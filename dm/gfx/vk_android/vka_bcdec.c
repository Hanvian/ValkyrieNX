// GPU compute decoder for BC4 / BC5 / BC6H / BC7 blocks on Vulkan devices that lack hardware BC4-7 support.
// Invoked from vka_exec.c (buf2tex / copy_t2t) and vka_draw.c (bcrt_flush). Replay thread safe; internal pipeline creation guarded by g_mu.
#include "vka.h"
#include "vka_bcn.h"
#include "vka_bcdec_spv.h"
#include <pthread.h>
#include <stdint.h>

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static VkDescriptorSetLayout g_dsl;
static VkPipelineLayout g_lay;
static VkPipeline g_pipe;
static VkDevice g_dev;

static int ensure_pipe(void) {
    if (!V.push_desc || !V.dev) return 0;
    pthread_mutex_lock(&g_mu);
    if (g_dev == V.dev && g_pipe) {
        pthread_mutex_unlock(&g_mu);
        return 1;
    }
    g_dsl = VK_NULL_HANDLE; g_lay = VK_NULL_HANDLE; g_pipe = VK_NULL_HANDLE; g_dev = V.dev;
    VkDescriptorSetLayoutBinding b[2] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dl = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
        .bindingCount = 2,
        .pBindings = b,
    };
    VkPushConstantRange pc = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkPipelineLayoutCreateInfo li = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &g_dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pc,
    };
    VkShaderModuleCreateInfo mi = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof SPV_BCDEC_COMP,
        .pCode = SPV_BCDEC_COMP,
    };
    VkShaderModule m = VK_NULL_HANDLE;
    VkResult r = vk.vkCreateDescriptorSetLayout(V.dev, &dl, NULL, &g_dsl);
    if (!r) r = vk.vkCreatePipelineLayout(V.dev, &li, NULL, &g_lay);
    if (!r) r = vk.vkCreateShaderModule(V.dev, &mi, NULL, &m);
    if (!r) {
        VkComputePipelineCreateInfo ci = {
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
            .layout = g_lay,
        };
        r = vk.vkCreateComputePipelines(V.dev, VK_NULL_HANDLE, 1, &ci, NULL, &g_pipe);
    }
    if (m) vk.vkDestroyShaderModule(V.dev, m, NULL);
    if (r) {
        VRED_ONCE("GPU BC4-7 decoder pipeline: %d", r);
        if (g_pipe) vk.vkDestroyPipeline(V.dev, g_pipe, NULL);
        if (g_lay) vk.vkDestroyPipelineLayout(V.dev, g_lay, NULL);
        if (g_dsl) vk.vkDestroyDescriptorSetLayout(V.dev, g_dsl, NULL);
        g_pipe = VK_NULL_HANDLE; g_lay = VK_NULL_HANDLE; g_dsl = VK_NULL_HANDLE; g_dev = VK_NULL_HANDLE;
    }
    pthread_mutex_unlock(&g_mu);
    return g_pipe != VK_NULL_HANDLE;
}

int vka_bcdec_dispatch(VkCommandBuffer cb, VkBuffer src_buf, VkDeviceSize src_off, uint32_t src_row_bytes, uint32_t src_layer_bytes,
                       vka_img *dst_im, uint32_t dst_lv, uint32_t dst_l0, int32_t xoff, int32_t yoff, int32_t zoff,
                       uint32_t w, uint32_t h, uint32_t layers) {
    if (!dst_im || !dst_im->f.bc_dec || !w || !h || !layers) return 0;
    if (!ensure_pipe()) return 0;

    uint32_t kind = dst_im->f.bc_dec;
    uint32_t src_bpb = dst_im->f.src_bpb;
    uint32_t bx = (w + 3u) / 4u, by = (h + 3u) / 4u;
    uint32_t row_tight = bx * src_bpb;
    if (!src_row_bytes) src_row_bytes = row_tight;
    if (!src_layer_bytes) src_layer_bytes = src_row_bytes * by;

    VkDeviceSize sb_al = V.sb_align > 16 ? V.sb_align : 16;
    if (src_off % sb_al != 0 || (src_row_bytes & 3u) != 0 || (src_layer_bytes & 3u) != 0) {
        uint64_t bnc_layer = (uint64_t)row_tight * by;
        vka_span src_bnc = {0};
        if (!vka_scratch(bnc_layer * layers, sb_al, &src_bnc)) return 0;
        for (uint32_t z = 0; z < layers; z++) {
            if (src_row_bytes == row_tight) {
                VkBufferCopy bc = {src_off + (VkDeviceSize)z * src_layer_bytes, src_bnc.off + (VkDeviceSize)z * bnc_layer, bnc_layer};
                vk.vkCmdCopyBuffer(cb, src_buf, src_bnc.buf, 1, &bc);
            } else {
                for (uint32_t y = 0; y < by; y++) {
                    VkBufferCopy bc = {
                        src_off + (VkDeviceSize)z * src_layer_bytes + (VkDeviceSize)y * src_row_bytes,
                        src_bnc.off + (VkDeviceSize)z * bnc_layer + (VkDeviceSize)y * row_tight,
                        row_tight,
                    };
                    vk.vkCmdCopyBuffer(cb, src_buf, src_bnc.buf, 1, &bc);
                }
            }
        }
        vka_barrier(cb);
        src_buf = src_bnc.buf; src_off = src_bnc.off;
        src_row_bytes = row_tight; src_layer_bytes = (uint32_t)bnc_layer;
    }

    uint32_t wpw = kind <= VKA_BC4_SNORM ? 1u : kind <= VKA_BC5_SNORM ? 2u : kind == VKA_BC7 ? 4u : 8u;
    uint32_t dst_row_words = bx * wpw;
    uint32_t dst_layer_words = by * 4u * dst_row_words;
    VkDeviceSize dst_layer_bytes = (VkDeviceSize)dst_layer_words * 4u;
    VkDeviceSize dst_bytes = dst_layer_bytes * layers;
    vka_span dst = {0};
    if (!vka_scratch(dst_bytes, sb_al, &dst)) return 0;

    VkDeviceSize src_span = (VkDeviceSize)(layers - 1u) * src_layer_bytes + (VkDeviceSize)(by - 1u) * src_row_bytes + row_tight;
    VkDescriptorBufferInfo bi[2] = {{src_buf, src_off, src_span}, {dst.buf, dst.off, dst_bytes}};
    VkWriteDescriptorSet wr[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[0]},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[1]},
    };
    uint32_t pc[8] = {kind, bx, by, layers, src_row_bytes / 4u, src_layer_bytes / 4u, dst_row_words, dst_layer_words};

    vka_disp_settle(cb);
    vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipe);
    vk.vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_lay, 0, 2, wr);
    vka_dyn_invalidate();
    vk.vkCmdPushConstants(cb, g_lay, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vk.vkCmdDispatch(cb, (bx + 7u) / 8u, (by + 7u) / 8u, layers);
    vka_barrier(cb);

    int is3d = dst_im->depth > 1 || dst_im->d.target == NVN_TEXTURE_TARGET_3D;
    if (is3d) {
        VkBufferImageCopy rg = {
            dst.off, bx * 4u, by * 4u,
            {VK_IMAGE_ASPECT_COLOR_BIT, dst_lv, 0, 1},
            {xoff, yoff, zoff},
            {w, h, layers},
        };
        vk.vkCmdCopyBufferToImage(cb, dst.buf, dst_im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
    } else {
        VkBufferImageCopy rg[64];
        uint32_t n = 0;
        for (uint32_t z = 0; z < layers; z++) {
            rg[n++] = (VkBufferImageCopy){
                dst.off + (VkDeviceSize)z * dst_layer_bytes, bx * 4u, by * 4u,
                {VK_IMAGE_ASPECT_COLOR_BIT, dst_lv, dst_l0 + z, 1},
                {xoff, yoff, 0},
                {w, h, 1},
            };
            if (n == 64 || z + 1 == layers) {
                vk.vkCmdCopyBufferToImage(cb, dst.buf, dst_im->img, VK_IMAGE_LAYOUT_GENERAL, n, rg);
                n = 0;
            }
        }
    }
    vka_img_written(dst_im);
    vka_barrier(cb);
    return 1;
}

void vka_bcdec_from_image(VkCommandBuffer cb, vka_img *src_im, uint32_t slv, uint32_t sl0, int32_t sx0, int32_t sy0, int32_t sz0,
                          vka_img *dst_im, uint32_t dlv, uint32_t dl0, int32_t dx0, int32_t dy0, int32_t dz0,
                          uint32_t w_px, uint32_t h_px, uint32_t layers) {
    if (!src_im || !dst_im || !dst_im->f.bc_dec || !w_px || !h_px || !layers) return;
    if (src_im->f.kind != 1 || src_im->f.bpb != dst_im->f.src_bpb) return;
    uint32_t bx = (w_px + 3u) / 4u, by = (h_px + 3u) / 4u;
    uint32_t sw = src_im->w >> slv ? src_im->w >> slv : 1u;
    uint32_t sh = src_im->h >> slv ? src_im->h >> slv : 1u;
    if (sx0 < 0 || sy0 < 0 || sz0 < 0 || (uint32_t)sx0 + bx > sw || (uint32_t)sy0 + by > sh) return;

    uint32_t row_bytes = bx * src_im->f.bpb;
    uint32_t layer_bytes = row_bytes * by;
    VkDeviceSize sb_al = V.sb_align > 16 ? V.sb_align : 16;
    vka_span raw = {0};
    if (!vka_scratch((VkDeviceSize)layer_bytes * layers, sb_al, &raw)) return;

    int src3d = src_im->depth > 1 || src_im->d.target == NVN_TEXTURE_TARGET_3D;
    if (src3d) {
        VkBufferImageCopy rg = {
            raw.off, 0, 0,
            {VK_IMAGE_ASPECT_COLOR_BIT, slv, 0, 1},
            {sx0, sy0, sz0},
            {bx, by, layers},
        };
        vk.vkCmdCopyImageToBuffer(cb, src_im->img, VK_IMAGE_LAYOUT_GENERAL, raw.buf, 1, &rg);
    } else {
        VkBufferImageCopy rg[64];
        uint32_t n = 0;
        for (uint32_t z = 0; z < layers; z++) {
            rg[n++] = (VkBufferImageCopy){
                raw.off + (VkDeviceSize)z * layer_bytes, 0, 0,
                {VK_IMAGE_ASPECT_COLOR_BIT, slv, sl0 + (uint32_t)sz0 + z, 1},
                {sx0, sy0, 0},
                {bx, by, 1},
            };
            if (n == 64 || z + 1 == layers) {
                vk.vkCmdCopyImageToBuffer(cb, src_im->img, VK_IMAGE_LAYOUT_GENERAL, raw.buf, n, rg);
                n = 0;
            }
        }
    }
    vka_barrier(cb);
    if (!vka_bcdec_dispatch(cb, raw.buf, raw.off, row_bytes, layer_bytes, dst_im, dlv, dl0, dx0, dy0, dz0, w_px, h_px, layers))
        VRED_ONCE("GPU BC4-7 decoder: could not decode blocks from texture serial %llu", (unsigned long long)src_im->serial);
}
