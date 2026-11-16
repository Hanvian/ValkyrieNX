// Optional implementation-dependent CPU alias of exported Vulkan memory. OPAQUE_FD does not promise mmap or ARM64 exclusive (LL/SC) support: never use a path
// until the actual device's export, bidirectional alias, mprotect, and ARM64 ldxr/stxr probe passes. When OPAQUE_FD fails LL/SC (Exynos Xclipse SPAL), fall back to
// cacheable AHardwareBuffer BLOB DMA-BUF imported via VK_ANDROID_external_memory_android_hardware_buffer with explicit EL0 D-cache clean/invalidate.
#pragma once
#include <android/hardware_buffer.h>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER
#define AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER (1ULL << 24)
#endif

static VkBufferUsageFlags vka_guest_usage(int bda) {
    return VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
           VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
           VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | (bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);
}

static size_t vka_fd_cache_line_bytes(void) {
#if defined(__aarch64__)
    static size_t line;
    if (!line) {
        uint64_t ctr = 0; __asm__ __volatile__("mrs %0, ctr_el0" : "=r"(ctr));
        size_t d = (size_t)4u << ((ctr >> 16) & 0xfu);
        line = d >= 16 ? d : 64;
    }
    return line;
#else
    return 64;
#endif
}

// inval=0: dc cvac (clean dirty CPU cache lines to PoC for GPU reads); inval=1: dc civac (clean+invalidate for CPU reads after GPU writes). Caller must ensure [p, p+n) has PROT_READ.
static inline void vka_fd_cache_op(const void *p, size_t n, int inval) {
#if defined(__aarch64__)
    if (!p || !n) return;
    size_t step = vka_fd_cache_line_bytes(), off = (uintptr_t)p & (step - 1), tot = n + off;
    uintptr_t a = (uintptr_t)p - off, e = tot >= n && a + tot >= a ? a + tot : UINTPTR_MAX;
    if (inval) { for (; a < e; a += step) __asm__ __volatile__("dc civac, %0" :: "r"(a) : "memory"); }
    else { for (; a < e; a += step) __asm__ __volatile__("dc cvac, %0" :: "r"(a) : "memory"); }
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    (void)p; (void)n; (void)inval;
#endif
}

static int vka_fd_llsc_ok(volatile uint32_t *w) {
#if defined(__aarch64__)
    *w = 0x10u;
    for (int attempt = 0; attempt < 4; attempt++) {
        uint32_t tmp, st;
        __asm__ __volatile__(
            "ldxr %w0, [%2]\n"
            "add  %w0, %w0, #1\n"
            "stxr %w1, %w0, [%2]\n"
            : "=&r"(tmp), "=&r"(st)
            : "r"(w)
            : "memory");
        if (st == 0) return *w == 0x11u;
    }
    return 0;
#else
    (void)w; return 1;
#endif
}

typedef struct { int version; int numFds; int numInts; int data[1]; } vka_native_handle;
typedef const vka_native_handle *(*vka_pfn_ahb_get_handle)(const AHardwareBuffer *);

static vka_pfn_ahb_get_handle vka_ahb_get_handle_fn(void) {
    static vka_pfn_ahb_get_handle fn; static int tried;
    if (!__atomic_load_n(&tried, __ATOMIC_ACQUIRE)) {
        void *lib = dlopen("libnativewindow.so", RTLD_NOW | RTLD_LOCAL);
        vka_pfn_ahb_get_handle p = (vka_pfn_ahb_get_handle)(lib ? dlsym(lib, "AHardwareBuffer_getNativeHandle") : dlsym(RTLD_DEFAULT, "AHardwareBuffer_getNativeHandle"));
        __atomic_store_n(&fn, p, __ATOMIC_RELAXED);
        __atomic_store_n(&tried, 1, __ATOMIC_RELEASE);
    }
    return fn;
}

typedef struct { VkBuffer buf; VkDeviceMemory mem; void *mapped; int fd; int fd_map_only; size_t map_size; } vka_fd_memory;

static void vka_fd_destroy(vka_fns *f, VkDevice dev, vka_fd_memory *m) {
    if (m->fd >= 0) close(m->fd);
    if (m->mapped) {
        if (m->fd_map_only) munmap(m->mapped, m->map_size);
        else f->vkUnmapMemory(dev, m->mem);
    }
    if (m->buf) f->vkDestroyBuffer(dev, m->buf, NULL);
    if (m->mem) f->vkFreeMemory(dev, m->mem, NULL);
    *m = (vka_fd_memory){.fd = -1};
}

static uint32_t vka_fd_pick_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits, int need_hv) {
    uint32_t type = UINT32_MAX;
    const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mp->memoryTypeCount; i++) if ((bits >> i) & 1u) {
        VkMemoryPropertyFlags pf = mp->memoryTypes[i].propertyFlags;
        if (need_hv) {
            if ((pf & hv) == hv && (type == UINT32_MAX || (pf & VK_MEMORY_PROPERTY_HOST_CACHED_BIT))) type = i;
        } else {
            if (type == UINT32_MAX || ((pf & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(mp->memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))) type = i;
        }
    }
    return type;
}

static VkResult vka_fd_create_ahb(vka_fns *f, VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp, VkDeviceSize n, int bda, vka_fd_memory *out) {
    vka_pfn_ahb_get_handle get_handle = vka_ahb_get_handle_fn();
    if (!f->vkGetAndroidHardwareBufferPropertiesANDROID || !get_handle) return VK_ERROR_EXTENSION_NOT_PRESENT;
    if (!n || n > (VkDeviceSize)UINT32_MAX) return VK_ERROR_OUT_OF_HOST_MEMORY;
    AHardwareBuffer_Desc desc = {
        .width = (uint32_t)n, .height = 1, .layers = 1, .format = AHARDWAREBUFFER_FORMAT_BLOB,
        .usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN | AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER,
    };
    AHardwareBuffer *ahb = NULL;
    if (AHardwareBuffer_allocate(&desc, &ahb) || !ahb) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    const vka_native_handle *nh = get_handle(ahb);
    int dup_fd = (nh && nh->numFds >= 1 && nh->data[0] >= 0) ? dup(nh->data[0]) : -1;
    if (dup_fd < 0) { AHardwareBuffer_release(ahb); return VK_ERROR_OUT_OF_HOST_MEMORY; }
    out->fd = dup_fd;

    VkAndroidHardwareBufferPropertiesANDROID ahbp = {VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
    VkResult r = f->vkGetAndroidHardwareBufferPropertiesANDROID(dev, ahb, &ahbp);
    if (r) goto fail;
    VkExternalMemoryBufferCreateInfo ei = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &ei, .size = n, .usage = vka_guest_usage(bda), .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if ((r = f->vkCreateBuffer(dev, &bi, NULL, &out->buf))) goto fail;
    VkMemoryRequirements mr; f->vkGetBufferMemoryRequirements(dev, out->buf, &mr);
    uint32_t bits = ahbp.memoryTypeBits & mr.memoryTypeBits;
    uint32_t type = vka_fd_pick_type(mp, bits, 0);
    if (type == UINT32_MAX) { r = VK_ERROR_FEATURE_NOT_PRESENT; goto fail; }
    VkImportAndroidHardwareBufferInfoANDROID imp = {VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID, .buffer = ahb};
    VkMemoryAllocateFlagsInfo af = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .pNext = &imp, .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = bda ? (void *)&af : (void *)&imp,
                               .allocationSize = ahbp.allocationSize > mr.size ? ahbp.allocationSize : mr.size, .memoryTypeIndex = type};
    if ((r = f->vkAllocateMemory(dev, &ai, NULL, &out->mem)) || (r = f->vkBindBufferMemory(dev, out->buf, out->mem, 0))) goto fail;
    AHardwareBuffer_release(ahb); ahb = NULL;
    void *p = mmap(NULL, (size_t)n, PROT_READ | PROT_WRITE, MAP_SHARED, out->fd, 0);
    if (p == MAP_FAILED) { r = VK_ERROR_MEMORY_MAP_FAILED; goto fail; }
    out->mapped = p; out->fd_map_only = 1; out->map_size = (size_t)n;
    return VK_SUCCESS;
fail:
    if (ahb) AHardwareBuffer_release(ahb);
    vka_fd_destroy(f, dev, out); return r;
}

static VkResult vka_fd_create(vka_fns *f, VkPhysicalDevice pd, VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
                              VkDeviceSize n, int bda, int use_ahb, vka_fd_memory *out) {
    *out = (vka_fd_memory){.fd = -1};
    if (use_ahb) return vka_fd_create_ahb(f, dev, mp, n, bda, out);
    if (!f->vkGetMemoryFdKHR || !f->vkGetPhysicalDeviceExternalBufferProperties) return VK_ERROR_EXTENSION_NOT_PRESENT;
    VkPhysicalDeviceExternalBufferInfo qi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO, .usage = vka_guest_usage(bda),
                                            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
    VkExternalBufferProperties ep = {VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES}; f->vkGetPhysicalDeviceExternalBufferProperties(pd, &qi, &ep);
    if (!(ep.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT)) return VK_ERROR_FEATURE_NOT_PRESENT;
    VkExternalMemoryBufferCreateInfo ei = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = qi.handleType};
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &ei, .size = n, .usage = qi.usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkResult r = f->vkCreateBuffer(dev, &bi, NULL, &out->buf); if (r) return r;
    VkMemoryRequirements mr; f->vkGetBufferMemoryRequirements(dev, out->buf, &mr);
    int host_visible = 1;
    uint32_t type = vka_fd_pick_type(mp, mr.memoryTypeBits, 1);
    if (type == UINT32_MAX) { host_visible = 0; type = vka_fd_pick_type(mp, mr.memoryTypeBits, 0); }
    if (type == UINT32_MAX) { r = VK_ERROR_FEATURE_NOT_PRESENT; goto fail; }
    VkMemoryAllocateFlagsInfo af = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryDedicatedAllocateInfo ded = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .pNext = bda ? &af : NULL, .buffer = out->buf};
    VkExportMemoryAllocateInfo ex = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, .pNext = &ded, .handleTypes = qi.handleType};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &ex, .allocationSize = mr.size, .memoryTypeIndex = type};
    if ((r = f->vkAllocateMemory(dev, &ai, NULL, &out->mem)) || (r = f->vkBindBufferMemory(dev, out->buf, out->mem, 0))) goto fail;
    VkMemoryGetFdInfoKHR gi = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = out->mem, .handleType = qi.handleType};
    if ((r = f->vkGetMemoryFdKHR(dev, &gi, &out->fd))) goto fail;
    if (host_visible) {
        if ((r = f->vkMapMemory(dev, out->mem, 0, VK_WHOLE_SIZE, 0, &out->mapped))) goto fail;
    } else {
        size_t msz = (size_t)(mr.size ? mr.size : n);
        void *p = mmap(NULL, msz, PROT_READ | PROT_WRITE, MAP_SHARED, out->fd, 0);
        if (p == MAP_FAILED) { r = VK_ERROR_MEMORY_MAP_FAILED; goto fail; }
        out->mapped = p; out->fd_map_only = 1; out->map_size = msz;
    }
    return VK_SUCCESS;
fail:
    vka_fd_destroy(f, dev, out); return r;
}

static int vka_fd_enabled(void) { const char *e = getenv("DM_VK_FD_ALIAS"); return !e || strcmp(e, "0"); }

static int vka_fd_probe_one(vka_fns *f, VkPhysicalDevice pd, VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
                            int bda, int use_ahb, char *why, size_t whyn) {
    const size_t n = 65536; vka_fd_memory m; VkResult r = vka_fd_create(f, pd, dev, mp, n, bda, use_ahb, &m);
    const char *kind = use_ahb ? "AHardwareBuffer" : "opaque_fd";
    if (r) { snprintf(why, whyn, "%s allocation failed (%d)", kind, r); return 0; }
    void *at = mmap(NULL, n, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); int ok = 0;
    if (at != MAP_FAILED) {
        void *a = mmap(at, n, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, m.fd, 0);
        if (a != MAP_FAILED) {
            volatile uint32_t *original = m.mapped, *alias = a; ok = 1;
            for (size_t i = 0; i < n / 4; i++) original[i] = (uint32_t)i ^ 0x5eed1234u;
            for (size_t i = 0; i < n / 4; i++) if (alias[i] != ((uint32_t)i ^ 0x5eed1234u)) { ok = 0; break; }
            for (size_t i = 0; i < n / 4; i++) alias[i] = (uint32_t)i ^ 0xc0ffee00u;
            if (mprotect((char *)a + 4096, 4096, PROT_READ) || mprotect((char *)a + 4096, 4096, PROT_READ | PROT_WRITE)) ok = 0;
            for (size_t i = 0; i < n / 4; i++) if (original[i] != ((uint32_t)i ^ 0xc0ffee00u) || alias[i] != original[i]) { ok = 0; break; }
            int llsc = ok ? vka_fd_llsc_ok((volatile uint32_t *)((char *)a + 8192)) : 0;
            if (ok && use_ahb) vka_fd_cache_op(a, n, 0);
            if (!ok) snprintf(why, whyn, "%s bidirectional fixed alias or mprotect failed", kind);
            else if (!llsc) { ok = 0; snprintf(why, whyn, "%s lacks ARM64 exclusive (LL/SC) support", kind); }
            else snprintf(why, whyn, "passed %s fixed alias, mprotect, and ARM64 LL/SC%s", kind, use_ahb ? " (cache maintenance enabled)" : "");
        } else snprintf(why, whyn, "%s fd mmap failed (errno %d)", kind, errno);
        munmap(at, n);
    } else snprintf(why, whyn, "alias reservation failed (errno %d)", errno);
    vka_fd_destroy(f, dev, &m); return ok;
}

static int vka_fd_probe(vka_fns *f, VkPhysicalDevice pd, VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
                        int bda, int x_fd, int x_ahb, int *noncoherent, char *why, size_t whyn) {
    *noncoherent = 0;
    const char *e = getenv("DM_VK_FD_ALIAS");
    int force_ahb = e && (!strcmp(e, "ahb") || !strcmp(e, "2"));
    int force_opaque = e && !strcmp(e, "opaque");
    char w_op[160] = "not probed", w_ahb[160] = "not probed";
    if (!force_ahb && x_fd && vka_fd_probe_one(f, pd, dev, mp, bda, 0, w_op, sizeof w_op)) {
        snprintf(why, whyn, "%s", w_op); return 1;
    }
    if (!force_opaque && x_ahb && vka_fd_probe_one(f, pd, dev, mp, bda, 1, w_ahb, sizeof w_ahb)) {
        *noncoherent = 1; snprintf(why, whyn, "%s", w_ahb); return 1;
    }
    if (!force_ahb && x_fd) snprintf(why, whyn, "%s%s%s", w_op, (!force_opaque && x_ahb) ? "; " : "", (!force_opaque && x_ahb) ? w_ahb : "");
    else snprintf(why, whyn, "%s", x_ahb ? w_ahb : "no external fd/AHB extension");
    return 0;
}
