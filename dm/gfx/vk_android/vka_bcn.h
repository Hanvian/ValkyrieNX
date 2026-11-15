// Software fallback decoder for BC4 / BC5 / BC6H / BC7 when the Vulkan device lacks those compressed formats
// (e.g., Samsung Xclipse SPAL driver supports BC1-BC3 in hardware, not BC4-BC7).
// Tables and bitfield definitions follow the Khronos Data Format Specification (BPTC / RGTC).
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum {
    VKA_BC_NONE = 0,
    VKA_BC4_UNORM = 1,
    VKA_BC4_SNORM = 2,
    VKA_BC5_UNORM = 3,
    VKA_BC5_SNORM = 4,
    VKA_BC6H_SFLOAT = 5,
    VKA_BC6H_UFLOAT = 6,
    VKA_BC7 = 7,
};

static inline uint32_t vka_bc_bits(const uint8_t *b, int off, int n) {
    if (n <= 0) return 0;
    uint32_t out = 0;
    int byte = off >> 3, bit = off & 7, pos = 0;
    while (n > 0) {
        int take = 8 - bit;
        if (take > n) take = n;
        out |= (uint32_t)((b[byte] >> bit) & ((1u << take) - 1u)) << pos;
        n -= take;
        pos += take;
        byte++;
        bit = 0;
    }
    return out;
}

static inline int32_t vka_bc_sext(int32_t v, int bits) {
    int32_t m = 1 << (bits - 1);
    return (v ^ m) - m;
}

static inline uint8_t vka_bc7_expand(uint8_t v, int bits) {
    return bits == 8 ? v : (uint8_t)((v << (8 - bits)) | (v >> (2 * bits - 8)));
}

static inline int32_t vka_bptc_interp(int32_t a, int32_t b, int idx, int bits) {
    static const uint8_t w2[4] = {0, 21, 43, 64};
    static const uint8_t w3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
    static const uint8_t w4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};
    const int w = bits == 2 ? w2[idx] : bits == 3 ? w3[idx] : w4[idx];
    return ((64 - w) * a + w * b + 32) >> 6;
}

static void vka_bc4_block_u(const uint8_t *b, uint8_t *dst, size_t dst_pitch, uint32_t w, uint32_t h, uint32_t step) {
    uint8_t p[8];
    p[0] = b[0];
    p[1] = b[1];
    if (b[0] > b[1]) {
        for (int i = 0; i < 6; i++) p[i + 2] = (uint8_t)(((6 - i) * (int)b[0] + (i + 1) * (int)b[1] + 3) / 7);
    } else {
        for (int i = 0; i < 4; i++) p[i + 2] = (uint8_t)(((4 - i) * (int)b[0] + (i + 1) * (int)b[1] + 2) / 5);
        p[6] = 0;
        p[7] = 255;
    }
    uint64_t idx = 0;
    for (int i = 0; i < 6; i++) idx |= (uint64_t)b[2 + i] << (8 * i);
    for (uint32_t y = 0; y < h; y++) {
        uint8_t *row = dst + (size_t)y * dst_pitch;
        for (uint32_t x = 0; x < w; x++) {
            uint32_t c = (uint32_t)((idx >> (3 * (y * 4 + x))) & 7);
            row[x * step] = p[c];
        }
    }
}

static void vka_bc4_block_s(const uint8_t *b, uint8_t *dst, size_t dst_pitch, uint32_t w, uint32_t h, uint32_t step) {
    int8_t s0 = (int8_t)b[0], s1 = (int8_t)b[1];
    int e0 = s0 == -128 ? -127 : (int)s0;
    int e1 = s1 == -128 ? -127 : (int)s1;
    int8_t p[8];
    p[0] = (int8_t)e0;
    p[1] = (int8_t)e1;
    if (e0 > e1) {
        for (int i = 0; i < 6; i++) {
            int v = (6 - i) * e0 + (i + 1) * e1;
            p[i + 2] = (int8_t)(v >= 0 ? (v + 3) / 7 : (v - 3) / 7);
        }
    } else {
        for (int i = 0; i < 4; i++) {
            int v = (4 - i) * e0 + (i + 1) * e1;
            p[i + 2] = (int8_t)(v >= 0 ? (v + 2) / 5 : (v - 2) / 5);
        }
        p[6] = -127;
        p[7] = 127;
    }
    uint64_t idx = 0;
    for (int i = 0; i < 6; i++) idx |= (uint64_t)b[2 + i] << (8 * i);
    for (uint32_t y = 0; y < h; y++) {
        uint8_t *row = dst + (size_t)y * dst_pitch;
        for (uint32_t x = 0; x < w; x++) {
            uint32_t c = (uint32_t)((idx >> (3 * (y * 4 + x))) & 7);
            row[x * step] = (uint8_t)p[c];
        }
    }
}

static const uint32_t k_bptc_part2[64] = {
    0x50505050u, 0x40404040u, 0x54545454u, 0x54505040u,
    0x50404000u, 0x55545450u, 0x55545040u, 0x54504000u,
    0x50400000u, 0x55555450u, 0x55544000u, 0x54400000u,
    0x55555440u, 0x55550000u, 0x55555500u, 0x55000000u,
    0x55150100u, 0x00004054u, 0x15010000u, 0x00405054u,
    0x00004050u, 0x15050100u, 0x05010000u, 0x40505054u,
    0x00404050u, 0x05010100u, 0x14141414u, 0x05141450u,
    0x01155440u, 0x00555500u, 0x15014054u, 0x05414150u,
    0x44444444u, 0x55005500u, 0x11441144u, 0x05055050u,
    0x05500550u, 0x11114444u, 0x41144114u, 0x44111144u,
    0x15055054u, 0x01055040u, 0x05041050u, 0x05455150u,
    0x14414114u, 0x50050550u, 0x41411414u, 0x00141400u,
    0x00041504u, 0x00105410u, 0x10541000u, 0x04150400u,
    0x50410514u, 0x41051450u, 0x05415014u, 0x14054150u,
    0x41050514u, 0x41505014u, 0x40011554u, 0x54150140u,
    0x50505500u, 0x00555050u, 0x15151010u, 0x54540404u,
};

static const uint32_t k_bptc_part3[64] = {
    0xaa685050u, 0x6a5a5040u, 0x5a5a4200u, 0x5450a0a8u,
    0xa5a50000u, 0xa0a05050u, 0x5555a0a0u, 0x5a5a5050u,
    0xaa550000u, 0xaa555500u, 0xaaaa5500u, 0x90909090u,
    0x94949494u, 0xa4a4a4a4u, 0xa9a59450u, 0x2a0a4250u,
    0xa5945040u, 0x0a425054u, 0xa5a5a500u, 0x55a0a0a0u,
    0xa8a85454u, 0x6a6a4040u, 0xa4a45000u, 0x1a1a0500u,
    0x0050a4a4u, 0xaaa59090u, 0x14696914u, 0x69691400u,
    0xa08585a0u, 0xaa821414u, 0x50a4a450u, 0x6a5a0200u,
    0xa9a58000u, 0x5090a0a8u, 0xa8a09050u, 0x24242424u,
    0x00aa5500u, 0x24924924u, 0x24499224u, 0x50a50a50u,
    0x500aa550u, 0xaaaa4444u, 0x66660000u, 0xa5a0a5a0u,
    0x50a050a0u, 0x69286928u, 0x44aaaa44u, 0x66666600u,
    0xaa444444u, 0x54a854a8u, 0x95809580u, 0x96969600u,
    0xa85454a8u, 0x80959580u, 0xaa141414u, 0x96960000u,
    0xaaaa1414u, 0xa05050a0u, 0xa0a5a5a0u, 0x96000000u,
    0x40804080u, 0xa9a8a9a8u, 0xaaaaaa44u, 0x2a4a5254u,
};

static const uint8_t k_bptc_anchor[3][64] = {
    {
        0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,
        0xf,0x2,0x8,0x2,0x2,0x8,0x8,0xf,0x2,0x8,0x2,0x2,0x8,0x8,0x2,0x2,
        0xf,0xf,0x6,0x8,0x2,0x8,0xf,0xf,0x2,0x8,0x2,0x2,0x2,0xf,0xf,0x6,
        0x6,0x2,0x6,0x8,0xf,0xf,0x2,0x2,0xf,0xf,0xf,0xf,0xf,0x2,0x2,0xf
    },
    {
        0x3,0x3,0xf,0xf,0x8,0x3,0xf,0xf,0x8,0x8,0x6,0x6,0x6,0x5,0x3,0x3,
        0x3,0x3,0x8,0xf,0x3,0x3,0x6,0xa,0x5,0x8,0x8,0x6,0x8,0x5,0xf,0xf,
        0x8,0xf,0x3,0x5,0x6,0xa,0x8,0xf,0xf,0x3,0xf,0x5,0xf,0xf,0xf,0xf,
        0x3,0xf,0x5,0x5,0x5,0x8,0x5,0xa,0x5,0xa,0x8,0xd,0xf,0xc,0x3,0x3
    },
    {
        0xf,0x8,0x8,0x3,0xf,0xf,0x3,0x8,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0x8,
        0xf,0x8,0xf,0x3,0xf,0x8,0xf,0x8,0x3,0xf,0x6,0xa,0xf,0xf,0xa,0x8,
        0xf,0x3,0xf,0xa,0xa,0x8,0x9,0xa,0x6,0xf,0x8,0xf,0x3,0x6,0x6,0x8,
        0xf,0x3,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0x3,0xf,0xf,0x8
    },
};

static inline int vka_bptc_is_anchor(int n_subsets, int part, int texel) {
    if (texel == 0) return 1;
    if (n_subsets == 2) return k_bptc_anchor[0][part] == texel;
    if (n_subsets == 3) return k_bptc_anchor[1][part] == texel || k_bptc_anchor[2][part] == texel;
    return 0;
}

static inline int vka_bptc_anchors_before(int n_subsets, int part, int texel) {
    int c = texel > 0 ? 1 : 0;
    if (n_subsets == 2 && texel > k_bptc_anchor[0][part]) c++;
    else if (n_subsets == 3) {
        if (texel > k_bptc_anchor[1][part]) c++;
        if (texel > k_bptc_anchor[2][part]) c++;
    }
    return c;
}

typedef struct {
    uint8_t n_subsets, n_part_bits, has_rot, has_idx_sel, n_color_bits, n_alpha_bits, has_ep_pbits, has_sh_pbits, n_idx_bits, n_idx2_bits;
} vka_bc7_mode;

static const vka_bc7_mode k_bc7_modes[8] = {
    {3, 4, 0, 0, 4, 0, 1, 0, 3, 0},
    {2, 6, 0, 0, 6, 0, 0, 1, 3, 0},
    {3, 6, 0, 0, 5, 0, 0, 0, 2, 0},
    {2, 6, 0, 0, 7, 0, 1, 0, 2, 0},
    {1, 0, 1, 1, 5, 6, 0, 0, 2, 3},
    {1, 0, 1, 0, 7, 8, 0, 0, 2, 2},
    {1, 0, 0, 0, 7, 7, 1, 0, 4, 0},
    {2, 6, 0, 0, 5, 5, 1, 0, 2, 0},
};

static void vka_bc7_block(const uint8_t *b, uint8_t *dst, size_t dst_pitch, uint32_t w, uint32_t h) {
    if (!b[0]) {
        for (uint32_t y = 0; y < h; y++) memset(dst + (size_t)y * dst_pitch, 0, (size_t)w * 4);
        return;
    }
    int m = 0;
    while (((b[0] >> m) & 1) == 0) m++;
    const vka_bc7_mode *mode = &k_bc7_modes[m];
    int bit = m + 1;

    int part = (int)vka_bc_bits(b, bit, mode->n_part_bits);
    bit += mode->n_part_bits;
    uint32_t subsets = mode->n_subsets == 1 ? 0 : mode->n_subsets == 2 ? k_bptc_part2[part] : k_bptc_part3[part];

    int rot = 0;
    if (mode->has_rot) { rot = (int)vka_bc_bits(b, bit, 2); bit += 2; }
    int idx_sel = 0;
    if (mode->has_idx_sel) { idx_sel = (int)vka_bc_bits(b, bit, 1); bit += 1; }

    uint8_t ep[6][4];
    for (int c = 0; c < 3; c++)
        for (int s = 0; s < mode->n_subsets; s++)
            for (int e = 0; e < 2; e++) {
                ep[s * 2 + e][c] = (uint8_t)vka_bc_bits(b, bit, mode->n_color_bits);
                bit += mode->n_color_bits;
            }
    int n_comp = 3;
    if (mode->n_alpha_bits > 0) {
        for (int s = 0; s < mode->n_subsets; s++)
            for (int e = 0; e < 2; e++) {
                ep[s * 2 + e][3] = (uint8_t)vka_bc_bits(b, bit, mode->n_alpha_bits);
                bit += mode->n_alpha_bits;
            }
        n_comp = 4;
    } else {
        for (int s = 0; s < mode->n_subsets; s++)
            for (int e = 0; e < 2; e++)
                ep[s * 2 + e][3] = 255;
    }
    if (mode->has_ep_pbits) {
        for (int s = 0; s < mode->n_subsets; s++)
            for (int e = 0; e < 2; e++) {
                uint8_t pb = (uint8_t)vka_bc_bits(b, bit, 1);
                bit += 1;
                for (int c = 0; c < n_comp; c++) ep[s * 2 + e][c] = (uint8_t)((ep[s * 2 + e][c] << 1) | pb);
            }
    } else if (mode->has_sh_pbits) {
        for (int s = 0; s < mode->n_subsets; s++) {
            uint8_t pb = (uint8_t)vka_bc_bits(b, bit, 1);
            bit += 1;
            for (int e = 0; e < 2; e++)
                for (int c = 0; c < n_comp; c++) ep[s * 2 + e][c] = (uint8_t)((ep[s * 2 + e][c] << 1) | pb);
        }
    }
    int cb = mode->n_color_bits + mode->has_ep_pbits + mode->has_sh_pbits;
    int ab = mode->n_alpha_bits + mode->has_ep_pbits + mode->has_sh_pbits;
    for (int s = 0; s < mode->n_subsets; s++)
        for (int e = 0; e < 2; e++) {
            for (int c = 0; c < 3; c++) ep[s * 2 + e][c] = vka_bc7_expand(ep[s * 2 + e][c], cb);
            if (mode->n_alpha_bits > 0) ep[s * 2 + e][3] = vka_bc7_expand(ep[s * 2 + e][3], ab);
        }

    for (uint32_t y = 0; y < h; y++) {
        uint8_t *px = dst + (size_t)y * dst_pitch;
        for (uint32_t x = 0; x < w; x++) {
            int texel = (int)(x + y * 4);
            int anc = vka_bptc_anchors_before(mode->n_subsets, part, texel);
            int is_anc = vka_bptc_is_anchor(mode->n_subsets, part, texel);
            int off1 = bit + mode->n_idx_bits * texel - anc;
            int off2 = bit + 16 * mode->n_idx_bits - mode->n_subsets + mode->n_idx2_bits * texel - anc;
            int sub = (int)((subsets >> (texel * 2)) & 3u);
            int idx[2] = {0, 0};
            idx[0] = (int)vka_bc_bits(b, off1, mode->n_idx_bits - is_anc);
            if (mode->n_idx2_bits) idx[1] = (int)vka_bc_bits(b, off2, mode->n_idx2_bits - is_anc);

            int ci = idx[idx_sel];
            int cb_i = idx_sel ? mode->n_idx2_bits : mode->n_idx_bits;
            for (int c = 0; c < 3; c++) px[c] = (uint8_t)vka_bptc_interp(ep[sub * 2][c], ep[sub * 2 + 1][c], ci, cb_i);

            int ai = (mode->n_idx2_bits && !idx_sel) ? idx[1] : idx[0];
            int ab_i = (mode->n_idx2_bits && !idx_sel) ? mode->n_idx2_bits : mode->n_idx_bits;
            px[3] = (uint8_t)vka_bptc_interp(ep[sub * 2][3], ep[sub * 2 + 1][3], ai, ab_i);

            if (rot) {
                uint8_t t = px[rot - 1];
                px[rot - 1] = px[3];
                px[3] = t;
            }
            px += 4;
        }
    }
}

typedef struct { int8_t ep; uint8_t comp, off, nbits, rev; } vka_bc6_bf;
typedef struct { uint8_t res, tr, part_bits, ep_bits, idx_bits, d_bits[3]; vka_bc6_bf bf[24]; } vka_bc6_mode;

static const vka_bc6_mode k_bc6_modes[34] = {
    {0, 1, 5, 10, 3, {5, 5, 5}, {{2,1,4,1,0},{2,2,4,1,0},{3,2,4,1,0},{0,0,0,10,0},{0,1,0,10,0},{0,2,0,10,0},{1,0,0,5,0},{3,1,4,1,0},{2,1,0,4,0},{1,1,0,5,0},{3,2,0,1,0},{3,1,0,4,0},{1,2,0,5,0},{3,2,1,1,0},{2,2,0,4,0},{2,0,0,5,0},{3,2,2,1,0},{3,0,0,5,0},{3,2,3,1,0},{-1}}},
    {0, 1, 5, 7,  3, {6, 6, 6}, {{2,1,5,1,0},{3,1,4,1,0},{3,1,5,1,0},{0,0,0,7,0},{3,2,0,1,0},{3,2,1,1,0},{2,2,4,1,0},{0,1,0,7,0},{2,2,5,1,0},{3,2,2,1,0},{2,1,4,1,0},{0,2,0,7,0},{3,2,3,1,0},{3,2,5,1,0},{3,2,4,1,0},{1,0,0,6,0},{2,1,0,4,0},{1,1,0,6,0},{3,1,0,4,0},{1,2,0,6,0},{2,2,0,4,0},{2,0,0,6,0},{3,0,0,6,0},{-1}}},
    {0, 1, 5, 11, 3, {5, 4, 4}, {{0,0,0,10,0},{0,1,0,10,0},{0,2,0,10,0},{1,0,0,5,0},{0,0,10,1,0},{2,1,0,4,0},{1,1,0,4,0},{0,1,10,1,0},{3,2,0,1,0},{3,1,0,4,0},{1,2,0,4,0},{0,2,10,1,0},{3,2,1,1,0},{2,2,0,4,0},{2,0,0,5,0},{3,2,2,1,0},{3,0,0,5,0},{3,2,3,1,0},{-1}}},
    {0, 0, 0, 10, 4, {10,10,10},{{0,0,0,10,0},{0,1,0,10,0},{0,2,0,10,0},{1,0,0,10,0},{1,1,0,10,0},{1,2,0,10,0},{-1}}},
    {0, 1, 5, 11, 3, {4, 5, 4}, {{0,0,0,10,0},{0,1,0,10,0},{0,2,0,10,0},{1,0,0,4,0},{0,0,10,1,0},{3,1,4,1,0},{2,1,0,4,0},{1,1,0,5,0},{0,1,10,1,0},{3,1,0,4,0},{1,2,0,4,0},{0,2,10,1,0},{3,2,1,1,0},{2,2,0,4,0},{2,0,0,4,0},{3,2,0,1,0},{3,2,2,1,0},{3,0,0,4,0},{2,1,4,1,0},{3,2,3,1,0},{-1}}},
    {0, 1, 0, 11, 4, {9, 9, 9}, {{0,0,0,10,0},{0,1,0,10,0},{0,2,0,10,0},{1,0,0,9,0},{0,0,10,1,0},{1,1,0,9,0},{0,1,10,1,0},{1,2,0,9,0},{0,2,10,1,0},{-1}}},
    {0, 1, 5, 11, 3, {4, 4, 5}, {{0,0,0,10,0},{0,1,0,10,0},{0,2,0,10,0},{1,0,0,4,0},{0,0,10,1,0},{2,2,4,1,0},{2,1,0,4,0},{1,1,0,4,0},{0,1,10,1,0},{3,2,0,1,0},{3,1,0,4,0},{1,2,0,5,0},{0,2,10,1,0},{2,2,0,4,0},{2,0,0,4,0},{3,2,1,1,0},{3,2,2,1,0},{3,0,0,4,0},{3,2,4,1,0},{3,2,3,1,0},{-1}}},
    {0, 1, 0, 12, 4, {8, 8, 8}, {{0,0,0,10,0},{0,1,0,10,0},{0,2,0,10,0},{1,0,0,8,0},{0,0,10,2,1},{1,1,0,8,0},{0,1,10,2,1},{1,2,0,8,0},{0,2,10,2,1},{-1}}},
    {0, 1, 5, 9,  3, {5, 5, 5}, {{0,0,0,9,0},{2,2,4,1,0},{0,1,0,9,0},{2,1,4,1,0},{0,2,0,9,0},{3,2,4,1,0},{1,0,0,5,0},{3,1,4,1,0},{2,1,0,4,0},{1,1,0,5,0},{3,2,0,1,0},{3,1,0,4,0},{1,2,0,5,0},{3,2,1,1,0},{2,2,0,4,0},{2,0,0,5,0},{3,2,2,1,0},{3,0,0,5,0},{3,2,3,1,0},{-1}}},
    {0, 1, 0, 16, 4, {4, 4, 4}, {{0,0,0,10,0},{0,1,0,10,0},{0,2,0,10,0},{1,0,0,4,0},{0,0,10,6,1},{1,1,0,4,0},{0,1,10,6,1},{1,2,0,4,0},{0,2,10,6,1},{-1}}},
    {0, 1, 5, 8,  3, {6, 5, 5}, {{0,0,0,8,0},{3,1,4,1,0},{2,2,4,1,0},{0,1,0,8,0},{3,2,2,1,0},{2,1,4,1,0},{0,2,0,8,0},{3,2,3,1,0},{3,2,4,1,0},{1,0,0,6,0},{2,1,0,4,0},{1,1,0,5,0},{3,2,0,1,0},{3,1,0,4,0},{1,2,0,5,0},{3,2,1,1,0},{2,2,0,4,0},{2,0,0,6,0},{3,0,0,6,0},{-1}}},
    {1},
    {0, 1, 5, 8,  3, {5, 6, 5}, {{0,0,0,8,0},{3,2,0,1,0},{2,2,4,1,0},{0,1,0,8,0},{2,1,5,1,0},{2,1,4,1,0},{0,2,0,8,0},{3,1,5,1,0},{3,2,4,1,0},{1,0,0,5,0},{3,1,4,1,0},{2,1,0,4,0},{1,1,0,6,0},{3,1,0,4,0},{1,2,0,5,0},{3,2,1,1,0},{2,2,0,4,0},{2,0,0,5,0},{3,2,2,1,0},{3,0,0,5,0},{3,2,3,1,0},{-1}}},
    {1},
    {0, 1, 5, 8,  3, {5, 5, 6}, {{0,0,0,8,0},{3,2,1,1,0},{2,2,4,1,0},{0,1,0,8,0},{2,2,5,1,0},{2,1,4,1,0},{0,2,0,8,0},{3,2,5,1,0},{3,2,4,1,0},{1,0,0,5,0},{3,1,4,1,0},{2,1,0,4,0},{1,1,0,5,0},{3,2,0,1,0},{3,1,0,4,0},{1,2,0,6,0},{2,2,0,4,0},{2,0,0,5,0},{3,2,2,1,0},{3,0,0,5,0},{3,2,3,1,0},{-1}}},
    {1},
    {0, 0, 5, 6,  3, {6, 6, 6}, {{0,0,0,6,0},{3,1,4,1,0},{3,2,0,1,0},{3,2,1,1,0},{2,2,4,1,0},{0,1,0,6,0},{2,1,5,1,0},{2,2,5,1,0},{3,2,2,1,0},{2,1,4,1,0},{0,2,0,6,0},{3,1,5,1,0},{3,2,3,1,0},{3,2,5,1,0},{3,2,4,1,0},{1,0,0,6,0},{2,1,0,4,0},{1,1,0,6,0},{3,1,0,4,0},{1,2,0,6,0},{2,2,0,4,0},{2,0,0,6,0},{3,0,0,6,0},{-1}}},
    {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1},
};

static inline int32_t vka_bc6_unq(int32_t v, int bits, int is_signed) {
    if (is_signed) {
        if (bits >= 16 || v == 0) return v;
        int neg = v < 0;
        if (neg) v = -v;
        v = (v >= (1 << (bits - 1)) - 1) ? 0x7fff : ((v << 15) + 0x4000) >> (bits - 1);
        return neg ? -v : v;
    }
    if (bits >= 15 || v == 0) return v;
    if (v == (1 << bits) - 1) return 0xffff;
    return ((v << 15) + 0x4000) >> (bits - 1);
}

static void vka_bc6h_block(const uint8_t *b, uint16_t *dst, size_t dst_pitch_u16, uint32_t w, uint32_t h, int is_signed) {
    int mode_num, bit;
    if (b[0] & 0x2) { mode_num = (((b[0] >> 1) & 0xe) | (b[0] & 1)) + 2; bit = 5; }
    else { mode_num = b[0] & 3; bit = 2; }
    const vka_bc6_mode *mode = mode_num < 34 ? &k_bc6_modes[mode_num] : &k_bc6_modes[17];
    if (mode->res) {
        for (uint32_t y = 0; y < h; y++) {
            uint16_t *p = dst + (size_t)y * dst_pitch_u16;
            for (uint32_t x = 0; x < w; x++) { p[0] = 0; p[1] = 0; p[2] = 0; p[3] = 0x3c00; p += 4; }
        }
        return;
    }
    int n_ep = mode->part_bits ? 4 : 2;
    int32_t ep[4][3] = {{0}};
    for (const vka_bc6_bf *bf = mode->bf; bf->ep != -1; bf++) {
        uint32_t v = vka_bc_bits(b, bit, bf->nbits);
        bit += bf->nbits;
        if (bf->rev) {
            for (int i = 0; i < bf->nbits; i++)
                if (v & (1u << i)) ep[bf->ep][bf->comp] |= 1 << ((bf->nbits - 1 - i) + bf->off);
        } else {
            ep[bf->ep][bf->comp] |= (int32_t)v << bf->off;
        }
    }
    if (mode->tr) {
        int32_t mask = (1 << mode->ep_bits) - 1;
        for (int e = 1; e < n_ep; e++)
            for (int c = 0; c < 3; c++) {
                int32_t d = vka_bc_sext(ep[e][c], mode->d_bits[c]);
                ep[e][c] = (ep[0][c] + d) & mask;
            }
    }
    for (int e = 0; e < n_ep; e++)
        for (int c = 0; c < 3; c++) {
            int32_t v = is_signed ? vka_bc_sext(ep[e][c], mode->ep_bits) : ep[e][c];
            ep[e][c] = vka_bc6_unq(v, mode->ep_bits, is_signed);
        }

    int part = 0, n_sub = 1;
    uint32_t subsets = 0;
    if (mode->part_bits) {
        part = (int)vka_bc_bits(b, bit, mode->part_bits);
        bit += mode->part_bits;
        subsets = k_bptc_part2[part & 31];
        n_sub = 2;
    }

    for (uint32_t y = 0; y < h; y++) {
        uint16_t *px = dst + (size_t)y * dst_pitch_u16;
        for (uint32_t x = 0; x < w; x++) {
            int texel = (int)(x + y * 4);
            int anc = vka_bptc_anchors_before(n_sub, part & 31, texel);
            int is_anc = vka_bptc_is_anchor(n_sub, part & 31, texel);
            int sub = (int)((subsets >> (texel * 2)) & 3u);
            int idx = (int)vka_bc_bits(b, bit + mode->idx_bits * texel - anc, mode->idx_bits - is_anc);
            for (int c = 0; c < 3; c++) {
                int32_t v = vka_bptc_interp(ep[sub * 2][c], ep[sub * 2 + 1][c], idx, mode->idx_bits);
                if (is_signed) px[c] = (uint16_t)(v < 0 ? ((-v * 31 / 32) | 0x8000) : (v * 31 / 32));
                else px[c] = (uint16_t)(v * 31 / 64);
            }
            px[3] = 0x3c00;
            px += 4;
        }
    }
}

static inline void vka_bcn_decode(int kind, uint32_t w, uint32_t h_rem, uint32_t bx,
                                  const uint8_t *src, size_t src_pitch, uint8_t *dst, size_t dst_pitch) {
    for (uint32_t by = 0; by * 4 < h_rem; by++) {
        uint32_t bh = h_rem - by * 4;
        if (bh > 4) bh = 4;
        const uint8_t *srow = src + (size_t)by * src_pitch;
        uint8_t *drow = dst + (size_t)by * 4 * dst_pitch;
        for (uint32_t x = 0; x < bx; x++) {
            uint32_t px = x * 4;
            if (px >= w) break;
            uint32_t bw = w - px;
            if (bw > 4) bw = 4;
            switch (kind) {
            case VKA_BC4_UNORM:
                vka_bc4_block_u(srow + (size_t)x * 8, drow + px, dst_pitch, bw, bh, 1);
                break;
            case VKA_BC4_SNORM:
                vka_bc4_block_s(srow + (size_t)x * 8, drow + px, dst_pitch, bw, bh, 1);
                break;
            case VKA_BC5_UNORM:
                vka_bc4_block_u(srow + (size_t)x * 16, drow + (size_t)px * 2, dst_pitch, bw, bh, 2);
                vka_bc4_block_u(srow + (size_t)x * 16 + 8, drow + (size_t)px * 2 + 1, dst_pitch, bw, bh, 2);
                break;
            case VKA_BC5_SNORM:
                vka_bc4_block_s(srow + (size_t)x * 16, drow + (size_t)px * 2, dst_pitch, bw, bh, 2);
                vka_bc4_block_s(srow + (size_t)x * 16 + 8, drow + (size_t)px * 2 + 1, dst_pitch, bw, bh, 2);
                break;
            case VKA_BC6H_SFLOAT:
            case VKA_BC6H_UFLOAT:
                vka_bc6h_block(srow + (size_t)x * 16, (uint16_t *)(drow + (size_t)px * 8), dst_pitch / 2, bw, bh, kind == VKA_BC6H_SFLOAT);
                break;
            case VKA_BC7:
                vka_bc7_block(srow + (size_t)x * 16, drow + (size_t)px * 4, dst_pitch, bw, bh);
                break;
            }
        }
    }
}
