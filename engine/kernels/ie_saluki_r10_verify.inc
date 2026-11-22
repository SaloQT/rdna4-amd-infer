/* ============================================================================
 * Round 10: Bitwise-exact T=8 verify-decode raw-quant GEMVs
 *   Addresses the top post-R9-win3 decode/verify bottlenecks on RX 9070 XT
 *   (gfx1201, 256 threads/WG, 32 rows/WG) using two proven zero-spill,
 *   high-occupancy architectures:
 *
 *   A. Coalesced 4-rows/wave + 0-bank-conflict LDS activation staging
 *      (`r4_lds`, 32 rows/WG, 8x fewer global activation loads, 100% coalesced
 *      contiguous weight loads, branchless 1-row software pipeline with zero
 *      `loadcnt` HOL blocking, <= 96 VGPRs, 0 spills):
 *        1. ie_saluki_iq2xxs_r4_lds  (IQ2_XXS, 10240 B LDS -> 6 WGs/WGP, ~74 VGPR)
 *        2. ie_saluki_q2k_r4_lds     (Q2_K,    10240 B LDS -> 6 WGs/WGP, ~86 VGPR;
 *                                     b128 weights + exact _Float16x2 lxs sums)
 *        3. ie_saluki_iq2s_r4_lds    (IQ2_S,   13312 B LDS -> 4 WGs/WGP, ~80 VGPR;
 *                                     packed 4 KiB u32 LUT + single-mul TR8_P2)
 *        4. ie_saluki_iq2xs_r4_lds   (IQ2_XS,  11264 B LDS -> 5 WGs/WGP, ~78 VGPR;
 *                                     packed 2 KiB u32 LUT + TR8_KSW)
 *        5. ie_saluki_iq3s_r4_lds    (IQ3_S,   11264 B LDS -> 5 WGs/WGP, ~82 VGPR;
 *                                     2 KiB u32 LUT + single-mul odd-grid ^0xFE)
 *
 *   B. Fixed 1-row/lane scalar-activation pipelined (`tr9_r10`, 32 rows/WG,
 *      hoisted xt[8] SGPRs, no per-4-tok asm barrier, branchless hot loop
 *      eliminating the j=3 CFG-merge `s_wait_loadcnt 0` stall, unioned 8 KiB
 *      LDS -> 8 WGs/WGP, <= 68 VGPRs, 0 spills):
 *        6. ie_saluki_iq2xxs_tr9_r10 (IQ2_XXS,  8192 B LDS -> 8 WGs/WGP, ~60 VGPR)
 *        7. ie_saluki_iq1s_tr9_r10   (IQ1_S,    8192 B LDS -> 8 WGs/WGP, ~62 VGPR;
 *                                     packed u32 LUT + exact folded 8*g +- 1)
 *        8. ie_saluki_iq1m_tr9_r10   (IQ1_M,    8192 B LDS -> 8 WGs/WGP, ~66 VGPR;
 *                                     packed u32 LUT + folded 8*g +- 1)
 *
 * Included at the end of kernels/ie_saluki_kernels.c (after Round 9).
 * All local helpers/types/macros are prefixed r10_v_ / R10_V_; all macros are
 * #undef'd at EOF.
 * Bitwise equal (0 bit diffs) to base ie_gemv_raw_*_tr8 and R0/R4/R8/R9.
 * ========================================================================== */

typedef _Float16 r10_v_f16x2 __attribute__((ext_vector_type(2)));

/* 4-byte-aligned u32x4 load helper (emits global_load_b128 on gfx1201 with
 * zero alignment UB for Q2_K's 84-byte stride blocks). */
struct __attribute__((packed, aligned(4))) r10_v_u32x4_p4 { u32x4 v; };
static inline __attribute__((always_inline)) u32x4 r10_v_ld_u32x4_a4(const G u8 *p, const u32 o) {
  return ((const G struct r10_v_u32x4_p4 *)(p + o))->v;
}

/* ============================================================================
 * Shared epilogues & reduction helpers (exact FP32 reduction order)
 * ========================================================================== */

/* Exact wave_tree_f reduction and epilogue for tr8 R=4 kernels.
 * Safe when r0 >= rows (trailing waves on non-multiple-of-32 rows). */
static inline __attribute__((always_inline)) void r10_v_tr8_epilogue4(
    const u32 r0, const u32 rows, const u32 l,
    float a[4][8], const G float *bias, const G float *res, G float *y, const u32 ys, const u32 rs) {
#pragma clang fp contract(off)
  float o[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    o[k] = 0.0f;
#pragma unroll
    for (u32 t = 0; t < 8u; t++) {
      const float sv = I2F(__builtin_amdgcn_readlane(F2I(wave_tree_f(a[k][t])), 0));
      if (l == t) o[k] = sv;
    }
  }
  if (l < 8u) {
    const G float *rt = res ? (const G float *)((const G u8 *)res + (unsigned long)l * rs) : res;
    G float *yt = (G float *)((G u8 *)y + (unsigned long)l * ys);
#pragma unroll
    for (u32 k = 0; k < 4u; k++) if (bias && r0 + k < rows) o[k] = o[k] + bias[r0 + k];
#pragma unroll
    for (u32 k = 0; k < 4u; k++) if (rt && r0 + k < rows) o[k] = rt[r0 + k] + o[k];
#pragma unroll
    for (u32 k = 0; k < 4u; k++) if (r0 + k < rows) yt[r0 + k] = o[k];
  }
}

/* Exact 5-level binary reduction and epilogue for tr9 (32 rows/WG).
 * Caller MUST execute lbarrier() before this if red is unioned with ulut. */
static inline __attribute__((always_inline)) void r10_v_tr9_reduce32(
    LDSP float *red, const u32 w, const u32 l, const u32 r0, const u32 rows,
    float a[4][8], const G float *bias, const G float *res, G float *y, const u32 ys, const u32 rs) {
#pragma clang fp contract(off)
#pragma unroll
  for (u32 t = 0; t < 8u; t++) red[(w * 8u + t) * 32u + l] = (a[0][t] + a[2][t]) + (a[1][t] + a[3][t]);
  lbarrier();
  const u32 i = tid();
  const u32 rr = i & 31u, t = i >> 5u, row = r0 + rr;
  float v[8];
#pragma unroll
  for (u32 x = 0; x < 8u; x++) v[x] = red[(x * 8u + t) * 32u + rr];
  const float x0 = v[0] + v[4], x1 = v[1] + v[5], x2 = v[2] + v[6], x3 = v[3] + v[7];
  float out = (x0 + x2) + (x1 + x3);
  if (row < rows) {
    if (bias) out = out + bias[row];
    if (res) out = ((const G float *)((const G u8 *)res + (unsigned long)t * rs))[row] + out;
    ((G float *)((G u8 *)y + (unsigned long)t * ys))[row] = out;
  }
}

/* ============================================================================
 * Shared r4_lds activation staging & LDS-backed 8-token dot helpers
 *   Layout: lx[(t * 8 + w) * 32 + l], ldav[t * 32 + l] (bank = l -> 0 conflicts)
 *   Each wave wv (0..7) stages token wv once per 32-subblock tile (8x less X
 *   global traffic). Reading lx/ldav uses dscnt, completely decoupled from the
 *   next row's weight global_load (loadcnt) -> zero loadcnt HOL blocking.
 * ========================================================================== */

static inline __attribute__((always_inline)) void r10_v_r4lds_stage_x(
    const G u8 *xb, const u32 xo, const u32 dao,
    LDSP u32 *lx, LDSP float *ldav, const u32 wv, const u32 l) {
  const float dav = *(const G float *)(xb + dao);
  const u32x4 x0 = *(const G u32x4 *)(xb + xo);
  const u32x4 x1 = *(const G u32x4 *)(xb + xo + 16u);
  const u32 base = wv * 256u + l;
  ldav[wv * 32u + l] = dav;
  lx[base + 0u] = x0.x;
  lx[base + 32u] = x0.y;
  lx[base + 64u] = x0.z;
  lx[base + 96u] = x0.w;
  lx[base + 128u] = x1.x;
  lx[base + 160u] = x1.y;
  lx[base + 192u] = x1.z;
  lx[base + 224u] = x1.w;
}

static inline __attribute__((always_inline)) void r10_v_r4lds_stage_x_q2k(
    const G u8 *xb, const u32 xo, const u32 dao,
    LDSP u32 *lx, LDSP float *ldav, LDSP u32 *lxs, const u32 wv, const u32 l) {
#pragma clang fp contract(off)
  const float dav = *(const G float *)(xb + dao);
  const u32x4 x0 = *(const G u32x4 *)(xb + xo);
  const u32x4 x1 = *(const G u32x4 *)(xb + xo + 16u);
  /* Every sum of 16 signed int8 activations lies in [-2048, +2032], which is
   * bitwise exact in IEEE-754 binary16 (_Float16). */
  const u32 xs0 = ie_dot4_ss(ie_dot4_ss(0u, x0.x, 0x01010101u), x0.y, 0x01010101u) +
                  ie_dot4_ss(ie_dot4_ss(0u, x0.z, 0x01010101u), x0.w, 0x01010101u);
  const u32 xs1 = ie_dot4_ss(ie_dot4_ss(0u, x1.x, 0x01010101u), x1.y, 0x01010101u) +
                  ie_dot4_ss(ie_dot4_ss(0u, x1.z, 0x01010101u), x1.w, 0x01010101u);
  const _Float16 h0 = (_Float16)(int)xs0;
  const _Float16 h1 = (_Float16)(int)xs1;
  const u32 pxs = (u32)__builtin_bit_cast(u16, h0) | ((u32)__builtin_bit_cast(u16, h1) << 16u);
  const u32 base = wv * 256u + l;
  ldav[wv * 32u + l] = dav;
  lxs[wv * 32u + l] = pxs;
  lx[base + 0u] = x0.x;
  lx[base + 32u] = x0.y;
  lx[base + 64u] = x0.z;
  lx[base + 96u] = x0.w;
  lx[base + 128u] = x1.x;
  lx[base + 160u] = x1.y;
  lx[base + 192u] = x1.z;
  lx[base + 224u] = x1.w;
}

static inline __attribute__((always_inline)) void r10_v_r4lds_dot8(
    const u32 wl[4], const u32 wh[4], const float dw,
    const LDSP u32 *lx, const LDSP float *ldav, const u32 l, float ak[8]) {
#pragma clang fp contract(off)
#pragma unroll
  for (u32 t = 0; t < 8u; t++) {
    const u32 base = t * 256u + l;
    const float dav = ldav[t * 32u + l];
    u32 idot = ie_dot4_ss(0u, wl[0], lx[base + 0u]);
    idot = ie_dot4_ss(idot, wh[0], lx[base + 32u]);
    idot = ie_dot4_ss(idot, wl[1], lx[base + 64u]);
    idot = ie_dot4_ss(idot, wh[1], lx[base + 96u]);
    idot = ie_dot4_ss(idot, wl[2], lx[base + 128u]);
    idot = ie_dot4_ss(idot, wh[2], lx[base + 160u]);
    idot = ie_dot4_ss(idot, wl[3], lx[base + 192u]);
    idot = ie_dot4_ss(idot, wh[3], lx[base + 224u]);
    ak[t] += dav * (dw * (float)(int)idot);
    /* Fence every 2 tokens: overlaps 18 ds_load_b32 with 24 VALU while capping
     * live activation VGPRs at <= 18 so total VGPRs stay <= 96 (0 spills). */
    if ((t & 1u) == 1u) __builtin_amdgcn_sched_barrier(0);
  }
}

static inline __attribute__((always_inline)) void r10_v_r4lds_dot8_split16(
    const u32 wl[4], const u32 wh[4], const float dw0, const float dw1,
    const LDSP u32 *lx, const LDSP float *ldav, const u32 l, float ak[8]) {
#pragma clang fp contract(off)
#pragma unroll
  for (u32 t = 0; t < 8u; t++) {
    const u32 base = t * 256u + l;
    const float dav = ldav[t * 32u + l];
    u32 id0 = ie_dot4_ss(0u, wl[0], lx[base + 0u]);
    id0 = ie_dot4_ss(id0, wh[0], lx[base + 32u]);
    id0 = ie_dot4_ss(id0, wl[1], lx[base + 64u]);
    id0 = ie_dot4_ss(id0, wh[1], lx[base + 96u]);
    u32 id1 = ie_dot4_ss(0u, wl[2], lx[base + 128u]);
    id1 = ie_dot4_ss(id1, wh[2], lx[base + 160u]);
    id1 = ie_dot4_ss(id1, wl[3], lx[base + 192u]);
    id1 = ie_dot4_ss(id1, wh[3], lx[base + 224u]);
    ak[t] += dav * (dw0 * (float)(int)id0 + dw1 * (float)(int)id1);
    if ((t & 1u) == 1u) __builtin_amdgcn_sched_barrier(0);
  }
}

/* ============================================================================
 * Shared tr9_r10 hoisted-xt scalar-load 8-token dot helpers
 *   Fixes R8/R9 tr9_pipe1 overhead: xt[0..7] is tr8_uniform'd ONCE before the
 *   loop (0 v_readfirstlane inside loop), and removes the per-4-token asm pin
 *   on aj so s_load of t=4..7 overlaps freely with t=0..3 (critical on nb=544).
 * ========================================================================== */

static inline __attribute__((always_inline)) void r10_v_tr9_dot8_sload(
    const u32 wl[4], const u32 wh[4], const float dw,
    const TR8_CSP u8 *const xt[8], const u32 xo, const u32 dao, float aj[8]) {
#pragma clang fp contract(off)
#pragma unroll
  for (u32 t = 0; t < 8u; t++) {
    const TR8_CSP u8 *xb = xt[t];
    const float dav = *(const TR8_CSP float *)(xb + dao);
    const u32x4 x0 = *(const TR8_CSP u32x4 *)(xb + xo);
    const u32x4 x1 = *(const TR8_CSP u32x4 *)(xb + xo + 16u);
    u32 idot = ie_dot4_ss(0u, wl[0], x0.x);
    idot = ie_dot4_ss(idot, wh[0], x0.y);
    idot = ie_dot4_ss(idot, wl[1], x0.z);
    idot = ie_dot4_ss(idot, wh[1], x0.w);
    idot = ie_dot4_ss(idot, wl[2], x1.x);
    idot = ie_dot4_ss(idot, wh[2], x1.y);
    idot = ie_dot4_ss(idot, wl[3], x1.z);
    idot = ie_dot4_ss(idot, wh[3], x1.w);
    aj[t] += dav * (dw * (float)(int)idot);
  }
}

static inline __attribute__((always_inline)) void r10_v_tr9_dot8_sload_split16(
    const u32 wl[4], const u32 wh[4], const float dw0, const float dw1,
    const TR8_CSP u8 *const xt[8], const u32 xo, const u32 dao, float aj[8]) {
#pragma clang fp contract(off)
#pragma unroll
  for (u32 t = 0; t < 8u; t++) {
    const TR8_CSP u8 *xb = xt[t];
    const float dav = *(const TR8_CSP float *)(xb + dao);
    const u32x4 x0 = *(const TR8_CSP u32x4 *)(xb + xo);
    const u32x4 x1 = *(const TR8_CSP u32x4 *)(xb + xo + 16u);
    u32 id0 = ie_dot4_ss(0u, wl[0], x0.x);
    id0 = ie_dot4_ss(id0, wh[0], x0.y);
    id0 = ie_dot4_ss(id0, wl[1], x0.z);
    id0 = ie_dot4_ss(id0, wh[1], x0.w);
    u32 id1 = ie_dot4_ss(0u, wl[2], x1.x);
    id1 = ie_dot4_ss(id1, wh[2], x1.y);
    id1 = ie_dot4_ss(id1, wl[3], x1.z);
    id1 = ie_dot4_ss(id1, wh[3], x1.w);
    aj[t] += dav * (dw0 * (float)(int)id0 + dw1 * (float)(int)id1);
  }
}

/* ============================================================================
 * Per-subtype decode helpers (100% bitwise exact, fp contract off)
 * ========================================================================== */

static inline __attribute__((always_inline)) void r10_v_decode_iq2xxs(
    const LDSP u32 *ulut, const u32 d16, const u32 a0, const u32 a1,
    u32 wl[4], u32 wh[4], float *dw_out) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d16);
  *dw_out = (d * 0.25f) * (0.5f + (float)(a1 >> 28u));
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 cp = ulut[ie_byte(a0, g)];
    TR8_KSW(cp & 0x03030303u, (cp >> 4u) & 0x03030303u, (a1 >> (7u * g)) & 127u, wl[g], wh[g]);
  }
}

static inline __attribute__((always_inline)) void r10_v_decode_iq2s(
    const LDSP u32 *ulut, const u32 d16, const u32 qlo, const u32 sgn,
    const u32 qh, const u32 sc, u32 wl[4], u32 wh[4], float *dw0_out, float *dw1_out) {
#pragma clang fp contract(off)
  const float d4 = ie_rq_h2f(d16) * 0.25f;
  *dw0_out = d4 * (0.5f + (float)(sc & 15u));
  *dw1_out = d4 * (0.5f + (float)(sc >> 4u));
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 idx = ie_byte(qlo, g) | ((qh << (8u - 2u * g)) & 0x300u);
    const u32 sb = ie_byte(sgn, g);
    const u32 cp = ulut[idx];
    const u32 p_ = (sb & 127u) * 0x00810204u;
    wl[g] = TR8_P2((p_ & 0x04040404u) | (cp & 0x03030303u));
    wh[g] = TR8_P2((((sb << 19u) | (p_ >> 4u)) & 0x04040404u) | ((cp >> 4u) & 0x03030303u));
  }
}

static inline __attribute__((always_inline)) void r10_v_decode_iq2xs(
    const LDSP u32 *ulut, const u32 d16, const u32 sc, const u32 q01, const u32 q23,
    u32 wl[4], u32 wh[4], float *dw0_out, float *dw1_out) {
#pragma clang fp contract(off)
  const float d4 = ie_rq_h2f(d16) * 0.25f;
  *dw0_out = d4 * (0.5f + (float)(sc & 15u));
  *dw1_out = d4 * (0.5f + (float)(sc >> 4u));
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 q2 = ((g < 2u ? q01 : q23) >> (16u * (g & 1u))) & 0xFFFFu;
    const u32 cp = ulut[q2 & 511u];
    TR8_KSW(cp & 0x03030303u, (cp >> 4u) & 0x03030303u, q2 >> 9u, wl[g], wh[g]);
  }
}

/* All 512 ie_rq_iq3s_grid entries hold strictly odd bytes ({1,3,5,7,9,11,13,15}),
 * so -g == g ^ 0xFE across all 4 bytes with zero carry. Single-mul 8-bit sign
 * decode (identical to r9_rq_iq3s_ksw). */
static inline __attribute__((always_inline)) void r10_v_decode_iq3s(
    const LDSP u32 *ulut, const u32 d16, const u32 qh, const u32 sc_b, const u32 sgn,
    const u32 q01, const u32 q23, const u32 sc_sh,
    u32 wl[4], u32 wh[4], float *dw_out) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d16);
  const u32 sc = (sc_b >> sc_sh) & 15u;
  *dw_out = d * (float)(int)(1u + 2u * sc);
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 qp = ((g < 2u ? q01 : q23) >> (16u * (g & 1u))) & 0xFFFFu;
    const u32 i1 = (qp & 255u) | (((qh >> (2u * g)) & 1u) << 8u);
    const u32 i2 = (qp >> 8u) | (((qh >> (2u * g + 1u)) & 1u) << 8u);
    const u32 g1 = ulut[i1], g2 = ulut[i2];
    const u32 sb = ie_byte(sgn, g);
    const u32 p_ = (sb & 127u) * 0x00204081u;
    const u32 t0 = p_ & 0x01010101u;
    const u32 t1 = ((p_ >> 4u) | (sb << 17u)) & 0x01010101u;
    wl[g] = g1 ^ __builtin_amdgcn_perm(0u, 0x0000FE00u, t0);
    wh[g] = g2 ^ __builtin_amdgcn_perm(0u, 0x0000FE00u, t1);
  }
}

/* Folded IQ1_S: weights (8*g +- 1) in {-9,-7,-1,+1,+7,+9} via rq_pack_iq1 +
 * rq_iq1_map, with exact power-of-2 scale (d * (2*ls+1)) * 0.125f.
 * Bitwise identical to dw * ((float)(int)idot + (+-0.125f) * (float)(int)as)
 * for every finite binary16 d and every int8 activation block (|8*I +- A| <= 36864 < 2^24,
 * and (+0.0f) + (-0.0f) == +0.0f). Eliminates all `as` loads and adds. */
static inline __attribute__((always_inline)) void r10_v_decode_iq1s(
    const LDSP u32 *ulut, const u32 d16, const u32 qlo, const u32 qh,
    u32 wl[4], u32 wh[4], float *dw8_out) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d16);
  *dw8_out = (d * (float)(int)(2u * ((qh >> 12u) & 7u) + 1u)) * 0.125f;
  const u32 tbl = (qh & 0x8000u) != 0u ? RQ_IQ1_TBL_NEG : RQ_IQ1_TBL_POS;
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 idx = ie_byte(qlo, g) | (((qh >> (3u * g)) & 7u) << 8u);
    const u32 cp = ulut[idx];
    wl[g] = rq_iq1_map(cp, tbl);
    wh[g] = rq_iq1_map(cp >> 4u, tbl);
  }
}

static inline __attribute__((always_inline)) void r10_v_decode_iq1m(
    const LDSP u32 *ulut, const u32 slo, const u32 shi, const u32 qhb, const u32 qlo,
    const int use_shi, const u32 sc_sh16, const u32 sh,
    u32 wl[4], u32 wh[4], float *dq0_out, float *dq1_out) {
#pragma clang fp contract(off)
  const u32 s0 = slo & 0xFFFFu, s1 = slo >> 16u, s2 = shi & 0xFFFFu, s3 = shi >> 16u;
  const float d = ie_rq_h2f((s0 >> 12u) | ((s1 >> 8u) & 0x00F0u) | ((s2 >> 4u) & 0x0F00u) | (s3 & 0xF000u));
  const u32 scv = (use_shi ? shi : slo) >> sc_sh16;
  *dq0_out = (d * (float)(int)(2u * ((scv >> sh) & 7u) + 1u)) * 0.125f;
  *dq1_out = (d * (float)(int)(2u * ((scv >> (sh + 3u)) & 7u) + 1u)) * 0.125f;
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 qh = g < 2u ? (qhb & 255u) : (qhb >> 8u);
    const u32 idx = ie_byte(qlo, g) | ((qh << (8u - 4u * (g & 1u))) & 0x700u);
    const u32 cp = ulut[idx];
    const u32 tbl = (qh & ((g & 1u) != 0u ? 0x80u : 0x08u)) != 0u ? RQ_IQ1_TBL_NEG : RQ_IQ1_TBL_POS;
    wl[g] = rq_iq1_map(cp, tbl);
    wh[g] = rq_iq1_map(cp >> 4u, tbl);
  }
}

static inline __attribute__((always_inline)) void r10_v_q2k_r4lds_dot8(
    const u32 d_mn, const u32 sc16, const u32x4 q0_raw, const u32x4 q1_raw,
    const u32 sh, const u32 qmask,
    const LDSP u32 *lx, const LDSP float *ldav, const LDSP u32 *lxs, const u32 l, float ak[8]) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d_mn & 0xFFFFu);
  const float mn = ie_rq_h2f(d_mn >> 16u);
  const float dw0 = d * (float)(sc16 & 15u), dm0 = mn * (float)((sc16 >> 4u) & 15u);
  const float dw1 = d * (float)((sc16 >> 8u) & 15u), dm1 = mn * (float)(sc16 >> 12u);
  const u32 q00 = q0_raw.x & qmask, q01 = q0_raw.y & qmask;
  const u32 q02 = q0_raw.z & qmask, q03 = q0_raw.w & qmask;
  const u32 q10 = q1_raw.x & qmask, q11 = q1_raw.y & qmask;
  const u32 q12 = q1_raw.z & qmask, q13 = q1_raw.w & qmask;
#pragma unroll
  for (u32 t = 0; t < 8u; t++) {
    const u32 base = t * 256u + l;
    const float dav = ldav[t * 32u + l];
    const u32 pxs = lxs[t * 32u + l];
    const r10_v_f16x2 hs = __builtin_bit_cast(r10_v_f16x2, pxs);
    const float xsf0 = (float)hs.x;
    const float xsf1 = (float)hs.y;
    u32 id0 = ie_dot4_us(0u, q00, lx[base + 0u]);
    id0 = ie_dot4_us(id0, q01, lx[base + 32u]);
    id0 = ie_dot4_us(id0, q02, lx[base + 64u]);
    id0 = ie_dot4_us(id0, q03, lx[base + 96u]);
    const float t0 = dav * (dw0 * (float)((int)id0 >> sh) - dm0 * xsf0);
    u32 id1 = ie_dot4_us(0u, q10, lx[base + 128u]);
    id1 = ie_dot4_us(id1, q11, lx[base + 160u]);
    id1 = ie_dot4_us(id1, q12, lx[base + 192u]);
    id1 = ie_dot4_us(id1, q13, lx[base + 224u]);
    const float t1 = dav * (dw1 * (float)((int)id1 >> sh) - dm1 * xsf1);
    ak[t] += (t0 + t1);
    if ((t & 1u) == 1u) __builtin_amdgcn_sched_barrier(0);
  }
}

/* ============================================================================
 * R10-V1: ie_saluki_iq2xxs_r4_lds
 *   T=8 IQ2_XXS, 4 rows/wave (32 rows/WG), 100% coalesced 264B/row loads,
 *   0-conflict LDS activation staging (lx + ldav + ulut = 10240 B -> 6 WGs/WGP,
 *   fitting all 160 WGs of 5120x17408 in 1 full wave!), branchless 1-row pipe1.
 * ========================================================================== */
KERNEL ie_saluki_iq2xxs_r4_lds(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                               const G float *bias, const G float *res, const G float *nw, G float *ny,
                               G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  const u32 wg_r0 = 32u * wgid();
  if (T != 8u || wg_r0 >= rows) return;
  static LDS u32 ulut[256] __attribute__((aligned(16)));
  static LDS u32 lx[8u * 8u * 32u] __attribute__((aligned(128)));
  static LDS float ldav[8u * 32u] __attribute__((aligned(128)));
  ulut[tid()] = rq_pack_iq2(ie_rq_iq2xxs_grid[tid()]);
  lbarrier();

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 r0 = ie_gemv_row(wgid(), wv) * 4u, l = lane();
  const u32 mb = nb >> 3u;
  const G u8 *row[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 rr = (r0 + k < rows) ? (r0 + k) : 0u;
    row[k] = tr8_uniform(q + (unsigned long)rr * mb * 66u);
  }
  const G u8 *xw = tr8_uniform(xq + (unsigned long)wv * xs);
  float a[4][8];
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[k][t] = 0.0f;

  const u32 s = l & 7u;
  u32 off_d = (l >> 3u) * 66u;
  u32 off_q = off_d + 2u + 8u * s;
  u32 xo = 32u * l, dao = 32u * nb + 4u * l;
  const u32 nb32 = nb & ~31u;

  if (nb32 > 0u) {
    u32 d_c = ie_rq_a16(row[0] + off_d, 0u);
    u32 a0_c = ie_rq_a32(row[0] + off_q, 0u);
    u32 a1_c = ie_rq_a32(row[0] + off_q, 4u);
    __asm__ volatile("" : "+v"(d_c), "+v"(a0_c), "+v"(a1_c));

    for (u32 b0 = 0; b0 + 32u < nb32; b0 += 32u) {
      r10_v_r4lds_stage_x(xw, xo, dao, lx, ldav, wv, l);
      lbarrier();
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        u32 d_n, a0_n, a1_n;
        if (k < 3u) {
          d_n = ie_rq_a16(row[k + 1u] + off_d, 0u);
          a0_n = ie_rq_a32(row[k + 1u] + off_q, 0u);
          a1_n = ie_rq_a32(row[k + 1u] + off_q, 4u);
        } else {
          d_n = ie_rq_a16(row[0] + off_d, 264u);
          a0_n = ie_rq_a32(row[0] + off_q, 264u);
          a1_n = ie_rq_a32(row[0] + off_q, 268u);
        }
        __builtin_amdgcn_sched_barrier(0);
        u32 wl[4], wh[4]; float dw;
        r10_v_decode_iq2xxs(ulut, d_c, a0_c, a1_c, wl, wh, &dw);
        r10_v_r4lds_dot8(wl, wh, dw, lx, ldav, l, a[k]);
        d_c = d_n; a0_c = a0_n; a1_c = a1_n;
      }
      lbarrier();
      off_d += 264u; off_q += 264u;
      xo += 1024u; dao += 128u;
    }

    /* Last full 32-subblock tile (branchless: k=0..2 prefetch k+1, k=3 none). */
    r10_v_r4lds_stage_x(xw, xo, dao, lx, ldav, wv, l);
    lbarrier();
#pragma unroll
    for (u32 k = 0; k < 3u; k++) {
      const u32 d_n = ie_rq_a16(row[k + 1u] + off_d, 0u);
      const u32 a0_n = ie_rq_a32(row[k + 1u] + off_q, 0u);
      const u32 a1_n = ie_rq_a32(row[k + 1u] + off_q, 4u);
      __builtin_amdgcn_sched_barrier(0);
      u32 wl[4], wh[4]; float dw;
      r10_v_decode_iq2xxs(ulut, d_c, a0_c, a1_c, wl, wh, &dw);
      r10_v_r4lds_dot8(wl, wh, dw, lx, ldav, l, a[k]);
      d_c = d_n; a0_c = a0_n; a1_c = a1_n;
    }
    {
      u32 wl[4], wh[4]; float dw;
      r10_v_decode_iq2xxs(ulut, d_c, a0_c, a1_c, wl, wh, &dw);
      r10_v_r4lds_dot8(wl, wh, dw, lx, ldav, l, a[3]);
    }
    off_d += 264u; off_q += 264u;
    xo += 1024u; dao += 128u;
  }

  if (nb32 < nb) {
    if (nb32 > 0u) lbarrier();
    if (nb32 + l < nb) r10_v_r4lds_stage_x(xw, xo, dao, lx, ldav, wv, l);
    lbarrier();
    if (nb32 + l < nb) {
      for (u32 k = 0; k < 4u; k++) {
        u32 wl[4], wh[4]; float dw;
        r10_v_decode_iq2xxs(ulut, ie_rq_a16(row[k] + off_d, 0u),
                            ie_rq_a32(row[k] + off_q, 0u), ie_rq_a32(row[k] + off_q, 4u),
                            wl, wh, &dw);
        r10_v_r4lds_dot8(wl, wh, dw, lx, ldav, l, a[k]);
      }
    }
  }
  r10_v_tr8_epilogue4(r0, rows, l, a, bias, res, y, ys, rs);
}

/* ============================================================================
 * R10-V2: ie_saluki_q2k_r4_lds
 *   T=8 Q2_K, 4 rows/wave (32 rows/WG), b128 coalesced weights (4 vs 11 loads),
 *   0-conflict LDS staging for lx + ldav + exact _Float16x2 lxs (10240 B LDS ->
 *   6 WGs/WGP = 2x occupancy over R4's 195-VGPR 3 WGs/WGP!), 8x less X loads
 *   and xs0/xs1 ALU, branchless 1-row pipe1 with zero loadcnt HOL blocking.
 * ========================================================================== */
KERNEL ie_saluki_q2k_r4_lds(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                            const G float *bias, const G float *res, const G float *nw, G float *ny,
                            G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  const u32 wg_r0 = 32u * wgid();
  if (T != 8u || wg_r0 >= rows) return;
  static LDS u32 lx[8u * 8u * 32u] __attribute__((aligned(128)));
  static LDS float ldav[8u * 32u] __attribute__((aligned(128)));
  static LDS u32 lxs[8u * 32u] __attribute__((aligned(128)));

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 r0 = ie_gemv_row(wgid(), wv) * 4u, l = lane();
  const u32 mb = nb >> 3u;
  const G u8 *row[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 rr = (r0 + k < rows) ? (r0 + k) : 0u;
    row[k] = tr8_uniform(q + (unsigned long)rr * mb * 84u);
  }
  const G u8 *xw = tr8_uniform(xq + (unsigned long)wv * xs);
  float a[4][8];
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[k][t] = 0.0f;

  const u32 s = l & 7u;
  const u32 sh = 2u * (s & 3u), qmask = 0x03030303u << sh;
  u32 off_d = (l >> 3u) * 84u;
  u32 off_ps = off_d + 2u * s;
  u32 off_pq = off_d + 16u + (s >> 2u) * 32u;
  u32 xo = 32u * l, dao = 32u * nb + 4u * l;
  const u32 nb32 = nb & ~31u;

  if (nb32 > 0u) {
    u32 d_mn_c = ie_rq_a32(row[0] + off_d, 80u);
    u32 sc16_c = ie_rq_a16(row[0] + off_ps, 0u);
    u32x4 q0_c = r10_v_ld_u32x4_a4(row[0] + off_pq, 0u);
    u32x4 q1_c = r10_v_ld_u32x4_a4(row[0] + off_pq, 16u);
    __asm__ volatile("" : "+v"(d_mn_c), "+v"(sc16_c), "+v"(q0_c), "+v"(q1_c));

    for (u32 b0 = 0; b0 + 32u < nb32; b0 += 32u) {
      r10_v_r4lds_stage_x_q2k(xw, xo, dao, lx, ldav, lxs, wv, l);
      lbarrier();
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        u32 d_mn_n, sc16_n;
        u32x4 q0_n, q1_n;
        if (k < 3u) {
          d_mn_n = ie_rq_a32(row[k + 1u] + off_d, 80u);
          sc16_n = ie_rq_a16(row[k + 1u] + off_ps, 0u);
          q0_n = r10_v_ld_u32x4_a4(row[k + 1u] + off_pq, 0u);
          q1_n = r10_v_ld_u32x4_a4(row[k + 1u] + off_pq, 16u);
        } else {
          d_mn_n = ie_rq_a32(row[0] + off_d, 416u);
          sc16_n = ie_rq_a16(row[0] + off_ps, 336u);
          q0_n = r10_v_ld_u32x4_a4(row[0] + off_pq, 336u);
          q1_n = r10_v_ld_u32x4_a4(row[0] + off_pq, 352u);
        }
        __builtin_amdgcn_sched_barrier(0);
        r10_v_q2k_r4lds_dot8(d_mn_c, sc16_c, q0_c, q1_c, sh, qmask, lx, ldav, lxs, l, a[k]);
        d_mn_c = d_mn_n; sc16_c = sc16_n; q0_c = q0_n; q1_c = q1_n;
      }
      lbarrier();
      off_d += 336u; off_ps += 336u; off_pq += 336u;
      xo += 1024u; dao += 128u;
    }

    r10_v_r4lds_stage_x_q2k(xw, xo, dao, lx, ldav, lxs, wv, l);
    lbarrier();
#pragma unroll
    for (u32 k = 0; k < 3u; k++) {
      const u32 d_mn_n = ie_rq_a32(row[k + 1u] + off_d, 80u);
      const u32 sc16_n = ie_rq_a16(row[k + 1u] + off_ps, 0u);
      const u32x4 q0_n = r10_v_ld_u32x4_a4(row[k + 1u] + off_pq, 0u);
      const u32x4 q1_n = r10_v_ld_u32x4_a4(row[k + 1u] + off_pq, 16u);
      __builtin_amdgcn_sched_barrier(0);
      r10_v_q2k_r4lds_dot8(d_mn_c, sc16_c, q0_c, q1_c, sh, qmask, lx, ldav, lxs, l, a[k]);
      d_mn_c = d_mn_n; sc16_c = sc16_n; q0_c = q0_n; q1_c = q1_n;
    }
    r10_v_q2k_r4lds_dot8(d_mn_c, sc16_c, q0_c, q1_c, sh, qmask, lx, ldav, lxs, l, a[3]);
    off_d += 336u; off_ps += 336u; off_pq += 336u;
    xo += 1024u; dao += 128u;
  }

  if (nb32 < nb) {
    if (nb32 > 0u) lbarrier();
    if (nb32 + l < nb) r10_v_r4lds_stage_x_q2k(xw, xo, dao, lx, ldav, lxs, wv, l);
    lbarrier();
    if (nb32 + l < nb) {
      for (u32 k = 0; k < 4u; k++) {
        r10_v_q2k_r4lds_dot8(ie_rq_a32(row[k] + off_d, 80u), ie_rq_a16(row[k] + off_ps, 0u),
                             r10_v_ld_u32x4_a4(row[k] + off_pq, 0u), r10_v_ld_u32x4_a4(row[k] + off_pq, 16u),
                             sh, qmask, lx, ldav, lxs, l, a[k]);
      }
    }
  }
  r10_v_tr8_epilogue4(r0, rows, l, a, bias, res, y, ys, rs);
}

/* ============================================================================
 * R10-V3: ie_saluki_iq2s_r4_lds
 *   T=8 IQ2_S, 4 rows/wave (32 rows/WG), packed 4 KiB u32 LUT (half bank
 *   conflicts vs u64) + 0-conflict LDS X (13312 B -> 4 WGs/WGP vs R0's 3!),
 *   100% coalesced 328 B/row loads, single-mul TR8_P2 decode, branchless pipe1.
 * ========================================================================== */
KERNEL ie_saluki_iq2s_r4_lds(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                             const G float *bias, const G float *res, const G float *nw, G float *ny,
                             G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  const u32 wg_r0 = 32u * wgid();
  if (T != 8u || wg_r0 >= rows) return;
  static LDS u32 ulut[1024] __attribute__((aligned(16)));
  static LDS u32 lx[8u * 8u * 32u] __attribute__((aligned(128)));
  static LDS float ldav[8u * 32u] __attribute__((aligned(128)));
  for (u32 i = tid(); i < 1024u; i += NT) ulut[i] = rq_pack_iq2(ie_rq_iq2s_grid[i]);
  lbarrier();

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 r0 = ie_gemv_row(wgid(), wv) * 4u, l = lane();
  const u32 mb = nb >> 3u;
  const G u8 *row[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 rr = (r0 + k < rows) ? (r0 + k) : 0u;
    row[k] = tr8_uniform(q + (unsigned long)rr * mb * 82u);
  }
  const G u8 *xw = tr8_uniform(xq + (unsigned long)wv * xs);
  float a[4][8];
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[k][t] = 0.0f;

  const u32 s = l & 7u;
  u32 off_d = (l >> 3u) * 82u;
  u32 off_qlo = off_d + 2u + 4u * s;
  u32 off_qh = off_d + 66u + s;
  u32 xo = 32u * l, dao = 32u * nb + 4u * l;
  const u32 nb32 = nb & ~31u;

  if (nb32 > 0u) {
    u32 d_c = ie_rq_a16(row[0] + off_d, 0u);
    u32 qlo_c = ie_rq_a32(row[0] + off_qlo, 0u);
    u32 sgn_c = ie_rq_a32(row[0] + off_qlo, 32u);
    u32 qh_c = (u32)(row[0] + off_qh)[0];
    u32 sc_c = (u32)(row[0] + off_qh)[8];
    __asm__ volatile("" : "+v"(d_c), "+v"(qlo_c), "+v"(sgn_c), "+v"(qh_c), "+v"(sc_c));

    for (u32 b0 = 0; b0 + 32u < nb32; b0 += 32u) {
      r10_v_r4lds_stage_x(xw, xo, dao, lx, ldav, wv, l);
      lbarrier();
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        u32 d_n, qlo_n, sgn_n, qh_n, sc_n;
        if (k < 3u) {
          d_n = ie_rq_a16(row[k + 1u] + off_d, 0u);
          qlo_n = ie_rq_a32(row[k + 1u] + off_qlo, 0u);
          sgn_n = ie_rq_a32(row[k + 1u] + off_qlo, 32u);
          qh_n = (u32)(row[k + 1u] + off_qh)[0];
          sc_n = (u32)(row[k + 1u] + off_qh)[8];
        } else {
          d_n = ie_rq_a16(row[0] + off_d, 328u);
          qlo_n = ie_rq_a32(row[0] + off_qlo, 328u);
          sgn_n = ie_rq_a32(row[0] + off_qlo, 360u);
          qh_n = (u32)(row[0] + off_qh)[328];
          sc_n = (u32)(row[0] + off_qh)[336];
        }
        __builtin_amdgcn_sched_barrier(0);
        u32 wl[4], wh[4]; float dw0, dw1;
        r10_v_decode_iq2s(ulut, d_c, qlo_c, sgn_c, qh_c, sc_c, wl, wh, &dw0, &dw1);
        r10_v_r4lds_dot8_split16(wl, wh, dw0, dw1, lx, ldav, l, a[k]);
        d_c = d_n; qlo_c = qlo_n; sgn_c = sgn_n; qh_c = qh_n; sc_c = sc_n;
      }
      lbarrier();
      off_d += 328u; off_qlo += 328u; off_qh += 328u;
      xo += 1024u; dao += 128u;
    }

    r10_v_r4lds_stage_x(xw, xo, dao, lx, ldav, wv, l);
    lbarrier();
#pragma unroll
    for (u32 k = 0; k < 3u; k++) {
      const u32 d_n = ie_rq_a16(row[k + 1u] + off_d, 0u);
      const u32 qlo_n = ie_rq_a32(row[k + 1u] + off_qlo, 0u);
      const u32 sgn_n = ie_rq_a32(row[k + 1u] + off_qlo, 32u);
      const u32 qh_n = (u32)(row[k + 1u] + off_qh)[0];
      const u32 sc_n = (u32)(row[k + 1u] + off_qh)[8];
      __builtin_amdgcn_sched_barrier(0);
      u32 wl[4], wh[4]; float dw0, dw1;
      r10_v_decode_iq2s(ulut, d_c, qlo_c, sgn_c, qh_c, sc_c, wl, wh, &dw0, &dw1);
      r10_v_r4lds_dot8_split16(wl, wh, dw0, dw1, lx, ldav, l, a[k]);
      d_c = d_n; qlo_c = qlo_n; sgn_c = sgn_n; qh_c = qh_n; sc_c = sc_n;
    }
    {
      u32 wl[4], wh[4]; float dw0, dw1;
      r10_v_decode_iq2s(ulut, d_c, qlo_c, sgn_c, qh_c, sc_c, wl, wh, &dw0, &dw1);
      r10_v_r4lds_dot8_split16(wl, wh, dw0, dw1, lx, ldav, l, a[3]);
    }
    off_d += 328u; off_qlo += 328u; off_qh += 328u;
    xo += 1024u; dao += 128u;
  }

  if (nb32 < nb) {
    if (nb32 > 0u) lbarrier();
    if (nb32 + l < nb) r10_v_r4lds_stage_x(xw, xo, dao, lx, ldav, wv, l);
    lbarrier();
    if (nb32 + l < nb) {
      for (u32 k = 0; k < 4u; k++) {
        u32 wl[4], wh[4]; float dw0, dw1;
        r10_v_decode_iq2s(ulut, ie_rq_a16(row[k] + off_d, 0u),
                          ie_rq_a32(row[k] + off_qlo, 0u), ie_rq_a32(row[k] + off_qlo, 32u),
                          (u32)(row[k] + off_qh)[0], (u32)(row[k] + off_qh)[8],
                          wl, wh, &dw0, &dw1);
        r10_v_r4lds_dot8_split16(wl, wh, dw0, dw1, lx, ldav, l, a[k]);
      }
    }
  }
  r10_v_tr8_epilogue4(r0, rows, l, a, bias, res, y, ys, rs);
}

/* ============================================================================
 * R10-V4: ie_saluki_iq2xs_r4_lds
 *   T=8 IQ2_XS, 4 rows/wave (32 rows/WG), packed 2 KiB u32 LUT + 0-conflict
 *   LDS X (11264 B -> 5 WGs/WGP), 100% coalesced 296 B/row loads, branchless
 *   1-row pipe1 with zero loadcnt HOL blocking.
 * ========================================================================== */
KERNEL ie_saluki_iq2xs_r4_lds(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                              const G float *bias, const G float *res, const G float *nw, G float *ny,
                              G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  const u32 wg_r0 = 32u * wgid();
  if (T != 8u || wg_r0 >= rows) return;
  static LDS u32 ulut[512] __attribute__((aligned(16)));
  static LDS u32 lx[8u * 8u * 32u] __attribute__((aligned(128)));
  static LDS float ldav[8u * 32u] __attribute__((aligned(128)));
  ulut[tid()] = rq_pack_iq2(ie_rq_iq2xs_grid[tid()]);
  ulut[tid() + 256u] = rq_pack_iq2(ie_rq_iq2xs_grid[tid() + 256u]);
  lbarrier();

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 r0 = ie_gemv_row(wgid(), wv) * 4u, l = lane();
  const u32 mb = nb >> 3u;
  const G u8 *row[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 rr = (r0 + k < rows) ? (r0 + k) : 0u;
    row[k] = tr8_uniform(q + (unsigned long)rr * mb * 74u);
  }
  const G u8 *xw = tr8_uniform(xq + (unsigned long)wv * xs);
  float a[4][8];
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[k][t] = 0.0f;

  const u32 s = l & 7u;
  u32 off_d = (l >> 3u) * 74u;
  u32 off_q = off_d + 2u + 8u * s;
  u32 off_sc = off_d + 66u + s;
  u32 xo = 32u * l, dao = 32u * nb + 4u * l;
  const u32 nb32 = nb & ~31u;

  if (nb32 > 0u) {
    u32 d_c = ie_rq_a16(row[0] + off_d, 0u);
    u32 sc_c = (u32)(row[0] + off_sc)[0];
    u32 q01_c = ie_rq_a32(row[0] + off_q, 0u);
    u32 q23_c = ie_rq_a32(row[0] + off_q, 4u);
    __asm__ volatile("" : "+v"(d_c), "+v"(sc_c), "+v"(q01_c), "+v"(q23_c));

    for (u32 b0 = 0; b0 + 32u < nb32; b0 += 32u) {
      r10_v_r4lds_stage_x(xw, xo, dao, lx, ldav, wv, l);
      lbarrier();
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        u32 d_n, sc_n, q01_n, q23_n;
        if (k < 3u) {
          d_n = ie_rq_a16(row[k + 1u] + off_d, 0u);
          sc_n = (u32)(row[k + 1u] + off_sc)[0];
          q01_n = ie_rq_a32(row[k + 1u] + off_q, 0u);
          q23_n = ie_rq_a32(row[k + 1u] + off_q, 4u);
        } else {
          d_n = ie_rq_a16(row[0] + off_d, 296u);
          sc_n = (u32)(row[0] + off_sc)[296];
          q01_n = ie_rq_a32(row[0] + off_q, 296u);
          q23_n = ie_rq_a32(row[0] + off_q, 300u);
        }
        __builtin_amdgcn_sched_barrier(0);
        u32 wl[4], wh[4]; float dw0, dw1;
        r10_v_decode_iq2xs(ulut, d_c, sc_c, q01_c, q23_c, wl, wh, &dw0, &dw1);
        r10_v_r4lds_dot8_split16(wl, wh, dw0, dw1, lx, ldav, l, a[k]);
        d_c = d_n; sc_c = sc_n; q01_c = q01_n; q23_c = q23_n;
      }
      lbarrier();
      off_d += 296u; off_q += 296u; off_sc += 296u;
      xo += 1024u; dao += 128u;
    }

    r10_v_r4lds_stage_x(xw, xo, dao, lx, ldav, wv, l);
    lbarrier();
#pragma unroll
    for (u32 k = 0; k < 3u; k++) {
      const u32 d_n = ie_rq_a16(row[k + 1u] + off_d, 0u);
      const u32 sc_n = (u32)(row[k + 1u] + off_sc)[0];
      const u32 q01_n = ie_rq_a32(row[k + 1u] + off_q, 0u);
      const u32 q23_n = ie_rq_a32(row[k + 1u] + off_q, 4u);
      __builtin_amdgcn_sched_barrier(0);
      u32 wl[4], wh[4]; float dw0, dw1;
      r10_v_decode_iq2xs(ulut, d_c, sc_c, q01_c, q23_c, wl, wh, &dw0, &dw1);
      r10_v_r4lds_dot8_split16(wl, wh, dw0, dw1, lx, ldav, l, a[k]);
      d_c = d_n; sc_c = sc_n; q01_c = q01_n; q23_c = q23_n;
    }
    {
      u32 wl[4], wh[4]; float dw0, dw1;
      r10_v_decode_iq2xs(ulut, d_c, sc_c, q01_c, q23_c, wl, wh, &dw0, &dw1);
      r10_v_r4lds_dot8_split16(wl, wh, dw0, dw1, lx, ldav, l, a[3]);
    }
    off_d += 296u; off_q += 296u; off_sc += 296u;
    xo += 1024u; dao += 128u;
  }

  if (nb32 < nb) {
    if (nb32 > 0u) lbarrier();
    if (nb32 + l < nb) r10_v_r4lds_stage_x(xw, xo, dao, lx, ldav, wv, l);
    lbarrier();
    if (nb32 + l < nb) {
      for (u32 k = 0; k < 4u; k++) {
        u32 wl[4], wh[4]; float dw0, dw1;
        r10_v_decode_iq2xs(ulut, ie_rq_a16(row[k] + off_d, 0u), (u32)(row[k] + off_sc)[0],
                           ie_rq_a32(row[k] + off_q, 0u), ie_rq_a32(row[k] + off_q, 4u),
                           wl, wh, &dw0, &dw1);
        r10_v_r4lds_dot8_split16(wl, wh, dw0, dw1, lx, ldav, l, a[k]);
      }
    }
  }
  r10_v_tr8_epilogue4(r0, rows, l, a, bias, res, y, ys, rs);
}

/* ============================================================================
 * R10-V5: ie_saluki_iq3s_r4_lds
 *   T=8 IQ3_S, 4 rows/wave (32 rows/WG), 2 KiB u32 LUT + 0-conflict LDS X
 *   (11264 B -> 5 WGs/WGP), hoisted single-mul odd-grid ^0xFE sign decode,
 *   coalesced 440 B/row loads, branchless 1-row pipe1. Replaces unhoisted r2
 *   on 17408x5120 (180.1 ms) and base tr8 on 5120x17408 (58.3 ms).
 * ========================================================================== */
KERNEL ie_saluki_iq3s_r4_lds(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                             const G float *bias, const G float *res, const G float *nw, G float *ny,
                             G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  const u32 wg_r0 = 32u * wgid();
  if (T != 8u || wg_r0 >= rows) return;
  static LDS u32 ulut[512] __attribute__((aligned(16)));
  static LDS u32 lx[8u * 8u * 32u] __attribute__((aligned(128)));
  static LDS float ldav[8u * 32u] __attribute__((aligned(128)));
  ulut[tid()] = ie_rq_iq3s_grid[tid()];
  ulut[tid() + 256u] = ie_rq_iq3s_grid[tid() + 256u];
  lbarrier();

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 r0 = ie_gemv_row(wgid(), wv) * 4u, l = lane();
  const u32 mb = nb >> 3u;
  const G u8 *row[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 rr = (r0 + k < rows) ? (r0 + k) : 0u;
    row[k] = tr8_uniform(q + (unsigned long)rr * mb * 110u);
  }
  const G u8 *xw = tr8_uniform(xq + (unsigned long)wv * xs);
  float a[4][8];
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[k][t] = 0.0f;

  const u32 s = l & 7u, sc_sh = 4u * (s & 1u);
  u32 off_d = (l >> 3u) * 110u;
  u32 off_q = off_d + 2u + 8u * s;
  u32 off_qh = off_d + 66u + s;
  u32 off_sgn = off_d + 74u + 4u * s;
  u32 off_sc = off_d + 106u + (s >> 1u);
  u32 xo = 32u * l, dao = 32u * nb + 4u * l;
  const u32 nb32 = nb & ~31u;

  if (nb32 > 0u) {
    u32 d_c = ie_rq_a16(row[0] + off_d, 0u);
    u32 qh_c = (u32)(row[0] + off_qh)[0];
    u32 sc_c = (u32)(row[0] + off_sc)[0];
    u32 sgn_c = ie_rq_a32(row[0] + off_sgn, 0u);
    u32 q01_c = ie_rq_a32(row[0] + off_q, 0u);
    u32 q23_c = ie_rq_a32(row[0] + off_q, 4u);
    __asm__ volatile("" : "+v"(d_c), "+v"(qh_c), "+v"(sc_c), "+v"(sgn_c), "+v"(q01_c), "+v"(q23_c));

    for (u32 b0 = 0; b0 + 32u < nb32; b0 += 32u) {
      r10_v_r4lds_stage_x(xw, xo, dao, lx, ldav, wv, l);
      lbarrier();
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        u32 d_n, qh_n, sc_n, sgn_n, q01_n, q23_n;
        if (k < 3u) {
          d_n = ie_rq_a16(row[k + 1u] + off_d, 0u);
          qh_n = (u32)(row[k + 1u] + off_qh)[0];
          sc_n = (u32)(row[k + 1u] + off_sc)[0];
          sgn_n = ie_rq_a32(row[k + 1u] + off_sgn, 0u);
          q01_n = ie_rq_a32(row[k + 1u] + off_q, 0u);
          q23_n = ie_rq_a32(row[k + 1u] + off_q, 4u);
        } else {
          d_n = ie_rq_a16(row[0] + off_d, 440u);
          qh_n = (u32)(row[0] + off_qh)[440];
          sc_n = (u32)(row[0] + off_sc)[440];
          sgn_n = ie_rq_a32(row[0] + off_sgn, 440u);
          q01_n = ie_rq_a32(row[0] + off_q, 440u);
          q23_n = ie_rq_a32(row[0] + off_q, 444u);
        }
        __builtin_amdgcn_sched_barrier(0);
        u32 wl[4], wh[4]; float dw;
        r10_v_decode_iq3s(ulut, d_c, qh_c, sc_c, sgn_c, q01_c, q23_c, sc_sh, wl, wh, &dw);
        r10_v_r4lds_dot8(wl, wh, dw, lx, ldav, l, a[k]);
        d_c = d_n; qh_c = qh_n; sc_c = sc_n; sgn_c = sgn_n; q01_c = q01_n; q23_c = q23_n;
      }
      lbarrier();
      off_d += 440u; off_q += 440u; off_qh += 440u; off_sgn += 440u; off_sc += 440u;
      xo += 1024u; dao += 128u;
    }

    r10_v_r4lds_stage_x(xw, xo, dao, lx, ldav, wv, l);
    lbarrier();
#pragma unroll
    for (u32 k = 0; k < 3u; k++) {
      const u32 d_n = ie_rq_a16(row[k + 1u] + off_d, 0u);
      const u32 qh_n = (u32)(row[k + 1u] + off_qh)[0];
      const u32 sc_n = (u32)(row[k + 1u] + off_sc)[0];
      const u32 sgn_n = ie_rq_a32(row[k + 1u] + off_sgn, 0u);
      const u32 q01_n = ie_rq_a32(row[k + 1u] + off_q, 0u);
      const u32 q23_n = ie_rq_a32(row[k + 1u] + off_q, 4u);
      __builtin_amdgcn_sched_barrier(0);
      u32 wl[4], wh[4]; float dw;
      r10_v_decode_iq3s(ulut, d_c, qh_c, sc_c, sgn_c, q01_c, q23_c, sc_sh, wl, wh, &dw);
      r10_v_r4lds_dot8(wl, wh, dw, lx, ldav, l, a[k]);
      d_c = d_n; qh_c = qh_n; sc_c = sc_n; sgn_c = sgn_n; q01_c = q01_n; q23_c = q23_n;
    }
    {
      u32 wl[4], wh[4]; float dw;
      r10_v_decode_iq3s(ulut, d_c, qh_c, sc_c, sgn_c, q01_c, q23_c, sc_sh, wl, wh, &dw);
      r10_v_r4lds_dot8(wl, wh, dw, lx, ldav, l, a[3]);
    }
    off_d += 440u; off_q += 440u; off_qh += 440u; off_sgn += 440u; off_sc += 440u;
    xo += 1024u; dao += 128u;
  }

  if (nb32 < nb) {
    if (nb32 > 0u) lbarrier();
    if (nb32 + l < nb) r10_v_r4lds_stage_x(xw, xo, dao, lx, ldav, wv, l);
    lbarrier();
    if (nb32 + l < nb) {
      for (u32 k = 0; k < 4u; k++) {
        u32 wl[4], wh[4]; float dw;
        r10_v_decode_iq3s(ulut, ie_rq_a16(row[k] + off_d, 0u), (u32)(row[k] + off_qh)[0],
                          (u32)(row[k] + off_sc)[0], ie_rq_a32(row[k] + off_sgn, 0u),
                          ie_rq_a32(row[k] + off_q, 0u), ie_rq_a32(row[k] + off_q, 4u),
                          sc_sh, wl, wh, &dw);
        r10_v_r4lds_dot8(wl, wh, dw, lx, ldav, l, a[k]);
      }
    }
  }
  r10_v_tr8_epilogue4(r0, rows, l, a, bias, res, y, ys, rs);
}

/* ============================================================================
 * R10-V6: ie_saluki_iq2xxs_tr9_r10
 *   Fixed 1-row/lane tr9_pipe1 for T=8 IQ2_XXS (32 rows/WG):
 *     - Hoisted xt[8] SGPRs (removes 544x tr8_uniform on 5120x17408!)
 *     - No per-4-token asm barrier on aj (unblocks s_load overlap across t=0..7)
 *     - Branchless hot loop k0+32 < nb32 (eliminates j=3 s_wait_loadcnt 0!)
 *     - Immediate-offset bp_d / bp_q (1 ptr increment per 4 superblocks)
 *     - Unioned 8192 B LDS -> 8 WGs/WGP (vs 7 WGs/WGP in R4/R8!), ~60 VGPRs
 * ========================================================================== */
KERNEL ie_saluki_iq2xxs_tr9_r10(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                                const G float *bias, const G float *res, const G float *nw, G float *ny,
                                G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  const u32 r0 = 32u * wgid();
  if (T != 8u || r0 >= rows) return;
  static LDS union {
    u32 ulut[256];
    float red[8u * 8u * 32u];
  } sm __attribute__((aligned(16)));
  sm.ulut[tid()] = rq_pack_iq2(ie_rq_iq2xxs_grid[tid()]);
  lbarrier();

  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 r = r0 + l;
  const u32 mb = nb >> 3u;
  const G u8 *rowp = q + (unsigned long)(r < rows ? r : r0) * mb * 66u;
  const TR8_CSP u8 *xt[8];
#pragma unroll
  for (u32 t = 0; t < 8u; t++) xt[t] = (const TR8_CSP u8 *)tr8_uniform(xq + (unsigned long)t * xs);
  float a[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[j][t] = 0.0f;

  const G u8 *bp_d = rowp;
  const G u8 *bp_q = rowp + (2u + 8u * w);
  u32 xo = 32u * w, dao = 32u * nb + 4u * w;
  const u32 nb32 = nb & ~31u;

  if (nb32 > 0u) {
    u32 d_c = ie_rq_a16(bp_d, 0u);
    u32 a0_c = ie_rq_a32(bp_q, 0u);
    u32 a1_c = ie_rq_a32(bp_q, 4u);
    __asm__ volatile("" : "+v"(d_c), "+v"(a0_c), "+v"(a1_c));

    for (u32 k0 = 0; k0 + 32u < nb32; k0 += 32u) {
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
        const u32 noff = 66u * (j + 1u);
        const u32 d_n = ie_rq_a16(bp_d, noff);
        const u32 a0_n = ie_rq_a32(bp_q, noff);
        const u32 a1_n = ie_rq_a32(bp_q, noff + 4u);
        __builtin_amdgcn_sched_barrier(0);
        u32 wl[4], wh[4]; float dw;
        r10_v_decode_iq2xxs(sm.ulut, d_c, a0_c, a1_c, wl, wh, &dw);
        r10_v_tr9_dot8_sload(wl, wh, dw, xt, xo + 256u * j, dao + 32u * j, a[j]);
        __builtin_amdgcn_sched_barrier(0);
        d_c = d_n; a0_c = a0_n; a1_c = a1_n;
      }
      bp_d += 264u; bp_q += 264u;
      xo += 1024u; dao += 128u;
    }

#pragma unroll
    for (u32 j = 0; j < 3u; j++) {
      const u32 noff = 66u * (j + 1u);
      const u32 d_n = ie_rq_a16(bp_d, noff);
      const u32 a0_n = ie_rq_a32(bp_q, noff);
      const u32 a1_n = ie_rq_a32(bp_q, noff + 4u);
      __builtin_amdgcn_sched_barrier(0);
      u32 wl[4], wh[4]; float dw;
      r10_v_decode_iq2xxs(sm.ulut, d_c, a0_c, a1_c, wl, wh, &dw);
      r10_v_tr9_dot8_sload(wl, wh, dw, xt, xo + 256u * j, dao + 32u * j, a[j]);
      __builtin_amdgcn_sched_barrier(0);
      d_c = d_n; a0_c = a0_n; a1_c = a1_n;
    }
    {
      u32 wl[4], wh[4]; float dw;
      r10_v_decode_iq2xxs(sm.ulut, d_c, a0_c, a1_c, wl, wh, &dw);
      r10_v_tr9_dot8_sload(wl, wh, dw, xt, xo + 768u, dao + 96u, a[3]);
    }
    bp_d += 264u; bp_q += 264u;
    xo += 1024u; dao += 128u;
  }

  if (nb32 < nb) {
    for (u32 j = 0; j < 4u; j++) {
      if (nb32 + w + 8u * j < nb) {
        const u32 off = 66u * j;
        u32 wl[4], wh[4]; float dw;
        r10_v_decode_iq2xxs(sm.ulut, ie_rq_a16(bp_d, off),
                            ie_rq_a32(bp_q, off), ie_rq_a32(bp_q, off + 4u),
                            wl, wh, &dw);
        r10_v_tr9_dot8_sload(wl, wh, dw, xt, xo + 256u * j, dao + 32u * j, a[j]);
      }
    }
  }
  lbarrier();
  r10_v_tr9_reduce32(sm.red, w, l, r0, rows, a, bias, res, y, ys, rs);
}

/* ============================================================================
 * R10-V7: ie_saluki_iq1s_tr9_r10
 *   T=8 IQ1_S, 1 row/lane (32 rows/WG), packed 8 KiB u32 LUT unioned with red
 *   (8192 B LDS -> 8 WGs/WGP = 2x current r4 occupancy; ds_load_b32 halves
 *   LUT bank conflicts!), bitwise-exact folded 8*g +- 1 (zero `as` loads/adds),
 *   scalar s_load X with hoisted xt[8], branchless 1-SB pipe1, ~62 VGPRs.
 * ========================================================================== */
KERNEL ie_saluki_iq1s_tr9_r10(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                              const G float *bias, const G float *res, const G float *nw, G float *ny,
                              G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  const u32 r0 = 32u * wgid();
  if (T != 8u || r0 >= rows) return;
  static LDS union {
    u32 ulut[2048];
    float red[8u * 8u * 32u];
  } sm __attribute__((aligned(16)));
  for (u32 i = tid(); i < 2048u; i += NT) sm.ulut[i] = rq_pack_iq1(ie_rq_iq1s_grid[i]);
  lbarrier();

  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 r = r0 + l;
  const u32 mb = nb >> 3u;
  const G u8 *rowp = q + (unsigned long)(r < rows ? r : r0) * mb * 50u;
  const TR8_CSP u8 *xt[8];
#pragma unroll
  for (u32 t = 0; t < 8u; t++) xt[t] = (const TR8_CSP u8 *)tr8_uniform(xq + (unsigned long)t * xs);
  float a[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[j][t] = 0.0f;

  const G u8 *bp_d = rowp;
  const G u8 *bp_qlo = rowp + (2u + 4u * w);
  const G u8 *bp_qh = rowp + (34u + 2u * w);
  u32 xo = 32u * w, dao = 32u * nb + 4u * w;
  const u32 nb32 = nb & ~31u;

  if (nb32 > 0u) {
    u32 d_c = ie_rq_a16(bp_d, 0u);
    u32 qlo_c = ie_rq_a32(bp_qlo, 0u);
    u32 qh_c = ie_rq_a16(bp_qh, 0u);
    __asm__ volatile("" : "+v"(d_c), "+v"(qlo_c), "+v"(qh_c));

    for (u32 k0 = 0; k0 + 32u < nb32; k0 += 32u) {
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
        const u32 noff = 50u * (j + 1u);
        const u32 d_n = ie_rq_a16(bp_d, noff);
        const u32 qlo_n = ie_rq_a32(bp_qlo, noff);
        const u32 qh_n = ie_rq_a16(bp_qh, noff);
        __builtin_amdgcn_sched_barrier(0);
        u32 wl[4], wh[4]; float dw8;
        r10_v_decode_iq1s(sm.ulut, d_c, qlo_c, qh_c, wl, wh, &dw8);
        r10_v_tr9_dot8_sload(wl, wh, dw8, xt, xo + 256u * j, dao + 32u * j, a[j]);
        __builtin_amdgcn_sched_barrier(0);
        d_c = d_n; qlo_c = qlo_n; qh_c = qh_n;
      }
      bp_d += 200u; bp_qlo += 200u; bp_qh += 200u;
      xo += 1024u; dao += 128u;
    }

#pragma unroll
    for (u32 j = 0; j < 3u; j++) {
      const u32 noff = 50u * (j + 1u);
      const u32 d_n = ie_rq_a16(bp_d, noff);
      const u32 qlo_n = ie_rq_a32(bp_qlo, noff);
      const u32 qh_n = ie_rq_a16(bp_qh, noff);
      __builtin_amdgcn_sched_barrier(0);
      u32 wl[4], wh[4]; float dw8;
      r10_v_decode_iq1s(sm.ulut, d_c, qlo_c, qh_c, wl, wh, &dw8);
      r10_v_tr9_dot8_sload(wl, wh, dw8, xt, xo + 256u * j, dao + 32u * j, a[j]);
      __builtin_amdgcn_sched_barrier(0);
      d_c = d_n; qlo_c = qlo_n; qh_c = qh_n;
    }
    {
      u32 wl[4], wh[4]; float dw8;
      r10_v_decode_iq1s(sm.ulut, d_c, qlo_c, qh_c, wl, wh, &dw8);
      r10_v_tr9_dot8_sload(wl, wh, dw8, xt, xo + 768u, dao + 96u, a[3]);
    }
    bp_d += 200u; bp_qlo += 200u; bp_qh += 200u;
    xo += 1024u; dao += 128u;
  }

  if (nb32 < nb) {
    for (u32 j = 0; j < 4u; j++) {
      if (nb32 + w + 8u * j < nb) {
        const u32 off = 50u * j;
        u32 wl[4], wh[4]; float dw8;
        r10_v_decode_iq1s(sm.ulut, ie_rq_a16(bp_d, off),
                          ie_rq_a32(bp_qlo, off), ie_rq_a16(bp_qh, off),
                          wl, wh, &dw8);
        r10_v_tr9_dot8_sload(wl, wh, dw8, xt, xo + 256u * j, dao + 32u * j, a[j]);
      }
    }
  }
  lbarrier();
  r10_v_tr9_reduce32(sm.red, w, l, r0, rows, a, bias, res, y, ys, rs);
}

/* ============================================================================
 * R10-V8: ie_saluki_iq1m_tr9_r10
 *   T=8 IQ1_M, 1 row/lane (32 rows/WG), packed 8 KiB u32 LUT unioned with red
 *   (8192 B LDS -> 8 WGs/WGP = 2x current r4 occupancy; ds_load_b32 halves
 *   LUT bank conflicts!), scalar s_load X with hoisted xt[8], branchless 1-SB
 *   pipe1, ~66 VGPRs.
 * ========================================================================== */
KERNEL ie_saluki_iq1m_tr9_r10(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                              const G float *bias, const G float *res, const G float *nw, G float *ny,
                              G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  const u32 r0 = 32u * wgid();
  if (T != 8u || r0 >= rows) return;
  static LDS union {
    u32 ulut[2048];
    float red[8u * 8u * 32u];
  } sm __attribute__((aligned(16)));
  for (u32 i = tid(); i < 2048u; i += NT) sm.ulut[i] = rq_pack_iq1(ie_rq_iq1s_grid[i]);
  lbarrier();

  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 r = r0 + l;
  const u32 mb = nb >> 3u;
  const G u8 *rowp = q + (unsigned long)(r < rows ? r : r0) * mb * 56u;
  const TR8_CSP u8 *xt[8];
#pragma unroll
  for (u32 t = 0; t < 8u; t++) xt[t] = (const TR8_CSP u8 *)tr8_uniform(xq + (unsigned long)t * xs);
  float a[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[j][t] = 0.0f;

  const int use_shi = (w & 4u) != 0u;
  const u32 sc_sh16 = 16u * ((w >> 1u) & 1u);
  const u32 sh = 6u * (w & 1u);
  const G u8 *bp_sc = rowp + 48u;
  const G u8 *bp_qlo = rowp + 4u * w;
  const G u8 *bp_qhb = rowp + (32u + 2u * w);
  u32 xo = 32u * w, dao = 32u * nb + 4u * w;
  const u32 nb32 = nb & ~31u;

  if (nb32 > 0u) {
    u32 slo_c = ie_rq_a32(bp_sc, 0u);
    u32 shi_c = ie_rq_a32(bp_sc, 4u);
    u32 qlo_c = ie_rq_a32(bp_qlo, 0u);
    u32 qhb_c = ie_rq_a16(bp_qhb, 0u);
    __asm__ volatile("" : "+v"(slo_c), "+v"(shi_c), "+v"(qlo_c), "+v"(qhb_c));

    for (u32 k0 = 0; k0 + 32u < nb32; k0 += 32u) {
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
        const u32 noff = 56u * (j + 1u);
        const u32 slo_n = ie_rq_a32(bp_sc, noff);
        const u32 shi_n = ie_rq_a32(bp_sc, noff + 4u);
        const u32 qlo_n = ie_rq_a32(bp_qlo, noff);
        const u32 qhb_n = ie_rq_a16(bp_qhb, noff);
        __builtin_amdgcn_sched_barrier(0);
        u32 wl[4], wh[4]; float dq0, dq1;
        r10_v_decode_iq1m(sm.ulut, slo_c, shi_c, qhb_c, qlo_c, use_shi, sc_sh16, sh, wl, wh, &dq0, &dq1);
        r10_v_tr9_dot8_sload_split16(wl, wh, dq0, dq1, xt, xo + 256u * j, dao + 32u * j, a[j]);
        __builtin_amdgcn_sched_barrier(0);
        slo_c = slo_n; shi_c = shi_n; qlo_c = qlo_n; qhb_c = qhb_n;
      }
      bp_sc += 224u; bp_qlo += 224u; bp_qhb += 224u;
      xo += 1024u; dao += 128u;
    }

#pragma unroll
    for (u32 j = 0; j < 3u; j++) {
      const u32 noff = 56u * (j + 1u);
      const u32 slo_n = ie_rq_a32(bp_sc, noff);
      const u32 shi_n = ie_rq_a32(bp_sc, noff + 4u);
      const u32 qlo_n = ie_rq_a32(bp_qlo, noff);
      const u32 qhb_n = ie_rq_a16(bp_qhb, noff);
      __builtin_amdgcn_sched_barrier(0);
      u32 wl[4], wh[4]; float dq0, dq1;
      r10_v_decode_iq1m(sm.ulut, slo_c, shi_c, qhb_c, qlo_c, use_shi, sc_sh16, sh, wl, wh, &dq0, &dq1);
      r10_v_tr9_dot8_sload_split16(wl, wh, dq0, dq1, xt, xo + 256u * j, dao + 32u * j, a[j]);
      __builtin_amdgcn_sched_barrier(0);
      slo_c = slo_n; shi_c = shi_n; qlo_c = qlo_n; qhb_c = qhb_n;
    }
    {
      u32 wl[4], wh[4]; float dq0, dq1;
      r10_v_decode_iq1m(sm.ulut, slo_c, shi_c, qhb_c, qlo_c, use_shi, sc_sh16, sh, wl, wh, &dq0, &dq1);
      r10_v_tr9_dot8_sload_split16(wl, wh, dq0, dq1, xt, xo + 768u, dao + 96u, a[3]);
    }
    bp_sc += 224u; bp_qlo += 224u; bp_qhb += 224u;
    xo += 1024u; dao += 128u;
  }

  if (nb32 < nb) {
    for (u32 j = 0; j < 4u; j++) {
      if (nb32 + w + 8u * j < nb) {
        const u32 off = 56u * j;
        u32 wl[4], wh[4]; float dq0, dq1;
        r10_v_decode_iq1m(sm.ulut, ie_rq_a32(bp_sc, off), ie_rq_a32(bp_sc, off + 4u),
                          ie_rq_a16(bp_qhb, off), ie_rq_a32(bp_qlo, off),
                          use_shi, sc_sh16, sh, wl, wh, &dq0, &dq1);
        r10_v_tr9_dot8_sload_split16(wl, wh, dq0, dq1, xt, xo + 256u * j, dao + 32u * j, a[j]);
      }
    }
  }
  lbarrier();
  r10_v_tr9_reduce32(sm.red, w, l, r0, rows, a, bias, res, y, ys, rs);
}
