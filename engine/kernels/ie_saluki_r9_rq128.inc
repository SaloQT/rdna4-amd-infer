/* ============================================================================
 * Round 9: Bitwise-exact pipelined 128x128 prefill raw-quant WMMA GEMMs
 *   Rolls out Round 8's fixed software pipeline (prologue/latch cur-load
 *   drain -> 0 s_wait_loadcnt across all 64 WMMAs), scalar float acc[8][8] +
 *   R9_RQ_PIN_ACC8 (100% v_dual_fmac_f32), hoisted exact power-of-2 d scales,
 *   hoisted qh/sc alignments, single-mul sign decoders, and chunk_j = 2u
 *   (with recomputed activation row offsets to stay comfortably <= 192 VGPRs,
 *   0 spills, 2 WGs/CU) to the remaining 5 prefill RQ128 subtypes:
 *     - ie_saluki_gemm_rq128_iq2s_pipe   (IQ2_S,   split16, lut[1024], 22528 B LDS)
 *     - ie_saluki_gemm_rq128_iq3s_pipe   (IQ3_S,            lut[512],  20480 B LDS)
 *     - ie_saluki_gemm_rq128_iq2xs_pipe  (IQ2_XS,  split16, lut[512],  20480 B LDS)
 *     - ie_saluki_gemm_rq128_iq1m_pipe   (IQ1_M,   split16, lut[2048], 26624 B LDS)
 *     - ie_saluki_gemm_rq128_iq3xxs_pipe (IQ3_XXS,          lut[256],  19456 B LDS)
 *   (IQ1_S skipped: tested and discarded in Round 8.)
 *
 * Included at the end of kernels/ie_saluki_kernels.c (after Round 8).
 * All helpers/macros are prefixed r9_rq_ / R9_RQ_ to avoid collisions.
 * Bitwise equal (0 bit diffs) to ie_gemm_rq_<type> and ie_saluki_gemm_rq128_<type>.
 * ========================================================================== */

/* Tie 8 independent VGPR_32 accumulators to the SelectionDAG chain without
 * forming a VReg_256 super-register: preserves 0 spills and enables 100%
 * v_dual_fmac_f32 / v_fmac_f32 shrinking. */
#define R9_RQ_PIN_ACC8(a) __asm__ volatile("" : "+v"((a)[0]), "+v"((a)[1]), "+v"((a)[2]), "+v"((a)[3]), \
                                                "+v"((a)[4]), "+v"((a)[5]), "+v"((a)[6]), "+v"((a)[7]))

/* Drain ONLY the actually-loaded live global-load words of cur (never dead zeroed
 * rq_hblk_t slots) before the loop header and at the loop latch so SIInsertWaitcnts
 * never stalls on nxt/rx prefetches before the 64 WMMAs. */
static inline __attribute__((always_inline)) void r9_rq_touch_hblk(const u32 type, rq_hblk_t *x) {
  if (type == IE_RQ_IQ2_XS) {
    /* rq_hload(IQ2_XS): r[0](b32), r[1..2](b64), r[3..6](b128), r[7..10](b128), r[11](b32) */
    __asm__ volatile("" : "+v"(x->r[0]), "+v"(x->r[1]), "+v"(x->r[2]), "+v"(x->r[3]), "+v"(x->r[7]), "+v"(x->r[11]));
  } else if (type == IE_RQ_IQ2_S) {
    /* rq_hload(IQ2_S): r[0](b32), r[1..4](b128), r[5..8](b128), r[9](b32), r[10..13](b128), r[14](b32) */
    __asm__ volatile("" : "+v"(x->r[0]), "+v"(x->r[1]), "+v"(x->r[5]), "+v"(x->r[9]), "+v"(x->r[10]), "+v"(x->r[14]));
  } else if (type == IE_RQ_IQ3_XXS) {
    /* rq_hload(IQ3_XXS): r[0](b32), r[1..4](b128), r[5](b32), r[6..9](b128), r[10..13](b128), r[14](b32) */
    __asm__ volatile("" : "+v"(x->r[0]), "+v"(x->r[1]), "+v"(x->r[5]), "+v"(x->r[6]), "+v"(x->r[10]), "+v"(x->r[14]));
  } else if (type == IE_RQ_IQ3_S) {
    /* rq_hload(IQ3_S): r[0](b32), r[1..2](b64), r[3](b32), r[4..7](b128), r[8](b32),
     *                  r[9..12](b128), r[13..16](b128), r[17](b32) */
    __asm__ volatile("" : "+v"(x->r[0]), "+v"(x->r[1]), "+v"(x->r[2]), "+v"(x->r[3]),
                          "+v"(x->r[4]), "+v"(x->r[8]), "+v"(x->r[9]), "+v"(x->r[13]), "+v"(x->r[17]));
  } else if (type == IE_RQ_IQ1_M) {
    /* rq_hload(IQ1_M): r[0..1](b64), r[2..3](b64), r[4..7](b128) */
    __asm__ volatile("" : "+v"(x->r[0]), "+v"(x->r[1]), "+v"(x->r[2]), "+v"(x->r[3]), "+v"(x->r[4]));
  }
}

/* Single-mul IQ2_S 8-bit sign+code decode (bitwise identical to 2x RQ_SGN_B).
 * Caller guarantees sb in [0, 255]. s7 = sb & 127 avoids carry into bit 9;
 * bit 7 of sb lands at bit 26 via sb << 19 (zero at bits 2, 10, 18). */
static inline __attribute__((always_inline)) v2i r9_rq_iq2s_ksw(const u32 cp, const u32 sb) {
  const u32 s7 = sb & 127u;
  const u32 p_ = s7 * 0x00810204u;
  const u32 lo = RQ_P2((p_ & 0x04040404u) | (cp & 0x03030303u));
  const u32 hi = RQ_P2((((sb << 19u) | (p_ >> 4u)) & 0x04040404u) | ((cp >> 4u) & 0x03030303u));
  const v2i v = {(int)lo, (int)hi};
  return v;
}

/* Single-mul IQ3_XXS 7-bit+parity sign decode (bitwise identical to rq_ksign7 +
 * 2x ie_rq_ssign(g, ie_rq_mask4(...))). s7 in [0, 127]: s7 * 0x00204081 has zero
 * bit overlap across all 4 shifted copies, and m & 0x01010101u == t. */
static inline __attribute__((always_inline)) v2i r9_rq_iq3xxs_ksw(const u32 g0, const u32 g1, const u32 s7) {
  const u32 p_ = s7 * 0x00204081u;
  const u32 t0 = p_ & 0x01010101u;
  const u32 t1 = ((p_ >> 4u) | ((u32)__builtin_popcount(s7) << 24u)) & 0x01010101u;
  const u32 m0 = __builtin_amdgcn_perm(0u, 0u, t0 | 0x0C0C0C0Cu);
  const u32 m1 = __builtin_amdgcn_perm(0u, 0u, t1 | 0x0C0C0C0Cu);
  const v2i v = {(int)((g0 ^ m0) + t0), (int)((g1 ^ m1) + t1)};
  return v;
}

/* Single-mul IQ3_S 8-bit sign decode (bitwise identical to 2x ie_rq_ssign(g, ie_rq_mask4(...))).
 * Every byte in ie_rq_iq3s_grid[512] is odd ({1,3,5,7,9,11,13,15}), so -g == g ^ 0xFE
 * across all 4 bytes with zero carry; v_perm_b32 maps {0,1} -> {0x00,0xFE}.
 * Caller guarantees sb in [0, 255]; sb << 17 lands bit 7 at bit 24 (zero at bits 0, 8, 16). */
static inline __attribute__((always_inline)) v2i r9_rq_iq3s_ksw(const u32 g0, const u32 g1, const u32 sb) {
  const u32 s7 = sb & 127u;
  const u32 p_ = s7 * 0x00204081u;
  const u32 t0 = p_ & 0x01010101u;
  const u32 t1 = ((p_ >> 4u) | (sb << 17u)) & 0x01010101u;
  const u32 m0 = __builtin_amdgcn_perm(0u, 0x0000FE00u, t0);
  const u32 m1 = __builtin_amdgcn_perm(0u, 0x0000FE00u, t1);
  const v2i v = {(int)(g0 ^ m0), (int)(g1 ^ m1)};
  return v;
}

/* Non-split16 WMMA + scalar dual-fmac accumulate (chunk_j = 2u, ntok = 128u, wj = 8u). */
static inline __attribute__((always_inline)) void r9_rq_wmma_acc_j2(
    const v2i w0, const v2i w1, const float s0, const u32 k,
    const LDSP u8 *lx, const LDSP float *lsx, const u32 l, const u32 h, float acc[][8]) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  const u32 r16 = l & 15u;
#pragma unroll
  for (u32 j0 = 0; j0 < 8u; j0 += 2u) {
    v2i a0v[2], a1v[2];
    v8i c0[2];
#pragma unroll
    for (u32 j = 0; j < 2u; j++) {
      const u32 tk = 16u * (j0 + j) + r16;
      a0v[j] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k) + 8u * h);
      a1v[j] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k + 1u) + 8u * h);
    }
#pragma unroll
    for (u32 j = 0; j < 2u; j++) {
      c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0v[j], 1, w0, z, 0);
      c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1v[j], 1, w1, c0[j], 0);
    }
#pragma unroll
    for (u32 j = 0; j < 2u; j++) {
      const u32 ts = 16u * (j0 + j) + 8u * h;
      const f8 dx = *(const LDSP f8 *)(lsx + k * 128u + ts);
      const f8 t = __builtin_convertvector(c0[j], f8) * s0;
#pragma unroll
      for (u32 v = 0; v < 8u; v++) acc[j0 + j][v] += t[v] * dx[v];
      R9_RQ_PIN_ACC8(acc[j0 + j]);
    }
  }
}

/* Split-16 WMMA + scalar dual-fmac accumulate (chunk_j = 2u, ntok = 128u, wj = 8u). */
static inline __attribute__((always_inline)) void r9_rq_wmma_split16_j2(
    const v2i w0, const v2i w1, const float s0, const float s1, const u32 k,
    const LDSP u8 *lx, const LDSP float *lsx, const u32 l, const u32 h, float acc[][8]) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  const u32 r16 = l & 15u;
#pragma unroll
  for (u32 j0 = 0; j0 < 8u; j0 += 2u) {
    v2i a0v[2], a1v[2];
    v8i c0[2], c1[2];
#pragma unroll
    for (u32 j = 0; j < 2u; j++) {
      const u32 tk = 16u * (j0 + j) + r16;
      a0v[j] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k) + 8u * h);
      a1v[j] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k + 1u) + 8u * h);
    }
#pragma unroll
    for (u32 j = 0; j < 2u; j++) {
      c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0v[j], 1, w0, z, 0);
      c1[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1v[j], 1, w1, z, 0);
    }
#pragma unroll
    for (u32 j = 0; j < 2u; j++) {
      const u32 ts = 16u * (j0 + j) + 8u * h;
      const f8 dx = *(const LDSP f8 *)(lsx + k * 128u + ts);
      f8 t = __builtin_convertvector(c0[j], f8) * s0;
      t += __builtin_convertvector(c1[j], f8) * s1;
#pragma unroll
      for (u32 v = 0; v < 8u; v++) acc[j0 + j][v] += t[v] * dx[v];
      R9_RQ_PIN_ACC8(acc[j0 + j]);
    }
  }
}

/* IQ2_XS half-block math: hoisted exact d4 = d * 0.25f, hoisted sc4 alignment, single-mul r8_iq2_ksw. */
static inline __attribute__((always_inline)) void r9_rq_iq2xs_math_half(
    const rq_hblk_t *cur, const LDSP u32 *lut, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 l, const u32 h, float acc[][8]) {
  const u32 sh = cur->sh, qsh = sh ^ 16u;
  const u32 psel = h ? 0x07060302u : 0x05040100u;
  float d4 = ie_rq_h2f((cur->r[0] >> sh) & 0xFFFFu) * 0.25f;
  __asm__ volatile("" : "+v"(d4));
  const u32 sc4 = rq_align(cur->r[1], cur->r[2], qsh);
#pragma unroll
  for (u32 k = 0; k < RQ_KB; k++) {
    const u32 sc = ie_byte(sc4, k);
    const float s0 = d4 * (0.5f + (float)(sc & 15u));
    const float s1 = d4 * (0.5f + (float)(sc >> 4u));
    const u32 w0r = rq_align(cur->r[3u + 2u * k], cur->r[4u + 2u * k], qsh);
    const u32 w1r = rq_align(cur->r[4u + 2u * k], cur->r[5u + 2u * k], qsh);
    const u32 qp = __builtin_amdgcn_perm(w1r, w0r, psel);
    const u32 q2_0 = qp & 0xFFFFu, q2_1 = qp >> 16u;
    const u32 cp0 = lut[q2_0 & 511u];
    const u32 cp1 = lut[q2_1 & 511u];
    const v2i w0 = r8_iq2_ksw(cp0, q2_0 >> 9u);
    const v2i w1 = r8_iq2_ksw(cp1, q2_1 >> 9u);
    r9_rq_wmma_split16_j2(w0, w1, s0, s1, k, lx, lsx, l, h, acc);
  }
}

/* IQ2_S half-block math: hoisted exact d4 = d * 0.25f, hoisted qh4/sc4 alignments, single-mul r9_rq_iq2s_ksw. */
static inline __attribute__((always_inline)) void r9_rq_iq2s_math_half(
    const rq_hblk_t *cur, const LDSP u32 *lut, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 l, const u32 h, float acc[][8]) {
  const u32 sh = cur->sh, qsh = sh ^ 16u;
  float d4 = ie_rq_h2f((cur->r[0] >> sh) & 0xFFFFu) * 0.25f;
  __asm__ volatile("" : "+v"(d4));
  const u32 qh4 = rq_align(cur->r[1], cur->r[2], qsh);
  const u32 sc4 = rq_align(cur->r[3], cur->r[4], qsh);
  const u32 g0 = h, g1 = 2u + h;
  const u32 sh0 = 8u - 2u * g0, sh1 = 8u - 2u * g1;
#pragma unroll
  for (u32 k = 0; k < RQ_KB; k++) {
    const u32 qh = ie_byte(qh4, k);
    const u32 sc = ie_byte(sc4, k);
    const float s0 = d4 * (0.5f + (float)(sc & 15u));
    const float s1 = d4 * (0.5f + (float)(sc >> 4u));
    const u32 qlo = rq_align(cur->r[5u + k], cur->r[6u + k], qsh);
    const u32 sgn = rq_align(cur->r[10u + k], cur->r[11u + k], qsh);
    const u32 cp0 = lut[ie_byte(qlo, g0) | ((qh << sh0) & 0x300u)];
    const u32 cp1 = lut[ie_byte(qlo, g1) | ((qh << sh1) & 0x300u)];
    const v2i w0 = r9_rq_iq2s_ksw(cp0, ie_byte(sgn, g0));
    const v2i w1 = r9_rq_iq2s_ksw(cp1, ie_byte(sgn, g1));
    r9_rq_wmma_split16_j2(w0, w1, s0, s1, k, lx, lsx, l, h, acc);
  }
}

/* IQ1_M half-block math: hoisted exact d8 = d * 0.125f, batched LDS LUT gathers. */
static inline __attribute__((always_inline)) void r9_rq_iq1m_math_half(
    const rq_hblk_t *cur, const LDSP u32 *lut, const u32 hb, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 l, const u32 h, float acc[][8]) {
  const u32 s0w = cur->r[0] & 0xFFFFu, s1w = cur->r[0] >> 16u;
  const u32 s2w = cur->r[1] & 0xFFFFu, s3w = cur->r[1] >> 16u;
  float d8 = ie_rq_h2f((s0w >> 12u) | ((s1w >> 8u) & 0x00F0u) | ((s2w >> 4u) & 0x0F00u) | (s3w & 0xF000u)) * 0.125f;
  __asm__ volatile("" : "+v"(d8));
  const u32 pair = hb != 0u ? cur->r[1] : cur->r[0];
  const u32 dmask = h != 0u ? 0x80u : 0x08u;
  const u32 qhsh = 8u - 4u * h;
  const u32 g0 = h, g1 = 2u + h;
#pragma unroll
  for (u32 k = 0; k < RQ_KB; k++) {
    const u32 scv = pair >> (16u * (k >> 1u)), ksh = 6u * (k & 1u);
    const float s0 = d8 * (float)(int)(2u * ((scv >> ksh) & 7u) + 1u);
    const float s1 = d8 * (float)(int)(2u * ((scv >> (ksh + 3u)) & 7u) + 1u);
    const u32 qhb = (cur->r[2u + (k >> 1u)] >> (16u * (k & 1u))) & 0xFFFFu;
    const u32 qlo = cur->r[4u + k];
    const u32 qh0 = qhb & 255u, qh1 = qhb >> 8u;
    const u32 cp0 = lut[ie_byte(qlo, g0) | ((qh0 << qhsh) & 0x700u)];
    const u32 cp1 = lut[ie_byte(qlo, g1) | ((qh1 << qhsh) & 0x700u)];
    const u32 tbl0 = (qh0 & dmask) != 0u ? RQ_IQ1_TBL_NEG : RQ_IQ1_TBL_POS;
    const u32 tbl1 = (qh1 & dmask) != 0u ? RQ_IQ1_TBL_NEG : RQ_IQ1_TBL_POS;
    const v2i w0 = {(int)rq_iq1_map(cp0, tbl0), (int)rq_iq1_map(cp0 >> 4u, tbl0)};
    const v2i w1 = {(int)rq_iq1_map(cp1, tbl1), (int)rq_iq1_map(cp1 >> 4u, tbl1)};
    r9_rq_wmma_split16_j2(w0, w1, s0, s1, k, lx, lsx, l, h, acc);
  }
}

/* IQ3_XXS half-block math: hoisted exact d2 = d * 0.5f, batched 4x LUT loads, single-mul r9_rq_iq3xxs_ksw. */
static inline __attribute__((always_inline)) void r9_rq_iq3xxs_math_half(
    const rq_hblk_t *cur, const LDSP u32 *lut, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 l, const u32 h, float acc[][8]) {
  const u32 sh = cur->sh, qsh = sh ^ 16u;
  const u32 psel = h ? 0x07060302u : 0x05040100u;
  float d2 = ie_rq_h2f((cur->r[0] >> sh) & 0xFFFFu) * 0.5f;
  __asm__ volatile("" : "+v"(d2));
  const u32 sh0 = 7u * h, sh1 = 7u * (2u + h);
#pragma unroll
  for (u32 k = 0; k < RQ_KB; k++) {
    const u32 aux = rq_align(cur->r[1u + k], cur->r[2u + k], qsh);
    const u32 w0r = rq_align(cur->r[6u + 2u * k], cur->r[7u + 2u * k], qsh);
    const u32 w1r = rq_align(cur->r[7u + 2u * k], cur->r[8u + 2u * k], qsh);
    const u32 qp = __builtin_amdgcn_perm(w1r, w0r, psel);
    const float s0 = d2 * (0.5f + (float)(aux >> 28u));
    const u32 w16_0 = qp & 0xFFFFu, w16_1 = qp >> 16u;
    const u32 g00 = lut[w16_0 & 255u], g01 = lut[w16_0 >> 8u];
    const u32 g10 = lut[w16_1 & 255u], g11 = lut[w16_1 >> 8u];
    const v2i w0 = r9_rq_iq3xxs_ksw(g00, g01, (aux >> sh0) & 127u);
    const v2i w1 = r9_rq_iq3xxs_ksw(g10, g11, (aux >> sh1) & 127u);
    r9_rq_wmma_acc_j2(w0, w1, s0, k, lx, lsx, l, h, acc);
  }
}

/* IQ3_S half-block math: hoisted d/qh4/sc16, batched 4x LUT loads, single-mul odd-grid ^0xFE r9_rq_iq3s_ksw. */
static inline __attribute__((always_inline)) void r9_rq_iq3s_math_half(
    const rq_hblk_t *cur, const LDSP u32 *lut, const u32 hb, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 l, const u32 h, float acc[][8]) {
  const u32 sh = cur->sh, qsh = sh ^ 16u;
  const u32 psel = h ? 0x07060302u : 0x05040100u;
  float d = ie_rq_h2f((cur->r[0] >> sh) & 0xFFFFu);
  __asm__ volatile("" : "+v"(d));
  const u32 qh4 = rq_align(cur->r[1], cur->r[2], qsh);
  const u32 sc16 = cur->r[3] >> (sh ^ (hb ? 0u : 16u));
  const u32 g0 = h, g1 = 2u + h;
#pragma unroll
  for (u32 k = 0; k < RQ_KB; k++) {
    const u32 qh = ie_byte(qh4, k);
    const u32 sc = (sc16 >> (4u * k)) & 15u;
    const float s0 = d * (float)(int)(1u + 2u * sc);
    const u32 sgn = rq_align(cur->r[4u + k], cur->r[5u + k], qsh);
    const u32 w0r = rq_align(cur->r[9u + 2u * k], cur->r[10u + 2u * k], qsh);
    const u32 w1r = rq_align(cur->r[10u + 2u * k], cur->r[11u + 2u * k], qsh);
    const u32 qp = __builtin_amdgcn_perm(w1r, w0r, psel);
    const u32 w16_0 = qp & 0xFFFFu, w16_1 = qp >> 16u;
    const u32 i00 = (w16_0 & 255u) | (((qh >> (2u * g0)) & 1u) << 8u);
    const u32 i01 = (w16_0 >> 8u)  | (((qh >> (2u * g0 + 1u)) & 1u) << 8u);
    const u32 i10 = (w16_1 & 255u) | (((qh >> (2u * g1)) & 1u) << 8u);
    const u32 i11 = (w16_1 >> 8u)  | (((qh >> (2u * g1 + 1u)) & 1u) << 8u);
    const u32 l00 = lut[i00], l01 = lut[i01], l10 = lut[i10], l11 = lut[i11];
    const v2i w0 = r9_rq_iq3s_ksw(l00, l01, ie_byte(sgn, g0));
    const v2i w1 = r9_rq_iq3s_ksw(l10, l11, ie_byte(sgn, g1));
    r9_rq_wmma_acc_j2(w0, w1, s0, k, lx, lsx, l, h, acc);
  }
}

/* Pipelined 128x128 runner: drains prologue/latch cur loads via r9_rq_touch_hblk,
 * true if(more)-gated nxt + activation prefetch, scalar float acc[8][8] for 100%
 * v_dual_fmac_f32, and on-the-fly xq row addressing (saves 8 VGPRs vs cached xp[4]
 * to keep even 18-word IQ3_S and split16 IQ2_S safely <= 192 VGPRs with 0 spills). */
static inline __attribute__((always_inline)) void r9_rq128_pipe_run(
    const u32 type, const G u8 *q, const G u8 *xq, G u8 *y,
    const u32 rows, const u32 nb, const u32 T, const G float *bias, const G u8 *res,
    const u32 xs, const u32 ys, const u32 rs, LDSP u8 *lx, LDSP float *lsx, LDSP u32 *lut) {
  if (T == 0u || rows == 0u || nb < 8u || (nb & 7u) != 0u || !q || !xq || !y) return;
  const u32 ntt = (T + 127u) / 128u, tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 rbase = rt * 128u, tbase = tt * 128u;
  if (rbase >= rows || tbase >= T) return;
  const u32 t0 = tid(), w = t0 >> 5u, l = lane(), h = l >> 4u;
  const u32 r_loc = 16u * w + (l & 15u), row = rbase + r_loc, rr = row < rows ? row : rows - 1u;
  const u32 mb = nb >> 3u, bs = ie_rq_bs(type);
  const G u8 *rowp = q + (unsigned long)rr * mb * bs;
  float acc[8][8];
#pragma unroll
  for (u32 j = 0; j < 8u; j++)
#pragma unroll
    for (u32 v = 0; v < 8u; v++) acc[j][v] = 0.0f;
  rq_stage_lut_raw(type, (LDSP ie_rq_u64 *)0, lut, 0);

  const u32 tk0 = t0 >> 3u, c8_off = (t0 & 7u) * 16u;
  const u32 gs = tbase + (t0 & 127u) < T ? tbase + (t0 & 127u) : T - 1u;
  const G float *sxp = (const G float *)(xq + (unsigned long)gs * xs + 32ul * nb);
  u32x4 rx[4];
  f32x4 rsx = (f32x4)(0.0f);
#pragma unroll
  for (u32 i = 0; i < 4u; i++) {
    const u32 gt = tbase + tk0 + 32u * i < T ? tbase + tk0 + 32u * i : T - 1u;
    rx[i] = *(const G u32x4 *)(xq + (unsigned long)gt * xs + c8_off);
  }
  if (t0 < 128u) rsx = rq_load_f32x4(sxp);
  rq_hblk_t cur = rq_hload(type, rowp, 0u, h);
#pragma unroll
  for (u32 i = 0; i < 4u; i++) {
    const u32 p = t0 + 256u * i, tk = p >> 3u, c8 = p & 7u;
    *rqc_lx_slot(lx, tk, c8) = rx[i];
  }
  if (t0 < 128u) { lsx[t0] = rsx.x; lsx[128u + t0] = rsx.y; lsx[256u + t0] = rsx.z; lsx[384u + t0] = rsx.w; }
  lbarrier();
  r9_rq_touch_hblk(type, &cur);

  const u32 nc = 2u * mb;
#pragma unroll 1
  for (u32 c = 0; c < nc; c++) {
    const int more = c + 1u < nc;
    rq_hblk_t nxt;
    if (more) {
      const u32 cn = c + 1u;
      nxt = rq_hload(type, rowp + (unsigned long)(cn >> 1u) * bs, cn & 1u, h);
      const u32 kb = 4u * cn;
      const unsigned long kb_off = (unsigned long)kb * 32ul + c8_off;
#pragma unroll
      for (u32 i = 0; i < 4u; i++) {
        const u32 gt = tbase + tk0 + 32u * i < T ? tbase + tk0 + 32u * i : T - 1u;
        rx[i] = *(const G u32x4 *)(xq + (unsigned long)gt * xs + kb_off);
      }
      if (t0 < 128u) rsx = rq_load_f32x4(sxp + kb);
    }
    __builtin_amdgcn_sched_barrier(0);
    if (type == IE_RQ_IQ2_S) r9_rq_iq2s_math_half(&cur, lut, lx, lsx, l, h, acc);
    else if (type == IE_RQ_IQ3_S) r9_rq_iq3s_math_half(&cur, lut, c & 1u, lx, lsx, l, h, acc);
    else if (type == IE_RQ_IQ2_XS) r9_rq_iq2xs_math_half(&cur, lut, lx, lsx, l, h, acc);
    else if (type == IE_RQ_IQ1_M) r9_rq_iq1m_math_half(&cur, lut, c & 1u, lx, lsx, l, h, acc);
    else if (type == IE_RQ_IQ3_XXS) r9_rq_iq3xxs_math_half(&cur, lut, lx, lsx, l, h, acc);
    __builtin_amdgcn_sched_barrier(0);
    if (more) {
      lbarrier();
#pragma unroll
      for (u32 i = 0; i < 4u; i++) {
        const u32 p = t0 + 256u * i, tk = p >> 3u, c8 = p & 7u;
        *rqc_lx_slot(lx, tk, c8) = rx[i];
      }
      if (t0 < 128u) { lsx[t0] = rsx.x; lsx[128u + t0] = rsx.y; lsx[256u + t0] = rsx.z; lsx[384u + t0] = rsx.w; }
      lbarrier();
      cur = nxt;
      r9_rq_touch_hblk(type, &cur);
    }
  }
  r8_rq_store_out(y, bias, res, rows, T, ys, rs, row, tbase, 0u, 8u, h, acc);
}

#define R9_DEF_RQ128_PIPE(tag_, type_, nlut_) \
  KERNEL ie_saluki_gemm_rq128_##tag_##_pipe(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T, \
                                            const G float *bias, const G u8 *res, u32 xs, u32 ys, u32 rs) { \
    static LDS u8 lx[128u * 128u] __attribute__((aligned(16))); \
    static LDS float lsx[4u * 128u] __attribute__((aligned(32))); \
    static LDS u32 lut[nlut_] __attribute__((aligned(16))); \
    r9_rq128_pipe_run(type_, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, lut); \
  }

R9_DEF_RQ128_PIPE(iq2s,   IE_RQ_IQ2_S,   1024u)
R9_DEF_RQ128_PIPE(iq3s,   IE_RQ_IQ3_S,   512u)
R9_DEF_RQ128_PIPE(iq2xs,  IE_RQ_IQ2_XS,  512u)
R9_DEF_RQ128_PIPE(iq1m,   IE_RQ_IQ1_M,   2048u)
R9_DEF_RQ128_PIPE(iq3xxs, IE_RQ_IQ3_XXS, 256u)

#undef R9_DEF_RQ128_PIPE
#undef R9_RQ_PIN_ACC8
