/* ============================================================================
 * Round 11 Group 2: Bitwise-exact 128x128 double-buffered (db64), 20 KiB LDS,
 *                   zero-stall split16 pipelined IQ2_S prefill WMMA GEMM
 *                   for RX 9070 XT (gfx1201, wave32).
 *
 * Target: ie_saluki_gemm_rq128_iq2s_r11 (replaces ie_saluki_gemm_rq128_iq2s_pipe,
 *         1,813.1 ms trimmed, n=2,317 — #1 largest kernel in Group 2, 28.9% of
 *         Group 2's 6,273 ms in logs/saluki-opt-r10-final-ab/traced_cand/summary.json).
 *
 * Architectural improvements (targeting <= 192 VGPRs, 0 spills, 20,480 B LDS):
 *   1. Interleaved 16-bit packed IQ2_S grid LUT (u16 lut[1024] = 2,048 B vs
 *      4,096 B u32): halves LUT LDS footprint & bank traffic, dropping total
 *      LDS from 22,528 B to 20,480 B (matching IQ2_XS/Q2_K at 3 WGs/64 KiB).
 *      Unpacks via 2x v_lshl_or_b32 (zero multiplies), exhaustively verified
 *      0-bit-diff across all 1,024 grid entries x 256 sign bytes (262,144 cases).
 *   2. Double-buffered 64-token LDS staging (lx[2][64*128] + lsx[2][4*64]):
 *      eliminates R9's back-to-back empty lbarrier() pairs and lockstep 16 KiB
 *      ds_store bursts, halves rx (8 vs 16 VGPRs) and rsx wave work.
 *   3. Base-aligned whole-SB weight sharing with v_perm_b32 (12 -> 8 global
 *      weight loads/SB): base = rq_ap(bp, 0) folds fp16 d into base[0] (zero
 *      separate d_word load) and loads qh[0..7]+sc[0..7] once per superblock;
 *      hb1 tail is just 2x global_load_b128 (base+4, base+12).
 *   4. Zero-stall 1-tile split16 WMMA/post-op pipeline: exploits zero RAW
 *      dependency between split16's c0 and c1 WMMAs (both start from z) to
 *      halve live WMMA accumulators (16 vs 32 VGPRs), a0/a1 (4 vs 8 VGPRs),
 *      and dx (8 vs 16 VGPRs) while hiding dx (24+ VALU cyc) and next a0/a1
 *      (28+ VALU cyc) behind isolated 8x cvt + 4x dual_mul -> 8x cvt +
 *      4x dual_fmac -> 4x dual_fmac stages (0 s_wait_dscnt stalls).
 *   5. Mid-B-half (k=2) weight prefetch + on-the-fly per-k s0/s1 from 1-VGPR
 *      sc4 + d4: keeps xdc[4] at 16 VGPRs and issues hb1_tail (Phase 2) and
 *      nxt_hb0 (Phase 4) right after xdc[0..1] (8 VGPRs) die, keeping peak
 *      VGPRs safely <= 192 (8 waves/SIMD, 0 spills) with >550 wave cycles of
 *      weight-prefetch hiding and FIFO-ordered rx/rsx ahead of weight loads.
 *   6. Batched 4-k decode_all overlapping all 8 LUT ds_load_u16 + step-0
 *      a0/a1 loads behind 8x single-mul sign-mask VALU (0 LDS stalls).
 *   7. 32-bit voffset + 64-bit uniform saddr activation loads (eliminates R9's
 *      4x 64-bit gt*xs multiplies per half-block) + compact 32-deep batched
 *      residual epilogue (32x MLP on 5120x6144 o_proj, preserves -0.0f).
 *
 * Included at EOF of kernels/ie_saluki_kernels.c by r11-wiring.
 * All local types/helpers/macros are prefixed r11_g02_ / R11_G02_ and all
 * macros are #undef'd at EOF. 100% bitwise identical (0 bit diffs) to
 * ie_gemm_rq_iq2s, ie_saluki_gemm_rq128_iq2s, and ie_saluki_gemm_rq128_iq2s_pipe.
 * ========================================================================== */

_Static_assert(sizeof(u16) == 2u, "r11_g02 u16 LUT requires 2-byte u16");

struct __attribute__((packed, aligned(4))) r11_g02_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 r11_g02_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct r11_g02_u32x4p *)p)->v;
}

/* Tie 8 independent VGPR_32 accumulators to the SelectionDAG chain without
 * forming a VReg_256 super-register: preserves 0 spills and enables 100%
 * v_dual_fmac_f32 shrinking. */
#define R11_G02_PIN_ACC8(a)                                                                                        \
  __asm__ volatile(""                                                                                              \
                   : "+v"((a)[0]), "+v"((a)[1]), "+v"((a)[2]), "+v"((a)[3]), "+v"((a)[4]), "+v"((a)[5]),           \
                     "+v"((a)[6]), "+v"((a)[7]))

typedef struct {
  v2i w0, w1;
} r11_g02_dc_t;

/* Pack 8 IQ2 magnitudes {0x08, 0x19, 0x2b} (high nibbles {0, 1, 2}) into 16 bits:
 *   bits  0.. 1: m0,  bits  2.. 3: m4
 *   bits  4.. 5: m2,  bits  6.. 7: m6
 *   bits  8.. 9: m1,  bits 10..11: m5
 *   bits 12..13: m3,  bits 14..15: m7
 * Allows zero-multiply unpack via (cp | (cp << 12)) & 0x03030303 and
 * ((cp >> 2) | (cp << 10)) & 0x03030303. */
static inline __attribute__((always_inline)) u16 r11_g02_pack_iq2_u16(const ie_rq_u64 g) {
  const u32 lo = (u32)g, hi = (u32)(g >> 32u);
  const u32 m0 = (lo >> 4u)  & 3u, m1 = (lo >> 12u) & 3u, m2 = (lo >> 20u) & 3u, m3 = (lo >> 28u) & 3u;
  const u32 m4 = (hi >> 4u)  & 3u, m5 = (hi >> 12u) & 3u, m6 = (hi >> 20u) & 3u, m7 = (hi >> 28u) & 3u;
  return (u16)(m0 | (m4 << 2u) | (m2 << 4u) | (m6 << 6u) |
               (m1 << 8u) | (m5 << 10u) | (m3 << 12u) | (m7 << 14u));
}

/* Branchless 2-byte-shifted word alignment from base = rq_ap(bp, 0):
 *   bp % 4 == 0 -> psel = 0x05040302u -> (lo >> 16) | (hi << 16)
 *   bp % 4 == 2 -> psel = 0x07060504u -> hi */
static inline __attribute__((always_inline)) u32 r11_g02_perm_al(const u32 lo, const u32 hi, const u32 psel) {
  return __builtin_amdgcn_perm(hi, lo, psel);
}

/* Combine precomputed IQ2_S sign masks (bits 2,10,18,26) with interleaved u16
 * magnitude code cp16 into two signed int8x4 weight words via RQ_P2. */
static inline __attribute__((always_inline)) v2i r11_g02_iq2s_finish(const u32 cp16, const u32 sm_lo,
                                                                     const u32 sm_hi) {
  const u32 c_lo = (cp16 | (cp16 << 12u)) & 0x03030303u;
  const u32 c_hi = ((cp16 >> 2u) | (cp16 << 10u)) & 0x03030303u;
  const u32 lo = RQ_P2(sm_lo | c_lo);
  const u32 hi = RQ_P2(sm_hi | c_hi);
  const v2i v = {(int)lo, (int)hi};
  return v;
}

/* Stage 64 tokens x 128 int8 activations + 4*64 Q8_0 scales into one db64 LDS bank.
 * Uses single precomputed st_off0 (since (t0+256)>>3 == (t0>>3)+32 preserves (tk&7) and (t0&7),
 * so slot 1 is exactly st_off0 + 4096u -> 1 VGPR instead of 4 split address VGPRs). */
static inline __attribute__((always_inline)) void r11_g02_stage_x64(LDSP u8 *lx, LDSP float *lsx, const u32 st_off0,
                                                                    const u32 t0, const int is_s, const u32x4 rx[2],
                                                                    const f32x4 rsx) {
  *(LDSP u32x4 *)(lx + st_off0) = rx[0];
  *(LDSP u32x4 *)(lx + st_off0 + 4096u) = rx[1];
  if (is_s) {
    lsx[t0] = rsx.x;
    lsx[64u + t0] = rsx.y;
    lsx[128u + t0] = rsx.z;
    lsx[192u + t0] = rsx.w;
  }
}

/* Load one 16x16x16 A-matrix pair (half 0 and half 1 of 32-weight sub-block k) for tile j. */
static inline __attribute__((always_inline)) void r11_g02_load_a(const LDSP u8 *lx, const u32 k, const u32 j,
                                                                 const u32 r16, const u32 h, v2i *a0, v2i *a1) {
  const u32 tk = 16u * j + r16;
  *a0 = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k) + 8u * h);
  *a1 = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k + 1u) + 8u * h);
}

/* Load IQ2_S hb0 + shared d/qh[0..7]/sc[0..7] in 6 global loads (15 words) off base = rq_ap(p, 0):
 *   w0[0..4]  = base[ 0.. 4] (d + qlo_hb0)
 *   w8[0..4]  = base[ 8..12] (sgn_hb0; w8[0]=base[8] also closes qlo_hb1)
 *   w16[0..4] = base[16..20] (qh_hb0, qh_hb1, sc_hb0, sc_hb1; w16[0]=base[16] also closes sgn_hb1)
 * Max read extent: p + 83 (p%4==0) or p + 81 (p%4==2), strictly within rq_hload's p + 85 bound. */
static inline __attribute__((always_inline)) void r11_g02_iq2s_load_hb0(const G u8 *p, u32 w0[5], u32 w8[5],
                                                                        u32 w16[5]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v0 = r11_g02_ld4(base + 0u);
  w0[0] = v0.x; w0[1] = v0.y; w0[2] = v0.z; w0[3] = v0.w;
  w0[4] = base[4];
  const u32x4 v8 = r11_g02_ld4(base + 8u);
  w8[0] = v8.x; w8[1] = v8.y; w8[2] = v8.z; w8[3] = v8.w;
  w8[4] = base[12];
  const u32x4 v16 = r11_g02_ld4(base + 16u);
  w16[0] = v16.x; w16[1] = v16.y; w16[2] = v16.z; w16[3] = v16.w;
  w16[4] = base[20];
}

/* Load IQ2_S hb1 tail in 2 global_load_b128s (8 words) off base = rq_ap(p, 0):
 *   q1[0..3] = base[ 4.. 7] (paired with carried base[8]  -> qlo_hb1)
 *   s1[0..3] = base[12..15] (paired with carried base[16] -> sgn_hb1) */
static inline __attribute__((always_inline)) void r11_g02_iq2s_load_hb1_tail(const G u8 *p, u32 q1[4], u32 s1[4]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v4 = r11_g02_ld4(base + 4u);
  q1[0] = v4.x; q1[1] = v4.y; q1[2] = v4.z; q1[3] = v4.w;
  const u32x4 v12 = r11_g02_ld4(base + 12u);
  s1[0] = v12.x; s1[1] = v12.y; s1[2] = v12.z; s1[3] = v12.w;
}

/* Batched 4-k IQ2_S decode:
 *   1. Issues all 8 u16 LUT ds_load_u16 + step-0 (k=0, j=0) A-tile ds_load_b64.
 *   2. Computes all 8 sign masks (8x single-mul s7 * 0x00810204) while LUT loads return.
 *   3. Finishes 8x (v_lshl_or_b32 + RQ_P2) into compact 16-VGPR dc[4] with 0 LDS stalls.
 * Derives g0/g1/sh0/sh1 locally from pinned h_off = 32*h so LLVM does not hoist 4 extra
 * loop-invariant shift VGPRs across apply_k_range. */
static inline __attribute__((always_inline)) void r11_g02_iq2s_decode_all(
    const u32 qlo_w[5], const u32 sgn_w[5], const u32 qh4, const u32 psel,
    const LDSP u16 *lut, const LDSP u8 *lx, const u32 r16, const u32 h, const u32 h_off,
    r11_g02_dc_t dc[4], v2i *a0_0, v2i *a1_0) {
  u32 ho = h_off;
  __asm__ volatile("" : "+v"(ho));
  const u32 g0 = ho >> 5u, g1 = 2u + g0;
  const u32 sh0 = 8u - 2u * g0, sh1 = 8u - 2u * g1;
  u32 sgnv[4], cp0v[4], cp1v[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 qh = ie_byte(qh4, k);
    const u32 qlo = r11_g02_perm_al(qlo_w[k], qlo_w[k + 1u], psel);
    sgnv[k] = r11_g02_perm_al(sgn_w[k], sgn_w[k + 1u], psel);
    cp0v[k] = (u32)lut[ie_byte(qlo, g0) | ((qh << sh0) & 0x300u)];
    cp1v[k] = (u32)lut[ie_byte(qlo, g1) | ((qh << sh1) & 0x300u)];
  }
  r11_g02_load_a(lx, 0u, 0u, r16, h, a0_0, a1_0);
  __builtin_amdgcn_sched_barrier(0);
  u32 sm0_lo[4], sm0_hi[4], sm1_lo[4], sm1_hi[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 sb0 = ie_byte(sgnv[k], g0), sb1 = ie_byte(sgnv[k], g1);
    const u32 p0 = (sb0 & 127u) * 0x00810204u;
    const u32 p1 = (sb1 & 127u) * 0x00810204u;
    sm0_lo[k] = p0 & 0x04040404u;
    sm0_hi[k] = ((sb0 << 19u) | (p0 >> 4u)) & 0x04040404u;
    sm1_lo[k] = p1 & 0x04040404u;
    sm1_hi[k] = ((sb1 << 19u) | (p1 >> 4u)) & 0x04040404u;
  }
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    dc[k].w0 = r11_g02_iq2s_finish(cp0v[k], sm0_lo[k], sm0_hi[k]);
    dc[k].w1 = r11_g02_iq2s_finish(cp1v[k], sm1_lo[k], sm1_hi[k]);
  }
}

/* Zero-stall 1-tile split16 WMMA + post-op pipeline for k in [k_beg, k_end), j in 0..3.
 *   - Stage 1: 2 independent WMMAs (c0, c1 from z -> 0 RAW hazard, only 16 VGPRs live).
 *   - Stage 2: FIFO-ordered dx (consumed in Stage 4, 24+ VALU cyc later -> s_wait_dscnt 2, 0 stall)
 *              then next step's (a0_c, a1_c) reusing dead a0/a1 VGPRs (28+ cyc later -> 0 stall).
 *   - Stage 3A: 8x v_cvt_f32_i32 + 4x v_dual_mul_f32  (tv = (float)c0 * s0).
 *   - Stage 3B: 8x v_cvt_f32_i32 + 4x v_dual_fmac_f32 (tv += (float)c1 * s1, exact fma).
 *   - Stage 4:  4x v_dual_fmac_f32                    (acc[j] += tv * dx, exact fma). */
static inline __attribute__((always_inline)) void r11_g02_apply_k_range(
    const r11_g02_dc_t dc[4], const u32 sc4, const float d4,
    const LDSP u8 *lx, const LDSP float *lsx, const u32 r16, const u32 h,
    const u32 k_beg, const u32 k_end, v2i *a0_c, v2i *a1_c, float acc[4][8]) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (u32 k = k_beg; k < k_end; k++) {
    const u32 sc = ie_byte(sc4, k);
    const float s0 = d4 * (0.5f + (float)(sc & 15u));
    const float s1 = d4 * (0.5f + (float)(sc >> 4u));
#pragma unroll
    for (u32 j = 0; j < 4u; j++) {
      const u32 step = 4u * k + j;
      const v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, *a0_c, 1, dc[k].w0, z, 0);
      const v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, *a1_c, 1, dc[k].w1, z, 0);
      __builtin_amdgcn_sched_barrier(0);
      const f8 dx = *(const LDSP f8 *)(lsx + k * 64u + 16u * j + 8u * h);
      if (step + 1u < 16u) {
        const u32 kn = (step + 1u) >> 2u;
        const u32 jn = (step + 1u) & 3u;
        r11_g02_load_a(lx, kn, jn, r16, h, a0_c, a1_c);
      }
      __builtin_amdgcn_sched_barrier(0);
      float tv[8];
#pragma unroll
      for (u32 v = 0; v < 8u; v++) tv[v] = (float)c0[v] * s0;
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (u32 v = 0; v < 8u; v++) tv[v] += (float)c1[v] * s1;
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (u32 v = 0; v < 8u; v++) acc[j][v] += tv[v] * dx[v];
      R11_G02_PIN_ACC8(acc[j]);
      __builtin_amdgcn_sched_barrier(0);
    }
  }
}

/* Compact, 32-deep batched 64-token half epilogue. Caller guarantees row < rows.
 * Preserves exact epilogue() FP order: if (bias) v = v + bval; if (res) v = rval + v;
 * Never adds 0.0f when bias == NULL (preserves -0.0f). */
static inline __attribute__((always_inline)) void r11_g02_store_half(
    G u8 *y, const G float *bias, const G u8 *res, const u32 T, const u32 ys, const u32 rs,
    const u32 row, const u32 t0_half, const u32 h, const float bval, const float acc[4][8]) {
  if (t0_half >= T) return;
  if (t0_half + 64u <= T) {
    G u8 *yp = y + (unsigned long)(t0_half + 8u * h) * ys + (unsigned long)row * 4ul;
    const unsigned long ysu = (unsigned long)ys;
    if (res) {
      const G u8 *rp = res + (unsigned long)(t0_half + 8u * h) * rs + (unsigned long)row * 4ul;
      const unsigned long rsu = (unsigned long)rs;
      float rv[4][8];
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
#pragma unroll
        for (u32 v = 0; v < 8u; v++) rv[j][v] = *(const G float *)(rp + (16ul * j + v) * rsu);
      }
      __builtin_amdgcn_sched_barrier(0);
      if (bias) {
#pragma unroll
        for (u32 j = 0; j < 4u; j++) {
#pragma unroll
          for (u32 v = 0; v < 8u; v++) *(G float *)(yp + (16ul * j + v) * ysu) = rv[j][v] + (acc[j][v] + bval);
        }
      } else {
#pragma unroll
        for (u32 j = 0; j < 4u; j++) {
#pragma unroll
          for (u32 v = 0; v < 8u; v++) *(G float *)(yp + (16ul * j + v) * ysu) = rv[j][v] + acc[j][v];
        }
      }
    } else if (bias) {
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
#pragma unroll
        for (u32 v = 0; v < 8u; v++) *(G float *)(yp + (16ul * j + v) * ysu) = acc[j][v] + bval;
      }
    } else {
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
#pragma unroll
        for (u32 v = 0; v < 8u; v++) *(G float *)(yp + (16ul * j + v) * ysu) = acc[j][v];
      }
    }
  } else {
#pragma unroll
    for (u32 j = 0; j < 4u; j++) {
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const u32 t = t0_half + 16u * j + 8u * h + v;
        if (t < T) {
          float val = acc[j][v];
          if (bias) val = val + bval;
          if (res) val = ((const G float *)(res + (unsigned long)t * rs))[row] + val;
          ((G float *)(y + (unsigned long)t * ys))[row] = val;
        }
      }
    }
  }
}

KERNEL ie_saluki_gemm_rq128_iq2s_r11(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T,
                                     const G float *bias, const G u8 *res, u32 xs, u32 ys, u32 rs) {
  static LDS u8 lx[2u * 64u * 128u] __attribute__((aligned(16)));
  static LDS float lsx[2u * 4u * 64u] __attribute__((aligned(32)));
  static LDS u16 lut[1024] __attribute__((aligned(16)));
  if (T == 0u || rows == 0u || nb < 8u || (nb & 7u) != 0u || !q || !xq || !y) return;
  const u32 ntt = (T + 127u) / 128u, tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 rbase = rt * 128u, tbase = tt * 128u;
  if (rbase >= rows || tbase >= T) return;
  const u32 t0 = tid(), w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u)), l = lane();
  const u32 r16 = l & 15u, h = l >> 4u;
  const int is_s = (w < 2u);
  const u32 r_loc = 16u * w + r16, row = rbase + r_loc, rr = row < rows ? row : rows - 1u;
  const u32 mb = nb >> 3u;
  const G u8 *bp = q + (unsigned long)rr * mb * 82u;
  u32 cur_psel = ((unsigned long)bp & 2ul) ? 0x07060504u : 0x05040302u;
  const u32 h_off = 32u * h;
  const u32 st_tk = t0 >> 3u, st_c8 = t0 & 7u;
  const u32 st_off0 = st_tk * 128u + ((st_c8 ^ (st_tk & 7u)) * 16u);

  LDSP u8 *lx0 = lx, *lx1 = lx + 64u * 128u;
  LDSP float *lsx0 = lsx, *lsx1 = lsx + 4u * 64u;

  float acc0[4][8], acc1[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++) {
#pragma unroll
    for (u32 v = 0; v < 8u; v++) acc0[j][v] = 0.0f, acc1[j][v] = 0.0f;
  }

  /* Stage compact 2 KiB interleaved u16 IQ2_S grid LUT (4 iterations x 256 threads). */
#pragma unroll
  for (u32 i = 0; i < 4u; i++) {
    const u32 idx = t0 + i * NT;
    lut[idx] = r11_g02_pack_iq2_u16(ie_rq_iq2s_grid[idx]);
  }

  /* Uniform tile bases + 32-bit lane offsets -> saddr + voffset global_load_b128. */
  const G u8 *xq_tile = xq + (unsigned long)tbase * xs;
  const G u8 *sx_tile = xq_tile + 32ul * nb;
  const u32 dt_max = T - 1u - tbase;
  u32 xoffA[2], xoffB[2];
#pragma unroll
  for (u32 i = 0; i < 2u; i++) {
    const u32 p = t0 + 256u * i, tk = p >> 3u, c8 = p & 7u;
    const u32 dtA = tk <= dt_max ? tk : dt_max;
    const u32 dtB = 64u + tk <= dt_max ? 64u + tk : dt_max;
    xoffA[i] = dtA * xs + c8 * 16u;
    xoffB[i] = dtB * xs + c8 * 16u;
  }
  const u32 s_tk = t0 & 63u;
  const u32 sxoffA = (s_tk <= dt_max ? s_tk : dt_max) * xs;
  const u32 sxoffB = (64u + s_tk <= dt_max ? 64u + s_tk : dt_max) * xs;

  /* Prologue: load sb=0 hb0 A-half activations + sb=0 hb0/shared weights. */
  u32x4 rx[2];
  f32x4 rsx = (f32x4)(0.0f);
#pragma unroll
  for (u32 i = 0; i < 2u; i++) rx[i] = r11_g02_ld4(xq_tile + (unsigned long)xoffA[i]);
  if (is_s) rsx = rq_load_f32x4((const G float *)(sx_tile + (unsigned long)sxoffA));
  u32 cur_w0[5], cur_w8[5], cur_w16[5];
  r11_g02_iq2s_load_hb0(bp, cur_w0, cur_w8, cur_w16);
  r11_g02_stage_x64(lx0, lsx0, st_off0, t0, is_s, rx, rsx);
  lbarrier();
  __asm__ volatile("" : "+v"(cur_w0[0]), "+v"(cur_w0[4]), "+v"(cur_w8[0]), "+v"(cur_w8[4]),
                        "+v"(cur_w16[0]), "+v"(cur_w16[4]));

#pragma unroll 1
  for (u32 sb = 0u; ; sb++) {
    const u32 kb0 = 8u * sb;
    const u32 kb1 = kb0 + 4u;
    const u32 kb0_x = kb0 * 32u, kb0_s = kb0 * 4u;
    const u32 kb1_x = kb1 * 32u, kb1_s = kb1 * 4u;

    /* Phase 1 (hb0 A-half): prefetch hb0 B-half activations; decode & apply hb0 on lx0 -> acc0. */
#pragma unroll
    for (u32 i = 0; i < 2u; i++) rx[i] = r11_g02_ld4(xq_tile + (unsigned long)(xoffB[i] + kb0_x));
    if (is_s) rsx = rq_load_f32x4((const G float *)(sx_tile + (unsigned long)(sxoffB + kb0_s)));
    __builtin_amdgcn_sched_barrier(0);

    float d4 = ie_rq_h2f((cur_w0[0] >> ((cur_psel & 4u) << 2u)) & 0xFFFFu) * 0.25f;
    __asm__ volatile("" : "+v"(d4));
    const u32 qh0 = r11_g02_perm_al(cur_w16[0], cur_w16[1], cur_psel);
    const u32 qh1 = r11_g02_perm_al(cur_w16[1], cur_w16[2], cur_psel);
    const u32 sc0 = r11_g02_perm_al(cur_w16[2], cur_w16[3], cur_psel);
    const u32 sc1 = r11_g02_perm_al(cur_w16[3], cur_w16[4], cur_psel);
    const u32 carry_b8 = cur_w8[0];
    const u32 carry_b16 = cur_w16[0];

    r11_g02_dc_t xdc[4];
    v2i a0_0, a1_0;
    r11_g02_iq2s_decode_all(cur_w0, cur_w8, qh0, cur_psel, lut, lx0, r16, h, h_off, xdc, &a0_0, &a1_0);
    __builtin_amdgcn_sched_barrier(0);
    r11_g02_apply_k_range(xdc, sc0, d4, lx0, lsx0, r16, h, 0u, 4u, &a0_0, &a1_0, acc0);

    r11_g02_stage_x64(lx1, lsx1, st_off0, t0, is_s, rx, rsx);
    lbarrier();

    /* Phase 2 (hb0 B-half): prefetch hb1 A-half activations before k=0..1, then issue 2x b128
     * hb1_tail after xdc[0..1] die at k=2 (FIFO after rx/rsx -> stage_x64 waits only for rx/rsx). */
    r11_g02_load_a(lx1, 0u, 0u, r16, h, &a0_0, &a1_0);
#pragma unroll
    for (u32 i = 0; i < 2u; i++) rx[i] = r11_g02_ld4(xq_tile + (unsigned long)(xoffA[i] + kb1_x));
    if (is_s) rsx = rq_load_f32x4((const G float *)(sx_tile + (unsigned long)(sxoffA + kb1_s)));
    __builtin_amdgcn_sched_barrier(0);
    r11_g02_apply_k_range(xdc, sc0, d4, lx1, lsx1, r16, h, 0u, 2u, &a0_0, &a1_0, acc1);

    u32 hb1_q1[4], hb1_s1[4];
    r11_g02_iq2s_load_hb1_tail(bp, hb1_q1, hb1_s1);
    __builtin_amdgcn_sched_barrier(0);
    r11_g02_apply_k_range(xdc, sc0, d4, lx1, lsx1, r16, h, 2u, 4u, &a0_0, &a1_0, acc1);

    r11_g02_stage_x64(lx0, lsx0, st_off0, t0, is_s, rx, rsx);
    lbarrier();

    /* Phase 3 (hb1 A-half): drain hb1_tail (in flight since Phase 2 k=2), prefetch hb1 B-half X,
     * decode & apply hb1 on lx0 -> acc0. */
    u32 qlo1_w[5] = {hb1_q1[0], hb1_q1[1], hb1_q1[2], hb1_q1[3], carry_b8};
    u32 sgn1_w[5] = {hb1_s1[0], hb1_s1[1], hb1_s1[2], hb1_s1[3], carry_b16};
    __asm__ volatile("" : "+v"(qlo1_w[0]), "+v"(sgn1_w[0]));
#pragma unroll
    for (u32 i = 0; i < 2u; i++) rx[i] = r11_g02_ld4(xq_tile + (unsigned long)(xoffB[i] + kb1_x));
    if (is_s) rsx = rq_load_f32x4((const G float *)(sx_tile + (unsigned long)(sxoffB + kb1_s)));
    __builtin_amdgcn_sched_barrier(0);

    r11_g02_iq2s_decode_all(qlo1_w, sgn1_w, qh1, cur_psel, lut, lx0, r16, h, h_off, xdc, &a0_0, &a1_0);
    __builtin_amdgcn_sched_barrier(0);
    r11_g02_apply_k_range(xdc, sc1, d4, lx0, lsx0, r16, h, 0u, 4u, &a0_0, &a1_0, acc0);

    r11_g02_stage_x64(lx1, lsx1, st_off0, t0, is_s, rx, rsx);
    lbarrier();

    /* Phase 4 (hb1 B-half): advance to sb+1, prefetch next SB A-half activations before k=0..1,
     * then issue next SB hb0 weights after xdc[0..1] die at k=2 (reusing freed VGPRs).
     * Explicit if (!more) break eliminates the !more backedge phi so cur_w0/w8/w16 (15 VGPRs)
     * die immediately after Phase 1 decode_all instead of remaining live across the whole loop. */
    bp += 82u;
    cur_psel ^= 0x02020606u;
    const int more = sb + 1u < mb;
    r11_g02_load_a(lx1, 0u, 0u, r16, h, &a0_0, &a1_0);
    if (more) {
      const u32 kbn = 8u * (sb + 1u);
      const u32 kbn_x = kbn * 32u, kbn_s = kbn * 4u;
#pragma unroll
      for (u32 i = 0; i < 2u; i++) rx[i] = r11_g02_ld4(xq_tile + (unsigned long)(xoffA[i] + kbn_x));
      if (is_s) rsx = rq_load_f32x4((const G float *)(sx_tile + (unsigned long)(sxoffA + kbn_s)));
    }
    __builtin_amdgcn_sched_barrier(0);
    r11_g02_apply_k_range(xdc, sc1, d4, lx1, lsx1, r16, h, 0u, 2u, &a0_0, &a1_0, acc1);

    u32 nxt_w0[5], nxt_w8[5], nxt_w16[5];
    if (more) {
      r11_g02_iq2s_load_hb0(bp, nxt_w0, nxt_w8, nxt_w16);
    }
    __builtin_amdgcn_sched_barrier(0);
    r11_g02_apply_k_range(xdc, sc1, d4, lx1, lsx1, r16, h, 2u, 4u, &a0_0, &a1_0, acc1);

    if (!more) break;
    r11_g02_stage_x64(lx0, lsx0, st_off0, t0, is_s, rx, rsx);
    lbarrier();
#pragma unroll
    for (u32 i = 0; i < 5u; i++) {
      cur_w0[i] = nxt_w0[i];
      cur_w8[i] = nxt_w8[i];
      cur_w16[i] = nxt_w16[i];
    }
    __asm__ volatile("" : "+v"(cur_w0[0]), "+v"(cur_w0[4]), "+v"(cur_w8[0]), "+v"(cur_w8[4]),
                          "+v"(cur_w16[0]), "+v"(cur_w16[4]));
  }

  if (row >= rows) return;
  u32 ho_ep = h_off;
  __asm__ volatile("" : "+v"(ho_ep));
  const u32 h_ep = ho_ep >> 5u;
  const float bval = bias ? bias[row] : 0.0f;
  r11_g02_store_half(y, bias, res, T, ys, rs, row, tbase, h_ep, bval, acc0);
  r11_g02_store_half(y, bias, res, T, ys, rs, row, tbase + 64u, h_ep, bval, acc1);
}

#undef R11_G02_PIN_ACC8
