/* ============================================================================
 * Inkling-2 Round 7 (Rank 1): Superblock-Aligned 8-Lane/Row Q4_K Verify Output
 * Head GEMV + Optional Fused Logit-Cap for RX 9070 XT (gfx1201, wave32, NT=256).
 *
 * Target: output.weight (MAT_Q4K, 201024 x 2304, nb = 72 = 9 super-blocks),
 *         verify T = 1..8 (especially T = 2..8 DSpark verify where base
 *         ie_gemv_q4kq8_ts2 hits 193 VGPR = 1 WG/CU, bypasses cache with
 *         TH_LOAD_NT across TS=2 waves = 2x DRAM reads of 260.5 MB, and wastes
 *         25% of lanes on nb = 72 = 32 + 32 + 8).
 *
 * Exported Kernel:
 *   KERNEL ie_inkling_gemv_q4k_head_ver_r7(
 *       const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y,
 *       u32 rows, u32 nb, const G float *bias, const G float *res,
 *       u32 T, u32 xs, u32 ys, u32 rs, u32 unpadded, float cap)
 *
 * Grid:
 *   ((rows + 31u) / 32u, 1u, 1u) with NT = 256 (8 waves x 4 rows/wave = 32 rows/WG).
 *   For Inkling-2 output.weight (rows = 201024): grid = (6282, 1, 1).
 *   Safe for padded grids (e.g. 253..256 -> 257).
 *
 * Arguments & Stride Contract (matches ie_gemv_q4kq8_ts2 / _tr8):
 *   - xs       : per-token BYTE stride of Q8_0 activation buffer xq.
 *   - ys       : per-token FLOAT stride of output buffer y (g->bufs[o->b].stride / 4u).
 *   - rs       : per-token FLOAT stride of residual res (or 0u if res == NULL).
 *   - unpadded : if unpadded > 0u and r >= unpadded, stores -1e30f directly for
 *                all tokens 0..T-1 (skipping Q4_K dot products on padded rows,
 *                matching ie_logit_cap). Pass 0u for unfused Q4_K GEMV.
 *   - cap      : if cap > 0.0f, applies exact cap * ie_tanhf_s(val / cap) after
 *                epilogue(val, r, bias, rp) for rows < unpadded (or all rows if
 *                unpadded == 0u). Pass 0.0f for unfused Q4_K GEMV.
 *
 * Dispatch Guards (src/hip.c):
 *   - Unfused OP_GEMV_Q4K (T = 2..8, or T == 1 without fused norm):
 *       !mk.rec && T >= 1u && T <= 8u && rows > 0u && nb > 0u && (nb & 7u) == 0u
 *       && !NW
 *     Pass unpadded = 0u, cap = 0.0f.
 *   - Fused [OP_GEMV_Q4K, OP_LOGIT_CAP] in ink_r2_fuse (T = 1..8):
 *       !mk.rec && T >= 1u && T <= 8u && o0->kind == OP_GEMV_Q4K && !NW &&
 *       o0->w->rows > 0u && o0->w->nb > 0u && (o0->w->nb & 7u) == 0u &&
 *       o1->kind == OP_LOGIT_CAP && o1->a == o0->b && o1->n == o0->w->rows &&
 *       (o1->b == o0->b || buf_disjoint(g, o0->a, o1->b))
 *     Pass y = B(o1->b), ys = g->bufs[o1->b].stride / 4u,
 *     unpadded = o1->nh, cap = o1->clamp (consumes 2 ops).
 *
 * Parity & Resources:
 *   - 100% bitwise-exact (0-bit-diff) vs ie_gemv_q4kq8 / ie_gemv_q4kq8_ts2
 *     (+ ie_logit_cap when fused): preserves exact per-lane block accumulation
 *     across 4 virtual 32-lane accumulators (a0..a3), #pragma clang fp contract(off),
 *     explicit __builtin_fmaf(-mm, xa, __builtin_fmaf(sc * (float)dq, da, acc)),
 *     and exact wave_tree_f (+16 -> u0/u1, +8 -> s, +4/+2/+1 DPP shl_i) order.
 *   - Target resources: <= 96 VGPR (4 WGs/CU = 16 waves/SIMD), 0 B LDS, 0/0 spills.
 * ========================================================================== */

#define INK_R7_Q4K_TMAX 8u

struct __attribute__((packed, aligned(4))) ink_r7_q4k_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r7_q4k_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r7_q4k_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) void ink_r7_q4k_sb_step(
    const G u32 *row_qw, const G u16 *row_sm, const G u32 *row_dd32, const G u8 *xq,
    u32 nb, u32 T, u32 xs, u32 sb_u, u32 sub, u32 sub32, u32 sub4, v8f *acc) {
#pragma clang fp contract(off)
  const u32 b = (sb_u << 3u) + sub;

  /* Cached contiguous 128 B/row Q4_K weight load (0 TH_LOAD_NT, 1x DRAM pass). */
  const u32x4 w4 = ink_r7_q4k_ld4(row_qw + (unsigned long)b * 4ul);
  const u32 sm = (u32)row_sm[b];
  const u32 dd_raw = row_dd32[sb_u];
  const u16 d_bits = (u16)(dd_raw & 0xFFFFu);
  const u16 dm_bits = (u16)(dd_raw >> 16u);
  const float d = (float)__builtin_bit_cast(f16, d_bits);
  const float dmin = (float)__builtin_bit_cast(f16, dm_bits);
  const float sc = d * (float)(sm & 63u);
  const float mm = dmin * (float)(sm >> 8u);

  const u32 lo0 = ie_q4k_lo(w4.x), hi0 = ie_q4k_hi(w4.x);
  const u32 lo1 = ie_q4k_lo(w4.y), hi1 = ie_q4k_hi(w4.y);
  const u32 lo2 = ie_q4k_lo(w4.z), hi2 = ie_q4k_hi(w4.z);
  const u32 lo3 = ie_q4k_lo(w4.w), hi3 = ie_q4k_hi(w4.w);

  const unsigned long aw_sb = (unsigned long)sb_u * 256ul;
  const unsigned long da_sb = (unsigned long)nb * 32ul + (unsigned long)sb_u * 32ul;
  const unsigned long as_sb = (unsigned long)nb * 36ul + (unsigned long)sb_u * 32ul;

  v8f av = *acc;
#pragma unroll
  for (u32 j = 0u; j < INK_R7_Q4K_TMAX; j++) {
    if (j < T) {
      const G u8 *xj = xq + (unsigned long)j * (unsigned long)xs;
      const G u8 *xj_aw = xj + aw_sb;
      const u32x4 act0 = ink_r7_q4k_ld4(xj_aw + sub32);
      const u32x4 act1 = ink_r7_q4k_ld4(xj_aw + sub32 + 16u);
      const float da = *(const G float *)(xj + da_sb + sub4);
      const int as = (int)(*(const G u32 *)(xj + as_sb + sub4));
      const float xa = da * (float)as;

      u32 dq = ie_dot4_us(0u, lo0, act0.x);
      dq = ie_dot4_us(dq, hi0, act1.x);
      dq = ie_dot4_us(dq, lo1, act0.y);
      dq = ie_dot4_us(dq, hi1, act1.y);
      dq = ie_dot4_us(dq, lo2, act0.z);
      dq = ie_dot4_us(dq, hi2, act1.z);
      dq = ie_dot4_us(dq, lo3, act0.w);
      dq = ie_dot4_us(dq, hi3, act1.w);

      const float prod = sc * (float)(int)dq;
      const float a_mid = __builtin_fmaf(prod, da, av[j]);
      av[j] = __builtin_fmaf(-mm, xa, a_mid);
      if ((j & 1u) == 1u) __builtin_amdgcn_sched_barrier(0);
    }
  }
  *acc = av;
  __builtin_amdgcn_sched_barrier(0);
}

/* ============================================================================
 * 8-lane/row Q4_K GEMV + optional fused logit-cap (T = 1..8, 4 rows/wave).
 * Uses SSA v8f accumulators (a0..a3) for 0 private stack / 0 spills and exact
 * 0-bit-diff wave_tree_f parity, with 4x unrolled superblock loop eliminating
 * runtime slot branches and pair-wise sched_barrier to stay <= 96 VGPR.
 * ========================================================================== */
KERNEL ie_inkling_gemv_q4k_head_ver_r7(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y,
                                       u32 rows, u32 nb, const G float *bias, const G float *res,
                                       u32 T, u32 xs, u32 ys, u32 rs, u32 unpadded, float cap) {
#pragma clang fp contract(off)
  if (T == 0u || T > INK_R7_Q4K_TMAX || rows == 0u || nb == 0u || (nb & 7u) != 0u || !qw || !qs || !xq || !y)
    return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 r = (wgid() << 5u) + (w << 2u) + (l >> 3u);
  if (r >= rows) return;

  /* Padded vocabulary tail fast-exit when fused with OP_LOGIT_CAP (matches ie_logit_cap). */
  if (unpadded > 0u && r >= unpadded) {
    if (sub == 0u) {
#pragma unroll
      for (u32 j = 0u; j < INK_R7_Q4K_TMAX; j++) {
        if (j < T) y[(unsigned long)j * (unsigned long)ys + (unsigned long)r] = -1e30f;
      }
    }
    return;
  }

  const u32 nsb = nb >> 3u;
  const G u32 *row_qw = qw + (unsigned long)r * (unsigned long)nb * 4ul;
  const G u16 *row_sm = qs + (unsigned long)r * (unsigned long)nb;
  const G u32 *row_dd32 = (const G u32 *)(qs + (unsigned long)rows * (unsigned long)nb +
                                          (unsigned long)r * (unsigned long)nsb * 2ul);
  const u32 sub32 = sub << 5u;
  const u32 sub4 = sub << 2u;

  /* Four virtual 32-lane accumulators per token (SSA v8f -> guaranteed 0 private memory). */
  v8f a0 = (v8f)(0.0f), a1 = (v8f)(0.0f), a2 = (v8f)(0.0f), a3 = (v8f)(0.0f);

  const u32 nsb4 = nsb & ~3u;
#pragma clang loop unroll(disable)
  for (u32 sb0 = 0u; sb0 < nsb4; sb0 += 4u) {
    const u32 sb_u = __builtin_amdgcn_readfirstlane(sb0);
    ink_r7_q4k_sb_step(row_qw, row_sm, row_dd32, xq, nb, T, xs, sb_u + 0u, sub, sub32, sub4, &a0);
    ink_r7_q4k_sb_step(row_qw, row_sm, row_dd32, xq, nb, T, xs, sb_u + 1u, sub, sub32, sub4, &a1);
    ink_r7_q4k_sb_step(row_qw, row_sm, row_dd32, xq, nb, T, xs, sb_u + 2u, sub, sub32, sub4, &a2);
    ink_r7_q4k_sb_step(row_qw, row_sm, row_dd32, xq, nb, T, xs, sb_u + 3u, sub, sub32, sub4, &a3);
  }
  if (nsb4 < nsb) {
    const u32 sb_t = __builtin_amdgcn_readfirstlane(nsb4);
    ink_r7_q4k_sb_step(row_qw, row_sm, row_dd32, xq, nb, T, xs, sb_t + 0u, sub, sub32, sub4, &a0);
    if (sb_t + 1u < nsb)
      ink_r7_q4k_sb_step(row_qw, row_sm, row_dd32, xq, nb, T, xs, sb_t + 1u, sub, sub32, sub4, &a1);
    if (sb_t + 2u < nsb)
      ink_r7_q4k_sb_step(row_qw, row_sm, row_dd32, xq, nb, T, xs, sb_t + 2u, sub, sub32, sub4, &a2);
  }

  /* Exact wave_tree_f order: +16 (a0+a2, a1+a3), +8 (u0+u1), then DPP +4, +2, +1. */
  const int use_cap = (cap > 0.0f);
  const float inv_cap = use_cap ? (1.0f / cap) : 0.0f;
#pragma unroll
  for (u32 j = 0u; j < INK_R7_Q4K_TMAX; j++) {
    if (j < T) {
      const float u0 = a0[j] + a2[j];
      const float u1 = a1[j] + a3[j];
      float s = u0 + u1;
      s = s + I2F(shl_i(F2I(s), 4));
      s = s + I2F(shl_i(F2I(s), 2));
      s = s + I2F(shl_i(F2I(s), 1));
      if (sub == 0u) {
        const G float *rp = res ? (res + (unsigned long)j * (unsigned long)rs) : (const G float *)0;
        float val = epilogue(s, r, bias, rp);
        if (use_cap) val = cap * ie_tanhf_s(val * inv_cap);
        y[(unsigned long)j * (unsigned long)ys + (unsigned long)r] = val;
      }
    }
  }
}

#undef INK_R7_Q4K_TMAX
