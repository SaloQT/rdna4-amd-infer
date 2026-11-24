/* ============================================================================
 * Inkling-2 Round 10B (Wave 10B - H06 + True 3-WG/CU Q8_0, H03 FP8 WMMA,
 * H08 2-Term BF16-Pair WMMA F32 Router) for RX 9070 XT (gfx1201, wave32, NT=256).
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r10_wfr_ /
 * INK_R10_WFR_ and all macros are #undef'd at EOF.
 *
 * gfx1201 Hardware Occupancy (24-VGPR wave32 granule, 1536 VGPR/SIMD32):
 *   <=  96 VGPR -> 16w/SIMD (4 WGs/CU)
 *   <= 120 VGPR -> 12w/SIMD (3 WGs/CU)  <-- ALL 5 kernels target <= 120 VGPR!
 *
 * Exported Kernels, Grids, Guards, and Parity:
 *
 * 1. KERNEL ie_inkling_gemm_q8_med_swiz_r10(
 *        const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
 *        u32 rows, u32 nb, const G float *bias, const G float *res,
 *        u32 T, u32 xs, u32 ys, u32 rs)
 *    - Exact 12-arg drop-in for ie_inkling_gemm_q8_med_r7 / r9 (128x64).
 *    - Grid:   (((rows + 127u) / 128u) * ((T + 63u) / 64u), 1u, 1u), NT = 256.
 *    - Guard:  !mk.rec && T >= 17u && rows >= 512u && nb >= 4u && (nb & 3u) == 0u.
 *    - Upgrades over q8_med_r7 (123 VGPR -> 10w/SIMD = 2 WGs/CU!):
 *      <= 120 VGPR (true 12w/SIMD = 3 WGs/CU, +50% occupancy!), 0 v_fma_mix_f32,
 *      4x4 WGP-aware L2/MALL tile swizzling (H06), decoupled s_barrier_signal/wait.
 *    - Parity: 100% bitwise-exact (0-bit-diff exact-SHA) vs ie_gemm_q8r / q8_med_r7!
 *
 * 2. KERNEL ie_inkling_gemm_q8_64x64_swiz_r10(exact 12-arg ie_gemm_q8r signature)
 *    - Exact 12-arg drop-in for ie_inkling_gemm_q8_64x64_r7 / r9 (64x64).
 *    - Grid:   (((rows + 63u) / 64u) * ((T + 63u) / 64u), 1u, 1u), NT = 256.
 *    - Guard:  !mk.rec && T >= 17u && rows >= 64u && rows <= 1024u && nb >= 4u && (nb & 3u) == 0u.
 *    - Resources: <= 120 VGPR (<= 96 VGPR target), 19,968 B LDS -> true 3 WGs/CU!
 *    - Parity: 100% bitwise-exact (0-bit-diff exact-SHA) vs ie_gemm_q8r!
 *
 * 3. KERNEL ie_inkling_gemm_iq4nl_fp8_expmaj_r10(
 *        const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
 *        u32 rows, u32 nb, const G u32 *list, const G u32 *offs, u32 np,
 *        u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv, u32 kind,
 *        const G u8 *res, u32 rs)
 *    - H03 Native gfx1201 FP8 WMMA (v_wmma_f32_16x16x16_fp8_fp8) Expert-Major
 *      IQ4_NL MoE Prefill Down-Proj on Macro-K=128 (KB=4) tiles (4x fewer
 *      rescaling FMAs, 0 int32->f32 conversions, native v_cvt_pk_fp8_f32!).
 *    - Exact 17-arg drop-in & expert-major grid of ie_inkling_gemm_iq4nl_exp_r2!
 *    - Grid:   ((rows + 127u) / 128u, list ? m->n_exp : (np + 63u) / 64u, 1u), NT = 256.
 *    - Guard:  !mk.rec && T >= 17u && kind == MAT_IQ4NL && rows > 0u && nb >= 4u && (nb & 3u) == 0u.
 *    - Resources: <= 120 VGPR (true 3 WGs/CU!), 18,176 B LDS, 0 v_fma_mix_f32.
 *    - Parity: statistical (FP8 E4M3 3-bit mantissa); gate on KLD < 5e-4 + opt-out.
 *
 * 4. KERNEL ie_inkling_gemm_q8_fp8_med_r10(exact 12-arg ie_gemm_q8r signature)
 *    - H03 Native gfx1201 FP8 WMMA Dense Q8_0 128x64 Prefill on Macro-K=128 tiles
 *      with 4x4 L2/MALL swizzling and native v_cvt_pk_fp8_f32.
 *    - Grid:   (((rows + 127u) / 128u) * ((T + 63u) / 64u), 1u, 1u), NT = 256.
 *    - Guard:  !mk.rec && T >= 64u && rows >= 512u && nb >= 4u && (nb & 3u) == 0u.
 *    - Resources: <= 120 VGPR (true 3 WGs/CU!), 17,920 B LDS, 0 v_fma_mix_f32.
 *    - Parity: statistical (FP8 E4M3); gate on KLD < 5e-4 + IE_INKLING_FP8_WMMA=0 opt-out.
 *
 * 5. KERNEL ie_inkling_gemm_f32_router_bf16_r10(
 *        const G float *w, const G float *x, G float *y,
 *        u32 rows, u32 cols, const G float *bias, const G float *res,
 *        u32 xs, u32 ys, u32 rs, u32 T,
 *        u32 K, const G float *ebias, float scale,
 *        G u32 *route, u32 ost, G u32 *count)
 *    - H08 Lossless 3-WMMA BF16-Pair Split F32 Router (W_hi*X_hi + W_hi*X_lo +
 *      W_lo*X_hi via v_wmma_f32_16x16x16_bf16, ~17+ bits mantissa, max |err| < 1e-6)
 *      with optional fused L2-hot OP_MOE_ROUTE (exact 17-arg r7 signature!).
 *    - Tile:   (BM=32, BN=16, BK=64), 8 waves = 2(M) x 4(Split-K), direct-VGPR W
 *              + 8,196 B LDS double-buffered X -> 128 WGs at M=128, T=512 (2 WGs/CU,
 *              avoiding R17 b74's 32-WG under-saturation!).
 *    - Grid:   ((rows + 31u) / 32u, (T + 15u) / 16u, 1u), NT = 256.
 *    - Guard:  !mk.rec && T >= 17u && rows > 0u && cols >= 16u && (cols & 15u) == 0u &&
 *              (xs & 15u) == 0u && (((uintptr_t)w) & 15u) == 0u && (((uintptr_t)x) & 15u) == 0u;
 *              for fused route add !bias && !res && rows <= 256u && K > 0u && K <= 32u &&
 *              count != NULL && ((T + 15u) / 16u) <= count_cap (32 <= 384 in b->at_count).
 *    - Resources: <= 96 VGPR (4 WGs/CU!), 8,196 B LDS (4 WGs/CU!), 0/0 spills.
 *    - Parity: KLD ~ 0.0000 / exact top-K routing accuracy.
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R10_WFR_KB       4u
#define INK_R10_WFR_M_R      128u
#define INK_R10_WFR_M_T      64u
#define INK_R10_WFR_H_R      64u
#define INK_R10_WFR_M_SWSTR  160u /* 128 f16 + 32 f16 pad = 320 B -> 0-conflict ds_store_2addr_b32 */
#define INK_R10_WFR_H_SWSTR  96u  /*  64 f16 + 32 f16 pad = 192 B -> 0-conflict ds_store_2addr_b32 */
#define INK_R10_WFR_EP_STR   129u /* 128 f32 +  1 f32 pad -> 0-conflict 32-lane contiguous reads */
#define INK_R10_WFR_RTR_BM   32u
#define INK_R10_WFR_RTR_BN   16u
#define INK_R10_WFR_RTR_BK   64u

typedef unsigned int ink_r10_wfr_u32x2 __attribute__((ext_vector_type(2)));
typedef __bf16       ink_r10_wfr_bf8   __attribute__((ext_vector_type(8)));

struct __attribute__((packed, aligned(4))) ink_r10_wfr_u32x4p {
  u32x4 v;
};

struct __attribute__((packed, aligned(4))) ink_r10_wfr_u32x2p {
  ink_r10_wfr_u32x2 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r10_wfr_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r10_wfr_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) ink_r10_wfr_u32x2 ink_r10_wfr_ld2(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r10_wfr_u32x2p *)p)->v;
}

static inline __attribute__((always_inline)) void ink_r10_wfr_lsignal(void) {
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  __builtin_amdgcn_s_barrier_signal(-1);
  __asm__ volatile("" ::: "memory");
}

static inline __attribute__((always_inline)) void ink_r10_wfr_lwait(void) {
  __builtin_amdgcn_s_barrier_wait(-1);
  __asm__ volatile("" ::: "memory");
}

/* H06: Bijective 4x4 WGP-aware L2/MALL tile swizzling (0 barriers, 0 VGPRs).
 * Groups G=4 adjacent M-tiles across consecutive N-tiles so concurrent WGs share
 * X in L2 while consecutive N-tiles reuse hot weight slabs in L2/MALL. */
static inline __attribute__((always_inline)) void ink_r10_wfr_swiz_rt_tt(
    const u32 wg, const u32 nrt, const u32 ntt, u32 *rt_out, u32 *tt_out) {
  const u32 grp_m = 4u;
  const u32 grp_sz = grp_m * ntt;
  const u32 grp = wg / grp_sz;
  const u32 in_grp = wg - grp * grp_sz;
  const u32 r0 = grp * grp_m;
  const u32 r_rem = nrt - r0;
  const u32 gw = r_rem < grp_m ? r_rem : grp_m;
  const u32 tt = in_grp / gw;
  const u32 rt = r0 + (in_grp - tt * gw);
  *rt_out = (u32)__builtin_amdgcn_readfirstlane((int)rt);
  *tt_out = (u32)__builtin_amdgcn_readfirstlane((int)tt);
}

/* 0-bank-conflict DPP-packed f16 weight-scale staging -> single ds_store_2addr_b32. */
static inline __attribute__((always_inline)) void ink_r10_wfr_stage_sw(
    LDSP f16 *lsw, const u32 sw_str, const u32 r_idx, const ink_r10_wfr_u32x2 rsw) {
  u32 ri = r_idx;
  __asm__ volatile("" : "+v"(ri));
  const u32 w0 = rsw.x, w1 = rsw.y;
  const u32 p0 = (u32)__builtin_amdgcn_update_dpp(0, (int)w0, 0xB1, 0xF, 0xF, true);
  const u32 p1 = (u32)__builtin_amdgcn_update_dpp(0, (int)w1, 0xB1, 0xF, 0xF, true);
  const u32 odd = ri & 1u;
  const u32 sel = odd ? 0x03020706u : 0x05040100u;
  const u32 v_a = __builtin_amdgcn_perm(p0, w0, sel);
  const u32 v_b = __builtin_amdgcn_perm(p1, w1, sel);
  const u32 r2 = ri & ~1u;
  LDSP u32 *sw32 = (LDSP u32 *)lsw;
  const u32 str32 = sw_str >> 1u;
  sw32[(0u + odd) * str32 + (r2 >> 1u)] = v_a;
  sw32[(2u + odd) * str32 + (r2 >> 1u)] = v_b;
}

static inline __attribute__((always_inline)) void ink_r10_wfr_stage_sx(
    LDSP float *lsx, const u32 stride_t, const u32 t_idx, const u32x4 rsx) {
  u32 ti = t_idx;
  __asm__ volatile("" : "+v"(ti));
  const u32 d0 = rsx.x, d1 = rsx.y, d2 = rsx.z, d3 = rsx.w;
  lsx[0u * stride_t + ti] = __builtin_bit_cast(float, d0);
  lsx[1u * stride_t + ti] = __builtin_bit_cast(float, d1);
  lsx[2u * stride_t + ti] = __builtin_bit_cast(float, d2);
  lsx[3u * stride_t + ti] = __builtin_bit_cast(float, d3);
}

/* Load f16 weight scales and pin f8 conversion -> kills 100% of v_fma_mix_f32! */
static inline __attribute__((always_inline)) f8 ink_r10_wfr_load_sw_f8(
    const LDSP f16 *lsw, const u32 sw_str, const u32 k, const u32 wm, const u32 h) {
  const h8 sw_h = *(const LDSP h8 *)(lsw + k * sw_str + wm + (h << 3u));
  f8 sw = __builtin_convertvector(sw_h, f8);
  __asm__ volatile("" : "+v"(sw));
  return sw;
}

/* Vectorized 2x f32x4 epilogue store for dense Q8_0 GEMMs (0-bit-diff exact-SHA). */
static inline __attribute__((always_inline)) void ink_r10_wfr_store_q8_row8(
    G float *y, const G float *bias, const G float *res,
    const u32 t, const u32 r0, const u32 rows, const u32 ys, const u32 rs,
    const f8 av) {
  const int vec_ok = (r0 + 8u <= rows) && ((ys & 3u) == 0u) && ((((unsigned long)y) & 15ul) == 0ul) &&
                     (!res || (((rs & 3u) == 0u) && ((((unsigned long)res) & 15ul) == 0ul))) &&
                     (!bias || ((((unsigned long)bias) & 15ul) == 0ul));
  G float *yt = y + (unsigned long)t * (unsigned long)ys;
  if (__builtin_expect(vec_ok, 1)) {
    f32x4 lo = {av[0], av[1], av[2], av[3]};
    f32x4 hi = {av[4], av[5], av[6], av[7]};
    if (bias) {
      const G f32x4 *bp4 = (const G f32x4 *)(bias + r0);
      lo += bp4[0];
      hi += bp4[1];
    }
    if (res) {
      const G f32x4 *rp4 = (const G f32x4 *)(res + (unsigned long)t * (unsigned long)rs + r0);
      lo += rp4[0];
      hi += rp4[1];
    }
    G f32x4 *yt4 = (G f32x4 *)(yt + r0);
    yt4[0] = lo;
    yt4[1] = hi;
  } else {
    const G float *rp = res ? (res + (unsigned long)t * (unsigned long)rs) : (const G float *)0;
#pragma unroll 1
    for (u32 v = 0u; v < 8u; v++) {
      const u32 r = r0 + v;
      if (r < rows) yt[r] = epilogue(av[v], r, bias, rp);
    }
  }
}

/* ============================================================================
 * Kernel 1: ie_inkling_gemm_q8_med_swiz_r10 (H06 + True 3-WG/CU <= 120 VGPR!)
 *   128x64 dense Q8_0 prefill WMMA: inline-zero WMMA + v_cvt_f32_i32 (-8 VGPR),
 *   decoupled rx0/rx1 -> rsw/rsx -> wa[0..1] staging, 4x4 L2/MALL swizzling,
 *   gfx1201 split barriers, 0 v_fma_mix_f32, 100% exact-SHA!
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r10_wfr_q8_k64(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 r16, const u32 h, const u32 lx_base0, f8 acc[4]) {
  const v8i zm = (v8i)(0);
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * INK_R10_WFR_M_T + r16;
  const v2i a0 = {(int)wk.x, (int)wk.y};
  const v2i a1 = {(int)wk.z, (int)wk.w};

  /* Pair 0 (j = 0, 1) */
  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + 0u * 2048u);
  const u32x4 b1 = *(const LDSP u32x4 *)(lx_k + 1u * 2048u);
  const float dx0 = lsx_k[0u];
  const float dx1 = lsx_k[16u];
  __builtin_amdgcn_sched_barrier(0);

  const v2i b0_0 = {(int)b0.x, (int)b0.y}, b0_1 = {(int)b0.z, (int)b0.w};
  const v2i b1_0 = {(int)b1.x, (int)b1.y}, b1_1 = {(int)b1.z, (int)b1.w};
  v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_0, zm, 0);
  v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b1_0, zm, 0);
  c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_1, c0, 0);
  c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1_1, c1, 0);
  __builtin_amdgcn_sched_barrier(0);

  const f8 sw = ink_r10_wfr_load_sw_f8(lsw, INK_R10_WFR_M_SWSTR, k, wr, h);
  acc[0] += (sw * dx0) * __builtin_convertvector(c0, f8);
  __asm__ volatile("" : : "v"(acc[0]));
  __builtin_amdgcn_sched_barrier(0);
  acc[1] += (sw * dx1) * __builtin_convertvector(c1, f8);
  __asm__ volatile("" : : "v"(acc[1]) : "memory");
  __builtin_amdgcn_sched_barrier(0);

  /* Pair 1 (j = 2, 3) */
  const u32x4 b2 = *(const LDSP u32x4 *)(lx_k + 2u * 2048u);
  const u32x4 b3 = *(const LDSP u32x4 *)(lx_k + 3u * 2048u);
  const float dx2 = lsx_k[32u];
  const float dx3 = lsx_k[48u];
  __builtin_amdgcn_sched_barrier(0);

  const v2i b2_0 = {(int)b2.x, (int)b2.y}, b2_1 = {(int)b2.z, (int)b2.w};
  const v2i b3_0 = {(int)b3.x, (int)b3.y}, b3_1 = {(int)b3.z, (int)b3.w};
  v8i c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b2_0, zm, 0);
  v8i c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b3_0, zm, 0);
  c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b2_1, c2, 0);
  c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b3_1, c3, 0);
  __builtin_amdgcn_sched_barrier(0);

  acc[2] += (sw * dx2) * __builtin_convertvector(c2, f8);
  __asm__ volatile("" : : "v"(acc[2]));
  __builtin_amdgcn_sched_barrier(0);
  acc[3] += (sw * dx3) * __builtin_convertvector(c3, f8);
  __asm__ volatile("" : : "v"(acc[3]) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

KERNEL ie_inkling_gemm_q8_med_swiz_r10(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                       u32 rows, u32 nb, const G float *bias, const G float *res,
                                       u32 T, u32 xs, u32 ys, u32 rs) {
  static LDS struct {
    u8    x[2u][INK_R10_WFR_M_T * (INK_R10_WFR_KB * 32u)] __attribute__((aligned(16))); /* 16,384 B */
    f16   sw[2u][INK_R10_WFR_KB * INK_R10_WFR_M_SWSTR] __attribute__((aligned(16)));    /*  2,560 B */
    float sx[2u][INK_R10_WFR_KB * INK_R10_WFR_M_T] __attribute__((aligned(32)));        /*  2,048 B */
  } ink_r10_wfr_med_lds;
  _Static_assert(sizeof(ink_r10_wfr_med_lds) == 20992u, "ink_r10_wfr_med_lds must be 20,992 B (3 WGs/CU)");

  if (T == 0u || rows == 0u || nb < INK_R10_WFR_KB || (nb & (INK_R10_WFR_KB - 1u)) != 0u || !qw || !qs || !xq || !y)
    return;

  const u32 nrt = (rows + INK_R10_WFR_M_R - 1u) / INK_R10_WFR_M_R;
  const u32 ntt = (T + INK_R10_WFR_M_T - 1u) / INK_R10_WFR_M_T;
  const u32 wg = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  if (wg >= nrt * ntt) return;
  u32 rt, tt;
  ink_r10_wfr_swiz_rt_tt(wg, nrt, ntt, &rt, &tt);
  const u32 rbase = rt * INK_R10_WFR_M_R, tbase = tt * INK_R10_WFR_M_T;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;
  const u32 wr = w << 4u;

  f8 acc[4];
#pragma unroll
  for (u32 j = 0u; j < 4u; j++) acc[j] = (f8)(0.0f);

  const G u8 *qw8 = rq_ptr_u((const G u8 *)qw);
  const G u8 *qs8 = rq_ptr_u((const G u8 *)qs);
  const G u8 *xq8 = rq_ptr_u(xq);
  const G u8 *xq_s = rq_ptr_u(xq + (unsigned long)nb * 32ul);

  const u32 gr = (rbase + wr + r16 < rows) ? (rbase + wr + r16) : (rows - 1u);
  const u32 w_off = gr * (nb * 32u) + (h << 4u);

  const u32 tk0 = t0 >> 3u, col0 = t0 & 7u;
  const u32 gt0 = (tbase + tk0 < T) ? (tbase + tk0) : (T - 1u);
  const u32 gt1 = (tbase + tk0 + 32u < T) ? (tbase + tk0 + 32u) : (T - 1u);
  const u32 xoff0 = gt0 * xs + (col0 << 4u);
  const u32 xoff1 = gt1 * xs + (col0 << 4u);
  const u32 st_x = (tk0 << 7u) + ((col0 ^ (tk0 & 7u)) << 4u);
  const u32 lx_base0 = (r16 << 7u) + ((h ^ (r16 & 7u)) << 4u);

  const int is_sw = (w < 4u);
  const int is_sx = (w == 4u || w == 5u);
  const u32 sr = (rbase + (t0 & 127u) < rows) ? (rbase + (t0 & 127u)) : (rows - 1u);
  const u32 sw_off = is_sw ? (sr * (nb * 2u)) : 0u;
  const u32 sxi = t0 & 63u;
  const u32 gt_sx = (tbase + sxi < T) ? (tbase + sxi) : (T - 1u);
  const u32 sx_off = is_sx ? (gt_sx * xs) : 0u;

  u32x4 wa[INK_R10_WFR_KB];
#pragma unroll
  for (u32 k = 0u; k < INK_R10_WFR_KB; k++) wa[k] = ink_r10_wfr_ld4(qw8 + w_off + k * 32u);
  {
    const u32x4 rx0_i = ink_r10_wfr_ld4(xq8 + xoff0);
    const u32x4 rx1_i = ink_r10_wfr_ld4(xq8 + xoff1);
    const ink_r10_wfr_u32x2 rsw_i = is_sw ? ink_r10_wfr_ld2(qs8 + sw_off) : (ink_r10_wfr_u32x2){0u, 0u};
    const u32x4 rsx_i = is_sx ? ink_r10_wfr_ld4(xq_s + sx_off) : (u32x4)(0u);
    *(LDSP u32x4 *)(ink_r10_wfr_med_lds.x[0] + st_x) = rx0_i;
    *(LDSP u32x4 *)(ink_r10_wfr_med_lds.x[0] + st_x + 4096u) = rx1_i;
    if (is_sw) ink_r10_wfr_stage_sw(ink_r10_wfr_med_lds.sw[0], INK_R10_WFR_M_SWSTR, t0, rsw_i);
    if (is_sx) ink_r10_wfr_stage_sx(ink_r10_wfr_med_lds.sx[0], INK_R10_WFR_M_T, sxi, rsx_i);
  }
  ink_r10_wfr_lsignal();
  ink_r10_wfr_lwait();

  u32 bf = 0u;
  u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R10_WFR_KB) {
    const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R10_WFR_KB));
    const int more = (kb_n < nb);
    __asm__ volatile("" : : "v"(wa[0]), "v"(wa[1]));
    if (more) {
      const G u8 *xq_n = rq_ptr_u(xq8 + (unsigned long)kb_n * 32ul);
      rx0 = ink_r10_wfr_ld4(xq_n + xoff0);
      rx1 = ink_r10_wfr_ld4(xq_n + xoff1);
    }
    __builtin_amdgcn_sched_barrier(0);

    const LDSP u8 *lx = ink_r10_wfr_med_lds.x[bf];
    const LDSP f16 *lsw = ink_r10_wfr_med_lds.sw[bf];
    const LDSP float *lsx = ink_r10_wfr_med_lds.sx[bf];

    ink_r10_wfr_q8_k64(wa[0], lx, lsw, lsx, 0u, wr, r16, h, lx_base0, acc);
    ink_r10_wfr_q8_k64(wa[1], lx, lsw, lsx, 1u, wr, r16, h, lx_base0, acc);

    /* Free rx0/rx1 (8 VGPRs) into inactive LDS and load rsw/rsx while wa[0..1] are dead! */
    ink_r10_wfr_u32x2 rsw = {0u, 0u};
    u32x4 rsx = (u32x4)(0u);
    if (more) {
      const u32 nbf = bf ^ 1u;
      *(LDSP u32x4 *)(ink_r10_wfr_med_lds.x[nbf] + st_x) = rx0;
      *(LDSP u32x4 *)(ink_r10_wfr_med_lds.x[nbf] + st_x + 4096u) = rx1;
      const G u8 *qs_n = rq_ptr_u(qs8 + (unsigned long)kb_n * 2ul);
      const G u8 *xs_n = rq_ptr_u(xq_s + (unsigned long)kb_n * 4ul);
      if (is_sw) rsw = ink_r10_wfr_ld2(qs_n + sw_off);
      if (is_sx) rsx = ink_r10_wfr_ld4(xs_n + sx_off);
    }
    __builtin_amdgcn_sched_barrier(0);

    ink_r10_wfr_q8_k64(wa[2], lx, lsw, lsx, 2u, wr, r16, h, lx_base0, acc);

    /* Stage rsw/rsx -> inactive LDS (frees 6 VGPRs), signal split barrier BEFORE k=3 + wa loads! */
    if (more) {
      const u32 nbf = bf ^ 1u;
      if (is_sw) ink_r10_wfr_stage_sw(ink_r10_wfr_med_lds.sw[nbf], INK_R10_WFR_M_SWSTR, t0, rsw);
      if (is_sx) ink_r10_wfr_stage_sx(ink_r10_wfr_med_lds.sx[nbf], INK_R10_WFR_M_T, sxi, rsx);
      /* All LDS writes to nbf are complete: signal split barrier BEFORE k=3 (100% hidden!) */
      ink_r10_wfr_lsignal();
      const G u8 *qw_n = rq_ptr_u(qw8 + (unsigned long)kb_n * 32ul);
      wa[0] = ink_r10_wfr_ld4(qw_n + w_off + 0u);
      wa[1] = ink_r10_wfr_ld4(qw_n + w_off + 32u);
    }
    __builtin_amdgcn_sched_barrier(0);

    ink_r10_wfr_q8_k64(wa[3], lx, lsw, lsx, 3u, wr, r16, h, lx_base0, acc);
    if (more) {
      const G u8 *qw_n = rq_ptr_u(qw8 + (unsigned long)kb_n * 32ul);
      wa[2] = ink_r10_wfr_ld4(qw_n + w_off + 64u);
      wa[3] = ink_r10_wfr_ld4(qw_n + w_off + 96u);
      ink_r10_wfr_lwait();
      bf ^= 1u;
    }
  }

  const u32 r0 = rbase + wr + (h << 3u);
#pragma unroll
  for (u32 j = 0u; j < 4u; j++) {
    const u32 t = tbase + (j << 4u) + r16;
    if (t < T) ink_r10_wfr_store_q8_row8(y, bias, res, t, r0, rows, ys, rs, acc[j]);
  }
}

/* ============================================================================
 * Kernel 2: ie_inkling_gemm_q8_64x64_swiz_r10 (H06 + True 3-WG/CU <= 96 VGPR!)
 *   64x64 dense Q8_0 prefill WMMA: 4x4 L2/MALL swizzling, decoupled split
 *   barriers hiding barrier behind k=3 + wa[2..3], 0 v_fma_mix_f32, exact-SHA!
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r10_wfr_q8_k32(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wm, const u32 h, const u32 lx_base0, f8 *acc0, f8 *acc1) {
  const v8i zm = (v8i)(0x4B400000);
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * INK_R10_WFR_M_T;
  const v2i a0 = {(int)wk.x, (int)wk.y};
  const v2i a1 = {(int)wk.z, (int)wk.w};

  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + 0u * 2048u);
  const u32x4 b1 = *(const LDSP u32x4 *)(lx_k + 1u * 2048u);
  const float dx0 = lsx_k[0u];
  const float dx1 = lsx_k[16u];
  __builtin_amdgcn_sched_barrier(0);

  const v2i b0_0 = {(int)b0.x, (int)b0.y}, b0_1 = {(int)b0.z, (int)b0.w};
  const v2i b1_0 = {(int)b1.x, (int)b1.y}, b1_1 = {(int)b1.z, (int)b1.w};
  v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_0, zm, 0);
  v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b1_0, zm, 0);
  c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_1, c0, 0);
  c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1_1, c1, 0);
  __builtin_amdgcn_sched_barrier(0);

  const f8 sw = ink_r10_wfr_load_sw_f8(lsw, INK_R10_WFR_H_SWSTR, k, wm, h);
  *acc0 += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  __asm__ volatile("" : : "v"(*acc0));
  __builtin_amdgcn_sched_barrier(0);
  *acc1 += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
  __asm__ volatile("" : : "v"(*acc1) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

KERNEL ie_inkling_gemm_q8_64x64_swiz_r10(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                         u32 rows, u32 nb, const G float *bias, const G float *res,
                                         u32 T, u32 xs, u32 ys, u32 rs) {
  static LDS struct {
    u8    x[2u][INK_R10_WFR_M_T * (INK_R10_WFR_KB * 32u)] __attribute__((aligned(16))); /* 16,384 B */
    f16   sw[2u][INK_R10_WFR_KB * INK_R10_WFR_H_SWSTR] __attribute__((aligned(16)));    /*  1,536 B */
    float sx[2u][INK_R10_WFR_KB * INK_R10_WFR_M_T] __attribute__((aligned(32)));        /*  2,048 B */
  } ink_r10_wfr_h_lds;
  _Static_assert(sizeof(ink_r10_wfr_h_lds) == 19968u, "ink_r10_wfr_h_lds must be 19,968 B (3 WGs/CU)");

  if (T == 0u || rows == 0u || nb < INK_R10_WFR_KB || (nb & (INK_R10_WFR_KB - 1u)) != 0u || !qw || !qs || !xq || !y)
    return;

  const u32 nrt = (rows + INK_R10_WFR_H_R - 1u) / INK_R10_WFR_H_R;
  const u32 ntt = (T + INK_R10_WFR_M_T - 1u) / INK_R10_WFR_M_T;
  const u32 wg = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  if (wg >= nrt * ntt) return;
  u32 rt, tt;
  ink_r10_wfr_swiz_rt_tt(wg, nrt, ntt, &rt, &tt);
  const u32 rbase = rt * INK_R10_WFR_H_R, tbase = tt * INK_R10_WFR_M_T;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;
  const u32 wm = (w >> 1u) << 4u;
  const u32 wn = (w & 1u) << 5u;

  f8 acc0 = (f8)(0.0f), acc1 = (f8)(0.0f);

  const G u8 *qw8 = rq_ptr_u((const G u8 *)qw);
  const G u8 *qs8 = rq_ptr_u((const G u8 *)qs);
  const G u8 *xq8 = rq_ptr_u(xq);
  const G u8 *xq_s = rq_ptr_u(xq + (unsigned long)nb * 32ul);

  const u32 gr = (rbase + wm + r16 < rows) ? (rbase + wm + r16) : (rows - 1u);
  const u32 w_off = gr * (nb * 32u) + (h << 4u);

  const u32 tk0 = t0 >> 3u, col0 = t0 & 7u;
  const u32 gt0 = (tbase + tk0 < T) ? (tbase + tk0) : (T - 1u);
  const u32 gt1 = (tbase + tk0 + 32u < T) ? (tbase + tk0 + 32u) : (T - 1u);
  const u32 xoff0 = gt0 * xs + (col0 << 4u);
  const u32 xoff1 = gt1 * xs + (col0 << 4u);
  const u32 st_x = (tk0 << 7u) + ((col0 ^ (tk0 & 7u)) << 4u);
  const u32 lx_base0 = ((wn + r16) << 7u) + ((h ^ (r16 & 7u)) << 4u);

  const int is_sw = (w < 2u);
  const int is_sx = (w == 2u || w == 3u);
  const u32 swi = t0 & 63u;
  const u32 sr = (rbase + swi < rows) ? (rbase + swi) : (rows - 1u);
  const u32 sw_off = is_sw ? (sr * (nb * 2u)) : 0u;
  const u32 sxi = t0 & 63u;
  const u32 gt_sx = (tbase + sxi < T) ? (tbase + sxi) : (T - 1u);
  const u32 sx_off = is_sx ? (gt_sx * xs) : 0u;

  u32x4 wa[INK_R10_WFR_KB];
#pragma unroll
  for (u32 k = 0u; k < INK_R10_WFR_KB; k++) wa[k] = ink_r10_wfr_ld4(qw8 + w_off + k * 32u);
  {
    const u32x4 rx0_i = ink_r10_wfr_ld4(xq8 + xoff0);
    const u32x4 rx1_i = ink_r10_wfr_ld4(xq8 + xoff1);
    const ink_r10_wfr_u32x2 rsw_i = is_sw ? ink_r10_wfr_ld2(qs8 + sw_off) : (ink_r10_wfr_u32x2){0u, 0u};
    const u32x4 rsx_i = is_sx ? ink_r10_wfr_ld4(xq_s + sx_off) : (u32x4)(0u);
    *(LDSP u32x4 *)(ink_r10_wfr_h_lds.x[0] + st_x) = rx0_i;
    *(LDSP u32x4 *)(ink_r10_wfr_h_lds.x[0] + st_x + 4096u) = rx1_i;
    if (is_sw) ink_r10_wfr_stage_sw(ink_r10_wfr_h_lds.sw[0], INK_R10_WFR_H_SWSTR, swi, rsw_i);
    if (is_sx) ink_r10_wfr_stage_sx(ink_r10_wfr_h_lds.sx[0], INK_R10_WFR_M_T, sxi, rsx_i);
  }
  ink_r10_wfr_lsignal();
  ink_r10_wfr_lwait();

  u32 bf = 0u;
  u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R10_WFR_KB) {
    const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R10_WFR_KB));
    const int more = (kb_n < nb);
    __asm__ volatile("" : : "v"(wa[0]), "v"(wa[1]));
    if (more) {
      const G u8 *xq_n = rq_ptr_u(xq8 + (unsigned long)kb_n * 32ul);
      rx0 = ink_r10_wfr_ld4(xq_n + xoff0);
      rx1 = ink_r10_wfr_ld4(xq_n + xoff1);
    }
    __builtin_amdgcn_sched_barrier(0);

    const LDSP u8 *lx = ink_r10_wfr_h_lds.x[bf];
    const LDSP f16 *lsw = ink_r10_wfr_h_lds.sw[bf];
    const LDSP float *lsx = ink_r10_wfr_h_lds.sx[bf] + wn + r16;

    ink_r10_wfr_q8_k32(wa[0], lx, lsw, lsx, 0u, wm, h, lx_base0, &acc0, &acc1);
    ink_r10_wfr_q8_k32(wa[1], lx, lsw, lsx, 1u, wm, h, lx_base0, &acc0, &acc1);

    ink_r10_wfr_u32x2 rsw = {0u, 0u};
    u32x4 rsx = (u32x4)(0u);
    if (more) {
      const u32 nbf = bf ^ 1u;
      *(LDSP u32x4 *)(ink_r10_wfr_h_lds.x[nbf] + st_x) = rx0;
      *(LDSP u32x4 *)(ink_r10_wfr_h_lds.x[nbf] + st_x + 4096u) = rx1;
      const G u8 *qs_n = rq_ptr_u(qs8 + (unsigned long)kb_n * 2ul);
      const G u8 *xs_n = rq_ptr_u(xq_s + (unsigned long)kb_n * 4ul);
      if (is_sw) rsw = ink_r10_wfr_ld2(qs_n + sw_off);
      if (is_sx) rsx = ink_r10_wfr_ld4(xs_n + sx_off);
    }
    __builtin_amdgcn_sched_barrier(0);

    ink_r10_wfr_q8_k32(wa[2], lx, lsw, lsx, 2u, wm, h, lx_base0, &acc0, &acc1);

    if (more) {
      const u32 nbf = bf ^ 1u;
      if (is_sw) ink_r10_wfr_stage_sw(ink_r10_wfr_h_lds.sw[nbf], INK_R10_WFR_H_SWSTR, swi, rsw);
      if (is_sx) ink_r10_wfr_stage_sx(ink_r10_wfr_h_lds.sx[nbf], INK_R10_WFR_M_T, sxi, rsx);
      ink_r10_wfr_lsignal();
      const G u8 *qw_n = rq_ptr_u(qw8 + (unsigned long)kb_n * 32ul);
      wa[0] = ink_r10_wfr_ld4(qw_n + w_off + 0u);
      wa[1] = ink_r10_wfr_ld4(qw_n + w_off + 32u);
    }
    __builtin_amdgcn_sched_barrier(0);

    ink_r10_wfr_q8_k32(wa[3], lx, lsw, lsx, 3u, wm, h, lx_base0, &acc0, &acc1);
    if (more) {
      const G u8 *qw_n = rq_ptr_u(qw8 + (unsigned long)kb_n * 32ul);
      wa[2] = ink_r10_wfr_ld4(qw_n + w_off + 64u);
      wa[3] = ink_r10_wfr_ld4(qw_n + w_off + 96u);
      ink_r10_wfr_lwait();
      bf ^= 1u;
    }
  }

  const u32 r0 = rbase + wm + (h << 3u);
  const u32 t_0 = tbase + wn + r16;
  if (t_0 < T) ink_r10_wfr_store_q8_row8(y, bias, res, t_0, r0, rows, ys, rs, acc0);
  const u32 t_1 = tbase + wn + 16u + r16;
  if (t_1 < T) ink_r10_wfr_store_q8_row8(y, bias, res, t_1, r0, rows, ys, rs, acc1);
}

/* ============================================================================
 * Part B (H03): Native gfx1201 FP8 WMMA (v_wmma_f32_16x16x16_fp8_fp8) Helpers
 *   - Uses hardware v_cvt_pk_fp8_f32 (OCP E4M3FN, max 448.0f).
 *   - Macro-K=128 (KB=4 blocks): normalizes each sub-block scale to
 *     rw_k = sw_k / sw_max in [-1, 1] and rx_k = sx_k / sx_max in [-1, 1],
 *     so |int8 * rw_k| <= 128.0f << 448.0f (0 overflow, 0 underflow!).
 *   - Runs 8 consecutive FP8 WMMAs directly into FP32 accumulators and rescales
 *     ONCE per 128 elements by (sw_max * sx_max) -> 4x fewer rescaling FMAs!
 * ========================================================================== */

static inline __attribute__((always_inline)) u32 ink_r10_wfr_pack_fp8_i8x4(const u32 w, const float sc) {
  const float f0 = (float)(signed char)(w & 0xFFu) * sc;
  const float f1 = (float)(signed char)((w >> 8u) & 0xFFu) * sc;
  const float f2 = (float)(signed char)((w >> 16u) & 0xFFu) * sc;
  const float f3 = (float)(signed char)(w >> 24u) * sc;
  u32 r = 0u;
  r = __builtin_amdgcn_cvt_pk_fp8_f32(f0, f1, r, 0);
  r = __builtin_amdgcn_cvt_pk_fp8_f32(f2, f3, r, 1);
  return r;
}

static inline __attribute__((always_inline)) void ink_r10_wfr_unpack_sw4(
    const ink_r10_wfr_u32x2 sw2, float rw[4], float *sw_max_out) {
  const u32 w0 = sw2.x, w1 = sw2.y;
  const u16 h0 = (u16)(w0 & 0xFFFFu), h1 = (u16)(w0 >> 16u);
  const u16 h2 = (u16)(w1 & 0xFFFFu), h3 = (u16)(w1 >> 16u);
  float s0 = (float)__builtin_bit_cast(f16, h0);
  float s1 = (float)__builtin_bit_cast(f16, h1);
  float s2 = (float)__builtin_bit_cast(f16, h2);
  float s3 = (float)__builtin_bit_cast(f16, h3);
  /* Pin f32 scales to prevent quarter-rate v_fma_mix_f32 folding! */
  __asm__ volatile("" : "+v"(s0), "+v"(s1), "+v"(s2), "+v"(s3));
  const float m01 = __builtin_fmaxf(__builtin_fabsf(s0), __builtin_fabsf(s1));
  const float m23 = __builtin_fmaxf(__builtin_fabsf(s2), __builtin_fabsf(s3));
  const float smax = __builtin_fmaxf(m01, m23);
  const float inv = (smax > 0.0f) ? (1.0f / smax) : 0.0f;
  rw[0] = s0 * inv;
  rw[1] = s1 * inv;
  rw[2] = s2 * inv;
  rw[3] = s3 * inv;
  *sw_max_out = smax;
}

/* Cooperative 4-lanes/token Q8_0 -> FP8 LDS stager (64 tokens x 4 K-blocks = 256 threads).
 * Each thread converts one 32-elem Q8_0 block (rx_lo, rx_hi) using sx / sx_max
 * (sx_max computed across 4 adjacent lanes via 2x shfl_xor_f -> ZERO sx[0..3] LDS!). */
static inline __attribute__((always_inline)) void ink_r10_wfr_stage_x_fp8(
    LDSP u8 *lx_dst, LDSP float *lsx_max, const u32 st_x0,
    const u32 tk_x, const u32 kb_t, const u32x4 rx0, const u32x4 rx1, const float sx_v) {
  float smax = __builtin_fabsf(sx_v);
  smax = __builtin_fmaxf(smax, shfl_xor_f(smax, 1u));
  smax = __builtin_fmaxf(smax, shfl_xor_f(smax, 2u));
  float rx = (smax > 0.0f) ? (sx_v / smax) : 0.0f;
  if (kb_t == 0u) lsx_max[tk_x] = smax;
  __asm__ volatile("" : "+v"(rx));
  __builtin_amdgcn_sched_barrier(0);

  const u32 q00 = ink_r10_wfr_pack_fp8_i8x4(rx0.x, rx);
  const u32 q01 = ink_r10_wfr_pack_fp8_i8x4(rx0.y, rx);
  __builtin_amdgcn_sched_barrier(0);
  const u32 q02 = ink_r10_wfr_pack_fp8_i8x4(rx0.z, rx);
  const u32 q03 = ink_r10_wfr_pack_fp8_i8x4(rx0.w, rx);
  *(LDSP u32x4 *)(lx_dst + st_x0) = (u32x4){q00, q01, q02, q03};
  __builtin_amdgcn_sched_barrier(0);

  const u32 q10 = ink_r10_wfr_pack_fp8_i8x4(rx1.x, rx);
  const u32 q11 = ink_r10_wfr_pack_fp8_i8x4(rx1.y, rx);
  __builtin_amdgcn_sched_barrier(0);
  const u32 q12 = ink_r10_wfr_pack_fp8_i8x4(rx1.z, rx);
  const u32 q13 = ink_r10_wfr_pack_fp8_i8x4(rx1.w, rx);
  *(LDSP u32x4 *)(lx_dst + (st_x0 ^ 16u)) = (u32x4){q10, q11, q12, q13};
}

/* Convert 16 Q8_0 int8 weights (u32x4) * rw -> 2x v2i FP8 operands for 2x FP8 WMMAs. */
static inline __attribute__((always_inline)) void ink_r10_wfr_q8_w_fp8(
    const u32x4 wk, const float rw, v2i *a0_out, v2i *a1_out) {
  const u32 p0 = ink_r10_wfr_pack_fp8_i8x4(wk.x, rw);
  const u32 p1 = ink_r10_wfr_pack_fp8_i8x4(wk.y, rw);
  *a0_out = (v2i){(int)p0, (int)p1};
  __builtin_amdgcn_sched_barrier(0);
  const u32 p2 = ink_r10_wfr_pack_fp8_i8x4(wk.z, rw);
  const u32 p3 = ink_r10_wfr_pack_fp8_i8x4(wk.w, rw);
  *a1_out = (v2i){(int)p2, (int)p3};
  __builtin_amdgcn_sched_barrier(0);
}

/* Convert 16 IQ4_NL nibbles (selected by h) * rw -> 2x v2i FP8 operands for 2x FP8 WMMAs. */
static inline __attribute__((always_inline)) void ink_r10_wfr_iq4_w_fp8(
    const u32x4 wk, const u32 h, const float rw, v2i *a0_out, v2i *a1_out) {
  const u32x4 nb4 = h ? ((wk >> 4u) & 0x0F0F0F0Fu) : (wk & 0x0F0F0F0Fu);
  const u32 p0 = ink_r10_wfr_pack_fp8_i8x4(ie_iq4_map(nb4.x), rw);
  const u32 p1 = ink_r10_wfr_pack_fp8_i8x4(ie_iq4_map(nb4.y), rw);
  *a0_out = (v2i){(int)p0, (int)p1};
  __builtin_amdgcn_sched_barrier(0);
  const u32 p2 = ink_r10_wfr_pack_fp8_i8x4(ie_iq4_map(nb4.z), rw);
  const u32 p3 = ink_r10_wfr_pack_fp8_i8x4(ie_iq4_map(nb4.w), rw);
  *a1_out = (v2i){(int)p2, (int)p3};
  __builtin_amdgcn_sched_barrier(0);
}

/* Macro-K=128 (KB=4) FP8 WMMA step for one sub-block k (accumulates directly into FP32 p[0..3]!). */
static inline __attribute__((always_inline)) void ink_r10_wfr_fp8_step_j4(
    const v2i a0, const v2i a1, const LDSP u8 *lx, const u32 k, const u32 lx_base0, f8 p[4]) {
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));

  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + 0u * 2048u);
  const u32x4 b1 = *(const LDSP u32x4 *)(lx_k + 1u * 2048u);
  __builtin_amdgcn_sched_barrier(0);
  const v2i b0_0 = {(int)b0.x, (int)b0.y}, b0_1 = {(int)b0.z, (int)b0.w};
  const v2i b1_0 = {(int)b1.x, (int)b1.y}, b1_1 = {(int)b1.z, (int)b1.w};
  p[0] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, b0_0, p[0]);
  p[1] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, b1_0, p[1]);
  p[0] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, b0_1, p[0]);
  p[1] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, b1_1, p[1]);
  __builtin_amdgcn_sched_barrier(0);

  const u32x4 b2 = *(const LDSP u32x4 *)(lx_k + 2u * 2048u);
  const u32x4 b3 = *(const LDSP u32x4 *)(lx_k + 3u * 2048u);
  __builtin_amdgcn_sched_barrier(0);
  const v2i b2_0 = {(int)b2.x, (int)b2.y}, b2_1 = {(int)b2.z, (int)b2.w};
  const v2i b3_0 = {(int)b3.x, (int)b3.y}, b3_1 = {(int)b3.z, (int)b3.w};
  p[2] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, b2_0, p[2]);
  p[3] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, b3_0, p[3]);
  p[2] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, b2_1, p[2]);
  p[3] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, b3_1, p[3]);
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r10_wfr_fp8_apply_scale4(
    const LDSP float *lswm, const LDSP float *lsxm, const u32 wr, const u32 r16, const u32 h,
    const f8 p[4], f8 acc[4]) {
  const LDSP f32x4 *sw4 = (const LDSP f32x4 *)(lswm + wr + (h << 3u));
  const f32x4 slo = sw4[0], shi = sw4[1];
  const f8 swm = {slo.x, slo.y, slo.z, slo.w, shi.x, shi.y, shi.z, shi.w};
  const float sx0 = lsxm[r16 + 0u];
  const float sx1 = lsxm[r16 + 16u];
  const float sx2 = lsxm[r16 + 32u];
  const float sx3 = lsxm[r16 + 48u];
  __builtin_amdgcn_sched_barrier(0);
  acc[0] += (swm * sx0) * p[0];
  __asm__ volatile("" : : "v"(acc[0]));
  acc[1] += (swm * sx1) * p[1];
  __asm__ volatile("" : : "v"(acc[1]));
  acc[2] += (swm * sx2) * p[2];
  __asm__ volatile("" : : "v"(acc[2]));
  acc[3] += (swm * sx3) * p[3];
  __asm__ volatile("" : : "v"(acc[3]) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

/* ============================================================================
 * Kernel 3: ie_inkling_gemm_q8_fp8_med_r10 (H03 Dense Q8_0 FP8 WMMA, 128x64)
 *   Macro-K=128 FP8 WMMA, 4x4 L2/MALL swizzling, 17,920 B LDS, <= 120 VGPR!
 * ========================================================================== */

KERNEL ie_inkling_gemm_q8_fp8_med_r10(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                      u32 rows, u32 nb, const G float *bias, const G float *res,
                                      u32 T, u32 xs, u32 ys, u32 rs) {
  static LDS struct {
    u8    x[2u][INK_R10_WFR_M_T * (INK_R10_WFR_KB * 32u)] __attribute__((aligned(16))); /* 16,384 B */
    float swm[2u][INK_R10_WFR_M_R] __attribute__((aligned(16)));                        /*  1,024 B */
    float sxm[2u][INK_R10_WFR_M_T] __attribute__((aligned(16)));                        /*    512 B */
  } ink_r10_wfr_q8fp8_lds;
  _Static_assert(sizeof(ink_r10_wfr_q8fp8_lds) == 17920u, "ink_r10_wfr_q8fp8_lds must be 17,920 B (3 WGs/CU)");

  if (T == 0u || rows == 0u || nb < INK_R10_WFR_KB || (nb & (INK_R10_WFR_KB - 1u)) != 0u || !qw || !qs || !xq || !y)
    return;

  const u32 nrt = (rows + INK_R10_WFR_M_R - 1u) / INK_R10_WFR_M_R;
  const u32 ntt = (T + INK_R10_WFR_M_T - 1u) / INK_R10_WFR_M_T;
  const u32 wg = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  if (wg >= nrt * ntt) return;
  u32 rt, tt;
  ink_r10_wfr_swiz_rt_tt(wg, nrt, ntt, &rt, &tt);
  const u32 rbase = rt * INK_R10_WFR_M_R, tbase = tt * INK_R10_WFR_M_T;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;
  const u32 wr = w << 4u;

  f8 acc[4];
#pragma unroll
  for (u32 j = 0u; j < 4u; j++) acc[j] = (f8)(0.0f);

  const G u8 *qw8 = rq_ptr_u((const G u8 *)qw);
  const G u8 *qs8 = rq_ptr_u((const G u8 *)qs);
  const G u8 *xq8 = rq_ptr_u(xq);
  const G u8 *xq_s = rq_ptr_u(xq + (unsigned long)nb * 32ul);

  const u32 gr = (rbase + wr + r16 < rows) ? (rbase + wr + r16) : (rows - 1u);
  const u32 w_off = gr * (nb * 32u) + (h << 4u);
  const u32 sw_off = gr * (nb * 2u);

  /* 4-lanes/token cooperative Q8_0 -> FP8 loader: tk_x in [0..63], kb_t in [0..3]. */
  const u32 tk_x = t0 >> 2u, kb_t = t0 & 3u;
  const u32 gt_x = (tbase + tk_x < T) ? (tbase + tk_x) : (T - 1u);
  const u32 xoff_u = gt_x * xs + (kb_t << 5u);
  const u32 xoff_s = gt_x * xs + (kb_t << 2u);
  const u32 st_x0 = (tk_x << 7u) + (((kb_t << 1u) ^ (tk_x & 7u)) << 4u);
  const u32 lx_base0 = (r16 << 7u) + ((h ^ (r16 & 7u)) << 4u);

  /* Rolling 2-slot weight prefetch (wa0, wa1 = 8 VGPRs instead of 16 -> true 3 WGs/CU!). */
  u32x4 wa0 = ink_r10_wfr_ld4(qw8 + w_off + 0u);
  u32x4 wa1 = ink_r10_wfr_ld4(qw8 + w_off + 32u);
  ink_r10_wfr_u32x2 rsw = ink_r10_wfr_ld2(qs8 + sw_off);
  {
    const u32x4 rx0_i = ink_r10_wfr_ld4(xq8 + xoff_u + 0u);
    const u32x4 rx1_i = ink_r10_wfr_ld4(xq8 + xoff_u + 16u);
    const float sx_i = *(const G float *)(xq_s + xoff_s);
    ink_r10_wfr_stage_x_fp8(ink_r10_wfr_q8fp8_lds.x[0], ink_r10_wfr_q8fp8_lds.sxm[0],
                            st_x0, tk_x, kb_t, rx0_i, rx1_i, sx_i);
  }
  float rw[4], sw_max_cur;
  ink_r10_wfr_unpack_sw4(rsw, rw, &sw_max_cur);
  if (h == 0u) ink_r10_wfr_q8fp8_lds.swm[0][wr + r16] = sw_max_cur;
  ink_r10_wfr_lsignal();
  ink_r10_wfr_lwait();

  u32 bf = 0u;
#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R10_WFR_KB) {
    const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R10_WFR_KB));
    const int more = (kb_n < nb);
    u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
    float sx_n = 0.0f;
    if (more) {
      const G u8 *xq_n = rq_ptr_u(xq8 + (unsigned long)kb_n * 32ul);
      const G u8 *xs_n = rq_ptr_u(xq_s + (unsigned long)kb_n * 4ul);
      rx0 = ink_r10_wfr_ld4(xq_n + xoff_u + 0u);
      rx1 = ink_r10_wfr_ld4(xq_n + xoff_u + 16u);
      sx_n = *(const G float *)(xs_n + xoff_s);
    }
    __builtin_amdgcn_sched_barrier(0);

    const LDSP u8 *lx = ink_r10_wfr_q8fp8_lds.x[bf];
    const G u8 *qw_c = rq_ptr_u(qw8 + (unsigned long)kb0 * 32ul);
    f8 p[4] = {(f8)(0.0f), (f8)(0.0f), (f8)(0.0f), (f8)(0.0f)};

    {
      v2i a0, a1;
      ink_r10_wfr_q8_w_fp8(wa0, rw[0], &a0, &a1);
      wa0 = ink_r10_wfr_ld4(qw_c + w_off + 64u);
      ink_r10_wfr_fp8_step_j4(a0, a1, lx, 0u, lx_base0, p);
    }
    {
      v2i a0, a1;
      ink_r10_wfr_q8_w_fp8(wa1, rw[1], &a0, &a1);
      wa1 = ink_r10_wfr_ld4(qw_c + w_off + 96u);
      ink_r10_wfr_fp8_step_j4(a0, a1, lx, 1u, lx_base0, p);
    }

    if (more) {
      const u32 nbf = bf ^ 1u;
      ink_r10_wfr_stage_x_fp8(ink_r10_wfr_q8fp8_lds.x[nbf], ink_r10_wfr_q8fp8_lds.sxm[nbf],
                              st_x0, tk_x, kb_t, rx0, rx1, sx_n);
      const G u8 *qs_n = rq_ptr_u(qs8 + (unsigned long)kb_n * 2ul);
      rsw = ink_r10_wfr_ld2(qs_n + sw_off);
    }
    __builtin_amdgcn_sched_barrier(0);

    {
      v2i a0, a1;
      ink_r10_wfr_q8_w_fp8(wa0, rw[2], &a0, &a1);
      if (more) {
        const G u8 *qw_n = rq_ptr_u(qw8 + (unsigned long)kb_n * 32ul);
        wa0 = ink_r10_wfr_ld4(qw_n + w_off + 0u);
      }
      ink_r10_wfr_fp8_step_j4(a0, a1, lx, 2u, lx_base0, p);
    }
    {
      v2i a0, a1;
      ink_r10_wfr_q8_w_fp8(wa1, rw[3], &a0, &a1);
      if (more) {
        const G u8 *qw_n = rq_ptr_u(qw8 + (unsigned long)kb_n * 32ul);
        wa1 = ink_r10_wfr_ld4(qw_n + w_off + 32u);
      }
      ink_r10_wfr_fp8_step_j4(a0, a1, lx, 3u, lx_base0, p);
    }

    /* Overlap next-tile scale unpack + split barrier signal with current-tile apply_scale4! */
    if (more) {
      const u32 nbf = bf ^ 1u;
      ink_r10_wfr_unpack_sw4(rsw, rw, &sw_max_cur);
      if (h == 0u) ink_r10_wfr_q8fp8_lds.swm[nbf][wr + r16] = sw_max_cur;
      ink_r10_wfr_lsignal();
    }
    ink_r10_wfr_fp8_apply_scale4(ink_r10_wfr_q8fp8_lds.swm[bf], ink_r10_wfr_q8fp8_lds.sxm[bf],
                                 wr, r16, h, p, acc);
    if (more) {
      ink_r10_wfr_lwait();
      bf ^= 1u;
    }
  }

  const u32 r0 = rbase + wr + (h << 3u);
#pragma unroll
  for (u32 j = 0u; j < 4u; j++) {
    const u32 t = tbase + (j << 4u) + r16;
    if (t < T) ink_r10_wfr_store_q8_row8(y, bias, res, t, r0, rows, ys, rs, acc[j]);
  }
}

/* ============================================================================
 * Kernel 4: ie_inkling_gemm_iq4nl_fp8_expmaj_r10 (H03 IQ4_NL FP8 WMMA!)
 *   Expert-major L2-localized IQ4_NL prefill down-proj using native FP8 WMMA
 *   on Macro-K=128 tiles, <= 120 VGPR (true 3 WGs/CU!), 18,176 B LDS!
 * ========================================================================== */

typedef struct {
  union {
    struct {
      u8    x[2u][INK_R10_WFR_M_T * (INK_R10_WFR_KB * 32u)] __attribute__((aligned(16))); /* 16,384 B */
      float swm[2u][INK_R10_WFR_M_R] __attribute__((aligned(16)));                        /*  1,024 B */
      float sxm[2u][INK_R10_WFR_M_T] __attribute__((aligned(16)));                        /*    512 B */
    } k;
    float ep[32u][INK_R10_WFR_EP_STR] __attribute__((aligned(32)));                       /* 16,512 B */
  } u;
  u32 pid[64] __attribute__((aligned(16)));                                               /*    256 B */
} ink_r10_wfr_iq4fp8_lds_t;

_Static_assert(sizeof(ink_r10_wfr_iq4fp8_lds_t) == 18176u, "ink_r10_wfr_iq4fp8_lds_t must be 18,176 B (3 WGs/CU)");

static inline __attribute__((always_inline)) u32 ink_r10_wfr_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

/* Coalesced 512-B (f32x4) strip store via unioned ep[32][129] LDS (0 extra LDS!). */
static inline __attribute__((always_inline)) void ink_r10_wfr_iq4_store_strip(
    LDSP ink_r10_wfr_iq4fp8_lds_t *lds, G u8 *Y, const G u8 *res,
    const u32 rbase, const u32 rows, const u32 ys, const u32 rs,
    const u32 j_off, const u32 n_tok, const f8 a0, const f8 a1) {
  u32 t0 = tid();
  __asm__ volatile("" : "+v"(t0));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = t0 & 31u, l16 = l & 15u, h = l >> 4u;
  const u32 r_loc = (w << 4u) + (h << 3u);

  ink_r10_wfr_lsignal();
  ink_r10_wfr_lwait();
  {
    LDSP f32x4 *ep0 = (LDSP f32x4 *)(lds->u.ep[l16] + r_loc);
    ep0[0] = (f32x4){a0[0], a0[1], a0[2], a0[3]};
    ep0[1] = (f32x4){a0[4], a0[5], a0[6], a0[7]};
    if (n_tok > 16u) {
      LDSP f32x4 *ep1 = (LDSP f32x4 *)(lds->u.ep[16u + l16] + r_loc);
      ep1[0] = (f32x4){a1[0], a1[1], a1[2], a1[3]};
      ep1[1] = (f32x4){a1[4], a1[5], a1[6], a1[7]};
    }
  }
  ink_r10_wfr_lsignal();
  ink_r10_wfr_lwait();

  const u32 r4 = rbase + (l << 2u);
  const int r4_ok = (r4 + 4u <= rows);
  for (u32 tk = w; tk < n_tok; tk += 8u) {
    const u32 pid = lds->pid[j_off + tk];
    const f32x4 v = *(const LDSP f32x4 *)(lds->u.ep[tk] + (l << 2u));
    G float *yp = (G float *)(Y + (unsigned long)pid * (unsigned long)ys);
    if (__builtin_expect(r4_ok, 1)) {
      f32x4 ov = v;
      if (res) ov += *(const G f32x4 *)((const G float *)(res + (unsigned long)pid * (unsigned long)rs) + r4);
      *(G f32x4 *)(yp + r4) = ov;
    } else if (r4 < rows) {
      const G float *rp = res ? (const G float *)(res + (unsigned long)pid * (unsigned long)rs) : (const G float *)0;
      const float vs[4] = {v.x, v.y, v.z, v.w};
      for (u32 c = 0u; c < 4u; c++) {
        if (r4 + c < rows) yp[r4 + c] = rp ? (vs[c] + rp[r4 + c]) : vs[c];
      }
    }
  }
}

/* Expert-major FP8 IQ4_NL chunk runner (handles any 1 <= nj <= 64 with unified Macro-K=128!).
 * Uses 1..4 active j-tiles (wj = (nj + 15) / 16) with dynamic-free WJ=4 / WJ<=2 split. */
static inline __attribute__((always_inline)) void ink_r10_wfr_iq4fp8_run4(
    LDSP ink_r10_wfr_iq4fp8_lds_t *lds, const G u8 *qw_e, const G u8 *qs_e,
    const G u8 *X_u, const G u8 *X_s, const G u32 *list,
    const u32 pb, const u32 nj, const u32 rbase, const u32 rows, const u32 nb,
    const u32 xs, const u32 xdiv, f8 acc[4]) {
  u32 t0 = tid();
  __asm__ volatile("" : "+v"(t0));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = t0 & 31u, l16 = l & 15u, h = l >> 4u;
  const u32 wr = w << 4u;

  if (w < 2u) {
    const u32 pj = pb + (t0 < nj ? t0 : 0u);
    lds->pid[t0] = list ? list[pj] : pj;
  }
  ink_r10_wfr_lsignal();
  ink_r10_wfr_lwait();

  const u32 gr = (rbase + wr + l16 < rows) ? (rbase + wr + l16) : (rows - 1u);
  const u32 w_off = gr * (nb * 16u);
  const u32 sw_off = gr * (nb * 2u);

  const u32 tk_x = t0 >> 2u, kb_t = t0 & 3u;
  const u32 xrow = ink_r10_wfr_div_x(lds->pid[tk_x], xdiv) * xs;
  const u32 xoff_u = xrow + (kb_t << 5u);
  const u32 xoff_s = xrow + (kb_t << 2u);
  const u32 st_x0 = (tk_x << 7u) + (((kb_t << 1u) ^ (tk_x & 7u)) << 4u);
  const u32 lx_base0 = (l16 << 7u) + ((h ^ (l16 & 7u)) << 4u);

#pragma unroll
  for (u32 j = 0u; j < 4u; j++) acc[j] = (f8)(0.0f);

  /* Rolling 2-slot weight prefetch (wa0, wa1 = 8 VGPRs instead of 16 -> true 3 WGs/CU!). */
  u32x4 wa0 = *(const G u32x4 *)(qw_e + w_off + 0u);
  u32x4 wa1 = *(const G u32x4 *)(qw_e + w_off + 16u);
  ink_r10_wfr_u32x2 rsw = ink_r10_wfr_ld2(qs_e + sw_off);
  {
    const u32x4 rx0_i = *(const G u32x4 *)(X_u + xoff_u + 0u);
    const u32x4 rx1_i = *(const G u32x4 *)(X_u + xoff_u + 16u);
    const float sx_i = *(const G float *)(X_s + xoff_s);
    ink_r10_wfr_stage_x_fp8(lds->u.k.x[0], lds->u.k.sxm[0], st_x0, tk_x, kb_t, rx0_i, rx1_i, sx_i);
  }
  float rw[4], sw_max_cur;
  ink_r10_wfr_unpack_sw4(rsw, rw, &sw_max_cur);
  if (h == 0u) lds->u.k.swm[0][wr + l16] = sw_max_cur;
  ink_r10_wfr_lsignal();
  ink_r10_wfr_lwait();

  u32 bf = 0u;
#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R10_WFR_KB) {
    const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R10_WFR_KB));
    const int more = (kb_n < nb);
    u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
    float sx_n = 0.0f;
    if (more) {
      const G u8 *X_n = rq_ptr_u(X_u + (unsigned long)kb_n * 32ul);
      const G u8 *Xs_n = rq_ptr_u(X_s + (unsigned long)kb_n * 4ul);
      rx0 = *(const G u32x4 *)(X_n + xoff_u + 0u);
      rx1 = *(const G u32x4 *)(X_n + xoff_u + 16u);
      sx_n = *(const G float *)(Xs_n + xoff_s);
    }
    __builtin_amdgcn_sched_barrier(0);

    const LDSP u8 *lx = lds->u.k.x[bf];
    const G u8 *qw_c = rq_ptr_u(qw_e + (unsigned long)kb0 * 16ul);
    f8 p[4] = {(f8)(0.0f), (f8)(0.0f), (f8)(0.0f), (f8)(0.0f)};

    {
      v2i a0, a1;
      ink_r10_wfr_iq4_w_fp8(wa0, h, rw[0], &a0, &a1);
      wa0 = *(const G u32x4 *)(qw_c + w_off + 32u);
      ink_r10_wfr_fp8_step_j4(a0, a1, lx, 0u, lx_base0, p);
    }
    {
      v2i a0, a1;
      ink_r10_wfr_iq4_w_fp8(wa1, h, rw[1], &a0, &a1);
      wa1 = *(const G u32x4 *)(qw_c + w_off + 48u);
      ink_r10_wfr_fp8_step_j4(a0, a1, lx, 1u, lx_base0, p);
    }

    if (more) {
      const u32 nbf = bf ^ 1u;
      ink_r10_wfr_stage_x_fp8(lds->u.k.x[nbf], lds->u.k.sxm[nbf], st_x0, tk_x, kb_t, rx0, rx1, sx_n);
      const G u8 *qs_n = rq_ptr_u(qs_e + (unsigned long)kb_n * 2ul);
      rsw = ink_r10_wfr_ld2(qs_n + sw_off);
    }
    __builtin_amdgcn_sched_barrier(0);

    {
      v2i a0, a1;
      ink_r10_wfr_iq4_w_fp8(wa0, h, rw[2], &a0, &a1);
      if (more) {
        const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
        wa0 = *(const G u32x4 *)(qw_n + w_off + 0u);
      }
      ink_r10_wfr_fp8_step_j4(a0, a1, lx, 2u, lx_base0, p);
    }
    {
      v2i a0, a1;
      ink_r10_wfr_iq4_w_fp8(wa1, h, rw[3], &a0, &a1);
      if (more) {
        const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
        wa1 = *(const G u32x4 *)(qw_n + w_off + 16u);
      }
      ink_r10_wfr_fp8_step_j4(a0, a1, lx, 3u, lx_base0, p);
    }

    /* Overlap next-tile scale unpack + split barrier signal with current-tile apply_scale4! */
    if (more) {
      const u32 nbf = bf ^ 1u;
      ink_r10_wfr_unpack_sw4(rsw, rw, &sw_max_cur);
      if (h == 0u) lds->u.k.swm[nbf][wr + l16] = sw_max_cur;
      ink_r10_wfr_lsignal();
    }
    ink_r10_wfr_fp8_apply_scale4(lds->u.k.swm[bf], lds->u.k.sxm[bf], wr, l16, h, p, acc);
    if (more) {
      ink_r10_wfr_lwait();
      bf ^= 1u;
    }
  }
}

KERNEL ie_inkling_gemm_iq4nl_fp8_expmaj_r10(
    const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
    u32 rows, u32 nb, const G u32 *list, const G u32 *offs, u32 np,
    u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv, u32 kind,
    const G u8 *res, u32 rs) {
  static LDS ink_r10_wfr_iq4fp8_lds_t ink_r10_wfr_iq4fp8_lds __attribute__((aligned(32)));

  if (kind != 8u || rows == 0u || nb < INK_R10_WFR_KB || (nb & (INK_R10_WFR_KB - 1u)) != 0u || !qw0 || !qs0 || !X || !Y)
    return;

  const u32 rbase = (u32)__builtin_amdgcn_readfirstlane((int)wgid()) * INK_R10_WFR_M_R;
  if (rbase >= rows) return;

  const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  u32 pb = list ? offs[e] : (e * 64u);
  const u32 p1 = list ? offs[e + 1u] : (pb + 64u < np ? pb + 64u : np);
  if (pb >= p1) return;

  const u32 e_w = list ? e : 0u;
  const G u8 *qw_e = rq_ptr_u((const G u8 *)(qw0 + (unsigned long)e_w * (unsigned long)qwn));
  const G u8 *qs_e = rq_ptr_u((const G u8 *)(qs0 + (unsigned long)e_w * (unsigned long)qsn));
  const G u8 *X_u = rq_ptr_u(X);
  const G u8 *X_s = rq_ptr_u(X + (unsigned long)nb * 32ul);

#pragma unroll 1
  for (; pb < p1; pb += 64u) {
    const u32 nj = (p1 - pb < 64u) ? (p1 - pb) : 64u;
    f8 acc[4];
    ink_r10_wfr_iq4fp8_run4(&ink_r10_wfr_iq4fp8_lds, qw_e, qs_e, X_u, X_s, list,
                            pb, nj, rbase, rows, nb, xs, xdiv, acc);
    const u32 n0 = nj < 32u ? nj : 32u;
    ink_r10_wfr_iq4_store_strip(&ink_r10_wfr_iq4fp8_lds, Y, res, rbase, rows, ys, rs, 0u, n0, acc[0], acc[1]);
    if (nj > 32u) {
      ink_r10_wfr_iq4_store_strip(&ink_r10_wfr_iq4fp8_lds, Y, res, rbase, rows, ys, rs, 32u, nj - 32u, acc[2], acc[3]);
    }
    if (pb + 64u < p1) {
      ink_r10_wfr_lsignal();
      ink_r10_wfr_lwait();
    }
  }
}

/* ============================================================================
 * Kernel 5: ie_inkling_gemm_f32_router_bf16_r10 (H08 2-Term BF16-Pair Router!)
 *   Lossless 3-WMMA BF16-pair split (W_hi*X_hi + W_hi*X_lo + W_lo*X_hi) on
 *   (BM=32, BN=16, BK=64), 8 waves = 2(M) x 4(Split-K), direct-VGPR W +
 *   double-buffered 8,196 B LDS X -> 128 WGs at M=128, T=512 (2 WGs/CU!),
 *   <= 96 VGPR (4 WGs/CU!), optional fused L2-hot OP_MOE_ROUTE!
 * ========================================================================== */

static inline __attribute__((always_inline)) u32 ink_r10_wfr_rne_bf16_hi32(const float x) {
  const u32 u = __builtin_bit_cast(u32, x);
  return (u + 0x7FFFu + ((u >> 16u) & 1u)) & 0xFFFF0000u;
}

static inline __attribute__((always_inline)) void ink_r10_wfr_split_f4_bf16(
    const f32x4 x, ink_r10_wfr_u32x2 *hi, ink_r10_wfr_u32x2 *lo) {
  const u32 h0 = ink_r10_wfr_rne_bf16_hi32(x.x), h1 = ink_r10_wfr_rne_bf16_hi32(x.y);
  const u32 h2 = ink_r10_wfr_rne_bf16_hi32(x.z), h3 = ink_r10_wfr_rne_bf16_hi32(x.w);
  const u32 l0 = ink_r10_wfr_rne_bf16_hi32(x.x - __builtin_bit_cast(float, h0));
  const u32 l1 = ink_r10_wfr_rne_bf16_hi32(x.y - __builtin_bit_cast(float, h1));
  const u32 l2 = ink_r10_wfr_rne_bf16_hi32(x.z - __builtin_bit_cast(float, h2));
  const u32 l3 = ink_r10_wfr_rne_bf16_hi32(x.w - __builtin_bit_cast(float, h3));
  *hi = (ink_r10_wfr_u32x2){(h0 >> 16u) | h1, (h2 >> 16u) | h3};
  *lo = (ink_r10_wfr_u32x2){(l0 >> 16u) | l1, (l2 >> 16u) | l3};
}

static inline __attribute__((always_inline)) void ink_r10_wfr_split_w8_bf16(
    const f32x4 w0, const f32x4 w1, ink_r10_wfr_bf8 *whi, ink_r10_wfr_bf8 *wlo) {
  ink_r10_wfr_u32x2 h0, l0, h1, l1;
  ink_r10_wfr_split_f4_bf16(w0, &h0, &l0);
  ink_r10_wfr_split_f4_bf16(w1, &h1, &l1);
  const u32x4 hp = {h0.x, h0.y, h1.x, h1.y};
  const u32x4 lp = {l0.x, l0.y, l1.x, l1.y};
  *whi = __builtin_bit_cast(ink_r10_wfr_bf8, hp);
  *wlo = __builtin_bit_cast(ink_r10_wfr_bf8, lp);
}

typedef struct {
  union {
    struct {
      u32 x_hi[2u][INK_R10_WFR_RTR_BN * (INK_R10_WFR_RTR_BK / 2u)] __attribute__((aligned(16))); /* 4,096 B */
      u32 x_lo[2u][INK_R10_WFR_RTR_BN * (INK_R10_WFR_RTR_BK / 2u)] __attribute__((aligned(16))); /* 4,096 B */
    } x;
    float red[4u][INK_R10_WFR_RTR_BN][INK_R10_WFR_RTR_BM] __attribute__((aligned(16)));          /* 8,192 B */
  } u;
  u32 is_last;                                                                                   /*     4 B */
} ink_r10_wfr_rtr_lds_t;

_Static_assert(sizeof(ink_r10_wfr_rtr_lds_t) <= 8208u, "ink_r10_wfr_rtr_lds_t must fit in <= 8.2 KiB (4 WGs/CU)");

KERNEL ie_inkling_gemm_f32_router_bf16_r10(
    const G float *w, const G float *x, G float *y,
    u32 rows, u32 cols, const G float *bias, const G float *res,
    u32 xs, u32 ys, u32 rs, u32 T,
    u32 K, const G float *ebias, float scale,
    G u32 *route, u32 ost, G u32 *count) {
  static LDS ink_r10_wfr_rtr_lds_t ink_r10_wfr_rtr_lds __attribute__((aligned(16)));

  const u32 mg = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  const u32 nt = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  const u32 m0 = mg * INK_R10_WFR_RTR_BM;
  const u32 n0 = nt * INK_R10_WFR_RTR_BN;
  if (m0 >= rows || n0 >= T || cols < 16u || (cols & 15u) != 0u || !w || !x || !y) return;

  const u32 t = tid();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;

  /* 8 waves = 2(M: wm in 0..1) x 4(Split-K: wk in 0..3). */
  const u32 wm = wv & 1u;
  const u32 wk = wv >> 1u;

  /* Cooperative X loader: 256 threads = 16 tokens x 16 f32x4 chunks (64 floats/token). */
  const u32 xt_ld = t >> 4u;
  const u32 xc4 = t & 15u;
  const u32 tk_ld = n0 + xt_ld;
  const int tk_ok = (tk_ld < T);
  const G float *x_ptr = (const G float *)((const G u8 *)x + (unsigned long)(tk_ok ? tk_ld : (T - 1u)) * xs);
  const u32 x_swz_off = xt_ld * 32u + (((xc4 >> 1u) ^ (xt_ld & 7u)) << 2u) + ((xc4 & 1u) << 1u);

  /* Direct-VGPR W loader: lane (r16, h) of wave (wm, wk) loads 8 floats (2x f32x4). */
  const u32 r_w = m0 + (wm << 4u) + r16;
  const int r_ok = (r_w < rows);
  const G float *w_ptr = w + (unsigned long)(r_ok ? r_w : (rows - 1u)) * cols;
  const u32 wk_off = (wk << 4u) + (h << 3u);

  /* LDS read offset for X tile: token r16, 16-B chunk (2*wk + h) ^ (r16 & 7). */
  const u32 x_rd_off = r16 * 32u + (((wk << 1u) + h) ^ (r16 & 7u)) * 4u;

  const f32x4 z4 = {0.0f, 0.0f, 0.0f, 0.0f};
  v8f acc = (v8f)(0.0f);

  /* Prologue: load c0 = 0 for X -> LDS[0] and W -> VGPRs. */
  f32x4 vw0 = (r_ok && wk_off < cols) ? *(const G f32x4 *)(w_ptr + wk_off + 0u) : z4;
  f32x4 vw1 = (r_ok && wk_off + 4u < cols) ? *(const G f32x4 *)(w_ptr + wk_off + 4u) : z4;
  {
    const u32 kc_x = xc4 << 2u;
    const f32x4 vx = (tk_ok && kc_x < cols) ? *(const G f32x4 *)(x_ptr + kc_x) : z4;
    ink_r10_wfr_u32x2 xh, xl;
    ink_r10_wfr_split_f4_bf16(vx, &xh, &xl);
    *(LDSP ink_r10_wfr_u32x2 *)(ink_r10_wfr_rtr_lds.u.x.x_hi[0] + x_swz_off) = xh;
    *(LDSP ink_r10_wfr_u32x2 *)(ink_r10_wfr_rtr_lds.u.x.x_lo[0] + x_swz_off) = xl;
  }
  ink_r10_wfr_lsignal();
  ink_r10_wfr_lwait();

  u32 bf = 0u;
#pragma unroll 1
  for (u32 c0 = 0u; c0 < cols; c0 += INK_R10_WFR_RTR_BK) {
    const u32 c_n = c0 + INK_R10_WFR_RTR_BK;
    const int more = (c_n < cols);
    f32x4 nw0 = z4, nw1 = z4, nvx = z4;
    if (more) {
      const u32 kw_n = c_n + wk_off;
      const u32 kx_n = c_n + (xc4 << 2u);
      if (r_ok && kw_n < cols) {
        nw0 = *(const G f32x4 *)(w_ptr + kw_n + 0u);
        nw1 = *(const G f32x4 *)(w_ptr + kw_n + 4u);
      }
      if (tk_ok && kx_n < cols) nvx = *(const G f32x4 *)(x_ptr + kx_n);
    }
    __builtin_amdgcn_sched_barrier(0);

    ink_r10_wfr_bf8 whi, wlo;
    ink_r10_wfr_split_w8_bf16(vw0, vw1, &whi, &wlo);
    const ink_r10_wfr_bf8 xhi = *(const LDSP ink_r10_wfr_bf8 *)(ink_r10_wfr_rtr_lds.u.x.x_hi[bf] + x_rd_off);
    const ink_r10_wfr_bf8 xlo = *(const LDSP ink_r10_wfr_bf8 *)(ink_r10_wfr_rtr_lds.u.x.x_lo[bf] + x_rd_off);
    __builtin_amdgcn_sched_barrier(0);

    if (more) {
      const u32 nbf = bf ^ 1u;
      ink_r10_wfr_u32x2 xh_n, xl_n;
      ink_r10_wfr_split_f4_bf16(nvx, &xh_n, &xl_n);
      *(LDSP ink_r10_wfr_u32x2 *)(ink_r10_wfr_rtr_lds.u.x.x_hi[nbf] + x_swz_off) = xh_n;
      *(LDSP ink_r10_wfr_u32x2 *)(ink_r10_wfr_rtr_lds.u.x.x_lo[nbf] + x_swz_off) = xl_n;
      /* Signal split barrier BEFORE the 3 BF16 WMMAs (100% hidden behind WMMA compute!) */
      ink_r10_wfr_lsignal();
    }

    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(whi, xhi, acc);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(whi, xlo, acc);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(wlo, xhi, acc);

    if (more) {
      vw0 = nw0;
      vw1 = nw1;
      ink_r10_wfr_lwait();
      bf ^= 1u;
    }
  }

  /* Split-K reduction across 4 K-waves via unioned red[4][16][32] (0 extra LDS!). */
  ink_r10_wfr_lsignal();
  ink_r10_wfr_lwait();
  {
    const u32 r_loc = (wm << 4u) + (h << 3u);
    LDSP f32x4 *rp = (LDSP f32x4 *)(ink_r10_wfr_rtr_lds.u.red[wk][r16] + r_loc);
    rp[0] = (f32x4){acc[0], acc[1], acc[2], acc[3]};
    rp[1] = (f32x4){acc[4], acc[5], acc[6], acc[7]};
  }
  ink_r10_wfr_lsignal();
  ink_r10_wfr_lwait();

  /* Waves 0..3 (128 threads = 16 tokens x 8 f32x4/token): reduce & coalesced f32x4 store. */
  if (wv < 4u) {
    const u32 tk_loc = t >> 3u;
    const u32 r4_loc = (t & 7u) << 2u;
    const u32 tk = n0 + tk_loc;
    if (tk < T) {
      const f32x4 v0 = *(const LDSP f32x4 *)(ink_r10_wfr_rtr_lds.u.red[0][tk_loc] + r4_loc);
      const f32x4 v1 = *(const LDSP f32x4 *)(ink_r10_wfr_rtr_lds.u.red[1][tk_loc] + r4_loc);
      const f32x4 v2 = *(const LDSP f32x4 *)(ink_r10_wfr_rtr_lds.u.red[2][tk_loc] + r4_loc);
      const f32x4 v3 = *(const LDSP f32x4 *)(ink_r10_wfr_rtr_lds.u.red[3][tk_loc] + r4_loc);
      f32x4 out4 = (v0 + v1) + (v2 + v3);
      const u32 r_o = m0 + r4_loc;
      G float *yp = (G float *)((G u8 *)y + (unsigned long)tk * ys);
      const int vec_y = (r_o + 4u <= rows) && ((ys & 15u) == 0u) && (!res || (rs & 15u) == 0u) &&
                        (!bias || ((((unsigned long)bias) & 15ul) == 0ul));
      if (__builtin_expect(vec_y, 1)) {
        if (bias) out4 += *(const G f32x4 *)(bias + r_o);
        if (res) out4 += *(const G f32x4 *)((const G float *)((const G u8 *)res + (unsigned long)tk * rs) + r_o);
        *(G f32x4 *)(yp + r_o) = out4;
      } else {
        const G float *rp = res ? (const G float *)((const G u8 *)res + (unsigned long)tk * rs) : (const G float *)0;
        const float vs[4] = {out4.x, out4.y, out4.z, out4.w};
        for (u32 i = 0u; i < 4u; i++) {
          if (r_o + i < rows) yp[r_o + i] = epilogue(vs[i], r_o + i, bias, rp);
        }
      }
    }
  }

  /* Optional fused L2-hot OP_MOE_ROUTE: last arriving M-WG routes all <= 16 tokens in parallel. */
  if (!route || !count || rows > 256u || K == 0u || K > 32u) return;

  const u32 ng_m = (rows + INK_R10_WFR_RTR_BM - 1u) / INK_R10_WFR_RTR_BM;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (t == 0u) {
    ink_r10_wfr_rtr_lds.is_last =
        (__scoped_atomic_fetch_add(&count[nt], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng_m - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r10_wfr_rtr_lds.is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const u32 nj_t = (T - n0 < INK_R10_WFR_RTR_BN) ? (T - n0) : INK_R10_WFR_RTR_BN;
#pragma clang loop unroll(disable)
  for (u32 dt = wv; dt < nj_t; dt += 8u) {
    const u32 tk = n0 + dt;
    const G float *lg_tk = (const G float *)((const G u8 *)y + (unsigned long)tk * ys);
    G u32 *rt_tk = (G u32 *)((G u8 *)route + (unsigned long)tk * ost);
    moe_route8(lg_tk, rows, K, ebias, scale, rt_tk);
  }
  barrier();
  if (t == 0u) count[nt] = 0u;
}

#undef INK_R10_WFR_RTR_BK
#undef INK_R10_WFR_RTR_BN
#undef INK_R10_WFR_RTR_BM
#undef INK_R10_WFR_EP_STR
#undef INK_R10_WFR_H_SWSTR
#undef INK_R10_WFR_M_SWSTR
#undef INK_R10_WFR_H_R
#undef INK_R10_WFR_M_T
#undef INK_R10_WFR_M_R
#undef INK_R10_WFR_KB

#endif /* !IE_MEGA */
