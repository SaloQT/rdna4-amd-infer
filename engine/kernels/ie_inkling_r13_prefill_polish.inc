/* ============================================================================
 * Inkling-2 Round 13 (Wave 13 - H07 Horizontal Multi-Proj Q8_0, H12 CU-Saturating
 * Prefill Glue, H10 L0$ Cached LUT Bypass FLUTE IQ3_XXS) for RX 9070 XT
 * (gfx1201, wave32, NT=256).
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r13_pf_ /
 * INK_R13_PF_ and all macros are #undef'd at EOF.
 *
 * Exported Kernels, Grids, Guards, and Parity:
 *
 * 1. KERNEL ie_inkling_gemm_q8_multi3_r13(
 *        const G u8 *xq, u32 nb, u32 T, u32 xs,
 *        const G u32 *qwA, const G f16 *qsA, G float *yA, u32 rowsA, u32 ysA,
 *        const G u32 *qwB, const G f16 *qsB, G float *yB, u32 rowsB, u32 ysB,
 *        const G u32 *qwC, const G f16 *qsC, G float *yC, u32 rowsC, u32 ysC,
 *        const G float *av_v, u32 a_vclip_hd, float a_vclip_clamp,
 *        u32 a_vclip_norope, float a_vclip_escale, u32 a_vclip_win, u32 a_vclip_pos0)
 *    - H07 Horizontal Multi-Projection Dense Q8_0 Prefill (64x64 WMMA):
 *      Combines up to 3 matrices (A, B, C) sharing quantized input `xq` into
 *      ONE launch (e.g. Wrel 384 + inline hd=16 r_clip + Wsink 24 + Wagt 3072
 *      -> kills q8_small_r1 #1 launch hotspot + r_clip = -144 launches/chunk;
 *      or looped/global split Wq 3072 + Wk 1024 + Wv 1024 = -60 launches/chunk!).
 *    - Grid:   ((((rowsA+63u)/64u) + ((rowsB+63u)/64u) + ((rowsC+63u)/64u)) *
 *              ((T + 63u) / 64u), 1u, 1u), NT = 256.
 *    - Strides: xs in BYTES; ysA, ysB, ysC in FLOATS (matching ie_gemm_q8r).
 *    - Guard:  !mk.rec && T >= 17u && nb >= 4u && (nb & 3u) == 0u && rowsA > 0u &&
 *              (a_vclip_hd == 0u || (a_vclip_hd == 16u && (rowsA & 15u) == 0u)).
 *    - Resources: <= 120 VGPR (true 12w/SIMD = 3 WGs/CU!), 19,968 B LDS, 0 v_fma_mix.
 *    - Parity: 100% bitwise-exact (0-bit-diff exact-SHA) vs ie_gemm_q8r [+ vclip_r1]!
 *
 * 2. KERNEL ie_inkling_kv_sconv_vclip_store_pf_r13(...)
 *    - H12 CU-Saturating TTILE=4 fix for R18's 16-WG under-saturation!
 *    - Exact 21-arg signature of ie_inkling_kv_sconv_vclip_store_pf_r8.
 *    - Grid:   (1u, (T + 3u) / 4u, 1u), NT = 256 (128 WGs at T=512 = 2.0 WGs/CU!).
 *    - Resources: <= 96 VGPR (4 WGs/CU!), 0 v_fma_mix*_f16 on KV stores, exact-SHA!
 *
 * 3. KERNEL ie_inkling_sconv_post_pf_r13(...)
 *    - H12 CU-Saturating TTILE=8 fix for R18's 48-WG under-saturation!
 *    - Exact 15-arg signature of ie_inkling_sconv_post_pf_r8.
 *    - Grid:   (((n >> 2u) + 255u) / 256u, (T + 7u) / 8u, 1u), NT = 256
 *              ((3, 64, 1) = 192 WGs at dim=2304, T=512 = 3.0 WGs/CU!).
 *    - Resources: <= 96 VGPR (4 WGs/CU!), 100% exact-SHA!
 *
 * 4. KERNEL ie_inkling_attn_res_norm_q8_pf_r13(...)
 *    - H12 T-WG prefill AttnRes + RMSNorm + Q8_0 (exact 25-arg r7 signature!).
 *    - Grid:   (1u, T, 1u), NT = 256 (512 WGs at T=512).
 *    - Uses default cached L0$/L2 loads (TH_LOAD_RT) on shared q/nw weights,
 *      fast K==1 path, split barriers, <= 96 VGPR (4 WGs/CU!), exact-SHA!
 *
 * 5. KERNEL ie_inkling_gemm_rq_exp_iq3xxs_flute_l0lut_r13(
 *        const G u8 *qw, const G u8 *X, G u8 *Qout, u32 eff, u32 nb,
 *        const G u32 *list, const G u32 *tiles, const G u32 *n_tiles,
 *        u32 q_bytes, u32 xs, u32 qstr, u32 xdiv, float beta, float lbeta)
 *    - H10 L0$ LUT bypass variant of 110-VGPR flute_situ_r10:
 *      reads ie_rq_iq3xxs_grid[256] directly via default cached L0$ loads
 *      (TH_LOAD_RT, 0 TH_LOAD_HT/NT), killing 100% of LUT LDS bank conflicts
 *      and saving 1,024 B LDS (19,712 -> 18,688 B)!
 *    - Exact 14-arg drop-in & grid of ie_inkling_gemm_rq_exp_iq3xxs_flute_situ_r10.
 *    - Resources: <= 112 VGPR (true 3 WGs/CU!), 18,688 B LDS, 0/0 spills, exact-SHA!
 *
 * 6. KERNEL ie_inkling_gemm_f32_router_bf16_r13(
 *        const G float *w, const G float *x, G float *y,
 *        u32 rows, u32 cols, const G float *bias, const G float *res,
 *        u32 xs, u32 ys, u32 rs, u32 T,
 *        u32 K, const G float *ebias, float scale,
 *        G u32 *route, u32 ost, G u32 *count)
 *    - H08 R49 forensic fix: Scaled 3-WMMA FP16-Pair Split (W*64, X*1 -> *1/64,
 *      22-bit mantissa via v_wmma_f32_16x16x16_f16_w32_gfx12 + hardware
 *      v_cvt_f16_f32 -> 18.7x lower RMS error than R10's 16-bit BF16-pair and
 *      3x less conversion VALU!) on (BM=32, BN=32, BK=64) with 2 N-tiles/wave
 *      (2x less W global traffic and W conversion vs R10's BN=16) + fused L2-hot
 *      OP_MOE_ROUTE (exact 17-arg drop-in for router_route_pf_r7 / bf16_r10!).
 *    - Grid:   ((rows + 31u) / 32u, (T + 31u) / 32u, 1u), NT = 256
 *              (64 WGs at T=512, 128 WGs at promoted --chunk 1024!).
 *    - Guard:  !mk.rec && T >= 17u && rows > 0u && cols >= 16u && (cols & 15u) == 0u &&
 *              (xs & 15u) == 0u && (((uintptr_t)w) & 15u) == 0u && (((uintptr_t)x) & 15u) == 0u;
 *              for fused route add !bias && !res && rows <= 256u && K > 0u && K <= 32u &&
 *              count != NULL && ((T + 31u) / 32u) <= count_cap (32 <= 384 in b->at_count).
 *    - Resources: <= 96 VGPR (4 WGs/CU!), 16,384 B LDS (4 WGs/CU!), 0/0 spills,
 *      0 v_fma_mix_f32, 0 TH_LOAD_NT/HT.
 *
 * 7. KERNEL ie_inkling_gemm_f32_router_exact_r13(exact 17-arg router signature)
 *    - 0-ULP exact-SHA companion to Kernel 6: sequential k=0..cols-1 FP32 FMA
 *      (32x32x32, 2x2 reg blocking, 16,384 B unioned LDS, <= 64 VGPR = 4 WGs/CU!)
 *      + fused L2-hot OP_MOE_ROUTE on the exact same ((rows+31)/32, (T+31)/32) grid.
 *    - Replaces 211-VGPR 32-WG unfused router_r2 with 100% 0-bit-diff logits &
 *      routing (0.000000 KLD delta, 4/4 guaranteed exact R37 SHA!).
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R13_PF_KB       4u
#define INK_R13_PF_H_R      64u
#define INK_R13_PF_M_T      64u
#define INK_R13_PF_H_SWSTR  96u
#define INK_R13_PF_KV_TTILE 4u
#define INK_R13_PF_SC_TTILE 8u
#define INK_R13_PF_AR_MAXM  32u
#define INK_R13_PF_AR_SLOTS 128u
#define INK_R13_PF_RTR_BM   32u
#define INK_R13_PF_RTR_BN   32u
#define INK_R13_PF_RTR_BK   64u
#define INK_R13_PF_EXR_BK   32u

typedef unsigned int ink_r13_pf_u32x2 __attribute__((ext_vector_type(2)));
typedef float        ink_r13_pf_f2    __attribute__((ext_vector_type(2)));

struct __attribute__((packed, aligned(4))) ink_r13_pf_u32x4p {
  u32x4 v;
};

struct __attribute__((packed, aligned(4))) ink_r13_pf_u32x2p {
  ink_r13_pf_u32x2 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r13_pf_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r13_pf_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) ink_r13_pf_u32x2 ink_r13_pf_ld2(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r13_pf_u32x2p *)p)->v;
}

static inline __attribute__((always_inline)) void ink_r13_pf_lsignal(void) {
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  __builtin_amdgcn_s_barrier_signal(-1);
  __asm__ volatile("" ::: "memory");
}

static inline __attribute__((always_inline)) void ink_r13_pf_lwait(void) {
  __builtin_amdgcn_s_barrier_wait(-1);
  __asm__ volatile("" ::: "memory");
}

/* ============================================================================
 * Kernel 1 (H07): ie_inkling_gemm_q8_multi3_r13
 *   Horizontal 3-projection 64x64 dense Q8_0 prefill WMMA sharing `xq` +
 *   optional exact-SHA inline hd=16 vclip on Matrix A (Wrel + r_clip!).
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r13_pf_m3_stage_sw(
    LDSP f16 *lsw, const u32 r_idx, const ink_r13_pf_u32x2 rsw) {
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
  const u32 str32 = INK_R13_PF_H_SWSTR >> 1u;
  sw32[(0u + odd) * str32 + (r2 >> 1u)] = v_a;
  sw32[(2u + odd) * str32 + (r2 >> 1u)] = v_b;
}

static inline __attribute__((always_inline)) void ink_r13_pf_m3_stage_sx(
    LDSP float *lsx, const u32 t_idx, const u32x4 rsx) {
  u32 ti = t_idx;
  __asm__ volatile("" : "+v"(ti));
  const u32 d0 = rsx.x, d1 = rsx.y, d2 = rsx.z, d3 = rsx.w;
  lsx[0u * INK_R13_PF_M_T + ti] = __builtin_bit_cast(float, d0);
  lsx[1u * INK_R13_PF_M_T + ti] = __builtin_bit_cast(float, d1);
  lsx[2u * INK_R13_PF_M_T + ti] = __builtin_bit_cast(float, d2);
  lsx[3u * INK_R13_PF_M_T + ti] = __builtin_bit_cast(float, d3);
}

static inline __attribute__((always_inline)) void ink_r13_pf_m3_k32(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wm, const u32 h, const u32 lx_base0, f8 *acc0, f8 *acc1) {
  const v8i zm = (v8i)(0x4B400000);
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * INK_R13_PF_M_T;
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

  const h8 sw_h = *(const LDSP h8 *)(lsw + k * INK_R13_PF_H_SWSTR + wm + (h << 3u));
  f8 sw = __builtin_convertvector(sw_h, f8);
  __asm__ volatile("" : "+v"(sw));
  *acc0 += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  __asm__ volatile("" : : "v"(*acc0));
  __builtin_amdgcn_sched_barrier(0);
  *acc1 += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
  __asm__ volatile("" : : "v"(*acc1) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) float ink_r13_pf_vclip_scale(
    const float ss, const float inv_hd, const float clamp, const u32 norope,
    const float escale, const u32 win, const u32 pos0, const u32 t) {
  float f;
  if (clamp > 0.0f) {
    f = clamp / (__builtin_sqrtf(ss) + 1e-6f);
    if (f > 1.0f) f = 1.0f;
    if (norope) {
      const float ratio = (float)(pos0 + t + 1u) / (float)win;
      if (ratio > 1.0f) f *= 1.0f + escale * __builtin_logf(ratio);
    }
  } else {
    f = 1.0f / __builtin_sqrtf(ss * inv_hd + escale);
  }
  return f;
}

/* Exact-SHA hd=16 inline vclip + store for one token's 16-element head (lane h=0: 0..7, h=1: 8..15).
 * Matches ie_inkling_vclip_r1 tree order ((ss0 + ss2) + (ss1 + ss3)) bit-for-bit! */
static inline __attribute__((always_inline)) void ink_r13_pf_m3_store_row8(
    G float *y, const u32 t, const u32 T, const u32 r0, const u32 rows, const u32 ys,
    f8 av, const int do_vclip16, const G float *av_v, const float clamp,
    const u32 norope, const float escale, const u32 win, const u32 pos0) {
  if (do_vclip16) {
    f32x4 lo = {av[0], av[1], av[2], av[3]};
    f32x4 hi = {av[4], av[5], av[6], av[7]};
    f32x4 v0 = {1.0f, 1.0f, 1.0f, 1.0f}, v1 = {1.0f, 1.0f, 1.0f, 1.0f};
    if (av_v && r0 + 8u <= rows) {
      const G f32x4 *vp4 = (const G f32x4 *)(av_v + r0);
      v0 = vp4[0];
      v1 = vp4[1];
      if (clamp > 0.0f) {
        lo *= v0;
        hi *= v1;
      }
    }
    const float s_lo = (lo.x * lo.x + lo.y * lo.y) + (lo.z * lo.z + lo.w * lo.w);
    const float s_hi = (hi.x * hi.x + hi.y * hi.y) + (hi.z * hi.z + hi.w * hi.w);
    const float a_p = s_lo + I2F(xrow_i(F2I(s_lo)));
    const float b_p = s_hi + I2F(xrow_i(F2I(s_hi)));
    const float ss = a_p + b_p;
    if (t < T && r0 + 8u <= rows) {
      const float f = ink_r13_pf_vclip_scale(ss, 1.0f / 16.0f, clamp, norope, escale, win, pos0, t);
      lo *= f;
      hi *= f;
      if (clamp == 0.0f && av_v) {
        lo *= v0;
        hi *= v1;
      }
      G float *yt = y + (unsigned long)t * (unsigned long)ys;
      if (((ys & 3u) == 0u) && ((((unsigned long)y) & 15ul) == 0ul)) {
        G f32x4 *yt4 = (G f32x4 *)(yt + r0);
        yt4[0] = lo;
        yt4[1] = hi;
      } else {
        yt[r0 + 0u] = lo.x; yt[r0 + 1u] = lo.y; yt[r0 + 2u] = lo.z; yt[r0 + 3u] = lo.w;
        yt[r0 + 4u] = hi.x; yt[r0 + 5u] = hi.y; yt[r0 + 6u] = hi.z; yt[r0 + 7u] = hi.w;
      }
    }
    return;
  }

  if (t >= T) return;
  G float *yt = y + (unsigned long)t * (unsigned long)ys;
  if (__builtin_expect((r0 + 8u <= rows) && ((ys & 3u) == 0u) && ((((unsigned long)y) & 15ul) == 0ul), 1)) {
    G f32x4 *yt4 = (G f32x4 *)(yt + r0);
    yt4[0] = (f32x4){av[0], av[1], av[2], av[3]};
    yt4[1] = (f32x4){av[4], av[5], av[6], av[7]};
  } else {
#pragma unroll 1
    for (u32 v = 0u; v < 8u; v++) {
      const u32 r = r0 + v;
      if (r < rows) yt[r] = av[v];
    }
  }
}

KERNEL ie_inkling_gemm_q8_multi3_r13(
    const G u8 *xq, u32 nb, u32 T, u32 xs,
    const G u32 *qwA, const G f16 *qsA, G float *yA, u32 rowsA, u32 ysA,
    const G u32 *qwB, const G f16 *qsB, G float *yB, u32 rowsB, u32 ysB,
    const G u32 *qwC, const G f16 *qsC, G float *yC, u32 rowsC, u32 ysC,
    const G float *av_v, u32 a_vclip_hd, float a_vclip_clamp,
    u32 a_vclip_norope, float a_vclip_escale, u32 a_vclip_win, u32 a_vclip_pos0) {
  static LDS struct {
    u8    x[2u][INK_R13_PF_M_T * (INK_R13_PF_KB * 32u)] __attribute__((aligned(16))); /* 16,384 B */
    f16   sw[2u][INK_R13_PF_KB * INK_R13_PF_H_SWSTR] __attribute__((aligned(16)));    /*  1,536 B */
    float sx[2u][INK_R13_PF_KB * INK_R13_PF_M_T] __attribute__((aligned(32)));        /*  2,048 B */
  } ink_r13_pf_m3_lds;
  _Static_assert(sizeof(ink_r13_pf_m3_lds) == 19968u, "ink_r13_pf_m3_lds must be 19,968 B (3 WGs/CU)");

  if (T == 0u || nb < INK_R13_PF_KB || (nb & (INK_R13_PF_KB - 1u)) != 0u || !xq || rowsA == 0u) return;

  const u32 nrtA = (rowsA + INK_R13_PF_H_R - 1u) / INK_R13_PF_H_R;
  const u32 nrtB = (rowsB + INK_R13_PF_H_R - 1u) / INK_R13_PF_H_R;
  const u32 nrtC = (rowsC + INK_R13_PF_H_R - 1u) / INK_R13_PF_H_R;
  const u32 nrt_tot = nrtA + nrtB + nrtC;
  const u32 ntt = (T + INK_R13_PF_M_T - 1u) / INK_R13_PF_M_T;
  const u32 wg = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  if (wg >= nrt_tot * ntt) return;

  /* 4x4 WGP-aware L2/MALL swizzling across the unified row-tile space (shares xq in L2!). */
  const u32 grp_m = 4u;
  const u32 grp_sz = grp_m * ntt;
  const u32 grp = wg / grp_sz;
  const u32 in_grp = wg - grp * grp_sz;
  const u32 r_grp0 = grp * grp_m;
  const u32 r_rem = nrt_tot - r_grp0;
  const u32 gw = r_rem < grp_m ? r_rem : grp_m;
  const u32 tt = (u32)__builtin_amdgcn_readfirstlane((int)(in_grp / gw));
  const u32 rt_g = (u32)__builtin_amdgcn_readfirstlane((int)(r_grp0 + (in_grp - tt * gw)));

  /* Select target projection (A, B, or C) in SGPRs. */
  const G u32 *qw;
  const G f16 *qs;
  G float *y;
  u32 rows, ys, rt_loc;
  int do_vclip16 = 0;
  if (rt_g < nrtA) {
    qw = qwA; qs = qsA; y = yA; rows = rowsA; ys = ysA; rt_loc = rt_g;
    do_vclip16 = (a_vclip_hd == 16u);
  } else if (rt_g < nrtA + nrtB) {
    qw = qwB; qs = qsB; y = yB; rows = rowsB; ys = ysB; rt_loc = rt_g - nrtA;
  } else {
    qw = qwC; qs = qsC; y = yC; rows = rowsC; ys = ysC; rt_loc = rt_g - nrtA - nrtB;
  }
  if (!qw || !qs || !y || rows == 0u) return;

  const u32 rbase = rt_loc * INK_R13_PF_H_R;
  const u32 tbase = tt * INK_R13_PF_M_T;

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

  u32x4 wa[INK_R13_PF_KB];
#pragma unroll
  for (u32 k = 0u; k < INK_R13_PF_KB; k++) wa[k] = ink_r13_pf_ld4(qw8 + w_off + k * 32u);
  {
    const u32x4 rx0_i = ink_r13_pf_ld4(xq8 + xoff0);
    const u32x4 rx1_i = ink_r13_pf_ld4(xq8 + xoff1);
    const ink_r13_pf_u32x2 rsw_i = is_sw ? ink_r13_pf_ld2(qs8 + sw_off) : (ink_r13_pf_u32x2){0u, 0u};
    const u32x4 rsx_i = is_sx ? ink_r13_pf_ld4(xq_s + sx_off) : (u32x4)(0u);
    *(LDSP u32x4 *)(ink_r13_pf_m3_lds.x[0] + st_x) = rx0_i;
    *(LDSP u32x4 *)(ink_r13_pf_m3_lds.x[0] + st_x + 4096u) = rx1_i;
    if (is_sw) ink_r13_pf_m3_stage_sw(ink_r13_pf_m3_lds.sw[0], swi, rsw_i);
    if (is_sx) ink_r13_pf_m3_stage_sx(ink_r13_pf_m3_lds.sx[0], sxi, rsx_i);
  }
  ink_r13_pf_lsignal();
  ink_r13_pf_lwait();

  u32 bf = 0u;
  u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R13_PF_KB) {
    const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R13_PF_KB));
    const int more = (kb_n < nb);
    __asm__ volatile("" : : "v"(wa[0]), "v"(wa[1]));
    if (more) {
      const G u8 *xq_n = rq_ptr_u(xq8 + (unsigned long)kb_n * 32ul);
      rx0 = ink_r13_pf_ld4(xq_n + xoff0);
      rx1 = ink_r13_pf_ld4(xq_n + xoff1);
    }
    __builtin_amdgcn_sched_barrier(0);

    const LDSP u8 *lx = ink_r13_pf_m3_lds.x[bf];
    const LDSP f16 *lsw = ink_r13_pf_m3_lds.sw[bf];
    const LDSP float *lsx = ink_r13_pf_m3_lds.sx[bf] + wn + r16;

    ink_r13_pf_m3_k32(wa[0], lx, lsw, lsx, 0u, wm, h, lx_base0, &acc0, &acc1);
    ink_r13_pf_m3_k32(wa[1], lx, lsw, lsx, 1u, wm, h, lx_base0, &acc0, &acc1);

    ink_r13_pf_u32x2 rsw = {0u, 0u};
    u32x4 rsx = (u32x4)(0u);
    if (more) {
      const u32 nbf = bf ^ 1u;
      *(LDSP u32x4 *)(ink_r13_pf_m3_lds.x[nbf] + st_x) = rx0;
      *(LDSP u32x4 *)(ink_r13_pf_m3_lds.x[nbf] + st_x + 4096u) = rx1;
      const G u8 *qs_n = rq_ptr_u(qs8 + (unsigned long)kb_n * 2ul);
      const G u8 *xs_n = rq_ptr_u(xq_s + (unsigned long)kb_n * 4ul);
      if (is_sw) rsw = ink_r13_pf_ld2(qs_n + sw_off);
      if (is_sx) rsx = ink_r13_pf_ld4(xs_n + sx_off);
    }
    __builtin_amdgcn_sched_barrier(0);

    ink_r13_pf_m3_k32(wa[2], lx, lsw, lsx, 2u, wm, h, lx_base0, &acc0, &acc1);

    if (more) {
      const u32 nbf = bf ^ 1u;
      if (is_sw) ink_r13_pf_m3_stage_sw(ink_r13_pf_m3_lds.sw[nbf], swi, rsw);
      if (is_sx) ink_r13_pf_m3_stage_sx(ink_r13_pf_m3_lds.sx[nbf], sxi, rsx);
      ink_r13_pf_lsignal();
      const G u8 *qw_n = rq_ptr_u(qw8 + (unsigned long)kb_n * 32ul);
      wa[0] = ink_r13_pf_ld4(qw_n + w_off + 0u);
      wa[1] = ink_r13_pf_ld4(qw_n + w_off + 32u);
    }
    __builtin_amdgcn_sched_barrier(0);

    ink_r13_pf_m3_k32(wa[3], lx, lsw, lsx, 3u, wm, h, lx_base0, &acc0, &acc1);
    if (more) {
      const G u8 *qw_n = rq_ptr_u(qw8 + (unsigned long)kb_n * 32ul);
      wa[2] = ink_r13_pf_ld4(qw_n + w_off + 64u);
      wa[3] = ink_r13_pf_ld4(qw_n + w_off + 96u);
      ink_r13_pf_lwait();
      bf ^= 1u;
    }
  }

  const u32 r0 = rbase + wm + (h << 3u);
  const u32 t_0 = tbase + wn + r16;
  const u32 t_1 = tbase + wn + 16u + r16;
  ink_r13_pf_m3_store_row8(y, t_0, T, r0, rows, ys, acc0, do_vclip16, av_v, a_vclip_clamp,
                           a_vclip_norope, a_vclip_escale, a_vclip_win, a_vclip_pos0);
  ink_r13_pf_m3_store_row8(y, t_1, T, r0, rows, ys, acc1, do_vclip16, av_v, a_vclip_clamp,
                           a_vclip_norope, a_vclip_escale, a_vclip_win, a_vclip_pos0);
}

/* ============================================================================
 * Kernels 2 & 3 (H12): CU-Saturating TTILE=4 / TTILE=8 Prefill Short-Conv Glue
 *   Fixes R18's TTILE=32 under-saturation (16 WGs -> 128 WGs; 48 -> 192 WGs!).
 *   <= 96 VGPR (4 WGs/CU!), 0 v_fma_mix*_f16 on KV stores, 100% exact-SHA!
 * ========================================================================== */

static inline __attribute__((always_inline)) float ink_r13_pf_sc_pick4(const f32x4 v, const u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline __attribute__((always_inline)) void ink_r13_pf_sc_load_sw(
    const G float *v, const u32 i0, f32x4 *sw0, f32x4 *sw1, f32x4 *sw2, f32x4 *sw3) {
  const G f32x4 *v4 = (const G f32x4 *)v + i0;
  const f32x4 vw0 = v4[0], vw1 = v4[1], vw2 = v4[2], vw3 = v4[3];
  *sw0 = (f32x4){vw0.w, vw1.w, vw2.w, vw3.w};
  *sw1 = (f32x4){vw0.z, vw1.z, vw2.z, vw3.z};
  *sw2 = (f32x4){vw0.y, vw1.y, vw2.y, vw3.y};
  *sw3 = (f32x4){vw0.x, vw1.x, vw2.x, vw3.x};
}

static inline __attribute__((always_inline)) void ink_r13_pf_sc_seed_ring(
    const G float *ring, const u32 i0, const u32 pos0, f32x4 *x1, f32x4 *x2, f32x4 *x3) {
  *x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (pos0 > 0u) {
    const G f32x4 *rg = (const G f32x4 *)ring + i0;
    const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
    const u32 s1 = (pos0 - 1u) & 3u;
    *x1 = (f32x4){ink_r13_pf_sc_pick4(r0, s1), ink_r13_pf_sc_pick4(r1, s1),
                  ink_r13_pf_sc_pick4(r2, s1), ink_r13_pf_sc_pick4(r3, s1)};
    if (pos0 >= 2u) {
      const u32 s2 = (pos0 - 2u) & 3u;
      *x2 = (f32x4){ink_r13_pf_sc_pick4(r0, s2), ink_r13_pf_sc_pick4(r1, s2),
                    ink_r13_pf_sc_pick4(r2, s2), ink_r13_pf_sc_pick4(r3, s2)};
    }
    if (pos0 >= 3u) {
      const u32 s3 = (pos0 - 3u) & 3u;
      *x3 = (f32x4){ink_r13_pf_sc_pick4(r0, s3), ink_r13_pf_sc_pick4(r1, s3),
                    ink_r13_pf_sc_pick4(r2, s3), ink_r13_pf_sc_pick4(r3, s3)};
    }
  }
}

/* Exact double-rounded f32->f16 KV store barrier: kills 100% of v_fma_mix*_f16! */
static inline __attribute__((always_inline)) kv4 ink_r13_pf_f32x4_to_kv4(const f32x4 v) {
  float x0 = v.x, x1 = v.y, x2 = v.z, x3 = v.w;
  __asm__ volatile("" : "+v"(x0), "+v"(x1), "+v"(x2), "+v"(x3));
  return __builtin_convertvector((f32x4){x0, x1, x2, x3}, kv4);
}

static inline __attribute__((always_inline)) f32x4 ink_r13_pf_sc_conv4(
    const f32x4 x0, const f32x4 x1, const f32x4 x2, const f32x4 x3,
    const f32x4 w0, const f32x4 w1, const f32x4 w2, const f32x4 w3, const float escale) {
#pragma clang fp contract(off)
  const f32x4 y = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3;
  return (x0 + y) * escale;
}

static inline __attribute__((always_inline)) void ink_r13_pf_sc_ring_commit4(
    const G float *a, G float *ring, const u32 i4, const u32 i0,
    const u32 pos0, const u32 T, const u32 as) {
  if (T >= 4u) {
    const u32 k0 = T - 4u;
    const u32 s0 = (pos0 + k0) & 3u;
    const u32 t_s0 = k0 + ((0u - s0) & 3u);
    const u32 t_s1 = k0 + ((1u - s0) & 3u);
    const u32 t_s2 = k0 + ((2u - s0) & 3u);
    const u32 t_s3 = k0 + ((3u - s0) & 3u);
    const f32x4 v0 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t_s0 * (unsigned long)as))[i4];
    const f32x4 v1 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t_s1 * (unsigned long)as))[i4];
    const f32x4 v2 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t_s2 * (unsigned long)as))[i4];
    const f32x4 v3 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t_s3 * (unsigned long)as))[i4];
    G f32x4 *rg = (G f32x4 *)ring + i0;
    rg[0] = (f32x4){v0.x, v1.x, v2.x, v3.x};
    rg[1] = (f32x4){v0.y, v1.y, v2.y, v3.y};
    rg[2] = (f32x4){v0.z, v1.z, v2.z, v3.z};
    rg[3] = (f32x4){v0.w, v1.w, v2.w, v3.w};
  } else {
    for (u32 tk = 0u; tk < T; tk++) {
      const f32x4 v = ((const G f32x4 *)((const G u8 *)a + (unsigned long)tk * (unsigned long)as))[i4];
      const u32 slot = (pos0 + tk) & 3u;
      ring[(unsigned long)(i0 + 0u) * 4ul + slot] = v.x;
      ring[(unsigned long)(i0 + 1u) * 4ul + slot] = v.y;
      ring[(unsigned long)(i0 + 2u) * 4ul + slot] = v.z;
      ring[(unsigned long)(i0 + 3u) * 4ul + slot] = v.w;
    }
  }
}

KERNEL ie_inkling_kv_sconv_vclip_store_pf_r13(
    const G float *ka, const G float *vk, const G float *ck, G float *ring_k,
    const G float *va, const G float *vv, const G float *cv, G float *ring_v,
    G kvt *kc0, G kvt *vc0, float sc_escale, float k_clamp, float v_eps,
    u32 pos0, u32 T, u32 kv_len, u32 kas, u32 cks, u32 vas, u32 cvs,
    G u32 *ring_cnt) {
  static LDS u32 ink_r13_pf_kv_last;

  if (T == 0u || kv_len == 0u) return;
  const u32 n_tiles = (T + INK_R13_PF_KV_TTILE - 1u) / INK_R13_PF_KV_TTILE;
  const u32 tile = tokid();
  if (tile >= n_tiles) return;

  const u32 t0 = tile * INK_R13_PF_KV_TTILE;
  const u32 t_end = (t0 + INK_R13_PF_KV_TTILE < T) ? (t0 + INK_R13_PF_KV_TTILE) : T;
  const u32 i4 = tid();
  const u32 i0 = i4 << 2u;
  const u32 h = i4 >> 5u;
  const u32 p0 = (pos0 + t0) % kv_len;

  /* K path (TTILE = 4 -> 128 WGs at T=512 = 2.0 WGs/CU!). */
  {
    f32x4 sw0, sw1, sw2, sw3;
    ink_r13_pf_sc_load_sw(vk, i0, &sw0, &sw1, &sw2, &sw3);
    f32x4 x1, x2, x3;
    if (t0 >= 3u) {
      x3 = ((const G f32x4 *)((const G u8 *)ka + (unsigned long)(t0 - 3u) * (unsigned long)kas))[i4];
      x2 = ((const G f32x4 *)((const G u8 *)ka + (unsigned long)(t0 - 2u) * (unsigned long)kas))[i4];
      x1 = ((const G f32x4 *)((const G u8 *)ka + (unsigned long)(t0 - 1u) * (unsigned long)kas))[i4];
    } else {
      ink_r13_pf_sc_seed_ring(ring_k, i0, pos0, &x1, &x2, &x3);
    }
    u32 p = p0;
    for (u32 tk = t0; tk < t_end; tk++) {
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)ka + (unsigned long)tk * (unsigned long)kas))[i4];
      f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
      if (ck) {
        const f32x4 ct = ((const G f32x4 *)((const G u8 *)ck + (unsigned long)tk * (unsigned long)cks))[h];
        w0 += ct.x; w1 += ct.y; w2 += ct.z; w3 += ct.w;
      }
      const f32x4 kx = ink_r13_pf_sc_conv4(x0, x1, x2, x3, w0, w1, w2, w3, sc_escale);
      float ss = (kx.x * kx.x + kx.y * kx.y) + (kx.z * kx.z + kx.w * kx.w);
      ss = wave_sum_all(ss);
      float fk = k_clamp / (__builtin_sqrtf(ss) + 1e-6f);
      if (fk > 1.0f) fk = 1.0f;
      ((G kv4 *)kc0)[(unsigned long)p * 256ul + (unsigned long)i4] = ink_r13_pf_f32x4_to_kv4(kx * fk);
      x3 = x2; x2 = x1; x1 = x0;
      p++;
      if (p == kv_len) p = 0u;
    }
  }

  /* V path (TTILE = 4). */
  {
    f32x4 sw0, sw1, sw2, sw3;
    ink_r13_pf_sc_load_sw(vv, i0, &sw0, &sw1, &sw2, &sw3);
    f32x4 x1, x2, x3;
    if (t0 >= 3u) {
      x3 = ((const G f32x4 *)((const G u8 *)va + (unsigned long)(t0 - 3u) * (unsigned long)vas))[i4];
      x2 = ((const G f32x4 *)((const G u8 *)va + (unsigned long)(t0 - 2u) * (unsigned long)vas))[i4];
      x1 = ((const G f32x4 *)((const G u8 *)va + (unsigned long)(t0 - 1u) * (unsigned long)vas))[i4];
    } else {
      ink_r13_pf_sc_seed_ring(ring_v, i0, pos0, &x1, &x2, &x3);
    }
    u32 p = p0;
    for (u32 tk = t0; tk < t_end; tk++) {
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)va + (unsigned long)tk * (unsigned long)vas))[i4];
      f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
      if (cv) {
        const f32x4 ct = ((const G f32x4 *)((const G u8 *)cv + (unsigned long)tk * (unsigned long)cvs))[h];
        w0 += ct.x; w1 += ct.y; w2 += ct.z; w3 += ct.w;
      }
      const f32x4 vx = ink_r13_pf_sc_conv4(x0, x1, x2, x3, w0, w1, w2, w3, sc_escale);
      float ss = (vx.x * vx.x + vx.y * vx.y) + (vx.z * vx.z + vx.w * vx.w);
      ss = wave_sum_all(ss);
      const float fv = 1.0f / __builtin_sqrtf(ss * (1.0f / 128.0f) + v_eps);
      ((G kv4 *)vc0)[(unsigned long)p * 256ul + (unsigned long)i4] = ink_r13_pf_f32x4_to_kv4(vx * fv);
      x3 = x2; x2 = x1; x1 = x0;
      p++;
      if (p == kv_len) p = 0u;
    }
  }

  if (!ring_cnt) return;
  if (n_tiles > 1u) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (tid() == 0u) {
      ink_r13_pf_kv_last =
          (__scoped_atomic_fetch_add(ring_cnt, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == n_tiles - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r13_pf_kv_last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (tid() == 0u) __scoped_atomic_store_n(ring_cnt, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  }
  ink_r13_pf_sc_ring_commit4(ka, ring_k, i4, i0, pos0, T, kas);
  ink_r13_pf_sc_ring_commit4(va, ring_v, i4, i0, pos0, T, vas);
}

KERNEL ie_inkling_sconv_post_pf_r13(
    const G float *a, const G float *v, G float *ring,
    const G float *res, G float *out, G float *blk_dst,
    u32 n, float escale, u32 pos0, u32 T,
    u32 as, u32 rs, u32 os, u32 bs, G u32 *ring_cnt) {
#pragma clang fp contract(off)
  static LDS u32 ink_r13_pf_post_last;

  if (T == 0u || n == 0u) return;
  const u32 n4 = n >> 2u;
  const u32 gx = wgid();
  const u32 i4 = gx * NT + tid();
  const u32 n_tiles = (T + INK_R13_PF_SC_TTILE - 1u) / INK_R13_PF_SC_TTILE;
  const u32 tile = tokid();
  if (tile >= n_tiles || gx * NT >= n4) return;

  const u32 t0 = tile * INK_R13_PF_SC_TTILE;
  const u32 t_end = (t0 + INK_R13_PF_SC_TTILE < T) ? (t0 + INK_R13_PF_SC_TTILE) : T;

  if (i4 < n4) {
    const u32 i0 = i4 << 2u;
    f32x4 sw0, sw1, sw2, sw3;
    ink_r13_pf_sc_load_sw(v, i0, &sw0, &sw1, &sw2, &sw3);

    f32x4 x1, x2, x3;
    if (t0 >= 3u) {
      x3 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)(t0 - 3u) * (unsigned long)as))[i4];
      x2 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)(t0 - 2u) * (unsigned long)as))[i4];
      x1 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)(t0 - 1u) * (unsigned long)as))[i4];
    } else {
      ink_r13_pf_sc_seed_ring(ring, i0, pos0, &x1, &x2, &x3);
    }

    for (u32 tk = t0; tk < t_end; tk++) {
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)tk * (unsigned long)as))[i4];
      const f32x4 rv = ((const G f32x4 *)((const G u8 *)res + (unsigned long)tk * (unsigned long)rs))[i4];
      const f32x4 sc = ink_r13_pf_sc_conv4(x0, x1, x2, x3, sw0, sw1, sw2, sw3, escale);
      if (blk_dst) {
        ((G f32x4 *)((G u8 *)blk_dst + (unsigned long)tk * (unsigned long)bs))[i4] = rv;
        ((G f32x4 *)((G u8 *)out + (unsigned long)tk * (unsigned long)os))[i4] = sc;
      } else {
        ((G f32x4 *)((G u8 *)out + (unsigned long)tk * (unsigned long)os))[i4] = rv + sc;
      }
      x3 = x2; x2 = x1; x1 = x0;
    }
  }

  if (!ring_cnt) return;
  if (n_tiles > 1u) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (tid() == 0u) {
      ink_r13_pf_post_last =
          (__scoped_atomic_fetch_add(ring_cnt + gx, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == n_tiles - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r13_pf_post_last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (tid() == 0u) __scoped_atomic_store_n(ring_cnt + gx, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  }

  if (i4 < n4) {
    ink_r13_pf_sc_ring_commit4(a, ring, i4, i4 << 2u, pos0, T, as);
  }
}

/* ============================================================================
 * Kernel 4 (H12): ie_inkling_attn_res_norm_q8_pf_r13
 *   T-WG prefill AttnRes + RMSNorm + Q8_0 with default cached loads on shared
 *   q/nw weights, fast K==1 specialization, split barriers, <= 96 VGPR, exact-SHA!
 * ========================================================================== */

typedef struct {
  union {
    float ar_red[INK_R13_PF_AR_SLOTS * 8u]; /* 4,096 B */
    struct {
      float rn_red[8u];
      float rn_sh[8u][128u];                /* 4,096 B */
    } rn;
  } u;
} ink_r13_pf_ar_lds_t;

static LDS ink_r13_pf_ar_lds_t ink_r13_pf_ar_lds __attribute__((aligned(16)));

static inline __attribute__((always_inline)) float ink_r13_pf_wave_max(float v) {
  v = __builtin_fmaxf(v, shfl_xor_f(v, 16u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 8u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 4u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 2u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 1u));
  return v;
}

static inline __attribute__((always_inline)) float ink_r13_pf_wave_softmax(
    const float logit, const u32 l, const u32 M) {
  const float v = (l < M) ? logit : -__builtin_inff();
  const float mx = ink_r13_pf_wave_max(v);
  const float e = (l < M) ? __builtin_expf(logit - mx) : 0.0f;
  const float ls = wave_sum_all(e);
  return (l < M) ? (e * (1.0f / ls)) : 0.0f;
}

static inline __attribute__((always_inline)) void ink_r13_pf_quant_chunk4(
    const f32x4 y4, const u32 wv, const u32 l, const u32 b0, G u8 *q) {
  *(LDSP f32x4 *)&ink_r13_pf_ar_lds.u.rn.rn_sh[wv][l << 2u] = y4;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 m = 0u; m < 4u; m++) quant_wave(ink_r13_pf_ar_lds.u.rn.rn_sh[wv][(m << 5u) + l], b0 + m, q, 72u);
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

static inline __attribute__((always_inline)) void ink_r13_pf_norm_2304(
    const f32x4 x0, const f32x4 x1, const float x2, const G float *nw, G float *y, G u8 *q,
    const float eps, const u32 th, const u32 wv, const u32 l) {
  const f32x4 w0 = *(const G f32x4 *)(nw + (th << 2u));
  const f32x4 w1 = *(const G f32x4 *)(nw + 1024u + (th << 2u));
  const float w2 = nw[2048u + th];

  float ss = (x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w);
  ss += (x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w);
  ss += x2 * x2;
  ss = wave_tree_f(ss);
  if (l == 0u) ink_r13_pf_ar_lds.u.rn.rn_red[wv] = ss;
  ink_r13_pf_lsignal();
  ink_r13_pf_lwait();

  float tot = 0.0f;
#pragma unroll
  for (u32 w = 0u; w < 8u; w++) tot += ink_r13_pf_ar_lds.u.rn.rn_red[w];
  const float s = 1.0f / __builtin_sqrtf(tot / 2304.0f + eps);

  const f32x4 y0 = x0 * s * w0;
  const f32x4 y1 = x1 * s * w1;
  const float y2 = x2 * s * w2;
  if (y) {
    *(G f32x4 *)(y + (th << 2u)) = y0;
    *(G f32x4 *)(y + 1024u + (th << 2u)) = y1;
    y[2048u + th] = y2;
  }
  if (q) {
    ink_r13_pf_quant_chunk4(y0, wv, l, wv << 2u, q);
    ink_r13_pf_quant_chunk4(y1, wv, l, 32u + (wv << 2u), q);
    quant_wave(y2, 64u + wv, q, 72u);
  }
}

KERNEL ie_inkling_attn_res_norm_q8_pf_r13(
    const G float *a, const G float *c, const G float *q,
    const G float *nw0, G float *y0, G u8 *q0,
    const G float *nw1, G float *y1, G u8 *q1,
    const G float *nw2, G float *y2, G u8 *q2,
    u32 M, u32 K, u32 n, float eps,
    u32 as, u32 cs, u32 y0s, u32 q0s, u32 y1s, u32 q1s, u32 y2s, u32 q2s,
    int has_c) {
  if (n != 2304u || M == 0u || M > INK_R13_PF_AR_MAXM || K == 0u || K > 3u) return;

  const u32 th = tid();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(th >> 5u));
  const u32 l = lane();
  if (a) a = TOKC(a, as);
  if (has_c && c) c = TOKC(c, cs);
  if (y0) y0 = TOK(y0, y0s);
  if (q0) q0 = TOK(q0, q0s);

  const u32 n_a = has_c ? (M - 1u) : M;
  const float inv_sqn = 1.0f / __builtin_sqrtf(2304.0f);
  const f32x4 z4 = {0.0f, 0.0f, 0.0f, 0.0f};

  /* Fast K == 1 specialization (all 48 primary AttnRes + RMSNorm + Q8_0 prefill visits!). */
  if (K == 1u) {
    const f32x4 q0_0 = *(const G f32x4 *)(q + (th << 2u));
    const f32x4 q1_0 = *(const G f32x4 *)(q + 1024u + (th << 2u));
    const float q2_0 = q[2048u + th];

    for (u32 s = 0u; s < M; s++) {
      const G float *sp = (s < n_a) ? (a + (unsigned long)s * 2304ul) : c;
      const f32x4 x0 = *(const G f32x4 *)(sp + (th << 2u));
      const f32x4 x1 = *(const G f32x4 *)(sp + 1024u + (th << 2u));
      const float x2 = sp[2048u + th];
      float sq = ((x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w)) +
                 ((x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w)) + x2 * x2;
      sq = wave_tree_f(sq);
      const f32x4 p0 = x0 * q0_0, p1 = x1 * q1_0;
      float d0 = ((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w)) + x2 * q2_0;
      d0 = wave_tree_f(d0);
      const u32 rbase = s * 16u;
      if (l == 0u) {
        ink_r13_pf_ar_lds.u.ar_red[rbase + wv] = sq;
        ink_r13_pf_ar_lds.u.ar_red[rbase + 8u + wv] = d0;
      }
    }
    ink_r13_pf_lsignal();
    ink_r13_pf_lwait();

    float my_inv = 0.0f, lg0 = 0.0f;
    if (l < M) {
      const u32 rbase = l * 16u;
      float sum_sq = 0.0f, sd0 = 0.0f;
#pragma unroll
      for (u32 x = 0u; x < 8u; x++) {
        sum_sq += ink_r13_pf_ar_lds.u.ar_red[rbase + x];
        sd0 += ink_r13_pf_ar_lds.u.ar_red[rbase + 8u + x];
      }
      my_inv = 1.0f / __builtin_sqrtf(sum_sq / 2304.0f + eps);
      lg0 = sd0 * (my_inv * inv_sqn);
    }
    const float my_prob0 = ink_r13_pf_wave_softmax(lg0, l, M);
    ink_r13_pf_lsignal();
    ink_r13_pf_lwait();

    f32x4 a0_0 = z4, a1_0 = z4;
    float a2_0 = 0.0f;
    for (u32 s = 0u; s < M; s++) {
      const G float *sp = (s < n_a) ? (a + (unsigned long)s * 2304ul) : c;
      const f32x4 x0 = *(const G f32x4 *)(sp + (th << 2u));
      const f32x4 x1 = *(const G f32x4 *)(sp + 1024u + (th << 2u));
      const float x2 = sp[2048u + th];
      const float inv_s = I2F(__builtin_amdgcn_readlane(F2I(my_inv), (int)s));
      const float p0 = I2F(__builtin_amdgcn_readlane(F2I(my_prob0), (int)s));
      a0_0 += (x0 * inv_s) * p0;
      a1_0 += (x1 * inv_s) * p0;
      a2_0 += (x2 * inv_s) * p0;
    }
    ink_r13_pf_norm_2304(a0_0, a1_0, a2_0, nw0, y0, q0, eps, th, wv, l);
    return;
  }

  /* General K = 2..3 path (exact-SHA match to ie_inkling_attn_res_norm_q8_r7). */
  if (y1) y1 = TOK(y1, y1s);
  if (q1) q1 = TOK(q1, q1s);
  if (K > 2u) {
    if (y2) y2 = TOK(y2, y2s);
    if (q2) q2 = TOK(q2, q2s);
  }

  const f32x4 q0_0 = *(const G f32x4 *)(q + (th << 2u));
  const f32x4 q1_0 = *(const G f32x4 *)(q + 1024u + (th << 2u));
  const float q2_0 = q[2048u + th];
  const G float *qk1 = q + 2304u;
  const f32x4 q0_1 = *(const G f32x4 *)(qk1 + (th << 2u));
  const f32x4 q1_1 = *(const G f32x4 *)(qk1 + 1024u + (th << 2u));
  const float q2_1 = qk1[2048u + th];
  f32x4 q0_2 = z4, q1_2 = z4;
  float q2_2 = 0.0f;
  if (K > 2u) {
    const G float *qk2 = q + 4608u;
    q0_2 = *(const G f32x4 *)(qk2 + (th << 2u));
    q1_2 = *(const G f32x4 *)(qk2 + 1024u + (th << 2u));
    q2_2 = qk2[2048u + th];
  }

  const u32 stride_s = K + 1u;
  for (u32 s = 0u; s < M; s++) {
    const G float *sp = (s < n_a) ? (a + (unsigned long)s * 2304ul) : c;
    const f32x4 x0 = *(const G f32x4 *)(sp + (th << 2u));
    const f32x4 x1 = *(const G f32x4 *)(sp + 1024u + (th << 2u));
    const float x2 = sp[2048u + th];
    float sq = ((x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w)) +
               ((x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w)) + x2 * x2;
    sq = wave_tree_f(sq);
    const u32 rbase = s * stride_s * 8u;
    if (l == 0u) ink_r13_pf_ar_lds.u.ar_red[rbase + wv] = sq;
    {
      const f32x4 p0 = x0 * q0_0, p1 = x1 * q1_0;
      const float d0 = wave_tree_f(((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w)) + x2 * q2_0);
      if (l == 0u) ink_r13_pf_ar_lds.u.ar_red[rbase + 8u + wv] = d0;
    }
    {
      const f32x4 p0 = x0 * q0_1, p1 = x1 * q1_1;
      const float d1 = wave_tree_f(((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w)) + x2 * q2_1);
      if (l == 0u) ink_r13_pf_ar_lds.u.ar_red[rbase + 16u + wv] = d1;
    }
    if (K > 2u) {
      const f32x4 p0 = x0 * q0_2, p1 = x1 * q1_2;
      const float d2 = wave_tree_f(((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w)) + x2 * q2_2);
      if (l == 0u) ink_r13_pf_ar_lds.u.ar_red[rbase + 24u + wv] = d2;
    }
  }
  ink_r13_pf_lsignal();
  ink_r13_pf_lwait();

  float my_inv = 0.0f, lg0 = 0.0f, lg1 = 0.0f, lg2 = 0.0f;
  if (l < M) {
    const u32 rbase = l * stride_s * 8u;
    float sum_sq = 0.0f, sd0 = 0.0f, sd1 = 0.0f, sd2 = 0.0f;
#pragma unroll
    for (u32 x = 0u; x < 8u; x++) {
      sum_sq += ink_r13_pf_ar_lds.u.ar_red[rbase + x];
      sd0 += ink_r13_pf_ar_lds.u.ar_red[rbase + 8u + x];
      sd1 += ink_r13_pf_ar_lds.u.ar_red[rbase + 16u + x];
      if (K > 2u) sd2 += ink_r13_pf_ar_lds.u.ar_red[rbase + 24u + x];
    }
    my_inv = 1.0f / __builtin_sqrtf(sum_sq / 2304.0f + eps);
    const float sc = my_inv * inv_sqn;
    lg0 = sd0 * sc;
    lg1 = sd1 * sc;
    if (K > 2u) lg2 = sd2 * sc;
  }
  const float my_prob0 = ink_r13_pf_wave_softmax(lg0, l, M);
  const float my_prob1 = ink_r13_pf_wave_softmax(lg1, l, M);
  const float my_prob2 = (K > 2u) ? ink_r13_pf_wave_softmax(lg2, l, M) : 0.0f;
  ink_r13_pf_lsignal();
  ink_r13_pf_lwait();

  f32x4 a0_0 = z4, a1_0 = z4, a0_1 = z4, a1_1 = z4, a0_2 = z4, a1_2 = z4;
  float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f;
  for (u32 s = 0u; s < M; s++) {
    const G float *sp = (s < n_a) ? (a + (unsigned long)s * 2304ul) : c;
    const f32x4 x0 = *(const G f32x4 *)(sp + (th << 2u));
    const f32x4 x1 = *(const G f32x4 *)(sp + 1024u + (th << 2u));
    const float x2 = sp[2048u + th];
    const float inv_s = I2F(__builtin_amdgcn_readlane(F2I(my_inv), (int)s));
    const f32x4 xn0 = x0 * inv_s, xn1 = x1 * inv_s;
    const float xn2 = x2 * inv_s;
    const float p0 = I2F(__builtin_amdgcn_readlane(F2I(my_prob0), (int)s));
    const float p1 = I2F(__builtin_amdgcn_readlane(F2I(my_prob1), (int)s));
    a0_0 += xn0 * p0; a1_0 += xn1 * p0; a2_0 += xn2 * p0;
    a0_1 += xn0 * p1; a1_1 += xn1 * p1; a2_1 += xn2 * p1;
    if (K > 2u) {
      const float p2 = I2F(__builtin_amdgcn_readlane(F2I(my_prob2), (int)s));
      a0_2 += xn0 * p2; a1_2 += xn1 * p2; a2_2 += xn2 * p2;
    }
  }

  ink_r13_pf_norm_2304(a0_0, a1_0, a2_0, nw0, y0, q0, eps, th, wv, l);
  ink_r13_pf_lsignal();
  ink_r13_pf_lwait();
  ink_r13_pf_norm_2304(a0_1, a1_1, a2_1, nw1, y1, q1, eps, th, wv, l);
  if (K > 2u) {
    ink_r13_pf_lsignal();
    ink_r13_pf_lwait();
    ink_r13_pf_norm_2304(a0_2, a1_2, a2_2, nw2, y2, q2, eps, th, wv, l);
  }
}

/* ============================================================================
 * Kernel 5 (H10): ie_inkling_gemm_rq_exp_iq3xxs_flute_l0lut_r13
 *   L0$ cached LUT bypass variant of 110-VGPR flute_situ_r10:
 *   Reads ie_rq_iq3xxs_grid[256] directly via default cached L0$ loads
 *   (TH_LOAD_RT, 0 TH_LOAD_HT/NT), killing 100% of LUT LDS bank conflicts
 *   and shrinking LDS 19,712 B -> 18,688 B (<= 112 VGPR, 3 WGs/CU, exact-SHA!).
 * ========================================================================== */

typedef struct {
  v2i w0, w1;
  float s0;
} ink_r13_pf_fl_dc_t;

typedef struct {
  union {
    struct {
      u8 x[2][64u * 128u] __attribute__((aligned(16)));
      float sx[2][4u * 64u] __attribute__((aligned(32)));
    } k;
    float ep[2][32u][66u] __attribute__((aligned(32)));
  } u;
  u32 pid[64] __attribute__((aligned(16)));
} ink_r13_pf_fl_lds_t;

_Static_assert(sizeof(ink_r13_pf_fl_lds_t) == 18688u, "ink_r13_pf_fl_lds_t must be 18,688 B (-1 KiB vs flute_situ_r10)");

static inline __attribute__((always_inline)) u32 ink_r13_pf_fl_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

static inline __attribute__((always_inline)) v2i ink_r13_pf_fl_ksw(const u32 g0, const u32 g1, const u32 s7) {
  const u32 p_ = s7 * 0x00204081u;
  const u32 t0 = p_ & 0x01010101u;
  const u32 t1 = ((p_ >> 4u) | ((u32)__builtin_popcount(s7) << 24u)) & 0x01010101u;
  const u32 m0 = __builtin_amdgcn_perm(0u, 0u, t0 | 0x0C0C0C0Cu);
  const u32 m1 = __builtin_amdgcn_perm(0u, 0u, t1 | 0x0C0C0C0Cu);
  const v2i v = {(int)((g0 ^ m0) + t0), (int)((g1 ^ m1) + t1)};
  return v;
}

/* L0$ default-cached LUT decode: 4x global_load_b32 (TH_LOAD_RT, 0 LDS bank conflicts!). */
static inline __attribute__((always_inline)) void ink_r13_pf_fl_decode_l0(
    const u32 qp, const u32 auxv, const float d2, const u32 h, ink_r13_pf_fl_dc_t *dc) {
  const u32 g00 = ie_rq_iq3xxs_grid[qp & 255u];
  const u32 g01 = ie_rq_iq3xxs_grid[(qp >> 8u) & 255u];
  const u32 g10 = ie_rq_iq3xxs_grid[(qp >> 16u) & 255u];
  const u32 g11 = ie_rq_iq3xxs_grid[qp >> 24u];
  __builtin_amdgcn_sched_barrier(0);
  const u32 s_sh = h ? 14u : 0u;
  dc->s0 = d2 * (0.5f + (float)(auxv >> 28u));
  dc->w0 = ink_r13_pf_fl_ksw(g00, g01, (auxv >> s_sh) & 127u);
  dc->w1 = ink_r13_pf_fl_ksw(g10, g11, (auxv >> (s_sh + 7u)) & 127u);
}

static inline __attribute__((always_inline)) u32 ink_r13_pf_fl_qp0(const u32x4 qv, const u32 h) {
  const u32 q0 = qv.x, q1 = qv.y;
  const u32 q1_x = (u32)xrow_i((int)q1);
  return h ? q1_x : q0;
}

static inline __attribute__((always_inline)) u32 ink_r13_pf_fl_qp1(const u32x4 qv, const u32 h) {
  const u32 q2 = qv.z, q3 = qv.w;
  const u32 q3_x = (u32)xrow_i((int)q3);
  return h ? q3_x : q2;
}

static inline __attribute__((always_inline)) u32 ink_r13_pf_fl_qp2(const u32x4 qv, const u32 h) {
  const u32 q0 = qv.x, q1 = qv.y;
  const u32 q0_x = (u32)xrow_i((int)q0);
  return h ? q1 : q0_x;
}

static inline __attribute__((always_inline)) u32 ink_r13_pf_fl_qp3(const u32x4 qv, const u32 h) {
  const u32 q2 = qv.z, q3 = qv.w;
  const u32 q2_x = (u32)xrow_i((int)q2);
  return h ? q3 : q2_x;
}

static inline __attribute__((always_inline)) void ink_r13_pf_fl_accum_tile(
    const v8i r, const float s0, const LDSP float *sx0, f8 *acc_j) {
  const f32x4 dlo = *(const LDSP f32x4 *)(sx0);
  const f32x4 dhi = *(const LDSP f32x4 *)(sx0 + 4u);
  const f8 rf = (__builtin_bit_cast(f8, r) - 12582912.0f) * s0;
  __builtin_amdgcn_sched_barrier(0);
  const f8 dx = {dlo[0], dlo[1], dlo[2], dlo[3], dhi[0], dhi[1], dhi[2], dhi[3]};
  *acc_j += rf * dx;
  __asm__ volatile("" : : "v"(*acc_j) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r13_pf_fl_apply_one_j(
    const ink_r13_pf_fl_dc_t dck, const LDSP u8 *lx_k, const LDSP float *lsx_k,
    const u32 j0, f8 *acc) {
  const v8i zm = (v8i)(0x4B400000);
  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + j0 * 2048u);
  __builtin_amdgcn_sched_barrier(0);
  const v2i a0 = {(int)b0.x, (int)b0.y}, a1 = {(int)b0.z, (int)b0.w};
  v8i r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, dck.w0, zm, 0);
  r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, dck.w1, r0, 0);
  __builtin_amdgcn_sched_barrier(0);
  ink_r13_pf_fl_accum_tile(r0, dck.s0, lsx_k + 16u * j0, &acc[j0]);
}

static inline __attribute__((always_inline)) void ink_r13_pf_fl_apply_pair_j(
    const ink_r13_pf_fl_dc_t dck, const LDSP u8 *lx_k, const LDSP float *lsx_k,
    const u32 j0, f8 *acc) {
  const v8i zm = (v8i)(0x4B400000);
  const u32 j_off = j0 * 2048u;
  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + j_off);
  const u32x4 b1 = *(const LDSP u32x4 *)(lx_k + j_off + 2048u);
  __builtin_amdgcn_sched_barrier(0);
  const v2i a0_0 = {(int)b0.x, (int)b0.y}, a0_1 = {(int)b0.z, (int)b0.w};
  const v2i a1_0 = {(int)b1.x, (int)b1.y}, a1_1 = {(int)b1.z, (int)b1.w};
  v8i r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_0, 1, dck.w0, zm, 0);
  v8i r1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_0, 1, dck.w0, zm, 0);
  r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_1, 1, dck.w1, r0, 0);
  r1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_1, 1, dck.w1, r1, 0);
  __builtin_amdgcn_sched_barrier(0);
  ink_r13_pf_fl_accum_tile(r0, dck.s0, lsx_k + 16u * j0, &acc[j0]);
  ink_r13_pf_fl_accum_tile(r1, dck.s0, lsx_k + 16u * (j0 + 1u), &acc[j0 + 1u]);
}

static inline __attribute__((always_inline)) void ink_r13_pf_fl_kstep(
    const ink_r13_pf_fl_dc_t dck, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 k, const u32 lx_base0, const u32 h, const u32 wj, f8 *acc) {
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const u32 sx_str = (wj >= 3u) ? 64u : 32u;
  const LDSP float *lsx_k = lsx + k * sx_str + 8u * h;
  if (wj == 1u) {
    ink_r13_pf_fl_apply_one_j(dck, lx_k, lsx_k, 0u, acc);
  } else if (wj == 2u) {
    ink_r13_pf_fl_apply_pair_j(dck, lx_k, lsx_k, 0u, acc);
  } else if (wj == 3u) {
    ink_r13_pf_fl_apply_pair_j(dck, lx_k, lsx_k, 0u, acc);
    ink_r13_pf_fl_apply_one_j(dck, lx_k, lsx_k, 2u, acc);
  } else {
    ink_r13_pf_fl_apply_pair_j(dck, lx_k, lsx_k, 0u, acc);
    ink_r13_pf_fl_apply_pair_j(dck, lx_k, lsx_k, 2u, acc);
  }
}

static inline __attribute__((always_inline)) void ink_r13_pf_fl_stage_sx(
    LDSP float *lsx, const u32 stride, const u32 t0, const int is_s, const f32x4 rsx) {
  if (is_s) {
    u32 ti = t0;
    __asm__ volatile("" : "+v"(ti));
    lsx[0u * stride + ti] = rsx.x;
    lsx[1u * stride + ti] = rsx.y;
    lsx[2u * stride + ti] = rsx.z;
    lsx[3u * stride + ti] = rsx.w;
  }
}

/* Exact SITU (#pragma clang fp contract(off)) + SoA Q8_0 (quant_wave) epilogue strip. */
static inline __attribute__((always_inline)) void ink_r13_pf_fl_epilogue_strip(
    LDSP float *ep, const LDSP u32 *lpid, G u8 *Qout,
    const u32 is_up, const u32 r_loc, const u32 h, const u32 w, const u32 l,
    const u32 strip_base, const u32 nj_strip, const u32 blk0, const u32 nblk,
    const u32 nb_out, const u32 qstr, const float beta, const float lbeta,
    const u32 j_base, const f8 *acc) {
#pragma clang fp contract(off)
  LDSP float *ep_wr = ep + (is_up * 32u + 8u * h) * 66u + r_loc;
#pragma unroll
  for (u32 j_loc = 0u; j_loc < 2u; j_loc++) {
    if (16u * j_loc < nj_strip) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        ep_wr[(16u * j_loc + v) * 66u] = acc[j_base + j_loc][v];
      }
    }
  }
  lbarrier();

  const int use_beta = beta > 0.0f;
  const int use_lbeta = lbeta > 0.0f;
  for (u32 tk = w; tk < nj_strip; tk += 8u) {
    const u32 pid = (u32)__builtin_amdgcn_readfirstlane((int)lpid[strip_base + tk]);
    G u8 *qp = Qout + (unsigned long)pid * qstr;
    const LDSP float *ep_g = ep + tk * 66u + l;
#pragma unroll
    for (u32 b = 0u; b < 2u; b++) {
      if (b < nblk) {
        const float g = ep_g[32u * b];
        const float u = ep_g[2112u + 32u * b];
        const float gs = use_beta ? beta * ie_tanhf_s(g / beta) : g;
        const float us = use_lbeta ? lbeta * ie_tanhf_s(u / lbeta) : u;
        const float val = gs * sigmoidf(g) * us;
        quant_wave(val, blk0 + b, qp, nb_out);
      }
    }
  }
}

/* Shared L0$-LUT FLUTE K-loop body for bucket WJ in {1, 2, 3, 4} (100% exact flute_situ_r10
 * 4-row SoA layout + ping-pong q_cur/q_nxt schedule, reading ie_rq_iq3xxs_grid via default L0$!). */
static inline __attribute__((always_inline)) void ink_r13_pf_fl_run_wj(
    LDSP ink_r13_pf_fl_lds_t *lds, const G u8 *p_qs_e, const G u8 *p_aux_e, const G u8 *p_d_e,
    const G u8 *X_u, const G u8 *X_s, const G u32 *list,
    const u32 pb, const u32 nj, const u32 mb, const u32 xs, const u32 xdiv,
    const u32 w, const u32 t0, const u32 h, const u32 q_off, const u32 a_off, const u32 d_off,
    const u32 tk_ld, const u32 c8_ld, const u32 st_x, const u32 lx_base0,
    const u32 wj, f8 *acc) {
  const int ld_hi = (wj == 4u) || ((wj == 3u) && (tk_ld < 16u));
  const int is_s = (wj == 4u) ? (w < 2u) : (wj == 3u ? (t0 < 48u) : (wj == 2u ? (w == 0u) : (t0 < 16u)));
  const u32 sx_str = (wj >= 3u) ? 64u : 32u;

  if (is_s) {
    const u32 pj = pb + (t0 < nj ? t0 : 0u);
    lds->pid[t0] = list[pj];
  }
  lbarrier();

  const u32 xoff0 = ink_r13_pf_fl_div_x(lds->pid[tk_ld], xdiv) * xs + c8_ld * 16u;
  const u32 xoff1 = ld_hi ? (ink_r13_pf_fl_div_x(lds->pid[tk_ld + 32u], xdiv) * xs + c8_ld * 16u) : 0u;
  const u32 sxoff = is_s ? ink_r13_pf_fl_div_x(lds->pid[t0 & 63u], xdiv) * xs : 0u;

  u32x4 q_cur = *(const G u32x4 *)(p_qs_e + q_off);
  u32x4 a_cur = *(const G u32x4 *)(p_aux_e + a_off);
  u32 d_cur = (u32)(*(const G u16 *)(p_d_e + d_off));
  {
    const u32x4 rx0_i = ink_r13_pf_ld4(X_u + xoff0);
    const u32x4 rx1_i = ld_hi ? ink_r13_pf_ld4(X_u + xoff1) : (u32x4)(0u);
    const f32x4 rsx_i = is_s ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
    *(LDSP u32x4 *)(lds->u.k.x[0] + st_x) = rx0_i;
    if (ld_hi) *(LDSP u32x4 *)(lds->u.k.x[0] + st_x + 4096u) = rx1_i;
    ink_r13_pf_fl_stage_sx(lds->u.k.sx[0], sx_str, t0, is_s, rsx_i);
  }
  lbarrier();

  u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
  for (u32 sb = 0u; ; sb++) {
    const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
    const u32 kb1 = 8u * sb_u + 4u;
    const G u8 *p_qs_sb  = rq_ptr_u(p_qs_e  + (unsigned long)sb_u * 256ul);
    const G u8 *p_aux_sb = rq_ptr_u(p_aux_e + (unsigned long)sb_u * 128ul);

    /* -------- hb0 (k = 0..3) -------- */
    const float d2 = ie_rq_h2f(d_cur) * 0.5f;
    ink_r13_pf_fl_dc_t xdc;
    ink_r13_pf_fl_decode_l0(ink_r13_pf_fl_qp0(q_cur, h), a_cur.x, d2, h, &xdc);
    __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
    __builtin_amdgcn_sched_barrier(0);
    {
      const G u8 *X_hb1 = rq_ptr_u(X_u + (unsigned long)kb1 * 32ul);
      rx0 = ink_r13_pf_ld4(X_hb1 + xoff0);
      if (ld_hi) rx1 = ink_r13_pf_ld4(X_hb1 + xoff1);
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r13_pf_fl_kstep(xdc, lds->u.k.x[0], lds->u.k.sx[0], 0u, lx_base0, h, wj, acc);

    /* hb0 k=1: issue hb1 aligned 1x b128 qs into q_nxt (3 K-blocks + barrier hidden!). */
    ink_r13_pf_fl_decode_l0(ink_r13_pf_fl_qp1(q_cur, h), a_cur.y, d2, h, &xdc);
    const u32x4 q_nxt = *(const G u32x4 *)(p_qs_sb + q_off + 32u);
    __builtin_amdgcn_sched_barrier(0);
    ink_r13_pf_fl_kstep(xdc, lds->u.k.x[0], lds->u.k.sx[0], 1u, lx_base0, h, wj, acc);

    /* hb0 k=2 */
    ink_r13_pf_fl_decode_l0(ink_r13_pf_fl_qp2(q_cur, h), a_cur.z, d2, h, &xdc);
    __builtin_amdgcn_sched_barrier(0);
    ink_r13_pf_fl_kstep(xdc, lds->u.k.x[0], lds->u.k.sx[0], 2u, lx_base0, h, wj, acc);

    *(LDSP u32x4 *)(lds->u.k.x[1] + st_x) = rx0;
    if (ld_hi) *(LDSP u32x4 *)(lds->u.k.x[1] + st_x + 4096u) = rx1;
    f32x4 rsx = (f32x4)(0.0f);
    if (is_s) {
      const G u8 *Xs_hb1 = rq_ptr_u(X_s + (unsigned long)kb1 * 4ul);
      rsx = rq_load_f32x4((const G float *)(Xs_hb1 + sxoff));
    }
    __builtin_amdgcn_sched_barrier(0);

    /* hb0 k=3 */
    ink_r13_pf_fl_decode_l0(ink_r13_pf_fl_qp3(q_cur, h), a_cur.w, d2, h, &xdc);
    __builtin_amdgcn_sched_barrier(0);
    ink_r13_pf_fl_kstep(xdc, lds->u.k.x[0], lds->u.k.sx[0], 3u, lx_base0, h, wj, acc);

    ink_r13_pf_fl_stage_sx(lds->u.k.sx[1], sx_str, t0, is_s, rsx);
    a_cur = *(const G u32x4 *)(p_aux_sb + a_off + 16u);
    lbarrier();

    /* -------- hb1 (k = 0..3) -------- */
    const int more = (sb_u + 1u < mb);
    const u32 kbn = 8u * (sb_u + 1u);
    const G u8 *p_qs_nxt  = rq_ptr_u(p_qs_e  + (unsigned long)(sb_u + 1u) * 256ul);
    const G u8 *p_aux_nxt = rq_ptr_u(p_aux_e + (unsigned long)(sb_u + 1u) * 128ul);
    const G u8 *p_d_nxt   = rq_ptr_u(p_d_e   + (unsigned long)(sb_u + 1u) * 8ul);

    ink_r13_pf_fl_decode_l0(ink_r13_pf_fl_qp0(q_nxt, h), a_cur.x, d2, h, &xdc);
    __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
    __builtin_amdgcn_sched_barrier(0);
    if (more) {
      const G u8 *X_nxt = rq_ptr_u(X_u + (unsigned long)kbn * 32ul);
      rx0 = ink_r13_pf_ld4(X_nxt + xoff0);
      if (ld_hi) rx1 = ink_r13_pf_ld4(X_nxt + xoff1);
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r13_pf_fl_kstep(xdc, lds->u.k.x[1], lds->u.k.sx[1], 0u, lx_base0, h, wj, acc);

    /* hb1 k=1 */
    ink_r13_pf_fl_decode_l0(ink_r13_pf_fl_qp1(q_nxt, h), a_cur.y, d2, h, &xdc);
    if (more) q_cur = *(const G u32x4 *)(p_qs_nxt + q_off);
    __builtin_amdgcn_sched_barrier(0);
    ink_r13_pf_fl_kstep(xdc, lds->u.k.x[1], lds->u.k.sx[1], 1u, lx_base0, h, wj, acc);

    /* hb1 k=2 */
    ink_r13_pf_fl_decode_l0(ink_r13_pf_fl_qp2(q_nxt, h), a_cur.z, d2, h, &xdc);
    __builtin_amdgcn_sched_barrier(0);
    ink_r13_pf_fl_kstep(xdc, lds->u.k.x[1], lds->u.k.sx[1], 2u, lx_base0, h, wj, acc);

    f32x4 rsx1 = (f32x4)(0.0f);
    if (more) {
      *(LDSP u32x4 *)(lds->u.k.x[0] + st_x) = rx0;
      if (ld_hi) *(LDSP u32x4 *)(lds->u.k.x[0] + st_x + 4096u) = rx1;
      if (is_s) {
        const G u8 *Xs_nxt = rq_ptr_u(X_s + (unsigned long)kbn * 4ul);
        rsx1 = rq_load_f32x4((const G float *)(Xs_nxt + sxoff));
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    /* hb1 k=3 */
    ink_r13_pf_fl_decode_l0(ink_r13_pf_fl_qp3(q_nxt, h), a_cur.w, d2, h, &xdc);
    __builtin_amdgcn_sched_barrier(0);
    ink_r13_pf_fl_kstep(xdc, lds->u.k.x[1], lds->u.k.sx[1], 3u, lx_base0, h, wj, acc);

    if (!more) break;
    ink_r13_pf_fl_stage_sx(lds->u.k.sx[0], sx_str, t0, is_s, rsx1);
    a_cur = *(const G u32x4 *)(p_aux_nxt + a_off);
    d_cur = (u32)(*(const G u16 *)(p_d_nxt + d_off));
    lbarrier();
  }
}

KERNEL ie_inkling_gemm_rq_exp_iq3xxs_flute_l0lut_r13(
    const G u8 *qw, const G u8 *X, G u8 *Qout, u32 eff, u32 nb,
    const G u32 *list, const G u32 *tiles, const G u32 *n_tiles,
    u32 q_bytes, u32 xs, u32 qstr, u32 xdiv, float beta, float lbeta) {
  static LDS ink_r13_pf_fl_lds_t ink_r13_pf_fl_lds;
  if (!n_tiles || !tiles || !list || !qw || !X || !Qout) return;

  const u32 ti = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  const u32 nt_act = (u32)__builtin_amdgcn_readfirstlane((int)(*n_tiles));
  const u32 gbase = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() * 64u));
  if (ti >= nt_act || gbase >= eff || eff == 0u || (eff & 31u) != 0u || nb < 8u || (nb & 7u) != 0u || xdiv == 0u)
    return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)tiles[ti]);
  const u32 e = desc & 0xFFu;
  const u32 pb = (desc >> 8u) & 0x1FFFu;
  const u32 nj = (desc >> 21u) & 0x7Fu;
  if (nj == 0u || nj > 64u) return;

  const u32 cls = (nj - 1u) >> 4u; /* 3: 49..64, 2: 33..48, 1: 17..32, 0: 1..16 */
  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;

  const u32 is_up = w >> 2u;
  const u32 r_loc = 16u * (w & 3u) + r16;
  const u32 ch = gbase + r_loc;
  const u32 ch_c = ch < eff ? ch : eff - 1u;
  const u32 w_row = is_up ? (eff + ch_c) : ch_c;

  const u32 mb = nb >> 3u;
  const u32 nb_out = eff >> 5u;
  const u32 blk0 = gbase >> 5u;
  const u32 nblk = (eff - gbase >= 64u) ? 2u : 1u;

  /* FLUTE 3-plane SGPR base pointers (exact 98 B/row/sb, 128-B aligned!). */
  const unsigned long n_sb_tot = (unsigned long)(eff >> 1u) * (unsigned long)mb;
  const G u8 *p_qs_e  = rq_ptr_u(qw + (unsigned long)e * (unsigned long)q_bytes);
  const G u8 *p_aux_e = rq_ptr_u(p_qs_e + n_sb_tot * 256ul);
  const G u8 *p_d_e   = rq_ptr_u(p_qs_e + n_sb_tot * 384ul);
  const G u8 *X_u = rq_ptr_u(X);
  const G u8 *X_s = rq_ptr_u(X + 32ul * nb);

  const u32 r_idx = ((w_row >> 2u) * mb) * 4u + (w_row & 3u);
  const u32 q_off = (r_idx << 6u) + (h << 4u);
  const u32 a_off = r_idx << 5u;
  const u32 d_off = r_idx << 1u;

  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  const u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);
  const u32 lx_base0 = r16 * 128u + ((h ^ (r16 & 7u)) << 4u);

  f8 acc[4];
#pragma unroll
  for (u32 j = 0u; j < 4u; j++) acc[j] = (f8)(0.0f);

  if (cls == 3u) {
    ink_r13_pf_fl_run_wj(&ink_r13_pf_fl_lds, p_qs_e, p_aux_e, p_d_e, X_u, X_s, list,
                         pb, nj, mb, xs, xdiv, w, t0, h, q_off, a_off, d_off, tk_ld, c8_ld, st_x, lx_base0, 4u, acc);
  } else if (cls == 2u) {
    ink_r13_pf_fl_run_wj(&ink_r13_pf_fl_lds, p_qs_e, p_aux_e, p_d_e, X_u, X_s, list,
                         pb, nj, mb, xs, xdiv, w, t0, h, q_off, a_off, d_off, tk_ld, c8_ld, st_x, lx_base0, 3u, acc);
  } else if (cls == 1u) {
    ink_r13_pf_fl_run_wj(&ink_r13_pf_fl_lds, p_qs_e, p_aux_e, p_d_e, X_u, X_s, list,
                         pb, nj, mb, xs, xdiv, w, t0, h, q_off, a_off, d_off, tk_ld, c8_ld, st_x, lx_base0, 2u, acc);
  } else {
    ink_r13_pf_fl_run_wj(&ink_r13_pf_fl_lds, p_qs_e, p_aux_e, p_d_e, X_u, X_s, list,
                         pb, nj, mb, xs, xdiv, w, t0, h, q_off, a_off, d_off, tk_ld, c8_ld, st_x, lx_base0, 1u, acc);
  }

  lbarrier();
  const u32 nj0 = nj < 32u ? nj : 32u;
  ink_r13_pf_fl_epilogue_strip(&ink_r13_pf_fl_lds.u.ep[0][0][0], ink_r13_pf_fl_lds.pid, Qout,
                               is_up, r_loc, h, w, l, 0u, nj0, blk0, nblk, nb_out, qstr, beta, lbeta, 0u, acc);
  if (nj > 32u) {
    lbarrier();
    ink_r13_pf_fl_epilogue_strip(&ink_r13_pf_fl_lds.u.ep[0][0][0], ink_r13_pf_fl_lds.pid, Qout,
                                 is_up, r_loc, h, w, l, 32u, nj - 32u, blk0, nblk, nb_out, qstr, beta, lbeta, 2u, acc);
  }
}

/* ============================================================================
 * Kernel 6 (H08 R49 Fix): ie_inkling_gemm_f32_router_bf16_r13
 *   Scaled 3-WMMA FP16-Pair Split (W*64, X*1 -> *1/64, 22-bit mantissa via
 *   v_wmma_f32_16x16x16_f16_w32_gfx12 + hardware v_cvt_f16_f32 -> 18.7x lower
 *   RMS error than R10 BF16-pair!) on (BM=32, BN=32, BK=64), 2 N-tiles/wave,
 *   <= 96 VGPR (4 WGs/CU!), 16,384 B LDS (4 WGs/CU!), fused OP_MOE_ROUTE!
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r13_pf_split_f4_f16(
    const f32x4 x, const float sc, ink_r13_pf_u32x2 *hi, ink_r13_pf_u32x2 *lo) {
  float a0 = x.x * sc, a1 = x.y * sc, a2 = x.z * sc, a3 = x.w * sc;
  if (sc != 1.0f) {
    __asm__ volatile("" : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3));
  }
  const f16 h0 = (f16)a0, h1 = (f16)a1, h2 = (f16)a2, h3 = (f16)a3;
  float hf0 = (float)h0, hf1 = (float)h1, hf2 = (float)h2, hf3 = (float)h3;
  __asm__ volatile("" : "+v"(hf0), "+v"(hf1), "+v"(hf2), "+v"(hf3));
  const float r0 = a0 - hf0, r1 = a1 - hf1, r2 = a2 - hf2, r3 = a3 - hf3;
  const f16 l0 = (f16)r0, l1 = (f16)r1, l2 = (f16)r2, l3 = (f16)r3;
  const u16 hu0 = __builtin_bit_cast(u16, h0), hu1 = __builtin_bit_cast(u16, h1);
  const u16 hu2 = __builtin_bit_cast(u16, h2), hu3 = __builtin_bit_cast(u16, h3);
  const u16 lu0 = __builtin_bit_cast(u16, l0), lu1 = __builtin_bit_cast(u16, l1);
  const u16 lu2 = __builtin_bit_cast(u16, l2), lu3 = __builtin_bit_cast(u16, l3);
  *hi = (ink_r13_pf_u32x2){(u32)hu0 | ((u32)hu1 << 16u), (u32)hu2 | ((u32)hu3 << 16u)};
  *lo = (ink_r13_pf_u32x2){(u32)lu0 | ((u32)lu1 << 16u), (u32)lu2 | ((u32)lu3 << 16u)};
}

static inline __attribute__((always_inline)) void ink_r13_pf_split_w8_f16(
    const f32x4 w0, const f32x4 w1, h8 *whi, h8 *wlo) {
  ink_r13_pf_u32x2 h0, l0, h1, l1;
  ink_r13_pf_split_f4_f16(w0, 64.0f, &h0, &l0);
  ink_r13_pf_split_f4_f16(w1, 64.0f, &h1, &l1);
  const u32x4 hp = {h0.x, h0.y, h1.x, h1.y};
  const u32x4 lp = {l0.x, l0.y, l1.x, l1.y};
  *whi = __builtin_bit_cast(h8, hp);
  *wlo = __builtin_bit_cast(h8, lp);
}

typedef union {
  struct {
    u32 x_hi[2u][INK_R13_PF_RTR_BN * (INK_R13_PF_RTR_BK / 2u)] __attribute__((aligned(16))); /* 8,192 B */
    u32 x_lo[2u][INK_R13_PF_RTR_BN * (INK_R13_PF_RTR_BK / 2u)] __attribute__((aligned(16))); /* 8,192 B */
  } x;                                                                                       /* 16,384 B */
  float red[4u][INK_R13_PF_RTR_BN][INK_R13_PF_RTR_BM] __attribute__((aligned(16)));          /* 16,384 B */
  u32 is_last;                                                                               /*      4 B */
} ink_r13_pf_rtr_lds_t;

_Static_assert(sizeof(ink_r13_pf_rtr_lds_t) == 16384u, "ink_r13_pf_rtr_lds_t must be 16 KiB (4 WGs/CU)");

KERNEL ie_inkling_gemm_f32_router_bf16_r13(
    const G float *w, const G float *x, G float *y,
    u32 rows, u32 cols, const G float *bias, const G float *res,
    u32 xs, u32 ys, u32 rs, u32 T,
    u32 K, const G float *ebias, float scale,
    G u32 *route, u32 ost, G u32 *count) {
  static LDS ink_r13_pf_rtr_lds_t ink_r13_pf_rtr_lds __attribute__((aligned(16)));

  const u32 mg = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  const u32 nt = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  const u32 m0 = mg * INK_R13_PF_RTR_BM;
  const u32 n0 = nt * INK_R13_PF_RTR_BN;
  if (m0 >= rows || n0 >= T || cols < 16u || (cols & 15u) != 0u || !w || !x || !y) return;

  const u32 t = tid();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;

  /* 8 waves = 2(M: wm in 0..1) x 4(Split-K: wk in 0..3); each wave covers 2 N-tiles (32 toks). */
  const u32 wm = wv & 1u;
  const u32 wk = wv >> 1u;

  /* Cooperative X loader: 256 threads load 32 tokens x 64 floats (2x f32x4/thread). */
  const u32 xt_ld = t >> 4u;
  const u32 xc4 = t & 15u;
  const u32 tk_ld0 = n0 + xt_ld;
  const u32 tk_ld1 = tk_ld0 + 16u;
  const int tk0_ok = (tk_ld0 < T);
  const int tk1_ok = (tk_ld1 < T);
  const G float *x_ptr0 = (const G float *)((const G u8 *)x + (unsigned long)(tk0_ok ? tk_ld0 : (T - 1u)) * xs);
  const G float *x_ptr1 = (const G float *)((const G u8 *)x + (unsigned long)(tk1_ok ? tk_ld1 : (T - 1u)) * xs);
  const u32 x_swz_off0 = xt_ld * 32u + (((xc4 >> 1u) ^ (xt_ld & 7u)) << 2u) + ((xc4 & 1u) << 1u);
  const u32 x_swz_off1 = x_swz_off0 + 512u;

  /* Direct-VGPR W loader: lane (r16, h) of wave (wm, wk) loads 8 floats ONCE for both N-tiles. */
  const u32 r_w = m0 + (wm << 4u) + r16;
  const int r_ok = (r_w < rows);
  const G float *w_ptr = w + (unsigned long)(r_ok ? r_w : (rows - 1u)) * cols;
  const u32 wk_off = (wk << 4u) + (h << 3u);

  /* LDS read offsets for X tiles: token n_sub*16 + r16, 16-B chunk (2*wk + h) ^ (r16 & 7). */
  const u32 x_rd_off0 = r16 * 32u + (((wk << 1u) + h) ^ (r16 & 7u)) * 4u;
  const u32 x_rd_off1 = x_rd_off0 + 512u;

  const f32x4 z4 = {0.0f, 0.0f, 0.0f, 0.0f};
  v8f acc0 = (v8f)(0.0f);
  v8f acc1 = (v8f)(0.0f);

  /* Prologue: load c0 = 0 for W -> VGPRs and X (32 tokens) -> LDS[0]. */
  f32x4 vw0 = (r_ok && wk_off < cols) ? *(const G f32x4 *)(w_ptr + wk_off + 0u) : z4;
  f32x4 vw1 = (r_ok && wk_off + 4u < cols) ? *(const G f32x4 *)(w_ptr + wk_off + 4u) : z4;
  {
    const u32 kc_x = xc4 << 2u;
    const int cok = (kc_x < cols);
    const f32x4 vx0 = (tk0_ok && cok) ? *(const G f32x4 *)(x_ptr0 + kc_x) : z4;
    const f32x4 vx1 = (tk1_ok && cok) ? *(const G f32x4 *)(x_ptr1 + kc_x) : z4;
    ink_r13_pf_u32x2 xh0, xl0, xh1, xl1;
    ink_r13_pf_split_f4_f16(vx0, 1.0f, &xh0, &xl0);
    ink_r13_pf_split_f4_f16(vx1, 1.0f, &xh1, &xl1);
    *(LDSP ink_r13_pf_u32x2 *)(ink_r13_pf_rtr_lds.x.x_hi[0] + x_swz_off0) = xh0;
    *(LDSP ink_r13_pf_u32x2 *)(ink_r13_pf_rtr_lds.x.x_lo[0] + x_swz_off0) = xl0;
    *(LDSP ink_r13_pf_u32x2 *)(ink_r13_pf_rtr_lds.x.x_hi[0] + x_swz_off1) = xh1;
    *(LDSP ink_r13_pf_u32x2 *)(ink_r13_pf_rtr_lds.x.x_lo[0] + x_swz_off1) = xl1;
  }
  ink_r13_pf_lsignal();
  ink_r13_pf_lwait();

  u32 bf = 0u;
#pragma unroll 1
  for (u32 c0 = 0u; c0 < cols; c0 += INK_R13_PF_RTR_BK) {
    const u32 c_n = c0 + INK_R13_PF_RTR_BK;
    const int more = (c_n < cols);
    f32x4 nw0 = z4, nw1 = z4, nvx0 = z4, nvx1 = z4;
    if (more) {
      const u32 kw_n = c_n + wk_off;
      const u32 kx_n = c_n + (xc4 << 2u);
      if (r_ok && kw_n < cols) {
        nw0 = *(const G f32x4 *)(w_ptr + kw_n + 0u);
        nw1 = *(const G f32x4 *)(w_ptr + kw_n + 4u);
      }
      if (kx_n < cols) {
        if (tk0_ok) nvx0 = *(const G f32x4 *)(x_ptr0 + kx_n);
        if (tk1_ok) nvx1 = *(const G f32x4 *)(x_ptr1 + kx_n);
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    h8 whi, wlo;
    ink_r13_pf_split_w8_f16(vw0, vw1, &whi, &wlo);

    /* N-tile 0 (tokens n0 + 0..15): 3 FP16-pair WMMAs. */
    {
      const h8 xhi0 = *(const LDSP h8 *)(ink_r13_pf_rtr_lds.x.x_hi[bf] + x_rd_off0);
      const h8 xlo0 = *(const LDSP h8 *)(ink_r13_pf_rtr_lds.x.x_lo[bf] + x_rd_off0);
      acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(whi, xhi0, acc0);
      acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(whi, xlo0, acc0);
      acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(wlo, xhi0, acc0);
    }
    __builtin_amdgcn_sched_barrier(0);

    /* Load N-tile 1 X from active LDS[bf], then stage next X -> inactive LDS[nbf] & signal! */
    const h8 xhi1 = *(const LDSP h8 *)(ink_r13_pf_rtr_lds.x.x_hi[bf] + x_rd_off1);
    const h8 xlo1 = *(const LDSP h8 *)(ink_r13_pf_rtr_lds.x.x_lo[bf] + x_rd_off1);
    __builtin_amdgcn_sched_barrier(0);

    if (more) {
      const u32 nbf = bf ^ 1u;
      ink_r13_pf_u32x2 xh0_n, xl0_n, xh1_n, xl1_n;
      ink_r13_pf_split_f4_f16(nvx0, 1.0f, &xh0_n, &xl0_n);
      ink_r13_pf_split_f4_f16(nvx1, 1.0f, &xh1_n, &xl1_n);
      *(LDSP ink_r13_pf_u32x2 *)(ink_r13_pf_rtr_lds.x.x_hi[nbf] + x_swz_off0) = xh0_n;
      *(LDSP ink_r13_pf_u32x2 *)(ink_r13_pf_rtr_lds.x.x_lo[nbf] + x_swz_off0) = xl0_n;
      *(LDSP ink_r13_pf_u32x2 *)(ink_r13_pf_rtr_lds.x.x_hi[nbf] + x_swz_off1) = xh1_n;
      *(LDSP ink_r13_pf_u32x2 *)(ink_r13_pf_rtr_lds.x.x_lo[nbf] + x_swz_off1) = xl1_n;
      /* Signal split barrier BEFORE N-tile 1's 3 FP16 WMMAs (100% hidden behind compute!). */
      ink_r13_pf_lsignal();
    }

    acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(whi, xhi1, acc1);
    acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(whi, xlo1, acc1);
    acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(wlo, xhi1, acc1);

    if (more) {
      vw0 = nw0;
      vw1 = nw1;
      ink_r13_pf_lwait();
      bf ^= 1u;
    }
  }

  /* Split-K reduction across 4 K-waves via unioned red[4][32][32] (16,384 B exact!). */
  ink_r13_pf_lsignal();
  ink_r13_pf_lwait();
  {
    const u32 r_loc = (wm << 4u) + (h << 3u);
    LDSP f32x4 *rp0 = (LDSP f32x4 *)(ink_r13_pf_rtr_lds.red[wk][r16] + r_loc);
    rp0[0] = (f32x4){acc0[0], acc0[1], acc0[2], acc0[3]};
    rp0[1] = (f32x4){acc0[4], acc0[5], acc0[6], acc0[7]};
    LDSP f32x4 *rp1 = (LDSP f32x4 *)(ink_r13_pf_rtr_lds.red[wk][16u + r16] + r_loc);
    rp1[0] = (f32x4){acc1[0], acc1[1], acc1[2], acc1[3]};
    rp1[1] = (f32x4){acc1[4], acc1[5], acc1[6], acc1[7]};
  }
  ink_r13_pf_lsignal();
  ink_r13_pf_lwait();

  /* ALL 8 waves (256 threads = 32 tokens x 8 f32x4/token): reduce, exact 2^-6 unscale, store. */
  {
    const u32 tk_loc = t >> 3u;
    const u32 r4_loc = (t & 7u) << 2u;
    const u32 tk = n0 + tk_loc;
    if (tk < T) {
      const f32x4 v0 = *(const LDSP f32x4 *)(ink_r13_pf_rtr_lds.red[0][tk_loc] + r4_loc);
      const f32x4 v1 = *(const LDSP f32x4 *)(ink_r13_pf_rtr_lds.red[1][tk_loc] + r4_loc);
      const f32x4 v2 = *(const LDSP f32x4 *)(ink_r13_pf_rtr_lds.red[2][tk_loc] + r4_loc);
      const f32x4 v3 = *(const LDSP f32x4 *)(ink_r13_pf_rtr_lds.red[3][tk_loc] + r4_loc);
      f32x4 out4 = ((v0 + v1) + (v2 + v3)) * 0.015625f; /* exact 2^-6 power-of-2 unscale */
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

  /* Optional fused L2-hot OP_MOE_ROUTE: last arriving M-WG routes all <= 32 tokens across 8 waves. */
  if (!route || !count || rows > 256u || K == 0u || K > 32u) return;

  const u32 ng_m = (rows + INK_R13_PF_RTR_BM - 1u) / INK_R13_PF_RTR_BM;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier(); /* red[] is now 100% dead; safe to reuse union for is_last! */
  if (t == 0u) {
    ink_r13_pf_rtr_lds.is_last =
        (__scoped_atomic_fetch_add(&count[nt], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng_m - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r13_pf_rtr_lds.is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const u32 nj_t = (T - n0 < INK_R13_PF_RTR_BN) ? (T - n0) : INK_R13_PF_RTR_BN;
#pragma clang loop unroll(disable)
  for (u32 dt = wv; dt < nj_t; dt += 8u) {
    const u32 tk = n0 + dt;
    const G float *lg_tk = (const G float *)((const G u8 *)y + (unsigned long)tk * ys);
    G u32 *rt_tk = (G u32 *)((G u8 *)route + (unsigned long)tk * ost);
    moe_route8(lg_tk, rows, K, ebias, scale, rt_tk);
  }
  barrier();
  if (t == 0u) __scoped_atomic_store_n(&count[nt], 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
}

/* ============================================================================
 * Kernel 7 (H08 Exact-SHA Companion): ie_inkling_gemm_f32_router_exact_r13
 *   0-ULP bitwise-identical sequential k=0..cols-1 FP32 FMA router on
 *   (BM=32, BN=32, BK=32), <= 64 VGPR (4 WGs/CU!), 16,384 B unioned LDS
 *   (4 WGs/CU!), decoupled split barriers, + fused L2-hot OP_MOE_ROUTE!
 *   Grid: ((rows + 31u) / 32u, (T + 31u) / 32u, 1u) = 64 WGs@512 / 128@1024!
 * ========================================================================== */

typedef union {
  struct {
    float w[2u][INK_R13_PF_EXR_BK][INK_R13_PF_RTR_BM] __attribute__((aligned(16))); /*  8,192 B */
    float x[2u][INK_R13_PF_EXR_BK][INK_R13_PF_RTR_BN] __attribute__((aligned(16))); /*  8,192 B */
  } wx;                                                                             /* 16,384 B */
  u32 is_last;                                                                      /*      4 B */
} ink_r13_pf_exr_lds_t;

_Static_assert(sizeof(ink_r13_pf_exr_lds_t) == 16384u, "ink_r13_pf_exr_lds_t must be 16 KiB (4 WGs/CU)");

static inline __attribute__((always_inline)) void ink_r13_pf_exr_store(
    LDSP ink_r13_pf_exr_lds_t *lds, const u32 buf, const u32 c4, const u32 sm,
    const f32x4 vw, const f32x4 vx) {
  const u32 k0 = c4 << 2u;
  lds->wx.w[buf][k0 + 0u][sm] = vw.x;
  lds->wx.w[buf][k0 + 1u][sm] = vw.y;
  lds->wx.w[buf][k0 + 2u][sm] = vw.z;
  lds->wx.w[buf][k0 + 3u][sm] = vw.w;
  lds->wx.x[buf][k0 + 0u][sm] = vx.x;
  lds->wx.x[buf][k0 + 1u][sm] = vx.y;
  lds->wx.x[buf][k0 + 2u][sm] = vx.z;
  lds->wx.x[buf][k0 + 3u][sm] = vx.w;
}

static inline __attribute__((always_inline)) void ink_r13_pf_exr_mma_half(
    const LDSP ink_r13_pf_exr_lds_t *lds, const u32 buf, const u32 k_start, const u32 k_end,
    const u32 rm, const u32 rn, float acc[2][2]) {
#pragma clang loop unroll(disable)
  for (u32 k0 = k_start; k0 < k_end; k0 += 4u) {
    const u32 sm = rm ^ k0;
    const u32 sn = rn ^ k0;
#pragma unroll
    for (u32 dk = 0u; dk < 4u; dk++) {
      const u32 k = k0 + dk;
      const ink_r13_pf_f2 aw = *(const LDSP ink_r13_pf_f2 *)(&lds->wx.w[buf][k][sm]);
      const ink_r13_pf_f2 bx = *(const LDSP ink_r13_pf_f2 *)(&lds->wx.x[buf][k][sn]);
      acc[0][0] = __builtin_fmaf(aw.x, bx.x, acc[0][0]);
      acc[0][1] = __builtin_fmaf(aw.x, bx.y, acc[0][1]);
      acc[1][0] = __builtin_fmaf(aw.y, bx.x, acc[1][0]);
      acc[1][1] = __builtin_fmaf(aw.y, bx.y, acc[1][1]);
    }
    __builtin_amdgcn_sched_barrier(0);
  }
}

KERNEL ie_inkling_gemm_f32_router_exact_r13(
    const G float *w, const G float *x, G float *y,
    u32 rows, u32 cols, const G float *bias, const G float *res,
    u32 xs, u32 ys, u32 rs, u32 T,
    u32 K, const G float *ebias, float scale,
    G u32 *route, u32 ost, G u32 *count) {
  static LDS ink_r13_pf_exr_lds_t ink_r13_pf_exr_lds __attribute__((aligned(16)));

  const u32 mg = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  const u32 nt = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  const u32 m0 = mg * INK_R13_PF_RTR_BM;
  const u32 n0 = nt * INK_R13_PF_RTR_BN;
  if (m0 >= rows || n0 >= T || cols == 0u || !w || !x || !y) return;

  const u32 t = tid();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = lane();

  /* Cooperative f32x4 tile loaders: 8 threads along K (32 floats) x 32 rows/tokens. */
  const u32 lrow = t >> 3u;
  const u32 c4 = t & 7u;
  const u32 sm = lrow ^ (c4 << 2u);

  const u32 r_ld = m0 + lrow;
  const u32 tk_ld = n0 + lrow;
  const int r_ok = (r_ld < rows);
  const int tk_ok = (tk_ld < T);

  const G float *w_ptr = w + (unsigned long)(r_ok ? r_ld : (rows - 1u)) * cols;
  const G float *x_ptr = (const G float *)((const G u8 *)x + (unsigned long)(tk_ok ? tk_ld : (T - 1u)) * xs);

  /* Wave/lane output tile: 8 waves = 2(M) x 4(N) of 16x8; each lane computes 2(M) x 2(N). */
  const u32 wm = (wv & 1u) * 16u;
  const u32 wn = (wv >> 1u) * 8u;
  const u32 rm = wm + (l >> 2u) * 2u;
  const u32 rn = wn + (l & 3u) * 2u;

  float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  const f32x4 z4 = {0.0f, 0.0f, 0.0f, 0.0f};

  {
    const u32 c = c4 << 2u;
    const int cok = (c < cols);
    const f32x4 vw = (r_ok && cok) ? *(const G f32x4 *)(w_ptr + c) : z4;
    const f32x4 vx = (tk_ok && cok) ? *(const G f32x4 *)(x_ptr + c) : z4;
    ink_r13_pf_exr_store(&ink_r13_pf_exr_lds, 0u, c4, sm, vw, vx);
  }
  ink_r13_pf_lsignal();
  ink_r13_pf_lwait();

  u32 buf = 0u;
#pragma unroll 1
  for (u32 c0 = INK_R13_PF_EXR_BK; c0 < cols; c0 += INK_R13_PF_EXR_BK) {
    const u32 c = c0 + (c4 << 2u);
    const int cok = (c < cols);
    const f32x4 vw = (r_ok && cok) ? *(const G f32x4 *)(w_ptr + c) : z4;
    const f32x4 vx = (tk_ok && cok) ? *(const G f32x4 *)(x_ptr + c) : z4;
    __builtin_amdgcn_sched_barrier(0);

    /* Exact sequential k = 0..31 (all LDS[buf] reads complete BEFORE storing LDS[buf^1]!). */
    ink_r13_pf_exr_mma_half(&ink_r13_pf_exr_lds, buf, 0u, INK_R13_PF_EXR_BK, rm, rn, acc);
    ink_r13_pf_exr_store(&ink_r13_pf_exr_lds, buf ^ 1u, c4, sm, vw, vx);
    ink_r13_pf_lsignal();
    ink_r13_pf_lwait();
    buf ^= 1u;
  }

  ink_r13_pf_exr_mma_half(&ink_r13_pf_exr_lds, buf, 0u, INK_R13_PF_EXR_BK, rm, rn, acc);

  /* Coalesced f32x2 stores (exact epilogue order). */
  const u32 r_o = m0 + rm;
  const int vec_y = (r_o + 2u <= rows) && ((ys & 7u) == 0u) && (!res || (rs & 7u) == 0u) &&
                    (!bias || ((((unsigned long)bias) & 7ul) == 0ul));
#pragma unroll
  for (u32 j = 0u; j < 2u; j++) {
    const u32 tk = n0 + rn + j;
    if (tk < T) {
      G float *yp = (G float *)((G u8 *)y + (unsigned long)tk * ys);
      if (__builtin_expect(vec_y, 1)) {
        ink_r13_pf_f2 out2 = {acc[0][j], acc[1][j]};
        if (bias) out2 += *(const G ink_r13_pf_f2 *)(bias + r_o);
        if (res) out2 += *(const G ink_r13_pf_f2 *)((const G float *)((const G u8 *)res + (unsigned long)tk * rs) + r_o);
        *(G ink_r13_pf_f2 *)(yp + r_o) = out2;
      } else {
        const G float *rp = res ? (const G float *)((const G u8 *)res + (unsigned long)tk * rs) : (const G float *)0;
        if (r_o + 0u < rows) yp[r_o + 0u] = epilogue(acc[0][j], r_o + 0u, bias, rp);
        if (r_o + 1u < rows) yp[r_o + 1u] = epilogue(acc[1][j], r_o + 1u, bias, rp);
      }
    }
  }

  /* Optional fused L2-hot OP_MOE_ROUTE: last arriving M-WG routes all <= 32 tokens across 8 waves. */
  if (!route || !count || rows > 256u || K == 0u || K > 32u) return;

  const u32 ng_m = (rows + INK_R13_PF_RTR_BM - 1u) / INK_R13_PF_RTR_BM;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier(); /* wx[] is now 100% dead; safe to reuse union for is_last! */
  if (t == 0u) {
    ink_r13_pf_exr_lds.is_last =
        (__scoped_atomic_fetch_add(&count[nt], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng_m - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r13_pf_exr_lds.is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const u32 nj_t = (T - n0 < INK_R13_PF_RTR_BN) ? (T - n0) : INK_R13_PF_RTR_BN;
#pragma clang loop unroll(disable)
  for (u32 dt = wv; dt < nj_t; dt += 8u) {
    const u32 tk = n0 + dt;
    const G float *lg_tk = (const G float *)((const G u8 *)y + (unsigned long)tk * ys);
    G u32 *rt_tk = (G u32 *)((G u8 *)route + (unsigned long)tk * ost);
    moe_route8(lg_tk, rows, K, ebias, scale, rt_tk);
  }
  barrier();
  if (t == 0u) __scoped_atomic_store_n(&count[nt], 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
}

#undef INK_R13_PF_EXR_BK
#undef INK_R13_PF_RTR_BK
#undef INK_R13_PF_RTR_BN
#undef INK_R13_PF_RTR_BM
#undef INK_R13_PF_AR_SLOTS
#undef INK_R13_PF_AR_MAXM
#undef INK_R13_PF_SC_TTILE
#undef INK_R13_PF_KV_TTILE
#undef INK_R13_PF_H_SWSTR
#undef INK_R13_PF_M_T
#undef INK_R13_PF_H_R
#undef INK_R13_PF_KB

#endif /* !IE_MEGA */
