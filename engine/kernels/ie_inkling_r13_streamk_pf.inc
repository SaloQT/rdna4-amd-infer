/* ============================================================================
 * Inkling-2 Round 13B (Wave 13 - H11 Hybrid Stream-K Dense Q8_0 Prefill &
 * H05b FlashAttention-4 Polynomial FMA exp2 Interleaved Prefill Attention)
 * for RX 9070 XT (gfx1201, wave32, 64 CUs, NT=256).
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r13_sk_ /
 * INK_R13_SK_ and all macros are #undef'd at EOF.
 *
 * gfx1201 Hardware Occupancy (24-VGPR wave32 granule, 1536 VGPR/SIMD32):
 *   <= 120 VGPR -> 12w/SIMD (3 WGs/CU -> 192 resident WGs across 64 CUs!)
 *
 * Exported Kernels, Grids, Guards, and Parity:
 *
 * 1. KERNEL ie_inkling_gemm_q8_med_sk_r13(
 *        const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
 *        u32 rows, u32 nb, const G float *bias, const G float *res,
 *        u32 T, u32 xs, u32 ys, u32 rs,
 *        G float *sk_part, G u32 *sk_flag, u32 sk_wgs)
 *    - H11 Hybrid Stream-K Dense Q8_0 128x64 Prefill WMMA:
 *      Eliminates the 25-33% tail-wave idle bubble across 192 resident WGs
 *      (64 CUs x 3 WGs/CU) on attn_o (144 tiles), wqkv (320 tiles = 192 DP +
 *      128 SK), and wk/wv (64 tiles at 128x64).
 *      - Phase 1: Full data-parallel waves for floor(total_tiles / sk_wgs) *
 *        sk_wgs tiles (4x4 L2/MALL swizzled, direct store, 0 scratch).
 *      - Phase 2: Lock-free, ZERO-spin-wait (100% WDDM deadlock-proof!)
 *        Stream-K split across all sk_wgs WGs for remainder sk_tiles =
 *        total_tiles % sk_wgs. Each split tile's last-arriving WG (tracked
 *        by self-resetting sk_flag[sk_t]) sums peer partials from sk_part
 *        in exact ascending K-order (deterministic f32-tol ~1e-6) and stores
 *        y with bias/res!
 *    - Grid:      (sk_wgs, 1u, 1u), NT = 256 (default sk_wgs = 192).
 *    - Strides:   xs in BYTES; ys, rs in FLOATS (matching ie_gemm_q8r).
 *    - Scratch:   sk_part >= 2 * sk_wgs * 8192 floats (3,145,728 floats =
 *                 12.0 MiB at sk_wgs=192 -> fits in existing b->at_part
 *                 3,194,880 floats, 0 extra VRAM!); sk_flag >= sk_wgs u32s
 *                 pre-zeroed (self-reset -> fits in existing b->at_count!).
 *                 If sk_part==NULL || sk_flag==NULL, falls back to pure DP.
 *    - Guard:     !mk.rec && T >= 17u && rows >= 512u && nb >= 4u &&
 *                 (nb & 3u) == 0u && sk_wgs > 0u (best when total_tiles %
 *                 sk_wgs is in [32 .. sk_wgs - 32]).
 *    - Resources: <= 120 VGPR (true 12w/SIMD = 3 WGs/CU!), <= 21,504 B LDS,
 *                 0 v_fma_mix_f32, 0/0 spills.
 *    - Parity:    Deterministic f32-tol ~1e-6 (KLD <= 1e-4).
 *
 * 2. KERNEL ie_inkling_rel_attn_pf_far_poly_r13(
 *        const G float *q, const G kvt *kc, const G kvt *vc,
 *        const G float *sink, G float *part,
 *        u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
 *        u32 rel_extent, u32 qs, u32 ss)
 *    - H05b FlashAttention-4 Polynomial FMA exp2 interleaved variant of
 *      ie_inkling_rel_attn_pf_far_r10 (exact 14-arg drop-in ahead of
 *      ie_inkling_rel_attn_pf_near_merge_r10!).
 *    - Interleaves paired degree-4 Horner FMA polynomial exp2 (even lanes
 *      v=0,2 and v=4,6 on v_dual_fmac_f32 + integer exponent shift; max rel
 *      err 2.9e-6 << 5e-5, exact 1.0f at 0, exact 0.0f at -inf; 0 SFU ops!)
 *      with hardware v_exp_f32 (odd lanes v=1,3 and v=5,7) so FMA and SFU
 *      pipes run concurrently!
 *    - Grid:      ((T + 31u) / 32u, n_kv, 1u), NT = 256.
 *    - Resources: <= 120 VGPR (true 12w/SIMD = 3 WGs/CU!), 16,384 B LDS,
 *                 0/0 spills.
 *    - Parity:    KLD < 1e-5 (max exp2 rel err 2.9e-6 << f16 9.7e-4 ULP!).
 *
 * 3. KERNEL ie_inkling_rel_attn_pf_sb_poly_r13(
 *        const G float *q, const G kvt *kc, const G kvt *vc,
 *        const G float *R, const G float *rel_bias, const G float *sink, G float *out,
 *        u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
 *        u32 d_rel, u32 rel_extent, u32 win,
 *        u32 qs, u32 rs, u32 ss, u32 os)
 *    - H05b FA4 Polynomial FMA exp2 interleaved companion of pf_sb_r10 for
 *      ALL 40 SWA layers and Global fallback (exact 20-arg pf_r7 drop-in!).
 *    - Grid:      ((T + 31u) / 32u, n_kv, 1u), NT = 256.
 *    - Resources: <= 160 VGPR (2 WGs/CU), 32,128 B LDS, 0/0 spills.
 *    - Parity:    KLD < 1e-5.
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(kvt) == 2u, "ie_inkling_r13_streamk_pf requires f16 KV cache");

#define INK_R13_SK_KB       4u
#define INK_R13_SK_M_R      128u
#define INK_R13_SK_M_T      64u
#define INK_R13_SK_M_SWSTR  160u
#define INK_R13_SK_SLOT_F   8192u

#define INK_R13_SK_NB16     16u
#define INK_R13_SK_NB32     32u
#define INK_R13_SK_D        3u
#define INK_R13_SK_LOG2E    1.44269504088896340736f
#define INK_R13_SK_QSC      (0.08838834764831844299f * 1.44269504088896340736f)
#define INK_R13_SK_ROWB     256u
#define INK_R13_SK_PSTR     130u

/* Minimax degree-4 polynomial for 2^f on f in [-0.5, 0.5] with c0 = 1.0f exact:
 * max relative error = 2.915e-6 across [-126, 0] (<< 5e-5 spec and << f16 ULP),
 * exact 1.0f at x=0, exact 0.0f at x<=-127 or -inf, 0 SFU instructions! */
#define INK_R13_SK_EXP_C1   0.693124235f
#define INK_R13_SK_EXP_C2   0.240240976f
#define INK_R13_SK_EXP_C3   0.0559059493f
#define INK_R13_SK_EXP_C4   0.00958269928f
#define INK_R13_SK_EXP_MAG  12582912.0f

typedef unsigned int ink_r13_sk_u2 __attribute__((ext_vector_type(2)));
typedef float        ink_r13_sk_f2 __attribute__((ext_vector_type(2)));

struct __attribute__((packed, aligned(4))) ink_r13_sk_u32x4p {
  u32x4 v;
};

struct __attribute__((packed, aligned(4))) ink_r13_sk_u2p {
  ink_r13_sk_u2 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r13_sk_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r13_sk_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) ink_r13_sk_u2 ink_r13_sk_ld2(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r13_sk_u2p *)p)->v;
}

static inline __attribute__((always_inline)) void ink_r13_sk_lsignal(void) {
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  __builtin_amdgcn_s_barrier_signal(-1);
  __asm__ volatile("" ::: "memory");
}

static inline __attribute__((always_inline)) void ink_r13_sk_lwait(void) {
  __builtin_amdgcn_s_barrier_wait(-1);
  __asm__ volatile("" ::: "memory");
}

/* ============================================================================
 * Part A (H11): Hybrid Stream-K Dense Q8_0 128x64 Prefill WMMA (<= 120 VGPR!)
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r13_sk_swiz_rt_tt(
    const u32 tile, const u32 nrt, const u32 ntt, u32 *rt_out, u32 *tt_out) {
  const u32 grp_m = 4u;
  const u32 grp_sz = grp_m * ntt;
  const u32 grp = tile / grp_sz;
  const u32 in_grp = tile - grp * grp_sz;
  const u32 r0 = grp * grp_m;
  const u32 r_rem = nrt - r0;
  const u32 gw = r_rem < grp_m ? r_rem : grp_m;
  const u32 tt = in_grp / gw;
  const u32 rt = r0 + (in_grp - tt * gw);
  *rt_out = (u32)__builtin_amdgcn_readfirstlane((int)rt);
  *tt_out = (u32)__builtin_amdgcn_readfirstlane((int)tt);
}

static inline __attribute__((always_inline)) void ink_r13_sk_stage_sw(
    LDSP f16 *lsw, const u32 r_idx, const ink_r13_sk_u2 rsw) {
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
  const u32 str32 = INK_R13_SK_M_SWSTR >> 1u;
  sw32[(0u + odd) * str32 + (r2 >> 1u)] = v_a;
  sw32[(2u + odd) * str32 + (r2 >> 1u)] = v_b;
}

static inline __attribute__((always_inline)) void ink_r13_sk_stage_sx(
    LDSP float *lsx, const u32 t_idx, const u32x4 rsx) {
  u32 ti = t_idx;
  __asm__ volatile("" : "+v"(ti));
  const u32 d0 = rsx.x, d1 = rsx.y, d2 = rsx.z, d3 = rsx.w;
  lsx[0u * INK_R13_SK_M_T + ti] = __builtin_bit_cast(float, d0);
  lsx[1u * INK_R13_SK_M_T + ti] = __builtin_bit_cast(float, d1);
  lsx[2u * INK_R13_SK_M_T + ti] = __builtin_bit_cast(float, d2);
  lsx[3u * INK_R13_SK_M_T + ti] = __builtin_bit_cast(float, d3);
}

static inline __attribute__((always_inline)) f8 ink_r13_sk_load_sw_f8(
    const LDSP f16 *lsw, const u32 k, const u32 wr, const u32 h) {
  const h8 sw_h = *(const LDSP h8 *)(lsw + k * INK_R13_SK_M_SWSTR + wr + (h << 3u));
  f8 sw = __builtin_convertvector(sw_h, f8);
  __asm__ volatile("" : "+v"(sw));
  return sw;
}

static inline __attribute__((always_inline)) void ink_r13_sk_q8_k64(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 r16, const u32 h, const u32 lx_base0, f8 acc[4]) {
  const v8i zm = (v8i)(0);
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * INK_R13_SK_M_T + r16;
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

  const f8 sw = ink_r13_sk_load_sw_f8(lsw, k, wr, h);
  acc[0] += (sw * dx0) * __builtin_convertvector(c0, f8);
  __asm__ volatile("" : : "v"(acc[0]));
  __builtin_amdgcn_sched_barrier(0);
  acc[1] += (sw * dx1) * __builtin_convertvector(c1, f8);
  __asm__ volatile("" : : "v"(acc[1]) : "memory");
  __builtin_amdgcn_sched_barrier(0);

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

static inline __attribute__((always_inline)) void ink_r13_sk_store_q8_row8(
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

/* Exact O(1) Stream-K owner function: for total_iters >= active_sk >= 1 and 0 <= i < total_iters,
 * returns the unique WG w in [0, active_sk - 1] with iter_start(w) <= i < iter_end(w). */
static inline __attribute__((always_inline)) u32 ink_r13_sk_owner(
    const u32 iter, const u32 active_sk, const u32 total_iters) {
  const unsigned long num = (unsigned long)iter * (unsigned long)active_sk + (unsigned long)(active_sk - 1u);
  return (u32)__builtin_amdgcn_readfirstlane((int)(num / (unsigned long)total_iters));
}

KERNEL ie_inkling_gemm_q8_med_sk_r13(
    const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
    u32 rows, u32 nb, const G float *bias, const G float *res,
    u32 T, u32 xs, u32 ys, u32 rs,
    G float *sk_part, G u32 *sk_flag, u32 sk_wgs) {
  static LDS struct {
    u8    x[2u][INK_R13_SK_M_T * (INK_R13_SK_KB * 32u)] __attribute__((aligned(16))); /* 16,384 B */
    f16   sw[2u][INK_R13_SK_KB * INK_R13_SK_M_SWSTR] __attribute__((aligned(16)));    /*  2,560 B */
    float sx[2u][INK_R13_SK_KB * INK_R13_SK_M_T] __attribute__((aligned(32)));        /*  2,048 B */
    u32   is_last;                                                                    /*      4 B */
  } ink_r13_sk_med_lds;
  _Static_assert(sizeof(ink_r13_sk_med_lds) <= 21504u, "ink_r13_sk_med_lds must fit 3 WGs/CU (<= 21,504 B)");

  if (T == 0u || rows == 0u || nb < INK_R13_SK_KB || (nb & (INK_R13_SK_KB - 1u)) != 0u ||
      !qw || !qs || !xq || !y || sk_wgs == 0u)
    return;

  const u32 nrt = (rows + INK_R13_SK_M_R - 1u) / INK_R13_SK_M_R;
  const u32 ntt = (T + INK_R13_SK_M_T - 1u) / INK_R13_SK_M_T;
  const u32 total_tiles = nrt * ntt;
  const u32 wg = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  if (wg >= sk_wgs) return;

  const int use_sk = (sk_part != (G float *)0) && (sk_flag != (G u32 *)0);
  const u32 dp_tiles = use_sk ? ((total_tiles / sk_wgs) * sk_wgs) : total_tiles;
  const u32 sk_tiles = total_tiles - dp_tiles;
  const u32 n_kb = nb >> 2u;
  const u32 total_iters = sk_tiles * n_kb;
  const u32 active_sk = (total_iters < sk_wgs) ? total_iters : sk_wgs;

  u32 sk_cur = 0u, sk_end = 0u;
  if (sk_tiles > 0u && wg < active_sk) {
    sk_cur = (u32)__builtin_amdgcn_readfirstlane((int)(((unsigned long)wg * total_iters) / active_sk));
    sk_end = (u32)__builtin_amdgcn_readfirstlane((int)(((unsigned long)(wg + 1u) * total_iters) / active_sk));
  }

  const G u8 *qw8 = rq_ptr_u((const G u8 *)qw);
  const G u8 *qs8 = rq_ptr_u((const G u8 *)qs);
  const G u8 *xq8 = rq_ptr_u(xq);
  const G u8 *xq_s = rq_ptr_u(xq + (unsigned long)nb * 32ul);

  u32 dp_next = wg;
  u32 seg_idx = 0u;

  /* Unified single-K-loop state machine: Phase 1 (full DP waves) -> Phase 2 (Stream-K remainder). */
#pragma unroll 1
  for (;;) {
    u32 tile, kb_start, kb_end, sk_t = 0u;
    int is_sk_seg = 0;

    if (dp_next < dp_tiles) {
      tile = dp_next;
      dp_next += sk_wgs;
      kb_start = 0u;
      kb_end = nb;
    } else if (sk_cur < sk_end) {
      sk_t = (u32)__builtin_amdgcn_readfirstlane((int)(sk_cur / n_kb));
      tile = dp_tiles + sk_t;
      const u32 tile_i0 = sk_t * n_kb;
      const u32 tile_i1 = tile_i0 + n_kb;
      const u32 seg_end = (sk_end < tile_i1) ? sk_end : tile_i1;
      kb_start = (sk_cur - tile_i0) << 2u;
      kb_end = (seg_end - tile_i0) << 2u;
      sk_cur = seg_end;
      is_sk_seg = (kb_start > 0u || kb_end < nb) ? 1 : 0;
    } else {
      break;
    }

    if (seg_idx > 0u) {
      ink_r13_sk_lsignal();
      ink_r13_sk_lwait();
    }
    seg_idx++;

    u32 rt, tt;
    ink_r13_sk_swiz_rt_tt(tile, nrt, ntt, &rt, &tt);
    const u32 rbase = rt * INK_R13_SK_M_R;
    const u32 tbase = tt * INK_R13_SK_M_T;

    u32 t0 = tid();
    __asm__ volatile("" : "+v"(t0));
    const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
    const u32 l = t0 & 31u, r16 = l & 15u, h = l >> 4u;
    const u32 wr = w << 4u;

    f8 acc[4];
#pragma unroll
    for (u32 j = 0u; j < 4u; j++) acc[j] = (f8)(0.0f);

    {
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

      const G u8 *qw_0 = rq_ptr_u(qw8 + (unsigned long)kb_start * 32ul);
      const G u8 *qs_0 = rq_ptr_u(qs8 + (unsigned long)kb_start * 2ul);
      const G u8 *xq_0 = rq_ptr_u(xq8 + (unsigned long)kb_start * 32ul);
      const G u8 *xs_0 = rq_ptr_u(xq_s + (unsigned long)kb_start * 4ul);

      u32x4 wa[INK_R13_SK_KB];
#pragma unroll
      for (u32 k = 0u; k < INK_R13_SK_KB; k++) wa[k] = ink_r13_sk_ld4(qw_0 + w_off + k * 32u);
      {
        const u32x4 rx0_i = ink_r13_sk_ld4(xq_0 + xoff0);
        const u32x4 rx1_i = ink_r13_sk_ld4(xq_0 + xoff1);
        const ink_r13_sk_u2 rsw_i = is_sw ? ink_r13_sk_ld2(qs_0 + sw_off) : (ink_r13_sk_u2){0u, 0u};
        const u32x4 rsx_i = is_sx ? ink_r13_sk_ld4(xs_0 + sx_off) : (u32x4)(0u);
        *(LDSP u32x4 *)(ink_r13_sk_med_lds.x[0] + st_x) = rx0_i;
        *(LDSP u32x4 *)(ink_r13_sk_med_lds.x[0] + st_x + 4096u) = rx1_i;
        if (is_sw) ink_r13_sk_stage_sw(ink_r13_sk_med_lds.sw[0], t0, rsw_i);
        if (is_sx) ink_r13_sk_stage_sx(ink_r13_sk_med_lds.sx[0], sxi, rsx_i);
      }
      ink_r13_sk_lsignal();
      ink_r13_sk_lwait();

      u32 bf = 0u;
      u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
      for (u32 kb0 = kb_start; kb0 < kb_end; kb0 += INK_R13_SK_KB) {
        const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R13_SK_KB));
        const int more = (kb_n < kb_end);
        __asm__ volatile("" : : "v"(wa[0]), "v"(wa[1]));
        if (more) {
          const G u8 *xq_n = rq_ptr_u(xq8 + (unsigned long)kb_n * 32ul);
          rx0 = ink_r13_sk_ld4(xq_n + xoff0);
          rx1 = ink_r13_sk_ld4(xq_n + xoff1);
        }
        __builtin_amdgcn_sched_barrier(0);

        const LDSP u8 *lx = ink_r13_sk_med_lds.x[bf];
        const LDSP f16 *lsw = ink_r13_sk_med_lds.sw[bf];
        const LDSP float *lsx = ink_r13_sk_med_lds.sx[bf];

        ink_r13_sk_q8_k64(wa[0], lx, lsw, lsx, 0u, wr, r16, h, lx_base0, acc);
        ink_r13_sk_q8_k64(wa[1], lx, lsw, lsx, 1u, wr, r16, h, lx_base0, acc);

        ink_r13_sk_u2 rsw = {0u, 0u};
        u32x4 rsx = (u32x4)(0u);
        if (more) {
          const u32 nbf = bf ^ 1u;
          *(LDSP u32x4 *)(ink_r13_sk_med_lds.x[nbf] + st_x) = rx0;
          *(LDSP u32x4 *)(ink_r13_sk_med_lds.x[nbf] + st_x + 4096u) = rx1;
          const G u8 *qs_n = rq_ptr_u(qs8 + (unsigned long)kb_n * 2ul);
          const G u8 *xs_n = rq_ptr_u(xq_s + (unsigned long)kb_n * 4ul);
          if (is_sw) rsw = ink_r13_sk_ld2(qs_n + sw_off);
          if (is_sx) rsx = ink_r13_sk_ld4(xs_n + sx_off);
        }
        __builtin_amdgcn_sched_barrier(0);

        ink_r13_sk_q8_k64(wa[2], lx, lsw, lsx, 2u, wr, r16, h, lx_base0, acc);

        if (more) {
          const u32 nbf = bf ^ 1u;
          if (is_sw) ink_r13_sk_stage_sw(ink_r13_sk_med_lds.sw[nbf], t0, rsw);
          if (is_sx) ink_r13_sk_stage_sx(ink_r13_sk_med_lds.sx[nbf], sxi, rsx);
          ink_r13_sk_lsignal();
          const G u8 *qw_n = rq_ptr_u(qw8 + (unsigned long)kb_n * 32ul);
          wa[0] = ink_r13_sk_ld4(qw_n + w_off + 0u);
          wa[1] = ink_r13_sk_ld4(qw_n + w_off + 32u);
        }
        __builtin_amdgcn_sched_barrier(0);

        ink_r13_sk_q8_k64(wa[3], lx, lsw, lsx, 3u, wr, r16, h, lx_base0, acc);
        if (more) {
          const G u8 *qw_n = rq_ptr_u(qw8 + (unsigned long)kb_n * 32ul);
          wa[2] = ink_r13_sk_ld4(qw_n + w_off + 64u);
          wa[3] = ink_r13_sk_ld4(qw_n + w_off + 96u);
          ink_r13_sk_lwait();
          bf ^= 1u;
        }
      }
    }

    /* Epilogue: direct store for full tiles; lock-free last-arriver deterministic merge for split tiles. */
    int do_store = 1;
    if (is_sk_seg) {
      const u32 tile_i0 = sk_t * n_kb;
      const u32 w_first = ink_r13_sk_owner(tile_i0, active_sk, total_iters);
      const u32 w_last = ink_r13_sk_owner(tile_i0 + n_kb - 1u, active_sk, total_iters);
      const u32 n_contrib = w_last - w_first + 1u;
      const u32 my_slot = (wg == w_first) ? (2u * wg + 1u) : (2u * wg);

      G f32x4 *dst_p = (G f32x4 *)(sk_part + (unsigned long)my_slot * (unsigned long)INK_R13_SK_SLOT_F);
#pragma unroll
      for (u32 j = 0u; j < 4u; j++) {
        dst_p[(2u * j + 0u) * 256u + t0] = (f32x4){acc[j][0], acc[j][1], acc[j][2], acc[j][3]};
        dst_p[(2u * j + 1u) * 256u + t0] = (f32x4){acc[j][4], acc[j][5], acc[j][6], acc[j][7]};
      }
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      barrier();
      if (t0 == 0u) {
        ink_r13_sk_med_lds.is_last =
            (__scoped_atomic_fetch_add(&sk_flag[sk_t], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == n_contrib - 1u)
                ? 1u
                : 0u;
      }
      barrier();
      do_store = (ink_r13_sk_med_lds.is_last != 0u);
      if (do_store) {
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        if (t0 == 0u) __scoped_atomic_store_n(&sk_flag[sk_t], 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
#pragma unroll
        for (u32 j = 0u; j < 4u; j++) acc[j] = (f8)(0.0f);
        /* Exact deterministic ascending K-order reduction across all contributors [w_first..w_last]. */
#pragma unroll 1
        for (u32 wc = w_first; wc <= w_last; wc++) {
          const u32 slot = (wc == w_first) ? (2u * wc + 1u) : (2u * wc);
          const G f32x4 *src_p = (const G f32x4 *)(sk_part + (unsigned long)slot * (unsigned long)INK_R13_SK_SLOT_F);
#pragma unroll
          for (u32 j = 0u; j < 4u; j++) {
            const f32x4 v0 = src_p[(2u * j + 0u) * 256u + t0];
            const f32x4 v1 = src_p[(2u * j + 1u) * 256u + t0];
            acc[j] += (f8){v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
          }
        }
      }
    }

    if (do_store) {
      const u32 r0 = rbase + wr + (h << 3u);
#pragma unroll
      for (u32 j = 0u; j < 4u; j++) {
        const u32 t = tbase + (j << 4u) + r16;
        if (t < T) ink_r13_sk_store_q8_row8(y, bias, res, t, r0, rows, ys, rs, acc[j]);
      }
    }
  }
}

/* ============================================================================
 * Part B (H05b): FA4 Degree-4 Minimax Horner FMA exp2 + Dual-Pipe Interleave
 *   - Runs 100% on full-rate VALU/FMA pipe (0 SFU ops, 0 v_rndne, 0 v_cvt!).
 *   - Paired evaluation (a, b) unlocks gfx1201 v_dual_fmac_f32 (2 FMAs/cycle)
 *     while interleaved odd lanes execute v_exp_f32 concurrently on the SFU!
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r13_sk_exp2_poly_pair(
    const float xa, const float xb, float *ra_out, float *rb_out) {
  const float ca = __builtin_fmaxf(xa, -127.0f);
  const float cb = __builtin_fmaxf(xb, -127.0f);
  float za = ca + INK_R13_SK_EXP_MAG;
  float zb = cb + INK_R13_SK_EXP_MAG;
  /* Pin rounded magic sums so LLVM never folds (c + 12582912) - 12582912 -> c. */
  __asm__ volatile("" : "+v"(za), "+v"(zb));
  const float fa = ca - (za - INK_R13_SK_EXP_MAG);
  const float fb = cb - (zb - INK_R13_SK_EXP_MAG);
  const u32 ua = __builtin_bit_cast(u32, za);
  const u32 ub = __builtin_bit_cast(u32, zb);
  const u32 sa_u = (ua << 23u) + 0x3F800000u;
  const u32 sb_u = (ub << 23u) + 0x3F800000u;
  const float sa = __builtin_bit_cast(float, sa_u);
  const float sb = __builtin_bit_cast(float, sb_u);
  float pa = __builtin_fmaf(INK_R13_SK_EXP_C4, fa, INK_R13_SK_EXP_C3);
  float pb = __builtin_fmaf(INK_R13_SK_EXP_C4, fb, INK_R13_SK_EXP_C3);
  pa = __builtin_fmaf(pa, fa, INK_R13_SK_EXP_C2);
  pb = __builtin_fmaf(pb, fb, INK_R13_SK_EXP_C2);
  pa = __builtin_fmaf(pa, fa, INK_R13_SK_EXP_C1);
  pb = __builtin_fmaf(pb, fb, INK_R13_SK_EXP_C1);
  pa = __builtin_fmaf(pa, fa, 1.0f);
  pb = __builtin_fmaf(pb, fb, 1.0f);
  float ra = pa * sa;
  float rb = pb * sb;
  __asm__ volatile("" : "+v"(ra), "+v"(rb));
  *ra_out = ra;
  *rb_out = rb;
}

/* Evaluate 8 softmax probabilities with concurrent FMA-poly (even) + SFU v_exp_f32 (odd)! */
static inline __attribute__((always_inline)) float ink_r13_sk_exp2_interleaved8(
    const float s0, const float s1, const float s2, const float s3,
    const float s4, const float s5, const float s6, const float s7,
    const float mn, h8 *pb_out) {
  float p0, p2, p4, p6;
  /* Group A: SFU (p1, p3) launched alongside paired FMA-poly (p0, p2) -> dual-pipe overlap! */
  const float p1 = __builtin_amdgcn_exp2f(s1 - mn);
  const float p3 = __builtin_amdgcn_exp2f(s3 - mn);
  ink_r13_sk_exp2_poly_pair(s0 - mn, s2 - mn, &p0, &p2);
  float ps = (p0 + p1) + (p2 + p3);
  const f16 h0 = (f16)p0, h1 = (f16)p1, h2 = (f16)p2, h3 = (f16)p3;
  __builtin_amdgcn_sched_barrier(0);

  /* Group B: SFU (p5, p7) launched alongside paired FMA-poly (p4, p6)! */
  const float p5 = __builtin_amdgcn_exp2f(s5 - mn);
  const float p7 = __builtin_amdgcn_exp2f(s7 - mn);
  ink_r13_sk_exp2_poly_pair(s4 - mn, s6 - mn, &p4, &p6);
  ps += (p4 + p5) + (p6 + p7);
  *pb_out = (h8){h0, h1, h2, h3, (f16)p4, (f16)p5, (f16)p6, (f16)p7};
  return ps;
}

/* ============================================================================
 * Shared Attention Helpers (self-contained ink_r13_sk_ prefix)
 * ========================================================================== */

typedef struct {
  f16 k[INK_R13_SK_NB32 * 128u];     /*  8,192 B */
  f16 vt[2][128u * INK_R13_SK_NB16]; /*  8,192 B */
} ink_r13_sk_far_lds_t;

_Static_assert(sizeof(ink_r13_sk_far_lds_t) == 16384u, "ink_r13_sk_far_lds_t must be 16,384 B");

typedef struct {
  f16   k[2][INK_R13_SK_NB16 * 128u];  /*  8,192 B */
  f16   vt[2][128u * INK_R13_SK_NB16]; /*  8,192 B */
  f16   rb[2][48u * 16u];              /*  3,072 B */
  float rbsc[6][16][33];               /* 12,672 B */
} ink_r13_sk_nf_lds_t;

_Static_assert(sizeof(ink_r13_sk_nf_lds_t) <= 32768u, "ink_r13_sk_nf_lds_t must fit in 32 KiB");

static inline __attribute__((always_inline)) u32 ink_r13_sk_k_off(const u32 r, const u32 c) {
  return r * 128u + 8u * (c ^ (r & 15u));
}

static inline __attribute__((always_inline)) u32 ink_r13_sk_vt_off(const u32 col, const u32 h) {
  return 8u * ((2u * col + h) ^ ((col >> 2u) & 1u) ^ ((col >> 3u) & 7u));
}

static inline __attribute__((always_inline)) float ink_r13_sk_xrow(const float v) {
  return I2F(xrow_i(F2I(v)));
}

static inline __attribute__((always_inline)) u32 ink_r13_sk_far_nblk32(
    const u32 qpos_min, const u32 rel_extent, const u32 kv_len) {
  const u32 eff_ext = rel_extent > 0u ? rel_extent : 1u;
  if (qpos_min + 1u <= eff_ext) return 0u;
  const u32 lim = qpos_min + 1u - eff_ext;
  const u32 lim_nw = (lim <= kv_len) ? lim : kv_len;
  return lim_nw >> 5u;
}

static inline __attribute__((always_inline)) void ink_r13_sk_transpose8(const u32 *k, u32x4 *ev, u32x4 *od) {
  u32 e[4], o[4];
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    e[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x05040100u);
    o[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x07060302u);
  }
  *ev = (u32x4){e[0], e[1], e[2], e[3]};
  *od = (u32x4){o[0], o[1], o[2], o[3]};
}

static inline __attribute__((always_inline)) u32 ink_r13_sk_far_lane(void) {
  u32 l;
  __asm__ volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(l));
  return l;
}

/* ============================================================================
 * Kernel 2: ie_inkling_rel_attn_pf_far_poly_r13 (H05b Far-Field, <= 120 VGPR!)
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r13_sk_far_load_k32(
    u32x4 kx[8], const G u8 *kb_kh, const unsigned long row_step, const u32 kb, const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
  const G u8 *k_w = kb_kh + (unsigned long)(kb + 16u * wi) * row_step;
  const unsigned long k_off = (unsigned long)sub_r * row_step + 16ul * c;
#pragma unroll
  for (u32 i = 0u; i < 8u; i++) {
    kx[i] = *(const G u32x4 *)(k_w + (unsigned long)(2u * i) * row_step + k_off);
  }
}

static inline __attribute__((always_inline)) void ink_r13_sk_far_pub_k32(
    LDSP f16 *k_lds, const u32x4 kx[8], const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
#pragma unroll
  for (u32 i = 0u; i < 8u; i++) {
    const u32 r = 16u * wi + 2u * i + sub_r;
    *(LDSP u32x4 *)(k_lds + ink_r13_sk_k_off(r, c)) = kx[i];
  }
}

static inline __attribute__((always_inline)) void ink_r13_sk_far_load_v32(
    ink_r13_sk_u2 vx0[8], ink_r13_sk_u2 vx1[8], const G u8 *vb_kh,
    const unsigned long row_step, const u32 kb, const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const unsigned long v_off = 8ul * L;
  const G u8 *v_w0 = vb_kh + (unsigned long)(kb + 8u * wi) * row_step;
  const G u8 *v_w1 = vb_kh + (unsigned long)(kb + 16u + 8u * wi) * row_step;
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    vx0[u] = *(const G ink_r13_sk_u2 *)(v_w0 + (unsigned long)u * row_step + v_off);
    vx1[u] = *(const G ink_r13_sk_u2 *)(v_w1 + (unsigned long)u * row_step + v_off);
  }
}

static inline __attribute__((always_inline)) void ink_r13_sk_far_pub_vt16(
    LDSP f16 *vt_lds, const ink_r13_sk_u2 vx[8], const u32 wi, const u32 L) {
  u32 vx0[8], vx1[8];
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    vx0[u] = vx[u].x;
    vx1[u] = vx[u].y;
  }
  u32x4 ev0, od0, ev1, od1;
  ink_r13_sk_transpose8(vx0, &ev0, &od0);
  ink_r13_sk_transpose8(vx1, &ev1, &od1);
  const u32 c0 = 4u * L;
  *(LDSP u32x4 *)(vt_lds + ink_r13_sk_vt_off(c0 + 0u, wi)) = ev0;
  *(LDSP u32x4 *)(vt_lds + ink_r13_sk_vt_off(c0 + 1u, wi)) = od0;
  *(LDSP u32x4 *)(vt_lds + ink_r13_sk_vt_off(c0 + 2u, wi)) = ev1;
  *(LDSP u32x4 *)(vt_lds + ink_r13_sk_vt_off(c0 + 3u, wi)) = od1;
}

static inline __attribute__((always_inline)) v8f ink_r13_sk_far_qk16(
    const LDSP f16 *k_lds, const h8 qa[8]) {
  const u32 l_loc = ink_r13_sk_far_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 kb_b = (rl << 8u) | ((rl ^ hl) << 4u);
  const LDSP u8 *kp = (const LDSP u8 *)k_lds;

  v8f sa = (v8f)(0.0f);
  h8 k0 = *(const LDSP h8 *)(kp + (kb_b ^ 0u));
  h8 k1 = *(const LDSP h8 *)(kp + (kb_b ^ 32u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[0], sa);
  k0 = *(const LDSP h8 *)(kp + (kb_b ^ 64u));
  __builtin_amdgcn_sched_barrier(0);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[1], sa);
  k1 = *(const LDSP h8 *)(kp + (kb_b ^ 96u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[2], sa);
  k0 = *(const LDSP h8 *)(kp + (kb_b ^ 128u));
  __builtin_amdgcn_sched_barrier(0);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[3], sa);
  k1 = *(const LDSP h8 *)(kp + (kb_b ^ 160u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[4], sa);
  k0 = *(const LDSP h8 *)(kp + (kb_b ^ 192u));
  __builtin_amdgcn_sched_barrier(0);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[5], sa);
  k1 = *(const LDSP h8 *)(kp + (kb_b ^ 224u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[6], sa);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[7], sa);
  __builtin_amdgcn_sched_barrier(0);
  return sa;
}

static inline __attribute__((always_inline)) v8f ink_r13_sk_far_qk16_s1(
    const LDSP f16 *k_lds, const h8 qa[8]) {
  const u32 l_loc = ink_r13_sk_far_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  u32 kb_b = (rl << 8u) | ((rl ^ hl) << 4u);
  const LDSP u8 *kp = (const LDSP u8 *)k_lds;

  v8f sa = (v8f)(0.0f);
  h8 k0;

  k0 = *(const LDSP h8 *)(kp + kb_b);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[0], sa);
  kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);

  k0 = *(const LDSP h8 *)(kp + kb_b);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[1], sa);
  kb_b ^= 96u;
  __builtin_amdgcn_sched_barrier(0);

  k0 = *(const LDSP h8 *)(kp + kb_b);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[2], sa);
  kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);

  k0 = *(const LDSP h8 *)(kp + kb_b);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[3], sa);
  kb_b ^= 224u;
  __builtin_amdgcn_sched_barrier(0);

  k0 = *(const LDSP h8 *)(kp + kb_b);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[4], sa);
  kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);

  k0 = *(const LDSP h8 *)(kp + kb_b);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[5], sa);
  kb_b ^= 96u;
  __builtin_amdgcn_sched_barrier(0);

  k0 = *(const LDSP h8 *)(kp + kb_b);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[6], sa);
  kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);

  k0 = *(const LDSP h8 *)(kp + kb_b);
  ink_r13_sk_lsignal();
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[7], sa);
  __builtin_amdgcn_sched_barrier(0);
  return sa;
}

static inline __attribute__((always_inline)) void ink_r13_sk_far_pv16(
    const LDSP f16 *vt_lds, const h8 pb, v8f o[8], const int do_signal) {
  const u32 l_loc = ink_r13_sk_far_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2u) & 1u) ^ (rl >> 3u);
  const u32 vtb = 16u * vt_lane;
  const LDSP u8 *vp = (const LDSP u8 *)vt_lds;

  h8 v0 = *(const LDSP h8 *)(vp + 0u + vtb);
  h8 v1 = *(const LDSP h8 *)(vp + 512u + (vtb ^ 32u));
  __builtin_amdgcn_sched_barrier(0);

  o[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[0]);
  v0 = *(const LDSP h8 *)(vp + 1024u + (vtb ^ 64u));
  __builtin_amdgcn_sched_barrier(0);
  o[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[1]);
  v1 = *(const LDSP h8 *)(vp + 1536u + (vtb ^ 96u));
  __builtin_amdgcn_sched_barrier(0);

  o[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[2]);
  v0 = *(const LDSP h8 *)(vp + 2048u + vtb);
  __builtin_amdgcn_sched_barrier(0);
  o[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[3]);
  v1 = *(const LDSP h8 *)(vp + 2560u + (vtb ^ 32u));
  __builtin_amdgcn_sched_barrier(0);

  o[4] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[4]);
  v0 = *(const LDSP h8 *)(vp + 3072u + (vtb ^ 64u));
  __builtin_amdgcn_sched_barrier(0);
  o[5] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[5]);
  v1 = *(const LDSP h8 *)(vp + 3584u + (vtb ^ 96u));
  if (do_signal) {
    ink_r13_sk_lsignal();
  } else {
    __builtin_amdgcn_sched_barrier(0);
  }

  o[6] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[6]);
  o[7] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[7]);
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r13_sk_far_blk32_poly_step(
    const LDSP f16 *k_lds, const LDSP f16 *vt0, const LDSP f16 *vt1,
    const h8 qa[8], const int has_next, float *m_p, float *ls_p, v8f o[8]) {
  float m = *m_p;
  float ls = *ls_p;

  /* Half 0: QK_0 + FA4 interleaved FMA-poly / SFU exp2 + conditional rescale. */
  const v8f s0 = ink_r13_sk_far_qk16(k_lds, qa);
  h8 pb0 = (h8)(0.0f);
  int active0 = 0;
  {
    float bm0 = s0[0];
#pragma unroll
    for (u32 v = 1u; v < 8u; v++) bm0 = __builtin_fmaxf(bm0, s0[v]);
    bm0 = __builtin_fmaxf(bm0, ink_r13_sk_xrow(bm0));
    active0 = (__builtin_amdgcn_ballot_w32(bm0 > -1e30f && !(bm0 - m < -30.0f)) != 0u);
    if (active0) {
      const int nr0 = (__builtin_amdgcn_ballot_w32(bm0 > m || m <= -1e30f) != 0u);
      const float mn0 = nr0 ? __builtin_fmaxf(m, bm0) : m;
      float ps0 = 0.0f;
      if (mn0 > -1e30f) {
        ps0 = ink_r13_sk_exp2_interleaved8(s0[0], s0[1], s0[2], s0[3], s0[4], s0[5], s0[6], s0[7], mn0, &pb0);
      }
      float c0 = 1.0f;
      if (nr0) {
        c0 = (m > -1e30f) ? __builtin_amdgcn_exp2f(m - mn0) : 0.0f;
#pragma unroll
        for (u32 nt = 0u; nt < 8u; nt++)
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c0;
      }
      ls = ls * c0 + ps0;
      m = mn0;
    }
  }
  __builtin_amdgcn_sched_barrier(0);

  /* Half 1: QK_1 + FA4 interleaved FMA-poly / SFU exp2 -> frees s1 before lwait! */
  const v8f s1 = ink_r13_sk_far_qk16_s1(k_lds + 2048u, qa);
  h8 pb1 = (h8)(0.0f);
  float c1 = 1.0f;
  int active1 = 0, nr1 = 0;
  {
    float bm1 = s1[0];
#pragma unroll
    for (u32 v = 1u; v < 8u; v++) bm1 = __builtin_fmaxf(bm1, s1[v]);
    bm1 = __builtin_fmaxf(bm1, ink_r13_sk_xrow(bm1));
    active1 = (__builtin_amdgcn_ballot_w32(bm1 > -1e30f && !(bm1 - m < -30.0f)) != 0u);
    if (active1) {
      nr1 = (__builtin_amdgcn_ballot_w32(bm1 > m || m <= -1e30f) != 0u);
      const float mn1 = nr1 ? __builtin_fmaxf(m, bm1) : m;
      float ps1 = 0.0f;
      if (mn1 > -1e30f) {
        ps1 = ink_r13_sk_exp2_interleaved8(s1[0], s1[1], s1[2], s1[3], s1[4], s1[5], s1[6], s1[7], mn1, &pb1);
      }
      if (nr1) c1 = (m > -1e30f) ? __builtin_amdgcn_exp2f(m - mn1) : 0.0f;
      ls = ls * c1 + ps1;
      m = mn1;
    }
  }
  *m_p = m;
  *ls_p = ls;

  ink_r13_sk_lwait();

  if (active0) ink_r13_sk_far_pv16(vt0, pb0, o, 0);
  if (active1) {
    if (nr1) {
#pragma unroll
      for (u32 nt = 0u; nt < 8u; nt++)
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c1;
    }
    ink_r13_sk_far_pv16(vt1, pb1, o, has_next);
  }
  if (has_next) {
    if (!active1) ink_r13_sk_lsignal();
    ink_r13_sk_lwait();
  }
}

__attribute__((amdgpu_waves_per_eu(12, 16)))
KERNEL ie_inkling_rel_attn_pf_far_poly_r13(
    const G float *q, const G kvt *kc, const G kvt *vc,
    const G float *sink, G float *part,
    u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
    u32 rel_extent, u32 qs, u32 ss) {
  static LDS ink_r13_sk_far_lds_t ink_r13_sk_far_lds __attribute__((aligned(16)));
  if (hd != 128u || n_kv == 0u || n_head != n_kv * 3u || T == 0u || kv_len == 0u || !part) return;

  const u32 ntile = (T + 31u) >> 5u;
  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  if (wgid() >= ntile || kh >= n_kv) return;

  const u32 tile = (ntile == 16u && n_kv == 8u && kh < 4u) ? wgid() : ((ntile - 1u) - wgid());
  const u32 t0 = tile << 5u;
  const u32 qpos_min = pos0 + t0;
  const u32 nblk32 = (u32)__builtin_amdgcn_readfirstlane((int)ink_r13_sk_far_nblk32(qpos_min, rel_extent, kv_len));
  if (nblk32 == 0u) return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));

  if (w >= 6u) {
    const u32 pt = tid() - 192u;
    const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
    const u32 L = pt & 31u;
    const unsigned long row_step = (unsigned long)n_kv * INK_R13_SK_ROWB;
    const G u8 *kb_kh = (const G u8 *)kc + (unsigned long)kh * INK_R13_SK_ROWB;
    const G u8 *vb_kh = (const G u8 *)vc + (unsigned long)kh * INK_R13_SK_ROWB;

    u32x4 kx[8];
    ink_r13_sk_u2 vx0[8], vx1[8];

    ink_r13_sk_far_load_k32(kx, kb_kh, row_step, 0u, pt);
    __builtin_amdgcn_sched_barrier(0);
    ink_r13_sk_far_pub_k32(ink_r13_sk_far_lds.k, kx, pt);
    ink_r13_sk_lsignal();
    ink_r13_sk_far_load_v32(vx0, vx1, vb_kh, row_step, 0u, pt);
    if (1u < nblk32) ink_r13_sk_far_load_k32(kx, kb_kh, row_step, 32u, pt);
    ink_r13_sk_lwait();

#pragma clang loop unroll(disable)
    for (u32 b = 0u; b < nblk32; b++) {
      const u32 b_u = (u32)__builtin_amdgcn_readfirstlane((int)b);
      ink_r13_sk_far_pub_vt16(ink_r13_sk_far_lds.vt[0], vx0, wi, L);
      ink_r13_sk_far_pub_vt16(ink_r13_sk_far_lds.vt[1], vx1, wi, L);
      ink_r13_sk_lsignal();
      if (b_u + 1u < nblk32) {
        ink_r13_sk_far_load_v32(vx0, vx1, vb_kh, row_step, (b_u + 1u) << 5u, pt);
      }
      ink_r13_sk_lwait();

      if (b_u + 1u < nblk32) {
        ink_r13_sk_far_pub_k32(ink_r13_sk_far_lds.k, kx, pt);
        ink_r13_sk_lsignal();
        if (b_u + 2u < nblk32) {
          ink_r13_sk_far_load_k32(kx, kb_kh, row_step, (b_u + 2u) << 5u, pt);
        }
        ink_r13_sk_lwait();
      }
    }
    return;
  }

  const u32 tsub = w >= 3u ? 1u : 0u;
  const u32 hsub = w - 3u * tsub;
  const u32 t_sub0 = t0 + 16u * tsub;
  if (t_sub0 >= T) {
    const u32 nbar = 2u * nblk32;
    for (u32 i = 0u; i < nbar; i++) {
      ink_r13_sk_lsignal();
      ink_r13_sk_lwait();
    }
    return;
  }

  ink_r13_sk_lsignal();

  const u32 h = (u32)__builtin_amdgcn_readfirstlane((int)(kh * 3u + hsub));
  h8 qa[8];
  float m, ls;
  {
    const u32 l = lane(), hl = l >> 4u, rl = l & 15u;
    const u32 tk = t_sub0 + rl;
    const u32 tq = tk < T ? tk : (T - 1u);
    const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (unsigned long)tq * qs) +
                                           (unsigned long)h * 128ul);
#pragma unroll
    for (u32 ks = 0u; ks < 8u; ks++) {
      const f32x4 v0 = qp4[4u * ks + 2u * hl + 0u];
      const f32x4 v1 = qp4[4u * ks + 2u * hl + 1u];
      qa[ks] = (h8){(f16)(v0.x * INK_R13_SK_QSC), (f16)(v0.y * INK_R13_SK_QSC),
                    (f16)(v0.z * INK_R13_SK_QSC), (f16)(v0.w * INK_R13_SK_QSC),
                    (f16)(v1.x * INK_R13_SK_QSC), (f16)(v1.y * INK_R13_SK_QSC),
                    (f16)(v1.z * INK_R13_SK_QSC), (f16)(v1.w * INK_R13_SK_QSC)};
      __builtin_amdgcn_sched_barrier(0);
    }
    const float sink_val = ((const G float *)((const G u8 *)sink + (unsigned long)tq * ss))[h];
    m = sink_val * INK_R13_SK_LOG2E;
    ls = (hl == 0u) ? 1.0f : 0.0f;
  }
  __builtin_amdgcn_sched_barrier(0);

  v8f o[8];
#pragma unroll
  for (u32 nt = 0u; nt < 8u; nt++) o[nt] = (v8f)(0.0f);
  __builtin_amdgcn_sched_barrier(0);

  ink_r13_sk_lwait();

#pragma clang loop unroll(disable)
  for (u32 b = 0u; b < nblk32; b++) {
    const u32 b_u = (u32)__builtin_amdgcn_readfirstlane((int)b);
    const int has_next = (b_u + 1u < nblk32);
    ink_r13_sk_far_blk32_poly_step(ink_r13_sk_far_lds.k, ink_r13_sk_far_lds.vt[0],
                                   ink_r13_sk_far_lds.vt[1], qa, has_next, &m, &ls, o);
  }

  const float l_tot = ls + ink_r13_sk_xrow(ls);
  const u32 l_ep = ink_r13_sk_far_lane();
  const u32 rl_ep = l_ep & 15u, hl_ep = l_ep >> 4u;
  const u32 tk_ep = t_sub0 + rl_ep;
  if (tk_ep < T) {
    G float *pp = part + (unsigned long)(tk_ep * n_head + h) * INK_R13_SK_PSTR;
    if (hl_ep == 0u) {
      pp[0] = m;
      pp[1] = l_tot;
    }
    G ink_r13_sk_f2 *po2 = (G ink_r13_sk_f2 *)(pp + 2u);
#pragma unroll
    for (u32 nt = 0u; nt < 8u; nt++) {
      const u32 base2 = 8u * nt + 4u * hl_ep;
      po2[base2 + 0u] = (ink_r13_sk_f2){o[nt][0], o[nt][1]};
      po2[base2 + 1u] = (ink_r13_sk_f2){o[nt][2], o[nt][3]};
      po2[base2 + 2u] = (ink_r13_sk_f2){o[nt][4], o[nt][5]};
      po2[base2 + 3u] = (ink_r13_sk_f2){o[nt][6], o[nt][7]};
    }
  }
}

/* ============================================================================
 * Kernel 3: ie_inkling_rel_attn_pf_sb_poly_r13 (H05b SWA/All-Phase Poly exp2)
 * ========================================================================== */

static LDS ink_r13_sk_nf_lds_t ink_r13_sk_nf_lds __attribute__((aligned(16)));

typedef struct {
  u32x4         kx[4];
  ink_r13_sk_u2 vx[8];
} ink_r13_sk_slot16_t;

static inline __attribute__((always_inline)) u32 ink_r13_sk_wrap_s(u32 v, const u32 kv_len) {
  if (kv_len >= 16u) return (v >= kv_len) ? (v - kv_len) : v;
  while (v >= kv_len) v -= kv_len;
  return v;
}

static inline __attribute__((always_inline)) u32 ink_r13_sk_kv_adv(const u32 kv_b, const u32 kv_len) {
  return ink_r13_sk_wrap_s(kv_b + INK_R13_SK_NB16, kv_len);
}

static inline __attribute__((always_inline)) void ink_r13_sk_load16_nowrap(
    ink_r13_sk_slot16_t *s, const G u8 *kb_kh, const G u8 *vb_kh,
    const unsigned long row_step, const u32 kb, const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
  const G u8 *k_w = kb_kh + (unsigned long)(kb + 8u * wi) * row_step;
  const unsigned long k_off = (unsigned long)sub_r * row_step + 16ul * c;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    s->kx[i] = *(const G u32x4 *)(k_w + (unsigned long)(2u * i) * row_step + k_off);
  }
  const G u8 *v_w = vb_kh + (unsigned long)(kb + 8u * wi) * row_step;
  const unsigned long v_off = 8ul * L;
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    s->vx[u] = *(const G ink_r13_sk_u2 *)(v_w + (unsigned long)u * row_step + v_off);
  }
}

static inline __attribute__((always_inline)) void ink_r13_sk_load16_ring(
    ink_r13_sk_slot16_t *s, const G u8 *kb_kh, const G u8 *vb_kh,
    const unsigned long row_step, const u32 kv_len, const u32 kb, const u32 kv_b,
    const u32 kend, const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
  const u32 r_base = 8u * wi;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r0 = r_base + 2u * i;
    const u32 p0 = kb + r0;
    const u32 p1 = p0 + 1u;
    const u32 kvp0 = ink_r13_sk_wrap_s(kv_b + r0, kv_len);
    const u32 kvp1 = ink_r13_sk_wrap_s(kv_b + r0 + 1u, kv_len);
    const u32 p = sub_r ? p1 : p0;
    const u32 kvp = sub_r ? kvp1 : kvp0;
    s->kx[i] = (u32x4)(0u);
    if (p < kend) {
      s->kx[i] = *(const G u32x4 *)(kb_kh + (unsigned long)kvp * row_step + 16ul * c);
    }
  }
  const unsigned long v_off = 8ul * L;
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const u32 r = r_base + u;
    const u32 p = kb + r;
    s->vx[u] = (ink_r13_sk_u2)(0u);
    if (p < kend) {
      const u32 kvp = ink_r13_sk_wrap_s(kv_b + r, kv_len);
      const G u8 *v_row = vb_kh + (unsigned long)kvp * row_step;
      s->vx[u] = *(const G ink_r13_sk_u2 *)(v_row + v_off);
    }
  }
}

static inline __attribute__((always_inline)) void ink_r13_sk_pub16_kv(
    const ink_r13_sk_slot16_t *s, const u32 bf, const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r = 8u * wi + 2u * i + sub_r;
    *(LDSP u32x4 *)(ink_r13_sk_nf_lds.k[bf] + ink_r13_sk_k_off(r, c)) = s->kx[i];
  }
  u32 vx0[8], vx1[8];
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    vx0[u] = s->vx[u].x;
    vx1[u] = s->vx[u].y;
  }
  u32x4 ev0, od0, ev1, od1;
  ink_r13_sk_transpose8(vx0, &ev0, &od0);
  ink_r13_sk_transpose8(vx1, &ev1, &od1);
  const u32 c0 = 4u * L;
  *(LDSP u32x4 *)(ink_r13_sk_nf_lds.vt[bf] + ink_r13_sk_vt_off(c0 + 0u, wi)) = ev0;
  *(LDSP u32x4 *)(ink_r13_sk_nf_lds.vt[bf] + ink_r13_sk_vt_off(c0 + 1u, wi)) = od0;
  *(LDSP u32x4 *)(ink_r13_sk_nf_lds.vt[bf] + ink_r13_sk_vt_off(c0 + 2u, wi)) = ev1;
  *(LDSP u32x4 *)(ink_r13_sk_nf_lds.vt[bf] + ink_r13_sk_vt_off(c0 + 3u, wi)) = od1;
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r13_sk_pub16_rb(
    const G float *rel_bias, const u32 qpos_min, const u32 extent_lim,
    const u32 kb, const u32 bf, const u32 pt) {
  const int d_min_s = (int)qpos_min - (int)(kb + (INK_R13_SK_NB16 - 1u));
  const u32 d_base_wg = d_min_s > 0 ? (u32)d_min_s : 0u;
  if (d_base_wg < extent_lim && pt < 48u) {
    const u32 d = d_base_wg + pt;
    h8 r0 = (h8)(0.0f), r1 = (h8)(0.0f);
    if (d < extent_lim) {
      const G f32x4 *rp = (const G f32x4 *)(rel_bias + (unsigned long)d * 16ul);
      const f32x4 v0 = rp[0], v1 = rp[1], v2 = rp[2], v3 = rp[3];
      r0 = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
      r1 = (h8){(f16)v2.x, (f16)v2.y, (f16)v2.z, (f16)v2.w, (f16)v3.x, (f16)v3.y, (f16)v3.z, (f16)v3.w};
    }
    *(LDSP h8 *)(ink_r13_sk_nf_lds.rb[bf] + pt * 16u + 0u) = r0;
    *(LDSP h8 *)(ink_r13_sk_nf_lds.rb[bf] + pt * 16u + 8u) = r1;
  }
}

static inline __attribute__((always_inline)) void ink_r13_sk_producer16_sb(
    const G u8 *kb_kh, const G u8 *vb_kh, const G float *rel_bias,
    const unsigned long row_step, const u32 kv_len, const u32 k_start,
    const u32 kend, const u32 qpos_min, const u32 extent_lim,
    const u32 nblk, const u32 blk_far_end, const int no_wrap, const u32 pt) {
  if (nblk == 0u) return;

  const u32 nblk_full = (kend > k_start) ? ((kend - k_start) >> 4u) : 0u;
  const u32 fast_lim = no_wrap ? (nblk_full < nblk ? nblk_full : nblk) : 0u;

  u32 kv_load = no_wrap ? 0u : (k_start % kv_len);
  u32 k0 = 0u;
  ink_r13_sk_slot16_t ring[INK_R13_SK_D];

  if (no_wrap && fast_lim >= 6u && blk_far_end >= 6u) {
    const u32 far_loc = blk_far_end < fast_lim ? blk_far_end : fast_lim;
    ink_r13_sk_load16_nowrap(&ring[0], kb_kh, vb_kh, row_step, k_start + 0u, pt);
    ink_r13_sk_load16_nowrap(&ring[1], kb_kh, vb_kh, row_step, k_start + 16u, pt);
    ink_r13_sk_load16_nowrap(&ring[2], kb_kh, vb_kh, row_step, k_start + 32u, pt);
    __builtin_amdgcn_sched_barrier(0);
    ink_r13_sk_pub16_kv(&ring[0], 0u, pt);
    ink_r13_sk_lsignal();
    ink_r13_sk_load16_nowrap(&ring[0], kb_kh, vb_kh, row_step, k_start + 48u, pt);
    ink_r13_sk_lwait();

#pragma clang loop unroll(disable)
    for (; k0 + 6u < far_loc; k0 += 3u) {
      ink_r13_sk_pub16_kv(&ring[1], (k0 + 1u) & 1u, pt);
      ink_r13_sk_lsignal();
      ink_r13_sk_load16_nowrap(&ring[1], kb_kh, vb_kh, row_step, k_start + 16u * (k0 + 4u), pt);
      ink_r13_sk_lwait();

      ink_r13_sk_pub16_kv(&ring[2], (k0 + 2u) & 1u, pt);
      ink_r13_sk_lsignal();
      ink_r13_sk_load16_nowrap(&ring[2], kb_kh, vb_kh, row_step, k_start + 16u * (k0 + 5u), pt);
      ink_r13_sk_lwait();

      ink_r13_sk_pub16_kv(&ring[0], (k0 + 3u) & 1u, pt);
      ink_r13_sk_lsignal();
      ink_r13_sk_load16_nowrap(&ring[0], kb_kh, vb_kh, row_step, k_start + 16u * (k0 + 6u), pt);
      ink_r13_sk_lwait();
    }
  } else {
#pragma unroll
    for (u32 s = 0u; s < INK_R13_SK_D; s++) {
      if (s < nblk) {
        const u32 kbs = k_start + 16u * s;
        if (s < fast_lim) {
          ink_r13_sk_load16_nowrap(&ring[s], kb_kh, vb_kh, row_step, kbs, pt);
        } else {
          ink_r13_sk_load16_ring(&ring[s], kb_kh, vb_kh, row_step, kv_len, kbs, no_wrap ? kbs : kv_load, kend, pt);
        }
        if (!no_wrap) kv_load = ink_r13_sk_kv_adv(kv_load, kv_len);
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r13_sk_pub16_kv(&ring[0], 0u, pt);
    if (0u >= blk_far_end) {
      ink_r13_sk_pub16_rb(rel_bias, qpos_min, extent_lim, k_start, 0u, pt);
    }
    ink_r13_sk_lsignal();
    if (INK_R13_SK_D < nblk) {
      const u32 kbd = k_start + 16u * INK_R13_SK_D;
      if (INK_R13_SK_D < fast_lim) {
        ink_r13_sk_load16_nowrap(&ring[0], kb_kh, vb_kh, row_step, kbd, pt);
      } else {
        ink_r13_sk_load16_ring(&ring[0], kb_kh, vb_kh, row_step, kv_len, kbd, no_wrap ? kbd : kv_load, kend, pt);
      }
      if (!no_wrap) kv_load = ink_r13_sk_kv_adv(kv_load, kv_len);
    }
    ink_r13_sk_lwait();
  }

#pragma clang loop unroll(disable)
  for (; k0 < nblk; k0 += INK_R13_SK_D) {
#pragma unroll
    for (u32 j = 0u; j < INK_R13_SK_D; j++) {
      const u32 k = k0 + j;
      if (k + 1u < nblk) {
        const u32 nxt = (j + 1u) % INK_R13_SK_D;
        const u32 kn = k + 1u;
        ink_r13_sk_pub16_kv(&ring[nxt], kn & 1u, pt);
        if (kn >= blk_far_end) {
          ink_r13_sk_pub16_rb(rel_bias, qpos_min, extent_lim, k_start + 16u * kn, kn & 1u, pt);
        }
        ink_r13_sk_lsignal();
        const u32 kl = k + 1u + INK_R13_SK_D;
        if (kl < nblk) {
          const u32 kbl = k_start + 16u * kl;
          if (kl < fast_lim) {
            ink_r13_sk_load16_nowrap(&ring[nxt], kb_kh, vb_kh, row_step, kbl, pt);
          } else {
            ink_r13_sk_load16_ring(&ring[nxt], kb_kh, vb_kh, row_step, kv_len, kbl, no_wrap ? kbl : kv_load, kend, pt);
          }
          if (!no_wrap) kv_load = ink_r13_sk_kv_adv(kv_load, kv_len);
        }
        ink_r13_sk_lwait();
      }
    }
  }
}

static inline __attribute__((always_inline)) void ink_r13_sk_qk16_pipe(
    const u32 bf, const h8 qa[8], float s_out[8]) {
  u32 l_loc = lane();
  __asm__ volatile("" : "+v"(l_loc));
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 kb_b = (rl << 8u) | ((rl ^ hl) << 4u);
  const LDSP u8 *kp = (const LDSP u8 *)ink_r13_sk_nf_lds.k[bf];
  const v8f zero = (v8f)(0.0f);
  v8f sa = zero, sb = zero;

  h8 k0 = *(const LDSP h8 *)(kp + (kb_b ^ 0u));
  h8 k1 = *(const LDSP h8 *)(kp + (kb_b ^ 32u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[0], sa);
  k0 = *(const LDSP h8 *)(kp + (kb_b ^ 64u));
  sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[1], sb);
  k1 = *(const LDSP h8 *)(kp + (kb_b ^ 96u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[2], sa);
  k0 = *(const LDSP h8 *)(kp + (kb_b ^ 128u));
  sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[3], sb);
  k1 = *(const LDSP h8 *)(kp + (kb_b ^ 160u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[4], sa);
  k0 = *(const LDSP h8 *)(kp + (kb_b ^ 192u));
  sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[5], sb);
  k1 = *(const LDSP h8 *)(kp + (kb_b ^ 224u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[6], sa);
  sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[7], sb);

#pragma unroll
  for (u32 v = 0u; v < 8u; v++) s_out[v] = sa[v] + sb[v];
}

static inline __attribute__((always_inline)) void ink_r13_sk_pv16_pipe_sb(
    const u32 bf, const h8 pb, v8f o[8], const int has_next) {
  u32 l_loc = lane();
  __asm__ volatile("" : "+v"(l_loc));
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2u) & 1u) ^ (rl >> 3u);

  const LDSP f16 *vp0 = ink_r13_sk_nf_lds.vt[bf] + 8u * vt_lane;
  const LDSP f16 *vp1 = ink_r13_sk_nf_lds.vt[bf] + 256u + 8u * (vt_lane ^ 2u);
  const LDSP f16 *vp2 = ink_r13_sk_nf_lds.vt[bf] + 512u + 8u * (vt_lane ^ 4u);
  const LDSP f16 *vp3 = ink_r13_sk_nf_lds.vt[bf] + 768u + 8u * (vt_lane ^ 6u);

  h8 v0 = *(const LDSP h8 *)(vp0);
  h8 v1 = *(const LDSP h8 *)(vp1);
  __builtin_amdgcn_sched_barrier(0);

  o[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[0]);
  v0 = *(const LDSP h8 *)(vp2);
  __builtin_amdgcn_sched_barrier(0);
  o[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[1]);
  v1 = *(const LDSP h8 *)(vp3);
  __builtin_amdgcn_sched_barrier(0);

  o[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[2]);
  v0 = *(const LDSP h8 *)(vp0 + 1024u);
  __builtin_amdgcn_sched_barrier(0);
  o[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[3]);
  v1 = *(const LDSP h8 *)(vp1 + 1024u);
  __builtin_amdgcn_sched_barrier(0);

  o[4] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[4]);
  v0 = *(const LDSP h8 *)(vp2 + 1024u);
  __builtin_amdgcn_sched_barrier(0);
  o[5] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[5]);
  v1 = *(const LDSP h8 *)(vp3 + 1024u);
  if (has_next) {
    ink_r13_sk_lsignal();
  } else {
    __builtin_amdgcn_sched_barrier(0);
  }

  o[6] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[6]);
  o[7] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[7]);
}

static inline __attribute__((always_inline)) void ink_r13_sk_softmax_pv16_poly_sb(
    const float s[8], const u32 bf, const int has_next,
    float *m_p, float *ls_p, v8f o[8]) {
  float bm = s[0];
#pragma unroll
  for (u32 v = 1u; v < 8u; v++) bm = __builtin_fmaxf(bm, s[v]);
  bm = __builtin_fmaxf(bm, ink_r13_sk_xrow(bm));

  const float m = *m_p;
  const int active = (__builtin_amdgcn_ballot_w32(bm > -1e30f && !(bm - m < -30.0f)) != 0u);
  if (active) {
    const int need_rescale = (__builtin_amdgcn_ballot_w32(bm > m || m <= -1e30f) != 0u);
    float mn = m;
    float c = 1.0f;
    if (need_rescale) {
      mn = __builtin_fmaxf(m, bm);
      c = (m > -1e30f) ? __builtin_amdgcn_exp2f(m - mn) : 0.0f;
    }

    h8 pb;
    float ps = 0.0f;
    if (mn > -1e30f) {
      ps = ink_r13_sk_exp2_interleaved8(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], mn, &pb);
    } else {
      pb = (h8)(0.0f);
    }
    *ls_p = (*ls_p) * c + ps;
    *m_p = mn;

    if (need_rescale) {
#pragma unroll
      for (u32 nt = 0u; nt < 8u; nt++)
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c;
    }

    ink_r13_sk_pv16_pipe_sb(bf, pb, o, has_next);
  }
  if (has_next) {
    if (!active) ink_r13_sk_lsignal();
    ink_r13_sk_lwait();
  }
}

static inline __attribute__((always_inline)) void ink_r13_sk_consumer16_poly_sb(
    const u32 w, const u32 k_start, const u32 qpos_min, const u32 sub_qpos_min, const u32 sub_qpos_max,
    const u32 win, const u32 extent_lim, const u32 nblk, const u32 blk_far_end,
    const h8 qa[8], const h8 ra, float *m_p, float *ls_p, v8f o[8]) {
  const v8f zero = (v8f)(0.0f);
  const u32 nfar = blk_far_end < nblk ? blk_far_end : nblk;

#pragma clang loop unroll(disable)
  for (u32 k = 0u; k < nfar; k++) {
    const u32 k_u = (u32)__builtin_amdgcn_readfirstlane((int)k);
    const u32 bf = k_u & 1u;
    float s[8];
    ink_r13_sk_qk16_pipe(bf, qa, s);
    ink_r13_sk_softmax_pv16_poly_sb(s, bf, (k_u + 1u < nblk), m_p, ls_p, o);
  }

#pragma clang loop unroll(disable)
  for (u32 k = nfar; k < nblk; k++) {
    const u32 k_u = (u32)__builtin_amdgcn_readfirstlane((int)k);
    const u32 kb = k_start + INK_R13_SK_NB16 * k_u;
    const u32 bf = k_u & 1u;
    float s[8];
    ink_r13_sk_qk16_pipe(bf, qa, s);

    u32 l_loc = lane();
    __asm__ volatile("" : "+v"(l_loc));
    const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
    const u32 tp = sub_qpos_min + rl;

    const int d_min_sub_s = (int)sub_qpos_min - (int)(kb + (INK_R13_SK_NB16 - 1u));
    const u32 d_base_sub = d_min_sub_s > 0 ? (u32)d_min_sub_s : 0u;
    if (d_base_sub < extent_lim) {
      const int d_min_wg_s = (int)qpos_min - (int)(kb + (INK_R13_SK_NB16 - 1u));
      const u32 d_base_wg = d_min_wg_s > 0 ? (u32)d_min_wg_s : 0u;
      const u32 off = d_base_sub - d_base_wg;
      const h8 b0 = *(const LDSP h8 *)(ink_r13_sk_nf_lds.rb[bf] + (off + rl) * 16u + 8u * hl);
      const h8 b1 = *(const LDSP h8 *)(ink_r13_sk_nf_lds.rb[bf] + (off + 16u + rl) * 16u + 8u * hl);
      const v8f rb0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0, ra, zero);
      const v8f rb1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1, ra, zero);
      LDSP float *sc_row = ink_r13_sk_nf_lds.rbsc[w][rl];
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        sc_row[8u * hl + v] = rb0[v];
        sc_row[16u + 8u * hl + v] = rb1[v];
      }
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 kp = kb + 8u * hl + v;
        if (kp <= tp) {
          const u32 dist = tp - kp;
          if (dist < extent_lim) s[v] += sc_row[dist - d_base_sub];
        }
      }
    }

    if (kb + (INK_R13_SK_NB16 - 1u) > sub_qpos_min || (win > 0u && sub_qpos_max >= kb + win)) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 kp = kb + 8u * hl + v;
        if (kp > tp || (win > 0u && (tp - kp) >= win)) s[v] = -__builtin_inff();
      }
    }

    ink_r13_sk_softmax_pv16_poly_sb(s, bf, (k_u + 1u < nblk), m_p, ls_p, o);
  }
}

static inline __attribute__((always_inline)) u32 ink_r13_sk_calc_far_end16(
    const u32 win, const u32 qpos_min, const u32 extent_lim, const u32 k_start, const u32 nblk) {
  if (win > 0u) return 0u;
  const u32 guard = extent_lim + (INK_R13_SK_NB16 - 1u);
  if (qpos_min < guard || qpos_min - guard < k_start) return 0u;
  const u32 b_far = ((qpos_min - guard - k_start) >> 4u) + 1u;
  return b_far < nblk ? b_far : nblk;
}

KERNEL ie_inkling_rel_attn_pf_sb_poly_r13(
    const G float *q, const G kvt *kc, const G kvt *vc,
    const G float *R, const G float *rel_bias, const G float *sink, G float *out,
    u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
    u32 d_rel, u32 rel_extent, u32 win,
    u32 qs, u32 rs, u32 ss, u32 os) {
  if (hd != 128u || d_rel != 16u || n_kv == 0u || n_head != n_kv * 3u || T == 0u || kv_len == 0u) return;

  const u32 ntile = (T + 31u) >> 5u;
  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  if (wgid() >= ntile || kh >= n_kv) return;

  const u32 tile = (ntile == 16u && n_kv == 8u && win == 0u && kh < 4u) ? wgid() : ((ntile - 1u) - wgid());
  const u32 t0 = tile << 5u;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));

  const u32 tlast = (t0 + 31u < T) ? (t0 + 31u) : (T - 1u);
  const u32 qpos_min = pos0 + t0;
  const u32 qpos_max = pos0 + tlast;
  const u32 k_start = (win > 0u && qpos_min + 1u > win) ? (qpos_min + 1u - win) : 0u;
  const u32 kend = qpos_max + 1u;
  const u32 nblk = (kend > k_start) ? ((kend - k_start + INK_R13_SK_NB16 - 1u) >> 4u) : 0u;
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const u32 blk_far_end = ink_r13_sk_calc_far_end16(win, qpos_min, extent_lim, k_start, nblk);

  if (w >= 6u) {
    const unsigned long row_step = (unsigned long)n_kv * INK_R13_SK_ROWB;
    const G u8 *kb_kh = (const G u8 *)kc + (unsigned long)kh * INK_R13_SK_ROWB;
    const G u8 *vb_kh = (const G u8 *)vc + (unsigned long)kh * INK_R13_SK_ROWB;
    const int no_wrap = (kend <= kv_len);
    ink_r13_sk_producer16_sb(kb_kh, vb_kh, rel_bias, row_step, kv_len, k_start, kend,
                             qpos_min, extent_lim, nblk, blk_far_end, no_wrap, tid() - 192u);
    return;
  }

  const u32 tsub = w >= 3u ? 1u : 0u;
  const u32 hsub = w - 3u * tsub;
  const u32 t_sub0 = t0 + 16u * tsub;
  if (t_sub0 >= T) {
    for (u32 k = 0u; k < nblk; k++) {
      ink_r13_sk_lsignal();
      ink_r13_sk_lwait();
    }
    return;
  }

  if (nblk > 0u) ink_r13_sk_lsignal();

  const u32 h = (u32)__builtin_amdgcn_readfirstlane((int)(kh * 3u + hsub));
  const u32 sub_qpos_min = pos0 + t_sub0;
  const u32 sub_qpos_max = pos0 + (t_sub0 + 15u);

  h8 qa[8];
  h8 ra;
  float m, ls;
  {
    const u32 l = lane(), hl = l >> 4u, rl = l & 15u;
    const u32 tk = t_sub0 + rl;
    const u32 tq = tk < T ? tk : (T - 1u);
    const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (unsigned long)tq * qs) +
                                           (unsigned long)h * 128ul);
#pragma unroll
    for (u32 ks = 0u; ks < 8u; ks++) {
      const f32x4 v0 = qp4[4u * ks + 2u * hl + 0u];
      const f32x4 v1 = qp4[4u * ks + 2u * hl + 1u];
      qa[ks] = (h8){(f16)(v0.x * INK_R13_SK_QSC), (f16)(v0.y * INK_R13_SK_QSC),
                    (f16)(v0.z * INK_R13_SK_QSC), (f16)(v0.w * INK_R13_SK_QSC),
                    (f16)(v1.x * INK_R13_SK_QSC), (f16)(v1.y * INK_R13_SK_QSC),
                    (f16)(v1.z * INK_R13_SK_QSC), (f16)(v1.w * INK_R13_SK_QSC)};
    }
    const G f32x4 *rp4 = (const G f32x4 *)((const G float *)((const G u8 *)R + (unsigned long)tq * rs) +
                                           (unsigned long)h * 16ul);
    const f32x4 r0 = rp4[2u * hl + 0u];
    const f32x4 r1 = rp4[2u * hl + 1u];
    ra = (h8){(f16)(r0.x * INK_R13_SK_LOG2E), (f16)(r0.y * INK_R13_SK_LOG2E),
              (f16)(r0.z * INK_R13_SK_LOG2E), (f16)(r0.w * INK_R13_SK_LOG2E),
              (f16)(r1.x * INK_R13_SK_LOG2E), (f16)(r1.y * INK_R13_SK_LOG2E),
              (f16)(r1.z * INK_R13_SK_LOG2E), (f16)(r1.w * INK_R13_SK_LOG2E)};

    const float sink_val = ((const G float *)((const G u8 *)sink + (unsigned long)tq * ss))[h];
    m = sink_val * INK_R13_SK_LOG2E;
    ls = (hl == 0u) ? 1.0f : 0.0f;
  }

  v8f o[8];
#pragma unroll
  for (u32 nt = 0u; nt < 8u; nt++) o[nt] = (v8f)(0.0f);

  if (nblk > 0u) {
    ink_r13_sk_lwait();
    ink_r13_sk_consumer16_poly_sb(w, k_start, qpos_min, sub_qpos_min, sub_qpos_max,
                                  win, extent_lim, nblk, blk_far_end, qa, ra, &m, &ls, o);
  }

  const float inv = 1.0f / (ls + ink_r13_sk_xrow(ls));
  u32 l_ep = lane();
  __asm__ volatile("" : "+v"(l_ep));
  const u32 rl_ep = l_ep & 15u, hl_ep = l_ep >> 4u;
  const u32 tk_ep = t_sub0 + rl_ep;
  if (tk_ep < T) {
    G f32x4 *op4 = (G f32x4 *)((G float *)((G u8 *)out + (unsigned long)tk_ep * os) + (unsigned long)h * 128ul);
#pragma unroll
    for (u32 nt = 0u; nt < 8u; nt++) {
      const f32x4 a0 = {o[nt][0] * inv, o[nt][1] * inv, o[nt][2] * inv, o[nt][3] * inv};
      const f32x4 a1 = {o[nt][4] * inv, o[nt][5] * inv, o[nt][6] * inv, o[nt][7] * inv};
      op4[4u * nt + 2u * hl_ep + 0u] = a0;
      op4[4u * nt + 2u * hl_ep + 1u] = a1;
    }
  }
}

#undef INK_R13_SK_EXP_MAG
#undef INK_R13_SK_EXP_C4
#undef INK_R13_SK_EXP_C3
#undef INK_R13_SK_EXP_C2
#undef INK_R13_SK_EXP_C1
#undef INK_R13_SK_PSTR
#undef INK_R13_SK_ROWB
#undef INK_R13_SK_QSC
#undef INK_R13_SK_LOG2E
#undef INK_R13_SK_D
#undef INK_R13_SK_NB32
#undef INK_R13_SK_NB16
#undef INK_R13_SK_SLOT_F
#undef INK_R13_SK_M_SWSTR
#undef INK_R13_SK_M_T
#undef INK_R13_SK_M_R
#undef INK_R13_SK_KB

#endif /* !IE_MEGA */
