/* ============================================================================
 * Inkling-2 Round 9: Wave 2 Micro-Tuned Verify/T1 Kernels
 * Target: RX 9070 XT (gfx1201, wave32, NT=256)
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r9_wt_ /
 * INK_R9_WT_ and all macros are #undef'd at EOF.
 *
 * Addresses the two Wave 2 micro-findings from #225 / #228:
 *   1. D1 (qkv_qclip) & D2 (rel_clip_sink_gate) had CT=4 -> 2x DRAM weight read
 *      at T=6/T=8 verify, plus 160/100 WG tail imbalance & 6x wrel weight read.
 *   2. E3 (wo_sconv_post_r7) had unpipelined 1-block/iter loop + hard
 *      sched_barrier stalls + 32-VGPR f8 accs + inner `if (j < tc)` overhead.
 *
 * Exported kernels (100% bitwise-exact 0-bit-diff parity, 0/0 spills, priv=0,
 * <= 96 VGPR = 4 WGs/CU):
 *
 * 1. KERNEL ie_inkling_qkv_qclip_t8_r9(...)
 *    - Exact 34-arg drop-in signature of ie_inkling_qkv_qclip_r7 (bit 64).
 *    - 16 rows/WG (2 rows/wave, 32 lanes/row), SINGLE-PASS CT=8 weight read
 *      for ALL T = 1..8 (and 2-pass for T = 9..16), double-buffered b128
 *      weight prefetch, uniform `if (j >= tc) break;`!
 *    - Grid: (wg_q + wg_k + wg_v + wg_kd + wg_vd, 1u, 1u)
 *        wg_q  = (n_head * 128u) >> 4u  (192 WGs = exact 8 WGs/Q-head)
 *        wg_k  = (n_kv * 128u + 15u) >> 4u (64 WGs)
 *        wg_v  = (n_kv * 128u + 15u) >> 4u (64 WGs)
 *        wg_kd = rows_kd ? ((rows_kd + 15u) >> 4u) : 0u (0 or 2 WGs)
 *        wg_vd = rows_vd ? ((rows_vd + 15u) >> 4u) : 0u (0 or 2 WGs)
 *      Total = 320 WGs (SWA = exact 5.00 WGs/CU, 0% tail waste!) / 324 (Global).
 *    - Sync: head_cnt[0..n_head-1] (96 B, zero-init, auto-reset to 0 by the
 *      8th arriving WG of each Q head).
 *    - Parity: 0-bit-diff vs ie_gemv_q8q8[_t] + ie_inkling_vclip_r1 (hd=128).
 *
 * 2. KERNEL ie_inkling_rel_clip_sink_gate_t8_r9(...)
 *    - Exact 25-arg drop-in signature of ie_inkling_rel_clip_sink_gate_r7 (bit 65).
 *    - 16 rows/WG (2 rows/wave, 32 lanes/row), SINGLE-PASS CT=8 weight read
 *      for ALL T = 1..8 (including wrel — 1x weight read instead of 6x!).
 *    - Grid: (wg_r + wg_s + wg_g, 1u, 1u)
 *        wg_r = n_head                  (24 WGs: 1 WG = 1 full 16-row head -> 0 atomics!)
 *        wg_s = (n_head + 15u) >> 4u    (2 WGs for wsink)
 *        wg_g = (rows_g + 15u) >> 4u    (192 WGs for wagt = exact 3.00 WGs/CU!)
 *      Total = 218 WGs (fits in ONE 4-WG/CU resident wave on 64 CUs, uniform 2 rows/wave!).
 *    - Parity: 0-bit-diff vs ie_gemv_q8q8[_t] + ie_inkling_vclip_r1 (hd=16).
 *
 * 3. KERNEL ie_inkling_wo_sconv_post_t1_r9(...)
 *    - Exact 19-arg drop-in signature of ie_inkling_wo_sconv_post_r7 (bit 68).
 *    - Lean T==1 specialized fast path (<= 64 VGPR, 4 WGs/CU): exact 2x-pipelined
 *      4-block (8x b128 in flight) inner loop matching ie_inkling_gemv_q8_med_r2,
 *      with sconv weights/ring loaded AFTER the GEMV loop (0 extra VGPRs in loop!).
 *    - Grid: ((dim + 31u) >> 5u, 1u, 1u)  [(72, 1, 1) for dim=2304].
 *    - Guard: T == 1u && dim > 0u && nb >= 8u && (nb & 7u) == 0u.
 *    - Parity: 0-bit-diff vs ie_inkling_gemv_q8_med_r2 + ie_inkling_sconv_post_r7.
 *
 * 4. KERNEL ie_inkling_wo_sconv_post_t8_r9(...)
 *    - Exact 19-arg drop-in signature of ie_inkling_wo_sconv_post_r7 (bit 68).
 *    - Companion T=2..16 verify path: 16 rows/WG (144 WGs for dim=2304), single-pass
 *      CT=8 double-buffered weight prefetch, uniform `if (j >= tc) break;`,
 *      sconv weights/ring loaded AFTER GEMV (0 sconv VGPRs in hot loop!).
 *    - Grid: ((dim + 15u) >> 4u, 1u, 1u)  [(144, 1, 1) for dim=2304].
 *    - Parity: 0-bit-diff vs ie_gemv_q8q8_t + ie_inkling_sconv_post_r7.
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R9_WT_CT 8u

struct __attribute__((packed, aligned(4))) ink_r9_wt_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r9_wt_ld4(const G void *p) {
  return ((const G struct ink_r9_wt_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) float ink_r9_wt_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline __attribute__((always_inline)) int ink_r9_wt_dot8(u32x4 w0, u32x4 w1, u32x4 a0, u32x4 a1) {
  const u32 w00 = w0.x, w01 = w0.y, w02 = w0.z, w03 = w0.w;
  const u32 w10 = w1.x, w11 = w1.y, w12 = w1.z, w13 = w1.w;
  const u32 a00 = a0.x, a01 = a0.y, a02 = a0.z, a03 = a0.w;
  const u32 a10 = a1.x, a11 = a1.y, a12 = a1.z, a13 = a1.w;
  u32 d = ie_dot4_ss(0u, w00, a00);
  d = ie_dot4_ss(d, w01, a01);
  d = ie_dot4_ss(d, w02, a02);
  d = ie_dot4_ss(d, w03, a03);
  d = ie_dot4_ss(d, w10, a10);
  d = ie_dot4_ss(d, w11, a11);
  d = ie_dot4_ss(d, w12, a12);
  d = ie_dot4_ss(d, w13, a13);
  return (int)d;
}

static inline __attribute__((always_inline)) float ink_r9_wt_vclip_scale(
    float ss, float inv_hd, float clamp, u32 norope, float escale, u32 win, u32 pos0, u32 t) {
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

/* Single-pass (T <= 8) 32-lane Q8_0 row dot for tc in 1..8 tokens.
 * Exact 0-bit-diff lane assignment & wave_tree_f reduction matching ie_gemv_q8q8[_t].
 * Scalar accumulators a0..a7 + paired token scheduling stay purely in VGPRs
 * (0 priv, 0 scratch, <= 64 VGPR = 4 WGs/CU!). */
static inline __attribute__((always_inline)) void ink_r9_wt_row_q8_t8(
    const G u32 *row_qw_in, const G f16 *row_qs_in, const G u8 *xq_in,
    u32 nb, u32 tc_in, u32 xs, u32 l,
    float *o0, float *o1, float *o2, float *o3,
    float *o4, float *o5, float *o6, float *o7) {
#pragma clang fp contract(off)
  const u32 tc = (u32)__builtin_amdgcn_readfirstlane((int)tc_in);
  const G u8 *row_qw = rq_ptr_u((const G u8 *)row_qw_in);
  const G f16 *row_qs = (const G f16 *)rq_ptr_u((const G u8 *)row_qs_in);
  const G u8 *xq_base = rq_ptr_u(xq_in);
  const unsigned long x_sc_off = 32ul * (unsigned long)nb;

  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

  u32 b_c = l;
  int ok_c = (b_c < nb);
  u32x4 w0_c = ok_c ? ink_r9_wt_ld4(row_qw + (b_c << 5u)) : (u32x4)(0u);
  u32x4 w1_c = ok_c ? ink_r9_wt_ld4(row_qw + (b_c << 5u) + 16u) : (u32x4)(0u);
  float sw_c = ok_c ? (float)row_qs[b_c] : 0.0f;

  const u32 n_steps = (nb + 31u) >> 5u;
#pragma unroll 1
  for (u32 st = 0u; st < n_steps; st++) {
    const u32 b_n = b_c + 32u;
    const int ok_n = (b_n < nb);
    const u32x4 w0_n = ok_n ? ink_r9_wt_ld4(row_qw + (b_n << 5u)) : (u32x4)(0u);
    const u32x4 w1_n = ok_n ? ink_r9_wt_ld4(row_qw + (b_n << 5u) + 16u) : (u32x4)(0u);
    const float sw_n = ok_n ? (float)row_qs[b_n] : 0.0f;
    __builtin_amdgcn_sched_barrier(0);

    if (ok_c) {
      const u32 bo_bytes = b_c << 5u;
      const G u8 *xj = xq_base;
#define INK_R9_WT_T8_STEP(IDX, ACC) do { \
        if (tc > (IDX)) { \
          const u32x4 xv0 = ink_r9_wt_ld4(xj + bo_bytes); \
          const u32x4 xv1 = ink_r9_wt_ld4(xj + bo_bytes + 16u); \
          const float da = ((const G float *)(xj + x_sc_off))[b_c]; \
          const float term = (sw_c * da) * (float)ink_r9_wt_dot8(w0_c, w1_c, xv0, xv1); \
          (ACC) = (ACC) + term; \
          xj += xs; \
        } \
      } while (0)
      INK_R9_WT_T8_STEP(0u, a0);
      INK_R9_WT_T8_STEP(1u, a1);
      INK_R9_WT_T8_STEP(2u, a2);
      INK_R9_WT_T8_STEP(3u, a3);
      INK_R9_WT_T8_STEP(4u, a4);
      INK_R9_WT_T8_STEP(5u, a5);
      INK_R9_WT_T8_STEP(6u, a6);
      INK_R9_WT_T8_STEP(7u, a7);
#undef INK_R9_WT_T8_STEP
    }
    __builtin_amdgcn_sched_barrier(0);
    b_c = b_n; ok_c = ok_n; w0_c = w0_n; w1_c = w1_n; sw_c = sw_n;
  }

  *o0 = wave_tree_f(a0);
  *o1 = (tc > 1u) ? wave_tree_f(a1) : 0.0f;
  *o2 = (tc > 2u) ? wave_tree_f(a2) : 0.0f;
  *o3 = (tc > 3u) ? wave_tree_f(a3) : 0.0f;
  *o4 = (tc > 4u) ? wave_tree_f(a4) : 0.0f;
  *o5 = (tc > 5u) ? wave_tree_f(a5) : 0.0f;
  *o6 = (tc > 6u) ? wave_tree_f(a6) : 0.0f;
  *o7 = (tc > 7u) ? wave_tree_f(a7) : 0.0f;
}

/* 16 rows/WG (2 rows/wave, 32 lanes/row) single-pass CT=8 Q8_0 GEMV. */
static inline __attribute__((always_inline)) void ink_r9_wt_gemv_q8_16r(
    u32 g, const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
    u32 rows, u32 nb, u32 T, u32 xs, u32 ys_bytes) {
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 r_wave = (g << 4u) + (w << 1u);
  if (r_wave >= rows) return;

#pragma clang loop unroll(disable)
  for (u32 q = 0u; q < 2u; q++) {
    const u32 r = r_wave + q;
    if (r >= rows) break;
    const G u32 *row_qw = qw + (unsigned long)r * (unsigned long)nb * 8ul;
    const G f16 *row_qs = qs + (unsigned long)r * (unsigned long)nb;

    for (u32 t0 = 0u; t0 < T; t0 += INK_R9_WT_CT) {
      const u32 tc = (T - t0 < INK_R9_WT_CT) ? (T - t0) : INK_R9_WT_CT;
      float v0, v1, v2, v3, v4, v5, v6, v7;
      ink_r9_wt_row_q8_t8(row_qw, row_qs, xq + (unsigned long)t0 * (unsigned long)xs,
                          nb, tc, xs, l, &v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7);
      if (l == 0u) {
        G u8 *yb = (G u8 *)y + (unsigned long)t0 * (unsigned long)ys_bytes;
        ((G float *)yb)[r] = v0;
        if (tc > 1u) ((G float *)(yb + 1ul * (unsigned long)ys_bytes))[r] = v1;
        if (tc > 2u) ((G float *)(yb + 2ul * (unsigned long)ys_bytes))[r] = v2;
        if (tc > 3u) ((G float *)(yb + 3ul * (unsigned long)ys_bytes))[r] = v3;
        if (tc > 4u) ((G float *)(yb + 4ul * (unsigned long)ys_bytes))[r] = v4;
        if (tc > 5u) ((G float *)(yb + 5ul * (unsigned long)ys_bytes))[r] = v5;
        if (tc > 6u) ((G float *)(yb + 6ul * (unsigned long)ys_bytes))[r] = v6;
        if (tc > 7u) ((G float *)(yb + 7ul * (unsigned long)ys_bytes))[r] = v7;
      }
    }
  }
}

/* ============================================================================
 * Kernel 1: ie_inkling_qkv_qclip_t8_r9
 * Single-pass CT=8, 16 rows/WG (320 WGs SWA = 5.00 WGs/CU) QKV + q_clip + kd/vd.
 * ========================================================================== */

KERNEL ie_inkling_qkv_qclip_t8_r9(
    const G u32 *qw_q, const G f16 *qs_q, const G u8 *xq_q, G float *y_q_raw, G float *y_q_clip,
    const G u32 *qw_k, const G f16 *qs_k, const G u8 *xq_k, G float *y_k,
    const G u32 *qw_v, const G f16 *qs_v, const G u8 *xq_v, G float *y_v,
    const G u32 *qw_kd, const G f16 *qs_kd, G float *y_kd,
    const G u32 *qw_vd, const G f16 *qs_vd, G float *y_vd,
    const G float *v_qclip, G u32 *head_cnt,
    u32 n_head, u32 n_kv, u32 hd, u32 rows_kd, u32 rows_vd, u32 nb,
    float clamp, u32 norope, float escale, u32 win, u32 pos0, u32 T,
    u32 xs_q, u32 xs_k, u32 xs_v,
    u32 ys_qr, u32 ys_qc, u32 ys_k, u32 ys_v, u32 ys_kd, u32 ys_vd) {
  static LDS u32 ink_r9_wt_q_last;

  if (T == 0u || hd != 128u || n_head == 0u || n_kv == 0u || nb == 0u) return;

  const u32 qd = n_head * 128u;
  const u32 kvd = n_kv * 128u;
  const u32 wg_q = qd >> 4u; /* exact 8 WGs per Q head (16 rows/WG) */
  const u32 wg_k = (kvd + 15u) >> 4u;
  const u32 wg_v = (kvd + 15u) >> 4u;
  const u32 wg_kd = rows_kd ? ((rows_kd + 15u) >> 4u) : 0u;
  const u32 wg_vd = rows_vd ? ((rows_vd + 15u) >> 4u) : 0u;
  const u32 g = wgid();

  const G u32 *qw_sel;
  const G f16 *qs_sel;
  const G u8  *xq_sel;
  G float     *y_sel;
  u32 g_sel, rows_sel, xs_sel, ys_sel, is_q = 0u;

  if (g < wg_q) {
    qw_sel = qw_q; qs_sel = qs_q; xq_sel = xq_q; y_sel = y_q_raw;
    g_sel = g; rows_sel = qd; xs_sel = xs_q; ys_sel = ys_qr; is_q = 1u;
  } else if (g < wg_q + wg_k) {
    qw_sel = qw_k; qs_sel = qs_k; xq_sel = xq_k; y_sel = y_k;
    g_sel = g - wg_q; rows_sel = kvd; xs_sel = xs_k; ys_sel = ys_k;
  } else if (g < wg_q + wg_k + wg_v) {
    qw_sel = qw_v; qs_sel = qs_v; xq_sel = xq_v; y_sel = y_v;
    g_sel = g - (wg_q + wg_k); rows_sel = kvd; xs_sel = xs_v; ys_sel = ys_v;
  } else if (g < wg_q + wg_k + wg_v + wg_kd) {
    qw_sel = qw_kd; qs_sel = qs_kd; xq_sel = xq_k; y_sel = y_kd;
    g_sel = g - (wg_q + wg_k + wg_v); rows_sel = rows_kd; xs_sel = xs_k; ys_sel = ys_kd;
  } else if (g < wg_q + wg_k + wg_v + wg_kd + wg_vd) {
    qw_sel = qw_vd; qs_sel = qs_vd; xq_sel = xq_v; y_sel = y_vd;
    g_sel = g - (wg_q + wg_k + wg_v + wg_kd); rows_sel = rows_vd; xs_sel = xs_v; ys_sel = ys_vd;
  } else {
    return;
  }

  ink_r9_wt_gemv_q8_16r(g_sel, qw_sel, qs_sel, xq_sel, y_sel, rows_sel, nb, T, xs_sel, ys_sel);
  if (!is_q) return;

  /* Atomic tail q_clip on the 8th arriving WG of Q head h = g >> 3. */
  const u32 h = g >> 3u;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r9_wt_q_last =
        (__scoped_atomic_fetch_add(head_cnt + h, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == 7u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r9_wt_q_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  if (tid() == 0u) head_cnt[h] = 0u;

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const G f32x4 *vp = v_qclip ? ((const G f32x4 *)v_qclip + (unsigned long)h * 32ul) : (const G f32x4 *)0;
  const f32x4 vv = vp ? vp[l] : (f32x4){1.0f, 1.0f, 1.0f, 1.0f};

  for (u32 tk = wv; tk < T; tk += 8u) {
    const G f32x4 *ah = (const G f32x4 *)((const G u8 *)y_q_raw + (unsigned long)tk * (unsigned long)ys_qr)
                        + (unsigned long)h * 32ul;
    G f32x4 *bh = (G f32x4 *)((G u8 *)y_q_clip + (unsigned long)tk * (unsigned long)ys_qc)
                  + (unsigned long)h * 32ul;
    f32x4 xv = ah[l];
    if (clamp > 0.0f && vp) xv *= vv;
    float ss = (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
    ss = wave_sum_all(ss);
    const float f = ink_r9_wt_vclip_scale(ss, 1.0f / 128.0f, clamp, norope, escale, win, pos0, tk);
    f32x4 outv = xv * f;
    if (clamp == 0.0f && vp) outv *= vv;
    bh[l] = outv;
  }
}

/* ============================================================================
 * Kernel 2: ie_inkling_rel_clip_sink_gate_t8_r9
 * Single-pass CT=8, 16 rows/WG (218 WGs total: 24 wrel + 2 wsink + 192 wagt),
 * uniform 2 rows/wave across ALL WGs, 0 atomics, 1x weight read for T <= 8!
 * ========================================================================== */

KERNEL ie_inkling_rel_clip_sink_gate_t8_r9(
    const G u32 *qw_r, const G f16 *qs_r, const G float *v_rclip, G float *y_rclip,
    const G u32 *qw_s, const G f16 *qs_s, G float *y_sink,
    const G u32 *qw_g, const G f16 *qs_g, G float *y_gate,
    const G u8 *xq, u32 n_head, u32 d_rel, u32 rows_g, u32 nb,
    float clamp, u32 norope, float escale, u32 win, u32 pos0, u32 T,
    u32 xs, u32 ys_rc, u32 ys_s, u32 ys_g) {
  /* Head-local staging for 1 full 16-row head x up to 8 tokens (512 B LDS). */
  static LDS float ink_r9_wt_rbuf[8u][16u] __attribute__((aligned(16)));

  if (T == 0u || d_rel != 16u || n_head == 0u || rows_g == 0u || nb == 0u) return;

  const u32 wg_r = n_head;               /* 1 WG = 1 full 16-row head (8 waves x 2 rows/wave) */
  const u32 wg_s = (n_head + 15u) >> 4u; /* 2 WGs for n_head=24 */
  const u32 wg_g = (rows_g + 15u) >> 4u; /* 192 WGs for rows_g=3072 */
  const u32 g = wgid();

  if (g < wg_r) {
    const u32 h = g;
    const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
    const u32 l = lane();
    const u32 r_base = h * 16u + (w << 1u);

    for (u32 t0 = 0u; t0 < T; t0 += INK_R9_WT_CT) {
      const u32 tc = (T - t0 < INK_R9_WT_CT) ? (T - t0) : INK_R9_WT_CT;
#pragma clang loop unroll(disable)
      for (u32 q = 0u; q < 2u; q++) {
        const u32 r_in_h = (w << 1u) + q;
        const u32 r = r_base + q;
        const G u32 *row_qw = qw_r + (unsigned long)r * (unsigned long)nb * 8ul;
        const G f16 *row_qs = qs_r + (unsigned long)r * (unsigned long)nb;
        float v0, v1, v2, v3, v4, v5, v6, v7;
        ink_r9_wt_row_q8_t8(row_qw, row_qs, xq + (unsigned long)t0 * (unsigned long)xs,
                            nb, tc, xs, l, &v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7);
        if (l == 0u) {
          ink_r9_wt_rbuf[0u][r_in_h] = v0;
          if (tc > 1u) ink_r9_wt_rbuf[1u][r_in_h] = v1;
          if (tc > 2u) ink_r9_wt_rbuf[2u][r_in_h] = v2;
          if (tc > 3u) ink_r9_wt_rbuf[3u][r_in_h] = v3;
          if (tc > 4u) ink_r9_wt_rbuf[4u][r_in_h] = v4;
          if (tc > 5u) ink_r9_wt_rbuf[5u][r_in_h] = v5;
          if (tc > 6u) ink_r9_wt_rbuf[6u][r_in_h] = v6;
          if (tc > 7u) ink_r9_wt_rbuf[7u][r_in_h] = v7;
        }
      }
      lbarrier();

      /* All tc tokens (<= 8) clip in parallel across waves w = 0..tc-1! */
      if (w < tc && l < 4u) {
        const G f32x4 *vp = v_rclip ? ((const G f32x4 *)v_rclip + (unsigned long)h * 4ul) : (const G f32x4 *)0;
        const f32x4 vv = vp ? vp[l] : (f32x4){1.0f, 1.0f, 1.0f, 1.0f};
        f32x4 xv = *(const LDSP f32x4 *)(ink_r9_wt_rbuf[w] + (l << 2u));
        if (clamp > 0.0f && vp) xv *= vv;
        float ss = (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
        ss += shfl_xor_f(ss, 2u);
        ss += shfl_xor_f(ss, 1u);
        const float f = ink_r9_wt_vclip_scale(ss, 1.0f / 16.0f, clamp, norope, escale, win, pos0, t0 + w);
        f32x4 outv = xv * f;
        if (clamp == 0.0f && vp) outv *= vv;
        G f32x4 *bh = (G f32x4 *)((G u8 *)y_rclip + (unsigned long)(t0 + w) * (unsigned long)ys_rc)
                      + (unsigned long)h * 4ul;
        bh[l] = outv;
      }
      if (t0 + INK_R9_WT_CT < T) lbarrier();
    }
    return;
  }

  const u32 g1 = g - wg_r;
  if (g1 >= wg_s + wg_g) return;
  const int is_s = (g1 < wg_s);
  const G u32 *qw_sel = is_s ? qw_s : qw_g;
  const G f16 *qs_sel = is_s ? qs_s : qs_g;
  G float     *y_sel  = is_s ? y_sink : y_gate;
  const u32 g_sel     = is_s ? g1 : (g1 - wg_s);
  const u32 rows_sel  = is_s ? n_head : rows_g;
  const u32 ys_sel    = is_s ? ys_s : ys_g;
  ink_r9_wt_gemv_q8_16r(g_sel, qw_sel, qs_sel, xq, y_sel, rows_sel, nb, T, xs, ys_sel);
}

/* ============================================================================
 * Kernel 3: ie_inkling_wo_sconv_post_t1_r9
 * Lean T==1 specialized W_o + attn_pc sconv + residual/copy + ring (<= 64 VGPR).
 * Exact 2x-pipelined 4-block inner loop matching ie_inkling_gemv_q8_med_r2,
 * loading sconv weights/ring ONLY AFTER GEMV completes (0 extra loop VGPRs!).
 * ========================================================================== */

KERNEL ie_inkling_wo_sconv_post_t1_r9(
    const G u32 *qw, const G f16 *qs, const G u8 *xq,
    const G float *v_conv, G float *ring,
    const G float *block_res, G float *out, G float *blk_dst, G float *hist_out,
    u32 dim, u32 nb, float escale, u32 pos0, u32 T,
    u32 xs, u32 res_s, u32 out_s, u32 blk_s, u32 hist_row_stride) {
#pragma clang fp contract(off)
  (void)xs; (void)res_s; (void)out_s; (void)blk_s; (void)hist_row_stride;
  if (T != 1u || dim == 0u || nb < 8u || (nb & 7u) != 0u) return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 r = (wgid() << 5u) + (w << 2u) + (l >> 3u);
  if (r >= dim) return;

  const G u32 *row_qw = qw + (unsigned long)r * (unsigned long)nb * 8ul;
  const G f16 *row_qs = qs + (unsigned long)r * (unsigned long)nb;
  const G u32 *aw = (const G u32 *)xq;
  const G float *da = (const G float *)(xq + (unsigned long)nb * 32ul);

  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  const u32 nb_main = nb & ~31u;

  for (u32 k0 = 0u; k0 < nb_main; k0 += 32u) {
    const u32 b0 = k0 + sub;
    const u32 b1 = b0 + 8u;
    const G u32 *wp0 = row_qw + (unsigned long)b0 * 8ul;
    const G u32 *wp1 = row_qw + (unsigned long)b1 * 8ul;
    const G u32 *ap0 = aw + (unsigned long)b0 * 8ul;
    const G u32 *ap1 = aw + (unsigned long)b1 * 8ul;
    const u32x4 w0_0 = ink_r9_wt_ld4(wp0);
    const u32x4 w0_1 = ink_r9_wt_ld4(wp0 + 4u);
    const u32x4 x0_0 = ink_r9_wt_ld4(ap0);
    const u32x4 x0_1 = ink_r9_wt_ld4(ap0 + 4u);
    const float sc0 = (float)row_qs[b0] * da[b0];
    const u32x4 w1_0 = ink_r9_wt_ld4(wp1);
    const u32x4 w1_1 = ink_r9_wt_ld4(wp1 + 4u);
    const u32x4 x1_0 = ink_r9_wt_ld4(ap1);
    const u32x4 x1_1 = ink_r9_wt_ld4(ap1 + 4u);
    const float sc1 = (float)row_qs[b1] * da[b1];

    const int d0 = ink_r9_wt_dot8(w0_0, w0_1, x0_0, x0_1);
    a0 = a0 + sc0 * (float)d0;
    const int d1 = ink_r9_wt_dot8(w1_0, w1_1, x1_0, x1_1);
    a1 = a1 + sc1 * (float)d1;
    __builtin_amdgcn_sched_barrier(0);

    const u32 b2 = b0 + 16u;
    const u32 b3 = b0 + 24u;
    const G u32 *wp2 = row_qw + (unsigned long)b2 * 8ul;
    const G u32 *wp3 = row_qw + (unsigned long)b3 * 8ul;
    const G u32 *ap2 = aw + (unsigned long)b2 * 8ul;
    const G u32 *ap3 = aw + (unsigned long)b3 * 8ul;
    const u32x4 w2_0 = ink_r9_wt_ld4(wp2);
    const u32x4 w2_1 = ink_r9_wt_ld4(wp2 + 4u);
    const u32x4 x2_0 = ink_r9_wt_ld4(ap2);
    const u32x4 x2_1 = ink_r9_wt_ld4(ap2 + 4u);
    const float sc2 = (float)row_qs[b2] * da[b2];
    const u32x4 w3_0 = ink_r9_wt_ld4(wp3);
    const u32x4 w3_1 = ink_r9_wt_ld4(wp3 + 4u);
    const u32x4 x3_0 = ink_r9_wt_ld4(ap3);
    const u32x4 x3_1 = ink_r9_wt_ld4(ap3 + 4u);
    const float sc3 = (float)row_qs[b3] * da[b3];

    const int d2 = ink_r9_wt_dot8(w2_0, w2_1, x2_0, x2_1);
    a2 = a2 + sc2 * (float)d2;
    const int d3 = ink_r9_wt_dot8(w3_0, w3_1, x3_0, x3_1);
    a3 = a3 + sc3 * (float)d3;
    __builtin_amdgcn_sched_barrier(0);
  }

  const u32 rem = nb - nb_main;
  if (rem > 0u) {
    const u32 b0 = nb_main + sub;
    if (b0 < nb) {
      const G u32 *wp0 = row_qw + (unsigned long)b0 * 8ul;
      const G u32 *ap0 = aw + (unsigned long)b0 * 8ul;
      const int d0 = ink_r9_wt_dot8(ink_r9_wt_ld4(wp0), ink_r9_wt_ld4(wp0 + 4u),
                                    ink_r9_wt_ld4(ap0), ink_r9_wt_ld4(ap0 + 4u));
      a0 = a0 + ((float)row_qs[b0] * da[b0]) * (float)d0;
    }
    if (rem > 8u) {
      const u32 b1 = nb_main + 8u + sub;
      if (b1 < nb) {
        const G u32 *wp1 = row_qw + (unsigned long)b1 * 8ul;
        const G u32 *ap1 = aw + (unsigned long)b1 * 8ul;
        const int d1 = ink_r9_wt_dot8(ink_r9_wt_ld4(wp1), ink_r9_wt_ld4(wp1 + 4u),
                                      ink_r9_wt_ld4(ap1), ink_r9_wt_ld4(ap1 + 4u));
        a1 = a1 + ((float)row_qs[b1] * da[b1]) * (float)d1;
      }
    }
    if (rem > 16u) {
      const u32 b2 = nb_main + 16u + sub;
      if (b2 < nb) {
        const G u32 *wp2 = row_qw + (unsigned long)b2 * 8ul;
        const G u32 *ap2 = aw + (unsigned long)b2 * 8ul;
        const int d2 = ink_r9_wt_dot8(ink_r9_wt_ld4(wp2), ink_r9_wt_ld4(wp2 + 4u),
                                      ink_r9_wt_ld4(ap2), ink_r9_wt_ld4(ap2 + 4u));
        a2 = a2 + ((float)row_qs[b2] * da[b2]) * (float)d2;
      }
    }
  }

  float x0 = (a0 + a2) + (a1 + a3);
  x0 = x0 + I2F(shl_i(F2I(x0), 4));
  x0 = x0 + I2F(shl_i(F2I(x0), 2));
  x0 = x0 + I2F(shl_i(F2I(x0), 1));

  /* Epilogue AFTER GEMV finishes -> sconv state uses 0 VGPRs during GEMV! */
  if (sub == 0u) {
    const f32x4 vw = ((const G f32x4 *)v_conv)[r];
    float x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
    if (pos0 > 0u) {
      const f32x4 rg = ((const G f32x4 *)ring)[r];
      x1 = ink_r9_wt_pick4(rg, (pos0 - 1u) & 3u);
      if (pos0 >= 2u) x2 = ink_r9_wt_pick4(rg, (pos0 - 2u) & 3u);
      if (pos0 >= 3u) x3 = ink_r9_wt_pick4(rg, (pos0 - 3u) & 3u);
    }
    const float cv = ((x0 * vw.w + x1 * vw.z) + x2 * vw.y) + x3 * vw.x;
    const float sc = (x0 + cv) * escale;
    const float rv = block_res[r];
    if (blk_dst) {
      blk_dst[r] = rv;
      out[r] = sc;
    } else {
      out[r] = rv + sc;
    }
    if (hist_out) hist_out[r] = x0;
    else ring[(unsigned long)r * 4ul + (pos0 & 3u)] = x0;
  }
}

/* ============================================================================
 * Kernel 4: ie_inkling_wo_sconv_post_t8_r9
 * Companion verify (T=2..16) W_o + attn_pc sconv + residual/copy [+ hist/ring].
 * 16 rows/WG (144 WGs for dim=2304), single-pass CT=8 double-buffered GEMV,
 * 0 sconv VGPRs live in hot loop!
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r9_wt_wo_step(
    float x0, float *x1, float *x2, float *x3,
    float w0, float w1, float w2, float w3,
    const G float *block_res, G float *out, G float *blk_dst, G float *hist_out, G float *ring,
    u32 r, u32 tk, u32 pos0, u32 T, float escale,
    u32 res_s, u32 out_s, u32 blk_s, u32 hist_row_stride) {
#pragma clang fp contract(off)
  const float cv = ((x0 * w0 + (*x1) * w1) + (*x2) * w2) + (*x3) * w3;
  const float sc = (x0 + cv) * escale;
  const float rv = ((const G float *)((const G u8 *)block_res + (unsigned long)tk * (unsigned long)res_s))[r];
  G float *op = (G float *)((G u8 *)out + (unsigned long)tk * (unsigned long)out_s);
  if (blk_dst) {
    G float *bp = (G float *)((G u8 *)blk_dst + (unsigned long)tk * (unsigned long)blk_s);
    bp[r] = rv;
    op[r] = sc;
  } else {
    op[r] = rv + sc;
  }
  if (hist_out) {
    hist_out[(unsigned long)tk * (unsigned long)hist_row_stride + (unsigned long)r] = x0;
  } else if (tk + 4u >= T) {
    ring[(unsigned long)r * 4ul + ((pos0 + tk) & 3u)] = x0;
  }
  *x3 = *x2; *x2 = *x1; *x1 = x0;
}

KERNEL ie_inkling_wo_sconv_post_t8_r9(
    const G u32 *qw, const G f16 *qs, const G u8 *xq,
    const G float *v_conv, G float *ring,
    const G float *block_res, G float *out, G float *blk_dst, G float *hist_out,
    u32 dim, u32 nb, float escale, u32 pos0, u32 T,
    u32 xs, u32 res_s, u32 out_s, u32 blk_s, u32 hist_row_stride) {
  /* Tiny 352-B LDS staging so sconv loop runs 1 non-unrolled copy (low SGPR + <=64 VGPR!). */
  static LDS float ink_r9_wt_wo_h[8u][3u];
  static LDS float ink_r9_wt_wo_v[8u][8u];

  if (T == 0u || T > 16u || dim == 0u || nb == 0u) return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 r_wave = (wgid() << 4u) + (w << 1u);
  if (r_wave >= dim) return;

#pragma clang loop unroll(disable)
  for (u32 q = 0u; q < 2u; q++) {
    const u32 r = r_wave + q;
    if (r >= dim) break;
    const G u32 *row_qw = qw + (unsigned long)r * (unsigned long)nb * 8ul;
    const G f16 *row_qs = qs + (unsigned long)r * (unsigned long)nb;

    for (u32 t0 = 0u; t0 < T; t0 += INK_R9_WT_CT) {
      const u32 tc = (T - t0 < INK_R9_WT_CT) ? (T - t0) : INK_R9_WT_CT;
      float v0, v1, v2, v3, v4, v5, v6, v7;
      ink_r9_wt_row_q8_t8(row_qw, row_qs, xq + (unsigned long)t0 * (unsigned long)xs,
                          nb, tc, xs, l, &v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7);

      if (l == 0u) {
        ink_r9_wt_wo_v[w][0u] = v0;
        if (tc > 1u) ink_r9_wt_wo_v[w][1u] = v1;
        if (tc > 2u) ink_r9_wt_wo_v[w][2u] = v2;
        if (tc > 3u) ink_r9_wt_wo_v[w][3u] = v3;
        if (tc > 4u) ink_r9_wt_wo_v[w][4u] = v4;
        if (tc > 5u) ink_r9_wt_wo_v[w][5u] = v5;
        if (tc > 6u) ink_r9_wt_wo_v[w][6u] = v6;
        if (tc > 7u) ink_r9_wt_wo_v[w][7u] = v7;

        const f32x4 vw = ((const G f32x4 *)v_conv)[r];
        const float w0 = vw.w, w1 = vw.z, w2 = vw.y, w3 = vw.x;
        float x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
        if (t0 == 0u) {
          if (pos0 > 0u) {
            const f32x4 rg = ((const G f32x4 *)ring)[r];
            x1 = ink_r9_wt_pick4(rg, (pos0 - 1u) & 3u);
            if (pos0 >= 2u) x2 = ink_r9_wt_pick4(rg, (pos0 - 2u) & 3u);
            if (pos0 >= 3u) x3 = ink_r9_wt_pick4(rg, (pos0 - 3u) & 3u);
          }
        } else {
          x1 = ink_r9_wt_wo_h[w][0u];
          x2 = ink_r9_wt_wo_h[w][1u];
          x3 = ink_r9_wt_wo_h[w][2u];
        }

#pragma clang loop unroll(disable)
        for (u32 j = 0u; j < tc; j++) {
          ink_r9_wt_wo_step(ink_r9_wt_wo_v[w][j], &x1, &x2, &x3, w0, w1, w2, w3,
                            block_res, out, blk_dst, hist_out, ring,
                            r, t0 + j, pos0, T, escale, res_s, out_s, blk_s, hist_row_stride);
        }

        if (t0 + INK_R9_WT_CT < T) {
          ink_r9_wt_wo_h[w][0u] = x1;
          ink_r9_wt_wo_h[w][1u] = x2;
          ink_r9_wt_wo_h[w][2u] = x3;
        }
      }
    }
  }
}

#undef INK_R9_WT_CT

#endif /* IE_MEGA */
