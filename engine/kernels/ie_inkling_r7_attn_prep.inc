/* ============================================================================
 * Inkling-2 Round 7: Fused Attention Prep Kernels (Fusion D1 + Fusion D2)
 * Target: RX 9070 XT (gfx1201, wave32, NT=256)
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r7_ap_ /
 * INK_R7_AP_ and all macros are #undef'd at EOF.
 *
 * Exported kernels (100% bitwise-exact 0-bit-diff parity, T=1 and T=2..16):
 *
 * 1. KERNEL ie_inkling_qkv_qclip_r7(...)
 *    - Fusion D1: fuses
 *        * 10 SWA pass=-1 visits: wqkv(5120x2304) + q_clip              (2 -> 1, -10)
 *        * 30 SWA pass>=0 visits: wq + wk + wv + q_clip                 (4 -> 1, -90)
 *        *  8 Global visits:      wq + wk + wv + q_clip + k_dyn + v_dyn (6 -> 1, -40)
 *      Total: -140 launches/step across all 48 visits!
 *    - Works seamlessly for packed wqkv (pass qw_k = qw_q + qd*nb*8,
 *      qs_k = qs_q + qd*nb, qw_v = qw_q + (qd+kvd)*nb*8, etc., xq_q==xq_k==xq_v)
 *      AND separate wq/wk/wv with shared or distinct (xq_q, xq_k, xq_v).
 *    - Grid: (wg_q + wg_k + wg_v + wg_kd + wg_vd, 1, 1)
 *        wg_q  = (n_head * hd) >> 5    (96 WGs = exact 4 WGs per Q head for hd=128)
 *        wg_k  = (n_kv * hd + 31) >> 5 (32 WGs)
 *        wg_v  = (n_kv * hd + 31) >> 5 (32 WGs)
 *        wg_kd = rows_kd ? ((rows_kd + 31) >> 5) : 0 (0 or 1 WG)
 *        wg_vd = rows_vd ? ((rows_vd + 31) >> 5) : 0 (0 or 1 WG)
 *      Total = 160 WGs (SWA) or 162 WGs (Global) -> full 64-CU saturation.
 *    - Synchronization: per-Q-head counter head_cnt[0..n_head-1] (96 B, zero-init,
 *      auto-reset to 0 by the 4th arriving WG of each Q head). The 4th WG of head h
 *      runs exact ie_inkling_vclip_r1 (hd=128) over L2-hot y_q_raw[0..T-1][h]
 *      across parallel waves (tk = wv; tk < T; tk += 8) -> y_q_clip[0..T-1][h].
 *      Safe both out-of-place and in-place (y_q_clip == y_q_raw).
 *    - Exact host guard:
 *        hd == 128u && n_head > 0u && n_kv > 0u && nb >= 8u && (nb & 7u) == 0u &&
 *        T >= 1u && T <= 16u && (ys_qr & 15u) == 0u && (ys_qc & 15u) == 0u &&
 *        all outputs mutually disjoint from inputs (and on 20 SWA pass>0 + 8 Global
 *        visits, kq/vq kept live past q_clip and v_dyn hoisted/disjoint).
 *    - Parity: 0-bit-diff vs ie_gemv_q8q8 (T=1) / ie_gemv_q8q8_tr (T=2..16)
 *      and ie_inkling_vclip_r1 (hd=128).
 *
 * 2. KERNEL ie_inkling_rel_clip_sink_gate_r7(...)
 *    - Fusion D2: fuses the 4 consecutive shared-qq ops on ALL 48 visits:
 *        wrel(384x2304) + r_clip(24x16) + wsink(24x2304) + wagt(3072x2304)
 *      (4 -> 1 across 48 visits = -144 launches/step!).
 *    - ZERO atomic counters, ZERO model.c edits, completely elides r_raw writes!
 *    - Grid: (wg_r + wg_s + wg_g, 1, 1)
 *        wg_r = (n_head + 7) >> 3      (3 WGs: 1 wave owns 1 full 16-row r head!)
 *        wg_s = (n_head + 31) >> 5     (1 WG for wsink, 24 rows)
 *        wg_g = (rows_g + 31) >> 5     (96 WGs for wagt, 3072 rows)
 *      Total = 100 WGs -> all 100 WGs share L2-hot xq (qq).
 *    - Exact host guard:
 *        d_rel == 16u && n_head > 0u && rows_g > 0u && nb >= 8u && (nb & 7u) == 0u &&
 *        T >= 1u && T <= 16u && (ys_rc & 15u) == 0u &&
 *        (y_rclip, y_sink, y_gate) mutually disjoint and disjoint from xq.
 *    - Parity: 0-bit-diff vs ie_gemv_q8q8[_tr] + ie_inkling_vclip_r1 (hd=16).
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R7_AP_MT_CT 4u /* Token chunk for T=2..16: keeps VGPR <= 96 (4 WGs/CU) */

struct __attribute__((packed, aligned(4))) ink_r7_ap_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r7_ap_ld4(const G void *p) {
  return ((const G struct ink_r7_ap_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) u32x4 ink_r7_ap_ld4_nt(const G u32 *p) {
  return __builtin_nontemporal_load((const G u32x4 *)p);
}

static inline __attribute__((always_inline)) int ink_r7_ap_dot8(u32x4 w0, u32x4 w1, u32x4 a0, u32x4 a1) {
  u32 d = ie_dot4_ss(0u, w0.x, a0.x);
  d = ie_dot4_ss(d, w0.y, a0.y);
  d = ie_dot4_ss(d, w0.z, a0.z);
  d = ie_dot4_ss(d, w0.w, a0.w);
  d = ie_dot4_ss(d, w1.x, a1.x);
  d = ie_dot4_ss(d, w1.y, a1.y);
  d = ie_dot4_ss(d, w1.z, a1.z);
  d = ie_dot4_ss(d, w1.w, a1.w);
  return (int)d;
}

/* Exact ie_inkling_vclip_r1 scale formula (matches ink_r1_g08_vclip_scale). */
static inline __attribute__((always_inline)) float ink_r7_ap_vclip_scale(
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

/* 0-bit-diff T=1 Q8_0 GEMV (4 rows/wave, 8 lanes/row, 32 rows/WG).
 * Exact 4-acc virtual-lane wave_tree_f order matching ie_gemv_q8q8 / ink_r2_g10_gemv_q8_4rw. */
static inline __attribute__((always_inline)) float ink_r7_ap_q8_dot_8l(
    const G u32 *row_qw, const G f16 *row_qs, const G u8 *xq, u32 nb, u32 sub) {
#pragma clang fp contract(off)
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
    const u32x4 w0_0 = ink_r7_ap_ld4(wp0);
    const u32x4 w0_1 = ink_r7_ap_ld4(wp0 + 4u);
    const u32x4 x0_0 = ink_r7_ap_ld4(ap0);
    const u32x4 x0_1 = ink_r7_ap_ld4(ap0 + 4u);
    const float sc0 = (float)row_qs[b0] * da[b0];
    const u32x4 w1_0 = ink_r7_ap_ld4(wp1);
    const u32x4 w1_1 = ink_r7_ap_ld4(wp1 + 4u);
    const u32x4 x1_0 = ink_r7_ap_ld4(ap1);
    const u32x4 x1_1 = ink_r7_ap_ld4(ap1 + 4u);
    const float sc1 = (float)row_qs[b1] * da[b1];

    const int d0 = ink_r7_ap_dot8(w0_0, w0_1, x0_0, x0_1);
    a0 = a0 + sc0 * (float)d0;
    const int d1 = ink_r7_ap_dot8(w1_0, w1_1, x1_0, x1_1);
    a1 = a1 + sc1 * (float)d1;

    const u32 b2 = b0 + 16u;
    const u32 b3 = b0 + 24u;
    const G u32 *wp2 = row_qw + (unsigned long)b2 * 8ul;
    const G u32 *wp3 = row_qw + (unsigned long)b3 * 8ul;
    const G u32 *ap2 = aw + (unsigned long)b2 * 8ul;
    const G u32 *ap3 = aw + (unsigned long)b3 * 8ul;
    const u32x4 w2_0 = ink_r7_ap_ld4(wp2);
    const u32x4 w2_1 = ink_r7_ap_ld4(wp2 + 4u);
    const u32x4 x2_0 = ink_r7_ap_ld4(ap2);
    const u32x4 x2_1 = ink_r7_ap_ld4(ap2 + 4u);
    const float sc2 = (float)row_qs[b2] * da[b2];
    const u32x4 w3_0 = ink_r7_ap_ld4(wp3);
    const u32x4 w3_1 = ink_r7_ap_ld4(wp3 + 4u);
    const u32x4 x3_0 = ink_r7_ap_ld4(ap3);
    const u32x4 x3_1 = ink_r7_ap_ld4(ap3 + 4u);
    const float sc3 = (float)row_qs[b3] * da[b3];

    const int d2 = ink_r7_ap_dot8(w2_0, w2_1, x2_0, x2_1);
    a2 = a2 + sc2 * (float)d2;
    const int d3 = ink_r7_ap_dot8(w3_0, w3_1, x3_0, x3_1);
    a3 = a3 + sc3 * (float)d3;
  }

  const u32 rem = nb - nb_main;
  if (rem > 0u) {
    const u32 b0 = nb_main + sub;
    if (b0 < nb) {
      const G u32 *wp0 = row_qw + (unsigned long)b0 * 8ul;
      const G u32 *ap0 = aw + (unsigned long)b0 * 8ul;
      const int d0 = ink_r7_ap_dot8(ink_r7_ap_ld4(wp0), ink_r7_ap_ld4(wp0 + 4u),
                                    ink_r7_ap_ld4(ap0), ink_r7_ap_ld4(ap0 + 4u));
      a0 = a0 + ((float)row_qs[b0] * da[b0]) * (float)d0;
    }
    if (rem > 8u) {
      const u32 b1 = nb_main + 8u + sub;
      if (b1 < nb) {
        const G u32 *wp1 = row_qw + (unsigned long)b1 * 8ul;
        const G u32 *ap1 = aw + (unsigned long)b1 * 8ul;
        const int d1 = ink_r7_ap_dot8(ink_r7_ap_ld4(wp1), ink_r7_ap_ld4(wp1 + 4u),
                                      ink_r7_ap_ld4(ap1), ink_r7_ap_ld4(ap1 + 4u));
        a1 = a1 + ((float)row_qs[b1] * da[b1]) * (float)d1;
      }
    }
    if (rem > 16u) {
      const u32 b2 = nb_main + 16u + sub;
      if (b2 < nb) {
        const G u32 *wp2 = row_qw + (unsigned long)b2 * 8ul;
        const G u32 *ap2 = aw + (unsigned long)b2 * 8ul;
        const int d2 = ink_r7_ap_dot8(ink_r7_ap_ld4(wp2), ink_r7_ap_ld4(wp2 + 4u),
                                      ink_r7_ap_ld4(ap2), ink_r7_ap_ld4(ap2 + 4u));
        a2 = a2 + ((float)row_qs[b2] * da[b2]) * (float)d2;
      }
    }
  }

  const float u0 = a0 + a2;
  const float u1 = a1 + a3;
  float s = u0 + u1;
  s = s + I2F(shl_i(F2I(s), 4));
  s = s + I2F(shl_i(F2I(s), 2));
  s = s + I2F(shl_i(F2I(s), 1));
  return s;
}

static inline __attribute__((always_inline)) void ink_r7_ap_gemv_q8_t1(
    u32 g, const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb) {
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 r = (g << 5u) + (w << 2u) + (l >> 3u);
  if (r < rows) {
    const G u32 *row_qw = qw + (unsigned long)r * (unsigned long)nb * 8ul;
    const G f16 *row_qs = qs + (unsigned long)r * (unsigned long)nb;
    const float s = ink_r7_ap_q8_dot_8l(row_qw, row_qs, xq, nb, sub);
    if (sub == 0u) y[r] = s;
  }
}

/* 0-bit-diff T=2..16 Q8_0 GEMV (4 rows/wave, 32 lanes/row, 32 rows/WG).
 * Matches ie_gemv_q8q8_tr / ie_gemv_q8q8 lane assignment and wave_tree_f exactly.
 * Processes 2 rows per inner half-pass so weight VGPRs stay at 16 (total <= 80 VGPR -> 4 WGs/CU). */
static inline __attribute__((always_inline)) void ink_r7_ap_gemv_q8_mt(
    u32 g, const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
    u32 rows, u32 nb, u32 T, u32 xs, u32 ys_bytes) {
#pragma clang fp contract(off)
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 r_wave = (g << 5u) + (w << 2u);
  if (r_wave >= rows) return;

#pragma clang loop unroll(disable)
  for (u32 half = 0u; half < 4u; half += 2u) {
    const u32 r0 = r_wave + half;
    if (r0 >= rows) break;

    for (u32 t0 = 0u; t0 < T; t0 += INK_R7_AP_MT_CT) {
      const u32 tc = (T - t0 < INK_R7_AP_MT_CT) ? (T - t0) : INK_R7_AP_MT_CT;
      float acc[2u][INK_R7_AP_MT_CT];
#pragma unroll
      for (u32 q = 0u; q < 2u; q++)
#pragma unroll
        for (u32 j = 0u; j < INK_R7_AP_MT_CT; j++) acc[q][j] = 0.0f;

      for (u32 b = l; b < nb; b += 32u) {
        u32x4 w0[2u], w1[2u];
        float sw[2u];
#pragma unroll
        for (u32 q = 0u; q < 2u; q++) {
          const u32 r = (r0 + q < rows) ? (r0 + q) : r0;
          const G u32 *wp = qw + ((unsigned long)r * (unsigned long)nb + (unsigned long)b) * 8ul;
          w0[q] = ink_r7_ap_ld4_nt(wp);
          w1[q] = ink_r7_ap_ld4_nt(wp + 4u);
          sw[q] = (float)qs[(unsigned long)r * (unsigned long)nb + (unsigned long)b];
        }
#pragma unroll
        for (u32 j = 0u; j < INK_R7_AP_MT_CT; j++) {
          if (j < tc) {
            const G u8 *xj = xq + (unsigned long)(t0 + j) * (unsigned long)xs;
            const G u32 *aw = (const G u32 *)xj;
            const u32x4 a0 = ink_r7_ap_ld4(aw + (unsigned long)b * 8ul);
            const u32x4 a1 = ink_r7_ap_ld4(aw + (unsigned long)b * 8ul + 4ul);
            const float da = ((const G float *)(xj + (unsigned long)nb * 32ul))[b];
#pragma unroll
            for (u32 q = 0u; q < 2u; q++) {
              const int dot = ink_r7_ap_dot8(w0[q], w1[q], a0, a1);
              const float sc = sw[q] * da;
              acc[q][j] = acc[q][j] + sc * (float)dot;
            }
          }
        }
      }

#pragma unroll
      for (u32 q = 0u; q < 2u; q++) {
        const u32 r = r0 + q;
#pragma unroll
        for (u32 j = 0u; j < INK_R7_AP_MT_CT; j++) {
          if (j < tc) {
            const float v = wave_tree_f(acc[q][j]);
            if (l == 0u && r < rows) {
              G float *yj = (G float *)((G u8 *)y + (unsigned long)(t0 + j) * (unsigned long)ys_bytes);
              yj[r] = v;
            }
          }
        }
      }
    }
  }
}

static inline __attribute__((always_inline)) void ink_r7_ap_gemv_q8(
    u32 g, const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
    u32 rows, u32 nb, u32 T, u32 xs, u32 ys_bytes) {
  if (T == 1u) {
    ink_r7_ap_gemv_q8_t1(g, qw, qs, xq, y, rows, nb);
  } else {
    ink_r7_ap_gemv_q8_mt(g, qw, qs, xq, y, rows, nb, T, xs, ys_bytes);
  }
}

/* ============================================================================
 * Kernel D1: ie_inkling_qkv_qclip_r7
 * Fuses (wqkv | wq+wk+wv) + q_clip [+ k_dyn + v_dyn] into 1 launch.
 * Single unified GEMV call keeps SGPRs low (0/0 spills) and max 4 WGs/CU.
 * ========================================================================== */

KERNEL ie_inkling_qkv_qclip_r7(
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
  static LDS u32 ink_r7_ap_q_last;

  if (T == 0u || hd != 128u || n_head == 0u || n_kv == 0u || nb == 0u) return;

  const u32 qd = n_head * 128u;
  const u32 kvd = n_kv * 128u;
  const u32 wg_q = qd >> 5u; /* exact 4 WGs per Q head */
  const u32 wg_k = (kvd + 31u) >> 5u;
  const u32 wg_v = (kvd + 31u) >> 5u;
  const u32 wg_kd = rows_kd ? ((rows_kd + 31u) >> 5u) : 0u;
  const u32 wg_vd = rows_vd ? ((rows_vd + 31u) >> 5u) : 0u;
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

  ink_r7_ap_gemv_q8(g_sel, qw_sel, qs_sel, xq_sel, y_sel, rows_sel, nb, T, xs_sel, ys_sel);
  if (!is_q) return;

  /* Atomic tail q_clip on the 4th arriving WG of Q head h = g >> 2. */
  const u32 h = g >> 2u;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r7_ap_q_last =
        (__scoped_atomic_fetch_add(head_cnt + h, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == 3u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r7_ap_q_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  if (tid() == 0u) head_cnt[h] = 0u;

  /* Exact ie_inkling_vclip_r1 (hd == 128): 1 wave per token, up to 8 tokens in parallel. */
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
    const float f = ink_r7_ap_vclip_scale(ss, 1.0f / 128.0f, clamp, norope, escale, win, pos0, tk);
    f32x4 outv = xv * f;
    if (clamp == 0.0f && vp) outv *= vv;
    bh[l] = outv;
  }
}

/* ============================================================================
 * Kernel D2: ie_inkling_rel_clip_sink_gate_r7
 * Fuses wrel + r_clip + wsink + wagt (all 4 sharing xq = qq) with 0 atomics.
 * ========================================================================== */

KERNEL ie_inkling_rel_clip_sink_gate_r7(
    const G u32 *qw_r, const G f16 *qs_r, const G float *v_rclip, G float *y_rclip,
    const G u32 *qw_s, const G f16 *qs_s, G float *y_sink,
    const G u32 *qw_g, const G f16 *qs_g, G float *y_gate,
    const G u8 *xq, u32 n_head, u32 d_rel, u32 rows_g, u32 nb,
    float clamp, u32 norope, float escale, u32 win, u32 pos0, u32 T,
    u32 xs, u32 ys_rc, u32 ys_s, u32 ys_g) {
  /* Wave-local staging for 1 full 16-row r head (8 * 16 * 4 = 512 B LDS). */
  static LDS float ink_r7_ap_rbuf[8u][16u] __attribute__((aligned(16)));

  if (T == 0u || d_rel != 16u || n_head == 0u || rows_g == 0u || nb == 0u) return;

  const u32 wg_r = (n_head + 7u) >> 3u;  /* 8 heads/WG -> 3 WGs for n_head=24 */
  const u32 wg_s = (n_head + 31u) >> 5u; /* 1 WG for n_head=24 */
  const u32 wg_g = (rows_g + 31u) >> 5u; /* 96 WGs for rows_g=3072 */
  const u32 g = wgid();

  if (g < wg_r) {
    /* Each wave w computes ALL 16 rows of head h = g*8 + w per token, then clips directly to y_rclip.
     * Reuses exact 0-bit-diff 8-lane Q8 dot (weights are 313 KiB/WG -> L2-hot for T>1) while
     * keeping the whole kernel <= 92 VGPR (4 WGs/CU) for all 96 wagt WGs! */
    const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
    const u32 l = lane();
    const u32 h = (g << 3u) + w;
    const u32 sub = l & 7u;
    const u32 qrow = l >> 3u;

    for (u32 tk = 0u; tk < T; tk++) {
      const G u8 *xt = xq + (unsigned long)tk * (unsigned long)xs;
      if (h < n_head) {
#pragma clang loop unroll(disable)
        for (u32 grp = 0u; grp < 4u; grp++) {
          const u32 r = h * 16u + grp * 4u + qrow;
          const G u32 *row_qw = qw_r + (unsigned long)r * (unsigned long)nb * 8ul;
          const G f16 *row_qs = qs_r + (unsigned long)r * (unsigned long)nb;
          const float s = ink_r7_ap_q8_dot_8l(row_qw, row_qs, xt, nb, sub);
          if (sub == 0u) ink_r7_ap_rbuf[w][grp * 4u + qrow] = s;
        }
      }
      lbarrier();
      if (h < n_head && l < 4u) {
        const G f32x4 *vp = v_rclip ? ((const G f32x4 *)v_rclip + (unsigned long)h * 4ul) : (const G f32x4 *)0;
        const f32x4 vv = vp ? vp[l] : (f32x4){1.0f, 1.0f, 1.0f, 1.0f};
        f32x4 xv = *(const LDSP f32x4 *)(ink_r7_ap_rbuf[w] + (l << 2u));
        if (clamp > 0.0f && vp) xv *= vv;
        float ss = (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
        ss += shfl_xor_f(ss, 2u);
        ss += shfl_xor_f(ss, 1u);
        const float f = ink_r7_ap_vclip_scale(ss, 1.0f / 16.0f, clamp, norope, escale, win, pos0, tk);
        f32x4 outv = xv * f;
        if (clamp == 0.0f && vp) outv *= vv;
        G f32x4 *bh = (G f32x4 *)((G u8 *)y_rclip + (unsigned long)tk * (unsigned long)ys_rc)
                      + (unsigned long)h * 4ul;
        bh[l] = outv;
      }
      if (T > 1u) lbarrier();
    }
    return;
  }

  /* Unified wsink + wagt GEMV call -> single code copy, low SGPR & VGPR. */
  const u32 g1 = g - wg_r;
  if (g1 >= wg_s + wg_g) return;
  const int is_s = (g1 < wg_s);
  const G u32 *qw_sel = is_s ? qw_s : qw_g;
  const G f16 *qs_sel = is_s ? qs_s : qs_g;
  G float     *y_sel  = is_s ? y_sink : y_gate;
  const u32 g_sel     = is_s ? g1 : (g1 - wg_s);
  const u32 rows_sel  = is_s ? n_head : rows_g;
  const u32 ys_sel    = is_s ? ys_s : ys_g;
  ink_r7_ap_gemv_q8(g_sel, qw_sel, qs_sel, xq, y_sel, rows_sel, nb, T, xs, ys_sel);
}

#undef INK_R7_AP_MT_CT

#endif /* IE_MEGA */
