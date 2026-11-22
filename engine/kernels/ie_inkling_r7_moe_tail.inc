/* ============================================================================
 * Inkling-2-Mini Round 7 (Fusion B): MoE Down-Projection + Ordered 8-Slot Combine +
 *                                    Post-FFN Short-Conv + Residual Add [+ sc_hist]
 * Target: RX 9070 XT (gfx1201, wave32, NT=256), Clang 21 bit_cast-safe, 0 spills.
 *
 * Included at EOF of kernels/ie_inkling_kernels.c (after ie_kernels.c).
 * All local symbols/LDS/macros prefixed ink_r7_mt_ / INK_R7_MT_; macros #undef'd at EOF.
 *
 * Replaces across ALL 48 logical visits (dim=2304, eff=1152, nb=36, n_exp=128, K=8, IQ4_NL):
 *   - B1 (3-op tail, 0 counters): [OP_MOE_SUM(res=-1), OP_SCONV(ffn_pc), OP_ADD (+ sc_hist)]
 *       * T == 1:     3 -> 1 launch/visit (-96/step)
 *       * T == 2..16: 3 -> 1 launch/visit (-96/step, kills moe_sum + sconv_add + sconv_hist)
 *   - B2 (full 4-op, 288 B rcount): [OP_GEMV_EXP(IQ4_NL), OP_MOE_SUM, OP_SCONV, OP_ADD (+ sc_hist)]
 *       * T == 1:     4 -> 1 launch/visit (-144/step, preserves full 576-WG BW!)
 *       * T == 2..16: 4 -> 1 launch/visit (-144/step, preserves compact_r6 active-pack BW!)
 *
 * CRITICAL ARENA LIFETIME REQUIREMENT (src/model.c:3305):
 *   In model.c:3292-3305, x2 (OP_ADD output) is created AFTER mq (dies at exp_down),
 *   yd (dies at moe_sum), and rt (dies at moe_sum). Without extending lifetimes,
 *   plan_arena WILL alias x2 onto mq/yd/rt!
 *   - Required 1-line model.c edit right after OP_ADD(ffn_inp, fpc, x2):
 *       use_last(B, mq); use_last(B, yd); use_last(B, rt);
 *   - Required hip.c disjointness guards:
 *       B1: buf_disjoint(g, o0->a, o2->c) && buf_disjoint(g, o0->c, o2->c)
 *           (o2->c may alias res).
 *       B2: buf_disjoint(g, o0->a, o3->c) && buf_disjoint(g, o0->b, o3->c) &&
 *           buf_disjoint(g, o0->c, o3->c) && buf_disjoint(g, o0->a, o0->b) &&
 *           buf_disjoint(g, o0->c, o0->b).
 *
 * Device counter requirement for B2 (Kernels 2 & 3):
 *   - Allocate u32 *ink_moe_rcount of size ((m->dim + 31u) / 32u) * 4u (288 B for dim=2304),
 *     zeroed once at gpu_open and freed in gpu_free. Each row-block self-resets rcount[rb] = 0u.
 *
 * Exported KERNELs, Grids, Guards, and Parity:
 *
 * 1. KERNEL ie_inkling_moe_sum_sconv_add_r7(
 *        const G float *y, const G u32 *route, const G float *v, G float *ring,
 *        const G float *res, G float *out, G float *hist,
 *        u32 n, u32 K, float escale, u32 pos0, u32 T,
 *        u32 ys, u32 rts, u32 rs, u32 os, u32 hrs)
 *    - Grid:   (((n >> 2u) + 255u) / 256u, 1u, 1u)  [(3, 1, 1) for dim=2304, T=1..16]
 *    - Guard:  !mk.rec && T >= 1u && T <= 16u && o0->kind == OP_MOE_SUM && o0->res < 0 &&
 *              o1->kind == OP_SCONV && o1->c < 0 && o1->a == o0->b && o1->aoff == 0u &&
 *              o1->n == o0->n && o2->kind == OP_ADD && o2->n == o0->n &&
 *              (o2->a == o1->b || o2->b == o1->b) && (n & 3u) == 0u && K > 0u &&
 *              16-byte align on y, v, ring, res, out (, hist) &&
 *              (T == 1u || ((ys | rs | os) & 15u) == 0u) && (!hist || (hrs & 3u) == 0u).
 *              NOTE: ys = g->bufs[o0->a].stride (token byte stride of exp_down [T, K*dim]),
 *                    rts = g->bufs[o0->c].stride, hrs = b->sc_hist_stride (floats).
 *    - Parity: 100% bitwise-exact (0-bit-diff) vs ie_inkling_moe_sum_r1 + ie_inkling_sconv_r1 +
 *              ie_add[_t] + (ie_inkling_sconv_hist_r4 | ie_sconv_ring_store).
 *
 * 2. KERNEL ie_inkling_exp_down_tail_t1_r7(
 *        const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *yd,
 *        u32 rows, u32 nb, const G u32 *route, u32 K, u32 qwn, u32 qsn,
 *        u32 astr, u32 bstr,
 *        const G float *v, G float *ring, const G float *res, G float *out,
 *        float escale, u32 pos0, G u32 *rcount)
 *    - Grid:   ((rows + 31u) / 32u, K, 1u)  [(72, 8, 1) = 576 WGs]
 *    - Guard:  !mk.rec && T == 1u && o0->w->kind == MAT_IQ4NL (8u) &&
 *              rows > 0u && (rows & 3u) == 0u && nb > 0u && K > 0u &&
 *              o0->bstr == rows * 4u && b->ink_moe_rcount != NULL &&
 *              16-byte align on yd, v, ring, res, out && disjointness above.
 *    - Parity: 100% bitwise-exact (0-bit-diff) vs ie_inkling_gexp_iq4nl_r1 +
 *              ie_inkling_moe_sum_r1 + ie_inkling_sconv_r1 + ie_add.
 *
 * 3. KERNEL ie_inkling_exp_down_tail_ver_r7(
 *        const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
 *        u32 rows, u32 nb,
 *        const G u32 *list, const G u32 *packs, const G u32 *n_packs,
 *        u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv,
 *        const G u32 *route, u32 K, u32 rts,
 *        const G float *v, G float *ring, const G float *res, G float *out, G float *hist,
 *        float escale, u32 pos0, u32 T, u32 yts, u32 rs, u32 os, u32 hrs,
 *        G u32 *rcount)
 *    - Grid:   ((rows + 31u) / 32u, max_packs, 1u)
 *              where max_packs = min(T * K, n_exp + (T * K) / 4u).
 *    - Guard:  !mk.rec && T >= 2u && T <= 16u && o0->w->kind == MAT_IQ4NL (8u) &&
 *              b->ms_list && b->ms_packs && b->ms_npacks && b->ink_moe_rcount &&
 *              rows > 0u && (rows & 3u) == 0u && nb > 0u && K > 0u &&
 *              ys == rows * 4u && yts == K * ys &&
 *              16-byte align on Y, v, ring, res, out (, hist) &&
 *              ((yts | rs | os) & 15u) == 0u && (!hist || (hrs & 3u) == 0u) &&
 *              disjointness above.
 *              NOTE: ys = o0->bstr (slot byte stride = rows * 4), yts = g->bufs[o0->b].stride
 *                    (token byte stride = K * rows * 4).
 *    - Parity: 100% bitwise-exact (0-bit-diff) vs ie_inkling_gexp_iq4nl_compact_r6 +
 *              ie_inkling_moe_sum_r1 + ie_inkling_sconv_add_r2 +
 *              (ie_inkling_sconv_hist_r4 | ie_sconv_ring_store).
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(f16) == 2u, "ink_r7_mt f16 must be 2 bytes");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r7_mt float/u32 must be 4 bytes");

#define INK_R7_MT_ACC_FMA(a0_, a1_, a2_, a3_, j_, v_, d_)          \
  do {                                                             \
    if ((j_) == 0u) (a0_) = __builtin_fmaf((v_), (d_), (a0_));     \
    else if ((j_) == 1u) (a1_) = __builtin_fmaf((v_), (d_), (a1_));\
    else if ((j_) == 2u) (a2_) = __builtin_fmaf((v_), (d_), (a2_));\
    else (a3_) = __builtin_fmaf((v_), (d_), (a3_));                \
  } while (0)

static inline __attribute__((always_inline)) u32 ink_r7_mt_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

static inline __attribute__((always_inline)) float ink_r7_mt_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline __attribute__((always_inline)) float ink_r7_mt_oct_tree_f(float v) {
#pragma clang fp contract(off)
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}

/* Clang 21 safe IQ4_NL dot4 (scalar lane extraction). */
static inline __attribute__((always_inline)) int ink_r7_mt_iq4_dot4(
    const u32x4 w4, const u32x4 a0, const u32x4 a1) {
  const u32 wx = w4.x, wy = w4.y, wz = w4.z, ww = w4.w;
  const u32 a00 = a0.x, a01 = a0.y, a02 = a0.z, a03 = a0.w;
  const u32 a10 = a1.x, a11 = a1.y, a12 = a1.z, a13 = a1.w;
  u32 d = 0u;
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(wx & 0x0F0F0F0Fu), a00), ie_iq4_map((wx >> 4u) & 0x0F0F0F0Fu), a10);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(wy & 0x0F0F0F0Fu), a01), ie_iq4_map((wy >> 4u) & 0x0F0F0F0Fu), a11);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(wz & 0x0F0F0F0Fu), a02), ie_iq4_map((wz >> 4u) & 0x0F0F0F0Fu), a12);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(ww & 0x0F0F0F0Fu), a03), ie_iq4_map((ww >> 4u) & 0x0F0F0F0Fu), a13);
  return (int)d;
}

static inline __attribute__((always_inline)) int ink_r7_mt_iq4_dot_pre(
    const u32 wl0, const u32 wh0, const u32 wl1, const u32 wh1,
    const u32 wl2, const u32 wh2, const u32 wl3, const u32 wh3,
    const u32x4 a0, const u32x4 a1) {
  const u32 a00 = a0.x, a01 = a0.y, a02 = a0.z, a03 = a0.w;
  const u32 a10 = a1.x, a11 = a1.y, a12 = a1.z, a13 = a1.w;
  u32 d = 0u;
  d = ie_dot4_ss(ie_dot4_ss(d, wl0, a00), wh0, a10);
  d = ie_dot4_ss(ie_dot4_ss(d, wl1, a01), wh1, a11);
  d = ie_dot4_ss(ie_dot4_ss(d, wl2, a02), wh2, a12);
  d = ie_dot4_ss(ie_dot4_ss(d, wl3, a03), wh3, a13);
  return (int)d;
}

/* Exact ordered K-slot weighted MoE combine for 1 f32x4 at channel group i4. */
static inline __attribute__((always_inline)) f32x4 ink_r7_mt_moe_combine4(
    const G f32x4 *y4, const G u32 *rt, u32 n4, u32 K, u32 i4) {
#pragma clang fp contract(off)
  const G float *w = (const G float *)(rt + K);
  f32x4 x0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (K == 8u) {
    const float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    const float w4 = w[4], w5 = w[5], w6 = w[6], w7 = w[7];
    const f32x4 y0 = y4[0ul * n4 + i4];
    const f32x4 y1 = y4[1ul * n4 + i4];
    const f32x4 y2 = y4[2ul * n4 + i4];
    const f32x4 y3 = y4[3ul * n4 + i4];
    const f32x4 y4v = y4[4ul * n4 + i4];
    const f32x4 y5 = y4[5ul * n4 + i4];
    const f32x4 y6 = y4[6ul * n4 + i4];
    const f32x4 y7 = y4[7ul * n4 + i4];
#pragma unroll
    for (u32 c = 0u; c < 4u; c++) {
      float acc = __builtin_fmaf(w0, y0[c], 0.0f);
      acc = __builtin_fmaf(w1, y1[c], acc);
      acc = __builtin_fmaf(w2, y2[c], acc);
      acc = __builtin_fmaf(w3, y3[c], acc);
      acc = __builtin_fmaf(w4, y4v[c], acc);
      acc = __builtin_fmaf(w5, y5[c], acc);
      acc = __builtin_fmaf(w6, y6[c], acc);
      acc = __builtin_fmaf(w7, y7[c], acc);
      x0[c] = acc;
    }
  } else {
    for (u32 k = 0u; k < K; k++) {
      const float wk = w[k];
      const f32x4 yk = y4[(unsigned long)k * n4 + i4];
#pragma unroll
      for (u32 c = 0u; c < 4u; c++) x0[c] = __builtin_fmaf(wk, yk[c], x0[c]);
    }
  }
  return x0;
}

/* Shared FFN tail step over t = 0..T-1 for channel group i4 (4 floats = i0..i0+3).
 * Exact K-slot ordered fmaf + exact non-FMA static 4-tap sconv + residual add + hist/ring. */
static inline __attribute__((always_inline)) void ink_r7_mt_tail_loop(
    const G u8 *y_base, const G u8 *rt_base, const G float *v, G float *ring,
    const G u8 *res_base, G u8 *out_base, G float *hist,
    u32 n4, u32 i4, u32 K, float escale, u32 pos0, u32 T,
    u32 yts, u32 rts, u32 rs, u32 os, u32 hrs) {
#pragma clang fp contract(off)
  const u32 i0 = i4 << 2u;

  /* Load 4-tap static conv weights once for all T tokens. */
  const G f32x4 *v4 = (const G f32x4 *)v + i0;
  const f32x4 vw0 = v4[0], vw1 = v4[1], vw2 = v4[2], vw3 = v4[3];
  const f32x4 sw0 = (f32x4){vw0.w, vw1.w, vw2.w, vw3.w};
  const f32x4 sw1 = (f32x4){vw0.z, vw1.z, vw2.z, vw3.z};
  const f32x4 sw2 = (f32x4){vw0.y, vw1.y, vw2.y, vw3.y};
  const f32x4 sw3 = (f32x4){vw0.x, vw1.x, vw2.x, vw3.x};

  /* Seed 3-token lag window from ring once. */
  f32x4 x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 x4 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (pos0 > 0u) {
    const G f32x4 *rg = (const G f32x4 *)ring + i0;
    const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
    const u32 s1 = (pos0 - 1u) & 3u;
    x1 = (f32x4){ink_r7_mt_pick4(r0, s1), ink_r7_mt_pick4(r1, s1),
                 ink_r7_mt_pick4(r2, s1), ink_r7_mt_pick4(r3, s1)};
    if (pos0 >= 2u) {
      const u32 s2 = (pos0 - 2u) & 3u;
      x2 = (f32x4){ink_r7_mt_pick4(r0, s2), ink_r7_mt_pick4(r1, s2),
                   ink_r7_mt_pick4(r2, s2), ink_r7_mt_pick4(r3, s2)};
    }
    if (pos0 >= 3u) {
      const u32 s3 = (pos0 - 3u) & 3u;
      x3 = (f32x4){ink_r7_mt_pick4(r0, s3), ink_r7_mt_pick4(r1, s3),
                   ink_r7_mt_pick4(r2, s3), ink_r7_mt_pick4(r3, s3)};
    }
  }

  for (u32 t = 0u; t < T; t++) {
    const G f32x4 *y4 = (const G f32x4 *)(y_base + (unsigned long)t * yts);
    const G u32 *rt = (const G u32 *)(rt_base + (unsigned long)t * rts);
    const f32x4 rv = ((const G f32x4 *)(res_base + (unsigned long)t * rs))[i4];

    const f32x4 x0 = ink_r7_mt_moe_combine4(y4, rt, n4, K, i4);
    if (hist) ((G f32x4 *)(hist + (unsigned long)t * hrs))[i4] = x0;

    const f32x4 cv = ((x0 * sw0 + x1 * sw1) + x2 * sw2) + x3 * sw3;
    const f32x4 sc = (x0 + cv) * escale;
    ((G f32x4 *)(out_base + (unsigned long)t * os))[i4] = rv + sc;
    x4 = x3; x3 = x2; x2 = x1; x1 = x0;
  }

  if (!hist) {
    if (T >= 4u) {
      const u32 s0 = (pos0 + T - 4u) & 3u;
      const f32x4 v0 = (s0 == 0u) ? x4 : ((s0 == 3u) ? x3 : ((s0 == 2u) ? x2 : x1));
      const f32x4 v1 = (s0 == 1u) ? x4 : ((s0 == 0u) ? x3 : ((s0 == 3u) ? x2 : x1));
      const f32x4 v2 = (s0 == 2u) ? x4 : ((s0 == 1u) ? x3 : ((s0 == 0u) ? x2 : x1));
      const f32x4 v3 = (s0 == 3u) ? x4 : ((s0 == 2u) ? x3 : ((s0 == 1u) ? x2 : x1));
      G f32x4 *rg = (G f32x4 *)ring + i0;
      rg[0] = (f32x4){v0.x, v1.x, v2.x, v3.x};
      rg[1] = (f32x4){v0.y, v1.y, v2.y, v3.y};
      rg[2] = (f32x4){v0.z, v1.z, v2.z, v3.z};
      rg[3] = (f32x4){v0.w, v1.w, v2.w, v3.w};
    } else {
      if (T >= 3u) {
        const u32 sl = (pos0 + T - 3u) & 3u;
        ring[(i0 + 0u) * 4u + sl] = x3.x;
        ring[(i0 + 1u) * 4u + sl] = x3.y;
        ring[(i0 + 2u) * 4u + sl] = x3.z;
        ring[(i0 + 3u) * 4u + sl] = x3.w;
      }
      if (T >= 2u) {
        const u32 sl = (pos0 + T - 2u) & 3u;
        ring[(i0 + 0u) * 4u + sl] = x2.x;
        ring[(i0 + 1u) * 4u + sl] = x2.y;
        ring[(i0 + 2u) * 4u + sl] = x2.z;
        ring[(i0 + 3u) * 4u + sl] = x2.w;
      }
      {
        const u32 sl = (pos0 + T - 1u) & 3u;
        ring[(i0 + 0u) * 4u + sl] = x1.x;
        ring[(i0 + 1u) * 4u + sl] = x1.y;
        ring[(i0 + 2u) * 4u + sl] = x1.z;
        ring[(i0 + 3u) * 4u + sl] = x1.w;
      }
    }
  }
}

/* ============================================================================
 * Kernel 1 (B1): Standalone 3-Op FFN Tail (moe_sum + ffn_pc sconv + add [+ hist/ring])
 *   Grid: (((n >> 2u) + 255u) / 256u, 1u, 1u)  [(3, 1, 1) for dim=2304, T=1..16].
 *   0 LDS, 0 counters, 0 spills, 100% bitwise-exact.
 * ========================================================================== */

KERNEL ie_inkling_moe_sum_sconv_add_r7(
    const G float *y, const G u32 *route, const G float *v, G float *ring,
    const G float *res, G float *out, G float *hist,
    u32 n, u32 K, float escale, u32 pos0, u32 T,
    u32 ys, u32 rts, u32 rs, u32 os, u32 hrs) {
  if (T == 0u || n == 0u || (n & 3u) != 0u || K == 0u || !y || !route || !v || !ring || !res || !out) return;
  const u32 n4 = n >> 2u;
  const u32 i4 = wgid() * NT + tid();
  if (i4 >= n4) return;

  ink_r7_mt_tail_loop((const G u8 *)y, (const G u8 *)route, v, ring,
                      (const G u8 *)res, (G u8 *)out, hist,
                      n4, i4, K, escale, pos0, T, ys, rts, rs, os, hrs);
}

/* ============================================================================
 * Kernel 2 (B2-T1): Full 4-Op T=1 Decode FFN (exp_down IQ4_NL + moe_sum + sconv + add)
 *   Grid: ((rows + 31u) / 32u, K, 1u)  [(72, 8, 1) = 576 WGs].
 *   Preserves full 576-WG DRAM BW; 8th WG of each 32-row block combines its L2-hot
 *   yd[0..K-1][rb*32..rb*32+31] in exact order + sconv + ring + residual add.
 * ========================================================================== */

KERNEL ie_inkling_exp_down_tail_t1_r7(
    const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *yd,
    u32 rows, u32 nb, const G u32 *route, u32 K, u32 qwn, u32 qsn,
    u32 astr, u32 bstr,
    const G float *v, G float *ring, const G float *res, G float *out,
    float escale, u32 pos0, G u32 *rcount) {
#pragma clang fp contract(off)
  static LDS u32 ink_r7_mt_t1_last;
  if (rows == 0u || (rows & 3u) != 0u || nb == 0u || K == 0u ||
      !qw || !qs || !xq || !yd || !route || !v || !ring || !res || !out || !rcount)
    return;

  const u32 rb = wgid();
  const u32 rbase = rb * 32u;
  if (rbase >= rows) return;

  const u32 s = tokid();
  if (s >= K) return;

  const u32 t0 = tid();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 r_wave = rbase + wv * 4u;
  const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
  const u32 r = r_wave + sub;

  const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[s]);
  const G u8 *w_base = rq_ptr_u((const G u8 *)(qw + (unsigned long)e * qwn));
  const G u8 *q_base = rq_ptr_u((const G u8 *)(qs + (unsigned long)e * qsn));
  const G u8 *xp = rq_ptr_u(xq + (unsigned long)s * astr);
  G float *o = (G float *)((G u8 *)yd + (unsigned long)s * bstr);

  /* Phase 1: exact r3_g19 / r1_g04 4-row/wave IQ4_NL GEMV (all waves stay convergent for barrier). */
  if (r_wave < rows) {
    const u32 rr = r < rows ? r : rows - 1u;
    const G u8 *rw = w_base + (unsigned long)rr * ((unsigned long)nb * 16ul);
    const G f16 *rs_p = (const G f16 *)q_base + (unsigned long)rr * nb;
    const G u32 *aw = (const G u32 *)xp;
    const G float *da = (const G float *)(xp + 32ul * (unsigned long)nb);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const u32 nstep = (nb + 7u) >> 3u;

    u32x4 w_c = (u32x4)(0u), x0_c = (u32x4)(0u), x1_c = (u32x4)(0u);
    f16 sc_c = (f16)0.0f;
    float d_c = 0.0f;
    if (sl < nb) {
      w_c = *(const G u32x4 *)(rw + sl * 16u);
      sc_c = rs_p[sl];
      x0_c = *(const G u32x4 *)(aw + sl * 8u);
      x1_c = *(const G u32x4 *)(aw + sl * 8u + 4u);
      d_c = da[sl];
    }
    __builtin_amdgcn_sched_barrier(0);

#pragma unroll 1
    for (u32 step = 0u; step < nstep; step++) {
      const u32 b_c = step * 8u + sl;
      const u32 bn = b_c + 8u;
      u32x4 w_n = (u32x4)(0u), x0_n = (u32x4)(0u), x1_n = (u32x4)(0u);
      f16 sc_n = (f16)0.0f;
      float d_n = 0.0f;
      if (bn < nb) {
        w_n = *(const G u32x4 *)(rw + bn * 16u);
        sc_n = rs_p[bn];
        x0_n = *(const G u32x4 *)(aw + bn * 8u);
        x1_n = *(const G u32x4 *)(aw + bn * 8u + 4u);
        d_n = da[bn];
      }
      __builtin_amdgcn_sched_barrier(0);
      if (b_c < nb) {
        const float val = (float)sc_c * (float)ink_r7_mt_iq4_dot4(w_c, x0_c, x1_c);
        const u32 j = step & 3u;
        INK_R7_MT_ACC_FMA(a0, a1, a2, a3, j, val, d_c);
      }
      __builtin_amdgcn_sched_barrier(0);
      w_c = w_n; sc_c = sc_n; x0_c = x0_n; x1_c = x1_n; d_c = d_n;
    }

    const int full_wave = (nb > 16u);
    const float acc = ink_r7_mt_oct_tree_f(full_wave ? ((a0 + a2) + (a1 + a3)) : (a0 + a1));
    if (sl == 0u && r < rows) o[r] = acc;
  }

  /* Per-row-block agent-scope release/acquire: K-th slot WG finishes this 32-row slice. */
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (t0 == 0u) {
    ink_r7_mt_t1_last =
        (__scoped_atomic_fetch_add(&rcount[rb], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == K - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r7_mt_t1_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  /* Phase 2: 8 threads (8 x f32x4 = 32 rows) run exact ordered combine + sconv + ring + add. */
  const u32 n4 = rows >> 2u;
  if (t0 < 8u) {
    const u32 i4 = rb * 8u + t0;
    if (i4 < n4) {
      ink_r7_mt_tail_loop((const G u8 *)yd, (const G u8 *)route, v, ring,
                          (const G u8 *)res, (G u8 *)out, (G float *)0,
                          n4, i4, K, escale, pos0, 1u, 0u, 0u, 0u, 0u, 0u);
    }
  }
  if (t0 == 0u) rcount[rb] = 0u;
}

/* ============================================================================
 * Kernel 3 (B2-Ver): Full 4-Op T=2..16 Verify FFN
 *   (compact IQ4_NL exp_down + ordered moe_sum + ffn_pc sconv + add [+ sc_hist / ring])
 *   Grid: ((rows + 31u) / 32u, max_packs, 1u).
 *   Inactive packs (by >= *n_packs) return immediately before touching rcount.
 *   npk-th active pack of row-block rb finishes all T<=16 tokens for its 32 L2-hot rows.
 * ========================================================================== */

KERNEL ie_inkling_exp_down_tail_ver_r7(
    const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
    u32 rows, u32 nb,
    const G u32 *list, const G u32 *packs, const G u32 *n_packs,
    u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv,
    const G u32 *route, u32 K, u32 rts,
    const G float *v, G float *ring, const G float *res, G float *out, G float *hist,
    float escale, u32 pos0, u32 T, u32 yts, u32 rs, u32 os, u32 hrs,
    G u32 *rcount) {
#pragma clang fp contract(off)
  static LDS u32 ink_r7_mt_ver_last;
  if (rows == 0u || (rows & 3u) != 0u || nb == 0u || T == 0u || K == 0u ||
      !qw0 || !qs0 || !X || !Y || !list || !packs || !n_packs ||
      !route || !v || !ring || !res || !out || !rcount)
    return;

  const u32 by = tokid();
  const u32 npk = (u32)__builtin_amdgcn_readfirstlane((int)(*n_packs));
  const u32 rb = wgid();
  const u32 rbase = rb * 32u;
  if (by >= npk || rbase >= rows) return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)packs[by]);
  const u32 e = desc & 0xFFu;
  const u32 lo = (desc >> 8u) & 0xFFFu;
  const u32 nj = (desc >> 20u) & 0xFu;

  const u32 t0 = tid();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 r_wave = rbase + wv * 4u;

  /* Phase 1: exact ie_inkling_gexp_iq4nl_compact_r6 (all waves stay convergent for barrier). */
  if (nj > 0u && r_wave < rows) {
    const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
    const u32 r = r_wave + sub;
    const u32 rr = r < rows ? r : rows - 1u;

    const G u8 *w_base = rq_ptr_u((const G u8 *)(qw0 + (unsigned long)e * qwn));
    const G u8 *q_base = rq_ptr_u((const G u8 *)(qs0 + (unsigned long)e * qsn));
    const G u8 *rw = w_base + (unsigned long)rr * ((unsigned long)nb * 16ul);
    const G f16 *rs_p = (const G f16 *)q_base + (unsigned long)rr * nb;
    const unsigned long x_scale_off = 32ul * (unsigned long)nb;
    const int full_wave = (nb > 16u);

    const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
    const G u8 *xp0 = rq_ptr_u(X + (unsigned long)ink_r7_mt_div_x(p0, xdiv) * xs);
    const G u32 *aw0 = (const G u32 *)xp0;
    const G float *da0 = (const G float *)(xp0 + x_scale_off);

    if (nj == 1u) {
      if (r < rows) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        const u32 nstep = (nb + 7u) >> 3u;
        u32x4 w_c = (u32x4)(0u), x0_c = (u32x4)(0u), x1_c = (u32x4)(0u);
        f16 sc_c = (f16)0.0f;
        float d_c = 0.0f;
        if (sl < nb) {
          w_c = *(const G u32x4 *)(rw + sl * 16u);
          sc_c = rs_p[sl];
          x0_c = *(const G u32x4 *)(aw0 + sl * 8u);
          x1_c = *(const G u32x4 *)(aw0 + sl * 8u + 4u);
          d_c = da0[sl];
        }
        __builtin_amdgcn_sched_barrier(0);

#pragma unroll 1
        for (u32 step = 0u; step < nstep; step++) {
          const u32 b_c = step * 8u + sl;
          const u32 bn = b_c + 8u;
          u32x4 w_n = (u32x4)(0u), x0_n = (u32x4)(0u), x1_n = (u32x4)(0u);
          f16 sc_n = (f16)0.0f;
          float d_n = 0.0f;
          if (bn < nb) {
            w_n = *(const G u32x4 *)(rw + bn * 16u);
            sc_n = rs_p[bn];
            x0_n = *(const G u32x4 *)(aw0 + bn * 8u);
            x1_n = *(const G u32x4 *)(aw0 + bn * 8u + 4u);
            d_n = da0[bn];
          }
          __builtin_amdgcn_sched_barrier(0);
          if (b_c < nb) {
            const float val = (float)sc_c * (float)ink_r7_mt_iq4_dot4(w_c, x0_c, x1_c);
            const u32 j = step & 3u;
            INK_R7_MT_ACC_FMA(a0, a1, a2, a3, j, val, d_c);
          }
          __builtin_amdgcn_sched_barrier(0);
          w_c = w_n; sc_c = sc_n; x0_c = x0_n; x1_c = x1_n; d_c = d_n;
        }

        const float acc0 = ink_r7_mt_oct_tree_f(full_wave ? ((a0 + a2) + (a1 + a3)) : (a0 + a1));
        if (sl == 0u) {
          G float *yp0 = (G float *)(Y + (unsigned long)p0 * ys);
          yp0[r] = acc0;
        }
      }
    } else {
      const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 1u]);
      const u32 p2 = nj > 2u ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 2u]) : p0;
      const u32 p3 = nj > 3u ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 3u]) : p0;

      const G u8 *xp1 = rq_ptr_u(X + (unsigned long)ink_r7_mt_div_x(p1, xdiv) * xs);
      const G u8 *xp2 = nj > 2u ? rq_ptr_u(X + (unsigned long)ink_r7_mt_div_x(p2, xdiv) * xs) : xp0;
      const G u8 *xp3 = nj > 3u ? rq_ptr_u(X + (unsigned long)ink_r7_mt_div_x(p3, xdiv) * xs) : xp0;

      const G u32 *aw1 = (const G u32 *)xp1;
      const G u32 *aw2 = (const G u32 *)xp2;
      const G u32 *aw3 = (const G u32 *)xp3;
      const G float *da1 = (const G float *)(xp1 + x_scale_off);
      const G float *da2 = (const G float *)(xp2 + x_scale_off);
      const G float *da3 = (const G float *)(xp3 + x_scale_off);

      float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
      float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
      float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
      float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;

      const u32 nstep = (nb + 7u) >> 3u;
      const u32 nb_last = nb - 1u;

      u32x4 w_c = (u32x4)(0u);
      f16 sc_c = (f16)0.0f;
      if (sl < nb) {
        w_c = *(const G u32x4 *)(rw + sl * 16u);
        sc_c = rs_p[sl];
      }
      __builtin_amdgcn_sched_barrier(0);

#pragma unroll 1
      for (u32 step = 0u; step < nstep; step++) {
        const u32 b_c = step * 8u + sl;
        const u32 bn = b_c + 8u;
        const u32 j_acc = step & 3u;

        u32x4 w_n = (u32x4)(0u);
        f16 sc_n = (f16)0.0f;

        if (b_c < nb) {
          u32x4 x0_a = *(const G u32x4 *)(aw0 + b_c * 8u);
          u32x4 x1_a = *(const G u32x4 *)(aw0 + b_c * 8u + 4u);
          float d_a = da0[b_c];

          u32x4 x0_b = *(const G u32x4 *)(aw1 + b_c * 8u);
          u32x4 x1_b = *(const G u32x4 *)(aw1 + b_c * 8u + 4u);
          float d_b = da1[b_c];

          const u32 bn_safe = bn < nb ? bn : nb_last;
          w_n = *(const G u32x4 *)(rw + bn_safe * 16u);
          sc_n = rs_p[bn_safe];
          __builtin_amdgcn_sched_barrier(0);

          const float sc_f = (float)sc_c;
          const u32 wcx = w_c.x, wcy = w_c.y, wcz = w_c.z, wcw = w_c.w;
          const u32 wl0 = ie_iq4_map(wcx & 0x0F0F0F0Fu);
          const u32 wh0 = ie_iq4_map((wcx >> 4u) & 0x0F0F0F0Fu);
          const u32 wl1 = ie_iq4_map(wcy & 0x0F0F0F0Fu);
          const u32 wh1 = ie_iq4_map((wcy >> 4u) & 0x0F0F0F0Fu);
          const u32 wl2 = ie_iq4_map(wcz & 0x0F0F0F0Fu);
          const u32 wh2 = ie_iq4_map((wcz >> 4u) & 0x0F0F0F0Fu);
          const u32 wl3 = ie_iq4_map(wcw & 0x0F0F0F0Fu);
          const u32 wh3 = ie_iq4_map((wcw >> 4u) & 0x0F0F0F0Fu);

          const float v0 = sc_f * (float)ink_r7_mt_iq4_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_a, x1_a);
          INK_R7_MT_ACC_FMA(a0_0, a0_1, a0_2, a0_3, j_acc, v0, d_a);

          if (nj > 2u) {
            x0_a = *(const G u32x4 *)(aw2 + b_c * 8u);
            x1_a = *(const G u32x4 *)(aw2 + b_c * 8u + 4u);
            d_a = da2[b_c];
          }
          __builtin_amdgcn_sched_barrier(0);

          const float v1 = sc_f * (float)ink_r7_mt_iq4_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_b, x1_b);
          INK_R7_MT_ACC_FMA(a1_0, a1_1, a1_2, a1_3, j_acc, v1, d_b);

          if (nj > 2u) {
            if (nj > 3u) {
              x0_b = *(const G u32x4 *)(aw3 + b_c * 8u);
              x1_b = *(const G u32x4 *)(aw3 + b_c * 8u + 4u);
              d_b = da3[b_c];
            }
            __builtin_amdgcn_sched_barrier(0);
            const float v2 = sc_f * (float)ink_r7_mt_iq4_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_a, x1_a);
            INK_R7_MT_ACC_FMA(a2_0, a2_1, a2_2, a2_3, j_acc, v2, d_a);

            if (nj > 3u) {
              __builtin_amdgcn_sched_barrier(0);
              const float v3 = sc_f * (float)ink_r7_mt_iq4_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_b, x1_b);
              INK_R7_MT_ACC_FMA(a3_0, a3_1, a3_2, a3_3, j_acc, v3, d_b);
            }
          }
        }
        __builtin_amdgcn_sched_barrier(0);
        w_c = w_n; sc_c = sc_n;
      }

      const float acc0 = ink_r7_mt_oct_tree_f(full_wave ? ((a0_0 + a0_2) + (a0_1 + a0_3)) : (a0_0 + a0_1));
      if (sl == 0u && r < rows) ((G float *)(Y + (unsigned long)p0 * ys))[r] = acc0;

      const float acc1 = ink_r7_mt_oct_tree_f(full_wave ? ((a1_0 + a1_2) + (a1_1 + a1_3)) : (a1_0 + a1_1));
      if (sl == 0u && r < rows) ((G float *)(Y + (unsigned long)p1 * ys))[r] = acc1;

      if (nj > 2u) {
        const float acc2 = ink_r7_mt_oct_tree_f(full_wave ? ((a2_0 + a2_2) + (a2_1 + a2_3)) : (a2_0 + a2_1));
        if (sl == 0u && r < rows) ((G float *)(Y + (unsigned long)p2 * ys))[r] = acc2;
      }
      if (nj > 3u) {
        const float acc3 = ink_r7_mt_oct_tree_f(full_wave ? ((a3_0 + a3_2) + (a3_1 + a3_3)) : (a3_0 + a3_1));
        if (sl == 0u && r < rows) ((G float *)(Y + (unsigned long)p3 * ys))[r] = acc3;
      }
    }
  }

  /* Per-row-block agent-scope release/acquire: npk-th active pack WG finishes all T tokens
   * for this 32-row slice (yd is L2-hot!). */
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (t0 == 0u) {
    ink_r7_mt_ver_last =
        (__scoped_atomic_fetch_add(&rcount[rb], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == npk - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r7_mt_ver_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  /* Phase 2: 8 threads (8 x f32x4 = 32 rows) loop t=0..T-1 over exact combine + sconv + add + hist/ring. */
  const u32 n4 = rows >> 2u;
  if (t0 < 8u) {
    const u32 i4 = rb * 8u + t0;
    if (i4 < n4) {
      ink_r7_mt_tail_loop((const G u8 *)Y, (const G u8 *)route, v, ring,
                          (const G u8 *)res, (G u8 *)out, hist,
                          n4, i4, K, escale, pos0, T, yts, rts, rs, os, hrs);
    }
  }
  if (t0 == 0u) rcount[rb] = 0u;
}

#undef INK_R7_MT_ACC_FMA

#endif /* !IE_MEGA */
