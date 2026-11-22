/* ============================================================================
 * Inkling-2 Round 4: Verify Split-K Relative Attention & sconv Ring History
 *                    for RX 9070 XT (gfx1201, wave32, NT=256, 0-spill).
 *
 * Kernels:
 *   1. [Bit 42] ie_inkling_rel_attn_ver_r4
 *      Signature:
 *        KERNEL ie_inkling_rel_attn_ver_r4(
 *          const G float *q, const G kvt *kc, const G kvt *vc,
 *          const G float *R, const G float *rel_bias, const G float *sink,
 *          G float *part, G float *out,
 *          u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
 *          u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
 *          u32 qs, u32 rs, u32 ss, u32 os, G u32 *count)
 *      Grid: (n_kv * nsplit, T, 1u) with NT=256 (8 waves of 32).
 *        1 WG processes 3 GQA Q heads (gqa=3) for 1 KV head kh = wgid()/nsplit,
 *        1 token t = tokid() (at pos = pos0 + t), and split s = wgid()%nsplit.
 *      Guards (src/hip.c):
 *        !b->kvq8 && !b->kvq4 && T >= 2u && T <= 16u && hd == 128u &&
 *        d_rel == 16u && n_kv > 0u && n_head == 3u * n_kv && kv_len > 0u &&
 *        nsplit >= 1u && (nsplit == 1u || (part && count)) &&
 *        ((qs | os) & 15u) == 0u && ((uintptr_t)q & 15u) == 0u
 *      Buffer requirements (when nsplit > 1u; unused when nsplit == 1u):
 *        - part:  >= T * n_head * nsplit * 130 floats (520 bytes/slot).
 *                 Slot index: ((t * n_head + h) * nsplit + s) * 130
 *                 Record layout: {M, L, O[0..127]}.
 *        - count: >= T * n_kv u32 elements, zero-initialized before first use.
 *                 Counter index: t * n_kv + kh (self-reset to 0 by last WG).
 *      Parity:
 *        f32-tol vs ie_rel_attn_pf / ie_rel_attn (contiguous f16x4 lane
 *        reduction order vs strided scalar f16; exact relative bias, sink,
 *        online softmax, causal bound pos0+t, SWA win, and ring % kv_len).
 *
 *   2. [Bit 43] ie_inkling_sconv_hist_r4
 *      Signature:
 *        KERNEL ie_inkling_sconv_hist_r4(
 *          const G float *a, G float *hist,
 *          u32 n, u32 T, u32 as, u32 hist_row_stride)
 *      Grid: (((n >> 2u) + 255u) / 256u, T, 1u) with NT=256.
 *      Guards (src/hip.c):
 *        (n & 3u) == 0u && (as & 15u) == 0u && (hist_row_stride & 3u) == 0u &&
 *        ((uintptr_t)a & 15u) == 0u && ((uintptr_t)hist & 15u) == 0u
 *      Behavior:
 *        Copies a[t * (as/4) + c] -> hist[t * hist_row_stride + c] for
 *        t = 0..T-1, c = 0..n-1 using coalesced f32x4 loads/stores, leaving
 *        sc_ring untouched during speculative verify.
 *      Parity: bitwise-exact copy.
 *
 *   3. [Bit 43] ie_inkling_sconv_commit_r4
 *      Signature:
 *        KERNEL ie_inkling_sconv_commit_r4(
 *          G float *ring, const G float *hist,
 *          u32 tot, u32 pos0, u32 n_acc, u32 hist_row_stride)
 *      Grid: ((tot + 255u) / 256u, 1u, 1u) with NT=256.
 *      Guards (src/hip.c):
 *        tot > 0u && n_acc >= 1u && ((uintptr_t)ring & 15u) == 0u
 *      Behavior:
 *        Commits accepted tokens t = (n_acc > 4u ? n_acc - 4u : 0u) .. n_acc-1
 *        into ring[c * 4u + ((pos0 + t) & 3u)] = hist[t * hist_row_stride + c]
 *        for all c in 0..tot-1 (coalesced f32x4 store when n_acc >= 4u).
 *      Parity: bitwise-exact vs running ie_sconv_ring_store on accepted prefix.
 * ========================================================================== */

#define INK_R4_ATTN_HD 128u
#define INK_R4_ATTN_DREL 16u
#define INK_R4_ATTN_GQA 3u
#define INK_R4_ATTN_PSTR 130u

/* 16-lane (half-wave) sum broadcast to all 32 lanes; bitwise identical to
 * wave_sum_all(v) when lanes 16..31 are 0.0f, saving 1x v_permlanex16. */
static inline __attribute__((always_inline)) float ink_r4_attn_halfwave_sum_all(float v) {
  v += I2F(shl_i(F2I(v), 8));
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return I2F(__builtin_amdgcn_readlane(F2I(v), 0));
}

static inline __attribute__((always_inline)) float ink_r4_attn_dot4(f32x4 a, f32x4 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

/* ----------------------------------------------------------------------------
 * [Bit 42] Kernel 1: Multi-token (T=2..16) split-K GQA 3:1 f16-KV relative attn
 * -------------------------------------------------------------------------- */

KERNEL_CALL ie_inkling_rel_attn_ver_r4(const G float *q, const G kvt *kc, const G kvt *vc, const G float *R,
                                       const G float *rel_bias, const G float *sink, G float *part, G float *out,
                                       u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv, u32 d_rel,
                                       u32 rel_extent, u32 win, u32 nsplit, u32 qs, u32 rs, u32 ss, u32 os,
                                       G u32 *count) {
#ifdef IE_MEGA
#define ink_r4_attn_m (mk_ink_lds.rel_attn.m)
#define ink_r4_attn_l (mk_ink_lds.rel_attn.l)
#define ink_r4_attn_o (mk_ink_lds.rel_attn.o)
#define ink_r4_attn_last (mk_ink_lds.rel_attn.last)
#else
  static LDS float ink_r4_attn_m[INK_R4_ATTN_GQA][8];
  static LDS float ink_r4_attn_l[INK_R4_ATTN_GQA][8];
  static LDS __attribute__((aligned(16))) float ink_r4_attn_o[INK_R4_ATTN_GQA][8][INK_R4_ATTN_HD];
  static LDS u32 ink_r4_attn_last;
#endif

  const u32 t = tokid();
  if (t >= T || hd != INK_R4_ATTN_HD || d_rel != INK_R4_ATTN_DREL || n_kv == 0u ||
      n_head != INK_R4_ATTN_GQA * n_kv || kv_len == 0u || nsplit == 0u)
    return;

  const u32 kh = wgid() / nsplit;
  const u32 s = wgid() % nsplit;
  const u32 h0 = kh * INK_R4_ATTN_GQA;
  const u32 kvd = n_kv * INK_R4_ATTN_HD;
  const u32 w = tid() >> 5u;
  const u32 l = lane();

  const u32 pos = pos0 + t;
  const u32 pend = pos + 1u;
  const u32 w0 = (win > 0u && pend > win) ? (pend - win) : 0u;
  const u32 np = pend - w0;
  const u32 ch = (np + nsplit - 1u) / nsplit;
  const u32 off0 = s * ch;
  const u32 p0 = (off0 < np) ? (w0 + off0) : pend;
  const u32 p1 = (off0 < np && off0 + ch < np) ? (w0 + off0 + ch) : pend;

  const G float *qt = (const G float *)((const G u8 *)q + (unsigned long)t * qs);
  const G float *Rt = (const G float *)((const G u8 *)R + (unsigned long)t * rs);
  const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
  G float *ot = (G float *)((G u8 *)out + (unsigned long)t * os);

  if (nsplit > 1u && p0 >= p1) {
    /* Empty tail split (s * ch >= np): write sentinel partial record so
     * split-K merge skips this split while still advancing count[t * n_kv + kh]. */
    for (u32 idx = tid(); idx < INK_R4_ATTN_GQA * INK_R4_ATTN_HD; idx += NT) {
      const u32 g = idx >> 7u;
      const u32 e = idx & 127u;
      const u32 h = h0 + g;
      G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R4_ATTN_PSTR;
      pp[2u + e] = 0.0f;
      if (e == 0u) {
        pp[0] = -1e30f;
        pp[1] = 0.0f;
      }
    }
  } else {
    const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
    const float scale = 1.0f / __builtin_sqrtf((float)INK_R4_ATTN_HD);

    /* Load Q (3 heads x 128 f32 -> 1 f32x4/lane/head) and R (3 heads x 16 f32). */
    const f32x4 q0 = ((const G f32x4 *)(qt + (unsigned long)(h0 + 0u) * INK_R4_ATTN_HD))[l];
    const f32x4 q1 = ((const G f32x4 *)(qt + (unsigned long)(h0 + 1u) * INK_R4_ATTN_HD))[l];
    const f32x4 q2 = ((const G f32x4 *)(qt + (unsigned long)(h0 + 2u) * INK_R4_ATTN_HD))[l];

    const float rv0 = (l < INK_R4_ATTN_DREL) ? Rt[(unsigned long)(h0 + 0u) * INK_R4_ATTN_DREL + l] : 0.0f;
    const float rv1 = (l < INK_R4_ATTN_DREL) ? Rt[(unsigned long)(h0 + 1u) * INK_R4_ATTN_DREL + l] : 0.0f;
    const float rv2 = (l < INK_R4_ATTN_DREL) ? Rt[(unsigned long)(h0 + 2u) * INK_R4_ATTN_DREL + l] : 0.0f;

    f32x4 acc0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};

    float m0 = -__builtin_inff(), sum0 = 0.0f;
    float m1 = -__builtin_inff(), sum1 = 0.0f;
    float m2 = -__builtin_inff(), sum2 = 0.0f;

    const G kvt *kc_h = kc + (unsigned long)kh * INK_R4_ATTN_HD;
    const G kvt *vc_h = vc + (unsigned long)kh * INK_R4_ATTN_HD;

    /* Software-pipelined GQA-3 KV loop: 1x coalesced f16x4 K load + 1x V load
     * per wave shared across all 3 Q heads, with 1-step KV lookahead. */
    u32 kp = p0 + w;
    if (kp < p1) {
      u32 kvp = kp % kv_len;
      const G kv4 *krow = (const G kv4 *)(kc_h + (unsigned long)kvp * kvd);
      const G kv4 *vrow = (const G kv4 *)(vc_h + (unsigned long)kvp * kvd);
      kv4 kh4 = krow[l];
      kv4 vh4 = vrow[l];

      for (;;) {
        const u32 dist = pos - kp;
        const int has_rel = (dist < extent_lim);
        float rb = 0.0f;
        if (has_rel && l < INK_R4_ATTN_DREL) rb = rel_bias[(unsigned long)dist * INK_R4_ATTN_DREL + l];

        const u32 kp_nxt = kp + 8u;
        u32 kvp_nxt = kvp;
        kv4 kh4_nxt = kh4;
        kv4 vh4_nxt = vh4;
        if (kp_nxt < p1) {
          kvp_nxt = kvp + 8u;
          if (kvp_nxt >= kv_len) kvp_nxt = (kv_len >= 8u) ? (kvp_nxt - kv_len) : (kp_nxt % kv_len);
          const G kv4 *krow_n = (const G kv4 *)(kc_h + (unsigned long)kvp_nxt * kvd);
          const G kv4 *vrow_n = (const G kv4 *)(vc_h + (unsigned long)kvp_nxt * kvd);
          kh4_nxt = krow_n[l];
          vh4_nxt = vrow_n[l];
        }

        const f32x4 kf = __builtin_convertvector(kh4, f32x4);
        float d0 = wave_sum_all(ink_r4_attn_dot4(q0, kf)) * scale;
        float d1 = wave_sum_all(ink_r4_attn_dot4(q1, kf)) * scale;
        float d2 = wave_sum_all(ink_r4_attn_dot4(q2, kf)) * scale;

        if (has_rel) {
          d0 += ink_r4_attn_halfwave_sum_all(rv0 * rb);
          d1 += ink_r4_attn_halfwave_sum_all(rv1 * rb);
          d2 += ink_r4_attn_halfwave_sum_all(rv2 * rb);
        }

        const float mn0 = __builtin_fmaxf(m0, d0);
        const float c0 = (sum0 > 0.0f) ? __builtin_expf(m0 - mn0) : 0.0f;
        const float e0 = __builtin_expf(d0 - mn0);
        sum0 = sum0 * c0 + e0;
        m0 = mn0;

        const float mn1 = __builtin_fmaxf(m1, d1);
        const float c1 = (sum1 > 0.0f) ? __builtin_expf(m1 - mn1) : 0.0f;
        const float e1 = __builtin_expf(d1 - mn1);
        sum1 = sum1 * c1 + e1;
        m1 = mn1;

        const float mn2 = __builtin_fmaxf(m2, d2);
        const float c2 = (sum2 > 0.0f) ? __builtin_expf(m2 - mn2) : 0.0f;
        const float e2 = __builtin_expf(d2 - mn2);
        sum2 = sum2 * c2 + e2;
        m2 = mn2;

        const f32x4 vf = __builtin_convertvector(vh4, f32x4);
        acc0 = acc0 * c0 + e0 * vf;
        acc1 = acc1 * c1 + e1 * vf;
        acc2 = acc2 * c2 + e2 * vf;

        if (kp_nxt >= p1) break;
        kp = kp_nxt;
        kvp = kvp_nxt;
        kh4 = kh4_nxt;
        vh4 = vh4_nxt;
      }
    }

    /* Stage wave-level (m, sum, acc[128]) for all 3 Q heads into LDS. */
    if (l == 0u) {
      ink_r4_attn_m[0][w] = m0;
      ink_r4_attn_l[0][w] = sum0;
      ink_r4_attn_m[1][w] = m1;
      ink_r4_attn_l[1][w] = sum1;
      ink_r4_attn_m[2][w] = m2;
      ink_r4_attn_l[2][w] = sum2;
    }
    ((LDSP f32x4 *)&ink_r4_attn_o[0][w][0])[l] = acc0;
    ((LDSP f32x4 *)&ink_r4_attn_o[1][w][0])[l] = acc1;
    ((LDSP f32x4 *)&ink_r4_attn_o[2][w][0])[l] = acc2;
    lbarrier();

    /* Fast path: nsplit == 1 (no global scratch or atomics). */
    if (nsplit == 1u) {
      for (u32 idx = tid(); idx < INK_R4_ATTN_GQA * INK_R4_ATTN_HD; idx += NT) {
        const u32 g = idx >> 7u;
        const u32 e = idx & 127u;
        const u32 h = h0 + g;
        const float sv = st[h];
        float M = sv;
        for (u32 x = 0u; x < 8u; x++)
          if (ink_r4_attn_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r4_attn_m[g][x]);
        float L = __builtin_expf(sv - M);
        float o = 0.0f;
        for (u32 x = 0u; x < 8u; x++) {
          const float lx = ink_r4_attn_l[g][x];
          if (lx > 0.0f) {
            const float wx = __builtin_expf(ink_r4_attn_m[g][x] - M);
            L += lx * wx;
            o += ink_r4_attn_o[g][x][e] * wx;
          }
        }
        ot[(unsigned long)h * INK_R4_ATTN_HD + e] = o / L;
      }
      return;
    }

    /* Split-K path: write 3 partial records into part[((t*n_head + h)*nsplit + s)*130]. */
    for (u32 idx = tid(); idx < INK_R4_ATTN_GQA * INK_R4_ATTN_HD; idx += NT) {
      const u32 g = idx >> 7u;
      const u32 e = idx & 127u;
      const u32 h = h0 + g;
      float M = -__builtin_inff();
      for (u32 x = 0u; x < 8u; x++)
        if (ink_r4_attn_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r4_attn_m[g][x]);
      G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R4_ATTN_PSTR;
      float o = 0.0f;
      float L = 0.0f;
      for (u32 x = 0u; x < 8u; x++) {
        const float lx = ink_r4_attn_l[g][x];
        if (lx > 0.0f) {
          const float wx = __builtin_expf(ink_r4_attn_m[g][x] - M);
          o += ink_r4_attn_o[g][x][e] * wx;
          if (e == 0u) L += lx * wx;
        }
      }
      pp[2u + e] = o;
      if (e == 0u) {
        pp[0] = M;
        pp[1] = L;
      }
    }
  }

  const u32 cidx = t * n_kv + kh;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r4_attn_last =
        (__scoped_atomic_fetch_add(&count[cidx], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u);
  }
  barrier();
  if (!ink_r4_attn_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  /* Last WG for (token t, KV head kh) merges all 3 Q heads across sp = 0..nsplit-1. */
  G float *ph0 = part + (unsigned long)((t * n_head + (h0 + 0u)) * nsplit) * INK_R4_ATTN_PSTR;
  G float *ph1 = part + (unsigned long)((t * n_head + (h0 + 1u)) * nsplit) * INK_R4_ATTN_PSTR;
  G float *ph2 = part + (unsigned long)((t * n_head + (h0 + 2u)) * nsplit) * INK_R4_ATTN_PSTR;

  /* Pass 1: global max M2[0..2] including per-token, per-head sink. */
  float mv0 = (tid() == 0u) ? st[h0 + 0u] : -__builtin_inff();
  float mv1 = (tid() == 0u) ? st[h0 + 1u] : -__builtin_inff();
  float mv2 = (tid() == 0u) ? st[h0 + 2u] : -__builtin_inff();
  for (u32 sp = tid(); sp < nsplit; sp += NT) {
    const unsigned long off = (unsigned long)sp * INK_R4_ATTN_PSTR;
    if (ph0[off + 1u] > 0.0f) mv0 = __builtin_fmaxf(mv0, ph0[off]);
    if (ph1[off + 1u] > 0.0f) mv1 = __builtin_fmaxf(mv1, ph1[off]);
    if (ph2[off + 1u] > 0.0f) mv2 = __builtin_fmaxf(mv2, ph2[off]);
  }
  for (u32 mask = 16u; mask >= 1u; mask >>= 1u) {
    mv0 = __builtin_fmaxf(mv0, shfl_xor_f(mv0, mask));
    mv1 = __builtin_fmaxf(mv1, shfl_xor_f(mv1, mask));
    mv2 = __builtin_fmaxf(mv2, shfl_xor_f(mv2, mask));
  }
  if (l == 0u) {
    ink_r4_attn_m[0][w] = mv0;
    ink_r4_attn_m[1][w] = mv1;
    ink_r4_attn_m[2][w] = mv2;
  }
  lbarrier();

  float M2[INK_R4_ATTN_GQA];
  for (u32 g = 0u; g < INK_R4_ATTN_GQA; g++) {
    float mx = ink_r4_attn_m[g][0];
    for (u32 x = 1u; x < 8u; x++) mx = __builtin_fmaxf(mx, ink_r4_attn_m[g][x]);
    M2[g] = mx;
  }

  /* Pass 2: per-split weights + global denominator L2[0..2].
   * For nsplit <= 128, stage weights in dead LDS ink_r4_attn_o[g][0][sp]
   * to avoid global round-trips through part. */
  const int wt_in_lds = (nsplit <= INK_R4_ATTN_HD);
  float lv0 = 0.0f, lv1 = 0.0f, lv2 = 0.0f;
  for (u32 sp = tid(); sp < nsplit; sp += NT) {
    const unsigned long off = (unsigned long)sp * INK_R4_ATTN_PSTR;

    const float lp0 = ph0[off + 1u];
    const float wt0 = (lp0 > 0.0f) ? __builtin_expf(ph0[off] - M2[0]) : 0.0f;
    if (wt_in_lds) ink_r4_attn_o[0][0][sp] = wt0;
    else ph0[off] = wt0;
    lv0 += lp0 * wt0;

    const float lp1 = ph1[off + 1u];
    const float wt1 = (lp1 > 0.0f) ? __builtin_expf(ph1[off] - M2[1]) : 0.0f;
    if (wt_in_lds) ink_r4_attn_o[1][0][sp] = wt1;
    else ph1[off] = wt1;
    lv1 += lp1 * wt1;

    const float lp2 = ph2[off + 1u];
    const float wt2 = (lp2 > 0.0f) ? __builtin_expf(ph2[off] - M2[2]) : 0.0f;
    if (wt_in_lds) ink_r4_attn_o[2][0][sp] = wt2;
    else ph2[off] = wt2;
    lv2 += lp2 * wt2;
  }
  lv0 = wave_tree_f(lv0);
  lv1 = wave_tree_f(lv1);
  lv2 = wave_tree_f(lv2);
  if (l == 0u) {
    ink_r4_attn_l[0][w] = lv0;
    ink_r4_attn_l[1][w] = lv1;
    ink_r4_attn_l[2][w] = lv2;
  }
  if (wt_in_lds) lbarrier();
  else barrier();

  float L2[INK_R4_ATTN_GQA];
  for (u32 g = 0u; g < INK_R4_ATTN_GQA; g++) {
    float ls = 0.0f;
    for (u32 x = 0u; x < 8u; x++) ls += ink_r4_attn_l[g][x];
    L2[g] = ls + __builtin_expf(st[h0 + g] - M2[g]);
  }

  /* Pass 3: weighted sum of partial outputs -> ot[h * 128 + e]. */
  for (u32 idx = tid(); idx < INK_R4_ATTN_GQA * INK_R4_ATTN_HD; idx += NT) {
    const u32 g = idx >> 7u;
    const u32 e = idx & 127u;
    const u32 h = h0 + g;
    const G float *ph = part + (unsigned long)((t * n_head + h) * nsplit) * INK_R4_ATTN_PSTR;
    float o = 0.0f;
    if (wt_in_lds) {
      for (u32 sp = 0u; sp < nsplit; sp++) o += ph[(unsigned long)sp * INK_R4_ATTN_PSTR + 2u + e] * ink_r4_attn_o[g][0][sp];
    } else {
      for (u32 sp = 0u; sp < nsplit; sp++) o += ph[(unsigned long)sp * INK_R4_ATTN_PSTR + 2u + e] * ph[(unsigned long)sp * INK_R4_ATTN_PSTR];
    }
    ot[(unsigned long)h * INK_R4_ATTN_HD + e] = o / L2[g];
  }

  if (tid() == 0u) count[cidx] = 0u;
}

/* ----------------------------------------------------------------------------
 * [Bit 43] Kernel 2: Speculative verify sconv ring history capture
 * -------------------------------------------------------------------------- */

KERNEL ie_inkling_sconv_hist_r4(const G float *a, G float *hist, u32 n, u32 T, u32 as, u32 hist_row_stride) {
  const u32 n4 = n >> 2u;
  const u32 i4 = wgid() * NT + tid();
  const u32 t = tokid();
  if (i4 >= n4 || t >= T) return;
  const G f32x4 *at = (const G f32x4 *)((const G u8 *)a + (unsigned long)t * as);
  G f32x4 *ht = (G f32x4 *)(hist + (unsigned long)t * hist_row_stride);
  ht[i4] = at[i4];
}

/* ----------------------------------------------------------------------------
 * [Bit 43] Kernel 3: Speculative accept sconv ring commit
 * -------------------------------------------------------------------------- */

KERNEL ie_inkling_sconv_commit_r4(G float *ring, const G float *hist, u32 tot, u32 pos0, u32 n_acc,
                                  u32 hist_row_stride) {
  const u32 c = wgid() * NT + tid();
  if (c >= tot || n_acc == 0u) return;

  if (n_acc >= 4u) {
    /* Last 4 accepted tokens fill all 4 ring slots: 1x coalesced f32x4 store. */
    const u32 t0 = n_acc - 4u;
    const u32 s0 = (pos0 + t0) & 3u;
    const u32 r0 = t0 + ((0u - s0) & 3u);
    const u32 r1 = t0 + ((1u - s0) & 3u);
    const u32 r2 = t0 + ((2u - s0) & 3u);
    const u32 r3 = t0 + ((3u - s0) & 3u);
    const float v0 = hist[(unsigned long)r0 * hist_row_stride + c];
    const float v1 = hist[(unsigned long)r1 * hist_row_stride + c];
    const float v2 = hist[(unsigned long)r2 * hist_row_stride + c];
    const float v3 = hist[(unsigned long)r3 * hist_row_stride + c];
    ((G f32x4 *)ring)[c] = (f32x4){v0, v1, v2, v3};
    return;
  }

  for (u32 t = 0u; t < n_acc; t++) {
    ring[(unsigned long)c * 4u + ((pos0 + t) & 3u)] = hist[(unsigned long)t * hist_row_stride + c];
  }
}

#ifdef IE_MEGA
#undef ink_r4_attn_m
#undef ink_r4_attn_l
#undef ink_r4_attn_o
#undef ink_r4_attn_last
#endif
#undef INK_R4_ATTN_HD
#undef INK_R4_ATTN_DREL
#undef INK_R4_ATTN_GQA
#undef INK_R4_ATTN_PSTR
