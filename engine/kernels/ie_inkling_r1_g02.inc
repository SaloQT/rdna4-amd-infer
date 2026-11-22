/* ============================================================================
 * Inkling-2 Round 1 Group 2: GQA 3:1 f16-KV Decode Relative Attention
 *                            for RX 9070 XT (gfx1201, wave32, NT=256).
 *
 * Signature:
 *   KERNEL ie_inkling_rel_attn_dec_r1(const G float *q, const G kvt *kc,
 *     const G kvt *vc, const G float *R, const G float *rel_bias,
 *     const G float *sink, G float *part, G float *out, u32 pos, u32 kv_len,
 *     u32 hd, u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win,
 *     u32 nsplit, u32 ch, G u32 *count)
 *   (Exact same 19-argument signature as ie_rel_attn.)
 *
 * Grid: (n_kv * nsplit, 1, 1) with 256 threads/WG (8 waves of 32).
 *   Recommended host nsplit for GQA-3 (preserving max_sp <= 64 and ch):
 *     u32 nsplit = (np >= 128u && b->at_count) ? (np + 63u) / 64u : 1u;
 *     (Gives 128 WGs = 2 WGs/CU on SWA win=1024 and up to 512 WGs on Global;
 *      also works unchanged with base ie_rel_attn's (np + 127u) / 128u.)
 *
 * Guards (src/hip.c):
 *   T == 1u && !b->kvq8 && !b->kvq4 && hd == 128u && d_rel == 16u &&
 *   n_kv > 0u && n_head == 3u * n_kv
 *
 * Split-K compatibility:
 *   Uses identical `part` layout: part[(h * nsplit + s) * (hd + 2)] = {M, L, O[128]}
 *   and `count[kh]` (kh < n_kv <= n_head), resetting count[kh] = 0 before exit.
 *
 * Parity expectation:
 *   f32-tol vs ie_rel_attn (contiguous f16x4 lane reduction order vs strided
 *   scalar f16; identical relative bias, sink, online softmax, and split-K math).
 * ========================================================================== */

#define INK_R1_G02_HD 128u
#define INK_R1_G02_DREL 16u
#define INK_R1_G02_GQA 3u
#define INK_R1_G02_PSTR 130u

/* 16-lane (half-wave) sum broadcast to all 32 lanes; bitwise identical to
 * wave_sum_all(v) when lanes 16..31 are 0.0f, saving 1x v_permlanex16. */
static inline __attribute__((always_inline)) float ink_r1_g02_halfwave_sum_all(float v) {
  v += I2F(shl_i(F2I(v), 8));
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return I2F(__builtin_amdgcn_readlane(F2I(v), 0));
}

static inline __attribute__((always_inline)) float ink_r1_g02_dot4(f32x4 a, f32x4 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

KERNEL_CALL ie_inkling_rel_attn_dec_r1(const G float *q, const G kvt *kc, const G kvt *vc, const G float *R,
                                       const G float *rel_bias, const G float *sink, G float *part, G float *out, u32 pos,
                                       u32 kv_len, u32 hd, u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win,
                                       u32 nsplit, u32 ch, G u32 *count) {
#ifdef IE_MEGA
#define ink_r1_g02_m (mk_ink_lds.rel_attn.m)
#define ink_r1_g02_l (mk_ink_lds.rel_attn.l)
#define ink_r1_g02_o (mk_ink_lds.rel_attn.o)
#define ink_r1_g02_last (mk_ink_lds.rel_attn.last)
#else
  static LDS float ink_r1_g02_m[INK_R1_G02_GQA][8];
  static LDS float ink_r1_g02_l[INK_R1_G02_GQA][8];
  static LDS __attribute__((aligned(16))) float ink_r1_g02_o[INK_R1_G02_GQA][8][INK_R1_G02_HD];
  static LDS u32 ink_r1_g02_last;
#endif

  if (hd != INK_R1_G02_HD || d_rel != INK_R1_G02_DREL || n_head != INK_R1_G02_GQA * n_kv) return;

  const u32 kh = wgid() / nsplit;
  const u32 s = wgid() % nsplit;
  const u32 h0 = kh * INK_R1_G02_GQA;
  const u32 kvd = n_kv * INK_R1_G02_HD;
  const u32 w = tid() >> 5u;
  const u32 l = lane();
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const float scale = 1.0f / __builtin_sqrtf((float)INK_R1_G02_HD);

  /* Load Q (3 heads x 128 f32 -> 1 f32x4/lane/head) and R (3 heads x 16 f32). */
  const f32x4 q0 = ((const G f32x4 *)(q + (unsigned long)(h0 + 0u) * INK_R1_G02_HD))[l];
  const f32x4 q1 = ((const G f32x4 *)(q + (unsigned long)(h0 + 1u) * INK_R1_G02_HD))[l];
  const f32x4 q2 = ((const G f32x4 *)(q + (unsigned long)(h0 + 2u) * INK_R1_G02_HD))[l];

  const float rv0 = (l < INK_R1_G02_DREL) ? R[(unsigned long)(h0 + 0u) * INK_R1_G02_DREL + l] : 0.0f;
  const float rv1 = (l < INK_R1_G02_DREL) ? R[(unsigned long)(h0 + 1u) * INK_R1_G02_DREL + l] : 0.0f;
  const float rv2 = (l < INK_R1_G02_DREL) ? R[(unsigned long)(h0 + 2u) * INK_R1_G02_DREL + l] : 0.0f;

  f32x4 acc0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 acc1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 acc2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};

  float m0 = -__builtin_inff(), sum0 = 0.0f;
  float m1 = -__builtin_inff(), sum1 = 0.0f;
  float m2 = -__builtin_inff(), sum2 = 0.0f;

  const u32 w0 = (win > 0u && pos + 1u > win) ? (pos + 1u - win) : 0u;
  const u32 p0 = w0 + s * ch;
  const u32 pend = pos + 1u;
  const u32 p1 = (w0 + (s + 1u) * ch < pend) ? (w0 + (s + 1u) * ch) : pend;

  const G kvt *kc_h = kc + (unsigned long)kh * INK_R1_G02_HD;
  const G kvt *vc_h = vc + (unsigned long)kh * INK_R1_G02_HD;

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
      if (has_rel && l < INK_R1_G02_DREL) rb = rel_bias[(unsigned long)dist * INK_R1_G02_DREL + l];

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
      float d0 = wave_sum_all(ink_r1_g02_dot4(q0, kf)) * scale;
      float d1 = wave_sum_all(ink_r1_g02_dot4(q1, kf)) * scale;
      float d2 = wave_sum_all(ink_r1_g02_dot4(q2, kf)) * scale;

      if (has_rel) {
        d0 += ink_r1_g02_halfwave_sum_all(rv0 * rb);
        d1 += ink_r1_g02_halfwave_sum_all(rv1 * rb);
        d2 += ink_r1_g02_halfwave_sum_all(rv2 * rb);
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
    ink_r1_g02_m[0][w] = m0;
    ink_r1_g02_l[0][w] = sum0;
    ink_r1_g02_m[1][w] = m1;
    ink_r1_g02_l[1][w] = sum1;
    ink_r1_g02_m[2][w] = m2;
    ink_r1_g02_l[2][w] = sum2;
  }
  ((LDSP f32x4 *)&ink_r1_g02_o[0][w][0])[l] = acc0;
  ((LDSP f32x4 *)&ink_r1_g02_o[1][w][0])[l] = acc1;
  ((LDSP f32x4 *)&ink_r1_g02_o[2][w][0])[l] = acc2;
  lbarrier();

  /* Fast path: nsplit == 1 (no global scratch or atomics). */
  if (nsplit == 1u) {
    for (u32 idx = tid(); idx < INK_R1_G02_GQA * INK_R1_G02_HD; idx += NT) {
      const u32 g = idx >> 7u;
      const u32 e = idx & 127u;
      const u32 h = h0 + g;
      const float sv = sink[h];
      float M = sv;
      for (u32 x = 0u; x < 8u; x++)
        if (ink_r1_g02_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r1_g02_m[g][x]);
      float L = __builtin_expf(sv - M);
      float o = 0.0f;
      for (u32 x = 0u; x < 8u; x++) {
        const float lx = ink_r1_g02_l[g][x];
        if (lx > 0.0f) {
          const float wx = __builtin_expf(ink_r1_g02_m[g][x] - M);
          L += lx * wx;
          o += ink_r1_g02_o[g][x][e] * wx;
        }
      }
      out[(unsigned long)h * INK_R1_G02_HD + e] = o / L;
    }
    return;
  }

  /* Split-K path: write 3 partial records into part[(h * nsplit + s) * 130]. */
  for (u32 idx = tid(); idx < INK_R1_G02_GQA * INK_R1_G02_HD; idx += NT) {
    const u32 g = idx >> 7u;
    const u32 e = idx & 127u;
    const u32 h = h0 + g;
    float M = -__builtin_inff();
    for (u32 x = 0u; x < 8u; x++)
      if (ink_r1_g02_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r1_g02_m[g][x]);
    G float *pp = part + (unsigned long)(h * nsplit + s) * INK_R1_G02_PSTR;
    float o = 0.0f;
    float L = 0.0f;
    for (u32 x = 0u; x < 8u; x++) {
      const float lx = ink_r1_g02_l[g][x];
      if (lx > 0.0f) {
        const float wx = __builtin_expf(ink_r1_g02_m[g][x] - M);
        o += ink_r1_g02_o[g][x][e] * wx;
        if (e == 0u) L += lx * wx;
      }
    }
    pp[2u + e] = o;
    if (e == 0u) {
      pp[0] = M;
      pp[1] = L;
    }
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r1_g02_last =
        (__scoped_atomic_fetch_add(&count[kh], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u);
  }
  barrier();
  if (!ink_r1_g02_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  /* Last WG for KV head kh merges all 3 Q heads across s = 0 .. nsplit - 1. */
  G float *ph0 = part + (unsigned long)(h0 + 0u) * nsplit * INK_R1_G02_PSTR;
  G float *ph1 = part + (unsigned long)(h0 + 1u) * nsplit * INK_R1_G02_PSTR;
  G float *ph2 = part + (unsigned long)(h0 + 2u) * nsplit * INK_R1_G02_PSTR;

  /* Pass 1: global max M2[0..2] including per-head sink. */
  float mv0 = (tid() == 0u) ? sink[h0 + 0u] : -__builtin_inff();
  float mv1 = (tid() == 0u) ? sink[h0 + 1u] : -__builtin_inff();
  float mv2 = (tid() == 0u) ? sink[h0 + 2u] : -__builtin_inff();
  for (u32 t = tid(); t < nsplit; t += NT) {
    const unsigned long off = (unsigned long)t * INK_R1_G02_PSTR;
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
    ink_r1_g02_m[0][w] = mv0;
    ink_r1_g02_m[1][w] = mv1;
    ink_r1_g02_m[2][w] = mv2;
  }
  lbarrier();

  float M2[INK_R1_G02_GQA];
  for (u32 g = 0u; g < INK_R1_G02_GQA; g++) {
    float mx = ink_r1_g02_m[g][0];
    for (u32 x = 1u; x < 8u; x++) mx = __builtin_fmaxf(mx, ink_r1_g02_m[g][x]);
    M2[g] = mx;
  }

  /* Pass 2: per-split weights + global denominator L2[0..2].
   * For nsplit <= 128 (always true when max_sp <= 64), stage weights in dead
   * LDS ink_r1_g02_o[g][0][t] to avoid global round-trips through part. */
  const int wt_in_lds = (nsplit <= INK_R1_G02_HD);
  float lv0 = 0.0f, lv1 = 0.0f, lv2 = 0.0f;
  for (u32 t = tid(); t < nsplit; t += NT) {
    const unsigned long off = (unsigned long)t * INK_R1_G02_PSTR;

    const float lp0 = ph0[off + 1u];
    const float wt0 = (lp0 > 0.0f) ? __builtin_expf(ph0[off] - M2[0]) : 0.0f;
    if (wt_in_lds) ink_r1_g02_o[0][0][t] = wt0;
    else ph0[off] = wt0;
    lv0 += lp0 * wt0;

    const float lp1 = ph1[off + 1u];
    const float wt1 = (lp1 > 0.0f) ? __builtin_expf(ph1[off] - M2[1]) : 0.0f;
    if (wt_in_lds) ink_r1_g02_o[1][0][t] = wt1;
    else ph1[off] = wt1;
    lv1 += lp1 * wt1;

    const float lp2 = ph2[off + 1u];
    const float wt2 = (lp2 > 0.0f) ? __builtin_expf(ph2[off] - M2[2]) : 0.0f;
    if (wt_in_lds) ink_r1_g02_o[2][0][t] = wt2;
    else ph2[off] = wt2;
    lv2 += lp2 * wt2;
  }
  lv0 = wave_tree_f(lv0);
  lv1 = wave_tree_f(lv1);
  lv2 = wave_tree_f(lv2);
  if (l == 0u) {
    ink_r1_g02_l[0][w] = lv0;
    ink_r1_g02_l[1][w] = lv1;
    ink_r1_g02_l[2][w] = lv2;
  }
  if (wt_in_lds) lbarrier();
  else barrier();

  float L2[INK_R1_G02_GQA];
  for (u32 g = 0u; g < INK_R1_G02_GQA; g++) {
    float ls = 0.0f;
    for (u32 x = 0u; x < 8u; x++) ls += ink_r1_g02_l[g][x];
    L2[g] = ls + __builtin_expf(sink[h0 + g] - M2[g]);
  }

  /* Pass 3: weighted sum of partial outputs -> out[h * 128 + e]. */
  for (u32 idx = tid(); idx < INK_R1_G02_GQA * INK_R1_G02_HD; idx += NT) {
    const u32 g = idx >> 7u;
    const u32 e = idx & 127u;
    const u32 h = h0 + g;
    const G float *ph = part + (unsigned long)h * nsplit * INK_R1_G02_PSTR;
    float o = 0.0f;
    if (wt_in_lds) {
      for (u32 t = 0u; t < nsplit; t++) o += ph[(unsigned long)t * INK_R1_G02_PSTR + 2u + e] * ink_r1_g02_o[g][0][t];
    } else {
      for (u32 t = 0u; t < nsplit; t++) o += ph[(unsigned long)t * INK_R1_G02_PSTR + 2u + e] * ph[(unsigned long)t * INK_R1_G02_PSTR];
    }
    out[(unsigned long)h * INK_R1_G02_HD + e] = o / L2[g];
  }

  if (tid() == 0u) count[kh] = 0u;
}

#ifdef IE_MEGA
#undef ink_r1_g02_m
#undef ink_r1_g02_l
#undef ink_r1_g02_o
#undef ink_r1_g02_last
#endif
#undef INK_R1_G02_HD
#undef INK_R1_G02_DREL
#undef INK_R1_G02_GQA
#undef INK_R1_G02_PSTR
