/* ie_raw_swiglu.inc -- fused dual-matrix raw IQ/K gate + up GEMV + SwiGLU + Q8
 * quantization, supporting any (type0, type1) pair of the 8 IE_RQ_* raw types.
 * Workgroup owns 32 consecutive output rows (rbase = 32 * wgid(), 4 rows/wave
 * across 8 waves = 1 Q8 block per token per workgroup).
 * Included inside #ifndef IE_MEGA in kernels/ie_kernels.c. */
#ifndef IE_MEGA

#ifndef LDSP
#define LDSP __attribute__((address_space(3)))
#endif

#define SW_TG 128u
LDS static ie_rq_u64 sw_tab[SW_TG + 2048u];
LDS static float sw_out[16][32];

#define SW_P2(sel) __builtin_amdgcn_perm(0x00D5E7F8u, 0x002B1908u, (sel))
#define SW_SGN_M(c, m) SW_P2(((m) & 0x04040404u) | (c))
#define SW_SGN_B(c, b) SW_P2((((b) * 0x00810204u) & 0x04040404u) | (c))
#define SW_TAB32(i) ((LDSP ie_rq_u32 *)(sw_tab + SW_TG))[(i)]
#define SW_X(w) (*(const G ie_rq_u32 *)(xb + xo + 4u * (w)))

static inline __attribute__((always_inline)) ie_rq_u64 sw_code8(ie_rq_u64 g) {
  ie_rq_u64 c = 0u;
  for (ie_rq_u32 i = 0; i < 8u; i++) {
    const ie_rq_u32 b = (ie_rq_u32)(g >> (8u * i)) & 255u;
    c |= (ie_rq_u64)((b > 8u ? 1u : 0u) + (b > 25u ? 1u : 0u)) << (8u * i);
  }
  return c;
}

static inline __attribute__((always_inline)) const G u8 *sw_uniform(const G u8 *p) {
  const unsigned long v = (unsigned long)p;
  const u32 lo = (u32)__builtin_amdgcn_readfirstlane((u32)v);
  const u32 hi = (u32)__builtin_amdgcn_readfirstlane((u32)(v >> 32u));
  return (const G u8 *)(((unsigned long)hi << 32u) | (unsigned long)lo);
}

static inline __attribute__((always_inline)) int sw_same_lut(u32 t0, u32 t1) {
  if (t0 == t1) return 1;
  if ((t0 == IE_RQ_IQ1_S || t0 == IE_RQ_IQ1_M) && (t1 == IE_RQ_IQ1_S || t1 == IE_RQ_IQ1_M)) return 1;
  return 0;
}

static inline __attribute__((always_inline)) void sw_stage_lut(u32 type) {
  const u32 t = tid();
  if (type == IE_RQ_IQ2_XXS || type == IE_RQ_IQ2_XS || type == IE_RQ_IQ3_XXS) {
    if (t < 128u) sw_tab[t] = ie_rq_ksigns64[t];
  }
  if (type == IE_RQ_IQ2_XXS) sw_tab[SW_TG + t] = sw_code8(ie_rq_iq2xxs_grid[t]);
  if (type == IE_RQ_IQ2_XS) {
    for (u32 i = t; i < 512u; i += NT) sw_tab[SW_TG + i] = sw_code8(ie_rq_iq2xs_grid[i]);
  }
  if (type == IE_RQ_IQ2_S) {
    for (u32 i = t; i < 1024u; i += NT) sw_tab[SW_TG + i] = sw_code8(ie_rq_iq2s_grid[i]);
  }
  if (type == IE_RQ_IQ1_S || type == IE_RQ_IQ1_M) {
    for (u32 i = t; i < 2048u; i += NT) sw_tab[SW_TG + i] = ie_rq_iq1s_grid[i];
  }
  if (type == IE_RQ_IQ3_XXS) SW_TAB32(t) = ie_rq_iq3xxs_grid[t];
  if (type == IE_RQ_IQ3_S) {
    for (u32 i = t; i < 512u; i += NT) SW_TAB32(i) = ie_rq_iq3s_grid[i];
  }
  if (type != IE_RQ_Q2_K) lbarrier();
}

static inline __attribute__((always_inline)) float sw_dot_sub(const ie_rq_u32 type, IE_RQP *row, ie_rq_u32 po,
                                                              IE_RQP *xb, ie_rq_u32 nb, ie_rq_u32 b) {
#pragma clang fp contract(off)
  const ie_rq_u32 s = b & 7u, xo = 32u * b;
  const float dav = *(const G float *)(xb + (32u * nb + 4u * b));
  IE_RQP *pd = row + po;

  switch (type) {
    case IE_RQ_Q2_K: {
      IE_RQP *ps = row + (po + 2u * s);
      IE_RQP *pq = row + (po + (s >> 2u) * 32u);
      const float d = ie_rq_h2f(ie_rq_a16(pd, 80u)), mn = ie_rq_h2f(ie_rq_a16(pd, 82u));
      const ie_rq_u32 sc16 = ie_rq_a16(ps, 0u);
      const ie_rq_u32 sh = 2u * (s & 3u), qmask = 0x03030303u << sh;
      float acc = 0.0f;
      for (ie_rq_u32 h = 0; h < 2u; h++) {
        const ie_rq_u32 sc = (sc16 >> (8u * h)) & 255u;
        ie_rq_u32 idots = 0u;
        for (ie_rq_u32 w = 0; w < 4u; w++) idots = ie_dot4_us(idots, ie_rq_a32(pq, 16u + 16u * h + 4u * w) & qmask, SW_X(4u * h + w));
        const ie_rq_u32 idot = (ie_rq_u32)((int)idots >> sh);
        const float dw = d * (float)(sc & 15u), dm = mn * (float)(sc >> 4u);
        const ie_rq_u32 xs = ie_dot4_ss(ie_dot4_ss(0u, SW_X(4u * h), 0x01010101u), SW_X(4u * h + 1u), 0x01010101u) +
                             ie_dot4_ss(ie_dot4_ss(0u, SW_X(4u * h + 2u), 0x01010101u), SW_X(4u * h + 3u), 0x01010101u);
        const float term = dav * (dw * (float)(int)idot - dm * (float)(int)xs);
        acc = h == 0u ? term : acc + term;
      }
      return acc;
    }
    case IE_RQ_IQ2_XXS: {
      IE_RQP *ps = row + (po + 8u * s);
      const float d = ie_rq_h2f(ie_rq_a16(pd, 0u));
      const ie_rq_u32 a0 = ie_rq_a32(ps, 2u), a1 = ie_rq_a32(ps, 6u);
      const float dw = d * (0.5f + (float)(a1 >> 28u)) * 0.25f;
      ie_rq_u32 idot = 0u;
      for (ie_rq_u32 l = 0; l < 4u; l++) {
        const ie_rq_u64 m = sw_tab[(a1 >> (7u * l)) & 127u], g = sw_tab[SW_TG + ie_byte(a0, l)];
        idot = ie_dot4_ss(ie_dot4_ss(idot, SW_SGN_M((ie_rq_u32)g, (ie_rq_u32)m), SW_X(2u * l)),
                          SW_SGN_M((ie_rq_u32)(g >> 32u), (ie_rq_u32)(m >> 32u)), SW_X(2u * l + 1u));
      }
      return dav * (dw * (float)(int)idot);
    }
    case IE_RQ_IQ2_XS: {
      IE_RQP *ps = row + (po + 8u * s);
      IE_RQP *psc = row + (po + 2u * (s >> 1u));
      const float d = ie_rq_h2f(ie_rq_a16(pd, 0u));
      const ie_rq_u32 sc = (ie_rq_a16(psc, 66u) >> (8u * (s & 1u))) & 255u;
      const float dw0 = d * (0.5f + (float)(sc & 15u)) * 0.25f, dw1 = d * (0.5f + (float)(sc >> 4u)) * 0.25f;
      ie_rq_u32 id0 = 0u, id1 = 0u;
      for (ie_rq_u32 l = 0; l < 4u; l++) {
        const ie_rq_u32 q2 = ie_rq_a16(ps, 2u + 2u * l);
        const ie_rq_u64 m = sw_tab[q2 >> 9u], g = sw_tab[SW_TG + (q2 & 511u)];
        const ie_rq_u32 id = ie_dot4_ss(ie_dot4_ss(0u, SW_SGN_M((ie_rq_u32)g, (ie_rq_u32)m), SW_X(2u * l)),
                                        SW_SGN_M((ie_rq_u32)(g >> 32u), (ie_rq_u32)(m >> 32u)), SW_X(2u * l + 1u));
        if (l < 2u) id0 += id; else id1 += id;
      }
      return dav * (dw0 * (float)(int)id0 + dw1 * (float)(int)id1);
    }
    case IE_RQ_IQ2_S: {
      IE_RQP *ps4 = row + (po + 4u * s);
      IE_RQP *ps2 = row + (po + 2u * (s >> 1u));
      const float d = ie_rq_h2f(ie_rq_a16(pd, 0u));
      const ie_rq_u32 qh = (ie_rq_a16(ps2, 66u) >> (8u * (s & 1u))) & 255u;
      const ie_rq_u32 sc = (ie_rq_a16(ps2, 74u) >> (8u * (s & 1u))) & 255u;
      const ie_rq_u32 qlo = ie_rq_a32(ps4, 2u), sgn = ie_rq_a32(ps4, 34u);
      const float dw0 = d * (0.5f + (float)(sc & 15u)) * 0.25f, dw1 = d * (0.5f + (float)(sc >> 4u)) * 0.25f;
      ie_rq_u32 id0 = 0u, id1 = 0u;
      for (ie_rq_u32 l = 0; l < 4u; l++) {
        const ie_rq_u32 idx = ie_byte(qlo, l) | ((qh << (8u - 2u * l)) & 0x300u);
        const ie_rq_u32 sb = ie_byte(sgn, l);
        const ie_rq_u64 g = sw_tab[SW_TG + idx];
        const ie_rq_u32 id = ie_dot4_ss(ie_dot4_ss(0u, SW_SGN_B((ie_rq_u32)g, sb & 15u), SW_X(2u * l)),
                                        SW_SGN_B((ie_rq_u32)(g >> 32u), sb >> 4u), SW_X(2u * l + 1u));
        if (l < 2u) id0 += id; else id1 += id;
      }
      return dav * (dw0 * (float)(int)id0 + dw1 * (float)(int)id1);
    }
    case IE_RQ_IQ3_XXS: {
      IE_RQP *ps8 = row + (po + 8u * s);
      IE_RQP *ps4 = row + (po + 4u * s);
      const float d = ie_rq_h2f(ie_rq_a16(pd, 0u));
      const ie_rq_u32 aux = ie_rq_a32(ps4, 66u);
      const float dw = d * (0.5f + (float)(aux >> 28u)) * 0.5f;
      ie_rq_u32 idot = 0u;
      for (ie_rq_u32 l = 0; l < 4u; l++) {
        const ie_rq_u32 qp = ie_rq_a16(ps8, 2u + 2u * l);
        const ie_rq_u32 g1 = SW_TAB32(qp & 255u), g2 = SW_TAB32(qp >> 8u);
        const ie_rq_u64 m = sw_tab[(aux >> (7u * l)) & 127u];
        idot = ie_dot4_ss(ie_dot4_ss(idot, ie_rq_ssign(g1, (ie_rq_u32)m), SW_X(2u * l)),
                          ie_rq_ssign(g2, (ie_rq_u32)(m >> 32u)), SW_X(2u * l + 1u));
      }
      return dav * (dw * (float)(int)idot);
    }
    case IE_RQ_IQ3_S: {
      IE_RQP *ps8 = row + (po + 8u * s);
      IE_RQP *ps4 = row + (po + 4u * s);
      IE_RQP *ps2 = row + (po + 2u * (s >> 1u));
      IE_RQP *psc = row + (po + 2u * (s >> 2u));
      const float d = ie_rq_h2f(ie_rq_a16(pd, 0u));
      const ie_rq_u32 qh = (ie_rq_a16(ps2, 66u) >> (8u * (s & 1u))) & 255u;
      const ie_rq_u32 sc = (ie_rq_a16(psc, 106u) >> (4u * (s & 3u))) & 15u;
      const ie_rq_u32 sgn = ie_rq_a32(ps4, 74u);
      const float dw = d * (float)(int)(1u + 2u * sc);
      ie_rq_u32 idot = 0u;
      for (ie_rq_u32 l = 0; l < 4u; l++) {
        const ie_rq_u32 qp = ie_rq_a16(ps8, 2u + 2u * l);
        const ie_rq_u32 i1 = (qp & 255u) | (((qh >> (2u * l)) & 1u) << 8u);
        const ie_rq_u32 i2 = (qp >> 8u) | (((qh >> (2u * l + 1u)) & 1u) << 8u);
        const ie_rq_u32 g1 = SW_TAB32(i1), g2 = SW_TAB32(i2);
        const ie_rq_u64 m = ie_rq_mask8(ie_byte(sgn, l));
        idot = ie_dot4_ss(ie_dot4_ss(idot, ie_rq_ssign(g1, (ie_rq_u32)m), SW_X(2u * l)),
                          ie_rq_ssign(g2, (ie_rq_u32)(m >> 32u)), SW_X(2u * l + 1u));
      }
      return dav * (dw * (float)(int)idot);
    }
    case IE_RQ_IQ1_S: {
      IE_RQP *pi = row + (po + 4u * s);
      IE_RQP *ph = row + (po + 2u * s);
      const float d = ie_rq_h2f(ie_rq_a16(pd, 0u));
      const ie_rq_u32 qh = ie_rq_a16(ph, 34u), qlo = ie_rq_a32(pi, 2u);
      const float dw = d * (float)(int)(2u * ((qh >> 12u) & 7u) + 1u);
      const float del = (qh & 0x8000u) != 0u ? -0.125f : 0.125f;
      ie_rq_u32 idot = 0u;
      for (ie_rq_u32 l = 0; l < 4u; l++) {
        const ie_rq_u32 idx = ie_byte(qlo, l) | (((qh >> (3u * l)) & 7u) << 8u);
        const ie_rq_u64 g = sw_tab[SW_TG + idx];
        idot = ie_dot4_ss(ie_dot4_ss(idot, (ie_rq_u32)g, SW_X(2u * l)), (ie_rq_u32)(g >> 32u), SW_X(2u * l + 1u));
      }
      const ie_rq_u32 as = *(const G ie_rq_u32 *)(xb + (36u * nb + 4u * b));
      return dav * (dw * ((float)(int)idot + del * (float)(int)as));
    }
    case IE_RQ_IQ1_M: {
      IE_RQP *pi = row + (po + 4u * s);
      IE_RQP *ph = row + (po + 2u * s);
      const ie_rq_u32 slo = ie_rq_a32(pd, 48u), shi = ie_rq_a32(pd, 52u);
      const ie_rq_u32 s0 = slo & 0xFFFFu, s1 = slo >> 16u, s2 = shi & 0xFFFFu, s3 = shi >> 16u;
      const float d = ie_rq_h2f((s0 >> 12u) | ((s1 >> 8u) & 0x00F0u) | ((s2 >> 4u) & 0x0F00u) | (s3 & 0xF000u));
      const ie_rq_u32 pair = (s & 4u) != 0u ? shi : slo;
      const ie_rq_u32 scv = pair >> (16u * ((s >> 1u) & 1u)), qhb = ie_rq_a16(ph, 32u), qlo = ie_rq_a32(pi, 0u), sh = 6u * (s & 1u);
      const float dq0 = (d * (float)(int)(2u * ((scv >> sh) & 7u) + 1u)) * 0.125f;
      const float dq1 = (d * (float)(int)(2u * ((scv >> (sh + 3u)) & 7u) + 1u)) * 0.125f;
      ie_rq_u32 c0 = 0u, c1 = 0u;
      for (ie_rq_u32 l = 0; l < 4u; l++) {
        const ie_rq_u32 qh = l < 2u ? (qhb & 255u) : (qhb >> 8u);
        const ie_rq_u32 idx = ie_byte(qlo, l) | ((qh << (8u - 4u * (l & 1u))) & 0x700u);
        const ie_rq_u64 g = sw_tab[SW_TG + idx];
        const ie_rq_u32 tbl = (qh & ((l & 1u) != 0u ? 0x80u : 0x08u)) != 0u ? 0xF70007FFu : 0xF9000901u;
        const ie_rq_u32 wl = __builtin_amdgcn_perm(0u, tbl, (ie_rq_u32)g & 0x03030303u);
        const ie_rq_u32 wh = __builtin_amdgcn_perm(0u, tbl, (ie_rq_u32)(g >> 32u) & 0x03030303u);
        const ie_rq_u32 xl = SW_X(2u * l), xh = SW_X(2u * l + 1u);
        if (l < 2u) c0 = ie_dot4_ss(ie_dot4_ss(c0, wl, xl), wh, xh);
        else c1 = ie_dot4_ss(ie_dot4_ss(c1, wl, xl), wh, xh);
      }
      return dav * (dq0 * (float)(int)c0 + dq1 * (float)(int)c1);
    }
    default: return 0.0f;
  }
}

#undef SW_X

/* T = 1 fast path when type0 == type1: shares activation loads across 4 gate + 4 up rows per wave. */
static inline __attribute__((always_inline)) void sw_t1_same(const u32 type, const G u8 *q0, const G u8 *q1,
                                                             const G u8 *xt0, u32 rbase, u32 rows, u32 nb) {
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 r0 = rbase + 4u * wv;
  if (r0 >= rows) {
    if (l == 0u) {
      _Pragma("unroll") for (u32 k = 0; k < 4u; k++) sw_out[0][4u * wv + k] = 0.0f;
    }
    return;
  }
  const u32 mb = nb / 8u, bs = ie_rq_bs(type);
  const unsigned long rstride = (unsigned long)mb * bs;
  const G u8 *row0[4], *row1[4];
  _Pragma("unroll") for (u32 k = 0; k < 4u; k++) {
    const u32 rk = r0 + k < rows ? r0 + k : r0;
    row0[k] = sw_uniform(q0 + (unsigned long)rk * rstride);
    row1[k] = sw_uniform(q1 + (unsigned long)rk * rstride);
  }
  float a0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, a1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (u32 b = l; b < nb; b += 32u) {
    const u32 po = (b >> 3u) * bs;
    _Pragma("unroll") for (u32 k = 0; k < 4u; k++) a0[k] += sw_dot_sub(type, row0[k], po, xt0, nb, b);
    _Pragma("unroll") for (u32 k = 0; k < 4u; k++) a1[k] += sw_dot_sub(type, row1[k], po, xt0, nb, b);
  }
  _Pragma("unroll") for (u32 k = 0; k < 4u; k++) {
    const float g = wave_tree_f(a0[k]), u = wave_tree_f(a1[k]);
    if (l == 0u) sw_out[0][4u * wv + k] = r0 + k < rows ? (g / (1.0f + __builtin_expf(-g))) * u : 0.0f;
  }
}

/* T = 1 single-matrix pass (pass 0 = gate, pass 1 = up + SwiGLU) for mixed (type0 != type1) layers. */
static inline __attribute__((always_inline)) void sw_t1_pass(const u32 type, const G u8 *q, const G u8 *xt0,
                                                             u32 rbase, u32 rows, u32 nb, u32 pass) {
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 r0 = rbase + 4u * wv;
  if (r0 >= rows) {
    if (l == 0u) {
      _Pragma("unroll") for (u32 k = 0; k < 4u; k++) sw_out[0][4u * wv + k] = 0.0f;
    }
    return;
  }
  const u32 mb = nb / 8u, bs = ie_rq_bs(type);
  const unsigned long rstride = (unsigned long)mb * bs;
  const G u8 *row[4];
  _Pragma("unroll") for (u32 k = 0; k < 4u; k++) {
    const u32 rk = r0 + k < rows ? r0 + k : r0;
    row[k] = sw_uniform(q + (unsigned long)rk * rstride);
  }
  float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (u32 b = l; b < nb; b += 32u) {
    const u32 po = (b >> 3u) * bs;
    _Pragma("unroll") for (u32 k = 0; k < 4u; k++) a[k] += sw_dot_sub(type, row[k], po, xt0, nb, b);
  }
  _Pragma("unroll") for (u32 k = 0; k < 4u; k++) {
    const float s = wave_tree_f(a[k]);
    if (l == 0u) {
      const u32 rr = 4u * wv + k;
      if (r0 + k >= rows) {
        sw_out[0][rr] = 0.0f;
      } else if (pass == 0u) {
        sw_out[0][rr] = s;
      } else {
        const float g = sw_out[0][rr];
        sw_out[0][rr] = (g / (1.0f + __builtin_expf(-g))) * s;
      }
    }
  }
}

/* Multi-token pass: processes 2 rows/sub-pass x 2 = 4 rows/wave so VGPRs stay low (~85 at T=4, ~120 at T=8, ~140 at T=16, 0 spills). */
#define SW_GRPN(n) ((n) > 9u ? 8u : (n))
static inline __attribute__((always_inline)) void sw_tr_run(const u32 type, const u32 NTOK, const G u8 *q,
                                                            const G u8 *xq, u32 rbase, u32 rows, u32 nb,
                                                            u32 xs, u32 pass) {
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 mb = nb / 8u, bs = ie_rq_bs(type);
  const unsigned long rstride = (unsigned long)mb * bs;
  const G u8 *xt[16];
  _Pragma("unroll") for (u32 t = 0; t < NTOK; t++) xt[t] = sw_uniform(xq + (unsigned long)t * xs);

#pragma clang loop unroll(disable)
  for (u32 rp = 0; rp < 2u; rp++) {
    const u32 r0 = rbase + 4u * wv + 2u * rp;
    if (r0 >= rows) {
      if (l == 0u) {
        _Pragma("unroll") for (u32 k = 0; k < 2u; k++)
          _Pragma("unroll") for (u32 t = 0; t < NTOK; t++) sw_out[t][4u * wv + 2u * rp + k] = 0.0f;
      }
      continue;
    }
    const G u8 *row[2];
    row[0] = sw_uniform(q + (unsigned long)r0 * rstride);
    row[1] = sw_uniform(q + (unsigned long)(r0 + 1u < rows ? r0 + 1u : r0) * rstride);
    float a[2][16];
    _Pragma("unroll") for (u32 k = 0; k < 2u; k++)
      _Pragma("unroll") for (u32 t = 0; t < NTOK; t++) a[k][t] = 0.0f;
    for (u32 b = l; b < nb; b += 32u) {
      const u32 po = (b >> 3u) * bs;
      _Pragma("unroll") for (u32 g0 = 0; g0 < NTOK; g0 += SW_GRPN(NTOK)) {
        _Pragma("unroll") for (u32 k = 0; k < 2u; k++) {
          _Pragma("unroll") for (u32 t = g0; t < g0 + SW_GRPN(NTOK) && t < NTOK; t++)
            a[k][t] += sw_dot_sub(type, row[k], po, xt[t], nb, b);
        }
        if (SW_GRPN(NTOK) < NTOK) __builtin_amdgcn_sched_barrier(0);
      }
    }
    _Pragma("unroll") for (u32 k = 0; k < 2u; k++) {
      const u32 rr = 4u * wv + 2u * rp + k;
      _Pragma("unroll") for (u32 t = 0; t < NTOK; t++) {
        const float s = wave_tree_f(a[k][t]);
        if (l == 0u) {
          if (r0 + k >= rows) {
            sw_out[t][rr] = 0.0f;
          } else if (pass == 0u) {
            sw_out[t][rr] = s;
          } else {
            const float g = sw_out[t][rr];
            sw_out[t][rr] = (g / (1.0f + __builtin_expf(-g))) * s;
          }
        }
      }
    }
  }
}

#define SW_T1_SAME_CASE(tp) case tp: sw_t1_same(tp, q0, q1, xt0, rbase, rows, nb); break;
#define SW_T1_PASS_CASE(tp) case tp: sw_t1_pass(tp, q, xt0, rbase, rows, nb, pass); break;

KERNEL ie_gemv_raw_swiglu(const G u8 *q0, u32 type0, const G u8 *q1, u32 type1, const G u8 *xq,
                          G float *y, G u8 *oq, u32 rows, u32 nb, u32 T, u32 xs, u32 ys, u32 qs) {
  (void)xs, (void)ys, (void)qs;
  const u32 wg = wgid(), rbase = 32u * wg;
  if (T == 0u || rbase >= rows) return;
  const G u8 *xt0 = sw_uniform(xq);

  if (type0 == type1) {
    sw_stage_lut(type0);
    switch (type0) {
      SW_T1_SAME_CASE(IE_RQ_Q2_K)
      SW_T1_SAME_CASE(IE_RQ_IQ2_XXS)
      SW_T1_SAME_CASE(IE_RQ_IQ2_XS)
      SW_T1_SAME_CASE(IE_RQ_IQ3_XXS)
      SW_T1_SAME_CASE(IE_RQ_IQ1_S)
      SW_T1_SAME_CASE(IE_RQ_IQ2_S)
      SW_T1_SAME_CASE(IE_RQ_IQ1_M)
      SW_T1_SAME_CASE(IE_RQ_IQ3_S)
      default: return;
    }
  } else {
#pragma clang loop unroll(disable)
    for (u32 pass = 0; pass < 2u; pass++) {
      const u32 type = pass == 0u ? type0 : type1;
      const G u8 *q = pass == 0u ? q0 : q1;
      if (pass == 0u) {
        sw_stage_lut(type0);
      } else if (!sw_same_lut(type0, type1) && type1 != IE_RQ_Q2_K) {
        lbarrier();
        sw_stage_lut(type1);
      }
      switch (type) {
        SW_T1_PASS_CASE(IE_RQ_Q2_K)
        SW_T1_PASS_CASE(IE_RQ_IQ2_XXS)
        SW_T1_PASS_CASE(IE_RQ_IQ2_XS)
        SW_T1_PASS_CASE(IE_RQ_IQ3_XXS)
        SW_T1_PASS_CASE(IE_RQ_IQ1_S)
        SW_T1_PASS_CASE(IE_RQ_IQ2_S)
        SW_T1_PASS_CASE(IE_RQ_IQ1_M)
        SW_T1_PASS_CASE(IE_RQ_IQ3_S)
        default: break;
      }
    }
  }

  lbarrier();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  if (w == 0u) {
    const float v = sw_out[0][l];
    if (y && rbase + l < rows) y[rbase + l] = v;
    if (oq) quant_wave(v, wg, oq, rows >> 5u);
  }
}

#define SW_TR4_TYPE_CASE(tp)                                                      \
  case tp:                                                                        \
    switch (T) {                                                                  \
      case 1u: sw_tr_run(tp, 1u, q, xq, rbase, rows, nb, xs, pass); break;        \
      case 2u: sw_tr_run(tp, 2u, q, xq, rbase, rows, nb, xs, pass); break;        \
      case 3u: sw_tr_run(tp, 3u, q, xq, rbase, rows, nb, xs, pass); break;        \
      default: sw_tr_run(tp, 4u, q, xq, rbase, rows, nb, xs, pass); break;        \
    }                                                                             \
    break;

KERNEL ie_gemv_raw_swiglu_tr4(const G u8 *q0, u32 type0, const G u8 *q1, u32 type1, const G u8 *xq,
                              G float *y, G u8 *oq, u32 rows, u32 nb, u32 T, u32 xs, u32 ys, u32 qs) {
  const u32 wg = wgid(), rbase = 32u * wg;
  if (T == 0u || T > 4u || rbase >= rows) return;

#pragma clang loop unroll(disable)
  for (u32 pass = 0; pass < 2u; pass++) {
    const u32 type = pass == 0u ? type0 : type1;
    const G u8 *q = pass == 0u ? q0 : q1;
    if (pass == 0u) {
      sw_stage_lut(type0);
    } else if (!sw_same_lut(type0, type1) && type1 != IE_RQ_Q2_K) {
      lbarrier();
      sw_stage_lut(type1);
    }
    switch (type) {
      SW_TR4_TYPE_CASE(IE_RQ_Q2_K)
      SW_TR4_TYPE_CASE(IE_RQ_IQ2_XXS)
      SW_TR4_TYPE_CASE(IE_RQ_IQ2_XS)
      SW_TR4_TYPE_CASE(IE_RQ_IQ3_XXS)
      SW_TR4_TYPE_CASE(IE_RQ_IQ1_S)
      SW_TR4_TYPE_CASE(IE_RQ_IQ2_S)
      SW_TR4_TYPE_CASE(IE_RQ_IQ1_M)
      SW_TR4_TYPE_CASE(IE_RQ_IQ3_S)
      default: break;
    }
  }

  lbarrier();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  if (w < T) {
    const float v = sw_out[w][l];
    if (y && rbase + l < rows) ((G float *)((G u8 *)y + (unsigned long)w * ys))[rbase + l] = v;
    if (oq) quant_wave(v, wg, oq + (unsigned long)w * qs, rows >> 5u);
  }
}

#define SW_TR8_TYPE_CASE(tp)                                                      \
  case tp:                                                                        \
    switch (T) {                                                                  \
      case 5u: sw_tr_run(tp, 5u, q, xq, rbase, rows, nb, xs, pass); break;        \
      case 6u: sw_tr_run(tp, 6u, q, xq, rbase, rows, nb, xs, pass); break;        \
      case 7u: sw_tr_run(tp, 7u, q, xq, rbase, rows, nb, xs, pass); break;        \
      case 8u: sw_tr_run(tp, 8u, q, xq, rbase, rows, nb, xs, pass); break;        \
      default: sw_tr_run(tp, 9u, q, xq, rbase, rows, nb, xs, pass); break;        \
    }                                                                             \
    break;

KERNEL ie_gemv_raw_swiglu_tr8(const G u8 *q0, u32 type0, const G u8 *q1, u32 type1, const G u8 *xq,
                              G float *y, G u8 *oq, u32 rows, u32 nb, u32 T, u32 xs, u32 ys, u32 qs) {
  const u32 wg = wgid(), rbase = 32u * wg;
  if (T < 5u || T > 9u || rbase >= rows) return;

#pragma clang loop unroll(disable)
  for (u32 pass = 0; pass < 2u; pass++) {
    const u32 type = pass == 0u ? type0 : type1;
    const G u8 *q = pass == 0u ? q0 : q1;
    if (pass == 0u) {
      sw_stage_lut(type0);
    } else if (!sw_same_lut(type0, type1) && type1 != IE_RQ_Q2_K) {
      lbarrier();
      sw_stage_lut(type1);
    }
    switch (type) {
      SW_TR8_TYPE_CASE(IE_RQ_Q2_K)
      SW_TR8_TYPE_CASE(IE_RQ_IQ2_XXS)
      SW_TR8_TYPE_CASE(IE_RQ_IQ2_XS)
      SW_TR8_TYPE_CASE(IE_RQ_IQ3_XXS)
      SW_TR8_TYPE_CASE(IE_RQ_IQ1_S)
      SW_TR8_TYPE_CASE(IE_RQ_IQ2_S)
      SW_TR8_TYPE_CASE(IE_RQ_IQ1_M)
      SW_TR8_TYPE_CASE(IE_RQ_IQ3_S)
      default: break;
    }
  }

  lbarrier();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  for (u32 t = w; t < T; t += 8u) {
    const float v = sw_out[t][l];
    if (y && rbase + l < rows) ((G float *)((G u8 *)y + (unsigned long)t * ys))[rbase + l] = v;
    if (oq) quant_wave(v, wg, oq + (unsigned long)t * qs, rows >> 5u);
  }
}

#define SW_TR16_TYPE_CASE(tp)                                                      \
  case tp:                                                                         \
    switch (T) {                                                                   \
      case 10u: sw_tr_run(tp, 10u, q, xq, rbase, rows, nb, xs, pass); break;       \
      case 11u: sw_tr_run(tp, 11u, q, xq, rbase, rows, nb, xs, pass); break;       \
      case 12u: sw_tr_run(tp, 12u, q, xq, rbase, rows, nb, xs, pass); break;       \
      case 13u: sw_tr_run(tp, 13u, q, xq, rbase, rows, nb, xs, pass); break;       \
      case 14u: sw_tr_run(tp, 14u, q, xq, rbase, rows, nb, xs, pass); break;       \
      case 15u: sw_tr_run(tp, 15u, q, xq, rbase, rows, nb, xs, pass); break;       \
      default: sw_tr_run(tp, 16u, q, xq, rbase, rows, nb, xs, pass); break;        \
    }                                                                              \
    break;

KERNEL ie_gemv_raw_swiglu_tr16(const G u8 *q0, u32 type0, const G u8 *q1, u32 type1, const G u8 *xq,
                               G float *y, G u8 *oq, u32 rows, u32 nb, u32 T, u32 xs, u32 ys, u32 qs) {
  const u32 wg = wgid(), rbase = 32u * wg;
  if (T < 10u || T > 16u || rbase >= rows) return;

#pragma clang loop unroll(disable)
  for (u32 pass = 0; pass < 2u; pass++) {
    const u32 type = pass == 0u ? type0 : type1;
    const G u8 *q = pass == 0u ? q0 : q1;
    if (pass == 0u) {
      sw_stage_lut(type0);
    } else if (!sw_same_lut(type0, type1) && type1 != IE_RQ_Q2_K) {
      lbarrier();
      sw_stage_lut(type1);
    }
    switch (type) {
      SW_TR16_TYPE_CASE(IE_RQ_Q2_K)
      SW_TR16_TYPE_CASE(IE_RQ_IQ2_XXS)
      SW_TR16_TYPE_CASE(IE_RQ_IQ2_XS)
      SW_TR16_TYPE_CASE(IE_RQ_IQ3_XXS)
      SW_TR16_TYPE_CASE(IE_RQ_IQ1_S)
      SW_TR16_TYPE_CASE(IE_RQ_IQ2_S)
      SW_TR16_TYPE_CASE(IE_RQ_IQ1_M)
      SW_TR16_TYPE_CASE(IE_RQ_IQ3_S)
      default: break;
    }
  }

  lbarrier();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  for (u32 t = w; t < T; t += 8u) {
    const float v = sw_out[t][l];
    if (y && rbase + l < rows) ((G float *)((G u8 *)y + (unsigned long)t * ys))[rbase + l] = v;
    if (oq) quant_wave(v, wg, oq + (unsigned long)t * qs, rows >> 5u);
  }
}

/* Fused MAT_Q4K gate + up + SwiGLU + Q8 quantization for T = 1 .. 16 tokens (e.g. Saluki blk.64 MTP FFN).
 * Supports both concatenated [gate | up] (qw1 == qw0, roff1 == ffn, rows0 == rows1 == 2 * ffn) and separate
 * wgate + wup (roff1 == 0, rows0 == rows1 == ffn). Bitwise identical to ie_gemv_q4kq8 + ie_swiglu. */
KERNEL ie_gemv_q4k_swiglu(const G u32 *qw0, const G u16 *qs0, u32 rows0, const G u32 *qw1, const G u16 *qs1,
                          u32 rows1, u32 roff1, const G u8 *xq, G float *y, G u8 *oq, u32 rows, u32 nb, u32 T,
                          u32 xs, u32 ys, u32 qs) {
#pragma clang fp contract(off)
  const u32 wg = wgid(), rbase = 32u * wg;
  if (T == 0u || T > 16u || rbase >= rows) return;
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();

  if (T == 1u) {
    const u32 r0 = rbase + 4u * wv;
    float ag[4] = {0.0f, 0.0f, 0.0f, 0.0f}, au[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (r0 < rows) {
      const G u32 *aw = (const G u32 *)xq;
      const G float *da = (const G float *)(xq + 32u * nb);
      const G u32 *asum = (const G u32 *)(xq + 36u * nb);
      for (u32 t = 0; ie_lane_blk(l, t) < nb; t += 2u) {
#pragma unroll
        for (u32 u = 0; u < 2u; u++) {
          const int okb = ie_lane_blk(l, t + u) < nb;
          if (!okb) continue;
          const u32 b = ie_lane_blk(l, t + u);
          const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u);
          const float dav = da[b], xa = dav * (float)(int)asum[b];
          const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
          const Mem am = {av};
#pragma unroll
          for (u32 q = 0; q < 4u; q++) {
            const u32 rg = r0 + q < rows ? r0 + q : r0, ru = roff1 + rg;
            const u32x4 wg4 = *(const G u32x4 *)(qw0 + ie_q4_dst_word(rg, b, 0u, nb));
            const u32x4 wu4 = *(const G u32x4 *)(qw1 + ie_q4_dst_word(ru, b, 0u, nb));
            const u32 xg[4] = {wg4.x, wg4.y, wg4.z, wg4.w}, xu[4] = {wu4.x, wu4.y, wu4.z, wu4.w};
            u32 wvg[8], wvu[8];
#pragma unroll
            for (u32 k = 0; k < 4u; k++) {
              wvg[2u * k] = ie_q4k_lo(xg[k]), wvg[2u * k + 1u] = ie_q4k_hi(xg[k]);
              wvu[2u * k] = ie_q4k_lo(xu[k]), wvu[2u * k + 1u] = ie_q4k_hi(xu[k]);
            }
            const u32 smg = qs0[ie_q4k_sm(rg, b, nb)], ddg = ie_q4k_dd(rows0, rg, b, nb);
            const float scg = (float)((const G f16 *)qs0)[ddg] * (float)(smg & 63u);
            const float mmg = (float)((const G f16 *)qs0)[ddg + 1u] * (float)(smg >> 8);
            const u32 smu = qs1[ie_q4k_sm(ru, b, nb)], ddu = ie_q4k_dd(rows1, ru, b, nb);
            const float scu = (float)((const G f16 *)qs1)[ddu] * (float)(smu & 63u);
            const float mmu = (float)((const G f16 *)qs1)[ddu + 1u] * (float)(smu >> 8);
            const Mem umg = {wvg}, umu = {wvu};
            const int dqg = (int)ie_q4k_dot_u(umg, 0u, am, 0u);
            const int dqu = (int)ie_q4k_dot_u(umu, 0u, am, 0u);
            ag[q] = __builtin_fmaf(-mmg, xa, __builtin_fmaf(scg * (float)dqg, dav, ag[q]));
            au[q] = __builtin_fmaf(-mmu, xa, __builtin_fmaf(scu * (float)dqu, dav, au[q]));
          }
        }
      }
    }
#pragma unroll
    for (u32 q = 0; q < 4u; q++) {
      const float g = wave_tree_f(ag[q]), u = wave_tree_f(au[q]);
      if (l == 0u) sw_out[0][4u * wv + q] = r0 + q < rows ? (g / (1.0f + __builtin_expf(-g))) * u : 0.0f;
    }
    lbarrier();
    if (wv == 0u) {
      const float v = sw_out[0][l];
      if (y && rbase + l < rows) y[rbase + l] = v;
      if (oq) quant_wave(v, wg, oq, rows >> 5u);
    }
    return;
  }

#pragma clang loop unroll(disable)
  for (u32 rp = 0; rp < 2u; rp++) {
    const u32 r0 = rbase + 4u * wv + 2u * rp;
    float ag[2][16], au[2][16];
#pragma unroll
    for (u32 q = 0; q < 2u; q++)
#pragma unroll
      for (u32 j = 0; j < 16u; j++) ag[q][j] = 0.0f, au[q][j] = 0.0f;
    if (r0 < rows) {
      for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) {
        const u32 b = ie_lane_blk(l, t);
        u32 wvg[2][8], wvu[2][8];
        float scg[2], mmg[2], scu[2], mmu[2];
#pragma unroll
        for (u32 q = 0; q < 2u; q++) {
          const u32 rg = r0 + q < rows ? r0 + q : r0, ru = roff1 + rg;
          const u32x4 wg4 = *(const G u32x4 *)(qw0 + ie_q4_dst_word(rg, b, 0u, nb));
          const u32x4 wu4 = *(const G u32x4 *)(qw1 + ie_q4_dst_word(ru, b, 0u, nb));
          const u32 xg[4] = {wg4.x, wg4.y, wg4.z, wg4.w}, xu[4] = {wu4.x, wu4.y, wu4.z, wu4.w};
#pragma unroll
          for (u32 k = 0; k < 4u; k++) {
            wvg[q][2u * k] = ie_q4k_lo(xg[k]), wvg[q][2u * k + 1u] = ie_q4k_hi(xg[k]);
            wvu[q][2u * k] = ie_q4k_lo(xu[k]), wvu[q][2u * k + 1u] = ie_q4k_hi(xu[k]);
          }
          const u32 smg = qs0[ie_q4k_sm(rg, b, nb)], ddg = ie_q4k_dd(rows0, rg, b, nb);
          scg[q] = (float)((const G f16 *)qs0)[ddg] * (float)(smg & 63u);
          mmg[q] = (float)((const G f16 *)qs0)[ddg + 1u] * (float)(smg >> 8);
          const u32 smu = qs1[ie_q4k_sm(ru, b, nb)], ddu = ie_q4k_dd(rows1, ru, b, nb);
          scu[q] = (float)((const G f16 *)qs1)[ddu] * (float)(smu & 63u);
          mmu[q] = (float)((const G f16 *)qs1)[ddu + 1u] * (float)(smu >> 8);
        }
#pragma unroll
        for (u32 j = 0; j < 16u; j++) {
          if (j < T) {
            const G u8 *xj = xq + (unsigned long)j * xs;
            const G u32 *aw = (const G u32 *)xj;
            const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u);
            const float dav = ((const G float *)(xj + 32u * nb))[b];
            const float xa = dav * (float)(int)((const G u32 *)(xj + 36u * nb))[b];
            const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const Mem am = {av};
#pragma unroll
            for (u32 q = 0; q < 2u; q++) {
              const Mem umg = {wvg[q]}, umu = {wvu[q]};
              const int dqg = (int)ie_q4k_dot_u(umg, 0u, am, 0u);
              const int dqu = (int)ie_q4k_dot_u(umu, 0u, am, 0u);
              ag[q][j] = __builtin_fmaf(-mmg[q], xa, __builtin_fmaf(scg[q] * (float)dqg, dav, ag[q][j]));
              au[q][j] = __builtin_fmaf(-mmu[q], xa, __builtin_fmaf(scu[q] * (float)dqu, dav, au[q][j]));
            }
          }
        }
      }
    }
#pragma unroll
    for (u32 q = 0; q < 2u; q++) {
      const u32 c = 4u * wv + 2u * rp + q;
#pragma unroll
      for (u32 j = 0; j < 16u; j++) {
        if (j < T) {
          const float g = wave_tree_f(ag[q][j]), u = wave_tree_f(au[q][j]);
          if (l == 0u) sw_out[j][c] = rbase + c < rows ? (g / (1.0f + __builtin_expf(-g))) * u : 0.0f;
        }
      }
    }
  }

  lbarrier();
  for (u32 t = wv; t < T; t += 8u) {
    const float v = sw_out[t][l];
    if (y && rbase + l < rows) ((G float *)((G u8 *)y + (unsigned long)t * ys))[rbase + l] = v;
    if (oq) quant_wave(v, wg, oq + (unsigned long)t * qs, rows >> 5u);
  }
}

#undef SW_GRPN
#undef SW_T1_SAME_CASE
#undef SW_T1_PASS_CASE
#undef SW_TR4_TYPE_CASE
#undef SW_TR8_TYPE_CASE
#undef SW_TR16_TYPE_CASE
#undef SW_TAB32
#undef SW_SGN_B
#undef SW_SGN_M
#undef SW_P2
#undef SW_TG

#endif /* !IE_MEGA */
