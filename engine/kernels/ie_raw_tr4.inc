/* ie_raw_tr4.inc -- T = 1..4 multi-token raw IQ/K GEMV kernels (2 rows/wave,
 * 16 rows/WG) and multi-matrix ie_gemv_raw_tr4_m. Included inside #ifndef IE_MEGA
 * in kernels/ie_kernels.c. Bitwise identical to ie_gemv_raw_*_tr and CPU. */
#ifndef IE_MEGA

#ifndef LDSP
#define LDSP __attribute__((address_space(3)))
#endif

#ifndef IE_TR4_BASE_LDS
#define IE_TR4_BASE_LDS
LDS static ie_rq_u64 tr4_xxs_grid[256];
LDS static ie_rq_u64 tr4_ks[128];
#endif

#define TR4_TG 128u
LDS static ie_rq_u64 tr4_g2xs[512], tr4_g2s[1024], tr4_g1s[2048];
LDS static ie_rq_u32 tr4_g3xxs[256], tr4_g3s[512];
LDS static ie_rq_u64 tr4_mtab[TR4_TG + 2048u];
LDS static float tr4_ep[4][16];

#define TR4_P2(sel) __builtin_amdgcn_perm(0x00D5E7F8u, 0x002B1908u, (sel))
#define TR4_SGN_M(c, m) TR4_P2(((m) & 0x04040404u) | (c))
#define TR4_SGN_B(c, b) TR4_P2((((b) * 0x00810204u) & 0x04040404u) | (c))

static inline __attribute__((always_inline)) ie_rq_u64 tr4_code8(ie_rq_u64 g) {
  ie_rq_u64 c = 0u;
  for (ie_rq_u32 i = 0; i < 8u; i++) {
    const ie_rq_u32 b = (ie_rq_u32)(g >> (8u * i)) & 255u;
    c |= (ie_rq_u64)((b > 8u ? 1u : 0u) + (b > 25u ? 1u : 0u)) << (8u * i);
  }
  return c;
}

static inline __attribute__((always_inline)) const G u8 *tr4_uniform(const G u8 *p) {
  const unsigned long v = (unsigned long)p;
  const u32 lo = (u32)__builtin_amdgcn_readfirstlane((u32)v);
  const u32 hi = (u32)__builtin_amdgcn_readfirstlane((u32)(v >> 32u));
  return (const G u8 *)(((unsigned long)hi << 32u) | (unsigned long)lo);
}

#define TR4_MTAB32(i) ((LDSP ie_rq_u32 *)(tr4_mtab + TR4_TG))[(i)]
#define TR4_KS(i) (mt ? tr4_mtab[(i)] : tr4_ks[(i)])
#define TR4_G2XXS(i) (mt ? tr4_mtab[TR4_TG + (i)] : tr4_xxs_grid[(i)])
#define TR4_G2XS(i) (mt ? tr4_mtab[TR4_TG + (i)] : tr4_g2xs[(i)])
#define TR4_G2S(i) (mt ? tr4_mtab[TR4_TG + (i)] : tr4_g2s[(i)])
#define TR4_G1S(i) (mt ? tr4_mtab[TR4_TG + (i)] : tr4_g1s[(i)])
#define TR4_G3XXS(i) (mt ? TR4_MTAB32(i) : tr4_g3xxs[(i)])
#define TR4_G3S(i) (mt ? TR4_MTAB32(i) : tr4_g3s[(i)])
#define TR4_X(w) (*(const G ie_rq_u32 *)(xb + xo + 4u * (w)))

static inline __attribute__((always_inline)) float tr4_dot_sub(const ie_rq_u32 type, const int mt, IE_RQP *row, ie_rq_u32 po,
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
        for (ie_rq_u32 w = 0; w < 4u; w++) idots = ie_dot4_us(idots, ie_rq_a32(pq, 16u + 16u * h + 4u * w) & qmask, TR4_X(4u * h + w));
        const ie_rq_u32 idot = (ie_rq_u32)((int)idots >> sh);
        const float dw = d * (float)(sc & 15u), dm = mn * (float)(sc >> 4u);
        const ie_rq_u32 xs = ie_dot4_ss(ie_dot4_ss(0u, TR4_X(4u * h), 0x01010101u), TR4_X(4u * h + 1u), 0x01010101u) +
                             ie_dot4_ss(ie_dot4_ss(0u, TR4_X(4u * h + 2u), 0x01010101u), TR4_X(4u * h + 3u), 0x01010101u);
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
        const ie_rq_u64 m = TR4_KS((a1 >> (7u * l)) & 127u), g = TR4_G2XXS(ie_byte(a0, l));
        idot = ie_dot4_ss(ie_dot4_ss(idot, TR4_SGN_M((ie_rq_u32)g, (ie_rq_u32)m), TR4_X(2u * l)),
                          TR4_SGN_M((ie_rq_u32)(g >> 32u), (ie_rq_u32)(m >> 32u)), TR4_X(2u * l + 1u));
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
        const ie_rq_u64 m = TR4_KS(q2 >> 9u), g = TR4_G2XS(q2 & 511u);
        const ie_rq_u32 id = ie_dot4_ss(ie_dot4_ss(0u, TR4_SGN_M((ie_rq_u32)g, (ie_rq_u32)m), TR4_X(2u * l)),
                                        TR4_SGN_M((ie_rq_u32)(g >> 32u), (ie_rq_u32)(m >> 32u)), TR4_X(2u * l + 1u));
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
        const ie_rq_u64 g = TR4_G2S(idx);
        const ie_rq_u32 id = ie_dot4_ss(ie_dot4_ss(0u, TR4_SGN_B((ie_rq_u32)g, sb & 15u), TR4_X(2u * l)),
                                        TR4_SGN_B((ie_rq_u32)(g >> 32u), sb >> 4u), TR4_X(2u * l + 1u));
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
        const ie_rq_u32 g1 = TR4_G3XXS(qp & 255u), g2 = TR4_G3XXS(qp >> 8u);
        const ie_rq_u64 m = TR4_KS((aux >> (7u * l)) & 127u);
        idot = ie_dot4_ss(ie_dot4_ss(idot, ie_rq_ssign(g1, (ie_rq_u32)m), TR4_X(2u * l)),
                          ie_rq_ssign(g2, (ie_rq_u32)(m >> 32u)), TR4_X(2u * l + 1u));
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
        const ie_rq_u32 g1 = TR4_G3S(i1), g2 = TR4_G3S(i2);
        const ie_rq_u64 m = ie_rq_mask8(ie_byte(sgn, l));
        idot = ie_dot4_ss(ie_dot4_ss(idot, ie_rq_ssign(g1, (ie_rq_u32)m), TR4_X(2u * l)),
                          ie_rq_ssign(g2, (ie_rq_u32)(m >> 32u)), TR4_X(2u * l + 1u));
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
        const ie_rq_u64 g = TR4_G1S(idx);
        idot = ie_dot4_ss(ie_dot4_ss(idot, (ie_rq_u32)g, TR4_X(2u * l)), (ie_rq_u32)(g >> 32u), TR4_X(2u * l + 1u));
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
        const ie_rq_u64 g = TR4_G1S(idx);
        const ie_rq_u32 tbl = (qh & ((l & 1u) != 0u ? 0x80u : 0x08u)) != 0u ? 0xF70007FFu : 0xF9000901u;
        const ie_rq_u32 wl = __builtin_amdgcn_perm(0u, tbl, (ie_rq_u32)g & 0x03030303u);
        const ie_rq_u32 wh = __builtin_amdgcn_perm(0u, tbl, (ie_rq_u32)(g >> 32u) & 0x03030303u);
        const ie_rq_u32 xl = TR4_X(2u * l), xh = TR4_X(2u * l + 1u);
        if (l < 2u) c0 = ie_dot4_ss(ie_dot4_ss(c0, wl, xl), wh, xh);
        else c1 = ie_dot4_ss(ie_dot4_ss(c1, wl, xl), wh, xh);
      }
      return dav * (dq0 * (float)(int)c0 + dq1 * (float)(int)c1);
    }
    default: return 0.0f;
  }
}

#undef TR4_X

static inline __attribute__((always_inline)) void gemv_raw_tr4_run(const u32 type, const int mt, const u32 NTOK, u32 wg,
                                                                   const G u8 *q, const G u8 *xq, G float *y, u32 rows,
                                                                   u32 nb, u32 xs, u32 ys, u32 rs,
                                                                   const G float *bias, const G float *res) {
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 rbase = 16u * wg, r0 = rbase + 2u * wv, l = lane();
  if (r0 < rows) {
    const u32 mb = nb / 8u, bs = ie_rq_bs(type);
    const unsigned long rstride = (unsigned long)mb * bs;
    const G u8 *row[2];
    row[0] = tr4_uniform(q + (unsigned long)r0 * rstride);
    row[1] = tr4_uniform(q + (unsigned long)(r0 + 1u < rows ? r0 + 1u : r0) * rstride);
    const G u8 *xt[4];
    _Pragma("unroll") for (u32 t = 0; t < NTOK; t++) xt[t] = tr4_uniform(xq + (unsigned long)t * xs);
    float a[2][4];
    _Pragma("unroll") for (u32 k = 0; k < 2u; k++)
      _Pragma("unroll") for (u32 t = 0; t < NTOK; t++) a[k][t] = 0.0f;
    for (u32 b = l; b < nb; b += 32u) {
      const u32 po = (b >> 3u) * bs;
      _Pragma("unroll") for (u32 k = 0; k < 2u; k++) {
        _Pragma("unroll") for (u32 t = 0; t < NTOK; t++)
          a[k][t] += tr4_dot_sub(type, mt, row[k], po, xt[t], nb, b);
      }
    }
    _Pragma("unroll") for (u32 k = 0; k < 2u; k++) {
      _Pragma("unroll") for (u32 t = 0; t < NTOK; t++) {
        const float s = wave_tree_f(a[k][t]);
        if (l == 0u) tr4_ep[t][2u * wv + k] = s;
      }
    }
  }
  lbarrier();
  const u32 i = tid();
  if (i < 16u * NTOK) {
    const u32 t = i >> 4u, rr = i & 15u, r = rbase + rr;
    if (r < rows) {
      const G float *rt = res ? (const G float *)((const G u8 *)res + (unsigned long)t * rs) : res;
      G float *yt = (G float *)((G u8 *)y + (unsigned long)t * ys);
      yt[r] = epilogue(tr4_ep[t][rr], r, bias, rt);
    }
  }
}

static inline __attribute__((always_inline)) void tr4_stage_lut(const u32 type, const int mt) {
  const u32 t = tid();
  if (type == IE_RQ_IQ2_XXS || type == IE_RQ_IQ2_XS || type == IE_RQ_IQ3_XXS) {
    if (t < 128u) {
      if (mt) tr4_mtab[t] = ie_rq_ksigns64[t];
      else tr4_ks[t] = ie_rq_ksigns64[t];
    }
  }
  if (type == IE_RQ_IQ2_XXS) {
    const ie_rq_u64 c = tr4_code8(ie_rq_iq2xxs_grid[t]);
    if (mt) tr4_mtab[TR4_TG + t] = c;
    else tr4_xxs_grid[t] = c;
  }
  if (type == IE_RQ_IQ2_XS) {
    for (u32 i = t; i < 512u; i += NT) {
      const ie_rq_u64 c = tr4_code8(ie_rq_iq2xs_grid[i]);
      if (mt) tr4_mtab[TR4_TG + i] = c;
      else tr4_g2xs[i] = c;
    }
  }
  if (type == IE_RQ_IQ2_S) {
    for (u32 i = t; i < 1024u; i += NT) {
      const ie_rq_u64 c = tr4_code8(ie_rq_iq2s_grid[i]);
      if (mt) tr4_mtab[TR4_TG + i] = c;
      else tr4_g2s[i] = c;
    }
  }
  if (type == IE_RQ_IQ1_S || type == IE_RQ_IQ1_M) {
    for (u32 i = t; i < 2048u; i += NT) {
      const ie_rq_u64 g = ie_rq_iq1s_grid[i];
      if (mt) tr4_mtab[TR4_TG + i] = g;
      else tr4_g1s[i] = g;
    }
  }
  if (type == IE_RQ_IQ3_XXS) {
    const ie_rq_u32 g = ie_rq_iq3xxs_grid[t];
    if (mt) TR4_MTAB32(t) = g;
    else tr4_g3xxs[t] = g;
  }
  if (type == IE_RQ_IQ3_S) {
    for (u32 i = t; i < 512u; i += NT) {
      const ie_rq_u32 g = ie_rq_iq3s_grid[i];
      if (mt) TR4_MTAB32(i) = g;
      else tr4_g3s[i] = g;
    }
  }
  if (type != IE_RQ_Q2_K) lbarrier();
}

static inline __attribute__((always_inline)) void gemv_raw_tr4(const u32 type, const G u8 *q, const G u8 *xq, G float *y,
                                                               u32 rows, u32 nb, u32 T, const G float *bias,
                                                               const G float *res, u32 xs, u32 ys, u32 rs) {
  const u32 wg = wgid();
  if (T == 0u || 16u * wg >= rows) return;
  tr4_stage_lut(type, 0);
  switch (T) {
    case 1u: gemv_raw_tr4_run(type, 0, 1u, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 2u: gemv_raw_tr4_run(type, 0, 2u, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 3u: gemv_raw_tr4_run(type, 0, 3u, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    default: gemv_raw_tr4_run(type, 0, 4u, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
  }
}

KERNEL ie_gemv_raw_q2k_tr4(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                           const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                           float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count;
  gemv_raw_tr4(IE_RQ_Q2_K, q, xq, y, rows, nb, T, bias, res, xs, ys, rs);
}
KERNEL ie_gemv_raw_iq2xxs_tr4(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                              const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                              float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count;
  gemv_raw_tr4(IE_RQ_IQ2_XXS, q, xq, y, rows, nb, T, bias, res, xs, ys, rs);
}
KERNEL ie_gemv_raw_iq2xs_tr4(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                             const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                             float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count;
  gemv_raw_tr4(IE_RQ_IQ2_XS, q, xq, y, rows, nb, T, bias, res, xs, ys, rs);
}
KERNEL ie_gemv_raw_iq3xxs_tr4(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                              const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                              float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count;
  gemv_raw_tr4(IE_RQ_IQ3_XXS, q, xq, y, rows, nb, T, bias, res, xs, ys, rs);
}
KERNEL ie_gemv_raw_iq1s_tr4(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                            const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                            float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count;
  gemv_raw_tr4(IE_RQ_IQ1_S, q, xq, y, rows, nb, T, bias, res, xs, ys, rs);
}
KERNEL ie_gemv_raw_iq2s_tr4(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                            const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                            float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count;
  gemv_raw_tr4(IE_RQ_IQ2_S, q, xq, y, rows, nb, T, bias, res, xs, ys, rs);
}
KERNEL ie_gemv_raw_iq1m_tr4(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                            const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                            float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count;
  gemv_raw_tr4(IE_RQ_IQ1_M, q, xq, y, rows, nb, T, bias, res, xs, ys, rs);
}
KERNEL ie_gemv_raw_iq3s_tr4(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                            const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                            float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count;
  gemv_raw_tr4(IE_RQ_IQ3_S, q, xq, y, rows, nb, T, bias, res, xs, ys, rs);
}

#ifndef IE_TR4M_TYPES
#define IE_TR4M_TYPES (1u << IE_RQ_Q2_K | 1u << IE_RQ_IQ2_XXS | 1u << IE_RQ_IQ2_XS | 1u << IE_RQ_IQ3_XXS | \
                       1u << IE_RQ_IQ1_S | 1u << IE_RQ_IQ2_S | 1u << IE_RQ_IQ1_M | 1u << IE_RQ_IQ3_S)
#endif

static inline __attribute__((always_inline)) void gemv_raw_tr4_mseg(const u32 type, u32 wg, const G u8 *q, const G u8 *xq,
                                                                    G float *y, u32 rows, u32 nb, u32 T,
                                                                    const G float *bias, const G float *res,
                                                                    u32 xs, u32 ys, u32 rs) {
  if (T == 0u || 16u * wg >= rows) return;
  tr4_stage_lut(type, 1);
  switch (T) {
    case 1u: gemv_raw_tr4_run(type, 1, 1u, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 2u: gemv_raw_tr4_run(type, 1, 2u, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 3u: gemv_raw_tr4_run(type, 1, 3u, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    default: gemv_raw_tr4_run(type, 1, 4u, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
  }
}

#define TR4M_SEG(t) \
  case t: \
    if ((IE_TR4M_TYPES) >> t & 1u) gemv_raw_tr4_mseg(t, wg, q, xq, y, rows, nb, T, bias, res, xs, ys, rs); \
    return;

KERNEL ie_gemv_raw_tr4_m(const G u8 *xq, u32 nb, u32 T, u32 xs, u32 g1, u32 g2,
                         const G u8 *q0, G float *y0, u32 rows0, u32 type0, u32 ys0, const G float *res0, u32 rs0, const G float *bias0,
                         const G u8 *q1, G float *y1, u32 rows1, u32 type1, u32 ys1, const G float *res1, u32 rs1, const G float *bias1,
                         const G u8 *q2, G float *y2, u32 rows2, u32 type2, u32 ys2, const G float *res2, u32 rs2, const G float *bias2) {
  const u32 w = wgid(), s = w < g1 ? 0u : w < g2 ? 1u : 2u, wg = w - (s == 0u ? 0u : s == 1u ? g1 : g2);
  const G u8 *q = s == 0u ? q0 : s == 1u ? q1 : q2;
  G float *y = s == 0u ? y0 : s == 1u ? y1 : y2;
  const G float *res = s == 0u ? res0 : s == 1u ? res1 : res2, *bias = s == 0u ? bias0 : s == 1u ? bias1 : bias2;
  const u32 rows = s == 0u ? rows0 : s == 1u ? rows1 : rows2, type = s == 0u ? type0 : s == 1u ? type1 : type2;
  const u32 ys = s == 0u ? ys0 : s == 1u ? ys1 : ys2, rs = s == 0u ? rs0 : s == 1u ? rs1 : rs2;
  switch (type) {
    TR4M_SEG(IE_RQ_Q2_K)
    TR4M_SEG(IE_RQ_IQ2_XXS)
    TR4M_SEG(IE_RQ_IQ2_XS)
    TR4M_SEG(IE_RQ_IQ3_XXS)
    TR4M_SEG(IE_RQ_IQ1_S)
    TR4M_SEG(IE_RQ_IQ2_S)
    TR4M_SEG(IE_RQ_IQ1_M)
    TR4M_SEG(IE_RQ_IQ3_S)
    case IE_RQ_Q8_0:
      gemv_q8q8_mseg(wg, q, xq, y, rows, nb, T, bias, res, xs, ys, rs);
      return;
    default: return;
  }
}

#undef TR4M_SEG
#undef TR4_MTAB32
#undef TR4_KS
#undef TR4_G2XXS
#undef TR4_G2XS
#undef TR4_G2S
#undef TR4_G1S
#undef TR4_G3XXS
#undef TR4_G3S

#endif /* !IE_MEGA */
