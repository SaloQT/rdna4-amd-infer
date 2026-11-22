/* ============================================================================
 * Inkling-2 Round 7 (Rank 2): Multi-Query Verify Relative Attention for SWA
 * (40 layers, win = 1024) and Single-Pass T<=8 FP16 WMMA Global Verify Attention
 * (8 layers, win = 0, 16K-32K) on RX 9070 XT (gfx1201, wave32, NT=256).
 *
 * Kernels Exported (both match exact 23-argument signature of ver_r4 / qtile_r6):
 *
 * 1. KERNEL_CALL ie_inkling_rel_attn_ver_swa_qtile_r7(...)
 *    - Target: 40 SWA layers (win > 0, e.g. win = 1024) during T = 2..16 verify.
 *              Replaces per-token ie_inkling_rel_attn_ver_r4 which re-reads
 *              167.8 MB of SWA KV 6x per verify step (~1.01 GB > 64 MB L3).
 *    - Grid:   (n_kv * nsplit, (T + 3u) / 4u, 1u), NT = 256 (QT = 4 tokens/WG).
 *              Cuts SWA verify KV traffic 3x at T=6, 4x at T=4/8.
 *    - Guards: !b->kvq8 && !b->kvq4 && win > 0u && T >= 2u && T <= 16u &&
 *              hd == 128u && d_rel == 16u && n_kv > 0u && n_head == 3u * n_kv &&
 *              kv_len > 0u && nsplit >= 1u && (nsplit == 1u || (part && count)) &&
 *              ((qs | os) & 15u) == 0u && ((uintptr_t)q & 15u) == 0u
 *              Suggested nsplit:
 *                u32 np_u = (pos + T > win) ? (win + 3u) : (pos + T);
 *                u32 nsplit = (np_u >= 128u && b->at_part && b->at_count) ? (np_u + 63u) / 64u : 1u;
 *                if (nsplit > 64u) nsplit = 64u;
 *    - Counter: count[t0 * n_kv + kh] (self-reset to 0 by last WG; t0 = tokid()*4).
 *    - Resources: <= 128 VGPR (3 WGs/CU), 12,484 B LDS (<= 16 KiB), 0/0 spills.
 *    - Parity: f32-tol (~1e-6) vs ie_inkling_rel_attn_ver_r4 (exact causal/SWA
 *              window [max(0, pos_q+1-win), pos_q], ring wrap % kv_len, rel_bias, sink).
 *
 * 2. KERNEL ie_inkling_rel_attn_ver_glob_wmma_r7(...)
 *    - Target: 8 Global layers (win == 0) during T = 2..8 verify at long context
 *              (pos0 >= 2048, especially 16K-32K). Processes ALL T <= 8 tokens
 *              (up to 24 GQA queries = 2 WMMA N-tiles) in ONE SINGLE PASS over
 *              global K/V -> cuts 16K-32K Global KV DRAM reads in HALF vs
 *              qtile_r6 (which launches 2 token tiles for T = 5..8)!
 *    - Grid:   (n_kv * nsplit, 1u, 1u), NT = 256.
 *              Suggested nsplit:
 *                u32 np = pos + T;
 *                u32 nsplit = (np >= 256u && b->at_part && b->at_count) ? (np + 255u) / 256u : 1u;
 *                if (nsplit > 64u) nsplit = 64u;
 *                (e.g. 16K -> nsplit=64 -> 512 WGs, 256 keys/WG = 2 chunks of 128;
 *                 32K -> nsplit=64 -> 512 WGs, 512 keys/WG = 4 chunks of 128).
 *    - Guards: !b->kvq8 && !b->kvq4 && !mk.rec && win == 0u && pos >= 2048u &&
 *              T >= 2u && T <= 8u && hd == 128u && d_rel == 16u && n_kv > 0u &&
 *              n_head == 3u * n_kv && kv_len > 0u && nsplit >= 1u &&
 *              (nsplit == 1u || (part && count)) &&
 *              ((qs | os) & 15u) == 0u && ((uintptr_t)q & 15u) == 0u
 *    - Counter: count[kh] (1 counter per KV head, self-reset to 0 by last WG).
 *    - Buffer:  part >= T * n_head * nsplit * 130 floats (exact ver_r4/qtile_r6 layout).
 *    - Resources: <= 128 VGPR (3 WGs/CU), 15,360 B LDS (<= 16 KiB), 0/0 spills.
 *    - Parity: f32-tol (fp16 WMMA QK^T & PV, exact far-field & near-field rel_bias
 *              + causal mask + dynamic per-head sink).
 * ========================================================================== */

_Static_assert(sizeof(kvt) == 2u, "ie_inkling_r7_attn_ver requires f16 KV cache");

#define INK_R7_AV_HD    128u
#define INK_R7_AV_DREL  16u
#define INK_R7_AV_GQA   3u
#define INK_R7_AV_QT    4u
#define INK_R7_AV_PSTR  130u
#define INK_R7_AV_LOG2E 1.44269504088896340736f
#define INK_R7_AV_LN2   0.69314718055994530942f
#define INK_R7_AV_QSC   (0.08838834764831844299f * 1.44269504088896340736f)

static inline __attribute__((always_inline)) float ink_r7_av_sgpr_f(float v) {
  return I2F(__builtin_amdgcn_readfirstlane(F2I(v)));
}

static inline __attribute__((always_inline)) float ink_r7_av_halfwave_sum_all(float v) {
  v += I2F(shl_i(F2I(v), 8));
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return I2F(__builtin_amdgcn_readlane(F2I(v), 0));
}

static inline __attribute__((always_inline)) float ink_r7_av_dot4(f32x4 a, f32x4 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

/* ============================================================================
 * Kernel 1: ie_inkling_rel_attn_ver_swa_qtile_r7 (SWA win > 0, QT=4, <=128 VGPR)
 * ========================================================================== */

#define INK_R7_SWA_QK3(qi, kf_val, sc_val, lane_id, d0_out, d1_out, d2_out)                            \
  do {                                                                                                 \
    const LDSP f32x4 *_r7_qp =                                                                         \
        (const LDSP f32x4 *)(ink_r7_swa_q + (unsigned long)(qi) * (INK_R7_AV_GQA * INK_R7_AV_HD));     \
    const f32x4 _r7_q0 = _r7_qp[0u * 32u + (lane_id)];                                                 \
    const f32x4 _r7_q1 = _r7_qp[1u * 32u + (lane_id)];                                                 \
    const f32x4 _r7_q2 = _r7_qp[2u * 32u + (lane_id)];                                                 \
    (d0_out) = ink_r7_av_sgpr_f(wave_sum_all(ink_r7_av_dot4(_r7_q0, (kf_val))) * (sc_val));            \
    (d1_out) = ink_r7_av_sgpr_f(wave_sum_all(ink_r7_av_dot4(_r7_q1, (kf_val))) * (sc_val));            \
    (d2_out) = ink_r7_av_sgpr_f(wave_sum_all(ink_r7_av_dot4(_r7_q2, (kf_val))) * (sc_val));            \
  } while (0)

#define INK_R7_SWA_ADD_RB3(qi, rb_val, lane_id, d0_io, d1_io, d2_io)                                       \
  do {                                                                                                     \
    const LDSP float *_r7_rp = ink_r7_swa_r + (unsigned long)(qi) * (INK_R7_AV_GQA * INK_R7_AV_DREL);      \
    const float _r7_rv0 = ((lane_id) < INK_R7_AV_DREL) ? _r7_rp[0u * INK_R7_AV_DREL + (lane_id)] : 0.0f;   \
    const float _r7_rv1 = ((lane_id) < INK_R7_AV_DREL) ? _r7_rp[1u * INK_R7_AV_DREL + (lane_id)] : 0.0f;   \
    const float _r7_rv2 = ((lane_id) < INK_R7_AV_DREL) ? _r7_rp[2u * INK_R7_AV_DREL + (lane_id)] : 0.0f;   \
    (d0_io) = ink_r7_av_sgpr_f((d0_io) + ink_r7_av_halfwave_sum_all(_r7_rv0 * (rb_val)));                  \
    (d1_io) = ink_r7_av_sgpr_f((d1_io) + ink_r7_av_halfwave_sum_all(_r7_rv1 * (rb_val)));                  \
    (d2_io) = ink_r7_av_sgpr_f((d2_io) + ink_r7_av_halfwave_sum_all(_r7_rv2 * (rb_val)));                  \
  } while (0)

#define INK_R7_SWA_UPD(m_var, s_var, a_var, d_val, vf_val)                           \
  do {                                                                               \
    const float _r7_mn = __builtin_fmaxf((m_var), (d_val));                          \
    const float _r7_c = ((s_var) > 0.0f) ? __builtin_expf((m_var) - _r7_mn) : 0.0f;  \
    const float _r7_e = __builtin_expf((d_val) - _r7_mn);                            \
    (s_var) = ink_r7_av_sgpr_f((s_var) * _r7_c + _r7_e);                             \
    (m_var) = ink_r7_av_sgpr_f(_r7_mn);                                              \
    (a_var) = (a_var) * _r7_c + _r7_e * (vf_val);                                    \
  } while (0)

#define INK_R7_SWA_EDGE_TOK(qi, kf_val, vf_val, sc_val, lane_id, kp_val, pos_q, win_v, ext_lim,       \
                            m0_v, s0_v, a0_v, m1_v, s1_v, a1_v, m2_v, s2_v, a2_v)                     \
  do {                                                                                                \
    if ((kp_val) <= (pos_q) && ((pos_q) - (kp_val)) < (win_v)) {                                      \
      const u32 _r7_dist = (pos_q) - (kp_val);                                                        \
      float _r7_rb = 0.0f;                                                                            \
      if (_r7_dist < (ext_lim) && (lane_id) < INK_R7_AV_DREL) {                                       \
        _r7_rb = rel_bias[(unsigned long)_r7_dist * INK_R7_AV_DREL + (lane_id)];                      \
      }                                                                                               \
      float _r7_d0, _r7_d1, _r7_d2;                                                                   \
      INK_R7_SWA_QK3((qi), (kf_val), (sc_val), (lane_id), _r7_d0, _r7_d1, _r7_d2);                    \
      if (_r7_dist < (ext_lim)) {                                                                     \
        INK_R7_SWA_ADD_RB3((qi), _r7_rb, (lane_id), _r7_d0, _r7_d1, _r7_d2);                          \
      }                                                                                               \
      INK_R7_SWA_UPD((m0_v), (s0_v), (a0_v), _r7_d0, (vf_val));                                       \
      INK_R7_SWA_UPD((m1_v), (s1_v), (a1_v), _r7_d1, (vf_val));                                       \
      INK_R7_SWA_UPD((m2_v), (s2_v), (a2_v), _r7_d2, (vf_val));                                       \
    }                                                                                                 \
  } while (0)

KERNEL_CALL ie_inkling_rel_attn_ver_swa_qtile_r7(const G float *q, const G kvt *kc, const G kvt *vc,
                                                 const G float *R, const G float *rel_bias, const G float *sink,
                                                 G float *part, G float *out, u32 pos0, u32 T, u32 kv_len, u32 hd,
                                                 u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
                                                 u32 qs, u32 rs, u32 ss, u32 os, G u32 *count) {
#ifdef IE_MEGA
  _Static_assert(sizeof(mk_ink_lds.rel_attn.o) >=
                     (INK_R7_AV_QT * INK_R7_AV_GQA * (INK_R7_AV_HD + INK_R7_AV_DREL)) * sizeof(float),
                 "mk_ink_lds.rel_attn.o too small for QT=4 SWA Q/R staging");
#define ink_r7_swa_m    (mk_ink_lds.rel_attn.m)
#define ink_r7_swa_l    (mk_ink_lds.rel_attn.l)
#define ink_r7_swa_o    (mk_ink_lds.rel_attn.o)
#define ink_r7_swa_last (mk_ink_lds.rel_attn.last)
#define ink_r7_swa_q    ((LDSP float *)&mk_ink_lds.rel_attn.o[0][0][0])
#define ink_r7_swa_r    (((LDSP float *)&mk_ink_lds.rel_attn.o[0][0][0]) + \
                         (INK_R7_AV_QT * INK_R7_AV_GQA * INK_R7_AV_HD))
#else
  static LDS union {
    struct {
      float q[INK_R7_AV_QT * INK_R7_AV_GQA * INK_R7_AV_HD] __attribute__((aligned(16)));
      float r[INK_R7_AV_QT * INK_R7_AV_GQA * INK_R7_AV_DREL] __attribute__((aligned(16)));
    } qr;
    struct {
      float m[INK_R7_AV_GQA][8];
      float l[INK_R7_AV_GQA][8];
      float o[INK_R7_AV_GQA][8][INK_R7_AV_HD] __attribute__((aligned(16)));
    } red;
  } ink_r7_swa_lds __attribute__((aligned(16)));
  static LDS u32 ink_r7_swa_last;
  _Static_assert(sizeof(ink_r7_swa_lds) == 12480u, "ink_r7_swa_lds must be 12,480 B");
#define ink_r7_swa_m (ink_r7_swa_lds.red.m)
#define ink_r7_swa_l (ink_r7_swa_lds.red.l)
#define ink_r7_swa_o (ink_r7_swa_lds.red.o)
#define ink_r7_swa_q (ink_r7_swa_lds.qr.q)
#define ink_r7_swa_r (ink_r7_swa_lds.qr.r)
#endif

  const u32 t0 = __builtin_amdgcn_readfirstlane(tokid() * INK_R7_AV_QT);
  if (t0 >= T || win == 0u || hd != INK_R7_AV_HD || d_rel != INK_R7_AV_DREL || n_kv == 0u ||
      n_head != INK_R7_AV_GQA * n_kv || kv_len == 0u || nsplit == 0u)
    return;

  const u32 kh = __builtin_amdgcn_readfirstlane(wgid() / nsplit);
  const u32 s = __builtin_amdgcn_readfirstlane(wgid() % nsplit);
  if (kh >= n_kv) return;

  const u32 h0 = __builtin_amdgcn_readfirstlane(kh * INK_R7_AV_GQA);
  const u32 kvd = __builtin_amdgcn_readfirstlane(n_kv * INK_R7_AV_HD);
  const u32 w = __builtin_amdgcn_readfirstlane(tid() >> 5u);
  const u32 l = lane();

  const u32 rem = T - t0;
  const u32 nq = __builtin_amdgcn_readfirstlane(rem < INK_R7_AV_QT ? rem : INK_R7_AV_QT);
  const u32 pos_t0 = __builtin_amdgcn_readfirstlane(pos0 + t0);
  const u32 pos_last = __builtin_amdgcn_readfirstlane(pos_t0 + nq - 1u);

  /* Union of SWA windows for tokens t0 .. t0 + nq - 1: [w_union, pos_last + 1). */
  const u32 w_union = __builtin_amdgcn_readfirstlane((pos_t0 + 1u > win) ? (pos_t0 + 1u - win) : 0u);
  const u32 w_last = __builtin_amdgcn_readfirstlane((pos_last + 1u > win) ? (pos_last + 1u - win) : 0u);
  const u32 pend = __builtin_amdgcn_readfirstlane(pos_last + 1u);
  const u32 np = __builtin_amdgcn_readfirstlane(pend - w_union);
  const u32 ch = __builtin_amdgcn_readfirstlane((np + nsplit - 1u) / nsplit);
  const u32 off0 = __builtin_amdgcn_readfirstlane(s * ch);
  const u32 p0 = __builtin_amdgcn_readfirstlane((off0 < np) ? (w_union + off0) : pend);
  const u32 p1 = __builtin_amdgcn_readfirstlane((off0 < np && off0 + ch < np) ? (w_union + off0 + ch) : pend);

  if (nsplit > 1u && p0 >= p1) {
#pragma clang loop unroll(disable)
    for (u32 qi = 0u; qi < nq; qi++) {
      const u32 t = t0 + qi;
      for (u32 idx = tid(); idx < INK_R7_AV_GQA * INK_R7_AV_HD; idx += NT) {
        const u32 g = idx >> 7u;
        const u32 e = idx & 127u;
        const u32 h = h0 + g;
        G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R7_AV_PSTR;
        pp[2u + e] = 0.0f;
        if (e == 0u) {
          pp[0] = -1e30f;
          pp[1] = 0.0f;
        }
      }
    }
  } else {
    const u32 extent_lim = __builtin_amdgcn_readfirstlane((win < rel_extent) ? win : rel_extent);
    /* Interior fast path: all keys [p0, p1) are valid for ALL nq tokens AND every distance < extent_lim. */
    const int swa_interior =
        __builtin_amdgcn_readfirstlane((p0 >= w_last) && (p1 <= pos_t0 + 1u) && (pos_last - p0 < extent_lim));
    const float scale = 1.0f / __builtin_sqrtf((float)INK_R7_AV_HD);

    /* Phase 1: Stage Q (nq * 3 * 128 f32) and R (nq * 3 * 16 f32) into LDS. */
    const u32 nq_vecs = __builtin_amdgcn_readfirstlane(nq * (INK_R7_AV_GQA * (INK_R7_AV_HD / 4u)));
    for (u32 idx = tid(); idx < nq_vecs; idx += NT) {
      const u32 qg = idx >> 5u;
      const u32 e4 = idx & 31u;
      const u32 qi = qg / INK_R7_AV_GQA;
      const u32 gi = qg - qi * INK_R7_AV_GQA;
      const G float *qt = (const G float *)((const G u8 *)q + (unsigned long)(t0 + qi) * qs);
      ((LDSP f32x4 *)(ink_r7_swa_q + (unsigned long)qg * INK_R7_AV_HD))[e4] =
          ((const G f32x4 *)(qt + (unsigned long)(h0 + gi) * INK_R7_AV_HD))[e4];
    }
    if (extent_lim > 0u) {
      const u32 idx = tid();
      if (idx < nq * (INK_R7_AV_GQA * INK_R7_AV_DREL)) {
        const u32 qg = idx >> 4u;
        const u32 e = idx & 15u;
        const u32 qi = qg / INK_R7_AV_GQA;
        const u32 gi = qg - qi * INK_R7_AV_GQA;
        const G float *Rt = (const G float *)((const G u8 *)R + (unsigned long)(t0 + qi) * rs);
        ink_r7_swa_r[(unsigned long)qg * INK_R7_AV_DREL + e] = Rt[(unsigned long)(h0 + gi) * INK_R7_AV_DREL + e];
      }
    }
    lbarrier();

    f32x4 acc00 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc01 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc02 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc10 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc11 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc12 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc20 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc21 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc22 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc30 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc31 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc32 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};

    float m00 = -__builtin_inff(), s00 = 0.0f;
    float m01 = -__builtin_inff(), s01 = 0.0f;
    float m02 = -__builtin_inff(), s02 = 0.0f;
    float m10 = -__builtin_inff(), s10 = 0.0f;
    float m11 = -__builtin_inff(), s11 = 0.0f;
    float m12 = -__builtin_inff(), s12 = 0.0f;
    float m20 = -__builtin_inff(), s20 = 0.0f;
    float m21 = -__builtin_inff(), s21 = 0.0f;
    float m22 = -__builtin_inff(), s22 = 0.0f;
    float m30 = -__builtin_inff(), s30 = 0.0f;
    float m31 = -__builtin_inff(), s31 = 0.0f;
    float m32 = -__builtin_inff(), s32 = 0.0f;

    const G kvt *kc_h = kc + (unsigned long)kh * INK_R7_AV_HD;
    const G kvt *vc_h = vc + (unsigned long)kh * INK_R7_AV_HD;

    u32 kp = __builtin_amdgcn_readfirstlane(p0 + w);
    if (kp < p1) {
      u32 kvp = __builtin_amdgcn_readfirstlane((kp < kv_len) ? kp : (kp % kv_len));
      const G kv4 *krow = (const G kv4 *)(kc_h + (unsigned long)kvp * kvd);
      const G kv4 *vrow = (const G kv4 *)(vc_h + (unsigned long)kvp * kvd);
      kv4 kh4 = krow[l];
      kv4 vh4 = vrow[l];

      if (swa_interior) {
        /* Interior SWA split (>85% of splits at pos >= 1024): zero mask branches,
         * pipelined L1-hot rel_bias loads overlapped with QK dot products. */
        for (;;) {
          const u32 kp_nxt = __builtin_amdgcn_readfirstlane(kp + 8u);
          u32 kvp_nxt = kvp;
          kv4 kh4_nxt = kh4;
          kv4 vh4_nxt = vh4;
          if (kp_nxt < p1) {
            u32 kn = kvp + 8u;
            if (kn >= kv_len) kn = (kv_len >= 8u) ? (kn - kv_len) : (kp_nxt % kv_len);
            kvp_nxt = __builtin_amdgcn_readfirstlane(kn);
            const G kv4 *krow_n = (const G kv4 *)(kc_h + (unsigned long)kvp_nxt * kvd);
            const G kv4 *vrow_n = (const G kv4 *)(vc_h + (unsigned long)kvp_nxt * kvd);
            kh4_nxt = krow_n[l];
            vh4_nxt = vrow_n[l];
          }

          /* Pre-issue contiguous L1-hot rel_bias rows for dist0 .. dist0 + nq - 1. */
          const u32 d0 = pos_t0 - kp;
          const int has_rb = (l < INK_R7_AV_DREL);
          const float rb0 = has_rb ? rel_bias[(unsigned long)(d0 + 0u) * INK_R7_AV_DREL + l] : 0.0f;
          const float rb1 = (has_rb && nq > 1u) ? rel_bias[(unsigned long)(d0 + 1u) * INK_R7_AV_DREL + l] : 0.0f;
          const float rb2 = (has_rb && nq > 2u) ? rel_bias[(unsigned long)(d0 + 2u) * INK_R7_AV_DREL + l] : 0.0f;
          const float rb3 = (has_rb && nq > 3u) ? rel_bias[(unsigned long)(d0 + 3u) * INK_R7_AV_DREL + l] : 0.0f;

          const f32x4 kf = __builtin_convertvector(kh4, f32x4);
          float d00, d01, d02;
          float d10 = 0.0f, d11 = 0.0f, d12 = 0.0f;
          float d20 = 0.0f, d21 = 0.0f, d22 = 0.0f;
          float d30 = 0.0f, d31 = 0.0f, d32 = 0.0f;

          INK_R7_SWA_QK3(0u, kf, scale, l, d00, d01, d02);
          if (nq > 1u) INK_R7_SWA_QK3(1u, kf, scale, l, d10, d11, d12);
          if (nq > 2u) INK_R7_SWA_QK3(2u, kf, scale, l, d20, d21, d22);
          if (nq > 3u) INK_R7_SWA_QK3(3u, kf, scale, l, d30, d31, d32);

          INK_R7_SWA_ADD_RB3(0u, rb0, l, d00, d01, d02);
          if (nq > 1u) INK_R7_SWA_ADD_RB3(1u, rb1, l, d10, d11, d12);
          if (nq > 2u) INK_R7_SWA_ADD_RB3(2u, rb2, l, d20, d21, d22);
          if (nq > 3u) INK_R7_SWA_ADD_RB3(3u, rb3, l, d30, d31, d32);

          const f32x4 vf = __builtin_convertvector(vh4, f32x4);
          INK_R7_SWA_UPD(m00, s00, acc00, d00, vf);
          INK_R7_SWA_UPD(m01, s01, acc01, d01, vf);
          INK_R7_SWA_UPD(m02, s02, acc02, d02, vf);
          if (nq > 1u) {
            INK_R7_SWA_UPD(m10, s10, acc10, d10, vf);
            INK_R7_SWA_UPD(m11, s11, acc11, d11, vf);
            INK_R7_SWA_UPD(m12, s12, acc12, d12, vf);
          }
          if (nq > 2u) {
            INK_R7_SWA_UPD(m20, s20, acc20, d20, vf);
            INK_R7_SWA_UPD(m21, s21, acc21, d21, vf);
            INK_R7_SWA_UPD(m22, s22, acc22, d22, vf);
          }
          if (nq > 3u) {
            INK_R7_SWA_UPD(m30, s30, acc30, d30, vf);
            INK_R7_SWA_UPD(m31, s31, acc31, d31, vf);
            INK_R7_SWA_UPD(m32, s32, acc32, d32, vf);
          }

          if (kp_nxt >= p1) break;
          kp = kp_nxt;
          kvp = kvp_nxt;
          kh4 = kh4_nxt;
          vh4 = vh4_nxt;
        }
      } else {
        /* Left-edge (SWA start) & right-edge (causal end) boundary splits. */
        const u32 pos_q0 = pos_t0;
        const u32 pos_q1 = __builtin_amdgcn_readfirstlane(pos_t0 + 1u);
        const u32 pos_q2 = __builtin_amdgcn_readfirstlane(pos_t0 + 2u);
        const u32 pos_q3 = __builtin_amdgcn_readfirstlane(pos_t0 + 3u);

        for (;;) {
          const u32 kp_nxt = __builtin_amdgcn_readfirstlane(kp + 8u);
          u32 kvp_nxt = kvp;
          kv4 kh4_nxt = kh4;
          kv4 vh4_nxt = vh4;
          if (kp_nxt < p1) {
            u32 kn = kvp + 8u;
            if (kn >= kv_len) kn = (kv_len >= 8u) ? (kn - kv_len) : (kp_nxt % kv_len);
            kvp_nxt = __builtin_amdgcn_readfirstlane(kn);
            const G kv4 *krow_n = (const G kv4 *)(kc_h + (unsigned long)kvp_nxt * kvd);
            const G kv4 *vrow_n = (const G kv4 *)(vc_h + (unsigned long)kvp_nxt * kvd);
            kh4_nxt = krow_n[l];
            vh4_nxt = vrow_n[l];
          }

          const f32x4 kf = __builtin_convertvector(kh4, f32x4);
          const f32x4 vf = __builtin_convertvector(vh4, f32x4);

          INK_R7_SWA_EDGE_TOK(0u, kf, vf, scale, l, kp, pos_q0, win, extent_lim,
                              m00, s00, acc00, m01, s01, acc01, m02, s02, acc02);
          if (nq > 1u) {
            INK_R7_SWA_EDGE_TOK(1u, kf, vf, scale, l, kp, pos_q1, win, extent_lim,
                                m10, s10, acc10, m11, s11, acc11, m12, s12, acc12);
          }
          if (nq > 2u) {
            INK_R7_SWA_EDGE_TOK(2u, kf, vf, scale, l, kp, pos_q2, win, extent_lim,
                                m20, s20, acc20, m21, s21, acc21, m22, s22, acc22);
          }
          if (nq > 3u) {
            INK_R7_SWA_EDGE_TOK(3u, kf, vf, scale, l, kp, pos_q3, win, extent_lim,
                                m30, s30, acc30, m31, s31, acc31, m32, s32, acc32);
          }

          if (kp_nxt >= p1) break;
          kp = kp_nxt;
          kvp = kvp_nxt;
          kh4 = kh4_nxt;
          vh4 = vh4_nxt;
        }
      }
    }

    lbarrier();

    /* Phase 3: Sequentially reduce qi = 0..nq-1 across the 8 waves. */
#pragma clang loop unroll(disable)
    for (u32 qi = 0u; qi < nq; qi++) {
      if (qi > 0u) lbarrier();
      float rm0, rm1, rm2, rl0, rl1, rl2;
      f32x4 ra0, ra1, ra2;
      switch (qi) {
        case 0u:
          rm0 = m00; rm1 = m01; rm2 = m02;
          rl0 = s00; rl1 = s01; rl2 = s02;
          ra0 = acc00; ra1 = acc01; ra2 = acc02;
          break;
        case 1u:
          rm0 = m10; rm1 = m11; rm2 = m12;
          rl0 = s10; rl1 = s11; rl2 = s12;
          ra0 = acc10; ra1 = acc11; ra2 = acc12;
          break;
        case 2u:
          rm0 = m20; rm1 = m21; rm2 = m22;
          rl0 = s20; rl1 = s21; rl2 = s22;
          ra0 = acc20; ra1 = acc21; ra2 = acc22;
          break;
        default:
          rm0 = m30; rm1 = m31; rm2 = m32;
          rl0 = s30; rl1 = s31; rl2 = s32;
          ra0 = acc30; ra1 = acc31; ra2 = acc32;
          break;
      }

      if (l == 0u) {
        ink_r7_swa_m[0][w] = rm0; ink_r7_swa_l[0][w] = rl0;
        ink_r7_swa_m[1][w] = rm1; ink_r7_swa_l[1][w] = rl1;
        ink_r7_swa_m[2][w] = rm2; ink_r7_swa_l[2][w] = rl2;
      }
      ((LDSP f32x4 *)&ink_r7_swa_o[0][w][0])[l] = ra0;
      ((LDSP f32x4 *)&ink_r7_swa_o[1][w][0])[l] = ra1;
      ((LDSP f32x4 *)&ink_r7_swa_o[2][w][0])[l] = ra2;
      lbarrier();

      const u32 t = t0 + qi;
      if (nsplit == 1u) {
        const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
        G float *ot = (G float *)((G u8 *)out + (unsigned long)t * os);
        for (u32 idx = tid(); idx < INK_R7_AV_GQA * INK_R7_AV_HD; idx += NT) {
          const u32 g = idx >> 7u;
          const u32 e = idx & 127u;
          const u32 h = h0 + g;
          const float sv = st[h];
          float M = sv;
          for (u32 x = 0u; x < 8u; x++)
            if (ink_r7_swa_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r7_swa_m[g][x]);
          float L = __builtin_expf(sv - M);
          float o = 0.0f;
          for (u32 x = 0u; x < 8u; x++) {
            const float lx = ink_r7_swa_l[g][x];
            if (lx > 0.0f) {
              const float wx = __builtin_expf(ink_r7_swa_m[g][x] - M);
              L += lx * wx;
              o += ink_r7_swa_o[g][x][e] * wx;
            }
          }
          ot[(unsigned long)h * INK_R7_AV_HD + e] = o / L;
        }
      } else {
        for (u32 idx = tid(); idx < INK_R7_AV_GQA * INK_R7_AV_HD; idx += NT) {
          const u32 g = idx >> 7u;
          const u32 e = idx & 127u;
          const u32 h = h0 + g;
          float M = -__builtin_inff();
          for (u32 x = 0u; x < 8u; x++)
            if (ink_r7_swa_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r7_swa_m[g][x]);
          G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R7_AV_PSTR;
          float o = 0.0f, L = 0.0f;
          for (u32 x = 0u; x < 8u; x++) {
            const float lx = ink_r7_swa_l[g][x];
            if (lx > 0.0f) {
              const float wx = __builtin_expf(ink_r7_swa_m[g][x] - M);
              o += ink_r7_swa_o[g][x][e] * wx;
              if (e == 0u) L += lx * wx;
            }
          }
          pp[2u + e] = o;
          if (e == 0u) {
            pp[0] = (L > 0.0f) ? M : -1e30f;
            pp[1] = L;
          }
        }
      }
    }

    if (nsplit == 1u) return;
  }

  /* Phase 4: Split-K arrival & last-WG 3-pass merge. */
  const u32 cidx = __builtin_amdgcn_readfirstlane(t0 * n_kv + kh);
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r7_swa_last =
        (__scoped_atomic_fetch_add(&count[cidx], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u);
  }
  barrier();
  if (!ink_r7_swa_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const int wt_in_lds = (nsplit <= INK_R7_AV_HD);
#pragma clang loop unroll(disable)
  for (u32 qi = 0u; qi < nq; qi++) {
    if (qi > 0u) lbarrier();
    const u32 t = t0 + qi;
    const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
    G float *ot = (G float *)((G u8 *)out + (unsigned long)t * os);

    G float *ph0 = part + (unsigned long)((t * n_head + (h0 + 0u)) * nsplit) * INK_R7_AV_PSTR;
    G float *ph1 = part + (unsigned long)((t * n_head + (h0 + 1u)) * nsplit) * INK_R7_AV_PSTR;
    G float *ph2 = part + (unsigned long)((t * n_head + (h0 + 2u)) * nsplit) * INK_R7_AV_PSTR;

    float mv0 = (tid() == 0u) ? st[h0 + 0u] : -__builtin_inff();
    float mv1 = (tid() == 0u) ? st[h0 + 1u] : -__builtin_inff();
    float mv2 = (tid() == 0u) ? st[h0 + 2u] : -__builtin_inff();
    for (u32 sp = tid(); sp < nsplit; sp += NT) {
      const unsigned long off = (unsigned long)sp * INK_R7_AV_PSTR;
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
      ink_r7_swa_m[0][w] = mv0;
      ink_r7_swa_m[1][w] = mv1;
      ink_r7_swa_m[2][w] = mv2;
    }
    lbarrier();

    float M2[INK_R7_AV_GQA];
    for (u32 g = 0u; g < INK_R7_AV_GQA; g++) {
      float mx = ink_r7_swa_m[g][0];
      for (u32 x = 1u; x < 8u; x++) mx = __builtin_fmaxf(mx, ink_r7_swa_m[g][x]);
      M2[g] = mx;
    }

    float lv0 = 0.0f, lv1 = 0.0f, lv2 = 0.0f;
    for (u32 sp = tid(); sp < nsplit; sp += NT) {
      const unsigned long off = (unsigned long)sp * INK_R7_AV_PSTR;
      const float lp0 = ph0[off + 1u];
      const float wt0 = (lp0 > 0.0f) ? __builtin_expf(ph0[off] - M2[0]) : 0.0f;
      if (wt_in_lds) ink_r7_swa_o[0][0][sp] = wt0;
      else ph0[off] = wt0;
      lv0 += lp0 * wt0;

      const float lp1 = ph1[off + 1u];
      const float wt1 = (lp1 > 0.0f) ? __builtin_expf(ph1[off] - M2[1]) : 0.0f;
      if (wt_in_lds) ink_r7_swa_o[1][0][sp] = wt1;
      else ph1[off] = wt1;
      lv1 += lp1 * wt1;

      const float lp2 = ph2[off + 1u];
      const float wt2 = (lp2 > 0.0f) ? __builtin_expf(ph2[off] - M2[2]) : 0.0f;
      if (wt_in_lds) ink_r7_swa_o[2][0][sp] = wt2;
      else ph2[off] = wt2;
      lv2 += lp2 * wt2;
    }
    lv0 = wave_tree_f(lv0);
    lv1 = wave_tree_f(lv1);
    lv2 = wave_tree_f(lv2);
    if (l == 0u) {
      ink_r7_swa_l[0][w] = lv0;
      ink_r7_swa_l[1][w] = lv1;
      ink_r7_swa_l[2][w] = lv2;
    }
    if (wt_in_lds) lbarrier();
    else barrier();

    float L2[INK_R7_AV_GQA];
    for (u32 g = 0u; g < INK_R7_AV_GQA; g++) {
      float ls = 0.0f;
      for (u32 x = 0u; x < 8u; x++) ls += ink_r7_swa_l[g][x];
      L2[g] = ls + __builtin_expf(st[h0 + g] - M2[g]);
    }

    for (u32 idx = tid(); idx < INK_R7_AV_GQA * INK_R7_AV_HD; idx += NT) {
      const u32 g = idx >> 7u;
      const u32 e = idx & 127u;
      const u32 h = h0 + g;
      const G float *ph = part + (unsigned long)((t * n_head + h) * nsplit) * INK_R7_AV_PSTR;
      float o = 0.0f;
      if (wt_in_lds) {
        for (u32 sp = 0u; sp < nsplit; sp++)
          o += ph[(unsigned long)sp * INK_R7_AV_PSTR + 2u + e] * ink_r7_swa_o[g][0][sp];
      } else {
        for (u32 sp = 0u; sp < nsplit; sp++)
          o += ph[(unsigned long)sp * INK_R7_AV_PSTR + 2u + e] * ph[(unsigned long)sp * INK_R7_AV_PSTR];
      }
      ot[(unsigned long)h * INK_R7_AV_HD + e] = o / L2[g];
    }
  }

  barrier();
  if (tid() == 0u) count[cidx] = 0u;
}

/* ============================================================================
 * Kernel 2: ie_inkling_rel_attn_ver_glob_wmma_r7
 * Single-pass T<=8 (up to 24 queries = 2 WMMA N-tiles) FP16 WMMA Global Verify
 * Attention: reads 16K-32K Global KV ONCE across all T<=8 tokens (2x less KV
 * DRAM traffic than qtile_r6 at T=5..8), <= 128 VGPR (3 WGs/CU), 15,360 B LDS.
 * ========================================================================== */

#ifndef IE_MEGA

typedef union {
  struct {
    f16 q[32u * 128u] __attribute__((aligned(16)));      /*  8,192 B: 2x16 queries x 128 f16 (XOR-swizzled) */
    f16 r[32u * 16u] __attribute__((aligned(16)));       /*  1,024 B: 2x16 queries x 16 f16 (near-field R) */
    f16 p_tile[2u][2u][256u] __attribute__((aligned(16))); /* 2,048 B: double-buffered 2-pg 16x16 f16 P */
    union {
      u32 v_sc[8u][128u] __attribute__((aligned(16)));   /*  4,096 B: 8 waves x 512 B 0-conflict V^T / rb_tmp */
      struct {
        float mx[2u][128u];                              /*  1,024 B: 2 pg x 8 waves x 16 queries chunk max */
        float ls[2u][128u];                              /*  1,024 B: 2 pg x 8 waves x 16 queries chunk sum */
      } red;
    } scratch;                                           /*  4,096 B */
  } kv;                                                  /* 15,360 B */
  struct {
    float o_stg[8u][128u] __attribute__((aligned(16)));  /*  4,096 B: Phase-4 coalesced f32x4 staging */
    float m[32u];                                        /*    128 B: per-query split M (natural log) */
    float l[32u];                                        /*    128 B: per-query split L */
    float wt[8u][128u] __attribute__((aligned(16)));     /*  4,096 B: last-WG per-wave split merge weights */
    u32   is_last;                                       /*      4 B */
  } ph4;                                                 /*  8,452 B <= 15,360 B */
} ink_r7_gw_lds_t;

static LDS ink_r7_gw_lds_t ink_r7_gw_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r7_gw_lds_t) == 15360u, "ink_r7_gw_lds_t must be 15,360 B (<= 16 KiB)");

static inline __attribute__((always_inline)) u32 ink_r7_gw_q_off(u32 row, u32 c8) {
  return row * 128u + 8u * (c8 ^ (row & 15u));
}

static inline __attribute__((always_inline)) u32 ink_r7_gw_p_off(u32 rl, u32 hl) {
  return rl * 16u + 8u * (hl ^ ((rl >> 2u) & 1u));
}

static inline __attribute__((always_inline)) float ink_r7_gw_xrow(float v) {
  return I2F(xrow_i(F2I(v)));
}

/* 0-bank-conflict in-wave 16x16 f16 transpose via 1x DPP(0xB1) + perm + 512 B v_sc[w].
 * Input vr: lane (hl, rl) holds key rl (0..15), dims 8*hl..8*hl+7.
 * Output  : lane (hl, rl) holds dim rl (0..15), keys 8*hl..8*hl+7. */
static inline __attribute__((always_inline)) h8 ink_r7_gw_v_transpose(
    const u32x4 vr, LDSP u32 *v_sc_w, const u32 hl, const u32 rl) {
  const u32 sel = (rl & 1u) ? 0x03020706u : 0x05040100u;
  const u32 kh = rl >> 3u;
  const u32 k4 = (rl >> 1u) & 3u;
  const u32 w_arr[4] = {vr.x, vr.y, vr.z, vr.w};
#pragma unroll
  for (u32 q = 0u; q < 4u; q++) {
    const u32 x = w_arr[q];
    const u32 y = (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0xB1, 0xF, 0xF, true);
    const u32 t = __builtin_amdgcn_perm(y, x, sel);
    const u32 d = 8u * hl + 2u * q + (rl & 1u);
    const u32 s4 = ((d & 7u) ^ ((d >> 3u) << 1u) ^ (kh << 2u)) + ((d >> 3u) << 3u) + (kh << 4u);
    v_sc_w[s4 * 4u + k4] = t;
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  const u32 s4_r = ((rl & 7u) ^ ((rl >> 3u) << 1u) ^ (hl << 2u)) + ((rl >> 3u) << 3u) + (hl << 4u);
  return *(const LDSP h8 *)(v_sc_w + s4_r * 4u);
}

/* Apply a 16-distance rel_bias WMMA tile into sv using wave-private 512 B scratch (0 extra LDS). */
static inline __attribute__((always_inline)) void ink_r7_gw_apply_rb16(
    const v8f rb, const u32 base_off, const u32 hl, const u32 kb, const u32 pos_q,
    const u32 rel_extent, const u32 d_base, LDSP float *sc_row, v8f *sv) {
  if (hl == 0u) {
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) sc_row[v] = rb[v];
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 v = 0u; v < 8u; v++) {
    const u32 kp = kb + 8u * hl + v;
    if (kp <= pos_q) {
      const u32 dist = pos_q - kp;
      if (dist < rel_extent) {
        const u32 rel = (dist - d_base) - base_off;
        if (rel < 8u) (*sv)[v] += sc_row[rel];
      }
    }
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  if (hl == 1u) {
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) sc_row[v] = rb[v];
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 v = 0u; v < 8u; v++) {
    const u32 kp = kb + 8u * hl + v;
    if (kp <= pos_q) {
      const u32 dist = pos_q - kp;
      if (dist < rel_extent) {
        const u32 rel = (dist - d_base) - (base_off + 8u);
        if (rel < 8u) (*sv)[v] += sc_row[rel];
      }
    }
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

/* Wave-level split-K merge of nsplit partials for one (t, h) pair (1 wave = 32 lanes x 4 floats = 128). */
static inline __attribute__((always_inline)) void ink_r7_gw_merge_head(
    const G float *ph, const float sink_val, const u32 nsplit, const u32 l,
    LDSP float *wt_w, G float *dst_h) {
  float mv = (l == 0u) ? sink_val : -__builtin_inff();
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const unsigned long off = (unsigned long)sp * INK_R7_AV_PSTR;
    if (ph[off + 1u] > 0.0f) mv = __builtin_fmaxf(mv, ph[off]);
  }
#pragma unroll
  for (u32 mask = 16u; mask >= 1u; mask >>= 1u) mv = __builtin_fmaxf(mv, shfl_xor_f(mv, mask));
  const float M2 = ink_r7_av_sgpr_f(mv);

  float lv = 0.0f;
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const unsigned long off = (unsigned long)sp * INK_R7_AV_PSTR;
    const float lp = ph[off + 1u];
    const float wt = (lp > 0.0f) ? __builtin_expf(ph[off] - M2) : 0.0f;
    if (sp < 128u) wt_w[sp] = wt;
    lv += lp * wt;
  }
  const float L2 = ink_r7_av_sgpr_f(wave_sum_all(lv) + __builtin_expf(sink_val - M2));
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");

  f32x4 acc = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  for (u32 sp = 0u; sp < nsplit; sp++) {
    const float wt = (sp < 128u) ? wt_w[sp]
                                 : ((ph[(unsigned long)sp * INK_R7_AV_PSTR + 1u] > 0.0f)
                                        ? __builtin_expf(ph[(unsigned long)sp * INK_R7_AV_PSTR] - M2)
                                        : 0.0f);
    if (wt > 0.0f) {
      const G float *po = ph + (unsigned long)sp * INK_R7_AV_PSTR + 2u + 4u * l;
      const f32x4 ov = {po[0], po[1], po[2], po[3]};
      acc += ov * wt;
    }
  }
  const float inv = 1.0f / L2;
  ((G f32x4 *)dst_h)[l] = acc * inv;
}

KERNEL ie_inkling_rel_attn_ver_glob_wmma_r7(const G float *q, const G kvt *kc, const G kvt *vc,
                                            const G float *R, const G float *rel_bias, const G float *sink,
                                            G float *part, G float *out, u32 pos0, u32 T, u32 kv_len, u32 hd,
                                            u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
                                            u32 qs, u32 rs, u32 ss, u32 os, G u32 *count) {
  if (T == 0u || T > 8u || win != 0u || hd != INK_R7_AV_HD || d_rel != INK_R7_AV_DREL || n_kv == 0u ||
      n_head != INK_R7_AV_GQA * n_kv || kv_len == 0u || nsplit == 0u)
    return;

  const u32 kh = __builtin_amdgcn_readfirstlane(wgid() / nsplit);
  const u32 s = __builtin_amdgcn_readfirstlane(wgid() % nsplit);
  if (kh >= n_kv) return;

  const u32 h0 = __builtin_amdgcn_readfirstlane(kh * INK_R7_AV_GQA);
  const u32 kvd = __builtin_amdgcn_readfirstlane(n_kv * INK_R7_AV_HD);
  const u32 w = __builtin_amdgcn_readfirstlane(tid() >> 5u);
  const u32 l = lane(), hl = l >> 4u, rl = l & 15u;

  const u32 nq_tot = __builtin_amdgcn_readfirstlane(T * INK_R7_AV_GQA);
  const u32 npg = __builtin_amdgcn_readfirstlane(nq_tot > 16u ? 2u : 1u);
  const u32 np = __builtin_amdgcn_readfirstlane(pos0 + T);
  const u32 ch = __builtin_amdgcn_readfirstlane((np + nsplit - 1u) / nsplit);
  const u32 off0 = __builtin_amdgcn_readfirstlane(s * ch);
  const u32 p0 = __builtin_amdgcn_readfirstlane((off0 < np) ? off0 : np);
  const u32 p1 = __builtin_amdgcn_readfirstlane((off0 < np && off0 + ch < np) ? (off0 + ch) : np);

  if (nsplit > 1u && p0 >= p1) {
    /* Empty tail split: emit sentinel partials for all nq_tot queries. */
    for (u32 qi = w; qi < nq_tot; qi += 8u) {
      const u32 t = qi / INK_R7_AV_GQA;
      const u32 h = h0 + (qi - t * INK_R7_AV_GQA);
      G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R7_AV_PSTR;
      if (l == 0u) {
        pp[0] = -1e30f;
        pp[1] = 0.0f;
      }
      pp[2u + 4u * l + 0u] = 0.0f;
      pp[2u + 4u * l + 1u] = 0.0f;
      pp[2u + 4u * l + 2u] = 0.0f;
      pp[2u + 4u * l + 3u] = 0.0f;
    }
  } else {
    const int far_split = __builtin_amdgcn_readfirstlane((p1 <= pos0 + 1u) && ((pos0 + 1u - p1) >= rel_extent));

    /* Phase 1: Cooperatively stage all nq_tot <= 24 queries (f16, scaled by log2(e)/sqrt(128))
     * into XOR-swizzled q_lds (8 KiB, 0 bank conflicts). Pad rows nq_tot..31 with 0. */
    const u32 q_rows = npg * 16u;
    for (u32 idx = tid(); idx < q_rows * 16u; idx += NT) {
      const u32 qr = idx >> 4u;
      const u32 c8 = idx & 15u;
      h8 qh = (h8)(0.0f);
      if (qr < nq_tot) {
        const u32 tj = qr / INK_R7_AV_GQA;
        const u32 gi = qr - tj * INK_R7_AV_GQA;
        const G f32x4 *qp = (const G f32x4 *)((const G float *)((const G u8 *)q + (unsigned long)tj * qs) +
                                              (unsigned long)(h0 + gi) * INK_R7_AV_HD);
        const f32x4 v0 = qp[2u * c8 + 0u];
        const f32x4 v1 = qp[2u * c8 + 1u];
        qh = (h8){(f16)(v0.x * INK_R7_AV_QSC), (f16)(v0.y * INK_R7_AV_QSC),
                  (f16)(v0.z * INK_R7_AV_QSC), (f16)(v0.w * INK_R7_AV_QSC),
                  (f16)(v1.x * INK_R7_AV_QSC), (f16)(v1.y * INK_R7_AV_QSC),
                  (f16)(v1.z * INK_R7_AV_QSC), (f16)(v1.w * INK_R7_AV_QSC)};
      }
      *(LDSP h8 *)(ink_r7_gw_lds.kv.q + ink_r7_gw_q_off(qr, c8)) = qh;
    }
    if (!far_split && rel_extent > 0u) {
      if (tid() < q_rows * 2u) {
        const u32 qr = tid() >> 1u;
        const u32 h8i = tid() & 1u;
        h8 rh = (h8)(0.0f);
        if (qr < nq_tot) {
          const u32 tj = qr / INK_R7_AV_GQA;
          const u32 gi = qr - tj * INK_R7_AV_GQA;
          const G f32x4 *rp = (const G f32x4 *)((const G float *)((const G u8 *)R + (unsigned long)tj * rs) +
                                                (unsigned long)(h0 + gi) * INK_R7_AV_DREL);
          const f32x4 v0 = rp[2u * h8i + 0u];
          const f32x4 v1 = rp[2u * h8i + 1u];
          rh = (h8){(f16)(v0.x * INK_R7_AV_LOG2E), (f16)(v0.y * INK_R7_AV_LOG2E),
                    (f16)(v0.z * INK_R7_AV_LOG2E), (f16)(v0.w * INK_R7_AV_LOG2E),
                    (f16)(v1.x * INK_R7_AV_LOG2E), (f16)(v1.y * INK_R7_AV_LOG2E),
                    (f16)(v1.z * INK_R7_AV_LOG2E), (f16)(v1.w * INK_R7_AV_LOG2E)};
        }
        *(LDSP h8 *)(ink_r7_gw_lds.kv.r + qr * 16u + 8u * h8i) = rh;
      }
    }
    lbarrier();

    const G kvt *kc_h = kc + (unsigned long)kh * INK_R7_AV_HD;
    const G kvt *vc_h = vc + (unsigned long)kh * INK_R7_AV_HD;
    const int no_wrap = __builtin_amdgcn_readfirstlane(p1 <= kv_len);

    /* Per-wave persistent output accumulators (wave w owns dims 16*w .. 16*w + 15 for all <= 24 queries). */
    const v8f zero = (v8f)(0.0f);
    v8f acc0 = zero, acc1 = zero;
    float m0 = -__builtin_inff(), ls0 = 0.0f;
    float m1 = -__builtin_inff(), ls1 = 0.0f;

    const u32 p_off = ink_r7_gw_p_off(rl, hl);
    const u32 pos_q0 = pos0 + (rl / INK_R7_AV_GQA);
    const u32 pos_q1 = pos0 + ((16u + rl) / INK_R7_AV_GQA);

    /* Phase 2+3: Process split [p0, p1) in 128-key chunks (8 waves x 16 keys). */
#pragma clang loop unroll(disable)
    for (u32 c0 = p0; c0 < p1; c0 += 128u) {
      const u32 kb = __builtin_amdgcn_readfirstlane(c0 + 16u * w);
      const u32 chunk_end = __builtin_amdgcn_readfirstlane((c0 + 128u < p1) ? (c0 + 128u) : p1);
      const int far_chunk =
          __builtin_amdgcn_readfirstlane(far_split || ((chunk_end <= pos0 + 1u) && ((pos0 + 1u - chunk_end) >= rel_extent)));

      v8f sp0 = (v8f)(-__builtin_inff());
      v8f sp1 = (v8f)(-__builtin_inff());

      if (kb < chunk_end) {
        const u32 kp_row = kb + rl;
        const int row_ok = (kp_row < chunk_end);
        const u32 kv_row = no_wrap ? (row_ok ? kp_row : (chunk_end - 1u))
                                   : ((row_ok ? kp_row : (chunk_end - 1u)) % kv_len);
        const G f16 *k_base = kc_h + (unsigned long)kv_row * kvd + 8u * hl;

        /* Stream K across hd = 128 (8 x 16 f16) with 2-deep K prefetch, shared across both pg. */
        sp0 = zero;
        sp1 = zero;
        h8 kk0 = row_ok ? *(const G h8 *)(k_base + 0u) : (h8)(0.0f);
        h8 kk1 = row_ok ? *(const G h8 *)(k_base + 16u) : (h8)(0.0f);
#pragma unroll
        for (u32 ks = 0u; ks < 8u; ks += 2u) {
          const u32 c_a = 2u * ks + hl;
          const u32 c_b = 2u * (ks + 1u) + hl;
          const h8 q0_a = *(const LDSP h8 *)(ink_r7_gw_lds.kv.q + ink_r7_gw_q_off(rl, c_a));
          const h8 q0_b = *(const LDSP h8 *)(ink_r7_gw_lds.kv.q + ink_r7_gw_q_off(rl, c_b));
          sp0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk0, q0_a, sp0);
          sp0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk1, q0_b, sp0);
          if (npg > 1u) {
            const h8 q1_a = *(const LDSP h8 *)(ink_r7_gw_lds.kv.q + ink_r7_gw_q_off(16u + rl, c_a));
            const h8 q1_b = *(const LDSP h8 *)(ink_r7_gw_lds.kv.q + ink_r7_gw_q_off(16u + rl, c_b));
            sp1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk0, q1_a, sp1);
            sp1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk1, q1_b, sp1);
          }
          if (ks + 2u < 8u) {
            kk0 = row_ok ? *(const G h8 *)(k_base + 16u * (ks + 2u)) : (h8)(0.0f);
            kk1 = row_ok ? *(const G h8 *)(k_base + 16u * (ks + 3u)) : (h8)(0.0f);
          }
          __builtin_amdgcn_sched_barrier(0);
        }

        if (!far_chunk) {
          /* Near-field rel_bias via WMMA + wave-private 512 B scratch (0 extra LDS). */
          const int d_min_s = (int)pos0 - (int)(kb + 15u);
          const u32 d_base = d_min_s > 0 ? (u32)d_min_s : 0u;
          if (d_base < rel_extent) {
            LDSP float *sc_row = ((LDSP float *)ink_r7_gw_lds.kv.scratch.v_sc[w]) + rl * 8u;
            const h8 ra0 = *(const LDSP h8 *)(ink_r7_gw_lds.kv.r + rl * 16u + 8u * hl);
            const u32 d0 = d_base + rl;
            h8 b0 = (h8)(0.0f);
            if (d0 < rel_extent) {
              const G f32x4 *rp = (const G f32x4 *)(rel_bias + (unsigned long)d0 * 16ul + 8ul * hl);
              const f32x4 v0 = rp[0], v1 = rp[1];
              b0 = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
            }
            const v8f rb0_0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0, ra0, zero);
            ink_r7_gw_apply_rb16(rb0_0, 0u, hl, kb, pos_q0, rel_extent, d_base, sc_row, &sp0);
            if (npg > 1u) {
              const h8 ra1 = *(const LDSP h8 *)(ink_r7_gw_lds.kv.r + (16u + rl) * 16u + 8u * hl);
              const v8f rb0_1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0, ra1, zero);
              ink_r7_gw_apply_rb16(rb0_1, 0u, hl, kb, pos_q1, rel_extent, d_base, sc_row, &sp1);
            }
            if (d_base + 16u < rel_extent && (pos0 + T - 1u) >= kb + d_base + 16u) {
              const u32 d1 = d_base + 16u + rl;
              h8 b1 = (h8)(0.0f);
              if (d1 < rel_extent) {
                const G f32x4 *rp = (const G f32x4 *)(rel_bias + (unsigned long)d1 * 16ul + 8ul * hl);
                const f32x4 v0 = rp[0], v1 = rp[1];
                b1 = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
              }
              const v8f rb1_0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1, ra0, zero);
              ink_r7_gw_apply_rb16(rb1_0, 16u, hl, kb, pos_q0, rel_extent, d_base, sc_row, &sp0);
              if (npg > 1u) {
                const h8 ra1 = *(const LDSP h8 *)(ink_r7_gw_lds.kv.r + (16u + rl) * 16u + 8u * hl);
                const v8f rb1_1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1, ra1, zero);
                ink_r7_gw_apply_rb16(rb1_1, 16u, hl, kb, pos_q1, rel_extent, d_base, sc_row, &sp1);
              }
            }
          }
          /* Causal + split-end mask. */
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) {
            const u32 kp = kb + 8u * hl + v;
            if (kp >= chunk_end || kp > pos_q0) sp0[v] = -__builtin_inff();
            if (npg > 1u && (kp >= chunk_end || kp > pos_q1)) sp1[v] = -__builtin_inff();
          }
        } else if (kb + 16u > chunk_end) {
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) {
            if (kb + 8u * hl + v >= chunk_end) {
              sp0[v] = -__builtin_inff();
              if (npg > 1u) sp1[v] = -__builtin_inff();
            }
          }
        }
      }

      /* Prefetch V tile 0 (keys c0..c0+15, dims 16*w..16*w+15) ahead of softmax barriers. */
      u32x4 vr0 = (u32x4)(0u);
      {
        const u32 vp = c0 + rl;
        if (vp < chunk_end) {
          const u32 kv_v = no_wrap ? vp : (vp % kv_len);
          vr0 = *(const G u32x4 *)(vc_h + (unsigned long)kv_v * kvd + 16u * w + 8u * hl);
        }
      }

      /* Guard scratch union transition: near-field apply_rb16 uses v_sc[w] before red.mx aliases scratch. */
      if (!far_chunk) lbarrier();

      /* 2-barrier online softmax across the 128-key chunk (union red with v_sc). */
      float mx0 = sp0[0];
#pragma unroll
      for (u32 v = 1u; v < 8u; v++) mx0 = __builtin_fmaxf(mx0, sp0[v]);
      mx0 = __builtin_fmaxf(mx0, ink_r7_gw_xrow(mx0));
      if (hl == 0u) ink_r7_gw_lds.kv.scratch.red.mx[0][16u * w + rl] = mx0;

      if (npg > 1u) {
        float mx1 = sp1[0];
#pragma unroll
        for (u32 v = 1u; v < 8u; v++) mx1 = __builtin_fmaxf(mx1, sp1[v]);
        mx1 = __builtin_fmaxf(mx1, ink_r7_gw_xrow(mx1));
        if (hl == 0u) ink_r7_gw_lds.kv.scratch.red.mx[1][16u * w + rl] = mx1;
      }
      lbarrier();

      float mc0 = ink_r7_gw_lds.kv.scratch.red.mx[0][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) mc0 = __builtin_fmaxf(mc0, ink_r7_gw_lds.kv.scratch.red.mx[0][16u * x + rl]);
      const float mn0 = __builtin_fmaxf(m0, mc0);
      const float rs0 = (ls0 > 0.0f && mn0 > -1e30f) ? __builtin_amdgcn_exp2f(m0 - mn0) : 0.0f;
      acc0 *= rs0;
      ls0 *= rs0;
      m0 = mn0;

      h8 ph0 = (h8)(0.0f);
      float lc0 = 0.0f;
      if (mn0 > -1e30f) {
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) {
          const float p = __builtin_amdgcn_exp2f(sp0[v] - mn0);
          lc0 += p;
          ph0[v] = (f16)p;
        }
      }
      lc0 += ink_r7_gw_xrow(lc0);
      if (hl == 0u) ink_r7_gw_lds.kv.scratch.red.ls[0][16u * w + rl] = lc0;

      h8 ph1 = (h8)(0.0f);
      if (npg > 1u) {
        float mc1 = ink_r7_gw_lds.kv.scratch.red.mx[1][rl];
#pragma unroll
        for (u32 x = 1u; x < 8u; x++) mc1 = __builtin_fmaxf(mc1, ink_r7_gw_lds.kv.scratch.red.mx[1][16u * x + rl]);
        const float mn1 = __builtin_fmaxf(m1, mc1);
        const float rs1 = (ls1 > 0.0f && mn1 > -1e30f) ? __builtin_amdgcn_exp2f(m1 - mn1) : 0.0f;
        acc1 *= rs1;
        ls1 *= rs1;
        m1 = mn1;

        float lc1 = 0.0f;
        if (mn1 > -1e30f) {
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) {
            const float p = __builtin_amdgcn_exp2f(sp1[v] - mn1);
            lc1 += p;
            ph1[v] = (f16)p;
          }
        }
        lc1 += ink_r7_gw_xrow(lc1);
        if (hl == 0u) ink_r7_gw_lds.kv.scratch.red.ls[1][16u * w + rl] = lc1;
      }

      /* Wave 0 publishes kt=0 p_tile[0] before the softmax-end lbarrier(). */
      if (w == 0u) {
        *(LDSP h8 *)(ink_r7_gw_lds.kv.p_tile[0][0] + p_off) = ph0;
        if (npg > 1u) *(LDSP h8 *)(ink_r7_gw_lds.kv.p_tile[0][1] + p_off) = ph1;
      }
      lbarrier();

      float sum_l0 = ink_r7_gw_lds.kv.scratch.red.ls[0][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) sum_l0 += ink_r7_gw_lds.kv.scratch.red.ls[0][16u * x + rl];
      ls0 += sum_l0;
      if (npg > 1u) {
        float sum_l1 = ink_r7_gw_lds.kv.scratch.red.ls[1][rl];
#pragma unroll
        for (u32 x = 1u; x < 8u; x++) sum_l1 += ink_r7_gw_lds.kv.scratch.red.ls[1][16u * x + rl];
        ls1 += sum_l1;
      }
      /* Ensure red.ls reads complete before v_sc aliases scratch! */
      lbarrier();

      /* Phase 3: Double-buffered p_tile + 1-pass 0-conflict V^T + WMMA PV over kt = 0..7. */
      LDSP u32 *v_sc_w = ink_r7_gw_lds.kv.scratch.v_sc[w];
#pragma clang loop unroll(disable)
      for (u32 kt = 0u; kt < 8u; kt++) {
        if (c0 + 16u * kt >= chunk_end) break;
        const int has_nxt = (kt + 1u < 8u) && (c0 + 16u * (kt + 1u) < chunk_end);
        u32x4 vr_nxt = (u32x4)(0u);
        if (has_nxt) {
          const u32 vp_n = c0 + 16u * (kt + 1u) + rl;
          if (vp_n < chunk_end) {
            const u32 kv_vn = no_wrap ? vp_n : (vp_n % kv_len);
            vr_nxt = *(const G u32x4 *)(vc_h + (unsigned long)kv_vn * kvd + 16u * w + 8u * hl);
          }
        }
        __builtin_amdgcn_sched_barrier(0);

        const h8 vt_a = ink_r7_gw_v_transpose(vr0, v_sc_w, hl, rl);
        const h8 pt0 = *(const LDSP h8 *)(ink_r7_gw_lds.kv.p_tile[kt & 1u][0] + p_off);
        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt0, acc0);
        if (npg > 1u) {
          const h8 pt1 = *(const LDSP h8 *)(ink_r7_gw_lds.kv.p_tile[kt & 1u][1] + p_off);
          acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt1, acc1);
        }

        if (has_nxt && w == kt + 1u) {
          *(LDSP h8 *)(ink_r7_gw_lds.kv.p_tile[(kt + 1u) & 1u][0] + p_off) = ph0;
          if (npg > 1u) *(LDSP h8 *)(ink_r7_gw_lds.kv.p_tile[(kt + 1u) & 1u][1] + p_off) = ph1;
        }
        lbarrier();
        vr0 = vr_nxt;
      }
    }

    /* Phase 4A: Stage (m, ls) into ph4.m/l, then coalesce 8 queries at a time via o_stg[8][128]. */
    lbarrier();
    if (w == 0u && hl == 0u) {
      ink_r7_gw_lds.ph4.m[rl] = (ls0 > 0.0f) ? (m0 * INK_R7_AV_LN2) : -1e30f;
      ink_r7_gw_lds.ph4.l[rl] = ls0;
      if (npg > 1u) {
        ink_r7_gw_lds.ph4.m[16u + rl] = (ls1 > 0.0f) ? (m1 * INK_R7_AV_LN2) : -1e30f;
        ink_r7_gw_lds.ph4.l[16u + rl] = ls1;
      }
    }
    lbarrier();

    /* Each half-pg (8 queries) transposes wave-split dims (16/wave) -> contiguous 128 floats/query. */
#pragma clang loop unroll(disable)
    for (u32 pg = 0u; pg < npg; pg++) {
      const v8f a_cur = (pg == 0u) ? acc0 : acc1;
#pragma unroll
      for (u32 qb = 0u; qb < 2u; qb++) {
        const u32 q_base = 16u * pg + 8u * qb;
        if (q_base >= nq_tot) break;
        if ((rl >> 3u) == qb) {
          const u32 q_loc = rl & 7u;
          LDSP float *dst_w = ink_r7_gw_lds.ph4.o_stg[q_loc] + 16u * w + 8u * hl;
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) dst_w[v] = a_cur[v];
        }
        lbarrier();

        const u32 qi = q_base + w;
        if (qi < nq_tot) {
          const u32 t = qi / INK_R7_AV_GQA;
          const u32 h = h0 + (qi - t * INK_R7_AV_GQA);
          const f32x4 ov = ((const LDSP f32x4 *)ink_r7_gw_lds.ph4.o_stg[w])[l];
          if (nsplit == 1u) {
            const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
            const float sv = st[h];
            const float mq = ink_r7_gw_lds.ph4.m[qi];
            const float lq = ink_r7_gw_lds.ph4.l[qi];
            const float M = (lq > 0.0f) ? __builtin_fmaxf(sv, mq) : sv;
            const float wx = (lq > 0.0f) ? __builtin_expf(mq - M) : 0.0f;
            const float L = __builtin_expf(sv - M) + lq * wx;
            const float sc_o = wx / L;
            G float *ot = (G float *)((G u8 *)out + (unsigned long)t * os) + (unsigned long)h * INK_R7_AV_HD;
            ((G f32x4 *)ot)[l] = ov * sc_o;
          } else {
            G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R7_AV_PSTR;
            if (l == 0u) {
              pp[0] = ink_r7_gw_lds.ph4.m[qi];
              pp[1] = ink_r7_gw_lds.ph4.l[qi];
            }
            G float *po = pp + 2u + 4u * l;
            po[0] = ov.x; po[1] = ov.y; po[2] = ov.z; po[3] = ov.w;
          }
        }
        lbarrier();
      }
    }

    if (nsplit == 1u) return;
  }

  /* Phase 4B: Single-atomic arrival per KV head (count[kh]) + parallel 8-wave split-K merge. */
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r7_gw_lds.ph4.is_last =
        (__scoped_atomic_fetch_add(&count[kh], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r7_gw_lds.ph4.is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  for (u32 qi = w; qi < nq_tot; qi += 8u) {
    const u32 t = qi / INK_R7_AV_GQA;
    const u32 h = h0 + (qi - t * INK_R7_AV_GQA);
    const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
    G float *ot = (G float *)((G u8 *)out + (unsigned long)t * os) + (unsigned long)h * INK_R7_AV_HD;
    const G float *ph = part + (unsigned long)((t * n_head + h) * nsplit) * INK_R7_AV_PSTR;
    ink_r7_gw_merge_head(ph, st[h], nsplit, l, ink_r7_gw_lds.ph4.wt[w], ot);
  }
  barrier();
  if (tid() == 0u) count[kh] = 0u;
}

#endif /* !IE_MEGA */

#undef ink_r7_swa_m
#undef ink_r7_swa_l
#undef ink_r7_swa_o
#ifdef IE_MEGA
#undef ink_r7_swa_last
#endif
#undef ink_r7_swa_q
#undef ink_r7_swa_r
#undef INK_R7_SWA_EDGE_TOK
#undef INK_R7_SWA_UPD
#undef INK_R7_SWA_ADD_RB3
#undef INK_R7_SWA_QK3
#undef INK_R7_AV_QSC
#undef INK_R7_AV_LN2
#undef INK_R7_AV_LOG2E
#undef INK_R7_AV_PSTR
#undef INK_R7_AV_QT
#undef INK_R7_AV_GQA
#undef INK_R7_AV_DREL
#undef INK_R7_AV_HD
