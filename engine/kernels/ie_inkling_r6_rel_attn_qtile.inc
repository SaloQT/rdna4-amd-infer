/* ============================================================================
 * Inkling-2 Round 6 (Audit Exp #5): Multi-Query Global K/V-Sharing Verify
 * Relative Attention for RX 9070 XT (gfx1201, wave32, NT=256, 0-spill).
 *
 * Kernel:
 *   KERNEL ie_inkling_rel_attn_ver_qtile_r6(
 *     const G float *q, const G kvt *kc, const G kvt *vc,
 *     const G float *R, const G float *rel_bias, const G float *sink,
 *     G float *part, G float *out,
 *     u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
 *     u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
 *     u32 qs, u32 rs, u32 ss, u32 os, G u32 *count)
 *   (Exact same 23-argument signature as ie_inkling_rel_attn_ver_r4.)
 *
 * Grid:
 *   (n_kv * nsplit, (T + 3u) / 4u, 1u) with NT=256 (8 waves of 32).
 *   1 WG processes QT=4 consecutive verify tokens t0..min(t0+3, T-1)
 *   (t0 = tokid() * 4u) x 3 GQA Q heads (up to 12 queries) for 1 KV head
 *   kh = wgid() / nsplit and split s = wgid() % nsplit, reading each global
 *   K and V row ONCE across all 3 * QT queries (4x K/V traffic cut at T=4,8;
 *   3x cut at T=6 DSpark 5-draft verify).
 *
 * Guards (src/hip.c):
 *   !b->kvq8 && !b->kvq4 && win == 0u && pos >= 2048u && T >= 2u && T <= 16u &&
 *   hd == 128u && d_rel == 16u && n_kv > 0u && n_head == 3u * n_kv &&
 *   kv_len > 0u && nsplit >= 1u && (nsplit == 1u || (part && count)) &&
 *   ((qs | os) & 15u) == 0u && ((uintptr_t)q & 15u) == 0u
 *   Keep SWA (win > 0u) and short global contexts (pos < 2048u) on
 *   ie_inkling_rel_attn_ver_r4.
 *
 * Buffer & Counter Contract (identical to ie_inkling_rel_attn_ver_r4):
 *   - part:  >= T * n_head * nsplit * 130 floats (520 B/slot, unused if nsplit==1).
 *            Slot index: ((t * n_head + h) * nsplit + s) * 130 -> {M, L, O[0..127]}.
 *   - count: >= T * n_kv u32 elements, zero-initialized before first use.
 *            Uses 1 counter per token tile: count[t0 * n_kv + kh] (self-reset to 0
 *            by the last WG; counters for t0+1..t0+nq-1 remain untouched at 0).
 *
 * Resource Design (gfx1201 wave32):
 *   - LDS: 12,484 B (~12.2 KiB <= 24 KiB; also aliases mk_ink_lds.rel_attn
 *          with 0 extra bytes under IE_MEGA). Q (6,144 B) + R (768 B) are staged
 *          in LDS during the KV loop, then overlaid by the 12,480 B 8-wave
 *          reduction buffer processed 1 token at a time.
 *   - VGPR: Target <= 128 VGPR (0 spills -> 3 WGs/CU). Running (m, sum) and
 *           wave-uniform QK scores are forced to SGPRs via readfirstlane;
 *           acc[4][3] stays in 48 VGPRs via explicit scalar variables + switch.
 *   - Split specialization: Far-field interior splits (p1 <= pos0+t0+1 and
 *           pos0+t0+1-p1 >= rel_extent; >95% of splits at 16K-32K) execute a
 *           zero-mask, zero-rel_bias fast loop; near-field/tail splits enforce
 *           exact per-query causal mask (kp <= pos_q) and per-query distance
 *           rel_bias (pos_q - kp < rel_extent).
 *
 * Parity Expectation:
 *   Bitwise-identical to ie_inkling_rel_attn_ver_r4 whenever tile split size
 *   ch = ceil((pos0 + t_end + 1) / nsplit) matches per-token ch; f32-tol
 *   (~1e-6) across split-size boundaries and vs ie_rel_attn_pf / ie_rel_attn.
 * ========================================================================== */

_Static_assert(sizeof(kvt) == 2u, "ie_inkling_rel_attn_ver_qtile_r6 requires f16 KV cache");

#define INK_R6_ATTN_HD   128u
#define INK_R6_ATTN_DREL 16u
#define INK_R6_ATTN_GQA  3u
#define INK_R6_ATTN_QT   4u
#define INK_R6_ATTN_PSTR 130u

static inline __attribute__((always_inline)) float ink_r6_attn_sgpr_f(float v) {
  return I2F(__builtin_amdgcn_readfirstlane(F2I(v)));
}

/* 16-lane (half-wave) sum broadcast to all 32 lanes; bitwise identical to
 * ink_r4_attn_halfwave_sum_all / wave_sum_all when lanes 16..31 are 0.0f. */
static inline __attribute__((always_inline)) float ink_r6_attn_halfwave_sum_all(float v) {
  v += I2F(shl_i(F2I(v), 8));
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return I2F(__builtin_amdgcn_readlane(F2I(v), 0));
}

static inline __attribute__((always_inline)) float ink_r6_attn_dot4(f32x4 a, f32x4 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

#define INK_R6_ATTN_QK3(qi, kf_val, sc_val, lane_id, d0_out, d1_out, d2_out)                            \
  do {                                                                                                  \
    const LDSP f32x4 *_r6_qp =                                                                          \
        (const LDSP f32x4 *)(ink_r6_q_base + (unsigned long)(qi) * (INK_R6_ATTN_GQA * INK_R6_ATTN_HD)); \
    const f32x4 _r6_q0 = _r6_qp[0u * 32u + (lane_id)];                                                  \
    const f32x4 _r6_q1 = _r6_qp[1u * 32u + (lane_id)];                                                  \
    const f32x4 _r6_q2 = _r6_qp[2u * 32u + (lane_id)];                                                  \
    (d0_out) = ink_r6_attn_sgpr_f(wave_sum_all(ink_r6_attn_dot4(_r6_q0, (kf_val))) * (sc_val));         \
    (d1_out) = ink_r6_attn_sgpr_f(wave_sum_all(ink_r6_attn_dot4(_r6_q1, (kf_val))) * (sc_val));         \
    (d2_out) = ink_r6_attn_sgpr_f(wave_sum_all(ink_r6_attn_dot4(_r6_q2, (kf_val))) * (sc_val));         \
  } while (0)

#define INK_R6_ATTN_UPD(m_var, s_var, a_var, d_val, vf_val)                           \
  do {                                                                                \
    const float _r6_mn = __builtin_fmaxf((m_var), (d_val));                           \
    const float _r6_c = ((s_var) > 0.0f) ? __builtin_expf((m_var) - _r6_mn) : 0.0f;   \
    const float _r6_e = __builtin_expf((d_val) - _r6_mn);                             \
    (s_var) = ink_r6_attn_sgpr_f((s_var) * _r6_c + _r6_e);                            \
    (m_var) = ink_r6_attn_sgpr_f(_r6_mn);                                             \
    (a_var) = (a_var) * _r6_c + _r6_e * (vf_val);                                     \
  } while (0)

#define INK_R6_ATTN_NEAR_TOK(qi, kf_val, vf_val, sc_val, lane_id, kp_val, pos_q, rel_ext,             \
                             m0_v, s0_v, a0_v, m1_v, s1_v, a1_v, m2_v, s2_v, a2_v)                    \
  do {                                                                                                \
    if ((kp_val) <= (pos_q)) {                                                                        \
      float _r6_d0, _r6_d1, _r6_d2;                                                                   \
      INK_R6_ATTN_QK3((qi), (kf_val), (sc_val), (lane_id), _r6_d0, _r6_d1, _r6_d2);                   \
      const u32 _r6_dist = (pos_q) - (kp_val);                                                        \
      if (_r6_dist < (rel_ext)) {                                                                     \
        const float _r6_rb = ((lane_id) < INK_R6_ATTN_DREL)                                           \
                                 ? rel_bias[(unsigned long)_r6_dist * INK_R6_ATTN_DREL + (lane_id)]   \
                                 : 0.0f;                                                              \
        const LDSP float *_r6_rp =                                                                    \
            ink_r6_r_base + (unsigned long)(qi) * (INK_R6_ATTN_GQA * INK_R6_ATTN_DREL);               \
        const float _r6_rv0 = ((lane_id) < INK_R6_ATTN_DREL) ? _r6_rp[0u * INK_R6_ATTN_DREL + (lane_id)] : 0.0f; \
        const float _r6_rv1 = ((lane_id) < INK_R6_ATTN_DREL) ? _r6_rp[1u * INK_R6_ATTN_DREL + (lane_id)] : 0.0f; \
        const float _r6_rv2 = ((lane_id) < INK_R6_ATTN_DREL) ? _r6_rp[2u * INK_R6_ATTN_DREL + (lane_id)] : 0.0f; \
        _r6_d0 = ink_r6_attn_sgpr_f(_r6_d0 + ink_r6_attn_halfwave_sum_all(_r6_rv0 * _r6_rb));         \
        _r6_d1 = ink_r6_attn_sgpr_f(_r6_d1 + ink_r6_attn_halfwave_sum_all(_r6_rv1 * _r6_rb));         \
        _r6_d2 = ink_r6_attn_sgpr_f(_r6_d2 + ink_r6_attn_halfwave_sum_all(_r6_rv2 * _r6_rb));         \
      }                                                                                               \
      INK_R6_ATTN_UPD((m0_v), (s0_v), (a0_v), _r6_d0, (vf_val));                                      \
      INK_R6_ATTN_UPD((m1_v), (s1_v), (a1_v), _r6_d1, (vf_val));                                      \
      INK_R6_ATTN_UPD((m2_v), (s2_v), (a2_v), _r6_d2, (vf_val));                                      \
    }                                                                                                 \
  } while (0)

KERNEL_CALL ie_inkling_rel_attn_ver_qtile_r6(const G float *q, const G kvt *kc, const G kvt *vc, const G float *R,
                                             const G float *rel_bias, const G float *sink, G float *part, G float *out,
                                             u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv, u32 d_rel,
                                             u32 rel_extent, u32 win, u32 nsplit, u32 qs, u32 rs, u32 ss, u32 os,
                                             G u32 *count) {
#ifdef IE_MEGA
  _Static_assert(sizeof(mk_ink_lds.rel_attn.o) >=
                     (INK_R6_ATTN_QT * INK_R6_ATTN_GQA * (INK_R6_ATTN_HD + INK_R6_ATTN_DREL)) * sizeof(float),
                 "mk_ink_lds.rel_attn.o too small for QT=4 Q/R staging");
#define ink_r6_m      (mk_ink_lds.rel_attn.m)
#define ink_r6_l      (mk_ink_lds.rel_attn.l)
#define ink_r6_o      (mk_ink_lds.rel_attn.o)
#define ink_r6_last   (mk_ink_lds.rel_attn.last)
#define ink_r6_q_base ((LDSP float *)&mk_ink_lds.rel_attn.o[0][0][0])
#define ink_r6_r_base (((LDSP float *)&mk_ink_lds.rel_attn.o[0][0][0]) + \
                       (INK_R6_ATTN_QT * INK_R6_ATTN_GQA * INK_R6_ATTN_HD))
#else
  static LDS union {
    struct {
      float q[INK_R6_ATTN_QT * INK_R6_ATTN_GQA * INK_R6_ATTN_HD] __attribute__((aligned(16)));
      float r[INK_R6_ATTN_QT * INK_R6_ATTN_GQA * INK_R6_ATTN_DREL] __attribute__((aligned(16)));
    } qr;
    struct {
      float m[INK_R6_ATTN_GQA][8];
      float l[INK_R6_ATTN_GQA][8];
      float o[INK_R6_ATTN_GQA][8][INK_R6_ATTN_HD] __attribute__((aligned(16)));
    } red;
  } ink_r6_lds __attribute__((aligned(16)));
  static LDS u32 ink_r6_last;
#define ink_r6_m      (ink_r6_lds.red.m)
#define ink_r6_l      (ink_r6_lds.red.l)
#define ink_r6_o      (ink_r6_lds.red.o)
#define ink_r6_q_base (ink_r6_lds.qr.q)
#define ink_r6_r_base (ink_r6_lds.qr.r)
#endif

  const u32 t0 = __builtin_amdgcn_readfirstlane(tokid() * INK_R6_ATTN_QT);
  if (t0 >= T || win != 0u || hd != INK_R6_ATTN_HD || d_rel != INK_R6_ATTN_DREL || n_kv == 0u ||
      n_head != INK_R6_ATTN_GQA * n_kv || kv_len == 0u || nsplit == 0u)
    return;

  const u32 kh = __builtin_amdgcn_readfirstlane(wgid() / nsplit);
  const u32 s = __builtin_amdgcn_readfirstlane(wgid() % nsplit);
  if (kh >= n_kv) return;

  const u32 h0 = __builtin_amdgcn_readfirstlane(kh * INK_R6_ATTN_GQA);
  const u32 kvd = __builtin_amdgcn_readfirstlane(n_kv * INK_R6_ATTN_HD);
  const u32 w = __builtin_amdgcn_readfirstlane(tid() >> 5u);
  const u32 l = lane();

  const u32 rem = T - t0;
  const u32 nq = __builtin_amdgcn_readfirstlane(rem < INK_R6_ATTN_QT ? rem : INK_R6_ATTN_QT);
  const u32 pos_t0 = __builtin_amdgcn_readfirstlane(pos0 + t0);
  const u32 np = __builtin_amdgcn_readfirstlane(pos_t0 + nq); /* win == 0: union [0, pos0 + t0 + nq) */
  const u32 ch = __builtin_amdgcn_readfirstlane((np + nsplit - 1u) / nsplit);
  const u32 off0 = __builtin_amdgcn_readfirstlane(s * ch);
  const u32 p0 = __builtin_amdgcn_readfirstlane((off0 < np) ? off0 : np);
  const u32 p1 = __builtin_amdgcn_readfirstlane((off0 < np && off0 + ch < np) ? (off0 + ch) : np);

  if (nsplit > 1u && p0 >= p1) {
    /* Empty tail split (s * ch >= np): emit sentinel partials for all nq tokens
     * so split-K merge skips this split while advancing count[t0 * n_kv + kh]. */
    #pragma clang loop unroll(disable)
    for (u32 qi = 0u; qi < nq; qi++) {
      const u32 t = t0 + qi;
      for (u32 idx = tid(); idx < INK_R6_ATTN_GQA * INK_R6_ATTN_HD; idx += NT) {
        const u32 g = idx >> 7u;
        const u32 e = idx & 127u;
        const u32 h = h0 + g;
        G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R6_ATTN_PSTR;
        pp[2u + e] = 0.0f;
        if (e == 0u) {
          pp[0] = -1e30f;
          pp[1] = 0.0f;
        }
      }
    }
  } else {
    const int far_interior = __builtin_amdgcn_readfirstlane((p1 <= pos_t0 + 1u) && ((pos_t0 + 1u - p1) >= rel_extent));
    const float scale = 1.0f / __builtin_sqrtf((float)INK_R6_ATTN_HD);

    /* Phase 1: Cooperatively stage Q (nq * 3 * 128 f32) and optional R (nq * 3 * 16 f32) into LDS. */
    const u32 nq_vecs = __builtin_amdgcn_readfirstlane(nq * (INK_R6_ATTN_GQA * (INK_R6_ATTN_HD / 4u)));
    for (u32 idx = tid(); idx < nq_vecs; idx += NT) {
      const u32 qg = idx >> 5u;
      const u32 e4 = idx & 31u;
      const u32 qi = qg / INK_R6_ATTN_GQA;
      const u32 gi = qg - qi * INK_R6_ATTN_GQA;
      const G float *qt = (const G float *)((const G u8 *)q + (unsigned long)(t0 + qi) * qs);
      ((LDSP f32x4 *)(ink_r6_q_base + (unsigned long)qg * INK_R6_ATTN_HD))[e4] =
          ((const G f32x4 *)(qt + (unsigned long)(h0 + gi) * INK_R6_ATTN_HD))[e4];
    }
    if (!far_interior && rel_extent > 0u) {
      const u32 idx = tid();
      if (idx < nq * (INK_R6_ATTN_GQA * INK_R6_ATTN_DREL)) {
        const u32 qg = idx >> 4u;
        const u32 e = idx & 15u;
        const u32 qi = qg / INK_R6_ATTN_GQA;
        const u32 gi = qg - qi * INK_R6_ATTN_GQA;
        const G float *Rt = (const G float *)((const G u8 *)R + (unsigned long)(t0 + qi) * rs);
        ink_r6_r_base[(unsigned long)qg * INK_R6_ATTN_DREL + e] = Rt[(unsigned long)(h0 + gi) * INK_R6_ATTN_DREL + e];
      }
    }
    lbarrier();

    /* Explicit per-(token, head) accumulators (48 VGPRs) and SGPR softmax state (24 SGPRs). */
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

    const G kvt *kc_h = kc + (unsigned long)kh * INK_R6_ATTN_HD;
    const G kvt *vc_h = vc + (unsigned long)kh * INK_R6_ATTN_HD;

    /* Phase 2: Software-pipelined KV loop sharing 1 coalesced f16x4 K load + 1 V load
     * per wave across all 3 * nq queries (up to 12 queries per K/V load). */
    u32 kp = __builtin_amdgcn_readfirstlane(p0 + w);
    if (kp < p1) {
      u32 kvp = __builtin_amdgcn_readfirstlane((kp < kv_len) ? kp : (kp % kv_len));
      const G kv4 *krow = (const G kv4 *)(kc_h + (unsigned long)kvp * kvd);
      const G kv4 *vrow = (const G kv4 *)(vc_h + (unsigned long)kvp * kvd);
      kv4 kh4 = krow[l];
      kv4 vh4 = vrow[l];

      if (far_interior) {
        /* Far-field interior fast path: no causal mask, no rel_bias, no R loads. */
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

          /* Compute all Q*K scores first while vh4 + next K/V loads complete. */
          const f32x4 kf = __builtin_convertvector(kh4, f32x4);
          float d00, d01, d02;
          float d10 = 0.0f, d11 = 0.0f, d12 = 0.0f;
          float d20 = 0.0f, d21 = 0.0f, d22 = 0.0f;
          float d30 = 0.0f, d31 = 0.0f, d32 = 0.0f;

          INK_R6_ATTN_QK3(0u, kf, scale, l, d00, d01, d02);
          if (nq > 1u) INK_R6_ATTN_QK3(1u, kf, scale, l, d10, d11, d12);
          if (nq > 2u) INK_R6_ATTN_QK3(2u, kf, scale, l, d20, d21, d22);
          if (nq > 3u) INK_R6_ATTN_QK3(3u, kf, scale, l, d30, d31, d32);

          /* Online softmax + V accumulate (kf is dead, freeing its 4 VGPRs for vf). */
          const f32x4 vf = __builtin_convertvector(vh4, f32x4);
          INK_R6_ATTN_UPD(m00, s00, acc00, d00, vf);
          INK_R6_ATTN_UPD(m01, s01, acc01, d01, vf);
          INK_R6_ATTN_UPD(m02, s02, acc02, d02, vf);
          if (nq > 1u) {
            INK_R6_ATTN_UPD(m10, s10, acc10, d10, vf);
            INK_R6_ATTN_UPD(m11, s11, acc11, d11, vf);
            INK_R6_ATTN_UPD(m12, s12, acc12, d12, vf);
          }
          if (nq > 2u) {
            INK_R6_ATTN_UPD(m20, s20, acc20, d20, vf);
            INK_R6_ATTN_UPD(m21, s21, acc21, d21, vf);
            INK_R6_ATTN_UPD(m22, s22, acc22, d22, vf);
          }
          if (nq > 3u) {
            INK_R6_ATTN_UPD(m30, s30, acc30, d30, vf);
            INK_R6_ATTN_UPD(m31, s31, acc31, d31, vf);
            INK_R6_ATTN_UPD(m32, s32, acc32, d32, vf);
          }

          if (kp_nxt >= p1) break;
          kp = kp_nxt;
          kvp = kvp_nxt;
          kh4 = kh4_nxt;
          vh4 = vh4_nxt;
        }
      } else {
        /* Near-field & tail path: exact per-query causal mask + relative distance bias. */
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

          INK_R6_ATTN_NEAR_TOK(0u, kf, vf, scale, l, kp, pos_q0, rel_extent,
                               m00, s00, acc00, m01, s01, acc01, m02, s02, acc02);
          if (nq > 1u) {
            INK_R6_ATTN_NEAR_TOK(1u, kf, vf, scale, l, kp, pos_q1, rel_extent,
                                 m10, s10, acc10, m11, s11, acc11, m12, s12, acc12);
          }
          if (nq > 2u) {
            INK_R6_ATTN_NEAR_TOK(2u, kf, vf, scale, l, kp, pos_q2, rel_extent,
                                 m20, s20, acc20, m21, s21, acc21, m22, s22, acc22);
          }
          if (nq > 3u) {
            INK_R6_ATTN_NEAR_TOK(3u, kf, vf, scale, l, kp, pos_q3, rel_extent,
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

    /* Ensure all 8 waves finish reading Q/R LDS before overlaying the reduction buffer. */
    lbarrier();

    /* Phase 3: Sequentially reduce each token qi = 0..nq-1 across the 8 waves
     * using the single 12.2 KiB LDS reduction buffer and emit out (nsplit==1) or part. */
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
        ink_r6_m[0][w] = rm0;
        ink_r6_l[0][w] = rl0;
        ink_r6_m[1][w] = rm1;
        ink_r6_l[1][w] = rl1;
        ink_r6_m[2][w] = rm2;
        ink_r6_l[2][w] = rl2;
      }
      ((LDSP f32x4 *)&ink_r6_o[0][w][0])[l] = ra0;
      ((LDSP f32x4 *)&ink_r6_o[1][w][0])[l] = ra1;
      ((LDSP f32x4 *)&ink_r6_o[2][w][0])[l] = ra2;
      lbarrier();

      const u32 t = t0 + qi;
      if (nsplit == 1u) {
        const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
        G float *ot = (G float *)((G u8 *)out + (unsigned long)t * os);
        for (u32 idx = tid(); idx < INK_R6_ATTN_GQA * INK_R6_ATTN_HD; idx += NT) {
          const u32 g = idx >> 7u;
          const u32 e = idx & 127u;
          const u32 h = h0 + g;
          const float sv = st[h];
          float M = sv;
          for (u32 x = 0u; x < 8u; x++)
            if (ink_r6_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r6_m[g][x]);
          float L = __builtin_expf(sv - M);
          float o = 0.0f;
          for (u32 x = 0u; x < 8u; x++) {
            const float lx = ink_r6_l[g][x];
            if (lx > 0.0f) {
              const float wx = __builtin_expf(ink_r6_m[g][x] - M);
              L += lx * wx;
              o += ink_r6_o[g][x][e] * wx;
            }
          }
          ot[(unsigned long)h * INK_R6_ATTN_HD + e] = o / L;
        }
      } else {
        for (u32 idx = tid(); idx < INK_R6_ATTN_GQA * INK_R6_ATTN_HD; idx += NT) {
          const u32 g = idx >> 7u;
          const u32 e = idx & 127u;
          const u32 h = h0 + g;
          float M = -__builtin_inff();
          for (u32 x = 0u; x < 8u; x++)
            if (ink_r6_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r6_m[g][x]);
          G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R6_ATTN_PSTR;
          float o = 0.0f;
          float L = 0.0f;
          for (u32 x = 0u; x < 8u; x++) {
            const float lx = ink_r6_l[g][x];
            if (lx > 0.0f) {
              const float wx = __builtin_expf(ink_r6_m[g][x] - M);
              o += ink_r6_o[g][x][e] * wx;
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

  /* Phase 4: Split-K arrival & last-WG 3-pass merge for each token qi = 0..nq-1. */
  const u32 cidx = __builtin_amdgcn_readfirstlane(t0 * n_kv + kh);
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r6_last =
        (__scoped_atomic_fetch_add(&count[cidx], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u);
  }
  barrier();
  if (!ink_r6_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const int wt_in_lds = (nsplit <= INK_R6_ATTN_HD);
  #pragma clang loop unroll(disable)
  for (u32 qi = 0u; qi < nq; qi++) {
    if (qi > 0u) lbarrier();
    const u32 t = t0 + qi;
    const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
    G float *ot = (G float *)((G u8 *)out + (unsigned long)t * os);

    G float *ph0 = part + (unsigned long)((t * n_head + (h0 + 0u)) * nsplit) * INK_R6_ATTN_PSTR;
    G float *ph1 = part + (unsigned long)((t * n_head + (h0 + 1u)) * nsplit) * INK_R6_ATTN_PSTR;
    G float *ph2 = part + (unsigned long)((t * n_head + (h0 + 2u)) * nsplit) * INK_R6_ATTN_PSTR;

    /* Pass 1: global max M2[0..2] including per-token, per-head sink. */
    float mv0 = (tid() == 0u) ? st[h0 + 0u] : -__builtin_inff();
    float mv1 = (tid() == 0u) ? st[h0 + 1u] : -__builtin_inff();
    float mv2 = (tid() == 0u) ? st[h0 + 2u] : -__builtin_inff();
    for (u32 sp = tid(); sp < nsplit; sp += NT) {
      const unsigned long off = (unsigned long)sp * INK_R6_ATTN_PSTR;
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
      ink_r6_m[0][w] = mv0;
      ink_r6_m[1][w] = mv1;
      ink_r6_m[2][w] = mv2;
    }
    lbarrier();

    float M2[INK_R6_ATTN_GQA];
    for (u32 g = 0u; g < INK_R6_ATTN_GQA; g++) {
      float mx = ink_r6_m[g][0];
      for (u32 x = 1u; x < 8u; x++) mx = __builtin_fmaxf(mx, ink_r6_m[g][x]);
      M2[g] = mx;
    }

    /* Pass 2: per-split weights + global denominator L2[0..2]. */
    float lv0 = 0.0f, lv1 = 0.0f, lv2 = 0.0f;
    for (u32 sp = tid(); sp < nsplit; sp += NT) {
      const unsigned long off = (unsigned long)sp * INK_R6_ATTN_PSTR;

      const float lp0 = ph0[off + 1u];
      const float wt0 = (lp0 > 0.0f) ? __builtin_expf(ph0[off] - M2[0]) : 0.0f;
      if (wt_in_lds) ink_r6_o[0][0][sp] = wt0;
      else ph0[off] = wt0;
      lv0 += lp0 * wt0;

      const float lp1 = ph1[off + 1u];
      const float wt1 = (lp1 > 0.0f) ? __builtin_expf(ph1[off] - M2[1]) : 0.0f;
      if (wt_in_lds) ink_r6_o[1][0][sp] = wt1;
      else ph1[off] = wt1;
      lv1 += lp1 * wt1;

      const float lp2 = ph2[off + 1u];
      const float wt2 = (lp2 > 0.0f) ? __builtin_expf(ph2[off] - M2[2]) : 0.0f;
      if (wt_in_lds) ink_r6_o[2][0][sp] = wt2;
      else ph2[off] = wt2;
      lv2 += lp2 * wt2;
    }
    lv0 = wave_tree_f(lv0);
    lv1 = wave_tree_f(lv1);
    lv2 = wave_tree_f(lv2);
    if (l == 0u) {
      ink_r6_l[0][w] = lv0;
      ink_r6_l[1][w] = lv1;
      ink_r6_l[2][w] = lv2;
    }
    if (wt_in_lds) lbarrier();
    else barrier();

    float L2[INK_R6_ATTN_GQA];
    for (u32 g = 0u; g < INK_R6_ATTN_GQA; g++) {
      float ls = 0.0f;
      for (u32 x = 0u; x < 8u; x++) ls += ink_r6_l[g][x];
      L2[g] = ls + __builtin_expf(st[h0 + g] - M2[g]);
    }

    /* Pass 3: weighted sum of partial outputs -> ot[h * 128 + e]. */
    for (u32 idx = tid(); idx < INK_R6_ATTN_GQA * INK_R6_ATTN_HD; idx += NT) {
      const u32 g = idx >> 7u;
      const u32 e = idx & 127u;
      const u32 h = h0 + g;
      const G float *ph = part + (unsigned long)((t * n_head + h) * nsplit) * INK_R6_ATTN_PSTR;
      float o = 0.0f;
      if (wt_in_lds) {
        for (u32 sp = 0u; sp < nsplit; sp++)
          o += ph[(unsigned long)sp * INK_R6_ATTN_PSTR + 2u + e] * ink_r6_o[g][0][sp];
      } else {
        for (u32 sp = 0u; sp < nsplit; sp++)
          o += ph[(unsigned long)sp * INK_R6_ATTN_PSTR + 2u + e] * ph[(unsigned long)sp * INK_R6_ATTN_PSTR];
      }
      ot[(unsigned long)h * INK_R6_ATTN_HD + e] = o / L2[g];
    }
  }

  if (tid() == 0u) count[cidx] = 0u;
}

#undef ink_r6_m
#undef ink_r6_l
#undef ink_r6_o
#ifdef IE_MEGA
#undef ink_r6_last
#endif
#undef ink_r6_q_base
#undef ink_r6_r_base
#undef INK_R6_ATTN_NEAR_TOK
#undef INK_R6_ATTN_UPD
#undef INK_R6_ATTN_QK3
#undef INK_R6_ATTN_HD
#undef INK_R6_ATTN_DREL
#undef INK_R6_ATTN_GQA
#undef INK_R6_ATTN_QT
#undef INK_R6_ATTN_PSTR
