/* ============================================================================
 * Inkling-2 Round 7: Fused Q4_K Head + Softcap + Online Top-K/Argmax (Bit 84)
 *                    & 0-VRAM id_map Gather DSpark Shortlist Suite (Bit 88)
 * Target: RX 9070 XT (gfx1201, wave32, NT=256)
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r7_fh_ /
 * INK_R7_FH_ and all macros are #undef'd at EOF.
 *
 * Exported kernels:
 *
 * 1. Bit 84: KERNEL ie_inkling_q4k_head_cap_topk_r7(...)
 *    - Persistent grid-stride fused Q4_K output.weight (201024x2304, nb=72) +
 *      inline cap*tanh(x/cap) + tile-local online Argmax (K==1) OR Top-K
 *      (2 <= K <= 32) directly to output / host-pinned buffer in ONE launch!
 *    - Eliminates ALL writes & re-reads of g->logits (up to 12.8 MB saved at
 *      T=8) AND collapses [OP_GEMV_Q4K, OP_LOGIT_CAP, T x OP_ARGMAX | topk]
 *      into 1 single launch!
 *    - Grid: (ng, 1u, 1u) with 1 <= ng <= 128 (e.g. ng=128 = 2 WGs/CU for both
 *      greedy K==1 and sampled K<=32 -> nc = ng * K <= 4096 exact merge!).
 *    - Exact 0-bit-diff parity vs ie_inkling_gemv_q4k_head_ver_r7 + ie_argmax /
 *      ie_inkling_topk_batched_r7 (exact 8-lane/row Q4_K wave_tree_f, exact
 *      lowest-index tie-breaking!).
 *    - Scratch:
 *        part_v, part_i: T * ng * K floats / u32s (at T=8, ng=128, K=32: 128 KiB
 *                        each = 256 KiB total; fits inside existing b->tk_buf!).
 *        done:           1 zero-initialized u32 counter (auto-reset to 0 by last WG).
 *    - Output:
 *        If K == 1u: writes [best_id, 0u] at out_am + t * (am_stride / 4u)
 *                    (or also ov[t]=best_v, oi[t]=best_id if ov/oi non-NULL).
 *        If K >  1u: writes sorted ov[t*K + 0..K-1], oi[t*K + 0..K-1] (value
 *                    descending, smaller token index first on ties).
 *
 * 2. Bit 88 (R30 0-VRAM 60K+4K Shortlist Gather Suite, matching wave6-7-host-specs.md):
 *    (a) KERNEL ie_inkling_ds_base_q4k_idmap_r7(...)
 *        - Gathers n_act (<= 65536) rows tok = id_map[s] directly from target's
 *          existing Q4_K output.weight (0 extra VRAM!) for T = 1..8 drafts ->
 *          writes compact base[j * ys + s]. Grid: ((n_act + 31u) >> 5u, 1u, 1u).
 *    (b) KERNEL ie_inkling_ds_gemv_p_q4_idmap_zsync_r7(...)
 *        - Gathers n_act rows tok = id_map[s] directly from existing Q4_0 split P
 *          (nb=8, 1 exact 128-B cache line/row!) with ds_ctrl[1] early exit ->
 *          writes logits[s] = base_row[s] + dot. Grid: ((n_act + 31u) >> 5u, 1u, 1u).
 *    (c) KERNEL ie_inkling_ds_cap_sample_idmap_zsync_r7(...)
 *        - Wave 6 greedy/probabilistic zsync + ForeSpark entropy kernel with
 *          id_map[0..n_act-1] canonical token ID mapping & canonical tie-breaking
 *          (compares tok = id_map[s], writes canonical tok to ds_ctrl[0],
 *          draft_ids[], and q_ids[] so RNN & spec_rej_step need 0 changes!).
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R7_FH_TMAX  8u
#define INK_R7_FH_KMAX  32u
#define INK_R7_FH_NGMAX 128u
#define INK_R7_FH_CAP   1024u

typedef float ink_r7_fh_f8 __attribute__((ext_vector_type(8)));
typedef unsigned long long ink_r7_fh_u64;

struct __attribute__((packed, aligned(4))) ink_r7_fh_u32x4p {
  u32x4 v;
};

struct __attribute__((packed, aligned(4))) ink_r7_fh_f32x4p {
  f32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r7_fh_ld4(const G void *p) {
  return ((const G struct ink_r7_fh_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) float ink_r7_fh_cap(float v, float cap) {
  return (cap > 0.0f) ? (cap * ie_tanhf_s(v / cap)) : v;
}

static inline __attribute__((always_inline)) float ink_r7_fh_cap_inv(float v, float cap, float inv_cap) {
  return (cap > 0.0f) ? (cap * ie_tanhf_s(v * inv_cap)) : v;
}

static inline __attribute__((always_inline)) int ink_r7_fh_better(float v, u32 i, float bv, u32 bi) {
  return (v > bv) || (v == bv && i < bi);
}

static inline __attribute__((always_inline)) u32 ink_r7_fh_f2u(float f) {
  u32 u = __builtin_bit_cast(u32, f);
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static inline __attribute__((always_inline)) float ink_r7_fh_u2f(u32 k) {
  const u32 u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  return __builtin_bit_cast(float, u);
}

static inline __attribute__((always_inline)) u32 ink_r7_fh_cswap(u32 v, u32 ov, u32 hi, u32 asc) {
  return (hi ^ asc) ? (v < ov ? v : ov) : (v > ov ? v : ov);
}

static inline __attribute__((always_inline)) u32 ink_r7_fh_wave_step(u32 v, u32 t, u32 stride, u32 asc) {
  const u32 ov = (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ stride) * 4u), (int)v);
  return ink_r7_fh_cswap(v, ov, (u32)((t & stride) != 0u), asc);
}

static inline __attribute__((always_inline)) u32 ink_r7_fh_lds_step(
    u32 v, u32 t, u32 stride, u32 asc, LDSP u32 *buf) {
  buf[t] = v;
  lbarrier();
  const u32 ov = buf[t ^ stride];
  return ink_r7_fh_cswap(v, ov, (u32)((t & stride) != 0u), asc);
}

/* Exact 8-lane/row Q4_K dot product for 1 row r across T <= 8 tokens (matches r7_q4k_head). */
static inline __attribute__((always_inline)) void ink_r7_fh_q4k_row(
    const G u32 *qw, const G u16 *qs, const G u8 *xq,
    u32 r, u32 rows, u32 nb, u32 T, u32 xs, u32 sub,
    ink_r7_fh_f8 *out_v) {
#pragma clang fp contract(off)
  const u32 nsb = nb >> 3u;
  const G u32 *row_qw = qw + (unsigned long)r * (unsigned long)nb * 4ul;
  const G u16 *row_sm = qs + (unsigned long)r * (unsigned long)nb;
  const G u32 *row_dd32 = (const G u32 *)(qs + (unsigned long)rows * (unsigned long)nb +
                                          (unsigned long)r * (unsigned long)nsb * 2ul);
  const u32 sub32 = sub << 5u;
  const u32 sub4 = sub << 2u;

  ink_r7_fh_f8 a0 = (ink_r7_fh_f8)(0.0f);
  ink_r7_fh_f8 a1 = (ink_r7_fh_f8)(0.0f);
  ink_r7_fh_f8 a2 = (ink_r7_fh_f8)(0.0f);
  ink_r7_fh_f8 a3 = (ink_r7_fh_f8)(0.0f);

#pragma clang loop unroll(disable)
  for (u32 sb = 0u; sb < nsb; sb++) {
    const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
    const u32 slot = (u32)__builtin_amdgcn_readfirstlane((int)(sb_u & 3u));
    const u32 b = (sb_u << 3u) + sub;

    const u32x4 w4 = ink_r7_fh_ld4(row_qw + (unsigned long)b * 4ul);
    const u32 sm = (u32)row_sm[b];
    const u32 dd_raw = row_dd32[sb_u];
    const u16 d_bits = (u16)(dd_raw & 0xFFFFu);
    const u16 dm_bits = (u16)(dd_raw >> 16u);
    const float d = (float)__builtin_bit_cast(f16, d_bits);
    const float dmin = (float)__builtin_bit_cast(f16, dm_bits);
    const float sc = d * (float)(sm & 63u);
    const float mm = dmin * (float)(sm >> 8u);

    const u32 lo0 = ie_q4k_lo(w4.x), hi0 = ie_q4k_hi(w4.x);
    const u32 lo1 = ie_q4k_lo(w4.y), hi1 = ie_q4k_hi(w4.y);
    const u32 lo2 = ie_q4k_lo(w4.z), hi2 = ie_q4k_hi(w4.z);
    const u32 lo3 = ie_q4k_lo(w4.w), hi3 = ie_q4k_hi(w4.w);

    const unsigned long aw_sb = (unsigned long)sb_u * 256ul;
    const unsigned long da_sb = (unsigned long)nb * 32ul + (unsigned long)sb_u * 32ul;
    const unsigned long as_sb = (unsigned long)nb * 36ul + (unsigned long)sb_u * 32ul;

#pragma unroll
    for (u32 j = 0u; j < INK_R7_FH_TMAX; j++) {
      if (j < T) {
        const G u8 *xj = xq + (unsigned long)j * (unsigned long)xs;
        const G u8 *xj_aw = xj + aw_sb;
        const u32x4 act0 = ink_r7_fh_ld4(xj_aw + sub32);
        const u32x4 act1 = ink_r7_fh_ld4(xj_aw + sub32 + 16u);
        const float da = *(const G float *)(xj + da_sb + sub4);
        const int as = (int)(*(const G u32 *)(xj + as_sb + sub4));
        const float xa = da * (float)as;

        u32 dq = ie_dot4_us(0u, lo0, act0.x);
        dq = ie_dot4_us(dq, hi0, act1.x);
        dq = ie_dot4_us(dq, lo1, act0.y);
        dq = ie_dot4_us(dq, hi1, act1.y);
        dq = ie_dot4_us(dq, lo2, act0.z);
        dq = ie_dot4_us(dq, hi2, act1.z);
        dq = ie_dot4_us(dq, lo3, act0.w);
        dq = ie_dot4_us(dq, hi3, act1.w);

        const float prod = sc * (float)(int)dq;
        if (slot == 0u) {
          a0[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a0[j]));
        } else if (slot == 1u) {
          a1[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a1[j]));
        } else if (slot == 2u) {
          a2[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a2[j]));
        } else {
          a3[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a3[j]));
        }
        if ((j & 1u) == 1u) __builtin_amdgcn_sched_barrier(0);
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  ink_r7_fh_f8 res = (ink_r7_fh_f8)(0.0f);
#pragma unroll
  for (u32 j = 0u; j < INK_R7_FH_TMAX; j++) {
    if (j < T) {
      float s = (a0[j] + a2[j]) + (a1[j] + a3[j]);
      s = s + I2F(shl_i(F2I(s), 4));
      s = s + I2F(shl_i(F2I(s), 2));
      s = s + I2F(shl_i(F2I(s), 1));
      res[j] = s;
    }
  }
  *out_v = res;
}

/* ============================================================================
 * Kernel 1 (Bit 84): ie_inkling_q4k_head_cap_topk_r7
 * Persistent Q4_K output.weight + softcap + tile-local online Argmax / Top-K.
 * ========================================================================== */

typedef struct {
  union {
    struct {
      /* Tile-local top-K heap per token (8 * 32 * 8 = 2,048 B) + 32-row chunk staging (1,536 B). */
      float kv[INK_R7_FH_TMAX][INK_R7_FH_KMAX];
      u32   ki[INK_R7_FH_TMAX][INK_R7_FH_KMAX];
      float min_v[INK_R7_FH_TMAX];
      u32   min_i[INK_R7_FH_TMAX];
      u32   min_pos[INK_R7_FH_TMAX];
      float ch_v[INK_R7_FH_TMAX][32u];
      u32   ch_i[32u];
      u32   any_pass[INK_R7_FH_TMAX];
    } st1;
    u32 sm[2u][NT];
    struct {
      float cv[INK_R7_FH_CAP];
      u32   ci[INK_R7_FH_CAP];
    } cand;
  } u;
  u32   wsum[2u][8u];
  float wv[8u];
  u32   wi[8u];
  u32   last;
} ink_r7_fh_head_lds_t;

static LDS ink_r7_fh_head_lds_t ink_r7_fh_hlds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r7_fh_head_lds_t) <= 16384u, "ink_r7_fh_hlds must stay <= 16 KiB");

static inline __attribute__((always_inline)) u32 ink_r7_fh_bitonic_kth_256(u32 v, u32 K) {
  const u32 t = tid();
#pragma clang loop unroll(disable)
  for (u32 size = 2u; size <= 32u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride > 0u; stride >>= 1u) {
      v = ink_r7_fh_wave_step(v, t, stride, a);
    }
  }
  u32 buf_id = 0u;
#pragma clang loop unroll(disable)
  for (u32 size = 64u; size <= 256u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride >= 32u; stride >>= 1u) {
      v = ink_r7_fh_lds_step(v, t, stride, a, ink_r7_fh_hlds.u.sm[buf_id]);
      buf_id ^= 1u;
    }
#pragma clang loop unroll(disable)
    for (u32 stride = 16u; stride > 0u; stride >>= 1u) {
      v = ink_r7_fh_wave_step(v, t, stride, a);
    }
  }
  if (t == K - 1u) ink_r7_fh_hlds.u.sm[0u][0u] = v;
  lbarrier();
  return ink_r7_fh_hlds.u.sm[0u][0u];
}

static inline __attribute__((always_inline)) u32 ink_r7_fh_wg_scan_u32(
    u32 v, LDSP u32 *wsum, u32 *total_out) {
  const u32 l = lane();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  u32 x = v;
#pragma clang loop unroll(disable)
  for (u32 d = 1u; d < 32u; d <<= 1u) {
    const u32 src = (l >= d) ? (l - d) : 0u;
    const u32 ov = (u32)__builtin_amdgcn_ds_bpermute((int)(src * 4u), (int)x);
    if (l >= d) x += ov;
  }
  if (l == 31u) wsum[w] = x;
  lbarrier();
  u32 wb = 0u, tot = 0u;
#pragma clang loop unroll(disable)
  for (u32 j = 0u; j < 8u; j++) {
    const u32 wv = wsum[j];
    if (j < w) wb += wv;
    tot += wv;
  }
  *total_out = tot;
  return wb + (x - v);
}

static inline __attribute__((always_inline)) void ink_r7_fh_classify16(
    const float rv[16u], u32 vmask, u32 tmax_k, u32 tau_k, float tau,
    u32 *gt_mask_out, u32 *eq_mask_out) {
  u32 gm = 0u, em = 0u;
  if (tau_k == 0u) {
    gm = vmask;
  } else if (tmax_k == tau_k) {
#pragma unroll
    for (u32 k = 0u; k < 16u; k++) {
      if ((vmask & (1u << k)) && rv[k] == tau) em |= (1u << k);
    }
  } else if (tmax_k > tau_k) {
#pragma unroll
    for (u32 k = 0u; k < 16u; k++) {
      if (vmask & (1u << k)) {
        const float v = rv[k];
        if (v > tau) gm |= (1u << k);
        else if (v == tau) em |= (1u << k);
      }
    }
  }
  *gt_mask_out = gm;
  *eq_mask_out = em;
}

static inline __attribute__((always_inline)) void ink_r7_fh_rank_write(
    u32 c, u32 K, G float *out_v, G u32 *out_i) {
  for (u32 j = tid(); j < c; j += NT) {
    const float a = ink_r7_fh_hlds.u.cand.cv[j];
    const u32 ai = ink_r7_fh_hlds.u.cand.ci[j];
    u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
    for (u32 u = 0u; u < c; u++) {
      const float b = ink_r7_fh_hlds.u.cand.cv[u];
      rank += (u32)((b > a) | ((b == a) & (ink_r7_fh_hlds.u.cand.ci[u] < ai)));
    }
    if (rank < K) {
      out_v[rank] = a;
      out_i[rank] = ai;
    }
  }
}

KERNEL ie_inkling_q4k_head_cap_topk_r7(
    const G u32 *qw, const G u16 *qs, const G u8 *xq,
    const G float *bias,
    u32 rows, u32 unpadded, u32 nb, u32 T, u32 xs,
    float cap, u32 K, u32 ng,
    G float *part_v, G u32 *part_i, G u32 *done,
    G u32 *out_am, u32 am_stride,
    G float *ov, G u32 *oi) {
  const u32 g = wgid();
  const u32 t = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 row_in_wg = (w << 2u) + (l >> 3u);
  const u32 nv = (unpadded > 0u && unpadded < rows) ? unpadded : rows;

  if (g >= ng || ng == 0u || ng > INK_R7_FH_NGMAX || T == 0u || T > INK_R7_FH_TMAX ||
      K == 0u || K > INK_R7_FH_KMAX || nb == 0u || (nb & 7u) != 0u || nv == 0u) return;

  /* Initialize tile-local top-K buffer in LDS for all T tokens. */
  if (t < T * K) {
    const u32 tj = t / K;
    const u32 tk = t - tj * K;
    ink_r7_fh_hlds.u.st1.kv[tj][tk] = -__builtin_inff();
    ink_r7_fh_hlds.u.st1.ki[tj][tk] = 0xFFFFFFFFu;
  }
  if (t < T) {
    ink_r7_fh_hlds.u.st1.min_v[t] = -__builtin_inff();
    ink_r7_fh_hlds.u.st1.min_i[t] = 0xFFFFFFFFu;
    ink_r7_fh_hlds.u.st1.min_pos[t] = 0u;
    ink_r7_fh_hlds.u.st1.any_pass[t] = 0u;
  }
  lbarrier();

  /* Persistent grid-stride loop over 32-row chunks of output.weight [0..nv-1]. */
  for (u32 rbase = g << 5u; rbase < nv; rbase += (ng << 5u)) {
    const u32 r = rbase + row_in_wg;
    const u32 r_valid = (r < nv) ? 1u : 0u;
    ink_r7_fh_f8 vals = (ink_r7_fh_f8)(-__builtin_inff());
    if (r_valid) {
      ink_r7_fh_q4k_row(qw, qs, xq, r, rows, nb, T, xs, sub, &vals);
    }

    if (sub == 0u) {
      ink_r7_fh_hlds.u.st1.ch_i[row_in_wg] = r_valid ? r : 0xFFFFFFFFu;
      const float bv_bias = (r_valid && bias) ? bias[r] : 0.0f;
#pragma unroll
      for (u32 j = 0u; j < INK_R7_FH_TMAX; j++) {
        if (j < T) {
          float v = r_valid ? ink_r7_fh_cap(vals[j] + bv_bias, cap) : -__builtin_inff();
          ink_r7_fh_hlds.u.st1.ch_v[j][row_in_wg] = v;
          if (r_valid && ink_r7_fh_better(v, r, ink_r7_fh_hlds.u.st1.min_v[j], ink_r7_fh_hlds.u.st1.min_i[j])) {
            ink_r7_fh_hlds.u.st1.any_pass[j] = 1u;
          }
        }
      }
    }
    lbarrier();

    /* Wave j (0..T-1) updates token j's top-K heap in parallel ONLY if any of the 32 rows beat min_v[j]!
     * After chunk 0, >99% of chunks skip completely in 1 branch! */
    if (w < T && ink_r7_fh_hlds.u.st1.any_pass[w] != 0u) {
      if (l == 0u) {
        float cur_mv = ink_r7_fh_hlds.u.st1.min_v[w];
        u32 cur_mi = ink_r7_fh_hlds.u.st1.min_i[w];
        u32 cur_mp = ink_r7_fh_hlds.u.st1.min_pos[w];
#pragma clang loop unroll(disable)
        for (u32 c = 0u; c < 32u; c++) {
          const u32 ci = ink_r7_fh_hlds.u.st1.ch_i[c];
          if (ci == 0xFFFFFFFFu) continue;
          const float cv = ink_r7_fh_hlds.u.st1.ch_v[w][c];
          if (ink_r7_fh_better(cv, ci, cur_mv, cur_mi)) {
            ink_r7_fh_hlds.u.st1.kv[w][cur_mp] = cv;
            ink_r7_fh_hlds.u.st1.ki[w][cur_mp] = ci;
            if (K == 1u) {
              cur_mv = cv;
              cur_mi = ci;
            } else {
              /* Find new worst element in [0..K-1] (K <= 32). */
              float wv_min = ink_r7_fh_hlds.u.st1.kv[w][0u];
              u32 wi_min = ink_r7_fh_hlds.u.st1.ki[w][0u];
              u32 wp_min = 0u;
#pragma clang loop unroll(disable)
              for (u32 p = 1u; p < K; p++) {
                const float pv = ink_r7_fh_hlds.u.st1.kv[w][p];
                const u32 pi = ink_r7_fh_hlds.u.st1.ki[w][p];
                /* Worse means: pi == ~0, OR (wi_min != ~0 && (pv < wv_min || (pv == wv_min && pi > wi_min))). */
                if (pi == 0xFFFFFFFFu) {
                  if (wi_min != 0xFFFFFFFFu) { wv_min = pv; wi_min = pi; wp_min = p; }
                } else if (wi_min != 0xFFFFFFFFu && (pv < wv_min || (pv == wv_min && pi > wi_min))) {
                  wv_min = pv; wi_min = pi; wp_min = p;
                }
              }
              cur_mv = wv_min; cur_mi = wi_min; cur_mp = wp_min;
            }
          }
        }
        ink_r7_fh_hlds.u.st1.min_v[w] = cur_mv;
        ink_r7_fh_hlds.u.st1.min_i[w] = cur_mi;
        ink_r7_fh_hlds.u.st1.min_pos[w] = cur_mp;
        ink_r7_fh_hlds.u.st1.any_pass[w] = 0u;
      }
    }
    lbarrier();
  }

  /* Export this WG's K winners per token to part_v / part_i. */
  const u32 tk_total = T * K;
  for (u32 idx = t; idx < tk_total; idx += NT) {
    const u32 tj = idx / K;
    const u32 tk = idx - tj * K;
    const unsigned long dst = (((unsigned long)tj * (unsigned long)ng + (unsigned long)g) * (unsigned long)K) + tk;
    part_v[dst] = ink_r7_fh_hlds.u.st1.kv[tj][tk];
    part_i[dst] = ink_r7_fh_hlds.u.st1.ki[tj][tk];
  }

  /* Agent release/acquire + last-WG election across all ng WGs. */
  if (ng > 1u) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (t == 0u) {
      ink_r7_fh_hlds.last =
          (__scoped_atomic_fetch_add(done, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r7_fh_hlds.last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (t == 0u) *done = 0u;
  } else {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  }

  /* Stage 2 (last WG): merge nc = ng * K <= 4096 candidates for each token j = 0..T-1! */
  const u32 nc = ng * K;
  for (u32 tj = 0u; tj < T; tj++) {
    const G float *cv_row = part_v + (unsigned long)tj * (unsigned long)nc;
    const G u32   *ci_row = part_i + (unsigned long)tj * (unsigned long)nc;

    if (K == 1u) {
      float bv = -__builtin_inff();
      u32 bi = 0xFFFFFFFFu;
      for (u32 u = t; u < ng; u += NT) {
        const float vu = cv_row[u];
        const u32 iu = ci_row[u];
        if (ink_r7_fh_better(vu, iu, bv, bi)) { bv = vu; bi = iu; }
      }
#pragma unroll
      for (u32 m = 16u; m > 0u; m >>= 1u) {
        const float ov_m = shfl_xor_f(bv, m);
        const u32 oi_m = (u32)__builtin_amdgcn_ds_bpermute((int)((l ^ m) * 4u), (int)bi);
        if (ink_r7_fh_better(ov_m, oi_m, bv, bi)) { bv = ov_m; bi = oi_m; }
      }
      if (l == 0u) { ink_r7_fh_hlds.wv[w] = bv; ink_r7_fh_hlds.wi[w] = bi; }
      lbarrier();
      if (t == 0u) {
        float fv = ink_r7_fh_hlds.wv[0u];
        u32 fi = ink_r7_fh_hlds.wi[0u];
#pragma clang loop unroll(disable)
        for (u32 j = 1u; j < 8u; j++) {
          if (ink_r7_fh_better(ink_r7_fh_hlds.wv[j], ink_r7_fh_hlds.wi[j], fv, fi)) {
            fv = ink_r7_fh_hlds.wv[j]; fi = ink_r7_fh_hlds.wi[j];
          }
        }
        if (out_am) {
          G u32 *am_r = (G u32 *)((G u8 *)out_am + (unsigned long)tj * (unsigned long)am_stride);
          am_r[0u] = fi;
          am_r[1u] = 0u;
        }
        if (ov) ov[tj] = fv;
        if (oi) oi[tj] = fi;
      }
      lbarrier();
      continue;
    }

    G float *ov_row = ov + (unsigned long)tj * (unsigned long)K;
    G u32   *oi_row = oi + (unsigned long)tj * (unsigned long)K;

    if (nc <= NT) {
      float mv = -__builtin_inff();
      u32 mi = 0xFFFFFFFFu;
      if (t < nc) {
        mv = cv_row[t];
        mi = ci_row[t];
        ink_r7_fh_hlds.u.cand.cv[t] = mv;
        ink_r7_fh_hlds.u.cand.ci[t] = mi;
      }
      const u32 is_val = (t < nc && mi != 0xFFFFFFFFu) ? 1u : 0u;
      const u32 wcnt = (u32)__builtin_popcount(__builtin_amdgcn_ballot_w32(is_val != 0u));
      if (l == 0u) ink_r7_fh_hlds.wsum[0u][w] = wcnt;
      lbarrier();
      u32 nvalid = 0u;
#pragma clang loop unroll(disable)
      for (u32 j = 0u; j < 8u; j++) nvalid += ink_r7_fh_hlds.wsum[0u][j];
      if (is_val) {
        u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
        for (u32 u = 0u; u < nc; u++) {
          const u32 ui = ink_r7_fh_hlds.u.cand.ci[u];
          if (ui != 0xFFFFFFFFu) {
            const float b = ink_r7_fh_hlds.u.cand.cv[u];
            rank += (u32)((b > mv) | ((b == mv) & (ui < mi)));
          }
        }
        if (rank < K) { ov_row[rank] = mv; oi_row[rank] = mi; }
      }
      if (nvalid < K && t < K - nvalid) {
        ov_row[nvalid + t] = -__builtin_inff();
        oi_row[nvalid + t] = 0xFFFFFFFFu;
      }
      lbarrier();
    } else {
      float mv[16u];
      u32 vm2 = 0u;
      float tm2 = -__builtin_inff();
#pragma unroll
      for (u32 k = 0u; k < 16u; k++) {
        const u32 p = t + k * NT;
        if (p < nc) {
          const u32 idx = ci_row[p];
          const float v = cv_row[p];
          mv[k] = v;
          if (idx != 0xFFFFFFFFu) {
            vm2 |= (1u << k);
            if (v > tm2) tm2 = v;
          }
        } else {
          mv[k] = -__builtin_inff();
        }
      }
      const u32 tm2_k = vm2 ? ink_r7_fh_f2u(tm2) : 0u;
      const u32 tau2_k = ink_r7_fh_bitonic_kth_256(tm2_k, K);
      const float tau2 = tau2_k ? ink_r7_fh_u2f(tau2_k) : -__builtin_inff();

      u32 gm2 = 0u, em2 = 0u;
      ink_r7_fh_classify16(mv, vm2, tm2_k, tau2_k, tau2, &gm2, &em2);
      const u32 my_gt2 = (u32)__builtin_popcount(gm2);
      const u32 my_eq2 = (u32)__builtin_popcount(em2);
      u32 ptot2 = 0u;
      const u32 poff2 = ink_r7_fh_wg_scan_u32((my_eq2 << 16u) | my_gt2, ink_r7_fh_hlds.wsum[1u], &ptot2);
      const u32 bgt2 = poff2 & 0xFFFFu;
      const u32 beq2 = poff2 >> 16u;
      const u32 tgt2 = ptot2 & 0xFFFFu;
      const u32 teq2 = ptot2 >> 16u;

      if (gm2) {
        u32 pos = bgt2;
#pragma unroll
        for (u32 k = 0u; k < 16u; k++) {
          if (gm2 & (1u << k)) {
            ink_r7_fh_hlds.u.cand.cv[pos] = mv[k];
            ink_r7_fh_hlds.u.cand.ci[pos] = ci_row[t + k * NT];
            pos++;
          }
        }
      }

      const u32 need2 = (tgt2 < K) ? (K - tgt2) : 0u;
      u32 c2 = tgt2;
      if (need2 > 0u && teq2 <= need2) {
        if (em2) {
          u32 pos = tgt2 + beq2;
#pragma unroll
          for (u32 k = 0u; k < 16u; k++) {
            if (em2 & (1u << k)) {
              ink_r7_fh_hlds.u.cand.cv[pos] = mv[k];
              ink_r7_fh_hlds.u.cand.ci[pos] = ci_row[t + k * NT];
              pos++;
            }
          }
        }
        c2 = tgt2 + teq2;
      }
      lbarrier();

      if (need2 > 0u && teq2 > need2) {
        u32 taken2 = 0u;
#pragma clang loop unroll(disable)
        for (u32 k = 0u; k < 16u; k++) {
          if (taken2 < need2) {
            const u32 hit = (em2 >> k) & 1u;
            u32 stot = 0u;
            const u32 soff = ink_r7_fh_wg_scan_u32(hit, ink_r7_fh_hlds.wsum[k & 1u], &stot);
            if (hit && taken2 + soff < need2) {
              const u32 p = t + k * NT;
              const u32 rk = tgt2 + taken2 + soff;
              ov_row[rk] = cv_row[p];
              oi_row[rk] = ci_row[p];
            }
            taken2 += stot;
          }
        }
      }
      ink_r7_fh_rank_write(c2, K, ov_row, oi_row);
      if (tau2_k == 0u && c2 < K && t < K - c2) {
        ov_row[c2 + t] = -__builtin_inff();
        oi_row[c2 + t] = 0xFFFFFFFFu;
      }
      lbarrier();
    }
  }
}

/* ============================================================================
 * Kernel 2a (Bit 88): ie_inkling_ds_base_q4k_idmap_r7
 * 0-VRAM Q4_K output.weight gather for DSpark base[0..T-1][0..n_act-1].
 * ========================================================================== */

KERNEL ie_inkling_ds_base_q4k_idmap_r7(
    const G u32 *qw, const G u16 *qs, const G u8 *xq,
    const G u32 *id_map, G float *base_out,
    u32 full_rows, u32 n_act, u32 nb, u32 T, u32 xs, u32 ys_f32) {
  if (T == 0u || T > INK_R7_FH_TMAX || n_act == 0u || nb == 0u || (nb & 7u) != 0u) return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 s = (wgid() << 5u) + (w << 2u) + (l >> 3u);
  if (s >= n_act) return;

  const u32 tok = id_map ? id_map[s] : s;
  if (tok >= full_rows) return;

  ink_r7_fh_f8 vals;
  ink_r7_fh_q4k_row(qw, qs, xq, tok, full_rows, nb, T, xs, sub, &vals);
  if (sub == 0u) {
#pragma unroll
    for (u32 j = 0u; j < INK_R7_FH_TMAX; j++) {
      if (j < T) base_out[(unsigned long)j * (unsigned long)ys_f32 + (unsigned long)s] = vals[j];
    }
  }
}

/* ============================================================================
 * Kernel 2b (Bit 88): ie_inkling_ds_gemv_p_q4_idmap_zsync_r7
 * 0-VRAM Q4_0 split P gather (128 B/row = 1 exact cache line!) + base_row add.
 * ========================================================================== */

static inline __attribute__((always_inline)) float ink_r7_fh_q4q8_blk(
    const G u32 *row_qw, const G f16 *row_qs, const G u32 *aw,
    const G float *da, const G u32 *asum, u32 b) {
  const u32x4 w4 = ink_r7_fh_ld4(row_qw + (unsigned long)b * 4ul);
  const u32x4 a0 = ink_r7_fh_ld4(aw + (unsigned long)b * 8ul);
  const u32x4 a1 = ink_r7_fh_ld4(aw + (unsigned long)b * 8ul + 4ul);
  const float sc = (float)row_qs[b] * da[b];
  const u32 as = asum[b];
  u32 d = ie_q4q8_word(0u, w4.x, a0.x, a1.x);
  d = ie_q4q8_word(d, w4.y, a0.y, a1.y);
  d = ie_q4q8_word(d, w4.z, a0.z, a1.z);
  d = ie_q4q8_word(d, w4.w, a0.w, a1.w);
  const int dot = (int)(d - 8u * as);
  return sc * (float)dot;
}

KERNEL ie_inkling_ds_gemv_p_q4_idmap_zsync_r7(
    const G u32 *qw, const G f16 *qs, const G u8 *xq,
    const G float *base_row, const G u32 *id_map, G float *logits,
    u32 full_rows, u32 n_act, u32 nb, const G u32 *ds_ctrl) {
#pragma clang fp contract(off)
  if (ds_ctrl && ds_ctrl[1u] != 0u) return;
  if (n_act == 0u || nb == 0u) return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 s = (wgid() << 5u) + (w << 2u) + (l >> 3u);
  if (s >= n_act) return;

  const u32 tok = id_map ? id_map[s] : s;
  if (tok >= full_rows) return;

  const G u32 *row_qw = qw + (unsigned long)tok * (unsigned long)nb * 4ul;
  const G f16 *row_qs = qs + (unsigned long)tok * (unsigned long)nb;
  const G u32 *aw = (const G u32 *)xq;
  const G float *da = (const G float *)(xq + (unsigned long)nb * 32ul);
  const G u32 *asum = (const G u32 *)(xq + (unsigned long)nb * 36ul);

  float acc;
  if (nb == 8u) {
    acc = 0.0f + ink_r7_fh_q4q8_blk(row_qw, row_qs, aw, da, asum, sub);
  } else {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (u32 k0 = 0u; k0 < nb; k0 += 32u) {
      const u32 b0 = k0 + sub;
      if (b0 < nb) a0 = a0 + ink_r7_fh_q4q8_blk(row_qw, row_qs, aw, da, asum, b0);
      const u32 b1 = b0 + 8u;
      if (b1 < nb) a1 = a1 + ink_r7_fh_q4q8_blk(row_qw, row_qs, aw, da, asum, b1);
      const u32 b2 = b0 + 16u;
      if (b2 < nb) a2 = a2 + ink_r7_fh_q4q8_blk(row_qw, row_qs, aw, da, asum, b2);
      const u32 b3 = b0 + 24u;
      if (b3 < nb) a3 = a3 + ink_r7_fh_q4q8_blk(row_qw, row_qs, aw, da, asum, b3);
    }
    acc = (a0 + a2) + (a1 + a3);
  }

  acc = acc + I2F(shl_i(F2I(acc), 4));
  acc = acc + I2F(shl_i(F2I(acc), 2));
  acc = acc + I2F(shl_i(F2I(acc), 1));
  if (sub == 0u) logits[s] = base_row ? (base_row[s] + acc) : acc;
}

/* ============================================================================
 * Kernel 2c (Bit 88): ie_inkling_ds_cap_sample_idmap_zsync_r7
 * Shortlist zsync cap+argmax/sample with canonical id_map[s] tie-breaking.
 * ========================================================================== */

typedef struct {
  union {
    u32 sm[2u][NT];
    struct {
      float cv[INK_R7_FH_CAP];
      u32   ci[INK_R7_FH_CAP];
    } cand;
    struct {
      float wv[8u];
      u32   wi[8u];
      float ws[8u];
    } gr;
  } u;
  u32   wsum[2u][8u];
  float fv[INK_R7_FH_KMAX];
  u32   fi[INK_R7_FH_KMAX];
  float fq[INK_R7_FH_KMAX];
  u32   last;
} ink_r7_fh_dsl_lds_t;

static LDS ink_r7_fh_dsl_lds_t ink_r7_fh_dlds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r7_fh_dsl_lds_t) <= 16384u, "ink_r7_fh_dlds must stay <= 16 KiB");

static inline __attribute__((always_inline)) void ink_r7_fh_dgr_best(float *bv, u32 *bi) {
  const u32 l = lane();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
#pragma unroll
  for (u32 m = 16u; m > 0u; m >>= 1u) {
    const float ov = shfl_xor_f(*bv, m);
    const u32 oi = (u32)__builtin_amdgcn_ds_bpermute((int)((l ^ m) * 4u), (int)*bi);
    if (ink_r7_fh_better(ov, oi, *bv, *bi)) { *bv = ov; *bi = oi; }
  }
  if (l == 0u) { ink_r7_fh_dlds.u.gr.wv[w] = *bv; ink_r7_fh_dlds.u.gr.wi[w] = *bi; }
  lbarrier();
  float v = ink_r7_fh_dlds.u.gr.wv[0u];
  u32 idx = ink_r7_fh_dlds.u.gr.wi[0u];
#pragma clang loop unroll(disable)
  for (u32 j = 1u; j < 8u; j++) {
    if (ink_r7_fh_better(ink_r7_fh_dlds.u.gr.wv[j], ink_r7_fh_dlds.u.gr.wi[j], v, idx)) {
      v = ink_r7_fh_dlds.u.gr.wv[j]; idx = ink_r7_fh_dlds.u.gr.wi[j];
    }
  }
  *bv = v; *bi = idx;
}

static inline __attribute__((always_inline)) u32 ink_r7_fh_dbitonic(u32 v, u32 K) {
  const u32 t = tid();
#pragma clang loop unroll(disable)
  for (u32 size = 2u; size <= 32u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride > 0u; stride >>= 1u) v = ink_r7_fh_wave_step(v, t, stride, a);
  }
  u32 buf_id = 0u;
#pragma clang loop unroll(disable)
  for (u32 size = 64u; size <= 256u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride >= 32u; stride >>= 1u) {
      v = ink_r7_fh_lds_step(v, t, stride, a, ink_r7_fh_dlds.u.sm[buf_id]);
      buf_id ^= 1u;
    }
#pragma clang loop unroll(disable)
    for (u32 stride = 16u; stride > 0u; stride >>= 1u) v = ink_r7_fh_wave_step(v, t, stride, a);
  }
  if (t == K - 1u) ink_r7_fh_dlds.u.sm[0u][0u] = v;
  lbarrier();
  return ink_r7_fh_dlds.u.sm[0u][0u];
}

KERNEL ie_inkling_ds_cap_sample_idmap_zsync_r7(
    const G float *logits, const G u32 *id_map,
    G float *part_v, G u32 *part_i, G float *part_sum,
    G u32 *am_count, G u32 *ds_ctrl, G ink_r7_fh_u64 *rng_state,
    G u32 *draft_ids, G float *draft_probs, G float *draft_entropy,
    G u32 *q_n, G u32 *q_ids, G float *q_probs,
    u32 n_act, u32 ng, u32 K, u32 K_stride, u32 step_idx,
    float cap, float temp, float top_p, float pmin, float surv_a, float surv_b) {
  if (ds_ctrl[1u] != 0u) return;
  const u32 g = wgid(), t = tid();
  if (g >= ng || ng == 0u || n_act == 0u) return;

  /* Mode A: Greedy (temp <= 0.0f) with canonical id_map[s] tie-breaking! */
  if (temp <= 0.0f) {
    const float inv_cap = (cap > 0.0f) ? (1.0f / cap) : 0.0f;
    float bv = -__builtin_inff();
    u32 bi = 0xFFFFFFFFu;
    float se = 0.0f;
    for (u32 s = g * NT + t; s < n_act; s += ng * NT) {
      const float fv = ink_r7_fh_cap_inv(logits[s], cap, inv_cap);
      const u32 tok = id_map ? id_map[s] : s;
      if (fv > bv) {
        se = se * __builtin_expf(bv - fv) + 1.0f;
        bv = fv; bi = tok;
      } else {
        if (fv == bv && tok < bi) bi = tok;
        if (fv > -__builtin_inff()) se += __builtin_expf(fv - bv);
      }
    }
    const float my_bm = bv;
    ink_r7_fh_dgr_best(&bv, &bi);
    float sm = (se > 0.0f) ? (se * __builtin_expf(my_bm - bv)) : 0.0f;
    sm = wave_sum_all(sm);
    if (lane() == 0u) ink_r7_fh_dlds.u.gr.ws[(u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u))] = sm;
    lbarrier();
    float wg_se = 0.0f;
#pragma clang loop unroll(disable)
    for (u32 j = 0u; j < 8u; j++) wg_se += ink_r7_fh_dlds.u.gr.ws[j];

    if (t == 0u) { part_v[g] = bv; part_i[g] = bi; part_sum[g] = wg_se; }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (t == 0u) {
      ink_r7_fh_dlds.last =
          (__scoped_atomic_fetch_add(am_count, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r7_fh_dlds.last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

    bv = -__builtin_inff(); bi = 0xFFFFFFFFu;
    for (u32 u = t; u < ng; u += NT) {
      if (ink_r7_fh_better(part_v[u], part_i[u], bv, bi)) { bv = part_v[u]; bi = part_i[u]; }
    }
    ink_r7_fh_dgr_best(&bv, &bi);
    float S = 0.0f;
    for (u32 u = t; u < ng; u += NT) {
      if (part_sum[u] > 0.0f) S += part_sum[u] * __builtin_expf(part_v[u] - bv);
    }
    S = wave_sum_all(S);
    if (lane() == 0u) ink_r7_fh_dlds.u.gr.ws[(u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u))] = S;
    lbarrier();
    S = 0.0f;
#pragma clang loop unroll(disable)
    for (u32 j = 0u; j < 8u; j++) S += ink_r7_fh_dlds.u.gr.ws[j];

    if (t == 0u) {
      const float prob = (S > 0.0f) ? (1.0f / S) : 0.0f;
      draft_ids[step_idx] = bi;
      draft_probs[step_idx] = prob;
      if (draft_entropy) draft_entropy[step_idx] = 0.0f;
      if (q_n) q_n[step_idx] = 0u;
      ds_ctrl[0u] = bi;
      ds_ctrl[2u] = step_idx + 1u;
      if (prob < pmin) ds_ctrl[1u] = 1u;
      *am_count = 0u;
    }
    return;
  }

  /* Mode B: Probabilistic Wave 6 + canonical id_map[s] mapping (K <= 32, ng <= 64, cs <= 4096). */
  if (K == 0u || K > INK_R7_FH_KMAX || ng > 64u) return;
  const u32 cs = (n_act + ng - 1u) / ng;
  const u32 cs4 = (cs + 3u) & ~3u;
  if (cs4 > 4096u) return;

  float rv[16u];
  u32 vmask = 0u;
  float tmax = -__builtin_inff();
  const u32 nvec = cs4 >> 2u;
  const u32 i0 = g * cs4;
  u32 in_bounds = 0u;
#pragma unroll
  for (u32 q = 0u; q < 4u; q++) {
    const u32 vidx = t + q * NT;
    const u32 base = i0 + (vidx << 2u);
    if (vidx < nvec && base + 4u <= n_act) {
      const f32x4 v4 = ((const G struct ink_r7_fh_f32x4p *)(logits + base))->v;
      rv[4u * q + 0u] = v4.x;
      rv[4u * q + 1u] = v4.y;
      rv[4u * q + 2u] = v4.z;
      rv[4u * q + 3u] = v4.w;
      in_bounds |= (0xFu << (4u * q));
    } else if (vidx < nvec && base < n_act) {
#pragma unroll
      for (u32 e = 0u; e < 4u; e++) {
        const u32 ok = (base + e < n_act);
        rv[4u * q + e] = ok ? logits[base + e] : -__builtin_inff();
        if (ok) in_bounds |= (1u << (4u * q + e));
      }
    } else {
      rv[4u * q + 0u] = rv[4u * q + 1u] = rv[4u * q + 2u] = rv[4u * q + 3u] = -__builtin_inff();
    }
  }
#pragma unroll
  for (u32 k = 0u; k < 16u; k++) {
    if (in_bounds & (1u << k)) {
      const float v = rv[k];
      if (v == v) { vmask |= (1u << k); if (v > tmax) tmax = v; }
    }
  }

  const u32 tmax_k = vmask ? ink_r7_fh_f2u(tmax) : 0u;
  const u32 tau_k = ink_r7_fh_dbitonic(tmax_k, K);
  const float tau = tau_k ? ink_r7_fh_u2f(tau_k) : -__builtin_inff();
  u32 gt_mask = 0u, eq_mask = 0u;
  ink_r7_fh_classify16(rv, vmask, tmax_k, tau_k, tau, &gt_mask, &eq_mask);

  const u32 my_gt = (u32)__builtin_popcount(gt_mask);
  const u32 my_eq = (u32)__builtin_popcount(eq_mask);
  u32 ptot = 0u;
  const u32 poff = ink_r7_fh_wg_scan_u32((my_eq << 16u) | my_gt, ink_r7_fh_dlds.wsum[1u], &ptot);
  const u32 base_gt = poff & 0xFFFFu, base_eq = poff >> 16u;
  const u32 total_gt = ptot & 0xFFFFu, total_eq = ptot >> 16u;

  if (gt_mask) {
    u32 pos = base_gt;
#pragma unroll
    for (u32 q = 0u; q < 4u; q++) {
      const u32 base = i0 + ((q * NT + t) << 2u);
#pragma unroll
      for (u32 e = 0u; e < 4u; e++) {
        if (gt_mask & (1u << (4u * q + e))) {
          const u32 sl = base + e;
          ink_r7_fh_dlds.u.cand.cv[pos] = rv[4u * q + e];
          ink_r7_fh_dlds.u.cand.ci[pos] = id_map ? id_map[sl] : sl;
          pos++;
        }
      }
    }
  }
  const u32 need = (total_gt < K) ? (K - total_gt) : 0u;
  u32 cand_n = total_gt;
  if (need > 0u && total_eq <= need) {
    if (eq_mask) {
      u32 pos = total_gt + base_eq;
#pragma unroll
      for (u32 q = 0u; q < 4u; q++) {
        const u32 base = i0 + ((q * NT + t) << 2u);
#pragma unroll
        for (u32 e = 0u; e < 4u; e++) {
          if (eq_mask & (1u << (4u * q + e))) {
            const u32 sl = base + e;
            ink_r7_fh_dlds.u.cand.cv[pos] = rv[4u * q + e];
            ink_r7_fh_dlds.u.cand.ci[pos] = id_map ? id_map[sl] : sl;
            pos++;
          }
        }
      }
    }
    cand_n = total_gt + total_eq;
  }
  lbarrier();

  /* Write canonical (val, tok) segment winners to part_v / part_i (or directly to fv/fi if ng==1). */
  G float *seg_v = (ng > 1u) ? (part_v + (unsigned long)g * (unsigned long)K) : (G float *)0;
  G u32   *seg_i = (ng > 1u) ? (part_i + (unsigned long)g * (unsigned long)K) : (G u32 *)0;
  if (need > 0u && total_eq > need) {
    u32 taken = 0u;
#pragma clang loop unroll(disable)
    for (u32 q = 0u; q < 4u; q++) {
      if (taken < need) {
        const u32 m4 = (eq_mask >> (4u * q)) & 0xFu;
        u32 stot = 0u;
        const u32 soff = ink_r7_fh_wg_scan_u32((u32)__builtin_popcount(m4), ink_r7_fh_dlds.wsum[q & 1u], &stot);
        if (m4) {
          u32 off = taken + soff;
          const u32 base = i0 + ((q * NT + t) << 2u);
#pragma unroll
          for (u32 e = 0u; e < 4u; e++) {
            if (m4 & (1u << e)) {
              if (off < need) {
                const u32 sl = base + e;
                const float val = logits[sl];
                const u32 tok = id_map ? id_map[sl] : sl;
                if (ng > 1u) { seg_v[total_gt + off] = val; seg_i[total_gt + off] = tok; }
                else { ink_r7_fh_dlds.fv[total_gt + off] = val; ink_r7_fh_dlds.fi[total_gt + off] = tok; }
              }
              off++;
            }
          }
        }
        taken += stot;
      }
    }
  }
  for (u32 j = t; j < cand_n; j += NT) {
    const float a = ink_r7_fh_dlds.u.cand.cv[j];
    const u32 ai = ink_r7_fh_dlds.u.cand.ci[j];
    u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
    for (u32 u = 0u; u < cand_n; u++) {
      const float b = ink_r7_fh_dlds.u.cand.cv[u];
      rank += (u32)((b > a) | ((b == a) & (ink_r7_fh_dlds.u.cand.ci[u] < ai)));
    }
    if (rank < K) {
      if (ng > 1u) { seg_v[rank] = a; seg_i[rank] = ai; }
      else { ink_r7_fh_dlds.fv[rank] = a; ink_r7_fh_dlds.fi[rank] = ai; }
    }
  }
  if (tau_k == 0u && cand_n < K && t < K - cand_n) {
    if (ng > 1u) { seg_v[cand_n + t] = -__builtin_inff(); seg_i[cand_n + t] = 0xFFFFFFFFu; }
    else { ink_r7_fh_dlds.fv[cand_n + t] = -__builtin_inff(); ink_r7_fh_dlds.fi[cand_n + t] = 0xFFFFFFFFu; }
  }

  if (ng > 1u) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (t == 0u) {
      ink_r7_fh_dlds.last =
          (__scoped_atomic_fetch_add(am_count, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r7_fh_dlds.last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (t == 0u) *am_count = 0u;

    const u32 nc = ng * K;
    if (nc <= NT) {
      float mv = -__builtin_inff();
      u32 mi = 0xFFFFFFFFu;
      if (t < nc) {
        mv = part_v[t]; mi = part_i[t];
        ink_r7_fh_dlds.u.cand.cv[t] = mv;
        ink_r7_fh_dlds.u.cand.ci[t] = mi;
      }
      const u32 is_val = (t < nc && mi != 0xFFFFFFFFu) ? 1u : 0u;
      const u32 wcnt = (u32)__builtin_popcount(__builtin_amdgcn_ballot_w32(is_val != 0u));
      if (lane() == 0u) ink_r7_fh_dlds.wsum[0u][(u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u))] = wcnt;
      lbarrier();
      u32 nvalid = 0u;
#pragma clang loop unroll(disable)
      for (u32 j = 0u; j < 8u; j++) nvalid += ink_r7_fh_dlds.wsum[0u][j];
      if (is_val) {
        u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
        for (u32 u = 0u; u < nc; u++) {
          const u32 ui = ink_r7_fh_dlds.u.cand.ci[u];
          if (ui != 0xFFFFFFFFu) {
            const float b = ink_r7_fh_dlds.u.cand.cv[u];
            rank += (u32)((b > mv) | ((b == mv) & (ui < mi)));
          }
        }
        if (rank < K) { ink_r7_fh_dlds.fv[rank] = mv; ink_r7_fh_dlds.fi[rank] = mi; }
      }
      if (nvalid < K && t < K - nvalid) {
        ink_r7_fh_dlds.fv[nvalid + t] = -__builtin_inff();
        ink_r7_fh_dlds.fi[nvalid + t] = 0xFFFFFFFFu;
      }
    } else {
      float mv[16u];
      u32 vm2 = 0u;
      float tm2 = -__builtin_inff();
#pragma unroll
      for (u32 k = 0u; k < 16u; k++) {
        const u32 p = t + k * NT;
        if (p < nc) {
          const u32 idx = part_i[p];
          const float v = part_v[p];
          mv[k] = v;
          if (idx != 0xFFFFFFFFu) { vm2 |= (1u << k); if (v > tm2) tm2 = v; }
        } else {
          mv[k] = -__builtin_inff();
        }
      }
      const u32 tm2_k = vm2 ? ink_r7_fh_f2u(tm2) : 0u;
      const u32 tau2_k = ink_r7_fh_dbitonic(tm2_k, K);
      const float tau2 = tau2_k ? ink_r7_fh_u2f(tau2_k) : -__builtin_inff();
      u32 gm2 = 0u, em2 = 0u;
      ink_r7_fh_classify16(mv, vm2, tm2_k, tau2_k, tau2, &gm2, &em2);
      const u32 my_gt2 = (u32)__builtin_popcount(gm2);
      const u32 my_eq2 = (u32)__builtin_popcount(em2);
      u32 ptot2 = 0u;
      const u32 poff2 = ink_r7_fh_wg_scan_u32((my_eq2 << 16u) | my_gt2, ink_r7_fh_dlds.wsum[1u], &ptot2);
      const u32 bgt2 = poff2 & 0xFFFFu, beq2 = poff2 >> 16u;
      const u32 tgt2 = ptot2 & 0xFFFFu, teq2 = ptot2 >> 16u;

      if (gm2) {
        u32 pos = bgt2;
#pragma unroll
        for (u32 k = 0u; k < 16u; k++) {
          if (gm2 & (1u << k)) {
            ink_r7_fh_dlds.u.cand.cv[pos] = mv[k];
            ink_r7_fh_dlds.u.cand.ci[pos] = part_i[t + k * NT];
            pos++;
          }
        }
      }
      const u32 need2 = (tgt2 < K) ? (K - tgt2) : 0u;
      u32 c2 = tgt2;
      if (need2 > 0u && teq2 <= need2) {
        if (em2) {
          u32 pos = tgt2 + beq2;
#pragma unroll
          for (u32 k = 0u; k < 16u; k++) {
            if (em2 & (1u << k)) {
              ink_r7_fh_dlds.u.cand.cv[pos] = mv[k];
              ink_r7_fh_dlds.u.cand.ci[pos] = part_i[t + k * NT];
              pos++;
            }
          }
        }
        c2 = tgt2 + teq2;
      }
      lbarrier();

      if (need2 > 0u && teq2 > need2) {
        u32 taken2 = 0u;
#pragma clang loop unroll(disable)
        for (u32 k = 0u; k < 16u; k++) {
          if (taken2 < need2) {
            const u32 hit = (em2 >> k) & 1u;
            u32 stot = 0u;
            const u32 soff = ink_r7_fh_wg_scan_u32(hit, ink_r7_fh_dlds.wsum[k & 1u], &stot);
            if (hit && taken2 + soff < need2) {
              const u32 p = t + k * NT;
              const u32 rk = tgt2 + taken2 + soff;
              ink_r7_fh_dlds.fv[rk] = part_v[p];
              ink_r7_fh_dlds.fi[rk] = part_i[p];
            }
            taken2 += stot;
          }
        }
      }
      for (u32 j = t; j < c2; j += NT) {
        const float a = ink_r7_fh_dlds.u.cand.cv[j];
        const u32 ai = ink_r7_fh_dlds.u.cand.ci[j];
        u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
        for (u32 u = 0u; u < c2; u++) {
          const float b = ink_r7_fh_dlds.u.cand.cv[u];
          rank += (u32)((b > a) | ((b == a) & (ink_r7_fh_dlds.u.cand.ci[u] < ai)));
        }
        if (rank < K) { ink_r7_fh_dlds.fv[rank] = a; ink_r7_fh_dlds.fi[rank] = ai; }
      }
      if (tau2_k == 0u && c2 < K && t < K - c2) {
        ink_r7_fh_dlds.fv[c2 + t] = -__builtin_inff();
        ink_r7_fh_dlds.fi[c2 + t] = 0xFFFFFFFFu;
      }
    }
  }
  lbarrier();

  /* Stage 3 (Wave 0): Softmax(temp) + top_p nucleus + ForeSpark H(q) + xorshift64*. */
  if (t < 32u) {
    const u32 c = t;
    const u32 ci = (c < K) ? ink_r7_fh_dlds.fi[c] : 0xFFFFFFFFu;
    const float cv_raw = (c < K) ? ink_r7_fh_dlds.fv[c] : -__builtin_inff();
    const float v0_raw = ink_r7_fh_dlds.fv[0u];
    const u32 valid = (c < K && ci != 0xFFFFFFFFu && cv_raw == cv_raw && v0_raw == v0_raw) ? 1u : 0u;
    const float cv = valid ? ink_r7_fh_cap(cv_raw, cap) : -__builtin_inff();
    const float v0 = (v0_raw == v0_raw) ? ink_r7_fh_cap(v0_raw, cap) : v0_raw;

    const float wc = valid ? __builtin_expf((cv - v0) / temp) : 0.0f;
    const float z = wave_sum_all(wc);

    float cum = wc;
#pragma clang loop unroll(disable)
    for (u32 d = 1u; d < 32u; d <<= 1u) {
      const u32 src = (c >= d) ? (c - d) : 0u;
      const u32 bits = (u32)__builtin_amdgcn_ds_bpermute((int)(src * 4u), (int)__builtin_bit_cast(u32, cum));
      if (c >= d) cum += __builtin_bit_cast(float, bits);
    }
    const float cum_excl = cum - wc;
    const u32 in_nuc = (valid && (c == 0u || top_p >= 1.0f || cum_excl < top_p * z)) ? 1u : 0u;
    const u32 m = (u32)__builtin_popcount(__builtin_amdgcn_ballot_w32(in_nuc != 0u));

    const float zq = wave_sum_all(in_nuc ? wc : 0.0f);
    const float qc = (in_nuc && zq > 0.0f) ? (wc / zq) : ((c == 0u && valid) ? 1.0f : 0.0f);
    const float H = wave_sum_all(qc > 0.0f ? (-qc * __builtin_logf(qc)) : 0.0f);

    ink_r7_fh_dlds.fq[c] = qc;
    if (c < m && c < K_stride) {
      if (q_ids) q_ids[(unsigned long)step_idx * (unsigned long)K_stride + (unsigned long)c] = ci;
      if (q_probs) q_probs[(unsigned long)step_idx * (unsigned long)K_stride + (unsigned long)c] = qc;
    }

    if (c == 0u) {
      u32 chosen = 0u;
      u32 chosen_tok = (m > 0u) ? ink_r7_fh_dlds.fi[0u] : 0u;
      float chosen_p = (m > 0u) ? ink_r7_fh_dlds.fq[0u] : 0.0f;
      if (m > 1u && rng_state) {
        ink_r7_fh_u64 rs = *rng_state;
        if (rs == 0ull) rs = 0x9E3779B97F4A7C15ull ^ ((ink_r7_fh_u64)step_idx + 1ull);
        rs ^= rs >> 12u; rs ^= rs << 25u; rs ^= rs >> 27u;
        *rng_state = rs;
        const ink_r7_fh_u64 rnd = rs * 2685821657736338717ull;
        float u = (float)((u32)(rnd >> 40u)) * (1.0f / 16777216.0f);
        u32 last_nz = 0u;
#pragma clang loop unroll(disable)
        for (u32 j = 0u; j < m; j++) {
          const float pj = ink_r7_fh_dlds.fq[j];
          if (pj > 0.0f) last_nz = j;
          u -= pj;
          if (u <= 0.0f && pj > 0.0f) { chosen = j; break; }
          if (j + 1u == m) chosen = last_nz;
        }
        chosen_tok = ink_r7_fh_dlds.fi[chosen];
        chosen_p = ink_r7_fh_dlds.fq[chosen];
      }

      draft_ids[step_idx] = chosen_tok;
      draft_probs[step_idx] = chosen_p;
      if (draft_entropy) draft_entropy[step_idx] = H;
      if (q_n) q_n[step_idx] = m;
      ds_ctrl[0u] = chosen_tok;
      ds_ctrl[2u] = step_idx + 1u;

      if (surv_a > 0.0f) {
        float alpha_hat = surv_a - surv_b * H;
        if (alpha_hat < 0.0f) alpha_hat = 0.0f;
        if (alpha_hat > 1.0f) alpha_hat = 1.0f;
        const u32 prev_bits = ds_ctrl[3u];
        const float prev_surv = (step_idx == 0u) ? 1.0f : __builtin_bit_cast(float, prev_bits);
        const float cum_surv = prev_surv * alpha_hat;
        ds_ctrl[3u] = __builtin_bit_cast(u32, cum_surv);
        if (cum_surv < pmin) ds_ctrl[1u] = 1u;
      } else if (pmin > 0.0f && chosen_p < pmin) {
        ds_ctrl[1u] = 1u;
      }
    }
  }
}

#undef INK_R7_FH_CAP
#undef INK_R7_FH_NGMAX
#undef INK_R7_FH_KMAX
#undef INK_R7_FH_TMAX

#endif /* IE_MEGA */
