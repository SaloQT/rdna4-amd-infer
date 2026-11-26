/* ============================================================================
 * Inkling-2 Round 7: DSpark Wave 6 Probabilistic ZSync & Split-KV Long Attention
 * Target: RX 9070 XT (gfx1201, wave32, NT=256)
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r7_ds_ /
 * INK_R7_DS_ and all macros are #undef'd at EOF.
 *
 * Exported kernels:
 *
 * 1. KERNEL ie_inkling_ds_cap_sample_zsync_r7(...)
 *    - Drop-in Wave 6 (W6-1 + W6-2) upgrade to ie_dspark_cap_argmax_zsync_r5!
 *      Preserves exact ds_ctrl[4] = {prev_tok, stop_flag, n_ran, cum_surv_bits}
 *      ABI so ALL existing R5 zsync Markov kernels stay unchanged.
 *    - Early exits immediately if ds_ctrl[1] != 0u.
 *    - temp <= 0.0f (Greedy mode):
 *        Single-pass vectorized f32x4 softcap + online (max, sum_exp) + exact
 *        lowest-index tie-breaking (100% bitwise-exact best_id vs R5), updates
 *        ds_ctrl[0]=best_id, ds_ctrl[2]=step_idx+1, ds_ctrl[1]=(prob < pmin).
 *        Supports 1 <= ng <= 128.
 *    - temp > 0.0f (Probabilistic W6-1 + ForeSpark W6-2 mode):
 *        * Single-pass vectorized f32x4 softcap + zero-overflow Bitonic-256
 *          top-K (1 <= K <= 32, 1 <= ng <= 64, cs <= 4096 -> ng=64 for V=200058).
 *        * Last WG merges nc = ng * K <= 2048 candidates -> sorted top-K in LDS.
 *        * Wave 0 (32 lanes, < 0.5 us) computes exact softmax at temperature
 *          `temp`, exact `top_p` nucleus cutoff `m` (matching gen.c:782),
 *          normalized `q[0..m-1]`, and ForeSpark Shannon entropy
 *          `H = -sum q[c] * log(q[c])`.
 *        * Thread 0 draws `chosen ~ q` using xorshift64* on `*rng_state`
 *          (exact multiplier 2685821657736338717ULL matching gen.c:35/735),
 *          writes `q_n[step_idx]=m`, `q_ids[step_idx*K_stride..]`,
 *          `q_probs[step_idx*K_stride..]`, `draft_ids[step_idx]=tok`,
 *          `draft_probs[step_idx]=q[chosen]`, `draft_entropy[step_idx]=H`,
 *          sets `ds_ctrl[0]=tok` (next Markov RNN step conditions on sampled tok!),
 *          `ds_ctrl[2]=step_idx+1`.
 *        * Early-stop cutoff:
 *            - If surv_a > 0.0f (ForeSpark W6-2):
 *                alpha_hat = clamp(surv_a - surv_b * H, 0.0f, 1.0f)
 *                cum_surv  = (step_idx == 0 ? 1.0f : ds_ctrl[3]) * alpha_hat
 *                ds_ctrl[3] = bit_cast(u32, cum_surv)
 *                if (cum_surv < pmin) ds_ctrl[1] = 1u
 *            - Else:
 *                if (pmin > 0.0f && q[chosen] < pmin) ds_ctrl[1] = 1u
 *        * All per-step outputs pack contiguously in zbuf -> 1 single D2H copy
 *          populates d[1..nd] AND qnode[0..nd-1] for exact spec_rej_step()!
 *
 * 2. KERNEL ie_inkling_ds_attn_long_r7(...)
 *    - Split-KV GQA-3 x Block-Token-8 vectorized dwordx4 attention for DSpark
 *      (specifically targets layer 1 Global which hit 1.45 ms = 40% of ds_draft
 *      at 16K in R00 trace, plus SWA layers!).
 *    - Eliminates the 24x redundant KV cache reads of ie_dspark_attn_r4:
 *        * 1 WG owns 1 KV head kh x 1 split sp x up to 8 block tokens (1 wave
 *          per block token, holding ALL 3 GQA Q heads + outputs in 30 VGPRs!).
 *        * Cooperative 128-bit global_load_b128 (u32x4) loads BK=16 context KV
 *          positions into padded 8,704 B LDS -> shared across ALL 24 (gqa, bt)
 *          queries!
 *        * Zero integer modulo inside the inner loop (single initial p0 % kv_cap
 *          per split, then conditional wrap).
 *        * Bidirectional f32 block KV [0..block_size-1] included in split sp==0.
 *        * Single-launch Split-KV: last arriving split WG for (btile, kh)
 *          performs exact 2-pass online softmax merge across nsplit partials and
 *          resets count[btile * n_kv + kh] = 0u.
 *    - Grid: (n_kv * nsplit, (block_size + 7u) >> 3u, 1), block: (256, 1, 1).
 *    - Host guard: hd == 128u && n_kv > 0u && n_head == n_kv * 3u &&
 *      block_size > 0u && nsplit >= 1u && (qs & 15u) == 0u && (ks & 15u) == 0u &&
 *      (os & 15u) == 0u.
 *    - Scratch (when nsplit > 1):
 *        part:  block_size * n_head * nsplit * 132 floats (16-B aligned slots).
 *        count: ((block_size + 7u) >> 3u) * n_kv zero-init u32 counters (auto-reset).
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R7_DS_KMAX  32u
#define INK_R7_DS_NGMAX 64u
#define INK_R7_DS_CSMAX 4096u
#define INK_R7_DS_CAP   1024u
#define INK_R7_DS_HD    128u
#define INK_R7_DS_BK    16u
#define INK_R7_DS_LSTR  136u /* 128 + 8 f16 pad = 272 B (16-B aligned, low bank conflict) */
#define INK_R7_DS_PSTR  132u /* 128 acc + m + l + 2 pad = 528 B (16-B aligned) */

typedef unsigned long long ink_r7_ds_u64;
typedef _Float16 ink_r7_ds_f16x4 __attribute__((ext_vector_type(4)));

struct __attribute__((packed, aligned(4))) ink_r7_ds_f32x4_a4 {
  f32x4 v;
};

struct __attribute__((packed, aligned(4))) ink_r7_ds_u32x4_a4 {
  u32x4 v;
};

static inline __attribute__((always_inline)) float ink_r7_ds_cap(float v, float cap) {
  return (cap > 0.0f) ? (cap * ie_tanhf_s(v / cap)) : v;
}

static inline __attribute__((always_inline)) float ink_r7_ds_cap_inv(float v, float cap, float inv_cap) {
  return (cap > 0.0f) ? (cap * ie_tanhf_s(v * inv_cap)) : v;
}

static inline __attribute__((always_inline)) int ink_r7_ds_better(float v, u32 i, float bv, u32 bi) {
  return (v > bv) || (v == bv && i < bi);
}

static inline __attribute__((always_inline)) u32 ink_r7_ds_f2u(float f) {
  u32 u = __builtin_bit_cast(u32, f);
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static inline __attribute__((always_inline)) float ink_r7_ds_u2f(u32 k) {
  const u32 u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  return __builtin_bit_cast(float, u);
}

static inline __attribute__((always_inline)) u32 ink_r7_ds_cswap(u32 v, u32 ov, u32 hi, u32 asc) {
  return (hi ^ asc) ? (v < ov ? v : ov) : (v > ov ? v : ov);
}

static inline __attribute__((always_inline)) u32 ink_r7_ds_wave_step(u32 v, u32 t, u32 stride, u32 asc) {
  const u32 ov = (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ stride) * 4u), (int)v);
  return ink_r7_ds_cswap(v, ov, (u32)((t & stride) != 0u), asc);
}

static inline __attribute__((always_inline)) u32 ink_r7_ds_lds_step(
    u32 v, u32 t, u32 stride, u32 asc, LDSP u32 *buf) {
  buf[t] = v;
  lbarrier();
  const u32 ov = buf[t ^ stride];
  return ink_r7_ds_cswap(v, ov, (u32)((t & stride) != 0u), asc);
}

/* ============================================================================
 * Kernel 1: ie_inkling_ds_cap_sample_zsync_r7
 * ========================================================================== */

typedef struct {
  union {
    u32 sm[2u][NT];
    struct {
      float cv[INK_R7_DS_CAP];
      u32   ci[INK_R7_DS_CAP];
    } cand;
    struct {
      float wv[8u];
      u32   wi[8u];
      float ws[8u];
    } gr;
  } u;
  u32   wsum[2u][8u];
  float fv[INK_R7_DS_KMAX];
  u32   fi[INK_R7_DS_KMAX];
  float fq[INK_R7_DS_KMAX];
  u32   last;
} ink_r7_ds_samp_lds_t;

static LDS ink_r7_ds_samp_lds_t ink_r7_ds_samp_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r7_ds_samp_lds_t) <= 16384u, "ink_r7_ds_samp_lds must stay <= 16 KiB");

static inline __attribute__((always_inline)) u32 ink_r7_ds_bitonic_kth_256(u32 v, u32 K) {
  const u32 t = tid();
#pragma clang loop unroll(disable)
  for (u32 size = 2u; size <= 32u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride > 0u; stride >>= 1u) {
      v = ink_r7_ds_wave_step(v, t, stride, a);
    }
  }
  u32 buf_id = 0u;
#pragma clang loop unroll(disable)
  for (u32 size = 64u; size <= 256u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride >= 32u; stride >>= 1u) {
      v = ink_r7_ds_lds_step(v, t, stride, a, ink_r7_ds_samp_lds.u.sm[buf_id]);
      buf_id ^= 1u;
    }
#pragma clang loop unroll(disable)
    for (u32 stride = 16u; stride > 0u; stride >>= 1u) {
      v = ink_r7_ds_wave_step(v, t, stride, a);
    }
  }
  if (t == K - 1u) ink_r7_ds_samp_lds.u.sm[0u][0u] = v;
  lbarrier();
  return ink_r7_ds_samp_lds.u.sm[0u][0u];
}

static inline __attribute__((always_inline)) u32 ink_r7_ds_wg_scan_u32(
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

static inline __attribute__((always_inline)) void ink_r7_ds_classify16(
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

static inline __attribute__((always_inline)) void ink_r7_ds_rank_write_g(
    u32 c, u32 K, G float *out_v, G u32 *out_i) {
  for (u32 j = tid(); j < c; j += NT) {
    const float a = ink_r7_ds_samp_lds.u.cand.cv[j];
    const u32 ai = ink_r7_ds_samp_lds.u.cand.ci[j];
    u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
    for (u32 u = 0u; u < c; u++) {
      const float b = ink_r7_ds_samp_lds.u.cand.cv[u];
      rank += (u32)((b > a) | ((b == a) & (ink_r7_ds_samp_lds.u.cand.ci[u] < ai)));
    }
    if (rank < K) {
      out_v[rank] = a;
      out_i[rank] = ai;
    }
  }
}

static inline __attribute__((always_inline)) void ink_r7_ds_rank_write_lds(u32 c, u32 K) {
  for (u32 j = tid(); j < c; j += NT) {
    const float a = ink_r7_ds_samp_lds.u.cand.cv[j];
    const u32 ai = ink_r7_ds_samp_lds.u.cand.ci[j];
    u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
    for (u32 u = 0u; u < c; u++) {
      const float b = ink_r7_ds_samp_lds.u.cand.cv[u];
      rank += (u32)((b > a) | ((b == a) & (ink_r7_ds_samp_lds.u.cand.ci[u] < ai)));
    }
    if (rank < K) {
      ink_r7_ds_samp_lds.fv[rank] = a;
      ink_r7_ds_samp_lds.fi[rank] = ai;
    }
  }
}

static inline __attribute__((always_inline)) void ink_r7_ds_gr_wg_best(float *bv, u32 *bi) {
  const u32 l = lane();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
#pragma unroll
  for (u32 m = 16u; m > 0u; m >>= 1u) {
    const float ov = shfl_xor_f(*bv, m);
    const u32 oi = (u32)__builtin_amdgcn_ds_bpermute((int)((l ^ m) * 4u), (int)*bi);
    if (ink_r7_ds_better(ov, oi, *bv, *bi)) {
      *bv = ov;
      *bi = oi;
    }
  }
  if (l == 0u) {
    ink_r7_ds_samp_lds.u.gr.wv[w] = *bv;
    ink_r7_ds_samp_lds.u.gr.wi[w] = *bi;
  }
  lbarrier();
  float v = ink_r7_ds_samp_lds.u.gr.wv[0u];
  u32 idx = ink_r7_ds_samp_lds.u.gr.wi[0u];
#pragma clang loop unroll(disable)
  for (u32 j = 1u; j < 8u; j++) {
    const float jv = ink_r7_ds_samp_lds.u.gr.wv[j];
    const u32 ji = ink_r7_ds_samp_lds.u.gr.wi[j];
    if (ink_r7_ds_better(jv, ji, v, idx)) {
      v = jv;
      idx = ji;
    }
  }
  *bv = v;
  *bi = idx;
}

KERNEL ie_inkling_ds_cap_sample_zsync_r7(
    const G float *logits,
    G float *part_v, G u32 *part_i, G float *part_sum,
    G u32 *am_count, G u32 *ds_ctrl, G ink_r7_ds_u64 *rng_state,
    G u32 *draft_ids, G float *draft_probs, G float *draft_entropy,
    G u32 *q_n, G u32 *q_ids, G float *q_probs,
    u32 V, u32 ng, u32 K, u32 K_stride, u32 step_idx,
    float cap, float temp, float top_p, float pmin, float surv_a, float surv_b) {
  /* Zero-sync early exit if a prior step hit the stopping threshold. */
  if (ds_ctrl[1u] != 0u) return;

  const u32 g = wgid();
  const u32 t = tid();
  if (g >= ng || ng == 0u || V == 0u) return;

  /* ------------------------------------------------------------------------
   * Mode A: Greedy (temp <= 0.0f) -- exact bitwise parity with R5 zsync.
   * ---------------------------------------------------------------------- */
  if (temp <= 0.0f) {
    const float inv_cap = (cap > 0.0f) ? (1.0f / cap) : 0.0f;
    float bv = -__builtin_inff();
    u32 bi = 0xFFFFFFFFu;
    float se = 0.0f;

    const u32 n4 = V >> 2u;
    const u32 stride4 = ng * NT;
    for (u32 v4i = g * NT + t; v4i < n4; v4i += stride4) {
      const u32 i0 = v4i << 2u;
      const f32x4 raw = ((const G struct ink_r7_ds_f32x4_a4 *)(logits + i0))->v;
      const float f0 = ink_r7_ds_cap_inv(raw.x, cap, inv_cap);
      const float f1 = ink_r7_ds_cap_inv(raw.y, cap, inv_cap);
      const float f2 = ink_r7_ds_cap_inv(raw.z, cap, inv_cap);
      const float f3 = ink_r7_ds_cap_inv(raw.w, cap, inv_cap);

      if (f0 > bv) { se = se * __builtin_expf(bv - f0) + 1.0f; bv = f0; bi = i0; }
      else { if (f0 == bv && i0 < bi) bi = i0; if (f0 > -__builtin_inff()) se += __builtin_expf(f0 - bv); }

      if (f1 > bv) { se = se * __builtin_expf(bv - f1) + 1.0f; bv = f1; bi = i0 + 1u; }
      else { if (f1 == bv && (i0 + 1u) < bi) bi = i0 + 1u; if (f1 > -__builtin_inff()) se += __builtin_expf(f1 - bv); }

      if (f2 > bv) { se = se * __builtin_expf(bv - f2) + 1.0f; bv = f2; bi = i0 + 2u; }
      else { if (f2 == bv && (i0 + 2u) < bi) bi = i0 + 2u; if (f2 > -__builtin_inff()) se += __builtin_expf(f2 - bv); }

      if (f3 > bv) { se = se * __builtin_expf(bv - f3) + 1.0f; bv = f3; bi = i0 + 3u; }
      else { if (f3 == bv && (i0 + 3u) < bi) bi = i0 + 3u; if (f3 > -__builtin_inff()) se += __builtin_expf(f3 - bv); }
    }
    const u32 tail_base = n4 << 2u;
    if (g == 0u && tail_base + t < V) {
      const u32 idx = tail_base + t;
      const float fv = ink_r7_ds_cap_inv(logits[idx], cap, inv_cap);
      if (fv > bv) { se = se * __builtin_expf(bv - fv) + 1.0f; bv = fv; bi = idx; }
      else { if (fv == bv && idx < bi) bi = idx; if (fv > -__builtin_inff()) se += __builtin_expf(fv - bv); }
    }

    const float my_bm = bv;
    ink_r7_ds_gr_wg_best(&bv, &bi);
    float s = (se > 0.0f) ? (se * __builtin_expf(my_bm - bv)) : 0.0f;
    s = wave_sum_all(s);
    if (lane() == 0u) ink_r7_ds_samp_lds.u.gr.ws[(u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u))] = s;
    lbarrier();
    float wg_se = 0.0f;
#pragma clang loop unroll(disable)
    for (u32 j = 0u; j < 8u; j++) wg_se += ink_r7_ds_samp_lds.u.gr.ws[j];

    if (t == 0u) {
      part_v[g] = bv;
      part_i[g] = bi;
      part_sum[g] = wg_se;
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (t == 0u) {
      ink_r7_ds_samp_lds.last =
          (__scoped_atomic_fetch_add(am_count, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r7_ds_samp_lds.last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

    bv = -__builtin_inff();
    bi = 0xFFFFFFFFu;
    for (u32 u = t; u < ng; u += NT) {
      const float vu = part_v[u];
      const u32 iu = part_i[u];
      if (ink_r7_ds_better(vu, iu, bv, bi)) { bv = vu; bi = iu; }
    }
    ink_r7_ds_gr_wg_best(&bv, &bi);

    float S = 0.0f;
    for (u32 u = t; u < ng; u += NT) {
      const float sg = part_sum[u];
      if (sg > 0.0f) S += sg * __builtin_expf(part_v[u] - bv);
    }
    S = wave_sum_all(S);
    if (lane() == 0u) ink_r7_ds_samp_lds.u.gr.ws[(u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u))] = S;
    lbarrier();
    S = 0.0f;
#pragma clang loop unroll(disable)
    for (u32 j = 0u; j < 8u; j++) S += ink_r7_ds_samp_lds.u.gr.ws[j];

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

  /* ------------------------------------------------------------------------
   * Mode B: Probabilistic Wave 6 (temp > 0.0f, 1 <= K <= 32, ng <= 64).
   * Single-pass f32x4 load + inline softcap + zero-overflow Bitonic-256 top-K
   * -> exact top_p nucleus + ForeSpark H(q_i) + GPU xorshift64* sample.
   * ---------------------------------------------------------------------- */
  if (K == 0u || K > INK_R7_DS_KMAX || ng > INK_R7_DS_NGMAX) return;
  const u32 cs = (V + ng - 1u) / ng;
  const u32 cs4 = (cs + 3u) & ~3u;
  if (cs4 > INK_R7_DS_CSMAX) return;

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
    if (vidx < nvec && base + 4u <= V) {
      const f32x4 v4 = ((const G struct ink_r7_ds_f32x4_a4 *)(logits + base))->v;
      rv[4u * q + 0u] = v4.x;
      rv[4u * q + 1u] = v4.y;
      rv[4u * q + 2u] = v4.z;
      rv[4u * q + 3u] = v4.w;
      in_bounds |= (0xFu << (4u * q));
    } else if (vidx < nvec && base < V) {
#pragma unroll
      for (u32 e = 0u; e < 4u; e++) {
        const u32 ok = (base + e < V);
        rv[4u * q + e] = ok ? logits[base + e] : -__builtin_inff();
        if (ok) in_bounds |= (1u << (4u * q + e));
      }
    } else {
      rv[4u * q + 0u] = -__builtin_inff();
      rv[4u * q + 1u] = -__builtin_inff();
      rv[4u * q + 2u] = -__builtin_inff();
      rv[4u * q + 3u] = -__builtin_inff();
    }
  }
#pragma unroll
  for (u32 k = 0u; k < 16u; k++) {
    if (in_bounds & (1u << k)) {
      const float v = rv[k];
      if (v == v) {
        vmask |= (1u << k);
        if (v > tmax) tmax = v;
      }
    }
  }

  const u32 tmax_k = vmask ? ink_r7_ds_f2u(tmax) : 0u;
  const u32 tau_k = ink_r7_ds_bitonic_kth_256(tmax_k, K);
  const float tau = tau_k ? ink_r7_ds_u2f(tau_k) : -__builtin_inff();

  u32 gt_mask = 0u, eq_mask = 0u;
  ink_r7_ds_classify16(rv, vmask, tmax_k, tau_k, tau, &gt_mask, &eq_mask);

  const u32 my_gt = (u32)__builtin_popcount(gt_mask);
  const u32 my_eq = (u32)__builtin_popcount(eq_mask);
  u32 packed_tot = 0u;
  const u32 packed_off = ink_r7_ds_wg_scan_u32((my_eq << 16u) | my_gt, ink_r7_ds_samp_lds.wsum[1u], &packed_tot);
  const u32 base_gt = packed_off & 0xFFFFu;
  const u32 base_eq = packed_off >> 16u;
  const u32 total_gt = packed_tot & 0xFFFFu;
  const u32 total_eq = packed_tot >> 16u;

  if (gt_mask) {
    u32 pos = base_gt;
#pragma unroll
    for (u32 q = 0u; q < 4u; q++) {
      const u32 base = i0 + ((q * NT + t) << 2u);
#pragma unroll
      for (u32 e = 0u; e < 4u; e++) {
        if (gt_mask & (1u << (4u * q + e))) {
          ink_r7_ds_samp_lds.u.cand.cv[pos] = rv[4u * q + e];
          ink_r7_ds_samp_lds.u.cand.ci[pos] = base + e;
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
            ink_r7_ds_samp_lds.u.cand.cv[pos] = rv[4u * q + e];
            ink_r7_ds_samp_lds.u.cand.ci[pos] = base + e;
            pos++;
          }
        }
      }
    }
    cand_n = total_gt + total_eq;
  }
  lbarrier();

  if (ng > 1u) {
    G float *seg_v = part_v + (unsigned long)g * (unsigned long)K;
    G u32   *seg_i = part_i + (unsigned long)g * (unsigned long)K;
    if (need > 0u && total_eq > need) {
      u32 taken = 0u;
#pragma clang loop unroll(disable)
      for (u32 q = 0u; q < 4u; q++) {
        if (taken < need) {
          const u32 m4 = (eq_mask >> (4u * q)) & 0xFu;
          const u32 cnt4 = (u32)__builtin_popcount(m4);
          u32 step_tot = 0u;
          const u32 step_off = ink_r7_ds_wg_scan_u32(cnt4, ink_r7_ds_samp_lds.wsum[q & 1u], &step_tot);
          if (m4) {
            u32 off = taken + step_off;
            const u32 base = i0 + ((q * NT + t) << 2u);
#pragma unroll
            for (u32 e = 0u; e < 4u; e++) {
              if (m4 & (1u << e)) {
                if (off < need) {
                  seg_v[total_gt + off] = logits[base + e];
                  seg_i[total_gt + off] = base + e;
                }
                off++;
              }
            }
          }
          taken += step_tot;
        }
      }
    }
    ink_r7_ds_rank_write_g(cand_n, K, seg_v, seg_i);
    if (tau_k == 0u && cand_n < K && t < K - cand_n) {
      seg_v[cand_n + t] = -__builtin_inff();
      seg_i[cand_n + t] = 0xFFFFFFFFu;
    }

    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (t == 0u) {
      ink_r7_ds_samp_lds.last =
          (__scoped_atomic_fetch_add(am_count, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r7_ds_samp_lds.last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (t == 0u) *am_count = 0u;

    /* Stage 2: merge nc = ng * K <= 2048 into ink_r7_ds_samp_lds.fv / fi [0..K-1]. */
    const u32 nc = ng * K;
    if (nc <= NT) {
      float mv = -__builtin_inff();
      u32 mi = 0xFFFFFFFFu;
      if (t < nc) {
        mv = part_v[t];
        mi = part_i[t];
        ink_r7_ds_samp_lds.u.cand.cv[t] = mv;
        ink_r7_ds_samp_lds.u.cand.ci[t] = mi;
      }
      const u32 is_val = (t < nc && mi != 0xFFFFFFFFu) ? 1u : 0u;
      const u32 wcnt = (u32)__builtin_popcount(__builtin_amdgcn_ballot_w32(is_val != 0u));
      if (lane() == 0u) {
        ink_r7_ds_samp_lds.wsum[0u][(u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u))] = wcnt;
      }
      lbarrier();
      u32 nvalid = 0u;
#pragma clang loop unroll(disable)
      for (u32 j = 0u; j < 8u; j++) nvalid += ink_r7_ds_samp_lds.wsum[0u][j];
      if (is_val) {
        u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
        for (u32 u = 0u; u < nc; u++) {
          const u32 ui = ink_r7_ds_samp_lds.u.cand.ci[u];
          if (ui != 0xFFFFFFFFu) {
            const float b = ink_r7_ds_samp_lds.u.cand.cv[u];
            rank += (u32)((b > mv) | ((b == mv) & (ui < mi)));
          }
        }
        if (rank < K) {
          ink_r7_ds_samp_lds.fv[rank] = mv;
          ink_r7_ds_samp_lds.fi[rank] = mi;
        }
      }
      if (nvalid < K && t < K - nvalid) {
        ink_r7_ds_samp_lds.fv[nvalid + t] = -__builtin_inff();
        ink_r7_ds_samp_lds.fi[nvalid + t] = 0xFFFFFFFFu;
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
          if (idx != 0xFFFFFFFFu) {
            vm2 |= (1u << k);
            if (v > tm2) tm2 = v;
          }
        } else {
          mv[k] = -__builtin_inff();
        }
      }

      const u32 tm2_k = vm2 ? ink_r7_ds_f2u(tm2) : 0u;
      const u32 tau2_k = ink_r7_ds_bitonic_kth_256(tm2_k, K);
      const float tau2 = tau2_k ? ink_r7_ds_u2f(tau2_k) : -__builtin_inff();

      u32 gm2 = 0u, em2 = 0u;
      ink_r7_ds_classify16(mv, vm2, tm2_k, tau2_k, tau2, &gm2, &em2);

      const u32 my_gt2 = (u32)__builtin_popcount(gm2);
      const u32 my_eq2 = (u32)__builtin_popcount(em2);
      u32 ptot2 = 0u;
      const u32 poff2 = ink_r7_ds_wg_scan_u32((my_eq2 << 16u) | my_gt2, ink_r7_ds_samp_lds.wsum[1u], &ptot2);
      const u32 bgt2 = poff2 & 0xFFFFu;
      const u32 beq2 = poff2 >> 16u;
      const u32 tgt2 = ptot2 & 0xFFFFu;
      const u32 teq2 = ptot2 >> 16u;

      if (gm2) {
        u32 pos = bgt2;
#pragma unroll
        for (u32 k = 0u; k < 16u; k++) {
          if (gm2 & (1u << k)) {
            ink_r7_ds_samp_lds.u.cand.cv[pos] = mv[k];
            ink_r7_ds_samp_lds.u.cand.ci[pos] = part_i[t + k * NT];
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
              ink_r7_ds_samp_lds.u.cand.cv[pos] = mv[k];
              ink_r7_ds_samp_lds.u.cand.ci[pos] = part_i[t + k * NT];
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
            const u32 soff = ink_r7_ds_wg_scan_u32(hit, ink_r7_ds_samp_lds.wsum[k & 1u], &stot);
            if (hit && taken2 + soff < need2) {
              const u32 p = t + k * NT;
              const u32 rk = tgt2 + taken2 + soff;
              ink_r7_ds_samp_lds.fv[rk] = part_v[p];
              ink_r7_ds_samp_lds.fi[rk] = part_i[p];
            }
            taken2 += stot;
          }
        }
      }
      ink_r7_ds_rank_write_lds(c2, K);
      if (tau2_k == 0u && c2 < K && t < K - c2) {
        ink_r7_ds_samp_lds.fv[c2 + t] = -__builtin_inff();
        ink_r7_ds_samp_lds.fi[c2 + t] = 0xFFFFFFFFu;
      }
    }
  } else {
    /* ng == 1u: write directly into ink_r7_ds_samp_lds.fv / fi. */
    if (need > 0u && total_eq > need) {
      u32 taken = 0u;
#pragma clang loop unroll(disable)
      for (u32 q = 0u; q < 4u; q++) {
        if (taken < need) {
          const u32 m4 = (eq_mask >> (4u * q)) & 0xFu;
          const u32 cnt4 = (u32)__builtin_popcount(m4);
          u32 step_tot = 0u;
          const u32 step_off = ink_r7_ds_wg_scan_u32(cnt4, ink_r7_ds_samp_lds.wsum[q & 1u], &step_tot);
          if (m4) {
            u32 off = taken + step_off;
            const u32 base = i0 + ((q * NT + t) << 2u);
#pragma unroll
            for (u32 e = 0u; e < 4u; e++) {
              if (m4 & (1u << e)) {
                if (off < need) {
                  ink_r7_ds_samp_lds.fv[total_gt + off] = logits[base + e];
                  ink_r7_ds_samp_lds.fi[total_gt + off] = base + e;
                }
                off++;
              }
            }
          }
          taken += step_tot;
        }
      }
    }
    ink_r7_ds_rank_write_lds(cand_n, K);
    if (tau_k == 0u && cand_n < K && t < K - cand_n) {
      ink_r7_ds_samp_lds.fv[cand_n + t] = -__builtin_inff();
      ink_r7_ds_samp_lds.fi[cand_n + t] = 0xFFFFFFFFu;
    }
  }
  lbarrier();

  /* Stage 3 (Wave 0, 32 lanes): Softmax(temp) + exact top_p nucleus + entropy + xorshift64*. */
  if (t < 32u) {
    const u32 c = t;
    const u32 ci = (c < K) ? ink_r7_ds_samp_lds.fi[c] : 0xFFFFFFFFu;
    const float cv_raw = (c < K) ? ink_r7_ds_samp_lds.fv[c] : -__builtin_inff();
    const float v0_raw = ink_r7_ds_samp_lds.fv[0u];
    const u32 valid = (c < K && ci != 0xFFFFFFFFu && cv_raw == cv_raw && v0_raw == v0_raw) ? 1u : 0u;
    const float cv = valid ? ink_r7_ds_cap(cv_raw, cap) : -__builtin_inff();
    const float v0 = (v0_raw == v0_raw) ? ink_r7_ds_cap(v0_raw, cap) : v0_raw;

    const float wc = valid ? __builtin_expf((cv - v0) / temp) : 0.0f;
    const float z = wave_sum_all(wc);

    /* Inclusive prefix sum across wave 0 to evaluate exact gen.c:782 nucleus condition. */
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
    const float hc = (qc > 0.0f) ? (-qc * __builtin_logf(qc)) : 0.0f;
    const float H = wave_sum_all(hc);

    ink_r7_ds_samp_lds.fq[c] = qc;
    if (c < m && c < K_stride) {
      if (q_ids) q_ids[(unsigned long)step_idx * (unsigned long)K_stride + (unsigned long)c] = ci;
      if (q_probs) q_probs[(unsigned long)step_idx * (unsigned long)K_stride + (unsigned long)c] = qc;
    }

    if (c == 0u) {
      u32 chosen = 0u;
      u32 chosen_tok = (m > 0u) ? ink_r7_ds_samp_lds.fi[0u] : 0u;
      float chosen_p = (m > 0u) ? ink_r7_ds_samp_lds.fq[0u] : 0.0f;

      if (m > 1u && rng_state) {
        ink_r7_ds_u64 rs = *rng_state;
        if (rs == 0ull) rs = 0x9E3779B97F4A7C15ull ^ ((ink_r7_ds_u64)step_idx + 1ull);
        rs ^= rs >> 12u;
        rs ^= rs << 25u;
        rs ^= rs >> 27u;
        *rng_state = rs;
        const ink_r7_ds_u64 rnd = rs * 2685821657736338717ull;
        float u = (float)((u32)(rnd >> 40u)) * (1.0f / 16777216.0f);
        u32 last_nz = 0u;
#pragma clang loop unroll(disable)
        for (u32 j = 0u; j < m; j++) {
          const float pj = ink_r7_ds_samp_lds.fq[j];
          if (pj > 0.0f) last_nz = j;
          u -= pj;
          if (u <= 0.0f && pj > 0.0f) {
            chosen = j;
            break;
          }
          if (j + 1u == m) chosen = last_nz;
        }
        chosen_tok = ink_r7_ds_samp_lds.fi[chosen];
        chosen_p = ink_r7_ds_samp_lds.fq[chosen];
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

/* ============================================================================
 * Kernel 2: ie_inkling_ds_attn_long_r7
 * Split-KV GQA-3 x Block-Token-8 dwordx4 Attention for DSpark (hd=128).
 * ========================================================================== */

typedef struct {
  f16 k[INK_R7_DS_BK][INK_R7_DS_LSTR]; /* 16 * 136 * 2 = 4,352 B */
  f16 v[INK_R7_DS_BK][INK_R7_DS_LSTR]; /* 16 * 136 * 2 = 4,352 B */
  u32 last;
} ink_r7_ds_attn_lds_t;

static LDS ink_r7_ds_attn_lds_t ink_r7_ds_attn_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r7_ds_attn_lds_t) <= 16384u, "ink_r7_ds_attn_lds must stay <= 16 KiB");

KERNEL ie_inkling_ds_attn_long_r7(
    const G float *q, const G kvt *kc_ctx, const G kvt *vc_ctx,
    const G float *k_blk, const G float *v_blk, G float *out,
    u32 ctx_len, u32 kv_cap, u32 win, u32 block_size,
    u32 n_head, u32 n_kv, u32 hd, u32 qs, u32 ks, u32 os,
    u32 nsplit, G float *part, G u32 *count) {
  if (hd != INK_R7_DS_HD || n_kv == 0u || n_head != n_kv * 3u || block_size == 0u || nsplit == 0u) return;

  const u32 g = wgid();
  const u32 kh = g / nsplit;
  const u32 sp = g - kh * nsplit;
  const u32 btile = tokid();
  const u32 bt_base = btile << 3u;
  if (kh >= n_kv || bt_base >= block_size) return;

  const u32 t = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = lane();
  const u32 bt = bt_base + w;
  const u32 h0 = kh * 3u;
  const u32 kvd = n_kv * INK_R7_DS_HD;
  const float scale = 1.0f / __builtin_sqrtf((float)INK_R7_DS_HD);

  /* Each wave w loads its 3 GQA Q heads for block token bt into 12 VGPRs. */
  f32x4 q0 = {0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 q1 = {0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 q2 = {0.0f, 0.0f, 0.0f, 0.0f};
  if (bt < block_size) {
    const G float *qb = (const G float *)((const G u8 *)q + (unsigned long)bt * (unsigned long)qs)
                        + (unsigned long)h0 * INK_R7_DS_HD;
    q0 = ((const G struct ink_r7_ds_f32x4_a4 *)(qb + 0u * INK_R7_DS_HD + 4u * l))->v;
    q1 = ((const G struct ink_r7_ds_f32x4_a4 *)(qb + 1u * INK_R7_DS_HD + 4u * l))->v;
    q2 = ((const G struct ink_r7_ds_f32x4_a4 *)(qb + 2u * INK_R7_DS_HD + 4u * l))->v;
  }

  f32x4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
  float m0 = -__builtin_inff(), l0 = 0.0f;
  float m1 = -__builtin_inff(), l1 = 0.0f;
  float m2 = -__builtin_inff(), l2 = 0.0f;

  /* 1) Context KV slice for this split: BK=16 tiled dwordx4 load into LDS -> shared by 24 queries! */
  const u32 ctx_start = (win > 0u && ctx_len > win) ? (ctx_len - win) : 0u;
  const u32 ctx_n = (kv_cap > 0u && ctx_len > ctx_start) ? (ctx_len - ctx_start) : 0u;
  const u32 chunk = (ctx_n + nsplit - 1u) / nsplit;
  const u32 p_begin = ctx_start + sp * chunk;
  const u32 p_end = (p_begin < ctx_len) ? ((p_begin + chunk < ctx_len) ? (p_begin + chunk) : ctx_len) : p_begin;

  if (p_begin < p_end) {
    const u32 ld_r = t >> 4u; /* 0..15 */
    const u32 ld_c = t & 15u; /* 0..15 (each = 8 f16 = 16 B) */
    const u32 fast_wrap = (kv_cap >= INK_R7_DS_BK) ? 1u : 0u;
    u32 ring_p = p_begin % kv_cap;

    for (u32 p0 = p_begin; p0 < p_end; p0 += INK_R7_DS_BK) {
      const u32 n_tile = (p_end - p0 < INK_R7_DS_BK) ? (p_end - p0) : INK_R7_DS_BK;
      if (ld_r < n_tile) {
        u32 rp = ring_p + ld_r;
        if (fast_wrap) {
          if (rp >= kv_cap) rp -= kv_cap;
        } else {
          rp %= kv_cap;
        }
        const unsigned long off = (unsigned long)rp * (unsigned long)kvd
                                + (unsigned long)kh * INK_R7_DS_HD + (unsigned long)(ld_c << 3u);
        *(LDSP u32x4 *)(ink_r7_ds_attn_lds.k[ld_r] + (ld_c << 3u)) =
            ((const G struct ink_r7_ds_u32x4_a4 *)(kc_ctx + off))->v;
        *(LDSP u32x4 *)(ink_r7_ds_attn_lds.v[ld_r] + (ld_c << 3u)) =
            ((const G struct ink_r7_ds_u32x4_a4 *)(vc_ctx + off))->v;
      }
      lbarrier();

      if (bt < block_size) {
#pragma clang loop unroll(disable)
        for (u32 r = 0u; r < n_tile; r++) {
          const ink_r7_ds_f16x4 kh4 = *(const LDSP ink_r7_ds_f16x4 *)(ink_r7_ds_attn_lds.k[r] + (l << 2u));
          const f32x4 kv = {(float)kh4.x, (float)kh4.y, (float)kh4.z, (float)kh4.w};
          const ink_r7_ds_f16x4 vh4 = *(const LDSP ink_r7_ds_f16x4 *)(ink_r7_ds_attn_lds.v[r] + (l << 2u));
          const f32x4 vv = {(float)vh4.x, (float)vh4.y, (float)vh4.z, (float)vh4.w};

          {
            float d = (q0.x * kv.x + q0.y * kv.y) + (q0.z * kv.z + q0.w * kv.w);
            d = wave_sum_all(d) * scale;
            const float mn = __builtin_fmaxf(m0, d);
            const float cf = l0 > 0.0f ? __builtin_expf(m0 - mn) : 0.0f;
            const float ef = __builtin_expf(d - mn);
            l0 = l0 * cf + ef;
            acc0 = acc0 * cf + vv * ef;
            m0 = mn;
          }
          {
            float d = (q1.x * kv.x + q1.y * kv.y) + (q1.z * kv.z + q1.w * kv.w);
            d = wave_sum_all(d) * scale;
            const float mn = __builtin_fmaxf(m1, d);
            const float cf = l1 > 0.0f ? __builtin_expf(m1 - mn) : 0.0f;
            const float ef = __builtin_expf(d - mn);
            l1 = l1 * cf + ef;
            acc1 = acc1 * cf + vv * ef;
            m1 = mn;
          }
          {
            float d = (q2.x * kv.x + q2.y * kv.y) + (q2.z * kv.z + q2.w * kv.w);
            d = wave_sum_all(d) * scale;
            const float mn = __builtin_fmaxf(m2, d);
            const float cf = l2 > 0.0f ? __builtin_expf(m2 - mn) : 0.0f;
            const float ef = __builtin_expf(d - mn);
            l2 = l2 * cf + ef;
            acc2 = acc2 * cf + vv * ef;
            m2 = mn;
          }
        }
      }
      lbarrier();

      if (fast_wrap) {
        ring_p += INK_R7_DS_BK;
        if (ring_p >= kv_cap) ring_p -= kv_cap;
      } else {
        ring_p = (ring_p + INK_R7_DS_BK) % kv_cap;
      }
    }
  }

  /* 2) Bidirectional block KV [0..block_size-1] (f32): handled in split sp == 0. */
  if (sp == 0u && bt < block_size) {
    for (u32 b = 0u; b < block_size; b++) {
      const G float *kbp = (const G float *)((const G u8 *)k_blk + (unsigned long)b * (unsigned long)ks)
                         + (unsigned long)kh * INK_R7_DS_HD;
      const G float *vbp = (const G float *)((const G u8 *)v_blk + (unsigned long)b * (unsigned long)ks)
                         + (unsigned long)kh * INK_R7_DS_HD;
      const f32x4 kv = ((const G struct ink_r7_ds_f32x4_a4 *)(kbp + 4u * l))->v;
      const f32x4 vv = ((const G struct ink_r7_ds_f32x4_a4 *)(vbp + 4u * l))->v;

      {
        float d = (q0.x * kv.x + q0.y * kv.y) + (q0.z * kv.z + q0.w * kv.w);
        d = wave_sum_all(d) * scale;
        const float mn = __builtin_fmaxf(m0, d);
        const float cf = l0 > 0.0f ? __builtin_expf(m0 - mn) : 0.0f;
        const float ef = __builtin_expf(d - mn);
        l0 = l0 * cf + ef;
        acc0 = acc0 * cf + vv * ef;
        m0 = mn;
      }
      {
        float d = (q1.x * kv.x + q1.y * kv.y) + (q1.z * kv.z + q1.w * kv.w);
        d = wave_sum_all(d) * scale;
        const float mn = __builtin_fmaxf(m1, d);
        const float cf = l1 > 0.0f ? __builtin_expf(m1 - mn) : 0.0f;
        const float ef = __builtin_expf(d - mn);
        l1 = l1 * cf + ef;
        acc1 = acc1 * cf + vv * ef;
        m1 = mn;
      }
      {
        float d = (q2.x * kv.x + q2.y * kv.y) + (q2.z * kv.z + q2.w * kv.w);
        d = wave_sum_all(d) * scale;
        const float mn = __builtin_fmaxf(m2, d);
        const float cf = l2 > 0.0f ? __builtin_expf(m2 - mn) : 0.0f;
        const float ef = __builtin_expf(d - mn);
        l2 = l2 * cf + ef;
        acc2 = acc2 * cf + vv * ef;
        m2 = mn;
      }
    }
  }

  /* 3) Direct output if nsplit == 1, else single-launch Split-KV reduction in last WG. */
  if (nsplit == 1u) {
    if (bt < block_size) {
      G float *ob = (G float *)((G u8 *)out + (unsigned long)bt * (unsigned long)os)
                  + (unsigned long)h0 * INK_R7_DS_HD;
      const float i0_s = l0 > 0.0f ? (1.0f / l0) : 0.0f;
      const float i1_s = l1 > 0.0f ? (1.0f / l1) : 0.0f;
      const float i2_s = l2 > 0.0f ? (1.0f / l2) : 0.0f;
      ((G struct ink_r7_ds_f32x4_a4 *)(ob + 0u * INK_R7_DS_HD + 4u * l))->v = acc0 * i0_s;
      ((G struct ink_r7_ds_f32x4_a4 *)(ob + 1u * INK_R7_DS_HD + 4u * l))->v = acc1 * i1_s;
      ((G struct ink_r7_ds_f32x4_a4 *)(ob + 2u * INK_R7_DS_HD + 4u * l))->v = acc2 * i2_s;
    }
    return;
  }

  if (bt < block_size) {
    const unsigned long b0 = (((unsigned long)bt * (unsigned long)n_head + (unsigned long)(h0 + 0u)) * nsplit + sp) * INK_R7_DS_PSTR;
    const unsigned long b1 = (((unsigned long)bt * (unsigned long)n_head + (unsigned long)(h0 + 1u)) * nsplit + sp) * INK_R7_DS_PSTR;
    const unsigned long b2 = (((unsigned long)bt * (unsigned long)n_head + (unsigned long)(h0 + 2u)) * nsplit + sp) * INK_R7_DS_PSTR;
    ((G struct ink_r7_ds_f32x4_a4 *)(part + b0 + 4u * l))->v = acc0;
    ((G struct ink_r7_ds_f32x4_a4 *)(part + b1 + 4u * l))->v = acc1;
    ((G struct ink_r7_ds_f32x4_a4 *)(part + b2 + 4u * l))->v = acc2;
    if (l == 0u) {
      part[b0 + 128u] = m0; part[b0 + 129u] = l0;
      part[b1 + 128u] = m1; part[b1 + 129u] = l1;
      part[b2 + 128u] = m2; part[b2 + 129u] = l2;
    }
  }

  const u32 cidx = btile * n_kv + kh;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (t == 0u) {
    ink_r7_ds_attn_lds.last =
        (__scoped_atomic_fetch_add(count + cidx, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r7_ds_attn_lds.last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  if (t == 0u) count[cidx] = 0u;

  if (bt < block_size) {
    G float *ob = (G float *)((G u8 *)out + (unsigned long)bt * (unsigned long)os)
                + (unsigned long)h0 * INK_R7_DS_HD;
#pragma clang loop unroll(disable)
    for (u32 qh = 0u; qh < 3u; qh++) {
      const unsigned long hbase = ((unsigned long)bt * (unsigned long)n_head + (unsigned long)(h0 + qh)) * nsplit;
      float M = -__builtin_inff();
      for (u32 s = 0u; s < nsplit; s++) {
        const G float *ps = part + (hbase + s) * INK_R7_DS_PSTR;
        if (ps[129u] > 0.0f) M = __builtin_fmaxf(M, ps[128u]);
      }
      float L = 0.0f;
      f32x4 O = {0.0f, 0.0f, 0.0f, 0.0f};
      for (u32 s = 0u; s < nsplit; s++) {
        const G float *ps = part + (hbase + s) * INK_R7_DS_PSTR;
        const float ls = ps[129u];
        if (ls > 0.0f) {
          const float wt = __builtin_expf(ps[128u] - M);
          L += ls * wt;
          O += ((const G struct ink_r7_ds_f32x4_a4 *)(ps + 4u * l))->v * wt;
        }
      }
      const float inv = (L > 0.0f) ? (1.0f / L) : 0.0f;
      ((G struct ink_r7_ds_f32x4_a4 *)(ob + (unsigned long)qh * INK_R7_DS_HD + 4u * l))->v = O * inv;
    }
  }
}

#undef INK_R7_DS_PSTR
#undef INK_R7_DS_LSTR
#undef INK_R7_DS_BK
#undef INK_R7_DS_HD
#undef INK_R7_DS_CAP
#undef INK_R7_DS_CSMAX
#undef INK_R7_DS_NGMAX
#undef INK_R7_DS_KMAX

#endif /* IE_MEGA */
