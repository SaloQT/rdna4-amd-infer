/* ============================================================================
 * Inkling-2 Round 7: Batched Single-Launch Verify Argmax & Top-K Kernels
 * Target: RX 9070 XT (gfx1201, wave32, NT=256)
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r7_vt_ /
 * INK_R7_VT_ and all macros are #undef'd at EOF.
 *
 * Exported kernels:
 *
 * 1. KERNEL ie_inkling_argmax_batched_r7(...)
 *    - Replaces T separate ie_argmax launches (0.72-1.12 ms/step in R00 trace)
 *      with ONE single vectorized launch for T=1..16.
 *    - Grid: (ng, T, 1), block: (256, 1, 1), 1 <= ng <= 128 (e.g. ng=32 for T>=4,
 *      ng=64 for T=2..3, ng=128 for T=1).
 *    - Optional inline logit softcap: when cap > 0.0f, applies
 *      cap * ie_tanhf_s(v / cap) during the single-pass f32x4 load (allowing
 *      OP_LOGIT_CAP + OP_ARGMAX fusion on greedy decode/verify!). When cap <= 0.0f,
 *      reads already-capped logits unmodified.
 *    - Single-pass f32x4 vectorized read:
 *        * need_prob == 0u: ZERO expf() calls (pure bandwidth argmax, ~15 us at T=8),
 *          writes [best_id, 0u].
 *        * need_prob != 0u: single-pass online (max, sum_exp), writes
 *          [best_id, __builtin_bit_cast(u32, 1.0f / S)] matching ie_argmax layout.
 *    - Exact lowest-index tie-breaking matching ie_argmax (better()).
 *    - Scratch:
 *        part:  T * ng * 3 u32 words (row r at part + r * ng * 3).
 *        count: T zero-initialized u32 counters (auto-reset to 0 by row r's last WG).
 *
 * 2. KERNEL ie_inkling_topk_batched_r7(...)
 *    - Single-launch, single-pass, zero-overflow wave32 batched Top-K for T=1..16,
 *      K <= 64, nv = vocab_unpadded (200058), ng <= 64, cs <= 4096.
 *    - Identical tk_buf layout to ie_saluki_topk_r11 (hip.c:7318-7324), plus
 *      optional inline float cap softcap (cap > 0.0f applies cap * ie_tanhf_s(v / cap)
 *      on load, allowing OP_LOGIT_CAP + topk fusion on sampled verify; cap <= 0.0f
 *      matches ie_topk_max + ie_topk_sel bit-for-bit, 0 ULP on lse[r]!).
 *    - Grid: (ng, n_rows, 1), block: (256, 1, 1), LDS: 8,336 B (4 WGs/CU max occ).
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R7_VT_KMAX  64u
#define INK_R7_VT_NGMAX 64u
#define INK_R7_VT_CSMAX 4096u
#define INK_R7_VT_CAP   1024u

struct __attribute__((packed, aligned(4))) ink_r7_vt_f32x4_a4 {
  f32x4 v;
};

static inline __attribute__((always_inline)) float ink_r7_vt_cap(float v, float cap) {
  return (cap > 0.0f) ? (cap * ie_tanhf_s(v / cap)) : v;
}

static inline __attribute__((always_inline)) int ink_r7_vt_better(float v, u32 i, float bv, u32 bi) {
  return (v > bv) || (v == bv && i < bi);
}

/* ============================================================================
 * Kernel 1: ie_inkling_argmax_batched_r7
 * ========================================================================== */

typedef struct {
  float wv[8u];
  u32   wi[8u];
  float ws[8u];
  u32   last;
} ink_r7_vt_am_lds_t;

static inline __attribute__((always_inline)) void ink_r7_vt_am_wg_best(
    float *bv, u32 *bi, LDSP ink_r7_vt_am_lds_t *lds) {
  const u32 l = lane();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
#pragma unroll
  for (u32 m = 16u; m > 0u; m >>= 1u) {
    const float ov = shfl_xor_f(*bv, m);
    const u32 oi = (u32)__builtin_amdgcn_ds_bpermute((int)((l ^ m) * 4u), (int)*bi);
    if (ink_r7_vt_better(ov, oi, *bv, *bi)) {
      *bv = ov;
      *bi = oi;
    }
  }
  if (l == 0u) {
    lds->wv[w] = *bv;
    lds->wi[w] = *bi;
  }
  lbarrier();
  float v = lds->wv[0u];
  u32 idx = lds->wi[0u];
#pragma clang loop unroll(disable)
  for (u32 j = 1u; j < 8u; j++) {
    const float jv = lds->wv[j];
    const u32 ji = lds->wi[j];
    if (ink_r7_vt_better(jv, ji, v, idx)) {
      v = jv;
      idx = ji;
    }
  }
  *bv = v;
  *bi = idx;
}

static inline __attribute__((always_inline)) float ink_r7_vt_am_wg_online_sum(
    float bm, float se, float wg_max, LDSP ink_r7_vt_am_lds_t *lds) {
  const u32 l = lane();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  float s = (se > 0.0f) ? (se * __builtin_expf(bm - wg_max)) : 0.0f;
  s = wave_sum_all(s);
  if (l == 0u) lds->ws[w] = s;
  lbarrier();
  float tot = 0.0f;
#pragma clang loop unroll(disable)
  for (u32 j = 0u; j < 8u; j++) tot += lds->ws[j];
  return tot;
}

KERNEL ie_inkling_argmax_batched_r7(
    const G float *x, u32 n, u32 xs_bytes, float cap, u32 need_prob,
    G u32 *out, u32 os_bytes, G u32 *part, G u32 *count, u32 ng) {
  static LDS ink_r7_vt_am_lds_t ink_r7_vt_am_lds __attribute__((aligned(16)));

  const u32 g = wgid();
  const u32 r = tokid();
  const u32 t = tid();
  if (g >= ng || ng == 0u || n == 0u) return;

  const G float *row = (const G float *)((const G u8 *)x + (unsigned long)r * (unsigned long)xs_bytes);
  G u32 *out_r = (G u32 *)((G u8 *)out + (unsigned long)r * (unsigned long)os_bytes);

  float bv = -__builtin_inff();
  u32 bi = 0xFFFFFFFFu;
  float se = 0.0f;

  const u32 n4 = n >> 2u;
  const u32 stride4 = ng * NT;
  if (need_prob) {
    for (u32 v4i = g * NT + t; v4i < n4; v4i += stride4) {
      const u32 i0 = v4i << 2u;
      const f32x4 raw = ((const G struct ink_r7_vt_f32x4_a4 *)(row + i0))->v;
      const float f0 = ink_r7_vt_cap(raw.x, cap);
      const float f1 = ink_r7_vt_cap(raw.y, cap);
      const float f2 = ink_r7_vt_cap(raw.z, cap);
      const float f3 = ink_r7_vt_cap(raw.w, cap);

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
    if (g == 0u && tail_base + t < n) {
      const u32 idx = tail_base + t;
      const float fv = ink_r7_vt_cap(row[idx], cap);
      if (fv > bv) { se = se * __builtin_expf(bv - fv) + 1.0f; bv = fv; bi = idx; }
      else { if (fv == bv && idx < bi) bi = idx; if (fv > -__builtin_inff()) se += __builtin_expf(fv - bv); }
    }
  } else {
    for (u32 v4i = g * NT + t; v4i < n4; v4i += stride4) {
      const u32 i0 = v4i << 2u;
      const f32x4 raw = ((const G struct ink_r7_vt_f32x4_a4 *)(row + i0))->v;
      const float f0 = ink_r7_vt_cap(raw.x, cap);
      const float f1 = ink_r7_vt_cap(raw.y, cap);
      const float f2 = ink_r7_vt_cap(raw.z, cap);
      const float f3 = ink_r7_vt_cap(raw.w, cap);
      if (ink_r7_vt_better(f0, i0, bv, bi)) { bv = f0; bi = i0; }
      if (ink_r7_vt_better(f1, i0 + 1u, bv, bi)) { bv = f1; bi = i0 + 1u; }
      if (ink_r7_vt_better(f2, i0 + 2u, bv, bi)) { bv = f2; bi = i0 + 2u; }
      if (ink_r7_vt_better(f3, i0 + 3u, bv, bi)) { bv = f3; bi = i0 + 3u; }
    }
    const u32 tail_base = n4 << 2u;
    if (g == 0u && tail_base + t < n) {
      const u32 idx = tail_base + t;
      const float fv = ink_r7_vt_cap(row[idx], cap);
      if (ink_r7_vt_better(fv, idx, bv, bi)) { bv = fv; bi = idx; }
    }
  }

  const float my_bm = bv;
  ink_r7_vt_am_wg_best(&bv, &bi, &ink_r7_vt_am_lds);
  const float wg_se = need_prob ? ink_r7_vt_am_wg_online_sum(my_bm, se, bv, &ink_r7_vt_am_lds) : 0.0f;

  if (ng == 1u) {
    if (t == 0u) {
      const float prob = (need_prob && wg_se > 0.0f) ? (1.0f / wg_se) : 0.0f;
      out_r[0u] = bi;
      out_r[1u] = need_prob ? __builtin_bit_cast(u32, prob) : 0u;
    }
    return;
  }

  G u32 *part_r = part + (unsigned long)r * (unsigned long)ng * 3ul;
  if (t == 0u) {
    const float bv_s = bv;
    const float se_s = wg_se;
    part_r[3u * g + 0u] = __builtin_bit_cast(u32, bv_s);
    part_r[3u * g + 1u] = bi;
    part_r[3u * g + 2u] = __builtin_bit_cast(u32, se_s);
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (t == 0u) {
    ink_r7_vt_am_lds.last =
        (__scoped_atomic_fetch_add(count + r, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r7_vt_am_lds.last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  bv = -__builtin_inff();
  bi = 0xFFFFFFFFu;
  for (u32 u = t; u < ng; u += NT) {
    const u32 vb = __scoped_atomic_load_n(&part_r[3u * u + 0u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    const u32 iu = __scoped_atomic_load_n(&part_r[3u * u + 1u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    const float vu = __builtin_bit_cast(float, vb);
    if (ink_r7_vt_better(vu, iu, bv, bi)) {
      bv = vu;
      bi = iu;
    }
  }
  ink_r7_vt_am_wg_best(&bv, &bi, &ink_r7_vt_am_lds);

  float S = 0.0f;
  if (need_prob) {
    for (u32 u = t; u < ng; u += NT) {
      const u32 vb = __scoped_atomic_load_n(&part_r[3u * u + 0u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      const u32 sb = __scoped_atomic_load_n(&part_r[3u * u + 2u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      const float vu = __builtin_bit_cast(float, vb);
      const float su = __builtin_bit_cast(float, sb);
      if (su > 0.0f) S += su * __builtin_expf(vu - bv);
    }
    S = wave_sum_all(S);
    if (lane() == 0u) ink_r7_vt_am_lds.ws[(u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u))] = S;
    lbarrier();
    S = 0.0f;
#pragma clang loop unroll(disable)
    for (u32 j = 0u; j < 8u; j++) S += ink_r7_vt_am_lds.ws[j];
  }

  if (t == 0u) {
    const float prob = (need_prob && S > 0.0f) ? (1.0f / S) : 0.0f;
    out_r[0u] = bi;
    out_r[1u] = need_prob ? __builtin_bit_cast(u32, prob) : 0u;
    count[r] = 0u;
  }
}

/* ============================================================================
 * Kernel 2: ie_inkling_topk_batched_r7
 * Single-launch, single-pass, zero-overflow wave32 Top-K + optional softcap/LSE.
 * ========================================================================== */

typedef struct {
  union {
    u32 sm[2u][NT];
    struct {
      float cv[INK_R7_VT_CAP];
      u32   ci[INK_R7_VT_CAP];
    } cand;
  } u;
  u32   wsum[2u][8u];
  float lse_m[8u];
  float lse_l[8u];
  u32   last;
} ink_r7_vt_tk_lds_t;

static LDS ink_r7_vt_tk_lds_t ink_r7_vt_tk_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r7_vt_tk_lds_t) <= 16384u, "ink_r7_vt_tk_lds must stay <= 16 KiB");

static inline __attribute__((always_inline)) u32 ink_r7_vt_f2u(float f) {
  u32 u = __builtin_bit_cast(u32, f);
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static inline __attribute__((always_inline)) float ink_r7_vt_u2f(u32 k) {
  const u32 u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  return __builtin_bit_cast(float, u);
}

static inline __attribute__((always_inline)) u32 ink_r7_vt_cswap(u32 v, u32 ov, u32 hi, u32 asc) {
  return (hi ^ asc) ? (v < ov ? v : ov) : (v > ov ? v : ov);
}

static inline __attribute__((always_inline)) u32 ink_r7_vt_wave_step(u32 v, u32 t, u32 stride, u32 asc) {
  const u32 ov = (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ stride) * 4u), (int)v);
  return ink_r7_vt_cswap(v, ov, (u32)((t & stride) != 0u), asc);
}

static inline __attribute__((always_inline)) u32 ink_r7_vt_lds_step(
    u32 v, u32 t, u32 stride, u32 asc, LDSP u32 *buf) {
  buf[t] = v;
  lbarrier();
  const u32 ov = buf[t ^ stride];
  return ink_r7_vt_cswap(v, ov, (u32)((t & stride) != 0u), asc);
}

static inline __attribute__((always_inline)) u32 ink_r7_vt_bitonic_kth_256(u32 v, u32 K) {
  const u32 t = tid();
#pragma clang loop unroll(disable)
  for (u32 size = 2u; size <= 32u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride > 0u; stride >>= 1u) {
      v = ink_r7_vt_wave_step(v, t, stride, a);
    }
  }
  u32 buf_id = 0u;
#pragma clang loop unroll(disable)
  for (u32 size = 64u; size <= 256u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride >= 32u; stride >>= 1u) {
      v = ink_r7_vt_lds_step(v, t, stride, a, ink_r7_vt_tk_lds.u.sm[buf_id]);
      buf_id ^= 1u;
    }
#pragma clang loop unroll(disable)
    for (u32 stride = 16u; stride > 0u; stride >>= 1u) {
      v = ink_r7_vt_wave_step(v, t, stride, a);
    }
  }
  if (t == K - 1u) ink_r7_vt_tk_lds.u.sm[0u][0u] = v;
  lbarrier();
  return ink_r7_vt_tk_lds.u.sm[0u][0u];
}

static inline __attribute__((always_inline)) u32 ink_r7_vt_wg_scan_u32(
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

static inline __attribute__((always_inline)) void ink_r7_vt_rank_write(
    u32 c, u32 K, G float *out_v, G u32 *out_i) {
  for (u32 j = tid(); j < c; j += NT) {
    const float a = ink_r7_vt_tk_lds.u.cand.cv[j];
    const u32 ai = ink_r7_vt_tk_lds.u.cand.ci[j];
    u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
    for (u32 u = 0u; u < c; u++) {
      const float b = ink_r7_vt_tk_lds.u.cand.cv[u];
      rank += (u32)((b > a) | ((b == a) & (ink_r7_vt_tk_lds.u.cand.ci[u] < ai)));
    }
    if (rank < K) {
      out_v[rank] = a;
      out_i[rank] = ai;
    }
  }
}

static inline __attribute__((always_inline)) void ink_r7_vt_classify16(
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

KERNEL ie_inkling_topk_batched_r7(
    const G float *x, u32 n, u32 xs_f32, u32 K, u32 cs, u32 ng, float cap, u32 need_lse,
    G float *smax, G float *cv, G u32 *ci, G u32 *done,
    G float *ov, G u32 *oi) {
  const u32 r = tokid(), g = wgid();
  if (g >= ng || n == 0u || K == 0u || K > INK_R7_VT_KMAX || ng == 0u || ng > INK_R7_VT_NGMAX ||
      cs == 0u || cs > INK_R7_VT_CSMAX) return;

  const u32 t = tid();

  /* Stage 1: Single-pass load (+ optional softcap + exact online LSE if need_lse != 0). */
  float rv[16u];
  u32 vmask = 0u;
  float tmax = -__builtin_inff();
  u32 i0 = 0u;

  {
    const G float *row = x + (unsigned long)r * (unsigned long)xs_f32;
    if (need_lse) {
      i0 = g * cs;
      u32 in_bounds = 0u;
      if (cs == INK_R7_VT_CSMAX && i0 + INK_R7_VT_CSMAX <= n) {
#pragma unroll
        for (u32 k = 0u; k < 16u; k++) rv[k] = ink_r7_vt_cap(row[i0 + t + k * NT], cap);
        in_bounds = 0xFFFFu;
      } else {
#pragma unroll
        for (u32 k = 0u; k < 16u; k++) {
          const u32 p = t + k * NT;
          const u32 ok = (p < cs && i0 + p < n);
          rv[k] = ok ? ink_r7_vt_cap(row[i0 + p], cap) : -__builtin_inff();
          if (ok) in_bounds |= (1u << k);
        }
      }
      float m = -__builtin_inff(), l = 0.0f;
#pragma unroll
      for (u32 k = 0u; k < 16u; k++) {
        if (in_bounds & (1u << k)) {
          const float v = rv[k];
          if (v > m) {
            l = l * __builtin_expf(m - v) + 1.0f;
            m = v;
          } else if (v > -__builtin_inff()) {
            l += __builtin_expf(v - m);
          }
          if (v == v) {
            vmask |= (1u << k);
            if (v > tmax) tmax = v;
          }
        }
      }
#pragma clang loop unroll(disable)
      for (u32 s = 16u; s > 0u; s >>= 1u) {
        const float om = shfl_xor_f(m, s), ol = shfl_xor_f(l, s);
        if (om > m) {
          l = l * __builtin_expf(m - om) + ol;
          m = om;
        } else if (m > -__builtin_inff()) {
          l += ol * __builtin_expf(om - m);
        }
      }
      if (lane() == 0u) {
        const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
        ink_r7_vt_tk_lds.lse_m[w] = m;
        ink_r7_vt_tk_lds.lse_l[w] = l;
      }
    } else {
      const u32 cs4 = (cs + 3u) & ~3u;
      const u32 nvec = cs4 >> 2u;
      i0 = g * cs4;
      u32 in_bounds = 0u;
#pragma unroll
      for (u32 q = 0u; q < 4u; q++) {
        const u32 vidx = t + q * NT;
        const u32 base = i0 + (vidx << 2u);
        if (vidx < nvec && base + 4u <= n) {
          const f32x4 v4 = ((const G struct ink_r7_vt_f32x4_a4 *)(row + base))->v;
          rv[4u * q + 0u] = ink_r7_vt_cap(v4.x, cap);
          rv[4u * q + 1u] = ink_r7_vt_cap(v4.y, cap);
          rv[4u * q + 2u] = ink_r7_vt_cap(v4.z, cap);
          rv[4u * q + 3u] = ink_r7_vt_cap(v4.w, cap);
          in_bounds |= (0xFu << (4u * q));
        } else if (vidx < nvec && base < n) {
#pragma unroll
          for (u32 e = 0u; e < 4u; e++) {
            const u32 ok = (base + e < n);
            rv[4u * q + e] = ok ? ink_r7_vt_cap(row[base + e], cap) : -__builtin_inff();
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
    }
  }

  const u32 tmax_k = vmask ? ink_r7_vt_f2u(tmax) : 0u;
  const u32 tau_k = ink_r7_vt_bitonic_kth_256(tmax_k, K);
  const float tau = tau_k ? ink_r7_vt_u2f(tau_k) : -__builtin_inff();

  u32 gt_mask = 0u, eq_mask = 0u;
  ink_r7_vt_classify16(rv, vmask, tmax_k, tau_k, tau, &gt_mask, &eq_mask);

  const u32 my_gt = (u32)__builtin_popcount(gt_mask);
  const u32 my_eq = (u32)__builtin_popcount(eq_mask);
  u32 packed_tot = 0u;
  const u32 packed_off = ink_r7_vt_wg_scan_u32((my_eq << 16u) | my_gt, ink_r7_vt_tk_lds.wsum[1u], &packed_tot);
  const u32 base_gt = packed_off & 0xFFFFu;
  const u32 base_eq = packed_off >> 16u;
  const u32 total_gt = packed_tot & 0xFFFFu;
  const u32 total_eq = packed_tot >> 16u;

  if (gt_mask) {
    u32 pos = base_gt;
    if (need_lse) {
#pragma unroll
      for (u32 k = 0u; k < 16u; k++) {
        if (gt_mask & (1u << k)) {
          ink_r7_vt_tk_lds.u.cand.cv[pos] = rv[k];
          ink_r7_vt_tk_lds.u.cand.ci[pos] = i0 + t + k * NT;
          pos++;
        }
      }
    } else {
#pragma unroll
      for (u32 q = 0u; q < 4u; q++) {
        const u32 base = i0 + ((q * NT + t) << 2u);
#pragma unroll
        for (u32 e = 0u; e < 4u; e++) {
          if (gt_mask & (1u << (4u * q + e))) {
            ink_r7_vt_tk_lds.u.cand.cv[pos] = rv[4u * q + e];
            ink_r7_vt_tk_lds.u.cand.ci[pos] = base + e;
            pos++;
          }
        }
      }
    }
  }

  const u32 need = (total_gt < K) ? (K - total_gt) : 0u;
  u32 cand_n = total_gt;
  if (need > 0u && total_eq <= need) {
    if (eq_mask) {
      u32 pos = total_gt + base_eq;
      if (need_lse) {
#pragma unroll
        for (u32 k = 0u; k < 16u; k++) {
          if (eq_mask & (1u << k)) {
            ink_r7_vt_tk_lds.u.cand.cv[pos] = rv[k];
            ink_r7_vt_tk_lds.u.cand.ci[pos] = i0 + t + k * NT;
            pos++;
          }
        }
      } else {
#pragma unroll
        for (u32 q = 0u; q < 4u; q++) {
          const u32 base = i0 + ((q * NT + t) << 2u);
#pragma unroll
          for (u32 e = 0u; e < 4u; e++) {
            if (eq_mask & (1u << (4u * q + e))) {
              ink_r7_vt_tk_lds.u.cand.cv[pos] = rv[4u * q + e];
              ink_r7_vt_tk_lds.u.cand.ci[pos] = base + e;
              pos++;
            }
          }
        }
      }
    }
    cand_n = total_gt + total_eq;
  }
  lbarrier();

  {
    const unsigned long seg_off = (ng == 1u) ? ((unsigned long)r * K) : (((unsigned long)r * ng + g) * K);
    G float *seg_v = (ng == 1u) ? (ov + seg_off) : (cv + seg_off);
    G u32   *seg_i = (ng == 1u) ? (oi + seg_off) : (ci + seg_off);

    if (need > 0u && total_eq > need) {
      const G float *row_eq = x + (unsigned long)r * (unsigned long)xs_f32;
      u32 taken = 0u;
      if (need_lse) {
#pragma clang loop unroll(disable)
        for (u32 k = 0u; k < 16u; k++) {
          if (taken < need) {
            const u32 hit = (eq_mask >> k) & 1u;
            u32 step_tot = 0u;
            const u32 step_off = ink_r7_vt_wg_scan_u32(hit, ink_r7_vt_tk_lds.wsum[k & 1u], &step_tot);
            if (hit && taken + step_off < need) {
              const u32 idx = i0 + t + k * NT;
              const u32 rk = total_gt + taken + step_off;
              seg_v[rk] = ink_r7_vt_cap(row_eq[idx], cap);
              seg_i[rk] = idx;
            }
            taken += step_tot;
          }
        }
      } else {
#pragma clang loop unroll(disable)
        for (u32 q = 0u; q < 4u; q++) {
          if (taken < need) {
            const u32 m4 = (eq_mask >> (4u * q)) & 0xFu;
            const u32 cnt4 = (u32)__builtin_popcount(m4);
            u32 step_tot = 0u;
            const u32 step_off = ink_r7_vt_wg_scan_u32(cnt4, ink_r7_vt_tk_lds.wsum[q & 1u], &step_tot);
            if (m4) {
              u32 off = taken + step_off;
              const u32 base = i0 + ((q * NT + t) << 2u);
#pragma unroll
              for (u32 e = 0u; e < 4u; e++) {
                if (m4 & (1u << e)) {
                  if (off < need) {
                    seg_v[total_gt + off] = ink_r7_vt_cap(row_eq[base + e], cap);
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
    }

    ink_r7_vt_rank_write(cand_n, K, seg_v, seg_i);
    if (tau_k == 0u && cand_n < K && t < K - cand_n) {
      seg_v[cand_n + t] = -__builtin_inff();
      seg_i[cand_n + t] = 0xFFFFFFFFu;
    }
  }

  if (need_lse && t == 0u) {
    float tm = ink_r7_vt_tk_lds.lse_m[0u], tl = ink_r7_vt_tk_lds.lse_l[0u];
#pragma clang loop unroll(disable)
    for (u32 w = 1u; w < 8u; w++) {
      const float wm = ink_r7_vt_tk_lds.lse_m[w], wl = ink_r7_vt_tk_lds.lse_l[w];
      if (wm > tm) {
        tl = tl * __builtin_expf(tm - wm) + wl;
        tm = wm;
      } else if (tm > -__builtin_inff()) {
        tl += wl * __builtin_expf(wm - tm);
      }
    }
    if (ng == 1u) {
      smax[2048u + r] = tl > 0.0f ? (tm + __builtin_logf(tl)) : -__builtin_inff();
    } else {
      smax[r * 64u + g] = tm;
      smax[1024u + r * 64u + g] = tl;
    }
  }

  if (ng == 1u) return;

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (t == 0u) {
    ink_r7_vt_tk_lds.last =
        (__scoped_atomic_fetch_add(done + r, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r7_vt_tk_lds.last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  if (t == 0u) done[r] = 0u;

  /* Stage 2 (last WG of row r): merge nc = ng * K <= 4096 candidates. */
  const u32 nc = ng * K;
  const G float *cv_row = cv + (unsigned long)r * (unsigned long)nc;
  const G u32   *ci_row = ci + (unsigned long)r * (unsigned long)nc;
  G float *ov_row = ov + (unsigned long)r * (unsigned long)K;
  G u32   *oi_row = oi + (unsigned long)r * (unsigned long)K;

  if (nc <= NT) {
    float mv = -__builtin_inff();
    u32 mi = 0xFFFFFFFFu;
    if (t < nc) {
      mv = cv_row[t];
      mi = ci_row[t];
      ink_r7_vt_tk_lds.u.cand.cv[t] = mv;
      ink_r7_vt_tk_lds.u.cand.ci[t] = mi;
    }
    const u32 is_val = (t < nc && mi != 0xFFFFFFFFu) ? 1u : 0u;
    const u32 wcnt = (u32)__builtin_popcount(__builtin_amdgcn_ballot_w32(is_val != 0u));
    if (lane() == 0u) {
      const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
      ink_r7_vt_tk_lds.wsum[0u][w] = wcnt;
    }
    lbarrier();
    u32 nvalid = 0u;
#pragma clang loop unroll(disable)
    for (u32 j = 0u; j < 8u; j++) nvalid += ink_r7_vt_tk_lds.wsum[0u][j];
    if (is_val) {
      u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
      for (u32 u = 0u; u < nc; u++) {
        const u32 ui = ink_r7_vt_tk_lds.u.cand.ci[u];
        if (ui != 0xFFFFFFFFu) {
          const float b = ink_r7_vt_tk_lds.u.cand.cv[u];
          rank += (u32)((b > mv) | ((b == mv) & (ui < mi)));
        }
      }
      if (rank < K) {
        ov_row[rank] = mv;
        oi_row[rank] = mi;
      }
    }
    if (nvalid < K && t < K - nvalid) {
      ov_row[nvalid + t] = -__builtin_inff();
      oi_row[nvalid + t] = 0xFFFFFFFFu;
    }
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

    const u32 tm2_k = vm2 ? ink_r7_vt_f2u(tm2) : 0u;
    const u32 tau2_k = ink_r7_vt_bitonic_kth_256(tm2_k, K);
    const float tau2 = tau2_k ? ink_r7_vt_u2f(tau2_k) : -__builtin_inff();

    u32 gm2 = 0u, em2 = 0u;
    ink_r7_vt_classify16(mv, vm2, tm2_k, tau2_k, tau2, &gm2, &em2);

    const u32 my_gt2 = (u32)__builtin_popcount(gm2);
    const u32 my_eq2 = (u32)__builtin_popcount(em2);
    u32 ptot2 = 0u;
    const u32 poff2 = ink_r7_vt_wg_scan_u32((my_eq2 << 16u) | my_gt2, ink_r7_vt_tk_lds.wsum[1u], &ptot2);
    const u32 bgt2 = poff2 & 0xFFFFu;
    const u32 beq2 = poff2 >> 16u;
    const u32 tgt2 = ptot2 & 0xFFFFu;
    const u32 teq2 = ptot2 >> 16u;

    if (gm2) {
      u32 pos = bgt2;
#pragma unroll
      for (u32 k = 0u; k < 16u; k++) {
        if (gm2 & (1u << k)) {
          ink_r7_vt_tk_lds.u.cand.cv[pos] = mv[k];
          ink_r7_vt_tk_lds.u.cand.ci[pos] = ci_row[t + k * NT];
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
            ink_r7_vt_tk_lds.u.cand.cv[pos] = mv[k];
            ink_r7_vt_tk_lds.u.cand.ci[pos] = ci_row[t + k * NT];
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
          const u32 soff = ink_r7_vt_wg_scan_u32(hit, ink_r7_vt_tk_lds.wsum[k & 1u], &stot);
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

    ink_r7_vt_rank_write(c2, K, ov_row, oi_row);
    if (tau2_k == 0u && c2 < K && t < K - c2) {
      ov_row[c2 + t] = -__builtin_inff();
      oi_row[c2 + t] = 0xFFFFFFFFu;
    }
  }

  if (need_lse && t == 0u) {
    float rm = -__builtin_inff(), rl = 0.0f;
#pragma clang loop unroll(disable)
    for (u32 u = 0u; u < ng; u++) {
      const float um = smax[r * 64u + u], ul = smax[1024u + r * 64u + u];
      if (um > rm) {
        rl = rl * __builtin_expf(rm - um) + ul;
        rm = um;
      } else if (rm > -__builtin_inff()) {
        rl += ul * __builtin_expf(um - rm);
      }
    }
    smax[2048u + r] = rl > 0.0f ? (rm + __builtin_logf(rl)) : -__builtin_inff();
  }
}

#undef INK_R7_VT_CAP
#undef INK_R7_VT_CSMAX
#undef INK_R7_VT_NGMAX
#undef INK_R7_VT_KMAX

#endif /* IE_MEGA */
