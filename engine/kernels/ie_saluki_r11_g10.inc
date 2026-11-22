/* ============================================================================
 * Round 11 Group 10: Single-Launch, Single-Pass, Zero-Overflow Top-K + LSE
 *                    for Saluki 27B on RX 9070 XT (gfx1201, wave32, NT=256)
 *
 * Included at the end of kernels/ie_saluki_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed r11_g10_ /
 * R11_G10_ and all macros are #undef'd at EOF.
 *
 * Kernel exported (100% bitwise-exact replacement for ie_topk_max + ie_topk_sel):
 *   ie_saluki_topk_r11
 *
 * Key optimizations over 2-launch, 2-pass ie_topk_max + ie_topk_sel:
 *   1. Fuses both launches into ONE kernel launch (eliminates 2,391 launches,
 *      inter-kernel drain/barrier overhead, and the entire 2nd full-vocab read).
 *   2. Single-pass global memory read:
 *      - need_lse == 0 (target verify: nv=248320, T=8, K=20): eliminates ALL
 *        1.99M expf() calls and loads logits via 128-bit f32x4 (global_load_b128)
 *        using 4-byte-aligned packed loads (zero C11 UB).
 *      - need_lse != 0 (MTP draft: nv=65536, T=1..2, K=8): issues all 16
 *        coalesced global_load_b32 loads in flight before the exact register
 *        online-softmax loop, matching ie_topk_max + ie_topk_sel bit-for-bit
 *        (0 ULP diff on lse[r]).
 *   3. Zero-overflow in-workgroup threshold selection (AirTopK / Bitonic-256
 *      pigeonhole bound):
 *      - tau = K-th largest of 256 per-thread maxima (via zero-bank-conflict
 *        double-buffered 256-thread bitonic sort on canonical u32 float keys).
 *      - Strictly fewer than K threads exceed tau -> strictly-greater count
 *        total_gt <= (K - 1) * 16 <= 1008 < 1024 ALWAYS (zero overflow, zero
 *        slow-path fallback even on all-equal or all -inf inputs!).
 *      - Packed 16+16-bit wave32 prefix scan computes (gt, eq) offsets in one
 *        barrier; exact ascending-index tie-breaker handles heavy ties at tau
 *        while preserving original float bit patterns (incl. -0.0f vs +0.0f).
 *      - 240+ threads skip candidate filtering via tmax_k < tau_k.
 *   4. Stage-2 last-WG merge via agent release/acquire + done[r] counter:
 *      - Fast path nc = ng * K <= 256 (ALL MTP 64K K=8 calls: nc = 128):
 *        single-barrier LDS rank-and-write (~1 KB L2 read, 0 sorting).
 *      - General path nc > 256 (target verify: nc = 1220, up to 4096):
 *        reuses the exact zero-overflow Bitonic-256 threshold selector
 *        (re-reading L1-hot ci_row only for surviving candidates to save 16
 *        VGPRs).
 *   5. Low-SGPR / 0-spill codegen hygiene:
 *      - Rolled Bitonic-256 & wave32 scan loops (uniform readfirstlane wave id),
 *        non-vectorized rank loop (prevents 48-SGPR v_cmp explosion), rolled
 *        rare heavy-tie loops (re-reading L1-hot input for <=64 ties), and late
 *        pointer materialization keep SGPRs ~45-60 (0 SGPR spills, 0 VGPR
 *        spills, 0 B private scratch, 8,336 B LDS -> 8 WGs/WGP max occupancy).
 * ========================================================================== */

#define R11_G10_KMAX 64u
#define R11_G10_NGMAX 64u
#define R11_G10_CSMAX 4096u
#define R11_G10_CAP 1024u

/* 4-byte-aligned f32x4 load helper (emits global_load_b128 on gfx1201 with
 * zero C11 alignment UB for any 4-byte-aligned float pointer). */
struct __attribute__((packed, aligned(4))) r11_g10_f32x4_a4 { f32x4 v; };

/* 8,336 B unioned LDS -> 8 WGs/WGP (4 WGs/CU = 100% max occupancy on gfx1201).
 * Disjoint phase lifetimes (proven by intervening lbarrier()):
 *   - u.sm[2][256] (2,048 B): double-buffered bitonic-256 sort keys; read of
 *     u.sm[0][0] completes before the wg_scan lbarrier() precedes u.cand writes.
 *   - u.cand.{cv,ci}[1024] (8,192 B): surviving candidates for exact ranking.
 *   - wsum[2][8] (64 B): double-buffered wave scan totals.
 *   - lse_m[8], lse_l[8] (64 B): per-wave online softmax state (Stage 1 only).
 *   - last (4 B): last-WG election flag. */
typedef struct {
  union {
    u32 sm[2][NT];
    struct {
      float cv[R11_G10_CAP];
      u32   ci[R11_G10_CAP];
    } cand;
  } u;
  u32   wsum[2][8];
  float lse_m[8];
  float lse_l[8];
  u32   last;
} r11_g10_lds_t;

static LDS r11_g10_lds_t r11_g10_lds __attribute__((aligned(16)));
_Static_assert(sizeof(r11_g10_lds_t) <= 16384u, "r11_g10_lds must stay <= 16 KiB");

/* Order-preserving bijection from non-NaN IEEE-754 float to u32 in [0x007FFFFF, 0xFF800000].
 * Key 0u is reserved for "no valid (non-NaN) element", strictly below -inf (0x007FFFFF).
 * Canonicalizes -0.0f to +0.0f so both map to 0x80000000u (matching float == comparison);
 * original float bit patterns are preserved separately in rv[] and written unmodified to ov[]. */
static inline __attribute__((always_inline)) u32 r11_g10_f2u(float f) {
  u32 u = __builtin_bit_cast(u32, f);
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static inline __attribute__((always_inline)) float r11_g10_u2f(u32 k) {
  const u32 u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  return __builtin_bit_cast(float, u);
}

/* Bitonic compare-exchange: asc==0 sorts descending (low gets max), asc==1 ascending (low gets min). */
static inline __attribute__((always_inline)) u32 r11_g10_cswap(u32 v, u32 ov, u32 hi, u32 asc) {
  return (hi ^ asc) ? (v < ov ? v : ov) : (v > ov ? v : ov);
}

static inline __attribute__((always_inline)) u32 r11_g10_wave_step(u32 v, u32 t, u32 stride, u32 asc) {
  const u32 ov = (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ stride) * 4u), (int)v);
  return r11_g10_cswap(v, ov, (u32)((t & stride) != 0u), asc);
}

static inline __attribute__((always_inline)) u32 r11_g10_lds_step(
    u32 v, u32 t, u32 stride, u32 asc, LDSP u32 *buf) {
  buf[t] = v;
  lbarrier();
  const u32 ov = buf[t ^ stride];
  return r11_g10_cswap(v, ov, (u32)((t & stride) != 0u), asc);
}

/* Exact K-th largest (1-indexed, 1 <= K <= 64) of 256 u32 keys (1 per thread).
 * Rolled loops prevent cross-stage SGPR predicate hoisting (saves ~28 SGPRs).
 * Bank-conflict-free (stride >= 32 keeps (t ^ stride) & 31 == t & 31) and
 * double-buffered across sm[0]/sm[1] for zero WAR hazards in 7 lbarriers.
 * Caller MUST execute an lbarrier() before writing r11_g10_lds.u.cand. */
static inline __attribute__((always_inline)) u32 r11_g10_bitonic_kth_256(u32 v, u32 K) {
  const u32 t = tid();
#pragma clang loop unroll(disable)
  for (u32 size = 2u; size <= 32u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride > 0u; stride >>= 1u) {
      v = r11_g10_wave_step(v, t, stride, a);
    }
  }
  u32 buf_id = 0u;
#pragma clang loop unroll(disable)
  for (u32 size = 64u; size <= 256u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride >= 32u; stride >>= 1u) {
      v = r11_g10_lds_step(v, t, stride, a, r11_g10_lds.u.sm[buf_id]);
      buf_id ^= 1u;
    }
#pragma clang loop unroll(disable)
    for (u32 stride = 16u; stride > 0u; stride >>= 1u) {
      v = r11_g10_wave_step(v, t, stride, a);
    }
  }
  /* Six cross-wave swaps alternate buf_id: 0,1,0,1,0,1, so all reads of sm[0]
   * completed before the last sm[1] lbarrier(). Broadcast sorted[K - 1] via sm[0][0]. */
  if (t == K - 1u) r11_g10_lds.u.sm[0][0] = v;
  lbarrier();
  return r11_g10_lds.u.sm[0][0];
}

/* 256-thread exclusive prefix sum (supports packed 16+16-bit non-overflowing lanes).
 * Uses wave-uniform readfirstlane wave id + rolled loops so 0 persistent SGPR
 * predicate masks are hoisted across the kernel. Contains one lbarrier(), also
 * retiring reads of u.sm[0][0]. */
static inline __attribute__((always_inline)) u32 r11_g10_wg_scan_u32(
    u32 v, LDSP u32 *wsum, u32 *total_out) {
  const u32 l = lane();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5u);
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

/* Exact rank-and-write of c <= 1008 non-NaN distinct-index candidates in r11_g10_lds.u.cand
 * into out_v[0..min(c,K)-1] / out_i[0..min(c,K)-1]. Value descending, equal values smaller index first.
 * Disabling vectorize/interleave/unroll prevents LLVM from hoisting 48 concurrent v_cmp SGPR masks. */
static inline __attribute__((always_inline)) void r11_g10_rank_write(
    u32 c, u32 K, G float *out_v, G u32 *out_i) {
  for (u32 j = tid(); j < c; j += NT) {
    const float a = r11_g10_lds.u.cand.cv[j];
    const u32 ai = r11_g10_lds.u.cand.ci[j];
    u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
    for (u32 u = 0u; u < c; u++) {
      const float b = r11_g10_lds.u.cand.cv[u];
      rank += (u32)((b > a) | ((b == a) & (r11_g10_lds.u.cand.ci[u] < ai)));
    }
    if (rank < K) {
      out_v[rank] = a;
      out_i[rank] = ai;
    }
  }
}

/* Compute (gt_mask, eq_mask) over a thread's 16 register floats given (vmask, tmax_k, tau_k, tau).
 * All rv[k] accesses use compile-time constant k -> 100% VGPR-resident, 0 scratch spills. */
static inline __attribute__((always_inline)) void r11_g10_classify16(
    const float rv[16], u32 vmask, u32 tmax_k, u32 tau_k, float tau,
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

/* Single-launch, single-pass Top-K + exact LSE kernel.
 * Grid: (ng, nrows, 1), block: (256, 1, 1), LDS: 8,336 B.
 * Contract (exact bitwise match to ie_topk_max + ie_topk_sel):
 *   - ov[r*K + 0..K-1], oi[r*K + 0..K-1]: top K non-NaN logits per row r, value descending,
 *     equal values broken by smaller token index first; missing entries padded (-inf, 0xFFFFFFFFu).
 *   - If need_lse != 0: smax[2048u + r] = exact row log-sum-exp (0 ULP vs ie_topk_max + ie_topk_sel).
 *   - Buffers:
 *       cv, ci: TK_ROWS * TK_NG * TK_MAX entries each (disjoint from smax and done).
 *       smax:   2064 floats ([0..1023] seg max, [1024..2047] seg exp-sum, [2048..2063] row lse).
 *       done:   TK_ROWS zero-initialized u32 counters (auto-reset to 0 by each row's last WG). */
KERNEL ie_saluki_topk_r11(const G float *x, u32 n, u32 xs, u32 K, u32 cs, u32 ng, u32 need_lse,
                          G float *smax, G float *cv, G u32 *ci, G u32 *done,
                          G float *ov, G u32 *oi) {
  const u32 r = tokid(), g = wgid();
  if (g >= ng || n == 0u || K == 0u || K > R11_G10_KMAX || ng == 0u || ng > R11_G10_NGMAX ||
      cs == 0u || cs > R11_G10_CSMAX) return;

  const u32 t = tid();

  /* ========================================================================
   * STAGE 1: Single-pass load (+ exact online softmax iff need_lse != 0)
   *          and zero-overflow exact per-segment Top-K -> seg_v / seg_i.
   * ====================================================================== */
  float rv[16];
  u32 vmask = 0u;
  float tmax = -__builtin_inff();
  u32 i0 = 0u;

  {
    const G float *row = x + (unsigned long)r * xs;
    if (need_lse) {
      /* Exact ie_topk_max segment partition & accumulation order (0 ULP on lse).
       * Issue all 16 coalesced global_load_b32 loads in flight before expf(). */
      i0 = g * cs;
      u32 in_bounds = 0u;
      if (cs == R11_G10_CSMAX && i0 + R11_G10_CSMAX <= n) {
#pragma unroll
        for (u32 k = 0u; k < 16u; k++) rv[k] = row[i0 + t + k * NT];
        in_bounds = 0xFFFFu;
      } else {
#pragma unroll
        for (u32 k = 0u; k < 16u; k++) {
          const u32 p = t + k * NT;
          const u32 ok = (p < cs && i0 + p < n);
          rv[k] = ok ? row[i0 + p] : -__builtin_inff();
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
        const u32 w = (u32)__builtin_amdgcn_readfirstlane(t >> 5u);
        r11_g10_lds.lse_m[w] = m;
        r11_g10_lds.lse_l[w] = l;
      }
    } else {
      /* need_lse == 0: 0 expf() calls; 128-bit vectorized f32x4 loads over contiguous
       * 4-float-aligned segments of cs4 = ceil4(cs) <= 4096 floats per WG. */
      const u32 cs4 = (cs + 3u) & ~3u;
      const u32 nvec = cs4 >> 2u;
      i0 = g * cs4;
      u32 in_bounds = 0u;
#pragma unroll
      for (u32 q = 0u; q < 4u; q++) {
        const u32 vidx = t + q * NT;
        const u32 base = i0 + vidx * 4u;
        if (vidx < nvec && base + 4u <= n) {
          const f32x4 v4 = ((const G struct r11_g10_f32x4_a4 *)(row + base))->v;
          rv[4u * q + 0u] = v4[0];
          rv[4u * q + 1u] = v4[1];
          rv[4u * q + 2u] = v4[2];
          rv[4u * q + 3u] = v4[3];
          in_bounds |= (0xFu << (4u * q));
        } else if (vidx < nvec && base < n) {
#pragma unroll
          for (u32 e = 0u; e < 4u; e++) {
            const u32 ok = (base + e < n);
            rv[4u * q + e] = ok ? row[base + e] : -__builtin_inff();
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

  /* Per-WG threshold tau_k = K-th largest of 256 per-thread maxima. */
  const u32 tmax_k = vmask ? r11_g10_f2u(tmax) : 0u;
  const u32 tau_k = r11_g10_bitonic_kth_256(tmax_k, K);
  const float tau = tau_k ? r11_g10_u2f(tau_k) : -__builtin_inff();

  u32 gt_mask = 0u, eq_mask = 0u;
  r11_g10_classify16(rv, vmask, tmax_k, tau_k, tau, &gt_mask, &eq_mask);

  const u32 my_gt = (u32)__builtin_popcount(gt_mask);
  const u32 my_eq = (u32)__builtin_popcount(eq_mask);
  u32 packed_tot = 0u;
  const u32 packed_off = r11_g10_wg_scan_u32((my_eq << 16u) | my_gt, r11_g10_lds.wsum[1], &packed_tot);
  const u32 base_gt = packed_off & 0xFFFFu;
  const u32 base_eq = packed_off >> 16u;
  const u32 total_gt = packed_tot & 0xFFFFu;
  const u32 total_eq = packed_tot >> 16u;

  /* Compact strictly-greater candidates (guaranteed total_gt <= (K-1)*16 <= 1008 < 1024). */
  if (gt_mask) {
    u32 pos = base_gt;
    if (need_lse) {
#pragma unroll
      for (u32 k = 0u; k < 16u; k++) {
        if (gt_mask & (1u << k)) {
          r11_g10_lds.u.cand.cv[pos] = rv[k];
          r11_g10_lds.u.cand.ci[pos] = i0 + t + k * NT;
          pos++;
        }
      }
    } else {
#pragma unroll
      for (u32 q = 0u; q < 4u; q++) {
        const u32 base = i0 + (q * NT + t) * 4u;
#pragma unroll
        for (u32 e = 0u; e < 4u; e++) {
          if (gt_mask & (1u << (4u * q + e))) {
            r11_g10_lds.u.cand.cv[pos] = rv[4u * q + e];
            r11_g10_lds.u.cand.ci[pos] = base + e;
            pos++;
          }
        }
      }
    }
  }

  const u32 need = (total_gt < K) ? (K - total_gt) : 0u;
  u32 cand_n = total_gt;
  if (need > 0u && total_eq <= need) {
    /* Fast path (99.99% of calls): all equal-to-tau candidates fit (total <= K <= 64). */
    if (eq_mask) {
      u32 pos = total_gt + base_eq;
      if (need_lse) {
#pragma unroll
        for (u32 k = 0u; k < 16u; k++) {
          if (eq_mask & (1u << k)) {
            r11_g10_lds.u.cand.cv[pos] = rv[k];
            r11_g10_lds.u.cand.ci[pos] = i0 + t + k * NT;
            pos++;
          }
        }
      } else {
#pragma unroll
        for (u32 q = 0u; q < 4u; q++) {
          const u32 base = i0 + (q * NT + t) * 4u;
#pragma unroll
          for (u32 e = 0u; e < 4u; e++) {
            if (eq_mask & (1u << (4u * q + e))) {
              r11_g10_lds.u.cand.cv[pos] = rv[4u * q + e];
              r11_g10_lds.u.cand.ci[pos] = base + e;
              pos++;
            }
          }
        }
      }
    }
    cand_n = total_gt + total_eq;
  }
  lbarrier(); /* u.cand visible, all reads of wsum[1] done, rv[16] now dead */

  /* Late Stage-1 output pointer materialization (keeps pointer SGPRs off Stage-1 hot path). */
  {
    const unsigned long seg_off = (ng == 1u) ? ((unsigned long)r * K) : (((unsigned long)r * ng + g) * K);
    G float *seg_v = (ng == 1u) ? (ov + seg_off) : (cv + seg_off);
    G u32   *seg_i = (ng == 1u) ? (oi + seg_off) : (ci + seg_off);

    if (need > 0u && total_eq > need) {
      /* Rare heavy-tie path (e.g. all-equal or all -inf): rolled loops avoid 20x
       * wg_scan inlining; re-read L1-hot row[idx] only for the <= 64 winning ties
       * (preserves exact -0.0f vs +0.0f bits with 0 rv[] scratch spills). */
      const G float *row_eq = x + (unsigned long)r * xs;
      u32 taken = 0u;
      if (need_lse) {
#pragma clang loop unroll(disable)
        for (u32 k = 0u; k < 16u; k++) {
          if (taken < need) {
            const u32 hit = (eq_mask >> k) & 1u;
            u32 step_tot = 0u;
            const u32 step_off = r11_g10_wg_scan_u32(hit, r11_g10_lds.wsum[k & 1u], &step_tot);
            if (hit && taken + step_off < need) {
              const u32 idx = i0 + t + k * NT;
              const u32 rk = total_gt + taken + step_off;
              seg_v[rk] = row_eq[idx];
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
            const u32 step_off = r11_g10_wg_scan_u32(cnt4, r11_g10_lds.wsum[q & 1u], &step_tot);
            if (m4) {
              u32 off = taken + step_off;
              const u32 base = i0 + (q * NT + t) * 4u;
#pragma unroll
              for (u32 e = 0u; e < 4u; e++) {
                if (m4 & (1u << e)) {
                  if (off < need) {
                    seg_v[total_gt + off] = row_eq[base + e];
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

    r11_g10_rank_write(cand_n, K, seg_v, seg_i);
    if (tau_k == 0u && cand_n < K && t < K - cand_n) {
      seg_v[cand_n + t] = -__builtin_inff();
      seg_i[cand_n + t] = 0xFFFFFFFFu;
    }
  }

  /* Complete per-segment exact softmax reduction (lse_m/lse_l written >8 lbarriers ago). */
  if (need_lse && t == 0u) {
    float tm = r11_g10_lds.lse_m[0], tl = r11_g10_lds.lse_l[0];
#pragma clang loop unroll(disable)
    for (u32 w = 1u; w < 8u; w++) {
      const float wm = r11_g10_lds.lse_m[w], wl = r11_g10_lds.lse_l[w];
      if (wm > tm) {
        tl = tl * __builtin_expf(tm - wm) + wl;
        tm = wm;
      } else if (tm > -__builtin_inff()) {
        tl += wl * __builtin_expf(wm - tm);
      }
    }
    if (ng == 1u) {
      smax[2048u + r] = tl > 0.0f ? tm + __builtin_logf(tl) : -__builtin_inff();
    } else {
      smax[r * 64u + g] = tm;
      smax[1024u + r * 64u + g] = tl;
    }
  }

  if (ng == 1u) return;

  /* ========================================================================
   * INTER-WG HANDOFF: Publish seg_v/seg_i (+ smax) and elect last WG of row r.
   * ====================================================================== */
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __builtin_amdgcn_s_barrier();
  if (t == 0u)
    r11_g10_lds.last = (__scoped_atomic_fetch_add(done + r, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u);
  lbarrier();
  if (!r11_g10_lds.last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  if (t == 0u) done[r] = 0u;

  /* ========================================================================
   * STAGE 2 (Last WG of row r): Merge nc = ng * K <= 4096 segment winners
   *          into ov_row / oi_row (+ exact row lse[r] iff need_lse != 0).
   * ====================================================================== */
  const u32 nc = ng * K;
  const G float *cv_row = cv + (unsigned long)r * nc;
  const G u32   *ci_row = ci + (unsigned long)r * nc;
  G float *ov_row = ov + (unsigned long)r * K;
  G u32   *oi_row = oi + (unsigned long)r * K;

  if (nc <= NT) {
    /* Fast path (covers ALL MTP 64K K=8 calls: nc = 16 * 8 = 128): 1-barrier LDS rank. */
    float mv = -__builtin_inff();
    u32 mi = 0xFFFFFFFFu;
    if (t < nc) {
      mv = cv_row[t];
      mi = ci_row[t];
      r11_g10_lds.u.cand.cv[t] = mv;
      r11_g10_lds.u.cand.ci[t] = mi;
    }
    const u32 is_val = (t < nc && mi != 0xFFFFFFFFu) ? 1u : 0u;
    const u32 wcnt = (u32)__builtin_popcount(__builtin_amdgcn_ballot_w32(is_val != 0u));
    if (lane() == 0u) {
      const u32 w = (u32)__builtin_amdgcn_readfirstlane(t >> 5u);
      r11_g10_lds.wsum[0][w] = wcnt;
    }
    lbarrier();
    u32 nvalid = 0u;
#pragma clang loop unroll(disable)
    for (u32 j = 0u; j < 8u; j++) nvalid += r11_g10_lds.wsum[0][j];
    if (is_val) {
      u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
      for (u32 u = 0u; u < nc; u++) {
        const u32 ui = r11_g10_lds.u.cand.ci[u];
        if (ui != 0xFFFFFFFFu) {
          const float b = r11_g10_lds.u.cand.cv[u];
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
    /* General path (covers target verify nc = 61 * 20 = 1220, up to 64 * 64 = 4096):
     * Reuse the exact zero-overflow Bitonic-256 selector. Re-read L1-hot ci_row only for
     * surviving candidates to avoid holding mi[16] in VGPRs.
     * Equal values across p = 0..nc-1 are strictly ascending in original token index ci_row[p]
     * because segments are contiguous disjoint index ranges sorted ascending on ties. */
    float mv[16];
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

    const u32 tm2_k = vm2 ? r11_g10_f2u(tm2) : 0u;
    const u32 tau2_k = r11_g10_bitonic_kth_256(tm2_k, K);
    const float tau2 = tau2_k ? r11_g10_u2f(tau2_k) : -__builtin_inff();

    u32 gm2 = 0u, em2 = 0u;
    r11_g10_classify16(mv, vm2, tm2_k, tau2_k, tau2, &gm2, &em2);

    const u32 my_gt2 = (u32)__builtin_popcount(gm2);
    const u32 my_eq2 = (u32)__builtin_popcount(em2);
    u32 ptot2 = 0u;
    const u32 poff2 = r11_g10_wg_scan_u32((my_eq2 << 16u) | my_gt2, r11_g10_lds.wsum[1], &ptot2);
    const u32 bgt2 = poff2 & 0xFFFFu;
    const u32 beq2 = poff2 >> 16u;
    const u32 tgt2 = ptot2 & 0xFFFFu;
    const u32 teq2 = ptot2 >> 16u;

    if (gm2) {
      u32 pos = bgt2;
#pragma unroll
      for (u32 k = 0u; k < 16u; k++) {
        if (gm2 & (1u << k)) {
          r11_g10_lds.u.cand.cv[pos] = mv[k];
          r11_g10_lds.u.cand.ci[pos] = ci_row[t + k * NT];
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
            r11_g10_lds.u.cand.cv[pos] = mv[k];
            r11_g10_lds.u.cand.ci[pos] = ci_row[t + k * NT];
            pos++;
          }
        }
      }
      c2 = tgt2 + teq2;
    }
    lbarrier(); /* u.cand visible, all reads of wsum[1] done, mv[16] now dead */

    if (need2 > 0u && teq2 > need2) {
      u32 taken2 = 0u;
#pragma clang loop unroll(disable)
      for (u32 k = 0u; k < 16u; k++) {
        if (taken2 < need2) {
          const u32 hit = (em2 >> k) & 1u;
          u32 stot = 0u;
          const u32 soff = r11_g10_wg_scan_u32(hit, r11_g10_lds.wsum[k & 1u], &stot);
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

    r11_g10_rank_write(c2, K, ov_row, oi_row);
    if (tau2_k == 0u && c2 < K && t < K - c2) {
      ov_row[c2 + t] = -__builtin_inff();
      oi_row[c2 + t] = 0xFFFFFFFFu;
    }
  }

  /* Final row LSE reduction on thread 0 (exact ie_topk_sel order, 0 ULP diff). */
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
    smax[2048u + r] = rl > 0.0f ? rm + __builtin_logf(rl) : -__builtin_inff();
  }
}

#undef R11_G10_CAP
#undef R11_G10_CSMAX
#undef R11_G10_NGMAX
#undef R11_G10_KMAX
