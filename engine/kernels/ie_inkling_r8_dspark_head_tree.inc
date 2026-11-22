/* ============================================================================
 * Inkling-2 Round 8: Fused DSpark P+Sample (Bit 85), Adaptive MoE Verify
 *                    (Bit 86), & EAGLE-2/Sequoia Tree Verify Attn (Bit 89)
 * Target: RX 9070 XT (gfx1201, wave32, NT=256)
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r8_dh_ /
 * INK_R8_DH_ and all macros are #undef'd at EOF.
 *
 * Exported kernels:
 *
 * 1. Bit 85 (R27): KERNEL ie_inkling_ds_gemv_p_cap_sample_zsync_r8(...)
 *    - 1-launch persistent fused DSpark Q4_0 P*lat + base_row + softcap +
 *      online Argmax (temp <= 0) OR Top-K/top_p/ForeSpark/xorshift64* sample
 *      (temp > 0)! Supports BOTH full vocab (id_map == NULL) AND R30 60K+4K
 *      shortlist (id_map != NULL, canonical id_map[s] tie-breaking).
 *    - Completely eliminates writing & re-reading b->ds.logits (800 KiB x 2 per
 *      draft step = 8-11 MB DRAM traffic + 5-7 launches saved per step!).
 *    - Preserves exact ds_ctrl[4] = {prev_tok, stop_flag, n_ran, cum_surv_bits}
 *      ABI for zero-diff compatibility with all R5 zsync kernels.
 *    - Grid: (ng, 1u, 1u) with 1 <= ng <= 64 (e.g. ng=64 = 1 WG/CU -> nc=ng*K<=2048).
 *
 * 2. Bit 86 (R28): Adaptive 1-Route vs 2..4-Route Specialized Verify MoE
 *    - ie_inkling_moe_sort_adap_r8: 1-WG sort emitting exact offs/list PLUS two
 *      disjoint compact pack lists: packs1[0..*n_packs1-1] (nj==1) and
 *      packsM[0..*n_packsM-1] (nj==2..4) -> 0 wasted WGs!
 *    - ie_inkling_gexp_iq3xxs_situ_nj1_r8: lean single-pair IQ3+SITU+Q8 (~56 VGPR!).
 *    - ie_inkling_gexp_iq3xxs_situ_njm_r8: 2..4-pair shared-weight IQ3+SITU+Q8.
 *    - ie_inkling_gexp_iq4nl_nj1_r8:       lean single-pair IQ4_NL (~48 VGPR!).
 *    - ie_inkling_gexp_iq4nl_njm_r8:       2..4-pair shared-weight IQ4_NL.
 *    (Also safe with unified packs[] from r6_compact: nj1 skips nj!=1 & njm skips nj<2!)
 *    - Parity: 100% bitwise-exact (0-bit-diff).
 *
 * 3. Bit 89 (R32): KERNEL ie_inkling_rel_attn_ver_tree_r8(...)
 *    - Split-KV GQA-3 f16-KV relative attention for T <= 8 speculative tree
 *      verification (EAGLE-2 / Sequoia, both SWA win>0 and Global win==0!).
 *    - Same 23-arg base signature & part/count scratch as ie_inkling_rel_attn_ver_r4,
 *      plus:
 *        * u64 tree_mask: 8x8 packed ancestor bits; bit (t*8 + j) == 1 iff
 *          tree node j (stored at KV pos0+j) is an ancestor of (or equal to) node t.
 *        * u32 pos_off_packed: 4-bit logical tree depth per node (node j depth =
 *          (pos_off_packed >> (4*j)) & 0xFu; logical_pos(j) = pos0 + dep_j).
 *    - Reduces identically to ie_inkling_rel_attn_ver_r4 for linear chains
 *      (lower-triangular tree_mask, pos_off_packed = 0x76543210u).
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R8_DH_KMAX      32u
#define INK_R8_DH_NGMAX     64u
#define INK_R8_DH_CAP       1024u
#define INK_R8_DH_SB_BYTES  98u
#define INK_R8_DH_MAX_EXP   256u
#define INK_R8_DH_MAX_PAIRS 4096u
#define INK_R8_DH_HD        128u
#define INK_R8_DH_DREL      16u
#define INK_R8_DH_GQA       3u
#define INK_R8_DH_PSTR      130u

typedef unsigned long long ink_r8_dh_u64;

struct __attribute__((packed, aligned(4))) ink_r8_dh_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r8_dh_ld4(const G void *p) {
  return ((const G struct ink_r8_dh_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) float ink_r8_dh_cap(float v, float cap) {
  return (cap > 0.0f) ? (cap * ie_tanhf_s(v / cap)) : v;
}

static inline __attribute__((always_inline)) int ink_r8_dh_better(float v, u32 i, float bv, u32 bi) {
  return (v > bv) || (v == bv && i < bi);
}

static inline __attribute__((always_inline)) u32 ink_r8_dh_div_x(u32 pid, u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

static inline __attribute__((always_inline)) float ink_r8_dh_oct_tree_f(float v) {
#pragma clang fp contract(off)
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}

static inline __attribute__((always_inline)) u32 ink_r8_dh_f2u(float f) {
  u32 u = __builtin_bit_cast(u32, f);
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static inline __attribute__((always_inline)) float ink_r8_dh_u2f(u32 k) {
  const u32 u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  return __builtin_bit_cast(float, u);
}

static inline __attribute__((always_inline)) u32 ink_r8_dh_cswap(u32 v, u32 ov, u32 hi, u32 asc) {
  return (hi ^ asc) ? (v < ov ? v : ov) : (v > ov ? v : ov);
}

static inline __attribute__((always_inline)) u32 ink_r8_dh_wave_step(u32 v, u32 t, u32 stride, u32 asc) {
  const u32 ov = (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ stride) * 4u), (int)v);
  return ink_r8_dh_cswap(v, ov, (u32)((t & stride) != 0u), asc);
}

static inline __attribute__((always_inline)) u32 ink_r8_dh_lds_step(
    u32 v, u32 t, u32 stride, u32 asc, LDSP u32 *buf) {
  buf[t] = v;
  lbarrier();
  const u32 ov = buf[t ^ stride];
  return ink_r8_dh_cswap(v, ov, (u32)((t & stride) != 0u), asc);
}

static inline __attribute__((always_inline)) u32 ink_r8_dh_wave_scan_inc(u32 x) {
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x111, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x112, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x114, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x118, 0xF, 0xF, true);
  const u32 lo16 = (u32)__builtin_amdgcn_readlane((int)x, 15);
  if (lane() >= 16u) x += lo16;
  return x;
}

static inline __attribute__((always_inline)) float ink_r8_dh_q4q8_blk(
    const G u32 *row_qw, const G f16 *row_qs, const G u32 *aw,
    const G float *da, const G u32 *asum, u32 b) {
  const u32x4 w4 = ink_r8_dh_ld4(row_qw + (unsigned long)b * 4ul);
  const u32x4 a0 = ink_r8_dh_ld4(aw + (unsigned long)b * 8ul);
  const u32x4 a1 = ink_r8_dh_ld4(aw + (unsigned long)b * 8ul + 4ul);
  const float sc = (float)row_qs[b] * da[b];
  const u32 as = asum[b];
  u32 d = ie_q4q8_word(0u, w4.x, a0.x, a1.x);
  d = ie_q4q8_word(d, w4.y, a0.y, a1.y);
  d = ie_q4q8_word(d, w4.z, a0.z, a1.z);
  d = ie_q4q8_word(d, w4.w, a0.w, a1.w);
  const int dot = (int)(d - 8u * as);
  return sc * (float)dot;
}

/* ============================================================================
 * Kernel 1 (Bit 85): ie_inkling_ds_gemv_p_cap_sample_zsync_r8
 * Persistent 1-launch fused Q4_0 P*lat + base_row + softcap + argmax/sample.
 * Zero b->ds.logits writes or reads!
 * ========================================================================== */

typedef struct {
  union {
    struct {
      float kv[INK_R8_DH_KMAX];
      u32   ki[INK_R8_DH_KMAX];
      float ch_v[32u];
      u32   ch_i[32u];
      float min_v;
      u32   min_i;
      u32   min_pos;
      u32   any_pass;
    } st1;
    u32 sm[2u][NT];
    struct {
      float cv[INK_R8_DH_CAP];
      u32   ci[INK_R8_DH_CAP];
    } cand;
    struct {
      float wv[8u];
      u32   wi[8u];
      float ws[8u];
    } gr;
  } u;
  u32   wsum[2u][8u];
  float fv[INK_R8_DH_KMAX];
  u32   fi[INK_R8_DH_KMAX];
  float fq[INK_R8_DH_KMAX];
  u32   last;
} ink_r8_dh_ds_lds_t;

static LDS ink_r8_dh_ds_lds_t ink_r8_dh_dlds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r8_dh_ds_lds_t) <= 16384u, "ink_r8_dh_dlds must stay <= 16 KiB");

static inline __attribute__((always_inline)) void ink_r8_dh_gr_best(float *bv, u32 *bi) {
  const u32 l = lane();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
#pragma unroll
  for (u32 m = 16u; m > 0u; m >>= 1u) {
    const float ov = shfl_xor_f(*bv, m);
    const u32 oi = (u32)__builtin_amdgcn_ds_bpermute((int)((l ^ m) * 4u), (int)*bi);
    if (ink_r8_dh_better(ov, oi, *bv, *bi)) { *bv = ov; *bi = oi; }
  }
  if (l == 0u) { ink_r8_dh_dlds.u.gr.wv[w] = *bv; ink_r8_dh_dlds.u.gr.wi[w] = *bi; }
  lbarrier();
  float v = ink_r8_dh_dlds.u.gr.wv[0u];
  u32 idx = ink_r8_dh_dlds.u.gr.wi[0u];
#pragma clang loop unroll(disable)
  for (u32 j = 1u; j < 8u; j++) {
    if (ink_r8_dh_better(ink_r8_dh_dlds.u.gr.wv[j], ink_r8_dh_dlds.u.gr.wi[j], v, idx)) {
      v = ink_r8_dh_dlds.u.gr.wv[j]; idx = ink_r8_dh_dlds.u.gr.wi[j];
    }
  }
  *bv = v; *bi = idx;
}

static inline __attribute__((always_inline)) u32 ink_r8_dh_bitonic(u32 v, u32 K) {
  const u32 t = tid();
#pragma clang loop unroll(disable)
  for (u32 size = 2u; size <= 32u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride > 0u; stride >>= 1u) v = ink_r8_dh_wave_step(v, t, stride, a);
  }
  u32 buf_id = 0u;
#pragma clang loop unroll(disable)
  for (u32 size = 64u; size <= 256u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride >= 32u; stride >>= 1u) {
      v = ink_r8_dh_lds_step(v, t, stride, a, ink_r8_dh_dlds.u.sm[buf_id]);
      buf_id ^= 1u;
    }
#pragma clang loop unroll(disable)
    for (u32 stride = 16u; stride > 0u; stride >>= 1u) v = ink_r8_dh_wave_step(v, t, stride, a);
  }
  if (t == K - 1u) ink_r8_dh_dlds.u.sm[0u][0u] = v;
  lbarrier();
  return ink_r8_dh_dlds.u.sm[0u][0u];
}

static inline __attribute__((always_inline)) u32 ink_r8_dh_wg_scan_u32(
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

static inline __attribute__((always_inline)) float ink_r8_dh_p_row_val(
    const G u32 *qw, const G f16 *qs, const G u8 *xq, const G float *base_row,
    u32 tok, u32 s, u32 nb, u32 sub, float cap) {
#pragma clang fp contract(off)
  const G u32 *row_qw = qw + (unsigned long)tok * (unsigned long)nb * 4ul;
  const G f16 *row_qs = qs + (unsigned long)tok * (unsigned long)nb;
  const G u32 *aw = (const G u32 *)xq;
  const G float *da = (const G float *)(xq + (unsigned long)nb * 32ul);
  const G u32 *asum = (const G u32 *)(xq + (unsigned long)nb * 36ul);

  float acc;
  if (nb == 8u) {
    acc = 0.0f + ink_r8_dh_q4q8_blk(row_qw, row_qs, aw, da, asum, sub);
  } else {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (u32 k0 = 0u; k0 < nb; k0 += 32u) {
      const u32 b0 = k0 + sub;
      if (b0 < nb) a0 = a0 + ink_r8_dh_q4q8_blk(row_qw, row_qs, aw, da, asum, b0);
      const u32 b1 = b0 + 8u;
      if (b1 < nb) a1 = a1 + ink_r8_dh_q4q8_blk(row_qw, row_qs, aw, da, asum, b1);
      const u32 b2 = b0 + 16u;
      if (b2 < nb) a2 = a2 + ink_r8_dh_q4q8_blk(row_qw, row_qs, aw, da, asum, b2);
      const u32 b3 = b0 + 24u;
      if (b3 < nb) a3 = a3 + ink_r8_dh_q4q8_blk(row_qw, row_qs, aw, da, asum, b3);
    }
    acc = (a0 + a2) + (a1 + a3);
  }
  acc = ink_r8_dh_oct_tree_f(acc);
  const float raw = base_row ? (base_row[s] + acc) : acc;
  return ink_r8_dh_cap(raw, cap);
}

KERNEL ie_inkling_ds_gemv_p_cap_sample_zsync_r8(
    const G u32 *p_qw, const G f16 *p_qs, const G u8 *lat_q8,
    const G float *base_row, const G u32 *id_map,
    G float *part_v, G u32 *part_i, G float *part_sum,
    G u32 *am_count, G u32 *ds_ctrl, G ink_r8_dh_u64 *rng_state,
    G u32 *draft_ids, G float *draft_probs, G float *draft_entropy,
    G u32 *q_n, G u32 *q_ids, G float *q_probs,
    u32 full_rows, u32 n_act, u32 nb, u32 ng, u32 K, u32 K_stride, u32 step_idx,
    float cap, float temp, float top_p, float pmin, float surv_a, float surv_b) {
  if (ds_ctrl[1u] != 0u) return;

  const u32 g = wgid();
  const u32 t = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 row_in_wg = (w << 2u) + (l >> 3u);
  if (g >= ng || ng == 0u || ng > INK_R8_DH_NGMAX || n_act == 0u || nb == 0u) return;

  /* Mode A: Greedy (temp <= 0.0f) -- 0 LDS in hot loop, 0 logits materialization! */
  if (temp <= 0.0f) {
    float bv = -__builtin_inff();
    u32 bi = 0xFFFFFFFFu;
    float se = 0.0f;

    for (u32 sbase = g << 5u; sbase < n_act; sbase += (ng << 5u)) {
      const u32 s = sbase + row_in_wg;
      if (s < n_act) {
        const u32 tok = id_map ? id_map[s] : s;
        if (tok < full_rows) {
          const float fv = ink_r8_dh_p_row_val(p_qw, p_qs, lat_q8, base_row, tok, s, nb, sub, cap);
          if (sub == 0u) {
            if (fv > bv) {
              se = se * __builtin_expf(bv - fv) + 1.0f;
              bv = fv; bi = tok;
            } else {
              if (fv == bv && tok < bi) bi = tok;
              if (fv > -__builtin_inff()) se += __builtin_expf(fv - bv);
            }
          }
        }
      }
    }

    const float my_bm = bv;
    ink_r8_dh_gr_best(&bv, &bi);
    float sm = (sub == 0u && se > 0.0f) ? (se * __builtin_expf(my_bm - bv)) : 0.0f;
    sm = wave_sum_all(sm);
    if (l == 0u) ink_r8_dh_dlds.u.gr.ws[w] = sm;
    lbarrier();
    float wg_se = 0.0f;
#pragma clang loop unroll(disable)
    for (u32 j = 0u; j < 8u; j++) wg_se += ink_r8_dh_dlds.u.gr.ws[j];

    if (t == 0u) { part_v[g] = bv; part_i[g] = bi; part_sum[g] = wg_se; }
    if (ng > 1u) {
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      barrier();
      if (t == 0u) {
        ink_r8_dh_dlds.last =
            (__scoped_atomic_fetch_add(am_count, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u) ? 1u : 0u;
      }
      barrier();
      if (!ink_r8_dh_dlds.last) return;
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

      bv = -__builtin_inff(); bi = 0xFFFFFFFFu;
      for (u32 u = t; u < ng; u += NT) {
        if (ink_r8_dh_better(part_v[u], part_i[u], bv, bi)) { bv = part_v[u]; bi = part_i[u]; }
      }
      ink_r8_dh_gr_best(&bv, &bi);
      float S = 0.0f;
      for (u32 u = t; u < ng; u += NT) {
        if (part_sum[u] > 0.0f) S += part_sum[u] * __builtin_expf(part_v[u] - bv);
      }
      S = wave_sum_all(S);
      if (l == 0u) ink_r8_dh_dlds.u.gr.ws[w] = S;
      lbarrier();
      wg_se = 0.0f;
#pragma clang loop unroll(disable)
      for (u32 j = 0u; j < 8u; j++) wg_se += ink_r8_dh_dlds.u.gr.ws[j];
    }

    if (t == 0u) {
      const float prob = (wg_se > 0.0f) ? (1.0f / wg_se) : 0.0f;
      draft_ids[step_idx] = bi;
      draft_probs[step_idx] = prob;
      if (draft_entropy) draft_entropy[step_idx] = 0.0f;
      if (q_n) q_n[step_idx] = 0u;
      ds_ctrl[0u] = bi;
      ds_ctrl[2u] = step_idx + 1u;
      if (prob < pmin) ds_ctrl[1u] = 1u;
      if (ng > 1u) *am_count = 0u;
    }
    return;
  }

  /* Mode B: Probabilistic Wave 6 (temp > 0.0f, 1 <= K <= 32, ng <= 64). */
  if (K == 0u || K > INK_R8_DH_KMAX) return;
  if (t < K) {
    ink_r8_dh_dlds.u.st1.kv[t] = -__builtin_inff();
    ink_r8_dh_dlds.u.st1.ki[t] = 0xFFFFFFFFu;
  }
  if (t == 0u) {
    ink_r8_dh_dlds.u.st1.min_v = -__builtin_inff();
    ink_r8_dh_dlds.u.st1.min_i = 0xFFFFFFFFu;
    ink_r8_dh_dlds.u.st1.min_pos = 0u;
    ink_r8_dh_dlds.u.st1.any_pass = 0u;
  }
  lbarrier();

  for (u32 sbase = g << 5u; sbase < n_act; sbase += (ng << 5u)) {
    const u32 s = sbase + row_in_wg;
    const u32 s_ok = (s < n_act) ? 1u : 0u;
    const u32 tok = s_ok ? (id_map ? id_map[s] : s) : 0xFFFFFFFFu;
    const u32 ok = (s_ok && tok < full_rows) ? 1u : 0u;
    float fv = -__builtin_inff();
    if (ok) {
      fv = ink_r8_dh_p_row_val(p_qw, p_qs, lat_q8, base_row, tok, s, nb, sub, cap);
    }
    if (sub == 0u) {
      ink_r8_dh_dlds.u.st1.ch_v[row_in_wg] = fv;
      ink_r8_dh_dlds.u.st1.ch_i[row_in_wg] = ok ? tok : 0xFFFFFFFFu;
      if (ok && ink_r8_dh_better(fv, tok, ink_r8_dh_dlds.u.st1.min_v, ink_r8_dh_dlds.u.st1.min_i)) {
        ink_r8_dh_dlds.u.st1.any_pass = 1u;
      }
    }
    lbarrier();

    if (t == 0u && ink_r8_dh_dlds.u.st1.any_pass != 0u) {
      float cur_mv = ink_r8_dh_dlds.u.st1.min_v;
      u32 cur_mi = ink_r8_dh_dlds.u.st1.min_i;
      u32 cur_mp = ink_r8_dh_dlds.u.st1.min_pos;
#pragma clang loop unroll(disable)
      for (u32 c = 0u; c < 32u; c++) {
        const u32 ci = ink_r8_dh_dlds.u.st1.ch_i[c];
        if (ci == 0xFFFFFFFFu) continue;
        const float cv = ink_r8_dh_dlds.u.st1.ch_v[c];
        if (ink_r8_dh_better(cv, ci, cur_mv, cur_mi)) {
          ink_r8_dh_dlds.u.st1.kv[cur_mp] = cv;
          ink_r8_dh_dlds.u.st1.ki[cur_mp] = ci;
          if (K == 1u) {
            cur_mv = cv; cur_mi = ci;
          } else {
            float wv_min = ink_r8_dh_dlds.u.st1.kv[0u];
            u32 wi_min = ink_r8_dh_dlds.u.st1.ki[0u];
            u32 wp_min = 0u;
#pragma clang loop unroll(disable)
            for (u32 p = 1u; p < K; p++) {
              const float pv = ink_r8_dh_dlds.u.st1.kv[p];
              const u32 pi = ink_r8_dh_dlds.u.st1.ki[p];
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
      ink_r8_dh_dlds.u.st1.min_v = cur_mv;
      ink_r8_dh_dlds.u.st1.min_i = cur_mi;
      ink_r8_dh_dlds.u.st1.min_pos = cur_mp;
      ink_r8_dh_dlds.u.st1.any_pass = 0u;
    }
    lbarrier();
  }

  /* Export this WG's K winners to part_v/part_i (or rank directly if ng == 1). */
  if (t < K) {
    part_v[(unsigned long)g * (unsigned long)K + t] = ink_r8_dh_dlds.u.st1.kv[t];
    part_i[(unsigned long)g * (unsigned long)K + t] = ink_r8_dh_dlds.u.st1.ki[t];
  }

  if (ng > 1u) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (t == 0u) {
      ink_r8_dh_dlds.last =
          (__scoped_atomic_fetch_add(am_count, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r8_dh_dlds.last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (t == 0u) *am_count = 0u;
  } else {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  }

  /* Stage 2: merge nc = ng * K <= 2048 into ink_r8_dh_dlds.fv / fi. */
  const u32 nc = ng * K;
  if (nc <= NT) {
    float mv = -__builtin_inff();
    u32 mi = 0xFFFFFFFFu;
    if (t < nc) {
      mv = part_v[t]; mi = part_i[t];
      ink_r8_dh_dlds.u.cand.cv[t] = mv;
      ink_r8_dh_dlds.u.cand.ci[t] = mi;
    }
    const u32 is_val = (t < nc && mi != 0xFFFFFFFFu) ? 1u : 0u;
    const u32 wcnt = (u32)__builtin_popcount(__builtin_amdgcn_ballot_w32(is_val != 0u));
    if (l == 0u) ink_r8_dh_dlds.wsum[0u][w] = wcnt;
    lbarrier();
    u32 nvalid = 0u;
#pragma clang loop unroll(disable)
    for (u32 j = 0u; j < 8u; j++) nvalid += ink_r8_dh_dlds.wsum[0u][j];
    if (is_val) {
      u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
      for (u32 u = 0u; u < nc; u++) {
        const u32 ui = ink_r8_dh_dlds.u.cand.ci[u];
        if (ui != 0xFFFFFFFFu) {
          const float b = ink_r8_dh_dlds.u.cand.cv[u];
          rank += (u32)((b > mv) | ((b == mv) & (ui < mi)));
        }
      }
      if (rank < K) { ink_r8_dh_dlds.fv[rank] = mv; ink_r8_dh_dlds.fi[rank] = mi; }
    }
    if (nvalid < K && t < K - nvalid) {
      ink_r8_dh_dlds.fv[nvalid + t] = -__builtin_inff();
      ink_r8_dh_dlds.fi[nvalid + t] = 0xFFFFFFFFu;
    }
  } else {
    float mv[8u]; /* nc <= 2048 -> <= 8 elems/thread! */
    u32 vm2 = 0u;
    float tm2 = -__builtin_inff();
#pragma unroll
    for (u32 k = 0u; k < 8u; k++) {
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
    const u32 tm2_k = vm2 ? ink_r8_dh_f2u(tm2) : 0u;
    const u32 tau2_k = ink_r8_dh_bitonic(tm2_k, K);
    const float tau2 = tau2_k ? ink_r8_dh_u2f(tau2_k) : -__builtin_inff();

    u32 gm2 = 0u, em2 = 0u;
    if (tau2_k == 0u) {
      gm2 = vm2;
    } else if (tm2_k == tau2_k) {
#pragma unroll
      for (u32 k = 0u; k < 8u; k++) if ((vm2 & (1u << k)) && mv[k] == tau2) em2 |= (1u << k);
    } else if (tm2_k > tau2_k) {
#pragma unroll
      for (u32 k = 0u; k < 8u; k++) {
        if (vm2 & (1u << k)) {
          if (mv[k] > tau2) gm2 |= (1u << k);
          else if (mv[k] == tau2) em2 |= (1u << k);
        }
      }
    }

    u32 ptot2 = 0u;
    const u32 poff2 = ink_r8_dh_wg_scan_u32(
        (((u32)__builtin_popcount(em2)) << 16u) | (u32)__builtin_popcount(gm2),
        ink_r8_dh_dlds.wsum[1u], &ptot2);
    const u32 bgt2 = poff2 & 0xFFFFu, beq2 = poff2 >> 16u;
    const u32 tgt2 = ptot2 & 0xFFFFu, teq2 = ptot2 >> 16u;

    if (gm2) {
      u32 pos = bgt2;
#pragma unroll
      for (u32 k = 0u; k < 8u; k++) {
        if (gm2 & (1u << k)) {
          ink_r8_dh_dlds.u.cand.cv[pos] = mv[k];
          ink_r8_dh_dlds.u.cand.ci[pos] = part_i[t + k * NT];
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
        for (u32 k = 0u; k < 8u; k++) {
          if (em2 & (1u << k)) {
            ink_r8_dh_dlds.u.cand.cv[pos] = mv[k];
            ink_r8_dh_dlds.u.cand.ci[pos] = part_i[t + k * NT];
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
      for (u32 k = 0u; k < 8u; k++) {
        if (taken2 < need2) {
          const u32 hit = (em2 >> k) & 1u;
          u32 stot = 0u;
          const u32 soff = ink_r8_dh_wg_scan_u32(hit, ink_r8_dh_dlds.wsum[k & 1u], &stot);
          if (hit && taken2 + soff < need2) {
            const u32 p = t + k * NT;
            ink_r8_dh_dlds.fv[tgt2 + taken2 + soff] = part_v[p];
            ink_r8_dh_dlds.fi[tgt2 + taken2 + soff] = part_i[p];
          }
          taken2 += stot;
        }
      }
    }
    for (u32 j = t; j < c2; j += NT) {
      const float a = ink_r8_dh_dlds.u.cand.cv[j];
      const u32 ai = ink_r8_dh_dlds.u.cand.ci[j];
      u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
      for (u32 u = 0u; u < c2; u++) {
        const float b = ink_r8_dh_dlds.u.cand.cv[u];
        rank += (u32)((b > a) | ((b == a) & (ink_r8_dh_dlds.u.cand.ci[u] < ai)));
      }
      if (rank < K) { ink_r8_dh_dlds.fv[rank] = a; ink_r8_dh_dlds.fi[rank] = ai; }
    }
    if (tau2_k == 0u && c2 < K && t < K - c2) {
      ink_r8_dh_dlds.fv[c2 + t] = -__builtin_inff();
      ink_r8_dh_dlds.fi[c2 + t] = 0xFFFFFFFFu;
    }
  }
  lbarrier();

  /* Stage 3 (Wave 0): exact Softmax(temp) + top_p nucleus + ForeSpark H(q) + xorshift64*. */
  if (t < 32u) {
    const u32 c = t;
    const u32 ci = (c < K) ? ink_r8_dh_dlds.fi[c] : 0xFFFFFFFFu;
    const float cv = (c < K) ? ink_r8_dh_dlds.fv[c] : -__builtin_inff();
    const float v0 = ink_r8_dh_dlds.fv[0u];
    const u32 valid = (c < K && ci != 0xFFFFFFFFu && cv == cv && v0 == v0) ? 1u : 0u;

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

    ink_r8_dh_dlds.fq[c] = qc;
    if (c < m && c < K_stride) {
      if (q_ids) q_ids[(unsigned long)step_idx * (unsigned long)K_stride + (unsigned long)c] = ci;
      if (q_probs) q_probs[(unsigned long)step_idx * (unsigned long)K_stride + (unsigned long)c] = qc;
    }

    if (c == 0u) {
      u32 chosen = 0u;
      u32 chosen_tok = (m > 0u) ? ink_r8_dh_dlds.fi[0u] : 0u;
      float chosen_p = (m > 0u) ? ink_r8_dh_dlds.fq[0u] : 0.0f;
      if (m > 1u && rng_state) {
        ink_r8_dh_u64 rs = *rng_state;
        if (rs == 0ull) rs = 0x9E3779B97F4A7C15ull ^ ((ink_r8_dh_u64)step_idx + 1ull);
        rs ^= rs >> 12u; rs ^= rs << 25u; rs ^= rs >> 27u;
        *rng_state = rs;
        const ink_r8_dh_u64 rnd = rs * 2685821657736338717ull;
        float u = (float)((u32)(rnd >> 40u)) * (1.0f / 16777216.0f);
        u32 last_nz = 0u;
#pragma clang loop unroll(disable)
        for (u32 j = 0u; j < m; j++) {
          const float pj = ink_r8_dh_dlds.fq[j];
          if (pj > 0.0f) last_nz = j;
          u -= pj;
          if (u <= 0.0f && pj > 0.0f) { chosen = j; break; }
          if (j + 1u == m) chosen = last_nz;
        }
        chosen_tok = ink_r8_dh_dlds.fi[chosen];
        chosen_p = ink_r8_dh_dlds.fq[chosen];
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
 * Kernel 2 (Bit 86): Adaptive 1-Route vs 2..4-Route Specialized Verify MoE
 * ========================================================================== */

KERNEL ie_inkling_moe_sort_adap_r8(
    const G u32 *route, u32 rst, u32 T, u32 K, u32 n_exp,
    G u32 *offs, G u32 *list,
    G u32 *packs1, G u32 *n_packs1,
    G u32 *packsM, G u32 *n_packsM) {
  static LDS u32 ink_r8_dh_scnt[INK_R8_DH_MAX_EXP];
  static LDS u32 ink_r8_dh_ser[INK_R8_DH_MAX_PAIRS];
  static LDS u32 ink_r8_dh_wsum[9u];
  static LDS u32 ink_r8_dh_p1sum[9u];
  static LDS u32 ink_r8_dh_pmsum[9u];

  const u32 t = tid(), w = t >> 5u, l = lane();
  const u32 np = T * K;
  if (n_exp == 0u || n_exp > INK_R8_DH_MAX_EXP || np > INK_R8_DH_MAX_PAIRS) return;
  if (t < n_exp) ink_r8_dh_scnt[t] = 0u;
  lbarrier();

  for (u32 p = t; p < np; p += NT) {
    const u32 tk = (K == 8u) ? (p >> 3u) : (p / K);
    const u32 kk = (K == 8u) ? (p & 7u) : (p - tk * K);
    const u32 e = ((const G u32 *)((const G u8 *)route + (unsigned long)tk * (unsigned long)rst))[kk];
    const u32 r = __atomic_fetch_add(&ink_r8_dh_scnt[e], 1u, __ATOMIC_RELAXED);
    ink_r8_dh_ser[p] = (e << 16u) | r;
  }
  lbarrier();

  const u32 c = (t < n_exp) ? ink_r8_dh_scnt[t] : 0u;
  /* Split packs into nj==1 (packs1) vs nj in 2..4 (packsM).
   * Note: c == 5 splits into 4 (packsM) + 1 (packs1)! */
  const u32 n_m = c >> 2u;          /* full 4-route packs */
  const u32 rem = c & 3u;
  const u32 p1_c = (rem == 1u) ? 1u : 0u;
  const u32 pm_c = n_m + ((rem >= 2u) ? 1u : 0u);

  const u32 c_inc  = ink_r8_dh_wave_scan_inc(c);
  const u32 p1_inc = ink_r8_dh_wave_scan_inc(p1_c);
  const u32 pm_inc = ink_r8_dh_wave_scan_inc(pm_c);
  if (l == 31u) {
    ink_r8_dh_wsum[w] = c_inc;
    ink_r8_dh_p1sum[w] = p1_inc;
    ink_r8_dh_pmsum[w] = pm_inc;
  }
  lbarrier();

  if (w == 0u) {
    const u32 ws = (l < 8u) ? ink_r8_dh_wsum[l] : 0u;
    const u32 p1s = (l < 8u) ? ink_r8_dh_p1sum[l] : 0u;
    const u32 pms = (l < 8u) ? ink_r8_dh_pmsum[l] : 0u;
    const u32 wi = ink_r8_dh_wave_scan_inc(ws);
    const u32 p1i = ink_r8_dh_wave_scan_inc(p1s);
    const u32 pmi = ink_r8_dh_wave_scan_inc(pms);
    if (l < 8u) {
      ink_r8_dh_wsum[l] = wi - ws;
      ink_r8_dh_p1sum[l] = p1i - p1s;
      ink_r8_dh_pmsum[l] = pmi - pms;
    }
    if (l == 7u) {
      ink_r8_dh_wsum[8u] = wi;
      ink_r8_dh_p1sum[8u] = p1i;
      ink_r8_dh_pmsum[8u] = pmi;
    }
  }
  lbarrier();

  if (t < n_exp) {
    const u32 off = ink_r8_dh_wsum[w] + (c_inc - c);
    const u32 p1off = ink_r8_dh_p1sum[w] + (p1_inc - p1_c);
    const u32 pmoff = ink_r8_dh_pmsum[w] + (pm_inc - pm_c);
    ink_r8_dh_scnt[t] = off;
    if (offs) offs[t] = off;

    if (c > 0u) {
      u32 lo_p = off;
      u32 mi = pmoff;
      for (u32 k = 0u; k < n_m; k++) {
        if (packsM) packsM[mi++] = (t & 0xFFu) | ((lo_p & 0xFFFu) << 8u) | (4u << 20u);
        lo_p += 4u;
      }
      if (rem >= 2u) {
        if (packsM) packsM[mi++] = (t & 0xFFu) | ((lo_p & 0xFFFu) << 8u) | ((rem & 0xFu) << 20u);
      } else if (rem == 1u) {
        if (packs1) packs1[p1off] = (t & 0xFFu) | ((lo_p & 0xFFFu) << 8u) | (1u << 20u);
      }
    }
  }
  if (t == 0u) {
    if (offs) offs[n_exp] = ink_r8_dh_wsum[8u];
    if (n_packs1) *n_packs1 = ink_r8_dh_p1sum[8u];
    if (n_packsM) *n_packsM = ink_r8_dh_pmsum[8u];
  }
  lbarrier();

  for (u32 p = t; p < np; p += NT) {
    const u32 er = ink_r8_dh_ser[p];
    list[ink_r8_dh_scnt[er >> 16u] + (er & 0xFFFFu)] = p;
  }
}

static inline __attribute__((always_inline)) float ink_r8_dh_iq3_stream_dot(
    const LDSP u32 *ulut, u32 d16, u32 aux, u32 q01, u32 q23,
    u32x4 x0, u32x4 x1, float dav) {
#pragma clang fp contract(off)
  const float dw = ie_rq_h2f(d16) * (0.5f + (float)(aux >> 28u)) * 0.5f;
  const u32 qp0 = q01 & 0xFFFFu, qp1 = q01 >> 16u;
  const u32 qp2 = q23 & 0xFFFFu, qp3 = q23 >> 16u;
  const u32 g_lo0 = ulut[qp0 & 255u], g_hi0 = ulut[qp0 >> 8u];
  const u32 g_lo1 = ulut[qp1 & 255u], g_hi1 = ulut[qp1 >> 8u];
  const u32 g_lo2 = ulut[qp2 & 255u], g_hi2 = ulut[qp2 >> 8u];
  const u32 g_lo3 = ulut[qp3 & 255u], g_hi3 = ulut[qp3 >> 8u];
  __builtin_amdgcn_sched_barrier(0);
  const u32 x00 = x0.x, x01 = x0.y, x02 = x0.z, x03 = x0.w;
  const u32 x10 = x1.x, x11 = x1.y, x12 = x1.z, x13 = x1.w;
  const u32 s8_0 = rq_ksign7(aux & 127u);
  u32 idot = ie_dot4_ss(0u, ie_rq_ssign(g_lo0, ie_rq_mask4(s8_0 & 15u)), x00);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_hi0, ie_rq_mask4(s8_0 >> 4u)), x01);
  const u32 s8_1 = rq_ksign7((aux >> 7u) & 127u);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_lo1, ie_rq_mask4(s8_1 & 15u)), x02);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_hi1, ie_rq_mask4(s8_1 >> 4u)), x03);
  const u32 s8_2 = rq_ksign7((aux >> 14u) & 127u);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_lo2, ie_rq_mask4(s8_2 & 15u)), x10);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_hi2, ie_rq_mask4(s8_2 >> 4u)), x11);
  const u32 s8_3 = rq_ksign7((aux >> 21u) & 127u);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_lo3, ie_rq_mask4(s8_3 & 15u)), x12);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_hi3, ie_rq_mask4(s8_3 >> 4u)), x13);
  return dav * (dw * (float)(int)idot);
}

static inline __attribute__((always_inline)) float ink_r8_dh_iq3_pre_dot(
    u32 wl0, u32 wh0, u32 wl1, u32 wh1, u32 wl2, u32 wh2, u32 wl3, u32 wh3,
    float dw, u32x4 x0, u32x4 x1, float dav) {
#pragma clang fp contract(off)
  const u32 x00 = x0.x, x01 = x0.y, x02 = x0.z, x03 = x0.w;
  const u32 x10 = x1.x, x11 = x1.y, x12 = x1.z, x13 = x1.w;
  u32 idot = ie_dot4_ss(0u, wl0, x00);
  idot = ie_dot4_ss(idot, wh0, x01);
  idot = ie_dot4_ss(idot, wl1, x02);
  idot = ie_dot4_ss(idot, wh1, x03);
  idot = ie_dot4_ss(idot, wl2, x10);
  idot = ie_dot4_ss(idot, wh2, x11);
  idot = ie_dot4_ss(idot, wl3, x12);
  idot = ie_dot4_ss(idot, wh3, x13);
  return dav * (dw * (float)(int)idot);
}

static inline __attribute__((always_inline)) int ink_r8_dh_iq4_dot4(u32x4 w4, u32x4 a0, u32x4 a1) {
  const u32 wx = w4.x, wy = w4.y, wz = w4.z, ww = w4.w;
  const u32 a00 = a0.x, a01 = a0.y, a02 = a0.z, a03 = a0.w;
  const u32 a10 = a1.x, a11 = a1.y, a12 = a1.z, a13 = a1.w;
  u32 d = 0u;
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(wx & 0x0F0F0F0Fu), a00), ie_iq4_map((wx >> 4u) & 0x0F0F0F0Fu), a10);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(wy & 0x0F0F0F0Fu), a01), ie_iq4_map((wy >> 4u) & 0x0F0F0F0Fu), a11);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(wz & 0x0F0F0F0Fu), a02), ie_iq4_map((wz >> 4u) & 0x0F0F0F0Fu), a12);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(ww & 0x0F0F0F0Fu), a03), ie_iq4_map((ww >> 4u) & 0x0F0F0F0Fu), a13);
  return (int)d;
}

static inline __attribute__((always_inline)) int ink_r8_dh_iq4_dot_pre(
    u32 wl0, u32 wh0, u32 wl1, u32 wh1, u32 wl2, u32 wh2, u32 wl3, u32 wh3,
    u32x4 a0, u32x4 a1) {
  const u32 a00 = a0.x, a01 = a0.y, a02 = a0.z, a03 = a0.w;
  const u32 a10 = a1.x, a11 = a1.y, a12 = a1.z, a13 = a1.w;
  u32 d = ie_dot4_ss(0u, wl0, a00);
  d = ie_dot4_ss(d, wh0, a10);
  d = ie_dot4_ss(d, wl1, a01);
  d = ie_dot4_ss(d, wh1, a11);
  d = ie_dot4_ss(d, wl2, a02);
  d = ie_dot4_ss(d, wh2, a12);
  d = ie_dot4_ss(d, wl3, a03);
  d = ie_dot4_ss(d, wh3, a13);
  return (int)d;
}

#define INK_R8_DH_ADD4(a0_, a1_, a2_, a3_, j_, v_) \
  do {                                             \
    if ((j_) == 0u) (a0_) += (v_);                 \
    else if ((j_) == 1u) (a1_) += (v_);            \
    else if ((j_) == 2u) (a2_) += (v_);            \
    else (a3_) += (v_);                            \
  } while (0)

#define INK_R8_DH_FMA4(a0_, a1_, a2_, a3_, j_, p_, d_)   \
  do {                                                   \
    if ((j_) == 0u) (a0_) = __builtin_fmaf((p_), (d_), (a0_)); \
    else if ((j_) == 1u) (a1_) = __builtin_fmaf((p_), (d_), (a1_)); \
    else if ((j_) == 2u) (a2_) = __builtin_fmaf((p_), (d_), (a2_)); \
    else (a3_) = __builtin_fmaf((p_), (d_), (a3_));      \
  } while (0)

/* Lean 1-route-only IQ3_XXS Gate/Up + SITU + Q8 (zero multi-pair register overhead -> <= 64 VGPR!). */
KERNEL ie_inkling_gexp_iq3xxs_situ_nj1_r8(
    const G u8 *qw, const G u8 *X, G float *b_out, G u8 *q_out,
    u32 eff, u32 nb, const G u32 *list, const G u32 *packs1, const G u32 *n_packs1,
    u32 q_bytes, u32 xs, u32 bstr, u32 qstr, u32 xdiv, float beta, float lbeta) {
#pragma clang fp contract(off)
  static LDS u32   ink_r8_dh_iq3_lut1[256u] __attribute__((aligned(16)));
  static LDS float ink_r8_dh_situ1[32u] __attribute__((aligned(16)));

  const u32 by = tokid();
  const u32 npk = (u32)__builtin_amdgcn_readfirstlane((int)(*n_packs1));
  const u32 blk = wgid();
  const u32 nb_out = eff >> 5u;
  if (by >= npk || blk >= nb_out) return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)packs1[by]);
  const u32 nj = (desc >> 20u) & 0xFu;
  if (nj != 1u) return;

  const u32 e = desc & 0xFFu;
  const u32 lo = (desc >> 8u) & 0xFFFu;
  const u32 t0 = tid();
  ink_r8_dh_iq3_lut1[t0] = ie_rq_iq3xxs_grid[t0];
  lbarrier();

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 r_in_blk = (wv << 2u) + k;
  const u32 r = (blk << 5u) + r_in_blk;
  const u32 mb = nb >> 3u;
  const u32 row_bytes = mb * INK_R8_DH_SB_BYTES;
  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * (unsigned long)q_bytes);
  const G u8 *bpg = ebase + (unsigned long)r * (unsigned long)row_bytes;
  const G u8 *bpu = ebase + (unsigned long)(eff + r) * (unsigned long)row_bytes;
  const u32 qo = 2u + (s << 3u), auxo = 66u + (s << 2u);
  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
  const G u8 *xp0 = rq_ptr_u(X + (unsigned long)ink_r8_dh_div_x(p0, xdiv) * (unsigned long)xs);
  const G float *dap0 = (const G float *)(xp0 + 32ul * (unsigned long)nb);

  float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
  float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;

  u32 dg_c = ie_rq_a16(bpg, 0u), auxg_c = ie_rq_a32(bpg, auxo);
  u32 q01g_c = ie_rq_a32(bpg, qo), q23g_c = ie_rq_a32(bpg, qo + 4u);
  u32x4 x0_c = ink_r8_dh_ld4(xp0 + (s << 5u));
  u32x4 x1_c = ink_r8_dh_ld4(xp0 + (s << 5u) + 16u);
  float dav_c = dap0[s];

#pragma unroll 1
  for (u32 sb = 0u; sb < mb; sb++) {
    const u32 j_acc = sb & 3u;
    const u32 du_c = ie_rq_a16(bpu, 0u), auxu_c = ie_rq_a32(bpu, auxo);
    const u32 q01u_c = ie_rq_a32(bpu, qo), q23u_c = ie_rq_a32(bpu, qo + 4u);
    __builtin_amdgcn_sched_barrier(0);

    const float vg = ink_r8_dh_iq3_stream_dot(ink_r8_dh_iq3_lut1, dg_c, auxg_c, q01g_c, q23g_c, x0_c, x1_c, dav_c);
    if (j_acc == 0u) ag0 += vg; else if (j_acc == 1u) ag1 += vg; else if (j_acc == 2u) ag2 += vg; else ag3 += vg;
    __builtin_amdgcn_sched_barrier(0);

    bpg += INK_R8_DH_SB_BYTES; bpu += INK_R8_DH_SB_BYTES;
    u32 dg_n = 0u, auxg_n = 0u, q01g_n = 0u, q23g_n = 0u;
    u32x4 x0_n = (u32x4)(0u), x1_n = (u32x4)(0u);
    float dav_n = 0.0f;
    if (sb + 1u < mb) {
      const u32 bn = ((sb + 1u) << 3u) + s;
      dg_n = ie_rq_a16(bpg, 0u); auxg_n = ie_rq_a32(bpg, auxo);
      q01g_n = ie_rq_a32(bpg, qo); q23g_n = ie_rq_a32(bpg, qo + 4u);
      x0_n = ink_r8_dh_ld4(xp0 + (bn << 5u));
      x1_n = ink_r8_dh_ld4(xp0 + (bn << 5u) + 16u);
      dav_n = dap0[bn];
    }
    __builtin_amdgcn_sched_barrier(0);

    const float vu = ink_r8_dh_iq3_stream_dot(ink_r8_dh_iq3_lut1, du_c, auxu_c, q01u_c, q23u_c, x0_c, x1_c, dav_c);
    if (j_acc == 0u) au0 += vu; else if (j_acc == 1u) au1 += vu; else if (j_acc == 2u) au2 += vu; else au3 += vu;
    __builtin_amdgcn_sched_barrier(0);

    dg_c = dg_n; auxg_c = auxg_n; q01g_c = q01g_n; q23g_c = q23g_n;
    x0_c = x0_n; x1_c = x1_n; dav_c = dav_n;
  }

  const float gv = ink_r8_dh_oct_tree_f((ag0 + ag2) + (ag1 + ag3));
  const float uv = ink_r8_dh_oct_tree_f((au0 + au2) + (au1 + au3));
  if (s == 0u) {
    const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
    const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
    ink_r8_dh_situ1[r_in_blk] = gs * sigmoidf(gv) * us;
  }
  lbarrier();
  if (wv == 0u) {
    const float v = ink_r8_dh_situ1[l];
    if (b_out) ((G float *)((G u8 *)b_out + (unsigned long)p0 * (unsigned long)bstr))[(blk << 5u) + l] = v;
    if (q_out) quant_wave(v, blk, q_out + (unsigned long)p0 * (unsigned long)qstr, nb_out);
  }
}

/* Dedicated 2..4-route IQ3_XXS Gate/Up + SITU + Q8 (zero nj==1 branch overhead). */
KERNEL ie_inkling_gexp_iq3xxs_situ_njm_r8(
    const G u8 *qw, const G u8 *X, G float *b_out, G u8 *q_out,
    u32 eff, u32 nb, const G u32 *list, const G u32 *packsM, const G u32 *n_packsM,
    u32 q_bytes, u32 xs, u32 bstr, u32 qstr, u32 xdiv, float beta, float lbeta) {
#pragma clang fp contract(off)
  static LDS u32   ink_r8_dh_iq3_lutm[256u] __attribute__((aligned(16)));
  static LDS float ink_r8_dh_situm[4u][32u] __attribute__((aligned(16)));

  const u32 by = tokid();
  const u32 npk = (u32)__builtin_amdgcn_readfirstlane((int)(*n_packsM));
  const u32 blk = wgid();
  const u32 nb_out = eff >> 5u;
  if (by >= npk || blk >= nb_out) return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)packsM[by]);
  const u32 nj = (desc >> 20u) & 0xFu;
  if (nj < 2u || nj > 4u) return;

  const u32 e = desc & 0xFFu;
  const u32 lo = (desc >> 8u) & 0xFFFu;
  const u32 t0 = tid();
  ink_r8_dh_iq3_lutm[t0] = ie_rq_iq3xxs_grid[t0];
  lbarrier();

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 r_in_blk = (wv << 2u) + k;
  const u32 r = (blk << 5u) + r_in_blk;
  const u32 mb = nb >> 3u;
  const u32 row_bytes = mb * INK_R8_DH_SB_BYTES;
  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * (unsigned long)q_bytes);
  const G u8 *bpg = ebase + (unsigned long)r * (unsigned long)row_bytes;
  const G u8 *bpu = ebase + (unsigned long)(eff + r) * (unsigned long)row_bytes;
  const u32 qo = 2u + (s << 3u), auxo = 66u + (s << 2u);
  const unsigned long x_scale_off = 32ul * (unsigned long)nb;

  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
  const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 1u]);
  const u32 p2 = (nj > 2u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 2u]) : p0;
  const u32 p3 = (nj > 3u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 3u]) : p0;
  const G u8 *xp0 = rq_ptr_u(X + (unsigned long)ink_r8_dh_div_x(p0, xdiv) * (unsigned long)xs);
  const G u8 *xp1 = rq_ptr_u(X + (unsigned long)ink_r8_dh_div_x(p1, xdiv) * (unsigned long)xs);
  const G u8 *xp2 = (nj > 2u) ? rq_ptr_u(X + (unsigned long)ink_r8_dh_div_x(p2, xdiv) * (unsigned long)xs) : xp0;
  const G u8 *xp3 = (nj > 3u) ? rq_ptr_u(X + (unsigned long)ink_r8_dh_div_x(p3, xdiv) * (unsigned long)xs) : xp0;
  const G float *dap0 = (const G float *)(xp0 + x_scale_off);
  const G float *dap1 = (const G float *)(xp1 + x_scale_off);
  const G float *dap2 = (const G float *)(xp2 + x_scale_off);
  const G float *dap3 = (const G float *)(xp3 + x_scale_off);

  float g0_0 = 0.0f, g0_1 = 0.0f, g0_2 = 0.0f, g0_3 = 0.0f;
  float g1_0 = 0.0f, g1_1 = 0.0f, g1_2 = 0.0f, g1_3 = 0.0f;
  float g2_0 = 0.0f, g2_1 = 0.0f, g2_2 = 0.0f, g2_3 = 0.0f;
  float g3_0 = 0.0f, g3_1 = 0.0f, g3_2 = 0.0f, g3_3 = 0.0f;

  /* Pass 1: Gate row (decoded ONCE across 2..4 pairs). */
#pragma unroll 1
  for (u32 sb = 0u; sb < mb; sb++) {
    const u32 bn = (sb << 3u) + s;
    const u32 d16 = ie_rq_a16(bpg, 0u);
    const u32 aux = ie_rq_a32(bpg, auxo);
    const u32 q01 = ie_rq_a32(bpg, qo), q23 = ie_rq_a32(bpg, qo + 4u);
    bpg += INK_R8_DH_SB_BYTES;
    const float dw = ie_rq_h2f(d16) * (0.5f + (float)(aux >> 28u)) * 0.5f;
    const u32 qp0 = q01 & 0xFFFFu, qp1 = q01 >> 16u, qp2 = q23 & 0xFFFFu, qp3 = q23 >> 16u;
    const u32 s8_0 = rq_ksign7(aux & 127u), s8_1 = rq_ksign7((aux >> 7u) & 127u);
    const u32 s8_2 = rq_ksign7((aux >> 14u) & 127u), s8_3 = rq_ksign7((aux >> 21u) & 127u);
    const u32 wl0 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp0 & 255u], ie_rq_mask4(s8_0 & 15u));
    const u32 wh0 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp0 >> 8u],   ie_rq_mask4(s8_0 >> 4u));
    const u32 wl1 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp1 & 255u], ie_rq_mask4(s8_1 & 15u));
    const u32 wh1 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp1 >> 8u],   ie_rq_mask4(s8_1 >> 4u));
    const u32 wl2 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp2 & 255u], ie_rq_mask4(s8_2 & 15u));
    const u32 wh2 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp2 >> 8u],   ie_rq_mask4(s8_2 >> 4u));
    const u32 wl3 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp3 & 255u], ie_rq_mask4(s8_3 & 15u));
    const u32 wh3 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp3 >> 8u],   ie_rq_mask4(s8_3 >> 4u));
    __builtin_amdgcn_sched_barrier(0);

    const u32 j_acc = sb & 3u;
    const u32 bo = bn << 5u;
    INK_R8_DH_ADD4(g0_0, g0_1, g0_2, g0_3, j_acc,
                   ink_r8_dh_iq3_pre_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                         ink_r8_dh_ld4(xp0 + bo), ink_r8_dh_ld4(xp0 + bo + 16u), dap0[bn]));
    INK_R8_DH_ADD4(g1_0, g1_1, g1_2, g1_3, j_acc,
                   ink_r8_dh_iq3_pre_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                         ink_r8_dh_ld4(xp1 + bo), ink_r8_dh_ld4(xp1 + bo + 16u), dap1[bn]));
    if (nj > 2u) {
      INK_R8_DH_ADD4(g2_0, g2_1, g2_2, g2_3, j_acc,
                     ink_r8_dh_iq3_pre_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                           ink_r8_dh_ld4(xp2 + bo), ink_r8_dh_ld4(xp2 + bo + 16u), dap2[bn]));
      if (nj > 3u) {
        INK_R8_DH_ADD4(g3_0, g3_1, g3_2, g3_3, j_acc,
                       ink_r8_dh_iq3_pre_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                             ink_r8_dh_ld4(xp3 + bo), ink_r8_dh_ld4(xp3 + bo + 16u), dap3[bn]));
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  const float gv0 = ink_r8_dh_oct_tree_f((g0_0 + g0_2) + (g0_1 + g0_3));
  const float gv1 = ink_r8_dh_oct_tree_f((g1_0 + g1_2) + (g1_1 + g1_3));
  const float gv2 = (nj > 2u) ? ink_r8_dh_oct_tree_f((g2_0 + g2_2) + (g2_1 + g2_3)) : 0.0f;
  const float gv3 = (nj > 3u) ? ink_r8_dh_oct_tree_f((g3_0 + g3_2) + (g3_1 + g3_3)) : 0.0f;

  g0_0 = g0_1 = g0_2 = g0_3 = 0.0f;
  g1_0 = g1_1 = g1_2 = g1_3 = 0.0f;
  g2_0 = g2_1 = g2_2 = g2_3 = 0.0f;
  g3_0 = g3_1 = g3_2 = g3_3 = 0.0f;

  /* Pass 2: Up row (decoded ONCE across 2..4 pairs, reuses accumulator registers!). */
#pragma unroll 1
  for (u32 sb = 0u; sb < mb; sb++) {
    const u32 bn = (sb << 3u) + s;
    const u32 d16 = ie_rq_a16(bpu, 0u);
    const u32 aux = ie_rq_a32(bpu, auxo);
    const u32 q01 = ie_rq_a32(bpu, qo), q23 = ie_rq_a32(bpu, qo + 4u);
    bpu += INK_R8_DH_SB_BYTES;
    const float dw = ie_rq_h2f(d16) * (0.5f + (float)(aux >> 28u)) * 0.5f;
    const u32 qp0 = q01 & 0xFFFFu, qp1 = q01 >> 16u, qp2 = q23 & 0xFFFFu, qp3 = q23 >> 16u;
    const u32 s8_0 = rq_ksign7(aux & 127u), s8_1 = rq_ksign7((aux >> 7u) & 127u);
    const u32 s8_2 = rq_ksign7((aux >> 14u) & 127u), s8_3 = rq_ksign7((aux >> 21u) & 127u);
    const u32 wl0 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp0 & 255u], ie_rq_mask4(s8_0 & 15u));
    const u32 wh0 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp0 >> 8u],   ie_rq_mask4(s8_0 >> 4u));
    const u32 wl1 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp1 & 255u], ie_rq_mask4(s8_1 & 15u));
    const u32 wh1 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp1 >> 8u],   ie_rq_mask4(s8_1 >> 4u));
    const u32 wl2 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp2 & 255u], ie_rq_mask4(s8_2 & 15u));
    const u32 wh2 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp2 >> 8u],   ie_rq_mask4(s8_2 >> 4u));
    const u32 wl3 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp3 & 255u], ie_rq_mask4(s8_3 & 15u));
    const u32 wh3 = ie_rq_ssign(ink_r8_dh_iq3_lutm[qp3 >> 8u],   ie_rq_mask4(s8_3 >> 4u));
    __builtin_amdgcn_sched_barrier(0);

    const u32 j_acc = sb & 3u;
    const u32 bo = bn << 5u;
    INK_R8_DH_ADD4(g0_0, g0_1, g0_2, g0_3, j_acc,
                   ink_r8_dh_iq3_pre_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                         ink_r8_dh_ld4(xp0 + bo), ink_r8_dh_ld4(xp0 + bo + 16u), dap0[bn]));
    INK_R8_DH_ADD4(g1_0, g1_1, g1_2, g1_3, j_acc,
                   ink_r8_dh_iq3_pre_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                         ink_r8_dh_ld4(xp1 + bo), ink_r8_dh_ld4(xp1 + bo + 16u), dap1[bn]));
    if (nj > 2u) {
      INK_R8_DH_ADD4(g2_0, g2_1, g2_2, g2_3, j_acc,
                     ink_r8_dh_iq3_pre_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                           ink_r8_dh_ld4(xp2 + bo), ink_r8_dh_ld4(xp2 + bo + 16u), dap2[bn]));
      if (nj > 3u) {
        INK_R8_DH_ADD4(g3_0, g3_1, g3_2, g3_3, j_acc,
                       ink_r8_dh_iq3_pre_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                             ink_r8_dh_ld4(xp3 + bo), ink_r8_dh_ld4(xp3 + bo + 16u), dap3[bn]));
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  const float uv0 = ink_r8_dh_oct_tree_f((g0_0 + g0_2) + (g0_1 + g0_3));
  const float uv1 = ink_r8_dh_oct_tree_f((g1_0 + g1_2) + (g1_1 + g1_3));
  const float uv2 = (nj > 2u) ? ink_r8_dh_oct_tree_f((g2_0 + g2_2) + (g2_1 + g2_3)) : 0.0f;
  const float uv3 = (nj > 3u) ? ink_r8_dh_oct_tree_f((g3_0 + g3_2) + (g3_1 + g3_3)) : 0.0f;

  if (s == 0u) {
    float gs, us;
    gs = beta > 0.0f ? beta * ie_tanhf_s(gv0 / beta) : gv0;
    us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv0 / lbeta) : uv0;
    ink_r8_dh_situm[0u][r_in_blk] = gs * sigmoidf(gv0) * us;
    gs = beta > 0.0f ? beta * ie_tanhf_s(gv1 / beta) : gv1;
    us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv1 / lbeta) : uv1;
    ink_r8_dh_situm[1u][r_in_blk] = gs * sigmoidf(gv1) * us;
    if (nj > 2u) {
      gs = beta > 0.0f ? beta * ie_tanhf_s(gv2 / beta) : gv2;
      us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv2 / lbeta) : uv2;
      ink_r8_dh_situm[2u][r_in_blk] = gs * sigmoidf(gv2) * us;
    }
    if (nj > 3u) {
      gs = beta > 0.0f ? beta * ie_tanhf_s(gv3 / beta) : gv3;
      us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv3 / lbeta) : uv3;
      ink_r8_dh_situm[3u][r_in_blk] = gs * sigmoidf(gv3) * us;
    }
  }
  lbarrier();

  if (wv < nj) {
    const u32 pw = (wv == 0u) ? p0 : (wv == 1u ? p1 : (wv == 2u ? p2 : p3));
    const float v = ink_r8_dh_situm[wv][l];
    if (b_out) ((G float *)((G u8 *)b_out + (unsigned long)pw * (unsigned long)bstr))[(blk << 5u) + l] = v;
    if (q_out) quant_wave(v, blk, q_out + (unsigned long)pw * (unsigned long)qstr, nb_out);
  }
}

/* Lean 1-route-only IQ4_NL verify (zero multi-pair overhead -> <= 52 VGPR!). */
KERNEL ie_inkling_gexp_iq4nl_nj1_r8(
    const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
    u32 rows, u32 nb, const G u32 *list, const G u32 *packs1, const G u32 *n_packs1,
    u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv, u32 kind, const G u8 *res, u32 rs) {
#pragma clang fp contract(off)
  (void)kind;
  const u32 by = tokid();
  const u32 npk = (u32)__builtin_amdgcn_readfirstlane((int)(*n_packs1));
  const u32 rbase = wgid() << 5u;
  if (by >= npk || rbase >= rows) return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)packs1[by]);
  if (((desc >> 20u) & 0xFu) != 1u) return;

  const u32 e = desc & 0xFFu;
  const u32 lo = (desc >> 8u) & 0xFFFu;
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
  const u32 r = rbase + (wv << 2u) + sub;
  const u32 rr = (r < rows) ? r : (rows - 1u);

  const G u8 *rw = (const G u8 *)(qw0 + (unsigned long)e * (unsigned long)qwn + (unsigned long)rr * (unsigned long)nb * 4ul);
  const G f16 *rs_p = (const G f16 *)(qs0 + (unsigned long)e * (unsigned long)qsn) + (unsigned long)rr * (unsigned long)nb;
  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
  const G u8 *xp0 = rq_ptr_u(X + (unsigned long)ink_r8_dh_div_x(p0, xdiv) * (unsigned long)xs);
  const G u32 *aw0 = (const G u32 *)xp0;
  const G float *da0 = (const G float *)(xp0 + 32ul * (unsigned long)nb);

  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  const u32 n_steps = (nb + 7u) >> 3u;
  u32 b_c = sl;
  const int ok0 = (b_c < nb);
  u32x4 w_c = ok0 ? ink_r8_dh_ld4(rw + (unsigned long)b_c * 16ul) : (u32x4)(0u);
  float sc_c = ok0 ? (float)rs_p[b_c] : 0.0f;
  u32x4 x0_c = ok0 ? ink_r8_dh_ld4(aw0 + (unsigned long)b_c * 8ul) : (u32x4)(0u);
  u32x4 x1_c = ok0 ? ink_r8_dh_ld4(aw0 + (unsigned long)b_c * 8ul + 4ul) : (u32x4)(0u);
  float d_c = ok0 ? da0[b_c] : 0.0f;

#pragma unroll 1
  for (u32 st = 0u; st < n_steps; st++) {
    const u32 b_n = b_c + 8u;
    const int ok_n = (b_n < nb);
    const u32x4 w_n = ok_n ? ink_r8_dh_ld4(rw + (unsigned long)b_n * 16ul) : (u32x4)(0u);
    const float sc_n = ok_n ? (float)rs_p[b_n] : 0.0f;
    const u32x4 x0_n = ok_n ? ink_r8_dh_ld4(aw0 + (unsigned long)b_n * 8ul) : (u32x4)(0u);
    const u32x4 x1_n = ok_n ? ink_r8_dh_ld4(aw0 + (unsigned long)b_n * 8ul + 4ul) : (u32x4)(0u);
    const float d_n = ok_n ? da0[b_n] : 0.0f;
    __builtin_amdgcn_sched_barrier(0);

    if (b_c < nb) {
      const float prod = sc_c * (float)ink_r8_dh_iq4_dot4(w_c, x0_c, x1_c);
      const u32 j_acc = st & 3u;
      if (j_acc == 0u) a0 = __builtin_fmaf(prod, d_c, a0);
      else if (j_acc == 1u) a1 = __builtin_fmaf(prod, d_c, a1);
      else if (j_acc == 2u) a2 = __builtin_fmaf(prod, d_c, a2);
      else a3 = __builtin_fmaf(prod, d_c, a3);
    }
    __builtin_amdgcn_sched_barrier(0);
    b_c = b_n; w_c = w_n; sc_c = sc_n; x0_c = x0_n; x1_c = x1_n; d_c = d_n;
  }

  const float acc0 = ink_r8_dh_oct_tree_f((a0 + a2) + (a1 + a3));
  if (sl == 0u && r < rows) {
    G float *yp0 = (G float *)(Y + (unsigned long)p0 * (unsigned long)ys);
    const G float *rp0 = res ? (const G float *)(res + (unsigned long)p0 * (unsigned long)rs) : (const G float *)0;
    yp0[r] = rp0 ? (rp0[r] + acc0) : acc0;
  }
}

/* Dedicated 2..4-route IQ4_NL verify (shared decoded nibbles + double-buffered weights). */
KERNEL ie_inkling_gexp_iq4nl_njm_r8(
    const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
    u32 rows, u32 nb, const G u32 *list, const G u32 *packsM, const G u32 *n_packsM,
    u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv, u32 kind, const G u8 *res, u32 rs) {
#pragma clang fp contract(off)
  (void)kind;
  const u32 by = tokid();
  const u32 npk = (u32)__builtin_amdgcn_readfirstlane((int)(*n_packsM));
  const u32 rbase = wgid() << 5u;
  if (by >= npk || rbase >= rows) return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)packsM[by]);
  const u32 nj = (desc >> 20u) & 0xFu;
  if (nj < 2u || nj > 4u) return;

  const u32 e = desc & 0xFFu;
  const u32 lo = (desc >> 8u) & 0xFFFu;
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
  const u32 r = rbase + (wv << 2u) + sub;
  const u32 rr = (r < rows) ? r : (rows - 1u);

  const G u8 *rw = (const G u8 *)(qw0 + (unsigned long)e * (unsigned long)qwn + (unsigned long)rr * (unsigned long)nb * 4ul);
  const G f16 *rs_p = (const G f16 *)(qs0 + (unsigned long)e * (unsigned long)qsn) + (unsigned long)rr * (unsigned long)nb;
  const unsigned long x_scale_off = 32ul * (unsigned long)nb;

  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
  const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 1u]);
  const u32 p2 = (nj > 2u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 2u]) : p0;
  const u32 p3 = (nj > 3u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 3u]) : p0;
  const G u8 *xp0 = rq_ptr_u(X + (unsigned long)ink_r8_dh_div_x(p0, xdiv) * (unsigned long)xs);
  const G u8 *xp1 = rq_ptr_u(X + (unsigned long)ink_r8_dh_div_x(p1, xdiv) * (unsigned long)xs);
  const G u8 *xp2 = (nj > 2u) ? rq_ptr_u(X + (unsigned long)ink_r8_dh_div_x(p2, xdiv) * (unsigned long)xs) : xp0;
  const G u8 *xp3 = (nj > 3u) ? rq_ptr_u(X + (unsigned long)ink_r8_dh_div_x(p3, xdiv) * (unsigned long)xs) : xp0;
  const G u32 *aw0 = (const G u32 *)xp0, *aw1 = (const G u32 *)xp1;
  const G u32 *aw2 = (const G u32 *)xp2, *aw3 = (const G u32 *)xp3;
  const G float *da0 = (const G float *)(xp0 + x_scale_off);
  const G float *da1 = (const G float *)(xp1 + x_scale_off);
  const G float *da2 = (const G float *)(xp2 + x_scale_off);
  const G float *da3 = (const G float *)(xp3 + x_scale_off);

  float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
  float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
  float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
  float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;

  const u32 n_steps = (nb + 7u) >> 3u;
  u32 b_c = sl;
  const int ok0 = (b_c < nb);
  u32x4 w_c = ok0 ? ink_r8_dh_ld4(rw + (unsigned long)b_c * 16ul) : (u32x4)(0u);
  float sc_c = ok0 ? (float)rs_p[b_c] : 0.0f;

#pragma unroll 1
  for (u32 st = 0u; st < n_steps; st++) {
    const u32 b_n = b_c + 8u;
    const int ok_n = (b_n < nb);
    const u32x4 w_n = ok_n ? ink_r8_dh_ld4(rw + (unsigned long)b_n * 16ul) : (u32x4)(0u);
    const float sc_n = ok_n ? (float)rs_p[b_n] : 0.0f;
    __builtin_amdgcn_sched_barrier(0);

    if (b_c < nb) {
      const u32 wx = w_c.x, wy = w_c.y, wz = w_c.z, ww = w_c.w;
      const u32 wl0 = ie_iq4_map(wx & 0x0F0F0F0Fu), wh0 = ie_iq4_map((wx >> 4u) & 0x0F0F0F0Fu);
      const u32 wl1 = ie_iq4_map(wy & 0x0F0F0F0Fu), wh1 = ie_iq4_map((wy >> 4u) & 0x0F0F0F0Fu);
      const u32 wl2 = ie_iq4_map(wz & 0x0F0F0F0Fu), wh2 = ie_iq4_map((wz >> 4u) & 0x0F0F0F0Fu);
      const u32 wl3 = ie_iq4_map(ww & 0x0F0F0F0Fu), wh3 = ie_iq4_map((ww >> 4u) & 0x0F0F0F0Fu);
      const u32 j_acc = st & 3u;

      const float prod0 = sc_c * (float)ink_r8_dh_iq4_dot_pre(
          wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3,
          ink_r8_dh_ld4(aw0 + (unsigned long)b_c * 8ul), ink_r8_dh_ld4(aw0 + (unsigned long)b_c * 8ul + 4ul));
      INK_R8_DH_FMA4(a0_0, a0_1, a0_2, a0_3, j_acc, prod0, da0[b_c]);

      const float prod1 = sc_c * (float)ink_r8_dh_iq4_dot_pre(
          wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3,
          ink_r8_dh_ld4(aw1 + (unsigned long)b_c * 8ul), ink_r8_dh_ld4(aw1 + (unsigned long)b_c * 8ul + 4ul));
      INK_R8_DH_FMA4(a1_0, a1_1, a1_2, a1_3, j_acc, prod1, da1[b_c]);

      if (nj > 2u) {
        const float prod2 = sc_c * (float)ink_r8_dh_iq4_dot_pre(
            wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3,
            ink_r8_dh_ld4(aw2 + (unsigned long)b_c * 8ul), ink_r8_dh_ld4(aw2 + (unsigned long)b_c * 8ul + 4ul));
        INK_R8_DH_FMA4(a2_0, a2_1, a2_2, a2_3, j_acc, prod2, da2[b_c]);
        if (nj > 3u) {
          const float prod3 = sc_c * (float)ink_r8_dh_iq4_dot_pre(
              wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3,
              ink_r8_dh_ld4(aw3 + (unsigned long)b_c * 8ul), ink_r8_dh_ld4(aw3 + (unsigned long)b_c * 8ul + 4ul));
          INK_R8_DH_FMA4(a3_0, a3_1, a3_2, a3_3, j_acc, prod3, da3[b_c]);
        }
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    b_c = b_n; w_c = w_n; sc_c = sc_n;
  }

  const float acc0 = ink_r8_dh_oct_tree_f((a0_0 + a0_2) + (a0_1 + a0_3));
  const float acc1 = ink_r8_dh_oct_tree_f((a1_0 + a1_2) + (a1_1 + a1_3));
  const float acc2 = (nj > 2u) ? ink_r8_dh_oct_tree_f((a2_0 + a2_2) + (a2_1 + a2_3)) : 0.0f;
  const float acc3 = (nj > 3u) ? ink_r8_dh_oct_tree_f((a3_0 + a3_2) + (a3_1 + a3_3)) : 0.0f;

  if (sl == 0u && r < rows) {
    G float *yp0 = (G float *)(Y + (unsigned long)p0 * (unsigned long)ys);
    const G float *rp0 = res ? (const G float *)(res + (unsigned long)p0 * (unsigned long)rs) : (const G float *)0;
    yp0[r] = rp0 ? (rp0[r] + acc0) : acc0;
    G float *yp1 = (G float *)(Y + (unsigned long)p1 * (unsigned long)ys);
    const G float *rp1 = res ? (const G float *)(res + (unsigned long)p1 * (unsigned long)rs) : (const G float *)0;
    yp1[r] = rp1 ? (rp1[r] + acc1) : acc1;
    if (nj > 2u) {
      G float *yp2 = (G float *)(Y + (unsigned long)p2 * (unsigned long)ys);
      const G float *rp2 = res ? (const G float *)(res + (unsigned long)p2 * (unsigned long)rs) : (const G float *)0;
      yp2[r] = rp2 ? (rp2[r] + acc2) : acc2;
    }
    if (nj > 3u) {
      G float *yp3 = (G float *)(Y + (unsigned long)p3 * (unsigned long)ys);
      const G float *rp3 = res ? (const G float *)(res + (unsigned long)p3 * (unsigned long)rs) : (const G float *)0;
      yp3[r] = rp3 ? (rp3[r] + acc3) : acc3;
    }
  }
}

/* ============================================================================
 * Kernel 3 (Bit 89): ie_inkling_rel_attn_ver_tree_r8
 * Split-KV GQA-3 f16-KV Relative Attention for T <= 8 Speculative Tree Verify.
 * ========================================================================== */

static inline __attribute__((always_inline)) float ink_r8_dh_hw_sum(float v) {
  v += I2F(shl_i(F2I(v), 8));
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return I2F(__builtin_amdgcn_readlane(F2I(v), 0));
}

static inline __attribute__((always_inline)) float ink_r8_dh_dot4(f32x4 a, f32x4 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

KERNEL ie_inkling_rel_attn_ver_tree_r8(
    const G float *q, const G kvt *kc, const G kvt *vc,
    const G float *R, const G float *rel_bias, const G float *sink,
    G float *part, G float *out,
    u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
    u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
    u32 qs, u32 rs, u32 ss, u32 os, G u32 *count,
    ink_r8_dh_u64 tree_mask, u32 pos_off_packed) {
  static LDS float ink_r8_dh_tm[INK_R8_DH_GQA][8u];
  static LDS float ink_r8_dh_tl[INK_R8_DH_GQA][8u];
  static LDS __attribute__((aligned(16))) float ink_r8_dh_to[INK_R8_DH_GQA][8u][INK_R8_DH_HD];
  static LDS u32 ink_r8_dh_tlast;

  const u32 t = tokid();
  if (t >= T || T == 0u || T > 8u || hd != INK_R8_DH_HD || d_rel != INK_R8_DH_DREL ||
      n_kv == 0u || n_head != INK_R8_DH_GQA * n_kv || kv_len == 0u || nsplit == 0u) return;

  const u32 kh = wgid() / nsplit;
  const u32 s = wgid() - kh * nsplit;
  if (kh >= n_kv) return;

  const u32 h0 = kh * INK_R8_DH_GQA;
  const u32 kvd = n_kv * INK_R8_DH_HD;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();

  /* Logical sequence position of tree node t = pos0 + dep_t. */
  const u32 dep_t = (pos_off_packed >> (t << 2u)) & 0xFu;
  const u32 logical_pos = pos0 + dep_t;
  const u32 logical_pend = logical_pos + 1u;
  const u32 w0 = (win > 0u && logical_pend > win) ? (logical_pend - win) : 0u;
  const u32 np_pfx = (pos0 > w0) ? (pos0 - w0) : 0u;

  /* Partition committed prefix [w0 .. pos0 - 1] across nsplit splits; split 0 also adds intra-tree ancestors. */
  const u32 ch = (np_pfx + nsplit - 1u) / nsplit;
  const u32 off0 = s * ch;
  const u32 p0 = (off0 < np_pfx) ? (w0 + off0) : pos0;
  const u32 p1 = (off0 < np_pfx && off0 + ch < np_pfx) ? (w0 + off0 + ch) : pos0;

  const G float *qt = (const G float *)((const G u8 *)q + (unsigned long)t * (unsigned long)qs);
  const G float *Rt = (const G float *)((const G u8 *)R + (unsigned long)t * (unsigned long)rs);
  const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * (unsigned long)ss);
  G float *ot = (G float *)((G u8 *)out + (unsigned long)t * (unsigned long)os);

  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const float scale = 1.0f / __builtin_sqrtf((float)INK_R8_DH_HD);

  const f32x4 q0 = ((const G f32x4 *)(qt + (unsigned long)(h0 + 0u) * INK_R8_DH_HD))[l];
  const f32x4 q1 = ((const G f32x4 *)(qt + (unsigned long)(h0 + 1u) * INK_R8_DH_HD))[l];
  const f32x4 q2 = ((const G f32x4 *)(qt + (unsigned long)(h0 + 2u) * INK_R8_DH_HD))[l];
  const float rv0 = (l < INK_R8_DH_DREL) ? Rt[(unsigned long)(h0 + 0u) * INK_R8_DH_DREL + l] : 0.0f;
  const float rv1 = (l < INK_R8_DH_DREL) ? Rt[(unsigned long)(h0 + 1u) * INK_R8_DH_DREL + l] : 0.0f;
  const float rv2 = (l < INK_R8_DH_DREL) ? Rt[(unsigned long)(h0 + 2u) * INK_R8_DH_DREL + l] : 0.0f;

  f32x4 acc0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 acc1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 acc2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  float m0 = -__builtin_inff(), sum0 = 0.0f;
  float m1 = -__builtin_inff(), sum1 = 0.0f;
  float m2 = -__builtin_inff(), sum2 = 0.0f;

  const G kvt *kc_h = kc + (unsigned long)kh * INK_R8_DH_HD;
  const G kvt *vc_h = vc + (unsigned long)kh * INK_R8_DH_HD;

  /* 1) Committed prefix KV [p0 .. p1 - 1]: exact relative distance = logical_pos - kp. */
  u32 kp = p0 + w;
  if (kp < p1) {
    u32 kvp = kp % kv_len;
    kv4 kh4 = ((const G kv4 *)(kc_h + (unsigned long)kvp * (unsigned long)kvd))[l];
    kv4 vh4 = ((const G kv4 *)(vc_h + (unsigned long)kvp * (unsigned long)kvd))[l];
    for (;;) {
      const u32 dist = logical_pos - kp;
      const int has_rel = (dist < extent_lim);
      float rb = (has_rel && l < INK_R8_DH_DREL) ? rel_bias[(unsigned long)dist * INK_R8_DH_DREL + l] : 0.0f;

      const u32 kp_nxt = kp + 8u;
      u32 kvp_nxt = kvp;
      kv4 kh4_nxt = kh4, vh4_nxt = vh4;
      if (kp_nxt < p1) {
        kvp_nxt = kvp + 8u;
        if (kvp_nxt >= kv_len) kvp_nxt = (kv_len >= 8u) ? (kvp_nxt - kv_len) : (kp_nxt % kv_len);
        kh4_nxt = ((const G kv4 *)(kc_h + (unsigned long)kvp_nxt * (unsigned long)kvd))[l];
        vh4_nxt = ((const G kv4 *)(vc_h + (unsigned long)kvp_nxt * (unsigned long)kvd))[l];
      }

      const f32x4 kf = __builtin_convertvector(kh4, f32x4);
      float d0 = wave_sum_all(ink_r8_dh_dot4(q0, kf)) * scale;
      float d1 = wave_sum_all(ink_r8_dh_dot4(q1, kf)) * scale;
      float d2 = wave_sum_all(ink_r8_dh_dot4(q2, kf)) * scale;
      if (has_rel) {
        d0 += ink_r8_dh_hw_sum(rv0 * rb);
        d1 += ink_r8_dh_hw_sum(rv1 * rb);
        d2 += ink_r8_dh_hw_sum(rv2 * rb);
      }

      const float mn0 = __builtin_fmaxf(m0, d0);
      const float c0 = (sum0 > 0.0f) ? __builtin_expf(m0 - mn0) : 0.0f;
      const float e0 = __builtin_expf(d0 - mn0);
      sum0 = sum0 * c0 + e0; m0 = mn0;

      const float mn1 = __builtin_fmaxf(m1, d1);
      const float c1 = (sum1 > 0.0f) ? __builtin_expf(m1 - mn1) : 0.0f;
      const float e1 = __builtin_expf(d1 - mn1);
      sum1 = sum1 * c1 + e1; m1 = mn1;

      const float mn2 = __builtin_fmaxf(m2, d2);
      const float c2 = (sum2 > 0.0f) ? __builtin_expf(m2 - mn2) : 0.0f;
      const float e2 = __builtin_expf(d2 - mn2);
      sum2 = sum2 * c2 + e2; m2 = mn2;

      const f32x4 vf = __builtin_convertvector(vh4, f32x4);
      acc0 = acc0 * c0 + e0 * vf;
      acc1 = acc1 * c1 + e1 * vf;
      acc2 = acc2 * c2 + e2 * vf;

      if (kp_nxt >= p1) break;
      kp = kp_nxt; kvp = kvp_nxt; kh4 = kh4_nxt; vh4 = vh4_nxt;
    }
  }

  /* 2) Intra-tree ancestors [0 .. T - 1] handled in split s == 0 (1 wave per tree node w < T!). */
  if (s == 0u && w < T) {
    const u32 row_mask = (u32)((tree_mask >> (t << 3u)) & 0xFFull);
    if ((row_mask & (1u << w)) != 0u) {
      const u32 dep_j = (pos_off_packed >> (w << 2u)) & 0xFu;
      const u32 dist = (dep_t >= dep_j) ? (dep_t - dep_j) : 0u;
      if (win == 0u || dist < win) {
        const u32 kvp = (pos0 + w) % kv_len;
        const kv4 kh4 = ((const G kv4 *)(kc_h + (unsigned long)kvp * (unsigned long)kvd))[l];
        const kv4 vh4 = ((const G kv4 *)(vc_h + (unsigned long)kvp * (unsigned long)kvd))[l];
        const f32x4 kf = __builtin_convertvector(kh4, f32x4);
        const f32x4 vf = __builtin_convertvector(vh4, f32x4);
        const int has_rel = (dist < extent_lim);
        const float rb = (has_rel && l < INK_R8_DH_DREL) ? rel_bias[(unsigned long)dist * INK_R8_DH_DREL + l] : 0.0f;

        float d0 = wave_sum_all(ink_r8_dh_dot4(q0, kf)) * scale;
        float d1 = wave_sum_all(ink_r8_dh_dot4(q1, kf)) * scale;
        float d2 = wave_sum_all(ink_r8_dh_dot4(q2, kf)) * scale;
        if (has_rel) {
          d0 += ink_r8_dh_hw_sum(rv0 * rb);
          d1 += ink_r8_dh_hw_sum(rv1 * rb);
          d2 += ink_r8_dh_hw_sum(rv2 * rb);
        }

        const float mn0 = __builtin_fmaxf(m0, d0);
        const float c0 = (sum0 > 0.0f) ? __builtin_expf(m0 - mn0) : 0.0f;
        const float e0 = __builtin_expf(d0 - mn0);
        sum0 = sum0 * c0 + e0; m0 = mn0; acc0 = acc0 * c0 + e0 * vf;

        const float mn1 = __builtin_fmaxf(m1, d1);
        const float c1 = (sum1 > 0.0f) ? __builtin_expf(m1 - mn1) : 0.0f;
        const float e1 = __builtin_expf(d1 - mn1);
        sum1 = sum1 * c1 + e1; m1 = mn1; acc1 = acc1 * c1 + e1 * vf;

        const float mn2 = __builtin_fmaxf(m2, d2);
        const float c2 = (sum2 > 0.0f) ? __builtin_expf(m2 - mn2) : 0.0f;
        const float e2 = __builtin_expf(d2 - mn2);
        sum2 = sum2 * c2 + e2; m2 = mn2; acc2 = acc2 * c2 + e2 * vf;
      }
    }
  }

  /* 3) Wave merge + nsplit == 1 direct write OR Split-KV merge (identical to r4_attn). */
  if (l == 0u) {
    ink_r8_dh_tm[0u][w] = m0; ink_r8_dh_tl[0u][w] = sum0;
    ink_r8_dh_tm[1u][w] = m1; ink_r8_dh_tl[1u][w] = sum1;
    ink_r8_dh_tm[2u][w] = m2; ink_r8_dh_tl[2u][w] = sum2;
  }
  ((LDSP f32x4 *)&ink_r8_dh_to[0u][w][0u])[l] = acc0;
  ((LDSP f32x4 *)&ink_r8_dh_to[1u][w][0u])[l] = acc1;
  ((LDSP f32x4 *)&ink_r8_dh_to[2u][w][0u])[l] = acc2;
  lbarrier();

  if (nsplit == 1u) {
    for (u32 idx = tid(); idx < INK_R8_DH_GQA * INK_R8_DH_HD; idx += NT) {
      const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
      const float sv = st[h];
      float M = sv;
      for (u32 x = 0u; x < 8u; x++) if (ink_r8_dh_tl[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r8_dh_tm[g][x]);
      float L = __builtin_expf(sv - M), o = 0.0f;
      for (u32 x = 0u; x < 8u; x++) {
        const float lx = ink_r8_dh_tl[g][x];
        if (lx > 0.0f) {
          const float wx = __builtin_expf(ink_r8_dh_tm[g][x] - M);
          L += lx * wx; o += ink_r8_dh_to[g][x][e] * wx;
        }
      }
      ot[(unsigned long)h * INK_R8_DH_HD + e] = o / L;
    }
    return;
  }

  for (u32 idx = tid(); idx < INK_R8_DH_GQA * INK_R8_DH_HD; idx += NT) {
    const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
    float M = -__builtin_inff();
    for (u32 x = 0u; x < 8u; x++) if (ink_r8_dh_tl[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r8_dh_tm[g][x]);
    G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R8_DH_PSTR;
    float o = 0.0f, L = 0.0f;
    for (u32 x = 0u; x < 8u; x++) {
      const float lx = ink_r8_dh_tl[g][x];
      if (lx > 0.0f) {
        const float wx = __builtin_expf(ink_r8_dh_tm[g][x] - M);
        o += ink_r8_dh_to[g][x][e] * wx;
        if (e == 0u) L += lx * wx;
      }
    }
    pp[2u + e] = o;
    if (e == 0u) { pp[0u] = (L > 0.0f) ? M : -1e30f; pp[1u] = L; }
  }

  const u32 cidx = t * n_kv + kh;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r8_dh_tlast =
        (__scoped_atomic_fetch_add(&count[cidx], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r8_dh_tlast) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  if (tid() == 0u) count[cidx] = 0u;

  for (u32 idx = tid(); idx < INK_R8_DH_GQA * INK_R8_DH_HD; idx += NT) {
    const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
    const G float *ph = part + (unsigned long)((t * n_head + h) * nsplit) * INK_R8_DH_PSTR;
    const float sv = st[h];
    float M = sv;
    for (u32 sp = 0u; sp < nsplit; sp++) {
      const float ls = ph[(unsigned long)sp * INK_R8_DH_PSTR + 1u];
      if (ls > 0.0f) M = __builtin_fmaxf(M, ph[(unsigned long)sp * INK_R8_DH_PSTR]);
    }
    float L = __builtin_expf(sv - M), o = 0.0f;
    for (u32 sp = 0u; sp < nsplit; sp++) {
      const float ls = ph[(unsigned long)sp * INK_R8_DH_PSTR + 1u];
      if (ls > 0.0f) {
        const float wx = __builtin_expf(ph[(unsigned long)sp * INK_R8_DH_PSTR] - M);
        L += ls * wx;
        o += ph[(unsigned long)sp * INK_R8_DH_PSTR + 2u + e] * wx;
      }
    }
    ot[(unsigned long)h * INK_R8_DH_HD + e] = o / L;
  }
}

#undef INK_R8_DH_FMA4
#undef INK_R8_DH_ADD4
#undef INK_R8_DH_PSTR
#undef INK_R8_DH_GQA
#undef INK_R8_DH_DREL
#undef INK_R8_DH_HD
#undef INK_R8_DH_MAX_PAIRS
#undef INK_R8_DH_MAX_EXP
#undef INK_R8_DH_SB_BYTES
#undef INK_R8_DH_CAP
#undef INK_R8_DH_NGMAX
#undef INK_R8_DH_KMAX

#endif /* IE_MEGA */
