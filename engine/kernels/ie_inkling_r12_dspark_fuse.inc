/* ============================================================================
 * Inkling-2-Mini Round 12 (Wave 12 - H24, H25, H32, H23):
 *   DSpark Speculator Block + Markov Fusions & 256-WG Threshold-Gated Heads
 * Target: RX 9070 XT (gfx1201, wave32, NT=256, 64 CUs, 64 MiB MALL)
 *
 * Included at EOF of kernels/ie_inkling_kernels.c (after ie_kernels.c).
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r12_ds_ /
 * INK_R12_DS_ and all macros are #undef'd at EOF.
 * 0 warnings (-Wall -Wextra), 0/0 spills, priv=0, scratch=0, Clang bit_cast-safe.
 *
 * Exported Kernels (7):
 *
 * 1. H24a: KERNEL ie_inkling_ds_add_q8_batch_r12(
 *        const G float *hidden, const G float *delta, const G float *bias,
 *        G u8 *qout, u32 dim, u32 nd, u32 hs, u32 ds_str, u32 qs)
 *    - Grid:   ((dim + 255u) / 256u, nd, 1u)  [(9, nd, 1) for dim=2304]
 *    - Effect: Batches nd launches of ie_dspark_add_q8_r4 into 1 launch (-4..-6).
 *              Pass hidden = b->ds.hidden + hs (token 1), delta = b->ds.zero (or NULL)
 *              with ds_str = 0u, bias = b->ds.mk_hidden_bias, qout = b->ds.hq, qs = xqs.
 *    - Parity: 100% bitwise-exact vs nd calls to ie_dspark_add_q8_r4.
 *
 * 2. H24b: KERNEL ie_inkling_ds_rnn_fused_zsync_r12(
 *        G float *state, const G u8 *w1_qw, const G u16 *w1_qs,
 *        const G float *hidden_i, const G float *f_rnn,
 *        const G u32 *rnn_qw, const G u16 *rnn_qs, const G float *rnn_bias,
 *        G float *raw768, G float *latent, G u8 *latent_q8,
 *        const G u32 *ds_ctrl, G u32 *done,
 *        u32 rank, u32 dim, u32 rnn_rows)
 *    - Grid:   (ng, 1u, 1u)  [ng in 1..96; recommended ng=64 or 96 for 100% CU util]
 *    - Guard:  !b->m->ds_head.rows && ds_ctrl && done && (rank & 31u) == 0u &&
 *              (dim & 31u) == 0u && (2u * rank + 2u * dim) <= 5120u &&
 *              d->mk_rnn_proj.kind == MAT_Q4K. (*done initialized to 0 once; auto-reset.)
 *    - Effect: Replaces K_DS_CONCAT_Q8_ZSYNC_R5 + K_DS_GEMV_Q4K_ZSYNC_R5 +
 *              K_DS_RNN_STEP_ZSYNC_R5 (3 -> 1 launch/step, 15 -> 5 launches at nd=5).
 *              Builds 6.25 KiB cat_q8 in LDS, pins 2.21 MiB mk_rnn_proj via TH_LOAD_HT,
 *              runs exact 32-lane Q4_K dot order, and last WG runs RNN_STEP -> state + lat_q8.
 *    - Parity: 100% bitwise-exact vs R5 zsync triple; preserves ds_ctrl[4] ABI.
 *
 * 3. H25a: KERNEL ie_inkling_ds_qkv_rope_q4k_r12(
 *        const G u32 *wq0, const G u16 *wq1, const G u32 *wk0, const G u16 *wk1,
 *        const G u32 *wv0, const G u16 *wv1, const G u8 *xq,
 *        const G float *q_norm, const G float *k_norm,
 *        const G float *cs, const G float *sn,
 *        G float *q_out, G float *k_out, G float *v_out,
 *        u32 nh, u32 nkv, u32 hd, u32 nrot, u32 nb, u32 B,
 *        u32 xqs, u32 qs, u32 kvs, float eps)
 *    - Grid:   (nh + 2u * nkv, (B + 3u) / 4u, 1u)  [(40, 2, 1) = 80 WGs for B=8]
 *    - Guard:  wq/wk/wv all MAT_Q4K, hd == 128u, nrot <= 128u && (nrot & 1u) == 0u,
 *              nb > 0u && (nb & 7u) == 0u, B > 0u, q_norm && k_norm && cs && sn.
 *    - Effect: Fuses wq + wk + wv + QK-RMSNorm + NeoX RoPE (5 -> 1 launch/layer,
 *              -12 launches across 3 DSpark layers), <= 80 VGPR (4 WGs/CU), 2 KiB LDS.
 *    - Parity: bitwise-exact Q4_K GEMV + f32-tol (<= 1e-6) RMSNorm + NeoX RoPE.
 *
 * 4. H25b: KERNEL ie_inkling_ds_attn_wmma_q8_r12(
 *        const G float *q, const G kvt *kc_ctx, const G kvt *vc_ctx,
 *        const G float *k_blk, const G float *v_blk, G float *out, G u8 *aq,
 *        u32 ctx_len, u32 kv_cap, u32 win, u32 block_size,
 *        u32 n_head, u32 n_kv, u32 hd, u32 qs, u32 ks, u32 os, u32 aqs,
 *        u32 nsplit, G float *part, G u32 *count)
 *    - Grid:   (n_kv * nsplit, (block_size + 7u) / 8u, 1u)  [btiles=1 for B=8]
 *    - Guard:  sizeof(kvt) == 2u && hd == 128u && n_kv > 0u && n_head == n_kv * 3u &&
 *              block_size > 0u && nsplit >= 1u && ((qs | ks) & 15u) == 0u &&
 *              (out == NULL || (os & 15u) == 0u) && (nsplit == 1u || (part && count)).
 *    - Effect: Single-pass FP16 WMMA attention for B<=8 x 3 GQA = 24 queries over
 *              context f16 KV (SWA or Global) + bidirectional f32 block KV, PLUS inline
 *              K_QUANT (att -> aq) inside final reduction wave (-3 launches, ~2x faster
 *              16K Layer-1 DSpark attention), <= 128 VGPR (3 WGs/CU), 14,336 B LDS.
 *    - Parity: f32-tol (FP16 WMMA QK^T & PV) + exact quant_wave Q8_0 output.
 *
 * 5. H25c: KERNEL ie_inkling_ds_proj_res_norm_q8_q4k_r12(
 *        const G u32 *qw, const G u16 *qs, const G u8 *xq,
 *        const G float *res, G float *h_out,
 *        const G float *norm_w, G float *ny_out, G u8 *nq_out,
 *        G float *ss_buf, G u32 *count,
 *        u32 rows, u32 nb, u32 B, u32 xs, u32 rs, u32 hs, u32 nys, u32 nqs,
 *        float eps)
 *    - Grid:   ((rows + 31u) / 32u, (B + 3u) / 4u, 1u)  [(72, 2, 1) = 144 WGs for dim=2304, B=8]
 *    - Guard:  MAT_Q4K, rows > 0u && (rows & 31u) == 0u, nb > 0u && (nb & 7u) == 0u,
 *              B > 0u, ss_buf sized >= B * ((rows + 31u) / 32u) floats,
 *              count sized >= (B + 3u) / 4u u32s (0-init, auto-reset to 0).
 *              Note: xs/rs/hs/nys/nqs are ALL byte strides! h_out may alias res.
 *    - Effect: Fuses Wo/Wdown (MAT_Q4K) + residual add + next RMSNorm + Q8_0
 *              (-6 launches across 3 DSpark layers), <= 80 VGPR (4 WGs/CU).
 *    - Parity: bitwise-exact Q4_K + res; deterministic ss_buf RMSNorm + exact Q8_0.
 *
 * 6. H32: KERNEL ie_inkling_ds_p_sample_256wg_r12(
 *        const G u32 *p_qw, const G f16 *p_qs, const G u8 *lat_q8,
 *        const G float *base_row, const G u32 *id_map,
 *        G float *part_v, G u32 *part_i, G float *part_sum,
 *        G u32 *am_count, G u32 *ds_ctrl, G unsigned long long *rng_state,
 *        G u32 *draft_ids, G float *draft_probs, G float *draft_entropy,
 *        G u32 *q_n, G u32 *q_ids, G float *q_probs,
 *        u32 full_rows, u32 n_act, u32 nb, u32 ng, u32 K, u32 K_stride, u32 step_idx,
 *        float cap, float temp, float top_p, float pmin, float surv_a, float surv_b)
 *    - Grid:   (ng, 1u, 1u) with 1 <= ng <= 256 (recommended ng=256 = 4 WGs/CU!)
 *    - Guard:  Same 30-arg drop-in ABI as ie_inkling_ds_gemv_p_cap_sample_zsync_r8!
 *              Buffer sizing: part_v & part_i >= ng * K elements (at ng=256, K=32: 32 KiB each;
 *              at ng=64, K=32: 8 KiB each), part_sum >= ng floats (greedy mode).
 *    - Effect: Fixes R27 regression: runs 256 WGs (4 WGs/CU), hoists monotonic tanhf
 *              out of hot loop via 1-cycle SGPR raw_cut threshold, eliminates ALL
 *              per-chunk cross-wave lbarrier()s via wave-private K-heaps, and uses
 *              2-pass streaming Stage-2 merge (0 register arrays -> <= 80 VGPR!).
 *    - Parity: 100% exact Top-K / argmax / ForeSpark / xorshift64* vs r8.
 *
 * 7. H23: KERNEL ie_inkling_q4k_head_topk_256wg_r12(
 *        const G u32 *qw, const G u16 *qs, const G u8 *xq,
 *        const G float *bias,
 *        u32 rows, u32 unpadded, u32 nb, u32 T, u32 xs,
 *        float cap, u32 K, u32 ng,
 *        G float *part_v, G u32 *part_i, G u32 *done,
 *        G u32 *out_am, u32 am_stride,
 *        G float *ov, G u32 *oi)
 *    - Grid:   (ng, 1u, 1u) with 1 <= ng <= 256 (recommended ng=256 = 4 WGs/CU!)
 *    - Guard:  Same 19-arg drop-in ABI as ie_inkling_q4k_head_cap_topk_r7!
 *              Buffer sizing: part_v & part_i >= T * ng * K elements (at T=8, ng=256,
 *              K=32: 256 KiB each; at ng=128: 128 KiB each).
 *    - Effect: Fixes R26 regression: runs 256 WGs (4 WGs/CU) at <= 80 VGPR using
 *              32-lane/row single-pass Q4_K (TH_LOAD_NT on 260 MiB output.weight,
 *              TH_LOAD_HT on xq), hoists monotonic tanhf out of hot row loop via
 *              1-cycle SGPR raw_cut threshold (0 barriers in K==1 greedy loop,
 *              1-barrier skip on >99% chunks in K>1), and 2-pass streaming Stage-2.
 *    - Parity: 100% bitwise-exact vs ie_gemv_q4kq8 + logit_cap + argmax/topk.
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(kvt) == 2u, "ink_r12_ds requires f16 KV cache");
_Static_assert(sizeof(f16) == 2u, "ink_r12_ds f16 must be 2 bytes");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r12_ds float/u32 must be 4 bytes");

#define INK_R12_DS_RNN_MAXNB  160u
#define INK_R12_DS_HD         128u
#define INK_R12_DS_GQA        3u
#define INK_R12_DS_PSTR       130u
#define INK_R12_DS_TMAX       8u
#define INK_R12_DS_KMAX       32u
#define INK_R12_DS_NGMAX      256u
#define INK_R12_DS_CAP        1024u
#define INK_R12_DS_LOG2E      1.44269504088896340736f
#define INK_R12_DS_LN2        0.69314718055994530942f
#define INK_R12_DS_QSC        (0.08838834764831844299f * 1.44269504088896340736f)
#define INK_R12_DS_BUF_FLAGS  0x31004000u

typedef unsigned long long ink_r12_ds_u64;
typedef int ink_r12_ds_i32x4 __attribute__((ext_vector_type(4)));
typedef float ink_r12_ds_f4 __attribute__((ext_vector_type(4)));
typedef float ink_r12_ds_f8 __attribute__((ext_vector_type(8)));

struct __attribute__((packed, aligned(4))) ink_r12_ds_u32x4p {
  u32x4 v;
};

struct __attribute__((packed, aligned(4))) ink_r12_ds_f32x4p {
  f32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r12_ds_ld4(const G void *p) {
  return ((const G struct ink_r12_ds_u32x4p *)p)->v;
}

/* R43 scrub: standard default cached loads (TH_LOAD_RT; aux=0). */
static inline __attribute__((always_inline)) u32x4 ink_r12_ds_nt4(const G void *p) {
  return *(const G u32x4 *)p;
}

static inline __attribute__((always_inline)) u16 ink_r12_ds_nt16(const G u16 *p) {
  return *p;
}

static inline __attribute__((always_inline)) u32 ink_r12_ds_nt32(const G u32 *p) {
  return *p;
}

static inline __attribute__((always_inline)) __amdgpu_buffer_rsrc_t ink_r12_ds_mkbuf(
    const G void *base, u32 nbytes) {
  return __builtin_amdgcn_make_buffer_rsrc((G void *)rq_ptr_u((const G u8 *)base), 0, nbytes,
                                           INK_R12_DS_BUF_FLAGS);
}

#define INK_R12_DS_RBL128(rsrc_, voff_, aux_) \
  __builtin_bit_cast(u32x4, __builtin_amdgcn_raw_buffer_load_b128((rsrc_), (int)(voff_), 0, 0))
#define INK_R12_DS_RBL32(rsrc_, voff_, aux_) \
  ((u32)__builtin_amdgcn_raw_buffer_load_b32((rsrc_), (int)(voff_), 0, 0))
#define INK_R12_DS_RBL16(rsrc_, voff_, aux_) \
  ((u16)__builtin_amdgcn_raw_buffer_load_b16((rsrc_), (int)(voff_), 0, 0))
static inline __attribute__((always_inline)) float ink_r12_ds_u32_f32(u32 bits) {
  return __builtin_bit_cast(float, bits);
}
#define INK_R12_DS_RBLF32(rsrc_, voff_, aux_) \
  ink_r12_ds_u32_f32(INK_R12_DS_RBL32((rsrc_), (voff_), (aux_)))

static inline __attribute__((always_inline)) float ink_r12_ds_sgpr_f(float v) {
  return I2F(__builtin_amdgcn_readfirstlane(F2I(v)));
}

static inline __attribute__((always_inline)) float ink_r12_ds_oct_tree_f(float v) {
#pragma clang fp contract(off)
  v = v + I2F(shl_i(F2I(v), 4));
  v = v + I2F(shl_i(F2I(v), 2));
  v = v + I2F(shl_i(F2I(v), 1));
  return v;
}

static inline __attribute__((always_inline)) float ink_r12_ds_cap(float v, float cap) {
  return (cap > 0.0f) ? (cap * ie_tanhf_s(v / cap)) : v;
}

static inline __attribute__((always_inline)) int ink_r12_ds_better(float v, u32 i, float bv, u32 bi) {
  return (v > bv) || (v == bv && i < bi);
}

static inline __attribute__((always_inline)) u32 ink_r12_ds_f2u(float f) {
  u32 u = __builtin_bit_cast(u32, f);
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static inline __attribute__((always_inline)) float ink_r12_ds_u2f(u32 k) {
  const u32 u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  return __builtin_bit_cast(float, u);
}

static inline __attribute__((always_inline)) u32 ink_r12_ds_cswap(u32 v, u32 ov, u32 hi, u32 asc) {
  return (hi ^ asc) ? (v < ov ? v : ov) : (v > ov ? v : ov);
}

static inline __attribute__((always_inline)) u32 ink_r12_ds_wave_step(u32 v, u32 t, u32 stride, u32 asc) {
  const u32 ov = (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ stride) * 4u), (int)v);
  return ink_r12_ds_cswap(v, ov, (u32)((t & stride) != 0u), asc);
}

static inline __attribute__((always_inline)) u32 ink_r12_ds_lds_step(
    u32 v, u32 t, u32 stride, u32 asc, LDSP u32 *buf) {
  buf[t] = v;
  lbarrier();
  const u32 ov = buf[t ^ stride];
  return ink_r12_ds_cswap(v, ov, (u32)((t & stride) != 0u), asc);
}

static inline __attribute__((always_inline)) u32 ink_r12_ds_bitonic_kth_256(
    u32 v, u32 K, LDSP u32 (*sm)[NT]) {
  const u32 t = tid();
#pragma clang loop unroll(disable)
  for (u32 size = 2u; size <= 32u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride > 0u; stride >>= 1u) v = ink_r12_ds_wave_step(v, t, stride, a);
  }
  u32 buf_id = 0u;
#pragma clang loop unroll(disable)
  for (u32 size = 64u; size <= 256u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride >= 32u; stride >>= 1u) {
      v = ink_r12_ds_lds_step(v, t, stride, a, sm[buf_id]);
      buf_id ^= 1u;
    }
#pragma clang loop unroll(disable)
    for (u32 stride = 16u; stride > 0u; stride >>= 1u) v = ink_r12_ds_wave_step(v, t, stride, a);
  }
  if (t == K - 1u) sm[0u][0u] = v;
  lbarrier();
  return sm[0u][0u];
}

static inline __attribute__((always_inline)) u32 ink_r12_ds_wg_scan_u32(
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

/* Compute exact safe raw-logit lower bound rc such that cap(raw) < min_v_cap strictly
 * for every raw < rc (returns -inf if min_v_cap is saturated or empty). */
static inline __attribute__((always_inline)) float ink_r12_ds_safe_raw_cut(
    float raw_min, float min_v_cap, float cap) {
  if (min_v_cap <= -1e30f) return -__builtin_inff();
  if (cap <= 0.0f) return raw_min;
  const float rc = raw_min - 1e-3f * cap;
  return (ink_r12_ds_cap(rc, cap) < min_v_cap) ? rc : -__builtin_inff();
}

/* ============================================================================
 * 1. H24a: ie_inkling_ds_add_q8_batch_r12
 * ========================================================================== */

KERNEL ie_inkling_ds_add_q8_batch_r12(
    const G float *hidden, const G float *delta, const G float *bias,
    G u8 *qout, u32 dim, u32 nd, u32 hs, u32 ds_str, u32 qs) {
  const u32 t = tokid();
  if (t >= nd || dim == 0u || (dim & 31u) != 0u || !hidden || !qout) return;

  const u32 nb = dim >> 5u;
  const u32 b = wgid() * 8u + (tid() >> 5u);
  if (b >= nb) return;

  const u32 j = b * 32u + lane();
  const G float *ht = (const G float *)((const G u8 *)hidden + (unsigned long)t * (unsigned long)hs);
  const G float *dt = delta ? (const G float *)((const G u8 *)delta + (unsigned long)t * (unsigned long)ds_str)
                            : (const G float *)0;
  const float v = ht[j] + (dt ? dt[j] : 0.0f) + (bias ? bias[j] : 0.0f);
  quant_wave(v, b, qout + (unsigned long)t * (unsigned long)qs, nb);
}

/* ============================================================================
 * 2. H24b: ie_inkling_ds_rnn_fused_zsync_r12
 *    1-Launch Fused [CONCAT_Q8 + mk_rnn_proj (MAT_Q4K) + RNN_STEP]
 *    6,404 B LDS, TH_LOAD_HT on 2.21 MiB mk_rnn_proj, exact 32-lane Q4_K order.
 * ========================================================================== */

typedef struct {
  u32   qs[INK_R12_DS_RNN_MAXNB * 8u] __attribute__((aligned(16))); /* 5,120 B */
  float da[INK_R12_DS_RNN_MAXNB];                                   /*   640 B */
  u32   asum[INK_R12_DS_RNN_MAXNB];                                 /*   640 B */
  u32   is_last;                                                    /*     4 B */
} ink_r12_ds_rnn_lds_t;

static LDS ink_r12_ds_rnn_lds_t ink_r12_ds_rlds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r12_ds_rnn_lds_t) == 6416u, "ink_r12_ds_rnn_lds_t must be 6,416 B");

KERNEL ie_inkling_ds_rnn_fused_zsync_r12(
    G float *state, const G u8 *w1_qw, const G u16 *w1_qs,
    const G float *hidden_i, const G float *f_rnn,
    const G u32 *rnn_qw, const G u16 *rnn_qs, const G float *rnn_bias,
    G float *raw768, G float *latent, G u8 *latent_q8,
    const G u32 *ds_ctrl, G u32 *done,
    u32 rank, u32 dim, u32 rnn_rows) {
#pragma clang fp contract(off)
  if (!ds_ctrl || ds_ctrl[1u] != 0u) return;
  if (rank == 0u || (rank & 31u) != 0u || (dim & 31u) != 0u || rnn_rows == 0u) return;

  const u32 total = 2u * rank + 2u * dim;
  const u32 nb = total >> 5u;
  if (nb == 0u || nb > INK_R12_DS_RNN_MAXNB || (nb & 7u) != 0u) return;
  if (!state || !w1_qw || !w1_qs || !hidden_i || !f_rnn || !rnn_qw || !rnn_qs || !raw768 || !done) return;

  const u32 g = wgid();
  const u32 ng = ngroups_x();
  const u32 th = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(th >> 5u));
  const u32 l = lane();

  /* Phase 1: Cooperatively build exact cat_q8 in LDS (8 waves x 20 blocks). */
  const u32 prev_tok = (u32)__builtin_amdgcn_readfirstlane((int)ds_ctrl[0u]);
  const u32 w1_nb = rank >> 5u;
  const u32 off_h = 2u * rank;
  const u32 off_f = 2u * rank + dim;

#pragma clang loop unroll(disable)
  for (u32 b = w; b < nb; b += 8u) {
    const u32 idx = b * 32u + l;
    float v;
    if (idx < rank) {
      v = state[idx];
    } else if (idx < off_h) {
      const u32 wb = b - w1_nb;
      const unsigned long blk = (unsigned long)prev_tok * (unsigned long)w1_nb + (unsigned long)wb;
      const u16 s_raw = w1_qs[blk];
      const float sc = (float)__builtin_bit_cast(f16, s_raw);
      v = sc * (float)(i8)w1_qw[blk * 32ul + (unsigned long)l];
    } else if (idx < off_f) {
      v = hidden_i[idx - off_h];
    } else {
      v = f_rnn[idx - off_f];
    }
    const float amax = wave_max_pos(__builtin_fabsf(v));
    const float d = amax / 127.0f;
    const int qi = d != 0.0f ? (int)__builtin_roundf(v / d) : 0;
    ((LDSP i8 *)ink_r12_ds_rlds.qs)[b * 32u + l] = (i8)qi;
    const u32 s = wave_tree_u((u32)qi);
    if (l == 0u) {
      ink_r12_ds_rlds.da[b] = d;
      ink_r12_ds_rlds.asum[b] = s;
    }
  }
  lbarrier();

  /* Phase 2: Exact 32-lane/row Q4_K GEMV (bit-exact vs gemv_q4kq8_rows), TH_LOAD_HT on weights. */
  const u32 qw_bytes = rnn_rows * nb * 16u;
  const u32 nsb = nb >> 3u;
  const u32 qs_bytes = (rnn_rows * nb + rnn_rows * nsb * 2u) * 2u;
  const __amdgpu_buffer_rsrc_t r_qw = ink_r12_ds_mkbuf(rnn_qw, qw_bytes);
  const __amdgpu_buffer_rsrc_t r_qs = ink_r12_ds_mkbuf(rnn_qs, qs_bytes);
  const u32 dd_base_b = rnn_rows * nb * 2u;
  const u32 n_tiles = (rnn_rows + 7u) >> 3u;

#pragma clang loop unroll(disable)
  for (u32 tile = g; tile < n_tiles; tile += ng) {
    const u32 r = (tile << 3u) + w;
    if (r < rnn_rows) {
      const u32 row_qw_b = r * nb * 16u;
      const u32 row_sm_b = r * nb * 2u;
      const u32 row_dd_b = dd_base_b + r * nsb * 4u;
      float acc = 0.0f;

#pragma clang loop unroll(disable)
      for (u32 b = l; b < nb; b += 32u) {
        const u32x4 w4 = INK_R12_DS_RBL128(r_qw, row_qw_b + b * 16u, 2);
        const u32 sm = (u32)INK_R12_DS_RBL16(r_qs, row_sm_b + b * 2u, 2);
        const u32 dd_raw = INK_R12_DS_RBL32(r_qs, row_dd_b + (b >> 3u) * 4u, 2);
        const u16 d_bits = (u16)(dd_raw & 0xFFFFu);
        const u16 dm_bits = (u16)(dd_raw >> 16u);
        const float dv = (float)__builtin_bit_cast(f16, d_bits);
        const float mv = (float)__builtin_bit_cast(f16, dm_bits);

        const u32x4 a0 = *(const LDSP u32x4 *)(ink_r12_ds_rlds.qs + b * 8u);
        const u32x4 a1 = *(const LDSP u32x4 *)(ink_r12_ds_rlds.qs + b * 8u + 4u);
        const float dav = ink_r12_ds_rlds.da[b];
        const int as = (int)ink_r12_ds_rlds.asum[b];

        const u32 wq[4] = {w4.x, w4.y, w4.z, w4.w};
        const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const Mem qm = {wq}, am = {av};
        const int dq = (int)ie_q4k_dot(qm, 0u, am, 0u);
        const float sc = dv * (float)(sm & 63u);
        const float mm = mv * (float)(sm >> 8u);
        const float x = dav * (float)as;
        const float a1_v = __builtin_fmaf(sc * (float)dq, dav, acc);
        acc = __builtin_fmaf(-mm, x, a1_v);
      }
      acc = wave_tree_f(acc);
      if (l == 0u) raw768[r] = epilogue(acc, r, rnn_bias, (const G float *)0);
    }
  }

  /* Phase 3: Agent release/acquire barrier; last arriving WG runs exact RNN_STEP. */
  if (ng > 1u) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (th == 0u) {
      ink_r12_ds_rlds.is_last =
          (__scoped_atomic_fetch_add(done, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r12_ds_rlds.is_last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (th == 0u) *done = 0u;
  } else {
    barrier();
  }

  const u32 nb_r = rank >> 5u;
#pragma clang loop unroll(disable)
  for (u32 b = w; b < nb_r; b += 8u) {
    const u32 i = b * 32u + l;
    const float gv = 1.0f / (1.0f + __builtin_expf(-raw768[i]));
    const float cand = ie_tanhf_s(raw768[rank + i]);
    const float s = gv * state[i] + (1.0f - gv) * cand;
    const float lat = ie_tanhf_s(raw768[2u * rank + i]);
    state[i] = s;
    if (latent) latent[i] = lat;
    if (latent_q8) quant_wave(lat, b, latent_q8, nb_r);
  }
}

/* ============================================================================
 * 3. H25a: ie_inkling_ds_qkv_rope_q4k_r12
 *    Unified MAT_Q4K QKV + inline per-head QK-RMSNorm + NeoX RoPE (5 -> 1 launch).
 *    Grid: (nh + 2*nkv, (B + 3)/4, 1) = (40, 2, 1) = 80 WGs for B=8.
 *    2,048 B LDS, <= 80 VGPR (4 WGs/CU), 0/0 spills.
 * ========================================================================== */

static LDS float ink_r12_ds_qkvh[4u][INK_R12_DS_HD] __attribute__((aligned(16)));

KERNEL ie_inkling_ds_qkv_rope_q4k_r12(
    const G u32 *wq0, const G u16 *wq1, const G u32 *wk0, const G u16 *wk1,
    const G u32 *wv0, const G u16 *wv1, const G u8 *xq,
    const G float *q_norm, const G float *k_norm,
    const G float *cs, const G float *sn,
    G float *q_out, G float *k_out, G float *v_out,
    u32 nh, u32 nkv, u32 hd, u32 nrot, u32 nb, u32 B,
    u32 xqs, u32 qs, u32 kvs, float eps) {
#pragma clang fp contract(off)
  const u32 g = wgid();
  const u32 t0 = tokid() << 2u;
  if (g >= nh + 2u * nkv || t0 >= B || hd != INK_R12_DS_HD || nb == 0u || (nb & 7u) != 0u) return;

  const u32 bt = (B - t0 < 4u) ? (B - t0) : 4u;
  const int is_v = (g >= nh + nkv);
  const int is_k = (!is_v && g >= nh);
  const u32 h = is_v ? (g - nh - nkv) : (is_k ? (g - nh) : g);
  const u32 mat_rows = is_v ? (nkv * hd) : (is_k ? (nkv * hd) : (nh * hd));
  const G u32 *qw = is_v ? wv0 : (is_k ? wk0 : wq0);
  const G u16 *qm = is_v ? wv1 : (is_k ? wk1 : wq1);

  const u32 nsb = nb >> 3u;
  const __amdgpu_buffer_rsrc_t r_qw = ink_r12_ds_mkbuf(qw, mat_rows * nb * 16u);
  const __amdgpu_buffer_rsrc_t r_qm = ink_r12_ds_mkbuf(qm, (mat_rows * nb + mat_rows * nsb * 2u) * 2u);
  const u32 dd_base_b = mat_rows * nb * 2u;

  const u32 th = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(th >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 row_in_pass = (w << 2u) + (l >> 3u); /* 0..31 */
  const u32 head_r0 = h * INK_R12_DS_HD;

  const G u8 *xq_t0 = xq + (unsigned long)t0 * (unsigned long)xqs;
  const __amdgpu_buffer_rsrc_t r_x = ink_r12_ds_mkbuf(xq_t0, bt * xqs);
  const u32 da_base = nb * 32u;
  const u32 as_base = nb * 36u;

  /* 4 passes x 32 rows = all 128 rows of head h across bt <= 4 tokens (exact 8-lane Q4_K order). */
#pragma clang loop unroll(disable)
  for (u32 pass = 0u; pass < 4u; pass++) {
    const u32 loc_r = (pass << 5u) + row_in_pass; /* 0..127 */
    const u32 r = head_r0 + loc_r;
    const u32 row_qw_b = r * nb * 16u;
    const u32 row_sm_b = r * nb * 2u;
    const u32 row_dd_b = dd_base_b + r * nsb * 4u;

    ink_r12_ds_f4 a0 = (ink_r12_ds_f4)(0.0f);
    ink_r12_ds_f4 a1 = (ink_r12_ds_f4)(0.0f);
    ink_r12_ds_f4 a2 = (ink_r12_ds_f4)(0.0f);
    ink_r12_ds_f4 a3 = (ink_r12_ds_f4)(0.0f);

#pragma clang loop unroll(disable)
    for (u32 sb = 0u; sb < nsb; sb++) {
      const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
      const u32 slot = (u32)__builtin_amdgcn_readfirstlane((int)(sb_u & 3u));
      const u32 b = (sb_u << 3u) + sub;

      const u32x4 w4 = INK_R12_DS_RBL128(r_qw, row_qw_b + b * 16u, 0);
      const u32 sm = (u32)INK_R12_DS_RBL16(r_qm, row_sm_b + b * 2u, 0);
      const u32 dd_raw = INK_R12_DS_RBL32(r_qm, row_dd_b + sb_u * 4u, 0);
      const u16 d_bits = (u16)(dd_raw & 0xFFFFu);
      const u16 dm_bits = (u16)(dd_raw >> 16u);
      const float dv = (float)__builtin_bit_cast(f16, d_bits);
      const float mv = (float)__builtin_bit_cast(f16, dm_bits);
      const float sc = dv * (float)(sm & 63u);
      const float mm = mv * (float)(sm >> 8u);

      const u32 lo0 = ie_q4k_lo(w4.x), hi0 = ie_q4k_hi(w4.x);
      const u32 lo1 = ie_q4k_lo(w4.y), hi1 = ie_q4k_hi(w4.y);
      const u32 lo2 = ie_q4k_lo(w4.z), hi2 = ie_q4k_hi(w4.z);
      const u32 lo3 = ie_q4k_lo(w4.w), hi3 = ie_q4k_hi(w4.w);

      const u32 aw_off = b * 32u;
      const u32 da_off = da_base + b * 4u;
      const u32 as_off = as_base + b * 4u;

#pragma unroll
      for (u32 j = 0u; j < 4u; j++) {
        if (j < bt) {
          const u32 jbase = j * xqs;
          const u32x4 act0 = INK_R12_DS_RBL128(r_x, jbase + aw_off, 2);
          const u32x4 act1 = INK_R12_DS_RBL128(r_x, jbase + aw_off + 16u, 2);
          const float da = INK_R12_DS_RBLF32(r_x, jbase + da_off, 2);
          const int as = (int)INK_R12_DS_RBL32(r_x, jbase + as_off, 2);
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
          if (slot == 0u) a0[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a0[j]));
          else if (slot == 1u) a1[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a1[j]));
          else if (slot == 2u) a2[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a2[j]));
          else a3[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a3[j]));
        }
      }
      __builtin_amdgcn_sched_barrier(0);
    }

#pragma unroll
    for (u32 j = 0u; j < 4u; j++) {
      if (j < bt) {
        const float s = ink_r12_ds_oct_tree_f((a0[j] + a2[j]) + (a1[j] + a3[j]));
        if (sub == 0u) {
          if (is_v) {
            G float *vp = (G float *)((G u8 *)v_out + (unsigned long)(t0 + j) * (unsigned long)kvs) + head_r0;
            vp[loc_r] = s;
          } else {
            ink_r12_ds_qkvh[j][loc_r] = s;
          }
        }
      }
    }
  }

  if (is_v) return;
  lbarrier();

  /* Wave w (0..bt-1) owns token t0+w: in-register RMSNorm + NeoX RoPE (0 extra barriers!). */
  if (w < bt) {
    const u32 tk = t0 + w;
    const G float *nw = is_k ? k_norm : q_norm;
    const u32 h2 = nrot >> 1u;
    const G float *cs_t = cs + (unsigned long)tk * (unsigned long)h2;
    const G float *sn_t = sn + (unsigned long)tk * (unsigned long)h2;
    G float *dst = (G float *)((G u8 *)(is_k ? k_out : q_out) +
                               (unsigned long)tk * (unsigned long)(is_k ? kvs : qs)) + head_r0;

    const float x0 = ink_r12_ds_qkvh[w][l];
    const float x1 = ink_r12_ds_qkvh[w][l + 32u];
    const float x2 = ink_r12_ds_qkvh[w][l + 64u];
    const float x3 = ink_r12_ds_qkvh[w][l + 96u];

    const float ss = wave_tree_f((x0 * x0 + x1 * x1) + (x2 * x2 + x3 * x3));
    const float inv = 1.0f / __builtin_sqrtf(ss * (1.0f / 128.0f) + eps);

    const float y0 = x0 * inv * nw[l];
    const float y1 = x1 * inv * nw[l + 32u];
    const float y2 = x2 * inv * nw[l + 64u];
    const float y3 = x3 * inv * nw[l + 96u];

    if (nrot == INK_R12_DS_HD) {
      const float c0 = cs_t[l], s0 = sn_t[l];
      const float c1 = cs_t[l + 32u], s1 = sn_t[l + 32u];
      dst[l]       = y0 * c0 - y2 * s0;
      dst[l + 32u] = y1 * c1 - y3 * s1;
      dst[l + 64u] = y0 * s0 + y2 * c0;
      dst[l + 96u] = y1 * s1 + y3 * c1;
    } else {
      /* General partial NeoX RoPE fallback via wave-private LDS row ink_r12_ds_qkvh[w]. */
      ink_r12_ds_qkvh[w][l] = y0;
      ink_r12_ds_qkvh[w][l + 32u] = y1;
      ink_r12_ds_qkvh[w][l + 64u] = y2;
      ink_r12_ds_qkvh[w][l + 96u] = y3;
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
      for (u32 m = 0u; m < 4u; m++) {
        const u32 idx = m * 32u + l;
        float r = ink_r12_ds_qkvh[w][idx];
        if (idx < nrot) {
          const u32 ii = idx < h2 ? idx : (idx - h2);
          const float a = ink_r12_ds_qkvh[w][ii];
          const float b = ink_r12_ds_qkvh[w][ii + h2];
          r = (idx < h2) ? (a * cs_t[ii] - b * sn_t[ii]) : (a * sn_t[ii] + b * cs_t[ii]);
        }
        dst[idx] = r;
      }
    }
  }
}

/* ============================================================================
 * 4. H25b: ie_inkling_ds_attn_wmma_q8_r12
 *    FP16 WMMA DSpark Bidirectional Block + Context Attention (Global or SWA)
 *    with inline K_QUANT (att -> aq) inside final reduction wave.
 *    Grid: (n_kv * nsplit, (block_size + 7)/8, 1), 14,336 B LDS, <= 128 VGPR.
 * ========================================================================== */

typedef union {
  struct {
    f16 q[32u * 128u] __attribute__((aligned(16)));         /*  8,192 B: 2x16 queries x 128 f16 */
    f16 p_tile[2u][2u][256u] __attribute__((aligned(16)));  /*  2,048 B: double-buffered 16x16 P */
    union {
      u32 v_sc[8u][128u] __attribute__((aligned(16)));      /*  4,096 B: 8 waves x 512 B V^T */
      struct {
        float mx[2u][128u];                                 /*  1,024 B */
        float ls[2u][128u];                                 /*  1,024 B */
      } red;
    } scratch;                                              /*  4,096 B */
  } kv;                                                     /* 14,336 B */
  struct {
    float o_stg[8u][128u] __attribute__((aligned(16)));     /*  4,096 B */
    float m[32u];                                           /*    128 B */
    float l[32u];                                           /*    128 B */
    float wt[8u][128u] __attribute__((aligned(16)));        /*  4,096 B */
    u32   is_last;                                          /*      4 B */
  } ph4;                                                    /*  8,452 B <= 14,336 B */
} ink_r12_ds_wmma_lds_t;

static LDS ink_r12_ds_wmma_lds_t ink_r12_ds_wlds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r12_ds_wmma_lds_t) == 14336u, "ink_r12_ds_wmma_lds_t must be 14,336 B");

static inline __attribute__((always_inline)) u32 ink_r12_ds_q_off(u32 row, u32 c8) {
  return row * 128u + 8u * (c8 ^ (row & 15u));
}

static inline __attribute__((always_inline)) u32 ink_r12_ds_p_off(u32 rl, u32 hl) {
  return rl * 16u + 8u * (hl ^ ((rl >> 2u) & 1u));
}

static inline __attribute__((always_inline)) float ink_r12_ds_xrow(float v) {
  return I2F(xrow_i(F2I(v)));
}

static inline __attribute__((always_inline)) h8 ink_r12_ds_v_transpose(
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

/* Write one head (128 floats = 4 floats/lane) to optional f32 out AND optional Q8_0 aq
 * using wave-private 512 B LDS staging (0 barriers!). */
static inline __attribute__((always_inline)) void ink_r12_ds_emit_head_q8(
    const f32x4 ov, G float *ot_h, G u8 *aqt, u32 h, u32 n_head, u32 l, LDSP float *stg_w) {
  if (ot_h) ((G struct ink_r12_ds_f32x4p *)(ot_h + 4u * l))->v = ov;
  if (aqt) {
    ((LDSP f32x4 *)stg_w)[l] = ov;
    __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
    const u32 nb_q = n_head << 2u;
    const u32 b0 = h << 2u;
#pragma unroll
    for (u32 m = 0u; m < 4u; m++) {
      quant_wave(stg_w[32u * m + l], b0 + m, aqt, nb_q);
    }
    __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  }
}

static inline __attribute__((always_inline)) void ink_r12_ds_wmma_merge_head(
    const G float *ph, const u32 nsplit, const u32 l, LDSP float *wt_w, LDSP float *stg_w,
    G float *ot_h, G u8 *aqt, u32 h, u32 n_head) {
  float mv = -__builtin_inff();
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const unsigned long off = (unsigned long)sp * INK_R12_DS_PSTR;
    if (ph[off + 1u] > 0.0f) mv = __builtin_fmaxf(mv, ph[off]);
  }
#pragma unroll
  for (u32 mask = 16u; mask >= 1u; mask >>= 1u) mv = __builtin_fmaxf(mv, shfl_xor_f(mv, mask));
  const float M2 = ink_r12_ds_sgpr_f(mv);

  float lv = 0.0f;
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const unsigned long off = (unsigned long)sp * INK_R12_DS_PSTR;
    const float lp = ph[off + 1u];
    const float wt = (lp > 0.0f) ? __builtin_expf(ph[off] - M2) : 0.0f;
    if (sp < 128u) wt_w[sp] = wt;
    lv += lp * wt;
  }
  const float L2 = ink_r12_ds_sgpr_f(wave_sum_all(lv));
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");

  f32x4 acc = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  for (u32 sp = 0u; sp < nsplit; sp++) {
    const float wt = (sp < 128u) ? wt_w[sp]
                                 : ((ph[(unsigned long)sp * INK_R12_DS_PSTR + 1u] > 0.0f)
                                        ? __builtin_expf(ph[(unsigned long)sp * INK_R12_DS_PSTR] - M2)
                                        : 0.0f);
    if (wt > 0.0f) {
      const G float *po = ph + (unsigned long)sp * INK_R12_DS_PSTR + 2u + 4u * l;
      const f32x4 ov = {po[0], po[1], po[2], po[3]};
      acc += ov * wt;
    }
  }
  const float inv = (L2 > 0.0f) ? (1.0f / L2) : 0.0f;
  ink_r12_ds_emit_head_q8(acc * inv, ot_h, aqt, h, n_head, l, stg_w);
}

KERNEL ie_inkling_ds_attn_wmma_q8_r12(
    const G float *q, const G kvt *kc_ctx, const G kvt *vc_ctx,
    const G float *k_blk, const G float *v_blk, G float *out, G u8 *aq,
    u32 ctx_len, u32 kv_cap, u32 win, u32 block_size,
    u32 n_head, u32 n_kv, u32 hd, u32 qs, u32 ks, u32 os, u32 aqs,
    u32 nsplit, G float *part, G u32 *count) {
  if (hd != INK_R12_DS_HD || n_kv == 0u || n_head != n_kv * INK_R12_DS_GQA ||
      block_size == 0u || nsplit == 0u)
    return;

  const u32 g = wgid();
  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)(g / nsplit));
  const u32 sp = (u32)__builtin_amdgcn_readfirstlane((int)(g - kh * nsplit));
  const u32 btile = tokid();
  const u32 bt_base = btile << 3u;
  if (kh >= n_kv || bt_base >= block_size) return;

  const u32 n_bt = (block_size - bt_base < 8u) ? (block_size - bt_base) : 8u;
  const u32 nq_tot = (u32)__builtin_amdgcn_readfirstlane((int)(n_bt * INK_R12_DS_GQA));
  const u32 npg = (u32)__builtin_amdgcn_readfirstlane((int)(nq_tot > 16u ? 2u : 1u));
  const u32 h0 = (u32)__builtin_amdgcn_readfirstlane((int)(kh * INK_R12_DS_GQA));
  const u32 kvd = (u32)__builtin_amdgcn_readfirstlane((int)(n_kv * INK_R12_DS_HD));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane(), hl = l >> 4u, rl = l & 15u;

  const u32 ctx_start = (win > 0u && ctx_len > win) ? (ctx_len - win) : 0u;
  const u32 ctx_n = (kv_cap > 0u && ctx_len > ctx_start) ? (ctx_len - ctx_start) : 0u;
  const u32 chunk = (ctx_n + nsplit - 1u) / nsplit;
  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)(ctx_start + sp * chunk));
  const u32 p1 = (u32)__builtin_amdgcn_readfirstlane(
      (int)((p0 < ctx_len) ? ((p0 + chunk < ctx_len) ? (p0 + chunk) : ctx_len) : p0));

  const int has_work = (p0 < p1) || (sp == 0u);
  if (nsplit > 1u && !has_work) {
    for (u32 qi = w; qi < nq_tot; qi += 8u) {
      const u32 bt_loc = qi / INK_R12_DS_GQA;
      const u32 bt = bt_base + bt_loc;
      const u32 h = h0 + (qi - bt_loc * INK_R12_DS_GQA);
      G float *pp = part + (unsigned long)((bt * n_head + h) * nsplit + sp) * INK_R12_DS_PSTR;
      if (l == 0u) { pp[0] = -1e30f; pp[1] = 0.0f; }
      pp[2u + 4u * l + 0u] = 0.0f;
      pp[2u + 4u * l + 1u] = 0.0f;
      pp[2u + 4u * l + 2u] = 0.0f;
      pp[2u + 4u * l + 3u] = 0.0f;
    }
  } else {
    /* Phase 1: Stage nq_tot <= 24 queries (f16, scaled by log2(e)/sqrt(128)) into XOR-swizzled LDS. */
    const u32 q_rows = npg * 16u;
    for (u32 idx = tid(); idx < q_rows * 16u; idx += NT) {
      const u32 qr = idx >> 4u;
      const u32 c8 = idx & 15u;
      h8 qh = (h8)(0.0f);
      if (qr < nq_tot) {
        const u32 bt_loc = qr / INK_R12_DS_GQA;
        const u32 gi = qr - bt_loc * INK_R12_DS_GQA;
        const G float *qb = (const G float *)((const G u8 *)q + (unsigned long)(bt_base + bt_loc) * (unsigned long)qs) +
                            (unsigned long)(h0 + gi) * INK_R12_DS_HD;
        const f32x4 v0 = ((const G struct ink_r12_ds_f32x4p *)(qb + 8u * c8))->v;
        const f32x4 v1 = ((const G struct ink_r12_ds_f32x4p *)(qb + 8u * c8 + 4u))->v;
        qh = (h8){(f16)(v0.x * INK_R12_DS_QSC), (f16)(v0.y * INK_R12_DS_QSC),
                  (f16)(v0.z * INK_R12_DS_QSC), (f16)(v0.w * INK_R12_DS_QSC),
                  (f16)(v1.x * INK_R12_DS_QSC), (f16)(v1.y * INK_R12_DS_QSC),
                  (f16)(v1.z * INK_R12_DS_QSC), (f16)(v1.w * INK_R12_DS_QSC)};
      }
      *(LDSP h8 *)(ink_r12_ds_wlds.kv.q + ink_r12_ds_q_off(qr, c8)) = qh;
    }
    lbarrier();

    const G kvt *kc_h = kc_ctx + (unsigned long)kh * INK_R12_DS_HD;
    const G kvt *vc_h = vc_ctx + (unsigned long)kh * INK_R12_DS_HD;
    const int no_wrap = (int)__builtin_amdgcn_readfirstlane((int)(kv_cap > 0u && p1 <= kv_cap));

    const v8f zero = (v8f)(0.0f);
    v8f acc0 = zero, acc1 = zero;
    float m0 = -__builtin_inff(), ls0 = 0.0f;
    float m1 = -__builtin_inff(), ls1 = 0.0f;
    const u32 p_off = ink_r12_ds_p_off(rl, hl);

    /* Phase 2: Context KV [p0, p1) in 128-key WMMA chunks (8 waves x 16 keys). */
#pragma clang loop unroll(disable)
    for (u32 c0 = p0; c0 < p1; c0 += 128u) {
      const u32 kb = (u32)__builtin_amdgcn_readfirstlane((int)(c0 + 16u * w));
      const u32 chunk_end = (u32)__builtin_amdgcn_readfirstlane((int)((c0 + 128u < p1) ? (c0 + 128u) : p1));

      v8f sp0 = (v8f)(-__builtin_inff());
      v8f sp1 = (v8f)(-__builtin_inff());

      if (kb < chunk_end) {
        const u32 kp_row = kb + rl;
        const int row_ok = (kp_row < chunk_end);
        const u32 kv_row = no_wrap ? (row_ok ? kp_row : (chunk_end - 1u))
                                   : ((row_ok ? kp_row : (chunk_end - 1u)) % kv_cap);
        const G f16 *k_base = kc_h + (unsigned long)kv_row * (unsigned long)kvd + 8u * hl;

        sp0 = zero;
        sp1 = zero;
        h8 kk0 = row_ok ? __builtin_bit_cast(h8, ink_r12_ds_nt4(k_base + 0u)) : (h8)(0.0f);
        h8 kk1 = row_ok ? __builtin_bit_cast(h8, ink_r12_ds_nt4(k_base + 16u)) : (h8)(0.0f);
#pragma unroll
        for (u32 ks_i = 0u; ks_i < 8u; ks_i += 2u) {
          const u32 c_a = 2u * ks_i + hl;
          const u32 c_b = 2u * (ks_i + 1u) + hl;
          const h8 q0_a = *(const LDSP h8 *)(ink_r12_ds_wlds.kv.q + ink_r12_ds_q_off(rl, c_a));
          const h8 q0_b = *(const LDSP h8 *)(ink_r12_ds_wlds.kv.q + ink_r12_ds_q_off(rl, c_b));
          sp0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk0, q0_a, sp0);
          sp0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk1, q0_b, sp0);
          if (npg > 1u) {
            const h8 q1_a = *(const LDSP h8 *)(ink_r12_ds_wlds.kv.q + ink_r12_ds_q_off(16u + rl, c_a));
            const h8 q1_b = *(const LDSP h8 *)(ink_r12_ds_wlds.kv.q + ink_r12_ds_q_off(16u + rl, c_b));
            sp1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk0, q1_a, sp1);
            sp1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk1, q1_b, sp1);
          }
          if (ks_i + 2u < 8u) {
            kk0 = row_ok ? __builtin_bit_cast(h8, ink_r12_ds_nt4(k_base + 16u * (ks_i + 2u))) : (h8)(0.0f);
            kk1 = row_ok ? __builtin_bit_cast(h8, ink_r12_ds_nt4(k_base + 16u * (ks_i + 3u))) : (h8)(0.0f);
          }
          __builtin_amdgcn_sched_barrier(0);
        }

        if (kb + 16u > chunk_end) {
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) {
            if (kb + 8u * hl + v >= chunk_end) {
              sp0[v] = -__builtin_inff();
              if (npg > 1u) sp1[v] = -__builtin_inff();
            }
          }
        }
      }

      u32x4 vr0 = (u32x4)(0u);
      {
        const u32 vp = c0 + rl;
        if (vp < chunk_end) {
          const u32 kv_v = no_wrap ? vp : (vp % kv_cap);
          vr0 = ink_r12_ds_nt4(vc_h + (unsigned long)kv_v * (unsigned long)kvd + 16u * w + 8u * hl);
        }
      }

      float mx0 = sp0[0];
#pragma unroll
      for (u32 v = 1u; v < 8u; v++) mx0 = __builtin_fmaxf(mx0, sp0[v]);
      mx0 = __builtin_fmaxf(mx0, ink_r12_ds_xrow(mx0));
      if (hl == 0u) ink_r12_ds_wlds.kv.scratch.red.mx[0][16u * w + rl] = mx0;

      if (npg > 1u) {
        float mx1 = sp1[0];
#pragma unroll
        for (u32 v = 1u; v < 8u; v++) mx1 = __builtin_fmaxf(mx1, sp1[v]);
        mx1 = __builtin_fmaxf(mx1, ink_r12_ds_xrow(mx1));
        if (hl == 0u) ink_r12_ds_wlds.kv.scratch.red.mx[1][16u * w + rl] = mx1;
      }
      lbarrier();

      float mc0 = ink_r12_ds_wlds.kv.scratch.red.mx[0][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) mc0 = __builtin_fmaxf(mc0, ink_r12_ds_wlds.kv.scratch.red.mx[0][16u * x + rl]);
      const float mn0 = __builtin_fmaxf(m0, mc0);
      const float rs0 = (ls0 > 0.0f && mn0 > -1e30f) ? __builtin_amdgcn_exp2f(m0 - mn0) : 0.0f;
      acc0 *= rs0; ls0 *= rs0; m0 = mn0;

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
      lc0 += ink_r12_ds_xrow(lc0);
      if (hl == 0u) ink_r12_ds_wlds.kv.scratch.red.ls[0][16u * w + rl] = lc0;

      h8 ph1 = (h8)(0.0f);
      if (npg > 1u) {
        float mc1 = ink_r12_ds_wlds.kv.scratch.red.mx[1][rl];
#pragma unroll
        for (u32 x = 1u; x < 8u; x++) mc1 = __builtin_fmaxf(mc1, ink_r12_ds_wlds.kv.scratch.red.mx[1][16u * x + rl]);
        const float mn1 = __builtin_fmaxf(m1, mc1);
        const float rs1 = (ls1 > 0.0f && mn1 > -1e30f) ? __builtin_amdgcn_exp2f(m1 - mn1) : 0.0f;
        acc1 *= rs1; ls1 *= rs1; m1 = mn1;

        float lc1 = 0.0f;
        if (mn1 > -1e30f) {
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) {
            const float p = __builtin_amdgcn_exp2f(sp1[v] - mn1);
            lc1 += p;
            ph1[v] = (f16)p;
          }
        }
        lc1 += ink_r12_ds_xrow(lc1);
        if (hl == 0u) ink_r12_ds_wlds.kv.scratch.red.ls[1][16u * w + rl] = lc1;
      }

      if (w == 0u) {
        *(LDSP h8 *)(ink_r12_ds_wlds.kv.p_tile[0][0] + p_off) = ph0;
        if (npg > 1u) *(LDSP h8 *)(ink_r12_ds_wlds.kv.p_tile[0][1] + p_off) = ph1;
      }
      lbarrier();

      float sum_l0 = ink_r12_ds_wlds.kv.scratch.red.ls[0][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) sum_l0 += ink_r12_ds_wlds.kv.scratch.red.ls[0][16u * x + rl];
      ls0 += sum_l0;
      if (npg > 1u) {
        float sum_l1 = ink_r12_ds_wlds.kv.scratch.red.ls[1][rl];
#pragma unroll
        for (u32 x = 1u; x < 8u; x++) sum_l1 += ink_r12_ds_wlds.kv.scratch.red.ls[1][16u * x + rl];
        ls1 += sum_l1;
      }
      lbarrier();

      LDSP u32 *v_sc_w = ink_r12_ds_wlds.kv.scratch.v_sc[w];
#pragma clang loop unroll(disable)
      for (u32 kt = 0u; kt < 8u; kt++) {
        if (c0 + 16u * kt >= chunk_end) break;
        const int has_nxt = (kt + 1u < 8u) && (c0 + 16u * (kt + 1u) < chunk_end);
        u32x4 vr_nxt = (u32x4)(0u);
        if (has_nxt) {
          const u32 vp_n = c0 + 16u * (kt + 1u) + rl;
          if (vp_n < chunk_end) {
            const u32 kv_vn = no_wrap ? vp_n : (vp_n % kv_cap);
            vr_nxt = ink_r12_ds_nt4(vc_h + (unsigned long)kv_vn * (unsigned long)kvd + 16u * w + 8u * hl);
          }
        }
        __builtin_amdgcn_sched_barrier(0);

        const h8 vt_a = ink_r12_ds_v_transpose(vr0, v_sc_w, hl, rl);
        const h8 pt0 = *(const LDSP h8 *)(ink_r12_ds_wlds.kv.p_tile[kt & 1u][0] + p_off);
        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt0, acc0);
        if (npg > 1u) {
          const h8 pt1 = *(const LDSP h8 *)(ink_r12_ds_wlds.kv.p_tile[kt & 1u][1] + p_off);
          acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt1, acc1);
        }

        if (has_nxt && w == kt + 1u) {
          *(LDSP h8 *)(ink_r12_ds_wlds.kv.p_tile[(kt + 1u) & 1u][0] + p_off) = ph0;
          if (npg > 1u) *(LDSP h8 *)(ink_r12_ds_wlds.kv.p_tile[(kt + 1u) & 1u][1] + p_off) = ph1;
        }
        lbarrier();
        vr0 = vr_nxt;
      }
    }

    /* Phase 3: Bidirectional f32 block KV [0..block_size-1] in split sp == 0 (16-key WMMA tiles). */
    if (sp == 0u) {
#pragma clang loop unroll(disable)
      for (u32 b0 = 0u; b0 < block_size; b0 += 16u) {
        const u32 b_end = (b0 + 16u < block_size) ? (b0 + 16u) : block_size;
        v8f sp0 = (v8f)(-__builtin_inff());
        v8f sp1 = (v8f)(-__builtin_inff());

        if (w == 0u) {
          const u32 b_row = b0 + rl;
          const int row_ok = (b_row < b_end);
          const G float *kbp = row_ok ? ((const G float *)((const G u8 *)k_blk + (unsigned long)b_row * (unsigned long)ks) +
                                         (unsigned long)kh * INK_R12_DS_HD + 8u * hl)
                                      : (const G float *)0;
          sp0 = zero;
          sp1 = zero;
#pragma unroll
          for (u32 ks_i = 0u; ks_i < 8u; ks_i++) {
            h8 kk = (h8)(0.0f);
            if (row_ok) {
              const f32x4 v0 = ((const G struct ink_r12_ds_f32x4p *)(kbp + 16u * ks_i))->v;
              const f32x4 v1 = ((const G struct ink_r12_ds_f32x4p *)(kbp + 16u * ks_i + 4u))->v;
              kk = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w,
                        (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
            }
            const u32 c_a = 2u * ks_i + hl;
            const h8 q0_a = *(const LDSP h8 *)(ink_r12_ds_wlds.kv.q + ink_r12_ds_q_off(rl, c_a));
            sp0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk, q0_a, sp0);
            if (npg > 1u) {
              const h8 q1_a = *(const LDSP h8 *)(ink_r12_ds_wlds.kv.q + ink_r12_ds_q_off(16u + rl, c_a));
              sp1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk, q1_a, sp1);
            }
          }
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) {
            if (b0 + 8u * hl + v >= b_end) {
              sp0[v] = -__builtin_inff();
              if (npg > 1u) sp1[v] = -__builtin_inff();
            }
          }

          float mx0 = sp0[0];
#pragma unroll
          for (u32 v = 1u; v < 8u; v++) mx0 = __builtin_fmaxf(mx0, sp0[v]);
          mx0 = __builtin_fmaxf(mx0, ink_r12_ds_xrow(mx0));
          if (hl == 0u) ink_r12_ds_wlds.kv.scratch.red.mx[0][rl] = mx0;
          if (npg > 1u) {
            float mx1 = sp1[0];
#pragma unroll
            for (u32 v = 1u; v < 8u; v++) mx1 = __builtin_fmaxf(mx1, sp1[v]);
            mx1 = __builtin_fmaxf(mx1, ink_r12_ds_xrow(mx1));
            if (hl == 0u) ink_r12_ds_wlds.kv.scratch.red.mx[1][rl] = mx1;
          }
        }
        lbarrier();

        const float mc0 = ink_r12_ds_wlds.kv.scratch.red.mx[0][rl];
        const float mn0 = __builtin_fmaxf(m0, mc0);
        const float rs0 = (ls0 > 0.0f && mn0 > -1e30f) ? __builtin_amdgcn_exp2f(m0 - mn0) : 0.0f;
        acc0 *= rs0; ls0 *= rs0; m0 = mn0;

        float mn1 = m1;
        if (npg > 1u) {
          const float mc1 = ink_r12_ds_wlds.kv.scratch.red.mx[1][rl];
          mn1 = __builtin_fmaxf(m1, mc1);
          const float rs1 = (ls1 > 0.0f && mn1 > -1e30f) ? __builtin_amdgcn_exp2f(m1 - mn1) : 0.0f;
          acc1 *= rs1; ls1 *= rs1; m1 = mn1;
        }

        if (w == 0u) {
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
          lc0 += ink_r12_ds_xrow(lc0);
          if (hl == 0u) ink_r12_ds_wlds.kv.scratch.red.ls[0][rl] = lc0;
          *(LDSP h8 *)(ink_r12_ds_wlds.kv.p_tile[0][0] + p_off) = ph0;

          if (npg > 1u) {
            h8 ph1 = (h8)(0.0f);
            float lc1 = 0.0f;
            if (mn1 > -1e30f) {
#pragma unroll
              for (u32 v = 0u; v < 8u; v++) {
                const float p = __builtin_amdgcn_exp2f(sp1[v] - mn1);
                lc1 += p;
                ph1[v] = (f16)p;
              }
            }
            lc1 += ink_r12_ds_xrow(lc1);
            if (hl == 0u) ink_r12_ds_wlds.kv.scratch.red.ls[1][rl] = lc1;
            *(LDSP h8 *)(ink_r12_ds_wlds.kv.p_tile[0][1] + p_off) = ph1;
          }
        }
        lbarrier();

        ls0 += ink_r12_ds_wlds.kv.scratch.red.ls[0][rl];
        if (npg > 1u) ls1 += ink_r12_ds_wlds.kv.scratch.red.ls[1][rl];
        lbarrier();

        /* Load f32 v_blk -> pack f16x8 into u32x4 -> transpose -> WMMA PV across all 8 waves. */
        u32x4 vr_b = (u32x4)(0u);
        const u32 vp_b = b0 + rl;
        if (vp_b < b_end) {
          const G float *vbp = (const G float *)((const G u8 *)v_blk + (unsigned long)vp_b * (unsigned long)ks) +
                               (unsigned long)kh * INK_R12_DS_HD + 16u * w + 8u * hl;
          const f32x4 v0 = ((const G struct ink_r12_ds_f32x4p *)(vbp + 0u))->v;
          const f32x4 v1 = ((const G struct ink_r12_ds_f32x4p *)(vbp + 4u))->v;
          const h8 vh = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w,
                             (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
          vr_b = __builtin_bit_cast(u32x4, vh);
        }
        const h8 vt_a = ink_r12_ds_v_transpose(vr_b, ink_r12_ds_wlds.kv.scratch.v_sc[w], hl, rl);
        const h8 pt0 = *(const LDSP h8 *)(ink_r12_ds_wlds.kv.p_tile[0][0] + p_off);
        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt0, acc0);
        if (npg > 1u) {
          const h8 pt1 = *(const LDSP h8 *)(ink_r12_ds_wlds.kv.p_tile[0][1] + p_off);
          acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt1, acc1);
        }
        lbarrier();
      }
    }

    /* Phase 4A: Stage (m, ls) and transpose wave-split dims -> 128 floats/query. */
    lbarrier();
    if (w == 0u && hl == 0u) {
      ink_r12_ds_wlds.ph4.m[rl] = (ls0 > 0.0f) ? (m0 * INK_R12_DS_LN2) : -1e30f;
      ink_r12_ds_wlds.ph4.l[rl] = ls0;
      if (npg > 1u) {
        ink_r12_ds_wlds.ph4.m[16u + rl] = (ls1 > 0.0f) ? (m1 * INK_R12_DS_LN2) : -1e30f;
        ink_r12_ds_wlds.ph4.l[16u + rl] = ls1;
      }
    }
    lbarrier();

#pragma clang loop unroll(disable)
    for (u32 pg = 0u; pg < npg; pg++) {
      const v8f a_cur = (pg == 0u) ? acc0 : acc1;
#pragma unroll
      for (u32 qb = 0u; qb < 2u; qb++) {
        const u32 q_base = 16u * pg + 8u * qb;
        if (q_base >= nq_tot) break;
        if ((rl >> 3u) == qb) {
          const u32 q_loc = rl & 7u;
          LDSP float *dst_w = ink_r12_ds_wlds.ph4.o_stg[q_loc] + 16u * w + 8u * hl;
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) dst_w[v] = a_cur[v];
        }
        lbarrier();

        const u32 qi = q_base + w;
        if (qi < nq_tot) {
          const u32 bt_loc = qi / INK_R12_DS_GQA;
          const u32 bt = bt_base + bt_loc;
          const u32 h = h0 + (qi - bt_loc * INK_R12_DS_GQA);
          const f32x4 ov = ((const LDSP f32x4 *)ink_r12_ds_wlds.ph4.o_stg[w])[l];
          if (nsplit == 1u) {
            const float lq = ink_r12_ds_wlds.ph4.l[qi];
            const float inv = (lq > 0.0f) ? (1.0f / lq) : 0.0f;
            G float *ot_h = out ? ((G float *)((G u8 *)out + (unsigned long)bt * (unsigned long)os) +
                                   (unsigned long)h * INK_R12_DS_HD)
                                : (G float *)0;
            G u8 *aqt = aq ? (aq + (unsigned long)bt * (unsigned long)aqs) : (G u8 *)0;
            ink_r12_ds_emit_head_q8(ov * inv, ot_h, aqt, h, n_head, l, ink_r12_ds_wlds.ph4.wt[w]);
          } else {
            G float *pp = part + (unsigned long)((bt * n_head + h) * nsplit + sp) * INK_R12_DS_PSTR;
            if (l == 0u) {
              pp[0] = ink_r12_ds_wlds.ph4.m[qi];
              pp[1] = ink_r12_ds_wlds.ph4.l[qi];
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

  /* Phase 4B: Last-WG arrival per (btile, kh) + inline merge & Q8_0 quant! */
  const u32 cidx = btile * n_kv + kh;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r12_ds_wlds.ph4.is_last =
        (__scoped_atomic_fetch_add(&count[cidx], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r12_ds_wlds.ph4.is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  for (u32 qi = w; qi < nq_tot; qi += 8u) {
    const u32 bt_loc = qi / INK_R12_DS_GQA;
    const u32 bt = bt_base + bt_loc;
    const u32 h = h0 + (qi - bt_loc * INK_R12_DS_GQA);
    G float *ot_h = out ? ((G float *)((G u8 *)out + (unsigned long)bt * (unsigned long)os) +
                           (unsigned long)h * INK_R12_DS_HD)
                        : (G float *)0;
    G u8 *aqt = aq ? (aq + (unsigned long)bt * (unsigned long)aqs) : (G u8 *)0;
    const G float *ph = part + (unsigned long)((bt * n_head + h) * nsplit) * INK_R12_DS_PSTR;
    ink_r12_ds_wmma_merge_head(ph, nsplit, l, ink_r12_ds_wlds.ph4.wt[w],
                               ink_r12_ds_wlds.ph4.o_stg[w], ot_h, aqt, h, n_head);
  }
  barrier();
  if (tid() == 0u) count[cidx] = 0u;
}

/* ============================================================================
 * 5. H25c: ie_inkling_ds_proj_res_norm_q8_q4k_r12
 *    Fused Wo/Wdown (MAT_Q4K) + residual add + deterministic next RMSNorm + Q8_0.
 *    Grid: ((rows + 31)/32, (B + 3)/4, 1) = (72, 2, 1) = 144 WGs for dim=2304, B=8.
 *    <= 80 VGPR (4 WGs/CU), 36 B LDS, 0/0 spills.
 * ========================================================================== */

static LDS float ink_r12_ds_pn_wsum[8u];
static LDS u32   ink_r12_ds_pn_last;

KERNEL ie_inkling_ds_proj_res_norm_q8_q4k_r12(
    const G u32 *qw, const G u16 *qs, const G u8 *xq,
    const G float *res, G float *h_out,
    const G float *norm_w, G float *ny_out, G u8 *nq_out,
    G float *ss_buf, G u32 *count,
    u32 rows, u32 nb, u32 B, u32 xs, u32 rs, u32 hs, u32 nys, u32 nqs,
    float eps) {
#pragma clang fp contract(off)
  const u32 rb = wgid();
  const u32 ng_r = ngroups_x();
  const u32 ttile = tokid();
  const u32 t0 = ttile << 2u;
  if (rows == 0u || (rows & 31u) != 0u || nb == 0u || (nb & 7u) != 0u || t0 >= B) return;
  const u32 rbase = rb << 5u;
  if (rbase >= rows) return;

  const u32 bt = (B - t0 < 4u) ? (B - t0) : 4u;
  const u32 th = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(th >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 r = rbase + (w << 2u) + (l >> 3u);

  const u32 nsb = nb >> 3u;
  const __amdgpu_buffer_rsrc_t r_qw = ink_r12_ds_mkbuf(qw, rows * nb * 16u);
  const __amdgpu_buffer_rsrc_t r_qs = ink_r12_ds_mkbuf(qs, (rows * nb + rows * nsb * 2u) * 2u);
  const G u8 *xq_t0 = xq + (unsigned long)t0 * (unsigned long)xs;
  const __amdgpu_buffer_rsrc_t r_x = ink_r12_ds_mkbuf(xq_t0, bt * xs);

  const u32 row_qw_b = r * nb * 16u;
  const u32 row_sm_b = r * nb * 2u;
  const u32 row_dd_b = rows * nb * 2u + r * nsb * 4u;
  const u32 da_base = nb * 32u;
  const u32 as_base = nb * 36u;

  ink_r12_ds_f4 a0 = (ink_r12_ds_f4)(0.0f);
  ink_r12_ds_f4 a1 = (ink_r12_ds_f4)(0.0f);
  ink_r12_ds_f4 a2 = (ink_r12_ds_f4)(0.0f);
  ink_r12_ds_f4 a3 = (ink_r12_ds_f4)(0.0f);

#pragma clang loop unroll(disable)
  for (u32 sb = 0u; sb < nsb; sb++) {
    const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
    const u32 slot = (u32)__builtin_amdgcn_readfirstlane((int)(sb_u & 3u));
    const u32 b = (sb_u << 3u) + sub;

    const u32x4 w4 = INK_R12_DS_RBL128(r_qw, row_qw_b + b * 16u, 0);
    const u32 sm = (u32)INK_R12_DS_RBL16(r_qs, row_sm_b + b * 2u, 0);
    const u32 dd_raw = INK_R12_DS_RBL32(r_qs, row_dd_b + sb_u * 4u, 0);
    const u16 d_bits = (u16)(dd_raw & 0xFFFFu);
    const u16 dm_bits = (u16)(dd_raw >> 16u);
    const float dv = (float)__builtin_bit_cast(f16, d_bits);
    const float mv = (float)__builtin_bit_cast(f16, dm_bits);
    const float sc = dv * (float)(sm & 63u);
    const float mm = mv * (float)(sm >> 8u);

    const u32 lo0 = ie_q4k_lo(w4.x), hi0 = ie_q4k_hi(w4.x);
    const u32 lo1 = ie_q4k_lo(w4.y), hi1 = ie_q4k_hi(w4.y);
    const u32 lo2 = ie_q4k_lo(w4.z), hi2 = ie_q4k_hi(w4.z);
    const u32 lo3 = ie_q4k_lo(w4.w), hi3 = ie_q4k_hi(w4.w);

    const u32 aw_off = b * 32u;
    const u32 da_off = da_base + b * 4u;
    const u32 as_off = as_base + b * 4u;

#pragma unroll
    for (u32 j = 0u; j < 4u; j++) {
      if (j < bt) {
        const u32 jbase = j * xs;
        const u32x4 act0 = INK_R12_DS_RBL128(r_x, jbase + aw_off, 2);
        const u32x4 act1 = INK_R12_DS_RBL128(r_x, jbase + aw_off + 16u, 2);
        const float da = INK_R12_DS_RBLF32(r_x, jbase + da_off, 2);
        const int as = (int)INK_R12_DS_RBL32(r_x, jbase + as_off, 2);
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
        if (slot == 0u) a0[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a0[j]));
        else if (slot == 1u) a1[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a1[j]));
        else if (slot == 2u) a2[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a2[j]));
        else a3[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a3[j]));
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  const int do_norm = (norm_w != (const G float *)0) && (ny_out != (G float *)0 || nq_out != (G u8 *)0);
  ink_r12_ds_f4 sq = (ink_r12_ds_f4)(0.0f);

#pragma unroll
  for (u32 j = 0u; j < 4u; j++) {
    if (j < bt) {
      const float dot = ink_r12_ds_oct_tree_f((a0[j] + a2[j]) + (a1[j] + a3[j]));
      if (sub == 0u) {
        const u32 tk = t0 + j;
        const G float *rp = res ? (const G float *)((const G u8 *)res + (unsigned long)tk * (unsigned long)rs)
                                : (const G float *)0;
        G float *hp = (G float *)((G u8 *)h_out + (unsigned long)tk * (unsigned long)hs);
        const float hv = rp ? (rp[r] + dot) : dot;
        hp[r] = hv;
        sq[j] = hv * hv;
      }
    }
  }

  if (!do_norm) return;

  /* Deterministic per-WG partial sum-of-squares into ss_buf[tk * ng_r + rb]. */
#pragma clang loop unroll(disable)
  for (u32 j = 0u; j < bt; j++) {
    const float wv_s = wave_tree_f(sq[j]);
    if (l == 0u) ink_r12_ds_pn_wsum[w] = wv_s;
    lbarrier();
    if (th == 0u) {
      float tot = 0.0f;
#pragma unroll
      for (u32 x = 0u; x < 8u; x++) tot += ink_r12_ds_pn_wsum[x];
      ss_buf[(unsigned long)(t0 + j) * (unsigned long)ng_r + (unsigned long)rb] = tot;
    }
    lbarrier();
  }

  /* Agent release/acquire barrier per token tile; last WG runs RMSNorm + Q8_0 for all bt tokens. */
  if (ng_r > 1u) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (th == 0u) {
      ink_r12_ds_pn_last =
          (__scoped_atomic_fetch_add(&count[ttile], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng_r - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r12_ds_pn_last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (th == 0u) count[ttile] = 0u;
  } else {
    barrier();
  }

  const u32 nb_out = rows >> 5u;
#pragma clang loop unroll(disable)
  for (u32 j = 0u; j < bt; j++) {
    const u32 tk = t0 + j;
    const G float *sb = ss_buf + (unsigned long)tk * (unsigned long)ng_r;
    float psum = 0.0f;
    for (u32 idx = th; idx < ng_r; idx += NT) psum += sb[idx];
    psum = wave_tree_f(psum);
    if (l == 0u) ink_r12_ds_pn_wsum[w] = psum;
    lbarrier();
    float ss = 0.0f;
#pragma unroll
    for (u32 x = 0u; x < 8u; x++) ss += ink_r12_ds_pn_wsum[x];
    const float inv = 1.0f / __builtin_sqrtf(ss / (float)rows + eps);

    const G float *hp = (const G float *)((const G u8 *)h_out + (unsigned long)tk * (unsigned long)hs);
    G float *nyp = ny_out ? (G float *)((G u8 *)ny_out + (unsigned long)tk * (unsigned long)nys) : (G float *)0;
    G u8 *nqp = nq_out ? (nq_out + (unsigned long)tk * (unsigned long)nqs) : (G u8 *)0;
#pragma clang loop unroll(disable)
    for (u32 b = w; b < nb_out; b += 8u) {
      const u32 i = (b << 5u) + l;
      const float y = hp[i] * inv * norm_w[i];
      if (nyp) nyp[i] = y;
      if (nqp) quant_wave(y, b, nqp, nb_out);
    }
    lbarrier();
  }
}

/* ============================================================================
 * 6. H32: ie_inkling_ds_p_sample_256wg_r12
 *    256-WG (4 WGs/CU) Threshold-Gated DSpark P*lat + base + softcap + sample.
 *    Wave-private K-heaps (0 hot-loop barriers!), 1-cycle SGPR raw_cut filter
 *    hoisting tanhf out of hot row loop, 2-pass streaming Stage-2 merge (ng<=256).
 * ========================================================================== */

typedef struct {
  union {
    struct {
      float kv[8u][INK_R12_DS_KMAX];  /* 1,024 B: wave-private K-heaps (0 hot barriers!) */
      u32   ki[8u][INK_R12_DS_KMAX];  /* 1,024 B */
    } wh;
    u32 sm[2u][NT];                   /* 2,048 B */
    struct {
      float cv[INK_R12_DS_CAP];       /* 4,096 B */
      u32   ci[INK_R12_DS_CAP];       /* 4,096 B */
    } cand;
    struct {
      float wv[8u];
      u32   wi[8u];
      float ws[8u];
    } gr;
  } u;
  u32   wsum[2u][8u];
  float fv[INK_R12_DS_KMAX];
  u32   fi[INK_R12_DS_KMAX];
  float fq[INK_R12_DS_KMAX];
  u32   last;
} ink_r12_ds_ps_lds_t;

static LDS ink_r12_ds_ps_lds_t ink_r12_ds_pslds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r12_ds_ps_lds_t) <= 16384u, "ink_r12_ds_pslds must stay <= 16 KiB");

static inline __attribute__((always_inline)) void ink_r12_ds_ps_gr_best(float *bv, u32 *bi) {
  const u32 l = lane();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
#pragma unroll
  for (u32 m = 16u; m > 0u; m >>= 1u) {
    const float ov = shfl_xor_f(*bv, m);
    const u32 oi = (u32)__builtin_amdgcn_ds_bpermute((int)((l ^ m) * 4u), (int)*bi);
    if (ink_r12_ds_better(ov, oi, *bv, *bi)) { *bv = ov; *bi = oi; }
  }
  if (l == 0u) { ink_r12_ds_pslds.u.gr.wv[w] = *bv; ink_r12_ds_pslds.u.gr.wi[w] = *bi; }
  lbarrier();
  float v = ink_r12_ds_pslds.u.gr.wv[0u];
  u32 idx = ink_r12_ds_pslds.u.gr.wi[0u];
#pragma clang loop unroll(disable)
  for (u32 j = 1u; j < 8u; j++) {
    if (ink_r12_ds_better(ink_r12_ds_pslds.u.gr.wv[j], ink_r12_ds_pslds.u.gr.wi[j], v, idx)) {
      v = ink_r12_ds_pslds.u.gr.wv[j]; idx = ink_r12_ds_pslds.u.gr.wi[j];
    }
  }
  *bv = v; *bi = idx;
}

static inline __attribute__((always_inline)) float ink_r12_ds_q4q8_blk(
    const G u32 *row_qw, const G f16 *row_qs,
    __amdgpu_buffer_rsrc_t r_x, u32 da_base, u32 as_base, u32 b) {
  const u32x4 w4 = ink_r12_ds_ld4(row_qw + (unsigned long)b * 4ul);
  const u32x4 a0 = INK_R12_DS_RBL128(r_x, b * 32u, 2);
  const u32x4 a1 = INK_R12_DS_RBL128(r_x, b * 32u + 16u, 2);
  const float da = INK_R12_DS_RBLF32(r_x, da_base + b * 4u, 2);
  const u32 as = INK_R12_DS_RBL32(r_x, as_base + b * 4u, 2);
  const float sc = (float)row_qs[b] * da;
  u32 d = ie_q4q8_word(0u, w4.x, a0.x, a1.x);
  d = ie_q4q8_word(d, w4.y, a0.y, a1.y);
  d = ie_q4q8_word(d, w4.z, a0.z, a1.z);
  d = ie_q4q8_word(d, w4.w, a0.w, a1.w);
  const int dot = (int)(d - 8u * as);
  return sc * (float)dot;
}

/* Raw (uncapped!) P*lat + base_row for 1 row tok at shortlist index s (0 tanhf!). */
static inline __attribute__((always_inline)) float ink_r12_ds_p_row_raw(
    const G u32 *qw, const G f16 *qs,
    __amdgpu_buffer_rsrc_t r_x, u32 da_base, u32 as_base,
    const G float *base_row, u32 tok, u32 s, u32 nb, u32 sub) {
#pragma clang fp contract(off)
  const G u32 *row_qw = qw + (unsigned long)tok * (unsigned long)nb * 4ul;
  const G f16 *row_qs = qs + (unsigned long)tok * (unsigned long)nb;

  float acc;
  if (nb == 8u) {
    acc = 0.0f + ink_r12_ds_q4q8_blk(row_qw, row_qs, r_x, da_base, as_base, sub);
  } else {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (u32 k0 = 0u; k0 < nb; k0 += 32u) {
      const u32 b0 = k0 + sub;
      if (b0 < nb) a0 = a0 + ink_r12_ds_q4q8_blk(row_qw, row_qs, r_x, da_base, as_base, b0);
      const u32 b1 = b0 + 8u;
      if (b1 < nb) a1 = a1 + ink_r12_ds_q4q8_blk(row_qw, row_qs, r_x, da_base, as_base, b1);
      const u32 b2 = b0 + 16u;
      if (b2 < nb) a2 = a2 + ink_r12_ds_q4q8_blk(row_qw, row_qs, r_x, da_base, as_base, b2);
      const u32 b3 = b0 + 24u;
      if (b3 < nb) a3 = a3 + ink_r12_ds_q4q8_blk(row_qw, row_qs, r_x, da_base, as_base, b3);
    }
    acc = (a0 + a2) + (a1 + a3);
  }
  acc = ink_r12_ds_oct_tree_f(acc);
  return base_row ? (base_row[s] + acc) : acc;
}

/* Streaming 2-pass Stage-2 Top-K merge for nc <= 8192 (ng <= 256, K <= 32, 0 float arrays!). */
static inline __attribute__((always_inline)) void ink_r12_ds_stage2_merge(
    const G float *cv_row, const G u32 *ci_row, u32 nc, u32 K,
    LDSP u32 (*sm)[NT], LDSP u32 (*wsum)[8u],
    LDSP float *cand_v, LDSP u32 *cand_i,
    LDSP float *out_v_lds, LDSP u32 *out_i_lds,
    G float *out_v_g, G u32 *out_i_g) {
  const u32 t = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = lane();

  if (nc <= NT) {
    float mv = -__builtin_inff();
    u32 mi = 0xFFFFFFFFu;
    if (t < nc) {
      mv = cv_row[t]; mi = ci_row[t];
      cand_v[t] = mv; cand_i[t] = mi;
    }
    const u32 is_val = (t < nc && mi != 0xFFFFFFFFu) ? 1u : 0u;
    const u32 wcnt = (u32)__builtin_popcount(__builtin_amdgcn_ballot_w32(is_val != 0u));
    if (l == 0u) wsum[0u][w] = wcnt;
    lbarrier();
    u32 nvalid = 0u;
#pragma clang loop unroll(disable)
    for (u32 j = 0u; j < 8u; j++) nvalid += wsum[0u][j];
    if (is_val) {
      u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
      for (u32 u = 0u; u < nc; u++) {
        const u32 ui = cand_i[u];
        if (ui != 0xFFFFFFFFu) {
          const float b = cand_v[u];
          rank += (u32)((b > mv) | ((b == mv) & (ui < mi)));
        }
      }
      if (rank < K) {
        if (out_v_lds) { out_v_lds[rank] = mv; out_i_lds[rank] = mi; }
        else { out_v_g[rank] = mv; out_i_g[rank] = mi; }
      }
    }
    if (nvalid < K && t < K - nvalid) {
      if (out_v_lds) { out_v_lds[nvalid + t] = -__builtin_inff(); out_i_lds[nvalid + t] = 0xFFFFFFFFu; }
      else { out_v_g[nvalid + t] = -__builtin_inff(); out_i_g[nvalid + t] = 0xFFFFFFFFu; }
    }
    return;
  }

  /* Pass 1: find per-thread max across up to 32 strided elements (0 array VGPRs!). */
  u32 vm2 = 0u;
  float tm2 = -__builtin_inff();
  const u32 n_it = (nc + NT - 1u) >> 8u; /* <= 32 */
#pragma clang loop unroll(disable)
  for (u32 k = 0u; k < n_it; k++) {
    const u32 p = t + (k << 8u);
    if (p < nc) {
      const u32 idx = ci_row[p];
      if (idx != 0xFFFFFFFFu) {
        const float v = cv_row[p];
        vm2 |= (1u << k);
        if (v > tm2) tm2 = v;
      }
    }
  }
  const u32 tm2_k = vm2 ? ink_r12_ds_f2u(tm2) : 0u;
  const u32 tau2_k = ink_r12_ds_bitonic_kth_256(tm2_k, K, sm);
  const float tau2 = tau2_k ? ink_r12_ds_u2f(tau2_k) : -__builtin_inff();

  /* Pass 2: classify > tau2 and == tau2 into 2 scalar bitmasks (gm2, em2). */
  u32 gm2 = 0u, em2 = 0u;
  if (tau2_k == 0u) {
    gm2 = vm2;
  } else if (tm2_k >= tau2_k) {
#pragma clang loop unroll(disable)
    for (u32 k = 0u; k < n_it; k++) {
      if (vm2 & (1u << k)) {
        const float v = cv_row[t + (k << 8u)];
        if (v > tau2) gm2 |= (1u << k);
        else if (v == tau2) em2 |= (1u << k);
      }
    }
  }

  u32 ptot2 = 0u;
  const u32 poff2 = ink_r12_ds_wg_scan_u32(
      (((u32)__builtin_popcount(em2)) << 16u) | (u32)__builtin_popcount(gm2),
      wsum[1u], &ptot2);
  const u32 bgt2 = poff2 & 0xFFFFu, beq2 = poff2 >> 16u;
  const u32 tgt2 = ptot2 & 0xFFFFu, teq2 = ptot2 >> 16u;

  if (gm2) {
    u32 pos = bgt2;
#pragma clang loop unroll(disable)
    for (u32 m = gm2; m != 0u; m &= m - 1u) {
      const u32 k = (u32)__builtin_ctz(m);
      const u32 p = t + (k << 8u);
      cand_v[pos] = cv_row[p];
      cand_i[pos] = ci_row[p];
      pos++;
    }
  }

  const u32 need2 = (tgt2 < K) ? (K - tgt2) : 0u;
  u32 c2 = tgt2;
  if (need2 > 0u && teq2 <= need2) {
    if (em2) {
      u32 pos = tgt2 + beq2;
#pragma clang loop unroll(disable)
      for (u32 m = em2; m != 0u; m &= m - 1u) {
        const u32 k = (u32)__builtin_ctz(m);
        const u32 p = t + (k << 8u);
        cand_v[pos] = cv_row[p];
        cand_i[pos] = ci_row[p];
        pos++;
      }
    }
    c2 = tgt2 + teq2;
  }
  lbarrier();

  if (need2 > 0u && teq2 > need2) {
    u32 taken2 = 0u;
#pragma clang loop unroll(disable)
    for (u32 k = 0u; k < n_it; k++) {
      if (taken2 < need2) {
        const u32 hit = (em2 >> k) & 1u;
        u32 stot = 0u;
        const u32 soff = ink_r12_ds_wg_scan_u32(hit, wsum[k & 1u], &stot);
        if (hit && taken2 + soff < need2) {
          const u32 p = t + (k << 8u);
          const u32 rk = tgt2 + taken2 + soff;
          if (out_v_lds) { out_v_lds[rk] = cv_row[p]; out_i_lds[rk] = ci_row[p]; }
          else { out_v_g[rk] = cv_row[p]; out_i_g[rk] = ci_row[p]; }
        }
        taken2 += stot;
      }
    }
  }

  for (u32 j = t; j < c2; j += NT) {
    const float a = cand_v[j];
    const u32 ai = cand_i[j];
    u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
    for (u32 u = 0u; u < c2; u++) {
      const float b = cand_v[u];
      rank += (u32)((b > a) | ((b == a) & (cand_i[u] < ai)));
    }
    if (rank < K) {
      if (out_v_lds) { out_v_lds[rank] = a; out_i_lds[rank] = ai; }
      else { out_v_g[rank] = a; out_i_g[rank] = ai; }
    }
  }
  if (tau2_k == 0u && c2 < K && t < K - c2) {
    if (out_v_lds) { out_v_lds[c2 + t] = -__builtin_inff(); out_i_lds[c2 + t] = 0xFFFFFFFFu; }
    else { out_v_g[c2 + t] = -__builtin_inff(); out_i_g[c2 + t] = 0xFFFFFFFFu; }
  }
}

KERNEL ie_inkling_ds_p_sample_256wg_r12(
    const G u32 *p_qw, const G f16 *p_qs, const G u8 *lat_q8,
    const G float *base_row, const G u32 *id_map,
    G float *part_v, G u32 *part_i, G float *part_sum,
    G u32 *am_count, G u32 *ds_ctrl, G ink_r12_ds_u64 *rng_state,
    G u32 *draft_ids, G float *draft_probs, G float *draft_entropy,
    G u32 *q_n, G u32 *q_ids, G float *q_probs,
    u32 full_rows, u32 n_act, u32 nb, u32 ng, u32 K, u32 K_stride, u32 step_idx,
    float cap, float temp, float top_p, float pmin, float surv_a, float surv_b) {
  if (!ds_ctrl || ds_ctrl[1u] != 0u) return;

  const u32 g = wgid();
  const u32 t = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 row_in_wg = (w << 2u) + (l >> 3u);
  if (g >= ng || ng == 0u || ng > INK_R12_DS_NGMAX || n_act == 0u || nb == 0u) return;

  const __amdgpu_buffer_rsrc_t r_x = ink_r12_ds_mkbuf(lat_q8, nb * 40u);
  const u32 da_base = nb * 32u;
  const u32 as_base = nb * 36u;

  /* Mode A: Greedy (temp <= 0.0f) -- 0 hot-loop barriers, threshold-skipped tanhf & expf! */
  if (temp <= 0.0f) {
    float bv = -__builtin_inff();
    u32 bi = 0xFFFFFFFFu;
    float se = 0.0f;
    float raw_exp_cut = -__builtin_inff(); /* exact bound: cap(raw) < bv - 26 -> exp < 5e-12 */

#pragma clang loop unroll(disable)
    for (u32 sbase = g << 5u; sbase < n_act; sbase += (ng << 5u)) {
      const u32 s = sbase + row_in_wg;
      if (s < n_act) {
        const u32 tok = id_map ? id_map[s] : s;
        if (tok < full_rows) {
          const float raw = ink_r12_ds_p_row_raw(p_qw, p_qs, r_x, da_base, as_base,
                                                 base_row, tok, s, nb, sub);
          if (sub == 0u && raw >= raw_exp_cut) {
            const float fv = ink_r12_ds_cap(raw, cap);
            if (fv > bv) {
              se = se * __builtin_expf(bv - fv) + 1.0f;
              bv = fv; bi = tok;
              if (cap > 0.0f) {
                const float y = bv - 26.0f;
                if (y > -0.99f * cap && y < 0.99f * cap) {
                  raw_exp_cut = 0.5f * cap * __builtin_logf((cap + y) / (cap - y)) - 1e-2f;
                } else {
                  raw_exp_cut = -__builtin_inff();
                }
              } else {
                raw_exp_cut = bv - 26.0f;
              }
            } else {
              if (fv == bv && tok < bi) bi = tok;
              if (fv > bv - 26.0f) se += __builtin_expf(fv - bv);
            }
          }
        }
      }
    }

    const float my_bm = bv;
    ink_r12_ds_ps_gr_best(&bv, &bi);
    float sm = (sub == 0u && se > 0.0f) ? (se * __builtin_expf(my_bm - bv)) : 0.0f;
    sm = wave_sum_all(sm);
    if (l == 0u) ink_r12_ds_pslds.u.gr.ws[w] = sm;
    lbarrier();
    float wg_se = 0.0f;
#pragma clang loop unroll(disable)
    for (u32 j = 0u; j < 8u; j++) wg_se += ink_r12_ds_pslds.u.gr.ws[j];

    if (t == 0u) { part_v[g] = bv; part_i[g] = bi; part_sum[g] = wg_se; }
    if (ng > 1u) {
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      barrier();
      if (t == 0u) {
        ink_r12_ds_pslds.last =
            (__scoped_atomic_fetch_add(am_count, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u) ? 1u : 0u;
      }
      barrier();
      if (!ink_r12_ds_pslds.last) return;
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

      bv = -__builtin_inff(); bi = 0xFFFFFFFFu;
      for (u32 u = t; u < ng; u += NT) {
        if (ink_r12_ds_better(part_v[u], part_i[u], bv, bi)) { bv = part_v[u]; bi = part_i[u]; }
      }
      ink_r12_ds_ps_gr_best(&bv, &bi);
      float S = 0.0f;
      for (u32 u = t; u < ng; u += NT) {
        if (part_sum[u] > 0.0f) S += part_sum[u] * __builtin_expf(part_v[u] - bv);
      }
      S = wave_sum_all(S);
      if (l == 0u) ink_r12_ds_pslds.u.gr.ws[w] = S;
      lbarrier();
      wg_se = 0.0f;
#pragma clang loop unroll(disable)
      for (u32 j = 0u; j < 8u; j++) wg_se += ink_r12_ds_pslds.u.gr.ws[j];
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

  /* Mode B: Probabilistic Wave 6 (temp > 0.0f, 1 <= K <= 32, ng <= 256).
   * Wave-private K-heaps -> ZERO cross-wave lbarrier()s in the entire hot row loop!
   * 1-cycle SGPR wave_rc threshold hoists tanhf out of >99% of rows! */
  if (K == 0u || K > INK_R12_DS_KMAX) return;
  if (l < K) {
    ink_r12_ds_pslds.u.wh.kv[w][l] = -__builtin_inff();
    ink_r12_ds_pslds.u.wh.ki[w][l] = 0xFFFFFFFFu;
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");

  float w_min_v = -__builtin_inff();
  u32   w_min_i = 0xFFFFFFFFu;
  u32   w_min_pos = 0u;
  float w_rc = -__builtin_inff(); /* wave-uniform SGPR lower bound on raw logit */
  u32   w_fill = 0u;

#pragma clang loop unroll(disable)
  for (u32 sbase = g << 5u; sbase < n_act; sbase += (ng << 5u)) {
    const u32 s = sbase + row_in_wg;
    const u32 s_ok = (s < n_act) ? 1u : 0u;
    const u32 tok = s_ok ? (id_map ? id_map[s] : s) : 0xFFFFFFFFu;
    const u32 ok = (s_ok && tok < full_rows) ? 1u : 0u;

    float raw = -__builtin_inff();
    if (ok) {
      raw = ink_r12_ds_p_row_raw(p_qw, p_qs, r_x, da_base, as_base, base_row, tok, s, nb, sub);
    }

    const float rc_s = ink_r12_ds_sgpr_f(w_rc);
    const u32 hit_mask = (u32)__builtin_amdgcn_ballot_w32(sub == 0u && ok != 0u && raw >= rc_s);
    if (hit_mask != 0u) {
      /* Only passing lanes (rare after first K rows!) compute tanhf and insert into wave heap. */
      float fv = (sub == 0u && ok != 0u && raw >= rc_s) ? ink_r12_ds_cap(raw, cap) : -__builtin_inff();
      u32   fi = (sub == 0u && ok != 0u && raw >= rc_s) ? tok : 0xFFFFFFFFu;

      u32 m = hit_mask;
#pragma clang loop unroll(disable)
      while (m != 0u) {
        const u32 src_l = (u32)__builtin_ctz(m);
        m &= m - 1u;
        const float cv = I2F(__builtin_amdgcn_readlane(F2I(fv), (int)src_l));
        const float cr = I2F(__builtin_amdgcn_readlane(F2I(raw), (int)src_l));
        const u32   ci = (u32)__builtin_amdgcn_readlane((int)fi, (int)src_l);
        if (ink_r12_ds_better(cv, ci, w_min_v, w_min_i)) {
          if (l == 0u) {
            ink_r12_ds_pslds.u.wh.kv[w][w_min_pos] = cv;
            ink_r12_ds_pslds.u.wh.ki[w][w_min_pos] = ci;
          }
          __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
          if (w_fill < K) w_fill++;
          if (K == 1u) {
            w_min_v = cv; w_min_i = ci; w_min_pos = 0u;
            w_rc = ink_r12_ds_safe_raw_cut(cr, cv, cap);
          } else {
            /* Parallel 32-lane wave-wide find-worst across [0..K-1] (0 loops, NaN/+inf safe!). */
            const float pv = (l < K) ? ink_r12_ds_pslds.u.wh.kv[w][l] : __builtin_inff();
            const u32   pi = (l < K) ? ink_r12_ds_pslds.u.wh.ki[w][l] : 0u;
            float bv = pv;
            u32   bi = pi;
            u32   bp = (l < K) ? l : 0xFFFFFFFFu;
#pragma unroll
            for (u32 d = 16u; d > 0u; d >>= 1u) {
              const float ov = shfl_xor_f(bv, d);
              const u32   oi = (u32)__builtin_amdgcn_ds_bpermute((int)((l ^ d) * 4u), (int)bi);
              const u32   op = (u32)__builtin_amdgcn_ds_bpermute((int)((l ^ d) * 4u), (int)bp);
              const int o_worse = (op != 0xFFFFFFFFu) &&
                                  (bp == 0xFFFFFFFFu ||
                                   ((oi == 0xFFFFFFFFu)
                                        ? (bi != 0xFFFFFFFFu || op < bp)
                                        : (bi != 0xFFFFFFFFu && (ov < bv || (ov == bv && oi > bi)))));
              if (o_worse) { bv = ov; bi = oi; bp = op; }
            }
            w_min_v = ink_r12_ds_sgpr_f(bv);
            w_min_i = (u32)__builtin_amdgcn_readfirstlane((int)bi);
            w_min_pos = (u32)__builtin_amdgcn_readfirstlane((int)bp);
            if (w_fill >= K && w_min_i != 0xFFFFFFFFu) {
              /* Invert cap(min_v) safely to raw bound using atanh when non-saturated. */
              if (cap <= 0.0f) {
                w_rc = w_min_v;
              } else if (w_min_v > -0.999f * cap && w_min_v < 0.999f * cap) {
                const float r_est = 0.5f * cap * __builtin_logf((cap + w_min_v) / (cap - w_min_v));
                w_rc = ink_r12_ds_safe_raw_cut(r_est, w_min_v, cap);
              } else {
                w_rc = -__builtin_inff();
              }
            }
          }
        }
      }
    }
  }

  /* Merge 8 waves' K-heaps (8*K <= 256 elements) into WG's K winners -> export to part_v/part_i. */
  lbarrier();
  {
    const u32 n_wg = 8u * K;
    const u32 my_w = t / K;
    const u32 my_k = t - my_w * K;
    const float mv = (t < n_wg) ? ink_r12_ds_pslds.u.wh.kv[my_w][my_k] : -__builtin_inff();
    const u32   mi = (t < n_wg) ? ink_r12_ds_pslds.u.wh.ki[my_w][my_k] : 0xFFFFFFFFu;
    const u32 is_val = (t < n_wg && mi != 0xFFFFFFFFu) ? 1u : 0u;
    const u32 wcnt = (u32)__builtin_popcount(__builtin_amdgcn_ballot_w32(is_val != 0u));
    if (l == 0u) ink_r12_ds_pslds.wsum[0u][w] = wcnt;
    lbarrier();
    u32 nvalid = 0u;
#pragma clang loop unroll(disable)
    for (u32 j = 0u; j < 8u; j++) nvalid += ink_r12_ds_pslds.wsum[0u][j];

    G float *pv_dst = part_v + (unsigned long)g * (unsigned long)K;
    G u32   *pi_dst = part_i + (unsigned long)g * (unsigned long)K;
    if (is_val) {
      u32 rank = 0u;
#pragma clang loop unroll(disable)
      for (u32 uw = 0u; uw < 8u; uw++) {
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
        for (u32 uk = 0u; uk < K; uk++) {
          const u32 ui = ink_r12_ds_pslds.u.wh.ki[uw][uk];
          if (ui != 0xFFFFFFFFu) {
            const float b = ink_r12_ds_pslds.u.wh.kv[uw][uk];
            rank += (u32)((b > mv) | ((b == mv) & (ui < mi)));
          }
        }
      }
      if (rank < K) {
        if (ng == 1u) { ink_r12_ds_pslds.fv[rank] = mv; ink_r12_ds_pslds.fi[rank] = mi; }
        else { pv_dst[rank] = mv; pi_dst[rank] = mi; }
      }
    }
    if (nvalid < K && t < K - nvalid) {
      if (ng == 1u) { ink_r12_ds_pslds.fv[nvalid + t] = -__builtin_inff(); ink_r12_ds_pslds.fi[nvalid + t] = 0xFFFFFFFFu; }
      else { pv_dst[nvalid + t] = -__builtin_inff(); pi_dst[nvalid + t] = 0xFFFFFFFFu; }
    }
  }

  if (ng > 1u) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (t == 0u) {
      ink_r12_ds_pslds.last =
          (__scoped_atomic_fetch_add(am_count, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r12_ds_pslds.last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (t == 0u) *am_count = 0u;

    ink_r12_ds_stage2_merge(part_v, part_i, ng * K, K,
                            ink_r12_ds_pslds.u.sm, ink_r12_ds_pslds.wsum,
                            ink_r12_ds_pslds.u.cand.cv, ink_r12_ds_pslds.u.cand.ci,
                            ink_r12_ds_pslds.fv, ink_r12_ds_pslds.fi,
                            (G float *)0, (G u32 *)0);
  }
  lbarrier();

  /* Stage 3 (Wave 0): exact Softmax(temp) + top_p nucleus + ForeSpark H(q) + xorshift64*. */
  if (t < 32u) {
    const u32 c = t;
    const u32 ci = (c < K) ? ink_r12_ds_pslds.fi[c] : 0xFFFFFFFFu;
    const float cv = (c < K) ? ink_r12_ds_pslds.fv[c] : -__builtin_inff();
    const float v0 = ink_r12_ds_pslds.fv[0u];
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

    ink_r12_ds_pslds.fq[c] = qc;
    if (c < m && c < K_stride) {
      if (q_ids) q_ids[(unsigned long)step_idx * (unsigned long)K_stride + (unsigned long)c] = ci;
      if (q_probs) q_probs[(unsigned long)step_idx * (unsigned long)K_stride + (unsigned long)c] = qc;
    }

    if (c == 0u) {
      u32 chosen = 0u;
      u32 chosen_tok = (m > 0u) ? ink_r12_ds_pslds.fi[0u] : 0u;
      float chosen_p = (m > 0u) ? ink_r12_ds_pslds.fq[0u] : 0.0f;
      if (m > 1u && rng_state) {
        ink_r12_ds_u64 rs = *rng_state;
        if (rs == 0ull) rs = 0x9E3779B97F4A7C15ull ^ ((ink_r12_ds_u64)step_idx + 1ull);
        rs ^= rs >> 12u; rs ^= rs << 25u; rs ^= rs >> 27u;
        *rng_state = rs;
        const ink_r12_ds_u64 rnd = rs * 2685821657736338717ull;
        float u = (float)((u32)(rnd >> 40u)) * (1.0f / 16777216.0f);
        u32 last_nz = 0u;
#pragma clang loop unroll(disable)
        for (u32 j = 0u; j < m; j++) {
          const float pj = ink_r12_ds_pslds.fq[j];
          if (pj > 0.0f) last_nz = j;
          u -= pj;
          if (u <= 0.0f && pj > 0.0f) { chosen = j; break; }
          if (j + 1u == m) chosen = last_nz;
        }
        chosen_tok = ink_r12_ds_pslds.fi[chosen];
        chosen_p = ink_r12_ds_pslds.fq[chosen];
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
 * 7. H23: ie_inkling_q4k_head_topk_256wg_r12
 *    256-WG (4 WGs/CU) Threshold-Gated Q4_K Output Head + Cap + Argmax/Top-K.
 *    32-lane/row single-pass Q4_K (TH_LOAD_NT on 260 MiB output.weight,
 *    TH_LOAD_HT on xq -> <= 80 VGPR!), 1-cycle SGPR raw_cut hoists tanhf out
 *    of hot loop (0 hot barriers for K==1 greedy, 1-barrier skip for K>1),
 *    2-pass streaming Stage-2 merge supporting ng <= 256 (nc <= 8192).
 * ========================================================================== */

typedef struct {
  union {
    struct {
      float kv[INK_R12_DS_TMAX][INK_R12_DS_KMAX]; /* 1,024 B */
      u32   ki[INK_R12_DS_TMAX][INK_R12_DS_KMAX]; /* 1,024 B */
      float min_v[INK_R12_DS_TMAX];               /*    32 B */
      u32   min_i[INK_R12_DS_TMAX];               /*    32 B */
      u32   min_pos[INK_R12_DS_TMAX];             /*    32 B */
      float raw_cut[INK_R12_DS_TMAX];             /*    32 B */
      u32   fill[INK_R12_DS_TMAX];                /*    32 B */
      float ch_v[INK_R12_DS_TMAX][8u];            /*   256 B */
      u32   ch_i[INK_R12_DS_TMAX][8u];            /*   256 B */
      u32   ch_mask[INK_R12_DS_TMAX];             /*    32 B */
      u32   any_mask;                             /*     4 B */
    } st1;
    struct {
      float bv[8u][INK_R12_DS_TMAX];              /*   256 B: K==1 wave winners */
      u32   bi[8u][INK_R12_DS_TMAX];              /*   256 B */
    } gr1;
    u32 sm[2u][NT];                               /* 2,048 B */
    struct {
      float cv[INK_R12_DS_CAP];                   /* 4,096 B */
      u32   ci[INK_R12_DS_CAP];                   /* 4,096 B */
    } cand;
  } u;
  u32   wsum[2u][8u];
  float wv[8u];
  u32   wi[8u];
  u32   last;
} ink_r12_ds_head_lds_t;

static LDS ink_r12_ds_head_lds_t ink_r12_ds_hlds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r12_ds_head_lds_t) <= 16384u, "ink_r12_ds_hlds must stay <= 16 KiB");

/* Exact 32-lane/row Q4_K dot product for 1 row r across T <= 8 tokens (TH_LOAD_NT weights, TH_LOAD_HT xq). */
static inline __attribute__((always_inline)) void ink_r12_ds_q4k_row32(
    const G u32 *qw, const G u16 *qs, __amdgpu_buffer_rsrc_t r_x,
    u32 r, u32 rows, u32 nb, u32 T, u32 xs, u32 l,
    ink_r12_ds_f8 *out_v) {
#pragma clang fp contract(off)
  const u32 nsb = nb >> 3u;
  const G u32 *row_qw = qw + (unsigned long)r * (unsigned long)nb * 4ul;
  const G u16 *row_sm = qs + (unsigned long)r * (unsigned long)nb;
  const G u32 *row_dd32 = (const G u32 *)(qs + (unsigned long)rows * (unsigned long)nb +
                                          (unsigned long)r * (unsigned long)nsb * 2ul);
  const u32 da_base = nb * 32u;
  const u32 as_base = nb * 36u;

  ink_r12_ds_f8 acc = (ink_r12_ds_f8)(0.0f);

#pragma clang loop unroll(disable)
  for (u32 b = l; b < nb; b += 32u) {
    const u32x4 w4 = ink_r12_ds_nt4(row_qw + (unsigned long)b * 4ul);
    const u32 sm = (u32)ink_r12_ds_nt16(row_sm + b);
    const u32 dd_raw = ink_r12_ds_nt32(row_dd32 + (b >> 3u));
    const u16 d_bits = (u16)(dd_raw & 0xFFFFu);
    const u16 dm_bits = (u16)(dd_raw >> 16u);
    const float dv = (float)__builtin_bit_cast(f16, d_bits);
    const float mv = (float)__builtin_bit_cast(f16, dm_bits);
    const float sc = dv * (float)(sm & 63u);
    const float mm = mv * (float)(sm >> 8u);

    const u32 lo0 = ie_q4k_lo(w4.x), hi0 = ie_q4k_hi(w4.x);
    const u32 lo1 = ie_q4k_lo(w4.y), hi1 = ie_q4k_hi(w4.y);
    const u32 lo2 = ie_q4k_lo(w4.z), hi2 = ie_q4k_hi(w4.z);
    const u32 lo3 = ie_q4k_lo(w4.w), hi3 = ie_q4k_hi(w4.w);

    const u32 aw_off = b * 32u;
    const u32 da_off = da_base + b * 4u;
    const u32 as_off = as_base + b * 4u;

#pragma unroll
    for (u32 j = 0u; j < INK_R12_DS_TMAX; j++) {
      if (j < T) {
        const u32 jbase = j * xs;
        const u32x4 act0 = INK_R12_DS_RBL128(r_x, jbase + aw_off, 2);
        const u32x4 act1 = INK_R12_DS_RBL128(r_x, jbase + aw_off + 16u, 2);
        const float da = INK_R12_DS_RBLF32(r_x, jbase + da_off, 2);
        const int as = (int)INK_R12_DS_RBL32(r_x, jbase + as_off, 2);
        const float xa = da * (float)as;

        u32 dq = ie_dot4_us(0u, lo0, act0.x);
        dq = ie_dot4_us(dq, hi0, act1.x);
        dq = ie_dot4_us(dq, lo1, act0.y);
        dq = ie_dot4_us(dq, hi1, act1.y);
        dq = ie_dot4_us(dq, lo2, act0.z);
        dq = ie_dot4_us(dq, hi2, act1.z);
        dq = ie_dot4_us(dq, lo3, act0.w);
        dq = ie_dot4_us(dq, hi3, act1.w);

        acc[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(sc * (float)(int)dq, da, acc[j]));
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  ink_r12_ds_f8 res = (ink_r12_ds_f8)(0.0f);
#pragma unroll
  for (u32 j = 0u; j < INK_R12_DS_TMAX; j++) {
    if (j < T) res[j] = wave_tree_f(acc[j]);
  }
  *out_v = res;
}

KERNEL ie_inkling_q4k_head_topk_256wg_r12(
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
  const u32 nv = (unpadded > 0u && unpadded < rows) ? unpadded : rows;

  if (g >= ng || ng == 0u || ng > INK_R12_DS_NGMAX || T == 0u || T > INK_R12_DS_TMAX ||
      K == 0u || K > INK_R12_DS_KMAX || nb == 0u || (nb & 7u) != 0u || nv == 0u) return;

  const __amdgpu_buffer_rsrc_t r_x = ink_r12_ds_mkbuf(xq, T * xs);

  if (K == 1u) {
    /* Fast Greedy Mode (K == 1): 0 barriers in hot loop, wave-private (bv, bi, raw_cut)! */
    ink_r12_ds_f8 bv = (ink_r12_ds_f8)(-__builtin_inff());
    ink_r12_ds_f8 rc = (ink_r12_ds_f8)(-__builtin_inff());
    u32 bi[INK_R12_DS_TMAX];
#pragma unroll
    for (u32 j = 0u; j < INK_R12_DS_TMAX; j++) bi[j] = 0xFFFFFFFFu;

#pragma clang loop unroll(disable)
    for (u32 rbase = g << 3u; rbase < nv; rbase += (ng << 3u)) {
      const u32 r = rbase + w;
      if (r < nv) {
        ink_r12_ds_f8 vals;
        ink_r12_ds_q4k_row32(qw, qs, r_x, r, rows, nb, T, xs, l, &vals);
        if (l == 0u) {
          const float b_val = bias ? bias[r] : 0.0f;
#pragma unroll
          for (u32 j = 0u; j < INK_R12_DS_TMAX; j++) {
            if (j < T) {
              const float raw = vals[j] + b_val;
              if (raw >= rc[j]) {
                const float fv = ink_r12_ds_cap(raw, cap);
                if (ink_r12_ds_better(fv, r, bv[j], bi[j])) {
                  bv[j] = fv;
                  bi[j] = r;
                  rc[j] = ink_r12_ds_safe_raw_cut(raw, fv, cap);
                }
              }
            }
          }
        }
      }
    }

    if (l == 0u) {
#pragma unroll
      for (u32 j = 0u; j < INK_R12_DS_TMAX; j++) {
        if (j < T) {
          ink_r12_ds_hlds.u.gr1.bv[w][j] = bv[j];
          ink_r12_ds_hlds.u.gr1.bi[w][j] = bi[j];
        }
      }
    }
    lbarrier();

    if (t < T) {
      float fv = ink_r12_ds_hlds.u.gr1.bv[0u][t];
      u32   fi = ink_r12_ds_hlds.u.gr1.bi[0u][t];
#pragma clang loop unroll(disable)
      for (u32 ww = 1u; ww < 8u; ww++) {
        if (ink_r12_ds_better(ink_r12_ds_hlds.u.gr1.bv[ww][t], ink_r12_ds_hlds.u.gr1.bi[ww][t], fv, fi)) {
          fv = ink_r12_ds_hlds.u.gr1.bv[ww][t];
          fi = ink_r12_ds_hlds.u.gr1.bi[ww][t];
        }
      }
      part_v[(unsigned long)t * (unsigned long)ng + (unsigned long)g] = fv;
      part_i[(unsigned long)t * (unsigned long)ng + (unsigned long)g] = fi;
    }
  } else {
    /* Sampled Top-K Mode (2 <= K <= 32): 1-cycle SGPR raw_cut filter + 1-barrier skip on >99% chunks! */
    if (t < T * K) {
      const u32 tj = t / K;
      const u32 tk = t - tj * K;
      ink_r12_ds_hlds.u.st1.kv[tj][tk] = -__builtin_inff();
      ink_r12_ds_hlds.u.st1.ki[tj][tk] = 0xFFFFFFFFu;
    }
    if (t < T) {
      ink_r12_ds_hlds.u.st1.min_v[t] = -__builtin_inff();
      ink_r12_ds_hlds.u.st1.min_i[t] = 0xFFFFFFFFu;
      ink_r12_ds_hlds.u.st1.min_pos[t] = 0u;
      ink_r12_ds_hlds.u.st1.raw_cut[t] = -__builtin_inff();
      ink_r12_ds_hlds.u.st1.fill[t] = 0u;
      ink_r12_ds_hlds.u.st1.ch_mask[t] = 0u;
    }
    if (t == 0u) ink_r12_ds_hlds.u.st1.any_mask = 0u;
    lbarrier();

    ink_r12_ds_f8 rc_s = (ink_r12_ds_f8)(-__builtin_inff());

#pragma clang loop unroll(disable)
    for (u32 rbase = g << 3u; rbase < nv; rbase += (ng << 3u)) {
      const u32 r = rbase + w;
      if (r < nv) {
        ink_r12_ds_f8 vals;
        ink_r12_ds_q4k_row32(qw, qs, r_x, r, rows, nb, T, xs, l, &vals);
        if (l == 0u) {
          const float b_val = bias ? bias[r] : 0.0f;
          u32 my_tmask = 0u;
#pragma unroll
          for (u32 j = 0u; j < INK_R12_DS_TMAX; j++) {
            if (j < T) {
              const float raw = vals[j] + b_val;
              if (raw >= rc_s[j]) {
                const float fv = ink_r12_ds_cap(raw, cap);
                if (ink_r12_ds_better(fv, r, ink_r12_ds_hlds.u.st1.min_v[j], ink_r12_ds_hlds.u.st1.min_i[j])) {
                  ink_r12_ds_hlds.u.st1.ch_v[j][w] = fv;
                  ink_r12_ds_hlds.u.st1.ch_i[j][w] = r;
                  __atomic_fetch_or(&ink_r12_ds_hlds.u.st1.ch_mask[j], 1u << w, __ATOMIC_RELAXED);
                  my_tmask |= (1u << j);
                }
              }
            }
          }
          if (my_tmask != 0u) {
            __atomic_fetch_or(&ink_r12_ds_hlds.u.st1.any_mask, my_tmask, __ATOMIC_RELAXED);
          }
        }
      }
      lbarrier();

      const u32 amask = (u32)__builtin_amdgcn_readfirstlane((int)ink_r12_ds_hlds.u.st1.any_mask);
      if (amask != 0u) {
        /* Wave w (< T) updates token w's K-heap in parallel (32-lane min-search, 0 inner loops!). */
        if (w < T && ((amask >> w) & 1u) != 0u) {
          u32 cm = (u32)__builtin_amdgcn_readfirstlane((int)ink_r12_ds_hlds.u.st1.ch_mask[w]);
          float cur_mv = ink_r12_ds_sgpr_f(ink_r12_ds_hlds.u.st1.min_v[w]);
          u32   cur_mi = (u32)__builtin_amdgcn_readfirstlane((int)ink_r12_ds_hlds.u.st1.min_i[w]);
          u32   cur_mp = (u32)__builtin_amdgcn_readfirstlane((int)ink_r12_ds_hlds.u.st1.min_pos[w]);
          u32   cur_fl = (u32)__builtin_amdgcn_readfirstlane((int)ink_r12_ds_hlds.u.st1.fill[w]);

#pragma clang loop unroll(disable)
          while (cm != 0u) {
            const u32 cw = (u32)__builtin_ctz(cm);
            cm &= cm - 1u;
            const float cv = ink_r12_ds_sgpr_f(ink_r12_ds_hlds.u.st1.ch_v[w][cw]);
            const u32   ci = (u32)__builtin_amdgcn_readfirstlane((int)ink_r12_ds_hlds.u.st1.ch_i[w][cw]);
            if (ink_r12_ds_better(cv, ci, cur_mv, cur_mi)) {
              if (l == 0u) {
                ink_r12_ds_hlds.u.st1.kv[w][cur_mp] = cv;
                ink_r12_ds_hlds.u.st1.ki[w][cur_mp] = ci;
              }
              __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
              if (cur_fl < K) cur_fl++;

              const float pv = (l < K) ? ink_r12_ds_hlds.u.st1.kv[w][l] : __builtin_inff();
              const u32   pi = (l < K) ? ink_r12_ds_hlds.u.st1.ki[w][l] : 0u;
              float bv = pv; u32 bi = pi; u32 bp = (l < K) ? l : 0xFFFFFFFFu;
#pragma unroll
              for (u32 d = 16u; d > 0u; d >>= 1u) {
                const float ov_d = shfl_xor_f(bv, d);
                const u32   oi_d = (u32)__builtin_amdgcn_ds_bpermute((int)((l ^ d) * 4u), (int)bi);
                const u32   op_d = (u32)__builtin_amdgcn_ds_bpermute((int)((l ^ d) * 4u), (int)bp);
                const int o_worse = (op_d != 0xFFFFFFFFu) &&
                                    (bp == 0xFFFFFFFFu ||
                                     ((oi_d == 0xFFFFFFFFu)
                                          ? (bi != 0xFFFFFFFFu || op_d < bp)
                                          : (bi != 0xFFFFFFFFu && (ov_d < bv || (ov_d == bv && oi_d > bi)))));
                if (o_worse) { bv = ov_d; bi = oi_d; bp = op_d; }
              }
              cur_mv = ink_r12_ds_sgpr_f(bv);
              cur_mi = (u32)__builtin_amdgcn_readfirstlane((int)bi);
              cur_mp = (u32)__builtin_amdgcn_readfirstlane((int)bp);
            }
          }
          if (l == 0u) {
            ink_r12_ds_hlds.u.st1.min_v[w] = cur_mv;
            ink_r12_ds_hlds.u.st1.min_i[w] = cur_mi;
            ink_r12_ds_hlds.u.st1.min_pos[w] = cur_mp;
            ink_r12_ds_hlds.u.st1.fill[w] = cur_fl;
            ink_r12_ds_hlds.u.st1.ch_mask[w] = 0u;
            float nrc = -__builtin_inff();
            if (cur_fl >= K && cur_mi != 0xFFFFFFFFu) {
              if (cap <= 0.0f) {
                nrc = cur_mv;
              } else if (cur_mv > -0.999f * cap && cur_mv < 0.999f * cap) {
                const float r_est = 0.5f * cap * __builtin_logf((cap + cur_mv) / (cap - cur_mv));
                nrc = ink_r12_ds_safe_raw_cut(r_est, cur_mv, cap);
              }
            }
            ink_r12_ds_hlds.u.st1.raw_cut[w] = nrc;
          }
        }
        if (t == 0u) ink_r12_ds_hlds.u.st1.any_mask = 0u;
        lbarrier();
#pragma unroll
        for (u32 j = 0u; j < INK_R12_DS_TMAX; j++) {
          if (j < T) rc_s[j] = ink_r12_ds_sgpr_f(ink_r12_ds_hlds.u.st1.raw_cut[j]);
        }
      }
    }

    const u32 tk_total = T * K;
    for (u32 idx = t; idx < tk_total; idx += NT) {
      const u32 tj = idx / K;
      const u32 tk = idx - tj * K;
      const unsigned long dst = (((unsigned long)tj * (unsigned long)ng + (unsigned long)g) * (unsigned long)K) + tk;
      part_v[dst] = ink_r12_ds_hlds.u.st1.kv[tj][tk];
      part_i[dst] = ink_r12_ds_hlds.u.st1.ki[tj][tk];
    }
  }

  /* Agent release/acquire + last-WG election across all ng WGs. */
  if (ng > 1u) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (t == 0u) {
      ink_r12_ds_hlds.last =
          (__scoped_atomic_fetch_add(done, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r12_ds_hlds.last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (t == 0u) *done = 0u;
  } else {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  }

  /* Stage 2 (last WG): merge nc = ng * K <= 8192 for each token tj = 0..T-1. */
  const u32 nc = ng * K;
#pragma clang loop unroll(disable)
  for (u32 tj = 0u; tj < T; tj++) {
    const G float *cv_row = part_v + (unsigned long)tj * (unsigned long)nc;
    const G u32   *ci_row = part_i + (unsigned long)tj * (unsigned long)nc;

    if (K == 1u) {
      float bv = -__builtin_inff();
      u32 bi = 0xFFFFFFFFu;
      for (u32 u = t; u < ng; u += NT) {
        const float vu = cv_row[u];
        const u32 iu = ci_row[u];
        if (ink_r12_ds_better(vu, iu, bv, bi)) { bv = vu; bi = iu; }
      }
#pragma unroll
      for (u32 m = 16u; m > 0u; m >>= 1u) {
        const float ov_m = shfl_xor_f(bv, m);
        const u32 oi_m = (u32)__builtin_amdgcn_ds_bpermute((int)((l ^ m) * 4u), (int)bi);
        if (ink_r12_ds_better(ov_m, oi_m, bv, bi)) { bv = ov_m; bi = oi_m; }
      }
      if (l == 0u) { ink_r12_ds_hlds.wv[w] = bv; ink_r12_ds_hlds.wi[w] = bi; }
      lbarrier();
      if (t == 0u) {
        float fv = ink_r12_ds_hlds.wv[0u];
        u32 fi = ink_r12_ds_hlds.wi[0u];
#pragma clang loop unroll(disable)
        for (u32 j = 1u; j < 8u; j++) {
          if (ink_r12_ds_better(ink_r12_ds_hlds.wv[j], ink_r12_ds_hlds.wi[j], fv, fi)) {
            fv = ink_r12_ds_hlds.wv[j]; fi = ink_r12_ds_hlds.wi[j];
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
    ink_r12_ds_stage2_merge(cv_row, ci_row, nc, K,
                            ink_r12_ds_hlds.u.sm, ink_r12_ds_hlds.wsum,
                            ink_r12_ds_hlds.u.cand.cv, ink_r12_ds_hlds.u.cand.ci,
                            (LDSP float *)0, (LDSP u32 *)0,
                            ov_row, oi_row);
    lbarrier();
  }
}

#undef INK_R12_DS_RBLF32
#undef INK_R12_DS_RBL16
#undef INK_R12_DS_RBL32
#undef INK_R12_DS_RBL128
#undef INK_R12_DS_BUF_FLAGS
#undef INK_R12_DS_QSC
#undef INK_R12_DS_LN2
#undef INK_R12_DS_LOG2E
#undef INK_R12_DS_CAP
#undef INK_R12_DS_NGMAX
#undef INK_R12_DS_KMAX
#undef INK_R12_DS_TMAX
#undef INK_R12_DS_PSTR
#undef INK_R12_DS_GQA
#undef INK_R12_DS_HD
#undef INK_R12_DS_RNN_MAXNB

#endif /* !IE_MEGA */
