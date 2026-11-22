/* ============================================================================
 * Inkling-2-Mini Round 5 (r5-dspark-kernels): Zero-Sync Device-Side DSpark
 * Markov Loop Kernels + Small-Row Multi-Token Verify Q8_0 GEMV for RX 9070 XT
 * (gfx1201, wave32, NT=256).
 *
 * All local symbols are prefixed ink_r5_ / INK_R5_, all macros are #undef'd
 * at EOF, Clang 21 __builtin_bit_cast-safe, 0 register spills.
 *
 * Device control block layout (ds_ctrl[4], u32):
 *   ds_ctrl[0] = prev_tok   (initialized to anchor_id; updated to best_id each step)
 *   ds_ctrl[1] = stop_flag  (initialized to 0; set to 1 when best_prob < pmin)
 *   ds_ctrl[2] = n_ran      (initialized to 0; updated to step_idx + 1 each active step)
 *   ds_ctrl[3] = 0          (reserved)
 *
 * Exported Kernels, Grids, Guards, and Parity:
 *
 * --- Part A: Zero-Sync Device-Side DSpark Markov Loop ---
 *
 * 1. KERNEL ie_dspark_block_init_zsync_r5(
 *        const G u8 *emb_qw, const G u16 *emb_qs, const G float *emb_norm_w,
 *        const G float *mask_embed, G float *block_out, G float *ds_state,
 *        G u32 *ds_ctrl, u32 anchor_id, u32 dim, u32 nb, float eps,
 *        u32 block_size, u32 rank)
 *    - Grid:   (1u, block_size, 1u)  [block_size=8, dim=2304, nb=72, rank=256]
 *    - Guard:  block_size > 0u && dim > 0u && nb * 32u >= dim && emb_qw && emb_qs &&
 *              mask_embed && block_out && ds_ctrl && (rank == 0u || ds_state);
 *              for dim==2304 fast path, 16-byte aligned emb_norm_w, mask_embed,
 *              block_out (4-byte aligned emb_qw).
 *    - Effect: Same as ie_dspark_block_init_r4, PLUS t==0 zeros ds_state[0..rank-1]
 *              (eliminates hipMemset) and sets ds_ctrl = {anchor_id, 0, 0, 0}.
 *    - Parity: bitwise-exact vs ie_dspark_block_init_r4 + hipMemset(ds_state, 0).
 *
 * 2. KERNEL ie_dspark_concat_q8_zsync_r5(
 *        const G float *state, const G u8 *w1_qw, const G u16 *w1_qs,
 *        const G float *hidden_i, const G float *f_rnn, G u8 *qout,
 *        const G u32 *ds_ctrl, u32 rank, u32 dim)
 *    - Grid:   ((2u * rank + 2u * dim + 255u) / 256u, 1u, 1u)  [20 WGs for 5120]
 *    - Guard:  !b->m->ds_head.rows (identity vocab IDs in ds_ctrl[0]), ds_ctrl &&
 *              (rank & 31u) == 0u && (dim & 31u) == 0u && (rank + dim) > 0u &&
 *              state && w1_qw && w1_qs && hidden_i && f_rnn && qout.
 *    - Effect: Early-returns if ds_ctrl[1] != 0u; reads prev_tok = ds_ctrl[0].
 *    - Parity: bitwise-exact vs ie_dspark_concat_q8_r4 (0 LDS).
 *
 * 3. KERNEL ie_dspark_rnn_step_zsync_r5(
 *        const G float *raw768, G float *state, G float *latent,
 *        G u8 *latent_q8, const G u32 *ds_ctrl, u32 rank)
 *    - Grid:   (1u, 1u, 1u)  [rank=256]
 *    - Guard:  rank > 0u && (rank & 31u) == 0u && raw768 && state.
 *    - Effect: Early-returns if ds_ctrl && ds_ctrl[1] != 0u.
 *    - Parity: bitwise-exact vs ie_dspark_rnn_step_r4 (0 LDS).
 *
 * 4. KERNEL ie_dspark_add_q8_zsync_r5(
 *        const G float *hidden_i, const G float *delta, const G float *bias,
 *        G u8 *qout, const G u32 *ds_ctrl, u32 dim)
 *    - Grid:   ((dim + 255u) / 256u, 1u, 1u)  [9 WGs for dim=2304]
 *    - Guard:  dim > 0u && (dim & 31u) == 0u && hidden_i && delta && qout.
 *    - Effect: Early-returns if ds_ctrl && ds_ctrl[1] != 0u.
 *    - Parity: bitwise-exact vs ie_dspark_add_q8_r4 (0 LDS).
 *
 * 5. KERNEL ie_dspark_gemv_p_q4_zsync_r5(
 *        const G u32 *qw, const G f16 *qs, const G u8 *xq,
 *        const G float *base_row, G float *logits, u32 rows, u32 nb,
 *        const G u32 *ds_ctrl)
 *    - Grid:   ((rows + 31u) / 32u, 1u, 1u)  [6282 WGs for rows=201024, nb=8]
 *    - Guard:  rows > 0u && nb > 0u && qw && qs && xq && logits (repack_q4 Q4_0 layout).
 *    - Effect: Early-returns if ds_ctrl && ds_ctrl[1] != 0u; computes
 *              logits[r] = dot_q4_q8(P[r], xq) + (base_row ? base_row[r] : 0)
 *              at 4 rows/wave (8 lanes/row = 1 lane/block for nb=8, 100% util,
 *              512 B coalesced wave reads, L3-cached for 28.95 MB P, 0 LDS).
 *    - Parity: bitwise-exact vs ie_gemv_q4q8(..., bias=NULL, res=base_row).
 *
 * 6. KERNEL ie_dspark_cap_argmax_zsync_r5(
 *        const G float *logits, G float *part_max, G u32 *part_id,
 *        G float *part_sum, G u32 *count, G u32 *ds_ctrl,
 *        G u32 *draft_ids, G float *draft_probs,
 *        u32 V, float cap, float pmin, u32 step_idx)
 *    - Grid:   (128u, 1u, 1u)  [ng = ngroups_x() == 128; part_* sized >= ng]
 *    - Guard:  !b->m->ds_head.rows && V > 0u && logits && part_max && part_id &&
 *              part_sum && count && ds_ctrl (*count initialized to 0 before first call).
 *    - Effect: Early-returns if ds_ctrl[1] != 0u (leaves *count untouched).
 *              Single-pass register logit_cap (cap > 0 ? cap * ie_tanhf_s(x / cap) : x)
 *              + argmax + online sum_exp across logits[0..V-1] (0 logits writeback).
 *              Last WG resets *count = 0, writes draft_ids[step_idx] = best_id,
 *              draft_probs[step_idx] = 1/sum_exp, ds_ctrl[0] = best_id,
 *              ds_ctrl[2] = step_idx + 1u, and sets ds_ctrl[1] = 1u if best_prob < pmin.
 *    - Parity: bitwise-exact best_id vs ie_logit_cap + ie_argmax(n=V);
 *              f32-tol (<= 1e-6 rel) on online softmax best_prob (100 B LDS).
 *
 * 7. KERNEL ie_dspark_gemv_q8_zsync_r5 / ie_dspark_gemv_q4k_zsync_r5 /
 *           ie_dspark_gemv_q6k_zsync_r5
 *    - Grid:   ((rows + 7u) / 8u, 1u, 1u)  [exact ie_gemv_ngroups(rows)]
 *    - Guard:  same as ie_gemv_q8q8 / ie_gemv_q4kq8 / ie_gemv_q6kq8 (no fused norm).
 *    - Effect: Early-returns if ds_ctrl && ds_ctrl[1] != 0u, skipping rnn_proj,
 *              hidden_out, and unsplit lm_head (260.5 MB) after stop_flag latches.
 *    - Parity: bitwise-exact vs ie_gemv_q8q8 / ie_gemv_q4kq8 / ie_gemv_q6kq8 (0 LDS).
 *
 * --- Part B: Small-Row Multi-Token Verify Q8_0 GEMV ---
 *
 * 8. KERNEL ie_inkling_gemv_q8_small_ver_r5(
 *        const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
 *        u32 rows, u32 nb, const G float *bias, const G float *res,
 *        u32 T, u32 xs, u32 ys, u32 rs)
 *    - Grid:   (rows, 1u, 1u)  [1 WG per row, 8 waves/row; safe for padded grid]
 *    - Guard:  T >= 2u && T <= 16u && rows > 0u && rows <= 128u && nb > 0u &&
 *              qw && qs && xq && y (target: 64x4608, 24x2304, 32x2304 verify ops).
 *    - Effect: Reads each Q8_0 weight block ONCE per chunk of <= 8 tokens and dots
 *              against all tokens in the chunk (T<=8: 1 pass = Tx less weight traffic;
 *              9<=T<=16: 2 passes), with all <=8 waves reducing tokens in parallel
 *              via 8 KiB conflict-free LDS.
 *    - Parity: bitwise-exact (0-bit-diff) vs ie_gemv_q8q8_ts / ie_gemv_q8q8_t.
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(f16) == 2u, "ink_r5 f16 must be 2 bytes");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r5 float/u32 must be 4 bytes");

#define INK_R5_DIM_FAST  2304u
#define INK_R5_NB_FAST   72u
#define INK_R5_VER_CHUNK 8u

struct __attribute__((packed, aligned(4))) ink_r5_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r5_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r5_u32x4p *)p)->v;
}

/* 1-barrier LDS workgroup sum for ie_dspark_block_init_zsync_r5 (32 B LDS). */
static LDS float ink_r5_rn_red[8];

static inline __attribute__((always_inline)) float ink_r5_wg_sum_1b(float v, u32 wv, u32 l) {
  v = wave_tree_f(v);
  if (l == 0u) ink_r5_rn_red[wv] = v;
  lbarrier();
  float s = 0.0f;
#pragma unroll
  for (u32 w = 0u; w < 8u; w++) s += ink_r5_rn_red[w];
  return s;
}

/* Exact Q4_0 x Q8_0 single-block scaled dot product (matches ie_q4q8_block). */
static inline __attribute__((always_inline)) float ink_r5_q4q8_blk(
    const G u32 *row_qw, const G f16 *row_qs, const G u32 *aw,
    const G float *da, const G u32 *asum, u32 b) {
  const u32x4 w4 = ink_r5_ld4(row_qw + (unsigned long)b * 4ul);
  const u32x4 a0 = ink_r5_ld4(aw + (unsigned long)b * 8ul);
  const u32x4 a1 = ink_r5_ld4(aw + (unsigned long)b * 8ul + 4ul);
  const float sc = (float)row_qs[b] * da[b];
  const u32 as = asum[b];
  u32 d = ie_q4q8_word(0u, w4.x, a0.x, a1.x);
  d = ie_q4q8_word(d, w4.y, a0.y, a1.y);
  d = ie_q4q8_word(d, w4.z, a0.z, a1.z);
  d = ie_q4q8_word(d, w4.w, a0.w, a1.w);
  const int dot = (int)(d - 8u * as);
  return sc * (float)dot;
}

/* Exact Q8_0 x Q8_0 single-block integer dot product (matches ie_q8q8_block). */
static inline __attribute__((always_inline)) int ink_r5_dot8_ss(
    u32x4 w0, u32x4 w1, u32x4 a0, u32x4 a1) {
  u32 d = ie_dot4_ss(0u, w0.x, a0.x);
  d = ie_dot4_ss(d, w0.y, a0.y);
  d = ie_dot4_ss(d, w0.z, a0.z);
  d = ie_dot4_ss(d, w0.w, a0.w);
  d = ie_dot4_ss(d, w1.x, a1.x);
  d = ie_dot4_ss(d, w1.y, a1.y);
  d = ie_dot4_ss(d, w1.z, a1.z);
  d = ie_dot4_ss(d, w1.w, a1.w);
  return (int)d;
}

/* Online (max, argmax, sum_exp) merge: preserves exact tie-breaking (smaller index wins)
 * and keeps expf arguments <= 0 (NaN-free even for -inf inputs). */
static inline __attribute__((always_inline)) void ink_r5_am_merge(
    float *va, u32 *ia, float *sa, float vb, u32 ib, float sb) {
  if (ib == 0xFFFFFFFFu || sb <= 0.0f) return;
  if (*ia == 0xFFFFFFFFu || *sa <= 0.0f) {
    *va = vb;
    *ia = ib;
    *sa = sb;
    return;
  }
  if (better(vb, ib, *va, *ia)) {
    const float sc = (*va == vb) ? 1.0f : __builtin_expf(*va - vb);
    *sa = sb + (*sa) * sc;
    *va = vb;
    *ia = ib;
  } else {
    const float sc = (vb == *va) ? 1.0f : __builtin_expf(vb - *va);
    *sa = (*sa) + sb * sc;
  }
}

/* Wave-wide (32-lane) online (max, argmax, sum_exp) reduction via XOR bpermute (0 LDS). */
static inline __attribute__((always_inline)) void ink_r5_am_wave_reduce(
    float *bv, u32 *bi, float *se, u32 l) {
  float v = *bv, s = *se;
  u32 i = *bi;
#pragma unroll
  for (u32 m = 16u; m > 0u; m >>= 1u) {
    const int addr = (int)((l ^ m) * 4u);
    const float ov = __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(addr, __builtin_bit_cast(int, v)));
    const u32 oi = (u32)__builtin_amdgcn_ds_bpermute(addr, (int)i);
    const float os = __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(addr, __builtin_bit_cast(int, s)));
    ink_r5_am_merge(&v, &i, &s, ov, oi, os);
  }
  *bv = v;
  *bi = i;
  *se = s;
}

/* ============================================================================
 * 1. ie_dspark_block_init_zsync_r5
 * ========================================================================== */
KERNEL ie_dspark_block_init_zsync_r5(const G u8 *emb_qw, const G u16 *emb_qs, const G float *emb_norm_w,
                                     const G float *mask_embed, G float *block_out, G float *ds_state,
                                     G u32 *ds_ctrl, u32 anchor_id, u32 dim, u32 nb, float eps,
                                     u32 block_size, u32 rank) {
  const u32 t = tokid();
  if (t >= block_size || dim == 0u) return;

  const u32 th = tid();
  G float *dst = block_out + (unsigned long)t * (unsigned long)dim;

  /* t >= 1: vectorized copy of mask_embed -> block_out[t, 0..dim-1]. */
  if (t >= 1u) {
    if ((dim & 3u) == 0u) {
      const u32 n4 = dim >> 2u;
      const G f32x4 *src4 = (const G f32x4 *)mask_embed;
      G f32x4 *dst4 = (G f32x4 *)dst;
      for (u32 i = th; i < n4; i += NT) dst4[i] = src4[i];
    } else {
      for (u32 i = th; i < dim; i += NT) dst[i] = mask_embed[i];
    }
    return;
  }

  /* t == 0: zero ds_state[0..rank-1] and initialize ds_ctrl[0..3] on device. */
  if (ds_state) {
    for (u32 i = th; i < rank; i += NT) ds_state[i] = 0.0f;
  }
  if (ds_ctrl && th == 0u) {
    ds_ctrl[0] = anchor_id;
    ds_ctrl[1] = 0u;
    ds_ctrl[2] = 0u;
    ds_ctrl[3] = 0u;
  }

  /* t == 0: dequantize row anchor_id of Q8_0 token_embd + optional RMSNorm. */
  const u32 wv = th >> 5u, l = lane();
  const G u8 *row_q = emb_qw + (unsigned long)anchor_id * (unsigned long)nb * 32ul;
  const G u16 *row_s = emb_qs + (unsigned long)anchor_id * (unsigned long)nb;

  if (dim == INK_R5_DIM_FAST && nb == INK_R5_NB_FAST) {
    const G u32 *row_qw = (const G u32 *)row_q;
    const u32 w0 = row_qw[th];
    const u32 w1 = row_qw[256u + th];
    const u16 s0_raw = row_s[th >> 3u];
    const u16 s1_raw = row_s[32u + (th >> 3u)];
    const u16 s2_raw = row_s[64u + wv];
    const float sc0 = (float)__builtin_bit_cast(f16, s0_raw);
    const float sc1 = (float)__builtin_bit_cast(f16, s1_raw);
    const float sc2 = (float)__builtin_bit_cast(f16, s2_raw);

    const f32x4 x0 = {
        sc0 * (float)(i8)(w0 & 255u),
        sc0 * (float)(i8)((w0 >> 8u) & 255u),
        sc0 * (float)(i8)((w0 >> 16u) & 255u),
        sc0 * (float)(i8)(w0 >> 24u),
    };
    const f32x4 x1 = {
        sc1 * (float)(i8)(w1 & 255u),
        sc1 * (float)(i8)((w1 >> 8u) & 255u),
        sc1 * (float)(i8)((w1 >> 16u) & 255u),
        sc1 * (float)(i8)(w1 >> 24u),
    };
    const float x2 = sc2 * (float)(i8)row_q[2048u + th];

    const u32 off0 = th * 4u;
    const u32 off1 = 1024u + th * 4u;
    const u32 off2 = 2048u + th;

    if (!emb_norm_w) {
      *(G f32x4 *)(dst + off0) = x0;
      *(G f32x4 *)(dst + off1) = x1;
      dst[off2] = x2;
      return;
    }

    float ss = (x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w);
    ss += (x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w);
    ss += x2 * x2;
    ss = ink_r5_wg_sum_1b(ss, wv, l);
    const float s = 1.0f / __builtin_sqrtf(ss / 2304.0f + eps);

    *(G f32x4 *)(dst + off0) = x0 * s * (*(const G f32x4 *)(emb_norm_w + off0));
    *(G f32x4 *)(dst + off1) = x1 * s * (*(const G f32x4 *)(emb_norm_w + off1));
    dst[off2] = x2 * s * emb_norm_w[off2];
    return;
  }

  /* General path for arbitrary dim / nb. */
  if (!emb_norm_w) {
    for (u32 i = th; i < dim; i += NT) {
      const u16 s_raw = row_s[i >> 5u];
      const float sc = (float)__builtin_bit_cast(f16, s_raw);
      dst[i] = sc * (float)(i8)row_q[i];
    }
    return;
  }

  float ss = 0.0f;
  for (u32 i = th; i < dim; i += NT) {
    const u16 s_raw = row_s[i >> 5u];
    const float sc = (float)__builtin_bit_cast(f16, s_raw);
    const float v = sc * (float)(i8)row_q[i];
    dst[i] = v;
    ss += v * v;
  }
  ss = ink_r5_wg_sum_1b(ss, wv, l);
  const float s = 1.0f / __builtin_sqrtf(ss / (float)dim + eps);
  for (u32 i = th; i < dim; i += NT) dst[i] = dst[i] * s * emb_norm_w[i];
}

/* ============================================================================
 * 2. ie_dspark_concat_q8_zsync_r5
 * ========================================================================== */
KERNEL ie_dspark_concat_q8_zsync_r5(const G float *state, const G u8 *w1_qw, const G u16 *w1_qs,
                                    const G float *hidden_i, const G float *f_rnn, G u8 *qout,
                                    const G u32 *ds_ctrl, u32 rank, u32 dim) {
  if (!ds_ctrl || ds_ctrl[1] != 0u) return;
  if ((rank & 31u) != 0u || (dim & 31u) != 0u) return;

  const u32 total = 2u * rank + 2u * dim;
  const u32 nb = total >> 5u;
  const u32 b = wgid() * 8u + (tid() >> 5u);
  if (b >= nb) return; /* whole waves exit together */

  const u32 prev_tok = ds_ctrl[0];
  const u32 l = lane();
  const u32 idx = b * 32u + l;
  float v;
  if (idx < rank) {
    v = state[idx];
  } else if (idx < 2u * rank) {
    const u32 w1_nb = rank >> 5u;
    const u32 wb = b - w1_nb;
    const unsigned long blk = (unsigned long)prev_tok * (unsigned long)w1_nb + (unsigned long)wb;
    const u16 s_raw = w1_qs[blk];
    const float sc = (float)__builtin_bit_cast(f16, s_raw);
    v = sc * (float)(i8)w1_qw[blk * 32ul + (unsigned long)l];
  } else if (idx < 2u * rank + dim) {
    v = hidden_i[idx - 2u * rank];
  } else {
    v = f_rnn[idx - (2u * rank + dim)];
  }
  quant_wave(v, b, qout, nb);
}

/* ============================================================================
 * 3. ie_dspark_rnn_step_zsync_r5
 * ========================================================================== */
KERNEL ie_dspark_rnn_step_zsync_r5(const G float *raw768, G float *state, G float *latent,
                                   G u8 *latent_q8, const G u32 *ds_ctrl, u32 rank) {
  if (ds_ctrl && ds_ctrl[1] != 0u) return;
  if (rank == 0u || (rank & 31u) != 0u) return;

  const u32 nb = rank >> 5u;
  const u32 wv = tid() >> 5u, l = lane();
  for (u32 b = wv; b < nb; b += 8u) {
    const u32 i = b * 32u + l;
    const float g = 1.0f / (1.0f + __builtin_expf(-raw768[i]));
    const float cand = ie_tanhf_s(raw768[rank + i]);
    const float s = g * state[i] + (1.0f - g) * cand;
    const float lat = ie_tanhf_s(raw768[2u * rank + i]);
    state[i] = s;
    if (latent) latent[i] = lat;
    if (latent_q8) quant_wave(lat, b, latent_q8, nb);
  }
}

/* ============================================================================
 * 4. ie_dspark_add_q8_zsync_r5
 * ========================================================================== */
KERNEL ie_dspark_add_q8_zsync_r5(const G float *hidden_i, const G float *delta, const G float *bias,
                                 G u8 *qout, const G u32 *ds_ctrl, u32 dim) {
  if (ds_ctrl && ds_ctrl[1] != 0u) return;
  if (dim == 0u || (dim & 31u) != 0u) return;

  const u32 nb = dim >> 5u;
  const u32 b = wgid() * 8u + (tid() >> 5u);
  if (b >= nb) return; /* whole waves exit together */

  const u32 j = b * 32u + lane();
  const float v = hidden_i[j] + delta[j] + (bias ? bias[j] : 0.0f);
  quant_wave(v, b, qout, nb);
}

/* ============================================================================
 * 5. ie_dspark_gemv_p_q4_zsync_r5
 * Q4_0 P matrix GEMV + base_row residual: 4 rows/wave, 8 lanes/row (32 rows/WG).
 * For nb==8 (rank=256): 1 lane/block, 100% lane utilization, 512 B coalesced
 * wave reads, cached in Infinity Cache (L3), 0 LDS, bitwise-exact vs ie_gemv_q4q8.
 * ========================================================================== */
KERNEL ie_dspark_gemv_p_q4_zsync_r5(const G u32 *qw, const G f16 *qs, const G u8 *xq,
                                    const G float *base_row, G float *logits, u32 rows, u32 nb,
                                    const G u32 *ds_ctrl) {
#pragma clang fp contract(off)
  if (ds_ctrl && ds_ctrl[1] != 0u) return;
  if (rows == 0u || nb == 0u) return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 r = (wgid() << 5u) + (w << 2u) + (l >> 3u);
  if (r >= rows) return;

  const G u32 *row_qw = qw + (unsigned long)r * (unsigned long)nb * 4ul;
  const G f16 *row_qs = qs + (unsigned long)r * (unsigned long)nb;
  const G u32 *aw = (const G u32 *)xq;
  const G float *da = (const G float *)(xq + (unsigned long)nb * 32ul);
  const G u32 *asum = (const G u32 *)(xq + (unsigned long)nb * 36ul);

  float s;
  if (nb == 8u) {
    s = 0.0f + ink_r5_q4q8_blk(row_qw, row_qs, aw, da, asum, sub);
  } else {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (u32 k0 = 0u; k0 < nb; k0 += 32u) {
      const u32 b0 = k0 + sub;
      if (b0 < nb) a0 = a0 + ink_r5_q4q8_blk(row_qw, row_qs, aw, da, asum, b0);
      const u32 b1 = b0 + 8u;
      if (b1 < nb) a1 = a1 + ink_r5_q4q8_blk(row_qw, row_qs, aw, da, asum, b1);
      const u32 b2 = b0 + 16u;
      if (b2 < nb) a2 = a2 + ink_r5_q4q8_blk(row_qw, row_qs, aw, da, asum, b2);
      const u32 b3 = b0 + 24u;
      if (b3 < nb) a3 = a3 + ink_r5_q4q8_blk(row_qw, row_qs, aw, da, asum, b3);
    }
    const float u0 = a0 + a2;
    const float u1 = a1 + a3;
    s = u0 + u1;
  }

  /* Exact wave_tree_f steps (+4, +2, +1) inside each 8-lane row group. */
  s = s + I2F(shl_i(F2I(s), 4));
  s = s + I2F(shl_i(F2I(s), 2));
  s = s + I2F(shl_i(F2I(s), 1));
  if (sub == 0u) logits[r] = base_row ? (base_row[r] + s) : s;
}

/* ============================================================================
 * 6. ie_dspark_cap_argmax_zsync_r5
 * Single-pass fused logit_cap + argmax + online softmax probability + device
 * control-block update (grid (128u, 1u, 1u), 100 B LDS, 0 logits writeback).
 * ========================================================================== */
static LDS float ink_r5_am_wv[8];
static LDS u32 ink_r5_am_wi[8];
static LDS float ink_r5_am_ws[8];
static LDS u32 ink_r5_am_last;

KERNEL ie_dspark_cap_argmax_zsync_r5(const G float *logits, G float *part_max, G u32 *part_id,
                                     G float *part_sum, G u32 *count, G u32 *ds_ctrl,
                                     G u32 *draft_ids, G float *draft_probs,
                                     u32 V, float cap, float pmin, u32 step_idx) {
  if (!ds_ctrl || ds_ctrl[1] != 0u) return;
  if (V == 0u || !logits || !part_max || !part_id || !part_sum || !count) return;

  const u32 g = wgid();
  const u32 ng = ngroups_x();
  const u32 th = tid();
  const u32 w = th >> 5u;
  const u32 l = lane();
  const u32 stride = ng * NT;

  float bv = -__builtin_inff();
  u32 bi = 0xFFFFFFFFu;
  float se = 0.0f;

  if (cap > 0.0f) {
    for (u32 i = g * NT + th; i < V; i += stride) {
      const float x = cap * ie_tanhf_s(logits[i] / cap);
      if (better(x, i, bv, bi)) {
        if (se > 0.0f) se = (bv == x ? se : se * __builtin_expf(bv - x)) + 1.0f;
        else se = 1.0f;
        bv = x;
        bi = i;
      } else {
        se += (x == bv) ? 1.0f : __builtin_expf(x - bv);
      }
    }
  } else {
    for (u32 i = g * NT + th; i < V; i += stride) {
      const float x = logits[i];
      if (better(x, i, bv, bi)) {
        if (se > 0.0f) se = (bv == x ? se : se * __builtin_expf(bv - x)) + 1.0f;
        else se = 1.0f;
        bv = x;
        bi = i;
      } else {
        se += (x == bv) ? 1.0f : __builtin_expf(x - bv);
      }
    }
  }

  /* Wave reduction (32 lanes -> lane 0) then 8-wave WG merge via 96 B LDS. */
  ink_r5_am_wave_reduce(&bv, &bi, &se, l);
  if (l == 0u) {
    ink_r5_am_wv[w] = bv;
    ink_r5_am_wi[w] = bi;
    ink_r5_am_ws[w] = se;
  }
  lbarrier();

  if (th == 0u) {
    float wg_v = ink_r5_am_wv[0];
    u32 wg_i = ink_r5_am_wi[0];
    float wg_s = ink_r5_am_ws[0];
#pragma unroll
    for (u32 k = 1u; k < 8u; k++) {
      ink_r5_am_merge(&wg_v, &wg_i, &wg_s, ink_r5_am_wv[k], ink_r5_am_wi[k], ink_r5_am_ws[k]);
    }
    const u32 wg_vu = __builtin_bit_cast(u32, wg_v);
    const u32 wg_su = __builtin_bit_cast(u32, wg_s);
    __scoped_atomic_store_n((G u32 *)(part_max + g), wg_vu, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    __scoped_atomic_store_n(part_id + g, wg_i, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    __scoped_atomic_store_n((G u32 *)(part_sum + g), wg_su, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    ink_r5_am_last = (__scoped_atomic_fetch_add(count, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r5_am_last) return;

  /* Last arriving WG merges all ng partials and updates device control block. */
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  bv = -__builtin_inff();
  bi = 0xFFFFFFFFu;
  se = 0.0f;
  for (u32 pg = th; pg < ng; pg += NT) {
    const u32 vu = __scoped_atomic_load_n((G u32 *)(part_max + pg), __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    const u32 pi = __scoped_atomic_load_n(part_id + pg, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    const u32 su = __scoped_atomic_load_n((G u32 *)(part_sum + pg), __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    ink_r5_am_merge(&bv, &bi, &se, __builtin_bit_cast(float, vu), pi, __builtin_bit_cast(float, su));
  }

  ink_r5_am_wave_reduce(&bv, &bi, &se, l);
  if (l == 0u) {
    ink_r5_am_wv[w] = bv;
    ink_r5_am_wi[w] = bi;
    ink_r5_am_ws[w] = se;
  }
  lbarrier();

  if (th == 0u) {
    float best_v = ink_r5_am_wv[0];
    u32 best_id = ink_r5_am_wi[0];
    float sum_exp = ink_r5_am_ws[0];
#pragma unroll
    for (u32 k = 1u; k < 8u; k++) {
      ink_r5_am_merge(&best_v, &best_id, &sum_exp, ink_r5_am_wv[k], ink_r5_am_wi[k], ink_r5_am_ws[k]);
    }
    const float best_prob = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    if (draft_ids) draft_ids[step_idx] = best_id;
    if (draft_probs) draft_probs[step_idx] = best_prob;
    ds_ctrl[0] = best_id;
    ds_ctrl[2] = step_idx + 1u;
    if (best_prob < pmin) ds_ctrl[1] = 1u;
    __scoped_atomic_store_n(count, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  }
}

/* ============================================================================
 * 7. Compact Guarded T=1 GEMV Wrappers for Zero-Sync Markov Loop
 * Allow rnn_proj, hidden_out, and unsplit lm_head (260.5 MB) to early-exit
 * immediately once ds_ctrl[1] != 0u (exact ie_gemv_* arithmetic, 0 LDS).
 * ========================================================================== */
KERNEL ie_dspark_gemv_q8_zsync_r5(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                  u32 rows, u32 nb, const G float *bias, const G float *res,
                                  const G u32 *ds_ctrl) {
  if (ds_ctrl && ds_ctrl[1] != 0u) return;
  gemv_q8q8_rows(wgid(), qw, qs, xq, y, rows, nb, bias, res);
}

KERNEL ie_dspark_gemv_q4k_zsync_r5(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y,
                                   u32 rows, u32 nb, const G float *bias, const G float *res,
                                   const G u32 *ds_ctrl) {
  if (ds_ctrl && ds_ctrl[1] != 0u) return;
  gemv_q4kq8_rows(wgid(), qw, qs, xq, y, rows, nb, bias, res);
}

KERNEL ie_dspark_gemv_q6k_zsync_r5(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y,
                                   u32 rows, u32 nb, const G float *bias, const G float *res,
                                   const G u32 *ds_ctrl) {
  if (ds_ctrl && ds_ctrl[1] != 0u) return;
  gemv_q6kq8_rows(wgid(), qw, qs, xq, y, rows, nb, bias, res);
}

/* ============================================================================
 * 8. ie_inkling_gemv_q8_small_ver_r5
 * High-occupancy small-row (rows <= 128) multi-token (T = 2..16) Q8_0 verify
 * GEMV: 1 WG (8 waves) per output row. Loads each weight block ONCE per chunk
 * of <= 8 tokens and dots against all tokens in the chunk; all <= 8 waves then
 * reduce distinct tokens in parallel via 8 KiB conflict-free LDS.
 * Preserves exact ie_gemv_q8q8_ts / ie_gemv_q8q8_t per-lane block order and
 * wave_tree_f reduction (0-bit-diff).
 * ========================================================================== */
static LDS float ink_r5_qv_lds[INK_R5_VER_CHUNK][8u][32u] __attribute__((aligned(16)));

KERNEL ie_inkling_gemv_q8_small_ver_r5(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                       u32 rows, u32 nb, const G float *bias, const G float *res,
                                       u32 T, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  _Static_assert(sizeof(ink_r5_qv_lds) == 8192u, "ink_r5_qv_lds must be 8 KiB");
  if (T == 0u || rows == 0u || nb == 0u || !qw || !qs || !xq || !y) return;

  const u32 r = wgid();
  if (r >= rows) return;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane();
  const u32 nchunks = (nb + 255u) >> 8u;
  const G u32 *row_qw = qw + (unsigned long)r * (unsigned long)nb * 8ul;
  const G f16 *row_qs = qs + (unsigned long)r * (unsigned long)nb;

  for (u32 t_base = 0u; t_base < T; t_base += INK_R5_VER_CHUNK) {
    const u32 tn = (T - t_base) < INK_R5_VER_CHUNK ? (T - t_base) : INK_R5_VER_CHUNK;
    float acc = 0.0f;

    for (u32 ch = 0u; ch < nchunks; ch++) {
      const u32 base = ch << 8u;
      const u32 b = base + t0;
      if (b < nb) {
        const G u32 *wp = row_qw + (unsigned long)b * 8ul;
        const u32x4 w0 = ink_r5_ld4(wp);
        const u32x4 w1 = ink_r5_ld4(wp + 4u);
        const float sw = (float)row_qs[b];

        u32 j = 0u;
        for (; j + 1u < tn; j += 2u) {
          const G u8 *xj0 = xq + (unsigned long)(t_base + j) * (unsigned long)xs;
          const G u8 *xj1 = xq + (unsigned long)(t_base + j + 1u) * (unsigned long)xs;
          const G u32 *aw0 = (const G u32 *)xj0;
          const G u32 *aw1 = (const G u32 *)xj1;
          const u32x4 a0_0 = ink_r5_ld4(aw0 + (unsigned long)b * 8ul);
          const u32x4 a1_0 = ink_r5_ld4(aw0 + (unsigned long)b * 8ul + 4ul);
          const float da0 = ((const G float *)(xj0 + (unsigned long)nb * 32ul))[b];
          const u32x4 a0_1 = ink_r5_ld4(aw1 + (unsigned long)b * 8ul);
          const u32x4 a1_1 = ink_r5_ld4(aw1 + (unsigned long)b * 8ul + 4ul);
          const float da1 = ((const G float *)(xj1 + (unsigned long)nb * 32ul))[b];
          const float sc0 = sw * da0;
          const float sc1 = sw * da1;
          const int d0 = ink_r5_dot8_ss(w0, w1, a0_0, a1_0);
          const int d1 = ink_r5_dot8_ss(w0, w1, a0_1, a1_1);
          ink_r5_qv_lds[j][w][l] = sc0 * (float)d0;
          ink_r5_qv_lds[j + 1u][w][l] = sc1 * (float)d1;
        }
        if (j < tn) {
          const G u8 *xj0 = xq + (unsigned long)(t_base + j) * (unsigned long)xs;
          const G u32 *aw0 = (const G u32 *)xj0;
          const u32x4 a0_0 = ink_r5_ld4(aw0 + (unsigned long)b * 8ul);
          const u32x4 a1_0 = ink_r5_ld4(aw0 + (unsigned long)b * 8ul + 4ul);
          const float da0 = ((const G float *)(xj0 + (unsigned long)nb * 32ul))[b];
          const float sc0 = sw * da0;
          const int d0 = ink_r5_dot8_ss(w0, w1, a0_0, a1_0);
          ink_r5_qv_lds[j][w][l] = sc0 * (float)d0;
        }
      }

      lbarrier();

      /* Wave w reduces token (t_base + w) in exact ascending block order. */
      if (w < tn) {
        const u32 rem = nb - base;
        const u32 nw_act = rem >= 256u ? 8u : ((rem + 31u) >> 5u);
        for (u32 k = 0u; k < nw_act; k++) {
          if (base + (k << 5u) + l < nb) acc = acc + ink_r5_qv_lds[w][k][l];
        }
      }

      if (ch + 1u < nchunks) lbarrier();
    }

    if (w < tn) {
      const u32 tok = t_base + w;
      acc = wave_tree_f(acc);
      if (l == 0u) {
        const G float *rp = res ? res + (unsigned long)tok * (unsigned long)rs : (const G float *)0;
        y[(unsigned long)tok * (unsigned long)ys + (unsigned long)r] = epilogue(acc, r, bias, rp);
      }
    }

    if (t_base + INK_R5_VER_CHUNK < T) lbarrier();
  }
}

#undef INK_R5_VER_CHUNK
#undef INK_R5_NB_FAST
#undef INK_R5_DIM_FAST

#endif /* !IE_MEGA */
