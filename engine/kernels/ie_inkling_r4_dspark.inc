/* ============================================================================
 * Inkling-2-Mini Round 4 (dspark-kernels-r4): Self-Contained DSpark Speculator
 * GPU Kernels for RX 9070 XT (gfx1201, wave32, NT=256, f16 context KV).
 *
 * All local symbols are prefixed ink_r4_ds_ / INK_R4_DS_, all macros are
 * #undef'd at EOF, Clang 21 __builtin_bit_cast-safe, 0 register spills.
 *
 * Exported Kernels, Grids, Guards, and Parity:
 *
 * 1. KERNEL ie_dspark_dflare_mix_r4(
 *        const G float *f0, const G float *f1, const G float *f2, const G float *f3,
 *        const G float *fuse_w, const G float *norm_w, G float *out, G u8 *qout,
 *        u32 layer_idx, u32 dim, float eps, u32 fs, u32 os, u32 qs, u32 T)
 *    - Grid:   (1u, T, 1u)
 *    - Guard:  T > 0u && dim > 0u && (dim & 31u) == 0u && f0 && f1 && f2 && f3 &&
 *              fuse_w && (out || qout); for dim==2304 fast path, 16-byte aligned
 *              f0..f3, norm_w, out, and (T == 1u || ((fs|os) & 15u) == 0u).
 *    - Parity: f32-tol (<= 1e-6 vs FP32 softmax + mix + RMSNorm + ie_quant_q8;
 *              exact quant_wave Q8_0 output from normalized f32 values).
 *
 * 2. KERNEL ie_dspark_block_init_r4(
 *        const G u8 *emb_qw, const G u16 *emb_qs, const G float *emb_norm_w,
 *        const G float *mask_embed, G float *block_out,
 *        u32 anchor_id, u32 dim, u32 nb, float eps, u32 block_size)
 *    - Grid:   (1u, block_size, 1u)  [block_size=8, dim=2304, nb=72]
 *    - Guard:  block_size > 0u && dim > 0u && nb * 32u >= dim && emb_qw && emb_qs &&
 *              mask_embed && block_out; for dim==2304 fast path, 16-byte aligned
 *              emb_norm_w, mask_embed, block_out (4-byte aligned emb_qw).
 *    - Parity: bitwise-exact vs ie_embed_q8 (when emb_norm_w == NULL) and copy
 *              of mask_embed (t >= 1); f32-tol (<= 1e-6) vs RMSNorm when emb_norm_w != NULL.
 *
 * 3. KERNEL ie_dspark_attn_r4(
 *        const G float *q, const G kvt *kc_ctx, const G kvt *vc_ctx,
 *        const G float *k_blk, const G float *v_blk, G float *out,
 *        u32 ctx_len, u32 kv_cap, u32 win, u32 block_size,
 *        u32 n_head, u32 n_kv, u32 hd, u32 qs, u32 ks, u32 os)
 *    - Grid:   (n_head, block_size, 1u)  [24 x 8 = 192 WGs]
 *    - Guard:  sizeof(kvt) == 2u && n_head > 0u && n_kv > 0u && (n_head % n_kv) == 0u &&
 *              hd > 0u && hd <= 256u && block_size > 0u && (ctx_len == 0u || kv_cap > 0u);
 *              for hd==128 fast path, 16-byte aligned q, k_blk, v_blk, out and
 *              (block_size == 1u || ((qs|ks|os) & 15u) == 0u).
 *    - Parity: f32-tol (<= 1e-5 vs FP32 bidirectional block + windowed/global f16-KV
 *              attention with scale = 1/sqrt(hd), online softmax, 8-wave LDS merge,
 *              LDS = 4,160 B <= 8 KiB).
 *
 * 4. KERNEL ie_dspark_concat_q8_r4(
 *        const G float *state, const G u8 *w1_qw, const G u16 *w1_qs,
 *        const G float *hidden_i, const G float *f_rnn, G u8 *qout,
 *        u32 prev_tok, u32 rank, u32 dim)
 *    - Grid:   ((2u * rank + 2u * dim + 255u) / 256u, 1u, 1u)  [20 WGs for 5120]
 *    - Guard:  (rank & 31u) == 0u && (dim & 31u) == 0u && (rank + dim) > 0u &&
 *              state && w1_qw && w1_qs && hidden_i && f_rnn && qout.
 *    - Parity: bitwise-exact vs concat([state, dequant(w1[prev_tok]), hidden_i, f_rnn])
 *              followed by ie_quant_q8 (0 LDS).
 *
 * 5. KERNEL ie_dspark_rnn_step_r4(
 *        const G float *raw768, G float *state, G float *latent,
 *        G u8 *latent_q8, u32 rank)
 *    - Grid:   (1u, 1u, 1u)  [rank=256, 1 thread/rank]
 *    - Guard:  rank > 0u && (rank & 31u) == 0u && raw768 && state.
 *    - Parity: bitwise-exact vs scalar sigmoid/ie_tanhf_s RNN update + ie_quant_q8 (0 LDS).
 *
 * 6. KERNEL ie_dspark_add_q8_r4(
 *        const G float *hidden_i, const G float *delta, const G float *bias,
 *        G u8 *qout, u32 dim)
 *    - Grid:   ((dim + 255u) / 256u, 1u, 1u)  [9 WGs for dim=2304]
 *    - Guard:  dim > 0u && (dim & 31u) == 0u && hidden_i && delta && qout.
 *    - Parity: bitwise-exact vs (hidden_i[j] + delta[j] + (bias ? bias[j] : 0))
 *              followed by ie_quant_q8 (0 LDS).
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(kvt) == 2u, "ie_dspark_attn_r4 requires f16 KV cache");

#define INK_R4_DS_DIM_FAST 2304u
#define INK_R4_DS_NB_FAST  72u
#define INK_R4_DS_HD_FAST  128u

/* Shared LDS for 1-barrier RMSNorm reduction + wave-private f32x4 quant transpose. */
static LDS float ink_r4_ds_rn_red[8];
static LDS float ink_r4_ds_rn_sh[8][128] __attribute__((aligned(16)));

/* LDS for ie_dspark_attn_r4 8-wave online softmax merge: 4,160 B <= 8 KiB. */
typedef struct {
  float o[8][128]; /* 4,096 B: 8 waves x 128 f32 elements (0 bank conflicts) */
  float m[8];      /*    32 B: per-wave running max */
  float l[8];      /*    32 B: per-wave running exp sum */
} ink_r4_ds_attn_lds_t;

_Static_assert(sizeof(ink_r4_ds_attn_lds_t) <= 8192u, "ink_r4_ds_attn_lds_t must be <= 8 KiB");

/* 1-barrier LDS-only workgroup sum (safe when called once per WG lifetime). */
static inline __attribute__((always_inline)) float ink_r4_ds_wg_sum_1b(float v, u32 wv, u32 l) {
  v = wave_tree_f(v);
  if (l == 0u) ink_r4_ds_rn_red[wv] = v;
  lbarrier();
  float s = 0.0f;
#pragma unroll
  for (u32 w = 0u; w < 8u; w++) s += ink_r4_ds_rn_red[w];
  return s;
}

/* Transpose 4 floats/lane (128 floats = 4 Q8_0 blocks per wave) via wave-private LDS and quantize. */
static inline __attribute__((always_inline)) void ink_r4_ds_quant_chunk4(
    f32x4 y4, u32 wv, u32 l, u32 b0, G u8 *q, u32 nb) {
  *(LDSP f32x4 *)&ink_r4_ds_rn_sh[wv][l * 4u] = y4;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 m = 0u; m < 4u; m++) quant_wave(ink_r4_ds_rn_sh[wv][32u * m + l], b0 + m, q, nb);
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

/* ----------------------------------------------------------------------------
 * 1. ie_dspark_dflare_mix_r4
 * -------------------------------------------------------------------------- */
KERNEL ie_dspark_dflare_mix_r4(const G float *f0, const G float *f1, const G float *f2, const G float *f3,
                               const G float *fuse_w, const G float *norm_w, G float *out, G u8 *qout,
                               u32 layer_idx, u32 dim, float eps, u32 fs, u32 os, u32 qs, u32 T) {
  const u32 t = tokid();
  if (t >= T || dim == 0u || (dim & 31u) != 0u) return;

  f0 = TOKC(f0, fs);
  f1 = TOKC(f1, fs);
  f2 = TOKC(f2, fs);
  f3 = TOKC(f3, fs);
  out = TOK(out, os);
  qout = TOK(qout, qs);

  /* Uniform FP32 softmax over fuse_w[layer_idx*4 .. layer_idx*4+3]. */
  const G float *fw = fuse_w + (unsigned long)layer_idx * 4ul;
  const float fw0 = fw[0], fw1 = fw[1], fw2 = fw[2], fw3 = fw[3];
  const float mx = __builtin_fmaxf(__builtin_fmaxf(fw0, fw1), __builtin_fmaxf(fw2, fw3));
  const float e0 = __builtin_expf(fw0 - mx);
  const float e1 = __builtin_expf(fw1 - mx);
  const float e2 = __builtin_expf(fw2 - mx);
  const float e3 = __builtin_expf(fw3 - mx);
  const float inv_e = 1.0f / ((e0 + e1) + (e2 + e3));
  const float a0 = e0 * inv_e;
  const float a1 = e1 * inv_e;
  const float a2 = e2 * inv_e;
  const float a3 = e3 * inv_e;

  const u32 th = tid(), wv = th >> 5u, l = lane();

  /* Fast path: dim == 2304 (9 floats/thread in VGPRs, 1 lbarrier(), 0 spills). */
  if (dim == INK_R4_DS_DIM_FAST) {
    const u32 off0 = th * 4u;
    const u32 off1 = 1024u + th * 4u;
    const u32 off2 = 2048u + th;

    const f32x4 m0 = (*(const G f32x4 *)(f0 + off0)) * a0
                   + (*(const G f32x4 *)(f1 + off0)) * a1
                   + (*(const G f32x4 *)(f2 + off0)) * a2
                   + (*(const G f32x4 *)(f3 + off0)) * a3;
    const f32x4 m1 = (*(const G f32x4 *)(f0 + off1)) * a0
                   + (*(const G f32x4 *)(f1 + off1)) * a1
                   + (*(const G f32x4 *)(f2 + off1)) * a2
                   + (*(const G f32x4 *)(f3 + off1)) * a3;
    const float m2 = f0[off2] * a0 + f1[off2] * a1 + f2[off2] * a2 + f3[off2] * a3;

    float ss = (m0.x * m0.x + m0.y * m0.y) + (m0.z * m0.z + m0.w * m0.w);
    ss += (m1.x * m1.x + m1.y * m1.y) + (m1.z * m1.z + m1.w * m1.w);
    ss += m2 * m2;
    ss = ink_r4_ds_wg_sum_1b(ss, wv, l);
    const float s = 1.0f / __builtin_sqrtf(ss / 2304.0f + eps);

    f32x4 y0 = m0 * s;
    f32x4 y1 = m1 * s;
    float y2 = m2 * s;
    if (norm_w) {
      y0 = y0 * (*(const G f32x4 *)(norm_w + off0));
      y1 = y1 * (*(const G f32x4 *)(norm_w + off1));
      y2 = y2 * norm_w[off2];
    }

    if (out) {
      *(G f32x4 *)(out + off0) = y0;
      *(G f32x4 *)(out + off1) = y1;
      out[off2] = y2;
    }
    if (qout) {
      ink_r4_ds_quant_chunk4(y0, wv, l, wv * 4u, qout, INK_R4_DS_NB_FAST);
      ink_r4_ds_quant_chunk4(y1, wv, l, 32u + wv * 4u, qout, INK_R4_DS_NB_FAST);
      quant_wave(y2, 64u + wv, qout, INK_R4_DS_NB_FAST);
    }
    return;
  }

  /* General path for any (dim & 31u) == 0u. */
  float ss = 0.0f;
  for (u32 i = th; i < dim; i += NT) {
    const float m = f0[i] * a0 + f1[i] * a1 + f2[i] * a2 + f3[i] * a3;
    ss += m * m;
  }
  ss = ink_r4_ds_wg_sum_1b(ss, wv, l);
  const float s = 1.0f / __builtin_sqrtf(ss / (float)dim + eps);

  const u32 nb = dim >> 5u;
  for (u32 b = wv; b < nb; b += 8u) {
    const u32 i = b * 32u + l;
    const float m = f0[i] * a0 + f1[i] * a1 + f2[i] * a2 + f3[i] * a3;
    const float y = norm_w ? (m * s * norm_w[i]) : (m * s);
    if (out) out[i] = y;
    if (qout) quant_wave(y, b, qout, nb);
  }
}

/* ----------------------------------------------------------------------------
 * 2. ie_dspark_block_init_r4
 * -------------------------------------------------------------------------- */
KERNEL ie_dspark_block_init_r4(const G u8 *emb_qw, const G u16 *emb_qs, const G float *emb_norm_w,
                               const G float *mask_embed, G float *block_out, u32 anchor_id, u32 dim,
                               u32 nb, float eps, u32 block_size) {
  const u32 t = tokid();
  if (t >= block_size || dim == 0u) return;

  const u32 th = tid();
  G float *dst = block_out + (unsigned long)t * dim;

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

  /* t == 0: dequantize row anchor_id of Q8_0 token_embd + optional RMSNorm. */
  const u32 wv = th >> 5u, l = lane();
  const G u8 *row_q = emb_qw + (unsigned long)anchor_id * nb * 32ul;
  const G u16 *row_s = emb_qs + (unsigned long)anchor_id * nb;

  if (dim == INK_R4_DS_DIM_FAST && nb == INK_R4_DS_NB_FAST) {
    const G u32 *row_qw = (const G u32 *)row_q;
    const u32 w0 = row_qw[th];
    const u32 w1 = row_qw[256u + th];
    const float sc0 = (float)__builtin_bit_cast(f16, row_s[th >> 3u]);
    const float sc1 = (float)__builtin_bit_cast(f16, row_s[32u + (th >> 3u)]);
    const float sc2 = (float)__builtin_bit_cast(f16, row_s[64u + wv]);

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
    ss = ink_r4_ds_wg_sum_1b(ss, wv, l);
    const float s = 1.0f / __builtin_sqrtf(ss / 2304.0f + eps);

    *(G f32x4 *)(dst + off0) = x0 * s * (*(const G f32x4 *)(emb_norm_w + off0));
    *(G f32x4 *)(dst + off1) = x1 * s * (*(const G f32x4 *)(emb_norm_w + off1));
    dst[off2] = x2 * s * emb_norm_w[off2];
    return;
  }

  /* General path for arbitrary dim / nb. */
  if (!emb_norm_w) {
    for (u32 i = th; i < dim; i += NT) {
      const float sc = (float)__builtin_bit_cast(f16, row_s[i >> 5u]);
      dst[i] = sc * (float)(i8)row_q[i];
    }
    return;
  }

  float ss = 0.0f;
  for (u32 i = th; i < dim; i += NT) {
    const float sc = (float)__builtin_bit_cast(f16, row_s[i >> 5u]);
    const float v = sc * (float)(i8)row_q[i];
    dst[i] = v;
    ss += v * v;
  }
  ss = ink_r4_ds_wg_sum_1b(ss, wv, l);
  const float s = 1.0f / __builtin_sqrtf(ss / (float)dim + eps);
  for (u32 i = th; i < dim; i += NT) dst[i] = dst[i] * s * emb_norm_w[i];
}

/* ----------------------------------------------------------------------------
 * 3. ie_dspark_attn_r4
 * -------------------------------------------------------------------------- */
KERNEL ie_dspark_attn_r4(const G float *q, const G kvt *kc_ctx, const G kvt *vc_ctx,
                         const G float *k_blk, const G float *v_blk, G float *out,
                         u32 ctx_len, u32 kv_cap, u32 win, u32 block_size,
                         u32 n_head, u32 n_kv, u32 hd, u32 qs, u32 ks, u32 os) {
  static LDS ink_r4_ds_attn_lds_t ink_r4_ds_attn_lds __attribute__((aligned(16)));

  const u32 h = wgid();
  const u32 t = tokid();
  if (h >= n_head || t >= block_size || n_kv == 0u || hd == 0u || hd > 256u) return;

  const u32 kh = h / (n_head / n_kv);
  const u32 kvd = n_kv * hd;
  const u32 th = tid(), w = th >> 5u, l = lane();
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  const u32 start = (win > 0u && ctx_len > win) ? (ctx_len - win) : 0u;

  const G float *qp = (const G float *)((const G u8 *)q + (unsigned long)t * qs) + (unsigned long)h * hd;
  G float *op = (G float *)((G u8 *)out + (unsigned long)t * os) + (unsigned long)h * hd;

  /* Fast path: hd == 128 (4 floats/lane in f32x4 VGPRs, vectorized kv4 f16x4 loads, 1 lbarrier()). */
  if (hd == INK_R4_DS_HD_FAST) {
    const f32x4 qv = *(const G f32x4 *)(qp + 4u * l);
    f32x4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    float m = -__builtin_inff(), sum = 0.0f;

    /* 1) Context KV: p in [start .. ctx_len-1], ring-indexed % kv_cap, 8 waves strided. */
    if (kv_cap > 0u) {
      for (u32 p = start + w; p < ctx_len; p += 8u) {
        const u32 kvp = p % kv_cap;
        const unsigned long row_off = (unsigned long)kvp * kvd + (unsigned long)kh * INK_R4_DS_HD_FAST;
        const f32x4 kv = KV4(kc_ctx + row_off, l);
        float d = (qv.x * kv.x + qv.y * kv.y) + (qv.z * kv.z + qv.w * kv.w);
        d = wave_sum_all(d) * scale;
        const float mn = __builtin_fmaxf(m, d);
        const float c = sum > 0.0f ? __builtin_expf(m - mn) : 0.0f;
        const float e = __builtin_expf(d - mn);
        const f32x4 vv = KV4(vc_ctx + row_off, l);
        sum = sum * c + e;
        acc = acc * c + vv * e;
        m = mn;
      }
    }

    /* 2) Block KV: ALL block_size keys b in 0..block_size-1 (bidirectional, no causal mask). */
    for (u32 b = w; b < block_size; b += 8u) {
      const G float *kbp = (const G float *)((const G u8 *)k_blk + (unsigned long)b * ks)
                         + (unsigned long)kh * INK_R4_DS_HD_FAST;
      const G float *vbp = (const G float *)((const G u8 *)v_blk + (unsigned long)b * ks)
                         + (unsigned long)kh * INK_R4_DS_HD_FAST;
      const f32x4 kv = *(const G f32x4 *)(kbp + 4u * l);
      float d = (qv.x * kv.x + qv.y * kv.y) + (qv.z * kv.z + qv.w * kv.w);
      d = wave_sum_all(d) * scale;
      const float mn = __builtin_fmaxf(m, d);
      const float c = sum > 0.0f ? __builtin_expf(m - mn) : 0.0f;
      const float e = __builtin_expf(d - mn);
      const f32x4 vv = *(const G f32x4 *)(vbp + 4u * l);
      sum = sum * c + e;
      acc = acc * c + vv * e;
      m = mn;
    }

    /* 3) Final 8-wave online softmax merge via 4,160 B LDS (1 lbarrier()). */
    if (l == 0u) {
      ink_r4_ds_attn_lds.m[w] = m;
      ink_r4_ds_attn_lds.l[w] = sum;
    }
    *(LDSP f32x4 *)(ink_r4_ds_attn_lds.o[w] + 4u * l) = acc;
    lbarrier();

    if (th < INK_R4_DS_HD_FAST) {
      float M = -__builtin_inff();
#pragma unroll
      for (u32 x = 0u; x < 8u; x++)
        if (ink_r4_ds_attn_lds.l[x] > 0.0f) M = __builtin_fmaxf(M, ink_r4_ds_attn_lds.m[x]);
      float L = 0.0f, o = 0.0f;
#pragma unroll
      for (u32 x = 0u; x < 8u; x++) {
        if (ink_r4_ds_attn_lds.l[x] > 0.0f) {
          const float wt = __builtin_expf(ink_r4_ds_attn_lds.m[x] - M);
          L += ink_r4_ds_attn_lds.l[x] * wt;
          o += ink_r4_ds_attn_lds.o[x][th] * wt;
        }
      }
      op[th] = (L > 0.0f) ? (o / L) : 0.0f;
    }
    return;
  }

  /* General fallback for any 1 <= hd <= 256 (stays within 4,160 B LDS). */
  float qv[8], acc[8];
#pragma unroll
  for (u32 j = 0u; j < 8u; j++) {
    const u32 i = l + 32u * j;
    qv[j] = (i < hd) ? qp[i] : 0.0f;
    acc[j] = 0.0f;
  }
  float m = -__builtin_inff(), sum = 0.0f;

  if (kv_cap > 0u) {
    for (u32 p = start + w; p < ctx_len; p += 8u) {
      const u32 kvp = p % kv_cap;
      const G kvt *kr = kc_ctx + (unsigned long)kvp * kvd + (unsigned long)kh * hd;
      const G kvt *vr = vc_ctx + (unsigned long)kvp * kvd + (unsigned long)kh * hd;
      float d = 0.0f;
#pragma unroll
      for (u32 j = 0u; j < 8u; j++) {
        const u32 i = l + 32u * j;
        if (i < hd) d += qv[j] * (float)kr[i];
      }
      d = wave_sum_all(d) * scale;
      const float mn = __builtin_fmaxf(m, d);
      const float c = sum > 0.0f ? __builtin_expf(m - mn) : 0.0f;
      const float e = __builtin_expf(d - mn);
      sum = sum * c + e;
#pragma unroll
      for (u32 j = 0u; j < 8u; j++) {
        const u32 i = l + 32u * j;
        if (i < hd) acc[j] = acc[j] * c + e * (float)vr[i];
      }
      m = mn;
    }
  }

  for (u32 b = w; b < block_size; b += 8u) {
    const G float *kbp = (const G float *)((const G u8 *)k_blk + (unsigned long)b * ks) + (unsigned long)kh * hd;
    const G float *vbp = (const G float *)((const G u8 *)v_blk + (unsigned long)b * ks) + (unsigned long)kh * hd;
    float d = 0.0f;
#pragma unroll
    for (u32 j = 0u; j < 8u; j++) {
      const u32 i = l + 32u * j;
      if (i < hd) d += qv[j] * kbp[i];
    }
    d = wave_sum_all(d) * scale;
    const float mn = __builtin_fmaxf(m, d);
    const float c = sum > 0.0f ? __builtin_expf(m - mn) : 0.0f;
    const float e = __builtin_expf(d - mn);
    sum = sum * c + e;
#pragma unroll
    for (u32 j = 0u; j < 8u; j++) {
      const u32 i = l + 32u * j;
      if (i < hd) acc[j] = acc[j] * c + e * vbp[i];
    }
    m = mn;
  }

  if (l == 0u) {
    ink_r4_ds_attn_lds.m[w] = m;
    ink_r4_ds_attn_lds.l[w] = sum;
  }
  const u32 hd_lo = hd < 128u ? hd : 128u;
#pragma unroll
  for (u32 j = 0u; j < 4u; j++) {
    const u32 i = l + 32u * j;
    if (i < hd_lo) ink_r4_ds_attn_lds.o[w][i] = acc[j];
  }
  lbarrier();

  float M = -__builtin_inff();
#pragma unroll
  for (u32 x = 0u; x < 8u; x++)
    if (ink_r4_ds_attn_lds.l[x] > 0.0f) M = __builtin_fmaxf(M, ink_r4_ds_attn_lds.m[x]);
  float L = 0.0f;
#pragma unroll
  for (u32 x = 0u; x < 8u; x++)
    if (ink_r4_ds_attn_lds.l[x] > 0.0f) L += ink_r4_ds_attn_lds.l[x] * __builtin_expf(ink_r4_ds_attn_lds.m[x] - M);
  const float inv_L = (L > 0.0f) ? (1.0f / L) : 0.0f;

  if (th < hd_lo) {
    float o = 0.0f;
#pragma unroll
    for (u32 x = 0u; x < 8u; x++)
      if (ink_r4_ds_attn_lds.l[x] > 0.0f) o += ink_r4_ds_attn_lds.o[x][th] * __builtin_expf(ink_r4_ds_attn_lds.m[x] - M);
    op[th] = o * inv_L;
  }

  if (hd > 128u) {
    lbarrier();
    const u32 hd_hi = hd - 128u;
#pragma unroll
    for (u32 j = 0u; j < 4u; j++) {
      const u32 i = l + 32u * j;
      if (i < hd_hi) ink_r4_ds_attn_lds.o[w][i] = acc[4u + j];
    }
    lbarrier();
    if (th < hd_hi) {
      float o = 0.0f;
#pragma unroll
      for (u32 x = 0u; x < 8u; x++)
        if (ink_r4_ds_attn_lds.l[x] > 0.0f) o += ink_r4_ds_attn_lds.o[x][th] * __builtin_expf(ink_r4_ds_attn_lds.m[x] - M);
      op[128u + th] = o * inv_L;
    }
  }
}

/* ----------------------------------------------------------------------------
 * 4. ie_dspark_concat_q8_r4
 * -------------------------------------------------------------------------- */
KERNEL ie_dspark_concat_q8_r4(const G float *state, const G u8 *w1_qw, const G u16 *w1_qs,
                              const G float *hidden_i, const G float *f_rnn, G u8 *qout,
                              u32 prev_tok, u32 rank, u32 dim) {
  if ((rank & 31u) != 0u || (dim & 31u) != 0u) return;

  const u32 total = 2u * rank + 2u * dim;
  const u32 nb = total >> 5u;
  const u32 b = wgid() * 8u + (tid() >> 5u);
  if (b >= nb) return; /* whole waves exit together */

  const u32 l = lane();
  const u32 idx = b * 32u + l;
  float v;
  if (idx < rank) {
    v = state[idx];
  } else if (idx < 2u * rank) {
    const u32 w1_nb = rank >> 5u;
    const u32 wb = b - w1_nb;
    const unsigned long blk = (unsigned long)prev_tok * w1_nb + wb;
    const float sc = (float)__builtin_bit_cast(f16, w1_qs[blk]);
    v = sc * (float)(i8)w1_qw[blk * 32ul + l];
  } else if (idx < 2u * rank + dim) {
    v = hidden_i[idx - 2u * rank];
  } else {
    v = f_rnn[idx - (2u * rank + dim)];
  }
  quant_wave(v, b, qout, nb);
}

/* ----------------------------------------------------------------------------
 * 5. ie_dspark_rnn_step_r4
 * -------------------------------------------------------------------------- */
KERNEL ie_dspark_rnn_step_r4(const G float *raw768, G float *state, G float *latent,
                             G u8 *latent_q8, u32 rank) {
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

/* ----------------------------------------------------------------------------
 * 6. ie_dspark_add_q8_r4
 * -------------------------------------------------------------------------- */
KERNEL ie_dspark_add_q8_r4(const G float *hidden_i, const G float *delta, const G float *bias,
                           G u8 *qout, u32 dim) {
  if (dim == 0u || (dim & 31u) != 0u) return;

  const u32 nb = dim >> 5u;
  const u32 b = wgid() * 8u + (tid() >> 5u);
  if (b >= nb) return; /* whole waves exit together */

  const u32 j = b * 32u + lane();
  const float v = hidden_i[j] + delta[j] + (bias ? bias[j] : 0.0f);
  quant_wave(v, b, qout, nb);
}

#undef INK_R4_DS_DIM_FAST
#undef INK_R4_DS_NB_FAST
#undef INK_R4_DS_HD_FAST

#endif /* !IE_MEGA */
