/* ie_norm_fuse.inc -- High-occupancy multi-token RMSNorm (D4.2) and single-launch
 * fused MTP prenorm + Q8_0 quantization (D3.2) for gfx1201 (wave32, NT = 256).
 *
 * Included by kernels/ie_kernels.c outside the megakernel (#ifndef IE_MEGA).
 *
 * Kernels provided:
 *   1. ie_rmsnorm_mt      -- Drop-in multi-token replacement for ie_rmsnorm_f /
 *                            ie_rmsnorm_t. Grid: ((n + 1023) / 1024, T) via launch_t.
 *                            Each WG owns 1024 elements (32 Q8_0 blocks, 4 per wave),
 *                            issuing 128-bit f32x4 loads in flight with zero second
 *                            read of x, wave-private LDS staging for quant_wave, and
 *                            full dead_b (y == NULL) support. Cuts redundant ss reads
 *                            4x vs ie_rmsnorm_f while saturating all 64 CUs at T=2..16
 *                            (unlike 1-WG-per-token ie_rmsnorm_t).
 *   2. ie_rmsnorm_res_mt  -- Same 1024-elem/WG multi-token design with optional fused
 *                            residual add (u = res + x -> optional x_out) + RMSNorm +
 *                            optional y + Q8_0 quant_wave.
 *   3. ie_mtp_prenorm_q8  -- Collapses MTP's 4-op / 5..19-launch prenorm sequence
 *                            ([optional OP_EMBED] + enorm(e) + hnorm(h) + concat Q8_0)
 *                            into 1 kernel launch for T = 1..16!
 *                            Grid: (2 * ((dim + 1023) / 1024), T) via launch_t.
 *                            Supports both on-the-fly embedding (tok_kind = 1: MAT_Q4,
 *                            2: MAT_Q8, 3: MAT_F32 for up to 16 token IDs in kernarg)
 *                            and pre-embedded float input e (tok_kind = 0), plus
 *                            cat == NULL and e_out == NULL dead-buffer skipping.
 */

#ifndef IE_MEGA

static LDS float nf_sh[8][128] __attribute__((aligned(16)));

/* Normalize + optional FP32 store + optional Q8_0 quantize (into block range
 * [q_blk_off .. q_blk_off + (n >> 5)) of a Q8_0 buffer with nb_tot total blocks)
 * for workgroup slice g (1024 floats = 32 Q8_0 blocks per WG). */
static inline void nf_rmsnorm_slice_f32(const G float *x, const G float *w, G float *y, u32 n, float eps,
                                        float scale_in, G float *x_copy, G u8 *q, u32 q_blk_off, u32 nb_tot, u32 g) {
  const u32 th = tid(), wv = th >> 5, l = lane();
  const u32 nb = n >> 5, b0 = g * 32u + wv * 4u, base = g * 1024u + th * 4u;
  f32x4 own_x = {0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 own_w = {0.0f, 0.0f, 0.0f, 0.0f};
  float ss = 0.0f;

  if ((n & 1023u) == 0u && n <= 16384u) {
    f32x4 v[16];
#pragma unroll
    for (u32 k = 0; k < 16u; k++)
      v[k] = (k * 1024u < n) ? (*(const G f32x4 *)(x + k * 1024u + th * 4u) * scale_in)
                             : (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    if (base < n) own_w = *(const G f32x4 *)(w + base);
#pragma unroll
    for (u32 k = 0; k < 16u; k++) {
      if (k * 1024u < n) {
        ss += (v[k].x * v[k].x + v[k].y * v[k].y) + (v[k].z * v[k].z + v[k].w * v[k].w);
        if (k == g) own_x = v[k];
      }
    }
  } else {
    u32 k = 0u;
    for (; (k + 1u) * 1024u <= n; k++) {
      const f32x4 vk = *(const G f32x4 *)(x + k * 1024u + th * 4u) * scale_in;
      ss += (vk.x * vk.x + vk.y * vk.y) + (vk.z * vk.z + vk.w * vk.w);
      if (k == g) own_x = vk;
    }
    for (u32 i = k * 1024u + th; i < n; i += NT) {
      const float xi = x[i] * scale_in;
      ss += xi * xi;
    }
    if ((g + 1u) * 1024u <= n) own_w = *(const G f32x4 *)(w + base);
  }

  if (x_copy) {
    if ((g + 1u) * 1024u <= n) {
      *(G f32x4 *)(x_copy + base) = own_x;
    } else {
      for (u32 m = 0; m < 4u; m++) {
        const u32 b = b0 + m;
        if (b < nb) x_copy[b * 32u + l] = x[b * 32u + l] * scale_in;
      }
    }
  }

  ss = wg_sum(ss);
  const float s = 1.0f / __builtin_sqrtf(ss / (float)n + eps);

  if ((g + 1u) * 1024u <= n) {
    const f32x4 y4 = own_x * s * own_w;
    if (y) *(G f32x4 *)(y + base) = y4;
    if (q) {
      *(LDSP f32x4 *)&nf_sh[wv][l * 4u] = y4;
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
      for (u32 m = 0; m < 4u; m++) quant_wave(nf_sh[wv][32u * m + l], q_blk_off + b0 + m, q, nb_tot);
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
    }
  } else {
    for (u32 m = 0; m < 4u; m++) {
      const u32 b = b0 + m;
      if (b < nb) {
        const u32 i = b * 32u + l;
        const float v = (x[i] * scale_in) * s * w[i];
        if (y) y[i] = v;
        if (q) quant_wave(v, q_blk_off + b, q, nb_tot);
      }
    }
  }
}

/* 1. Multi-token vector-loaded RMSNorm (drop-in signature match for ie_rmsnorm_f / ie_rmsnorm_t).
 *    Grid: launch_t(b, K_RMSNORM_MT, (n + 1023u) / 1024u, T, args). */
KERNEL ie_rmsnorm_mt(const G float *x, const G float *w, G float *y, u32 n, float eps, G u8 *q,
                     u32 xs, u32 ys, u32 qs) {
  x = TOKC(x, xs), y = TOK(y, ys), q = TOK(q, qs);
  nf_rmsnorm_slice_f32(x, w, y, n, eps, 1.0f, (G float *)0, q, 0u, n >> 5, wgid());
}

/* 2. Multi-token RMSNorm with optional fused residual add (u = res + x -> optional x_out).
 *    Grid: launch_t(b, K_RMSNORM_RES_MT, (n + 1023u) / 1024u, T, args).
 *    Note: if x_out aliases x or res (in-place residual update), launch with grid_x = 1u;
 *    otherwise launch with grid_x = (n + 1023u) / 1024u. */
KERNEL ie_rmsnorm_res_mt(const G float *x, const G float *res, G float *x_out, const G float *w, G float *y,
                         u32 n, float eps, G u8 *q, u32 xs, u32 rs, u32 xos, u32 ys, u32 qs) {
  x = TOKC(x, xs), res = res ? TOKC(res, rs) : (const G float *)0;
  x_out = TOK(x_out, xos), y = TOK(y, ys), q = TOK(q, qs);
  if (!res) {
    nf_rmsnorm_slice_f32(x, w, y, n, eps, 1.0f, x_out, q, 0u, n >> 5, wgid());
    return;
  }
  const u32 g = wgid(), th = tid(), wv = th >> 5, l = lane();
  const u32 nb = n >> 5;
  if (x_out && (x_out == x || x_out == res)) {
    float ss = 0.0f;
    for (u32 i4 = th * 4u; i4 + 4u <= n; i4 += 1024u) {
      const f32x4 u4 = *(const G f32x4 *)(res + i4) + *(const G f32x4 *)(x + i4);
      if (x_out) *(G f32x4 *)(x_out + i4) = u4;
      ss += (u4.x * u4.x + u4.y * u4.y) + (u4.z * u4.z + u4.w * u4.w);
    }
    for (u32 i = (n & ~1023u) + th; i < n; i += NT) {
      const float ui = res[i] + x[i];
      if (x_out) x_out[i] = ui;
      ss += ui * ui;
    }
    ss = wg_sum(ss);
    const float s = 1.0f / __builtin_sqrtf(ss / (float)n + eps);
    for (u32 b = wv; b < nb; b += 8u) {
      const u32 i = b * 32u + l;
      const float ui = x_out ? x_out[i] : (res[i] + x[i]);
      const float v = ui * s * w[i];
      if (y) y[i] = v;
      if (q) quant_wave(v, b, q, nb);
    }
    return;
  }

  const u32 b0 = g * 32u + wv * 4u, base = g * 1024u + th * 4u;
  f32x4 own_u = {0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 own_w = {0.0f, 0.0f, 0.0f, 0.0f};
  float ss = 0.0f;
  if ((n & 1023u) == 0u && n <= 16384u) {
    f32x4 u_v[16];
#pragma unroll
    for (u32 k = 0; k < 16u; k++)
      u_v[k] = (k * 1024u < n) ? (*(const G f32x4 *)(res + k * 1024u + th * 4u) + *(const G f32x4 *)(x + k * 1024u + th * 4u))
                               : (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    if (base < n) own_w = *(const G f32x4 *)(w + base);
#pragma unroll
    for (u32 k = 0; k < 16u; k++) {
      if (k * 1024u < n) {
        ss += (u_v[k].x * u_v[k].x + u_v[k].y * u_v[k].y) + (u_v[k].z * u_v[k].z + u_v[k].w * u_v[k].w);
        if (k == g) own_u = u_v[k];
      }
    }
  } else {
    u32 k = 0u;
    for (; (k + 1u) * 1024u <= n; k++) {
      const f32x4 uk = *(const G f32x4 *)(res + k * 1024u + th * 4u) + *(const G f32x4 *)(x + k * 1024u + th * 4u);
      ss += (uk.x * uk.x + uk.y * uk.y) + (uk.z * uk.z + uk.w * uk.w);
      if (k == g) own_u = uk;
    }
    for (u32 i = k * 1024u + th; i < n; i += NT) {
      const float ui = res[i] + x[i];
      ss += ui * ui;
    }
    if ((g + 1u) * 1024u <= n) own_w = *(const G f32x4 *)(w + base);
  }

  ss = wg_sum(ss);
  const float s = 1.0f / __builtin_sqrtf(ss / (float)n + eps);

  if ((g + 1u) * 1024u <= n) {
    if (x_out) *(G f32x4 *)(x_out + base) = own_u;
    const f32x4 y4 = own_u * s * own_w;
    if (y) *(G f32x4 *)(y + base) = y4;
    if (q) {
      *(LDSP f32x4 *)&nf_sh[wv][l * 4u] = y4;
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
      for (u32 m = 0; m < 4u; m++) quant_wave(nf_sh[wv][32u * m + l], b0 + m, q, nb);
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
    }
  } else {
    for (u32 m = 0; m < 4u; m++) {
      const u32 b = b0 + m;
      if (b < nb) {
        const u32 i = b * 32u + l;
        const float ui = res[i] + x[i];
        if (x_out) x_out[i] = ui;
        const float v = ui * s * w[i];
        if (y) y[i] = v;
        if (q) quant_wave(v, b, q, nb);
      }
    }
  }
}

static inline float nf_embed_elem(const G u32 *qw, const G f16 *qs, u32 nb, u32 tok, u32 b, u32 j, float sc, u32 kind) {
  if (kind == 1u) { /* MAT_Q4 */
    const u32 w = qw[ie_q4_dst_word(tok, b, ie_q4_word_of(j), nb)];
    return (float)qs[ie_q4_dst_scale(tok, b, nb)] * (float)((int)ie_q4_nib(w, j) - 8) * sc;
  }
  /* kind == 2u: MAT_Q8 */
  const u32 w = qw[ie_q8_dst_word(tok, b, j >> 2, nb)];
  return (float)qs[ie_q8_dst_scale(tok, b, nb)] * (float)(i8)ie_byte(w, j & 3u) * sc;
}

/* 3. Fused MTP prenorm + Q8_0 quantization (1 launch for T = 1..16):
 *    Computes [enorm(embed(tok) or e) | hnorm(h)] into optional FP32 cat (2*dim)
 *    and Q8_0 catq (nb_cat = 2*dim / 32 blocks).
 *    tok_kind:
 *      0 -> read pre-embedded float buffer e (stride es bytes; tp0..tp7 unused)
 *      1 -> fuse OP_EMBED from MAT_Q4 (tw0 = qw, tw1 = qs), optionally write e_out
 *      2 -> fuse OP_EMBED from MAT_Q8 (tw0 = qw, tw1 = qs), optionally write e_out
 *      3 -> fuse OP_EMBED from MAT_F32 (tw0 = float table), optionally write e_out
 *    tp0..tp7: 8 uint64_t values packing up to 16 token IDs:
 *      tp_k = (uint64_t)toks[2*k] | ((uint64_t)toks[2*k + 1] << 32).
 *    Grid: launch_t(b, K_MTP_PRENORM_Q8, 2u * ((dim + 1023u) / 1024u), T, args). */
KERNEL ie_mtp_prenorm_q8(const G float *e, G float *e_out, const G float *h,
                         const G float *ew, const G float *hw,
                         G float *cat, G u8 *catq,
                         const G void *tw0, const G void *tw1,
                         u32 dim, float eps, float escale, u32 tok_kind,
                         u32 es, u32 hs, u32 cs, u32 qs,
                         unsigned long tp0, unsigned long tp1, unsigned long tp2, unsigned long tp3,
                         unsigned long tp4, unsigned long tp5, unsigned long tp6, unsigned long tp7) {
  const u32 ng = (dim + 1023u) / 1024u, wg = wgid();
  if (wg >= 2u * ng || dim == 0u) return;

  const u32 tk = tokid();
  e = e ? TOKC(e, es) : (const G float *)0;
  e_out = TOK(e_out, es);
  h = TOKC(h, hs);
  cat = TOK(cat, cs);
  catq = TOK(catq, qs);

  const u32 nb_half = dim >> 5, nb_cat = dim >> 4;
  if (wg >= ng) {
    /* hnorm(h) -> second half of cat (cat + dim) and blocks [nb_half .. 2*nb_half) of catq */
    const u32 g = wg - ng;
    G float *cat_h = cat ? (cat + dim) : (G float *)0;
    nf_rmsnorm_slice_f32(h, hw, cat_h, dim, eps, 1.0f, (G float *)0, catq, nb_half, nb_cat, g);
    return;
  }

  /* enorm(e or embed(tok)) -> first half of cat and blocks [0 .. nb_half) of catq */
  const u32 g = wg;
  if (tok_kind == 0u) {
    nf_rmsnorm_slice_f32(e, ew, cat, dim, eps, 1.0f, (G float *)0, catq, 0u, nb_cat, g);
    return;
  }

  const u32 pi = tk >> 1;
  const unsigned long tp = pi == 0u ? tp0 : pi == 1u ? tp1 : pi == 2u ? tp2 : pi == 3u ? tp3
                         : pi == 4u ? tp4 : pi == 5u ? tp5 : pi == 6u ? tp6 : tp7;
  const u32 tok = (tk & 1u) ? (u32)(tp >> 32) : (u32)tp;

  if (tok_kind == 3u) { /* MAT_F32 */
    const G float *erow = (const G float *)tw0 + (unsigned long)tok * dim;
    nf_rmsnorm_slice_f32(erow, ew, cat, dim, eps, escale, e_out, catq, 0u, nb_cat, g);
    return;
  }

  /* MAT_Q4 (1) or MAT_Q8 (2): wave-coalesced dequantize + single-pass own-slice capture */
  const G u32 *qw = (const G u32 *)tw0;
  const G f16 *qsc = (const G f16 *)tw1;
  const u32 wv = tid() >> 5, l = lane(), b0 = g * 32u + wv * 4u;
  float own_e[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float ss = 0.0f;

  for (u32 k = 0; k < ng; k++) {
    const u32 bk0 = k * 32u + wv * 4u;
#pragma unroll
    for (u32 m = 0; m < 4u; m++) {
      const u32 b = bk0 + m;
      if (b < nb_half) {
        const float ev = nf_embed_elem(qw, qsc, nb_half, tok, b, l, escale, tok_kind);
        ss += ev * ev;
        if (k == g) own_e[m] = ev;
      }
    }
  }

  ss = wg_sum(ss);
  const float s = 1.0f / __builtin_sqrtf(ss / (float)dim + eps);

#pragma unroll
  for (u32 m = 0; m < 4u; m++) {
    const u32 b = b0 + m;
    if (b < nb_half) {
      const u32 i = b * 32u + l;
      const float ev = own_e[m];
      if (e_out) e_out[i] = ev;
      const float v = ev * s * ew[i];
      if (cat) cat[i] = v;
      if (catq) quant_wave(v, b, catq, nb_cat);
    }
  }
}

#endif /* !IE_MEGA */
