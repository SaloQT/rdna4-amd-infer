/* ============================================================================
 * Inkling-2 Round 2 Group 16: HIP Graph T=1 Decode *_ctx Kernels &
 *                             Batched Prefill Q8_0 Embedding
 *                             for RX 9070 XT (gfx1201, wave32, NT=256).
 *
 * Device context struct layout:
 *   const G u32 *ctx -> ctx[0] = tok, ctx[1] = pos
 *
 * Exported KERNELs, exact grids (gx, gy, 1), dispatch guards, and parity:
 *
 *   [Bit 33] Q8_0 Token Embedding (HIP Graph T=1 ctx + Batched Prefill):
 *   1. KERNEL ie_inkling_embed_q8_ctx_r2(const G u8 *qw, const G u16 *qs,
 *        G float *out, u32 nb, const G u32 *ctx)
 *      Grid:   ((nb + 7u) / 8u, 1, 1)  [1 wave/block of 32, 8 blocks/WG; gx=9 for dim=2304]
 *      Guard:  T == 1u && tm->kind == MAT_Q8 && (o->escale <= 0.0f || o->escale == 1.0f) && nb > 0u
 *      Parity: 100% bitwise-exact vs ie_embed_q8 (with sc == 1.0f).
 *
 *   2. KERNEL ie_inkling_embed_q8_batch_r2(const G u8 *qw, const G u16 *qs,
 *        const G u32 *toks, G float *out, u32 nb, u32 os, u32 T)
 *      Grid:   ((nb + 7u) / 8u, T, 1)  [1 single launch for all T prefill tokens; os = byte stride]
 *      Guard:  T > 1u && tm->kind == MAT_Q8 && (o->escale <= 0.0f || o->escale == 1.0f) && nb > 0u
 *      Parity: 100% bitwise-exact vs T launches of ie_embed_q8 (with sc == 1.0f).
 *
 *   [Bit 34] HIP Graph T=1 Decode Position-Dependent Elementwise/Conv/KV Kernels:
 *   3. KERNEL ie_inkling_vclip_ctx_r2(const G float *a, G float *b, const G float *v,
 *        u32 nh, u32 hd, float clamp, u32 norope, float escale, u32 win, const G u32 *ctx)
 *      Grid:   (gx, 1, 1) where
 *                hd == 128u -> gx = max(1u, min((nh + 7u) / 8u, 256u))   [1 wave/head, 8 heads/WG]
 *                hd == 16u  -> gx = max(1u, min((nh + 63u) / 64u, 256u)) [4 lanes/head, 64 heads/WG]
 *      Guard:  T == 1u && (hd == 128u || hd == 16u) && 16B-aligned a, b(, v)
 *      Parity: 100% bitwise-exact vs ie_inkling_vclip_r1 (f32-tol ~1e-7 vs ie_vclip).
 *
 *   4. KERNEL ie_inkling_sconv_ctx_r2(const G float *a, const G float *v, const G float *c,
 *        G float *ring, G float *b, u32 n, u32 nh, float escale, const G u32 *ctx)
 *      Grid:   (((n >> 2u) + 255u) / 256u, 1, 1)  [gx=3 for dim=2304, gx=1 for kvd=1024]
 *      Guard:  T == 1u && (n & 3u) == 0u && (!c || (nh > 0u && ((n / nh) & 3u) == 0u)) &&
 *              16B-aligned a, v, ring, b(, c)
 *      Parity: 100% bitwise-exact vs ie_sconv / ie_inkling_sconv_r1 (fp contract off, in-place ring update).
 *
 *   5. KERNEL ie_inkling_sconv_add_ctx_r2(const G float *a, const G float *v, const G float *c,
 *        G float *ring, const G float *res, G float *out, u32 n, u32 nh, float escale, const G u32 *ctx)
 *      Grid:   (((n >> 2u) + 255u) / 256u, 1, 1)  [gx=3 for dim=2304]
 *      Guard:  T == 1u && (n & 3u) == 0u && (!c || (nh > 0u && ((n / nh) & 3u) == 0u)) &&
 *              16B-aligned a, v, ring, out(, c, res)
 *      Parity: 100% bitwise-exact vs ie_sconv + ie_add (fp contract off, in-place ring update).
 *
 *   6. KERNEL ie_inkling_moe_sum_sconv_add_ctx_r2(const G float *y, const G u32 *route,
 *        const G float *v, G float *ring, const G float *res, G float *out, u32 n, u32 K,
 *        float escale, const G u32 *ctx)
 *      Grid:   (((n >> 2u) + 255u) / 256u, 1, 1)  [gx=3 for dim=2304]
 *      Guard:  T == 1u && (n & 3u) == 0u && K >= 1u && 16B-aligned y, v, ring, out(, res)
 *      Parity: 100% bitwise-exact vs ie_moe_sum(res=NULL) + ie_sconv(c=NULL) + ie_add(res, fpc)
 *              (fp contract off, exact fmaf order, in-place ring update).
 *
 *   7. KERNEL ie_inkling_kv_store_ctx_r2(const G float *k, const G float *v,
 *        G kvt *kc0, G kvt *vc0, u32 kv_len, u32 kvd, const G u32 *ctx)
 *      Grid:   (((kvd >> 2u) + 255u) / 256u, 1, 1)  [gx=1 for kvd=1024; writes row (ctx[1] % kv_len)]
 *      Guard:  T == 1u && !b->kvq8 && !b->kvq4 && kv_len > 0u && (kvd & 3u) == 0u &&
 *              16B-aligned k, v && 8B-aligned kc0, vc0
 *      Parity: 100% bitwise-exact vs ie_kv_store (f16 KV).
 *
 *   [Bit 35] HIP Graph T=1 Decode GQA-3 f16-KV Relative Attention:
 *   8. KERNEL ie_inkling_rel_attn_dec_ctx_r2(const G float *q, const G kvt *kc, const G kvt *vc,
 *        const G float *R, const G float *rel_bias, const G float *sink, G float *part, G float *out,
 *        u32 kv_len, u32 hd, u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win,
 *        u32 nsplit, G u32 *count, const G u32 *ctx)
 *      Grid:   (n_kv * nsplit, 1, 1)  [nsplit bucketed e.g. {1,4,8,16,32,64}; ch computed on device]
 *      Guard:  T == 1u && !b->kvq8 && !b->kvq4 && hd == 128u && d_rel == 16u &&
 *              n_kv > 0u && n_head == 3u * n_kv && nsplit >= 1u && kv_len > 0u
 *      Split-K: Handles empty tail splits (s * ch >= np) safely: writes M=-1e30f, L=0.0f, O[0..127]=0.0f,
 *               participates in atomicAdd(&count[kh], 1u), merges cleanly, and resets count[kh] = 0u.
 *      Parity: 100% bitwise-exact vs ie_inkling_rel_attn_dec_r1 when nsplit matches (f32-tol vs ie_rel_attn).
 *
 * Included at EOF of kernels/ie_inkling_kernels.c.
 * All local helpers/LDS/macros prefixed ink_r2_g16_ / INK_R2_G16_; macros #undef'd at EOF.
 * ========================================================================== */

#define INK_R2_G16_HD 128u
#define INK_R2_G16_DREL 16u
#define INK_R2_G16_GQA 3u
#define INK_R2_G16_PSTR 130u

/* ----------------------------------------------------------------------------
 * Local Helpers
 * -------------------------------------------------------------------------- */

static inline __attribute__((always_inline)) float ink_r2_g16_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline __attribute__((always_inline)) float ink_r2_g16_vclip_scale(float ss, float inv_hd, float clamp,
                                                                          u32 norope, float escale, u32 win, u32 pos) {
  float f;
  if (clamp > 0.0f) {
    f = clamp / (__builtin_sqrtf(ss) + 1e-6f);
    if (f > 1.0f) f = 1.0f;
    if (norope) {
      const float ratio = (float)(pos + 1u) / (float)win;
      if (ratio > 1.0f) f *= 1.0f + escale * __builtin_logf(ratio);
    }
  } else {
    f = 1.0f / __builtin_sqrtf(ss * inv_hd + escale);
  }
  return f;
}

static inline __attribute__((always_inline)) float ink_r2_g16_halfwave_sum_all(float v) {
  v += I2F(shl_i(F2I(v), 8));
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return I2F(__builtin_amdgcn_readlane(F2I(v), 0));
}

static inline __attribute__((always_inline)) float ink_r2_g16_dot4(f32x4 a, f32x4 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

/* ----------------------------------------------------------------------------
 * [Bit 33] Kernels 1 & 2: Q8_0 Embedding (HIP Graph T=1 ctx & Batched Prefill)
 * -------------------------------------------------------------------------- */

KERNEL ie_inkling_embed_q8_ctx_r2(const G u8 *qw, const G u16 *qs, G float *out, u32 nb, const G u32 *ctx) {
  const u32 b = wgid() * 8u + (tid() >> 5u);
  if (b >= nb) return;
  const u32 l = lane();
  const u32 tok = ctx[0];
  const unsigned long blk = (unsigned long)tok * nb + b;
  const float s = (float)((const G f16 *)qs)[blk];
  const i8 qv = ((const G i8 *)qw)[blk * 32ul + l];
  out[b * 32u + l] = s * (float)qv;
}

KERNEL ie_inkling_embed_q8_batch_r2(const G u8 *qw, const G u16 *qs, const G u32 *toks, G float *out, u32 nb, u32 os,
                                    u32 T) {
  const u32 t = tokid();
  if (t >= T) return;
  const u32 b = wgid() * 8u + (tid() >> 5u);
  if (b >= nb) return;
  const u32 l = lane();
  const u32 tok = toks[t];
  const unsigned long blk = (unsigned long)tok * nb + b;
  const float s = (float)((const G f16 *)qs)[blk];
  const i8 qv = ((const G i8 *)qw)[blk * 32ul + l];
  G float *ot = (G float *)((G u8 *)out + (unsigned long)t * os);
  ot[b * 32u + l] = s * (float)qv;
}

/* ----------------------------------------------------------------------------
 * [Bit 34] Kernel 3: ie_inkling_vclip_ctx_r2 (T=1 decode, pos = ctx[1])
 * -------------------------------------------------------------------------- */

KERNEL ie_inkling_vclip_ctx_r2(const G float *a, G float *b, const G float *v, u32 nh, u32 hd, float clamp, u32 norope,
                               float escale, u32 win, const G u32 *ctx) {
  const u32 pos = ctx[1];

  if (hd == 128u) {
    /* 1 wave (32 lanes) per head, 1 f32x4 per lane, 8 heads per WG. */
    const u32 wv = tid() >> 5u, l = lane();
    const u32 step = ngroups_x() * 8u;
    for (u32 h = wgid() * 8u + wv; h < nh; h += step) {
      const G f32x4 *ah = (const G f32x4 *)a + (unsigned long)h * 32u;
      G f32x4 *bh = (G f32x4 *)b + (unsigned long)h * 32u;
      const G f32x4 *vp = v ? ((const G f32x4 *)v + (unsigned long)h * 32u) : 0;

      f32x4 xv = ah[l];
      const f32x4 vv = vp ? vp[l] : (f32x4){1.0f, 1.0f, 1.0f, 1.0f};
      if (clamp > 0.0f && vp) xv *= vv;
      float ss = (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
      ss = wave_sum_all(ss);
      const float f = ink_r2_g16_vclip_scale(ss, 1.0f / 128.0f, clamp, norope, escale, win, pos);
      f32x4 outv = xv * f;
      if (clamp == 0.0f && vp) outv *= vv;
      bh[l] = outv;
    }
    return;
  }

  if (hd == 16u) {
    /* 4 lanes per head (1 f32x4 per lane), 8 heads per wave, 64 heads per WG. */
    const u32 sub = tid() >> 2u;
    const u32 sl = tid() & 3u;
    const u32 step = ngroups_x() * 64u;
    for (u32 h = wgid() * 64u + sub; h < nh; h += step) {
      const G f32x4 *ah = (const G f32x4 *)a + (unsigned long)h * 4u;
      G f32x4 *bh = (G f32x4 *)b + (unsigned long)h * 4u;
      const G f32x4 *vp = v ? ((const G f32x4 *)v + (unsigned long)h * 4u) : 0;

      f32x4 xv = ah[sl];
      const f32x4 vv = vp ? vp[sl] : (f32x4){1.0f, 1.0f, 1.0f, 1.0f};
      if (clamp > 0.0f && vp) xv *= vv;
      float ss = (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
      ss += shfl_xor_f(ss, 2u);
      ss += shfl_xor_f(ss, 1u);
      const float f = ink_r2_g16_vclip_scale(ss, 1.0f / 16.0f, clamp, norope, escale, win, pos);
      f32x4 outv = xv * f;
      if (clamp == 0.0f && vp) outv *= vv;
      bh[sl] = outv;
    }
  }
}

/* ----------------------------------------------------------------------------
 * [Bit 34] Kernel 4: ie_inkling_sconv_ctx_r2 (T=1 decode, pos = ctx[1])
 * -------------------------------------------------------------------------- */

KERNEL ie_inkling_sconv_ctx_r2(const G float *a, const G float *v, const G float *c, G float *ring, G float *b, u32 n,
                               u32 nh, float escale, const G u32 *ctx) {
#pragma clang fp contract(off)
  const u32 n4 = n >> 2u;
  const u32 i4 = wgid() * NT + tid();
  if (i4 >= n4) return;
  const u32 i0 = i4 << 2u;
  const u32 pos = ctx[1];

  const G f32x4 *v4 = (const G f32x4 *)v + i0;
  const f32x4 vw0 = v4[0], vw1 = v4[1], vw2 = v4[2], vw3 = v4[3];
  f32x4 w0 = (f32x4){vw0.w, vw1.w, vw2.w, vw3.w};
  f32x4 w1 = (f32x4){vw0.z, vw1.z, vw2.z, vw3.z};
  f32x4 w2 = (f32x4){vw0.y, vw1.y, vw2.y, vw3.y};
  f32x4 w3 = (f32x4){vw0.x, vw1.x, vw2.x, vw3.x};

  const f32x4 x0 = ((const G f32x4 *)a)[i4];
  f32x4 x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (pos > 0u) {
    const G f32x4 *rg = (const G f32x4 *)ring + i0;
    const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
    const u32 s1 = (pos - 1u) & 3u;
    x1 = (f32x4){ink_r2_g16_pick4(r0, s1), ink_r2_g16_pick4(r1, s1), ink_r2_g16_pick4(r2, s1),
                 ink_r2_g16_pick4(r3, s1)};
    if (pos >= 2u) {
      const u32 s2 = (pos - 2u) & 3u;
      x2 = (f32x4){ink_r2_g16_pick4(r0, s2), ink_r2_g16_pick4(r1, s2), ink_r2_g16_pick4(r2, s2),
                   ink_r2_g16_pick4(r3, s2)};
    }
    if (pos >= 3u) {
      const u32 s3 = (pos - 3u) & 3u;
      x3 = (f32x4){ink_r2_g16_pick4(r0, s3), ink_r2_g16_pick4(r1, s3), ink_r2_g16_pick4(r2, s3),
                   ink_r2_g16_pick4(r3, s3)};
    }
  }
  if (c) {
    const u32 h = i0 / (n / nh);
    const f32x4 ct = ((const G f32x4 *)c)[h];
    w0 += ct.x, w1 += ct.y, w2 += ct.z, w3 += ct.w;
  }
  const f32x4 y = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3;
  ((G f32x4 *)b)[i4] = (x0 + y) * escale;

  const u32 slot = pos & 3u;
  ring[(i0 + 0u) * 4u + slot] = x0.x;
  ring[(i0 + 1u) * 4u + slot] = x0.y;
  ring[(i0 + 2u) * 4u + slot] = x0.z;
  ring[(i0 + 3u) * 4u + slot] = x0.w;
}

/* ----------------------------------------------------------------------------
 * [Bit 34] Kernel 5: ie_inkling_sconv_add_ctx_r2 (T=1 fused sconv + residual add)
 * -------------------------------------------------------------------------- */

KERNEL ie_inkling_sconv_add_ctx_r2(const G float *a, const G float *v, const G float *c, G float *ring,
                                   const G float *res, G float *out, u32 n, u32 nh, float escale, const G u32 *ctx) {
#pragma clang fp contract(off)
  const u32 n4 = n >> 2u;
  const u32 i4 = wgid() * NT + tid();
  if (i4 >= n4) return;
  const u32 i0 = i4 << 2u;
  const u32 pos = ctx[1];

  const G f32x4 *v4 = (const G f32x4 *)v + i0;
  const f32x4 vw0 = v4[0], vw1 = v4[1], vw2 = v4[2], vw3 = v4[3];
  f32x4 w0 = (f32x4){vw0.w, vw1.w, vw2.w, vw3.w};
  f32x4 w1 = (f32x4){vw0.z, vw1.z, vw2.z, vw3.z};
  f32x4 w2 = (f32x4){vw0.y, vw1.y, vw2.y, vw3.y};
  f32x4 w3 = (f32x4){vw0.x, vw1.x, vw2.x, vw3.x};

  const f32x4 x0 = ((const G f32x4 *)a)[i4];
  f32x4 x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (pos > 0u) {
    const G f32x4 *rg = (const G f32x4 *)ring + i0;
    const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
    const u32 s1 = (pos - 1u) & 3u;
    x1 = (f32x4){ink_r2_g16_pick4(r0, s1), ink_r2_g16_pick4(r1, s1), ink_r2_g16_pick4(r2, s1),
                 ink_r2_g16_pick4(r3, s1)};
    if (pos >= 2u) {
      const u32 s2 = (pos - 2u) & 3u;
      x2 = (f32x4){ink_r2_g16_pick4(r0, s2), ink_r2_g16_pick4(r1, s2), ink_r2_g16_pick4(r2, s2),
                   ink_r2_g16_pick4(r3, s2)};
    }
    if (pos >= 3u) {
      const u32 s3 = (pos - 3u) & 3u;
      x3 = (f32x4){ink_r2_g16_pick4(r0, s3), ink_r2_g16_pick4(r1, s3), ink_r2_g16_pick4(r2, s3),
                   ink_r2_g16_pick4(r3, s3)};
    }
  }
  if (c) {
    const u32 h = i0 / (n / nh);
    const f32x4 ct = ((const G f32x4 *)c)[h];
    w0 += ct.x, w1 += ct.y, w2 += ct.z, w3 += ct.w;
  }
  const f32x4 y = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3;
  const f32x4 sc = (x0 + y) * escale;
  ((G f32x4 *)out)[i4] = res ? (((const G f32x4 *)res)[i4] + sc) : sc;

  const u32 slot = pos & 3u;
  ring[(i0 + 0u) * 4u + slot] = x0.x;
  ring[(i0 + 1u) * 4u + slot] = x0.y;
  ring[(i0 + 2u) * 4u + slot] = x0.z;
  ring[(i0 + 3u) * 4u + slot] = x0.w;
}

/* ----------------------------------------------------------------------------
 * [Bit 34] Kernel 6: ie_inkling_moe_sum_sconv_add_ctx_r2 (T=1 FFN tail fusion)
 * -------------------------------------------------------------------------- */

KERNEL ie_inkling_moe_sum_sconv_add_ctx_r2(const G float *y, const G u32 *route, const G float *v, G float *ring,
                                           const G float *res, G float *out, u32 n, u32 K, float escale,
                                           const G u32 *ctx) {
#pragma clang fp contract(off)
  const u32 n4 = n >> 2u;
  const u32 i4 = wgid() * NT + tid();
  if (i4 >= n4) return;
  const u32 i0 = i4 << 2u;
  const u32 pos = ctx[1];

  /* Step 1: exact MoE weighted sum over K experts -> x0. */
  const G float *rw = (const G float *)(route + K);
  const G f32x4 *y4 = (const G f32x4 *)y;
  f32x4 x0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (K == 8u) {
    const float w0 = rw[0], w1 = rw[1], w2 = rw[2], w3 = rw[3];
    const float w4 = rw[4], w5 = rw[5], w6 = rw[6], w7 = rw[7];
    const f32x4 y0 = y4[0ul * n4 + i4];
    const f32x4 y1 = y4[1ul * n4 + i4];
    const f32x4 y2 = y4[2ul * n4 + i4];
    const f32x4 y3 = y4[3ul * n4 + i4];
    const f32x4 y4v = y4[4ul * n4 + i4];
    const f32x4 y5 = y4[5ul * n4 + i4];
    const f32x4 y6 = y4[6ul * n4 + i4];
    const f32x4 y7 = y4[7ul * n4 + i4];
#pragma unroll
    for (u32 c = 0u; c < 4u; c++) {
      float a = __builtin_fmaf(w0, y0[c], 0.0f);
      a = __builtin_fmaf(w1, y1[c], a);
      a = __builtin_fmaf(w2, y2[c], a);
      a = __builtin_fmaf(w3, y3[c], a);
      a = __builtin_fmaf(w4, y4v[c], a);
      a = __builtin_fmaf(w5, y5[c], a);
      a = __builtin_fmaf(w6, y6[c], a);
      a = __builtin_fmaf(w7, y7[c], a);
      x0[c] = a;
    }
  } else {
    for (u32 k = 0u; k < K; k++) {
      const float wk = rw[k];
      const f32x4 yk = y4[(unsigned long)k * n4 + i4];
#pragma unroll
      for (u32 c = 0u; c < 4u; c++) x0[c] = __builtin_fmaf(wk, yk[c], x0[c]);
    }
  }

  /* Step 2: exact 4-tap static post-conv (c == NULL) + in-place ring update. */
  const G f32x4 *v4 = (const G f32x4 *)v + i0;
  const f32x4 vw0 = v4[0], vw1 = v4[1], vw2 = v4[2], vw3 = v4[3];
  const f32x4 sw0 = (f32x4){vw0.w, vw1.w, vw2.w, vw3.w};
  const f32x4 sw1 = (f32x4){vw0.z, vw1.z, vw2.z, vw3.z};
  const f32x4 sw2 = (f32x4){vw0.y, vw1.y, vw2.y, vw3.y};
  const f32x4 sw3 = (f32x4){vw0.x, vw1.x, vw2.x, vw3.x};

  f32x4 x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (pos > 0u) {
    const G f32x4 *rg = (const G f32x4 *)ring + i0;
    const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
    const u32 s1 = (pos - 1u) & 3u;
    x1 = (f32x4){ink_r2_g16_pick4(r0, s1), ink_r2_g16_pick4(r1, s1), ink_r2_g16_pick4(r2, s1),
                 ink_r2_g16_pick4(r3, s1)};
    if (pos >= 2u) {
      const u32 s2 = (pos - 2u) & 3u;
      x2 = (f32x4){ink_r2_g16_pick4(r0, s2), ink_r2_g16_pick4(r1, s2), ink_r2_g16_pick4(r2, s2),
                   ink_r2_g16_pick4(r3, s2)};
    }
    if (pos >= 3u) {
      const u32 s3 = (pos - 3u) & 3u;
      x3 = (f32x4){ink_r2_g16_pick4(r0, s3), ink_r2_g16_pick4(r1, s3), ink_r2_g16_pick4(r2, s3),
                   ink_r2_g16_pick4(r3, s3)};
    }
  }

  const f32x4 sy = ((x0 * sw0 + x1 * sw1) + x2 * sw2) + x3 * sw3;
  const f32x4 fpc = (x0 + sy) * escale;

  const u32 slot = pos & 3u;
  ring[(i0 + 0u) * 4u + slot] = x0.x;
  ring[(i0 + 1u) * 4u + slot] = x0.y;
  ring[(i0 + 2u) * 4u + slot] = x0.z;
  ring[(i0 + 3u) * 4u + slot] = x0.w;

  /* Step 3: residual add -> out. */
  ((G f32x4 *)out)[i4] = res ? (((const G f32x4 *)res)[i4] + fpc) : fpc;
}

/* ----------------------------------------------------------------------------
 * [Bit 34] Kernel 7: ie_inkling_kv_store_ctx_r2 (T=1 fallback KV store, pos = ctx[1])
 * -------------------------------------------------------------------------- */

KERNEL ie_inkling_kv_store_ctx_r2(const G float *k, const G float *v, G kvt *kc0, G kvt *vc0, u32 kv_len, u32 kvd,
                                  const G u32 *ctx) {
  const u32 n4 = kvd >> 2u;
  const u32 i4 = wgid() * NT + tid();
  if (i4 >= n4) return;
  const u32 slot = ctx[1] % kv_len;
  const unsigned long base4 = ((unsigned long)slot * kvd) >> 2u;
  const f32x4 kv = ((const G f32x4 *)k)[i4];
  const f32x4 vv = ((const G f32x4 *)v)[i4];
  ((G kv4 *)kc0)[base4 + i4] = (kv4){(kvt)kv.x, (kvt)kv.y, (kvt)kv.z, (kvt)kv.w};
  ((G kv4 *)vc0)[base4 + i4] = (kv4){(kvt)vv.x, (kvt)vv.y, (kvt)vv.z, (kvt)vv.w};
}

/* ----------------------------------------------------------------------------
 * [Bit 35] Kernel 8: ie_inkling_rel_attn_dec_ctx_r2 (HIP Graph GQA-3 f16-KV Decode)
 * -------------------------------------------------------------------------- */

KERNEL ie_inkling_rel_attn_dec_ctx_r2(const G float *q, const G kvt *kc, const G kvt *vc, const G float *R,
                                      const G float *rel_bias, const G float *sink, G float *part, G float *out,
                                      u32 kv_len, u32 hd, u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win,
                                      u32 nsplit, G u32 *count, const G u32 *ctx) {
  static LDS float ink_r2_g16_m[INK_R2_G16_GQA][8];
  static LDS float ink_r2_g16_l[INK_R2_G16_GQA][8];
  static LDS __attribute__((aligned(16))) float ink_r2_g16_o[INK_R2_G16_GQA][8][INK_R2_G16_HD];
  static LDS u32 ink_r2_g16_last;

  if (hd != INK_R2_G16_HD || d_rel != INK_R2_G16_DREL || n_head != INK_R2_G16_GQA * n_kv) return;

  const u32 kh = wgid() / nsplit;
  const u32 s = wgid() % nsplit;
  const u32 h0 = kh * INK_R2_G16_GQA;
  const u32 kvd = n_kv * INK_R2_G16_HD;
  const u32 w = tid() >> 5u;
  const u32 l = lane();

  /* Compute dynamic position bounds and split chunk size on device from ctx[1]. */
  const u32 pos = ctx[1];
  const u32 pend = pos + 1u;
  const u32 w0 = (win > 0u && pend > win) ? (pend - win) : 0u;
  const u32 np = pend - w0;
  const u32 ch = (np + nsplit - 1u) / nsplit;
  const u32 off0 = s * ch;
  const u32 p0 = (off0 < np) ? (w0 + off0) : pend;
  const u32 p1 = (off0 < np && off0 + ch < np) ? (w0 + off0 + ch) : pend;

  if (nsplit > 1u && p0 >= p1) {
    /* Empty tail split (bucketed nsplit > np): write sentinel partial record
     * and still participate in atomicAdd(&count[kh], 1u) for split-K merge. */
    for (u32 idx = tid(); idx < INK_R2_G16_GQA * INK_R2_G16_HD; idx += NT) {
      const u32 g = idx >> 7u;
      const u32 e = idx & 127u;
      G float *pp = part + (unsigned long)((h0 + g) * nsplit + s) * INK_R2_G16_PSTR;
      pp[2u + e] = 0.0f;
      if (e == 0u) {
        pp[0] = -1e30f;
        pp[1] = 0.0f;
      }
    }
  } else {
    const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
    const float scale = 1.0f / __builtin_sqrtf((float)INK_R2_G16_HD);

    /* Load Q (3 heads x 128 f32 -> 1 f32x4/lane/head) and R (3 heads x 16 f32). */
    const f32x4 q0 = ((const G f32x4 *)(q + (unsigned long)(h0 + 0u) * INK_R2_G16_HD))[l];
    const f32x4 q1 = ((const G f32x4 *)(q + (unsigned long)(h0 + 1u) * INK_R2_G16_HD))[l];
    const f32x4 q2 = ((const G f32x4 *)(q + (unsigned long)(h0 + 2u) * INK_R2_G16_HD))[l];

    const float rv0 = (l < INK_R2_G16_DREL) ? R[(unsigned long)(h0 + 0u) * INK_R2_G16_DREL + l] : 0.0f;
    const float rv1 = (l < INK_R2_G16_DREL) ? R[(unsigned long)(h0 + 1u) * INK_R2_G16_DREL + l] : 0.0f;
    const float rv2 = (l < INK_R2_G16_DREL) ? R[(unsigned long)(h0 + 2u) * INK_R2_G16_DREL + l] : 0.0f;

    f32x4 acc0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};

    float m0 = -__builtin_inff(), sum0 = 0.0f;
    float m1 = -__builtin_inff(), sum1 = 0.0f;
    float m2 = -__builtin_inff(), sum2 = 0.0f;

    const G kvt *kc_h = kc + (unsigned long)kh * INK_R2_G16_HD;
    const G kvt *vc_h = vc + (unsigned long)kh * INK_R2_G16_HD;

    /* Software-pipelined GQA-3 KV loop: 1x coalesced f16x4 K load + 1x V load
     * per wave shared across all 3 Q heads, with 1-step KV lookahead. */
    u32 kp = p0 + w;
    if (kp < p1) {
      u32 kvp = kp % kv_len;
      const G kv4 *krow = (const G kv4 *)(kc_h + (unsigned long)kvp * kvd);
      const G kv4 *vrow = (const G kv4 *)(vc_h + (unsigned long)kvp * kvd);
      kv4 kh4 = krow[l];
      kv4 vh4 = vrow[l];

      for (;;) {
        const u32 dist = pos - kp;
        const int has_rel = (dist < extent_lim);
        float rb = 0.0f;
        if (has_rel && l < INK_R2_G16_DREL) rb = rel_bias[(unsigned long)dist * INK_R2_G16_DREL + l];

        const u32 kp_nxt = kp + 8u;
        u32 kvp_nxt = kvp;
        kv4 kh4_nxt = kh4;
        kv4 vh4_nxt = vh4;
        if (kp_nxt < p1) {
          kvp_nxt = kvp + 8u;
          if (kvp_nxt >= kv_len) kvp_nxt = (kv_len >= 8u) ? (kvp_nxt - kv_len) : (kp_nxt % kv_len);
          const G kv4 *krow_n = (const G kv4 *)(kc_h + (unsigned long)kvp_nxt * kvd);
          const G kv4 *vrow_n = (const G kv4 *)(vc_h + (unsigned long)kvp_nxt * kvd);
          kh4_nxt = krow_n[l];
          vh4_nxt = vrow_n[l];
        }

        const f32x4 kf = __builtin_convertvector(kh4, f32x4);
        float d0 = wave_sum_all(ink_r2_g16_dot4(q0, kf)) * scale;
        float d1 = wave_sum_all(ink_r2_g16_dot4(q1, kf)) * scale;
        float d2 = wave_sum_all(ink_r2_g16_dot4(q2, kf)) * scale;

        if (has_rel) {
          d0 += ink_r2_g16_halfwave_sum_all(rv0 * rb);
          d1 += ink_r2_g16_halfwave_sum_all(rv1 * rb);
          d2 += ink_r2_g16_halfwave_sum_all(rv2 * rb);
        }

        const float mn0 = __builtin_fmaxf(m0, d0);
        const float c0 = (sum0 > 0.0f) ? __builtin_expf(m0 - mn0) : 0.0f;
        const float e0 = __builtin_expf(d0 - mn0);
        sum0 = sum0 * c0 + e0;
        m0 = mn0;

        const float mn1 = __builtin_fmaxf(m1, d1);
        const float c1 = (sum1 > 0.0f) ? __builtin_expf(m1 - mn1) : 0.0f;
        const float e1 = __builtin_expf(d1 - mn1);
        sum1 = sum1 * c1 + e1;
        m1 = mn1;

        const float mn2 = __builtin_fmaxf(m2, d2);
        const float c2 = (sum2 > 0.0f) ? __builtin_expf(m2 - mn2) : 0.0f;
        const float e2 = __builtin_expf(d2 - mn2);
        sum2 = sum2 * c2 + e2;
        m2 = mn2;

        const f32x4 vf = __builtin_convertvector(vh4, f32x4);
        acc0 = acc0 * c0 + e0 * vf;
        acc1 = acc1 * c1 + e1 * vf;
        acc2 = acc2 * c2 + e2 * vf;

        if (kp_nxt >= p1) break;
        kp = kp_nxt;
        kvp = kvp_nxt;
        kh4 = kh4_nxt;
        vh4 = vh4_nxt;
      }
    }

    /* Stage wave-level (m, sum, acc[128]) for all 3 Q heads into LDS. */
    if (l == 0u) {
      ink_r2_g16_m[0][w] = m0;
      ink_r2_g16_l[0][w] = sum0;
      ink_r2_g16_m[1][w] = m1;
      ink_r2_g16_l[1][w] = sum1;
      ink_r2_g16_m[2][w] = m2;
      ink_r2_g16_l[2][w] = sum2;
    }
    ((LDSP f32x4 *)&ink_r2_g16_o[0][w][0])[l] = acc0;
    ((LDSP f32x4 *)&ink_r2_g16_o[1][w][0])[l] = acc1;
    ((LDSP f32x4 *)&ink_r2_g16_o[2][w][0])[l] = acc2;
    lbarrier();

    /* Fast path: nsplit == 1 (no global scratch or atomics). */
    if (nsplit == 1u) {
      for (u32 idx = tid(); idx < INK_R2_G16_GQA * INK_R2_G16_HD; idx += NT) {
        const u32 g = idx >> 7u;
        const u32 e = idx & 127u;
        const u32 h = h0 + g;
        const float sv = sink[h];
        float M = sv;
        for (u32 x = 0u; x < 8u; x++)
          if (ink_r2_g16_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r2_g16_m[g][x]);
        float L = __builtin_expf(sv - M);
        float o = 0.0f;
        for (u32 x = 0u; x < 8u; x++) {
          const float lx = ink_r2_g16_l[g][x];
          if (lx > 0.0f) {
            const float wx = __builtin_expf(ink_r2_g16_m[g][x] - M);
            L += lx * wx;
            o += ink_r2_g16_o[g][x][e] * wx;
          }
        }
        out[(unsigned long)h * INK_R2_G16_HD + e] = o / L;
      }
      return;
    }

    /* Split-K path: write 3 partial records into part[(h * nsplit + s) * 130]. */
    for (u32 idx = tid(); idx < INK_R2_G16_GQA * INK_R2_G16_HD; idx += NT) {
      const u32 g = idx >> 7u;
      const u32 e = idx & 127u;
      const u32 h = h0 + g;
      float M = -__builtin_inff();
      for (u32 x = 0u; x < 8u; x++)
        if (ink_r2_g16_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r2_g16_m[g][x]);
      G float *pp = part + (unsigned long)(h * nsplit + s) * INK_R2_G16_PSTR;
      float o = 0.0f;
      float L = 0.0f;
      for (u32 x = 0u; x < 8u; x++) {
        const float lx = ink_r2_g16_l[g][x];
        if (lx > 0.0f) {
          const float wx = __builtin_expf(ink_r2_g16_m[g][x] - M);
          o += ink_r2_g16_o[g][x][e] * wx;
          if (e == 0u) L += lx * wx;
        }
      }
      pp[2u + e] = o;
      if (e == 0u) {
        pp[0] = M;
        pp[1] = L;
      }
    }
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r2_g16_last =
        (__scoped_atomic_fetch_add(&count[kh], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u);
  }
  barrier();
  if (!ink_r2_g16_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  /* Last WG for KV head kh merges all 3 Q heads across s = 0 .. nsplit - 1. */
  G float *ph0 = part + (unsigned long)(h0 + 0u) * nsplit * INK_R2_G16_PSTR;
  G float *ph1 = part + (unsigned long)(h0 + 1u) * nsplit * INK_R2_G16_PSTR;
  G float *ph2 = part + (unsigned long)(h0 + 2u) * nsplit * INK_R2_G16_PSTR;

  /* Pass 1: global max M2[0..2] including per-head sink (skips empty splits with L == 0). */
  float mv0 = (tid() == 0u) ? sink[h0 + 0u] : -__builtin_inff();
  float mv1 = (tid() == 0u) ? sink[h0 + 1u] : -__builtin_inff();
  float mv2 = (tid() == 0u) ? sink[h0 + 2u] : -__builtin_inff();
  for (u32 t = tid(); t < nsplit; t += NT) {
    const unsigned long off = (unsigned long)t * INK_R2_G16_PSTR;
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
    ink_r2_g16_m[0][w] = mv0;
    ink_r2_g16_m[1][w] = mv1;
    ink_r2_g16_m[2][w] = mv2;
  }
  lbarrier();

  float M2[INK_R2_G16_GQA];
  for (u32 g = 0u; g < INK_R2_G16_GQA; g++) {
    float mx = ink_r2_g16_m[g][0];
    for (u32 x = 1u; x < 8u; x++) mx = __builtin_fmaxf(mx, ink_r2_g16_m[g][x]);
    M2[g] = mx;
  }

  /* Pass 2: per-split weights + global denominator L2[0..2]. */
  const int wt_in_lds = (nsplit <= INK_R2_G16_HD);
  float lv0 = 0.0f, lv1 = 0.0f, lv2 = 0.0f;
  for (u32 t = tid(); t < nsplit; t += NT) {
    const unsigned long off = (unsigned long)t * INK_R2_G16_PSTR;

    const float lp0 = ph0[off + 1u];
    const float wt0 = (lp0 > 0.0f) ? __builtin_expf(ph0[off] - M2[0]) : 0.0f;
    if (wt_in_lds) ink_r2_g16_o[0][0][t] = wt0;
    else ph0[off] = wt0;
    lv0 += lp0 * wt0;

    const float lp1 = ph1[off + 1u];
    const float wt1 = (lp1 > 0.0f) ? __builtin_expf(ph1[off] - M2[1]) : 0.0f;
    if (wt_in_lds) ink_r2_g16_o[1][0][t] = wt1;
    else ph1[off] = wt1;
    lv1 += lp1 * wt1;

    const float lp2 = ph2[off + 1u];
    const float wt2 = (lp2 > 0.0f) ? __builtin_expf(ph2[off] - M2[2]) : 0.0f;
    if (wt_in_lds) ink_r2_g16_o[2][0][t] = wt2;
    else ph2[off] = wt2;
    lv2 += lp2 * wt2;
  }
  lv0 = wave_tree_f(lv0);
  lv1 = wave_tree_f(lv1);
  lv2 = wave_tree_f(lv2);
  if (l == 0u) {
    ink_r2_g16_l[0][w] = lv0;
    ink_r2_g16_l[1][w] = lv1;
    ink_r2_g16_l[2][w] = lv2;
  }
  if (wt_in_lds) lbarrier();
  else barrier();

  float L2[INK_R2_G16_GQA];
  for (u32 g = 0u; g < INK_R2_G16_GQA; g++) {
    float ls = 0.0f;
    for (u32 x = 0u; x < 8u; x++) ls += ink_r2_g16_l[g][x];
    L2[g] = ls + __builtin_expf(sink[h0 + g] - M2[g]);
  }

  /* Pass 3: weighted sum of partial outputs -> out[h * 128 + e] (skips zero-weight empty splits). */
  for (u32 idx = tid(); idx < INK_R2_G16_GQA * INK_R2_G16_HD; idx += NT) {
    const u32 g = idx >> 7u;
    const u32 e = idx & 127u;
    const u32 h = h0 + g;
    const G float *ph = part + (unsigned long)h * nsplit * INK_R2_G16_PSTR;
    float o = 0.0f;
    if (wt_in_lds) {
      for (u32 t = 0u; t < nsplit; t++) {
        const float wt = ink_r2_g16_o[g][0][t];
        if (wt > 0.0f) o += ph[(unsigned long)t * INK_R2_G16_PSTR + 2u + e] * wt;
      }
    } else {
      for (u32 t = 0u; t < nsplit; t++) {
        const float wt = ph[(unsigned long)t * INK_R2_G16_PSTR];
        if (wt > 0.0f) o += ph[(unsigned long)t * INK_R2_G16_PSTR + 2u + e] * wt;
      }
    }
    out[(unsigned long)h * INK_R2_G16_HD + e] = o / L2[g];
  }

  if (tid() == 0u) count[kh] = 0u;
}

#undef INK_R2_G16_HD
#undef INK_R2_G16_DREL
#undef INK_R2_G16_GQA
#undef INK_R2_G16_PSTR
