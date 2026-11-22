/* ie_inkling_r2_g12.inc -- Fused 5-op Inkling-2 KV pipeline
 * (sconv(k) -> vclip(k) -> sconv(v) -> vclip(v) -> f16 kv_store) + HIP Graph *_ctx variant
 * for RX 9070 XT (gfx1201, wave32, NT = 256).
 *
 * Specialized for Inkling-2 KV geometry: n_kv = 8, hd = 128, kvd = 1024 (f16 KV cache).
 * 1 WG of 256 threads = 8 waves (1 wave per KV head, 1 f32x4 per lane); zero LDS, zero barriers.
 *
 * Kernels exported:
 *   1. KERNEL ie_inkling_kv_sconv_vclip_store_r2(
 *          const G float *ka, const G float *vk, const G float *ck, G float *ring_k,
 *          const G float *va, const G float *vv, const G float *cv, G float *ring_v,
 *          G kvt *kc0, G kvt *vc0, float sc_escale, float k_clamp, float v_eps,
 *          u32 pos0, u32 T, u32 kv_len, u32 kas, u32 cks, u32 vas, u32 cvs)
 *      Bit: 25 (IE_INKLING_KERNEL_MASK).
 *      Grid:
 *        - T == 1u: (1, 1, 1)                     [updates ring_k & ring_v in-place; 1 launch total]
 *        - T >  1u: (1, (T + 3u) / 4u, 1)         [TTILE = 4 tokens/WG; ring_k & ring_v are read-only
 *                                                  to avoid inter-WG race; follow with 1 launch of
 *                                                  ie_inkling_kv_ring_store_r2 below]
 *      Guard: !b->kvq8 && !b->kvq4 && !mk.rec && n_kv == 8u && hd == 128u && kv_len > 0u &&
 *             k_clamp > 0.0f && k_vclip.v == NULL && k_vclip.norope == 0u &&
 *             v_vclip.v == NULL && v_vclip.clamp == 0.0f &&
 *             k_sconv.escale == v_sconv.escale &&
 *             ptrs ka, vk, ring_k, va, vv, ring_v, kc0, vc0 (, ck, cv) 16-byte aligned &&
 *             (T == 1u || ((kas | vas | (ck ? cks : 0u) | (cv ? cvs : 0u)) & 15u) == 0u).
 *      Note: On full layers, ck = k_dyn and cv = v_dyn (both non-NULL, [n_kv][4] per token);
 *            on SWA layers, ck = NULL and cv = NULL. On full layers, v_dyn matvec must run
 *            before this fused launch (it depends only on vn/vq, independent of k_conv/k_clip).
 *      Parity: sconv is bitwise-exact vs ie_sconv / ie_inkling_sconv_r1 (fp contract off, exact
 *              left-to-right sum); vclip is bitwise-exact vs ie_inkling_vclip_r1 (f32-tol ~1e-7
 *              vs base ie_vclip due to f32x4 sum-of-squares); f16 KV store is exact round-to-even.
 *
 *   2. KERNEL ie_inkling_kv_ring_store_r2(
 *          const G float *ka, const G float *va, G float *ring_k, G float *ring_v,
 *          u32 pos0, u32 T, u32 kas, u32 vas)
 *      Bit: 25 (T > 1u companion to ie_inkling_kv_sconv_vclip_store_r2; replaces 2x ie_sconv_ring_store).
 *      Grid: (1, 1, 1).
 *      Guard: T > 1u && same alignment/geometry guard as ie_inkling_kv_sconv_vclip_store_r2.
 *      Parity: bitwise-exact vs 2x ie_sconv_ring_store (coalesced f32x4 ring writes when T >= 4u).
 *
 *   3. KERNEL ie_inkling_kv_sconv_vclip_store_ctx_r2(
 *          const G float *ka, const G float *vk, const G float *ck, G float *ring_k,
 *          const G float *va, const G float *vv, const G float *cv, G float *ring_v,
 *          G kvt *kc0, G kvt *vc0, float sc_escale, float k_clamp, float v_eps,
 *          const G u32 *ctx, u32 kv_len)
 *      Bit: 26 (IE_INKLING_KERNEL_MASK; HIP Graph T == 1u decode variant).
 *      Grid: (1, 1, 1).
 *      Guard: T == 1u && ctx != NULL (reads pos = ctx[1] from ie_ink_dec_ctx { u32 tok; u32 pos; }) &&
 *             same geometry/alignment/param guards as #1. All args are 100% static for graph capture.
 *      Parity: bitwise-exact vs ie_inkling_kv_sconv_vclip_store_r2 at T == 1u.
 */

#define INK_R2_G12_TTILE 4u

static inline float ink_r2_g12_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline void ink_r2_g12_load_sw(const G float *v, u32 i0, f32x4 *sw0, f32x4 *sw1, f32x4 *sw2, f32x4 *sw3) {
  const G f32x4 *v4 = (const G f32x4 *)v + i0;
  const f32x4 vw0 = v4[0], vw1 = v4[1], vw2 = v4[2], vw3 = v4[3];
  *sw0 = (f32x4){vw0.w, vw1.w, vw2.w, vw3.w};
  *sw1 = (f32x4){vw0.z, vw1.z, vw2.z, vw3.z};
  *sw2 = (f32x4){vw0.y, vw1.y, vw2.y, vw3.y};
  *sw3 = (f32x4){vw0.x, vw1.x, vw2.x, vw3.x};
}

static inline void ink_r2_g12_seed_ring(const G float *ring, u32 i0, u32 pos0, f32x4 *x1, f32x4 *x2, f32x4 *x3) {
  *x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (pos0 > 0u) {
    const G f32x4 *rg = (const G f32x4 *)ring + i0;
    const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
    const u32 s1 = (pos0 - 1u) & 3u;
    *x1 = (f32x4){ink_r2_g12_pick4(r0, s1), ink_r2_g12_pick4(r1, s1), ink_r2_g12_pick4(r2, s1),
                  ink_r2_g12_pick4(r3, s1)};
    if (pos0 >= 2u) {
      const u32 s2 = (pos0 - 2u) & 3u;
      *x2 = (f32x4){ink_r2_g12_pick4(r0, s2), ink_r2_g12_pick4(r1, s2), ink_r2_g12_pick4(r2, s2),
                    ink_r2_g12_pick4(r3, s2)};
    }
    if (pos0 >= 3u) {
      const u32 s3 = (pos0 - 3u) & 3u;
      *x3 = (f32x4){ink_r2_g12_pick4(r0, s3), ink_r2_g12_pick4(r1, s3), ink_r2_g12_pick4(r2, s3),
                    ink_r2_g12_pick4(r3, s3)};
    }
  }
}

static inline f32x4 ink_r2_g12_sconv4(f32x4 x0, f32x4 x1, f32x4 x2, f32x4 x3, f32x4 w0, f32x4 w1, f32x4 w2, f32x4 w3,
                                      float escale) {
#pragma clang fp contract(off)
  const f32x4 y = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3;
  return (x0 + y) * escale;
}

static inline void ink_r2_g12_kv_step1(const G float *ka, const G float *vk, const G float *ck, G float *ring_k,
                                       const G float *va, const G float *vv, const G float *cv, G float *ring_v,
                                       G kvt *kc0, G kvt *vc0, float sc_escale, float k_clamp, float v_eps, u32 pos,
                                       u32 kv_len) {
  const u32 i4 = tid();
  const u32 i0 = i4 << 2;
  const u32 h = i4 >> 5;
  const u32 slot = pos & 3u;
  const unsigned long kv_off = (unsigned long)(pos % kv_len) * 256u + i4;

  /* K: sconv(k) -> per-head L2 vclip(k) -> f16 kc store + in-place ring_k update. */
  {
    f32x4 w0, w1, w2, w3;
    ink_r2_g12_load_sw(vk, i0, &w0, &w1, &w2, &w3);
    if (ck) {
      const f32x4 ct = ((const G f32x4 *)ck)[h];
      w0 += ct.x, w1 += ct.y, w2 += ct.z, w3 += ct.w;
    }
    const f32x4 x0 = ((const G f32x4 *)ka)[i4];
    f32x4 x1, x2, x3;
    ink_r2_g12_seed_ring(ring_k, i0, pos, &x1, &x2, &x3);
    const f32x4 kx = ink_r2_g12_sconv4(x0, x1, x2, x3, w0, w1, w2, w3, sc_escale);

    float ss = (kx.x * kx.x + kx.y * kx.y) + (kx.z * kx.z + kx.w * kx.w);
    ss = wave_sum_all(ss);
    float fk = k_clamp / (__builtin_sqrtf(ss) + 1e-6f);
    if (fk > 1.0f) fk = 1.0f;
    ((G kv4 *)kc0)[kv_off] = __builtin_convertvector(kx * fk, kv4);

    ring_k[(i0 + 0u) * 4u + slot] = x0.x;
    ring_k[(i0 + 1u) * 4u + slot] = x0.y;
    ring_k[(i0 + 2u) * 4u + slot] = x0.z;
    ring_k[(i0 + 3u) * 4u + slot] = x0.w;
  }

  /* V: sconv(v) -> per-head RMSNorm vclip(v) -> f16 vc store + in-place ring_v update. */
  {
    f32x4 w0, w1, w2, w3;
    ink_r2_g12_load_sw(vv, i0, &w0, &w1, &w2, &w3);
    if (cv) {
      const f32x4 ct = ((const G f32x4 *)cv)[h];
      w0 += ct.x, w1 += ct.y, w2 += ct.z, w3 += ct.w;
    }
    const f32x4 x0 = ((const G f32x4 *)va)[i4];
    f32x4 x1, x2, x3;
    ink_r2_g12_seed_ring(ring_v, i0, pos, &x1, &x2, &x3);
    const f32x4 vx = ink_r2_g12_sconv4(x0, x1, x2, x3, w0, w1, w2, w3, sc_escale);

    float ss = (vx.x * vx.x + vx.y * vx.y) + (vx.z * vx.z + vx.w * vx.w);
    ss = wave_sum_all(ss);
    const float fv = 1.0f / __builtin_sqrtf(ss * (1.0f / 128.0f) + v_eps);
    ((G kv4 *)vc0)[kv_off] = __builtin_convertvector(vx * fv, kv4);

    ring_v[(i0 + 0u) * 4u + slot] = x0.x;
    ring_v[(i0 + 1u) * 4u + slot] = x0.y;
    ring_v[(i0 + 2u) * 4u + slot] = x0.z;
    ring_v[(i0 + 3u) * 4u + slot] = x0.w;
  }
}

KERNEL ie_inkling_kv_sconv_vclip_store_r2(const G float *ka, const G float *vk, const G float *ck, G float *ring_k,
                                          const G float *va, const G float *vv, const G float *cv, G float *ring_v,
                                          G kvt *kc0, G kvt *vc0, float sc_escale, float k_clamp, float v_eps,
                                          u32 pos0, u32 T, u32 kv_len, u32 kas, u32 cks, u32 vas, u32 cvs) {
  if (T == 1u) {
    ink_r2_g12_kv_step1(ka, vk, ck, ring_k, va, vv, cv, ring_v, kc0, vc0, sc_escale, k_clamp, v_eps, pos0, kv_len);
    return;
  }

  const u32 t0 = tokid() * INK_R2_G12_TTILE;
  if (t0 >= T) return;
  const u32 t_end = (t0 + INK_R2_G12_TTILE < T) ? (t0 + INK_R2_G12_TTILE) : T;
  const u32 i4 = tid();
  const u32 i0 = i4 << 2;
  const u32 h = i4 >> 5;
  const u32 p0 = (pos0 + t0) % kv_len;

  /* K path over t = t0 .. t_end - 1 (ring_k is read-only; updated by ie_inkling_kv_ring_store_r2). */
  {
    f32x4 sw0, sw1, sw2, sw3;
    ink_r2_g12_load_sw(vk, i0, &sw0, &sw1, &sw2, &sw3);
    f32x4 x1, x2, x3;
    if (t0 >= 3u) {
      x3 = ((const G f32x4 *)((const G u8 *)ka + (unsigned long)(t0 - 3u) * kas))[i4];
      x2 = ((const G f32x4 *)((const G u8 *)ka + (unsigned long)(t0 - 2u) * kas))[i4];
      x1 = ((const G f32x4 *)((const G u8 *)ka + (unsigned long)(t0 - 1u) * kas))[i4];
    } else {
      ink_r2_g12_seed_ring(ring_k, i0, pos0, &x1, &x2, &x3);
    }
    u32 p = p0;
    for (u32 t = t0; t < t_end; t++) {
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)ka + (unsigned long)t * kas))[i4];
      f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
      if (ck) {
        const f32x4 ct = ((const G f32x4 *)((const G u8 *)ck + (unsigned long)t * cks))[h];
        w0 += ct.x, w1 += ct.y, w2 += ct.z, w3 += ct.w;
      }
      const f32x4 kx = ink_r2_g12_sconv4(x0, x1, x2, x3, w0, w1, w2, w3, sc_escale);
      float ss = (kx.x * kx.x + kx.y * kx.y) + (kx.z * kx.z + kx.w * kx.w);
      ss = wave_sum_all(ss);
      float fk = k_clamp / (__builtin_sqrtf(ss) + 1e-6f);
      if (fk > 1.0f) fk = 1.0f;
      ((G kv4 *)kc0)[(unsigned long)p * 256u + i4] = __builtin_convertvector(kx * fk, kv4);
      x3 = x2;
      x2 = x1;
      x1 = x0;
      p++;
      if (p == kv_len) p = 0u;
    }
  }

  /* V path over t = t0 .. t_end - 1 (ring_v is read-only; updated by ie_inkling_kv_ring_store_r2). */
  {
    f32x4 sw0, sw1, sw2, sw3;
    ink_r2_g12_load_sw(vv, i0, &sw0, &sw1, &sw2, &sw3);
    f32x4 x1, x2, x3;
    if (t0 >= 3u) {
      x3 = ((const G f32x4 *)((const G u8 *)va + (unsigned long)(t0 - 3u) * vas))[i4];
      x2 = ((const G f32x4 *)((const G u8 *)va + (unsigned long)(t0 - 2u) * vas))[i4];
      x1 = ((const G f32x4 *)((const G u8 *)va + (unsigned long)(t0 - 1u) * vas))[i4];
    } else {
      ink_r2_g12_seed_ring(ring_v, i0, pos0, &x1, &x2, &x3);
    }
    u32 p = p0;
    for (u32 t = t0; t < t_end; t++) {
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)va + (unsigned long)t * vas))[i4];
      f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
      if (cv) {
        const f32x4 ct = ((const G f32x4 *)((const G u8 *)cv + (unsigned long)t * cvs))[h];
        w0 += ct.x, w1 += ct.y, w2 += ct.z, w3 += ct.w;
      }
      const f32x4 vx = ink_r2_g12_sconv4(x0, x1, x2, x3, w0, w1, w2, w3, sc_escale);
      float ss = (vx.x * vx.x + vx.y * vx.y) + (vx.z * vx.z + vx.w * vx.w);
      ss = wave_sum_all(ss);
      const float fv = 1.0f / __builtin_sqrtf(ss * (1.0f / 128.0f) + v_eps);
      ((G kv4 *)vc0)[(unsigned long)p * 256u + i4] = __builtin_convertvector(vx * fv, kv4);
      x3 = x2;
      x2 = x1;
      x1 = x0;
      p++;
      if (p == kv_len) p = 0u;
    }
  }
}

static inline void ink_r2_g12_ring_store4(const G float *a, G float *ring, u32 i4, u32 i0, u32 pos0, u32 T, u32 as) {
  if (T >= 4u) {
    const u32 k0 = T - 4u;
    const u32 s0 = (pos0 + k0) & 3u;
    const u32 t_s0 = k0 + ((0u - s0) & 3u);
    const u32 t_s1 = k0 + ((1u - s0) & 3u);
    const u32 t_s2 = k0 + ((2u - s0) & 3u);
    const u32 t_s3 = k0 + ((3u - s0) & 3u);
    const f32x4 v0 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t_s0 * as))[i4];
    const f32x4 v1 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t_s1 * as))[i4];
    const f32x4 v2 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t_s2 * as))[i4];
    const f32x4 v3 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t_s3 * as))[i4];
    G f32x4 *rg = (G f32x4 *)ring + i0;
    rg[0] = (f32x4){v0.x, v1.x, v2.x, v3.x};
    rg[1] = (f32x4){v0.y, v1.y, v2.y, v3.y};
    rg[2] = (f32x4){v0.z, v1.z, v2.z, v3.z};
    rg[3] = (f32x4){v0.w, v1.w, v2.w, v3.w};
  } else {
    for (u32 t = 0u; t < T; t++) {
      const u32 slot = (pos0 + t) & 3u;
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t * as))[i4];
      ring[(i0 + 0u) * 4u + slot] = x0.x;
      ring[(i0 + 1u) * 4u + slot] = x0.y;
      ring[(i0 + 2u) * 4u + slot] = x0.z;
      ring[(i0 + 3u) * 4u + slot] = x0.w;
    }
  }
}

KERNEL ie_inkling_kv_ring_store_r2(const G float *ka, const G float *va, G float *ring_k, G float *ring_v, u32 pos0,
                                   u32 T, u32 kas, u32 vas) {
  const u32 i4 = tid();
  const u32 i0 = i4 << 2;
  ink_r2_g12_ring_store4(ka, ring_k, i4, i0, pos0, T, kas);
  ink_r2_g12_ring_store4(va, ring_v, i4, i0, pos0, T, vas);
}

KERNEL ie_inkling_kv_sconv_vclip_store_ctx_r2(const G float *ka, const G float *vk, const G float *ck, G float *ring_k,
                                              const G float *va, const G float *vv, const G float *cv, G float *ring_v,
                                              G kvt *kc0, G kvt *vc0, float sc_escale, float k_clamp, float v_eps,
                                              const G u32 *ctx, u32 kv_len) {
  const u32 pos = ctx[1];
  ink_r2_g12_kv_step1(ka, vk, ck, ring_k, va, vv, cv, ring_v, kc0, vc0, sc_escale, k_clamp, v_eps, pos, kv_len);
}

#undef INK_R2_G12_TTILE
