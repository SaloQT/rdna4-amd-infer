/* ie_inkling_r1_g08.inc -- Vectorized, full-CU Inkling-2 loop_gate, sconv, and vclip kernels
 * for RX 9070 XT (gfx1201, wave32, NT = 256).
 *
 * Kernels exported:
 *   1. KERNEL ie_inkling_loop_gate_r1(const G float *a, const G float *c, const G float *gt, G float *b,
 *                                     u32 n, u32 T, u32 as, u32 cs, u32 gs, u32 bs)
 *      Grid: (gx, 1, 1) where total4 = T * (n / 4u),
 *            gx = (T == 1u) ? ((total4 + 255u) / 256u) : ((total4 + 1023u) / 1024u) [min 1].
 *      Guard: (n & 3u) == 0u && ptrs A,C,GT,B 16-byte aligned &&
 *             (T == 1u || ((as|cs|gs|bs) & 15u) == 0u).
 *      Parity: bitwise-exact vs ie_loop_gate (fp contract off, exact ie_tanhf_s).
 *
 *   2. KERNEL ie_inkling_sconv_r1(const G float *a, const G float *v, const G float *c, G float *ring, G float *b,
 *                                 u32 n, u32 nh, float escale, u32 pos0, u32 T, u32 as, u32 cs, u32 bs)
 *      Grid: ((n / 4u + 255u) / 256u, (T + 7u) / 8u, 1)  [TTILE = 8 tokens/WG].
 *      Guard: (n & 3u) == 0u && (!C || (nh > 0u && ((n / nh) & 3u) == 0u)) &&
 *             ptrs A,B,v,ring(,C) 16-byte aligned && (T == 1u || ((as|bs|(C?cs:0u)) & 15u) == 0u).
 *      Note: Exact same parameter list as ie_sconv. When T == 1u updates ring in-place;
 *            when T > 1u host still runs ie_sconv_ring_store after (avoids inter-WG ring race).
 *      Parity: bitwise-exact vs ie_sconv (fp contract off, exact left-to-right sum order).
 *
 *   3. KERNEL ie_inkling_vclip_r1(const G float *a, G float *b, const G float *v, u32 nh, u32 hd, float clamp,
 *                                 u32 norope, float escale, u32 win, u32 pos0, u32 T, u32 as, u32 bs)
 *      Grid: (gx, 1, 1) where total_h = T * nh:
 *            hd == 128u -> gx = min((total_h + 7u) / 8u, 256u)   [1 wave/head, 8 heads/WG]
 *            hd == 16u  -> gx = min((total_h + 63u) / 64u, 256u) [4 lanes/head, 64 heads/WG].
 *      Guard: (hd == 128u || hd == 16u) && ptrs A,B(,v) 16-byte aligned &&
 *             (T == 1u || ((as|bs) & 15u) == 0u).
 *      Parity: f32-tol (~1e-7 relative, non-negative sum-of-squares reordered for f32x4).
 */

#define INK_R1_G08_SCONV_TTILE 8u

static inline f32x4 ink_r1_g08_loop_gate4(f32x4 va, f32x4 vc, f32x4 vgf, f32x4 vgg) {
#pragma clang fp contract(off)
  const float f0 = ie_tanhf_s(vgf.x), g0 = ie_tanhf_s(vgg.x);
  const float f1 = ie_tanhf_s(vgf.y), g1 = ie_tanhf_s(vgg.y);
  const float f2 = ie_tanhf_s(vgf.z), g2 = ie_tanhf_s(vgg.z);
  const float f3 = ie_tanhf_s(vgf.w), g3 = ie_tanhf_s(vgg.w);
  return (f32x4){
      (1.0f - f0) * va.x + g0 * vc.x,
      (1.0f - f1) * va.y + g1 * vc.y,
      (1.0f - f2) * va.z + g2 * vc.z,
      (1.0f - f3) * va.w + g3 * vc.w,
  };
}

KERNEL ie_inkling_loop_gate_r1(const G float *a, const G float *c, const G float *gt, G float *b, u32 n, u32 T, u32 as,
                               u32 cs, u32 gs, u32 bs) {
#pragma clang fp contract(off)
  const u32 n4 = n >> 2;
  if (T == 1u) {
    const u32 i4 = wgid() * NT + tid();
    if (i4 >= n4) return;
    const f32x4 va = ((const G f32x4 *)a)[i4];
    const f32x4 vc = ((const G f32x4 *)c)[i4];
    const f32x4 vgf = ((const G f32x4 *)gt)[i4];
    const f32x4 vgg = ((const G f32x4 *)gt)[n4 + i4];
    ((G f32x4 *)b)[i4] = ink_r1_g08_loop_gate4(va, vc, vgf, vgg);
    return;
  }

  const u32 total4 = T * n4;
  const u32 stride = ngroups_x() * NT;
  for (u32 idx = wgid() * NT + tid(); idx < total4; idx += stride) {
    const u32 t = idx / n4;
    const u32 i4 = idx - t * n4;
    const G f32x4 *at = (const G f32x4 *)((const G u8 *)a + (unsigned long)t * as);
    const G f32x4 *ct = (const G f32x4 *)((const G u8 *)c + (unsigned long)t * cs);
    const G f32x4 *gtt = (const G f32x4 *)((const G u8 *)gt + (unsigned long)t * gs);
    G f32x4 *bt = (G f32x4 *)((G u8 *)b + (unsigned long)t * bs);
    const f32x4 va = at[i4];
    const f32x4 vc = ct[i4];
    const f32x4 vgf = gtt[i4];
    const f32x4 vgg = gtt[n4 + i4];
    bt[i4] = ink_r1_g08_loop_gate4(va, vc, vgf, vgg);
  }
}

static inline float ink_r1_g08_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

KERNEL ie_inkling_sconv_r1(const G float *a, const G float *v, const G float *c, G float *ring, G float *b, u32 n,
                           u32 nh, float escale, u32 pos0, u32 T, u32 as, u32 cs, u32 bs) {
#pragma clang fp contract(off)
  const u32 n4 = n >> 2;
  const u32 i4 = wgid() * NT + tid();
  if (i4 >= n4) return;
  const u32 i0 = i4 << 2;

  /* Load 4-tap static conv weights once for all tokens in this WG tile. */
  const G f32x4 *v4 = (const G f32x4 *)v + i0;
  const f32x4 vw0 = v4[0], vw1 = v4[1], vw2 = v4[2], vw3 = v4[3];
  const f32x4 sw0 = (f32x4){vw0.w, vw1.w, vw2.w, vw3.w};
  const f32x4 sw1 = (f32x4){vw0.z, vw1.z, vw2.z, vw3.z};
  const f32x4 sw2 = (f32x4){vw0.y, vw1.y, vw2.y, vw3.y};
  const f32x4 sw3 = (f32x4){vw0.x, vw1.x, vw2.x, vw3.x};

  const u32 h = c ? (i0 / (n / nh)) : 0u;

  if (T == 1u) {
    const f32x4 x0 = ((const G f32x4 *)a)[i4];
    f32x4 x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    if (pos0 > 0u) {
      const G f32x4 *rg = (const G f32x4 *)ring + i0;
      const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
      const u32 s1 = (pos0 - 1u) & 3u;
      x1 = (f32x4){ink_r1_g08_pick4(r0, s1), ink_r1_g08_pick4(r1, s1), ink_r1_g08_pick4(r2, s1),
                   ink_r1_g08_pick4(r3, s1)};
      if (pos0 >= 2u) {
        const u32 s2 = (pos0 - 2u) & 3u;
        x2 = (f32x4){ink_r1_g08_pick4(r0, s2), ink_r1_g08_pick4(r1, s2), ink_r1_g08_pick4(r2, s2),
                     ink_r1_g08_pick4(r3, s2)};
      }
      if (pos0 >= 3u) {
        const u32 s3 = (pos0 - 3u) & 3u;
        x3 = (f32x4){ink_r1_g08_pick4(r0, s3), ink_r1_g08_pick4(r1, s3), ink_r1_g08_pick4(r2, s3),
                     ink_r1_g08_pick4(r3, s3)};
      }
    }
    f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
    if (c) {
      const f32x4 ct = ((const G f32x4 *)c)[h];
      w0 += ct.x, w1 += ct.y, w2 += ct.z, w3 += ct.w;
    }
    const f32x4 y = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3;
    ((G f32x4 *)b)[i4] = (x0 + y) * escale;
    const u32 slot = pos0 & 3u;
    ring[(i0 + 0u) * 4u + slot] = x0.x;
    ring[(i0 + 1u) * 4u + slot] = x0.y;
    ring[(i0 + 2u) * 4u + slot] = x0.z;
    ring[(i0 + 3u) * 4u + slot] = x0.w;
    return;
  }

  const u32 t0 = tokid() * INK_R1_G08_SCONV_TTILE;
  if (t0 >= T) return;
  const u32 t_end = (t0 + INK_R1_G08_SCONV_TTILE < T) ? (t0 + INK_R1_G08_SCONV_TTILE) : T;

  /* Prime the 3-token sliding register window [x3, x2, x1] before stepping t = t0 .. t_end - 1. */
  f32x4 x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (t0 >= 3u) {
    x3 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)(t0 - 3u) * as))[i4];
    x2 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)(t0 - 2u) * as))[i4];
    x1 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)(t0 - 1u) * as))[i4];
  } else if (pos0 > 0u) {
    /* t0 == 0u (since TTILE == 8 >= 3): seed from ring for lags 1..3. */
    const G f32x4 *rg = (const G f32x4 *)ring + i0;
    const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
    const u32 s1 = (pos0 - 1u) & 3u;
    x1 = (f32x4){ink_r1_g08_pick4(r0, s1), ink_r1_g08_pick4(r1, s1), ink_r1_g08_pick4(r2, s1),
                 ink_r1_g08_pick4(r3, s1)};
    if (pos0 >= 2u) {
      const u32 s2 = (pos0 - 2u) & 3u;
      x2 = (f32x4){ink_r1_g08_pick4(r0, s2), ink_r1_g08_pick4(r1, s2), ink_r1_g08_pick4(r2, s2),
                   ink_r1_g08_pick4(r3, s2)};
    }
    if (pos0 >= 3u) {
      const u32 s3 = (pos0 - 3u) & 3u;
      x3 = (f32x4){ink_r1_g08_pick4(r0, s3), ink_r1_g08_pick4(r1, s3), ink_r1_g08_pick4(r2, s3),
                   ink_r1_g08_pick4(r3, s3)};
    }
  }

  for (u32 t = t0; t < t_end; t++) {
    const f32x4 x0 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t * as))[i4];
    f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
    if (c) {
      const f32x4 ct = ((const G f32x4 *)((const G u8 *)c + (unsigned long)t * cs))[h];
      w0 += ct.x, w1 += ct.y, w2 += ct.z, w3 += ct.w;
    }
    const f32x4 y = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3;
    ((G f32x4 *)((G u8 *)b + (unsigned long)t * bs))[i4] = (x0 + y) * escale;
    x3 = x2;
    x2 = x1;
    x1 = x0;
  }
}

static inline float ink_r1_g08_vclip_scale(float ss, float inv_hd, float clamp, u32 norope, float escale, u32 win,
                                           u32 pos0, u32 t) {
  float f;
  if (clamp > 0.0f) {
    f = clamp / (__builtin_sqrtf(ss) + 1e-6f);
    if (f > 1.0f) f = 1.0f;
    if (norope) {
      const float ratio = (float)(pos0 + t + 1u) / (float)win;
      if (ratio > 1.0f) f *= 1.0f + escale * __builtin_logf(ratio);
    }
  } else {
    f = 1.0f / __builtin_sqrtf(ss * inv_hd + escale);
  }
  return f;
}

KERNEL ie_inkling_vclip_r1(const G float *a, G float *b, const G float *v, u32 nh, u32 hd, float clamp, u32 norope,
                           float escale, u32 win, u32 pos0, u32 T, u32 as, u32 bs) {
  const u32 total_h = T * nh;

  if (hd == 128u) {
    /* 1 wave (32 lanes) per head, 1 f32x4 per lane, 8 heads per WG. */
    const u32 wv = tid() >> 5, l = lane();
    const u32 step = ngroups_x() * 8u;
    for (u32 idx = wgid() * 8u + wv; idx < total_h; idx += step) {
      const u32 t = (T == 1u) ? 0u : (idx / nh);
      const u32 h = (T == 1u) ? idx : (idx - t * nh);
      const G f32x4 *ah = (const G f32x4 *)((const G u8 *)a + (unsigned long)t * as) + (unsigned long)h * 32u;
      G f32x4 *bh = (G f32x4 *)((G u8 *)b + (unsigned long)t * bs) + (unsigned long)h * 32u;
      const G f32x4 *vp = v ? ((const G f32x4 *)v + (unsigned long)h * 32u) : 0;

      f32x4 xv = ah[l];
      const f32x4 vv = vp ? vp[l] : (f32x4){1.0f, 1.0f, 1.0f, 1.0f};
      if (clamp > 0.0f && vp) xv *= vv;
      float ss = (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
      ss = wave_sum_all(ss);
      const float f = ink_r1_g08_vclip_scale(ss, 1.0f / 128.0f, clamp, norope, escale, win, pos0, t);
      f32x4 outv = xv * f;
      if (clamp == 0.0f && vp) outv *= vv;
      bh[l] = outv;
    }
    return;
  }

  if (hd == 16u) {
    /* 4 lanes per head (1 f32x4 per lane), 8 heads per wave, 64 heads per WG. */
    const u32 sub = tid() >> 2;
    const u32 sl = tid() & 3u;
    const u32 step = ngroups_x() * 64u;
    for (u32 idx = wgid() * 64u + sub; idx < total_h; idx += step) {
      const u32 t = (T == 1u) ? 0u : (idx / nh);
      const u32 h = (T == 1u) ? idx : (idx - t * nh);
      const G f32x4 *ah = (const G f32x4 *)((const G u8 *)a + (unsigned long)t * as) + (unsigned long)h * 4u;
      G f32x4 *bh = (G f32x4 *)((G u8 *)b + (unsigned long)t * bs) + (unsigned long)h * 4u;
      const G f32x4 *vp = v ? ((const G f32x4 *)v + (unsigned long)h * 4u) : 0;

      f32x4 xv = ah[sl];
      const f32x4 vv = vp ? vp[sl] : (f32x4){1.0f, 1.0f, 1.0f, 1.0f};
      if (clamp > 0.0f && vp) xv *= vv;
      float ss = (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
      ss += shfl_xor_f(ss, 2u);
      ss += shfl_xor_f(ss, 1u);
      const float f = ink_r1_g08_vclip_scale(ss, 1.0f / 16.0f, clamp, norope, escale, win, pos0, t);
      f32x4 outv = xv * f;
      if (clamp == 0.0f && vp) outv *= vv;
      bh[sl] = outv;
    }
  }
}

#undef INK_R1_G08_SCONV_TTILE
