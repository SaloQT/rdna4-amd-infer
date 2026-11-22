/* ============================================================================
 * Inkling-2 Round 8: Prefill Fused Short-Conv Pipelines (ink_mask2 bit 75, R18)
 * Target: RX 9070 XT (gfx1201, wave32, NT=256)
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r8_sc_ /
 * INK_R8_SC_ and all macros are #undef'd at EOF.
 *
 * Exported kernels (100% bitwise-exact 0-bit-diff parity for T=1..2048 prefill):
 *
 * 1. KERNEL ie_inkling_kv_sconv_vclip_store_pf_r8(...)
 *    - Fuses the ENTIRE prefill 6-op KV pipeline:
 *        [sconv(k) -> vclip(k) -> sconv(v) -> vclip(v) -> f16 kv_store -> ring_k/v update]
 *      into ONE SINGLE LAUNCH (eliminates 48 separate ie_inkling_kv_ring_store_r2
 *      launches per prefill chunk AND raises TTILE 4 -> 32 for 8x fewer static-weight
 *      & halo token reloads!).
 *    - Race-free single-launch ring update: when ring_cnt != NULL (1 zero-init u32,
 *      auto-reset to 0), all n_tiles WGs read ring_k/ring_v safely first; the last
 *      arriving WG performs agent acquire + coalesced f32x4 store of the last 4
 *      tokens (T-4..T-1) into ring_k & ring_v and resets *ring_cnt = 0u.
 *      (If ring_cnt == NULL, ring is left read-only for external ring commit.)
 *    - Grid: (1u, (T + 31u) / 32u, 1u), NT = 256 (16 WGs at T=512).
 *    - Guard: !kvq8 && !kvq4 && !mk.rec && n_kv == 8u && hd == 128u && kv_len > 0u &&
 *             T >= 1u && k_clamp > 0.0f && k_vclip.v == NULL && k_vclip.norope == 0u &&
 *             v_vclip.v == NULL && v_vclip.clamp == 0.0f && k_sconv.escale == v_sconv.escale &&
 *             16-B aligned pointers & strides (kas, vas, cks, cvs).
 *    - Parity: 0-bit-diff vs ie_inkling_kv_sconv_vclip_store_r2 + ie_inkling_kv_ring_store_r2.
 *
 * 2. KERNEL ie_inkling_sconv_post_pf_r8(...)
 *    - Fuses prefill attn_pc (48 visits) & ffn_pc (48 visits):
 *        [OP_SCONV(static 4-tap, c=NULL) + (OP_ADD(res, sc -> out) |
 *         OP_COPY(res -> blk_dst, sc -> out)) + race-free ring update]
 *      into ONE SINGLE LAUNCH (eliminates 96 ie_sconv_ring_store launches + 8 OP_COPY
 *      launches per prefill chunk AND raises TTILE 8 -> 32 for 4x fewer weight/halo loads!).
 *    - Grid: (((n >> 2u) + 255u) / 256u, (T + 31u) / 32u, 1u) [(3, 16, 1) at dim=2304, T=512].
 *    - Counter: ring_cnt[0..gx-1] (gx zero-init u32s = 12 B for dim=2304, auto-reset to 0;
 *      if ring_cnt == NULL, ring is left read-only).
 *    - Guard: (n & 3u) == 0u && T >= 1u && 16-B aligned pointers & strides (as, rs, os, bs).
 *             (out may alias res when blk_dst == NULL.)
 *    - Parity: 0-bit-diff vs ie_inkling_sconv_add_r2 (+ OP_COPY) + ie_sconv_ring_store.
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R8_SC_TTILE 32u

static inline __attribute__((always_inline)) float ink_r8_sc_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline __attribute__((always_inline)) void ink_r8_sc_load_sw(
    const G float *v, u32 i0, f32x4 *sw0, f32x4 *sw1, f32x4 *sw2, f32x4 *sw3) {
  const G f32x4 *v4 = (const G f32x4 *)v + i0;
  const f32x4 vw0 = v4[0], vw1 = v4[1], vw2 = v4[2], vw3 = v4[3];
  *sw0 = (f32x4){vw0.w, vw1.w, vw2.w, vw3.w};
  *sw1 = (f32x4){vw0.z, vw1.z, vw2.z, vw3.z};
  *sw2 = (f32x4){vw0.y, vw1.y, vw2.y, vw3.y};
  *sw3 = (f32x4){vw0.x, vw1.x, vw2.x, vw3.x};
}

static inline __attribute__((always_inline)) void ink_r8_sc_seed_ring(
    const G float *ring, u32 i0, u32 pos0, f32x4 *x1, f32x4 *x2, f32x4 *x3) {
  *x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (pos0 > 0u) {
    const G f32x4 *rg = (const G f32x4 *)ring + i0;
    const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
    const u32 s1 = (pos0 - 1u) & 3u;
    *x1 = (f32x4){ink_r8_sc_pick4(r0, s1), ink_r8_sc_pick4(r1, s1),
                  ink_r8_sc_pick4(r2, s1), ink_r8_sc_pick4(r3, s1)};
    if (pos0 >= 2u) {
      const u32 s2 = (pos0 - 2u) & 3u;
      *x2 = (f32x4){ink_r8_sc_pick4(r0, s2), ink_r8_sc_pick4(r1, s2),
                    ink_r8_sc_pick4(r2, s2), ink_r8_sc_pick4(r3, s2)};
    }
    if (pos0 >= 3u) {
      const u32 s3 = (pos0 - 3u) & 3u;
      *x3 = (f32x4){ink_r8_sc_pick4(r0, s3), ink_r8_sc_pick4(r1, s3),
                    ink_r8_sc_pick4(r2, s3), ink_r8_sc_pick4(r3, s3)};
    }
  }
}

static inline __attribute__((always_inline)) kv4 ink_r8_sc_f32x4_to_kv4(f32x4 v) {
  float x0 = v.x, x1 = v.y, x2 = v.z, x3 = v.w;
  /* Prevent AMDGPU ISel from folding fp_round_f16(fmul(x, f)) into single-rounded
   * v_fma_mixlo_f16 / v_fma_mixhi_f16 (preserves exact double-rounded f32->f16
   * parity with ie_inkling_vclip_r1 + ie_kv_store). */
  __asm__ volatile("" : "+v"(x0), "+v"(x1), "+v"(x2), "+v"(x3));
  return __builtin_convertvector((f32x4){x0, x1, x2, x3}, kv4);
}

static inline __attribute__((always_inline)) f32x4 ink_r8_sc_conv4(
    f32x4 x0, f32x4 x1, f32x4 x2, f32x4 x3,
    f32x4 w0, f32x4 w1, f32x4 w2, f32x4 w3, float escale) {
#pragma clang fp contract(off)
  const f32x4 y = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3;
  return (x0 + y) * escale;
}

static inline __attribute__((always_inline)) void ink_r8_sc_ring_commit4(
    const G float *a, G float *ring, u32 i4, u32 i0, u32 pos0, u32 T, u32 as) {
  if (T >= 4u) {
    const u32 k0 = T - 4u;
    const u32 s0 = (pos0 + k0) & 3u;
    const u32 t_s0 = k0 + ((0u - s0) & 3u);
    const u32 t_s1 = k0 + ((1u - s0) & 3u);
    const u32 t_s2 = k0 + ((2u - s0) & 3u);
    const u32 t_s3 = k0 + ((3u - s0) & 3u);
    const f32x4 v0 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t_s0 * (unsigned long)as))[i4];
    const f32x4 v1 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t_s1 * (unsigned long)as))[i4];
    const f32x4 v2 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t_s2 * (unsigned long)as))[i4];
    const f32x4 v3 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t_s3 * (unsigned long)as))[i4];
    G f32x4 *rg = (G f32x4 *)ring + i0;
    rg[0] = (f32x4){v0.x, v1.x, v2.x, v3.x};
    rg[1] = (f32x4){v0.y, v1.y, v2.y, v3.y};
    rg[2] = (f32x4){v0.z, v1.z, v2.z, v3.z};
    rg[3] = (f32x4){v0.w, v1.w, v2.w, v3.w};
  } else {
    for (u32 tk = 0u; tk < T; tk++) {
      const f32x4 v = ((const G f32x4 *)((const G u8 *)a + (unsigned long)tk * (unsigned long)as))[i4];
      const u32 slot = (pos0 + tk) & 3u;
      ring[(unsigned long)(i0 + 0u) * 4ul + slot] = v.x;
      ring[(unsigned long)(i0 + 1u) * 4ul + slot] = v.y;
      ring[(unsigned long)(i0 + 2u) * 4ul + slot] = v.z;
      ring[(unsigned long)(i0 + 3u) * 4ul + slot] = v.w;
    }
  }
}

/* ============================================================================
 * Kernel 1: ie_inkling_kv_sconv_vclip_store_pf_r8
 * Prefill 1-launch K/V ShortConv + VClip + f16 KV-Store + Race-Free Ring Commit.
 * ========================================================================== */

KERNEL ie_inkling_kv_sconv_vclip_store_pf_r8(
    const G float *ka, const G float *vk, const G float *ck, G float *ring_k,
    const G float *va, const G float *vv, const G float *cv, G float *ring_v,
    G kvt *kc0, G kvt *vc0, float sc_escale, float k_clamp, float v_eps,
    u32 pos0, u32 T, u32 kv_len, u32 kas, u32 cks, u32 vas, u32 cvs,
    G u32 *ring_cnt) {
  static LDS u32 ink_r8_sc_kv_last;

  if (T == 0u || kv_len == 0u) return;
  const u32 n_tiles = (T + INK_R8_SC_TTILE - 1u) / INK_R8_SC_TTILE;
  const u32 tile = tokid();
  if (tile >= n_tiles) return;

  const u32 t0 = tile * INK_R8_SC_TTILE;
  const u32 t_end = (t0 + INK_R8_SC_TTILE < T) ? (t0 + INK_R8_SC_TTILE) : T;
  const u32 i4 = tid();
  const u32 i0 = i4 << 2u;
  const u32 h = i4 >> 5u;
  const u32 p0 = (pos0 + t0) % kv_len;

  /* K path: 32 tokens per WG -> 1x static weight & halo load per 32 tokens. */
  {
    f32x4 sw0, sw1, sw2, sw3;
    ink_r8_sc_load_sw(vk, i0, &sw0, &sw1, &sw2, &sw3);
    f32x4 x1, x2, x3;
    if (t0 >= 3u) {
      x3 = ((const G f32x4 *)((const G u8 *)ka + (unsigned long)(t0 - 3u) * (unsigned long)kas))[i4];
      x2 = ((const G f32x4 *)((const G u8 *)ka + (unsigned long)(t0 - 2u) * (unsigned long)kas))[i4];
      x1 = ((const G f32x4 *)((const G u8 *)ka + (unsigned long)(t0 - 1u) * (unsigned long)kas))[i4];
    } else {
      ink_r8_sc_seed_ring(ring_k, i0, pos0, &x1, &x2, &x3);
    }
    u32 p = p0;
    for (u32 tk = t0; tk < t_end; tk++) {
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)ka + (unsigned long)tk * (unsigned long)kas))[i4];
      f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
      if (ck) {
        const f32x4 ct = ((const G f32x4 *)((const G u8 *)ck + (unsigned long)tk * (unsigned long)cks))[h];
        w0 += ct.x; w1 += ct.y; w2 += ct.z; w3 += ct.w;
      }
      const f32x4 kx = ink_r8_sc_conv4(x0, x1, x2, x3, w0, w1, w2, w3, sc_escale);
      float ss = (kx.x * kx.x + kx.y * kx.y) + (kx.z * kx.z + kx.w * kx.w);
      ss = wave_sum_all(ss);
      float fk = k_clamp / (__builtin_sqrtf(ss) + 1e-6f);
      if (fk > 1.0f) fk = 1.0f;
      ((G kv4 *)kc0)[(unsigned long)p * 256ul + (unsigned long)i4] = ink_r8_sc_f32x4_to_kv4(kx * fk);
      x3 = x2; x2 = x1; x1 = x0;
      p++;
      if (p == kv_len) p = 0u;
    }
  }

  /* V path: 32 tokens per WG. */
  {
    f32x4 sw0, sw1, sw2, sw3;
    ink_r8_sc_load_sw(vv, i0, &sw0, &sw1, &sw2, &sw3);
    f32x4 x1, x2, x3;
    if (t0 >= 3u) {
      x3 = ((const G f32x4 *)((const G u8 *)va + (unsigned long)(t0 - 3u) * (unsigned long)vas))[i4];
      x2 = ((const G f32x4 *)((const G u8 *)va + (unsigned long)(t0 - 2u) * (unsigned long)vas))[i4];
      x1 = ((const G f32x4 *)((const G u8 *)va + (unsigned long)(t0 - 1u) * (unsigned long)vas))[i4];
    } else {
      ink_r8_sc_seed_ring(ring_v, i0, pos0, &x1, &x2, &x3);
    }
    u32 p = p0;
    for (u32 tk = t0; tk < t_end; tk++) {
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)va + (unsigned long)tk * (unsigned long)vas))[i4];
      f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
      if (cv) {
        const f32x4 ct = ((const G f32x4 *)((const G u8 *)cv + (unsigned long)tk * (unsigned long)cvs))[h];
        w0 += ct.x; w1 += ct.y; w2 += ct.z; w3 += ct.w;
      }
      const f32x4 vx = ink_r8_sc_conv4(x0, x1, x2, x3, w0, w1, w2, w3, sc_escale);
      float ss = (vx.x * vx.x + vx.y * vx.y) + (vx.z * vx.z + vx.w * vx.w);
      ss = wave_sum_all(ss);
      const float fv = 1.0f / __builtin_sqrtf(ss * (1.0f / 128.0f) + v_eps);
      ((G kv4 *)vc0)[(unsigned long)p * 256ul + (unsigned long)i4] = ink_r8_sc_f32x4_to_kv4(vx * fv);
      x3 = x2; x2 = x1; x1 = x0;
      p++;
      if (p == kv_len) p = 0u;
    }
  }

  /* Race-free single-launch ring commit: last arriving tile writes last <=4 tokens to ring_k & ring_v. */
  if (!ring_cnt) return;
  if (n_tiles > 1u) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (tid() == 0u) {
      ink_r8_sc_kv_last =
          (__scoped_atomic_fetch_add(ring_cnt, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == n_tiles - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r8_sc_kv_last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (tid() == 0u) *ring_cnt = 0u;
  }
  ink_r8_sc_ring_commit4(ka, ring_k, i4, i0, pos0, T, kas);
  ink_r8_sc_ring_commit4(va, ring_v, i4, i0, pos0, T, vas);
}

/* ============================================================================
 * Kernel 2: ie_inkling_sconv_post_pf_r8
 * Prefill 1-launch attn_pc / ffn_pc ShortConv + (Add | Block Copy) + Ring Commit.
 * ========================================================================== */

KERNEL ie_inkling_sconv_post_pf_r8(
    const G float *a, const G float *v, G float *ring,
    const G float *res, G float *out, G float *blk_dst,
    u32 n, float escale, u32 pos0, u32 T,
    u32 as, u32 rs, u32 os, u32 bs, G u32 *ring_cnt) {
#pragma clang fp contract(off)
  static LDS u32 ink_r8_sc_post_last;

  if (T == 0u || n == 0u) return;
  const u32 n4 = n >> 2u;
  const u32 gx = wgid();
  const u32 i4 = gx * NT + tid();
  const u32 n_tiles = (T + INK_R8_SC_TTILE - 1u) / INK_R8_SC_TTILE;
  const u32 tile = tokid();
  if (tile >= n_tiles || gx * NT >= n4) return;

  const u32 t0 = tile * INK_R8_SC_TTILE;
  const u32 t_end = (t0 + INK_R8_SC_TTILE < T) ? (t0 + INK_R8_SC_TTILE) : T;

  if (i4 < n4) {
    const u32 i0 = i4 << 2u;
    f32x4 sw0, sw1, sw2, sw3;
    ink_r8_sc_load_sw(v, i0, &sw0, &sw1, &sw2, &sw3);

    f32x4 x1, x2, x3;
    if (t0 >= 3u) {
      x3 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)(t0 - 3u) * (unsigned long)as))[i4];
      x2 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)(t0 - 2u) * (unsigned long)as))[i4];
      x1 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)(t0 - 1u) * (unsigned long)as))[i4];
    } else {
      ink_r8_sc_seed_ring(ring, i0, pos0, &x1, &x2, &x3);
    }

    for (u32 tk = t0; tk < t_end; tk++) {
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)tk * (unsigned long)as))[i4];
      const f32x4 rv = ((const G f32x4 *)((const G u8 *)res + (unsigned long)tk * (unsigned long)rs))[i4];
      const f32x4 sc = ink_r8_sc_conv4(x0, x1, x2, x3, sw0, sw1, sw2, sw3, escale);
      if (blk_dst) {
        ((G f32x4 *)((G u8 *)blk_dst + (unsigned long)tk * (unsigned long)bs))[i4] = rv;
        ((G f32x4 *)((G u8 *)out + (unsigned long)tk * (unsigned long)os))[i4] = sc;
      } else {
        ((G f32x4 *)((G u8 *)out + (unsigned long)tk * (unsigned long)os))[i4] = rv + sc;
      }
      x3 = x2; x2 = x1; x1 = x0;
    }
  }

  if (!ring_cnt) return;
  if (n_tiles > 1u) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (tid() == 0u) {
      ink_r8_sc_post_last =
          (__scoped_atomic_fetch_add(ring_cnt + gx, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == n_tiles - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r8_sc_post_last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (tid() == 0u) ring_cnt[gx] = 0u;
  }

  if (i4 < n4) {
    ink_r8_sc_ring_commit4(a, ring, i4, i4 << 2u, pos0, T, as);
  }
}

#undef INK_R8_SC_TTILE

#endif /* IE_MEGA */
