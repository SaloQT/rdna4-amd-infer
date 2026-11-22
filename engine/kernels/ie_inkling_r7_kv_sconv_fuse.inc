/* ============================================================================
 * Inkling-2-Mini Round 7 (Fusion C1 + C2): Unified Short-Conv + Clip + KV/Hist
 *                                          & Post-Attn Short-Conv + Residual/BlockCopy
 * Target: RX 9070 XT (gfx1201, wave32, NT=256), Clang 21 bit_cast-safe, 0 spills.
 *
 * Included at EOF of kernels/ie_inkling_kernels.c (after ie_kernels.c).
 * All local symbols/LDS/macros prefixed ink_r7_kv_ / INK_R7_KV_; macros #undef'd at EOF.
 *
 * Replaces across ALL 48 logical visits (40 SWA + 8 Global, T=1..16):
 *   - C1 (KV pipeline): [OP_SCONV(k), OP_VCLIP(k), OP_SCONV(v), OP_VCLIP(v), OP_KV (+ 2x sc_hist / kv_ring)]
 *       * T == 1:     5 -> 1 launch/visit (-192/step all 48, or -160/step SWA-only)
 *       * T == 2..16: 7 -> 1 launch/visit (-288/step all 48, or -240/step SWA-only)
 *       * NOTE on 8 Global layers: old R2 bit 25 hoisted v_dyn before sconv(k) without
 *         checking buf_disjoint(v_dyn, k_dyn/k_raw), so plan_arena clobbered k_dyn!
 *         Emitting v_dyn right after k_dyn in model.c:3209 (1-line move) makes all 48
 *         visits contiguous 5-op with zero aliasing hazard!
 *   - C2 (Post-Attn / Standalone SConv):
 *       * 40 visits (ip % 6 != 0): [OP_SCONV(attn_pc), OP_ADD(block_res, apc -> x1) (+ sc_hist)] -> 1 launch
 *       *  8 visits (ip % 6 == 0): [OP_SCONV(attn_pc), OP_COPY(block_res -> blocks_buf) (+ sc_hist)] -> 1 launch
 *       * T == 1:     -48/step (96 -> 48); T == 2..16: -56/step (104 -> 48).
 *       * (Also works for standalone ffn_pc [OP_SCONV, OP_ADD (+ sc_hist)] if B is off!)
 *
 * Exported KERNELs, Grids, Guards, Disjointness, and Parity:
 *
 * 1. KERNEL ie_inkling_kv_sconv_vclip_store_r7(
 *        const G float *ka, const G float *vk, const G float *ck, G float *ring_k, G float *hist_k,
 *        const G float *va, const G float *vv, const G float *cv, G float *ring_v, G float *hist_v,
 *        G kvt *kc0, G kvt *vc0, float sc_escale, float k_clamp, float v_eps,
 *        u32 pos0, u32 T, u32 kv_len,
 *        u32 kas, u32 cks, u32 vas, u32 cvs, u32 hrs)
 *    - Grid:   (1u, 1u, 1u) for T = 1..16 (8 waves = 8 KV heads, 32 lanes x f32x4 = 1024 floats).
 *              Single WG owns all t = 0..T-1 -> ZERO inter-WG race!
 *              When hist_k != NULL (b->sc_defer), writes sc_hist_k/v inline (hrs = b->sc_hist_stride
 *              in floats) and leaves ring_k/v read-only.
 *              When hist_k == NULL (T==1 or !sc_defer), updates ring_k/v in-place from the
 *              4-token register window (0 re-reads, 0 second kernel!).
 *    - Guard:  !mk.rec && T >= 1u && T <= 16u && !b->kvq8 && !b->kvq4 &&
 *              m->n_kv == 8u && m->hd == 128u && o0->n == 1024u && ov->n == 1024u &&
 *              okc->clamp > 0.0f && !okc->v && !okc->norope && !ovc->v && ovc->clamp == 0.0f &&
 *              o0->escale == ov->escale && g->kv_len[okv->kvi] > 0u &&
 *              16-byte alignment on ka, vk, ring_k, va, vv, ring_v, kc0, vc0 (, ck, cv, hist_k, hist_v) &&
 *              (T == 1u || ((kas | vas | (ck ? cks : 0u) | (cv ? cvs : 0u)) & 15u) == 0u) &&
 *              (!hist_k || (hrs & 3u) == 0u).
 *              If still using has_vdyn hoisting without model.c reorder, MUST additionally check:
 *              buf_disjoint(g, ovd->b, o0->a) && (o0->c < 0 || buf_disjoint(g, ovd->b, o0->c)).
 *    - Res:    0 B LDS, <= 64 VGPR, 0 spills, 4 WGs/CU.
 *    - Parity: 100% bitwise-exact (0-bit-diff) vs ie_inkling_sconv_r1 + ie_inkling_vclip_r1 +
 *              ie_kv_store[_t] + ie_inkling_sconv_hist_r4 / ie_sconv_ring_store.
 *
 * 2. KERNEL ie_inkling_sconv_post_r7(
 *        const G float *a, const G float *v, G float *ring,
 *        const G float *res, G float *out, G float *blk_dst, G float *hist,
 *        u32 n, float escale, u32 pos0, u32 T,
 *        u32 as, u32 rs, u32 os, u32 bds, u32 hrs)
 *    - Grid:   (((n >> 2u) + 255u) / 256u, 1u, 1u)  [(3, 1, 1) for dim=2304, T=1..16]
 *    - Modes:
 *        * blk_dst == NULL: [OP_SCONV(c=-1), OP_ADD(res, sc -> out)] (40 visits attn_pc, 48 ffn_pc).
 *        * blk_dst != NULL: [OP_SCONV(c=-1 -> out), OP_COPY(res -> blk_dst)] (8 ip%6==0 visits).
 *        * hist != NULL (b->sc_defer): writes hist[t * hrs + i] = a[t][i] inline, ring read-only.
 *        * hist == NULL (T==1 or !sc_defer): updates ring in-place from 4-token register window.
 *    - Guard:  !mk.rec && T >= 1u && T <= 16u && o0->c < 0 && (n & 3u) == 0u &&
 *              16-byte alignment on a, v, ring, res, out (, blk_dst, hist) &&
 *              (T == 1u || ((as | rs | os | (blk_dst ? bds : 0u)) & 15u) == 0u) &&
 *              (!hist || (hrs & 3u) == 0u).
 *              Disjointness:
 *                - add mode: out may alias res; if T > 1u require buf_disjoint(o0->a, o1->c).
 *                - copy mode: buf_disjoint(o0->a, o1->b) && buf_disjoint(o0->b, o1->b) &&
 *                             (T == 1u || buf_disjoint(o0->a, o0->b)).
 *    - Model:  ZERO model.c edits required.
 *    - Res:    0 B LDS, <= 56 VGPR, 0 spills, 4 WGs/CU.
 *    - Parity: 100% bitwise-exact (0-bit-diff) vs ie_inkling_sconv_r1 + (ie_add[_t] | ie_copy_rows) +
 *              (ie_inkling_sconv_hist_r4 | ie_sconv_ring_store).
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(kvt) == 2u, "ink_r7_kv requires f16 KV cache");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r7_kv float/u32 must be 4 bytes");

static inline __attribute__((always_inline)) float ink_r7_kv_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline __attribute__((always_inline)) void ink_r7_kv_load_sw(
    const G float *v, u32 i0, f32x4 *sw0, f32x4 *sw1, f32x4 *sw2, f32x4 *sw3) {
  const G f32x4 *v4 = (const G f32x4 *)v + i0;
  const f32x4 vw0 = v4[0], vw1 = v4[1], vw2 = v4[2], vw3 = v4[3];
  *sw0 = (f32x4){vw0.w, vw1.w, vw2.w, vw3.w};
  *sw1 = (f32x4){vw0.z, vw1.z, vw2.z, vw3.z};
  *sw2 = (f32x4){vw0.y, vw1.y, vw2.y, vw3.y};
  *sw3 = (f32x4){vw0.x, vw1.x, vw2.x, vw3.x};
}

static inline __attribute__((always_inline)) void ink_r7_kv_seed_ring(
    const G float *ring, u32 i0, u32 pos0, f32x4 *x1, f32x4 *x2, f32x4 *x3) {
  *x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (pos0 > 0u) {
    const G f32x4 *rg = (const G f32x4 *)ring + i0;
    const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
    const u32 s1 = (pos0 - 1u) & 3u;
    *x1 = (f32x4){ink_r7_kv_pick4(r0, s1), ink_r7_kv_pick4(r1, s1),
                  ink_r7_kv_pick4(r2, s1), ink_r7_kv_pick4(r3, s1)};
    if (pos0 >= 2u) {
      const u32 s2 = (pos0 - 2u) & 3u;
      *x2 = (f32x4){ink_r7_kv_pick4(r0, s2), ink_r7_kv_pick4(r1, s2),
                    ink_r7_kv_pick4(r2, s2), ink_r7_kv_pick4(r3, s2)};
    }
    if (pos0 >= 3u) {
      const u32 s3 = (pos0 - 3u) & 3u;
      *x3 = (f32x4){ink_r7_kv_pick4(r0, s3), ink_r7_kv_pick4(r1, s3),
                    ink_r7_kv_pick4(r2, s3), ink_r7_kv_pick4(r3, s3)};
    }
  }
}

/* Exact non-FMA 4-tap shortconv matching ie_inkling_sconv_r1 & ie_sconv. */
static inline __attribute__((always_inline)) f32x4 ink_r7_kv_sconv4(
    f32x4 x0, f32x4 x1, f32x4 x2, f32x4 x3,
    f32x4 w0, f32x4 w1, f32x4 w2, f32x4 w3, float escale) {
#pragma clang fp contract(off)
  const f32x4 y = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3;
  return (x0 + y) * escale;
}

static inline __attribute__((always_inline)) kv4 ink_r7_kv_f32x4_to_kv4(f32x4 v) {
  float x0 = v.x, x1 = v.y, x2 = v.z, x3 = v.w;
  /* Prevent AMDGPU ISel from folding fp_round_f16(fmul(x, f)) into single-rounded
   * v_fma_mixlo_f16 / v_fma_mixhi_f16 (preserves exact double-rounded f32->f16
   * parity with ie_inkling_vclip_r1 + ie_kv_store). */
  __asm__ volatile("" : "+v"(x0), "+v"(x1), "+v"(x2), "+v"(x3));
  return __builtin_convertvector((f32x4){x0, x1, x2, x3}, kv4);
}

static inline __attribute__((always_inline)) f32x4 ink_r7_kv_vclip_k(f32x4 kx, float k_clamp) {
  float ss = (kx.x * kx.x + kx.y * kx.y) + (kx.z * kx.z + kx.w * kx.w);
  ss = wave_sum_all(ss);
  float fk = k_clamp / (__builtin_sqrtf(ss) + 1e-6f);
  if (fk > 1.0f) fk = 1.0f;
  return kx * fk;
}

static inline __attribute__((always_inline)) f32x4 ink_r7_kv_vclip_v(f32x4 vx, float v_eps) {
  float ss = (vx.x * vx.x + vx.y * vx.y) + (vx.z * vx.z + vx.w * vx.w);
  ss = wave_sum_all(ss);
  const float fv = 1.0f / __builtin_sqrtf(ss * (1.0f / 128.0f) + v_eps);
  return vx * fv;
}

/* Commit the last min(T, 4) tokens held in shift-window [x4(T-4), x3(T-3), x2(T-2), x1(T-1)]
 * directly into ring[i0..i0+3][0..3] with ZERO global re-reads. */
static inline __attribute__((always_inline)) void ink_r7_kv_ring_commit(
    G float *ring, u32 i0, u32 pos0, u32 T, f32x4 x1, f32x4 x2, f32x4 x3, f32x4 x4) {
  if (T >= 4u) {
    /* s0 = ring slot of token (T-4) = x4; s0+1 -> x3; s0+2 -> x2; s0+3 -> x1. */
    const u32 s0 = (pos0 + T - 4u) & 3u;
    const f32x4 v0 = (s0 == 0u) ? x4 : ((s0 == 3u) ? x3 : ((s0 == 2u) ? x2 : x1));
    const f32x4 v1 = (s0 == 1u) ? x4 : ((s0 == 0u) ? x3 : ((s0 == 3u) ? x2 : x1));
    const f32x4 v2 = (s0 == 2u) ? x4 : ((s0 == 1u) ? x3 : ((s0 == 0u) ? x2 : x1));
    const f32x4 v3 = (s0 == 3u) ? x4 : ((s0 == 2u) ? x3 : ((s0 == 1u) ? x2 : x1));
    G f32x4 *rg = (G f32x4 *)ring + i0;
    rg[0] = (f32x4){v0.x, v1.x, v2.x, v3.x};
    rg[1] = (f32x4){v0.y, v1.y, v2.y, v3.y};
    rg[2] = (f32x4){v0.z, v1.z, v2.z, v3.z};
    rg[3] = (f32x4){v0.w, v1.w, v2.w, v3.w};
  } else {
    /* T in {1, 2, 3}: write oldest -> newest into their ring slots. */
    if (T >= 3u) {
      const u32 sl = (pos0 + T - 3u) & 3u;
      ring[(i0 + 0u) * 4u + sl] = x3.x;
      ring[(i0 + 1u) * 4u + sl] = x3.y;
      ring[(i0 + 2u) * 4u + sl] = x3.z;
      ring[(i0 + 3u) * 4u + sl] = x3.w;
    }
    if (T >= 2u) {
      const u32 sl = (pos0 + T - 2u) & 3u;
      ring[(i0 + 0u) * 4u + sl] = x2.x;
      ring[(i0 + 1u) * 4u + sl] = x2.y;
      ring[(i0 + 2u) * 4u + sl] = x2.z;
      ring[(i0 + 3u) * 4u + sl] = x2.w;
    }
    {
      const u32 sl = (pos0 + T - 1u) & 3u;
      ring[(i0 + 0u) * 4u + sl] = x1.x;
      ring[(i0 + 1u) * 4u + sl] = x1.y;
      ring[(i0 + 2u) * 4u + sl] = x1.z;
      ring[(i0 + 3u) * 4u + sl] = x1.w;
    }
  }
}

/* ============================================================================
 * Kernel 1 (C1): Fused K/V Short-Conv + VClip + f16 KV Store [+ Inline sc_hist / Ring]
 *   Grid: (1, 1, 1) for T = 1..16 (8 waves = 8 KV heads, 1024 floats, 0 LDS, 0 barriers).
 * ========================================================================== */

KERNEL ie_inkling_kv_sconv_vclip_store_r7(
    const G float *ka, const G float *vk, const G float *ck, G float *ring_k, G float *hist_k,
    const G float *va, const G float *vv, const G float *cv, G float *ring_v, G float *hist_v,
    G kvt *kc0, G kvt *vc0, float sc_escale, float k_clamp, float v_eps,
    u32 pos0, u32 T, u32 kv_len,
    u32 kas, u32 cks, u32 vas, u32 cvs, u32 hrs) {
  if (T == 0u || kv_len == 0u || !ka || !vk || !ring_k || !va || !vv || !ring_v || !kc0 || !vc0) return;

  const u32 i4 = tid();
  const u32 i0 = i4 << 2u;
  const u32 h = i4 >> 5u;
  const u32 p0 = pos0 % kv_len;

  /* K path: sconv(k) -> per-head L2 vclip(k) -> f16 kc store + inline hist_k or ring_k. */
  {
    f32x4 sw0, sw1, sw2, sw3;
    ink_r7_kv_load_sw(vk, i0, &sw0, &sw1, &sw2, &sw3);
    f32x4 x1, x2, x3, x4 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    ink_r7_kv_seed_ring(ring_k, i0, pos0, &x1, &x2, &x3);

    u32 p = p0;
    for (u32 t = 0u; t < T; t++) {
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)ka + (unsigned long)t * kas))[i4];
      if (hist_k) ((G f32x4 *)(hist_k + (unsigned long)t * hrs))[i4] = x0;

      f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
      if (ck) {
        const f32x4 ct = ((const G f32x4 *)((const G u8 *)ck + (unsigned long)t * cks))[h];
        w0 += ct.x; w1 += ct.y; w2 += ct.z; w3 += ct.w;
      }
      const f32x4 kx = ink_r7_kv_sconv4(x0, x1, x2, x3, w0, w1, w2, w3, sc_escale);
      const f32x4 kxc = ink_r7_kv_vclip_k(kx, k_clamp);
      ((G kv4 *)kc0)[(unsigned long)p * 256ul + i4] = ink_r7_kv_f32x4_to_kv4(kxc);

      x4 = x3; x3 = x2; x2 = x1; x1 = x0;
      p++;
      if (p == kv_len) p = 0u;
    }
    if (!hist_k) ink_r7_kv_ring_commit(ring_k, i0, pos0, T, x1, x2, x3, x4);
  }

  /* V path: sconv(v) -> per-head RMSNorm vclip(v) -> f16 vc store + inline hist_v or ring_v. */
  {
    f32x4 sw0, sw1, sw2, sw3;
    ink_r7_kv_load_sw(vv, i0, &sw0, &sw1, &sw2, &sw3);
    f32x4 x1, x2, x3, x4 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    ink_r7_kv_seed_ring(ring_v, i0, pos0, &x1, &x2, &x3);

    u32 p = p0;
    for (u32 t = 0u; t < T; t++) {
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)va + (unsigned long)t * vas))[i4];
      if (hist_v) ((G f32x4 *)(hist_v + (unsigned long)t * hrs))[i4] = x0;

      f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
      if (cv) {
        const f32x4 ct = ((const G f32x4 *)((const G u8 *)cv + (unsigned long)t * cvs))[h];
        w0 += ct.x; w1 += ct.y; w2 += ct.z; w3 += ct.w;
      }
      const f32x4 vx = ink_r7_kv_sconv4(x0, x1, x2, x3, w0, w1, w2, w3, sc_escale);
      const f32x4 vxc = ink_r7_kv_vclip_v(vx, v_eps);
      ((G kv4 *)vc0)[(unsigned long)p * 256ul + i4] = ink_r7_kv_f32x4_to_kv4(vxc);

      x4 = x3; x3 = x2; x2 = x1; x1 = x0;
      p++;
      if (p == kv_len) p = 0u;
    }
    if (!hist_v) ink_r7_kv_ring_commit(ring_v, i0, pos0, T, x1, x2, x3, x4);
  }
}

/* ============================================================================
 * Kernel 2 (C2): Post-Attention Short-Conv + Residual Add OR Block-Save Copy
 *                [+ Inline sc_hist / Ring Update] for T = 1..16
 *   Grid: (((n >> 2u) + 255u) / 256u, 1u, 1u)  [(3, 1, 1) for dim=2304; 0 LDS, 0 barriers].
 * ========================================================================== */

KERNEL ie_inkling_sconv_post_r7(
    const G float *a, const G float *v, G float *ring,
    const G float *res, G float *out, G float *blk_dst, G float *hist,
    u32 n, float escale, u32 pos0, u32 T,
    u32 as, u32 rs, u32 os, u32 bds, u32 hrs) {
#pragma clang fp contract(off)
  if (T == 0u || n == 0u || (n & 3u) != 0u || !a || !v || !ring || !res || !out) return;

  const u32 n4 = n >> 2u;
  const u32 i4 = wgid() * NT + tid();
  if (i4 >= n4) return;
  const u32 i0 = i4 << 2u;

  f32x4 sw0, sw1, sw2, sw3;
  ink_r7_kv_load_sw(v, i0, &sw0, &sw1, &sw2, &sw3);

  f32x4 x1, x2, x3, x4 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  ink_r7_kv_seed_ring(ring, i0, pos0, &x1, &x2, &x3);

  if (!blk_dst) {
    /* Add mode (40 attn_pc visits + optional 48 ffn_pc visits): out = res + sconv(a). */
    for (u32 t = 0u; t < T; t++) {
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t * as))[i4];
      const f32x4 rv = ((const G f32x4 *)((const G u8 *)res + (unsigned long)t * rs))[i4];
      if (hist) ((G f32x4 *)(hist + (unsigned long)t * hrs))[i4] = x0;

      const f32x4 sc = ink_r7_kv_sconv4(x0, x1, x2, x3, sw0, sw1, sw2, sw3, escale);
      ((G f32x4 *)((G u8 *)out + (unsigned long)t * os))[i4] = rv + sc;
      x4 = x3; x3 = x2; x2 = x1; x1 = x0;
    }
  } else {
    /* Block-save copy mode (8 ip%6==0 visits): out = sconv(a); blk_dst = res. */
    for (u32 t = 0u; t < T; t++) {
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t * as))[i4];
      const f32x4 rv = ((const G f32x4 *)((const G u8 *)res + (unsigned long)t * rs))[i4];
      if (hist) ((G f32x4 *)(hist + (unsigned long)t * hrs))[i4] = x0;

      const f32x4 sc = ink_r7_kv_sconv4(x0, x1, x2, x3, sw0, sw1, sw2, sw3, escale);
      ((G f32x4 *)((G u8 *)out + (unsigned long)t * os))[i4] = sc;
      ((G f32x4 *)((G u8 *)blk_dst + (unsigned long)t * bds))[i4] = rv;
      x4 = x3; x3 = x2; x2 = x1; x1 = x0;
    }
  }

  if (!hist) ink_r7_kv_ring_commit(ring, i0, pos0, T, x1, x2, x3, x4);
}

#endif /* !IE_MEGA */
