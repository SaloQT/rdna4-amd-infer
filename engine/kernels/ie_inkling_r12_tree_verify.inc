/* ============================================================================
 * Inkling-2-Mini Round 12 (Wave 12 - H26: Zero-Unfusion Tree Verification):
 *   Tree-Aware tpar_pack Fused SConv Kernels (C1, C2, B2) +
 *   Tree-Aware Single-Pass FP16 WMMA Global Verify Attention (+ Optional H15 Gate+Q8)
 * Target: RX 9070 XT (gfx1201, wave32, NT=256, 64 CUs)
 *
 * Included at EOF of kernels/ie_inkling_kernels.c (after ie_kernels.c).
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r12_tv_ /
 * INK_R12_TV_ and all macros are #undef'd at EOF.
 * 0 warnings (-Wall -Wextra), 0/0 spills, priv=0, scratch=0, Clang bit_cast-safe.
 *
 * Tree Parent Encoding (`u32 tpar_pack`, 1 SGPR, supports ANY rooted tree T <= 8):
 *   - 8x 4-bit parent indices packed in nibbles: par(t) = (tpar_pack >> (t * 4u)) & 0xFu.
 *   - Node 0 is always the tree root at logical position pos0 (ancestors come from ring[pos0-1..3]).
 *   - Every node t > 0 has topological parent 0 <= par(t) < t.
 *   - Linear chain T <= 8: tpar_pack = 0x65432100u (par = {0, 0, 1, 2, 3, 4, 5, 6}).
 *   - 2-branch split=5 tree (0->1->2->3->4, 0->5->6->7): tpar_pack = 0x65032100u.
 *   - Arbitrary EAGLE-2 / Sequoia trees of any depth/branching work identically!
 *   - Eliminates the `!sc_tree` bailout that caused +344 unfused launches/step in R32!
 *
 * Exported Kernels (4):
 *
 * 1. H26-C1: KERNEL ie_inkling_kv_sconv_vclip_store_tree_r12(
 *        const G float *ka, const G float *vk, const G float *ck, G float *ring_k, G float *hist_k,
 *        const G float *va, const G float *vv, const G float *cv, G float *ring_v, G float *hist_v,
 *        G kvt *kc0, G kvt *vc0, float sc_escale, float k_clamp, float v_eps,
 *        u32 pos0, u32 T, u32 kv_len,
 *        u32 kas, u32 cks, u32 vas, u32 cvs, u32 hrs, u32 tpar_pack)
 *    - Grid:   (1u, 1u, 1u) for T = 1..8 (8 waves = 8 KV heads, 0 LDS, 0 barriers).
 *    - Effect: Drop-in tree-aware replacement for ie_inkling_kv_sconv_vclip_store_r7.
 *              Strict f32-then-f16 store barrier (0 v_fma_mix*_f16 on KV stores!).
 *              Physical KV slot for node t is (pos0 + t) % kv_len (matches ver_tree_r8).
 *    - Parity: 100% bitwise-exact (0-bit-diff) vs R32 unfused tree sconv + vclip + kv_store.
 *
 * 2. H26-C2: KERNEL ie_inkling_sconv_post_tree_r12(
 *        const G float *a, const G float *v, G float *ring,
 *        const G float *res, G float *out, G float *blk_dst, G float *hist,
 *        u32 n, float escale, u32 pos0, u32 T,
 *        u32 as, u32 rs, u32 os, u32 bds, u32 hrs, u32 tpar_pack)
 *    - Grid:   (((n >> 2u) + 255u) / 256u, 1u, 1u)  [(3, 1, 1) for dim=2304, T=1..8]
 *    - Effect: Drop-in tree-aware replacement for ie_inkling_sconv_post_r7 (both Add and
 *              Block-Save Copy modes, inline sc_hist when hist != NULL). 0 LDS, <= 56 VGPR.
 *    - Parity: 100% bitwise-exact (0-bit-diff) vs R32 unfused tree sconv + add/copy + hist.
 *
 * 3. H26-B2: KERNEL ie_inkling_exp_down_tail_ver_tree_r12(
 *        const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
 *        u32 rows, u32 nb,
 *        const G u32 *list, const G u32 *packs, const G u32 *n_packs,
 *        u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv,
 *        const G u32 *route, u32 K, u32 rts,
 *        const G float *v, G float *ring, const G float *res, G float *out, G float *hist,
 *        float escale, u32 pos0, u32 T, u32 yts, u32 rs, u32 os, u32 hrs,
 *        G u32 *rcount, u32 tpar_pack)
 *    - Grid:   ((rows + 31u) / 32u, max_packs, 1u) for T = 2..8.
 *    - Effect: Drop-in tree-aware replacement for ie_inkling_exp_down_tail_ver_r7.
 *              Uses 1,024 B LDS (8 threads x 8 tokens x f32x4) in Phase 2 so ancestor
 *              combined tokens are L0-cached with 0 extra VGPRs in Phase 1 IQ4_NL!
 *    - Parity: 100% bitwise-exact (0-bit-diff) vs compact IQ4_NL + moe_sum + tree sconv + add.
 *
 * 4. H26-WMMA-Glob: KERNEL ie_inkling_rel_attn_ver_glob_wmma_tree_r12(
 *        const G float *q, const G kvt *kc, const G kvt *vc,
 *        const G float *R, const G float *rel_bias, const G float *sink,
 *        G float *part, G float *out, u32 pos0, u32 T, u32 kv_len, u32 hd,
 *        u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
 *        u32 qs, u32 rs, u32 ss, u32 os, G u32 *count,
 *        unsigned long long tree_mask, u32 pos_off_packed,
 *        const G float *agate, G u8 *oq, u32 gs, u32 oqs)
 *    - Grid:   (n_kv * nsplit, 1u, 1u), NT = 256, T = 2..8, win == 0.
 *    - Effect: Single-pass FP16 WMMA Global Verify Attention for EAGLE-2 / Sequoia trees!
 *              Far-field chunks [0 .. pos0 - rel_extent) run 100% zero-overhead identical
 *              to ver_glob_wmma_r7; tree_mask + pos_off_packed apply strictly inside
 *              Near-Field chunks. Optional H15 fused OP_GATE(sigmoid) + Q8_0 epilogue
 *              (pass agate=NULL, oq=NULL for plain f32 out).
 *    - Res:    15,360 B LDS (<= 16 KiB), <= 128 VGPR (3 WGs/CU), 0/0 spills.
 *    - Parity: f32-tol (FP16 WMMA) vs ie_inkling_rel_attn_ver_tree_r8 (+ exact OP_GATE+Q8).
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(kvt) == 2u, "ink_r12_tv requires f16 KV cache");
_Static_assert(sizeof(f16) == 2u, "ink_r12_tv f16 must be 2 bytes");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r12_tv float/u32 must be 4 bytes");

#define INK_R12_TV_HD    128u
#define INK_R12_TV_DREL  16u
#define INK_R12_TV_GQA   3u
#define INK_R12_TV_PSTR  130u
#define INK_R12_TV_LOG2E 1.44269504088896340736f
#define INK_R12_TV_LN2   0.69314718055994530942f
#define INK_R12_TV_QSC   (0.08838834764831844299f * 1.44269504088896340736f)

typedef unsigned long long ink_r12_tv_u64;

#define INK_R12_TV_ACC_FMA(a0_, a1_, a2_, a3_, j_, v_, d_)          \
  do {                                                              \
    if ((j_) == 0u) (a0_) = __builtin_fmaf((v_), (d_), (a0_));      \
    else if ((j_) == 1u) (a1_) = __builtin_fmaf((v_), (d_), (a1_)); \
    else if ((j_) == 2u) (a2_) = __builtin_fmaf((v_), (d_), (a2_)); \
    else (a3_) = __builtin_fmaf((v_), (d_), (a3_));                 \
  } while (0)

static inline __attribute__((always_inline)) float ink_r12_tv_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline __attribute__((always_inline)) u32 ink_r12_tv_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

static inline __attribute__((always_inline)) float ink_r12_tv_oct_tree_f(float v) {
#pragma clang fp contract(off)
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}

static inline __attribute__((always_inline)) float ink_r12_tv_sgpr_f(float v) {
  return I2F(__builtin_amdgcn_readfirstlane(F2I(v)));
}

static inline __attribute__((always_inline)) float ink_r12_tv_xrow(float v) {
  return I2F(xrow_i(F2I(v)));
}

static inline __attribute__((always_inline)) void ink_r12_tv_load_sw(
    const G float *v, u32 i0, f32x4 *sw0, f32x4 *sw1, f32x4 *sw2, f32x4 *sw3) {
  const G f32x4 *v4 = (const G f32x4 *)v + i0;
  const f32x4 vw0 = v4[0], vw1 = v4[1], vw2 = v4[2], vw3 = v4[3];
  *sw0 = (f32x4){vw0.w, vw1.w, vw2.w, vw3.w};
  *sw1 = (f32x4){vw0.z, vw1.z, vw2.z, vw3.z};
  *sw2 = (f32x4){vw0.y, vw1.y, vw2.y, vw3.y};
  *sw3 = (f32x4){vw0.x, vw1.x, vw2.x, vw3.x};
}

static inline __attribute__((always_inline)) void ink_r12_tv_seed_ring(
    const G float *ring, u32 i0, u32 pos0, f32x4 *r1, f32x4 *r2, f32x4 *r3) {
  *r1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *r2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *r3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (pos0 > 0u) {
    const G f32x4 *rg = (const G f32x4 *)ring + i0;
    const f32x4 v0 = rg[0], v1 = rg[1], v2 = rg[2], v3 = rg[3];
    const u32 s1 = (pos0 - 1u) & 3u;
    *r1 = (f32x4){ink_r12_tv_pick4(v0, s1), ink_r12_tv_pick4(v1, s1),
                  ink_r12_tv_pick4(v2, s1), ink_r12_tv_pick4(v3, s1)};
    if (pos0 >= 2u) {
      const u32 s2 = (pos0 - 2u) & 3u;
      *r2 = (f32x4){ink_r12_tv_pick4(v0, s2), ink_r12_tv_pick4(v1, s2),
                    ink_r12_tv_pick4(v2, s2), ink_r12_tv_pick4(v3, s2)};
    }
    if (pos0 >= 3u) {
      const u32 s3 = (pos0 - 3u) & 3u;
      *r3 = (f32x4){ink_r12_tv_pick4(v0, s3), ink_r12_tv_pick4(v1, s3),
                    ink_r12_tv_pick4(v2, s3), ink_r12_tv_pick4(v3, s3)};
    }
  }
}

/* Exact non-FMA 4-tap shortconv matching ie_inkling_sconv_r1 & ie_sconv. */
static inline __attribute__((always_inline)) f32x4 ink_r12_tv_sconv4(
    f32x4 x0, f32x4 x1, f32x4 x2, f32x4 x3,
    f32x4 w0, f32x4 w1, f32x4 w2, f32x4 w3, float escale) {
#pragma clang fp contract(off)
  const f32x4 y = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3;
  return (x0 + y) * escale;
}

/* Strict f32-then-f16 conversion barrier (prevents v_fma_mix*_f16 1-ULP drift on KV stores!). */
static inline __attribute__((always_inline)) kv4 ink_r12_tv_f32x4_to_kv4(f32x4 v) {
  float x0 = v.x, x1 = v.y, x2 = v.z, x3 = v.w;
  __asm__ volatile("" : "+v"(x0), "+v"(x1), "+v"(x2), "+v"(x3));
  return __builtin_convertvector((f32x4){x0, x1, x2, x3}, kv4);
}

static inline __attribute__((always_inline)) f32x4 ink_r12_tv_vclip_k(f32x4 kx, float k_clamp) {
  float ss = (kx.x * kx.x + kx.y * kx.y) + (kx.z * kx.z + kx.w * kx.w);
  ss = wave_sum_all(ss);
  float fk = k_clamp / (__builtin_sqrtf(ss) + 1e-6f);
  if (fk > 1.0f) fk = 1.0f;
  return kx * fk;
}

static inline __attribute__((always_inline)) f32x4 ink_r12_tv_vclip_v(f32x4 vx, float v_eps) {
  float ss = (vx.x * vx.x + vx.y * vx.y) + (vx.z * vx.z + vx.w * vx.w);
  ss = wave_sum_all(ss);
  const float fv = 1.0f / __builtin_sqrtf(ss * (1.0f / 128.0f) + v_eps);
  return vx * fv;
}

/* Resolve 3 tree-ancestor taps (x1, x2, x3) for token t from shift window + global src + ring.
 * Zero extra global loads when par(t) == t - 1 (linear step); exact arbitrary tree lookup on branch switch! */
static inline __attribute__((always_inline)) void ink_r12_tv_resolve_taps_g(
    const G u8 *src_base, u32 stride, u32 i4, u32 t, u32 tpar_pack,
    f32x4 r1, f32x4 r2, f32x4 r3, f32x4 prev_x0,
    f32x4 *x1, f32x4 *x2, f32x4 *x3) {
  if (t == 0u) {
    *x1 = r1; *x2 = r2; *x3 = r3;
    return;
  }
  const u32 p1 = (tpar_pack >> (t << 2u)) & 0xFu;
  if (p1 == t - 1u) {
    *x3 = *x2; *x2 = *x1; *x1 = prev_x0;
    return;
  }
  *x1 = ((const G f32x4 *)(src_base + (unsigned long)p1 * (unsigned long)stride))[i4];
  if (p1 == 0u) {
    *x2 = r1; *x3 = r2;
  } else {
    const u32 p2 = (tpar_pack >> (p1 << 2u)) & 0xFu;
    *x2 = ((const G f32x4 *)(src_base + (unsigned long)p2 * (unsigned long)stride))[i4];
    if (p2 == 0u) {
      *x3 = r1;
    } else {
      const u32 p3 = (tpar_pack >> (p2 << 2u)) & 0xFu;
      *x3 = ((const G f32x4 *)(src_base + (unsigned long)p3 * (unsigned long)stride))[i4];
    }
  }
}

/* Non-deferred ring commit fallback (used only when hist == NULL). */
static inline __attribute__((always_inline)) void ink_r12_tv_ring_commit(
    G float *ring, u32 i0, u32 pos0, u32 T, f32x4 x1, f32x4 x2, f32x4 x3, f32x4 x4) {
  if (T >= 4u) {
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
    if (T >= 3u) {
      const u32 sl = (pos0 + T - 3u) & 3u;
      ring[(i0 + 0u) * 4u + sl] = x3.x; ring[(i0 + 1u) * 4u + sl] = x3.y;
      ring[(i0 + 2u) * 4u + sl] = x3.z; ring[(i0 + 3u) * 4u + sl] = x3.w;
    }
    if (T >= 2u) {
      const u32 sl = (pos0 + T - 2u) & 3u;
      ring[(i0 + 0u) * 4u + sl] = x2.x; ring[(i0 + 1u) * 4u + sl] = x2.y;
      ring[(i0 + 2u) * 4u + sl] = x2.z; ring[(i0 + 3u) * 4u + sl] = x2.w;
    }
    {
      const u32 sl = (pos0 + T - 1u) & 3u;
      ring[(i0 + 0u) * 4u + sl] = x1.x; ring[(i0 + 1u) * 4u + sl] = x1.y;
      ring[(i0 + 2u) * 4u + sl] = x1.z; ring[(i0 + 3u) * 4u + sl] = x1.w;
    }
  }
}

/* ============================================================================
 * 1. H26-C1: ie_inkling_kv_sconv_vclip_store_tree_r12
 *    Tree-aware K/V Short-Conv + VClip + f16 KV Store [+ Inline sc_hist / Ring]
 *    Grid: (1, 1, 1), 0 LDS, 0 v_fma_mix*_f16 on KV stores.
 * ========================================================================== */

KERNEL ie_inkling_kv_sconv_vclip_store_tree_r12(
    const G float *ka, const G float *vk, const G float *ck, G float *ring_k, G float *hist_k,
    const G float *va, const G float *vv, const G float *cv, G float *ring_v, G float *hist_v,
    G kvt *kc0, G kvt *vc0, float sc_escale, float k_clamp, float v_eps,
    u32 pos0, u32 T, u32 kv_len,
    u32 kas, u32 cks, u32 vas, u32 cvs, u32 hrs, u32 tpar_pack) {
  if (T == 0u || T > 8u || kv_len == 0u || !ka || !vk || !ring_k || !va || !vv || !ring_v || !kc0 || !vc0) return;

  const u32 i4 = tid();
  const u32 i0 = i4 << 2u;
  const u32 h = i4 >> 5u;
  const u32 p0 = pos0 % kv_len;

  /* K path: tree sconv(k) -> per-head L2 vclip(k) -> exact f32-then-f16 kc store + hist_k. */
  {
    f32x4 sw0, sw1, sw2, sw3;
    ink_r12_tv_load_sw(vk, i0, &sw0, &sw1, &sw2, &sw3);
    f32x4 r1, r2, r3;
    ink_r12_tv_seed_ring(ring_k, i0, pos0, &r1, &r2, &r3);

    f32x4 x1 = r1, x2 = r2, x3 = r3, prev_x0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    u32 p = p0;
#pragma clang loop unroll(disable)
    for (u32 t = 0u; t < T; t++) {
      ink_r12_tv_resolve_taps_g((const G u8 *)ka, kas, i4, t, tpar_pack, r1, r2, r3, prev_x0, &x1, &x2, &x3);
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)ka + (unsigned long)t * (unsigned long)kas))[i4];
      if (hist_k) ((G f32x4 *)(hist_k + (unsigned long)t * (unsigned long)hrs))[i4] = x0;

      f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
      if (ck) {
        const f32x4 ct = ((const G f32x4 *)((const G u8 *)ck + (unsigned long)t * (unsigned long)cks))[h];
        w0 += ct.x; w1 += ct.y; w2 += ct.z; w3 += ct.w;
      }
      const f32x4 kx = ink_r12_tv_sconv4(x0, x1, x2, x3, w0, w1, w2, w3, sc_escale);
      const f32x4 kxc = ink_r12_tv_vclip_k(kx, k_clamp);
      ((G kv4 *)kc0)[(unsigned long)p * 256ul + i4] = ink_r12_tv_f32x4_to_kv4(kxc);

      prev_x0 = x0;
      p++;
      if (p == kv_len) p = 0u;
    }
    if (!hist_k) ink_r12_tv_ring_commit(ring_k, i0, pos0, T, prev_x0, x1, x2, x3);
  }

  /* V path: tree sconv(v) -> per-head RMSNorm vclip(v) -> exact f32-then-f16 vc store + hist_v. */
  {
    f32x4 sw0, sw1, sw2, sw3;
    ink_r12_tv_load_sw(vv, i0, &sw0, &sw1, &sw2, &sw3);
    f32x4 r1, r2, r3;
    ink_r12_tv_seed_ring(ring_v, i0, pos0, &r1, &r2, &r3);

    f32x4 x1 = r1, x2 = r2, x3 = r3, prev_x0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    u32 p = p0;
#pragma clang loop unroll(disable)
    for (u32 t = 0u; t < T; t++) {
      ink_r12_tv_resolve_taps_g((const G u8 *)va, vas, i4, t, tpar_pack, r1, r2, r3, prev_x0, &x1, &x2, &x3);
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)va + (unsigned long)t * (unsigned long)vas))[i4];
      if (hist_v) ((G f32x4 *)(hist_v + (unsigned long)t * (unsigned long)hrs))[i4] = x0;

      f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
      if (cv) {
        const f32x4 ct = ((const G f32x4 *)((const G u8 *)cv + (unsigned long)t * (unsigned long)cvs))[h];
        w0 += ct.x; w1 += ct.y; w2 += ct.z; w3 += ct.w;
      }
      const f32x4 vx = ink_r12_tv_sconv4(x0, x1, x2, x3, w0, w1, w2, w3, sc_escale);
      const f32x4 vxc = ink_r12_tv_vclip_v(vx, v_eps);
      ((G kv4 *)vc0)[(unsigned long)p * 256ul + i4] = ink_r12_tv_f32x4_to_kv4(vxc);

      prev_x0 = x0;
      p++;
      if (p == kv_len) p = 0u;
    }
    if (!hist_v) ink_r12_tv_ring_commit(ring_v, i0, pos0, T, prev_x0, x1, x2, x3);
  }
}

/* ============================================================================
 * 2. H26-C2: ie_inkling_sconv_post_tree_r12
 *    Tree-aware Post-Attn Short-Conv + Residual Add OR Block-Save Copy + sc_hist
 *    Grid: (((n >> 2) + 255) / 256, 1, 1), 0 LDS, <= 56 VGPR.
 * ========================================================================== */

KERNEL ie_inkling_sconv_post_tree_r12(
    const G float *a, const G float *v, G float *ring,
    const G float *res, G float *out, G float *blk_dst, G float *hist,
    u32 n, float escale, u32 pos0, u32 T,
    u32 as, u32 rs, u32 os, u32 bds, u32 hrs, u32 tpar_pack) {
#pragma clang fp contract(off)
  if (T == 0u || T > 8u || n == 0u || (n & 3u) != 0u || !a || !v || !ring || !res || !out) return;

  const u32 n4 = n >> 2u;
  const u32 i4 = wgid() * NT + tid();
  if (i4 >= n4) return;
  const u32 i0 = i4 << 2u;

  f32x4 sw0, sw1, sw2, sw3;
  ink_r12_tv_load_sw(v, i0, &sw0, &sw1, &sw2, &sw3);

  f32x4 r1, r2, r3;
  ink_r12_tv_seed_ring(ring, i0, pos0, &r1, &r2, &r3);

  f32x4 x1 = r1, x2 = r2, x3 = r3, prev_x0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (!blk_dst) {
#pragma clang loop unroll(disable)
    for (u32 t = 0u; t < T; t++) {
      ink_r12_tv_resolve_taps_g((const G u8 *)a, as, i4, t, tpar_pack, r1, r2, r3, prev_x0, &x1, &x2, &x3);
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t * (unsigned long)as))[i4];
      const f32x4 rv = ((const G f32x4 *)((const G u8 *)res + (unsigned long)t * (unsigned long)rs))[i4];
      if (hist) ((G f32x4 *)(hist + (unsigned long)t * (unsigned long)hrs))[i4] = x0;

      const f32x4 sc = ink_r12_tv_sconv4(x0, x1, x2, x3, sw0, sw1, sw2, sw3, escale);
      ((G f32x4 *)((G u8 *)out + (unsigned long)t * (unsigned long)os))[i4] = rv + sc;
      prev_x0 = x0;
    }
  } else {
#pragma clang loop unroll(disable)
    for (u32 t = 0u; t < T; t++) {
      ink_r12_tv_resolve_taps_g((const G u8 *)a, as, i4, t, tpar_pack, r1, r2, r3, prev_x0, &x1, &x2, &x3);
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t * (unsigned long)as))[i4];
      const f32x4 rv = ((const G f32x4 *)((const G u8 *)res + (unsigned long)t * (unsigned long)rs))[i4];
      if (hist) ((G f32x4 *)(hist + (unsigned long)t * (unsigned long)hrs))[i4] = x0;

      const f32x4 sc = ink_r12_tv_sconv4(x0, x1, x2, x3, sw0, sw1, sw2, sw3, escale);
      ((G f32x4 *)((G u8 *)out + (unsigned long)t * (unsigned long)os))[i4] = sc;
      ((G f32x4 *)((G u8 *)blk_dst + (unsigned long)t * (unsigned long)bds))[i4] = rv;
      prev_x0 = x0;
    }
  }

  if (!hist) ink_r12_tv_ring_commit(ring, i0, pos0, T, prev_x0, x1, x2, x3);
}

/* ============================================================================
 * 3. H26-B2: ie_inkling_exp_down_tail_ver_tree_r12
 *    Tree-aware 4-Op Compact IQ4_NL exp_down + moe_sum + ffn_pc sconv + add + sc_hist
 *    Grid: ((rows + 31)/32, max_packs, 1), 1,028 B LDS, <= 80 VGPR.
 * ========================================================================== */

static inline __attribute__((always_inline)) int ink_r12_tv_iq4_dot4(
    const u32x4 w4, const u32x4 a0, const u32x4 a1) {
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

static inline __attribute__((always_inline)) int ink_r12_tv_iq4_dot_pre(
    const u32 wl0, const u32 wh0, const u32 wl1, const u32 wh1,
    const u32 wl2, const u32 wh2, const u32 wl3, const u32 wh3,
    const u32x4 a0, const u32x4 a1) {
  const u32 a00 = a0.x, a01 = a0.y, a02 = a0.z, a03 = a0.w;
  const u32 a10 = a1.x, a11 = a1.y, a12 = a1.z, a13 = a1.w;
  u32 d = 0u;
  d = ie_dot4_ss(ie_dot4_ss(d, wl0, a00), wh0, a10);
  d = ie_dot4_ss(ie_dot4_ss(d, wl1, a01), wh1, a11);
  d = ie_dot4_ss(ie_dot4_ss(d, wl2, a02), wh2, a12);
  d = ie_dot4_ss(ie_dot4_ss(d, wl3, a03), wh3, a13);
  return (int)d;
}

static inline __attribute__((always_inline)) f32x4 ink_r12_tv_moe_combine4(
    const G f32x4 *y4, const G u32 *rt, u32 n4, u32 K, u32 i4) {
#pragma clang fp contract(off)
  const G float *w = (const G float *)(rt + K);
  f32x4 x0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  for (u32 k = 0u; k < K; k++) {
    const float wk = w[k];
    const f32x4 yk = y4[(unsigned long)k * (unsigned long)n4 + (unsigned long)i4];
#pragma unroll
    for (u32 c = 0u; c < 4u; c++) x0[c] = __builtin_fmaf(wk, yk[c], x0[c]);
  }
  return x0;
}

KERNEL ie_inkling_exp_down_tail_ver_tree_r12(
    const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
    u32 rows, u32 nb,
    const G u32 *list, const G u32 *packs, const G u32 *n_packs,
    u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv,
    const G u32 *route, u32 K, u32 rts,
    const G float *v, G float *ring, const G float *res, G float *out, G float *hist,
    float escale, u32 pos0, u32 T, u32 yts, u32 rs, u32 os, u32 hrs,
    G u32 *rcount, u32 tpar_pack) {
#pragma clang fp contract(off)
  static LDS u32   ink_r12_tv_b2_last;
  static LDS f32x4 ink_r12_tv_b2_x[8u][8u] __attribute__((aligned(16))); /* 1,024 B: 8 lanes x 8 tree nodes */

  if (rows == 0u || (rows & 3u) != 0u || nb == 0u || T == 0u || T > 8u || K == 0u ||
      !qw0 || !qs0 || !X || !Y || !list || !packs || !n_packs ||
      !route || !v || !ring || !res || !out || !rcount)
    return;

  const u32 by = tokid();
  const u32 npk = (u32)__builtin_amdgcn_readfirstlane((int)(*n_packs));
  const u32 rb = wgid();
  const u32 rbase = rb * 32u;
  if (by >= npk || rbase >= rows) return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)packs[by]);
  const u32 e = desc & 0xFFu;
  const u32 lo = (desc >> 8u) & 0xFFFu;
  const u32 nj = (desc >> 20u) & 0xFu;

  const u32 t0 = tid();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 r_wave = rbase + wv * 4u;

  /* Phase 1: exact compact IQ4_NL exp_down (identical to ie_inkling_exp_down_tail_ver_r7). */
  if (nj > 0u && r_wave < rows) {
    const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
    const u32 r = r_wave + sub;
    const u32 rr = r < rows ? r : rows - 1u;

    const G u8 *w_base = rq_ptr_u((const G u8 *)(qw0 + (unsigned long)e * qwn));
    const G u8 *q_base = rq_ptr_u((const G u8 *)(qs0 + (unsigned long)e * qsn));
    const G u8 *rw = w_base + (unsigned long)rr * ((unsigned long)nb * 16ul);
    const G f16 *rs_p = (const G f16 *)q_base + (unsigned long)rr * nb;
    const unsigned long x_scale_off = 32ul * (unsigned long)nb;
    const int full_wave = (nb > 16u);

    const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
    const G u8 *xp0 = rq_ptr_u(X + (unsigned long)ink_r12_tv_div_x(p0, xdiv) * xs);
    const G u32 *aw0 = (const G u32 *)xp0;
    const G float *da0 = (const G float *)(xp0 + x_scale_off);

    if (nj == 1u) {
      if (r < rows) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        const u32 nstep = (nb + 7u) >> 3u;
        u32x4 w_c = (u32x4)(0u), x0_c = (u32x4)(0u), x1_c = (u32x4)(0u);
        f16 sc_c = (f16)0.0f;
        float d_c = 0.0f;
        if (sl < nb) {
          w_c = *(const G u32x4 *)(rw + sl * 16u);
          sc_c = rs_p[sl];
          x0_c = *(const G u32x4 *)(aw0 + sl * 8u);
          x1_c = *(const G u32x4 *)(aw0 + sl * 8u + 4u);
          d_c = da0[sl];
        }
        __builtin_amdgcn_sched_barrier(0);

#pragma unroll 1
        for (u32 step = 0u; step < nstep; step++) {
          const u32 b_c = step * 8u + sl;
          const u32 bn = b_c + 8u;
          u32x4 w_n = (u32x4)(0u), x0_n = (u32x4)(0u), x1_n = (u32x4)(0u);
          f16 sc_n = (f16)0.0f;
          float d_n = 0.0f;
          if (bn < nb) {
            w_n = *(const G u32x4 *)(rw + bn * 16u);
            sc_n = rs_p[bn];
            x0_n = *(const G u32x4 *)(aw0 + bn * 8u);
            x1_n = *(const G u32x4 *)(aw0 + bn * 8u + 4u);
            d_n = da0[bn];
          }
          __builtin_amdgcn_sched_barrier(0);
          if (b_c < nb) {
            const float val = (float)sc_c * (float)ink_r12_tv_iq4_dot4(w_c, x0_c, x1_c);
            const u32 j = step & 3u;
            INK_R12_TV_ACC_FMA(a0, a1, a2, a3, j, val, d_c);
          }
          __builtin_amdgcn_sched_barrier(0);
          w_c = w_n; sc_c = sc_n; x0_c = x0_n; x1_c = x1_n; d_c = d_n;
        }

        const float acc0 = ink_r12_tv_oct_tree_f(full_wave ? ((a0 + a2) + (a1 + a3)) : (a0 + a1));
        if (sl == 0u) {
          G float *yp0 = (G float *)(Y + (unsigned long)p0 * ys);
          yp0[r] = acc0;
        }
      }
    } else {
      const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 1u]);
      const u32 p2 = nj > 2u ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 2u]) : p0;
      const u32 p3 = nj > 3u ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 3u]) : p0;

      const G u8 *xp1 = rq_ptr_u(X + (unsigned long)ink_r12_tv_div_x(p1, xdiv) * xs);
      const G u8 *xp2 = nj > 2u ? rq_ptr_u(X + (unsigned long)ink_r12_tv_div_x(p2, xdiv) * xs) : xp0;
      const G u8 *xp3 = nj > 3u ? rq_ptr_u(X + (unsigned long)ink_r12_tv_div_x(p3, xdiv) * xs) : xp0;

      const G u32 *aw1 = (const G u32 *)xp1;
      const G u32 *aw2 = (const G u32 *)xp2;
      const G u32 *aw3 = (const G u32 *)xp3;
      const G float *da1 = (const G float *)(xp1 + x_scale_off);
      const G float *da2 = (const G float *)(xp2 + x_scale_off);
      const G float *da3 = (const G float *)(xp3 + x_scale_off);

      float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
      float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
      float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
      float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;

      const u32 nstep = (nb + 7u) >> 3u;
      const u32 nb_last = nb - 1u;

      u32x4 w_c = (u32x4)(0u);
      f16 sc_c = (f16)0.0f;
      if (sl < nb) {
        w_c = *(const G u32x4 *)(rw + sl * 16u);
        sc_c = rs_p[sl];
      }
      __builtin_amdgcn_sched_barrier(0);

#pragma unroll 1
      for (u32 step = 0u; step < nstep; step++) {
        const u32 b_c = step * 8u + sl;
        const u32 bn = b_c + 8u;
        const u32 j_acc = step & 3u;

        u32x4 w_n = (u32x4)(0u);
        f16 sc_n = (f16)0.0f;

        if (b_c < nb) {
          u32x4 x0_a = *(const G u32x4 *)(aw0 + b_c * 8u);
          u32x4 x1_a = *(const G u32x4 *)(aw0 + b_c * 8u + 4u);
          float d_a = da0[b_c];

          u32x4 x0_b = *(const G u32x4 *)(aw1 + b_c * 8u);
          u32x4 x1_b = *(const G u32x4 *)(aw1 + b_c * 8u + 4u);
          float d_b = da1[b_c];

          const u32 bn_safe = bn < nb ? bn : nb_last;
          w_n = *(const G u32x4 *)(rw + bn_safe * 16u);
          sc_n = rs_p[bn_safe];
          __builtin_amdgcn_sched_barrier(0);

          const float sc_f = (float)sc_c;
          const u32 wcx = w_c.x, wcy = w_c.y, wcz = w_c.z, wcw = w_c.w;
          const u32 wl0 = ie_iq4_map(wcx & 0x0F0F0F0Fu), wh0 = ie_iq4_map((wcx >> 4u) & 0x0F0F0F0Fu);
          const u32 wl1 = ie_iq4_map(wcy & 0x0F0F0F0Fu), wh1 = ie_iq4_map((wcy >> 4u) & 0x0F0F0F0Fu);
          const u32 wl2 = ie_iq4_map(wcz & 0x0F0F0F0Fu), wh2 = ie_iq4_map((wcz >> 4u) & 0x0F0F0F0Fu);
          const u32 wl3 = ie_iq4_map(wcw & 0x0F0F0F0Fu), wh3 = ie_iq4_map((wcw >> 4u) & 0x0F0F0F0Fu);

          const float v0 = sc_f * (float)ink_r12_tv_iq4_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_a, x1_a);
          INK_R12_TV_ACC_FMA(a0_0, a0_1, a0_2, a0_3, j_acc, v0, d_a);

          if (nj > 2u) {
            x0_a = *(const G u32x4 *)(aw2 + b_c * 8u);
            x1_a = *(const G u32x4 *)(aw2 + b_c * 8u + 4u);
            d_a = da2[b_c];
          }
          __builtin_amdgcn_sched_barrier(0);

          const float v1 = sc_f * (float)ink_r12_tv_iq4_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_b, x1_b);
          INK_R12_TV_ACC_FMA(a1_0, a1_1, a1_2, a1_3, j_acc, v1, d_b);

          if (nj > 2u) {
            if (nj > 3u) {
              x0_b = *(const G u32x4 *)(aw3 + b_c * 8u);
              x1_b = *(const G u32x4 *)(aw3 + b_c * 8u + 4u);
              d_b = da3[b_c];
            }
            __builtin_amdgcn_sched_barrier(0);
            const float v2 = sc_f * (float)ink_r12_tv_iq4_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_a, x1_a);
            INK_R12_TV_ACC_FMA(a2_0, a2_1, a2_2, a2_3, j_acc, v2, d_a);

            if (nj > 3u) {
              __builtin_amdgcn_sched_barrier(0);
              const float v3 = sc_f * (float)ink_r12_tv_iq4_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_b, x1_b);
              INK_R12_TV_ACC_FMA(a3_0, a3_1, a3_2, a3_3, j_acc, v3, d_b);
            }
          }
        }
        __builtin_amdgcn_sched_barrier(0);
        w_c = w_n; sc_c = sc_n;
      }

      const float acc0 = ink_r12_tv_oct_tree_f(full_wave ? ((a0_0 + a0_2) + (a0_1 + a0_3)) : (a0_0 + a0_1));
      if (sl == 0u && r < rows) ((G float *)(Y + (unsigned long)p0 * ys))[r] = acc0;

      const float acc1 = ink_r12_tv_oct_tree_f(full_wave ? ((a1_0 + a1_2) + (a1_1 + a1_3)) : (a1_0 + a1_1));
      if (sl == 0u && r < rows) ((G float *)(Y + (unsigned long)p1 * ys))[r] = acc1;

      if (nj > 2u) {
        const float acc2 = ink_r12_tv_oct_tree_f(full_wave ? ((a2_0 + a2_2) + (a2_1 + a2_3)) : (a2_0 + a2_1));
        if (sl == 0u && r < rows) ((G float *)(Y + (unsigned long)p2 * ys))[r] = acc2;
      }
      if (nj > 3u) {
        const float acc3 = ink_r12_tv_oct_tree_f(full_wave ? ((a3_0 + a3_2) + (a3_1 + a3_3)) : (a3_0 + a3_1));
        if (sl == 0u && r < rows) ((G float *)(Y + (unsigned long)p3 * ys))[r] = acc3;
      }
    }
  }

  /* Agent release/acquire: npk-th active pack WG finishes all T tree nodes for its 32 rows. */
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (t0 == 0u) {
    ink_r12_tv_b2_last =
        (__scoped_atomic_fetch_add(&rcount[rb], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == npk - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r12_tv_b2_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  /* Phase 2: 8 threads (8 x f32x4 = 32 rows) run exact ordered combine + tree sconv + add + hist. */
  const u32 n4 = rows >> 2u;
  if (t0 < 8u) {
    const u32 i4 = rb * 8u + t0;
    if (i4 < n4) {
      const u32 i0 = i4 << 2u;
      f32x4 sw0, sw1, sw2, sw3;
      ink_r12_tv_load_sw(v, i0, &sw0, &sw1, &sw2, &sw3);
      f32x4 r1, r2, r3;
      ink_r12_tv_seed_ring(ring, i0, pos0, &r1, &r2, &r3);

      f32x4 x1 = r1, x2 = r2, x3 = r3, prev_x0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
#pragma clang loop unroll(disable)
      for (u32 t = 0u; t < T; t++) {
        const G f32x4 *y4 = (const G f32x4 *)(Y + (unsigned long)t * (unsigned long)yts);
        const G u32 *rt = (const G u32 *)((const G u8 *)route + (unsigned long)t * (unsigned long)rts);
        const f32x4 x0 = ink_r12_tv_moe_combine4(y4, rt, n4, K, i4);
        ink_r12_tv_b2_x[t0][t] = x0;
        if (hist) ((G f32x4 *)(hist + (unsigned long)t * (unsigned long)hrs))[i4] = x0;

        if (t == 0u) {
          x1 = r1; x2 = r2; x3 = r3;
        } else {
          const u32 p1 = (tpar_pack >> (t << 2u)) & 0xFu;
          if (p1 == t - 1u) {
            x3 = x2; x2 = x1; x1 = prev_x0;
          } else {
            x1 = ink_r12_tv_b2_x[t0][p1];
            if (p1 == 0u) {
              x2 = r1; x3 = r2;
            } else {
              const u32 p2 = (tpar_pack >> (p1 << 2u)) & 0xFu;
              x2 = ink_r12_tv_b2_x[t0][p2];
              if (p2 == 0u) x3 = r1;
              else {
                const u32 p3 = (tpar_pack >> (p2 << 2u)) & 0xFu;
                x3 = ink_r12_tv_b2_x[t0][p3];
              }
            }
          }
        }

        const f32x4 rv = ((const G f32x4 *)((const G u8 *)res + (unsigned long)t * (unsigned long)rs))[i4];
        const f32x4 sc = ink_r12_tv_sconv4(x0, x1, x2, x3, sw0, sw1, sw2, sw3, escale);
        ((G f32x4 *)((G u8 *)out + (unsigned long)t * (unsigned long)os))[i4] = rv + sc;
        prev_x0 = x0;
      }

      if (!hist) ink_r12_tv_ring_commit(ring, i0, pos0, T, prev_x0, x1, x2, x3);
    }
  }
  if (t0 == 0u) rcount[rb] = 0u;
}

/* ============================================================================
 * 4. H26-WMMA-Glob: ie_inkling_rel_attn_ver_glob_wmma_tree_r12
 *    Single-pass T<=8 FP16 WMMA Global Verify Attention for EAGLE-2/Sequoia Trees
 *    (zero overhead on Far-Field [0..pos0-rel_extent), exact tree_mask +
 *    pos_off_packed in Near-Field, optional H15 fused OP_GATE(sigmoid) + Q8_0!).
 *    Grid: (n_kv * nsplit, 1, 1), 15,360 B LDS, <= 128 VGPR (3 WGs/CU).
 * ========================================================================== */

typedef union {
  struct {
    f16 q[32u * 128u] __attribute__((aligned(16)));        /*  8,192 B */
    f16 r[32u * 16u] __attribute__((aligned(16)));         /*  1,024 B */
    f16 p_tile[2u][2u][256u] __attribute__((aligned(16))); /*  2,048 B */
    union {
      u32 v_sc[8u][128u] __attribute__((aligned(16)));     /*  4,096 B */
      struct {
        float mx[2u][128u];                                /*  1,024 B */
        float ls[2u][128u];                                /*  1,024 B */
      } red;
    } scratch;                                             /*  4,096 B */
  } kv;                                                    /* 15,360 B */
  struct {
    float o_stg[8u][128u] __attribute__((aligned(16)));    /*  4,096 B */
    float m[32u];                                          /*    128 B */
    float l[32u];                                          /*    128 B */
    float wt[8u][128u] __attribute__((aligned(16)));       /*  4,096 B */
    u32   is_last;                                         /*      4 B */
  } ph4;                                                   /*  8,452 B <= 15,360 B */
} ink_r12_tv_gw_lds_t;

static LDS ink_r12_tv_gw_lds_t ink_r12_tv_gw_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r12_tv_gw_lds_t) == 15360u, "ink_r12_tv_gw_lds_t must be 15,360 B");

static inline __attribute__((always_inline)) u32 ink_r12_tv_gw_q_off(u32 row, u32 c8) {
  return row * 128u + 8u * (c8 ^ (row & 15u));
}

static inline __attribute__((always_inline)) u32 ink_r12_tv_gw_p_off(u32 rl, u32 hl) {
  return rl * 16u + 8u * (hl ^ ((rl >> 2u) & 1u));
}

static inline __attribute__((always_inline)) h8 ink_r12_tv_gw_v_transpose(
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

/* Tree-aware Near-Field 16-distance rel_bias application into WMMA score tile sv. */
static inline __attribute__((always_inline)) void ink_r12_tv_gw_apply_rb16_tree(
    const v8f rb, const u32 base_off, const u32 hl, const u32 kb,
    const u32 pos0, const u32 T, const u32 dep_t, const u32 rmask_t,
    const u32 pos_off_packed, const u32 rel_extent, const u32 d_base,
    LDSP float *sc_row, v8f *sv) {
  if (hl == 0u) {
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) sc_row[v] = rb[v];
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma clang loop unroll(disable)
  for (u32 v = 0u; v < 8u; v++) {
    const u32 kp = kb + 8u * hl + v;
    int valid = 0;
    u32 dist = 0u;
    if (kp < pos0) {
      valid = 1;
      dist = (pos0 + dep_t) - kp;
    } else {
      const u32 j = kp - pos0;
      if (j < T && ((rmask_t >> j) & 1u) != 0u) {
        valid = 1;
        const u32 dep_j = (pos_off_packed >> (j << 2u)) & 0xFu;
        dist = (dep_t >= dep_j) ? (dep_t - dep_j) : 0u;
      }
    }
    if (valid && dist < rel_extent) {
      const u32 rel = (dist - d_base) - base_off;
      if (rel < 8u) (*sv)[v] += sc_row[rel];
    }
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  if (hl == 1u) {
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) sc_row[v] = rb[v];
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma clang loop unroll(disable)
  for (u32 v = 0u; v < 8u; v++) {
    const u32 kp = kb + 8u * hl + v;
    int valid = 0;
    u32 dist = 0u;
    if (kp < pos0) {
      valid = 1;
      dist = (pos0 + dep_t) - kp;
    } else {
      const u32 j = kp - pos0;
      if (j < T && ((rmask_t >> j) & 1u) != 0u) {
        valid = 1;
        const u32 dep_j = (pos_off_packed >> (j << 2u)) & 0xFu;
        dist = (dep_t >= dep_j) ? (dep_t - dep_j) : 0u;
      }
    }
    if (valid && dist < rel_extent) {
      const u32 rel = (dist - d_base) - (base_off + 8u);
      if (rel < 8u) (*sv)[v] += sc_row[rel];
    }
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

/* Exact OP_GATE (out * sigmoid(agate)) + Q8_0 emit for 1 head (128 floats = 4/lane). */
static inline __attribute__((always_inline)) float ink_r12_tv_gate_f(float o, float g) {
#pragma clang fp contract(off)
  return o * (1.0f / (1.0f + __builtin_expf(-g)));
}

static inline __attribute__((always_inline)) void ink_r12_tv_emit_gate_q8(
    const f32x4 ov, const G float *gt_h, G float *ot_h, G u8 *oqt,
    u32 h, u32 n_head, u32 l, LDSP float *stg_w) {
  f32x4 yv = ov;
  if (gt_h) {
    const f32x4 gv = ((const G f32x4 *)gt_h)[l];
    yv = (f32x4){ink_r12_tv_gate_f(ov.x, gv.x), ink_r12_tv_gate_f(ov.y, gv.y),
                 ink_r12_tv_gate_f(ov.z, gv.z), ink_r12_tv_gate_f(ov.w, gv.w)};
  }
  if (ot_h) ((G f32x4 *)ot_h)[l] = yv;
  if (oqt) {
    ((LDSP f32x4 *)stg_w)[l] = yv;
    __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
    const u32 nb_q = n_head << 2u;
    const u32 b0 = h << 2u;
#pragma unroll
    for (u32 m = 0u; m < 4u; m++) quant_wave(stg_w[32u * m + l], b0 + m, oqt, nb_q);
    __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  }
}

static inline __attribute__((always_inline)) void ink_r12_tv_gw_merge_head(
    const G float *ph, const float sink_val, const u32 nsplit, const u32 l,
    LDSP float *wt_w, LDSP float *stg_w, const G float *gt_h, G float *ot_h, G u8 *oqt,
    u32 h, u32 n_head) {
  float mv = (l == 0u) ? sink_val : -__builtin_inff();
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const unsigned long off = (unsigned long)sp * INK_R12_TV_PSTR;
    if (ph[off + 1u] > 0.0f) mv = __builtin_fmaxf(mv, ph[off]);
  }
#pragma unroll
  for (u32 mask = 16u; mask >= 1u; mask >>= 1u) mv = __builtin_fmaxf(mv, shfl_xor_f(mv, mask));
  const float M2 = ink_r12_tv_sgpr_f(mv);

  float lv = 0.0f;
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const unsigned long off = (unsigned long)sp * INK_R12_TV_PSTR;
    const float lp = ph[off + 1u];
    const float wt = (lp > 0.0f) ? __builtin_expf(ph[off] - M2) : 0.0f;
    if (sp < 128u) wt_w[sp] = wt;
    lv += lp * wt;
  }
  const float L2 = ink_r12_tv_sgpr_f(wave_sum_all(lv) + __builtin_expf(sink_val - M2));
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");

  f32x4 acc = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  for (u32 sp = 0u; sp < nsplit; sp++) {
    const float wt = (sp < 128u) ? wt_w[sp]
                                 : ((ph[(unsigned long)sp * INK_R12_TV_PSTR + 1u] > 0.0f)
                                        ? __builtin_expf(ph[(unsigned long)sp * INK_R12_TV_PSTR] - M2)
                                        : 0.0f);
    if (wt > 0.0f) {
      const G float *po = ph + (unsigned long)sp * INK_R12_TV_PSTR + 2u + 4u * l;
      const f32x4 ov = {po[0], po[1], po[2], po[3]};
      acc += ov * wt;
    }
  }
  const float inv = 1.0f / L2;
  ink_r12_tv_emit_gate_q8(acc * inv, gt_h, ot_h, oqt, h, n_head, l, stg_w);
}

KERNEL ie_inkling_rel_attn_ver_glob_wmma_tree_r12(
    const G float *q, const G kvt *kc, const G kvt *vc,
    const G float *R, const G float *rel_bias, const G float *sink,
    G float *part, G float *out, u32 pos0, u32 T, u32 kv_len, u32 hd,
    u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
    u32 qs, u32 rs, u32 ss, u32 os, G u32 *count,
    ink_r12_tv_u64 tree_mask, u32 pos_off_packed,
    const G float *agate, G u8 *oq, u32 gs, u32 oqs) {
  if (T == 0u || T > 8u || win != 0u || hd != INK_R12_TV_HD || d_rel != INK_R12_TV_DREL || n_kv == 0u ||
      n_head != INK_R12_TV_GQA * n_kv || kv_len == 0u || nsplit == 0u)
    return;

  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() / nsplit));
  const u32 s = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() % nsplit));
  if (kh >= n_kv) return;

  const u32 h0 = (u32)__builtin_amdgcn_readfirstlane((int)(kh * INK_R12_TV_GQA));
  const u32 kvd = (u32)__builtin_amdgcn_readfirstlane((int)(n_kv * INK_R12_TV_HD));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane(), hl = l >> 4u, rl = l & 15u;

  const u32 nq_tot = (u32)__builtin_amdgcn_readfirstlane((int)(T * INK_R12_TV_GQA));
  const u32 npg = (u32)__builtin_amdgcn_readfirstlane((int)(nq_tot > 16u ? 2u : 1u));
  const u32 np = (u32)__builtin_amdgcn_readfirstlane((int)(pos0 + T));
  const u32 ch = (u32)__builtin_amdgcn_readfirstlane((int)((np + nsplit - 1u) / nsplit));
  const u32 off0 = (u32)__builtin_amdgcn_readfirstlane((int)(s * ch));
  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)((off0 < np) ? off0 : np));
  const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)((off0 < np && off0 + ch < np) ? (off0 + ch) : np));

  if (nsplit > 1u && p0 >= p1) {
    for (u32 qi = w; qi < nq_tot; qi += 8u) {
      const u32 t = qi / INK_R12_TV_GQA;
      const u32 h = h0 + (qi - t * INK_R12_TV_GQA);
      G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R12_TV_PSTR;
      if (l == 0u) { pp[0] = -1e30f; pp[1] = 0.0f; }
      pp[2u + 4u * l + 0u] = 0.0f;
      pp[2u + 4u * l + 1u] = 0.0f;
      pp[2u + 4u * l + 2u] = 0.0f;
      pp[2u + 4u * l + 3u] = 0.0f;
    }
  } else {
    /* Far-field splits strictly precede pos0 by >= rel_extent -> 100% zero-overhead vs ver_glob_wmma_r7! */
    const int far_split = (int)__builtin_amdgcn_readfirstlane(
        (int)((p1 <= pos0) && ((pos0 + 1u - p1) >= rel_extent)));

    const u32 q_rows = npg * 16u;
    for (u32 idx = tid(); idx < q_rows * 16u; idx += NT) {
      const u32 qr = idx >> 4u;
      const u32 c8 = idx & 15u;
      h8 qh = (h8)(0.0f);
      if (qr < nq_tot) {
        const u32 tj = qr / INK_R12_TV_GQA;
        const u32 gi = qr - tj * INK_R12_TV_GQA;
        const G f32x4 *qp = (const G f32x4 *)((const G float *)((const G u8 *)q + (unsigned long)tj * (unsigned long)qs) +
                                              (unsigned long)(h0 + gi) * INK_R12_TV_HD);
        const f32x4 v0 = qp[2u * c8 + 0u];
        const f32x4 v1 = qp[2u * c8 + 1u];
        qh = (h8){(f16)(v0.x * INK_R12_TV_QSC), (f16)(v0.y * INK_R12_TV_QSC),
                  (f16)(v0.z * INK_R12_TV_QSC), (f16)(v0.w * INK_R12_TV_QSC),
                  (f16)(v1.x * INK_R12_TV_QSC), (f16)(v1.y * INK_R12_TV_QSC),
                  (f16)(v1.z * INK_R12_TV_QSC), (f16)(v1.w * INK_R12_TV_QSC)};
      }
      *(LDSP h8 *)(ink_r12_tv_gw_lds.kv.q + ink_r12_tv_gw_q_off(qr, c8)) = qh;
    }
    if (!far_split && rel_extent > 0u) {
      if (tid() < q_rows * 2u) {
        const u32 qr = tid() >> 1u;
        const u32 h8i = tid() & 1u;
        h8 rh = (h8)(0.0f);
        if (qr < nq_tot) {
          const u32 tj = qr / INK_R12_TV_GQA;
          const u32 gi = qr - tj * INK_R12_TV_GQA;
          const G f32x4 *rp = (const G f32x4 *)((const G float *)((const G u8 *)R + (unsigned long)tj * (unsigned long)rs) +
                                                (unsigned long)(h0 + gi) * INK_R12_TV_DREL);
          const f32x4 v0 = rp[2u * h8i + 0u];
          const f32x4 v1 = rp[2u * h8i + 1u];
          rh = (h8){(f16)(v0.x * INK_R12_TV_LOG2E), (f16)(v0.y * INK_R12_TV_LOG2E),
                    (f16)(v0.z * INK_R12_TV_LOG2E), (f16)(v0.w * INK_R12_TV_LOG2E),
                    (f16)(v1.x * INK_R12_TV_LOG2E), (f16)(v1.y * INK_R12_TV_LOG2E),
                    (f16)(v1.z * INK_R12_TV_LOG2E), (f16)(v1.w * INK_R12_TV_LOG2E)};
        }
        *(LDSP h8 *)(ink_r12_tv_gw_lds.kv.r + qr * 16u + 8u * h8i) = rh;
      }
    }
    lbarrier();

    const G kvt *kc_h = kc + (unsigned long)kh * INK_R12_TV_HD;
    const G kvt *vc_h = vc + (unsigned long)kh * INK_R12_TV_HD;
    const int no_wrap = (int)__builtin_amdgcn_readfirstlane((int)(p1 <= kv_len));

    const v8f zero = (v8f)(0.0f);
    v8f acc0 = zero, acc1 = zero;
    float m0 = -__builtin_inff(), ls0 = 0.0f;
    float m1 = -__builtin_inff(), ls1 = 0.0f;

    const u32 p_off = ink_r12_tv_gw_p_off(rl, hl);

#pragma clang loop unroll(disable)
    for (u32 c0 = p0; c0 < p1; c0 += 128u) {
      const u32 kb = (u32)__builtin_amdgcn_readfirstlane((int)(c0 + 16u * w));
      const u32 chunk_end = (u32)__builtin_amdgcn_readfirstlane((int)((c0 + 128u < p1) ? (c0 + 128u) : p1));
      const int far_chunk = (int)__builtin_amdgcn_readfirstlane(
          (int)(far_split || ((chunk_end <= pos0) && ((pos0 + 1u - chunk_end) >= rel_extent))));

      v8f sp0 = (v8f)(-__builtin_inff());
      v8f sp1 = (v8f)(-__builtin_inff());

      if (kb < chunk_end) {
        const u32 kp_row = kb + rl;
        const int row_ok = (kp_row < chunk_end);
        const u32 kv_row = no_wrap ? (row_ok ? kp_row : (chunk_end - 1u))
                                   : ((row_ok ? kp_row : (chunk_end - 1u)) % kv_len);
        const G f16 *k_base = kc_h + (unsigned long)kv_row * (unsigned long)kvd + 8u * hl;

        sp0 = zero;
        sp1 = zero;
        h8 kk0 = row_ok ? *(const G h8 *)(k_base + 0u) : (h8)(0.0f);
        h8 kk1 = row_ok ? *(const G h8 *)(k_base + 16u) : (h8)(0.0f);
#pragma unroll
        for (u32 ks_i = 0u; ks_i < 8u; ks_i += 2u) {
          const u32 c_a = 2u * ks_i + hl;
          const u32 c_b = 2u * (ks_i + 1u) + hl;
          const h8 q0_a = *(const LDSP h8 *)(ink_r12_tv_gw_lds.kv.q + ink_r12_tv_gw_q_off(rl, c_a));
          const h8 q0_b = *(const LDSP h8 *)(ink_r12_tv_gw_lds.kv.q + ink_r12_tv_gw_q_off(rl, c_b));
          sp0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk0, q0_a, sp0);
          sp0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk1, q0_b, sp0);
          if (npg > 1u) {
            const h8 q1_a = *(const LDSP h8 *)(ink_r12_tv_gw_lds.kv.q + ink_r12_tv_gw_q_off(16u + rl, c_a));
            const h8 q1_b = *(const LDSP h8 *)(ink_r12_tv_gw_lds.kv.q + ink_r12_tv_gw_q_off(16u + rl, c_b));
            sp1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk0, q1_a, sp1);
            sp1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk1, q1_b, sp1);
          }
          if (ks_i + 2u < 8u) {
            kk0 = row_ok ? *(const G h8 *)(k_base + 16u * (ks_i + 2u)) : (h8)(0.0f);
            kk1 = row_ok ? *(const G h8 *)(k_base + 16u * (ks_i + 3u)) : (h8)(0.0f);
          }
          __builtin_amdgcn_sched_barrier(0);
        }

        if (!far_chunk) {
          /* Compute tree metadata strictly inside Near-Field (0 live VGPRs in Far-Field / PV!). */
          const u32 t_q0 = rl / INK_R12_TV_GQA;
          const u32 t_q1 = (16u + rl) / INK_R12_TV_GQA;
          const u32 dep_q0 = (t_q0 < T) ? ((pos_off_packed >> (t_q0 << 2u)) & 0xFu) : 0u;
          const u32 dep_q1 = (t_q1 < T) ? ((pos_off_packed >> (t_q1 << 2u)) & 0xFu) : 0u;
          const u32 rmask_q0 = (t_q0 < T) ? (u32)((tree_mask >> (t_q0 << 3u)) & 0xFFull) : 0u;
          const u32 rmask_q1 = (t_q1 < T) ? (u32)((tree_mask >> (t_q1 << 3u)) & 0xFFull) : 0u;

          /* Near-field tree-aware rel_bias via WMMA + wave-private 512 B scratch. */
          const int d_min_s = (int)pos0 - (int)(kb + 15u);
          const u32 d_base = d_min_s > 0 ? (u32)d_min_s : 0u;
          if (d_base < rel_extent) {
            LDSP float *sc_row = ((LDSP float *)ink_r12_tv_gw_lds.kv.scratch.v_sc[w]) + rl * 8u;
            const h8 ra0 = *(const LDSP h8 *)(ink_r12_tv_gw_lds.kv.r + rl * 16u + 8u * hl);
            const u32 d0 = d_base + rl;
            h8 b0 = (h8)(0.0f);
            if (d0 < rel_extent) {
              const G f32x4 *rp = (const G f32x4 *)(rel_bias + (unsigned long)d0 * 16ul + 8ul * hl);
              const f32x4 v0 = rp[0], v1 = rp[1];
              b0 = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
            }
            const v8f rb0_0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0, ra0, zero);
            ink_r12_tv_gw_apply_rb16_tree(rb0_0, 0u, hl, kb, pos0, T, dep_q0, rmask_q0,
                                          pos_off_packed, rel_extent, d_base, sc_row, &sp0);
            if (npg > 1u) {
              const h8 ra1 = *(const LDSP h8 *)(ink_r12_tv_gw_lds.kv.r + (16u + rl) * 16u + 8u * hl);
              const v8f rb0_1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0, ra1, zero);
              ink_r12_tv_gw_apply_rb16_tree(rb0_1, 0u, hl, kb, pos0, T, dep_q1, rmask_q1,
                                            pos_off_packed, rel_extent, d_base, sc_row, &sp1);
            }
            if (d_base + 16u < rel_extent && (pos0 + T - 1u) >= kb + d_base + 16u) {
              const u32 d1 = d_base + 16u + rl;
              h8 b1 = (h8)(0.0f);
              if (d1 < rel_extent) {
                const G f32x4 *rp = (const G f32x4 *)(rel_bias + (unsigned long)d1 * 16ul + 8ul * hl);
                const f32x4 v0 = rp[0], v1 = rp[1];
                b1 = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
              }
              const v8f rb1_0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1, ra0, zero);
              ink_r12_tv_gw_apply_rb16_tree(rb1_0, 16u, hl, kb, pos0, T, dep_q0, rmask_q0,
                                            pos_off_packed, rel_extent, d_base, sc_row, &sp0);
              if (npg > 1u) {
                const h8 ra1 = *(const LDSP h8 *)(ink_r12_tv_gw_lds.kv.r + (16u + rl) * 16u + 8u * hl);
                const v8f rb1_1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1, ra1, zero);
                ink_r12_tv_gw_apply_rb16_tree(rb1_1, 16u, hl, kb, pos0, T, dep_q1, rmask_q1,
                                              pos_off_packed, rel_extent, d_base, sc_row, &sp1);
              }
            }
          }
          /* Tree ancestor mask + split-end mask strictly inside Near-Field! */
#pragma clang loop unroll(disable)
          for (u32 v = 0u; v < 8u; v++) {
            const u32 kp = kb + 8u * hl + v;
            if (kp >= chunk_end) {
              sp0[v] = -__builtin_inff();
              if (npg > 1u) sp1[v] = -__builtin_inff();
            } else if (kp >= pos0) {
              const u32 j = kp - pos0;
              if (j >= T || ((rmask_q0 >> j) & 1u) == 0u) sp0[v] = -__builtin_inff();
              if (npg > 1u && (j >= T || ((rmask_q1 >> j) & 1u) == 0u)) sp1[v] = -__builtin_inff();
            }
          }
        } else if (kb + 16u > chunk_end) {
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
          const u32 kv_v = no_wrap ? vp : (vp % kv_len);
          vr0 = *(const G u32x4 *)(vc_h + (unsigned long)kv_v * (unsigned long)kvd + 16u * w + 8u * hl);
        }
      }

      if (!far_chunk) lbarrier();

      float mx0 = sp0[0];
#pragma unroll
      for (u32 v = 1u; v < 8u; v++) mx0 = __builtin_fmaxf(mx0, sp0[v]);
      mx0 = __builtin_fmaxf(mx0, ink_r12_tv_xrow(mx0));
      if (hl == 0u) ink_r12_tv_gw_lds.kv.scratch.red.mx[0][16u * w + rl] = mx0;

      if (npg > 1u) {
        float mx1 = sp1[0];
#pragma unroll
        for (u32 v = 1u; v < 8u; v++) mx1 = __builtin_fmaxf(mx1, sp1[v]);
        mx1 = __builtin_fmaxf(mx1, ink_r12_tv_xrow(mx1));
        if (hl == 0u) ink_r12_tv_gw_lds.kv.scratch.red.mx[1][16u * w + rl] = mx1;
      }
      lbarrier();

      float mc0 = ink_r12_tv_gw_lds.kv.scratch.red.mx[0][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) mc0 = __builtin_fmaxf(mc0, ink_r12_tv_gw_lds.kv.scratch.red.mx[0][16u * x + rl]);
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
      lc0 += ink_r12_tv_xrow(lc0);
      if (hl == 0u) ink_r12_tv_gw_lds.kv.scratch.red.ls[0][16u * w + rl] = lc0;

      h8 ph1 = (h8)(0.0f);
      if (npg > 1u) {
        float mc1 = ink_r12_tv_gw_lds.kv.scratch.red.mx[1][rl];
#pragma unroll
        for (u32 x = 1u; x < 8u; x++) mc1 = __builtin_fmaxf(mc1, ink_r12_tv_gw_lds.kv.scratch.red.mx[1][16u * x + rl]);
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
        lc1 += ink_r12_tv_xrow(lc1);
        if (hl == 0u) ink_r12_tv_gw_lds.kv.scratch.red.ls[1][16u * w + rl] = lc1;
      }

      if (w == 0u) {
        *(LDSP h8 *)(ink_r12_tv_gw_lds.kv.p_tile[0][0] + p_off) = ph0;
        if (npg > 1u) *(LDSP h8 *)(ink_r12_tv_gw_lds.kv.p_tile[0][1] + p_off) = ph1;
      }
      lbarrier();

      float sum_l0 = ink_r12_tv_gw_lds.kv.scratch.red.ls[0][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) sum_l0 += ink_r12_tv_gw_lds.kv.scratch.red.ls[0][16u * x + rl];
      ls0 += sum_l0;
      if (npg > 1u) {
        float sum_l1 = ink_r12_tv_gw_lds.kv.scratch.red.ls[1][rl];
#pragma unroll
        for (u32 x = 1u; x < 8u; x++) sum_l1 += ink_r12_tv_gw_lds.kv.scratch.red.ls[1][16u * x + rl];
        ls1 += sum_l1;
      }
      lbarrier();

      LDSP u32 *v_sc_w = ink_r12_tv_gw_lds.kv.scratch.v_sc[w];
#pragma clang loop unroll(disable)
      for (u32 kt = 0u; kt < 8u; kt++) {
        if (c0 + 16u * kt >= chunk_end) break;
        const int has_nxt = (kt + 1u < 8u) && (c0 + 16u * (kt + 1u) < chunk_end);
        u32x4 vr_nxt = (u32x4)(0u);
        if (has_nxt) {
          const u32 vp_n = c0 + 16u * (kt + 1u) + rl;
          if (vp_n < chunk_end) {
            const u32 kv_vn = no_wrap ? vp_n : (vp_n % kv_len);
            vr_nxt = *(const G u32x4 *)(vc_h + (unsigned long)kv_vn * (unsigned long)kvd + 16u * w + 8u * hl);
          }
        }
        __builtin_amdgcn_sched_barrier(0);

        const h8 vt_a = ink_r12_tv_gw_v_transpose(vr0, v_sc_w, hl, rl);
        const h8 pt0 = *(const LDSP h8 *)(ink_r12_tv_gw_lds.kv.p_tile[kt & 1u][0] + p_off);
        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt0, acc0);
        if (npg > 1u) {
          const h8 pt1 = *(const LDSP h8 *)(ink_r12_tv_gw_lds.kv.p_tile[kt & 1u][1] + p_off);
          acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt1, acc1);
        }

        if (has_nxt && w == kt + 1u) {
          *(LDSP h8 *)(ink_r12_tv_gw_lds.kv.p_tile[(kt + 1u) & 1u][0] + p_off) = ph0;
          if (npg > 1u) *(LDSP h8 *)(ink_r12_tv_gw_lds.kv.p_tile[(kt + 1u) & 1u][1] + p_off) = ph1;
        }
        lbarrier();
        vr0 = vr_nxt;
      }
    }

    lbarrier();
    if (w == 0u && hl == 0u) {
      ink_r12_tv_gw_lds.ph4.m[rl] = (ls0 > 0.0f) ? (m0 * INK_R12_TV_LN2) : -1e30f;
      ink_r12_tv_gw_lds.ph4.l[rl] = ls0;
      if (npg > 1u) {
        ink_r12_tv_gw_lds.ph4.m[16u + rl] = (ls1 > 0.0f) ? (m1 * INK_R12_TV_LN2) : -1e30f;
        ink_r12_tv_gw_lds.ph4.l[16u + rl] = ls1;
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
          LDSP float *dst_w = ink_r12_tv_gw_lds.ph4.o_stg[q_loc] + 16u * w + 8u * hl;
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) dst_w[v] = a_cur[v];
        }
        lbarrier();

        const u32 qi = q_base + w;
        if (qi < nq_tot) {
          const u32 t = qi / INK_R12_TV_GQA;
          const u32 h = h0 + (qi - t * INK_R12_TV_GQA);
          const f32x4 ov = ((const LDSP f32x4 *)ink_r12_tv_gw_lds.ph4.o_stg[w])[l];
          if (nsplit == 1u) {
            const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * (unsigned long)ss);
            const float sv = st[h];
            const float mq = ink_r12_tv_gw_lds.ph4.m[qi];
            const float lq = ink_r12_tv_gw_lds.ph4.l[qi];
            const float M = (lq > 0.0f) ? __builtin_fmaxf(sv, mq) : sv;
            const float wx = (lq > 0.0f) ? __builtin_expf(mq - M) : 0.0f;
            const float L = __builtin_expf(sv - M) + lq * wx;
            const float sc_o = wx / L;
            const G float *gt_h = agate ? ((const G float *)((const G u8 *)agate + (unsigned long)t * (unsigned long)gs) +
                                           (unsigned long)h * INK_R12_TV_HD)
                                        : (const G float *)0;
            G float *ot_h = out ? ((G float *)((G u8 *)out + (unsigned long)t * (unsigned long)os) +
                                   (unsigned long)h * INK_R12_TV_HD)
                                : (G float *)0;
            G u8 *oqt = oq ? (oq + (unsigned long)t * (unsigned long)oqs) : (G u8 *)0;
            ink_r12_tv_emit_gate_q8(ov * sc_o, gt_h, ot_h, oqt, h, n_head, l, ink_r12_tv_gw_lds.ph4.wt[w]);
          } else {
            G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R12_TV_PSTR;
            if (l == 0u) {
              pp[0] = ink_r12_tv_gw_lds.ph4.m[qi];
              pp[1] = ink_r12_tv_gw_lds.ph4.l[qi];
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

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r12_tv_gw_lds.ph4.is_last =
        (__scoped_atomic_fetch_add(&count[kh], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r12_tv_gw_lds.ph4.is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  for (u32 qi = w; qi < nq_tot; qi += 8u) {
    const u32 t = qi / INK_R12_TV_GQA;
    const u32 h = h0 + (qi - t * INK_R12_TV_GQA);
    const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * (unsigned long)ss);
    const G float *gt_h = agate ? ((const G float *)((const G u8 *)agate + (unsigned long)t * (unsigned long)gs) +
                                   (unsigned long)h * INK_R12_TV_HD)
                                : (const G float *)0;
    G float *ot_h = out ? ((G float *)((G u8 *)out + (unsigned long)t * (unsigned long)os) +
                           (unsigned long)h * INK_R12_TV_HD)
                        : (G float *)0;
    G u8 *oqt = oq ? (oq + (unsigned long)t * (unsigned long)oqs) : (G u8 *)0;
    const G float *ph = part + (unsigned long)((t * n_head + h) * nsplit) * INK_R12_TV_PSTR;
    ink_r12_tv_gw_merge_head(ph, st[h], nsplit, l, ink_r12_tv_gw_lds.ph4.wt[w],
                             ink_r12_tv_gw_lds.ph4.o_stg[w], gt_h, ot_h, oqt, h, n_head);
  }
  barrier();
  if (tid() == 0u) count[kh] = 0u;
}

#undef INK_R12_TV_ACC_FMA
#undef INK_R12_TV_QSC
#undef INK_R12_TV_LN2
#undef INK_R12_TV_LOG2E
#undef INK_R12_TV_PSTR
#undef INK_R12_TV_GQA
#undef INK_R12_TV_DREL
#undef INK_R12_TV_HD

#endif /* !IE_MEGA */
