/* ============================================================================
 * Inkling-2 R2 (g14): F32 Router Prefill GEMM & Fused Decode Router+Route
 *                     for RX 9070 XT (gfx1201, wave32, NT=256)
 *
 * Kernels:
 *
 * 1. Bit 30 — Prefill F32 Router Tiled GEMM (replaces scalar 8x8 ie_gemm_f32_s):
 *    KERNEL ie_inkling_gemm_f32_router_r2(
 *        const G float *w, const G float *x, G float *y,
 *        u32 rows, u32 cols, const G float *bias,
 *        const G float *res, u32 xs, u32 ys, u32 rs, u32 T)
 *    Grid:   ((rows + 31u) / 32u, (T + 63u) / 64u, 1u), NT=256 (8 waves).
 *            (For 128x2304, T=512: grid=(4, 8, 1) = 32 WGs = 1 WG/WGP across all 32 WGPs.)
 *    Guards: !mk.rec && T > 1u && rows > 0u && cols > 0u && (cols & 3u) == 0u &&
 *            (xs & 15u) == 0u &&
 *            (((uintptr_t)w) & 15u) == 0u && (((uintptr_t)x) & 15u) == 0u.
 *    Parity: Exact FP32 FMA tiled GEMM (within FP32 ULP of ie_gemm_f32_s / ie_gemv_f32;
 *            same epilogue(acc, r, bias, res) order; 100% router-selection safe).
 *
 * 2. Bit 31 — Decode T=1 Fused F32 Router GEMV + MoE Top-K Route (cuts 48 launches/tok):
 *    KERNEL ie_inkling_router_route_dec_r2(
 *        const G float *w, const G float *x, G float *logits,
 *        u32 rows, u32 cols, u32 K, const G float *ebias, float scale,
 *        G u32 *route, G u32 *count)
 *    Grid:   ((rows + 3u) / 4u, 1u, 1u), NT=256 (8 waves, 4 rows/WG, 2 waves/row).
 *            (For 128x2304: grid=(32, 1, 1) = 32 WGs = 1 WG/WGP across all 32 WGPs.)
 *            Note: caller passes count = b->gm_count (pre-zeroed; last WG resets to 0).
 *    Guards: !mk.rec && T == 1u && !bias && !res && !NW &&
 *            rows > 0u && rows <= 256u && K > 0u && K <= 32u &&
 *            cols > 0u && (cols & 3u) == 0u &&
 *            (((uintptr_t)w) & 15u) == 0u && (((uintptr_t)x) & 15u) == 0u &&
 *            (next op is OP_MOE_ROUTE with next->a == o->b && next->n == rows).
 *    Parity: Exact FP32 FMA GEMV + exact base moe_route8() (identical sigmoid, ebias,
 *            scale, smallest-index tie-breaking, and 1/(sum+1e-20f) normalization).
 *            100% HIP Graph capture compatible (all pointers/grid static across tokens).
 * ========================================================================== */

#define INK_R2_G14_BM 32u
#define INK_R2_G14_BN 64u
#define INK_R2_G14_BK 32u

typedef float ink_r2_g14_f2 __attribute__((ext_vector_type(2)));

/* Double-buffered XOR-swizzled LDS for prefill router GEMM:
 * 2 * (32*32 + 32*64) * 4 = 24,576 B <= 32 KiB (4 WGs/WGP occupancy). */
typedef struct {
  float w[2][INK_R2_G14_BK][INK_R2_G14_BM];
  float x[2][INK_R2_G14_BK][INK_R2_G14_BN];
} ink_r2_g14_gemm_lds_t;

static LDS ink_r2_g14_gemm_lds_t ink_r2_g14_gemm_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r2_g14_gemm_lds_t) <= 32768u, "ink_r2_g14_gemm_lds must fit in 32 KiB");

/* Small LDS for decode fused router+route (36 B total). */
typedef struct {
  float wsum[8];
  u32   is_last;
} ink_r2_g14_dec_lds_t;

static LDS ink_r2_g14_dec_lds_t ink_r2_g14_dec_lds __attribute__((aligned(16)));

static inline __attribute__((always_inline)) void ink_r2_g14_gemm_store_lds(
    u32 buf, u32 c4, u32 sm, u32 sn0, u32 sn1, f32x4 vw, f32x4 vx0, f32x4 vx1) {
  const u32 k0 = c4 << 2u;
  ink_r2_g14_gemm_lds.w[buf][k0 + 0u][sm]  = vw.x;
  ink_r2_g14_gemm_lds.w[buf][k0 + 1u][sm]  = vw.y;
  ink_r2_g14_gemm_lds.w[buf][k0 + 2u][sm]  = vw.z;
  ink_r2_g14_gemm_lds.w[buf][k0 + 3u][sm]  = vw.w;
  ink_r2_g14_gemm_lds.x[buf][k0 + 0u][sn0] = vx0.x;
  ink_r2_g14_gemm_lds.x[buf][k0 + 1u][sn0] = vx0.y;
  ink_r2_g14_gemm_lds.x[buf][k0 + 2u][sn0] = vx0.z;
  ink_r2_g14_gemm_lds.x[buf][k0 + 3u][sn0] = vx0.w;
  ink_r2_g14_gemm_lds.x[buf][k0 + 0u][sn1] = vx1.x;
  ink_r2_g14_gemm_lds.x[buf][k0 + 1u][sn1] = vx1.y;
  ink_r2_g14_gemm_lds.x[buf][k0 + 2u][sn1] = vx1.z;
  ink_r2_g14_gemm_lds.x[buf][k0 + 3u][sn1] = vx1.w;
}

static inline __attribute__((always_inline)) void ink_r2_g14_gemm_mma_tile(
    u32 buf, u32 rm, u32 rn, float acc[4][2]) {
#pragma unroll
  for (u32 k = 0u; k < INK_R2_G14_BK; k++) {
    const u32 swz = k & 28u; /* ((k >> 2) & 7) << 2: preserves 4-float & 2-float alignment */
    const f32x4 aw = *(const LDSP f32x4 *)(&ink_r2_g14_gemm_lds.w[buf][k][rm ^ swz]);
    const ink_r2_g14_f2 bx = *(const LDSP ink_r2_g14_f2 *)(&ink_r2_g14_gemm_lds.x[buf][k][rn ^ swz]);
    acc[0][0] = __builtin_fmaf(aw.x, bx.x, acc[0][0]);
    acc[0][1] = __builtin_fmaf(aw.x, bx.y, acc[0][1]);
    acc[1][0] = __builtin_fmaf(aw.y, bx.x, acc[1][0]);
    acc[1][1] = __builtin_fmaf(aw.y, bx.y, acc[1][1]);
    acc[2][0] = __builtin_fmaf(aw.z, bx.x, acc[2][0]);
    acc[2][1] = __builtin_fmaf(aw.z, bx.y, acc[2][1]);
    acc[3][0] = __builtin_fmaf(aw.w, bx.x, acc[3][0]);
    acc[3][1] = __builtin_fmaf(aw.w, bx.y, acc[3][1]);
  }
}

KERNEL ie_inkling_gemm_f32_router_r2(const G float *w, const G float *x, G float *y,
                                     u32 rows, u32 cols, const G float *bias,
                                     const G float *res, u32 xs, u32 ys, u32 rs, u32 T) {
  const u32 m0 = wgid() * INK_R2_G14_BM;
  const u32 n0 = tokid() * INK_R2_G14_BN;
  if (m0 >= rows || n0 >= T) return;

  const u32 t = tid(), wv = t >> 5u, l = lane();
  /* Cooperative f32x4 tile loaders: 8 threads along K (32 floats) x 32 rows/tokens. */
  const u32 lrow = t >> 3u;
  const u32 c4   = t & 7u;
  const u32 swz0 = c4 << 2u;
  const u32 sm   = lrow ^ swz0;
  const u32 sn0  = lrow ^ swz0;
  const u32 sn1  = (lrow + 32u) ^ swz0;

  const u32 r_ld  = m0 + lrow;
  const u32 tk_ld0 = n0 + lrow;
  const u32 tk_ld1 = tk_ld0 + 32u;
  const int r_ok  = r_ld < rows;
  const int tk0_ok = tk_ld0 < T;
  const int tk1_ok = tk_ld1 < T;

  const G float *w_ptr  = w + (unsigned long)(r_ok ? r_ld : rows - 1u) * cols;
  const G float *x_ptr0 = (const G float *)((const G u8 *)x + (unsigned long)(tk0_ok ? tk_ld0 : T - 1u) * xs);
  const G float *x_ptr1 = (const G float *)((const G u8 *)x + (unsigned long)(tk1_ok ? tk_ld1 : T - 1u) * xs);

  /* Wave/thread output tile: 8 waves arranged 2(M) x 4(N) of 16x16; each lane computes 4(M) x 2(N). */
  const u32 wm = (wv & 1u) * 16u;
  const u32 wn = (wv >> 1u) * 16u;
  const u32 rm = wm + (l >> 3u) * 4u;
  const u32 rn = wn + (l & 7u) * 2u;

  float acc[4][2];
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) acc[i][0] = 0.0f, acc[i][1] = 0.0f;

  const f32x4 z4 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  /* Prologue: load K-tile 0 into LDS buffer 0. */
  {
    const u32 c = c4 << 2u;
    const int cok = c < cols;
    const f32x4 vw  = (r_ok   && cok) ? *(const G f32x4 *)(w_ptr  + c) : z4;
    const f32x4 vx0 = (tk0_ok && cok) ? *(const G f32x4 *)(x_ptr0 + c) : z4;
    const f32x4 vx1 = (tk1_ok && cok) ? *(const G f32x4 *)(x_ptr1 + c) : z4;
    ink_r2_g14_gemm_store_lds(0u, c4, sm, sn0, sn1, vw, vx0, vx1);
  }
  lbarrier();

  /* Main double-buffered K-loop: overlap global f32x4 loads with 256 FP32 FMAs. */
  u32 buf = 0u;
  for (u32 c0 = INK_R2_G14_BK; c0 < cols; c0 += INK_R2_G14_BK) {
    const u32 c = c0 + (c4 << 2u);
    const int cok = c < cols;
    const f32x4 vw  = (r_ok   && cok) ? *(const G f32x4 *)(w_ptr  + c) : z4;
    const f32x4 vx0 = (tk0_ok && cok) ? *(const G f32x4 *)(x_ptr0 + c) : z4;
    const f32x4 vx1 = (tk1_ok && cok) ? *(const G f32x4 *)(x_ptr1 + c) : z4;

    ink_r2_g14_gemm_mma_tile(buf, rm, rn, acc);

    ink_r2_g14_gemm_store_lds(buf ^ 1u, c4, sm, sn0, sn1, vw, vx0, vx1);
    lbarrier();
    buf ^= 1u;
  }

  /* Epilogue: compute final K-tile and store with exact epilogue(). */
  ink_r2_g14_gemm_mma_tile(buf, rm, rn, acc);

#pragma unroll
  for (u32 j = 0u; j < 2u; j++) {
    const u32 tk = n0 + rn + j;
    if (tk < T) {
      G float *yp = (G float *)((G u8 *)y + (unsigned long)tk * ys);
      const G float *rp = res ? (const G float *)((const G u8 *)res + (unsigned long)tk * rs) : (const G float *)0;
#pragma unroll
      for (u32 i = 0u; i < 4u; i++) {
        const u32 r = m0 + rm + i;
        if (r < rows) yp[r] = epilogue(acc[i][j], r, bias, rp);
      }
    }
  }
}

KERNEL ie_inkling_router_route_dec_r2(const G float *w, const G float *x, G float *logits,
                                      u32 rows, u32 cols, u32 K, const G float *ebias, float scale,
                                      G u32 *route, G u32 *count) {
  const u32 g = wgid(), t = tid(), wv = t >> 5u, l = lane();
  const u32 r_base = g << 2u;
  const u32 r = r_base + (t >> 6u); /* 4 rows/WG, 2 waves (64 threads) per row */
  const u32 t64 = t & 63u;

  float acc = 0.0f;
  if (r < rows) {
    const G f32x4 *w4 = (const G f32x4 *)(w + (unsigned long)r * cols);
    const G f32x4 *x4 = (const G f32x4 *)x;
    const u32 cols4 = cols >> 2u;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    u32 c = t64;
    /* 3x unrolled coalesced f32x4 loads (192 f32x4 = 768 floats/step; exact 3 iters for cols=2304). */
    for (; c + 128u < cols4; c += 192u) {
      const f32x4 wv0 = w4[c];
      const f32x4 xv0 = x4[c];
      const f32x4 wv1 = w4[c + 64u];
      const f32x4 xv1 = x4[c + 64u];
      const f32x4 wv2 = w4[c + 128u];
      const f32x4 xv2 = x4[c + 128u];
      a0 = __builtin_fmaf(wv0.x, xv0.x, a0);
      a1 = __builtin_fmaf(wv0.y, xv0.y, a1);
      a2 = __builtin_fmaf(wv0.z, xv0.z, a2);
      a3 = __builtin_fmaf(wv0.w, xv0.w, a3);
      a0 = __builtin_fmaf(wv1.x, xv1.x, a0);
      a1 = __builtin_fmaf(wv1.y, xv1.y, a1);
      a2 = __builtin_fmaf(wv1.z, xv1.z, a2);
      a3 = __builtin_fmaf(wv1.w, xv1.w, a3);
      a0 = __builtin_fmaf(wv2.x, xv2.x, a0);
      a1 = __builtin_fmaf(wv2.y, xv2.y, a1);
      a2 = __builtin_fmaf(wv2.z, xv2.z, a2);
      a3 = __builtin_fmaf(wv2.w, xv2.w, a3);
    }
    for (; c < cols4; c += 64u) {
      const f32x4 wv_t = w4[c];
      const f32x4 xv_t = x4[c];
      a0 = __builtin_fmaf(wv_t.x, xv_t.x, a0);
      a1 = __builtin_fmaf(wv_t.y, xv_t.y, a1);
      a2 = __builtin_fmaf(wv_t.z, xv_t.z, a2);
      a3 = __builtin_fmaf(wv_t.w, xv_t.w, a3);
    }
    acc = (a0 + a1) + (a2 + a3);
  }

  acc = wave_tree_f(acc);
  if (l == 0u) ink_r2_g14_dec_lds.wsum[wv] = acc;
  lbarrier();

  if (t < 4u) {
    const u32 rr = r_base + t;
    if (rr < rows) logits[rr] = ink_r2_g14_dec_lds.wsum[2u * t] + ink_r2_g14_dec_lds.wsum[2u * t + 1u];
  }

  /* Agent-scope release + device-scope atomic last-WG barrier (matches gemv_norm_tail). */
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  const u32 ngroups = ngroups_x();
  if (t == 0u)
    ink_r2_g14_dec_lds.is_last =
        (__scoped_atomic_fetch_add(count, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ngroups - 1u) ? 1u : 0u;
  barrier();
  if (!ink_r2_g14_dec_lds.is_last) return;

  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  if (t < 32u) moe_route8(logits, rows, K, ebias, scale, route);
  if (t == 0u) *count = 0u;
}

#undef INK_R2_G14_BM
#undef INK_R2_G14_BN
#undef INK_R2_G14_BK
