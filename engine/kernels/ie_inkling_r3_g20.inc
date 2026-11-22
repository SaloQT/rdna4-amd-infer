/* ============================================================================
 * Inkling-2 R3 (g20): Low-VGPR / High-Occupancy Prefill Router GEMM &
 *                     Relative-Position Attention for RX 9070 XT
 *                     (gfx1201, wave32, NT=256)
 *
 * Kernels:
 *
 * 1. Bit 40 — Prefill F32 Router Tiled GEMM (replaces 211-VGPR router_r2):
 *    KERNEL ie_inkling_gemm_f32_router_r3(
 *        const G float *w, const G float *x, G float *y,
 *        u32 rows, u32 cols, const G float *bias,
 *        const G float *res, u32 xs, u32 ys, u32 rs, u32 T)
 *    Grid:   ((rows + 31u) / 32u, (T + 31u) / 32u, 1u), NT=256 (8 waves).
 *            (For 128x2304, T=512: grid=(4, 16, 1) = 64 WGs -> saturates all 64 CUs!)
 *    Guards: !mk.rec && T > 1u && rows > 0u && cols > 0u && (cols & 3u) == 0u &&
 *            (xs & 15u) == 0u &&
 *            (((uintptr_t)w) & 15u) == 0u && (((uintptr_t)x) & 15u) == 0u.
 *    Budget: <= 64 VGPR (0 spills, 3-4 WGs/CU), 16,384 B (16 KiB) double-buffered
 *            XOR-swizzled LDS (32x32x32, 2x2 reg blocking, 4-step K sched_barrier).
 *    Parity: Exact FP32 __builtin_fmaf tiled GEMM (within FP32 ULP of ie_gemm_f32_s /
 *            ie_inkling_gemm_f32_router_r2; exact epilogue(acc, r, bias, res) order).
 *
 * 2. Bit 41 — Prefill Relative-Position WMMA Attention (replaces 174-VGPR / 32 KiB r1):
 *    KERNEL ie_inkling_rel_attn_pf_r3(
 *        const G float *q, const G kvt *kc, const G kvt *vc,
 *        const G float *R, const G float *rel_bias, const G float *sink, G float *out,
 *        u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
 *        u32 d_rel, u32 rel_extent, u32 win, u32 qs, u32 rs, u32 ss, u32 os)
 *    Grid:   ((T + 31u) / 32u, n_kv, 1u), NT=256 (8 waves: 2 producer + 6 consumer).
 *    Guards: !b->kvq8 && !b->kvq4 && T > 1u && hd == 128u && d_rel == 16u &&
 *            n_kv > 0u && n_head == n_kv * 3u && kv_len > 0u.
 *    Budget: <= 144 VGPR (0 spills, 2 WGs/CU on gfx1201), 22,912 B (22.4 KiB <= 24 KiB)
 *            LDS (D=2 producer ring, single-acc QK^T + 2-step sched_barrier, 2-tile PV
 *            sched_barrier, 8-float half-wave rbsc[6][16][9] = 3,456 B vs 12,672 B).
 *    Parity: f32-tol (identical fp16 WMMA QK^T, R*rel_bias, PV math, float rbsc,
 *            log2 online softmax with dynamic sink and causal/SWA masking).
 * ========================================================================== */

_Static_assert(sizeof(kvt) == 2u, "ie_inkling_rel_attn_pf_r3 requires f16 KV cache");

/* ----------------------------------------------------------------------------
 * Bit 40: ie_inkling_gemm_f32_router_r3 (32x32x32 double-buffered, <= 16 KiB LDS)
 * -------------------------------------------------------------------------- */

#define INK_R3_G20_BM 32u
#define INK_R3_G20_BN 32u
#define INK_R3_G20_BK 32u

typedef float ink_r3_g20_f2 __attribute__((ext_vector_type(2)));

/* Double-buffered XOR-swizzled LDS: 2 * (32*32 + 32*32) * 4 = 16,384 B (16 KiB). */
typedef struct {
  float w[2][INK_R3_G20_BK][INK_R3_G20_BM];
  float x[2][INK_R3_G20_BK][INK_R3_G20_BN];
} ink_r3_g20_gemm_lds_t;

static LDS ink_r3_g20_gemm_lds_t ink_r3_g20_gemm_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r3_g20_gemm_lds_t) <= 16384u, "ink_r3_g20_gemm_lds must fit in 16 KiB");

static inline __attribute__((always_inline)) void ink_r3_g20_gemm_store_lds(
    u32 buf, u32 c4, u32 sm, f32x4 vw, f32x4 vx) {
  const u32 k0 = c4 << 2u;
  ink_r3_g20_gemm_lds.w[buf][k0 + 0u][sm] = vw.x;
  ink_r3_g20_gemm_lds.w[buf][k0 + 1u][sm] = vw.y;
  ink_r3_g20_gemm_lds.w[buf][k0 + 2u][sm] = vw.z;
  ink_r3_g20_gemm_lds.w[buf][k0 + 3u][sm] = vw.w;
  ink_r3_g20_gemm_lds.x[buf][k0 + 0u][sm] = vx.x;
  ink_r3_g20_gemm_lds.x[buf][k0 + 1u][sm] = vx.y;
  ink_r3_g20_gemm_lds.x[buf][k0 + 2u][sm] = vx.z;
  ink_r3_g20_gemm_lds.x[buf][k0 + 3u][sm] = vx.w;
}

static inline __attribute__((always_inline)) void ink_r3_g20_gemm_mma_tile(
    u32 buf, u32 rm, u32 rn, float acc[2][2]) {
  /* Rolled outer K-chunks of 4 + sched_barrier(0) prevents LLVM from hoisting
   * all 32 LDS loads and blowing up VGPRs (keeps VGPRs ~44-56 vs 211 in R2). */
#pragma clang loop unroll(disable)
  for (u32 k0 = 0u; k0 < INK_R3_G20_BK; k0 += 4u) {
    const u32 sm = rm ^ k0;
    const u32 sn = rn ^ k0;
#pragma unroll
    for (u32 dk = 0u; dk < 4u; dk++) {
      const u32 k = k0 + dk;
      const ink_r3_g20_f2 aw = *(const LDSP ink_r3_g20_f2 *)(&ink_r3_g20_gemm_lds.w[buf][k][sm]);
      const ink_r3_g20_f2 bx = *(const LDSP ink_r3_g20_f2 *)(&ink_r3_g20_gemm_lds.x[buf][k][sn]);
      acc[0][0] = __builtin_fmaf(aw.x, bx.x, acc[0][0]);
      acc[0][1] = __builtin_fmaf(aw.x, bx.y, acc[0][1]);
      acc[1][0] = __builtin_fmaf(aw.y, bx.x, acc[1][0]);
      acc[1][1] = __builtin_fmaf(aw.y, bx.y, acc[1][1]);
    }
    __builtin_amdgcn_sched_barrier(0);
  }
}

KERNEL ie_inkling_gemm_f32_router_r3(const G float *w, const G float *x, G float *y,
                                     u32 rows, u32 cols, const G float *bias,
                                     const G float *res, u32 xs, u32 ys, u32 rs, u32 T) {
  const u32 m0 = wgid() * INK_R3_G20_BM;
  const u32 n0 = tokid() * INK_R3_G20_BN;
  if (m0 >= rows || n0 >= T) return;

  const u32 t = tid(), wv = t >> 5u, l = lane();
  /* Cooperative f32x4 tile loaders: 8 threads along K (32 floats) x 32 rows/tokens. */
  const u32 lrow = t >> 3u;
  const u32 c4   = t & 7u;
  const u32 sm   = lrow ^ (c4 << 2u);

  const u32 r_ld  = m0 + lrow;
  const u32 tk_ld = n0 + lrow;
  const int r_ok  = r_ld < rows;
  const int tk_ok = tk_ld < T;

  const G float *w_ptr = w + (unsigned long)(r_ok ? r_ld : rows - 1u) * cols;
  const G float *x_ptr = (const G float *)((const G u8 *)x + (unsigned long)(tk_ok ? tk_ld : T - 1u) * xs);

  /* Wave/thread output tile: 8 waves arranged 2(M) x 4(N) of 16x8; each lane computes 2(M) x 2(N). */
  const u32 wm = (wv & 1u) * 16u;
  const u32 wn = (wv >> 1u) * 8u;
  const u32 rm = wm + (l >> 2u) * 2u;
  const u32 rn = wn + (l & 3u) * 2u;

  float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  const f32x4 z4 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};

  /* Prologue: load K-tile 0 into LDS buffer 0. */
  {
    const u32 c = c4 << 2u;
    const int cok = c < cols;
    const f32x4 vw = (r_ok  && cok) ? *(const G f32x4 *)(w_ptr + c) : z4;
    const f32x4 vx = (tk_ok && cok) ? *(const G f32x4 *)(x_ptr + c) : z4;
    ink_r3_g20_gemm_store_lds(0u, c4, sm, vw, vx);
  }
  lbarrier();

  /* Main double-buffered K-loop: overlap global f32x4 loads with 128 FP32 FMAs/thread. */
  u32 buf = 0u;
  for (u32 c0 = INK_R3_G20_BK; c0 < cols; c0 += INK_R3_G20_BK) {
    const u32 c = c0 + (c4 << 2u);
    const int cok = c < cols;
    const f32x4 vw = (r_ok  && cok) ? *(const G f32x4 *)(w_ptr + c) : z4;
    const f32x4 vx = (tk_ok && cok) ? *(const G f32x4 *)(x_ptr + c) : z4;
    __builtin_amdgcn_sched_barrier(0);

    ink_r3_g20_gemm_mma_tile(buf, rm, rn, acc);

    ink_r3_g20_gemm_store_lds(buf ^ 1u, c4, sm, vw, vx);
    lbarrier();
    buf ^= 1u;
  }

  /* Epilogue: compute final K-tile and store with exact epilogue(). */
  ink_r3_g20_gemm_mma_tile(buf, rm, rn, acc);

#pragma unroll
  for (u32 j = 0u; j < 2u; j++) {
    const u32 tk = n0 + rn + j;
    if (tk < T) {
      G float *yp = (G float *)((G u8 *)y + (unsigned long)tk * ys);
      const G float *rp = res ? (const G float *)((const G u8 *)res + (unsigned long)tk * rs) : (const G float *)0;
#pragma unroll
      for (u32 i = 0u; i < 2u; i++) {
        const u32 r = m0 + rm + i;
        if (r < rows) yp[r] = epilogue(acc[i][j], r, bias, rp);
      }
    }
  }
}

/* ----------------------------------------------------------------------------
 * Bit 41: ie_inkling_rel_attn_pf_r3 (<= 144 VGPR, 22,912 B <= 24 KiB LDS)
 * -------------------------------------------------------------------------- */

#define INK_R3_G20_NB    16u
#define INK_R3_G20_D     2u
#define INK_R3_G20_LOG2E 1.44269504088896340736f
/* 1.0f / sqrtf(128.0f) * log2(e) */
#define INK_R3_G20_QSC   (0.08838834764831844299f * 1.44269504088896340736f)
#define INK_R3_G20_ROWB  256u /* hd=128 f16 elements = 256 bytes per (pos, kh) row */

typedef unsigned int ink_r3_g20_u2 __attribute__((ext_vector_type(2)));

/* Total LDS = 8,192 + 8,192 + 3,072 + 3,456 = 22,912 B <= 24 KiB (22.375 KiB). */
typedef struct {
  f16   k[2][INK_R3_G20_NB * 128u];  /*  8,192 B: double-buffered XOR-swizzled K */
  f16   vt[2][128u * INK_R3_G20_NB]; /*  8,192 B: double-buffered XOR-swizzled V^T */
  f16   rb[2][48u * 16u];            /*  3,072 B: double-buffered rel_bias (48 dists x 16) */
  float rbsc[6][16][9];              /*  3,456 B: half-wave 8-float R*rel_bias scratch (stride 9 -> 0 conflicts) */
} ink_r3_g20_attn_lds_t;

static LDS ink_r3_g20_attn_lds_t ink_r3_g20_attn_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r3_g20_attn_lds_t) <= 24576u, "ink_r3_g20_attn_lds must fit in 24 KiB");

static inline __attribute__((always_inline)) u32 ink_r3_g20_k_off(u32 r, u32 c) {
  return r * 128u + 8u * (c ^ r);
}

static inline __attribute__((always_inline)) u32 ink_r3_g20_vt_off(u32 col, u32 h) {
  return 8u * ((2u * col + h) ^ ((col >> 2u) & 1u) ^ ((col >> 3u) & 7u));
}

static inline __attribute__((always_inline)) float ink_r3_g20_xrow(float v) {
  return __builtin_bit_cast(float, xrow_i(__builtin_bit_cast(int, v)));
}

typedef struct {
  u32x4         kx[4]; /* K: 4 chunks of 8 halves (16 B each) per producer thread */
  ink_r3_g20_u2 vx[8]; /* V: 8 keys x 4 contiguous halves (2 column pairs = 8 B each) */
} ink_r3_g20_slot_t;

static inline __attribute__((always_inline)) void ink_r3_g20_transpose8(const u32 *k, u32x4 *ev, u32x4 *od) {
  u32 e[4], o[4];
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    e[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x05040100u);
    o[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x07060302u);
  }
  *ev = (u32x4){e[0], e[1], e[2], e[3]};
  *od = (u32x4){o[0], o[1], o[2], o[3]};
}

static inline __attribute__((always_inline)) void ink_r3_g20_attn_load(
    ink_r3_g20_slot_t *s, const G u8 *kc, const G u8 *vc,
    u32 n_kv, u32 kh, u32 kv_len, u32 k_start, u32 kend, u32 b, u32 pt) {
  const u32 wi = pt >> 5u, L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
  const u32 kb = k_start + INK_R3_G20_NB * b;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r = 8u * wi + 2u * i + sub_r;
    const u32 p = kb + r;
    s->kx[i] = (u32x4)(0u);
    if (p < kend) {
      const u32 kvp = p % kv_len;
      s->kx[i] = *(const G u32x4 *)(kc + ((unsigned long)kvp * n_kv + kh) * INK_R3_G20_ROWB + 16u * c);
    }
  }
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const u32 p = kb + 8u * wi + u;
    s->vx[u] = (ink_r3_g20_u2)(0u);
    if (p < kend) {
      const u32 kvp = p % kv_len;
      s->vx[u] = *(const G ink_r3_g20_u2 *)(vc + ((unsigned long)kvp * n_kv + kh) * INK_R3_G20_ROWB + 8u * L);
    }
  }
}

static inline __attribute__((always_inline)) void ink_r3_g20_attn_publish(
    const ink_r3_g20_slot_t *s, const G float *rel_bias,
    u32 qpos_min, u32 extent_lim, u32 k_start, u32 b, u32 bf, u32 pt) {
  const u32 wi = pt >> 5u, L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r = 8u * wi + 2u * i + sub_r;
    *(LDSP u32x4 *)(ink_r3_g20_attn_lds.k[bf] + ink_r3_g20_k_off(r, c)) = s->kx[i];
  }
  const u32 c0 = 4u * L;
  {
    u32 vx0[8];
#pragma unroll
    for (u32 u = 0u; u < 8u; u++) vx0[u] = s->vx[u].x;
    u32x4 ev0, od0;
    ink_r3_g20_transpose8(vx0, &ev0, &od0);
    *(LDSP u32x4 *)(ink_r3_g20_attn_lds.vt[bf] + ink_r3_g20_vt_off(c0 + 0u, wi)) = ev0;
    *(LDSP u32x4 *)(ink_r3_g20_attn_lds.vt[bf] + ink_r3_g20_vt_off(c0 + 1u, wi)) = od0;
  }
  __builtin_amdgcn_sched_barrier(0);
  {
    u32 vx1[8];
#pragma unroll
    for (u32 u = 0u; u < 8u; u++) vx1[u] = s->vx[u].y;
    u32x4 ev1, od1;
    ink_r3_g20_transpose8(vx1, &ev1, &od1);
    *(LDSP u32x4 *)(ink_r3_g20_attn_lds.vt[bf] + ink_r3_g20_vt_off(c0 + 2u, wi)) = ev1;
    *(LDSP u32x4 *)(ink_r3_g20_attn_lds.vt[bf] + ink_r3_g20_vt_off(c0 + 3u, wi)) = od1;
  }
  __builtin_amdgcn_sched_barrier(0);

  const u32 kb = k_start + INK_R3_G20_NB * b;
  const int d_min_s = (int)qpos_min - (int)(kb + (INK_R3_G20_NB - 1u));
  const u32 d_base_wg = d_min_s > 0 ? (u32)d_min_s : 0u;
  if (d_base_wg < extent_lim && pt < 48u) {
    const u32 d = d_base_wg + pt;
    h8 r0 = (h8)(0.0f), r1 = (h8)(0.0f);
    if (d < extent_lim) {
      const G f32x4 *rp = (const G f32x4 *)(rel_bias + (unsigned long)d * 16ul);
      const f32x4 v0 = rp[0], v1 = rp[1], v2 = rp[2], v3 = rp[3];
      r0 = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
      r1 = (h8){(f16)v2.x, (f16)v2.y, (f16)v2.z, (f16)v2.w, (f16)v3.x, (f16)v3.y, (f16)v3.z, (f16)v3.w};
    }
    *(LDSP h8 *)(ink_r3_g20_attn_lds.rb[bf] + pt * 16u + 0u) = r0;
    *(LDSP h8 *)(ink_r3_g20_attn_lds.rb[bf] + pt * 16u + 8u) = r1;
  }
}

/* Apply a 16-distance WMMA tile (rb) into sv using an 8-float half-wave LDS scratch
 * (exact float precision, 0 bank conflicts on stride 9, 3,456 B vs 12,672 B in R1). */
static inline __attribute__((always_inline)) void ink_r3_g20_apply_rb16(
    v8f rb, u32 base_off, u32 hl, u32 kb, u32 tp, u32 extent_lim, u32 d_base_sub,
    LDSP float *sc_row, v8f *sv) {
  if (hl == 0u) {
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) sc_row[v] = rb[v];
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 v = 0u; v < 8u; v++) {
    const u32 kp = kb + 8u * hl + v;
    if (kp <= tp) {
      const u32 dist = tp - kp;
      if (dist < extent_lim) {
        const u32 rel = (dist - d_base_sub) - base_off;
        if (rel < 8u) (*sv)[v] += sc_row[rel];
      }
    }
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  if (hl == 1u) {
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) sc_row[v] = rb[v];
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 v = 0u; v < 8u; v++) {
    const u32 kp = kb + 8u * hl + v;
    if (kp <= tp) {
      const u32 dist = tp - kp;
      if (dist < extent_lim) {
        const u32 rel = (dist - d_base_sub) - (base_off + 8u);
        if (rel < 8u) (*sv)[v] += sc_row[rel];
      }
    }
  }
}

KERNEL ie_inkling_rel_attn_pf_r3(const G float *q, const G kvt *kc, const G kvt *vc,
                                 const G float *R, const G float *rel_bias,
                                 const G float *sink, G float *out,
                                 u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
                                 u32 d_rel, u32 rel_extent, u32 win,
                                 u32 qs, u32 rs, u32 ss, u32 os) {
  if (hd != 128u || d_rel != 16u || n_kv == 0u || n_head != n_kv * 3u || T == 0u || kv_len == 0u) return;

  const u32 ntile = (T + 31u) >> 5u;
  /* Descending longest-tile-first scheduling reduces causal tail imbalance. */
  const u32 tile = (ntile - 1u) - wgid();
  const u32 kh = tokid();
  const u32 t0 = tile << 5u;
  const u32 w = __builtin_amdgcn_readfirstlane(tid() >> 5u);
  const u32 l = lane(), hl = l >> 4u, rl = l & 15u;

  const u32 tlast = (t0 + 31u < T) ? (t0 + 31u) : (T - 1u);
  const u32 qpos_min = pos0 + t0;
  const u32 qpos_max = pos0 + tlast;
  const u32 k_start = (win > 0u && qpos_min + 1u > win) ? (qpos_min + 1u - win) : 0u;
  const u32 kend = qpos_max + 1u;
  const u32 nblk = (kend > k_start) ? ((kend - k_start + INK_R3_G20_NB - 1u) >> 4u) : 0u;
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;

  if (w >= 6u) { /* Producer waves 6, 7: 2-deep async K/V pipeline + rel_bias staging */
    const G u8 *kb8 = (const G u8 *)kc;
    const G u8 *vb8 = (const G u8 *)vc;
    const u32 pt = tid() - 192u;
    ink_r3_g20_slot_t ring[INK_R3_G20_D];
#pragma unroll
    for (u32 s = 0u; s < INK_R3_G20_D; s++)
      if (s < nblk) ink_r3_g20_attn_load(&ring[s], kb8, vb8, n_kv, kh, kv_len, k_start, kend, s, pt);
    if (nblk > 0u) {
      ink_r3_g20_attn_publish(&ring[0], rel_bias, qpos_min, extent_lim, k_start, 0u, 0u, pt);
      if (INK_R3_G20_D < nblk)
        ink_r3_g20_attn_load(&ring[0], kb8, vb8, n_kv, kh, kv_len, k_start, kend, INK_R3_G20_D, pt);
    }
    for (u32 k0 = 0u; k0 < nblk; k0 += INK_R3_G20_D) {
#pragma unroll
      for (u32 j = 0u; j < INK_R3_G20_D; j++) {
        const u32 k = k0 + j;
        if (k < nblk) {
          lbarrier();
          if (k + 1u < nblk) {
            const u32 nxt = j ^ 1u;
            ink_r3_g20_attn_publish(&ring[nxt], rel_bias, qpos_min, extent_lim, k_start, k + 1u, (k + 1u) & 1u, pt);
            if (k + 1u + INK_R3_G20_D < nblk)
              ink_r3_g20_attn_load(&ring[nxt], kb8, vb8, n_kv, kh, kv_len, k_start, kend, k + 1u + INK_R3_G20_D, pt);
          }
        }
      }
    }
    return;
  }

  /* Consumer waves 0..5: 2 sub-tiles of 16 tokens x 3 GQA query heads */
  const u32 tsub = w >= 3u ? 1u : 0u;
  const u32 hsub = w - 3u * tsub;
  const u32 t_sub0 = t0 + 16u * tsub;
  if (t_sub0 >= T) { /* Idle tail sub-tile (uniform across wave): participate in barriers only */
    for (u32 k = 0u; k < nblk; k++) lbarrier();
    return;
  }

  const u32 h = kh * 3u + hsub;
  const u32 tk = t_sub0 + rl;
  const u32 tq = tk < T ? tk : (T - 1u);
  const u32 tp = pos0 + tk;
  const u32 sub_qpos_min = pos0 + t_sub0;
  const u32 sub_qpos_max = pos0 + (t_sub0 + 15u);

  /* Preload Q (scaled by log2(e)/sqrt(128)) and R (scaled by log2(e)) into VGPRs (36 VGPRs total). */
  h8 qa[8];
  {
    const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (unsigned long)tq * qs) +
                                           (unsigned long)h * 128ul);
#pragma unroll
    for (u32 ks = 0u; ks < 8u; ks++) {
      const f32x4 v0 = qp4[4u * ks + 2u * hl + 0u];
      const f32x4 v1 = qp4[4u * ks + 2u * hl + 1u];
      qa[ks] = (h8){(f16)(v0.x * INK_R3_G20_QSC), (f16)(v0.y * INK_R3_G20_QSC),
                    (f16)(v0.z * INK_R3_G20_QSC), (f16)(v0.w * INK_R3_G20_QSC),
                    (f16)(v1.x * INK_R3_G20_QSC), (f16)(v1.y * INK_R3_G20_QSC),
                    (f16)(v1.z * INK_R3_G20_QSC), (f16)(v1.w * INK_R3_G20_QSC)};
    }
  }
  h8 ra;
  {
    const G f32x4 *rp4 = (const G f32x4 *)((const G float *)((const G u8 *)R + (unsigned long)tq * rs) +
                                           (unsigned long)h * 16ul);
    const f32x4 r0 = rp4[2u * hl + 0u];
    const f32x4 r1 = rp4[2u * hl + 1u];
    ra = (h8){(f16)(r0.x * INK_R3_G20_LOG2E), (f16)(r0.y * INK_R3_G20_LOG2E),
              (f16)(r0.z * INK_R3_G20_LOG2E), (f16)(r0.w * INK_R3_G20_LOG2E),
              (f16)(r1.x * INK_R3_G20_LOG2E), (f16)(r1.y * INK_R3_G20_LOG2E),
              (f16)(r1.z * INK_R3_G20_LOG2E), (f16)(r1.w * INK_R3_G20_LOG2E)};
  }
  __builtin_amdgcn_sched_barrier(0);

  /* Dynamic per-token, per-head sink initializes log2 online softmax. */
  const float sink_val = ((const G float *)((const G u8 *)sink + (unsigned long)tq * ss))[h];
  float m = sink_val * INK_R3_G20_LOG2E;
  float ls = (hl == 0u) ? 1.0f : 0.0f;

  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2u) & 1u) ^ (rl >> 3u);
  const v8f zero = (v8f)(0.0f);
  v8f o[8];
#pragma unroll
  for (u32 nt = 0u; nt < 8u; nt++) o[nt] = zero;

  for (u32 k = 0u; k < nblk; k++) {
    lbarrier();
    const u32 kb = k_start + INK_R3_G20_NB * k;
    const u32 bf = k & 1u;

    /* S^T = K * Q^T (16x16x128): single accumulator + 2-step sched_barrier trims ~32 VGPRs. */
    v8f sv = zero;
#pragma unroll
    for (u32 ks = 0u; ks < 8u; ks += 2u) {
      const h8 k0 = *(const LDSP h8 *)(ink_r3_g20_attn_lds.k[bf] + ink_r3_g20_k_off(rl, 2u * ks + hl));
      const h8 k1 = *(const LDSP h8 *)(ink_r3_g20_attn_lds.k[bf] + ink_r3_g20_k_off(rl, 2u * (ks + 1u) + hl));
      sv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[ks], sv);
      sv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[ks + 1u], sv);
      __builtin_amdgcn_sched_barrier(0);
    }

    /* Relative position bias: sequential 16-dist passes reuse rb0 regs and 3,456 B rbsc. */
    const int d_min_sub_s = (int)sub_qpos_min - (int)(kb + (INK_R3_G20_NB - 1u));
    const u32 d_base_sub = d_min_sub_s > 0 ? (u32)d_min_sub_s : 0u;
    if (d_base_sub < extent_lim) {
      const int d_min_wg_s = (int)qpos_min - (int)(kb + (INK_R3_G20_NB - 1u));
      const u32 d_base_wg = d_min_wg_s > 0 ? (u32)d_min_wg_s : 0u;
      const u32 off = d_base_sub - d_base_wg;
      LDSP float *sc_row = ink_r3_g20_attn_lds.rbsc[w][rl];

      const h8 b0 = *(const LDSP h8 *)(ink_r3_g20_attn_lds.rb[bf] + (off + rl) * 16u + 8u * hl);
      const v8f rb0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0, ra, zero);
      ink_r3_g20_apply_rb16(rb0, 0u, hl, kb, tp, extent_lim, d_base_sub, sc_row, &sv);

      if (d_base_sub + 16u < extent_lim && sub_qpos_max >= kb + d_base_sub + 16u) {
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        const h8 b1 = *(const LDSP h8 *)(ink_r3_g20_attn_lds.rb[bf] + (off + 16u + rl) * 16u + 8u * hl);
        const v8f rb1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1, ra, zero);
        ink_r3_g20_apply_rb16(rb1, 16u, hl, kb, tp, extent_lim, d_base_sub, sc_row, &sv);
      }
      __builtin_amdgcn_sched_barrier(0);
    }

    /* Causal + SWA window mask (uniform branch skips interior unmasked blocks). */
    if (kb + (INK_R3_G20_NB - 1u) > sub_qpos_min || (win > 0u && sub_qpos_max >= kb + win)) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 kp = kb + 8u * hl + v;
        if (kp > tp || (win > 0u && (tp - kp) >= win)) sv[v] = -__builtin_inff();
      }
    }

    float bm = sv[0];
#pragma unroll
    for (u32 v = 1u; v < 8u; v++) bm = __builtin_fmaxf(bm, sv[v]);
    bm = __builtin_fmaxf(bm, ink_r3_g20_xrow(bm));

    /* Exact zero-tile PV skip: also skips blocks fully masked out for this sub-tile. */
    if (__builtin_amdgcn_ballot_w32(!(bm - m < -30.0f)) == 0u) continue;

    const float mn = __builtin_fmaxf(m, bm);
    const float c = __builtin_amdgcn_exp2f(m - mn);
    h8 pb;
    float ps = 0.0f;
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      const float p = __builtin_amdgcn_exp2f(sv[v] - mn);
      ps += p;
      pb[v] = (f16)p;
    }
    ls = ls * c + ps;
    m = mn;

    if (__builtin_amdgcn_ballot_w32(c != 1.0f) != 0u) {
#pragma unroll
      for (u32 nt = 0u; nt < 8u; nt++)
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c;
    }

    /* O^T += V^T * P^T (8 WMMAs across hd=128; 2-tile sched_barrier caps live V loads to 8 VGPRs). */
    const LDSP f16 *vt_b = ink_r3_g20_attn_lds.vt[bf];
#pragma unroll
    for (u32 g = 0u; g < 2u; g++) {
      const LDSP f16 *vg = vt_b + 1024u * g;
      const h8 v0 = *(const LDSP h8 *)(vg + 0u   + 8u * vt_lane);
      const h8 v1 = *(const LDSP h8 *)(vg + 256u + 8u * (vt_lane ^ 2u));
      o[4u * g + 0u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[4u * g + 0u]);
      o[4u * g + 1u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[4u * g + 1u]);
      __builtin_amdgcn_sched_barrier(0);
      const h8 v2 = *(const LDSP h8 *)(vg + 512u + 8u * (vt_lane ^ 4u));
      const h8 v3 = *(const LDSP h8 *)(vg + 768u + 8u * (vt_lane ^ 6u));
      o[4u * g + 2u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v2, pb, o[4u * g + 2u]);
      o[4u * g + 3u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v3, pb, o[4u * g + 3u]);
      __builtin_amdgcn_sched_barrier(0);
    }
  }

  /* Final normalization (wave-uniform xrow before tail mask) and coalesced f32x4 output stores. */
  const float inv = 1.0f / (ls + ink_r3_g20_xrow(ls));
  if (tk < T) {
    G f32x4 *op4 = (G f32x4 *)((G float *)((G u8 *)out + (unsigned long)tk * os) + (unsigned long)h * 128ul);
#pragma unroll
    for (u32 nt = 0u; nt < 8u; nt++) {
      const f32x4 a0 = {o[nt][0] * inv, o[nt][1] * inv, o[nt][2] * inv, o[nt][3] * inv};
      const f32x4 a1 = {o[nt][4] * inv, o[nt][5] * inv, o[nt][6] * inv, o[nt][7] * inv};
      op4[4u * nt + 2u * hl + 0u] = a0;
      op4[4u * nt + 2u * hl + 1u] = a1;
    }
  }
}

#undef INK_R3_G20_BM
#undef INK_R3_G20_BN
#undef INK_R3_G20_BK
#undef INK_R3_G20_NB
#undef INK_R3_G20_D
#undef INK_R3_G20_LOG2E
#undef INK_R3_G20_QSC
#undef INK_R3_G20_ROWB
