/* ============================================================================
 * Inkling-2 Round 10 (Wave 9/10, H02):
 *   Expert-Major L2-Localized 3-WGs/CU Zero-fma_mix IQ4_NL MoE Prefill Down
 *   WMMA GEMM + Fused Coalesced Staggered Weighted Scatter-Accumulate (and
 *   Exact-SHA Non-Atomic Companion) for RX 9070 XT (gfx1201, wave32, NT=256).
 *
 * Target: MoE ffn_down_exps MAT_IQ4NL (kind=8, 2304x1152, nb=36, 128E/K=8) +
 *         OP_MOE_SUM (~22% of 4K/16K prefill + 37.7 MB/visit exp_down traffic
 *         + 384 moe_sum launches/4K!).
 *
 * Critical R15 (b71) / R36 (b99) Root-Cause Lesson Applied:
 *   - Keeps r2_g15's EXPERT-MAJOR WG scheduling (grid = (rows/128, n_exp, 1))
 *     so all 18 row-tiles of each hot expert stay L2-localized AND each WG
 *     reuses its own 83 KiB weight slice across consecutive 64-pair chunks in
 *     L0/L1/L2 (never scattering experts across the flat tile worklist!).
 *   - Separates `while (pb < p1 && p1 - pb > 48u)` (WJ=4 full 64-pair loop) from a
 *     branch-free single-pass tail (WJ=3 / WJ=2 / WJ=1 for rem <= 48) and
 *     derives per-phase lane/address offsets locally from pinned tid() so ZERO
 *     address/epilogue VGPRs span across K-loops -> <= 128 VGPR (3 WGs/CU =
 *     +50% occupancy over r2_g15!), 0/0 spills, 0 v_fma_mix_f32!
 *   - Reuses dead K-loop LDS (union ep[32][129] inside existing 21,248 B LDS
 *     = 0 extra LDS bytes!) to transpose outputs per 32-token strip:
 *       1) scatter_r10: loads route weight rw[pid] ONCE per token in SGPR,
 *          staggers token order by (e & 7) and 32-row cache-line order by
 *          (e & 3), and issues fully COALESCED 128-B wave32 native FP32
 *          __builtin_amdgcn_global_atomic_fadd_f32 into pre-zeroed ysum[tok, rows]
 *          (16x fewer L2 atomic cache-line transactions than lane-major scatter!).
 *       2) expmaj_r10: non-atomic exact-SHA companion writing exp_down via
 *          fully coalesced 512-B global_store_b128 (f32x4) stores (drop-in for
 *          ie_inkling_gemm_iq4nl_exp_r2 to A/B 3-WG/CU expert-major independently!).
 *
 * Exported Kernels, Grids, Guards, and Parity:
 *
 * 1. KERNEL ie_inkling_gemm_iq4nl_scatter_r10(
 *        const G u32 *qw0, const G u16 *qs0, const G u8 *X, G float *ysum,
 *        u32 rows, u32 nb, const G u32 *list, const G u32 *offs,
 *        const G u32 *route, u32 rts, u32 K,
 *        u32 qwn, u32 qsn, u32 xs, u32 os, u32 xdiv)
 *    - Fuses [OP_GEMV_EXP(MAT_IQ4NL), OP_MOE_SUM(res=-1)] at T >= 17.
 *    - Precondition: host pre-zeros ysum[0 .. T*os - 1] (hipMemsetAsync 0).
 *    - Grid:   ((rows + 127u) / 128u, n_exp, 1u), NT = 256.
 *    - Guard:  !mk.rec && T >= 17u && kind == MAT_IQ4NL (8u) && rows > 0u &&
 *              nb > 0u && (nb & 3u) == 0u && list && offs && route && K > 0u &&
 *              o1->res < 0 && buf_disjoint(X, ysum).
 *    - Parity: f32-tol (~1e-6 due to non-deterministic atomic FP32 addition order;
 *              verify KLD <= 0.0005 vs R00 reference).
 *
 * 2. KERNEL ie_inkling_gemm_iq4nl_expmaj_r10(
 *        const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
 *        u32 rows, u32 nb, const G u32 *list, const G u32 *offs, u32 np,
 *        u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv, u32 kind,
 *        const G u8 *res, u32 rs)
 *    - Exact drop-in signature & grid of ie_inkling_gemm_iq4nl_exp_r2!
 *    - Grid:   ((rows + 127u) / 128u, list ? m->n_exp : (np + 63u) / 64u, 1u), NT = 256.
 *    - Parity: 100% bitwise-exact (0-bit-diff exact-SHA) vs ie_inkling_gemm_iq4nl_exp_r2!
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R10_IQ4_KB     4u
#define INK_R10_IQ4_R      128u
#define INK_R10_IQ4_SW_STR 160u /* 128 f16 + 32 f16 pad = 320 B -> 0-conflict 2x ds_store_b32 */
#define INK_R10_IQ4_EP_STR 129u /* 128 f32 + 1 f32 pad -> 0-conflict 32-lane contiguous reads */

typedef unsigned int ink_r10_iq4_u32x2 __attribute__((ext_vector_type(2)));

struct __attribute__((packed, aligned(4))) ink_r10_iq4_u32x2p {
  ink_r10_iq4_u32x2 v;
};

static inline __attribute__((always_inline)) ink_r10_iq4_u32x2 ink_r10_iq4_ld2(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r10_iq4_u32x2p *)p)->v;
}

typedef struct {
  union {
    struct {
      u8    x[2][64u * 128u] __attribute__((aligned(16)));                           /* 16,384 B */
      f16   sw[2][INK_R10_IQ4_KB * INK_R10_IQ4_SW_STR] __attribute__((aligned(16))); /*  2,560 B */
      float sx[2][INK_R10_IQ4_KB * 64u] __attribute__((aligned(32)));                /*  2,048 B */
    } k;
    float ep[32u][INK_R10_IQ4_EP_STR] __attribute__((aligned(32)));                  /* 16,512 B */
  } u;
  u32 pid[64] __attribute__((aligned(16)));                                          /*    256 B */
} ink_r10_iq4_lds_t;

_Static_assert(sizeof(ink_r10_iq4_lds_t) == 21248u, "ink_r10_iq4_lds_t must be 21,248 B (<= 21.5 KiB = 3 WGs/CU)");

static inline __attribute__((always_inline)) u32 ink_r10_iq4_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

/* 0-bank-conflict 2x ds_store_b32 f16 weight-scale staging (waves 0..3, 128 rows). */
static inline __attribute__((always_inline)) void ink_r10_iq4_stage_sw(
    LDSP f16 *lsw, const u32 t0, const int is_sw, const ink_r10_iq4_u32x2 rsw) {
  if (is_sw) {
    u32 t_loc = t0;
    __asm__ volatile("" : "+v"(t_loc));
    const u32 w0 = rsw.x, w1 = rsw.y;
    const u32 p0 = (u32)__builtin_amdgcn_update_dpp(0, (int)w0, 0xB1, 0xF, 0xF, true);
    const u32 p1 = (u32)__builtin_amdgcn_update_dpp(0, (int)w1, 0xB1, 0xF, 0xF, true);
    const u32 odd = t_loc & 1u;
    const u32 sel = odd ? 0x03020706u : 0x05040100u;
    const u32 v_a = __builtin_amdgcn_perm(p0, w0, sel);
    const u32 v_b = __builtin_amdgcn_perm(p1, w1, sel);
    const u32 r2 = t_loc & ~1u;
    LDSP u32 *sw32 = (LDSP u32 *)lsw;
    const u32 str32 = INK_R10_IQ4_SW_STR >> 1u;
    sw32[(0u + odd) * str32 + (r2 >> 1u)] = v_a;
    sw32[(2u + odd) * str32 + (r2 >> 1u)] = v_b;
  }
}

static inline __attribute__((always_inline)) void ink_r10_iq4_stage_scales(
    LDSP f16 *lsw, LDSP float *lsx, const u32 t0, const u32 sx_stride,
    const int is_sw, const int is_sx, const ink_r10_iq4_u32x2 rsw, const f32x4 rsx) {
  ink_r10_iq4_stage_sw(lsw, t0, is_sw, rsw);
  if (is_sx) {
    u32 t_loc = t0;
    __asm__ volatile("" : "+v"(t_loc));
    lsx[0u * sx_stride + t_loc] = rsx.x;
    lsx[1u * sx_stride + t_loc] = rsx.y;
    lsx[2u * sx_stride + t_loc] = rsx.z;
    lsx[3u * sx_stride + t_loc] = rsx.w;
  }
}

/* Load f16 weight scales (1x ds_load_b128) and pin f32 conversion to kill 100% of v_fma_mix_f32! */
static inline __attribute__((always_inline)) f8 ink_r10_iq4_load_sw_f8(
    const LDSP f16 *lsw, const u32 k, const u32 wr, const u32 h) {
  const h8 sw_h = *(const LDSP h8 *)(lsw + k * INK_R10_IQ4_SW_STR + wr + 8u * h);
  f8 sw = __builtin_convertvector(sw_h, f8);
  __asm__ volatile("" : "+v"(sw));
  return sw;
}

static inline __attribute__((always_inline)) void ink_r10_iq4_decode_w(
    const u32x4 wk, const u32 h, v2i *a0_out, v2i *a1_out) {
  u32x4 nb4 = h ? ((wk >> 4u) & 0x0F0F0F0Fu) : (wk & 0x0F0F0F0Fu);
  nb4.x = ie_iq4_map(nb4.x);
  nb4.y = ie_iq4_map(nb4.y);
  nb4.z = ie_iq4_map(nb4.z);
  nb4.w = ie_iq4_map(nb4.w);
  *a0_out = (v2i){(int)nb4.x, (int)nb4.y};
  *a1_out = (v2i){(int)nb4.z, (int)nb4.w};
}

static inline __attribute__((always_inline)) void ink_r10_iq4_k_wj4(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 l16, const u32 h, const u32 lx_base0, f8 acc[4]) {
  /* Use inline-zero WMMA acc + exact v_cvt_f32_i32 instead of 8-VGPR 0x4B400000
   * constant -> frees 8 VGPRs so WJ=4 hits <= 120 VGPR (true 12w/SIMD = 3 WGs/CU!).
   * Exact-SHA: |c| <= 32*128*128 = 524,288 < 2^23 -> bit-identical IEEE-754 float. */
  const v8i zm = (v8i)(0);
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * 64u + l16;

  v2i a0, a1;
  ink_r10_iq4_decode_w(wk, h, &a0, &a1);

  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + 0u * 2048u);
  const u32x4 b1 = *(const LDSP u32x4 *)(lx_k + 1u * 2048u);
  const float dx0 = lsx_k[0u];
  const float dx1 = lsx_k[16u];
  __builtin_amdgcn_sched_barrier(0);

  const v2i b0_lo = {(int)b0.x, (int)b0.y}, b0_hi = {(int)b0.z, (int)b0.w};
  const v2i b1_lo = {(int)b1.x, (int)b1.y}, b1_hi = {(int)b1.z, (int)b1.w};
  v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_lo, zm, 0);
  v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b1_lo, zm, 0);
  c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_hi, c0, 0);
  c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1_hi, c1, 0);
  __builtin_amdgcn_sched_barrier(0);

  const f8 sw = ink_r10_iq4_load_sw_f8(lsw, k, wr, h);
  acc[0] += (sw * dx0) * __builtin_convertvector(c0, f8);
  __asm__ volatile("" : : "v"(acc[0]));
  __builtin_amdgcn_sched_barrier(0);
  acc[1] += (sw * dx1) * __builtin_convertvector(c1, f8);
  __asm__ volatile("" : : "v"(acc[1]) : "memory");
  __builtin_amdgcn_sched_barrier(0);

  const u32x4 b2 = *(const LDSP u32x4 *)(lx_k + 2u * 2048u);
  const u32x4 b3 = *(const LDSP u32x4 *)(lx_k + 3u * 2048u);
  const float dx2 = lsx_k[32u];
  const float dx3 = lsx_k[48u];
  __builtin_amdgcn_sched_barrier(0);

  const v2i b2_lo = {(int)b2.x, (int)b2.y}, b2_hi = {(int)b2.z, (int)b2.w};
  const v2i b3_lo = {(int)b3.x, (int)b3.y}, b3_hi = {(int)b3.z, (int)b3.w};
  v8i c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b2_lo, zm, 0);
  v8i c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b3_lo, zm, 0);
  c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b2_hi, c2, 0);
  c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b3_hi, c3, 0);
  __builtin_amdgcn_sched_barrier(0);

  acc[2] += (sw * dx2) * __builtin_convertvector(c2, f8);
  __asm__ volatile("" : : "v"(acc[2]));
  __builtin_amdgcn_sched_barrier(0);
  acc[3] += (sw * dx3) * __builtin_convertvector(c3, f8);
  __asm__ volatile("" : : "v"(acc[3]) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r10_iq4_k_wj3(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 l16, const u32 h, const u32 lx_base0, f8 acc[3]) {
  const v8i zm = (v8i)(0x4B400000);
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * 64u + l16;

  v2i a0, a1;
  ink_r10_iq4_decode_w(wk, h, &a0, &a1);

  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + 0u * 2048u);
  const u32x4 b1 = *(const LDSP u32x4 *)(lx_k + 1u * 2048u);
  const float dx0 = lsx_k[0u];
  const float dx1 = lsx_k[16u];
  __builtin_amdgcn_sched_barrier(0);

  const v2i b0_lo = {(int)b0.x, (int)b0.y}, b0_hi = {(int)b0.z, (int)b0.w};
  const v2i b1_lo = {(int)b1.x, (int)b1.y}, b1_hi = {(int)b1.z, (int)b1.w};
  v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_lo, zm, 0);
  v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b1_lo, zm, 0);
  c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_hi, c0, 0);
  c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1_hi, c1, 0);
  __builtin_amdgcn_sched_barrier(0);

  const f8 sw = ink_r10_iq4_load_sw_f8(lsw, k, wr, h);
  acc[0] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  __asm__ volatile("" : : "v"(acc[0]));
  __builtin_amdgcn_sched_barrier(0);
  acc[1] += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
  __asm__ volatile("" : : "v"(acc[1]) : "memory");
  __builtin_amdgcn_sched_barrier(0);

  const u32x4 b2 = *(const LDSP u32x4 *)(lx_k + 2u * 2048u);
  const float dx2 = lsx_k[32u];
  __builtin_amdgcn_sched_barrier(0);

  const v2i b2_lo = {(int)b2.x, (int)b2.y}, b2_hi = {(int)b2.z, (int)b2.w};
  v8i c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b2_lo, zm, 0);
  c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b2_hi, c2, 0);
  __builtin_amdgcn_sched_barrier(0);

  acc[2] += (sw * dx2) * (__builtin_bit_cast(f8, c2) - 12582912.0f);
  __asm__ volatile("" : : "v"(acc[2]) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r10_iq4_k_wj2(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 l16, const u32 h, const u32 lx_base0, f8 acc[2]) {
  const v8i zm = (v8i)(0x4B400000);
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * 32u + l16;

  v2i a0, a1;
  ink_r10_iq4_decode_w(wk, h, &a0, &a1);

  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + 0u * 2048u);
  const u32x4 b1 = *(const LDSP u32x4 *)(lx_k + 1u * 2048u);
  const float dx0 = lsx_k[0u];
  const float dx1 = lsx_k[16u];
  __builtin_amdgcn_sched_barrier(0);

  const v2i b0_lo = {(int)b0.x, (int)b0.y}, b0_hi = {(int)b0.z, (int)b0.w};
  const v2i b1_lo = {(int)b1.x, (int)b1.y}, b1_hi = {(int)b1.z, (int)b1.w};
  v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_lo, zm, 0);
  v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b1_lo, zm, 0);
  c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_hi, c0, 0);
  c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1_hi, c1, 0);
  __builtin_amdgcn_sched_barrier(0);

  const f8 sw = ink_r10_iq4_load_sw_f8(lsw, k, wr, h);
  acc[0] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  __asm__ volatile("" : : "v"(acc[0]));
  __builtin_amdgcn_sched_barrier(0);
  acc[1] += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
  __asm__ volatile("" : : "v"(acc[1]) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r10_iq4_k_wj1(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 l16, const u32 h, const u32 lx_base0, f8 *acc0) {
  const v8i zm = (v8i)(0x4B400000);
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * 32u + l16;

  v2i a0, a1;
  ink_r10_iq4_decode_w(wk, h, &a0, &a1);

  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + 0u * 2048u);
  const float dx0 = lsx_k[0u];
  __builtin_amdgcn_sched_barrier(0);

  const v2i b0_lo = {(int)b0.x, (int)b0.y}, b0_hi = {(int)b0.z, (int)b0.w};
  v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_lo, zm, 0);
  c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_hi, c0, 0);
  __builtin_amdgcn_sched_barrier(0);

  const f8 sw = ink_r10_iq4_load_sw_f8(lsw, k, wr, h);
  *acc0 += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  __asm__ volatile("" : : "v"(*acc0) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

/* Shared K-loop bodies for WJ = 4, 3, 2, 1.
 * Every helper derives lane/address indices locally from a pinned tid() copy
 * so ZERO tile-invariant address VGPRs leak across the outer expert-major loop! */
static inline __attribute__((always_inline)) void ink_r10_iq4_run_wj4(
    LDSP ink_r10_iq4_lds_t *lds, const G u8 *qw_e, const G u8 *qs_e,
    const G u8 *X_u, const G u8 *X_s, const G u32 *list,
    const u32 pb, const u32 nj, const u32 rbase, const u32 rows, const u32 nb,
    const u32 xs, const u32 xdiv, f8 acc[4]) {
  u32 t0 = tid();
  __asm__ volatile("" : "+v"(t0));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = t0 & 31u, l16 = l & 15u, h = l >> 4u;
  const u32 wr = 16u * w;
  const int is_sw = (w < 4u);
  const int is_sx = (w < 2u);

  if (is_sx) {
    const u32 pj = pb + (t0 < nj ? t0 : 0u);
    lds->pid[t0] = list ? list[pj] : pj;
  }
  lbarrier();

  const u32 rw_idx = rbase + wr + l16;
  const u32 w_off = (rw_idx < rows ? rw_idx : rows - 1u) * (nb * 16u);
  const u32 sr = rbase + (is_sw ? t0 : 0u);
  const u32 sw_off = (sr < rows ? sr : rows - 1u) * (nb * 2u);
  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  const u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);
  const u32 lx_base0 = l16 * 128u + ((h ^ (l16 & 7u)) << 4u);

  const u32 xoff0 = ink_r10_iq4_div_x(lds->pid[tk_ld], xdiv) * xs + c8_ld * 16u;
  const u32 xoff1 = ink_r10_iq4_div_x(lds->pid[tk_ld + 32u], xdiv) * xs + c8_ld * 16u;
  const u32 sxoff = is_sx ? ink_r10_iq4_div_x(lds->pid[t0 & 63u], xdiv) * xs : 0u;

#pragma unroll
  for (u32 j = 0u; j < 4u; j++) acc[j] = (f8)(0.0f);

  u32x4 wa[INK_R10_IQ4_KB];
#pragma unroll
  for (u32 k = 0u; k < INK_R10_IQ4_KB; k++) wa[k] = *(const G u32x4 *)(qw_e + w_off + k * 16u);
  {
    const u32x4 rx0_i = *(const G u32x4 *)(X_u + xoff0);
    const u32x4 rx1_i = *(const G u32x4 *)(X_u + xoff1);
    const ink_r10_iq4_u32x2 rsw_i = is_sw ? ink_r10_iq4_ld2(qs_e + sw_off) : (ink_r10_iq4_u32x2){0u, 0u};
    const f32x4 rsx_i = is_sx ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
    *(LDSP u32x4 *)(lds->u.k.x[0] + st_x) = rx0_i;
    *(LDSP u32x4 *)(lds->u.k.x[0] + st_x + 4096u) = rx1_i;
    ink_r10_iq4_stage_scales(lds->u.k.sw[0], lds->u.k.sx[0], t0, 64u, is_sw, is_sx, rsw_i, rsx_i);
  }
  lbarrier();

  u32 bf = 0u;
  u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R10_IQ4_KB) {
    const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R10_IQ4_KB));
    const int more = (kb_n < nb);
    if (more) {
      const G u8 *X_n = rq_ptr_u(X_u + (unsigned long)kb_n * 32ul);
      rx0 = *(const G u32x4 *)(X_n + xoff0);
      rx1 = *(const G u32x4 *)(X_n + xoff1);
    }
    __builtin_amdgcn_sched_barrier(0);
    const LDSP u8 *lx = lds->u.k.x[bf];
    const LDSP f16 *lsw = lds->u.k.sw[bf];
    const LDSP float *lsx = lds->u.k.sx[bf];

    ink_r10_iq4_k_wj4(wa[0], lx, lsw, lsx, 0u, wr, l16, h, lx_base0, acc);
    ink_r10_iq4_k_wj4(wa[1], lx, lsw, lsx, 1u, wr, l16, h, lx_base0, acc);

    /* Store rx0/rx1 -> inactive LDS (frees 8 VGPRs) and prefetch rsw/rsx while wa[0..1] are dead! */
    ink_r10_iq4_u32x2 rsw = {0u, 0u};
    f32x4 rsx = (f32x4)(0.0f);
    if (more) {
      const u32 nbf = bf ^ 1u;
      *(LDSP u32x4 *)(lds->u.k.x[nbf] + st_x) = rx0;
      *(LDSP u32x4 *)(lds->u.k.x[nbf] + st_x + 4096u) = rx1;
      const G u8 *qs_n = rq_ptr_u(qs_e + (unsigned long)kb_n * 2ul);
      const G u8 *Xs_n = rq_ptr_u(X_s + (unsigned long)kb_n * 4ul);
      if (is_sw) rsw = ink_r10_iq4_ld2(qs_n + sw_off);
      if (is_sx) rsx = rq_load_f32x4((const G float *)(Xs_n + sxoff));
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r10_iq4_k_wj4(wa[2], lx, lsw, lsx, 2u, wr, l16, h, lx_base0, acc);

    /* Stage rsw/rsx -> inactive LDS (frees 6 VGPRs!) and reload wa[0..1] while wa[2] is dead! */
    if (more) {
      const u32 nbf = bf ^ 1u;
      ink_r10_iq4_stage_scales(lds->u.k.sw[nbf], lds->u.k.sx[nbf], t0, 64u, is_sw, is_sx, rsw, rsx);
      __builtin_amdgcn_sched_barrier(0);
      const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
      wa[0] = *(const G u32x4 *)(qw_n + w_off + 0u);
      wa[1] = *(const G u32x4 *)(qw_n + w_off + 16u);
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r10_iq4_k_wj4(wa[3], lx, lsw, lsx, 3u, wr, l16, h, lx_base0, acc);
    if (more) {
      const u32 nbf = bf ^ 1u;
      const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
      wa[2] = *(const G u32x4 *)(qw_n + w_off + 32u);
      wa[3] = *(const G u32x4 *)(qw_n + w_off + 48u);
      lbarrier();
      bf = nbf;
    }
  }
}

static inline __attribute__((always_inline)) void ink_r10_iq4_run_wj3(
    LDSP ink_r10_iq4_lds_t *lds, const G u8 *qw_e, const G u8 *qs_e,
    const G u8 *X_u, const G u8 *X_s, const G u32 *list,
    const u32 pb, const u32 nj, const u32 rbase, const u32 rows, const u32 nb,
    const u32 xs, const u32 xdiv, f8 acc[3]) {
  u32 t0 = tid();
  __asm__ volatile("" : "+v"(t0));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = t0 & 31u, l16 = l & 15u, h = l >> 4u;
  const u32 wr = 16u * w;
  const int is_sw = (w < 4u);
  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  const int ld_hi = (tk_ld < 16u);
  const int is_sx = (t0 < 48u);

  if (is_sx) {
    const u32 pj = pb + (t0 < nj ? t0 : 0u);
    lds->pid[t0] = list ? list[pj] : pj;
  }
  lbarrier();

  const u32 rw_idx = rbase + wr + l16;
  const u32 w_off = (rw_idx < rows ? rw_idx : rows - 1u) * (nb * 16u);
  const u32 sr = rbase + (is_sw ? t0 : 0u);
  const u32 sw_off = (sr < rows ? sr : rows - 1u) * (nb * 2u);
  const u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);
  const u32 lx_base0 = l16 * 128u + ((h ^ (l16 & 7u)) << 4u);

  const u32 xoff0 = ink_r10_iq4_div_x(lds->pid[tk_ld], xdiv) * xs + c8_ld * 16u;
  const u32 xoff1 = ld_hi ? (ink_r10_iq4_div_x(lds->pid[tk_ld + 32u], xdiv) * xs + c8_ld * 16u) : 0u;
  const u32 sxoff = is_sx ? ink_r10_iq4_div_x(lds->pid[t0], xdiv) * xs : 0u;

  acc[0] = (f8)(0.0f);
  acc[1] = (f8)(0.0f);
  acc[2] = (f8)(0.0f);

  u32x4 wa[INK_R10_IQ4_KB];
#pragma unroll
  for (u32 k = 0u; k < INK_R10_IQ4_KB; k++) wa[k] = *(const G u32x4 *)(qw_e + w_off + k * 16u);
  {
    const u32x4 rx0_i = *(const G u32x4 *)(X_u + xoff0);
    const u32x4 rx1_i = ld_hi ? *(const G u32x4 *)(X_u + xoff1) : (u32x4)(0u);
    const ink_r10_iq4_u32x2 rsw_i = is_sw ? ink_r10_iq4_ld2(qs_e + sw_off) : (ink_r10_iq4_u32x2){0u, 0u};
    const f32x4 rsx_i = is_sx ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
    *(LDSP u32x4 *)(lds->u.k.x[0] + st_x) = rx0_i;
    if (ld_hi) *(LDSP u32x4 *)(lds->u.k.x[0] + st_x + 4096u) = rx1_i;
    ink_r10_iq4_stage_scales(lds->u.k.sw[0], lds->u.k.sx[0], t0, 64u, is_sw, is_sx, rsw_i, rsx_i);
  }
  lbarrier();

  u32 bf = 0u;
  u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R10_IQ4_KB) {
    const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R10_IQ4_KB));
    const int more = (kb_n < nb);
    if (more) {
      const G u8 *X_n = rq_ptr_u(X_u + (unsigned long)kb_n * 32ul);
      rx0 = *(const G u32x4 *)(X_n + xoff0);
      if (ld_hi) rx1 = *(const G u32x4 *)(X_n + xoff1);
    }
    __builtin_amdgcn_sched_barrier(0);
    const LDSP u8 *lx = lds->u.k.x[bf];
    const LDSP f16 *lsw = lds->u.k.sw[bf];
    const LDSP float *lsx = lds->u.k.sx[bf];

    ink_r10_iq4_k_wj3(wa[0], lx, lsw, lsx, 0u, wr, l16, h, lx_base0, acc);
    ink_r10_iq4_k_wj3(wa[1], lx, lsw, lsx, 1u, wr, l16, h, lx_base0, acc);
    if (more) {
      const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
      wa[0] = *(const G u32x4 *)(qw_n + w_off + 0u);
      wa[1] = *(const G u32x4 *)(qw_n + w_off + 16u);
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r10_iq4_k_wj3(wa[2], lx, lsw, lsx, 2u, wr, l16, h, lx_base0, acc);

    ink_r10_iq4_u32x2 rsw = {0u, 0u};
    f32x4 rsx = (f32x4)(0.0f);
    if (more) {
      const u32 nbf = bf ^ 1u;
      *(LDSP u32x4 *)(lds->u.k.x[nbf] + st_x) = rx0;
      if (ld_hi) *(LDSP u32x4 *)(lds->u.k.x[nbf] + st_x + 4096u) = rx1;
      const G u8 *qs_n = rq_ptr_u(qs_e + (unsigned long)kb_n * 2ul);
      const G u8 *Xs_n = rq_ptr_u(X_s + (unsigned long)kb_n * 4ul);
      if (is_sw) rsw = ink_r10_iq4_ld2(qs_n + sw_off);
      if (is_sx) rsx = rq_load_f32x4((const G float *)(Xs_n + sxoff));
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r10_iq4_k_wj3(wa[3], lx, lsw, lsx, 3u, wr, l16, h, lx_base0, acc);
    if (more) {
      const u32 nbf = bf ^ 1u;
      ink_r10_iq4_stage_scales(lds->u.k.sw[nbf], lds->u.k.sx[nbf], t0, 64u, is_sw, is_sx, rsw, rsx);
      const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
      wa[2] = *(const G u32x4 *)(qw_n + w_off + 32u);
      wa[3] = *(const G u32x4 *)(qw_n + w_off + 48u);
      lbarrier();
      bf = nbf;
    }
  }
}

static inline __attribute__((always_inline)) void ink_r10_iq4_run_wj2(
    LDSP ink_r10_iq4_lds_t *lds, const G u8 *qw_e, const G u8 *qs_e,
    const G u8 *X_u, const G u8 *X_s, const G u32 *list,
    const u32 pb, const u32 nj, const u32 rbase, const u32 rows, const u32 nb,
    const u32 xs, const u32 xdiv, f8 acc[2]) {
  u32 t0 = tid();
  __asm__ volatile("" : "+v"(t0));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = t0 & 31u, l16 = l & 15u, h = l >> 4u;
  const u32 wr = 16u * w;
  const int is_sw = (w < 4u);
  const int is_sx = (w == 0u);

  if (is_sx) {
    const u32 pj = pb + (t0 < nj ? t0 : 0u);
    lds->pid[t0] = list ? list[pj] : pj;
  }
  lbarrier();

  const u32 rw_idx = rbase + wr + l16;
  const u32 w_off = (rw_idx < rows ? rw_idx : rows - 1u) * (nb * 16u);
  const u32 sr = rbase + (is_sw ? t0 : 0u);
  const u32 sw_off = (sr < rows ? sr : rows - 1u) * (nb * 2u);
  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  const u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);
  const u32 lx_base0 = l16 * 128u + ((h ^ (l16 & 7u)) << 4u);

  const u32 xoff0 = ink_r10_iq4_div_x(lds->pid[tk_ld], xdiv) * xs + c8_ld * 16u;
  const u32 sxoff = is_sx ? ink_r10_iq4_div_x(lds->pid[t0], xdiv) * xs : 0u;

  acc[0] = (f8)(0.0f);
  acc[1] = (f8)(0.0f);

  u32x4 wa[INK_R10_IQ4_KB];
#pragma unroll
  for (u32 k = 0u; k < INK_R10_IQ4_KB; k++) wa[k] = *(const G u32x4 *)(qw_e + w_off + k * 16u);
  {
    const u32x4 rx0_i = *(const G u32x4 *)(X_u + xoff0);
    const ink_r10_iq4_u32x2 rsw_i = is_sw ? ink_r10_iq4_ld2(qs_e + sw_off) : (ink_r10_iq4_u32x2){0u, 0u};
    const f32x4 rsx_i = is_sx ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
    *(LDSP u32x4 *)(lds->u.k.x[0] + st_x) = rx0_i;
    ink_r10_iq4_stage_scales(lds->u.k.sw[0], lds->u.k.sx[0], t0, 32u, is_sw, is_sx, rsw_i, rsx_i);
  }
  lbarrier();

  u32 bf = 0u;
  u32x4 rx0 = (u32x4)(0u);
#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R10_IQ4_KB) {
    const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R10_IQ4_KB));
    const int more = (kb_n < nb);
    if (more) {
      const G u8 *X_n = rq_ptr_u(X_u + (unsigned long)kb_n * 32ul);
      rx0 = *(const G u32x4 *)(X_n + xoff0);
    }
    __builtin_amdgcn_sched_barrier(0);
    const LDSP u8 *lx = lds->u.k.x[bf];
    const LDSP f16 *lsw = lds->u.k.sw[bf];
    const LDSP float *lsx = lds->u.k.sx[bf];

    ink_r10_iq4_k_wj2(wa[0], lx, lsw, lsx, 0u, wr, l16, h, lx_base0, acc);
    ink_r10_iq4_k_wj2(wa[1], lx, lsw, lsx, 1u, wr, l16, h, lx_base0, acc);
    if (more) {
      const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
      wa[0] = *(const G u32x4 *)(qw_n + w_off + 0u);
      wa[1] = *(const G u32x4 *)(qw_n + w_off + 16u);
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r10_iq4_k_wj2(wa[2], lx, lsw, lsx, 2u, wr, l16, h, lx_base0, acc);

    ink_r10_iq4_u32x2 rsw = {0u, 0u};
    f32x4 rsx = (f32x4)(0.0f);
    if (more) {
      const u32 nbf = bf ^ 1u;
      *(LDSP u32x4 *)(lds->u.k.x[nbf] + st_x) = rx0;
      const G u8 *qs_n = rq_ptr_u(qs_e + (unsigned long)kb_n * 2ul);
      const G u8 *Xs_n = rq_ptr_u(X_s + (unsigned long)kb_n * 4ul);
      if (is_sw) rsw = ink_r10_iq4_ld2(qs_n + sw_off);
      if (is_sx) rsx = rq_load_f32x4((const G float *)(Xs_n + sxoff));
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r10_iq4_k_wj2(wa[3], lx, lsw, lsx, 3u, wr, l16, h, lx_base0, acc);
    if (more) {
      const u32 nbf = bf ^ 1u;
      ink_r10_iq4_stage_scales(lds->u.k.sw[nbf], lds->u.k.sx[nbf], t0, 32u, is_sw, is_sx, rsw, rsx);
      const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
      wa[2] = *(const G u32x4 *)(qw_n + w_off + 32u);
      wa[3] = *(const G u32x4 *)(qw_n + w_off + 48u);
      lbarrier();
      bf = nbf;
    }
  }
}

static inline __attribute__((always_inline)) void ink_r10_iq4_run_wj1(
    LDSP ink_r10_iq4_lds_t *lds, const G u8 *qw_e, const G u8 *qs_e,
    const G u8 *X_u, const G u8 *X_s, const G u32 *list,
    const u32 pb, const u32 nj, const u32 rbase, const u32 rows, const u32 nb,
    const u32 xs, const u32 xdiv, f8 *acc0) {
  u32 t0 = tid();
  __asm__ volatile("" : "+v"(t0));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = t0 & 31u, l16 = l & 15u, h = l >> 4u;
  const u32 wr = 16u * w;
  const int is_sw = (w < 4u);
  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  const int ld_lo = (tk_ld < 16u);
  const int is_sx = (t0 < 16u);

  if (is_sx) {
    const u32 pj = pb + (t0 < nj ? t0 : 0u);
    lds->pid[t0] = list ? list[pj] : pj;
  }
  lbarrier();

  const u32 rw_idx = rbase + wr + l16;
  const u32 w_off = (rw_idx < rows ? rw_idx : rows - 1u) * (nb * 16u);
  const u32 sr = rbase + (is_sw ? t0 : 0u);
  const u32 sw_off = (sr < rows ? sr : rows - 1u) * (nb * 2u);
  const u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);
  const u32 lx_base0 = l16 * 128u + ((h ^ (l16 & 7u)) << 4u);

  const u32 xoff0 = ld_lo ? (ink_r10_iq4_div_x(lds->pid[tk_ld], xdiv) * xs + c8_ld * 16u) : 0u;
  const u32 sxoff = is_sx ? ink_r10_iq4_div_x(lds->pid[t0], xdiv) * xs : 0u;

  *acc0 = (f8)(0.0f);

  u32x4 wa[INK_R10_IQ4_KB];
#pragma unroll
  for (u32 k = 0u; k < INK_R10_IQ4_KB; k++) wa[k] = *(const G u32x4 *)(qw_e + w_off + k * 16u);
  {
    const u32x4 rx0_i = ld_lo ? *(const G u32x4 *)(X_u + xoff0) : (u32x4)(0u);
    const ink_r10_iq4_u32x2 rsw_i = is_sw ? ink_r10_iq4_ld2(qs_e + sw_off) : (ink_r10_iq4_u32x2){0u, 0u};
    const f32x4 rsx_i = is_sx ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
    if (ld_lo) *(LDSP u32x4 *)(lds->u.k.x[0] + st_x) = rx0_i;
    ink_r10_iq4_stage_scales(lds->u.k.sw[0], lds->u.k.sx[0], t0, 32u, is_sw, is_sx, rsw_i, rsx_i);
  }
  lbarrier();

  u32 bf = 0u;
  u32x4 rx0 = (u32x4)(0u);
#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R10_IQ4_KB) {
    const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R10_IQ4_KB));
    const int more = (kb_n < nb);
    if (more && ld_lo) {
      const G u8 *X_n = rq_ptr_u(X_u + (unsigned long)kb_n * 32ul);
      rx0 = *(const G u32x4 *)(X_n + xoff0);
    }
    __builtin_amdgcn_sched_barrier(0);
    const LDSP u8 *lx = lds->u.k.x[bf];
    const LDSP f16 *lsw = lds->u.k.sw[bf];
    const LDSP float *lsx = lds->u.k.sx[bf];

    ink_r10_iq4_k_wj1(wa[0], lx, lsw, lsx, 0u, wr, l16, h, lx_base0, acc0);
    ink_r10_iq4_k_wj1(wa[1], lx, lsw, lsx, 1u, wr, l16, h, lx_base0, acc0);
    if (more) {
      const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
      wa[0] = *(const G u32x4 *)(qw_n + w_off + 0u);
      wa[1] = *(const G u32x4 *)(qw_n + w_off + 16u);
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r10_iq4_k_wj1(wa[2], lx, lsw, lsx, 2u, wr, l16, h, lx_base0, acc0);

    ink_r10_iq4_u32x2 rsw = {0u, 0u};
    f32x4 rsx = (f32x4)(0.0f);
    if (more) {
      const u32 nbf = bf ^ 1u;
      if (ld_lo) *(LDSP u32x4 *)(lds->u.k.x[nbf] + st_x) = rx0;
      const G u8 *qs_n = rq_ptr_u(qs_e + (unsigned long)kb_n * 2ul);
      const G u8 *Xs_n = rq_ptr_u(X_s + (unsigned long)kb_n * 4ul);
      if (is_sw) rsw = ink_r10_iq4_ld2(qs_n + sw_off);
      if (is_sx) rsx = rq_load_f32x4((const G float *)(Xs_n + sxoff));
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r10_iq4_k_wj1(wa[3], lx, lsw, lsx, 3u, wr, l16, h, lx_base0, acc0);
    if (more) {
      const u32 nbf = bf ^ 1u;
      ink_r10_iq4_stage_scales(lds->u.k.sw[nbf], lds->u.k.sx[nbf], t0, 32u, is_sw, is_sx, rsw, rsx);
      const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
      wa[2] = *(const G u32x4 *)(qw_n + w_off + 32u);
      wa[3] = *(const G u32x4 *)(qw_n + w_off + 48u);
      lbarrier();
      bf = nbf;
    }
  }
}

/* LDS-transposed coalesced staggered FP32 atomic scatter-add strip (<= 32 tokens).
 * Re-derives all lane offsets from pinned tid() so ZERO epilogue VGPRs span the K-loop! */
static inline __attribute__((always_inline)) void ink_r10_iq4_scatter_strip(
    LDSP float *ep, const LDSP u32 *lpid, G float *ysum,
    const G u32 *route, const u32 rts, const u32 K, const u32 e,
    const u32 rbase, const u32 rows, const u32 os,
    const u32 strip_base, const u32 nj_strip, const u32 j_base, const f8 *acc) {
  u32 t_ep = tid();
  __asm__ volatile("" : "+v"(t_ep));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t_ep >> 5u));
  const u32 l = t_ep & 31u, l16 = l & 15u, h = l >> 4u;
  const u32 r_wr = 16u * w + 8u * h;

#pragma unroll
  for (u32 j_loc = 0u; j_loc < 2u; j_loc++) {
    const u32 slot_loc = 16u * j_loc + l16;
    if (slot_loc < nj_strip) {
      LDSP float *dst = ep + slot_loc * INK_R10_IQ4_EP_STR + r_wr;
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) dst[v] = acc[j_base + j_loc][v];
    }
  }
  lbarrier();

  /* Bijective token stagger by (e & 7) + 32-row cache-line stagger by (e & 3). */
  u32 stag = e & 7u;
  if (stag >= nj_strip) stag = 0u;
  const u32 c0 = e & 3u;

  for (u32 tk_i = w; tk_i < nj_strip; tk_i += 8u) {
    u32 tk = tk_i + stag;
    if (tk >= nj_strip) tk -= nj_strip;
    const u32 pid = (u32)__builtin_amdgcn_readfirstlane((int)lpid[strip_base + tk]);
    const u32 tok = (K == 8u) ? (pid >> 3u) : (pid / K);
    const u32 slot_k = (K == 8u) ? (pid & 7u) : (pid - tok * K);
    const G float *rw_p = (const G float *)((const G u8 *)route + (unsigned long)tok * rts) + K;
    const float rw_val = rw_p[slot_k];
    G float *ytok = (G float *)((G u8 *)ysum + (unsigned long)tok * os);
    const LDSP float *src = ep + tk * INK_R10_IQ4_EP_STR;

#pragma unroll
    for (u32 ci = 0u; ci < 4u; ci++) {
      const u32 c = (ci + c0) & 3u;
      const u32 r_loc = (c << 5u) + l;
      const u32 r = rbase + r_loc;
      if (r < rows) {
        const float val = src[r_loc] * rw_val;
        __builtin_amdgcn_global_atomic_fadd_f32(ytok + r, val);
      }
    }
  }
}

/* LDS-transposed coalesced 512-B global_store_b128 epilogue strip for non-atomic expmaj_r10. */
static inline __attribute__((always_inline)) void ink_r10_iq4_store_strip(
    LDSP float *ep, const LDSP u32 *lpid, G u8 *Y, const G u8 *res,
    const u32 rbase, const u32 rows, const u32 ys, const u32 rs,
    const u32 strip_base, const u32 nj_strip, const u32 j_base, const f8 *acc) {
  u32 t_ep = tid();
  __asm__ volatile("" : "+v"(t_ep));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t_ep >> 5u));
  const u32 l = t_ep & 31u, l16 = l & 15u, h = l >> 4u;
  const u32 r_wr = 16u * w + 8u * h;

#pragma unroll
  for (u32 j_loc = 0u; j_loc < 2u; j_loc++) {
    const u32 slot_loc = 16u * j_loc + l16;
    if (slot_loc < nj_strip) {
      LDSP float *dst = ep + slot_loc * INK_R10_IQ4_EP_STR + r_wr;
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) dst[v] = acc[j_base + j_loc][v];
    }
  }
  lbarrier();

  const int vec_ok = (rbase + 128u <= rows) && ((ys & 15u) == 0u) && ((((unsigned long)Y) & 15ul) == 0ul) &&
                     (!res || (((rs & 15u) == 0u) && ((((unsigned long)res) & 15ul) == 0ul)));
  if (__builtin_expect(vec_ok, 1)) {
    for (u32 tk = w; tk < nj_strip; tk += 8u) {
      const u32 pid = (u32)__builtin_amdgcn_readfirstlane((int)lpid[strip_base + tk]);
      const LDSP float *src = ep + tk * INK_R10_IQ4_EP_STR + 4u * l;
      f32x4 v = {src[0], src[1], src[2], src[3]};
      if (res) {
        const G f32x4 *rr4 = (const G f32x4 *)((const G float *)(res + (unsigned long)pid * rs) + rbase);
        v += rr4[l];
      }
      G f32x4 *yr4 = (G f32x4 *)((G float *)(Y + (unsigned long)pid * ys) + rbase);
      yr4[l] = v;
    }
  } else {
    for (u32 tk = w; tk < nj_strip; tk += 8u) {
      const u32 pid = (u32)__builtin_amdgcn_readfirstlane((int)lpid[strip_base + tk]);
      const LDSP float *src = ep + tk * INK_R10_IQ4_EP_STR + 4u * l;
      G float *yr = (G float *)(Y + (unsigned long)pid * ys);
      const G float *rr = res ? (const G float *)(res + (unsigned long)pid * rs) : (const G float *)0;
#pragma unroll 1
      for (u32 c = 0u; c < 4u; c++) {
        const u32 r = rbase + 4u * l + c;
        if (r < rows) yr[r] = epilogue(src[c], r, 0, rr);
      }
    }
  }
}

/* ============================================================================
 * Kernel 1: ie_inkling_gemm_iq4nl_scatter_r10
 *   Expert-major L2-localized 3-WG/CU IQ4_NL down-proj + fused coalesced
 *   staggered FP32 atomic weighted scatter-accumulate into pre-zeroed ysum!
 * ========================================================================== */
KERNEL ie_inkling_gemm_iq4nl_scatter_r10(
    const G u32 *qw0, const G u16 *qs0, const G u8 *X, G float *ysum,
    u32 rows, u32 nb, const G u32 *list, const G u32 *offs,
    const G u32 *route, u32 rts, u32 K,
    u32 qwn, u32 qsn, u32 xs, u32 os, u32 xdiv) {
  static LDS ink_r10_iq4_lds_t ink_r10_iq4_sc_lds;
  if (!qw0 || !qs0 || !X || !ysum || !list || !offs || !route ||
      rows == 0u || nb == 0u || (nb & 3u) != 0u || xdiv == 0u || K == 0u)
    return;

  const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)offs[e]);
  const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)offs[e + 1u]);
  const u32 rbase = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() * INK_R10_IQ4_R));
  if (p0 >= p1 || rbase >= rows) return;

  const G u8 *qw_e = rq_ptr_u((const G u8 *)(qw0 + (unsigned long)e * qwn));
  const G u8 *qs_e = rq_ptr_u((const G u8 *)(qs0 + (unsigned long)e * qsn));
  const G u8 *X_u = rq_ptr_u(X);
  const G u8 *X_s = rq_ptr_u(X + 32ul * nb);

  u32 pb = p0;
  /* Phase 1: Expert-major 64-pair chunks (WJ=4), reusing hot weights in L0/L1/L2! */
#pragma unroll 1
  while (pb < p1 && p1 - pb > 48u) {
    const u32 pb_u = (u32)__builtin_amdgcn_readfirstlane((int)pb);
    const u32 rem = p1 - pb_u;
    const u32 nj = rem < 64u ? rem : 64u;
    f8 acc[4];
    ink_r10_iq4_run_wj4(&ink_r10_iq4_sc_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                        rbase, rows, nb, xs, xdiv, acc);
    lbarrier();
    ink_r10_iq4_scatter_strip(&ink_r10_iq4_sc_lds.u.ep[0][0], ink_r10_iq4_sc_lds.pid, ysum,
                              route, rts, K, e, rbase, rows, os, 0u, 32u, 0u, acc);
    lbarrier();
    ink_r10_iq4_scatter_strip(&ink_r10_iq4_sc_lds.u.ep[0][0], ink_r10_iq4_sc_lds.pid, ysum,
                              route, rts, K, e, rbase, rows, os, 32u, nj - 32u, 2u, acc);
    pb = pb_u + nj;
    if (pb < p1) lbarrier();
  }

  /* Phase 2: Single-pass adaptive tail (rem in 1..48 -> WJ = 3/2/1, zero back-edge!). */
  if (pb < p1) {
    const u32 pb_u = (u32)__builtin_amdgcn_readfirstlane((int)pb);
    const u32 nj = p1 - pb_u;
    f8 acc[3];
    if (nj > 32u) {
      ink_r10_iq4_run_wj3(&ink_r10_iq4_sc_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                          rbase, rows, nb, xs, xdiv, acc);
      lbarrier();
      ink_r10_iq4_scatter_strip(&ink_r10_iq4_sc_lds.u.ep[0][0], ink_r10_iq4_sc_lds.pid, ysum,
                                route, rts, K, e, rbase, rows, os, 0u, 32u, 0u, acc);
      lbarrier();
      ink_r10_iq4_scatter_strip(&ink_r10_iq4_sc_lds.u.ep[0][0], ink_r10_iq4_sc_lds.pid, ysum,
                                route, rts, K, e, rbase, rows, os, 32u, nj - 32u, 2u, acc);
    } else if (nj > 16u) {
      ink_r10_iq4_run_wj2(&ink_r10_iq4_sc_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                          rbase, rows, nb, xs, xdiv, acc);
      lbarrier();
      ink_r10_iq4_scatter_strip(&ink_r10_iq4_sc_lds.u.ep[0][0], ink_r10_iq4_sc_lds.pid, ysum,
                                route, rts, K, e, rbase, rows, os, 0u, nj, 0u, acc);
    } else {
      ink_r10_iq4_run_wj1(&ink_r10_iq4_sc_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                          rbase, rows, nb, xs, xdiv, &acc[0]);
      lbarrier();
      ink_r10_iq4_scatter_strip(&ink_r10_iq4_sc_lds.u.ep[0][0], ink_r10_iq4_sc_lds.pid, ysum,
                                route, rts, K, e, rbase, rows, os, 0u, nj, 0u, acc);
    }
  }
}

/* ============================================================================
 * Kernel 2: ie_inkling_gemm_iq4nl_expmaj_r10
 *   Non-atomic exact-SHA 3-WG/CU expert-major IQ4_NL prefill down-proj with
 *   coalesced 512-B b128 stores (drop-in for ie_inkling_gemm_iq4nl_exp_r2!).
 * ========================================================================== */
KERNEL ie_inkling_gemm_iq4nl_expmaj_r10(
    const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
    u32 rows, u32 nb, const G u32 *list, const G u32 *offs, u32 np,
    u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv, u32 kind,
    const G u8 *res, u32 rs) {
  static LDS ink_r10_iq4_lds_t ink_r10_iq4_em_lds;
  (void)kind;
  if (!qw0 || !qs0 || !X || !Y || rows == 0u || nb == 0u || (nb & 3u) != 0u || xdiv == 0u) return;

  const u32 e = list ? (u32)__builtin_amdgcn_readfirstlane((int)tokid()) : 0u;
  const u32 p0 = list ? (u32)__builtin_amdgcn_readfirstlane((int)offs[e])
                      : (u32)__builtin_amdgcn_readfirstlane((int)(tokid() * 64u));
  const u32 p1 = list ? (u32)__builtin_amdgcn_readfirstlane((int)offs[e + 1u])
                      : (p0 + 64u < np ? p0 + 64u : np);
  const u32 rbase = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() * INK_R10_IQ4_R));
  if (p0 >= p1 || rbase >= rows) return;

  const G u8 *qw_e = rq_ptr_u((const G u8 *)(qw0 + (unsigned long)e * qwn));
  const G u8 *qs_e = rq_ptr_u((const G u8 *)(qs0 + (unsigned long)e * qsn));
  const G u8 *X_u = rq_ptr_u(X);
  const G u8 *X_s = rq_ptr_u(X + 32ul * nb);

  u32 pb = p0;
#pragma unroll 1
  while (pb < p1 && p1 - pb > 48u) {
    const u32 pb_u = (u32)__builtin_amdgcn_readfirstlane((int)pb);
    const u32 rem = p1 - pb_u;
    const u32 nj = rem < 64u ? rem : 64u;
    f8 acc[4];
    ink_r10_iq4_run_wj4(&ink_r10_iq4_em_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                        rbase, rows, nb, xs, xdiv, acc);
    lbarrier();
    ink_r10_iq4_store_strip(&ink_r10_iq4_em_lds.u.ep[0][0], ink_r10_iq4_em_lds.pid, Y, res,
                            rbase, rows, ys, rs, 0u, 32u, 0u, acc);
    lbarrier();
    ink_r10_iq4_store_strip(&ink_r10_iq4_em_lds.u.ep[0][0], ink_r10_iq4_em_lds.pid, Y, res,
                            rbase, rows, ys, rs, 32u, nj - 32u, 2u, acc);
    pb = pb_u + nj;
    if (pb < p1) lbarrier();
  }

  if (pb < p1) {
    const u32 pb_u = (u32)__builtin_amdgcn_readfirstlane((int)pb);
    const u32 nj = p1 - pb_u;
    f8 acc[3];
    if (nj > 32u) {
      ink_r10_iq4_run_wj3(&ink_r10_iq4_em_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                          rbase, rows, nb, xs, xdiv, acc);
      lbarrier();
      ink_r10_iq4_store_strip(&ink_r10_iq4_em_lds.u.ep[0][0], ink_r10_iq4_em_lds.pid, Y, res,
                              rbase, rows, ys, rs, 0u, 32u, 0u, acc);
      lbarrier();
      ink_r10_iq4_store_strip(&ink_r10_iq4_em_lds.u.ep[0][0], ink_r10_iq4_em_lds.pid, Y, res,
                              rbase, rows, ys, rs, 32u, nj - 32u, 2u, acc);
    } else if (nj > 16u) {
      ink_r10_iq4_run_wj2(&ink_r10_iq4_em_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                          rbase, rows, nb, xs, xdiv, acc);
      lbarrier();
      ink_r10_iq4_store_strip(&ink_r10_iq4_em_lds.u.ep[0][0], ink_r10_iq4_em_lds.pid, Y, res,
                              rbase, rows, ys, rs, 0u, nj, 0u, acc);
    } else {
      ink_r10_iq4_run_wj1(&ink_r10_iq4_em_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                          rbase, rows, nb, xs, xdiv, &acc[0]);
      lbarrier();
      ink_r10_iq4_store_strip(&ink_r10_iq4_em_lds.u.ep[0][0], ink_r10_iq4_em_lds.pid, Y, res,
                              rbase, rows, ys, rs, 0u, nj, 0u, acc);
    }
  }
}

#undef INK_R10_IQ4_EP_STR
#undef INK_R10_IQ4_SW_STR
#undef INK_R10_IQ4_R
#undef INK_R10_IQ4_KB

#endif /* !IE_MEGA */
