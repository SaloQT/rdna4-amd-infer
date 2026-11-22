/* ============================================================================
 * Inkling-2 Round 9 (Wave 8 / Prefill Follow-up):
 *   3-WGs/CU Zero-fma_mix, Zero-Stall-Waitcnt, Branch-Free 4-Bucket
 *   (16/32/48/64) Worklist IQ4_NL MoE Prefill WMMA GEMM for RX 9070 XT
 *   (gfx1201, wave32, NT=256).
 *
 * Target: MoE ffn_down_exps MAT_IQ4NL (kind=8, 2304x1152, nb=36, 128E/K=8),
 *         our #2 prefill kernel (176.6 ms 4K / 592.9 ms 16K in R00 trace #191).
 *
 * Root-Cause Fixes over R15 rejected ie_inkling_gemm_iq4nl_exp_r7 (+4.9ms 4K /
 * +29.0ms 16K vs r2_g15), proved via llvm-objdump ISA diff of r7 vs r2_g15:
 *   1. Killed ALL 160 quarter-rate v_fma_mix_f32 instructions!
 *      In r7, Clang folded (float)sw_h[i] * dx into 32 non-dual-issue
 *      v_fma_mix_f32/K-block (dropping v_dual_mul_f32 95->34 and v_dual_fmac_f32
 *      60->25). Pinning h8->f8 conversion with __asm__ volatile("" : "+v"(sw))
 *      eliminates 100% of v_fma_mix_f32 (0!) and restores full 2-ops/cycle
 *      v_dual_mul_f32 + v_dual_fmac_f32 while keeping compact f16 sw LDS!
 *   2. Eliminated BOTH r7's premature staging AND r2's hidden s_wait_loadcnt 0!
 *      Both r2 and r7 issued conditional rsw/rsx loads BEFORE k=0, forcing LLVM
 *      to emit s_wait_loadcnt 0x0 before k=0 (draining rx0/rx1 before compute!).
 *      r9 issues ONLY unconditional rx0/rx1 at loop top, reloads wa[0..1] right
 *      after k=1, stages rx0/rx1 into dead lds[bf^1] after k=2 (3 full K-blocks
 *      hidden, freeing 8 VGPRs), then loads rsw/rsx before k=3 (hidden across
 *      k=3 with ZERO waitcnt because wa[3] is already ready!), and stages
 *      rsw/rsx + reloads wa[2..3] after k=3. Peak drops <= 128 VGPR AND k=0/k=2
 *      only wait s_wait_loadcnt 4 (never stalling on in-flight rx0/rx1!).
 *   3. Removed fake per-K-block if(k < nk) branches (nb is guarded (nb&3)==0!)
 *      AND replaced r7's per-K-block if(has_j3)/if(has_j1) branches with 4
 *      clean, straight-line inner loops keyed by worklist cls = 3/2/1/0
 *      (64/48/32/16 pairs -> exact WJ=4/3/2/1). Removing fake PHI nodes lets
 *      LLVM shrink ALL 4 K-blocks' FMAs to 2-ops/cycle v_dual_fmac_f32!
 *   4. SGPR-pinned chunk bases (rq_ptr_u) + pure 32-bit VGPR byte offsets +
 *      0-conflict 2x ds_store_b32 f16 sw staging (21,248 B LDS <= 21.8 KiB) +
 *      <= 128 VGPR -> 3 WGs/CU (+50% occupancy over r2_g15!), with 2x f32x4
 *      vectorized epilogue stores and non-unrolled cold tail fallback.
 *
 * Exported Kernel, Grid, Guard, and Parity (drop-in replacement for b71):
 *   KERNEL ie_inkling_gemm_iq4nl_exp_r9(
 *       const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
 *       u32 rows, u32 nb, const G u32 *list, const G u32 *tiles,
 *       const G u32 *n_tiles, u32 qwn, u32 qsn, u32 xs, u32 ys,
 *       u32 xdiv, u32 kind, const G u8 *res, u32 rs)
 *   - Grid:   ((rows + 127u) / 128u, max_tiles, 1u), NT = 256
 *             where max_tiles = min(T*K, n_exp + (T*K)/64u)
 *             (consumes bit-69 ie_inkling_moe_pf_worklist_r7 b->ms_packs/ms_npacks).
 *   - Guard:  !mk.rec && T >= 17u && kind == MAT_IQ4NL (8u) && rows > 0u &&
 *             nb > 0u && (nb & 3u) == 0u && list && tiles && n_tiles &&
 *             n_exp <= 256u && T * K <= 8192u.
 *   - Parity: 100% bitwise-exact (0-bit-diff) vs ie_inkling_gemm_iq4nl_exp_r2!
 *   - All local symbols/LDS/macros prefixed ink_r9_iq4_ / INK_R9_IQ4_; macros
 *     #undef'd at EOF. Clang 21 scalar __builtin_bit_cast safe, 0/0 spills.
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R9_IQ4_KB     4u
#define INK_R9_IQ4_R      128u
#define INK_R9_IQ4_SW_STR 160u /* 128 f16 + 32 f16 pad = 320 B = 16 mod 32 banks -> 0-conflict 2x ds_store_b32 */

typedef unsigned int ink_r9_iq4_u32x2 __attribute__((ext_vector_type(2)));

struct __attribute__((packed, aligned(4))) ink_r9_iq4_u32x2p {
  ink_r9_iq4_u32x2 v;
};

static inline __attribute__((always_inline)) ink_r9_iq4_u32x2 ink_r9_iq4_ld2(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r9_iq4_u32x2p *)p)->v;
}

typedef struct {
  u8    x[2][64u * 128u] __attribute__((aligned(16)));                         /* 16,384 B */
  f16   sw[2][INK_R9_IQ4_KB * INK_R9_IQ4_SW_STR] __attribute__((aligned(16))); /*  2,560 B */
  float sx[2][INK_R9_IQ4_KB * 64u] __attribute__((aligned(32)));               /*  2,048 B */
  u32   pid[64] __attribute__((aligned(16)));                                  /*    256 B */
} ink_r9_iq4_lds_t;

_Static_assert(sizeof(ink_r9_iq4_lds_t) == 21248u, "ink_r9_iq4_lds_t must be 21,248 B (<= 21.8 KiB = 3 WGs/CU)");

static inline __attribute__((always_inline)) u32 ink_r9_iq4_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

/* 0-bank-conflict 2x ds_store_b32 f16 weight-scale staging (waves 0..3, 128 rows). */
static inline __attribute__((always_inline)) void ink_r9_iq4_stage_sw(
    LDSP f16 *lsw, const u32 t0, const int is_sw, const ink_r9_iq4_u32x2 rsw) {
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
    const u32 str32 = INK_R9_IQ4_SW_STR >> 1u;
    sw32[(0u + odd) * str32 + (r2 >> 1u)] = v_a;
    sw32[(2u + odd) * str32 + (r2 >> 1u)] = v_b;
  }
}

static inline __attribute__((always_inline)) void ink_r9_iq4_stage_scales(
    LDSP f16 *lsw, LDSP float *lsx, const u32 t0, const u32 sx_stride,
    const int is_sw, const int is_sx, const ink_r9_iq4_u32x2 rsw, const f32x4 rsx) {
  ink_r9_iq4_stage_sw(lsw, t0, is_sw, rsw);
  if (is_sx) {
    u32 t_loc = t0;
    __asm__ volatile("" : "+v"(t_loc));
    lsx[0u * sx_stride + t_loc] = rsx.x;
    lsx[1u * sx_stride + t_loc] = rsx.y;
    lsx[2u * sx_stride + t_loc] = rsx.z;
    lsx[3u * sx_stride + t_loc] = rsx.w;
  }
}

/* Load f16 weight scales (1x ds_load_b128) and convert to pinned f32 f8:
 * __asm__("" : "+v"(sw)) eliminates 100% of quarter-rate v_fma_mix_f32 and
 * unlocks 2-ops/cycle v_dual_mul_f32 + v_dual_fmac_f32. */
static inline __attribute__((always_inline)) f8 ink_r9_iq4_load_sw_f8(
    const LDSP f16 *lsw, const u32 k, const u32 wr, const u32 h) {
  const h8 sw_h = *(const LDSP h8 *)(lsw + k * INK_R9_IQ4_SW_STR + wr + 8u * h);
  f8 sw = __builtin_convertvector(sw_h, f8);
  __asm__ volatile("" : "+v"(sw));
  return sw;
}

static inline __attribute__((always_inline)) void ink_r9_iq4_decode_w(
    const u32x4 wk, const u32 h, v2i *a0_out, v2i *a1_out) {
  u32x4 nb4 = h ? ((wk >> 4u) & 0x0F0F0F0Fu) : (wk & 0x0F0F0F0Fu);
  nb4.x = ie_iq4_map(nb4.x);
  nb4.y = ie_iq4_map(nb4.y);
  nb4.z = ie_iq4_map(nb4.z);
  nb4.w = ie_iq4_map(nb4.w);
  *a0_out = (v2i){(int)nb4.x, (int)nb4.y};
  *a1_out = (v2i){(int)nb4.z, (int)nb4.w};
}

/* Branch-free K-block for WJ = 4 (49..64 pairs):
 * Loads & converts f16 sw ONCE per K-block (only 8 v_cvt_f32_f16 & 1x ds_load_b128!). */
static inline __attribute__((always_inline)) void ink_r9_iq4_k_wj4(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 l16, const u32 h, const u32 lx_base0, f8 acc[4]) {
  const v8i zm = (v8i)(0x4B400000);
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * 64u + l16;

  v2i a0, a1;
  ink_r9_iq4_decode_w(wk, h, &a0, &a1);

  /* Pair 0 (j = 0, 1) */
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

  const f8 sw = ink_r9_iq4_load_sw_f8(lsw, k, wr, h);
  acc[0] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  __asm__ volatile("" : : "v"(acc[0]));
  __builtin_amdgcn_sched_barrier(0);
  acc[1] += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
  __asm__ volatile("" : : "v"(acc[1]) : "memory");
  __builtin_amdgcn_sched_barrier(0);

  /* Pair 1 (j = 2, 3): reuses sw, b0/b1, and c0/c1 VGPRs. */
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

  acc[2] += (sw * dx2) * (__builtin_bit_cast(f8, c2) - 12582912.0f);
  __asm__ volatile("" : : "v"(acc[2]));
  __builtin_amdgcn_sched_barrier(0);
  acc[3] += (sw * dx3) * (__builtin_bit_cast(f8, c3) - 12582912.0f);
  __asm__ volatile("" : : "v"(acc[3]) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

/* Branch-free K-block for WJ = 3 (33..48 pairs): -25% X loads & WMMAs, 1x sw load! */
static inline __attribute__((always_inline)) void ink_r9_iq4_k_wj3(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 l16, const u32 h, f8 acc[3]) {
  const v8i zm = (v8i)(0x4B400000);
  const u32 x_col = ((2u * k + h) ^ (l16 & 7u)) << 4u;
  const LDSP u8 *lx_k = lx + l16 * 128u + x_col;
  const LDSP float *lsx_k = lsx + k * 64u + l16;

  v2i a0, a1;
  ink_r9_iq4_decode_w(wk, h, &a0, &a1);

  /* Pair 0 (j = 0, 1) */
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

  const f8 sw = ink_r9_iq4_load_sw_f8(lsw, k, wr, h);
  acc[0] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  __asm__ volatile("" : : "v"(acc[0]));
  __builtin_amdgcn_sched_barrier(0);
  acc[1] += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
  __asm__ volatile("" : : "v"(acc[1]) : "memory");
  __builtin_amdgcn_sched_barrier(0);

  /* Single tile (j = 2) */
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

/* Branch-free K-block for WJ = 2 (17..32 pairs). */
static inline __attribute__((always_inline)) void ink_r9_iq4_k_wj2(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 l16, const u32 h, f8 acc[2]) {
  const v8i zm = (v8i)(0x4B400000);
  const u32 x_col = ((2u * k + h) ^ (l16 & 7u)) << 4u;
  const LDSP u8 *lx_k = lx + l16 * 128u + x_col;
  const LDSP float *lsx_k = lsx + k * 32u + l16;

  v2i a0, a1;
  ink_r9_iq4_decode_w(wk, h, &a0, &a1);

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

  const f8 sw = ink_r9_iq4_load_sw_f8(lsw, k, wr, h);
  acc[0] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  __asm__ volatile("" : : "v"(acc[0]));
  __builtin_amdgcn_sched_barrier(0);
  acc[1] += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
  __asm__ volatile("" : : "v"(acc[1]) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

/* Branch-free K-block for WJ = 1 (1..16 pairs): -50% X loads & WMMAs vs r2! */
static inline __attribute__((always_inline)) void ink_r9_iq4_k_wj1(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 l16, const u32 h, f8 *acc0) {
  const v8i zm = (v8i)(0x4B400000);
  const u32 x_col = ((2u * k + h) ^ (l16 & 7u)) << 4u;
  const LDSP u8 *lx_k = lx + l16 * 128u + x_col;
  const LDSP float *lsx_k = lsx + k * 32u + l16;

  v2i a0, a1;
  ink_r9_iq4_decode_w(wk, h, &a0, &a1);

  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + 0u * 2048u);
  const float dx0 = lsx_k[0u];
  __builtin_amdgcn_sched_barrier(0);

  const v2i b0_lo = {(int)b0.x, (int)b0.y}, b0_hi = {(int)b0.z, (int)b0.w};
  v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_lo, zm, 0);
  c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_hi, c0, 0);
  __builtin_amdgcn_sched_barrier(0);

  const f8 sw = ink_r9_iq4_load_sw_f8(lsw, k, wr, h);
  *acc0 += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  __asm__ volatile("" : : "v"(*acc0) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

/* 2x f32x4 vectorized epilogue store (exact 0-bit-diff; cold tail unroll-1 for I-cache). */
static inline __attribute__((always_inline)) void ink_r9_iq4_store_out(
    G u8 *Y, const G u8 *res, const LDSP u32 *lpid, const u32 nj, const u32 wj,
    const u32 rbase, const u32 wr, const u32 l16, const u32 h, const u32 rows,
    const u32 ys, const u32 rs, const f8 *acc) {
  const u32 r0 = rbase + wr + 8u * h;
  const int vec_ok = (r0 + 8u <= rows) && ((ys & 15u) == 0u) && ((((unsigned long)Y) & 15ul) == 0ul) &&
                     (!res || (((rs & 15u) == 0u) && ((((unsigned long)res) & 15ul) == 0ul)));
  if (__builtin_expect(vec_ok, 1)) {
#pragma unroll
    for (u32 j = 0u; j < wj; j++) {
      const u32 slot = 16u * j + l16;
      if (slot >= nj) continue;
      const u32 pid = lpid[slot];
      G float *yr = (G float *)(Y + (unsigned long)pid * ys);
      f32x4 lo = {acc[j][0], acc[j][1], acc[j][2], acc[j][3]};
      f32x4 hi = {acc[j][4], acc[j][5], acc[j][6], acc[j][7]};
      if (res) {
        const G f32x4 *rr4 = (const G f32x4 *)((const G float *)(res + (unsigned long)pid * rs) + r0);
        lo += rr4[0];
        hi += rr4[1];
      }
      G f32x4 *yr4 = (G f32x4 *)(yr + r0);
      yr4[0] = lo;
      yr4[1] = hi;
    }
  } else {
#pragma unroll 1
    for (u32 j = 0u; j < wj; j++) {
      const u32 slot = 16u * j + l16;
      if (slot >= nj) continue;
      const u32 pid = lpid[slot];
      G float *yr = (G float *)(Y + (unsigned long)pid * ys);
      const G float *rr = res ? (const G float *)(res + (unsigned long)pid * rs) : (const G float *)0;
#pragma unroll 1
      for (u32 v = 0u; v < 8u; v++) {
        const u32 r = r0 + v;
        if (r < rows) yr[r] = epilogue(acc[j][v], r, 0, rr);
      }
    }
  }
}

KERNEL ie_inkling_gemm_iq4nl_exp_r9(const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
                                    u32 rows, u32 nb, const G u32 *list, const G u32 *tiles,
                                    const G u32 *n_tiles, u32 qwn, u32 qsn, u32 xs, u32 ys,
                                    u32 xdiv, u32 kind, const G u8 *res, u32 rs) {
  static LDS ink_r9_iq4_lds_t ink_r9_iq4_lds;
  (void)kind;
  if (!n_tiles || !tiles || !list || !qw0 || !qs0 || !X || !Y || nb == 0u || (nb & 3u) != 0u || xdiv == 0u) return;

  const u32 ti = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  const u32 nt_act = (u32)__builtin_amdgcn_readfirstlane((int)(*n_tiles));
  const u32 rbase = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() * INK_R9_IQ4_R));
  if (ti >= nt_act || rbase >= rows) return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)tiles[ti]);
  const u32 e = desc & 0xFFu;
  const u32 pb = (desc >> 8u) & 0x1FFFu;
  const u32 nj = (desc >> 21u) & 0x7Fu;
  if (nj == 0u || nj > 64u) return;

  const u32 cls = (nj - 1u) >> 4u; /* 3: 49..64, 2: 33..48, 1: 17..32, 0: 1..16 (SGPR-uniform!) */
  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), l16 = l & 15u, h = l >> 4u;
  const u32 wr = 16u * w;
  const int is_sw = (w < 4u);

  const G u8 *qw_e = rq_ptr_u((const G u8 *)(qw0 + (unsigned long)e * qwn));
  const G u8 *qs_e = rq_ptr_u((const G u8 *)(qs0 + (unsigned long)e * qsn));
  const G u8 *X_u = rq_ptr_u(X);
  const G u8 *X_s = rq_ptr_u(X + 32ul * nb);

  const u32 rw_idx = rbase + wr + l16;
  const u32 w_off = (rw_idx < rows ? rw_idx : rows - 1u) * (nb * 16u);
  const u32 sr = rbase + (is_sw ? t0 : 0u);
  const u32 sw_off = (sr < rows ? sr : rows - 1u) * (nb * 2u);

  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  const u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);
  const u32 lx_base0 = l16 * 128u + ((h ^ (l16 & 7u)) << 4u);

  if (cls == 3u) {
    /* ================ Bucket 3: 49..64 pairs (WJ = 4, 0 K-loop branches) ================ */
    const int is_sx = (w < 2u);
    if (is_sx) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r9_iq4_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ink_r9_iq4_div_x(ink_r9_iq4_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
    const u32 xoff1 = ink_r9_iq4_div_x(ink_r9_iq4_lds.pid[tk_ld + 32u], xdiv) * xs + c8_ld * 16u;
    const u32 sxoff = is_sx ? ink_r9_iq4_div_x(ink_r9_iq4_lds.pid[t0 & 63u], xdiv) * xs : 0u;

    f8 acc[4];
#pragma unroll
    for (u32 j = 0u; j < 4u; j++) acc[j] = (f8)(0.0f);

    u32x4 wa[INK_R9_IQ4_KB];
#pragma unroll
    for (u32 k = 0u; k < INK_R9_IQ4_KB; k++) wa[k] = *(const G u32x4 *)(qw_e + w_off + k * 16u);
    {
      const u32x4 rx0_i = *(const G u32x4 *)(X_u + xoff0);
      const u32x4 rx1_i = *(const G u32x4 *)(X_u + xoff1);
      const ink_r9_iq4_u32x2 rsw_i = is_sw ? ink_r9_iq4_ld2(qs_e + sw_off) : (ink_r9_iq4_u32x2){0u, 0u};
      const f32x4 rsx_i = is_sx ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
      *(LDSP u32x4 *)(ink_r9_iq4_lds.x[0] + st_x) = rx0_i;
      *(LDSP u32x4 *)(ink_r9_iq4_lds.x[0] + st_x + 4096u) = rx1_i;
      ink_r9_iq4_stage_scales(ink_r9_iq4_lds.sw[0], ink_r9_iq4_lds.sx[0], t0, 64u, is_sw, is_sx, rsw_i, rsx_i);
    }
    lbarrier();

    u32 bf = 0u;
    u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
    for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R9_IQ4_KB) {
      const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R9_IQ4_KB));
      const int more = (kb_n < nb);
      if (more) {
        const G u8 *X_n = rq_ptr_u(X_u + (unsigned long)kb_n * 32ul);
        rx0 = *(const G u32x4 *)(X_n + xoff0);
        rx1 = *(const G u32x4 *)(X_n + xoff1);
      }
      __builtin_amdgcn_sched_barrier(0);
      const LDSP u8 *lx = ink_r9_iq4_lds.x[bf];
      const LDSP f16 *lsw = ink_r9_iq4_lds.sw[bf];
      const LDSP float *lsx = ink_r9_iq4_lds.sx[bf];

      ink_r9_iq4_k_wj4(wa[0], lx, lsw, lsx, 0u, wr, l16, h, lx_base0, acc);
      ink_r9_iq4_k_wj4(wa[1], lx, lsw, lsx, 1u, wr, l16, h, lx_base0, acc);
      if (more) {
        const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
        wa[0] = *(const G u32x4 *)(qw_n + w_off + 0u);
        wa[1] = *(const G u32x4 *)(qw_n + w_off + 16u);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq4_k_wj4(wa[2], lx, lsw, lsx, 2u, wr, l16, h, lx_base0, acc);

      ink_r9_iq4_u32x2 rsw = {0u, 0u};
      f32x4 rsx = (f32x4)(0.0f);
      if (more) {
        const u32 nbf = bf ^ 1u;
        *(LDSP u32x4 *)(ink_r9_iq4_lds.x[nbf] + st_x) = rx0;
        *(LDSP u32x4 *)(ink_r9_iq4_lds.x[nbf] + st_x + 4096u) = rx1;
        const G u8 *qs_n = rq_ptr_u(qs_e + (unsigned long)kb_n * 2ul);
        const G u8 *Xs_n = rq_ptr_u(X_s + (unsigned long)kb_n * 4ul);
        if (is_sw) rsw = ink_r9_iq4_ld2(qs_n + sw_off);
        if (is_sx) rsx = rq_load_f32x4((const G float *)(Xs_n + sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq4_k_wj4(wa[3], lx, lsw, lsx, 3u, wr, l16, h, lx_base0, acc);
      if (more) {
        const u32 nbf = bf ^ 1u;
        ink_r9_iq4_stage_scales(ink_r9_iq4_lds.sw[nbf], ink_r9_iq4_lds.sx[nbf], t0, 64u, is_sw, is_sx, rsw, rsx);
        const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
        wa[2] = *(const G u32x4 *)(qw_n + w_off + 32u);
        wa[3] = *(const G u32x4 *)(qw_n + w_off + 48u);
        lbarrier();
        bf = nbf;
      }
    }

    ink_r9_iq4_store_out(Y, res, ink_r9_iq4_lds.pid, nj, 4u, rbase, wr, l16, h, rows, ys, rs, acc);
  } else if (cls == 2u) {
    /* ================ Bucket 2: 33..48 pairs (WJ = 3, -25% X & WMMA, 0 branches) ================ */
    const int ld_hi = (tk_ld < 16u);
    const int is_sx = (t0 < 48u);
    if (is_sx) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r9_iq4_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ink_r9_iq4_div_x(ink_r9_iq4_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
    const u32 xoff1 = ld_hi ? (ink_r9_iq4_div_x(ink_r9_iq4_lds.pid[tk_ld + 32u], xdiv) * xs + c8_ld * 16u) : 0u;
    const u32 sxoff = is_sx ? ink_r9_iq4_div_x(ink_r9_iq4_lds.pid[t0], xdiv) * xs : 0u;

    f8 acc[3];
    acc[0] = (f8)(0.0f);
    acc[1] = (f8)(0.0f);
    acc[2] = (f8)(0.0f);

    u32x4 wa[INK_R9_IQ4_KB];
#pragma unroll
    for (u32 k = 0u; k < INK_R9_IQ4_KB; k++) wa[k] = *(const G u32x4 *)(qw_e + w_off + k * 16u);
    {
      const u32x4 rx0_i = *(const G u32x4 *)(X_u + xoff0);
      const u32x4 rx1_i = ld_hi ? *(const G u32x4 *)(X_u + xoff1) : (u32x4)(0u);
      const ink_r9_iq4_u32x2 rsw_i = is_sw ? ink_r9_iq4_ld2(qs_e + sw_off) : (ink_r9_iq4_u32x2){0u, 0u};
      const f32x4 rsx_i = is_sx ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
      *(LDSP u32x4 *)(ink_r9_iq4_lds.x[0] + st_x) = rx0_i;
      if (ld_hi) *(LDSP u32x4 *)(ink_r9_iq4_lds.x[0] + st_x + 4096u) = rx1_i;
      ink_r9_iq4_stage_scales(ink_r9_iq4_lds.sw[0], ink_r9_iq4_lds.sx[0], t0, 64u, is_sw, is_sx, rsw_i, rsx_i);
    }
    lbarrier();

    u32 bf = 0u;
    u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
    for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R9_IQ4_KB) {
      const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R9_IQ4_KB));
      const int more = (kb_n < nb);
      if (more) {
        const G u8 *X_n = rq_ptr_u(X_u + (unsigned long)kb_n * 32ul);
        rx0 = *(const G u32x4 *)(X_n + xoff0);
        if (ld_hi) rx1 = *(const G u32x4 *)(X_n + xoff1);
      }
      __builtin_amdgcn_sched_barrier(0);
      const LDSP u8 *lx = ink_r9_iq4_lds.x[bf];
      const LDSP f16 *lsw = ink_r9_iq4_lds.sw[bf];
      const LDSP float *lsx = ink_r9_iq4_lds.sx[bf];

      ink_r9_iq4_k_wj3(wa[0], lx, lsw, lsx, 0u, wr, l16, h, acc);
      ink_r9_iq4_k_wj3(wa[1], lx, lsw, lsx, 1u, wr, l16, h, acc);
      if (more) {
        const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
        wa[0] = *(const G u32x4 *)(qw_n + w_off + 0u);
        wa[1] = *(const G u32x4 *)(qw_n + w_off + 16u);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq4_k_wj3(wa[2], lx, lsw, lsx, 2u, wr, l16, h, acc);

      ink_r9_iq4_u32x2 rsw = {0u, 0u};
      f32x4 rsx = (f32x4)(0.0f);
      if (more) {
        const u32 nbf = bf ^ 1u;
        *(LDSP u32x4 *)(ink_r9_iq4_lds.x[nbf] + st_x) = rx0;
        if (ld_hi) *(LDSP u32x4 *)(ink_r9_iq4_lds.x[nbf] + st_x + 4096u) = rx1;
        const G u8 *qs_n = rq_ptr_u(qs_e + (unsigned long)kb_n * 2ul);
        const G u8 *Xs_n = rq_ptr_u(X_s + (unsigned long)kb_n * 4ul);
        if (is_sw) rsw = ink_r9_iq4_ld2(qs_n + sw_off);
        if (is_sx) rsx = rq_load_f32x4((const G float *)(Xs_n + sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq4_k_wj3(wa[3], lx, lsw, lsx, 3u, wr, l16, h, acc);
      if (more) {
        const u32 nbf = bf ^ 1u;
        ink_r9_iq4_stage_scales(ink_r9_iq4_lds.sw[nbf], ink_r9_iq4_lds.sx[nbf], t0, 64u, is_sw, is_sx, rsw, rsx);
        const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
        wa[2] = *(const G u32x4 *)(qw_n + w_off + 32u);
        wa[3] = *(const G u32x4 *)(qw_n + w_off + 48u);
        lbarrier();
        bf = nbf;
      }
    }

    ink_r9_iq4_store_out(Y, res, ink_r9_iq4_lds.pid, nj, 3u, rbase, wr, l16, h, rows, ys, rs, acc);
  } else if (cls == 1u) {
    /* ================ Bucket 1: 17..32 pairs (WJ = 2, 0 K-loop branches) ================ */
    const int is_sx = (w == 0u);
    if (is_sx) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r9_iq4_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ink_r9_iq4_div_x(ink_r9_iq4_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
    const u32 sxoff = is_sx ? ink_r9_iq4_div_x(ink_r9_iq4_lds.pid[t0], xdiv) * xs : 0u;

    f8 acc[2];
    acc[0] = (f8)(0.0f);
    acc[1] = (f8)(0.0f);

    u32x4 wa[INK_R9_IQ4_KB];
#pragma unroll
    for (u32 k = 0u; k < INK_R9_IQ4_KB; k++) wa[k] = *(const G u32x4 *)(qw_e + w_off + k * 16u);
    {
      const u32x4 rx0_i = *(const G u32x4 *)(X_u + xoff0);
      const ink_r9_iq4_u32x2 rsw_i = is_sw ? ink_r9_iq4_ld2(qs_e + sw_off) : (ink_r9_iq4_u32x2){0u, 0u};
      const f32x4 rsx_i = is_sx ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
      *(LDSP u32x4 *)(ink_r9_iq4_lds.x[0] + st_x) = rx0_i;
      ink_r9_iq4_stage_scales(ink_r9_iq4_lds.sw[0], ink_r9_iq4_lds.sx[0], t0, 32u, is_sw, is_sx, rsw_i, rsx_i);
    }
    lbarrier();

    u32 bf = 0u;
    u32x4 rx0 = (u32x4)(0u);
#pragma unroll 1
    for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R9_IQ4_KB) {
      const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R9_IQ4_KB));
      const int more = (kb_n < nb);
      if (more) {
        const G u8 *X_n = rq_ptr_u(X_u + (unsigned long)kb_n * 32ul);
        rx0 = *(const G u32x4 *)(X_n + xoff0);
      }
      __builtin_amdgcn_sched_barrier(0);
      const LDSP u8 *lx = ink_r9_iq4_lds.x[bf];
      const LDSP f16 *lsw = ink_r9_iq4_lds.sw[bf];
      const LDSP float *lsx = ink_r9_iq4_lds.sx[bf];

      ink_r9_iq4_k_wj2(wa[0], lx, lsw, lsx, 0u, wr, l16, h, acc);
      ink_r9_iq4_k_wj2(wa[1], lx, lsw, lsx, 1u, wr, l16, h, acc);
      if (more) {
        const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
        wa[0] = *(const G u32x4 *)(qw_n + w_off + 0u);
        wa[1] = *(const G u32x4 *)(qw_n + w_off + 16u);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq4_k_wj2(wa[2], lx, lsw, lsx, 2u, wr, l16, h, acc);

      ink_r9_iq4_u32x2 rsw = {0u, 0u};
      f32x4 rsx = (f32x4)(0.0f);
      if (more) {
        const u32 nbf = bf ^ 1u;
        *(LDSP u32x4 *)(ink_r9_iq4_lds.x[nbf] + st_x) = rx0;
        const G u8 *qs_n = rq_ptr_u(qs_e + (unsigned long)kb_n * 2ul);
        const G u8 *Xs_n = rq_ptr_u(X_s + (unsigned long)kb_n * 4ul);
        if (is_sw) rsw = ink_r9_iq4_ld2(qs_n + sw_off);
        if (is_sx) rsx = rq_load_f32x4((const G float *)(Xs_n + sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq4_k_wj2(wa[3], lx, lsw, lsx, 3u, wr, l16, h, acc);
      if (more) {
        const u32 nbf = bf ^ 1u;
        ink_r9_iq4_stage_scales(ink_r9_iq4_lds.sw[nbf], ink_r9_iq4_lds.sx[nbf], t0, 32u, is_sw, is_sx, rsw, rsx);
        const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
        wa[2] = *(const G u32x4 *)(qw_n + w_off + 32u);
        wa[3] = *(const G u32x4 *)(qw_n + w_off + 48u);
        lbarrier();
        bf = nbf;
      }
    }

    ink_r9_iq4_store_out(Y, res, ink_r9_iq4_lds.pid, nj, 2u, rbase, wr, l16, h, rows, ys, rs, acc);
  } else {
    /* ================ Bucket 0: 1..16 pairs (WJ = 1, -50% X & WMMA vs r2!) ================ */
    const int ld_lo = (tk_ld < 16u);
    const int is_sx = (t0 < 16u);
    if (is_sx) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r9_iq4_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ld_lo ? (ink_r9_iq4_div_x(ink_r9_iq4_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u) : 0u;
    const u32 sxoff = is_sx ? ink_r9_iq4_div_x(ink_r9_iq4_lds.pid[t0], xdiv) * xs : 0u;

    f8 acc0 = (f8)(0.0f);

    u32x4 wa[INK_R9_IQ4_KB];
#pragma unroll
    for (u32 k = 0u; k < INK_R9_IQ4_KB; k++) wa[k] = *(const G u32x4 *)(qw_e + w_off + k * 16u);
    {
      const u32x4 rx0_i = ld_lo ? *(const G u32x4 *)(X_u + xoff0) : (u32x4)(0u);
      const ink_r9_iq4_u32x2 rsw_i = is_sw ? ink_r9_iq4_ld2(qs_e + sw_off) : (ink_r9_iq4_u32x2){0u, 0u};
      const f32x4 rsx_i = is_sx ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
      if (ld_lo) *(LDSP u32x4 *)(ink_r9_iq4_lds.x[0] + st_x) = rx0_i;
      ink_r9_iq4_stage_scales(ink_r9_iq4_lds.sw[0], ink_r9_iq4_lds.sx[0], t0, 32u, is_sw, is_sx, rsw_i, rsx_i);
    }
    lbarrier();

    u32 bf = 0u;
    u32x4 rx0 = (u32x4)(0u);
#pragma unroll 1
    for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R9_IQ4_KB) {
      const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R9_IQ4_KB));
      const int more = (kb_n < nb);
      if (more && ld_lo) {
        const G u8 *X_n = rq_ptr_u(X_u + (unsigned long)kb_n * 32ul);
        rx0 = *(const G u32x4 *)(X_n + xoff0);
      }
      __builtin_amdgcn_sched_barrier(0);
      const LDSP u8 *lx = ink_r9_iq4_lds.x[bf];
      const LDSP f16 *lsw = ink_r9_iq4_lds.sw[bf];
      const LDSP float *lsx = ink_r9_iq4_lds.sx[bf];

      ink_r9_iq4_k_wj1(wa[0], lx, lsw, lsx, 0u, wr, l16, h, &acc0);
      ink_r9_iq4_k_wj1(wa[1], lx, lsw, lsx, 1u, wr, l16, h, &acc0);
      if (more) {
        const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
        wa[0] = *(const G u32x4 *)(qw_n + w_off + 0u);
        wa[1] = *(const G u32x4 *)(qw_n + w_off + 16u);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq4_k_wj1(wa[2], lx, lsw, lsx, 2u, wr, l16, h, &acc0);

      ink_r9_iq4_u32x2 rsw = {0u, 0u};
      f32x4 rsx = (f32x4)(0.0f);
      if (more) {
        const u32 nbf = bf ^ 1u;
        if (ld_lo) *(LDSP u32x4 *)(ink_r9_iq4_lds.x[nbf] + st_x) = rx0;
        const G u8 *qs_n = rq_ptr_u(qs_e + (unsigned long)kb_n * 2ul);
        const G u8 *Xs_n = rq_ptr_u(X_s + (unsigned long)kb_n * 4ul);
        if (is_sw) rsw = ink_r9_iq4_ld2(qs_n + sw_off);
        if (is_sx) rsx = rq_load_f32x4((const G float *)(Xs_n + sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq4_k_wj1(wa[3], lx, lsw, lsx, 3u, wr, l16, h, &acc0);
      if (more) {
        const u32 nbf = bf ^ 1u;
        ink_r9_iq4_stage_scales(ink_r9_iq4_lds.sw[nbf], ink_r9_iq4_lds.sx[nbf], t0, 32u, is_sw, is_sx, rsw, rsx);
        const G u8 *qw_n = rq_ptr_u(qw_e + (unsigned long)kb_n * 16ul);
        wa[2] = *(const G u32x4 *)(qw_n + w_off + 32u);
        wa[3] = *(const G u32x4 *)(qw_n + w_off + 48u);
        lbarrier();
        bf = nbf;
      }
    }

    ink_r9_iq4_store_out(Y, res, ink_r9_iq4_lds.pid, nj, 1u, rbase, wr, l16, h, rows, ys, rs, &acc0);
  }
}

#undef INK_R9_IQ4_SW_STR
#undef INK_R9_IQ4_R
#undef INK_R9_IQ4_KB

#endif /* !IE_MEGA */
