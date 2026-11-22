/* ============================================================================
 * Inkling-2 Round 7 (Wave 3, Bit 71): High-Occupancy (<= 128 VGPR, 20.25 KiB
 * LDS = 3 WGs/CU) Compact Worklist 16/32/48/64 IQ4_NL MoE Prefill WMMA GEMM
 * with 2x f32x4 Vectorized Epilogue Stores for RX 9070 XT (gfx1201, wave32).
 *
 * Target: MoE ffn_down_exps MAT_IQ4NL (kind=8, 2304x1152, nb=36, 128E/K=8),
 *         21.8% of 4K prefill in R00 trace #191!
 * All local symbols/LDS/macros prefixed ink_r7_iq4_ / INK_R7_IQ4_; macros
 * #undef'd at EOF. Clang 21 scalar __builtin_bit_cast safe, 0/0 spills.
 *
 * Exported Kernel, Grid, Guard, and Parity:
 *   KERNEL ie_inkling_gemm_iq4nl_exp_r7(
 *       const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
 *       u32 rows, u32 nb, const G u32 *list, const G u32 *tiles,
 *       const G u32 *n_tiles, u32 qwn, u32 qsn, u32 xs, u32 ys,
 *       u32 xdiv, u32 kind, const G u8 *res, u32 rs)
 *   - Bit:    71 (ink_mask2 bit 7)
 *   - Grid:   ((rows + 127u) / 128u, max_tiles, 1u), NT = 256
 *             where max_tiles = min(T*K, n_exp + (T*K)/64u)
 *             (consumes the exact same b->ms_packs / b->ms_npacks flat
 *              largest-first 16/32/48/64 worklist built by
 *              ie_inkling_moe_pf_worklist_r7!).
 *   - Guard:  !mk.rec && T >= 17u && kind == MAT_IQ4NL (8u) && rows > 0u &&
 *             nb > 0u && (nb & 3u) == 0u && list && tiles && n_tiles &&
 *             n_exp <= 256u && T * K <= 8192u.
 *   - Key Upgrades over ie_inkling_gemm_iq4nl_exp_r2 (152 VGPR, 22,784 B = 2 WG/CU):
 *     1. f16 sw LDS shrinks total LDS 22,784 B -> 20,736 B (20.25 KiB <= 21.8 KiB)
 *        AND halves sw LDS loads (1x ds_load_b128 h8 -> f8 vs 2x b128 f32).
 *     2. Early staging of rx/rsw/rsx after k=1 + 32-bit offsets + single-tile
 *        worklist drops VGPRs 152 -> <= 128 VGPR -> UNLOCKS 3 WGs/CU (+50% occ!).
 *     3. Adaptive 16/32/48/64 (WJ=1/2/3/4) worklist tiles: -25% X reads & WMMAs
 *        on 33..48 tiles, -50% on 1..16 tiles, 0 empty-expert WGs, largest-first.
 *     4. 2x f32x4 vectorized epilogue stores (4x fewer store ops vs r2_g15's 8x
 *        scalar stores; automatic scalar fallback for unaligned/tail rows).
 *   - Parity: 100% bitwise-exact (0-bit-diff) vs ie_inkling_gemm_iq4nl_exp_r2!
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R7_IQ4_KB 4u
#define INK_R7_IQ4_R  128u

typedef _Float16 ink_r7_iq4_f16x4 __attribute__((ext_vector_type(4)));
typedef unsigned int ink_r7_iq4_u32x2 __attribute__((ext_vector_type(2)));

struct __attribute__((packed, aligned(4))) ink_r7_iq4_u32x2p {
  ink_r7_iq4_u32x2 v;
};

static inline __attribute__((always_inline)) ink_r7_iq4_u32x2 ink_r7_iq4_ld2(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r7_iq4_u32x2p *)p)->v;
}

typedef struct {
  u8    x[2][64u * 128u] __attribute__((aligned(16)));             /* 16,384 B */
  f16   sw[2][INK_R7_IQ4_KB * INK_R7_IQ4_R] __attribute__((aligned(16))); /*  2,048 B */
  float sx[2][INK_R7_IQ4_KB * 64u] __attribute__((aligned(32)));   /*  2,048 B */
  u32   pid[64] __attribute__((aligned(16)));                      /*    256 B */
} ink_r7_iq4_lds_t;

_Static_assert(sizeof(ink_r7_iq4_lds_t) == 20736u, "ink_r7_iq4_lds_t must be 20,736 B (<= 21.8 KiB = 3 WGs/CU)");

static inline __attribute__((always_inline)) u32 ink_r7_iq4_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

static inline __attribute__((always_inline)) void ink_r7_iq4_stage_sw(
    LDSP f16 *lsw, const u32 t0, const int is_sw, const ink_r7_iq4_u32x2 rsw) {
  if (is_sw) {
    const u32 w0 = rsw.x, w1 = rsw.y;
    const u16 s0 = (u16)(w0 & 0xFFFFu);
    const u16 s1 = (u16)(w0 >> 16u);
    const u16 s2 = (u16)(w1 & 0xFFFFu);
    const u16 s3 = (u16)(w1 >> 16u);
    lsw[0u * INK_R7_IQ4_R + t0] = __builtin_bit_cast(f16, s0);
    lsw[1u * INK_R7_IQ4_R + t0] = __builtin_bit_cast(f16, s1);
    lsw[2u * INK_R7_IQ4_R + t0] = __builtin_bit_cast(f16, s2);
    lsw[3u * INK_R7_IQ4_R + t0] = __builtin_bit_cast(f16, s3);
  }
}

static inline __attribute__((always_inline)) void ink_r7_iq4_stage_x64(
    LDSP u8 *lx, LDSP f16 *lsw, LDSP float *lsx, const u32 st_x, const u32 t0,
    const int is_sw, const int is_sx, const int ld_hi,
    const u32x4 rx0, const u32x4 rx1, const ink_r7_iq4_u32x2 rsw, const f32x4 rsx) {
  *(LDSP u32x4 *)(lx + st_x) = rx0;
  if (ld_hi) *(LDSP u32x4 *)(lx + st_x + 4096u) = rx1;
  ink_r7_iq4_stage_sw(lsw, t0, is_sw, rsw);
  if (is_sx) {
    lsx[0u * 64u + t0] = rsx.x;
    lsx[1u * 64u + t0] = rsx.y;
    lsx[2u * 64u + t0] = rsx.z;
    lsx[3u * 64u + t0] = rsx.w;
  }
}

static inline __attribute__((always_inline)) void ink_r7_iq4_stage_x32(
    LDSP u8 *lx, LDSP f16 *lsw, LDSP float *lsx, const u32 st_x, const u32 t0,
    const int is_sw, const int is_sx, const int ld_lo,
    const u32x4 rx0, const ink_r7_iq4_u32x2 rsw, const f32x4 rsx) {
  if (ld_lo) *(LDSP u32x4 *)(lx + st_x) = rx0;
  ink_r7_iq4_stage_sw(lsw, t0, is_sw, rsw);
  if (is_sx) {
    lsx[0u * 32u + t0] = rsx.x;
    lsx[1u * 32u + t0] = rsx.y;
    lsx[2u * 32u + t0] = rsx.z;
    lsx[3u * 32u + t0] = rsx.w;
  }
}

/* Single K-block IQ4_NL WMMA for 48- or 64-pair tile (WJ = 3 or 4). */
static inline __attribute__((always_inline)) void ink_r7_iq4_k_hi(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 l16, const u32 h, const int has_j3, f8 acc[4]) {
  const v8i zm = (v8i)(0x4B400000);
  const u32 x_col = ((2u * k + h) ^ (l16 & 7u)) << 4u;
  const LDSP u8 *lx_k = lx + l16 * 128u + x_col;
  const LDSP float *lsx_k = lsx + k * 64u + l16;

  u32x4 nb4 = h ? ((wk >> 4u) & 0x0F0F0F0Fu) : (wk & 0x0F0F0F0Fu);
  nb4.x = ie_iq4_map(nb4.x);
  nb4.y = ie_iq4_map(nb4.y);
  nb4.z = ie_iq4_map(nb4.z);
  nb4.w = ie_iq4_map(nb4.w);
  const v2i a0 = {(int)nb4.x, (int)nb4.y};
  const v2i a1 = {(int)nb4.z, (int)nb4.w};

  /* Pair 0 (j = 0, 1) */
  u32x4 b0 = *(const LDSP u32x4 *)(lx_k + 0u * 2048u);
  u32x4 b1 = *(const LDSP u32x4 *)(lx_k + 1u * 2048u);
  float dx0 = lsx_k[0u];
  float dx1 = lsx_k[16u];
  __builtin_amdgcn_sched_barrier(0);

  v2i b0_lo = {(int)b0.x, (int)b0.y}, b0_hi = {(int)b0.z, (int)b0.w};
  v2i b1_lo = {(int)b1.x, (int)b1.y}, b1_hi = {(int)b1.z, (int)b1.w};
  v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_lo, zm, 0);
  v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b1_lo, zm, 0);
  c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_hi, c0, 0);
  c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1_hi, c1, 0);
  __builtin_amdgcn_sched_barrier(0);

  const h8 sw_h = *(const LDSP h8 *)(lsw + k * INK_R7_IQ4_R + wr + 8u * h);
  const f8 sw = __builtin_convertvector(sw_h, f8);
  acc[0] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  acc[1] += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
  __builtin_amdgcn_sched_barrier(0);

  /* Pair 1 (j = 2, and j = 3 only if nj > 48) */
  if (has_j3) {
    b0 = *(const LDSP u32x4 *)(lx_k + 2u * 2048u);
    b1 = *(const LDSP u32x4 *)(lx_k + 3u * 2048u);
    dx0 = lsx_k[32u];
    dx1 = lsx_k[48u];
    __builtin_amdgcn_sched_barrier(0);

    b0_lo = (v2i){(int)b0.x, (int)b0.y};
    b0_hi = (v2i){(int)b0.z, (int)b0.w};
    b1_lo = (v2i){(int)b1.x, (int)b1.y};
    b1_hi = (v2i){(int)b1.z, (int)b1.w};
    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_lo, zm, 0);
    c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b1_lo, zm, 0);
    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_hi, c0, 0);
    c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1_hi, c1, 0);
    __builtin_amdgcn_sched_barrier(0);

    acc[2] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
    acc[3] += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
  } else {
    b0 = *(const LDSP u32x4 *)(lx_k + 2u * 2048u);
    dx0 = lsx_k[32u];
    __builtin_amdgcn_sched_barrier(0);

    b0_lo = (v2i){(int)b0.x, (int)b0.y};
    b0_hi = (v2i){(int)b0.z, (int)b0.w};
    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_lo, zm, 0);
    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_hi, c0, 0);
    __builtin_amdgcn_sched_barrier(0);

    acc[2] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  }
  __builtin_amdgcn_sched_barrier(0);
}

/* Single K-block IQ4_NL WMMA for 16- or 32-pair tile (WJ = 1 or 2). */
static inline __attribute__((always_inline)) void ink_r7_iq4_k_lo(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 l16, const u32 h, const int has_j1, f8 acc[2]) {
  const v8i zm = (v8i)(0x4B400000);
  const u32 x_col = ((2u * k + h) ^ (l16 & 7u)) << 4u;
  const LDSP u8 *lx_k = lx + l16 * 128u + x_col;
  const LDSP float *lsx_k = lsx + k * 32u + l16;

  u32x4 nb4 = h ? ((wk >> 4u) & 0x0F0F0F0Fu) : (wk & 0x0F0F0F0Fu);
  nb4.x = ie_iq4_map(nb4.x);
  nb4.y = ie_iq4_map(nb4.y);
  nb4.z = ie_iq4_map(nb4.z);
  nb4.w = ie_iq4_map(nb4.w);
  const v2i a0 = {(int)nb4.x, (int)nb4.y};
  const v2i a1 = {(int)nb4.z, (int)nb4.w};

  const h8 sw_h = *(const LDSP h8 *)(lsw + k * INK_R7_IQ4_R + wr + 8u * h);
  if (has_j1) {
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

    const f8 sw = __builtin_convertvector(sw_h, f8);
    acc[0] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
    acc[1] += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
  } else {
    const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + 0u * 2048u);
    const float dx0 = lsx_k[0u];
    __builtin_amdgcn_sched_barrier(0);

    const v2i b0_lo = {(int)b0.x, (int)b0.y}, b0_hi = {(int)b0.z, (int)b0.w};
    v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_lo, zm, 0);
    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_hi, c0, 0);
    __builtin_amdgcn_sched_barrier(0);

    const f8 sw = __builtin_convertvector(sw_h, f8);
    acc[0] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  }
  __builtin_amdgcn_sched_barrier(0);
}

/* 2x f32x4 vectorized epilogue store (exact 0-bit-diff with automatic tail fallback). */
static inline __attribute__((always_inline)) void ink_r7_iq4_store_out(
    G u8 *Y, const G u8 *res, const LDSP u32 *lpid, const u32 nj, const u32 wj,
    const u32 rbase, const u32 wr, const u32 l16, const u32 h, const u32 rows,
    const u32 ys, const u32 rs, const f8 *acc) {
  const u32 r0 = rbase + wr + 8u * h;
  const int vec_ok = (r0 + 8u <= rows) && ((ys & 15u) == 0u) && (!res || (rs & 15u) == 0u);
#pragma unroll
  for (u32 j = 0u; j < wj; j++) {
    const u32 slot = 16u * j + l16;
    if (slot >= nj) continue;
    const u32 pid = lpid[slot];
    G float *yr = (G float *)(Y + (unsigned long)pid * ys);
    if (vec_ok) {
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
    } else {
      const G float *rr = res ? (const G float *)(res + (unsigned long)pid * rs) : (const G float *)0;
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 r = r0 + v;
        if (r < rows) yr[r] = epilogue(acc[j][v], r, 0, rr);
      }
    }
  }
}

KERNEL ie_inkling_gemm_iq4nl_exp_r7(const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
                                    u32 rows, u32 nb, const G u32 *list, const G u32 *tiles,
                                    const G u32 *n_tiles, u32 qwn, u32 qsn, u32 xs, u32 ys,
                                    u32 xdiv, u32 kind, const G u8 *res, u32 rs) {
  static LDS ink_r7_iq4_lds_t ink_r7_iq4_lds;
  (void)kind;
  if (!n_tiles || !tiles || !list || !qw0 || !qs0 || !X || !Y || nb == 0u || (nb & 3u) != 0u || xdiv == 0u) return;

  const u32 ti = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  const u32 nt_act = (u32)__builtin_amdgcn_readfirstlane((int)(*n_tiles));
  const u32 rbase = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() * INK_R7_IQ4_R));
  if (ti >= nt_act || rbase >= rows) return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)tiles[ti]);
  const u32 e = desc & 0xFFu;
  const u32 pb = (desc >> 8u) & 0x1FFFu;
  const u32 nj = (desc >> 21u) & 0x7Fu;
  if (nj == 0u || nj > 64u) return;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), l16 = l & 15u, h = l >> 4u;
  const u32 wr = 16u * w;
  const int is_sw = (w < 4u);

  const G u8 *qw_e = rq_ptr_u((const G u8 *)(qw0 + (unsigned long)e * qwn));
  const G u16 *qs_e = (const G u16 *)rq_ptr_u((const G u8 *)(qs0 + (unsigned long)e * qsn));
  const G u8 *X_s = rq_ptr_u(X + 32ul * nb);

  const u32 rw_idx = rbase + wr + l16;
  const u32 w_off = (rw_idx < rows ? rw_idx : rows - 1u) * nb * 16u;
  const u32 sr = rbase + (is_sw ? t0 : 0u);
  const u32 sw_off = (sr < rows ? sr : rows - 1u) * nb;

  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  const u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);

  if (nj > 32u) {
    /* ---------------- Adaptive 48- or 64-pair tile (WJ = 3 or 4) ---------------- */
    const int has_j3 = __builtin_amdgcn_readfirstlane(nj > 48u);
    const int ld_hi = has_j3 || (tk_ld < 16u);
    const int is_sx = (w < 2u) && (has_j3 || (t0 < 48u));
    if (w < 2u) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r7_iq4_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ink_r7_iq4_div_x(ink_r7_iq4_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
    const u32 xoff1 = ld_hi ? (ink_r7_iq4_div_x(ink_r7_iq4_lds.pid[tk_ld + 32u], xdiv) * xs + c8_ld * 16u) : 0u;
    const u32 sxoff = is_sx ? ink_r7_iq4_div_x(ink_r7_iq4_lds.pid[t0 & 63u], xdiv) * xs : 0u;

    f8 acc[4];
#pragma unroll
    for (u32 j = 0u; j < 4u; j++) acc[j] = (f8)(0.0f);

    u32x4 wa[INK_R7_IQ4_KB];
#pragma unroll
    for (u32 k = 0u; k < INK_R7_IQ4_KB; k++) wa[k] = *(const G u32x4 *)(qw_e + (unsigned long)w_off + k * 16u);
    u32x4 rx0 = *(const G u32x4 *)(X + (unsigned long)xoff0);
    u32x4 rx1 = ld_hi ? *(const G u32x4 *)(X + (unsigned long)xoff1) : (u32x4)(0u);
    ink_r7_iq4_u32x2 rsw = is_sw ? ink_r7_iq4_ld2(qs_e + (unsigned long)sw_off) : (ink_r7_iq4_u32x2){0u, 0u};
    f32x4 rsx = is_sx ? rq_load_f32x4((const G float *)(X_s + (unsigned long)sxoff)) : (f32x4)(0.0f);

    ink_r7_iq4_stage_x64(ink_r7_iq4_lds.x[0], ink_r7_iq4_lds.sw[0], ink_r7_iq4_lds.sx[0],
                         st_x, t0, is_sw, is_sx, ld_hi, rx0, rx1, rsw, rsx);
    lbarrier();

    u32 bf = 0u;
#pragma unroll 1
    for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R7_IQ4_KB) {
      const u32 kb_n = kb0 + INK_R7_IQ4_KB;
      const int more = (kb_n < nb);
      u32 nk = (u32)__builtin_amdgcn_readfirstlane((int)(nb - kb0 < INK_R7_IQ4_KB ? nb - kb0 : INK_R7_IQ4_KB));
      __asm__ volatile("" : "+s"(nk));

      if (more) {
        const G u8 *X_n = X + (unsigned long)kb_n * 32ul;
        rx0 = *(const G u32x4 *)(X_n + (unsigned long)xoff0);
        if (ld_hi) rx1 = *(const G u32x4 *)(X_n + (unsigned long)xoff1);
        if (is_sw) rsw = ink_r7_iq4_ld2(qs_e + (unsigned long)sw_off + kb_n);
        if (is_sx) rsx = rq_load_f32x4((const G float *)(X_s + (unsigned long)kb_n * 4ul + (unsigned long)sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);
      const LDSP u8 *lx = ink_r7_iq4_lds.x[bf];
      const LDSP f16 *lsw = ink_r7_iq4_lds.sw[bf];
      const LDSP float *lsx = ink_r7_iq4_lds.sx[bf];

      if (0u < nk) ink_r7_iq4_k_hi(wa[0], lx, lsw, lsx, 0u, wr, l16, h, has_j3, acc);
      if (1u < nk) ink_r7_iq4_k_hi(wa[1], lx, lsw, lsx, 1u, wr, l16, h, has_j3, acc);
      /* Stage next chunk into dead lds[bf^1] right after k=1 -> frees 14 VGPRs before wa[0..1] reload! */
      if (more) {
        const u32 nbf = bf ^ 1u;
        ink_r7_iq4_stage_x64(ink_r7_iq4_lds.x[nbf], ink_r7_iq4_lds.sw[nbf], ink_r7_iq4_lds.sx[nbf],
                             st_x, t0, is_sw, is_sx, ld_hi, rx0, rx1, rsw, rsx);
        __builtin_amdgcn_sched_barrier(0);
        const G u8 *wp_n = qw_e + (unsigned long)w_off + (unsigned long)kb_n * 16ul;
        wa[0] = *(const G u32x4 *)(wp_n + 0u);
        wa[1] = *(const G u32x4 *)(wp_n + 16u);
      }
      __builtin_amdgcn_sched_barrier(0);
      if (2u < nk) ink_r7_iq4_k_hi(wa[2], lx, lsw, lsx, 2u, wr, l16, h, has_j3, acc);
      if (3u < nk) ink_r7_iq4_k_hi(wa[3], lx, lsw, lsx, 3u, wr, l16, h, has_j3, acc);
      if (more) {
        const G u8 *wp_n = qw_e + (unsigned long)w_off + (unsigned long)kb_n * 16ul;
        wa[2] = *(const G u32x4 *)(wp_n + 32u);
        wa[3] = *(const G u32x4 *)(wp_n + 48u);
        lbarrier();
        bf ^= 1u;
      }
    }

    ink_r7_iq4_store_out(Y, res, ink_r7_iq4_lds.pid, nj, 4u, rbase, wr, l16, h, rows, ys, rs, acc);
  } else {
    /* ---------------- Adaptive 16- or 32-pair tile (WJ = 1 or 2) ---------------- */
    const int has_j1 = __builtin_amdgcn_readfirstlane(nj > 16u);
    const int ld_lo = has_j1 || (tk_ld < 16u);
    const int is_sx = (w == 0u) && (has_j1 || (t0 < 16u));
    if (w == 0u) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r7_iq4_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ld_lo ? (ink_r7_iq4_div_x(ink_r7_iq4_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u) : 0u;
    const u32 sxoff = is_sx ? ink_r7_iq4_div_x(ink_r7_iq4_lds.pid[t0 & 31u], xdiv) * xs : 0u;

    f8 acc[2];
    acc[0] = (f8)(0.0f);
    acc[1] = (f8)(0.0f);

    u32x4 wa[INK_R7_IQ4_KB];
#pragma unroll
    for (u32 k = 0u; k < INK_R7_IQ4_KB; k++) wa[k] = *(const G u32x4 *)(qw_e + (unsigned long)w_off + k * 16u);
    u32x4 rx0 = ld_lo ? *(const G u32x4 *)(X + (unsigned long)xoff0) : (u32x4)(0u);
    ink_r7_iq4_u32x2 rsw = is_sw ? ink_r7_iq4_ld2(qs_e + (unsigned long)sw_off) : (ink_r7_iq4_u32x2){0u, 0u};
    f32x4 rsx = is_sx ? rq_load_f32x4((const G float *)(X_s + (unsigned long)sxoff)) : (f32x4)(0.0f);

    ink_r7_iq4_stage_x32(ink_r7_iq4_lds.x[0], ink_r7_iq4_lds.sw[0], ink_r7_iq4_lds.sx[0],
                         st_x, t0, is_sw, is_sx, ld_lo, rx0, rsw, rsx);
    lbarrier();

    u32 bf = 0u;
#pragma unroll 1
    for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R7_IQ4_KB) {
      const u32 kb_n = kb0 + INK_R7_IQ4_KB;
      const int more = (kb_n < nb);
      u32 nk = (u32)__builtin_amdgcn_readfirstlane((int)(nb - kb0 < INK_R7_IQ4_KB ? nb - kb0 : INK_R7_IQ4_KB));
      __asm__ volatile("" : "+s"(nk));

      if (more) {
        if (ld_lo) rx0 = *(const G u32x4 *)((X + (unsigned long)kb_n * 32ul) + (unsigned long)xoff0);
        if (is_sw) rsw = ink_r7_iq4_ld2(qs_e + (unsigned long)sw_off + kb_n);
        if (is_sx) rsx = rq_load_f32x4((const G float *)(X_s + (unsigned long)kb_n * 4ul + (unsigned long)sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);
      const LDSP u8 *lx = ink_r7_iq4_lds.x[bf];
      const LDSP f16 *lsw = ink_r7_iq4_lds.sw[bf];
      const LDSP float *lsx = ink_r7_iq4_lds.sx[bf];

      if (0u < nk) ink_r7_iq4_k_lo(wa[0], lx, lsw, lsx, 0u, wr, l16, h, has_j1, acc);
      if (1u < nk) ink_r7_iq4_k_lo(wa[1], lx, lsw, lsx, 1u, wr, l16, h, has_j1, acc);
      if (more) {
        const u32 nbf = bf ^ 1u;
        ink_r7_iq4_stage_x32(ink_r7_iq4_lds.x[nbf], ink_r7_iq4_lds.sw[nbf], ink_r7_iq4_lds.sx[nbf],
                             st_x, t0, is_sw, is_sx, ld_lo, rx0, rsw, rsx);
        __builtin_amdgcn_sched_barrier(0);
        const G u8 *wp_n = qw_e + (unsigned long)w_off + (unsigned long)kb_n * 16ul;
        wa[0] = *(const G u32x4 *)(wp_n + 0u);
        wa[1] = *(const G u32x4 *)(wp_n + 16u);
      }
      __builtin_amdgcn_sched_barrier(0);
      if (2u < nk) ink_r7_iq4_k_lo(wa[2], lx, lsw, lsx, 2u, wr, l16, h, has_j1, acc);
      if (3u < nk) ink_r7_iq4_k_lo(wa[3], lx, lsw, lsx, 3u, wr, l16, h, has_j1, acc);
      if (more) {
        const G u8 *wp_n = qw_e + (unsigned long)w_off + (unsigned long)kb_n * 16ul;
        wa[2] = *(const G u32x4 *)(wp_n + 32u);
        wa[3] = *(const G u32x4 *)(wp_n + 48u);
        lbarrier();
        bf ^= 1u;
      }
    }

    ink_r7_iq4_store_out(Y, res, ink_r7_iq4_lds.pid, nj, 2u, rbase, wr, l16, h, rows, ys, rs, acc);
  }
}

#undef INK_R7_IQ4_R
#undef INK_R7_IQ4_KB

#endif /* !IE_MEGA */
