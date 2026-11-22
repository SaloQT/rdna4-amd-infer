/* ============================================================================
 * Inkling-2 Round 2 Group 15 (g15-iq4nl-pf-r2): Adaptive 128x64 / 128x32
 * IQ4_NL MoE Prefill WMMA GEMM for RX 9070 XT (gfx1201, wave32, NT=256).
 *
 * Target: MoE ffn_down_exps MAT_IQ4NL (kind=8), shape 2304x1152 (nb=36), 128E/K=8.
 * Bitmask: IE_INKLING_KERNEL_MASK bit 32 (0x100000000ULL).
 *
 * Exported Kernel, Grid, Guard, and Parity:
 * - KERNEL ie_inkling_gemm_iq4nl_exp_r2(const G u32 *qw0, const G u16 *qs0,
 *     const G u8 *X, G u8 *Y, u32 rows, u32 nb, const G u32 *list,
 *     const G u32 *offs, u32 np, u32 qwn, u32 qsn, u32 xs, u32 ys,
 *     u32 xdiv, u32 kind, const G u8 *res, u32 rs)
 *   - Replaces: ie_inkling_gemm_iq4nl_exp_r1 / ie_gemm_kx (T>1 MAT_IQ4NL)
 *   - Grid:     ((rows + 127u) / 128u, list ? m->n_exp : (np + 63u) / 64u, 1)
 *               [e.g. (18, 128, 1) = 2304 WGs = 36 WGs/CU for Inkling-2]
 *   - Guard:    T > 1u && kind == MAT_IQ4NL (8u) && nb > 0u && (nb & 3u) == 0u
 *   - Parity:   bitwise-identical to ie_inkling_gemm_iq4nl_exp_r1 (f32-tol ~1e-5
 *               vs ie_gexp_t; exact same WMMA + FMA K-block accumulation order)
 *
 * Key R2 Optimizations:
 * - Adaptive 128x64 (rem > 32, WJ=4) / 128x32 (rem <= 32, WJ=2) token tiling:
 *   halves IQ4_NL weight memory traffic & decode work on all >32-pair experts
 *   (~50% of experts carrying ~65% of pairs at T=512, K=8, E=128) while keeping
 *   zero wasted WMMAs/X loads on <=32-pair experts.
 * - XOR-swizzled bank-conflict-free 128 B/row activation LDS + K-major sx LDS
 *   (128-bit rq_load_f32x4) + 64-bit f16x4 weight-scale loads + cached pid[64]
 *   (22,784 B LDS -> 2 WGs/CU with 19.5 KiB spare).
 * - Dead-weight in-place reload (wa[0..1] after k=1, wa[2..3] after k=3; zero
 *   wn[4] array -> -16 VGPRs), nk-separated k basic blocks, and sched_barrier-
 *   sequenced two-pair WMMA micro-loop (reusing b0/b1 and c0/c1 VGPRs) ->
 *   <= 160 VGPR, 0 spills, 2 WGs/CU.
 * ============================================================================ */

#define INK_R2_G15_KB 4u
#define INK_R2_G15_R  128u

typedef _Float16 ink_r2_g15_f16x4 __attribute__((ext_vector_type(4)));

typedef struct {
  u8 x[2][64u * 128u] __attribute__((aligned(16)));
  float sw[2][INK_R2_G15_KB * INK_R2_G15_R] __attribute__((aligned(32)));
  float sx[2][INK_R2_G15_KB * 64u] __attribute__((aligned(32)));
  u32 pid[64] __attribute__((aligned(16)));
} ink_r2_g15_lds_t;

_Static_assert(sizeof(ink_r2_g15_lds_t) == 22784u, "ink_r2_g15_lds_t must be 22,784 B");

static inline __attribute__((always_inline)) u32 ink_r2_g15_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

static inline __attribute__((always_inline)) void ink_r2_g15_stage_sw(
    LDSP float *lsw, const u32 t0, const int is_sw, const ink_r2_g15_f16x4 rsw) {
  if (is_sw) {
    lsw[0u * INK_R2_G15_R + t0] = (float)rsw[0];
    lsw[1u * INK_R2_G15_R + t0] = (float)rsw[1];
    lsw[2u * INK_R2_G15_R + t0] = (float)rsw[2];
    lsw[3u * INK_R2_G15_R + t0] = (float)rsw[3];
  }
}

static inline __attribute__((always_inline)) void ink_r2_g15_stage_x64(
    LDSP u8 *lx, LDSP float *lsw, LDSP float *lsx, const u32 st_x, const u32 t0,
    const int is_sw, const int is_sx, const u32x4 rx0, const u32x4 rx1,
    const ink_r2_g15_f16x4 rsw, const f32x4 rsx) {
  *(LDSP u32x4 *)(lx + st_x) = rx0;
  *(LDSP u32x4 *)(lx + st_x + 4096u) = rx1;
  ink_r2_g15_stage_sw(lsw, t0, is_sw, rsw);
  if (is_sx) {
    lsx[0u * 64u + t0] = rsx.x;
    lsx[1u * 64u + t0] = rsx.y;
    lsx[2u * 64u + t0] = rsx.z;
    lsx[3u * 64u + t0] = rsx.w;
  }
}

static inline __attribute__((always_inline)) void ink_r2_g15_stage_x32(
    LDSP u8 *lx, LDSP float *lsw, LDSP float *lsx, const u32 st_x, const u32 t0,
    const int is_sw, const int is_sx, const u32x4 rx0,
    const ink_r2_g15_f16x4 rsw, const f32x4 rsx) {
  *(LDSP u32x4 *)(lx + st_x) = rx0;
  ink_r2_g15_stage_sw(lsw, t0, is_sw, rsw);
  if (is_sx) {
    lsx[0u * 32u + t0] = rsx.x;
    lsx[1u * 32u + t0] = rsx.y;
    lsx[2u * 32u + t0] = rsx.z;
    lsx[3u * 32u + t0] = rsx.w;
  }
}

/* Single K-block IQ4_NL WMMA for 64-pair tile (WJ=4), split into two strictly
 * sequenced 2-tile pairs so b0/b1 and c0/c1 VGPRs are reused without hoisting. */
static inline __attribute__((always_inline)) void ink_r2_g15_iq4_k64(
    const u32x4 wk, const LDSP u8 *lx, const LDSP float *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 l16, const u32 h, f8 acc[4]) {
  const v8i zm = (v8i)(0x4B400000);
  const u32 x_col = ((2u * k + h) ^ (l16 & 7u)) << 4u;
  const LDSP u8 *lx_k = lx + l16 * 128u + x_col;
  const LDSP float *lsx_k = lsx + k * 64u + l16;

  u32x4 nb4 = h ? (wk >> 4u) & 0x0F0F0F0Fu : wk & 0x0F0F0F0Fu;
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

  const f8 sw = *(const LDSP f8 *)(lsw + k * INK_R2_G15_R + wr + 8u * h);
  acc[0] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  acc[1] += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
  __builtin_amdgcn_sched_barrier(0);

  /* Pair 1 (j = 2, 3): reuses b0/b1 and c0/c1 VGPRs. */
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
  __builtin_amdgcn_sched_barrier(0);
}

/* Single K-block IQ4_NL WMMA for 32-pair tile (WJ=2). */
static inline __attribute__((always_inline)) void ink_r2_g15_iq4_k32(
    const u32x4 wk, const LDSP u8 *lx, const LDSP float *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 l16, const u32 h, f8 acc[2]) {
  const v8i zm = (v8i)(0x4B400000);
  const u32 x_col = ((2u * k + h) ^ (l16 & 7u)) << 4u;
  const LDSP u8 *lx_k = lx + l16 * 128u + x_col;
  const LDSP float *lsx_k = lsx + k * 32u + l16;

  u32x4 nb4 = h ? (wk >> 4u) & 0x0F0F0F0Fu : wk & 0x0F0F0F0Fu;
  nb4.x = ie_iq4_map(nb4.x);
  nb4.y = ie_iq4_map(nb4.y);
  nb4.z = ie_iq4_map(nb4.z);
  nb4.w = ie_iq4_map(nb4.w);
  const v2i a0 = {(int)nb4.x, (int)nb4.y};
  const v2i a1 = {(int)nb4.z, (int)nb4.w};

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

  const f8 sw = *(const LDSP f8 *)(lsw + k * INK_R2_G15_R + wr + 8u * h);
  acc[0] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  acc[1] += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r2_g15_store_out(
    G u8 *Y, const G u8 *res, const LDSP u32 *lpid, const u32 nj, const u32 wj,
    const u32 rbase, const u32 wr, const u32 l16, const u32 h, const u32 rows,
    const u32 ys, const u32 rs, const f8 *acc) {
#pragma unroll
  for (u32 j = 0u; j < wj; j++) {
    const u32 slot = 16u * j + l16;
    if (slot >= nj) continue;
    const u32 pid = lpid[slot];
    G float *yr = (G float *)(Y + (unsigned long)pid * ys);
    const G float *rr = res ? (const G float *)(res + (unsigned long)pid * rs) : (const G float *)0;
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      const u32 r = rbase + wr + 8u * h + v;
      if (r < rows) yr[r] = epilogue(acc[j][v], r, 0, rr);
    }
  }
}

KERNEL ie_inkling_gemm_iq4nl_exp_r2(const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y, u32 rows, u32 nb,
                                    const G u32 *list, const G u32 *offs, u32 np, u32 qwn, u32 qsn, u32 xs, u32 ys,
                                    u32 xdiv, u32 kind, const G u8 *res, u32 rs) {
  static LDS ink_r2_g15_lds_t ink_r2_g15_lds;
  (void)kind;
  const u32 e = list ? tokid() : 0u;
  const u32 p0 = list ? offs[e] : tokid() * 64u;
  const u32 p1 = list ? offs[e + 1u] : (p0 + 64u < np ? p0 + 64u : np);
  const u32 rbase = wgid() * INK_R2_G15_R;
  if (p0 >= p1 || rbase >= rows || nb == 0u) return;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), l16 = l & 15u, h = l >> 4u;
  const u32 wr = 16u * w;
  const int is_sw = (w < 4u);

  const G u32 *qw = qw0 + (unsigned long)e * qwn;
  const G u16 *qs = qs0 + (unsigned long)e * qsn;

  /* Weight and weight-scale pointers are invariant across token tiles (pb). */
  const u32 rw_idx = rbase + wr + l16;
  const G u8 *wp = (const G u8 *)qw + (unsigned long)(rw_idx < rows ? rw_idx : rows - 1u) * nb * 16u;
  const u32 sr = rbase + (is_sw ? t0 : 0u);
  const u32 srr = sr < rows ? sr : rows - 1u;
  const G u16 *swp = qs + (unsigned long)srr * nb;

  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  const u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);

  for (u32 pb = p0; pb < p1; ) {
    if (pb > p0) lbarrier();
    const u32 rem = p1 - pb;

    if (rem > 32u) {
      /* ---------------- Adaptive 64-pair tile (128x64, WJ=4) ---------------- */
      const u32 nj = rem < 64u ? rem : 64u;
      const int is_sx = (w < 2u);
      if (is_sx) {
        const u32 pj = pb + (t0 < nj ? t0 : 0u);
        ink_r2_g15_lds.pid[t0] = list ? list[pj] : pj;
      }
      lbarrier();

      const G u8 *xp0 = X + (unsigned long)ink_r2_g15_div_x(ink_r2_g15_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
      const G u8 *xp1 = X + (unsigned long)ink_r2_g15_div_x(ink_r2_g15_lds.pid[tk_ld + 32u], xdiv) * xs + c8_ld * 16u;
      const G float *sxp = (const G float *)(X + (unsigned long)ink_r2_g15_div_x(ink_r2_g15_lds.pid[t0 & 63u], xdiv) * xs + 32ul * nb);

      f8 acc[4];
#pragma unroll
      for (u32 j = 0u; j < 4u; j++) acc[j] = (f8)(0.0f);

      u32x4 wa[INK_R2_G15_KB];
#pragma unroll
      for (u32 k = 0u; k < INK_R2_G15_KB; k++) wa[k] = *(const G u32x4 *)(wp + k * 16u);
      u32x4 rx0 = *(const G u32x4 *)xp0;
      u32x4 rx1 = *(const G u32x4 *)xp1;
      ink_r2_g15_f16x4 rsw = is_sw ? *(const G ink_r2_g15_f16x4 *)swp
                                   : (ink_r2_g15_f16x4){0.0f, 0.0f, 0.0f, 0.0f};
      f32x4 rsx = is_sx ? rq_load_f32x4(sxp) : (f32x4)(0.0f);

      ink_r2_g15_stage_x64(ink_r2_g15_lds.x[0], ink_r2_g15_lds.sw[0], ink_r2_g15_lds.sx[0],
                           st_x, t0, is_sw, is_sx, rx0, rx1, rsw, rsx);
      lbarrier();

      u32 bf = 0u;
#pragma unroll 1
      for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R2_G15_KB) {
        const u32 kb_n = kb0 + INK_R2_G15_KB;
        const int more = (kb_n < nb);
        const u32 nk = nb - kb0 < INK_R2_G15_KB ? nb - kb0 : INK_R2_G15_KB; /* separates k blocks for scheduler */
        if (more) {
          rx0 = *(const G u32x4 *)(xp0 + kb_n * 32u);
          rx1 = *(const G u32x4 *)(xp1 + kb_n * 32u);
          if (is_sw) rsw = *(const G ink_r2_g15_f16x4 *)(swp + kb_n);
          if (is_sx) rsx = rq_load_f32x4(sxp + kb_n);
        }
        __builtin_amdgcn_sched_barrier(0);
        const LDSP u8 *lx = ink_r2_g15_lds.x[bf];
        const LDSP float *lsw = ink_r2_g15_lds.sw[bf];
        const LDSP float *lsx = ink_r2_g15_lds.sx[bf];

        if (0u < nk) ink_r2_g15_iq4_k64(wa[0], lx, lsw, lsx, 0u, wr, l16, h, acc);
        if (1u < nk) ink_r2_g15_iq4_k64(wa[1], lx, lsw, lsx, 1u, wr, l16, h, acc);
        /* wa[0..1] are dead: reload next chunk's wa[0..1] in-place (zero wn[4] regs). */
        if (more) {
          wa[0] = *(const G u32x4 *)(wp + (kb_n + 0u) * 16u);
          wa[1] = *(const G u32x4 *)(wp + (kb_n + 1u) * 16u);
        }
        __builtin_amdgcn_sched_barrier(0);
        if (2u < nk) ink_r2_g15_iq4_k64(wa[2], lx, lsw, lsx, 2u, wr, l16, h, acc);
        if (3u < nk) ink_r2_g15_iq4_k64(wa[3], lx, lsw, lsx, 3u, wr, l16, h, acc);
        if (more) {
          wa[2] = *(const G u32x4 *)(wp + (kb_n + 2u) * 16u);
          wa[3] = *(const G u32x4 *)(wp + (kb_n + 3u) * 16u);
          const u32 nbf = bf ^ 1u;
          ink_r2_g15_stage_x64(ink_r2_g15_lds.x[nbf], ink_r2_g15_lds.sw[nbf], ink_r2_g15_lds.sx[nbf],
                               st_x, t0, is_sw, is_sx, rx0, rx1, rsw, rsx);
          lbarrier();
          bf = nbf;
        }
      }

      ink_r2_g15_store_out(Y, res, ink_r2_g15_lds.pid, nj, 4u, rbase, wr, l16, h, rows, ys, rs, acc);
      pb += 64u;
    } else {
      /* ---------------- Adaptive 32-pair tile (128x32, WJ=2) ---------------- */
      const u32 nj = rem;
      const int is_sx = (w == 0u);
      if (is_sx) {
        const u32 pj = pb + (t0 < nj ? t0 : 0u);
        ink_r2_g15_lds.pid[t0] = list ? list[pj] : pj;
      }
      lbarrier();

      const G u8 *xp0 = X + (unsigned long)ink_r2_g15_div_x(ink_r2_g15_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
      const G float *sxp = (const G float *)(X + (unsigned long)ink_r2_g15_div_x(ink_r2_g15_lds.pid[t0 & 31u], xdiv) * xs + 32ul * nb);

      f8 acc[2];
      acc[0] = (f8)(0.0f);
      acc[1] = (f8)(0.0f);

      u32x4 wa[INK_R2_G15_KB];
#pragma unroll
      for (u32 k = 0u; k < INK_R2_G15_KB; k++) wa[k] = *(const G u32x4 *)(wp + k * 16u);
      u32x4 rx0 = *(const G u32x4 *)xp0;
      ink_r2_g15_f16x4 rsw = is_sw ? *(const G ink_r2_g15_f16x4 *)swp
                                   : (ink_r2_g15_f16x4){0.0f, 0.0f, 0.0f, 0.0f};
      f32x4 rsx = is_sx ? rq_load_f32x4(sxp) : (f32x4)(0.0f);

      ink_r2_g15_stage_x32(ink_r2_g15_lds.x[0], ink_r2_g15_lds.sw[0], ink_r2_g15_lds.sx[0],
                           st_x, t0, is_sw, is_sx, rx0, rsw, rsx);
      lbarrier();

      u32 bf = 0u;
#pragma unroll 1
      for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R2_G15_KB) {
        const u32 kb_n = kb0 + INK_R2_G15_KB;
        const int more = (kb_n < nb);
        const u32 nk = nb - kb0 < INK_R2_G15_KB ? nb - kb0 : INK_R2_G15_KB;
        if (more) {
          rx0 = *(const G u32x4 *)(xp0 + kb_n * 32u);
          if (is_sw) rsw = *(const G ink_r2_g15_f16x4 *)(swp + kb_n);
          if (is_sx) rsx = rq_load_f32x4(sxp + kb_n);
        }
        __builtin_amdgcn_sched_barrier(0);
        const LDSP u8 *lx = ink_r2_g15_lds.x[bf];
        const LDSP float *lsw = ink_r2_g15_lds.sw[bf];
        const LDSP float *lsx = ink_r2_g15_lds.sx[bf];

        if (0u < nk) ink_r2_g15_iq4_k32(wa[0], lx, lsw, lsx, 0u, wr, l16, h, acc);
        if (1u < nk) ink_r2_g15_iq4_k32(wa[1], lx, lsw, lsx, 1u, wr, l16, h, acc);
        if (more) {
          wa[0] = *(const G u32x4 *)(wp + (kb_n + 0u) * 16u);
          wa[1] = *(const G u32x4 *)(wp + (kb_n + 1u) * 16u);
        }
        __builtin_amdgcn_sched_barrier(0);
        if (2u < nk) ink_r2_g15_iq4_k32(wa[2], lx, lsw, lsx, 2u, wr, l16, h, acc);
        if (3u < nk) ink_r2_g15_iq4_k32(wa[3], lx, lsw, lsx, 3u, wr, l16, h, acc);
        if (more) {
          wa[2] = *(const G u32x4 *)(wp + (kb_n + 2u) * 16u);
          wa[3] = *(const G u32x4 *)(wp + (kb_n + 3u) * 16u);
          const u32 nbf = bf ^ 1u;
          ink_r2_g15_stage_x32(ink_r2_g15_lds.x[nbf], ink_r2_g15_lds.sw[nbf], ink_r2_g15_lds.sx[nbf],
                               st_x, t0, is_sw, is_sx, rx0, rsw, rsx);
          lbarrier();
          bf = nbf;
        }
      }

      ink_r2_g15_store_out(Y, res, ink_r2_g15_lds.pid, nj, 2u, rbase, wr, l16, h, rows, ys, rs, acc);
      pb += 32u;
    }
  }
}

#undef INK_R2_G15_KB
#undef INK_R2_G15_R
