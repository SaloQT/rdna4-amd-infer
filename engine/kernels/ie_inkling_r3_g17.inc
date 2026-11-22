/* ============================================================================
 * Inkling-2 Round 3 Group 17 (g17-iq3xxs-pf-r3): Low-VGPR Adaptive 128x64 /
 * 128x32 IQ3_XXS MoE Prefill WMMA GEMM for RX 9070 XT (gfx1201, wave32, NT=256).
 *
 * Target: MoE ffn_gate_up_exps IQ3_XXS (2304x2304x128, nb=72, K=8; ~39% of R2
 *         prefill time at 230 VGPR / 1 WG/CU in r1).
 * Bitmask: IE_INKLING_KERNEL_MASK bit 36 (1ULL << 36).
 *
 * Exported Kernel, Grid, Guard, and Parity:
 * - KERNEL ie_inkling_gemm_rq_exp_iq3xxs_r3(const G u8 *qw, const G u8 *X,
 *     G float *Y, u32 rows, u32 nb, const G u32 *list, const G u32 *offs,
 *     u32 np, u32 q_bytes, u32 xs, u32 ys, u32 xdiv, const G float *res,
 *     u32 rs)
 *   - Replaces: ie_inkling_gemm_rq_exp_iq3xxs_r1 / ie_gemm_rq_exp_iq3xxs
 *   - Grid:     ((rows + 127u) / 128u, n_exp, 1u)  [e.g. (18, 128, 1) = 2304 WGs]
 *   - Guard:    rows > 0u && (rows & 31u) == 0u && nb >= 8u && (nb & 7u) == 0u
 *   - Parity:   bitwise-exact (0 bit diffs) vs ie_gemm_rq_exp_iq3xxs / r1
 *
 * Key R3 Optimizations (combining g03 adaptive tiling + g15/g09 low-VGPR):
 * - Adaptive 128x64 (rem > 32, WJ=4) / 128x32 (rem <= 32, WJ=2) token tiling:
 *   halves IQ3_XXS weight reads & decode work on >32-pair experts while keeping
 *   zero wasted WMMAs/X loads on <=32-pair experts.
 * - Strictly sequenced two-pair WMMA micro-loop (j=0..1 then j=2..3) with
 *   sched_barrier(0) between A-loads, WMMAs, dx-loads, cvt+mul, and FMAs so
 *   a0/a1, r0/r1, and dx_lo/dx_hi VGPRs are reused without hoisting.
 * - 2-k batched decode (xdc[2]) + in-place dead-weight reload into cur_q/cur_a
 *   right after k=2..3 decode (single weight buffer, zero hb1_q/q1/nxt_q/carry
 *   copies, overlapped with k=2..3 WMMAs + FMAs).
 * - SGPR-pinned nk block-separation guards (if (k < nk)) prevent LLVM from
 *   merging unrolled K steps into one register-spilling basic block.
 * - 19,712 B XOR-swizzled bank-conflict-free double-buffered LDS + lbarrier()
 *   -> <= 160 VGPR, 0 spills, 2 WGs/CU.
 * ============================================================================ */

struct __attribute__((packed, aligned(4))) ink_r3_g17_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r3_g17_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r3_g17_u32x4p *)p)->v;
}

typedef struct {
  v2i w0, w1;
  float s0;
} ink_r3_g17_dc_t;

typedef struct {
  u8 x[2][64u * 128u] __attribute__((aligned(16)));
  float sx[2][4u * 64u] __attribute__((aligned(32)));
  u32 lut[256] __attribute__((aligned(16)));
  u32 pid[64] __attribute__((aligned(16)));
} ink_r3_g17_exp_lds_t;

_Static_assert(sizeof(ink_r3_g17_exp_lds_t) == 19712u, "ink_r3_g17_exp_lds_t must be 19,712 B");

static inline __attribute__((always_inline)) u32 ink_r3_g17_perm_al(const u32 lo, const u32 hi, const u32 psel) {
  return __builtin_amdgcn_perm(hi, lo, psel);
}

static inline __attribute__((always_inline)) u32 ink_r3_g17_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

/* Load IQ3_XXS hb0 (base[0..8] = d+qs_hb0, base[16..20] = aux_hb0) in 5 global loads.
 * base[8] and base[20] are carried forward into cur_q[0]/cur_a[0] for hb1. */
static inline __attribute__((always_inline)) void ink_r3_g17_load_hb0(const G u8 *p, u32 qw[9], u32 aw[5]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v0 = ink_r3_g17_ld4(base + 0u);
  const u32x4 v1 = ink_r3_g17_ld4(base + 4u);
  qw[0] = v0.x; qw[1] = v0.y; qw[2] = v0.z; qw[3] = v0.w;
  qw[4] = v1.x; qw[5] = v1.y; qw[6] = v1.z; qw[7] = v1.w;
  qw[8] = base[8];
  const u32x4 a0 = ink_r3_g17_ld4(base + 16u);
  aw[0] = a0.x; aw[1] = a0.y; aw[2] = a0.z; aw[3] = a0.w;
  aw[4] = base[20];
}

/* Load IQ3_XXS hb1 tail (base[9..16] = qs_hb1, base[21..24] = aux_hb1) in 3x global_load_b128. */
static inline __attribute__((always_inline)) void ink_r3_g17_load_hb1_tail(const G u8 *p, u32 qt[8], u32 at[4]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v0 = ink_r3_g17_ld4(base + 9u);
  const u32x4 v1 = ink_r3_g17_ld4(base + 13u);
  qt[0] = v0.x; qt[1] = v0.y; qt[2] = v0.z; qt[3] = v0.w;
  qt[4] = v1.x; qt[5] = v1.y; qt[6] = v1.z; qt[7] = v1.w;
  const u32x4 a0 = ink_r3_g17_ld4(base + 21u);
  at[0] = a0.x; at[1] = a0.y; at[2] = a0.z; at[3] = a0.w;
}

/* 2-k batched IQ3_XXS decode (k = k0, k0+1) overlapping 8 LUT ds_load_b32.
 * Caller passes &cur_q[2*k0] and &cur_a[k0]. */
static inline __attribute__((always_inline)) void ink_r3_g17_decode_pair(
    const u32 *qw, const u32 *aw, const u32 psel, const float d2, const LDSP u32 *lut,
    const u32 h, ink_r3_g17_dc_t dc[2]) {
  const u32 psel_q = h ? 0x07060302u : 0x05040100u;
  u32 auxv[2], g00[2], g01[2], g10[2], g11[2];
#pragma unroll
  for (u32 dk = 0u; dk < 2u; dk++) {
    auxv[dk] = ink_r3_g17_perm_al(aw[dk], aw[dk + 1u], psel);
    const u32 w0 = ink_r3_g17_perm_al(qw[2u * dk], qw[2u * dk + 1u], psel);
    const u32 w1 = ink_r3_g17_perm_al(qw[2u * dk + 1u], qw[2u * dk + 2u], psel);
    const u32 qp = __builtin_amdgcn_perm(w1, w0, psel_q);
    g00[dk] = lut[qp & 255u];
    g01[dk] = lut[(qp >> 8u) & 255u];
    g10[dk] = lut[(qp >> 16u) & 255u];
    g11[dk] = lut[qp >> 24u];
  }
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for (u32 dk = 0u; dk < 2u; dk++) {
    dc[dk].s0 = d2 * (0.5f + (float)(auxv[dk] >> 28u));
  }
#pragma unroll
  for (u32 dk = 0u; dk < 2u; dk++) {
    const u32 s8_0 = rq_ksign7((auxv[dk] >> (7u * h)) & 127u);
    const u32 s8_1 = rq_ksign7((auxv[dk] >> (7u * (2u + h))) & 127u);
    const v2i v0 = {(int)ie_rq_ssign(g00[dk], ie_rq_mask4(s8_0 & 15u)),
                    (int)ie_rq_ssign(g01[dk], ie_rq_mask4(s8_0 >> 4u))};
    const v2i v1 = {(int)ie_rq_ssign(g10[dk], ie_rq_mask4(s8_1 & 15u)),
                    (int)ie_rq_ssign(g11[dk], ie_rq_mask4(s8_1 >> 4u))};
    dc[dk].w0 = v0;
    dc[dk].w1 = v1;
  }
}

/* Strictly sequenced 2-tile (j0, j0+1) WMMA + exact FMA epilogue.
 * Reuses a0/a1, r0/r1, and dx_lo/dx_hi VGPRs cleanly across pairs; overlaps
 * dx_lo/dx_hi LDS loads with (float)r[v] * s0 conversion+scaling. */
static inline __attribute__((always_inline)) void ink_r3_g17_apply_pair_j(
    const ink_r3_g17_dc_t dck, const LDSP u8 *lx_r, const LDSP float *lsx_k,
    const u32 c0, const u32 c1, const u32 j0, float acc[][8]) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  const u32 j_off = j0 * 2048u;
  const v2i a0_0 = *(const LDSP v2i *)(lx_r + c0 + j_off);
  const v2i a0_1 = *(const LDSP v2i *)(lx_r + c0 + j_off + 2048u);
  const v2i a1_0 = *(const LDSP v2i *)(lx_r + c1 + j_off);
  const v2i a1_1 = *(const LDSP v2i *)(lx_r + c1 + j_off + 2048u);
  __builtin_amdgcn_sched_barrier(0);

  v8i r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_0, 1, dck.w0, z, 0);
  v8i r1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_1, 1, dck.w0, z, 0);
  r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_0, 1, dck.w1, r0, 0);
  r1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_1, 1, dck.w1, r1, 0);
  __builtin_amdgcn_sched_barrier(0);

  const float s0 = dck.s0;
  const LDSP float *sx0 = lsx_k + 16u * j0;
  f32x4 dx_lo = *(const LDSP f32x4 *)(sx0);
  f32x4 dx_hi = *(const LDSP f32x4 *)(sx0 + 4u);
  __builtin_amdgcn_sched_barrier(0);
  float t0[8];
#pragma unroll
  for (u32 v = 0u; v < 8u; v++) t0[v] = (float)r0[v] * s0;
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for (u32 v = 0u; v < 4u; v++) acc[j0][v] += t0[v] * dx_lo[v];
#pragma unroll
  for (u32 v = 0u; v < 4u; v++) acc[j0][4u + v] += t0[4u + v] * dx_hi[v];
  __builtin_amdgcn_sched_barrier(0);

  dx_lo = *(const LDSP f32x4 *)(sx0 + 16u);
  dx_hi = *(const LDSP f32x4 *)(sx0 + 20u);
  __builtin_amdgcn_sched_barrier(0);
  float t1[8];
#pragma unroll
  for (u32 v = 0u; v < 8u; v++) t1[v] = (float)r1[v] * s0;
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for (u32 v = 0u; v < 4u; v++) acc[j0 + 1u][v] += t1[v] * dx_lo[v];
#pragma unroll
  for (u32 v = 0u; v < 4u; v++) acc[j0 + 1u][4u + v] += t1[4u + v] * dx_hi[v];
  __builtin_amdgcn_sched_barrier(0);
}

/* Single K-block IQ3_XXS WMMA for 64-pair tile (WJ=4), split into two
 * sequenced 2-tile pairs (j=0..1, then j=2..3). */
static inline __attribute__((always_inline)) void ink_r3_g17_apply_k64(
    const ink_r3_g17_dc_t dck, const LDSP u8 *lx_r, const LDSP float *lsx_h,
    const u32 k, const u32 r16, float acc[4][8]) {
  const u32 c0 = ((2u * k) ^ (r16 & 7u)) << 4u;
  const u32 c1 = c0 ^ 16u;
  const LDSP float *lsx_k = lsx_h + k * 64u;
  ink_r3_g17_apply_pair_j(dck, lx_r, lsx_k, c0, c1, 0u, acc);
  ink_r3_g17_apply_pair_j(dck, lx_r, lsx_k, c0, c1, 2u, acc);
}

/* Single K-block IQ3_XXS WMMA for 32-pair tile (WJ=2, j=0..1). */
static inline __attribute__((always_inline)) void ink_r3_g17_apply_k32(
    const ink_r3_g17_dc_t dck, const LDSP u8 *lx_r, const LDSP float *lsx_h,
    const u32 k, const u32 r16, float acc[2][8]) {
  const u32 c0 = ((2u * k) ^ (r16 & 7u)) << 4u;
  const u32 c1 = c0 ^ 16u;
  const LDSP float *lsx_k = lsx_h + k * 32u;
  ink_r3_g17_apply_pair_j(dck, lx_r, lsx_k, c0, c1, 0u, acc);
}

static inline __attribute__((always_inline)) void ink_r3_g17_stage_x64(
    LDSP u8 *lx, LDSP float *lsx, const u32 st_x, const u32 t0, const int is_s,
    const u32x4 rx0, const u32x4 rx1, const f32x4 rsx) {
  *(LDSP u32x4 *)(lx + st_x) = rx0;
  *(LDSP u32x4 *)(lx + st_x + 4096u) = rx1;
  if (is_s) {
    lsx[t0] = rsx.x;
    lsx[64u + t0] = rsx.y;
    lsx[128u + t0] = rsx.z;
    lsx[192u + t0] = rsx.w;
  }
}

static inline __attribute__((always_inline)) void ink_r3_g17_stage_x32(
    LDSP u8 *lx, LDSP float *lsx, const u32 st_x, const u32 t0, const int is_s,
    const u32x4 rx0, const f32x4 rsx) {
  *(LDSP u32x4 *)(lx + st_x) = rx0;
  if (is_s) {
    lsx[t0] = rsx.x;
    lsx[32u + t0] = rsx.y;
    lsx[64u + t0] = rsx.z;
    lsx[96u + t0] = rsx.w;
  }
}

static inline __attribute__((always_inline)) void ink_r3_g17_store_exp(
    G float *Y, const G float *res, const LDSP u32 *lpid, const u32 nj, const u32 wj,
    const u32 row, const u32 h, const u32 ys, const u32 rs, float acc[][8]) {
  if (res) {
#pragma unroll
    for (u32 j = 0u; j < wj; j++) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 slot = 16u * j + 8u * h + v;
        if (slot < nj) {
          const u32 pid = lpid[slot];
          G float *yp = (G float *)((G u8 *)Y + (unsigned long)pid * ys);
          const G float *rp = (const G float *)((const G u8 *)res + (unsigned long)pid * rs);
          yp[row] = rp[row] + acc[j][v];
        }
      }
    }
  } else {
#pragma unroll
    for (u32 j = 0u; j < wj; j++) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 slot = 16u * j + 8u * h + v;
        if (slot < nj) {
          const u32 pid = lpid[slot];
          G float *yp = (G float *)((G u8 *)Y + (unsigned long)pid * ys);
          yp[row] = acc[j][v];
        }
      }
    }
  }
}

KERNEL ie_inkling_gemm_rq_exp_iq3xxs_r3(const G u8 *qw, const G u8 *X, G float *Y, u32 rows, u32 nb,
                                        const G u32 *list, const G u32 *offs, u32 np, u32 q_bytes,
                                        u32 xs, u32 ys, u32 xdiv, const G float *res, u32 rs) {
  static LDS ink_r3_g17_exp_lds_t ink_r3_g17_exp_lds;
  if (rows == 0u || (rows & 31u) != 0u || nb < 8u || (nb & 7u) != 0u || !qw || !X || !Y) return;
  const u32 e = tokid();
  const u32 p0 = list ? offs[e] : 0u;
  const u32 p1 = list ? offs[e + 1u] : np;
  const u32 rbase = wgid() * 128u;
  if (p0 >= p1 || rbase >= rows) return;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;
  const u32 row = rbase + 16u * w + r16;
  const u32 rr = row < rows ? row : rows - 1u;
  const u32 mb = nb >> 3u;
  const G u8 *rowp = qw + (unsigned long)e * q_bytes + (unsigned long)rr * mb * 98u;
  const u32 init_psel = ((unsigned long)rowp & 2ul) ? 0x07060504u : 0x05040302u;

  /* Stage 1 KiB IQ3_XXS grid LUT (synchronized by first tile's pid lbarrier). */
  ink_r3_g17_exp_lds.lut[t0] = ie_rq_iq3xxs_grid[t0];

  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  const u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);
  const u32 x_r_off = r16 * 128u + 8u * h;

  for (u32 pb = p0; pb < p1; ) {
    if (pb > p0) lbarrier();
    const u32 rem = p1 - pb;
    if (rem > 32u) {
      /* ---------------- Adaptive 64-pair tile (128x64, WJ=4) ---------------- */
      const u32 nj = rem < 64u ? rem : 64u;
      const int is_s = (w < 2u);
      if (is_s) {
        const u32 pj = pb + (t0 < nj ? t0 : 0u);
        ink_r3_g17_exp_lds.pid[t0] = list ? list[pj] : pj;
      }
      lbarrier();

      const G u8 *xp0 = X + (unsigned long)ink_r3_g17_div_x(ink_r3_g17_exp_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
      const G u8 *xp1 = X + (unsigned long)ink_r3_g17_div_x(ink_r3_g17_exp_lds.pid[tk_ld + 32u], xdiv) * xs + c8_ld * 16u;
      const G float *sxp = (const G float *)(X + (unsigned long)ink_r3_g17_div_x(ink_r3_g17_exp_lds.pid[t0 & 63u], xdiv) * xs + 32ul * nb);

      float acc[4][8];
#pragma unroll
      for (u32 j = 0u; j < 4u; j++) {
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) acc[j][v] = 0.0f;
      }

      u32x4 rx0 = ink_r3_g17_ld4(xp0);
      u32x4 rx1 = ink_r3_g17_ld4(xp1);
      f32x4 rsx = is_s ? rq_load_f32x4(sxp) : (f32x4)(0.0f);
      const G u8 *bp = rowp;
      u32 cur_psel = init_psel;
      u32 cur_q[9], cur_a[5];
      ink_r3_g17_load_hb0(bp, cur_q, cur_a);
      ink_r3_g17_stage_x64(ink_r3_g17_exp_lds.x[0], ink_r3_g17_exp_lds.sx[0], st_x, t0, is_s, rx0, rx1, rsx);
      lbarrier();

      const LDSP u8 *lx0_r = ink_r3_g17_exp_lds.x[0] + x_r_off;
      const LDSP u8 *lx1_r = ink_r3_g17_exp_lds.x[1] + x_r_off;
      const LDSP float *lsx0_h = ink_r3_g17_exp_lds.sx[0] + 8u * h;
      const LDSP float *lsx1_h = ink_r3_g17_exp_lds.sx[1] + 8u * h;

#pragma unroll 1
      for (u32 sb = 0u; ; sb++) {
        const u32 kb1 = 8u * sb + 4u;
        /* SGPR-pinned block-separation guard prevents LLVM from merging K blocks. */
        u32 nk = nb - 8u * sb < 4u ? nb - 8u * sb : 4u;
        __asm__ volatile("" : "+s"(nk));

        /* hb0: prefetch hb1 activations FIRST (FIFO ahead of hb1_tail). */
        rx0 = ink_r3_g17_ld4(xp0 + kb1 * 32u);
        rx1 = ink_r3_g17_ld4(xp1 + kb1 * 32u);
        if (is_s) rsx = rq_load_f32x4(sxp + kb1);
        __builtin_amdgcn_sched_barrier(0);

        const float d2 = ie_rq_h2f((cur_q[0] >> ((cur_psel & 4u) << 2u)) & 0xFFFFu) * 0.5f;
        ink_r3_g17_dc_t xdc[2];

        /* hb0 k = 0, 1 */
        ink_r3_g17_decode_pair(cur_q, cur_a, cur_psel, d2, ink_r3_g17_exp_lds.lut, h, xdc);
        __builtin_amdgcn_sched_barrier(0);
        if (0u < nk) ink_r3_g17_apply_k64(xdc[0], lx0_r, lsx0_h, 0u, r16, acc);
        if (1u < nk) ink_r3_g17_apply_k64(xdc[1], lx0_r, lsx0_h, 1u, r16, acc);

        /* hb0 k = 2, 3: cur_q/cur_a die right after decode -> in-place hb1_tail reload! */
        ink_r3_g17_decode_pair(cur_q + 4u, cur_a + 2u, cur_psel, d2, ink_r3_g17_exp_lds.lut, h, xdc);
        __builtin_amdgcn_sched_barrier(0);
        cur_q[0] = cur_q[8];
        cur_a[0] = cur_a[4];
        ink_r3_g17_load_hb1_tail(bp, cur_q + 1u, cur_a + 1u);
        __builtin_amdgcn_sched_barrier(0);
        if (2u < nk) ink_r3_g17_apply_k64(xdc[0], lx0_r, lsx0_h, 2u, r16, acc);
        if (3u < nk) ink_r3_g17_apply_k64(xdc[1], lx0_r, lsx0_h, 3u, r16, acc);

        ink_r3_g17_stage_x64(ink_r3_g17_exp_lds.x[1], ink_r3_g17_exp_lds.sx[1], st_x, t0, is_s, rx0, rx1, rsx);
        lbarrier();

        /* hb1: cur_q/cur_a already hold carried base[8]/base[20] + hb1_tail. */
        const u32 psel_hb1 = cur_psel;
        bp += 98u;
        cur_psel ^= 0x02020606u;
        const int more = (sb + 1u < mb);
        if (more) {
          const u32 kbn = 8u * (sb + 1u);
          rx0 = ink_r3_g17_ld4(xp0 + kbn * 32u);
          rx1 = ink_r3_g17_ld4(xp1 + kbn * 32u);
          if (is_s) rsx = rq_load_f32x4(sxp + kbn);
        }
        __builtin_amdgcn_sched_barrier(0);

        /* hb1 k = 0, 1 */
        ink_r3_g17_decode_pair(cur_q, cur_a, psel_hb1, d2, ink_r3_g17_exp_lds.lut, h, xdc);
        __builtin_amdgcn_sched_barrier(0);
        if (0u < nk) ink_r3_g17_apply_k64(xdc[0], lx1_r, lsx1_h, 0u, r16, acc);
        if (1u < nk) ink_r3_g17_apply_k64(xdc[1], lx1_r, lsx1_h, 1u, r16, acc);

        /* hb1 k = 2, 3: cur_q/cur_a die right after decode -> in-place next-SB hb0 reload! */
        ink_r3_g17_decode_pair(cur_q + 4u, cur_a + 2u, psel_hb1, d2, ink_r3_g17_exp_lds.lut, h, xdc);
        __builtin_amdgcn_sched_barrier(0);
        if (more) ink_r3_g17_load_hb0(bp, cur_q, cur_a);
        __builtin_amdgcn_sched_barrier(0);
        if (2u < nk) ink_r3_g17_apply_k64(xdc[0], lx1_r, lsx1_h, 2u, r16, acc);
        if (3u < nk) ink_r3_g17_apply_k64(xdc[1], lx1_r, lsx1_h, 3u, r16, acc);

        if (!more) break;
        ink_r3_g17_stage_x64(ink_r3_g17_exp_lds.x[0], ink_r3_g17_exp_lds.sx[0], st_x, t0, is_s, rx0, rx1, rsx);
        lbarrier();
      }

      if (row < rows) {
        ink_r3_g17_store_exp(Y, res, ink_r3_g17_exp_lds.pid, nj, 4u, row, h, ys, rs, acc);
      }
      pb += 64u;
    } else {
      /* ---------------- Adaptive 32-pair tile (128x32, WJ=2) ---------------- */
      const u32 nj = rem;
      const int is_s = (w == 0u);
      if (is_s) {
        const u32 pj = pb + (t0 < nj ? t0 : 0u);
        ink_r3_g17_exp_lds.pid[t0] = list ? list[pj] : pj;
      }
      lbarrier();

      const G u8 *xp0 = X + (unsigned long)ink_r3_g17_div_x(ink_r3_g17_exp_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
      const G float *sxp = (const G float *)(X + (unsigned long)ink_r3_g17_div_x(ink_r3_g17_exp_lds.pid[t0 & 31u], xdiv) * xs + 32ul * nb);

      float acc[2][8];
#pragma unroll
      for (u32 j = 0u; j < 2u; j++) {
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) acc[j][v] = 0.0f;
      }

      u32x4 rx0 = ink_r3_g17_ld4(xp0);
      f32x4 rsx = is_s ? rq_load_f32x4(sxp) : (f32x4)(0.0f);
      const G u8 *bp = rowp;
      u32 cur_psel = init_psel;
      u32 cur_q[9], cur_a[5];
      ink_r3_g17_load_hb0(bp, cur_q, cur_a);
      ink_r3_g17_stage_x32(ink_r3_g17_exp_lds.x[0], ink_r3_g17_exp_lds.sx[0], st_x, t0, is_s, rx0, rsx);
      lbarrier();

      const LDSP u8 *lx0_r = ink_r3_g17_exp_lds.x[0] + x_r_off;
      const LDSP u8 *lx1_r = ink_r3_g17_exp_lds.x[1] + x_r_off;
      const LDSP float *lsx0_h = ink_r3_g17_exp_lds.sx[0] + 8u * h;
      const LDSP float *lsx1_h = ink_r3_g17_exp_lds.sx[1] + 8u * h;

#pragma unroll 1
      for (u32 sb = 0u; ; sb++) {
        const u32 kb1 = 8u * sb + 4u;
        u32 nk = nb - 8u * sb < 4u ? nb - 8u * sb : 4u;
        __asm__ volatile("" : "+s"(nk));

        rx0 = ink_r3_g17_ld4(xp0 + kb1 * 32u);
        if (is_s) rsx = rq_load_f32x4(sxp + kb1);
        __builtin_amdgcn_sched_barrier(0);

        const float d2 = ie_rq_h2f((cur_q[0] >> ((cur_psel & 4u) << 2u)) & 0xFFFFu) * 0.5f;
        ink_r3_g17_dc_t xdc[2];

        /* hb0 k = 0, 1 */
        ink_r3_g17_decode_pair(cur_q, cur_a, cur_psel, d2, ink_r3_g17_exp_lds.lut, h, xdc);
        __builtin_amdgcn_sched_barrier(0);
        if (0u < nk) ink_r3_g17_apply_k32(xdc[0], lx0_r, lsx0_h, 0u, r16, acc);
        if (1u < nk) ink_r3_g17_apply_k32(xdc[1], lx0_r, lsx0_h, 1u, r16, acc);

        /* hb0 k = 2, 3 */
        ink_r3_g17_decode_pair(cur_q + 4u, cur_a + 2u, cur_psel, d2, ink_r3_g17_exp_lds.lut, h, xdc);
        __builtin_amdgcn_sched_barrier(0);
        cur_q[0] = cur_q[8];
        cur_a[0] = cur_a[4];
        ink_r3_g17_load_hb1_tail(bp, cur_q + 1u, cur_a + 1u);
        __builtin_amdgcn_sched_barrier(0);
        if (2u < nk) ink_r3_g17_apply_k32(xdc[0], lx0_r, lsx0_h, 2u, r16, acc);
        if (3u < nk) ink_r3_g17_apply_k32(xdc[1], lx0_r, lsx0_h, 3u, r16, acc);

        ink_r3_g17_stage_x32(ink_r3_g17_exp_lds.x[1], ink_r3_g17_exp_lds.sx[1], st_x, t0, is_s, rx0, rsx);
        lbarrier();

        const u32 psel_hb1 = cur_psel;
        bp += 98u;
        cur_psel ^= 0x02020606u;
        const int more = (sb + 1u < mb);
        if (more) {
          const u32 kbn = 8u * (sb + 1u);
          rx0 = ink_r3_g17_ld4(xp0 + kbn * 32u);
          if (is_s) rsx = rq_load_f32x4(sxp + kbn);
        }
        __builtin_amdgcn_sched_barrier(0);

        /* hb1 k = 0, 1 */
        ink_r3_g17_decode_pair(cur_q, cur_a, psel_hb1, d2, ink_r3_g17_exp_lds.lut, h, xdc);
        __builtin_amdgcn_sched_barrier(0);
        if (0u < nk) ink_r3_g17_apply_k32(xdc[0], lx1_r, lsx1_h, 0u, r16, acc);
        if (1u < nk) ink_r3_g17_apply_k32(xdc[1], lx1_r, lsx1_h, 1u, r16, acc);

        /* hb1 k = 2, 3 */
        ink_r3_g17_decode_pair(cur_q + 4u, cur_a + 2u, psel_hb1, d2, ink_r3_g17_exp_lds.lut, h, xdc);
        __builtin_amdgcn_sched_barrier(0);
        if (more) ink_r3_g17_load_hb0(bp, cur_q, cur_a);
        __builtin_amdgcn_sched_barrier(0);
        if (2u < nk) ink_r3_g17_apply_k32(xdc[0], lx1_r, lsx1_h, 2u, r16, acc);
        if (3u < nk) ink_r3_g17_apply_k32(xdc[1], lx1_r, lsx1_h, 3u, r16, acc);

        if (!more) break;
        ink_r3_g17_stage_x32(ink_r3_g17_exp_lds.x[0], ink_r3_g17_exp_lds.sx[0], st_x, t0, is_s, rx0, rsx);
        lbarrier();
      }

      if (row < rows) {
        ink_r3_g17_store_exp(Y, res, ink_r3_g17_exp_lds.pid, nj, 2u, row, h, ys, rs, acc);
      }
      pb += 32u;
    }
  }
}
