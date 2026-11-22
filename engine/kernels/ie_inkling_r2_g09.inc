/* ============================================================================
 * Inkling-2 R2 Group 09: Low-VGPR IQ3_XXS Prefill GEMM & Decode Fused
 * IQ3_XXS+SITU+Q8 MoE Kernels (gfx1201, wave32, NT=256)
 *
 * Exports (included into kernels/ie_inkling_kernels.c after ie_kernels.c):
 *   1. KERNEL ie_inkling_gemm_rq_exp_iq3xxs_r2(const G u8 *qw, const G u8 *X,
 *        G float *Y, u32 rows, u32 nb, const G u32 *list, const G u32 *offs,
 *        u32 np, u32 q_bytes, u32 xs, u32 ys, u32 xdiv, const G float *res,
 *        u32 rs)                                                     [bit 18]
 *      - Replaces: ie_inkling_gemm_rq_exp_iq3xxs_r1 (prefill MoE IQ3_XXS)
 *      - Grid:     ((rows + 127u) / 128u, n_exp, 1u)  [e.g. (18, 128, 1)]
 *      - Guard:    rows > 0u && (rows & 31u) == 0u && nb >= 8u && (nb & 7u) == 0u
 *      - Parity:   bitwise-exact (0 bit diffs) vs ie_gemm_rq_exp_iq3xxs
 *      - Opts:     Dedicated 128x32 (acc[2][8]) + 2-k batched decode (xdc[2]) +
 *                  dead-weight register reuse (hb1_tail / next-hb0 issued only
 *                  after k=2..3 decode frees cur_q/cur_a) + 10.1 KiB XOR-swizzled
 *                  double-buffered LDS + lbarrier() -> <= 160 VGPR (2 WGs/CU).
 *
 *   2. KERNEL ie_inkling_gexp_iq3xxs_situ_r2(const G u8 *qw, const G u8 *xq,
 *        G float *b, G u8 *q, u32 eff, u32 nb, const G u32 *route, u32 q_bytes,
 *        u32 astr, u32 bstr, u32 qstr, float beta, float lbeta)      [bit 19]
 *      - Replaces: T=1 decode OP_GEMV_EXP(egu IQ3_XXS [2*eff, nb*32]) + OP_SITU
 *      - Grid:     ((eff + 31u) / 32u, K, 1u)         [e.g. (36, 8, 1) = 288 WGs]
 *      - Guard:    T == 1u && type == IE_RQ_IQ3_XXS && eff > 0u &&
 *                  (eff & 31u) == 0u && rows == 2u * eff && nb >= 8u &&
 *                  (nb & 7u) == 0u && (b != NULL || q != NULL)
 *      - Parity:   100% bitwise-exact (0 bit diffs) vs ie_gexp_raw + ie_situ
 *                  (exact 4-acc wave_tree_f order, #pragma clang fp contract(off),
 *                  ie_tanhf_s / sigmoidf / quant_wave).
 * ========================================================================== */

struct __attribute__((packed, aligned(4))) ink_r2_g09_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r2_g09_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r2_g09_u32x4p *)p)->v;
}

#define INK_R2_G09_PIN_ACC8(a)                                                                                     \
  __asm__ volatile(""                                                                                              \
                   : "+v"((a)[0]), "+v"((a)[1]), "+v"((a)[2]), "+v"((a)[3]), "+v"((a)[4]), "+v"((a)[5]),           \
                     "+v"((a)[6]), "+v"((a)[7]))

/* ============================================================================
 * Kernel 1: T>1 Prefill MoE IQ3_XXS WMMA GEMM (Dedicated 128x32, <= 160 VGPR)
 *   - Dedicated 32-pair tile (acc[2][8] = 16 VGPRs, no duplicated 64-pair path).
 *   - 2-k batched decode (xdc[2] = 10 VGPRs) halves decode register pressure.
 *   - Next-half weight loads (hb1_tail / next-hb0) issue immediately after
 *     k=2..3 decode, when cur_q/cur_a are completely dead -> zero overlap of
 *     cur and next weight buffers in VGPRs while still overlapping global loads
 *     with k=2..3 WMMAs and FMA epilogues.
 *   - LDS: 10,368 B (swizzled double-buffered 32-tok lx + lsx + 1 KiB lut +
 *     pid[32]) -> <= 10.2 KiB LDS, lbarrier() only.
 * ========================================================================== */

typedef struct {
  v2i w0, w1;
  float s0;
} ink_r2_g09_dc_t;

typedef struct {
  u8 x[2][32u * 128u] __attribute__((aligned(16)));
  float sx[2][4u * 32u] __attribute__((aligned(32)));
  u32 lut[256] __attribute__((aligned(16)));
  u32 pid[32] __attribute__((aligned(16)));
} ink_r2_g09_exp_lds_t;

_Static_assert(sizeof(ink_r2_g09_exp_lds_t) == 10368u, "ink_r2_g09_exp_lds_t must be 10,368 B");

static inline __attribute__((always_inline)) u32 ink_r2_g09_perm_al(const u32 lo, const u32 hi, const u32 psel) {
  return __builtin_amdgcn_perm(hi, lo, psel);
}

/* Load IQ3_XXS hb0 (base[0..8] = d+qs_hb0, base[16..20] = aux_hb0) in 5 global loads.
 * base[8] and base[20] are carried forward to close hb1 (zero redundant reloads). */
static inline __attribute__((always_inline)) void ink_r2_g09_load_hb0(const G u8 *p, u32 qw[9], u32 aw[5]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v0 = ink_r2_g09_ld4(base + 0u);
  const u32x4 v1 = ink_r2_g09_ld4(base + 4u);
  qw[0] = v0.x; qw[1] = v0.y; qw[2] = v0.z; qw[3] = v0.w;
  qw[4] = v1.x; qw[5] = v1.y; qw[6] = v1.z; qw[7] = v1.w;
  qw[8] = base[8];
  const u32x4 a0 = ink_r2_g09_ld4(base + 16u);
  aw[0] = a0.x; aw[1] = a0.y; aw[2] = a0.z; aw[3] = a0.w;
  aw[4] = base[20];
}

/* Load IQ3_XXS hb1 tail (base[9..16] = qs_hb1, base[21..24] = aux_hb1) in 3x global_load_b128. */
static inline __attribute__((always_inline)) void ink_r2_g09_load_hb1_tail(const G u8 *p, u32 qt[8], u32 at[4]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v0 = ink_r2_g09_ld4(base + 9u);
  const u32x4 v1 = ink_r2_g09_ld4(base + 13u);
  qt[0] = v0.x; qt[1] = v0.y; qt[2] = v0.z; qt[3] = v0.w;
  qt[4] = v1.x; qt[5] = v1.y; qt[6] = v1.z; qt[7] = v1.w;
  const u32x4 a0 = ink_r2_g09_ld4(base + 21u);
  at[0] = a0.x; at[1] = a0.y; at[2] = a0.z; at[3] = a0.w;
}

static inline __attribute__((always_inline)) void ink_r2_g09_load_a_pair(
    const LDSP u8 *lx, const u32 k, const u32 r16, const u32 h, v2i a0[2], v2i a1[2]) {
  const u32 tk0 = r16;
  const u32 tk1 = r16 + 16u;
  a0[0] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk0, 2u * k) + 8u * h);
  a0[1] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk1, 2u * k) + 8u * h);
  a1[0] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk0, 2u * k + 1u) + 8u * h);
  a1[1] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk1, 2u * k + 1u) + 8u * h);
}

/* 2-k batched IQ3_XXS decode (k = k0, k0+1) overlapping 8 LUT ds_load_b32 +
 * step-k0 A/dx LDS loads. Caller passes &qw[2*k0] and &aw[k0]. */
static inline __attribute__((always_inline)) void ink_r2_g09_decode_pair(
    const u32 *qw, const u32 *aw, const u32 psel, const float d2, const LDSP u32 *lut,
    const LDSP u8 *lx, const LDSP float *lsx, const u32 k0, const u32 r16, const u32 h,
    const u32 h_off, ink_r2_g09_dc_t dc[2], v2i a0_0[2], v2i a1_0[2], f32x4 *dx_lo, f32x4 *dx_hi) {
  u32 h_loc = h_off >> 5u;
  __asm__ volatile("" : "+v"(h_loc));
  const u32 psel_q = h_loc ? 0x07060302u : 0x05040100u;
  u32 auxv[2], g00[2], g01[2], g10[2], g11[2];
#pragma unroll
  for (u32 dk = 0u; dk < 2u; dk++) {
    auxv[dk] = ink_r2_g09_perm_al(aw[dk], aw[dk + 1u], psel);
    const u32 w0 = ink_r2_g09_perm_al(qw[2u * dk], qw[2u * dk + 1u], psel);
    const u32 w1 = ink_r2_g09_perm_al(qw[2u * dk + 1u], qw[2u * dk + 2u], psel);
    const u32 qp = __builtin_amdgcn_perm(w1, w0, psel_q);
    g00[dk] = lut[qp & 255u];
    g01[dk] = lut[(qp >> 8u) & 255u];
    g10[dk] = lut[(qp >> 16u) & 255u];
    g11[dk] = lut[qp >> 24u];
  }
  ink_r2_g09_load_a_pair(lx, k0, r16, h, a0_0, a1_0);
  const u32 ts0 = k0 * 32u;
  *dx_lo = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx + ts0) + h_off);
  *dx_hi = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx + ts0 + 4u) + h_off);
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for (u32 dk = 0u; dk < 2u; dk++) {
    dc[dk].s0 = d2 * (0.5f + (float)(auxv[dk] >> 28u));
  }
#pragma unroll
  for (u32 dk = 0u; dk < 2u; dk++) {
    const u32 s8_0 = rq_ksign7((auxv[dk] >> (7u * h_loc)) & 127u);
    const u32 s8_1 = rq_ksign7((auxv[dk] >> (7u * (2u + h_loc))) & 127u);
    const v2i v0 = {(int)ie_rq_ssign(g00[dk], ie_rq_mask4(s8_0 & 15u)),
                    (int)ie_rq_ssign(g01[dk], ie_rq_mask4(s8_0 >> 4u))};
    const v2i v1 = {(int)ie_rq_ssign(g10[dk], ie_rq_mask4(s8_1 & 15u)),
                    (int)ie_rq_ssign(g11[dk], ie_rq_mask4(s8_1 >> 4u))};
    dc[dk].w0 = v0;
    dc[dk].w1 = v1;
  }
}

/* Pipelined 2-k apply (k = k0, k0+1) for 32-pair tile (j = 0, 1). */
static inline __attribute__((always_inline)) void ink_r2_g09_apply_pair(
    const ink_r2_g09_dc_t dc[2], const LDSP u8 *lx, const LDSP float *lsx,
    const u32 r16, const u32 h, const u32 h_off, v2i a0_c[2], v2i a1_c[2],
    f32x4 *dx_lo_p, f32x4 *dx_hi_p, float acc[2][8], const u32 k0) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  f32x4 dx_lo = *dx_lo_p;
  f32x4 dx_hi = *dx_hi_p;
#pragma unroll
  for (u32 dk = 0u; dk < 2u; dk++) {
    const u32 k = k0 + dk;
    const float s0 = dc[dk].s0;
    v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_c[0], 1, dc[dk].w0, z, 0);
    v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_c[1], 1, dc[dk].w0, z, 0);
    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_c[0], 1, dc[dk].w1, c0, 0);
    c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_c[1], 1, dc[dk].w1, c1, 0);
    __builtin_amdgcn_sched_barrier(0);
    if (dk == 0u) {
      ink_r2_g09_load_a_pair(lx, k0 + 1u, r16, h, a0_c, a1_c);
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (u32 v = 0u; v < 4u; v++) acc[0][v] += ((float)c0[v] * s0) * dx_lo[v];
#pragma unroll
    for (u32 v = 0u; v < 4u; v++) acc[0][4u + v] += ((float)c0[4u + v] * s0) * dx_hi[v];
    INK_R2_G09_PIN_ACC8(acc[0]);
    __builtin_amdgcn_sched_barrier(0);
    const u32 ts1 = k * 32u + 16u;
    dx_lo = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx + ts1) + h_off);
    dx_hi = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx + ts1 + 4u) + h_off);
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (u32 v = 0u; v < 4u; v++) acc[1][v] += ((float)c1[v] * s0) * dx_lo[v];
#pragma unroll
    for (u32 v = 0u; v < 4u; v++) acc[1][4u + v] += ((float)c1[4u + v] * s0) * dx_hi[v];
    INK_R2_G09_PIN_ACC8(acc[1]);
    __builtin_amdgcn_sched_barrier(0);
    if (dk == 0u) {
      const u32 tsn = (k0 + 1u) * 32u;
      dx_lo = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx + tsn) + h_off);
      dx_hi = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx + tsn + 4u) + h_off);
    }
    __builtin_amdgcn_sched_barrier(0);
  }
  *dx_lo_p = dx_lo;
  *dx_hi_p = dx_hi;
}

static inline __attribute__((always_inline)) void ink_r2_g09_stage_x32(
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

static inline __attribute__((always_inline)) u32 ink_r2_g09_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

KERNEL ie_inkling_gemm_rq_exp_iq3xxs_r2(const G u8 *qw, const G u8 *X, G float *Y, u32 rows, u32 nb,
                                        const G u32 *list, const G u32 *offs, u32 np, u32 q_bytes,
                                        u32 xs, u32 ys, u32 xdiv, const G float *res, u32 rs) {
  static LDS ink_r2_g09_exp_lds_t ink_r2_g09_exp_lds;
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
  ink_r2_g09_exp_lds.lut[t0] = ie_rq_iq3xxs_grid[t0];

  u32 xso_ = xs, yso_ = ys, rso_ = rs;
  __asm__ volatile("" : "+v"(xso_), "+v"(yso_), "+v"(rso_));
  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);
  __asm__ volatile("" : "+v"(st_x));
  u32 h_off = 32u * h;
  __asm__ volatile("" : "+v"(h_off));
  const int is_s = (w == 0u);

  for (u32 pb = p0; pb < p1; pb += 32u) {
    if (pb > p0) lbarrier();
    const u32 rem = p1 - pb;
    const u32 nj = rem < 32u ? rem : 32u;
    if (is_s) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r2_g09_exp_lds.pid[t0] = list ? list[pj] : pj;
    }
    lbarrier();

    const G u8 *xp0 = X + (unsigned long)ink_r2_g09_div_x(ink_r2_g09_exp_lds.pid[tk_ld], xdiv) * xso_ + c8_ld * 16u;
    const G float *sxp = (const G float *)(X + (unsigned long)ink_r2_g09_div_x(ink_r2_g09_exp_lds.pid[t0 & 31u], xdiv) * xso_ + 32ul * nb);

    float acc[2][8];
#pragma unroll
    for (u32 j = 0u; j < 2u; j++) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) acc[j][v] = 0.0f;
    }

    u32x4 rx0 = ink_r2_g09_ld4(xp0);
    f32x4 rsx = is_s ? rq_load_f32x4(sxp) : (f32x4)(0.0f);
    const G u8 *bp = rowp;
    u32 cur_psel = init_psel;
    u32 cur_q[9], cur_a[5];
    ink_r2_g09_load_hb0(bp, cur_q, cur_a);
    ink_r2_g09_stage_x32(ink_r2_g09_exp_lds.x[0], ink_r2_g09_exp_lds.sx[0], st_x, t0, is_s, rx0, rsx);
    lbarrier();
    __asm__ volatile("" : "+v"(cur_q[0]), "+v"(cur_q[4]), "+v"(cur_q[8]), "+v"(cur_a[0]), "+v"(cur_a[4]));

#pragma unroll 1
    for (u32 sb = 0u; ; sb++) {
      const u32 kb1 = 8u * sb + 4u;
      /* hb0: prefetch hb1 activations FIRST so they return ahead of hb1_tail. */
      rx0 = ink_r2_g09_ld4(xp0 + kb1 * 32u);
      if (is_s) rsx = rq_load_f32x4(sxp + kb1);
      __builtin_amdgcn_sched_barrier(0);

      float d2 = ie_rq_h2f((cur_q[0] >> ((cur_psel & 4u) << 2u)) & 0xFFFFu) * 0.5f;
      __asm__ volatile("" : "+v"(d2));
      const u32 carry_b8 = cur_q[8];
      const u32 carry_b20 = cur_a[4];

      ink_r2_g09_dc_t xdc[2];
      v2i a0_c[2], a1_c[2];
      f32x4 dx_lo, dx_hi;

      /* hb0 k = 0, 1 */
      ink_r2_g09_decode_pair(cur_q, cur_a, cur_psel, d2, ink_r2_g09_exp_lds.lut,
                             ink_r2_g09_exp_lds.x[0], ink_r2_g09_exp_lds.sx[0],
                             0u, r16, h, h_off, xdc, a0_c, a1_c, &dx_lo, &dx_hi);
      __builtin_amdgcn_sched_barrier(0);
      ink_r2_g09_apply_pair(xdc, ink_r2_g09_exp_lds.x[0], ink_r2_g09_exp_lds.sx[0],
                            r16, h, h_off, a0_c, a1_c, &dx_lo, &dx_hi, acc, 0u);

      /* hb0 k = 2, 3: cur_q/cur_a die right after decode -> reuse regs for hb1_tail! */
      ink_r2_g09_decode_pair(cur_q + 4u, cur_a + 2u, cur_psel, d2, ink_r2_g09_exp_lds.lut,
                             ink_r2_g09_exp_lds.x[0], ink_r2_g09_exp_lds.sx[0],
                             2u, r16, h, h_off, xdc, a0_c, a1_c, &dx_lo, &dx_hi);
      __builtin_amdgcn_sched_barrier(0);
      u32 hb1_q[8], hb1_a[4];
      ink_r2_g09_load_hb1_tail(bp, hb1_q, hb1_a);
      __builtin_amdgcn_sched_barrier(0);
      ink_r2_g09_apply_pair(xdc, ink_r2_g09_exp_lds.x[0], ink_r2_g09_exp_lds.sx[0],
                            r16, h, h_off, a0_c, a1_c, &dx_lo, &dx_hi, acc, 2u);

      ink_r2_g09_stage_x32(ink_r2_g09_exp_lds.x[1], ink_r2_g09_exp_lds.sx[1], st_x, t0, is_s, rx0, rsx);
      lbarrier();

      /* hb1: assemble q1/a1 from carried words + hb1_tail; prefetch next SB activations. */
      u32 q1[9] = {carry_b8, hb1_q[0], hb1_q[1], hb1_q[2], hb1_q[3], hb1_q[4], hb1_q[5], hb1_q[6], hb1_q[7]};
      u32 a1[5] = {carry_b20, hb1_a[0], hb1_a[1], hb1_a[2], hb1_a[3]};
      __asm__ volatile("" : "+v"(q1[1]), "+v"(q1[5]), "+v"(a1[1]));
      const u32 psel_hb1 = cur_psel;
      bp += 98u;
      cur_psel ^= 0x02020606u;
      const int more = (sb + 1u < mb);
      if (more) {
        const u32 kbn = 8u * (sb + 1u);
        rx0 = ink_r2_g09_ld4(xp0 + kbn * 32u);
        if (is_s) rsx = rq_load_f32x4(sxp + kbn);
      }
      __builtin_amdgcn_sched_barrier(0);

      /* hb1 k = 0, 1 */
      ink_r2_g09_decode_pair(q1, a1, psel_hb1, d2, ink_r2_g09_exp_lds.lut,
                             ink_r2_g09_exp_lds.x[1], ink_r2_g09_exp_lds.sx[1],
                             0u, r16, h, h_off, xdc, a0_c, a1_c, &dx_lo, &dx_hi);
      __builtin_amdgcn_sched_barrier(0);
      ink_r2_g09_apply_pair(xdc, ink_r2_g09_exp_lds.x[1], ink_r2_g09_exp_lds.sx[1],
                            r16, h, h_off, a0_c, a1_c, &dx_lo, &dx_hi, acc, 0u);

      /* hb1 k = 2, 3: q1/a1 die right after decode -> load next SB hb0 directly into cur_q/cur_a! */
      ink_r2_g09_decode_pair(q1 + 4u, a1 + 2u, psel_hb1, d2, ink_r2_g09_exp_lds.lut,
                             ink_r2_g09_exp_lds.x[1], ink_r2_g09_exp_lds.sx[1],
                             2u, r16, h, h_off, xdc, a0_c, a1_c, &dx_lo, &dx_hi);
      __builtin_amdgcn_sched_barrier(0);
      if (more) ink_r2_g09_load_hb0(bp, cur_q, cur_a);
      __builtin_amdgcn_sched_barrier(0);
      ink_r2_g09_apply_pair(xdc, ink_r2_g09_exp_lds.x[1], ink_r2_g09_exp_lds.sx[1],
                            r16, h, h_off, a0_c, a1_c, &dx_lo, &dx_hi, acc, 2u);

      if (!more) break;
      ink_r2_g09_stage_x32(ink_r2_g09_exp_lds.x[0], ink_r2_g09_exp_lds.sx[0], st_x, t0, is_s, rx0, rsx);
      lbarrier();
      __asm__ volatile("" : "+v"(cur_q[0]), "+v"(cur_q[4]), "+v"(cur_q[8]), "+v"(cur_a[0]), "+v"(cur_a[4]));
    }

    if (row < rows) {
      const u32 h_ep = h_off >> 5u;
      if (res) {
#pragma unroll
        for (u32 j = 0u; j < 2u; j++) {
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) {
            const u32 slot = 16u * j + 8u * h_ep + v;
            if (slot < nj) {
              const u32 pid = ink_r2_g09_exp_lds.pid[slot];
              G float *yp = (G float *)((G u8 *)Y + (unsigned long)pid * yso_);
              const G float *rp = (const G float *)((const G u8 *)res + (unsigned long)pid * rso_);
              yp[row] = rp[row] + acc[j][v];
            }
          }
        }
      } else {
#pragma unroll
        for (u32 j = 0u; j < 2u; j++) {
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) {
            const u32 slot = 16u * j + 8u * h_ep + v;
            if (slot < nj) {
              const u32 pid = ink_r2_g09_exp_lds.pid[slot];
              G float *yp = (G float *)((G u8 *)Y + (unsigned long)pid * yso_);
              yp[row] = acc[j][v];
            }
          }
        }
      }
    }
  }
}

/* ============================================================================
 * Kernel 2: T=1 Decode Fused IQ3_XXS gate_up + SITU + Q8_0 Quant
 *   - Each WG owns 1 output Q8_0 block (32 gate rows 0..eff-1 + 32 paired up
 *     rows eff..2*eff-1) for 1 active expert slot (grid = ((eff+31)/32, K, 1)).
 *   - 4 paired (gate, up) rows per wave, 8 lanes/row (1 lane per IQ3_XXS
 *     sub-block): halves xq reads vs separate gate/up, interleaves up_c loads
 *     with gate_c compute and next (x_n, gate_n) loads with up_c compute.
 *   - Exact 4-acc wave_tree_f reduction + #pragma clang fp contract(off) +
 *     ie_tanhf_s / sigmoidf / quant_wave -> 100% bitwise-exact vs
 *     ie_gexp_raw + ie_situ, eliminating exp_gu global round-trip & 48 OP_SITU
 *     launches/tok.
 * ========================================================================== */

static inline __attribute__((always_inline)) float ink_r2_g09_iq3xxs_sub_dot(
    const LDSP u32 *ulut, const u32 d16, const u32 aux, const u32 q01, const u32 q23,
    const u32x4 x0, const u32x4 x1, const float dav) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d16);
  const float dw = d * (0.5f + (float)(aux >> 28u)) * 0.5f;
  u32 g_lo[4], g_hi[4];
#pragma unroll
  for (u32 g = 0u; g < 4u; g++) {
    const u32 qp = ((g < 2u ? q01 : q23) >> (16u * (g & 1u))) & 0xFFFFu;
    g_lo[g] = ulut[qp & 255u];
    g_hi[g] = ulut[qp >> 8u];
  }
  __builtin_amdgcn_sched_barrier(0);
  u32 wl[4], wh[4];
#pragma unroll
  for (u32 g = 0u; g < 4u; g++) {
    const u32 s8 = rq_ksign7((aux >> (7u * g)) & 127u);
    wl[g] = ie_rq_ssign(g_lo[g], ie_rq_mask4(s8 & 15u));
    wh[g] = ie_rq_ssign(g_hi[g], ie_rq_mask4(s8 >> 4u));
  }
  u32 idot = ie_dot4_ss(0u, wl[0], x0.x);
  idot = ie_dot4_ss(idot, wh[0], x0.y);
  idot = ie_dot4_ss(idot, wl[1], x0.z);
  idot = ie_dot4_ss(idot, wh[1], x0.w);
  idot = ie_dot4_ss(idot, wl[2], x1.x);
  idot = ie_dot4_ss(idot, wh[2], x1.y);
  idot = ie_dot4_ss(idot, wl[3], x1.z);
  idot = ie_dot4_ss(idot, wh[3], x1.w);
  return dav * (dw * (float)(int)idot);
}

KERNEL ie_inkling_gexp_iq3xxs_situ_r2(const G u8 *qw, const G u8 *xq, G float *b, G u8 *q,
                                      u32 eff, u32 nb, const G u32 *route, u32 q_bytes,
                                      u32 astr, u32 bstr, u32 qstr, float beta, float lbeta) {
#pragma clang fp contract(off)
  static LDS u32 ink_r2_g09_dec_lut[256] __attribute__((aligned(16)));
  static LDS float ink_r2_g09_situ_v[32] __attribute__((aligned(16)));
  if (eff == 0u || (eff & 31u) != 0u || nb < 8u || (nb & 7u) != 0u || !qw || !xq || (!b && !q)) return;
  const u32 blk = wgid();
  const u32 nb_out = eff >> 5u;
  if (blk >= nb_out) return;

  const u32 t0 = tid();
  ink_r2_g09_dec_lut[t0] = ie_rq_iq3xxs_grid[t0];
  lbarrier();

  const u32 slot = tokid();
  const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[slot]);
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 r_in_blk = wv * 4u + k;
  const u32 r = blk * 32u + r_in_blk;
  const u32 mb = nb >> 3u;
  const unsigned long row_bytes = (unsigned long)mb * 98ul;
  const G u8 *ebase = qw + (unsigned long)e * q_bytes;
  const G u8 *bpg = ebase + (unsigned long)r * row_bytes;
  const G u8 *bpu = ebase + (unsigned long)(eff + r) * row_bytes;
  const G u8 *xp = rq_ptr_u(xq + (unsigned long)slot * astr);
  const G float *dap = (const G float *)(xp + 32ul * nb);

  const u32 qo = 2u + 8u * s;
  const u32 auxo = 66u + 4u * s;
  float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
  float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;

  /* Prologue: prefetch superblock 0 activations + gate weights. */
  u32 dg_c = ie_rq_a16(bpg, 0u);
  u32 auxg_c = ie_rq_a32(bpg, auxo);
  u32 q01g_c = ie_rq_a32(bpg, qo);
  u32 q23g_c = ie_rq_a32(bpg, qo + 4u);
  u32x4 x0_c = *(const G u32x4 *)(xp + 32u * s);
  u32x4 x1_c = *(const G u32x4 *)(xp + 32u * s + 16u);
  float dav_c = dap[s];
  __asm__ volatile("" : "+v"(dg_c), "+v"(auxg_c), "+v"(q01g_c), "+v"(q23g_c));

#pragma unroll 1
  for (u32 sb = 0u; sb < mb; sb++) {
    const u32 j = sb & 3u;
    const int has_nxt = (sb + 1u < mb);

    /* Overlap current-SB up loads with current-SB gate compute. */
    const u32 du_c = ie_rq_a16(bpu, 0u);
    const u32 auxu_c = ie_rq_a32(bpu, auxo);
    const u32 q01u_c = ie_rq_a32(bpu, qo);
    const u32 q23u_c = ie_rq_a32(bpu, qo + 4u);
    __builtin_amdgcn_sched_barrier(0);
    const float vg = ink_r2_g09_iq3xxs_sub_dot(ink_r2_g09_dec_lut, dg_c, auxg_c, q01g_c, q23g_c, x0_c, x1_c, dav_c);
    if (j == 0u) ag0 += vg;
    else if (j == 1u) ag1 += vg;
    else if (j == 2u) ag2 += vg;
    else ag3 += vg;
    __builtin_amdgcn_sched_barrier(0);

    bpg += 98u;
    bpu += 98u;
    u32 dg_n = 0u, auxg_n = 0u, q01g_n = 0u, q23g_n = 0u;
    u32x4 x0_n = (u32x4)(0u), x1_n = (u32x4)(0u);
    float dav_n = 0.0f;
    if (has_nxt) {
      const u32 bn = 8u * (sb + 1u) + s;
      dg_n = ie_rq_a16(bpg, 0u);
      auxg_n = ie_rq_a32(bpg, auxo);
      q01g_n = ie_rq_a32(bpg, qo);
      q23g_n = ie_rq_a32(bpg, qo + 4u);
      x0_n = *(const G u32x4 *)(xp + 32u * bn);
      x1_n = *(const G u32x4 *)(xp + 32u * bn + 16u);
      dav_n = dap[bn];
    }
    __builtin_amdgcn_sched_barrier(0);
    const float vu = ink_r2_g09_iq3xxs_sub_dot(ink_r2_g09_dec_lut, du_c, auxu_c, q01u_c, q23u_c, x0_c, x1_c, dav_c);
    if (j == 0u) au0 += vu;
    else if (j == 1u) au1 += vu;
    else if (j == 2u) au2 += vu;
    else au3 += vu;
    __builtin_amdgcn_sched_barrier(0);

    dg_c = dg_n; auxg_c = auxg_n; q01g_c = q01g_n; q23g_c = q23g_n;
    x0_c = x0_n; x1_c = x1_n; dav_c = dav_n;
  }

  /* Exact wave_tree_f parenthesization: (lane s + s+16) + (s+8 + s+24), then shl 4, 2, 1. */
  float g = (ag0 + ag2) + (ag1 + ag3);
  g += I2F(shl_i(F2I(g), 4));
  g += I2F(shl_i(F2I(g), 2));
  g += I2F(shl_i(F2I(g), 1));

  float u = (au0 + au2) + (au1 + au3);
  u += I2F(shl_i(F2I(u), 4));
  u += I2F(shl_i(F2I(u), 2));
  u += I2F(shl_i(F2I(u), 1));

  /* Exact SITU activation on the 32 block elements (s == 0u). */
  if (s == 0u) {
    const float gs = beta > 0.0f ? beta * ie_tanhf_s(g / beta) : g;
    const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(u / lbeta) : u;
    ink_r2_g09_situ_v[r_in_blk] = gs * sigmoidf(g) * us;
  }
  lbarrier();

  /* Wave 0 writes optional f32 b and exact Q8_0 quantized output via quant_wave. */
  if (wv == 0u) {
    const float v = ink_r2_g09_situ_v[l];
    if (b) ((G float *)((G u8 *)b + (unsigned long)slot * bstr))[blk * 32u + l] = v;
    if (q) quant_wave(v, blk, q + (unsigned long)slot * qstr, nb_out);
  }
}

#undef INK_R2_G09_PIN_ACC8
