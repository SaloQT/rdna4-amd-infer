/* ============================================================================
 * Inkling-2 R1 Group 03: Specialized IQ3_XXS MoE Expert Kernels (gfx1201, wave32, NT=256)
 *
 * Exports (included into kernels/ie_inkling_kernels.c after ie_kernels.c):
 *   1. KERNEL ie_inkling_gexp_raw_iq3xxs_r1(const G u8 *qw, const G u8 *xq, G float *y,
 *        u32 rows, u32 nb, const G u32 *route, u32 q_bytes, u32 astr, u32 bstr, u32 type)
 *      - Replaces: ie_gexp_raw (T=1 decode MoE ffn_gate_up_exps, IQ3_XXS 2304x2304x128, K=8)
 *      - Grid:     ((rows + 31u) / 32u, K, 1u)   [e.g. (72, 8, 1) for Inkling-2]
 *      - Guard:    type == IE_RQ_IQ3_XXS && rows > 0u && nb >= 8u && (nb & 7u) == 0u
 *      - Parity:   bitwise-exact (0 bit diffs) vs ie_gexp_raw
 *
 *   2. KERNEL ie_inkling_gemm_rq_exp_iq3xxs_r1(const G u8 *qw, const G u8 *X, G float *Y,
 *        u32 rows, u32 nb, const G u32 *list, const G u32 *offs, u32 np, u32 q_bytes,
 *        u32 xs, u32 ys, u32 xdiv, const G float *res, u32 rs)
 *      - Replaces: ie_gemm_rq_exp_iq3xxs (T>1 prefill MoE ffn_gate_up_exps, IQ3_XXS)
 *      - Grid:     ((rows + 127u) / 128u, n_exp, 1u)   [e.g. (18, 128, 1) for Inkling-2]
 *      - Guard:    rows > 0u && (rows & 31u) == 0u && nb >= 8u && (nb & 7u) == 0u
 *      - Parity:   bitwise-exact (0 bit diffs) vs ie_gemm_rq_exp_iq3xxs
 * ========================================================================== */

struct __attribute__((packed, aligned(4))) ink_r1_g03_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r1_g03_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r1_g03_u32x4p *)p)->v;
}

#define INK_R1_G03_PIN_ACC8(a)                                                                                     \
  __asm__ volatile(""                                                                                              \
                   : "+v"((a)[0]), "+v"((a)[1]), "+v"((a)[2]), "+v"((a)[3]), "+v"((a)[4]), "+v"((a)[5]),           \
                     "+v"((a)[6]), "+v"((a)[7]))

/* ============================================================================
 * Kernel 1: T=1 Decode MoE IQ3_XXS (4 rows/wave, 8 lanes/row, 32 rows/WG)
 *   - 1 KiB LDS grid LUT (vs 16 KiB in ie_gexp_raw) + ALU parity sign masks -> max occupancy.
 *   - 8 lanes per row map 1-to-1 to the 8 sub-blocks of each 98-byte IQ3_XXS superblock:
 *     100% coalesced 64B qs + 32B aux loads, 0 idle lanes on nb=72 (mb=9), 1-SB lookahead.
 *   - 4-accumulator a[0..3] + 3-step DPP shl_i(4,2,1) preserves exact wave_tree_f FP32 order.
 * ========================================================================== */

static inline __attribute__((always_inline)) float ink_r1_g03_iq3xxs_sub_dot(
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

KERNEL_LOOP ie_inkling_gexp_raw_iq3xxs_r1(const G u8 *qw, const G u8 *xq, G float *y, u32 rows, u32 nb,
                                          const G u32 *route, u32 q_bytes, u32 astr, u32 bstr, u32 type) {
#pragma clang fp contract(off)
#ifdef IE_MEGA
#define ink_r1_g03_dec_lut mk_iq3_lut
  if (type != IE_RQ_IQ3_XXS || rows == 0u || nb < 8u || (nb & 7u) != 0u) return;
  const u32 t0 = tid();
#else
  static LDS u32 ink_r1_g03_dec_lut[256] __attribute__((aligned(16)));
  if (type != IE_RQ_IQ3_XXS || rows == 0u || nb < 8u || (nb & 7u) != 0u) return;
  const u32 t0 = tid();
  ink_r1_g03_dec_lut[t0] = ie_rq_iq3xxs_grid[t0];
  lbarrier();
#endif

  const u32 slot = tokid(), e = route[slot];
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 r = wgid() * 32u + wv * 4u + k;
  const u32 rr = r < rows ? r : rows - 1u;
  const u32 mb = nb >> 3u;
  const G u8 *bp = qw + (unsigned long)e * q_bytes + (unsigned long)rr * mb * 98u;
  const G u8 *xp = rq_ptr_u(xq + (unsigned long)slot * astr);
  const G float *dap = (const G float *)(xp + 32ul * nb);

  const u32 qo = 2u + 8u * s;
  const u32 auxo = 66u + 4u * s;
  float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  /* Prologue: prefetch superblock 0 (weights + activations). */
  u32 d_c = ie_rq_a16(bp, 0u);
  u32 aux_c = ie_rq_a32(bp, auxo);
  u32 q01_c = ie_rq_a32(bp, qo);
  u32 q23_c = ie_rq_a32(bp, qo + 4u);
  u32x4 x0_c = *(const G u32x4 *)(xp + 32u * s);
  u32x4 x1_c = *(const G u32x4 *)(xp + 32u * s + 16u);
  float dav_c = dap[s];
  __asm__ volatile("" : "+v"(d_c), "+v"(aux_c), "+v"(q01_c), "+v"(q23_c));

  const u32 mb4 = mb & ~3u;
#pragma unroll 1
  for (u32 sb0 = 0u; sb0 < mb4; sb0 += 4u) {
#pragma unroll
    for (u32 j = 0u; j < 4u; j++) {
      const u32 sb = sb0 + j;
      const int has_nxt = (sb + 1u < mb);
      bp += 98u;
      u32 d_n = 0u, aux_n = 0u, q01_n = 0u, q23_n = 0u;
      u32x4 x0_n = (u32x4)(0u), x1_n = (u32x4)(0u);
      float dav_n = 0.0f;
      if (has_nxt) {
        const u32 bn = 8u * (sb + 1u) + s;
        d_n = ie_rq_a16(bp, 0u);
        aux_n = ie_rq_a32(bp, auxo);
        q01_n = ie_rq_a32(bp, qo);
        q23_n = ie_rq_a32(bp, qo + 4u);
        x0_n = *(const G u32x4 *)(xp + 32u * bn);
        x1_n = *(const G u32x4 *)(xp + 32u * bn + 16u);
        dav_n = dap[bn];
      }
      __builtin_amdgcn_sched_barrier(0);
      a[j] += ink_r1_g03_iq3xxs_sub_dot(ink_r1_g03_dec_lut, d_c, aux_c, q01_c, q23_c, x0_c, x1_c, dav_c);
      __builtin_amdgcn_sched_barrier(0);
      d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
      x0_c = x0_n; x1_c = x1_n; dav_c = dav_n;
    }
  }

  if (mb4 < mb) {
#pragma unroll
    for (u32 j = 0u; j < 4u; j++) {
      const u32 sb = mb4 + j;
      if (sb < mb) {
        const int has_nxt = (sb + 1u < mb);
        bp += 98u;
        u32 d_n = 0u, aux_n = 0u, q01_n = 0u, q23_n = 0u;
        u32x4 x0_n = (u32x4)(0u), x1_n = (u32x4)(0u);
        float dav_n = 0.0f;
        if (has_nxt) {
          const u32 bn = 8u * (sb + 1u) + s;
          d_n = ie_rq_a16(bp, 0u);
          aux_n = ie_rq_a32(bp, auxo);
          q01_n = ie_rq_a32(bp, qo);
          q23_n = ie_rq_a32(bp, qo + 4u);
          x0_n = *(const G u32x4 *)(xp + 32u * bn);
          x1_n = *(const G u32x4 *)(xp + 32u * bn + 16u);
          dav_n = dap[bn];
        }
        __builtin_amdgcn_sched_barrier(0);
        a[j] += ink_r1_g03_iq3xxs_sub_dot(ink_r1_g03_dec_lut, d_c, aux_c, q01_c, q23_c, x0_c, x1_c, dav_c);
        __builtin_amdgcn_sched_barrier(0);
        d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
        x0_c = x0_n; x1_c = x1_n; dav_c = dav_n;
      }
    }
  }

  /* Exact wave_tree_f parenthesization: (lane s + s+16) + (s+8 + s+24), then shl 4, 2, 1. */
  float acc = (a[0] + a[2]) + (a[1] + a[3]);
  acc += I2F(shl_i(F2I(acc), 4));
  acc += I2F(shl_i(F2I(acc), 2));
  acc += I2F(shl_i(F2I(acc), 1));
  if (s == 0u && r < rows) {
    ((G float *)((G u8 *)y + (unsigned long)slot * bstr))[r] = acc;
  }
}

#ifdef IE_MEGA
#undef ink_r1_g03_dec_lut
#else
/* ============================================================================
 * Kernel 2: T>1 Prefill MoE IQ3_XXS WMMA GEMM (Adaptive 32/64 pairs x 128 rows)
 *   - LDS: 19,712 B (swizzled double-buffered 64-tok lx + lsx + 1 KiB lut + pid[64];
 *     drops unused lsa and 7 KiB excess LUT -> 3 WGs/CU LDS budget).
 *   - Adaptive token tile per expert: rem > 32 -> 128x64 (halves weight bandwidth on
 *     33..64-pair experts); rem <= 32 -> 128x32 (zero wasted WMMAs/X loads).
 *   - Base-aligned direct-VGPR IQ3_XXS weight prefetch (8 loads/SB: 5 hb0 + 3x b128 hb1),
 *     hoisted d*0.5f, batched 4-k decode_all overlapping 16 LUT + step-0 A/dx loads,
 *     single-dx zero-stall WMMA/post-op pipeline, mid-phase weight prefetch (<= 192 VGPR).
 * ========================================================================== */

typedef struct {
  v2i w0, w1;
  float s0;
} ink_r1_g03_dc_t;

typedef struct {
  u8 x[2][64u * 128u] __attribute__((aligned(16)));
  float sx[2][4u * 64u] __attribute__((aligned(32)));
  u32 lut[256] __attribute__((aligned(16)));
  u32 pid[64] __attribute__((aligned(16)));
} ink_r1_g03_exp_lds_t;

_Static_assert(sizeof(ink_r1_g03_exp_lds_t) == 19712u, "ink_r1_g03_exp_lds_t must be 19,712 B");

static inline __attribute__((always_inline)) u32 ink_r1_g03_perm_al(const u32 lo, const u32 hi, const u32 psel) {
  return __builtin_amdgcn_perm(hi, lo, psel);
}

/* Load IQ3_XXS hb0 (base[0..8] = d+qs_hb0, base[16..20] = aux_hb0) in 5 global loads.
 * base[8] and base[20] are carried forward to close hb1 (zero redundant reloads). */
static inline __attribute__((always_inline)) void ink_r1_g03_load_hb0(const G u8 *p, u32 qw[9], u32 aw[5]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v0 = ink_r1_g03_ld4(base + 0u);
  const u32x4 v1 = ink_r1_g03_ld4(base + 4u);
  qw[0] = v0.x; qw[1] = v0.y; qw[2] = v0.z; qw[3] = v0.w;
  qw[4] = v1.x; qw[5] = v1.y; qw[6] = v1.z; qw[7] = v1.w;
  qw[8] = base[8];
  const u32x4 a0 = ink_r1_g03_ld4(base + 16u);
  aw[0] = a0.x; aw[1] = a0.y; aw[2] = a0.z; aw[3] = a0.w;
  aw[4] = base[20];
}

/* Load IQ3_XXS hb1 tail (base[9..16] = qs_hb1, base[21..24] = aux_hb1) in 3x global_load_b128. */
static inline __attribute__((always_inline)) void ink_r1_g03_load_hb1_tail(const G u8 *p, u32 qt[8], u32 at[4]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v0 = ink_r1_g03_ld4(base + 9u);
  const u32x4 v1 = ink_r1_g03_ld4(base + 13u);
  qt[0] = v0.x; qt[1] = v0.y; qt[2] = v0.z; qt[3] = v0.w;
  qt[4] = v1.x; qt[5] = v1.y; qt[6] = v1.z; qt[7] = v1.w;
  const u32x4 a0 = ink_r1_g03_ld4(base + 21u);
  at[0] = a0.x; at[1] = a0.y; at[2] = a0.z; at[3] = a0.w;
}

static inline __attribute__((always_inline)) void ink_r1_g03_load_a_pair(
    const LDSP u8 *lx, const u32 k, const u32 j0, const u32 r16, const u32 h, v2i a0[2], v2i a1[2]) {
  const u32 tk0 = 16u * j0 + r16;
  const u32 tk1 = tk0 + 16u;
  a0[0] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk0, 2u * k) + 8u * h);
  a0[1] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk1, 2u * k) + 8u * h);
  a1[0] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk0, 2u * k + 1u) + 8u * h);
  a1[1] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk1, 2u * k + 1u) + 8u * h);
}

/* Batched 4-k IQ3_XXS decode overlapping all 16 LUT ds_load_b32 + step-0 A/dx loads. */
static inline __attribute__((always_inline)) void ink_r1_g03_decode_all(
    const u32 qw[9], const u32 aw[5], const u32 psel, const float d2, const LDSP u32 *lut,
    const LDSP u8 *lx, const LDSP float *lsx, const u32 r16, const u32 h, const u32 h_off,
    ink_r1_g03_dc_t dc[4], v2i a0_0[2], v2i a1_0[2], f32x4 *dx_lo, f32x4 *dx_hi) {
  u32 h_loc = h_off >> 5u;
  __asm__ volatile("" : "+v"(h_loc));
  const u32 psel_q = h_loc ? 0x07060302u : 0x05040100u;
  u32 auxv[4], g00[4], g01[4], g10[4], g11[4];
#pragma unroll
  for (u32 k = 0u; k < 4u; k++) {
    auxv[k] = ink_r1_g03_perm_al(aw[k], aw[k + 1u], psel);
    const u32 w0 = ink_r1_g03_perm_al(qw[2u * k], qw[2u * k + 1u], psel);
    const u32 w1 = ink_r1_g03_perm_al(qw[2u * k + 1u], qw[2u * k + 2u], psel);
    const u32 qp = __builtin_amdgcn_perm(w1, w0, psel_q);
    g00[k] = lut[qp & 255u];
    g01[k] = lut[(qp >> 8u) & 255u];
    g10[k] = lut[(qp >> 16u) & 255u];
    g11[k] = lut[qp >> 24u];
  }
  ink_r1_g03_load_a_pair(lx, 0u, 0u, r16, h, a0_0, a1_0);
  *dx_lo = *(const LDSP f32x4 *)((const LDSP u8 *)lsx + h_off);
  *dx_hi = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx + 4u) + h_off);
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for (u32 k = 0u; k < 4u; k++) {
    dc[k].s0 = d2 * (0.5f + (float)(auxv[k] >> 28u));
  }
#pragma unroll
  for (u32 k = 0u; k < 4u; k++) {
    const u32 s8_0 = rq_ksign7((auxv[k] >> (7u * h_loc)) & 127u);
    const u32 s8_1 = rq_ksign7((auxv[k] >> (7u * (2u + h_loc))) & 127u);
    const v2i v0 = {(int)ie_rq_ssign(g00[k], ie_rq_mask4(s8_0 & 15u)),
                    (int)ie_rq_ssign(g01[k], ie_rq_mask4(s8_0 >> 4u))};
    const v2i v1 = {(int)ie_rq_ssign(g10[k], ie_rq_mask4(s8_1 & 15u)),
                    (int)ie_rq_ssign(g11[k], ie_rq_mask4(s8_1 >> 4u))};
    dc[k].w0 = v0;
    dc[k].w1 = v1;
  }
}

/* Single-dx zero-stall pipelined apply for 64-pair tile (nsteps=8: 2 pairs/k x 4 k)
 * or 32-pair tile (nsteps=4: 1 pair/k x 4 k). ntok_ is compile-time 64u or 32u. */
static inline __attribute__((always_inline)) void ink_r1_g03_apply_range(
    const ink_r1_g03_dc_t dc[4], const LDSP u8 *lx, const LDSP float *lsx, const u32 ntok,
    const u32 r16, const u32 h, const u32 h_off, v2i a0_c[2], v2i a1_c[2],
    f32x4 *dx_lo_p, f32x4 *dx_hi_p, float acc[][8], const u32 step_beg, const u32 step_end) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  const u32 nsteps = (ntok == 64u) ? 8u : 4u;
  f32x4 dx_lo = *dx_lo_p;
  f32x4 dx_hi = *dx_hi_p;
#pragma unroll
  for (u32 step = step_beg; step < step_end; step++) {
    const u32 k = (ntok == 64u) ? (step >> 1u) : step;
    const u32 j0 = (ntok == 64u) ? ((step & 1u) << 1u) : 0u;
    const float s0 = dc[k].s0;
    v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_c[0], 1, dc[k].w0, z, 0);
    v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_c[1], 1, dc[k].w0, z, 0);
    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_c[0], 1, dc[k].w1, c0, 0);
    c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_c[1], 1, dc[k].w1, c1, 0);
    __builtin_amdgcn_sched_barrier(0);
    if (step + 1u < nsteps) {
      const u32 kn = (ntok == 64u) ? ((step + 1u) >> 1u) : (step + 1u);
      const u32 j0n = (ntok == 64u) ? (((step + 1u) & 1u) << 1u) : 0u;
      ink_r1_g03_load_a_pair(lx, kn, j0n, r16, h, a0_c, a1_c);
    }
    __builtin_amdgcn_sched_barrier(0);
    {
      float t0[8];
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) t0[v] = (float)c0[v] * s0;
#pragma unroll
      for (u32 v = 0u; v < 4u; v++) acc[j0][v] += t0[v] * dx_lo[v];
#pragma unroll
      for (u32 v = 0u; v < 4u; v++) acc[j0][4u + v] += t0[4u + v] * dx_hi[v];
      INK_R1_G03_PIN_ACC8(acc[j0]);
    }
    __builtin_amdgcn_sched_barrier(0);
    const u32 ts1 = k * ntok + 16u * (j0 + 1u);
    dx_lo = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx + ts1) + h_off);
    dx_hi = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx + ts1 + 4u) + h_off);
    __builtin_amdgcn_sched_barrier(0);
    float t1[8];
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) t1[v] = (float)c1[v] * s0;
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (u32 v = 0u; v < 4u; v++) acc[j0 + 1u][v] += t1[v] * dx_lo[v];
#pragma unroll
    for (u32 v = 0u; v < 4u; v++) acc[j0 + 1u][4u + v] += t1[4u + v] * dx_hi[v];
    INK_R1_G03_PIN_ACC8(acc[j0 + 1u]);
    __builtin_amdgcn_sched_barrier(0);
    if (step + 1u < nsteps) {
      const u32 kn = (ntok == 64u) ? ((step + 1u) >> 1u) : (step + 1u);
      const u32 j0n = (ntok == 64u) ? (((step + 1u) & 1u) << 1u) : 0u;
      const u32 tsn = kn * ntok + 16u * j0n;
      dx_lo = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx + tsn) + h_off);
      dx_hi = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx + tsn + 4u) + h_off);
    }
    __builtin_amdgcn_sched_barrier(0);
  }
  *dx_lo_p = dx_lo;
  *dx_hi_p = dx_hi;
}

static inline __attribute__((always_inline)) void ink_r1_g03_stage_x32(
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

static inline __attribute__((always_inline)) void ink_r1_g03_stage_x64(
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

static inline __attribute__((always_inline)) u32 ink_r1_g03_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

static inline __attribute__((always_inline)) void ink_r1_g03_store_exp(
    G float *Y, const G float *res, const LDSP u32 *lpid, const u32 nj, const u32 wj,
    const u32 row, const u32 h_ep, const u32 yso, const u32 rso, float acc[][8]) {
  if (res) {
#pragma unroll
    for (u32 j = 0u; j < wj; j++) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 slot = 16u * j + 8u * h_ep + v;
        if (slot < nj) {
          const u32 pid = lpid[slot];
          G float *yp = (G float *)((G u8 *)Y + (unsigned long)pid * yso);
          const G float *rp = (const G float *)((const G u8 *)res + (unsigned long)pid * rso);
          yp[row] = rp[row] + acc[j][v];
        }
      }
    }
  } else {
#pragma unroll
    for (u32 j = 0u; j < wj; j++) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 slot = 16u * j + 8u * h_ep + v;
        if (slot < nj) {
          const u32 pid = lpid[slot];
          G float *yp = (G float *)((G u8 *)Y + (unsigned long)pid * yso);
          yp[row] = acc[j][v];
        }
      }
    }
  }
}

KERNEL ie_inkling_gemm_rq_exp_iq3xxs_r1(const G u8 *qw, const G u8 *X, G float *Y, u32 rows, u32 nb,
                                        const G u32 *list, const G u32 *offs, u32 np, u32 q_bytes,
                                        u32 xs, u32 ys, u32 xdiv, const G float *res, u32 rs) {
  static LDS ink_r1_g03_exp_lds_t ink_r1_g03_exp_lds;
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
  ink_r1_g03_exp_lds.lut[t0] = ie_rq_iq3xxs_grid[t0];

  u32 xso_ = xs, yso_ = ys, rso_ = rs;
  __asm__ volatile("" : "+v"(xso_), "+v"(yso_), "+v"(rso_));
  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);
  __asm__ volatile("" : "+v"(st_x));
  u32 h_off = 32u * h;
  __asm__ volatile("" : "+v"(h_off));

  for (u32 pb = p0; pb < p1; ) {
    if (pb > p0) lbarrier();
    const u32 rem = p1 - pb;
    if (rem > 32u) {
      /* ---------------- Adaptive 64-pair tile (128x64, wj=4) ---------------- */
      const u32 nj = rem < 64u ? rem : 64u;
      const int is_s = (w < 2u);
      if (is_s) {
        const u32 pj = pb + (t0 < nj ? t0 : 0u);
        ink_r1_g03_exp_lds.pid[t0] = list ? list[pj] : pj;
      }
      lbarrier();

      const G u8 *xp0 = X + (unsigned long)ink_r1_g03_div_x(ink_r1_g03_exp_lds.pid[tk_ld], xdiv) * xso_ + c8_ld * 16u;
      const G u8 *xp1 = X + (unsigned long)ink_r1_g03_div_x(ink_r1_g03_exp_lds.pid[tk_ld + 32u], xdiv) * xso_ + c8_ld * 16u;
      const G float *sxp = (const G float *)(X + (unsigned long)ink_r1_g03_div_x(ink_r1_g03_exp_lds.pid[t0 & 63u], xdiv) * xso_ + 32ul * nb);

      float acc[4][8];
#pragma unroll
      for (u32 j = 0u; j < 4u; j++) {
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) acc[j][v] = 0.0f;
      }

      u32x4 rx0 = ink_r1_g03_ld4(xp0);
      u32x4 rx1 = ink_r1_g03_ld4(xp1);
      f32x4 rsx = is_s ? rq_load_f32x4(sxp) : (f32x4)(0.0f);
      const G u8 *bp = rowp;
      u32 cur_psel = init_psel;
      u32 cur_q[9], cur_a[5];
      ink_r1_g03_load_hb0(bp, cur_q, cur_a);
      ink_r1_g03_stage_x64(ink_r1_g03_exp_lds.x[0], ink_r1_g03_exp_lds.sx[0], st_x, t0, is_s, rx0, rx1, rsx);
      lbarrier();
      __asm__ volatile("" : "+v"(cur_q[0]), "+v"(cur_q[4]), "+v"(cur_q[8]), "+v"(cur_a[0]), "+v"(cur_a[4]));

#pragma unroll 1
      for (u32 sb = 0u; ; sb++) {
        const u32 kb1 = 8u * sb + 4u;
        /* hb0: prefetch hb1 activations FIRST (FIFO ahead of hb1_tail). */
        rx0 = ink_r1_g03_ld4(xp0 + kb1 * 32u);
        rx1 = ink_r1_g03_ld4(xp1 + kb1 * 32u);
        if (is_s) rsx = rq_load_f32x4(sxp + kb1);
        __builtin_amdgcn_sched_barrier(0);

        float d2 = ie_rq_h2f((cur_q[0] >> ((cur_psel & 4u) << 2u)) & 0xFFFFu) * 0.5f;
        __asm__ volatile("" : "+v"(d2));
        const u32 carry_b8 = cur_q[8];
        const u32 carry_b20 = cur_a[4];

        ink_r1_g03_dc_t xdc[4];
        v2i a0_c[2], a1_c[2];
        f32x4 dx_lo, dx_hi;
        ink_r1_g03_decode_all(cur_q, cur_a, cur_psel, d2, ink_r1_g03_exp_lds.lut,
                              ink_r1_g03_exp_lds.x[0], ink_r1_g03_exp_lds.sx[0],
                              r16, h, h_off, xdc, a0_c, a1_c, &dx_lo, &dx_hi);
        __builtin_amdgcn_sched_barrier(0);
        ink_r1_g03_apply_range(xdc, ink_r1_g03_exp_lds.x[0], ink_r1_g03_exp_lds.sx[0],
                               64u, r16, h, h_off, a0_c, a1_c, &dx_lo, &dx_hi, acc, 0u, 4u);
        u32 hb1_q[8], hb1_a[4];
        ink_r1_g03_load_hb1_tail(bp, hb1_q, hb1_a);
        __builtin_amdgcn_sched_barrier(0);
        ink_r1_g03_apply_range(xdc, ink_r1_g03_exp_lds.x[0], ink_r1_g03_exp_lds.sx[0],
                               64u, r16, h, h_off, a0_c, a1_c, &dx_lo, &dx_hi, acc, 4u, 8u);

        ink_r1_g03_stage_x64(ink_r1_g03_exp_lds.x[1], ink_r1_g03_exp_lds.sx[1], st_x, t0, is_s, rx0, rx1, rsx);
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
          rx0 = ink_r1_g03_ld4(xp0 + kbn * 32u);
          rx1 = ink_r1_g03_ld4(xp1 + kbn * 32u);
          if (is_s) rsx = rq_load_f32x4(sxp + kbn);
        }
        __builtin_amdgcn_sched_barrier(0);

        ink_r1_g03_decode_all(q1, a1, psel_hb1, d2, ink_r1_g03_exp_lds.lut,
                              ink_r1_g03_exp_lds.x[1], ink_r1_g03_exp_lds.sx[1],
                              r16, h, h_off, xdc, a0_c, a1_c, &dx_lo, &dx_hi);
        __builtin_amdgcn_sched_barrier(0);
        ink_r1_g03_apply_range(xdc, ink_r1_g03_exp_lds.x[1], ink_r1_g03_exp_lds.sx[1],
                               64u, r16, h, h_off, a0_c, a1_c, &dx_lo, &dx_hi, acc, 0u, 4u);
        u32 nxt_q[9], nxt_a[5];
        if (more) ink_r1_g03_load_hb0(bp, nxt_q, nxt_a);
        __builtin_amdgcn_sched_barrier(0);
        ink_r1_g03_apply_range(xdc, ink_r1_g03_exp_lds.x[1], ink_r1_g03_exp_lds.sx[1],
                               64u, r16, h, h_off, a0_c, a1_c, &dx_lo, &dx_hi, acc, 4u, 8u);

        if (!more) break;
        ink_r1_g03_stage_x64(ink_r1_g03_exp_lds.x[0], ink_r1_g03_exp_lds.sx[0], st_x, t0, is_s, rx0, rx1, rsx);
        lbarrier();
#pragma unroll
        for (u32 i = 0u; i < 9u; i++) cur_q[i] = nxt_q[i];
#pragma unroll
        for (u32 i = 0u; i < 5u; i++) cur_a[i] = nxt_a[i];
        __asm__ volatile("" : "+v"(cur_q[0]), "+v"(cur_q[4]), "+v"(cur_q[8]), "+v"(cur_a[0]), "+v"(cur_a[4]));
      }

      if (row < rows) {
        const u32 h_ep = h_off >> 5u;
        ink_r1_g03_store_exp(Y, res, ink_r1_g03_exp_lds.pid, nj, 4u, row, h_ep, yso_, rso_, acc);
      }
      pb += 64u;
    } else {
      /* ---------------- Adaptive 32-pair tile (128x32, wj=2) ---------------- */
      const u32 nj = rem;
      const int is_s = (w == 0u);
      if (is_s) {
        const u32 pj = pb + (t0 < nj ? t0 : 0u);
        ink_r1_g03_exp_lds.pid[t0] = list ? list[pj] : pj;
      }
      lbarrier();

      const G u8 *xp0 = X + (unsigned long)ink_r1_g03_div_x(ink_r1_g03_exp_lds.pid[tk_ld], xdiv) * xso_ + c8_ld * 16u;
      const G float *sxp = (const G float *)(X + (unsigned long)ink_r1_g03_div_x(ink_r1_g03_exp_lds.pid[t0 & 31u], xdiv) * xso_ + 32ul * nb);

      float acc[2][8];
#pragma unroll
      for (u32 j = 0u; j < 2u; j++) {
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) acc[j][v] = 0.0f;
      }

      u32x4 rx0 = ink_r1_g03_ld4(xp0);
      f32x4 rsx = is_s ? rq_load_f32x4(sxp) : (f32x4)(0.0f);
      const G u8 *bp = rowp;
      u32 cur_psel = init_psel;
      u32 cur_q[9], cur_a[5];
      ink_r1_g03_load_hb0(bp, cur_q, cur_a);
      ink_r1_g03_stage_x32(ink_r1_g03_exp_lds.x[0], ink_r1_g03_exp_lds.sx[0], st_x, t0, is_s, rx0, rsx);
      lbarrier();
      __asm__ volatile("" : "+v"(cur_q[0]), "+v"(cur_q[4]), "+v"(cur_q[8]), "+v"(cur_a[0]), "+v"(cur_a[4]));

#pragma unroll 1
      for (u32 sb = 0u; ; sb++) {
        const u32 kb1 = 8u * sb + 4u;
        rx0 = ink_r1_g03_ld4(xp0 + kb1 * 32u);
        if (is_s) rsx = rq_load_f32x4(sxp + kb1);
        __builtin_amdgcn_sched_barrier(0);

        float d2 = ie_rq_h2f((cur_q[0] >> ((cur_psel & 4u) << 2u)) & 0xFFFFu) * 0.5f;
        __asm__ volatile("" : "+v"(d2));
        const u32 carry_b8 = cur_q[8];
        const u32 carry_b20 = cur_a[4];

        ink_r1_g03_dc_t xdc[4];
        v2i a0_c[2], a1_c[2];
        f32x4 dx_lo, dx_hi;
        ink_r1_g03_decode_all(cur_q, cur_a, cur_psel, d2, ink_r1_g03_exp_lds.lut,
                              ink_r1_g03_exp_lds.x[0], ink_r1_g03_exp_lds.sx[0],
                              r16, h, h_off, xdc, a0_c, a1_c, &dx_lo, &dx_hi);
        __builtin_amdgcn_sched_barrier(0);
        ink_r1_g03_apply_range(xdc, ink_r1_g03_exp_lds.x[0], ink_r1_g03_exp_lds.sx[0],
                               32u, r16, h, h_off, a0_c, a1_c, &dx_lo, &dx_hi, acc, 0u, 2u);
        u32 hb1_q[8], hb1_a[4];
        ink_r1_g03_load_hb1_tail(bp, hb1_q, hb1_a);
        __builtin_amdgcn_sched_barrier(0);
        ink_r1_g03_apply_range(xdc, ink_r1_g03_exp_lds.x[0], ink_r1_g03_exp_lds.sx[0],
                               32u, r16, h, h_off, a0_c, a1_c, &dx_lo, &dx_hi, acc, 2u, 4u);

        ink_r1_g03_stage_x32(ink_r1_g03_exp_lds.x[1], ink_r1_g03_exp_lds.sx[1], st_x, t0, is_s, rx0, rsx);
        lbarrier();

        u32 q1[9] = {carry_b8, hb1_q[0], hb1_q[1], hb1_q[2], hb1_q[3], hb1_q[4], hb1_q[5], hb1_q[6], hb1_q[7]};
        u32 a1[5] = {carry_b20, hb1_a[0], hb1_a[1], hb1_a[2], hb1_a[3]};
        __asm__ volatile("" : "+v"(q1[1]), "+v"(q1[5]), "+v"(a1[1]));
        const u32 psel_hb1 = cur_psel;
        bp += 98u;
        cur_psel ^= 0x02020606u;
        const int more = (sb + 1u < mb);
        if (more) {
          const u32 kbn = 8u * (sb + 1u);
          rx0 = ink_r1_g03_ld4(xp0 + kbn * 32u);
          if (is_s) rsx = rq_load_f32x4(sxp + kbn);
        }
        __builtin_amdgcn_sched_barrier(0);

        ink_r1_g03_decode_all(q1, a1, psel_hb1, d2, ink_r1_g03_exp_lds.lut,
                              ink_r1_g03_exp_lds.x[1], ink_r1_g03_exp_lds.sx[1],
                              r16, h, h_off, xdc, a0_c, a1_c, &dx_lo, &dx_hi);
        __builtin_amdgcn_sched_barrier(0);
        ink_r1_g03_apply_range(xdc, ink_r1_g03_exp_lds.x[1], ink_r1_g03_exp_lds.sx[1],
                               32u, r16, h, h_off, a0_c, a1_c, &dx_lo, &dx_hi, acc, 0u, 2u);
        u32 nxt_q[9], nxt_a[5];
        if (more) ink_r1_g03_load_hb0(bp, nxt_q, nxt_a);
        __builtin_amdgcn_sched_barrier(0);
        ink_r1_g03_apply_range(xdc, ink_r1_g03_exp_lds.x[1], ink_r1_g03_exp_lds.sx[1],
                               32u, r16, h, h_off, a0_c, a1_c, &dx_lo, &dx_hi, acc, 2u, 4u);

        if (!more) break;
        ink_r1_g03_stage_x32(ink_r1_g03_exp_lds.x[0], ink_r1_g03_exp_lds.sx[0], st_x, t0, is_s, rx0, rsx);
        lbarrier();
#pragma unroll
        for (u32 i = 0u; i < 9u; i++) cur_q[i] = nxt_q[i];
#pragma unroll
        for (u32 i = 0u; i < 5u; i++) cur_a[i] = nxt_a[i];
        __asm__ volatile("" : "+v"(cur_q[0]), "+v"(cur_q[4]), "+v"(cur_q[8]), "+v"(cur_a[0]), "+v"(cur_a[4]));
      }

      if (row < rows) {
        const u32 h_ep = h_off >> 5u;
        ink_r1_g03_store_exp(Y, res, ink_r1_g03_exp_lds.pid, nj, 2u, row, h_ep, yso_, rso_, acc);
      }
      pb += 32u;
    }
  }
}
#endif /* !IE_MEGA */

#undef INK_R1_G03_PIN_ACC8
