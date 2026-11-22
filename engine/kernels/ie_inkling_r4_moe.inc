/* ============================================================================
 * Inkling-2 R4 MoE Speculative Verify Kernels (T=2..16, gfx1201, wave32, NT=256)
 *
 * Exports (included into kernels/ie_inkling_kernels.c after ie_kernels.c):
 *
 * 1. KERNEL ie_inkling_gexp_raw_iq3xxs_ver_r4(
 *      const G u8 *qw, const G u8 *X, G float *Y, u32 rows, u32 nb,
 *      const G u32 *list, const G u32 *offs, u32 np, u32 q_bytes,
 *      u32 xs, u32 ys, u32 xdiv, const G float *res, u32 rs)        [bit 44]
 *    - Signature: exact ie_gemm_rq_exp_iq3xxs signature
 *    - Replaces:  K_GEXP_RAW_T for T=2..16 (MoE ffn_gate_up_exps, IQ3_XXS)
 *    - Grid:      ((rows + 31u) / 32u, n_exp, 1u)  [e.g. (72, 128, 1)]
 *    - Guard:     T >= 2u && T <= 16u && type == IE_RQ_IQ3_XXS (18u) &&
 *                 rows > 0u && nb >= 8u && (nb & 7u) == 0u
 *    - Parity:    100% bitwise-exact (0 bit diffs) vs ie_gexp_raw_t
 *                 (#pragma clang fp contract(off), 4-acc wave_tree_f order)
 *    - Design:    Immediate WG early-return on lo >= hi (0 LDS/barrier for
 *                 empty experts), 1 KiB LDS LUT, 4 rows/wave (8 lanes/row,
 *                 0 idle lanes on nb=72), reads+decodes IQ3_XXS weights ONCE
 *                 per chunk of up to 4 routed pairs with 1-SB weight lookahead
 *                 and ping-pong L1/L2 X loads (<= 96 VGPR, 0 spills).
 *
 * 2. KERNEL ie_inkling_gexp_iq4nl_ver_r4(
 *      const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
 *      u32 rows, u32 nb, const G u32 *list, const G u32 *offs, u32 np,
 *      u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv, u32 kind,
 *      const G u8 *res, u32 rs)                                      [bit 45]
 *    - Signature: exact ie_gemm_kx signature
 *    - Replaces:  K_GEXP_T for T=2..16 (MoE ffn_down_exps, MAT_IQ4NL)
 *    - Grid:      ((rows + 31u) / 32u, n_exp, 1u)  [e.g. (72, 128, 1)]
 *    - Guard:     T >= 2u && T <= 16u && kind == MAT_IQ4NL (8u) &&
 *                 rows > 0u && nb > 0u
 *    - Parity:    100% bitwise-exact (0 bit diffs) vs ie_gexp_t
 *                 (#pragma clang fp contract(off), 4-acc wave_tree_f /
 *                  half_tree_f order, exact epilogue())
 *    - Design:    Immediate WG early-return on lo >= hi, 0 LDS, 4 rows/wave
 *                 (8 lanes/row, 90% util on nb=36), coalesced u32x4 loads,
 *                 reads+decodes IQ4_NL weights ONCE per chunk of up to 4
 *                 routed pairs with 1-step weight lookahead (<= 96 VGPR, 0 spills).
 *
 * 3. KERNEL ie_inkling_gemv_f32_router_ver_r4(
 *      const G float *w, const G float *x, G float *y, u32 rows, u32 cols,
 *      const G float *bias, const G float *res, u32 xs, u32 ys, u32 rs,
 *      u32 T)                                                        [bit 46]
 *    - Signature: exact ie_gemm_f32_s signature
 *    - Replaces:  K_GEMV_F32_T for T=2..16 (MoE router, 128x2304 f32)
 *    - Grid:      ((rows + 3u) / 4u, 1u, 1u)       [e.g. (32, 1, 1)]
 *    - Guard:     T >= 2u && T <= 16u && rows > 0u && cols > 0u
 *    - Parity:    f32-tol (~1e-6 vs ie_gemv_f32_t; exact __builtin_fmaf +
 *                 wave_tree_f + 2-wave sum + epilogue())
 *    - Design:    4 rows/WG, 2 waves/row (64 threads x f32x4 = 256 cols/step,
 *                 exact 9 steps on cols=2304), reads each f32x4 weight ONCE
 *                 per 8-token chunk (T-chunked by 8 for T=9..16), 128 B LDS
 *                 2-wave reduction, 0 spills.
 * ========================================================================== */

struct __attribute__((packed, aligned(4))) ink_r4_moe_f32x4p {
  f32x4 v;
};

static inline __attribute__((always_inline)) f32x4 ink_r4_moe_ld_f4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r4_moe_f32x4p *)p)->v;
}

static inline __attribute__((always_inline)) u32 ink_r4_moe_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

/* 8-lane subgroup reduction inside each 16-lane DPP row (lanes 0, 8, 16, 24
 * hold the exact sum of their 8-lane subgroup). */
static inline __attribute__((always_inline)) float ink_r4_moe_oct_tree_f(float v) {
#pragma clang fp contract(off)
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}

#define INK_R4_MOE_ACC_ADD(a0_, a1_, a2_, a3_, j_, v_) \
  do {                                                 \
    if ((j_) == 0u) (a0_) += (v_);                     \
    else if ((j_) == 1u) (a1_) += (v_);                \
    else if ((j_) == 2u) (a2_) += (v_);                \
    else (a3_) += (v_);                                \
  } while (0)

#define INK_R4_MOE_ACC_FMA(a0_, a1_, a2_, a3_, j_, v_, d_)          \
  do {                                                              \
    if ((j_) == 0u) (a0_) = __builtin_fmaf((v_), (d_), (a0_));      \
    else if ((j_) == 1u) (a1_) = __builtin_fmaf((v_), (d_), (a1_)); \
    else if ((j_) == 2u) (a2_) = __builtin_fmaf((v_), (d_), (a2_)); \
    else (a3_) = __builtin_fmaf((v_), (d_), (a3_));                 \
  } while (0)

/* ============================================================================
 * Kernel 1: T=2..16 Verify MoE IQ3_XXS (4 rows/wave, 8 lanes/row, 32 rows/WG)
 *   - Immediate early-return on lo >= hi before touching LDS or lbarrier().
 *   - Reads & decodes IQ3_XXS weights ONCE per chunk of up to 4 routed pairs.
 *   - 100% bitwise-exact vs ie_gexp_raw_t.
 * ========================================================================== */

static inline __attribute__((always_inline)) float ink_r4_moe_iq3_dot(
    const u32 wl0, const u32 wh0, const u32 wl1, const u32 wh1,
    const u32 wl2, const u32 wh2, const u32 wl3, const u32 wh3,
    const float dw, const u32x4 x0, const u32x4 x1, const float dav) {
#pragma clang fp contract(off)
  u32 idot = ie_dot4_ss(0u, wl0, x0.x);
  idot = ie_dot4_ss(idot, wh0, x0.y);
  idot = ie_dot4_ss(idot, wl1, x0.z);
  idot = ie_dot4_ss(idot, wh1, x0.w);
  idot = ie_dot4_ss(idot, wl2, x1.x);
  idot = ie_dot4_ss(idot, wh2, x1.y);
  idot = ie_dot4_ss(idot, wl3, x1.z);
  idot = ie_dot4_ss(idot, wh3, x1.w);
  return dav * (dw * (float)(int)idot);
}

KERNEL_LOOP ie_inkling_gexp_raw_iq3xxs_ver_r4(const G u8 *qw, const G u8 *X, G float *Y, u32 rows, u32 nb,
                                              const G u32 *list, const G u32 *offs, u32 np, u32 q_bytes,
                                              u32 xs, u32 ys, u32 xdiv, const G float *res, u32 rs) {
#pragma clang fp contract(off)
#ifdef IE_MEGA
#define ink_r4_moe_iq3_lut mk_iq3_lut
#else
  static LDS u32 ink_r4_moe_iq3_lut[256] __attribute__((aligned(16)));
#endif
  if (rows == 0u || nb < 8u || (nb & 7u) != 0u || !qw || !X || !Y) return;

  const u32 e = tokid();
  const u32 lo = list ? offs[e] : 0u;
  const u32 hi = list ? offs[e + 1u] : np;
  const u32 rbase = wgid() * 32u;
  if (lo >= hi || rbase >= rows) return;

  const u32 t0 = tid();
#ifndef IE_MEGA
  ink_r4_moe_iq3_lut[t0] = ie_rq_iq3xxs_grid[t0];
  lbarrier();
#endif

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 r = rbase + wv * 4u + k;
  const u32 rr = r < rows ? r : rows - 1u;
  const u32 mb = nb >> 3u;
  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * q_bytes);
  const G u8 *rowp = ebase + (unsigned long)rr * ((unsigned long)mb * 98ul);
  const u32 qo = 2u + 8u * s;
  const u32 auxo = 66u + 4u * s;
  const unsigned long x_scale_off = 32ul * (unsigned long)nb;

  for (u32 pb = lo; pb < hi; pb += 4u) {
    const u32 nj = (u32)__builtin_amdgcn_readfirstlane((int)(hi - pb < 4u ? hi - pb : 4u));
    const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)(list ? list[pb] : pb));
    const u32 p1 = nj > 1u ? (u32)__builtin_amdgcn_readfirstlane((int)(list ? list[pb + 1u] : pb + 1u)) : p0;
    const u32 p2 = nj > 2u ? (u32)__builtin_amdgcn_readfirstlane((int)(list ? list[pb + 2u] : pb + 2u)) : p0;
    const u32 p3 = nj > 3u ? (u32)__builtin_amdgcn_readfirstlane((int)(list ? list[pb + 3u] : pb + 3u)) : p0;

    const G u8 *xp0 = rq_ptr_u(X + (unsigned long)ink_r4_moe_div_x(p0, xdiv) * xs);
    const G u8 *xp1 = nj > 1u ? rq_ptr_u(X + (unsigned long)ink_r4_moe_div_x(p1, xdiv) * xs) : xp0;
    const G u8 *xp2 = nj > 2u ? rq_ptr_u(X + (unsigned long)ink_r4_moe_div_x(p2, xdiv) * xs) : xp0;
    const G u8 *xp3 = nj > 3u ? rq_ptr_u(X + (unsigned long)ink_r4_moe_div_x(p3, xdiv) * xs) : xp0;

    const G float *dap0 = (const G float *)(xp0 + x_scale_off);
    const G float *dap1 = (const G float *)(xp1 + x_scale_off);
    const G float *dap2 = (const G float *)(xp2 + x_scale_off);
    const G float *dap3 = (const G float *)(xp3 + x_scale_off);

    float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;

    const G u8 *bp = rowp;
    u32 d_c = ie_rq_a16(bp, 0u);
    u32 aux_c = ie_rq_a32(bp, auxo);
    u32 q01_c = ie_rq_a32(bp, qo);
    u32 q23_c = ie_rq_a32(bp, qo + 4u);
    __asm__ volatile("" : "+v"(d_c), "+v"(aux_c), "+v"(q01_c), "+v"(q23_c));

#pragma unroll 1
    for (u32 sb = 0u; sb < mb; sb++) {
      const u32 b = 8u * sb + s;
      const u32 j_acc = sb & 3u;
      const int has_nxt = (sb + 1u < mb);
      bp += 98u;

      /* Prefetch next superblock weights (overlaps current decode + all nj dots). */
      u32 d_n = 0u, aux_n = 0u, q01_n = 0u, q23_n = 0u;
      if (has_nxt) {
        d_n = ie_rq_a16(bp, 0u);
        aux_n = ie_rq_a32(bp, auxo);
        q01_n = ie_rq_a32(bp, qo);
        q23_n = ie_rq_a32(bp, qo + 4u);
      }

      /* Issue pair 0 X load + all 8 LDS grid LUT lookups together. */
      u32x4 x0_a = *(const G u32x4 *)(xp0 + 32u * b);
      u32x4 x1_a = *(const G u32x4 *)(xp0 + 32u * b + 16u);
      float dav_a = dap0[b];

      const u32 qp0 = q01_c & 0xFFFFu;
      const u32 qp1 = q01_c >> 16u;
      const u32 qp2 = q23_c & 0xFFFFu;
      const u32 qp3 = q23_c >> 16u;
      const u32 g_lo0 = ink_r4_moe_iq3_lut[qp0 & 255u];
      const u32 g_hi0 = ink_r4_moe_iq3_lut[qp0 >> 8u];
      const u32 g_lo1 = ink_r4_moe_iq3_lut[qp1 & 255u];
      const u32 g_hi1 = ink_r4_moe_iq3_lut[qp1 >> 8u];
      const u32 g_lo2 = ink_r4_moe_iq3_lut[qp2 & 255u];
      const u32 g_hi2 = ink_r4_moe_iq3_lut[qp2 >> 8u];
      const u32 g_lo3 = ink_r4_moe_iq3_lut[qp3 & 255u];
      const u32 g_hi3 = ink_r4_moe_iq3_lut[qp3 >> 8u];
      __builtin_amdgcn_sched_barrier(0);

      /* Decode IQ3_XXS scale & signed int8 weights ONCE for all nj pairs. */
      const float d = ie_rq_h2f(d_c);
      const float dw = d * (0.5f + (float)(aux_c >> 28u)) * 0.5f;
      const u32 s8_0 = rq_ksign7(aux_c & 127u);
      const u32 s8_1 = rq_ksign7((aux_c >> 7u) & 127u);
      const u32 s8_2 = rq_ksign7((aux_c >> 14u) & 127u);
      const u32 s8_3 = rq_ksign7((aux_c >> 21u) & 127u);
      const u32 wl0 = ie_rq_ssign(g_lo0, ie_rq_mask4(s8_0 & 15u));
      const u32 wh0 = ie_rq_ssign(g_hi0, ie_rq_mask4(s8_0 >> 4u));
      const u32 wl1 = ie_rq_ssign(g_lo1, ie_rq_mask4(s8_1 & 15u));
      const u32 wh1 = ie_rq_ssign(g_hi1, ie_rq_mask4(s8_1 >> 4u));
      const u32 wl2 = ie_rq_ssign(g_lo2, ie_rq_mask4(s8_2 & 15u));
      const u32 wh2 = ie_rq_ssign(g_hi2, ie_rq_mask4(s8_2 >> 4u));
      const u32 wl3 = ie_rq_ssign(g_lo3, ie_rq_mask4(s8_3 & 15u));
      const u32 wh3 = ie_rq_ssign(g_hi3, ie_rq_mask4(s8_3 >> 4u));

      u32x4 x0_b = (u32x4)(0u), x1_b = (u32x4)(0u);
      float dav_b = 0.0f;
      if (nj > 1u) {
        x0_b = *(const G u32x4 *)(xp1 + 32u * b);
        x1_b = *(const G u32x4 *)(xp1 + 32u * b + 16u);
        dav_b = dap1[b];
      }
      __builtin_amdgcn_sched_barrier(0);

      const float v0 = ink_r4_moe_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_a, x1_a, dav_a);
      INK_R4_MOE_ACC_ADD(a0_0, a0_1, a0_2, a0_3, j_acc, v0);

      if (nj > 1u) {
        if (nj > 2u) {
          x0_a = *(const G u32x4 *)(xp2 + 32u * b);
          x1_a = *(const G u32x4 *)(xp2 + 32u * b + 16u);
          dav_a = dap2[b];
        }
        __builtin_amdgcn_sched_barrier(0);
        const float v1 = ink_r4_moe_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_b, x1_b, dav_b);
        INK_R4_MOE_ACC_ADD(a1_0, a1_1, a1_2, a1_3, j_acc, v1);

        if (nj > 2u) {
          if (nj > 3u) {
            x0_b = *(const G u32x4 *)(xp3 + 32u * b);
            x1_b = *(const G u32x4 *)(xp3 + 32u * b + 16u);
            dav_b = dap3[b];
          }
          __builtin_amdgcn_sched_barrier(0);
          const float v2 = ink_r4_moe_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_a, x1_a, dav_a);
          INK_R4_MOE_ACC_ADD(a2_0, a2_1, a2_2, a2_3, j_acc, v2);

          if (nj > 3u) {
            __builtin_amdgcn_sched_barrier(0);
            const float v3 = ink_r4_moe_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_b, x1_b, dav_b);
            INK_R4_MOE_ACC_ADD(a3_0, a3_1, a3_2, a3_3, j_acc, v3);
          }
        }
      }
      __builtin_amdgcn_sched_barrier(0);
      d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
    }

    /* Exact wave_tree_f reduction + store for each active pair in this chunk. */
    const float acc0 = ink_r4_moe_oct_tree_f((a0_0 + a0_2) + (a0_1 + a0_3));
    if (s == 0u && r < rows) {
      G float *yp0 = (G float *)((G u8 *)Y + (unsigned long)p0 * ys);
      const G float *rp0 = res ? (const G float *)((const G u8 *)res + (unsigned long)p0 * rs) : 0;
      yp0[r] = rp0 ? rp0[r] + acc0 : acc0;
    }
    if (nj > 1u) {
      const float acc1 = ink_r4_moe_oct_tree_f((a1_0 + a1_2) + (a1_1 + a1_3));
      if (s == 0u && r < rows) {
        G float *yp1 = (G float *)((G u8 *)Y + (unsigned long)p1 * ys);
        const G float *rp1 = res ? (const G float *)((const G u8 *)res + (unsigned long)p1 * rs) : 0;
        yp1[r] = rp1 ? rp1[r] + acc1 : acc1;
      }
    }
    if (nj > 2u) {
      const float acc2 = ink_r4_moe_oct_tree_f((a2_0 + a2_2) + (a2_1 + a2_3));
      if (s == 0u && r < rows) {
        G float *yp2 = (G float *)((G u8 *)Y + (unsigned long)p2 * ys);
        const G float *rp2 = res ? (const G float *)((const G u8 *)res + (unsigned long)p2 * rs) : 0;
        yp2[r] = rp2 ? rp2[r] + acc2 : acc2;
      }
    }
    if (nj > 3u) {
      const float acc3 = ink_r4_moe_oct_tree_f((a3_0 + a3_2) + (a3_1 + a3_3));
      if (s == 0u && r < rows) {
        G float *yp3 = (G float *)((G u8 *)Y + (unsigned long)p3 * ys);
        const G float *rp3 = res ? (const G float *)((const G u8 *)res + (unsigned long)p3 * rs) : 0;
        yp3[r] = rp3 ? rp3[r] + acc3 : acc3;
      }
    }
  }
}

/* ============================================================================
 * Kernel 2: T=2..16 Verify MoE IQ4_NL (4 rows/wave, 8 lanes/row, 32 rows/WG)
 *   - Immediate early-return on lo >= hi, 0 LDS, coalesced u32x4 loads.
 *   - Reads & decodes IQ4_NL weights ONCE per chunk of up to 4 routed pairs.
 *   - 100% bitwise-exact vs ie_gexp_t (exact 4-acc wave_tree_f / half_tree_f).
 * ========================================================================== */

static inline __attribute__((always_inline)) int ink_r4_moe_iq4_dot(
    const u32 wl0, const u32 wh0, const u32 wl1, const u32 wh1,
    const u32 wl2, const u32 wh2, const u32 wl3, const u32 wh3,
    const u32x4 a0, const u32x4 a1) {
  u32 d = 0u;
  d = ie_dot4_ss(ie_dot4_ss(d, wl0, a0.x), wh0, a1.x);
  d = ie_dot4_ss(ie_dot4_ss(d, wl1, a0.y), wh1, a1.y);
  d = ie_dot4_ss(ie_dot4_ss(d, wl2, a0.z), wh2, a1.z);
  d = ie_dot4_ss(ie_dot4_ss(d, wl3, a0.w), wh3, a1.w);
  return (int)d;
}

KERNEL_LOOP ie_inkling_gexp_iq4nl_ver_r4(const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
                                         u32 rows, u32 nb, const G u32 *list, const G u32 *offs, u32 np,
                                         u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv, u32 kind,
                                         const G u8 *res, u32 rs) {
#pragma clang fp contract(off)
  (void)kind;
  if (rows == 0u || nb == 0u || !qw0 || !qs0 || !X || !Y) return;

  const u32 e = tokid();
  const u32 lo = list ? offs[e] : 0u;
  const u32 hi = list ? offs[e + 1u] : np;
  const u32 rbase = wgid() * 32u;
  if (lo >= hi || rbase >= rows) return;

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 r_wave = rbase + wv * 4u;
  if (r_wave >= rows) return;

  const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
  const u32 r = r_wave + sub;
  const u32 rr = r < rows ? r : rows - 1u;

  const G u8 *w_base = rq_ptr_u((const G u8 *)(qw0 + (unsigned long)e * qwn));
  const G u8 *q_base = rq_ptr_u((const G u8 *)(qs0 + (unsigned long)e * qsn));
  const G u8 *rw = w_base + (unsigned long)rr * ((unsigned long)nb * 16ul);
  const G f16 *rs_p = (const G f16 *)q_base + (unsigned long)rr * nb;
  const u32 nstep = (nb + 7u) >> 3u;
  const unsigned long x_scale_off = 32ul * (unsigned long)nb;

  for (u32 pb = lo; pb < hi; pb += 4u) {
    const u32 nj = (u32)__builtin_amdgcn_readfirstlane((int)(hi - pb < 4u ? hi - pb : 4u));
    const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)(list ? list[pb] : pb));
    const u32 p1 = nj > 1u ? (u32)__builtin_amdgcn_readfirstlane((int)(list ? list[pb + 1u] : pb + 1u)) : p0;
    const u32 p2 = nj > 2u ? (u32)__builtin_amdgcn_readfirstlane((int)(list ? list[pb + 2u] : pb + 2u)) : p0;
    const u32 p3 = nj > 3u ? (u32)__builtin_amdgcn_readfirstlane((int)(list ? list[pb + 3u] : pb + 3u)) : p0;

    const G u8 *xp0 = rq_ptr_u(X + (unsigned long)ink_r4_moe_div_x(p0, xdiv) * xs);
    const G u8 *xp1 = nj > 1u ? rq_ptr_u(X + (unsigned long)ink_r4_moe_div_x(p1, xdiv) * xs) : xp0;
    const G u8 *xp2 = nj > 2u ? rq_ptr_u(X + (unsigned long)ink_r4_moe_div_x(p2, xdiv) * xs) : xp0;
    const G u8 *xp3 = nj > 3u ? rq_ptr_u(X + (unsigned long)ink_r4_moe_div_x(p3, xdiv) * xs) : xp0;

    const G u32 *aw0 = (const G u32 *)xp0;
    const G u32 *aw1 = (const G u32 *)xp1;
    const G u32 *aw2 = (const G u32 *)xp2;
    const G u32 *aw3 = (const G u32 *)xp3;
    const G float *da0 = (const G float *)(xp0 + x_scale_off);
    const G float *da1 = (const G float *)(xp1 + x_scale_off);
    const G float *da2 = (const G float *)(xp2 + x_scale_off);
    const G float *da3 = (const G float *)(xp3 + x_scale_off);

    float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;

    /* Prologue: prefetch step 0 weights (blocks 0..7). */
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

      /* Prefetch next step weights (overlaps current decode + all nj dots). */
      u32x4 w_n = (u32x4)(0u);
      f16 sc_n = (f16)0.0f;
      if (bn < nb) {
        w_n = *(const G u32x4 *)(rw + bn * 16u);
        sc_n = rs_p[bn];
      }

      if (b_c < nb) {
        /* Issue pair 0 X load while decoding IQ4_NL weights ONCE for all nj pairs. */
        u32x4 x0_a = *(const G u32x4 *)(aw0 + b_c * 8u);
        u32x4 x1_a = *(const G u32x4 *)(aw0 + b_c * 8u + 4u);
        float d_a = da0[b_c];

        const float sc_f = (float)sc_c;
        const u32 wl0 = ie_iq4_map(w_c.x & 0x0F0F0F0Fu);
        const u32 wh0 = ie_iq4_map((w_c.x >> 4u) & 0x0F0F0F0Fu);
        const u32 wl1 = ie_iq4_map(w_c.y & 0x0F0F0F0Fu);
        const u32 wh1 = ie_iq4_map((w_c.y >> 4u) & 0x0F0F0F0Fu);
        const u32 wl2 = ie_iq4_map(w_c.z & 0x0F0F0F0Fu);
        const u32 wh2 = ie_iq4_map((w_c.z >> 4u) & 0x0F0F0F0Fu);
        const u32 wl3 = ie_iq4_map(w_c.w & 0x0F0F0F0Fu);
        const u32 wh3 = ie_iq4_map((w_c.w >> 4u) & 0x0F0F0F0Fu);

        u32x4 x0_b = (u32x4)(0u), x1_b = (u32x4)(0u);
        float d_b = 0.0f;
        if (nj > 1u) {
          x0_b = *(const G u32x4 *)(aw1 + b_c * 8u);
          x1_b = *(const G u32x4 *)(aw1 + b_c * 8u + 4u);
          d_b = da1[b_c];
        }
        __builtin_amdgcn_sched_barrier(0);

        const float v0 = sc_f * (float)ink_r4_moe_iq4_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_a, x1_a);
        INK_R4_MOE_ACC_FMA(a0_0, a0_1, a0_2, a0_3, j_acc, v0, d_a);

        if (nj > 1u) {
          if (nj > 2u) {
            x0_a = *(const G u32x4 *)(aw2 + b_c * 8u);
            x1_a = *(const G u32x4 *)(aw2 + b_c * 8u + 4u);
            d_a = da2[b_c];
          }
          __builtin_amdgcn_sched_barrier(0);
          const float v1 = sc_f * (float)ink_r4_moe_iq4_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_b, x1_b);
          INK_R4_MOE_ACC_FMA(a1_0, a1_1, a1_2, a1_3, j_acc, v1, d_b);

          if (nj > 2u) {
            if (nj > 3u) {
              x0_b = *(const G u32x4 *)(aw3 + b_c * 8u);
              x1_b = *(const G u32x4 *)(aw3 + b_c * 8u + 4u);
              d_b = da3[b_c];
            }
            __builtin_amdgcn_sched_barrier(0);
            const float v2 = sc_f * (float)ink_r4_moe_iq4_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_a, x1_a);
            INK_R4_MOE_ACC_FMA(a2_0, a2_1, a2_2, a2_3, j_acc, v2, d_a);

            if (nj > 3u) {
              __builtin_amdgcn_sched_barrier(0);
              const float v3 = sc_f * (float)ink_r4_moe_iq4_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_b, x1_b);
              INK_R4_MOE_ACC_FMA(a3_0, a3_1, a3_2, a3_3, j_acc, v3, d_b);
            }
          }
        }
      }
      __builtin_amdgcn_sched_barrier(0);
      w_c = w_n; sc_c = sc_n;
    }

    /* Exact wave_tree_f (nb > 16) / half_tree_f (nb <= 16) reduction + epilogue store. */
    const int full_wave = (nb > 16u);
    const float acc0 = ink_r4_moe_oct_tree_f(full_wave ? ((a0_0 + a0_2) + (a0_1 + a0_3)) : (a0_0 + a0_1));
    if (sl == 0u && r < rows) {
      G float *yp0 = (G float *)(Y + (unsigned long)p0 * ys);
      const G float *rp0 = res ? (const G float *)(res + (unsigned long)p0 * rs) : 0;
      yp0[r] = epilogue(acc0, r, 0, rp0);
    }
    if (nj > 1u) {
      const float acc1 = ink_r4_moe_oct_tree_f(full_wave ? ((a1_0 + a1_2) + (a1_1 + a1_3)) : (a1_0 + a1_1));
      if (sl == 0u && r < rows) {
        G float *yp1 = (G float *)(Y + (unsigned long)p1 * ys);
        const G float *rp1 = res ? (const G float *)(res + (unsigned long)p1 * rs) : 0;
        yp1[r] = epilogue(acc1, r, 0, rp1);
      }
    }
    if (nj > 2u) {
      const float acc2 = ink_r4_moe_oct_tree_f(full_wave ? ((a2_0 + a2_2) + (a2_1 + a2_3)) : (a2_0 + a2_1));
      if (sl == 0u && r < rows) {
        G float *yp2 = (G float *)(Y + (unsigned long)p2 * ys);
        const G float *rp2 = res ? (const G float *)(res + (unsigned long)p2 * rs) : 0;
        yp2[r] = epilogue(acc2, r, 0, rp2);
      }
    }
    if (nj > 3u) {
      const float acc3 = ink_r4_moe_oct_tree_f(full_wave ? ((a3_0 + a3_2) + (a3_1 + a3_3)) : (a3_0 + a3_1));
      if (sl == 0u && r < rows) {
        G float *yp3 = (G float *)(Y + (unsigned long)p3 * ys);
        const G float *rp3 = res ? (const G float *)(res + (unsigned long)p3 * rs) : 0;
        yp3[r] = epilogue(acc3, r, 0, rp3);
      }
    }
  }
}

/* ============================================================================
 * Kernel 3: T=2..16 Verify MoE F32 Router (4 rows/WG, 2 waves/row, 32 WGs)
 *   - Reads each f32x4 weight ONCE per 8-token chunk (T-chunked by 8 for T=9..16).
 *   - 64 threads x f32x4 = 256 cols/step (exact 9 steps on cols=2304).
 *   - Exact __builtin_fmaf + wave_tree_f + 2-wave LDS sum (128 B) + epilogue().
 * ========================================================================== */

#define INK_R4_MOE_FMA4(acc_, w_, x_)            \
  do {                                           \
    (acc_) = __builtin_fmaf((w_).x, (x_).x, (acc_)); \
    (acc_) = __builtin_fmaf((w_).y, (x_).y, (acc_)); \
    (acc_) = __builtin_fmaf((w_).z, (x_).z, (acc_)); \
    (acc_) = __builtin_fmaf((w_).w, (x_).w, (acc_)); \
  } while (0)

#define INK_R4_MOE_RTR_STORE(k_, acc_)                                                                \
  do {                                                                                                \
    const u32 tk_ = t0 + (k_);                                                                        \
    const float sum_ = (acc_) + ink_r4_moe_rtr_red[row_local][(k_)];                                  \
    G float *yp_ = (G float *)((G u8 *)y + (unsigned long)tk_ * ys);                                  \
    const G float *rp_ = res ? (const G float *)((const G u8 *)res + (unsigned long)tk_ * rs) : 0;    \
    yp_[r] = epilogue(sum_, r, bias, rp_);                                                            \
  } while (0)

KERNEL_LOOP ie_inkling_gemv_f32_router_ver_r4(const G float *w, const G float *x, G float *y, u32 rows, u32 cols,
                                              const G float *bias, const G float *res, u32 xs, u32 ys, u32 rs, u32 T) {
#ifdef IE_MEGA
#define ink_r4_moe_rtr_red (mk_ink_lds.router_ver.red)
#else
  static LDS float ink_r4_moe_rtr_red[4][8] __attribute__((aligned(16)));
#endif
  if (rows == 0u || cols == 0u || T == 0u || wgid() * 4u >= rows || !w || !x || !y) return;

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 row_local = wv >> 1u;
  const u32 w_sub = wv & 1u;
  const u32 r = wgid() * 4u + row_local;
  const u32 rr = r < rows ? r : rows - 1u;
  const u32 lane64 = (w_sub << 5u) | l;

  const G float *rowp = (const G float *)rq_ptr_u((const G u8 *)(w + (unsigned long)rr * cols));
  const u32 cols4 = cols >> 2u;

  for (u32 t0 = 0u; t0 < T; t0 += 8u) {
    const u32 nt = (u32)__builtin_amdgcn_readfirstlane((int)(T - t0 < 8u ? T - t0 : 8u));
    const G float *xp0 = (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 0u) * xs);
    const G float *xp1 = nt > 1u ? (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 1u) * xs) : xp0;
    const G float *xp2 = nt > 2u ? (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 2u) * xs) : xp0;
    const G float *xp3 = nt > 3u ? (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 3u) * xs) : xp0;
    const G float *xp4 = nt > 4u ? (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 4u) * xs) : xp0;
    const G float *xp5 = nt > 5u ? (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 5u) * xs) : xp0;
    const G float *xp6 = nt > 6u ? (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 6u) * xs) : xp0;
    const G float *xp7 = nt > 7u ? (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 7u) * xs) : xp0;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

    if (lane64 < cols4) {
      f32x4 w_c = ink_r4_moe_ld_f4(rowp + (unsigned long)lane64 * 4ul);
      __builtin_amdgcn_sched_barrier(0);

#pragma unroll 1
      for (u32 c4 = lane64; c4 < cols4; c4 += 64u) {
        const u32 cn = c4 + 64u;
        f32x4 w_n = (f32x4)(0.0f);
        if (cn < cols4) w_n = ink_r4_moe_ld_f4(rowp + (unsigned long)cn * 4ul);

        const unsigned long off = (unsigned long)c4 * 4ul;
        f32x4 xv0 = ink_r4_moe_ld_f4(xp0 + off);
        f32x4 xv1 = nt > 1u ? ink_r4_moe_ld_f4(xp1 + off) : (f32x4)(0.0f);
        __builtin_amdgcn_sched_barrier(0);
        INK_R4_MOE_FMA4(a0, w_c, xv0);

        if (nt > 1u) {
          if (nt > 2u) xv0 = ink_r4_moe_ld_f4(xp2 + off);
          __builtin_amdgcn_sched_barrier(0);
          INK_R4_MOE_FMA4(a1, w_c, xv1);
          if (nt > 2u) {
            if (nt > 3u) xv1 = ink_r4_moe_ld_f4(xp3 + off);
            __builtin_amdgcn_sched_barrier(0);
            INK_R4_MOE_FMA4(a2, w_c, xv0);
            if (nt > 3u) {
              if (nt > 4u) xv0 = ink_r4_moe_ld_f4(xp4 + off);
              __builtin_amdgcn_sched_barrier(0);
              INK_R4_MOE_FMA4(a3, w_c, xv1);
              if (nt > 4u) {
                if (nt > 5u) xv1 = ink_r4_moe_ld_f4(xp5 + off);
                __builtin_amdgcn_sched_barrier(0);
                INK_R4_MOE_FMA4(a4, w_c, xv0);
                if (nt > 5u) {
                  if (nt > 6u) xv0 = ink_r4_moe_ld_f4(xp6 + off);
                  __builtin_amdgcn_sched_barrier(0);
                  INK_R4_MOE_FMA4(a5, w_c, xv1);
                  if (nt > 6u) {
                    if (nt > 7u) xv1 = ink_r4_moe_ld_f4(xp7 + off);
                    __builtin_amdgcn_sched_barrier(0);
                    INK_R4_MOE_FMA4(a6, w_c, xv0);
                    if (nt > 7u) {
                      __builtin_amdgcn_sched_barrier(0);
                      INK_R4_MOE_FMA4(a7, w_c, xv1);
                    }
                  }
                }
              }
            }
          }
        }
        __builtin_amdgcn_sched_barrier(0);
        w_c = w_n;
      }
    }

    /* Scalar tail when cols % 4 != 0 (skipped via uniform branch on cols=2304). */
    if (cols & 3u) {
      const u32 ct = (cols4 << 2u) + lane64;
      if (ct < cols) {
        const float ws = rowp[ct];
        a0 = __builtin_fmaf(ws, xp0[ct], a0);
        if (nt > 1u) a1 = __builtin_fmaf(ws, xp1[ct], a1);
        if (nt > 2u) a2 = __builtin_fmaf(ws, xp2[ct], a2);
        if (nt > 3u) a3 = __builtin_fmaf(ws, xp3[ct], a3);
        if (nt > 4u) a4 = __builtin_fmaf(ws, xp4[ct], a4);
        if (nt > 5u) a5 = __builtin_fmaf(ws, xp5[ct], a5);
        if (nt > 6u) a6 = __builtin_fmaf(ws, xp6[ct], a6);
        if (nt > 7u) a7 = __builtin_fmaf(ws, xp7[ct], a7);
      }
    }

    /* Wave-level tree reduction for active tokens. */
    a0 = wave_tree_f(a0);
    if (nt > 1u) a1 = wave_tree_f(a1);
    if (nt > 2u) a2 = wave_tree_f(a2);
    if (nt > 3u) a3 = wave_tree_f(a3);
    if (nt > 4u) a4 = wave_tree_f(a4);
    if (nt > 5u) a5 = wave_tree_f(a5);
    if (nt > 6u) a6 = wave_tree_f(a6);
    if (nt > 7u) a7 = wave_tree_f(a7);

    /* Combine the 2 waves of each row via 128 B LDS. */
    if (w_sub == 1u && l == 0u) {
      ink_r4_moe_rtr_red[row_local][0] = a0;
      if (nt > 1u) ink_r4_moe_rtr_red[row_local][1] = a1;
      if (nt > 2u) ink_r4_moe_rtr_red[row_local][2] = a2;
      if (nt > 3u) ink_r4_moe_rtr_red[row_local][3] = a3;
      if (nt > 4u) ink_r4_moe_rtr_red[row_local][4] = a4;
      if (nt > 5u) ink_r4_moe_rtr_red[row_local][5] = a5;
      if (nt > 6u) ink_r4_moe_rtr_red[row_local][6] = a6;
      if (nt > 7u) ink_r4_moe_rtr_red[row_local][7] = a7;
    }
    lbarrier();

    if (w_sub == 0u && l == 0u && r < rows) {
      INK_R4_MOE_RTR_STORE(0u, a0);
      if (nt > 1u) INK_R4_MOE_RTR_STORE(1u, a1);
      if (nt > 2u) INK_R4_MOE_RTR_STORE(2u, a2);
      if (nt > 3u) INK_R4_MOE_RTR_STORE(3u, a3);
      if (nt > 4u) INK_R4_MOE_RTR_STORE(4u, a4);
      if (nt > 5u) INK_R4_MOE_RTR_STORE(5u, a5);
      if (nt > 6u) INK_R4_MOE_RTR_STORE(6u, a6);
      if (nt > 7u) INK_R4_MOE_RTR_STORE(7u, a7);
    }
    if (t0 + 8u < T) lbarrier();
  }
}

#ifdef IE_MEGA
#undef ink_r4_moe_iq3_lut
#undef ink_r4_moe_rtr_red
#endif
#undef INK_R4_MOE_ACC_ADD
#undef INK_R4_MOE_ACC_FMA
#undef INK_R4_MOE_FMA4
#undef INK_R4_MOE_RTR_STORE
