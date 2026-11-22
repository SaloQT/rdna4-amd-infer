/* ============================================================================
 * Inkling-2 Round 1 Group 5: Q8_0 Projection Kernels (gfx1201, wave32, NT=256)
 *
 * Exported kernels (exact parameter lists of ie_gemm_q8r / ie_gemm_q8 / ie_gemv_q8q8):
 *
 * 1. KERNEL ie_inkling_gemm_q8_small_r1(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
 *                                       u32 rows, u32 nb, const G float *bias, const G float *res,
 *                                       u32 T, u32 xs, u32 ys, u32 rs)
 *    - Grid:   (((rows + 15u) / 16u) * ((T + 15u) / 16u), 1, 1)  [pad 253..256 -> 257 ok]
 *    - Guard:  T >= 17u && rows > 0u && rows <= 128u (also safe/fast up to rows <= 384u for wrel)
 *              && nb >= 4u && (nb & 3u) == 0u && nb <= 65536u && xs <= 0x0FFFFFFFu
 *    - Target: prefill wsink (24x2304), k/v_conv_proj (32x2304), loop_down (64x4608), wrel (384x2304)
 *    - Parity: bitwise-exact (0-bit-diff) vs ie_gemm_q8r
 *
 * 2. KERNEL ie_inkling_gemm_q8_nb2_r1(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
 *                                     u32 rows, u32 nb, const G float *bias, const G float *res,
 *                                     u32 T, u32 xs, u32 ys, u32 rs)
 *    - Grid:   (((rows + 127u) / 128u) * ((T + 63u) / 64u), 1, 1)
 *    - Guard:  T >= 17u && rows > 0u && nb == 2u
 *    - Target: prefill loop_wfg (4608x64, nb=2) — swapped WMMA (A=X, B=W) for 100% coalesced y stores
 *    - Parity: bitwise-exact (0-bit-diff) vs ie_gemm_q8
 *
 * 3. KERNEL ie_inkling_gemv_q8_nb2_r1(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
 *                                     u32 rows, u32 nb, const G float *bias, const G float *res,
 *                                     const G float *nw, G float *ny, G u8 *nq, float eps, G u32 *count)
 *    - Grid:   ((rows + 63u) / 64u, 1, 1)
 *    - Guard:  T == 1u && rows > 0u && nb == 2u
 *    - Target: decode loop_wfg (4608x64, nb=2) — 8 rows/wave, 4 lanes/row (100% lane util, 72 WGs/64 CUs)
 *    - Parity: bitwise-exact (0-bit-diff) vs ie_gemv_q8q8
 *
 * 4. KERNEL ie_inkling_gemv_q8_small_r1(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
 *                                       u32 rows, u32 nb, const G float *bias, const G float *res,
 *                                       const G float *nw, G float *ny, G u8 *nq, float eps, G u32 *count)
 *    - Grid:   (rows, 1, 1)
 *    - Guard:  T == 1u && rows > 0u && rows <= 64u && nb >= 32u
 *    - Target: decode wsink (24x2304), k/v_conv_proj (32x2304), loop_down (64x4608) — 1 WG/row (8x WGs)
 *    - Parity: bitwise-exact (0-bit-diff) vs ie_gemv_q8q8
 * ========================================================================== */

#define INK_R1_G05_TILE 16u
#define INK_R1_G05_KSTEP 8u
#define INK_R1_G05_SC_STRIDE 12u
#define INK_R1_G05_DOT_BUF 2048u /* 8 waves * 16 tokens * 16 rows */
#define INK_R1_G05_SC_BUF 192u   /* 16 rows/tokens * 12-float padded stride */
#define INK_R1_G05_NB2_R 128u
#define INK_R1_G05_NB2_T 64u
#define INK_R1_G05_NB2_XLS 80u   /* 64 B (nb=2) + 16 B pad -> 0 bank conflicts on b128 reads */

typedef _Float16 ink_r1_g05_f16x2 __attribute__((ext_vector_type(2)));
typedef unsigned int ink_r1_g05_u32x2 __attribute__((ext_vector_type(2)));

struct __attribute__((packed, aligned(4))) ink_r1_g05_u32x4p {
  u32x4 v;
};

struct __attribute__((packed, aligned(4))) ink_r1_g05_u32x2p {
  ink_r1_g05_u32x2 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r1_g05_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r1_g05_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) ink_r1_g05_u32x2 ink_r1_g05_ld2(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r1_g05_u32x2p *)p)->v;
}

#ifndef IE_MEGA
/* ============================================================================
 * Kernel 1: ie_inkling_gemm_q8_small_r1
 * 16x16 WMMA tile, 8-way wave-cooperative K-step (KSTEP=8, plus 4-block tail
 * when nb % 8 == 4). Direct-VGPR W & X loads, zero-bank-conflict XOR-swizzled
 * double-buffered LDS, exact-order FMA reduction (0-bit-diff vs ie_gemm_q8r).
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r1_g05_compute_store8(
    const u32x4 w_vec, const u32x4 x_vec, const u32x4 sc_vec, const u32 w,
    const u32 r16, const u32 h, const u32 st_col0, const u32 st_col1,
    LDSP float *dot_b, LDSP float *sw_b, LDSP float *dx_b) {
  const v8i zm = (v8i)(0x4B400000);
  const v2i a0 = {(int)w_vec.x, (int)w_vec.y};
  const v2i a1 = {(int)w_vec.z, (int)w_vec.w};
  const v2i b0 = {(int)x_vec.x, (int)x_vec.y};
  const v2i b1 = {(int)x_vec.z, (int)x_vec.w};
  v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0, zm, 0);
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1, c, 0);
  const f8 df = __builtin_bit_cast(f8, c) - 12582912.0f;
  LDSP float *dp = dot_b + (w << 8u) + (r16 << 4u);
  *(LDSP f32x4 *)(dp + st_col0) = (f32x4){df[0], df[1], df[2], df[3]};
  *(LDSP f32x4 *)(dp + st_col1) = (f32x4){df[4], df[5], df[6], df[7]};

  if (w == 0u) {
    if (h == 0u) {
      const u32 u0 = sc_vec.x, u1 = sc_vec.y, u2 = sc_vec.z, u3 = sc_vec.w;
      const ink_r1_g05_f16x2 p0 = __builtin_bit_cast(ink_r1_g05_f16x2, u0);
      const ink_r1_g05_f16x2 p1 = __builtin_bit_cast(ink_r1_g05_f16x2, u1);
      const ink_r1_g05_f16x2 p2 = __builtin_bit_cast(ink_r1_g05_f16x2, u2);
      const ink_r1_g05_f16x2 p3 = __builtin_bit_cast(ink_r1_g05_f16x2, u3);
      LDSP float *sp = sw_b + r16 * INK_R1_G05_SC_STRIDE;
      *(LDSP f32x4 *)(sp + 0u) = (f32x4){(float)p0.x, (float)p0.y, (float)p1.x, (float)p1.y};
      *(LDSP f32x4 *)(sp + 4u) = (f32x4){(float)p2.x, (float)p2.y, (float)p3.x, (float)p3.y};
    }
  } else if (w == 1u) {
    const u32 u0 = sc_vec.x, u1 = sc_vec.y, u2 = sc_vec.z, u3 = sc_vec.w;
    const f32x4 dxv = {__builtin_bit_cast(float, u0), __builtin_bit_cast(float, u1),
                       __builtin_bit_cast(float, u2), __builtin_bit_cast(float, u3)};
    *(LDSP f32x4 *)(dx_b + r16 * INK_R1_G05_SC_STRIDE + 4u * h) = dxv;
  }
}

static inline __attribute__((always_inline)) float ink_r1_g05_reduce_buf8(
    const LDSP float *dot_b, const LDSP float *sw_b, const LDSP float *dx_b,
    const u32 t_out, const u32 r_out, const u32 r_sw, float acc) {
  const LDSP float *dp = dot_b + (t_out << 4u) + r_sw;
  const LDSP float *sp = sw_b + r_out * INK_R1_G05_SC_STRIDE;
  const LDSP float *xp = dx_b + t_out * INK_R1_G05_SC_STRIDE;
  const f32x4 sw0 = *(const LDSP f32x4 *)(sp + 0u);
  const f32x4 dx0 = *(const LDSP f32x4 *)(xp + 0u);
  const float d0 = dp[0u << 8u];
  const float d1 = dp[1u << 8u];
  const float d2 = dp[2u << 8u];
  const float d3 = dp[3u << 8u];
  const f32x4 sw1 = *(const LDSP f32x4 *)(sp + 4u);
  const f32x4 dx1 = *(const LDSP f32x4 *)(xp + 4u);
  const float d4 = dp[4u << 8u];
  const float d5 = dp[5u << 8u];
  const float d6 = dp[6u << 8u];
  const float d7 = dp[7u << 8u];
  __builtin_amdgcn_sched_barrier(0);
  const f32x4 sc0 = sw0 * dx0;
  acc = __builtin_fmaf(sc0.x, d0, acc);
  acc = __builtin_fmaf(sc0.y, d1, acc);
  acc = __builtin_fmaf(sc0.z, d2, acc);
  acc = __builtin_fmaf(sc0.w, d3, acc);
  const f32x4 sc1 = sw1 * dx1;
  acc = __builtin_fmaf(sc1.x, d4, acc);
  acc = __builtin_fmaf(sc1.y, d5, acc);
  acc = __builtin_fmaf(sc1.z, d6, acc);
  acc = __builtin_fmaf(sc1.w, d7, acc);
  return acc;
}

KERNEL ie_inkling_gemm_q8_small_r1(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                   u32 rows, u32 nb, const G float *bias, const G float *res,
                                   u32 T, u32 xs, u32 ys, u32 rs) {
  static LDS float ink_r1_g05_lds_dot[2u * INK_R1_G05_DOT_BUF] __attribute__((aligned(16)));
  static LDS float ink_r1_g05_lds_sw[2u * INK_R1_G05_SC_BUF] __attribute__((aligned(16)));
  static LDS float ink_r1_g05_lds_dx[2u * INK_R1_G05_SC_BUF] __attribute__((aligned(16)));
  _Static_assert(sizeof(f16) == 2u, "ink_r1_g05 f16 must be 2 bytes");
  _Static_assert(sizeof(ink_r1_g05_lds_dot) + sizeof(ink_r1_g05_lds_sw) + sizeof(ink_r1_g05_lds_dx) == 19456u,
                 "ink_r1_g05 small GEMM LDS must be 19456 B");

  if (T == 0u || rows == 0u || nb < 4u || (nb & 3u) != 0u || !qw || !qs || !xq || !y) return;
  const u32 ntt = (T + INK_R1_G05_TILE - 1u) >> 4u;
  const u32 tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 rbase = rt * INK_R1_G05_TILE, tbase = tt * INK_R1_G05_TILE;
  if (rbase >= rows || tbase >= T) return;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;

  const u32 r_out = t0 & 15u;
  const u32 t_out = t0 >> 4u;
  const u32 r_sw = r_out ^ ((t_out & 6u) << 1u);

  const u32 st_swiz = (r16 & 6u) << 1u;
  const u32 st_col0 = (8u * h) ^ st_swiz;
  const u32 st_col1 = (8u * h + 4u) ^ st_swiz;

  const u32 dr_max = rows - 1u - rbase;
  const u32 dt_max = T - 1u - tbase;
  const u32 dr = r16 <= dr_max ? r16 : dr_max;
  const u32 dt = r16 <= dt_max ? r16 : dt_max;

  const G u8 *qw_tile = (const G u8 *)qw + (unsigned long)rbase * (unsigned long)nb * 32ul;
  const G u8 *xq_tile = xq + (unsigned long)tbase * (unsigned long)xs;
  const G u8 *sc_tile = (w == 0u) ? ((const G u8 *)qs + (unsigned long)rbase * (unsigned long)nb * 2ul)
                                  : (xq_tile + (unsigned long)nb * 32ul);
  const u32 sc_shift = (w == 0u) ? 1u : 2u;

  const u32 woff = dr * (nb * 32u) + w * 32u + 16u * h;
  const u32 xoff = dt * xs + w * 32u + 16u * h;
  const u32 scoff = (w == 0u) ? (dr * (nb * 2u)) : (dt * xs + 16u * h);

  float acc = 0.0f;
  __asm__ volatile("" : "+v"(acc));

  const u32 nb_main = nb & ~7u;
  u32 bf = 0u;
  if (nb_main > 0u) {
    u32x4 cur_w = ink_r1_g05_ld4(qw_tile + (unsigned long)woff);
    u32x4 cur_x = ink_r1_g05_ld4(xq_tile + (unsigned long)xoff);
    u32x4 cur_sc = (u32x4)(0u);
    if (w < 2u) {
      if (w == 1u || h == 0u) cur_sc = ink_r1_g05_ld4(sc_tile + (unsigned long)scoff);
    }
    __asm__ volatile("" : "+v"(cur_w), "+v"(cur_x));
    if (w < 2u) __asm__ volatile("" : "+v"(cur_sc));

    u32x4 nxt_w = (u32x4)(0u), nxt_x = (u32x4)(0u), nxt_sc = (u32x4)(0u);
    if (nb_main > INK_R1_G05_KSTEP) {
      const G u8 *qw_1 = qw_tile + (unsigned long)INK_R1_G05_KSTEP * 32ul;
      const G u8 *xq_1 = xq_tile + (unsigned long)INK_R1_G05_KSTEP * 32ul;
      const G u8 *sc_1 = sc_tile + ((unsigned long)INK_R1_G05_KSTEP << sc_shift);
      nxt_w = ink_r1_g05_ld4(qw_1 + (unsigned long)woff);
      nxt_x = ink_r1_g05_ld4(xq_1 + (unsigned long)xoff);
      if (w < 2u) {
        if (w == 1u || h == 0u) nxt_sc = ink_r1_g05_ld4(sc_1 + (unsigned long)scoff);
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    ink_r1_g05_compute_store8(cur_w, cur_x, cur_sc, w, r16, h, st_col0, st_col1,
                              ink_r1_g05_lds_dot, ink_r1_g05_lds_sw, ink_r1_g05_lds_dx);
    lbarrier();

#pragma unroll 1
    for (u32 kb0 = INK_R1_G05_KSTEP; kb0 < nb_main; kb0 += INK_R1_G05_KSTEP) {
      acc = ink_r1_g05_reduce_buf8(ink_r1_g05_lds_dot + bf * INK_R1_G05_DOT_BUF,
                                   ink_r1_g05_lds_sw + bf * INK_R1_G05_SC_BUF,
                                   ink_r1_g05_lds_dx + bf * INK_R1_G05_SC_BUF,
                                   t_out, r_out, r_sw, acc);

      cur_w = nxt_w;
      cur_x = nxt_x;
      if (w < 2u) cur_sc = nxt_sc;
      __asm__ volatile("" : "+v"(cur_w), "+v"(cur_x));
      if (w < 2u) __asm__ volatile("" : "+v"(cur_sc));

      const u32 kbn = kb0 + INK_R1_G05_KSTEP;
      if (kbn < nb_main) {
        const G u8 *qw_n = qw_tile + (unsigned long)kbn * 32ul;
        const G u8 *xq_n = xq_tile + (unsigned long)kbn * 32ul;
        const G u8 *sc_n = sc_tile + ((unsigned long)kbn << sc_shift);
        nxt_w = ink_r1_g05_ld4(qw_n + (unsigned long)woff);
        nxt_x = ink_r1_g05_ld4(xq_n + (unsigned long)xoff);
        if (w < 2u) {
          if (w == 1u || h == 0u) nxt_sc = ink_r1_g05_ld4(sc_n + (unsigned long)scoff);
        }
      }
      __builtin_amdgcn_sched_barrier(0);

      const u32 nbf = bf ^ 1u;
      ink_r1_g05_compute_store8(cur_w, cur_x, cur_sc, w, r16, h, st_col0, st_col1,
                                ink_r1_g05_lds_dot + nbf * INK_R1_G05_DOT_BUF,
                                ink_r1_g05_lds_sw + nbf * INK_R1_G05_SC_BUF,
                                ink_r1_g05_lds_dx + nbf * INK_R1_G05_SC_BUF);
      lbarrier();
      bf = nbf;
    }

    acc = ink_r1_g05_reduce_buf8(ink_r1_g05_lds_dot + bf * INK_R1_G05_DOT_BUF,
                                 ink_r1_g05_lds_sw + bf * INK_R1_G05_SC_BUF,
                                 ink_r1_g05_lds_dx + bf * INK_R1_G05_SC_BUF,
                                 t_out, r_out, r_sw, acc);
  }

  /* Optional 4-block tail when nb % 8 == 4 (zero cost when nb % 8 == 0). */
  if ((nb & 4u) != 0u) {
    const u32 tbf = bf ^ 1u;
    LDSP float *dot_b = ink_r1_g05_lds_dot + tbf * INK_R1_G05_DOT_BUF;
    LDSP float *sw_b = ink_r1_g05_lds_sw + tbf * INK_R1_G05_SC_BUF;
    LDSP float *dx_b = ink_r1_g05_lds_dx + tbf * INK_R1_G05_SC_BUF;
    if (w < 4u) {
      const u32x4 tw = ink_r1_g05_ld4(qw_tile + (unsigned long)nb_main * 32ul + (unsigned long)woff);
      const u32x4 tx = ink_r1_g05_ld4(xq_tile + (unsigned long)nb_main * 32ul + (unsigned long)xoff);
      const v8i zm = (v8i)(0x4B400000);
      const v2i a0 = {(int)tw.x, (int)tw.y}, a1 = {(int)tw.z, (int)tw.w};
      const v2i b0 = {(int)tx.x, (int)tx.y}, b1 = {(int)tx.z, (int)tx.w};
      v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0, zm, 0);
      c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1, c, 0);
      const f8 df = __builtin_bit_cast(f8, c) - 12582912.0f;
      LDSP float *dp = dot_b + (w << 8u) + (r16 << 4u);
      *(LDSP f32x4 *)(dp + st_col0) = (f32x4){df[0], df[1], df[2], df[3]};
      *(LDSP f32x4 *)(dp + st_col1) = (f32x4){df[4], df[5], df[6], df[7]};
      if (w == 0u && h == 0u) {
        const ink_r1_g05_u32x2 sv = ink_r1_g05_ld2(sc_tile + (unsigned long)nb_main * 2ul + (unsigned long)scoff);
        const u32 s0 = sv.x, s1 = sv.y;
        const ink_r1_g05_f16x2 p0 = __builtin_bit_cast(ink_r1_g05_f16x2, s0);
        const ink_r1_g05_f16x2 p1 = __builtin_bit_cast(ink_r1_g05_f16x2, s1);
        *(LDSP f32x4 *)(sw_b + r16 * INK_R1_G05_SC_STRIDE) = (f32x4){(float)p0.x, (float)p0.y, (float)p1.x, (float)p1.y};
      } else if (w == 1u && h == 0u) {
        const u32x4 xv = ink_r1_g05_ld4(sc_tile + (unsigned long)nb_main * 4ul + (unsigned long)scoff);
        const u32 u0 = xv.x, u1 = xv.y, u2 = xv.z, u3 = xv.w;
        *(LDSP f32x4 *)(dx_b + r16 * INK_R1_G05_SC_STRIDE) = (f32x4){
            __builtin_bit_cast(float, u0), __builtin_bit_cast(float, u1),
            __builtin_bit_cast(float, u2), __builtin_bit_cast(float, u3)};
      }
    }
    lbarrier();
    const LDSP float *dp = dot_b + (t_out << 4u) + r_sw;
    const f32x4 sw0 = *(const LDSP f32x4 *)(sw_b + r_out * INK_R1_G05_SC_STRIDE);
    const f32x4 dx0 = *(const LDSP f32x4 *)(dx_b + t_out * INK_R1_G05_SC_STRIDE);
    const f32x4 sc0 = sw0 * dx0;
    acc = __builtin_fmaf(sc0.x, dp[0u << 8u], acc);
    acc = __builtin_fmaf(sc0.y, dp[1u << 8u], acc);
    acc = __builtin_fmaf(sc0.z, dp[2u << 8u], acc);
    acc = __builtin_fmaf(sc0.w, dp[3u << 8u], acc);
  }

  const u32 r = rbase + r_out;
  const u32 t = tbase + t_out;
  if (r < rows && t < T) {
    const G float *rp = res ? res + (unsigned long)t * (unsigned long)rs : (const G float *)0;
    y[(unsigned long)t * (unsigned long)ys + r] = epilogue(acc, r, bias, rp);
  }
}

/* ============================================================================
 * Kernel 2: ie_inkling_gemm_q8_nb2_r1
 * Dedicated nb==2 (cols==64) Q8_0 WMMA GEMM for loop_wfg (4608x64).
 * Uses 128x64 WG tile (16 rows/wave x 64 tokens/wave), 1-shot LDS staging of
 * the 4 KiB activation tile, direct-VGPR weight loads, and SWAPPED WMMA
 * (A = X, B = W) so output D[v] has 16 contiguous rows across lanes 0..15 for
 * a fixed token -> 100% coalesced y stores (eliminates ie_gemm_q8's 16x write
 * amplification) with only 1 lbarrier() total and 0-bit-diff vs ie_gemm_q8.
 * ========================================================================== */

KERNEL ie_inkling_gemm_q8_nb2_r1(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                 u32 rows, u32 nb, const G float *bias, const G float *res,
                                 u32 T, u32 xs, u32 ys, u32 rs) {
  static LDS u8 ink_r1_g05_nb2_x[INK_R1_G05_NB2_T * INK_R1_G05_NB2_XLS] __attribute__((aligned(16)));
  static LDS float ink_r1_g05_nb2_dx[2u][INK_R1_G05_NB2_T] __attribute__((aligned(32)));

  if (T == 0u || rows == 0u || nb != 2u || !qw || !qs || !xq || !y) return;
  const u32 ntt = (T + INK_R1_G05_NB2_T - 1u) / INK_R1_G05_NB2_T;
  const u32 tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 rbase = rt * INK_R1_G05_NB2_R, tbase = tt * INK_R1_G05_NB2_T;
  if (rbase >= rows || tbase >= T) return;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;

  /* Stage 64 tokens x 64 B of X into LDS (1x global_load_b128 per thread). */
  const u32 tk_ld = t0 >> 2u;
  const u32 ch_ld = t0 & 3u;
  const u32 gt_ld = tbase + tk_ld < T ? tbase + tk_ld : T - 1u;
  const u32x4 xv_ld = ink_r1_g05_ld4(xq + (unsigned long)gt_ld * (unsigned long)xs + (unsigned long)(ch_ld << 4u));

  /* Threads 0..63 also load the 2 f32 activation scales per token. */
  float dx_ld0 = 0.0f, dx_ld1 = 0.0f;
  if (t0 < INK_R1_G05_NB2_T) {
    const u32 gt_sc = tbase + t0 < T ? tbase + t0 : T - 1u;
    const ink_r1_g05_u32x2 dv = ink_r1_g05_ld2(xq + (unsigned long)gt_sc * (unsigned long)xs + 64ul);
    const u32 d0 = dv.x, d1 = dv.y;
    dx_ld0 = __builtin_bit_cast(float, d0);
    dx_ld1 = __builtin_bit_cast(float, d1);
  }

  /* Load this wave's 16 weight rows directly into VGPRs (in flight with X). */
  const u32 r = rbase + (w << 4u) + r16;
  const u32 gr = r < rows ? r : rows - 1u;
  const G u8 *wp = (const G u8 *)qw + (unsigned long)gr * 64ul + (unsigned long)(h << 4u);
  const u32x4 w0 = ink_r1_g05_ld4(wp);
  const u32x4 w1 = ink_r1_g05_ld4(wp + 32ul);
  const u32 sw_raw = *(const G u32 *)(qs + (unsigned long)gr * 2ul);
  const ink_r1_g05_f16x2 sw_p = __builtin_bit_cast(ink_r1_g05_f16x2, sw_raw);
  const float sw0 = (float)sw_p.x;
  const float sw1 = (float)sw_p.y;

  *(LDSP u32x4 *)(ink_r1_g05_nb2_x + tk_ld * INK_R1_G05_NB2_XLS + (ch_ld << 4u)) = xv_ld;
  if (t0 < INK_R1_G05_NB2_T) {
    ink_r1_g05_nb2_dx[0u][t0] = dx_ld0;
    ink_r1_g05_nb2_dx[1u][t0] = dx_ld1;
  }
  lbarrier();

  const v8i zm = (v8i)(0x4B400000);
  const v2i b0_0 = {(int)w0.x, (int)w0.y}, b0_1 = {(int)w0.z, (int)w0.w};
  const v2i b1_0 = {(int)w1.x, (int)w1.y}, b1_1 = {(int)w1.z, (int)w1.w};
  const float bval = (bias && r < rows) ? bias[r] : 0.0f;

#pragma unroll
  for (u32 j = 0u; j < 4u; j++) {
    const LDSP u8 *xp = ink_r1_g05_nb2_x + ((j << 4u) + r16) * INK_R1_G05_NB2_XLS + (h << 4u);
    const u32x4 x0 = *(const LDSP u32x4 *)(xp);
    const u32x4 x1 = *(const LDSP u32x4 *)(xp + 32u);
    const f8 dx0 = *(const LDSP f8 *)(ink_r1_g05_nb2_dx[0u] + (j << 4u) + (h << 3u));
    const f8 dx1 = *(const LDSP f8 *)(ink_r1_g05_nb2_dx[1u] + (j << 4u) + (h << 3u));

    const v2i a0_0 = {(int)x0.x, (int)x0.y}, a0_1 = {(int)x0.z, (int)x0.w};
    v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_0, 1, b0_0, zm, 0);
    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_1, 1, b0_1, c0, 0);
    const f8 d0 = __builtin_bit_cast(f8, c0) - 12582912.0f;

    const v2i a1_0 = {(int)x1.x, (int)x1.y}, a1_1 = {(int)x1.z, (int)x1.w};
    v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_0, 1, b1_0, zm, 0);
    c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_1, 1, b1_1, c1, 0);
    const f8 d1 = __builtin_bit_cast(f8, c1) - 12582912.0f;

    const f8 sc0 = (f8)(sw0) * dx0;
    const f8 sc1 = (f8)(sw1) * dx1;
    const u32 t_base_v = tbase + (j << 4u) + (h << 3u);
    if (r < rows) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 t = t_base_v + v;
        if (t < T) {
          float acc = __builtin_fmaf(sc0[v], d0[v], 0.0f);
          acc = __builtin_fmaf(sc1[v], d1[v], acc);
          if (bias) acc = acc + bval;
          if (res) acc = res[(unsigned long)t * (unsigned long)rs + r] + acc;
          y[(unsigned long)t * (unsigned long)ys + r] = acc;
        }
      }
    }
  }
}
#endif /* !IE_MEGA */

/* ============================================================================
 * Kernel 3: ie_inkling_gemv_q8_nb2_r1
 * Dedicated nb==2 (cols==64) decode Q8_0 GEMV for loop_wfg (4608x64).
 * 8 rows/wave, 4 lanes/row (100% lane utilization, 512 B contiguous coalesced
 * wave weight reads, 72 WGs across 64 CUs, 2 DPP shl_i steps, 0 LDS, 0-bit-diff
 * vs ie_gemv_q8q8).
 * ========================================================================== */

KERNEL ie_inkling_gemv_q8_nb2_r1(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                 u32 rows, u32 nb, const G float *bias, const G float *res,
                                 const G float *nw, G float *ny, G u8 *nq, float eps, G u32 *count) {
#pragma clang fp contract(off)
  const u32 l = lane();
  const u32 sub = l & 3u;
  const u32 blk = sub >> 1u;
  const u32 r = wgid() * 64u + (tid() >> 5u) * 8u + (l >> 2u);
  if (r < rows && nb == 2u) {
    const u32x4 wv = ink_r1_g05_ld4((const G u8 *)qw + (unsigned long)r * 64ul + (unsigned long)(sub << 4u));
    const u32x4 av = ink_r1_g05_ld4(xq + (unsigned long)(sub << 4u));
    const u32 d_half = ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(0u, wv.x, av.x), wv.y, av.y), wv.z, av.z), wv.w, av.w);
    const int dot = (int)d_half + shl_i((int)d_half, 1);
    const float sc = (float)qs[(unsigned long)r * 2ul + blk] * ((const G float *)(xq + 64u))[blk];
    const float term = sc * (float)dot;
    const float sum = term + I2F(shl_i(F2I(term), 2));
    if (sub == 0u) y[r] = epilogue(sum, r, bias, res);
  }
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

/* ============================================================================
 * Kernel 4: ie_inkling_gemv_q8_small_r1
 * High-occupancy decode Q8_0 GEMV for small-row shapes (wsink 24x2304,
 * k/v_conv_proj 32x2304, loop_down 64x4608): 1 WG (8 waves) per output row
 * -> 8x higher CU occupancy (64 WGs on 64 CUs for loop_down vs 8 WGs before).
 * Preserves exact ie_gemv_q8q8 per-lane block accumulation and wave_tree_f
 * reduction order (0-bit-diff vs ie_gemv_q8q8).
 * ========================================================================== */

KERNEL ie_inkling_gemv_q8_small_r1(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                   u32 rows, u32 nb, const G float *bias, const G float *res,
                                   const G float *nw, G float *ny, G u8 *nq, float eps, G u32 *count) {
#pragma clang fp contract(off)
#ifdef IE_MEGA
#define ink_r1_g05_gv_lds (mk_ink_lds.q8_small.gv_lds)
#else
  static LDS float ink_r1_g05_gv_lds[8u][32u] __attribute__((aligned(16)));
#endif
  const u32 r = wgid();
  const u32 t0 = tid(), w = t0 >> 5u, l = lane();
  if (r < rows) {
    const G u32 *aw = (const G u32 *)xq;
    const G float *da = (const G float *)(xq + (unsigned long)nb * 32ul);
    float acc = 0.0f;
    const u32 nchunks = (nb + 255u) >> 8u;
    for (u32 ch = 0u; ch < nchunks; ch++) {
      const u32 base = ch << 8u;
      const u32 b = base + t0;
      float term = 0.0f;
      if (b < nb) {
        const G u32 *wp = qw + ((unsigned long)r * (unsigned long)nb + (unsigned long)b) * 8ul;
        const u32x4 w0 = *(const G u32x4 *)wp;
        const u32x4 w1 = *(const G u32x4 *)(wp + 4u);
        const u32x4 a0 = *(const G u32x4 *)(aw + (unsigned long)b * 8ul);
        const u32x4 a1 = *(const G u32x4 *)(aw + (unsigned long)b * 8ul + 4ul);
        const float sc = (float)qs[(unsigned long)r * (unsigned long)nb + (unsigned long)b] * da[b];
        const u32 wq[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const Mem qm = {wq}, am = {av};
        const int dot = (int)ie_q8q8_block(qm, 0u, am, 0u);
        term = sc * (float)dot;
      }
      ink_r1_g05_gv_lds[w][l] = term;
      lbarrier();
      if (w == 0u) {
        const u32 rem = nb - base;
        const u32 nw_act = rem >= 256u ? 8u : ((rem + 31u) >> 5u);
        for (u32 k = 0u; k < nw_act; k++) {
          if (base + (k << 5u) + l < nb) acc = acc + ink_r1_g05_gv_lds[k][l];
        }
      }
      if (ch + 1u < nchunks) lbarrier();
    }
    if (w == 0u) {
      acc = wave_tree_f(acc);
      if (l == 0u) y[r] = epilogue(acc, r, bias, res);
    }
  }
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

#ifdef IE_MEGA
#undef ink_r1_g05_gv_lds
#endif
#undef INK_R1_G05_NB2_XLS
#undef INK_R1_G05_NB2_T
#undef INK_R1_G05_NB2_R
#undef INK_R1_G05_SC_BUF
#undef INK_R1_G05_DOT_BUF
#undef INK_R1_G05_SC_STRIDE
#undef INK_R1_G05_KSTEP
#undef INK_R1_G05_TILE
