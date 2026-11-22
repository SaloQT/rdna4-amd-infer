/* ============================================================================
 * Inkling-2 Round 2 Group 10: Medium Q8_0 Decode/Prefill Kernels
 * Target: RX 9070 XT (gfx1201, wave32, NT=256)
 *
 * Exported kernels:
 *
 * 1. KERNEL ie_inkling_gemv_q8_med_r2(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
 *                                     u32 rows, u32 nb, const G float *bias, const G float *res,
 *                                     const G float *nw, G float *ny, G u8 *nq, float eps, G u32 *count)
 *    - Bit:    20 (exact ie_gemv_q8q8 signature)
 *    - Grid:   ((rows + 31u) / 32u, 1, 1)  [4 rows/wave, 8 lanes/row, 32 rows/WG]
 *    - Guard:  T == 1u && rows >= 128u && rows <= 8192u && nb >= 32u && (nb & 7u) == 0u
 *              (leaves output.weight 201024x2304 on ie_gemv_q8q8; safe for any nb >= 8)
 *    - Target: decode 3072x2304 (86x), 1024x2304 (76x), 384x2304 (48x), 2304x3072 (48x), 5120x2304 (10x)
 *              -> 100% lane utilization on nb=72 (9 full iters) and nb=96 (12 full iters)
 *    - Parity: bitwise-exact (0-bit-diff) vs ie_gemv_q8q8 (4-acc exact wave_tree_f, fp contract off)
 *
 * 2. KERNEL ie_inkling_gemv_q8_multi2_r2(const G u32 *qw0, const G f16 *qs0, const G float *bias0,
 *                                        G float *y0, u32 rows0,
 *                                        const G u32 *qw1, const G f16 *qs1, const G float *bias1,
 *                                        G float *y1, u32 rows1,
 *                                        const G u8 *xq, u32 nb)
 *    - Bit:    21 (decode T=1 fused dual-output Q8_0 GEMV sharing xq)
 *    - Grid:   (((rows0 + 31u) / 32u) + ((rows1 + 31u) / 32u), 1, 1)
 *    - Guard:  T == 1u && rows0 > 0u && rows1 > 0u && nb >= 8u && (nb & 7u) == 0u
 *              (wsink [24,2304] + wagt [3072,2304] sharing qq -> 1+96 = 97 WGs, -48 launches/tok)
 *    - Parity: bitwise-exact (0-bit-diff) vs two separate ie_gemv_q8q8 launches
 *
 * 3. KERNEL ie_inkling_gemm_q8_med_r2(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
 *                                     u32 rows, u32 nb, const G float *bias, const G float *res,
 *                                     u32 T, u32 xs, u32 ys, u32 rs)
 *    - Bit:    22 (exact ie_gemm_q8r signature)
 *    - Grid:   (((rows + 127u) / 128u) * ((T + 63u) / 64u), 1, 1)  [128x64 tile, WI=1, WJ=4, KB=4]
 *    - Guard:  T >= 17u && rows >= 512u && nb >= 4u && (nb & 3u) == 0u
 *              (also safe for any rows > 0; leaves rows <= 384 on ie_inkling_gemm_q8_small_r1)
 *    - Target: prefill wk/wv 1024x2304 (64 WGs vs 32), wq/wagt 3072x2304 (192 WGs = 3x64),
 *              wo 2304x3072 (144 WGs), wqkv 5120x2304 (320 WGs = 5x64)
 *              -> 64-bit f16x4 scale loads, XOR-swizzled 0-bank-conflict LDS (22 KiB), <=160 VGPR (2 WG/CU)
 *    - Parity: bitwise-exact (0-bit-diff) vs ie_gemm_q8r
 * ========================================================================== */

#define INK_R2_G10_M_WI 1u
#define INK_R2_G10_M_R  128u /* 8 waves * 16 rows * WI=1 */
#define INK_R2_G10_M_T  64u  /* 4 * 16 tokens per WG */
#define INK_R2_G10_M_WJ 4u   /* M_T / 16 */
#define INK_R2_G10_M_KB 4u   /* 4 Q8_0 blocks (128 bytes) per K chunk */

typedef _Float16 ink_r2_g10_f16x2 __attribute__((ext_vector_type(2)));
typedef unsigned int ink_r2_g10_u32x2 __attribute__((ext_vector_type(2)));

struct __attribute__((packed, aligned(4))) ink_r2_g10_u32x4p {
  u32x4 v;
};

struct __attribute__((packed, aligned(4))) ink_r2_g10_u32x2p {
  ink_r2_g10_u32x2 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r2_g10_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r2_g10_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) ink_r2_g10_u32x2 ink_r2_g10_ld2(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r2_g10_u32x2p *)p)->v;
}

static inline __attribute__((always_inline)) int ink_r2_g10_dot8(u32x4 w0, u32x4 w1, u32x4 a0, u32x4 a1) {
  u32 d = ie_dot4_ss(0u, w0.x, a0.x);
  d = ie_dot4_ss(d, w0.y, a0.y);
  d = ie_dot4_ss(d, w0.z, a0.z);
  d = ie_dot4_ss(d, w0.w, a0.w);
  d = ie_dot4_ss(d, w1.x, a1.x);
  d = ie_dot4_ss(d, w1.y, a1.y);
  d = ie_dot4_ss(d, w1.z, a1.z);
  d = ie_dot4_ss(d, w1.w, a1.w);
  return (int)d;
}

/* ============================================================================
 * Shared 4-rows/wave (8 lanes/row, 32 rows/WG) Q8_0 GEMV core
 * Lane sub (0..7) accumulates 4 virtual ie_gemv_q8q8 lanes:
 *   a0: sub + 32*t,  a1: sub + 8 + 32*t,  a2: sub + 16 + 32*t,  a3: sub + 24 + 32*t
 * Then reduces in exact wave_tree_f order:
 *   step 1 (+16): u0 = a0 + a2, u1 = a1 + a3
 *   step 2 (+8):  s  = u0 + u1
 *   steps 3..5 (+4, +2, +1): DPP row_shl within each 8-lane row group.
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r2_g10_gemv_q8_4rw(
    u32 g, const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
    u32 rows, u32 nb, const G float *bias, const G float *res) {
#pragma clang fp contract(off)
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 r = (g << 5u) + (w << 2u) + (l >> 3u);
  if (r < rows) {
    const G u32 *row_qw = qw + (unsigned long)r * (unsigned long)nb * 8ul;
    const G f16 *row_qs = qs + (unsigned long)r * (unsigned long)nb;
    const G u32 *aw = (const G u32 *)xq;
    const G float *da = (const G float *)(xq + (unsigned long)nb * 32ul);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const u32 nb_main = nb & ~31u;

    for (u32 k0 = 0u; k0 < nb_main; k0 += 32u) {
      const u32 b0 = k0 + sub;
      const u32 b1 = b0 + 8u;
      const G u32 *wp0 = row_qw + (unsigned long)b0 * 8ul;
      const G u32 *wp1 = row_qw + (unsigned long)b1 * 8ul;
      const G u32 *ap0 = aw + (unsigned long)b0 * 8ul;
      const G u32 *ap1 = aw + (unsigned long)b1 * 8ul;
      const u32x4 w0_0 = ink_r2_g10_ld4(wp0);
      const u32x4 w0_1 = ink_r2_g10_ld4(wp0 + 4u);
      const u32x4 x0_0 = ink_r2_g10_ld4(ap0);
      const u32x4 x0_1 = ink_r2_g10_ld4(ap0 + 4u);
      const float sc0 = (float)row_qs[b0] * da[b0];
      const u32x4 w1_0 = ink_r2_g10_ld4(wp1);
      const u32x4 w1_1 = ink_r2_g10_ld4(wp1 + 4u);
      const u32x4 x1_0 = ink_r2_g10_ld4(ap1);
      const u32x4 x1_1 = ink_r2_g10_ld4(ap1 + 4u);
      const float sc1 = (float)row_qs[b1] * da[b1];

      const int d0 = ink_r2_g10_dot8(w0_0, w0_1, x0_0, x0_1);
      const float t0 = sc0 * (float)d0;
      a0 = a0 + t0;
      const int d1 = ink_r2_g10_dot8(w1_0, w1_1, x1_0, x1_1);
      const float t1 = sc1 * (float)d1;
      a1 = a1 + t1;

      const u32 b2 = b0 + 16u;
      const u32 b3 = b0 + 24u;
      const G u32 *wp2 = row_qw + (unsigned long)b2 * 8ul;
      const G u32 *wp3 = row_qw + (unsigned long)b3 * 8ul;
      const G u32 *ap2 = aw + (unsigned long)b2 * 8ul;
      const G u32 *ap3 = aw + (unsigned long)b3 * 8ul;
      const u32x4 w2_0 = ink_r2_g10_ld4(wp2);
      const u32x4 w2_1 = ink_r2_g10_ld4(wp2 + 4u);
      const u32x4 x2_0 = ink_r2_g10_ld4(ap2);
      const u32x4 x2_1 = ink_r2_g10_ld4(ap2 + 4u);
      const float sc2 = (float)row_qs[b2] * da[b2];
      const u32x4 w3_0 = ink_r2_g10_ld4(wp3);
      const u32x4 w3_1 = ink_r2_g10_ld4(wp3 + 4u);
      const u32x4 x3_0 = ink_r2_g10_ld4(ap3);
      const u32x4 x3_1 = ink_r2_g10_ld4(ap3 + 4u);
      const float sc3 = (float)row_qs[b3] * da[b3];

      const int d2 = ink_r2_g10_dot8(w2_0, w2_1, x2_0, x2_1);
      const float t2 = sc2 * (float)d2;
      a2 = a2 + t2;
      const int d3 = ink_r2_g10_dot8(w3_0, w3_1, x3_0, x3_1);
      const float t3 = sc3 * (float)d3;
      a3 = a3 + t3;
    }

    const u32 rem = nb - nb_main;
    if (rem > 0u) {
      const u32 b0 = nb_main + sub;
      if (b0 < nb) {
        const G u32 *wp0 = row_qw + (unsigned long)b0 * 8ul;
        const G u32 *ap0 = aw + (unsigned long)b0 * 8ul;
        const u32x4 w0_0 = ink_r2_g10_ld4(wp0);
        const u32x4 w0_1 = ink_r2_g10_ld4(wp0 + 4u);
        const u32x4 x0_0 = ink_r2_g10_ld4(ap0);
        const u32x4 x0_1 = ink_r2_g10_ld4(ap0 + 4u);
        const float sc0 = (float)row_qs[b0] * da[b0];
        const int d0 = ink_r2_g10_dot8(w0_0, w0_1, x0_0, x0_1);
        a0 = a0 + sc0 * (float)d0;
      }
      if (rem > 8u) {
        const u32 b1 = nb_main + 8u + sub;
        if (b1 < nb) {
          const G u32 *wp1 = row_qw + (unsigned long)b1 * 8ul;
          const G u32 *ap1 = aw + (unsigned long)b1 * 8ul;
          const u32x4 w1_0 = ink_r2_g10_ld4(wp1);
          const u32x4 w1_1 = ink_r2_g10_ld4(wp1 + 4u);
          const u32x4 x1_0 = ink_r2_g10_ld4(ap1);
          const u32x4 x1_1 = ink_r2_g10_ld4(ap1 + 4u);
          const float sc1 = (float)row_qs[b1] * da[b1];
          const int d1 = ink_r2_g10_dot8(w1_0, w1_1, x1_0, x1_1);
          a1 = a1 + sc1 * (float)d1;
        }
      }
      if (rem > 16u) {
        const u32 b2 = nb_main + 16u + sub;
        if (b2 < nb) {
          const G u32 *wp2 = row_qw + (unsigned long)b2 * 8ul;
          const G u32 *ap2 = aw + (unsigned long)b2 * 8ul;
          const u32x4 w2_0 = ink_r2_g10_ld4(wp2);
          const u32x4 w2_1 = ink_r2_g10_ld4(wp2 + 4u);
          const u32x4 x2_0 = ink_r2_g10_ld4(ap2);
          const u32x4 x2_1 = ink_r2_g10_ld4(ap2 + 4u);
          const float sc2 = (float)row_qs[b2] * da[b2];
          const int d2 = ink_r2_g10_dot8(w2_0, w2_1, x2_0, x2_1);
          a2 = a2 + sc2 * (float)d2;
        }
      }
    }

    const float u0 = a0 + a2;
    const float u1 = a1 + a3;
    float s = u0 + u1;
    s = s + I2F(shl_i(F2I(s), 4));
    s = s + I2F(shl_i(F2I(s), 2));
    s = s + I2F(shl_i(F2I(s), 1));
    if (sub == 0u) y[r] = epilogue(s, r, bias, res);
  }
}

/* ============================================================================
 * Kernel 1: ie_inkling_gemv_q8_med_r2 (bit 20)
 * ========================================================================== */

KERNEL ie_inkling_gemv_q8_med_r2(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                 u32 rows, u32 nb, const G float *bias, const G float *res,
                                 const G float *nw, G float *ny, G u8 *nq, float eps, G u32 *count) {
  ink_r2_g10_gemv_q8_4rw(wgid(), qw, qs, xq, y, rows, nb, bias, res);
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

/* ============================================================================
 * Kernel 2: ie_inkling_gemv_q8_multi2_r2 (bit 21)
 * Fused dual-output Q8_0 GEMV sharing xq (wsink [24,2304] + wagt [3072,2304]).
 * WG-uniform dispatch: WGs [0, wg0) compute y0, WGs [wg0, wg0+wg1) compute y1.
 * ========================================================================== */

KERNEL ie_inkling_gemv_q8_multi2_r2(const G u32 *qw0, const G f16 *qs0, const G float *bias0,
                                    G float *y0, u32 rows0,
                                    const G u32 *qw1, const G f16 *qs1, const G float *bias1,
                                    G float *y1, u32 rows1,
                                    const G u8 *xq, u32 nb) {
  const u32 wg0 = (rows0 + 31u) >> 5u;
  const u32 g = wgid();
  if (g < wg0) {
    ink_r2_g10_gemv_q8_4rw(g, qw0, qs0, xq, y0, rows0, nb, bias0, (const G float *)0);
  } else {
    ink_r2_g10_gemv_q8_4rw(g - wg0, qw1, qs1, xq, y1, rows1, nb, bias1, (const G float *)0);
  }
}

/* ============================================================================
 * Kernel 3: ie_inkling_gemm_q8_med_r2 (bit 22)
 * 128x64 (WI=1, WJ=4, KB=4) direct-VGPR W + XOR-swizzled bank-conflict-free
 * LDS X/scales (22 KiB total -> 2 WGs/CU), 64-bit f16x4 weight scale loads,
 * 128-bit f32x4 activation scale loads balanced across waves, <=160 VGPR.
 * ========================================================================== */

KERNEL ie_inkling_gemm_q8_med_r2(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                 u32 rows, u32 nb, const G float *bias, const G float *res,
                                 u32 T, u32 xs, u32 ys, u32 rs) {
  static LDS u8 ink_r2_g10_x[2u][INK_R2_G10_M_T * (INK_R2_G10_M_KB * 32u)] __attribute__((aligned(16)));
  static LDS float ink_r2_g10_sw[2u][INK_R2_G10_M_KB * INK_R2_G10_M_R] __attribute__((aligned(32)));
  static LDS float ink_r2_g10_sx[2u][INK_R2_G10_M_KB * INK_R2_G10_M_T] __attribute__((aligned(16)));
  _Static_assert(sizeof(f16) == 2u, "ink_r2_g10 f16 must be 2 bytes");
  _Static_assert(sizeof(ink_r2_g10_x) + sizeof(ink_r2_g10_sw) + sizeof(ink_r2_g10_sx) == 22528u,
                 "ink_r2_g10 med GEMM LDS must be 22528 B");

  if (T == 0u || rows == 0u || nb < INK_R2_G10_M_KB || (nb & (INK_R2_G10_M_KB - 1u)) != 0u ||
      !qw || !qs || !xq || !y) return;

  const u32 ntt = (T + INK_R2_G10_M_T - 1u) / INK_R2_G10_M_T;
  const u32 tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 rbase = rt * INK_R2_G10_M_R, tbase = tt * INK_R2_G10_M_T;
  if (rbase >= rows || tbase >= T) return;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;
  const u32 wr = w << 4u;

  f8 acc[INK_R2_G10_M_WJ];
#pragma unroll
  for (u32 j = 0u; j < INK_R2_G10_M_WJ; j++) acc[j] = (f8)(0.0f);

  /* Direct-VGPR weight pointer for this wave's 16 rows (WI=1). */
  const u32 gr = rbase + wr + r16 < rows ? rbase + wr + r16 : rows - 1u;
  const G u8 *wp = (const G u8 *)qw + (unsigned long)gr * (unsigned long)nb * 32ul + (unsigned long)(h << 4u);

  /* Cooperative X staging pointers: 256 threads * 2 * 16 B = 64 tokens * 128 B. */
  const G u8 *xp[2u];
  u32 x_st_off[2u];
#pragma unroll
  for (u32 i = 0u; i < 2u; i++) {
    const u32 p = t0 + NT * i;
    const u32 tk = p >> 3u;
    const u32 col = p & 7u;
    const u32 gt = tbase + tk < T ? tbase + tk : T - 1u;
    xp[i] = xq + (unsigned long)gt * (unsigned long)xs + (unsigned long)(col << 4u);
    x_st_off[i] = (tk << 7u) + ((col ^ (tk & 7u)) << 4u);
  }

  /* Weight scales (waves 0..3: 128 threads * f16x4) and activation scales (waves 4..5: 64 threads * f32x4). */
  const u32 sr = rbase + (t0 & 127u) < rows ? rbase + (t0 & 127u) : rows - 1u;
  const G f16 *swp = qs + (unsigned long)sr * (unsigned long)nb;
  const u32 sxi = t0 & 63u;
  const u32 gt_sx = tbase + sxi < T ? tbase + sxi : T - 1u;
  const G u8 *sxp = xq + (unsigned long)gt_sx * (unsigned long)xs + (unsigned long)nb * 32ul;

  u32x4 wa[INK_R2_G10_M_KB], wn[INK_R2_G10_M_KB];
  u32x4 rx[2u];
  u32x4 rsc = (u32x4)(0u);

#define INK_R2_G10_FETCH_W(dst_, kb0_)                                              \
  _Pragma("unroll") for (u32 k_ = 0u; k_ < INK_R2_G10_M_KB; k_++)                   \
      dst_[k_] = ink_r2_g10_ld4(wp + (unsigned long)((kb0_) + k_) * 32ul)

#define INK_R2_G10_FETCH_X(kb0_)                                                    \
  do {                                                                              \
    _Pragma("unroll") for (u32 i_ = 0u; i_ < 2u; i_++)                              \
        rx[i_] = ink_r2_g10_ld4(xp[i_] + (unsigned long)(kb0_) * 32ul);             \
    if (t0 < INK_R2_G10_M_R) {                                                      \
      const ink_r2_g10_u32x2 sw2_ = ink_r2_g10_ld2(swp + (unsigned long)(kb0_));    \
      rsc.x = sw2_.x;                                                               \
      rsc.y = sw2_.y;                                                               \
    } else if (t0 < INK_R2_G10_M_R + INK_R2_G10_M_T) {                              \
      rsc = ink_r2_g10_ld4(sxp + (unsigned long)(kb0_) * 4ul);                      \
    }                                                                               \
  } while (0)

#define INK_R2_G10_STORE_X(bf_)                                                     \
  do {                                                                              \
    _Pragma("unroll") for (u32 i_ = 0u; i_ < 2u; i_++)                              \
        *(LDSP u32x4 *)(ink_r2_g10_x[bf_] + x_st_off[i_]) = rx[i_];                 \
    if (t0 < INK_R2_G10_M_R) {                                                      \
      const u32 s0_ = rsc.x, s1_ = rsc.y;                                           \
      const ink_r2_g10_f16x2 p0_ = __builtin_bit_cast(ink_r2_g10_f16x2, s0_);       \
      const ink_r2_g10_f16x2 p1_ = __builtin_bit_cast(ink_r2_g10_f16x2, s1_);       \
      ink_r2_g10_sw[bf_][0u * INK_R2_G10_M_R + t0] = (float)p0_.x;                  \
      ink_r2_g10_sw[bf_][1u * INK_R2_G10_M_R + t0] = (float)p0_.y;                  \
      ink_r2_g10_sw[bf_][2u * INK_R2_G10_M_R + t0] = (float)p1_.x;                  \
      ink_r2_g10_sw[bf_][3u * INK_R2_G10_M_R + t0] = (float)p1_.y;                  \
    } else if (t0 < INK_R2_G10_M_R + INK_R2_G10_M_T) {                              \
      const u32 d0_ = rsc.x, d1_ = rsc.y, d2_ = rsc.z, d3_ = rsc.w;                 \
      ink_r2_g10_sx[bf_][0u * INK_R2_G10_M_T + sxi] = __builtin_bit_cast(float, d0_); \
      ink_r2_g10_sx[bf_][1u * INK_R2_G10_M_T + sxi] = __builtin_bit_cast(float, d1_); \
      ink_r2_g10_sx[bf_][2u * INK_R2_G10_M_T + sxi] = __builtin_bit_cast(float, d2_); \
      ink_r2_g10_sx[bf_][3u * INK_R2_G10_M_T + sxi] = __builtin_bit_cast(float, d3_); \
    }                                                                               \
  } while (0)

  INK_R2_G10_FETCH_W(wa, 0u);
  INK_R2_G10_FETCH_X(0u);
  INK_R2_G10_STORE_X(0u);
  lbarrier();

  const u32 swiz_t = r16 & 7u;
  const v8i zm = (v8i)(0x4B400000);
  u32 bf = 0u;

#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R2_G10_M_KB) {
    const int more = kb0 + INK_R2_G10_M_KB < nb;
    if (more) {
      INK_R2_G10_FETCH_W(wn, kb0 + INK_R2_G10_M_KB);
      INK_R2_G10_FETCH_X(kb0 + INK_R2_G10_M_KB);
    }
    __builtin_amdgcn_sched_barrier(0);

    const LDSP u8 *lx = ink_r2_g10_x[bf];
    const LDSP float *lsw = ink_r2_g10_sw[bf];
    const LDSP float *lsx = ink_r2_g10_sx[bf];
    /* Always == INK_R2_G10_M_KB; the branch keeps k-blocks in separate BBs so
     * LLVM's pre-RA scheduler does not overlap all 4 k-steps and spill VGPRs. */
    const u32 nk = nb - kb0 < INK_R2_G10_M_KB ? nb - kb0 : INK_R2_G10_M_KB;

#pragma unroll
    for (u32 k = 0u; k < INK_R2_G10_M_KB; k++) {
      if (k >= nk) break;
      const u32 col_sw = ((k << 1u) + h) ^ swiz_t;
      const v2i a0 = {(int)wa[k].x, (int)wa[k].y};
      const v2i a1 = {(int)wa[k].z, (int)wa[k].w};
      const f8 sw = *(const LDSP f8 *)(lsw + k * INK_R2_G10_M_R + wr + (h << 3u));

      /* Process 2 token tiles at a time (j=0,1 then j=2,3) to halve live b[]/c[] VGPRs. */
#pragma unroll
      for (u32 jp = 0u; jp < INK_R2_G10_M_WJ; jp += 2u) {
        const u32x4 b0 = *(const LDSP u32x4 *)(lx + (((jp << 4u) + r16) << 7u) + (col_sw << 4u));
        const u32x4 b1 = *(const LDSP u32x4 *)(lx + ((((jp + 1u) << 4u) + r16) << 7u) + (col_sw << 4u));
        const float dx0 = lsx[k * INK_R2_G10_M_T + (jp << 4u) + r16];
        const float dx1 = lsx[k * INK_R2_G10_M_T + ((jp + 1u) << 4u) + r16];
        const v2i b0_0 = {(int)b0.x, (int)b0.y}, b0_1 = {(int)b0.z, (int)b0.w};
        const v2i b1_0 = {(int)b1.x, (int)b1.y}, b1_1 = {(int)b1.z, (int)b1.w};
        v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_0, zm, 0);
        v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b1_0, zm, 0);
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_1, c0, 0);
        c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1_1, c1, 0);
        acc[jp] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
        acc[jp + 1u] += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
      }
      __asm__ volatile("" : "+v"(acc[0]), "+v"(acc[1]), "+v"(acc[2]), "+v"(acc[3]));
      __builtin_amdgcn_sched_barrier(0);
    }

    if (more) INK_R2_G10_STORE_X(bf ^ 1u);
    lbarrier();
    bf ^= 1u;
#pragma unroll
    for (u32 k = 0u; k < INK_R2_G10_M_KB; k++) wa[k] = wn[k];
  }

#pragma unroll
  for (u32 j = 0u; j < INK_R2_G10_M_WJ; j++) {
    const u32 t = tbase + (j << 4u) + r16;
    if (t >= T) continue;
    const G float *rp = res ? res + (unsigned long)t * (unsigned long)rs : (const G float *)0;
    for (u32 v = 0u; v < 8u; v++) {
      const u32 r = rbase + wr + (h << 3u) + v;
      if (r < rows) {
        y[(unsigned long)t * (unsigned long)ys + r] = epilogue(acc[j][v], r, bias, rp);
      }
    }
  }

#undef INK_R2_G10_STORE_X
#undef INK_R2_G10_FETCH_X
#undef INK_R2_G10_FETCH_W
}

#undef INK_R2_G10_M_KB
#undef INK_R2_G10_M_WJ
#undef INK_R2_G10_M_T
#undef INK_R2_G10_M_R
#undef INK_R2_G10_M_WI
