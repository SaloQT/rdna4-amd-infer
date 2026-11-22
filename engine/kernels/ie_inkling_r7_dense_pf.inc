/* ============================================================================
 * Inkling-2 Round 7 (Wave 3, Bits 72-74): High-Occupancy Dense Q8_0 Prefill
 * WMMA GEMMs (128x64, 32x64, 64x64) + 32x64x16 F32 Router GEMM with Optional
 * Fused OP_MOE_ROUTE for RX 9070 XT (gfx1201, wave32, NT=256).
 *
 * Target: 18.3% of 4K prefill (q8_med_r2 + q8_small_r1 + router_r3 + moe_route).
 * All local symbols/LDS/macros prefixed ink_r7_dpf_ / INK_R7_DPF_; macros
 * #undef'd at EOF. Clang 21 scalar __builtin_bit_cast safe, 0/0 spills.
 *
 * Exported Kernels, Grids, Guards, and Parity:
 *
 * 1. KERNEL ie_inkling_gemm_q8_med_r7(
 *        const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
 *        u32 rows, u32 nb, const G float *bias, const G float *res,
 *        u32 T, u32 xs, u32 ys, u32 rs)
 *    - Bit:    72 (ink_mask2 bit 8); exact drop-in for ie_inkling_gemm_q8_med_r2
 *    - Grid:   (((rows + 127u) / 128u) * ((T + 63u) / 64u), 1u, 1u), NT = 256
 *    - Guard:  !mk.rec && T >= 17u && rows >= 512u && nb >= 4u && (nb & 3u) == 0u
 *    - Upgrades over q8_med_r2 (152 VGPR, 22,528 B LDS = 2 WGs/CU):
 *      (a) f16 sw LDS shrinks LDS 22,528 B -> 20,480 B (20.0 KiB <= 21.8 KiB)
 *      (b) Eliminates wn[4] (-16 VGPRs) via in-place wa[0..1]/wa[2..3] reload +
 *          early X/scale staging after k=1 + 32-bit offsets -> <= 128 VGPR
 *          (UNLOCKS 3 WGs/CU = +50% occupancy!)
 *      (c) 2x f32x4 vectorized epilogue stores (4x fewer stores; scalar fallback)
 *    - Parity: 100% bitwise-exact (0-bit-diff) vs ie_gemm_q8r / q8_med_r2!
 *
 * 2a. KERNEL ie_inkling_gemm_q8_small_r7(exact 12-arg ie_gemm_q8r signature)
 *     - Bit:   73 (ink_mask2 bit 9); 32x64 (BM=32, BN=64, KB=4) direct-VGPR W
 *              (8 waves = 2(M) x 4(N), 16x16 per wave -> 1x f8 acc = <= 96 VGPR,
 *               18,944 B LDS = 3 WGs/CU, ZERO LDS dot reduction, 2x f32x4 stores).
 *     - Grid:  (((rows + 31u) / 32u) * ((T + 63u) / 64u), 1u, 1u), NT = 256
 *              (wrel 384x2304 at T=512 -> 96 WGs, 4x less W & 2x less X traffic
 *               than 16x16 small_r1; also covers loop_down 64x4608, wsink, etc.).
 *     - Guard: !mk.rec && T >= 17u && rows > 0u && rows <= 512u && nb >= 4u &&
 *              (nb & 3u) == 0u.
 *     - Parity: 100% bitwise-exact (0-bit-diff) vs ie_gemm_q8r!
 *
 * 2b. KERNEL ie_inkling_gemm_q8_64x64_r7(exact 12-arg ie_gemm_q8r signature)
 *     - Bit:   73 companion (64x64: 8 waves = 4(M) x 2(N), 16x32 per wave,
 *              <= 96 VGPR, 19,456 B LDS = 3 WGs/CU, 2x f32x4 stores).
 *     - Grid:  (((rows + 63u) / 64u) * ((T + 63u) / 64u), 1u, 1u), NT = 256
 *              (wk/wv 1024x2304 at T=512 -> 128 WGs = 2.00 full waves vs 64 WGs
 *               in 128x64; also A/B selectable for wrel 384x2304 / rows 64..1024).
 *     - Parity: 100% bitwise-exact (0-bit-diff) vs ie_gemm_q8r!
 *
 * 3. KERNEL ie_inkling_router_route_pf_r7(
 *        const G float *w, const G float *x, G float *y,
 *        u32 rows, u32 cols, const G float *bias, const G float *res,
 *        u32 xs, u32 ys, u32 rs, u32 T,
 *        u32 K, const G float *ebias, float scale, G u32 *route, u32 ost,
 *        G u32 *count)
 *    - Bit:    74 (ink_mask2 bit 10)
 *    - Grid:   ((rows + 31u) / 32u, (T + 63u) / 64u, 1u), NT = 256
 *    - Resources: <= 64 VGPR (<= 96 VGPR = 4 WGs/CU!), 12,288 B LDS (<= 16 KiB =
 *                 4 WGs/CU!), 0-bank-conflict XOR swizzle (swz = c4 << 3).
 *    - Modes:
 *      (a) Standalone GEMM (route == NULL): pass K=0, ebias=NULL, scale=0,
 *          route=NULL, ost=0, count=NULL. Exact drop-in for router_r3 (xs/ys/rs
 *          in BYTES).
 *      (b) Fused [OP_GEMV_F32(router), OP_MOE_ROUTE] (route != NULL): eliminates
 *          48 ie_moe_route launches/chunk! Last arriving M-WG of each 64-token
 *          tile (tracked via pre-zeroed count[tokid()], e.g. b->at_count, self-
 *          reset to 0) runs 8 waves in parallel calling exact moe_route8() on
 *          L2-hot logits!
 *    - Guards: !mk.rec && T >= 17u && rows > 0u && cols > 0u && (cols & 3u) == 0u &&
 *              (xs & 15u) == 0u && (((uintptr_t)w) & 15u) == 0u &&
 *              (((uintptr_t)x) & 15u) == 0u; for fused route add !bias && !res &&
 *              rows <= 256u && K > 0u && K <= 32u && count != NULL &&
 *              ((T + 63u) / 64u) <= count_cap (384 in b->at_count).
 *    - Parity: 100% bitwise-exact (0-bit-diff sequential k=0..cols-1 FMA order
 *              + exact moe_route8) vs ie_inkling_gemm_f32_router_r3 + ie_moe_route!
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R7_DPF_KB   4u
#define INK_R7_DPF_M_R  128u
#define INK_R7_DPF_M_T  64u
#define INK_R7_DPF_S_R  32u
#define INK_R7_DPF_H_R  64u
#define INK_R7_DPF_R_BM 32u
#define INK_R7_DPF_R_BN 64u
#define INK_R7_DPF_R_BK 16u

typedef unsigned int ink_r7_dpf_u32x2 __attribute__((ext_vector_type(2)));
typedef float ink_r7_dpf_f2 __attribute__((ext_vector_type(2)));

struct __attribute__((packed, aligned(4))) ink_r7_dpf_u32x4p {
  u32x4 v;
};

struct __attribute__((packed, aligned(4))) ink_r7_dpf_u32x2p {
  ink_r7_dpf_u32x2 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r7_dpf_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r7_dpf_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) ink_r7_dpf_u32x2 ink_r7_dpf_ld2(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r7_dpf_u32x2p *)p)->v;
}

static inline __attribute__((always_inline)) void ink_r7_dpf_stage_sw_f16(
    LDSP f16 *lsw, const u32 stride_r, const u32 r_idx, const ink_r7_dpf_u32x2 rsw) {
  const u32 w0 = rsw.x, w1 = rsw.y;
  const u16 s0 = (u16)(w0 & 0xFFFFu);
  const u16 s1 = (u16)(w0 >> 16u);
  const u16 s2 = (u16)(w1 & 0xFFFFu);
  const u16 s3 = (u16)(w1 >> 16u);
  lsw[0u * stride_r + r_idx] = __builtin_bit_cast(f16, s0);
  lsw[1u * stride_r + r_idx] = __builtin_bit_cast(f16, s1);
  lsw[2u * stride_r + r_idx] = __builtin_bit_cast(f16, s2);
  lsw[3u * stride_r + r_idx] = __builtin_bit_cast(f16, s3);
}

static inline __attribute__((always_inline)) void ink_r7_dpf_stage_sx_f32(
    LDSP float *lsx, const u32 stride_t, const u32 t_idx, const u32x4 rsx) {
  const u32 d0 = rsx.x, d1 = rsx.y, d2 = rsx.z, d3 = rsx.w;
  lsx[0u * stride_t + t_idx] = __builtin_bit_cast(float, d0);
  lsx[1u * stride_t + t_idx] = __builtin_bit_cast(float, d1);
  lsx[2u * stride_t + t_idx] = __builtin_bit_cast(float, d2);
  lsx[3u * stride_t + t_idx] = __builtin_bit_cast(float, d3);
}

/* Vectorized 2x f32x4 epilogue store for Q8_0 GEMMs (0-bit-diff vs epilogue()). */
static inline __attribute__((always_inline)) void ink_r7_dpf_store_q8_row8(
    G float *y, const G float *bias, const G float *res,
    const u32 t, const u32 r0, const u32 rows, const u32 ys, const u32 rs,
    const f8 av) {
  const int vec_ok = (r0 + 8u <= rows) && ((ys & 3u) == 0u) && (!res || (rs & 3u) == 0u) &&
                     (!bias || ((((unsigned long)bias) & 15ul) == 0ul));
  G float *yt = y + (unsigned long)t * (unsigned long)ys;
  if (vec_ok) {
    f32x4 lo = {av[0], av[1], av[2], av[3]};
    f32x4 hi = {av[4], av[5], av[6], av[7]};
    if (bias) {
      const G f32x4 *bp4 = (const G f32x4 *)(bias + r0);
      lo += bp4[0];
      hi += bp4[1];
    }
    if (res) {
      const G f32x4 *rp4 = (const G f32x4 *)(res + (unsigned long)t * (unsigned long)rs + r0);
      lo += rp4[0];
      hi += rp4[1];
    }
    G f32x4 *yt4 = (G f32x4 *)(yt + r0);
    yt4[0] = lo;
    yt4[1] = hi;
  } else {
    const G float *rp = res ? (res + (unsigned long)t * (unsigned long)rs) : (const G float *)0;
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      const u32 r = r0 + v;
      if (r < rows) yt[r] = epilogue(av[v], r, bias, rp);
    }
  }
}

/* ============================================================================
 * Kernel 1: ie_inkling_gemm_q8_med_r7 (Bit 72, 128x64, <= 128 VGPR, 20 KiB LDS)
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r7_dpf_q8_k64(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 r16, const u32 h, f8 acc[4]) {
  const v8i zm = (v8i)(0x4B400000);
  const u32 col_sw = ((k << 1u) + h) ^ (r16 & 7u);
  const LDSP u8 *lx_k = lx + (r16 << 7u) + (col_sw << 4u);
  const LDSP float *lsx_k = lsx + k * INK_R7_DPF_M_T + r16;
  const v2i a0 = {(int)wk.x, (int)wk.y};
  const v2i a1 = {(int)wk.z, (int)wk.w};

  /* Pair 0 (j = 0, 1) */
  u32x4 b0 = *(const LDSP u32x4 *)(lx_k + 0u * 2048u);
  u32x4 b1 = *(const LDSP u32x4 *)(lx_k + 1u * 2048u);
  float dx0 = lsx_k[0u];
  float dx1 = lsx_k[16u];
  __builtin_amdgcn_sched_barrier(0);

  v2i b0_0 = {(int)b0.x, (int)b0.y}, b0_1 = {(int)b0.z, (int)b0.w};
  v2i b1_0 = {(int)b1.x, (int)b1.y}, b1_1 = {(int)b1.z, (int)b1.w};
  v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_0, zm, 0);
  v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b1_0, zm, 0);
  c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_1, c0, 0);
  c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1_1, c1, 0);
  __builtin_amdgcn_sched_barrier(0);

  const h8 sw_h = *(const LDSP h8 *)(lsw + k * INK_R7_DPF_M_R + wr + (h << 3u));
  const f8 sw = __builtin_convertvector(sw_h, f8);
  acc[0] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  acc[1] += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
  __builtin_amdgcn_sched_barrier(0);

  /* Pair 1 (j = 2, 3) */
  b0 = *(const LDSP u32x4 *)(lx_k + 2u * 2048u);
  b1 = *(const LDSP u32x4 *)(lx_k + 3u * 2048u);
  dx0 = lsx_k[32u];
  dx1 = lsx_k[48u];
  __builtin_amdgcn_sched_barrier(0);

  b0_0 = (v2i){(int)b0.x, (int)b0.y}; b0_1 = (v2i){(int)b0.z, (int)b0.w};
  b1_0 = (v2i){(int)b1.x, (int)b1.y}; b1_1 = (v2i){(int)b1.z, (int)b1.w};
  c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_0, zm, 0);
  c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b1_0, zm, 0);
  c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_1, c0, 0);
  c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1_1, c1, 0);
  __builtin_amdgcn_sched_barrier(0);

  acc[2] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  acc[3] += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
  __builtin_amdgcn_sched_barrier(0);
}

KERNEL ie_inkling_gemm_q8_med_r7(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                 u32 rows, u32 nb, const G float *bias, const G float *res,
                                 u32 T, u32 xs, u32 ys, u32 rs) {
  static LDS struct {
    u8    x[2u][INK_R7_DPF_M_T * (INK_R7_DPF_KB * 32u)] __attribute__((aligned(16)));
    f16   sw[2u][INK_R7_DPF_KB * INK_R7_DPF_M_R] __attribute__((aligned(16)));
    float sx[2u][INK_R7_DPF_KB * INK_R7_DPF_M_T] __attribute__((aligned(16)));
  } ink_r7_dpf_med_lds;
  _Static_assert(sizeof(ink_r7_dpf_med_lds) == 20480u, "ink_r7_dpf_med_lds must be 20,480 B (<= 21.8 KiB = 3 WGs/CU)");

  if (T == 0u || rows == 0u || nb < INK_R7_DPF_KB || (nb & (INK_R7_DPF_KB - 1u)) != 0u || !qw || !qs || !xq || !y)
    return;

  const u32 ntt = (T + INK_R7_DPF_M_T - 1u) / INK_R7_DPF_M_T;
  const u32 wg = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  const u32 tt = wg % ntt, rt = wg / ntt;
  const u32 rbase = rt * INK_R7_DPF_M_R, tbase = tt * INK_R7_DPF_M_T;
  if (rbase >= rows || tbase >= T) return;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;
  const u32 wr = w << 4u;

  f8 acc[4];
#pragma unroll
  for (u32 j = 0u; j < 4u; j++) acc[j] = (f8)(0.0f);

  const G u8 *qw8 = rq_ptr_u((const G u8 *)qw);
  const G u8 *qs8 = rq_ptr_u((const G u8 *)qs);
  const G u8 *xq_s = rq_ptr_u(xq + (unsigned long)nb * 32ul);

  const u32 gr = (rbase + wr + r16 < rows) ? (rbase + wr + r16) : (rows - 1u);
  const u32 w_off = gr * (nb * 32u) + (h << 4u);

  const u32 tk0 = t0 >> 3u, col0 = t0 & 7u;
  const u32 gt0 = (tbase + tk0 < T) ? (tbase + tk0) : (T - 1u);
  const u32 gt1 = (tbase + tk0 + 32u < T) ? (tbase + tk0 + 32u) : (T - 1u);
  const u32 xoff0 = gt0 * xs + (col0 << 4u);
  const u32 xoff1 = gt1 * xs + (col0 << 4u);
  const u32 st_x = (tk0 << 7u) + ((col0 ^ (tk0 & 7u)) << 4u);

  const int is_sw = (w < 4u);
  const int is_sx = (w == 4u || w == 5u);
  const u32 sr = (rbase + (t0 & 127u) < rows) ? (rbase + (t0 & 127u)) : (rows - 1u);
  const u32 sw_off = is_sw ? (sr * (nb * 2u)) : 0u;
  const u32 sxi = t0 & 63u;
  const u32 gt_sx = (tbase + sxi < T) ? (tbase + sxi) : (T - 1u);
  const u32 sx_off = is_sx ? (gt_sx * xs) : 0u;

  /* Single weight buffer wa[4] (no wn[4] -> -16 VGPRs!). */
  u32x4 wa[INK_R7_DPF_KB];
#pragma unroll
  for (u32 k = 0u; k < INK_R7_DPF_KB; k++) wa[k] = ink_r7_dpf_ld4(qw8 + (unsigned long)w_off + (unsigned long)k * 32ul);
  u32x4 rx0 = ink_r7_dpf_ld4(xq + (unsigned long)xoff0);
  u32x4 rx1 = ink_r7_dpf_ld4(xq + (unsigned long)xoff1);
  u32x4 rsc = (u32x4)(0u);
  if (is_sw) {
    const ink_r7_dpf_u32x2 sw2 = ink_r7_dpf_ld2(qs8 + (unsigned long)sw_off);
    rsc.x = sw2.x;
    rsc.y = sw2.y;
  } else if (is_sx) {
    rsc = ink_r7_dpf_ld4(xq_s + (unsigned long)sx_off);
  }

  *(LDSP u32x4 *)(ink_r7_dpf_med_lds.x[0] + st_x) = rx0;
  *(LDSP u32x4 *)(ink_r7_dpf_med_lds.x[0] + st_x + 4096u) = rx1;
  if (is_sw) {
    const ink_r7_dpf_u32x2 sw2 = {rsc.x, rsc.y};
    ink_r7_dpf_stage_sw_f16(ink_r7_dpf_med_lds.sw[0], INK_R7_DPF_M_R, t0, sw2);
  } else if (is_sx) {
    ink_r7_dpf_stage_sx_f32(ink_r7_dpf_med_lds.sx[0], INK_R7_DPF_M_T, sxi, rsc);
  }
  lbarrier();

  u32 bf = 0u;
#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R7_DPF_KB) {
    const u32 kb_n = kb0 + INK_R7_DPF_KB;
    const int more = (kb_n < nb);
    u32 nk = (u32)__builtin_amdgcn_readfirstlane((int)(nb - kb0 < INK_R7_DPF_KB ? nb - kb0 : INK_R7_DPF_KB));
    __asm__ volatile("" : "+s"(nk));

    if (more) {
      const G u8 *xq_n = xq + (unsigned long)kb_n * 32ul;
      rx0 = ink_r7_dpf_ld4(xq_n + (unsigned long)xoff0);
      rx1 = ink_r7_dpf_ld4(xq_n + (unsigned long)xoff1);
      if (is_sw) {
        const ink_r7_dpf_u32x2 sw2 = ink_r7_dpf_ld2(qs8 + (unsigned long)sw_off + (unsigned long)kb_n * 2ul);
        rsc.x = sw2.x;
        rsc.y = sw2.y;
      } else if (is_sx) {
        rsc = ink_r7_dpf_ld4(xq_s + (unsigned long)kb_n * 4ul + (unsigned long)sx_off);
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    const LDSP u8 *lx = ink_r7_dpf_med_lds.x[bf];
    const LDSP f16 *lsw = ink_r7_dpf_med_lds.sw[bf];
    const LDSP float *lsx = ink_r7_dpf_med_lds.sx[bf];

    if (0u < nk) ink_r7_dpf_q8_k64(wa[0], lx, lsw, lsx, 0u, wr, r16, h, acc);
    if (1u < nk) ink_r7_dpf_q8_k64(wa[1], lx, lsw, lsx, 1u, wr, r16, h, acc);

    /* Stage next X/scales into dead lds[bf^1] right after k=1 -> frees 12 VGPRs before wa[0..1] reload! */
    if (more) {
      const u32 nbf = bf ^ 1u;
      *(LDSP u32x4 *)(ink_r7_dpf_med_lds.x[nbf] + st_x) = rx0;
      *(LDSP u32x4 *)(ink_r7_dpf_med_lds.x[nbf] + st_x + 4096u) = rx1;
      if (is_sw) {
        const ink_r7_dpf_u32x2 sw2 = {rsc.x, rsc.y};
        ink_r7_dpf_stage_sw_f16(ink_r7_dpf_med_lds.sw[nbf], INK_R7_DPF_M_R, t0, sw2);
      } else if (is_sx) {
        ink_r7_dpf_stage_sx_f32(ink_r7_dpf_med_lds.sx[nbf], INK_R7_DPF_M_T, sxi, rsc);
      }
      __builtin_amdgcn_sched_barrier(0);
      const G u8 *wp_n = qw8 + (unsigned long)w_off + (unsigned long)kb_n * 32ul;
      wa[0] = ink_r7_dpf_ld4(wp_n + 0ul);
      wa[1] = ink_r7_dpf_ld4(wp_n + 32ul);
    }
    __builtin_amdgcn_sched_barrier(0);

    if (2u < nk) ink_r7_dpf_q8_k64(wa[2], lx, lsw, lsx, 2u, wr, r16, h, acc);
    if (3u < nk) ink_r7_dpf_q8_k64(wa[3], lx, lsw, lsx, 3u, wr, r16, h, acc);
    if (more) {
      const G u8 *wp_n = qw8 + (unsigned long)w_off + (unsigned long)kb_n * 32ul;
      wa[2] = ink_r7_dpf_ld4(wp_n + 64ul);
      wa[3] = ink_r7_dpf_ld4(wp_n + 96ul);
      lbarrier();
      bf ^= 1u;
    }
  }

  const u32 r0 = rbase + wr + (h << 3u);
#pragma unroll
  for (u32 j = 0u; j < 4u; j++) {
    const u32 t = tbase + (j << 4u) + r16;
    if (t < T) ink_r7_dpf_store_q8_row8(y, bias, res, t, r0, rows, ys, rs, acc[j]);
  }
}

/* ============================================================================
 * Kernel 2A: ie_inkling_gemm_q8_small_r7 (Bit 73, 32x64, <= 96 VGPR, 18.5 KiB LDS)
 *   8 waves = 2(M) x 4(N), each wave owns ONE 16x16 output tile (1x f8 acc!).
 *   Zero LDS dot-reduction, 4x less W traffic & 2x less X traffic than small_r1.
 *   wrel (384x2304, T=512) -> 96 WGs; loop_down (64x4608), wsink (24x2304).
 * ========================================================================== */

KERNEL ie_inkling_gemm_q8_small_r7(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                   u32 rows, u32 nb, const G float *bias, const G float *res,
                                   u32 T, u32 xs, u32 ys, u32 rs) {
  static LDS struct {
    u8    x[2u][INK_R7_DPF_M_T * (INK_R7_DPF_KB * 32u)] __attribute__((aligned(16)));
    f16   sw[2u][INK_R7_DPF_KB * INK_R7_DPF_S_R] __attribute__((aligned(16)));
    float sx[2u][INK_R7_DPF_KB * INK_R7_DPF_M_T] __attribute__((aligned(16)));
  } ink_r7_dpf_sm_lds;
  _Static_assert(sizeof(ink_r7_dpf_sm_lds) == 18944u, "ink_r7_dpf_sm_lds must be 18,944 B (<= 21.8 KiB = 3 WGs/CU)");

  if (T == 0u || rows == 0u || nb < INK_R7_DPF_KB || (nb & (INK_R7_DPF_KB - 1u)) != 0u || !qw || !qs || !xq || !y)
    return;

  const u32 ntt = (T + INK_R7_DPF_M_T - 1u) / INK_R7_DPF_M_T;
  const u32 wg = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  const u32 tt = wg % ntt, rt = wg / ntt;
  const u32 rbase = rt * INK_R7_DPF_S_R, tbase = tt * INK_R7_DPF_M_T;
  if (rbase >= rows || tbase >= T) return;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;
  /* 8 waves = 2(M: 0..15, 16..31) x 4(N: 0..15, 16..31, 32..47, 48..63). */
  const u32 wm = (w >> 2u) << 4u;
  const u32 wn = (w & 3u) << 4u;

  f8 acc = (f8)(0.0f);

  const G u8 *qw8 = rq_ptr_u((const G u8 *)qw);
  const G u8 *qs8 = rq_ptr_u((const G u8 *)qs);
  const G u8 *xq_s = rq_ptr_u(xq + (unsigned long)nb * 32ul);

  const u32 gr = (rbase + wm + r16 < rows) ? (rbase + wm + r16) : (rows - 1u);
  const u32 w_off = gr * (nb * 32u) + (h << 4u);

  const u32 tk0 = t0 >> 3u, col0 = t0 & 7u;
  const u32 gt0 = (tbase + tk0 < T) ? (tbase + tk0) : (T - 1u);
  const u32 gt1 = (tbase + tk0 + 32u < T) ? (tbase + tk0 + 32u) : (T - 1u);
  const u32 xoff0 = gt0 * xs + (col0 << 4u);
  const u32 xoff1 = gt1 * xs + (col0 << 4u);
  const u32 st_x = (tk0 << 7u) + ((col0 ^ (tk0 & 7u)) << 4u);

  /* Wave 0 loads 32 f16x4 weight scales; waves 1..2 load 64 f32x4 activation scales. */
  const int is_sw = (w == 0u);
  const int is_sx = (w == 1u || w == 2u);
  const u32 sr = (rbase + l < rows) ? (rbase + l) : (rows - 1u);
  const u32 sw_off = is_sw ? (sr * (nb * 2u)) : 0u;
  const u32 sxi = (t0 - 32u) & 63u;
  const u32 gt_sx = (tbase + sxi < T) ? (tbase + sxi) : (T - 1u);
  const u32 sx_off = is_sx ? (gt_sx * xs) : 0u;

  u32x4 wa[INK_R7_DPF_KB];
#pragma unroll
  for (u32 k = 0u; k < INK_R7_DPF_KB; k++) wa[k] = ink_r7_dpf_ld4(qw8 + (unsigned long)w_off + (unsigned long)k * 32ul);
  u32x4 rx0 = ink_r7_dpf_ld4(xq + (unsigned long)xoff0);
  u32x4 rx1 = ink_r7_dpf_ld4(xq + (unsigned long)xoff1);
  u32x4 rsc = (u32x4)(0u);
  if (is_sw) {
    const ink_r7_dpf_u32x2 sw2 = ink_r7_dpf_ld2(qs8 + (unsigned long)sw_off);
    rsc.x = sw2.x;
    rsc.y = sw2.y;
  } else if (is_sx) {
    rsc = ink_r7_dpf_ld4(xq_s + (unsigned long)sx_off);
  }

  *(LDSP u32x4 *)(ink_r7_dpf_sm_lds.x[0] + st_x) = rx0;
  *(LDSP u32x4 *)(ink_r7_dpf_sm_lds.x[0] + st_x + 4096u) = rx1;
  if (is_sw) {
    const ink_r7_dpf_u32x2 sw2 = {rsc.x, rsc.y};
    ink_r7_dpf_stage_sw_f16(ink_r7_dpf_sm_lds.sw[0], INK_R7_DPF_S_R, l, sw2);
  } else if (is_sx) {
    ink_r7_dpf_stage_sx_f32(ink_r7_dpf_sm_lds.sx[0], INK_R7_DPF_M_T, sxi, rsc);
  }
  lbarrier();

  const v8i zm = (v8i)(0x4B400000);
  const u32 x_t_base = (wn + r16) << 7u;
  const u32 swiz_t = r16 & 7u;
  u32 bf = 0u;

#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R7_DPF_KB) {
    const u32 kb_n = kb0 + INK_R7_DPF_KB;
    const int more = (kb_n < nb);
    u32 nk = (u32)__builtin_amdgcn_readfirstlane((int)(nb - kb0 < INK_R7_DPF_KB ? nb - kb0 : INK_R7_DPF_KB));
    __asm__ volatile("" : "+s"(nk));

    if (more) {
      const G u8 *xq_n = xq + (unsigned long)kb_n * 32ul;
      rx0 = ink_r7_dpf_ld4(xq_n + (unsigned long)xoff0);
      rx1 = ink_r7_dpf_ld4(xq_n + (unsigned long)xoff1);
      if (is_sw) {
        const ink_r7_dpf_u32x2 sw2 = ink_r7_dpf_ld2(qs8 + (unsigned long)sw_off + (unsigned long)kb_n * 2ul);
        rsc.x = sw2.x;
        rsc.y = sw2.y;
      } else if (is_sx) {
        rsc = ink_r7_dpf_ld4(xq_s + (unsigned long)kb_n * 4ul + (unsigned long)sx_off);
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    const LDSP u8 *lx = ink_r7_dpf_sm_lds.x[bf] + x_t_base;
    const LDSP f16 *lsw = ink_r7_dpf_sm_lds.sw[bf];
    const LDSP float *lsx = ink_r7_dpf_sm_lds.sx[bf] + wn + r16;

#pragma unroll
    for (u32 k = 0u; k < INK_R7_DPF_KB; k++) {
      if (k < nk) {
        const u32 col_sw = ((k << 1u) + h) ^ swiz_t;
        const u32x4 b0 = *(const LDSP u32x4 *)(lx + (col_sw << 4u));
        const h8 sw_h = *(const LDSP h8 *)(lsw + k * INK_R7_DPF_S_R + wm + (h << 3u));
        const float dx0 = lsx[k * INK_R7_DPF_M_T];
        __builtin_amdgcn_sched_barrier(0);
        const v2i a0 = {(int)wa[k].x, (int)wa[k].y};
        const v2i a1 = {(int)wa[k].z, (int)wa[k].w};
        const v2i b0_0 = {(int)b0.x, (int)b0.y};
        const v2i b0_1 = {(int)b0.z, (int)b0.w};
        v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_0, zm, 0);
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_1, c0, 0);
        const f8 sw = __builtin_convertvector(sw_h, f8);
        acc += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
        __builtin_amdgcn_sched_barrier(0);
      }
      if (k == 1u && more) {
        const u32 nbf = bf ^ 1u;
        *(LDSP u32x4 *)(ink_r7_dpf_sm_lds.x[nbf] + st_x) = rx0;
        *(LDSP u32x4 *)(ink_r7_dpf_sm_lds.x[nbf] + st_x + 4096u) = rx1;
        if (is_sw) {
          const ink_r7_dpf_u32x2 sw2 = {rsc.x, rsc.y};
          ink_r7_dpf_stage_sw_f16(ink_r7_dpf_sm_lds.sw[nbf], INK_R7_DPF_S_R, l, sw2);
        } else if (is_sx) {
          ink_r7_dpf_stage_sx_f32(ink_r7_dpf_sm_lds.sx[nbf], INK_R7_DPF_M_T, sxi, rsc);
        }
        __builtin_amdgcn_sched_barrier(0);
        const G u8 *wp_n = qw8 + (unsigned long)w_off + (unsigned long)kb_n * 32ul;
        wa[0] = ink_r7_dpf_ld4(wp_n + 0ul);
        wa[1] = ink_r7_dpf_ld4(wp_n + 32ul);
        __builtin_amdgcn_sched_barrier(0);
      }
    }

    if (more) {
      const G u8 *wp_n = qw8 + (unsigned long)w_off + (unsigned long)kb_n * 32ul;
      wa[2] = ink_r7_dpf_ld4(wp_n + 64ul);
      wa[3] = ink_r7_dpf_ld4(wp_n + 96ul);
      lbarrier();
      bf ^= 1u;
    }
  }

  const u32 t = tbase + wn + r16;
  const u32 r0 = rbase + wm + (h << 3u);
  if (t < T) ink_r7_dpf_store_q8_row8(y, bias, res, t, r0, rows, ys, rs, acc);
}

/* ============================================================================
 * Kernel 2B: ie_inkling_gemm_q8_64x64_r7 (Bit 73 companion, 64x64, <= 96 VGPR)
 *   8 waves = 4(M) x 2(N), each wave computes 16x32 (acc[2]), 19,456 B LDS
 *   (3 WGs/CU). Exact 128 WGs (2.00 full waves) on wk/wv (1024x2304, T=512).
 * ========================================================================== */

KERNEL ie_inkling_gemm_q8_64x64_r7(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                   u32 rows, u32 nb, const G float *bias, const G float *res,
                                   u32 T, u32 xs, u32 ys, u32 rs) {
  static LDS struct {
    u8    x[2u][INK_R7_DPF_M_T * (INK_R7_DPF_KB * 32u)] __attribute__((aligned(16)));
    f16   sw[2u][INK_R7_DPF_KB * INK_R7_DPF_H_R] __attribute__((aligned(16)));
    float sx[2u][INK_R7_DPF_KB * INK_R7_DPF_M_T] __attribute__((aligned(16)));
  } ink_r7_dpf_h_lds;
  _Static_assert(sizeof(ink_r7_dpf_h_lds) == 19456u, "ink_r7_dpf_h_lds must be 19,456 B (<= 21.8 KiB = 3 WGs/CU)");

  if (T == 0u || rows == 0u || nb < INK_R7_DPF_KB || (nb & (INK_R7_DPF_KB - 1u)) != 0u || !qw || !qs || !xq || !y)
    return;

  const u32 ntt = (T + INK_R7_DPF_M_T - 1u) / INK_R7_DPF_M_T;
  const u32 wg = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  const u32 tt = wg % ntt, rt = wg / ntt;
  const u32 rbase = rt * INK_R7_DPF_H_R, tbase = tt * INK_R7_DPF_M_T;
  if (rbase >= rows || tbase >= T) return;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;
  /* 8 waves = 4(M: 0,16,32,48) x 2(N: 0..31, 32..63). */
  const u32 wm = (w >> 1u) << 4u;
  const u32 wn = (w & 1u) << 5u;

  f8 acc0 = (f8)(0.0f), acc1 = (f8)(0.0f);

  const G u8 *qw8 = rq_ptr_u((const G u8 *)qw);
  const G u8 *qs8 = rq_ptr_u((const G u8 *)qs);
  const G u8 *xq_s = rq_ptr_u(xq + (unsigned long)nb * 32ul);

  const u32 gr = (rbase + wm + r16 < rows) ? (rbase + wm + r16) : (rows - 1u);
  const u32 w_off = gr * (nb * 32u) + (h << 4u);

  const u32 tk0 = t0 >> 3u, col0 = t0 & 7u;
  const u32 gt0 = (tbase + tk0 < T) ? (tbase + tk0) : (T - 1u);
  const u32 gt1 = (tbase + tk0 + 32u < T) ? (tbase + tk0 + 32u) : (T - 1u);
  const u32 xoff0 = gt0 * xs + (col0 << 4u);
  const u32 xoff1 = gt1 * xs + (col0 << 4u);
  const u32 st_x = (tk0 << 7u) + ((col0 ^ (tk0 & 7u)) << 4u);

  /* Waves 0..1 load 64 f16x4 weight scales; waves 2..3 load 64 f32x4 activation scales. */
  const int is_sw = (w < 2u);
  const int is_sx = (w == 2u || w == 3u);
  const u32 swi = t0 & 63u;
  const u32 sr = (rbase + swi < rows) ? (rbase + swi) : (rows - 1u);
  const u32 sw_off = is_sw ? (sr * (nb * 2u)) : 0u;
  const u32 sxi = t0 & 63u;
  const u32 gt_sx = (tbase + sxi < T) ? (tbase + sxi) : (T - 1u);
  const u32 sx_off = is_sx ? (gt_sx * xs) : 0u;

  u32x4 wa[INK_R7_DPF_KB];
#pragma unroll
  for (u32 k = 0u; k < INK_R7_DPF_KB; k++) wa[k] = ink_r7_dpf_ld4(qw8 + (unsigned long)w_off + (unsigned long)k * 32ul);
  u32x4 rx0 = ink_r7_dpf_ld4(xq + (unsigned long)xoff0);
  u32x4 rx1 = ink_r7_dpf_ld4(xq + (unsigned long)xoff1);
  u32x4 rsc = (u32x4)(0u);
  if (is_sw) {
    const ink_r7_dpf_u32x2 sw2 = ink_r7_dpf_ld2(qs8 + (unsigned long)sw_off);
    rsc.x = sw2.x;
    rsc.y = sw2.y;
  } else if (is_sx) {
    rsc = ink_r7_dpf_ld4(xq_s + (unsigned long)sx_off);
  }

  *(LDSP u32x4 *)(ink_r7_dpf_h_lds.x[0] + st_x) = rx0;
  *(LDSP u32x4 *)(ink_r7_dpf_h_lds.x[0] + st_x + 4096u) = rx1;
  if (is_sw) {
    const ink_r7_dpf_u32x2 sw2 = {rsc.x, rsc.y};
    ink_r7_dpf_stage_sw_f16(ink_r7_dpf_h_lds.sw[0], INK_R7_DPF_H_R, swi, sw2);
  } else if (is_sx) {
    ink_r7_dpf_stage_sx_f32(ink_r7_dpf_h_lds.sx[0], INK_R7_DPF_M_T, sxi, rsc);
  }
  lbarrier();

  const v8i zm = (v8i)(0x4B400000);
  const u32 x_t_base = (wn + r16) << 7u;
  const u32 swiz_t = r16 & 7u;
  u32 bf = 0u;

#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R7_DPF_KB) {
    const u32 kb_n = kb0 + INK_R7_DPF_KB;
    const int more = (kb_n < nb);
    u32 nk = (u32)__builtin_amdgcn_readfirstlane((int)(nb - kb0 < INK_R7_DPF_KB ? nb - kb0 : INK_R7_DPF_KB));
    __asm__ volatile("" : "+s"(nk));

    if (more) {
      const G u8 *xq_n = xq + (unsigned long)kb_n * 32ul;
      rx0 = ink_r7_dpf_ld4(xq_n + (unsigned long)xoff0);
      rx1 = ink_r7_dpf_ld4(xq_n + (unsigned long)xoff1);
      if (is_sw) {
        const ink_r7_dpf_u32x2 sw2 = ink_r7_dpf_ld2(qs8 + (unsigned long)sw_off + (unsigned long)kb_n * 2ul);
        rsc.x = sw2.x;
        rsc.y = sw2.y;
      } else if (is_sx) {
        rsc = ink_r7_dpf_ld4(xq_s + (unsigned long)kb_n * 4ul + (unsigned long)sx_off);
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    const LDSP u8 *lx = ink_r7_dpf_h_lds.x[bf] + x_t_base;
    const LDSP f16 *lsw = ink_r7_dpf_h_lds.sw[bf];
    const LDSP float *lsx = ink_r7_dpf_h_lds.sx[bf] + wn + r16;

#pragma unroll
    for (u32 k = 0u; k < INK_R7_DPF_KB; k++) {
      if (k < nk) {
        const u32 col_sw = ((k << 1u) + h) ^ swiz_t;
        const u32x4 b0 = *(const LDSP u32x4 *)(lx + (col_sw << 4u));
        const u32x4 b1 = *(const LDSP u32x4 *)(lx + 2048u + (col_sw << 4u));
        const float dx0 = lsx[k * INK_R7_DPF_M_T + 0u];
        const float dx1 = lsx[k * INK_R7_DPF_M_T + 16u];
        __builtin_amdgcn_sched_barrier(0);
        const v2i a0 = {(int)wa[k].x, (int)wa[k].y};
        const v2i a1 = {(int)wa[k].z, (int)wa[k].w};
        const v2i b0_0 = {(int)b0.x, (int)b0.y}, b0_1 = {(int)b0.z, (int)b0.w};
        const v2i b1_0 = {(int)b1.x, (int)b1.y}, b1_1 = {(int)b1.z, (int)b1.w};
        v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_0, zm, 0);
        v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b1_0, zm, 0);
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_1, c0, 0);
        c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1_1, c1, 0);
        __builtin_amdgcn_sched_barrier(0);
        const h8 sw_h = *(const LDSP h8 *)(lsw + k * INK_R7_DPF_H_R + wm + (h << 3u));
        const f8 sw = __builtin_convertvector(sw_h, f8);
        acc0 += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
        acc1 += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
        __builtin_amdgcn_sched_barrier(0);
      }
      if (k == 1u && more) {
        const u32 nbf = bf ^ 1u;
        *(LDSP u32x4 *)(ink_r7_dpf_h_lds.x[nbf] + st_x) = rx0;
        *(LDSP u32x4 *)(ink_r7_dpf_h_lds.x[nbf] + st_x + 4096u) = rx1;
        if (is_sw) {
          const ink_r7_dpf_u32x2 sw2 = {rsc.x, rsc.y};
          ink_r7_dpf_stage_sw_f16(ink_r7_dpf_h_lds.sw[nbf], INK_R7_DPF_H_R, swi, sw2);
        } else if (is_sx) {
          ink_r7_dpf_stage_sx_f32(ink_r7_dpf_h_lds.sx[nbf], INK_R7_DPF_M_T, sxi, rsc);
        }
        __builtin_amdgcn_sched_barrier(0);
        const G u8 *wp_n = qw8 + (unsigned long)w_off + (unsigned long)kb_n * 32ul;
        wa[0] = ink_r7_dpf_ld4(wp_n + 0ul);
        wa[1] = ink_r7_dpf_ld4(wp_n + 32ul);
        __builtin_amdgcn_sched_barrier(0);
      }
    }

    if (more) {
      const G u8 *wp_n = qw8 + (unsigned long)w_off + (unsigned long)kb_n * 32ul;
      wa[2] = ink_r7_dpf_ld4(wp_n + 64ul);
      wa[3] = ink_r7_dpf_ld4(wp_n + 96ul);
      lbarrier();
      bf ^= 1u;
    }
  }

  const u32 r0 = rbase + wm + (h << 3u);
  const u32 t_0 = tbase + wn + r16;
  if (t_0 < T) ink_r7_dpf_store_q8_row8(y, bias, res, t_0, r0, rows, ys, rs, acc0);
  const u32 t_1 = tbase + wn + 16u + r16;
  if (t_1 < T) ink_r7_dpf_store_q8_row8(y, bias, res, t_1, r0, rows, ys, rs, acc1);
}

/* ============================================================================
 * Kernel 3: ie_inkling_router_route_pf_r7 (Bit 74)
 *   32x64x16 double-buffered F32 Router GEMM (<= 96 VGPR, 12,292 B <= 16 KiB LDS
 *   = 4 WGs/CU!) + optional fused L2-hot OP_MOE_ROUTE (eliminates 48 launches/chunk).
 *   0-bank-conflict XOR swizzle (swz = c4 << 3) on both stores and loads.
 *   Exact sequential k=0..cols-1 __builtin_fmaf order -> 0-bit-diff logits & routing!
 * ========================================================================== */

typedef struct {
  float w[2][INK_R7_DPF_R_BK][INK_R7_DPF_R_BM] __attribute__((aligned(16))); /*  4,096 B */
  float x[2][INK_R7_DPF_R_BK][INK_R7_DPF_R_BN] __attribute__((aligned(16))); /*  8,192 B */
  u32   is_last;                                                             /*      4 B */
} ink_r7_dpf_rtr_lds_t;

static LDS ink_r7_dpf_rtr_lds_t ink_r7_dpf_rtr_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r7_dpf_rtr_lds_t) <= 16384u, "ink_r7_dpf_rtr_lds must fit in 16 KiB (4 WGs/CU)");

static inline __attribute__((always_inline)) void ink_r7_dpf_rtr_store(
    const u32 buf, const u32 k0, const u32 is_w, const u32 sm, const u32 sn,
    const f32x4 vw, const f32x4 vx) {
  if (is_w) {
    ink_r7_dpf_rtr_lds.w[buf][k0 + 0u][sm] = vw.x;
    ink_r7_dpf_rtr_lds.w[buf][k0 + 1u][sm] = vw.y;
    ink_r7_dpf_rtr_lds.w[buf][k0 + 2u][sm] = vw.z;
    ink_r7_dpf_rtr_lds.w[buf][k0 + 3u][sm] = vw.w;
  }
  ink_r7_dpf_rtr_lds.x[buf][k0 + 0u][sn] = vx.x;
  ink_r7_dpf_rtr_lds.x[buf][k0 + 1u][sn] = vx.y;
  ink_r7_dpf_rtr_lds.x[buf][k0 + 2u][sn] = vx.z;
  ink_r7_dpf_rtr_lds.x[buf][k0 + 3u][sn] = vx.w;
}

static inline __attribute__((always_inline)) void ink_r7_dpf_rtr_mma(
    const u32 buf, const u32 rm, const u32 rn, float acc[4][2]) {
#pragma clang loop unroll(disable)
  for (u32 k0 = 0u; k0 < INK_R7_DPF_R_BK; k0 += 4u) {
    const u32 swz = k0 << 1u; /* (k0 / 4) * 8 in {0, 8, 16, 24}: preserves f32x4 & f32x2 align */
    const u32 sm = rm ^ swz;
    const u32 sn = rn ^ swz;
#pragma unroll
    for (u32 dk = 0u; dk < 4u; dk++) {
      const u32 k = k0 + dk;
      const f32x4 aw = *(const LDSP f32x4 *)(&ink_r7_dpf_rtr_lds.w[buf][k][sm]);
      const ink_r7_dpf_f2 bx = *(const LDSP ink_r7_dpf_f2 *)(&ink_r7_dpf_rtr_lds.x[buf][k][sn]);
      acc[0][0] = __builtin_fmaf(aw.x, bx.x, acc[0][0]);
      acc[0][1] = __builtin_fmaf(aw.x, bx.y, acc[0][1]);
      acc[1][0] = __builtin_fmaf(aw.y, bx.x, acc[1][0]);
      acc[1][1] = __builtin_fmaf(aw.y, bx.y, acc[1][1]);
      acc[2][0] = __builtin_fmaf(aw.z, bx.x, acc[2][0]);
      acc[2][1] = __builtin_fmaf(aw.z, bx.y, acc[2][1]);
      acc[3][0] = __builtin_fmaf(aw.w, bx.x, acc[3][0]);
      acc[3][1] = __builtin_fmaf(aw.w, bx.y, acc[3][1]);
    }
    __builtin_amdgcn_sched_barrier(0);
  }
}

KERNEL ie_inkling_router_route_pf_r7(const G float *w, const G float *x, G float *y,
                                     u32 rows, u32 cols, const G float *bias, const G float *res,
                                     u32 xs, u32 ys, u32 rs, u32 T,
                                     u32 K, const G float *ebias, float scale,
                                     G u32 *route, u32 ost, G u32 *count) {
  const u32 mg = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  const u32 nt = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  const u32 m0 = mg * INK_R7_DPF_R_BM;
  const u32 n0 = nt * INK_R7_DPF_R_BN;
  if (m0 >= rows || n0 >= T || cols == 0u || !w || !x || !y) return;

  const u32 t = tid();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = lane();

  /* Cooperative f32x4 loaders: BK=16 = 4 f32x4/row.
   *   W (32 x 4 = 128 loads): waves 0..3 (is_w = wv < 4), lrow = t >> 2 (0..31), c4 = t & 3.
   *   X (64 x 4 = 256 loads): all 8 waves, xrow = t >> 2 (0..63), c4 = t & 3. */
  const u32 is_w = (wv < 4u);
  const u32 xrow = t >> 2u;
  const u32 c4 = t & 3u;
  const u32 k0_ld = c4 << 2u;
  const u32 swz_ld = c4 << 3u;
  const u32 sm_ld = (xrow & 31u) ^ swz_ld;
  const u32 sn_ld = xrow ^ swz_ld;

  const u32 r_ld = m0 + (xrow & 31u);
  const u32 tk_ld = n0 + xrow;
  const int r_ok = is_w && (r_ld < rows);
  const int tk_ok = tk_ld < T;

  const G float *w_ptr = w + (unsigned long)(r_ok ? r_ld : (rows - 1u)) * cols;
  const G float *x_ptr = (const G float *)((const G u8 *)x + (unsigned long)(tk_ok ? tk_ld : (T - 1u)) * xs);

  /* Wave/lane output tile: 8 waves = 2(M) x 4(N) of 16x16; each lane computes 4(M) x 2(N). */
  const u32 wm = (wv & 1u) * 16u;
  const u32 wn = (wv >> 1u) * 16u;
  const u32 rm = wm + (l >> 3u) * 4u;
  const u32 rn = wn + (l & 7u) * 2u;

  float acc[4][2];
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    acc[i][0] = 0.0f;
    acc[i][1] = 0.0f;
  }

  const f32x4 z4 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  {
    const int cok = k0_ld < cols;
    const f32x4 vw = (r_ok && cok) ? *(const G f32x4 *)(w_ptr + k0_ld) : z4;
    const f32x4 vx = (tk_ok && cok) ? *(const G f32x4 *)(x_ptr + k0_ld) : z4;
    ink_r7_dpf_rtr_store(0u, k0_ld, is_w, sm_ld, sn_ld, vw, vx);
  }
  lbarrier();

  u32 buf = 0u;
#pragma unroll 1
  for (u32 c0 = INK_R7_DPF_R_BK; c0 < cols; c0 += INK_R7_DPF_R_BK) {
    const u32 c = c0 + k0_ld;
    const int cok = c < cols;
    const f32x4 vw = (r_ok && cok) ? *(const G f32x4 *)(w_ptr + c) : z4;
    const f32x4 vx = (tk_ok && cok) ? *(const G f32x4 *)(x_ptr + c) : z4;
    __builtin_amdgcn_sched_barrier(0);

    ink_r7_dpf_rtr_mma(buf, rm, rn, acc);

    ink_r7_dpf_rtr_store(buf ^ 1u, k0_ld, is_w, sm_ld, sn_ld, vw, vx);
    lbarrier();
    buf ^= 1u;
  }

  ink_r7_dpf_rtr_mma(buf, rm, rn, acc);

  /* Store FP32 logits (vectorized f32x4 when 4 contiguous rows fit and aligned). */
  const u32 r_base_o = m0 + rm;
  const int vec_y = (r_base_o + 4u <= rows) && ((ys & 15u) == 0u) && (!res || (rs & 15u) == 0u) &&
                    (!bias || ((((unsigned long)bias) & 15ul) == 0ul));
#pragma unroll
  for (u32 j = 0u; j < 2u; j++) {
    const u32 tk = n0 + rn + j;
    if (tk < T) {
      G float *yp = (G float *)((G u8 *)y + (unsigned long)tk * ys);
      if (vec_y) {
        f32x4 out4 = {acc[0][j], acc[1][j], acc[2][j], acc[3][j]};
        if (bias) out4 += *(const G f32x4 *)(bias + r_base_o);
        if (res) out4 += *(const G f32x4 *)((const G float *)((const G u8 *)res + (unsigned long)tk * rs) + r_base_o);
        *(G f32x4 *)(yp + r_base_o) = out4;
      } else {
        const G float *rp = res ? (const G float *)((const G u8 *)res + (unsigned long)tk * rs) : (const G float *)0;
#pragma unroll
        for (u32 i = 0u; i < 4u; i++) {
          const u32 r = r_base_o + i;
          if (r < rows) yp[r] = epilogue(acc[i][j], r, bias, rp);
        }
      }
    }
  }

  /* Optional fused OP_MOE_ROUTE: last arriving M-WG for N-tile nt routes all <= 64
   * tokens in parallel across all 8 waves using exact moe_route8() on L2-hot logits! */
  if (!route || !count || rows > 256u || K == 0u || K > 32u) return;

  const u32 ng_m = (rows + INK_R7_DPF_R_BM - 1u) / INK_R7_DPF_R_BM;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (t == 0u) {
    ink_r7_dpf_rtr_lds.is_last =
        (__scoped_atomic_fetch_add(&count[nt], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng_m - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r7_dpf_rtr_lds.is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const u32 nj_t = (T - n0 < INK_R7_DPF_R_BN) ? (T - n0) : INK_R7_DPF_R_BN;
#pragma clang loop unroll(disable)
  for (u32 dt = wv; dt < nj_t; dt += 8u) {
    const u32 tk = n0 + dt;
    const G float *lg_tk = (const G float *)((const G u8 *)y + (unsigned long)tk * ys);
    G u32 *rt_tk = (G u32 *)((G u8 *)route + (unsigned long)tk * ost);
    moe_route8(lg_tk, rows, K, ebias, scale, rt_tk);
  }
  barrier();
  if (t == 0u) count[nt] = 0u;
}

#undef INK_R7_DPF_R_BK
#undef INK_R7_DPF_R_BN
#undef INK_R7_DPF_R_BM
#undef INK_R7_DPF_H_R
#undef INK_R7_DPF_S_R
#undef INK_R7_DPF_M_T
#undef INK_R7_DPF_M_R
#undef INK_R7_DPF_KB

#endif /* !IE_MEGA */
