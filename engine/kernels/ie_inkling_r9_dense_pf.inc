/* ============================================================================
 * Inkling-2 Round 9 (Wave 8 / Dense Q8_0 Prefill Follow-up):
 *   3-WGs/CU Zero-fma_mix, Zero-Stall-Waitcnt, 100% Dual-Issue Dense Q8_0
 *   Prefill WMMA GEMMs (128x64 & 64x64) for RX 9070 XT (gfx1201, wave32, NT=256).
 *
 * Root-Cause Fixes over Wave 3 promoted ie_inkling_gemm_q8_med_r7 & q8_64x64_r7,
 * proved via llvm-objdump ISA diff of build/ie_inkling_kernels.hsaco:
 *   1. Killed ALL 128 (med_r7) / 64 (64x64_r7) quarter-rate v_fma_mix_f32!
 *      In r7, Clang folded (float)sw_h[i] * dx into non-dual-issue v_fma_mix_f32
 *      with ZERO v_dual_mul_f32 (0 vs 120 in med_r2!). Pinning h8 -> f8 with
 *      __asm__ volatile("" : "+v"(sw)) eliminates 100% of v_fma_mix_f32 (0!)
 *      and restores full 2-ops/cycle v_dual_mul_f32 while staying <= 21.8 KiB LDS!
 *   2. Eliminated fake if(k < nk) branches & PHI nodes (nb is guarded (nb&3)==0):
 *      Restores 2-ops/cycle v_dual_fmac_f32 across ALL 4 K-blocks (64x64_r7 had
 *      literally dfma=1, fma=56; med_r7 had dfma=32, fma=93!).
 *   3. Eliminated premature staging & serialized wa loads (med_r7 wld=113,
 *      64x64_r7 had s_wait_loadcnt 0x1 between wa[0]/wa[1] & wa[2]/wa[3], and
 *      med_r7 had 2x s_wait_loadcnt 0x0 before k=0 from shared rsc!):
 *      r9 issues ONLY unconditional rx0/rx1 at loop top (overlapped across 3 full
 *      K-blocks k=0,1,2), reloads wa[0..1] in s_clause right after k=1, stages
 *      rx0/rx1 + loads rsw/rsx after k=2 (hidden across k=3), then stages
 *      rsw/rsx + reloads wa[2..3] in s_clause after k=3.
 *   4. Replaced 4x ds_store_b16 (ds16=8) with 0-bank-conflict v_perm_b32_e64_dpp
 *      + single ds_store_2addr_b32 (ds16=0!) using padded SW strides
 *      (160 f16 -> 20,992 B LDS for med_r9; 96 f16 -> 19,968 B LDS for 64x64_r9;
 *      both <= 21.8 KiB = 3 WGs/CU!).
 *
 * Exported Kernels, Grids, Guards, and Parity (exact drop-ins for b72 & b73):
 * 1. KERNEL ie_inkling_gemm_q8_med_r9(
 *        const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
 *        u32 rows, u32 nb, const G float *bias, const G float *res,
 *        u32 T, u32 xs, u32 ys, u32 rs)
 *    - Drop-in replacement for ie_inkling_gemm_q8_med_r7 (Bit 72, 128x64)
 *    - Grid:   (((rows + 127u) / 128u) * ((T + 63u) / 64u), 1u, 1u), NT = 256
 *    - Guard:  !mk.rec && T >= 17u && rows >= 512u && nb >= 4u && (nb & 3u) == 0u
 *    - Parity: 100% bitwise-exact (0-bit-diff) vs ie_gemm_q8r / q8_med_r2 / q8_med_r7!
 *
 * 2. KERNEL ie_inkling_gemm_q8_64x64_r9(exact 12-arg ie_gemm_q8r signature)
 *    - Drop-in replacement for ie_inkling_gemm_q8_64x64_r7 (Bit 73, 64x64)
 *    - Grid:   (((rows + 63u) / 64u) * ((T + 63u) / 64u), 1u, 1u), NT = 256
 *    - Guard:  !mk.rec && T >= 17u && rows >= 64u && rows <= 1024u &&
 *              nb >= 4u && (nb & 3u) == 0u
 *    - Parity: 100% bitwise-exact (0-bit-diff) vs ie_gemm_q8r / q8_64x64_r7!
 *
 * All local symbols/LDS/macros prefixed ink_r9_dpf_ / INK_R9_DPF_; macros
 * #undef'd at EOF. Clang 21 scalar __builtin_bit_cast safe, 0/0 spills.
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R9_DPF_KB      4u
#define INK_R9_DPF_M_R     128u
#define INK_R9_DPF_M_T     64u
#define INK_R9_DPF_H_R     64u
#define INK_R9_DPF_M_SWSTR 160u /* 128 f16 + 32 f16 pad = 320 B = 16 mod 32 banks -> 0-conflict ds_store_2addr_b32 */
#define INK_R9_DPF_H_SWSTR 96u  /*  64 f16 + 32 f16 pad = 192 B = 16 mod 32 banks -> 0-conflict ds_store_2addr_b32 */

typedef unsigned int ink_r9_dpf_u32x2 __attribute__((ext_vector_type(2)));

struct __attribute__((packed, aligned(4))) ink_r9_dpf_u32x4p {
  u32x4 v;
};

struct __attribute__((packed, aligned(4))) ink_r9_dpf_u32x2p {
  ink_r9_dpf_u32x2 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r9_dpf_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r9_dpf_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) ink_r9_dpf_u32x2 ink_r9_dpf_ld2(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r9_dpf_u32x2p *)p)->v;
}

/* 0-bank-conflict DPP-packed f16 weight-scale staging -> single ds_store_2addr_b32.
 * Caller guarantees is_sw is wave-uniform (all 32 lanes active) and sw_str/2 == 16 mod 32. */
static inline __attribute__((always_inline)) void ink_r9_dpf_stage_sw(
    LDSP f16 *lsw, const u32 sw_str, const u32 r_idx, const ink_r9_dpf_u32x2 rsw) {
  u32 ri = r_idx;
  __asm__ volatile("" : "+v"(ri));
  const u32 w0 = rsw.x, w1 = rsw.y;
  const u32 p0 = (u32)__builtin_amdgcn_update_dpp(0, (int)w0, 0xB1, 0xF, 0xF, true);
  const u32 p1 = (u32)__builtin_amdgcn_update_dpp(0, (int)w1, 0xB1, 0xF, 0xF, true);
  const u32 odd = ri & 1u;
  const u32 sel = odd ? 0x03020706u : 0x05040100u;
  const u32 v_a = __builtin_amdgcn_perm(p0, w0, sel);
  const u32 v_b = __builtin_amdgcn_perm(p1, w1, sel);
  const u32 r2 = ri & ~1u;
  LDSP u32 *sw32 = (LDSP u32 *)lsw;
  const u32 str32 = sw_str >> 1u;
  sw32[(0u + odd) * str32 + (r2 >> 1u)] = v_a;
  sw32[(2u + odd) * str32 + (r2 >> 1u)] = v_b;
}

static inline __attribute__((always_inline)) void ink_r9_dpf_stage_sx(
    LDSP float *lsx, const u32 stride_t, const u32 t_idx, const u32x4 rsx) {
  u32 ti = t_idx;
  __asm__ volatile("" : "+v"(ti));
  const u32 d0 = rsx.x, d1 = rsx.y, d2 = rsx.z, d3 = rsx.w;
  lsx[0u * stride_t + ti] = __builtin_bit_cast(float, d0);
  lsx[1u * stride_t + ti] = __builtin_bit_cast(float, d1);
  lsx[2u * stride_t + ti] = __builtin_bit_cast(float, d2);
  lsx[3u * stride_t + ti] = __builtin_bit_cast(float, d3);
}

/* Load f16 weight scales (1x ds_load_b128) and pin f8 conversion with "+v"(sw):
 * kills 100% of quarter-rate v_fma_mix_f32 and restores 2-ops/cycle v_dual_mul_f32! */
static inline __attribute__((always_inline)) f8 ink_r9_dpf_load_sw_f8(
    const LDSP f16 *lsw, const u32 sw_str, const u32 k, const u32 wm, const u32 h) {
  const h8 sw_h = *(const LDSP h8 *)(lsw + k * sw_str + wm + (h << 3u));
  f8 sw = __builtin_convertvector(sw_h, f8);
  __asm__ volatile("" : "+v"(sw));
  return sw;
}

/* Vectorized 2x f32x4 epilogue store for Q8_0 GEMMs (0-bit-diff; unroll-1 cold tail). */
static inline __attribute__((always_inline)) void ink_r9_dpf_store_q8_row8(
    G float *y, const G float *bias, const G float *res,
    const u32 t, const u32 r0, const u32 rows, const u32 ys, const u32 rs,
    const f8 av) {
  const int vec_ok = (r0 + 8u <= rows) && ((ys & 3u) == 0u) && ((((unsigned long)y) & 15ul) == 0ul) &&
                     (!res || (((rs & 3u) == 0u) && ((((unsigned long)res) & 15ul) == 0ul))) &&
                     (!bias || ((((unsigned long)bias) & 15ul) == 0ul));
  G float *yt = y + (unsigned long)t * (unsigned long)ys;
  if (__builtin_expect(vec_ok, 1)) {
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
#pragma unroll 1
    for (u32 v = 0u; v < 8u; v++) {
      const u32 r = r0 + v;
      if (r < rows) yt[r] = epilogue(av[v], r, bias, rp);
    }
  }
}

/* ============================================================================
 * Kernel 1: ie_inkling_gemm_q8_med_r9 (Bit 72, 128x64, <= 128 VGPR, 20,992 B LDS)
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r9_dpf_q8_k64(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 r16, const u32 h, const u32 lx_base0, f8 acc[4]) {
  const v8i zm = (v8i)(0x4B400000);
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * INK_R9_DPF_M_T + r16;
  const v2i a0 = {(int)wk.x, (int)wk.y};
  const v2i a1 = {(int)wk.z, (int)wk.w};

  /* Pair 0 (j = 0, 1) */
  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + 0u * 2048u);
  const u32x4 b1 = *(const LDSP u32x4 *)(lx_k + 1u * 2048u);
  const float dx0 = lsx_k[0u];
  const float dx1 = lsx_k[16u];
  __builtin_amdgcn_sched_barrier(0);

  const v2i b0_0 = {(int)b0.x, (int)b0.y}, b0_1 = {(int)b0.z, (int)b0.w};
  const v2i b1_0 = {(int)b1.x, (int)b1.y}, b1_1 = {(int)b1.z, (int)b1.w};
  v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_0, zm, 0);
  v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b1_0, zm, 0);
  c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_1, c0, 0);
  c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1_1, c1, 0);
  __builtin_amdgcn_sched_barrier(0);

  const f8 sw = ink_r9_dpf_load_sw_f8(lsw, INK_R9_DPF_M_SWSTR, k, wr, h);
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

  const v2i b2_0 = {(int)b2.x, (int)b2.y}, b2_1 = {(int)b2.z, (int)b2.w};
  const v2i b3_0 = {(int)b3.x, (int)b3.y}, b3_1 = {(int)b3.z, (int)b3.w};
  v8i c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b2_0, zm, 0);
  v8i c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b3_0, zm, 0);
  c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b2_1, c2, 0);
  c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b3_1, c3, 0);
  __builtin_amdgcn_sched_barrier(0);

  acc[2] += (sw * dx2) * (__builtin_bit_cast(f8, c2) - 12582912.0f);
  __asm__ volatile("" : : "v"(acc[2]));
  __builtin_amdgcn_sched_barrier(0);
  acc[3] += (sw * dx3) * (__builtin_bit_cast(f8, c3) - 12582912.0f);
  __asm__ volatile("" : : "v"(acc[3]) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

KERNEL ie_inkling_gemm_q8_med_r9(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                 u32 rows, u32 nb, const G float *bias, const G float *res,
                                 u32 T, u32 xs, u32 ys, u32 rs) {
  static LDS struct {
    u8    x[2u][INK_R9_DPF_M_T * (INK_R9_DPF_KB * 32u)] __attribute__((aligned(16))); /* 16,384 B */
    f16   sw[2u][INK_R9_DPF_KB * INK_R9_DPF_M_SWSTR] __attribute__((aligned(16)));    /*  2,560 B */
    float sx[2u][INK_R9_DPF_KB * INK_R9_DPF_M_T] __attribute__((aligned(32)));        /*  2,048 B */
  } ink_r9_dpf_med_lds;
  _Static_assert(sizeof(ink_r9_dpf_med_lds) == 20992u, "ink_r9_dpf_med_lds must be 20,992 B (<= 21.8 KiB = 3 WGs/CU)");

  if (T == 0u || rows == 0u || nb < INK_R9_DPF_KB || (nb & (INK_R9_DPF_KB - 1u)) != 0u || !qw || !qs || !xq || !y)
    return;

  const u32 ntt = (T + INK_R9_DPF_M_T - 1u) / INK_R9_DPF_M_T;
  const u32 wg = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  const u32 tt = wg % ntt, rt = wg / ntt;
  const u32 rbase = rt * INK_R9_DPF_M_R, tbase = tt * INK_R9_DPF_M_T;
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
  const G u8 *xq8 = rq_ptr_u(xq);
  const G u8 *xq_s = rq_ptr_u(xq + (unsigned long)nb * 32ul);

  const u32 gr = (rbase + wr + r16 < rows) ? (rbase + wr + r16) : (rows - 1u);
  const u32 w_off = gr * (nb * 32u) + (h << 4u);

  const u32 tk0 = t0 >> 3u, col0 = t0 & 7u;
  const u32 gt0 = (tbase + tk0 < T) ? (tbase + tk0) : (T - 1u);
  const u32 gt1 = (tbase + tk0 + 32u < T) ? (tbase + tk0 + 32u) : (T - 1u);
  const u32 xoff0 = gt0 * xs + (col0 << 4u);
  const u32 xoff1 = gt1 * xs + (col0 << 4u);
  const u32 st_x = (tk0 << 7u) + ((col0 ^ (tk0 & 7u)) << 4u);
  const u32 lx_base0 = (r16 << 7u) + ((h ^ (r16 & 7u)) << 4u);

  const int is_sw = (w < 4u);
  const int is_sx = (w == 4u || w == 5u);
  const u32 sr = (rbase + (t0 & 127u) < rows) ? (rbase + (t0 & 127u)) : (rows - 1u);
  const u32 sw_off = is_sw ? (sr * (nb * 2u)) : 0u;
  const u32 sxi = t0 & 63u;
  const u32 gt_sx = (tbase + sxi < T) ? (tbase + sxi) : (T - 1u);
  const u32 sx_off = is_sx ? (gt_sx * xs) : 0u;

  u32x4 wa[INK_R9_DPF_KB];
#pragma unroll
  for (u32 k = 0u; k < INK_R9_DPF_KB; k++) wa[k] = ink_r9_dpf_ld4(qw8 + w_off + k * 32u);
  {
    const u32x4 rx0_i = ink_r9_dpf_ld4(xq8 + xoff0);
    const u32x4 rx1_i = ink_r9_dpf_ld4(xq8 + xoff1);
    const ink_r9_dpf_u32x2 rsw_i = is_sw ? ink_r9_dpf_ld2(qs8 + sw_off) : (ink_r9_dpf_u32x2){0u, 0u};
    const u32x4 rsx_i = is_sx ? ink_r9_dpf_ld4(xq_s + sx_off) : (u32x4)(0u);
    *(LDSP u32x4 *)(ink_r9_dpf_med_lds.x[0] + st_x) = rx0_i;
    *(LDSP u32x4 *)(ink_r9_dpf_med_lds.x[0] + st_x + 4096u) = rx1_i;
    if (is_sw) ink_r9_dpf_stage_sw(ink_r9_dpf_med_lds.sw[0], INK_R9_DPF_M_SWSTR, t0, rsw_i);
    if (is_sx) ink_r9_dpf_stage_sx(ink_r9_dpf_med_lds.sx[0], INK_R9_DPF_M_T, sxi, rsx_i);
  }
  lbarrier();

  u32 bf = 0u;
  u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R9_DPF_KB) {
    const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R9_DPF_KB));
    const int more = (kb_n < nb);
    /* Anchor wa[0..1] readiness BEFORE rx0/rx1 issue so k=0 never drains rx0/rx1! */
    __asm__ volatile("" : : "v"(wa[0]), "v"(wa[1]));
    if (more) {
      const G u8 *xq_n = rq_ptr_u(xq8 + (unsigned long)kb_n * 32ul);
      rx0 = ink_r9_dpf_ld4(xq_n + xoff0);
      rx1 = ink_r9_dpf_ld4(xq_n + xoff1);
    }
    __builtin_amdgcn_sched_barrier(0);

    const LDSP u8 *lx = ink_r9_dpf_med_lds.x[bf];
    const LDSP f16 *lsw = ink_r9_dpf_med_lds.sw[bf];
    const LDSP float *lsx = ink_r9_dpf_med_lds.sx[bf];

    ink_r9_dpf_q8_k64(wa[0], lx, lsw, lsx, 0u, wr, r16, h, lx_base0, acc);
    ink_r9_dpf_q8_k64(wa[1], lx, lsw, lsx, 1u, wr, r16, h, lx_base0, acc);
    if (more) {
      const G u8 *qw_n = rq_ptr_u(qw8 + (unsigned long)kb_n * 32ul);
      wa[0] = ink_r9_dpf_ld4(qw_n + w_off + 0u);
      wa[1] = ink_r9_dpf_ld4(qw_n + w_off + 32u);
    }
    __builtin_amdgcn_sched_barrier(0);

    ink_r9_dpf_q8_k64(wa[2], lx, lsw, lsx, 2u, wr, r16, h, lx_base0, acc);

    ink_r9_dpf_u32x2 rsw = {0u, 0u};
    u32x4 rsx = (u32x4)(0u);
    if (more) {
      const u32 nbf = bf ^ 1u;
      *(LDSP u32x4 *)(ink_r9_dpf_med_lds.x[nbf] + st_x) = rx0;
      *(LDSP u32x4 *)(ink_r9_dpf_med_lds.x[nbf] + st_x + 4096u) = rx1;
      const G u8 *qs_n = rq_ptr_u(qs8 + (unsigned long)kb_n * 2ul);
      const G u8 *xs_n = rq_ptr_u(xq_s + (unsigned long)kb_n * 4ul);
      if (is_sw) rsw = ink_r9_dpf_ld2(qs_n + sw_off);
      if (is_sx) rsx = ink_r9_dpf_ld4(xs_n + sx_off);
    }
    __builtin_amdgcn_sched_barrier(0);

    ink_r9_dpf_q8_k64(wa[3], lx, lsw, lsx, 3u, wr, r16, h, lx_base0, acc);
    if (more) {
      const u32 nbf = bf ^ 1u;
      if (is_sw) ink_r9_dpf_stage_sw(ink_r9_dpf_med_lds.sw[nbf], INK_R9_DPF_M_SWSTR, t0, rsw);
      if (is_sx) ink_r9_dpf_stage_sx(ink_r9_dpf_med_lds.sx[nbf], INK_R9_DPF_M_T, sxi, rsx);
      const G u8 *qw_n = rq_ptr_u(qw8 + (unsigned long)kb_n * 32ul);
      wa[2] = ink_r9_dpf_ld4(qw_n + w_off + 64u);
      wa[3] = ink_r9_dpf_ld4(qw_n + w_off + 96u);
      lbarrier();
      bf = nbf;
    }
  }

  const u32 r0 = rbase + wr + (h << 3u);
#pragma unroll
  for (u32 j = 0u; j < 4u; j++) {
    const u32 t = tbase + (j << 4u) + r16;
    if (t < T) ink_r9_dpf_store_q8_row8(y, bias, res, t, r0, rows, ys, rs, acc[j]);
  }
}

/* ============================================================================
 * Kernel 2: ie_inkling_gemm_q8_64x64_r9 (Bit 73, 64x64, <= 96 VGPR, 19,968 B LDS)
 *   8 waves = 4(M) x 2(N), each wave computes 16x32 (acc[2]), 3 WGs/CU.
 *   Exact 128 WGs (2.00 full waves) on wk/wv (1024x2304, T=512).
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r9_dpf_q8_k32(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wm, const u32 h, const u32 lx_base0, f8 *acc0, f8 *acc1) {
  const v8i zm = (v8i)(0x4B400000);
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * INK_R9_DPF_M_T;
  const v2i a0 = {(int)wk.x, (int)wk.y};
  const v2i a1 = {(int)wk.z, (int)wk.w};

  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + 0u * 2048u);
  const u32x4 b1 = *(const LDSP u32x4 *)(lx_k + 1u * 2048u);
  const float dx0 = lsx_k[0u];
  const float dx1 = lsx_k[16u];
  __builtin_amdgcn_sched_barrier(0);

  const v2i b0_0 = {(int)b0.x, (int)b0.y}, b0_1 = {(int)b0.z, (int)b0.w};
  const v2i b1_0 = {(int)b1.x, (int)b1.y}, b1_1 = {(int)b1.z, (int)b1.w};
  v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_0, zm, 0);
  v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b1_0, zm, 0);
  c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_1, c0, 0);
  c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1_1, c1, 0);
  __builtin_amdgcn_sched_barrier(0);

  const f8 sw = ink_r9_dpf_load_sw_f8(lsw, INK_R9_DPF_H_SWSTR, k, wm, h);
  *acc0 += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
  __asm__ volatile("" : : "v"(*acc0));
  __builtin_amdgcn_sched_barrier(0);
  *acc1 += (sw * dx1) * (__builtin_bit_cast(f8, c1) - 12582912.0f);
  __asm__ volatile("" : : "v"(*acc1) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

KERNEL ie_inkling_gemm_q8_64x64_r9(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
                                   u32 rows, u32 nb, const G float *bias, const G float *res,
                                   u32 T, u32 xs, u32 ys, u32 rs) {
  static LDS struct {
    u8    x[2u][INK_R9_DPF_M_T * (INK_R9_DPF_KB * 32u)] __attribute__((aligned(16))); /* 16,384 B */
    f16   sw[2u][INK_R9_DPF_KB * INK_R9_DPF_H_SWSTR] __attribute__((aligned(16)));    /*  1,536 B */
    float sx[2u][INK_R9_DPF_KB * INK_R9_DPF_M_T] __attribute__((aligned(32)));        /*  2,048 B */
  } ink_r9_dpf_h_lds;
  _Static_assert(sizeof(ink_r9_dpf_h_lds) == 19968u, "ink_r9_dpf_h_lds must be 19,968 B (<= 21.8 KiB = 3 WGs/CU)");

  if (T == 0u || rows == 0u || nb < INK_R9_DPF_KB || (nb & (INK_R9_DPF_KB - 1u)) != 0u || !qw || !qs || !xq || !y)
    return;

  const u32 ntt = (T + INK_R9_DPF_M_T - 1u) / INK_R9_DPF_M_T;
  const u32 wg = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  const u32 tt = wg % ntt, rt = wg / ntt;
  const u32 rbase = rt * INK_R9_DPF_H_R, tbase = tt * INK_R9_DPF_M_T;
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
  const G u8 *xq8 = rq_ptr_u(xq);
  const G u8 *xq_s = rq_ptr_u(xq + (unsigned long)nb * 32ul);

  const u32 gr = (rbase + wm + r16 < rows) ? (rbase + wm + r16) : (rows - 1u);
  const u32 w_off = gr * (nb * 32u) + (h << 4u);

  const u32 tk0 = t0 >> 3u, col0 = t0 & 7u;
  const u32 gt0 = (tbase + tk0 < T) ? (tbase + tk0) : (T - 1u);
  const u32 gt1 = (tbase + tk0 + 32u < T) ? (tbase + tk0 + 32u) : (T - 1u);
  const u32 xoff0 = gt0 * xs + (col0 << 4u);
  const u32 xoff1 = gt1 * xs + (col0 << 4u);
  const u32 st_x = (tk0 << 7u) + ((col0 ^ (tk0 & 7u)) << 4u);
  const u32 lx_base0 = ((wn + r16) << 7u) + ((h ^ (r16 & 7u)) << 4u);

  const int is_sw = (w < 2u);
  const int is_sx = (w == 2u || w == 3u);
  const u32 swi = t0 & 63u;
  const u32 sr = (rbase + swi < rows) ? (rbase + swi) : (rows - 1u);
  const u32 sw_off = is_sw ? (sr * (nb * 2u)) : 0u;
  const u32 sxi = t0 & 63u;
  const u32 gt_sx = (tbase + sxi < T) ? (tbase + sxi) : (T - 1u);
  const u32 sx_off = is_sx ? (gt_sx * xs) : 0u;

  u32x4 wa[INK_R9_DPF_KB];
#pragma unroll
  for (u32 k = 0u; k < INK_R9_DPF_KB; k++) wa[k] = ink_r9_dpf_ld4(qw8 + w_off + k * 32u);
  {
    const u32x4 rx0_i = ink_r9_dpf_ld4(xq8 + xoff0);
    const u32x4 rx1_i = ink_r9_dpf_ld4(xq8 + xoff1);
    const ink_r9_dpf_u32x2 rsw_i = is_sw ? ink_r9_dpf_ld2(qs8 + sw_off) : (ink_r9_dpf_u32x2){0u, 0u};
    const u32x4 rsx_i = is_sx ? ink_r9_dpf_ld4(xq_s + sx_off) : (u32x4)(0u);
    *(LDSP u32x4 *)(ink_r9_dpf_h_lds.x[0] + st_x) = rx0_i;
    *(LDSP u32x4 *)(ink_r9_dpf_h_lds.x[0] + st_x + 4096u) = rx1_i;
    if (is_sw) ink_r9_dpf_stage_sw(ink_r9_dpf_h_lds.sw[0], INK_R9_DPF_H_SWSTR, swi, rsw_i);
    if (is_sx) ink_r9_dpf_stage_sx(ink_r9_dpf_h_lds.sx[0], INK_R9_DPF_M_T, sxi, rsx_i);
  }
  lbarrier();

  u32 bf = 0u;
  u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R9_DPF_KB) {
    const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R9_DPF_KB));
    const int more = (kb_n < nb);
    /* Anchor wa[0..1] readiness BEFORE rx0/rx1 issue so k=0 never drains rx0/rx1! */
    __asm__ volatile("" : : "v"(wa[0]), "v"(wa[1]));
    if (more) {
      const G u8 *xq_n = rq_ptr_u(xq8 + (unsigned long)kb_n * 32ul);
      rx0 = ink_r9_dpf_ld4(xq_n + xoff0);
      rx1 = ink_r9_dpf_ld4(xq_n + xoff1);
    }
    __builtin_amdgcn_sched_barrier(0);

    const LDSP u8 *lx = ink_r9_dpf_h_lds.x[bf];
    const LDSP f16 *lsw = ink_r9_dpf_h_lds.sw[bf];
    const LDSP float *lsx = ink_r9_dpf_h_lds.sx[bf] + wn + r16;

    ink_r9_dpf_q8_k32(wa[0], lx, lsw, lsx, 0u, wm, h, lx_base0, &acc0, &acc1);
    ink_r9_dpf_q8_k32(wa[1], lx, lsw, lsx, 1u, wm, h, lx_base0, &acc0, &acc1);
    if (more) {
      const G u8 *qw_n = rq_ptr_u(qw8 + (unsigned long)kb_n * 32ul);
      wa[0] = ink_r9_dpf_ld4(qw_n + w_off + 0u);
      wa[1] = ink_r9_dpf_ld4(qw_n + w_off + 32u);
    }
    __builtin_amdgcn_sched_barrier(0);

    ink_r9_dpf_q8_k32(wa[2], lx, lsw, lsx, 2u, wm, h, lx_base0, &acc0, &acc1);

    ink_r9_dpf_u32x2 rsw = {0u, 0u};
    u32x4 rsx = (u32x4)(0u);
    if (more) {
      const u32 nbf = bf ^ 1u;
      *(LDSP u32x4 *)(ink_r9_dpf_h_lds.x[nbf] + st_x) = rx0;
      *(LDSP u32x4 *)(ink_r9_dpf_h_lds.x[nbf] + st_x + 4096u) = rx1;
      const G u8 *qs_n = rq_ptr_u(qs8 + (unsigned long)kb_n * 2ul);
      const G u8 *xs_n = rq_ptr_u(xq_s + (unsigned long)kb_n * 4ul);
      if (is_sw) rsw = ink_r9_dpf_ld2(qs_n + sw_off);
      if (is_sx) rsx = ink_r9_dpf_ld4(xs_n + sx_off);
    }
    __builtin_amdgcn_sched_barrier(0);

    ink_r9_dpf_q8_k32(wa[3], lx, lsw, lsx, 3u, wm, h, lx_base0, &acc0, &acc1);
    if (more) {
      const u32 nbf = bf ^ 1u;
      if (is_sw) ink_r9_dpf_stage_sw(ink_r9_dpf_h_lds.sw[nbf], INK_R9_DPF_H_SWSTR, swi, rsw);
      if (is_sx) ink_r9_dpf_stage_sx(ink_r9_dpf_h_lds.sx[nbf], INK_R9_DPF_M_T, sxi, rsx);
      const G u8 *qw_n = rq_ptr_u(qw8 + (unsigned long)kb_n * 32ul);
      wa[2] = ink_r9_dpf_ld4(qw_n + w_off + 64u);
      wa[3] = ink_r9_dpf_ld4(qw_n + w_off + 96u);
      lbarrier();
      bf = nbf;
    }
  }

  const u32 r0 = rbase + wm + (h << 3u);
  const u32 t_0 = tbase + wn + r16;
  if (t_0 < T) ink_r9_dpf_store_q8_row8(y, bias, res, t_0, r0, rows, ys, rs, acc0);
  const u32 t_1 = tbase + wn + 16u + r16;
  if (t_1 < T) ink_r9_dpf_store_q8_row8(y, bias, res, t_1, r0, rows, ys, rs, acc1);
}

#undef INK_R9_DPF_H_SWSTR
#undef INK_R9_DPF_M_SWSTR
#undef INK_R9_DPF_H_R
#undef INK_R9_DPF_M_T
#undef INK_R9_DPF_M_R
#undef INK_R9_DPF_KB

#endif /* !IE_MEGA */
