/* ============================================================================
 * Round 11 Group 1: Bitwise-exact, 2-WG/CU (VGPR <= 192, 0-spill) 128x128
 *                   double-buffered (db64) prefill IQ2_XXS WMMA GEMM for
 *                   RX 9070 XT (gfx1201, wave32).
 *
 * Target: ie_saluki_gemm_rq128_iq2xxs_r10 (5,848.3 ms = 25.7% of all prefill,
 *         6,318 launches in logs/saluki-opt-r10-final-ab/traced_cand/summary.json).
 *
 * Root cause fixed:
 *   In R10, ie_saluki_gemm_rq128_iq2xxs_r10 compiled to 204 VGPRs (12 VGPRs
 *   above gfx1201's 192-VGPR occupancy cliff = 1536 VGPRs/SIMD / 8 waves/SIMD),
 *   dropping occupancy in half from 2 WGs/CU (128 concurrent WGs) to 1 WG/CU
 *   (64 concurrent WGs). At 1 WG/CU, every lbarrier() (4/superblock = 272 on
 *   5120x17408) drained all 4 SIMD waves simultaneously, leaving only +0.87%
 *   on 17408x5120 (res == NULL) despite R10's large pipeline/VALU wins.
 *
 * Architectural optimizations (all 100% bitwise-exact vs ie_gemm_rq_iq2xxs):
 *   1. Single-dx split-pipeline apply_dc (-8 VGPRs + eliminates 2 VReg_256
 *      tuples): replaces simultaneous f8 dx0_c + dx1_c (16 VGPRs) with a
 *      single f32x4 pair (dx_lo, dx_hi = 8 VGPRs), hiding dx1 ds_load_b128
 *      behind c1's 8x v_cvt_f32_i32 + 4x v_dual_mul_f32 (12 wave cycles x
 *      8 waves/SIMD) and next dx0 behind next step's 4 WMMAs + c0 cvt/mul
 *      (48 wave cycles), while preserving exact s_wait_dscnt FIFO overlap.
 *   2. Mid-phase staggered weight prefetch (-18 peak VGPRs): moves hb1_t[8]
 *      out of Phase 1 (where it overlapped cur_w[0..8]) to mid-Phase 2 after
 *      step 3 (k=0..1 done -> xdc[0..1] = 10 VGPRs dead), and moves nxt_w[9]
 *      to mid-Phase 4 after step 3 (xdc[0..1] = 10 VGPRs dead). Keeps rx/rsx
 *      FIFO-ahead of weight prefetches (stage_x64 never waits on weights) and
 *      hides weight loads behind 16 WMMAs + 128 VALU + stage_x64 + lbarrier().
 *   3. Branchless 32-bit voffset + uniform-saddr activation loads, single-exit
 *      SB loop, pinned st_x & local decode h_loc:
 *      - Uses (xq_tile + (unsigned long)(xoff[i] + kb_x)) to block 64-bit GEP
 *        reassociation (6 u32 VGPRs instead of 12, single-insn saddr+voffset).
 *      - Single-exit for(sb=0;;sb++) + if (!more) break; kills the !more false
 *        backedge PHI so cur_w[0..7] & cur_dw die immediately after Phase 1
 *        decode_all (-9 VGPRs at Phase 2/4 peak).
 *      - Pinned 1-VGPR st_x + immediate offsets (+0/+4096/+8192/+12288) for
 *        all 4 ds_store_b128 slots, cur_sh ^= 16u toggle, and local h_loc in
 *        decode_all (-7 VGPRs).
 *   4. Scalar-t0/t1 post-op + LUT-overlapped s0 decode: avoids VReg_256
 *      fragmentation on t0/t1 (enabling in-place c0/c1 reuse + 100%
 *      v_dual_mul_f32 / v_dual_fmac_f32) and hides 8x LUT ds_load_b32 behind
 *      4x d4 * (0.5f + scale) VALU before r11_g01_iq2_ksw.
 *   5. Low-VGPR uniform-saddr 16-deep batched epilogue (max v133 verified):
 *      replaces 32-deep rv[4][8] + 64-bit per-lane rp/yp carry chains with
 *      2x 16-deep rv[2][8] batches + 64-bit SGPR base + single 32-bit voffset
 *      (guarded against >4 GiB strides with exact 64-bit fallback).
 *   6. Naturally lands <= 192 VGPRs (~172-180 VGPRs, 8 waves/SIMD = 2 WGs/CU =
 *      128 active WGs, 0 VGPR/SGPR spills, 0 B scratch, 19,456 B LDS).
 *
 * Included at the end of kernels/ie_saluki_kernels.c by r11-wiring.
 * All local types/helpers/macros are prefixed r11_g01_ / R11_G01_ and all
 * macros are #undef'd at EOF. 100% bitwise identical (0 bit diffs) to
 * ie_gemm_rq_iq2xxs and ie_saluki_gemm_rq128_iq2xxs_r10.
 * ========================================================================== */

struct __attribute__((packed, aligned(4))) r11_g01_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 r11_g01_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct r11_g01_u32x4p *)p)->v;
}

/* Tie 8 independent VGPR_32 accumulators to the SelectionDAG chain without
 * forming a VReg_256 super-register: preserves 0 spills and 100% v_dual_fmac_f32. */
#define R11_G01_PIN_ACC8(a)                                                                                        \
  __asm__ volatile(""                                                                                              \
                   : "+v"((a)[0]), "+v"((a)[1]), "+v"((a)[2]), "+v"((a)[3]), "+v"((a)[4]), "+v"((a)[5]),           \
                     "+v"((a)[6]), "+v"((a)[7]))

typedef struct {
  v2i w0, w1;
  float s0;
} r11_g01_dc_t;

/* Single-mul IQ2_XXS sign+code decode (bitwise identical to rq_ksign7 + 2x RQ_SGN_B). */
static inline __attribute__((always_inline)) v2i r11_g01_iq2_ksw(const u32 cp, const u32 s7) {
  const u32 p_ = s7 * 0x00810204u;
  const u32 lo = RQ_P2((p_ & 0x04040404u) | (cp & 0x03030303u));
  const u32 hi =
      RQ_P2(((((u32)__builtin_popcount(s7) << 26u) | (p_ >> 4u)) & 0x04040404u) | ((cp >> 4u) & 0x03030303u));
  const v2i v = {(int)lo, (int)hi};
  return v;
}

/* Branchless 32-bit voffset + uniform 64-bit saddr 64-token activation prefetch.
 * Adding kb_x/kb_s inside u32 blocks LLVM 64-bit GEP reassociation (keeps xq_tile/sx_tile
 * as uniform SGPR saddr and xoff/sxoff as 6 x 32-bit VGPRs; valid for xs <= 0x01FFFFFFu). */
static inline __attribute__((always_inline)) void r11_g01_load_x64(
    const G u8 *xq_tile, const G u8 *sx_tile, const u32 kb_x, const u32 kb_s, const u32 xoff[2], const u32 sxoff,
    const int is_s, u32x4 rx[2], f32x4 *rsx) {
  rx[0] = r11_g01_ld4(xq_tile + (unsigned long)(xoff[0] + kb_x));
  rx[1] = r11_g01_ld4(xq_tile + (unsigned long)(xoff[1] + kb_x));
  if (is_s) *rsx = rq_load_f32x4((const G float *)(sx_tile + (unsigned long)(sxoff + kb_s)));
}

/* Stage 64 tokens x 128 int8 activations + 4*64 Q8 scales into swizzled LDS.
 * Since ((tk + 32) & 7) == (tk & 7), slot(tk + 32, c8) == slot(tk, c8) + 4096 B
 * -> single loop-invariant st_x VGPR + immediate ds_store_b128 offsets. */
static inline __attribute__((always_inline)) void r11_g01_stage_x64(LDSP u8 *lx_dst, LDSP float *lsx_dst,
                                                                    const u32 st_x, const u32 t0, const int is_s,
                                                                    const u32x4 rx[2], const f32x4 rsx) {
  *(LDSP u32x4 *)(lx_dst + st_x) = rx[0];
  *(LDSP u32x4 *)(lx_dst + st_x + 4096u) = rx[1];
  if (is_s) {
    lsx_dst[t0] = rsx.x;
    lsx_dst[64u + t0] = rsx.y;
    lsx_dst[128u + t0] = rsx.z;
    lsx_dst[192u + t0] = rsx.w;
  }
}

/* Load one pair of 16x16x16 A-matrix tiles (j0, j0+1) for sub-block k from 64-tok swizzled LDS. */
static inline __attribute__((always_inline)) void r11_g01_load_a_pair(const LDSP u8 *lx, const u32 k, const u32 j0,
                                                                      const u32 r16, const u32 h, v2i a0[2],
                                                                      v2i a1[2]) {
  const u32 tk0 = 16u * j0 + r16;
  const u32 tk1 = tk0 + 16u;
  a0[0] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk0, 2u * k) + 8u * h);
  a0[1] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk1, 2u * k) + 8u * h);
  a1[0] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk0, 2u * k + 1u) + 8u * h);
  a1[1] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk1, 2u * k + 1u) + 8u * h);
}

/* Paired-superblock IQ2_XXS weight fetch: 4 loads for hb0 (d_word + 9 words), 2 loads for hb1 tail (8 words). */
static inline __attribute__((always_inline)) void r11_g01_xxs_load_hb0(const G u8 *p, u32 *d_word, u32 w[9]) {
  *d_word = *rq_ap(p, 0u);
  const G u32 *qp = rq_ap(p, 2u);
  const u32x4 v0 = r11_g01_ld4(qp + 0u);
  const u32x4 v1 = r11_g01_ld4(qp + 4u);
  w[0] = v0.x; w[1] = v0.y; w[2] = v0.z; w[3] = v0.w;
  w[4] = v1.x; w[5] = v1.y; w[6] = v1.z; w[7] = v1.w;
  w[8] = qp[8];
}

static inline __attribute__((always_inline)) void r11_g01_xxs_load_hb1_tail(const G u8 *p, u32 tail[8]) {
  const G u32 *qp = rq_ap(p, 2u);
  const u32x4 v0 = r11_g01_ld4(qp + 9u);
  const u32x4 v1 = r11_g01_ld4(qp + 13u);
  tail[0] = v0.x; tail[1] = v0.y; tail[2] = v0.z; tail[3] = v0.w;
  tail[4] = v1.x; tail[5] = v1.y; tail[6] = v1.z; tail[7] = v1.w;
}

/* Batched 4-k IQ2_XXS decode overlapping all 8 LUT loads + step=0 lx0/lsx0 loads.
 * Computes dc[0..3].s0 while LUT loads return, keeping a0_0/a1_0/dx in flight across ksw VALU.
 * Derives local h_loc from h_off (= 32*h) so 8*h/16+8*h/7*h/14+7*h die before apply_dc_range. */
static inline __attribute__((always_inline)) void r11_g01_xxs_decode_all(
    const u32 r[9], const u32 qsh, const float d4, const LDSP u32 *lut, const LDSP u8 *lx0, const LDSP float *lsx0,
    const u32 r16, const u32 h, const u32 h_off, r11_g01_dc_t dc[4], v2i a0_0[2], v2i a1_0[2], f32x4 *dx_lo,
    f32x4 *dx_hi) {
  u32 h_loc = h_off >> 5u;
  __asm__ volatile("" : "+v"(h_loc));
  u32 a1v[4], cp0v[4], cp1v[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 a0 = rq_align(r[2u * k], r[1u + 2u * k], qsh);
    a1v[k] = rq_align(r[1u + 2u * k], r[2u + 2u * k], qsh);
    cp0v[k] = lut[(a0 >> (8u * h_loc)) & 255u];
    cp1v[k] = lut[(a0 >> (8u * (2u + h_loc))) & 255u];
  }
  r11_g01_load_a_pair(lx0, 0u, 0u, r16, h, a0_0, a1_0);
  *dx_lo = *(const LDSP f32x4 *)((const LDSP u8 *)lsx0 + h_off);
  *dx_hi = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx0 + 4u) + h_off);
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for (u32 k = 0; k < 4u; k++) dc[k].s0 = d4 * (0.5f + (float)(a1v[k] >> 28u));
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    dc[k].w0 = r11_g01_iq2_ksw(cp0v[k], (a1v[k] >> (7u * h_loc)) & 127u);
    dc[k].w1 = r11_g01_iq2_ksw(cp1v[k], (a1v[k] >> (7u * (2u + h_loc))) & 127u);
  }
}

/* Single-dx zero-stall pipelined 64-token apply for steps [step_beg, step_end) in 0..8.
 * Caller pre-issues step_beg's (a0_c, a1_c, dx_lo, dx_hi). Splitting [0,4) + [4,8) lets
 * Phase 2 / Phase 4 issue weight prefetches after k=0..1 die (freeing 10 VGPRs of xdc). */
static inline __attribute__((always_inline)) void r11_g01_apply_dc_range(
    const r11_g01_dc_t dc[4], const LDSP u8 *lx, const LDSP float *lsx, const u32 r16, const u32 h, const u32 h_off,
    v2i a0_c[2], v2i a1_c[2], f32x4 *dx_lo_p, f32x4 *dx_hi_p, float acc[4][8], const u32 step_beg,
    const u32 step_end) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  f32x4 dx_lo = *dx_lo_p;
  f32x4 dx_hi = *dx_hi_p;
#pragma unroll
  for (u32 step = step_beg; step < step_end; step++) {
    const u32 k = step >> 1u;
    const u32 j0 = (step & 1u) << 1u;
    const float s0 = dc[k].s0;
    /* Stage 1: 4 WMMAs (32 cycles) consuming pre-loaded a0_c, a1_c (s_wait_dscnt 2 -> 0 stall). */
    v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_c[0], 1, dc[k].w0, z, 0);
    v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_c[1], 1, dc[k].w0, z, 0);
    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_c[0], 1, dc[k].w1, c0, 0);
    c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_c[1], 1, dc[k].w1, c1, 0);
    __builtin_amdgcn_sched_barrier(0);
    /* Stage 2: Issue next step's a0/a1 (reusing dead a0_c/a1_c), then c0 post-op (kills c0 & dx). */
    if (step + 1u < 8u) {
      const u32 kn = (step + 1u) >> 1u;
      const u32 j0n = ((step + 1u) & 1u) << 1u;
      r11_g01_load_a_pair(lx, kn, j0n, r16, h, a0_c, a1_c);
    }
    __builtin_amdgcn_sched_barrier(0);
    {
      float t0[8];
#pragma unroll
      for (u32 v = 0; v < 8u; v++) t0[v] = (float)c0[v] * s0;
#pragma unroll
      for (u32 v = 0; v < 4u; v++) acc[j0][v] += t0[v] * dx_lo[v];
#pragma unroll
      for (u32 v = 0; v < 4u; v++) acc[j0][4u + v] += t0[4u + v] * dx_hi[v];
      R11_G01_PIN_ACC8(acc[j0]);
    }
    __builtin_amdgcn_sched_barrier(0);
    /* Stage 3A: Issue dx1 into now-dead dx_lo/dx_hi (0 overlap with dx0 -> -8 VGPRs). */
    const u32 ts1 = k * 64u + 16u * (j0 + 1u);
    dx_lo = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx + ts1) + h_off);
    dx_hi = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx + ts1 + 4u) + h_off);
    __builtin_amdgcn_sched_barrier(0);
    /* Stage 3B: c1 8x v_cvt_f32_i32 + 4x v_dual_mul_f32 hides dx1 LDS load across 8 waves/SIMD. */
    float t1[8];
#pragma unroll
    for (u32 v = 0; v < 8u; v++) t1[v] = (float)c1[v] * s0;
    __builtin_amdgcn_sched_barrier(0);
    /* Stage 4A: c1 4x v_dual_fmac_f32 (kills t1 & dx1). */
#pragma unroll
    for (u32 v = 0; v < 4u; v++) acc[j0 + 1u][v] += t1[v] * dx_lo[v];
#pragma unroll
    for (u32 v = 0; v < 4u; v++) acc[j0 + 1u][4u + v] += t1[4u + v] * dx_hi[v];
    R11_G01_PIN_ACC8(acc[j0 + 1u]);
    __builtin_amdgcn_sched_barrier(0);
    /* Stage 4B: Issue next step's dx0 (hidden across next step's 4 WMMAs + a_next + c0 cvt/mul). */
    if (step + 1u < 8u) {
      const u32 kn = (step + 1u) >> 1u;
      const u32 j0n = ((step + 1u) & 1u) << 1u;
      const u32 tsn = kn * 64u + 16u * j0n;
      dx_lo = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx + tsn) + h_off);
      dx_hi = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx + tsn + 4u) + h_off);
    }
    __builtin_amdgcn_sched_barrier(0);
  }
  *dx_lo_p = dx_lo;
  *dx_hi_p = dx_hi;
}

/* Low-VGPR uniform-saddr 16-deep batched 64-token half epilogue. Caller guarantees row < rows.
 * Preserves exact epilogue() FP order: if (bias) v = v + bval; if (res) v = rval + v; */
static inline __attribute__((always_inline)) void r11_g01_store_half(
    G u8 *y, const G float *bias, const G u8 *res, const u32 rows, const u32 T, const u32 ys, const u32 rs,
    const u32 row, const u32 t0_half, const u32 h, const float bval, float acc[4][8]) {
  if (t0_half >= T) return;
  const int y_u32 = ((unsigned long)rows * 4ul + 8ul * ys <= 0xFFFFFFFFul);
  const int r_u32 = (!res) || ((unsigned long)rows * 4ul + 8ul * rs <= 0xFFFFFFFFul);
  if (T - t0_half >= 64u && y_u32 && r_u32) {
    const unsigned long ysu = (unsigned long)ys;
    const u32 y_voff = row * 4u + (h ? 8u * ys : 0u);
    G u8 *yb = y + (unsigned long)t0_half * ysu;
    if (res) {
      const unsigned long rsu = (unsigned long)rs;
      const u32 r_voff = row * 4u + (h ? 8u * rs : 0u);
      const G u8 *rb = res + (unsigned long)t0_half * rsu;
#pragma unroll
      for (u32 j0 = 0u; j0 < 4u; j0 += 2u) {
        float rv[2][8];
#pragma unroll
        for (u32 jj = 0u; jj < 2u; jj++) {
          const G u8 *rj = rb + (unsigned long)(16u * (j0 + jj)) * rsu;
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) rv[jj][v] = *(const G float *)((rj + (unsigned long)v * rsu) + (unsigned long)r_voff);
        }
        __builtin_amdgcn_sched_barrier(0);
        if (bias) {
#pragma unroll
          for (u32 jj = 0u; jj < 2u; jj++) {
            G u8 *yj = yb + (unsigned long)(16u * (j0 + jj)) * ysu;
#pragma unroll
            for (u32 v = 0u; v < 8u; v++)
              *(G float *)((yj + (unsigned long)v * ysu) + (unsigned long)y_voff) = rv[jj][v] + (acc[j0 + jj][v] + bval);
          }
        } else {
#pragma unroll
          for (u32 jj = 0u; jj < 2u; jj++) {
            G u8 *yj = yb + (unsigned long)(16u * (j0 + jj)) * ysu;
#pragma unroll
            for (u32 v = 0u; v < 8u; v++)
              *(G float *)((yj + (unsigned long)v * ysu) + (unsigned long)y_voff) = rv[jj][v] + acc[j0 + jj][v];
          }
        }
        __builtin_amdgcn_sched_barrier(0);
      }
    } else if (bias) {
#pragma unroll
      for (u32 j = 0u; j < 4u; j++) {
        G u8 *yj = yb + (unsigned long)(16u * j) * ysu;
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) *(G float *)((yj + (unsigned long)v * ysu) + (unsigned long)y_voff) = acc[j][v] + bval;
      }
    } else {
#pragma unroll
      for (u32 j = 0u; j < 4u; j++) {
        G u8 *yj = yb + (unsigned long)(16u * j) * ysu;
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) *(G float *)((yj + (unsigned long)v * ysu) + (unsigned long)y_voff) = acc[j][v];
      }
    }
  } else {
#pragma unroll
    for (u32 j = 0u; j < 4u; j++) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 t = t0_half + 16u * j + 8u * h + v;
        if (t < T) {
          float val = acc[j][v];
          if (bias) val = val + bval;
          if (res) val = ((const G float *)(res + (unsigned long)t * rs))[row] + val;
          ((G float *)(y + (unsigned long)t * ys))[row] = val;
        }
      }
    }
  }
}

KERNEL ie_saluki_gemm_rq128_iq2xxs_r11(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T,
                                       const G float *bias, const G u8 *res, u32 xs, u32 ys, u32 rs) {
  static LDS u8 lx[2u * 64u * 128u] __attribute__((aligned(16)));
  static LDS float lsx[2u * 4u * 64u] __attribute__((aligned(32)));
  static LDS u32 lut[256] __attribute__((aligned(16)));
  _Static_assert(sizeof(lx) + sizeof(lsx) + sizeof(lut) == 19456u, "r11_g01 LDS must be 19,456 B");
  if (T == 0u || rows == 0u || nb < 8u || (nb & 7u) != 0u || !q || !xq || !y) return;
  const u32 ntt = (T + 127u) / 128u, tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 rbase = rt * 128u, tbase = tt * 128u;
  if (rbase >= rows || tbase >= T) return;
  const u32 t0 = tid(), w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u)), l = lane();
  const u32 r16 = l & 15u, h = l >> 4u;
  const int is_s = (w < 2u);
  const u32 r_loc = 16u * w + r16, row = rbase + r_loc, rr = row < rows ? row : rows - 1u;
  const u32 mb = nb >> 3u;
  const G u8 *bp = q + (unsigned long)rr * mb * 66u;

  LDSP u8 *lx0 = lx, *lx1 = lx + 64u * 128u;
  LDSP float *lsx0 = lsx, *lsx1 = lsx + 4u * 64u;

  float acc0[4][8], acc1[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++) {
#pragma unroll
    for (u32 v = 0; v < 8u; v++) acc0[j][v] = 0.0f, acc1[j][v] = 0.0f;
  }
  rq_stage_lut_raw(IE_RQ_IQ2_XXS, (LDSP ie_rq_u64 *)0, lut, 0);

  const G u8 *xq_tile = xq + (unsigned long)tbase * xs;
  const G u8 *sx_tile = xq_tile + 32ul * nb;
  const u32 dt_max = T - 1u - tbase;
  u32 xoffA[2], xoffB[2];
#pragma unroll
  for (u32 i = 0; i < 2u; i++) {
    const u32 p = t0 + 256u * i, tk = p >> 3u, c8 = p & 7u;
    const u32 dtA = tk <= dt_max ? tk : dt_max;
    const u32 dtB = 64u + tk <= dt_max ? 64u + tk : dt_max;
    xoffA[i] = dtA * xs + c8 * 16u;
    xoffB[i] = dtB * xs + c8 * 16u;
  }
  const u32 s_tk = t0 & 63u;
  const u32 sxoffA = (s_tk <= dt_max ? s_tk : dt_max) * xs;
  const u32 sxoffB = (64u + s_tk <= dt_max ? 64u + s_tk : dt_max) * xs;
  u32 st_x = (u32)((const LDSP u8 *)rqc_lx_slot(lx, t0 >> 3u, t0 & 7u) - lx);
  __asm__ volatile("" : "+v"(st_x));
  u32 h_off = 32u * h;
  __asm__ volatile("" : "+v"(h_off));

  /* Prologue: load sb=0 hb0 A-half activations + hb0 weights (4 loads). */
  u32x4 rx[2];
  f32x4 rsx = (f32x4)(0.0f);
  r11_g01_load_x64(xq_tile, sx_tile, 0u, 0u, xoffA, sxoffA, is_s, rx, &rsx);
  u32 cur_dw, cur_w[9];
  u32 cur_sh = (u32)((unsigned long)bp & 2ul) << 3u;
  r11_g01_xxs_load_hb0(bp, &cur_dw, cur_w);
  r11_g01_stage_x64(lx0, lsx0, st_x, t0, is_s, rx, rsx);
  lbarrier();
  __asm__ volatile("" : "+v"(cur_dw), "+v"(cur_w[0]), "+v"(cur_w[4]), "+v"(cur_w[8]));

  /* Single-exit loop (mb >= 1u guaranteed above): kills the !more backedge PHI so
   * cur_w[0..7] & cur_dw die right after Phase 1 decode_all (-9 VGPRs at Phase 2/4 peak). */
#pragma unroll 1
  for (u32 sb = 0u;; sb++) {
    const u32 kb0 = 8u * sb;
    const u32 kb1 = kb0 + 4u;

    /* Phase 1 (hb0 A-half): prefetch rxB/rsxB only (hb1_t deferred to mid-Phase 2 -> -8 VGPRs). */
    r11_g01_load_x64(xq_tile, sx_tile, kb0 * 32u, kb0 * 4u, xoffB, sxoffB, is_s, rx, &rsx);
    __builtin_amdgcn_sched_barrier(0);

    float d4 = ie_rq_h2f((cur_dw >> cur_sh) & 0xFFFFu) * 0.25f;
    __asm__ volatile("" : "+v"(d4));
    const u32 qsh = cur_sh ^ 16u;
    r11_g01_dc_t xdc[4];
    v2i a0_0[2], a1_0[2];
    f32x4 dx_lo, dx_hi;
    r11_g01_xxs_decode_all(cur_w, qsh, d4, lut, lx0, lsx0, r16, h, h_off, xdc, a0_0, a1_0, &dx_lo, &dx_hi);
    __builtin_amdgcn_sched_barrier(0);
    r11_g01_apply_dc_range(xdc, lx0, lsx0, r16, h, h_off, a0_0, a1_0, &dx_lo, &dx_hi, acc0, 0u, 8u);

    /* Store hb0 B-half -> lx1 (race-free: all waves finished reading lx1 before prior lbarrier). */
    r11_g01_stage_x64(lx1, lsx1, st_x, t0, is_s, rx, rsx);
    lbarrier();

    /* Phase 2 (hb0 B-half): prefetch rxA_hb1 at start; issue hb1_t after k=0..1 die (step 4). */
    r11_g01_load_a_pair(lx1, 0u, 0u, r16, h, a0_0, a1_0);
    dx_lo = *(const LDSP f32x4 *)((const LDSP u8 *)lsx1 + h_off);
    dx_hi = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx1 + 4u) + h_off);
    r11_g01_load_x64(xq_tile, sx_tile, kb1 * 32u, kb1 * 4u, xoffA, sxoffA, is_s, rx, &rsx);
    __builtin_amdgcn_sched_barrier(0);
    r11_g01_apply_dc_range(xdc, lx1, lsx1, r16, h, h_off, a0_0, a1_0, &dx_lo, &dx_hi, acc1, 0u, 4u);
    u32 hb1_t[8];
    r11_g01_xxs_load_hb1_tail(bp, hb1_t);
    __builtin_amdgcn_sched_barrier(0);
    r11_g01_apply_dc_range(xdc, lx1, lsx1, r16, h, h_off, a0_0, a1_0, &dx_lo, &dx_hi, acc1, 4u, 8u);

    /* Store hb1 A-half -> lx0 (FIFO: rxA_hb1 ahead of hb1_t, so stage_x64 never waits on hb1_t). */
    r11_g01_stage_x64(lx0, lsx0, st_x, t0, is_s, rx, rsx);
    lbarrier();

    /* Phase 3 (hb1 A-half): assemble cur1 from cur_w[8] + hb1_t[0..7], prefetch rxB_hb1. */
    u32 cur1[9];
    cur1[0] = cur_w[8];
#pragma unroll
    for (u32 i = 0; i < 8u; i++) cur1[1u + i] = hb1_t[i];
    __asm__ volatile("" : "+v"(cur1[1]), "+v"(cur1[5]));
    r11_g01_load_x64(xq_tile, sx_tile, kb1 * 32u, kb1 * 4u, xoffB, sxoffB, is_s, rx, &rsx);
    __builtin_amdgcn_sched_barrier(0);

    r11_g01_xxs_decode_all(cur1, qsh, d4, lut, lx0, lsx0, r16, h, h_off, xdc, a0_0, a1_0, &dx_lo, &dx_hi);
    __builtin_amdgcn_sched_barrier(0);
    r11_g01_apply_dc_range(xdc, lx0, lsx0, r16, h, h_off, a0_0, a1_0, &dx_lo, &dx_hi, acc0, 0u, 8u);

    r11_g01_stage_x64(lx1, lsx1, st_x, t0, is_s, rx, rsx);
    lbarrier();

    /* Phase 4 (hb1 B-half): prefetch next SB A-half at start; issue nxt_w after k=0..1 die (step 4). */
    bp += 66u;
    const int more = (sb + 1u < mb);
    r11_g01_load_a_pair(lx1, 0u, 0u, r16, h, a0_0, a1_0);
    dx_lo = *(const LDSP f32x4 *)((const LDSP u8 *)lsx1 + h_off);
    dx_hi = *(const LDSP f32x4 *)((const LDSP u8 *)(lsx1 + 4u) + h_off);
    if (more) {
      const u32 kbn = 8u * (sb + 1u);
      r11_g01_load_x64(xq_tile, sx_tile, kbn * 32u, kbn * 4u, xoffA, sxoffA, is_s, rx, &rsx);
    }
    __builtin_amdgcn_sched_barrier(0);
    r11_g01_apply_dc_range(xdc, lx1, lsx1, r16, h, h_off, a0_0, a1_0, &dx_lo, &dx_hi, acc1, 0u, 4u);
    u32 nxt_dw = 0u, nxt_w[9];
    if (more) r11_g01_xxs_load_hb0(bp, &nxt_dw, nxt_w);
    __builtin_amdgcn_sched_barrier(0);
    r11_g01_apply_dc_range(xdc, lx1, lsx1, r16, h, h_off, a0_0, a1_0, &dx_lo, &dx_hi, acc1, 4u, 8u);

    if (!more) break;
    r11_g01_stage_x64(lx0, lsx0, st_x, t0, is_s, rx, rsx);
    lbarrier();
    cur_dw = nxt_dw;
    cur_sh ^= 16u;
#pragma unroll
    for (u32 i = 0; i < 9u; i++) cur_w[i] = nxt_w[i];
    __asm__ volatile("" : "+v"(cur_dw), "+v"(cur_w[0]), "+v"(cur_w[4]), "+v"(cur_w[8]));
  }

  if (row >= rows) return;
  const u32 h_ep = h_off >> 5u;
  const float bval = bias ? bias[row] : 0.0f;
  r11_g01_store_half(y, bias, res, rows, T, ys, rs, row, tbase, h_ep, bval, acc0);
  r11_g01_store_half(y, bias, res, rows, T, ys, rs, row, tbase + 64u, h_ep, bval, acc1);
}

#undef R11_G01_PIN_ACC8
