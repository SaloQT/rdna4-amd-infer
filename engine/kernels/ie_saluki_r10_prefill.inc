/* ============================================================================
 * Round 10: Bitwise-exact 128x128 double-buffered (db64), zero-LDS-stall
 *           pipelined prefill raw-quant WMMA GEMMs for RX 9070 XT (gfx1201).
 *
 * Targets the top 3 prefill raw-quant GEMM families (41.2% of total prefill
 * in post-R9-win3 trace logs/saluki-opt-r9-win3-ab/traced_cand/summary.json):
 *   1. ie_saluki_gemm_rq128_iq2xxs_r10  (IQ2_XXS, 5,832.1 ms = 25.2%, 19,456 B LDS, 3 WGs/WGP)
 *   2. ie_saluki_gemm_rq128_q2k_r10     (Q2_K,    2,103.8 ms =  9.1%, 20,480 B LDS, 3 WGs/WGP)
 *   3. ie_saluki_gemm_rq128_iq1s_r10    (IQ1_S,   1,593.8 ms =  6.9%, 26,624 B LDS, 2 WGs/WGP)
 *
 * Architectural improvements (verified against gfx1201 disassembly):
 *   - Double-buffered 64-token LDS staging (lx[2][64*128], lsx[2][4*64], plus
 *     lsa[2][4*64] for Q2_K): identical LDS footprint and WGs/WGP occupancy as
 *     single-buffered 128-token kernels, but eliminates back-to-back empty
 *     lbarrier() pairs and lockstep 16 KiB LDS write bursts, halves live rx
 *     VGPRs (rx[2] = 8 VGPRs vs rx[4] = 16 VGPRs), and halves rsx wave work.
 *   - Whole-superblock weight reuse (paired hb0/hb1 for IQ2_XXS & Q2_K = 6 ld/SB;
 *     single 5-load full-SB fetch for IQ1_S vs 10 ld/SB) with FIFO-ordered
 *     rx/rsx ahead of weight prefetches so ds_store never waits on next weights.
 *   - Batched 4-k decode_all before A-half (reused for B-half): issues all 8
 *     LUT ds_load_b32s + step=0 lx0/lsx0 loads in one batch before decode VALU.
 *   - Zero-stall software-pipelined chunk_j=2 apply_dc across all 8 (k, j0)
 *     steps: prefetches next a0/a1 and dx1 right after 4 WMMAs and next dx0
 *     after c0 post-op, eliminating per-j0 s_wait_dscnt stalls and isolating
 *     each 8-wide cvt -> 4x v_dual_mul_f32 -> 4x v_dual_fmac_f32 cleanly to
 *     prevent HB1 dual-issue collapse.
 *   - Q2_K direct ds_load_b128 broadcast of packed f16x2 lsa: eliminates all
 *     256 v_permlane16_b32 + ~170 v_mov_b32 0 instructions per half-block.
 *   - 32-bit voffset + 64-bit uniform saddr activation loads (xq_tile / sx_tile):
 *     eliminates per-half-block 64-bit VALU carry chains and saves 4-6 VGPRs.
 *   - Compact, batched epilogue: hoists row < rows, loads bias[row] once, and
 *     issues 32-deep batched res loads on full 64-token halves (32x MLP on
 *     5120x17408 ffn_down vs 64x serial s_wait_loadcnt 0; cuts ~11 KiB I$ bloat).
 *
 * Included at the end of kernels/ie_saluki_kernels.c by r10-wiring.
 * All local helpers/types/macros are prefixed r10_pf_ / R10_PF_ and all macros
 * are #undef'd at EOF. 100% bitwise identical (0 bit diffs) to ie_gemm_rq_*.
 * ========================================================================== */

typedef _Float16 r10_pf_f16;
typedef _Float16 r10_pf_f16x2 __attribute__((ext_vector_type(2)));

struct __attribute__((packed, aligned(4))) r10_pf_u32x4p {
  u32x4 v;
};
struct __attribute__((packed, aligned(4))) r10_pf_u32x2p {
  rq_u32x2 v;
};

static inline __attribute__((always_inline)) u32x4 r10_pf_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct r10_pf_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) rq_u32x2 r10_pf_ld2(const G void *p) {
  return ((const __attribute__((address_space(1))) struct r10_pf_u32x2p *)p)->v;
}

/* Tie 8 independent VGPR_32 accumulators to the SelectionDAG chain without
 * forming a VReg_256 super-register: preserves 0 spills and enables 100%
 * v_dual_fmac_f32 shrinking. */
#define R10_PF_PIN_ACC8(a)                                                                                         \
  __asm__ volatile(""                                                                                              \
                   : "+v"((a)[0]), "+v"((a)[1]), "+v"((a)[2]), "+v"((a)[3]), "+v"((a)[4]), "+v"((a)[5]),           \
                     "+v"((a)[6]), "+v"((a)[7]))

typedef struct {
  v2i w0, w1;
  float s0;
} r10_pf_dc_t;

typedef struct {
  v2i w0, w1;
  float nm0, nm1;
} r10_pf_q2k_dc_t;

/* Single-mul IQ2_XXS sign+code decode (bitwise identical to rq_ksign7 + 2x RQ_SGN_B). */
static inline __attribute__((always_inline)) v2i r10_pf_iq2_ksw(const u32 cp, const u32 s7) {
  const u32 p_ = s7 * 0x00810204u;
  const u32 lo = RQ_P2((p_ & 0x04040404u) | (cp & 0x03030303u));
  const u32 hi =
      RQ_P2(((((u32)__builtin_popcount(s7) << 26u) | (p_ >> 4u)) & 0x04040404u) | ((cp >> 4u) & 0x03030303u));
  const v2i v = {(int)lo, (int)hi};
  return v;
}

/* Exact packed f16x2 min-term subtraction for Q2_K (sum16 is an exact integer in [-2048, 2032]). */
static inline __attribute__((always_inline)) float r10_pf_q2k_sub_f16x2(float t, const u32 pair, const float nm0,
                                                                        const float nm1) {
  const r10_pf_f16x2 s = __builtin_bit_cast(r10_pf_f16x2, pair);
  t += (float)s.x * nm0;
  t += (float)s.y * nm1;
  return t;
}

static inline __attribute__((always_inline)) void r10_pf_q2k_store_lsa(LDSP u32 *lsa, const u32 ntok, const u32 tk,
                                                                       const u32 c8, const u32x4 rx) {
  const r10_pf_f16 hf = (r10_pf_f16)rqc_sum16i(rx);
  const u32 bits = (u32)__builtin_bit_cast(u16, hf);
  const u32 other = (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ 1u) * 4u), (int)bits);
  if ((c8 & 1u) == 0u) lsa[(c8 >> 1u) * ntok + tk] = bits | (other << 16u);
}

/* Stage 64 tokens x 128 int8 activations + 4*64 scales (+ optional packed f16x2 Q2_K sums) to LDS. */
static inline __attribute__((always_inline)) void r10_pf_stage_x64(LDSP u8 *lx, LDSP float *lsx, LDSP u32 *lsa,
                                                                   const int do_lsa, const u32 t0, const int is_s,
                                                                   const u32x4 rx[2], const f32x4 rsx) {
#pragma unroll
  for (u32 i = 0; i < 2u; i++) {
    const u32 p = t0 + 256u * i, tk = p >> 3u, c8 = p & 7u;
    *rqc_lx_slot(lx, tk, c8) = rx[i];
    if (do_lsa) r10_pf_q2k_store_lsa(lsa, 64u, tk, c8, rx[i]);
  }
  if (is_s) {
    lsx[t0] = rsx.x;
    lsx[64u + t0] = rsx.y;
    lsx[128u + t0] = rsx.z;
    lsx[192u + t0] = rsx.w;
  }
}

/* Load one pair of 16x16x16 A-matrix tiles (j0, j0+1) for sub-block k from 64-tok swizzled LDS. */
static inline __attribute__((always_inline)) void r10_pf_load_a_pair(const LDSP u8 *lx, const u32 k, const u32 j0,
                                                                     const u32 r16, const u32 h, v2i a0[2], v2i a1[2]) {
  const u32 tk0 = 16u * j0 + r16;
  const u32 tk1 = tk0 + 16u;
  a0[0] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk0, 2u * k) + 8u * h);
  a0[1] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk1, 2u * k) + 8u * h);
  a1[0] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk0, 2u * k + 1u) + 8u * h);
  a1[1] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk1, 2u * k + 1u) + 8u * h);
}

/* Zero-stall software-pipelined 64-token apply for single-scale subtypes (IQ2_XXS, IQ1_S).
 * Caller pre-issues step=0 (a0_c, a1_c, dx0_c) so it overlaps decode_all or global prefetches. */
static inline __attribute__((always_inline)) void r10_pf_apply_dc_body(const r10_pf_dc_t dc[4], const LDSP u8 *lx,
                                                                       const LDSP float *lsx, const u32 r16,
                                                                       const u32 h, v2i a0_c[2], v2i a1_c[2], f8 dx0_c,
                                                                       float acc[4][8]) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (u32 step = 0; step < 8u; step++) {
    const u32 k = step >> 1u;
    const u32 j0 = (step & 1u) << 1u;
    /* Stage 1: 4 WMMAs (32 cycles) consuming pre-loaded a0_c, a1_c. */
    v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_c[0], 1, dc[k].w0, z, 0);
    v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_c[1], 1, dc[k].w0, z, 0);
    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_c[0], 1, dc[k].w1, c0, 0);
    c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_c[1], 1, dc[k].w1, c1, 0);
    __builtin_amdgcn_sched_barrier(0);
    /* Stage 2: Issue dx1_c + next step's a0/a1 (reusing now-dead a0_c/a1_c VGPRs). */
    const f8 dx1_c = *(const LDSP f8 *)(lsx + k * 64u + 16u * (j0 + 1u) + 8u * h);
    if (step + 1u < 8u) {
      const u32 kn = (step + 1u) >> 1u;
      const u32 j0n = ((step + 1u) & 1u) << 1u;
      r10_pf_load_a_pair(lx, kn, j0n, r16, h, a0_c, a1_c);
    }
    __builtin_amdgcn_sched_barrier(0);
    /* Stage 3: Pure c0 post-op -> 8x v_cvt_f32_i32, 4x v_dual_mul_f32, 4x v_dual_fmac_f32. */
    {
      const float s0 = dc[k].s0;
      const f8 t0 = __builtin_convertvector(c0, f8) * s0;
#pragma unroll
      for (u32 v = 0; v < 8u; v++) acc[j0][v] += t0[v] * dx0_c[v];
      R10_PF_PIN_ACC8(acc[j0]);
    }
    __builtin_amdgcn_sched_barrier(0);
    /* Stage 4: Issue next dx0_c (reusing now-dead dx0_c VGPRs) + pure c1 post-op. */
    if (step + 1u < 8u) {
      const u32 kn = (step + 1u) >> 1u;
      const u32 j0n = ((step + 1u) & 1u) << 1u;
      dx0_c = *(const LDSP f8 *)(lsx + kn * 64u + 16u * j0n + 8u * h);
    }
    {
      const float s0 = dc[k].s0;
      const f8 t1 = __builtin_convertvector(c1, f8) * s0;
#pragma unroll
      for (u32 v = 0; v < 8u; v++) acc[j0 + 1u][v] += t1[v] * dx1_c[v];
      R10_PF_PIN_ACC8(acc[j0 + 1u]);
    }
    __builtin_amdgcn_sched_barrier(0);
  }
}

/* Compact, batched 64-token half epilogue. Caller guarantees row < rows.
 * Preserves exact epilogue() FP order: if (bias) v = v + bval; if (res) v = rval + v; */
static inline __attribute__((always_inline)) void r10_pf_store_half(G u8 *y, const G float *bias, const G u8 *res,
                                                                    const u32 T, const u32 ys, const u32 rs,
                                                                    const u32 row, const u32 t0_half, const u32 h,
                                                                    const float bval, const float acc[4][8]) {
  if (t0_half >= T) return;
  if (t0_half + 64u <= T) {
    G u8 *yp = y + (unsigned long)(t0_half + 8u * h) * ys + (unsigned long)row * 4ul;
    const unsigned long ysu = (unsigned long)ys;
    if (res) {
      const G u8 *rp = res + (unsigned long)(t0_half + 8u * h) * rs + (unsigned long)row * 4ul;
      const unsigned long rsu = (unsigned long)rs;
      float rv[4][8];
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
#pragma unroll
        for (u32 v = 0; v < 8u; v++) rv[j][v] = *(const G float *)(rp + (16ul * j + v) * rsu);
      }
      __builtin_amdgcn_sched_barrier(0);
      if (bias) {
#pragma unroll
        for (u32 j = 0; j < 4u; j++) {
#pragma unroll
          for (u32 v = 0; v < 8u; v++) *(G float *)(yp + (16ul * j + v) * ysu) = rv[j][v] + (acc[j][v] + bval);
        }
      } else {
#pragma unroll
        for (u32 j = 0; j < 4u; j++) {
#pragma unroll
          for (u32 v = 0; v < 8u; v++) *(G float *)(yp + (16ul * j + v) * ysu) = rv[j][v] + acc[j][v];
        }
      }
    } else if (bias) {
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
#pragma unroll
        for (u32 v = 0; v < 8u; v++) *(G float *)(yp + (16ul * j + v) * ysu) = acc[j][v] + bval;
      }
    } else {
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
#pragma unroll
        for (u32 v = 0; v < 8u; v++) *(G float *)(yp + (16ul * j + v) * ysu) = acc[j][v];
      }
    }
  } else {
#pragma unroll
    for (u32 j = 0; j < 4u; j++) {
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
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

/* ============================================================================
 * 1. ie_saluki_gemm_rq128_iq2xxs_r10 (IQ2_XXS, 19,456 B LDS = 3 WGs/WGP)
 * ========================================================================== */
static inline __attribute__((always_inline)) void r10_pf_xxs_load_hb0(const G u8 *p, u32 *d_word, u32 w[9],
                                                                      u32 *sh_out) {
  *sh_out = (u32)((unsigned long)p & 2ul) << 3u;
  *d_word = *rq_ap(p, 0u);
  const G u32 *qp = rq_ap(p, 2u);
  const u32x4 v0 = r10_pf_ld4(qp + 0u);
  const u32x4 v1 = r10_pf_ld4(qp + 4u);
  w[0] = v0.x; w[1] = v0.y; w[2] = v0.z; w[3] = v0.w;
  w[4] = v1.x; w[5] = v1.y; w[6] = v1.z; w[7] = v1.w;
  w[8] = qp[8];
}

static inline __attribute__((always_inline)) void r10_pf_xxs_load_hb1_tail(const G u8 *p, u32 tail[8]) {
  const G u32 *qp = rq_ap(p, 2u);
  const u32x4 v0 = r10_pf_ld4(qp + 9u);
  const u32x4 v1 = r10_pf_ld4(qp + 13u);
  tail[0] = v0.x; tail[1] = v0.y; tail[2] = v0.z; tail[3] = v0.w;
  tail[4] = v1.x; tail[5] = v1.y; tail[6] = v1.z; tail[7] = v1.w;
}

/* Batched 4-k IQ2_XXS decode overlapping all 8 LUT loads + step=0 lx0/lsx0 loads. */
static inline __attribute__((always_inline)) void r10_pf_xxs_decode_all(const u32 r[9], const u32 qsh, const float d4,
                                                                        const LDSP u32 *lut, const LDSP u8 *lx0,
                                                                        const LDSP float *lsx0, const u32 r16,
                                                                        const u32 h, r10_pf_dc_t dc[4], v2i a0_0[2],
                                                                        v2i a1_0[2], f8 *dx0_0) {
  u32 a1v[4], cp0v[4], cp1v[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 a0 = rq_align(r[2u * k], r[1u + 2u * k], qsh);
    a1v[k] = rq_align(r[1u + 2u * k], r[2u + 2u * k], qsh);
    cp0v[k] = lut[(a0 >> (8u * h)) & 255u];
    cp1v[k] = lut[(a0 >> (8u * (2u + h))) & 255u];
  }
  r10_pf_load_a_pair(lx0, 0u, 0u, r16, h, a0_0, a1_0);
  *dx0_0 = *(const LDSP f8 *)(lsx0 + 8u * h);
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    dc[k].s0 = d4 * (0.5f + (float)(a1v[k] >> 28u));
    dc[k].w0 = r10_pf_iq2_ksw(cp0v[k], (a1v[k] >> (7u * h)) & 127u);
    dc[k].w1 = r10_pf_iq2_ksw(cp1v[k], (a1v[k] >> (7u * (2u + h))) & 127u);
  }
}

KERNEL ie_saluki_gemm_rq128_iq2xxs_r10(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T,
                                       const G float *bias, const G u8 *res, u32 xs, u32 ys, u32 rs) {
  static LDS u8 lx[2u * 64u * 128u] __attribute__((aligned(16)));
  static LDS float lsx[2u * 4u * 64u] __attribute__((aligned(32)));
  static LDS u32 lut[256] __attribute__((aligned(16)));
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

  /* Uniform tile bases + 32-bit lane offsets -> saddr + voffset global_load_b128. */
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

  /* Prologue: load sb=0 hb0 A-half activations + hb0 weights. */
  u32x4 rx[2];
  f32x4 rsx = (f32x4)(0.0f);
#pragma unroll
  for (u32 i = 0; i < 2u; i++) rx[i] = r10_pf_ld4(xq_tile + (unsigned long)xoffA[i]);
  if (is_s) rsx = rq_load_f32x4((const G float *)(sx_tile + (unsigned long)sxoffA));
  u32 cur_dw, cur_w[9], cur_sh;
  r10_pf_xxs_load_hb0(bp, &cur_dw, cur_w, &cur_sh);
  r10_pf_stage_x64(lx0, lsx0, (LDSP u32 *)0, 0, t0, is_s, rx, rsx);
  lbarrier();
  __asm__ volatile("" : "+v"(cur_dw), "+v"(cur_w[0]), "+v"(cur_w[4]), "+v"(cur_w[8]));

#pragma unroll 1
  for (u32 sb = 0; sb < mb; sb++) {
    const u32 kb0 = 8u * sb;
    const u32 kb1 = kb0 + 4u;
    const G u8 *xq_hb0 = xq_tile + (unsigned long)kb0 * 32ul;
    const G u8 *sx_hb0 = sx_tile + (unsigned long)kb0 * 4ul;

    /* Phase 1 (hb0 A-half): issue rxB/rsxB FIRST, then hb1_t SECOND so ds_store(lx1)
     * waits only for rxB/rsxB while hb1_t stays in flight across A-half and B-half. */
#pragma unroll
    for (u32 i = 0; i < 2u; i++) rx[i] = r10_pf_ld4(xq_hb0 + (unsigned long)xoffB[i]);
    if (is_s) rsx = rq_load_f32x4((const G float *)(sx_hb0 + (unsigned long)sxoffB));
    u32 hb1_t[8];
    r10_pf_xxs_load_hb1_tail(bp, hb1_t);
    __builtin_amdgcn_sched_barrier(0);

    float d4 = ie_rq_h2f((cur_dw >> cur_sh) & 0xFFFFu) * 0.25f;
    __asm__ volatile("" : "+v"(d4));
    const u32 qsh = cur_sh ^ 16u;
    r10_pf_dc_t xdc[4];
    v2i a0_0[2], a1_0[2];
    f8 dx0_0;
    r10_pf_xxs_decode_all(cur_w, qsh, d4, lut, lx0, lsx0, r16, h, xdc, a0_0, a1_0, &dx0_0);
    __builtin_amdgcn_sched_barrier(0);
    r10_pf_apply_dc_body(xdc, lx0, lsx0, r16, h, a0_0, a1_0, dx0_0, acc0);

    /* Store hb0 B-half -> lx1 (race-free: all waves finished reading lx1 before prior lbarrier). */
    r10_pf_stage_x64(lx1, lsx1, (LDSP u32 *)0, 0, t0, is_s, rx, rsx);
    lbarrier();

    /* Phase 2 (hb0 B-half): overlap step=0 lx1/lsx1 loads with rxA_hb1 prefetch. */
    r10_pf_load_a_pair(lx1, 0u, 0u, r16, h, a0_0, a1_0);
    dx0_0 = *(const LDSP f8 *)(lsx1 + 8u * h);
    const G u8 *xq_hb1 = xq_tile + (unsigned long)kb1 * 32ul;
    const G u8 *sx_hb1 = sx_tile + (unsigned long)kb1 * 4ul;
#pragma unroll
    for (u32 i = 0; i < 2u; i++) rx[i] = r10_pf_ld4(xq_hb1 + (unsigned long)xoffA[i]);
    if (is_s) rsx = rq_load_f32x4((const G float *)(sx_hb1 + (unsigned long)sxoffA));
    __builtin_amdgcn_sched_barrier(0);
    r10_pf_apply_dc_body(xdc, lx1, lsx1, r16, h, a0_0, a1_0, dx0_0, acc1);

    /* Store hb1 A-half -> lx0 (race-free: all waves finished reading lx0 before Phase 1 lbarrier). */
    r10_pf_stage_x64(lx0, lsx0, (LDSP u32 *)0, 0, t0, is_s, rx, rsx);
    lbarrier();

    /* Phase 3 (hb1 A-half): assemble cur1 from cur_w[8] + hb1_t[0..7], prefetch rxB_hb1. */
    u32 cur1[9];
    cur1[0] = cur_w[8];
#pragma unroll
    for (u32 i = 0; i < 8u; i++) cur1[1u + i] = hb1_t[i];
    __asm__ volatile("" : "+v"(cur1[1]), "+v"(cur1[5]));
#pragma unroll
    for (u32 i = 0; i < 2u; i++) rx[i] = r10_pf_ld4(xq_hb1 + (unsigned long)xoffB[i]);
    if (is_s) rsx = rq_load_f32x4((const G float *)(sx_hb1 + (unsigned long)sxoffB));
    __builtin_amdgcn_sched_barrier(0);

    r10_pf_xxs_decode_all(cur1, qsh, d4, lut, lx0, lsx0, r16, h, xdc, a0_0, a1_0, &dx0_0);
    __builtin_amdgcn_sched_barrier(0);
    r10_pf_apply_dc_body(xdc, lx0, lsx0, r16, h, a0_0, a1_0, dx0_0, acc0);

    r10_pf_stage_x64(lx1, lsx1, (LDSP u32 *)0, 0, t0, is_s, rx, rsx);
    lbarrier();

    /* Phase 4 (hb1 B-half): prefetch next superblock A-half + nxt_w FIRST/SECOND inside if (more). */
    bp += 66u;
    const int more = sb + 1u < mb;
    r10_pf_load_a_pair(lx1, 0u, 0u, r16, h, a0_0, a1_0);
    dx0_0 = *(const LDSP f8 *)(lsx1 + 8u * h);
    u32 nxt_dw = 0u, nxt_w[9], nxt_sh = 0u;
    if (more) {
      const u32 kbn = 8u * (sb + 1u);
      const G u8 *xq_nxt = xq_tile + (unsigned long)kbn * 32ul;
      const G u8 *sx_nxt = sx_tile + (unsigned long)kbn * 4ul;
#pragma unroll
      for (u32 i = 0; i < 2u; i++) rx[i] = r10_pf_ld4(xq_nxt + (unsigned long)xoffA[i]);
      if (is_s) rsx = rq_load_f32x4((const G float *)(sx_nxt + (unsigned long)sxoffA));
      r10_pf_xxs_load_hb0(bp, &nxt_dw, nxt_w, &nxt_sh);
    }
    __builtin_amdgcn_sched_barrier(0);
    r10_pf_apply_dc_body(xdc, lx1, lsx1, r16, h, a0_0, a1_0, dx0_0, acc1);

    if (more) {
      r10_pf_stage_x64(lx0, lsx0, (LDSP u32 *)0, 0, t0, is_s, rx, rsx);
      lbarrier();
      cur_dw = nxt_dw;
      cur_sh = nxt_sh;
#pragma unroll
      for (u32 i = 0; i < 9u; i++) cur_w[i] = nxt_w[i];
      __asm__ volatile("" : "+v"(cur_dw), "+v"(cur_w[0]), "+v"(cur_w[4]), "+v"(cur_w[8]));
    }
  }

  if (row >= rows) return;
  const float bval = bias ? bias[row] : 0.0f;
  r10_pf_store_half(y, bias, res, T, ys, rs, row, tbase, h, bval, acc0);
  r10_pf_store_half(y, bias, res, T, ys, rs, row, tbase + 64u, h, bval, acc1);
}

/* ============================================================================
 * 2. ie_saluki_gemm_rq128_q2k_r10 (Q2_K, 20,480 B LDS = 3 WGs/WGP)
 * ========================================================================== */
static inline __attribute__((always_inline)) void r10_pf_q2k_load_hb0(const G u8 *p, const u32 h, u32 *dm_out,
                                                                      u32 sc_out[4], u32 q0_out[2], u32 q1_out[2]) {
  *dm_out = *(const G u32 *)(p + 80u);
  const u32x4 sc = r10_pf_ld4(p + 0u);
  sc_out[0] = sc.x; sc_out[1] = sc.y; sc_out[2] = sc.z; sc_out[3] = sc.w;
  const u32 qs = 16u + 8u * h;
  const rq_u32x2 q0 = r10_pf_ld2(p + qs);
  const rq_u32x2 q1 = r10_pf_ld2(p + qs + 16u);
  q0_out[0] = q0.x; q0_out[1] = q0.y;
  q1_out[0] = q1.x; q1_out[1] = q1.y;
}

static inline __attribute__((always_inline)) void r10_pf_q2k_load_hb1_tail(const G u8 *p, const u32 h, u32 q2_out[2],
                                                                           u32 q3_out[2]) {
  const u32 qs = 48u + 8u * h;
  const rq_u32x2 q2 = r10_pf_ld2(p + qs);
  const rq_u32x2 q3 = r10_pf_ld2(p + qs + 16u);
  q2_out[0] = q2.x; q2_out[1] = q2.y;
  q3_out[0] = q3.x; q3_out[1] = q3.y;
}

static inline __attribute__((always_inline)) void r10_pf_q2k_decode_all(const u32 sc_lo, const u32 sc_hi,
                                                                        const u32 q0[2], const u32 q1[2],
                                                                        const float mn, r10_pf_q2k_dc_t dc[4]) {
  const u32 sc_pair[2] = {sc_lo, sc_hi};
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 sw = sc_pair[k >> 1u] >> (16u * (k & 1u));
    const u32 sc0 = sw & 255u, sc1 = (sw >> 8u) & 255u, ksh = 2u * k;
    dc[k].w0.x = (int)(((q0[0] >> ksh) & 0x03030303u) * (sc0 & 15u));
    dc[k].w0.y = (int)(((q0[1] >> ksh) & 0x03030303u) * (sc0 & 15u));
    dc[k].w1.x = (int)(((q1[0] >> ksh) & 0x03030303u) * (sc1 & 15u));
    dc[k].w1.y = (int)(((q1[1] >> ksh) & 0x03030303u) * (sc1 & 15u));
    float nm0 = -(mn * (float)(sc0 >> 4u));
    float nm1 = -(mn * (float)(sc1 >> 4u));
    __asm__ volatile("" : "+v"(nm0), "+v"(nm1));
    dc[k].nm0 = nm0;
    dc[k].nm1 = nm1;
  }
}

/* Zero-stall pipelined 64-token Q2_K apply with direct ds_load_b128 broadcast of packed f16x2 lsa
 * (zero v_permlane16_b32, zero v_mov 0, clean cvt+dual_mul -> fma_mix -> dual_fmac stages). */
static inline __attribute__((always_inline)) void r10_pf_q2k_apply_dc_body(
    const float d, const r10_pf_q2k_dc_t dc[4], const LDSP u8 *lx, const LDSP float *lsx, const LDSP u32 *lsa,
    const u32 r16, const u32 h, v2i a0_c[2], v2i a1_c[2], u32x4 sa0_lo, u32x4 sa0_hi, f8 dx0_c, float acc[4][8]) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (u32 step = 0; step < 8u; step++) {
    const u32 k = step >> 1u;
    const u32 j0 = (step & 1u) << 1u;
    const float nm0 = dc[k].nm0, nm1 = dc[k].nm1;
    /* Stage 1: 4 WMMAs (32 cycles) consuming pre-loaded a0_c, a1_c. */
    v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_c[0], 1, dc[k].w0, z, 0);
    v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_c[1], 1, dc[k].w0, z, 0);
    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_c[0], 1, dc[k].w1, c0, 0);
    c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_c[1], 1, dc[k].w1, c1, 0);
    __builtin_amdgcn_sched_barrier(0);
    /* Stage 2: Issue sa1 (2x b128) + next step's a0/a1 (reusing dead a0_c/a1_c). */
    const u32 ts1 = k * 64u + 16u * (j0 + 1u) + 8u * h;
    const u32x4 sa1_lo = *(const LDSP u32x4 *)(lsa + ts1);
    const u32x4 sa1_hi = *(const LDSP u32x4 *)(lsa + ts1 + 4u);
    if (step + 1u < 8u) {
      const u32 kn = (step + 1u) >> 1u;
      const u32 j0n = ((step + 1u) & 1u) << 1u;
      r10_pf_load_a_pair(lx, kn, j0n, r16, h, a0_c, a1_c);
    }
    __builtin_amdgcn_sched_barrier(0);
    /* Stage 3A: c0 cvt + 4x dual_mul -> 16x v_fma_mix_f32 (consumes sa0). */
    float tv0[8];
#pragma unroll
    for (u32 v = 0; v < 8u; v++) tv0[v] = (float)c0[v] * d;
    __builtin_amdgcn_sched_barrier(0);
    {
      const u32 s0 = sa0_lo[0], s1 = sa0_lo[1], s2 = sa0_lo[2], s3 = sa0_lo[3];
      const u32 s4 = sa0_hi[0], s5 = sa0_hi[1], s6 = sa0_hi[2], s7 = sa0_hi[3];
      tv0[0] = r10_pf_q2k_sub_f16x2(tv0[0], s0, nm0, nm1);
      tv0[1] = r10_pf_q2k_sub_f16x2(tv0[1], s1, nm0, nm1);
      tv0[2] = r10_pf_q2k_sub_f16x2(tv0[2], s2, nm0, nm1);
      tv0[3] = r10_pf_q2k_sub_f16x2(tv0[3], s3, nm0, nm1);
      tv0[4] = r10_pf_q2k_sub_f16x2(tv0[4], s4, nm0, nm1);
      tv0[5] = r10_pf_q2k_sub_f16x2(tv0[5], s5, nm0, nm1);
      tv0[6] = r10_pf_q2k_sub_f16x2(tv0[6], s6, nm0, nm1);
      tv0[7] = r10_pf_q2k_sub_f16x2(tv0[7], s7, nm0, nm1);
    }
    __builtin_amdgcn_sched_barrier(0);
    /* Stage 3B: Issue dx1_c (reusing now-dead sa0 VGPRs) + c0 4x dual_fmac (consumes dx0_c). */
    const f8 dx1_c = *(const LDSP f8 *)(lsx + ts1);
#pragma unroll
    for (u32 v = 0; v < 8u; v++) acc[j0][v] += tv0[v] * dx0_c[v];
    R10_PF_PIN_ACC8(acc[j0]);
    __builtin_amdgcn_sched_barrier(0);
    /* Stage 4A: Issue next sa0 + dx0 (reusing dead dx0_c/tv0) + c1 cvt + 4x dual_mul -> 16x fma_mix. */
    if (step + 1u < 8u) {
      const u32 kn = (step + 1u) >> 1u;
      const u32 j0n = ((step + 1u) & 1u) << 1u;
      const u32 tsn = kn * 64u + 16u * j0n + 8u * h;
      sa0_lo = *(const LDSP u32x4 *)(lsa + tsn);
      sa0_hi = *(const LDSP u32x4 *)(lsa + tsn + 4u);
      dx0_c = *(const LDSP f8 *)(lsx + tsn);
    }
    float tv1[8];
#pragma unroll
    for (u32 v = 0; v < 8u; v++) tv1[v] = (float)c1[v] * d;
    __builtin_amdgcn_sched_barrier(0);
    {
      const u32 s0 = sa1_lo[0], s1 = sa1_lo[1], s2 = sa1_lo[2], s3 = sa1_lo[3];
      const u32 s4 = sa1_hi[0], s5 = sa1_hi[1], s6 = sa1_hi[2], s7 = sa1_hi[3];
      tv1[0] = r10_pf_q2k_sub_f16x2(tv1[0], s0, nm0, nm1);
      tv1[1] = r10_pf_q2k_sub_f16x2(tv1[1], s1, nm0, nm1);
      tv1[2] = r10_pf_q2k_sub_f16x2(tv1[2], s2, nm0, nm1);
      tv1[3] = r10_pf_q2k_sub_f16x2(tv1[3], s3, nm0, nm1);
      tv1[4] = r10_pf_q2k_sub_f16x2(tv1[4], s4, nm0, nm1);
      tv1[5] = r10_pf_q2k_sub_f16x2(tv1[5], s5, nm0, nm1);
      tv1[6] = r10_pf_q2k_sub_f16x2(tv1[6], s6, nm0, nm1);
      tv1[7] = r10_pf_q2k_sub_f16x2(tv1[7], s7, nm0, nm1);
    }
    __builtin_amdgcn_sched_barrier(0);
    /* Stage 4B: c1 4x dual_fmac (consumes dx1_c, issued 32 cycles ago). */
#pragma unroll
    for (u32 v = 0; v < 8u; v++) acc[j0 + 1u][v] += tv1[v] * dx1_c[v];
    R10_PF_PIN_ACC8(acc[j0 + 1u]);
    __builtin_amdgcn_sched_barrier(0);
  }
}

KERNEL ie_saluki_gemm_rq128_q2k_r10(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T,
                                    const G float *bias, const G u8 *res, u32 xs, u32 ys, u32 rs) {
  static LDS u8 lx[2u * 64u * 128u] __attribute__((aligned(16)));
  static LDS float lsx[2u * 4u * 64u] __attribute__((aligned(32)));
  static LDS u32 lsa[2u * 4u * 64u] __attribute__((aligned(16)));
  if (T == 0u || rows == 0u || nb < 8u || (nb & 7u) != 0u || !q || !xq || !y) return;
  const u32 ntt = (T + 127u) / 128u, tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 rbase = rt * 128u, tbase = tt * 128u;
  if (rbase >= rows || tbase >= T) return;
  const u32 t0 = tid(), w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u)), l = lane();
  const u32 r16 = l & 15u, h = l >> 4u;
  const int is_s = (w < 2u);
  const u32 r_loc = 16u * w + r16, row = rbase + r_loc, rr = row < rows ? row : rows - 1u;
  const u32 mb = nb >> 3u;
  const G u8 *bp = q + (unsigned long)rr * mb * 84u;

  LDSP u8 *lx0 = lx, *lx1 = lx + 64u * 128u;
  LDSP float *lsx0 = lsx, *lsx1 = lsx + 4u * 64u;
  LDSP u32 *lsa0 = lsa, *lsa1 = lsa + 4u * 64u;

  float acc0[4][8], acc1[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++) {
#pragma unroll
    for (u32 v = 0; v < 8u; v++) acc0[j][v] = 0.0f, acc1[j][v] = 0.0f;
  }

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

  /* Prologue: load sb=0 hb0 A-half activations + hb0 weights (dm + sc[0..3] + q0,q1). */
  u32x4 rx[2];
  f32x4 rsx = (f32x4)(0.0f);
#pragma unroll
  for (u32 i = 0; i < 2u; i++) rx[i] = r10_pf_ld4(xq_tile + (unsigned long)xoffA[i]);
  if (is_s) rsx = rq_load_f32x4((const G float *)(sx_tile + (unsigned long)sxoffA));
  u32 cur_dm, cur_sc[4], cur_q0[2], cur_q1[2];
  r10_pf_q2k_load_hb0(bp, h, &cur_dm, cur_sc, cur_q0, cur_q1);
  r10_pf_stage_x64(lx0, lsx0, lsa0, 1, t0, is_s, rx, rsx);
  lbarrier();
  __asm__ volatile("" : "+v"(cur_dm), "+v"(cur_sc[0]), "+v"(cur_q0[0]), "+v"(cur_q1[0]));

#pragma unroll 1
  for (u32 sb = 0; sb < mb; sb++) {
    const u32 kb0 = 8u * sb;
    const u32 kb1 = kb0 + 4u;
    const G u8 *xq_hb0 = xq_tile + (unsigned long)kb0 * 32ul;
    const G u8 *sx_hb0 = sx_tile + (unsigned long)kb0 * 4ul;

    /* Phase 1 (hb0 A-half): issue step=0 LDS loads + rxB/rsxB FIRST + hb1 q2/q3 SECOND. */
    v2i a0_0[2], a1_0[2];
    r10_pf_load_a_pair(lx0, 0u, 0u, r16, h, a0_0, a1_0);
    u32x4 sa0_lo = *(const LDSP u32x4 *)(lsa0 + 8u * h);
    u32x4 sa0_hi = *(const LDSP u32x4 *)(lsa0 + 8u * h + 4u);
    f8 dx0_0 = *(const LDSP f8 *)(lsx0 + 8u * h);
#pragma unroll
    for (u32 i = 0; i < 2u; i++) rx[i] = r10_pf_ld4(xq_hb0 + (unsigned long)xoffB[i]);
    if (is_s) rsx = rq_load_f32x4((const G float *)(sx_hb0 + (unsigned long)sxoffB));
    u32 hb1_q2[2], hb1_q3[2];
    r10_pf_q2k_load_hb1_tail(bp, h, hb1_q2, hb1_q3);
    __builtin_amdgcn_sched_barrier(0);

    float d = ie_rq_h2f(cur_dm & 0xFFFFu);
    float mn = ie_rq_h2f(cur_dm >> 16u);
    __asm__ volatile("" : "+v"(d), "+v"(mn));
    r10_pf_q2k_dc_t qdc[4];
    r10_pf_q2k_decode_all(cur_sc[0], cur_sc[1], cur_q0, cur_q1, mn, qdc);
    __builtin_amdgcn_sched_barrier(0);
    r10_pf_q2k_apply_dc_body(d, qdc, lx0, lsx0, lsa0, r16, h, a0_0, a1_0, sa0_lo, sa0_hi, dx0_0, acc0);

    r10_pf_stage_x64(lx1, lsx1, lsa1, 1, t0, is_s, rx, rsx);
    lbarrier();

    /* Phase 2 (hb0 B-half): issue step=0 lx1/lsa1/lsx1 + prefetch rxA_hb1. */
    r10_pf_load_a_pair(lx1, 0u, 0u, r16, h, a0_0, a1_0);
    sa0_lo = *(const LDSP u32x4 *)(lsa1 + 8u * h);
    sa0_hi = *(const LDSP u32x4 *)(lsa1 + 8u * h + 4u);
    dx0_0 = *(const LDSP f8 *)(lsx1 + 8u * h);
    const G u8 *xq_hb1 = xq_tile + (unsigned long)kb1 * 32ul;
    const G u8 *sx_hb1 = sx_tile + (unsigned long)kb1 * 4ul;
#pragma unroll
    for (u32 i = 0; i < 2u; i++) rx[i] = r10_pf_ld4(xq_hb1 + (unsigned long)xoffA[i]);
    if (is_s) rsx = rq_load_f32x4((const G float *)(sx_hb1 + (unsigned long)sxoffA));
    __builtin_amdgcn_sched_barrier(0);
    r10_pf_q2k_apply_dc_body(d, qdc, lx1, lsx1, lsa1, r16, h, a0_0, a1_0, sa0_lo, sa0_hi, dx0_0, acc1);

    r10_pf_stage_x64(lx0, lsx0, lsa0, 1, t0, is_s, rx, rsx);
    lbarrier();

    /* Phase 3 (hb1 A-half): drain hb1_q2/q3 (in flight since Phase 1), decode hb1, prefetch rxB_hb1. */
    __asm__ volatile("" : "+v"(hb1_q2[0]), "+v"(hb1_q3[0]));
    r10_pf_load_a_pair(lx0, 0u, 0u, r16, h, a0_0, a1_0);
    sa0_lo = *(const LDSP u32x4 *)(lsa0 + 8u * h);
    sa0_hi = *(const LDSP u32x4 *)(lsa0 + 8u * h + 4u);
    dx0_0 = *(const LDSP f8 *)(lsx0 + 8u * h);
#pragma unroll
    for (u32 i = 0; i < 2u; i++) rx[i] = r10_pf_ld4(xq_hb1 + (unsigned long)xoffB[i]);
    if (is_s) rsx = rq_load_f32x4((const G float *)(sx_hb1 + (unsigned long)sxoffB));
    __builtin_amdgcn_sched_barrier(0);

    r10_pf_q2k_decode_all(cur_sc[2], cur_sc[3], hb1_q2, hb1_q3, mn, qdc);
    __builtin_amdgcn_sched_barrier(0);
    r10_pf_q2k_apply_dc_body(d, qdc, lx0, lsx0, lsa0, r16, h, a0_0, a1_0, sa0_lo, sa0_hi, dx0_0, acc0);

    r10_pf_stage_x64(lx1, lsx1, lsa1, 1, t0, is_s, rx, rsx);
    lbarrier();

    /* Phase 4 (hb1 B-half): prefetch next superblock A-half + next hb0 weights inside if (more). */
    bp += 84u;
    const int more = sb + 1u < mb;
    r10_pf_load_a_pair(lx1, 0u, 0u, r16, h, a0_0, a1_0);
    sa0_lo = *(const LDSP u32x4 *)(lsa1 + 8u * h);
    sa0_hi = *(const LDSP u32x4 *)(lsa1 + 8u * h + 4u);
    dx0_0 = *(const LDSP f8 *)(lsx1 + 8u * h);
    u32 nxt_dm = 0u, nxt_sc[4], nxt_q0[2], nxt_q1[2];
    if (more) {
      const u32 kbn = 8u * (sb + 1u);
      const G u8 *xq_nxt = xq_tile + (unsigned long)kbn * 32ul;
      const G u8 *sx_nxt = sx_tile + (unsigned long)kbn * 4ul;
#pragma unroll
      for (u32 i = 0; i < 2u; i++) rx[i] = r10_pf_ld4(xq_nxt + (unsigned long)xoffA[i]);
      if (is_s) rsx = rq_load_f32x4((const G float *)(sx_nxt + (unsigned long)sxoffA));
      r10_pf_q2k_load_hb0(bp, h, &nxt_dm, nxt_sc, nxt_q0, nxt_q1);
    }
    __builtin_amdgcn_sched_barrier(0);
    r10_pf_q2k_apply_dc_body(d, qdc, lx1, lsx1, lsa1, r16, h, a0_0, a1_0, sa0_lo, sa0_hi, dx0_0, acc1);

    if (more) {
      r10_pf_stage_x64(lx0, lsx0, lsa0, 1, t0, is_s, rx, rsx);
      lbarrier();
      cur_dm = nxt_dm;
#pragma unroll
      for (u32 i = 0; i < 4u; i++) cur_sc[i] = nxt_sc[i];
      cur_q0[0] = nxt_q0[0]; cur_q0[1] = nxt_q0[1];
      cur_q1[0] = nxt_q1[0]; cur_q1[1] = nxt_q1[1];
      __asm__ volatile("" : "+v"(cur_dm), "+v"(cur_sc[0]), "+v"(cur_q0[0]), "+v"(cur_q1[0]));
    }
  }

  if (row >= rows) return;
  const float bval = bias ? bias[row] : 0.0f;
  r10_pf_store_half(y, bias, res, T, ys, rs, row, tbase, h, bval, acc0);
  r10_pf_store_half(y, bias, res, T, ys, rs, row, tbase + 64u, h, bval, acc1);
}

/* ============================================================================
 * 3. ie_saluki_gemm_rq128_iq1s_r10 (IQ1_S, 26,624 B LDS = 2 WGs/WGP)
 * ========================================================================== */
/* Single 5-load full 50-byte IQ1_S superblock fetch (vs 10 loads/SB in rq_hload).
 * Safe for both p % 4 == 0 (sh=0, qsh=16, qp=p) and p % 4 == 2 (sh=16, qsh=0, qp=p+2). */
static inline __attribute__((always_inline)) void r10_pf_iq1s_load_sb(const G u8 *p, u32 *d_word, u32 w[13],
                                                                      u32 *sh_out) {
  *sh_out = (u32)((unsigned long)p & 2ul) << 3u;
  *d_word = *rq_ap(p, 0u);
  const G u32 *qp = rq_ap(p, 2u);
  const u32x4 v0 = r10_pf_ld4(qp + 0u);
  const u32x4 v1 = r10_pf_ld4(qp + 4u);
  const u32x4 v2 = r10_pf_ld4(qp + 8u);
  w[0] = v0.x; w[1] = v0.y; w[2] = v0.z; w[3] = v0.w;
  w[4] = v1.x; w[5] = v1.y; w[6] = v1.z; w[7] = v1.w;
  w[8] = v2.x; w[9] = v2.y; w[10] = v2.z; w[11] = v2.w;
  w[12] = qp[12];
}

/* Batched 4-k IQ1_S decode overlapping all 8 LUT[2048] loads + step=0 lx0/lsx0 loads. */
static inline __attribute__((always_inline)) void r10_pf_iq1s_decode_all(const u32 ql_w[5], const u32 qh_w[3],
                                                                         const u32 qsh, const float d8,
                                                                         const LDSP u32 *lut, const LDSP u8 *lx0,
                                                                         const LDSP float *lsx0, const u32 r16,
                                                                         const u32 h, r10_pf_dc_t dc[4], v2i a0_0[2],
                                                                         v2i a1_0[2], f8 *dx0_0) {
  const u32 qhw0 = rq_align(qh_w[0], qh_w[1], qsh);
  const u32 qhw1 = rq_align(qh_w[1], qh_w[2], qsh);
  u32 tblv[4], cp0v[4], cp1v[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 qhw = (k < 2u) ? qhw0 : qhw1;
    const u32 qh = (qhw >> (16u * (k & 1u))) & 0xFFFFu;
    dc[k].s0 = d8 * (float)(int)(2u * ((qh >> 12u) & 7u) + 1u);
    tblv[k] = (qh & 0x8000u) != 0u ? RQ_IQ1_TBL_NEG : RQ_IQ1_TBL_POS;
    const u32 qlo = rq_align(ql_w[k], ql_w[k + 1u], qsh);
    const u32 g0 = h, g1 = 2u + h;
    cp0v[k] = lut[ie_byte(qlo, g0) | (((qh >> (3u * g0)) & 7u) << 8u)];
    cp1v[k] = lut[ie_byte(qlo, g1) | (((qh >> (3u * g1)) & 7u) << 8u)];
  }
  r10_pf_load_a_pair(lx0, 0u, 0u, r16, h, a0_0, a1_0);
  *dx0_0 = *(const LDSP f8 *)(lsx0 + 8u * h);
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    dc[k].w0 = (v2i){(int)rq_iq1_map(cp0v[k], tblv[k]), (int)rq_iq1_map(cp0v[k] >> 4u, tblv[k])};
    dc[k].w1 = (v2i){(int)rq_iq1_map(cp1v[k], tblv[k]), (int)rq_iq1_map(cp1v[k] >> 4u, tblv[k])};
  }
}

KERNEL ie_saluki_gemm_rq128_iq1s_r10(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T,
                                     const G float *bias, const G u8 *res, u32 xs, u32 ys, u32 rs) {
  static LDS u8 lx[2u * 64u * 128u] __attribute__((aligned(16)));
  static LDS float lsx[2u * 4u * 64u] __attribute__((aligned(32)));
  static LDS u32 lut[2048] __attribute__((aligned(16)));
  if (T == 0u || rows == 0u || nb < 8u || (nb & 7u) != 0u || !q || !xq || !y) return;
  const u32 ntt = (T + 127u) / 128u, tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 rbase = rt * 128u, tbase = tt * 128u;
  if (rbase >= rows || tbase >= T) return;
  const u32 t0 = tid(), w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u)), l = lane();
  const u32 r16 = l & 15u, h = l >> 4u;
  const int is_s = (w < 2u);
  const u32 r_loc = 16u * w + r16, row = rbase + r_loc, rr = row < rows ? row : rows - 1u;
  const u32 mb = nb >> 3u;
  const G u8 *bp = q + (unsigned long)rr * mb * 50u;

  LDSP u8 *lx0 = lx, *lx1 = lx + 64u * 128u;
  LDSP float *lsx0 = lsx, *lsx1 = lsx + 4u * 64u;

  float acc0[4][8], acc1[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++) {
#pragma unroll
    for (u32 v = 0; v < 8u; v++) acc0[j][v] = 0.0f, acc1[j][v] = 0.0f;
  }
  rq_stage_lut_raw(IE_RQ_IQ1_S, (LDSP ie_rq_u64 *)0, lut, 0);

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

  /* Prologue: load sb=0 hb0 A-half activations + full sb=0 superblock weights (5 loads). */
  u32x4 rx[2];
  f32x4 rsx = (f32x4)(0.0f);
#pragma unroll
  for (u32 i = 0; i < 2u; i++) rx[i] = r10_pf_ld4(xq_tile + (unsigned long)xoffA[i]);
  if (is_s) rsx = rq_load_f32x4((const G float *)(sx_tile + (unsigned long)sxoffA));
  u32 cur_dw, cur_w[13], cur_sh;
  r10_pf_iq1s_load_sb(bp, &cur_dw, cur_w, &cur_sh);
  r10_pf_stage_x64(lx0, lsx0, (LDSP u32 *)0, 0, t0, is_s, rx, rsx);
  lbarrier();
  __asm__ volatile("" : "+v"(cur_dw), "+v"(cur_w[0]), "+v"(cur_w[4]), "+v"(cur_w[8]), "+v"(cur_w[12]));

#pragma unroll 1
  for (u32 sb = 0; sb < mb; sb++) {
    const u32 kb0 = 8u * sb;
    const u32 kb1 = kb0 + 4u;
    const G u8 *xq_hb0 = xq_tile + (unsigned long)kb0 * 32ul;
    const G u8 *sx_hb0 = sx_tile + (unsigned long)kb0 * 4ul;

    /* Phase 1 (hb0 A-half): zero weight loads needed (whole SB already loaded). */
#pragma unroll
    for (u32 i = 0; i < 2u; i++) rx[i] = r10_pf_ld4(xq_hb0 + (unsigned long)xoffB[i]);
    if (is_s) rsx = rq_load_f32x4((const G float *)(sx_hb0 + (unsigned long)sxoffB));
    __builtin_amdgcn_sched_barrier(0);

    float d8 = ie_rq_h2f((cur_dw >> cur_sh) & 0xFFFFu) * 0.125f;
    __asm__ volatile("" : "+v"(d8));
    const u32 qsh = cur_sh ^ 16u;
    r10_pf_dc_t xdc[4];
    v2i a0_0[2], a1_0[2];
    f8 dx0_0;
    r10_pf_iq1s_decode_all(&cur_w[0], &cur_w[8], qsh, d8, lut, lx0, lsx0, r16, h, xdc, a0_0, a1_0, &dx0_0);
    __builtin_amdgcn_sched_barrier(0);
    r10_pf_apply_dc_body(xdc, lx0, lsx0, r16, h, a0_0, a1_0, dx0_0, acc0);

    r10_pf_stage_x64(lx1, lsx1, (LDSP u32 *)0, 0, t0, is_s, rx, rsx);
    lbarrier();

    /* Phase 2 (hb0 B-half). */
    r10_pf_load_a_pair(lx1, 0u, 0u, r16, h, a0_0, a1_0);
    dx0_0 = *(const LDSP f8 *)(lsx1 + 8u * h);
    const G u8 *xq_hb1 = xq_tile + (unsigned long)kb1 * 32ul;
    const G u8 *sx_hb1 = sx_tile + (unsigned long)kb1 * 4ul;
#pragma unroll
    for (u32 i = 0; i < 2u; i++) rx[i] = r10_pf_ld4(xq_hb1 + (unsigned long)xoffA[i]);
    if (is_s) rsx = rq_load_f32x4((const G float *)(sx_hb1 + (unsigned long)sxoffA));
    __builtin_amdgcn_sched_barrier(0);
    r10_pf_apply_dc_body(xdc, lx1, lsx1, r16, h, a0_0, a1_0, dx0_0, acc1);

    r10_pf_stage_x64(lx0, lsx0, (LDSP u32 *)0, 0, t0, is_s, rx, rsx);
    lbarrier();

    /* Phase 3 (hb1 A-half): decode hb1 from already-loaded cur_w[4..8] and cur_w[10..12]. */
#pragma unroll
    for (u32 i = 0; i < 2u; i++) rx[i] = r10_pf_ld4(xq_hb1 + (unsigned long)xoffB[i]);
    if (is_s) rsx = rq_load_f32x4((const G float *)(sx_hb1 + (unsigned long)sxoffB));
    __builtin_amdgcn_sched_barrier(0);

    r10_pf_iq1s_decode_all(&cur_w[4], &cur_w[10], qsh, d8, lut, lx0, lsx0, r16, h, xdc, a0_0, a1_0, &dx0_0);
    __builtin_amdgcn_sched_barrier(0);
    r10_pf_apply_dc_body(xdc, lx0, lsx0, r16, h, a0_0, a1_0, dx0_0, acc0);

    r10_pf_stage_x64(lx1, lsx1, (LDSP u32 *)0, 0, t0, is_s, rx, rsx);
    lbarrier();

    /* Phase 4 (hb1 B-half): prefetch next superblock A-half + full next SB weights (5 loads). */
    bp += 50u;
    const int more = sb + 1u < mb;
    r10_pf_load_a_pair(lx1, 0u, 0u, r16, h, a0_0, a1_0);
    dx0_0 = *(const LDSP f8 *)(lsx1 + 8u * h);
    u32 nxt_dw = 0u, nxt_w[13], nxt_sh = 0u;
    if (more) {
      const u32 kbn = 8u * (sb + 1u);
      const G u8 *xq_nxt = xq_tile + (unsigned long)kbn * 32ul;
      const G u8 *sx_nxt = sx_tile + (unsigned long)kbn * 4ul;
#pragma unroll
      for (u32 i = 0; i < 2u; i++) rx[i] = r10_pf_ld4(xq_nxt + (unsigned long)xoffA[i]);
      if (is_s) rsx = rq_load_f32x4((const G float *)(sx_nxt + (unsigned long)sxoffA));
      r10_pf_iq1s_load_sb(bp, &nxt_dw, nxt_w, &nxt_sh);
    }
    __builtin_amdgcn_sched_barrier(0);
    r10_pf_apply_dc_body(xdc, lx1, lsx1, r16, h, a0_0, a1_0, dx0_0, acc1);

    if (more) {
      r10_pf_stage_x64(lx0, lsx0, (LDSP u32 *)0, 0, t0, is_s, rx, rsx);
      lbarrier();
      cur_dw = nxt_dw;
      cur_sh = nxt_sh;
#pragma unroll
      for (u32 i = 0; i < 13u; i++) cur_w[i] = nxt_w[i];
      __asm__ volatile("" : "+v"(cur_dw), "+v"(cur_w[0]), "+v"(cur_w[4]), "+v"(cur_w[8]), "+v"(cur_w[12]));
    }
  }

  if (row >= rows) return;
  const float bval = bias ? bias[row] : 0.0f;
  r10_pf_store_half(y, bias, res, T, ys, rs, row, tbase, h, bval, acc0);
  r10_pf_store_half(y, bias, res, T, ys, rs, row, tbase + 64u, h, bval, acc1);
}

#undef R10_PF_PIN_ACC8
