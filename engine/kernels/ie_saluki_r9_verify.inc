/* ============================================================================
 * Round 9: Bitwise-exact pipelined T=8 verify-decode raw-quant GEMVs
 *   Extends Round 8's 0-spill 1-subblock / 1-row lookahead prefetch pattern
 *   (ie_saluki_iq2xxs_tr9_pipe1) to the remaining 7 top T=8 verify GEMV
 *   subtypes while preserving each subtype's R4-proven winning layout and
 *   exact FP32 reduction order:
 *
 *   1-row/lane tr9_pipe1 (32 rows/WG, wave-uniform s_load xq -> kmcnt,
 *   unioned 8 KiB LDS -> 8 WGs/WGP, ~66-78 VGPRs <= 96, 0 spills):
 *     1. ie_saluki_iq2s_tr8_pipe1    (IQ2_S,   u64 ulut[1024] | red, 8192 B LDS)
 *     2. ie_saluki_iq2xs_tr9_pipe1   (IQ2_XS,  u32 ulut[512]  | red, 8192 B LDS)
 *     3. ie_saluki_iq3xxs_tr9_pipe1  (IQ3_XXS, u32 ulut[256]  | red, 8192 B LDS)
 *
 *   4-rows/wave tr8_r4_pipe1 (32 rows/WG, coalesced 1-row lookahead across
 *   k=0..3 and b->b+32 after register-resident x[0..7] -> zero loadcnt HOL
 *   blocking, <= 176 VGPRs <= 192 -> 4 WGs/WGP, 0 spills):
 *     4. ie_saluki_iq1s_tr8_pipe1    (IQ1_S,   u64 ulut[2048],      16384 B LDS)
 *     5. ie_saluki_q2k_tr8_r4_pipe1  (Q2_K,    4 loads/row b128,        0 B LDS)
 *     6. ie_saluki_iq1m_tr8_r4_pipe1 (IQ1_M,   u64 ulut[2048],      16384 B LDS)
 *
 *   2-rows/wave tr8_r2_pipe1 (16 rows/WG, 6 loads/row, loadcnt-clean 3-token
 *   t=5..7 prefetch overlap, ~78-88 VGPRs <= 96 -> 8 WGs/WGP, 0 spills):
 *     7. ie_saluki_iq3s_tr8_r2_pipe1 (IQ3_S,   u32 ulut[512],        2048 B LDS)
 *
 * Included at the end of kernels/ie_saluki_kernels.c (after Round 8 &
 * ie_saluki_r9_rq128.inc). All new helpers/types are prefixed r9_v_.
 * Bitwise equal (0 bit diffs) to base ie_gemv_raw_*_tr8 and R0/R4 kernels.
 * ========================================================================== */

/* 4-byte-aligned u32x4 load helper (emits global_load_b128 on gfx1201 with
 * zero alignment UB for Q2_K's 84-byte stride blocks). */
struct __attribute__((packed, aligned(4))) r9_v_u32x4_p4 { u32x4 v; };
static inline __attribute__((always_inline)) u32x4 r9_v_ld_u32x4_a4(const G u8 *p, const u32 o) {
  return ((const G struct r9_v_u32x4_p4 *)(p + o))->v;
}

/* Wave-uniform scalar-load 8-token dot for split-16 tr9 subtypes (IQ2_S, IQ2_XS).
 * Preserves exact dav * (dw0 * (float)(int)id0 + dw1 * (float)(int)id1) order. */
static inline __attribute__((always_inline)) void r9_v_tr9_dot8_sload_split16(
    const u32 wl[4], const u32 wh[4], const float dw0, const float dw1,
    const G u8 *xq, const u32 xs, const u32 xo, const u32 dao, float aj[8]) {
#pragma clang fp contract(off)
#pragma unroll
  for (u32 t = 0; t < 8u; t++) {
    const TR8_CSP u8 *xb = (const TR8_CSP u8 *)tr8_uniform(xq + (unsigned long)t * xs);
    const float dav = *(const TR8_CSP float *)(xb + dao);
    const u32x4 x0 = *(const TR8_CSP u32x4 *)(xb + xo);
    const u32x4 x1 = *(const TR8_CSP u32x4 *)(xb + xo + 16u);
    u32 id0 = ie_dot4_ss(0u, wl[0], x0.x);
    id0 = ie_dot4_ss(id0, wh[0], x0.y);
    id0 = ie_dot4_ss(id0, wl[1], x0.z);
    id0 = ie_dot4_ss(id0, wh[1], x0.w);
    u32 id1 = ie_dot4_ss(0u, wl[2], x1.x);
    id1 = ie_dot4_ss(id1, wh[2], x1.y);
    id1 = ie_dot4_ss(id1, wl[3], x1.z);
    id1 = ie_dot4_ss(id1, wh[3], x1.w);
    aj[t] += dav * (dw0 * (float)(int)id0 + dw1 * (float)(int)id1);
    if ((t & 3u) == 3u) __asm__ volatile("" : "+v"(aj[t - 3u]), "+v"(aj[t - 2u]), "+v"(aj[t - 1u]), "+v"(aj[t]));
  }
}

/* Exact wave_tree_f reduction and epilogue for tr8 R=2 / R=4 kernels. */
static inline __attribute__((always_inline)) void r9_v_tr8_epilogue(
    const u32 R, const u32 r0, const u32 rows, const u32 l,
    float a[][8], const G float *bias, const G float *res, G float *y, const u32 ys, const u32 rs) {
#pragma clang fp contract(off)
  float o[4];
#pragma unroll
  for (u32 k = 0; k < R; k++) {
    o[k] = 0.0f;
#pragma unroll
    for (u32 t = 0; t < 8u; t++) {
      const float sv = I2F(__builtin_amdgcn_readlane(F2I(wave_tree_f(a[k][t])), 0));
      if (l == t) o[k] = sv;
    }
  }
  if (l < 8u) {
    const G float *rt = res ? (const G float *)((const G u8 *)res + (unsigned long)l * rs) : res;
    G float *yt = (G float *)((G u8 *)y + (unsigned long)l * ys);
#pragma unroll
    for (u32 k = 0; k < R; k++) if (bias && r0 + k < rows) o[k] = o[k] + bias[r0 + k];
#pragma unroll
    for (u32 k = 0; k < R; k++) if (rt && r0 + k < rows) o[k] = rt[r0 + k] + o[k];
#pragma unroll
    for (u32 k = 0; k < R; k++) if (r0 + k < rows) yt[r0 + k] = o[k];
  }
}

/* --- R9-V1: ie_saluki_iq2s_tr8_pipe1 (T=8 verify IQ2_S, tr9 1 row/lane) --- */
static inline __attribute__((always_inline)) void r9_v_decode_iq2s(
    const LDSP ie_rq_u64 *ulut, const u32 d16, const u32 qlo, const u32 sgn,
    const u32 qh, const u32 sc, u32 wl[4], u32 wh[4], float *dw0_out, float *dw1_out) {
#pragma clang fp contract(off)
  const float d4 = ie_rq_h2f(d16) * 0.25f;
  *dw0_out = d4 * (0.5f + (float)(sc & 15u));
  *dw1_out = d4 * (0.5f + (float)(sc >> 4u));
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 idx = ie_byte(qlo, g) | ((qh << (8u - 2u * g)) & 0x300u);
    const u32 sb = ie_byte(sgn, g);
    const ie_rq_u64 gv = ulut[idx];
    const u32 p_ = (sb & 127u) * 0x00810204u;
    wl[g] = TR8_P2((p_ & 0x04040404u) | (u32)gv);
    wh[g] = TR8_P2((((sb << 19u) | (p_ >> 4u)) & 0x04040404u) | (u32)(gv >> 32u));
  }
}

KERNEL ie_saluki_iq2s_tr8_pipe1(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                                const G float *bias, const G float *res, const G float *nw, G float *ny,
                                G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  const u32 r0 = 32u * wgid();
  if (T != 8u || r0 >= rows) return;
  static LDS union {
    ie_rq_u64 ulut[1024];
    float red[8u * 8u * 32u];
  } sm __attribute__((aligned(16)));
  for (u32 i = tid(); i < 1024u; i += NT) sm.ulut[i] = TR8_G2C(ie_rq_iq2s_grid[i]);
  lbarrier();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 r = r0 + l;
  const u32 mb = nb >> 3u;
  const G u8 *rowp = q + (unsigned long)(r < rows ? r : r0) * mb * 82u;
  float a[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[j][t] = 0.0f;

  const u32 qlo_o = 2u + 4u * w, sgn_o = 34u + 4u * w, qh_o = 66u + w, sc_o = 74u + w;
  const u32 nb32 = nb & ~31u;
  if (nb32 > 0u) {
    const G u8 *bp = rowp;
    u32 d_c = ie_rq_a16(bp, 0u);
    u32 qlo_c = ie_rq_a32(bp, qlo_o), sgn_c = ie_rq_a32(bp, sgn_o);
    u32 qh_c = (u32)bp[qh_o], sc_c = (u32)bp[sc_o];
    __asm__ volatile("" : "+v"(d_c), "+v"(qlo_c), "+v"(sgn_c), "+v"(qh_c), "+v"(sc_c));
    for (u32 k0 = 0; k0 < nb32; k0 += 32u) {
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
        const u32 b = k0 + w + 8u * j;
        const int has_nxt = (j < 3u) || (k0 + 32u < nb32);
        bp += 82u;
        u32 d_n = 0u, qlo_n = 0u, sgn_n = 0u, qh_n = 0u, sc_n = 0u;
        if (has_nxt) {
          d_n = ie_rq_a16(bp, 0u);
          qlo_n = ie_rq_a32(bp, qlo_o);
          sgn_n = ie_rq_a32(bp, sgn_o);
          qh_n = (u32)bp[qh_o];
          sc_n = (u32)bp[sc_o];
        }
        __builtin_amdgcn_sched_barrier(0);
        u32 wl[4], wh[4]; float dw0, dw1;
        r9_v_decode_iq2s(sm.ulut, d_c, qlo_c, sgn_c, qh_c, sc_c, wl, wh, &dw0, &dw1);
        r9_v_tr9_dot8_sload_split16(wl, wh, dw0, dw1, xq, xs, 32u * b, 32u * nb + 4u * b, a[j]);
        __builtin_amdgcn_sched_barrier(0);
        d_c = d_n; qlo_c = qlo_n; sgn_c = sgn_n; qh_c = qh_n; sc_c = sc_n;
      }
    }
  }
  if (nb32 < nb) {
    for (u32 j = 0; j < 4u; j++) {
      const u32 b = nb32 + w + 8u * j;
      if (b < nb) {
        const G u8 *blk = rowp + (unsigned long)(b >> 3u) * 82u;
        u32 wl[4], wh[4]; float dw0, dw1;
        r9_v_decode_iq2s(sm.ulut, ie_rq_a16(blk, 0u), ie_rq_a32(blk, qlo_o), ie_rq_a32(blk, sgn_o),
                         (u32)blk[qh_o], (u32)blk[sc_o], wl, wh, &dw0, &dw1);
        r9_v_tr9_dot8_sload_split16(wl, wh, dw0, dw1, xq, xs, 32u * b, 32u * nb + 4u * b, a[j]);
      }
    }
  }
  lbarrier();
  r8_tr9_reduce32(sm.red, w, l, r0, rows, a, bias, res, y, ys, rs);
}

/* --- R9-V2: ie_saluki_iq2xs_tr9_pipe1 (T=8 verify IQ2_XS, tr9 1 row/lane) --- */
static inline __attribute__((always_inline)) void r9_v_decode_iq2xs(
    const LDSP u32 *ulut, const u32 d16, const u32 sc, const u32 q01, const u32 q23,
    u32 wl[4], u32 wh[4], float *dw0_out, float *dw1_out) {
#pragma clang fp contract(off)
  const float d4 = ie_rq_h2f(d16) * 0.25f;
  *dw0_out = d4 * (0.5f + (float)(sc & 15u));
  *dw1_out = d4 * (0.5f + (float)(sc >> 4u));
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 q2 = ((g < 2u ? q01 : q23) >> (16u * (g & 1u))) & 0xFFFFu;
    const u32 cp = ulut[q2 & 511u];
    TR8_KSW(cp & 0x0F0F0F0Fu, (cp >> 4u) & 0x0F0F0F0Fu, q2 >> 9u, wl[g], wh[g]);
  }
}

KERNEL ie_saluki_iq2xs_tr9_pipe1(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                                 const G float *bias, const G float *res, const G float *nw, G float *ny,
                                 G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  const u32 r0 = 32u * wgid();
  if (T != 8u || r0 >= rows) return;
  static LDS union {
    u32 ulut[512];
    float red[8u * 8u * 32u];
  } sm __attribute__((aligned(16)));
  sm.ulut[tid()] = r8_pack_iq2_code(ie_rq_iq2xs_grid[tid()]);
  sm.ulut[tid() + 256u] = r8_pack_iq2_code(ie_rq_iq2xs_grid[tid() + 256u]);
  lbarrier();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 r = r0 + l;
  const u32 mb = nb >> 3u;
  const G u8 *rowp = q + (unsigned long)(r < rows ? r : r0) * mb * 74u;
  float a[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[j][t] = 0.0f;

  const u32 qo = 2u + 8u * w, sc_o = 66u + w;
  const u32 nb32 = nb & ~31u;
  if (nb32 > 0u) {
    const G u8 *bp = rowp;
    u32 d_c = ie_rq_a16(bp, 0u);
    u32 sc_c = (u32)bp[sc_o];
    u32 q01_c = ie_rq_a32(bp, qo), q23_c = ie_rq_a32(bp, qo + 4u);
    __asm__ volatile("" : "+v"(d_c), "+v"(sc_c), "+v"(q01_c), "+v"(q23_c));
    for (u32 k0 = 0; k0 < nb32; k0 += 32u) {
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
        const u32 b = k0 + w + 8u * j;
        const int has_nxt = (j < 3u) || (k0 + 32u < nb32);
        bp += 74u;
        u32 d_n = 0u, sc_n = 0u, q01_n = 0u, q23_n = 0u;
        if (has_nxt) {
          d_n = ie_rq_a16(bp, 0u);
          sc_n = (u32)bp[sc_o];
          q01_n = ie_rq_a32(bp, qo);
          q23_n = ie_rq_a32(bp, qo + 4u);
        }
        __builtin_amdgcn_sched_barrier(0);
        u32 wl[4], wh[4]; float dw0, dw1;
        r9_v_decode_iq2xs(sm.ulut, d_c, sc_c, q01_c, q23_c, wl, wh, &dw0, &dw1);
        r9_v_tr9_dot8_sload_split16(wl, wh, dw0, dw1, xq, xs, 32u * b, 32u * nb + 4u * b, a[j]);
        __builtin_amdgcn_sched_barrier(0);
        d_c = d_n; sc_c = sc_n; q01_c = q01_n; q23_c = q23_n;
      }
    }
  }
  if (nb32 < nb) {
    for (u32 j = 0; j < 4u; j++) {
      const u32 b = nb32 + w + 8u * j;
      if (b < nb) {
        const G u8 *blk = rowp + (unsigned long)(b >> 3u) * 74u;
        u32 wl[4], wh[4]; float dw0, dw1;
        r9_v_decode_iq2xs(sm.ulut, ie_rq_a16(blk, 0u), (u32)blk[sc_o],
                          ie_rq_a32(blk, qo), ie_rq_a32(blk, qo + 4u), wl, wh, &dw0, &dw1);
        r9_v_tr9_dot8_sload_split16(wl, wh, dw0, dw1, xq, xs, 32u * b, 32u * nb + 4u * b, a[j]);
      }
    }
  }
  lbarrier();
  r8_tr9_reduce32(sm.red, w, l, r0, rows, a, bias, res, y, ys, rs);
}

/* --- R9-V3: ie_saluki_iq3xxs_tr9_pipe1 (T=8 verify IQ3_XXS, tr9 1 row/lane) --- */
static inline __attribute__((always_inline)) void r9_v_decode_iq3xxs(
    const LDSP u32 *ulut, const u32 d16, const u32 aux, const u32 q01, const u32 q23,
    u32 wl[4], u32 wh[4], float *dw_out) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d16);
  *dw_out = (d * 0.5f) * (0.5f + (float)(aux >> 28u));
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 qp = ((g < 2u ? q01 : q23) >> (16u * (g & 1u))) & 0xFFFFu;
    const u32 g1 = ulut[qp & 255u], g2 = ulut[qp >> 8u];
    const u32 s7 = (aux >> (7u * g)) & 127u;
    const u32 s8 = s7 | (((u32)__builtin_popcount(s7) & 1u) << 7u);
    const ie_rq_u64 m = ie_rq_mask8(s8);
    wl[g] = ie_rq_ssign(g1, (u32)m);
    wh[g] = ie_rq_ssign(g2, (u32)(m >> 32u));
  }
}

KERNEL ie_saluki_iq3xxs_tr9_pipe1(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                                  const G float *bias, const G float *res, const G float *nw, G float *ny,
                                  G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  const u32 r0 = 32u * wgid();
  if (T != 8u || r0 >= rows) return;
  static LDS union {
    u32 ulut[256];
    float red[8u * 8u * 32u];
  } sm __attribute__((aligned(16)));
  sm.ulut[tid()] = ie_rq_iq3xxs_grid[tid()];
  lbarrier();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 r = r0 + l;
  const u32 mb = nb >> 3u;
  const G u8 *rowp = q + (unsigned long)(r < rows ? r : r0) * mb * 98u;
  float a[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[j][t] = 0.0f;

  const u32 qo = 2u + 8u * w, auxo = 66u + 4u * w;
  const u32 nb32 = nb & ~31u;
  if (nb32 > 0u) {
    const G u8 *bp = rowp;
    u32 d_c = ie_rq_a16(bp, 0u);
    u32 aux_c = ie_rq_a32(bp, auxo);
    u32 q01_c = ie_rq_a32(bp, qo), q23_c = ie_rq_a32(bp, qo + 4u);
    __asm__ volatile("" : "+v"(d_c), "+v"(aux_c), "+v"(q01_c), "+v"(q23_c));
    for (u32 k0 = 0; k0 < nb32; k0 += 32u) {
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
        const u32 b = k0 + w + 8u * j;
        const int has_nxt = (j < 3u) || (k0 + 32u < nb32);
        bp += 98u;
        u32 d_n = 0u, aux_n = 0u, q01_n = 0u, q23_n = 0u;
        if (has_nxt) {
          d_n = ie_rq_a16(bp, 0u);
          aux_n = ie_rq_a32(bp, auxo);
          q01_n = ie_rq_a32(bp, qo);
          q23_n = ie_rq_a32(bp, qo + 4u);
        }
        __builtin_amdgcn_sched_barrier(0);
        u32 wl[4], wh[4]; float dw;
        r9_v_decode_iq3xxs(sm.ulut, d_c, aux_c, q01_c, q23_c, wl, wh, &dw);
        r8_tr9_dot8_sload(wl, wh, dw, xq, xs, 32u * b, 32u * nb + 4u * b, a[j]);
        __builtin_amdgcn_sched_barrier(0);
        d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
      }
    }
  }
  if (nb32 < nb) {
    for (u32 j = 0; j < 4u; j++) {
      const u32 b = nb32 + w + 8u * j;
      if (b < nb) {
        const G u8 *blk = rowp + (unsigned long)(b >> 3u) * 98u;
        u32 wl[4], wh[4]; float dw;
        r9_v_decode_iq3xxs(sm.ulut, ie_rq_a16(blk, 0u), ie_rq_a32(blk, auxo),
                           ie_rq_a32(blk, qo), ie_rq_a32(blk, qo + 4u), wl, wh, &dw);
        r8_tr9_dot8_sload(wl, wh, dw, xq, xs, 32u * b, 32u * nb + 4u * b, a[j]);
      }
    }
  }
  lbarrier();
  r8_tr9_reduce32(sm.red, w, l, r0, rows, a, bias, res, y, ys, rs);
}

/* --- R9-V4: ie_saluki_iq1s_tr8_pipe1 (T=8 verify IQ1_S, tr8 4 rows/wave) --- */
static inline __attribute__((always_inline)) void r9_v_iq1s_row_dot8(
    const LDSP ie_rq_u64 *ulut, const u32 d16, const u32 qlo, const u32 qh,
    const u32x4 x0v[8], const u32x4 x1v[8], const float dav_v[8], const float asf_v[8],
    float ak[8]) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d16);
  const float dw = d * (float)(int)(2u * ((qh >> 12u) & 7u) + 1u);
  const float del = (qh & 0x8000u) != 0u ? -0.125f : 0.125f;
  u32 wl[4], wh[4];
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 idx = ie_byte(qlo, g) | (((qh >> (3u * g)) & 7u) << 8u);
    const ie_rq_u64 gv = ulut[idx];
    wl[g] = (u32)gv;
    wh[g] = (u32)(gv >> 32u);
  }
#pragma unroll
  for (u32 t = 0; t < 8u; t++) {
    u32 idot = ie_dot4_ss(0u, wl[0], x0v[t].x);
    idot = ie_dot4_ss(idot, wh[0], x0v[t].y);
    idot = ie_dot4_ss(idot, wl[1], x0v[t].z);
    idot = ie_dot4_ss(idot, wh[1], x0v[t].w);
    idot = ie_dot4_ss(idot, wl[2], x1v[t].x);
    idot = ie_dot4_ss(idot, wh[2], x1v[t].y);
    idot = ie_dot4_ss(idot, wl[3], x1v[t].z);
    idot = ie_dot4_ss(idot, wh[3], x1v[t].w);
    ak[t] += dav_v[t] * (dw * ((float)(int)idot + del * asf_v[t]));
    if ((t & 3u) == 3u) __asm__ volatile("" : "+v"(ak[t - 3u]), "+v"(ak[t - 2u]), "+v"(ak[t - 1u]), "+v"(ak[t]));
  }
}

KERNEL ie_saluki_iq1s_tr8_pipe1(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                                const G float *bias, const G float *res, const G float *nw, G float *ny,
                                G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  if (T != 8u) return;
  static LDS ie_rq_u64 ulut[2048];
  for (u32 i = tid(); i < 2048u; i += NT) ulut[i] = ie_rq_iq1s_grid[i];
  lbarrier();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 r0 = ie_gemv_row(wgid(), wv) * 4u, l = lane();
  if (r0 >= rows) return;
  const u32 mb = nb >> 3u;
  const G u8 *row[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) row[k] = tr8_uniform(q + (unsigned long)(r0 + k < rows ? r0 + k : r0) * mb * 50u);
  const G u8 *xt[8];
#pragma unroll
  for (u32 t = 0; t < 8u; t++) xt[t] = tr8_uniform(xq + (unsigned long)t * xs);
  float a[4][8];
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[k][t] = 0.0f;

  const u32 s = l & 7u;
  u32 off_d = (l >> 3u) * 50u;
  u32 off_qlo = off_d + 2u + 4u * s;
  u32 off_qh = off_d + 34u + 2u * s;
  u32 xo = 32u * l, dao = 32u * nb + 4u * l, aso = 36u * nb + 4u * l;
  const u32 nb32 = nb & ~31u;
  if (nb32 > 0u) {
    u32 d_c = ie_rq_a16(row[0] + off_d, 0u);
    u32 qlo_c = ie_rq_a32(row[0] + off_qlo, 0u);
    u32 qh_c = ie_rq_a16(row[0] + off_qh, 0u);
    __asm__ volatile("" : "+v"(d_c), "+v"(qlo_c), "+v"(qh_c));
    for (u32 b0 = 0; b0 < nb32; b0 += 32u) {
      u32x4 x0v[8], x1v[8];
      float dav_v[8], asf_v[8];
      u32 as_v[8];
#pragma unroll
      for (u32 t = 0; t < 8u; t++) {
        const G u8 *xb = xt[t];
        dav_v[t] = *(const G float *)(xb + dao);
        as_v[t] = *(const G u32 *)(xb + aso);
        x0v[t] = *(const G u32x4 *)(xb + xo);
        x1v[t] = *(const G u32x4 *)(xb + xo + 16u);
      }
#pragma unroll
      for (u32 t = 0; t < 8u; t++) asf_v[t] = (float)(int)as_v[t];
      __builtin_amdgcn_sched_barrier(0);
      const int has_nxt_b = (b0 + 32u < nb32);
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        u32 d_n = 0u, qlo_n = 0u, qh_n = 0u;
        if (k < 3u) {
          d_n = ie_rq_a16(row[k + 1u] + off_d, 0u);
          qlo_n = ie_rq_a32(row[k + 1u] + off_qlo, 0u);
          qh_n = ie_rq_a16(row[k + 1u] + off_qh, 0u);
        } else if (has_nxt_b) {
          d_n = ie_rq_a16(row[0] + off_d, 200u);
          qlo_n = ie_rq_a32(row[0] + off_qlo, 200u);
          qh_n = ie_rq_a16(row[0] + off_qh, 200u);
        }
        __builtin_amdgcn_sched_barrier(0);
        r9_v_iq1s_row_dot8(ulut, d_c, qlo_c, qh_c, x0v, x1v, dav_v, asf_v, a[k]);
        __builtin_amdgcn_sched_barrier(0);
        d_c = d_n; qlo_c = qlo_n; qh_c = qh_n;
      }
      off_d += 200u; off_qlo += 200u; off_qh += 200u;
      xo += 1024u; dao += 128u; aso += 128u;
    }
  }
  if (nb32 < nb) {
    if (nb32 + l < nb) {
      for (u32 k = 0; k < 4u; k++) {
        const u32 d_c = ie_rq_a16(row[k] + off_d, 0u);
        const u32 qlo_c = ie_rq_a32(row[k] + off_qlo, 0u);
        const u32 qh_c = ie_rq_a16(row[k] + off_qh, 0u);
        const float d = ie_rq_h2f(d_c);
        const float dw = d * (float)(int)(2u * ((qh_c >> 12u) & 7u) + 1u);
        const float del = (qh_c & 0x8000u) != 0u ? -0.125f : 0.125f;
        u32 wl[4], wh[4];
#pragma unroll
        for (u32 g = 0; g < 4u; g++) {
          const u32 idx = ie_byte(qlo_c, g) | (((qh_c >> (3u * g)) & 7u) << 8u);
          const ie_rq_u64 gv = ulut[idx];
          wl[g] = (u32)gv;
          wh[g] = (u32)(gv >> 32u);
        }
        for (u32 t = 0; t < 8u; t++) {
          const G u8 *xb = xt[t];
          const float dav = *(const G float *)(xb + dao);
          const u32 as = *(const G u32 *)(xb + aso);
          const u32x4 x0 = *(const G u32x4 *)(xb + xo);
          const u32x4 x1 = *(const G u32x4 *)(xb + xo + 16u);
          u32 idot = ie_dot4_ss(0u, wl[0], x0.x);
          idot = ie_dot4_ss(idot, wh[0], x0.y);
          idot = ie_dot4_ss(idot, wl[1], x0.z);
          idot = ie_dot4_ss(idot, wh[1], x0.w);
          idot = ie_dot4_ss(idot, wl[2], x1.x);
          idot = ie_dot4_ss(idot, wh[2], x1.y);
          idot = ie_dot4_ss(idot, wl[3], x1.z);
          idot = ie_dot4_ss(idot, wh[3], x1.w);
          a[k][t] += dav * (dw * ((float)(int)idot + del * (float)(int)as));
        }
      }
    }
  }
  r9_v_tr8_epilogue(4u, r0, rows, l, a, bias, res, y, ys, rs);
}

/* --- R9-V5: ie_saluki_q2k_tr8_r4_pipe1 (T=8 verify Q2_K, tr8 4 rows/wave) --- */
static inline __attribute__((always_inline)) void r9_v_q2k_row_dot8(
    const u32 d_mn, const u32 sc16, const u32x4 q0_raw, const u32x4 q1_raw,
    const u32 sh, const u32 qmask,
    const u32x4 x0v[8], const u32x4 x1v[8], const float dav_v[8],
    const float xsf0[8], const float xsf1[8], float ak[8]) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d_mn & 0xFFFFu);
  const float mn = ie_rq_h2f(d_mn >> 16u);
  const float dw0 = d * (float)(sc16 & 15u), dm0 = mn * (float)((sc16 >> 4u) & 15u);
  const float dw1 = d * (float)((sc16 >> 8u) & 15u), dm1 = mn * (float)(sc16 >> 12u);
  const u32 q00 = q0_raw.x & qmask, q01 = q0_raw.y & qmask;
  const u32 q02 = q0_raw.z & qmask, q03 = q0_raw.w & qmask;
  const u32 q10 = q1_raw.x & qmask, q11 = q1_raw.y & qmask;
  const u32 q12 = q1_raw.z & qmask, q13 = q1_raw.w & qmask;
#pragma unroll
  for (u32 t = 0; t < 8u; t++) {
    u32 id0 = ie_dot4_us(0u, q00, x0v[t].x);
    id0 = ie_dot4_us(id0, q01, x0v[t].y);
    id0 = ie_dot4_us(id0, q02, x0v[t].z);
    id0 = ie_dot4_us(id0, q03, x0v[t].w);
    const float t0 = dav_v[t] * (dw0 * (float)((int)id0 >> sh) - dm0 * xsf0[t]);
    u32 id1 = ie_dot4_us(0u, q10, x1v[t].x);
    id1 = ie_dot4_us(id1, q11, x1v[t].y);
    id1 = ie_dot4_us(id1, q12, x1v[t].z);
    id1 = ie_dot4_us(id1, q13, x1v[t].w);
    const float t1 = dav_v[t] * (dw1 * (float)((int)id1 >> sh) - dm1 * xsf1[t]);
    ak[t] += (t0 + t1);
    if ((t & 3u) == 3u) __asm__ volatile("" : "+v"(ak[t - 3u]), "+v"(ak[t - 2u]), "+v"(ak[t - 1u]), "+v"(ak[t]));
  }
}

KERNEL ie_saluki_q2k_tr8_r4_pipe1(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                                  const G float *bias, const G float *res, const G float *nw, G float *ny,
                                  G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  if (T != 8u) return;
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 r0 = ie_gemv_row(wgid(), wv) * 4u, l = lane();
  if (r0 >= rows) return;
  const u32 mb = nb >> 3u;
  const G u8 *row[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) row[k] = tr8_uniform(q + (unsigned long)(r0 + k < rows ? r0 + k : r0) * mb * 84u);
  const G u8 *xt[8];
#pragma unroll
  for (u32 t = 0; t < 8u; t++) xt[t] = tr8_uniform(xq + (unsigned long)t * xs);
  float a[4][8];
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[k][t] = 0.0f;

  const u32 s = l & 7u;
  const u32 sh = 2u * (s & 3u), qmask = 0x03030303u << sh;
  u32 off_d = (l >> 3u) * 84u;
  u32 off_ps = off_d + 2u * s;
  u32 off_pq = off_d + 16u + (s >> 2u) * 32u;
  u32 xo = 32u * l, dao = 32u * nb + 4u * l;
  const u32 nb32 = nb & ~31u;
  if (nb32 > 0u) {
    u32 d_mn_c = ie_rq_a32(row[0] + off_d, 80u);
    u32 sc16_c = ie_rq_a16(row[0] + off_ps, 0u);
    u32x4 q0_c = r9_v_ld_u32x4_a4(row[0] + off_pq, 0u);
    u32x4 q1_c = r9_v_ld_u32x4_a4(row[0] + off_pq, 16u);
    __asm__ volatile("" : "+v"(d_mn_c), "+v"(sc16_c), "+v"(q0_c), "+v"(q1_c));
    for (u32 b0 = 0; b0 < nb32; b0 += 32u) {
      u32x4 x0v[8], x1v[8];
      float dav_v[8], xsf0[8], xsf1[8];
#pragma unroll
      for (u32 t = 0; t < 8u; t++) {
        const G u8 *xb = xt[t];
        dav_v[t] = *(const G float *)(xb + dao);
        x0v[t] = *(const G u32x4 *)(xb + xo);
        x1v[t] = *(const G u32x4 *)(xb + xo + 16u);
      }
#pragma unroll
      for (u32 t = 0; t < 8u; t++) {
        const u32 xs0 = ie_dot4_ss(ie_dot4_ss(0u, x0v[t].x, 0x01010101u), x0v[t].y, 0x01010101u) +
                        ie_dot4_ss(ie_dot4_ss(0u, x0v[t].z, 0x01010101u), x0v[t].w, 0x01010101u);
        const u32 xs1 = ie_dot4_ss(ie_dot4_ss(0u, x1v[t].x, 0x01010101u), x1v[t].y, 0x01010101u) +
                        ie_dot4_ss(ie_dot4_ss(0u, x1v[t].z, 0x01010101u), x1v[t].w, 0x01010101u);
        xsf0[t] = (float)(int)xs0;
        xsf1[t] = (float)(int)xs1;
      }
      __builtin_amdgcn_sched_barrier(0);
      const int has_nxt_b = (b0 + 32u < nb32);
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        u32 d_mn_n = 0u, sc16_n = 0u;
        u32x4 q0_n = (u32x4)(0u), q1_n = (u32x4)(0u);
        if (k < 3u) {
          d_mn_n = ie_rq_a32(row[k + 1u] + off_d, 80u);
          sc16_n = ie_rq_a16(row[k + 1u] + off_ps, 0u);
          q0_n = r9_v_ld_u32x4_a4(row[k + 1u] + off_pq, 0u);
          q1_n = r9_v_ld_u32x4_a4(row[k + 1u] + off_pq, 16u);
        } else if (has_nxt_b) {
          d_mn_n = ie_rq_a32(row[0] + off_d, 416u);
          sc16_n = ie_rq_a16(row[0] + off_ps, 336u);
          q0_n = r9_v_ld_u32x4_a4(row[0] + off_pq, 336u);
          q1_n = r9_v_ld_u32x4_a4(row[0] + off_pq, 352u);
        }
        __builtin_amdgcn_sched_barrier(0);
        r9_v_q2k_row_dot8(d_mn_c, sc16_c, q0_c, q1_c, sh, qmask, x0v, x1v, dav_v, xsf0, xsf1, a[k]);
        __builtin_amdgcn_sched_barrier(0);
        d_mn_c = d_mn_n; sc16_c = sc16_n; q0_c = q0_n; q1_c = q1_n;
      }
      off_d += 336u; off_ps += 336u; off_pq += 336u;
      xo += 1024u; dao += 128u;
    }
  }
  if (nb32 < nb) {
    if (nb32 + l < nb) {
      for (u32 k = 0; k < 4u; k++) {
        const u32 d_mn_c = ie_rq_a32(row[k] + off_d, 80u);
        const u32 sc16_c = ie_rq_a16(row[k] + off_ps, 0u);
        const u32x4 q0_c = r9_v_ld_u32x4_a4(row[k] + off_pq, 0u);
        const u32x4 q1_c = r9_v_ld_u32x4_a4(row[k] + off_pq, 16u);
        const float d = ie_rq_h2f(d_mn_c & 0xFFFFu), mn = ie_rq_h2f(d_mn_c >> 16u);
        const float dw0 = d * (float)(sc16_c & 15u), dm0 = mn * (float)((sc16_c >> 4u) & 15u);
        const float dw1 = d * (float)((sc16_c >> 8u) & 15u), dm1 = mn * (float)(sc16_c >> 12u);
        const u32 q00 = q0_c.x & qmask, q01 = q0_c.y & qmask;
        const u32 q02 = q0_c.z & qmask, q03 = q0_c.w & qmask;
        const u32 q10 = q1_c.x & qmask, q11 = q1_c.y & qmask;
        const u32 q12 = q1_c.z & qmask, q13 = q1_c.w & qmask;
        for (u32 t = 0; t < 8u; t++) {
          const G u8 *xb = xt[t];
          const float dav = *(const G float *)(xb + dao);
          const u32x4 x0 = *(const G u32x4 *)(xb + xo);
          const u32x4 x1 = *(const G u32x4 *)(xb + xo + 16u);
          const u32 xs0 = ie_dot4_ss(ie_dot4_ss(0u, x0.x, 0x01010101u), x0.y, 0x01010101u) +
                          ie_dot4_ss(ie_dot4_ss(0u, x0.z, 0x01010101u), x0.w, 0x01010101u);
          const u32 xs1 = ie_dot4_ss(ie_dot4_ss(0u, x1.x, 0x01010101u), x1.y, 0x01010101u) +
                          ie_dot4_ss(ie_dot4_ss(0u, x1.z, 0x01010101u), x1.w, 0x01010101u);
          u32 id0 = ie_dot4_us(0u, q00, x0.x);
          id0 = ie_dot4_us(id0, q01, x0.y);
          id0 = ie_dot4_us(id0, q02, x0.z);
          id0 = ie_dot4_us(id0, q03, x0.w);
          const float t0 = dav * (dw0 * (float)((int)id0 >> sh) - dm0 * (float)(int)xs0);
          u32 id1 = ie_dot4_us(0u, q10, x1.x);
          id1 = ie_dot4_us(id1, q11, x1.y);
          id1 = ie_dot4_us(id1, q12, x1.z);
          id1 = ie_dot4_us(id1, q13, x1.w);
          const float t1 = dav * (dw1 * (float)((int)id1 >> sh) - dm1 * (float)(int)xs1);
          a[k][t] += (t0 + t1);
        }
      }
    }
  }
  r9_v_tr8_epilogue(4u, r0, rows, l, a, bias, res, y, ys, rs);
}

/* --- R9-V6: ie_saluki_iq1m_tr8_r4_pipe1 (T=8 verify IQ1_M, tr8 4 rows/wave) --- */
static inline __attribute__((always_inline)) void r9_v_iq1m_row_dot8(
    const LDSP ie_rq_u64 *ulut, const u32 slo, const u32 shi, const u32 qhb, const u32 qlo,
    const u32 s, const u32 sc_sh16, const u32 sh,
    const u32x4 x0v[8], const u32x4 x1v[8], const float dav_v[8], float ak[8]) {
#pragma clang fp contract(off)
  const u32 s0 = slo & 0xFFFFu, s1 = slo >> 16u, s2 = shi & 0xFFFFu, s3 = shi >> 16u;
  const float d8 = ie_rq_h2f((s0 >> 12u) | ((s1 >> 8u) & 0x00F0u) | ((s2 >> 4u) & 0x0F00u) | (s3 & 0xF000u)) * 0.125f;
  const u32 scv = ((s & 4u) != 0u ? shi : slo) >> sc_sh16;
  const float dq0 = d8 * (float)(int)(2u * ((scv >> sh) & 7u) + 1u);
  const float dq1 = d8 * (float)(int)(2u * ((scv >> (sh + 3u)) & 7u) + 1u);
  u32 wl[4], wh[4];
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 qh = g < 2u ? (qhb & 255u) : (qhb >> 8u);
    const u32 idx = ie_byte(qlo, g) | ((qh << (8u - 4u * (g & 1u))) & 0x700u);
    const ie_rq_u64 gv = ulut[idx];
    const u32 tbl = (qh & ((g & 1u) != 0u ? 0x80u : 0x08u)) != 0u ? 0xF70007FFu : 0xF9000901u;
    wl[g] = __builtin_amdgcn_perm(0u, tbl, (u32)gv & 0x03030303u);
    wh[g] = __builtin_amdgcn_perm(0u, tbl, (u32)(gv >> 32u) & 0x03030303u);
  }
#pragma unroll
  for (u32 t = 0; t < 8u; t++) {
    u32 c0 = ie_dot4_ss(0u, wl[0], x0v[t].x);
    c0 = ie_dot4_ss(c0, wh[0], x0v[t].y);
    c0 = ie_dot4_ss(c0, wl[1], x0v[t].z);
    c0 = ie_dot4_ss(c0, wh[1], x0v[t].w);
    u32 c1 = ie_dot4_ss(0u, wl[2], x1v[t].x);
    c1 = ie_dot4_ss(c1, wh[2], x1v[t].y);
    c1 = ie_dot4_ss(c1, wl[3], x1v[t].z);
    c1 = ie_dot4_ss(c1, wh[3], x1v[t].w);
    ak[t] += dav_v[t] * (dq0 * (float)(int)c0 + dq1 * (float)(int)c1);
    if ((t & 3u) == 3u) __asm__ volatile("" : "+v"(ak[t - 3u]), "+v"(ak[t - 2u]), "+v"(ak[t - 1u]), "+v"(ak[t]));
  }
}

KERNEL ie_saluki_iq1m_tr8_r4_pipe1(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                                   const G float *bias, const G float *res, const G float *nw, G float *ny,
                                   G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  if (T != 8u) return;
  static LDS ie_rq_u64 ulut[2048];
  for (u32 i = tid(); i < 2048u; i += NT) ulut[i] = ie_rq_iq1s_grid[i];
  lbarrier();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 r0 = ie_gemv_row(wgid(), wv) * 4u, l = lane();
  if (r0 >= rows) return;
  const u32 mb = nb >> 3u;
  const G u8 *row[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) row[k] = tr8_uniform(q + (unsigned long)(r0 + k < rows ? r0 + k : r0) * mb * 56u);
  const G u8 *xt[8];
#pragma unroll
  for (u32 t = 0; t < 8u; t++) xt[t] = tr8_uniform(xq + (unsigned long)t * xs);
  float a[4][8];
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[k][t] = 0.0f;

  const u32 s = l & 7u, sc_sh16 = 16u * ((s >> 1u) & 1u), sh = 6u * (s & 1u);
  u32 off_d = (l >> 3u) * 56u;
  u32 off_qhb = off_d + 32u + 2u * s;
  u32 off_qlo = off_d + 4u * s;
  u32 xo = 32u * l, dao = 32u * nb + 4u * l;
  const u32 nb32 = nb & ~31u;
  if (nb32 > 0u) {
    u32 slo_c = ie_rq_a32(row[0] + off_d, 48u);
    u32 shi_c = ie_rq_a32(row[0] + off_d, 52u);
    u32 qhb_c = ie_rq_a16(row[0] + off_qhb, 0u);
    u32 qlo_c = ie_rq_a32(row[0] + off_qlo, 0u);
    __asm__ volatile("" : "+v"(slo_c), "+v"(shi_c), "+v"(qhb_c), "+v"(qlo_c));
    for (u32 b0 = 0; b0 < nb32; b0 += 32u) {
      u32x4 x0v[8], x1v[8];
      float dav_v[8];
#pragma unroll
      for (u32 t = 0; t < 8u; t++) {
        const G u8 *xb = xt[t];
        dav_v[t] = *(const G float *)(xb + dao);
        x0v[t] = *(const G u32x4 *)(xb + xo);
        x1v[t] = *(const G u32x4 *)(xb + xo + 16u);
      }
      __builtin_amdgcn_sched_barrier(0);
      const int has_nxt_b = (b0 + 32u < nb32);
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        u32 slo_n = 0u, shi_n = 0u, qhb_n = 0u, qlo_n = 0u;
        if (k < 3u) {
          slo_n = ie_rq_a32(row[k + 1u] + off_d, 48u);
          shi_n = ie_rq_a32(row[k + 1u] + off_d, 52u);
          qhb_n = ie_rq_a16(row[k + 1u] + off_qhb, 0u);
          qlo_n = ie_rq_a32(row[k + 1u] + off_qlo, 0u);
        } else if (has_nxt_b) {
          slo_n = ie_rq_a32(row[0] + off_d, 272u);
          shi_n = ie_rq_a32(row[0] + off_d, 276u);
          qhb_n = ie_rq_a16(row[0] + off_qhb, 224u);
          qlo_n = ie_rq_a32(row[0] + off_qlo, 224u);
        }
        __builtin_amdgcn_sched_barrier(0);
        r9_v_iq1m_row_dot8(ulut, slo_c, shi_c, qhb_c, qlo_c, s, sc_sh16, sh, x0v, x1v, dav_v, a[k]);
        __builtin_amdgcn_sched_barrier(0);
        slo_c = slo_n; shi_c = shi_n; qhb_c = qhb_n; qlo_c = qlo_n;
      }
      off_d += 224u; off_qhb += 224u; off_qlo += 224u;
      xo += 1024u; dao += 128u;
    }
  }
  if (nb32 < nb) {
    if (nb32 + l < nb) {
      for (u32 k = 0; k < 4u; k++) {
        const u32 slo_c = ie_rq_a32(row[k] + off_d, 48u);
        const u32 shi_c = ie_rq_a32(row[k] + off_d, 52u);
        const u32 qhb_c = ie_rq_a16(row[k] + off_qhb, 0u);
        const u32 qlo_c = ie_rq_a32(row[k] + off_qlo, 0u);
        const u32 s0 = slo_c & 0xFFFFu, s1 = slo_c >> 16u, s2 = shi_c & 0xFFFFu, s3 = shi_c >> 16u;
        const float d8 = ie_rq_h2f((s0 >> 12u) | ((s1 >> 8u) & 0x00F0u) | ((s2 >> 4u) & 0x0F00u) | (s3 & 0xF000u)) * 0.125f;
        const u32 scv = ((s & 4u) != 0u ? shi_c : slo_c) >> sc_sh16;
        const float dq0 = d8 * (float)(int)(2u * ((scv >> sh) & 7u) + 1u);
        const float dq1 = d8 * (float)(int)(2u * ((scv >> (sh + 3u)) & 7u) + 1u);
        u32 wl[4], wh[4];
#pragma unroll
        for (u32 g = 0; g < 4u; g++) {
          const u32 qh = g < 2u ? (qhb_c & 255u) : (qhb_c >> 8u);
          const u32 idx = ie_byte(qlo_c, g) | ((qh << (8u - 4u * (g & 1u))) & 0x700u);
          const ie_rq_u64 gv = ulut[idx];
          const u32 tbl = (qh & ((g & 1u) != 0u ? 0x80u : 0x08u)) != 0u ? 0xF70007FFu : 0xF9000901u;
          wl[g] = __builtin_amdgcn_perm(0u, tbl, (u32)gv & 0x03030303u);
          wh[g] = __builtin_amdgcn_perm(0u, tbl, (u32)(gv >> 32u) & 0x03030303u);
        }
        for (u32 t = 0; t < 8u; t++) {
          const G u8 *xb = xt[t];
          const float dav = *(const G float *)(xb + dao);
          const u32x4 x0 = *(const G u32x4 *)(xb + xo);
          const u32x4 x1 = *(const G u32x4 *)(xb + xo + 16u);
          u32 c0 = ie_dot4_ss(0u, wl[0], x0.x);
          c0 = ie_dot4_ss(c0, wh[0], x0.y);
          c0 = ie_dot4_ss(c0, wl[1], x0.z);
          c0 = ie_dot4_ss(c0, wh[1], x0.w);
          u32 c1 = ie_dot4_ss(0u, wl[2], x1.x);
          c1 = ie_dot4_ss(c1, wh[2], x1.y);
          c1 = ie_dot4_ss(c1, wl[3], x1.z);
          c1 = ie_dot4_ss(c1, wh[3], x1.w);
          a[k][t] += dav * (dq0 * (float)(int)c0 + dq1 * (float)(int)c1);
        }
      }
    }
  }
  r9_v_tr8_epilogue(4u, r0, rows, l, a, bias, res, y, ys, rs);
}

/* --- R9-V7: ie_saluki_iq3s_tr8_r2_pipe1 (T=8 verify IQ3_S, tr8 2 rows/wave) --- */
static inline __attribute__((always_inline)) void r9_v_iq3s_load(
    const G u8 *rp, const u32 base, const u32 s,
    u32 *d_out, u32 *qh_out, u32 *sc_out, u32 *sgn_out, u32 *q01_out, u32 *q23_out) {
  *d_out = ie_rq_a16(rp + base, 0u);
  *qh_out = (u32)(rp + (base + s))[66u];
  *sc_out = (u32)(rp + (base + (s >> 1u)))[106u];
  *sgn_out = ie_rq_a32(rp + (base + 4u * s), 74u);
  const u32 bq = base + 8u * s;
  *q01_out = ie_rq_a32(rp + bq, 2u);
  *q23_out = ie_rq_a32(rp + bq, 6u);
}

/* All 512 ie_rq_iq3s_grid entries hold strictly odd bytes (1..15), so
 * -m = (~m + 1) = m ^ 0xFE is bit-exact for every entry and sign bit. */
static inline __attribute__((always_inline)) void r9_v_decode_iq3s(
    const LDSP u32 *ulut, const u32 d16, const u32 qh, const u32 sc_b, const u32 sgn,
    const u32 q01, const u32 q23, const u32 sc_sh,
    u32 wl[4], u32 wh[4], float *dw_out) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d16);
  const u32 sc = (sc_b >> sc_sh) & 15u;
  *dw_out = d * (float)(int)(1u + 2u * sc);
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 qp = ((g < 2u ? q01 : q23) >> (16u * (g & 1u))) & 0xFFFFu;
    const u32 i1 = (qp & 255u) | (((qh >> (2u * g)) & 1u) << 8u);
    const u32 i2 = (qp >> 8u) | (((qh >> (2u * g + 1u)) & 1u) << 8u);
    const u32 g1 = ulut[i1], g2 = ulut[i2];
    const u32 sb = ie_byte(sgn, g);
    const u32 t0 = ((sb & 15u) * 0x00204081u) & 0x01010101u;
    const u32 t1 = ((sb >> 4u) * 0x00204081u) & 0x01010101u;
    wl[g] = g1 ^ __builtin_amdgcn_perm(0u, 0x0000FE00u, t0);
    wh[g] = g2 ^ __builtin_amdgcn_perm(0u, 0x0000FE00u, t1);
  }
}

KERNEL ie_saluki_iq3s_tr8_r2_pipe1(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                                   const G float *bias, const G float *res, const G float *nw, G float *ny,
                                   G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  if (T != 8u) return;
  static LDS u32 ulut[512];
  ulut[tid()] = ie_rq_iq3s_grid[tid()];
  ulut[tid() + 256u] = ie_rq_iq3s_grid[tid() + 256u];
  lbarrier();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 r0 = ie_gemv_row(wgid(), wv) * 2u, l = lane();
  if (r0 >= rows) return;
  const u32 mb = nb >> 3u;
  const G u8 *row[2];
#pragma unroll
  for (u32 k = 0; k < 2u; k++) row[k] = tr8_uniform(q + (unsigned long)(r0 + k < rows ? r0 + k : r0) * mb * 110u);
  const G u8 *xt[8];
#pragma unroll
  for (u32 t = 0; t < 8u; t++) xt[t] = tr8_uniform(xq + (unsigned long)t * xs);
  float a[2][8];
#pragma unroll
  for (u32 k = 0; k < 2u; k++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[k][t] = 0.0f;

  const u32 s = l & 7u, sc_sh = 4u * (s & 1u);
  u32 off_d = (l >> 3u) * 110u;
  u32 xo = 32u * l, dao = 32u * nb + 4u * l;
  const u32 nb32 = nb & ~31u;
  if (nb32 > 0u) {
    u32 d_c = 0u, qh_c = 0u, sc_c = 0u, sgn_c = 0u, q01_c = 0u, q23_c = 0u;
    r9_v_iq3s_load(row[0], off_d, s, &d_c, &qh_c, &sc_c, &sgn_c, &q01_c, &q23_c);
    __asm__ volatile("" : "+v"(d_c), "+v"(qh_c), "+v"(sc_c), "+v"(sgn_c), "+v"(q01_c), "+v"(q23_c));
    for (u32 b0 = 0; b0 < nb32; b0 += 32u) {
      const int has_nxt_b = (b0 + 32u < nb32);
#pragma unroll
      for (u32 k = 0; k < 2u; k++) {
        u32 wl[4], wh[4]; float dw;
        r9_v_decode_iq3s(ulut, d_c, qh_c, sc_c, sgn_c, q01_c, q23_c, sc_sh, wl, wh, &dw);
#pragma unroll
        for (u32 t = 0; t < 5u; t++) {
          const G u8 *xb = xt[t];
          const float dav = *(const G float *)(xb + dao);
          const u32x4 x0 = *(const G u32x4 *)(xb + xo);
          const u32x4 x1 = *(const G u32x4 *)(xb + xo + 16u);
          u32 idot = ie_dot4_ss(0u, wl[0], x0.x);
          idot = ie_dot4_ss(idot, wh[0], x0.y);
          idot = ie_dot4_ss(idot, wl[1], x0.z);
          idot = ie_dot4_ss(idot, wh[1], x0.w);
          idot = ie_dot4_ss(idot, wl[2], x1.x);
          idot = ie_dot4_ss(idot, wh[2], x1.y);
          idot = ie_dot4_ss(idot, wl[3], x1.z);
          idot = ie_dot4_ss(idot, wh[3], x1.w);
          a[k][t] += dav * (dw * (float)(int)idot);
          if (t == 3u) __asm__ volatile("" : "+v"(a[k][0]), "+v"(a[k][1]), "+v"(a[k][2]), "+v"(a[k][3]));
        }
        /* Preload t=5..7 (27 VGPRs) BEFORE issuing w_nxt so loadcnt FIFO never stalls on w_nxt. */
        u32x4 x0v[3], x1v[3];
        float dav_v[3];
#pragma unroll
        for (u32 t = 0; t < 3u; t++) {
          const G u8 *xb = xt[5u + t];
          dav_v[t] = *(const G float *)(xb + dao);
          x0v[t] = *(const G u32x4 *)(xb + xo);
          x1v[t] = *(const G u32x4 *)(xb + xo + 16u);
        }
        __builtin_amdgcn_sched_barrier(0);
        u32 d_n = 0u, qh_n = 0u, sc_n = 0u, sgn_n = 0u, q01_n = 0u, q23_n = 0u;
        if (k == 0u) {
          r9_v_iq3s_load(row[1], off_d, s, &d_n, &qh_n, &sc_n, &sgn_n, &q01_n, &q23_n);
        } else if (has_nxt_b) {
          r9_v_iq3s_load(row[0], off_d + 440u, s, &d_n, &qh_n, &sc_n, &sgn_n, &q01_n, &q23_n);
        }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (u32 t = 0; t < 3u; t++) {
          u32 idot = ie_dot4_ss(0u, wl[0], x0v[t].x);
          idot = ie_dot4_ss(idot, wh[0], x0v[t].y);
          idot = ie_dot4_ss(idot, wl[1], x0v[t].z);
          idot = ie_dot4_ss(idot, wh[1], x0v[t].w);
          idot = ie_dot4_ss(idot, wl[2], x1v[t].x);
          idot = ie_dot4_ss(idot, wh[2], x1v[t].y);
          idot = ie_dot4_ss(idot, wl[3], x1v[t].z);
          idot = ie_dot4_ss(idot, wh[3], x1v[t].w);
          a[k][5u + t] += dav_v[t] * (dw * (float)(int)idot);
        }
        __builtin_amdgcn_sched_barrier(0);
        d_c = d_n; qh_c = qh_n; sc_c = sc_n; sgn_c = sgn_n; q01_c = q01_n; q23_c = q23_n;
      }
      off_d += 440u; xo += 1024u; dao += 128u;
    }
  }
  if (nb32 < nb) {
    if (nb32 + l < nb) {
      for (u32 k = 0; k < 2u; k++) {
        u32 d_c = 0u, qh_c = 0u, sc_c = 0u, sgn_c = 0u, q01_c = 0u, q23_c = 0u;
        r9_v_iq3s_load(row[k], off_d, s, &d_c, &qh_c, &sc_c, &sgn_c, &q01_c, &q23_c);
        u32 wl[4], wh[4]; float dw;
        r9_v_decode_iq3s(ulut, d_c, qh_c, sc_c, sgn_c, q01_c, q23_c, sc_sh, wl, wh, &dw);
        for (u32 t = 0; t < 8u; t++) {
          const G u8 *xb = xt[t];
          const float dav = *(const G float *)(xb + dao);
          const u32x4 x0 = *(const G u32x4 *)(xb + xo);
          const u32x4 x1 = *(const G u32x4 *)(xb + xo + 16u);
          u32 idot = ie_dot4_ss(0u, wl[0], x0.x);
          idot = ie_dot4_ss(idot, wh[0], x0.y);
          idot = ie_dot4_ss(idot, wl[1], x0.z);
          idot = ie_dot4_ss(idot, wh[1], x0.w);
          idot = ie_dot4_ss(idot, wl[2], x1.x);
          idot = ie_dot4_ss(idot, wh[2], x1.y);
          idot = ie_dot4_ss(idot, wl[3], x1.z);
          idot = ie_dot4_ss(idot, wh[3], x1.w);
          a[k][t] += dav * (dw * (float)(int)idot);
        }
      }
    }
  }
  r9_v_tr8_epilogue(2u, r0, rows, l, a, bias, res, y, ys, rs);
}
