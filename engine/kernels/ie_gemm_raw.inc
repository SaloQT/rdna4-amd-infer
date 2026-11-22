/* ie_gemm_raw.inc -- prompt chunks of raw GGUF IQ/K weight blocks (MAT_RAWQ)
 * on the matrix cores:
 *   - ie_gemm_rq_<type>   (RQ_R = 128 rows x RQ_T = 64 tokens/WG, exact per-32 Q8_0 scales, LDS LUT + VGPR prefetch)
 *   - ie_gemm_rq32_<type> (RQ_R = 128 rows x RQ32_T = 32 tokens/WG, short prefills T = 17..64, exact per-32 Q8_0 scales)
 *   - ie_gemm_rq_swiglu   (128 rows x 32 tokens/WG, fused dual-matrix gate + up + SwiGLU + optional Q8_0 oq, same or mixed subtypes)
 *   - ie_gemm_ri_<type>   (RQ_R = 128 rows x RI_T = 64 tokens/WG, per-256 IE_RAW_GEMM_I256=1 path)
 *
 * Included at the end of kernels/ie_kernels.c (not in the megakernel builds).
 *
 * Operand roles: TOKENS are the A matrix (rows), WEIGHT ROWS are the B matrix
 * (columns). Lane l (0..31, h = l >> 4) holds token l % 16 in A, weight row
 * l % 16 in B, and in D element v (0..7) = token 8*h + v of weight row l % 16.
 * Thus every lane's weight scales are lane-local scalars (zero LDS exchange for
 * weights or weight scales), while token scales dx are contiguous f8 vectors in LDS.
 *
 * Exactness: every int32 is an exact integer dot product with exact per-32 Q8_0
 * activation scales da[b]:
 *   - IQ1_S / IQ1_M: weight = d * s * (g +- 1/8) with g in {-1, 0, 1}; folded into
 *     int8 weight 8*g +- 1 (+-1, +-7, +-9) with scale d * s / 8.
 *   - Q2_K: weight = d * sc * q - dmin * mn; int8 weight is sc * q (0..45) with
 *     scale d, and dmin * mn * sum16(a) is subtracted using exact 16-activation sums.
 *   - All IQ/K grid & sign tables are staged in LDS (compact nibble-packed u32 codes
 *     for IQ1/IQ2 grids + u32 grids for IQ3 + ksigns64): zero global LUT gathers. */
#ifndef IE_MEGA

#ifndef LDSP
#define LDSP __attribute__((address_space(3)))
#endif

#ifndef IE_RQ_WMMA_COMMON
#define IE_RQ_WMMA_COMMON

#define RQ_KB 4u                  /* 32-weight sub-blocks per K chunk (half a 256-weight raw block) */
#define RQ_LS (RQ_KB * 32u + 16u) /* LDS bytes per token row (144 B: 16-byte aligned, bank-conflict padded) */

/* IQ2 magnitude code {0, 1, 2} + sign -> signed int8 {+-8, +-25, +-43} via v_perm_b32. */
#define RQ_P2(sel) __builtin_amdgcn_perm(0x00D5E7F8u, 0x002B1908u, (sel))
#define RQ_SGN_M(c, m) RQ_P2(((m) & 0x04040404u) | (c))
#define RQ_SGN_B(c, b) RQ_P2((((b) * 0x00810204u) & 0x04040404u) | (c))

/* IQ1 grid bytes (-1, 0, 1 as 0xff, 0x00, 0x01) -> int8 weights 8*g +- 1. */
#define RQ_IQ1_TBL_POS 0xF9000901u /* delta +1: 1, 9, -, -7 */
#define RQ_IQ1_TBL_NEG 0xF70007FFu /* delta -1: -1, 7, -, -9 */
static inline u32 rq_iq1_map(u32 g, u32 tbl) { return __builtin_amdgcn_perm(0u, tbl, g & 0x03030303u); }

/* 16-weight split scales (IQ2_XS, IQ2_S, IQ1_M): one scale per 16-weight WMMA half. */
static inline int rq_split16(u32 type) { return type == IE_RQ_IQ2_XS || type == IE_RQ_IQ2_S || type == IE_RQ_IQ1_M; }

static inline int rq_same_lut(u32 t0, u32 t1) {
  if (t0 == t1) return 1;
  if ((t0 == IE_RQ_IQ1_S || t0 == IE_RQ_IQ1_M) && (t1 == IE_RQ_IQ1_S || t1 == IE_RQ_IQ1_M)) return 1;
  return 0;
}

/* Pack 8 grid bytes into 1 u32 (low nibbles = weights 0..3, high nibbles = weights 4..7):
 *   IQ2 grids hold {0x08, 0x19, 0x2b}, whose high nibbles are already {0, 1, 2};
 *   IQ1 grids hold {0xff, 0x00, 0x01}, needing only bits 0..1 per byte. */
static inline __attribute__((always_inline)) u32 rq_pack_iq2(ie_rq_u64 g) {
  return (((u32)g >> 4u) & 0x03030303u) | ((u32)(g >> 32u) & 0x30303030u);
}
static inline __attribute__((always_inline)) u32 rq_pack_iq1(ie_rq_u64 g) {
  return ((u32)g & 0x03030303u) | ((((u32)(g >> 32u)) & 0x03030303u) << 4u);
}

typedef unsigned int rq_u32x2 __attribute__((ext_vector_type(2)));

static inline __attribute__((always_inline)) u32 rq_ksign7(const u32 s7) {
  return s7 | (((u32)__builtin_popcount(s7) & 1u) << 7u);
}

static inline __attribute__((always_inline)) u32 rq_al16(const u32 lo, const u32 hi) {
  return (lo >> 16u) | (hi << 16u);
}

/* Branchless 4-byte-aligned global pointers and funnel-shift alignment (buffers have >= 32 zeroed tail bytes):
 * rq_hload issues ONLY pure aligned global_load_b32/b64/b128 instructions with zero ALU consumers and zero branches,
 * so LLVM never inserts s_wait_loadcnt before the WMMA compute tile. */
static inline __attribute__((always_inline)) const G u32 *rq_ap(const G u8 *p, const u32 o) {
  return (const G u32 *)(((unsigned long)(p + o)) & ~3ul);
}

static inline __attribute__((always_inline)) u32 rq_align(const u32 lo, const u32 hi, const u32 sh) {
  return __builtin_amdgcn_alignbit(hi, lo, sh);
}

/* Stage subtype's LUT into LDS (no barrier inside so caller can overlap with chunk-0 prefetch; ksigns uses ALU rq_ksign7). */
static inline __attribute__((always_inline)) void rq_stage_lut_raw(const u32 type, LDSP ie_rq_u64 *ksigns, LDSP u32 *lut,
                                                                   const int do_ksigns) {
  (void)ksigns;
  (void)do_ksigns;
  if (!lut) return;
  const u32 t = tid();
  switch (type) {
    case IE_RQ_IQ2_XXS:
      lut[t] = rq_pack_iq2(ie_rq_iq2xxs_grid[t]);
      break;
    case IE_RQ_IQ2_XS:
      for (u32 i = t; i < 512u; i += NT) lut[i] = rq_pack_iq2(ie_rq_iq2xs_grid[i]);
      break;
    case IE_RQ_IQ2_S:
      for (u32 i = t; i < 1024u; i += NT) lut[i] = rq_pack_iq2(ie_rq_iq2s_grid[i]);
      break;
    case IE_RQ_IQ3_XXS:
      lut[t] = ie_rq_iq3xxs_grid[t];
      break;
    case IE_RQ_IQ3_S:
      for (u32 i = t; i < 512u; i += NT) lut[i] = ie_rq_iq3s_grid[i];
      break;
    case IE_RQ_IQ1_S:
    case IE_RQ_IQ1_M:
      for (u32 i = t; i < 2048u; i += NT) lut[i] = rq_pack_iq1(ie_rq_iq1s_grid[i]);
      break;
    default:
      break;
  }
}

/* Exact sum of 16 signed int8 activations (for the Q2_K min term). */
static inline float rq_sum16(u32x4 v) {
  u32 a = ie_dot4_ss(0u, v.x, 0x01010101u);
  a = ie_dot4_ss(a, v.y, 0x01010101u);
  a = ie_dot4_ss(a, v.z, 0x01010101u);
  a = ie_dot4_ss(a, v.w, 0x01010101u);
  return (float)(int)a;
}

struct __attribute__((packed, aligned(4))) rq_f32x4p {
  f32x4 v;
};
static inline f32x4 rq_load_f32x4(const G float *p) {
  return ((const __attribute__((address_space(1))) struct rq_f32x4p *)p)->v;
}

/* Half-block (4 sub-blocks = 128 weights) raw weight words prefetched asynchronously into VGPRs. */
typedef struct {
  u32 r[18];
  u32 sh;
} rq_hblk_t;

/* Decoded int8 weights (8h..8h+7 in w0, 16+8h..16+8h+7 in w1) and exact float scales of one 32-weight sub-block. */
typedef struct {
  v2i w0, w1;
  float s0, s1, m0, m1;
} rq_sub_t;

static inline __attribute__((always_inline)) rq_hblk_t rq_hload(const u32 type, const G u8 *p, const u32 hb, const u32 h) {
  rq_hblk_t x;
#pragma unroll
  for (u32 i = 0; i < 18u; i++) x.r[i] = 0u;
  x.sh = (u32)((unsigned long)p & 2ul) << 3u;
  switch (type) {
    case IE_RQ_Q2_K: {
      x.sh = 0u;
      x.r[0] = *(const G u32 *)(p + 80u);
      const rq_u32x2 sc = *(const G rq_u32x2 *)(p + 8u * hb);
      x.r[1] = sc.x, x.r[2] = sc.y;
      const u32 qs = 16u + 32u * hb + 8u * h;
      const rq_u32x2 q0 = *(const G rq_u32x2 *)(p + qs);
      const rq_u32x2 q1 = *(const G rq_u32x2 *)(p + qs + 16u);
      x.r[3] = q0.x, x.r[4] = q0.y, x.r[5] = q1.x, x.r[6] = q1.y;
      break;
    }
    case IE_RQ_IQ2_XXS: {
      x.r[0] = *rq_ap(p, 0u);
      const G u32 *qp = rq_ap(p, 2u + 32u * hb);
      const u32x4 v0 = *(const G u32x4 *)(qp + 0u), v1 = *(const G u32x4 *)(qp + 4u);
      x.r[1] = v0.x, x.r[2] = v0.y, x.r[3] = v0.z, x.r[4] = v0.w;
      x.r[5] = v1.x, x.r[6] = v1.y, x.r[7] = v1.z, x.r[8] = v1.w;
      x.r[9] = qp[8];
      break;
    }
    case IE_RQ_IQ2_XS: {
      x.r[0] = *rq_ap(p, 0u);
      const rq_u32x2 sc = *(const G rq_u32x2 *)rq_ap(p, 66u + 4u * hb);
      x.r[1] = sc.x, x.r[2] = sc.y;
      const G u32 *qp = rq_ap(p, 2u + 32u * hb);
      const u32x4 v0 = *(const G u32x4 *)(qp + 0u), v1 = *(const G u32x4 *)(qp + 4u);
      x.r[3] = v0.x, x.r[4] = v0.y, x.r[5] = v0.z, x.r[6] = v0.w;
      x.r[7] = v1.x, x.r[8] = v1.y, x.r[9] = v1.z, x.r[10] = v1.w;
      x.r[11] = qp[8];
      break;
    }
    case IE_RQ_IQ2_S: {
      x.r[0] = *rq_ap(p, 0u);
      const u32x4 hp = *(const G u32x4 *)rq_ap(p, 66u + 4u * hb);
      x.r[1] = hp.x, x.r[2] = hp.y, x.r[3] = hp.z, x.r[4] = hp.w;
      const G u32 *qlp = rq_ap(p, 2u + 16u * hb);
      const u32x4 ql = *(const G u32x4 *)qlp;
      x.r[5] = ql.x, x.r[6] = ql.y, x.r[7] = ql.z, x.r[8] = ql.w, x.r[9] = qlp[4];
      const G u32 *sgp = rq_ap(p, 34u + 16u * hb);
      const u32x4 sg = *(const G u32x4 *)sgp;
      x.r[10] = sg.x, x.r[11] = sg.y, x.r[12] = sg.z, x.r[13] = sg.w, x.r[14] = sgp[4];
      break;
    }
    case IE_RQ_IQ3_XXS: {
      x.r[0] = *rq_ap(p, 0u);
      const G u32 *axp = rq_ap(p, 66u + 16u * hb);
      const u32x4 ax = *(const G u32x4 *)axp;
      x.r[1] = ax.x, x.r[2] = ax.y, x.r[3] = ax.z, x.r[4] = ax.w, x.r[5] = axp[4];
      const G u32 *qp = rq_ap(p, 2u + 32u * hb);
      const u32x4 v0 = *(const G u32x4 *)(qp + 0u), v1 = *(const G u32x4 *)(qp + 4u);
      x.r[6] = v0.x, x.r[7] = v0.y, x.r[8] = v0.z, x.r[9] = v0.w;
      x.r[10] = v1.x, x.r[11] = v1.y, x.r[12] = v1.z, x.r[13] = v1.w, x.r[14] = qp[8];
      break;
    }
    case IE_RQ_IQ3_S: {
      x.r[0] = *rq_ap(p, 0u);
      const rq_u32x2 qhv = *(const G rq_u32x2 *)rq_ap(p, 66u + 4u * hb);
      x.r[1] = qhv.x, x.r[2] = qhv.y;
      x.r[3] = *rq_ap(p, 106u + 2u * hb);
      const G u32 *sgp = rq_ap(p, 74u + 16u * hb);
      const u32x4 sg = *(const G u32x4 *)sgp;
      x.r[4] = sg.x, x.r[5] = sg.y, x.r[6] = sg.z, x.r[7] = sg.w, x.r[8] = sgp[4];
      const G u32 *qp = rq_ap(p, 2u + 32u * hb);
      const u32x4 v0 = *(const G u32x4 *)(qp + 0u), v1 = *(const G u32x4 *)(qp + 4u);
      x.r[9] = v0.x, x.r[10] = v0.y, x.r[11] = v0.z, x.r[12] = v0.w;
      x.r[13] = v1.x, x.r[14] = v1.y, x.r[15] = v1.z, x.r[16] = v1.w, x.r[17] = qp[8];
      break;
    }
    case IE_RQ_IQ1_S: {
      x.r[0] = *rq_ap(p, 0u);
      const G u32 *qhp = rq_ap(p, 34u + 8u * hb);
      const rq_u32x2 qhv = *(const G rq_u32x2 *)qhp;
      x.r[1] = qhv.x, x.r[2] = qhv.y, x.r[3] = qhp[2];
      const G u32 *qlp = rq_ap(p, 2u + 16u * hb);
      const u32x4 ql = *(const G u32x4 *)qlp;
      x.r[4] = ql.x, x.r[5] = ql.y, x.r[6] = ql.z, x.r[7] = ql.w, x.r[8] = qlp[4];
      break;
    }
    case IE_RQ_IQ1_M: {
      x.sh = 0u;
      const rq_u32x2 sc = *(const G rq_u32x2 *)(p + 48u);
      const rq_u32x2 qh = *(const G rq_u32x2 *)(p + 32u + 8u * hb);
      const u32x4 ql = *(const G u32x4 *)(p + 16u * hb);
      x.r[0] = sc.x, x.r[1] = sc.y;
      x.r[2] = qh.x, x.r[3] = qh.y;
      x.r[4] = ql.x, x.r[5] = ql.y, x.r[6] = ql.z, x.r[7] = ql.w;
      break;
    }
    default:
      break;
  }
  return x;
}

static inline __attribute__((always_inline)) rq_sub_t rq_hdec(const u32 type, const rq_hblk_t *x, const LDSP ie_rq_u64 *ksigns,
                                                              const LDSP u32 *lut, const u32 hb, const u32 k, const u32 h) {
  (void)ksigns;
  rq_sub_t r;
  r.s0 = r.s1 = r.m0 = r.m1 = 0.0f;
  r.w0 = r.w1 = (v2i)(0);
  const u32 sh = x->sh, qsh = sh ^ 16u;
  const u32 psel = h ? 0x07060302u : 0x05040100u;
  switch (type) {
    case IE_RQ_Q2_K: {
      const float d = ie_rq_h2f(x->r[0] & 0xFFFFu), mn = ie_rq_h2f(x->r[0] >> 16u);
      const u32 sw = x->r[1u + (k >> 1u)] >> (16u * (k & 1u));
      const u32 sc0 = sw & 255u, sc1 = (sw >> 8u) & 255u, ksh = 2u * k;
      r.w0.x = (int)(((x->r[3] >> ksh) & 0x03030303u) * (sc0 & 15u));
      r.w0.y = (int)(((x->r[4] >> ksh) & 0x03030303u) * (sc0 & 15u));
      r.w1.x = (int)(((x->r[5] >> ksh) & 0x03030303u) * (sc1 & 15u));
      r.w1.y = (int)(((x->r[6] >> ksh) & 0x03030303u) * (sc1 & 15u));
      r.s0 = d;
      r.m0 = mn * (float)(sc0 >> 4u);
      r.m1 = mn * (float)(sc1 >> 4u);
      break;
    }
    case IE_RQ_IQ2_XXS: {
      const float d = ie_rq_h2f((x->r[0] >> sh) & 0xFFFFu);
      const u32 a0 = rq_align(x->r[1u + 2u * k], x->r[2u + 2u * k], qsh);
      const u32 a1 = rq_align(x->r[2u + 2u * k], x->r[3u + 2u * k], qsh);
      r.s0 = d * (0.5f + (float)(a1 >> 28u)) * 0.25f;
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const u32 g8 = 2u * q + h;
        const u32 s8 = rq_ksign7((a1 >> (7u * g8)) & 127u);
        const u32 cp = lut[(a0 >> (8u * g8)) & 255u];
        const v2i v = {(int)RQ_SGN_B(cp & 0x03030303u, s8 & 15u), (int)RQ_SGN_B((cp >> 4u) & 0x03030303u, s8 >> 4u)};
        if (q == 0u) r.w0 = v;
        else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ2_XS: {
      const float d = ie_rq_h2f((x->r[0] >> sh) & 0xFFFFu);
      const u32 sc = (rq_align(x->r[1], x->r[2], qsh) >> (8u * k)) & 255u;
      r.s0 = d * (0.5f + (float)(sc & 15u)) * 0.25f;
      r.s1 = d * (0.5f + (float)(sc >> 4u)) * 0.25f;
      const u32 w0 = rq_align(x->r[3u + 2u * k], x->r[4u + 2u * k], qsh);
      const u32 w1 = rq_align(x->r[4u + 2u * k], x->r[5u + 2u * k], qsh);
      const u32 qp = __builtin_amdgcn_perm(w1, w0, psel);
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const u32 q2 = (qp >> (16u * q)) & 0xFFFFu;
        const u32 s8 = rq_ksign7(q2 >> 9u);
        const u32 cp = lut[q2 & 511u];
        const v2i v = {(int)RQ_SGN_B(cp & 0x03030303u, s8 & 15u), (int)RQ_SGN_B((cp >> 4u) & 0x03030303u, s8 >> 4u)};
        if (q == 0u) r.w0 = v;
        else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ2_S: {
      const float d = ie_rq_h2f((x->r[0] >> sh) & 0xFFFFu);
      const u32 qh = (rq_align(x->r[1], x->r[2], qsh) >> (8u * k)) & 255u;
      const u32 sc = (rq_align(x->r[3], x->r[4], qsh) >> (8u * k)) & 255u;
      r.s0 = d * (0.5f + (float)(sc & 15u)) * 0.25f;
      r.s1 = d * (0.5f + (float)(sc >> 4u)) * 0.25f;
      const u32 qlo = rq_align(x->r[5u + k], x->r[6u + k], qsh);
      const u32 sgn = rq_align(x->r[10u + k], x->r[11u + k], qsh);
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const u32 g8 = 2u * q + h;
        const u32 idx = ie_byte(qlo, g8) | ((qh << (8u - 2u * g8)) & 0x300u);
        const u32 sb = ie_byte(sgn, g8);
        const u32 cp = lut[idx];
        const v2i v = {(int)RQ_SGN_B(cp & 0x03030303u, sb & 15u), (int)RQ_SGN_B((cp >> 4u) & 0x03030303u, sb >> 4u)};
        if (q == 0u) r.w0 = v;
        else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ3_XXS: {
      const float d = ie_rq_h2f((x->r[0] >> sh) & 0xFFFFu);
      const u32 aux = rq_align(x->r[1u + k], x->r[2u + k], qsh);
      const u32 w0 = rq_align(x->r[6u + 2u * k], x->r[7u + 2u * k], qsh);
      const u32 w1 = rq_align(x->r[7u + 2u * k], x->r[8u + 2u * k], qsh);
      const u32 qp = __builtin_amdgcn_perm(w1, w0, psel);
      r.s0 = d * (0.5f + (float)(aux >> 28u)) * 0.5f;
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const u32 g8 = 2u * q + h, w16 = (qp >> (16u * q)) & 0xFFFFu;
        const u32 s8 = rq_ksign7((aux >> (7u * g8)) & 127u);
        const u32 g1 = lut[w16 & 255u], g2 = lut[w16 >> 8u];
        const v2i v = {(int)ie_rq_ssign(g1, ie_rq_mask4(s8 & 15u)), (int)ie_rq_ssign(g2, ie_rq_mask4(s8 >> 4u))};
        if (q == 0u) r.w0 = v;
        else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ3_S: {
      const float d = ie_rq_h2f((x->r[0] >> sh) & 0xFFFFu);
      const u32 qh = (rq_align(x->r[1], x->r[2], qsh) >> (8u * k)) & 255u;
      const u32 sc_sh = sh ^ (hb ? 0u : 16u);
      const u32 sc = ((x->r[3] >> sc_sh) >> (4u * k)) & 15u;
      r.s0 = d * (float)(int)(1u + 2u * sc);
      const u32 sgn = rq_align(x->r[4u + k], x->r[5u + k], qsh);
      const u32 w0 = rq_align(x->r[9u + 2u * k], x->r[10u + 2u * k], qsh);
      const u32 w1 = rq_align(x->r[10u + 2u * k], x->r[11u + 2u * k], qsh);
      const u32 qp = __builtin_amdgcn_perm(w1, w0, psel);
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const u32 g8 = 2u * q + h, w16 = (qp >> (16u * q)) & 0xFFFFu, sb = ie_byte(sgn, g8);
        const u32 i1 = (w16 & 255u) | (((qh >> (2u * g8)) & 1u) << 8u);
        const u32 i2 = (w16 >> 8u) | (((qh >> (2u * g8 + 1u)) & 1u) << 8u);
        const v2i v = {(int)ie_rq_ssign(lut[i1], ie_rq_mask4(sb & 15u)), (int)ie_rq_ssign(lut[i2], ie_rq_mask4(sb >> 4u))};
        if (q == 0u) r.w0 = v;
        else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ1_S: {
      const float d = ie_rq_h2f((x->r[0] >> sh) & 0xFFFFu);
      const u32 qhw = rq_align(x->r[1u + (k >> 1u)], x->r[2u + (k >> 1u)], qsh);
      const u32 qh = (qhw >> (16u * (k & 1u))) & 0xFFFFu;
      r.s0 = d * (float)(int)(2u * ((qh >> 12u) & 7u) + 1u) * 0.125f;
      const u32 tbl = (qh & 0x8000u) != 0u ? RQ_IQ1_TBL_NEG : RQ_IQ1_TBL_POS;
      const u32 qlo = rq_align(x->r[4u + k], x->r[5u + k], qsh);
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const u32 g8 = 2u * q + h;
        const u32 idx = ie_byte(qlo, g8) | (((qh >> (3u * g8)) & 7u) << 8u);
        const u32 cp = lut[idx];
        const v2i v = {(int)rq_iq1_map(cp, tbl), (int)rq_iq1_map(cp >> 4u, tbl)};
        if (q == 0u) r.w0 = v;
        else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ1_M: {
      const u32 s0 = x->r[0] & 0xFFFFu, s1 = x->r[0] >> 16u, s2 = x->r[1] & 0xFFFFu, s3 = x->r[1] >> 16u;
      const float d = ie_rq_h2f((s0 >> 12u) | ((s1 >> 8u) & 0x00F0u) | ((s2 >> 4u) & 0x0F00u) | (s3 & 0xF000u));
      const u32 pair = hb != 0u ? x->r[1] : x->r[0];
      const u32 scv = pair >> (16u * (k >> 1u)), ksh = 6u * (k & 1u);
      r.s0 = d * (float)(int)(2u * ((scv >> ksh) & 7u) + 1u) * 0.125f;
      r.s1 = d * (float)(int)(2u * ((scv >> (ksh + 3u)) & 7u) + 1u) * 0.125f;
      const u32 qhb = (x->r[2u + (k >> 1u)] >> (16u * (k & 1u))) & 0xFFFFu;
      const u32 qlo = x->r[4u + k];
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const u32 g8 = 2u * q + h, qh = q == 0u ? (qhb & 255u) : (qhb >> 8u);
        const u32 idx = ie_byte(qlo, g8) | ((qh << (8u - 4u * h)) & 0x700u);
        const u32 tbl = (qh & (h != 0u ? 0x80u : 0x08u)) != 0u ? RQ_IQ1_TBL_NEG : RQ_IQ1_TBL_POS;
        const u32 cp = lut[idx];
        const v2i v = {(int)rq_iq1_map(cp, tbl), (int)rq_iq1_map(cp >> 4u, tbl)};
        if (q == 0u) r.w0 = v;
        else r.w1 = v;
      }
      break;
    }
    default:
      break;
  }
  return r;
}

#endif /* IE_RQ_WMMA_COMMON */

/* ============================================================================
 * P1: Coalesced Shared-Weight Pipelined Matrix-Core Kernels (exact per-32 Q8_0 scales)
 *   - 8 waves split as 4 row-waves (64 rows/WG) x 2 token-waves sharing 64 weight rows in LDS
 *   - 100% 128-byte cache-line coalesced cooperative weight loads (1-2 global_load_b128/thread
 *     per 256-weight superblock, prefetched across all 64 WMMAs of both half-blocks)
 *   - ie_gemm_rq_<type>   : RQC_R = 64 x RQC_T = 128   (4 WMMA tiles/wave, <= 31 KiB LDS = 2 WGs/CU)
 *   - ie_gemm_rq32_<type> : RQC_R = 64 x RQC32_T = 64  (2 WMMA tiles/wave, <= 21.25 KiB LDS = 3 WGs/CU)
 * ========================================================================== */
#define RQC_R 128u
#define RQC_T 64u
#define RQC_WJ 4u
#define RQC32_R 64u
#define RQC32_T 64u
#define RQC32_WJ 2u
#define RQ_R 128u /* kept for ie_gemm_ri_<type> below */

typedef short rqc_s8 __attribute__((ext_vector_type(8)));

static inline __attribute__((always_inline)) int rqc_sum16i(const u32x4 v) {
  u32 a = ie_dot4_ss(0u, v.x, 0x01010101u);
  a = ie_dot4_ss(a, v.y, 0x01010101u);
  a = ie_dot4_ss(a, v.z, 0x01010101u);
  a = ie_dot4_ss(a, v.w, 0x01010101u);
  return (int)a;
}

static inline __attribute__((always_inline)) LDSP u32x4 *rqc_lx_slot(LDSP u8 *lx, const u32 tk, const u32 c8) {
  return (LDSP u32x4 *)(lx + tk * 128u + ((c8 ^ (tk & 7u)) << 4u));
}

static inline __attribute__((always_inline)) const LDSP u32x4 *rqc_lx_cslot(const LDSP u8 *lx, const u32 tk, const u32 c8) {
  return (const LDSP u32x4 *)(lx + tk * 128u + ((c8 ^ (tk & 7u)) << 4u));
}

static inline __attribute__((always_inline)) const LDSP u32 *rqc_lap(const LDSP u8 *lw_row, const u32 off16, const u32 o) {
  return (const LDSP u32 *)(lw_row + ((off16 + o) & ~3u));
}

static inline __attribute__((always_inline)) rq_hblk_t rq_hload_lds(const u32 type, const LDSP u8 *lw_row, const u32 off16,
                                                                    const u32 hb, const u32 h) {
  rq_hblk_t x;
#pragma unroll
  for (u32 i = 0; i < 18u; i++) x.r[i] = 0u;
  x.sh = (off16 & 2u) << 3u;
  switch (type) {
    case IE_RQ_Q2_K: {
      x.sh = 0u;
      const LDSP u32 *lp = (const LDSP u32 *)(lw_row + off16);
      x.r[0] = lp[20u];
      x.r[1] = lp[2u * hb + 0u];
      x.r[2] = lp[2u * hb + 1u];
      const u32 qw = 4u + 8u * hb + 2u * h;
      x.r[3] = lp[qw + 0u];
      x.r[4] = lp[qw + 1u];
      x.r[5] = lp[qw + 4u];
      x.r[6] = lp[qw + 5u];
      break;
    }
    case IE_RQ_IQ2_XXS: {
      x.r[0] = *rqc_lap(lw_row, off16, 0u);
      const LDSP u32 *qp = rqc_lap(lw_row, off16, 2u + 32u * hb);
#pragma unroll
      for (u32 i = 0; i < 9u; i++) x.r[1u + i] = qp[i];
      break;
    }
    case IE_RQ_IQ2_XS: {
      x.r[0] = *rqc_lap(lw_row, off16, 0u);
      const LDSP u32 *sp = rqc_lap(lw_row, off16, 66u + 4u * hb);
      x.r[1] = sp[0], x.r[2] = sp[1];
      const LDSP u32 *qp = rqc_lap(lw_row, off16, 2u + 32u * hb);
#pragma unroll
      for (u32 i = 0; i < 9u; i++) x.r[3u + i] = qp[i];
      break;
    }
    case IE_RQ_IQ2_S: {
      x.r[0] = *rqc_lap(lw_row, off16, 0u);
      const LDSP u32 *hp = rqc_lap(lw_row, off16, 66u + 4u * hb);
#pragma unroll
      for (u32 i = 0; i < 4u; i++) x.r[1u + i] = hp[i];
      const LDSP u32 *qlp = rqc_lap(lw_row, off16, 2u + 16u * hb);
#pragma unroll
      for (u32 i = 0; i < 5u; i++) x.r[5u + i] = qlp[i];
      const LDSP u32 *sgp = rqc_lap(lw_row, off16, 34u + 16u * hb);
#pragma unroll
      for (u32 i = 0; i < 5u; i++) x.r[10u + i] = sgp[i];
      break;
    }
    case IE_RQ_IQ3_XXS: {
      x.r[0] = *rqc_lap(lw_row, off16, 0u);
      const LDSP u32 *axp = rqc_lap(lw_row, off16, 66u + 16u * hb);
#pragma unroll
      for (u32 i = 0; i < 5u; i++) x.r[1u + i] = axp[i];
      const LDSP u32 *qp = rqc_lap(lw_row, off16, 2u + 32u * hb);
#pragma unroll
      for (u32 i = 0; i < 9u; i++) x.r[6u + i] = qp[i];
      break;
    }
    case IE_RQ_IQ3_S: {
      x.r[0] = *rqc_lap(lw_row, off16, 0u);
      const LDSP u32 *qhv = rqc_lap(lw_row, off16, 66u + 4u * hb);
      x.r[1] = qhv[0], x.r[2] = qhv[1];
      x.r[3] = *rqc_lap(lw_row, off16, 106u + 2u * hb);
      const LDSP u32 *sgp = rqc_lap(lw_row, off16, 74u + 16u * hb);
#pragma unroll
      for (u32 i = 0; i < 5u; i++) x.r[4u + i] = sgp[i];
      const LDSP u32 *qp = rqc_lap(lw_row, off16, 2u + 32u * hb);
#pragma unroll
      for (u32 i = 0; i < 9u; i++) x.r[9u + i] = qp[i];
      break;
    }
    case IE_RQ_IQ1_S: {
      x.r[0] = *rqc_lap(lw_row, off16, 0u);
      const LDSP u32 *qhp = rqc_lap(lw_row, off16, 34u + 8u * hb);
      x.r[1] = qhp[0], x.r[2] = qhp[1], x.r[3] = qhp[2];
      const LDSP u32 *qlp = rqc_lap(lw_row, off16, 2u + 16u * hb);
#pragma unroll
      for (u32 i = 0; i < 5u; i++) x.r[4u + i] = qlp[i];
      break;
    }
    case IE_RQ_IQ1_M: {
      x.sh = 0u;
      const LDSP u32 *lp = (const LDSP u32 *)(lw_row + off16);
      x.r[0] = lp[12u], x.r[1] = lp[13u];
      x.r[2] = lp[8u + 2u * hb], x.r[3] = lp[9u + 2u * hb];
#pragma unroll
      for (u32 i = 0; i < 4u; i++) x.r[4u + i] = lp[4u * hb + i];
      break;
    }
    default:
      break;
  }
  return x;
}

static inline __attribute__((always_inline)) void rqc_math_half(const u32 type, const LDSP u8 *lw_row, const u32 off16,
                                                                const LDSP u32 *lut, const u32 hb, const LDSP u8 *lx,
                                                                const LDSP float *lsx, const LDSP short *lsa, const u32 ntok,
                                                                const u32 t_wave, const u32 wj, const u32 l, const u32 h, f8 *acc) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  const rq_hblk_t cur = rq_hload_lds(type, lw_row, off16, hb, h);
#pragma unroll
  for (u32 k = 0; k < RQ_KB; k++) {
    const rq_sub_t dc = rq_hdec(type, &cur, (const LDSP ie_rq_u64 *)0, lut, hb, k, h);
    v2i a0[4], a1[4];
    v8i c0[4], c1[4];
#pragma unroll
    for (u32 j = 0; j < wj; j++) {
      const u32 tk = t_wave + 16u * j + (l & 15u);
      a0[j] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k) + 8u * h);
      a1[j] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k + 1u) + 8u * h);
    }
    if (rq_split16(type)) {
#pragma unroll
      for (u32 j = 0; j < wj; j++) {
        c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0[j], 1, dc.w0, z, 0);
        c1[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1[j], 1, dc.w1, z, 0);
      }
    } else {
#pragma unroll
      for (u32 j = 0; j < wj; j++) {
        c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0[j], 1, dc.w0, z, 0);
        c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1[j], 1, dc.w1, c0[j], 0);
      }
    }
#pragma unroll
    for (u32 j = 0; j < wj; j++) {
      const u32 ts = t_wave + 16u * j + 8u * h;
      const f8 dx = *(const LDSP f8 *)(lsx + k * ntok + ts);
      f8 t = __builtin_convertvector(c0[j], f8) * dc.s0;
      if (rq_split16(type)) {
        t += __builtin_convertvector(c1[j], f8) * dc.s1;
      } else if (type == IE_RQ_Q2_K) {
        const f8 sa0 = __builtin_convertvector(*(const LDSP rqc_s8 *)(lsa + (2u * k) * ntok + ts), f8);
        const f8 sa1 = __builtin_convertvector(*(const LDSP rqc_s8 *)(lsa + (2u * k + 1u) * ntok + ts), f8);
        t -= sa0 * dc.m0;
        t -= sa1 * dc.m1;
      }
      acc[j] += t * dx;
      __asm__ volatile("" : "+v"(acc[j]));
    }
  }
}

/* 128x64 Double-Buffered Activations + Single-Decode-Per-Row Pipelined Matrix-Core GEMM (exact per-32 Q8_0 scales):
 *   - All 8 waves own disjoint 16-row tiles (128 rows/WG x 64 tokens/WG, 4 WMMA tiles/wave) -> 1x weight decode!
 *   - Each wave loads & reads ONLY its own 16 weight rows in lw (zero cross-wave lw hazard).
 *   - Double-buffered lx[2] / lsx[2] / lsa[2] cuts superblock barriers from 4 down to 2, <= 39.4 KiB LDS = 3 WGs/WGP. */
#define RQC_DEF_PIPE128(fn_name_, wvec_, wstr_)                                                                                   \
  static inline __attribute__((always_inline)) void fn_name_(                                                                     \
      const u32 type, const G u8 *q, const G u8 *xq, G u8 *y, const u32 rows, const u32 nb, const u32 T, const G float *bias,     \
      const G u8 *res, const u32 xs, const u32 ys, const u32 rs, LDSP u8 *lx, LDSP float *lsx, LDSP u8 *lw, LDSP short *lsa,      \
      LDSP u32 *lut) {                                                                                                            \
    if (T == 0u || rows == 0u || nb < 8u || (nb & 7u) != 0u || !q || !xq || !y) return;                                           \
    const u32 ntt = (T + RQC_T - 1u) / RQC_T, tt = wgid() % ntt, rt = wgid() / ntt;                                               \
    const u32 rbase = rt * RQC_R, tbase = tt * RQC_T;                                                                             \
    if (rbase >= rows || tbase >= T) return;                                                                                      \
    const u32 t0 = tid(), w = t0 >> 5u, l = lane(), h = l >> 4u;                                                                  \
    const u32 r_loc = 16u * w + (l & 15u), row = rbase + r_loc, rr = row < rows ? row : rows - 1u;                                \
    const u32 mb = nb >> 3u, bs = ie_rq_bs(type);                                                                                 \
    const G u8 *rowp = q + (unsigned long)rr * mb * bs;                                                                           \
    const LDSP u8 *lw_row = lw + r_loc * (wstr_);                                                                                 \
    LDSP u8 *lx0 = lx, *lx1 = lx + RQC_T * 128u;                                                                                  \
    LDSP float *lsx0 = lsx, *lsx1 = lsx + RQ_KB * RQC_T;                                                                          \
    LDSP short *lsa0 = lsa, *lsa1 = lsa ? lsa + 2u * RQ_KB * RQC_T : (LDSP short *)0;                                             \
    f8 acc[RQC_WJ];                                                                                                               \
    _Pragma("unroll") for (u32 j = 0; j < RQC_WJ; j++) acc[j] = (f8)(0.0f);                                                       \
    rq_stage_lut_raw(type, (LDSP ie_rq_u64 *)0, lut, 0);                                                                          \
    if ((wvec_) == 4u && t0 < RQC_R) *(LDSP u32 *)(lw + t0 * (wstr_) + 64u) = 0u;                                                 \
    /* Wave w loads ONLY its own 16 weight rows 16*w .. 16*w + 15 (100% coalesced 128B/64B rows, zero cross-wave lw sharing) */   \
    const u32 r_w0 = 16u * w + ((wvec_) == 4u ? (l >> 2u) : (l >> 3u));                                                           \
    const u32 c_ld = (wvec_) == 4u ? (l & 3u) : (l & 7u);                                                                         \
    enum { NW = (wvec_) == 4u ? 2 : 4, RSTEP = (wvec_) == 4u ? 8u : 4u };                                                         \
    const G u8 *wp[NW];                                                                                                           \
    _Pragma("unroll") for (u32 p = 0; p < NW; p++) {                                                                              \
      const u32 gr = rbase + r_w0 + p * RSTEP < rows ? rbase + r_w0 + p * RSTEP : rows - 1u;                                      \
      wp[p] = q + (unsigned long)gr * mb * bs;                                                                                    \
    }                                                                                                                             \
    enum { NPC = (RQC_T * 8u) / NT };                                                                                             \
    const G u8 *xp[NPC];                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < NPC; i++) {                                                                             \
      const u32 p = t0 + NT * i, tk = p >> 3u, gt = tbase + tk < T ? tbase + tk : T - 1u;                                         \
      xp[i] = xq + (unsigned long)gt * xs + (p & 7u) * 16u;                                                                       \
    }                                                                                                                             \
    const u32 gs = tbase + (t0 & (RQC_T - 1u)) < T ? tbase + (t0 & (RQC_T - 1u)) : T - 1u;                                        \
    const G float *sxp = (const G float *)(xq + (unsigned long)gs * xs + 32ul * nb);                                              \
    u32x4 wn[NW];                                                                                                                 \
    _Pragma("unroll") for (u32 p = 0; p < NW; p++) wn[p] = *(const G u32x4 *)(((unsigned long)wp[p] & ~15ul) + 16ul * c_ld);      \
    u32x4 rx[NPC];                                                                                                                \
    f32x4 rsx = (f32x4)(0.0f);                                                                                                    \
    _Pragma("unroll") for (u32 i = 0; i < NPC; i++) rx[i] = *(const G u32x4 *)xp[i];                                              \
    if (t0 < RQC_T) rsx = rq_load_f32x4(sxp);                                                                                     \
    _Pragma("unroll") for (u32 p = 0; p < NW; p++) {                                                                              \
      LDSP u32 *dp = (LDSP u32 *)(lw + (r_w0 + p * RSTEP) * (wstr_) + 16u * c_ld);                                                \
      dp[0] = wn[p].x, dp[1] = wn[p].y, dp[2] = wn[p].z, dp[3] = wn[p].w;                                                         \
    }                                                                                                                             \
    _Pragma("unroll") for (u32 i = 0; i < NPC; i++) {                                                                             \
      const u32 p = t0 + NT * i, tk = p >> 3u, c8 = p & 7u;                                                                       \
      *rqc_lx_slot(lx0, tk, c8) = rx[i];                                                                                          \
      if (type == IE_RQ_Q2_K) lsa0[c8 * RQC_T + tk] = (short)rqc_sum16i(rx[i]);                                                   \
    }                                                                                                                             \
    if (t0 < RQC_T) {                                                                                                             \
      lsx0[0u * RQC_T + t0] = rsx.x, lsx0[1u * RQC_T + t0] = rsx.y;                                                               \
      lsx0[2u * RQC_T + t0] = rsx.z, lsx0[3u * RQC_T + t0] = rsx.w;                                                               \
    }                                                                                                                             \
    barrier();                                                                                                                    \
    _Pragma("unroll 1") for (u32 qb = 0; qb < mb; qb++) {                                                                         \
      const int more = qb + 1u < mb;                                                                                              \
      const u32 off16 = (u32)((unsigned long)(rowp + (unsigned long)qb * bs) & 15ul);                                             \
      {                                                                                                                           \
        const u32 kb1 = 8u * qb + 4u;                                                                                             \
        _Pragma("unroll") for (u32 i = 0; i < NPC; i++) rx[i] = *(const G u32x4 *)(xp[i] + kb1 * 32u);                            \
        if (t0 < RQC_T) rsx = rq_load_f32x4(sxp + kb1);                                                                           \
        if (more) {                                                                                                               \
          const unsigned long woff = (unsigned long)(qb + 1u) * bs;                                                               \
          _Pragma("unroll") for (u32 p = 0; p < NW; p++)                                                                          \
            wn[p] = *(const G u32x4 *)(((unsigned long)(wp[p] + woff) & ~15ul) + 16ul * c_ld);                                    \
        }                                                                                                                         \
      }                                                                                                                           \
      __builtin_amdgcn_sched_barrier(0);                                                                                          \
      rqc_math_half(type, lw_row, off16, lut, 0u, lx0, lsx0, lsa0, RQC_T, 0u, RQC_WJ, l, h, acc);                                 \
      _Pragma("unroll") for (u32 i = 0; i < NPC; i++) {                                                                           \
        const u32 p = t0 + NT * i, tk = p >> 3u, c8 = p & 7u;                                                                     \
        *rqc_lx_slot(lx1, tk, c8) = rx[i];                                                                                        \
        if (type == IE_RQ_Q2_K) lsa1[c8 * RQC_T + tk] = (short)rqc_sum16i(rx[i]);                                                 \
      }                                                                                                                           \
      if (t0 < RQC_T) {                                                                                                           \
        lsx1[0u * RQC_T + t0] = rsx.x, lsx1[1u * RQC_T + t0] = rsx.y;                                                             \
        lsx1[2u * RQC_T + t0] = rsx.z, lsx1[3u * RQC_T + t0] = rsx.w;                                                             \
      }                                                                                                                           \
      barrier(); /* Barrier 1 of 2: lx1 ready, and all waves finished reading lx0 */                                              \
      if (more) {                                                                                                                 \
        const u32 kb0 = 8u * (qb + 1u);                                                                                           \
        _Pragma("unroll") for (u32 i = 0; i < NPC; i++) rx[i] = *(const G u32x4 *)(xp[i] + kb0 * 32u);                            \
        if (t0 < RQC_T) rsx = rq_load_f32x4(sxp + kb0);                                                                           \
      }                                                                                                                           \
      __builtin_amdgcn_sched_barrier(0);                                                                                          \
      rqc_math_half(type, lw_row, off16, lut, 1u, lx1, lsx1, lsa1, RQC_T, 0u, RQC_WJ, l, h, acc);                                 \
      __builtin_amdgcn_sched_barrier(0);                                                                                          \
      if (more) {                                                                                                                 \
        _Pragma("unroll") for (u32 p = 0; p < NW; p++) {                                                                          \
          LDSP u32 *dp = (LDSP u32 *)(lw + (r_w0 + p * RSTEP) * (wstr_) + 16u * c_ld);                                            \
          dp[0] = wn[p].x, dp[1] = wn[p].y, dp[2] = wn[p].z, dp[3] = wn[p].w;                                                     \
        }                                                                                                                         \
        _Pragma("unroll") for (u32 i = 0; i < NPC; i++) {                                                                         \
          const u32 p = t0 + NT * i, tk = p >> 3u, c8 = p & 7u;                                                                   \
          *rqc_lx_slot(lx0, tk, c8) = rx[i];                                                                                      \
          if (type == IE_RQ_Q2_K) lsa0[c8 * RQC_T + tk] = (short)rqc_sum16i(rx[i]);                                               \
        }                                                                                                                         \
        if (t0 < RQC_T) {                                                                                                         \
          lsx0[0u * RQC_T + t0] = rsx.x, lsx0[1u * RQC_T + t0] = rsx.y;                                                           \
          lsx0[2u * RQC_T + t0] = rsx.z, lsx0[3u * RQC_T + t0] = rsx.w;                                                           \
        }                                                                                                                         \
        barrier(); /* Barrier 2 of 2: lw and lx0 ready, and all waves finished reading lx1 */                                     \
      }                                                                                                                           \
    }                                                                                                                             \
    _Pragma("unroll") for (u32 j = 0; j < RQC_WJ; j++) {                                                                          \
      _Pragma("unroll") for (u32 v = 0; v < 8u; v++) {                                                                            \
        const u32 t = tbase + 16u * j + 8u * h + v;                                                                               \
        if (t < T && row < rows) {                                                                                                \
          const G float *rp = res ? (const G float *)(res + (unsigned long)t * rs) : (const G float *)0;                          \
          ((G float *)(y + (unsigned long)t * ys))[row] = epilogue(acc[j][v], row, bias, rp);                                     \
        }                                                                                                                         \
      }                                                                                                                           \
    }                                                                                                                             \
  }

#define RQC_DEF_PIPE(fn_name_, ntok_, wj_, wvec_, wstr_)                                                                          \
  static inline __attribute__((always_inline)) void fn_name_(                                                                     \
      const u32 type, const G u8 *q, const G u8 *xq, G u8 *y, const u32 rows, const u32 nb, const u32 T, const G float *bias,     \
      const G u8 *res, const u32 xs, const u32 ys, const u32 rs, LDSP u8 *lx, LDSP float *lsx, LDSP u8 *lw, LDSP short *lsa,      \
      LDSP u32 *lut) {                                                                                                            \
    if (T == 0u || rows == 0u || nb < 8u || (nb & 7u) != 0u || !q || !xq || !y) return;                                           \
    const u32 ntt = (T + (ntok_) - 1u) / (ntok_), tt = wgid() % ntt, rt = wgid() / ntt;                                           \
    const u32 rbase = rt * RQC32_R, tbase = tt * (ntok_);                                                                         \
    if (rbase >= rows || tbase >= T) return;                                                                                      \
    const u32 t0 = tid(), w = t0 >> 5u, l = lane(), h = l >> 4u;                                                                  \
    const u32 w_row = w & 3u, w_tok = w >> 2u, t_wave = w_tok * ((ntok_) >> 1u);                                                  \
    const u32 r_loc = 16u * w_row + (l & 15u), row = rbase + r_loc, rr = row < rows ? row : rows - 1u;                            \
    const u32 mb = nb >> 3u, bs = ie_rq_bs(type);                                                                                 \
    const G u8 *rowp = q + (unsigned long)rr * mb * bs;                                                                           \
    const LDSP u8 *lw_row = lw + r_loc * (wstr_);                                                                                 \
    f8 acc[wj_];                                                                                                                  \
    _Pragma("unroll") for (u32 j = 0; j < (wj_); j++) acc[j] = (f8)(0.0f);                                                        \
    rq_stage_lut_raw(type, (LDSP ie_rq_u64 *)0, lut, 0);                                                                          \
    if ((wvec_) == 4u && t0 < RQC32_R) *(LDSP u32 *)(lw + t0 * (wstr_) + 64u) = 0u;                                               \
    const u32 r_ld0 = (wvec_) == 4u ? (t0 >> 2u) : (t0 >> 3u);                                                                    \
    const u32 c_ld = (wvec_) == 4u ? (t0 & 3u) : (t0 & 7u);                                                                       \
    const u32 gr0 = rbase + r_ld0 < rows ? rbase + r_ld0 : rows - 1u;                                                             \
    const u32 gr1 = rbase + 32u + r_ld0 < rows ? rbase + 32u + r_ld0 : rows - 1u;                                                 \
    const G u8 *wp0 = q + (unsigned long)gr0 * mb * bs;                                                                           \
    const G u8 *wp1 = q + (unsigned long)gr1 * mb * bs;                                                                           \
    enum { NPC = ((ntok_) * 8u) / NT };                                                                                           \
    const G u8 *xp[NPC];                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < NPC; i++) {                                                                             \
      const u32 p = t0 + NT * i, tk = p >> 3u, gt = tbase + tk < T ? tbase + tk : T - 1u;                                         \
      xp[i] = xq + (unsigned long)gt * xs + (p & 7u) * 16u;                                                                       \
    }                                                                                                                             \
    const u32 gs = tbase + (t0 & ((ntok_) - 1u)) < T ? tbase + (t0 & ((ntok_) - 1u)) : T - 1u;                                    \
    const G float *sxp = (const G float *)(xq + (unsigned long)gs * xs + 32ul * nb);                                              \
    u32x4 wn0 = *(const G u32x4 *)(((unsigned long)wp0 & ~15ul) + 16ul * c_ld);                                                   \
    u32x4 wn1 = (u32x4)(0u);                                                                                                      \
    if ((wvec_) == 8u) wn1 = *(const G u32x4 *)(((unsigned long)wp1 & ~15ul) + 16ul * c_ld);                                      \
    u32x4 rx[NPC];                                                                                                                \
    f32x4 rsx = (f32x4)(0.0f);                                                                                                    \
    _Pragma("unroll") for (u32 i = 0; i < NPC; i++) rx[i] = *(const G u32x4 *)xp[i];                                              \
    if (t0 < (ntok_)) rsx = rq_load_f32x4(sxp);                                                                                   \
    {                                                                                                                             \
      LDSP u32 *d0 = (LDSP u32 *)(lw + r_ld0 * (wstr_) + 16u * c_ld);                                                             \
      d0[0] = wn0.x, d0[1] = wn0.y, d0[2] = wn0.z, d0[3] = wn0.w;                                                                 \
      if ((wvec_) == 8u) {                                                                                                        \
        LDSP u32 *d1 = (LDSP u32 *)(lw + (32u + r_ld0) * (wstr_) + 16u * c_ld);                                                   \
        d1[0] = wn1.x, d1[1] = wn1.y, d1[2] = wn1.z, d1[3] = wn1.w;                                                               \
      }                                                                                                                           \
      _Pragma("unroll") for (u32 i = 0; i < NPC; i++) {                                                                           \
        const u32 p = t0 + NT * i, tk = p >> 3u, c8 = p & 7u;                                                                     \
        *rqc_lx_slot(lx, tk, c8) = rx[i];                                                                                         \
        if (type == IE_RQ_Q2_K) lsa[c8 * (ntok_) + tk] = (short)rqc_sum16i(rx[i]);                                                \
      }                                                                                                                           \
      if (t0 < (ntok_)) {                                                                                                         \
        lsx[0u * (ntok_) + t0] = rsx.x;                                                                                           \
        lsx[1u * (ntok_) + t0] = rsx.y;                                                                                           \
        lsx[2u * (ntok_) + t0] = rsx.z;                                                                                           \
        lsx[3u * (ntok_) + t0] = rsx.w;                                                                                           \
      }                                                                                                                           \
    }                                                                                                                             \
    barrier();                                                                                                                    \
    _Pragma("unroll 1") for (u32 qb = 0; qb < mb; qb++) {                                                                         \
      const int more = qb + 1u < mb;                                                                                              \
      const u32 off16 = (u32)((unsigned long)(rowp + (unsigned long)qb * bs) & 15ul);                                             \
      /* Prefetch hb=1 activations + next superblock weights asynchronously while computing hb=0 */                               \
      {                                                                                                                           \
        const u32 kb1 = 8u * qb + 4u;                                                                                             \
        _Pragma("unroll") for (u32 i = 0; i < NPC; i++) rx[i] = *(const G u32x4 *)(xp[i] + kb1 * 32u);                            \
        if (t0 < (ntok_)) rsx = rq_load_f32x4(sxp + kb1);                                                                         \
        if (more) {                                                                                                               \
          const unsigned long woff = (unsigned long)(qb + 1u) * bs;                                                               \
          wn0 = *(const G u32x4 *)(((unsigned long)(wp0 + woff) & ~15ul) + 16ul * c_ld);                                          \
          if ((wvec_) == 8u) wn1 = *(const G u32x4 *)(((unsigned long)(wp1 + woff) & ~15ul) + 16ul * c_ld);                       \
        }                                                                                                                         \
      }                                                                                                                           \
      __builtin_amdgcn_sched_barrier(0);                                                                                          \
      rqc_math_half(type, lw_row, off16, lut, 0u, lx, lsx, lsa, (ntok_), t_wave, (wj_), l, h, acc);                               \
      barrier();                                                                                                                  \
      _Pragma("unroll") for (u32 i = 0; i < NPC; i++) {                                                                           \
        const u32 p = t0 + NT * i, tk = p >> 3u, c8 = p & 7u;                                                                     \
        *rqc_lx_slot(lx, tk, c8) = rx[i];                                                                                         \
        if (type == IE_RQ_Q2_K) lsa[c8 * (ntok_) + tk] = (short)rqc_sum16i(rx[i]);                                                \
      }                                                                                                                           \
      if (t0 < (ntok_)) {                                                                                                         \
        lsx[0u * (ntok_) + t0] = rsx.x;                                                                                           \
        lsx[1u * (ntok_) + t0] = rsx.y;                                                                                           \
        lsx[2u * (ntok_) + t0] = rsx.z;                                                                                           \
        lsx[3u * (ntok_) + t0] = rsx.w;                                                                                           \
      }                                                                                                                           \
      barrier();                                                                                                                  \
      /* Prefetch next superblock hb=0 activations asynchronously while computing hb=1 */                                         \
      if (more) {                                                                                                                 \
        const u32 kb0 = 8u * (qb + 1u);                                                                                           \
        _Pragma("unroll") for (u32 i = 0; i < NPC; i++) rx[i] = *(const G u32x4 *)(xp[i] + kb0 * 32u);                            \
        if (t0 < (ntok_)) rsx = rq_load_f32x4(sxp + kb0);                                                                         \
      }                                                                                                                           \
      __builtin_amdgcn_sched_barrier(0);                                                                                          \
      rqc_math_half(type, lw_row, off16, lut, 1u, lx, lsx, lsa, (ntok_), t_wave, (wj_), l, h, acc);                               \
      if (more) {                                                                                                                 \
        barrier();                                                                                                                \
        LDSP u32 *d0 = (LDSP u32 *)(lw + r_ld0 * (wstr_) + 16u * c_ld);                                                           \
        d0[0] = wn0.x, d0[1] = wn0.y, d0[2] = wn0.z, d0[3] = wn0.w;                                                               \
        if ((wvec_) == 8u) {                                                                                                      \
          LDSP u32 *d1 = (LDSP u32 *)(lw + (32u + r_ld0) * (wstr_) + 16u * c_ld);                                                 \
          d1[0] = wn1.x, d1[1] = wn1.y, d1[2] = wn1.z, d1[3] = wn1.w;                                                             \
        }                                                                                                                         \
        _Pragma("unroll") for (u32 i = 0; i < NPC; i++) {                                                                         \
          const u32 p = t0 + NT * i, tk = p >> 3u, c8 = p & 7u;                                                                   \
          *rqc_lx_slot(lx, tk, c8) = rx[i];                                                                                       \
          if (type == IE_RQ_Q2_K) lsa[c8 * (ntok_) + tk] = (short)rqc_sum16i(rx[i]);                                              \
        }                                                                                                                         \
        if (t0 < (ntok_)) {                                                                                                       \
          lsx[0u * (ntok_) + t0] = rsx.x;                                                                                         \
          lsx[1u * (ntok_) + t0] = rsx.y;                                                                                         \
          lsx[2u * (ntok_) + t0] = rsx.z;                                                                                         \
          lsx[3u * (ntok_) + t0] = rsx.w;                                                                                         \
        }                                                                                                                         \
        barrier();                                                                                                                \
      }                                                                                                                           \
    }                                                                                                                             \
    _Pragma("unroll") for (u32 j = 0; j < (wj_); j++) {                                                                           \
      _Pragma("unroll") for (u32 v = 0; v < 8u; v++) {                                                                            \
        const u32 t = tbase + t_wave + 16u * j + 8u * h + v;                                                                      \
        if (t < T && row < rows) {                                                                                                \
          const G float *rp = res ? (const G float *)(res + (unsigned long)t * rs) : (const G float *)0;                          \
          ((G float *)(y + (unsigned long)t * ys))[row] = epilogue(acc[j][v], row, bias, rp);                                     \
        }                                                                                                                         \
      }                                                                                                                           \
    }                                                                                                                             \
  }

RQC_DEF_PIPE128(gemm_rq128_w8, 8u, 132u)
RQC_DEF_PIPE128(gemm_rq128_w4, 4u, 68u)
RQC_DEF_PIPE(gemm_rq64_w8, RQC32_T, RQC32_WJ, 8u, 132u)
RQC_DEF_PIPE(gemm_rq64_w4, RQC32_T, RQC32_WJ, 4u, 68u)

#define RQ_KERNEL_Q2K_DB(name_, run_fn_, wstr_)                                                                                   \
  KERNEL name_(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T, const G float *bias, const G u8 *res, u32 xs,     \
               u32 ys, u32 rs) {                                                                                                  \
    static LDS u8 lx[2u * RQC_T * 128u] __attribute__((aligned(16)));                                                             \
    static LDS float lsx[2u * RQ_KB * RQC_T] __attribute__((aligned(32)));                                                        \
    static LDS u8 lw[RQC_R * (wstr_)] __attribute__((aligned(16)));                                                               \
    static LDS short lsa[4u * RQ_KB * RQC_T] __attribute__((aligned(16)));                                                        \
    run_fn_(IE_RQ_Q2_K, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, lw, lsa, (LDSP u32 *)0);                           \
  }

#define RQ_KERNEL_IQ_DB(name_, type_, run_fn_, wstr_, nlut_)                                                                      \
  KERNEL name_(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T, const G float *bias, const G u8 *res, u32 xs,     \
               u32 ys, u32 rs) {                                                                                                  \
    static LDS u8 lx[2u * RQC_T * 128u] __attribute__((aligned(16)));                                                             \
    static LDS float lsx[2u * RQ_KB * RQC_T] __attribute__((aligned(32)));                                                        \
    static LDS u8 lw[RQC_R * (wstr_)] __attribute__((aligned(16)));                                                               \
    static LDS u32 lut[nlut_] __attribute__((aligned(16)));                                                                       \
    run_fn_(type_, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, lw, (LDSP short *)0, lut);                              \
  }

#define RQ_KERNEL_Q2K32(name_, run_fn_, ntok_, wstr_)                                                                             \
  KERNEL name_(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T, const G float *bias, const G u8 *res, u32 xs,     \
               u32 ys, u32 rs) {                                                                                                  \
    static LDS u8 lx[(ntok_) * 128u] __attribute__((aligned(16)));                                                                \
    static LDS float lsx[RQ_KB * (ntok_)] __attribute__((aligned(32)));                                                           \
    static LDS u8 lw[RQC32_R * (wstr_)] __attribute__((aligned(16)));                                                             \
    static LDS short lsa[2u * RQ_KB * (ntok_)] __attribute__((aligned(16)));                                                      \
    run_fn_(IE_RQ_Q2_K, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, lw, lsa, (LDSP u32 *)0);                           \
  }

#define RQ_KERNEL_IQ32(name_, type_, run_fn_, ntok_, wstr_, nlut_)                                                                \
  KERNEL name_(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T, const G float *bias, const G u8 *res, u32 xs,     \
               u32 ys, u32 rs) {                                                                                                  \
    static LDS u8 lx[(ntok_) * 128u] __attribute__((aligned(16)));                                                                \
    static LDS float lsx[RQ_KB * (ntok_)] __attribute__((aligned(32)));                                                           \
    static LDS u8 lw[RQC32_R * (wstr_)] __attribute__((aligned(16)));                                                             \
    static LDS u32 lut[nlut_] __attribute__((aligned(16)));                                                                       \
    run_fn_(type_, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, lw, (LDSP short *)0, lut);                              \
  }

/* P1a: RQC_R = 128 x RQC_T = 64 double-buffered (grid: ceil(rows / 128) * ceil(T / 64), <= 39.4 KiB LDS = 3 WGs/WGP). */
RQ_KERNEL_Q2K_DB(ie_gemm_rq_q2k, gemm_rq128_w8, 132u)
RQ_KERNEL_IQ_DB(ie_gemm_rq_iq2xxs, IE_RQ_IQ2_XXS, gemm_rq128_w8, 132u, 256u)
RQ_KERNEL_IQ_DB(ie_gemm_rq_iq2xs, IE_RQ_IQ2_XS, gemm_rq128_w8, 132u, 512u)
RQ_KERNEL_IQ_DB(ie_gemm_rq_iq3xxs, IE_RQ_IQ3_XXS, gemm_rq128_w8, 132u, 256u)
RQ_KERNEL_IQ_DB(ie_gemm_rq_iq1s, IE_RQ_IQ1_S, gemm_rq128_w4, 68u, 2048u)
RQ_KERNEL_IQ_DB(ie_gemm_rq_iq2s, IE_RQ_IQ2_S, gemm_rq128_w8, 132u, 1024u)
RQ_KERNEL_IQ_DB(ie_gemm_rq_iq1m, IE_RQ_IQ1_M, gemm_rq128_w4, 68u, 2048u)
RQ_KERNEL_IQ_DB(ie_gemm_rq_iq3s, IE_RQ_IQ3_S, gemm_rq128_w8, 132u, 512u)

/* P1b: RQC32_R = 64 x RQC32_T = 64 (grid: ceil(rows / 64) * ceil(T / 64), <= 21.25 KiB LDS). */
RQ_KERNEL_Q2K32(ie_gemm_rq32_q2k, gemm_rq64_w8, RQC32_T, 132u)
RQ_KERNEL_IQ32(ie_gemm_rq32_iq2xxs, IE_RQ_IQ2_XXS, gemm_rq64_w8, RQC32_T, 132u, 256u)
RQ_KERNEL_IQ32(ie_gemm_rq32_iq2xs, IE_RQ_IQ2_XS, gemm_rq64_w8, RQC32_T, 132u, 512u)
RQ_KERNEL_IQ32(ie_gemm_rq32_iq3xxs, IE_RQ_IQ3_XXS, gemm_rq64_w8, RQC32_T, 132u, 256u)
RQ_KERNEL_IQ32(ie_gemm_rq32_iq1s, IE_RQ_IQ1_S, gemm_rq64_w4, RQC32_T, 68u, 2048u)
RQ_KERNEL_IQ32(ie_gemm_rq32_iq2s, IE_RQ_IQ2_S, gemm_rq64_w8, RQC32_T, 132u, 1024u)
RQ_KERNEL_IQ32(ie_gemm_rq32_iq1m, IE_RQ_IQ1_M, gemm_rq64_w4, RQC32_T, 68u, 2048u)
RQ_KERNEL_IQ32(ie_gemm_rq32_iq3s, IE_RQ_IQ3_S, gemm_rq64_w8, RQC32_T, 132u, 512u)

#undef RQ_KERNEL_IQ
#undef RQ_KERNEL_Q2K
#undef RQC_DEF_PIPE

/* ============================================================================
 * P2: Fused Dual-Matrix Prefill SwiGLU: ie_gemm_rq_swiglu (T >= 17, LDS LUT <= 22 KiB = 2 WGs/CU)
 * ========================================================================== */
#define SWG_R 128u
#define SWG_T 32u
#define SWG_WJ 2u

typedef union {
  struct {
    u8 x[2][SWG_T * RQ_LS] __attribute__((aligned(16)));
    float sx[2][RQ_KB * SWG_T] __attribute__((aligned(32)));
    u32 lut[3072] __attribute__((aligned(16))); /* [0..2047] primary LUT, [2048..3071] secondary LUT or Q2_K sa[2][256] */
  } k;
  float out[SWG_T][SWG_R + 4u];
} rq_swg_lds_t;

static inline __attribute__((always_inline)) void rq_swg_step(const u32 type, const rq_sub_t *dc, const v2i *a0, const v2i *a1,
                                                              const f8 *dx, const LDSP float *lsa_k, const u32 h, f8 *acc) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  v8i c0[SWG_WJ], c1[SWG_WJ];
  if (rq_split16(type)) {
#pragma unroll
    for (u32 j = 0; j < SWG_WJ; j++) {
      c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0[j], 1, dc->w0, z, 0);
      c1[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1[j], 1, dc->w1, z, 0);
    }
  } else {
#pragma unroll
    for (u32 j = 0; j < SWG_WJ; j++) {
      c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0[j], 1, dc->w0, z, 0);
      c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1[j], 1, dc->w1, c0[j], 0);
    }
  }
#pragma unroll
  for (u32 j = 0; j < SWG_WJ; j++) {
    f8 t = __builtin_convertvector(c0[j], f8) * dc->s0;
    if (rq_split16(type)) {
      t += __builtin_convertvector(c1[j], f8) * dc->s1;
    } else if (type == IE_RQ_Q2_K) {
      const f8 sa0 = *(const LDSP f8 *)(lsa_k + 16u * j + 8u * h);
      const f8 sa1 = *(const LDSP f8 *)(lsa_k + SWG_T + 16u * j + 8u * h);
      t -= sa0 * dc->m0;
      t -= sa1 * dc->m1;
    }
    acc[j] += t * dx[j];
    __asm__ volatile("" : "+v"(acc[j]));
  }
}

KERNEL ie_gemm_rq_swiglu(const G u8 *q0, u32 type0, const G u8 *q1, u32 type1, const G u8 *xq,
                         G float *y, G u8 *oq, u32 rows, u32 nb, u32 T, u32 xs, u32 ys, u32 qs) {
  static LDS rq_swg_lds_t swg;
  if (T == 0u || rows == 0u || (rows & 31u) != 0u || nb < 8u || (nb & 7u) != 0u || !q0 || !q1 || !xq) return;
  type0 = (u32)__builtin_amdgcn_readfirstlane(type0);
  type1 = (u32)__builtin_amdgcn_readfirstlane(type1);
  const u32 bs0 = ie_rq_bs(type0), bs1 = ie_rq_bs(type1);
  if (bs0 == 0u || bs1 == 0u) return;

  const u32 ntt = (T + SWG_T - 1u) / SWG_T, tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 rbase = rt * SWG_R, tbase = tt * SWG_T;
  if (rbase >= rows || tbase >= T) return;

  const u32 t0 = tid(), w = t0 >> 5u, l = lane(), h = l >> 4u;
  const u32 row = rbase + 16u * w + (l & 15u), rr = row < rows ? row : rows - 1u;
  const u32 mb = nb >> 3u;
  const G u8 *rowp0 = q0 + (unsigned long)rr * mb * bs0;
  const G u8 *rowp1 = q1 + (unsigned long)rr * mb * bs1;
  const int same = rq_same_lut(type0, type1);
  const int has_q2k = (type0 == IE_RQ_Q2_K || type1 == IE_RQ_Q2_K);
  u32 off0 = 0u, off1 = 0u;
  if (!same) {
    if (type1 == IE_RQ_IQ1_S || type1 == IE_RQ_IQ1_M || type0 == IE_RQ_Q2_K) {
      off1 = 0u;
      off0 = 2048u;
    } else {
      off0 = 0u;
      off1 = 2048u;
    }
  }
  LDSP u32 *const lut0 = swg.k.lut + off0;
  LDSP u32 *const lut1 = swg.k.lut + off1;
  LDSP float *const lsa0 = (LDSP float *)(swg.k.lut + 2048u);
  LDSP float *const lsa1 = lsa0 + 2u * RQ_KB * SWG_T;

  rq_stage_lut_raw(type0, (LDSP ie_rq_u64 *)0, lut0, 0);
  if (!same) rq_stage_lut_raw(type1, (LDSP ie_rq_u64 *)0, lut1, 0);

  const u32 tk_ld = t0 >> 3u, pc_ld = t0 & 7u, gt_ld = tbase + tk_ld < T ? tbase + tk_ld : T - 1u;
  const G u8 *xp = xq + (unsigned long)gt_ld * xs + pc_ld * 16u;
  const u32 gs = tbase + (t0 & (SWG_T - 1u)) < T ? tbase + (t0 & (SWG_T - 1u)) : T - 1u;
  const G float *sxp = (const G float *)(xq + (unsigned long)gs * xs + 32ul * nb);

  u32x4 rx = *(const G u32x4 *)xp;
  f32x4 rsx = t0 < SWG_T ? rq_load_f32x4(sxp) : (f32x4)(0.0f);
  *(LDSP u32x4 *)(swg.k.x[0] + tk_ld * RQ_LS + pc_ld * 16u) = rx;
  if (has_q2k) lsa0[pc_ld * SWG_T + tk_ld] = rq_sum16(rx);
  if (t0 < SWG_T) {
    swg.k.sx[0][0u * SWG_T + t0] = rsx.x;
    swg.k.sx[0][1u * SWG_T + t0] = rsx.y;
    swg.k.sx[0][2u * SWG_T + t0] = rsx.z;
    swg.k.sx[0][3u * SWG_T + t0] = rsx.w;
  }

  rq_hblk_t cur0 = rq_hload(type0, rowp0, 0u, h);
  rq_hblk_t cur1 = rq_hload(type1, rowp1, 0u, h);
  barrier();

  f8 g_acc[SWG_WJ], u_acc[SWG_WJ];
#pragma unroll
  for (u32 j = 0; j < SWG_WJ; j++) g_acc[j] = (f8)(0.0f), u_acc[j] = (f8)(0.0f);

  const u32 nc = 2u * mb;
#pragma unroll 1
  for (u32 c = 0; c < nc; c++) {
    const u32 bf = c & 1u, hb = c & 1u;
    const int more = c + 1u < nc;
    const u32 cn = more ? c + 1u : c;
    const rq_hblk_t nxt0 = rq_hload(type0, rowp0 + (unsigned long)(cn >> 1u) * bs0, cn & 1u, h);
    const rq_hblk_t nxt1 = rq_hload(type1, rowp1 + (unsigned long)(cn >> 1u) * bs1, cn & 1u, h);
    if (more) {
      const u32 kb = 4u * (c + 1u);
      rx = *(const G u32x4 *)(xp + kb * 32u);
      if (t0 < SWG_T) rsx = rq_load_f32x4(sxp + kb);
    }
    __builtin_amdgcn_sched_barrier(0);

    const LDSP u8 *lx = swg.k.x[bf];
    const LDSP float *lsx = swg.k.sx[bf];
    const LDSP float *lsa = bf ? lsa1 : lsa0;
#pragma unroll
    for (u32 k = 0; k < RQ_KB; k++) {
      v2i a0[SWG_WJ], a1[SWG_WJ];
      f8 dx[SWG_WJ];
#pragma unroll
      for (u32 j = 0; j < SWG_WJ; j++) {
        const LDSP u8 *pa = lx + (16u * j + (l & 15u)) * RQ_LS + k * 32u + 8u * h;
        a0[j] = *(const LDSP v2i *)pa;
        a1[j] = *(const LDSP v2i *)(pa + 16u);
        dx[j] = *(const LDSP f8 *)(lsx + k * SWG_T + 16u * j + 8u * h);
      }
      const LDSP float *lsa_k = lsa + (2u * k) * SWG_T;
      const rq_sub_t dc0 = rq_hdec(type0, &cur0, (const LDSP ie_rq_u64 *)0, lut0, hb, k, h);
      const rq_sub_t dc1 = rq_hdec(type1, &cur1, (const LDSP ie_rq_u64 *)0, lut1, hb, k, h);
      rq_swg_step(type0, &dc0, a0, a1, dx, lsa_k, h, g_acc);
      rq_swg_step(type1, &dc1, a0, a1, dx, lsa_k, h, u_acc);
    }

    if (more) {
      const u32 nb_f = bf ^ 1u;
      *(LDSP u32x4 *)(swg.k.x[nb_f] + tk_ld * RQ_LS + pc_ld * 16u) = rx;
      if (has_q2k) (nb_f ? lsa1 : lsa0)[pc_ld * SWG_T + tk_ld] = rq_sum16(rx);
      if (t0 < SWG_T) {
        swg.k.sx[nb_f][0u * SWG_T + t0] = rsx.x;
        swg.k.sx[nb_f][1u * SWG_T + t0] = rsx.y;
        swg.k.sx[nb_f][2u * SWG_T + t0] = rsx.z;
        swg.k.sx[nb_f][3u * SWG_T + t0] = rsx.w;
      }
      lbarrier();
    }
    cur0 = nxt0;
    cur1 = nxt1;
  }

  f8 s_val[SWG_WJ];
#pragma unroll
  for (u32 j = 0; j < SWG_WJ; j++) {
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      const float g = g_acc[j][v], u = u_acc[j][v];
      const u32 t = tbase + 16u * j + 8u * h + v;
      const float sv = (t < T && row < rows) ? (g / (1.0f + __builtin_expf(-g))) * u : 0.0f;
      s_val[j][v] = sv;
      if (y && t < T && row < rows) {
        ((G float *)((G u8 *)y + (unsigned long)t * ys))[row] = sv;
      }
    }
  }

  if (oq) {
    lbarrier();
    const u32 rl = 16u * w + (l & 15u);
#pragma unroll
    for (u32 j = 0; j < SWG_WJ; j++) {
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        swg.out[16u * j + 8u * h + v][rl] = s_val[j][v];
      }
    }
    lbarrier();
    const u32 ntok = T - tbase < SWG_T ? T - tbase : SWG_T;
    const u32 rem_b = (rows - rbase) >> 5u;
    const u32 nblk = rem_b < 4u ? rem_b : 4u;
    const u32 total_q = ntok * nblk, nb_out = rows >> 5u;
    for (u32 idx = w; idx < total_q; idx += 8u) {
      const u32 tk = idx / nblk, blk = idx % nblk;
      const float vq = swg.out[tk][32u * blk + l];
      quant_wave(vq, (rbase >> 5u) + blk, oq + (unsigned long)(tbase + tk) * qs, nb_out);
    }
  }
}

#undef SWG_WJ
#undef SWG_T
#undef SWG_R

/* ============================================================================
 * One activation scale per 256 (IE_RAW_GEMM_I256=1): ie_gemm_ri_<type>
 * ========================================================================== */
typedef struct {
  v2i w0, w1;
  float s0, s1, m0, m1;
  int i0, i1, n0, n1;
  float fd, fm;
} rq_dec_t;

#define RI_T 64u /* tokens per workgroup */
#define RI_WJ (RI_T / 16u)
#define RI_LS (256u + 16u) /* LDS bytes per token row: a block of 256 plus 16 padding */
#define RI_NPC (RI_T * 16u / NT) /* 16-byte pieces of the activation block per thread */
static LDS u8 ri_x[2][RI_T * RI_LS] __attribute__((aligned(16)));
static LDS float ri_sx[2][RI_T] __attribute__((aligned(32))); /* the block's activation scale D per token */
static LDS u8 ri_sh[2][RI_T * 16u] __attribute__((aligned(16)));
static LDS u8 ri_sl[2][RI_T * 16u] __attribute__((aligned(16)));
static inline int rq_sum16i(u32x4 v) {
  u32 a = ie_dot4_ss(0u, v.x, 0x01010101u);
  a = ie_dot4_ss(a, v.y, 0x01010101u);
  a = ie_dot4_ss(a, v.z, 0x01010101u);
  a = ie_dot4_ss(a, v.w, 0x01010101u);
  return (int)a;
}

#define RP_NW 37u
typedef struct {
  u32 r[RP_NW];
} rp_raw_t;
typedef struct {
  ie_rq_u64 g[4][2];
  ie_rq_u64 m[4][2];
} rp_gat_t;

#ifndef RP_DEPTH
#define RP_DEPTH 2u
#endif
#ifndef RP_D_IQ2S
#define RP_D_IQ2S RP_DEPTH
#endif
#ifndef RP_D_IQ3S
#define RP_D_IQ3S RP_DEPTH
#endif
static inline u32 rp_depth(const u32 type) {
  return type == IE_RQ_IQ2_S ? RP_D_IQ2S : type == IE_RQ_IQ3_S ? RP_D_IQ3S : RP_DEPTH;
}
#ifndef RP_TT_FIRST
#define RP_TT_FIRST 1
#endif
#ifndef RP_GROUP
#define RP_GROUP 1u
#endif
#ifndef RP_G_IQ2S
#define RP_G_IQ2S 0u
#endif
#ifndef RP_G_IQ3S
#define RP_G_IQ3S RP_GROUP
#endif
static inline int rp_group(const u32 type) {
  return type == IE_RQ_Q2_K ? 0 : type == IE_RQ_IQ2_S ? RP_G_IQ2S : type == IE_RQ_IQ3_S ? RP_G_IQ3S : RP_GROUP;
}

static inline __attribute__((always_inline)) rp_raw_t rp_load(const u32 type, const G u8 *p, const u32 h) {
  rp_raw_t x;
#pragma unroll
  for (u32 i = 0; i < RP_NW; i++) x.r[i] = 0u;
  switch (type) {
    case IE_RQ_Q2_K: {
      x.r[0] = ie_rq_b32(p, 80u);
      x.r[13] = ie_rq_b32(p, 8u * h);
      x.r[14] = ie_rq_b32(p, 8u * h + 4u);
#pragma unroll
      for (u32 i = 0; i < 4u; i++) x.r[1u + i] = ie_rq_b32(p, 4u * i);
#pragma unroll
      for (u32 c = 0; c < 2u; c++) {
        const u32 qs = 16u + 32u * c + 8u * h;
        x.r[5u + 4u * c] = ie_rq_b32(p, qs);
        x.r[6u + 4u * c] = ie_rq_b32(p, qs + 4u);
        x.r[7u + 4u * c] = ie_rq_b32(p, qs + 16u);
        x.r[8u + 4u * c] = ie_rq_b32(p, qs + 20u);
      }
      break;
    }
    case IE_RQ_IQ2_XXS: {
      x.r[0] = ie_rq_b16(p, 0u);
#pragma unroll
      for (u32 s = 0; s < 8u; s++) {
        x.r[1u + 2u * s] = ie_rq_b32(p, 2u + 8u * s);
        x.r[2u + 2u * s] = ie_rq_b32(p, 6u + 8u * s);
      }
      break;
    }
    case IE_RQ_IQ2_XS: {
      x.r[0] = ie_rq_b16(p, 0u);
      x.r[1] = ie_rq_b32(p, 66u);
      x.r[2] = ie_rq_b32(p, 70u);
#pragma unroll
      for (u32 s = 0; s < 8u; s++)
#pragma unroll
        for (u32 q = 0; q < 2u; q++) x.r[3u + 2u * s + q] = ie_rq_b16(p, 2u + 8u * s + 4u * q + 2u * h);
      break;
    }
    case IE_RQ_IQ2_S: {
      x.r[0] = ie_rq_b16(p, 0u);
      x.r[1] = ie_rq_b32(p, 66u);
      x.r[2] = ie_rq_b32(p, 70u);
      x.r[3] = ie_rq_b32(p, 74u);
      x.r[4] = ie_rq_b32(p, 78u);
#pragma unroll
      for (u32 s = 0; s < 8u; s++)
#pragma unroll
        for (u32 q = 0; q < 2u; q++) {
          x.r[5u + 4u * s + 2u * q] = ie_rq_b(p, 2u + 4u * s + 2u * q + h);
          x.r[6u + 4u * s + 2u * q] = ie_rq_b(p, 34u + 4u * s + 2u * q + h);
        }
      break;
    }
    case IE_RQ_IQ3_XXS: {
      x.r[0] = ie_rq_b16(p, 0u);
#pragma unroll
      for (u32 s = 0; s < 8u; s++) x.r[1u + s] = ie_rq_b32(p, 66u + 4u * s);
#pragma unroll
      for (u32 s = 0; s < 8u; s++)
#pragma unroll
        for (u32 q = 0; q < 2u; q++) x.r[9u + 2u * s + q] = ie_rq_b16(p, 2u + 8u * s + 4u * q + 2u * h);
      break;
    }
    case IE_RQ_IQ3_S: {
      x.r[0] = ie_rq_b16(p, 0u);
      x.r[1] = ie_rq_b32(p, 66u);
      x.r[2] = ie_rq_b32(p, 70u);
      x.r[3] = ie_rq_b32(p, 106u);
#pragma unroll
      for (u32 s = 0; s < 8u; s++)
#pragma unroll
        for (u32 q = 0; q < 2u; q++) {
          x.r[4u + 2u * s + q] = ie_rq_b(p, 74u + 4u * s + 2u * q + h);
          x.r[20u + 2u * s + q] = ie_rq_b16(p, 2u + 8u * s + 4u * q + 2u * h);
        }
      break;
    }
    case IE_RQ_IQ1_S: {
      x.r[0] = ie_rq_b16(p, 0u);
#pragma unroll
      for (u32 j = 0; j < 4u; j++) x.r[1u + j] = ie_rq_b32(p, 34u + 4u * j);
#pragma unroll
      for (u32 s = 0; s < 8u; s++)
#pragma unroll
        for (u32 q = 0; q < 2u; q++) x.r[5u + 2u * s + q] = ie_rq_b(p, 2u + 4u * s + 2u * q + h);
      break;
    }
    case IE_RQ_IQ1_M: {
      x.r[0] = ie_rq_b32(p, 48u);
      x.r[1] = ie_rq_b32(p, 52u);
#pragma unroll
      for (u32 j = 0; j < 4u; j++) x.r[2u + j] = ie_rq_b32(p, 32u + 4u * j);
#pragma unroll
      for (u32 s = 0; s < 8u; s++)
#pragma unroll
        for (u32 q = 0; q < 2u; q++) x.r[6u + 2u * s + q] = ie_rq_b(p, 4u * s + 2u * q + h);
      break;
    }
    default: break;
  }
  return x;
}

static inline __attribute__((always_inline)) rp_gat_t rp_gather(const u32 type, const rp_raw_t *x, const u32 hb, const u32 h) {
  rp_gat_t o;
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
#pragma unroll
    for (u32 q = 0; q < 2u; q++) o.g[k][q] = 0, o.m[k][q] = 0;
  switch (type) {
    case IE_RQ_IQ2_XXS:
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        const u32 s = 4u * hb + k, a0 = x->r[1u + 2u * s], a1 = x->r[2u + 2u * s];
#pragma unroll
        for (u32 q = 0; q < 2u; q++) {
          const u32 g8 = 2u * q + h;
          o.m[k][q] = ie_rq_ksigns64[(a1 >> (7u * g8)) & 127u];
          o.g[k][q] = ie_rq_iq2xxs_grid[(a0 >> (8u * g8)) & 255u];
        }
      }
      break;
    case IE_RQ_IQ2_XS:
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        const u32 s = 4u * hb + k;
#pragma unroll
        for (u32 q = 0; q < 2u; q++) {
          const u32 q2 = x->r[3u + 2u * s + q];
          o.m[k][q] = ie_rq_ksigns64[q2 >> 9u];
          o.g[k][q] = ie_rq_iq2xs_grid[q2 & 511u];
        }
      }
      break;
    case IE_RQ_IQ2_S:
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        const u32 s = 4u * hb + k, qh = (x->r[1u + (s >> 2u)] >> (8u * (s & 3u))) & 255u;
#pragma unroll
        for (u32 q = 0; q < 2u; q++) {
          const u32 g8 = 2u * q + h;
          const u32 idx = x->r[5u + 4u * s + 2u * q] | ((qh << (8u - 2u * g8)) & 0x300u);
          o.g[k][q] = ie_rq_iq2s_grid[idx];
        }
      }
      break;
    case IE_RQ_IQ3_XXS:
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        const u32 s = 4u * hb + k, aux = x->r[1u + s];
#pragma unroll
        for (u32 q = 0; q < 2u; q++) {
          const u32 g8 = 2u * q + h, w = x->r[9u + 2u * s + q];
          const ie_rq_u32 g1 = ie_rq_iq3xxs_grid[w & 255u], g2 = ie_rq_iq3xxs_grid[(w >> 8u) & 255u];
          o.m[k][q] = ie_rq_ksigns64[(aux >> (7u * g8)) & 127u];
          o.g[k][q] = (ie_rq_u64)g1 | ((ie_rq_u64)g2 << 32u);
        }
      }
      break;
    case IE_RQ_IQ3_S:
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        const u32 s = 4u * hb + k, qh = (x->r[1u + (s >> 2u)] >> (8u * (s & 3u))) & 255u;
#pragma unroll
        for (u32 q = 0; q < 2u; q++) {
          const u32 g8 = 2u * q + h, w = x->r[20u + 2u * s + q];
          const u32 i1 = (w & 255u) | (((qh >> (2u * g8)) & 1u) << 8u);
          const u32 i2 = ((w >> 8u) & 255u) | (((qh >> (2u * g8 + 1u)) & 1u) << 8u);
          const ie_rq_u32 g1 = ie_rq_iq3s_grid[i1], g2 = ie_rq_iq3s_grid[i2];
          o.g[k][q] = (ie_rq_u64)g1 | ((ie_rq_u64)g2 << 32u);
        }
      }
      break;
    case IE_RQ_IQ1_S:
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        const u32 s = 4u * hb + k, qh = (x->r[1u + (s >> 1u)] >> (16u * (s & 1u))) & 0xFFFFu;
#pragma unroll
        for (u32 q = 0; q < 2u; q++) {
          const u32 g8 = 2u * q + h;
          const u32 idx = x->r[5u + 2u * s + q] | (((qh >> (3u * g8)) & 7u) << 8u);
          o.g[k][q] = ie_rq_iq1s_grid[idx];
          o.m[k][q] = (qh & 0x8000u) != 0u ? RQ_IQ1_TBL_NEG : RQ_IQ1_TBL_POS;
        }
      }
      break;
    case IE_RQ_IQ1_M:
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        const u32 s = 4u * hb + k, qhb = (x->r[2u + (s >> 1u)] >> (16u * (s & 1u))) & 0xFFFFu;
#pragma unroll
        for (u32 q = 0; q < 2u; q++) {
          const u32 qh = q == 0u ? (qhb & 255u) : (qhb >> 8u);
          const u32 idx = x->r[6u + 2u * s + q] | ((qh << (8u - 4u * h)) & 0x700u);
          o.g[k][q] = ie_rq_iq1s_grid[idx];
          o.m[k][q] = (qh & (h != 0u ? 0x80u : 0x08u)) != 0u ? RQ_IQ1_TBL_NEG : RQ_IQ1_TBL_POS;
        }
      }
      break;
    default: break;
  }
  return o;
}

static inline __attribute__((always_inline)) rq_dec_t rp_dec(const u32 type, const rp_raw_t *x, const rp_gat_t *t, const u32 hb, const u32 k,
                                                             const u32 h) {
  (void)h;
  rq_dec_t r;
  r.s0 = r.s1 = r.m0 = r.m1 = 0.0f;
  r.i0 = r.i1 = r.n0 = r.n1 = 0;
  r.fd = r.fm = 0.0f;
  r.w0 = r.w1 = (v2i)(0);
  const u32 s = 4u * hb + k;
  switch (type) {
    case IE_RQ_Q2_K: {
      const u32 sw = x->r[1u + (s >> 1u)], sc0 = (sw >> (16u * (s & 1u))) & 255u, sc1 = (sw >> (16u * (s & 1u) + 8u)) & 255u;
      const u32 c = s >> 2u, sh = 2u * (s & 3u);
      r.w0.x = (int)(((x->r[5u + 4u * c] >> sh) & 0x03030303u) * (sc0 & 15u));
      r.w0.y = (int)(((x->r[6u + 4u * c] >> sh) & 0x03030303u) * (sc0 & 15u));
      r.w1.x = (int)(((x->r[7u + 4u * c] >> sh) & 0x03030303u) * (sc1 & 15u));
      r.w1.y = (int)(((x->r[8u + 4u * c] >> sh) & 0x03030303u) * (sc1 & 15u));
      r.n0 = (int)(sc0 >> 4u), r.n1 = (int)(sc1 >> 4u);
      r.fd = ie_rq_h2f(x->r[0] & 0xFFFFu), r.fm = ie_rq_h2f(x->r[0] >> 16u);
      break;
    }
    case IE_RQ_IQ2_XXS: {
      const u32 a1 = x->r[2u + 2u * s];
      r.i0 = (int)(2u * (a1 >> 28u) + 1u), r.fd = ie_rq_h2f(x->r[0]) * 0.125f;
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const ie_rq_u64 m = t->m[k][q], g = t->g[k][q];
        const v2i v = {(int)ie_rq_ssign((ie_rq_u32)g, (ie_rq_u32)m), (int)ie_rq_ssign((ie_rq_u32)(g >> 32u), (ie_rq_u32)(m >> 32u))};
        if (q == 0u) r.w0 = v; else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ2_XS: {
      const u32 sc = (x->r[1u + (s >> 2u)] >> (8u * (s & 3u))) & 255u;
      r.i0 = (int)(2u * (sc & 15u) + 1u), r.i1 = (int)(2u * (sc >> 4u) + 1u), r.fd = ie_rq_h2f(x->r[0]) * 0.125f;
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const ie_rq_u64 m = t->m[k][q], g = t->g[k][q];
        const v2i v = {(int)ie_rq_ssign((ie_rq_u32)g, (ie_rq_u32)m), (int)ie_rq_ssign((ie_rq_u32)(g >> 32u), (ie_rq_u32)(m >> 32u))};
        if (q == 0u) r.w0 = v; else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ2_S: {
      const u32 sc = (x->r[3u + (s >> 2u)] >> (8u * (s & 3u))) & 255u;
      r.i0 = (int)(2u * (sc & 15u) + 1u), r.i1 = (int)(2u * (sc >> 4u) + 1u), r.fd = ie_rq_h2f(x->r[0]) * 0.125f;
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const ie_rq_u64 m = ie_rq_mask8(x->r[6u + 4u * s + 2u * q]), g = t->g[k][q];
        const v2i v = {(int)ie_rq_ssign((ie_rq_u32)g, (ie_rq_u32)m), (int)ie_rq_ssign((ie_rq_u32)(g >> 32u), (ie_rq_u32)(m >> 32u))};
        if (q == 0u) r.w0 = v; else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ3_XXS: {
      const u32 aux = x->r[1u + s];
      r.i0 = (int)(2u * (aux >> 28u) + 1u), r.fd = ie_rq_h2f(x->r[0]) * 0.25f;
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const ie_rq_u64 m = t->m[k][q], g = t->g[k][q];
        const v2i v = {(int)ie_rq_ssign((ie_rq_u32)g, (ie_rq_u32)m), (int)ie_rq_ssign((ie_rq_u32)(g >> 32u), (ie_rq_u32)(m >> 32u))};
        if (q == 0u) r.w0 = v; else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ3_S: {
      const u32 scb = (x->r[3] >> (4u * s)) & 15u;
      r.i0 = (int)(1u + 2u * scb), r.fd = ie_rq_h2f(x->r[0]);
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const ie_rq_u64 m = ie_rq_mask8(x->r[4u + 2u * s + q]), g = t->g[k][q];
        const v2i v = {(int)ie_rq_ssign((ie_rq_u32)g, (ie_rq_u32)m), (int)ie_rq_ssign((ie_rq_u32)(g >> 32u), (ie_rq_u32)(m >> 32u))};
        if (q == 0u) r.w0 = v; else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ1_S: {
      const u32 qh = (x->r[1u + (s >> 1u)] >> (16u * (s & 1u))) & 0xFFFFu;
      r.i0 = (int)(2u * ((qh >> 12u) & 7u) + 1u), r.fd = ie_rq_h2f(x->r[0]) * 0.125f;
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const ie_rq_u64 g = t->g[k][q];
        const u32 tbl = (u32)t->m[k][q];
        const v2i v = {(int)rq_iq1_map((ie_rq_u32)g, tbl), (int)rq_iq1_map((ie_rq_u32)(g >> 32u), tbl)};
        if (q == 0u) r.w0 = v; else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ1_M: {
      const ie_rq_u32 s0 = x->r[0] & 0xFFFFu, s1 = x->r[0] >> 16u, s2 = x->r[1] & 0xFFFFu, s3 = x->r[1] >> 16u;
      const float d = ie_rq_h2f((s0 >> 12u) | ((s1 >> 8u) & 0x00F0u) | ((s2 >> 4u) & 0x0F00u) | (s3 & 0xF000u));
      const ie_rq_u32 pair = (s & 4u) != 0u ? x->r[1] : x->r[0];
      const ie_rq_u32 scv = pair >> (16u * ((s >> 1u) & 1u)), sh = 6u * (s & 1u);
      r.i0 = (int)(2u * ((scv >> sh) & 7u) + 1u), r.i1 = (int)(2u * ((scv >> (sh + 3u)) & 7u) + 1u), r.fd = d * 0.125f;
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const ie_rq_u64 g = t->g[k][q];
        const u32 tbl = (u32)t->m[k][q];
        const v2i v = {(int)rq_iq1_map((ie_rq_u32)g, tbl), (int)rq_iq1_map((ie_rq_u32)(g >> 32u), tbl)};
        if (q == 0u) r.w0 = v; else r.w1 = v;
      }
      break;
    }
    default: break;
  }
  return r;
}

static inline __attribute__((always_inline)) void rp_math(const u32 type, const rq_dec_t *dc, const LDSP u8 *lx, const u32 hb, const u32 l,
                                                          const u32 h, v8i *ai) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (u32 j = 0; j < RI_WJ; j++) {
    if (rp_group(type)) {
      v2i fa0[4], fa1[4];
      v8i c0[4], c1[4];
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        const LDSP u8 *pa = lx + (16u * j + (l & 15u)) * RI_LS + (4u * hb + k) * 32u + 8u * h;
        fa0[k] = *(const LDSP v2i *)pa, fa1[k] = *(const LDSP v2i *)(pa + 16);
      }
#pragma unroll
      for (u32 k = 0; k < 4u; k++) c0[k] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, fa0[k], 1, dc[k].w0, z, 0);
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        if (rq_split16(type)) c1[k] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, fa1[k], 1, dc[k].w1, z, 0);
        else c0[k] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, fa1[k], 1, dc[k].w1, c0[k], 0);
      }
#pragma unroll
      for (u32 k = 0; k < 4u; k++) {
        if (rq_split16(type)) ai[j] += mul24v(c0[k], (v8i)(dc[k].i0)) + mul24v(c1[k], (v8i)(dc[k].i1));
        else ai[j] += mul24v(c0[k], (v8i)(dc[k].i0));
      }
      __asm__ volatile("" : "+v"(ai[j]));
      __builtin_amdgcn_sched_barrier(0);
      continue;
    }
#pragma unroll
    for (u32 k = 0; k < 4u; k++) {
      const LDSP u8 *pa = lx + (16u * j + (l & 15u)) * RI_LS + (4u * hb + k) * 32u + 8u * h;
      const v2i a0 = *(const LDSP v2i *)pa, a1 = *(const LDSP v2i *)(pa + 16);
      if (type == IE_RQ_Q2_K) {
        ai[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, dc[k].w0, ai[j], 0);
        ai[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, dc[k].w1, ai[j], 0);
      } else if (rq_split16(type)) {
        const v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, dc[k].w0, z, 0);
        const v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, dc[k].w1, z, 0);
        ai[j] += mul24v(c0, (v8i)(dc[k].i0)) + mul24v(c1, (v8i)(dc[k].i1));
      } else {
        v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, dc[k].w0, z, 0);
        c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, dc[k].w1, c, 0);
        ai[j] += mul24v(c, (v8i)(dc[k].i0));
      }
      __asm__ volatile("" : "+v"(ai[j]));
    }
    __builtin_amdgcn_sched_barrier(0);
  }
}

static inline __attribute__((always_inline)) rp_gat_t rp_gat_zero(void) {
  rp_gat_t o;
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
#pragma unroll
    for (u32 q = 0; q < 2u; q++) o.g[k][q] = 0, o.m[k][q] = 0;
  return o;
}

static inline __attribute__((always_inline)) void gemm_ri(const u32 type, const G u8 *q, const G u8 *xq, G u8 *y, const u32 rows, const u32 nb,
                                                          const u32 T, const G float *bias, const G u8 *res, const u32 xs, const u32 ys,
                                                          const u32 rs) {
  if (T == 0u || rows == 0u || nb < 8u) return;
#if RP_TT_FIRST
  const u32 ntr = (T + RI_T - 1u) / RI_T, nrt = (rows + RQ_R - 1u) / RQ_R, tt = wgid() % ntr, rt = wgid() / ntr;
#else
  const u32 nrt = (rows + RQ_R - 1u) / RQ_R, rt = wgid() % nrt, tt = wgid() / nrt;
#endif
  (void)nrt;
  const u32 t0 = tid(), w = t0 >> 5, l = lane(), h = l >> 4;
  const u32 tbase = tt * RI_T;
  const u32 row = rt * RQ_R + 16u * w + (l & 15u), rr = row < rows ? row : rows - 1u;
  const u32 mb = nb >> 3u, bs = ie_rq_bs(type);
  const G u8 *rowp = q + (unsigned long)rr * mb * bs;
  f8 acc[RI_WJ];
#pragma unroll
  for (u32 j = 0; j < RI_WJ; j++) acc[j] = (f8)(0.0f);
  const G u8 *xp[RI_NPC];
#pragma unroll
  for (u32 i = 0; i < RI_NPC; i++) {
    const u32 p = t0 + NT * i, tk = p >> 4, gt = tbase + tk < T ? tbase + tk : T - 1u;
    xp[i] = xq + (unsigned long)gt * xs + (p & 15u) * 16u;
  }
  const u32 gs = tbase + (t0 & (RI_T - 1u)) < T ? tbase + (t0 & (RI_T - 1u)) : T - 1u;
  const G float *sxp = (const G float *)(xq + (unsigned long)gs * xs + 32ul * nb);
  u32x4 rx[RI_NPC];
  float rsx = 0.0f;
#define RI_FETCH(qb_)                                                                                         \
  do {                                                                                                        \
    _Pragma("unroll") for (u32 i = 0; i < RI_NPC; i++) rx[i] = *(const G u32x4 *)(xp[i] + (unsigned long)(qb_) * 256ul); \
    if (t0 < RI_T) rsx = sxp[8u * (qb_)];                                                                     \
  } while (0)
#define RI_STORE(bf_)                                                                                         \
  do {                                                                                                        \
    _Pragma("unroll") for (u32 i = 0; i < RI_NPC; i++) {                                                      \
      const u32 p = t0 + NT * i;                                                                              \
      *(LDSP u32x4 *)(ri_x[bf_] + (p >> 4) * RI_LS + (p & 15u) * 16u) = rx[i];                                \
      if (type == IE_RQ_Q2_K) {                                                                               \
        const int s16 = rq_sum16i(rx[i]);                                                                     \
        ri_sh[bf_][p] = (u8)(s16 >> 4);                                                                       \
        ri_sl[bf_][p] = (u8)(s16 & 15);                                                                       \
      }                                                                                                       \
    }                                                                                                         \
    if (t0 < RI_T) ri_sx[bf_][t0] = rsx;                                                                      \
  } while (0)
  const u32 depth = rp_depth(type);
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  RI_FETCH(0u);
  RI_STORE(0u);
  barrier();
  rp_raw_t cur = rp_load(type, rowp, h);
  rp_gat_t g0 = rp_gat_zero(), g1 = rp_gat_zero();
  if (depth >= 2u) g0 = rp_gather(type, &cur, 0u, h);
#pragma unroll 1
  for (u32 qb = 0; qb < mb; qb++) {
    const u32 bf = qb & 1u;
    const int more = qb + 1u < mb;
    const u32 qn = more ? qb + 1u : qb;
    const rp_raw_t nxt = rp_load(type, rowp + (unsigned long)qn * bs, h);
    if (more) RI_FETCH(qb + 1u);
    v8i ai[RI_WJ], am[RI_WJ];
#pragma unroll
    for (u32 j = 0; j < RI_WJ; j++) ai[j] = z, am[j] = z;
    if (depth < 2u) g0 = rp_gather(type, &cur, 0u, h);
    if (depth >= 1u) g1 = rp_gather(type, &cur, 1u, h);
    {
      rq_dec_t dc[4];
#pragma unroll
      for (u32 k = 0; k < 4u; k++) dc[k] = rp_dec(type, &cur, &g0, 0u, k, h);
      __builtin_amdgcn_sched_barrier(0);
      rp_math(type, dc, ri_x[bf], 0u, l, h, ai);
    }
    if (depth >= 2u) g0 = rp_gather(type, &nxt, 0u, h);
    if (depth < 1u) g1 = rp_gather(type, &cur, 1u, h);
    float fd, fm;
    {
      rq_dec_t dc[4];
#pragma unroll
      for (u32 k = 0; k < 4u; k++) dc[k] = rp_dec(type, &cur, &g1, 1u, k, h);
      __builtin_amdgcn_sched_barrier(0);
      rp_math(type, dc, ri_x[bf], 1u, l, h, ai);
      fd = dc[0].fd, fm = dc[0].fm;
    }
    __builtin_amdgcn_sched_barrier(0);
    if (type == IE_RQ_Q2_K) {
      const v2i nbv = {(int)((cur.r[13] >> 4u) & 0x0F0F0F0Fu), (int)((cur.r[14] >> 4u) & 0x0F0F0F0Fu)};
#pragma unroll
      for (u32 j = 0; j < RI_WJ; j++) {
        const v2i ah = *(const LDSP v2i *)(ri_sh[bf] + (16u * j + (l & 15u)) * 16u + 8u * h);
        const v2i al = *(const LDSP v2i *)(ri_sl[bf] + (16u * j + (l & 15u)) * 16u + 8u * h);
        const v8i rh = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, ah, 1, nbv, z, 0);
        const v8i rl = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, al, 1, nbv, z, 0);
        am[j] = (rh << 4) + rl;
      }
    }
#pragma unroll
    for (u32 j = 0; j < RI_WJ; j++) {
      const f8 dx = *(const LDSP f8 *)(ri_sx[bf] + 16u * j + 8u * h);
      f8 t = __builtin_convertvector(ai[j], f8) * fd;
      if (type == IE_RQ_Q2_K) t -= __builtin_convertvector(am[j], f8) * fm;
      acc[j] += t * dx;
    }
    if (more) RI_STORE(bf ^ 1u);
    lbarrier();
    cur = nxt;
  }
#pragma unroll
  for (u32 j = 0; j < RI_WJ; j++)
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      const u32 t = tbase + 16u * j + 8u * h + v;
      if (t < T && row < rows)
        ((G float *)(y + (unsigned long)t * ys))[row] =
            epilogue(acc[j][v], row, bias, res ? (const G float *)(res + (unsigned long)t * rs) : (const G float *)0);
    }
#undef RI_FETCH
#undef RI_STORE
}

#define RI_KERNEL(name_, type_)                                                                               \
  KERNEL name_(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T, const G float *bias, const G u8 *res, u32 xs, \
               u32 ys, u32 rs) {                                                                              \
    gemm_ri(type_, q, xq, y, rows, nb, T, bias, res, xs, ys, rs);                                             \
  }
RI_KERNEL(ie_gemm_ri_q2k, IE_RQ_Q2_K)
RI_KERNEL(ie_gemm_ri_iq2xxs, IE_RQ_IQ2_XXS)
RI_KERNEL(ie_gemm_ri_iq2xs, IE_RQ_IQ2_XS)
RI_KERNEL(ie_gemm_ri_iq3xxs, IE_RQ_IQ3_XXS)
RI_KERNEL(ie_gemm_ri_iq1s, IE_RQ_IQ1_S)
RI_KERNEL(ie_gemm_ri_iq2s, IE_RQ_IQ2_S)
RI_KERNEL(ie_gemm_ri_iq1m, IE_RQ_IQ1_M)
RI_KERNEL(ie_gemm_ri_iq3s, IE_RQ_IQ3_S)

#endif /* !IE_MEGA */
