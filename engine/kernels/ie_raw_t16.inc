/* ie_raw_t16.inc -- single-pass matrix-core verify kernels for T = 1..16 tokens:
 *   - ie_gemm_rq_t16_<type> : 128 rows/WG x 16 tokens, all 8 MAT_RAWQ subtypes, fused bias + residual
 *   - ie_gemm_rq_t16_m      : multi-projection (up to 3 segments sharing xq: any mix of the 8 IE_RQ_* subtypes + MAT_Q8)
 *   - ie_gemm_rq_t16_swiglu : 128 rows/WG x 16 tokens, fused gate + up + SwiGLU + optional FP32 y + optional Q8_0 oq
 *
 * Tile: 128 rows/WG (8 waves x 16 rows/wave, NO split-K, NO inter-wave reduction)
 *       x N = 16 tokens (1 16x16x16_iu8 WMMA tile per wave, zero-padded in LDS for tk >= T).
 * Included inside kernels/ie_kernels.c (guarded by #ifndef IE_MEGA). */
#ifndef IE_MEGA

#ifndef LDSP
#define LDSP __attribute__((address_space(3)))
#endif

#ifndef IE_RQ_Q8_0
#define IE_RQ_Q8_0 8u /* GGML_Q8_0 / MAT_Q8 segment in ie_gemm_rq_t16_m: q = w0 (int8 qw), qs = w1 (f16 scales) */
#endif

#ifndef IE_RQ_WMMA_COMMON
#define IE_RQ_WMMA_COMMON

#define RQ_KB 4u                  /* 32-weight sub-blocks per K chunk (half a 256-weight raw block) */
#define RQ_LS (RQ_KB * 32u + 16u) /* LDS bytes per token row (144 B: 16-byte aligned, bank-conflict padded) */

#define RQ_P2(sel) __builtin_amdgcn_perm(0x00D5E7F8u, 0x002B1908u, (sel))
#define RQ_SGN_M(c, m) RQ_P2(((m) & 0x04040404u) | (c))
#define RQ_SGN_B(c, b) RQ_P2((((b) * 0x00810204u) & 0x04040404u) | (c))

#define RQ_IQ1_TBL_POS 0xF9000901u
#define RQ_IQ1_TBL_NEG 0xF70007FFu
static inline u32 rq_iq1_map(u32 g, u32 tbl) { return __builtin_amdgcn_perm(0u, tbl, g & 0x03030303u); }

static inline int rq_split16(u32 type) { return type == IE_RQ_IQ2_XS || type == IE_RQ_IQ2_S || type == IE_RQ_IQ1_M; }

static inline int rq_same_lut(u32 t0, u32 t1) {
  if (t0 == t1) return 1;
  if ((t0 == IE_RQ_IQ1_S || t0 == IE_RQ_IQ1_M) && (t1 == IE_RQ_IQ1_S || t1 == IE_RQ_IQ1_M)) return 1;
  return 0;
}

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

static inline __attribute__((always_inline)) const G u32 *rq_ap(const G u8 *p, const u32 o) {
  return (const G u32 *)(((unsigned long)(p + o)) & ~3ul);
}

static inline __attribute__((always_inline)) u32 rq_align(const u32 lo, const u32 hi, const u32 sh) {
  return __builtin_amdgcn_alignbit(hi, lo, sh);
}

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

typedef struct {
  u32 r[18];
  u32 sh;
} rq_hblk_t;

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

#define T16_R 128u
#define T16_T 16u

static inline __attribute__((always_inline)) void t16_step1(const u32 type, const rq_sub_t *dc, const v2i a0, const v2i a1,
                                                            const f8 dx, const LDSP float *lsa_k, const u32 h, f8 *acc) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  f8 t;
  if (rq_split16(type)) {
    const v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, dc->w0, z, 0);
    const v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, dc->w1, z, 0);
    t = __builtin_convertvector(c0, f8) * dc->s0 + __builtin_convertvector(c1, f8) * dc->s1;
  } else {
    v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, dc->w0, z, 0);
    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, dc->w1, c0, 0);
    t = __builtin_convertvector(c0, f8) * dc->s0;
    if (type == IE_RQ_Q2_K) {
      const f8 sa0 = *(const LDSP f8 *)(lsa_k + 8u * h);
      const f8 sa1 = *(const LDSP f8 *)(lsa_k + T16_T + 8u * h);
      t -= sa0 * dc->m0;
      t -= sa1 * dc->m1;
    }
  }
  *acc += t * dx;
  __asm__ volatile("" : "+v"(*acc));
}

/* Single-segment MAT_RAWQ T<=16 pipelined engine (type is a compile-time constant in every caller). */
static inline __attribute__((always_inline)) void t16_run_raw(const u32 type, const G u8 *q, const G u8 *xq, G u8 *y,
                                                              const u32 rbase, const u32 rows, const u32 nb, const u32 T,
                                                              const G float *bias, const G u8 *res, const u32 xs, const u32 ys,
                                                              const u32 rs, LDSP u8 (*lx_b)[T16_T * RQ_LS],
                                                              LDSP float (*lsx_b)[RQ_KB * T16_T], LDSP float *lsa_b0,
                                                              LDSP float *lsa_b1, LDSP ie_rq_u64 *ksigns, LDSP u32 *lut) {
  const u32 t0 = tid(), w = t0 >> 5u, l = lane(), h = l >> 4u;
  const u32 row = rbase + 16u * w + (l & 15u), rr = row < rows ? row : rows - 1u;
  const u32 mb = nb >> 3u, bs = ie_rq_bs(type);
  const G u8 *rowp = q + (unsigned long)rr * mb * bs;

  /* Waves 0..3 (t0 < 128) stage the 128 16-byte activation pieces; wave 4 lanes 0..15 (128..143) stage the 16 f32x4 da scales. */
  const u32 tk_ld = (t0 & 127u) >> 3u, pc_ld = t0 & 7u, gt_ld = tk_ld < T ? tk_ld : T - 1u;
  const G u8 *xp = xq + (unsigned long)gt_ld * xs + pc_ld * 16u;
  const u32 tk_sx = l & 15u, gs_sx = tk_sx < T ? tk_sx : T - 1u;
  const G float *sxp = (const G float *)(xq + (unsigned long)gs_sx * xs + 32ul * nb);

  rq_stage_lut_raw(type, ksigns, lut, 1);

  u32x4 rx = t0 < 128u ? *(const G u32x4 *)xp : (u32x4)(0);
  f32x4 rsx = (w == 4u && l < 16u) ? rq_load_f32x4(sxp) : (f32x4)(0.0f);
  if (t0 < 128u) {
    *(LDSP u32x4 *)(lx_b[0] + tk_ld * RQ_LS + pc_ld * 16u) = rx;
    if (type == IE_RQ_Q2_K) lsa_b0[pc_ld * T16_T + tk_ld] = rq_sum16(rx);
  }
  if (w == 4u && l < 16u) {
    lsx_b[0][0u * T16_T + tk_sx] = rsx.x;
    lsx_b[0][1u * T16_T + tk_sx] = rsx.y;
    lsx_b[0][2u * T16_T + tk_sx] = rsx.z;
    lsx_b[0][3u * T16_T + tk_sx] = rsx.w;
  }

  rq_hblk_t cur = rq_hload(type, rowp, 0u, h);
  barrier();

  f8 acc = (f8)(0.0f);
  const u32 nc = 2u * mb;
#pragma unroll 1
  for (u32 c = 0; c < nc; c++) {
    const u32 bf = c & 1u, hb = c & 1u;
    const int more = c + 1u < nc;
    const u32 cn = more ? c + 1u : c;
    const rq_hblk_t nxt = rq_hload(type, rowp + (unsigned long)(cn >> 1u) * bs, cn & 1u, h);
    if (more) {
      const u32 kb = 4u * (c + 1u);
      if (t0 < 128u) rx = *(const G u32x4 *)(xp + kb * 32u);
      if (w == 4u && l < 16u) rsx = rq_load_f32x4(sxp + kb);
    }
    __builtin_amdgcn_sched_barrier(0);

    {
      rq_sub_t dc[RQ_KB];
#pragma unroll
      for (u32 k = 0; k < RQ_KB; k++) dc[k] = rq_hdec(type, &cur, ksigns, lut, hb, k, h);
      const LDSP u8 *lx = lx_b[bf];
      const LDSP float *lsx = lsx_b[bf];
      const LDSP float *lsa = bf ? lsa_b1 : lsa_b0;
#pragma unroll
      for (u32 k = 0; k < RQ_KB; k++) {
        const LDSP u8 *pa = lx + (l & 15u) * RQ_LS + k * 32u + 8u * h;
        const v2i a0 = *(const LDSP v2i *)pa, a1 = *(const LDSP v2i *)(pa + 16u);
        const f8 dx = *(const LDSP f8 *)(lsx + k * T16_T + 8u * h);
        t16_step1(type, &dc[k], a0, a1, dx, type == IE_RQ_Q2_K ? (lsa + (2u * k) * T16_T) : (const LDSP float *)0, h, &acc);
      }
    }

    if (more) {
      const u32 nb_f = bf ^ 1u;
      if (t0 < 128u) {
        *(LDSP u32x4 *)(lx_b[nb_f] + tk_ld * RQ_LS + pc_ld * 16u) = rx;
        if (type == IE_RQ_Q2_K) (nb_f ? lsa_b1 : lsa_b0)[pc_ld * T16_T + tk_ld] = rq_sum16(rx);
      }
      if (w == 4u && l < 16u) {
        lsx_b[nb_f][0u * T16_T + tk_sx] = rsx.x;
        lsx_b[nb_f][1u * T16_T + tk_sx] = rsx.y;
        lsx_b[nb_f][2u * T16_T + tk_sx] = rsx.z;
        lsx_b[nb_f][3u * T16_T + tk_sx] = rsx.w;
      }
      lbarrier();
    }
    cur = nxt;
  }

#pragma unroll
  for (u32 v = 0; v < 8u; v++) {
    const u32 t = 8u * h + v;
    if (t < T && row < rows) {
      const G float *rp = res ? (const G float *)(res + (unsigned long)t * rs) : (const G float *)0;
      ((G float *)(y + (unsigned long)t * ys))[row] = epilogue(acc[v], row, bias, rp);
    }
  }
}

/* Single-segment MAT_Q8 (Q8_0 repacked: q = qw bytes, qs = f16 scales) T<=16 pipelined engine for ie_gemm_rq_t16_m. */
typedef struct {
  v2i w0[RQ_KB], w1[RQ_KB];
  float sw[RQ_KB];
} t16_q8_hblk_t;

static inline __attribute__((always_inline)) t16_q8_hblk_t t16_q8_load(const G u8 *qw_row, const G f16 *qs_row, const u32 kb0,
                                                                       const u32 h) {
  t16_q8_hblk_t o;
#pragma unroll
  for (u32 k = 0; k < RQ_KB; k++) {
    const G u8 *pw = qw_row + (unsigned long)(kb0 + k) * 32ul;
    o.w0[k] = *(const G v2i *)(pw + 8u * h);
    o.w1[k] = *(const G v2i *)(pw + 16u + 8u * h);
    o.sw[k] = (float)qs_row[kb0 + k];
  }
  return o;
}

static inline __attribute__((always_inline)) void t16_run_q8(const G u8 *qw, const G f16 *qs, const G u8 *xq, G u8 *y,
                                                             const u32 rbase, const u32 rows, const u32 nb, const u32 T,
                                                             const G float *bias, const G u8 *res, const u32 xs, const u32 ys,
                                                             const u32 rs, LDSP u8 (*lx_b)[T16_T * RQ_LS],
                                                             LDSP float (*lsx_b)[RQ_KB * T16_T]) {
  const u32 t0 = tid(), w = t0 >> 5u, l = lane(), h = l >> 4u;
  const u32 row = rbase + 16u * w + (l & 15u), rr = row < rows ? row : rows - 1u;
  const G u8 *qw_row = qw + (unsigned long)rr * nb * 32ul;
  const G f16 *qs_row = qs + (unsigned long)rr * nb;

  const u32 tk_ld = (t0 & 127u) >> 3u, pc_ld = t0 & 7u, gt_ld = tk_ld < T ? tk_ld : T - 1u;
  const G u8 *xp = xq + (unsigned long)gt_ld * xs + pc_ld * 16u;
  const u32 tk_sx = l & 15u, gs_sx = tk_sx < T ? tk_sx : T - 1u;
  const G float *sxp = (const G float *)(xq + (unsigned long)gs_sx * xs + 32ul * nb);

  u32x4 rx = t0 < 128u ? *(const G u32x4 *)xp : (u32x4)(0);
  f32x4 rsx = (w == 4u && l < 16u) ? rq_load_f32x4(sxp) : (f32x4)(0.0f);
  if (t0 < 128u) *(LDSP u32x4 *)(lx_b[0] + tk_ld * RQ_LS + pc_ld * 16u) = rx;
  if (w == 4u && l < 16u) {
    lsx_b[0][0u * T16_T + tk_sx] = rsx.x;
    lsx_b[0][1u * T16_T + tk_sx] = rsx.y;
    lsx_b[0][2u * T16_T + tk_sx] = rsx.z;
    lsx_b[0][3u * T16_T + tk_sx] = rsx.w;
  }

  t16_q8_hblk_t cur = t16_q8_load(qw_row, qs_row, 0u, h);
  barrier();

  f8 acc = (f8)(0.0f);
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  const u32 nc = nb >> 2u;
#pragma unroll 1
  for (u32 c = 0; c < nc; c++) {
    const u32 bf = c & 1u;
    const int more = c + 1u < nc;
    const u32 cn = more ? c + 1u : c;
    const t16_q8_hblk_t nxt = t16_q8_load(qw_row, qs_row, 4u * cn, h);
    if (more) {
      const u32 kb = 4u * (c + 1u);
      if (t0 < 128u) rx = *(const G u32x4 *)(xp + kb * 32u);
      if (w == 4u && l < 16u) rsx = rq_load_f32x4(sxp + kb);
    }

    const LDSP u8 *lx = lx_b[bf];
    const LDSP float *lsx = lsx_b[bf];
#pragma unroll
    for (u32 k = 0; k < RQ_KB; k++) {
      const LDSP u8 *pa = lx + (l & 15u) * RQ_LS + k * 32u + 8u * h;
      const v2i a0 = *(const LDSP v2i *)pa, a1 = *(const LDSP v2i *)(pa + 16u);
      const f8 dx = *(const LDSP f8 *)(lsx + k * T16_T + 8u * h);
      v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, cur.w0[k], z, 0);
      c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, cur.w1[k], c0, 0);
      acc += __builtin_convertvector(c0, f8) * (cur.sw[k] * dx);
      __asm__ volatile("" : "+v"(acc));
    }
    __builtin_amdgcn_sched_barrier(0);

    if (more) {
      const u32 nb_f = bf ^ 1u;
      if (t0 < 128u) *(LDSP u32x4 *)(lx_b[nb_f] + tk_ld * RQ_LS + pc_ld * 16u) = rx;
      if (w == 4u && l < 16u) {
        lsx_b[nb_f][0u * T16_T + tk_sx] = rsx.x;
        lsx_b[nb_f][1u * T16_T + tk_sx] = rsx.y;
        lsx_b[nb_f][2u * T16_T + tk_sx] = rsx.z;
        lsx_b[nb_f][3u * T16_T + tk_sx] = rsx.w;
      }
      lbarrier();
    }
    cur = nxt;
  }

#pragma unroll
  for (u32 v = 0; v < 8u; v++) {
    const u32 t = 8u * h + v;
    if (t < T && row < rows) {
      const G float *rp = res ? (const G float *)(res + (unsigned long)t * rs) : (const G float *)0;
      ((G float *)(y + (unsigned long)t * ys))[row] = epilogue(acc[v], row, bias, rp);
    }
  }
}

/* ============================================================================
 * D2a: Per-subtype Single-Pass Verify Kernels: ie_gemm_rq_t16_<type>
 *   - Grid: (rows + 127) / 128 workgroups of 256 threads (T = 1..16).
 * ========================================================================== */
#define T16_KERNEL_Q2K(name_)                                                                                                     \
  KERNEL name_(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T, const G float *bias, const G u8 *res, u32 xs,     \
               u32 ys, u32 rs) {                                                                                                  \
    static LDS u8 lx[2][T16_T * RQ_LS] __attribute__((aligned(16)));                                                              \
    static LDS float lsx[2][RQ_KB * T16_T] __attribute__((aligned(32)));                                                          \
    static LDS float lsa[2][2u * RQ_KB * T16_T] __attribute__((aligned(32)));                                                     \
    const u32 rbase = wgid() * T16_R;                                                                                             \
    if (T == 0u || T > T16_T || rows == 0u || rbase >= rows || nb < 8u || (nb & 7u) != 0u || !q || !xq || !y) return;             \
    t16_run_raw(IE_RQ_Q2_K, q, xq, y, rbase, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, lsa[0], lsa[1], (LDSP ie_rq_u64 *)0,    \
                (LDSP u32 *)0);                                                                                                   \
  }

#define T16_KERNEL_KS(name_, type_, nlut_)                                                                                        \
  KERNEL name_(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T, const G float *bias, const G u8 *res, u32 xs,     \
               u32 ys, u32 rs) {                                                                                                  \
    static LDS u8 lx[2][T16_T * RQ_LS] __attribute__((aligned(16)));                                                              \
    static LDS float lsx[2][RQ_KB * T16_T] __attribute__((aligned(32)));                                                          \
    static LDS u32 lut[nlut_] __attribute__((aligned(16)));                                                                       \
    const u32 rbase = wgid() * T16_R;                                                                                             \
    if (T == 0u || T > T16_T || rows == 0u || rbase >= rows || nb < 8u || (nb & 7u) != 0u || !q || !xq || !y) return;             \
    t16_run_raw(type_, q, xq, y, rbase, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, (LDSP float *)0, (LDSP float *)0,            \
                (LDSP ie_rq_u64 *)0, lut);                                                                                        \
  }

#define T16_KERNEL_NK(name_, type_, nlut_)                                                                                        \
  KERNEL name_(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T, const G float *bias, const G u8 *res, u32 xs,     \
               u32 ys, u32 rs) {                                                                                                  \
    static LDS u8 lx[2][T16_T * RQ_LS] __attribute__((aligned(16)));                                                              \
    static LDS float lsx[2][RQ_KB * T16_T] __attribute__((aligned(32)));                                                          \
    static LDS u32 lut[nlut_] __attribute__((aligned(16)));                                                                       \
    const u32 rbase = wgid() * T16_R;                                                                                             \
    if (T == 0u || T > T16_T || rows == 0u || rbase >= rows || nb < 8u || (nb & 7u) != 0u || !q || !xq || !y) return;             \
    t16_run_raw(type_, q, xq, y, rbase, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, (LDSP float *)0, (LDSP float *)0,            \
                (LDSP ie_rq_u64 *)0, lut);                                                                                        \
  }

T16_KERNEL_Q2K(ie_gemm_rq_t16_q2k)
T16_KERNEL_KS(ie_gemm_rq_t16_iq2xxs, IE_RQ_IQ2_XXS, 256u)
T16_KERNEL_KS(ie_gemm_rq_t16_iq2xs, IE_RQ_IQ2_XS, 512u)
T16_KERNEL_KS(ie_gemm_rq_t16_iq3xxs, IE_RQ_IQ3_XXS, 256u)
T16_KERNEL_NK(ie_gemm_rq_t16_iq1s, IE_RQ_IQ1_S, 2048u)
T16_KERNEL_NK(ie_gemm_rq_t16_iq2s, IE_RQ_IQ2_S, 1024u)
T16_KERNEL_NK(ie_gemm_rq_t16_iq1m, IE_RQ_IQ1_M, 2048u)
T16_KERNEL_NK(ie_gemm_rq_t16_iq3s, IE_RQ_IQ3_S, 512u)

#undef T16_KERNEL_NK
#undef T16_KERNEL_KS
#undef T16_KERNEL_Q2K

/* ============================================================================
 * D2b: Multi-Projection Single-Pass Verify: ie_gemm_rq_t16_m
 *   - Up to 3 segments sharing xq: any mix of the 8 IE_RQ_* subtypes plus
 *     optional IE_RQ_Q8_0 (8u = MAT_Q8, e.g. GDN's 96-row wba).
 *   - Grid: g1 = ceil(rows0 / 128), g2 = g1 + ceil(rows1 / 128),
 *           tot = g2 + ceil(rows2 / 128) workgroups of 256 threads.
 *   - Total LDS: 14.0 KiB -> 8 WGs/WGP (maximum hardware occupancy).
 * ========================================================================== */
KERNEL ie_gemm_rq_t16_m(const G u8 *xq, u32 nb, u32 T, u32 xs, u32 g1, u32 g2,
                        const G u8 *q0, G u8 *y0, u32 rows0, u32 type0, u32 ys0, const G u8 *res0, u32 rs0, const G float *bias0,
                        const G u8 *q1, G u8 *y1, u32 rows1, u32 type1, u32 ys1, const G u8 *res1, u32 rs1, const G float *bias1,
                        const G u8 *q2, G u8 *y2, u32 rows2, u32 type2, u32 ys2, const G u8 *res2, u32 rs2, const G float *bias2,
                        const G f16 *qs0, const G f16 *qs1, const G f16 *qs2) {
  static LDS u8 lx[2][T16_T * RQ_LS] __attribute__((aligned(16)));
  static LDS float lsx[2][RQ_KB * T16_T] __attribute__((aligned(32)));
  static LDS ie_rq_u64 ksigns[128] __attribute__((aligned(16)));
  static LDS u32 lut[2048] __attribute__((aligned(16))); /* also holds lsa[2][128] when segment type == IE_RQ_Q2_K */

  const u32 wg = wgid();
  const G u8 *q;
  const G f16 *qs;
  G u8 *y;
  const G u8 *res;
  const G float *bias;
  u32 rows, type, ys, rs, rbase;
  if (wg < g1) {
    q = q0, qs = qs0, y = y0, rows = rows0, type = type0, ys = ys0, res = res0, rs = rs0, bias = bias0;
    rbase = wg * T16_R;
  } else if (wg < g2) {
    q = q1, qs = qs1, y = y1, rows = rows1, type = type1, ys = ys1, res = res1, rs = rs1, bias = bias1;
    rbase = (wg - g1) * T16_R;
  } else {
    q = q2, qs = qs2, y = y2, rows = rows2, type = type2, ys = ys2, res = res2, rs = rs2, bias = bias2;
    rbase = (wg - g2) * T16_R;
  }
  type = (u32)__builtin_amdgcn_readfirstlane(type);
  if (T == 0u || T > T16_T || rows == 0u || rbase >= rows || nb < 8u || (nb & 7u) != 0u || !q || !xq || !y) return;

  LDSP float *const lsa0 = (LDSP float *)lut;
  LDSP float *const lsa1 = lsa0 + 2u * RQ_KB * T16_T;

  switch (type) {
    case IE_RQ_Q8_0:
      if (!qs) return;
      t16_run_q8(q, qs, xq, y, rbase, rows, nb, T, bias, res, xs, ys, rs, lx, lsx);
      break;
    case IE_RQ_Q2_K:
      t16_run_raw(IE_RQ_Q2_K, q, xq, y, rbase, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, lsa0, lsa1, (LDSP ie_rq_u64 *)0,
                  (LDSP u32 *)0);
      break;
    case IE_RQ_IQ2_XXS:
      t16_run_raw(IE_RQ_IQ2_XXS, q, xq, y, rbase, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, (LDSP float *)0, (LDSP float *)0,
                  ksigns, lut);
      break;
    case IE_RQ_IQ2_XS:
      t16_run_raw(IE_RQ_IQ2_XS, q, xq, y, rbase, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, (LDSP float *)0, (LDSP float *)0,
                  ksigns, lut);
      break;
    case IE_RQ_IQ3_XXS:
      t16_run_raw(IE_RQ_IQ3_XXS, q, xq, y, rbase, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, (LDSP float *)0, (LDSP float *)0,
                  ksigns, lut);
      break;
    case IE_RQ_IQ1_S:
      t16_run_raw(IE_RQ_IQ1_S, q, xq, y, rbase, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, (LDSP float *)0, (LDSP float *)0,
                  (LDSP ie_rq_u64 *)0, lut);
      break;
    case IE_RQ_IQ2_S:
      t16_run_raw(IE_RQ_IQ2_S, q, xq, y, rbase, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, (LDSP float *)0, (LDSP float *)0,
                  (LDSP ie_rq_u64 *)0, lut);
      break;
    case IE_RQ_IQ1_M:
      t16_run_raw(IE_RQ_IQ1_M, q, xq, y, rbase, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, (LDSP float *)0, (LDSP float *)0,
                  (LDSP ie_rq_u64 *)0, lut);
      break;
    case IE_RQ_IQ3_S:
      t16_run_raw(IE_RQ_IQ3_S, q, xq, y, rbase, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, (LDSP float *)0, (LDSP float *)0,
                  (LDSP ie_rq_u64 *)0, lut);
      break;
    default:
      break;
  }
}

/* ============================================================================
 * D2c: Fused Dual-Matrix Verify SwiGLU: ie_gemm_rq_t16_swiglu (T = 1..16)
 *   - Tile: 128 rows/WG x 16 tokens (grid: ceil(rows / 128) WGs of 256 threads).
 *   - Supports BOTH same-subtype and mixed-subtype (type0 != type1) in ONE pass.
 *   - Total LDS: 18.0 KiB -> 7 WGs/WGP.
 * ========================================================================== */
typedef union {
  struct {
    u8 x[2][T16_T * RQ_LS] __attribute__((aligned(16)));
    float sx[2][RQ_KB * T16_T] __attribute__((aligned(32)));
    ie_rq_u64 ksigns[128] __attribute__((aligned(16)));
    u32 lut[3072] __attribute__((aligned(16))); /* [0..2047] primary LUT, [2048..3071] secondary LUT or Q2_K sa[2][128] */
  } k;
  float out[T16_T][T16_R + 4u];
} t16_swg_lds_t;

KERNEL ie_gemm_rq_t16_swiglu(const G u8 *q0, u32 type0, const G u8 *q1, u32 type1, const G u8 *xq,
                             G float *y, G u8 *oq, u32 rows, u32 nb, u32 T, u32 xs, u32 ys, u32 qs) {
  static LDS t16_swg_lds_t swg;
  if (T == 0u || T > T16_T || rows == 0u || (rows & 31u) != 0u || nb < 8u || (nb & 7u) != 0u || !q0 || !q1 || !xq) return;
  type0 = (u32)__builtin_amdgcn_readfirstlane(type0);
  type1 = (u32)__builtin_amdgcn_readfirstlane(type1);
  const u32 bs0 = ie_rq_bs(type0), bs1 = ie_rq_bs(type1);
  if (bs0 == 0u || bs1 == 0u) return;

  const u32 rbase = wgid() * T16_R;
  if (rbase >= rows) return;

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
  LDSP float *const lsa1 = lsa0 + 2u * RQ_KB * T16_T;

  rq_stage_lut_raw(type0, swg.k.ksigns, lut0, 1);
  if (!same) {
    const int ks0 = (type0 == IE_RQ_IQ2_XXS || type0 == IE_RQ_IQ2_XS || type0 == IE_RQ_IQ3_XXS);
    rq_stage_lut_raw(type1, swg.k.ksigns, lut1, !ks0);
  }

  const u32 tk_ld = (t0 & 127u) >> 3u, pc_ld = t0 & 7u, gt_ld = tk_ld < T ? tk_ld : T - 1u;
  const G u8 *xp = xq + (unsigned long)gt_ld * xs + pc_ld * 16u;
  const u32 tk_sx = l & 15u, gs_sx = tk_sx < T ? tk_sx : T - 1u;
  const G float *sxp = (const G float *)(xq + (unsigned long)gs_sx * xs + 32ul * nb);

  u32x4 rx = t0 < 128u ? *(const G u32x4 *)xp : (u32x4)(0);
  f32x4 rsx = (w == 4u && l < 16u) ? rq_load_f32x4(sxp) : (f32x4)(0.0f);
  if (t0 < 128u) {
    *(LDSP u32x4 *)(swg.k.x[0] + tk_ld * RQ_LS + pc_ld * 16u) = rx;
    if (has_q2k) lsa0[pc_ld * T16_T + tk_ld] = rq_sum16(rx);
  }
  if (w == 4u && l < 16u) {
    swg.k.sx[0][0u * T16_T + tk_sx] = rsx.x;
    swg.k.sx[0][1u * T16_T + tk_sx] = rsx.y;
    swg.k.sx[0][2u * T16_T + tk_sx] = rsx.z;
    swg.k.sx[0][3u * T16_T + tk_sx] = rsx.w;
  }

  rq_hblk_t cur0 = rq_hload(type0, rowp0, 0u, h);
  rq_hblk_t cur1 = rq_hload(type1, rowp1, 0u, h);
  barrier();

  f8 g_acc = (f8)(0.0f), u_acc = (f8)(0.0f);
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
      if (t0 < 128u) rx = *(const G u32x4 *)(xp + kb * 32u);
      if (w == 4u && l < 16u) rsx = rq_load_f32x4(sxp + kb);
    }
    __builtin_amdgcn_sched_barrier(0);

    const LDSP u8 *lx = swg.k.x[bf];
    const LDSP float *lsx = swg.k.sx[bf];
    const LDSP float *lsa = bf ? lsa1 : lsa0;
#pragma unroll
    for (u32 k = 0; k < RQ_KB; k++) {
      const LDSP u8 *pa = lx + (l & 15u) * RQ_LS + k * 32u + 8u * h;
      const v2i a0 = *(const LDSP v2i *)pa, a1 = *(const LDSP v2i *)(pa + 16u);
      const f8 dx = *(const LDSP f8 *)(lsx + k * T16_T + 8u * h);
      const LDSP float *lsa_k = lsa + (2u * k) * T16_T;
      const rq_sub_t dc0 = rq_hdec(type0, &cur0, swg.k.ksigns, lut0, hb, k, h);
      const rq_sub_t dc1 = rq_hdec(type1, &cur1, swg.k.ksigns, lut1, hb, k, h);
      t16_step1(type0, &dc0, a0, a1, dx, lsa_k, h, &g_acc);
      t16_step1(type1, &dc1, a0, a1, dx, lsa_k, h, &u_acc);
    }

    if (more) {
      const u32 nb_f = bf ^ 1u;
      if (t0 < 128u) {
        *(LDSP u32x4 *)(swg.k.x[nb_f] + tk_ld * RQ_LS + pc_ld * 16u) = rx;
        if (has_q2k) (nb_f ? lsa1 : lsa0)[pc_ld * T16_T + tk_ld] = rq_sum16(rx);
      }
      if (w == 4u && l < 16u) {
        swg.k.sx[nb_f][0u * T16_T + tk_sx] = rsx.x;
        swg.k.sx[nb_f][1u * T16_T + tk_sx] = rsx.y;
        swg.k.sx[nb_f][2u * T16_T + tk_sx] = rsx.z;
        swg.k.sx[nb_f][3u * T16_T + tk_sx] = rsx.w;
      }
      lbarrier();
    }
    cur0 = nxt0;
    cur1 = nxt1;
  }

  f8 s_val;
#pragma unroll
  for (u32 v = 0; v < 8u; v++) {
    const float g = g_acc[v], u = u_acc[v];
    const u32 t = 8u * h + v;
    const float sv = (t < T && row < rows) ? (g / (1.0f + __builtin_expf(-g))) * u : 0.0f;
    s_val[v] = sv;
    if (y && t < T && row < rows) {
      ((G float *)((G u8 *)y + (unsigned long)t * ys))[row] = sv;
    }
  }

  if (oq) {
    lbarrier();
    const u32 rl = 16u * w + (l & 15u);
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      swg.out[8u * h + v][rl] = s_val[v];
    }
    lbarrier();
    const u32 rem_b = (rows - rbase) >> 5u;
    const u32 nblk = rem_b < 4u ? rem_b : 4u;
    const u32 total_q = T * nblk, nb_out = rows >> 5u;
    for (u32 idx = w; idx < total_q; idx += 8u) {
      const u32 tk = idx / nblk, blk = idx % nblk;
      const float vq = swg.out[tk][32u * blk + l];
      quant_wave(vq, (rbase >> 5u) + blk, oq + (unsigned long)tk * qs, nb_out);
    }
  }
}

#undef T16_T
#undef T16_R

#endif /* !IE_MEGA */
