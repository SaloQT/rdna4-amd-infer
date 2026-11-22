/* ============================================================================
 * Round 11 Group 4: ie_saluki_qkn_rope_kv_r11
 *   Bitwise-exact, 0-LDS, 0-barrier, 0-spill QK-RMSNorm + RoPE + Q8_0 KV Store
 *   with optional fused scaled-FP16 Q16 emission (eliminating ie_q_scale_f16
 *   and dead 12 MiB F32 qo writes during prefill) on RX 9070 XT (gfx1201).
 *
 * Replaces ie_qkn_rope_kv (+ ie_q_scale_f16 in prefill) across all phases
 * (prefill T=17..512, mtp_kv kv_only, verify T=4..16, tree verify, MTP T=1..8):
 *   1. 1 wave (32 lanes) per 256-dim head (8 heads/WG, 0 intra-wave divergence)
 *      with 128-bit coalesced f32x4 loads/stores (2x global_load_b128/head)
 *      instead of 1 WG per head with scalar b32 loads/stores and 1-byte b8 KV
 *      stores (eliminates 16,384 tiny WGs/launch at T=512 and 256-WG grid at
 *      T=8 verify).
 *   2. Zero LDS (0 B vs 256+32 B) and zero barriers (eliminates ALL 3 global
 *      L0-invalidating barrier() calls per head -> 49,152 -> 0 buffer_gl0_inv
 *      per T=512 launch, keeping qw/kw/cs/sn 100% resident in L0/L1 cache).
 *   3. Pure VALU register-only reductions & RoPE exchanges:
 *      - Exact 5-level binary tree sum-of-squares per 32-elem block via
 *        v_permlane16_b32 (lane^4) + DPP 0x4E (lane^2) + DPP 0xB1 (lane^1) +
 *        in-register stride-2/1 adds, followed by readlane(0,8,16,24) exact
 *        sequential block 0..7 accumulation matching base wg_sum(x*x) bit-for-bit.
 *      - Pure VALU RoPE partner exchange via v_permlane16_b32 (lane^8 for
 *        nrot=64) / v_permlanex16_b32 (lane^16 for nrot=128) / in-lane
 *        (nrot=256), preserving exact base LLVM IR fmaf(a,cs,b*(-sn)) and
 *        fmaf(a,sn,b*cs) rounding.
 *   4. Coalesced Q8_0 KV store: pure VALU 8-lane subgroup max (DPP 0xB1/0x4E +
 *      v_permlane16_b32 lane^4), packed u32 (4 int8/lane -> global_store_b32)
 *      plus single readlane-gathered 128-bit u32x4 scale store (1 vs 8 b16).
 *   5. Adaptive multi-token software pipeline (tpw=4 for T>=128, tpw=1 for
 *      T<128): reuses register-resident qw/kw across 4 tokens, pipelines
 *      next-token x0/x1 loads behind current-token cs/sn in the VMEM FIFO, and
 *      optionally writes scaled f16 Q16 (global_store_b64) while eliding dead
 *      F32 qo writes when qo == NULL (cutting T=512 QKV-prep traffic 47.1 ->
 *      23.1 MiB/layer and removing 1,104 ie_q_scale_f16 launches).
 *
 * Included at the end of kernels/ie_saluki_kernels.c.
 * All local types, helpers, and macros are prefixed r11_g04_ / R11_G04_,
 * and all macros are #undef'd at EOF.
 * ========================================================================== */

#define R11_G04_ROWB   272u
#define R11_G04_QSCALE (0.0625f * 1.44269504088896341f)
#define R11_G04_TPW(T) ((T) >= 128u ? 4u : 1u)

typedef _Float16 r11_g04_h4 __attribute__((ext_vector_type(4)));

_Static_assert(KVR(256u) == R11_G04_ROWB, "r11_g04 requires q8_0 KV record size 272 B");
_Static_assert(sizeof(r11_g04_h4) == 8u, "r11_g04_h4 must be 8 B");

/* Pure VALU intra-wave lane swaps (always invoked with EXEC == 0xFFFFFFFF on scalar locals). */
static inline __attribute__((always_inline)) float r11_g04_dpp_x1_f(float v) {
  const int b = __builtin_bit_cast(int, v);
  return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, b, 0xB1, 0xF, 0xF, true));
}

static inline __attribute__((always_inline)) float r11_g04_dpp_x2_f(float v) {
  const int b = __builtin_bit_cast(int, v);
  return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, b, 0x4E, 0xF, 0xF, true));
}

static inline __attribute__((always_inline)) float r11_g04_x4_f(float v) {
  const int b = __builtin_bit_cast(int, v);
  return __builtin_bit_cast(float, __builtin_amdgcn_permlane16(b, b, (int)0x32107654u, (int)0xBA98FEDCu, false, false));
}

static inline __attribute__((always_inline)) float r11_g04_x8_f(float v) {
  const int b = __builtin_bit_cast(int, v);
  return __builtin_bit_cast(float, __builtin_amdgcn_permlane16(b, b, (int)0xFEDCBA98u, (int)0x76543210u, false, false));
}

static inline __attribute__((always_inline)) float r11_g04_x16_f(float v) {
  const int b = __builtin_bit_cast(int, v);
  return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(b, b, (int)0x76543210u, (int)0xFEDCBA98u, false, false));
}

/* Exact 5-level binary tree sum of x^2 across each 8-lane (32-float) subgroup.
 * Lanes 0, 8, 16, 24 receive the exact wave_tree_f(x*x) sums of blocks 0..3 (or 4..7). */
static inline __attribute__((always_inline)) float r11_g04_sub8_ss(f32x4 x) {
#pragma clang fp contract(off)
  const float x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
  float p0 = x0 * x0;
  float p1 = x1 * x1;
  float p2 = x2 * x2;
  float p3 = x3 * x3;
  /* Level 1: stride 16 (lane sl <-> sl ^ 4) */
  p0 = p0 + r11_g04_x4_f(p0);
  p1 = p1 + r11_g04_x4_f(p1);
  p2 = p2 + r11_g04_x4_f(p2);
  p3 = p3 + r11_g04_x4_f(p3);
  /* Level 2: stride 8 (lane sl <-> sl ^ 2) */
  p0 = p0 + r11_g04_dpp_x2_f(p0);
  p1 = p1 + r11_g04_dpp_x2_f(p1);
  p2 = p2 + r11_g04_dpp_x2_f(p2);
  p3 = p3 + r11_g04_dpp_x2_f(p3);
  /* Level 3: stride 4 (lane sl <-> sl ^ 1) */
  p0 = p0 + r11_g04_dpp_x1_f(p0);
  p1 = p1 + r11_g04_dpp_x1_f(p1);
  p2 = p2 + r11_g04_dpp_x1_f(p2);
  p3 = p3 + r11_g04_dpp_x1_f(p3);
  /* Level 4 & 5: stride 2 and stride 1 in registers */
  const float d0 = p0 + p2;
  const float d1 = p1 + p3;
  return d0 + d1;
}

/* Pre-issue RoPE cos/sin 128-bit loads for token tk ahead of next-token x prefetch. */
static inline __attribute__((always_inline)) void r11_g04_load_rope_cs(
    const G float *cs, const G float *sn, u32 tk, u32 nrot, u32 l, f32x4 *c4o, f32x4 *s4o) {
  f32x4 c4 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 s4 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (nrot == 64u) {
    if (l < 16u) {
      const u32 i4 = l & 7u;
      const unsigned long off = (unsigned long)tk * 32ul + (unsigned long)(4u * i4);
      c4 = *(const G f32x4 *)(cs + off);
      s4 = *(const G f32x4 *)(sn + off);
    }
  } else if (nrot == 128u) {
    const u32 i4 = l & 15u;
    const unsigned long off = (unsigned long)tk * 64ul + (unsigned long)(4u * i4);
    c4 = *(const G f32x4 *)(cs + off);
    s4 = *(const G f32x4 *)(sn + off);
  } else if (nrot == 256u) {
    const unsigned long off = (unsigned long)tk * 128ul + (unsigned long)(4u * l);
    c4 = *(const G f32x4 *)(cs + off);
    s4 = *(const G f32x4 *)(sn + off);
  }
  *c4o = c4;
  *s4o = s4;
}

/* Bitwise-exact 256-dim head RMSNorm + RoPE (matches ie_kernels_q8.ll bit-for-bit).
 * Must be invoked with all 32 lanes active (EXEC == 0xFFFFFFFF). */
static inline __attribute__((always_inline)) void r11_g04_rmsnorm_rope(
    f32x4 x0, f32x4 x1, f32x4 w0, f32x4 w1, f32x4 c4, f32x4 s4,
    float eps, u32 nrot, u32 l, f32x4 *y0o, f32x4 *y1o) {
#pragma clang fp contract(off)
  const float sum_lo = r11_g04_sub8_ss(x0);
  const float sum_hi = r11_g04_sub8_ss(x1);
  const int b_lo = __builtin_bit_cast(int, sum_lo);
  const int b_hi = __builtin_bit_cast(int, sum_hi);
  const float s0 = __builtin_bit_cast(float, __builtin_amdgcn_readlane(b_lo, 0));
  const float s1 = __builtin_bit_cast(float, __builtin_amdgcn_readlane(b_lo, 8));
  const float s2 = __builtin_bit_cast(float, __builtin_amdgcn_readlane(b_lo, 16));
  const float s3 = __builtin_bit_cast(float, __builtin_amdgcn_readlane(b_lo, 24));
  const float s4v = __builtin_bit_cast(float, __builtin_amdgcn_readlane(b_hi, 0));
  const float s5 = __builtin_bit_cast(float, __builtin_amdgcn_readlane(b_hi, 8));
  const float s6 = __builtin_bit_cast(float, __builtin_amdgcn_readlane(b_hi, 16));
  const float s7 = __builtin_bit_cast(float, __builtin_amdgcn_readlane(b_hi, 24));
  float ss = 0.0f;
  ss = ss + s0; ss = ss + s1; ss = ss + s2; ss = ss + s3;
  ss = ss + s4v; ss = ss + s5; ss = ss + s6; ss = ss + s7;
  /* Exact IEEE-754 ss / 256.0f (power-of-2 scale) + eps without FMA contraction */
  const float mean = ss * 0.00390625f;
  const float inv = 1.0f / __builtin_sqrtf(mean + eps);
  f32x4 y0 = (f32x4){
      (x0[0] * inv) * w0[0], (x0[1] * inv) * w0[1],
      (x0[2] * inv) * w0[2], (x0[3] * inv) * w0[3]};
  f32x4 y1 = (f32x4){
      (x1[0] * inv) * w1[0], (x1[1] * inv) * w1[1],
      (x1[2] * inv) * w1[2], (x1[3] * inv) * w1[3]};

  if (nrot == 64u) {
    const float a0 = y0[0], a1 = y0[1], a2 = y0[2], a3 = y0[3];
    const float p0 = r11_g04_x8_f(a0);
    const float p1 = r11_g04_x8_f(a1);
    const float p2 = r11_g04_x8_f(a2);
    const float p3 = r11_g04_x8_f(a3);
    if (l < 16u) {
      if (l < 8u) {
        y0 = (f32x4){
            __builtin_fmaf(a0, c4[0], p0 * (-s4[0])),
            __builtin_fmaf(a1, c4[1], p1 * (-s4[1])),
            __builtin_fmaf(a2, c4[2], p2 * (-s4[2])),
            __builtin_fmaf(a3, c4[3], p3 * (-s4[3]))};
      } else {
        y0 = (f32x4){
            __builtin_fmaf(p0, s4[0], a0 * c4[0]),
            __builtin_fmaf(p1, s4[1], a1 * c4[1]),
            __builtin_fmaf(p2, s4[2], a2 * c4[2]),
            __builtin_fmaf(p3, s4[3], a3 * c4[3])};
      }
    }
  } else if (nrot == 128u) {
    const float a0 = y0[0], a1 = y0[1], a2 = y0[2], a3 = y0[3];
    const float p0 = r11_g04_x16_f(a0);
    const float p1 = r11_g04_x16_f(a1);
    const float p2 = r11_g04_x16_f(a2);
    const float p3 = r11_g04_x16_f(a3);
    if (l < 16u) {
      y0 = (f32x4){
          __builtin_fmaf(a0, c4[0], p0 * (-s4[0])),
          __builtin_fmaf(a1, c4[1], p1 * (-s4[1])),
          __builtin_fmaf(a2, c4[2], p2 * (-s4[2])),
          __builtin_fmaf(a3, c4[3], p3 * (-s4[3]))};
    } else {
      y0 = (f32x4){
          __builtin_fmaf(p0, s4[0], a0 * c4[0]),
          __builtin_fmaf(p1, s4[1], a1 * c4[1]),
          __builtin_fmaf(p2, s4[2], a2 * c4[2]),
          __builtin_fmaf(p3, s4[3], a3 * c4[3])};
    }
  } else if (nrot == 256u) {
    const float a0 = y0[0], a1 = y0[1], a2 = y0[2], a3 = y0[3];
    const float b0 = y1[0], b1 = y1[1], b2 = y1[2], b3 = y1[3];
    y0 = (f32x4){
        __builtin_fmaf(a0, c4[0], b0 * (-s4[0])),
        __builtin_fmaf(a1, c4[1], b1 * (-s4[1])),
        __builtin_fmaf(a2, c4[2], b2 * (-s4[2])),
        __builtin_fmaf(a3, c4[3], b3 * (-s4[3]))};
    y1 = (f32x4){
        __builtin_fmaf(a0, s4[0], b0 * c4[0]),
        __builtin_fmaf(a1, s4[1], b1 * c4[1]),
        __builtin_fmaf(a2, s4[2], b2 * c4[2]),
        __builtin_fmaf(a3, s4[3], b3 * c4[3])};
  }
  *y0o = y0;
  *y1o = y1;
}

/* Quantize 4 contiguous floats of an 8-lane (32-float) Q8_0 KV block into 1 u32;
 * returns the block's exact fp16 scale (bitwise identical to base kvq_store). */
static inline __attribute__((always_inline)) f16 r11_g04_kvq_pack4(f32x4 v, u32 *pk_out) {
#pragma clang fp contract(off)
  const float v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];
  float am = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(v0), __builtin_fabsf(v1)),
                             __builtin_fmaxf(__builtin_fabsf(v2), __builtin_fabsf(v3)));
  am = __builtin_fmaxf(am, r11_g04_dpp_x1_f(am));
  am = __builtin_fmaxf(am, r11_g04_dpp_x2_f(am));
  am = __builtin_fmaxf(am, r11_g04_x4_f(am));
  const f16 sh = (f16)(am / 127.0f);
  const float s = (float)sh;
  u32 pk = 0u;
  if (s != 0.0f) {
    int q0 = (int)__builtin_roundf(v0 / s);
    int q1 = (int)__builtin_roundf(v1 / s);
    int q2 = (int)__builtin_roundf(v2 / s);
    int q3 = (int)__builtin_roundf(v3 / s);
    q0 = q0 > 127 ? 127 : (q0 < -127 ? -127 : q0);
    q1 = q1 > 127 ? 127 : (q1 < -127 ? -127 : q1);
    q2 = q2 > 127 ? 127 : (q2 < -127 ? -127 : q2);
    q3 = q3 > 127 ? 127 : (q3 < -127 ? -127 : q3);
    pk = ((u32)(u8)(i8)q0) | (((u32)(u8)(i8)q1) << 8u) |
         (((u32)(u8)(i8)q2) << 16u) | (((u32)(u8)(i8)q3) << 24u);
  }
  *pk_out = pk;
  return sh;
}

/* Coalesced Q8_0 KV record store (256 i8 via 2x global_store_b32 + 8 f16 scales via 1x u32x4). */
static inline __attribute__((always_inline)) void r11_g04_kvq_store(
    f32x4 v0, f32x4 v1, G u8 *rec, u32 l) {
  u32 pk0, pk1;
  const f16 sh0 = r11_g04_kvq_pack4(v0, &pk0);
  const f16 sh1 = r11_g04_kvq_pack4(v1, &pk1);
  ((G u32 *)rec)[l] = pk0;
  ((G u32 *)rec)[32u + l] = pk1;
  const u32 s0b = (u32)__builtin_bit_cast(u16, sh0);
  const u32 s1b = (u32)__builtin_bit_cast(u16, sh1);
  const u32 b0 = (u32)__builtin_amdgcn_readlane((int)s0b, 0);
  const u32 b1 = (u32)__builtin_amdgcn_readlane((int)s0b, 8);
  const u32 b2 = (u32)__builtin_amdgcn_readlane((int)s0b, 16);
  const u32 b3 = (u32)__builtin_amdgcn_readlane((int)s0b, 24);
  const u32 b4 = (u32)__builtin_amdgcn_readlane((int)s1b, 0);
  const u32 b5 = (u32)__builtin_amdgcn_readlane((int)s1b, 8);
  const u32 b6 = (u32)__builtin_amdgcn_readlane((int)s1b, 16);
  const u32 b7 = (u32)__builtin_amdgcn_readlane((int)s1b, 24);
  if (l == 0u) {
    *(G u32x4 *)(rec + 256u) = (u32x4){
        b0 | (b1 << 16u), b2 | (b3 << 16u),
        b4 | (b5 << 16u), b6 | (b7 << 16u)};
  }
}

KERNEL ie_saluki_qkn_rope_kv_r11(const G float *qkv, G float *qo, G kvt *kc, G kvt *vc,
                                 const G float *qw, const G float *kw, const G float *cs, const G float *sn,
                                 u32 nq, u32 nkv, u32 hd, u32 nrot, float eps, u32 ins, u32 qos, u32 qstr,
                                 u32 T, G f16 *q16) {
#pragma clang fp contract(off)
  if (hd != 256u || T == 0u) return;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5u);
  const u32 g = (u32)__builtin_amdgcn_readfirstlane(wgid() * 8u + w);
  const u32 ntot = nq + 2u * nkv;
  if (g >= ntot) return;

  const u32 tpw = R11_G04_TPW(T);
  const u32 t0 = (u32)__builtin_amdgcn_readfirstlane(tokid() * tpw);
  if (t0 >= T) return;
  const u32 tend = t0 + tpw < T ? t0 + tpw : T;
  const u32 l = lane();

  if (g < nq) {
    /* Q head: register-cached qw + software-pipelined multi-token norm/RoPE + optional fused Q16. */
    const u32 h = g;
    const f32x4 w0 = *(const G f32x4 *)(qw + 4u * l);
    const f32x4 w1 = *(const G f32x4 *)(qw + 128u + 4u * l);
    const G float *src0 = (const G float *)((const G u8 *)qkv + (unsigned long)t0 * ins) + (unsigned long)h * qstr;
    f32x4 x0 = *(const G f32x4 *)(src0 + 4u * l);
    f32x4 x1 = *(const G f32x4 *)(src0 + 128u + 4u * l);
    for (u32 tk = t0; tk < tend; tk++) {
      f32x4 c4, s4;
      r11_g04_load_rope_cs(cs, sn, tk, nrot, l, &c4, &s4);
      f32x4 nx0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      f32x4 nx1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      if (tk + 1u < tend) {
        const G float *src_n = (const G float *)((const G u8 *)qkv + (unsigned long)(tk + 1u) * ins) + (unsigned long)h * qstr;
        nx0 = *(const G f32x4 *)(src_n + 4u * l);
        nx1 = *(const G f32x4 *)(src_n + 128u + 4u * l);
      }
      f32x4 y0, y1;
      r11_g04_rmsnorm_rope(x0, x1, w0, w1, c4, s4, eps, nrot, l, &y0, &y1);
      if (q16) {
        const float qs_f = R11_G04_QSCALE;
        G r11_g04_h4 *q16p = (G r11_g04_h4 *)(q16 + ((unsigned long)tk * nq + h) * 256ul);
        q16p[l] = (r11_g04_h4){
            (f16)(y0[0] * qs_f), (f16)(y0[1] * qs_f),
            (f16)(y0[2] * qs_f), (f16)(y0[3] * qs_f)};
        q16p[32u + l] = (r11_g04_h4){
            (f16)(y1[0] * qs_f), (f16)(y1[1] * qs_f),
            (f16)(y1[2] * qs_f), (f16)(y1[3] * qs_f)};
      }
      if (qo) {
        G float *qop = (G float *)((G u8 *)qo + (unsigned long)tk * qos) + (unsigned long)h * 256ul;
        *(G f32x4 *)(qop + 4u * l) = y0;
        *(G f32x4 *)(qop + 128u + 4u * l) = y1;
      }
      x0 = nx0;
      x1 = nx1;
    }
    return;
  }

  if (g < nq + nkv) {
    /* K head: register-cached kw + software-pipelined norm/RoPE + coalesced Q8_0 store. */
    const u32 h = g - nq;
    const unsigned long k_off = (unsigned long)nq * qstr + (unsigned long)h * 256ul;
    const f32x4 w0 = *(const G f32x4 *)(kw + 4u * l);
    const f32x4 w1 = *(const G f32x4 *)(kw + 128u + 4u * l);
    const G float *src0 = (const G float *)((const G u8 *)qkv + (unsigned long)t0 * ins) + k_off;
    f32x4 x0 = *(const G f32x4 *)(src0 + 4u * l);
    f32x4 x1 = *(const G f32x4 *)(src0 + 128u + 4u * l);
    for (u32 tk = t0; tk < tend; tk++) {
      f32x4 c4, s4;
      r11_g04_load_rope_cs(cs, sn, tk, nrot, l, &c4, &s4);
      f32x4 nx0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      f32x4 nx1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      if (tk + 1u < tend) {
        const G float *src_n = (const G float *)((const G u8 *)qkv + (unsigned long)(tk + 1u) * ins) + k_off;
        nx0 = *(const G f32x4 *)(src_n + 4u * l);
        nx1 = *(const G f32x4 *)(src_n + 128u + 4u * l);
      }
      f32x4 y0, y1;
      r11_g04_rmsnorm_rope(x0, x1, w0, w1, c4, s4, eps, nrot, l, &y0, &y1);
      G u8 *krec = (G u8 *)kc + ((unsigned long)tk * nkv + h) * R11_G04_ROWB;
      r11_g04_kvq_store(y0, y1, krec, l);
      x0 = nx0;
      x1 = nx1;
    }
    return;
  }

  /* V head: software-pipelined streaming f32x4 load + coalesced Q8_0 store. */
  const u32 h = g - nq - nkv;
  const unsigned long v_off = (unsigned long)nq * qstr + (unsigned long)(nkv + h) * 256ul;
  const G float *src0 = (const G float *)((const G u8 *)qkv + (unsigned long)t0 * ins) + v_off;
  f32x4 x0 = *(const G f32x4 *)(src0 + 4u * l);
  f32x4 x1 = *(const G f32x4 *)(src0 + 128u + 4u * l);
  for (u32 tk = t0; tk < tend; tk++) {
    f32x4 nx0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 nx1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    if (tk + 1u < tend) {
      const G float *src_n = (const G float *)((const G u8 *)qkv + (unsigned long)(tk + 1u) * ins) + v_off;
      nx0 = *(const G f32x4 *)(src_n + 4u * l);
      nx1 = *(const G f32x4 *)(src_n + 128u + 4u * l);
    }
    G u8 *vrec = (G u8 *)vc + ((unsigned long)tk * nkv + h) * R11_G04_ROWB;
    r11_g04_kvq_store(x0, x1, vrec, l);
    x0 = nx0;
    x1 = nx1;
  }
}

#undef R11_G04_TPW
#undef R11_G04_QSCALE
#undef R11_G04_ROWB
