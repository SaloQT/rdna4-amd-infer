/* ============================================================================
 * Inkling-2-Mini Round 14 (256K Long-Context + DSpark):
 *   Packed 4-Bit (Q4_0) Global-Only Target KV Cache Suite with Built-In
 *   Orthogonal Normalized FWHT-128 Rotation (rot_flags: bit 0 = K/Q, bit 1 = V/O)
 * Target: RX 9070 XT (gfx1201, wave32, NT=256, 64 CUs, 64 MiB MALL)
 *
 * Included at EOF of kernels/ie_inkling_kernels.c (after ie_kernels.c).
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r14_q4_ /
 * INK_R14_Q4_ and all macros are #undef'd at EOF.
 * 0 warnings (-Wall -Wextra), 0/0 spills, priv=0, scratch=0, TH_LOAD_NT=0,
 * TH_LOAD_HT=0 (R43 default cached loads), scalar __builtin_bit_cast safe.
 *
 * Compact Global-Only Q4_0 KV Record Format (hd == 128, n_kv == 8):
 *   - 72 B per (pos, kh) record (8-B aligned; 8 * 72 = 576 B/tok = 144 MiB per
 *     global tensor at 262,144 tokens -> exact 2.250 GiB total for 8 global K+V
 *     layers vs 8.000 GiB f16 = 3.56x smaller!):
 *       [ 0 .. 63]: 64 B packed 4-bit unsigned nibbles q in [0..15]
 *                   (even elem 2*i in low nibble, odd 2*i+1 in high nibble;
 *                    signed value = s * (float)((int)q - 8), exact zero at q=8)
 *       [64 .. 71]:  8 B = 4 x f16 scales (1 per 32-elem block, s = amax / 7.0f)
 *   - All 40 SWA layers stay in exact f16; ONLY the 8 Global layers (win == 0)
 *     use Q4_0; DSpark runs on its 8K FP16 sliding-window ring.
 *   - All KV byte offsets use 64-bit ink_r14_q4_u64 math (safe to 524,288+ pos).
 *
 * Built-In Orthogonal Normalized FWHT-128 Rotation (u32 rot_flags, 0 extra bytes):
 *   - H_128 = (1 / sqrt(128)) * H_2^{\otimes 7}: symmetric & self-inverse (H^2 = I).
 *   - rot_flags == 0u: plain Q4_0 (no rotation).
 *   - rot_flags & 1u:  rotate K before Q4_0 quant in kv_store; rotate Q in attention
 *                      prologue ((H Q) . (H K) == Q . K exact).
 *   - rot_flags & 2u:  rotate V before Q4_0 quant in kv_store; unrotate final O via
 *                      H_128 in attention epilogue before OP_GATE(sigmoid)+Q8_0 & out.
 *   - rot_flags == 3u: full K+V orthogonal rotation (slashes Q4 KV KLD).
 *
 * Exported Kernels (4):
 *   1. ie_inkling_kv_store_glob_q4_r14       (<= 48 VGPR, 4 WG/CU, 1,024 B LDS)
 *   2. ie_inkling_rel_attn_dec_glob_q4_r14   (<= 80 VGPR, 4 WG/CU, 12,484 B LDS)
 *   3. ie_inkling_rel_attn_ver_glob_q4_r14   (<= 96 VGPR, 4 WG/CU, 12,484 B LDS, QT=4)
 *   4. ie_inkling_rel_attn_pf_glob_q4_r14    (<= 120 VGPR, 3 WG/CU, 13,568 B LDS, WMMA)
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(f16) == 2u, "ink_r14_q4 f16 must be 2 bytes");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r14_q4 float/u32 must be 4 bytes");

typedef unsigned long long ink_r14_q4_u64;
typedef unsigned int       ink_r14_q4_u2 __attribute__((ext_vector_type(2)));
typedef float              ink_r14_q4_f2 __attribute__((ext_vector_type(2)));
typedef f16                ink_r14_q4_h4 __attribute__((ext_vector_type(4)));

_Static_assert(sizeof(ink_r14_q4_u64) == 8u, "ink_r14_q4_u64 must be 8 bytes");

#define INK_R14_Q4_HD       128u
#define INK_R14_Q4_DREL     16u
#define INK_R14_Q4_GQA      3u
#define INK_R14_Q4_RECB     72u  /* 64 B packed 4-bit nibbles + 8 B 4xf16 scales (8-B aligned) */
#define INK_R14_Q4_PSTR     130u
#define INK_R14_Q4_QT       4u
#define INK_R14_Q4_NB16     16u
#define INK_R14_Q4_LOG2E    1.44269504088896340736f
#define INK_R14_Q4_INV_SQRT 0.08838834764831844299f /* 1 / sqrt(128) */
#define INK_R14_Q4_QSC      (0.08838834764831844299f * 1.44269504088896340736f)

static inline __attribute__((always_inline)) float ink_r14_q4_sgpr_f(float v) {
  return I2F(__builtin_amdgcn_readfirstlane(F2I(v)));
}

static inline __attribute__((always_inline)) float ink_r14_q4_xrow(float v) {
  return I2F(xrow_i(F2I(v)));
}

static inline __attribute__((always_inline)) float ink_r14_q4_hw_sum(float v) {
  v += I2F(shl_i(F2I(v), 8));
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return I2F(__builtin_amdgcn_readlane(F2I(v), 0));
}

static inline __attribute__((always_inline)) float ink_r14_q4_dot4(f32x4 a, f32x4 b) {
  return (a.x * b.x + a.y * b.y) + (a.z * b.z + a.w * b.w);
}

static inline __attribute__((always_inline)) float ink_r14_q4_gate_f(float o, float g) {
#pragma clang fp contract(off)
  return o * (1.0f / (1.0f + __builtin_expf(-g)));
}

static inline __attribute__((always_inline)) u32 ink_r14_q4_shfl_xor_u32(u32 v, u32 m) {
  return (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ m) * 4u), (int)v);
}

/* Zero-LDS, zero-barrier exact normalized 128-pt Walsh-Hadamard transform on a
 * single wave32 holding 128 floats as f32x4 per lane (lane l -> elems 4*l..4*l+3).
 * Bitwise identical to wht_wg(..., hd=128) in ie_kernels.c! */
static inline __attribute__((always_inline)) f32x4 ink_r14_q4_fwht128_wave4(f32x4 v, u32 l) {
  /* Stage 0 (stride 1) & Stage 1 (stride 2): intra-lane 4-pt Hadamard */
  const float a0 = v.x + v.y, a1 = v.x - v.y;
  const float a2 = v.z + v.w, a3 = v.z - v.w;
  f32x4 r = {a0 + a2, a1 + a3, a0 - a2, a1 - a3};

  /* Stages 2..6 (strides 4, 8, 16, 32, 64 -> lane XOR masks 1, 2, 4, 8, 16) */
#pragma unroll
  for (u32 m = 1u; m <= 16u; m <<= 1u) {
    const float o0 = shfl_xor_f(r.x, m);
    const float o1 = shfl_xor_f(r.y, m);
    const float o2 = shfl_xor_f(r.z, m);
    const float o3 = shfl_xor_f(r.w, m);
    const int hi = (l & m) != 0u;
    r.x = hi ? (o0 - r.x) : (r.x + o0);
    r.y = hi ? (o1 - r.y) : (r.y + o1);
    r.z = hi ? (o2 - r.z) : (r.z + o2);
    r.w = hi ? (o3 - r.w) : (r.w + o3);
  }
  return r * INK_R14_Q4_INV_SQRT;
}

/* Standard cached RT dequant of 4 contiguous elements (lane l -> elems 4*l..4*l+3)
 * from 72-B Q4_0 record: 16-bit nibble load + 16-bit f16 scale load (4x less BW vs f16!). */
static inline __attribute__((always_inline)) f32x4 ink_r14_q4_ld4(const G u8 *rec, u32 l) {
  const u32 w = (u32)((const G u16 *)rec)[l];
  const u16 s_bits = ((const G u16 *)(rec + 64u))[l >> 3u];
  const float s = (float)__builtin_bit_cast(f16, s_bits);
  const float q0 = (float)((int)(w & 15u) - 8);
  const float q1 = (float)((int)((w >> 4u) & 15u) - 8);
  const float q2 = (float)((int)((w >> 8u) & 15u) - 8);
  const float q3 = (float)((int)(w >> 12u) - 8);
  return (f32x4){s * q0, s * q1, s * q2, s * q3};
}

/* ============================================================================
 * 1. ie_inkling_kv_store_glob_q4_r14
 *    Stores T tokens of K & V into 72 B/head/pos Global Q4_0 KV cache with
 *    optional orthogonal normalized FWHT-128 rotation (rot_flags: 1=K, 2=V, 3=KV).
 *    Grid: (n_kv, T, 1), NT=256 (Waves 0..3 = K blocks 0..3, Waves 4..7 = V blocks 0..3).
 *    1,024 B LDS, <= 32 VGPR (4 WG/CU), 64-bit byte offsets safe to 524,288+ pos.
 * ========================================================================== */

KERNEL ie_inkling_kv_store_glob_q4_r14(
    const G float *k, const G float *v, G u8 *kc_q4, G u8 *vc_q4,
    u32 pos0, u32 T, u32 kv_len, u32 n_kv, u32 hd, u32 ks, u32 vs,
    u32 rot_flags) {
  static LDS float ink_r14_q4_st_buf[2u][INK_R14_Q4_HD] __attribute__((aligned(16)));

  const u32 kh = wgid();
  const u32 t = tokid();
  if (kh >= n_kv || t >= T || hd != INK_R14_Q4_HD || kv_len == 0u || !k || !v || !kc_q4 || !vc_q4)
    return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 is_v = (w >= 4u) ? 1u : 0u;
  const u32 blk = w & 3u;
  const u32 e = (blk << 5u) + l; /* 0..127 within head */

  const G float *src_t = (const G float *)((const G u8 *)(is_v ? v : k) +
                                           (ink_r14_q4_u64)t * (ink_r14_q4_u64)(is_v ? vs : ks));
  float x = src_t[(ink_r14_q4_u64)kh * INK_R14_Q4_HD + e];

  /* Optional orthogonal FWHT-128 (all 8 waves execute identical lbarrier()s when rot_flags != 0). */
  if (rot_flags != 0u) {
    const int do_rot = (is_v ? (rot_flags & 2u) : (rot_flags & 1u)) != 0u;
    if (do_rot) {
#pragma unroll
      for (u32 s = 1u; s < 32u; s <<= 1u) {
        const float o = shfl_xor_f(x, s);
        x = (e & s) ? (o - x) : (x + o);
      }
      ink_r14_q4_st_buf[is_v][e] = x;
    }
    lbarrier();
    if (do_rot) {
      const float o32 = ink_r14_q4_st_buf[is_v][e ^ 32u];
      x = (e & 32u) ? (o32 - x) : (x + o32);
    }
    lbarrier();
    if (do_rot) {
      ink_r14_q4_st_buf[is_v][e] = x;
    }
    lbarrier();
    if (do_rot) {
      const float o64 = ink_r14_q4_st_buf[is_v][e ^ 64u];
      x = ((e & 64u) ? (o64 - x) : (x + o64)) * INK_R14_Q4_INV_SQRT;
    }
  }

  /* Per-32-elem symmetric Q4_0 quantization (q in 0..15, signed val = s * (q - 8)). */
  const float amax = wave_max_pos(__builtin_fabsf(x));
  const f16 sh = (f16)(amax * (1.0f / 7.0f));
  const float s = (float)sh;
  int qi = (s != 0.0f) ? (int)__builtin_roundf(x / s) : 0;
  if (qi > 7) qi = 7;
  if (qi < -8) qi = -8;
  const u32 q4 = (u32)(qi + 8);

  /* Pack 8 adjacent 4-bit nibbles into 4 coalesced u32 stores per 32-elem block (16 B). */
  const u32 q_hi = ink_r14_q4_shfl_xor_u32(q4, 1u);
  const u32 byte_pair = q4 | (q_hi << 4u);
  const u32 b_hi = ink_r14_q4_shfl_xor_u32(byte_pair, 2u);
  const u32 u16_pack = byte_pair | (b_hi << 8u);
  const u32 h_hi = ink_r14_q4_shfl_xor_u32(u16_pack, 4u);
  const u32 u32_pack = u16_pack | (h_hi << 16u);

  const u32 kvp = (pos0 + t) % kv_len;
  G u8 *dst = (is_v ? vc_q4 : kc_q4) +
              ((ink_r14_q4_u64)kvp * (ink_r14_q4_u64)n_kv + (ink_r14_q4_u64)kh) * INK_R14_Q4_RECB;
  if ((l & 7u) == 0u) {
    ((G u32 *)dst)[(blk << 2u) + (l >> 3u)] = u32_pack;
  }
  if (l == 0u) {
    ((G f16 *)(dst + 64u))[blk] = sh;
  }
}

/* ============================================================================
 * 2. ie_inkling_rel_attn_dec_glob_q4_r14
 *    T=1 Global Q4_0 Relative Attention + Built-In FWHT-128 + Optional OP_GATE+Q8.
 *    64-bit byte offsets safe to 524,288+ positions, standard cached RT loads,
 *    far-field zero-rb fast loop + exact near-field rel_bias + sink.
 *    Grid: (n_kv * nsplit, 1, 1), 12,484 B LDS, <= 80 VGPR (4 WG/CU).
 * ========================================================================== */

KERNEL ie_inkling_rel_attn_dec_glob_q4_r14(
    const G float *q, const G u8 *kc_q4, const G u8 *vc_q4, const G float *R,
    const G float *rel_bias, const G float *sink, G float *part, G float *out,
    u32 pos, u32 kv_len, u32 hd, u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent,
    u32 win, u32 nsplit, u32 ch, G u32 *count,
    const G float *agate, G u8 *oq, u32 rot_flags) {
  static LDS float ink_r14_q4_dg_m[INK_R14_Q4_GQA][8];
  static LDS float ink_r14_q4_dg_l[INK_R14_Q4_GQA][8];
  static LDS __attribute__((aligned(16))) float ink_r14_q4_dg_o[INK_R14_Q4_GQA][8][INK_R14_Q4_HD];
  static LDS u32 ink_r14_q4_dg_last;

  if (hd != INK_R14_Q4_HD || d_rel != INK_R14_Q4_DREL || n_kv == 0u ||
      n_head != INK_R14_Q4_GQA * n_kv || kv_len == 0u || nsplit == 0u)
    return;

  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() / nsplit));
  const u32 s = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() % nsplit));
  if (kh >= n_kv) return;

  const u32 h0 = kh * INK_R14_Q4_GQA;
  const ink_r14_q4_u64 row_step = (ink_r14_q4_u64)n_kv * INK_R14_Q4_RECB;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const float scale = INK_R14_Q4_INV_SQRT;

  f32x4 q0 = ((const G f32x4 *)(q + (ink_r14_q4_u64)(h0 + 0u) * INK_R14_Q4_HD))[l];
  f32x4 q1 = ((const G f32x4 *)(q + (ink_r14_q4_u64)(h0 + 1u) * INK_R14_Q4_HD))[l];
  f32x4 q2 = ((const G f32x4 *)(q + (ink_r14_q4_u64)(h0 + 2u) * INK_R14_Q4_HD))[l];
  if ((rot_flags & 1u) != 0u) {
    q0 = ink_r14_q4_fwht128_wave4(q0, l);
    q1 = ink_r14_q4_fwht128_wave4(q1, l);
    q2 = ink_r14_q4_fwht128_wave4(q2, l);
  }

  f32x4 acc0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 acc1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 acc2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  float m0 = -__builtin_inff(), sum0 = 0.0f;
  float m1 = -__builtin_inff(), sum1 = 0.0f;
  float m2 = -__builtin_inff(), sum2 = 0.0f;

  const u32 w0 = (win > 0u && pos + 1u > win) ? (pos + 1u - win) : 0u;
  const u32 p0 = w0 + s * ch;
  const u32 pend = pos + 1u;
  const u32 p1 = (w0 + (s + 1u) * ch < pend) ? (w0 + (s + 1u) * ch) : pend;

  const G u8 *kc_h = kc_q4 + (ink_r14_q4_u64)kh * INK_R14_Q4_RECB;
  const G u8 *vc_h = vc_q4 + (ink_r14_q4_u64)kh * INK_R14_Q4_RECB;

  u32 kp = p0 + w;
  if (kp < p1) {
    const int far_linear = (p1 <= kv_len) && (pos >= p1) && ((pos - (p1 - 1u)) >= extent_lim);
    if (far_linear) {
      f32x4 kf = ink_r14_q4_ld4(kc_h + (ink_r14_q4_u64)kp * row_step, l);
      f32x4 vf = ink_r14_q4_ld4(vc_h + (ink_r14_q4_u64)kp * row_step, l);
#pragma clang loop unroll(disable)
      for (;;) {
        const u32 kp_nxt = kp + 8u;
        f32x4 kf_n = kf, vf_n = vf;
        if (kp_nxt < p1) {
          kf_n = ink_r14_q4_ld4(kc_h + (ink_r14_q4_u64)kp_nxt * row_step, l);
          vf_n = ink_r14_q4_ld4(vc_h + (ink_r14_q4_u64)kp_nxt * row_step, l);
        }
        const float d0 = wave_sum_all(ink_r14_q4_dot4(q0, kf)) * scale;
        const float d1 = wave_sum_all(ink_r14_q4_dot4(q1, kf)) * scale;
        const float d2 = wave_sum_all(ink_r14_q4_dot4(q2, kf)) * scale;

        const float mn0 = __builtin_fmaxf(m0, d0);
        const float c0 = (sum0 > 0.0f) ? __builtin_expf(m0 - mn0) : 0.0f;
        const float e0 = __builtin_expf(d0 - mn0);
        sum0 = sum0 * c0 + e0; m0 = mn0;

        const float mn1 = __builtin_fmaxf(m1, d1);
        const float c1 = (sum1 > 0.0f) ? __builtin_expf(m1 - mn1) : 0.0f;
        const float e1 = __builtin_expf(d1 - mn1);
        sum1 = sum1 * c1 + e1; m1 = mn1;

        const float mn2 = __builtin_fmaxf(m2, d2);
        const float c2 = (sum2 > 0.0f) ? __builtin_expf(m2 - mn2) : 0.0f;
        const float e2 = __builtin_expf(d2 - mn2);
        sum2 = sum2 * c2 + e2; m2 = mn2;

        acc0 = acc0 * c0 + e0 * vf;
        acc1 = acc1 * c1 + e1 * vf;
        acc2 = acc2 * c2 + e2 * vf;

        if (kp_nxt >= p1) break;
        kp = kp_nxt; kf = kf_n; vf = vf_n;
      }
    } else {
      const float rv0 = (l < INK_R14_Q4_DREL) ? R[(ink_r14_q4_u64)(h0 + 0u) * INK_R14_Q4_DREL + l] : 0.0f;
      const float rv1 = (l < INK_R14_Q4_DREL) ? R[(ink_r14_q4_u64)(h0 + 1u) * INK_R14_Q4_DREL + l] : 0.0f;
      const float rv2 = (l < INK_R14_Q4_DREL) ? R[(ink_r14_q4_u64)(h0 + 2u) * INK_R14_Q4_DREL + l] : 0.0f;
      u32 kvp = kp % kv_len;
      f32x4 kf = ink_r14_q4_ld4(kc_h + (ink_r14_q4_u64)kvp * row_step, l);
      f32x4 vf = ink_r14_q4_ld4(vc_h + (ink_r14_q4_u64)kvp * row_step, l);

#pragma clang loop unroll(disable)
      for (;;) {
        const u32 dist = pos - kp;
        const int has_rel = (dist < extent_lim);
        const float rb = (has_rel && l < INK_R14_Q4_DREL)
                             ? rel_bias[(ink_r14_q4_u64)dist * INK_R14_Q4_DREL + l]
                             : 0.0f;

        const u32 kp_nxt = kp + 8u;
        u32 kvp_n = kvp;
        f32x4 kf_n = kf, vf_n = vf;
        if (kp_nxt < p1) {
          kvp_n = kvp + 8u;
          if (kvp_n >= kv_len) kvp_n = (kv_len >= 8u) ? (kvp_n - kv_len) : (kp_nxt % kv_len);
          kf_n = ink_r14_q4_ld4(kc_h + (ink_r14_q4_u64)kvp_n * row_step, l);
          vf_n = ink_r14_q4_ld4(vc_h + (ink_r14_q4_u64)kvp_n * row_step, l);
        }

        float d0 = wave_sum_all(ink_r14_q4_dot4(q0, kf)) * scale;
        float d1 = wave_sum_all(ink_r14_q4_dot4(q1, kf)) * scale;
        float d2 = wave_sum_all(ink_r14_q4_dot4(q2, kf)) * scale;
        if (has_rel) {
          d0 += ink_r14_q4_hw_sum(rv0 * rb);
          d1 += ink_r14_q4_hw_sum(rv1 * rb);
          d2 += ink_r14_q4_hw_sum(rv2 * rb);
        }

        const float mn0 = __builtin_fmaxf(m0, d0);
        const float c0 = (sum0 > 0.0f) ? __builtin_expf(m0 - mn0) : 0.0f;
        const float e0 = __builtin_expf(d0 - mn0);
        sum0 = sum0 * c0 + e0; m0 = mn0;

        const float mn1 = __builtin_fmaxf(m1, d1);
        const float c1 = (sum1 > 0.0f) ? __builtin_expf(m1 - mn1) : 0.0f;
        const float e1 = __builtin_expf(d1 - mn1);
        sum1 = sum1 * c1 + e1; m1 = mn1;

        const float mn2 = __builtin_fmaxf(m2, d2);
        const float c2 = (sum2 > 0.0f) ? __builtin_expf(m2 - mn2) : 0.0f;
        const float e2 = __builtin_expf(d2 - mn2);
        sum2 = sum2 * c2 + e2; m2 = mn2;

        acc0 = acc0 * c0 + e0 * vf;
        acc1 = acc1 * c1 + e1 * vf;
        acc2 = acc2 * c2 + e2 * vf;

        if (kp_nxt >= p1) break;
        kp = kp_nxt; kvp = kvp_n; kf = kf_n; vf = vf_n;
      }
    }
  }

  if (l == 0u) {
    ink_r14_q4_dg_m[0][w] = m0; ink_r14_q4_dg_l[0][w] = sum0;
    ink_r14_q4_dg_m[1][w] = m1; ink_r14_q4_dg_l[1][w] = sum1;
    ink_r14_q4_dg_m[2][w] = m2; ink_r14_q4_dg_l[2][w] = sum2;
  }
  ((LDSP f32x4 *)&ink_r14_q4_dg_o[0][w][0])[l] = acc0;
  ((LDSP f32x4 *)&ink_r14_q4_dg_o[1][w][0])[l] = acc1;
  ((LDSP f32x4 *)&ink_r14_q4_dg_o[2][w][0])[l] = acc2;
  lbarrier();

  const u32 nb_q = n_head << 2u;
  if (nsplit == 1u) {
    if ((rot_flags & 2u) != 0u) {
      /* Reduce into dg_o[g][0] -> unrotate via FWHT-128 in waves 0..2 -> gate+Q8. */
      for (u32 idx = tid(); idx < INK_R14_Q4_GQA * INK_R14_Q4_HD; idx += NT) {
        const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
        const float sv = sink[h];
        float M = sv;
        for (u32 x = 0u; x < 8u; x++) if (ink_r14_q4_dg_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r14_q4_dg_m[g][x]);
        float L = __builtin_expf(sv - M), o = 0.0f;
        for (u32 x = 0u; x < 8u; x++) {
          const float lx = ink_r14_q4_dg_l[g][x];
          if (lx > 0.0f) {
            const float wx = __builtin_expf(ink_r14_q4_dg_m[g][x] - M);
            L += lx * wx; o += ink_r14_q4_dg_o[g][x][e] * wx;
          }
        }
        ink_r14_q4_dg_o[g][1][e] = o / L;
      }
      lbarrier();
      if (w < INK_R14_Q4_GQA) {
        const f32x4 ov = ((const LDSP f32x4 *)&ink_r14_q4_dg_o[w][1][0])[l];
        ((LDSP f32x4 *)&ink_r14_q4_dg_o[w][1][0])[l] = ink_r14_q4_fwht128_wave4(ov, l);
      }
      lbarrier();
      for (u32 idx = tid(); idx < INK_R14_Q4_GQA * INK_R14_Q4_HD; idx += NT) {
        const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
        const u32 out_idx = h * INK_R14_Q4_HD + e;
        float v = ink_r14_q4_dg_o[g][1][e];
        if (agate) v = ink_r14_q4_gate_f(v, agate[out_idx]);
        if (out) out[out_idx] = v;
        if (oq) quant_wave(v, out_idx >> 5u, oq, nb_q);
      }
    } else {
      for (u32 idx = tid(); idx < INK_R14_Q4_GQA * INK_R14_Q4_HD; idx += NT) {
        const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
        const float sv = sink[h];
        float M = sv;
        for (u32 x = 0u; x < 8u; x++) if (ink_r14_q4_dg_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r14_q4_dg_m[g][x]);
        float L = __builtin_expf(sv - M), o = 0.0f;
        for (u32 x = 0u; x < 8u; x++) {
          const float lx = ink_r14_q4_dg_l[g][x];
          if (lx > 0.0f) {
            const float wx = __builtin_expf(ink_r14_q4_dg_m[g][x] - M);
            L += lx * wx; o += ink_r14_q4_dg_o[g][x][e] * wx;
          }
        }
        const u32 out_idx = h * INK_R14_Q4_HD + e;
        float v = o / L;
        if (agate) v = ink_r14_q4_gate_f(v, agate[out_idx]);
        if (out) out[out_idx] = v;
        if (oq) quant_wave(v, out_idx >> 5u, oq, nb_q);
      }
    }
    return;
  }

  for (u32 idx = tid(); idx < INK_R14_Q4_GQA * INK_R14_Q4_HD; idx += NT) {
    const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
    float M = -__builtin_inff();
    for (u32 x = 0u; x < 8u; x++) if (ink_r14_q4_dg_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r14_q4_dg_m[g][x]);
    G float *pp = part + (ink_r14_q4_u64)(h * nsplit + s) * INK_R14_Q4_PSTR;
    float o = 0.0f, L = 0.0f;
    for (u32 x = 0u; x < 8u; x++) {
      const float lx = ink_r14_q4_dg_l[g][x];
      if (lx > 0.0f) {
        const float wx = __builtin_expf(ink_r14_q4_dg_m[g][x] - M);
        o += ink_r14_q4_dg_o[g][x][e] * wx;
        if (e == 0u) L += lx * wx;
      }
    }
    pp[2u + e] = o;
    if (e == 0u) { pp[0] = (L > 0.0f) ? M : -1e30f; pp[1] = L; }
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r14_q4_dg_last =
        (__scoped_atomic_fetch_add(&count[kh], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r14_q4_dg_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  G float *ph0 = part + (ink_r14_q4_u64)(h0 + 0u) * nsplit * INK_R14_Q4_PSTR;
  G float *ph1 = part + (ink_r14_q4_u64)(h0 + 1u) * nsplit * INK_R14_Q4_PSTR;
  G float *ph2 = part + (ink_r14_q4_u64)(h0 + 2u) * nsplit * INK_R14_Q4_PSTR;

  float mv0 = (tid() == 0u) ? sink[h0 + 0u] : -__builtin_inff();
  float mv1 = (tid() == 0u) ? sink[h0 + 1u] : -__builtin_inff();
  float mv2 = (tid() == 0u) ? sink[h0 + 2u] : -__builtin_inff();
  for (u32 t = tid(); t < nsplit; t += NT) {
    const ink_r14_q4_u64 off = (ink_r14_q4_u64)t * INK_R14_Q4_PSTR;
    if (ph0[off + 1u] > 0.0f) mv0 = __builtin_fmaxf(mv0, ph0[off]);
    if (ph1[off + 1u] > 0.0f) mv1 = __builtin_fmaxf(mv1, ph1[off]);
    if (ph2[off + 1u] > 0.0f) mv2 = __builtin_fmaxf(mv2, ph2[off]);
  }
  for (u32 mask = 16u; mask >= 1u; mask >>= 1u) {
    mv0 = __builtin_fmaxf(mv0, shfl_xor_f(mv0, mask));
    mv1 = __builtin_fmaxf(mv1, shfl_xor_f(mv1, mask));
    mv2 = __builtin_fmaxf(mv2, shfl_xor_f(mv2, mask));
  }
  if (l == 0u) {
    ink_r14_q4_dg_m[0][w] = mv0; ink_r14_q4_dg_m[1][w] = mv1; ink_r14_q4_dg_m[2][w] = mv2;
  }
  lbarrier();

  float M2[INK_R14_Q4_GQA];
  for (u32 g = 0u; g < INK_R14_Q4_GQA; g++) {
    float mx = ink_r14_q4_dg_m[g][0];
    for (u32 x = 1u; x < 8u; x++) mx = __builtin_fmaxf(mx, ink_r14_q4_dg_m[g][x]);
    M2[g] = mx;
  }

  const int wt_in_lds = (nsplit <= INK_R14_Q4_HD);
  float lv0 = 0.0f, lv1 = 0.0f, lv2 = 0.0f;
  for (u32 t = tid(); t < nsplit; t += NT) {
    const ink_r14_q4_u64 off = (ink_r14_q4_u64)t * INK_R14_Q4_PSTR;
    const float lp0 = ph0[off + 1u], wt0 = (lp0 > 0.0f) ? __builtin_expf(ph0[off] - M2[0]) : 0.0f;
    if (wt_in_lds) ink_r14_q4_dg_o[0][0][t] = wt0; else ph0[off] = wt0;
    lv0 += lp0 * wt0;

    const float lp1 = ph1[off + 1u], wt1 = (lp1 > 0.0f) ? __builtin_expf(ph1[off] - M2[1]) : 0.0f;
    if (wt_in_lds) ink_r14_q4_dg_o[1][0][t] = wt1; else ph1[off] = wt1;
    lv1 += lp1 * wt1;

    const float lp2 = ph2[off + 1u], wt2 = (lp2 > 0.0f) ? __builtin_expf(ph2[off] - M2[2]) : 0.0f;
    if (wt_in_lds) ink_r14_q4_dg_o[2][0][t] = wt2; else ph2[off] = wt2;
    lv2 += lp2 * wt2;
  }
  lv0 = wave_tree_f(lv0); lv1 = wave_tree_f(lv1); lv2 = wave_tree_f(lv2);
  if (l == 0u) {
    ink_r14_q4_dg_l[0][w] = lv0; ink_r14_q4_dg_l[1][w] = lv1; ink_r14_q4_dg_l[2][w] = lv2;
  }
  if (wt_in_lds) lbarrier(); else barrier();

  float L2[INK_R14_Q4_GQA];
  for (u32 g = 0u; g < INK_R14_Q4_GQA; g++) {
    float ls = 0.0f;
    for (u32 x = 0u; x < 8u; x++) ls += ink_r14_q4_dg_l[g][x];
    L2[g] = ls + __builtin_expf(sink[h0 + g] - M2[g]);
  }

  if ((rot_flags & 2u) != 0u) {
    /* Store merged rotated O into row 1 (never aliases wt in row 0!), unrot, gate+Q8. */
    for (u32 idx = tid(); idx < INK_R14_Q4_GQA * INK_R14_Q4_HD; idx += NT) {
      const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
      const G float *ph = part + (ink_r14_q4_u64)h * nsplit * INK_R14_Q4_PSTR;
      float o = 0.0f;
      if (wt_in_lds) {
        for (u32 t = 0u; t < nsplit; t++) o += ph[(ink_r14_q4_u64)t * INK_R14_Q4_PSTR + 2u + e] * ink_r14_q4_dg_o[g][0][t];
      } else {
        for (u32 t = 0u; t < nsplit; t++) o += ph[(ink_r14_q4_u64)t * INK_R14_Q4_PSTR + 2u + e] * ph[(ink_r14_q4_u64)t * INK_R14_Q4_PSTR];
      }
      ink_r14_q4_dg_o[g][1][e] = o / L2[g];
    }
    lbarrier();
    if (w < INK_R14_Q4_GQA) {
      const f32x4 ov = ((const LDSP f32x4 *)&ink_r14_q4_dg_o[w][1][0])[l];
      ((LDSP f32x4 *)&ink_r14_q4_dg_o[w][1][0])[l] = ink_r14_q4_fwht128_wave4(ov, l);
    }
    lbarrier();
    for (u32 idx = tid(); idx < INK_R14_Q4_GQA * INK_R14_Q4_HD; idx += NT) {
      const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
      const u32 out_idx = h * INK_R14_Q4_HD + e;
      float v = ink_r14_q4_dg_o[g][1][e];
      if (agate) v = ink_r14_q4_gate_f(v, agate[out_idx]);
      if (out) out[out_idx] = v;
      if (oq) quant_wave(v, out_idx >> 5u, oq, nb_q);
    }
  } else {
    for (u32 idx = tid(); idx < INK_R14_Q4_GQA * INK_R14_Q4_HD; idx += NT) {
      const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
      const G float *ph = part + (ink_r14_q4_u64)h * nsplit * INK_R14_Q4_PSTR;
      float o = 0.0f;
      if (wt_in_lds) {
        for (u32 t = 0u; t < nsplit; t++) o += ph[(ink_r14_q4_u64)t * INK_R14_Q4_PSTR + 2u + e] * ink_r14_q4_dg_o[g][0][t];
      } else {
        for (u32 t = 0u; t < nsplit; t++) o += ph[(ink_r14_q4_u64)t * INK_R14_Q4_PSTR + 2u + e] * ph[(ink_r14_q4_u64)t * INK_R14_Q4_PSTR];
      }
      const u32 out_idx = h * INK_R14_Q4_HD + e;
      float v = o / L2[g];
      if (agate) v = ink_r14_q4_gate_f(v, agate[out_idx]);
      if (out) out[out_idx] = v;
      if (oq) quant_wave(v, out_idx >> 5u, oq, nb_q);
    }
  }
  if (tid() == 0u) count[kh] = 0u;
}

/* ============================================================================
 * 3. ie_inkling_rel_attn_ver_glob_q4_r14
 *    QT=4 Multi-Query Global Q4_0 Verify Attention + Built-In FWHT-128 +
 *    Optional Fused OP_GATE(sigmoid) + Q8_0.
 *    Reads each 72-B Q4_0 KV record ONCE per 4 tokens (14.2x less BW vs f16 ver_r4!).
 *    64-bit byte offsets safe to 524,288+ positions, standard cached RT loads,
 *    SGPR-hoisted m/s keeps <= 96 VGPR (4 WG/CU!), 12,484 B LDS.
 *    Grid: (n_kv * nsplit, (T + 3) / 4, 1), counter count[tokid * n_kv + kh].
 * ========================================================================== */

#define INK_R14_Q4_VQK3(qi_, kf_, sc_, l_, d0_, d1_, d2_)                                       \
  do {                                                                                          \
    const LDSP f32x4 *_qp = (const LDSP f32x4 *)(ink_r14_q4_vlds.qr.q +                         \
                                                 (unsigned long)(qi_) * (INK_R14_Q4_GQA * 128u)); \
    (d0_) = wave_sum_all(ink_r14_q4_dot4(_qp[0u * 32u + (l_)], (kf_))) * (sc_);                 \
    __builtin_amdgcn_sched_barrier(0);                                                          \
    (d1_) = wave_sum_all(ink_r14_q4_dot4(_qp[1u * 32u + (l_)], (kf_))) * (sc_);                 \
    __builtin_amdgcn_sched_barrier(0);                                                          \
    (d2_) = wave_sum_all(ink_r14_q4_dot4(_qp[2u * 32u + (l_)], (kf_))) * (sc_);                 \
  } while (0)

#define INK_R14_Q4_VRB3(qi_, rb_, l_, d0_, d1_, d2_)                                             \
  do {                                                                                           \
    const LDSP float *_rp = ink_r14_q4_vlds.qr.r + (unsigned long)(qi_) * (INK_R14_Q4_GQA * 16u); \
    const float _r0 = ((l_) < 16u) ? _rp[0u * 16u + (l_)] : 0.0f;                                \
    const float _r1 = ((l_) < 16u) ? _rp[1u * 16u + (l_)] : 0.0f;                                \
    const float _r2 = ((l_) < 16u) ? _rp[2u * 16u + (l_)] : 0.0f;                                \
    (d0_) = (d0_) + ink_r14_q4_hw_sum(_r0 * (rb_));                                              \
    (d1_) = (d1_) + ink_r14_q4_hw_sum(_r1 * (rb_));                                              \
    (d2_) = (d2_) + ink_r14_q4_hw_sum(_r2 * (rb_));                                              \
  } while (0)

#define INK_R14_Q4_VUPD(m_, s_, a_, d_, vf_)                               \
  do {                                                                     \
    const float _mn = __builtin_fmaxf((m_), (d_));                         \
    const float _c = ((s_) > 0.0f) ? __builtin_expf((m_) - _mn) : 0.0f;    \
    const float _e = __builtin_expf((d_) - _mn);                           \
    (s_) = ink_r14_q4_sgpr_f((s_) * _c + _e);                              \
    (m_) = ink_r14_q4_sgpr_f(_mn);                                         \
    (a_) = (a_) * _c + _e * (vf_);                                         \
  } while (0)

/* VGPR-resident online-softmax update (balances SGPR pressure <= 106 -> 0/0 spills!). */
#define INK_R14_Q4_VUPD_V(m_, s_, a_, d_, vf_)                             \
  do {                                                                     \
    const float _mn = __builtin_fmaxf((m_), (d_));                         \
    const float _c = ((s_) > 0.0f) ? __builtin_expf((m_) - _mn) : 0.0f;    \
    const float _e = __builtin_expf((d_) - _mn);                           \
    (s_) = (s_) * _c + _e;                                                 \
    (m_) = _mn;                                                            \
    (a_) = (a_) * _c + _e * (vf_);                                         \
  } while (0)

KERNEL ie_inkling_rel_attn_ver_glob_q4_r14(
    const G float *q, const G u8 *kc_q4, const G u8 *vc_q4,
    const G float *R, const G float *rel_bias, const G float *sink,
    G float *part, G float *out, u32 pos0, u32 T, u32 kv_len, u32 hd,
    u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
    u32 qs, u32 rs, u32 ss, u32 os, G u32 *count,
    const G float *agate, G u8 *oq, u32 gs, u32 oqs, u32 rot_flags) {
  static LDS union {
    struct {
      float q[INK_R14_Q4_QT * INK_R14_Q4_GQA * INK_R14_Q4_HD] __attribute__((aligned(16)));
      float r[INK_R14_Q4_QT * INK_R14_Q4_GQA * INK_R14_Q4_DREL] __attribute__((aligned(16)));
    } qr;
    struct {
      float m[INK_R14_Q4_GQA][8];
      float l[INK_R14_Q4_GQA][8];
      float o[INK_R14_Q4_GQA][8][INK_R14_Q4_HD] __attribute__((aligned(16)));
    } red;
  } ink_r14_q4_vlds __attribute__((aligned(16)));
  static LDS u32 ink_r14_q4_vlast;
  _Static_assert(sizeof(ink_r14_q4_vlds) == 12480u, "ink_r14_q4_vlds must be 12,480 B");

  const u32 t0 = (u32)__builtin_amdgcn_readfirstlane((int)(tokid() * INK_R14_Q4_QT));
  if (t0 >= T || hd != INK_R14_Q4_HD || d_rel != INK_R14_Q4_DREL || n_kv == 0u ||
      n_head != INK_R14_Q4_GQA * n_kv || kv_len == 0u || nsplit == 0u)
    return;

  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() / nsplit));
  const u32 s = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() % nsplit));
  if (kh >= n_kv) return;

  const u32 h0 = kh * INK_R14_Q4_GQA;
  const ink_r14_q4_u64 row_step = (ink_r14_q4_u64)n_kv * INK_R14_Q4_RECB;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();

  const u32 nq = (u32)__builtin_amdgcn_readfirstlane((int)((T - t0 < INK_R14_Q4_QT) ? (T - t0) : INK_R14_Q4_QT));
  const u32 pos_t0 = pos0 + t0;
  const u32 w_union = (win > 0u && pos_t0 + 1u > win) ? (pos_t0 + 1u - win) : 0u;
  const u32 pend = pos_t0 + nq;
  const u32 np = pend - w_union;
  const u32 ch = (np + nsplit - 1u) / nsplit;
  const u32 off0 = s * ch;
  const u32 p0 = (off0 < np) ? (w_union + off0) : pend;
  const u32 p1 = (off0 < np && off0 + ch < np) ? (w_union + off0 + ch) : pend;

  if (nsplit > 1u && p0 >= p1) {
#pragma clang loop unroll(disable)
    for (u32 qi = 0u; qi < nq; qi++) {
      const u32 t = t0 + qi;
      for (u32 idx = tid(); idx < INK_R14_Q4_GQA * INK_R14_Q4_HD; idx += NT) {
        const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
        G float *pp = part + (ink_r14_q4_u64)((t * n_head + h) * nsplit + s) * INK_R14_Q4_PSTR;
        pp[2u + e] = 0.0f;
        if (e == 0u) { pp[0] = -1e30f; pp[1] = 0.0f; }
      }
    }
  } else {
    const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
    const int far_linear = (p1 <= kv_len) && (p1 <= pos_t0 + 1u) && ((pos_t0 + 1u - p1) >= extent_lim);
    const float scale = INK_R14_Q4_INV_SQRT;

    const u32 nqg = nq * INK_R14_Q4_GQA;
    for (u32 qg = w; qg < nqg; qg += 8u) {
      const u32 qi = qg / INK_R14_Q4_GQA;
      const u32 gi = qg - qi * INK_R14_Q4_GQA;
      const G float *qt = (const G float *)((const G u8 *)q + (ink_r14_q4_u64)(t0 + qi) * (ink_r14_q4_u64)qs);
      f32x4 qv = ((const G f32x4 *)(qt + (ink_r14_q4_u64)(h0 + gi) * INK_R14_Q4_HD))[l];
      if ((rot_flags & 1u) != 0u) qv = ink_r14_q4_fwht128_wave4(qv, l);
      ((LDSP f32x4 *)(ink_r14_q4_vlds.qr.q + (unsigned long)qg * INK_R14_Q4_HD))[l] = qv;
    }
    if (!far_linear && extent_lim > 0u) {
      const u32 idx = tid();
      if (idx < nqg * INK_R14_Q4_DREL) {
        const u32 qg = idx >> 4u, e = idx & 15u;
        const u32 qi = qg / INK_R14_Q4_GQA, gi = qg - qi * INK_R14_Q4_GQA;
        const G float *Rt = (const G float *)((const G u8 *)R + (ink_r14_q4_u64)(t0 + qi) * (ink_r14_q4_u64)rs);
        ink_r14_q4_vlds.qr.r[(unsigned long)qg * INK_R14_Q4_DREL + e] =
            Rt[(ink_r14_q4_u64)(h0 + gi) * INK_R14_Q4_DREL + e];
      }
    }
    lbarrier();

    f32x4 acc00 = (f32x4)(0.0f), acc01 = (f32x4)(0.0f), acc02 = (f32x4)(0.0f);
    f32x4 acc10 = (f32x4)(0.0f), acc11 = (f32x4)(0.0f), acc12 = (f32x4)(0.0f);
    f32x4 acc20 = (f32x4)(0.0f), acc21 = (f32x4)(0.0f), acc22 = (f32x4)(0.0f);
    f32x4 acc30 = (f32x4)(0.0f), acc31 = (f32x4)(0.0f), acc32 = (f32x4)(0.0f);

    float m00 = -__builtin_inff(), s00 = 0.0f, m01 = -__builtin_inff(), s01 = 0.0f, m02 = -__builtin_inff(), s02 = 0.0f;
    float m10 = -__builtin_inff(), s10 = 0.0f, m11 = -__builtin_inff(), s11 = 0.0f, m12 = -__builtin_inff(), s12 = 0.0f;
    float m20 = -__builtin_inff(), s20 = 0.0f, m21 = -__builtin_inff(), s21 = 0.0f, m22 = -__builtin_inff(), s22 = 0.0f;
    float m30 = -__builtin_inff(), s30 = 0.0f, m31 = -__builtin_inff(), s31 = 0.0f, m32 = -__builtin_inff(), s32 = 0.0f;

    const G u8 *kc_h = kc_q4 + (ink_r14_q4_u64)kh * INK_R14_Q4_RECB;
    const G u8 *vc_h = vc_q4 + (ink_r14_q4_u64)kh * INK_R14_Q4_RECB;
    u32 kp = (u32)__builtin_amdgcn_readfirstlane((int)(p0 + w));

    if (kp < p1) {
      if (far_linear) {
#pragma clang loop unroll(disable)
        for (; kp < p1; kp += 8u) {
          const f32x4 kf = ink_r14_q4_ld4(kc_h + (ink_r14_q4_u64)kp * row_step, l);
          const f32x4 vf = ink_r14_q4_ld4(vc_h + (ink_r14_q4_u64)kp * row_step, l);
          float d0, d1, d2;
          INK_R14_Q4_VQK3(0u, kf, scale, l, d0, d1, d2);
          INK_R14_Q4_VUPD(m00, s00, acc00, d0, vf);
          INK_R14_Q4_VUPD(m01, s01, acc01, d1, vf);
          INK_R14_Q4_VUPD(m02, s02, acc02, d2, vf);
          if (nq > 1u) {
            INK_R14_Q4_VQK3(1u, kf, scale, l, d0, d1, d2);
            INK_R14_Q4_VUPD(m10, s10, acc10, d0, vf);
            INK_R14_Q4_VUPD(m11, s11, acc11, d1, vf);
            INK_R14_Q4_VUPD(m12, s12, acc12, d2, vf);
          }
          if (nq > 2u) {
            INK_R14_Q4_VQK3(2u, kf, scale, l, d0, d1, d2);
            INK_R14_Q4_VUPD(m20, s20, acc20, d0, vf);
            INK_R14_Q4_VUPD_V(m21, s21, acc21, d1, vf);
            INK_R14_Q4_VUPD_V(m22, s22, acc22, d2, vf);
          }
          if (nq > 3u) {
            INK_R14_Q4_VQK3(3u, kf, scale, l, d0, d1, d2);
            INK_R14_Q4_VUPD_V(m30, s30, acc30, d0, vf);
            INK_R14_Q4_VUPD_V(m31, s31, acc31, d1, vf);
            INK_R14_Q4_VUPD_V(m32, s32, acc32, d2, vf);
          }
        }
      } else {
#pragma clang loop unroll(disable)
        for (; kp < p1; kp += 8u) {
          const u32 kvp = (kp < kv_len) ? kp : (kp % kv_len);
          const f32x4 kf = ink_r14_q4_ld4(kc_h + (ink_r14_q4_u64)kvp * row_step, l);
          const f32x4 vf = ink_r14_q4_ld4(vc_h + (ink_r14_q4_u64)kvp * row_step, l);

          if (kp <= pos_t0 && (win == 0u || (pos_t0 - kp) < win)) {
            const u32 dist = pos_t0 - kp;
            const float rb = (dist < extent_lim && l < 16u) ? rel_bias[(ink_r14_q4_u64)dist * 16ull + l] : 0.0f;
            float d0, d1, d2;
            INK_R14_Q4_VQK3(0u, kf, scale, l, d0, d1, d2);
            if (dist < extent_lim) INK_R14_Q4_VRB3(0u, rb, l, d0, d1, d2);
            INK_R14_Q4_VUPD(m00, s00, acc00, d0, vf);
            INK_R14_Q4_VUPD(m01, s01, acc01, d1, vf);
            INK_R14_Q4_VUPD(m02, s02, acc02, d2, vf);
          }
          if (nq > 1u && kp <= pos_t0 + 1u && (win == 0u || (pos_t0 + 1u - kp) < win)) {
            const u32 dist = pos_t0 + 1u - kp;
            const float rb = (dist < extent_lim && l < 16u) ? rel_bias[(ink_r14_q4_u64)dist * 16ull + l] : 0.0f;
            float d0, d1, d2;
            INK_R14_Q4_VQK3(1u, kf, scale, l, d0, d1, d2);
            if (dist < extent_lim) INK_R14_Q4_VRB3(1u, rb, l, d0, d1, d2);
            INK_R14_Q4_VUPD(m10, s10, acc10, d0, vf);
            INK_R14_Q4_VUPD(m11, s11, acc11, d1, vf);
            INK_R14_Q4_VUPD(m12, s12, acc12, d2, vf);
          }
          if (nq > 2u && kp <= pos_t0 + 2u && (win == 0u || (pos_t0 + 2u - kp) < win)) {
            const u32 dist = pos_t0 + 2u - kp;
            const float rb = (dist < extent_lim && l < 16u) ? rel_bias[(ink_r14_q4_u64)dist * 16ull + l] : 0.0f;
            float d0, d1, d2;
            INK_R14_Q4_VQK3(2u, kf, scale, l, d0, d1, d2);
            if (dist < extent_lim) INK_R14_Q4_VRB3(2u, rb, l, d0, d1, d2);
            INK_R14_Q4_VUPD(m20, s20, acc20, d0, vf);
            INK_R14_Q4_VUPD_V(m21, s21, acc21, d1, vf);
            INK_R14_Q4_VUPD_V(m22, s22, acc22, d2, vf);
          }
          if (nq > 3u && kp <= pos_t0 + 3u && (win == 0u || (pos_t0 + 3u - kp) < win)) {
            const u32 dist = pos_t0 + 3u - kp;
            const float rb = (dist < extent_lim && l < 16u) ? rel_bias[(ink_r14_q4_u64)dist * 16ull + l] : 0.0f;
            float d0, d1, d2;
            INK_R14_Q4_VQK3(3u, kf, scale, l, d0, d1, d2);
            if (dist < extent_lim) INK_R14_Q4_VRB3(3u, rb, l, d0, d1, d2);
            INK_R14_Q4_VUPD_V(m30, s30, acc30, d0, vf);
            INK_R14_Q4_VUPD_V(m31, s31, acc31, d1, vf);
            INK_R14_Q4_VUPD_V(m32, s32, acc32, d2, vf);
          }
        }
      }
    }
    lbarrier();

#pragma clang loop unroll(disable)
    for (u32 qi = 0u; qi < nq; qi++) {
      if (qi > 0u) lbarrier();
      float rm0, rm1, rm2, rl0, rl1, rl2;
      f32x4 ra0, ra1, ra2;
      if (qi == 0u)      { rm0 = m00; rm1 = m01; rm2 = m02; rl0 = s00; rl1 = s01; rl2 = s02; ra0 = acc00; ra1 = acc01; ra2 = acc02; }
      else if (qi == 1u) { rm0 = m10; rm1 = m11; rm2 = m12; rl0 = s10; rl1 = s11; rl2 = s12; ra0 = acc10; ra1 = acc11; ra2 = acc12; }
      else if (qi == 2u) { rm0 = m20; rm1 = m21; rm2 = m22; rl0 = s20; rl1 = s21; rl2 = s22; ra0 = acc20; ra1 = acc21; ra2 = acc22; }
      else               { rm0 = m30; rm1 = m31; rm2 = m32; rl0 = s30; rl1 = s31; rl2 = s32; ra0 = acc30; ra1 = acc31; ra2 = acc32; }

      if (l == 0u) {
        ink_r14_q4_vlds.red.m[0][w] = rm0; ink_r14_q4_vlds.red.l[0][w] = rl0;
        ink_r14_q4_vlds.red.m[1][w] = rm1; ink_r14_q4_vlds.red.l[1][w] = rl1;
        ink_r14_q4_vlds.red.m[2][w] = rm2; ink_r14_q4_vlds.red.l[2][w] = rl2;
      }
      ((LDSP f32x4 *)&ink_r14_q4_vlds.red.o[0][w][0])[l] = ra0;
      ((LDSP f32x4 *)&ink_r14_q4_vlds.red.o[1][w][0])[l] = ra1;
      ((LDSP f32x4 *)&ink_r14_q4_vlds.red.o[2][w][0])[l] = ra2;
      lbarrier();

      const u32 t = t0 + qi;
      if (nsplit == 1u) {
        const G float *st = (const G float *)((const G u8 *)sink + (ink_r14_q4_u64)t * (ink_r14_q4_u64)ss);
        const G float *gt = agate ? (const G float *)((const G u8 *)agate + (ink_r14_q4_u64)t * (ink_r14_q4_u64)gs) : (const G float *)0;
        G float *ot = out ? (G float *)((G u8 *)out + (ink_r14_q4_u64)t * (ink_r14_q4_u64)os) : (G float *)0;
        G u8 *oqt = oq ? (oq + (ink_r14_q4_u64)t * (ink_r14_q4_u64)oqs) : (G u8 *)0;
        const u32 nb_q = n_head << 2u;
        for (u32 idx = tid(); idx < INK_R14_Q4_GQA * INK_R14_Q4_HD; idx += NT) {
          const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
          const float sv = st[h];
          float M = sv;
          for (u32 x = 0u; x < 8u; x++) if (ink_r14_q4_vlds.red.l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r14_q4_vlds.red.m[g][x]);
          float L = __builtin_expf(sv - M), o = 0.0f;
          for (u32 x = 0u; x < 8u; x++) {
            const float lx = ink_r14_q4_vlds.red.l[g][x];
            if (lx > 0.0f) {
              const float wx = __builtin_expf(ink_r14_q4_vlds.red.m[g][x] - M);
              L += lx * wx; o += ink_r14_q4_vlds.red.o[g][x][e] * wx;
            }
          }
          float v = o / L;
          if ((rot_flags & 2u) != 0u) {
            ink_r14_q4_vlds.red.o[g][1][e] = v;
          } else {
            const u32 out_idx = h * INK_R14_Q4_HD + e;
            if (gt) v = ink_r14_q4_gate_f(v, gt[out_idx]);
            if (ot) ot[out_idx] = v;
            if (oqt) quant_wave(v, out_idx >> 5u, oqt, nb_q);
          }
        }
        if ((rot_flags & 2u) != 0u) {
          lbarrier();
          if (w < INK_R14_Q4_GQA) {
            const f32x4 ov = ((const LDSP f32x4 *)&ink_r14_q4_vlds.red.o[w][1][0])[l];
            ((LDSP f32x4 *)&ink_r14_q4_vlds.red.o[w][1][0])[l] = ink_r14_q4_fwht128_wave4(ov, l);
          }
          lbarrier();
          for (u32 idx = tid(); idx < INK_R14_Q4_GQA * INK_R14_Q4_HD; idx += NT) {
            const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
            const u32 out_idx = h * INK_R14_Q4_HD + e;
            float v = ink_r14_q4_vlds.red.o[g][1][e];
            if (gt) v = ink_r14_q4_gate_f(v, gt[out_idx]);
            if (ot) ot[out_idx] = v;
            if (oqt) quant_wave(v, out_idx >> 5u, oqt, nb_q);
          }
        }
      } else {
        for (u32 idx = tid(); idx < INK_R14_Q4_GQA * INK_R14_Q4_HD; idx += NT) {
          const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
          float M = -__builtin_inff();
          for (u32 x = 0u; x < 8u; x++) if (ink_r14_q4_vlds.red.l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r14_q4_vlds.red.m[g][x]);
          G float *pp = part + (ink_r14_q4_u64)((t * n_head + h) * nsplit + s) * INK_R14_Q4_PSTR;
          float o = 0.0f, L = 0.0f;
          for (u32 x = 0u; x < 8u; x++) {
            const float lx = ink_r14_q4_vlds.red.l[g][x];
            if (lx > 0.0f) {
              const float wx = __builtin_expf(ink_r14_q4_vlds.red.m[g][x] - M);
              o += ink_r14_q4_vlds.red.o[g][x][e] * wx;
              if (e == 0u) L += lx * wx;
            }
          }
          pp[2u + e] = o;
          if (e == 0u) { pp[0] = (L > 0.0f) ? M : -1e30f; pp[1] = L; }
        }
      }
    }
    if (nsplit == 1u) return;
  }

  const u32 cidx = (t0 >> 2u) * n_kv + kh;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r14_q4_vlast =
        (__scoped_atomic_fetch_add(&count[cidx], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r14_q4_vlast) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const int wt_in_lds = (nsplit <= INK_R14_Q4_HD);
#pragma clang loop unroll(disable)
  for (u32 qi = 0u; qi < nq; qi++) {
    if (qi > 0u) lbarrier();
    const u32 t = t0 + qi;
    const G float *st = (const G float *)((const G u8 *)sink + (ink_r14_q4_u64)t * (ink_r14_q4_u64)ss);
    const G float *gt = agate ? (const G float *)((const G u8 *)agate + (ink_r14_q4_u64)t * (ink_r14_q4_u64)gs) : (const G float *)0;
    G float *ot = out ? (G float *)((G u8 *)out + (ink_r14_q4_u64)t * (ink_r14_q4_u64)os) : (G float *)0;
    G u8 *oqt = oq ? (oq + (ink_r14_q4_u64)t * (ink_r14_q4_u64)oqs) : (G u8 *)0;

    G float *ph0 = part + (ink_r14_q4_u64)((t * n_head + h0 + 0u) * nsplit) * INK_R14_Q4_PSTR;
    G float *ph1 = part + (ink_r14_q4_u64)((t * n_head + h0 + 1u) * nsplit) * INK_R14_Q4_PSTR;
    G float *ph2 = part + (ink_r14_q4_u64)((t * n_head + h0 + 2u) * nsplit) * INK_R14_Q4_PSTR;

    float mv0 = (tid() == 0u) ? st[h0 + 0u] : -__builtin_inff();
    float mv1 = (tid() == 0u) ? st[h0 + 1u] : -__builtin_inff();
    float mv2 = (tid() == 0u) ? st[h0 + 2u] : -__builtin_inff();
    for (u32 sp = tid(); sp < nsplit; sp += NT) {
      const ink_r14_q4_u64 off = (ink_r14_q4_u64)sp * INK_R14_Q4_PSTR;
      if (ph0[off + 1u] > 0.0f) mv0 = __builtin_fmaxf(mv0, ph0[off]);
      if (ph1[off + 1u] > 0.0f) mv1 = __builtin_fmaxf(mv1, ph1[off]);
      if (ph2[off + 1u] > 0.0f) mv2 = __builtin_fmaxf(mv2, ph2[off]);
    }
    for (u32 mask = 16u; mask >= 1u; mask >>= 1u) {
      mv0 = __builtin_fmaxf(mv0, shfl_xor_f(mv0, mask));
      mv1 = __builtin_fmaxf(mv1, shfl_xor_f(mv1, mask));
      mv2 = __builtin_fmaxf(mv2, shfl_xor_f(mv2, mask));
    }
    if (l == 0u) {
      ink_r14_q4_vlds.red.m[0][w] = mv0; ink_r14_q4_vlds.red.m[1][w] = mv1; ink_r14_q4_vlds.red.m[2][w] = mv2;
    }
    lbarrier();

    float M2[INK_R14_Q4_GQA];
    for (u32 g = 0u; g < INK_R14_Q4_GQA; g++) {
      float mx = ink_r14_q4_vlds.red.m[g][0];
      for (u32 x = 1u; x < 8u; x++) mx = __builtin_fmaxf(mx, ink_r14_q4_vlds.red.m[g][x]);
      M2[g] = mx;
    }

    float lv0 = 0.0f, lv1 = 0.0f, lv2 = 0.0f;
    for (u32 sp = tid(); sp < nsplit; sp += NT) {
      const ink_r14_q4_u64 off = (ink_r14_q4_u64)sp * INK_R14_Q4_PSTR;
      const float lp0 = ph0[off + 1u], wt0 = (lp0 > 0.0f) ? __builtin_expf(ph0[off] - M2[0]) : 0.0f;
      if (wt_in_lds) ink_r14_q4_vlds.red.o[0][0][sp] = wt0; else ph0[off] = wt0;
      lv0 += lp0 * wt0;

      const float lp1 = ph1[off + 1u], wt1 = (lp1 > 0.0f) ? __builtin_expf(ph1[off] - M2[1]) : 0.0f;
      if (wt_in_lds) ink_r14_q4_vlds.red.o[1][0][sp] = wt1; else ph1[off] = wt1;
      lv1 += lp1 * wt1;

      const float lp2 = ph2[off + 1u], wt2 = (lp2 > 0.0f) ? __builtin_expf(ph2[off] - M2[2]) : 0.0f;
      if (wt_in_lds) ink_r14_q4_vlds.red.o[2][0][sp] = wt2; else ph2[off] = wt2;
      lv2 += lp2 * wt2;
    }
    lv0 = wave_tree_f(lv0); lv1 = wave_tree_f(lv1); lv2 = wave_tree_f(lv2);
    if (l == 0u) {
      ink_r14_q4_vlds.red.l[0][w] = lv0; ink_r14_q4_vlds.red.l[1][w] = lv1; ink_r14_q4_vlds.red.l[2][w] = lv2;
    }
    if (wt_in_lds) lbarrier(); else barrier();

    float L2[INK_R14_Q4_GQA];
    for (u32 g = 0u; g < INK_R14_Q4_GQA; g++) {
      float ls = 0.0f;
      for (u32 x = 0u; x < 8u; x++) ls += ink_r14_q4_vlds.red.l[g][x];
      L2[g] = ls + __builtin_expf(st[h0 + g] - M2[g]);
    }

    const u32 nb_q = n_head << 2u;
    for (u32 idx = tid(); idx < INK_R14_Q4_GQA * INK_R14_Q4_HD; idx += NT) {
      const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
      const G float *ph = part + (ink_r14_q4_u64)((t * n_head + h) * nsplit) * INK_R14_Q4_PSTR;
      float o = 0.0f;
      if (wt_in_lds) {
        for (u32 sp = 0u; sp < nsplit; sp++) o += ph[(ink_r14_q4_u64)sp * INK_R14_Q4_PSTR + 2u + e] * ink_r14_q4_vlds.red.o[g][0][sp];
      } else {
        for (u32 sp = 0u; sp < nsplit; sp++) o += ph[(ink_r14_q4_u64)sp * INK_R14_Q4_PSTR + 2u + e] * ph[(ink_r14_q4_u64)sp * INK_R14_Q4_PSTR];
      }
      float v = o / L2[g];
      if ((rot_flags & 2u) != 0u) {
        ink_r14_q4_vlds.red.o[g][1][e] = v;
      } else {
        const u32 out_idx = h * INK_R14_Q4_HD + e;
        if (gt) v = ink_r14_q4_gate_f(v, gt[out_idx]);
        if (ot) ot[out_idx] = v;
        if (oqt) quant_wave(v, out_idx >> 5u, oqt, nb_q);
      }
    }
    if ((rot_flags & 2u) != 0u) {
      lbarrier();
      if (w < INK_R14_Q4_GQA) {
        const f32x4 ov = ((const LDSP f32x4 *)&ink_r14_q4_vlds.red.o[w][1][0])[l];
        ((LDSP f32x4 *)&ink_r14_q4_vlds.red.o[w][1][0])[l] = ink_r14_q4_fwht128_wave4(ov, l);
      }
      lbarrier();
      for (u32 idx = tid(); idx < INK_R14_Q4_GQA * INK_R14_Q4_HD; idx += NT) {
        const u32 g = idx >> 7u, e = idx & 127u, h = h0 + g;
        const u32 out_idx = h * INK_R14_Q4_HD + e;
        float v = ink_r14_q4_vlds.red.o[g][1][e];
        if (gt) v = ink_r14_q4_gate_f(v, gt[out_idx]);
        if (ot) ot[out_idx] = v;
        if (oqt) quant_wave(v, out_idx >> 5u, oqt, nb_q);
      }
    }
  }
  barrier();
  if (tid() == 0u) count[cidx] = 0u;
}

/* ============================================================================
 * 4. ie_inkling_rel_attn_pf_glob_q4_r14
 *    2-Wave Producer / 6-Wave Consumer FP16 WMMA Prefill reading 72 B Global Q4_0 KV
 *    with Built-In Orthogonal FWHT-128 Rotation (0 extra LDS, 0 extra K-loop VGPRs!).
 *    64-bit KV byte offsets safe to 524,288+ pos, standard cached RT loads,
 *    <= 120 VGPR (true 3 WGs/CU, 12w/SIMD!), 13,568 B LDS.
 *    Grid: ((T + 31) / 32, n_kv, 1).
 * ========================================================================== */

typedef struct {
  union {
    f16   k[INK_R14_Q4_NB16 * INK_R14_Q4_HD];  /* 4,096 B: 16 x 128 XOR-swizzled f16 K */
    float tmp[6u][INK_R14_Q4_HD];              /* 3,072 B: Prologue Q / Epilogue O FWHT-128 staging */
  } ku;
  f16 vt[INK_R14_Q4_HD * INK_R14_Q4_NB16];     /* 4,096 B: 128 x 16 XOR-swizzled f16 V^T */
  f16 rb[48u * 16u];                           /* 1,536 B: Near-field rel_bias tile (48 dists x 16) */
  f16 r[6u][16u][20u];                         /* 3,840 B: Consumer R*LOG2E (8-B aligned) */
} ink_r14_q4_pf_lds_t;

static LDS ink_r14_q4_pf_lds_t ink_r14_q4_pf_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r14_q4_pf_lds_t) == 13568u, "ink_r14_q4_pf_lds_t must be 13,568 B (<= 16 KiB)");

static inline __attribute__((always_inline)) u32 ink_r14_q4_pf_k_off(u32 r, u32 c) {
  return r * 128u + 8u * (c ^ (r & 15u));
}

static inline __attribute__((always_inline)) u32 ink_r14_q4_pf_vt_off(u32 d, u32 h) {
  return 8u * ((2u * d + h) ^ ((d >> 2u) & 1u) ^ ((d >> 3u) & 7u));
}

static inline __attribute__((always_inline)) u32 ink_r14_q4_pf_lane(void) {
  u32 l;
  __asm__ volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(l));
  return l;
}

static inline __attribute__((always_inline)) void ink_r14_q4_pf_t8(
    const u32 x[8], u32x4 *ev, u32x4 *od) {
  const u32 e0 = __builtin_amdgcn_perm(x[1], x[0], 0x05040100u);
  const u32 e1 = __builtin_amdgcn_perm(x[3], x[2], 0x05040100u);
  const u32 e2 = __builtin_amdgcn_perm(x[5], x[4], 0x05040100u);
  const u32 e3 = __builtin_amdgcn_perm(x[7], x[6], 0x05040100u);
  const u32 o0 = __builtin_amdgcn_perm(x[1], x[0], 0x07060302u);
  const u32 o1 = __builtin_amdgcn_perm(x[3], x[2], 0x07060302u);
  const u32 o2 = __builtin_amdgcn_perm(x[5], x[4], 0x07060302u);
  const u32 o3 = __builtin_amdgcn_perm(x[7], x[6], 0x07060302u);
  *ev = (u32x4){e0, e1, e2, e3};
  *od = (u32x4){o0, o1, o2, o3};
}

/* Dequantize 4 packed 4-bit nibbles (low 16 bits of w) * scale -> packed f16x4 (ink_r14_q4_u2). */
static inline __attribute__((always_inline)) ink_r14_q4_u2 ink_r14_q4_deq4n_h4(u32 w, float s) {
  const ink_r14_q4_h4 h = {(f16)(s * (float)((int)(w & 15u) - 8)),
                           (f16)(s * (float)((int)((w >> 4u) & 15u) - 8)),
                           (f16)(s * (float)((int)((w >> 8u) & 15u) - 8)),
                           (f16)(s * (float)((int)((w >> 12u) & 15u) - 8))};
  return __builtin_bit_cast(ink_r14_q4_u2, h);
}

static inline __attribute__((always_inline)) void ink_r14_q4_pf_prod_blk(
    const G u8 *kb_kh, const G u8 *vb_kh, const G float *rel_bias,
    ink_r14_q4_u64 row_step, u32 kv_len, u32 kb, u32 kend,
    u32 qpos_min, u32 extent_lim, int is_far, int no_wrap, u32 pt) {
  /* K: 16 rows x 16 c8-chunks = 256 chunks -> 4 chunks/thread; 1 u32 = 8 nibbles! */
  const u32 c = pt & 15u;
  const u32 r0 = pt >> 4u;
  const u32 blk_c = c >> 2u;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r = r0 + 4u * i;
    const u32 kp = kb + r;
    u32x4 kh8 = (u32x4)(0u);
    if (kp < kend) {
      const u32 kvp = no_wrap ? kp : (kp % kv_len);
      const G u8 *rec = kb_kh + (ink_r14_q4_u64)kvp * row_step;
      const u32 w8n = ((const G u32 *)rec)[c];
      const u16 s_bits = ((const G u16 *)(rec + 64u))[blk_c];
      const float sc = (float)__builtin_bit_cast(f16, s_bits);
      const ink_r14_q4_u2 h0 = ink_r14_q4_deq4n_h4(w8n, sc);
      const ink_r14_q4_u2 h1 = ink_r14_q4_deq4n_h4(w8n >> 16u, sc);
      kh8 = (u32x4){h0.x, h0.y, h1.x, h1.y};
    }
    *(LDSP u32x4 *)(ink_r14_q4_pf_lds.ku.k + ink_r14_q4_pf_k_off(r, c)) = kh8;
  }

  /* V^T: wi = pt>>5 (0..1 -> 8 keys each), L = pt&31 (4 dims -> 1 u16 = 4 nibbles!). */
  const u32 wi = pt >> 5u;
  const u32 L = pt & 31u;
  const u32 blk_v = L >> 3u;
  u32 vx0[8], vx1[8];
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const u32 kp = kb + 8u * wi + u;
    if (kp < kend) {
      const u32 kvp = no_wrap ? kp : (kp % kv_len);
      const G u8 *rec = vb_kh + (ink_r14_q4_u64)kvp * row_step;
      const u32 w4n = (u32)((const G u16 *)rec)[L];
      const u16 s_bits = ((const G u16 *)(rec + 64u))[blk_v];
      const float sc = (float)__builtin_bit_cast(f16, s_bits);
      const ink_r14_q4_u2 h4 = ink_r14_q4_deq4n_h4(w4n, sc);
      vx0[u] = h4.x;
      vx1[u] = h4.y;
    } else {
      vx0[u] = 0u;
      vx1[u] = 0u;
    }
  }
  u32x4 ev0, od0, ev1, od1;
  ink_r14_q4_pf_t8(vx0, &ev0, &od0);
  ink_r14_q4_pf_t8(vx1, &ev1, &od1);
  const u32 c0 = 4u * L;
  *(LDSP u32x4 *)(ink_r14_q4_pf_lds.vt + ink_r14_q4_pf_vt_off(c0 + 0u, wi)) = ev0;
  *(LDSP u32x4 *)(ink_r14_q4_pf_lds.vt + ink_r14_q4_pf_vt_off(c0 + 1u, wi)) = od0;
  *(LDSP u32x4 *)(ink_r14_q4_pf_lds.vt + ink_r14_q4_pf_vt_off(c0 + 2u, wi)) = ev1;
  *(LDSP u32x4 *)(ink_r14_q4_pf_lds.vt + ink_r14_q4_pf_vt_off(c0 + 3u, wi)) = od1;

  if (!is_far && extent_lim > 0u) {
    const int d_min_s = (int)qpos_min - (int)(kb + 15u);
    const u32 d_base = d_min_s > 0 ? (u32)d_min_s : 0u;
    if (pt < 48u) {
      const u32 d = d_base + pt;
      h8 lo = (h8)(0.0f), hi = (h8)(0.0f);
      if (d < extent_lim) {
        const G f32x4 *rp = (const G f32x4 *)(rel_bias + (ink_r14_q4_u64)d * 16ull);
        const f32x4 v0 = rp[0], v1 = rp[1], v2 = rp[2], v3 = rp[3];
        lo = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
        hi = (h8){(f16)v2.x, (f16)v2.y, (f16)v2.z, (f16)v2.w, (f16)v3.x, (f16)v3.y, (f16)v3.z, (f16)v3.w};
      }
      *(LDSP h8 *)(ink_r14_q4_pf_lds.rb + pt * 16u + 0u) = lo;
      *(LDSP h8 *)(ink_r14_q4_pf_lds.rb + pt * 16u + 8u) = hi;
    }
  }
}

static inline __attribute__((always_inline)) v8f ink_r14_q4_pf_qk16(const h8 qa[8]) {
  const u32 l_loc = ink_r14_q4_pf_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  u32 kb_b = (rl << 8u) | ((rl ^ hl) << 4u);
  const LDSP u8 *kp = (const LDSP u8 *)ink_r14_q4_pf_lds.ku.k;
  v8f sa = (v8f)(0.0f);
  h8 k0;
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[0], sa); kb_b ^= 32u; __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[1], sa); kb_b ^= 96u; __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[2], sa); kb_b ^= 32u; __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[3], sa); kb_b ^= 224u; __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[4], sa); kb_b ^= 32u; __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[5], sa); kb_b ^= 96u; __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[6], sa); kb_b ^= 32u; __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[7], sa); __builtin_amdgcn_sched_barrier(0);
  return sa;
}

static inline __attribute__((always_inline)) void ink_r14_q4_pf_pv16(const h8 pb, v8f o[8]) {
  const u32 l_loc = ink_r14_q4_pf_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2u) & 1u) ^ (rl >> 3u);
  const u32 vtb = 16u * vt_lane;
  const LDSP u8 *vp = (const LDSP u8 *)ink_r14_q4_pf_lds.vt;
  h8 v0;
  v0 = *(const LDSP h8 *)(vp +    0u + vtb);         o[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[0]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp +  512u + (vtb ^ 32u)); o[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[1]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 1024u + (vtb ^ 64u)); o[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[2]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 1536u + (vtb ^ 96u)); o[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[3]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 2048u + vtb);         o[4] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[4]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 2560u + (vtb ^ 32u)); o[5] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[5]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 3072u + (vtb ^ 64u)); o[6] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[6]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 3584u + (vtb ^ 96u)); o[7] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[7]); __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) float ink_r14_q4_pf_nf_adj(
    float sv, u32 kb, u32 v_idx, u32 qpos_sub, u32 win, u32 extent_lim, u32 d_base, u32 w) {
  const u32 l_loc = ink_r14_q4_pf_lane();
  const u32 rl_loc = l_loc & 15u, hl_loc = l_loc >> 4u;
  const u32 tp_loc = qpos_sub + rl_loc;
  const u32 kp = kb + (hl_loc << 3u) + v_idx;
  if (kp > tp_loc || (win > 0u && tp_loc - kp >= win)) return -__builtin_inff();
  if (extent_lim > 0u) {
    const u32 dist = tp_loc - kp;
    if (dist < extent_lim) {
      const LDSP f16 *rp = ink_r14_q4_pf_lds.r[w][rl_loc];
      const LDSP f16 *bp = ink_r14_q4_pf_lds.rb + (dist - d_base) * 16u;
      float rbs = 0.0f;
#pragma clang loop unroll(disable)
      for (u32 d = 0u; d < 16u; d += 4u) {
        const ink_r14_q4_h4 rv = *(const LDSP ink_r14_q4_h4 *)(rp + d);
        const ink_r14_q4_h4 bv = *(const LDSP ink_r14_q4_h4 *)(bp + d);
        rbs = __builtin_fmaf((float)rv.x, (float)bv.x, rbs);
        rbs = __builtin_fmaf((float)rv.y, (float)bv.y, rbs);
        rbs = __builtin_fmaf((float)rv.z, (float)bv.z, rbs);
        rbs = __builtin_fmaf((float)rv.w, (float)bv.w, rbs);
      }
      sv += rbs;
    }
  }
  return sv;
}

KERNEL ie_inkling_rel_attn_pf_glob_q4_r14(
    const G float *q, const G u8 *kc_q4, const G u8 *vc_q4,
    const G float *R, const G float *rel_bias,
    const G float *sink, G float *out,
    u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
    u32 d_rel, u32 rel_extent, u32 win,
    u32 qs, u32 rs, u32 ss, u32 os, u32 rot_flags) {
  if (hd != INK_R14_Q4_HD || d_rel != INK_R14_Q4_DREL || n_kv == 0u ||
      n_head != n_kv * INK_R14_Q4_GQA || T == 0u || kv_len == 0u)
    return;

  const u32 ntile = (T + 31u) >> 5u;
  const u32 kh = tokid();
  if (wgid() >= ntile || kh >= n_kv) return;

  const u32 tile = (ntile == 16u && n_kv == 8u && win == 0u && kh < 4u) ? wgid() : ((ntile - 1u) - wgid());
  const u32 t0 = tile << 5u;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));

  const u32 tlast = (t0 + 31u < T) ? (t0 + 31u) : (T - 1u);
  const u32 qpos_min = (u32)__builtin_amdgcn_readfirstlane((int)(pos0 + t0));
  const u32 qpos_max = (u32)__builtin_amdgcn_readfirstlane((int)(pos0 + tlast));
  const u32 k_start = (u32)__builtin_amdgcn_readfirstlane((int)((win > 0u && qpos_min + 1u > win) ? (qpos_min + 1u - win) : 0u));
  const u32 kend = (u32)__builtin_amdgcn_readfirstlane((int)(qpos_max + 1u));
  const u32 nblk = (u32)__builtin_amdgcn_readfirstlane((int)((kend > k_start) ? ((kend - k_start + 15u) >> 4u) : 0u));
  const u32 extent_lim = (u32)__builtin_amdgcn_readfirstlane((int)((win > 0u && win < rel_extent) ? win : rel_extent));
  u32 blk_far_end = 0u;
  if (win == 0u) {
    const u32 guard = extent_lim + 15u;
    if (qpos_min >= guard && qpos_min - guard >= k_start) {
      const u32 bf = ((qpos_min - guard - k_start) >> 4u) + 1u;
      blk_far_end = (bf < nblk) ? bf : nblk;
    }
  }
  blk_far_end = (u32)__builtin_amdgcn_readfirstlane((int)blk_far_end);

  /* Producer waves (6..7): wait if consumers use lds.ku.tmp for Q FWHT-128, then unpack 72-B Q4_0 K/V. */
  if (w >= 6u) {
    if ((rot_flags & 1u) != 0u) lbarrier();
    const ink_r14_q4_u64 row_step = (ink_r14_q4_u64)n_kv * INK_R14_Q4_RECB;
    const G u8 *kb_kh = kc_q4 + (ink_r14_q4_u64)kh * INK_R14_Q4_RECB;
    const G u8 *vb_kh = vc_q4 + (ink_r14_q4_u64)kh * INK_R14_Q4_RECB;
    const int no_wrap = (kend <= kv_len);
    const u32 pt = tid() - 192u;
#pragma clang loop unroll(disable)
    for (u32 b = 0u; b < nblk; b++) {
      ink_r14_q4_pf_prod_blk(kb_kh, vb_kh, rel_bias, row_step, kv_len,
                             k_start + (b << 4u), kend, qpos_min, extent_lim,
                             (b < blk_far_end) ? 1 : 0, no_wrap, pt);
      lbarrier(); /* publish K/V^T/rb to consumers */
      lbarrier(); /* wait for consumers to finish reading LDS before next block */
    }
    return;
  }

  const u32 tsub = w >= 3u ? 1u : 0u;
  const u32 hsub = w - 3u * tsub;
  const u32 t_sub0 = (u32)__builtin_amdgcn_readfirstlane((int)(t0 + 16u * tsub));
  if (t_sub0 >= T) {
    if ((rot_flags & 1u) != 0u) lbarrier();
#pragma clang loop unroll(disable)
    for (u32 b = 0u; b < nblk; b++) { lbarrier(); lbarrier(); }
    return;
  }

  const u32 h = (u32)__builtin_amdgcn_readfirstlane((int)(kh * INK_R14_Q4_GQA + hsub));
  const u32 qpos_sub = (u32)__builtin_amdgcn_readfirstlane((int)(pos0 + t_sub0));

  h8 qa[8];
  float m, ls;
  if ((rot_flags & 1u) != 0u) {
    /* Rotate each of the wave's 16 query tokens via in-register FWHT-128 -> wave-private 512 B LDS staging. */
    const u32 l_init = ink_r14_q4_pf_lane();
    const u32 rl_i = l_init & 15u, hl_i = l_init >> 4u;
    LDSP float *qtmp = ink_r14_q4_pf_lds.ku.tmp[w];
#pragma clang loop unroll(disable)
    for (u32 ti = 0u; ti < 16u; ti++) {
      const u32 tq_t = (t_sub0 + ti < T) ? (t_sub0 + ti) : (T - 1u);
      const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (ink_r14_q4_u64)tq_t * (ink_r14_q4_u64)qs) +
                                             (ink_r14_q4_u64)h * 128ull);
      const f32x4 qr = ink_r14_q4_fwht128_wave4(qp4[l_init], l_init);
      ((LDSP f32x4 *)qtmp)[l_init] = qr;
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
      if (rl_i == ti) {
        const LDSP f32x4 *tp4 = (const LDSP f32x4 *)qtmp;
#pragma unroll
        for (u32 ks_i = 0u; ks_i < 8u; ks_i++) {
          const f32x4 v0 = tp4[4u * ks_i + 2u * hl_i + 0u];
          const f32x4 v1 = tp4[4u * ks_i + 2u * hl_i + 1u];
          qa[ks_i] = (h8){(f16)(v0.x * INK_R14_Q4_QSC), (f16)(v0.y * INK_R14_Q4_QSC),
                          (f16)(v0.z * INK_R14_Q4_QSC), (f16)(v0.w * INK_R14_Q4_QSC),
                          (f16)(v1.x * INK_R14_Q4_QSC), (f16)(v1.y * INK_R14_Q4_QSC),
                          (f16)(v1.z * INK_R14_Q4_QSC), (f16)(v1.w * INK_R14_Q4_QSC)};
        }
      }
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
    }
    const u32 tk_i = t_sub0 + rl_i;
    const u32 tq_i = (tk_i < T) ? tk_i : (T - 1u);
    const float sink_val = ((const G float *)((const G u8 *)sink + (ink_r14_q4_u64)tq_i * (ink_r14_q4_u64)ss))[h];
    m = sink_val * INK_R14_Q4_LOG2E;
    ls = (hl_i == 0u) ? 1.0f : 0.0f;
    lbarrier(); /* release lds.ku.tmp -> lds.ku.k for producer */
  } else {
    const u32 l_init = ink_r14_q4_pf_lane();
    const u32 rl_i = l_init & 15u, hl_i = l_init >> 4u;
    const u32 tk_i = t_sub0 + rl_i;
    const u32 tq_i = (tk_i < T) ? tk_i : (T - 1u);
    const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (ink_r14_q4_u64)tq_i * (ink_r14_q4_u64)qs) +
                                           (ink_r14_q4_u64)h * 128ull);
#pragma unroll
    for (u32 ks_i = 0u; ks_i < 8u; ks_i++) {
      const f32x4 v0 = qp4[4u * ks_i + 2u * hl_i + 0u];
      const f32x4 v1 = qp4[4u * ks_i + 2u * hl_i + 1u];
      qa[ks_i] = (h8){(f16)(v0.x * INK_R14_Q4_QSC), (f16)(v0.y * INK_R14_Q4_QSC),
                      (f16)(v0.z * INK_R14_Q4_QSC), (f16)(v0.w * INK_R14_Q4_QSC),
                      (f16)(v1.x * INK_R14_Q4_QSC), (f16)(v1.y * INK_R14_Q4_QSC),
                      (f16)(v1.z * INK_R14_Q4_QSC), (f16)(v1.w * INK_R14_Q4_QSC)};
    }
    const float sink_val = ((const G float *)((const G u8 *)sink + (ink_r14_q4_u64)tq_i * (ink_r14_q4_u64)ss))[h];
    m = sink_val * INK_R14_Q4_LOG2E;
    ls = (hl_i == 0u) ? 1.0f : 0.0f;
  }

  const v8f zero = (v8f)(0.0f);
  v8f o[8];
#pragma unroll
  for (u32 nt = 0u; nt < 8u; nt++) o[nt] = zero;

  /* Far-Field loop: 0 rel_bias, 0 causal masks, <= 116 VGPR (true 3 WGs/CU!). */
#pragma clang loop unroll(disable)
  for (u32 b = 0u; b < blk_far_end; b++) {
    lbarrier();
    const v8f sa = ink_r14_q4_pf_qk16(qa);

    float mx = sa[0];
#pragma unroll
    for (u32 v = 1u; v < 8u; v++) mx = __builtin_fmaxf(mx, sa[v]);
    mx = __builtin_fmaxf(mx, ink_r14_q4_xrow(mx));
    const float mn = __builtin_fmaxf(m, mx);
    if (mn != m) {
      const float rsc = __builtin_amdgcn_exp2f(m - mn);
#pragma unroll
      for (u32 nt = 0u; nt < 8u; nt++) o[nt] *= rsc;
      ls *= rsc;
      m = mn;
    }
    h8 pb;
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      const float p = __builtin_amdgcn_exp2f(sa[v] - mn);
      ls += p;
      pb[v] = (f16)p;
    }
    ink_r14_q4_pf_pv16(pb, o);
    lbarrier();
  }

  /* Near-Field loop: R*LOG2E in 3.8 KiB LDS + scalar s0..s7 -> <= 120 VGPR (true 3 WG/CU!). */
  if (blk_far_end < nblk) {
    if (extent_lim > 0u) {
      const u32 l_r = ink_r14_q4_pf_lane();
      const u32 rl_r = l_r & 15u, hl_r = l_r >> 4u;
      const u32 tk_r = t_sub0 + rl_r;
      const u32 tq_r = (tk_r < T) ? tk_r : (T - 1u);
      const G f32x4 *rp4 = (const G f32x4 *)((const G float *)((const G u8 *)R + (ink_r14_q4_u64)tq_r * (ink_r14_q4_u64)rs) +
                                             (ink_r14_q4_u64)h * 16ull);
      const f32x4 r0 = rp4[2u * hl_r + 0u];
      const f32x4 r1 = rp4[2u * hl_r + 1u];
      *(LDSP ink_r14_q4_h4 *)(&ink_r14_q4_pf_lds.r[w][rl_r][8u * hl_r + 0u]) =
          (ink_r14_q4_h4){(f16)(r0.x * INK_R14_Q4_LOG2E), (f16)(r0.y * INK_R14_Q4_LOG2E),
                          (f16)(r0.z * INK_R14_Q4_LOG2E), (f16)(r0.w * INK_R14_Q4_LOG2E)};
      *(LDSP ink_r14_q4_h4 *)(&ink_r14_q4_pf_lds.r[w][rl_r][8u * hl_r + 4u]) =
          (ink_r14_q4_h4){(f16)(r1.x * INK_R14_Q4_LOG2E), (f16)(r1.y * INK_R14_Q4_LOG2E),
                          (f16)(r1.z * INK_R14_Q4_LOG2E), (f16)(r1.w * INK_R14_Q4_LOG2E)};
    }

#pragma clang loop unroll(disable)
    for (u32 b = blk_far_end; b < nblk; b++) {
      const u32 kb = (u32)__builtin_amdgcn_readfirstlane((int)(k_start + (b << 4u)));
      lbarrier();
      const v8f sa_v = ink_r14_q4_pf_qk16(qa);

      const int d_min_s = (int)qpos_min - (int)(kb + 15u);
      const u32 d_base = (u32)__builtin_amdgcn_readfirstlane(d_min_s > 0 ? d_min_s : 0);

      float s0 = sa_v[0], s1 = sa_v[1], s2 = sa_v[2], s3 = sa_v[3];
      float s4 = sa_v[4], s5 = sa_v[5], s6 = sa_v[6], s7 = sa_v[7];
      s0 = ink_r14_q4_pf_nf_adj(s0, kb, 0u, qpos_sub, win, extent_lim, d_base, w); __builtin_amdgcn_sched_barrier(0);
      s1 = ink_r14_q4_pf_nf_adj(s1, kb, 1u, qpos_sub, win, extent_lim, d_base, w); __builtin_amdgcn_sched_barrier(0);
      s2 = ink_r14_q4_pf_nf_adj(s2, kb, 2u, qpos_sub, win, extent_lim, d_base, w); __builtin_amdgcn_sched_barrier(0);
      s3 = ink_r14_q4_pf_nf_adj(s3, kb, 3u, qpos_sub, win, extent_lim, d_base, w); __builtin_amdgcn_sched_barrier(0);
      s4 = ink_r14_q4_pf_nf_adj(s4, kb, 4u, qpos_sub, win, extent_lim, d_base, w); __builtin_amdgcn_sched_barrier(0);
      s5 = ink_r14_q4_pf_nf_adj(s5, kb, 5u, qpos_sub, win, extent_lim, d_base, w); __builtin_amdgcn_sched_barrier(0);
      s6 = ink_r14_q4_pf_nf_adj(s6, kb, 6u, qpos_sub, win, extent_lim, d_base, w); __builtin_amdgcn_sched_barrier(0);
      s7 = ink_r14_q4_pf_nf_adj(s7, kb, 7u, qpos_sub, win, extent_lim, d_base, w); __builtin_amdgcn_sched_barrier(0);

      float mx = __builtin_fmaxf(__builtin_fmaxf(__builtin_fmaxf(s0, s1), __builtin_fmaxf(s2, s3)),
                                 __builtin_fmaxf(__builtin_fmaxf(s4, s5), __builtin_fmaxf(s6, s7)));
      mx = __builtin_fmaxf(mx, ink_r14_q4_xrow(mx));
      const float mn = __builtin_fmaxf(m, mx);
      if (mn != m) {
        const float rsc = __builtin_amdgcn_exp2f(m - mn);
#pragma unroll
        for (u32 nt = 0u; nt < 8u; nt++) o[nt] *= rsc;
        ls *= rsc;
        m = mn;
      }
      h8 pb = (h8)(0.0f);
      if (mn > -1e30f) {
        const float p0 = __builtin_amdgcn_exp2f(s0 - mn); ls += p0;
        const float p1 = __builtin_amdgcn_exp2f(s1 - mn); ls += p1;
        const float p2 = __builtin_amdgcn_exp2f(s2 - mn); ls += p2;
        const float p3 = __builtin_amdgcn_exp2f(s3 - mn); ls += p3;
        const float p4 = __builtin_amdgcn_exp2f(s4 - mn); ls += p4;
        const float p5 = __builtin_amdgcn_exp2f(s5 - mn); ls += p5;
        const float p6 = __builtin_amdgcn_exp2f(s6 - mn); ls += p6;
        const float p7 = __builtin_amdgcn_exp2f(s7 - mn); ls += p7;
        pb = (h8){(f16)p0, (f16)p1, (f16)p2, (f16)p3, (f16)p4, (f16)p5, (f16)p6, (f16)p7};
      }
      ink_r14_q4_pf_pv16(pb, o);
      lbarrier();
    }
  }

  /* Producer is 100% done with lds.ku.k after last block's lbarrier(); reuse as lds.ku.tmp[w]! */
  const float inv = 1.0f / (ls + ink_r14_q4_xrow(ls));
  /* Scale o[0..7] IN PLACE before rot_flags branch so LLVM never duplicates 64 VGPRs! */
#pragma unroll
  for (u32 nt = 0u; nt < 8u; nt++) o[nt] *= inv;
  __builtin_amdgcn_sched_barrier(0);

  const u32 l_out = ink_r14_q4_pf_lane();
  const u32 rl_o = l_out & 15u, hl_o = l_out >> 4u;
  if ((rot_flags & 2u) != 0u) {
    LDSP f32x4 *otmp4 = (LDSP f32x4 *)ink_r14_q4_pf_lds.ku.tmp[w];
    const u32 nt_sub = (T - t_sub0 < 16u) ? (T - t_sub0) : 16u;
#pragma clang loop unroll(disable)
    for (u32 ti = 0u; ti < nt_sub; ti++) {
      if (rl_o == ti) {
#pragma unroll
        for (u32 nt = 0u; nt < 8u; nt++) {
          const f32x4 v0 = {o[nt][0], o[nt][1], o[nt][2], o[nt][3]};
          const f32x4 v1 = {o[nt][4], o[nt][5], o[nt][6], o[nt][7]};
          otmp4[4u * nt + 2u * hl_o + 0u] = v0;
          otmp4[4u * nt + 2u * hl_o + 1u] = v1;
        }
      }
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
      const f32x4 ov = ink_r14_q4_fwht128_wave4(otmp4[l_out], l_out);
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
      G f32x4 *op4 = (G f32x4 *)((G float *)((G u8 *)out + (ink_r14_q4_u64)(t_sub0 + ti) * (ink_r14_q4_u64)os) +
                                 (ink_r14_q4_u64)h * 128ull);
      op4[l_out] = ov;
    }
  } else {
    const u32 tk_o = t_sub0 + rl_o;
    if (tk_o < T) {
      G f32x4 *op4 = (G f32x4 *)((G float *)((G u8 *)out + (ink_r14_q4_u64)tk_o * (ink_r14_q4_u64)os) +
                                 (ink_r14_q4_u64)h * 128ull);
#pragma unroll
      for (u32 nt = 0u; nt < 8u; nt++) {
        const f32x4 v0 = {o[nt][0], o[nt][1], o[nt][2], o[nt][3]};
        const f32x4 v1 = {o[nt][4], o[nt][5], o[nt][6], o[nt][7]};
        op4[4u * nt + 2u * hl_o + 0u] = v0;
        op4[4u * nt + 2u * hl_o + 1u] = v1;
      }
    }
  }
}

#undef INK_R14_Q4_VUPD_V
#undef INK_R14_Q4_VUPD
#undef INK_R14_Q4_VRB3
#undef INK_R14_Q4_VQK3
#undef INK_R14_Q4_QSC
#undef INK_R14_Q4_INV_SQRT
#undef INK_R14_Q4_LOG2E
#undef INK_R14_Q4_NB16
#undef INK_R14_Q4_QT
#undef INK_R14_Q4_PSTR
#undef INK_R14_Q4_RECB
#undef INK_R14_Q4_GQA
#undef INK_R14_Q4_DREL
#undef INK_R14_Q4_HD

#endif /* !IE_MEGA */
