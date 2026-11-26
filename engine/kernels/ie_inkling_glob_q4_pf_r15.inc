/* ============================================================================
 * Inkling-2-Mini Round 15 (256K Filled-Context Prefill & inc256 SOTA Suite):
 *   4 Zero-Spill 3-WGs/CU Global Q4_0 + FWHT-128 Prefill Attention Kernels
 *   for RX 9070 XT (gfx1201, wave32, NT=256, 64 CUs, 64 MiB MALL).
 *
 * Included at EOF of kernels/ie_inkling_kernels.c (after ie_kernels.c).
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r15_q4_ /
 * INK_R15_Q4_ and all macros are #undef'd at EOF.
 * 0 warnings (-Wall -Wextra), 0/0 spills, priv=0, scratch=0, TH_LOAD_NT=0,
 * TH_LOAD_HT=0 (R43 default cached loads), scalar __builtin_bit_cast safe.
 *
 * Exact Global Q4_0 KV Record Format (matches ie_inkling_glob_q4kv.inc 100%):
 *   - 72 B per (pos, kh) record (hd == 128, n_kv == 8, 8-B aligned):
 *       [ 0 .. 63]: 64 B packed 4-bit unsigned nibbles q in [0..15]
 *                   (even elem low nibble, odd elem high nibble;
 *                    signed val = s * (float)((int)q - 8), exact 0 at q=8)
 *       [64 .. 71]:  8 B = 4 x f16 scales (1 per 32-elem block)
 *   - Built-in Orthogonal Normalized FWHT-128 (u32 rot_flags: 0=plain, 1=K/Q,
 *     2=V/O, 3=K+V). All KV byte offsets use 64-bit unsigned long long math
 *     with strict kp < kend bounds checks (slack=0 safe up to 524,288+ pos).
 *
 * Key Architectural Upgrades over ie_inkling_rel_attn_pf_glob_q4_r14:
 *   1. True Async Producer-Consumer Pipeline: replaces r14's single-buffered
 *      2-barrier lockstep with double-buffered k[2]/vt[2]/rb[2] (19.0 KiB LDS)
 *      and D=2 raw-Q4 register prefetch ring (only 14 VGPR/block!) + gfx1201
 *      decoupled split barriers (s_barrier_signal/wait, 1 barrier/16 keys!).
 *   2. 2-Deep Pipelined QK16 + 2-Deep Pipelined PV16 WMMA schedules hiding
 *      ds_load_b128 latency behind v_wmma_f32_16x16x16_f16.
 *   3. Exact Wave-Level Conditional Rescale Skip (ballot_w32(bm > m)) +
 *      Zero-Tile PV Skip (bm - m < -30.0f) + Zero-Modulo Producer + Zero-LDS
 *      R*rel_bias via wave xrow_i exchange -> <= 120 VGPR (true 3 WGs/CU!).
 *   4. Dedicated Zero-rb NB=32 Far-Field + Near-Field Merge Pair (16,384 B LDS,
 *      2x fewer barriers on 99.6% of 256K KV blocks, reuses b->at_part up to
 *      T=1024 with 0 B extra VRAM!).
 *   5. Split-KV Incremental Prefill (ie_inkling_rel_attn_pf_split_glob_q4_r15):
 *      splits 261K KV across nsplit=2..16 (256 WGs = 4 WGs/CU for inc256!),
 *      reusing b->at_part (12.19 MiB = 0 extra VRAM) with in-register inverse
 *      FWHT-128 in the self-resetting final merge.
 *
 * Exported Kernels (4):
 *   1. ie_inkling_rel_attn_pf_split_glob_q4_r15 (<= 120 VGPR, 3 WG/CU, 19,472 B LDS)
 *   2. ie_inkling_rel_attn_pf_far_glob_q4_r15   (<= 120 VGPR, 3 WG/CU, 16,384 B LDS)
 *   3. ie_inkling_rel_attn_pf_near_glob_q4_r15  (<= 120 VGPR, 3 WG/CU, 19,456 B LDS)
 *   4. ie_inkling_rel_attn_pf_glob_q4_fast_r15  (<= 120 VGPR, 3 WG/CU, 19,456 B LDS)
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(f16) == 2u, "ink_r15_q4 f16 must be 2 bytes");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r15_q4 float/u32 must be 4 bytes");

typedef unsigned long long ink_r15_q4_u64;
typedef unsigned int       ink_r15_q4_u2 __attribute__((ext_vector_type(2)));
typedef float              ink_r15_q4_f2 __attribute__((ext_vector_type(2)));
typedef f16                ink_r15_q4_h4 __attribute__((ext_vector_type(4)));

_Static_assert(sizeof(ink_r15_q4_u64) == 8u, "ink_r15_q4_u64 must be 8 bytes");

#define INK_R15_Q4_HD       128u
#define INK_R15_Q4_DREL     16u
#define INK_R15_Q4_GQA      3u
#define INK_R15_Q4_RECB     72u
#define INK_R15_Q4_PSTR     132u
#define INK_R15_Q4_NB16     16u
#define INK_R15_Q4_NB32     32u
#define INK_R15_Q4_LOG2E    1.44269504088896340736f
#define INK_R15_Q4_LN2      0.69314718055994530942f
#define INK_R15_Q4_INV_SQRT 0.08838834764831844299f /* 1 / sqrt(128) */
#define INK_R15_Q4_QSC      (0.08838834764831844299f * 1.44269504088896340736f)
#define INK_R15_Q4_EXP_C1   0.693124235f
#define INK_R15_Q4_EXP_C2   0.240240976f
#define INK_R15_Q4_EXP_C3   0.0559059493f
#define INK_R15_Q4_EXP_C4   0.00958269928f
#define INK_R15_Q4_EXP_MAG  12582912.0f

static inline __attribute__((always_inline)) void ink_r15_q4_exp2_poly_pair(
    const float xa, const float xb, float *ra_out, float *rb_out) {
  const float ca = __builtin_fmaxf(xa, -127.0f);
  const float cb = __builtin_fmaxf(xb, -127.0f);
  float za = ca + INK_R15_Q4_EXP_MAG;
  float zb = cb + INK_R15_Q4_EXP_MAG;
  __asm__ volatile("" : "+v"(za), "+v"(zb));
  const float fa = ca - (za - INK_R15_Q4_EXP_MAG);
  const float fb = cb - (zb - INK_R15_Q4_EXP_MAG);
  const u32 ua = __builtin_bit_cast(u32, za);
  const u32 ub = __builtin_bit_cast(u32, zb);
  const float sa = __builtin_bit_cast(float, (ua << 23u) + 0x3F800000u);
  const float sb = __builtin_bit_cast(float, (ub << 23u) + 0x3F800000u);
  float pa = __builtin_fmaf(INK_R15_Q4_EXP_C4, fa, INK_R15_Q4_EXP_C3);
  float pb = __builtin_fmaf(INK_R15_Q4_EXP_C4, fb, INK_R15_Q4_EXP_C3);
  pa = __builtin_fmaf(pa, fa, INK_R15_Q4_EXP_C2);
  pb = __builtin_fmaf(pb, fb, INK_R15_Q4_EXP_C2);
  pa = __builtin_fmaf(pa, fa, INK_R15_Q4_EXP_C1);
  pb = __builtin_fmaf(pb, fb, INK_R15_Q4_EXP_C1);
  pa = __builtin_fmaf(pa, fa, 1.0f);
  pb = __builtin_fmaf(pb, fb, 1.0f);
  float ra = pa * sa;
  float rb = pb * sb;
  __asm__ volatile("" : "+v"(ra), "+v"(rb));
  *ra_out = ra;
  *rb_out = rb;
}

static inline __attribute__((always_inline)) float ink_r15_q4_exp2_interleaved8(
    const float s0, const float s1, const float s2, const float s3,
    const float s4, const float s5, const float s6, const float s7,
    const float mn, h8 *pb_out) {
  float p0, p2, p4, p6;
  const float p1 = __builtin_amdgcn_exp2f(s1 - mn);
  const float p3 = __builtin_amdgcn_exp2f(s3 - mn);
  ink_r15_q4_exp2_poly_pair(s0 - mn, s2 - mn, &p0, &p2);
  float ps = (p0 + p1) + (p2 + p3);
  const f16 h0 = (f16)p0, h1 = (f16)p1, h2 = (f16)p2, h3 = (f16)p3;
  __builtin_amdgcn_sched_barrier(0);
  const float p5 = __builtin_amdgcn_exp2f(s5 - mn);
  const float p7 = __builtin_amdgcn_exp2f(s7 - mn);
  ink_r15_q4_exp2_poly_pair(s4 - mn, s6 - mn, &p4, &p6);
  ps += (p4 + p5) + (p6 + p7);
  *pb_out = (h8){h0, h1, h2, h3, (f16)p4, (f16)p5, (f16)p6, (f16)p7};
  return ps;
}

/* ============================================================================
 * Shared Low-Level Primitives (Split Barriers, Swizzles, FWHT-128, Q4 Dequant)
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r15_q4_lsignal(void) {
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  __builtin_amdgcn_s_barrier_signal(-1);
  __asm__ volatile("" ::: "memory");
}

static inline __attribute__((always_inline)) void ink_r15_q4_lwait(void) {
  __builtin_amdgcn_s_barrier_wait(-1);
  __asm__ volatile("" ::: "memory");
}

/* Uniform full-exec lane ID (0..31) preventing LLVM LICM from hoisting LDS
 * addresses out of K-loops (saves 8-10 loop-carried VGPRs -> true 3 WGs/CU).
 * ONLY call where all 32 lanes are active (exec_lo == 0xffffffff). */
static inline __attribute__((always_inline)) u32 ink_r15_q4_lane(void) {
  u32 l;
  __asm__ volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(l));
  return l;
}

static inline __attribute__((always_inline)) float ink_r15_q4_xrow(float v) {
  return I2F(xrow_i(F2I(v)));
}

static inline __attribute__((always_inline)) u32 ink_r15_q4_k_off(u32 r, u32 c) {
  return r * 128u + 8u * (c ^ (r & 15u));
}

static inline __attribute__((always_inline)) u32 ink_r15_q4_vt_off(u32 d, u32 h) {
  return 8u * ((2u * d + h) ^ ((d >> 2u) & 1u) ^ ((d >> 3u) & 7u));
}

static inline __attribute__((always_inline)) u32 ink_r15_q4_wrap_s(u32 v, const u32 kv_len) {
  if (kv_len >= 16u) return (v >= kv_len) ? (v - kv_len) : v;
  while (v >= kv_len) v -= kv_len;
  return v;
}

static inline __attribute__((always_inline)) u32 ink_r15_q4_kv_adv16(u32 kv_b, const u32 kv_len) {
  return ink_r15_q4_wrap_s(kv_b + INK_R15_Q4_NB16, kv_len);
}

/* Exact NB=32 far-field block count (guaranteed no-wrap, zero rel_bias, and
 * leaves >= 1 key for near_glob_q4_r15 even if rel_extent == 0). */
static inline __attribute__((always_inline)) u32 ink_r15_q4_far_nblk32(
    const u32 qpos_min, const u32 rel_extent, const u32 kv_len) {
  const u32 eff_ext = (rel_extent > 0u) ? rel_extent : 1u;
  if (qpos_min + 1u <= eff_ext) return 0u;
  const u32 lim = qpos_min + 1u - eff_ext;
  const u32 lim_nw = (lim <= kv_len) ? lim : kv_len;
  return lim_nw >> 5u;
}

/* Exact NB=16 far-field block boundary in [0, nblk]. */
static inline __attribute__((always_inline)) u32 ink_r15_q4_calc_far_end16(
    const u32 win, const u32 qpos_min, const u32 extent_lim, const u32 k_start, const u32 nblk) {
  if (win > 0u) return 0u;
  const u32 guard = extent_lim + (INK_R15_Q4_NB16 - 1u);
  if (qpos_min < guard || qpos_min - guard < k_start) return 0u;
  const u32 b_far = ((qpos_min - guard - k_start) >> 4u) + 1u;
  return (b_far < nblk) ? b_far : nblk;
}

/* Zero-LDS, zero-barrier exact normalized 128-pt Walsh-Hadamard transform on a
 * single wave32 holding 128 floats as f32x4 per lane (matches ie_inkling_glob_q4kv.inc 100%). */
static inline __attribute__((always_inline)) f32x4 ink_r15_q4_fwht128_wave4(f32x4 v, u32 l) {
  const float a0 = v.x + v.y, a1 = v.x - v.y;
  const float a2 = v.z + v.w, a3 = v.z - v.w;
  f32x4 r = {a0 + a2, a1 + a3, a0 - a2, a1 - a3};
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
  return r * INK_R15_Q4_INV_SQRT;
}

static inline __attribute__((always_inline)) void ink_r15_q4_t8(
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

/* Exact bit-identical Q4_0 4-nibble dequant -> packed f16x4 (ink_r15_q4_u2).
 * Because f16 scale has 11 mantissa bits and qi in [-8..7] has <= 3 bits,
 * (f16)((float)sh * (float)qi) == sh * (f16)qi is 100% exact IEEE-754! */
static inline __attribute__((always_inline)) ink_r15_q4_u2 ink_r15_q4_deq4n_h4(u32 w, u16 s_bits) {
  const f16 sh = __builtin_bit_cast(f16, s_bits);
  const ink_r15_q4_h4 h = {sh * (f16)((int)(w & 15u) - 8),
                           sh * (f16)((int)((w >> 4u) & 15u) - 8),
                           sh * (f16)((int)((w >> 8u) & 15u) - 8),
                           sh * (f16)((int)((w >> 12u) & 15u) - 8)};
  return __builtin_bit_cast(ink_r15_q4_u2, h);
}

/* Load Q into qa[0..7] (with optional prologue FWHT-128 using wave-private tmp[w]). */
static inline __attribute__((always_inline)) void ink_r15_q4_load_q(
    const G float *q, u32 qs, u32 t_sub0, u32 T, u32 h, u32 rot_flags,
    LDSP float *qtmp, h8 qa[8]) {
  const u32 l_init = ink_r15_q4_lane();
  const u32 rl_i = l_init & 15u, hl_i = l_init >> 4u;
  if ((rot_flags & 1u) != 0u) {
#pragma clang loop unroll(disable)
    for (u32 ti = 0u; ti < 16u; ti++) {
      const u32 tq_t = (t_sub0 + ti < T) ? (t_sub0 + ti) : (T - 1u);
      const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (ink_r15_q4_u64)tq_t * (ink_r15_q4_u64)qs) +
                                             (ink_r15_q4_u64)h * 128ull);
      const f32x4 qr = ink_r15_q4_fwht128_wave4(qp4[l_init], l_init);
      ((LDSP f32x4 *)qtmp)[l_init] = qr;
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
      if (rl_i == ti) {
        const LDSP f32x4 *tp4 = (const LDSP f32x4 *)qtmp;
#pragma unroll
        for (u32 ks_i = 0u; ks_i < 8u; ks_i++) {
          const f32x4 v0 = tp4[4u * ks_i + 2u * hl_i + 0u];
          const f32x4 v1 = tp4[4u * ks_i + 2u * hl_i + 1u];
          qa[ks_i] = (h8){(f16)(v0.x * INK_R15_Q4_QSC), (f16)(v0.y * INK_R15_Q4_QSC),
                          (f16)(v0.z * INK_R15_Q4_QSC), (f16)(v0.w * INK_R15_Q4_QSC),
                          (f16)(v1.x * INK_R15_Q4_QSC), (f16)(v1.y * INK_R15_Q4_QSC),
                          (f16)(v1.z * INK_R15_Q4_QSC), (f16)(v1.w * INK_R15_Q4_QSC)};
        }
      }
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
    }
  } else {
    const u32 tk_i = t_sub0 + rl_i;
    const u32 tq_i = (tk_i < T) ? tk_i : (T - 1u);
    const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (ink_r15_q4_u64)tq_i * (ink_r15_q4_u64)qs) +
                                           (ink_r15_q4_u64)h * 128ull);
#pragma unroll
    for (u32 ks_i = 0u; ks_i < 8u; ks_i++) {
      const f32x4 v0 = qp4[4u * ks_i + 2u * hl_i + 0u];
      const f32x4 v1 = qp4[4u * ks_i + 2u * hl_i + 1u];
      qa[ks_i] = (h8){(f16)(v0.x * INK_R15_Q4_QSC), (f16)(v0.y * INK_R15_Q4_QSC),
                      (f16)(v0.z * INK_R15_Q4_QSC), (f16)(v0.w * INK_R15_Q4_QSC),
                      (f16)(v1.x * INK_R15_Q4_QSC), (f16)(v1.y * INK_R15_Q4_QSC),
                      (f16)(v1.z * INK_R15_Q4_QSC), (f16)(v1.w * INK_R15_Q4_QSC)};
      __builtin_amdgcn_sched_barrier(0);
    }
  }
}

/* Epilogue: normalize o[0..7] in place, optionally apply inverse FWHT-128 via
 * wave-private otmp (caller MUST ensure lds.k[0] is free via lbarrier()!), store out. */
static inline __attribute__((always_inline)) void ink_r15_q4_store_o(
    G float *out, u32 os, u32 t_sub0, u32 T, u32 h, u32 rot_flags,
    LDSP float *otmp, float ls, v8f o[8]) {
  const float inv = 1.0f / (ls + ink_r15_q4_xrow(ls));
#pragma unroll
  for (u32 nt = 0u; nt < 8u; nt++) o[nt] *= inv;
  __builtin_amdgcn_sched_barrier(0);

  const u32 l_out = ink_r15_q4_lane();
  const u32 rl_o = l_out & 15u, hl_o = l_out >> 4u;
  if ((rot_flags & 2u) != 0u) {
    LDSP f32x4 *otmp4 = (LDSP f32x4 *)otmp;
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
      const f32x4 ov = ink_r15_q4_fwht128_wave4(otmp4[l_out], l_out);
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
      G f32x4 *op4 = (G f32x4 *)((G float *)((G u8 *)out + (ink_r15_q4_u64)(t_sub0 + ti) * (ink_r15_q4_u64)os) +
                                 (ink_r15_q4_u64)h * 128ull);
      op4[l_out] = ov;
    }
  } else {
    const u32 tk_o = t_sub0 + rl_o;
    if (tk_o < T) {
      G f32x4 *op4 = (G f32x4 *)((G float *)((G u8 *)out + (ink_r15_q4_u64)tk_o * (ink_r15_q4_u64)os) +
                                 (ink_r15_q4_u64)h * 128ull);
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

/* ============================================================================
 * Shared Consumer WMMA Building Blocks (2-Deep QK16, 1-Deep QK16_S1, 2-Deep PV16)
 * ========================================================================== */

static inline __attribute__((always_inline)) v8f ink_r15_q4_qk16_2d(
    const LDSP f16 *k_lds, const h8 qa[8]) {
  const u32 l_loc = ink_r15_q4_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  u32 kb_b = (rl << 8u) | ((rl ^ hl) << 4u);
  const LDSP u8 *kp = (const LDSP u8 *)k_lds;

  v8f sa = (v8f)(0.0f);
  h8 k0 = *(const LDSP h8 *)(kp + kb_b); kb_b ^= 32u;
  h8 k1 = *(const LDSP h8 *)(kp + kb_b); kb_b ^= 96u;
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[0], sa);
  k0 = *(const LDSP h8 *)(kp + kb_b); kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[1], sa);
  k1 = *(const LDSP h8 *)(kp + kb_b); kb_b ^= 224u;
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[2], sa);
  k0 = *(const LDSP h8 *)(kp + kb_b); kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[3], sa);
  k1 = *(const LDSP h8 *)(kp + kb_b); kb_b ^= 96u;
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[4], sa);
  k0 = *(const LDSP h8 *)(kp + kb_b); kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[5], sa);
  k1 = *(const LDSP h8 *)(kp + kb_b);
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[6], sa);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[7], sa);
  __builtin_amdgcn_sched_barrier(0);
  return sa;
}

static inline __attribute__((always_inline)) v8f ink_r15_q4_qk16_s1_sig(
    const LDSP f16 *k_lds, const h8 qa[8]) {
  const u32 l_loc = ink_r15_q4_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  u32 kb_b = (rl << 8u) | ((rl ^ hl) << 4u);
  const LDSP u8 *kp = (const LDSP u8 *)k_lds;

  v8f sa = (v8f)(0.0f);
  h8 k0 = *(const LDSP h8 *)(kp + kb_b); kb_b ^= 32u;
  h8 k1 = *(const LDSP h8 *)(kp + kb_b); kb_b ^= 96u;
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[0], sa);
  k0 = *(const LDSP h8 *)(kp + kb_b); kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[1], sa);
  k1 = *(const LDSP h8 *)(kp + kb_b); kb_b ^= 224u;
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[2], sa);
  k0 = *(const LDSP h8 *)(kp + kb_b); kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[3], sa);
  k1 = *(const LDSP h8 *)(kp + kb_b); kb_b ^= 96u;
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[4], sa);
  k0 = *(const LDSP h8 *)(kp + kb_b); kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[5], sa);
  k1 = *(const LDSP h8 *)(kp + kb_b);
  ink_r15_q4_lsignal(); /* K LDS reads complete -> signal before final 2 QK WMMAs! */

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[6], sa);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[7], sa);
  __builtin_amdgcn_sched_barrier(0);
  return sa;
}

static inline __attribute__((always_inline)) v8f ink_r15_q4_qk16_1d(
    const LDSP f16 *k_lds, const h8 qa[8]) {
  const u32 l_loc = ink_r15_q4_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  u32 kb_b = (rl << 8u) | ((rl ^ hl) << 4u);
  const LDSP u8 *kp = (const LDSP u8 *)k_lds;

  v8f sa = (v8f)(0.0f);
  h8 k0;
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[0], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[1], sa); kb_b ^= 96u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[2], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[3], sa); kb_b ^= 224u; __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[4], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[5], sa); kb_b ^= 96u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[6], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[7], sa); __builtin_amdgcn_sched_barrier(0);
  return sa;
}

static inline __attribute__((always_inline)) void ink_r15_q4_pv16_1d(
    const LDSP f16 *vt_lds, const h8 pb, v8f o[8], const int do_signal) {
  const u32 l_loc = ink_r15_q4_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2u) & 1u) ^ (rl >> 3u);
  const u32 vtb = 16u * vt_lane;
  const LDSP u8 *vp = (const LDSP u8 *)vt_lds;
  h8 v0;
  v0 = *(const LDSP h8 *)(vp +    0u + vtb);         o[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[0]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp +  512u + (vtb ^ 32u)); o[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[1]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 1024u + (vtb ^ 64u)); o[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[2]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 1536u + (vtb ^ 96u)); o[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[3]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 2048u + vtb);         o[4] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[4]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 2560u + (vtb ^ 32u)); o[5] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[5]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 3072u + (vtb ^ 64u)); o[6] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[6]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 3584u + (vtb ^ 96u));
  if (do_signal) {
    ink_r15_q4_lsignal();
  } else {
    __builtin_amdgcn_sched_barrier(0);
  }
  o[7] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[7]);
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r15_q4_pv16_2d(
    const LDSP f16 *vt_lds, const h8 pb, v8f o[8], const int do_signal) {
  const u32 l_loc = ink_r15_q4_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2u) & 1u) ^ (rl >> 3u);
  u32 vtb = 16u * vt_lane;
  const LDSP u8 *vp = (const LDSP u8 *)vt_lds;

  h8 v0 = *(const LDSP h8 *)(vp + 0u + vtb); vtb ^= 32u;
  h8 v1 = *(const LDSP h8 *)(vp + 512u + vtb); vtb ^= 96u;
  __builtin_amdgcn_sched_barrier(0);

  o[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[0]);
  v0 = *(const LDSP h8 *)(vp + 1024u + vtb); vtb ^= 32u;
  __builtin_amdgcn_sched_barrier(0);
  o[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[1]);
  v1 = *(const LDSP h8 *)(vp + 1536u + vtb); vtb ^= 96u;
  __builtin_amdgcn_sched_barrier(0);

  o[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[2]);
  v0 = *(const LDSP h8 *)(vp + 2048u + vtb); vtb ^= 32u;
  __builtin_amdgcn_sched_barrier(0);
  o[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[3]);
  v1 = *(const LDSP h8 *)(vp + 2560u + vtb); vtb ^= 96u;
  __builtin_amdgcn_sched_barrier(0);

  o[4] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[4]);
  v0 = *(const LDSP h8 *)(vp + 3072u + vtb); vtb ^= 32u;
  __builtin_amdgcn_sched_barrier(0);
  o[5] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[5]);
  v1 = *(const LDSP h8 *)(vp + 3584u + vtb);
  if (do_signal) {
    ink_r15_q4_lsignal(); /* VT LDS reads complete -> signal before final 2 PV WMMAs! */
  } else {
    __builtin_amdgcn_sched_barrier(0);
  }

  o[6] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[6]);
  o[7] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[7]);
  __builtin_amdgcn_sched_barrier(0);
}

/* ============================================================================
 * PART A: Dedicated Zero-rb NB=32 Q4_0+FWHT-128 Far-Field Prefill (16,384 B LDS)
 *         Kernel: ie_inkling_rel_attn_pf_far_glob_q4_r15
 * ========================================================================== */

typedef struct {
  union {
    f16   k[INK_R15_Q4_NB32 * INK_R15_Q4_HD];  /* 8,192 B: 32 x 128 XOR-swizzled f16 K */
    float tmp[6u][INK_R15_Q4_HD];              /* 3,072 B: Prologue Q FWHT-128 staging */
  } ku;
  f16 vt[2u][INK_R15_Q4_HD * INK_R15_Q4_NB16]; /* 8,192 B: two 128 x 16 XOR-swizzled V^T */
} ink_r15_q4_far_lds_t;

_Static_assert(sizeof(ink_r15_q4_far_lds_t) == 16384u, "ink_r15_q4_far_lds_t must be 16,384 B");

typedef struct {
  u32 kw[8];
  u16 ks[8];
} ink_r15_q4_raw_k32_t;

typedef struct {
  u16 vw0[8], vs0[8];
  u16 vw1[8], vs1[8];
} ink_r15_q4_raw_v32_t;

static inline __attribute__((always_inline)) void ink_r15_q4_far_load_k32(
    ink_r15_q4_raw_k32_t *rk, const G u8 *kb_kh, const ink_r15_q4_u64 row_step,
    const u32 kb, const u32 pt) {
  const u32 c = pt & 15u;
  const u32 r0 = pt >> 4u;
  const u32 blk_c = c >> 2u;
#pragma unroll
  for (u32 i = 0u; i < 8u; i++) {
    const G u8 *rec = kb_kh + (ink_r15_q4_u64)(kb + r0 + 4u * i) * row_step;
    rk->kw[i] = ((const G u32 *)rec)[c];
    rk->ks[i] = ((const G u16 *)(rec + 64u))[blk_c];
  }
}

static inline __attribute__((always_inline)) void ink_r15_q4_far_pub_k32(
    LDSP f16 *k_lds, const ink_r15_q4_raw_k32_t *rk, const u32 pt) {
  const u32 c = pt & 15u;
  const u32 r0 = pt >> 4u;
#pragma unroll
  for (u32 i = 0u; i < 8u; i++) {
    const u32 r = r0 + 4u * i;
    const u32 w8n = rk->kw[i];
    const u16 s_bits = rk->ks[i];
    const ink_r15_q4_u2 h0 = ink_r15_q4_deq4n_h4(w8n, s_bits);
    const ink_r15_q4_u2 h1 = ink_r15_q4_deq4n_h4(w8n >> 16u, s_bits);
    *(LDSP u32x4 *)(k_lds + ink_r15_q4_k_off(r, c)) = (u32x4){h0.x, h0.y, h1.x, h1.y};
  }
}

static inline __attribute__((always_inline)) void ink_r15_q4_far_load_v32(
    ink_r15_q4_raw_v32_t *rv, const G u8 *vb_kh, const ink_r15_q4_u64 row_step,
    const u32 kb, const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 blk_v = L >> 3u;
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const G u8 *r0 = vb_kh + (ink_r15_q4_u64)(kb + 8u * wi + u) * row_step;
    const G u8 *r1 = vb_kh + (ink_r15_q4_u64)(kb + 16u + 8u * wi + u) * row_step;
    rv->vw0[u] = ((const G u16 *)r0)[L];
    rv->vs0[u] = ((const G u16 *)(r0 + 64u))[blk_v];
    rv->vw1[u] = ((const G u16 *)r1)[L];
    rv->vs1[u] = ((const G u16 *)(r1 + 64u))[blk_v];
  }
}

static inline __attribute__((always_inline)) void ink_r15_q4_far_pub_vt16(
    LDSP f16 *vt_lds, const u16 vw[8], const u16 vs[8], const u32 wi, const u32 L) {
  u32 vx0[8], vx1[8];
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const ink_r15_q4_u2 h4 = ink_r15_q4_deq4n_h4((u32)vw[u], vs[u]);
    vx0[u] = h4.x;
    vx1[u] = h4.y;
  }
  u32x4 ev0, od0, ev1, od1;
  ink_r15_q4_t8(vx0, &ev0, &od0);
  ink_r15_q4_t8(vx1, &ev1, &od1);
  const u32 c0 = 4u * L;
  *(LDSP u32x4 *)(vt_lds + ink_r15_q4_vt_off(c0 + 0u, wi)) = ev0;
  *(LDSP u32x4 *)(vt_lds + ink_r15_q4_vt_off(c0 + 1u, wi)) = od0;
  *(LDSP u32x4 *)(vt_lds + ink_r15_q4_vt_off(c0 + 2u, wi)) = ev1;
  *(LDSP u32x4 *)(vt_lds + ink_r15_q4_vt_off(c0 + 3u, wi)) = od1;
}

static inline __attribute__((always_inline)) void ink_r15_q4_far_blk32_step(
    const LDSP f16 *k_lds, const LDSP f16 *vt0, const LDSP f16 *vt1,
    const h8 qa[8], const int has_next, float *m_p, float *ls_p, v8f o[8]) {
  float m = *m_p;
  float ls = *ls_p;

  /* Half 0: QK_0 + exact conditional-rescale softmax -> frees s0 before QK_1! */
  const v8f s0 = ink_r15_q4_qk16_2d(k_lds, qa);
  h8 pb0 = (h8)(0.0f);
  int active0 = 0;
  {
    float bm0 = s0[0];
#pragma unroll
    for (u32 v = 1u; v < 8u; v++) bm0 = __builtin_fmaxf(bm0, s0[v]);
    bm0 = __builtin_fmaxf(bm0, ink_r15_q4_xrow(bm0));
    active0 = (__builtin_amdgcn_ballot_w32(bm0 > -1e30f && !(bm0 - m < -30.0f)) != 0u);
    if (active0) {
      const int nr0 = (__builtin_amdgcn_ballot_w32(bm0 > m || m <= -1e30f) != 0u);
      const float mn0 = nr0 ? __builtin_fmaxf(m, bm0) : m;
      float ps0 = 0.0f;
      if (mn0 > -1e30f) {
        ps0 = ink_r15_q4_exp2_interleaved8(s0[0], s0[1], s0[2], s0[3], s0[4], s0[5], s0[6], s0[7], mn0, &pb0);
      }
      float c0 = 1.0f;
      if (nr0) {
        c0 = (m > -1e30f) ? __builtin_amdgcn_exp2f(m - mn0) : 0.0f;
#pragma unroll
        for (u32 nt = 0u; nt < 8u; nt++)
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c0;
      }
      ls = ls * c0 + ps0;
      m = mn0;
    }
  }
  __builtin_amdgcn_sched_barrier(0);

  /* Half 1: QK_1 (signals K consumed before final WMMA) + softmax -> frees s1! */
  const v8f s1 = ink_r15_q4_qk16_s1_sig(k_lds + 2048u, qa);
  h8 pb1 = (h8)(0.0f);
  float c1 = 1.0f;
  int active1 = 0, nr1 = 0;
  {
    float bm1 = s1[0];
#pragma unroll
    for (u32 v = 1u; v < 8u; v++) bm1 = __builtin_fmaxf(bm1, s1[v]);
    bm1 = __builtin_fmaxf(bm1, ink_r15_q4_xrow(bm1));
    active1 = (__builtin_amdgcn_ballot_w32(bm1 > -1e30f && !(bm1 - m < -30.0f)) != 0u);
    if (active1) {
      nr1 = (__builtin_amdgcn_ballot_w32(bm1 > m || m <= -1e30f) != 0u);
      const float mn1 = nr1 ? __builtin_fmaxf(m, bm1) : m;
      float ps1 = 0.0f;
      if (mn1 > -1e30f) {
        ps1 = ink_r15_q4_exp2_interleaved8(s1[0], s1[1], s1[2], s1[3], s1[4], s1[5], s1[6], s1[7], mn1, &pb1);
      }
      if (nr1) c1 = (m > -1e30f) ? __builtin_amdgcn_exp2f(m - mn1) : 0.0f;
      ls = ls * c1 + ps1;
      m = mn1;
    }
  }
  *m_p = m;
  *ls_p = ls;

  ink_r15_q4_lwait(); /* Wait for VT[0..1] ready (both QKs + softmaxes overlapped!) */

  if (active0) ink_r15_q4_pv16_2d(vt0, pb0, o, 0);
  if (active1) {
    if (nr1) {
#pragma unroll
      for (u32 nt = 0u; nt < 8u; nt++)
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c1;
    }
    ink_r15_q4_pv16_2d(vt1, pb1, o, has_next);
  }
  if (has_next) {
    if (!active1) ink_r15_q4_lsignal();
    ink_r15_q4_lwait();
  }
}

/* ============================================================================
 * Kernel 2A: ie_inkling_rel_attn_pf_far_glob_q4_r15
 *   Dedicated Zero-rb NB=32 Global Q4_0+FWHT-128 Far-Field Prefill.
 *   Grid:   ((T + 31u) / 32u, n_kv, 1u), NT = 256.
 *   Buffer: part = b->at_part (stride 130 floats, fits T <= 1024 in 12.19 MiB!).
 *   Output: (M_far [log2], L_far, O_far [unnormalized, rotated if rot_flags & 2]).
 * ========================================================================== */
__attribute__((amdgpu_waves_per_eu(12, 16)))
KERNEL ie_inkling_rel_attn_pf_far_glob_q4_r15(
    const G float *q, const G u8 *kc_q4, const G u8 *vc_q4,
    const G float *sink, G float *part,
    u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
    u32 rel_extent, u32 qs, u32 ss, u32 rot_flags) {
  static LDS ink_r15_q4_far_lds_t ink_r15_q4_far_lds __attribute__((aligned(16)));
  if (hd != INK_R15_Q4_HD || n_kv == 0u || n_head != n_kv * INK_R15_Q4_GQA ||
      T == 0u || kv_len == 0u || !q || !kc_q4 || !vc_q4 || !sink || !part)
    return;

  const u32 ntile = (T + 31u) >> 5u;
  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  if (wgid() >= ntile || kh >= n_kv) return;

  const u32 tile = (ntile == 16u && n_kv == 8u && kh < 4u) ? wgid() : ((ntile - 1u) - wgid());
  const u32 t0 = tile << 5u;
  const u32 qpos_min = pos0 + t0;
  const u32 nblk32 = (u32)__builtin_amdgcn_readfirstlane((int)ink_r15_q4_far_nblk32(qpos_min, rel_extent, kv_len));
  if (nblk32 == 0u) return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));

  if (w >= 6u) {
    const u32 pt = tid() - 192u;
    const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
    const u32 L = pt & 31u;
    const ink_r15_q4_u64 row_step = (ink_r15_q4_u64)n_kv * INK_R15_Q4_RECB;
    const G u8 *kb_kh = kc_q4 + (ink_r15_q4_u64)kh * INK_R15_Q4_RECB;
    const G u8 *vb_kh = vc_q4 + (ink_r15_q4_u64)kh * INK_R15_Q4_RECB;

    ink_r15_q4_raw_k32_t rk;
    ink_r15_q4_raw_v32_t rv;

    /* Prefetch block 0 raw Q4 K while consumers run Q FWHT-128 (if rot_flags & 1). */
    ink_r15_q4_far_load_k32(&rk, kb_kh, row_step, 0u, pt);
    if ((rot_flags & 1u) != 0u) lbarrier();
    __builtin_amdgcn_sched_barrier(0);

    ink_r15_q4_far_pub_k32(ink_r15_q4_far_lds.ku.k, &rk, pt);
    ink_r15_q4_lsignal();
    ink_r15_q4_far_load_v32(&rv, vb_kh, row_step, 0u, pt);
    if (1u < nblk32) ink_r15_q4_far_load_k32(&rk, kb_kh, row_step, 32u, pt);
    ink_r15_q4_lwait();

#pragma clang loop unroll(disable)
    for (u32 b = 0u; b < nblk32; b++) {
      const u32 b_u = (u32)__builtin_amdgcn_readfirstlane((int)b);
      ink_r15_q4_far_pub_vt16(ink_r15_q4_far_lds.vt[0], rv.vw0, rv.vs0, wi, L);
      ink_r15_q4_far_pub_vt16(ink_r15_q4_far_lds.vt[1], rv.vw1, rv.vs1, wi, L);
      ink_r15_q4_lsignal();
      if (b_u + 1u < nblk32) {
        ink_r15_q4_far_load_v32(&rv, vb_kh, row_step, (b_u + 1u) << 5u, pt);
      }
      ink_r15_q4_lwait();

      if (b_u + 1u < nblk32) {
        ink_r15_q4_far_pub_k32(ink_r15_q4_far_lds.ku.k, &rk, pt);
        ink_r15_q4_lsignal();
        if (b_u + 2u < nblk32) {
          ink_r15_q4_far_load_k32(&rk, kb_kh, row_step, (b_u + 2u) << 5u, pt);
        }
        ink_r15_q4_lwait();
      }
    }
    return;
  }

  const u32 tsub = w >= 3u ? 1u : 0u;
  const u32 hsub = w - 3u * tsub;
  const u32 t_sub0 = (u32)__builtin_amdgcn_readfirstlane((int)(t0 + 16u * tsub));
  if (t_sub0 >= T) {
    if ((rot_flags & 1u) != 0u) lbarrier();
    const u32 nbar = 2u * nblk32;
    for (u32 i = 0u; i < nbar; i++) {
      ink_r15_q4_lsignal();
      ink_r15_q4_lwait();
    }
    return;
  }

  const u32 h = (u32)__builtin_amdgcn_readfirstlane((int)(kh * INK_R15_Q4_GQA + hsub));
  h8 qa[8];
  float m, ls;
  if ((rot_flags & 1u) == 0u) ink_r15_q4_lsignal();
  ink_r15_q4_load_q(q, qs, t_sub0, T, h, rot_flags, ink_r15_q4_far_lds.ku.tmp[w], qa);
  {
    const u32 l_i = ink_r15_q4_lane();
    const u32 rl_i = l_i & 15u, hl_i = l_i >> 4u;
    const u32 tk_i = t_sub0 + rl_i;
    const u32 tq_i = (tk_i < T) ? tk_i : (T - 1u);
    const float sink_val = ((const G float *)((const G u8 *)sink + (ink_r15_q4_u64)tq_i * (ink_r15_q4_u64)ss))[h];
    m = sink_val * INK_R15_Q4_LOG2E;
    ls = (hl_i == 0u) ? 1.0f : 0.0f;
  }
  if ((rot_flags & 1u) != 0u) {
    lbarrier(); /* release ku.tmp -> ku.k for producer */
    ink_r15_q4_lsignal();
  }

  v8f o[8];
#pragma unroll
  for (u32 nt = 0u; nt < 8u; nt++) o[nt] = (v8f)(0.0f);
  __builtin_amdgcn_sched_barrier(0);

  ink_r15_q4_lwait();

#pragma clang loop unroll(disable)
  for (u32 b = 0u; b < nblk32; b++) {
    const u32 b_u = (u32)__builtin_amdgcn_readfirstlane((int)b);
    const int has_next = (b_u + 1u < nblk32);
    ink_r15_q4_far_blk32_step(ink_r15_q4_far_lds.ku.k, ink_r15_q4_far_lds.vt[0],
                              ink_r15_q4_far_lds.vt[1], qa, has_next, &m, &ls, o);
  }

  const float l_tot = ls + ink_r15_q4_xrow(ls);
  const u32 l_ep = ink_r15_q4_lane();
  const u32 rl_ep = l_ep & 15u, hl_ep = l_ep >> 4u;
  const u32 tk_ep = t_sub0 + rl_ep;
  if (tk_ep < T) {
    G float *pp = part + (ink_r15_q4_u64)(tk_ep * n_head + h) * INK_R15_Q4_PSTR;
    if (hl_ep == 0u) {
      pp[0] = m;
      pp[1] = l_tot;
    }
    G f32x4 *po4 = (G f32x4 *)(pp + 4u);
#pragma unroll
    for (u32 nt = 0u; nt < 8u; nt++) {
      const u32 base4 = 4u * nt + 2u * hl_ep;
      po4[base4 + 0u] = (f32x4){o[nt][0], o[nt][1], o[nt][2], o[nt][3]};
      po4[base4 + 1u] = (f32x4){o[nt][4], o[nt][5], o[nt][6], o[nt][7]};
    }
  }
}

/* ============================================================================
 * PART B: Shared Double-Buffered NB=16 Q4_0+FWHT-128 Async Engine (19.0 KiB LDS)
 *         Used by pf_split_glob_q4_r15, pf_near_glob_q4_r15, pf_glob_q4_fast_r15.
 * ========================================================================== */

typedef struct {
  union {
    f16   k[2u][INK_R15_Q4_NB16 * INK_R15_Q4_HD]; /* 8,192 B: double-buffered K */
    float tmp[6u][INK_R15_Q4_HD];                 /* 3,072 B: Q/O FWHT-128 staging (aliases k[0]) */
  } ku;
  f16 vt[2u][INK_R15_Q4_HD * INK_R15_Q4_NB16];    /* 8,192 B: double-buffered V^T */
  f16 rb[2u][48u * INK_R15_Q4_DREL];              /* 3,072 B: double-buffered near-field rel_bias */
} ink_r15_q4_nb16_lds_t;

_Static_assert(sizeof(ink_r15_q4_nb16_lds_t) == 19456u, "ink_r15_q4_nb16_lds_t must be 19,456 B (<= 21.5 KiB = 3 WG/CU)");

typedef struct {
  u32 kw[4];
  u16 ks[4];
  u16 vw[8];
  u16 vs[8];
} ink_r15_q4_raw16_t;

static inline __attribute__((always_inline)) void ink_r15_q4_load16_nowrap(
    ink_r15_q4_raw16_t *s, const G u8 *kb_kh, const G u8 *vb_kh,
    const ink_r15_q4_u64 row_step, const u32 kb, const u32 pt) {
  const u32 c = pt & 15u;
  const u32 r0 = pt >> 4u;
  const u32 blk_c = c >> 2u;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const G u8 *rec = kb_kh + (ink_r15_q4_u64)(kb + r0 + 4u * i) * row_step;
    s->kw[i] = ((const G u32 *)rec)[c];
    s->ks[i] = ((const G u16 *)(rec + 64u))[blk_c];
  }
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 blk_v = L >> 3u;
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const G u8 *rec = vb_kh + (ink_r15_q4_u64)(kb + 8u * wi + u) * row_step;
    s->vw[u] = ((const G u16 *)rec)[L];
    s->vs[u] = ((const G u16 *)(rec + 64u))[blk_v];
  }
}

static inline __attribute__((always_inline)) void ink_r15_q4_load16_ring(
    ink_r15_q4_raw16_t *s, const G u8 *kb_kh, const G u8 *vb_kh,
    const ink_r15_q4_u64 row_step, const u32 kv_len, const u32 kb, const u32 kv_b,
    const u32 kend, const u32 pt) {
  const u32 c = pt & 15u;
  const u32 r0 = pt >> 4u;
  const u32 blk_c = c >> 2u;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r = r0 + 4u * i;
    const u32 kp = kb + r;
    if (kp < kend) {
      const u32 kvp = ink_r15_q4_wrap_s(kv_b + r, kv_len);
      const G u8 *rec = kb_kh + (ink_r15_q4_u64)kvp * row_step;
      s->kw[i] = ((const G u32 *)rec)[c];
      s->ks[i] = ((const G u16 *)(rec + 64u))[blk_c];
    } else {
      s->kw[i] = 0x88888888u;
      s->ks[i] = 0u;
    }
  }
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 blk_v = L >> 3u;
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const u32 r = 8u * wi + u;
    const u32 kp = kb + r;
    if (kp < kend) {
      const u32 kvp = ink_r15_q4_wrap_s(kv_b + r, kv_len);
      const G u8 *rec = vb_kh + (ink_r15_q4_u64)kvp * row_step;
      s->vw[u] = ((const G u16 *)rec)[L];
      s->vs[u] = ((const G u16 *)(rec + 64u))[blk_v];
    } else {
      s->vw[u] = 0x8888u;
      s->vs[u] = 0u;
    }
  }
}

static inline __attribute__((always_inline)) void ink_r15_q4_pub16_kv(
    LDSP ink_r15_q4_nb16_lds_t *lds, const ink_r15_q4_raw16_t *s, const u32 bf, const u32 pt) {
  const u32 c = pt & 15u;
  const u32 r0 = pt >> 4u;
  LDSP f16 *k_dst = lds->ku.k[bf];
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r = r0 + 4u * i;
    const u32 w8n = s->kw[i];
    const u16 sb = s->ks[i];
    const ink_r15_q4_u2 h0 = ink_r15_q4_deq4n_h4(w8n, sb);
    const ink_r15_q4_u2 h1 = ink_r15_q4_deq4n_h4(w8n >> 16u, sb);
    *(LDSP u32x4 *)(k_dst + ink_r15_q4_k_off(r, c)) = (u32x4){h0.x, h0.y, h1.x, h1.y};
  }
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  u32 vx0[8], vx1[8];
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const ink_r15_q4_u2 h4 = ink_r15_q4_deq4n_h4((u32)s->vw[u], s->vs[u]);
    vx0[u] = h4.x;
    vx1[u] = h4.y;
  }
  u32x4 ev0, od0, ev1, od1;
  ink_r15_q4_t8(vx0, &ev0, &od0);
  ink_r15_q4_t8(vx1, &ev1, &od1);
  const u32 c0 = 4u * L;
  LDSP f16 *vt_dst = lds->vt[bf];
  *(LDSP u32x4 *)(vt_dst + ink_r15_q4_vt_off(c0 + 0u, wi)) = ev0;
  *(LDSP u32x4 *)(vt_dst + ink_r15_q4_vt_off(c0 + 1u, wi)) = od0;
  *(LDSP u32x4 *)(vt_dst + ink_r15_q4_vt_off(c0 + 2u, wi)) = ev1;
  *(LDSP u32x4 *)(vt_dst + ink_r15_q4_vt_off(c0 + 3u, wi)) = od1;
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r15_q4_pub16_rb(
    LDSP ink_r15_q4_nb16_lds_t *lds, const G float *rel_bias, const u32 qpos_min,
    const u32 extent_lim, const u32 kb, const u32 bf, const u32 pt) {
  const int d_min_s = (int)qpos_min - (int)(kb + (INK_R15_Q4_NB16 - 1u));
  const u32 d_base_wg = (d_min_s > 0) ? (u32)d_min_s : 0u;
  if (d_base_wg < extent_lim && pt < 48u) {
    const u32 d = d_base_wg + pt;
    h8 r0 = (h8)(0.0f), r1 = (h8)(0.0f);
    if (d < extent_lim) {
      const G f32x4 *rp = (const G f32x4 *)(rel_bias + (ink_r15_q4_u64)d * 16ull);
      const f32x4 v0 = rp[0], v1 = rp[1], v2 = rp[2], v3 = rp[3];
      r0 = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
      r1 = (h8){(f16)v2.x, (f16)v2.y, (f16)v2.z, (f16)v2.w, (f16)v3.x, (f16)v3.y, (f16)v3.z, (f16)v3.w};
    }
    *(LDSP h8 *)(lds->rb[bf] + pt * 16u + 0u) = r0;
    *(LDSP h8 *)(lds->rb[bf] + pt * 16u + 8u) = r1;
  }
}

/* D=2 Async Double-Buffered Producer for block range [blk_beg, blk_end).
 * Emits: optional Sync_Pre (if wait_pre != 0), then EXACTLY (blk_end - blk_beg)
 * split barriers, then optional Sync_Post (if wait_post != 0). */
static inline __attribute__((always_inline)) void ink_r15_q4_producer16(
    LDSP ink_r15_q4_nb16_lds_t *lds, const G u8 *kb_kh, const G u8 *vb_kh,
    const G float *rel_bias, const ink_r15_q4_u64 row_step, const u32 kv_len,
    const u32 k_start, const u32 kend, const u32 qpos_min, const u32 extent_lim,
    const u32 blk_beg, const u32 blk_end, const u32 blk_far_end,
    const int no_wrap, const int wait_pre, const int wait_post, const u32 pt) {
  const u32 nblk_loc = blk_end - blk_beg;
  if (nblk_loc == 0u) return;

  const u32 kb0_start = k_start + INK_R15_Q4_NB16 * blk_beg;
  const u32 nblk_full = (kend > kb0_start) ? ((kend - kb0_start) >> 4u) : 0u;
  const u32 fast_lim = no_wrap ? ((nblk_full < nblk_loc) ? nblk_full : nblk_loc) : 0u;
  const u32 far_loc = (blk_far_end > blk_beg)
                          ? (((blk_far_end - blk_beg) < fast_lim) ? (blk_far_end - blk_beg) : fast_lim)
                          : 0u;

  u32 kv_load = no_wrap ? 0u : (kb0_start % kv_len);
  ink_r15_q4_raw16_t ring[2];

  /* Prime ring[0] (and ring[1]) BEFORE Sync_Pre so DRAM latency hides behind Q FWHT-128! */
  if (0u < fast_lim) {
    ink_r15_q4_load16_nowrap(&ring[0], kb_kh, vb_kh, row_step, kb0_start, pt);
  } else {
    ink_r15_q4_load16_ring(&ring[0], kb_kh, vb_kh, row_step, kv_len, kb0_start,
                           no_wrap ? kb0_start : kv_load, kend, pt);
  }
  if (!no_wrap) kv_load = ink_r15_q4_kv_adv16(kv_load, kv_len);

  if (1u < nblk_loc) {
    const u32 kb1 = kb0_start + 16u;
    if (1u < fast_lim) {
      ink_r15_q4_load16_nowrap(&ring[1], kb_kh, vb_kh, row_step, kb1, pt);
    } else {
      ink_r15_q4_load16_ring(&ring[1], kb_kh, vb_kh, row_step, kv_len, kb1,
                             no_wrap ? kb1 : kv_load, kend, pt);
    }
    if (!no_wrap) kv_load = ink_r15_q4_kv_adv16(kv_load, kv_len);
  }

  if (wait_pre) lbarrier();
  __builtin_amdgcn_sched_barrier(0);

  /* Publish block 0 into bf=0, signal Sync 0, prefetch block 2, wait Sync 0. */
  ink_r15_q4_pub16_kv(lds, &ring[0], 0u, pt);
  if (blk_beg >= blk_far_end) {
    ink_r15_q4_pub16_rb(lds, rel_bias, qpos_min, extent_lim, kb0_start, 0u, pt);
  }
  ink_r15_q4_lsignal();
  if (2u < nblk_loc) {
    const u32 kb2 = kb0_start + 32u;
    if (2u < fast_lim) {
      ink_r15_q4_load16_nowrap(&ring[0], kb_kh, vb_kh, row_step, kb2, pt);
    } else {
      ink_r15_q4_load16_ring(&ring[0], kb_kh, vb_kh, row_step, kv_len, kb2,
                             no_wrap ? kb2 : kv_load, kend, pt);
    }
    if (!no_wrap) kv_load = ink_r15_q4_kv_adv16(kv_load, kv_len);
  }
  ink_r15_q4_lwait();

  u32 k = 0u;
  /* Pure far-field no-wrap 2x unrolled hot loop (0 modulo, 0 bounds, 0 rb!). */
  if (no_wrap && far_loc >= 4u) {
#pragma clang loop unroll(disable)
    for (; k + 4u < far_loc; k += 2u) {
      ink_r15_q4_pub16_kv(lds, &ring[1], 1u, pt);
      ink_r15_q4_lsignal();
      ink_r15_q4_load16_nowrap(&ring[1], kb_kh, vb_kh, row_step, kb0_start + 16u * (k + 3u), pt);
      ink_r15_q4_lwait();

      ink_r15_q4_pub16_kv(lds, &ring[0], 0u, pt);
      ink_r15_q4_lsignal();
      ink_r15_q4_load16_nowrap(&ring[0], kb_kh, vb_kh, row_step, kb0_start + 16u * (k + 4u), pt);
      ink_r15_q4_lwait();
    }
  }

  /* Unified tail + near-field + ring loop (2x unrolled so ring[0]/ring[1] stay
   * compile-time constant -> 100% VGPR-promoted, 0 private stack / 0 spills!). */
#pragma clang loop unroll(disable)
  for (; k + 1u < nblk_loc; k += 2u) {
    {
      const u32 kn = k + 1u;
      ink_r15_q4_pub16_kv(lds, &ring[1], 1u, pt);
      if (blk_beg + kn >= blk_far_end) {
        ink_r15_q4_pub16_rb(lds, rel_bias, qpos_min, extent_lim, kb0_start + 16u * kn, 1u, pt);
      }
      ink_r15_q4_lsignal();
      const u32 kl = kn + 2u;
      if (kl < nblk_loc) {
        const u32 kbl = kb0_start + 16u * kl;
        if (kl < fast_lim) {
          ink_r15_q4_load16_nowrap(&ring[1], kb_kh, vb_kh, row_step, kbl, pt);
        } else {
          ink_r15_q4_load16_ring(&ring[1], kb_kh, vb_kh, row_step, kv_len, kbl,
                                 no_wrap ? kbl : kv_load, kend, pt);
        }
        if (!no_wrap) kv_load = ink_r15_q4_kv_adv16(kv_load, kv_len);
      }
      ink_r15_q4_lwait();
    }
    if (k + 2u < nblk_loc) {
      const u32 kn = k + 2u;
      ink_r15_q4_pub16_kv(lds, &ring[0], 0u, pt);
      if (blk_beg + kn >= blk_far_end) {
        ink_r15_q4_pub16_rb(lds, rel_bias, qpos_min, extent_lim, kb0_start + 16u * kn, 0u, pt);
      }
      ink_r15_q4_lsignal();
      const u32 kl = kn + 2u;
      if (kl < nblk_loc) {
        const u32 kbl = kb0_start + 16u * kl;
        if (kl < fast_lim) {
          ink_r15_q4_load16_nowrap(&ring[0], kb_kh, vb_kh, row_step, kbl, pt);
        } else {
          ink_r15_q4_load16_ring(&ring[0], kb_kh, vb_kh, row_step, kv_len, kbl,
                                 no_wrap ? kbl : kv_load, kend, pt);
        }
        if (!no_wrap) kv_load = ink_r15_q4_kv_adv16(kv_load, kv_len);
      }
      ink_r15_q4_lwait();
    }
  }

  if (wait_post) lbarrier();
}

static inline __attribute__((always_inline)) void ink_r15_q4_softmax_pv16_sb(
    const LDSP f16 *vt_lds, const v8f sa, const int has_next, const int use_1d,
    float *m_p, float *ls_p, v8f o[8]) {
  float bm = sa[0];
#pragma unroll
  for (u32 v = 1u; v < 8u; v++) bm = __builtin_fmaxf(bm, sa[v]);
  bm = __builtin_fmaxf(bm, ink_r15_q4_xrow(bm));

  const float m = *m_p;
  const int active = (__builtin_amdgcn_ballot_w32(bm > -1e30f && !(bm - m < -30.0f)) != 0u);
  if (active) {
    const int need_rescale = (__builtin_amdgcn_ballot_w32(bm > m || m <= -1e30f) != 0u);
    float mn = m;
    float c = 1.0f;
    if (need_rescale) {
      mn = __builtin_fmaxf(m, bm);
      c = (m > -1e30f) ? __builtin_amdgcn_exp2f(m - mn) : 0.0f;
    }

    h8 pb;
    float ps = 0.0f;
    if (mn > -1e30f) {
      f16 pa[8];
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const float pv = __builtin_amdgcn_exp2f(sa[v] - mn);
        pa[v] = (f16)pv;
        ps += pv;
      }
      pb = (h8){pa[0], pa[1], pa[2], pa[3], pa[4], pa[5], pa[6], pa[7]};
    } else {
      pb = (h8)(0.0f);
    }
    *ls_p = (*ls_p) * c + ps;
    *m_p = mn;

    if (need_rescale) {
#pragma unroll
      for (u32 nt = 0u; nt < 8u; nt++)
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c;
    }

    if (use_1d) {
      ink_r15_q4_pv16_1d(vt_lds, pb, o, has_next);
    } else {
      ink_r15_q4_pv16_2d(vt_lds, pb, o, has_next);
    }
  }
  if (has_next) {
    if (!active) ink_r15_q4_lsignal();
    ink_r15_q4_lwait();
  }
}

/* Zero-LDS, zero-loop-carried-VGPR near-field mask + exact L1-cached R * rel_bias
 * (R is only 6 KiB/WG -> 100% L1 hit, <= 4 temp VGPRs -> true 3 WGs/CU with 0 spills!). */
static inline __attribute__((always_inline)) float ink_r15_q4_nf_adj(
    float sv, const LDSP f16 *rb_lds, const G float *R, const u32 rs,
    const u32 t_sub0, const u32 T, const u32 h, const u32 kb, const u32 v_idx,
    const u32 qpos_sub, const u32 win, const u32 extent_lim, const u32 d_base_wg) {
  const u32 l_loc = ink_r15_q4_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 tp = qpos_sub + rl;
  const u32 kp = kb + (hl << 3u) + v_idx;
  if (kp > tp || (win > 0u && tp - kp >= win)) return -__builtin_inff();
  if (extent_lim > 0u) {
    const u32 dist = tp - kp;
    if (dist < extent_lim) {
      const u32 tq = (t_sub0 + rl < T) ? (t_sub0 + rl) : (T - 1u);
      const G float *rp = (const G float *)((const G u8 *)R + (ink_r15_q4_u64)tq * (ink_r15_q4_u64)rs) +
                          (ink_r15_q4_u64)h * 16ull;
      const LDSP f16 *bp = rb_lds + (dist - d_base_wg) * 16u;
      float rbs = 0.0f;
#pragma clang loop unroll(disable)
      for (u32 d = 0u; d < 16u; d += 4u) {
        const f32x4 rv = *(const G f32x4 *)(rp + d);
        const ink_r15_q4_h4 bv = *(const LDSP ink_r15_q4_h4 *)(bp + d);
        rbs = __builtin_fmaf((float)(f16)(rv.x * INK_R15_Q4_LOG2E), (float)bv.x, rbs);
        rbs = __builtin_fmaf((float)(f16)(rv.y * INK_R15_Q4_LOG2E), (float)bv.y, rbs);
        rbs = __builtin_fmaf((float)(f16)(rv.z * INK_R15_Q4_LOG2E), (float)bv.z, rbs);
        rbs = __builtin_fmaf((float)(f16)(rv.w * INK_R15_Q4_LOG2E), (float)bv.w, rbs);
      }
      sv += rbs;
    }
  }
  return sv;
}

/* Shared two-phase double-buffered consumer loop (Sync 0 ALREADY completed before entry!). */
static inline __attribute__((always_inline)) void ink_r15_q4_consumer16(
    const LDSP ink_r15_q4_nb16_lds_t *lds, const G float *R, u32 rs,
    const u32 t_sub0, const u32 T, const u32 h,
    const u32 k_start, const u32 qpos_min, const u32 sub_qpos_min,
    const u32 win, const u32 extent_lim,
    const u32 blk_beg, const u32 blk_end, const u32 blk_far_end,
    const h8 qa[8], float *m_p, float *ls_p, v8f o[8]) {
  const u32 nblk_loc = (u32)__builtin_amdgcn_readfirstlane((int)(blk_end - blk_beg));
  const u32 nfar = (u32)__builtin_amdgcn_readfirstlane((int)(
      (blk_far_end > blk_beg) ? (((blk_far_end < blk_end) ? blk_far_end : blk_end) - blk_beg) : 0u));

  /* Phase A: Pure Far-Field (1-deep QK/PV keeps <= 120 VGPR, 0/0 spills). */
#pragma clang loop unroll(disable)
  for (u32 k = 0u; k < nfar; k++) {
    const u32 k_u = (u32)__builtin_amdgcn_readfirstlane((int)k);
    const u32 bf = k_u & 1u;
    const v8f sa = ink_r15_q4_qk16_1d(lds->ku.k[bf], qa);
    ink_r15_q4_softmax_pv16_sb(lds->vt[bf], sa, (k_u + 1u < nblk_loc), 1, m_p, ls_p, o);
  }

  /* Phase B: Near-Field (<= 66 blocks: 1-deep QK/PV + exact L1-cached R*rb). */
  if (nfar < nblk_loc) {
#pragma clang loop unroll(disable)
    for (u32 k = nfar; k < nblk_loc; k++) {
      const u32 k_u = (u32)__builtin_amdgcn_readfirstlane((int)k);
      const u32 kb = (u32)__builtin_amdgcn_readfirstlane((int)(k_start + INK_R15_Q4_NB16 * (blk_beg + k_u)));
      const u32 bf = k_u & 1u;
      const v8f sa_v = ink_r15_q4_qk16_1d(lds->ku.k[bf], qa);

      const int d_min_wg_s = (int)qpos_min - (int)(kb + (INK_R15_Q4_NB16 - 1u));
      const u32 d_base_wg = (u32)__builtin_amdgcn_readfirstlane((d_min_wg_s > 0) ? d_min_wg_s : 0);
      const LDSP f16 *rb_lds = lds->rb[bf];

      float s0 = sa_v[0], s1 = sa_v[1], s2 = sa_v[2], s3 = sa_v[3];
      float s4 = sa_v[4], s5 = sa_v[5], s6 = sa_v[6], s7 = sa_v[7];
      s0 = ink_r15_q4_nf_adj(s0, rb_lds, R, rs, t_sub0, T, h, kb, 0u, sub_qpos_min, win, extent_lim, d_base_wg); __builtin_amdgcn_sched_barrier(0);
      s1 = ink_r15_q4_nf_adj(s1, rb_lds, R, rs, t_sub0, T, h, kb, 1u, sub_qpos_min, win, extent_lim, d_base_wg); __builtin_amdgcn_sched_barrier(0);
      s2 = ink_r15_q4_nf_adj(s2, rb_lds, R, rs, t_sub0, T, h, kb, 2u, sub_qpos_min, win, extent_lim, d_base_wg); __builtin_amdgcn_sched_barrier(0);
      s3 = ink_r15_q4_nf_adj(s3, rb_lds, R, rs, t_sub0, T, h, kb, 3u, sub_qpos_min, win, extent_lim, d_base_wg); __builtin_amdgcn_sched_barrier(0);
      s4 = ink_r15_q4_nf_adj(s4, rb_lds, R, rs, t_sub0, T, h, kb, 4u, sub_qpos_min, win, extent_lim, d_base_wg); __builtin_amdgcn_sched_barrier(0);
      s5 = ink_r15_q4_nf_adj(s5, rb_lds, R, rs, t_sub0, T, h, kb, 5u, sub_qpos_min, win, extent_lim, d_base_wg); __builtin_amdgcn_sched_barrier(0);
      s6 = ink_r15_q4_nf_adj(s6, rb_lds, R, rs, t_sub0, T, h, kb, 6u, sub_qpos_min, win, extent_lim, d_base_wg); __builtin_amdgcn_sched_barrier(0);
      s7 = ink_r15_q4_nf_adj(s7, rb_lds, R, rs, t_sub0, T, h, kb, 7u, sub_qpos_min, win, extent_lim, d_base_wg); __builtin_amdgcn_sched_barrier(0);

      const v8f sa_adj = (v8f){s0, s1, s2, s3, s4, s5, s6, s7};
      ink_r15_q4_softmax_pv16_sb(lds->vt[bf], sa_adj, (k_u + 1u < nblk_loc), 1, m_p, ls_p, o);
    }
  }
}

/* ============================================================================
 * Kernel 1: ie_inkling_rel_attn_pf_split_glob_q4_r15
 *   Split-KV Incremental Global Q4_0+FWHT-128 Prefill (T=17..256, nsplit=2..16).
 *   Grid:    (((T + 31u) / 32u) * nsplit, n_kv, 1u), NT = 256.
 *   Buffer:  part = b->at_part (stride 130 floats, 0 B extra VRAM:
 *            T=256 nsplit=4 -> 256 WGs; T=128 nsplit=8 -> 256 WGs!).
 *   Counter: count[tile * n_kv + kh] (self-reset to 0 by last arriving WG).
 * ========================================================================== */
__attribute__((amdgpu_waves_per_eu(12, 16)))
KERNEL ie_inkling_rel_attn_pf_split_glob_q4_r15(
    const G float *q, const G u8 *kc_q4, const G u8 *vc_q4,
    const G float *R, const G float *rel_bias, const G float *sink,
    G float *part, G float *out,
    u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
    u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
    u32 qs, u32 rs, u32 ss, u32 os, G u32 *count, u32 rot_flags) {
  static LDS ink_r15_q4_nb16_lds_t ink_r15_q4_sp_lds __attribute__((aligned(16)));
  static LDS u32 ink_r15_q4_sp_last;

  if (hd != INK_R15_Q4_HD || d_rel != INK_R15_Q4_DREL || n_kv == 0u ||
      n_head != n_kv * INK_R15_Q4_GQA || T == 0u || kv_len == 0u ||
      nsplit < 2u || nsplit > 16u || !q || !kc_q4 || !vc_q4 || !sink || !part || !out || !count)
    return;

  const u32 ntile = (T + 31u) >> 5u;
  const u32 wg = wgid();
  const u32 tile_raw = wg / nsplit;
  const u32 s = wg - tile_raw * nsplit;
  if (tile_raw >= ntile) return;
  const u32 tile = (ntile - 1u) - tile_raw;
  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  if (kh >= n_kv) return;

  const u32 t0 = tile << 5u;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 tlast = (t0 + 31u < T) ? (t0 + 31u) : (T - 1u);
  const u32 nq_tile = tlast - t0 + 1u;
  const u32 qpos_min = pos0 + t0;
  const u32 qpos_max = pos0 + tlast;
  const u32 k_start = (win > 0u && qpos_min + 1u > win) ? (qpos_min + 1u - win) : 0u;
  const u32 kend = qpos_max + 1u;
  const u32 nblk = (kend > k_start) ? ((kend - k_start + INK_R15_Q4_NB16 - 1u) >> 4u) : 0u;
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const u32 blk_far_end = ink_r15_q4_calc_far_end16(win, qpos_min, extent_lim, k_start, nblk);

  const u32 blk_per_s = (nblk + nsplit - 1u) / nsplit;
  const u32 blk_beg = (s * blk_per_s < nblk) ? (s * blk_per_s) : nblk;
  const u32 blk_end = (blk_beg + blk_per_s < nblk) ? (blk_beg + blk_per_s) : nblk;
  const u32 nblk_loc = blk_end - blk_beg;

  if (nblk_loc == 0u) {
    const u32 h0 = kh * INK_R15_Q4_GQA;
    const u32 l_z = lane();
    for (u32 qi = w; qi < nq_tile * INK_R15_Q4_GQA; qi += 8u) {
      const u32 tk = t0 + qi / INK_R15_Q4_GQA;
      const u32 h = h0 + qi % INK_R15_Q4_GQA;
      G float *pp = part + (ink_r15_q4_u64)((tk * n_head + h) * nsplit + s) * INK_R15_Q4_PSTR;
      if (l_z == 0u) {
        pp[0] = -1e30f;
        pp[1] = 0.0f;
      }
      ((G f32x4 *)(pp + 4u))[l_z] = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    }
  } else if (w >= 6u) {
    const ink_r15_q4_u64 row_step = (ink_r15_q4_u64)n_kv * INK_R15_Q4_RECB;
    const G u8 *kb_kh = kc_q4 + (ink_r15_q4_u64)kh * INK_R15_Q4_RECB;
    const G u8 *vb_kh = vc_q4 + (ink_r15_q4_u64)kh * INK_R15_Q4_RECB;
    const int no_wrap = (kend <= kv_len);
    ink_r15_q4_producer16(&ink_r15_q4_sp_lds, kb_kh, vb_kh, rel_bias, row_step, kv_len,
                          k_start, kend, qpos_min, extent_lim, blk_beg, blk_end, blk_far_end,
                          no_wrap, (rot_flags & 1u) != 0u, 0, tid() - 192u);
  } else {
    const u32 tsub = w >= 3u ? 1u : 0u;
    const u32 hsub = w - 3u * tsub;
    const u32 t_sub0 = (u32)__builtin_amdgcn_readfirstlane((int)(t0 + 16u * tsub));
    if (t_sub0 >= T) {
      if ((rot_flags & 1u) != 0u) lbarrier();
      for (u32 k = 0u; k < nblk_loc; k++) {
        ink_r15_q4_lsignal();
        ink_r15_q4_lwait();
      }
    } else {
      const u32 h = (u32)__builtin_amdgcn_readfirstlane((int)(kh * INK_R15_Q4_GQA + hsub));
      const u32 sub_qpos_min = (u32)__builtin_amdgcn_readfirstlane((int)(pos0 + t_sub0));

      h8 qa[8];
      if ((rot_flags & 1u) == 0u) ink_r15_q4_lsignal();
      ink_r15_q4_load_q(q, qs, t_sub0, T, h, rot_flags, ink_r15_q4_sp_lds.ku.tmp[w], qa);
      if ((rot_flags & 1u) != 0u) {
        lbarrier();
        ink_r15_q4_lsignal();
      }

      /* Split-KV omits sink here; dynamic per-token sink is added once in the final merge. */
      float m = -__builtin_inff();
      float ls = 0.0f;

      ink_r15_q4_lwait();
      v8f o[8];
#pragma unroll
      for (u32 nt = 0u; nt < 8u; nt++) {
        v8f z = (v8f)(0.0f);
        __asm__ volatile("" : "+v"(z));
        o[nt] = z;
      }
      __builtin_amdgcn_sched_barrier(0);

      ink_r15_q4_consumer16(&ink_r15_q4_sp_lds, R, rs, t_sub0, T, h,
                            k_start, qpos_min, sub_qpos_min,
                            win, extent_lim, blk_beg, blk_end, blk_far_end, qa, &m, &ls, o);

      const float l_tot = ls + ink_r15_q4_xrow(ls);
      const u32 l_ep = ink_r15_q4_lane();
      const u32 rl_ep = l_ep & 15u, hl_ep = l_ep >> 4u;
      const u32 tk_ep = t_sub0 + rl_ep;
      if (tk_ep < T) {
        G float *pp = part + (ink_r15_q4_u64)((tk_ep * n_head + h) * nsplit + s) * INK_R15_Q4_PSTR;
        if (hl_ep == 0u) {
          pp[0] = (l_tot > 0.0f) ? (m * INK_R15_Q4_LN2) : -1e30f;
          pp[1] = l_tot;
        }
        G f32x4 *po4 = (G f32x4 *)(pp + 4u);
#pragma unroll
        for (u32 nt = 0u; nt < 8u; nt++) {
          const u32 base4 = 4u * nt + 2u * hl_ep;
          po4[base4 + 0u] = (f32x4){o[nt][0], o[nt][1], o[nt][2], o[nt][3]};
          po4[base4 + 1u] = (f32x4){o[nt][4], o[nt][5], o[nt][6], o[nt][7]};
        }
      }
    }
  }

  /* Self-resetting split-K arrival & last-WG parallel merge + in-register inverse FWHT-128. */
  const u32 cidx = (u32)__builtin_amdgcn_readfirstlane((int)(tile * n_kv + kh));
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r15_q4_sp_last =
        (__scoped_atomic_fetch_add(&count[cidx], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r15_q4_sp_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const u32 h0 = kh * INK_R15_Q4_GQA;
  const u32 total_qh = nq_tile * INK_R15_Q4_GQA;
  const u32 l_m = lane();
  for (u32 qi = w; qi < total_qh; qi += 8u) {
    const u32 tk = t0 + qi / INK_R15_Q4_GQA;
    const u32 h = h0 + qi % INK_R15_Q4_GQA;
    const float sv = ((const G float *)((const G u8 *)sink + (ink_r15_q4_u64)tk * (ink_r15_q4_u64)ss))[h];
    const G float *ph = part + (ink_r15_q4_u64)((tk * n_head + h) * nsplit) * INK_R15_Q4_PSTR;

    float M2 = sv;
    for (u32 sp = 0u; sp < nsplit; sp++) {
      const ink_r15_q4_u64 off = (ink_r15_q4_u64)sp * INK_R15_Q4_PSTR;
      if (ph[off + 1u] > 0.0f) M2 = __builtin_fmaxf(M2, ph[off]);
    }

    float L2 = __builtin_expf(sv - M2);
    f32x4 acc = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 sp = 0u; sp < nsplit; sp++) {
      const ink_r15_q4_u64 off = (ink_r15_q4_u64)sp * INK_R15_Q4_PSTR;
      const float lp = ph[off + 1u];
      if (lp > 0.0f) {
        const float wt = __builtin_expf(ph[off] - M2);
        L2 += lp * wt;
        const f32x4 ov = ((const G f32x4 *)(ph + off + 4u))[l_m];
        acc += ov * wt;
      }
    }
    f32x4 res = acc * (1.0f / L2);
    if ((rot_flags & 2u) != 0u) res = ink_r15_q4_fwht128_wave4(res, l_m);
    G float *ot = (G float *)((G u8 *)out + (ink_r15_q4_u64)tk * (ink_r15_q4_u64)os) + (ink_r15_q4_u64)h * 128ull;
    ((G f32x4 *)ot)[l_m] = res;
  }

  barrier();
  if (tid() == 0u) count[cidx] = 0u;
}

/* ============================================================================
 * Kernel 2B: ie_inkling_rel_attn_pf_near_glob_q4_r15
 *   Near-Field Merge Companion consuming (M_far, L_far, O_far) from far_glob_q4_r15,
 *   running ONLY the <= 66 near-field blocks (exact rel_bias + causal mask +
 *   double-buffered split barriers) and applying inverse FWHT-128 before out.
 *   Grid: ((T + 31u) / 32u, n_kv, 1u), NT = 256.
 * ========================================================================== */
__attribute__((amdgpu_waves_per_eu(12, 16)))
KERNEL ie_inkling_rel_attn_pf_near_glob_q4_r15(
    const G float *q, const G u8 *kc_q4, const G u8 *vc_q4,
    const G float *R, const G float *rel_bias, const G float *sink,
    const G float *part, G float *out,
    u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
    u32 d_rel, u32 rel_extent, u32 win,
    u32 qs, u32 rs, u32 ss, u32 os, u32 rot_flags) {
  static LDS ink_r15_q4_nb16_lds_t ink_r15_q4_nf_lds __attribute__((aligned(16)));
  if (hd != INK_R15_Q4_HD || d_rel != INK_R15_Q4_DREL || n_kv == 0u ||
      n_head != n_kv * INK_R15_Q4_GQA || T == 0u || kv_len == 0u ||
      !q || !kc_q4 || !vc_q4 || !sink || !out)
    return;

  const u32 ntile = (T + 31u) >> 5u;
  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  if (wgid() >= ntile || kh >= n_kv) return;

  const u32 tile = (ntile == 16u && n_kv == 8u && win == 0u && kh < 4u) ? wgid() : ((ntile - 1u) - wgid());
  const u32 t0 = tile << 5u;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));

  const u32 tlast = (t0 + 31u < T) ? (t0 + 31u) : (T - 1u);
  const u32 qpos_min = pos0 + t0;
  const u32 qpos_max = pos0 + tlast;
  const u32 nblk32_far = (win == 0u && part) ? ink_r15_q4_far_nblk32(qpos_min, rel_extent, kv_len) : 0u;
  const u32 k_start = (nblk32_far > 0u)
                          ? (nblk32_far << 5u)
                          : ((win > 0u && qpos_min + 1u > win) ? (qpos_min + 1u - win) : 0u);
  const u32 kend = qpos_max + 1u;
  const u32 nblk = (kend > k_start) ? ((kend - k_start + INK_R15_Q4_NB16 - 1u) >> 4u) : 0u;
  if (nblk == 0u) return;
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const u32 blk_far_end = ink_r15_q4_calc_far_end16(win, qpos_min, extent_lim, k_start, nblk);

  if (w >= 6u) {
    const ink_r15_q4_u64 row_step = (ink_r15_q4_u64)n_kv * INK_R15_Q4_RECB;
    const G u8 *kb_kh = kc_q4 + (ink_r15_q4_u64)kh * INK_R15_Q4_RECB;
    const G u8 *vb_kh = vc_q4 + (ink_r15_q4_u64)kh * INK_R15_Q4_RECB;
    const int no_wrap = (kend <= kv_len);
    ink_r15_q4_producer16(&ink_r15_q4_nf_lds, kb_kh, vb_kh, rel_bias, row_step, kv_len,
                          k_start, kend, qpos_min, extent_lim, 0u, nblk, blk_far_end,
                          no_wrap, (rot_flags & 1u) != 0u, (rot_flags & 2u) != 0u, tid() - 192u);
    return;
  }

  const u32 tsub = w >= 3u ? 1u : 0u;
  const u32 hsub = w - 3u * tsub;
  const u32 t_sub0 = (u32)__builtin_amdgcn_readfirstlane((int)(t0 + 16u * tsub));
  if (t_sub0 >= T) {
    if ((rot_flags & 1u) != 0u) lbarrier();
    for (u32 k = 0u; k < nblk; k++) {
      ink_r15_q4_lsignal();
      ink_r15_q4_lwait();
    }
    if ((rot_flags & 2u) != 0u) lbarrier();
    return;
  }

  const u32 h = (u32)__builtin_amdgcn_readfirstlane((int)(kh * INK_R15_Q4_GQA + hsub));
  const u32 sub_qpos_min = (u32)__builtin_amdgcn_readfirstlane((int)(pos0 + t_sub0));

  h8 qa[8];
  if ((rot_flags & 1u) == 0u) ink_r15_q4_lsignal();
  ink_r15_q4_load_q(q, qs, t_sub0, T, h, rot_flags, ink_r15_q4_nf_lds.ku.tmp[w], qa);
  if ((rot_flags & 1u) != 0u) {
    lbarrier();
    ink_r15_q4_lsignal();
  }

  float m, ls;
  v8f o[8];
  {
    const u32 l_i = ink_r15_q4_lane();
    const u32 rl_i = l_i & 15u, hl_i = l_i >> 4u;
    const u32 tk_i = t_sub0 + rl_i;
    const u32 tq_i = (tk_i < T) ? tk_i : (T - 1u);
    if (nblk32_far > 0u) {
      const G float *pp = part + (ink_r15_q4_u64)(tq_i * n_head + h) * INK_R15_Q4_PSTR;
      m = pp[0];
      ls = (hl_i == 0u) ? pp[1] : 0.0f;
      const G f32x4 *po4 = (const G f32x4 *)(pp + 4u);
#pragma unroll
      for (u32 nt = 0u; nt < 8u; nt++) {
        const u32 base4 = 4u * nt + 2u * hl_i;
        const f32x4 a0 = po4[base4 + 0u];
        const f32x4 a1 = po4[base4 + 1u];
        o[nt] = (v8f){a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        __builtin_amdgcn_sched_barrier(0);
      }
    } else {
      const float sink_val = ((const G float *)((const G u8 *)sink + (ink_r15_q4_u64)tq_i * (ink_r15_q4_u64)ss))[h];
      m = sink_val * INK_R15_Q4_LOG2E;
      ls = (hl_i == 0u) ? 1.0f : 0.0f;
#pragma unroll
      for (u32 nt = 0u; nt < 8u; nt++) o[nt] = (v8f)(0.0f);
    }
  }
  __builtin_amdgcn_sched_barrier(0);

  ink_r15_q4_lwait();
  ink_r15_q4_consumer16(&ink_r15_q4_nf_lds, R, rs, t_sub0, T, h,
                        k_start, qpos_min, sub_qpos_min,
                        win, extent_lim, 0u, nblk, blk_far_end, qa, &m, &ls, o);
  if ((rot_flags & 2u) != 0u) lbarrier(); /* ensure all consumers done with ku.k[0] before ku.tmp */

  ink_r15_q4_store_o(out, os, t_sub0, T, h, rot_flags, ink_r15_q4_nf_lds.ku.tmp[w], ls, o);
}

/* ============================================================================
 * Kernel 3: ie_inkling_rel_attn_pf_glob_q4_fast_r15
 *   Drop-in single-launch upgrade of ie_inkling_rel_attn_pf_glob_q4_r14:
 *   double-buffered D=2 async Q4 producer + split barriers + 2-deep QK/PV +
 *   wave-level conditional rescale + zero-tile PV skip + zero-rb far-field!
 *   Grid: ((T + 31u) / 32u, n_kv, 1u), NT = 256.
 * ========================================================================== */
__attribute__((amdgpu_waves_per_eu(12, 16)))
KERNEL ie_inkling_rel_attn_pf_glob_q4_fast_r15(
    const G float *q, const G u8 *kc_q4, const G u8 *vc_q4,
    const G float *R, const G float *rel_bias,
    const G float *sink, G float *out,
    u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
    u32 d_rel, u32 rel_extent, u32 win,
    u32 qs, u32 rs, u32 ss, u32 os, u32 rot_flags) {
  static LDS ink_r15_q4_nb16_lds_t ink_r15_q4_fast_lds __attribute__((aligned(16)));
  if (hd != INK_R15_Q4_HD || d_rel != INK_R15_Q4_DREL || n_kv == 0u ||
      n_head != n_kv * INK_R15_Q4_GQA || T == 0u || kv_len == 0u ||
      !q || !kc_q4 || !vc_q4 || !sink || !out)
    return;

  const u32 ntile = (T + 31u) >> 5u;
  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  if (wgid() >= ntile || kh >= n_kv) return;

  const u32 tile = (ntile == 16u && n_kv == 8u && win == 0u && kh < 4u) ? wgid() : ((ntile - 1u) - wgid());
  const u32 t0 = tile << 5u;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));

  const u32 tlast = (t0 + 31u < T) ? (t0 + 31u) : (T - 1u);
  const u32 qpos_min = pos0 + t0;
  const u32 qpos_max = pos0 + tlast;
  const u32 k_start = (win > 0u && qpos_min + 1u > win) ? (qpos_min + 1u - win) : 0u;
  const u32 kend = qpos_max + 1u;
  const u32 nblk = (kend > k_start) ? ((kend - k_start + INK_R15_Q4_NB16 - 1u) >> 4u) : 0u;
  if (nblk == 0u) return;
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const u32 blk_far_end = ink_r15_q4_calc_far_end16(win, qpos_min, extent_lim, k_start, nblk);

  if (w >= 6u) {
    const ink_r15_q4_u64 row_step = (ink_r15_q4_u64)n_kv * INK_R15_Q4_RECB;
    const G u8 *kb_kh = kc_q4 + (ink_r15_q4_u64)kh * INK_R15_Q4_RECB;
    const G u8 *vb_kh = vc_q4 + (ink_r15_q4_u64)kh * INK_R15_Q4_RECB;
    const int no_wrap = (kend <= kv_len);
    ink_r15_q4_producer16(&ink_r15_q4_fast_lds, kb_kh, vb_kh, rel_bias, row_step, kv_len,
                          k_start, kend, qpos_min, extent_lim, 0u, nblk, blk_far_end,
                          no_wrap, (rot_flags & 1u) != 0u, (rot_flags & 2u) != 0u, tid() - 192u);
    return;
  }

  const u32 tsub = w >= 3u ? 1u : 0u;
  const u32 hsub = w - 3u * tsub;
  const u32 t_sub0 = (u32)__builtin_amdgcn_readfirstlane((int)(t0 + 16u * tsub));
  if (t_sub0 >= T) {
    if ((rot_flags & 1u) != 0u) lbarrier();
    for (u32 k = 0u; k < nblk; k++) {
      ink_r15_q4_lsignal();
      ink_r15_q4_lwait();
    }
    if ((rot_flags & 2u) != 0u) lbarrier();
    return;
  }

  const u32 h = (u32)__builtin_amdgcn_readfirstlane((int)(kh * INK_R15_Q4_GQA + hsub));
  const u32 sub_qpos_min = (u32)__builtin_amdgcn_readfirstlane((int)(pos0 + t_sub0));

  h8 qa[8];
  if ((rot_flags & 1u) == 0u) ink_r15_q4_lsignal();
  ink_r15_q4_load_q(q, qs, t_sub0, T, h, rot_flags, ink_r15_q4_fast_lds.ku.tmp[w], qa);
  if ((rot_flags & 1u) != 0u) {
    lbarrier();
    ink_r15_q4_lsignal();
  }

  float m, ls;
  {
    const u32 l_i = ink_r15_q4_lane();
    const u32 rl_i = l_i & 15u, hl_i = l_i >> 4u;
    const u32 tk_i = t_sub0 + rl_i;
    const u32 tq_i = (tk_i < T) ? tk_i : (T - 1u);
    const float sink_val = ((const G float *)((const G u8 *)sink + (ink_r15_q4_u64)tq_i * (ink_r15_q4_u64)ss))[h];
    m = sink_val * INK_R15_Q4_LOG2E;
    ls = (hl_i == 0u) ? 1.0f : 0.0f;
  }

  ink_r15_q4_lwait();
  v8f o[8];
#pragma unroll
  for (u32 nt = 0u; nt < 8u; nt++) {
    v8f z = (v8f)(0.0f);
    __asm__ volatile("" : "+v"(z));
    o[nt] = z;
  }
  __builtin_amdgcn_sched_barrier(0);

  ink_r15_q4_consumer16(&ink_r15_q4_fast_lds, R, rs, t_sub0, T, h,
                        k_start, qpos_min, sub_qpos_min,
                        win, extent_lim, 0u, nblk, blk_far_end, qa, &m, &ls, o);
  if ((rot_flags & 2u) != 0u) lbarrier();

  ink_r15_q4_store_o(out, os, t_sub0, T, h, rot_flags, ink_r15_q4_fast_lds.ku.tmp[w], ls, o);
}

#undef INK_R15_Q4_EXP_MAG
#undef INK_R15_Q4_EXP_C4
#undef INK_R15_Q4_EXP_C3
#undef INK_R15_Q4_EXP_C2
#undef INK_R15_Q4_EXP_C1
#undef INK_R15_Q4_QSC
#undef INK_R15_Q4_INV_SQRT
#undef INK_R15_Q4_LN2
#undef INK_R15_Q4_LOG2E
#undef INK_R15_Q4_NB32
#undef INK_R15_Q4_NB16
#undef INK_R15_Q4_PSTR
#undef INK_R15_Q4_RECB
#undef INK_R15_Q4_GQA
#undef INK_R15_Q4_DREL
#undef INK_R15_Q4_HD

#endif /* !IE_MEGA */
