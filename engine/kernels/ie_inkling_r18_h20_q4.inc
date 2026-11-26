/* ============================================================================
 * Inkling-2 Round 18 (R89): Single-Launch H20 QKV + QClip + Inline k_dyn/v_dyn
 * + K/V 4-Tap ShortConv + VClip + FWHT-128 + 72-B Q4_0 Global KV Store
 *
 * Target: RX 9070 XT (gfx1201, RDNA4, wave32, NT=256, 64 CUs, 64 MiB MALL)
 * Extends H20 (ie_inkling_qkv_sconv_kv_r11) to the 8 Global layers in 256K
 * q4rot mode, eliminating intermediate K/V round-trips and -8 launches/step.
 *
 * Grid: (wg_q + wg_k + wg_v, 1u, 1u) = (192 + 64 + 64, 1, 1) = 320 WGs
 *       (8 WGs/head x (24 Q + 8 K + 8 V heads), 16 rows/WG = 2 rows/wave).
 * ABI:  Exact drop-in signature of ie_inkling_qkv_sconv_kv_r11 with
 *       (G u8 *kc_q4, G u8 *vc_q4) replacing (G kvt *kc0, G kvt *vc0)
 *       plus trailing (u32 rot_flags) (bit 0 = rotate K, bit 1 = rotate V).
 *       64-bit KV byte offsets safe to 512K+, 0/0 spills, priv=0, 132 B LDS.
 *       100% bit-exact FWHT-128 + 72-B Q4_0 parity vs
 *       ie_inkling_c1_sconv_vclip_glob_q4_store_r16.
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R18_H20_CT       8u
#define INK_R18_H20_HD       128u
#define INK_R18_H20_RECB     72u
#define INK_R18_H20_INV_SQRT 0.08838834764831844299f

typedef unsigned long long ink_r18_h20_u64;
_Static_assert(sizeof(ink_r18_h20_u64) == 8u, "ink_r18_h20_u64 must be 8 bytes");

struct __attribute__((packed, aligned(4))) ink_r18_h20_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r18_h20_ld4(const G void *p) {
  return ((const G struct ink_r18_h20_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) float ink_r18_h20_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline __attribute__((always_inline)) int ink_r18_h20_dot8(
    u32x4 w0, u32x4 w1, u32x4 a0, u32x4 a1) {
  const u32 w00 = w0.x, w01 = w0.y, w02 = w0.z, w03 = w0.w;
  const u32 w10 = w1.x, w11 = w1.y, w12 = w1.z, w13 = w1.w;
  const u32 a00 = a0.x, a01 = a0.y, a02 = a0.z, a03 = a0.w;
  const u32 a10 = a1.x, a11 = a1.y, a12 = a1.z, a13 = a1.w;
  u32 d = ie_dot4_ss(0u, w00, a00);
  d = ie_dot4_ss(d, w01, a01);
  d = ie_dot4_ss(d, w02, a02);
  d = ie_dot4_ss(d, w03, a03);
  d = ie_dot4_ss(d, w10, a10);
  d = ie_dot4_ss(d, w11, a11);
  d = ie_dot4_ss(d, w12, a12);
  d = ie_dot4_ss(d, w13, a13);
  return (int)d;
}

static inline __attribute__((always_inline)) float ink_r18_h20_sub8_max_pos(float v) {
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0xB1, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0x4E, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_ds_swizzle(F2I(v), 0x101F)));
  return v;
}

static inline __attribute__((always_inline)) float ink_r18_h20_vclip_scale(
    float ss, float inv_hd, float clamp, u32 norope, float escale, u32 win, u32 pos0, u32 tk) {
  float f;
  if (clamp > 0.0f) {
    f = clamp / (__builtin_sqrtf(ss) + 1e-6f);
    if (f > 1.0f) f = 1.0f;
    if (norope) {
      const float ratio = (float)(pos0 + tk + 1u) / (float)win;
      if (ratio > 1.0f) f *= 1.0f + escale * __builtin_logf(ratio);
    }
  } else {
    f = 1.0f / __builtin_sqrtf(ss * inv_hd + escale);
  }
  return f;
}

static inline __attribute__((always_inline)) f32x4 ink_r18_h20_sconv4(
    f32x4 x0, f32x4 x1, f32x4 x2, f32x4 x3,
    f32x4 w0, f32x4 w1, f32x4 w2, f32x4 w3, float escale) {
#pragma clang fp contract(off)
  const f32x4 y = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3;
  return (x0 + y) * escale;
}

static inline __attribute__((always_inline)) void ink_r18_h20_load_sw(
    const G float *v, u32 i0, f32x4 *sw0, f32x4 *sw1, f32x4 *sw2, f32x4 *sw3) {
  const G f32x4 *v4 = (const G f32x4 *)v + i0;
  const f32x4 vw0 = v4[0], vw1 = v4[1], vw2 = v4[2], vw3 = v4[3];
  *sw0 = (f32x4){vw0.w, vw1.w, vw2.w, vw3.w};
  *sw1 = (f32x4){vw0.z, vw1.z, vw2.z, vw3.z};
  *sw2 = (f32x4){vw0.y, vw1.y, vw2.y, vw3.y};
  *sw3 = (f32x4){vw0.x, vw1.x, vw2.x, vw3.x};
}

static inline __attribute__((always_inline)) void ink_r18_h20_seed_ring(
    const G float *ring, u32 i0, u32 pos0, f32x4 *x1, f32x4 *x2, f32x4 *x3) {
  *x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (pos0 > 0u) {
    const G f32x4 *rg = (const G f32x4 *)ring + i0;
    const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
    const u32 s1 = (pos0 - 1u) & 3u;
    *x1 = (f32x4){ink_r18_h20_pick4(r0, s1), ink_r18_h20_pick4(r1, s1),
                  ink_r18_h20_pick4(r2, s1), ink_r18_h20_pick4(r3, s1)};
    if (pos0 >= 2u) {
      const u32 s2 = (pos0 - 2u) & 3u;
      *x2 = (f32x4){ink_r18_h20_pick4(r0, s2), ink_r18_h20_pick4(r1, s2),
                    ink_r18_h20_pick4(r2, s2), ink_r18_h20_pick4(r3, s2)};
    }
    if (pos0 >= 3u) {
      const u32 s3 = (pos0 - 3u) & 3u;
      *x3 = (f32x4){ink_r18_h20_pick4(r0, s3), ink_r18_h20_pick4(r1, s3),
                    ink_r18_h20_pick4(r2, s3), ink_r18_h20_pick4(r3, s3)};
    }
  }
}

static inline __attribute__((always_inline)) void ink_r18_h20_ring_commit(
    G float *ring, u32 i0, u32 pos0, u32 T, f32x4 x1, f32x4 x2, f32x4 x3, f32x4 x4) {
  if (T >= 4u) {
    const u32 s0 = (pos0 + T - 4u) & 3u;
    const f32x4 v0 = (s0 == 0u) ? x4 : ((s0 == 3u) ? x3 : ((s0 == 2u) ? x2 : x1));
    const f32x4 v1 = (s0 == 1u) ? x4 : ((s0 == 0u) ? x3 : ((s0 == 3u) ? x2 : x1));
    const f32x4 v2 = (s0 == 2u) ? x4 : ((s0 == 1u) ? x3 : ((s0 == 0u) ? x2 : x1));
    const f32x4 v3 = (s0 == 3u) ? x4 : ((s0 == 2u) ? x3 : ((s0 == 1u) ? x2 : x1));
    G f32x4 *rg = (G f32x4 *)ring + i0;
    rg[0] = (f32x4){v0.x, v1.x, v2.x, v3.x};
    rg[1] = (f32x4){v0.y, v1.y, v2.y, v3.y};
    rg[2] = (f32x4){v0.z, v1.z, v2.z, v3.z};
    rg[3] = (f32x4){v0.w, v1.w, v2.w, v3.w};
  } else {
    if (T >= 3u) {
      const u32 sl = (pos0 + T - 3u) & 3u;
      ring[(ink_r18_h20_u64)(i0 + 0u) * 4ull + sl] = x3.x;
      ring[(ink_r18_h20_u64)(i0 + 1u) * 4ull + sl] = x3.y;
      ring[(ink_r18_h20_u64)(i0 + 2u) * 4ull + sl] = x3.z;
      ring[(ink_r18_h20_u64)(i0 + 3u) * 4ull + sl] = x3.w;
    }
    if (T >= 2u) {
      const u32 sl = (pos0 + T - 2u) & 3u;
      ring[(ink_r18_h20_u64)(i0 + 0u) * 4ull + sl] = x2.x;
      ring[(ink_r18_h20_u64)(i0 + 1u) * 4ull + sl] = x2.y;
      ring[(ink_r18_h20_u64)(i0 + 2u) * 4ull + sl] = x2.z;
      ring[(ink_r18_h20_u64)(i0 + 3u) * 4ull + sl] = x2.w;
    }
    {
      const u32 sl = (pos0 + T - 1u) & 3u;
      ring[(ink_r18_h20_u64)(i0 + 0u) * 4ull + sl] = x1.x;
      ring[(ink_r18_h20_u64)(i0 + 1u) * 4ull + sl] = x1.y;
      ring[(ink_r18_h20_u64)(i0 + 2u) * 4ull + sl] = x1.z;
      ring[(ink_r18_h20_u64)(i0 + 3u) * 4ull + sl] = x1.w;
    }
  }
}

/* Zero-LDS, zero-barrier exact normalized 128-pt FWHT on 1 wave32 (matches r16 bit-for-bit). */
static inline __attribute__((always_inline)) f32x4 ink_r18_h20_fwht128_wave4(f32x4 v, u32 l) {
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
  return r * INK_R18_H20_INV_SQRT;
}

/* Lean-VGPR, low-SGPR bit-exact 72-B Q4_0 quantizer + coalesced store (matches r16 bit-for-bit, 0 spills). */
static inline __attribute__((always_inline)) void ink_r18_h20_q4_store_wave4_lean(
    f32x4 x, G u8 *dst, u32 l) {
  float am = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(x.x), __builtin_fabsf(x.y)),
                             __builtin_fmaxf(__builtin_fabsf(x.z), __builtin_fabsf(x.w)));
  am = ink_r18_h20_sub8_max_pos(am);
  const f16 sh = (f16)(am * (1.0f / 7.0f));
  float s = (float)sh;
  __asm__ volatile("" : "+v"(s));
  const int nz = (s != 0.0f);
  const float sd = nz ? s : 1.0f;
  u32 pk = 0u;
#pragma clang loop unroll(disable)
  for (u32 k = 0u; k < 4u; k++) {
    const float v = (k == 0u) ? x.x : ((k == 1u) ? x.y : ((k == 2u) ? x.z : x.w));
    const float vn = nz ? (v / sd) : 0.0f;
    const float qf = __builtin_fmaxf(-8.0f, __builtin_fminf(7.0f, __builtin_roundf(vn)));
    const int qi = (int)qf;
    pk |= ((u32)(qi + 8)) << (k << 2u);
  }
  ((G u16 *)dst)[l] = (u16)pk;
  if ((l & 7u) == 0u) {
    ((G f16 *)(dst + 64u))[l >> 3u] = sh;
  }
}

static inline __attribute__((always_inline)) void ink_r18_h20_q8_2rows_t8(
    const G u32 *qw, const G f16 *qs, const G u8 *xq,
    u32 r0, u32 r1, u32 nb, u32 T, u32 xs, u32 l,
    v8f *out0, v8f *out1) {
#pragma clang fp contract(off)
  const G u8 *rw0 = rq_ptr_u((const G u8 *)(qw + (ink_r18_h20_u64)r0 * (ink_r18_h20_u64)nb * 8ull));
  const G f16 *rs0 = (const G f16 *)rq_ptr_u((const G u8 *)(qs + (ink_r18_h20_u64)r0 * (ink_r18_h20_u64)nb));
  const G u8 *rw1 = rq_ptr_u((const G u8 *)(qw + (ink_r18_h20_u64)r1 * (ink_r18_h20_u64)nb * 8ull));
  const G f16 *rs1 = (const G f16 *)rq_ptr_u((const G u8 *)(qs + (ink_r18_h20_u64)r1 * (ink_r18_h20_u64)nb));
  const G u8 *xb = rq_ptr_u(xq);
  const ink_r18_h20_u64 sc_off = 32ull * (ink_r18_h20_u64)nb;

  v8f a0 = (v8f)(0.0f);
  v8f a1 = (v8f)(0.0f);

#pragma clang loop unroll(disable)
  for (u32 b = l; b < nb; b += 32u) {
    const u32 bo = b << 5u;
    const u32x4 w0a = ink_r18_h20_ld4(rw0 + bo);
    const u32x4 w0b = ink_r18_h20_ld4(rw0 + bo + 16u);
    const float sw0 = (float)rs0[b];
    const u32x4 w1a = ink_r18_h20_ld4(rw1 + bo);
    const u32x4 w1b = ink_r18_h20_ld4(rw1 + bo + 16u);
    const float sw1 = (float)rs1[b];
    __builtin_amdgcn_sched_barrier(0);

#pragma unroll
    for (u32 j = 0u; j < INK_R18_H20_CT; j++) {
      if (j < T) {
        const G u8 *xj = xb + (ink_r18_h20_u64)j * (ink_r18_h20_u64)xs;
        const u32x4 x0 = ink_r18_h20_ld4(xj + bo);
        const u32x4 x1 = ink_r18_h20_ld4(xj + bo + 16u);
        const float da = ((const G float *)(xj + sc_off))[b];
        a0[j] = a0[j] + (sw0 * da) * (float)ink_r18_h20_dot8(w0a, w0b, x0, x1);
        a1[j] = a1[j] + (sw1 * da) * (float)ink_r18_h20_dot8(w1a, w1b, x0, x1);
        if ((j & 1u) == 1u) __builtin_amdgcn_sched_barrier(0);
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  v8f res0 = (v8f)(0.0f);
  v8f res1 = (v8f)(0.0f);
#pragma unroll
  for (u32 j = 0u; j < INK_R18_H20_CT; j++) {
    if (j < T) {
      res0[j] = wave_tree_f(a0[j]);
      res1[j] = wave_tree_f(a1[j]);
    }
  }
  *out0 = res0;
  *out1 = res1;
}

static inline __attribute__((always_inline)) void ink_r18_h20_q8_1row_t8(
    const G u32 *qw, const G f16 *qs, const G u8 *xq,
    u32 r, u32 nb, u32 T, u32 xs, u32 l, v8f *out0) {
#pragma clang fp contract(off)
  const G u8 *rw0 = rq_ptr_u((const G u8 *)(qw + (ink_r18_h20_u64)r * (ink_r18_h20_u64)nb * 8ull));
  const G f16 *rs0 = (const G f16 *)rq_ptr_u((const G u8 *)(qs + (ink_r18_h20_u64)r * (ink_r18_h20_u64)nb));
  const G u8 *xb = rq_ptr_u(xq);
  const ink_r18_h20_u64 sc_off = 32ull * (ink_r18_h20_u64)nb;

  v8f a0 = (v8f)(0.0f);
#pragma clang loop unroll(disable)
  for (u32 b = l; b < nb; b += 32u) {
    const u32 bo = b << 5u;
    const u32x4 w0a = ink_r18_h20_ld4(rw0 + bo);
    const u32x4 w0b = ink_r18_h20_ld4(rw0 + bo + 16u);
    const float sw0 = (float)rs0[b];
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (u32 j = 0u; j < INK_R18_H20_CT; j++) {
      if (j < T) {
        const G u8 *xj = xb + (ink_r18_h20_u64)j * (ink_r18_h20_u64)xs;
        const u32x4 x0 = ink_r18_h20_ld4(xj + bo);
        const u32x4 x1 = ink_r18_h20_ld4(xj + bo + 16u);
        const float da = ((const G float *)(xj + sc_off))[b];
        a0[j] = a0[j] + (sw0 * da) * (float)ink_r18_h20_dot8(w0a, w0b, x0, x1);
      }
    }
  }
  v8f res0 = (v8f)(0.0f);
#pragma unroll
  for (u32 j = 0u; j < INK_R18_H20_CT; j++) {
    if (j < T) res0[j] = wave_tree_f(a0[j]);
  }
  *out0 = res0;
}

KERNEL ie_inkling_qkv_sconv_kv_glob_q4_r18(
    const G u32 *qw_q, const G f16 *qs_q, const G u8 *xq_q, G float *y_q_raw, G float *y_q_clip,
    const G u32 *qw_k, const G f16 *qs_k, const G u8 *xq_k, G float *y_k,
    const G u32 *qw_v, const G f16 *qs_v, const G u8 *xq_v, G float *y_v,
    const G u32 *qw_kd, const G f16 *qs_kd,
    const G u32 *qw_vd, const G f16 *qs_vd,
    const G float *v_qclip,
    const G float *vk, G float *ring_k, G float *hist_k,
    const G float *vv, G float *ring_v, G float *hist_v,
    G u8 *kc_q4, G u8 *vc_q4, G u32 *head_cnt,
    u32 n_head, u32 n_kv, u32 hd, u32 nb, u32 kv_len,
    float q_clamp, u32 norope, float q_escale, u32 win,
    float sc_escale, float k_clamp, float v_eps,
    u32 pos0, u32 T,
    u32 xs_q, u32 xs_k, u32 xs_v,
    u32 ys_qr, u32 ys_qc, u32 ys_k, u32 ys_v, u32 hrs, u32 rot_flags) {
  static LDS u32   ink_r18_h20_last;
  static LDS float ink_r18_h20_dyn[8u][4u] __attribute__((aligned(16))); /* 128 B */

  if (T == 0u || T > 8u || hd != INK_R18_H20_HD || n_head == 0u || n_kv == 0u ||
      nb == 0u || kv_len == 0u)
    return;

  const u32 wg_q = n_head << 3u; /* 192 WGs (8 WGs / Q head) */
  const u32 wg_k = n_kv << 3u;   /*  64 WGs (8 WGs / K head) */
  const u32 g = wgid();
  if (g >= wg_q + (wg_k << 1u)) return;

  u32 kind = 0u; /* 0 = Q, 1 = K, 2 = V */
  u32 g_sel = g, xs_sel = xs_q, ys_sel = ys_qr;
  const G u32 *qw_sel = qw_q;
  const G f16 *qs_sel = qs_q;
  const G u8  *xq_sel = xq_q;
  G float     *y_sel  = y_q_raw;

  if (g >= wg_q + wg_k) {
    kind = 2u; g_sel = g - (wg_q + wg_k);
    qw_sel = qw_v; qs_sel = qs_v; xq_sel = xq_v; y_sel = y_v; xs_sel = xs_v; ys_sel = ys_v;
  } else if (g >= wg_q) {
    kind = 1u; g_sel = g - wg_q;
    qw_sel = qw_k; qs_sel = qs_k; xq_sel = xq_k; y_sel = y_k; xs_sel = xs_k; ys_sel = ys_k;
  }

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 r0 = (g_sel << 4u) + (w << 1u);
  const u32 r1 = r0 + 1u;

  {
    v8f v0, v1;
    ink_r18_h20_q8_2rows_t8(qw_sel, qs_sel, xq_sel, r0, r1, nb, T, xs_sel, l, &v0, &v1);
    if (l == 0u) {
#pragma unroll
      for (u32 j = 0u; j < INK_R18_H20_CT; j++) {
        if (j < T) {
          G float *yt = (G float *)((G u8 *)y_sel + (ink_r18_h20_u64)j * (ink_r18_h20_u64)ys_sel);
          yt[r0] = v0[j];
          yt[r1] = v1[j];
        }
      }
    }
  }

  /* 8 WGs share each 128-dim head h = g_sel >> 3; atomic arrival elects 8th WG. */
  const u32 h = g_sel >> 3u;
  const u32 c_idx = (kind == 0u) ? h : ((kind == 1u) ? (n_head + h) : (n_head + n_kv + h));
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r18_h20_last =
        (__scoped_atomic_fetch_add(head_cnt + c_idx, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == 7u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r18_h20_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  if (tid() == 0u) head_cnt[c_idx] = 0u;

  if (kind == 0u) {
    /* Q-head tail: exact vclip(q) matching ie_inkling_qkv_sconv_kv_r11. */
    const G f32x4 *vp = v_qclip ? ((const G f32x4 *)v_qclip + (ink_r18_h20_u64)h * 32ull) : (const G f32x4 *)0;
    const f32x4 vqv = vp ? vp[l] : (f32x4){1.0f, 1.0f, 1.0f, 1.0f};
    if (w < T) {
      const G f32x4 *ah = (const G f32x4 *)((const G u8 *)y_sel + (ink_r18_h20_u64)w * (ink_r18_h20_u64)ys_sel)
                          + (ink_r18_h20_u64)h * 32ull;
      G f32x4 *bh = (G f32x4 *)((G u8 *)y_q_clip + (ink_r18_h20_u64)w * (ink_r18_h20_u64)ys_qc)
                    + (ink_r18_h20_u64)h * 32ull;
      f32x4 xv = ah[l];
      if (q_clamp > 0.0f && vp) xv *= vqv;
      float ss = (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
      ss = wave_sum_all(ss);
      const float f = ink_r18_h20_vclip_scale(ss, 1.0f / 128.0f, q_clamp, norope, q_escale, win, pos0, w);
      f32x4 outv = xv * f;
      if (q_clamp == 0.0f && vp) outv *= vqv;
      bh[l] = outv;
    }
    return;
  }

  /* K or V head tail: optional inline 4-row dynamic conv proj into LDS (0 global I/O!). */
  const int is_k = (kind == 1u);
  const G u32 *qw_d = is_k ? qw_kd : qw_vd;
  const G f16 *qs_d = is_k ? qs_kd : qs_vd;
  const int has_dyn = (qw_d != (const G u32 *)0 && qs_d != (const G f16 *)0);
  if (has_dyn) {
    if (w < 4u) {
      v8f dv;
      ink_r18_h20_q8_1row_t8(qw_d, qs_d, xq_sel, (h << 2u) + w, nb, T, xs_sel, l, &dv);
      if (l == 0u) {
#pragma unroll
        for (u32 j = 0u; j < INK_R18_H20_CT; j++) {
          if (j < T) ink_r18_h20_dyn[j][w] = dv[j];
        }
      }
    }
    lbarrier();
  }

  /* Wave 0 (32 lanes x f32x4 = 128 floats = 1 full KV head h):
   * exact 4-tap sconv + per-head vclip + inline FWHT-128 + 72-B Q4_0 KV store + hist/ring! */
  if (w == 0u) {
    const u32 i4 = (h << 5u) + l;
    const u32 i0 = i4 << 2u;
    const G float *vc_w = is_k ? vk : vv;
    G float *ring = is_k ? ring_k : ring_v;
    G float *hist = is_k ? hist_k : hist_v;
    G u8 *dst_q4 = is_k ? kc_q4 : vc_q4;
    if (!dst_q4) return;
    const u32 do_rot = (rot_flags >> (is_k ? 0u : 1u)) & 1u;

    f32x4 sw0, sw1, sw2, sw3;
    ink_r18_h20_load_sw(vc_w, i0, &sw0, &sw1, &sw2, &sw3);
    f32x4 x1, x2, x3, x4 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    ink_r18_h20_seed_ring(ring, i0, pos0, &x1, &x2, &x3);

    u32 p = pos0 % kv_len;
#pragma clang loop unroll(disable)
    for (u32 t = 0u; t < T; t++) {
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)y_sel + (ink_r18_h20_u64)t * (ink_r18_h20_u64)ys_sel))[i4];
      if (hist) ((G f32x4 *)(hist + (ink_r18_h20_u64)t * (ink_r18_h20_u64)hrs))[i4] = x0;

      f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
      if (has_dyn) {
        w0 += ink_r18_h20_dyn[t][0u];
        w1 += ink_r18_h20_dyn[t][1u];
        w2 += ink_r18_h20_dyn[t][2u];
        w3 += ink_r18_h20_dyn[t][3u];
      }
      const f32x4 sx = ink_r18_h20_sconv4(x0, x1, x2, x3, w0, w1, w2, w3, sc_escale);
      float ss = (sx.x * sx.x + sx.y * sx.y) + (sx.z * sx.z + sx.w * sx.w);
      ss = wave_sum_all(ss);
      float f;
      if (is_k) {
        f = k_clamp / (__builtin_sqrtf(ss) + 1e-6f);
        if (f > 1.0f) f = 1.0f;
      } else {
        f = 1.0f / __builtin_sqrtf(ss * (1.0f / 128.0f) + v_eps);
      }
      f32x4 xc = sx * f;
      if (do_rot != 0u) xc = ink_r18_h20_fwht128_wave4(xc, l);

      G u8 *dst = dst_q4 + ((ink_r18_h20_u64)p * (ink_r18_h20_u64)n_kv + (ink_r18_h20_u64)h) * INK_R18_H20_RECB;
      ink_r18_h20_q4_store_wave4_lean(xc, dst, l);

      x4 = x3; x3 = x2; x2 = x1; x1 = x0;
      p++;
      if (p == kv_len) p = 0u;
    }
    if (!hist) ink_r18_h20_ring_commit(ring, i0, pos0, T, x1, x2, x3, x4);
  }
}

#undef INK_R18_H20_INV_SQRT
#undef INK_R18_H20_RECB
#undef INK_R18_H20_HD
#undef INK_R18_H20_CT

#endif /* !IE_MEGA */
