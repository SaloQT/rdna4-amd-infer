/* ============================================================================
 * Inkling-2-Mini Round 16 (256K Q4-KV Tree-Verify, C1 Fusion & Fast PF Store):
 *   1. ie_inkling_rel_attn_ver_glob_q4_tree_r16 (R68 H26-Q4-TREE, bit 164)
 *        Single-pass T<=8 (up to 24 GQA queries) OPT-Tree FP16 WMMA Global
 *        Q4_0+FWHT-128 Verify Attention + optional fused OP_GATE(sigmoid)+Q8_0.
 *        Uses b->ink_tree_mask (uint64_t, 8 bits/row) & b->ink_tree_dep_pack
 *        (u32, 4-bit depth/node). >99.6% of 256K chunks run zero-overhead
 *        far-field fast loop; tree mask & tree-depth rel_bias apply strictly
 *        inside near-field chunks.
 *        Grid: (n_kv * nsplit, 1, 1), NT=256, 15,360 B LDS, <= 96 VGPR (4 WG/CU).
 *   2. ie_inkling_c1_sconv_vclip_glob_q4_store_r16 (R65 C1-GLOB-Q4, bit 163)
 *        Fuses [sconv(k), vclip(k), sconv(v), vclip(v), FWHT-128, 72-B Q4_0 KV
 *        store] + inline sc_hist/ring commit for T=1..8 (both linear chain
 *        tpar_pack=0x65432100u and arbitrary 8-node trees). Zero LDS, zero
 *        barriers, eliminates -16 launches/step on 8 global layers in q4rot!
 *        Grid: (1, 1, 1), NT=256 (8 waves = 8 KV heads), 0 B LDS, <= 64 VGPR.
 *   3. ie_inkling_kv_store_glob_q4_rot_pf_r16 (bit 165)
 *        Batched prefill (T=1..1024+) zero-LDS, zero-barrier wave-per-head
 *        optional vclip(k/v) + FWHT-128 + 72-B Q4_0 Global KV store (<0.2 ms
 *        for 1024 tokens x 8 global layers!).
 *        Grid: (((T * n_kv) + 7) / 8, 1, 1) [=(T, 1, 1) at n_kv=8], NT=256.
 *
 * Target: RX 9070 XT (gfx1201, RDNA4, wave32, NT=256, 64 CUs, 64 MiB MALL)
 * ABI: 72 B/head/pos planar Q4_0 + FWHT-128 (rot_flags in {0,1,2,3}), 64-bit
 *      KV byte offsets, slack=0 safe to 512K+, 0/0 spills, priv=0, scratch=0,
 *      TH_LOAD_NT=0, TH_LOAD_HT=0, scalar __builtin_bit_cast safe.
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(f16) == 2u, "ink_r16 f16 must be 2 bytes");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r16 float/u32 must be 4 bytes");

typedef unsigned long long ink_r16_u64;
typedef unsigned int       ink_r16_u2 __attribute__((ext_vector_type(2)));
typedef f16                ink_r16_h4 __attribute__((ext_vector_type(4)));
_Static_assert(sizeof(ink_r16_u64) == 8u, "ink_r16_u64 must be 8 bytes");

#define INK_R16_HD       128u
#define INK_R16_DREL     16u
#define INK_R16_GQA      3u
#define INK_R16_RECB     72u
#define INK_R16_PSTR     130u
#define INK_R16_LOG2E    1.44269504088896340736f
#define INK_R16_LN2      0.69314718055994530942f
#define INK_R16_INV_SQRT 0.08838834764831844299f
#define INK_R16_QSC      (0.08838834764831844299f * 1.44269504088896340736f)

static inline __attribute__((always_inline)) u32 ink_r16_lane_pin(void) {
  u32 l;
  __asm__ volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(l));
  return l;
}

static inline __attribute__((always_inline)) float ink_r16_sgpr_f(float v) {
  return I2F(__builtin_amdgcn_readfirstlane(F2I(v)));
}

static inline __attribute__((always_inline)) float ink_r16_xrow(float v) {
  return I2F(xrow_i(F2I(v)));
}

static inline __attribute__((always_inline)) float ink_r16_gate_f(float o, float g) {
#pragma clang fp contract(off)
  return o * (1.0f / (1.0f + __builtin_expf(-g)));
}

static inline __attribute__((always_inline)) float ink_r16_sub8_max_pos(float v) {
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0xB1, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0x4E, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_ds_swizzle(F2I(v), 0x101F)));
  return v;
}

static inline __attribute__((always_inline)) u32 ink_r16_sub8_sum_u32(u32 s) {
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0xB1, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0x4E, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_ds_swizzle((int)s, 0x101F);
  return s;
}

/* Zero-LDS, zero-barrier exact normalized 128-pt FWHT on 1 wave32 (lane l -> 4*l..4*l+3). */
static inline __attribute__((always_inline)) f32x4 ink_r16_fwht128_wave4(f32x4 v, u32 l) {
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
  return r * INK_R16_INV_SQRT;
}

/* Zero-LDS, zero-barrier bit-exact 72-B Q4_0 quantizer + coalesced store on 1 wave32.
 * Lane l (0..31) holds f32x4 for elements [4*l .. 4*l + 3]; subgroup sg=l>>3 (8 lanes = 32 elems). */
static inline __attribute__((always_inline)) void ink_r16_q4_store_wave4(
    f32x4 x, G u8 *dst, u32 l) {
  float am = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(x.x), __builtin_fabsf(x.y)),
                             __builtin_fmaxf(__builtin_fabsf(x.z), __builtin_fabsf(x.w)));
  am = ink_r16_sub8_max_pos(am);
  const f16 sh = (f16)(am * (1.0f / 7.0f));
  float s = (float)sh;
  __asm__ volatile("" : "+v"(s));
  const int nz = (s != 0.0f);
  int q0 = 0, q1 = 0, q2 = 0, q3 = 0;
  if (nz) {
    q0 = (int)__builtin_roundf(x.x / s);
    q1 = (int)__builtin_roundf(x.y / s);
    q2 = (int)__builtin_roundf(x.z / s);
    q3 = (int)__builtin_roundf(x.w / s);
    if (q0 > 7) q0 = 7; else if (q0 < -8) q0 = -8;
    if (q1 > 7) q1 = 7; else if (q1 < -8) q1 = -8;
    if (q2 > 7) q2 = 7; else if (q2 < -8) q2 = -8;
    if (q3 > 7) q3 = 7; else if (q3 < -8) q3 = -8;
  }
  const u16 pk16 = (u16)((u32)(q0 + 8) | (((u32)(q1 + 8)) << 4u) |
                         (((u32)(q2 + 8)) << 8u) | (((u32)(q3 + 8)) << 12u));
  ((G u16 *)dst)[l] = pk16;
  if ((l & 7u) == 0u) {
    ((G f16 *)(dst + 64u))[l >> 3u] = sh;
  }
}

/* Lean-VGPR variant of ink_r16_q4_store_wave4 (serializes the 4 IEEE divisions so only 1 div
 * is live at a time; 100% bit-identical output, 0 B scratch/priv, saves ~20 VGPRs). */
static inline __attribute__((always_inline)) void ink_r16_q4_store_wave4_lean(
    f32x4 x, G u8 *dst, u32 l) {
  float am = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(x.x), __builtin_fabsf(x.y)),
                             __builtin_fmaxf(__builtin_fabsf(x.z), __builtin_fabsf(x.w)));
  am = ink_r16_sub8_max_pos(am);
  const f16 sh = (f16)(am * (1.0f / 7.0f));
  float s = (float)sh;
  __asm__ volatile("" : "+v"(s));
  u32 pk = 0x8888u;
  if (s != 0.0f) {
    pk = 0u;
#pragma clang loop unroll(disable)
    for (u32 k = 0u; k < 4u; k++) {
      const float v = (k == 0u) ? x.x : ((k == 1u) ? x.y : ((k == 2u) ? x.z : x.w));
      int qi = (int)__builtin_roundf(v / s);
      if (qi > 7) qi = 7; else if (qi < -8) qi = -8;
      pk |= ((u32)(qi + 8)) << (k << 2u);
    }
  }
  ((G u16 *)dst)[l] = (u16)pk;
  if ((l & 7u) == 0u) {
    ((G f16 *)(dst + 64u))[l >> 3u] = sh;
  }
}

/* Zero-LDS in-register f32x4 OP_GATE(sigmoid) + Q8_0 epilogue for 1 head (hd=128=4x32). */
static inline __attribute__((always_inline)) void ink_r16_gate_q8_head_vec4(
    f32x4 ov, const G float *gt_h, G float *ot_h, G u8 *oqt, u32 h, u32 n_head, u32 l) {
#pragma clang fp contract(off)
  f32x4 vv = ov;
  if (gt_h) {
    const f32x4 gv = ((const G f32x4 *)gt_h)[l];
    vv = (f32x4){ink_r16_gate_f(ov.x, gv.x), ink_r16_gate_f(ov.y, gv.y),
                 ink_r16_gate_f(ov.z, gv.z), ink_r16_gate_f(ov.w, gv.w)};
  }
  if (ot_h) ((G f32x4 *)ot_h)[l] = vv;
  if (oqt) {
    const u32 sg = l >> 3u;
    const u32 sl = l & 7u;
    const u32 blk = (h << 2u) + sg;
    const u32 nb_q = n_head << 2u;
    float am = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(vv.x), __builtin_fabsf(vv.y)),
                               __builtin_fmaxf(__builtin_fabsf(vv.z), __builtin_fabsf(vv.w)));
    am = ink_r16_sub8_max_pos(am);
    const float d = am / 127.0f;
    int qi0 = 0, qi1 = 0, qi2 = 0, qi3 = 0;
    if (d != 0.0f) {
      qi0 = (int)__builtin_roundf(vv.x / d);
      qi1 = (int)__builtin_roundf(vv.y / d);
      qi2 = (int)__builtin_roundf(vv.z / d);
      qi3 = (int)__builtin_roundf(vv.w / d);
    }
    const u32 pk = ((u32)(u8)qi0) | (((u32)(u8)qi1) << 8u) |
                   (((u32)(u8)qi2) << 16u) | (((u32)(u8)qi3) << 24u);
    *(G u32 *)(oqt + (blk << 5u) + (sl << 2u)) = pk;
    const u32 s = ink_r16_sub8_sum_u32((u32)qi0 + (u32)qi1 + (u32)qi2 + (u32)qi3);
    if (sl == 0u) {
      ((G float *)(oqt + (nb_q << 5u)))[blk] = d;
      ((G u32 *)(oqt + nb_q * 36u))[blk] = s;
    }
  }
}

static inline __attribute__((always_inline)) h8 ink_r16_deq8n_h8(u32 w, float s) {
  return (h8){(f16)(s * (float)((int)(w & 15u) - 8)),
              (f16)(s * (float)((int)((w >> 4u) & 15u) - 8)),
              (f16)(s * (float)((int)((w >> 8u) & 15u) - 8)),
              (f16)(s * (float)((int)((w >> 12u) & 15u) - 8)),
              (f16)(s * (float)((int)((w >> 16u) & 15u) - 8)),
              (f16)(s * (float)((int)((w >> 20u) & 15u) - 8)),
              (f16)(s * (float)((int)((w >> 24u) & 15u) - 8)),
              (f16)(s * (float)((int)(w >> 28u) - 8))};
}

static inline __attribute__((always_inline)) f32x4 ink_r16_vclip_k(f32x4 kx, float k_clamp) {
  float ss = (kx.x * kx.x + kx.y * kx.y) + (kx.z * kx.z + kx.w * kx.w);
  ss = wave_sum_all(ss);
  float fk = k_clamp / (__builtin_sqrtf(ss) + 1e-6f);
  if (fk > 1.0f) fk = 1.0f;
  return kx * fk;
}

static inline __attribute__((always_inline)) f32x4 ink_r16_vclip_v(f32x4 vx, float v_eps) {
  float ss = (vx.x * vx.x + vx.y * vx.y) + (vx.z * vx.z + vx.w * vx.w);
  ss = wave_sum_all(ss);
  const float fv = 1.0f / __builtin_sqrtf(ss * (1.0f / 128.0f) + v_eps);
  return vx * fv;
}

/* ============================================================================
 * 1. ie_inkling_rel_attn_ver_glob_q4_tree_r16 (R68 H26-Q4-TREE, bit 164)
 *    Single-pass T<=8 OPT-Tree FP16 WMMA Global Q4_0+FWHT-128 Verify Attention
 *    + optional fused OP_GATE(sigmoid)+Q8_0.
 *    Grid: (n_kv * nsplit, 1, 1), NT=256, 15,360 B LDS, <= 96 VGPR (4 WG/CU).
 * ========================================================================== */

typedef union {
  struct {
    f16 q[32u * 128u] __attribute__((aligned(16)));         /* 8,192 B */
    f16 r[32u * 16u] __attribute__((aligned(16)));          /* 1,024 B */
    f16 p_tile[2u][2u][256u] __attribute__((aligned(16)));  /* 2,048 B */
    union {
      u32 v_sc[8u][128u] __attribute__((aligned(16)));      /* 4,096 B */
      struct {
        float mx[2u][128u];                                 /* 1,024 B */
        float ls[2u][128u];                                 /* 1,024 B */
      } red;
    } scratch;                                              /* 4,096 B */
  } kv;                                                     /* 15,360 B */
  struct {
    float o_stg[8u][128u] __attribute__((aligned(16)));     /* 4,096 B */
    float m[32u];                                           /*   128 B */
    float l[32u];                                           /*   128 B */
    float wt[8u][128u] __attribute__((aligned(16)));        /* 4,096 B */
    u32   is_last;                                          /*     4 B */
  } ph4;                                                    /* 8,452 B */
} ink_r16_tr_lds_t;

static LDS ink_r16_tr_lds_t ink_r16_tr_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r16_tr_lds_t) == 15360u, "ink_r16_tr_lds_t must be 15,360 B (<= 16 KiB)");

static inline __attribute__((always_inline)) u32 ink_r16_tr_q_off(u32 row, u32 c8) {
  return (row << 7u) + ((c8 ^ (row & 15u)) << 3u);
}

static inline __attribute__((always_inline)) u32 ink_r16_tr_p_off(u32 rl, u32 hl) {
  return (rl << 4u) + ((hl ^ ((rl >> 2u) & 1u)) << 3u);
}

static inline __attribute__((always_inline)) h8 ink_r16_tr_v_transpose(
    const h8 vh, LDSP u32 *v_sc_w, const u32 hl, const u32 rl) {
  const u32x4 vr = __builtin_bit_cast(u32x4, vh);
  const u32 sel = (rl & 1u) ? 0x03020706u : 0x05040100u;
  const u32 kh = rl >> 3u;
  const u32 k4 = (rl >> 1u) & 3u;
  const u32 w_arr[4] = {vr.x, vr.y, vr.z, vr.w};
#pragma unroll
  for (u32 q = 0u; q < 4u; q++) {
    const u32 x = w_arr[q];
    const u32 y = (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0xB1, 0xF, 0xF, true);
    const u32 t = __builtin_amdgcn_perm(y, x, sel);
    const u32 d = (hl << 3u) + (q << 1u) + (rl & 1u);
    const u32 s4 = ((d & 7u) ^ ((d >> 3u) << 1u) ^ (kh << 2u)) + ((d >> 3u) << 3u) + (kh << 4u);
    v_sc_w[(s4 << 2u) + k4] = t;
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  const u32 s4_r = ((rl & 7u) ^ ((rl >> 3u) << 1u) ^ (hl << 2u)) + ((rl >> 3u) << 3u) + (hl << 4u);
  return *(const LDSP h8 *)(v_sc_w + (s4_r << 2u));
}

/* Tree-aware near-field 16-distance rel_bias WMMA application (0 extra LDS). */
static inline __attribute__((always_inline)) void ink_r16_tr_apply_rb16(
    const v8f rb, const u32 base_off, const u32 hl, const u32 kb, const u32 pos0,
    const u32 T, const u32 dep_t, const u32 rmask_t, const u32 pos_off_packed,
    const u32 rel_extent, const u32 d_base, LDSP float *sc_row, v8f *sv) {
  if (hl == 0u) {
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) sc_row[v] = rb[v];
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma clang loop unroll(disable)
  for (u32 v = 0u; v < 8u; v++) {
    const u32 kp = kb + (hl << 3u) + v;
    int valid = 0;
    u32 dist = 0u;
    if (kp < pos0) {
      valid = 1;
      dist = (pos0 + dep_t) - kp;
    } else {
      const u32 j = kp - pos0;
      if (j < T && ((rmask_t >> j) & 1u) != 0u) {
        valid = 1;
        const u32 dep_j = (pos_off_packed >> (j << 2u)) & 0xFu;
        dist = (dep_t >= dep_j) ? (dep_t - dep_j) : 0u;
      }
    }
    if (valid && dist < rel_extent) {
      const u32 rel = (dist - d_base) - base_off;
      if (rel < 8u) (*sv)[v] += sc_row[rel];
    }
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  if (hl == 1u) {
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) sc_row[v] = rb[v];
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma clang loop unroll(disable)
  for (u32 v = 0u; v < 8u; v++) {
    const u32 kp = kb + (hl << 3u) + v;
    int valid = 0;
    u32 dist = 0u;
    if (kp < pos0) {
      valid = 1;
      dist = (pos0 + dep_t) - kp;
    } else {
      const u32 j = kp - pos0;
      if (j < T && ((rmask_t >> j) & 1u) != 0u) {
        valid = 1;
        const u32 dep_j = (pos_off_packed >> (j << 2u)) & 0xFu;
        dist = (dep_t >= dep_j) ? (dep_t - dep_j) : 0u;
      }
    }
    if (valid && dist < rel_extent) {
      const u32 rel = (dist - d_base) - (base_off + 8u);
      if (rel < 8u) (*sv)[v] += sc_row[rel];
    }
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

static inline __attribute__((always_inline)) void ink_r16_tr_merge_head(
    const G float *ph, const float sink_val, const u32 nsplit, const u32 l,
    LDSP float *wt_w, const G float *gt_h, G float *ot_h, G u8 *oqt,
    u32 h, u32 n_head, u32 rot_flags) {
  float mv = (l == 0u) ? sink_val : -__builtin_inff();
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const ink_r16_u64 off = (ink_r16_u64)sp * INK_R16_PSTR;
    if (ph[off + 1u] > 0.0f) mv = __builtin_fmaxf(mv, ph[off]);
  }
#pragma unroll
  for (u32 mask = 16u; mask >= 1u; mask >>= 1u) mv = __builtin_fmaxf(mv, shfl_xor_f(mv, mask));
  const float M2 = ink_r16_sgpr_f(mv);

  float lv = 0.0f;
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const ink_r16_u64 off = (ink_r16_u64)sp * INK_R16_PSTR;
    const float lp = ph[off + 1u];
    const float wt = (lp > 0.0f) ? __builtin_expf(ph[off] - M2) : 0.0f;
    if (sp < 128u) wt_w[sp] = wt;
    lv += lp * wt;
  }
  const float L2 = ink_r16_sgpr_f(wave_sum_all(lv) + __builtin_expf(sink_val - M2));
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");

  f32x4 acc = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  for (u32 sp = 0u; sp < nsplit; sp++) {
    const float wt = (sp < 128u) ? wt_w[sp]
                                 : ((ph[(ink_r16_u64)sp * INK_R16_PSTR + 1u] > 0.0f)
                                        ? __builtin_expf(ph[(ink_r16_u64)sp * INK_R16_PSTR] - M2)
                                        : 0.0f);
    if (wt > 0.0f) {
      const G float *po = ph + (ink_r16_u64)sp * INK_R16_PSTR + 2u + (l << 2u);
      const f32x4 ov = {po[0], po[1], po[2], po[3]};
      acc += ov * wt;
    }
  }
  f32x4 ov = acc * (1.0f / L2);
  if ((rot_flags & 2u) != 0u) ov = ink_r16_fwht128_wave4(ov, l);
  ink_r16_gate_q8_head_vec4(ov, gt_h, ot_h, oqt, h, n_head, l);
}

KERNEL ie_inkling_rel_attn_ver_glob_q4_tree_r16(
    const G float *q, const G u8 *kc_q4, const G u8 *vc_q4,
    const G float *R, const G float *rel_bias, const G float *sink,
    G float *part, G float *out, u32 pos0, u32 T, u32 kv_len, u32 hd,
    u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
    u32 qs, u32 rs, u32 ss, u32 os, G u32 *count,
    ink_r16_u64 tree_mask, u32 pos_off_packed,
    const G float *agate, G u8 *oq, u32 gs, u32 oqs, u32 rot_flags) {
  if (T == 0u || T > 8u || win != 0u || hd != INK_R16_HD || d_rel != INK_R16_DREL || n_kv == 0u ||
      n_head != INK_R16_GQA * n_kv || kv_len == 0u || nsplit == 0u)
    return;

  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() / nsplit));
  const u32 s = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() % nsplit));
  if (kh >= n_kv) return;

  const u32 h0 = (u32)__builtin_amdgcn_readfirstlane((int)(kh * INK_R16_GQA));
  const ink_r16_u64 row_step = (ink_r16_u64)n_kv * INK_R16_RECB;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));

  const u32 nq_tot = (u32)__builtin_amdgcn_readfirstlane((int)(T * INK_R16_GQA));
  const u32 npg = (u32)__builtin_amdgcn_readfirstlane((int)(nq_tot > 16u ? 2u : 1u));
  const u32 np = (u32)__builtin_amdgcn_readfirstlane((int)(pos0 + T));
  const u32 ch = (u32)__builtin_amdgcn_readfirstlane((int)((np + nsplit - 1u) / nsplit));
  const u32 off0 = (u32)__builtin_amdgcn_readfirstlane((int)(s * ch));
  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)((off0 < np) ? off0 : np));
  const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)((off0 < np && off0 + ch < np) ? (off0 + ch) : np));

  if (nsplit > 1u && p0 >= p1) {
    const u32 l_e = ink_r16_lane_pin();
    for (u32 qi = w; qi < nq_tot; qi += 8u) {
      const u32 t = qi / INK_R16_GQA;
      const u32 h = h0 + (qi - t * INK_R16_GQA);
      G float *pp = part + (ink_r16_u64)((t * n_head + h) * nsplit + s) * INK_R16_PSTR;
      if (l_e == 0u) { pp[0] = -1e30f; pp[1] = 0.0f; }
      G float *po = pp + 2u + (l_e << 2u);
      po[0] = 0.0f; po[1] = 0.0f; po[2] = 0.0f; po[3] = 0.0f;
    }
  } else {
    /* Strictly prefix keys >= rel_extent behind pos0: zero tree mask, zero rel_bias! */
    const int far_split = (int)__builtin_amdgcn_readfirstlane(
        (int)((p1 <= pos0) && ((pos0 + 1u - p1) >= rel_extent)));
    const u32 q_rows = npg << 4u;

    /* Phase 1: Stage Q (with optional FWHT-128) scaled by QSC into XOR-swizzled lds.kv.q. */
    {
      const u32 l_p1 = ink_r16_lane_pin();
      const u32 c8 = l_p1 >> 1u;
      const u32 sh = l_p1 & 1u;
      for (u32 qr = w; qr < q_rows; qr += 8u) {
        ink_r16_h4 qh4 = (ink_r16_h4)(0.0f);
        if (qr < nq_tot) {
          const u32 tj = qr / INK_R16_GQA;
          const u32 gi = qr - tj * INK_R16_GQA;
          const G float *qt = (const G float *)((const G u8 *)q + (ink_r16_u64)tj * (ink_r16_u64)qs);
          f32x4 qv = ((const G f32x4 *)(qt + (ink_r16_u64)(h0 + gi) * INK_R16_HD))[l_p1];
          if ((rot_flags & 1u) != 0u) qv = ink_r16_fwht128_wave4(qv, l_p1);
          qv *= INK_R16_QSC;
          qh4 = (ink_r16_h4){(f16)qv.x, (f16)qv.y, (f16)qv.z, (f16)qv.w};
        }
        ((LDSP ink_r16_h4 *)(ink_r16_tr_lds.kv.q + ink_r16_tr_q_off(qr, c8)))[sh] = qh4;
      }
      if (!far_split && rel_extent > 0u) {
        const u32 t_id = tid();
        if (t_id < (q_rows << 1u)) {
          const u32 qr = t_id >> 1u;
          const u32 h8i = t_id & 1u;
          h8 rh = (h8)(0.0f);
          if (qr < nq_tot) {
            const u32 tj = qr / INK_R16_GQA;
            const u32 gi = qr - tj * INK_R16_GQA;
            const G f32x4 *rp = (const G f32x4 *)((const G float *)((const G u8 *)R + (ink_r16_u64)tj * (ink_r16_u64)rs) +
                                                  (ink_r16_u64)(h0 + gi) * INK_R16_DREL);
            const f32x4 v0 = rp[(h8i << 1u) + 0u];
            const f32x4 v1 = rp[(h8i << 1u) + 1u];
            rh = (h8){(f16)(v0.x * INK_R16_LOG2E), (f16)(v0.y * INK_R16_LOG2E),
                      (f16)(v0.z * INK_R16_LOG2E), (f16)(v0.w * INK_R16_LOG2E),
                      (f16)(v1.x * INK_R16_LOG2E), (f16)(v1.y * INK_R16_LOG2E),
                      (f16)(v1.z * INK_R16_LOG2E), (f16)(v1.w * INK_R16_LOG2E)};
          }
          *(LDSP h8 *)(ink_r16_tr_lds.kv.r + (qr << 4u) + (h8i << 3u)) = rh;
        }
      }
    }
    lbarrier();

    const G u8 *kc_h = kc_q4 + (ink_r16_u64)kh * INK_R16_RECB;
    const G u8 *vc_h = vc_q4 + (ink_r16_u64)kh * INK_R16_RECB;
    const int no_wrap = (int)__builtin_amdgcn_readfirstlane((int)(p1 <= kv_len));

    const v8f zero = (v8f)(0.0f);
    v8f acc0 = zero, acc1 = zero;
    float m0 = -__builtin_inff(), ls0 = 0.0f;
    float m1 = -__builtin_inff(), ls1 = 0.0f;

    /* Phase 2+3: 128-key chunks (8 waves x 16 keys). */
#pragma clang loop unroll(disable)
    for (u32 c0 = p0; c0 < p1; c0 += 128u) {
      const u32 kb = (u32)__builtin_amdgcn_readfirstlane((int)(c0 + (w << 4u)));
      const u32 chunk_end = (u32)__builtin_amdgcn_readfirstlane((int)((c0 + 128u < p1) ? (c0 + 128u) : p1));
      const int far_chunk = (int)__builtin_amdgcn_readfirstlane(
          (int)(far_split || ((chunk_end <= pos0) && ((pos0 + 1u - chunk_end) >= rel_extent))));

      v8f sp0 = (v8f)(-__builtin_inff());
      v8f sp1 = (v8f)(-__builtin_inff());

      if (kb < chunk_end) {
        const u32 l_k = ink_r16_lane_pin();
        const u32 rl_k = l_k & 15u, hl_k = l_k >> 4u;
        const u32 kp_row = kb + rl_k;
        const int row_ok = (kp_row < chunk_end);
        const G u8 *rec_k = (const G u8 *)0;
        ink_r16_u2 ks_u2 = (ink_r16_u2)(0u);
        u32 wn_a = 0u, wn_b = 0u;
        if (row_ok) {
          const u32 kv_row = no_wrap ? kp_row : (kp_row % kv_len);
          rec_k = kc_h + (ink_r16_u64)kv_row * row_step;
          ks_u2 = *(const G ink_r16_u2 *)(rec_k + 64u);
          wn_a = ((const G u32 *)rec_k)[0u + hl_k];
          wn_b = ((const G u32 *)rec_k)[2u + hl_k];
        }
        const u32 ks_lo = ks_u2.x;
        const u32 ks_hi = ks_u2.y;

        sp0 = zero;
        sp1 = zero;
#pragma unroll
        for (u32 blk = 0u; blk < 4u; blk++) {
          const u32 w_a = wn_a, w_b = wn_b;
          if (blk + 1u < 4u && row_ok) {
            const u32 nb4 = (blk + 1u) << 2u;
            wn_a = ((const G u32 *)rec_k)[nb4 + 0u + hl_k];
            wn_b = ((const G u32 *)rec_k)[nb4 + 2u + hl_k];
          }
          const u32 s_word = (blk < 2u) ? ks_lo : ks_hi;
          const u16 s_bits = (u16)((blk & 1u) ? (s_word >> 16u) : (s_word & 0xFFFFu));
          const float sc = (float)__builtin_bit_cast(f16, s_bits);
          const h8 kk0 = ink_r16_deq8n_h8(w_a, sc);
          const h8 kk1 = ink_r16_deq8n_h8(w_b, sc);

          const u32 c_a = (blk << 2u) + 0u + hl_k;
          const u32 c_b = (blk << 2u) + 2u + hl_k;
          const h8 q0_a = *(const LDSP h8 *)(ink_r16_tr_lds.kv.q + ink_r16_tr_q_off(rl_k, c_a));
          const h8 q0_b = *(const LDSP h8 *)(ink_r16_tr_lds.kv.q + ink_r16_tr_q_off(rl_k, c_b));
          sp0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk0, q0_a, sp0);
          sp0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk1, q0_b, sp0);
          if (npg > 1u) {
            const h8 q1_a = *(const LDSP h8 *)(ink_r16_tr_lds.kv.q + ink_r16_tr_q_off(16u + rl_k, c_a));
            const h8 q1_b = *(const LDSP h8 *)(ink_r16_tr_lds.kv.q + ink_r16_tr_q_off(16u + rl_k, c_b));
            sp1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk0, q1_a, sp1);
            sp1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk1, q1_b, sp1);
          }
          __builtin_amdgcn_sched_barrier(0);
        }

        if (!far_chunk) {
          /* Tree metadata strictly scoped inside Near-Field (0 live VGPRs in Far-Field/PV!). */
          const u32 t_q0 = rl_k / INK_R16_GQA;
          const u32 t_q1 = (16u + rl_k) / INK_R16_GQA;
          const u32 dep_q0 = (t_q0 < T) ? ((pos_off_packed >> (t_q0 << 2u)) & 0xFu) : 0u;
          const u32 dep_q1 = (t_q1 < T) ? ((pos_off_packed >> (t_q1 << 2u)) & 0xFu) : 0u;
          const u32 rmask_q0 = (t_q0 < T) ? (u32)((tree_mask >> (t_q0 << 3u)) & 0xFFull) : 0u;
          const u32 rmask_q1 = (t_q1 < T) ? (u32)((tree_mask >> (t_q1 << 3u)) & 0xFFull) : 0u;

          const int d_min_s = (int)pos0 - (int)(kb + 15u);
          const u32 d_base = d_min_s > 0 ? (u32)d_min_s : 0u;
          if (d_base < rel_extent) {
            LDSP float *sc_row = ((LDSP float *)ink_r16_tr_lds.kv.scratch.v_sc[w]) + (rl_k << 3u);
            const h8 ra0 = *(const LDSP h8 *)(ink_r16_tr_lds.kv.r + (rl_k << 4u) + (hl_k << 3u));
            const u32 d0 = d_base + rl_k;
            h8 b0 = (h8)(0.0f);
            if (d0 < rel_extent) {
              const G f32x4 *rp = (const G f32x4 *)(rel_bias + (ink_r16_u64)d0 * 16ull + (ink_r16_u64)(hl_k << 3u));
              const f32x4 v0 = rp[0], v1 = rp[1];
              b0 = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
            }
            const v8f rb0_0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0, ra0, zero);
            ink_r16_tr_apply_rb16(rb0_0, 0u, hl_k, kb, pos0, T, dep_q0, rmask_q0,
                                  pos_off_packed, rel_extent, d_base, sc_row, &sp0);
            if (npg > 1u) {
              const h8 ra1 = *(const LDSP h8 *)(ink_r16_tr_lds.kv.r + ((16u + rl_k) << 4u) + (hl_k << 3u));
              const v8f rb0_1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0, ra1, zero);
              ink_r16_tr_apply_rb16(rb0_1, 0u, hl_k, kb, pos0, T, dep_q1, rmask_q1,
                                    pos_off_packed, rel_extent, d_base, sc_row, &sp1);
            }
            if (d_base + 16u < rel_extent && (pos0 + T - 1u) >= kb + d_base + 16u) {
              const u32 d1 = d_base + 16u + rl_k;
              h8 b1 = (h8)(0.0f);
              if (d1 < rel_extent) {
                const G f32x4 *rp = (const G f32x4 *)(rel_bias + (ink_r16_u64)d1 * 16ull + (ink_r16_u64)(hl_k << 3u));
                const f32x4 v0 = rp[0], v1 = rp[1];
                b1 = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
              }
              const v8f rb1_0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1, ra0, zero);
              ink_r16_tr_apply_rb16(rb1_0, 16u, hl_k, kb, pos0, T, dep_q0, rmask_q0,
                                    pos_off_packed, rel_extent, d_base, sc_row, &sp0);
              if (npg > 1u) {
                const h8 ra1 = *(const LDSP h8 *)(ink_r16_tr_lds.kv.r + ((16u + rl_k) << 4u) + (hl_k << 3u));
                const v8f rb1_1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1, ra1, zero);
                ink_r16_tr_apply_rb16(rb1_1, 16u, hl_k, kb, pos0, T, dep_q1, rmask_q1,
                                      pos_off_packed, rel_extent, d_base, sc_row, &sp1);
              }
            }
          }
          /* Tree ancestor mask + split-end mask strictly inside Near-Field! */
#pragma clang loop unroll(disable)
          for (u32 v = 0u; v < 8u; v++) {
            const u32 kp = kb + (hl_k << 3u) + v;
            if (kp >= chunk_end) {
              sp0[v] = -__builtin_inff();
              if (npg > 1u) sp1[v] = -__builtin_inff();
            } else if (kp >= pos0) {
              const u32 j = kp - pos0;
              if (j >= T || ((rmask_q0 >> j) & 1u) == 0u) sp0[v] = -__builtin_inff();
              if (npg > 1u && (j >= T || ((rmask_q1 >> j) & 1u) == 0u)) sp1[v] = -__builtin_inff();
            }
          }
        } else if (kb + 16u > chunk_end) {
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) {
            if (kb + (hl_k << 3u) + v >= chunk_end) {
              sp0[v] = -__builtin_inff();
              if (npg > 1u) sp1[v] = -__builtin_inff();
            }
          }
        }
      }

      /* Prefetch raw Q4_0 V tile 0 (2 VGPRs) ahead of softmax barriers. */
      const u32 l_s = ink_r16_lane_pin();
      const u32 rl_s = l_s & 15u, hl_s = l_s >> 4u;
      const u32 v_word = (w << 1u) + hl_s;
      const u32 v_blk = w >> 1u;
      u32 vq0 = 0u;
      u16 vs0 = 0u;
      {
        const u32 vp = c0 + rl_s;
        if (vp < chunk_end) {
          const u32 kv_v = no_wrap ? vp : (vp % kv_len);
          const G u8 *rv = vc_h + (ink_r16_u64)kv_v * row_step;
          vq0 = ((const G u32 *)rv)[v_word];
          vs0 = ((const G u16 *)(rv + 64u))[v_blk];
        }
      }

      if (!far_chunk) lbarrier();

      /* 2-barrier online softmax with conditional mn == m rescale skip. */
      float mx0 = sp0[0];
#pragma unroll
      for (u32 v = 1u; v < 8u; v++) mx0 = __builtin_fmaxf(mx0, sp0[v]);
      mx0 = __builtin_fmaxf(mx0, ink_r16_xrow(mx0));
      if (hl_s == 0u) ink_r16_tr_lds.kv.scratch.red.mx[0][(w << 4u) + rl_s] = mx0;

      if (npg > 1u) {
        float mx1 = sp1[0];
#pragma unroll
        for (u32 v = 1u; v < 8u; v++) mx1 = __builtin_fmaxf(mx1, sp1[v]);
        mx1 = __builtin_fmaxf(mx1, ink_r16_xrow(mx1));
        if (hl_s == 0u) ink_r16_tr_lds.kv.scratch.red.mx[1][(w << 4u) + rl_s] = mx1;
      }
      lbarrier();

      float mc0 = ink_r16_tr_lds.kv.scratch.red.mx[0][rl_s];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) mc0 = __builtin_fmaxf(mc0, ink_r16_tr_lds.kv.scratch.red.mx[0][(x << 4u) + rl_s]);
      const float mn0 = __builtin_fmaxf(m0, mc0);
      if (mn0 != m0) {
        const float rs0 = (ls0 > 0.0f && mn0 > -1e30f) ? __builtin_amdgcn_exp2f(m0 - mn0) : 0.0f;
        acc0 *= rs0;
        ls0 *= rs0;
        m0 = mn0;
      }

      h8 ph0 = (h8)(0.0f);
      float lc0 = 0.0f;
      if (mn0 > -1e30f) {
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) {
          const float p = __builtin_amdgcn_exp2f(sp0[v] - mn0);
          lc0 += p;
          ph0[v] = (f16)p;
        }
      }
      lc0 += ink_r16_xrow(lc0);
      if (hl_s == 0u) ink_r16_tr_lds.kv.scratch.red.ls[0][(w << 4u) + rl_s] = lc0;

      h8 ph1 = (h8)(0.0f);
      if (npg > 1u) {
        float mc1 = ink_r16_tr_lds.kv.scratch.red.mx[1][rl_s];
#pragma unroll
        for (u32 x = 1u; x < 8u; x++) mc1 = __builtin_fmaxf(mc1, ink_r16_tr_lds.kv.scratch.red.mx[1][(x << 4u) + rl_s]);
        const float mn1 = __builtin_fmaxf(m1, mc1);
        if (mn1 != m1) {
          const float rs1 = (ls1 > 0.0f && mn1 > -1e30f) ? __builtin_amdgcn_exp2f(m1 - mn1) : 0.0f;
          acc1 *= rs1;
          ls1 *= rs1;
          m1 = mn1;
        }
        float lc1 = 0.0f;
        if (mn1 > -1e30f) {
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) {
            const float p = __builtin_amdgcn_exp2f(sp1[v] - mn1);
            lc1 += p;
            ph1[v] = (f16)p;
          }
        }
        lc1 += ink_r16_xrow(lc1);
        if (hl_s == 0u) ink_r16_tr_lds.kv.scratch.red.ls[1][(w << 4u) + rl_s] = lc1;
      }

      const u32 p_off = ink_r16_tr_p_off(rl_s, hl_s);
      if (w == 0u) {
        *(LDSP h8 *)(ink_r16_tr_lds.kv.p_tile[0][0] + p_off) = ph0;
        if (npg > 1u) *(LDSP h8 *)(ink_r16_tr_lds.kv.p_tile[0][1] + p_off) = ph1;
      }
      lbarrier();

      float sum_l0 = ink_r16_tr_lds.kv.scratch.red.ls[0][rl_s];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) sum_l0 += ink_r16_tr_lds.kv.scratch.red.ls[0][(x << 4u) + rl_s];
      ls0 += sum_l0;
      if (npg > 1u) {
        float sum_l1 = ink_r16_tr_lds.kv.scratch.red.ls[1][rl_s];
#pragma unroll
        for (u32 x = 1u; x < 8u; x++) sum_l1 += ink_r16_tr_lds.kv.scratch.red.ls[1][(x << 4u) + rl_s];
        ls1 += sum_l1;
      }
      lbarrier();

      /* Phase 3: Double-buffered p_tile + raw Q4_0 V prefetch + 0-conflict V^T + WMMA PV. */
      LDSP u32 *v_sc_w = ink_r16_tr_lds.kv.scratch.v_sc[w];
#pragma clang loop unroll(disable)
      for (u32 kt = 0u; kt < 8u; kt++) {
        if (c0 + (kt << 4u) >= chunk_end) break;
        const int has_nxt = (kt + 1u < 8u) && (c0 + ((kt + 1u) << 4u) < chunk_end);
        u32 vq_n = 0u;
        u16 vs_n = 0u;
        if (has_nxt) {
          const u32 vp_n = c0 + ((kt + 1u) << 4u) + rl_s;
          if (vp_n < chunk_end) {
            const u32 kv_vn = no_wrap ? vp_n : (vp_n % kv_len);
            const G u8 *rv_n = vc_h + (ink_r16_u64)kv_vn * row_step;
            vq_n = ((const G u32 *)rv_n)[v_word];
            vs_n = ((const G u16 *)(rv_n + 64u))[v_blk];
          }
        }
        __builtin_amdgcn_sched_barrier(0);

        const h8 vh = ink_r16_deq8n_h8(vq0, (float)__builtin_bit_cast(f16, vs0));
        const h8 vt_a = ink_r16_tr_v_transpose(vh, v_sc_w, hl_s, rl_s);
        const h8 pt0 = *(const LDSP h8 *)(ink_r16_tr_lds.kv.p_tile[kt & 1u][0] + p_off);
        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt0, acc0);
        if (npg > 1u) {
          const h8 pt1 = *(const LDSP h8 *)(ink_r16_tr_lds.kv.p_tile[kt & 1u][1] + p_off);
          acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt1, acc1);
        }

        if (has_nxt && w == kt + 1u) {
          *(LDSP h8 *)(ink_r16_tr_lds.kv.p_tile[(kt + 1u) & 1u][0] + p_off) = ph0;
          if (npg > 1u) *(LDSP h8 *)(ink_r16_tr_lds.kv.p_tile[(kt + 1u) & 1u][1] + p_off) = ph1;
        }
        lbarrier();
        vq0 = vq_n;
        vs0 = vs_n;
      }
    }

    /* Phase 4A: Stage (m, ls) into ph4.m/l, then coalesce 8 queries at a time via o_stg[8][128]. */
    lbarrier();
    {
      const u32 l_4 = ink_r16_lane_pin();
      const u32 rl_4 = l_4 & 15u, hl_4 = l_4 >> 4u;
      if (w == 0u && hl_4 == 0u) {
        ink_r16_tr_lds.ph4.m[rl_4] = (ls0 > 0.0f) ? (m0 * INK_R16_LN2) : -1e30f;
        ink_r16_tr_lds.ph4.l[rl_4] = ls0;
        if (npg > 1u) {
          ink_r16_tr_lds.ph4.m[16u + rl_4] = (ls1 > 0.0f) ? (m1 * INK_R16_LN2) : -1e30f;
          ink_r16_tr_lds.ph4.l[16u + rl_4] = ls1;
        }
      }
    }
    lbarrier();

    const u32 l_out = ink_r16_lane_pin();
    const u32 rl_o = l_out & 15u, hl_o = l_out >> 4u;
#pragma clang loop unroll(disable)
    for (u32 pg = 0u; pg < npg; pg++) {
      const v8f a_cur = (pg == 0u) ? acc0 : acc1;
#pragma unroll
      for (u32 qb = 0u; qb < 2u; qb++) {
        const u32 q_base = (pg << 4u) + (qb << 3u);
        if (q_base >= nq_tot) break;
        if ((rl_o >> 3u) == qb) {
          const u32 q_loc = rl_o & 7u;
          LDSP float *dst_w = ink_r16_tr_lds.ph4.o_stg[q_loc] + (w << 4u) + (hl_o << 3u);
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) dst_w[v] = a_cur[v];
        }
        lbarrier();

        const u32 qi = q_base + w;
        if (qi < nq_tot) {
          const u32 t = qi / INK_R16_GQA;
          const u32 h = h0 + (qi - t * INK_R16_GQA);
          f32x4 ov = ((const LDSP f32x4 *)ink_r16_tr_lds.ph4.o_stg[w])[l_out];
          if (nsplit == 1u) {
            const G float *st = (const G float *)((const G u8 *)sink + (ink_r16_u64)t * (ink_r16_u64)ss);
            const float sv = st[h];
            const float mq = ink_r16_tr_lds.ph4.m[qi];
            const float lq = ink_r16_tr_lds.ph4.l[qi];
            const float M = (lq > 0.0f) ? __builtin_fmaxf(sv, mq) : sv;
            const float wx = (lq > 0.0f) ? __builtin_expf(mq - M) : 0.0f;
            const float L = __builtin_expf(sv - M) + lq * wx;
            ov *= (wx / L);
            if ((rot_flags & 2u) != 0u) ov = ink_r16_fwht128_wave4(ov, l_out);
            const G float *gt_h = agate ? ((const G float *)((const G u8 *)agate + (ink_r16_u64)t * (ink_r16_u64)gs) + (ink_r16_u64)h * INK_R16_HD) : (const G float *)0;
            G float *ot_h = out ? ((G float *)((G u8 *)out + (ink_r16_u64)t * (ink_r16_u64)os) + (ink_r16_u64)h * INK_R16_HD) : (G float *)0;
            G u8 *oqt = oq ? (oq + (ink_r16_u64)t * (ink_r16_u64)oqs) : (G u8 *)0;
            ink_r16_gate_q8_head_vec4(ov, gt_h, ot_h, oqt, h, n_head, l_out);
          } else {
            G float *pp = part + (ink_r16_u64)((t * n_head + h) * nsplit + s) * INK_R16_PSTR;
            if (l_out == 0u) {
              pp[0] = ink_r16_tr_lds.ph4.m[qi];
              pp[1] = ink_r16_tr_lds.ph4.l[qi];
            }
            G float *po = pp + 2u + (l_out << 2u);
            po[0] = ov.x; po[1] = ov.y; po[2] = ov.z; po[3] = ov.w;
          }
        }
        lbarrier();
      }
    }

    if (nsplit == 1u) return;
  }

  /* Phase 4B: Single-atomic arrival per KV head (count[kh]) + parallel 8-wave split-K merge. */
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r16_tr_lds.ph4.is_last =
        (__scoped_atomic_fetch_add(&count[kh], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r16_tr_lds.ph4.is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const u32 l_m = ink_r16_lane_pin();
  for (u32 qi = w; qi < nq_tot; qi += 8u) {
    const u32 t = qi / INK_R16_GQA;
    const u32 h = h0 + (qi - t * INK_R16_GQA);
    const G float *st = (const G float *)((const G u8 *)sink + (ink_r16_u64)t * (ink_r16_u64)ss);
    const G float *gt_h = agate ? ((const G float *)((const G u8 *)agate + (ink_r16_u64)t * (ink_r16_u64)gs) + (ink_r16_u64)h * INK_R16_HD) : (const G float *)0;
    G float *ot_h = out ? ((G float *)((G u8 *)out + (ink_r16_u64)t * (ink_r16_u64)os) + (ink_r16_u64)h * INK_R16_HD) : (G float *)0;
    G u8 *oqt = oq ? (oq + (ink_r16_u64)t * (ink_r16_u64)oqs) : (G u8 *)0;
    const G float *ph = part + (ink_r16_u64)((t * n_head + h) * nsplit) * INK_R16_PSTR;
    ink_r16_tr_merge_head(ph, st[h], nsplit, l_m, ink_r16_tr_lds.ph4.wt[w], gt_h, ot_h, oqt, h, n_head, rot_flags);
  }
  barrier();
  if (tid() == 0u) count[kh] = 0u;
}

/* ============================================================================
 * 2. ie_inkling_c1_sconv_vclip_glob_q4_store_r16 (R65 C1-GLOB-Q4, bit 163)
 *    Tree-aware 1-launch K/V Short-Conv + VClip + Inline FWHT-128 + 72-B Q4_0
 *    Global KV Store (+ inline sc_hist when hist!=NULL OR ring commit when NULL).
 *    Zero LDS, zero barriers, 64-bit KV offsets safe to 512K+.
 *    Linear chain T<=8: pass tpar_pack = 0x65432100u.
 *    Grid: (1, 1, 1), Block: (256, 1, 1) (8 waves = 8 KV heads), <= 64 VGPR.
 * ========================================================================== */

static inline __attribute__((always_inline)) f32x4 ink_r16_c1_load_tap(
    const G u8 *src_a, u32 a_s, const G float *ring_p, u32 i4, u32 i0, u32 pos0, int idx) {
  if (idx >= 0) {
    return ((const G f32x4 *)(src_a + (ink_r16_u64)(u32)idx * (ink_r16_u64)a_s))[i4];
  }
  const u32 d = (u32)(-idx);
  if (pos0 < d) return (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  const u32 sl = (pos0 - d) & 3u;
  return (f32x4){ring_p[(i0 + 0u) * 4u + sl], ring_p[(i0 + 1u) * 4u + sl],
                 ring_p[(i0 + 2u) * 4u + sl], ring_p[(i0 + 3u) * 4u + sl]};
}

static inline __attribute__((always_inline)) f32x4 ink_r16_c1_load_w_tap(
    const G float *v_w, u32 i0, u32 tap_col, float c_bias) {
#pragma clang fp contract(off)
  return (f32x4){v_w[(i0 + 0u) * 4u + tap_col] + c_bias,
                 v_w[(i0 + 1u) * 4u + tap_col] + c_bias,
                 v_w[(i0 + 2u) * 4u + tap_col] + c_bias,
                 v_w[(i0 + 3u) * 4u + tap_col] + c_bias};
}

KERNEL ie_inkling_c1_sconv_vclip_glob_q4_store_r16(
    const G float *ka, const G float *vk, const G float *ck, G float *ring_k, G float *hist_k,
    const G float *va, const G float *vv, const G float *cv, G float *ring_v, G float *hist_v,
    G u8 *kc_q4, G u8 *vc_q4, float sc_escale, float k_clamp, float v_eps,
    u32 pos0, u32 T, u32 kv_len,
    u32 kas, u32 cks, u32 vas, u32 cvs, u32 hrs, u32 tpar_pack, u32 rot_flags) {
  if (T == 0u || T > 8u || kv_len == 0u || !ka || !vk || !ring_k ||
      !va || !vv || !ring_v || !kc_q4 || !vc_q4)
    return;

  const u32 i4 = tid();
  const u32 i0 = i4 << 2u;
  const u32 h = (u32)__builtin_amdgcn_readfirstlane((int)(i4 >> 5u));
  const u32 l = lane();
  const u32 p0 = pos0 % kv_len;

  /* Unified 2-pass loop (kv=0: K + L2 vclip + kc_q4, kv=1: V + RMSNorm vclip + vc_q4).
   * Tap-by-tap exact non-FMA shortconv + serialized Q4 divider keeps <= 64 VGPR (4 WGs/CU). */
#pragma clang loop unroll(disable)
  for (u32 kv = 0u; kv < 2u; kv++) {
    const G u8 *src_a = (const G u8 *)(kv == 0u ? ka : va);
    const G float *v_w = (kv == 0u) ? vk : vv;
    const G u8 *c_p = (const G u8 *)(kv == 0u ? ck : cv);
    G float *ring_p = (kv == 0u) ? ring_k : ring_v;
    G float *hist_p = (kv == 0u) ? hist_k : hist_v;
    G u8 *dst_q4 = (kv == 0u) ? kc_q4 : vc_q4;
    const u32 a_s = (kv == 0u) ? kas : vas;
    const u32 c_s = (kv == 0u) ? cks : cvs;
    const u32 do_rot = (rot_flags >> kv) & 1u;

    u32 p = p0;
#pragma clang loop unroll(disable)
    for (u32 t = 0u; t < T; t++) {
      /* Wave-uniform scalar ancestor indices (idx >= 0 -> src_a[idx], idx < 0 -> ring[pos0+idx]). */
      int i1, i2, i3;
      if (t == 0u) {
        i1 = -1; i2 = -2; i3 = -3;
      } else {
        i1 = (int)((tpar_pack >> (t << 2u)) & 0xFu);
        if (i1 == 0) {
          i2 = -1; i3 = -2;
        } else {
          i2 = (int)((tpar_pack >> ((u32)i1 << 2u)) & 0xFu);
          if (i2 == 0) {
            i3 = -1;
          } else {
            i3 = (int)((tpar_pack >> ((u32)i2 << 2u)) & 0xFu);
          }
        }
      }

      const f32x4 x0 = ((const G f32x4 *)(src_a + (ink_r16_u64)t * (ink_r16_u64)a_s))[i4];
      if (hist_p) ((G f32x4 *)(hist_p + (ink_r16_u64)t * (ink_r16_u64)hrs))[i4] = x0;

      const f32x4 ct = c_p ? ((const G f32x4 *)(c_p + (ink_r16_u64)t * (ink_r16_u64)c_s))[h]
                           : (f32x4){0.0f, 0.0f, 0.0f, 0.0f};

      /* Exact non-FMA left-to-right 4-tap shortconv matching ie_inkling_sconv_r1 / r12. */
      f32x4 y;
      {
#pragma clang fp contract(off)
        const f32x4 w0 = ink_r16_c1_load_w_tap(v_w, i0, 3u, ct.x);
        y = x0 * w0;
        const f32x4 x1 = ink_r16_c1_load_tap(src_a, a_s, ring_p, i4, i0, pos0, i1);
        const f32x4 w1 = ink_r16_c1_load_w_tap(v_w, i0, 2u, ct.y);
        y = y + x1 * w1;
        const f32x4 x2 = ink_r16_c1_load_tap(src_a, a_s, ring_p, i4, i0, pos0, i2);
        const f32x4 w2 = ink_r16_c1_load_w_tap(v_w, i0, 1u, ct.z);
        y = y + x2 * w2;
        const f32x4 x3 = ink_r16_c1_load_tap(src_a, a_s, ring_p, i4, i0, pos0, i3);
        const f32x4 w3 = ink_r16_c1_load_w_tap(v_w, i0, 0u, ct.w);
        y = y + x3 * w3;
        y = (x0 + y) * sc_escale;
      }

      if (kv == 0u) y = ink_r16_vclip_k(y, k_clamp);
      else          y = ink_r16_vclip_v(y, v_eps);

      if (do_rot != 0u) y = ink_r16_fwht128_wave4(y, l);

      G u8 *dst = dst_q4 + ((ink_r16_u64)p * 8ull + (ink_r16_u64)h) * INK_R16_RECB;
      ink_r16_q4_store_wave4_lean(y, dst, l);

      p++;
      if (p == kv_len) p = 0u;
    }

    if (!hist_p) {
      const u32 n_com = (T >= 4u) ? 4u : T;
#pragma clang loop unroll(disable)
      for (u32 k = 0u; k < n_com; k++) {
        const u32 tk = T - n_com + k;
        const u32 sl = (pos0 + tk) & 3u;
        const f32x4 xv = ((const G f32x4 *)(src_a + (ink_r16_u64)tk * (ink_r16_u64)a_s))[i4];
        ring_p[(i0 + 0u) * 4u + sl] = xv.x;
        ring_p[(i0 + 1u) * 4u + sl] = xv.y;
        ring_p[(i0 + 2u) * 4u + sl] = xv.z;
        ring_p[(i0 + 3u) * 4u + sl] = xv.w;
      }
    }
  }
}

/* ============================================================================
 * 3. ie_inkling_kv_store_glob_q4_rot_pf_r16 (bit 165)
 *    Batched prefill (T=1..1024+) zero-LDS, zero-barrier wave-per-(tok,head)
 *    optional vclip(k/v) + FWHT-128 + 72-B Q4_0 Global KV store (<0.2 ms for
 *    1024 tokens x 8 global layers!).
 *    k_clamp > 0.0f enables inline L2 vclip(k); v_eps > 0.0f enables inline
 *    RMSNorm vclip(v). Pass 0.0f, 0.0f for drop-in post-vclip OP_KV store!
 *    Grid: (((T * n_kv) + 7u) / 8u, 1, 1) [=(T, 1, 1) at n_kv=8], Block: (256, 1, 1).
 * ========================================================================== */

KERNEL ie_inkling_kv_store_glob_q4_rot_pf_r16(
    const G float *k, const G float *v, G u8 *kc_q4, G u8 *vc_q4,
    u32 pos0, u32 T, u32 kv_len, u32 n_kv, u32 hd, u32 ks, u32 vs,
    float k_clamp, float v_eps, u32 rot_flags) {
  if (T == 0u || n_kv == 0u || hd != INK_R16_HD || kv_len == 0u || !k || !v || !kc_q4 || !vc_q4)
    return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 pair = (u32)__builtin_amdgcn_readfirstlane((int)((wgid() << 3u) + w));
  const u32 total = T * n_kv;
  if (pair >= total) return;

  const u32 t = (n_kv == 8u) ? (pair >> 3u) : (pair / n_kv);
  const u32 kh = (n_kv == 8u) ? (pair & 7u) : (pair - t * n_kv);
  const u32 l = lane();
  const u32 kvp = (pos0 + t) % kv_len;
  const ink_r16_u64 dst_off = ((ink_r16_u64)kvp * (ink_r16_u64)n_kv + (ink_r16_u64)kh) * INK_R16_RECB;

  /* K head: vectorized f32x4 load -> optional L2 vclip -> optional FWHT-128 -> 72-B Q4_0 store. */
  const G float *kt = (const G float *)((const G u8 *)k + (ink_r16_u64)t * (ink_r16_u64)ks);
  f32x4 kx = ((const G f32x4 *)(kt + (ink_r16_u64)kh * INK_R16_HD))[l];
  if (k_clamp > 0.0f) kx = ink_r16_vclip_k(kx, k_clamp);
  if ((rot_flags & 1u) != 0u) kx = ink_r16_fwht128_wave4(kx, l);
  ink_r16_q4_store_wave4(kx, kc_q4 + dst_off, l);

  /* V head: vectorized f32x4 load -> optional RMSNorm vclip -> optional FWHT-128 -> 72-B Q4_0 store. */
  const G float *vt = (const G float *)((const G u8 *)v + (ink_r16_u64)t * (ink_r16_u64)vs);
  f32x4 vx = ((const G f32x4 *)(vt + (ink_r16_u64)kh * INK_R16_HD))[l];
  if (v_eps > 0.0f) vx = ink_r16_vclip_v(vx, v_eps);
  if ((rot_flags & 2u) != 0u) vx = ink_r16_fwht128_wave4(vx, l);
  ink_r16_q4_store_wave4(vx, vc_q4 + dst_off, l);
}

#undef INK_R16_QSC
#undef INK_R16_INV_SQRT
#undef INK_R16_LN2
#undef INK_R16_LOG2E
#undef INK_R16_PSTR
#undef INK_R16_RECB
#undef INK_R16_GQA
#undef INK_R16_DREL
#undef INK_R16_HD

#endif /* !IE_MEGA */
