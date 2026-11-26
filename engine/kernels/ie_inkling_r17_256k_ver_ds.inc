/* ============================================================================
 * Inkling-2-Mini Round 17 (256K Verify + DSpark H25a Fix Suite):
 *   1. ie_inkling_ds_qkv_rope_q4k_r17
 *        Fixes R66 H25a Q/K RMSNorm bug in ie_inkling_ds_qkv_rope_q4k_r12
 *        (r12:622 called wave_tree_f() instead of wave_sum_all(), leaving
 *        lanes 1..15 and 17..31 with partial sums -> up to 4x distorted inv
 *        across 120/128 head dims of Q & K, dropping acceptance 95.7% -> 18.2%).
 *        Uses exact 4-wave-group wave_sum_all() matching wg_sum() in
 *        ie_qkn_rope_kv bit-for-bit, plus LDS-staged xq (<= 80 blocks) and
 *        1-SB weight prefetch, 0 TH_LOAD_NT/HT.
 *        Exact 24-arg drop-in ABI of ie_inkling_ds_qkv_rope_q4k_r12.
 *        Grid: (nh + 2*nkv, (B + 3)/4, 1), NT=256, 14,848 B LDS, <= 80 VGPR.
 *
 *   2. ie_inkling_flute_situ_ver_t8_r17
 *        Multi-token T=2..8 (and T<=16) FLUTE SoA IQ3_XXS Gate/Up + SITU + Q8_0
 *        verify kernel for -Draft 7 (128E core layers AND H38 Mode B remapped
 *        slots). Replaces r10's TH_LOAD_NT/HT with default cached loads, adds
 *        1-pass interleaved Gate+Up for nj==1 (0 LDS X copy) and nj==2 (1x LDS
 *        X read, 9 vs 18 iters), and cross-subpass prefetched wave-split for
 *        nj>=3 (handles nj=1..15 in <= 4-token chunks with <= 16 acc VGPRs).
 *        Exact 16-arg drop-in ABI of ie_inkling_gexp_iq3xxs_flute_situ_compact_r7.
 *        Grid: (eff / 32, max_packs, 1), NT=256, 13,568 B LDS, <= 88 VGPR.
 *
 *   3. ie_inkling_rel_attn_ver_glob_q4_wmma_fast_r17
 *        Next-gen 256K Global Q4_0 + FWHT-128 Verify & T=1 FP16 WMMA attention
 *        + optional fused OP_GATE(sigmoid) + Q8_0.
 *        4-wave async Producer (w=4..7, 128 threads, 2x MLP) + 4-wave Consumer
 *        (w=0..3, pg=w>>1, d_half=w&1 -> only 32 acc VGPRs + 32 Q VGPRs!),
 *        exact 16,384 B LDS (4 WGs/CU), gfx1201 decoupled split barriers
 *        (s_barrier_signal/wait), dedicated zero-rb NB=32 far-field loop
 *        (99.6% of 256K keys), exact q_valid-gated ballot_w32 conditional
 *        rescale skip + zero-tile PV skip (bm - m < -30.0f), and double-buffered
 *        NB=16 near-field tail with exact L1-cached f32 R*rel_bias.
 *        Exact 28-arg drop-in ABI of ie_inkling_rel_attn_ver_glob_q4_wmma_r15.
 *        Grid: (n_kv * nsplit, 1, 1), NT=256, 16,384 B LDS, <= 96 VGPR target.
 *
 * Target: RX 9070 XT (gfx1201, RDNA4, wave32, NT=256, 64 CUs, 64 MiB MALL)
 * Hygiene:
 *   - Wrapped in #ifndef IE_MEGA; all local symbols/macros prefixed
 *     ink_r17_vd_ / INK_R17_VD_; all macros #undef'd at EOF.
 *   - 0 TH_LOAD_NT, 0 TH_LOAD_HT (default cached loads only).
 *   - Scalar __builtin_bit_cast only (no direct ext_vector lane casts).
 *   - 64-bit KV byte offsets, slack=0 bounds-checked safe to 524K+.
 *   - 0/0 VGPR/SGPR spills target, priv=0, scratch=0, f16_mix=0.
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(f16) == 2u, "ink_r17_vd f16 must be 2 bytes");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r17_vd float/u32 must be 4 bytes");

typedef unsigned long long ink_r17_vd_u64;
typedef unsigned int       ink_r17_vd_u2 __attribute__((ext_vector_type(2)));
typedef float              ink_r17_vd_f4 __attribute__((ext_vector_type(4)));
typedef f16                ink_r17_vd_h4 __attribute__((ext_vector_type(4)));
typedef f16                ink_r17_vd_h8 __attribute__((ext_vector_type(8)));
typedef float              ink_r17_vd_v8f __attribute__((ext_vector_type(8)));

_Static_assert(sizeof(ink_r17_vd_u64) == 8u, "ink_r17_vd_u64 must be 8 bytes");

struct __attribute__((packed, aligned(4))) ink_r17_vd_u32x4p {
  u32x4 v;
};

struct __attribute__((packed, aligned(4))) ink_r17_vd_u32x2p {
  ink_r17_vd_u2 v;
};

#define INK_R17_VD_HD        128u
#define INK_R17_VD_DREL      16u
#define INK_R17_VD_GQA       3u
#define INK_R17_VD_RECB      72u
#define INK_R17_VD_PSTR      130u
#define INK_R17_VD_NB16      16u
#define INK_R17_VD_NB32      32u
#define INK_R17_VD_DS_MAXNB  80u
#define INK_R17_VD_FL_MAXNB  80u
#define INK_R17_VD_LOG2E     1.44269504088896340736f
#define INK_R17_VD_LN2       0.69314718055994530942f
#define INK_R17_VD_INV_SQRT  0.08838834764831844299f /* 1 / sqrt(128) */
#define INK_R17_VD_QSC       (0.08838834764831844299f * 1.44269504088896340736f)

#define INK_R17_VD_ADD4(a0_, a1_, a2_, a3_, j_, v_) \
  do {                                              \
    if ((j_) == 0u) (a0_) += (v_);                  \
    else if ((j_) == 1u) (a1_) += (v_);             \
    else if ((j_) == 2u) (a2_) += (v_);             \
    else (a3_) += (v_);                             \
  } while (0)

/* ============================================================================
 * Shared Primitives (Cached Loads, Split Barriers, Pin Lane, FWHT-128, Q8)
 * ========================================================================== */

static inline __attribute__((always_inline)) u32x4 ink_r17_vd_ld4(const G void *p) {
  return ((const G struct ink_r17_vd_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) ink_r17_vd_u2 ink_r17_vd_ld2(const G void *p) {
  return ((const G struct ink_r17_vd_u32x2p *)p)->v;
}

static inline __attribute__((always_inline)) u32 ink_r17_vd_lane_pin(void) {
  u32 l;
  __asm__ volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(l));
  return l;
}

static inline __attribute__((always_inline)) float ink_r17_vd_sgpr_f(float v) {
  return I2F(__builtin_amdgcn_readfirstlane(F2I(v)));
}

static inline __attribute__((always_inline)) float ink_r17_vd_xrow(float v) {
  return I2F(xrow_i(F2I(v)));
}

static inline __attribute__((always_inline)) float ink_r17_vd_oct_tree_f(float v) {
#pragma clang fp contract(off)
  v = v + I2F(shl_i(F2I(v), 4));
  v = v + I2F(shl_i(F2I(v), 2));
  v = v + I2F(shl_i(F2I(v), 1));
  return v;
}

static inline __attribute__((always_inline)) void ink_r17_vd_lsignal(void) {
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  __builtin_amdgcn_s_barrier_signal(-1);
  __asm__ volatile("" ::: "memory");
}

static inline __attribute__((always_inline)) void ink_r17_vd_lwait(void) {
  __builtin_amdgcn_s_barrier_wait(-1);
  __asm__ volatile("" ::: "memory");
}

static inline __attribute__((always_inline)) u32 ink_r17_vd_div_x(u32 pid, u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

static inline __attribute__((always_inline)) float ink_r17_vd_gate_f(float o, float g) {
#pragma clang fp contract(off)
  return o * (1.0f / (1.0f + __builtin_expf(-g)));
}

static inline __attribute__((always_inline)) float ink_r17_vd_sub8_max_pos(float v) {
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0xB1, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0x4E, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_ds_swizzle(F2I(v), 0x101F)));
  return v;
}

static inline __attribute__((always_inline)) u32 ink_r17_vd_sub8_sum_u32(u32 s) {
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0xB1, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0x4E, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_ds_swizzle((int)s, 0x101F);
  return s;
}

/* Zero-LDS, zero-barrier exact normalized 128-pt FWHT on 1 wave32 (lane l -> 4*l..4*l+3). */
static inline __attribute__((always_inline)) f32x4 ink_r17_vd_fwht128_wave4(f32x4 v, u32 l) {
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
  return r * INK_R17_VD_INV_SQRT;
}

/* Zero-LDS in-register f32x4 OP_GATE(sigmoid) + Q8_0 epilogue for 1 head (hd=128=4x32). */
static inline __attribute__((always_inline)) void ink_r17_vd_gate_q8_head_vec4(
    f32x4 ov, const G float *gt_h, G float *ot_h, G u8 *oqt, u32 h, u32 n_head, u32 l) {
#pragma clang fp contract(off)
  f32x4 vv = ov;
  if (gt_h) {
    const f32x4 gv = ((const G f32x4 *)gt_h)[l];
    vv = (f32x4){ink_r17_vd_gate_f(ov.x, gv.x), ink_r17_vd_gate_f(ov.y, gv.y),
                 ink_r17_vd_gate_f(ov.z, gv.z), ink_r17_vd_gate_f(ov.w, gv.w)};
  }
  if (ot_h) ((G f32x4 *)ot_h)[l] = vv;
  if (oqt) {
    const u32 sg = l >> 3u;
    const u32 sl = l & 7u;
    const u32 blk = (h << 2u) + sg;
    const u32 nb_q = n_head << 2u;
    float am = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(vv.x), __builtin_fabsf(vv.y)),
                               __builtin_fmaxf(__builtin_fabsf(vv.z), __builtin_fabsf(vv.w)));
    am = ink_r17_vd_sub8_max_pos(am);
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
    const u32 s = ink_r17_vd_sub8_sum_u32((u32)qi0 + (u32)qi1 + (u32)qi2 + (u32)qi3);
    if (sl == 0u) {
      ((G float *)(oqt + (nb_q << 5u)))[blk] = d;
      ((G u32 *)(oqt + nb_q * 36u))[blk] = s;
    }
  }
}

/* ============================================================================
 * KERNEL 1: ie_inkling_ds_qkv_rope_q4k_r17 (H25a Fix + LDS-Staged xq + Prefetch)
 *   Unified MAT_Q4K QKV + exact per-head QK-RMSNorm + NeoX RoPE (5 -> 1 launch).
 *   Bug fixed vs r12: r12:622 used wave_tree_f() (lane-0-only sum!) instead of
 *   wave_sum_all(), scrambling 120/128 head dims of Q & K. Here we use 4 exact
 *   32-lane wave_sum_all() reductions matching wg_sum() in ie_qkn_rope_kv
 *   bit-for-bit, plus LDS-staged xq (<= 80 blocks) and 1-SB weight prefetch.
 *   Grid: (nh + 2*nkv, (B + 3)/4, 1), NT=256, 14,848 B LDS, <= 80 VGPR (4 WG/CU).
 * ========================================================================== */

typedef struct {
  u32   xq_i8[4u][INK_R17_VD_DS_MAXNB * 8u] __attribute__((aligned(16))); /* 10,240 B */
  float xq_da[4u][INK_R17_VD_DS_MAXNB]      __attribute__((aligned(16))); /*  1,280 B */
  int   xq_as[4u][INK_R17_VD_DS_MAXNB]      __attribute__((aligned(16))); /*  1,280 B */
  float qkvh[4u][INK_R17_VD_HD]             __attribute__((aligned(16))); /*  2,048 B */
} ink_r17_vd_ds_qkv_lds_t;

static LDS ink_r17_vd_ds_qkv_lds_t ink_r17_vd_ds_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r17_vd_ds_qkv_lds_t) == 14848u, "ink_r17_vd_ds_qkv_lds_t must be 14,848 B (<= 16 KiB)");

/* 4 passes x 32 rows = all 128 rows of head h across bt <= 4 tokens (exact Q4_K order).
 * use_lds_x is a literal at each call site. Chosen inside the token loop instead, the LDS and
 * global activation loads share destination registers, and the compiler then drains the next
 * super-block's weight prefetch before the first dot of every super-block. */
static inline __attribute__((always_inline)) void ink_r17_vd_ds_qkv_passes(
    const int use_lds_x, const G u32 *qw, const G u16 *qm, const G u8 *xq_t0, G float *v_out,
    const u32 mat_rows, const u32 head_r0, const u32 nb, const u32 bt, const u32 xqs, const u32 kvs,
    const u32 t0, const int is_v, const u32 row_in_pass, const u32 sub) {
#pragma clang fp contract(off)
  const u32 nsb = nb >> 3u;
  const u32 da_base = nb << 5u;
  const u32 as_base = nb * 36u;
  const G u32 *dd_base = (const G u32 *)(qm + (ink_r17_vd_u64)mat_rows * (ink_r17_vd_u64)nb);

#pragma clang loop unroll(disable)
  for (u32 pass = 0u; pass < 4u; pass++) {
    const u32 loc_r = (pass << 5u) + row_in_pass; /* 0..127 */
    const u32 r = head_r0 + loc_r;
    const G u32 *qw_r = qw + ((ink_r17_vd_u64)r * (ink_r17_vd_u64)nb) * 4ull;
    const G u16 *sm_r = qm + (ink_r17_vd_u64)r * (ink_r17_vd_u64)nb;
    const G u32 *dd_r = dd_base + (ink_r17_vd_u64)r * (ink_r17_vd_u64)nsb;

    ink_r17_vd_f4 a0 = (ink_r17_vd_f4)(0.0f);
    ink_r17_vd_f4 a1 = (ink_r17_vd_f4)(0.0f);
    ink_r17_vd_f4 a2 = (ink_r17_vd_f4)(0.0f);
    ink_r17_vd_f4 a3 = (ink_r17_vd_f4)(0.0f);

    /* Prime 1-SB software prefetch for sb == 0. */
    u32x4 w4_c = ink_r17_vd_ld4(qw_r + (ink_r17_vd_u64)sub * 4ull);
    u32 sm_c = (u32)sm_r[sub];
    u32 dd_c = dd_r[0u];
    __asm__ volatile("" : "+v"(sm_c), "+v"(dd_c));

#pragma clang loop unroll(disable)
    for (u32 sb = 0u; sb < nsb; sb++) {
      const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
      const u32 slot = (u32)__builtin_amdgcn_readfirstlane((int)(sb_u & 3u));
      const u32 b = (sb_u << 3u) + sub;

      const u32 sb_n = (sb_u + 1u < nsb) ? (sb_u + 1u) : sb_u;
      const u32 bn = (sb_n << 3u) + sub;
      const u32x4 w4_n = ink_r17_vd_ld4(qw_r + (ink_r17_vd_u64)bn * 4ull);
      const u32 sm_n = (u32)sm_r[bn];
      const u32 dd_n = dd_r[sb_n];
      __builtin_amdgcn_sched_barrier(0);

      const u16 d_bits = (u16)(dd_c & 0xFFFFu);
      const u16 dm_bits = (u16)(dd_c >> 16u);
      const float dv = (float)__builtin_bit_cast(f16, d_bits);
      const float mv = (float)__builtin_bit_cast(f16, dm_bits);
      const float sc = dv * (float)(sm_c & 63u);
      const float mm = mv * (float)(sm_c >> 8u);

      const u32 wx = w4_c.x, wy = w4_c.y, wz = w4_c.z, ww = w4_c.w;
      const u32 lo0 = ie_q4k_lo(wx), hi0 = ie_q4k_hi(wx);
      const u32 lo1 = ie_q4k_lo(wy), hi1 = ie_q4k_hi(wy);
      const u32 lo2 = ie_q4k_lo(wz), hi2 = ie_q4k_hi(wz);
      const u32 lo3 = ie_q4k_lo(ww), hi3 = ie_q4k_hi(ww);

#pragma unroll
      for (u32 j = 0u; j < 4u; j++) {
        if (j < bt) {
          u32x4 act0, act1;
          float da;
          int as;
          if (use_lds_x) {
            act0 = *(const LDSP u32x4 *)(ink_r17_vd_ds_lds.xq_i8[j] + (b << 3u));
            act1 = *(const LDSP u32x4 *)(ink_r17_vd_ds_lds.xq_i8[j] + (b << 3u) + 4u);
            da = ink_r17_vd_ds_lds.xq_da[j][b];
            as = ink_r17_vd_ds_lds.xq_as[j][b];
          } else {
            const G u8 *xj = xq_t0 + (ink_r17_vd_u64)j * (ink_r17_vd_u64)xqs;
            act0 = ink_r17_vd_ld4(xj + ((ink_r17_vd_u64)b << 5u));
            act1 = ink_r17_vd_ld4(xj + ((ink_r17_vd_u64)b << 5u) + 16ull);
            da = ((const G float *)(xj + da_base))[b];
            as = (int)((const G u32 *)(xj + as_base))[b];
          }
          const float xa = da * (float)as;
          const u32 a00 = act0.x, a01 = act0.y, a02 = act0.z, a03 = act0.w;
          const u32 a10 = act1.x, a11 = act1.y, a12 = act1.z, a13 = act1.w;

          u32 dq = ie_dot4_us(0u, lo0, a00);
          dq = ie_dot4_us(dq, hi0, a10);
          dq = ie_dot4_us(dq, lo1, a01);
          dq = ie_dot4_us(dq, hi1, a11);
          dq = ie_dot4_us(dq, lo2, a02);
          dq = ie_dot4_us(dq, hi2, a12);
          dq = ie_dot4_us(dq, lo3, a03);
          dq = ie_dot4_us(dq, hi3, a13);

          const float prod = sc * (float)(int)dq;
          if (slot == 0u) a0[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a0[j]));
          else if (slot == 1u) a1[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a1[j]));
          else if (slot == 2u) a2[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a2[j]));
          else a3[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a3[j]));
        }
      }
      __builtin_amdgcn_sched_barrier(0);
      w4_c = w4_n;
      sm_c = sm_n;
      dd_c = dd_n;
    }

#pragma unroll
    for (u32 j = 0u; j < 4u; j++) {
      if (j < bt) {
        const float s = ink_r17_vd_oct_tree_f((a0[j] + a2[j]) + (a1[j] + a3[j]));
        if (sub == 0u) {
          if (is_v) {
            G float *vp = (G float *)((G u8 *)v_out + (ink_r17_vd_u64)(t0 + j) * (ink_r17_vd_u64)kvs) + head_r0;
            vp[loc_r] = s;
          } else {
            ink_r17_vd_ds_lds.qkvh[j][loc_r] = s;
          }
        }
      }
    }
  }
}

KERNEL ie_inkling_ds_qkv_rope_q4k_r17(
    const G u32 *wq0, const G u16 *wq1, const G u32 *wk0, const G u16 *wk1,
    const G u32 *wv0, const G u16 *wv1, const G u8 *xq,
    const G float *q_norm, const G float *k_norm,
    const G float *cs, const G float *sn,
    G float *q_out, G float *k_out, G float *v_out,
    u32 nh, u32 nkv, u32 hd, u32 nrot, u32 nb, u32 B,
    u32 xqs, u32 qs, u32 kvs, float eps) {
#pragma clang fp contract(off)
  const u32 g = wgid();
  const u32 t0 = tokid() << 2u;
  if (g >= nh + 2u * nkv || t0 >= B || hd != INK_R17_VD_HD || nrot > INK_R17_VD_HD ||
      (nrot & 1u) != 0u || nb == 0u || (nb & 7u) != 0u || !xq)
    return;

  const u32 bt = (B - t0 < 4u) ? (B - t0) : 4u;
  const int is_v = (g >= nh + nkv);
  const int is_k = (!is_v && g >= nh);
  const u32 h = is_v ? (g - nh - nkv) : (is_k ? (g - nh) : g);
  const u32 mat_rows = is_v ? (nkv * hd) : (is_k ? (nkv * hd) : (nh * hd));
  const G u32 *qw = is_v ? wv0 : (is_k ? wk0 : wq0);
  const G u16 *qm = is_v ? wv1 : (is_k ? wk1 : wq1);
  if (!qw || !qm) return;
  if (is_v) {
    if (!v_out) return;
  } else {
    if (!(is_k ? k_out : q_out) || !(is_k ? k_norm : q_norm)) return;
    if (nrot > 0u && (!cs || !sn)) return;
  }

  const u32 th = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(th >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 row_in_pass = (w << 2u) + (l >> 3u); /* 0..31 */
  const u32 head_r0 = h * INK_R17_VD_HD;

  const G u8 *xq_t0 = xq + (ink_r17_vd_u64)t0 * (ink_r17_vd_u64)xqs;

  if (nb <= INK_R17_VD_DS_MAXNB) {
    /* Cooperatively stage bt <= 4 Q8_0 tokens into LDS once for all 4 row passes. */
    const u32 n_v4 = nb << 1u;
    const u32 da_base = nb << 5u;
    const u32 as_base = nb * 36u;
#pragma clang loop unroll(disable)
    for (u32 j = 0u; j < bt; j++) {
      const G u8 *xj = xq_t0 + (ink_r17_vd_u64)j * (ink_r17_vd_u64)xqs;
      for (u32 idx = th; idx < n_v4; idx += NT) {
        ((LDSP u32x4 *)ink_r17_vd_ds_lds.xq_i8[j])[idx] = ink_r17_vd_ld4(xj + ((ink_r17_vd_u64)idx << 4u));
      }
      for (u32 idx = th; idx < nb; idx += NT) {
        ink_r17_vd_ds_lds.xq_da[j][idx] = ((const G float *)(xj + da_base))[idx];
        ink_r17_vd_ds_lds.xq_as[j][idx] = (int)((const G u32 *)(xj + as_base))[idx];
      }
    }
    lbarrier();
    ink_r17_vd_ds_qkv_passes(1, qw, qm, xq_t0, v_out, mat_rows, head_r0, nb, bt, xqs, kvs, t0, is_v,
                             row_in_pass, sub);
  } else {
    ink_r17_vd_ds_qkv_passes(0, qw, qm, xq_t0, v_out, mat_rows, head_r0, nb, bt, xqs, kvs, t0, is_v,
                             row_in_pass, sub);
  }

  if (is_v) return;
  lbarrier();

  /* Wave w (0..bt-1) owns token t0+w: bit-exact QK-RMSNorm + NeoX RoPE (0 extra barriers!). */
  if (w < bt) {
    const u32 tk = t0 + w;
    const G float *nw = is_k ? k_norm : q_norm;
    const u32 h2 = nrot >> 1u;
    const G float *cs_t = (nrot > 0u) ? (cs + (ink_r17_vd_u64)tk * (ink_r17_vd_u64)h2) : (const G float *)0;
    const G float *sn_t = (nrot > 0u) ? (sn + (ink_r17_vd_u64)tk * (ink_r17_vd_u64)h2) : (const G float *)0;
    G float *dst = (G float *)((G u8 *)(is_k ? k_out : q_out) +
                               (ink_r17_vd_u64)tk * (ink_r17_vd_u64)(is_k ? kvs : qs)) + head_r0;

    const float x0 = ink_r17_vd_ds_lds.qkvh[w][l];
    const float x1 = ink_r17_vd_ds_lds.qkvh[w][l + 32u];
    const float x2 = ink_r17_vd_ds_lds.qkvh[w][l + 64u];
    const float x3 = ink_r17_vd_ds_lds.qkvh[w][l + 96u];

    /* Exact 4-wave-group reduction broadcast to all 32 lanes (matches wg_sum in ie_qkn_rope_kv!). */
    const float ss0 = wave_sum_all(x0 * x0);
    const float ss1 = wave_sum_all(x1 * x1);
    const float ss2 = wave_sum_all(x2 * x2);
    const float ss3 = wave_sum_all(x3 * x3);
    const float ss = ((ss0 + ss1) + ss2) + ss3;
    const float inv = 1.0f / __builtin_sqrtf(ss * (1.0f / 128.0f) + eps);

    const float y0 = x0 * inv * nw[l];
    const float y1 = x1 * inv * nw[l + 32u];
    const float y2 = x2 * inv * nw[l + 64u];
    const float y3 = x3 * inv * nw[l + 96u];

    if (nrot == INK_R17_VD_HD) {
      const float c0 = cs_t[l], s0 = sn_t[l];
      const float c1 = cs_t[l + 32u], s1 = sn_t[l + 32u];
      dst[l]       = y0 * c0 - y2 * s0;
      dst[l + 32u] = y1 * c1 - y3 * s1;
      dst[l + 64u] = y0 * s0 + y2 * c0;
      dst[l + 96u] = y1 * s1 + y3 * c1;
    } else if (nrot == 0u) {
      dst[l]       = y0;
      dst[l + 32u] = y1;
      dst[l + 64u] = y2;
      dst[l + 96u] = y3;
    } else {
      /* General partial NeoX RoPE via wave-private LDS row qkvh[w]. */
      ink_r17_vd_ds_lds.qkvh[w][l]       = y0;
      ink_r17_vd_ds_lds.qkvh[w][l + 32u] = y1;
      ink_r17_vd_ds_lds.qkvh[w][l + 64u] = y2;
      ink_r17_vd_ds_lds.qkvh[w][l + 96u] = y3;
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
      for (u32 m = 0u; m < 4u; m++) {
        const u32 idx = (m << 5u) + l;
        float r = ink_r17_vd_ds_lds.qkvh[w][idx];
        if (idx < nrot) {
          const u32 ii = idx < h2 ? idx : (idx - h2);
          const float a = ink_r17_vd_ds_lds.qkvh[w][ii];
          const float b = ink_r17_vd_ds_lds.qkvh[w][ii + h2];
          r = (idx < h2) ? (a * cs_t[ii] - b * sn_t[ii]) : (a * sn_t[ii] + b * cs_t[ii]);
        }
        dst[idx] = r;
      }
    }
  }
}

/* ============================================================================
 * KERNEL 2: ie_inkling_flute_situ_ver_t8_r17 (T=2..8 FLUTE IQ3_XXS+SITU+Q8)
 *   Drop-in exact-SHA replacement for ie_inkling_gexp_iq3xxs_flute_situ_compact_r7
 *   and ..._nt_r10 on BOTH 128E core layers AND H38 Mode B remapped slots:
 *     - 0 TH_LOAD_NT, 0 TH_LOAD_HT (full 64 MiB MALL residency on RX 9070 XT).
 *     - nj == 1: 8-wave single-pass Gate+Up interleaved, 0 LDS X staging,
 *                branch-free 1-SB prefetch, 8 acc VGPRs.
 *     - nj == 2: 8-wave single-pass Gate+Up interleaved, LDS-staged X read ONCE
 *                per SB for both Gate & Up (9 vs 18 iters!), 16 acc VGPRs.
 *     - nj >= 3: Wave-split (Waves 0..3 Gate, Waves 4..7 Up) with cross-subpass
 *                1-SB prefetch + overlapped LDS X loads + 4-token chunking
 *                (handles nj in 1..15 in <= 16 acc VGPRs, <= 88 VGPR total).
 *   Grid: (eff / 32, max_packs, 1), NT=256, 13,568 B LDS (4 WGs/CU).
 * ========================================================================== */

static inline __attribute__((always_inline)) float ink_r17_vd_h2f(const u32 bits) {
  float f = ie_rq_h2f(bits);
  __asm__ volatile("" : "+v"(f));
  return f;
}

/* Single-mul IQ3_XXS 7-bit+parity sign decode (bitwise-identical to rq_ksign7 + ie_rq_ssign). */
static inline __attribute__((always_inline)) void ink_r17_vd_ksw(
    const u32 g0, const u32 g1, const u32 s7, u32 *w0_out, u32 *w1_out) {
  const u32 p_ = s7 * 0x00204081u;
  const u32 t0 = p_ & 0x01010101u;
  const u32 t1 = ((p_ >> 4u) | ((u32)__builtin_popcount(s7) << 24u)) & 0x01010101u;
  const u32 m0 = __builtin_amdgcn_perm(0u, 0u, t0 | 0x0C0C0C0Cu);
  const u32 m1 = __builtin_amdgcn_perm(0u, 0u, t1 | 0x0C0C0C0Cu);
  *w0_out = (g0 ^ m0) + t0;
  *w1_out = (g1 ^ m1) + t1;
}

static inline __attribute__((always_inline)) float ink_r17_vd_iq3_stream_dot(
    u32 d16, u32 aux, u32 q01, u32 q23,
    u32x4 x0, u32x4 x1, float dav) {
#pragma clang fp contract(off)
  const float dw = ink_r17_vd_h2f(d16) * (0.5f + (float)(aux >> 28u)) * 0.5f;
  const u32 qp0 = q01 & 0xFFFFu, qp1 = q01 >> 16u;
  const u32 qp2 = q23 & 0xFFFFu, qp3 = q23 >> 16u;
  const u32 g_lo0 = ie_rq_iq3xxs_grid[qp0 & 255u], g_hi0 = ie_rq_iq3xxs_grid[qp0 >> 8u];
  const u32 g_lo1 = ie_rq_iq3xxs_grid[qp1 & 255u], g_hi1 = ie_rq_iq3xxs_grid[qp1 >> 8u];
  const u32 g_lo2 = ie_rq_iq3xxs_grid[qp2 & 255u], g_hi2 = ie_rq_iq3xxs_grid[qp2 >> 8u];
  const u32 g_lo3 = ie_rq_iq3xxs_grid[qp3 & 255u], g_hi3 = ie_rq_iq3xxs_grid[qp3 >> 8u];
  __builtin_amdgcn_sched_barrier(0);

  const u32 x00 = x0.x, x01 = x0.y, x02 = x0.z, x03 = x0.w;
  const u32 x10 = x1.x, x11 = x1.y, x12 = x1.z, x13 = x1.w;

  u32 wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3;
  ink_r17_vd_ksw(g_lo0, g_hi0, aux & 127u, &wl0, &wh0);
  u32 idot = ie_dot4_ss(0u, wl0, x00);
  idot = ie_dot4_ss(idot, wh0, x01);

  ink_r17_vd_ksw(g_lo1, g_hi1, (aux >> 7u) & 127u, &wl1, &wh1);
  idot = ie_dot4_ss(idot, wl1, x02);
  idot = ie_dot4_ss(idot, wh1, x03);

  ink_r17_vd_ksw(g_lo2, g_hi2, (aux >> 14u) & 127u, &wl2, &wh2);
  idot = ie_dot4_ss(idot, wl2, x10);
  idot = ie_dot4_ss(idot, wh2, x11);

  ink_r17_vd_ksw(g_lo3, g_hi3, (aux >> 21u) & 127u, &wl3, &wh3);
  idot = ie_dot4_ss(idot, wl3, x12);
  idot = ie_dot4_ss(idot, wh3, x13);

  return dav * (dw * (float)(int)idot);
}

static inline __attribute__((always_inline)) float ink_r17_vd_iq3_dot_pre(
    u32 wl0, u32 wh0, u32 wl1, u32 wh1, u32 wl2, u32 wh2, u32 wl3, u32 wh3,
    float dw, u32x4 x0, u32x4 x1, float dav) {
#pragma clang fp contract(off)
  const u32 x00 = x0.x, x01 = x0.y, x02 = x0.z, x03 = x0.w;
  const u32 x10 = x1.x, x11 = x1.y, x12 = x1.z, x13 = x1.w;
  u32 idot = ie_dot4_ss(0u, wl0, x00);
  idot = ie_dot4_ss(idot, wh0, x01);
  idot = ie_dot4_ss(idot, wl1, x02);
  idot = ie_dot4_ss(idot, wh1, x03);
  idot = ie_dot4_ss(idot, wl2, x10);
  idot = ie_dot4_ss(idot, wh2, x11);
  idot = ie_dot4_ss(idot, wl3, x12);
  idot = ie_dot4_ss(idot, wh3, x13);
  return dav * (dw * (float)(int)idot);
}

typedef struct {
  u32   xq[4u][INK_R17_VD_FL_MAXNB * 8u] __attribute__((aligned(16))); /* 10,240 B */
  float xd[4u][INK_R17_VD_FL_MAXNB]      __attribute__((aligned(16))); /*  1,280 B */
  float g[4u][32u]                       __attribute__((aligned(16))); /*    512 B */
  float u[4u][32u]                       __attribute__((aligned(16))); /*    512 B */
} ink_r17_vd_flute_lds_t;

static LDS ink_r17_vd_flute_lds_t ink_r17_vd_fl_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r17_vd_flute_lds_t) == 12544u, "ink_r17_vd_flute_lds_t must be 12,544 B (<= 16 KiB)");

KERNEL ie_inkling_flute_situ_ver_t8_r17(
    const G u8 *qw, const G u8 *X, G float *b_out, G u8 *q_out,
    u32 eff, u32 nb, const G u32 *list, const G u32 *packs, const G u32 *n_packs,
    u32 q_bytes, u32 xs, u32 bstr, u32 qstr, u32 xdiv, float beta, float lbeta) {
#pragma clang fp contract(off)
  if (eff == 0u || (eff & 31u) != 0u || nb < 8u || (nb & 7u) != 0u ||
      !qw || !X || (!b_out && !q_out) || !list || !packs || !n_packs)
    return;

  const u32 by = tokid();
  const u32 npk = (u32)__builtin_amdgcn_readfirstlane((int)(*n_packs));
  const u32 blk = wgid();
  const u32 nb_out = eff >> 5u;
  if (by >= npk || blk >= nb_out) return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)packs[by]);
  const u32 e = desc & 0xFFu;
  const u32 lo = (desc >> 8u) & 0xFFFu;
  const u32 nj = (desc >> 20u) & 0xFu;
  if (nj == 0u) return;

  const u32 t0 = tid();

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 r_in_blk = (wv << 2u) + k;
  const u32 mb = nb >> 3u;

  const ink_r17_vd_u64 n_sb_tot = (ink_r17_vd_u64)(eff >> 1u) * (ink_r17_vd_u64)mb;
  const G u8 *ebase = rq_ptr_u(qw + (ink_r17_vd_u64)e * (ink_r17_vd_u64)q_bytes);
  const G ink_r17_vd_u2 *p_qs  = (const G ink_r17_vd_u2 *)ebase;
  const G u32           *p_aux = (const G u32 *)(ebase + n_sb_tot * 256ull);
  const G u16           *p_d   = (const G u16 *)(ebase + n_sb_tot * 384ull);
  const ink_r17_vd_u64 x_sc_off = 32ull * (ink_r17_vd_u64)nb;

  /* ------------------------------------------------------------------------
   * Path 1 (nj == 1): 8-wave single-pass Gate+Up interleaved, 0 LDS X copy,
   *                   branch-free 1-SB prefetch, cached loads only.
   * ---------------------------------------------------------------------- */
  if (nj == 1u) {
    const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
    const G u8 *xp0 = rq_ptr_u(X + (ink_r17_vd_u64)ink_r17_vd_div_x(p0, xdiv) * (ink_r17_vd_u64)xs);
    const G float *dap0 = (const G float *)(xp0 + x_sc_off);

    const u32 rg_g = (blk << 3u) + wv;
    const u32 rg_u = (eff >> 2u) + rg_g;
    const ink_r17_vd_u64 gg0 = (ink_r17_vd_u64)rg_g * (ink_r17_vd_u64)mb;
    const ink_r17_vd_u64 gu0 = (ink_r17_vd_u64)rg_u * (ink_r17_vd_u64)mb;
    const G ink_r17_vd_u2 *qs_g = p_qs + gg0 * 32ull + (ink_r17_vd_u64)l;
    const G u32 *aux_g = p_aux + gg0 * 32ull + (ink_r17_vd_u64)l;
    const G u16 *d_g = p_d + gg0 * 4ull + (ink_r17_vd_u64)k;
    const G ink_r17_vd_u2 *qs_u = p_qs + gu0 * 32ull + (ink_r17_vd_u64)l;
    const G u32 *aux_u = p_aux + gu0 * 32ull + (ink_r17_vd_u64)l;
    const G u16 *d_u = p_d + gu0 * 4ull + (ink_r17_vd_u64)k;

    float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
    float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;

    ink_r17_vd_u2 qg_c = ink_r17_vd_ld2(qs_g);
    u32 auxg_c = aux_g[0u];
    u32 dg_c = (u32)d_g[0u];
    u32x4 x0_c = ink_r17_vd_ld4(xp0 + ((ink_r17_vd_u64)s << 5u));
    u32x4 x1_c = ink_r17_vd_ld4(xp0 + ((ink_r17_vd_u64)s << 5u) + 16ull);
    float dav_c = dap0[s];
    __asm__ volatile("" : "+v"(dg_c), "+v"(auxg_c));

    /* Four super-blocks per pass, unrolled, so the accumulator index is a constant in each copy and
     * INK_R17_VD_ADD4 is one add instead of a compare/branch ladder on sb & 3. Every accumulator
     * still takes the same super-blocks in the same order. */
#pragma clang loop unroll(disable)
    for (u32 sb4 = 0u; sb4 < mb; sb4 += 4u) {
#pragma unroll
      for (u32 j_acc = 0u; j_acc < 4u; j_acc++) {
        const u32 sb = sb4 + j_acc;
        if (sb < mb) {
          const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
          const u32 sb_n = (sb_u + 1u < mb) ? (sb_u + 1u) : sb_u;
          const u32 bn = (sb_n << 3u) + s;

          const ink_r17_vd_u2 qu_c = ink_r17_vd_ld2(qs_u + (ink_r17_vd_u64)sb_u * 32ull);
          const u32 auxu_c = aux_u[(ink_r17_vd_u64)sb_u * 32ull];
          const u32 du_c = (u32)d_u[(ink_r17_vd_u64)sb_u * 4ull];
          __builtin_amdgcn_sched_barrier(0);

          const float vg = ink_r17_vd_iq3_stream_dot(
              dg_c, auxg_c, qg_c.x, qg_c.y, x0_c, x1_c, dav_c);
          INK_R17_VD_ADD4(ag0, ag1, ag2, ag3, j_acc, vg);
          __builtin_amdgcn_sched_barrier(0);

          const ink_r17_vd_u2 qg_n = ink_r17_vd_ld2(qs_g + (ink_r17_vd_u64)sb_n * 32ull);
          const u32 auxg_n = aux_g[(ink_r17_vd_u64)sb_n * 32ull];
          const u32 dg_n = (u32)d_g[(ink_r17_vd_u64)sb_n * 4ull];
          const u32x4 x0_n = ink_r17_vd_ld4(xp0 + ((ink_r17_vd_u64)bn << 5u));
          const u32x4 x1_n = ink_r17_vd_ld4(xp0 + ((ink_r17_vd_u64)bn << 5u) + 16ull);
          const float dav_n = dap0[bn];
          __builtin_amdgcn_sched_barrier(0);

          const float vu = ink_r17_vd_iq3_stream_dot(
              du_c, auxu_c, qu_c.x, qu_c.y, x0_c, x1_c, dav_c);
          INK_R17_VD_ADD4(au0, au1, au2, au3, j_acc, vu);
          __builtin_amdgcn_sched_barrier(0);

          qg_c = qg_n; auxg_c = auxg_n; dg_c = dg_n;
          x0_c = x0_n; x1_c = x1_n; dav_c = dav_n;
        }
      }
    }

    const float gv = ink_r17_vd_oct_tree_f((ag0 + ag2) + (ag1 + ag3));
    const float uv = ink_r17_vd_oct_tree_f((au0 + au2) + (au1 + au3));
    if (s == 0u) {
      const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
      const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
      ink_r17_vd_fl_lds.g[0u][r_in_blk] = gs * sigmoidf(gv) * us;
    }
    lbarrier();
    if (wv == 0u) {
      const float v = ink_r17_vd_fl_lds.g[0u][l];
      if (b_out) ((G float *)((G u8 *)b_out + (ink_r17_vd_u64)p0 * (ink_r17_vd_u64)bstr))[(blk << 5u) + l] = v;
      if (q_out) quant_wave(v, blk, q_out + (ink_r17_vd_u64)p0 * (ink_r17_vd_u64)qstr, nb_out);
    }
    return;
  }

  const int use_lds_x = (nb <= INK_R17_VD_FL_MAXNB);

  /* ------------------------------------------------------------------------
   * Path 2 (nj == 2): 8-wave single-pass Gate+Up interleaved (9 vs 18 iters!),
   *                   16 acc VGPRs, LDS-staged X read ONCE for both Gate & Up.
   * ---------------------------------------------------------------------- */
  if (nj == 2u) {
    const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
    const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 1u]);
    const G u8 *xp0 = rq_ptr_u(X + (ink_r17_vd_u64)ink_r17_vd_div_x(p0, xdiv) * (ink_r17_vd_u64)xs);
    const G u8 *xp1 = rq_ptr_u(X + (ink_r17_vd_u64)ink_r17_vd_div_x(p1, xdiv) * (ink_r17_vd_u64)xs);

    if (use_lds_x) {
      const u32 n_v4 = nb << 1u;
      for (u32 idx = t0; idx < n_v4; idx += NT) {
        ((LDSP u32x4 *)ink_r17_vd_fl_lds.xq[0u])[idx] = ink_r17_vd_ld4(xp0 + ((ink_r17_vd_u64)idx << 4u));
        ((LDSP u32x4 *)ink_r17_vd_fl_lds.xq[1u])[idx] = ink_r17_vd_ld4(xp1 + ((ink_r17_vd_u64)idx << 4u));
      }
      for (u32 idx = t0; idx < nb; idx += NT) {
        ink_r17_vd_fl_lds.xd[0u][idx] = ((const G float *)(xp0 + x_sc_off))[idx];
        ink_r17_vd_fl_lds.xd[1u][idx] = ((const G float *)(xp1 + x_sc_off))[idx];
      }
    }
    lbarrier();

    const G float *dap0 = (const G float *)(xp0 + x_sc_off);
    const G float *dap1 = (const G float *)(xp1 + x_sc_off);

    const u32 rg_g = (blk << 3u) + wv;
    const u32 rg_u = (eff >> 2u) + rg_g;
    const ink_r17_vd_u64 gg0 = (ink_r17_vd_u64)rg_g * (ink_r17_vd_u64)mb;
    const ink_r17_vd_u64 gu0 = (ink_r17_vd_u64)rg_u * (ink_r17_vd_u64)mb;
    const G ink_r17_vd_u2 *qs_g = p_qs + gg0 * 32ull + (ink_r17_vd_u64)l;
    const G u32 *aux_g = p_aux + gg0 * 32ull + (ink_r17_vd_u64)l;
    const G u16 *d_g = p_d + gg0 * 4ull + (ink_r17_vd_u64)k;
    const G ink_r17_vd_u2 *qs_u = p_qs + gu0 * 32ull + (ink_r17_vd_u64)l;
    const G u32 *aux_u = p_aux + gu0 * 32ull + (ink_r17_vd_u64)l;
    const G u16 *d_u = p_d + gu0 * 4ull + (ink_r17_vd_u64)k;

    float ag0_0 = 0.0f, ag0_1 = 0.0f, ag0_2 = 0.0f, ag0_3 = 0.0f;
    float ag1_0 = 0.0f, ag1_1 = 0.0f, ag1_2 = 0.0f, ag1_3 = 0.0f;
    float au0_0 = 0.0f, au0_1 = 0.0f, au0_2 = 0.0f, au0_3 = 0.0f;
    float au1_0 = 0.0f, au1_1 = 0.0f, au1_2 = 0.0f, au1_3 = 0.0f;

    ink_r17_vd_u2 qg_c = ink_r17_vd_ld2(qs_g);
    u32 auxg_c = aux_g[0u];
    u32 dg_c = (u32)d_g[0u];
    __asm__ volatile("" : "+v"(dg_c), "+v"(auxg_c));

    /* Unrolled by four for a constant accumulator index, as in the nj == 1 path. */
#pragma clang loop unroll(disable)
    for (u32 sb4 = 0u; sb4 < mb; sb4 += 4u) {
#pragma unroll
      for (u32 j_acc = 0u; j_acc < 4u; j_acc++) {
        const u32 sb = sb4 + j_acc;
        if (sb < mb) {
          const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
          const u32 b = (sb_u << 3u) + s;
          const u32 sb_n = (sb_u + 1u < mb) ? (sb_u + 1u) : sb_u;

          const ink_r17_vd_u2 qu_c = ink_r17_vd_ld2(qs_u + (ink_r17_vd_u64)sb_u * 32ull);
          const u32 auxu_c = aux_u[(ink_r17_vd_u64)sb_u * 32ull];
          const u32 du_c = (u32)d_u[(ink_r17_vd_u64)sb_u * 4ull];

          /* Load both tokens' X block once; reused for Gate AND Up! */
          u32x4 x0_0, x1_0, x0_1, x1_1;
          float d0, d1;
          if (use_lds_x) {
            x0_0 = *(const LDSP u32x4 *)(ink_r17_vd_fl_lds.xq[0u] + (b << 3u));
            x1_0 = *(const LDSP u32x4 *)(ink_r17_vd_fl_lds.xq[0u] + (b << 3u) + 4u);
            d0 = ink_r17_vd_fl_lds.xd[0u][b];
            x0_1 = *(const LDSP u32x4 *)(ink_r17_vd_fl_lds.xq[1u] + (b << 3u));
            x1_1 = *(const LDSP u32x4 *)(ink_r17_vd_fl_lds.xq[1u] + (b << 3u) + 4u);
            d1 = ink_r17_vd_fl_lds.xd[1u][b];
          } else {
            x0_0 = ink_r17_vd_ld4(xp0 + ((ink_r17_vd_u64)b << 5u));
            x1_0 = ink_r17_vd_ld4(xp0 + ((ink_r17_vd_u64)b << 5u) + 16ull);
            d0 = dap0[b];
            x0_1 = ink_r17_vd_ld4(xp1 + ((ink_r17_vd_u64)b << 5u));
            x1_1 = ink_r17_vd_ld4(xp1 + ((ink_r17_vd_u64)b << 5u) + 16ull);
            d1 = dap1[b];
          }

          /* Gate decode + 2-token dot4. */
          {
            const u32 q01 = qg_c.x, q23 = qg_c.y;
            const u32 qp0 = q01 & 0xFFFFu, qp1 = q01 >> 16u;
            const u32 qp2 = q23 & 0xFFFFu, qp3 = q23 >> 16u;
            const u32 g_lo0 = ie_rq_iq3xxs_grid[qp0 & 255u], g_hi0 = ie_rq_iq3xxs_grid[qp0 >> 8u];
            const u32 g_lo1 = ie_rq_iq3xxs_grid[qp1 & 255u], g_hi1 = ie_rq_iq3xxs_grid[qp1 >> 8u];
            const u32 g_lo2 = ie_rq_iq3xxs_grid[qp2 & 255u], g_hi2 = ie_rq_iq3xxs_grid[qp2 >> 8u];
            const u32 g_lo3 = ie_rq_iq3xxs_grid[qp3 & 255u], g_hi3 = ie_rq_iq3xxs_grid[qp3 >> 8u];
            __builtin_amdgcn_sched_barrier(0);

            const float dw = ink_r17_vd_h2f(dg_c) * (0.5f + (float)(auxg_c >> 28u)) * 0.5f;
            u32 wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3;
            ink_r17_vd_ksw(g_lo0, g_hi0, auxg_c & 127u, &wl0, &wh0);
            ink_r17_vd_ksw(g_lo1, g_hi1, (auxg_c >> 7u) & 127u, &wl1, &wh1);
            ink_r17_vd_ksw(g_lo2, g_hi2, (auxg_c >> 14u) & 127u, &wl2, &wh2);
            ink_r17_vd_ksw(g_lo3, g_hi3, (auxg_c >> 21u) & 127u, &wl3, &wh3);

            INK_R17_VD_ADD4(ag0_0, ag0_1, ag0_2, ag0_3, j_acc,
                            ink_r17_vd_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_0, x1_0, d0));
            INK_R17_VD_ADD4(ag1_0, ag1_1, ag1_2, ag1_3, j_acc,
                            ink_r17_vd_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_1, x1_1, d1));
          }
          __builtin_amdgcn_sched_barrier(0);

          /* Issue next-SB Gate prefetch before Up decode + dot4. */
          const ink_r17_vd_u2 qg_n = ink_r17_vd_ld2(qs_g + (ink_r17_vd_u64)sb_n * 32ull);
          const u32 auxg_n = aux_g[(ink_r17_vd_u64)sb_n * 32ull];
          const u32 dg_n = (u32)d_g[(ink_r17_vd_u64)sb_n * 4ull];
          __builtin_amdgcn_sched_barrier(0);

          /* Up decode + 2-token dot4. */
          {
            const u32 q01 = qu_c.x, q23 = qu_c.y;
            const u32 qp0 = q01 & 0xFFFFu, qp1 = q01 >> 16u;
            const u32 qp2 = q23 & 0xFFFFu, qp3 = q23 >> 16u;
            const u32 g_lo0 = ie_rq_iq3xxs_grid[qp0 & 255u], g_hi0 = ie_rq_iq3xxs_grid[qp0 >> 8u];
            const u32 g_lo1 = ie_rq_iq3xxs_grid[qp1 & 255u], g_hi1 = ie_rq_iq3xxs_grid[qp1 >> 8u];
            const u32 g_lo2 = ie_rq_iq3xxs_grid[qp2 & 255u], g_hi2 = ie_rq_iq3xxs_grid[qp2 >> 8u];
            const u32 g_lo3 = ie_rq_iq3xxs_grid[qp3 & 255u], g_hi3 = ie_rq_iq3xxs_grid[qp3 >> 8u];
            __builtin_amdgcn_sched_barrier(0);

            const float dw = ink_r17_vd_h2f(du_c) * (0.5f + (float)(auxu_c >> 28u)) * 0.5f;
            u32 wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3;
            ink_r17_vd_ksw(g_lo0, g_hi0, auxu_c & 127u, &wl0, &wh0);
            ink_r17_vd_ksw(g_lo1, g_hi1, (auxu_c >> 7u) & 127u, &wl1, &wh1);
            ink_r17_vd_ksw(g_lo2, g_hi2, (auxu_c >> 14u) & 127u, &wl2, &wh2);
            ink_r17_vd_ksw(g_lo3, g_hi3, (auxu_c >> 21u) & 127u, &wl3, &wh3);

            INK_R17_VD_ADD4(au0_0, au0_1, au0_2, au0_3, j_acc,
                            ink_r17_vd_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_0, x1_0, d0));
            INK_R17_VD_ADD4(au1_0, au1_1, au1_2, au1_3, j_acc,
                            ink_r17_vd_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_1, x1_1, d1));
          }
          __builtin_amdgcn_sched_barrier(0);

          qg_c = qg_n; auxg_c = auxg_n; dg_c = dg_n;
        }
      }
    }

    {
      const float gv0 = ink_r17_vd_oct_tree_f((ag0_0 + ag0_2) + (ag0_1 + ag0_3));
      const float uv0 = ink_r17_vd_oct_tree_f((au0_0 + au0_2) + (au0_1 + au0_3));
      const float gv1 = ink_r17_vd_oct_tree_f((ag1_0 + ag1_2) + (ag1_1 + ag1_3));
      const float uv1 = ink_r17_vd_oct_tree_f((au1_0 + au1_2) + (au1_1 + au1_3));
      if (s == 0u) {
        const float gs0 = beta > 0.0f ? beta * ie_tanhf_s(gv0 / beta) : gv0;
        const float us0 = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv0 / lbeta) : uv0;
        ink_r17_vd_fl_lds.g[0u][r_in_blk] = gs0 * sigmoidf(gv0) * us0;

        const float gs1 = beta > 0.0f ? beta * ie_tanhf_s(gv1 / beta) : gv1;
        const float us1 = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv1 / lbeta) : uv1;
        ink_r17_vd_fl_lds.g[1u][r_in_blk] = gs1 * sigmoidf(gv1) * us1;
      }
    }
    lbarrier();
    if (wv < 2u) {
      const u32 pw = (wv == 0u) ? p0 : p1;
      const float v = ink_r17_vd_fl_lds.g[wv][l];
      if (b_out) ((G float *)((G u8 *)b_out + (ink_r17_vd_u64)pw * (ink_r17_vd_u64)bstr))[(blk << 5u) + l] = v;
      if (q_out) quant_wave(v, blk, q_out + (ink_r17_vd_u64)pw * (ink_r17_vd_u64)qstr, nb_out);
    }
    return;
  }

  /* ------------------------------------------------------------------------
   * Path 3 (nj >= 3, handles nj=3..4 in 1 chunk and nj=5..15 in <=4 chunks):
   *   Wave-split (Waves 0..3 Gate, Waves 4..7 Up) with cross-subpass 1-SB
   *   weight prefetch + overlapped LDS X loads -> <= 16 acc VGPRs, <= 88 VGPR!
   * ---------------------------------------------------------------------- */
  const u32 is_up = wv >> 2u;
  const u32 wv4 = wv & 3u;
  const u32 rg_half_base = (is_up ? (eff >> 2u) : 0u) + (blk << 3u) + wv4;
  const ink_r17_vd_u64 g0_sp0 = (ink_r17_vd_u64)rg_half_base * (ink_r17_vd_u64)mb;
  const ink_r17_vd_u64 g0_sp1 = (ink_r17_vd_u64)(rg_half_base + 4u) * (ink_r17_vd_u64)mb;
  const G ink_r17_vd_u2 *qs_sp0 = p_qs + g0_sp0 * 32ull + (ink_r17_vd_u64)l;
  const G u32 *aux_sp0 = p_aux + g0_sp0 * 32ull + (ink_r17_vd_u64)l;
  const G u16 *d_sp0 = p_d + g0_sp0 * 4ull + (ink_r17_vd_u64)k;
  const G ink_r17_vd_u2 *qs_sp1 = p_qs + g0_sp1 * 32ull + (ink_r17_vd_u64)l;
  const G u32 *aux_sp1 = p_aux + g0_sp1 * 32ull + (ink_r17_vd_u64)l;
  const G u16 *d_sp1 = p_d + g0_sp1 * 4ull + (ink_r17_vd_u64)k;

#pragma clang loop unroll(disable)
  for (u32 j0 = 0u; j0 < nj; j0 += 4u) {
    const u32 nj_c = (nj - j0 < 4u) ? (nj - j0) : 4u;
    const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo + j0]);
    const u32 p1 = (nj_c > 1u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + j0 + 1u]) : p0;
    const u32 p2 = (nj_c > 2u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + j0 + 2u]) : p0;
    const u32 p3 = (nj_c > 3u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + j0 + 3u]) : p0;

    const G u8 *xp0 = rq_ptr_u(X + (ink_r17_vd_u64)ink_r17_vd_div_x(p0, xdiv) * (ink_r17_vd_u64)xs);
    const G u8 *xp1 = (nj_c > 1u) ? rq_ptr_u(X + (ink_r17_vd_u64)ink_r17_vd_div_x(p1, xdiv) * (ink_r17_vd_u64)xs) : xp0;
    const G u8 *xp2 = (nj_c > 2u) ? rq_ptr_u(X + (ink_r17_vd_u64)ink_r17_vd_div_x(p2, xdiv) * (ink_r17_vd_u64)xs) : xp0;
    const G u8 *xp3 = (nj_c > 3u) ? rq_ptr_u(X + (ink_r17_vd_u64)ink_r17_vd_div_x(p3, xdiv) * (ink_r17_vd_u64)xs) : xp0;

    if (use_lds_x) {
      const u32 n_v4 = nb << 1u;
#pragma clang loop unroll(disable)
      for (u32 j = 0u; j < nj_c; j++) {
        const G u8 *xpj = (j == 0u) ? xp0 : ((j == 1u) ? xp1 : ((j == 2u) ? xp2 : xp3));
        for (u32 idx = t0; idx < n_v4; idx += NT) {
          ((LDSP u32x4 *)ink_r17_vd_fl_lds.xq[j])[idx] = ink_r17_vd_ld4(xpj + ((ink_r17_vd_u64)idx << 4u));
        }
        for (u32 idx = t0; idx < nb; idx += NT) {
          ink_r17_vd_fl_lds.xd[j][idx] = ((const G float *)(xpj + x_sc_off))[idx];
        }
      }
    }
    lbarrier();

    const G float *dap0 = (const G float *)(xp0 + x_sc_off);
    const G float *dap1 = (const G float *)(xp1 + x_sc_off);
    const G float *dap2 = (const G float *)(xp2 + x_sc_off);
    const G float *dap3 = (const G float *)(xp3 + x_sc_off);

    /* Prime sp==0, sb==0 FLUTE weights; cross-subpass prefetch eliminates sp==1 bubble! */
    ink_r17_vd_u2 q_c = ink_r17_vd_ld2(qs_sp0);
    u32 aux_c = aux_sp0[0u];
    u32 d_c = (u32)d_sp0[0u];
    __asm__ volatile("" : "+v"(d_c), "+v"(aux_c));

#pragma clang loop unroll(disable)
    for (u32 sp = 0u; sp < 2u; sp++) {
      const G ink_r17_vd_u2 *qs_cur = (sp == 0u) ? qs_sp0 : qs_sp1;
      const G u32 *aux_cur = (sp == 0u) ? aux_sp0 : aux_sp1;
      const G u16 *d_cur = (sp == 0u) ? d_sp0 : d_sp1;

      float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
      float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
      float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
      float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;

#pragma clang loop unroll(disable)
      for (u32 sb = 0u; sb < mb; sb++) {
        const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
        const u32 b = (sb_u << 3u) + s;
        const u32 j_acc = sb_u & 3u;

        /* Cross-subpass prefetch: at (sp==0, sb==mb-1), prefetch (sp==1, sb==0)! */
        const int xpass = (sp == 0u) && (sb_u + 1u == mb);
        const G ink_r17_vd_u2 *qs_np = xpass ? qs_sp1 : (qs_cur + (ink_r17_vd_u64)((sb_u + 1u < mb) ? (sb_u + 1u) : sb_u) * 32ull);
        const G u32 *aux_np = xpass ? aux_sp1 : (aux_cur + (ink_r17_vd_u64)((sb_u + 1u < mb) ? (sb_u + 1u) : sb_u) * 32ull);
        const G u16 *d_np = xpass ? d_sp1 : (d_cur + (ink_r17_vd_u64)((sb_u + 1u < mb) ? (sb_u + 1u) : sb_u) * 4ull);

        const ink_r17_vd_u2 q_n = ink_r17_vd_ld2(qs_np);
        const u32 aux_n = aux_np[0u];
        const u32 d_n = (u32)d_np[0u];

        const u32 q01_c = q_c.x, q23_c = q_c.y;
        const u32 qp0 = q01_c & 0xFFFFu, qp1 = q01_c >> 16u;
        const u32 qp2 = q23_c & 0xFFFFu, qp3 = q23_c >> 16u;
        const u32 g_lo0 = ie_rq_iq3xxs_grid[qp0 & 255u], g_hi0 = ie_rq_iq3xxs_grid[qp0 >> 8u];
        const u32 g_lo1 = ie_rq_iq3xxs_grid[qp1 & 255u], g_hi1 = ie_rq_iq3xxs_grid[qp1 >> 8u];
        const u32 g_lo2 = ie_rq_iq3xxs_grid[qp2 & 255u], g_hi2 = ie_rq_iq3xxs_grid[qp2 >> 8u];
        const u32 g_lo3 = ie_rq_iq3xxs_grid[qp3 & 255u], g_hi3 = ie_rq_iq3xxs_grid[qp3 >> 8u];

        /* Issue token 0 X load BEFORE sign-mask decode to hide LDS latency! */
        const u32x4 x0_0 = use_lds_x ? *(const LDSP u32x4 *)(ink_r17_vd_fl_lds.xq[0u] + (b << 3u))
                                     : ink_r17_vd_ld4(xp0 + ((ink_r17_vd_u64)b << 5u));
        const u32x4 x1_0 = use_lds_x ? *(const LDSP u32x4 *)(ink_r17_vd_fl_lds.xq[0u] + (b << 3u) + 4u)
                                     : ink_r17_vd_ld4(xp0 + ((ink_r17_vd_u64)b << 5u) + 16ull);
        const float d0 = use_lds_x ? ink_r17_vd_fl_lds.xd[0u][b] : dap0[b];
        __builtin_amdgcn_sched_barrier(0);

        const float dw = ink_r17_vd_h2f(d_c) * (0.5f + (float)(aux_c >> 28u)) * 0.5f;
        u32 wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3;
        ink_r17_vd_ksw(g_lo0, g_hi0, aux_c & 127u, &wl0, &wh0);
        ink_r17_vd_ksw(g_lo1, g_hi1, (aux_c >> 7u) & 127u, &wl1, &wh1);
        ink_r17_vd_ksw(g_lo2, g_hi2, (aux_c >> 14u) & 127u, &wl2, &wh2);
        ink_r17_vd_ksw(g_lo3, g_hi3, (aux_c >> 21u) & 127u, &wl3, &wh3);

        INK_R17_VD_ADD4(a0_0, a0_1, a0_2, a0_3, j_acc,
                        ink_r17_vd_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_0, x1_0, d0));

        if (nj_c > 1u) {
          const u32x4 x0_1 = use_lds_x ? *(const LDSP u32x4 *)(ink_r17_vd_fl_lds.xq[1u] + (b << 3u))
                                       : ink_r17_vd_ld4(xp1 + ((ink_r17_vd_u64)b << 5u));
          const u32x4 x1_1 = use_lds_x ? *(const LDSP u32x4 *)(ink_r17_vd_fl_lds.xq[1u] + (b << 3u) + 4u)
                                       : ink_r17_vd_ld4(xp1 + ((ink_r17_vd_u64)b << 5u) + 16ull);
          const float d1 = use_lds_x ? ink_r17_vd_fl_lds.xd[1u][b] : dap1[b];
          INK_R17_VD_ADD4(a1_0, a1_1, a1_2, a1_3, j_acc,
                          ink_r17_vd_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_1, x1_1, d1));

          if (nj_c > 2u) {
            const u32x4 x0_2 = use_lds_x ? *(const LDSP u32x4 *)(ink_r17_vd_fl_lds.xq[2u] + (b << 3u))
                                         : ink_r17_vd_ld4(xp2 + ((ink_r17_vd_u64)b << 5u));
            const u32x4 x1_2 = use_lds_x ? *(const LDSP u32x4 *)(ink_r17_vd_fl_lds.xq[2u] + (b << 3u) + 4u)
                                         : ink_r17_vd_ld4(xp2 + ((ink_r17_vd_u64)b << 5u) + 16ull);
            const float d2 = use_lds_x ? ink_r17_vd_fl_lds.xd[2u][b] : dap2[b];
            INK_R17_VD_ADD4(a2_0, a2_1, a2_2, a2_3, j_acc,
                            ink_r17_vd_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_2, x1_2, d2));

            if (nj_c > 3u) {
              const u32x4 x0_3 = use_lds_x ? *(const LDSP u32x4 *)(ink_r17_vd_fl_lds.xq[3u] + (b << 3u))
                                           : ink_r17_vd_ld4(xp3 + ((ink_r17_vd_u64)b << 5u));
              const u32x4 x1_3 = use_lds_x ? *(const LDSP u32x4 *)(ink_r17_vd_fl_lds.xq[3u] + (b << 3u) + 4u)
                                           : ink_r17_vd_ld4(xp3 + ((ink_r17_vd_u64)b << 5u) + 16ull);
              const float d3 = use_lds_x ? ink_r17_vd_fl_lds.xd[3u][b] : dap3[b];
              INK_R17_VD_ADD4(a3_0, a3_1, a3_2, a3_3, j_acc,
                              ink_r17_vd_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_3, x1_3, d3));
            }
          }
        }
        __builtin_amdgcn_sched_barrier(0);
        q_c = q_n; aux_c = aux_n; d_c = d_n;
      }

      const u32 r_loc = (sp << 4u) + (wv4 << 2u) + k;
      const float acc0 = ink_r17_vd_oct_tree_f((a0_0 + a0_2) + (a0_1 + a0_3));
      if (s == 0u) {
        if (is_up) ink_r17_vd_fl_lds.u[0u][r_loc] = acc0;
        else ink_r17_vd_fl_lds.g[0u][r_loc] = acc0;
      }
      if (nj_c > 1u) {
        const float acc1 = ink_r17_vd_oct_tree_f((a1_0 + a1_2) + (a1_1 + a1_3));
        if (s == 0u) {
          if (is_up) ink_r17_vd_fl_lds.u[1u][r_loc] = acc1;
          else ink_r17_vd_fl_lds.g[1u][r_loc] = acc1;
        }
      }
      if (nj_c > 2u) {
        const float acc2 = ink_r17_vd_oct_tree_f((a2_0 + a2_2) + (a2_1 + a2_3));
        if (s == 0u) {
          if (is_up) ink_r17_vd_fl_lds.u[2u][r_loc] = acc2;
          else ink_r17_vd_fl_lds.g[2u][r_loc] = acc2;
        }
      }
      if (nj_c > 3u) {
        const float acc3 = ink_r17_vd_oct_tree_f((a3_0 + a3_2) + (a3_1 + a3_3));
        if (s == 0u) {
          if (is_up) ink_r17_vd_fl_lds.u[3u][r_loc] = acc3;
          else ink_r17_vd_fl_lds.g[3u][r_loc] = acc3;
        }
      }
    }
    lbarrier();

    if (wv < nj_c) {
      const u32 pw = (wv == 0u) ? p0 : ((wv == 1u) ? p1 : ((wv == 2u) ? p2 : p3));
      const float gv = ink_r17_vd_fl_lds.g[wv][l];
      const float uv = ink_r17_vd_fl_lds.u[wv][l];
      const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
      const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
      const float v = gs * sigmoidf(gv) * us;
      if (b_out) ((G float *)((G u8 *)b_out + (ink_r17_vd_u64)pw * (ink_r17_vd_u64)bstr))[(blk << 5u) + l] = v;
      if (q_out) quant_wave(v, blk, q_out + (ink_r17_vd_u64)pw * (ink_r17_vd_u64)qstr, nb_out);
    }
    if (j0 + 4u < nj) lbarrier();
  }
}

/* ============================================================================
 * KERNEL 3: ie_inkling_rel_attn_ver_glob_q4_wmma_fast_r17
 *   Next-Gen 256K Global Q4_0 + FWHT-128 Verify (T=1..8) FP16 WMMA Attention
 *   + optional fused OP_GATE(sigmoid) + Q8_0.
 *   Architecture:
 *     - 4 Producer Waves (w=4..7, 128 threads, 2x MLP) + 4 Consumer Waves
 *       (w=0..3, pg = w >> 1, d_half = w & 1 -> 32 acc VGPRs + 32 Q VGPRs!).
 *     - 16,384 B exact LDS (4 WGs/CU on gfx1201):
 *         Mode 1 (Far-Field NB=32): k[32x128] (8 KiB) + vt[2][128x16] (8 KiB)
 *         Mode 2 (Near-Field NB=16): k[2][16x128] (8 KiB) + vt[2][128x16] (8 KiB)
 *         Phase 4 (Merge/Writeout): o_stg[8][128] + m/l + wt[8][128] (8,452 B)
 *     - Dedicated zero-rb, zero-modulo NB=32 far-field loop (99.6% of 256K keys)
 *       with gfx1201 split barriers (s_barrier_signal/wait), q_valid-gated
 *       ballot_w32 conditional rescale skip, and zero-tile PV skip.
 *     - Exact L1-cached f32 R*rel_bias inside NB=16 near-field tail (0 extra LDS).
 *   Exact 28-arg drop-in ABI of ie_inkling_rel_attn_ver_glob_q4_wmma_r15.
 *   Grid: (n_kv * nsplit, 1, 1), NT=256, 16,384 B LDS, <= 96 VGPR target.
 * ========================================================================== */

typedef union {
  struct {
    union {
      f16   k[INK_R17_VD_NB32 * INK_R17_VD_HD] __attribute__((aligned(16))); /* 8,192 B */
      float qtmp[4u][INK_R17_VD_HD]            __attribute__((aligned(16))); /* 2,048 B */
    } ku;
    f16 vt[2u][INK_R17_VD_HD * INK_R17_VD_NB16] __attribute__((aligned(16))); /* 8,192 B */
  } far32;                                                                     /* 16,384 B */
  struct {
    f16 k[2u][INK_R17_VD_NB16 * INK_R17_VD_HD]  __attribute__((aligned(16))); /* 8,192 B */
    f16 vt[2u][INK_R17_VD_HD * INK_R17_VD_NB16] __attribute__((aligned(16))); /* 8,192 B */
  } nf16;                                                                      /* 16,384 B */
  struct {
    float o_stg[8u][INK_R17_VD_HD] __attribute__((aligned(16)));               /* 4,096 B */
    float m[32u];                                                              /*   128 B */
    float l[32u];                                                              /*   128 B */
    float wt[8u][128u]             __attribute__((aligned(16)));               /* 4,096 B */
    u32   is_last;                                                             /*     4 B */
  } ph4;                                                                       /* 8,452 B */
} ink_r17_vd_gw_lds_t;

static LDS ink_r17_vd_gw_lds_t ink_r17_vd_gw_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r17_vd_gw_lds_t) == 16384u, "ink_r17_vd_gw_lds_t must be 16,384 B (4 WGs/CU)");

typedef struct {
  u32 kw[4];
  u16 ks[4];
} ink_r17_vd_raw_k32_t;

typedef struct {
  u16 vw[8];
  u16 vs[8];
} ink_r17_vd_raw_v16_t;

static inline __attribute__((always_inline)) u32 ink_r17_vd_k_off(u32 r, u32 c) {
  return (r << 7u) + ((c ^ (r & 15u)) << 3u);
}

static inline __attribute__((always_inline)) u32 ink_r17_vd_vt_off(u32 d, u32 h) {
  return ((2u * d + h) ^ ((d >> 2u) & 1u) ^ ((d >> 3u) & 7u)) << 3u;
}

static inline __attribute__((always_inline)) void ink_r17_vd_t8(
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

/* Exact IEEE-754 Q4_0 4-nibble dequant -> packed f16x4 (ink_r17_vd_u2), f16_mix=0. */
static inline __attribute__((always_inline)) ink_r17_vd_u2 ink_r17_vd_deq4n_h4(u32 w, u16 s_bits) {
  const f16 sh = __builtin_bit_cast(f16, s_bits);
  const ink_r17_vd_h4 h = {sh * (f16)((int)(w & 15u) - 8),
                           sh * (f16)((int)((w >> 4u) & 15u) - 8),
                           sh * (f16)((int)((w >> 8u) & 15u) - 8),
                           sh * (f16)((int)((w >> 12u) & 15u) - 8)};
  return __builtin_bit_cast(ink_r17_vd_u2, h);
}

/* 4-wave Producer K32 load (128 threads x 4 rows = 32 keys x 128 elems, no-wrap). */
static inline __attribute__((always_inline)) void ink_r17_vd_far_load_k32(
    ink_r17_vd_raw_k32_t *rk, const G u8 *kc_h, const ink_r17_vd_u64 row_step,
    const u32 kb, const u32 pt) {
  const u32 c = pt & 15u;
  const u32 r0 = pt >> 4u;
  const u32 blk_c = c >> 2u;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const G u8 *rec = kc_h + (ink_r17_vd_u64)(kb + r0 + (i << 3u)) * row_step;
    rk->kw[i] = ((const G u32 *)rec)[c];
    rk->ks[i] = ((const G u16 *)(rec + 64u))[blk_c];
  }
}

static inline __attribute__((always_inline)) void ink_r17_vd_far_pub_k32(
    LDSP f16 *k_lds, const ink_r17_vd_raw_k32_t *rk, const u32 pt) {
  const u32 c = pt & 15u;
  const u32 r0 = pt >> 4u;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r = r0 + (i << 3u);
    const u32 w8n = rk->kw[i];
    const u16 sb = rk->ks[i];
    const ink_r17_vd_u2 h0 = ink_r17_vd_deq4n_h4(w8n, sb);
    const ink_r17_vd_u2 h1 = ink_r17_vd_deq4n_h4(w8n >> 16u, sb);
    *(LDSP u32x4 *)(k_lds + ink_r17_vd_k_off(r, c)) = (u32x4){h0.x, h0.y, h1.x, h1.y};
  }
}

/* 4-wave Producer V32 load (waves 0..1 -> vt[0], waves 2..3 -> vt[1], 8 rows/thread). */
static inline __attribute__((always_inline)) void ink_r17_vd_far_load_v32(
    ink_r17_vd_raw_v16_t *rv, const G u8 *vc_h, const ink_r17_vd_u64 row_step,
    const u32 kb, const u32 vh, const u32 wi, const u32 L) {
  const u32 blk_v = L >> 3u;
  const u32 row0 = kb + (vh << 4u) + (wi << 3u);
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const G u8 *rec = vc_h + (ink_r17_vd_u64)(row0 + u) * row_step;
    rv->vw[u] = ((const G u16 *)rec)[L];
    rv->vs[u] = ((const G u16 *)(rec + 64u))[blk_v];
  }
}

static inline __attribute__((always_inline)) void ink_r17_vd_pub_vt16(
    LDSP f16 *vt_lds, const ink_r17_vd_raw_v16_t *rv, const u32 wi, const u32 L) {
  u32 vx0[8], vx1[8];
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const ink_r17_vd_u2 h4 = ink_r17_vd_deq4n_h4((u32)rv->vw[u], rv->vs[u]);
    vx0[u] = h4.x;
    vx1[u] = h4.y;
  }
  u32x4 ev0, od0, ev1, od1;
  ink_r17_vd_t8(vx0, &ev0, &od0);
  ink_r17_vd_t8(vx1, &ev1, &od1);
  const u32 c0 = L << 2u;
  *(LDSP u32x4 *)(vt_lds + ink_r17_vd_vt_off(c0 + 0u, wi)) = ev0;
  *(LDSP u32x4 *)(vt_lds + ink_r17_vd_vt_off(c0 + 1u, wi)) = od0;
  *(LDSP u32x4 *)(vt_lds + ink_r17_vd_vt_off(c0 + 2u, wi)) = ev1;
  *(LDSP u32x4 *)(vt_lds + ink_r17_vd_vt_off(c0 + 3u, wi)) = od1;
}

/* Near-field NB=16 split-role 4-wave Producer (waves 0..1 load/pub V16, waves 2..3 load/pub K16). */
static inline __attribute__((always_inline)) void ink_r17_vd_nf_load_k16(
    ink_r17_vd_raw_k32_t *rk, const G u8 *kc_h, const ink_r17_vd_u64 row_step,
    const u32 kv_len, const u32 kb, const u32 p1, const int no_wrap, const u32 pt2) {
  const u32 c = pt2 & 15u;
  const u32 r0 = pt2 >> 4u; /* 0..3 */
  const u32 blk_c = c >> 2u;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 kp = kb + r0 + (i << 2u);
    if (kp < p1) {
      const u32 kvp = no_wrap ? kp : (kp % kv_len);
      const G u8 *rec = kc_h + (ink_r17_vd_u64)kvp * row_step;
      rk->kw[i] = ((const G u32 *)rec)[c];
      rk->ks[i] = ((const G u16 *)(rec + 64u))[blk_c];
    } else {
      rk->kw[i] = 0x88888888u;
      rk->ks[i] = 0u;
    }
  }
}

static inline __attribute__((always_inline)) void ink_r17_vd_nf_pub_k16(
    LDSP f16 *k_lds, const ink_r17_vd_raw_k32_t *rk, const u32 pt2) {
  const u32 c = pt2 & 15u;
  const u32 r0 = pt2 >> 4u; /* 0..3 */
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r = r0 + (i << 2u);
    const u32 w8n = rk->kw[i];
    const u16 sb = rk->ks[i];
    const ink_r17_vd_u2 h0 = ink_r17_vd_deq4n_h4(w8n, sb);
    const ink_r17_vd_u2 h1 = ink_r17_vd_deq4n_h4(w8n >> 16u, sb);
    *(LDSP u32x4 *)(k_lds + ink_r17_vd_k_off(r, c)) = (u32x4){h0.x, h0.y, h1.x, h1.y};
  }
}

static inline __attribute__((always_inline)) void ink_r17_vd_nf_load_v16(
    ink_r17_vd_raw_v16_t *rv, const G u8 *vc_h, const ink_r17_vd_u64 row_step,
    const u32 kv_len, const u32 kb, const u32 p1, const int no_wrap, const u32 wi, const u32 L) {
  const u32 blk_v = L >> 3u;
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const u32 kp = kb + (wi << 3u) + u;
    if (kp < p1) {
      const u32 kvp = no_wrap ? kp : (kp % kv_len);
      const G u8 *rec = vc_h + (ink_r17_vd_u64)kvp * row_step;
      rv->vw[u] = ((const G u16 *)rec)[L];
      rv->vs[u] = ((const G u16 *)(rec + 64u))[blk_v];
    } else {
      rv->vw[u] = 0x8888u;
      rv->vs[u] = 0u;
    }
  }
}

/* Consumer QK16 building blocks (2-deep, 1-deep+signal, 1-deep). */
static inline __attribute__((always_inline)) ink_r17_vd_v8f ink_r17_vd_qk16_2d(
    const LDSP f16 *k_lds, const ink_r17_vd_h8 qa[8]) {
  const u32 l_loc = ink_r17_vd_lane_pin();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  u32 kb_b = (rl << 8u) | ((rl ^ hl) << 4u);
  const LDSP u8 *kp = (const LDSP u8 *)k_lds;

  ink_r17_vd_v8f sa = (ink_r17_vd_v8f)(0.0f);
  ink_r17_vd_h8 k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); kb_b ^= 32u;
  ink_r17_vd_h8 k1 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); kb_b ^= 96u;
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[0], sa);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[1], sa);
  k1 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); kb_b ^= 224u;
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[2], sa);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[3], sa);
  k1 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); kb_b ^= 96u;
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[4], sa);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[5], sa);
  k1 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b);
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[6], sa);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[7], sa);
  __builtin_amdgcn_sched_barrier(0);
  return sa;
}

static inline __attribute__((always_inline)) ink_r17_vd_v8f ink_r17_vd_qk16_s1_sig(
    const LDSP f16 *k_lds, const ink_r17_vd_h8 qa[8]) {
  const u32 l_loc = ink_r17_vd_lane_pin();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  u32 kb_b = (rl << 8u) | ((rl ^ hl) << 4u);
  const LDSP u8 *kp = (const LDSP u8 *)k_lds;

  ink_r17_vd_v8f sa = (ink_r17_vd_v8f)(0.0f);
  ink_r17_vd_h8 k0;
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[0], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[1], sa); kb_b ^= 96u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[2], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[3], sa); kb_b ^= 224u; __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[4], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[5], sa); kb_b ^= 96u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[6], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b);
  ink_r17_vd_lsignal(); /* K32 LDS reads complete -> release K buffer before final WMMA! */
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[7], sa);
  __builtin_amdgcn_sched_barrier(0);
  return sa;
}

static inline __attribute__((always_inline)) ink_r17_vd_v8f ink_r17_vd_qk16_1d(
    const LDSP f16 *k_lds, const ink_r17_vd_h8 qa[8]) {
  const u32 l_loc = ink_r17_vd_lane_pin();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  u32 kb_b = (rl << 8u) | ((rl ^ hl) << 4u);
  const LDSP u8 *kp = (const LDSP u8 *)k_lds;

  ink_r17_vd_v8f sa = (ink_r17_vd_v8f)(0.0f);
  ink_r17_vd_h8 k0;
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[0], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[1], sa); kb_b ^= 96u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[2], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[3], sa); kb_b ^= 224u; __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[4], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[5], sa); kb_b ^= 96u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[6], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP ink_r17_vd_h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[7], sa); __builtin_amdgcn_sched_barrier(0);
  return sa;
}

/* 4-tile PV16 (64 output dims per wave: d_half==0 -> dims 0..63, d_half==1 -> dims 64..127). */
static inline __attribute__((always_inline)) void ink_r17_vd_pv16_2d_4t(
    const LDSP f16 *vt_lds, const ink_r17_vd_h8 pb, ink_r17_vd_v8f o[4],
    const u32 d_half, const int do_signal) {
  const u32 l_loc = ink_r17_vd_lane_pin();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2u) & 1u) ^ (rl >> 3u);
  const u32 vtb = vt_lane << 4u;
  const LDSP u8 *vp = (const LDSP u8 *)vt_lds + (d_half << 11u);

  ink_r17_vd_h8 v0 = *(const LDSP ink_r17_vd_h8 *)(vp +    0u + vtb);
  ink_r17_vd_h8 v1 = *(const LDSP ink_r17_vd_h8 *)(vp +  512u + (vtb ^ 32u));
  __builtin_amdgcn_sched_barrier(0);

  o[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[0]);
  v0 = *(const LDSP ink_r17_vd_h8 *)(vp + 1024u + (vtb ^ 64u));
  __builtin_amdgcn_sched_barrier(0);

  o[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[1]);
  v1 = *(const LDSP ink_r17_vd_h8 *)(vp + 1536u + (vtb ^ 96u));
  if (do_signal) {
    ink_r17_vd_lsignal(); /* VT LDS reads complete -> signal before final 2 PV WMMAs! */
  } else {
    __builtin_amdgcn_sched_barrier(0);
  }

  o[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[2]);
  o[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[3]);
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r17_vd_pv16_1d_4t(
    const LDSP f16 *vt_lds, const ink_r17_vd_h8 pb, ink_r17_vd_v8f o[4],
    const u32 d_half, const int do_signal) {
  const u32 l_loc = ink_r17_vd_lane_pin();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2u) & 1u) ^ (rl >> 3u);
  const u32 vtb = vt_lane << 4u;
  const LDSP u8 *vp = (const LDSP u8 *)vt_lds + (d_half << 11u);

  ink_r17_vd_h8 v0;
  v0 = *(const LDSP ink_r17_vd_h8 *)(vp +    0u + vtb);         o[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[0]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP ink_r17_vd_h8 *)(vp +  512u + (vtb ^ 32u)); o[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[1]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP ink_r17_vd_h8 *)(vp + 1024u + (vtb ^ 64u)); o[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[2]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP ink_r17_vd_h8 *)(vp + 1536u + (vtb ^ 96u));
  if (do_signal) {
    ink_r17_vd_lsignal();
  } else {
    __builtin_amdgcn_sched_barrier(0);
  }
  o[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[3]);
  __builtin_amdgcn_sched_barrier(0);
}

/* Far-field NB=32 consumer step with q_valid-gated ballot_w32 rescale skip & zero-tile PV skip. */
static inline __attribute__((always_inline)) void ink_r17_vd_far_blk32_step(
    const LDSP f16 *k_lds, const LDSP f16 *vt0, const LDSP f16 *vt1,
    const ink_r17_vd_h8 qa[8], const u32 d_half, const int q_valid, const int has_next,
    float *m_p, float *ls_p, ink_r17_vd_v8f o[4]) {
  float m = *m_p;
  float ls = *ls_p;

  /* Half 0: QK_0 + q_valid-gated conditional-rescale softmax -> frees s0 before QK_1! */
  const ink_r17_vd_v8f s0 = ink_r17_vd_qk16_2d(k_lds, qa);
  ink_r17_vd_h8 pb0 = (ink_r17_vd_h8)(0.0f);
  int active0 = 0;
  {
    float bm0 = s0[0];
#pragma unroll
    for (u32 v = 1u; v < 8u; v++) bm0 = __builtin_fmaxf(bm0, s0[v]);
    bm0 = __builtin_fmaxf(bm0, ink_r17_vd_xrow(bm0));
    active0 = (__builtin_amdgcn_ballot_w32(q_valid && bm0 > -1e30f && !(bm0 - m < -30.0f)) != 0u);
    if (active0) {
      const int nr0 = (__builtin_amdgcn_ballot_w32(q_valid && bm0 > -1e30f && (bm0 > m || m <= -1e30f)) != 0u);
      const float mn0 = (q_valid && nr0) ? __builtin_fmaxf(m, bm0) : m;
      float ps0 = 0.0f;
      if (q_valid && mn0 > -1e30f) {
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) {
          const float p0 = __builtin_amdgcn_exp2f(s0[v] - mn0);
          ps0 += p0;
          pb0[v] = (f16)p0;
        }
      }
      float c0 = 1.0f;
      if (nr0) {
        c0 = (q_valid && m > -1e30f) ? __builtin_amdgcn_exp2f(m - mn0) : 0.0f;
#pragma unroll
        for (u32 nt = 0u; nt < 4u; nt++)
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c0;
      }
      ls = ls * c0 + ps0;
      m = mn0;
    }
  }
  __builtin_amdgcn_sched_barrier(0);

  /* Half 1: QK_1 (signals K32 consumed before final WMMA) + softmax -> frees s1! */
  const ink_r17_vd_v8f s1 = ink_r17_vd_qk16_s1_sig(k_lds + 2048u, qa);
  ink_r17_vd_h8 pb1 = (ink_r17_vd_h8)(0.0f);
  float c1 = 1.0f;
  int active1 = 0, nr1 = 0;
  {
    float bm1 = s1[0];
#pragma unroll
    for (u32 v = 1u; v < 8u; v++) bm1 = __builtin_fmaxf(bm1, s1[v]);
    bm1 = __builtin_fmaxf(bm1, ink_r17_vd_xrow(bm1));
    active1 = (__builtin_amdgcn_ballot_w32(q_valid && bm1 > -1e30f && !(bm1 - m < -30.0f)) != 0u);
    if (active1) {
      nr1 = (__builtin_amdgcn_ballot_w32(q_valid && bm1 > -1e30f && (bm1 > m || m <= -1e30f)) != 0u);
      const float mn1 = (q_valid && nr1) ? __builtin_fmaxf(m, bm1) : m;
      float ps1 = 0.0f;
      if (q_valid && mn1 > -1e30f) {
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) {
          const float p1 = __builtin_amdgcn_exp2f(s1[v] - mn1);
          ps1 += p1;
          pb1[v] = (f16)p1;
        }
      }
      if (nr1) c1 = (q_valid && m > -1e30f) ? __builtin_amdgcn_exp2f(m - mn1) : 0.0f;
      ls = ls * c1 + ps1;
      m = mn1;
    }
  }
  *m_p = m;
  *ls_p = ls;

  ink_r17_vd_lwait(); /* Wait for VT[0..1] ready (both QKs + softmaxes overlapped!) */

  if (active0) ink_r17_vd_pv16_2d_4t(vt0, pb0, o, d_half, 0);
  if (active1) {
    if (nr1) {
#pragma unroll
      for (u32 nt = 0u; nt < 4u; nt++)
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c1;
    }
    ink_r17_vd_pv16_2d_4t(vt1, pb1, o, d_half, has_next);
  }
  if (has_next) {
    if (!active1) ink_r17_vd_lsignal();
    ink_r17_vd_lwait();
  }
}

/* Zero-LDS, zero-loop-carried-VGPR near-field causal mask + exact L1-cached f32 R * rel_bias. */
static inline __attribute__((always_inline)) float ink_r17_vd_nf_adj(
    float sv, const G float *R, const G float *rel_bias, const u32 rs,
    const u32 pg, const u32 nq_tot, const u32 h0, const u32 pos0,
    const u32 kb, const u32 p1, const u32 v_idx, const u32 rel_extent) {
  const u32 l_loc = ink_r17_vd_lane_pin();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 qi = (pg << 4u) + rl;
  if (qi >= nq_tot) return -__builtin_inff();
  const u32 tj = qi / INK_R17_VD_GQA;
  const u32 pos_q = pos0 + tj;
  const u32 kp = kb + (hl << 3u) + v_idx;
  if (kp >= p1 || kp > pos_q) return -__builtin_inff();
  if (rel_extent > 0u && R && rel_bias) {
    const u32 dist = pos_q - kp;
    if (dist < rel_extent) {
      const u32 h_q = h0 + (qi - tj * INK_R17_VD_GQA);
      const G f32x4 *rp = (const G f32x4 *)((const G float *)((const G u8 *)R + (ink_r17_vd_u64)tj * (ink_r17_vd_u64)rs) +
                                            (ink_r17_vd_u64)h_q * 16ull);
      const G f32x4 *bp = (const G f32x4 *)(rel_bias + (ink_r17_vd_u64)dist * 16ull);
      float rbs = 0.0f;
#pragma clang loop unroll(disable)
      for (u32 d = 0u; d < 4u; d++) {
        const f32x4 rv = rp[d];
        const f32x4 bv = bp[d];
        rbs = __builtin_fmaf(rv.x, bv.x, rbs);
        rbs = __builtin_fmaf(rv.y, bv.y, rbs);
        rbs = __builtin_fmaf(rv.z, bv.z, rbs);
        rbs = __builtin_fmaf(rv.w, bv.w, rbs);
      }
      sv = __builtin_fmaf(rbs, INK_R17_VD_LOG2E, sv);
    }
  }
  return sv;
}

/* Wave-level split-K merge + optional inverse FWHT-128 + optional OP_GATE(sigmoid)+Q8_0. */
static inline __attribute__((always_inline)) void ink_r17_vd_gw_merge_head(
    const G float *ph, const float sink_val, const u32 nsplit, const u32 l,
    LDSP float *wt_w, const G float *gt_h, G float *ot_h, G u8 *oqt,
    u32 h, u32 n_head, u32 rot_flags) {
  float mv = (l == 0u) ? sink_val : -__builtin_inff();
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const ink_r17_vd_u64 off = (ink_r17_vd_u64)sp * INK_R17_VD_PSTR;
    if (ph[off + 1u] > 0.0f) mv = __builtin_fmaxf(mv, ph[off]);
  }
#pragma unroll
  for (u32 mask = 16u; mask >= 1u; mask >>= 1u) mv = __builtin_fmaxf(mv, shfl_xor_f(mv, mask));
  const float M2 = ink_r17_vd_sgpr_f(mv);

  float lv = 0.0f;
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const ink_r17_vd_u64 off = (ink_r17_vd_u64)sp * INK_R17_VD_PSTR;
    const float lp = ph[off + 1u];
    const float wt = (lp > 0.0f) ? __builtin_expf(ph[off] - M2) : 0.0f;
    if (sp < 128u) wt_w[sp] = wt;
    lv += lp * wt;
  }
  const float L2 = ink_r17_vd_sgpr_f(wave_sum_all(lv) + __builtin_expf(sink_val - M2));
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");

  f32x4 acc = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  for (u32 sp = 0u; sp < nsplit; sp++) {
    const float wt = (sp < 128u) ? wt_w[sp]
                                 : ((ph[(ink_r17_vd_u64)sp * INK_R17_VD_PSTR + 1u] > 0.0f)
                                        ? __builtin_expf(ph[(ink_r17_vd_u64)sp * INK_R17_VD_PSTR] - M2)
                                        : 0.0f);
    if (wt > 0.0f) {
      const G float *po = ph + (ink_r17_vd_u64)sp * INK_R17_VD_PSTR + 2u + (l << 2u);
      const f32x4 ov = {po[0], po[1], po[2], po[3]};
      acc += ov * wt;
    }
  }
  f32x4 ov = acc * (1.0f / L2);
  if ((rot_flags & 2u) != 0u) ov = ink_r17_vd_fwht128_wave4(ov, l);
  ink_r17_vd_gate_q8_head_vec4(ov, gt_h, ot_h, oqt, h, n_head, l);
}

KERNEL ie_inkling_rel_attn_ver_glob_q4_wmma_fast_r17(
    const G float *q, const G u8 *kc_q4, const G u8 *vc_q4,
    const G float *R, const G float *rel_bias, const G float *sink,
    G float *part, G float *out, u32 pos0, u32 T, u32 kv_len, u32 hd,
    u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
    u32 qs, u32 rs, u32 ss, u32 os, G u32 *count,
    const G float *agate, G u8 *oq, u32 gs, u32 oqs, u32 rot_flags) {
  if (T == 0u || T > 8u || win != 0u || hd != INK_R17_VD_HD || d_rel != INK_R17_VD_DREL ||
      n_kv == 0u || n_head != INK_R17_VD_GQA * n_kv || kv_len == 0u || nsplit == 0u ||
      !q || !kc_q4 || !vc_q4 || !sink)
    return;
  if (nsplit > 1u && (!part || !count)) return;
  if (nsplit == 1u && !out && !oq) return;

  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() / nsplit));
  const u32 s = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() % nsplit));
  if (kh >= n_kv) return;

  const u32 h0 = (u32)__builtin_amdgcn_readfirstlane((int)(kh * INK_R17_VD_GQA));
  const ink_r17_vd_u64 row_step = (ink_r17_vd_u64)n_kv * INK_R17_VD_RECB;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));

  const u32 nq_tot = (u32)__builtin_amdgcn_readfirstlane((int)(T * INK_R17_VD_GQA));
  const u32 npg = (u32)__builtin_amdgcn_readfirstlane((int)(nq_tot > 16u ? 2u : 1u));
  const u32 np = (u32)__builtin_amdgcn_readfirstlane((int)(pos0 + T));
  const u32 ch = (u32)__builtin_amdgcn_readfirstlane((int)((np + nsplit - 1u) / nsplit));
  const u32 off0 = (u32)__builtin_amdgcn_readfirstlane((int)(s * ch));
  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)((off0 < np) ? off0 : np));
  const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)((off0 < np && off0 + ch < np) ? (off0 + ch) : np));

  if (nsplit > 1u && p0 >= p1) {
    const u32 l_e = ink_r17_vd_lane_pin();
    for (u32 qi = w; qi < nq_tot; qi += 8u) {
      const u32 t = qi / INK_R17_VD_GQA;
      const u32 h = h0 + (qi - t * INK_R17_VD_GQA);
      G float *pp = part + (ink_r17_vd_u64)((t * n_head + h) * nsplit + s) * INK_R17_VD_PSTR;
      if (l_e == 0u) { pp[0] = -1e30f; pp[1] = 0.0f; }
      G float *po = pp + 2u + (l_e << 2u);
      po[0] = 0.0f; po[1] = 0.0f; po[2] = 0.0f; po[3] = 0.0f;
    }
  } else {
    /* Partition [p0, p1) into exact no-wrap zero-rb far-field [p0, p_far) in NB=32 blocks
     * and near-field / tail [p_far, p1) in NB=16 blocks. */
    const u32 lim_caus = (pos0 + 1u > rel_extent) ? (pos0 + 1u - rel_extent) : 0u;
    const u32 lim_nw = (lim_caus < kv_len) ? lim_caus : kv_len;
    const u32 lim_far = (lim_nw < p1) ? lim_nw : p1;
    const u32 nblk32 = (u32)__builtin_amdgcn_readfirstlane((int)((lim_far > p0) ? ((lim_far - p0) >> 5u) : 0u));
    const u32 p_far = (u32)__builtin_amdgcn_readfirstlane((int)(p0 + (nblk32 << 5u)));
    const u32 nblk16 = (u32)__builtin_amdgcn_readfirstlane((int)((p1 > p_far) ? ((p1 - p_far + 15u) >> 4u) : 0u));
    const int no_wrap = (int)__builtin_amdgcn_readfirstlane((int)(p1 <= kv_len));

    const G u8 *kc_h = kc_q4 + (ink_r17_vd_u64)kh * INK_R17_VD_RECB;
    const G u8 *vc_h = vc_q4 + (ink_r17_vd_u64)kh * INK_R17_VD_RECB;

    ink_r17_vd_v8f o[4] = {(ink_r17_vd_v8f)(0.0f), (ink_r17_vd_v8f)(0.0f),
                           (ink_r17_vd_v8f)(0.0f), (ink_r17_vd_v8f)(0.0f)};
    float m = -__builtin_inff();
    float ls = 0.0f;

    if (w >= 4u) {
      /* ==================================================================
       * 4-Wave Producer (w=4..7, pt=0..127, 128 threads -> 2x MLP)
       * ================================================================ */
      const u32 pt = tid() - 128u;

      /* Prime Phase-A block 0 raw Q4 K ahead of Sync_Prologue to overlap Q FWHT-128! */
      ink_r17_vd_raw_k32_t rk0 = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};
      if (nblk32 > 0u) {
        ink_r17_vd_far_load_k32(&rk0, kc_h, row_step, p0, pt);
      }
      lbarrier(); /* Sync_Prologue: Consumers finished Q load + FWHT-128! */

      /* Phase A Producer: NB=32 Pure Far-Field (0 modulo, 0 bounds, 0 rb). */
      if (nblk32 > 0u) {
        const u32 pw = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
        const u32 vh = pw >> 1u;
        const u32 wi = pw & 1u;
        const u32 L = pt & 31u;
        ink_r17_vd_raw_v16_t rv;

        ink_r17_vd_far_pub_k32(ink_r17_vd_gw_lds.far32.ku.k, &rk0, pt);
        ink_r17_vd_lsignal();
        ink_r17_vd_far_load_v32(&rv, vc_h, row_step, p0, vh, wi, L);
        if (1u < nblk32) ink_r17_vd_far_load_k32(&rk0, kc_h, row_step, p0 + 32u, pt);
        ink_r17_vd_lwait();

#pragma clang loop unroll(disable)
        for (u32 b = 0u; b < nblk32; b++) {
          const u32 b_u = (u32)__builtin_amdgcn_readfirstlane((int)b);
          ink_r17_vd_pub_vt16(ink_r17_vd_gw_lds.far32.vt[vh], &rv, wi, L);
          ink_r17_vd_lsignal();
          if (b_u + 1u < nblk32) {
            ink_r17_vd_far_load_v32(&rv, vc_h, row_step, p0 + ((b_u + 1u) << 5u), vh, wi, L);
          }
          ink_r17_vd_lwait();

          if (b_u + 1u < nblk32) {
            ink_r17_vd_far_pub_k32(ink_r17_vd_gw_lds.far32.ku.k, &rk0, pt);
            ink_r17_vd_lsignal();
            if (b_u + 2u < nblk32) {
              ink_r17_vd_far_load_k32(&rk0, kc_h, row_step, p0 + ((b_u + 2u) << 5u), pt);
            }
            ink_r17_vd_lwait();
          }
        }
        lbarrier(); /* End Phase A: Consumers finished final VT read! */
      }

      /* Phase B Producer: Double-buffered NB=16 Near-Field / Tail (waves 4..5 V16, waves 6..7 K16). */
      if (nblk16 > 0u) {
        const u32 pt2 = pt & 63u;
        if (w >= 6u) {
          ink_r17_vd_raw_k32_t rk0_n;
          ink_r17_vd_raw_k32_t rk1_n = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};
          ink_r17_vd_nf_load_k16(&rk0_n, kc_h, row_step, kv_len, p_far, p1, no_wrap, pt2);
          if (1u < nblk16) ink_r17_vd_nf_load_k16(&rk1_n, kc_h, row_step, kv_len, p_far + 16u, p1, no_wrap, pt2);

          ink_r17_vd_nf_pub_k16(ink_r17_vd_gw_lds.nf16.k[0u], &rk0_n, pt2);
          ink_r17_vd_lsignal();
          if (2u < nblk16) ink_r17_vd_nf_load_k16(&rk0_n, kc_h, row_step, kv_len, p_far + 32u, p1, no_wrap, pt2);
          ink_r17_vd_lwait();

#pragma clang loop unroll(disable)
          for (u32 k = 0u; k + 1u < nblk16; k += 2u) {
            {
              const u32 kn = k + 1u;
              ink_r17_vd_nf_pub_k16(ink_r17_vd_gw_lds.nf16.k[1u], &rk1_n, pt2);
              ink_r17_vd_lsignal();
              if (kn + 2u < nblk16) {
                ink_r17_vd_nf_load_k16(&rk1_n, kc_h, row_step, kv_len, p_far + ((kn + 2u) << 4u), p1, no_wrap, pt2);
              }
              ink_r17_vd_lwait();
            }
            if (k + 2u < nblk16) {
              const u32 kn = k + 2u;
              ink_r17_vd_nf_pub_k16(ink_r17_vd_gw_lds.nf16.k[0u], &rk0_n, pt2);
              ink_r17_vd_lsignal();
              if (kn + 2u < nblk16) {
                ink_r17_vd_nf_load_k16(&rk0_n, kc_h, row_step, kv_len, p_far + ((kn + 2u) << 4u), p1, no_wrap, pt2);
              }
              ink_r17_vd_lwait();
            }
          }
        } else {
          const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt2 >> 5u));
          const u32 L = pt2 & 31u;
          ink_r17_vd_raw_v16_t rv0_n;
          ink_r17_vd_raw_v16_t rv1_n = {{0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}};
          ink_r17_vd_nf_load_v16(&rv0_n, vc_h, row_step, kv_len, p_far, p1, no_wrap, wi, L);
          if (1u < nblk16) ink_r17_vd_nf_load_v16(&rv1_n, vc_h, row_step, kv_len, p_far + 16u, p1, no_wrap, wi, L);

          ink_r17_vd_pub_vt16(ink_r17_vd_gw_lds.nf16.vt[0u], &rv0_n, wi, L);
          ink_r17_vd_lsignal();
          if (2u < nblk16) ink_r17_vd_nf_load_v16(&rv0_n, vc_h, row_step, kv_len, p_far + 32u, p1, no_wrap, wi, L);
          ink_r17_vd_lwait();

#pragma clang loop unroll(disable)
          for (u32 k = 0u; k + 1u < nblk16; k += 2u) {
            {
              const u32 kn = k + 1u;
              ink_r17_vd_pub_vt16(ink_r17_vd_gw_lds.nf16.vt[1u], &rv1_n, wi, L);
              ink_r17_vd_lsignal();
              if (kn + 2u < nblk16) {
                ink_r17_vd_nf_load_v16(&rv1_n, vc_h, row_step, kv_len, p_far + ((kn + 2u) << 4u), p1, no_wrap, wi, L);
              }
              ink_r17_vd_lwait();
            }
            if (k + 2u < nblk16) {
              const u32 kn = k + 2u;
              ink_r17_vd_pub_vt16(ink_r17_vd_gw_lds.nf16.vt[0u], &rv0_n, wi, L);
              ink_r17_vd_lsignal();
              if (kn + 2u < nblk16) {
                ink_r17_vd_nf_load_v16(&rv0_n, vc_h, row_step, kv_len, p_far + ((kn + 2u) << 4u), p1, no_wrap, wi, L);
              }
              ink_r17_vd_lwait();
            }
          }
        }
        lbarrier(); /* End Phase B: Consumers finished final nf16 read! */
      }
    } else {
      /* ==================================================================
       * 4-Wave Consumer (w=0..3, pg = w >> 1, d_half = w & 1)
       *   Each active wave holds 16 queries (qa[8]=32V) + 64 output dims (o[4]=32V)!
       * ================================================================ */
      const u32 pg = w >> 1u;
      const u32 d_half = w & 1u;

      if (pg >= npg) {
        /* Idle consumer pair when T <= 5 (npg == 1): lightweight barrier follower only. */
        lbarrier(); /* Sync_Prologue */
        if (nblk32 > 0u) {
          const u32 nbar = nblk32 << 1u;
#pragma clang loop unroll(disable)
          for (u32 i = 0u; i < nbar; i++) {
            ink_r17_vd_lsignal();
            ink_r17_vd_lwait();
          }
          lbarrier(); /* End Phase A */
        }
        if (nblk16 > 0u) {
#pragma clang loop unroll(disable)
          for (u32 i = 0u; i < nblk16; i++) {
            ink_r17_vd_lsignal();
            ink_r17_vd_lwait();
          }
          lbarrier(); /* End Phase B */
        }
      } else {
        /* Load 16 queries of pg (with optional FWHT-128 via wave-private far32.ku.qtmp[w]). */
        ink_r17_vd_h8 qa[8];
        {
          const u32 l_q = ink_r17_vd_lane_pin();
          const u32 rl_q = l_q & 15u, hl_q = l_q >> 4u;
          const u32 qi_r = (pg << 4u) + rl_q;
          if ((rot_flags & 1u) != 0u) {
#pragma unroll
            for (u32 ks_i = 0u; ks_i < 8u; ks_i++) qa[ks_i] = (ink_r17_vd_h8)(0.0f);
            LDSP f32x4 *qtmp4 = (LDSP f32x4 *)ink_r17_vd_gw_lds.far32.ku.qtmp[w];
            const u32 nq_pg = (nq_tot > (pg << 4u)) ? ((nq_tot - (pg << 4u) < 16u) ? (nq_tot - (pg << 4u)) : 16u) : 0u;
#pragma clang loop unroll(disable)
            for (u32 qi_loc = 0u; qi_loc < nq_pg; qi_loc++) {
              const u32 qi = (pg << 4u) + qi_loc;
              const u32 tj = qi / INK_R17_VD_GQA;
              const u32 gi = qi - tj * INK_R17_VD_GQA;
              const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (ink_r17_vd_u64)tj * (ink_r17_vd_u64)qs) +
                                                     (ink_r17_vd_u64)(h0 + gi) * INK_R17_VD_HD);
              const f32x4 qr = ink_r17_vd_fwht128_wave4(qp4[l_q], l_q);
              qtmp4[l_q] = qr;
              __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
              if (rl_q == qi_loc) {
#pragma unroll
                for (u32 ks_i = 0u; ks_i < 8u; ks_i++) {
                  const f32x4 v0 = qtmp4[(ks_i << 2u) + (hl_q << 1u) + 0u] * INK_R17_VD_QSC;
                  const f32x4 v1 = qtmp4[(ks_i << 2u) + (hl_q << 1u) + 1u] * INK_R17_VD_QSC;
                  qa[ks_i] = (ink_r17_vd_h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w,
                                             (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
                }
              }
              __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
            }
          } else {
            if (qi_r < nq_tot) {
              const u32 tj = qi_r / INK_R17_VD_GQA;
              const u32 gi = qi_r - tj * INK_R17_VD_GQA;
              const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (ink_r17_vd_u64)tj * (ink_r17_vd_u64)qs) +
                                                     (ink_r17_vd_u64)(h0 + gi) * INK_R17_VD_HD);
#pragma unroll
              for (u32 ks_i = 0u; ks_i < 8u; ks_i++) {
                const f32x4 v0 = qp4[(ks_i << 2u) + (hl_q << 1u) + 0u] * INK_R17_VD_QSC;
                const f32x4 v1 = qp4[(ks_i << 2u) + (hl_q << 1u) + 1u] * INK_R17_VD_QSC;
                qa[ks_i] = (ink_r17_vd_h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w,
                                           (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
                __builtin_amdgcn_sched_barrier(0);
              }
            } else {
#pragma unroll
              for (u32 ks_i = 0u; ks_i < 8u; ks_i++) qa[ks_i] = (ink_r17_vd_h8)(0.0f);
            }
          }
        }
#pragma unroll
        for (u32 nt = 0u; nt < 4u; nt++) o[nt] = (ink_r17_vd_v8f)(0.0f);

        lbarrier(); /* Sync_Prologue: release qtmp before Producer writes far32.ku.k / nf16.k! */

        const int q_valid = (((pg << 4u) + (ink_r17_vd_lane_pin() & 15u)) < nq_tot);

        /* Phase A Consumer: Pure Far-Field NB=32 loop (0 causal mask, 0 rb). */
        if (nblk32 > 0u) {
          ink_r17_vd_lsignal();
          ink_r17_vd_lwait(); /* Wait for K32 block 0 ready! */

#pragma clang loop unroll(disable)
          for (u32 b = 0u; b < nblk32; b++) {
            const u32 b_u = (u32)__builtin_amdgcn_readfirstlane((int)b);
            ink_r17_vd_far_blk32_step(ink_r17_vd_gw_lds.far32.ku.k,
                                      ink_r17_vd_gw_lds.far32.vt[0u],
                                      ink_r17_vd_gw_lds.far32.vt[1u],
                                      qa, d_half, q_valid, (b_u + 1u < nblk32),
                                      &m, &ls, o);
          }
          lbarrier(); /* End Phase A: release far32 LDS! */
        }

        /* Phase B Consumer: Double-Buffered NB=16 Near-Field / Tail with exact f32 R*rel_bias. */
        if (nblk16 > 0u) {
          ink_r17_vd_lsignal();
          ink_r17_vd_lwait(); /* Wait for nf16 bf=0 ready! */

#pragma clang loop unroll(disable)
          for (u32 k = 0u; k < nblk16; k++) {
            const u32 k_u = (u32)__builtin_amdgcn_readfirstlane((int)k);
            const u32 bf = k_u & 1u;
            const u32 kb = (u32)__builtin_amdgcn_readfirstlane((int)(p_far + (k_u << 4u)));
            const int has_next = (k_u + 1u < nblk16);

            const ink_r17_vd_v8f sa_v = ink_r17_vd_qk16_1d(ink_r17_vd_gw_lds.nf16.k[bf], qa);

            float s0 = ink_r17_vd_nf_adj(sa_v[0], R, rel_bias, rs, pg, nq_tot, h0, pos0, kb, p1, 0u, rel_extent); __builtin_amdgcn_sched_barrier(0);
            float s1 = ink_r17_vd_nf_adj(sa_v[1], R, rel_bias, rs, pg, nq_tot, h0, pos0, kb, p1, 1u, rel_extent); __builtin_amdgcn_sched_barrier(0);
            float s2 = ink_r17_vd_nf_adj(sa_v[2], R, rel_bias, rs, pg, nq_tot, h0, pos0, kb, p1, 2u, rel_extent); __builtin_amdgcn_sched_barrier(0);
            float s3 = ink_r17_vd_nf_adj(sa_v[3], R, rel_bias, rs, pg, nq_tot, h0, pos0, kb, p1, 3u, rel_extent); __builtin_amdgcn_sched_barrier(0);
            float s4 = ink_r17_vd_nf_adj(sa_v[4], R, rel_bias, rs, pg, nq_tot, h0, pos0, kb, p1, 4u, rel_extent); __builtin_amdgcn_sched_barrier(0);
            float s5 = ink_r17_vd_nf_adj(sa_v[5], R, rel_bias, rs, pg, nq_tot, h0, pos0, kb, p1, 5u, rel_extent); __builtin_amdgcn_sched_barrier(0);
            float s6 = ink_r17_vd_nf_adj(sa_v[6], R, rel_bias, rs, pg, nq_tot, h0, pos0, kb, p1, 6u, rel_extent); __builtin_amdgcn_sched_barrier(0);
            float s7 = ink_r17_vd_nf_adj(sa_v[7], R, rel_bias, rs, pg, nq_tot, h0, pos0, kb, p1, 7u, rel_extent); __builtin_amdgcn_sched_barrier(0);

            const ink_r17_vd_v8f sa_adj = (ink_r17_vd_v8f){s0, s1, s2, s3, s4, s5, s6, s7};
            float bm = __builtin_fmaxf(__builtin_fmaxf(__builtin_fmaxf(s0, s1), __builtin_fmaxf(s2, s3)),
                                       __builtin_fmaxf(__builtin_fmaxf(s4, s5), __builtin_fmaxf(s6, s7)));
            bm = __builtin_fmaxf(bm, ink_r17_vd_xrow(bm));

            const int active = (__builtin_amdgcn_ballot_w32(q_valid && bm > -1e30f && !(bm - m < -30.0f)) != 0u);
            if (active) {
              const int nr = (__builtin_amdgcn_ballot_w32(q_valid && bm > -1e30f && (bm > m || m <= -1e30f)) != 0u);
              const float mn = (q_valid && nr) ? __builtin_fmaxf(m, bm) : m;
              float c = 1.0f;
              if (nr) {
                c = (q_valid && m > -1e30f) ? __builtin_amdgcn_exp2f(m - mn) : 0.0f;
#pragma unroll
                for (u32 nt = 0u; nt < 4u; nt++)
#pragma unroll
                  for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c;
              }
              ink_r17_vd_h8 pb = (ink_r17_vd_h8)(0.0f);
              float ps = 0.0f;
              if (q_valid && mn > -1e30f) {
#pragma unroll
                for (u32 v = 0u; v < 8u; v++) {
                  const float p = __builtin_amdgcn_exp2f(sa_adj[v] - mn);
                  ps += p;
                  pb[v] = (f16)p;
                }
              }
              ls = ls * c + ps;
              m = mn;
              ink_r17_vd_pv16_1d_4t(ink_r17_vd_gw_lds.nf16.vt[bf], pb, o, d_half, has_next);
            }
            if (has_next) {
              if (!active) ink_r17_vd_lsignal();
              ink_r17_vd_lwait();
            }
          }
          lbarrier(); /* End Phase B: release nf16 LDS! */
        }
      }
    }

    /* ======================================================================
     * Phase 4A: Stage (m, l_tot) + o[0..3] from Consumer waves 0..3 into
     *           lds.ph4, then coalesce 8 queries at a time across all 8 waves.
     * ==================================================================== */
    if (w < 4u) {
      const u32 pg_w = w >> 1u;
      const u32 dh_w = w & 1u;
      if (pg_w < npg && dh_w == 0u) {
        const u32 l_4 = ink_r17_vd_lane_pin();
        const u32 rl_4 = l_4 & 15u, hl_4 = l_4 >> 4u;
        const float l_tot = ls + ink_r17_vd_xrow(ls);
        if (hl_4 == 0u) {
          const u32 qi = (pg_w << 4u) + rl_4;
          ink_r17_vd_gw_lds.ph4.m[qi] = (l_tot > 0.0f) ? (m * INK_R17_VD_LN2) : -1e30f;
          ink_r17_vd_gw_lds.ph4.l[qi] = l_tot;
        }
      }
    }
    lbarrier();

    const u32 l_out = ink_r17_vd_lane_pin();
    const u32 rl_o = l_out & 15u, hl_o = l_out >> 4u;
    const u32 pg_c = w >> 1u;
    const u32 dh_c = w & 1u;

#pragma clang loop unroll(disable)
    for (u32 pg_cur = 0u; pg_cur < npg; pg_cur++) {
#pragma unroll
      for (u32 qb = 0u; qb < 2u; qb++) {
        const u32 q_base = (pg_cur << 4u) + (qb << 3u);
        if (q_base >= nq_tot) break;
        if (w < 4u && pg_c == pg_cur && (rl_o >> 3u) == qb) {
          const u32 q_loc = rl_o & 7u;
          LDSP float *dst_q = ink_r17_vd_gw_lds.ph4.o_stg[q_loc] + (dh_c << 6u) + (hl_o << 3u);
#pragma unroll
          for (u32 t = 0u; t < 4u; t++) {
            LDSP float *dst_t = dst_q + (t << 4u);
#pragma unroll
            for (u32 v = 0u; v < 8u; v++) dst_t[v] = o[t][v];
          }
        }
        lbarrier();

        const u32 qi = q_base + w;
        if (qi < nq_tot) {
          const u32 t = qi / INK_R17_VD_GQA;
          const u32 h = h0 + (qi - t * INK_R17_VD_GQA);
          f32x4 ov = ((const LDSP f32x4 *)ink_r17_vd_gw_lds.ph4.o_stg[w])[l_out];
          if (nsplit == 1u) {
            const G float *st = (const G float *)((const G u8 *)sink + (ink_r17_vd_u64)t * (ink_r17_vd_u64)ss);
            const float sv = st[h];
            const float mq = ink_r17_vd_gw_lds.ph4.m[qi];
            const float lq = ink_r17_vd_gw_lds.ph4.l[qi];
            const float M = (lq > 0.0f) ? __builtin_fmaxf(sv, mq) : sv;
            const float wx = (lq > 0.0f) ? __builtin_expf(mq - M) : 0.0f;
            const float L = __builtin_expf(sv - M) + lq * wx;
            ov *= (wx / L);
            if ((rot_flags & 2u) != 0u) ov = ink_r17_vd_fwht128_wave4(ov, l_out);
            const G float *gt_h = agate ? ((const G float *)((const G u8 *)agate + (ink_r17_vd_u64)t * (ink_r17_vd_u64)gs) + (ink_r17_vd_u64)h * INK_R17_VD_HD) : (const G float *)0;
            G float *ot_h = out ? ((G float *)((G u8 *)out + (ink_r17_vd_u64)t * (ink_r17_vd_u64)os) + (ink_r17_vd_u64)h * INK_R17_VD_HD) : (G float *)0;
            G u8 *oqt = oq ? (oq + (ink_r17_vd_u64)t * (ink_r17_vd_u64)oqs) : (G u8 *)0;
            ink_r17_vd_gate_q8_head_vec4(ov, gt_h, ot_h, oqt, h, n_head, l_out);
          } else {
            G float *pp = part + (ink_r17_vd_u64)((t * n_head + h) * nsplit + s) * INK_R17_VD_PSTR;
            if (l_out == 0u) {
              pp[0] = ink_r17_vd_gw_lds.ph4.m[qi];
              pp[1] = ink_r17_vd_gw_lds.ph4.l[qi];
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
    ink_r17_vd_gw_lds.ph4.is_last =
        (__scoped_atomic_fetch_add(&count[kh], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r17_vd_gw_lds.ph4.is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const u32 l_m = ink_r17_vd_lane_pin();
  for (u32 qi = w; qi < nq_tot; qi += 8u) {
    const u32 t = qi / INK_R17_VD_GQA;
    const u32 h = h0 + (qi - t * INK_R17_VD_GQA);
    const G float *st = (const G float *)((const G u8 *)sink + (ink_r17_vd_u64)t * (ink_r17_vd_u64)ss);
    const G float *gt_h = agate ? ((const G float *)((const G u8 *)agate + (ink_r17_vd_u64)t * (ink_r17_vd_u64)gs) + (ink_r17_vd_u64)h * INK_R17_VD_HD) : (const G float *)0;
    G float *ot_h = out ? ((G float *)((G u8 *)out + (ink_r17_vd_u64)t * (ink_r17_vd_u64)os) + (ink_r17_vd_u64)h * INK_R17_VD_HD) : (G float *)0;
    G u8 *oqt = oq ? (oq + (ink_r17_vd_u64)t * (ink_r17_vd_u64)oqs) : (G u8 *)0;
    const G float *ph = part + (ink_r17_vd_u64)((t * n_head + h) * nsplit) * INK_R17_VD_PSTR;
    ink_r17_vd_gw_merge_head(ph, st[h], nsplit, l_m, ink_r17_vd_gw_lds.ph4.wt[w], gt_h, ot_h, oqt, h, n_head, rot_flags);
  }
  barrier();
  if (tid() == 0u) count[kh] = 0u;
}

#undef INK_R17_VD_HD
#undef INK_R17_VD_DREL
#undef INK_R17_VD_GQA
#undef INK_R17_VD_RECB
#undef INK_R17_VD_PSTR
#undef INK_R17_VD_NB16
#undef INK_R17_VD_NB32
#undef INK_R17_VD_DS_MAXNB
#undef INK_R17_VD_FL_MAXNB
#undef INK_R17_VD_LOG2E
#undef INK_R17_VD_LN2
#undef INK_R17_VD_INV_SQRT
#undef INK_R17_VD_QSC
#undef INK_R17_VD_ADD4

#endif /* IE_MEGA */
