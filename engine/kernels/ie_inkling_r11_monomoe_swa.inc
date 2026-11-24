/* ============================================================================
 * Inkling-2 Round 11: MonoMoE Persistent 1-Launch MoE Block (H16) &
 *                     SWA Verify Attention WMMA + QT=2 Exact-SHA (H21)
 * Target: RX 9070 XT (gfx1201, RDNA4, wave32, NT=256, 64 CUs, 64 MiB MALL)
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r11_ms_ /
 * INK_R11_MS_ and all macros are #undef'd at EOF.
 *
 * Exported Kernels (0/0 spills, priv=0, scratch=0):
 *   1. ie_inkling_monomoe_t1_r11 (H16, -144 launches/step on T=1)
 *        1-launch persistent MoE block collapsing ALL 8 ops (Router + Top-8
 *        Route + IQ3_XXS Gate/Up + SITU + Q8_0 + IQ4_NL Down + Ordered Combine
 *        + 4-Tap SConv + Ring + Residual Add) with grid <= 64, mandatory WDDM
 *        s_sendmsg_rtnl(0x83) 250ms/900ms poison guard on b->ink_sp_sync[2],
 *        TH_LOAD_NT on expert weights, TH_LOAD_HT on ffn_in_q8 / L2-hot
 *        exp_mid_q8, supporting BOTH AoS (flags bit 0 = 0) and FLUTE SoA
 *        (flags bit 0 = 1) + co-residency probe mode (flags bit 1 = 1)!
 *   2. ie_inkling_rel_attn_ver_swa_wmma_r11 (H21-WMMA)
 *        Single-pass T<=8 FP16 WMMA SWA verify attention (40 SWA layers,
 *        win=1024, 2.02-2.73 ms in #20) with exact both-sided sliding-window
 *        mask (kp >= w_q && kp <= pos_q), fast power-of-2 ring wrap, nsplit
 *        giving >= 128 WGs (e.g. nsplit=16 -> 128 WGs), optional fused
 *        OP_GATE(sigmoid) + Q8_0 epilogue (<= 128 VGPR = 3 WGs/CU, 15,360 B LDS).
 *   3. ie_inkling_rel_attn_ver_swa_qt2_r11 (H21-QT2)
 *        Ultra-lean <= 68 VGPR (4 WGs/CU) QT=2 exact-SHA FP32 scalar SWA verify
 *        companion (2x SWA KV bandwidth cut vs ver_r4, 0 waste at T=6 & T=8,
 *        optional fused OP_GATE(sigmoid) + Q8_0 epilogue, 12,484 B LDS).
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(kvt) == 2u, "ie_inkling_r11_monomoe_swa requires f16 KV cache");
_Static_assert(sizeof(f16) == 2u, "ink_r11_ms f16 must be 2 bytes");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r11_ms float/u32 must be 4 bytes");

#define INK_R11_MS_RSRC_W3       0x31004000u
#define INK_R11_MS_TH_HT         0 /* R43 scrub: standard default cached load (TH_LOAD_RT) */
#define INK_R11_MS_MAX_GRID      64u
#define INK_R11_MS_DEF_BAR_TICKS 25000000ull /* 250 ms @ 100 MHz s_sendmsg_rtnl(0x83) */
#define INK_R11_MS_DEF_KRN_TICKS 90000000ull /* 900 ms @ 100 MHz s_sendmsg_rtnl(0x83) */
#define INK_R11_MS_SB_BYTES      98u
#define INK_R11_MS_LDS_MAXNB     96u
#define INK_R11_MS_HD            128u
#define INK_R11_MS_DREL          16u
#define INK_R11_MS_GQA           3u
#define INK_R11_MS_PSTR          130u
#define INK_R11_MS_LOG2E         1.44269504088896340736f
#define INK_R11_MS_LN2           0.69314718055994530942f
#define INK_R11_MS_QSC           (0.08838834764831844299f * 1.44269504088896340736f)

typedef unsigned long long ink_r11_ms_u64;
typedef unsigned int ink_r11_ms_u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned int ink_r11_ms_u32_a2 __attribute__((aligned(2)));
typedef u32x4 ink_r11_ms_u32x4_a4 __attribute__((aligned(4)));
typedef ink_r11_ms_u32x2 ink_r11_ms_u32x2_a4 __attribute__((aligned(4)));

static inline __attribute__((always_inline)) __amdgpu_buffer_rsrc_t ink_r11_ms_rsrc(
    const G void *base_u, u32 nbytes) {
  G void *p = (G void *)rq_ptr_u((const G u8 *)base_u);
  return __builtin_amdgcn_make_buffer_rsrc(p, 0, nbytes, INK_R11_MS_RSRC_W3);
}

static inline __attribute__((always_inline)) u32x4 ink_r11_ms_nt_ld4(const G void *p) {
  return *(const G ink_r11_ms_u32x4_a4 *)p;
}

static inline __attribute__((always_inline)) ink_r11_ms_u32x2 ink_r11_ms_nt_ld2(const G void *p) {
  return *(const G ink_r11_ms_u32x2_a4 *)p;
}

static inline __attribute__((always_inline)) u32 ink_r11_ms_nt_ld1(const G void *p) {
  return *(const G u32 *)p;
}

static inline __attribute__((always_inline)) u32 ink_r11_ms_nt_a32(const G u8 *p, u32 o) {
  return *(const G ink_r11_ms_u32_a2 *)(p + o);
}

static inline __attribute__((always_inline)) u32 ink_r11_ms_nt_u16(const G void *p) {
  return (u32)(*(const G u16 *)p);
}

static inline __attribute__((always_inline)) f16 ink_r11_ms_nt_f16(const G void *p) {
  const u16 bits = *(const G u16 *)p;
  return __builtin_bit_cast(f16, bits);
}

static inline __attribute__((always_inline)) u32x4 ink_r11_ms_ht_ld4(
    __amdgpu_buffer_rsrc_t r, u32 voff) {
  return __builtin_amdgcn_raw_buffer_load_b128(r, voff, 0, INK_R11_MS_TH_HT);
}

static inline __attribute__((always_inline)) float ink_r11_ms_ht_f32(
    __amdgpu_buffer_rsrc_t r, u32 voff) {
  const u32 bits = __builtin_amdgcn_raw_buffer_load_b32(r, voff, 0, INK_R11_MS_TH_HT);
  return I2F((int)bits);
}

static inline __attribute__((always_inline)) float ink_r11_ms_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline __attribute__((always_inline)) float ink_r11_ms_oct_tree_f(float v) {
#pragma clang fp contract(off)
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}

static inline __attribute__((always_inline)) float ink_r11_ms_sgpr_f(float v) {
  return I2F(__builtin_amdgcn_readfirstlane(F2I(v)));
}

static inline __attribute__((always_inline)) float ink_r11_ms_xrow(float v) {
  return I2F(xrow_i(F2I(v)));
}

static inline __attribute__((always_inline)) float ink_r11_ms_halfwave_sum_all(float v) {
  v += I2F(shl_i(F2I(v), 8));
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return I2F(__builtin_amdgcn_readlane(F2I(v), 0));
}

static inline __attribute__((always_inline)) float ink_r11_ms_dot4(f32x4 a, f32x4 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

static inline __attribute__((always_inline)) float ink_r11_ms_gate_f(float a, float g) {
#pragma clang fp contract(off)
  return a * (1.0f / (1.0f + __builtin_expf(-g)));
}

static inline __attribute__((always_inline)) float ink_r11_ms_sub8_max_pos(float v) {
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0xB1, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0x4E, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_ds_swizzle(F2I(v), 0x101F)));
  return v;
}

static inline __attribute__((always_inline)) u32 ink_r11_ms_sub8_sum_u32(u32 s) {
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0xB1, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0x4E, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_ds_swizzle((int)s, 0x101F);
  return s;
}

static inline __attribute__((always_inline)) void ink_r11_ms_gate_q8_head_vec4(
    f32x4 ov, const G float *gt_h, G float *ot_h, G u8 *oqt, u32 h, u32 n_head, u32 l) {
#pragma clang fp contract(off)
  f32x4 vv = ov;
  if (gt_h) {
    const f32x4 gv = ((const G f32x4 *)gt_h)[l];
    vv = (f32x4){ink_r11_ms_gate_f(ov.x, gv.x), ink_r11_ms_gate_f(ov.y, gv.y),
                 ink_r11_ms_gate_f(ov.z, gv.z), ink_r11_ms_gate_f(ov.w, gv.w)};
  }
  if (ot_h) ((G f32x4 *)ot_h)[l] = vv;
  if (oqt) {
    const u32 sg = l >> 3u;
    const u32 sl = l & 7u;
    const u32 blk = (h << 2u) + sg;
    const u32 nb_q = n_head << 2u;
    float am = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(vv.x), __builtin_fabsf(vv.y)),
                               __builtin_fmaxf(__builtin_fabsf(vv.z), __builtin_fabsf(vv.w)));
    am = ink_r11_ms_sub8_max_pos(am);
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
    const u32 s = ink_r11_ms_sub8_sum_u32((u32)qi0 + (u32)qi1 + (u32)qi2 + (u32)qi3);
    if (sl == 0u) {
      ((G float *)(oqt + (nb_q << 5u)))[blk] = d;
      ((G u32 *)(oqt + nb_q * 36u))[blk] = s;
    }
  }
}

static inline __attribute__((always_inline)) int ink_r11_ms_iq4_dot4(
    u32x4 w4, u32x4 a0, u32x4 a1) {
  const u32 wx = w4.x, wy = w4.y, wz = w4.z, ww = w4.w;
  const u32 a00 = a0.x, a01 = a0.y, a02 = a0.z, a03 = a0.w;
  const u32 a10 = a1.x, a11 = a1.y, a12 = a1.z, a13 = a1.w;
  u32 d = 0u;
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(wx & 0x0F0F0F0Fu), a00), ie_iq4_map((wx >> 4u) & 0x0F0F0F0Fu), a10);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(wy & 0x0F0F0F0Fu), a01), ie_iq4_map((wy >> 4u) & 0x0F0F0F0Fu), a11);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(wz & 0x0F0F0F0Fu), a02), ie_iq4_map((wz >> 4u) & 0x0F0F0F0Fu), a12);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(ww & 0x0F0F0F0Fu), a03), ie_iq4_map((ww >> 4u) & 0x0F0F0F0Fu), a13);
  return (int)d;
}

static inline __attribute__((always_inline)) float ink_r11_ms_iq3_stream_dot(
    const LDSP u32 *ulut, u32 d16, u32 aux, u32 q01, u32 q23,
    u32x4 x0, u32x4 x1, float dav) {
#pragma clang fp contract(off)
  const float dw = ie_rq_h2f(d16) * (0.5f + (float)(aux >> 28u)) * 0.5f;
  const u32 qp0 = q01 & 0xFFFFu, qp1 = q01 >> 16u;
  const u32 qp2 = q23 & 0xFFFFu, qp3 = q23 >> 16u;
  const u32 g_lo0 = ulut[qp0 & 255u], g_hi0 = ulut[qp0 >> 8u];
  const u32 g_lo1 = ulut[qp1 & 255u], g_hi1 = ulut[qp1 >> 8u];
  const u32 g_lo2 = ulut[qp2 & 255u], g_hi2 = ulut[qp2 >> 8u];
  const u32 g_lo3 = ulut[qp3 & 255u], g_hi3 = ulut[qp3 >> 8u];
  __builtin_amdgcn_sched_barrier(0);

  const u32 x00 = x0.x, x01 = x0.y, x02 = x0.z, x03 = x0.w;
  const u32 x10 = x1.x, x11 = x1.y, x12 = x1.z, x13 = x1.w;

  const u32 s8_0 = rq_ksign7(aux & 127u);
  u32 idot = ie_dot4_ss(0u, ie_rq_ssign(g_lo0, ie_rq_mask4(s8_0 & 15u)), x00);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_hi0, ie_rq_mask4(s8_0 >> 4u)), x01);

  const u32 s8_1 = rq_ksign7((aux >> 7u) & 127u);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_lo1, ie_rq_mask4(s8_1 & 15u)), x02);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_hi1, ie_rq_mask4(s8_1 >> 4u)), x03);

  const u32 s8_2 = rq_ksign7((aux >> 14u) & 127u);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_lo2, ie_rq_mask4(s8_2 & 15u)), x10);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_hi2, ie_rq_mask4(s8_2 >> 4u)), x11);

  const u32 s8_3 = rq_ksign7((aux >> 21u) & 127u);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_lo3, ie_rq_mask4(s8_3 & 15u)), x12);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_hi3, ie_rq_mask4(s8_3 >> 4u)), x13);

  return dav * (dw * (float)(int)idot);
}

static inline __attribute__((always_inline)) f32x4 ink_r11_ms_moe_combine4(
    const G f32x4 *y4, const G u32 *rt, u32 n4, u32 K, u32 i4) {
#pragma clang fp contract(off)
  const G float *w = (const G float *)(rt + K);
  f32x4 x0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
#pragma clang loop unroll(disable)
  for (u32 k = 0u; k < K; k++) {
    const float wk = w[k];
    const f32x4 yk = y4[(unsigned long)k * n4 + i4];
    x0.x = __builtin_fmaf(wk, yk.x, x0.x);
    x0.y = __builtin_fmaf(wk, yk.y, x0.y);
    x0.z = __builtin_fmaf(wk, yk.z, x0.z);
    x0.w = __builtin_fmaf(wk, yk.w, x0.w);
  }
  return x0;
}

static inline __attribute__((always_inline)) u32 ink_r11_ms_grid_barrier(
    G u32 *sync, u32 ng, u32 *sense_io, LDSP u32 *lds_poison,
    ink_r11_ms_u64 k_t0, ink_r11_ms_u64 bar_lim, ink_r11_ms_u64 krn_lim) {
  if (ng <= 1u) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    return 0u;
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();

  const u32 next_sense = *sense_io + 1u;
  if (tid() == 0u) {
    u32 poisoned = __scoped_atomic_load_n(&sync[2u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    if (!poisoned) {
      const u32 prev = __scoped_atomic_fetch_add(&sync[0u], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      if (prev == ng - 1u) {
        __scoped_atomic_store_n(&sync[0u], 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __scoped_atomic_store_n(&sync[1u], next_sense, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      } else {
        const ink_r11_ms_u64 t0 = (ink_r11_ms_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
        for (;;) {
          const u32 cur = __scoped_atomic_load_n(&sync[1u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
          if ((int)(cur - next_sense) >= 0) break;
          if (__scoped_atomic_load_n(&sync[2u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) != 0u) {
            poisoned = 1u;
            break;
          }
          const ink_r11_ms_u64 now = (ink_r11_ms_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
          if ((now - t0) > bar_lim || (now - k_t0) > krn_lim) {
            __scoped_atomic_fetch_or(&sync[2u], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
            poisoned = 1u;
            break;
          }
          __builtin_amdgcn_s_sleep(2);
        }
        const ink_r11_ms_u64 dt_k = ((ink_r11_ms_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83) - t0) / 1000ull;
        __scoped_atomic_fetch_max(&sync[3u], (u32)dt_k, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      }
    }
    *lds_poison = poisoned;
  }
  barrier();
  *sense_io = next_sense;
  if (*lds_poison != 0u) return 1u;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  return 0u;
}

#define INK_R11_MS_ADD4(a0_, a1_, a2_, a3_, j_, v_) \
  do {                                              \
    if ((j_) == 0u) (a0_) += (v_);                  \
    else if ((j_) == 1u) (a1_) += (v_);             \
    else if ((j_) == 2u) (a2_) += (v_);             \
    else (a3_) += (v_);                             \
  } while (0)

#define INK_R11_MS_FMA4(a0_, a1_, a2_, a3_, j_, v_, d_)       \
  do {                                                        \
    if ((j_) == 0u) (a0_) = __builtin_fmaf((v_), (d_), (a0_)); \
    else if ((j_) == 1u) (a1_) = __builtin_fmaf((v_), (d_), (a1_)); \
    else if ((j_) == 2u) (a2_) = __builtin_fmaf((v_), (d_), (a2_)); \
    else (a3_) = __builtin_fmaf((v_), (d_), (a3_));           \
  } while (0)

/* ============================================================================
 * Kernel 1 (H16): ie_inkling_monomoe_t1_r11
 * Weight-Major 1-Launch Persistent MoE Block for T=1 (-144 launches/step!):
 *   Stage 0a: Exact Router GEMV [n_exp x dim] (gemv_f32_row, 2 rows/WG @ ng=64)
 *   Barrier 1 -> Stage 0b: Exact Top-K moe_route8 (WG 0, Wave 0)
 *   Barrier 2 -> Stage 1:  IQ3_XXS Gate/Up + SITU + Q8_0 -> exp_mid_q8 (10.2 KiB
 *                          L2-hot!; supports FLUTE SoA via flags bit 0 OR AoS;
 *                          TH_LOAD_NT weights, TH_LOAD_HT ffn_in_q8 staged in LDS!)
 *   Barrier 3 -> Stage 2a: IQ4_NL Down [dim x eff] (K * 72 tiles across ng<=64 WGs,
 *                          TH_LOAD_NT weights, TH_LOAD_HT L2-hot exp_mid_q8 -> yd)
 *   Barrier 4 -> Stage 2b: Exact ordered K-slot combine + 4-tap sconv + ring + add
 *
 * Flags: bit 0 (1u) = FLUTE_SOA layout for qw_gu (0 = AoS)
 *        bit 1 (2u) = PROBE_ONLY (runs 2 guarded barriers and returns cleanly)
 * Grid:  (ng, 1u, 1u) with ng <= 64 (hard-clamped), NT = 256, <= 80 VGPR, 4 WGs/CU.
 * Parity: 100% bitwise-exact (0-bit-diff) vs router_route_exact + iq3_situ + iq4_tail!
 * ========================================================================== */

typedef struct {
  u32   lut[256u] __attribute__((aligned(16)));
  u32   xq[INK_R11_MS_LDS_MAXNB * 8u] __attribute__((aligned(16)));
  float xd[INK_R11_MS_LDS_MAXNB] __attribute__((aligned(16)));
  float g[32u] __attribute__((aligned(16)));
  float u[32u] __attribute__((aligned(16)));
  u32   poison;
} ink_r11_ms_mm_lds_t;

static LDS ink_r11_ms_mm_lds_t ink_r11_ms_mm_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r11_ms_mm_lds_t) <= 8192u, "ink_r11_ms_mm_lds_t must stay <= 8 KiB");

KERNEL ie_inkling_monomoe_t1_r11(
    /* Router & Route (Stage 0) */
    const G float *w_rtr, const G float *ffn_in, G float *logits,
    const G float *ebias, G u32 *route, float r_scale,
    /* IQ3_XXS Gate/Up + SITU + Q8_0 (Stage 1) */
    const G u8 *qw_gu, const G u8 *ffn_in_q8, G u8 *exp_mid_q8,
    u32 q_bytes_gu, u32 qstr_mid, float beta, float lbeta,
    /* IQ4_NL Down + Combine + SConv + Ring + Add (Stage 2) */
    const G u32 *qw_dn, const G u16 *qs_dn, G float *yd,
    u32 qwn_dn, u32 qsn_dn,
    const G float *v_conv, G float *ring, const G float *res, G float *out,
    float escale, u32 pos0,
    /* Dimensions, Sync & WDDM Poison Guard */
    u32 dim, u32 eff, u32 n_exp, u32 K, u32 ng, u32 flags,
    G u32 *sync, ink_r11_ms_u64 bar_ticks, ink_r11_ms_u64 krn_ticks) {
#pragma clang fp contract(off)
  const u32 g = wgid();
  if (ng == 0u || ng > INK_R11_MS_MAX_GRID || g >= ng || !sync) return;
  if (__scoped_atomic_load_n(&sync[2u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) != 0u) return;

  const ink_r11_ms_u64 bar_lim = bar_ticks ? bar_ticks : INK_R11_MS_DEF_BAR_TICKS;
  const ink_r11_ms_u64 krn_lim = krn_ticks ? krn_ticks : INK_R11_MS_DEF_KRN_TICKS;
  const ink_r11_ms_u64 k_t0 = (ink_r11_ms_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
  u32 sense = __scoped_atomic_load_n(&sync[1u], __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);

  if (flags & 2u) {
    if (ink_r11_ms_grid_barrier(sync, ng, &sense, &ink_r11_ms_mm_lds.poison, k_t0, bar_lim, krn_lim)) return;
    ink_r11_ms_grid_barrier(sync, ng, &sense, &ink_r11_ms_mm_lds.poison, k_t0, bar_lim, krn_lim);
    return;
  }

  if (dim == 0u || (dim & 31u) != 0u || eff == 0u || (eff & 31u) != 0u ||
      n_exp == 0u || n_exp > 256u || K == 0u || K > 32u ||
      !w_rtr || !ffn_in || !logits || !route || !qw_gu || !ffn_in_q8 || !exp_mid_q8 ||
      !qw_dn || !qs_dn || !yd || !v_conv || !ring || !res || !out)
    return;

  const u32 t0 = tid();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane();
  const u32 nb_in = dim >> 5u;
  const u32 nb_mid = eff >> 5u;
  if (nb_in < 8u || (nb_in & 7u) != 0u) return;

  /* Stage 0a: Exact Router GEMV (2 rows/WG at n_exp=128, ng=64; 0-bit-diff!). */
  for (u32 r = g; r < n_exp; r += ng) {
    const float acc = gemv_f32_row(w_rtr + (unsigned long)r * (unsigned long)dim, ffn_in, dim);
    if (t0 == 0u) logits[r] = acc;
  }
  if (ink_r11_ms_grid_barrier(sync, ng, &sense, &ink_r11_ms_mm_lds.poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 0b: Exact Top-K moe_route8 in WG 0 Wave 0 while ALL WGs pre-stage LUT + ffn_in_q8! */
  if (g == 0u && t0 < 32u) {
    moe_route8(logits, n_exp, K, ebias, r_scale, route);
  }
  ink_r11_ms_mm_lds.lut[t0] = ie_rq_iq3xxs_grid[t0];
  const int use_lds_x = (nb_in <= INK_R11_MS_LDS_MAXNB);
  if (use_lds_x) {
    const __amdgpu_buffer_rsrc_t xr_in = ink_r11_ms_rsrc(ffn_in_q8, nb_in * 40u);
    const u32 n_v4 = nb_in << 1u;
    if (t0 < n_v4) ((LDSP u32x4 *)ink_r11_ms_mm_lds.xq)[t0] = ink_r11_ms_ht_ld4(xr_in, t0 << 4u);
    if (t0 < nb_in) ink_r11_ms_mm_lds.xd[t0] = ink_r11_ms_ht_f32(xr_in, (nb_in << 5u) + (t0 << 2u));
  }
  if (ink_r11_ms_grid_barrier(sync, ng, &sense, &ink_r11_ms_mm_lds.poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 1: Weight-Major IQ3_XXS Gate/Up + SITU + Q8_0 -> L2-hot exp_mid_q8!
   * Wave-split (Waves 0..3 gate, Waves 4..7 up, 2 sub-passes = 1 full 32-elem block)
   * keeps peak VGPR <= 64 while sharing LDS-staged ffn_in_q8 and TH_LOAD_NT weights! */
  {
    const u32 is_flute = flags & 1u;
    const u32 is_up = wv >> 2u;
    const u32 wv4 = wv & 3u;
    const u32 k = l >> 3u, s = l & 7u;
    const u32 mb = nb_in >> 3u;
    const u32 row_bytes = mb * INK_R11_MS_SB_BYTES;
    const unsigned long n_sb_tot = (unsigned long)(eff >> 1u) * (unsigned long)mb;
    const u32 qo = 2u + (s << 3u);
    const u32 auxo = 66u + (s << 2u);
    const u32 tot_s1 = K * nb_mid;
    const __amdgpu_buffer_rsrc_t xr_in = ink_r11_ms_rsrc(ffn_in_q8, nb_in * 40u);
    const u32 da_in_off = nb_in << 5u;

#pragma unroll 1
    for (u32 tile = g; tile < tot_s1; tile += ng) {
      const u32 slot = tile / nb_mid;
      const u32 blk = tile - slot * nb_mid;
      const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[slot]);
      const G u8 *ebase = rq_ptr_u(qw_gu + (unsigned long)e * (unsigned long)q_bytes_gu);

      if (is_flute) {
        const G ink_r11_ms_u32x2 *p_qs  = (const G ink_r11_ms_u32x2 *)ebase;
        const G u32              *p_aux = (const G u32 *)(ebase + n_sb_tot * 256ul);
        const G u16              *p_d   = (const G u16 *)(ebase + n_sb_tot * 384ul);
        const u32 rg_half = (is_up ? (eff >> 2u) : 0u) + (blk << 3u) + wv4;

#pragma unroll 1
        for (u32 sp = 0u; sp < 2u; sp++) {
          const u32 rg = rg_half + (sp << 2u);
          const unsigned long g0 = (unsigned long)rg * (unsigned long)mb;
          float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

          ink_r11_ms_u32x2 q_c = ink_r11_ms_nt_ld2(p_qs + g0 * 32ul + (unsigned long)l);
          u32 aux_c = ink_r11_ms_nt_ld1(p_aux + g0 * 32ul + (unsigned long)l);
          u32 d_c = ink_r11_ms_nt_u16(p_d + g0 * 4ul + (unsigned long)k);
          __asm__ volatile("" : "+v"(d_c), "+v"(aux_c));

#pragma unroll 1
          for (u32 sb = 0u; sb < mb; sb++) {
            const u32 b_idx = (sb << 3u) + s;
            const u32 sb_n = (sb + 1u < mb) ? (sb + 1u) : sb;
            const unsigned long gn = g0 + (unsigned long)sb_n;

            const u32x4 x0 = use_lds_x ? *(const LDSP u32x4 *)(ink_r11_ms_mm_lds.xq + (b_idx << 3u))
                                       : ink_r11_ms_ht_ld4(xr_in, b_idx << 5u);
            const u32x4 x1 = use_lds_x ? *(const LDSP u32x4 *)(ink_r11_ms_mm_lds.xq + (b_idx << 3u) + 4u)
                                       : ink_r11_ms_ht_ld4(xr_in, (b_idx << 5u) + 16u);
            const float dav = use_lds_x ? ink_r11_ms_mm_lds.xd[b_idx]
                                        : ink_r11_ms_ht_f32(xr_in, da_in_off + (b_idx << 2u));

            const ink_r11_ms_u32x2 q_n = ink_r11_ms_nt_ld2(p_qs + gn * 32ul + (unsigned long)l);
            const u32 aux_n = ink_r11_ms_nt_ld1(p_aux + gn * 32ul + (unsigned long)l);
            const u32 d_n = ink_r11_ms_nt_u16(p_d + gn * 4ul + (unsigned long)k);
            __builtin_amdgcn_sched_barrier(0);

            const float val = ink_r11_ms_iq3_stream_dot(
                ink_r11_ms_mm_lds.lut, d_c, aux_c, q_c.x, q_c.y, x0, x1, dav);
            INK_R11_MS_ADD4(a0, a1, a2, a3, sb & 3u, val);
            __builtin_amdgcn_sched_barrier(0);
            q_c = q_n; aux_c = aux_n; d_c = d_n;
          }

          const float acc = ink_r11_ms_oct_tree_f((a0 + a2) + (a1 + a3));
          if (s == 0u) {
            const u32 r_loc = (sp << 4u) + (wv4 << 2u) + k;
            if (is_up) ink_r11_ms_mm_lds.u[r_loc] = acc;
            else ink_r11_ms_mm_lds.g[r_loc] = acc;
          }
        }
      } else {
        /* AoS IQ3_XXS with TH_LOAD_NT weights + LDS-staged ffn_in_q8. */
#pragma unroll 1
        for (u32 sp = 0u; sp < 2u; sp++) {
          const u32 r_loc = (sp << 4u) + (wv4 << 2u) + k;
          const u32 r_mat = (is_up ? eff : 0u) + (blk << 5u) + r_loc;
          const G u8 *bp = ebase + (unsigned long)r_mat * (unsigned long)row_bytes;
          float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

          u32 d_c = ink_r11_ms_nt_u16(bp);
          u32 aux_c = ink_r11_ms_nt_a32(bp, auxo);
          u32 q01_c = ink_r11_ms_nt_a32(bp, qo);
          u32 q23_c = ink_r11_ms_nt_a32(bp, qo + 4u);
          __asm__ volatile("" : "+v"(d_c), "+v"(aux_c), "+v"(q01_c), "+v"(q23_c));

#pragma unroll 1
          for (u32 sb = 0u; sb < mb; sb++) {
            const u32 b_idx = (sb << 3u) + s;
            const G u8 *bp_n = (sb + 1u < mb) ? (bp + INK_R11_MS_SB_BYTES) : bp;

            const u32x4 x0 = use_lds_x ? *(const LDSP u32x4 *)(ink_r11_ms_mm_lds.xq + (b_idx << 3u))
                                       : ink_r11_ms_ht_ld4(xr_in, b_idx << 5u);
            const u32x4 x1 = use_lds_x ? *(const LDSP u32x4 *)(ink_r11_ms_mm_lds.xq + (b_idx << 3u) + 4u)
                                       : ink_r11_ms_ht_ld4(xr_in, (b_idx << 5u) + 16u);
            const float dav = use_lds_x ? ink_r11_ms_mm_lds.xd[b_idx]
                                        : ink_r11_ms_ht_f32(xr_in, da_in_off + (b_idx << 2u));

            const u32 d_n = ink_r11_ms_nt_u16(bp_n);
            const u32 aux_n = ink_r11_ms_nt_a32(bp_n, auxo);
            const u32 q01_n = ink_r11_ms_nt_a32(bp_n, qo);
            const u32 q23_n = ink_r11_ms_nt_a32(bp_n, qo + 4u);
            __builtin_amdgcn_sched_barrier(0);

            const float val = ink_r11_ms_iq3_stream_dot(
                ink_r11_ms_mm_lds.lut, d_c, aux_c, q01_c, q23_c, x0, x1, dav);
            INK_R11_MS_ADD4(a0, a1, a2, a3, sb & 3u, val);
            __builtin_amdgcn_sched_barrier(0);
            bp = bp_n;
            d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
          }

          const float acc = ink_r11_ms_oct_tree_f((a0 + a2) + (a1 + a3));
          if (s == 0u) {
            if (is_up) ink_r11_ms_mm_lds.u[r_loc] = acc;
            else ink_r11_ms_mm_lds.g[r_loc] = acc;
          }
        }
      }
      lbarrier();

      if (wv == 0u) {
        const float gv = ink_r11_ms_mm_lds.g[l];
        const float uv = ink_r11_ms_mm_lds.u[l];
        const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
        const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
        const float v = gs * sigmoidf(gv) * us;
        quant_wave(v, blk, exp_mid_q8 + (unsigned long)slot * (unsigned long)qstr_mid, nb_mid);
      }
      lbarrier();
    }
  }
  if (ink_r11_ms_grid_barrier(sync, ng, &sense, &ink_r11_ms_mm_lds.poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 2a: IQ4_NL Down [dim x eff] (K * (dim/32) = 576 tiles across ng<=64 WGs,
   * TH_LOAD_NT on IQ4_NL weights, TH_LOAD_HT on L2-hot exp_mid_q8!). */
  {
    const u32 n_rb = dim >> 5u;
    const u32 tot_s2 = K * n_rb;
    const u32 sub = l >> 3u, sl = l & 7u;
    const u32 nstep = (nb_mid + 7u) >> 3u;
    const u32 da_mid_off = nb_mid << 5u;
    const u32 bstr_yd = dim << 2u;

#pragma unroll 1
    for (u32 tile = g; tile < tot_s2; tile += ng) {
      const u32 slot = tile / n_rb;
      const u32 rb = tile - slot * n_rb;
      const u32 r = (rb << 5u) + (wv << 2u) + sub;
      const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[slot]);

      const G u8 *w_base = rq_ptr_u((const G u8 *)(qw_dn + (unsigned long)e * (unsigned long)qwn_dn));
      const G u8 *q_base = rq_ptr_u((const G u8 *)(qs_dn + (unsigned long)e * (unsigned long)qsn_dn));
      const G u8 *xp = rq_ptr_u(exp_mid_q8 + (unsigned long)slot * (unsigned long)qstr_mid);
      const __amdgpu_buffer_rsrc_t xr = ink_r11_ms_rsrc(xp, nb_mid * 40u);
      const G u8 *rw = w_base + (unsigned long)r * ((unsigned long)nb_mid * 16ul);
      const G f16 *rs_p = (const G f16 *)q_base + (unsigned long)r * (unsigned long)nb_mid;

      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
      u32x4 w_c = (u32x4)(0u);
      f16 sc_c = (f16)0.0f;
      if (sl < nb_mid) {
        w_c = ink_r11_ms_nt_ld4(rw + (unsigned long)sl * 16ul);
        sc_c = ink_r11_ms_nt_f16(rs_p + sl);
      }
      __builtin_amdgcn_sched_barrier(0);

#pragma unroll 1
      for (u32 step = 0u; step < nstep; step++) {
        const u32 b_c = (step << 3u) + sl;
        const u32 bn = b_c + 8u;
        u32x4 w_n = (u32x4)(0u);
        f16 sc_n = (f16)0.0f;
        if (bn < nb_mid) {
          w_n = ink_r11_ms_nt_ld4(rw + (unsigned long)bn * 16ul);
          sc_n = ink_r11_ms_nt_f16(rs_p + bn);
        }
        u32x4 x0_c = (u32x4)(0u), x1_c = (u32x4)(0u);
        float d_c = 0.0f;
        if (b_c < nb_mid) {
          x0_c = ink_r11_ms_ht_ld4(xr, b_c << 5u);
          x1_c = ink_r11_ms_ht_ld4(xr, (b_c << 5u) + 16u);
          d_c = ink_r11_ms_ht_f32(xr, da_mid_off + (b_c << 2u));
        }
        __builtin_amdgcn_sched_barrier(0);
        if (b_c < nb_mid) {
          const float val = (float)sc_c * (float)ink_r11_ms_iq4_dot4(w_c, x0_c, x1_c);
          INK_R11_MS_FMA4(a0, a1, a2, a3, step & 3u, val, d_c);
        }
        __builtin_amdgcn_sched_barrier(0);
        w_c = w_n; sc_c = sc_n;
      }

      const float acc = ink_r11_ms_oct_tree_f((nb_mid > 16u) ? ((a0 + a2) + (a1 + a3)) : (a0 + a1));
      if (sl == 0u) {
        ((G float *)((G u8 *)yd + (unsigned long)slot * (unsigned long)bstr_yd))[r] = acc;
      }
    }
  }
  if (ink_r11_ms_grid_barrier(sync, ng, &sense, &ink_r11_ms_mm_lds.poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 2b: Exact ordered K-slot combine + 4-tap sconv + ring + residual add! */
  {
    const u32 n4 = dim >> 2u;
    const u32 sl_r = pos0 & 3u;
#pragma clang loop unroll(disable)
    for (u32 i4 = g * NT + t0; i4 < n4; i4 += ng * NT) {
      const u32 i0 = i4 << 2u;
      const f32x4 x0 = ink_r11_ms_moe_combine4((const G f32x4 *)yd, route, n4, K, i4);
      const G f32x4 *v4 = (const G f32x4 *)v_conv + i0;
      const G f32x4 *rg = (const G f32x4 *)ring + i0;
      const f32x4 rv = ((const G f32x4 *)res)[i4];
      const u32 s1 = (pos0 - 1u) & 3u;
      const u32 s2 = (pos0 - 2u) & 3u;
      const u32 s3 = (pos0 - 3u) & 3u;
      f32x4 outv;
#pragma clang loop unroll(disable)
      for (u32 c = 0u; c < 4u; c++) {
        const float xc = x0[c];
        const f32x4 vwc = v4[c];
        float x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
        if (pos0 > 0u) {
          const f32x4 rc = rg[c];
          x1 = ink_r11_ms_pick4(rc, s1);
          if (pos0 >= 2u) x2 = ink_r11_ms_pick4(rc, s2);
          if (pos0 >= 3u) x3 = ink_r11_ms_pick4(rc, s3);
        }
        const float cv = ((xc * vwc.w + x1 * vwc.z) + x2 * vwc.y) + x3 * vwc.x;
        outv[c] = rv[c] + (xc + cv) * escale;
        ring[(unsigned long)(i0 + c) * 4ul + sl_r] = xc;
      }
      ((G f32x4 *)out)[i4] = outv;
    }
  }
}

/* ============================================================================
 * Kernel 2 (H21-WMMA): ie_inkling_rel_attn_ver_swa_wmma_r11
 * Single-pass T<=8 FP16 WMMA SWA Verify Attention (40 SWA layers, win=1024,
 * 2.02-2.73 ms in #20) with optional fused OP_GATE(sigmoid) + Q8_0 epilogue!
 *   - Packs all T*3 (<= 24) GQA queries into 1-2 16x16x16 FP16 WMMA tiles
 *     (reads SWA K/V ONCE across all T<=8 tokens -> 6x-8x KV traffic cut!).
 *   - Enforces exact BOTH-SIDED sliding-window mask (kp >= w_q && kp <= pos_q)
 *     and exact relative bias (dist < min(win, rel_extent)).
 *   - Fast power-of-2 ring-buffer wrap (single v_and_b32 for kv_len=2048!).
 *   - Grid: (n_kv * nsplit, 1u, 1u), recommended nsplit=16 -> 128 WGs (2/CU)
 *     or nsplit=8 -> 64 WGs (1/CU), NT=256, <= 128 VGPR (3 WGs/CU), 15,360 B LDS.
 * ========================================================================== */

typedef union {
  struct {
    f16 q[32u * 128u] __attribute__((aligned(16)));
    f16 r[32u * 16u] __attribute__((aligned(16)));
    f16 p_tile[2u][2u][256u] __attribute__((aligned(16)));
    union {
      u32 v_sc[8u][128u] __attribute__((aligned(16)));
      struct {
        float mx[2u][128u];
        float ls[2u][128u];
      } red;
    } scratch;
  } kv;
  struct {
    float o_stg[8u][128u] __attribute__((aligned(16)));
    float m[32u];
    float l[32u];
    float wt[8u][128u] __attribute__((aligned(16)));
    u32   is_last;
  } ph4;
} ink_r11_ms_sw_lds_t;

static LDS ink_r11_ms_sw_lds_t ink_r11_ms_sw_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r11_ms_sw_lds_t) == 15360u, "ink_r11_ms_sw_lds_t must be 15,360 B");

static inline __attribute__((always_inline)) u32 ink_r11_ms_sw_q_off(u32 row, u32 c8) {
  return row * 128u + 8u * (c8 ^ (row & 15u));
}

static inline __attribute__((always_inline)) u32 ink_r11_ms_sw_p_off(u32 rl, u32 hl) {
  return rl * 16u + 8u * (hl ^ ((rl >> 2u) & 1u));
}

static inline __attribute__((always_inline)) u32 ink_r11_ms_sw_wrap(
    u32 kp, u32 kv_len, u32 kv_mask, int is_pow2) {
  return is_pow2 ? (kp & kv_mask) : (kp % kv_len);
}

static inline __attribute__((always_inline)) h8 ink_r11_ms_sw_v_transpose(
    const u32x4 vr, LDSP u32 *v_sc_w, const u32 hl, const u32 rl) {
  const u32 sel = (rl & 1u) ? 0x03020706u : 0x05040100u;
  const u32 kh = rl >> 3u;
  const u32 k4 = (rl >> 1u) & 3u;
  const u32 w_arr[4] = {vr.x, vr.y, vr.z, vr.w};
#pragma unroll
  for (u32 q = 0u; q < 4u; q++) {
    const u32 x = w_arr[q];
    const u32 y = (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0xB1, 0xF, 0xF, true);
    const u32 t = __builtin_amdgcn_perm(y, x, sel);
    const u32 d = 8u * hl + 2u * q + (rl & 1u);
    const u32 s4 = ((d & 7u) ^ ((d >> 3u) << 1u) ^ (kh << 2u)) + ((d >> 3u) << 3u) + (kh << 4u);
    v_sc_w[s4 * 4u + k4] = t;
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  const u32 s4_r = ((rl & 7u) ^ ((rl >> 3u) << 1u) ^ (hl << 2u)) + ((rl >> 3u) << 3u) + (hl << 4u);
  return *(const LDSP h8 *)(v_sc_w + s4_r * 4u);
}

static inline __attribute__((always_inline)) void ink_r11_ms_sw_apply_rb16(
    const v8f rb, const u32 base_off, const u32 hl, const u32 kb, const u32 pos_q,
    const u32 w_q, const u32 extent_lim, const u32 d_base, LDSP float *sc_row, v8f *sv) {
  if (hl == 0u) {
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) sc_row[v] = rb[v];
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 v = 0u; v < 8u; v++) {
    const u32 kp = kb + 8u * hl + v;
    if (kp >= w_q && kp <= pos_q) {
      const u32 dist = pos_q - kp;
      if (dist < extent_lim) {
        const u32 rel = (dist - d_base) - base_off;
        if (rel < 8u) (*sv)[v] += sc_row[rel];
      }
    }
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  if (hl == 1u) {
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) sc_row[v] = rb[v];
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 v = 0u; v < 8u; v++) {
    const u32 kp = kb + 8u * hl + v;
    if (kp >= w_q && kp <= pos_q) {
      const u32 dist = pos_q - kp;
      if (dist < extent_lim) {
        const u32 rel = (dist - d_base) - (base_off + 8u);
        if (rel < 8u) (*sv)[v] += sc_row[rel];
      }
    }
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

static inline __attribute__((always_inline)) void ink_r11_ms_sw_merge_head_gate_q8(
    const G float *ph, const float sink_val, const u32 nsplit, const u32 l,
    LDSP float *wt_w, const G float *gt_h, G float *ot_h, G u8 *oqt, u32 h, u32 n_head) {
  float mv = (l == 0u) ? sink_val : -__builtin_inff();
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const unsigned long off = (unsigned long)sp * INK_R11_MS_PSTR;
    if (ph[off + 1u] > 0.0f) mv = __builtin_fmaxf(mv, ph[off]);
  }
#pragma unroll
  for (u32 mask = 16u; mask >= 1u; mask >>= 1u) mv = __builtin_fmaxf(mv, shfl_xor_f(mv, mask));
  const float M2 = ink_r11_ms_sgpr_f(mv);

  float lv = 0.0f;
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const unsigned long off = (unsigned long)sp * INK_R11_MS_PSTR;
    const float lp = ph[off + 1u];
    const float wt = (lp > 0.0f) ? __builtin_expf(ph[off] - M2) : 0.0f;
    if (sp < 128u) wt_w[sp] = wt;
    lv += lp * wt;
  }
  const float L2 = ink_r11_ms_sgpr_f(wave_sum_all(lv) + __builtin_expf(sink_val - M2));
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");

  f32x4 acc = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  for (u32 sp = 0u; sp < nsplit; sp++) {
    const float wt = (sp < 128u) ? wt_w[sp]
                                 : ((ph[(unsigned long)sp * INK_R11_MS_PSTR + 1u] > 0.0f)
                                        ? __builtin_expf(ph[(unsigned long)sp * INK_R11_MS_PSTR] - M2)
                                        : 0.0f);
    if (wt > 0.0f) {
      const G float *po = ph + (unsigned long)sp * INK_R11_MS_PSTR + 2u + 4u * l;
      const f32x4 ov = {po[0], po[1], po[2], po[3]};
      acc += ov * wt;
    }
  }
  ink_r11_ms_gate_q8_head_vec4(acc * (1.0f / L2), gt_h, ot_h, oqt, h, n_head, l);
}

KERNEL ie_inkling_rel_attn_ver_swa_wmma_r11(
    const G float *q, const G kvt *kc, const G kvt *vc,
    const G float *R, const G float *rel_bias, const G float *sink,
    G float *part, G float *out, u32 pos0, u32 T, u32 kv_len, u32 hd,
    u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
    u32 qs, u32 rs, u32 ss, u32 os, G u32 *count,
    const G float *agate, G u8 *oq, u32 gs, u32 oqs) {
  if (T == 0u || T > 8u || win == 0u || hd != INK_R11_MS_HD || d_rel != INK_R11_MS_DREL ||
      n_kv == 0u || n_head != INK_R11_MS_GQA * n_kv || kv_len == 0u || nsplit == 0u)
    return;

  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() / nsplit));
  const u32 s = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() % nsplit));
  if (kh >= n_kv) return;

  const u32 h0 = kh * INK_R11_MS_GQA;
  const u32 kvd = n_kv * INK_R11_MS_HD;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane(), hl = l >> 4u, rl = l & 15u;

  const u32 nq_tot = T * INK_R11_MS_GQA;
  const u32 npg = (nq_tot > 16u) ? 2u : 1u;
  const u32 pend = pos0 + T;
  const u32 w0 = (pos0 + 1u > win) ? (pos0 + 1u - win) : 0u;
  const u32 np = pend - w0;
  const u32 ch = (np + nsplit - 1u) / nsplit;
  const u32 off0 = s * ch;
  const u32 p0 = (off0 < np) ? (w0 + off0) : pend;
  const u32 p1 = (off0 < np && off0 + ch < np) ? (w0 + off0 + ch) : pend;

  if (nsplit > 1u && p0 >= p1) {
    for (u32 qi = w; qi < nq_tot; qi += 8u) {
      const u32 t = qi / INK_R11_MS_GQA;
      const u32 h = h0 + (qi - t * INK_R11_MS_GQA);
      G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R11_MS_PSTR;
      if (l == 0u) { pp[0] = -1e30f; pp[1] = 0.0f; }
      pp[2u + 4u * l + 0u] = 0.0f;
      pp[2u + 4u * l + 1u] = 0.0f;
      pp[2u + 4u * l + 2u] = 0.0f;
      pp[2u + 4u * l + 3u] = 0.0f;
    }
  } else {
    const u32 extent_lim = (win < rel_extent) ? win : rel_extent;
    const int is_pow2 = ((kv_len & (kv_len - 1u)) == 0u);
    const u32 kv_mask = kv_len - 1u;

    const u32 q_rows = npg * 16u;
    for (u32 idx = tid(); idx < q_rows * 16u; idx += NT) {
      const u32 qr = idx >> 4u;
      const u32 c8 = idx & 15u;
      h8 qh = (h8)(0.0f);
      if (qr < nq_tot) {
        const u32 tj = qr / INK_R11_MS_GQA;
        const u32 gi = qr - tj * INK_R11_MS_GQA;
        const G f32x4 *qp = (const G f32x4 *)((const G float *)((const G u8 *)q + (unsigned long)tj * qs) +
                                              (unsigned long)(h0 + gi) * INK_R11_MS_HD);
        const f32x4 v0 = qp[2u * c8 + 0u];
        const f32x4 v1 = qp[2u * c8 + 1u];
        qh = (h8){(f16)(v0.x * INK_R11_MS_QSC), (f16)(v0.y * INK_R11_MS_QSC),
                  (f16)(v0.z * INK_R11_MS_QSC), (f16)(v0.w * INK_R11_MS_QSC),
                  (f16)(v1.x * INK_R11_MS_QSC), (f16)(v1.y * INK_R11_MS_QSC),
                  (f16)(v1.z * INK_R11_MS_QSC), (f16)(v1.w * INK_R11_MS_QSC)};
      }
      *(LDSP h8 *)(ink_r11_ms_sw_lds.kv.q + ink_r11_ms_sw_q_off(qr, c8)) = qh;
    }
    if (extent_lim > 0u) {
      if (tid() < q_rows * 2u) {
        const u32 qr = tid() >> 1u;
        const u32 h8i = tid() & 1u;
        h8 rh = (h8)(0.0f);
        if (qr < nq_tot) {
          const u32 tj = qr / INK_R11_MS_GQA;
          const u32 gi = qr - tj * INK_R11_MS_GQA;
          const G f32x4 *rp = (const G f32x4 *)((const G float *)((const G u8 *)R + (unsigned long)tj * rs) +
                                                (unsigned long)(h0 + gi) * INK_R11_MS_DREL);
          const f32x4 v0 = rp[2u * h8i + 0u];
          const f32x4 v1 = rp[2u * h8i + 1u];
          rh = (h8){(f16)(v0.x * INK_R11_MS_LOG2E), (f16)(v0.y * INK_R11_MS_LOG2E),
                    (f16)(v0.z * INK_R11_MS_LOG2E), (f16)(v0.w * INK_R11_MS_LOG2E),
                    (f16)(v1.x * INK_R11_MS_LOG2E), (f16)(v1.y * INK_R11_MS_LOG2E),
                    (f16)(v1.z * INK_R11_MS_LOG2E), (f16)(v1.w * INK_R11_MS_LOG2E)};
        }
        *(LDSP h8 *)(ink_r11_ms_sw_lds.kv.r + qr * 16u + 8u * h8i) = rh;
      }
    }
    lbarrier();

    const G kvt *kc_h = kc + (unsigned long)kh * INK_R11_MS_HD;
    const G kvt *vc_h = vc + (unsigned long)kh * INK_R11_MS_HD;

    const v8f zero = (v8f)(0.0f);
    v8f acc0 = zero, acc1 = zero;
    float m0 = -__builtin_inff(), ls0 = 0.0f;
    float m1 = -__builtin_inff(), ls1 = 0.0f;

    const u32 p_off = ink_r11_ms_sw_p_off(rl, hl);
    const u32 pos_q0 = pos0 + (rl / INK_R11_MS_GQA);
    const u32 pos_q1 = pos0 + ((16u + rl) / INK_R11_MS_GQA);
    const u32 w_q0 = (pos_q0 + 1u > win) ? (pos_q0 + 1u - win) : 0u;
    const u32 w_q1 = (pos_q1 + 1u > win) ? (pos_q1 + 1u - win) : 0u;

#pragma clang loop unroll(disable)
    for (u32 c0 = p0; c0 < p1; c0 += 128u) {
      const u32 kb = (u32)__builtin_amdgcn_readfirstlane((int)(c0 + 16u * w));
      const u32 chunk_end = (u32)__builtin_amdgcn_readfirstlane((int)((c0 + 128u < p1) ? (c0 + 128u) : p1));

      v8f sp0 = (v8f)(-__builtin_inff());
      v8f sp1 = (v8f)(-__builtin_inff());

      if (kb < chunk_end) {
        const u32 kp_row = kb + rl;
        const int row_ok = (kp_row < chunk_end);
        const u32 kp_cl = row_ok ? kp_row : (chunk_end - 1u);
        const u32 kv_row = ink_r11_ms_sw_wrap(kp_cl, kv_len, kv_mask, is_pow2);
        const G f16 *k_base = kc_h + (unsigned long)kv_row * kvd + 8u * hl;

        sp0 = zero;
        sp1 = zero;
        h8 kk0 = row_ok ? *(const G h8 *)(k_base + 0u) : (h8)(0.0f);
        h8 kk1 = row_ok ? *(const G h8 *)(k_base + 16u) : (h8)(0.0f);
#pragma unroll
        for (u32 ks = 0u; ks < 8u; ks += 2u) {
          const u32 c_a = 2u * ks + hl;
          const u32 c_b = 2u * (ks + 1u) + hl;
          const h8 q0_a = *(const LDSP h8 *)(ink_r11_ms_sw_lds.kv.q + ink_r11_ms_sw_q_off(rl, c_a));
          const h8 q0_b = *(const LDSP h8 *)(ink_r11_ms_sw_lds.kv.q + ink_r11_ms_sw_q_off(rl, c_b));
          sp0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk0, q0_a, sp0);
          sp0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk1, q0_b, sp0);
          if (npg > 1u) {
            const h8 q1_a = *(const LDSP h8 *)(ink_r11_ms_sw_lds.kv.q + ink_r11_ms_sw_q_off(16u + rl, c_a));
            const h8 q1_b = *(const LDSP h8 *)(ink_r11_ms_sw_lds.kv.q + ink_r11_ms_sw_q_off(16u + rl, c_b));
            sp1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk0, q1_a, sp1);
            sp1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk1, q1_b, sp1);
          }
          if (ks + 2u < 8u) {
            kk0 = row_ok ? *(const G h8 *)(k_base + 16u * (ks + 2u)) : (h8)(0.0f);
            kk1 = row_ok ? *(const G h8 *)(k_base + 16u * (ks + 3u)) : (h8)(0.0f);
          }
          __builtin_amdgcn_sched_barrier(0);
        }

        if (extent_lim > 0u) {
          const int d_min_s = (int)pos0 - (int)(kb + 15u);
          const u32 d_base = d_min_s > 0 ? (u32)d_min_s : 0u;
          if (d_base < extent_lim) {
            LDSP float *sc_row = ((LDSP float *)ink_r11_ms_sw_lds.kv.scratch.v_sc[w]) + rl * 8u;
            const h8 ra0 = *(const LDSP h8 *)(ink_r11_ms_sw_lds.kv.r + rl * 16u + 8u * hl);
            const u32 d0 = d_base + rl;
            h8 b0 = (h8)(0.0f);
            if (d0 < extent_lim) {
              const G f32x4 *rp = (const G f32x4 *)(rel_bias + (unsigned long)d0 * 16ul + 8ul * hl);
              const f32x4 v0 = rp[0], v1 = rp[1];
              b0 = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
            }
            const v8f rb0_0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0, ra0, zero);
            ink_r11_ms_sw_apply_rb16(rb0_0, 0u, hl, kb, pos_q0, w_q0, extent_lim, d_base, sc_row, &sp0);
            if (npg > 1u) {
              const h8 ra1 = *(const LDSP h8 *)(ink_r11_ms_sw_lds.kv.r + (16u + rl) * 16u + 8u * hl);
              const v8f rb0_1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0, ra1, zero);
              ink_r11_ms_sw_apply_rb16(rb0_1, 0u, hl, kb, pos_q1, w_q1, extent_lim, d_base, sc_row, &sp1);
            }
            if (d_base + 16u < extent_lim && (pos0 + T - 1u) >= kb + d_base + 16u) {
              const u32 d1 = d_base + 16u + rl;
              h8 b1 = (h8)(0.0f);
              if (d1 < extent_lim) {
                const G f32x4 *rp = (const G f32x4 *)(rel_bias + (unsigned long)d1 * 16ul + 8ul * hl);
                const f32x4 v0 = rp[0], v1 = rp[1];
                b1 = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
              }
              const v8f rb1_0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1, ra0, zero);
              ink_r11_ms_sw_apply_rb16(rb1_0, 16u, hl, kb, pos_q0, w_q0, extent_lim, d_base, sc_row, &sp0);
              if (npg > 1u) {
                const h8 ra1 = *(const LDSP h8 *)(ink_r11_ms_sw_lds.kv.r + (16u + rl) * 16u + 8u * hl);
                const v8f rb1_1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1, ra1, zero);
                ink_r11_ms_sw_apply_rb16(rb1_1, 16u, hl, kb, pos_q1, w_q1, extent_lim, d_base, sc_row, &sp1);
              }
            }
          }
        }

        /* Exact BOTH-SIDED SWA mask: mask out kp < w_q, kp > pos_q, and kp >= chunk_end. */
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) {
          const u32 kp = kb + 8u * hl + v;
          if (kp >= chunk_end || kp < w_q0 || kp > pos_q0) sp0[v] = -__builtin_inff();
          if (npg > 1u && (kp >= chunk_end || kp < w_q1 || kp > pos_q1)) sp1[v] = -__builtin_inff();
        }
      }

      u32x4 vr0 = (u32x4)(0u);
      {
        const u32 vp = c0 + rl;
        if (vp < chunk_end) {
          const u32 kv_v = ink_r11_ms_sw_wrap(vp, kv_len, kv_mask, is_pow2);
          vr0 = *(const G u32x4 *)(vc_h + (unsigned long)kv_v * kvd + 16u * w + 8u * hl);
        }
      }
      lbarrier();

      float mx0 = sp0[0];
#pragma unroll
      for (u32 v = 1u; v < 8u; v++) mx0 = __builtin_fmaxf(mx0, sp0[v]);
      mx0 = __builtin_fmaxf(mx0, ink_r11_ms_xrow(mx0));
      if (hl == 0u) ink_r11_ms_sw_lds.kv.scratch.red.mx[0][16u * w + rl] = mx0;

      if (npg > 1u) {
        float mx1 = sp1[0];
#pragma unroll
        for (u32 v = 1u; v < 8u; v++) mx1 = __builtin_fmaxf(mx1, sp1[v]);
        mx1 = __builtin_fmaxf(mx1, ink_r11_ms_xrow(mx1));
        if (hl == 0u) ink_r11_ms_sw_lds.kv.scratch.red.mx[1][16u * w + rl] = mx1;
      }
      lbarrier();

      float mc0 = ink_r11_ms_sw_lds.kv.scratch.red.mx[0][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) mc0 = __builtin_fmaxf(mc0, ink_r11_ms_sw_lds.kv.scratch.red.mx[0][16u * x + rl]);
      const float mn0 = __builtin_fmaxf(m0, mc0);
      const float rs0 = (ls0 > 0.0f && mn0 > -1e30f) ? __builtin_amdgcn_exp2f(m0 - mn0) : 0.0f;
      acc0 *= rs0; ls0 *= rs0; m0 = mn0;

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
      lc0 += ink_r11_ms_xrow(lc0);
      if (hl == 0u) ink_r11_ms_sw_lds.kv.scratch.red.ls[0][16u * w + rl] = lc0;

      h8 ph1 = (h8)(0.0f);
      if (npg > 1u) {
        float mc1 = ink_r11_ms_sw_lds.kv.scratch.red.mx[1][rl];
#pragma unroll
        for (u32 x = 1u; x < 8u; x++) mc1 = __builtin_fmaxf(mc1, ink_r11_ms_sw_lds.kv.scratch.red.mx[1][16u * x + rl]);
        const float mn1 = __builtin_fmaxf(m1, mc1);
        const float rs1 = (ls1 > 0.0f && mn1 > -1e30f) ? __builtin_amdgcn_exp2f(m1 - mn1) : 0.0f;
        acc1 *= rs1; ls1 *= rs1; m1 = mn1;

        float lc1 = 0.0f;
        if (mn1 > -1e30f) {
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) {
            const float p = __builtin_amdgcn_exp2f(sp1[v] - mn1);
            lc1 += p;
            ph1[v] = (f16)p;
          }
        }
        lc1 += ink_r11_ms_xrow(lc1);
        if (hl == 0u) ink_r11_ms_sw_lds.kv.scratch.red.ls[1][16u * w + rl] = lc1;
      }

      if (w == 0u) {
        *(LDSP h8 *)(ink_r11_ms_sw_lds.kv.p_tile[0][0] + p_off) = ph0;
        if (npg > 1u) *(LDSP h8 *)(ink_r11_ms_sw_lds.kv.p_tile[0][1] + p_off) = ph1;
      }
      lbarrier();

      float sum_l0 = ink_r11_ms_sw_lds.kv.scratch.red.ls[0][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) sum_l0 += ink_r11_ms_sw_lds.kv.scratch.red.ls[0][16u * x + rl];
      ls0 += sum_l0;
      if (npg > 1u) {
        float sum_l1 = ink_r11_ms_sw_lds.kv.scratch.red.ls[1][rl];
#pragma unroll
        for (u32 x = 1u; x < 8u; x++) sum_l1 += ink_r11_ms_sw_lds.kv.scratch.red.ls[1][16u * x + rl];
        ls1 += sum_l1;
      }
      lbarrier();

      LDSP u32 *v_sc_w = ink_r11_ms_sw_lds.kv.scratch.v_sc[w];
#pragma clang loop unroll(disable)
      for (u32 kt = 0u; kt < 8u; kt++) {
        if (c0 + 16u * kt >= chunk_end) break;
        const int has_nxt = (kt + 1u < 8u) && (c0 + 16u * (kt + 1u) < chunk_end);
        u32x4 vr_nxt = (u32x4)(0u);
        if (has_nxt) {
          const u32 vp_n = c0 + 16u * (kt + 1u) + rl;
          if (vp_n < chunk_end) {
            const u32 kv_vn = ink_r11_ms_sw_wrap(vp_n, kv_len, kv_mask, is_pow2);
            vr_nxt = *(const G u32x4 *)(vc_h + (unsigned long)kv_vn * kvd + 16u * w + 8u * hl);
          }
        }
        __builtin_amdgcn_sched_barrier(0);

        const h8 vt_a = ink_r11_ms_sw_v_transpose(vr0, v_sc_w, hl, rl);
        const h8 pt0 = *(const LDSP h8 *)(ink_r11_ms_sw_lds.kv.p_tile[kt & 1u][0] + p_off);
        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt0, acc0);
        if (npg > 1u) {
          const h8 pt1 = *(const LDSP h8 *)(ink_r11_ms_sw_lds.kv.p_tile[kt & 1u][1] + p_off);
          acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt1, acc1);
        }

        if (has_nxt && w == kt + 1u) {
          *(LDSP h8 *)(ink_r11_ms_sw_lds.kv.p_tile[(kt + 1u) & 1u][0] + p_off) = ph0;
          if (npg > 1u) *(LDSP h8 *)(ink_r11_ms_sw_lds.kv.p_tile[(kt + 1u) & 1u][1] + p_off) = ph1;
        }
        lbarrier();
        vr0 = vr_nxt;
      }
    }

    lbarrier();
    if (w == 0u && hl == 0u) {
      ink_r11_ms_sw_lds.ph4.m[rl] = (ls0 > 0.0f) ? (m0 * INK_R11_MS_LN2) : -1e30f;
      ink_r11_ms_sw_lds.ph4.l[rl] = ls0;
      if (npg > 1u) {
        ink_r11_ms_sw_lds.ph4.m[16u + rl] = (ls1 > 0.0f) ? (m1 * INK_R11_MS_LN2) : -1e30f;
        ink_r11_ms_sw_lds.ph4.l[16u + rl] = ls1;
      }
    }
    lbarrier();

#pragma clang loop unroll(disable)
    for (u32 pg = 0u; pg < npg; pg++) {
      const v8f a_cur = (pg == 0u) ? acc0 : acc1;
#pragma unroll
      for (u32 qb = 0u; qb < 2u; qb++) {
        const u32 q_base = 16u * pg + 8u * qb;
        if (q_base >= nq_tot) break;
        if ((rl >> 3u) == qb) {
          const u32 q_loc = rl & 7u;
          LDSP float *dst_w = ink_r11_ms_sw_lds.ph4.o_stg[q_loc] + 16u * w + 8u * hl;
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) dst_w[v] = a_cur[v];
        }
        lbarrier();

        const u32 qi = q_base + w;
        if (qi < nq_tot) {
          const u32 t = qi / INK_R11_MS_GQA;
          const u32 h = h0 + (qi - t * INK_R11_MS_GQA);
          const f32x4 ov = ((const LDSP f32x4 *)ink_r11_ms_sw_lds.ph4.o_stg[w])[l];
          if (nsplit == 1u) {
            const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
            const float sv = st[h];
            const float mq = ink_r11_ms_sw_lds.ph4.m[qi];
            const float lq = ink_r11_ms_sw_lds.ph4.l[qi];
            const float M = (lq > 0.0f) ? __builtin_fmaxf(sv, mq) : sv;
            const float wx = (lq > 0.0f) ? __builtin_expf(mq - M) : 0.0f;
            const float L = __builtin_expf(sv - M) + lq * wx;
            const float sc_o = wx / L;
            const G float *gt_h = agate ? ((const G float *)((const G u8 *)agate + (unsigned long)t * gs) +
                                           (unsigned long)h * INK_R11_MS_HD)
                                        : (const G float *)0;
            G float *ot_h = out ? ((G float *)((G u8 *)out + (unsigned long)t * os) +
                                   (unsigned long)h * INK_R11_MS_HD)
                                : (G float *)0;
            G u8 *oqt = oq ? (oq + (unsigned long)t * oqs) : (G u8 *)0;
            ink_r11_ms_gate_q8_head_vec4(ov * sc_o, gt_h, ot_h, oqt, h, n_head, l);
          } else {
            G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R11_MS_PSTR;
            if (l == 0u) {
              pp[0] = ink_r11_ms_sw_lds.ph4.m[qi];
              pp[1] = ink_r11_ms_sw_lds.ph4.l[qi];
            }
            G float *po = pp + 2u + 4u * l;
            po[0] = ov.x; po[1] = ov.y; po[2] = ov.z; po[3] = ov.w;
          }
        }
        lbarrier();
      }
    }
    if (nsplit == 1u) return;
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r11_ms_sw_lds.ph4.is_last =
        (__scoped_atomic_fetch_add(&count[kh], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r11_ms_sw_lds.ph4.is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  for (u32 qi = w; qi < nq_tot; qi += 8u) {
    const u32 t = qi / INK_R11_MS_GQA;
    const u32 h = h0 + (qi - t * INK_R11_MS_GQA);
    const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
    const G float *gt_h = agate ? ((const G float *)((const G u8 *)agate + (unsigned long)t * gs) +
                                   (unsigned long)h * INK_R11_MS_HD)
                                : (const G float *)0;
    G float *ot_h = out ? ((G float *)((G u8 *)out + (unsigned long)t * os) +
                           (unsigned long)h * INK_R11_MS_HD)
                        : (G float *)0;
    G u8 *oqt = oq ? (oq + (unsigned long)t * oqs) : (G u8 *)0;
    const G float *ph = part + (unsigned long)((t * n_head + h) * nsplit) * INK_R11_MS_PSTR;
    ink_r11_ms_sw_merge_head_gate_q8(ph, st[h], nsplit, l, ink_r11_ms_sw_lds.ph4.wt[w],
                                     gt_h, ot_h, oqt, h, n_head);
  }
  barrier();
  if (tid() == 0u) count[kh] = 0u;
}

/* ============================================================================
 * Kernel 3 (H21-QT2): ie_inkling_rel_attn_ver_swa_qt2_r11
 * Ultra-lean <= 68 VGPR (4 WGs/CU) QT=2 Exact-SHA FP32 Scalar SWA Verify
 * Companion (with optional fused OP_GATE(sigmoid) + Q8_0 epilogue!).
 *   - 2 consecutive tokens/WG (6 GQA queries sharing each K/V load -> 2x fewer
 *     SWA KV reads vs ver_r4, 0 waste at T=6 & T=8, only 24 acc VGPRs -> 4 WGs/CU!).
 *   - Exact FP32 online-softmax arithmetic matching ver_r4 (no FP16 Q/P rounding!).
 *   - Grid: (n_kv * nsplit, (T + 1u) / 2u, 1u), NT = 256, 12,484 B LDS, 0 spills.
 *   - Counter: count[t0 * n_kv + kh] (self-resets to 0; compatible with b->at_count!).
 * ========================================================================== */

#define INK_R11_MS_QT2_UPD(m_v, s_v, a_v, d_val, vf_val)                           \
  do {                                                                             \
    const float _mn = __builtin_fmaxf((m_v), (d_val));                             \
    const float _c = ((s_v) > 0.0f) ? __builtin_expf((m_v) - _mn) : 0.0f;          \
    const float _e = __builtin_expf((d_val) - _mn);                                \
    (s_v) = ink_r11_ms_sgpr_f((s_v) * _c + _e);                                    \
    (m_v) = ink_r11_ms_sgpr_f(_mn);                                                \
    (a_v) = (a_v) * _c + _e * (vf_val);                                            \
  } while (0)

KERNEL ie_inkling_rel_attn_ver_swa_qt2_r11(
    const G float *q, const G kvt *kc, const G kvt *vc,
    const G float *R, const G float *rel_bias, const G float *sink,
    G float *part, G float *out, u32 pos0, u32 T, u32 kv_len, u32 hd,
    u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
    u32 qs, u32 rs, u32 ss, u32 os, G u32 *count,
    const G float *agate, G u8 *oq, u32 gs, u32 oqs) {
  static LDS union {
    struct {
      float q[2u * INK_R11_MS_GQA * INK_R11_MS_HD] __attribute__((aligned(16)));
      float r[2u * INK_R11_MS_GQA * INK_R11_MS_DREL] __attribute__((aligned(16)));
    } qr; /* 3,456 B */
    struct {
      float m[INK_R11_MS_GQA][8];
      float l[INK_R11_MS_GQA][8];
      float o[INK_R11_MS_GQA][8][INK_R11_MS_HD] __attribute__((aligned(16)));
    } red; /* 12,480 B */
  } ink_r11_ms_q2_lds __attribute__((aligned(16)));
  static LDS u32 ink_r11_ms_q2_last;

  const u32 t0 = (u32)__builtin_amdgcn_readfirstlane((int)(tokid() * 2u));
  if (t0 >= T || win == 0u || hd != INK_R11_MS_HD || d_rel != INK_R11_MS_DREL ||
      n_kv == 0u || n_head != INK_R11_MS_GQA * n_kv || kv_len == 0u || nsplit == 0u)
    return;

  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() / nsplit));
  const u32 s = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() % nsplit));
  if (kh >= n_kv) return;

  const u32 h0 = kh * INK_R11_MS_GQA;
  const u32 kvd = n_kv * INK_R11_MS_HD;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 nq = (T - t0 >= 2u) ? 2u : 1u;
  const u32 nb_q = n_head << 2u;

  const u32 pos_a = pos0 + t0;
  const u32 pos_b = pos_a + 1u;
  const u32 wa = (pos_a + 1u > win) ? (pos_a + 1u - win) : 0u;
  const u32 wb = (pos_b + 1u > win) ? (pos_b + 1u - win) : 0u;
  const u32 pend = pos_a + nq;
  const u32 np = pend - wa;
  const u32 ch = (np + nsplit - 1u) / nsplit;
  const u32 off0 = s * ch;
  const u32 p0 = (off0 < np) ? (wa + off0) : pend;
  const u32 p1 = (off0 < np && off0 + ch < np) ? (wa + off0 + ch) : pend;

  if (nsplit > 1u && p0 >= p1) {
    for (u32 qi = 0u; qi < nq; qi++) {
      const u32 t = t0 + qi;
      for (u32 idx = tid(); idx < INK_R11_MS_GQA * INK_R11_MS_HD; idx += NT) {
        const u32 g = idx >> 7u;
        const u32 e = idx & 127u;
        const u32 h = h0 + g;
        G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R11_MS_PSTR;
        pp[2u + e] = 0.0f;
        if (e == 0u) { pp[0] = -1e30f; pp[1] = 0.0f; }
      }
    }
  } else {
    const u32 extent_lim = (win < rel_extent) ? win : rel_extent;
    const float scale = 1.0f / __builtin_sqrtf((float)INK_R11_MS_HD);

    /* Stage Q (<= 6 heads * 128 f32 = 3 KiB) + R (<= 6 heads * 16 f32 = 384 B) in LDS. */
    const u32 nq_vecs = nq * (INK_R11_MS_GQA * 32u);
    for (u32 idx = tid(); idx < nq_vecs; idx += NT) {
      const u32 qg = idx >> 5u;
      const u32 e4 = idx & 31u;
      const u32 qi = qg / INK_R11_MS_GQA;
      const u32 gi = qg - qi * INK_R11_MS_GQA;
      const G float *qt = (const G float *)((const G u8 *)q + (unsigned long)(t0 + qi) * qs);
      ((LDSP f32x4 *)(ink_r11_ms_q2_lds.qr.q + (unsigned long)qg * INK_R11_MS_HD))[e4] =
          ((const G f32x4 *)(qt + (unsigned long)(h0 + gi) * INK_R11_MS_HD))[e4];
    }
    if (tid() < nq * (INK_R11_MS_GQA * INK_R11_MS_DREL)) {
      const u32 idx = tid();
      const u32 qg = idx >> 4u;
      const u32 e = idx & 15u;
      const u32 qi = qg / INK_R11_MS_GQA;
      const u32 gi = qg - qi * INK_R11_MS_GQA;
      const G float *Rt = (const G float *)((const G u8 *)R + (unsigned long)(t0 + qi) * rs);
      ink_r11_ms_q2_lds.qr.r[(unsigned long)qg * INK_R11_MS_DREL + e] = Rt[(unsigned long)(h0 + gi) * INK_R11_MS_DREL + e];
    }
    lbarrier();

    f32x4 a00 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 a01 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 a02 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 a10 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 a11 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 a12 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};

    float m00 = -__builtin_inff(), s00 = 0.0f;
    float m01 = -__builtin_inff(), s01 = 0.0f;
    float m02 = -__builtin_inff(), s02 = 0.0f;
    float m10 = -__builtin_inff(), s10 = 0.0f;
    float m11 = -__builtin_inff(), s11 = 0.0f;
    float m12 = -__builtin_inff(), s12 = 0.0f;

    const G kvt *kc_h = kc + (unsigned long)kh * INK_R11_MS_HD;
    const G kvt *vc_h = vc + (unsigned long)kh * INK_R11_MS_HD;

    u32 kp = p0 + w;
    if (kp < p1) {
      u32 kvp = kp % kv_len;
      kv4 kh4 = ((const G kv4 *)(kc_h + (unsigned long)kvp * kvd))[l];
      kv4 vh4 = ((const G kv4 *)(vc_h + (unsigned long)kvp * kvd))[l];

#pragma unroll 1
      for (;;) {
        const u32 kp_nxt = kp + 8u;
        u32 kvp_nxt = kvp;
        kv4 kh4_nxt = kh4, vh4_nxt = vh4;
        if (kp_nxt < p1) {
          kvp_nxt = kvp + 8u;
          if (kvp_nxt >= kv_len) kvp_nxt = (kv_len >= 8u) ? (kvp_nxt - kv_len) : (kp_nxt % kv_len);
          kh4_nxt = ((const G kv4 *)(kc_h + (unsigned long)kvp_nxt * kvd))[l];
          vh4_nxt = ((const G kv4 *)(vc_h + (unsigned long)kvp_nxt * kvd))[l];
        }

        const f32x4 kf = __builtin_convertvector(kh4, f32x4);
        const f32x4 vf = __builtin_convertvector(vh4, f32x4);

        if (kp <= pos_a) {
          const LDSP f32x4 *qp0 = (const LDSP f32x4 *)ink_r11_ms_q2_lds.qr.q;
          float d0 = ink_r11_ms_sgpr_f(wave_sum_all(ink_r11_ms_dot4(qp0[0u * 32u + l], kf)) * scale);
          float d1 = ink_r11_ms_sgpr_f(wave_sum_all(ink_r11_ms_dot4(qp0[1u * 32u + l], kf)) * scale);
          float d2 = ink_r11_ms_sgpr_f(wave_sum_all(ink_r11_ms_dot4(qp0[2u * 32u + l], kf)) * scale);
          const u32 dist = pos_a - kp;
          if (dist < extent_lim) {
            const float rb = (l < INK_R11_MS_DREL) ? rel_bias[(unsigned long)dist * INK_R11_MS_DREL + l] : 0.0f;
            const LDSP float *rp0 = ink_r11_ms_q2_lds.qr.r;
            const float r0 = (l < INK_R11_MS_DREL) ? rp0[0u * 16u + l] : 0.0f;
            const float r1 = (l < INK_R11_MS_DREL) ? rp0[1u * 16u + l] : 0.0f;
            const float r2 = (l < INK_R11_MS_DREL) ? rp0[2u * 16u + l] : 0.0f;
            d0 = ink_r11_ms_sgpr_f(d0 + ink_r11_ms_halfwave_sum_all(r0 * rb));
            d1 = ink_r11_ms_sgpr_f(d1 + ink_r11_ms_halfwave_sum_all(r1 * rb));
            d2 = ink_r11_ms_sgpr_f(d2 + ink_r11_ms_halfwave_sum_all(r2 * rb));
          }
          INK_R11_MS_QT2_UPD(m00, s00, a00, d0, vf);
          INK_R11_MS_QT2_UPD(m01, s01, a01, d1, vf);
          INK_R11_MS_QT2_UPD(m02, s02, a02, d2, vf);
        }

        if (nq > 1u && kp >= wb && kp <= pos_b) {
          const LDSP f32x4 *qp1 = (const LDSP f32x4 *)(ink_r11_ms_q2_lds.qr.q + INK_R11_MS_GQA * INK_R11_MS_HD);
          float d0 = ink_r11_ms_sgpr_f(wave_sum_all(ink_r11_ms_dot4(qp1[0u * 32u + l], kf)) * scale);
          float d1 = ink_r11_ms_sgpr_f(wave_sum_all(ink_r11_ms_dot4(qp1[1u * 32u + l], kf)) * scale);
          float d2 = ink_r11_ms_sgpr_f(wave_sum_all(ink_r11_ms_dot4(qp1[2u * 32u + l], kf)) * scale);
          const u32 dist = pos_b - kp;
          if (dist < extent_lim) {
            const float rb = (l < INK_R11_MS_DREL) ? rel_bias[(unsigned long)dist * INK_R11_MS_DREL + l] : 0.0f;
            const LDSP float *rp1 = ink_r11_ms_q2_lds.qr.r + INK_R11_MS_GQA * INK_R11_MS_DREL;
            const float r0 = (l < INK_R11_MS_DREL) ? rp1[0u * 16u + l] : 0.0f;
            const float r1 = (l < INK_R11_MS_DREL) ? rp1[1u * 16u + l] : 0.0f;
            const float r2 = (l < INK_R11_MS_DREL) ? rp1[2u * 16u + l] : 0.0f;
            d0 = ink_r11_ms_sgpr_f(d0 + ink_r11_ms_halfwave_sum_all(r0 * rb));
            d1 = ink_r11_ms_sgpr_f(d1 + ink_r11_ms_halfwave_sum_all(r1 * rb));
            d2 = ink_r11_ms_sgpr_f(d2 + ink_r11_ms_halfwave_sum_all(r2 * rb));
          }
          INK_R11_MS_QT2_UPD(m10, s10, a10, d0, vf);
          INK_R11_MS_QT2_UPD(m11, s11, a11, d1, vf);
          INK_R11_MS_QT2_UPD(m12, s12, a12, d2, vf);
        }

        if (kp_nxt >= p1) break;
        kp = kp_nxt; kvp = kvp_nxt; kh4 = kh4_nxt; vh4 = vh4_nxt;
      }
    }
    lbarrier();

    /* Reduce each token qi in {0, 1} through overlaid 12.2 KiB red LDS! */
    for (u32 qi = 0u; qi < nq; qi++) {
      const u32 t = t0 + qi;
      if (l == 0u) {
        ink_r11_ms_q2_lds.red.m[0][w] = (qi == 0u) ? m00 : m10;
        ink_r11_ms_q2_lds.red.l[0][w] = (qi == 0u) ? s00 : s10;
        ink_r11_ms_q2_lds.red.m[1][w] = (qi == 0u) ? m01 : m11;
        ink_r11_ms_q2_lds.red.l[1][w] = (qi == 0u) ? s01 : s11;
        ink_r11_ms_q2_lds.red.m[2][w] = (qi == 0u) ? m02 : m12;
        ink_r11_ms_q2_lds.red.l[2][w] = (qi == 0u) ? s02 : s12;
      }
      ((LDSP f32x4 *)&ink_r11_ms_q2_lds.red.o[0][w][0])[l] = (qi == 0u) ? a00 : a10;
      ((LDSP f32x4 *)&ink_r11_ms_q2_lds.red.o[1][w][0])[l] = (qi == 0u) ? a01 : a11;
      ((LDSP f32x4 *)&ink_r11_ms_q2_lds.red.o[2][w][0])[l] = (qi == 0u) ? a02 : a12;
      lbarrier();

      if (nsplit == 1u) {
        const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
        const G float *gt = agate ? (const G float *)((const G u8 *)agate + (unsigned long)t * gs) : (const G float *)0;
        G float *ot = out ? (G float *)((G u8 *)out + (unsigned long)t * os) : (G float *)0;
        G u8 *oqt = oq ? (oq + (unsigned long)t * oqs) : (G u8 *)0;
        for (u32 idx = tid(); idx < INK_R11_MS_GQA * INK_R11_MS_HD; idx += NT) {
          const u32 g = idx >> 7u;
          const u32 e = idx & 127u;
          const u32 h = h0 + g;
          const float sv = st[h];
          float M = sv;
          for (u32 x = 0u; x < 8u; x++)
            if (ink_r11_ms_q2_lds.red.l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r11_ms_q2_lds.red.m[g][x]);
          float L = __builtin_expf(sv - M);
          float o = 0.0f;
          for (u32 x = 0u; x < 8u; x++) {
            const float lx = ink_r11_ms_q2_lds.red.l[g][x];
            if (lx > 0.0f) {
              const float wx = __builtin_expf(ink_r11_ms_q2_lds.red.m[g][x] - M);
              L += lx * wx;
              o += ink_r11_ms_q2_lds.red.o[g][x][e] * wx;
            }
          }
          const u32 out_idx = h * INK_R11_MS_HD + e;
          float v = o / L;
          if (gt) v = ink_r11_ms_gate_f(v, gt[out_idx]);
          if (ot) ot[out_idx] = v;
          if (oqt) quant_wave(v, out_idx >> 5u, oqt, nb_q);
        }
      } else {
        for (u32 idx = tid(); idx < INK_R11_MS_GQA * INK_R11_MS_HD; idx += NT) {
          const u32 g = idx >> 7u;
          const u32 e = idx & 127u;
          const u32 h = h0 + g;
          float M = -__builtin_inff();
          for (u32 x = 0u; x < 8u; x++)
            if (ink_r11_ms_q2_lds.red.l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r11_ms_q2_lds.red.m[g][x]);
          G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R11_MS_PSTR;
          float o = 0.0f, L = 0.0f;
          for (u32 x = 0u; x < 8u; x++) {
            const float lx = ink_r11_ms_q2_lds.red.l[g][x];
            if (lx > 0.0f) {
              const float wx = __builtin_expf(ink_r11_ms_q2_lds.red.m[g][x] - M);
              o += ink_r11_ms_q2_lds.red.o[g][x][e] * wx;
              if (e == 0u) L += lx * wx;
            }
          }
          pp[2u + e] = o;
          if (e == 0u) { pp[0] = M; pp[1] = L; }
        }
      }
      if (qi + 1u < nq) lbarrier();
    }
    if (nsplit == 1u) return;
  }

  const u32 cidx = t0 * n_kv + kh;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r11_ms_q2_last =
        (__scoped_atomic_fetch_add(&count[cidx], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u);
  }
  barrier();
  if (!ink_r11_ms_q2_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const int wt_in_lds = (nsplit <= INK_R11_MS_HD);
  for (u32 qi = 0u; qi < nq; qi++) {
    const u32 t = t0 + qi;
    const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
    const G float *gt = agate ? (const G float *)((const G u8 *)agate + (unsigned long)t * gs) : (const G float *)0;
    G float *ot = out ? (G float *)((G u8 *)out + (unsigned long)t * os) : (G float *)0;
    G u8 *oqt = oq ? (oq + (unsigned long)t * oqs) : (G u8 *)0;

    G float *ph0 = part + (unsigned long)((t * n_head + (h0 + 0u)) * nsplit) * INK_R11_MS_PSTR;
    G float *ph1 = part + (unsigned long)((t * n_head + (h0 + 1u)) * nsplit) * INK_R11_MS_PSTR;
    G float *ph2 = part + (unsigned long)((t * n_head + (h0 + 2u)) * nsplit) * INK_R11_MS_PSTR;

    float mv0 = (tid() == 0u) ? st[h0 + 0u] : -__builtin_inff();
    float mv1 = (tid() == 0u) ? st[h0 + 1u] : -__builtin_inff();
    float mv2 = (tid() == 0u) ? st[h0 + 2u] : -__builtin_inff();
    for (u32 sp = tid(); sp < nsplit; sp += NT) {
      const unsigned long off = (unsigned long)sp * INK_R11_MS_PSTR;
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
      ink_r11_ms_q2_lds.red.m[0][w] = mv0;
      ink_r11_ms_q2_lds.red.m[1][w] = mv1;
      ink_r11_ms_q2_lds.red.m[2][w] = mv2;
    }
    lbarrier();

    float M2[INK_R11_MS_GQA];
    for (u32 g = 0u; g < INK_R11_MS_GQA; g++) {
      float mx = ink_r11_ms_q2_lds.red.m[g][0];
      for (u32 x = 1u; x < 8u; x++) mx = __builtin_fmaxf(mx, ink_r11_ms_q2_lds.red.m[g][x]);
      M2[g] = mx;
    }

    float lv0 = 0.0f, lv1 = 0.0f, lv2 = 0.0f;
    for (u32 sp = tid(); sp < nsplit; sp += NT) {
      const unsigned long off = (unsigned long)sp * INK_R11_MS_PSTR;
      const float lp0 = ph0[off + 1u];
      const float wt0 = (lp0 > 0.0f) ? __builtin_expf(ph0[off] - M2[0]) : 0.0f;
      if (wt_in_lds) ink_r11_ms_q2_lds.red.o[0][0][sp] = wt0; else ph0[off] = wt0;
      lv0 += lp0 * wt0;

      const float lp1 = ph1[off + 1u];
      const float wt1 = (lp1 > 0.0f) ? __builtin_expf(ph1[off] - M2[1]) : 0.0f;
      if (wt_in_lds) ink_r11_ms_q2_lds.red.o[1][0][sp] = wt1; else ph1[off] = wt1;
      lv1 += lp1 * wt1;

      const float lp2 = ph2[off + 1u];
      const float wt2 = (lp2 > 0.0f) ? __builtin_expf(ph2[off] - M2[2]) : 0.0f;
      if (wt_in_lds) ink_r11_ms_q2_lds.red.o[2][0][sp] = wt2; else ph2[off] = wt2;
      lv2 += lp2 * wt2;
    }
    lv0 = wave_tree_f(lv0);
    lv1 = wave_tree_f(lv1);
    lv2 = wave_tree_f(lv2);
    if (l == 0u) {
      ink_r11_ms_q2_lds.red.l[0][w] = lv0;
      ink_r11_ms_q2_lds.red.l[1][w] = lv1;
      ink_r11_ms_q2_lds.red.l[2][w] = lv2;
    }
    if (wt_in_lds) lbarrier(); else barrier();

    float L2[INK_R11_MS_GQA];
    for (u32 g = 0u; g < INK_R11_MS_GQA; g++) {
      float ls = 0.0f;
      for (u32 x = 0u; x < 8u; x++) ls += ink_r11_ms_q2_lds.red.l[g][x];
      L2[g] = ls + __builtin_expf(st[h0 + g] - M2[g]);
    }

    for (u32 idx = tid(); idx < INK_R11_MS_GQA * INK_R11_MS_HD; idx += NT) {
      const u32 g = idx >> 7u;
      const u32 e = idx & 127u;
      const u32 h = h0 + g;
      const G float *ph = part + (unsigned long)((t * n_head + h) * nsplit) * INK_R11_MS_PSTR;
      float o = 0.0f;
      if (wt_in_lds) {
        for (u32 sp = 0u; sp < nsplit; sp++) o += ph[(unsigned long)sp * INK_R11_MS_PSTR + 2u + e] * ink_r11_ms_q2_lds.red.o[g][0][sp];
      } else {
        for (u32 sp = 0u; sp < nsplit; sp++) o += ph[(unsigned long)sp * INK_R11_MS_PSTR + 2u + e] * ph[(unsigned long)sp * INK_R11_MS_PSTR];
      }
      const u32 out_idx = h * INK_R11_MS_HD + e;
      float v = o / L2[g];
      if (gt) v = ink_r11_ms_gate_f(v, gt[out_idx]);
      if (ot) ot[out_idx] = v;
      if (oqt) quant_wave(v, out_idx >> 5u, oqt, nb_q);
    }
    if (qi + 1u < nq) lbarrier();
  }

  if (tid() == 0u) count[cidx] = 0u;
}

#undef INK_R11_MS_QT2_UPD
#undef INK_R11_MS_FMA4
#undef INK_R11_MS_ADD4
#undef INK_R11_MS_QSC
#undef INK_R11_MS_LN2
#undef INK_R11_MS_LOG2E
#undef INK_R11_MS_PSTR
#undef INK_R11_MS_GQA
#undef INK_R11_MS_DREL
#undef INK_R11_MS_HD
#undef INK_R11_MS_LDS_MAXNB
#undef INK_R11_MS_SB_BYTES
#undef INK_R11_MS_DEF_KRN_TICKS
#undef INK_R11_MS_DEF_BAR_TICKS
#undef INK_R11_MS_MAX_GRID
#undef INK_R11_MS_TH_HT
#undef INK_R11_MS_RSRC_W3

#endif /* !IE_MEGA */
