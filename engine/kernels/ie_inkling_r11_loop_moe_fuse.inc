/* ============================================================================
 * Inkling-2 Round 11: Decode Launch Collapse & Verify Cliff-Fix Kernels
 *   (H17 1-Launch 4-Ctrl + T=6->8 Cliff Fix + H20 QKV+SConv+KV + H18 Wo+SConv
 *    + #20 #9 16K T=1 Global Attention)
 * Target: RX 9070 XT (gfx1201, RDNA4, wave32, NT=256, 64 CUs, 64 MiB MALL)
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r11_lm_ /
 * INK_R11_LM_ and all macros are #undef'd at EOF.
 *
 * Exported Kernels (0/0 spills, priv=0, scratch=0, 0 v_fma_mix*_f16 on KV):
 *   1. ie_inkling_lc_wfg_gate_norm_r11
 *        Drop-in 54-arg T=1..16 fix for #20's +1.70 ms T=6->8 cliff on r6
 *        (143 VGPR / 2 WGs/CU -> <= 64 VGPR / 4 WGs/CU, coalesced 32-lane
 *        inp/msg/tmp access, hoisted nw + double-buffered norm tail).
 *   2. ie_inkling_lc_pipe_attn_r11
 *        H17 1-launch 3-controller attention loop pipeline (prenorm+down+wfg+
 *        parallel-T RMSNorm+Q8, grid <= 64, WDDM 0x83 250ms/900ms poison guard,
 *        preserves msg_qm for lc_mlp, -48 launches/step).
 *   3. ie_inkling_lc_pipe_mlp_r11
 *        H17 1-launch MLP loop controller pipeline (optional Stage 0 attn_res
 *        + prenorm + down + wfg + parallel-T RMSNorm+Q8 -> ffn_in + ffn_in_q8,
 *        grid <= 64, WDDM 0x83 poison guard, -48/-72 launches/step).
 *   4. ie_inkling_qkv_sconv_kv_r11
 *        H20 single-launch QKV + q_clip + inline k_dyn/v_dyn + K/V 4-tap sconv
 *        + per-head vclip + strict f32->f16 KV store + hist/ring (320 WGs =
 *        5.00 WGs/CU on ALL 48 visits, T=1..8, <= 80 VGPR, -48 launches/step).
 *   5. ie_inkling_wo_sconv_post_t8_r11
 *        H18 lean T=2..16 W_o + SConv + Post fix (2 simultaneous rows/wave
 *        sharing xq, unrolled v8f accs, grid-flexible 72/128/144/192 WGs,
 *        optional TH_LOAD_HT flag for looped blk.6..17, <= 64 VGPR).
 *   6. ie_inkling_rel_attn_dec_glob_r11
 *        Fix #20 #9 (8 global layers = 1.335 ms = 96% of 233->16K T=1 slowdown):
 *        far-field zero-rb/zero-wrap fast loop + TH_LOAD_NT streaming global KV
 *        + optional fused OP_GATE(sigmoid) + Q8_0 epilogue (<= 68 VGPR, 4 WGs/CU).
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(kvt) == 2u, "ie_inkling_r11_loop_moe_fuse requires f16 KV cache");
_Static_assert(sizeof(f16) == 2u, "ink_r11_lm f16 must be 2 bytes");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r11_lm float/u32 must be 4 bytes");

#define INK_R11_LM_RSRC_W3       0x31004000u
#define INK_R11_LM_TH_HT         0 /* R43 scrub: standard default cached load (TH_LOAD_RT) */
#define INK_R11_LM_MAX_GRID      64u
#define INK_R11_LM_DEF_BAR_TICKS 25000000ull /* 250 ms @ 100 MHz s_sendmsg_rtnl(0x83) */
#define INK_R11_LM_DEF_KRN_TICKS 90000000ull /* 900 ms @ 100 MHz s_sendmsg_rtnl(0x83) */
#define INK_R11_LM_NB2_XLS       80u
#define INK_R11_LM_CT            8u
#define INK_R11_LM_HD            128u
#define INK_R11_LM_DREL          16u
#define INK_R11_LM_GQA           3u
#define INK_R11_LM_PSTR          130u
#define INK_R11_LM_AR_MAXM       32u

typedef unsigned long long ink_r11_lm_u64;
typedef unsigned int ink_r11_lm_u32x2 __attribute__((ext_vector_type(2)));
typedef ink_r11_lm_u32x2 ink_r11_lm_u32x2_a4 __attribute__((aligned(4)));

struct __attribute__((packed, aligned(4))) ink_r11_lm_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r11_lm_ld4(const G void *p) {
  return ((const G struct ink_r11_lm_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) __amdgpu_buffer_rsrc_t ink_r11_lm_rsrc(
    const G void *base_u, u32 nbytes) {
  G void *p = (G void *)rq_ptr_u((const G u8 *)base_u);
  return __builtin_amdgcn_make_buffer_rsrc(p, 0, nbytes, INK_R11_LM_RSRC_W3);
}

static inline __attribute__((always_inline)) u32x4 ink_r11_lm_ht_ld4(
    __amdgpu_buffer_rsrc_t r, u32 voff) {
  return __builtin_amdgcn_raw_buffer_load_b128(r, voff, 0, INK_R11_LM_TH_HT);
}

static inline __attribute__((always_inline)) float ink_r11_lm_ht_f32(
    __amdgpu_buffer_rsrc_t r, u32 voff) {
  const u32 bits = __builtin_amdgcn_raw_buffer_load_b32(r, voff, 0, INK_R11_LM_TH_HT);
  return I2F((int)bits);
}

static inline __attribute__((always_inline)) float ink_r11_lm_ht_f16f(
    __amdgpu_buffer_rsrc_t r, u32 voff) {
  const u16 bits = __builtin_amdgcn_raw_buffer_load_b16(r, voff, 0, INK_R11_LM_TH_HT);
  return (float)__builtin_bit_cast(f16, bits);
}

static inline __attribute__((always_inline)) u32x4 ink_r11_lm_rt_ld4(
    __amdgpu_buffer_rsrc_t r, u32 voff) {
  return __builtin_amdgcn_raw_buffer_load_b128(r, voff, 0, 0);
}

static inline __attribute__((always_inline)) float ink_r11_lm_rt_f32(
    __amdgpu_buffer_rsrc_t r, u32 voff) {
  const u32 bits = __builtin_amdgcn_raw_buffer_load_b32(r, voff, 0, 0);
  return I2F((int)bits);
}

static inline __attribute__((always_inline)) float ink_r11_lm_rt_f16f(
    __amdgpu_buffer_rsrc_t r, u32 voff) {
  const u16 bits = __builtin_amdgcn_raw_buffer_load_b16(r, voff, 0, 0);
  return (float)__builtin_bit_cast(f16, bits);
}

static inline __attribute__((always_inline)) kv4 ink_r11_lm_nt_kv4(const G kv4 *p) {
  const ink_r11_lm_u32x2 raw = *(const G ink_r11_lm_u32x2_a4 *)p;
  const u32 r0 = raw.x, r1 = raw.y;
  const u16 h0 = (u16)(r0 & 0xFFFFu), h1 = (u16)(r0 >> 16u);
  const u16 h2 = (u16)(r1 & 0xFFFFu), h3 = (u16)(r1 >> 16u);
  return (kv4){__builtin_bit_cast(f16, h0), __builtin_bit_cast(f16, h1),
               __builtin_bit_cast(f16, h2), __builtin_bit_cast(f16, h3)};
}

static inline __attribute__((always_inline)) float ink_r11_lm_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline __attribute__((always_inline)) int ink_r11_lm_dot8(
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

static inline __attribute__((always_inline)) float ink_r11_lm_loop_gate1(
    float a, float c, float f_raw, float g_raw) {
#pragma clang fp contract(off)
  const float f = ie_tanhf_s(f_raw);
  const float g = ie_tanhf_s(g_raw);
  return (1.0f - f) * a + g * c;
}

static inline __attribute__((always_inline)) void ink_r11_lm_quant_pack_lds(
    float v, u32 l, LDSP u8 *dst_q, LDSP float *dst_d) {
  const float amax = wave_max_pos(__builtin_fabsf(v));
  const float d = amax / 127.0f;
  const int qi = d != 0.0f ? (int)__builtin_roundf(v / d) : 0;
  const u32 b0 = (u32)(u8)(i8)qi;
  const u32 p2 = b0 | ((u32)shl_i((int)b0, 1) << 8u);
  const u32 p4 = p2 | ((u32)shl_i((int)p2, 2) << 16u);
  if ((l & 3u) == 0u) ((LDSP u32 *)dst_q)[l >> 2u] = p4;
  if (l == 0u) *dst_d = d;
}

static inline __attribute__((always_inline)) float ink_r11_lm_gate_f(float a, float g) {
#pragma clang fp contract(off)
  return a * (1.0f / (1.0f + __builtin_expf(-g)));
}

static inline __attribute__((always_inline)) float ink_r11_lm_halfwave_sum_all(float v) {
  v += I2F(shl_i(F2I(v), 8));
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return I2F(__builtin_amdgcn_readlane(F2I(v), 0));
}

static inline __attribute__((always_inline)) float ink_r11_lm_dot4(f32x4 a, f32x4 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

static inline __attribute__((always_inline)) float ink_r11_lm_wave_max(float v) {
  v = __builtin_fmaxf(v, shfl_xor_f(v, 16u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 8u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 4u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 2u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 1u));
  return v;
}

static inline __attribute__((always_inline)) float ink_r11_lm_wave_softmax(float logit, u32 l, u32 M) {
  const float v = (l < M) ? logit : -__builtin_inff();
  const float mx = ink_r11_lm_wave_max(v);
  const float e = (l < M) ? __builtin_expf(logit - mx) : 0.0f;
  const float ls = wave_sum_all(e);
  return (l < M) ? (e * (1.0f / ls)) : 0.0f;
}

static inline __attribute__((always_inline)) float ink_r11_lm_vclip_scale(
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

/* Strict double-rounded f32 -> f16 KV store barrier (0 v_fma_mix*_f16 drift). */
static inline __attribute__((always_inline)) kv4 ink_r11_lm_f32x4_to_kv4(f32x4 v) {
  float x0 = v.x, x1 = v.y, x2 = v.z, x3 = v.w;
  __asm__ volatile("" : "+v"(x0), "+v"(x1), "+v"(x2), "+v"(x3));
  return __builtin_convertvector((f32x4){x0, x1, x2, x3}, kv4);
}

static inline __attribute__((always_inline)) f32x4 ink_r11_lm_sconv4(
    f32x4 x0, f32x4 x1, f32x4 x2, f32x4 x3,
    f32x4 w0, f32x4 w1, f32x4 w2, f32x4 w3, float escale) {
#pragma clang fp contract(off)
  const f32x4 y = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3;
  return (x0 + y) * escale;
}

static inline __attribute__((always_inline)) void ink_r11_lm_load_sw(
    const G float *v, u32 i0, f32x4 *sw0, f32x4 *sw1, f32x4 *sw2, f32x4 *sw3) {
  const G f32x4 *v4 = (const G f32x4 *)v + i0;
  const f32x4 vw0 = v4[0], vw1 = v4[1], vw2 = v4[2], vw3 = v4[3];
  *sw0 = (f32x4){vw0.w, vw1.w, vw2.w, vw3.w};
  *sw1 = (f32x4){vw0.z, vw1.z, vw2.z, vw3.z};
  *sw2 = (f32x4){vw0.y, vw1.y, vw2.y, vw3.y};
  *sw3 = (f32x4){vw0.x, vw1.x, vw2.x, vw3.x};
}

static inline __attribute__((always_inline)) void ink_r11_lm_seed_ring(
    const G float *ring, u32 i0, u32 pos0, f32x4 *x1, f32x4 *x2, f32x4 *x3) {
  *x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (pos0 > 0u) {
    const G f32x4 *rg = (const G f32x4 *)ring + i0;
    const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
    const u32 s1 = (pos0 - 1u) & 3u;
    *x1 = (f32x4){ink_r11_lm_pick4(r0, s1), ink_r11_lm_pick4(r1, s1),
                  ink_r11_lm_pick4(r2, s1), ink_r11_lm_pick4(r3, s1)};
    if (pos0 >= 2u) {
      const u32 s2 = (pos0 - 2u) & 3u;
      *x2 = (f32x4){ink_r11_lm_pick4(r0, s2), ink_r11_lm_pick4(r1, s2),
                    ink_r11_lm_pick4(r2, s2), ink_r11_lm_pick4(r3, s2)};
    }
    if (pos0 >= 3u) {
      const u32 s3 = (pos0 - 3u) & 3u;
      *x3 = (f32x4){ink_r11_lm_pick4(r0, s3), ink_r11_lm_pick4(r1, s3),
                    ink_r11_lm_pick4(r2, s3), ink_r11_lm_pick4(r3, s3)};
    }
  }
}

static inline __attribute__((always_inline)) void ink_r11_lm_ring_commit(
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
      ring[(unsigned long)(i0 + 0u) * 4ul + sl] = x3.x;
      ring[(unsigned long)(i0 + 1u) * 4ul + sl] = x3.y;
      ring[(unsigned long)(i0 + 2u) * 4ul + sl] = x3.z;
      ring[(unsigned long)(i0 + 3u) * 4ul + sl] = x3.w;
    }
    if (T >= 2u) {
      const u32 sl = (pos0 + T - 2u) & 3u;
      ring[(unsigned long)(i0 + 0u) * 4ul + sl] = x2.x;
      ring[(unsigned long)(i0 + 1u) * 4ul + sl] = x2.y;
      ring[(unsigned long)(i0 + 2u) * 4ul + sl] = x2.z;
      ring[(unsigned long)(i0 + 3u) * 4ul + sl] = x2.w;
    }
    {
      const u32 sl = (pos0 + T - 1u) & 3u;
      ring[(unsigned long)(i0 + 0u) * 4ul + sl] = x1.x;
      ring[(unsigned long)(i0 + 1u) * 4ul + sl] = x1.y;
      ring[(unsigned long)(i0 + 2u) * 4ul + sl] = x1.z;
      ring[(unsigned long)(i0 + 3u) * 4ul + sl] = x1.w;
    }
  }
}

/* Wall-clock-guarded WDDM sense-reversing inter-WG barrier (sync = b->ink_sp_sync). */
static inline __attribute__((always_inline)) u32 ink_r11_lm_grid_barrier(
    G u32 *sync, u32 ng, u32 *sense_io, LDSP u32 *lds_poison,
    ink_r11_lm_u64 k_t0, ink_r11_lm_u64 bar_lim, ink_r11_lm_u64 krn_lim) {
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
        const ink_r11_lm_u64 t0 = (ink_r11_lm_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
        for (;;) {
          const u32 cur = __scoped_atomic_load_n(&sync[1u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
          if ((int)(cur - next_sense) >= 0) break;
          if (__scoped_atomic_load_n(&sync[2u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) != 0u) {
            poisoned = 1u;
            break;
          }
          const ink_r11_lm_u64 now = (ink_r11_lm_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
          if ((now - t0) > bar_lim || (now - k_t0) > krn_lim) {
            __scoped_atomic_fetch_or(&sync[2u], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
            poisoned = 1u;
            break;
          }
          __builtin_amdgcn_s_sleep(2);
        }
        const ink_r11_lm_u64 dt_k = ((ink_r11_lm_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83) - t0) / 1000ull;
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

/* ============================================================================
 * Kernel 1 (#20 #7 Cliff Fix): ie_inkling_lc_wfg_gate_norm_r11
 * Exact 54-arg drop-in replacement for ie_inkling_lc_wfg_gate_norm_r6 (T=1..16).
 *   - Drops VGPR 143 -> <= 64 (4 WGs/CU = 256 resident WGs > 216 QKV WGs!).
 *   - Coalesces T=2..16 inp/msg/tmp access via 2 KiB LDS staging -> 8 parallel
 *     waves x 32 contiguous lanes (eliminates 1-in-8 uncoalesced scalar writes).
 *   - Hoists nw once across all T tokens + double-buffered red[2][8] + next-tok
 *     prefetch in the RMSNorm+Q8 tail.
 *   - Grid: ((dim + 31u) / 32u, n_ctrl, 1u), NT = 256, 100% bitwise-exact!
 * ========================================================================== */

typedef union {
  struct {
    u8    x[16u * INK_R11_LM_NB2_XLS] __attribute__((aligned(16))); /* 1,280 B */
    float dx[2u][16u] __attribute__((aligned(16)));                /*   128 B */
    float fg[2u][8u][32u] __attribute__((aligned(16)));            /* 2,048 B */
  } wfg; /* 3,456 B */
  struct {
    float red[2u][8u];                                             /*    64 B */
    float sh[8u][128u] __attribute__((aligned(16)));               /* 4,096 B */
  } rn; /* 4,160 B */
} ink_r11_lm_wfg_lds_u;

static LDS ink_r11_lm_wfg_lds_u ink_r11_lm_wfg_lds __attribute__((aligned(16)));
static LDS u32 ink_r11_lm_wfg_last;

static inline __attribute__((always_inline)) void ink_r11_lm_rn_quant_chunk4(
    LDSP float (*sh)[128u], f32x4 y4, u32 wv, u32 l, u32 b0, G u8 *q, u32 nb) {
  *(LDSP f32x4 *)&sh[wv][l * 4u] = y4;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 m = 0u; m < 4u; m++) quant_wave(sh[wv][32u * m + l], b0 + m, q, nb);
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

static inline __attribute__((always_inline)) void ink_r11_lm_rmsnorm_1tok_gen(
    LDSP float (*red)[8u], LDSP float (*sh)[128u], u32 ping,
    const G float *x, const G float *nw, G float *ny, G u8 *nq,
    u32 n, float eps, u32 th, u32 wv, u32 l) {
  const u32 nb = n >> 5u;
  if (n <= NT) {
    const float xi = (th < n) ? x[th] : 0.0f;
    const float wi = (th < n) ? nw[th] : 0.0f;
    float ss = wave_tree_f(xi * xi);
    if (l == 0u) red[ping][wv] = ss;
    lbarrier();
    ss = 0.0f;
#pragma unroll
    for (u32 k = 0u; k < 8u; k++) ss += red[ping][k];
    const float s = 1.0f / __builtin_sqrtf(ss / (float)n + eps);
    if (th < n) {
      const float v = xi * s * wi;
      if (ny) ny[th] = v;
      if (nq) quant_wave(v, wv, nq, nb);
    }
    return;
  }

  float ss = 0.0f;
  u32 k = 0u;
  for (; (k + 1u) * 1024u <= n; k++) {
    const f32x4 vk = *(const G f32x4 *)(x + k * 1024u + th * 4u);
    ss += (vk.x * vk.x + vk.y * vk.y) + (vk.z * vk.z + vk.w * vk.w);
  }
  for (u32 i = k * 1024u + th; i < n; i += NT) {
    const float xi = x[i];
    ss += xi * xi;
  }
  ss = wave_tree_f(ss);
  if (l == 0u) red[ping][wv] = ss;
  lbarrier();
  ss = 0.0f;
#pragma unroll
  for (u32 ww = 0u; ww < 8u; ww++) ss += red[ping][ww];
  const float s = 1.0f / __builtin_sqrtf(ss / (float)n + eps);

  for (u32 ci = 0u; ci < k; ci++) {
    const u32 base = ci * 1024u + th * 4u;
    const f32x4 y4 = (*(const G f32x4 *)(x + base)) * s * (*(const G f32x4 *)(nw + base));
    if (ny) *(G f32x4 *)(ny + base) = y4;
    if (nq) ink_r11_lm_rn_quant_chunk4(sh, y4, wv, l, ci * 32u + wv * 4u, nq, nb);
  }
  for (u32 b = k * 32u + wv; b < nb; b += 8u) {
    const u32 i = b * 32u + l;
    const float v = x[i] * s * nw[i];
    if (ny) ny[i] = v;
    if (nq) quant_wave(v, b, nq, nb);
  }
}

/* Exact 0-bit-diff dim==2304 multi-token RMSNorm+Q8 with hoisted nw + ping-pong red + prefetch. */
static inline __attribute__((always_inline)) void ink_r11_lm_rmsnorm_2304_multi(
    LDSP float (*red)[8u], LDSP float (*sh)[128u],
    const G float *tmp, const G float *nw, G float *ny, G u8 *nq,
    u32 T, float eps, u32 ts, u32 ys, u32 qs_out, u32 th, u32 wv, u32 l) {
  const f32x4 w0 = *(const G f32x4 *)(nw + th * 4u);
  const f32x4 w1 = *(const G f32x4 *)(nw + 1024u + th * 4u);
  const float w2 = nw[2048u + th];

  const G float *xt_c = tmp;
  f32x4 x0 = *(const G f32x4 *)(xt_c + th * 4u);
  f32x4 x1 = *(const G f32x4 *)(xt_c + 1024u + th * 4u);
  float x2 = xt_c[2048u + th];

#pragma unroll 1
  for (u32 t = 0u; t < T; t++) {
    const u32 ping = t & 1u;
    float ss = (x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w);
    ss += (x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w);
    ss += x2 * x2;
    ss = wave_tree_f(ss);
    if (l == 0u) red[ping][wv] = ss;
    lbarrier();

    float sum = 0.0f;
#pragma unroll
    for (u32 ww = 0u; ww < 8u; ww++) sum += red[ping][ww];
    const float s = 1.0f / __builtin_sqrtf(sum / 2304.0f + eps);

    f32x4 x0_n = x0, x1_n = x1;
    float x2_n = x2;
    if (t + 1u < T) {
      const G float *xt_n = (const G float *)((const G u8 *)tmp + (unsigned long)(t + 1u) * (unsigned long)ts);
      x0_n = *(const G f32x4 *)(xt_n + th * 4u);
      x1_n = *(const G f32x4 *)(xt_n + 1024u + th * 4u);
      x2_n = xt_n[2048u + th];
    }

    const f32x4 y0 = x0 * s * w0;
    const f32x4 y1 = x1 * s * w1;
    const float y2 = x2 * s * w2;
    if (ny) {
      G float *nyt = (G float *)((G u8 *)ny + (unsigned long)t * (unsigned long)ys);
      *(G f32x4 *)(nyt + th * 4u) = y0;
      *(G f32x4 *)(nyt + 1024u + th * 4u) = y1;
      nyt[2048u + th] = y2;
    }
    if (nq) {
      G u8 *nqt = nq + (unsigned long)t * (unsigned long)qs_out;
      ink_r11_lm_rn_quant_chunk4(sh, y0, wv, l, wv * 4u, nqt, 72u);
      ink_r11_lm_rn_quant_chunk4(sh, y1, wv, l, 32u + wv * 4u, nqt, 72u);
      quant_wave(y2, 64u + wv, nqt, 72u);
    }
    x0 = x0_n; x1 = x1_n; x2 = x2_n;
  }
}

KERNEL ie_inkling_lc_wfg_gate_norm_r11(
    const G u32 *qw0, const G f16 *qs0, const G float *h0, const G float *inp0, const G float *msg0,
    G float *tmp0, const G float *nw0, G float *ny0, G u8 *nq0,
    const G u32 *qw1, const G f16 *qs1, const G float *h1, const G float *inp1, const G float *msg1,
    G float *tmp1, const G float *nw1, G float *ny1, G u8 *nq1,
    const G u32 *qw2, const G f16 *qs2, const G float *h2, const G float *inp2, const G float *msg2,
    G float *tmp2, const G float *nw2, G float *ny2, G u8 *nq2,
    const G u32 *qw3, const G f16 *qs3, const G float *h3, const G float *inp3, const G float *msg3,
    G float *tmp3, const G float *nw3, G float *ny3, G u8 *nq3,
    G u32 *count, u32 dim, u32 nb, u32 T, u32 n_ctrl, float eps,
    u32 hs, u32 is0, u32 is1, u32 is2, u32 is3,
    u32 ms0, u32 ms1, u32 ms2, u32 ms3,
    u32 ts, u32 ys, u32 qs_out) {
#pragma clang fp contract(off)
  _Static_assert(sizeof(ink_r11_lm_wfg_lds) == 4160u, "ink_r11_lm wfg/norm LDS union must be 4160 B");
  if (T == 0u || T > 16u || dim == 0u || (dim & 31u) != 0u || nb != 2u || n_ctrl == 0u || n_ctrl > 4u)
    return;

  const u32 bx = wgid();
  const u32 c = tokid();
  const u32 n_wgx = (dim + 31u) >> 5u;
  if (c >= n_ctrl || bx >= n_wgx) return;

  const G u32 *qw = qw0;
  const G f16 *qs = qs0;
  const G float *h = h0;
  const G float *inp = inp0;
  const G float *msg = msg0;
  G float *tmp = tmp0;
  const G float *nw = nw0;
  G float *ny = ny0;
  G u8 *nq = nq0;
  u32 is = is0;
  u32 ms = ms0;
  if (c == 1u) {
    qw = qw1; qs = qs1; h = h1; inp = inp1; msg = msg1;
    tmp = tmp1; nw = nw1; ny = ny1; nq = nq1; is = is1; ms = ms1;
  } else if (c == 2u) {
    qw = qw2; qs = qs2; h = h2; inp = inp2; msg = msg2;
    tmp = tmp2; nw = nw2; ny = ny2; nq = nq2; is = is2; ms = ms2;
  } else if (c == 3u) {
    qw = qw3; qs = qs3; h = h3; inp = inp3; msg = msg3;
    tmp = tmp3; nw = nw3; ny = ny3; nq = nq3; is = is3; ms = ms3;
  }
  if (!qw || !qs || !h || !inp || !msg || !tmp) return;

  const u32 th = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(th >> 5u));
  const u32 l = lane();
  const u32 grp = l >> 3u;
  const u32 is_gain = (l >> 2u) & 1u;
  const u32 sub = l & 3u;
  const u32 blk = sub >> 1u;
  const u32 i_ch = (bx << 5u) + (w << 2u) + grp;

  if (T == 1u) {
    if (w < 2u) {
      ink_r11_lm_quant_pack_lds(h[(w << 5u) + l], l,
                                ink_r11_lm_wfg_lds.wfg.x + (w << 5u),
                                &ink_r11_lm_wfg_lds.wfg.dx[w][0u]);
    }
    u32x4 wv = (u32x4)(0u);
    float sw = 0.0f, inp_val = 0.0f, msg_val = 0.0f;
    if (i_ch < dim) {
      const u32 row = is_gain ? (dim + i_ch) : i_ch;
      wv = ink_r11_lm_ld4((const G u8 *)qw + (unsigned long)row * 64ul + (unsigned long)(sub << 4u));
      sw = (float)qs[(unsigned long)row * 2ul + blk];
      if ((l & 7u) == 0u) {
        inp_val = inp[i_ch];
        msg_val = msg[i_ch];
      }
    }
    lbarrier();
    if (i_ch < dim) {
      const u32x4 av = *(const LDSP u32x4 *)(ink_r11_lm_wfg_lds.wfg.x + (sub << 4u));
      const u32 d_half =
          ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(0u, wv.x, av.x), wv.y, av.y), wv.z, av.z), wv.w, av.w);
      const int dot = (int)d_half + shl_i((int)d_half, 1);
      const float sc = sw * ink_r11_lm_wfg_lds.wfg.dx[blk][0u];
      const float term = sc * (float)dot;
      const float sum = term + I2F(shl_i(F2I(term), 2));
      const float gain_val = I2F(shl_i(F2I(sum), 4));
      if ((l & 7u) == 0u) tmp[i_ch] = ink_r11_lm_loop_gate1(inp_val, msg_val, sum, gain_val);
    }
  } else {
    /* Verify T = 2..16: quantize all T h-vectors into LDS, load wfg ONCE, then
     * chunk tokens in groups of 8 with coalesced 32-lane inp/msg/tmp I/O! */
    for (u32 tk = w; tk < T; tk += 8u) {
      const G float *ht = (const G float *)((const G u8 *)h + (unsigned long)tk * (unsigned long)hs);
      LDSP u8 *dst = ink_r11_lm_wfg_lds.wfg.x + tk * INK_R11_LM_NB2_XLS;
      ink_r11_lm_quant_pack_lds(ht[l], l, dst, &ink_r11_lm_wfg_lds.wfg.dx[0u][tk]);
      ink_r11_lm_quant_pack_lds(ht[32u + l], l, dst + 32u, &ink_r11_lm_wfg_lds.wfg.dx[1u][tk]);
    }

    u32x4 wv = (u32x4)(0u);
    float sw = 0.0f;
    if (i_ch < dim) {
      const u32 row = is_gain ? (dim + i_ch) : i_ch;
      wv = ink_r11_lm_ld4((const G u8 *)qw + (unsigned long)row * 64ul + (unsigned long)(sub << 4u));
      sw = (float)qs[(unsigned long)row * 2ul + blk];
    }
    lbarrier();

    const u32 ch_loc = (w << 2u) + grp;
    const u32 ch_coal = (bx << 5u) + l;
    for (u32 t0 = 0u; t0 < T; t0 += 8u) {
      const u32 tc = (T - t0 < 8u) ? (T - t0) : 8u;
      if (i_ch < dim) {
#pragma unroll 1
        for (u32 j = 0u; j < tc; j++) {
          const u32 t = t0 + j;
          const u32x4 av = *(const LDSP u32x4 *)(ink_r11_lm_wfg_lds.wfg.x + t * INK_R11_LM_NB2_XLS + (sub << 4u));
          const u32 d_half =
              ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(0u, wv.x, av.x), wv.y, av.y), wv.z, av.z), wv.w, av.w);
          const int dot = (int)d_half + shl_i((int)d_half, 1);
          const float sc = sw * ink_r11_lm_wfg_lds.wfg.dx[blk][t];
          const float term = sc * (float)dot;
          const float sum = term + I2F(shl_i(F2I(term), 2));
          const float gain_val = I2F(shl_i(F2I(sum), 4));
          if ((l & 7u) == 0u) {
            ink_r11_lm_wfg_lds.wfg.fg[0u][j][ch_loc] = sum;
            ink_r11_lm_wfg_lds.wfg.fg[1u][j][ch_loc] = gain_val;
          }
        }
      }
      lbarrier();

      /* All tc waves (<= 8) write 32 contiguous coalesced floats in parallel! */
      if (w < tc && ch_coal < dim) {
        const u32 t = t0 + w;
        const float f_raw = ink_r11_lm_wfg_lds.wfg.fg[0u][w][l];
        const float g_raw = ink_r11_lm_wfg_lds.wfg.fg[1u][w][l];
        const float a_val = ((const G float *)((const G u8 *)inp + (unsigned long)t * (unsigned long)is))[ch_coal];
        const float c_val = ((const G float *)((const G u8 *)msg + (unsigned long)t * (unsigned long)ms))[ch_coal];
        ((G float *)((G u8 *)tmp + (unsigned long)t * (unsigned long)ts))[ch_coal] =
            ink_r11_lm_loop_gate1(a_val, c_val, f_raw, g_raw);
      }
      if (t0 + 8u < T) lbarrier();
    }
  }

  if (nw && count && (ny || nq)) {
    G u32 *cnt_ptr = count + c;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (th == 0u) {
      ink_r11_lm_wfg_last =
          (__scoped_atomic_fetch_add(cnt_ptr, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == n_wgx - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r11_lm_wfg_last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

    if (dim == 2304u) {
      ink_r11_lm_rmsnorm_2304_multi(ink_r11_lm_wfg_lds.rn.red, ink_r11_lm_wfg_lds.rn.sh,
                                    tmp, nw, ny, nq, T, eps, ts, ys, qs_out, th, w, l);
    } else {
      for (u32 t = 0u; t < T; t++) {
        const G float *xt = (const G float *)((const G u8 *)tmp + (unsigned long)t * (unsigned long)ts);
        G float *nyt = ny ? (G float *)((G u8 *)ny + (unsigned long)t * (unsigned long)ys) : (G float *)0;
        G u8 *nqt = nq ? (nq + (unsigned long)t * (unsigned long)qs_out) : (G u8 *)0;
        ink_r11_lm_rmsnorm_1tok_gen(ink_r11_lm_wfg_lds.rn.red, ink_r11_lm_wfg_lds.rn.sh, t & 1u,
                                    xt, nw, nyt, nqt, dim, eps, th, w, l);
      }
    }
    if (th == 0u) __scoped_atomic_store_n(cnt_ptr, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  }
}

/* ============================================================================
 * Shared Helpers for H17 Persistent 1-Launch Loop Pipelines (Kernels 2 & 3)
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r11_lm_prenorm_2304(
    LDSP float *red, LDSP float (*sh)[128u],
    const G float *x, G u8 *d0, u32 q_blk_off, float eps, u32 th, u32 wv, u32 l) {
  const f32x4 x0 = *(const G f32x4 *)(x + th * 4u);
  const f32x4 x1 = *(const G f32x4 *)(x + 1024u + th * 4u);
  const float x2 = x[2048u + th];
  float ss = (x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w);
  ss += (x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w);
  ss += x2 * x2;
  ss = wave_tree_f(ss);
  if (l == 0u) red[wv] = ss;
  lbarrier();
  ss = 0.0f;
#pragma unroll
  for (u32 k = 0u; k < 8u; k++) ss += red[k];
  const float s = 1.0f / __builtin_sqrtf(ss / 2304.0f + eps);
  ink_r11_lm_rn_quant_chunk4(sh, x0 * s, wv, l, q_blk_off + wv * 4u, d0, 144u);
  ink_r11_lm_rn_quant_chunk4(sh, x1 * s, wv, l, q_blk_off + 32u + wv * 4u, d0, 144u);
  quant_wave(x2 * s, q_blk_off + 64u + wv, d0, 144u);
  lbarrier();
}

static inline __attribute__((always_inline)) void ink_r11_lm_down_1row(
    LDSP float (*dn)[8u][32u],
    const G u32 *qw, const G f16 *qs, const G u8 *xq_lo, const G u8 *xq, G float *y,
    u32 r, u32 nb, u32 T, u32 xs, u32 ys, u32 t0, u32 w, u32 l) {
#pragma clang fp contract(off)
  const u32 nb_half = nb >> 1u;
  const u32 nchunks = (nb + 255u) >> 8u;
  const G u32 *row_qw = qw + (unsigned long)r * (unsigned long)nb * 8ul;
  const G f16 *row_qs = qs + (unsigned long)r * (unsigned long)nb;

  if (T == 1u) {
    float acc = 0.0f;
    for (u32 ch = 0u; ch < nchunks; ch++) {
      const u32 base = ch << 8u;
      const u32 b = base + t0;
      float term = 0.0f;
      if (b < nb) {
        const G u32 *wp = row_qw + (unsigned long)b * 8ul;
        const u32x4 w0 = ink_r11_lm_ld4(wp);
        const u32x4 w1 = ink_r11_lm_ld4(wp + 4u);
        const float sw = (float)row_qs[b];
        const G u8 *xsrc = (b < nb_half) ? xq_lo : xq;
        const G u32 *aw = (const G u32 *)xsrc;
        const u32x4 a0 = ink_r11_lm_ld4(aw + (unsigned long)b * 8ul);
        const u32x4 a1 = ink_r11_lm_ld4(aw + (unsigned long)b * 8ul + 4ul);
        const float da = ((const G float *)(xsrc + (unsigned long)nb * 32ul))[b];
        term = (sw * da) * (float)ink_r11_lm_dot8(w0, w1, a0, a1);
      }
      dn[0u][w][l] = term;
      lbarrier();
      if (w == 0u) {
        const u32 rem = nb - base;
        const u32 nw_act = rem >= 256u ? 8u : ((rem + 31u) >> 5u);
        for (u32 k = 0u; k < nw_act; k++) {
          if (base + (k << 5u) + l < nb) acc = acc + dn[0u][k][l];
        }
      }
      lbarrier();
    }
    if (w == 0u) {
      acc = wave_tree_f(acc);
      if (l == 0u) y[r] = acc;
    }
    return;
  }

  float acc = 0.0f;
  for (u32 ch = 0u; ch < nchunks; ch++) {
    const u32 base = ch << 8u;
    const u32 b = base + t0;
    if (b < nb) {
      const G u32 *wp = row_qw + (unsigned long)b * 8ul;
      const u32x4 w0 = ink_r11_lm_ld4(wp);
      const u32x4 w1 = ink_r11_lm_ld4(wp + 4u);
      const float sw = (float)row_qs[b];
      const G u8 *xbase = (b < nb_half) ? xq_lo : xq;
      for (u32 j = 0u; j < T; j++) {
        const G u8 *xj = xbase + (unsigned long)j * (unsigned long)xs;
        const G u32 *aw = (const G u32 *)xj;
        const u32x4 a0 = ink_r11_lm_ld4(aw + (unsigned long)b * 8ul);
        const u32x4 a1 = ink_r11_lm_ld4(aw + (unsigned long)b * 8ul + 4ul);
        const float da = ((const G float *)(xj + (unsigned long)nb * 32ul))[b];
        dn[j][w][l] = (sw * da) * (float)ink_r11_lm_dot8(w0, w1, a0, a1);
      }
    }
    lbarrier();
    if (w < T) {
      const u32 rem = nb - base;
      const u32 nw_act = rem >= 256u ? 8u : ((rem + 31u) >> 5u);
      for (u32 k = 0u; k < nw_act; k++) {
        if (base + (k << 5u) + l < nb) acc = acc + dn[w][k][l];
      }
    }
    lbarrier();
  }
  if (w < T) {
    acc = wave_tree_f(acc);
    if (l == 0u) ((G float *)((G u8 *)y + (unsigned long)w * (unsigned long)ys))[r] = acc;
  }
}

static inline __attribute__((always_inline)) void ink_r11_lm_wfg_1blk(
    LDSP u8 *x_lds, LDSP float (*dx_lds)[8u], LDSP float (*fg_lds)[8u][32u],
    const G u32 *qw, const G f16 *qs, const G float *h, const G float *inp, const G float *msg,
    G float *tmp, u32 dim, u32 T, u32 hs, u32 is, u32 ms, u32 ts, u32 bx, u32 w, u32 l) {
#pragma clang fp contract(off)
  const u32 grp = l >> 3u;
  const u32 is_gain = (l >> 2u) & 1u;
  const u32 sub = l & 3u;
  const u32 blk = sub >> 1u;
  const u32 i_ch = (bx << 5u) + (w << 2u) + grp;

  if (T == 1u) {
    if (w < 2u) {
      ink_r11_lm_quant_pack_lds(h[(w << 5u) + l], l, x_lds + (w << 5u), &dx_lds[w][0u]);
    }
    u32x4 wv = (u32x4)(0u);
    float sw = 0.0f, inp_val = 0.0f, msg_val = 0.0f;
    if (i_ch < dim) {
      const u32 row = is_gain ? (dim + i_ch) : i_ch;
      wv = ink_r11_lm_ld4((const G u8 *)qw + (unsigned long)row * 64ul + (unsigned long)(sub << 4u));
      sw = (float)qs[(unsigned long)row * 2ul + blk];
      if ((l & 7u) == 0u) {
        inp_val = inp[i_ch];
        msg_val = msg[i_ch];
      }
    }
    lbarrier();
    if (i_ch < dim) {
      const u32x4 av = *(const LDSP u32x4 *)(x_lds + (sub << 4u));
      const u32 d_half =
          ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(0u, wv.x, av.x), wv.y, av.y), wv.z, av.z), wv.w, av.w);
      const int dot = (int)d_half + shl_i((int)d_half, 1);
      const float term = (sw * dx_lds[blk][0u]) * (float)dot;
      const float sum = term + I2F(shl_i(F2I(term), 2));
      const float gain_val = I2F(shl_i(F2I(sum), 4));
      if ((l & 7u) == 0u) tmp[i_ch] = ink_r11_lm_loop_gate1(inp_val, msg_val, sum, gain_val);
    }
    lbarrier();
    return;
  }

  if (w < T) {
    const G float *ht = (const G float *)((const G u8 *)h + (unsigned long)w * (unsigned long)hs);
    LDSP u8 *dst = x_lds + w * INK_R11_LM_NB2_XLS;
    ink_r11_lm_quant_pack_lds(ht[l], l, dst, &dx_lds[0u][w]);
    ink_r11_lm_quant_pack_lds(ht[32u + l], l, dst + 32u, &dx_lds[1u][w]);
  }
  u32x4 wv = (u32x4)(0u);
  float sw = 0.0f;
  if (i_ch < dim) {
    const u32 row = is_gain ? (dim + i_ch) : i_ch;
    wv = ink_r11_lm_ld4((const G u8 *)qw + (unsigned long)row * 64ul + (unsigned long)(sub << 4u));
    sw = (float)qs[(unsigned long)row * 2ul + blk];
  }
  lbarrier();

  const u32 ch_loc = (w << 2u) + grp;
  if (i_ch < dim) {
#pragma unroll 1
    for (u32 t = 0u; t < T; t++) {
      const u32x4 av = *(const LDSP u32x4 *)(x_lds + t * INK_R11_LM_NB2_XLS + (sub << 4u));
      const u32 d_half =
          ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(0u, wv.x, av.x), wv.y, av.y), wv.z, av.z), wv.w, av.w);
      const int dot = (int)d_half + shl_i((int)d_half, 1);
      const float term = (sw * dx_lds[blk][t]) * (float)dot;
      const float sum = term + I2F(shl_i(F2I(term), 2));
      const float gain_val = I2F(shl_i(F2I(sum), 4));
      if ((l & 7u) == 0u) {
        fg_lds[0u][t][ch_loc] = sum;
        fg_lds[1u][t][ch_loc] = gain_val;
      }
    }
  }
  lbarrier();

  const u32 ch_coal = (bx << 5u) + l;
  if (w < T && ch_coal < dim) {
    const float f_raw = fg_lds[0u][w][l];
    const float g_raw = fg_lds[1u][w][l];
    const float a_val = ((const G float *)((const G u8 *)inp + (unsigned long)w * (unsigned long)is))[ch_coal];
    const float c_val = ((const G float *)((const G u8 *)msg + (unsigned long)w * (unsigned long)ms))[ch_coal];
    ((G float *)((G u8 *)tmp + (unsigned long)w * (unsigned long)ts))[ch_coal] =
        ink_r11_lm_loop_gate1(a_val, c_val, f_raw, g_raw);
  }
  lbarrier();
}

/* ============================================================================
 * Kernel 2 (H17-Attn): ie_inkling_lc_pipe_attn_r11
 * 1-launch persistent 3-controller attention loop pipeline (24 ops -> 1 launch,
 * -48 launches/step on Inkling-2; T = 1..8, dim = 2304, nb_dn = 144, rank = 64).
 *   - Stage 0: LC_PRENORM_Q8_4X (shared_inp via flags bit 0 -> 4*T or 6*T WGs)
 *   - Stage 1: LC_DOWN_4X (3*64 = 192 rows across ng <= 64 WGs -> 3 rows/WG!)
 *   - Stage 2: LC_WFG_GATE (3*72 = 216 blocks across ng <= 64 WGs -> 3..4/WG!)
 *   - Stage 3: PARALLEL-T RMSNORM+Q8 (3*T <= 24 WGs run simultaneously!)
 *   - Preserves msg0 (msg_qm) read-only for downstream lc_mlp!
 *   - Flags: bit 0 = SHARED_INP (SWA), bit 2 = PROBE_ONLY (2 guarded barriers).
 *   - Grid: (ng, 1u, 1u) with ng <= 64 (e.g. 64), NT = 256, <= 80 VGPR, 8.2 KiB LDS.
 *   - Total args: 58 (<= 64 H13 packed kernarg limit!).
 * ========================================================================== */

typedef union {
  float dn[8u][8u][32u] __attribute__((aligned(16))); /* 8,192 B */
  struct {
    u8    x[8u * INK_R11_LM_NB2_XLS] __attribute__((aligned(16)));
    float dx[2u][8u] __attribute__((aligned(16)));
    float fg[2u][8u][32u] __attribute__((aligned(16)));
  } wfg; /* 2,752 B */
  struct {
    float red[8u];
    float sh[8u][128u] __attribute__((aligned(16)));
  } rn; /* 4,128 B */
} ink_r11_lm_pipe_lds_u;

static LDS ink_r11_lm_pipe_lds_u ink_r11_lm_pa_lds __attribute__((aligned(16)));
static LDS ink_r11_lm_u64 ink_r11_lm_pa_ptrs[36u] __attribute__((aligned(16)));
static LDS u32 ink_r11_lm_pa_u32s[12u] __attribute__((aligned(16)));
static LDS u32 ink_r11_lm_pa_poison;

KERNEL ie_inkling_lc_pipe_attn_r11(
    const G float *inp0, const G float *msg0, G u8 *cq0,
    const G u32 *qwd0, const G f16 *qsd0, G float *h0,
    const G u32 *qwf0, const G f16 *qsf0, G float *tmp0,
    const G float *nw0, G float *ny0, G u8 *nq0,
    const G float *inp1, const G float *msg1, G u8 *cq1,
    const G u32 *qwd1, const G f16 *qsd1, G float *h1,
    const G u32 *qwf1, const G f16 *qsf1, G float *tmp1,
    const G float *nw1, G float *ny1, G u8 *nq1,
    const G float *inp2, const G float *msg2, G u8 *cq2,
    const G u32 *qwd2, const G f16 *qsd2, G float *h2,
    const G u32 *qwf2, const G f16 *qsf2, G float *tmp2,
    const G float *nw2, G float *ny2, G u8 *nq2,
    G u32 *sync, u32 ng, u32 dim, u32 nb_dn, u32 rank, u32 T, u32 flags, float eps,
    u32 is0, u32 is1, u32 is2, u32 ms0, u32 ms1, u32 ms2,
    u32 cqs, u32 hs, u32 ts, u32 ys, u32 qso,
    ink_r11_lm_u64 bar_ticks, ink_r11_lm_u64 krn_ticks) {
  const u32 g = wgid();
  if (ng == 0u || ng > INK_R11_LM_MAX_GRID || g >= ng || !sync) return;
  if (__scoped_atomic_load_n(&sync[2u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) != 0u) return;

  const ink_r11_lm_u64 bar_lim = bar_ticks ? bar_ticks : INK_R11_LM_DEF_BAR_TICKS;
  const ink_r11_lm_u64 krn_lim = krn_ticks ? krn_ticks : INK_R11_LM_DEF_KRN_TICKS;
  const ink_r11_lm_u64 k_t0 = (ink_r11_lm_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
  u32 sense = __scoped_atomic_load_n(&sync[1u], __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);

  if (flags & 4u) {
    /* Co-residency probe launch: 2 guarded grid barriers and exit cleanly. */
    if (ink_r11_lm_grid_barrier(sync, ng, &sense, &ink_r11_lm_pa_poison, k_t0, bar_lim, krn_lim)) return;
    ink_r11_lm_grid_barrier(sync, ng, &sense, &ink_r11_lm_pa_poison, k_t0, bar_lim, krn_lim);
    return;
  }

  if (T == 0u || T > 8u || dim != 2304u || nb_dn != 144u || rank == 0u) return;

  const u32 th = tid();
  /* Stage 36 per-controller pointers + strides into 336 B LDS once so peak SGPRs stay <= 72 (0 spills!). */
  if (th == 0u) {
    ink_r11_lm_pa_ptrs[0u]  = (ink_r11_lm_u64)(unsigned long)inp0;
    ink_r11_lm_pa_ptrs[1u]  = (ink_r11_lm_u64)(unsigned long)inp1;
    ink_r11_lm_pa_ptrs[2u]  = (ink_r11_lm_u64)(unsigned long)inp2;
    ink_r11_lm_pa_ptrs[3u]  = (ink_r11_lm_u64)(unsigned long)msg0;
    ink_r11_lm_pa_ptrs[4u]  = (ink_r11_lm_u64)(unsigned long)msg1;
    ink_r11_lm_pa_ptrs[5u]  = (ink_r11_lm_u64)(unsigned long)msg2;
    ink_r11_lm_pa_ptrs[6u]  = (ink_r11_lm_u64)(unsigned long)cq0;
    ink_r11_lm_pa_ptrs[7u]  = (ink_r11_lm_u64)(unsigned long)cq1;
    ink_r11_lm_pa_ptrs[8u]  = (ink_r11_lm_u64)(unsigned long)cq2;
    ink_r11_lm_pa_ptrs[9u]  = (ink_r11_lm_u64)(unsigned long)qwd0;
    ink_r11_lm_pa_ptrs[10u] = (ink_r11_lm_u64)(unsigned long)qwd1;
    ink_r11_lm_pa_ptrs[11u] = (ink_r11_lm_u64)(unsigned long)qwd2;
    ink_r11_lm_pa_ptrs[12u] = (ink_r11_lm_u64)(unsigned long)qsd0;
    ink_r11_lm_pa_ptrs[13u] = (ink_r11_lm_u64)(unsigned long)qsd1;
    ink_r11_lm_pa_ptrs[14u] = (ink_r11_lm_u64)(unsigned long)qsd2;
    ink_r11_lm_pa_ptrs[15u] = (ink_r11_lm_u64)(unsigned long)h0;
    ink_r11_lm_pa_ptrs[16u] = (ink_r11_lm_u64)(unsigned long)h1;
    ink_r11_lm_pa_ptrs[17u] = (ink_r11_lm_u64)(unsigned long)h2;
    ink_r11_lm_pa_ptrs[18u] = (ink_r11_lm_u64)(unsigned long)qwf0;
    ink_r11_lm_pa_ptrs[19u] = (ink_r11_lm_u64)(unsigned long)qwf1;
    ink_r11_lm_pa_ptrs[20u] = (ink_r11_lm_u64)(unsigned long)qwf2;
    ink_r11_lm_pa_ptrs[21u] = (ink_r11_lm_u64)(unsigned long)qsf0;
    ink_r11_lm_pa_ptrs[22u] = (ink_r11_lm_u64)(unsigned long)qsf1;
    ink_r11_lm_pa_ptrs[23u] = (ink_r11_lm_u64)(unsigned long)qsf2;
    ink_r11_lm_pa_ptrs[24u] = (ink_r11_lm_u64)(unsigned long)tmp0;
    ink_r11_lm_pa_ptrs[25u] = (ink_r11_lm_u64)(unsigned long)tmp1;
    ink_r11_lm_pa_ptrs[26u] = (ink_r11_lm_u64)(unsigned long)tmp2;
    ink_r11_lm_pa_ptrs[27u] = (ink_r11_lm_u64)(unsigned long)nw0;
    ink_r11_lm_pa_ptrs[28u] = (ink_r11_lm_u64)(unsigned long)nw1;
    ink_r11_lm_pa_ptrs[29u] = (ink_r11_lm_u64)(unsigned long)nw2;
    ink_r11_lm_pa_ptrs[30u] = (ink_r11_lm_u64)(unsigned long)ny0;
    ink_r11_lm_pa_ptrs[31u] = (ink_r11_lm_u64)(unsigned long)ny1;
    ink_r11_lm_pa_ptrs[32u] = (ink_r11_lm_u64)(unsigned long)ny2;
    ink_r11_lm_pa_ptrs[33u] = (ink_r11_lm_u64)(unsigned long)nq0;
    ink_r11_lm_pa_ptrs[34u] = (ink_r11_lm_u64)(unsigned long)nq1;
    ink_r11_lm_pa_ptrs[35u] = (ink_r11_lm_u64)(unsigned long)nq2;
    ink_r11_lm_pa_u32s[0u]  = is0;
    ink_r11_lm_pa_u32s[1u]  = is1;
    ink_r11_lm_pa_u32s[2u]  = is2;
    ink_r11_lm_pa_u32s[3u]  = ms0;
    ink_r11_lm_pa_u32s[4u]  = ms1;
    ink_r11_lm_pa_u32s[5u]  = ms2;
    ink_r11_lm_pa_u32s[6u]  = cqs;
    ink_r11_lm_pa_u32s[7u]  = hs;
    ink_r11_lm_pa_u32s[8u]  = ts;
    ink_r11_lm_pa_u32s[9u]  = ys;
    ink_r11_lm_pa_u32s[10u] = qso;
    ink_r11_lm_pa_u32s[11u] = flags;
  }
  lbarrier();

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(th >> 5u));
  const u32 l = lane();
  const u32 shared = ink_r11_lm_pa_u32s[11u] & 1u;

  /* Stage 0: Prenorm + Q8 quant for all 3 controllers x T tokens. */
  {
    const u32 n_per_tok = shared ? 4u : 6u;
    const u32 tot0 = n_per_tok * T;
    const u32 cqs_u = ink_r11_lm_pa_u32s[6u];
    for (u32 task = g; task < tot0; task += ng) {
      const u32 tk = task / n_per_tok;
      const u32 sub = task - tk * n_per_tok;
      const G float *xp;
      G u8 *dp;
      u32 qoff;
      if (shared) {
        if (sub == 0u) {
          xp = (const G float *)((const G u8 *)(unsigned long)ink_r11_lm_pa_ptrs[0u] +
                                 (unsigned long)tk * (unsigned long)ink_r11_lm_pa_u32s[0u]);
          dp = (G u8 *)(unsigned long)ink_r11_lm_pa_ptrs[6u] + (unsigned long)tk * (unsigned long)cqs_u;
          qoff = 0u;
        } else {
          const u32 cc = sub - 1u;
          xp = (const G float *)((const G u8 *)(unsigned long)ink_r11_lm_pa_ptrs[3u + cc] +
                                 (unsigned long)tk * (unsigned long)ink_r11_lm_pa_u32s[3u + cc]);
          dp = (G u8 *)(unsigned long)ink_r11_lm_pa_ptrs[6u + cc] + (unsigned long)tk * (unsigned long)cqs_u;
          qoff = 72u;
        }
      } else {
        const u32 is_m = (sub >= 3u) ? 1u : 0u;
        const u32 cc = is_m ? (sub - 3u) : sub;
        const u32 p_base = is_m ? 3u : 0u;
        xp = (const G float *)((const G u8 *)(unsigned long)ink_r11_lm_pa_ptrs[p_base + cc] +
                               (unsigned long)tk * (unsigned long)ink_r11_lm_pa_u32s[p_base + cc]);
        dp = (G u8 *)(unsigned long)ink_r11_lm_pa_ptrs[6u + cc] + (unsigned long)tk * (unsigned long)cqs_u;
        qoff = is_m ? 72u : 0u;
      }
      ink_r11_lm_prenorm_2304(ink_r11_lm_pa_lds.rn.red, ink_r11_lm_pa_lds.rn.sh, xp, dp, qoff, eps, th, w, l);
    }
  }
  if (ink_r11_lm_grid_barrier(sync, ng, &sense, &ink_r11_lm_pa_poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 1: Down GEMV [rank x 4608] across 3 controllers (3 * 64 = 192 rows). */
  {
    const u32 tot1 = 3u * rank;
    const u32 cqs_u = ink_r11_lm_pa_u32s[6u];
    const u32 hs_u = ink_r11_lm_pa_u32s[7u];
    const G u8 *cq0_p = (const G u8 *)(unsigned long)ink_r11_lm_pa_ptrs[6u];
    for (u32 task = g; task < tot1; task += ng) {
      const u32 cc = task / rank;
      const u32 r = task - cc * rank;
      const G u32 *qw = (const G u32 *)(unsigned long)ink_r11_lm_pa_ptrs[9u + cc];
      const G f16 *qs = (const G f16 *)(unsigned long)ink_r11_lm_pa_ptrs[12u + cc];
      const G u8  *xq = (const G u8 *)(unsigned long)ink_r11_lm_pa_ptrs[6u + cc];
      G float     *yp = (G float *)(unsigned long)ink_r11_lm_pa_ptrs[15u + cc];
      const G u8  *xq_lo = shared ? cq0_p : xq;
      ink_r11_lm_down_1row(ink_r11_lm_pa_lds.dn, qw, qs, xq_lo, xq, yp, r, nb_dn, T, cqs_u, hs_u, th, w, l);
    }
  }
  if (ink_r11_lm_grid_barrier(sync, ng, &sense, &ink_r11_lm_pa_poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 2: WFG + Loop Gate across 3 controllers (3 * 72 = 216 blocks). */
  {
    const u32 n_wgx = 72u;
    const u32 tot2 = 3u * n_wgx;
    const u32 hs_u = ink_r11_lm_pa_u32s[7u];
    const u32 ts_u = ink_r11_lm_pa_u32s[8u];
    for (u32 task = g; task < tot2; task += ng) {
      const u32 cc = task / n_wgx;
      const u32 bx = task - cc * n_wgx;
      const G u32   *qw = (const G u32 *)(unsigned long)ink_r11_lm_pa_ptrs[18u + cc];
      const G f16   *qs = (const G f16 *)(unsigned long)ink_r11_lm_pa_ptrs[21u + cc];
      const G float *hp = (const G float *)(unsigned long)ink_r11_lm_pa_ptrs[15u + cc];
      const G float *ip = (const G float *)(unsigned long)ink_r11_lm_pa_ptrs[0u + cc];
      const G float *mp = (const G float *)(unsigned long)ink_r11_lm_pa_ptrs[3u + cc];
      G float       *tp = (G float *)(unsigned long)ink_r11_lm_pa_ptrs[24u + cc];
      const u32 is_u = ink_r11_lm_pa_u32s[0u + cc];
      const u32 ms_u = ink_r11_lm_pa_u32s[3u + cc];
      ink_r11_lm_wfg_1blk(ink_r11_lm_pa_lds.wfg.x, ink_r11_lm_pa_lds.wfg.dx, ink_r11_lm_pa_lds.wfg.fg,
                          qw, qs, hp, ip, mp, tp, dim, T, hs_u, is_u, ms_u, ts_u, bx, w, l);
    }
  }
  if (ink_r11_lm_grid_barrier(sync, ng, &sense, &ink_r11_lm_pa_poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 3: PARALLEL-T RMSNorm + Q8 across all 3 * T (<= 24) tokens! */
  {
    const u32 tot3 = 3u * T;
    const u32 ts_u = ink_r11_lm_pa_u32s[8u];
    const u32 ys_u = ink_r11_lm_pa_u32s[9u];
    const u32 qso_u = ink_r11_lm_pa_u32s[10u];
    for (u32 task = g; task < tot3; task += ng) {
      const u32 cc = task / T;
      const u32 tk = task - cc * T;
      const G float *tp  = (const G float *)(unsigned long)ink_r11_lm_pa_ptrs[24u + cc];
      const G float *nwp = (const G float *)(unsigned long)ink_r11_lm_pa_ptrs[27u + cc];
      G float       *nyp = (G float *)(unsigned long)ink_r11_lm_pa_ptrs[30u + cc];
      G u8          *nqp = (G u8 *)(unsigned long)ink_r11_lm_pa_ptrs[33u + cc];
      if (!nwp || (!nyp && !nqp)) continue;
      const G float *xt = (const G float *)((const G u8 *)tp + (unsigned long)tk * (unsigned long)ts_u);
      G float *nyt = nyp ? (G float *)((G u8 *)nyp + (unsigned long)tk * (unsigned long)ys_u) : (G float *)0;
      G u8 *nqt = nqp ? (nqp + (unsigned long)tk * (unsigned long)qso_u) : (G u8 *)0;
      const f32x4 x0 = *(const G f32x4 *)(xt + th * 4u);
      const f32x4 x1 = *(const G f32x4 *)(xt + 1024u + th * 4u);
      const float x2 = xt[2048u + th];
      const f32x4 w0 = *(const G f32x4 *)(nwp + th * 4u);
      const f32x4 w1 = *(const G f32x4 *)(nwp + 1024u + th * 4u);
      const float w2 = nwp[2048u + th];
      float ss = (x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w);
      ss += (x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w);
      ss += x2 * x2;
      ss = wave_tree_f(ss);
      if (l == 0u) ink_r11_lm_pa_lds.rn.red[w] = ss;
      lbarrier();
      ss = 0.0f;
#pragma unroll
      for (u32 k = 0u; k < 8u; k++) ss += ink_r11_lm_pa_lds.rn.red[k];
      const float s = 1.0f / __builtin_sqrtf(ss / 2304.0f + eps);
      const f32x4 y0 = x0 * s * w0;
      const f32x4 y1 = x1 * s * w1;
      const float y2 = x2 * s * w2;
      if (nyt) {
        *(G f32x4 *)(nyt + th * 4u) = y0;
        *(G f32x4 *)(nyt + 1024u + th * 4u) = y1;
        nyt[2048u + th] = y2;
      }
      if (nqt) {
        ink_r11_lm_rn_quant_chunk4(ink_r11_lm_pa_lds.rn.sh, y0, w, l, w * 4u, nqt, 72u);
        ink_r11_lm_rn_quant_chunk4(ink_r11_lm_pa_lds.rn.sh, y1, w, l, 32u + w * 4u, nqt, 72u);
        quant_wave(y2, 64u + w, nqt, 72u);
      }
      lbarrier();
    }
  }
}

/* ============================================================================
 * Kernel 3 (H17-MLP): ie_inkling_lc_pipe_mlp_r11
 * 1-launch persistent MLP loop controller pipeline for looped visits (T = 1..8,
 * dim = 2304, nb_dn = 144, rank = 64, grid <= 64, WDDM 0x83 poison guard).
 *   - Optional Stage 0 (when ar_q != NULL): OP_ATTN_RES(blocks_buf, ffn_inp -> fres)
 *     in parallel across T WGs (saves another -24 launches/step -> -72 total!).
 *   - Stage 1: LC_PRENORM_Q8(fres, msg_mlp -> cq) in parallel across 2*T WGs.
 *   - Stage 2: LC_DOWN(cq -> h) across 64 rows (1 row/WG at ng=64!).
 *   - Stage 3: LC_WFG_GATE(h, fres, msg_mlp -> minp) across 72 blocks.
 *   - Stage 4: PARALLEL-T RMSNORM+Q8(minp, ffn_norm -> ffn_in + ffn_in_q8).
 *   - Flags: bit 2 = PROBE_ONLY (2 guarded barriers).
 *   - Grid: (ng, 1u, 1u) with ng <= 64, NT = 256, <= 80 VGPR, 8.2 KiB LDS.
 * ========================================================================== */

typedef union {
  struct {
    float red[INK_R11_LM_AR_MAXM * 2u * 8u]; /* 2,048 B */
    float inv[INK_R11_LM_AR_MAXM];           /*   128 B */
    float prob[INK_R11_LM_AR_MAXM];          /*   128 B */
  } ar; /* 2,304 B */
  float dn[8u][8u][32u] __attribute__((aligned(16))); /* 8,192 B */
  struct {
    u8    x[8u * INK_R11_LM_NB2_XLS] __attribute__((aligned(16)));
    float dx[2u][8u] __attribute__((aligned(16)));
    float fg[2u][8u][32u] __attribute__((aligned(16)));
  } wfg; /* 2,752 B */
  struct {
    float red[8u];
    float sh[8u][128u] __attribute__((aligned(16)));
  } rn; /* 4,128 B */
} ink_r11_lm_pm_lds_u;

static LDS ink_r11_lm_pm_lds_u ink_r11_lm_pm_lds __attribute__((aligned(16)));
static LDS u32 ink_r11_lm_pm_poison;

KERNEL ie_inkling_lc_pipe_mlp_r11(
    const G float *blocks_buf, const G float *ffn_inp, const G float *ar_q,
    G float *fres, const G float *msg_mlp, G u8 *cq,
    const G u32 *qwd, const G f16 *qsd, G float *h,
    const G u32 *qwf, const G f16 *qsf, G float *minp,
    const G float *nw, G float *ffn_in, G u8 *ffn_in_q8,
    G u32 *sync, u32 ng, u32 M_ar, u32 dim, u32 nb_dn, u32 rank, u32 T, u32 flags, float eps,
    u32 bs_blk, u32 cs_inp, u32 fs, u32 ms, u32 cqs, u32 hs, u32 ts, u32 ys, u32 qso,
    ink_r11_lm_u64 bar_ticks, ink_r11_lm_u64 krn_ticks) {
  const u32 g = wgid();
  if (ng == 0u || ng > INK_R11_LM_MAX_GRID || g >= ng || !sync) return;
  if (__scoped_atomic_load_n(&sync[2u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) != 0u) return;

  const ink_r11_lm_u64 bar_lim = bar_ticks ? bar_ticks : INK_R11_LM_DEF_BAR_TICKS;
  const ink_r11_lm_u64 krn_lim = krn_ticks ? krn_ticks : INK_R11_LM_DEF_KRN_TICKS;
  const ink_r11_lm_u64 k_t0 = (ink_r11_lm_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
  u32 sense = __scoped_atomic_load_n(&sync[1u], __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);

  if (flags & 4u) {
    if (ink_r11_lm_grid_barrier(sync, ng, &sense, &ink_r11_lm_pm_poison, k_t0, bar_lim, krn_lim)) return;
    ink_r11_lm_grid_barrier(sync, ng, &sense, &ink_r11_lm_pm_poison, k_t0, bar_lim, krn_lim);
    return;
  }

  if (T == 0u || T > 8u || dim != 2304u || nb_dn != 144u || rank == 0u) return;

  const u32 th = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(th >> 5u));
  const u32 l = lane();

  /* Optional Stage 0: K=1 OP_ATTN_RES(blocks_buf, ffn_inp -> fres) across T WGs. */
  if (ar_q && M_ar > 0u && M_ar <= INK_R11_LM_AR_MAXM) {
    const float inv_sqn = 1.0f / __builtin_sqrtf(2304.0f);
    const u32 n_a = M_ar - 1u;
    const f32x4 q0 = *(const G f32x4 *)(ar_q + th * 4u);
    const f32x4 q1 = *(const G f32x4 *)(ar_q + 1024u + th * 4u);
    const float q2 = ar_q[2048u + th];
    for (u32 tk = g; tk < T; tk += ng) {
      const G float *at = blocks_buf ? (const G float *)((const G u8 *)blocks_buf + (unsigned long)tk * (unsigned long)bs_blk)
                                     : (const G float *)0;
      const G float *ct = (const G float *)((const G u8 *)ffn_inp + (unsigned long)tk * (unsigned long)cs_inp);
      G float *bt = (G float *)((G u8 *)fres + (unsigned long)tk * (unsigned long)fs);
      for (u32 s = 0u; s < M_ar; s++) {
        const G float *sp = (s < n_a) ? (at + (unsigned long)s * 2304ul) : ct;
        const f32x4 x0 = *(const G f32x4 *)(sp + th * 4u);
        const f32x4 x1 = *(const G f32x4 *)(sp + 1024u + th * 4u);
        const float x2 = sp[2048u + th];
        float sq = ((x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w)) +
                   ((x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w)) + x2 * x2;
        const f32x4 p0 = x0 * q0, p1 = x1 * q1;
        float dt = ((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w)) + x2 * q2;
        sq = wave_tree_f(sq);
        dt = wave_tree_f(dt);
        if (l == 0u) {
          ink_r11_lm_pm_lds.ar.red[s * 16u + w] = sq;
          ink_r11_lm_pm_lds.ar.red[s * 16u + 8u + w] = dt;
        }
      }
      lbarrier();
      if (w == 0u) {
        float lg = 0.0f, inv_s = 0.0f;
        if (l < M_ar) {
          float ssq = 0.0f, sdt = 0.0f;
#pragma unroll
          for (u32 x = 0u; x < 8u; x++) {
            ssq += ink_r11_lm_pm_lds.ar.red[l * 16u + x];
            sdt += ink_r11_lm_pm_lds.ar.red[l * 16u + 8u + x];
          }
          inv_s = 1.0f / __builtin_sqrtf(ssq / 2304.0f + eps);
          lg = sdt * inv_s * inv_sqn;
          ink_r11_lm_pm_lds.ar.inv[l] = inv_s;
        }
        const float pr = ink_r11_lm_wave_softmax(lg, l, M_ar);
        if (l < M_ar) ink_r11_lm_pm_lds.ar.prob[l] = pr;
      }
      lbarrier();
      f32x4 o0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      f32x4 o1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      float o2 = 0.0f;
      for (u32 s = 0u; s < M_ar; s++) {
        const G float *sp = (s < n_a) ? (at + (unsigned long)s * 2304ul) : ct;
        const float inv_s = ink_r11_lm_pm_lds.ar.inv[s];
        const float pr = ink_r11_lm_pm_lds.ar.prob[s];
        o0 += ((*(const G f32x4 *)(sp + th * 4u)) * inv_s) * pr;
        o1 += ((*(const G f32x4 *)(sp + 1024u + th * 4u)) * inv_s) * pr;
        o2 += (sp[2048u + th] * inv_s) * pr;
      }
      *(G f32x4 *)(bt + th * 4u) = o0;
      *(G f32x4 *)(bt + 1024u + th * 4u) = o1;
      bt[2048u + th] = o2;
      lbarrier();
    }
    if (ink_r11_lm_grid_barrier(sync, ng, &sense, &ink_r11_lm_pm_poison, k_t0, bar_lim, krn_lim)) return;
  }

  /* Stage 1: Prenorm + Q8 quant (fres -> cq[0..71], msg_mlp -> cq[72..143]). */
  {
    const u32 tot1 = 2u * T;
    for (u32 task = g; task < tot1; task += ng) {
      const u32 tk = task >> 1u;
      const u32 is_m = task & 1u;
      const G float *xp = is_m
                              ? (const G float *)((const G u8 *)msg_mlp + (unsigned long)tk * (unsigned long)ms)
                              : (const G float *)((const G u8 *)fres + (unsigned long)tk * (unsigned long)fs);
      G u8 *dp = cq + (unsigned long)tk * (unsigned long)cqs;
      ink_r11_lm_prenorm_2304(ink_r11_lm_pm_lds.rn.red, ink_r11_lm_pm_lds.rn.sh,
                              xp, dp, is_m ? 72u : 0u, eps, th, w, l);
    }
  }
  if (ink_r11_lm_grid_barrier(sync, ng, &sense, &ink_r11_lm_pm_poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 2: Down GEMV [rank x 4608] -> h (at ng=64, 1 row/WG!). */
  for (u32 r = g; r < rank; r += ng) {
    ink_r11_lm_down_1row(ink_r11_lm_pm_lds.dn, qwd, qsd, cq, cq, h, r, nb_dn, T, cqs, hs, th, w, l);
  }
  if (ink_r11_lm_grid_barrier(sync, ng, &sense, &ink_r11_lm_pm_poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 3: WFG + Loop Gate -> minp (72 blocks across ng <= 64 WGs). */
  for (u32 bx = g; bx < 72u; bx += ng) {
    ink_r11_lm_wfg_1blk(ink_r11_lm_pm_lds.wfg.x, ink_r11_lm_pm_lds.wfg.dx, ink_r11_lm_pm_lds.wfg.fg,
                        qwf, qsf, h, fres, msg_mlp, minp, dim, T, hs, fs, ms, ts, bx, w, l);
  }
  if (ink_r11_lm_grid_barrier(sync, ng, &sense, &ink_r11_lm_pm_poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 4: PARALLEL-T RMSNorm + Q8 -> ffn_in (non-NULL for F32 router!) + ffn_in_q8! */
  if (nw && (ffn_in || ffn_in_q8)) {
    const f32x4 w0 = *(const G f32x4 *)(nw + th * 4u);
    const f32x4 w1 = *(const G f32x4 *)(nw + 1024u + th * 4u);
    const float w2 = nw[2048u + th];
    for (u32 tk = g; tk < T; tk += ng) {
      const G float *xt = (const G float *)((const G u8 *)minp + (unsigned long)tk * (unsigned long)ts);
      G float *nyt = ffn_in ? (G float *)((G u8 *)ffn_in + (unsigned long)tk * (unsigned long)ys) : (G float *)0;
      G u8 *nqt = ffn_in_q8 ? (ffn_in_q8 + (unsigned long)tk * (unsigned long)qso) : (G u8 *)0;
      const f32x4 x0 = *(const G f32x4 *)(xt + th * 4u);
      const f32x4 x1 = *(const G f32x4 *)(xt + 1024u + th * 4u);
      const float x2 = xt[2048u + th];
      float ss = (x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w);
      ss += (x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w);
      ss += x2 * x2;
      ss = wave_tree_f(ss);
      if (l == 0u) ink_r11_lm_pm_lds.rn.red[w] = ss;
      lbarrier();
      ss = 0.0f;
#pragma unroll
      for (u32 k = 0u; k < 8u; k++) ss += ink_r11_lm_pm_lds.rn.red[k];
      const float s = 1.0f / __builtin_sqrtf(ss / 2304.0f + eps);
      const f32x4 y0 = x0 * s * w0;
      const f32x4 y1 = x1 * s * w1;
      const float y2 = x2 * s * w2;
      if (nyt) {
        *(G f32x4 *)(nyt + th * 4u) = y0;
        *(G f32x4 *)(nyt + 1024u + th * 4u) = y1;
        nyt[2048u + th] = y2;
      }
      if (nqt) {
        ink_r11_lm_rn_quant_chunk4(ink_r11_lm_pm_lds.rn.sh, y0, w, l, w * 4u, nqt, 72u);
        ink_r11_lm_rn_quant_chunk4(ink_r11_lm_pm_lds.rn.sh, y1, w, l, 32u + w * 4u, nqt, 72u);
        quant_wave(y2, 64u + w, nqt, 72u);
      }
      lbarrier();
    }
  }
}

/* ============================================================================
 * Kernel 4 (H20): ie_inkling_qkv_sconv_kv_r11
 * Single-launch QKV + q_clip + inline k_dyn/v_dyn + K/V 4-tap ShortConv +
 * per-KV-head VClip + strict double-rounded f32->f16 KV store + hist/ring!
 *   - Grid: (wg_q + wg_k + wg_v, 1u, 1u) = (192 + 64 + 64, 1, 1) = 320 WGs
 *           (EXACT 5.00 WGs/CU on ALL 48 visits: 40 SWA + 8 Global!).
 *   - 8 WGs per Q-head / K-head / V-head (16 rows/WG = 2 simultaneous rows/wave).
 *   - 8th arriving WG per Q-head runs exact q_clip via head_cnt[0..n_head-1].
 *   - 8th arriving WG per K-head / V-head runs optional 4-row k_dyn/v_dyn right
 *     into 128 B LDS (ZERO global k_dyn/v_dyn traffic!), then Wave 0 runs exact
 *     4-tap sconv + vclip + strict f32->f16 kc0/vc0 store + hist/ring via
 *     head_cnt[n_head..n_head+2*n_kv-1] (40 u32s = 160 B, zero-init, auto-reset).
 *   - T = 1..8, <= 80 VGPR (4 WGs/CU), 0 v_fma_mix*_f16 on KV stores, 0-bit-diff!
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r11_lm_q8_2rows_t8(
    const G u32 *qw, const G f16 *qs, const G u8 *xq,
    u32 r0, u32 r1, u32 nb, u32 T, u32 xs, u32 l,
    v8f *out0, v8f *out1) {
#pragma clang fp contract(off)
  const G u8 *rw0 = rq_ptr_u((const G u8 *)(qw + (unsigned long)r0 * (unsigned long)nb * 8ul));
  const G f16 *rs0 = (const G f16 *)rq_ptr_u((const G u8 *)(qs + (unsigned long)r0 * (unsigned long)nb));
  const G u8 *rw1 = rq_ptr_u((const G u8 *)(qw + (unsigned long)r1 * (unsigned long)nb * 8ul));
  const G f16 *rs1 = (const G f16 *)rq_ptr_u((const G u8 *)(qs + (unsigned long)r1 * (unsigned long)nb));
  const G u8 *xb = rq_ptr_u(xq);
  const unsigned long sc_off = 32ul * (unsigned long)nb;

  v8f a0 = (v8f)(0.0f);
  v8f a1 = (v8f)(0.0f);

#pragma clang loop unroll(disable)
  for (u32 b = l; b < nb; b += 32u) {
    const u32 bo = b << 5u;
    const u32x4 w0a = ink_r11_lm_ld4(rw0 + bo);
    const u32x4 w0b = ink_r11_lm_ld4(rw0 + bo + 16u);
    const float sw0 = (float)rs0[b];
    const u32x4 w1a = ink_r11_lm_ld4(rw1 + bo);
    const u32x4 w1b = ink_r11_lm_ld4(rw1 + bo + 16u);
    const float sw1 = (float)rs1[b];
    __builtin_amdgcn_sched_barrier(0);

#pragma unroll
    for (u32 j = 0u; j < INK_R11_LM_CT; j++) {
      if (j < T) {
        const G u8 *xj = xb + (unsigned long)j * (unsigned long)xs;
        const u32x4 x0 = ink_r11_lm_ld4(xj + bo);
        const u32x4 x1 = ink_r11_lm_ld4(xj + bo + 16u);
        const float da = ((const G float *)(xj + sc_off))[b];
        a0[j] = a0[j] + (sw0 * da) * (float)ink_r11_lm_dot8(w0a, w0b, x0, x1);
        a1[j] = a1[j] + (sw1 * da) * (float)ink_r11_lm_dot8(w1a, w1b, x0, x1);
        if ((j & 1u) == 1u) __builtin_amdgcn_sched_barrier(0);
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  v8f res0 = (v8f)(0.0f);
  v8f res1 = (v8f)(0.0f);
#pragma unroll
  for (u32 j = 0u; j < INK_R11_LM_CT; j++) {
    if (j < T) {
      res0[j] = wave_tree_f(a0[j]);
      res1[j] = wave_tree_f(a1[j]);
    }
  }
  *out0 = res0;
  *out1 = res1;
}

static inline __attribute__((always_inline)) void ink_r11_lm_q8_1row_t8(
    const G u32 *qw, const G f16 *qs, const G u8 *xq,
    u32 r, u32 nb, u32 T, u32 xs, u32 l, v8f *out0) {
#pragma clang fp contract(off)
  const G u8 *rw0 = rq_ptr_u((const G u8 *)(qw + (unsigned long)r * (unsigned long)nb * 8ul));
  const G f16 *rs0 = (const G f16 *)rq_ptr_u((const G u8 *)(qs + (unsigned long)r * (unsigned long)nb));
  const G u8 *xb = rq_ptr_u(xq);
  const unsigned long sc_off = 32ul * (unsigned long)nb;

  v8f a0 = (v8f)(0.0f);
#pragma clang loop unroll(disable)
  for (u32 b = l; b < nb; b += 32u) {
    const u32 bo = b << 5u;
    const u32x4 w0a = ink_r11_lm_ld4(rw0 + bo);
    const u32x4 w0b = ink_r11_lm_ld4(rw0 + bo + 16u);
    const float sw0 = (float)rs0[b];
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (u32 j = 0u; j < INK_R11_LM_CT; j++) {
      if (j < T) {
        const G u8 *xj = xb + (unsigned long)j * (unsigned long)xs;
        const u32x4 x0 = ink_r11_lm_ld4(xj + bo);
        const u32x4 x1 = ink_r11_lm_ld4(xj + bo + 16u);
        const float da = ((const G float *)(xj + sc_off))[b];
        a0[j] = a0[j] + (sw0 * da) * (float)ink_r11_lm_dot8(w0a, w0b, x0, x1);
      }
    }
  }
  v8f res0 = (v8f)(0.0f);
#pragma unroll
  for (u32 j = 0u; j < INK_R11_LM_CT; j++) {
    if (j < T) res0[j] = wave_tree_f(a0[j]);
  }
  *out0 = res0;
}

KERNEL ie_inkling_qkv_sconv_kv_r11(
    const G u32 *qw_q, const G f16 *qs_q, const G u8 *xq_q, G float *y_q_raw, G float *y_q_clip,
    const G u32 *qw_k, const G f16 *qs_k, const G u8 *xq_k, G float *y_k,
    const G u32 *qw_v, const G f16 *qs_v, const G u8 *xq_v, G float *y_v,
    const G u32 *qw_kd, const G f16 *qs_kd,
    const G u32 *qw_vd, const G f16 *qs_vd,
    const G float *v_qclip,
    const G float *vk, G float *ring_k, G float *hist_k,
    const G float *vv, G float *ring_v, G float *hist_v,
    G kvt *kc0, G kvt *vc0, G u32 *head_cnt,
    u32 n_head, u32 n_kv, u32 hd, u32 nb, u32 kv_len,
    float q_clamp, u32 norope, float q_escale, u32 win,
    float sc_escale, float k_clamp, float v_eps,
    u32 pos0, u32 T,
    u32 xs_q, u32 xs_k, u32 xs_v,
    u32 ys_qr, u32 ys_qc, u32 ys_k, u32 ys_v, u32 hrs) {
  static LDS u32   ink_r11_lm_qk_last;
  static LDS float ink_r11_lm_qk_dyn[8u][4u] __attribute__((aligned(16))); /* 128 B */

  if (T == 0u || T > 8u || hd != 128u || n_head == 0u || n_kv == 0u || nb == 0u || kv_len == 0u)
    return;

  const u32 qd = n_head * 128u;
  const u32 kvd = n_kv * 128u;
  const u32 wg_q = qd >> 4u;  /* 192 WGs (8 WGs / Q head) */
  const u32 wg_k = kvd >> 4u; /*  64 WGs (8 WGs / K head) */
  const u32 wg_v = kvd >> 4u; /*  64 WGs (8 WGs / V head) */
  const u32 g = wgid();
  if (g >= wg_q + wg_k + wg_v) return;

  u32 kind = 0u; /* 0 = Q, 1 = K, 2 = V */
  u32 g_sel = g, rows_sel = qd, xs_sel = xs_q, ys_sel = ys_qr;
  const G u32 *qw_sel = qw_q;
  const G f16 *qs_sel = qs_q;
  const G u8  *xq_sel = xq_q;
  G float     *y_sel  = y_q_raw;

  if (g >= wg_q + wg_k) {
    kind = 2u; g_sel = g - (wg_q + wg_k); rows_sel = kvd;
    qw_sel = qw_v; qs_sel = qs_v; xq_sel = xq_v; y_sel = y_v; xs_sel = xs_v; ys_sel = ys_v;
  } else if (g >= wg_q) {
    kind = 1u; g_sel = g - wg_q; rows_sel = kvd;
    qw_sel = qw_k; qs_sel = qs_k; xq_sel = xq_k; y_sel = y_k; xs_sel = xs_k; ys_sel = ys_k;
  }

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 r0 = (g_sel << 4u) + (w << 1u);
  const u32 r1 = (r0 + 1u < rows_sel) ? (r0 + 1u) : r0;

  {
    v8f v0, v1;
    ink_r11_lm_q8_2rows_t8(qw_sel, qs_sel, xq_sel, r0, r1, nb, T, xs_sel, l, &v0, &v1);
    if (l == 0u) {
#pragma unroll
      for (u32 j = 0u; j < INK_R11_LM_CT; j++) {
        if (j < T) {
          G float *yt = (G float *)((G u8 *)y_sel + (unsigned long)j * (unsigned long)ys_sel);
          yt[r0] = v0[j];
          if (r0 + 1u < rows_sel) yt[r0 + 1u] = v1[j];
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
    ink_r11_lm_qk_last =
        (__scoped_atomic_fetch_add(head_cnt + c_idx, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == 7u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r11_lm_qk_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  if (tid() == 0u) head_cnt[c_idx] = 0u;

  if (kind == 0u) {
    /* Q-head tail: exact vclip(q) matching ie_inkling_qkv_qclip_t8_r9. */
    const G f32x4 *vp = v_qclip ? ((const G f32x4 *)v_qclip + (unsigned long)h * 32ul) : (const G f32x4 *)0;
    const f32x4 vqv = vp ? vp[l] : (f32x4){1.0f, 1.0f, 1.0f, 1.0f};
    if (w < T) {
      const G f32x4 *ah = (const G f32x4 *)((const G u8 *)y_q_raw + (unsigned long)w * (unsigned long)ys_qr)
                          + (unsigned long)h * 32ul;
      G f32x4 *bh = (G f32x4 *)((G u8 *)y_q_clip + (unsigned long)w * (unsigned long)ys_qc)
                    + (unsigned long)h * 32ul;
      f32x4 xv = ah[l];
      if (q_clamp > 0.0f && vp) xv *= vqv;
      float ss = (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
      ss = wave_sum_all(ss);
      const float f = ink_r11_lm_vclip_scale(ss, 1.0f / 128.0f, q_clamp, norope, q_escale, win, pos0, w);
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
      ink_r11_lm_q8_1row_t8(qw_d, qs_d, is_k ? xq_k : xq_v, (h << 2u) + w, nb, T, is_k ? xs_k : xs_v, l, &dv);
      if (l == 0u) {
#pragma unroll
        for (u32 j = 0u; j < INK_R11_LM_CT; j++) {
          if (j < T) ink_r11_lm_qk_dyn[j][w] = dv[j];
        }
      }
    }
    lbarrier();
  }

  /* Wave 0 (32 lanes x f32x4 = 128 floats = 1 full KV head h):
   * exact 4-tap sconv + per-head vclip + strict f32->f16 KV store + hist/ring! */
  if (w == 0u) {
    const u32 i4 = (h << 5u) + l;
    const u32 i0 = i4 << 2u;
    const G float *vc_w = is_k ? vk : vv;
    G float *ring = is_k ? ring_k : ring_v;
    G float *hist = is_k ? hist_k : hist_v;
    G kvt *kv_dst = is_k ? kc0 : vc0;
    const G float *src_a = is_k ? y_k : y_v;
    const u32 as = is_k ? ys_k : ys_v;

    f32x4 sw0, sw1, sw2, sw3;
    ink_r11_lm_load_sw(vc_w, i0, &sw0, &sw1, &sw2, &sw3);
    f32x4 x1, x2, x3, x4 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    ink_r11_lm_seed_ring(ring, i0, pos0, &x1, &x2, &x3);

    u32 p = pos0 % kv_len;
    const u32 kvd4 = kvd >> 2u;
    for (u32 t = 0u; t < T; t++) {
      const f32x4 x0 = ((const G f32x4 *)((const G u8 *)src_a + (unsigned long)t * (unsigned long)as))[i4];
      if (hist) ((G f32x4 *)(hist + (unsigned long)t * (unsigned long)hrs))[i4] = x0;

      f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
      if (has_dyn) {
        w0 += ink_r11_lm_qk_dyn[t][0u];
        w1 += ink_r11_lm_qk_dyn[t][1u];
        w2 += ink_r11_lm_qk_dyn[t][2u];
        w3 += ink_r11_lm_qk_dyn[t][3u];
      }
      const f32x4 sx = ink_r11_lm_sconv4(x0, x1, x2, x3, w0, w1, w2, w3, sc_escale);
      float ss = (sx.x * sx.x + sx.y * sx.y) + (sx.z * sx.z + sx.w * sx.w);
      ss = wave_sum_all(ss);
      float f;
      if (is_k) {
        f = k_clamp / (__builtin_sqrtf(ss) + 1e-6f);
        if (f > 1.0f) f = 1.0f;
      } else {
        f = 1.0f / __builtin_sqrtf(ss * (1.0f / 128.0f) + v_eps);
      }
      const f32x4 xc = sx * f;
      ((G kv4 *)kv_dst)[(unsigned long)p * (unsigned long)kvd4 + (unsigned long)i4] = ink_r11_lm_f32x4_to_kv4(xc);

      x4 = x3; x3 = x2; x2 = x1; x1 = x0;
      p++;
      if (p == kv_len) p = 0u;
    }
    if (!hist) ink_r11_lm_ring_commit(ring, i0, pos0, T, x1, x2, x3, x4);
  }
}

/* ============================================================================
 * Kernel 5 (H18): ie_inkling_wo_sconv_post_t8_r11
 * Lean T=2..16 W_o + attn_pc ShortConv + Residual/BlockCopy [+ hist/ring].
 * Fixes R25 t8_r9's +0.20 ms regression:
 *   - 2 simultaneous rows/wave sharing xq loads (2x activation reuse!)
 *   - Unrolled v8f accumulators (0 inner `if (ju == ...)` branch ladder!)
 *   - Grid-flexible strided row-pair dispatch supporting gx in {72, 128, 144, 192}
 *     with 0 code changes so host can eliminate 144-WG tail-wave imbalance!
 *   - Optional flags bit 0 = TH_LOAD_HT on looped blk.6..17 W_o weights & xq!
 *   - Exact 19-arg signature of wo_sconv_post_t8_r9 + trailing `u32 flags`.
 *   - <= 64 VGPR (4 WGs/CU), 384 B LDS, 0/0 spills, 100% bitwise-exact!
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r11_lm_wo_step(
    float x0, float *x1, float *x2, float *x3,
    float w0, float w1, float w2, float w3,
    const G float *block_res, G float *out, G float *blk_dst, G float *hist_out, G float *ring,
    u32 r, u32 tk, u32 pos0, u32 T, float escale,
    u32 res_s, u32 out_s, u32 blk_s, u32 hist_row_stride) {
#pragma clang fp contract(off)
  const float cv = ((x0 * w0 + (*x1) * w1) + (*x2) * w2) + (*x3) * w3;
  const float sc = (x0 + cv) * escale;
  const float rv = ((const G float *)((const G u8 *)block_res + (unsigned long)tk * (unsigned long)res_s))[r];
  G float *op = (G float *)((G u8 *)out + (unsigned long)tk * (unsigned long)out_s);
  if (blk_dst) {
    G float *bp = (G float *)((G u8 *)blk_dst + (unsigned long)tk * (unsigned long)blk_s);
    bp[r] = rv;
    op[r] = sc;
  } else {
    op[r] = rv + sc;
  }
  if (hist_out) {
    hist_out[(unsigned long)tk * (unsigned long)hist_row_stride + (unsigned long)r] = x0;
  } else if (tk + 4u >= T) {
    ring[(unsigned long)r * 4ul + ((pos0 + tk) & 3u)] = x0;
  }
  *x3 = *x2; *x2 = *x1; *x1 = x0;
}

KERNEL ie_inkling_wo_sconv_post_t8_r11(
    const G u32 *qw, const G f16 *qs, const G u8 *xq,
    const G float *v_conv, G float *ring,
    const G float *block_res, G float *out, G float *blk_dst, G float *hist_out,
    u32 dim, u32 nb, float escale, u32 pos0, u32 T,
    u32 xs, u32 res_s, u32 out_s, u32 blk_s, u32 hist_row_stride, u32 flags) {
#pragma clang fp contract(off)
  static LDS float ink_r11_lm_wo_h[8u][2u][3u]; /* 192 B (2 rows/wave x 3 ring taps) */
  static LDS float ink_r11_lm_wo_v[8u][2u][8u]; /* 512 B (2 rows/wave x 8 token results -> <= 64 VGPR!) */
  if (T == 0u || T > 16u || dim == 0u || nb == 0u || !qw || !qs || !xq) return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 ng = (u32)__builtin_amdgcn_readfirstlane((int)ngroups_x());
  const u32 g_u = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  const u32 n_pairs = (dim + 1u) >> 1u;
  const u32 row_wbytes = nb << 5u;
  const u32 row_sbytes = nb << 1u;
  const u32 da_off0 = nb << 5u;
  const int use_ht = (flags & 1u) != 0u;
  const u32 x_tot = (T > 1u) ? (T * xs) : (nb * 40u);
  const __amdgpu_buffer_rsrc_t xr = ink_r11_lm_rsrc(xq, x_tot);

#pragma clang loop unroll(disable)
  for (u32 rp_raw = (g_u << 3u) + w; rp_raw < n_pairs; rp_raw += (ng << 3u)) {
    const u32 rp = (u32)__builtin_amdgcn_readfirstlane((int)rp_raw);
    const u32 r0 = rp << 1u;
    const u32 r1 = (r0 + 1u < dim) ? (r0 + 1u) : r0;
    const u32 r_rem = (r0 + 1u < dim) ? 2u : 1u;
    const u32 w1_off = (r0 + 1u < dim) ? row_wbytes : 0u;
    const u32 s1_off = (r0 + 1u < dim) ? row_sbytes : 0u;

    const __amdgpu_buffer_rsrc_t wr = ink_r11_lm_rsrc(
        (const G u8 *)qw + (unsigned long)r0 * (unsigned long)row_wbytes,
        r_rem * row_wbytes);
    const __amdgpu_buffer_rsrc_t sr = ink_r11_lm_rsrc(
        (const G u8 *)qs + (unsigned long)r0 * (unsigned long)row_sbytes,
        r_rem * row_sbytes);

#pragma clang loop unroll(disable)
    for (u32 t0 = 0u; t0 < T; t0 += INK_R11_LM_CT) {
      const u32 tc = (T - t0 < INK_R11_LM_CT) ? (T - t0) : INK_R11_LM_CT;
      {
        v8f acc0 = (v8f)(0.0f);
        v8f acc1 = (v8f)(0.0f);

        if (use_ht) {
#pragma clang loop unroll(disable)
          for (u32 b = l; b < nb; b += 32u) {
            const u32 bo = b << 5u;
            const u32x4 w0a = ink_r11_lm_ht_ld4(wr, bo);
            const u32x4 w0b = ink_r11_lm_ht_ld4(wr, bo + 16u);
            const float sw0 = ink_r11_lm_ht_f16f(sr, b << 1u);
            const u32x4 w1a = ink_r11_lm_ht_ld4(wr, w1_off + bo);
            const u32x4 w1b = ink_r11_lm_ht_ld4(wr, w1_off + bo + 16u);
            const float sw1 = ink_r11_lm_ht_f16f(sr, s1_off + (b << 1u));
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (u32 j = 0u; j < INK_R11_LM_CT; j++) {
              if (j < tc) {
                const u32 joff = (t0 + j) * xs;
                const u32x4 x0 = ink_r11_lm_ht_ld4(xr, joff + bo);
                const u32x4 x1 = ink_r11_lm_ht_ld4(xr, joff + bo + 16u);
                const float da = ink_r11_lm_ht_f32(xr, joff + da_off0 + (b << 2u));
                acc0[j] = acc0[j] + (sw0 * da) * (float)ink_r11_lm_dot8(w0a, w0b, x0, x1);
                acc1[j] = acc1[j] + (sw1 * da) * (float)ink_r11_lm_dot8(w1a, w1b, x0, x1);
                __builtin_amdgcn_sched_barrier(0);
              }
            }
          }
        } else {
#pragma clang loop unroll(disable)
          for (u32 b = l; b < nb; b += 32u) {
            const u32 bo = b << 5u;
            const u32x4 w0a = ink_r11_lm_rt_ld4(wr, bo);
            const u32x4 w0b = ink_r11_lm_rt_ld4(wr, bo + 16u);
            const float sw0 = ink_r11_lm_rt_f16f(sr, b << 1u);
            const u32x4 w1a = ink_r11_lm_rt_ld4(wr, w1_off + bo);
            const u32x4 w1b = ink_r11_lm_rt_ld4(wr, w1_off + bo + 16u);
            const float sw1 = ink_r11_lm_rt_f16f(sr, s1_off + (b << 1u));
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (u32 j = 0u; j < INK_R11_LM_CT; j++) {
              if (j < tc) {
                const u32 joff = (t0 + j) * xs;
                const u32x4 x0 = ink_r11_lm_rt_ld4(xr, joff + bo);
                const u32x4 x1 = ink_r11_lm_rt_ld4(xr, joff + bo + 16u);
                const float da = ink_r11_lm_rt_f32(xr, joff + da_off0 + (b << 2u));
                acc0[j] = acc0[j] + (sw0 * da) * (float)ink_r11_lm_dot8(w0a, w0b, x0, x1);
                acc1[j] = acc1[j] + (sw1 * da) * (float)ink_r11_lm_dot8(w1a, w1b, x0, x1);
                __builtin_amdgcn_sched_barrier(0);
              }
            }
          }
        }

#pragma unroll
        for (u32 j = 0u; j < INK_R11_LM_CT; j++) {
          if (j < tc) {
            const float s0 = wave_tree_f(acc0[j]);
            const float s1 = wave_tree_f(acc1[j]);
            if (l == 0u) {
              ink_r11_lm_wo_v[w][0u][j] = s0;
              ink_r11_lm_wo_v[w][1u][j] = s1;
            }
          }
        }
      }
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");

      /* Lane 0 finishes sconv+post for row r0; lane 1 finishes row r1 IN PARALLEL! */
      if (l < r_rem) {
        const u32 rr = (l == 0u) ? r0 : r1;
        const f32x4 vw = ((const G f32x4 *)v_conv)[rr];
        const float w0s = vw.w, w1s = vw.z, w2s = vw.y, w3s = vw.x;
        float x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
        if (t0 == 0u) {
          if (pos0 > 0u) {
            const f32x4 rg = ((const G f32x4 *)ring)[rr];
            x1 = ink_r11_lm_pick4(rg, (pos0 - 1u) & 3u);
            if (pos0 >= 2u) x2 = ink_r11_lm_pick4(rg, (pos0 - 2u) & 3u);
            if (pos0 >= 3u) x3 = ink_r11_lm_pick4(rg, (pos0 - 3u) & 3u);
          }
        } else {
          x1 = ink_r11_lm_wo_h[w][l][0u];
          x2 = ink_r11_lm_wo_h[w][l][1u];
          x3 = ink_r11_lm_wo_h[w][l][2u];
        }
#pragma clang loop unroll(disable)
        for (u32 j = 0u; j < tc; j++) {
          ink_r11_lm_wo_step(ink_r11_lm_wo_v[w][l][j], &x1, &x2, &x3, w0s, w1s, w2s, w3s,
                             block_res, out, blk_dst, hist_out, ring,
                             rr, t0 + j, pos0, T, escale, res_s, out_s, blk_s, hist_row_stride);
        }
        if (t0 + INK_R11_LM_CT < T) {
          ink_r11_lm_wo_h[w][l][0u] = x1;
          ink_r11_lm_wo_h[w][l][1u] = x2;
          ink_r11_lm_wo_h[w][l][2u] = x3;
        }
      }
    }
  }
}

/* ============================================================================
 * Kernel 6 (#20 #9 16K T=1 Global Attention): ie_inkling_rel_attn_dec_glob_r11
 * Specialized T=1 Global Attention targeting the 8 global layers (1.335 ms =
 * 96% of the 233->16K T=1 slowdown!):
 *   - Far-field split specialization (p1 <= pos+1 && pos+1-p1 >= extent_lim,
 *     >93% of splits at 16K): ZERO rel_bias loads, ZERO R reductions, ZERO
 *     modulo/wraparound when p1 <= kv_len (linear pointer advance!).
 *   - Streaming `__builtin_nontemporal_load` (`th:TH_LOAD_NT`) on global K/V
 *     so 512 MiB/step of single-read global KV never evicts hot looped weights
 *     or SWA KV from the 64 MiB Infinity Cache (MALL)!
 *   - Optional fused OP_GATE(out * sigmoid(agate)) + Q8_0 epilogue (out==NULL
 *     elides att; agate==NULL && oq==NULL acts as exact standalone drop-in!).
 *   - Grid: (n_kv * nsplit, 1u, 1u), <= 68 VGPR (4 WGs/CU), 12,484 B LDS, 0 spills.
 * ========================================================================== */

KERNEL ie_inkling_rel_attn_dec_glob_r11(
    const G float *q, const G kvt *kc, const G kvt *vc, const G float *R,
    const G float *rel_bias, const G float *sink, G float *part, G float *out,
    u32 pos, u32 kv_len, u32 hd, u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent,
    u32 win, u32 nsplit, u32 ch, G u32 *count,
    const G float *agate, G u8 *oq) {
  static LDS float ink_r11_lm_dg_m[INK_R11_LM_GQA][8];
  static LDS float ink_r11_lm_dg_l[INK_R11_LM_GQA][8];
  static LDS __attribute__((aligned(16))) float ink_r11_lm_dg_o[INK_R11_LM_GQA][8][INK_R11_LM_HD];
  static LDS u32 ink_r11_lm_dg_last;

  if (hd != INK_R11_LM_HD || d_rel != INK_R11_LM_DREL || n_head != INK_R11_LM_GQA * n_kv) return;

  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() / nsplit));
  const u32 s = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() % nsplit));
  const u32 h0 = kh * INK_R11_LM_GQA;
  const u32 kvd = n_kv * INK_R11_LM_HD;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const float scale = 1.0f / __builtin_sqrtf((float)INK_R11_LM_HD);

  const f32x4 q0 = ((const G f32x4 *)(q + (unsigned long)(h0 + 0u) * INK_R11_LM_HD))[l];
  const f32x4 q1 = ((const G f32x4 *)(q + (unsigned long)(h0 + 1u) * INK_R11_LM_HD))[l];
  const f32x4 q2 = ((const G f32x4 *)(q + (unsigned long)(h0 + 2u) * INK_R11_LM_HD))[l];

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

  const G kvt *kc_h = kc + (unsigned long)kh * INK_R11_LM_HD;
  const G kvt *vc_h = vc + (unsigned long)kh * INK_R11_LM_HD;
  const int far_split = (p1 <= pend) && ((pend - p1) >= extent_lim);
  const int no_wrap = (p1 <= kv_len);

  u32 kp = p0 + w;
  if (kp < p1) {
    if (far_split && no_wrap) {
      /* Far-field, non-wrapping global fast path (>93% of 16K splits):
       * TH_LOAD_NT streaming KV, zero rel_bias, zero wrap modulo, 1-step lookahead! */
      const unsigned long row_step = 8ul * (unsigned long)kvd;
      const G kvt *kr = kc_h + (unsigned long)kp * kvd;
      const G kvt *vr = vc_h + (unsigned long)kp * kvd;
      kv4 kh4 = ink_r11_lm_nt_kv4((const G kv4 *)kr + l);
      kv4 vh4 = ink_r11_lm_nt_kv4((const G kv4 *)vr + l);

#pragma unroll 1
      for (;;) {
        const u32 kp_nxt = kp + 8u;
        kv4 kh4_nxt = kh4, vh4_nxt = vh4;
        if (kp_nxt < p1) {
          kr += row_step;
          vr += row_step;
          kh4_nxt = ink_r11_lm_nt_kv4((const G kv4 *)kr + l);
          vh4_nxt = ink_r11_lm_nt_kv4((const G kv4 *)vr + l);
        }

        const f32x4 kf = __builtin_convertvector(kh4, f32x4);
        const float d0 = wave_sum_all(ink_r11_lm_dot4(q0, kf)) * scale;
        const float d1 = wave_sum_all(ink_r11_lm_dot4(q1, kf)) * scale;
        const float d2 = wave_sum_all(ink_r11_lm_dot4(q2, kf)) * scale;

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

        const f32x4 vf = __builtin_convertvector(vh4, f32x4);
        acc0 = acc0 * c0 + e0 * vf;
        acc1 = acc1 * c1 + e1 * vf;
        acc2 = acc2 * c2 + e2 * vf;

        if (kp_nxt >= p1) break;
        kp = kp_nxt;
        kh4 = kh4_nxt;
        vh4 = vh4_nxt;
      }
    } else {
      /* Near-field / wrapping path: exact rel_bias + TH_LOAD_NT streaming KV. */
      const float rv0 = (l < INK_R11_LM_DREL) ? R[(unsigned long)(h0 + 0u) * INK_R11_LM_DREL + l] : 0.0f;
      const float rv1 = (l < INK_R11_LM_DREL) ? R[(unsigned long)(h0 + 1u) * INK_R11_LM_DREL + l] : 0.0f;
      const float rv2 = (l < INK_R11_LM_DREL) ? R[(unsigned long)(h0 + 2u) * INK_R11_LM_DREL + l] : 0.0f;
      u32 kvp = kp % kv_len;
      kv4 kh4 = ink_r11_lm_nt_kv4((const G kv4 *)(kc_h + (unsigned long)kvp * kvd) + l);
      kv4 vh4 = ink_r11_lm_nt_kv4((const G kv4 *)(vc_h + (unsigned long)kvp * kvd) + l);

#pragma unroll 1
      for (;;) {
        const u32 dist = pos - kp;
        const int has_rel = (dist < extent_lim);
        float rb = 0.0f;
        if (has_rel && l < INK_R11_LM_DREL) rb = rel_bias[(unsigned long)dist * INK_R11_LM_DREL + l];

        const u32 kp_nxt = kp + 8u;
        u32 kvp_nxt = kvp;
        kv4 kh4_nxt = kh4, vh4_nxt = vh4;
        if (kp_nxt < p1) {
          kvp_nxt = kvp + 8u;
          if (kvp_nxt >= kv_len) kvp_nxt = (kv_len >= 8u) ? (kvp_nxt - kv_len) : (kp_nxt % kv_len);
          kh4_nxt = ink_r11_lm_nt_kv4((const G kv4 *)(kc_h + (unsigned long)kvp_nxt * kvd) + l);
          vh4_nxt = ink_r11_lm_nt_kv4((const G kv4 *)(vc_h + (unsigned long)kvp_nxt * kvd) + l);
        }

        const f32x4 kf = __builtin_convertvector(kh4, f32x4);
        float d0 = wave_sum_all(ink_r11_lm_dot4(q0, kf)) * scale;
        float d1 = wave_sum_all(ink_r11_lm_dot4(q1, kf)) * scale;
        float d2 = wave_sum_all(ink_r11_lm_dot4(q2, kf)) * scale;
        if (has_rel) {
          d0 += ink_r11_lm_halfwave_sum_all(rv0 * rb);
          d1 += ink_r11_lm_halfwave_sum_all(rv1 * rb);
          d2 += ink_r11_lm_halfwave_sum_all(rv2 * rb);
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

        const f32x4 vf = __builtin_convertvector(vh4, f32x4);
        acc0 = acc0 * c0 + e0 * vf;
        acc1 = acc1 * c1 + e1 * vf;
        acc2 = acc2 * c2 + e2 * vf;

        if (kp_nxt >= p1) break;
        kp = kp_nxt; kvp = kvp_nxt; kh4 = kh4_nxt; vh4 = vh4_nxt;
      }
    }
  }

  if (l == 0u) {
    ink_r11_lm_dg_m[0][w] = m0; ink_r11_lm_dg_l[0][w] = sum0;
    ink_r11_lm_dg_m[1][w] = m1; ink_r11_lm_dg_l[1][w] = sum1;
    ink_r11_lm_dg_m[2][w] = m2; ink_r11_lm_dg_l[2][w] = sum2;
  }
  ((LDSP f32x4 *)&ink_r11_lm_dg_o[0][w][0])[l] = acc0;
  ((LDSP f32x4 *)&ink_r11_lm_dg_o[1][w][0])[l] = acc1;
  ((LDSP f32x4 *)&ink_r11_lm_dg_o[2][w][0])[l] = acc2;
  lbarrier();

  const u32 nb_q = n_head << 2u;
  if (nsplit == 1u) {
    for (u32 idx = tid(); idx < INK_R11_LM_GQA * INK_R11_LM_HD; idx += NT) {
      const u32 g = idx >> 7u;
      const u32 e = idx & 127u;
      const u32 h = h0 + g;
      const float sv = sink[h];
      float M = sv;
      for (u32 x = 0u; x < 8u; x++)
        if (ink_r11_lm_dg_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r11_lm_dg_m[g][x]);
      float L = __builtin_expf(sv - M);
      float o = 0.0f;
      for (u32 x = 0u; x < 8u; x++) {
        const float lx = ink_r11_lm_dg_l[g][x];
        if (lx > 0.0f) {
          const float wx = __builtin_expf(ink_r11_lm_dg_m[g][x] - M);
          L += lx * wx;
          o += ink_r11_lm_dg_o[g][x][e] * wx;
        }
      }
      const u32 out_idx = h * INK_R11_LM_HD + e;
      float v = o / L;
      if (agate) v = ink_r11_lm_gate_f(v, agate[out_idx]);
      if (out) out[out_idx] = v;
      if (oq) quant_wave(v, out_idx >> 5u, oq, nb_q);
    }
    return;
  }

  for (u32 idx = tid(); idx < INK_R11_LM_GQA * INK_R11_LM_HD; idx += NT) {
    const u32 g = idx >> 7u;
    const u32 e = idx & 127u;
    const u32 h = h0 + g;
    float M = -__builtin_inff();
    for (u32 x = 0u; x < 8u; x++)
      if (ink_r11_lm_dg_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r11_lm_dg_m[g][x]);
    G float *pp = part + (unsigned long)(h * nsplit + s) * INK_R11_LM_PSTR;
    float o = 0.0f, L = 0.0f;
    for (u32 x = 0u; x < 8u; x++) {
      const float lx = ink_r11_lm_dg_l[g][x];
      if (lx > 0.0f) {
        const float wx = __builtin_expf(ink_r11_lm_dg_m[g][x] - M);
        o += ink_r11_lm_dg_o[g][x][e] * wx;
        if (e == 0u) L += lx * wx;
      }
    }
    pp[2u + e] = o;
    if (e == 0u) { pp[0] = M; pp[1] = L; }
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r11_lm_dg_last =
        (__scoped_atomic_fetch_add(&count[kh], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u);
  }
  barrier();
  if (!ink_r11_lm_dg_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  G float *ph0 = part + (unsigned long)(h0 + 0u) * nsplit * INK_R11_LM_PSTR;
  G float *ph1 = part + (unsigned long)(h0 + 1u) * nsplit * INK_R11_LM_PSTR;
  G float *ph2 = part + (unsigned long)(h0 + 2u) * nsplit * INK_R11_LM_PSTR;

  float mv0 = (tid() == 0u) ? sink[h0 + 0u] : -__builtin_inff();
  float mv1 = (tid() == 0u) ? sink[h0 + 1u] : -__builtin_inff();
  float mv2 = (tid() == 0u) ? sink[h0 + 2u] : -__builtin_inff();
  for (u32 t = tid(); t < nsplit; t += NT) {
    const unsigned long off = (unsigned long)t * INK_R11_LM_PSTR;
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
    ink_r11_lm_dg_m[0][w] = mv0;
    ink_r11_lm_dg_m[1][w] = mv1;
    ink_r11_lm_dg_m[2][w] = mv2;
  }
  lbarrier();

  float M2[INK_R11_LM_GQA];
  for (u32 g = 0u; g < INK_R11_LM_GQA; g++) {
    float mx = ink_r11_lm_dg_m[g][0];
    for (u32 x = 1u; x < 8u; x++) mx = __builtin_fmaxf(mx, ink_r11_lm_dg_m[g][x]);
    M2[g] = mx;
  }

  const int wt_in_lds = (nsplit <= INK_R11_LM_HD);
  float lv0 = 0.0f, lv1 = 0.0f, lv2 = 0.0f;
  for (u32 t = tid(); t < nsplit; t += NT) {
    const unsigned long off = (unsigned long)t * INK_R11_LM_PSTR;
    const float lp0 = ph0[off + 1u];
    const float wt0 = (lp0 > 0.0f) ? __builtin_expf(ph0[off] - M2[0]) : 0.0f;
    if (wt_in_lds) ink_r11_lm_dg_o[0][0][t] = wt0; else ph0[off] = wt0;
    lv0 += lp0 * wt0;

    const float lp1 = ph1[off + 1u];
    const float wt1 = (lp1 > 0.0f) ? __builtin_expf(ph1[off] - M2[1]) : 0.0f;
    if (wt_in_lds) ink_r11_lm_dg_o[1][0][t] = wt1; else ph1[off] = wt1;
    lv1 += lp1 * wt1;

    const float lp2 = ph2[off + 1u];
    const float wt2 = (lp2 > 0.0f) ? __builtin_expf(ph2[off] - M2[2]) : 0.0f;
    if (wt_in_lds) ink_r11_lm_dg_o[2][0][t] = wt2; else ph2[off] = wt2;
    lv2 += lp2 * wt2;
  }
  lv0 = wave_tree_f(lv0);
  lv1 = wave_tree_f(lv1);
  lv2 = wave_tree_f(lv2);
  if (l == 0u) {
    ink_r11_lm_dg_l[0][w] = lv0;
    ink_r11_lm_dg_l[1][w] = lv1;
    ink_r11_lm_dg_l[2][w] = lv2;
  }
  if (wt_in_lds) lbarrier(); else barrier();

  float L2[INK_R11_LM_GQA];
  for (u32 g = 0u; g < INK_R11_LM_GQA; g++) {
    float ls = 0.0f;
    for (u32 x = 0u; x < 8u; x++) ls += ink_r11_lm_dg_l[g][x];
    L2[g] = ls + __builtin_expf(sink[h0 + g] - M2[g]);
  }

  for (u32 idx = tid(); idx < INK_R11_LM_GQA * INK_R11_LM_HD; idx += NT) {
    const u32 g = idx >> 7u;
    const u32 e = idx & 127u;
    const u32 h = h0 + g;
    const G float *ph = part + (unsigned long)h * nsplit * INK_R11_LM_PSTR;
    float o = 0.0f;
    if (wt_in_lds) {
      for (u32 t = 0u; t < nsplit; t++) o += ph[(unsigned long)t * INK_R11_LM_PSTR + 2u + e] * ink_r11_lm_dg_o[g][0][t];
    } else {
      for (u32 t = 0u; t < nsplit; t++) o += ph[(unsigned long)t * INK_R11_LM_PSTR + 2u + e] * ph[(unsigned long)t * INK_R11_LM_PSTR];
    }
    const u32 out_idx = h * INK_R11_LM_HD + e;
    float v = o / L2[g];
    if (agate) v = ink_r11_lm_gate_f(v, agate[out_idx]);
    if (out) out[out_idx] = v;
    if (oq) quant_wave(v, out_idx >> 5u, oq, nb_q);
  }
  if (tid() == 0u) count[kh] = 0u;
}

#undef INK_R11_LM_AR_MAXM
#undef INK_R11_LM_PSTR
#undef INK_R11_LM_GQA
#undef INK_R11_LM_DREL
#undef INK_R11_LM_HD
#undef INK_R11_LM_CT
#undef INK_R11_LM_NB2_XLS
#undef INK_R11_LM_DEF_KRN_TICKS
#undef INK_R11_LM_DEF_BAR_TICKS
#undef INK_R11_LM_MAX_GRID
#undef INK_R11_LM_TH_HT
#undef INK_R11_LM_RSRC_W3

#endif /* !IE_MEGA */
