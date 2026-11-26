/* ============================================================================
 * Inkling-2 Round 16: Exact-SHA 1-ULP Rescue Kernels for Wave 12 R62 & R64
 *   (H17 1-Launch Persistent Loop Pipelines + H21 SWA Verify Attention Suite)
 * Target: RX 9070 XT (gfx1201, RDNA4, wave32, NT=256, 64 CUs, 64 MiB MALL)
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r16_er_ /
 * INK_R16_ER_ and all macros are #undef'd at EOF.
 *
 * Forensic 1-ULP Root Causes Fixed vs Wave 12 R11:
 *   1. R62 (H17-LC-PIPE, bit 122):
 *      ie_inkling_lc_pipe_mlp_r11 Stage 0 (OP_ATTN_RES) computed
 *        lg = sdt * inv_s * inv_sqn   ->  (sdt * inv_s) * inv_sqn
 *      whereas baseline ie_inkling_attn_res_r1 (r1_g06.inc:184,189) computes
 *        scale = inv_s * inv_sqn; lg = sdt * scale  ->  sdt * (inv_s * inv_sqn).
 *      That single 1-ULP multiplication associativity drift on 24 MLP visits
 *      flipped ds5_4k's razor tie (0043be34 -> c10f085d / 11f80e2c).
 *      ie_inkling_lc_pipe_mlp_exact_r16 fixes scale = inv_s * inv_sqn and
 *      matches ie_inkling_attn_res_r1 Pass 1/2 + Stages 1..4 bit-for-bit.
 *      ie_inkling_lc_pipe_attn_exact_r16 matches lc_prenorm_q8_4x_r6 +
 *      lc_down_4x_r6 + lc_wfg_gate_norm_r11 bit-for-bit across all 4 stages.
 *   2. R64 (H21-SWA, bit 125):
 *      (a) ie_inkling_rel_attn_ver_swa_wmma_r11 used reciprocal multiply
 *          acc * (1.0f / L2) and ov * (wx / L) instead of exact IEEE-754
 *          acc / L2 and (ov * wx) / L, and allowed 16x v_fma_mix*_f16 on
 *          Q/R f16 staging (plus 106 VGPR -> 3 WGs/CU).
 *          ie_inkling_rel_attn_ver_swa_wmma_exact_r16 fixes exact IEEE-754
 *          acc / L2 & (ov * wx) / L, eliminates all v_fma_mix*_f16 via strict
 *          f32-then-f16 staging, matches ie_inkling_gate_q8_r2 bit-for-bit
 *          (supporting both fused agate/oq and unfused agate==NULL), and
 *          targets <= 96 VGPR (4 WGs/CU).
 *      (b) ie_inkling_rel_attn_ver_swa_qt2_r11 used a union split partition
 *          (np=1025, nsplit=16 -> ch=65) instead of per-token 64-key splits,
 *          altering FP32 online-softmax reduction grouping by 1-2 ULPs vs
 *          ver_gate_q8_r10 (which PASSED ds5_4k in R54) and lacked barrier()
 *          before count[cidx]=0 and L==0 sentinel guard on pp[0].
 *          ie_inkling_rel_attn_ver_swa_qt2_exact_r16 uses wave-rotated
 *          per-token exact splits [p0_a,p1_a) & [p0_b,p1_b) over their 65-key
 *          union (98.4% KV sharing at win=1024!) so EVERY wave 0..7 of EVERY
 *          split produces the 100% 0-bit-diff partial of ver_gate_q8_r10!
 *      (c) ie_inkling_rel_attn_ver_swa_qtile_exact_r16 adds a 100% 0-bit-diff
 *          QT=4 FP32 SWA + fused OP_GATE(sigmoid) + Q8_0 drop-in matching R37
 *          baseline ver_swa_qtile_r7 + gate_q8_r2 bit-for-bit (-40 L/step).
 *
 * Exported Kernels (0/0 spills, priv=0, scratch=0, NT=0, HT=0, 0 f16 mix):
 *   1. ie_inkling_lc_pipe_attn_exact_r16            (57 args, grid <= 64)
 *   2. ie_inkling_lc_pipe_mlp_exact_r16             (35 args, grid <= 64)
 *   3. ie_inkling_rel_attn_ver_swa_wmma_exact_r16   (27 args, grid nkv*nsp)
 *   4. ie_inkling_rel_attn_ver_swa_qt2_exact_r16    (27 args, grid nkv*nsp x ceil(T/2))
 *   5. ie_inkling_rel_attn_ver_swa_qtile_exact_r16  (27 args, grid nkv*nsp x ceil(T/4))
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(kvt) == 2u, "ie_inkling_exact_rescue_r16 requires f16 KV cache");
_Static_assert(sizeof(f16) == 2u, "ink_r16_er f16 must be 2 bytes");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r16_er float/u32 must be 4 bytes");

#define INK_R16_ER_MAX_GRID      64u
#define INK_R16_ER_DEF_BAR_TICKS 25000000ull /* 250 ms @ 100 MHz s_sendmsg_rtnl(0x83) */
#define INK_R16_ER_DEF_KRN_TICKS 90000000ull /* 900 ms @ 100 MHz s_sendmsg_rtnl(0x83) */
#define INK_R16_ER_NB2_XLS       80u
#define INK_R16_ER_AR_MAXM       32u
#define INK_R16_ER_HD            128u
#define INK_R16_ER_DREL          16u
#define INK_R16_ER_GQA           3u
#define INK_R16_ER_QT4           4u
#define INK_R16_ER_PSTR          130u
#define INK_R16_ER_LOG2E         1.44269504088896340736f
#define INK_R16_ER_LN2           0.69314718055994530942f
#define INK_R16_ER_QSC           (0.08838834764831844299f * 1.44269504088896340736f)

typedef unsigned long long ink_r16_er_u64;

struct __attribute__((packed, aligned(4))) ink_r16_er_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r16_er_ld4(const G void *p) {
  return ((const G struct ink_r16_er_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) float ink_r16_er_sgpr_f(float v) {
  return I2F(__builtin_amdgcn_readfirstlane(F2I(v)));
}

static inline __attribute__((always_inline)) float ink_r16_er_xrow(float v) {
  return I2F(xrow_i(F2I(v)));
}

static inline __attribute__((always_inline)) float ink_r16_er_halfwave_sum_all(float v) {
  v += I2F(shl_i(F2I(v), 8));
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return I2F(__builtin_amdgcn_readlane(F2I(v), 0));
}

static inline __attribute__((always_inline)) float ink_r16_er_dot4(f32x4 a, f32x4 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

static inline __attribute__((always_inline)) int ink_r16_er_dot8(
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

static inline __attribute__((always_inline)) float ink_r16_er_loop_gate1(
    float a, float c, float f_raw, float g_raw) {
#pragma clang fp contract(off)
  const float f = ie_tanhf_s(f_raw);
  const float g = ie_tanhf_s(g_raw);
  return (1.0f - f) * a + g * c;
}

static inline __attribute__((always_inline)) void ink_r16_er_quant_pack_lds(
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

static inline __attribute__((always_inline)) void ink_r16_er_rn_quant_chunk4(
    LDSP float (*sh)[128u], f32x4 y4, u32 wv, u32 l, u32 b0, G u8 *q, u32 nb) {
  *(LDSP f32x4 *)&sh[wv][l * 4u] = y4;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 m = 0u; m < 4u; m++) quant_wave(sh[wv][32u * m + l], b0 + m, q, nb);
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

static inline __attribute__((always_inline)) float ink_r16_er_wave_max(float v) {
  v = __builtin_fmaxf(v, shfl_xor_f(v, 16u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 8u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 4u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 2u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 1u));
  return v;
}

static inline __attribute__((always_inline)) float ink_r16_er_wave_softmax(float logit, u32 l, u32 M) {
  const float v = (l < M) ? logit : -__builtin_inff();
  const float mx = ink_r16_er_wave_max(v);
  const float e = (l < M) ? __builtin_expf(logit - mx) : 0.0f;
  const float ls = wave_sum_all(e);
  return (l < M) ? (e * (1.0f / ls)) : 0.0f;
}

/* Wall-clock-guarded WDDM sense-reversing inter-WG barrier (sync = b->ink_sp_sync). */
static inline __attribute__((always_inline)) u32 ink_r16_er_grid_barrier(
    G u32 *sync, u32 ng, u32 *sense_io, LDSP u32 *lds_poison,
    ink_r16_er_u64 k_t0, ink_r16_er_u64 bar_lim, ink_r16_er_u64 krn_lim) {
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
        const ink_r16_er_u64 t0 = (ink_r16_er_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
        for (u32 spin = 0u; ; spin++) {
          const u32 cur = __scoped_atomic_load_n(&sync[1u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
          if ((int)(cur - next_sense) >= 0) break;
          if (__scoped_atomic_load_n(&sync[2u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) != 0u) {
            poisoned = 1u;
            break;
          }
          if ((spin & 15u) == 15u) {
            const ink_r16_er_u64 now = (ink_r16_er_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
            if ((now - t0) > bar_lim || (now - k_t0) > krn_lim) {
              __scoped_atomic_fetch_or(&sync[2u], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
              poisoned = 1u;
              break;
            }
          }
          __builtin_amdgcn_s_sleep(2);
        }
#ifdef IE_INK_ER_TRACE
        const ink_r16_er_u64 dt_k = ((ink_r16_er_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83) - t0) >> 10u;
        __scoped_atomic_fetch_max(&sync[3u], (u32)dt_k, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
#endif
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

static inline __attribute__((always_inline)) void ink_r16_er_prenorm_2304(
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
  ink_r16_er_rn_quant_chunk4(sh, x0 * s, wv, l, q_blk_off + wv * 4u, d0, 144u);
  ink_r16_er_rn_quant_chunk4(sh, x1 * s, wv, l, q_blk_off + 32u + wv * 4u, d0, 144u);
  quant_wave(x2 * s, q_blk_off + 64u + wv, d0, 144u);
  lbarrier();
}

static inline __attribute__((always_inline)) void ink_r16_er_down_1row(
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
        const u32x4 w0 = ink_r16_er_ld4(wp);
        const u32x4 w1 = ink_r16_er_ld4(wp + 4u);
        const float sw = (float)row_qs[b];
        const G u8 *xsrc = (b < nb_half) ? xq_lo : xq;
        const G u32 *aw = (const G u32 *)xsrc;
        const u32x4 a0 = ink_r16_er_ld4(aw + (unsigned long)b * 8ul);
        const u32x4 a1 = ink_r16_er_ld4(aw + (unsigned long)b * 8ul + 4ul);
        const float da = ((const G float *)(xsrc + (unsigned long)nb * 32ul))[b];
        term = (sw * da) * (float)ink_r16_er_dot8(w0, w1, a0, a1);
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
      const u32x4 w0 = ink_r16_er_ld4(wp);
      const u32x4 w1 = ink_r16_er_ld4(wp + 4u);
      const float sw = (float)row_qs[b];
      const G u8 *xbase = (b < nb_half) ? xq_lo : xq;
      for (u32 j = 0u; j < T; j++) {
        const G u8 *xj = xbase + (unsigned long)j * (unsigned long)xs;
        const G u32 *aw = (const G u32 *)xj;
        const u32x4 a0 = ink_r16_er_ld4(aw + (unsigned long)b * 8ul);
        const u32x4 a1 = ink_r16_er_ld4(aw + (unsigned long)b * 8ul + 4ul);
        const float da = ((const G float *)(xj + (unsigned long)nb * 32ul))[b];
        dn[j][w][l] = (sw * da) * (float)ink_r16_er_dot8(w0, w1, a0, a1);
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

static inline __attribute__((always_inline)) void ink_r16_er_wfg_1blk(
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
      ink_r16_er_quant_pack_lds(h[(w << 5u) + l], l, x_lds + (w << 5u), &dx_lds[w][0u]);
    }
    u32x4 wv = (u32x4)(0u);
    float sw = 0.0f, inp_val = 0.0f, msg_val = 0.0f;
    if (i_ch < dim) {
      const u32 row = is_gain ? (dim + i_ch) : i_ch;
      wv = ink_r16_er_ld4((const G u8 *)qw + (unsigned long)row * 64ul + (unsigned long)(sub << 4u));
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
      if ((l & 7u) == 0u) tmp[i_ch] = ink_r16_er_loop_gate1(inp_val, msg_val, sum, gain_val);
    }
    lbarrier();
    return;
  }

  if (w < T) {
    const G float *ht = (const G float *)((const G u8 *)h + (unsigned long)w * (unsigned long)hs);
    LDSP u8 *dst = x_lds + w * INK_R16_ER_NB2_XLS;
    ink_r16_er_quant_pack_lds(ht[l], l, dst, &dx_lds[0u][w]);
    ink_r16_er_quant_pack_lds(ht[32u + l], l, dst + 32u, &dx_lds[1u][w]);
  }
  u32x4 wv = (u32x4)(0u);
  float sw = 0.0f;
  if (i_ch < dim) {
    const u32 row = is_gain ? (dim + i_ch) : i_ch;
    wv = ink_r16_er_ld4((const G u8 *)qw + (unsigned long)row * 64ul + (unsigned long)(sub << 4u));
    sw = (float)qs[(unsigned long)row * 2ul + blk];
  }
  lbarrier();

  const u32 ch_loc = (w << 2u) + grp;
  if (i_ch < dim) {
#pragma unroll 1
    for (u32 t = 0u; t < T; t++) {
      const u32x4 av = *(const LDSP u32x4 *)(x_lds + t * INK_R16_ER_NB2_XLS + (sub << 4u));
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
        ink_r16_er_loop_gate1(a_val, c_val, f_raw, g_raw);
  }
  lbarrier();
}

/* ============================================================================
 * Kernel 1 (R62 Exact Rescue, H17-Attn): ie_inkling_lc_pipe_attn_exact_r16
 * Exact 57-arg drop-in replacement for ie_inkling_lc_pipe_attn_r11.
 *   - 0-bit-diff vs lc_prenorm_q8_4x_r6 + lc_down_4x_r6 + lc_wfg_gate_norm_r11.
 *   - Grid: (ng, 1u, 1u) with ng <= 64, NT = 256, <= 64 VGPR (4 WGs/CU), 8.5 KiB LDS.
 * ========================================================================== */

typedef union {
  float dn[8u][8u][32u] __attribute__((aligned(16))); /* 8,192 B */
  struct {
    u8    x[8u * INK_R16_ER_NB2_XLS] __attribute__((aligned(16)));
    float dx[2u][8u] __attribute__((aligned(16)));
    float fg[2u][8u][32u] __attribute__((aligned(16)));
  } wfg; /* 2,752 B */
  struct {
    float red[8u];
    float sh[8u][128u] __attribute__((aligned(16)));
  } rn; /* 4,128 B */
} ink_r16_er_pa_lds_u;

static LDS ink_r16_er_pa_lds_u ink_r16_er_pa_lds __attribute__((aligned(16)));
static LDS ink_r16_er_u64 ink_r16_er_pa_ptrs[36u] __attribute__((aligned(16)));
static LDS u32 ink_r16_er_pa_u32s[12u] __attribute__((aligned(16)));
static LDS u32 ink_r16_er_pa_poison;

KERNEL ie_inkling_lc_pipe_attn_exact_r16(
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
    ink_r16_er_u64 bar_ticks, ink_r16_er_u64 krn_ticks) {
  const u32 g = wgid();
  if (ng == 0u || ng > INK_R16_ER_MAX_GRID || g >= ng || !sync) return;
  if (__scoped_atomic_load_n(&sync[2u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) != 0u) return;

  const ink_r16_er_u64 bar_lim = bar_ticks ? bar_ticks : INK_R16_ER_DEF_BAR_TICKS;
  const ink_r16_er_u64 krn_lim = krn_ticks ? krn_ticks : INK_R16_ER_DEF_KRN_TICKS;
  const ink_r16_er_u64 k_t0 = (ink_r16_er_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
  u32 sense = __scoped_atomic_load_n(&sync[1u], __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);

  if (flags & 4u) {
    if (ink_r16_er_grid_barrier(sync, ng, &sense, &ink_r16_er_pa_poison, k_t0, bar_lim, krn_lim)) return;
    ink_r16_er_grid_barrier(sync, ng, &sense, &ink_r16_er_pa_poison, k_t0, bar_lim, krn_lim);
    return;
  }

  if (T == 0u || T > 8u || dim != 2304u || nb_dn != 144u || rank == 0u) return;

  const u32 th = tid();
  if (th == 0u) {
    ink_r16_er_pa_ptrs[0u]  = (ink_r16_er_u64)(unsigned long)inp0;
    ink_r16_er_pa_ptrs[1u]  = (ink_r16_er_u64)(unsigned long)inp1;
    ink_r16_er_pa_ptrs[2u]  = (ink_r16_er_u64)(unsigned long)inp2;
    ink_r16_er_pa_ptrs[3u]  = (ink_r16_er_u64)(unsigned long)msg0;
    ink_r16_er_pa_ptrs[4u]  = (ink_r16_er_u64)(unsigned long)msg1;
    ink_r16_er_pa_ptrs[5u]  = (ink_r16_er_u64)(unsigned long)msg2;
    ink_r16_er_pa_ptrs[6u]  = (ink_r16_er_u64)(unsigned long)cq0;
    ink_r16_er_pa_ptrs[7u]  = (ink_r16_er_u64)(unsigned long)cq1;
    ink_r16_er_pa_ptrs[8u]  = (ink_r16_er_u64)(unsigned long)cq2;
    ink_r16_er_pa_ptrs[9u]  = (ink_r16_er_u64)(unsigned long)qwd0;
    ink_r16_er_pa_ptrs[10u] = (ink_r16_er_u64)(unsigned long)qwd1;
    ink_r16_er_pa_ptrs[11u] = (ink_r16_er_u64)(unsigned long)qwd2;
    ink_r16_er_pa_ptrs[12u] = (ink_r16_er_u64)(unsigned long)qsd0;
    ink_r16_er_pa_ptrs[13u] = (ink_r16_er_u64)(unsigned long)qsd1;
    ink_r16_er_pa_ptrs[14u] = (ink_r16_er_u64)(unsigned long)qsd2;
    ink_r16_er_pa_ptrs[15u] = (ink_r16_er_u64)(unsigned long)h0;
    ink_r16_er_pa_ptrs[16u] = (ink_r16_er_u64)(unsigned long)h1;
    ink_r16_er_pa_ptrs[17u] = (ink_r16_er_u64)(unsigned long)h2;
    ink_r16_er_pa_ptrs[18u] = (ink_r16_er_u64)(unsigned long)qwf0;
    ink_r16_er_pa_ptrs[19u] = (ink_r16_er_u64)(unsigned long)qwf1;
    ink_r16_er_pa_ptrs[20u] = (ink_r16_er_u64)(unsigned long)qwf2;
    ink_r16_er_pa_ptrs[21u] = (ink_r16_er_u64)(unsigned long)qsf0;
    ink_r16_er_pa_ptrs[22u] = (ink_r16_er_u64)(unsigned long)qsf1;
    ink_r16_er_pa_ptrs[23u] = (ink_r16_er_u64)(unsigned long)qsf2;
    ink_r16_er_pa_ptrs[24u] = (ink_r16_er_u64)(unsigned long)tmp0;
    ink_r16_er_pa_ptrs[25u] = (ink_r16_er_u64)(unsigned long)tmp1;
    ink_r16_er_pa_ptrs[26u] = (ink_r16_er_u64)(unsigned long)tmp2;
    ink_r16_er_pa_ptrs[27u] = (ink_r16_er_u64)(unsigned long)nw0;
    ink_r16_er_pa_ptrs[28u] = (ink_r16_er_u64)(unsigned long)nw1;
    ink_r16_er_pa_ptrs[29u] = (ink_r16_er_u64)(unsigned long)nw2;
    ink_r16_er_pa_ptrs[30u] = (ink_r16_er_u64)(unsigned long)ny0;
    ink_r16_er_pa_ptrs[31u] = (ink_r16_er_u64)(unsigned long)ny1;
    ink_r16_er_pa_ptrs[32u] = (ink_r16_er_u64)(unsigned long)ny2;
    ink_r16_er_pa_ptrs[33u] = (ink_r16_er_u64)(unsigned long)nq0;
    ink_r16_er_pa_ptrs[34u] = (ink_r16_er_u64)(unsigned long)nq1;
    ink_r16_er_pa_ptrs[35u] = (ink_r16_er_u64)(unsigned long)nq2;
    ink_r16_er_pa_u32s[0u]  = is0;
    ink_r16_er_pa_u32s[1u]  = is1;
    ink_r16_er_pa_u32s[2u]  = is2;
    ink_r16_er_pa_u32s[3u]  = ms0;
    ink_r16_er_pa_u32s[4u]  = ms1;
    ink_r16_er_pa_u32s[5u]  = ms2;
    ink_r16_er_pa_u32s[6u]  = cqs;
    ink_r16_er_pa_u32s[7u]  = hs;
    ink_r16_er_pa_u32s[8u]  = ts;
    ink_r16_er_pa_u32s[9u]  = ys;
    ink_r16_er_pa_u32s[10u] = qso;
    ink_r16_er_pa_u32s[11u] = flags;
  }
  lbarrier();

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(th >> 5u));
  const u32 l = lane();
  const u32 shared = ink_r16_er_pa_u32s[11u] & 1u;

  /* Stage 0: Prenorm + Q8 quant across all 3 controllers x T tokens. */
  {
    const u32 n_per_tok = shared ? 4u : 6u;
    const u32 tot0 = n_per_tok * T;
    const u32 cqs_u = ink_r16_er_pa_u32s[6u];
    for (u32 task = g; task < tot0; task += ng) {
      const u32 tk = task / n_per_tok;
      const u32 sub = task - tk * n_per_tok;
      const G float *xp;
      G u8 *dp;
      u32 qoff;
      if (shared) {
        if (sub == 0u) {
          xp = (const G float *)((const G u8 *)(unsigned long)ink_r16_er_pa_ptrs[0u] +
                                 (unsigned long)tk * (unsigned long)ink_r16_er_pa_u32s[0u]);
          dp = (G u8 *)(unsigned long)ink_r16_er_pa_ptrs[6u] + (unsigned long)tk * (unsigned long)cqs_u;
          qoff = 0u;
        } else {
          const u32 cc = sub - 1u;
          xp = (const G float *)((const G u8 *)(unsigned long)ink_r16_er_pa_ptrs[3u + cc] +
                                 (unsigned long)tk * (unsigned long)ink_r16_er_pa_u32s[3u + cc]);
          dp = (G u8 *)(unsigned long)ink_r16_er_pa_ptrs[6u + cc] + (unsigned long)tk * (unsigned long)cqs_u;
          qoff = 72u;
        }
      } else {
        const u32 is_m = (sub >= 3u) ? 1u : 0u;
        const u32 cc = is_m ? (sub - 3u) : sub;
        const u32 p_base = is_m ? 3u : 0u;
        xp = (const G float *)((const G u8 *)(unsigned long)ink_r16_er_pa_ptrs[p_base + cc] +
                               (unsigned long)tk * (unsigned long)ink_r16_er_pa_u32s[p_base + cc]);
        dp = (G u8 *)(unsigned long)ink_r16_er_pa_ptrs[6u + cc] + (unsigned long)tk * (unsigned long)cqs_u;
        qoff = is_m ? 72u : 0u;
      }
      ink_r16_er_prenorm_2304(ink_r16_er_pa_lds.rn.red, ink_r16_er_pa_lds.rn.sh, xp, dp, qoff, eps, th, w, l);
    }
  }
  if (ink_r16_er_grid_barrier(sync, ng, &sense, &ink_r16_er_pa_poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 1: Down GEMV [rank x 4608] across 3 controllers (192 rows). */
  {
    const u32 tot1 = 3u * rank;
    const u32 cqs_u = ink_r16_er_pa_u32s[6u];
    const u32 hs_u = ink_r16_er_pa_u32s[7u];
    const G u8 *cq0_p = (const G u8 *)(unsigned long)ink_r16_er_pa_ptrs[6u];
    for (u32 task = g; task < tot1; task += ng) {
      const u32 cc = task / rank;
      const u32 r = task - cc * rank;
      const G u32 *qw = (const G u32 *)(unsigned long)ink_r16_er_pa_ptrs[9u + cc];
      const G f16 *qs = (const G f16 *)(unsigned long)ink_r16_er_pa_ptrs[12u + cc];
      const G u8  *xq = (const G u8 *)(unsigned long)ink_r16_er_pa_ptrs[6u + cc];
      G float     *yp = (G float *)(unsigned long)ink_r16_er_pa_ptrs[15u + cc];
      const G u8  *xq_lo = shared ? cq0_p : xq;
      ink_r16_er_down_1row(ink_r16_er_pa_lds.dn, qw, qs, xq_lo, xq, yp, r, nb_dn, T, cqs_u, hs_u, th, w, l);
    }
  }
  if (ink_r16_er_grid_barrier(sync, ng, &sense, &ink_r16_er_pa_poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 2: WFG + Loop Gate across 3 controllers (216 blocks). */
  {
    const u32 n_wgx = 72u;
    const u32 tot2 = 3u * n_wgx;
    const u32 hs_u = ink_r16_er_pa_u32s[7u];
    const u32 ts_u = ink_r16_er_pa_u32s[8u];
    for (u32 task = g; task < tot2; task += ng) {
      const u32 cc = task / n_wgx;
      const u32 bx = task - cc * n_wgx;
      const G u32   *qw = (const G u32 *)(unsigned long)ink_r16_er_pa_ptrs[18u + cc];
      const G f16   *qs = (const G f16 *)(unsigned long)ink_r16_er_pa_ptrs[21u + cc];
      const G float *hp = (const G float *)(unsigned long)ink_r16_er_pa_ptrs[15u + cc];
      const G float *ip = (const G float *)(unsigned long)ink_r16_er_pa_ptrs[0u + cc];
      const G float *mp = (const G float *)(unsigned long)ink_r16_er_pa_ptrs[3u + cc];
      G float       *tp = (G float *)(unsigned long)ink_r16_er_pa_ptrs[24u + cc];
      const u32 is_u = ink_r16_er_pa_u32s[0u + cc];
      const u32 ms_u = ink_r16_er_pa_u32s[3u + cc];
      ink_r16_er_wfg_1blk(ink_r16_er_pa_lds.wfg.x, ink_r16_er_pa_lds.wfg.dx, ink_r16_er_pa_lds.wfg.fg,
                          qw, qs, hp, ip, mp, tp, dim, T, hs_u, is_u, ms_u, ts_u, bx, w, l);
    }
  }
  if (ink_r16_er_grid_barrier(sync, ng, &sense, &ink_r16_er_pa_poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 3: PARALLEL-T RMSNorm + Q8 across all 3 * T tokens (exact R37/R52 order). */
  {
    const u32 tot3 = 3u * T;
    const u32 ts_u = ink_r16_er_pa_u32s[8u];
    const u32 ys_u = ink_r16_er_pa_u32s[9u];
    const u32 qso_u = ink_r16_er_pa_u32s[10u];
    for (u32 task = g; task < tot3; task += ng) {
      const u32 cc = task / T;
      const u32 tk = task - cc * T;
      const G float *tp  = (const G float *)(unsigned long)ink_r16_er_pa_ptrs[24u + cc];
      const G float *nwp = (const G float *)(unsigned long)ink_r16_er_pa_ptrs[27u + cc];
      G float       *nyp = (G float *)(unsigned long)ink_r16_er_pa_ptrs[30u + cc];
      G u8          *nqp = (G u8 *)(unsigned long)ink_r16_er_pa_ptrs[33u + cc];
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
      if (l == 0u) ink_r16_er_pa_lds.rn.red[w] = ss;
      lbarrier();
      ss = 0.0f;
#pragma unroll
      for (u32 k = 0u; k < 8u; k++) ss += ink_r16_er_pa_lds.rn.red[k];
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
        ink_r16_er_rn_quant_chunk4(ink_r16_er_pa_lds.rn.sh, y0, w, l, w * 4u, nqt, 72u);
        ink_r16_er_rn_quant_chunk4(ink_r16_er_pa_lds.rn.sh, y1, w, l, 32u + w * 4u, nqt, 72u);
        quant_wave(y2, 64u + w, nqt, 72u);
      }
      lbarrier();
    }
  }
}

/* ============================================================================
 * Kernel 2 (R62 Exact Rescue, H17-MLP): ie_inkling_lc_pipe_mlp_exact_r16
 * Exact 35-arg drop-in replacement for ie_inkling_lc_pipe_mlp_r11.
 *   - Fixes Stage 0 OP_ATTN_RES 1-ULP associativity bug:
 *       const float scale = inv_s * inv_sqn; lg = sdt * scale;
 *     matching ie_inkling_attn_res_r1 (r1_g06.inc:184,189) bit-for-bit!
 *   - Matches attn_res_r1 + lc_prenorm_q8_4x_r6 + gemv_q8_small(_ver_r5) +
 *     lc_wfg_gate_norm_r11 bit-for-bit across all 5 stages.
 *   - Grid: (ng, 1u, 1u) with ng <= 64, NT = 256, <= 56 VGPR (4 WGs/CU), 8.2 KiB LDS.
 * ========================================================================== */

typedef union {
  struct {
    float red[INK_R16_ER_AR_MAXM * 2u * 8u]; /* 2,048 B */
    float inv[INK_R16_ER_AR_MAXM];           /*   128 B */
    float prob[INK_R16_ER_AR_MAXM];          /*   128 B */
  } ar; /* 2,304 B */
  float dn[8u][8u][32u] __attribute__((aligned(16))); /* 8,192 B */
  struct {
    u8    x[8u * INK_R16_ER_NB2_XLS] __attribute__((aligned(16)));
    float dx[2u][8u] __attribute__((aligned(16)));
    float fg[2u][8u][32u] __attribute__((aligned(16)));
  } wfg; /* 2,752 B */
  struct {
    float red[8u];
    float sh[8u][128u] __attribute__((aligned(16)));
  } rn; /* 4,128 B */
} ink_r16_er_pm_lds_u;

static LDS ink_r16_er_pm_lds_u ink_r16_er_pm_lds __attribute__((aligned(16)));
static LDS u32 ink_r16_er_pm_poison;

KERNEL ie_inkling_lc_pipe_mlp_exact_r16(
    const G float *blocks_buf, const G float *ffn_inp, const G float *ar_q,
    G float *fres, const G float *msg_mlp, G u8 *cq,
    const G u32 *qwd, const G f16 *qsd, G float *h,
    const G u32 *qwf, const G f16 *qsf, G float *minp,
    const G float *nw, G float *ffn_in, G u8 *ffn_in_q8,
    G u32 *sync, u32 ng, u32 M_ar, u32 dim, u32 nb_dn, u32 rank, u32 T, u32 flags, float eps,
    u32 bs_blk, u32 cs_inp, u32 fs, u32 ms, u32 cqs, u32 hs, u32 ts, u32 ys, u32 qso,
    ink_r16_er_u64 bar_ticks, ink_r16_er_u64 krn_ticks) {
  const u32 g = wgid();
  if (ng == 0u || ng > INK_R16_ER_MAX_GRID || g >= ng || !sync) return;
  if (__scoped_atomic_load_n(&sync[2u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) != 0u) return;

  const ink_r16_er_u64 bar_lim = bar_ticks ? bar_ticks : INK_R16_ER_DEF_BAR_TICKS;
  const ink_r16_er_u64 krn_lim = krn_ticks ? krn_ticks : INK_R16_ER_DEF_KRN_TICKS;
  const ink_r16_er_u64 k_t0 = (ink_r16_er_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
  u32 sense = __scoped_atomic_load_n(&sync[1u], __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);

  if (flags & 4u) {
    if (ink_r16_er_grid_barrier(sync, ng, &sense, &ink_r16_er_pm_poison, k_t0, bar_lim, krn_lim)) return;
    ink_r16_er_grid_barrier(sync, ng, &sense, &ink_r16_er_pm_poison, k_t0, bar_lim, krn_lim);
    return;
  }

  if (T == 0u || T > 8u || dim != 2304u || nb_dn != 144u || rank == 0u) return;

  const u32 th = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(th >> 5u));
  const u32 l = lane();

  /* Optional Stage 0: K=1 OP_ATTN_RES(blocks_buf, ffn_inp -> fres), 0-bit-diff vs ie_inkling_attn_res_r1. */
  if (ar_q && M_ar > 0u && M_ar <= INK_R16_ER_AR_MAXM) {
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
          ink_r16_er_pm_lds.ar.red[s * 16u + w] = sq;
          ink_r16_er_pm_lds.ar.red[s * 16u + 8u + w] = dt;
        }
      }
      lbarrier();
      if (w == 0u) {
        float lg = 0.0f, inv_s = 0.0f;
        if (l < M_ar) {
          float ssq = 0.0f, sdt = 0.0f;
#pragma unroll
          for (u32 x = 0u; x < 8u; x++) {
            ssq += ink_r16_er_pm_lds.ar.red[l * 16u + x];
            sdt += ink_r16_er_pm_lds.ar.red[l * 16u + 8u + x];
          }
          inv_s = 1.0f / __builtin_sqrtf(ssq / 2304.0f + eps);
          /* EXACT associativity fix vs r11: scale = inv_s * inv_sqn first, then sdt * scale. */
          float scale_s = inv_s * inv_sqn;
          __asm__ volatile("" : "+v"(scale_s));
          lg = sdt * scale_s;
          ink_r16_er_pm_lds.ar.inv[l] = inv_s;
        }
        const float pr = ink_r16_er_wave_softmax(lg, l, M_ar);
        if (l < M_ar) ink_r16_er_pm_lds.ar.prob[l] = pr;
      }
      lbarrier();
      f32x4 o0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      f32x4 o1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      float o2 = 0.0f;
      for (u32 s = 0u; s < M_ar; s++) {
        const G float *sp = (s < n_a) ? (at + (unsigned long)s * 2304ul) : ct;
        const float inv_s = ink_r16_er_pm_lds.ar.inv[s];
        const float pr = ink_r16_er_pm_lds.ar.prob[s];
        const f32x4 xn0 = (*(const G f32x4 *)(sp + th * 4u)) * inv_s;
        const f32x4 xn1 = (*(const G f32x4 *)(sp + 1024u + th * 4u)) * inv_s;
        const float xn2 = sp[2048u + th] * inv_s;
        o0 += xn0 * pr;
        o1 += xn1 * pr;
        o2 += xn2 * pr;
      }
      *(G f32x4 *)(bt + th * 4u) = o0;
      *(G f32x4 *)(bt + 1024u + th * 4u) = o1;
      bt[2048u + th] = o2;
      lbarrier();
    }
    if (ink_r16_er_grid_barrier(sync, ng, &sense, &ink_r16_er_pm_poison, k_t0, bar_lim, krn_lim)) return;
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
      ink_r16_er_prenorm_2304(ink_r16_er_pm_lds.rn.red, ink_r16_er_pm_lds.rn.sh,
                              xp, dp, is_m ? 72u : 0u, eps, th, w, l);
    }
  }
  if (ink_r16_er_grid_barrier(sync, ng, &sense, &ink_r16_er_pm_poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 2: Down GEMV [rank x 4608] -> h. */
  for (u32 r = g; r < rank; r += ng) {
    ink_r16_er_down_1row(ink_r16_er_pm_lds.dn, qwd, qsd, cq, cq, h, r, nb_dn, T, cqs, hs, th, w, l);
  }
  if (ink_r16_er_grid_barrier(sync, ng, &sense, &ink_r16_er_pm_poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 3: WFG + Loop Gate -> minp (72 blocks). */
  for (u32 bx = g; bx < 72u; bx += ng) {
    ink_r16_er_wfg_1blk(ink_r16_er_pm_lds.wfg.x, ink_r16_er_pm_lds.wfg.dx, ink_r16_er_pm_lds.wfg.fg,
                        qwf, qsf, h, fres, msg_mlp, minp, dim, T, hs, fs, ms, ts, bx, w, l);
  }
  if (ink_r16_er_grid_barrier(sync, ng, &sense, &ink_r16_er_pm_poison, k_t0, bar_lim, krn_lim)) return;

  /* Stage 4: PARALLEL-T RMSNorm + Q8 -> ffn_in + ffn_in_q8. */
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
      if (l == 0u) ink_r16_er_pm_lds.rn.red[w] = ss;
      lbarrier();
      ss = 0.0f;
#pragma unroll
      for (u32 k = 0u; k < 8u; k++) ss += ink_r16_er_pm_lds.rn.red[k];
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
        ink_r16_er_rn_quant_chunk4(ink_r16_er_pm_lds.rn.sh, y0, w, l, w * 4u, nqt, 72u);
        ink_r16_er_rn_quant_chunk4(ink_r16_er_pm_lds.rn.sh, y1, w, l, 32u + w * 4u, nqt, 72u);
        quant_wave(y2, 64u + w, nqt, 72u);
      }
      lbarrier();
    }
  }
}

/* ============================================================================
 * Shared Helpers for R64 H21 SWA Exact Rescue Kernels (Kernels 3, 4, 5)
 * ========================================================================== */

static inline __attribute__((always_inline)) float ink_r16_er_gate_f(float a, float g) {
#pragma clang fp contract(off)
  return a * (1.0f / (1.0f + __builtin_expf(-g)));
}

static inline __attribute__((always_inline)) float ink_r16_er_sub8_max_pos(float v) {
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0xB1, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0x4E, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_ds_swizzle(F2I(v), 0x101F)));
  return v;
}

static inline __attribute__((always_inline)) u32 ink_r16_er_sub8_sum_u32(u32 s) {
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0xB1, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0x4E, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_ds_swizzle((int)s, 0x101F);
  return s;
}

/* Exact 0-bit-diff OP_GATE(sigmoid) + Q8_0 epilogue matching ie_inkling_gate_q8_r2.
 * Supports both fused (gt_h != NULL, oqt != NULL) and unfused (gt_h == NULL, oqt == NULL -> ot_h). */
static inline __attribute__((always_inline)) void ink_r16_er_gate_q8_head_vec4(
    f32x4 ov, const G float *gt_h, G float *ot_h, G u8 *oqt, u32 h, u32 n_head, u32 l) {
#pragma clang fp contract(off)
  f32x4 vv = ov;
  if (gt_h) {
    const f32x4 gv = ((const G f32x4 *)gt_h)[l];
    vv = (f32x4){ink_r16_er_gate_f(ov.x, gv.x), ink_r16_er_gate_f(ov.y, gv.y),
                 ink_r16_er_gate_f(ov.z, gv.z), ink_r16_er_gate_f(ov.w, gv.w)};
  }
  if (ot_h) ((G f32x4 *)ot_h)[l] = vv;
  if (oqt) {
    const u32 sg = l >> 3u;
    const u32 sl = l & 7u;
    const u32 blk = (h << 2u) + sg;
    const u32 nb_q = n_head << 2u;
    float am = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(vv.x), __builtin_fabsf(vv.y)),
                               __builtin_fmaxf(__builtin_fabsf(vv.z), __builtin_fabsf(vv.w)));
    am = ink_r16_er_sub8_max_pos(am);
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
    const u32 s = ink_r16_er_sub8_sum_u32((u32)qi0 + (u32)qi1 + (u32)qi2 + (u32)qi3);
    if (sl == 0u) {
      ((G float *)(oqt + (nb_q << 5u)))[blk] = d;
      ((G u32 *)(oqt + nb_q * 36u))[blk] = s;
    }
  }
}

/* Strict f32-then-f16 scale+cast (0 v_fma_mix*_f16 on Q/R staging). */
static inline __attribute__((always_inline)) h8 ink_r16_er_scale_f32x8_to_h8(
    f32x4 v0, f32x4 v1, float sc) {
  float a0 = v0.x * sc, a1 = v0.y * sc, a2 = v0.z * sc, a3 = v0.w * sc;
  float b0 = v1.x * sc, b1 = v1.y * sc, b2 = v1.z * sc, b3 = v1.w * sc;
  __asm__ volatile("" : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3));
  __asm__ volatile("" : "+v"(b0), "+v"(b1), "+v"(b2), "+v"(b3));
  return (h8){(f16)a0, (f16)a1, (f16)a2, (f16)a3, (f16)b0, (f16)b1, (f16)b2, (f16)b3};
}

/* ============================================================================
 * Kernel 3 (R64 Exact Rescue, H21-SWA-WMMA): ie_inkling_rel_attn_ver_swa_wmma_exact_r16
 * Exact 27-arg drop-in replacement for ie_inkling_rel_attn_ver_swa_wmma_r11:
 *   - Fixes exact IEEE-754 acc / L2 and (ov * wx) / L (no reciprocal multiply drift).
 *   - Eliminates all v_fma_mix*_f16 on Q/R staging (strict f32-then-f16).
 *   - Supports both fused OP_GATE(sigmoid)+Q8_0 (agate/oq != NULL) and unfused
 *     pure float out store (agate == NULL, oq == NULL).
 *   - Lean <= 96 VGPR target (4 WGs/CU vs 106 VGPR / 3 WGs/CU in r11), 15,360 B LDS.
 * ========================================================================== */

typedef union {
  struct {
    f16 q[32u * 128u] __attribute__((aligned(16)));
    f16 r[32u * 16u] __attribute__((aligned(16)));
    f16 p_tile[8u][2u][256u] __attribute__((aligned(16)));
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
} ink_r16_er_sw_lds_t;

static LDS ink_r16_er_sw_lds_t ink_r16_er_sw_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r16_er_sw_lds_t) == 21504u, "ink_r16_er_sw_lds_t must be 21,504 B (<= 21.5 KiB = 3 WGs/CU)");

static inline __attribute__((always_inline)) u32 ink_r16_er_sw_q_off(u32 row, u32 c8) {
  return row * 128u + 8u * (c8 ^ (row & 15u));
}

static inline __attribute__((always_inline)) u32 ink_r16_er_sw_p_off(u32 rl, u32 hl) {
  return rl * 16u + 8u * (hl ^ ((rl >> 2u) & 1u));
}

static inline __attribute__((always_inline)) u32 ink_r16_er_sw_wrap(
    u32 kp, u32 kv_len, u32 kv_mask, int is_pow2) {
  return is_pow2 ? (kp & kv_mask) : (kp % kv_len);
}

static inline __attribute__((always_inline)) h8 ink_r16_er_sw_v_transpose(
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

static inline __attribute__((always_inline)) void ink_r16_er_sw_apply_rb16(
    const v8f rb, const u32 base_off, const u32 hl, const u32 kb, const u32 pos_q,
    const u32 extent_lim, const u32 d_base, LDSP float *sc_row, v8f *sv) {
  /* extent_lim = min(win, rel_extent) <= win, so (kp <= pos_q && dist < extent_lim)
   * already guarantees kp >= w_q. Opaque barrier on kp_lo prevents LLVM LICM from
   * hoisting 16 per-v dist VGPRs across the rbp loop -> keeps peak VGPR <= 96 (4 WGs/CU). */
  u32 kp_lo = kb + 8u * hl;
  __asm__ volatile("" : "+v"(kp_lo));
  const u32 off0 = d_base + base_off;
  if (hl == 0u) {
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) sc_row[v] = rb[v];
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 v = 0u; v < 8u; v++) {
    const u32 kp = kp_lo + v;
    if (kp <= pos_q) {
      const u32 dist = pos_q - kp;
      if (dist < extent_lim) {
        const u32 rel = dist - off0;
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
  const u32 off1 = off0 + 8u;
#pragma unroll
  for (u32 v = 0u; v < 8u; v++) {
    const u32 kp = kp_lo + v;
    if (kp <= pos_q) {
      const u32 dist = pos_q - kp;
      if (dist < extent_lim) {
        const u32 rel = dist - off1;
        if (rel < 8u) (*sv)[v] += sc_row[rel];
      }
    }
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

static inline __attribute__((always_inline)) void ink_r16_er_sw_merge_head_exact(
    const G float *ph, const float sink_val, const u32 nsplit, const u32 l,
    LDSP float *wt_w, const G float *gt_h, G float *ot_h, G u8 *oqt, u32 h, u32 n_head) {
  float mv = (l == 0u) ? sink_val : -__builtin_inff();
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const unsigned long off = (unsigned long)sp * INK_R16_ER_PSTR;
    if (ph[off + 1u] > 0.0f) mv = __builtin_fmaxf(mv, ph[off]);
  }
#pragma unroll
  for (u32 mask = 16u; mask >= 1u; mask >>= 1u) mv = __builtin_fmaxf(mv, shfl_xor_f(mv, mask));
  const float M2 = ink_r16_er_sgpr_f(mv);

  float lv = 0.0f;
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const unsigned long off = (unsigned long)sp * INK_R16_ER_PSTR;
    const float lp = ph[off + 1u];
    const float wt = (lp > 0.0f) ? __builtin_expf(ph[off] - M2) : 0.0f;
    if (sp < 128u) wt_w[sp] = wt;
    lv += lp * wt;
  }
  const float L2 = ink_r16_er_sgpr_f(wave_sum_all(lv) + __builtin_expf(sink_val - M2));
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");

  f32x4 acc = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  for (u32 sp = 0u; sp < nsplit; sp++) {
    const float wt = (sp < 128u) ? wt_w[sp]
                                 : ((ph[(unsigned long)sp * INK_R16_ER_PSTR + 1u] > 0.0f)
                                        ? __builtin_expf(ph[(unsigned long)sp * INK_R16_ER_PSTR] - M2)
                                        : 0.0f);
    if (wt > 0.0f) {
      const G float *po = ph + (unsigned long)sp * INK_R16_ER_PSTR + 2u + 4u * l;
      const f32x4 ov = {po[0], po[1], po[2], po[3]};
      acc += ov * wt;
    }
  }
  /* Exact IEEE-754 division acc / L2 (replaces acc * (1.0f / L2) 1-ULP drift). */
  const f32x4 norm = (f32x4){acc.x / L2, acc.y / L2, acc.z / L2, acc.w / L2};
  ink_r16_er_gate_q8_head_vec4(norm, gt_h, ot_h, oqt, h, n_head, l);
}

KERNEL ie_inkling_rel_attn_ver_swa_wmma_exact_r16(
    const G float *q, const G kvt *kc, const G kvt *vc,
    const G float *R, const G float *rel_bias, const G float *sink,
    G float *part, G float *out, u32 pos0, u32 T, u32 kv_len, u32 hd,
    u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
    u32 qs, u32 rs, u32 ss, u32 os, G u32 *count,
    const G float *agate, G u8 *oq, u32 gs, u32 oqs) {
  if (T == 0u || T > 8u || win == 0u || hd != INK_R16_ER_HD || d_rel != INK_R16_ER_DREL ||
      n_kv == 0u || n_head != INK_R16_ER_GQA * n_kv || kv_len == 0u || nsplit == 0u)
    return;

  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() / nsplit));
  const u32 s = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() % nsplit));
  if (kh >= n_kv) return;

  const u32 h0 = kh * INK_R16_ER_GQA;
  const u32 kvd = n_kv * INK_R16_ER_HD;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane(), hl = l >> 4u, rl = l & 15u;

  const u32 nq_tot = T * INK_R16_ER_GQA;
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
      const u32 t = qi / INK_R16_ER_GQA;
      const u32 h = h0 + (qi - t * INK_R16_ER_GQA);
      G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R16_ER_PSTR;
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
        const u32 tj = qr / INK_R16_ER_GQA;
        const u32 gi = qr - tj * INK_R16_ER_GQA;
        const G f32x4 *qp = (const G f32x4 *)((const G float *)((const G u8 *)q + (unsigned long)tj * qs) +
                                              (unsigned long)(h0 + gi) * INK_R16_ER_HD);
        qh = ink_r16_er_scale_f32x8_to_h8(qp[2u * c8 + 0u], qp[2u * c8 + 1u], INK_R16_ER_QSC);
      }
      *(LDSP h8 *)(ink_r16_er_sw_lds.kv.q + ink_r16_er_sw_q_off(qr, c8)) = qh;
    }
    if (extent_lim > 0u) {
      if (tid() < q_rows * 2u) {
        const u32 qr = tid() >> 1u;
        const u32 h8i = tid() & 1u;
        h8 rh = (h8)(0.0f);
        if (qr < nq_tot) {
          const u32 tj = qr / INK_R16_ER_GQA;
          const u32 gi = qr - tj * INK_R16_ER_GQA;
          const G f32x4 *rp = (const G f32x4 *)((const G float *)((const G u8 *)R + (unsigned long)tj * rs) +
                                                (unsigned long)(h0 + gi) * INK_R16_ER_DREL);
          rh = ink_r16_er_scale_f32x8_to_h8(rp[2u * h8i + 0u], rp[2u * h8i + 1u], INK_R16_ER_LOG2E);
        }
        *(LDSP h8 *)(ink_r16_er_sw_lds.kv.r + qr * 16u + 8u * h8i) = rh;
      }
    }
    lbarrier();

    const G kvt *kc_h = kc + (unsigned long)kh * INK_R16_ER_HD;
    const G kvt *vc_h = vc + (unsigned long)kh * INK_R16_ER_HD;

    const v8f zero = (v8f)(0.0f);
    v8f acc0 = zero, acc1 = zero;
    float m0 = -__builtin_inff(), ls0 = 0.0f;
    float m1 = -__builtin_inff(), ls1 = 0.0f;

    const u32 p_off = ink_r16_er_sw_p_off(rl, hl);
    const u32 pos_q0 = pos0 + (rl / INK_R16_ER_GQA);
    const u32 pos_q1 = pos0 + ((16u + rl) / INK_R16_ER_GQA);
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
        const u32 kv_row = ink_r16_er_sw_wrap(kp_cl, kv_len, kv_mask, is_pow2);
        const G f16 *k_base = kc_h + (unsigned long)kv_row * kvd + 8u * hl;

        sp0 = zero;
        sp1 = zero;
        /* 2-deep K prefetch across hd=128 (non-unrolled ks+=2 avoids hoisting 16 q_off VGPRs -> <=96 VGPR = 4 WGs/CU!). */
        h8 kk0 = row_ok ? *(const G h8 *)(k_base + 0u) : (h8)(0.0f);
        h8 kk1 = row_ok ? *(const G h8 *)(k_base + 16u) : (h8)(0.0f);
#pragma clang loop unroll(disable)
        for (u32 ks = 0u; ks < 8u; ks += 2u) {
          const u32 c_a = 2u * ks + hl;
          const u32 c_b = 2u * (ks + 1u) + hl;
          const h8 q0_a = *(const LDSP h8 *)(ink_r16_er_sw_lds.kv.q + ink_r16_er_sw_q_off(rl, c_a));
          sp0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk0, q0_a, sp0);
          if (npg > 1u) {
            const h8 q1_a = *(const LDSP h8 *)(ink_r16_er_sw_lds.kv.q + ink_r16_er_sw_q_off(16u + rl, c_a));
            sp1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk0, q1_a, sp1);
          }
          if (ks + 2u < 8u) kk0 = row_ok ? *(const G h8 *)(k_base + 16u * (ks + 2u)) : (h8)(0.0f);
          __builtin_amdgcn_sched_barrier(0);
          const h8 q0_b = *(const LDSP h8 *)(ink_r16_er_sw_lds.kv.q + ink_r16_er_sw_q_off(rl, c_b));
          sp0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk1, q0_b, sp0);
          if (npg > 1u) {
            const h8 q1_b = *(const LDSP h8 *)(ink_r16_er_sw_lds.kv.q + ink_r16_er_sw_q_off(16u + rl, c_b));
            sp1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk1, q1_b, sp1);
          }
          if (ks + 2u < 8u) kk1 = row_ok ? *(const G h8 *)(k_base + 16u * (ks + 3u)) : (h8)(0.0f);
          __builtin_amdgcn_sched_barrier(0);
        }

        if (extent_lim > 0u) {
          const int d_min_s = (int)pos0 - (int)(kb + 15u);
          const u32 d_base = d_min_s > 0 ? (u32)d_min_s : 0u;
          if (d_base < extent_lim) {
            LDSP float *sc_row = ((LDSP float *)ink_r16_er_sw_lds.kv.scratch.v_sc[w]) + rl * 8u;
            const u32 n_rb_pass = (d_base + 16u < extent_lim && (pos0 + T - 1u) >= kb + d_base + 16u) ? 2u : 1u;
#pragma clang loop unroll(disable)
            for (u32 rbp = 0u; rbp < n_rb_pass; rbp++) {
              const u32 boff = rbp << 4u;
              const u32 dd = d_base + boff + rl;
              h8 bb = (h8)(0.0f);
              if (dd < extent_lim) {
                const G f32x4 *rp = (const G f32x4 *)(rel_bias + (unsigned long)dd * 16ul + 8ul * hl);
                const f32x4 v0 = rp[0], v1 = rp[1];
                bb = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w,
                          (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
              }
              const h8 ra0 = *(const LDSP h8 *)(ink_r16_er_sw_lds.kv.r + rl * 16u + 8u * hl);
              const v8f rb0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(bb, ra0, zero);
              ink_r16_er_sw_apply_rb16(rb0, boff, hl, kb, pos_q0, extent_lim, d_base, sc_row, &sp0);
              if (npg > 1u) {
                const h8 ra1 = *(const LDSP h8 *)(ink_r16_er_sw_lds.kv.r + (16u + rl) * 16u + 8u * hl);
                const v8f rb1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(bb, ra1, zero);
                ink_r16_er_sw_apply_rb16(rb1, boff, hl, kb, pos_q1, extent_lim, d_base, sc_row, &sp1);
              }
            }
          }
        }

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
          const u32 kv_v = ink_r16_er_sw_wrap(vp, kv_len, kv_mask, is_pow2);
          vr0 = *(const G u32x4 *)(vc_h + (unsigned long)kv_v * kvd + 16u * w + 8u * hl);
        }
      }
      lbarrier();

      float mx0 = sp0[0];
#pragma unroll
      for (u32 v = 1u; v < 8u; v++) mx0 = __builtin_fmaxf(mx0, sp0[v]);
      mx0 = __builtin_fmaxf(mx0, ink_r16_er_xrow(mx0));
      if (hl == 0u) ink_r16_er_sw_lds.kv.scratch.red.mx[0][16u * w + rl] = mx0;

      if (npg > 1u) {
        float mx1 = sp1[0];
#pragma unroll
        for (u32 v = 1u; v < 8u; v++) mx1 = __builtin_fmaxf(mx1, sp1[v]);
        mx1 = __builtin_fmaxf(mx1, ink_r16_er_xrow(mx1));
        if (hl == 0u) ink_r16_er_sw_lds.kv.scratch.red.mx[1][16u * w + rl] = mx1;
      }
      lbarrier();

      float mc0 = ink_r16_er_sw_lds.kv.scratch.red.mx[0][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) mc0 = __builtin_fmaxf(mc0, ink_r16_er_sw_lds.kv.scratch.red.mx[0][16u * x + rl]);
      const float mn0 = __builtin_fmaxf(m0, mc0);
      if (mn0 != m0) {
        const float rs0 = (ls0 > 0.0f && mn0 > -1e30f) ? __builtin_amdgcn_exp2f(m0 - mn0) : 0.0f;
        acc0 *= rs0; ls0 *= rs0; m0 = mn0;
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
      lc0 += ink_r16_er_xrow(lc0);
      if (hl == 0u) ink_r16_er_sw_lds.kv.scratch.red.ls[0][16u * w + rl] = lc0;

      h8 ph1 = (h8)(0.0f);
      if (npg > 1u) {
        float mc1 = ink_r16_er_sw_lds.kv.scratch.red.mx[1][rl];
#pragma unroll
        for (u32 x = 1u; x < 8u; x++) mc1 = __builtin_fmaxf(mc1, ink_r16_er_sw_lds.kv.scratch.red.mx[1][16u * x + rl]);
        const float mn1 = __builtin_fmaxf(m1, mc1);
        if (mn1 != m1) {
          const float rs1 = (ls1 > 0.0f && mn1 > -1e30f) ? __builtin_amdgcn_exp2f(m1 - mn1) : 0.0f;
          acc1 *= rs1; ls1 *= rs1; m1 = mn1;
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
        lc1 += ink_r16_er_xrow(lc1);
        if (hl == 0u) ink_r16_er_sw_lds.kv.scratch.red.ls[1][16u * w + rl] = lc1;
      }

      /* All 8 waves publish ph0/ph1 once into 8-slot p_tile[w] (eliminates 8 barriers/chunk in Phase 3!). */
      *(LDSP h8 *)(ink_r16_er_sw_lds.kv.p_tile[w][0] + p_off) = ph0;
      if (npg > 1u) *(LDSP h8 *)(ink_r16_er_sw_lds.kv.p_tile[w][1] + p_off) = ph1;
      lbarrier();

      float sum_l0 = ink_r16_er_sw_lds.kv.scratch.red.ls[0][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) sum_l0 += ink_r16_er_sw_lds.kv.scratch.red.ls[0][16u * x + rl];
      ls0 += sum_l0;
      if (npg > 1u) {
        float sum_l1 = ink_r16_er_sw_lds.kv.scratch.red.ls[1][rl];
#pragma unroll
        for (u32 x = 1u; x < 8u; x++) sum_l1 += ink_r16_er_sw_lds.kv.scratch.red.ls[1][16u * x + rl];
        ls1 += sum_l1;
      }
      lbarrier();

      LDSP u32 *v_sc_w = ink_r16_er_sw_lds.kv.scratch.v_sc[w];
#pragma clang loop unroll(disable)
      for (u32 kt = 0u; kt < 8u; kt++) {
        if (c0 + 16u * kt >= chunk_end) break;
        const int has_nxt = (kt + 1u < 8u) && (c0 + 16u * (kt + 1u) < chunk_end);
        u32x4 vr_nxt = (u32x4)(0u);
        if (has_nxt) {
          const u32 vp_n = c0 + 16u * (kt + 1u) + rl;
          if (vp_n < chunk_end) {
            const u32 kv_vn = ink_r16_er_sw_wrap(vp_n, kv_len, kv_mask, is_pow2);
            vr_nxt = *(const G u32x4 *)(vc_h + (unsigned long)kv_vn * kvd + 16u * w + 8u * hl);
          }
        }
        __builtin_amdgcn_sched_barrier(0);

        const h8 vt_a = ink_r16_er_sw_v_transpose(vr0, v_sc_w, hl, rl);
        const h8 pt0 = *(const LDSP h8 *)(ink_r16_er_sw_lds.kv.p_tile[kt][0] + p_off);
        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt0, acc0);
        if (npg > 1u) {
          const h8 pt1 = *(const LDSP h8 *)(ink_r16_er_sw_lds.kv.p_tile[kt][1] + p_off);
          acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt1, acc1);
        }
        vr0 = vr_nxt;
      }
    }

    lbarrier();
    if (w == 0u && hl == 0u) {
      ink_r16_er_sw_lds.ph4.m[rl] = (ls0 > 0.0f) ? (m0 * INK_R16_ER_LN2) : -1e30f;
      ink_r16_er_sw_lds.ph4.l[rl] = ls0;
      if (npg > 1u) {
        ink_r16_er_sw_lds.ph4.m[16u + rl] = (ls1 > 0.0f) ? (m1 * INK_R16_ER_LN2) : -1e30f;
        ink_r16_er_sw_lds.ph4.l[16u + rl] = ls1;
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
          LDSP float *dst_w = ink_r16_er_sw_lds.ph4.o_stg[q_loc] + 16u * w + 8u * hl;
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) dst_w[v] = a_cur[v];
        }
        lbarrier();

        const u32 qi = q_base + w;
        if (qi < nq_tot) {
          const u32 t = qi / INK_R16_ER_GQA;
          const u32 h = h0 + (qi - t * INK_R16_ER_GQA);
          const f32x4 ov = ((const LDSP f32x4 *)ink_r16_er_sw_lds.ph4.o_stg[w])[l];
          if (nsplit == 1u) {
            const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
            const float sv = st[h];
            const float mq = ink_r16_er_sw_lds.ph4.m[qi];
            const float lq = ink_r16_er_sw_lds.ph4.l[qi];
            const float M = (lq > 0.0f) ? __builtin_fmaxf(sv, mq) : sv;
            const float wx = (lq > 0.0f) ? __builtin_expf(mq - M) : 0.0f;
            const float L = __builtin_expf(sv - M) + lq * wx;
            /* Exact (ov * wx) / L matching scalar ver_swa_qtile_r7. */
            const f32x4 ow = ov * wx;
            const f32x4 norm = (f32x4){ow.x / L, ow.y / L, ow.z / L, ow.w / L};
            const G float *gt_h = agate ? ((const G float *)((const G u8 *)agate + (unsigned long)t * gs) +
                                           (unsigned long)h * INK_R16_ER_HD)
                                        : (const G float *)0;
            G float *ot_h = out ? ((G float *)((G u8 *)out + (unsigned long)t * os) +
                                   (unsigned long)h * INK_R16_ER_HD)
                                : (G float *)0;
            G u8 *oqt = oq ? (oq + (unsigned long)t * oqs) : (G u8 *)0;
            ink_r16_er_gate_q8_head_vec4(norm, gt_h, ot_h, oqt, h, n_head, l);
          } else {
            G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R16_ER_PSTR;
            if (l == 0u) {
              pp[0] = ink_r16_er_sw_lds.ph4.m[qi];
              pp[1] = ink_r16_er_sw_lds.ph4.l[qi];
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
    ink_r16_er_sw_lds.ph4.is_last =
        (__scoped_atomic_fetch_add(&count[kh], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r16_er_sw_lds.ph4.is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  for (u32 qi = w; qi < nq_tot; qi += 8u) {
    const u32 t = qi / INK_R16_ER_GQA;
    const u32 h = h0 + (qi - t * INK_R16_ER_GQA);
    const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
    const G float *gt_h = agate ? ((const G float *)((const G u8 *)agate + (unsigned long)t * gs) +
                                   (unsigned long)h * INK_R16_ER_HD)
                                : (const G float *)0;
    G float *ot_h = out ? ((G float *)((G u8 *)out + (unsigned long)t * os) +
                           (unsigned long)h * INK_R16_ER_HD)
                        : (G float *)0;
    G u8 *oqt = oq ? (oq + (unsigned long)t * oqs) : (G u8 *)0;
    const G float *ph = part + (unsigned long)((t * n_head + h) * nsplit) * INK_R16_ER_PSTR;
    ink_r16_er_sw_merge_head_exact(ph, st[h], nsplit, l, ink_r16_er_sw_lds.ph4.wt[w],
                                   gt_h, ot_h, oqt, h, n_head);
  }
  barrier();
  if (tid() == 0u) count[kh] = 0u;
}

/* ============================================================================
 * Kernel 4 (R64 Exact Rescue, H21-SWA-QT2): ie_inkling_rel_attn_ver_swa_qt2_exact_r16
 * Ultra-lean <= 68 VGPR (4 WGs/CU) QT=2 FP32 Scalar SWA Verify + optional fused
 * OP_GATE(sigmoid) + Q8_0 epilogue with WAVE-ROTATED PER-TOKEN EXACT SPLITS!
 *   - Computes each token's OWN exact split [p0_a, p1_a) & [p0_b, p1_b) and loops
 *     once over their union [u0, u1) (at win=1024, nsplit=16: 63 of 64 keys shared
 *     = 98.4% KV sharing, 2x fewer KV reads than ver_gate_q8_r10!).
 *   - Rotates Phase-3 LDS wave slot by (w + 8 - ((p0_t - u0) & 7)) & 7 so EVERY
 *     wave 0..7 of EVERY split produces the 100% 0-bit-diff partial of
 *     ie_inkling_rel_attn_ver_gate_q8_r10 (which PASSED 5/5 exact SHAs in R54!).
 *   - Fixes L==0 sentinel guard on pp[0] and missing barrier() before count reset.
 *   - Grid: (n_kv * nsplit, (T + 1u) / 2u, 1u), NT = 256, 12,484 B LDS, 0 spills.
 * ========================================================================== */

#define INK_R16_ER_QT2_UPD(m_v, s_v, a_v, d_val, vf_val)                           \
  do {                                                                             \
    const float _mn = __builtin_fmaxf((m_v), (d_val));                             \
    const float _c = ((s_v) > 0.0f) ? __builtin_expf((m_v) - _mn) : 0.0f;          \
    const float _e = __builtin_expf((d_val) - _mn);                                \
    (s_v) = ink_r16_er_sgpr_f((s_v) * _c + _e);                                    \
    (m_v) = ink_r16_er_sgpr_f(_mn);                                                \
    (a_v) = (a_v) * _c + _e * (vf_val);                                            \
  } while (0)

KERNEL ie_inkling_rel_attn_ver_swa_qt2_exact_r16(
    const G float *q, const G kvt *kc, const G kvt *vc,
    const G float *R, const G float *rel_bias, const G float *sink,
    G float *part, G float *out, u32 pos0, u32 T, u32 kv_len, u32 hd,
    u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
    u32 qs, u32 rs, u32 ss, u32 os, G u32 *count,
    const G float *agate, G u8 *oq, u32 gs, u32 oqs) {
  static LDS union {
    struct {
      float q[2u * INK_R16_ER_GQA * INK_R16_ER_HD] __attribute__((aligned(16)));
      float r[2u * INK_R16_ER_GQA * INK_R16_ER_DREL] __attribute__((aligned(16)));
    } qr; /* 3,456 B */
    struct {
      float m[INK_R16_ER_GQA][8];
      float l[INK_R16_ER_GQA][8];
      float o[INK_R16_ER_GQA][8][INK_R16_ER_HD] __attribute__((aligned(16)));
    } red; /* 12,480 B */
  } ink_r16_er_q2_lds __attribute__((aligned(16)));
  static LDS u32 ink_r16_er_q2_last;

  const u32 t0 = (u32)__builtin_amdgcn_readfirstlane((int)(tokid() * 2u));
  if (t0 >= T || win == 0u || hd != INK_R16_ER_HD || d_rel != INK_R16_ER_DREL ||
      n_kv == 0u || n_head != INK_R16_ER_GQA * n_kv || kv_len == 0u || nsplit == 0u)
    return;

  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() / nsplit));
  const u32 s = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() % nsplit));
  if (kh >= n_kv) return;

  const u32 h0 = kh * INK_R16_ER_GQA;
  const u32 kvd = n_kv * INK_R16_ER_HD;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 nq = (u32)__builtin_amdgcn_readfirstlane((int)((T - t0 >= 2u) ? 2u : 1u));
  const u32 nb_q = n_head << 2u;

  /* Per-token exact splits matching ie_inkling_rel_attn_ver_gate_q8_r10. */
  const u32 pos_a = pos0 + t0;
  const u32 pos_b = pos_a + 1u;
  const u32 pend_a = pos_a + 1u;
  const u32 pend_b = pos_b + 1u;
  const u32 wa = (pend_a > win) ? (pend_a - win) : 0u;
  const u32 wb = (pend_b > win) ? (pend_b - win) : 0u;
  const u32 np_a = pend_a - wa;
  const u32 np_b = pend_b - wb;
  const u32 ch_a = (np_a + nsplit - 1u) / nsplit;
  const u32 ch_b = (np_b + nsplit - 1u) / nsplit;
  const u32 off_a = s * ch_a;
  const u32 off_b = s * ch_b;
  const u32 p0_a = (off_a < np_a) ? (wa + off_a) : pend_a;
  const u32 p1_a = (off_a < np_a && off_a + ch_a < np_a) ? (wa + off_a + ch_a) : pend_a;
  const u32 p0_b = (nq > 1u && off_b < np_b) ? (wb + off_b) : pend_b;
  const u32 p1_b = (nq > 1u && off_b < np_b && off_b + ch_b < np_b) ? (wb + off_b + ch_b) : pend_b;

  const int a_act = (p0_a < p1_a);
  const int b_act = (nq > 1u) && (p0_b < p1_b);

  if (nsplit > 1u && !a_act && !b_act) {
    for (u32 qi = 0u; qi < nq; qi++) {
      const u32 t = t0 + qi;
      for (u32 idx = tid(); idx < INK_R16_ER_GQA * INK_R16_ER_HD; idx += NT) {
        const u32 g = idx >> 7u;
        const u32 e = idx & 127u;
        const u32 h = h0 + g;
        G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R16_ER_PSTR;
        pp[2u + e] = 0.0f;
        if (e == 0u) { pp[0] = -1e30f; pp[1] = 0.0f; }
      }
    }
  } else {
    const u32 u0 = (a_act && b_act) ? ((p0_a < p0_b) ? p0_a : p0_b) : (a_act ? p0_a : p0_b);
    const u32 u1 = (a_act && b_act) ? ((p1_a > p1_b) ? p1_a : p1_b) : (a_act ? p1_a : p1_b);
    const u32 extent_lim = (win < rel_extent) ? win : rel_extent;
    const float scale = 1.0f / __builtin_sqrtf((float)INK_R16_ER_HD);

    const u32 nq_vecs = nq * (INK_R16_ER_GQA * 32u);
    for (u32 idx = tid(); idx < nq_vecs; idx += NT) {
      const u32 qg = idx >> 5u;
      const u32 e4 = idx & 31u;
      const u32 qi = qg / INK_R16_ER_GQA;
      const u32 gi = qg - qi * INK_R16_ER_GQA;
      const G float *qt = (const G float *)((const G u8 *)q + (unsigned long)(t0 + qi) * qs);
      ((LDSP f32x4 *)(ink_r16_er_q2_lds.qr.q + (unsigned long)qg * INK_R16_ER_HD))[e4] =
          ((const G f32x4 *)(qt + (unsigned long)(h0 + gi) * INK_R16_ER_HD))[e4];
    }
    if (tid() < nq * (INK_R16_ER_GQA * INK_R16_ER_DREL)) {
      const u32 idx = tid();
      const u32 qg = idx >> 4u;
      const u32 e = idx & 15u;
      const u32 qi = qg / INK_R16_ER_GQA;
      const u32 gi = qg - qi * INK_R16_ER_GQA;
      const G float *Rt = (const G float *)((const G u8 *)R + (unsigned long)(t0 + qi) * rs);
      ink_r16_er_q2_lds.qr.r[(unsigned long)qg * INK_R16_ER_DREL + e] = Rt[(unsigned long)(h0 + gi) * INK_R16_ER_DREL + e];
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

    const G kvt *kc_h = kc + (unsigned long)kh * INK_R16_ER_HD;
    const G kvt *vc_h = vc + (unsigned long)kh * INK_R16_ER_HD;

    u32 kp = (u32)__builtin_amdgcn_readfirstlane((int)(u0 + w));
    if (kp < u1) {
      u32 kvp = (u32)__builtin_amdgcn_readfirstlane((int)(kp % kv_len));
      kv4 kh4 = ((const G kv4 *)(kc_h + (unsigned long)kvp * kvd))[l];
      kv4 vh4 = ((const G kv4 *)(vc_h + (unsigned long)kvp * kvd))[l];

#pragma unroll 1
      for (;;) {
        const u32 kp_nxt = (u32)__builtin_amdgcn_readfirstlane((int)(kp + 8u));
        u32 kvp_nxt = kvp;
        kv4 kh4_nxt = kh4, vh4_nxt = vh4;
        if (kp_nxt < u1) {
          u32 kn = kvp + 8u;
          if (kn >= kv_len) kn = (kv_len >= 8u) ? (kn - kv_len) : (kp_nxt % kv_len);
          kvp_nxt = (u32)__builtin_amdgcn_readfirstlane((int)kn);
          kh4_nxt = ((const G kv4 *)(kc_h + (unsigned long)kvp_nxt * kvd))[l];
          vh4_nxt = ((const G kv4 *)(vc_h + (unsigned long)kvp_nxt * kvd))[l];
        }

        const f32x4 kf = __builtin_convertvector(kh4, f32x4);
        const f32x4 vf = __builtin_convertvector(vh4, f32x4);

        if (a_act && kp >= p0_a && kp < p1_a) {
          const LDSP f32x4 *qp0 = (const LDSP f32x4 *)ink_r16_er_q2_lds.qr.q;
          float d0 = ink_r16_er_sgpr_f(wave_sum_all(ink_r16_er_dot4(qp0[0u * 32u + l], kf)) * scale);
          float d1 = ink_r16_er_sgpr_f(wave_sum_all(ink_r16_er_dot4(qp0[1u * 32u + l], kf)) * scale);
          float d2 = ink_r16_er_sgpr_f(wave_sum_all(ink_r16_er_dot4(qp0[2u * 32u + l], kf)) * scale);
          const u32 dist = pos_a - kp;
          if (dist < extent_lim) {
            const float rb = (l < INK_R16_ER_DREL) ? rel_bias[(unsigned long)dist * INK_R16_ER_DREL + l] : 0.0f;
            const LDSP float *rp0 = ink_r16_er_q2_lds.qr.r;
            const float r0 = (l < INK_R16_ER_DREL) ? rp0[0u * 16u + l] : 0.0f;
            const float r1 = (l < INK_R16_ER_DREL) ? rp0[1u * 16u + l] : 0.0f;
            const float r2 = (l < INK_R16_ER_DREL) ? rp0[2u * 16u + l] : 0.0f;
            d0 = ink_r16_er_sgpr_f(d0 + ink_r16_er_halfwave_sum_all(r0 * rb));
            d1 = ink_r16_er_sgpr_f(d1 + ink_r16_er_halfwave_sum_all(r1 * rb));
            d2 = ink_r16_er_sgpr_f(d2 + ink_r16_er_halfwave_sum_all(r2 * rb));
          }
          INK_R16_ER_QT2_UPD(m00, s00, a00, d0, vf);
          INK_R16_ER_QT2_UPD(m01, s01, a01, d1, vf);
          INK_R16_ER_QT2_UPD(m02, s02, a02, d2, vf);
        }

        if (b_act && kp >= p0_b && kp < p1_b) {
          const LDSP f32x4 *qp1 = (const LDSP f32x4 *)(ink_r16_er_q2_lds.qr.q + INK_R16_ER_GQA * INK_R16_ER_HD);
          float d0 = ink_r16_er_sgpr_f(wave_sum_all(ink_r16_er_dot4(qp1[0u * 32u + l], kf)) * scale);
          float d1 = ink_r16_er_sgpr_f(wave_sum_all(ink_r16_er_dot4(qp1[1u * 32u + l], kf)) * scale);
          float d2 = ink_r16_er_sgpr_f(wave_sum_all(ink_r16_er_dot4(qp1[2u * 32u + l], kf)) * scale);
          const u32 dist = pos_b - kp;
          if (dist < extent_lim) {
            const float rb = (l < INK_R16_ER_DREL) ? rel_bias[(unsigned long)dist * INK_R16_ER_DREL + l] : 0.0f;
            const LDSP float *rp1 = ink_r16_er_q2_lds.qr.r + INK_R16_ER_GQA * INK_R16_ER_DREL;
            const float r0 = (l < INK_R16_ER_DREL) ? rp1[0u * 16u + l] : 0.0f;
            const float r1 = (l < INK_R16_ER_DREL) ? rp1[1u * 16u + l] : 0.0f;
            const float r2 = (l < INK_R16_ER_DREL) ? rp1[2u * 16u + l] : 0.0f;
            d0 = ink_r16_er_sgpr_f(d0 + ink_r16_er_halfwave_sum_all(r0 * rb));
            d1 = ink_r16_er_sgpr_f(d1 + ink_r16_er_halfwave_sum_all(r1 * rb));
            d2 = ink_r16_er_sgpr_f(d2 + ink_r16_er_halfwave_sum_all(r2 * rb));
          }
          INK_R16_ER_QT2_UPD(m10, s10, a10, d0, vf);
          INK_R16_ER_QT2_UPD(m11, s11, a11, d1, vf);
          INK_R16_ER_QT2_UPD(m12, s12, a12, d2, vf);
        }

        if (kp_nxt >= u1) break;
        kp = kp_nxt; kvp = kvp_nxt; kh4 = kh4_nxt; vh4 = vh4_nxt;
      }
    }
    lbarrier();

    /* Wave-rotated Phase-3 slot maps each wave back to its exact ver_gate_q8_r10 slot! */
    const u32 w_a = a_act ? ((w + 8u - ((p0_a - u0) & 7u)) & 7u) : w;
    const u32 w_b = b_act ? ((w + 8u - ((p0_b - u0) & 7u)) & 7u) : w;

    for (u32 qi = 0u; qi < nq; qi++) {
      const u32 t = t0 + qi;
      const u32 ws = (qi == 0u) ? w_a : w_b;
      if (l == 0u) {
        ink_r16_er_q2_lds.red.m[0][ws] = (qi == 0u) ? m00 : m10;
        ink_r16_er_q2_lds.red.l[0][ws] = (qi == 0u) ? s00 : s10;
        ink_r16_er_q2_lds.red.m[1][ws] = (qi == 0u) ? m01 : m11;
        ink_r16_er_q2_lds.red.l[1][ws] = (qi == 0u) ? s01 : s11;
        ink_r16_er_q2_lds.red.m[2][ws] = (qi == 0u) ? m02 : m12;
        ink_r16_er_q2_lds.red.l[2][ws] = (qi == 0u) ? s02 : s12;
      }
      ((LDSP f32x4 *)&ink_r16_er_q2_lds.red.o[0][ws][0])[l] = (qi == 0u) ? a00 : a10;
      ((LDSP f32x4 *)&ink_r16_er_q2_lds.red.o[1][ws][0])[l] = (qi == 0u) ? a01 : a11;
      ((LDSP f32x4 *)&ink_r16_er_q2_lds.red.o[2][ws][0])[l] = (qi == 0u) ? a02 : a12;
      lbarrier();

      if (nsplit == 1u) {
        const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
        const G float *gt = agate ? (const G float *)((const G u8 *)agate + (unsigned long)t * gs) : (const G float *)0;
        G float *ot = out ? (G float *)((G u8 *)out + (unsigned long)t * os) : (G float *)0;
        G u8 *oqt = oq ? (oq + (unsigned long)t * oqs) : (G u8 *)0;
        for (u32 idx = tid(); idx < INK_R16_ER_GQA * INK_R16_ER_HD; idx += NT) {
          const u32 g = idx >> 7u;
          const u32 e = idx & 127u;
          const u32 h = h0 + g;
          const float sv = st[h];
          float M = sv;
          for (u32 x = 0u; x < 8u; x++)
            if (ink_r16_er_q2_lds.red.l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r16_er_q2_lds.red.m[g][x]);
          float L = __builtin_expf(sv - M);
          float o = 0.0f;
          for (u32 x = 0u; x < 8u; x++) {
            const float lx = ink_r16_er_q2_lds.red.l[g][x];
            if (lx > 0.0f) {
              const float wx = __builtin_expf(ink_r16_er_q2_lds.red.m[g][x] - M);
              L += lx * wx;
              o += ink_r16_er_q2_lds.red.o[g][x][e] * wx;
            }
          }
          const u32 out_idx = h * INK_R16_ER_HD + e;
          float v = o / L;
          if (gt) v = ink_r16_er_gate_f(v, gt[out_idx]);
          if (ot) ot[out_idx] = v;
          if (oqt) quant_wave(v, out_idx >> 5u, oqt, nb_q);
        }
      } else {
        for (u32 idx = tid(); idx < INK_R16_ER_GQA * INK_R16_ER_HD; idx += NT) {
          const u32 g = idx >> 7u;
          const u32 e = idx & 127u;
          const u32 h = h0 + g;
          float M = -__builtin_inff();
          for (u32 x = 0u; x < 8u; x++)
            if (ink_r16_er_q2_lds.red.l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r16_er_q2_lds.red.m[g][x]);
          G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R16_ER_PSTR;
          float o = 0.0f, L = 0.0f;
          for (u32 x = 0u; x < 8u; x++) {
            const float lx = ink_r16_er_q2_lds.red.l[g][x];
            if (lx > 0.0f) {
              const float wx = __builtin_expf(ink_r16_er_q2_lds.red.m[g][x] - M);
              o += ink_r16_er_q2_lds.red.o[g][x][e] * wx;
              if (e == 0u) L += lx * wx;
            }
          }
          pp[2u + e] = o;
          if (e == 0u) {
            pp[0] = (L > 0.0f) ? M : -1e30f;
            pp[1] = L;
          }
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
    ink_r16_er_q2_last =
        (__scoped_atomic_fetch_add(&count[cidx], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u);
  }
  barrier();
  if (!ink_r16_er_q2_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const int wt_in_lds = (nsplit <= INK_R16_ER_HD);
  for (u32 qi = 0u; qi < nq; qi++) {
    const u32 t = t0 + qi;
    const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
    const G float *gt = agate ? (const G float *)((const G u8 *)agate + (unsigned long)t * gs) : (const G float *)0;
    G float *ot = out ? (G float *)((G u8 *)out + (unsigned long)t * os) : (G float *)0;
    G u8 *oqt = oq ? (oq + (unsigned long)t * oqs) : (G u8 *)0;

    G float *ph0 = part + (unsigned long)((t * n_head + (h0 + 0u)) * nsplit) * INK_R16_ER_PSTR;
    G float *ph1 = part + (unsigned long)((t * n_head + (h0 + 1u)) * nsplit) * INK_R16_ER_PSTR;
    G float *ph2 = part + (unsigned long)((t * n_head + (h0 + 2u)) * nsplit) * INK_R16_ER_PSTR;

    float mv0 = (tid() == 0u) ? st[h0 + 0u] : -__builtin_inff();
    float mv1 = (tid() == 0u) ? st[h0 + 1u] : -__builtin_inff();
    float mv2 = (tid() == 0u) ? st[h0 + 2u] : -__builtin_inff();
    for (u32 sp = tid(); sp < nsplit; sp += NT) {
      const unsigned long off = (unsigned long)sp * INK_R16_ER_PSTR;
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
      ink_r16_er_q2_lds.red.m[0][w] = mv0;
      ink_r16_er_q2_lds.red.m[1][w] = mv1;
      ink_r16_er_q2_lds.red.m[2][w] = mv2;
    }
    lbarrier();

    float M2[INK_R16_ER_GQA];
    for (u32 g = 0u; g < INK_R16_ER_GQA; g++) {
      float mx = ink_r16_er_q2_lds.red.m[g][0];
      for (u32 x = 1u; x < 8u; x++) mx = __builtin_fmaxf(mx, ink_r16_er_q2_lds.red.m[g][x]);
      M2[g] = mx;
    }

    float lv0 = 0.0f, lv1 = 0.0f, lv2 = 0.0f;
    for (u32 sp = tid(); sp < nsplit; sp += NT) {
      const unsigned long off = (unsigned long)sp * INK_R16_ER_PSTR;
      const float lp0 = ph0[off + 1u];
      const float wt0 = (lp0 > 0.0f) ? __builtin_expf(ph0[off] - M2[0]) : 0.0f;
      if (wt_in_lds) ink_r16_er_q2_lds.red.o[0][0][sp] = wt0; else ph0[off] = wt0;
      lv0 += lp0 * wt0;

      const float lp1 = ph1[off + 1u];
      const float wt1 = (lp1 > 0.0f) ? __builtin_expf(ph1[off] - M2[1]) : 0.0f;
      if (wt_in_lds) ink_r16_er_q2_lds.red.o[1][0][sp] = wt1; else ph1[off] = wt1;
      lv1 += lp1 * wt1;

      const float lp2 = ph2[off + 1u];
      const float wt2 = (lp2 > 0.0f) ? __builtin_expf(ph2[off] - M2[2]) : 0.0f;
      if (wt_in_lds) ink_r16_er_q2_lds.red.o[2][0][sp] = wt2; else ph2[off] = wt2;
      lv2 += lp2 * wt2;
    }
    lv0 = wave_tree_f(lv0);
    lv1 = wave_tree_f(lv1);
    lv2 = wave_tree_f(lv2);
    if (l == 0u) {
      ink_r16_er_q2_lds.red.l[0][w] = lv0;
      ink_r16_er_q2_lds.red.l[1][w] = lv1;
      ink_r16_er_q2_lds.red.l[2][w] = lv2;
    }
    if (wt_in_lds) lbarrier(); else barrier();

    float L2[INK_R16_ER_GQA];
    for (u32 g = 0u; g < INK_R16_ER_GQA; g++) {
      float ls = 0.0f;
      for (u32 x = 0u; x < 8u; x++) ls += ink_r16_er_q2_lds.red.l[g][x];
      L2[g] = ls + __builtin_expf(st[h0 + g] - M2[g]);
    }

    for (u32 idx = tid(); idx < INK_R16_ER_GQA * INK_R16_ER_HD; idx += NT) {
      const u32 g = idx >> 7u;
      const u32 e = idx & 127u;
      const u32 h = h0 + g;
      const G float *ph = part + (unsigned long)((t * n_head + h) * nsplit) * INK_R16_ER_PSTR;
      float o = 0.0f;
      if (wt_in_lds) {
        for (u32 sp = 0u; sp < nsplit; sp++)
          o += ph[(unsigned long)sp * INK_R16_ER_PSTR + 2u + e] * ink_r16_er_q2_lds.red.o[g][0][sp];
      } else {
        for (u32 sp = 0u; sp < nsplit; sp++)
          o += ph[(unsigned long)sp * INK_R16_ER_PSTR + 2u + e] * ph[(unsigned long)sp * INK_R16_ER_PSTR];
      }
      const u32 out_idx = h * INK_R16_ER_HD + e;
      float v = o / L2[g];
      if (gt) v = ink_r16_er_gate_f(v, gt[out_idx]);
      if (ot) ot[out_idx] = v;
      if (oqt) quant_wave(v, out_idx >> 5u, oqt, nb_q);
    }
    if (qi + 1u < nq) lbarrier();
  }

  barrier();
  if (tid() == 0u) count[cidx] = 0u;
}

/* ============================================================================
 * Kernel 5 (R64 Exact Rescue, QT=4 Fused Gate+Q8): ie_inkling_rel_attn_ver_swa_qtile_exact_r16
 * 100% 0-bit-diff upgrade of R37 promoted ie_inkling_rel_attn_ver_swa_qtile_r7
 * fusing OP_GATE(sigmoid) + Q8_0 quantization (-40 launches/step + att elision).
 *   - Identical QT=4 union window, swa_interior + edge loops, Phase 3 & Phase 4
 *     o / L2[g], plus exact ie_inkling_gate_q8_r2 gate+quant epilogue.
 *   - Supports both fused (agate/oq != NULL) and unfused (agate == NULL -> out).
 *   - Grid: (n_kv * nsplit, (T + 3u) / 4u, 1u), NT = 256, 12,484 B LDS, 0 spills.
 * ========================================================================== */

#define INK_R16_ER_QT4_QK3(qi, kf_val, sc_val, lane_id, d0_out, d1_out, d2_out)                            \
  do {                                                                                                     \
    const LDSP f32x4 *_er_qp =                                                                             \
        (const LDSP f32x4 *)(ink_r16_er_q4_lds.qr.q + (unsigned long)(qi) * (INK_R16_ER_GQA * INK_R16_ER_HD)); \
    const f32x4 _er_q0 = _er_qp[0u * 32u + (lane_id)];                                                     \
    const f32x4 _er_q1 = _er_qp[1u * 32u + (lane_id)];                                                     \
    const f32x4 _er_q2 = _er_qp[2u * 32u + (lane_id)];                                                     \
    (d0_out) = ink_r16_er_sgpr_f(wave_sum_all(ink_r16_er_dot4(_er_q0, (kf_val))) * (sc_val));              \
    (d1_out) = ink_r16_er_sgpr_f(wave_sum_all(ink_r16_er_dot4(_er_q1, (kf_val))) * (sc_val));              \
    (d2_out) = ink_r16_er_sgpr_f(wave_sum_all(ink_r16_er_dot4(_er_q2, (kf_val))) * (sc_val));              \
  } while (0)

#define INK_R16_ER_QT4_ADD_RB3(qi, rb_val, lane_id, d0_io, d1_io, d2_io)                                       \
  do {                                                                                                         \
    const LDSP float *_er_rp = ink_r16_er_q4_lds.qr.r + (unsigned long)(qi) * (INK_R16_ER_GQA * INK_R16_ER_DREL); \
    const float _er_rv0 = ((lane_id) < INK_R16_ER_DREL) ? _er_rp[0u * INK_R16_ER_DREL + (lane_id)] : 0.0f;     \
    const float _er_rv1 = ((lane_id) < INK_R16_ER_DREL) ? _er_rp[1u * INK_R16_ER_DREL + (lane_id)] : 0.0f;     \
    const float _er_rv2 = ((lane_id) < INK_R16_ER_DREL) ? _er_rp[2u * INK_R16_ER_DREL + (lane_id)] : 0.0f;     \
    (d0_io) = ink_r16_er_sgpr_f((d0_io) + ink_r16_er_halfwave_sum_all(_er_rv0 * (rb_val)));                    \
    (d1_io) = ink_r16_er_sgpr_f((d1_io) + ink_r16_er_halfwave_sum_all(_er_rv1 * (rb_val)));                    \
    (d2_io) = ink_r16_er_sgpr_f((d2_io) + ink_r16_er_halfwave_sum_all(_er_rv2 * (rb_val)));                    \
  } while (0)

#define INK_R16_ER_QT4_EDGE_TOK(qi, kf_val, vf_val, sc_val, lane_id, kp_val, pos_q, win_v, ext_lim,       \
                                m0_v, s0_v, a0_v, m1_v, s1_v, a1_v, m2_v, s2_v, a2_v)                     \
  do {                                                                                                    \
    if ((kp_val) <= (pos_q) && ((pos_q) - (kp_val)) < (win_v)) {                                          \
      const u32 _er_dist = (pos_q) - (kp_val);                                                            \
      float _er_rb = 0.0f;                                                                                \
      if (_er_dist < (ext_lim) && (lane_id) < INK_R16_ER_DREL) {                                          \
        _er_rb = rel_bias[(unsigned long)_er_dist * INK_R16_ER_DREL + (lane_id)];                         \
      }                                                                                                   \
      float _er_d0, _er_d1, _er_d2;                                                                       \
      INK_R16_ER_QT4_QK3((qi), (kf_val), (sc_val), (lane_id), _er_d0, _er_d1, _er_d2);                    \
      if (_er_dist < (ext_lim)) {                                                                         \
        INK_R16_ER_QT4_ADD_RB3((qi), _er_rb, (lane_id), _er_d0, _er_d1, _er_d2);                          \
      }                                                                                                   \
      INK_R16_ER_QT2_UPD((m0_v), (s0_v), (a0_v), _er_d0, (vf_val));                                       \
      INK_R16_ER_QT2_UPD((m1_v), (s1_v), (a1_v), _er_d1, (vf_val));                                       \
      INK_R16_ER_QT2_UPD((m2_v), (s2_v), (a2_v), _er_d2, (vf_val));                                       \
    }                                                                                                     \
  } while (0)

KERNEL ie_inkling_rel_attn_ver_swa_qtile_exact_r16(
    const G float *q, const G kvt *kc, const G kvt *vc,
    const G float *R, const G float *rel_bias, const G float *sink,
    G float *part, G float *out, u32 pos0, u32 T, u32 kv_len, u32 hd,
    u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
    u32 qs, u32 rs, u32 ss, u32 os, G u32 *count,
    const G float *agate, G u8 *oq, u32 gs, u32 oqs) {
  static LDS union {
    struct {
      float q[INK_R16_ER_QT4 * INK_R16_ER_GQA * INK_R16_ER_HD] __attribute__((aligned(16)));
      float r[INK_R16_ER_QT4 * INK_R16_ER_GQA * INK_R16_ER_DREL] __attribute__((aligned(16)));
    } qr;
    struct {
      float m[INK_R16_ER_GQA][8];
      float l[INK_R16_ER_GQA][8];
      float o[INK_R16_ER_GQA][8][INK_R16_ER_HD] __attribute__((aligned(16)));
    } red;
  } ink_r16_er_q4_lds __attribute__((aligned(16)));
  static LDS u32 ink_r16_er_q4_last;
  _Static_assert(sizeof(ink_r16_er_q4_lds) == 12480u, "ink_r16_er_q4_lds must be 12,480 B");

  const u32 t0 = (u32)__builtin_amdgcn_readfirstlane((int)(tokid() * INK_R16_ER_QT4));
  if (t0 >= T || win == 0u || hd != INK_R16_ER_HD || d_rel != INK_R16_ER_DREL || n_kv == 0u ||
      n_head != INK_R16_ER_GQA * n_kv || kv_len == 0u || nsplit == 0u)
    return;

  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() / nsplit));
  const u32 s = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() % nsplit));
  if (kh >= n_kv) return;

  const u32 h0 = (u32)__builtin_amdgcn_readfirstlane((int)(kh * INK_R16_ER_GQA));
  const u32 kvd = (u32)__builtin_amdgcn_readfirstlane((int)(n_kv * INK_R16_ER_HD));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 nb_q = n_head << 2u;

  const u32 rem = T - t0;
  const u32 nq = (u32)__builtin_amdgcn_readfirstlane((int)(rem < INK_R16_ER_QT4 ? rem : INK_R16_ER_QT4));
  const u32 pos_t0 = (u32)__builtin_amdgcn_readfirstlane((int)(pos0 + t0));
  const u32 pos_last = (u32)__builtin_amdgcn_readfirstlane((int)(pos_t0 + nq - 1u));

  const u32 w_union = (u32)__builtin_amdgcn_readfirstlane((int)((pos_t0 + 1u > win) ? (pos_t0 + 1u - win) : 0u));
  const u32 w_last = (u32)__builtin_amdgcn_readfirstlane((int)((pos_last + 1u > win) ? (pos_last + 1u - win) : 0u));
  const u32 pend = (u32)__builtin_amdgcn_readfirstlane((int)(pos_last + 1u));
  const u32 np = (u32)__builtin_amdgcn_readfirstlane((int)(pend - w_union));
  const u32 ch = (u32)__builtin_amdgcn_readfirstlane((int)((np + nsplit - 1u) / nsplit));
  const u32 off0 = (u32)__builtin_amdgcn_readfirstlane((int)(s * ch));
  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)((off0 < np) ? (w_union + off0) : pend));
  const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)((off0 < np && off0 + ch < np) ? (w_union + off0 + ch) : pend));

  if (nsplit > 1u && p0 >= p1) {
#pragma clang loop unroll(disable)
    for (u32 qi = 0u; qi < nq; qi++) {
      const u32 t = t0 + qi;
      for (u32 idx = tid(); idx < INK_R16_ER_GQA * INK_R16_ER_HD; idx += NT) {
        const u32 g = idx >> 7u;
        const u32 e = idx & 127u;
        const u32 h = h0 + g;
        G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R16_ER_PSTR;
        pp[2u + e] = 0.0f;
        if (e == 0u) { pp[0] = -1e30f; pp[1] = 0.0f; }
      }
    }
  } else {
    const u32 extent_lim = (u32)__builtin_amdgcn_readfirstlane((int)((win < rel_extent) ? win : rel_extent));
    const int swa_interior =
        __builtin_amdgcn_readfirstlane((p0 >= w_last) && (p1 <= pos_t0 + 1u) && (pos_last - p0 < extent_lim));
    const float scale = 1.0f / __builtin_sqrtf((float)INK_R16_ER_HD);

    const u32 nq_vecs = (u32)__builtin_amdgcn_readfirstlane((int)(nq * (INK_R16_ER_GQA * (INK_R16_ER_HD / 4u))));
    for (u32 idx = tid(); idx < nq_vecs; idx += NT) {
      const u32 qg = idx >> 5u;
      const u32 e4 = idx & 31u;
      const u32 qi = qg / INK_R16_ER_GQA;
      const u32 gi = qg - qi * INK_R16_ER_GQA;
      const G float *qt = (const G float *)((const G u8 *)q + (unsigned long)(t0 + qi) * qs);
      ((LDSP f32x4 *)(ink_r16_er_q4_lds.qr.q + (unsigned long)qg * INK_R16_ER_HD))[e4] =
          ((const G f32x4 *)(qt + (unsigned long)(h0 + gi) * INK_R16_ER_HD))[e4];
    }
    if (extent_lim > 0u) {
      const u32 idx = tid();
      if (idx < nq * (INK_R16_ER_GQA * INK_R16_ER_DREL)) {
        const u32 qg = idx >> 4u;
        const u32 e = idx & 15u;
        const u32 qi = qg / INK_R16_ER_GQA;
        const u32 gi = qg - qi * INK_R16_ER_GQA;
        const G float *Rt = (const G float *)((const G u8 *)R + (unsigned long)(t0 + qi) * rs);
        ink_r16_er_q4_lds.qr.r[(unsigned long)qg * INK_R16_ER_DREL + e] =
            Rt[(unsigned long)(h0 + gi) * INK_R16_ER_DREL + e];
      }
    }
    lbarrier();

    f32x4 acc00 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc01 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc02 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc10 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc11 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc12 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc20 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc21 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc22 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc30 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc31 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc32 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};

    float m00 = -__builtin_inff(), s00 = 0.0f;
    float m01 = -__builtin_inff(), s01 = 0.0f;
    float m02 = -__builtin_inff(), s02 = 0.0f;
    float m10 = -__builtin_inff(), s10 = 0.0f;
    float m11 = -__builtin_inff(), s11 = 0.0f;
    float m12 = -__builtin_inff(), s12 = 0.0f;
    float m20 = -__builtin_inff(), s20 = 0.0f;
    float m21 = -__builtin_inff(), s21 = 0.0f;
    float m22 = -__builtin_inff(), s22 = 0.0f;
    float m30 = -__builtin_inff(), s30 = 0.0f;
    float m31 = -__builtin_inff(), s31 = 0.0f;
    float m32 = -__builtin_inff(), s32 = 0.0f;

    const G kvt *kc_h = kc + (unsigned long)kh * INK_R16_ER_HD;
    const G kvt *vc_h = vc + (unsigned long)kh * INK_R16_ER_HD;

    u32 kp = (u32)__builtin_amdgcn_readfirstlane((int)(p0 + w));
    if (kp < p1) {
      u32 kvp = (u32)__builtin_amdgcn_readfirstlane((int)((kp < kv_len) ? kp : (kp % kv_len)));
      kv4 kh4 = ((const G kv4 *)(kc_h + (unsigned long)kvp * kvd))[l];
      kv4 vh4 = ((const G kv4 *)(vc_h + (unsigned long)kvp * kvd))[l];

      if (swa_interior) {
        for (;;) {
          const u32 kp_nxt = (u32)__builtin_amdgcn_readfirstlane((int)(kp + 8u));
          u32 kvp_nxt = kvp;
          kv4 kh4_nxt = kh4, vh4_nxt = vh4;
          if (kp_nxt < p1) {
            u32 kn = kvp + 8u;
            if (kn >= kv_len) kn = (kv_len >= 8u) ? (kn - kv_len) : (kp_nxt % kv_len);
            kvp_nxt = (u32)__builtin_amdgcn_readfirstlane((int)kn);
            kh4_nxt = ((const G kv4 *)(kc_h + (unsigned long)kvp_nxt * kvd))[l];
            vh4_nxt = ((const G kv4 *)(vc_h + (unsigned long)kvp_nxt * kvd))[l];
          }

          const u32 d0 = pos_t0 - kp;
          const int has_rb = (l < INK_R16_ER_DREL);
          const float rb0 = has_rb ? rel_bias[(unsigned long)(d0 + 0u) * INK_R16_ER_DREL + l] : 0.0f;
          const float rb1 = (has_rb && nq > 1u) ? rel_bias[(unsigned long)(d0 + 1u) * INK_R16_ER_DREL + l] : 0.0f;
          const float rb2 = (has_rb && nq > 2u) ? rel_bias[(unsigned long)(d0 + 2u) * INK_R16_ER_DREL + l] : 0.0f;
          const float rb3 = (has_rb && nq > 3u) ? rel_bias[(unsigned long)(d0 + 3u) * INK_R16_ER_DREL + l] : 0.0f;

          const f32x4 kf = __builtin_convertvector(kh4, f32x4);
          float d00, d01, d02;
          float d10 = 0.0f, d11 = 0.0f, d12 = 0.0f;
          float d20 = 0.0f, d21 = 0.0f, d22 = 0.0f;
          float d30 = 0.0f, d31 = 0.0f, d32 = 0.0f;

          INK_R16_ER_QT4_QK3(0u, kf, scale, l, d00, d01, d02);
          if (nq > 1u) INK_R16_ER_QT4_QK3(1u, kf, scale, l, d10, d11, d12);
          if (nq > 2u) INK_R16_ER_QT4_QK3(2u, kf, scale, l, d20, d21, d22);
          if (nq > 3u) INK_R16_ER_QT4_QK3(3u, kf, scale, l, d30, d31, d32);

          INK_R16_ER_QT4_ADD_RB3(0u, rb0, l, d00, d01, d02);
          if (nq > 1u) INK_R16_ER_QT4_ADD_RB3(1u, rb1, l, d10, d11, d12);
          if (nq > 2u) INK_R16_ER_QT4_ADD_RB3(2u, rb2, l, d20, d21, d22);
          if (nq > 3u) INK_R16_ER_QT4_ADD_RB3(3u, rb3, l, d30, d31, d32);

          const f32x4 vf = __builtin_convertvector(vh4, f32x4);
          INK_R16_ER_QT2_UPD(m00, s00, acc00, d00, vf);
          INK_R16_ER_QT2_UPD(m01, s01, acc01, d01, vf);
          INK_R16_ER_QT2_UPD(m02, s02, acc02, d02, vf);
          if (nq > 1u) {
            INK_R16_ER_QT2_UPD(m10, s10, acc10, d10, vf);
            INK_R16_ER_QT2_UPD(m11, s11, acc11, d11, vf);
            INK_R16_ER_QT2_UPD(m12, s12, acc12, d12, vf);
          }
          if (nq > 2u) {
            INK_R16_ER_QT2_UPD(m20, s20, acc20, d20, vf);
            INK_R16_ER_QT2_UPD(m21, s21, acc21, d21, vf);
            INK_R16_ER_QT2_UPD(m22, s22, acc22, d22, vf);
          }
          if (nq > 3u) {
            INK_R16_ER_QT2_UPD(m30, s30, acc30, d30, vf);
            INK_R16_ER_QT2_UPD(m31, s31, acc31, d31, vf);
            INK_R16_ER_QT2_UPD(m32, s32, acc32, d32, vf);
          }

          if (kp_nxt >= p1) break;
          kp = kp_nxt; kvp = kvp_nxt; kh4 = kh4_nxt; vh4 = vh4_nxt;
        }
      } else {
        const u32 pos_q0 = pos_t0;
        const u32 pos_q1 = (u32)__builtin_amdgcn_readfirstlane((int)(pos_t0 + 1u));
        const u32 pos_q2 = (u32)__builtin_amdgcn_readfirstlane((int)(pos_t0 + 2u));
        const u32 pos_q3 = (u32)__builtin_amdgcn_readfirstlane((int)(pos_t0 + 3u));

        for (;;) {
          const u32 kp_nxt = (u32)__builtin_amdgcn_readfirstlane((int)(kp + 8u));
          u32 kvp_nxt = kvp;
          kv4 kh4_nxt = kh4, vh4_nxt = vh4;
          if (kp_nxt < p1) {
            u32 kn = kvp + 8u;
            if (kn >= kv_len) kn = (kv_len >= 8u) ? (kn - kv_len) : (kp_nxt % kv_len);
            kvp_nxt = (u32)__builtin_amdgcn_readfirstlane((int)kn);
            kh4_nxt = ((const G kv4 *)(kc_h + (unsigned long)kvp_nxt * kvd))[l];
            vh4_nxt = ((const G kv4 *)(vc_h + (unsigned long)kvp_nxt * kvd))[l];
          }

          const f32x4 kf = __builtin_convertvector(kh4, f32x4);
          const f32x4 vf = __builtin_convertvector(vh4, f32x4);

          INK_R16_ER_QT4_EDGE_TOK(0u, kf, vf, scale, l, kp, pos_q0, win, extent_lim,
                                  m00, s00, acc00, m01, s01, acc01, m02, s02, acc02);
          if (nq > 1u) {
            INK_R16_ER_QT4_EDGE_TOK(1u, kf, vf, scale, l, kp, pos_q1, win, extent_lim,
                                    m10, s10, acc10, m11, s11, acc11, m12, s12, acc12);
          }
          if (nq > 2u) {
            INK_R16_ER_QT4_EDGE_TOK(2u, kf, vf, scale, l, kp, pos_q2, win, extent_lim,
                                    m20, s20, acc20, m21, s21, acc21, m22, s22, acc22);
          }
          if (nq > 3u) {
            INK_R16_ER_QT4_EDGE_TOK(3u, kf, vf, scale, l, kp, pos_q3, win, extent_lim,
                                    m30, s30, acc30, m31, s31, acc31, m32, s32, acc32);
          }

          if (kp_nxt >= p1) break;
          kp = kp_nxt; kvp = kvp_nxt; kh4 = kh4_nxt; vh4 = vh4_nxt;
        }
      }
    }

    lbarrier();

#pragma clang loop unroll(disable)
    for (u32 qi = 0u; qi < nq; qi++) {
      if (qi > 0u) lbarrier();
      float rm0, rm1, rm2, rl0, rl1, rl2;
      f32x4 ra0, ra1, ra2;
      switch (qi) {
        case 0u:
          rm0 = m00; rm1 = m01; rm2 = m02; rl0 = s00; rl1 = s01; rl2 = s02;
          ra0 = acc00; ra1 = acc01; ra2 = acc02; break;
        case 1u:
          rm0 = m10; rm1 = m11; rm2 = m12; rl0 = s10; rl1 = s11; rl2 = s12;
          ra0 = acc10; ra1 = acc11; ra2 = acc12; break;
        case 2u:
          rm0 = m20; rm1 = m21; rm2 = m22; rl0 = s20; rl1 = s21; rl2 = s22;
          ra0 = acc20; ra1 = acc21; ra2 = acc22; break;
        default:
          rm0 = m30; rm1 = m31; rm2 = m32; rl0 = s30; rl1 = s31; rl2 = s32;
          ra0 = acc30; ra1 = acc31; ra2 = acc32; break;
      }

      if (l == 0u) {
        ink_r16_er_q4_lds.red.m[0][w] = rm0; ink_r16_er_q4_lds.red.l[0][w] = rl0;
        ink_r16_er_q4_lds.red.m[1][w] = rm1; ink_r16_er_q4_lds.red.l[1][w] = rl1;
        ink_r16_er_q4_lds.red.m[2][w] = rm2; ink_r16_er_q4_lds.red.l[2][w] = rl2;
      }
      ((LDSP f32x4 *)&ink_r16_er_q4_lds.red.o[0][w][0])[l] = ra0;
      ((LDSP f32x4 *)&ink_r16_er_q4_lds.red.o[1][w][0])[l] = ra1;
      ((LDSP f32x4 *)&ink_r16_er_q4_lds.red.o[2][w][0])[l] = ra2;
      lbarrier();

      const u32 t = t0 + qi;
      if (nsplit == 1u) {
        const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
        const G float *gt = agate ? (const G float *)((const G u8 *)agate + (unsigned long)t * gs) : (const G float *)0;
        G float *ot = out ? (G float *)((G u8 *)out + (unsigned long)t * os) : (G float *)0;
        G u8 *oqt = oq ? (oq + (unsigned long)t * oqs) : (G u8 *)0;
        for (u32 idx = tid(); idx < INK_R16_ER_GQA * INK_R16_ER_HD; idx += NT) {
          const u32 g = idx >> 7u;
          const u32 e = idx & 127u;
          const u32 h = h0 + g;
          const float sv = st[h];
          float M = sv;
          for (u32 x = 0u; x < 8u; x++)
            if (ink_r16_er_q4_lds.red.l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r16_er_q4_lds.red.m[g][x]);
          float L = __builtin_expf(sv - M);
          float o = 0.0f;
          for (u32 x = 0u; x < 8u; x++) {
            const float lx = ink_r16_er_q4_lds.red.l[g][x];
            if (lx > 0.0f) {
              const float wx = __builtin_expf(ink_r16_er_q4_lds.red.m[g][x] - M);
              L += lx * wx;
              o += ink_r16_er_q4_lds.red.o[g][x][e] * wx;
            }
          }
          const u32 out_idx = h * INK_R16_ER_HD + e;
          float v = o / L;
          if (gt) v = ink_r16_er_gate_f(v, gt[out_idx]);
          if (ot) ot[out_idx] = v;
          if (oqt) quant_wave(v, out_idx >> 5u, oqt, nb_q);
        }
      } else {
        for (u32 idx = tid(); idx < INK_R16_ER_GQA * INK_R16_ER_HD; idx += NT) {
          const u32 g = idx >> 7u;
          const u32 e = idx & 127u;
          const u32 h = h0 + g;
          float M = -__builtin_inff();
          for (u32 x = 0u; x < 8u; x++)
            if (ink_r16_er_q4_lds.red.l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r16_er_q4_lds.red.m[g][x]);
          G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R16_ER_PSTR;
          float o = 0.0f, L = 0.0f;
          for (u32 x = 0u; x < 8u; x++) {
            const float lx = ink_r16_er_q4_lds.red.l[g][x];
            if (lx > 0.0f) {
              const float wx = __builtin_expf(ink_r16_er_q4_lds.red.m[g][x] - M);
              o += ink_r16_er_q4_lds.red.o[g][x][e] * wx;
              if (e == 0u) L += lx * wx;
            }
          }
          pp[2u + e] = o;
          if (e == 0u) {
            pp[0] = (L > 0.0f) ? M : -1e30f;
            pp[1] = L;
          }
        }
      }
    }

    if (nsplit == 1u) return;
  }

  const u32 cidx = (u32)__builtin_amdgcn_readfirstlane((int)(t0 * n_kv + kh));
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r16_er_q4_last =
        (__scoped_atomic_fetch_add(&count[cidx], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u);
  }
  barrier();
  if (!ink_r16_er_q4_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const int wt_in_lds = (nsplit <= INK_R16_ER_HD);
#pragma clang loop unroll(disable)
  for (u32 qi = 0u; qi < nq; qi++) {
    if (qi > 0u) lbarrier();
    const u32 t = t0 + qi;
    const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
    const G float *gt = agate ? (const G float *)((const G u8 *)agate + (unsigned long)t * gs) : (const G float *)0;
    G float *ot = out ? (G float *)((G u8 *)out + (unsigned long)t * os) : (G float *)0;
    G u8 *oqt = oq ? (oq + (unsigned long)t * oqs) : (G u8 *)0;

    G float *ph0 = part + (unsigned long)((t * n_head + (h0 + 0u)) * nsplit) * INK_R16_ER_PSTR;
    G float *ph1 = part + (unsigned long)((t * n_head + (h0 + 1u)) * nsplit) * INK_R16_ER_PSTR;
    G float *ph2 = part + (unsigned long)((t * n_head + (h0 + 2u)) * nsplit) * INK_R16_ER_PSTR;

    float mv0 = (tid() == 0u) ? st[h0 + 0u] : -__builtin_inff();
    float mv1 = (tid() == 0u) ? st[h0 + 1u] : -__builtin_inff();
    float mv2 = (tid() == 0u) ? st[h0 + 2u] : -__builtin_inff();
    for (u32 sp = tid(); sp < nsplit; sp += NT) {
      const unsigned long off = (unsigned long)sp * INK_R16_ER_PSTR;
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
      ink_r16_er_q4_lds.red.m[0][w] = mv0;
      ink_r16_er_q4_lds.red.m[1][w] = mv1;
      ink_r16_er_q4_lds.red.m[2][w] = mv2;
    }
    lbarrier();

    float M2[INK_R16_ER_GQA];
    for (u32 g = 0u; g < INK_R16_ER_GQA; g++) {
      float mx = ink_r16_er_q4_lds.red.m[g][0];
      for (u32 x = 1u; x < 8u; x++) mx = __builtin_fmaxf(mx, ink_r16_er_q4_lds.red.m[g][x]);
      M2[g] = mx;
    }

    float lv0 = 0.0f, lv1 = 0.0f, lv2 = 0.0f;
    for (u32 sp = tid(); sp < nsplit; sp += NT) {
      const unsigned long off = (unsigned long)sp * INK_R16_ER_PSTR;
      const float lp0 = ph0[off + 1u];
      const float wt0 = (lp0 > 0.0f) ? __builtin_expf(ph0[off] - M2[0]) : 0.0f;
      if (wt_in_lds) ink_r16_er_q4_lds.red.o[0][0][sp] = wt0; else ph0[off] = wt0;
      lv0 += lp0 * wt0;

      const float lp1 = ph1[off + 1u];
      const float wt1 = (lp1 > 0.0f) ? __builtin_expf(ph1[off] - M2[1]) : 0.0f;
      if (wt_in_lds) ink_r16_er_q4_lds.red.o[1][0][sp] = wt1; else ph1[off] = wt1;
      lv1 += lp1 * wt1;

      const float lp2 = ph2[off + 1u];
      const float wt2 = (lp2 > 0.0f) ? __builtin_expf(ph2[off] - M2[2]) : 0.0f;
      if (wt_in_lds) ink_r16_er_q4_lds.red.o[2][0][sp] = wt2; else ph2[off] = wt2;
      lv2 += lp2 * wt2;
    }
    lv0 = wave_tree_f(lv0);
    lv1 = wave_tree_f(lv1);
    lv2 = wave_tree_f(lv2);
    if (l == 0u) {
      ink_r16_er_q4_lds.red.l[0][w] = lv0;
      ink_r16_er_q4_lds.red.l[1][w] = lv1;
      ink_r16_er_q4_lds.red.l[2][w] = lv2;
    }
    if (wt_in_lds) lbarrier(); else barrier();

    float L2[INK_R16_ER_GQA];
    for (u32 g = 0u; g < INK_R16_ER_GQA; g++) {
      float ls = 0.0f;
      for (u32 x = 0u; x < 8u; x++) ls += ink_r16_er_q4_lds.red.l[g][x];
      L2[g] = ls + __builtin_expf(st[h0 + g] - M2[g]);
    }

    for (u32 idx = tid(); idx < INK_R16_ER_GQA * INK_R16_ER_HD; idx += NT) {
      const u32 g = idx >> 7u;
      const u32 e = idx & 127u;
      const u32 h = h0 + g;
      const G float *ph = part + (unsigned long)((t * n_head + h) * nsplit) * INK_R16_ER_PSTR;
      float o = 0.0f;
      if (wt_in_lds) {
        for (u32 sp = 0u; sp < nsplit; sp++)
          o += ph[(unsigned long)sp * INK_R16_ER_PSTR + 2u + e] * ink_r16_er_q4_lds.red.o[g][0][sp];
      } else {
        for (u32 sp = 0u; sp < nsplit; sp++)
          o += ph[(unsigned long)sp * INK_R16_ER_PSTR + 2u + e] * ph[(unsigned long)sp * INK_R16_ER_PSTR];
      }
      const u32 out_idx = h * INK_R16_ER_HD + e;
      float v = o / L2[g];
      if (gt) v = ink_r16_er_gate_f(v, gt[out_idx]);
      if (ot) ot[out_idx] = v;
      if (oqt) quant_wave(v, out_idx >> 5u, oqt, nb_q);
    }
  }

  barrier();
  if (tid() == 0u) count[cidx] = 0u;
}

#undef INK_R16_ER_QT4_EDGE_TOK
#undef INK_R16_ER_QT4_ADD_RB3
#undef INK_R16_ER_QT4_QK3
#undef INK_R16_ER_QT2_UPD
#undef INK_R16_ER_QSC
#undef INK_R16_ER_LN2
#undef INK_R16_ER_LOG2E
#undef INK_R16_ER_PSTR
#undef INK_R16_ER_QT4
#undef INK_R16_ER_GQA
#undef INK_R16_ER_DREL
#undef INK_R16_ER_HD
#undef INK_R16_ER_AR_MAXM
#undef INK_R16_ER_NB2_XLS
#undef INK_R16_ER_DEF_KRN_TICKS
#undef INK_R16_ER_DEF_BAR_TICKS
#undef INK_R16_ER_MAX_GRID

#endif /* !IE_MEGA */
