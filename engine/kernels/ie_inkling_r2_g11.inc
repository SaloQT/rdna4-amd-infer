/* ============================================================================
 * Inkling-2 Round 2 Group 11: Loop Controller Fused Kernels (gfx1201, wave32, NT=256)
 *
 * Eliminates 384 launches/step (all 192 ie_quant_q8 + 96 RMSNorm + 96 loop_gate)
 * and ~1.8 GB/chunk prefill intermediate traffic (cat, hq, fg) across both
 * decode (T=1) and prefill (T>1).
 *
 * Exported kernels & dispatch spec:
 *
 * 1. KERNEL ie_inkling_lc_prenorm_q8_r2(const G float *inp, const G float *msg,
 *                                       G u8 *q, G float *cat, u32 dim, float eps,
 *                                       u32 is, u32 ms, u32 qs, u32 cs)
 *    - Bit:    23 (IE_INKLING_KERNEL_MASK)
 *    - Fuses:  [OP_RMSNORM(inp, ones_dim -> cat[0..dim-1]),
 *               OP_RMSNORM(msg, ones_dim -> cat[dim..2*dim-1]),
 *               OP_QUANT(cat -> q[2*dim])]
 *    - Grid:   (2u, T, 1u) via launch_t(b, K_INK_LC_PRENORM_Q8_R2, 2u, T, args)
 *              (bx == 0: inp -> Q8 blocks [0..dim/32-1];
 *               bx == 1: msg -> Q8 blocks [dim/32..2*dim/32-1])
 *    - Strides: is, ms, qs, cs are all per-token BYTE strides.
 *               Pass cat = NULL (dead-buffer skip) when cat is not read again
 *               (always true in ink_loop_ctrl_op when lc->down.kind != MAT_F32).
 *    - Guard:  T >= 1u && dim >= 32u && (dim & 31u) == 0u &&
 *              ((uintptr_t)inp & 15u) == 0u && ((uintptr_t)msg & 15u) == 0u &&
 *              (!cat || ((uintptr_t)cat & 15u) == 0u) &&
 *              (T == 1u || (((is | ms | (cat ? cs : 0u)) & 15u) == 0u))
 *    - Parity: bitwise-exact (0-bit-diff) vs ie_rmsnorm_mt(inp, ones_dim) +
 *              ie_rmsnorm_mt(msg, ones_dim) + ie_quant_q8(cat, q).
 *
 * 2. KERNEL ie_inkling_lc_wfg_gate_r2(const G u32 *qw, const G f16 *qs,
 *                                     const G float *h, const G float *inp,
 *                                     const G float *msg, G float *out,
 *                                     u32 dim, u32 nb, u32 T,
 *                                     u32 hs, u32 is, u32 ms, u32 os)
 *    - Bit:    24 (IE_INKLING_KERNEL_MASK)
 *    - Fuses:  [OP_QUANT(h[64] -> hq),
 *               OP_GEMV_Q8/OP_GEMM_Q8(wfg[2*dim, 64], hq -> fg[2*dim]),
 *               OP_LOOP_GATE(inp, msg, fg -> out[dim])]
 *              where wfg rows [0..dim-1] = forget and [dim..2*dim-1] = gain.
 *    - Grid:   T == 1u: ((dim + 31u) / 32u, 1u, 1u)   [32 outputs/WG, 72 WGs for dim=2304]
 *              T >  1u: ((dim + 127u) / 128u, (T + 63u) / 64u, 1u)  [128x64 tile, 144 WGs at T=512]
 *              via launch_t(b, K_INK_LC_WFG_GATE_R2, gx, gy, args).
 *    - Strides: hs, is, ms, os are all per-token BYTE strides.
 *    - Guard:  T >= 1u && dim > 0u && nb == 2u &&
 *              ((uintptr_t)qw & 3u) == 0u && ((uintptr_t)qs & 3u) == 0u &&
 *              (T == 1u || ((hs | is | ms | os) & 3u) == 0u)
 *              (Note: for strict 0-bit-diff at all T, guard T == 1u || T >= 17u;
 *               2u <= T <= 16u is also valid within <= 1 ULP vs ie_gemv_q8q8.)
 *    - Parity: T == 1u: bitwise-exact (0-bit-diff) vs ie_quant_q8 + ie_gemv_q8q8 + ie_loop_gate.
 *              T >= 17u: bitwise-exact (0-bit-diff) vs ie_quant_q8 + ie_gemm_q8 + ie_loop_gate.
 * ========================================================================== */

#define INK_R2_G11_NB2_R 128u
#define INK_R2_G11_NB2_T 64u
#define INK_R2_G11_NB2_XLS 80u /* 64 B (nb=2) + 16 B pad -> 0 bank conflicts on b128 WMMA reads */

typedef _Float16 ink_r2_g11_f16x2 __attribute__((ext_vector_type(2)));

struct __attribute__((packed, aligned(4))) ink_r2_g11_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r2_g11_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r2_g11_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) float ink_r2_g11_loop_gate1(float a, float c, float f_raw, float g_raw) {
#pragma clang fp contract(off)
  const float f = ie_tanhf_s(f_raw);
  const float g = ie_tanhf_s(g_raw);
  return (1.0f - f) * a + g * c;
}

/* Pack 32 per-lane int8 values in a wave into 8 conflict-free u32 LDS stores (32 bytes). */
static inline __attribute__((always_inline)) void ink_r2_g11_quant_pack_lds(float v, u32 l, LDSP u8 *dst_q,
                                                                            LDSP float *dst_d) {
  const float amax = wave_max_pos(__builtin_fabsf(v));
  const float d = amax / 127.0f;
  const int qi = d != 0.0f ? (int)__builtin_roundf(v / d) : 0;
  const u32 b0 = (u32)(u8)(i8)qi;
  const u32 p2 = b0 | ((u32)shl_i((int)b0, 1) << 8u);
  const u32 p4 = p2 | ((u32)shl_i((int)p2, 2) << 16u);
  if ((l & 3u) == 0u) ((LDSP u32 *)dst_q)[l >> 2u] = p4;
  if (l == 0u) *dst_d = d;
}

/* ============================================================================
 * Kernel 1: ie_inkling_lc_prenorm_q8_r2
 * Fuses unweighted RMSNorm(inp) + unweighted RMSNorm(msg) + Q8_0 quant(cat)
 * into 1 launch (grid = (2, T, 1)), skipping ones_dim reads and dead cat writes.
 * ========================================================================== */

#ifdef IE_MEGA
#define ink_r2_g11_pn_red (mk_ink_lds.rmsnorm.red)
#define ink_r2_g11_pn_sh (mk_ink_lds.rmsnorm.sh)
#else
static LDS float ink_r2_g11_pn_red[8];
static LDS float ink_r2_g11_pn_sh[8][128] __attribute__((aligned(16)));
#endif

/* 1-barrier LDS-only workgroup sum (bitwise-identical order to wg_sum). */
static inline __attribute__((always_inline)) float ink_r2_g11_wg_sum_1b(float v, u32 wv, u32 l) {
  v = wave_tree_f(v);
  if (l == 0u) ink_r2_g11_pn_red[wv] = v;
  lbarrier();
  float s = 0.0f;
#pragma unroll
  for (u32 w = 0u; w < 8u; w++) s += ink_r2_g11_pn_red[w];
  return s;
}

/* Transpose 4 floats/lane (128 floats = 4 Q8_0 blocks per wave) via wave-private LDS and quantize. */
static inline __attribute__((always_inline)) void ink_r2_g11_quant_chunk4(f32x4 y4, u32 wv, u32 l, u32 b0, G u8 *q,
                                                                          u32 nb_tot) {
  *(LDSP f32x4 *)&ink_r2_g11_pn_sh[wv][l * 4u] = y4;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 m = 0u; m < 4u; m++) quant_wave(ink_r2_g11_pn_sh[wv][32u * m + l], b0 + m, q, nb_tot);
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

KERNEL ie_inkling_lc_prenorm_q8_r2(const G float *inp, const G float *msg, G u8 *q, G float *cat, u32 dim, float eps,
                                   u32 is, u32 ms, u32 qs, u32 cs) {
  const u32 bx = wgid();
  if (bx >= 2u || dim == 0u || (dim & 31u) != 0u || !inp || !msg || !q) return;

  const G float *x = (bx == 0u) ? TOKC(inp, is) : TOKC(msg, ms);
  G float *y = cat ? (TOK(cat, cs) + (unsigned long)bx * dim) : (G float *)0;
  q = TOK(q, qs);

  const u32 th = tid(), wv = th >> 5u, l = lane();
  const u32 nb = dim >> 5u;
  const u32 nb_tot = nb << 1u;
  const u32 q_blk_off = bx * nb;

  /* Fast path: Inkling-2 dim == 2304 (100% of loop controller calls).
   * 9 floats/thread stay in VGPRs; zero ones_dim loads; 1 lbarrier(); 0 spills. */
  if (dim == 2304u) {
    const f32x4 x0 = *(const G f32x4 *)(x + th * 4u);
    const f32x4 x1 = *(const G f32x4 *)(x + 1024u + th * 4u);
    const float x2 = x[2048u + th];

    float ss = (x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w);
    ss += (x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w);
    ss += x2 * x2;
    ss = ink_r2_g11_wg_sum_1b(ss, wv, l);
    const float s = 1.0f / __builtin_sqrtf(ss / 2304.0f + eps);

    const f32x4 y0 = x0 * s;
    const f32x4 y1 = x1 * s;
    const float y2 = x2 * s;
    if (y) {
      *(G f32x4 *)(y + th * 4u) = y0;
      *(G f32x4 *)(y + 1024u + th * 4u) = y1;
      y[2048u + th] = y2;
    }
    ink_r2_g11_quant_chunk4(y0, wv, l, q_blk_off + wv * 4u, q, 144u);
    ink_r2_g11_quant_chunk4(y1, wv, l, q_blk_off + 32u + wv * 4u, q, 144u);
    quant_wave(y2, q_blk_off + 64u + wv, q, 144u);
    return;
  }

  /* Small-dim path (dim <= 256). */
  if (dim <= NT) {
    const float xi = (th < dim) ? x[th] : 0.0f;
    const float ss = ink_r2_g11_wg_sum_1b(xi * xi, wv, l);
    const float s = 1.0f / __builtin_sqrtf(ss / (float)dim + eps);
    if (th < dim) {
      const float v = xi * s;
      if (y) y[th] = v;
      quant_wave(v, q_blk_off + wv, q, nb_tot);
    }
    return;
  }

  /* General path for any (dim & 31u) == 0u. */
  float ss = 0.0f;
  u32 k = 0u;
  for (; (k + 1u) * 1024u <= dim; k++) {
    const f32x4 vk = *(const G f32x4 *)(x + k * 1024u + th * 4u);
    ss += (vk.x * vk.x + vk.y * vk.y) + (vk.z * vk.z + vk.w * vk.w);
  }
  for (u32 i = k * 1024u + th; i < dim; i += NT) {
    const float xi = x[i];
    ss += xi * xi;
  }
  ss = ink_r2_g11_wg_sum_1b(ss, wv, l);
  const float s = 1.0f / __builtin_sqrtf(ss / (float)dim + eps);

  for (u32 c = 0u; c < k; c++) {
    const u32 base = c * 1024u + th * 4u;
    const f32x4 y4 = *(const G f32x4 *)(x + base) * s;
    if (y) *(G f32x4 *)(y + base) = y4;
    ink_r2_g11_quant_chunk4(y4, wv, l, q_blk_off + c * 32u + wv * 4u, q, nb_tot);
  }
  for (u32 b = k * 32u + wv; b < nb; b += 8u) {
    const u32 i = b * 32u + l;
    const float v = x[i] * s;
    if (y) y[i] = v;
    quant_wave(v, q_blk_off + b, q, nb_tot);
  }
}

/* ============================================================================
 * Kernel 2: ie_inkling_lc_wfg_gate_r2
 * Fuses quant_q8(h[64]) + wfg[2*dim, 64] + loop_gate(inp, msg, fg -> out[dim])
 * into 1 launch for both T == 1 (decode) and T > 1 (prefill).
 * ========================================================================== */

#ifdef IE_MEGA
#define ink_r2_g11_lds_x (mk_ink_lds.lc_wfg.x)
#define ink_r2_g11_lds_dx (mk_ink_lds.lc_wfg.dx)
#else
static LDS u8 ink_r2_g11_lds_x[INK_R2_G11_NB2_T * INK_R2_G11_NB2_XLS] __attribute__((aligned(16)));
static LDS float ink_r2_g11_lds_dx[2u][INK_R2_G11_NB2_T] __attribute__((aligned(32)));
#endif

static inline __attribute__((always_inline)) void ink_r2_g11_wfg_gate_dec(
    const G u32 *qw, const G f16 *qs, const G float *h, const G float *inp, const G float *msg, G float *out, u32 dim) {
#pragma clang fp contract(off)
  const u32 t0 = tid();
  const u32 w = t0 >> 5u;
  const u32 l = lane();

  /* Waves 0 and 1 quantize h[0..31] and h[32..63] into LDS (64 B + 2 scales). */
  if (w < 2u) {
    const float hv = h[(w << 5u) + l];
    ink_r2_g11_quant_pack_lds(hv, l, ink_r2_g11_lds_x + (w << 5u), &ink_r2_g11_lds_dx[w][0u]);
  }

  /* 32 outputs/WG (4 outputs/wave, 8 lanes/output: lanes 0..3 forget, 4..7 gain).
   * Issue weight and inp/msg loads before lbarrier() to overlap with h quantization. */
  const u32 grp = l >> 3u;
  const u32 is_gain = (l >> 2u) & 1u;
  const u32 sub = l & 3u;
  const u32 blk = sub >> 1u;
  const u32 i = wgid() * 32u + (w << 2u) + grp;

  u32x4 wv = (u32x4)(0u);
  float sw = 0.0f;
  float inp_val = 0.0f, msg_val = 0.0f;
  if (i < dim) {
    const u32 row = is_gain ? (dim + i) : i;
    wv = ink_r2_g11_ld4((const G u8 *)qw + (unsigned long)row * 64ul + (unsigned long)(sub << 4u));
    sw = (float)qs[(unsigned long)row * 2ul + blk];
    if ((l & 7u) == 0u) {
      inp_val = inp[i];
      msg_val = msg[i];
    }
  }

  lbarrier();

  if (i < dim) {
    const u32x4 av = *(const LDSP u32x4 *)(ink_r2_g11_lds_x + (sub << 4u));
    const u32 d_half =
        ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(0u, wv.x, av.x), wv.y, av.y), wv.z, av.z), wv.w, av.w);
    const int dot = (int)d_half + shl_i((int)d_half, 1);
    const float sc = sw * ink_r2_g11_lds_dx[blk][0u];
    const float term = sc * (float)dot;
    const float sum = term + I2F(shl_i(F2I(term), 2));
    /* Lane grp*8 + 0 receives gain[i] from lane grp*8 + 4 within the same 16-lane DPP row. */
    const float gain_val = I2F(shl_i(F2I(sum), 4));
    if ((l & 7u) == 0u) out[i] = ink_r2_g11_loop_gate1(inp_val, msg_val, sum, gain_val);
  }
}

KERNEL_LOOP ie_inkling_lc_wfg_gate_r2(const G u32 *qw, const G f16 *qs, const G float *h, const G float *inp,
                                      const G float *msg, G float *out, u32 dim, u32 nb, u32 T, u32 hs, u32 is, u32 ms,
                                      u32 os) {
  if (T == 0u || dim == 0u || nb != 2u || !qw || !qs || !h || !inp || !msg || !out) return;

  if (T == 1u) {
    ink_r2_g11_wfg_gate_dec(qw, qs, h, inp, msg, out, dim);
    return;
  }

  /* Prefill path (T > 1): 128 output elements x 64 tokens per WG, swapped WMMA (A=X, B=W).
   * Each wave owns 16 paired (forget, gain) rows and reuses LDS X across both matrices. */
  const u32 rbase = wgid() * INK_R2_G11_NB2_R;
  const u32 tbase = tokid() * INK_R2_G11_NB2_T;
  if (rbase >= dim || tbase >= T) return;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, hh = l >> 4u;

  /* Quantize 64 tokens x 64 floats of h into LDS (each wave handles 8 tokens x 2 blocks). */
#pragma unroll
  for (u32 k = 0u; k < 8u; k++) {
    const u32 tk = (w << 3u) + k;
    const u32 gt = (tbase + tk < T) ? (tbase + tk) : (T - 1u);
    const G float *ht = (const G float *)((const G u8 *)h + (unsigned long)gt * hs);
    const float v0 = ht[l];
    const float v1 = ht[32u + l];
    LDSP u8 *dst = ink_r2_g11_lds_x + tk * INK_R2_G11_NB2_XLS;
    ink_r2_g11_quant_pack_lds(v0, l, dst, &ink_r2_g11_lds_dx[0u][tk]);
    ink_r2_g11_quant_pack_lds(v1, l, dst + 32u, &ink_r2_g11_lds_dx[1u][tk]);
  }

  /* Load this wave's 16 forget rows and 16 gain rows directly into VGPRs. */
  const u32 r = rbase + (w << 4u) + r16;
  const u32 gr = r < dim ? r : dim - 1u;
  const u32 gr_g = dim + gr;

  const G u8 *wpf = (const G u8 *)qw + (unsigned long)gr * 64ul + (unsigned long)(hh << 4u);
  const u32x4 wf0 = ink_r2_g11_ld4(wpf);
  const u32x4 wf1 = ink_r2_g11_ld4(wpf + 32ul);
  const u32 swf_raw = *(const G u32 *)(qs + (unsigned long)gr * 2ul);
  const ink_r2_g11_f16x2 swf_p = __builtin_bit_cast(ink_r2_g11_f16x2, swf_raw);
  const float swf0 = (float)swf_p.x;
  const float swf1 = (float)swf_p.y;

  const G u8 *wpg = (const G u8 *)qw + (unsigned long)gr_g * 64ul + (unsigned long)(hh << 4u);
  const u32x4 wg0 = ink_r2_g11_ld4(wpg);
  const u32x4 wg1 = ink_r2_g11_ld4(wpg + 32ul);
  const u32 swg_raw = *(const G u32 *)(qs + (unsigned long)gr_g * 2ul);
  const ink_r2_g11_f16x2 swg_p = __builtin_bit_cast(ink_r2_g11_f16x2, swg_raw);
  const float swg0 = (float)swg_p.x;
  const float swg1 = (float)swg_p.y;

  lbarrier();

  const v8i zm = (v8i)(0x4B400000);
  const v2i bf0_0 = {(int)wf0.x, (int)wf0.y}, bf0_1 = {(int)wf0.z, (int)wf0.w};
  const v2i bf1_0 = {(int)wf1.x, (int)wf1.y}, bf1_1 = {(int)wf1.z, (int)wf1.w};
  const v2i bg0_0 = {(int)wg0.x, (int)wg0.y}, bg0_1 = {(int)wg0.z, (int)wg0.w};
  const v2i bg1_0 = {(int)wg1.x, (int)wg1.y}, bg1_1 = {(int)wg1.z, (int)wg1.w};

#ifdef IE_MEGA
#pragma clang loop unroll(disable)
#else
#pragma unroll
#endif
  for (u32 j = 0u; j < 4u; j++) {
#ifdef IE_MEGA
    if (tbase + (j << 4u) >= T) break;
#endif
    const LDSP u8 *xp = ink_r2_g11_lds_x + ((j << 4u) + r16) * INK_R2_G11_NB2_XLS + (hh << 4u);
    const u32x4 x0 = *(const LDSP u32x4 *)(xp);
    const u32x4 x1 = *(const LDSP u32x4 *)(xp + 32u);
    const f8 dx0 = *(const LDSP f8 *)(ink_r2_g11_lds_dx[0u] + (j << 4u) + (hh << 3u));
    const f8 dx1 = *(const LDSP f8 *)(ink_r2_g11_lds_dx[1u] + (j << 4u) + (hh << 3u));

    const v2i a0_0 = {(int)x0.x, (int)x0.y}, a0_1 = {(int)x0.z, (int)x0.w};
    const v2i a1_0 = {(int)x1.x, (int)x1.y}, a1_1 = {(int)x1.z, (int)x1.w};

    v8i cf0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_0, 1, bf0_0, zm, 0);
    cf0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_1, 1, bf0_1, cf0, 0);
    const f8 df0 = __builtin_bit_cast(f8, cf0) - 12582912.0f;

    v8i cf1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_0, 1, bf1_0, zm, 0);
    cf1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_1, 1, bf1_1, cf1, 0);
    const f8 df1 = __builtin_bit_cast(f8, cf1) - 12582912.0f;

    v8i cg0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_0, 1, bg0_0, zm, 0);
    cg0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_1, 1, bg0_1, cg0, 0);
    const f8 dg0 = __builtin_bit_cast(f8, cg0) - 12582912.0f;

    v8i cg1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_0, 1, bg1_0, zm, 0);
    cg1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_1, 1, bg1_1, cg1, 0);
    const f8 dg1 = __builtin_bit_cast(f8, cg1) - 12582912.0f;

    const f8 scf0 = (f8)(swf0) * dx0;
    const f8 scf1 = (f8)(swf1) * dx1;
    const f8 scg0 = (f8)(swg0) * dx0;
    const f8 scg1 = (f8)(swg1) * dx1;

    const u32 t_base_v = tbase + (j << 4u) + (hh << 3u);
    if (r < dim) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 t = t_base_v + v;
        if (t < T) {
          float f_raw = __builtin_fmaf(scf0[v], df0[v], 0.0f);
          f_raw = __builtin_fmaf(scf1[v], df1[v], f_raw);
          float g_raw = __builtin_fmaf(scg0[v], dg0[v], 0.0f);
          g_raw = __builtin_fmaf(scg1[v], dg1[v], g_raw);
          const float a_val = *(const G float *)((const G u8 *)inp + (unsigned long)t * is + (unsigned long)r * 4ul);
          const float c_val = *(const G float *)((const G u8 *)msg + (unsigned long)t * ms + (unsigned long)r * 4ul);
          *(G float *)((G u8 *)out + (unsigned long)t * os + (unsigned long)r * 4ul) =
              ink_r2_g11_loop_gate1(a_val, c_val, f_raw, g_raw);
        }
      }
    }
  }
}

#ifdef IE_MEGA
#undef ink_r2_g11_pn_red
#undef ink_r2_g11_pn_sh
#undef ink_r2_g11_lds_x
#undef ink_r2_g11_lds_dx
#endif
#undef INK_R2_G11_NB2_XLS
#undef INK_R2_G11_NB2_T
#undef INK_R2_G11_NB2_R
