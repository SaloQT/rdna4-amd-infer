/* ============================================================================
 * Inkling-2 Round 1 Group 7: MoE Auxiliary Kernels (RX 9070 XT, gfx1201, wave32, NT=256)
 *
 * Exported KERNELs, exact grids (gx, gy, 1), dispatch guards, and parity:
 *   1. KERNEL ie_inkling_moe_sort_r1(const G u32 *route, u32 rst, u32 T, u32 K, u32 n_exp, G u32 *offs, G u32 *list)
 *      Grid:   (1, 1, 1)
 *      Guard:  T * K <= 8192u && n_exp <= 1024u
 *      Parity: offs[0..n_exp] bitwise-exact; list[offs[e]..offs[e+1]-1] is an exact
 *              permutation of expert e's pairs (block-monotone in 256-pair windows),
 *              producing 100% bitwise-exact outputs in downstream expert GEMMs.
 *   2. KERNEL ie_inkling_situ_r1(const G float *a, G float *b, G u8 *q, u32 n, u32 astr, u32 bstr, u32 qstr, float beta, float lbeta)
 *      Grid:   (gx, nh * T, 1) with gx = (nh * T >= 32u ? 1u : ((n >> 5u) + 31u) / 32u)
 *              (accepts any gx >= 1u via grid-stride over 32-block chunks)
 *      Guard:  (n & 31u) == 0u && (astr & 15u) == 0u && (!b || (bstr & 15u) == 0u) && (!q || (qstr & 3u) == 0u)
 *      Parity: 100% bitwise-exact against ie_situ (both f32 b and Q8_0 q).
 *   3. KERNEL ie_inkling_moe_sum_r1(const G float *y, G float *out, const G u32 *route, const G float *res, u32 n, u32 K, u32 ys, u32 os, u32 rts, u32 rs)
 *      Grid:   ((((n >> 2u) + 255u) / 256u), T, 1)  (i.e. (3, T, 1) for dim=2304)
 *      Guard:  (n & 3u) == 0u && (ys & 15u) == 0u && (os & 15u) == 0u && (!res || (rs & 15u) == 0u)
 *      Parity: 100% bitwise-exact against ie_moe_sum.
 *
 * Included at EOF of kernels/ie_inkling_kernels.c.
 * All local symbols/LDS/macros prefixed ink_r1_g07_ / INK_R1_G07_; macros #undef'd at EOF.
 * ========================================================================== */

#ifdef IE_MEGA
#define INK_R1_G07_MAX_EXP 256u
#define INK_R1_G07_MAX_PAIRS 512u
#define ink_r1_g07_cnt (mk_ink_lds.moe_sort.cnt)
#define ink_r1_g07_er (mk_ink_lds.moe_sort.er)
#define ink_r1_g07_wsum (mk_ink_lds.moe_sort.wsum)
#else
#define INK_R1_G07_MAX_EXP 1024u
#define INK_R1_G07_MAX_PAIRS 8192u

/* ----------------------------------------------------------------------------
 * Kernel 1: ie_inkling_moe_sort_r1
 *   Replaces single-thread (t==0) serial prefix sum + 4096-pair serial scatter
 *   with a fully parallel 256-thread single-WG histogram (recording per-pair
 *   in-expert rank in Pass 1) -> DPP wave-scan parallel prefix sum over n_exp
 *   -> zero-atomic parallel scatter of all T*K pairs in Pass 2.
 * -------------------------------------------------------------------------- */
static LDS u32 ink_r1_g07_cnt[INK_R1_G07_MAX_EXP];
static LDS u32 ink_r1_g07_er[INK_R1_G07_MAX_PAIRS];
static LDS u32 ink_r1_g07_wsum[9];
#endif

static inline __attribute__((always_inline)) u32 ink_r1_g07_wave_scan_inc(u32 x) {
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x111, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x112, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x114, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x118, 0xF, 0xF, true);
  const u32 lo16 = (u32)__builtin_amdgcn_readlane((int)x, 15);
  if (lane() >= 16u) x += lo16;
  return x;
}

KERNEL ie_inkling_moe_sort_r1(const G u32 *route, u32 rst, u32 T, u32 K, u32 n_exp, G u32 *offs, G u32 *list) {
  const u32 t = tid(), w = t >> 5u, l = lane();
  const u32 np = T * K;

  if (n_exp == 0u) {
    if (t == 0u) offs[0] = 0u;
    return;
  }
#ifdef IE_MEGA
  if (n_exp > INK_R1_G07_MAX_EXP || np > INK_R1_G07_MAX_PAIRS) return;
#endif

  for (u32 e = t; e < n_exp; e += NT) ink_r1_g07_cnt[e] = 0u;
  lbarrier();

  /* Pass 1: parallel histogram + record (expert << 16 | in_expert_rank) per pair. */
  if (K == 8u) {
    const u32 kk = t & 7u;
    const u32 tk0 = t >> 3u;
    for (u32 p = t, tk = tk0; p < np; p += NT, tk += 32u) {
      const u32 e = ((const G u32 *)((const G u8 *)route + (unsigned long)tk * rst))[kk];
      const u32 r = __atomic_fetch_add(&ink_r1_g07_cnt[e], 1u, __ATOMIC_RELAXED);
      ink_r1_g07_er[p] = (e << 16u) | r;
    }
  } else {
    for (u32 p = t; p < np; p += NT) {
      const u32 e = ((const G u32 *)((const G u8 *)route + (unsigned long)(p / K) * rst))[p % K];
      const u32 r = __atomic_fetch_add(&ink_r1_g07_cnt[e], 1u, __ATOMIC_RELAXED);
      ink_r1_g07_er[p] = (e << 16u) | r;
    }
  }
  lbarrier();

  /* Parallel exclusive prefix sum over n_exp (1 iteration for n_exp <= 256). */
  u32 base = 0u;
  for (u32 eb = 0u; eb < n_exp; eb += NT) {
    const u32 e = eb + t;
    const u32 c = e < n_exp ? ink_r1_g07_cnt[e] : 0u;
    const u32 inc = ink_r1_g07_wave_scan_inc(c);
    if (l == 31u) ink_r1_g07_wsum[w] = inc;
    lbarrier();
    if (w == 0u) {
      const u32 ws = l < 8u ? ink_r1_g07_wsum[l] : 0u;
      const u32 winc = ink_r1_g07_wave_scan_inc(ws);
      if (l < 8u) ink_r1_g07_wsum[l] = winc - ws;
      if (l == 7u) ink_r1_g07_wsum[8] = winc;
    }
    lbarrier();
    if (e < n_exp) {
      const u32 off = base + ink_r1_g07_wsum[w] + (inc - c);
      ink_r1_g07_cnt[e] = off;
      offs[e] = off;
    }
    base += ink_r1_g07_wsum[8];
    if (eb + NT < n_exp) lbarrier();
  }
  if (t == 0u) offs[n_exp] = base;
  lbarrier();

  /* Pass 2: zero-atomic parallel scatter of all T*K pairs into list[]. */
  for (u32 p = t; p < np; p += NT) {
    const u32 er = ink_r1_g07_er[p];
    const u32 e = er >> 16u;
    const u32 r = er & 0xFFFFu;
    list[ink_r1_g07_cnt[e] + r] = p;
  }
}

/* ----------------------------------------------------------------------------
 * Kernel 2: ie_inkling_situ_r1
 *   8-lane subgroup f32x4 SITU + fused Q8_0 quantization (32 blocks / 1024
 *   elements per WG, 0 LDS, 0 spills, packed u32 int8 stores).
 *   100% bitwise-exact against ie_situ (same ie_tanhf_s, sigmoidf, roundf,
 *   amax/127.0f, and u32 asum).
 * -------------------------------------------------------------------------- */
static inline __attribute__((always_inline)) float ink_r1_g07_sub8_max_pos(float v) {
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0xB1, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0x4E, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_ds_swizzle(F2I(v), 0x101F)));
  return v;
}

static inline __attribute__((always_inline)) u32 ink_r1_g07_sub8_sum_u32(u32 s) {
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0xB1, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0x4E, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_ds_swizzle((int)s, 0x101F);
  return s;
}

KERNEL ie_inkling_situ_r1(const G float *a, G float *b, G u8 *q, u32 n, u32 astr, u32 bstr, u32 qstr, float beta,
                          float lbeta) {
#pragma clang fp contract(off)
  const u32 nb = n >> 5u;
  const u32 sg = tid() >> 3u;
  const u32 sl = lane() & 7u;
  const u32 gx = ngroups_x();

  a = TOKC(a, astr);
  if (b) b = TOK(b, bstr);
  if (q) q = TOK(q, qstr);

  const int use_beta = beta > 0.0f;
  const int use_lbeta = lbeta > 0.0f;

  for (u32 blk = wgid() * 32u + sg; blk < nb; blk += gx * 32u) {
    const u32 off = blk * 32u + sl * 4u;
    const f32x4 gv = *(const G f32x4 *)(a + off);
    const f32x4 uv = *(const G f32x4 *)(a + n + off);

    f32x4 vv;
#pragma unroll
    for (u32 k = 0u; k < 4u; k++) {
      const float g = gv[k], u = uv[k];
      const float gs = use_beta ? beta * ie_tanhf_s(g / beta) : g;
      const float us = use_lbeta ? lbeta * ie_tanhf_s(u / lbeta) : u;
      vv[k] = gs * sigmoidf(g) * us;
    }

    if (b) *(G f32x4 *)(b + off) = vv;

    if (q) {
      float am = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(vv[0]), __builtin_fabsf(vv[1])),
                                 __builtin_fmaxf(__builtin_fabsf(vv[2]), __builtin_fabsf(vv[3])));
      am = ink_r1_g07_sub8_max_pos(am);
      const float d = am / 127.0f;
      int qi[4];
      if (d != 0.0f) {
#pragma unroll
        for (u32 k = 0u; k < 4u; k++) qi[k] = (int)__builtin_roundf(vv[k] / d);
      } else {
        qi[0] = qi[1] = qi[2] = qi[3] = 0;
      }
      const u32 pk = ((u32)(u8)qi[0]) | (((u32)(u8)qi[1]) << 8u) | (((u32)(u8)qi[2]) << 16u) |
                     (((u32)(u8)qi[3]) << 24u);
      *(G u32 *)(q + off) = pk;
      const u32 s = ink_r1_g07_sub8_sum_u32((u32)qi[0] + (u32)qi[1] + (u32)qi[2] + (u32)qi[3]);
      if (sl == 0u) {
        ((G float *)(q + 32u * nb))[blk] = d;
        ((G u32 *)(q + 36u * nb))[blk] = s;
      }
    }
  }
}

/* ----------------------------------------------------------------------------
 * Kernel 3: ie_inkling_moe_sum_r1
 *   Vectorized f32x4 loads/stores across dim (576 float4s for dim=2304 ->
 *   grid=(3, T, 1)), wave-uniform scalar weight loads, and unrolled K=8
 *   parallel-fetch + exact-order fmaf accumulation (100% bitwise-exact).
 * -------------------------------------------------------------------------- */
KERNEL ie_inkling_moe_sum_r1(const G float *y, G float *out, const G u32 *route, const G float *res, u32 n, u32 K,
                             u32 ys, u32 os, u32 rts, u32 rs) {
#pragma clang fp contract(off)
  const u32 n4 = n >> 2u;
  const u32 j4 = wgid() * NT + tid();
  if (j4 >= n4) return;

  y = TOKC(y, ys);
  out = TOK(out, os);
  route = TOKC(route, rts);
  if (res) res = TOKC(res, rs);

  const G float *w = (const G float *)(route + K);
  const G f32x4 *y4 = (const G f32x4 *)y;
  f32x4 acc = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};

  if (K == 8u) {
    const float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    const float w4 = w[4], w5 = w[5], w6 = w[6], w7 = w[7];
    const f32x4 y0 = y4[0ul * n4 + j4];
    const f32x4 y1 = y4[1ul * n4 + j4];
    const f32x4 y2 = y4[2ul * n4 + j4];
    const f32x4 y3 = y4[3ul * n4 + j4];
    const f32x4 y4v = y4[4ul * n4 + j4];
    const f32x4 y5 = y4[5ul * n4 + j4];
    const f32x4 y6 = y4[6ul * n4 + j4];
    const f32x4 y7 = y4[7ul * n4 + j4];
#pragma unroll
    for (u32 c = 0u; c < 4u; c++) {
      float a = __builtin_fmaf(w0, y0[c], 0.0f);
      a = __builtin_fmaf(w1, y1[c], a);
      a = __builtin_fmaf(w2, y2[c], a);
      a = __builtin_fmaf(w3, y3[c], a);
      a = __builtin_fmaf(w4, y4v[c], a);
      a = __builtin_fmaf(w5, y5[c], a);
      a = __builtin_fmaf(w6, y6[c], a);
      a = __builtin_fmaf(w7, y7[c], a);
      acc[c] = a;
    }
  } else {
    for (u32 k = 0u; k < K; k++) {
      const float wk = w[k];
      const f32x4 yk = y4[(unsigned long)k * n4 + j4];
#pragma unroll
      for (u32 c = 0u; c < 4u; c++) acc[c] = __builtin_fmaf(wk, yk[c], acc[c]);
    }
  }

  if (res) {
    const f32x4 rv = ((const G f32x4 *)res)[j4];
    acc = rv + acc;
  }
  ((G f32x4 *)out)[j4] = acc;
}

#ifdef IE_MEGA
#undef ink_r1_g07_cnt
#undef ink_r1_g07_er
#undef ink_r1_g07_wsum
#endif
#undef INK_R1_G07_MAX_EXP
#undef INK_R1_G07_MAX_PAIRS
