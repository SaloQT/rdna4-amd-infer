/* ie_inkling_r1_g06.inc -- Inkling-2-Mini (dim=2304, gfx1201, wave32, NT=256)
 * optimized Depth-Attention Residual (ie_attn_res) and 1-WG/token RMSNorm (ie_rmsnorm_mt).
 *
 * Exported kernels & dispatch spec:
 *   1. KERNEL ie_inkling_attn_res_r1(const G float *a, const G float *c, const G float *q, G float *b,
 *                                    u32 M, u32 K, u32 n, float eps, u32 as, u32 cs, u32 bs, int has_c)
 *      - Grid: (1u, T, 1u) via launch_t(b, K_INK_ATTN_RES_R1, 1u, T, args)  [same signature & grid as ie_attn_res]
 *      - Guard: M >= 1u && M <= 64u && K >= 1u && K <= 8u && n >= 1u && ((n & 3u) != 0u || (
 *               ((!has_c || M > 1u) ? (((uintptr_t)A & 15u) == 0u && (T == 1u || (as & 15u) == 0u)) : 1) &&
 *               (!has_c || (((uintptr_t)C & 15u) == 0u && (T == 1u || (cs & 15u) == 0u))) &&
 *               ((uintptr_t)Qp & 15u) == 0u && ((uintptr_t)B & 15u) == 0u && (T == 1u || (bs & 15u) == 0u)))
 *      - Parity: f32-tol (<= 1e-6 vs ie_attn_res; f32x4 grouping & wave-parallel softmax).
 *
 *   2. KERNEL ie_inkling_rmsnorm_mt_r1(const G float *x, const G float *w, G float *y, u32 n, float eps,
 *                                      G u8 *q, u32 xs, u32 ys, u32 qs)
 *      - Grid: (1u, T, 1u) via launch_t(b, K_INK_RMSNORM_MT_R1, 1u, T, args)  [1 WG/token instead of (n+1023)/1024]
 *      - Guard: (n & 31u) == 0u && n >= 32u && ((uintptr_t)A & 15u) == 0u && (T == 1u || (xs & 15u) == 0u) &&
 *               ((uintptr_t)w0 & 15u) == 0u && (!Bout || (((uintptr_t)Bout & 15u) == 0u && (T == 1u || (ys & 15u) == 0u)))
 *               (Note: in-place o->a == o->b IS safe because grid_x == 1u.)
 *      - Parity: bitwise-exact vs ie_rmsnorm_mt for all (n & 31u) == 0u (identical f32x4 + tail reduction order).
 */

#define INK_R1_G06_AR_MAXM 64u
#define INK_R1_G06_AR_MAXK 8u
#ifdef IE_MEGA
/* 256 * 8 = 2048 floats (8 KiB): covers all M<=64 for K<=3 and M<=32 for K<=7
 * (100% of Inkling-2 calls) while keeping pooled megakernel LDS <= 14 KiB. */
#define INK_R1_G06_AR_SLOTS 256u
#define ink_r1_g06_ar_red (mk_ink_lds.attn_res.ar_red)
#define ink_r1_g06_ar_inv (mk_ink_lds.attn_res.ar_inv)
#define ink_r1_g06_ar_prob (mk_ink_lds.attn_res.ar_prob)
#define ink_r1_g06_rn_red (mk_ink_lds.rmsnorm.red)
#define ink_r1_g06_rn_sh (mk_ink_lds.rmsnorm.sh)
#else
/* 384 * 8 = 3072 floats (12 KiB): holds all M*(K+1) wave partials with 0 in-loop barriers
 * whenever M*(K+1) <= 384 (all M<=64 for K<=5, M<=42 for K<=8). Total AR LDS = 14.25 KiB <= 16 KiB (4 WGs/CU). */
#define INK_R1_G06_AR_SLOTS 384u

static LDS float ink_r1_g06_ar_red[INK_R1_G06_AR_SLOTS * 8u];
static LDS float ink_r1_g06_ar_inv[INK_R1_G06_AR_MAXM];
static LDS float ink_r1_g06_ar_prob[INK_R1_G06_AR_MAXK][INK_R1_G06_AR_MAXM];

static LDS float ink_r1_g06_rn_red[8];
static LDS float ink_r1_g06_rn_sh[8][128] __attribute__((aligned(16)));
#endif

static inline float ink_r1_g06_wave_max(float v) {
  v = __builtin_fmaxf(v, shfl_xor_f(v, 16u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 8u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 4u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 2u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 1u));
  return v;
}

/* Wave-parallel softmax over lanes 0..M-1 (requires M <= 32, called uniformly by all 32 lanes). */
static inline float ink_r1_g06_wave_softmax(float logit, u32 l, u32 M) {
  const float v = (l < M) ? logit : -__builtin_inff();
  const float mx = ink_r1_g06_wave_max(v);
  const float e = (l < M) ? __builtin_expf(logit - mx) : 0.0f;
  const float ls = wave_sum_all(e);
  return (l < M) ? (e * (1.0f / ls)) : 0.0f;
}

/* 1-barrier LDS-only workgroup sum (bitwise-identical order to wg_sum).
 * Safe only when called once per workgroup lifetime (no subsequent overwrite of ink_r1_g06_rn_red). */
static inline float ink_r1_g06_wg_sum_1b(float v, u32 wv, u32 l) {
  v = wave_tree_f(v);
  if (l == 0u) ink_r1_g06_rn_red[wv] = v;
  lbarrier();
  float s = 0.0f;
#pragma unroll
  for (u32 w = 0; w < 8u; w++) s += ink_r1_g06_rn_red[w];
  return s;
}

/* Transpose 4 floats/lane (128 floats = 4 Q8_0 blocks per wave) via wave-private LDS and quantize. */
static inline void ink_r1_g06_quant_chunk4(f32x4 y4, u32 wv, u32 l, u32 b0, G u8 *q, u32 nb) {
  *(LDSP f32x4 *)&ink_r1_g06_rn_sh[wv][l * 4u] = y4;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 m = 0; m < 4u; m++) quant_wave(ink_r1_g06_rn_sh[wv][32u * m + l], b0 + m, q, nb);
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

/* 1. Optimized Depth-Attention Residual mixing (drop-in replacement for ie_attn_res).
 *    Fast path (n == 2304, M <= 32, K <= 3 -- 100% of Inkling-2's 144 calls):
 *      - Loads q[0..K-1] ONCE into VGPRs (128-bit f32x4 + 256-float coalesced tail).
 *      - Caches up to 4 slots in named VGPRs (zero scratch spills, zero second global read for M <= 4).
 *      - Collapses 2*M+1 full global-fence barriers down to ONE lbarrier().
 *      - Computes norms & softmax in parallel across lanes 0..M-1 in every wave (no serial tid()==0 loop).
 *      - Uses 128-bit f32x4 stores and inverted s-outside-k weighted sum. */
KERNEL_CALL ie_inkling_attn_res_r1(const G float *a, const G float *c, const G float *q, G float *b,
                                   u32 M, u32 K, u32 n, float eps, u32 as, u32 cs, u32 bs, int has_c) {
  if (M == 0u || M > INK_R1_G06_AR_MAXM || K == 0u || K > INK_R1_G06_AR_MAXK || n == 0u) return;
  const u32 th = tid(), wv = th >> 5, l = lane();
  a = TOKC(a, as), b = TOK(b, bs);
  if (has_c) c = TOKC(c, cs);
  const u32 n_a = has_c ? (M - 1u) : M;
  const float inv_sqn = 1.0f / __builtin_sqrtf((float)n);

  if (n == 2304u && M <= 32u && K <= 3u) {
    const f32x4 z4 = {0.0f, 0.0f, 0.0f, 0.0f};
    /* Load queries q[0..K-1] ONCE into named VGPRs (2304 = 1024 + 1024 + 256). */
    const f32x4 q0_0 = *(const G f32x4 *)(q + th * 4u);
    const f32x4 q1_0 = *(const G f32x4 *)(q + 1024u + th * 4u);
    const float q2_0 = q[2048u + th];
    f32x4 q0_1 = z4, q1_1 = z4, q0_2 = z4, q1_2 = z4;
    float q2_1 = 0.0f, q2_2 = 0.0f;
    if (K > 1u) {
      const G float *qk1 = q + 2304u;
      q0_1 = *(const G f32x4 *)(qk1 + th * 4u);
      q1_1 = *(const G f32x4 *)(qk1 + 1024u + th * 4u);
      q2_1 = qk1[2048u + th];
    }
    if (K > 2u) {
      const G float *qk2 = q + 4608u;
      q0_2 = *(const G f32x4 *)(qk2 + th * 4u);
      q1_2 = *(const G f32x4 *)(qk2 + 1024u + th * 4u);
      q2_2 = qk2[2048u + th];
    }

    /* Named VGPR cache for slots 0..3 (avoids dynamic VGPR indexing / scratch spills). */
    f32x4 c0_0 = z4, c1_0 = z4, c0_1 = z4, c1_1 = z4;
    f32x4 c0_2 = z4, c1_2 = z4, c0_3 = z4, c1_3 = z4;
    float c2_0 = 0.0f, c2_1 = 0.0f, c2_2 = 0.0f, c2_3 = 0.0f;

    /* Pass 1: all M slots' norms and K dot products with ZERO barriers inside the loop. */
    const u32 stride_s = K + 1u;
    for (u32 s = 0; s < M; s++) {
      const G float *sp = s < n_a ? (a + (unsigned long)s * 2304u) : c;
      const f32x4 x0 = *(const G f32x4 *)(sp + th * 4u);
      const f32x4 x1 = *(const G f32x4 *)(sp + 1024u + th * 4u);
      const float x2 = sp[2048u + th];
      if (s == 0u) {
        c0_0 = x0; c1_0 = x1; c2_0 = x2;
      } else if (s == 1u) {
        c0_1 = x0; c1_1 = x1; c2_1 = x2;
      } else if (s == 2u) {
        c0_2 = x0; c1_2 = x1; c2_2 = x2;
      } else if (s == 3u) {
        c0_3 = x0; c1_3 = x1; c2_3 = x2;
      }

      float sq = ((x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w)) +
                 ((x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w)) +
                 x2 * x2;
      sq = wave_tree_f(sq);
      const u32 rbase = s * stride_s * 8u;
      if (l == 0u) ink_r1_g06_ar_red[rbase + wv] = sq;

      {
        const f32x4 p0 = x0 * q0_0, p1 = x1 * q1_0;
        float d0 = ((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w)) + x2 * q2_0;
        d0 = wave_tree_f(d0);
        if (l == 0u) ink_r1_g06_ar_red[rbase + 8u + wv] = d0;
      }
      if (K > 1u) {
        const f32x4 p0 = x0 * q0_1, p1 = x1 * q1_1;
        float d1 = ((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w)) + x2 * q2_1;
        d1 = wave_tree_f(d1);
        if (l == 0u) ink_r1_g06_ar_red[rbase + 16u + wv] = d1;
      }
      if (K > 2u) {
        const f32x4 p0 = x0 * q0_2, p1 = x1 * q1_2;
        float d2 = ((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w)) + x2 * q2_2;
        d2 = wave_tree_f(d2);
        if (l == 0u) ink_r1_g06_ar_red[rbase + 24u + wv] = d2;
      }
    }

    /* Single LDS-only barrier for the entire kernel! */
    lbarrier();

    /* Every wave computes inv_s[0..M-1] and softmax prob[k][0..M-1] in parallel across lanes 0..M-1. */
    float my_inv = 0.0f;
    float lg0 = 0.0f, lg1 = 0.0f, lg2 = 0.0f;
    if (l < M) {
      const u32 rbase = l * stride_s * 8u;
      float sum_sq = 0.0f;
#pragma unroll
      for (u32 x = 0; x < 8u; x++) sum_sq += ink_r1_g06_ar_red[rbase + x];
      my_inv = 1.0f / __builtin_sqrtf(sum_sq / 2304.0f + eps);
      const float scale = my_inv * inv_sqn;

      float sd0 = 0.0f;
#pragma unroll
      for (u32 x = 0; x < 8u; x++) sd0 += ink_r1_g06_ar_red[rbase + 8u + x];
      lg0 = sd0 * scale;

      if (K > 1u) {
        float sd1 = 0.0f;
#pragma unroll
        for (u32 x = 0; x < 8u; x++) sd1 += ink_r1_g06_ar_red[rbase + 16u + x];
        lg1 = sd1 * scale;
      }
      if (K > 2u) {
        float sd2 = 0.0f;
#pragma unroll
        for (u32 x = 0; x < 8u; x++) sd2 += ink_r1_g06_ar_red[rbase + 24u + x];
        lg2 = sd2 * scale;
      }
    }
    const float my_prob0 = ink_r1_g06_wave_softmax(lg0, l, M);
    const float my_prob1 = (K > 1u) ? ink_r1_g06_wave_softmax(lg1, l, M) : 0.0f;
    const float my_prob2 = (K > 2u) ? ink_r1_g06_wave_softmax(lg2, l, M) : 0.0f;

    /* Pass 2: weighted sum (slots 0..3 from VGPRs, s >= 4 via f32x4; inverted s-outside-k). */
    f32x4 a0_0 = z4, a1_0 = z4, a0_1 = z4, a1_1 = z4, a0_2 = z4, a1_2 = z4;
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f;
    for (u32 s = 0; s < M; s++) {
      f32x4 x0, x1;
      float x2;
      if (s == 0u) {
        x0 = c0_0; x1 = c1_0; x2 = c2_0;
      } else if (s == 1u) {
        x0 = c0_1; x1 = c1_1; x2 = c2_1;
      } else if (s == 2u) {
        x0 = c0_2; x1 = c1_2; x2 = c2_2;
      } else if (s == 3u) {
        x0 = c0_3; x1 = c1_3; x2 = c2_3;
      } else {
        const G float *sp = s < n_a ? (a + (unsigned long)s * 2304u) : c;
        x0 = *(const G f32x4 *)(sp + th * 4u);
        x1 = *(const G f32x4 *)(sp + 1024u + th * 4u);
        x2 = sp[2048u + th];
      }
      const float inv_s = I2F(__builtin_amdgcn_readlane(F2I(my_inv), (int)s));
      const f32x4 xn0 = x0 * inv_s;
      const f32x4 xn1 = x1 * inv_s;
      const float xn2 = x2 * inv_s;

      const float p0 = I2F(__builtin_amdgcn_readlane(F2I(my_prob0), (int)s));
      a0_0 += xn0 * p0;
      a1_0 += xn1 * p0;
      a2_0 += xn2 * p0;
      if (K > 1u) {
        const float p1 = I2F(__builtin_amdgcn_readlane(F2I(my_prob1), (int)s));
        a0_1 += xn0 * p1;
        a1_1 += xn1 * p1;
        a2_1 += xn2 * p1;
      }
      if (K > 2u) {
        const float p2 = I2F(__builtin_amdgcn_readlane(F2I(my_prob2), (int)s));
        a0_2 += xn0 * p2;
        a1_2 += xn1 * p2;
        a2_2 += xn2 * p2;
      }
    }

    *(G f32x4 *)(b + th * 4u) = a0_0;
    *(G f32x4 *)(b + 1024u + th * 4u) = a1_0;
    b[2048u + th] = a2_0;
    if (K > 1u) {
      G float *b1 = b + 2304u;
      *(G f32x4 *)(b1 + th * 4u) = a0_1;
      *(G f32x4 *)(b1 + 1024u + th * 4u) = a1_1;
      b1[2048u + th] = a2_1;
    }
    if (K > 2u) {
      G float *b2 = b + 4608u;
      *(G f32x4 *)(b2 + th * 4u) = a0_2;
      *(G f32x4 *)(b2 + 1024u + th * 4u) = a1_2;
      b2[2048u + th] = a2_2;
    }
    return;
  }

  /* General fallback for any n, M <= 64, K <= 8. */
  const u32 stride_s = K + 1u;
  const int vec4_ok = ((n & 3u) == 0u);
  if (M * stride_s <= INK_R1_G06_AR_SLOTS) {
    /* Batched reduction: zero barriers inside the slot loop (only 2 lbarriers total). */
    for (u32 s = 0; s < M; s++) {
      const G float *sp = s < n_a ? (a + (unsigned long)s * n) : c;
      float sq = 0.0f;
      float dt[INK_R1_G06_AR_MAXK];
      for (u32 k = 0; k < K; k++) dt[k] = 0.0f;
      if (vec4_ok) {
        for (u32 i = th * 4u; i < n; i += NT * 4u) {
          const f32x4 x4 = *(const G f32x4 *)(sp + i);
          sq += (x4.x * x4.x + x4.y * x4.y) + (x4.z * x4.z + x4.w * x4.w);
          for (u32 k = 0; k < K; k++) {
            const f32x4 q4 = *(const G f32x4 *)(q + (unsigned long)k * n + i);
            const f32x4 p4 = x4 * q4;
            dt[k] += (p4.x + p4.y) + (p4.z + p4.w);
          }
        }
      } else {
        for (u32 i = th; i < n; i += NT) {
          const float x = sp[i];
          sq += x * x;
          for (u32 k = 0; k < K; k++) dt[k] += x * q[(unsigned long)k * n + i];
        }
      }
      const u32 rbase = s * stride_s * 8u;
      sq = wave_tree_f(sq);
      if (l == 0u) ink_r1_g06_ar_red[rbase + wv] = sq;
      for (u32 k = 0; k < K; k++) {
        const float dk = wave_tree_f(dt[k]);
        if (l == 0u) ink_r1_g06_ar_red[rbase + (1u + k) * 8u + wv] = dk;
      }
    }
    lbarrier();
    if (wv == 0u) {
      for (u32 s = l; s < M; s += 32u) {
        const u32 rbase = s * stride_s * 8u;
        float sum_sq = 0.0f;
#pragma unroll
        for (u32 x = 0; x < 8u; x++) sum_sq += ink_r1_g06_ar_red[rbase + x];
        const float inv_s = 1.0f / __builtin_sqrtf(sum_sq / (float)n + eps);
        ink_r1_g06_ar_inv[s] = inv_s;
        for (u32 k = 0; k < K; k++) {
          float sum_d = 0.0f;
#pragma unroll
          for (u32 x = 0; x < 8u; x++) sum_d += ink_r1_g06_ar_red[rbase + (1u + k) * 8u + x];
          ink_r1_g06_ar_prob[k][s] = sum_d * inv_s * inv_sqn;
        }
      }
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
      if (l == 0u) {
        for (u32 k = 0; k < K; k++) {
          float mx = ink_r1_g06_ar_prob[k][0];
          for (u32 s = 1u; s < M; s++) mx = __builtin_fmaxf(mx, ink_r1_g06_ar_prob[k][s]);
          float ls = 0.0f;
          for (u32 s = 0; s < M; s++) {
            const float e = __builtin_expf(ink_r1_g06_ar_prob[k][s] - mx);
            ink_r1_g06_ar_prob[k][s] = e;
            ls += e;
          }
          const float inv_l = 1.0f / ls;
          for (u32 s = 0; s < M; s++) ink_r1_g06_ar_prob[k][s] *= inv_l;
        }
      }
    }
    lbarrier();
  } else {
    /* Extreme (M, K) where M*(K+1) > 384: per-slot lbarrier fallback. */
    for (u32 s = 0; s < M; s++) {
      const G float *sp = s < n_a ? (a + (unsigned long)s * n) : c;
      float sq = 0.0f;
      float dt[INK_R1_G06_AR_MAXK];
      for (u32 k = 0; k < K; k++) dt[k] = 0.0f;
      if (vec4_ok) {
        for (u32 i = th * 4u; i < n; i += NT * 4u) {
          const f32x4 x4 = *(const G f32x4 *)(sp + i);
          sq += (x4.x * x4.x + x4.y * x4.y) + (x4.z * x4.z + x4.w * x4.w);
          for (u32 k = 0; k < K; k++) {
            const f32x4 q4 = *(const G f32x4 *)(q + (unsigned long)k * n + i);
            const f32x4 p4 = x4 * q4;
            dt[k] += (p4.x + p4.y) + (p4.z + p4.w);
          }
        }
      } else {
        for (u32 i = th; i < n; i += NT) {
          const float x = sp[i];
          sq += x * x;
          for (u32 k = 0; k < K; k++) dt[k] += x * q[(unsigned long)k * n + i];
        }
      }
      sq = wave_tree_f(sq);
      if (l == 0u) ink_r1_g06_ar_red[wv] = sq;
      for (u32 k = 0; k < K; k++) {
        const float dk = wave_tree_f(dt[k]);
        if (l == 0u) ink_r1_g06_ar_red[(1u + k) * 8u + wv] = dk;
      }
      lbarrier();
      if (th == 0u) {
        float sum_sq = 0.0f;
        for (u32 x = 0; x < 8u; x++) sum_sq += ink_r1_g06_ar_red[x];
        const float inv_s = 1.0f / __builtin_sqrtf(sum_sq / (float)n + eps);
        ink_r1_g06_ar_inv[s] = inv_s;
        for (u32 k = 0; k < K; k++) {
          float sum_d = 0.0f;
          for (u32 x = 0; x < 8u; x++) sum_d += ink_r1_g06_ar_red[(1u + k) * 8u + x];
          ink_r1_g06_ar_prob[k][s] = sum_d * inv_s * inv_sqn;
        }
      }
      lbarrier();
    }
    if (th == 0u) {
      for (u32 k = 0; k < K; k++) {
        float mx = ink_r1_g06_ar_prob[k][0];
        for (u32 s = 1u; s < M; s++) mx = __builtin_fmaxf(mx, ink_r1_g06_ar_prob[k][s]);
        float ls = 0.0f;
        for (u32 s = 0; s < M; s++) {
          const float e = __builtin_expf(ink_r1_g06_ar_prob[k][s] - mx);
          ink_r1_g06_ar_prob[k][s] = e;
          ls += e;
        }
        const float inv_l = 1.0f / ls;
        for (u32 s = 0; s < M; s++) ink_r1_g06_ar_prob[k][s] *= inv_l;
      }
    }
    lbarrier();
  }

  /* Weighted sum output (inverted s-outside-k so each slot is read once per i, not K times). */
  if (vec4_ok) {
    for (u32 i = th * 4u; i < n; i += NT * 4u) {
      f32x4 acc[INK_R1_G06_AR_MAXK];
      for (u32 k = 0; k < K; k++) acc[k] = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      for (u32 s = 0; s < M; s++) {
        const G float *sp = s < n_a ? (a + (unsigned long)s * n) : c;
        const f32x4 xn = *(const G f32x4 *)(sp + i) * ink_r1_g06_ar_inv[s];
        for (u32 k = 0; k < K; k++) acc[k] += xn * ink_r1_g06_ar_prob[k][s];
      }
      for (u32 k = 0; k < K; k++) *(G f32x4 *)(b + (unsigned long)k * n + i) = acc[k];
    }
  } else {
    for (u32 i = th; i < n; i += NT) {
      float acc[INK_R1_G06_AR_MAXK];
      for (u32 k = 0; k < K; k++) acc[k] = 0.0f;
      for (u32 s = 0; s < M; s++) {
        const G float *sp = s < n_a ? (a + (unsigned long)s * n) : c;
        const float xn = sp[i] * ink_r1_g06_ar_inv[s];
        for (u32 k = 0; k < K; k++) acc[k] += xn * ink_r1_g06_ar_prob[k][s];
      }
      for (u32 k = 0; k < K; k++) b[(unsigned long)k * n + i] = acc[k];
    }
  }
}

/* 2. 1-WG/token vectorized RMSNorm + optional fused Q8_0 quantization (drop-in signature for ie_rmsnorm_mt).
 *    Grid: launch_t(b, K_INK_RMSNORM_MT_R1, 1u, T, args).
 *    Eliminates 3x redundant x reads (g=3 -> g=1 at dim=2304), tail-WG scalar re-reads, and
 *    cuts 2 global-fence barriers down to 1 lbarrier(). Bitwise-identical to ie_rmsnorm_mt. */
KERNEL ie_inkling_rmsnorm_mt_r1(const G float *x, const G float *w, G float *y, u32 n, float eps,
                                G u8 *q, u32 xs, u32 ys, u32 qs) {
  x = TOKC(x, xs), y = TOK(y, ys), q = TOK(q, qs);
  const u32 th = tid(), wv = th >> 5, l = lane();
  const u32 nb = n >> 5;

  /* Fast path: Inkling-2 dim=2304 (100% of Inkling-2's 346 RMSNorm calls).
   * All 9 floats of x and 9 floats of w stay in VGPRs; zero second reads; 1 lbarrier(). */
  if (n == 2304u) {
    const f32x4 x0 = *(const G f32x4 *)(x + th * 4u);
    const f32x4 x1 = *(const G f32x4 *)(x + 1024u + th * 4u);
    const float x2 = x[2048u + th];
    const f32x4 w0 = *(const G f32x4 *)(w + th * 4u);
    const f32x4 w1 = *(const G f32x4 *)(w + 1024u + th * 4u);
    const float w2 = w[2048u + th];

    float ss = (x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w);
    ss += (x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w);
    ss += x2 * x2;
    ss = ink_r1_g06_wg_sum_1b(ss, wv, l);
    const float s = 1.0f / __builtin_sqrtf(ss / 2304.0f + eps);

    const f32x4 y0 = x0 * s * w0;
    const f32x4 y1 = x1 * s * w1;
    const float y2 = x2 * s * w2;
    if (y) {
      *(G f32x4 *)(y + th * 4u) = y0;
      *(G f32x4 *)(y + 1024u + th * 4u) = y1;
      y[2048u + th] = y2;
    }
    if (q) {
      ink_r1_g06_quant_chunk4(y0, wv, l, wv * 4u, q, 72u);
      ink_r1_g06_quant_chunk4(y1, wv, l, 32u + wv * 4u, q, 72u);
      quant_wave(y2, 64u + wv, q, 72u);
    }
    return;
  }

  /* Small-dim fast path (n <= 256, e.g. hd=128): single pass in registers, 1 lbarrier(). */
  if (n <= NT) {
    const float xi = (th < n) ? x[th] : 0.0f;
    const float wi = (th < n) ? w[th] : 0.0f;
    const float ss = ink_r1_g06_wg_sum_1b(xi * xi, wv, l);
    const float s = 1.0f / __builtin_sqrtf(ss / (float)n + eps);
    if (th < n) {
      const float v = xi * s * wi;
      if (y) y[th] = v;
      if (q) quant_wave(v, wv, q, nb);
    }
    return;
  }

  /* General path for any (n & 31u) == 0u: 1024-float f32x4 chunks + coalesced tail, 1 lbarrier(). */
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
  ss = ink_r1_g06_wg_sum_1b(ss, wv, l);
  const float s = 1.0f / __builtin_sqrtf(ss / (float)n + eps);

  for (u32 c = 0u; c < k; c++) {
    const u32 base = c * 1024u + th * 4u;
    const f32x4 y4 = *(const G f32x4 *)(x + base) * s * *(const G f32x4 *)(w + base);
    if (y) *(G f32x4 *)(y + base) = y4;
    if (q) ink_r1_g06_quant_chunk4(y4, wv, l, c * 32u + wv * 4u, q, nb);
  }
  for (u32 b = k * 32u + wv; b < nb; b += 8u) {
    const u32 i = b * 32u + l;
    const float v = x[i] * s * w[i];
    if (y) y[i] = v;
    if (q) quant_wave(v, b, q, nb);
  }
}

#ifdef IE_MEGA
#undef ink_r1_g06_ar_red
#undef ink_r1_g06_ar_inv
#undef ink_r1_g06_ar_prob
#undef ink_r1_g06_rn_red
#undef ink_r1_g06_rn_sh
#endif
#undef INK_R1_G06_AR_MAXM
#undef INK_R1_G06_AR_MAXK
#undef INK_R1_G06_AR_SLOTS
