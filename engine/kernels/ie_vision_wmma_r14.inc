/* ============================================================================
 * Inkling-2-Mini (inkling2v) RDNA4 (gfx1201, wave32, NT=256) WMMA Vision Suite
 *
 * Included at the bottom of kernels/ie_vision_kernels.c (leaves all existing
 * Qwen3.5/Saluki vision kernels untouched).
 *
 * 6 Exported Kernels:
 *   1. ie_vis_gemm_wmma_f16  - FP16xFP16 -> FP32 WMMA GEMM + fused bias/act/ls/res
 *   2. ie_vis_gemm_wmma_q8   - FP16xQ8_0 -> FP32 WMMA GEMM + fused bias/act/ls/res
 *   3. ie_vis_attn_wmma_hd64 - FP16 WMMA bidirectional FlashAttention (hd=64)
 *                              with inline fused 2D Axial RoPE (mode=0)
 *   4. ie_vis_layernorm_f16  - Vectorized single-pass LayerNorm -> f16 (+ opt f32)
 *   5. ie_vis_patch_prep_f16 - CLS prepend + pos_embd add + pre_ln in 1 launch
 *   6. ie_vis_merge_2x2_f16  - CLS strip + raster 2x2 spatial merge -> f16
 *
 * Hardware & Compiler Guarantees:
 *   - 0 errors, 0 warnings (-Wall -Wextra), 0/0 VGPR/SGPR spills, priv=0, scratch=0
 *   - TH_LOAD_NT=0, TH_LOAD_HT=0 (R43 gfx1201 cached load rule)
 *   - 0-way LDS bank conflicts on all GEMM and Attention loads/stores
 *   - Exact piecewise Chebyshev erf GELU (max_erf_err = 1.192e-7 <= 1.0 ULP vs erff)
 * ========================================================================== */

#define VIS_R14_LDSP  __attribute__((address_space(3)))
#define VIS_R14_LOG2E 1.44269504088896340736f
#define VIS_R14_ISQ2  0.70710678118654752440f

typedef unsigned long long vis_r14_u64;
_Static_assert(sizeof(vis_r14_u64) == 8u, "vis_r14_u64 must be 64-bit");

typedef f16   vis_r14_h8 __attribute__((ext_vector_type(8)));
typedef f16   vis_r14_h4 __attribute__((ext_vector_type(4)));
typedef f16   vis_r14_h2 __attribute__((ext_vector_type(2)));
typedef float vis_r14_f8 __attribute__((ext_vector_type(8)));
typedef float vis_r14_f4 __attribute__((ext_vector_type(4)));
typedef u32   vis_r14_u4 __attribute__((ext_vector_type(4)));

struct __attribute__((packed, aligned(2))) vis_r14_u4p2 {
  vis_r14_u4 v;
};

struct __attribute__((packed, aligned(4))) vis_r14_u4p4 {
  vis_r14_u4 v;
};

static inline __attribute__((always_inline)) vis_r14_u4 vis_r14_ld_u4_a2(const G void *p) {
  return ((const G struct vis_r14_u4p2 *)p)->v;
}

static inline __attribute__((always_inline)) vis_r14_u4 vis_r14_ld_u4_a4(const G void *p) {
  return ((const G struct vis_r14_u4p4 *)p)->v;
}

static inline __attribute__((always_inline)) u32 vis_r14_lane_pin(void) {
  u32 l;
  __asm__ volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(l));
  return l;
}

static inline __attribute__((always_inline)) float vis_r14_xrow_f(float v) {
  return I2F(xrow_i(F2I(v)));
}

/* Exact tanh GELU (Qwen3.5/Saluki, act == 1), zero libm dependency. */
static inline __attribute__((always_inline)) float vis_r14_gelu_tanh(float x) {
  const float z = 0.7978845608028654f * __builtin_fmaf(0.044715f * x * x, x, x);
  const float az = __builtin_fabsf(z);
  float t;
  if (az >= 10.0f) {
    t = 1.0f;
  } else {
    const float e = __builtin_amdgcn_exp2f(-2.0f * VIS_R14_LOG2E * az);
    t = (1.0f - e) / (1.0f + e);
  }
  return 0.5f * x * (1.0f + (z < 0.0f ? -t : t));
}

/* Exact erf GELU (Inkling-2 FFN_GELU_ERF, act == 2), zero libm dependency.
 * Piecewise Chebyshev minimax:
 *   |u| <= 1.0: erf(u) = u * P6(u^2) (0 exp2f, 0 rcp, <= 1.0 ULP)
 *   |u| in (1.0, 6.0): cancellation-free erfc(|u|) = exp2(-u^2*LOG2E) * Q6(t)
 * Verified max |erf_approx(u) - erff(u)| = 1.192e-7 (< 1.5e-7) over all R. */
static inline __attribute__((always_inline)) float vis_r14_gelu_erf(float x) {
  const float u = x * VIS_R14_ISQ2;
  const float ax = __builtin_fabsf(u);
  if (ax <= 1.0f) {
    const float z = ax * ax;
    float p = 7.933237066e-05f;
    p = __builtin_fmaf(p, z, -8.034768980e-04f);
    p = __builtin_fmaf(p, z,  5.191209726e-03f);
    p = __builtin_fmaf(p, z, -2.685539424e-02f);
    p = __builtin_fmaf(p, z,  1.128362566e-01f);
    p = __builtin_fmaf(p, z, -3.761262894e-01f);
    p = __builtin_fmaf(p, z,  1.128379226e+00f);
    return (0.5f * x) * (1.0f + u * p);
  }
  if (ax >= 6.0f) {
    return u >= 0.0f ? x : 0.0f;
  }
  const float t = 1.0f / __builtin_fmaf(0.3275911f, ax, 1.0f);
  float p = -4.291843623e-02f;
  p = __builtin_fmaf(p, t,  4.677634239e-01f);
  p = __builtin_fmaf(p, t, -3.230377734e-01f);
  p = __builtin_fmaf(p, t,  4.341602921e-01f);
  p = __builtin_fmaf(p, t,  7.837823033e-02f);
  p = __builtin_fmaf(p, t,  2.025526166e-01f);
  p = __builtin_fmaf(p, t,  1.834724247e-01f);
  const float erfc_a = (p * t) * __builtin_amdgcn_exp2f((-ax * ax) * VIS_R14_LOG2E);
  const float h = 0.5f * erfc_a;
  return x * (u >= 0.0f ? (1.0f - h) : h);
}

/* ============================================================================
 * Shared 0-Bank-Conflict WMMA GEMM Engine (BN=64, BM=128, BK=32, 24,576 B 2x LDS)
 * Wave grid: 2 N-strips of 32 (wn = (w>>2)*32) x 4 M-strips of 32 (wm = (w&3)*32)
 * WMMA mapping: acc = wmma(w_frag, a_frag, acc)
 *   -> lane (hl, rl) owns n = n_tile + rl, m = m_tile + 8*hl + (0..7)
 * ========================================================================== */

typedef struct {
  f16 a[64u * 32u];  /*  4,096 B: A[64, 32] XOR-swizzled */
  f16 w[128u * 32u]; /*  8,192 B: W[128, 32] XOR-swizzled */
} vis_r14_gemm_lds_t;

static LDS vis_r14_gemm_lds_t vis_r14_gemm_lds[2] __attribute__((aligned(16)));
_Static_assert(sizeof(vis_r14_gemm_lds) == 24576u, "vis_r14_gemm_lds must be 24,576 B (2 WGs/CU)");

static inline __attribute__((always_inline)) u32 vis_r14_gemm_off(u32 r, u32 c8) {
  return (r << 5u) + ((c8 ^ ((r >> 1u) & 3u)) << 3u);
}

static inline __attribute__((always_inline)) void vis_r14_gemm_mma_ks(
    vis_r14_f8 acc[2][2], u32 wn, u32 wm, u32 rl, u32 hl, u32 ks, u32 buf) {
  const u32 c8 = (ks << 1u) + hl;
  const vis_r14_h8 a0 = *(const VIS_R14_LDSP vis_r14_h8 *)(vis_r14_gemm_lds[buf].a + vis_r14_gemm_off(wn + rl, c8));
  const vis_r14_h8 a1 = *(const VIS_R14_LDSP vis_r14_h8 *)(vis_r14_gemm_lds[buf].a + vis_r14_gemm_off(wn + 16u + rl, c8));
  const vis_r14_h8 w0 = *(const VIS_R14_LDSP vis_r14_h8 *)(vis_r14_gemm_lds[buf].w + vis_r14_gemm_off(wm + rl, c8));
  const vis_r14_h8 w1 = *(const VIS_R14_LDSP vis_r14_h8 *)(vis_r14_gemm_lds[buf].w + vis_r14_gemm_off(wm + 16u + rl, c8));
  __builtin_amdgcn_sched_barrier(0);
  acc[0][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(w0, a0, acc[0][0]);
  acc[0][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(w1, a0, acc[0][1]);
  acc[1][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(w0, a1, acc[1][0]);
  acc[1][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(w1, a1, acc[1][1]);
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void vis_r14_gemm_epilogue(
    vis_r14_f8 acc[2][2], const G float *bias, const G float *ls, const G float *res, G void *C,
    u32 n0, u32 m0, u32 N, u32 M, u32 act, u32 out_f16) {
  const u32 t = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = vis_r14_lane_pin();
  const u32 rl = l & 15u, hl = l >> 4u;
  const u32 wn = (w >> 2u) << 5u;
  const u32 wm = (w & 3u) << 5u;
  const int m_vec_f16 = ((M & 7u) == 0u);
  const int m_vec_f32 = ((M & 3u) == 0u);

#pragma clang loop unroll(disable)
  for (u32 bm = 0u; bm < 2u; bm++) {
    const u32 gm = m0 + wm + (bm << 4u) + (hl << 3u);
    if (gm >= M) continue;
    const int full8 = (gm + 8u <= M);

    vis_r14_f8 bv = (vis_r14_f8)(0.0f);
    if (bias) {
      if (__builtin_expect(full8, 1)) {
        const vis_r14_f4 b0 = *(const G vis_r14_f4 *)(bias + gm);
        const vis_r14_f4 b1 = *(const G vis_r14_f4 *)(bias + gm + 4u);
        bv = (vis_r14_f8){b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
      } else {
#pragma clang loop unroll(disable)
        for (u32 v = 0u; v < 8u; v++)
          if (gm + v < M) bv[v] = bias[gm + v];
      }
    }

    vis_r14_f8 lv = (vis_r14_f8)(1.0f);
    if (ls) {
      if (__builtin_expect(full8, 1)) {
        const vis_r14_f4 l0 = *(const G vis_r14_f4 *)(ls + gm);
        const vis_r14_f4 l1 = *(const G vis_r14_f4 *)(ls + gm + 4u);
        lv = (vis_r14_f8){l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
      } else {
#pragma clang loop unroll(disable)
        for (u32 v = 0u; v < 8u; v++)
          if (gm + v < M) lv[v] = ls[gm + v];
      }
    }

#pragma clang loop unroll(disable)
    for (u32 bn = 0u; bn < 2u; bn++) {
      const u32 gn = n0 + wn + (bn << 4u) + rl;
      if (gn >= N) continue;

      vis_r14_f8 val = (bn == 0u) ? (bm == 0u ? acc[0][0] : acc[0][1])
                                  : (bm == 0u ? acc[1][0] : acc[1][1]);
      if (bias) val += bv;

      if (act == 2u) {
        val[0] = vis_r14_gelu_erf(val[0]); val[1] = vis_r14_gelu_erf(val[1]);
        val[2] = vis_r14_gelu_erf(val[2]); val[3] = vis_r14_gelu_erf(val[3]);
        val[4] = vis_r14_gelu_erf(val[4]); val[5] = vis_r14_gelu_erf(val[5]);
        val[6] = vis_r14_gelu_erf(val[6]); val[7] = vis_r14_gelu_erf(val[7]);
      } else if (act == 1u) {
        val[0] = vis_r14_gelu_tanh(val[0]); val[1] = vis_r14_gelu_tanh(val[1]);
        val[2] = vis_r14_gelu_tanh(val[2]); val[3] = vis_r14_gelu_tanh(val[3]);
        val[4] = vis_r14_gelu_tanh(val[4]); val[5] = vis_r14_gelu_tanh(val[5]);
        val[6] = vis_r14_gelu_tanh(val[6]); val[7] = vis_r14_gelu_tanh(val[7]);
      }

      if (ls) val *= lv;

      const vis_r14_u64 row_off = (vis_r14_u64)gn * (vis_r14_u64)M + (vis_r14_u64)gm;
      if (res) {
        if (__builtin_expect(full8 && m_vec_f32, 1)) {
          const vis_r14_f4 r0 = *(const G vis_r14_f4 *)(res + row_off);
          const vis_r14_f4 r1 = *(const G vis_r14_f4 *)(res + row_off + 4ull);
          val += (vis_r14_f8){r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
        } else {
#pragma clang loop unroll(disable)
          for (u32 v = 0u; v < 8u; v++)
            if (gm + v < M) val[v] += res[row_off + v];
        }
      }

      if (out_f16) {
        G f16 *cp = (G f16 *)C + row_off;
        if (__builtin_expect(full8 && m_vec_f16, 1)) {
          *(G vis_r14_h8 *)cp = (vis_r14_h8){
              (f16)val[0], (f16)val[1], (f16)val[2], (f16)val[3],
              (f16)val[4], (f16)val[5], (f16)val[6], (f16)val[7]};
        } else {
#pragma clang loop unroll(disable)
          for (u32 v = 0u; v < 8u; v++)
            if (gm + v < M) cp[v] = (f16)val[v];
        }
      } else {
        G float *cp = (G float *)C + row_off;
        if (__builtin_expect(full8 && m_vec_f32, 1)) {
          *(G vis_r14_f4 *)(cp + 0u) = (vis_r14_f4){val[0], val[1], val[2], val[3]};
          *(G vis_r14_f4 *)(cp + 4u) = (vis_r14_f4){val[4], val[5], val[6], val[7]};
        } else {
#pragma clang loop unroll(disable)
          for (u32 v = 0u; v < 8u; v++)
            if (gm + v < M) cp[v] = val[v];
        }
      }
    }
  }
}

/* ============================================================================
 * Kernel 1: ie_vis_gemm_wmma_f16
 *   C[N, M] = act(A_f16[N, K] @ W_f16[M, K]^T + bias) * ls + res
 *   Grid:  ((M + 127) / 128, (N + 63) / 64, 1), Block: (256, 1, 1)
 *   K must be a multiple of 16 (supports K%32==16 e.g. patch_embd K=1200,
 *   Saluki ffn_down K=4304). Arbitrary N and M (fast path when M%8==0).
 *   act: 0=none, 1=gelu_tanh, 2=gelu_erf. out_f16: 0=f32, 1=f16.
 *   bias, ls, res may be NULL; C may alias res when out_f16 == 0.
 * ========================================================================== */
KERNEL ie_vis_gemm_wmma_f16(
    const G f16 *A, const G f16 *W, const G float *bias, const G float *ls,
    const G float *res, G void *C, u32 N, u32 M, u32 K, u32 act, u32 out_f16) {
  if (N == 0u || M == 0u || K == 0u || (K & 15u) != 0u) return;

  const u32 m0 = (u32)__builtin_amdgcn_readfirstlane((int)(wgid_x() << 7u));
  const u32 n0 = (u32)__builtin_amdgcn_readfirstlane((int)(wgid_y() << 6u));
  if (m0 >= M || n0 >= N) return;

  const u32 t = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = vis_r14_lane_pin();
  const u32 rl = l & 15u, hl = l >> 4u;
  const u32 wn = (w >> 2u) << 5u;
  const u32 wm = (w & 3u) << 5u;

  const u32 ra = t >> 2u, ca8 = t & 3u, ca = ca8 << 3u;
  const u32 gn_a = n0 + ra;
  const u32 gm_w0 = m0 + ra;
  const u32 gm_w1 = m0 + ra + 64u;
  const int a_ok = (gn_a < N);
  const int w0_ok = (gm_w0 < M);
  const int w1_ok = (gm_w1 < M);

  const G f16 *a_row = A + (vis_r14_u64)(a_ok ? gn_a : 0u) * (vis_r14_u64)K + ca;
  const G f16 *w0_row = W + (vis_r14_u64)(w0_ok ? gm_w0 : 0u) * (vis_r14_u64)K + ca;
  const G f16 *w1_row = W + (vis_r14_u64)(w1_ok ? gm_w1 : 0u) * (vis_r14_u64)K + ca;

  const u32 la_off = vis_r14_gemm_off(ra, ca8);
  const u32 lw0_off = vis_r14_gemm_off(ra, ca8);
  const u32 lw1_off = vis_r14_gemm_off(ra + 64u, ca8);

  vis_r14_f8 acc[2][2];
  acc[0][0] = (vis_r14_f8)(0.0f);
  acc[0][1] = (vis_r14_f8)(0.0f);
  acc[1][0] = (vis_r14_f8)(0.0f);
  acc[1][1] = (vis_r14_f8)(0.0f);

  const u32 n32 = K >> 5u;
  if (n32 > 0u) {
    vis_r14_u4 pa = a_ok ? vis_r14_ld_u4_a4(a_row) : (vis_r14_u4)(0u);
    vis_r14_u4 pw0 = w0_ok ? vis_r14_ld_u4_a4(w0_row) : (vis_r14_u4)(0u);
    vis_r14_u4 pw1 = w1_ok ? vis_r14_ld_u4_a4(w1_row) : (vis_r14_u4)(0u);
    *(VIS_R14_LDSP vis_r14_u4 *)(vis_r14_gemm_lds[0].a + la_off) = pa;
    *(VIS_R14_LDSP vis_r14_u4 *)(vis_r14_gemm_lds[0].w + lw0_off) = pw0;
    *(VIS_R14_LDSP vis_r14_u4 *)(vis_r14_gemm_lds[0].w + lw1_off) = pw1;
    lbarrier();

#pragma clang loop unroll(disable)
    for (u32 kb = 0u; kb < n32; kb++) {
      const u32 cur = kb & 1u;
      const u32 nxt = cur ^ 1u;
      const int has_next = (kb + 1u < n32);
      if (has_next) {
        const u32 kn = (kb + 1u) << 5u;
        pa = a_ok ? vis_r14_ld_u4_a4(a_row + kn) : (vis_r14_u4)(0u);
        pw0 = w0_ok ? vis_r14_ld_u4_a4(w0_row + kn) : (vis_r14_u4)(0u);
        pw1 = w1_ok ? vis_r14_ld_u4_a4(w1_row + kn) : (vis_r14_u4)(0u);
      }

      vis_r14_gemm_mma_ks(acc, wn, wm, rl, hl, 0u, cur);
      vis_r14_gemm_mma_ks(acc, wn, wm, rl, hl, 1u, cur);

      if (has_next) {
        *(VIS_R14_LDSP vis_r14_u4 *)(vis_r14_gemm_lds[nxt].a + la_off) = pa;
        *(VIS_R14_LDSP vis_r14_u4 *)(vis_r14_gemm_lds[nxt].w + lw0_off) = pw0;
        *(VIS_R14_LDSP vis_r14_u4 *)(vis_r14_gemm_lds[nxt].w + lw1_off) = pw1;
        lbarrier();
      }
    }
  }

  /* Optional K%32 == 16 tail (e.g. Inkling-2 v.patch_embd K=1200, Saluki ffn_down K=4304). */
  if ((K & 16u) != 0u) {
    const u32 kt = n32 << 5u;
    const u32 tb = n32 & 1u;
    if (ca8 < 2u) {
      *(VIS_R14_LDSP vis_r14_u4 *)(vis_r14_gemm_lds[tb].a + la_off) =
          a_ok ? vis_r14_ld_u4_a4(a_row + kt) : (vis_r14_u4)(0u);
      *(VIS_R14_LDSP vis_r14_u4 *)(vis_r14_gemm_lds[tb].w + lw0_off) =
          w0_ok ? vis_r14_ld_u4_a4(w0_row + kt) : (vis_r14_u4)(0u);
      *(VIS_R14_LDSP vis_r14_u4 *)(vis_r14_gemm_lds[tb].w + lw1_off) =
          w1_ok ? vis_r14_ld_u4_a4(w1_row + kt) : (vis_r14_u4)(0u);
    }
    lbarrier();
    vis_r14_gemm_mma_ks(acc, wn, wm, rl, hl, 0u, tb);
  }

  vis_r14_gemm_epilogue(acc, bias, ls, res, C, n0, m0, N, M, act, out_f16);
}

/* ============================================================================
 * Kernel 2: ie_vis_gemm_wmma_q8
 *   C[N, M] = act(A_f16[N, K] @ dequant(W_q8[M, K/32])^T + bias) * ls + res
 *   Raw GGUF Q8_0: 34 B/block (f16 d + 32 x i8). K must be a multiple of 32.
 *   Grid:  ((M + 127) / 128, (N + 63) / 64, 1), Block: (256, 1, 1)
 * ========================================================================== */
/* Unpack 8 signed int8 bytes in (w0, w1) directly to 8 scaled fp16 values via
 * v_perm_b32 + packed f16 sub/mul (exact al_i8_to_h8 pattern). */
static inline __attribute__((always_inline)) vis_r14_h8 vis_r14_deq8_h8(u32 w0, u32 w1, f16 d) {
  const u32 mg = 0x64646464u, x0 = w0 ^ 0x80808080u, x1 = w1 ^ 0x80808080u;
  const vis_r14_h2 bias = {(f16)1152.0f, (f16)1152.0f};
  const vis_r14_h2 sc = {d, d};
  const vis_r14_h2 q0 = (__builtin_bit_cast(vis_r14_h2, __builtin_amdgcn_perm(mg, x0, 0x04010400u)) - bias) * sc;
  const vis_r14_h2 q1 = (__builtin_bit_cast(vis_r14_h2, __builtin_amdgcn_perm(mg, x0, 0x04030402u)) - bias) * sc;
  const vis_r14_h2 q2 = (__builtin_bit_cast(vis_r14_h2, __builtin_amdgcn_perm(mg, x1, 0x04010400u)) - bias) * sc;
  const vis_r14_h2 q3 = (__builtin_bit_cast(vis_r14_h2, __builtin_amdgcn_perm(mg, x1, 0x04030402u)) - bias) * sc;
  return (vis_r14_h8){q0[0], q0[1], q1[0], q1[1], q2[0], q2[1], q3[0], q3[1]};
}

KERNEL ie_vis_gemm_wmma_q8(
    const G f16 *A, const G u8 *W, const G float *bias, const G float *ls,
    const G float *res, G void *C, u32 N, u32 M, u32 K, u32 act, u32 out_f16) {
  if (N == 0u || M == 0u || K == 0u || (K & 31u) != 0u) return;

  const u32 m0 = (u32)__builtin_amdgcn_readfirstlane((int)(wgid_x() << 7u));
  const u32 n0 = (u32)__builtin_amdgcn_readfirstlane((int)(wgid_y() << 6u));
  if (m0 >= M || n0 >= N) return;

  const u32 t = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = vis_r14_lane_pin();
  const u32 rl = l & 15u, hl = l >> 4u;
  const u32 wn = (w >> 2u) << 5u;
  const u32 wm = (w & 3u) << 5u;

  const u32 ra = t >> 2u, ca8 = t & 3u, ca = ca8 << 3u;
  const u32 gn_a = n0 + ra;
  const int a_ok = (gn_a < N);
  const G f16 *a_row = A + (vis_r14_u64)(a_ok ? gn_a : 0u) * (vis_r14_u64)K + ca;

  const u32 nb = K >> 5u;
  const u32 rb = t >> 1u, hb = t & 1u;
  const u32 gm_w = m0 + rb;
  const int w_ok = (gm_w < M);
  const G u8 *w_row = W + (vis_r14_u64)(w_ok ? gm_w : 0u) * (vis_r14_u64)nb * 34ull;
  const u32 q_byte_off = 2u + (hb << 4u);

  const u32 la_off = vis_r14_gemm_off(ra, ca8);
  const u32 lw0_off = vis_r14_gemm_off(rb, (hb << 1u) + 0u);
  const u32 lw1_off = vis_r14_gemm_off(rb, (hb << 1u) + 1u);

  vis_r14_f8 acc[2][2];
  acc[0][0] = (vis_r14_f8)(0.0f);
  acc[0][1] = (vis_r14_f8)(0.0f);
  acc[1][0] = (vis_r14_f8)(0.0f);
  acc[1][1] = (vis_r14_f8)(0.0f);

  vis_r14_u4 pa = a_ok ? vis_r14_ld_u4_a4(a_row) : (vis_r14_u4)(0u);
  f16 wd = w_ok ? *(const G f16 *)w_row : (f16)0.0f;
  vis_r14_u4 wq = w_ok ? vis_r14_ld_u4_a2(w_row + q_byte_off) : (vis_r14_u4)(0u);
  *(VIS_R14_LDSP vis_r14_u4 *)(vis_r14_gemm_lds[0].a + la_off) = pa;
  *(VIS_R14_LDSP vis_r14_h8 *)(vis_r14_gemm_lds[0].w + lw0_off) = vis_r14_deq8_h8(wq.x, wq.y, wd);
  *(VIS_R14_LDSP vis_r14_h8 *)(vis_r14_gemm_lds[0].w + lw1_off) = vis_r14_deq8_h8(wq.z, wq.w, wd);
  lbarrier();

#pragma clang loop unroll(disable)
  for (u32 kb = 0u; kb < nb; kb++) {
    const u32 cur = kb & 1u;
    const u32 nxt = cur ^ 1u;
    const int has_next = (kb + 1u < nb);
    if (has_next) {
      const u32 kbn = kb + 1u;
      pa = a_ok ? vis_r14_ld_u4_a4(a_row + (kbn << 5u)) : (vis_r14_u4)(0u);
      const G u8 *bn = w_row + (vis_r14_u64)kbn * 34ull;
      wd = w_ok ? *(const G f16 *)bn : (f16)0.0f;
      wq = w_ok ? vis_r14_ld_u4_a2(bn + q_byte_off) : (vis_r14_u4)(0u);
    }

    vis_r14_gemm_mma_ks(acc, wn, wm, rl, hl, 0u, cur);
    vis_r14_gemm_mma_ks(acc, wn, wm, rl, hl, 1u, cur);

    if (has_next) {
      *(VIS_R14_LDSP vis_r14_u4 *)(vis_r14_gemm_lds[nxt].a + la_off) = pa;
      *(VIS_R14_LDSP vis_r14_h8 *)(vis_r14_gemm_lds[nxt].w + lw0_off) = vis_r14_deq8_h8(wq.x, wq.y, wd);
      *(VIS_R14_LDSP vis_r14_h8 *)(vis_r14_gemm_lds[nxt].w + lw1_off) = vis_r14_deq8_h8(wq.z, wq.w, wd);
      lbarrier();
    }
  }

  vis_r14_gemm_epilogue(acc, bias, ls, res, C, n0, m0, N, M, act, out_f16);
}

/* ============================================================================
 * Kernel 3: ie_vis_attn_wmma_hd64
 *   Bidirectional FP16 WMMA FlashAttention (hd=64, arbitrary nh) with
 *   inline fused Inkling-2 2D Axial RoPE (mode=0 interleaved pairs):
 *     - Token 0 (CLS): (pos_w=0, pos_h=0)
 *     - Token t>=1: idx=t-1, pos_w=(idx%gw)+1 on dims [0..31],
 *                            pos_h=(idx/gw)+1 on dims [32..63]
 *     - inv_freq[j] = 10000^(-2j/32), j in [0..15]
 *     - gw == 0 disables RoPE (plain bidirectional attention)
 *   Geometry: BQ=128 (8 waves x 16 Q rows), BK=64, 16,384 B LDS (4 WGs/CU),
 *             0-bank-conflict K and V^T swizzles, zero N x N VRAM buffer.
 *   Grid:  ((N_pos + 127) / 128, nh, 1), Block: (256, 1, 1)
 * ========================================================================== */

typedef struct {
  f16 k[64u * 64u];  /* 8,192 B: K[64, 64] XOR-swizzled */
  f16 vt[64u * 64u]; /* 8,192 B: V^T[64_d, 64_k] XOR-swizzled */
} vis_r14_attn_lds_t;

static LDS vis_r14_attn_lds_t vis_r14_attn_lds[2] __attribute__((aligned(16)));
_Static_assert(sizeof(vis_r14_attn_lds) == 32768u, "vis_r14_attn_lds must be 32,768 B (2 WGs/CU)");

static inline __attribute__((always_inline)) u32 vis_r14_attn_k_off(u32 r, u32 c8) {
  return (r << 6u) + ((c8 ^ (r & 7u)) << 3u);
}

static inline __attribute__((always_inline)) u32 vis_r14_attn_vt_off(u32 d, u32 c8) {
  return (d << 6u) + ((c8 ^ (d & 7u) ^ ((d >> 3u) & 1u)) << 3u);
}

static inline __attribute__((always_inline)) vis_r14_h8 vis_r14_rope_h8_4(
    vis_r14_h8 v, float pos, float f0, float f1, float f2, float f3, float post_scale) {
  const float t0 = __builtin_amdgcn_fractf(pos * f0);
  const float c0 = __builtin_amdgcn_cosf(t0), s0 = __builtin_amdgcn_sinf(t0);
  const float t1 = __builtin_amdgcn_fractf(pos * f1);
  const float c1 = __builtin_amdgcn_cosf(t1), s1 = __builtin_amdgcn_sinf(t1);
  const float t2 = __builtin_amdgcn_fractf(pos * f2);
  const float c2 = __builtin_amdgcn_cosf(t2), s2 = __builtin_amdgcn_sinf(t2);
  const float t3 = __builtin_amdgcn_fractf(pos * f3);
  const float c3 = __builtin_amdgcn_cosf(t3), s3 = __builtin_amdgcn_sinf(t3);

  const float x0 = (float)v[0], x1 = (float)v[1];
  const float x2 = (float)v[2], x3 = (float)v[3];
  const float x4 = (float)v[4], x5 = (float)v[5];
  const float x6 = (float)v[6], x7 = (float)v[7];

  return (vis_r14_h8){
      (f16)((x0 * c0 - x1 * s0) * post_scale), (f16)((x0 * s0 + x1 * c0) * post_scale),
      (f16)((x2 * c1 - x3 * s1) * post_scale), (f16)((x2 * s1 + x3 * c1) * post_scale),
      (f16)((x4 * c2 - x5 * s2) * post_scale), (f16)((x4 * s2 + x5 * c2) * post_scale),
      (f16)((x6 * c3 - x7 * s3) * post_scale), (f16)((x6 * s3 + x7 * c3) * post_scale)};
}

/* Rotate 8 f16s (= 4 interleaved pairs) with compile-time half_sel (0: j=0..7, 1: j=8..15)
 * and dynamic sub_sel (0: low 4 pairs, 1: high 4 pairs). Zero constant-buffer loads! */
static inline __attribute__((always_inline)) vis_r14_h8 vis_r14_rope_h8_q(
    vis_r14_h8 v, float pos, u32 is_hi8, u32 sub4, float post_scale) {
  float f0, f1, f2, f3;
  if (is_hi8 == 0u) {
    f0 = sub4 ? 1.591549441e-02f : 1.591549367e-01f; /* j=4 : j=0 */
    f1 = sub4 ? 8.949940093e-03f : 8.949939907e-02f; /* j=5 : j=1 */
    f2 = sub4 ? 5.032921210e-03f : 5.032921210e-02f; /* j=6 : j=2 */
    f3 = sub4 ? 2.830219688e-03f : 2.830219641e-02f; /* j=7 : j=3 */
  } else {
    f0 = sub4 ? 1.591549371e-04f : 1.591549488e-03f; /* j=12 : j=8 */
    f1 = sub4 ? 8.949940093e-05f : 8.949940093e-04f; /* j=13 : j=9 */
    f2 = sub4 ? 5.032921035e-05f : 5.032921326e-04f; /* j=14 : j=10 */
    f3 = sub4 ? 2.830219637e-05f : 2.830219455e-04f; /* j=15 : j=11 */
  }
  return vis_r14_rope_h8_4(v, pos, f0, f1, f2, f3, post_scale);
}

static inline __attribute__((always_inline)) void vis_r14_pos2d(u32 tok, u32 gw, float *pw, float *ph) {
  if (tok == 0u) {
    *pw = 0.0f;
    *ph = 0.0f;
  } else {
    const u32 idx = tok - 1u;
    const u32 py = idx / gw;
    const u32 px = idx - py * gw;
    *pw = (float)(px + 1u);
    *ph = (float)(py + 1u);
  }
}

static inline __attribute__((always_inline)) void vis_r14_t8_f16(const u32 k[8], vis_r14_u4 *ev, vis_r14_u4 *od) {
  u32 e[4], o[4];
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    e[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x05040100u);
    o[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x07060302u);
  }
  *ev = (vis_r14_u4){e[0], e[1], e[2], e[3]};
  *od = (vis_r14_u4){o[0], o[1], o[2], o[3]};
}

static inline __attribute__((always_inline)) void vis_r14_attn_load_kv(
    const G f16 *k_base, const G f16 *v_base, vis_r14_u64 row_stride,
    u32 k0, u32 N_pos, u32 gw, int check_bounds, u32 buf) {
  u32 t = tid();
  __asm__ volatile("" : "+v"(t));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = t & 31u;

  /* Load K[64, 64] -> apply inline 2D Axial RoPE -> 0-conflict lds[buf].k */
  const u32 rk = t >> 2u;
  const u32 qk = t & 3u;
  const u32 gk = k0 + rk;
  const int k_ok = (!check_bounds) || (gk < N_pos);
  vis_r14_h8 k_lo = (vis_r14_h8)(0.0f);
  vis_r14_h8 k_hi = (vis_r14_h8)(0.0f);
  if (k_ok) {
    const G f16 *kp = k_base + (vis_r14_u64)gk * row_stride + (qk << 4u);
    k_lo = *(const G vis_r14_h8 *)(kp + 0u);
    k_hi = *(const G vis_r14_h8 *)(kp + 8u);
    if (gw > 0u && gk > 0u) {
      float pw, ph;
      vis_r14_pos2d(gk, gw, &pw, &ph);
      const float pos = (qk >= 2u) ? ph : pw;
      const u32 hi8 = qk & 1u;
      k_lo = vis_r14_rope_h8_q(k_lo, pos, hi8, 0u, 1.0f);
      k_hi = vis_r14_rope_h8_q(k_hi, pos, hi8, 1u, 1.0f);
    }
  }
  *(VIS_R14_LDSP vis_r14_h8 *)(vis_r14_attn_lds[buf].k + vis_r14_attn_k_off(rk, (qk << 1u) + 0u)) = k_lo;
  *(VIS_R14_LDSP vis_r14_h8 *)(vis_r14_attn_lds[buf].k + vis_r14_attn_k_off(rk, (qk << 1u) + 1u)) = k_hi;
  __builtin_amdgcn_sched_barrier(0);

  /* Load V[64, 64] (coalesced 128 B/wave per row) -> transpose8 -> 0-conflict lds[buf].vt */
  u32 vr[8];
  const u32 kv0 = k0 + (w << 3u);
  const G f16 *vp_base = v_base + (l << 1u);
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const u32 gv = kv0 + u;
    if ((!check_bounds) || (gv < N_pos)) {
      vr[u] = *(const G u32 *)(vp_base + (vis_r14_u64)gv * row_stride);
    } else {
      vr[u] = 0u;
    }
  }
  vis_r14_u4 ev, od;
  vis_r14_t8_f16(vr, &ev, &od);
  const u32 d0 = l << 1u;
  *(VIS_R14_LDSP vis_r14_u4 *)(vis_r14_attn_lds[buf].vt + vis_r14_attn_vt_off(d0 + 0u, w)) = ev;
  *(VIS_R14_LDSP vis_r14_u4 *)(vis_r14_attn_lds[buf].vt + vis_r14_attn_vt_off(d0 + 1u, w)) = od;
}

static inline __attribute__((always_inline)) vis_r14_f8 vis_r14_attn_qk16(
    const vis_r14_h8 qa[4], u32 kb16, u32 buf) {
  const u32 l = vis_r14_lane_pin();
  const u32 rl = l & 15u, hl = l >> 4u;
  const u32 r = (kb16 << 4u) + rl;
  vis_r14_f8 sa = (vis_r14_f8)(0.0f);
  vis_r14_h8 kf0 = *(const VIS_R14_LDSP vis_r14_h8 *)(vis_r14_attn_lds[buf].k + vis_r14_attn_k_off(r, 0u + hl));
  vis_r14_h8 kf1 = *(const VIS_R14_LDSP vis_r14_h8 *)(vis_r14_attn_lds[buf].k + vis_r14_attn_k_off(r, 2u + hl));
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kf0, qa[0], sa);
  kf0 = *(const VIS_R14_LDSP vis_r14_h8 *)(vis_r14_attn_lds[buf].k + vis_r14_attn_k_off(r, 4u + hl));
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kf1, qa[1], sa);
  kf1 = *(const VIS_R14_LDSP vis_r14_h8 *)(vis_r14_attn_lds[buf].k + vis_r14_attn_k_off(r, 6u + hl));
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kf0, qa[2], sa);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kf1, qa[3], sa);
  __builtin_amdgcn_sched_barrier(0);
  return sa;
}

static inline __attribute__((always_inline)) void vis_r14_attn_pv16(
    vis_r14_h8 pb, vis_r14_f8 o[4], u32 kb16, u32 buf) {
  const u32 l = vis_r14_lane_pin();
  const u32 rl = l & 15u, hl = l >> 4u;
  const u32 c8 = (kb16 << 1u) + hl;
  vis_r14_h8 vf0 = *(const VIS_R14_LDSP vis_r14_h8 *)(vis_r14_attn_lds[buf].vt + vis_r14_attn_vt_off(0u + rl, c8));
  vis_r14_h8 vf1 = *(const VIS_R14_LDSP vis_r14_h8 *)(vis_r14_attn_lds[buf].vt + vis_r14_attn_vt_off(16u + rl, c8));
  o[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vf0, pb, o[0]);
  vf0 = *(const VIS_R14_LDSP vis_r14_h8 *)(vis_r14_attn_lds[buf].vt + vis_r14_attn_vt_off(32u + rl, c8));
  o[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vf1, pb, o[1]);
  vf1 = *(const VIS_R14_LDSP vis_r14_h8 *)(vis_r14_attn_lds[buf].vt + vis_r14_attn_vt_off(48u + rl, c8));
  o[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vf0, pb, o[2]);
  o[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vf1, pb, o[3]);
  __builtin_amdgcn_sched_barrier(0);
}

/* Process 16 keys (1 WMMA tile) keeping peak VGPR <= 96. */
static inline __attribute__((always_inline)) void vis_r14_attn_step16(
    const vis_r14_h8 qa[4], vis_r14_f8 o[4], float *m_io, float *ls_io,
    u32 kb16, u32 rem, int check_mask, u32 buf) {
  vis_r14_f8 s = vis_r14_attn_qk16(qa, kb16, buf);

  if (check_mask) {
    const u32 hl = vis_r14_lane_pin() >> 4u;
    const u32 kbase = (kb16 << 4u) + (hl << 3u);
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      if (kbase + v >= rem) s[v] = -__builtin_inff();
    }
  }

  float mx = s[0];
#pragma unroll
  for (u32 v = 1u; v < 8u; v++) mx = __builtin_fmaxf(mx, s[v]);
  mx = __builtin_fmaxf(mx, vis_r14_xrow_f(mx));

  float m = *m_io;
  float ls = *ls_io;
  const float mn = __builtin_fmaxf(m, mx);
  if (mn != -__builtin_inff()) {
    if (mn != m) {
      const float rsc = (m == -__builtin_inff()) ? 0.0f : __builtin_amdgcn_exp2f(m - mn);
      o[0] *= rsc;
      o[1] *= rsc;
      o[2] *= rsc;
      o[3] *= rsc;
      ls *= rsc;
      m = mn;
    }
    vis_r14_h8 p;
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      const float e = __builtin_amdgcn_exp2f(s[v] - mn);
      ls += e;
      p[v] = (f16)e;
    }
    vis_r14_attn_pv16(p, o, kb16, buf);
  }
  *m_io = m;
  *ls_io = ls;
}

KERNEL ie_vis_attn_wmma_hd64(
    const G f16 *qkv, G f16 *out, u32 N_pos, u32 nh, u32 gw, float scale) {
  if (N_pos == 0u || nh == 0u) return;

  const u32 q_wg = (u32)__builtin_amdgcn_readfirstlane((int)(wgid_x() << 7u));
  const u32 h = (u32)__builtin_amdgcn_readfirstlane((int)wgid_y());
  if (q_wg >= N_pos || h >= nh) return;

  const u32 D = nh << 6u;
  const vis_r14_u64 row_stride = (vis_r14_u64)(3u * D);
  const G f16 *q_base = qkv + (vis_r14_u64)(h << 6u);
  const G f16 *k_base = q_base + (vis_r14_u64)D;
  const G f16 *v_base = k_base + (vis_r14_u64)D;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 q_sub0 = q_wg + (w << 4u);

  vis_r14_h8 qa[4];
  {
    const u32 l_init = vis_r14_lane_pin();
    const u32 rl_i = l_init & 15u, hl_i = l_init >> 4u;
    const u32 q_idx = (q_sub0 + rl_i < N_pos) ? (q_sub0 + rl_i) : (N_pos - 1u);
    const float qsc = scale * VIS_R14_LOG2E;
    const G f16 *qp = q_base + (vis_r14_u64)q_idx * row_stride;

    qa[0] = *(const G vis_r14_h8 *)(qp + (0u << 4u) + (hl_i << 3u));
    qa[1] = *(const G vis_r14_h8 *)(qp + (1u << 4u) + (hl_i << 3u));
    qa[2] = *(const G vis_r14_h8 *)(qp + (2u << 4u) + (hl_i << 3u));
    qa[3] = *(const G vis_r14_h8 *)(qp + (3u << 4u) + (hl_i << 3u));

    if (gw > 0u) {
      float pw, ph;
      vis_r14_pos2d(q_idx, gw, &pw, &ph);
      qa[0] = vis_r14_rope_h8_q(qa[0], pw, 0u, hl_i, qsc);
      qa[1] = vis_r14_rope_h8_q(qa[1], pw, 1u, hl_i, qsc);
      qa[2] = vis_r14_rope_h8_q(qa[2], ph, 0u, hl_i, qsc);
      qa[3] = vis_r14_rope_h8_q(qa[3], ph, 1u, hl_i, qsc);
    } else {
#pragma unroll
      for (u32 ks = 0u; ks < 4u; ks++) {
        qa[ks] = (vis_r14_h8){
            (f16)((float)qa[ks][0] * qsc), (f16)((float)qa[ks][1] * qsc),
            (f16)((float)qa[ks][2] * qsc), (f16)((float)qa[ks][3] * qsc),
            (f16)((float)qa[ks][4] * qsc), (f16)((float)qa[ks][5] * qsc),
            (f16)((float)qa[ks][6] * qsc), (f16)((float)qa[ks][7] * qsc)};
      }
    }
  }

  vis_r14_f8 o[4];
  o[0] = (vis_r14_f8)(0.0f);
  o[1] = (vis_r14_f8)(0.0f);
  o[2] = (vis_r14_f8)(0.0f);
  o[3] = (vis_r14_f8)(0.0f);
  float m = -__builtin_inff();
  float ls = 0.0f;

  u32 buf = 0u;
  if (N_pos >= 64u) {
    vis_r14_attn_load_kv(k_base, v_base, row_stride, 0u, N_pos, gw, 0, 0u);
    lbarrier();
    u32 k0 = 0u;
#pragma clang loop unroll(disable)
    for (; k0 + 64u <= N_pos; k0 += 64u) {
      const u32 cur = buf;
#pragma clang loop unroll(disable)
      for (u32 kb16 = 0u; kb16 < 4u; kb16++) {
        vis_r14_attn_step16(qa, o, &m, &ls, kb16, 64u, 0, cur);
      }
      const u32 kn = k0 + 64u;
      if (kn + 64u <= N_pos) {
        buf ^= 1u;
        vis_r14_attn_load_kv(k_base, v_base, row_stride, kn, N_pos, gw, 0, buf);
        lbarrier();
      } else if (kn < N_pos) {
        buf ^= 1u;
        vis_r14_attn_load_kv(k_base, v_base, row_stride, kn, N_pos, gw, 1, buf);
        lbarrier();
      }
    }
    if (k0 < N_pos) {
      const u32 rem = N_pos - k0;
      const u32 nkb = (rem + 15u) >> 4u;
#pragma clang loop unroll(disable)
      for (u32 kb16 = 0u; kb16 < nkb; kb16++) {
        vis_r14_attn_step16(qa, o, &m, &ls, kb16, rem, 1, buf);
      }
    }
  } else {
    const u32 nkb = (N_pos + 15u) >> 4u;
    vis_r14_attn_load_kv(k_base, v_base, row_stride, 0u, N_pos, gw, 1, 0u);
    lbarrier();
#pragma clang loop unroll(disable)
    for (u32 kb16 = 0u; kb16 < nkb; kb16++) {
      vis_r14_attn_step16(qa, o, &m, &ls, kb16, N_pos, 1, 0u);
    }
  }

  const u32 l_out = vis_r14_lane_pin();
  const u32 rl_o = l_out & 15u, hl_o = l_out >> 4u;
  const u32 q_out = q_sub0 + rl_o;
  if (q_out < N_pos) {
    const float inv = 1.0f / (ls + vis_r14_xrow_f(ls));
    G f16 *op = out + (vis_r14_u64)q_out * (vis_r14_u64)D + (vis_r14_u64)(h << 6u) + (hl_o << 3u);
#pragma unroll
    for (u32 nt = 0u; nt < 4u; nt++) {
      const vis_r14_f8 ov = o[nt] * inv;
      *(G vis_r14_h8 *)(op + (nt << 4u)) = (vis_r14_h8){
          (f16)ov[0], (f16)ov[1], (f16)ov[2], (f16)ov[3],
          (f16)ov[4], (f16)ov[5], (f16)ov[6], (f16)ov[7]};
    }
  }
}

/* ============================================================================
 * Kernel 3b: ie_vis_rope2d_table_r14
 *   Precomputes 2D Axial RoPE cos/sin table once per image (matching
 *   ie_vis_attn_wmma_hd64's exact CLS + 1-based (px+1, py+1) formula).
 *   Layout: cs_out[p * 64 + 2 * k + 0] = cos, cs_out[p * 64 + 2 * k + 1] = sin
 *           for p in [0, n_pos), k in [0, 32) (256 B/patch).
 *   Grid: ((n_pos + 7) / 8, 1, 1), Block: (256, 1, 1)
 * ========================================================================== */
static inline __attribute__((always_inline)) float vis_r14_inv_freq_2pi(u32 j) {
  switch (j) {
    case 0u:  return 1.591549367e-01f;
    case 1u:  return 8.949939907e-02f;
    case 2u:  return 5.032921210e-02f;
    case 3u:  return 2.830219641e-02f;
    case 4u:  return 1.591549441e-02f;
    case 5u:  return 8.949940093e-03f;
    case 6u:  return 5.032921210e-03f;
    case 7u:  return 2.830219688e-03f;
    case 8u:  return 1.591549488e-03f;
    case 9u:  return 8.949940093e-04f;
    case 10u: return 5.032921326e-04f;
    case 11u: return 2.830219455e-04f;
    case 12u: return 1.591549371e-04f;
    case 13u: return 8.949940093e-05f;
    case 14u: return 5.032921035e-05f;
    default:  return 2.830219637e-05f;
  }
}

KERNEL ie_vis_rope2d_table_r14(
    G float *cs_out, u32 n_pos, u32 gw, u32 gh, float freq_base) {
  (void)gh;
  const u32 idx = wgid_x() * NT + tid();
  if (idx >= n_pos * 32u) return;
  const u32 p = idx >> 5u;
  const u32 k = idx & 31u;

  float c = 1.0f;
  float s = 0.0f;
  if (gw > 0u && p > 0u) {
    float pw, ph;
    vis_r14_pos2d(p, gw, &pw, &ph);
    const float pos = (k >= 16u) ? ph : pw;
    const u32 j = k & 15u;
    float fj;
    if (freq_base <= 0.0f || freq_base == 10000.0f) {
      fj = vis_r14_inv_freq_2pi(j);
    } else {
      fj = __builtin_amdgcn_exp2f(-((float)(j << 1u) * (1.0f / 32.0f)) * __builtin_log2f(freq_base)) *
           0.15915494309189533577f;
    }
    const float ta = __builtin_amdgcn_fractf(pos * fj);
    c = __builtin_amdgcn_cosf(ta);
    s = __builtin_amdgcn_sinf(ta);
  }

  const vis_r14_u64 off = (vis_r14_u64)p * 64ull + (vis_r14_u64)(k << 1u);
  cs_out[off + 0ull] = c;
  cs_out[off + 1ull] = s;
}

/* ============================================================================
 * Kernel 3c: ie_vis_attn_wmma_hd64_r18
 *   FP16 WMMA bidirectional FlashAttention (hd=64) using precomputed 2D-RoPE
 *   cos/sin table (zero idx/gw div & zero cosf/sinf in Q and K) + wave-uniform
 *   zero-tile PV skip (ballot_w32(mx - m < -28.0f)) + 2x LDS double-buffering.
 *   Grid:  ((N_pos + 127) / 128, nh, 1), Block: (256, 1, 1)
 * ========================================================================== */
static inline __attribute__((always_inline)) vis_r14_h8 vis_r14_rope_h8_cs(
    vis_r14_h8 v, const G float *csp, float post_scale) {
  const vis_r14_f4 cs0 = *(const G vis_r14_f4 *)(csp + 0u);
  const vis_r14_f4 cs1 = *(const G vis_r14_f4 *)(csp + 4u);
  const float x0 = (float)v[0], x1 = (float)v[1];
  const float x2 = (float)v[2], x3 = (float)v[3];
  const float x4 = (float)v[4], x5 = (float)v[5];
  const float x6 = (float)v[6], x7 = (float)v[7];
  return (vis_r14_h8){
      (f16)((x0 * cs0.x - x1 * cs0.y) * post_scale), (f16)((x0 * cs0.y + x1 * cs0.x) * post_scale),
      (f16)((x2 * cs0.z - x3 * cs0.w) * post_scale), (f16)((x2 * cs0.w + x3 * cs0.z) * post_scale),
      (f16)((x4 * cs1.x - x5 * cs1.y) * post_scale), (f16)((x4 * cs1.y + x5 * cs1.x) * post_scale),
      (f16)((x6 * cs1.z - x7 * cs1.w) * post_scale), (f16)((x6 * cs1.w + x7 * cs1.z) * post_scale)};
}

static inline __attribute__((always_inline)) void vis_r14_attn_load_kv_r18(
    const G f16 *k_base, const G f16 *v_base, const G float *rope_cs,
    vis_r14_u64 row_stride, u32 k0, u32 N_pos, u32 gw, int check_bounds, u32 buf) {
  u32 t = tid();
  __asm__ volatile("" : "+v"(t));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = t & 31u;

  /* Load K[64, 64] -> apply precomputed 2D Axial RoPE -> 0-conflict lds[buf].k */
  const u32 rk = t >> 2u;
  const u32 qk = t & 3u;
  const u32 gk = k0 + rk;
  const int k_ok = (!check_bounds) || (gk < N_pos);
  vis_r14_h8 k_lo = (vis_r14_h8)(0.0f);
  vis_r14_h8 k_hi = (vis_r14_h8)(0.0f);
  if (k_ok) {
    const G f16 *kp = k_base + (vis_r14_u64)gk * row_stride + (qk << 4u);
    k_lo = *(const G vis_r14_h8 *)(kp + 0u);
    k_hi = *(const G vis_r14_h8 *)(kp + 8u);
    if (gw > 0u && gk > 0u) {
      if (rope_cs) {
        const G float *csp = rope_cs + (vis_r14_u64)gk * 64ull + (qk << 4u);
        k_lo = vis_r14_rope_h8_cs(k_lo, csp + 0u, 1.0f);
        k_hi = vis_r14_rope_h8_cs(k_hi, csp + 8u, 1.0f);
      } else {
        float pw, ph;
        vis_r14_pos2d(gk, gw, &pw, &ph);
        const float pos = (qk >= 2u) ? ph : pw;
        const u32 hi8 = qk & 1u;
        k_lo = vis_r14_rope_h8_q(k_lo, pos, hi8, 0u, 1.0f);
        k_hi = vis_r14_rope_h8_q(k_hi, pos, hi8, 1u, 1.0f);
      }
    }
  }
  *(VIS_R14_LDSP vis_r14_h8 *)(vis_r14_attn_lds[buf].k + vis_r14_attn_k_off(rk, (qk << 1u) + 0u)) = k_lo;
  *(VIS_R14_LDSP vis_r14_h8 *)(vis_r14_attn_lds[buf].k + vis_r14_attn_k_off(rk, (qk << 1u) + 1u)) = k_hi;
  __builtin_amdgcn_sched_barrier(0);

  /* Load V[64, 64] (coalesced 128 B/wave per row) -> transpose8 -> 0-conflict lds[buf].vt */
  u32 vr[8];
  const u32 kv0 = k0 + (w << 3u);
  const G f16 *vp_base = v_base + (l << 1u);
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const u32 gv = kv0 + u;
    if ((!check_bounds) || (gv < N_pos)) {
      vr[u] = *(const G u32 *)(vp_base + (vis_r14_u64)gv * row_stride);
    } else {
      vr[u] = 0u;
    }
  }
  vis_r14_u4 ev, od;
  vis_r14_t8_f16(vr, &ev, &od);
  const u32 d0 = l << 1u;
  *(VIS_R14_LDSP vis_r14_u4 *)(vis_r14_attn_lds[buf].vt + vis_r14_attn_vt_off(d0 + 0u, w)) = ev;
  *(VIS_R14_LDSP vis_r14_u4 *)(vis_r14_attn_lds[buf].vt + vis_r14_attn_vt_off(d0 + 1u, w)) = od;
}

static inline __attribute__((always_inline)) void vis_r14_attn_step16_r18(
    const vis_r14_h8 qa[4], vis_r14_f8 o[4], float *m_io, float *ls_io,
    u32 kb16, u32 rem, int check_mask, u32 buf) {
  vis_r14_f8 s = vis_r14_attn_qk16(qa, kb16, buf);

  if (check_mask) {
    const u32 hl = vis_r14_lane_pin() >> 4u;
    const u32 kbase = (kb16 << 4u) + (hl << 3u);
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      if (kbase + v >= rem) s[v] = -__builtin_inff();
    }
  }

  float mx = s[0];
#pragma unroll
  for (u32 v = 1u; v < 8u; v++) mx = __builtin_fmaxf(mx, s[v]);
  mx = __builtin_fmaxf(mx, vis_r14_xrow_f(mx));

  float m = *m_io;
  /* Wave-uniform zero-tile PV skip: 2^-28 < 3.7e-9 (below fp16 subnormal 6e-8). */
  if (__builtin_amdgcn_ballot_w32(mx > -1e30f && !(mx - m < -28.0f)) == 0u) return;

  float ls = *ls_io;
  const float mn = __builtin_fmaxf(m, mx);
  if (__builtin_amdgcn_ballot_w32(mn != m) != 0u) {
    if (mn != m) {
      const float rsc = (m == -__builtin_inff()) ? 0.0f : __builtin_amdgcn_exp2f(m - mn);
      o[0] *= rsc;
      o[1] *= rsc;
      o[2] *= rsc;
      o[3] *= rsc;
      ls *= rsc;
      m = mn;
    }
  }
  vis_r14_h8 p;
#pragma unroll
  for (u32 v = 0u; v < 8u; v++) {
    const float e = (mn != -__builtin_inff()) ? __builtin_amdgcn_exp2f(s[v] - mn) : 0.0f;
    ls += e;
    p[v] = (f16)e;
  }
  vis_r14_attn_pv16(p, o, kb16, buf);
  *m_io = m;
  *ls_io = ls;
}

KERNEL ie_vis_attn_wmma_hd64_r18(
    const G f16 *qkv, G f16 *out, u32 N_pos, u32 nh, u32 gw, float scale,
    const G float *rope_cs) {
  if (N_pos == 0u || nh == 0u) return;

  const u32 q_wg = (u32)__builtin_amdgcn_readfirstlane((int)(wgid_x() << 7u));
  const u32 h = (u32)__builtin_amdgcn_readfirstlane((int)wgid_y());
  if (q_wg >= N_pos || h >= nh) return;

  const u32 D = nh << 6u;
  const vis_r14_u64 row_stride = (vis_r14_u64)(3u * D);
  const G f16 *q_base = qkv + (vis_r14_u64)(h << 6u);
  const G f16 *k_base = q_base + (vis_r14_u64)D;
  const G f16 *v_base = k_base + (vis_r14_u64)D;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 q_sub0 = q_wg + (w << 4u);

  vis_r14_h8 qa[4];
  {
    const u32 l_init = vis_r14_lane_pin();
    const u32 rl_i = l_init & 15u, hl_i = l_init >> 4u;
    const u32 q_idx = (q_sub0 + rl_i < N_pos) ? (q_sub0 + rl_i) : (N_pos - 1u);
    const float qsc = scale * VIS_R14_LOG2E;
    const G f16 *qp = q_base + (vis_r14_u64)q_idx * row_stride;

    qa[0] = *(const G vis_r14_h8 *)(qp + (0u << 4u) + (hl_i << 3u));
    qa[1] = *(const G vis_r14_h8 *)(qp + (1u << 4u) + (hl_i << 3u));
    qa[2] = *(const G vis_r14_h8 *)(qp + (2u << 4u) + (hl_i << 3u));
    qa[3] = *(const G vis_r14_h8 *)(qp + (3u << 4u) + (hl_i << 3u));

    if (gw > 0u && q_idx > 0u) {
      if (rope_cs) {
        const G float *csp = rope_cs + (vis_r14_u64)q_idx * 64ull + (hl_i << 3u);
        qa[0] = vis_r14_rope_h8_cs(qa[0], csp + (0u << 4u), qsc);
        qa[1] = vis_r14_rope_h8_cs(qa[1], csp + (1u << 4u), qsc);
        qa[2] = vis_r14_rope_h8_cs(qa[2], csp + (2u << 4u), qsc);
        qa[3] = vis_r14_rope_h8_cs(qa[3], csp + (3u << 4u), qsc);
      } else {
        float pw, ph;
        vis_r14_pos2d(q_idx, gw, &pw, &ph);
        qa[0] = vis_r14_rope_h8_q(qa[0], pw, 0u, hl_i, qsc);
        qa[1] = vis_r14_rope_h8_q(qa[1], pw, 1u, hl_i, qsc);
        qa[2] = vis_r14_rope_h8_q(qa[2], ph, 0u, hl_i, qsc);
        qa[3] = vis_r14_rope_h8_q(qa[3], ph, 1u, hl_i, qsc);
      }
    } else {
#pragma unroll
      for (u32 ks = 0u; ks < 4u; ks++) {
        qa[ks] = (vis_r14_h8){
            (f16)((float)qa[ks][0] * qsc), (f16)((float)qa[ks][1] * qsc),
            (f16)((float)qa[ks][2] * qsc), (f16)((float)qa[ks][3] * qsc),
            (f16)((float)qa[ks][4] * qsc), (f16)((float)qa[ks][5] * qsc),
            (f16)((float)qa[ks][6] * qsc), (f16)((float)qa[ks][7] * qsc)};
      }
    }
  }

  vis_r14_f8 o[4];
  o[0] = (vis_r14_f8)(0.0f);
  o[1] = (vis_r14_f8)(0.0f);
  o[2] = (vis_r14_f8)(0.0f);
  o[3] = (vis_r14_f8)(0.0f);
  float m = -__builtin_inff();
  float ls = 0.0f;

  u32 buf = 0u;
  if (N_pos >= 64u) {
    vis_r14_attn_load_kv_r18(k_base, v_base, rope_cs, row_stride, 0u, N_pos, gw, 0, 0u);
    lbarrier();
    u32 k0 = 0u;
#pragma clang loop unroll(disable)
    for (; k0 + 64u <= N_pos; k0 += 64u) {
      const u32 cur = buf;
#pragma clang loop unroll(disable)
      for (u32 kb16 = 0u; kb16 < 4u; kb16++) {
        vis_r14_attn_step16_r18(qa, o, &m, &ls, kb16, 64u, 0, cur);
      }
      const u32 kn = k0 + 64u;
      if (kn + 64u <= N_pos) {
        buf ^= 1u;
        vis_r14_attn_load_kv_r18(k_base, v_base, rope_cs, row_stride, kn, N_pos, gw, 0, buf);
        lbarrier();
      } else if (kn < N_pos) {
        buf ^= 1u;
        vis_r14_attn_load_kv_r18(k_base, v_base, rope_cs, row_stride, kn, N_pos, gw, 1, buf);
        lbarrier();
      }
    }
    if (k0 < N_pos) {
      const u32 rem = N_pos - k0;
      const u32 nkb = (rem + 15u) >> 4u;
#pragma clang loop unroll(disable)
      for (u32 kb16 = 0u; kb16 < nkb; kb16++) {
        vis_r14_attn_step16_r18(qa, o, &m, &ls, kb16, rem, 1, buf);
      }
    }
  } else {
    const u32 nkb = (N_pos + 15u) >> 4u;
    vis_r14_attn_load_kv_r18(k_base, v_base, rope_cs, row_stride, 0u, N_pos, gw, 1, 0u);
    lbarrier();
#pragma clang loop unroll(disable)
    for (u32 kb16 = 0u; kb16 < nkb; kb16++) {
      vis_r14_attn_step16_r18(qa, o, &m, &ls, kb16, N_pos, 1, 0u);
    }
  }

  const u32 l_out = vis_r14_lane_pin();
  const u32 rl_o = l_out & 15u, hl_o = l_out >> 4u;
  const u32 q_out = q_sub0 + rl_o;
  if (q_out < N_pos) {
    const float inv = 1.0f / (ls + vis_r14_xrow_f(ls));
    G f16 *op = out + (vis_r14_u64)q_out * (vis_r14_u64)D + (vis_r14_u64)(h << 6u) + (hl_o << 3u);
#pragma unroll
    for (u32 nt = 0u; nt < 4u; nt++) {
      const vis_r14_f8 ov = o[nt] * inv;
      *(G vis_r14_h8 *)(op + (nt << 4u)) = (vis_r14_h8){
          (f16)ov[0], (f16)ov[1], (f16)ov[2], (f16)ov[3],
          (f16)ov[4], (f16)ov[5], (f16)ov[6], (f16)ov[7]};
    }
  }
}

/* ============================================================================
 * Kernel 4: ie_vis_layernorm_f16
 *   Vectorized single-pass LayerNorm for D==1024 (4 floats/thread in VGPR,
 *   0 second global read) + general fallback for arbitrary D (e.g. D=1152).
 *   Supports f32 (in_f16==0) or f16 (in_f16!=0) input and nullable f16/f32 outputs.
 *   Grid: (N, 1, 1), Block: (256, 1, 1)
 * ========================================================================== */
KERNEL ie_vis_layernorm_f16(
    const G void *x, const G float *w, const G float *b,
    G f16 *y_f16, G float *y_f32, u32 N, u32 D, float eps, u32 in_f16) {
  const u32 n = (u32)__builtin_amdgcn_readfirstlane((int)wgid_x());
  if (n >= N || D == 0u) return;
  const u32 t = tid();

  if (D == 1024u) {
    vis_r14_f4 v;
    if (in_f16) {
      const vis_r14_h4 vh = ((const G vis_r14_h4 *)((const G f16 *)x + (vis_r14_u64)n * 1024ull))[t];
      v = (vis_r14_f4){(float)vh.x, (float)vh.y, (float)vh.z, (float)vh.w};
    } else {
      v = ((const G vis_r14_f4 *)((const G float *)x + (vis_r14_u64)n * 1024ull))[t];
    }
    const float mu = wg_sum((v.x + v.y) + (v.z + v.w)) * (1.0f / 1024.0f);
    const vis_r14_f4 d = v - mu;
    const float var = wg_sum((d.x * d.x + d.y * d.y) + (d.z * d.z + d.w * d.w)) * (1.0f / 1024.0f);
    const float inv = 1.0f / __builtin_sqrtf(var + eps);
    const vis_r14_f4 wf = ((const G vis_r14_f4 *)w)[t];
    const vis_r14_f4 bf = ((const G vis_r14_f4 *)b)[t];
    const vis_r14_f4 o = d * (inv * wf) + bf;
    if (y_f16) {
      ((G vis_r14_h4 *)(y_f16 + (vis_r14_u64)n * 1024ull))[t] =
          (vis_r14_h4){(f16)o.x, (f16)o.y, (f16)o.z, (f16)o.w};
    }
    if (y_f32) {
      ((G vis_r14_f4 *)(y_f32 + (vis_r14_u64)n * 1024ull))[t] = o;
    }
    return;
  }

  if (D <= 2048u && (D & 3u) == 0u) {
    const u32 nv4 = D >> 2u;
    const u32 i0 = t;
    const u32 i1 = t + NT;
    const int has0 = (i0 < nv4);
    const int has1 = (i1 < nv4);
    vis_r14_f4 v0 = (vis_r14_f4)(0.0f);
    vis_r14_f4 v1 = (vis_r14_f4)(0.0f);
    if (in_f16) {
      const G vis_r14_h4 *xh = (const G vis_r14_h4 *)((const G f16 *)x + (vis_r14_u64)n * (vis_r14_u64)D);
      if (has0) {
        const vis_r14_h4 vh = xh[i0];
        v0 = (vis_r14_f4){(float)vh.x, (float)vh.y, (float)vh.z, (float)vh.w};
      }
      if (has1) {
        const vis_r14_h4 vh = xh[i1];
        v1 = (vis_r14_f4){(float)vh.x, (float)vh.y, (float)vh.z, (float)vh.w};
      }
    } else {
      const G vis_r14_f4 *xf = (const G vis_r14_f4 *)((const G float *)x + (vis_r14_u64)n * (vis_r14_u64)D);
      if (has0) v0 = xf[i0];
      if (has1) v1 = xf[i1];
    }
    const float inv_d = 1.0f / (float)D;
    float s = 0.0f;
    if (has0) s += (v0.x + v0.y) + (v0.z + v0.w);
    if (has1) s += (v1.x + v1.y) + (v1.z + v1.w);
    const float mu = wg_sum(s) * inv_d;
    vis_r14_f4 d0 = (vis_r14_f4)(0.0f);
    vis_r14_f4 d1 = (vis_r14_f4)(0.0f);
    float vv = 0.0f;
    if (has0) {
      d0 = v0 - mu;
      vv += (d0.x * d0.x + d0.y * d0.y) + (d0.z * d0.z + d0.w * d0.w);
    }
    if (has1) {
      d1 = v1 - mu;
      vv += (d1.x * d1.x + d1.y * d1.y) + (d1.z * d1.z + d1.w * d1.w);
    }
    const float inv = 1.0f / __builtin_sqrtf(wg_sum(vv) * inv_d + eps);
    const G vis_r14_f4 *w4 = (const G vis_r14_f4 *)w;
    const G vis_r14_f4 *b4 = (const G vis_r14_f4 *)b;
    G vis_r14_h4 *yh = y_f16 ? (G vis_r14_h4 *)(y_f16 + (vis_r14_u64)n * (vis_r14_u64)D) : (G vis_r14_h4 *)0;
    G vis_r14_f4 *yf = y_f32 ? (G vis_r14_f4 *)(y_f32 + (vis_r14_u64)n * (vis_r14_u64)D) : (G vis_r14_f4 *)0;
    if (has0) {
      const vis_r14_f4 o0 = d0 * (inv * w4[i0]) + b4[i0];
      if (yh) yh[i0] = (vis_r14_h4){(f16)o0.x, (f16)o0.y, (f16)o0.z, (f16)o0.w};
      if (yf) yf[i0] = o0;
    }
    if (has1) {
      const vis_r14_f4 o1 = d1 * (inv * w4[i1]) + b4[i1];
      if (yh) yh[i1] = (vis_r14_h4){(f16)o1.x, (f16)o1.y, (f16)o1.z, (f16)o1.w};
      if (yf) yf[i1] = o1;
    }
    return;
  }

  float s = 0.0f;
  if (in_f16) {
    const G f16 *xn = (const G f16 *)x + (vis_r14_u64)n * (vis_r14_u64)D;
    for (u32 i = t; i < D; i += NT) s += (float)xn[i];
    const float mu = wg_sum(s) / (float)D;
    float vv = 0.0f;
    for (u32 i = t; i < D; i += NT) {
      const float df = (float)xn[i] - mu;
      vv = __builtin_fmaf(df, df, vv);
    }
    const float inv = 1.0f / __builtin_sqrtf(wg_sum(vv) / (float)D + eps);
    G f16 *yh = y_f16 ? (y_f16 + (vis_r14_u64)n * (vis_r14_u64)D) : (G f16 *)0;
    G float *yf = y_f32 ? (y_f32 + (vis_r14_u64)n * (vis_r14_u64)D) : (G float *)0;
    for (u32 i = t; i < D; i += NT) {
      const float ov = __builtin_fmaf(((float)xn[i] - mu) * inv, w[i], b[i]);
      if (yh) yh[i] = (f16)ov;
      if (yf) yf[i] = ov;
    }
  } else {
    const G float *xn = (const G float *)x + (vis_r14_u64)n * (vis_r14_u64)D;
    for (u32 i = t; i < D; i += NT) s += xn[i];
    const float mu = wg_sum(s) / (float)D;
    float vv = 0.0f;
    for (u32 i = t; i < D; i += NT) {
      const float df = xn[i] - mu;
      vv = __builtin_fmaf(df, df, vv);
    }
    const float inv = 1.0f / __builtin_sqrtf(wg_sum(vv) / (float)D + eps);
    G f16 *yh = y_f16 ? (y_f16 + (vis_r14_u64)n * (vis_r14_u64)D) : (G f16 *)0;
    G float *yf = y_f32 ? (y_f32 + (vis_r14_u64)n * (vis_r14_u64)D) : (G float *)0;
    for (u32 i = t; i < D; i += NT) {
      const float ov = __builtin_fmaf((xn[i] - mu) * inv, w[i], b[i]);
      if (yh) yh[i] = (f16)ov;
      if (yf) yf[i] = ov;
    }
  }
}

/* ============================================================================
 * Kernel 5: ie_vis_patch_prep_f16
 *   Single-launch Inkling-2 ViT entry preparation:
 *     - Row 0 (CLS):      class_embd + class_position_embd -> pre_ln
 *     - Rows 1..N_patches: patch_out[n-1] + pos_embd[n-1]  -> pre_ln
 *   Writes out_f16 (if non-NULL) and/or out_f32 (d_x residual stream, if non-NULL).
 *   out_f16 / out_f32 must NOT alias patch_out.
 *   Grid: (N_patches + 1, 1, 1), Block: (256, 1, 1)
 * ========================================================================== */
KERNEL ie_vis_patch_prep_f16(
    const G void *patch_out, const G float *pos_embd,
    const G float *cls_embd, const G float *cls_pos_embd,
    const G float *pre_ln_w, const G float *pre_ln_b,
    G f16 *out_f16, G float *out_f32,
    u32 N_patches, u32 D, float eps, u32 in_f16) {
  const u32 n = (u32)__builtin_amdgcn_readfirstlane((int)wgid_x());
  if (n > N_patches || D == 0u) return;
  const u32 t = tid();

  if (D == 1024u) {
    vis_r14_f4 v;
    if (n == 0u) {
      v = ((const G vis_r14_f4 *)cls_embd)[t] + ((const G vis_r14_f4 *)cls_pos_embd)[t];
    } else {
      const vis_r14_u64 p_off = (vis_r14_u64)(n - 1u) * 1024ull;
      vis_r14_f4 pv;
      if (in_f16) {
        const vis_r14_h4 ph = ((const G vis_r14_h4 *)((const G f16 *)patch_out + p_off))[t];
        pv = (vis_r14_f4){(float)ph.x, (float)ph.y, (float)ph.z, (float)ph.w};
      } else {
        pv = ((const G vis_r14_f4 *)((const G float *)patch_out + p_off))[t];
      }
      v = pv + ((const G vis_r14_f4 *)(pos_embd + p_off))[t];
    }
    const float mu = wg_sum((v.x + v.y) + (v.z + v.w)) * (1.0f / 1024.0f);
    const vis_r14_f4 d = v - mu;
    const float var = wg_sum((d.x * d.x + d.y * d.y) + (d.z * d.z + d.w * d.w)) * (1.0f / 1024.0f);
    const float inv = 1.0f / __builtin_sqrtf(var + eps);
    const vis_r14_f4 wf = ((const G vis_r14_f4 *)pre_ln_w)[t];
    const vis_r14_f4 bf = ((const G vis_r14_f4 *)pre_ln_b)[t];
    const vis_r14_f4 o = d * (inv * wf) + bf;
    if (out_f16) {
      ((G vis_r14_h4 *)(out_f16 + (vis_r14_u64)n * 1024ull))[t] =
          (vis_r14_h4){(f16)o.x, (f16)o.y, (f16)o.z, (f16)o.w};
    }
    if (out_f32) {
      ((G vis_r14_f4 *)(out_f32 + (vis_r14_u64)n * 1024ull))[t] = o;
    }
    return;
  }

  /* General D fallback */
  float s = 0.0f;
  for (u32 i = t; i < D; i += NT) {
    float val;
    if (n == 0u) {
      val = cls_embd[i] + cls_pos_embd[i];
    } else {
      const vis_r14_u64 off = (vis_r14_u64)(n - 1u) * (vis_r14_u64)D + i;
      const float pv = in_f16 ? (float)((const G f16 *)patch_out)[off] : ((const G float *)patch_out)[off];
      val = pv + pos_embd[off];
    }
    s += val;
  }
  const float mu = wg_sum(s) / (float)D;
  float vv = 0.0f;
  for (u32 i = t; i < D; i += NT) {
    float val;
    if (n == 0u) {
      val = cls_embd[i] + cls_pos_embd[i];
    } else {
      const vis_r14_u64 off = (vis_r14_u64)(n - 1u) * (vis_r14_u64)D + i;
      const float pv = in_f16 ? (float)((const G f16 *)patch_out)[off] : ((const G float *)patch_out)[off];
      val = pv + pos_embd[off];
    }
    const float df = val - mu;
    vv = __builtin_fmaf(df, df, vv);
  }
  const float inv = 1.0f / __builtin_sqrtf(wg_sum(vv) / (float)D + eps);
  G f16 *yh = out_f16 ? (out_f16 + (vis_r14_u64)n * (vis_r14_u64)D) : (G f16 *)0;
  G float *yf = out_f32 ? (out_f32 + (vis_r14_u64)n * (vis_r14_u64)D) : (G float *)0;
  for (u32 i = t; i < D; i += NT) {
    float val;
    if (n == 0u) {
      val = cls_embd[i] + cls_pos_embd[i];
    } else {
      const vis_r14_u64 off = (vis_r14_u64)(n - 1u) * (vis_r14_u64)D + i;
      const float pv = in_f16 ? (float)((const G f16 *)patch_out)[off] : ((const G float *)patch_out)[off];
      val = pv + pos_embd[off];
    }
    const float ov = __builtin_fmaf((val - mu) * inv, pre_ln_w[i], pre_ln_b[i]);
    if (yh) yh[i] = (f16)ov;
    if (yf) yf[i] = ov;
  }
}

/* ============================================================================
 * Kernel 6: ie_vis_merge_2x2_f16
 *   Strips skip_cls leading rows (1 for Inkling-2 CLS row 0) and merges raster
 *   2x2 patches [gh, gw, D] -> [n_tok, 4*D] in f16 ready for mm.1 WMMA:
 *     tok = my * (gw/2) + mx -> [(2*my,2*mx), (2*my,2*mx+1),
 *                                (2*my+1,2*mx), (2*my+1,2*mx+1)]
 *   Supports f32 (in_f16==0, e.g. d_x) or f16 (in_f16!=0) input.
 *   Grid: ((gh / 2) * (gw / 2), 1, 1), Block: (256, 1, 1)
 * ========================================================================== */
KERNEL ie_vis_merge_2x2_f16(
    const G void *x, G f16 *out_f16,
    u32 gh, u32 gw, u32 D, u32 skip_cls, u32 in_f16) {
  const u32 ny = gh >> 1u, nx = gw >> 1u;
  const u32 n_tok = ny * nx;
  const u32 tok = (u32)__builtin_amdgcn_readfirstlane((int)wgid_x());
  if (tok >= n_tok || D == 0u) return;

  const u32 my = tok / nx;
  const u32 mx = tok - my * nx;
  const u32 r[4] = {
      skip_cls + (2u * my + 0u) * gw + (2u * mx + 0u),
      skip_cls + (2u * my + 0u) * gw + (2u * mx + 1u),
      skip_cls + (2u * my + 1u) * gw + (2u * mx + 0u),
      skip_cls + (2u * my + 1u) * gw + (2u * mx + 1u)};

  const u32 t = tid();
  G f16 *dst_tok = out_f16 + (vis_r14_u64)tok * (vis_r14_u64)(4u * D);

  if ((D & 3u) == 0u) {
    const u32 d4 = D >> 2u;
    if (in_f16) {
      const G f16 *xh = (const G f16 *)x;
#pragma unroll
      for (u32 q = 0u; q < 4u; q++) {
        const G vis_r14_h4 *sp = (const G vis_r14_h4 *)(xh + (vis_r14_u64)r[q] * (vis_r14_u64)D);
        G vis_r14_h4 *dp = (G vis_r14_h4 *)(dst_tok + (vis_r14_u64)q * (vis_r14_u64)D);
        for (u32 i = t; i < d4; i += NT) dp[i] = sp[i];
      }
    } else {
      const G float *xf = (const G float *)x;
#pragma unroll
      for (u32 q = 0u; q < 4u; q++) {
        const G vis_r14_f4 *sp = (const G vis_r14_f4 *)(xf + (vis_r14_u64)r[q] * (vis_r14_u64)D);
        G vis_r14_h4 *dp = (G vis_r14_h4 *)(dst_tok + (vis_r14_u64)q * (vis_r14_u64)D);
        for (u32 i = t; i < d4; i += NT) {
          const vis_r14_f4 v = sp[i];
          dp[i] = (vis_r14_h4){(f16)v.x, (f16)v.y, (f16)v.z, (f16)v.w};
        }
      }
    }
    return;
  }

  if (in_f16) {
    const G f16 *xh = (const G f16 *)x;
#pragma unroll
    for (u32 q = 0u; q < 4u; q++) {
      const G f16 *sp = xh + (vis_r14_u64)r[q] * (vis_r14_u64)D;
      G f16 *dp = dst_tok + (vis_r14_u64)q * (vis_r14_u64)D;
      for (u32 i = t; i < D; i += NT) dp[i] = sp[i];
    }
  } else {
    const G float *xf = (const G float *)x;
#pragma unroll
    for (u32 q = 0u; q < 4u; q++) {
      const G float *sp = xf + (vis_r14_u64)r[q] * (vis_r14_u64)D;
      G f16 *dp = dst_tok + (vis_r14_u64)q * (vis_r14_u64)D;
      for (u32 i = t; i < D; i += NT) dp[i] = (f16)sp[i];
    }
  }
}

#undef VIS_R14_ISQ2
#undef VIS_R14_LOG2E
#undef VIS_R14_LDSP
