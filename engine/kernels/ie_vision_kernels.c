/* ie_vision_kernels.c -- gfx1201 (RDNA4, wave32) GPU kernels for Qwen3.5/3.8 ViT mmproj.
 * Standalone code object (build/ie_vision_kernels.hsaco), leaving base LLM/Saluki hsacos untouched.
 * All kernels launch with 256 threads per workgroup (8 waves of 32). */
#include <stdbool.h>
#include <stdint.h>

#define G __attribute__((address_space(1)))
#define LDS __attribute__((address_space(3), loader_uninitialized))
#define KERNEL __attribute__((amdgpu_kernel, visibility("protected"))) void
#define NT 256u

typedef unsigned int u32;
typedef unsigned char u8;
typedef unsigned short u16;
typedef signed char i8;
typedef _Float16 f16;

static inline u32 tid(void) { return __builtin_amdgcn_workitem_id_x(); }
static inline u32 wgid_x(void) { return __builtin_amdgcn_workgroup_id_x(); }
static inline u32 wgid_y(void) { return __builtin_amdgcn_workgroup_id_y(); }
static inline u32 lane(void) { return tid() & 31u; }

static inline void lbarrier(void) {
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  __builtin_amdgcn_s_barrier();
  __asm__ volatile("" ::: "memory");
}

static inline int xrow_i(int v) { return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false, false); }
static inline int shl_i(int v, int n) {
  switch (n) {
    case 8: return __builtin_amdgcn_update_dpp(0, v, 0x108, 0xF, 0xF, true);
    case 4: return __builtin_amdgcn_update_dpp(0, v, 0x104, 0xF, 0xF, true);
    case 2: return __builtin_amdgcn_update_dpp(0, v, 0x102, 0xF, 0xF, true);
    default: return __builtin_amdgcn_update_dpp(0, v, 0x101, 0xF, 0xF, true);
  }
}
#define F2I(x) __builtin_bit_cast(int, (x))
#define I2F(x) __builtin_bit_cast(float, (x))

static inline float wave_sum_all(float v) {
  v += I2F(xrow_i(F2I(v)));
  v += I2F(shl_i(F2I(v), 8));
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return I2F(__builtin_amdgcn_readlane(F2I(v), 0));
}

static inline float shfl_xor_f(float v, u32 m) {
  return __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((int)((lane() ^ m) * 4u), __builtin_bit_cast(int, v)));
}
static inline float wave_max_all(float v) {
  v = __builtin_fmaxf(v, shfl_xor_f(v, 16u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 8u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 4u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 2u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 1u));
  return v;
}

static LDS float vis_red[8];
static inline float wg_sum(float v) {
  v = wave_sum_all(v);
  if (lane() == 0u) vis_red[tid() >> 5] = v;
  lbarrier();
  float s = 0.0f;
  for (u32 w = 0; w < 8u; w++) s += vis_red[w];
  lbarrier();
  return s;
}

static inline float vis_gelu(float x) {
  /* Tanh GELU matching GGML_UNARY_OP_GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
  const float z = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  float t;
  if (z > 10.0f) t = 1.0f;
  else if (z < -10.0f) t = -1.0f;
  else {
    const float e = __builtin_expf(-2.0f * __builtin_fabsf(z));
    const float a = (1.0f - e) / (1.0f + e);
    t = z < 0.0f ? -a : a;
  }
  return 0.5f * x * (1.0f + t);
}

/* 1. LayerNorm: y[n, d] = (x[n, d] - mu) / sqrt(var + eps) * w[d] + b[d]
 * Grid: (N, 1, 1), Block: (256, 1, 1). */
KERNEL ie_vis_layernorm(const G float *x, const G float *w, const G float *b, G float *y, u32 N, u32 D, float eps) {
  const u32 n = wgid_x();
  if (n >= N) return;
  const G float *xn = x + (unsigned long)n * D;
  G float *yn = y + (unsigned long)n * D;
  float s = 0.0f;
  for (u32 i = tid(); i < D; i += NT) s += xn[i];
  const float mu = wg_sum(s) / (float)D;
  float v = 0.0f;
  for (u32 i = tid(); i < D; i += NT) {
    const float d = xn[i] - mu;
    v += d * d;
  }
  const float inv = 1.0f / __builtin_sqrtf(wg_sum(v) / (float)D + eps);
  for (u32 i = tid(); i < D; i += NT) yn[i] = (xn[i] - mu) * inv * w[i] + b[i];
}

/* 2. Tiled F32 x Q8_0 GEMM: C[N, M] = act(A[N, K] @ W_q8[M, K]^T + bias[M]) + res[N, M]
 * K must be a multiple of 32. Raw GGUF Q8_0: 34 bytes/block (f16 d + 32 int8).
 * Grid: ((M + 63) / 64, (N + 63) / 64, 1), Block: (256, 1, 1). */
static LDS float gemm_A[64][33];
static LDS float gemm_B[64][33];

KERNEL ie_vis_gemm_q8(const G float *A, const G u8 *W, const G float *bias, const G float *res, G float *C,
                      u32 N, u32 M, u32 K, u32 act) {
  const u32 m0 = wgid_x() * 64u, n0 = wgid_y() * 64u;
  const u32 t = tid(), tx = t & 15u, ty = t >> 4;
  const u32 nb = K >> 5;
  float acc[4][4];
#pragma unroll
  for (u32 a = 0; a < 4u; a++)
#pragma unroll
    for (u32 b = 0; b < 4u; b++) acc[a][b] = 0.0f;

  for (u32 kb = 0; kb < nb; kb++) {
    const u32 k0 = kb << 5;
#pragma unroll
    for (u32 s = 0; s < 8u; s++) {
      const u32 idx = t + s * 256u;
      const u32 r = idx >> 5, c = idx & 31u;
      const u32 gn = n0 + r;
      gemm_A[r][c] = gn < N ? A[(unsigned long)gn * K + k0 + c] : 0.0f;
      const u32 gm = m0 + r;
      if (gm < M) {
        const G u8 *blk = W + ((unsigned long)gm * nb + kb) * 34ul;
        const float d = (float)(*(const G f16 *)blk);
        gemm_B[r][c] = d * (float)((const G i8 *)(blk + 2))[c];
      } else {
        gemm_B[r][c] = 0.0f;
      }
    }
    lbarrier();
#pragma unroll
    for (u32 k = 0; k < 32u; k++) {
      float va[4], vb[4];
#pragma unroll
      for (u32 a = 0; a < 4u; a++) va[a] = gemm_A[ty + a * 16u][k];
#pragma unroll
      for (u32 b = 0; b < 4u; b++) vb[b] = gemm_B[tx + b * 16u][k];
#pragma unroll
      for (u32 a = 0; a < 4u; a++)
#pragma unroll
        for (u32 b = 0; b < 4u; b++) acc[a][b] = __builtin_fmaf(va[a], vb[b], acc[a][b]);
    }
    lbarrier();
  }

#pragma unroll
  for (u32 a = 0; a < 4u; a++) {
    const u32 gn = n0 + ty + a * 16u;
    if (gn >= N) continue;
#pragma unroll
    for (u32 b = 0; b < 4u; b++) {
      const u32 gm = m0 + tx + b * 16u;
      if (gm >= M) continue;
      float v = acc[a][b];
      if (bias) v += bias[gm];
      if (act == 1u) v = vis_gelu(v);
      const unsigned long off = (unsigned long)gn * M + gm;
      if (res) v += res[off];
      C[off] = v;
    }
  }
}

/* 3. Tiled F32 x F16 GEMM: C[N, M] = act(A[N, K] @ W_f16[M, K]^T + bias[M]) + res[N, M]
 * K must be a multiple of 16 (used for v.blk.*.ffn_down: K=4304, M=1152).
 * Grid: ((M + 63) / 64, (N + 63) / 64, 1), Block: (256, 1, 1). */
static LDS float gemm16_A[64][17];
static LDS float gemm16_B[64][17];

KERNEL ie_vis_gemm_f16(const G float *A, const G f16 *W, const G float *bias, const G float *res, G float *C,
                       u32 N, u32 M, u32 K, u32 act) {
  const u32 m0 = wgid_x() * 64u, n0 = wgid_y() * 64u;
  const u32 t = tid(), tx = t & 15u, ty = t >> 4;
  float acc[4][4];
#pragma unroll
  for (u32 a = 0; a < 4u; a++)
#pragma unroll
    for (u32 b = 0; b < 4u; b++) acc[a][b] = 0.0f;

  for (u32 k0 = 0; k0 < K; k0 += 16u) {
#pragma unroll
    for (u32 s = 0; s < 4u; s++) {
      const u32 idx = t + s * 256u;
      const u32 r = idx >> 4, c = idx & 15u;
      const u32 gn = n0 + r;
      gemm16_A[r][c] = gn < N ? A[(unsigned long)gn * K + k0 + c] : 0.0f;
      const u32 gm = m0 + r;
      gemm16_B[r][c] = gm < M ? (float)W[(unsigned long)gm * K + k0 + c] : 0.0f;
    }
    lbarrier();
#pragma unroll
    for (u32 k = 0; k < 16u; k++) {
      float va[4], vb[4];
#pragma unroll
      for (u32 a = 0; a < 4u; a++) va[a] = gemm16_A[ty + a * 16u][k];
#pragma unroll
      for (u32 b = 0; b < 4u; b++) vb[b] = gemm16_B[tx + b * 16u][k];
#pragma unroll
      for (u32 a = 0; a < 4u; a++)
#pragma unroll
        for (u32 b = 0; b < 4u; b++) acc[a][b] = __builtin_fmaf(va[a], vb[b], acc[a][b]);
    }
    lbarrier();
  }

#pragma unroll
  for (u32 a = 0; a < 4u; a++) {
    const u32 gn = n0 + ty + a * 16u;
    if (gn >= N) continue;
#pragma unroll
    for (u32 b = 0; b < 4u; b++) {
      const u32 gm = m0 + tx + b * 16u;
      if (gm >= M) continue;
      float v = acc[a][b];
      if (bias) v += bias[gm];
      if (act == 1u) v = vis_gelu(v);
      const unsigned long off = (unsigned long)gn * M + gm;
      if (res) v += res[off];
      C[off] = v;
    }
  }
}

/* 3b. Tiled F32 x F32 GEMM: C[N, M] = A[N, K] @ W_f32[M, K]^T + bias[M] + res[N, M]
 * K must be a multiple of 16 (used for v.patch_embd: K=768, M=1152).
 * Grid: ((M + 63) / 64, (N + 63) / 64, 1), Block: (256, 1, 1). */
KERNEL ie_vis_gemm_f32(const G float *A, const G float *W, const G float *bias, const G float *res, G float *C,
                       u32 N, u32 M, u32 K) {
  const u32 m0 = wgid_x() * 64u, n0 = wgid_y() * 64u;
  const u32 t = tid(), tx = t & 15u, ty = t >> 4;
  float acc[4][4];
#pragma unroll
  for (u32 a = 0; a < 4u; a++)
#pragma unroll
    for (u32 b = 0; b < 4u; b++) acc[a][b] = 0.0f;

  for (u32 k0 = 0; k0 < K; k0 += 16u) {
#pragma unroll
    for (u32 s = 0; s < 4u; s++) {
      const u32 idx = t + s * 256u;
      const u32 r = idx >> 4, c = idx & 15u;
      const u32 gn = n0 + r;
      gemm16_A[r][c] = gn < N ? A[(unsigned long)gn * K + k0 + c] : 0.0f;
      const u32 gm = m0 + r;
      gemm16_B[r][c] = gm < M ? W[(unsigned long)gm * K + k0 + c] : 0.0f;
    }
    lbarrier();
#pragma unroll
    for (u32 k = 0; k < 16u; k++) {
      float va[4], vb[4];
#pragma unroll
      for (u32 a = 0; a < 4u; a++) va[a] = gemm16_A[ty + a * 16u][k];
#pragma unroll
      for (u32 b = 0; b < 4u; b++) vb[b] = gemm16_B[tx + b * 16u][k];
#pragma unroll
      for (u32 a = 0; a < 4u; a++)
#pragma unroll
        for (u32 b = 0; b < 4u; b++) acc[a][b] = __builtin_fmaf(va[a], vb[b], acc[a][b]);
    }
    lbarrier();
  }

#pragma unroll
  for (u32 a = 0; a < 4u; a++) {
    const u32 gn = n0 + ty + a * 16u;
    if (gn >= N) continue;
#pragma unroll
    for (u32 b = 0; b < 4u; b++) {
      const u32 gm = m0 + tx + b * 16u;
      if (gm >= M) continue;
      float v = acc[a][b];
      if (bias) v += bias[gm];
      const unsigned long off = (unsigned long)gn * M + gm;
      if (res) v += res[off];
      C[off] = v;
    }
  }
}

/* 4. 2D Vision RoPE in-place on Q and K slices of qkv[N, 3 * D] (D = n_head * hd).
 * rcos, rsin: [N, hd/2] precomputed for (y, x).
 * Grid: (N, 1, 1), Block: (256, 1, 1). */
KERNEL ie_vis_rope2d(G float *qkv, const G float *rcos, const G float *rsin, u32 N, u32 n_head, u32 hd) {
  const u32 n = wgid_x();
  if (n >= N) return;
  const u32 half = hd >> 1;
  const u32 D = n_head * hd;
  const u32 total = n_head * half;
  const G float *cs = rcos + (unsigned long)n * half;
  const G float *sn = rsin + (unsigned long)n * half;
  G float *qn = qkv + (unsigned long)n * 3u * D;
  G float *kn = qn + D;
  for (u32 idx = tid(); idx < total; idx += NT) {
    const u32 h = idx / half, d = idx % half;
    const float c = cs[d], s = sn[d];
    const u32 o0 = h * hd + d, o1 = o0 + half;
    const float q0 = qn[o0], q1 = qn[o1];
    qn[o0] = q0 * c - q1 * s;
    qn[o1] = q0 * s + q1 * c;
    const float k0 = kn[o0], k1 = kn[o1];
    kn[o0] = k0 * c - k1 * s;
    kn[o1] = k0 * s + k1 * c;
  }
}

/* 5. Bidirectional FlashAttention on qkv[N, 3 * D] -> out[N, D] (hd <= 96, 8 queries/WG).
 * Grid: ((N + 7) / 8, n_head, 1), Block: (256, 1, 1). */
static LDS float att_K[32][96];
static LDS float att_V[32][96];

KERNEL ie_vis_attn(const G float *qkv, G float *out, u32 N, u32 n_head, u32 hd, float scale) {
  const u32 w = tid() >> 5, l = lane();
  const u32 qi = wgid_x() * 8u + w;
  const u32 h = wgid_y();
  const u32 D = n_head * hd;
  const u32 D3 = 3u * D;

  float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f;
  if (qi < N) {
    const G float *qp = qkv + (unsigned long)qi * D3 + h * hd;
    if (l < hd) q0 = qp[l];
    if (l + 32u < hd) q1 = qp[l + 32u];
    if (l + 64u < hd) q2 = qp[l + 64u];
  }

  float m = -__builtin_inff(), ls = 0.0f;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;

  for (u32 k0 = 0; k0 < N; k0 += 32u) {
    const u32 bk = (k0 + 32u <= N) ? 32u : (N - k0);
    const u32 tot = bk * hd;
    for (u32 idx = tid(); idx < tot; idx += NT) {
      const u32 j = idx / hd, d = idx % hd;
      const G float *kp = qkv + (unsigned long)(k0 + j) * D3 + D + h * hd;
      att_K[j][d] = kp[d];
      att_V[j][d] = kp[D + d];
    }
    lbarrier();
    if (qi < N) {
      /* Compute 32 Q*K scores into one per-lane register, then lane-parallel softmax once per 32-key tile. */
      float sc_lane = -__builtin_inff();
      for (u32 j = 0; j < bk; j++) {
        float dot = 0.0f;
        if (l < hd) dot = __builtin_fmaf(q0, att_K[j][l], dot);
        if (l + 32u < hd) dot = __builtin_fmaf(q1, att_K[j][l + 32u], dot);
        if (l + 64u < hd) dot = __builtin_fmaf(q2, att_K[j][l + 64u], dot);
        const float sc = wave_sum_all(dot) * scale;
        if (l == j) sc_lane = sc;
      }
      const float m_blk = wave_max_all(sc_lane);
      const float mn = __builtin_fmaxf(m, m_blk);
      const float alpha = m == -__builtin_inff() ? 0.0f : __builtin_expf(m - mn);
      const float p_lane = l < bk ? __builtin_expf(sc_lane - mn) : 0.0f;
      ls = ls * alpha + wave_sum_all(p_lane);
      a0 *= alpha, a1 *= alpha, a2 *= alpha;
      for (u32 j = 0; j < bk; j++) {
        const float p = I2F(__builtin_amdgcn_readlane(F2I(p_lane), j));
        if (l < hd) a0 = __builtin_fmaf(p, att_V[j][l], a0);
        if (l + 32u < hd) a1 = __builtin_fmaf(p, att_V[j][l + 32u], a1);
        if (l + 64u < hd) a2 = __builtin_fmaf(p, att_V[j][l + 64u], a2);
      }
      m = mn;
    }
    lbarrier();
  }

  if (qi < N) {
    const float inv = 1.0f / ls;
    G float *op = out + (unsigned long)qi * D + h * hd;
    if (l < hd) op[l] = a0 * inv;
    if (l + 32u < hd) op[l + 32u] = a1 * inv;
    if (l + 64u < hd) op[l + 64u] = a2 * inv;
  }
}

#include "ie_vision_wmma_r14.inc"
