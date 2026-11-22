/* Compile just the adapted Saluki kernels while reusing the base math. */
#define KERNEL static __attribute__((unused)) void
#define IE_KV_Q8
#define IE_SALUKI_ATTN_MP
#define IE_SALUKI_GDN1P
#include "ie_kernels.c"
#undef KERNEL
#define KERNEL __attribute__((amdgpu_kernel, visibility("protected"))) void

/* One wave per row, reusing the 32 weights across four tokens. Keep the
 * reduction and residual addition in the same order as ie_gemv_f32_tw. */
KERNEL ie_saluki_lora_b(const G float *w, const G float *x, G float *y, const G float *res,
                       u32 rows, u32 T, u32 xs, u32 ys, u32 rs) {
  const u32 r = ie_gemv_row(wgid(), tid() >> 5), l = lane(), t0 = tokid() * 4u;
  if (r >= rows) return;
  const float weight = w[(unsigned long)r * 32u + l];
#pragma unroll
  for (u32 t = 0; t < 4u; t++) {
    const u32 p = t0 + t;
    if (p < T) {
      const G float *xt = (const G float *)((const G u8 *)x + (unsigned long)p * xs);
      float value = 0.0f;
      value += weight * xt[l];
      value = wave_tree_f(value);
      if (l == 0u) {
        const G float *rt = (const G float *)((const G u8 *)res + (unsigned long)p * rs);
        ((G float *)((G u8 *)y + (unsigned long)p * ys))[r] = rt[r] + value;
      }
    }
  }
}

/* Eight tokens, four rows per wave, reusing the token activations between
 * four rows with the same per-row dot products and FP32 reductions. */
KERNEL ie_saluki_iq2xxs_tr8_r4(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                              const G float *bias, const G float *res, const G float *nw, G float *ny,
                              G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  if (T != 8u) return;
  tr4_xxs_grid[tid()] = TR8_G2C(ie_rq_iq2xxs_grid[tid()]);
  if (tid() < 128u) tr4_ks[tid()] = ie_rq_ksigns64[tid()];
  lbarrier();
  gemv_raw_tr8_run(IE_RQ_IQ2_XXS, 0, 8u, 4u, wgid(), q, xq, y, rows, nb, xs, ys, rs, bias, res);
}

static inline void saluki_iq1s_grid(void) {
  for (u32 i = tid(); i < 2048u; i += NT) tr8_g1s[i] = ie_rq_iq1s_grid[i];
  lbarrier();
}

/* Saluki's IQ1_S 16384x5120 and Q8_0 96x5120 projections share the
 * activation. Keep one launch and share each token's loads across four
 * IQ1_S rows per wave. The Q8 workgroups retain their original mapping. */
KERNEL ie_saluki_iq1s_q8_tr8(const G u8 *xq, u32 xs,
                             const G u8 *q0, G float *y0, u32 ys0, const G float *res0, u32 rs0, const G float *bias0,
                             const G u8 *q1, G float *y1, u32 ys1, const G float *res1, u32 rs1, const G u32 *qw1) {
  const u32 wg = wgid(), iq1_groups = 16384u / 32u;
  if (wg < iq1_groups) {
    saluki_iq1s_grid();
    gemv_raw_tr8_run(IE_RQ_IQ1_S, 0, 8u, 4u, wg, q0, xq, y0, 16384u, 160u, xs, ys0, rs0, bias0, res0);
  } else {
    gemv_q8q8_mseg(wg - iq1_groups, q1, xq, y1, 96u, 160u, 8u, (const G float *)qw1, res1, xs, ys1, rs1);
  }
}

/* One row per lane makes the eight tokens' activation loads wave-uniform.
 * gemv_raw_tr9_run preserves the original per-row FP32 reduction tree. */
KERNEL ie_saluki_iq2s_tr8(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                          const G float *bias, const G float *res, const G float *nw, G float *ny,
                          G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  if (T != 8u) return;
  for (u32 i = tid(); i < 1024u; i += NT) tr8_g2s[i] = TR8_G2C(ie_rq_iq2s_grid[i]);
  lbarrier();
  gemv_raw_tr9_run(IE_RQ_IQ2_S, 8u, wgid(), q, xq, y, rows, nb, xs, ys, rs, bias, res);
}

/* Fixed eight-token up projection, with no bias or residual. Keep four
 * rows per wave and the original reduction. */
KERNEL ie_saluki_iq1s_tr8(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                          const G float *bias, const G float *res, const G float *nw, G float *ny,
                          G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)rows; (void)nb; (void)T; (void)bias; (void)res; (void)nw;
  (void)ny; (void)nq; (void)eps; (void)count; (void)rs;
  saluki_iq1s_grid();
  gemv_raw_tr8_run(IE_RQ_IQ1_S, 0, 8u, 4u, wgid(), q, xq, y, 17408u, 160u, xs, ys, 0, 0, 0);
}

/* Fixed eight-token down projection with the original output epilogue. */
KERNEL ie_saluki_iq1s_down_tr8(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                               const G float *bias, const G float *res, const G float *nw, G float *ny,
                               G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)rows; (void)nb; (void)T; (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  saluki_iq1s_grid();
  gemv_raw_tr8_run(IE_RQ_IQ1_S, 0, 8u, 4u, wgid(), q, xq, y, 5120u, 544u, xs, ys, rs, bias, res);
}

/* Rank-32 LoRA-B projection: 32 rows per workgroup, up to 32 tokens per grid-y
 * block. Coalesced 32x32 LDS weight transpose -> 1 row per lane, wave-uniform
 * scalar activation loads, exact 5-level binary reduction matching wave_tree_f,
 * and 32-lane coalesced residual load + output store. Bitwise equal to
 * ie_gemv_f32_tw and ie_saluki_lora_b. */
static LDS float saluki_lb_w[32][33];

static inline __attribute__((always_inline)) float saluki_tree32_exact(const float p[32]) {
#pragma clang fp contract(off)
  float a[16], b[8], c[4], d[2];
#pragma unroll
  for (u32 i = 0; i < 16u; i++) a[i] = p[i] + p[i + 16u];
#pragma unroll
  for (u32 i = 0; i < 8u; i++) b[i] = a[i] + a[i + 8u];
#pragma unroll
  for (u32 i = 0; i < 4u; i++) c[i] = b[i] + b[i + 4u];
#pragma unroll
  for (u32 i = 0; i < 2u; i++) d[i] = c[i] + c[i + 2u];
  return d[0] + d[1];
}

KERNEL ie_saluki_lora_b_t32(const G float *w, const G float *x, G float *y, const G float *res,
                            u32 rows, u32 T, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  const u32 r0 = wgid() * 32u, tbase = tokid() * 32u;
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 rr = wv * 4u + k;
    saluki_lb_w[rr][l] = (r0 + rr < rows) ? w[(unsigned long)(r0 + rr) * 32u + l] : 0.0f;
  }
  lbarrier();
  const u32 ro = r0 + l;
  if (ro >= rows) return;
  float wr[32];
#pragma unroll
  for (u32 c = 0; c < 32u; c++) wr[c] = saluki_lb_w[l][c];

  const u32 tend = tbase + 32u < T ? tbase + 32u : T;
  for (u32 p = tbase + wv; p < tend; p += 8u) {
    const TR8_CSP float *xt = (const TR8_CSP float *)((const TR8_CSP u8 *)x + (unsigned long)p * xs);
    const float rv = ((const G float *)((const G u8 *)res + (unsigned long)p * rs))[ro];
    float pr[32];
#pragma unroll
    for (u32 c = 0; c < 32u; c++) pr[c] = 0.0f + wr[c] * xt[c];
    ((G float *)((G u8 *)y + (unsigned long)p * ys))[ro] = rv + saluki_tree32_exact(pr);
  }
}

/* Eight-token IQ2_XXS with 1 row per lane (32 rows/WG), wave-uniform scalar
 * activation loads, packed 1 KiB u32 code LUT, ALU sign/parity decode hoisted
 * out of the 8-token dot loop, and exact 8-token LDS tree reduction. Bitwise
 * equal to ie_gemv_raw_iq2xxs_tr8 and ie_saluki_iq2xxs_tr8_r4. */
static LDS u32 saluki_xxs_u32[256];
static LDS float saluki_tr9_red[8u * 8u * 32u];

static inline __attribute__((always_inline)) u32 saluki_pack_iq2_code(ie_rq_u64 g) {
  const ie_rq_u64 c = TR8_G2C(g);
  return (u32)c | ((u32)(c >> 32u) << 4u);
}

KERNEL ie_saluki_iq2xxs_tr9(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                            const G float *bias, const G float *res, const G float *nw, G float *ny,
                            G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  if (T != 8u) return;
  saluki_xxs_u32[tid()] = saluki_pack_iq2_code(ie_rq_iq2xxs_grid[tid()]);
  lbarrier();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 r0 = 32u * wgid(), r = r0 + l;
  const u32 mb = nb >> 3u;
  const G u8 *rowp = q + (unsigned long)(r < rows ? r : r0) * mb * 66u;
  const G u8 *xt[8];
#pragma unroll
  for (u32 t = 0; t < 8u; t++) xt[t] = tr8_uniform(xq + (unsigned long)t * xs);
  float a[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[j][t] = 0.0f;

  const u32 o = 2u + 8u * w;
  for (u32 k0 = 0; k0 < nb; k0 += 32u) {
#pragma unroll
    for (u32 j = 0; j < 4u; j++) {
      const u32 b = k0 + w + 8u * j;
      if (b < nb) {
        const G u8 *blk = rowp + (unsigned long)(b >> 3u) * 66u;
        const float d = ie_rq_h2f(ie_rq_a16(blk, 0u));
        const u32 a0 = ie_rq_a32(blk, o), a1 = ie_rq_a32(blk, o + 4u);
        const float dw = d * (0.5f + (float)(a1 >> 28u)) * 0.25f;
        u32 wl[4], wh[4];
#pragma unroll
        for (u32 g = 0; g < 4u; g++) {
          const u32 cp = saluki_xxs_u32[ie_byte(a0, g)];
          TR8_KSW(cp & 0x0F0F0F0Fu, (cp >> 4u) & 0x0F0F0F0Fu, (a1 >> (7u * g)) & 127u, wl[g], wh[g]);
        }
        const u32 xo = 32u * b, dao = 32u * nb + 4u * b;
#pragma unroll
        for (u32 t = 0; t < 8u; t++) {
          const TR8_CSP u8 *xb = (const TR8_CSP u8 *)xt[t];
          const float dav = *(const TR8_CSP float *)(xb + dao);
          const u32x4 x0 = *(const TR8_CSP u32x4 *)(xb + xo);
          const u32x4 x1 = *(const TR8_CSP u32x4 *)(xb + xo + 16u);
          u32 idot = ie_dot4_ss(0u, wl[0], x0.x);
          idot = ie_dot4_ss(idot, wh[0], x0.y);
          idot = ie_dot4_ss(idot, wl[1], x0.z);
          idot = ie_dot4_ss(idot, wh[1], x0.w);
          idot = ie_dot4_ss(idot, wl[2], x1.x);
          idot = ie_dot4_ss(idot, wh[2], x1.y);
          idot = ie_dot4_ss(idot, wl[3], x1.z);
          idot = ie_dot4_ss(idot, wh[3], x1.w);
          a[j][t] += dav * (dw * (float)(int)idot);
        }
      }
    }
  }
#pragma unroll
  for (u32 t = 0; t < 8u; t++) saluki_tr9_red[(w * 8u + t) * 32u + l] = (a[0][t] + a[2][t]) + (a[1][t] + a[3][t]);
  lbarrier();
  {
    const u32 i = tid();
    const u32 rr = i & 31u, t = i >> 5u, row = r0 + rr;
    float v[8];
#pragma unroll
    for (u32 x = 0; x < 8u; x++) v[x] = saluki_tr9_red[(x * 8u + t) * 32u + rr];
    const float x0 = v[0] + v[4], x1 = v[1] + v[5], x2 = v[2] + v[6], x3 = v[3] + v[7];
    float out = (x0 + x2) + (x1 + x3);
    if (row < rows) {
      if (bias) out = out + bias[row];
      if (res) out = ((const G float *)((const G u8 *)res + (unsigned long)t * rs))[row] + out;
      ((G float *)((G u8 *)y + (unsigned long)t * ys))[row] = out;
    }
  }
}

#define SALUKI_LDSP __attribute__((address_space(3)))
typedef float saluki_f32x4 __attribute__((ext_vector_type(4)));

/* Prefill rank-32 LoRA-A (rows=32, cols%256==0, no bias/res): all 32 rows in
 * one workgroup (4 rows/wave x 8 waves) with double-buffered 4-token x staging
 * in LDS (4x less input traffic). Preserves exact ie_gemm_f32_s order
 * (c = l, l+32, ... then wave_tree_f). */
static LDS float saluki_la_x4_db[2][4][256];
KERNEL ie_saluki_lora_a_t4(const G float *w, const G float *x, G float *y, u32 rows, u32 cols,
                           u32 xs, u32 ys, u32 T) {
  (void)rows;
  const u32 t0 = wgid() * 4u, wv = tid() >> 5, l = lane();
  if (t0 >= T) return;
  const u32 r0 = wv * 4u;
  const G float *w0 = w + (unsigned long)(r0 + 0u) * cols;
  const G float *w1 = w + (unsigned long)(r0 + 1u) * cols;
  const G float *w2 = w + (unsigned long)(r0 + 2u) * cols;
  const G float *w3 = w + (unsigned long)(r0 + 3u) * cols;
  float a0[4], a1[4], a2[4], a3[4];
#pragma unroll
  for (u32 t = 0; t < 4u; t++) a0[t] = 0.0f, a1[t] = 0.0f, a2[t] = 0.0f, a3[t] = 0.0f;
  const u32 lt0 = tid() >> 6, lc4 = tid() & 63u;
  const u32 tkA = t0 + lt0 < T ? t0 + lt0 : T - 1u;
  const G float *xA = (const G float *)((const G u8 *)x + (unsigned long)tkA * xs);
  *(SALUKI_LDSP saluki_f32x4 *)&saluki_la_x4_db[0][lt0][lc4 * 4u] = *(const G saluki_f32x4 *)(xA + lc4 * 4u);
  barrier();
  u32 cur = 0u;
  for (u32 c0 = 0; c0 < cols; c0 += 256u) {
    const u32 nxt = cur ^ 1u;
    saluki_f32x4 pfx;
    if (c0 + 256u < cols) pfx = *(const G saluki_f32x4 *)(xA + c0 + 256u + lc4 * 4u);
#pragma unroll
    for (u32 k = 0; k < 8u; k++) {
      const u32 c = l + 32u * k;
      const float k0 = w0[c0 + c], k1 = w1[c0 + c], k2 = w2[c0 + c], k3 = w3[c0 + c];
#pragma unroll
      for (u32 t = 0; t < 4u; t++) {
        const float xv = saluki_la_x4_db[cur][t][c];
        a0[t] += k0 * xv;
        a1[t] += k1 * xv;
        a2[t] += k2 * xv;
        a3[t] += k3 * xv;
      }
    }
    if (c0 + 256u < cols) {
      *(SALUKI_LDSP saluki_f32x4 *)&saluki_la_x4_db[nxt][lt0][lc4 * 4u] = pfx;
      barrier();
    }
    cur = nxt;
  }
#pragma unroll
  for (u32 t = 0; t < 4u; t++) {
    const float v0 = wave_tree_f(a0[t]), v1 = wave_tree_f(a1[t]);
    const float v2 = wave_tree_f(a2[t]), v3 = wave_tree_f(a3[t]);
    const u32 tk = t0 + t;
    if (l == 0u && tk < T) {
      G float *yt = (G float *)((G u8 *)y + (unsigned long)tk * ys);
      yt[r0 + 0u] = v0;
      yt[r0 + 1u] = v1;
      yt[r0 + 2u] = v2;
      yt[r0 + 3u] = v3;
    }
  }
}

/* Short-chunk prefill rank-32 LoRA-A (17 <= T <= 96, cols%1024==0): 1 token
 * x 32 rows per workgroup with double-buffered 1024-col LDS staging. Exact
 * ie_gemm_f32_s arithmetic order. */
static LDS float saluki_la_x1_db[2][1024];
KERNEL ie_saluki_lora_a_t1(const G float *w, const G float *x, G float *y, u32 rows, u32 cols,
                           u32 xs, u32 ys, u32 T) {
  (void)rows;
  const u32 tk = wgid(), wv = tid() >> 5, l = lane();
  if (tk >= T) return;
  const u32 r0 = wv * 4u;
  const G float *w0 = w + (unsigned long)(r0 + 0u) * cols;
  const G float *w1 = w + (unsigned long)(r0 + 1u) * cols;
  const G float *w2 = w + (unsigned long)(r0 + 2u) * cols;
  const G float *w3 = w + (unsigned long)(r0 + 3u) * cols;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  const G float *xA = (const G float *)((const G u8 *)x + (unsigned long)tk * xs);
  const u32 ti = tid();
  *(SALUKI_LDSP saluki_f32x4 *)&saluki_la_x1_db[0][ti * 4u] = *(const G saluki_f32x4 *)(xA + ti * 4u);
  barrier();
  u32 cur = 0u;
  for (u32 c0 = 0; c0 < cols; c0 += 1024u) {
    const u32 nxt = cur ^ 1u;
    saluki_f32x4 pfx;
    if (c0 + 1024u < cols) pfx = *(const G saluki_f32x4 *)(xA + c0 + 1024u + ti * 4u);
    for (u32 b = 0; b < 32u; b += 8u) {
#pragma unroll
      for (u32 k = 0; k < 8u; k++) {
        const u32 c = l + 32u * (b + k);
        const float xv = saluki_la_x1_db[cur][c];
        a0 += w0[c0 + c] * xv;
        a1 += w1[c0 + c] * xv;
        a2 += w2[c0 + c] * xv;
        a3 += w3[c0 + c] * xv;
      }
    }
    if (c0 + 1024u < cols) {
      *(SALUKI_LDSP saluki_f32x4 *)&saluki_la_x1_db[nxt][ti * 4u] = pfx;
      barrier();
    }
    cur = nxt;
  }
  const float v0 = wave_tree_f(a0), v1 = wave_tree_f(a1), v2 = wave_tree_f(a2), v3 = wave_tree_f(a3);
  if (l == 0u) {
    G float *yt = (G float *)((G u8 *)y + (unsigned long)tk * ys);
    yt[r0 + 0u] = v0;
    yt[r0 + 1u] = v1;
    yt[r0 + 2u] = v2;
    yt[r0 + 3u] = v3;
  }
}

/* Verify rank-32 LoRA-A (2 <= T <= 16, no bias/res): 1 row x 4 tokens per
 * workgroup reusing weight reads 4x, preserving exact gemv_f32_row + wg_sum
 * order. */
static LDS float saluki_lav_red[4][8];
KERNEL ie_saluki_lora_a_v4(const G float *w, const G float *x, G float *y, u32 rows, u32 cols,
                           u32 xs, u32 ys, u32 T) {
  const u32 r = wgid(), t0 = tokid() * 4u, ti = tid(), wv = ti >> 5, l = lane();
  if (r >= rows || t0 >= T) return;
  const G float *row = w + (unsigned long)r * cols;
  const G float *xp[4];
#pragma unroll
  for (u32 t = 0; t < 4u; t++) {
    const u32 tk = t0 + t < T ? t0 + t : T - 1u;
    xp[t] = (const G float *)((const G u8 *)x + (unsigned long)tk * xs);
  }
  float acc[4];
#pragma unroll
  for (u32 t = 0; t < 4u; t++) acc[t] = 0.0f;
  u32 c = ti;
  for (; c + 3u * NT < cols; c += 4u * NT) {
    float wv4[4];
#pragma unroll
    for (u32 k = 0; k < 4u; k++) wv4[k] = row[c + k * NT];
#pragma unroll
    for (u32 t = 0; t < 4u; t++) {
      float xv4[4];
#pragma unroll
      for (u32 k = 0; k < 4u; k++) xv4[k] = xp[t][c + k * NT];
#pragma unroll
      for (u32 k = 0; k < 4u; k++) acc[t] += wv4[k] * xv4[k];
    }
  }
  for (; c < cols; c += NT) {
    const float w0 = row[c];
#pragma unroll
    for (u32 t = 0; t < 4u; t++) acc[t] += w0 * xp[t][c];
  }
#pragma unroll
  for (u32 t = 0; t < 4u; t++) {
    const float v = wave_tree_f(acc[t]);
    if (l == 0u) saluki_lav_red[t][wv] = v;
  }
  barrier();
  if (ti < 4u && t0 + ti < T) {
    float s = 0.0f;
#pragma unroll
    for (u32 j = 0; j < 8u; j++) s += saluki_lav_red[ti][j];
    ((G float *)((G u8 *)y + (unsigned long)(t0 + ti) * ys))[r] = s;
  }
}

/* Eight-token Q2_K with 4 rows per wave (32 rows/WG), hoisted per-row decode
 * and shared activation sums across all 4 rows. Bitwise equal to
 * ie_gemv_raw_q2k_tr8. */
KERNEL ie_saluki_q2k_tr8_r4(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                            const G float *bias, const G float *res, const G float *nw, G float *ny,
                            G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  if (T != 8u) return;
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 r0 = ie_gemv_row(wgid(), wv) * 4u, l = lane();
  if (r0 >= rows) return;
  const u32 mb = nb / 8u;
  const G u8 *row[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) row[k] = tr8_uniform(q + (unsigned long)(r0 + k < rows ? r0 + k : r0) * mb * 84u);
  const G u8 *xt[8];
#pragma unroll
  for (u32 t = 0; t < 8u; t++) xt[t] = tr8_uniform(xq + (unsigned long)t * xs);
  float a[4][8];
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[k][t] = 0.0f;

  const u32 s = l & 7u, ps_o = 2u * s, pq_o = 16u + (s >> 2u) * 32u;
  const u32 sh = 2u * (s & 3u), qmask = 0x03030303u << sh;
  for (u32 b = l; b < nb; b += 32u) {
    const u32 qb = b >> 3u, xo = 32u * b, dao = 32u * nb + 4u * b;
    float dav_v[8], xsf0[8], xsf1[8];
#pragma unroll
    for (u32 t = 0; t < 8u; t++) {
      const G u8 *xb = xt[t];
      dav_v[t] = *(const G float *)(xb + dao);
      const u32x4 x0 = *(const G u32x4 *)(xb + xo);
      const u32x4 x1 = *(const G u32x4 *)(xb + xo + 16u);
      const u32 xs0 = ie_dot4_ss(ie_dot4_ss(0u, x0.x, 0x01010101u), x0.y, 0x01010101u) +
                      ie_dot4_ss(ie_dot4_ss(0u, x0.z, 0x01010101u), x0.w, 0x01010101u);
      const u32 xs1 = ie_dot4_ss(ie_dot4_ss(0u, x1.x, 0x01010101u), x1.y, 0x01010101u) +
                      ie_dot4_ss(ie_dot4_ss(0u, x1.z, 0x01010101u), x1.w, 0x01010101u);
      xsf0[t] = (float)(int)xs0;
      xsf1[t] = (float)(int)xs1;
    }
#pragma unroll
    for (u32 k = 0; k < 4u; k++) {
      const G u8 *blk = row[k] + (unsigned long)qb * 84u;
      const float d = ie_rq_h2f(ie_rq_a16(blk, 80u));
      const float mn = ie_rq_h2f(ie_rq_a16(blk, 82u));
      const u32 sc16 = ie_rq_a16(blk, ps_o);
      const float dw0 = d * (float)(sc16 & 15u), dm0 = mn * (float)((sc16 >> 4u) & 15u);
      const float dw1 = d * (float)((sc16 >> 8u) & 15u), dm1 = mn * (float)(sc16 >> 12u);
      const u32 q00 = ie_rq_a32(blk, pq_o + 0u) & qmask;
      const u32 q01 = ie_rq_a32(blk, pq_o + 4u) & qmask;
      const u32 q02 = ie_rq_a32(blk, pq_o + 8u) & qmask;
      const u32 q03 = ie_rq_a32(blk, pq_o + 12u) & qmask;
      const u32 q10 = ie_rq_a32(blk, pq_o + 16u) & qmask;
      const u32 q11 = ie_rq_a32(blk, pq_o + 20u) & qmask;
      const u32 q12 = ie_rq_a32(blk, pq_o + 24u) & qmask;
      const u32 q13 = ie_rq_a32(blk, pq_o + 28u) & qmask;
#pragma unroll
      for (u32 t = 0; t < 8u; t++) {
        const G u8 *xb = xt[t];
        const u32x4 x0 = *(const G u32x4 *)(xb + xo);
        const u32x4 x1 = *(const G u32x4 *)(xb + xo + 16u);
        u32 id0 = ie_dot4_us(0u, q00, x0.x);
        id0 = ie_dot4_us(id0, q01, x0.y);
        id0 = ie_dot4_us(id0, q02, x0.z);
        id0 = ie_dot4_us(id0, q03, x0.w);
        const float t0 = dav_v[t] * (dw0 * (float)((int)id0 >> sh) - dm0 * xsf0[t]);
        u32 id1 = ie_dot4_us(0u, q10, x1.x);
        id1 = ie_dot4_us(id1, q11, x1.y);
        id1 = ie_dot4_us(id1, q12, x1.z);
        id1 = ie_dot4_us(id1, q13, x1.w);
        const float t1 = dav_v[t] * (dw1 * (float)((int)id1 >> sh) - dm1 * xsf1[t]);
        a[k][t] += (t0 + t1);
      }
    }
  }
  float o[4];
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    o[k] = 0.0f;
#pragma unroll
    for (u32 t = 0; t < 8u; t++) {
      const float sv = I2F(__builtin_amdgcn_readlane(F2I(wave_tree_f(a[k][t])), 0));
      if (l == t) o[k] = sv;
    }
  }
  if (l < 8u) {
    const G float *rt = res ? (const G float *)((const G u8 *)res + (unsigned long)l * rs) : res;
    G float *yt = (G float *)((G u8 *)y + (unsigned long)l * ys);
#pragma unroll
    for (u32 k = 0; k < 4u; k++) if (bias && r0 + k < rows) o[k] = o[k] + bias[r0 + k];
#pragma unroll
    for (u32 k = 0; k < 4u; k++) if (rt && r0 + k < rows) o[k] = rt[r0 + k] + o[k];
#pragma unroll
    for (u32 k = 0; k < 4u; k++) if (r0 + k < rows) yt[r0 + k] = o[k];
  }
}

/* Eight-token IQ2_XS with 1 row/lane (32 rows/WG), wave-uniform scalar
 * activation loads, packed 2 KiB u32 code LUT, ALU sign/parity decode hoisted
 * outside the 8-token loop. Bitwise equal to ie_gemv_raw_iq2xs_tr8. */
static LDS u32 saluki_iq2xs_u32[512];
KERNEL ie_saluki_iq2xs_tr9(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                           const G float *bias, const G float *res, const G float *nw, G float *ny,
                           G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  if (T != 8u) return;
  saluki_iq2xs_u32[tid()] = saluki_pack_iq2_code(ie_rq_iq2xs_grid[tid()]);
  saluki_iq2xs_u32[tid() + 256u] = saluki_pack_iq2_code(ie_rq_iq2xs_grid[tid() + 256u]);
  lbarrier();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 r0 = 32u * wgid(), r = r0 + l;
  if (r0 >= rows) return;
  const u32 mb = nb >> 3u;
  const G u8 *rowp = q + (unsigned long)(r < rows ? r : r0) * mb * 74u;
  const G u8 *xt[8];
#pragma unroll
  for (u32 t = 0; t < 8u; t++) xt[t] = tr8_uniform(xq + (unsigned long)t * xs);
  float a[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[j][t] = 0.0f;

  const u32 qo = 2u + 8u * w, sco = 66u + 2u * (w >> 1u), scsh = 8u * (w & 1u);
  for (u32 k0 = 0; k0 < nb; k0 += 32u) {
#pragma unroll
    for (u32 j = 0; j < 4u; j++) {
      const u32 b = k0 + w + 8u * j;
      if (b < nb) {
        const G u8 *blk = rowp + (unsigned long)(b >> 3u) * 74u;
        const float d = ie_rq_h2f(ie_rq_a16(blk, 0u));
        const u32 sc = (ie_rq_a16(blk, sco) >> scsh) & 255u;
        const float dw0 = d * (0.5f + (float)(sc & 15u)) * 0.25f;
        const float dw1 = d * (0.5f + (float)(sc >> 4u)) * 0.25f;
        const u32 q01 = ie_rq_a32(blk, qo), q23 = ie_rq_a32(blk, qo + 4u);
        u32 wl[4], wh[4];
#pragma unroll
        for (u32 g = 0; g < 4u; g++) {
          const u32 q2 = ((g < 2u ? q01 : q23) >> (16u * (g & 1u))) & 0xFFFFu;
          const u32 cp = saluki_iq2xs_u32[q2 & 511u];
          TR8_KSW(cp & 0x0F0F0F0Fu, (cp >> 4u) & 0x0F0F0F0Fu, q2 >> 9u, wl[g], wh[g]);
        }
        const u32 xo = 32u * b, dao = 32u * nb + 4u * b;
#pragma unroll
        for (u32 t = 0; t < 8u; t++) {
          const TR8_CSP u8 *xb = (const TR8_CSP u8 *)xt[t];
          const float dav = *(const TR8_CSP float *)(xb + dao);
          const u32x4 x0 = *(const TR8_CSP u32x4 *)(xb + xo);
          const u32x4 x1 = *(const TR8_CSP u32x4 *)(xb + xo + 16u);
          u32 id0 = ie_dot4_ss(0u, wl[0], x0.x);
          id0 = ie_dot4_ss(id0, wh[0], x0.y);
          id0 = ie_dot4_ss(id0, wl[1], x0.z);
          id0 = ie_dot4_ss(id0, wh[1], x0.w);
          u32 id1 = ie_dot4_ss(0u, wl[2], x1.x);
          id1 = ie_dot4_ss(id1, wh[2], x1.y);
          id1 = ie_dot4_ss(id1, wl[3], x1.z);
          id1 = ie_dot4_ss(id1, wh[3], x1.w);
          a[j][t] += dav * (dw0 * (float)(int)id0 + dw1 * (float)(int)id1);
        }
      }
    }
  }
#pragma unroll
  for (u32 t = 0; t < 8u; t++) saluki_tr9_red[(w * 8u + t) * 32u + l] = (a[0][t] + a[2][t]) + (a[1][t] + a[3][t]);
  lbarrier();
  {
    const u32 i = tid();
    const u32 rr = i & 31u, t = i >> 5u, row = r0 + rr;
    float v[8];
#pragma unroll
    for (u32 x = 0; x < 8u; x++) v[x] = saluki_tr9_red[(x * 8u + t) * 32u + rr];
    const float x0 = v[0] + v[4], x1 = v[1] + v[5], x2 = v[2] + v[6], x3 = v[3] + v[7];
    float out = (x0 + x2) + (x1 + x3);
    if (row < rows) {
      if (bias) out = out + bias[row];
      if (res) out = ((const G float *)((const G u8 *)res + (unsigned long)t * rs))[row] + out;
      ((G float *)((G u8 *)y + (unsigned long)t * ys))[row] = out;
    }
  }
}

/* Eight-token shape-specialized IQ1_M (17408x5120, no bias/res, 4 rows/wave).
 * Bitwise equal to ie_gemv_raw_iq1m_tr8. */
KERNEL ie_saluki_iq1m_tr8_r4(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                             const G float *bias, const G float *res, const G float *nw, G float *ny,
                             G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nb; (void)bias; (void)res; (void)nw; (void)ny; (void)nq; (void)eps; (void)count; (void)rs;
  if (T != 8u) return;
  for (u32 i = tid(); i < 2048u; i += NT) tr8_g1s[i] = ie_rq_iq1s_grid[i];
  lbarrier();
  gemv_raw_tr8_run(IE_RQ_IQ1_M, 0, 8u, 4u, wgid(), q, xq, y, rows, 160u, xs, ys, 0, 0, 0);
}

/* Eight-token shape-specialized IQ3_S (nb=160, no bias/res): 4 rows/wave for
 * the 248320-row lm_head and 2 rows/wave for 17408x5120. Bitwise equal to
 * ie_gemv_raw_iq3s_tr8. */
KERNEL ie_saluki_iq3s_tr8_r4(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                             const G float *bias, const G float *res, const G float *nw, G float *ny,
                             G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nb; (void)bias; (void)res; (void)nw; (void)ny; (void)nq; (void)eps; (void)count; (void)rs;
  if (T != 8u) return;
  tr8_g3s[tid()] = ie_rq_iq3s_grid[tid()];
  tr8_g3s[tid() + 256u] = ie_rq_iq3s_grid[tid() + 256u];
  lbarrier();
  gemv_raw_tr8_run(IE_RQ_IQ3_S, 0, 8u, 4u, wgid(), q, xq, y, rows, 160u, xs, ys, 0, 0, 0);
}

KERNEL ie_saluki_iq3s_tr8_r2(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                             const G float *bias, const G float *res, const G float *nw, G float *ny,
                             G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nb; (void)bias; (void)res; (void)nw; (void)ny; (void)nq; (void)eps; (void)count; (void)rs;
  if (T != 8u) return;
  tr8_g3s[tid()] = ie_rq_iq3s_grid[tid()];
  tr8_g3s[tid() + 256u] = ie_rq_iq3s_grid[tid() + 256u];
  lbarrier();
  gemv_raw_tr8_run(IE_RQ_IQ3_S, 0, 8u, 2u, wgid(), q, xq, y, rows, 160u, xs, ys, 0, 0, 0);
}

/* Eight-token IQ3_XXS with 1 row/lane (32 rows/WG), 1 KiB grid LUT + ALU
 * parity sign hoisted outside the 8-token dot loop. Bitwise equal to
 * ie_gemv_raw_iq3xxs_tr8. */
static LDS u32 saluki_iq3xxs_u32[256];
KERNEL ie_saluki_iq3xxs_tr9(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                            const G float *bias, const G float *res, const G float *nw, G float *ny,
                            G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  if (T != 8u) return;
  saluki_iq3xxs_u32[tid()] = ie_rq_iq3xxs_grid[tid()];
  lbarrier();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 r0 = 32u * wgid(), r = r0 + l;
  if (r0 >= rows) return;
  const u32 mb = nb >> 3u;
  const G u8 *rowp = q + (unsigned long)(r < rows ? r : r0) * mb * 98u;
  const G u8 *xt[8];
#pragma unroll
  for (u32 t = 0; t < 8u; t++) xt[t] = tr8_uniform(xq + (unsigned long)t * xs);
  float a[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[j][t] = 0.0f;

  const u32 qo = 2u + 8u * w, auxo = 66u + 4u * w;
  for (u32 k0 = 0; k0 < nb; k0 += 32u) {
#pragma unroll
    for (u32 j = 0; j < 4u; j++) {
      const u32 b = k0 + w + 8u * j;
      if (b < nb) {
        const G u8 *blk = rowp + (unsigned long)(b >> 3u) * 98u;
        const float d = ie_rq_h2f(ie_rq_a16(blk, 0u));
        const u32 aux = ie_rq_a32(blk, auxo);
        const float dw = d * (0.5f + (float)(aux >> 28u)) * 0.5f;
        const u32 q01 = ie_rq_a32(blk, qo), q23 = ie_rq_a32(blk, qo + 4u);
        u32 wl[4], wh[4];
#pragma unroll
        for (u32 g = 0; g < 4u; g++) {
          const u32 qp = ((g < 2u ? q01 : q23) >> (16u * (g & 1u))) & 0xFFFFu;
          const u32 g1 = saluki_iq3xxs_u32[qp & 255u], g2 = saluki_iq3xxs_u32[qp >> 8u];
          const u32 s7 = (aux >> (7u * g)) & 127u;
          const u32 s8 = s7 | (((u32)__builtin_popcount(s7) & 1u) << 7u);
          const ie_rq_u64 m = ie_rq_mask8(s8);
          wl[g] = ie_rq_ssign(g1, (u32)m);
          wh[g] = ie_rq_ssign(g2, (u32)(m >> 32u));
        }
        const u32 xo = 32u * b, dao = 32u * nb + 4u * b;
#pragma unroll
        for (u32 t = 0; t < 8u; t++) {
          const TR8_CSP u8 *xb = (const TR8_CSP u8 *)xt[t];
          const float dav = *(const TR8_CSP float *)(xb + dao);
          const u32x4 x0 = *(const TR8_CSP u32x4 *)(xb + xo);
          const u32x4 x1 = *(const TR8_CSP u32x4 *)(xb + xo + 16u);
          u32 idot = ie_dot4_ss(0u, wl[0], x0.x);
          idot = ie_dot4_ss(idot, wh[0], x0.y);
          idot = ie_dot4_ss(idot, wl[1], x0.z);
          idot = ie_dot4_ss(idot, wh[1], x0.w);
          idot = ie_dot4_ss(idot, wl[2], x1.x);
          idot = ie_dot4_ss(idot, wh[2], x1.y);
          idot = ie_dot4_ss(idot, wl[3], x1.z);
          idot = ie_dot4_ss(idot, wh[3], x1.w);
          a[j][t] += dav * (dw * (float)(int)idot);
        }
      }
    }
  }
#pragma unroll
  for (u32 t = 0; t < 8u; t++) saluki_tr9_red[(w * 8u + t) * 32u + l] = (a[0][t] + a[2][t]) + (a[1][t] + a[3][t]);
  lbarrier();
  {
    const u32 i = tid();
    const u32 rr = i & 31u, t = i >> 5u, row = r0 + rr;
    float v[8];
#pragma unroll
    for (u32 x = 0; x < 8u; x++) v[x] = saluki_tr9_red[(x * 8u + t) * 32u + rr];
    const float x0 = v[0] + v[4], x1 = v[1] + v[5], x2 = v[2] + v[6], x3 = v[3] + v[7];
    float out = (x0 + x2) + (x1 + x3);
    if (row < rows) {
      if (bias) out = out + bias[row];
      if (res) out = ((const G float *)((const G u8 *)res + (unsigned long)t * rs))[row] + out;
      ((G float *)((G u8 *)y + (unsigned long)t * ys))[row] = out;
    }
  }
}

/* Round 6: Tree-verify T=8 specializations (zero extra launches, 2 WGs/CU on gfx1201).
 * 1) ie_saluki_attn_long_mp_tr: 1-launch Q8 long attention for tree verify
 *    (n_head=24, n_kv=4, T=8, npg=3, wbeg=0, tr!=NULL). amdgpu_waves_per_eu(8, 8)
 *    caps VGPRs at 192 (2 WGs/CU vs base ie_attn_long_mp's 229 VGPRs / 1 WG/CU),
 *    keeping the 99.5% interior splits at 2x occupancy. Bitwise equal to ie_attn_long_mp.
 * 2) ie_saluki_gdn1p_tr: 8-token tree-verify GDN records kernel (nk=16, nv=48,
 *    cd=10240, ns=1, rw=1, c=wfrom=0, tr!=NULL). 152 VGPRs, 0 spills, 18,752 B LDS
 *    (2-3 WGs/CU vs base ie_gdn1p's 255 VGPRs / 33.6 KiB LDS). Bitwise equal to ie_gdn1p. */
#define SALUKI_TR_DEPTH(d03, d47, j) ((((j) < 4u ? (d03) : (d47)) >> (8u * ((j) & 3u))) & 255u)

__attribute__((amdgpu_kernel, visibility("protected"), amdgpu_flat_work_group_size(256, 256), amdgpu_waves_per_eu(8, 8)))
void ie_saluki_attn_long_mp_tr(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos,
                               u32 n_head, u32 n_kv, u32 T, u32 npg, G u32 *count, G u32 *gcount, u32 gmax, G u8 *oq,
                               const G float *gate, u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride,
                               const G u8 *tr) {
  n_head = 24u; n_kv = 4u; T = 8u; npg = 3u;
  const u32 d03 = __builtin_amdgcn_readfirstlane(*(const G u32 *)(tr + TR_DEP));
  const u32 d47 = __builtin_amdgcn_readfirstlane(*(const G u32 *)(tr + TR_DEP + 4u));
  const u32 grp = 6u, npair = 48u;
  const u32 kh = __builtin_amdgcn_readfirstlane(wgid() & 3u);
  const u32 sg = __builtin_amdgcn_readfirstlane(wgid() >> 2u);
  const u32 w = tid() >> 5, l = lane(), hl = l >> 4, rl = l & 15u;
  const u32 kend = __builtin_amdgcn_readfirstlane(pos + (u32)tr[368]);
  const u32 c0 = __builtin_amdgcn_readfirstlane(sg * AL_CH);
  if (c0 >= kend) return;
  const int is_tail = __builtin_amdgcn_readfirstlane(c0 + AL_CH > pos);
  const v8f zero = (v8f)(0.0f);
  const unsigned long pst = pstride;
  const u32 kb = c0 + 16u * w;
  const int kon = kb < kend;
  v8f s_pg[3] = {zero, zero, zero};
  if (__builtin_expect(!is_tail, 1)) {
    u32x4 kq[8]; u32x4 ksc4 = (u32x4)(0u);
    if (kon) {
      const unsigned long row = kb + rl < kend ? kb + rl : kend - 1u;
      al_q8_kload(kc, row, n_kv, kh, hl, kq, &ksc4);
    }
#pragma unroll
    for (u32 pg = 0; pg < 3u; pg++) {
      mp_load_q_lds(q, pg, npair, grp, kh, qst);
      lbarrier();
      if (kon) {
        const LDSP f16 *qh = (const LDSP f16 *)mp_lds, *ql = qh + 16u * AL_QS;
        s_pg[pg] = al_q8_kqt_blocks(kq, ksc4, qh, ql, hl, rl);
      }
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
      lbarrier();
    }
  } else {
#pragma nounroll
    for (u32 pg = 0; pg < 3u; pg++) {
      mp_load_q_lds(q, pg, npair, grp, kh, qst);
      lbarrier();
      if (kon) {
        const LDSP f16 *qh = (const LDSP f16 *)mp_lds, *ql = qh + 16u * AL_QS;
        const u32 ci = 16u * pg + rl, cj = ci / grp;
        if (kb + 15u >= pos) {
          const u32 jlo = 16u * pg / grp, jhi = (16u * pg + 15u) / grp;
          v8f stmp = zero;
#pragma nounroll
          for (u32 jj = jlo; jj <= jhi; jj++) {
            u32x4 tkq[8]; u32x4 tksc4 = (u32x4)(0u);
            const unsigned long trow = al_q8_trow(tr, jj, kb + rl < kend ? kb + rl : kend - 1u, pos);
            al_q8_kload(kc, trow, n_kv, kh, hl, tkq, &tksc4);
            const v8f sj = al_q8_kqt_blocks(tkq, tksc4, qh, ql, hl, rl);
            if (cj == jj) stmp = sj;
          }
          s_pg[pg] = stmp;
        } else {
          u32x4 tkq[8]; u32x4 tksc4 = (u32x4)(0u);
          const unsigned long row = kb + rl < kend ? kb + rl : kend - 1u;
          al_q8_kload(kc, row, n_kv, kh, hl, tkq, &tksc4);
          s_pg[pg] = al_q8_kqt_blocks(tkq, tksc4, qh, ql, hl, rl);
        }
      }
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
      lbarrier();
    }
  }
  u32x4 vr_cur = (u32x4)(0u); f16 vsc_cur = (f16)0.0f;
  if (c0 < kend) {
    const unsigned long row0 = c0 + rl < kend ? c0 + rl : kend - 1u;
    const G u8 *rec0 = (const G u8 *)vc + (row0 * n_kv + kh) * 272ul;
    vr_cur = *(const G u32x4 *)(rec0 + 32ul * w + 16ul * hl);
    vsc_cur = ((const G f16 *)(rec0 + 256ul))[w];
  }
#pragma unroll
  for (u32 pg = 0; pg < 3u; pg++) {
    const u32 ci = 16u * pg + rl, cj = ci / grp, cpj = pos + SALUKI_TR_DEPTH(d03, d47, cj);
    float sv[8], mx = -__builtin_inff();
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      sv[v] = kb + 8u * hl + v <= cpj ? s_pg[pg][v] : -__builtin_inff();
      mx = __builtin_fmaxf(mx, sv[v]);
    }
    mx = __builtin_fmaxf(mx, I2F(xrow_i(F2I(mx))));
    if (hl == 0u) al_red[0][16u * w + rl] = mx;
    lbarrier();
    float M = al_red[0][rl];
#pragma unroll
    for (u32 x = 1; x < 8u; x++) M = __builtin_fmaxf(M, al_red[0][16u * x + rl]);
    h8 ph8, pl8; float ls = 0.0f;
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      const float p = kb + 8u * hl + v <= cpj ? __builtin_expf(sv[v] - M) : 0.0f;
      ls += p; const f16 hv = (f16)p; ph8[v] = hv; pl8[v] = (f16)(p - (float)hv);
    }
    ls += I2F(xrow_i(F2I(ls)));
    if (hl == 0u) al_red[1][16u * w + rl] = ls;
    LDSP f16 *pp = (LDSP f16 *)mp_lds + pg * MP_PG_HALVES;
    *(LDSP h8 *)(pp + rl * AL_PS + 16u * w + 8u * hl) = ph8;
    *(LDSP h8 *)(pp + 16u * AL_PS + rl * AL_PS + 16u * w + 8u * hl) = pl8;
    lbarrier();
    float L = al_red[1][rl];
#pragma unroll
    for (u32 x = 1; x < 8u; x++) L += al_red[1][16u * x + rl];
    if (w == 0u && hl == 0u && c0 <= cpj) {
      const u32 h = kh * grp + ci % grp, ns = al_ns(cpj);
      G float *pp_ml = part + (unsigned long)cj * pst + ((unsigned long)h * (ns + al_ng(ns)) + sg) * AL_PSZ;
      pp_ml[0] = M; pp_ml[1] = L;
    }
  }
  v8f acc[3][2] = {{zero, zero}, {zero, zero}, {zero, zero}};
  {
    LDSP f16 *sc = (LDSP f16 *)(mp_lds + MP_OFF_SC) + w * (16u * AL_SS);
    if (__builtin_expect(!is_tail, 1)) {
#pragma nounroll
      for (u32 kt = 0; kt < 8u; kt++) {
        if (c0 + 16u * kt >= kend) break;
        u32x4 vr_nxt = (u32x4)(0u); f16 vsc_nxt = (f16)0.0f;
        if (kt + 1u < 8u && c0 + 16u * (kt + 1u) < kend) {
          const unsigned long nrow = c0 + 16u * (kt + 1u) + rl < kend ? c0 + 16u * (kt + 1u) + rl : kend - 1u;
          const G u8 *nrec = (const G u8 *)vc + (nrow * n_kv + kh) * 272ul;
          vr_nxt = *(const G u32x4 *)(nrec + 32ul * w + 16ul * hl);
          vsc_nxt = ((const G f16 *)(nrec + 256ul))[w];
        }
        h8 b0, b1; v8f vs; float w_sc;
        mp_q8_vprep(vr_cur, vsc_cur, sc, w, hl, rl, &b0, &b1, &vs, &w_sc);
#pragma unroll
        for (u32 pg = 0; pg < 3u; pg++)
          mp_q8_pv_accum(b0, b1, vs, w_sc, (const LDSP f16 *)mp_lds + pg * MP_PG_HALVES, kt, hl, rl, 0xffu,
                         &acc[pg][0], &acc[pg][1]);
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        vr_cur = vr_nxt; vsc_cur = vsc_nxt;
      }
    } else {
#pragma nounroll
      for (u32 kt = 0; kt < 8u; kt++) {
        if (c0 + 16u * kt >= kend) break;
        u32x4 vr_nxt = (u32x4)(0u); f16 vsc_nxt = (f16)0.0f;
        if (kt + 1u < 8u && c0 + 16u * (kt + 1u) < kend) {
          const unsigned long nrow = c0 + 16u * (kt + 1u) + rl < kend ? c0 + 16u * (kt + 1u) + rl : kend - 1u;
          const G u8 *nrec = (const G u8 *)vc + (nrow * n_kv + kh) * 272ul;
          vr_nxt = *(const G u32x4 *)(nrec + 32ul * w + 16ul * hl);
          vsc_nxt = ((const G f16 *)(nrec + 256ul))[w];
        }
        if (c0 + 16u * kt + 15u >= pos) {
#pragma nounroll
          for (u32 jj = 0; jj < T; jj++) {
            const unsigned long key = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;
            const unsigned long trow = al_q8_trow(tr, jj, (u32)key, pos);
            const G u8 *trec = (const G u8 *)vc + (trow * n_kv + kh) * 272ul;
            const u32x4 vrj = *(const G u32x4 *)(trec + 32ul * w + 16ul * hl);
            const f16 vsj = ((const G f16 *)(trec + 256ul))[w];
            h8 b0, b1; v8f vs; float w_sc;
            mp_q8_vprep(vrj, vsj, sc, w, hl, rl, &b0, &b1, &vs, &w_sc);
#pragma unroll
            for (u32 pg = 0; pg < 3u; pg++) {
              if (16u * pg / grp <= jj && jj <= (16u * pg + 15u) / grp) {
                u32 vmask = 0u;
#pragma unroll
                for (u32 v = 0; v < 8u; v++)
                  if ((16u * pg + 8u * hl + v) / grp == jj) vmask |= 1u << v;
                mp_q8_pv_accum(b0, b1, vs, w_sc, (const LDSP f16 *)mp_lds + pg * MP_PG_HALVES, kt, hl, rl, vmask,
                               &acc[pg][0], &acc[pg][1]);
              }
            }
            __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
          }
        } else {
          h8 b0, b1; v8f vs; float w_sc;
          mp_q8_vprep(vr_cur, vsc_cur, sc, w, hl, rl, &b0, &b1, &vs, &w_sc);
#pragma unroll
          for (u32 pg = 0; pg < 3u; pg++)
            mp_q8_pv_accum(b0, b1, vs, w_sc, (const LDSP f16 *)mp_lds + pg * MP_PG_HALVES, kt, hl, rl, 0xffu,
                           &acc[pg][0], &acc[pg][1]);
          __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        }
        vr_cur = vr_nxt; vsc_cur = vsc_nxt;
      }
    }
  }
#pragma unroll
  for (u32 pg = 0; pg < 3u; pg++) {
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      const u32 pi = 16u * pg + 8u * hl + v, j = pi / grp, h = kh * grp + pi % grp;
      if (c0 <= pos + SALUKI_TR_DEPTH(d03, d47, j)) {
        const u32 ns = al_ns(pos + SALUKI_TR_DEPTH(d03, d47, j));
        G float *pp = part + (unsigned long)j * pst + ((unsigned long)h * (ns + al_ng(ns)) + sg) * AL_PSZ + 4u;
        pp[32u * w + rl] = acc[pg][0][v]; pp[32u * w + 16u + rl] = acc[pg][1][v];
      }
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (w < npg && hl == 0u) {
    const u32 ci = 16u * w + rl, cj = ci / grp, cpj = pos + SALUKI_TR_DEPTH(d03, d47, cj);
    u32 last = 0u;
    if (c0 <= cpj) {
      const u32 ns = al_ns(cpj), g = sg / AL_G, in_g = ns - g * AL_G < AL_G ? ns - g * AL_G : AL_G;
      const unsigned long gi = ((unsigned long)cj * n_head + kh * grp + ci % grp) * gmax + g;
      last = __scoped_atomic_fetch_add(&gcount[gi], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == in_g;
    }
    mp_last[ci] = last;
  }
  barrier();
  for (u32 pi = w; pi < 16u * npg; pi += 8u) {
    if (!mp_last[pi]) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const u32 j = pi / grp, h = kh * grp + pi % grp, ns = al_ns(pos + SALUKI_TR_DEPTH(d03, d47, j)), ng = al_ng(ns), g = sg / AL_G;
    const u32 in_g = ns - g * AL_G < AL_G ? ns - g * AL_G : AL_G;
    G float *pb = part + (unsigned long)j * pst + (unsigned long)h * (ns + ng) * AL_PSZ;
    G float *ox = out ? (G float *)((G u8 *)out + (unsigned long)j * ost) + (unsigned long)h * 256ul : (G float *)0;
    G u8 *oqx = oq ? oq + (unsigned long)j * oqst : oq;
    const G float *gxh = gate ? (const G float *)((const G u8 *)gate + (unsigned long)j * gst) + (unsigned long)h * gstride : gate;
    float M2, L2; f32x4 a0, a1;
    al_msum(pb + (unsigned long)g * AL_G * AL_PSZ, in_g, &M2, &L2, &a0, &a1);
    if (l == 0u) gcount[((unsigned long)j * n_head + h) * gmax + g] = 0u;
    if (ng == 1u) { al_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_head * 8u, (LDSP float *)mp_lds + w * 256u); continue; }
    G float *gp = pb + (unsigned long)(ns + g) * AL_PSZ;
    if (l == 0u) gp[0] = M2, gp[1] = L2;
    *(G f32x4 *)(gp + 4u + 4u * l) = a0; *(G f32x4 *)(gp + 132u + 4u * l) = a1;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    u32 done = 0u;
    if (l == 0u) done = __scoped_atomic_fetch_add(&count[j * n_head + h], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == ng;
    if (!__builtin_amdgcn_readfirstlane(done)) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    al_msum(pb + (unsigned long)ns * AL_PSZ, ng, &M2, &L2, &a0, &a1);
    if (l == 0u) count[j * n_head + h] = 0u;
    al_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_head * 8u, (LDSP float *)mp_lds + w * 256u);
  }
}
#undef SALUKI_TR_DEPTH

KERNEL ie_saluki_gdn1p_tr(const G float *in, G float *out, G u8 *oq, G float *ring, G float *S, const G float *cw,
                          const G float *dtb, const G float *sa, const G float *nw, u32 pos, u32 cd, u32 nk, u32 nv,
                          float eps, u32 T, u32 is, u32 os, u32 qs, u32 c, u32 wfrom, u32 rw, u32 ns, G float *rec,
                          const G u8 *tr, const G float *lazy_rec, u32 lazy_mask, unsigned long rls, u32 r_idx) {
  (void)cd; (void)nk; (void)nv; (void)T; (void)c; (void)wfrom; (void)rw; (void)ns;
  if (!rec || !tr) return;
  const u32 TT = 8u, NK = 16u, NV = 48u, CD = 10240u;
  const u32 h = wgid(), t0 = tid(), j = t0 & 127u, half = t0 >> 7, kh = h % NK, w = t0 >> 5, inner = NV * 128u;
  const unsigned long own = (unsigned long)h * 16384u + half * 64u * 128u + j;
  const u32 ca = half ? NK * 128u + kh * 128u + j : kh * 128u + j, cv = 2u * NK * 128u + h * 128u + j;
  G float *S_slot_own = S + own;
  float st[64];
  for (u32 ii = 0; ii < 64u; ii++) st[ii] = (pos || lazy_mask) ? S_slot_own[ii * 128u] : 0.0f;
  ie_gdn_lazy_apply(st, S_slot_own, lazy_rec, lazy_mask, rls, r_idx, NV, h, half, j, 1);
#pragma unroll 4
  for (u32 t = 0; t < TT; t++) {
    const float a = conv_ch_tr(in, is, ring, cw, ca, t, pos, CD, tr);
    const float vv = half ? 0.0f : conv_ch_tr(in, is, ring, cw, cv, t, pos, CD, tr);
    const float ss = wave_sum_all(a * a);
    if (lane() == 0u) gp_r[t][w] = ss;
    if (half) gp_k[t][j] = a; else gp_q[t][j] = a;
    if (!half) gp_v[t][j] = vv;
  }
  if (t0 < TT) {
    const G float *it = in + t0 * is;
    const float beta = sigmoidf(it[CD + inner + h]);
    const float al = it[CD + inner + NV + h] + dtb[h];
    const float sp = al > 20.0f ? al : __builtin_logf(1.0f + __builtin_expf(al));
    const float decay = __builtin_expf(sp * sa[h]);
    gp_bd[t0][0] = beta, gp_bd[t0][1] = decay;
    if (t0) rec[((unsigned long)(t0 - 1u) * NV + h) * GDN_RECF] = decay;
  }
  for (u32 t = 0; t < TT; t++) {
    if (h == kh) ring[((pos + t) & (GDN_RING - 1u)) * CD + ca] = in[t * is + ca];
    if (!half) ring[((pos + t) & (GDN_RING - 1u)) * CD + cv] = in[t * is + cv];
  }
  lbarrier();
  for (u32 t = 0; t < TT; t++) {
    const float n2 = half ? gp_r[t][4] + gp_r[t][5] + gp_r[t][6] + gp_r[t][7] : gp_r[t][0] + gp_r[t][1] + gp_r[t][2] + gp_r[t][3];
    const float an = (half ? gp_k[t][j] : gp_q[t][j]) * (1.0f / __builtin_sqrtf(n2 + eps));
    if (half) {
      gp_k[t][j] = an;
      if (t) rec[((unsigned long)(t - 1u) * NV + h) * GDN_RECF + GDN_RECK + j] = an;
    } else gp_q[t][j] = an;
  }
  lbarrier();
  for (u32 t = 0; t < TT; t++) {
    const float beta = gp_bd[t][0], decay = gp_bd[t][1];
    G float *rt = t ? rec + ((unsigned long)(t - 1u) * NV + h) * GDN_RECF : (G float *)0;
    if (t && tr[TR_PAR + t] != t - 1u) {
      const u32 pa = tr[TR_PAR + t];
      for (u32 ii = 0; ii < 64u; ii++) st[ii] = S_slot_own[ii * 128u];
      for (u32 m = pa ? tr_anc(tr, pa) : 0u; m; m &= m - 1u) {
        const u32 u = (u32)__builtin_ctz(m);
        const float du = gp_bd[u][1], dl = gp_d[u][j];
        for (u32 ii = 0; ii < 64u; ii++) {
          st[ii] = st[ii] * du;
          st[ii] = __builtin_fmaf(gp_k[u][half * 64u + ii], dl, st[ii]);
        }
      }
    }
    float pk = 0.0f;
    for (u32 ii = 0; ii < 64u; ii++) {
      st[ii] = st[ii] * decay;
      pk += st[ii] * gp_k[t][half * 64u + ii];
    }
    gp_px[half][j] = pk;
    lbarrier();
    const float delta = (gp_v[t][j] - (gp_px[0][j] + gp_px[1][j])) * beta;
    if (rt && !half) rt[GDN_RECD + j] = delta;
    if (!half) gp_d[t][j] = delta;
    float po = 0.0f;
    G float *Sw = S_slot_own;
    if (t == 0u) GDN_SWEEP_UPD(gp_k[t], gp_q[t], 1) else GDN_SWEEP_UPD(gp_k[t], gp_q[t], 0)
    gp_po[half][j] = po;
    lbarrier();
    if (!half) gp_q[t][j] = (gp_po[0][j] + gp_po[1][j]) * (1.0f / __builtin_sqrtf(128.0f));
  }
  lbarrier();
  {
    const u32 t = w, l = lane();
    float o[4], s[4];
    for (u32 q = 0; q < 4u; q++) o[q] = gp_q[t][32u * q + l], s[q] = wave_sum_all(o[q] * o[q]);
    const float n3 = s[0] + s[1] + s[2] + s[3];
    for (u32 q = 0; q < 4u; q++) {
      const u32 jj = 32u * q + l;
      const float z = in[t * is + CD + h * 128u + jj];
      const float y = o[q] * (1.0f / __builtin_sqrtf(n3 / 128.0f + eps)) * nw[jj] * (z / (1.0f + __builtin_expf(-z)));
      if (out) out[t * os + h * 128u + jj] = y;
      if (oq) quant_wave(y, (h * 128u + 32u * q) >> 5, oq + t * qs, inner >> 5);
    }
  }
}

/* ============================================================================
 * Round 7: Bitwise-exact prefill raw-quant WMMA GEMM specializations
 *   - Direct VGPR weight prefetch via rq_hload (eliminating 16.9 KiB lw LDS),
 *     dropping LDS to <= 20.5 KiB (3 WGs/CU for Q2_K/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ3_S,
 *     2 WGs/CU for IQ2_S/IQ1_S/IQ1_M) vs 35.3-39.4 KiB (1 WG/CU) in base ie_gemm_rq_*.
 *   - lbarrier() (LDS-only barrier: eliminates per-superblock global_inv L0 wipes).
 *   - Hoisted per-half-block float d (and mn for Q2_K).
 *   - Q2_K: exact packed _Float16x2 (sa0, sa1) sums in LDS (every int8 sum16 in
 *     [-2048, +2032] is bitwise exact in IEEE-754 binary16), 1x ds_load_b32 +
 *     v_permlane16_b32 half-wave broadcast + v_fma_mix_f32 (16x less lsa LDS traffic,
 *     zero short->float unpack ALU).
 *   - ie_saluki_gemm_rq128_<type>: 128 rows x 128 tokens/WG (2x less weight traffic
 *     and weight decode at T=512 / zero-pad multiples of 128).
 *   - ie_saluki_gemm_rq64_<type>: 128 rows x 64 tokens/WG (warm tails T=65..255).
 * Bitwise equal (0 bit diffs) to ie_gemm_rq_<type> across all shapes/bias/res/strides.
 * ========================================================================== */
typedef _Float16 saluki_f16;
typedef _Float16 saluki_f16x2 __attribute__((ext_vector_type(2)));

static inline __attribute__((always_inline)) u32 saluki_bcast16_u(const u32 v, const u32 idx) {
  const u32 sel = 0x11111111u * idx;
  return (u32)__builtin_amdgcn_permlane16(0, (int)v, (int)sel, (int)sel, false, false);
}

static inline __attribute__((always_inline)) float saluki_q2k_sub_f16x2(float t, const u32 pair, const float nm0, const float nm1) {
  const saluki_f16x2 s = __builtin_bit_cast(saluki_f16x2, pair);
  t += (float)s.x * nm0;
  t += (float)s.y * nm1;
  return t;
}

static inline __attribute__((always_inline)) float saluki_rq_hb_d(const u32 type, const rq_hblk_t *x) {
  float d;
  if (type == IE_RQ_Q2_K) {
    d = ie_rq_h2f(x->r[0] & 0xFFFFu);
  } else if (type == IE_RQ_IQ1_M) {
    const u32 s0 = x->r[0] & 0xFFFFu, s1 = x->r[0] >> 16u, s2 = x->r[1] & 0xFFFFu, s3 = x->r[1] >> 16u;
    d = ie_rq_h2f((s0 >> 12u) | ((s1 >> 8u) & 0x00F0u) | ((s2 >> 4u) & 0x0F00u) | (s3 & 0xF000u));
  } else {
    d = ie_rq_h2f((x->r[0] >> x->sh) & 0xFFFFu);
  }
  __asm__ volatile("" : "+v"(d));
  return d;
}

static inline __attribute__((always_inline)) rq_sub_t saluki_rq_hdec(
    const u32 type, const rq_hblk_t *x, const float d, const LDSP u32 *lut,
    const u32 hb, const u32 k, const u32 h) {
  rq_sub_t r;
  r.s0 = r.s1 = r.m0 = r.m1 = 0.0f;
  r.w0 = r.w1 = (v2i)(0);
  const u32 sh = x->sh, qsh = sh ^ 16u;
  const u32 psel = h ? 0x07060302u : 0x05040100u;
  switch (type) {
    case IE_RQ_IQ2_XXS: {
      const u32 a0 = rq_align(x->r[1u + 2u * k], x->r[2u + 2u * k], qsh);
      const u32 a1 = rq_align(x->r[2u + 2u * k], x->r[3u + 2u * k], qsh);
      r.s0 = (d * (0.5f + (float)(a1 >> 28u))) * 0.25f;
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const u32 g8 = 2u * q + h;
        const u32 s8 = rq_ksign7((a1 >> (7u * g8)) & 127u);
        const u32 cp = lut[(a0 >> (8u * g8)) & 255u];
        const v2i v = {(int)RQ_SGN_B(cp & 0x03030303u, s8 & 15u), (int)RQ_SGN_B((cp >> 4u) & 0x03030303u, s8 >> 4u)};
        if (q == 0u) r.w0 = v;
        else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ2_XS: {
      const u32 sc = (rq_align(x->r[1], x->r[2], qsh) >> (8u * k)) & 255u;
      r.s0 = (d * (0.5f + (float)(sc & 15u))) * 0.25f;
      r.s1 = (d * (0.5f + (float)(sc >> 4u))) * 0.25f;
      const u32 w0 = rq_align(x->r[3u + 2u * k], x->r[4u + 2u * k], qsh);
      const u32 w1 = rq_align(x->r[4u + 2u * k], x->r[5u + 2u * k], qsh);
      const u32 qp = __builtin_amdgcn_perm(w1, w0, psel);
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const u32 q2 = (qp >> (16u * q)) & 0xFFFFu;
        const u32 s8 = rq_ksign7(q2 >> 9u);
        const u32 cp = lut[q2 & 511u];
        const v2i v = {(int)RQ_SGN_B(cp & 0x03030303u, s8 & 15u), (int)RQ_SGN_B((cp >> 4u) & 0x03030303u, s8 >> 4u)};
        if (q == 0u) r.w0 = v;
        else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ2_S: {
      const u32 qh = (rq_align(x->r[1], x->r[2], qsh) >> (8u * k)) & 255u;
      const u32 sc = (rq_align(x->r[3], x->r[4], qsh) >> (8u * k)) & 255u;
      r.s0 = (d * (0.5f + (float)(sc & 15u))) * 0.25f;
      r.s1 = (d * (0.5f + (float)(sc >> 4u))) * 0.25f;
      const u32 qlo = rq_align(x->r[5u + k], x->r[6u + k], qsh);
      const u32 sgn = rq_align(x->r[10u + k], x->r[11u + k], qsh);
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const u32 g8 = 2u * q + h;
        const u32 idx = ie_byte(qlo, g8) | ((qh << (8u - 2u * g8)) & 0x300u);
        const u32 sb = ie_byte(sgn, g8);
        const u32 cp = lut[idx];
        const v2i v = {(int)RQ_SGN_B(cp & 0x03030303u, sb & 15u), (int)RQ_SGN_B((cp >> 4u) & 0x03030303u, sb >> 4u)};
        if (q == 0u) r.w0 = v;
        else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ3_XXS: {
      const u32 aux = rq_align(x->r[1u + k], x->r[2u + k], qsh);
      const u32 w0 = rq_align(x->r[6u + 2u * k], x->r[7u + 2u * k], qsh);
      const u32 w1 = rq_align(x->r[7u + 2u * k], x->r[8u + 2u * k], qsh);
      const u32 qp = __builtin_amdgcn_perm(w1, w0, psel);
      r.s0 = (d * (0.5f + (float)(aux >> 28u))) * 0.5f;
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const u32 g8 = 2u * q + h, w16 = (qp >> (16u * q)) & 0xFFFFu;
        const u32 s8 = rq_ksign7((aux >> (7u * g8)) & 127u);
        const u32 g1 = lut[w16 & 255u], g2 = lut[w16 >> 8u];
        const v2i v = {(int)ie_rq_ssign(g1, ie_rq_mask4(s8 & 15u)), (int)ie_rq_ssign(g2, ie_rq_mask4(s8 >> 4u))};
        if (q == 0u) r.w0 = v;
        else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ3_S: {
      const u32 qh = (rq_align(x->r[1], x->r[2], qsh) >> (8u * k)) & 255u;
      const u32 sc_sh = sh ^ (hb ? 0u : 16u);
      const u32 sc = ((x->r[3] >> sc_sh) >> (4u * k)) & 15u;
      r.s0 = d * (float)(int)(1u + 2u * sc);
      const u32 sgn = rq_align(x->r[4u + k], x->r[5u + k], qsh);
      const u32 w0 = rq_align(x->r[9u + 2u * k], x->r[10u + 2u * k], qsh);
      const u32 w1 = rq_align(x->r[10u + 2u * k], x->r[11u + 2u * k], qsh);
      const u32 qp = __builtin_amdgcn_perm(w1, w0, psel);
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const u32 g8 = 2u * q + h, w16 = (qp >> (16u * q)) & 0xFFFFu, sb = ie_byte(sgn, g8);
        const u32 i1 = (w16 & 255u) | (((qh >> (2u * g8)) & 1u) << 8u);
        const u32 i2 = (w16 >> 8u) | (((qh >> (2u * g8 + 1u)) & 1u) << 8u);
        const v2i v = {(int)ie_rq_ssign(lut[i1], ie_rq_mask4(sb & 15u)), (int)ie_rq_ssign(lut[i2], ie_rq_mask4(sb >> 4u))};
        if (q == 0u) r.w0 = v;
        else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ1_S: {
      const u32 qhw = rq_align(x->r[1u + (k >> 1u)], x->r[2u + (k >> 1u)], qsh);
      const u32 qh = (qhw >> (16u * (k & 1u))) & 0xFFFFu;
      r.s0 = (d * (float)(int)(2u * ((qh >> 12u) & 7u) + 1u)) * 0.125f;
      const u32 tbl = (qh & 0x8000u) != 0u ? RQ_IQ1_TBL_NEG : RQ_IQ1_TBL_POS;
      const u32 qlo = rq_align(x->r[4u + k], x->r[5u + k], qsh);
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const u32 g8 = 2u * q + h;
        const u32 idx = ie_byte(qlo, g8) | (((qh >> (3u * g8)) & 7u) << 8u);
        const u32 cp = lut[idx];
        const v2i v = {(int)rq_iq1_map(cp, tbl), (int)rq_iq1_map(cp >> 4u, tbl)};
        if (q == 0u) r.w0 = v;
        else r.w1 = v;
      }
      break;
    }
    case IE_RQ_IQ1_M: {
      const u32 pair = hb != 0u ? x->r[1] : x->r[0];
      const u32 scv = pair >> (16u * (k >> 1u)), ksh = 6u * (k & 1u);
      r.s0 = (d * (float)(int)(2u * ((scv >> ksh) & 7u) + 1u)) * 0.125f;
      r.s1 = (d * (float)(int)(2u * ((scv >> (ksh + 3u)) & 7u) + 1u)) * 0.125f;
      const u32 qhb = (x->r[2u + (k >> 1u)] >> (16u * (k & 1u))) & 0xFFFFu;
      const u32 qlo = x->r[4u + k];
#pragma unroll
      for (u32 q = 0; q < 2u; q++) {
        const u32 g8 = 2u * q + h, qh = q == 0u ? (qhb & 255u) : (qhb >> 8u);
        const u32 idx = ie_byte(qlo, g8) | ((qh << (8u - 4u * h)) & 0x700u);
        const u32 tbl = (qh & (h != 0u ? 0x80u : 0x08u)) != 0u ? RQ_IQ1_TBL_NEG : RQ_IQ1_TBL_POS;
        const u32 cp = lut[idx];
        const v2i v = {(int)rq_iq1_map(cp, tbl), (int)rq_iq1_map(cp >> 4u, tbl)};
        if (q == 0u) r.w0 = v;
        else r.w1 = v;
      }
      break;
    }
    default:
      break;
  }
  return r;
}

static inline __attribute__((always_inline)) void saluki_rq_store_out(
    G u8 *y, const G float *bias, const G u8 *res, const u32 rows, const u32 T,
    const u32 ys, const u32 rs, const u32 row, const u32 tbase, const u32 wj, const u32 h, const f8 *acc) {
#pragma unroll
  for (u32 j = 0; j < wj; j++) {
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      const u32 t = tbase + 16u * j + 8u * h + v;
      if (t < T && row < rows) {
        const G float *rp = res ? (const G float *)(res + (unsigned long)t * rs) : (const G float *)0;
        ((G float *)(y + (unsigned long)t * ys))[row] = epilogue(acc[j][v], row, bias, rp);
      }
    }
  }
}

static inline __attribute__((always_inline)) void saluki_rq_iq_half(
    const u32 type, const rq_hblk_t *cur, const LDSP u32 *lut, const u32 hb,
    const LDSP u8 *lx, const LDSP float *lsx, const u32 ntok, const u32 wj, const u32 chunk_j,
    const u32 l, const u32 h, f8 *acc) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  const float d = saluki_rq_hb_d(type, cur);
#pragma unroll
  for (u32 k = 0; k < RQ_KB; k++) {
    const rq_sub_t dc = saluki_rq_hdec(type, cur, d, lut, hb, k, h);
#pragma unroll
    for (u32 j0 = 0; j0 < wj; j0 += chunk_j) {
      v2i a0[4], a1[4];
      v8i c0[4], c1[4];
#pragma unroll
      for (u32 j = 0; j < chunk_j; j++) {
        const u32 tk = 16u * (j0 + j) + (l & 15u);
        a0[j] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k) + 8u * h);
        a1[j] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k + 1u) + 8u * h);
      }
      if (rq_split16(type)) {
#pragma unroll
        for (u32 j = 0; j < chunk_j; j++) {
          c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0[j], 1, dc.w0, z, 0);
          c1[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1[j], 1, dc.w1, z, 0);
        }
      } else {
#pragma unroll
        for (u32 j = 0; j < chunk_j; j++) {
          c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0[j], 1, dc.w0, z, 0);
          c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1[j], 1, dc.w1, c0[j], 0);
        }
      }
#pragma unroll
      for (u32 j = 0; j < chunk_j; j++) {
        const u32 ts = 16u * (j0 + j) + 8u * h;
        const f8 dx = *(const LDSP f8 *)(lsx + k * ntok + ts);
        f8 t = __builtin_convertvector(c0[j], f8) * dc.s0;
        if (rq_split16(type)) t += __builtin_convertvector(c1[j], f8) * dc.s1;
        acc[j0 + j] += t * dx;
        __asm__ volatile("" : "+v"(acc[j0 + j]));
      }
    }
  }
}

static inline __attribute__((always_inline)) void saluki_q2k_store_lsa(
    LDSP u32 *lsa, const u32 ntok, const u32 tk, const u32 c8, const u32x4 rx) {
  const saluki_f16 hf = (saluki_f16)rqc_sum16i(rx);
  const u32 bits = (u32)__builtin_bit_cast(u16, hf);
  const u32 other = (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ 1u) * 4u), (int)bits);
  if ((c8 & 1u) == 0u) lsa[(c8 >> 1u) * ntok + tk] = bits | (other << 16u);
}

static inline __attribute__((always_inline)) void saluki_rq_q2k_half(
    const rq_hblk_t *cur, const LDSP u8 *lx, const LDSP float *lsx, const LDSP u32 *lsa,
    const u32 ntok, const u32 wj, const u32 chunk_j, const u32 l, const u32 h, f8 *acc) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  float d = ie_rq_h2f(cur->r[0] & 0xFFFFu);
  float mn = ie_rq_h2f(cur->r[0] >> 16u);
  __asm__ volatile("" : "+v"(d), "+v"(mn));
  const u32 r16 = l & 15u;
#pragma unroll
  for (u32 k = 0; k < RQ_KB; k++) {
    const u32 sw = cur->r[1u + (k >> 1u)] >> (16u * (k & 1u));
    const u32 sc0 = sw & 255u, sc1 = (sw >> 8u) & 255u, ksh = 2u * k;
    v2i w0, w1;
    w0.x = (int)(((cur->r[3] >> ksh) & 0x03030303u) * (sc0 & 15u));
    w0.y = (int)(((cur->r[4] >> ksh) & 0x03030303u) * (sc0 & 15u));
    w1.x = (int)(((cur->r[5] >> ksh) & 0x03030303u) * (sc1 & 15u));
    w1.y = (int)(((cur->r[6] >> ksh) & 0x03030303u) * (sc1 & 15u));
    float nm0 = -(mn * (float)(sc0 >> 4u));
    float nm1 = -(mn * (float)(sc1 >> 4u));
    __asm__ volatile("" : "+v"(nm0), "+v"(nm1));
#pragma unroll
    for (u32 j0 = 0; j0 < wj; j0 += chunk_j) {
      v2i a0[4], a1[4];
      v8i c0[4];
#pragma unroll
      for (u32 j = 0; j < chunk_j; j++) {
        const u32 tk = 16u * (j0 + j) + (l & 15u);
        a0[j] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k) + 8u * h);
        a1[j] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k + 1u) + 8u * h);
      }
#pragma unroll
      for (u32 j = 0; j < chunk_j; j++) {
        c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0[j], 1, w0, z, 0);
        c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1[j], 1, w1, c0[j], 0);
      }
      const u32 perm_t = 16u * (j0 + (r16 >> 3u)) + 8u * h + (r16 & 7u);
      const u32 sa_p0 = lsa[k * ntok + perm_t];
      const u32 sa_p1 = (chunk_j == 4u) ? lsa[k * ntok + perm_t + 32u] : 0u;
#pragma unroll
      for (u32 j = 0; j < chunk_j; j++) {
        const u32 ts = 16u * (j0 + j) + 8u * h;
        const f8 dx = *(const LDSP f8 *)(lsx + k * ntok + ts);
        f8 t = __builtin_convertvector(c0[j], f8) * d;
        const u32 src = (j < 2u) ? sa_p0 : sa_p1;
        const u32 base = (j & 1u) * 8u;
#pragma unroll
        for (u32 v = 0; v < 8u; v++) t[v] = saluki_q2k_sub_f16x2(t[v], saluki_bcast16_u(src, base + v), nm0, nm1);
        acc[j0 + j] += t * dx;
        __asm__ volatile("" : "+v"(acc[j0 + j]));
      }
    }
  }
}

static inline __attribute__((always_inline)) void saluki_rq_run_128(
    const u32 type, const u32 chunk_j, const G u8 *q, const G u8 *xq, G u8 *y,
    const u32 rows, const u32 nb, const u32 T, const G float *bias, const G u8 *res,
    const u32 xs, const u32 ys, const u32 rs, LDSP u8 *lx, LDSP float *lsx, LDSP u32 *lut, LDSP u32 *lsa) {
  if (T == 0u || rows == 0u || nb < 8u || (nb & 7u) != 0u || !q || !xq || !y) return;
  const u32 ntt = (T + 127u) / 128u, tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 rbase = rt * 128u, tbase = tt * 128u;
  if (rbase >= rows || tbase >= T) return;
  const u32 t0 = tid(), w = t0 >> 5u, l = lane(), h = l >> 4u;
  const u32 r_loc = 16u * w + (l & 15u), row = rbase + r_loc, rr = row < rows ? row : rows - 1u;
  const u32 mb = nb >> 3u, bs = ie_rq_bs(type);
  const G u8 *rowp = q + (unsigned long)rr * mb * bs;
  f8 acc[8];
#pragma unroll
  for (u32 j = 0; j < 8u; j++) acc[j] = (f8)(0.0f);
  if (type != IE_RQ_Q2_K) rq_stage_lut_raw(type, (LDSP ie_rq_u64 *)0, lut, 0);
  const G u8 *xp[4];
#pragma unroll
  for (u32 i = 0; i < 4u; i++) {
    const u32 p = t0 + 256u * i, tk = p >> 3u, gt = tbase + tk < T ? tbase + tk : T - 1u;
    xp[i] = xq + (unsigned long)gt * xs + (p & 7u) * 16u;
  }
  const u32 gs = tbase + (t0 & 127u) < T ? tbase + (t0 & 127u) : T - 1u;
  const G float *sxp = (const G float *)(xq + (unsigned long)gs * xs + 32ul * nb);
  u32x4 rx[4];
  f32x4 rsx = (f32x4)(0.0f);
#pragma unroll
  for (u32 i = 0; i < 4u; i++) rx[i] = *(const G u32x4 *)xp[i];
  if (t0 < 128u) rsx = rq_load_f32x4(sxp);
  rq_hblk_t cur = rq_hload(type, rowp, 0u, h);
#pragma unroll
  for (u32 i = 0; i < 4u; i++) {
    const u32 p = t0 + 256u * i, tk = p >> 3u, c8 = p & 7u;
    *rqc_lx_slot(lx, tk, c8) = rx[i];
    if (type == IE_RQ_Q2_K) saluki_q2k_store_lsa(lsa, 128u, tk, c8, rx[i]);
  }
  if (t0 < 128u) { lsx[t0] = rsx.x; lsx[128u + t0] = rsx.y; lsx[256u + t0] = rsx.z; lsx[384u + t0] = rsx.w; }
  lbarrier();
  const u32 nc = 2u * mb;
#pragma unroll 1
  for (u32 c = 0; c < nc; c++) {
    const int more = c + 1u < nc;
    const u32 cn = more ? c + 1u : c;
    rq_hblk_t nxt = rq_hload(type, rowp + (unsigned long)(cn >> 1u) * bs, cn & 1u, h);
    if (more) {
      const u32 kb = 4u * (c + 1u);
#pragma unroll
      for (u32 i = 0; i < 4u; i++) rx[i] = *(const G u32x4 *)(xp[i] + kb * 32u);
      if (t0 < 128u) rsx = rq_load_f32x4(sxp + kb);
    }
    __builtin_amdgcn_sched_barrier(0);
    if (type == IE_RQ_Q2_K) saluki_rq_q2k_half(&cur, lx, lsx, lsa, 128u, 8u, chunk_j, l, h, acc);
    else saluki_rq_iq_half(type, &cur, lut, c & 1u, lx, lsx, 128u, 8u, chunk_j, l, h, acc);
    __builtin_amdgcn_sched_barrier(0);
    if (more) {
      lbarrier();
#pragma unroll
      for (u32 i = 0; i < 4u; i++) {
        const u32 p = t0 + 256u * i, tk = p >> 3u, c8 = p & 7u;
        *rqc_lx_slot(lx, tk, c8) = rx[i];
        if (type == IE_RQ_Q2_K) saluki_q2k_store_lsa(lsa, 128u, tk, c8, rx[i]);
      }
      if (t0 < 128u) { lsx[t0] = rsx.x; lsx[128u + t0] = rsx.y; lsx[256u + t0] = rsx.z; lsx[384u + t0] = rsx.w; }
      lbarrier();
    }
    cur = nxt;
  }
  saluki_rq_store_out(y, bias, res, rows, T, ys, rs, row, tbase, 8u, h, acc);
}

static inline __attribute__((always_inline)) void saluki_rq_run_64(
    const u32 type, const u32 chunk_j, const G u8 *q, const G u8 *xq, G u8 *y,
    const u32 rows, const u32 nb, const u32 T, const G float *bias, const G u8 *res,
    const u32 xs, const u32 ys, const u32 rs, LDSP u8 *lx, LDSP float *lsx, LDSP u32 *lut, LDSP u32 *lsa) {
  if (T == 0u || rows == 0u || nb < 8u || (nb & 7u) != 0u || !q || !xq || !y) return;
  const u32 ntt = (T + 63u) / 64u, tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 rbase = rt * 128u, tbase = tt * 64u;
  if (rbase >= rows || tbase >= T) return;
  const u32 t0 = tid(), w = t0 >> 5u, l = lane(), h = l >> 4u;
  const u32 r_loc = 16u * w + (l & 15u), row = rbase + r_loc, rr = row < rows ? row : rows - 1u;
  const u32 mb = nb >> 3u, bs = ie_rq_bs(type);
  const G u8 *rowp = q + (unsigned long)rr * mb * bs;
  LDSP u8 *lx0 = lx, *lx1 = lx + 64u * 128u;
  LDSP float *lsx0 = lsx, *lsx1 = lsx + 4u * 64u;
  LDSP u32 *lsa0 = lsa, *lsa1 = lsa ? lsa + 4u * 64u : (LDSP u32 *)0;
  f8 acc[4];
#pragma unroll
  for (u32 j = 0; j < 4u; j++) acc[j] = (f8)(0.0f);
  if (type != IE_RQ_Q2_K) rq_stage_lut_raw(type, (LDSP ie_rq_u64 *)0, lut, 0);
  const G u8 *xp[2];
#pragma unroll
  for (u32 i = 0; i < 2u; i++) {
    const u32 p = t0 + 256u * i, tk = p >> 3u, gt = tbase + tk < T ? tbase + tk : T - 1u;
    xp[i] = xq + (unsigned long)gt * xs + (p & 7u) * 16u;
  }
  const u32 gs = tbase + (t0 & 63u) < T ? tbase + (t0 & 63u) : T - 1u;
  const G float *sxp = (const G float *)(xq + (unsigned long)gs * xs + 32ul * nb);
  u32x4 rx[2];
  f32x4 rsx = (f32x4)(0.0f);
#pragma unroll
  for (u32 i = 0; i < 2u; i++) rx[i] = *(const G u32x4 *)xp[i];
  if (t0 < 64u) rsx = rq_load_f32x4(sxp);
  rq_hblk_t cur = rq_hload(type, rowp, 0u, h);
#pragma unroll
  for (u32 i = 0; i < 2u; i++) {
    const u32 p = t0 + 256u * i, tk = p >> 3u, c8 = p & 7u;
    *rqc_lx_slot(lx0, tk, c8) = rx[i];
    if (type == IE_RQ_Q2_K) saluki_q2k_store_lsa(lsa0, 64u, tk, c8, rx[i]);
  }
  if (t0 < 64u) { lsx0[t0] = rsx.x; lsx0[64u + t0] = rsx.y; lsx0[128u + t0] = rsx.z; lsx0[192u + t0] = rsx.w; }
  lbarrier();
  const u32 nc = 2u * mb;
#pragma unroll 1
  for (u32 c = 0; c < nc; c++) {
    const u32 bf = c & 1u;
    const int more = c + 1u < nc;
    const u32 cn = more ? c + 1u : c;
    rq_hblk_t nxt = rq_hload(type, rowp + (unsigned long)(cn >> 1u) * bs, cn & 1u, h);
    if (more) {
      const u32 kb = 4u * (c + 1u);
#pragma unroll
      for (u32 i = 0; i < 2u; i++) rx[i] = *(const G u32x4 *)(xp[i] + kb * 32u);
      if (t0 < 64u) rsx = rq_load_f32x4(sxp + kb);
    }
    __builtin_amdgcn_sched_barrier(0);
    if (type == IE_RQ_Q2_K)
      saluki_rq_q2k_half(&cur, bf ? lx1 : lx0, bf ? lsx1 : lsx0, bf ? lsa1 : lsa0, 64u, 4u, chunk_j, l, h, acc);
    else
      saluki_rq_iq_half(type, &cur, lut, bf, bf ? lx1 : lx0, bf ? lsx1 : lsx0, 64u, 4u, chunk_j, l, h, acc);
    __builtin_amdgcn_sched_barrier(0);
    if (more) {
      LDSP u8 *lxn = bf ? lx0 : lx1;
      LDSP float *lsxn = bf ? lsx0 : lsx1;
      LDSP u32 *lsan = bf ? lsa0 : lsa1;
#pragma unroll
      for (u32 i = 0; i < 2u; i++) {
        const u32 p = t0 + 256u * i, tk = p >> 3u, c8 = p & 7u;
        *rqc_lx_slot(lxn, tk, c8) = rx[i];
        if (type == IE_RQ_Q2_K) saluki_q2k_store_lsa(lsan, 64u, tk, c8, rx[i]);
      }
      if (t0 < 64u) { lsxn[t0] = rsx.x; lsxn[64u + t0] = rsx.y; lsxn[128u + t0] = rsx.z; lsxn[192u + t0] = rsx.w; }
      lbarrier();
    }
    cur = nxt;
  }
  saluki_rq_store_out(y, bias, res, rows, T, ys, rs, row, tbase, 4u, h, acc);
}

KERNEL ie_saluki_gemm_rq128_q2k(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T,
                                const G float *bias, const G u8 *res, u32 xs, u32 ys, u32 rs) {
  static LDS u8 lx[128u * 128u] __attribute__((aligned(16)));
  static LDS float lsx[4u * 128u] __attribute__((aligned(32)));
  static LDS u32 lsa[4u * 128u] __attribute__((aligned(16)));
  saluki_rq_run_128(IE_RQ_Q2_K, 2u, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, (LDSP u32 *)0, lsa);
}
KERNEL ie_saluki_gemm_rq64_q2k(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T,
                               const G float *bias, const G u8 *res, u32 xs, u32 ys, u32 rs) {
  static LDS u8 lx[2u * 64u * 128u] __attribute__((aligned(16)));
  static LDS float lsx[2u * 4u * 64u] __attribute__((aligned(32)));
  static LDS u32 lsa[2u * 4u * 64u] __attribute__((aligned(16)));
  saluki_rq_run_64(IE_RQ_Q2_K, 4u, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, (LDSP u32 *)0, lsa);
}

#define SALUKI_DEF_RQ_IQ(tag_, type_, nlut_, cj128_, cj64_) \
  KERNEL ie_saluki_gemm_rq128_##tag_(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T, \
                                     const G float *bias, const G u8 *res, u32 xs, u32 ys, u32 rs) { \
    static LDS u8 lx[128u * 128u] __attribute__((aligned(16))); \
    static LDS float lsx[4u * 128u] __attribute__((aligned(32))); \
    static LDS u32 lut[nlut_] __attribute__((aligned(16))); \
    saluki_rq_run_128(type_, cj128_, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, lut, (LDSP u32 *)0); \
  } \
  KERNEL ie_saluki_gemm_rq64_##tag_(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T, \
                                    const G float *bias, const G u8 *res, u32 xs, u32 ys, u32 rs) { \
    static LDS u8 lx[2u * 64u * 128u] __attribute__((aligned(16))); \
    static LDS float lsx[2u * 4u * 64u] __attribute__((aligned(32))); \
    static LDS u32 lut[nlut_] __attribute__((aligned(16))); \
    saluki_rq_run_64(type_, cj64_, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, lx, lsx, lut, (LDSP u32 *)0); \
  }

SALUKI_DEF_RQ_IQ(iq2xxs, IE_RQ_IQ2_XXS, 256u,  4u, 4u)
SALUKI_DEF_RQ_IQ(iq2xs,  IE_RQ_IQ2_XS,  512u,  2u, 4u)
SALUKI_DEF_RQ_IQ(iq3xxs, IE_RQ_IQ3_XXS, 256u,  4u, 4u)
SALUKI_DEF_RQ_IQ(iq1s,   IE_RQ_IQ1_S,   2048u, 4u, 4u)
SALUKI_DEF_RQ_IQ(iq2s,   IE_RQ_IQ2_S,   1024u, 2u, 4u)
SALUKI_DEF_RQ_IQ(iq1m,   IE_RQ_IQ1_M,   2048u, 2u, 4u)
SALUKI_DEF_RQ_IQ(iq3s,   IE_RQ_IQ3_S,   512u,  2u, 4u)
#undef SALUKI_DEF_RQ_IQ

/* ============================================================================
 * Round 8: Bitwise-exact prefill & verify-decode specializations (0 spills)
 *   - ie_saluki_attn_pl:              Q-in-VGPR + PL_D=3 branchless producer
 *                                     + 4-deep K/V LDS ping-pong (Saluki hd=256, nh=24, nkv=4, q8_0 KV)
 *   - ie_saluki_swiglu_v4:            8-lane subgroup f32x4 SwiGLU + Q8 quant
 *   - ie_saluki_gdn_seq_wmma_gnorm:   Wave-0 expG/expD + conflict-free 64b K^T + v_perm BF16 + 4-way gnorm
 *   - ie_saluki_gemm_rq128_iq2xxs_sb: Paired-superblock 128x128 IQ2_XXS (6 ld/SB, fixed waitcnt, 100% dual-fmac)
 *   - ie_saluki_gemm_rq128_q2k_pipe:  Pipelined 128x128 Q2_K (fixed waitcnt, scalar VGPR acc)
 *   - ie_saluki_iq2xxs_tr9_pipe1:     1-subblock pipelined T=8 IQ2_XXS verify (nb=160 shapes)
 *   - ie_saluki_attn_long_mp_pipe1:   Fixed V-prefetch pipeline + wave-private LDS key scales (T=8 std)
 *   - ie_saluki_attn_long_mp_tr_nospill: Zero-spill (48->0) tree-verify long attention + fixed V-prefetch
 * All 8 kernels are 100% bitwise-exact (0-bit-diff) against base ie_kernels_q8.hsaco.
 * ========================================================================== */

/* --- R8-1: ie_saluki_attn_pl (hd=256, n_head=24, n_kv=4, q8_0 KV) --- */
#define R8_PL_NB   16u
#define R8_PL_KS   256u
#define R8_PL_VS   16u
#define R8_PL_ROWB 272u
#define R8_PL_RSTR (4u * R8_PL_ROWB)
#define R8_PL_BSTR (16u * R8_PL_RSTR)

typedef _Float16 r8_h2 __attribute__((ext_vector_type(2)));
typedef unsigned int r8_u2 __attribute__((ext_vector_type(2)));

static LDS f16 r8_pl_k[2][R8_PL_NB * R8_PL_KS] __attribute__((aligned(512)));
static LDS f16 r8_pl_vt[2][256u * R8_PL_VS] __attribute__((aligned(512)));

static inline u32 r8_pl_k_off(u32 r, u32 c) { return r * 256u + 8u * (c ^ r); }
static inline u32 r8_pl_vt_off(u32 col, u32 h) { return 8u * ((2u * col + h) ^ ((col >> 2) & 1u) ^ ((col >> 3) & 7u)); }
static inline float r8_pl_xrow(float v) { return __builtin_bit_cast(float, xrow_i(__builtin_bit_cast(int, v))); }

static inline __attribute__((always_inline)) u32 r8_pl_k_base_b(u32 bf, u32 rl, u32 hl) {
  u32 b0 = (bf << 13) + 512u * rl + 16u * (rl ^ hl);
  __asm__("" : "+v"(b0));
  return b0;
}
static inline __attribute__((always_inline)) h8 r8_pl_ld_k(u32 k_b0, u32 ks) {
  const LDSP u8 *base = (const LDSP u8 *)r8_pl_k[0];
  return *(const LDSP h8 *)(base + (k_b0 ^ (32u * (ks & 7u))) + ((ks & 8u) ? 256u : 0u));
}
static inline __attribute__((always_inline)) u32 r8_pl_v_base_b(u32 bf, u32 vt_lane) {
  u32 b0 = (bf << 13) + 16u * vt_lane;
  __asm__("" : "+v"(b0));
  return b0;
}
static inline __attribute__((always_inline)) h8 r8_pl_ld_vt(u32 v_b0, u32 sub, u32 g) {
  const LDSP u8 *base = (const LDSP u8 *)r8_pl_vt[0];
  return *(const LDSP h8 *)(base + (v_b0 ^ (32u * sub)) + 512u * sub + 2048u * g);
}

static inline __attribute__((always_inline)) h8 r8_pl_deq8(u32 w0, u32 w1, u32 sbits) {
  const u32 mg = 0x64646464u, x0 = w0 ^ 0x80808080u, x1 = w1 ^ 0x80808080u;
  const f16 s = __builtin_bit_cast(f16, (u16)sbits);
  const r8_h2 sv = {s, s}, bias = {(f16)1152.0f, (f16)1152.0f};
  const r8_h2 r0 = (__builtin_bit_cast(r8_h2, __builtin_amdgcn_perm(mg, x0, 0x04010400u)) - bias) * sv;
  const r8_h2 r1 = (__builtin_bit_cast(r8_h2, __builtin_amdgcn_perm(mg, x0, 0x04030402u)) - bias) * sv;
  const r8_h2 r2 = (__builtin_bit_cast(r8_h2, __builtin_amdgcn_perm(mg, x1, 0x04010400u)) - bias) * sv;
  const r8_h2 r3 = (__builtin_bit_cast(r8_h2, __builtin_amdgcn_perm(mg, x1, 0x04030402u)) - bias) * sv;
  return (h8){r0[0], r0[1], r1[0], r1[1], r2[0], r2[1], r3[0], r3[1]};
}

static inline __attribute__((always_inline)) void r8_pl_deq4(u32 w, u32 sbits, u32 *a, u32 *b) {
  const u32 mg = 0x64646464u, x = w ^ 0x80808080u;
  const f16 s = __builtin_bit_cast(f16, (u16)sbits);
  const r8_h2 sv = {s, s}, bias = {(f16)1152.0f, (f16)1152.0f};
  *a = __builtin_bit_cast(u32, (__builtin_bit_cast(r8_h2, __builtin_amdgcn_perm(mg, x, 0x04010400u)) - bias) * sv);
  *b = __builtin_bit_cast(u32, (__builtin_bit_cast(r8_h2, __builtin_amdgcn_perm(mg, x, 0x04030402u)) - bias) * sv);
}

static inline __attribute__((always_inline)) void r8_pl_transpose8(const u32 *k, u32x4 *ev, u32x4 *od) {
  u32 e[4], o[4];
#pragma unroll
  for (u32 i = 0; i < 4u; i++) {
    e[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x05040100u);
    o[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x07060302u);
  }
  *ev = (u32x4){e[0], e[1], e[2], e[3]};
  *od = (u32x4){o[0], o[1], o[2], o[3]};
}

typedef struct {
  u32x4 kq[2][2];
  u32 ks0;
  u32 ks1;
  r8_u2 vq[8];
  u32 vsp;
} r8_pl_slot;

typedef struct {
  u32 k_off0;
  u32 ks_off0;
  u32 vs_off;
  u32 v_off0;
  u32 r0;
  u32 uk_base;
} r8_pl_poff;

static inline __attribute__((always_inline)) r8_pl_poff r8_pl_make_poff(u32 pt) {
  const u32 wi = pt >> 5, L = pt & 31u;
  const u32 r0 = pt >> 3, bi = pt & 7u;
  const u32 uk = L >> 2, wp = L & 3u;
  r8_pl_poff o;
  o.k_off0  = r0 * R8_PL_RSTR + 32u * bi;
  o.ks_off0 = r0 * R8_PL_RSTR + 256u + 2u * bi;
  o.vs_off  = (8u * wi + uk) * R8_PL_RSTR + 256u + 4u * wp;
  o.v_off0  = (8u * wi) * R8_PL_RSTR + 8u * L;
  o.r0      = r0;
  o.uk_base = 8u * wi + uk;
  return o;
}

static inline __attribute__((always_inline)) void r8_pl_load_fast(r8_pl_slot *s, const G u8 *kb_kh, const G u8 *vb_kh,
                                                                  const u32 b, const r8_pl_poff o) {
  const unsigned long boff = (unsigned long)b * R8_PL_BSTR;
  const G u8 *kb = kb_kh + boff;
  const G u8 *vb = vb_kh + boff;
  s->vsp = *(const G u32 *)(vb + o.vs_off);
  s->ks0 = (u32)(*(const G u16 *)(kb + o.ks_off0));
  s->ks1 = (u32)(*(const G u16 *)(kb + o.ks_off0 + 8u * R8_PL_RSTR));
  s->kq[0][0] = *(const G u32x4 *)(kb + o.k_off0);
  s->kq[0][1] = *(const G u32x4 *)(kb + o.k_off0 + 16u);
  s->kq[1][0] = *(const G u32x4 *)(kb + o.k_off0 + 8u * R8_PL_RSTR);
  s->kq[1][1] = *(const G u32x4 *)(kb + o.k_off0 + 8u * R8_PL_RSTR + 16u);
#pragma unroll
  for (u32 u = 0; u < 8u; u++)
    s->vq[u] = *(const G r8_u2 *)(vb + o.v_off0 + u * R8_PL_RSTR);
}

static inline __attribute__((always_inline)) void r8_pl_load(r8_pl_slot *s, const G u8 *kb_kh, const G u8 *vb_kh,
                                                             const u32 kend, const u32 b, const u32 pt,
                                                             const r8_pl_poff o) {
  const u32 p0 = 16u * b;
  if (p0 + 15u < kend) {
    r8_pl_load_fast(s, kb_kh, vb_kh, b, o);
  } else {
    const unsigned long boff = (unsigned long)b * R8_PL_BSTR;
    const G u8 *kb = kb_kh + boff;
    const G u8 *vb = vb_kh + boff;
    const u32 wi = pt >> 5;
    s->vsp = (p0 + o.uk_base < kend) ? *(const G u32 *)(vb + o.vs_off) : 0u;
    if (p0 + o.r0 < kend) {
      s->kq[0][0] = *(const G u32x4 *)(kb + o.k_off0);
      s->kq[0][1] = *(const G u32x4 *)(kb + o.k_off0 + 16u);
      s->ks0 = (u32)(*(const G u16 *)(kb + o.ks_off0));
    } else {
      s->kq[0][0] = s->kq[0][1] = (u32x4)(0u);
      s->ks0 = 0u;
    }
    if (p0 + o.r0 + 8u < kend) {
      s->kq[1][0] = *(const G u32x4 *)(kb + o.k_off0 + 8u * R8_PL_RSTR);
      s->kq[1][1] = *(const G u32x4 *)(kb + o.k_off0 + 8u * R8_PL_RSTR + 16u);
      s->ks1 = (u32)(*(const G u16 *)(kb + o.ks_off0 + 8u * R8_PL_RSTR));
    } else {
      s->kq[1][0] = s->kq[1][1] = (u32x4)(0u);
      s->ks1 = 0u;
    }
#pragma unroll
    for (u32 u = 0; u < 8u; u++) {
      const u32 p = p0 + 8u * wi + u;
      s->vq[u] = (p < kend) ? *(const G r8_u2 *)(vb + o.v_off0 + u * R8_PL_RSTR) : (r8_u2)(0u);
    }
  }
}

static inline __attribute__((always_inline)) void r8_pl_publish(const r8_pl_slot *s, const u32 bf, const u32 pt) {
  const u32 wi = pt >> 5, L = pt & 31u;
  const u32 r0 = pt >> 3, bi = pt & 7u;
  const u32 vbi = L >> 2, wp_need = vbi >> 1, sh_need = 16u * (vbi & 1u);

  u32 vs_raw[8];
#pragma unroll
  for (u32 u = 0; u < 8u; u++)
    vs_raw[u] = (u32)__builtin_amdgcn_ds_bpermute((int)((4u * u + wp_need) << 2), (int)s->vsp);

  {
    const u32 ks0 = s->ks0;
    const u32x4 w0 = s->kq[0][0], w1 = s->kq[0][1];
    *(LDSP h8 *)(r8_pl_k[bf] + r8_pl_k_off(r0, 4u * bi + 0u)) = r8_pl_deq8(w0.x, w0.y, ks0);
    *(LDSP h8 *)(r8_pl_k[bf] + r8_pl_k_off(r0, 4u * bi + 1u)) = r8_pl_deq8(w0.z, w0.w, ks0);
    *(LDSP h8 *)(r8_pl_k[bf] + r8_pl_k_off(r0, 4u * bi + 2u)) = r8_pl_deq8(w1.x, w1.y, ks0);
    *(LDSP h8 *)(r8_pl_k[bf] + r8_pl_k_off(r0, 4u * bi + 3u)) = r8_pl_deq8(w1.z, w1.w, ks0);

    const u32 ks1 = s->ks1;
    const u32x4 w2 = s->kq[1][0], w3 = s->kq[1][1];
    *(LDSP h8 *)(r8_pl_k[bf] + r8_pl_k_off(r0 + 8u, 4u * bi + 0u)) = r8_pl_deq8(w2.x, w2.y, ks1);
    *(LDSP h8 *)(r8_pl_k[bf] + r8_pl_k_off(r0 + 8u, 4u * bi + 1u)) = r8_pl_deq8(w2.z, w2.w, ks1);
    *(LDSP h8 *)(r8_pl_k[bf] + r8_pl_k_off(r0 + 8u, 4u * bi + 2u)) = r8_pl_deq8(w3.x, w3.y, ks1);
    *(LDSP h8 *)(r8_pl_k[bf] + r8_pl_k_off(r0 + 8u, 4u * bi + 3u)) = r8_pl_deq8(w3.z, w3.w, ks1);
  }

  u32 vs_sb[8];
#pragma unroll
  for (u32 u = 0; u < 8u; u++) vs_sb[u] = vs_raw[u] >> sh_need;

#pragma unroll
  for (u32 g = 0; g < 2u; g++) {
    u32 lo[8], hi[8];
#pragma unroll
    for (u32 u = 0; u < 8u; u++) {
      u32 a, b2;
      const u32 w = g ? s->vq[u].y : s->vq[u].x;
      r8_pl_deq4(w, vs_sb[u], &a, &b2);
      lo[u] = a;
      hi[u] = b2;
    }
    const u32 c0 = 8u * L + 4u * g;
    u32x4 e0, o0, e1, o1;
    r8_pl_transpose8(lo, &e0, &o0);
    r8_pl_transpose8(hi, &e1, &o1);
    *(LDSP u32x4 *)(r8_pl_vt[bf] + r8_pl_vt_off(c0 + 0u, wi)) = e0;
    *(LDSP u32x4 *)(r8_pl_vt[bf] + r8_pl_vt_off(c0 + 1u, wi)) = o0;
    *(LDSP u32x4 *)(r8_pl_vt[bf] + r8_pl_vt_off(c0 + 2u, wi)) = e1;
    *(LDSP u32x4 *)(r8_pl_vt[bf] + r8_pl_vt_off(c0 + 3u, wi)) = o1;
  }
}

static inline __attribute__((always_inline)) void r8_pl_epilogue(const v8f o[16], float ls, u32 t0, u32 rl, u32 hl,
                                                                 u32 T, u32 h, u32 n_head, G float *out,
                                                                 const G float *gate, u32 gstride, u32 os, u32 gs,
                                                                 G u8 *oq, u32 oqs) {
  const float inv = 1.0f / (ls + r8_pl_xrow(ls));
  const u32 tk = t0 + rl;
  const int inb = tk < T;
  const G float *gr = (gate && inb) ? (const G float *)((const G u8 *)gate + (unsigned long)tk * gs) + (unsigned long)h * gstride : (const G float *)0;
  G float *orow = (out && inb) ? (G float *)((G u8 *)out + (unsigned long)tk * os) + (unsigned long)h * 256u : (G float *)0;
  G u8 *qrow = (oq && inb) ? oq + (unsigned long)tk * oqs : (G u8 *)0;
  const u32 nbq = n_head * 8u;

#pragma clang loop unroll(disable)
  for (u32 blk = 0; blk < 8u; blk++) {
    v8f ot0, ot1;
    switch (blk) {
      case 0u: ot0 = o[0];  ot1 = o[1];  break;
      case 1u: ot0 = o[2];  ot1 = o[3];  break;
      case 2u: ot0 = o[4];  ot1 = o[5];  break;
      case 3u: ot0 = o[6];  ot1 = o[7];  break;
      case 4u: ot0 = o[8];  ot1 = o[9];  break;
      case 5u: ot0 = o[10]; ot1 = o[11]; break;
      case 6u: ot0 = o[12]; ot1 = o[13]; break;
      default: ot0 = o[14]; ot1 = o[15]; break;
    }
    const u32 dbase = 32u * blk + 8u * hl;
    float gv0[8], gv1[8];
    if (gr) {
      const G float *gb = gr + dbase;
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        gv0[v] = gb[v];
        gv1[v] = gb[v + 16u];
      }
    }
    float x0[8], x1[8];
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      float a0 = inb ? ot0[v] * inv : 0.0f;
      float a1 = inb ? ot1[v] * inv : 0.0f;
      if (gr) {
        a0 *= sigmoidf(gv0[v]);
        a1 *= sigmoidf(gv1[v]);
      }
      x0[v] = a0;
      x1[v] = a1;
      if (orow) {
        orow[dbase + v] = a0;
        orow[dbase + v + 16u] = a1;
      }
    }
    if (oq) {
      float am = 0.0f;
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        am = __builtin_fmaxf(am, __builtin_fabsf(x0[v]));
        am = __builtin_fmaxf(am, __builtin_fabsf(x1[v]));
      }
      const float amax = __builtin_fmaxf(am, r8_pl_xrow(am));
      const float d = amax / 127.0f;
      u32 sloc = 0u, p00 = 0u, p01 = 0u, p10 = 0u, p11 = 0u;
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const int q0 = d != 0.0f ? (int)__builtin_roundf(x0[v] / d) : 0;
        const int q1 = d != 0.0f ? (int)__builtin_roundf(x1[v] / d) : 0;
        sloc += (u32)q0 + (u32)q1;
        const u32 sh = 8u * (v & 3u);
        if (v < 4u) {
          p00 |= ((u32)(u8)q0) << sh;
          p10 |= ((u32)(u8)q1) << sh;
        } else {
          p01 |= ((u32)(u8)q0) << sh;
          p11 |= ((u32)(u8)q1) << sh;
        }
      }
      const u32 asum = sloc + (u32)xrow_i((int)sloc);
      if (qrow) {
        const u32 qb = h * 8u + blk;
        G u32 *qp = (G u32 *)(qrow + qb * 32u + 8u * hl);
        qp[0] = p00;
        qp[1] = p01;
        qp[4] = p10;
        qp[5] = p11;
        if (hl == 0u) {
          ((G float *)(qrow + 32u * nbq))[qb] = d;
          ((G u32 *)(qrow + 36u * nbq))[qb] = asum;
        }
      }
    }
  }
}

static inline __attribute__((always_inline)) void r8_pl_load_qreg(h8 qr_reg[16], const G float *q, const G f16 *q16,
                                                                  u32 tq, u32 h, u32 qs, u32 hl) {
  if (q16) {
    const G f16 *q16r = q16 + ((unsigned long)tq * 24u + h) * 256u + 8u * hl;
#pragma unroll
    for (u32 ks = 0u; ks < 16u; ks++) qr_reg[ks] = *(const G h8 *)(q16r + 16u * ks);
  } else {
    const float qscale = 0.0625f * 1.44269504088896341f;
    const G float *qr = (const G float *)((const G u8 *)q + (unsigned long)tq * qs + (unsigned long)h * 1024u) + 8u * hl;
#pragma unroll
    for (u32 ks = 0u; ks < 16u; ks++) {
      h8 qv;
#pragma unroll
      for (u32 e = 0u; e < 8u; e++) qv[e] = (f16)(qr[16u * ks + e] * qscale);
      qr_reg[ks] = qv;
    }
  }
}

__attribute__((amdgpu_waves_per_eu(4, 4)))
KERNEL ie_saluki_attn_pl(const G float *q, const G kvt *kc, const G kvt *vc, G float *out, u32 pos0, u32 hd,
                         u32 n_head, u32 n_kv, const G float *gate, u32 gstride, u32 qs, u32 os, u32 gs, u32 T,
                         G u8 *oq, u32 oqs, const G f16 *q16) {
  if (hd != 256u || n_head != 24u || n_kv != 4u || T == 0u) return;
  const u32 ntile = (T + 15u) >> 4, tile = wgid() % ntile, kh = wgid() / ntile, t0 = 16u * tile;
  if (kh >= 4u) return;
  const u32 w = __builtin_amdgcn_readfirstlane(tid() >> 5), l = lane(), hl = l >> 4, rl = l & 15u;
  const u32 tlast = t0 + 15u < T ? t0 + 15u : T - 1u, kend = pos0 + tlast + 1u;
  const u32 nblk = (kend + 15u) >> 4;
  if (w >= 6u) {
    const G u8 *kb_kh = (const G u8 *)kc + (unsigned long)kh * R8_PL_ROWB;
    const G u8 *vb_kh = (const G u8 *)vc + (unsigned long)kh * R8_PL_ROWB;
    const u32 pt = tid() - 192u;
    const r8_pl_poff po = r8_pl_make_poff(pt);
    const u32 nblk_full = kend >> 4;
    r8_pl_slot ring[3];
    if (nblk_full >= 6u) {
      r8_pl_load_fast(&ring[0], kb_kh, vb_kh, 0u, po);
      r8_pl_load_fast(&ring[1], kb_kh, vb_kh, 1u, po);
      r8_pl_load_fast(&ring[2], kb_kh, vb_kh, 2u, po);
      r8_pl_publish(&ring[0], 0u, pt);
      r8_pl_load_fast(&ring[0], kb_kh, vb_kh, 3u, po);
      u32 k0 = 0u;
      for (; k0 + 6u < nblk_full; k0 += 3u) {
        lbarrier();
        r8_pl_publish(&ring[1], (k0 + 1u) & 1u, pt);
        r8_pl_load_fast(&ring[1], kb_kh, vb_kh, k0 + 4u, po);
        lbarrier();
        r8_pl_publish(&ring[2], (k0 + 2u) & 1u, pt);
        r8_pl_load_fast(&ring[2], kb_kh, vb_kh, k0 + 5u, po);
        lbarrier();
        r8_pl_publish(&ring[0], (k0 + 3u) & 1u, pt);
        r8_pl_load_fast(&ring[0], kb_kh, vb_kh, k0 + 6u, po);
      }
#pragma clang loop unroll(disable)
      for (; k0 < nblk; k0 += 3u) {
#pragma unroll
        for (u32 j = 0; j < 3u; j++) {
          const u32 k = k0 + j;
          if (k < nblk) {
            lbarrier();
            if (k + 1u < nblk) {
              r8_pl_publish(&ring[(j + 1u) % 3u], (k + 1u) & 1u, pt);
              if (k + 4u < nblk) r8_pl_load(&ring[(j + 1u) % 3u], kb_kh, vb_kh, kend, k + 4u, pt, po);
            }
          }
        }
      }
    } else {
#pragma unroll
      for (u32 s = 0; s < 3u; s++)
        if (s < nblk) r8_pl_load(&ring[s], kb_kh, vb_kh, kend, s, pt, po);
      r8_pl_publish(&ring[0], 0u, pt);
      if (3u < nblk) r8_pl_load(&ring[0], kb_kh, vb_kh, kend, 3u, pt, po);
#pragma clang loop unroll(disable)
      for (u32 k0 = 0; k0 < nblk; k0 += 3u) {
#pragma unroll
        for (u32 j = 0; j < 3u; j++) {
          const u32 k = k0 + j;
          if (k < nblk) {
            lbarrier();
            if (k + 1u < nblk) {
              r8_pl_publish(&ring[(j + 1u) % 3u], (k + 1u) & 1u, pt);
              if (k + 4u < nblk) r8_pl_load(&ring[(j + 1u) % 3u], kb_kh, vb_kh, kend, k + 4u, pt, po);
            }
          }
        }
      }
    }
    return;
  }
  const u32 h = kh * 6u + w;
  const u32 tq = t0 + rl < T ? t0 + rl : T - 1u;
  h8 qr_reg[16];
  r8_pl_load_qreg(qr_reg, q, q16, tq, h, qs, hl);

  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2) & 1u) ^ (rl >> 3);
  const v8f zero = (v8f)(0.0f);
  v8f o[16];
#pragma unroll
  for (u32 nt = 0; nt < 16u; nt++) o[nt] = zero;
  float m = -__builtin_inff(), ls = 0.0f;
  const u32 tp = pos0 + t0 + rl;
  const u32 pos_t0 = pos0 + t0;

  for (u32 k = 0; k < nblk; k++) {
    lbarrier();
    const u32 kb = 16u * k, bf = k & 1u;
    const u32 k_b0 = r8_pl_k_base_b(bf, rl, hl);
    v8f sa = zero, sb = zero;
    h8 k0 = r8_pl_ld_k(k_b0, 0u);
    h8 k1 = r8_pl_ld_k(k_b0, 1u);
    h8 k2 = r8_pl_ld_k(k_b0, 2u);
    h8 k3 = r8_pl_ld_k(k_b0, 3u);
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (u32 g = 0u; g < 3u; g++) {
      const u32 ks = 4u * g;
      sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qr_reg[ks + 0u], sa);
      sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qr_reg[ks + 1u], sb);
      k0 = r8_pl_ld_k(k_b0, ks + 4u);
      k1 = r8_pl_ld_k(k_b0, ks + 5u);
      __builtin_amdgcn_sched_barrier(0);
      sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k2, qr_reg[ks + 2u], sa);
      sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k3, qr_reg[ks + 3u], sb);
      k2 = r8_pl_ld_k(k_b0, ks + 6u);
      k3 = r8_pl_ld_k(k_b0, ks + 7u);
      __builtin_amdgcn_sched_barrier(0);
    }
    sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qr_reg[12u], sa);
    sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qr_reg[13u], sb);
    sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k2, qr_reg[14u], sa);
    sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k3, qr_reg[15u], sb);

    float s[8];
#pragma unroll
    for (u32 v = 0; v < 8u; v++) s[v] = sa[v] + sb[v];
    if (kb + 15u > pos_t0) {
#pragma unroll
      for (u32 v = 0; v < 8u; v++)
        if (kb + 8u * hl + v > tp) s[v] = -__builtin_inff();
    }
    float bm = s[0];
#pragma unroll
    for (u32 v = 1; v < 8u; v++) bm = __builtin_fmaxf(bm, s[v]);
    bm = __builtin_fmaxf(bm, r8_pl_xrow(bm));
    if (__builtin_amdgcn_ballot_w32(!(bm - m < -28.0f)) == 0u) continue;
    const float mn = __builtin_fmaxf(m, bm);
    const float c = __builtin_amdgcn_exp2f(m - mn);
    h8 pb;
    float ps = 0.0f;
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      const float p = __builtin_amdgcn_exp2f(s[v] - mn);
      ps += p;
      pb[v] = (f16)p;
    }
    ls = ls * c + ps;
    m = mn;
    if (__builtin_amdgcn_ballot_w32(c != 1.0f) != 0u) {
#pragma unroll
      for (u32 nt = 0; nt < 16u; nt++)
#pragma unroll
        for (u32 v = 0; v < 8u; v++) o[nt][v] *= c;
    }
    const u32 v_b0 = r8_pl_v_base_b(bf, vt_lane);
    h8 v0 = r8_pl_ld_vt(v_b0, 0u, 0u);
    h8 v1 = r8_pl_ld_vt(v_b0, 1u, 0u);
    h8 v2 = r8_pl_ld_vt(v_b0, 2u, 0u);
    h8 v3 = r8_pl_ld_vt(v_b0, 3u, 0u);
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (u32 g = 0; g < 3u; g++) {
      o[4u * g + 0u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[4u * g + 0u]);
      o[4u * g + 1u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[4u * g + 1u]);
      v0 = r8_pl_ld_vt(v_b0, 0u, g + 1u);
      v1 = r8_pl_ld_vt(v_b0, 1u, g + 1u);
      __builtin_amdgcn_sched_barrier(0);
      o[4u * g + 2u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v2, pb, o[4u * g + 2u]);
      o[4u * g + 3u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v3, pb, o[4u * g + 3u]);
      v2 = r8_pl_ld_vt(v_b0, 2u, g + 1u);
      v3 = r8_pl_ld_vt(v_b0, 3u, g + 1u);
      __builtin_amdgcn_sched_barrier(0);
    }
    o[12u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[12u]);
    o[13u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[13u]);
    o[14u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v2, pb, o[14u]);
    o[15u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v3, pb, o[15u]);
  }
  r8_pl_epilogue(o, ls, t0, rl, hl, T, h, 24u, out, gate, gstride, os, gs, oq, oqs);
}

/* --- R8-2: ie_saluki_swiglu_v4 (8-lane subgroup f32x4 SwiGLU + Q8 quant) --- */
static inline __attribute__((always_inline)) float r8_sub8_max_pos(float v) {
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0xB1, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0x4E, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_ds_swizzle(F2I(v), 0x101F)));
  return v;
}

static inline __attribute__((always_inline)) u32 r8_sub8_sum_u32(u32 s) {
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0xB1, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0x4E, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_ds_swizzle((int)s, 0x101F);
  return s;
}

static inline __attribute__((always_inline)) void r8_swiglu_blk4(G float *a, const G float *b, u32 blk, u32 nb,
                                                                 u32 sl, int skip_f32, G u8 *q, float cl) {
  const u32 off = blk * 32u + sl * 4u;
  f32x4 gv = *(const G f32x4 *)(a + off);
  f32x4 uv = *(const G f32x4 *)(b + off);
  if (cl > 0.0f) {
#pragma unroll
    for (u32 k = 0; k < 4u; k++) {
      if (gv[k] > cl) gv[k] = cl;
      if (uv[k] > cl) uv[k] = cl;
      else if (uv[k] < -cl) uv[k] = -cl;
    }
  }
  f32x4 vv;
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
    vv[k] = gv[k] / (1.0f + __builtin_expf(-gv[k])) * uv[k];
  if (!skip_f32 || !q) *(G f32x4 *)(a + off) = vv;
  if (q) {
    float am = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(vv[0]), __builtin_fabsf(vv[1])),
                               __builtin_fmaxf(__builtin_fabsf(vv[2]), __builtin_fabsf(vv[3])));
    am = r8_sub8_max_pos(am);
    const float d = am / 127.0f;
    int qi[4];
    if (d != 0.0f) {
#pragma unroll
      for (u32 k = 0; k < 4u; k++) qi[k] = (int)__builtin_roundf(vv[k] / d);
    } else {
      qi[0] = qi[1] = qi[2] = qi[3] = 0;
    }
    const u32 pk = ((u32)(u8)qi[0]) | (((u32)(u8)qi[1]) << 8) | (((u32)(u8)qi[2]) << 16) | (((u32)(u8)qi[3]) << 24);
    *(G u32 *)(q + off) = pk;
    const u32 s = r8_sub8_sum_u32((u32)qi[0] + (u32)qi[1] + (u32)qi[2] + (u32)qi[3]);
    if (sl == 0u) {
      ((G float *)(q + 32u * nb))[blk] = d;
      ((G u32 *)(q + 36u * nb))[blk] = s;
    }
  }
}

KERNEL ie_saluki_swiglu_v4(G float *a, const G float *b, u32 n, G u8 *q, u32 as, u32 qs, float cl) {
  a = TOK(a, as);
  b = TOKC(b, as);
  q = TOK(q, qs);
  const int skip_f32 = (n >> 31) != 0u;
  n &= 0x7FFFFFFFu;
  const u32 nb = n >> 5;
  const u32 blk = wgid() * 32u + (tid() >> 3);
  if (blk >= nb) return;
  r8_swiglu_blk4(a, b, blk, nb, lane() & 7u, skip_f32, q, cl);
}

/* --- R8-3: ie_saluki_gdn_seq_wmma_gnorm (prefill GDN chunked recurrence + gnorm) --- */
static inline __attribute__((always_inline)) void r8_gw_split_f4(f4 x, gw_u2 *hi, gw_u2 *lo) {
  const float x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
  const u32 u0 = __builtin_bit_cast(u32, x0), u1 = __builtin_bit_cast(u32, x1);
  const u32 u2 = __builtin_bit_cast(u32, x2), u3 = __builtin_bit_cast(u32, x3);
  const u32 h0r = u0 + 0x7FFFu + ((u0 >> 16) & 1u);
  const u32 h1r = u1 + 0x7FFFu + ((u1 >> 16) & 1u);
  const u32 h2r = u2 + 0x7FFFu + ((u2 >> 16) & 1u);
  const u32 h3r = u3 + 0x7FFFu + ((u3 >> 16) & 1u);
  const float d0 = x0 - __builtin_bit_cast(float, h0r & 0xFFFF0000u);
  const float d1 = x1 - __builtin_bit_cast(float, h1r & 0xFFFF0000u);
  const float d2 = x2 - __builtin_bit_cast(float, h2r & 0xFFFF0000u);
  const float d3 = x3 - __builtin_bit_cast(float, h3r & 0xFFFF0000u);
  const u32 r0 = __builtin_bit_cast(u32, d0), r1 = __builtin_bit_cast(u32, d1);
  const u32 r2 = __builtin_bit_cast(u32, d2), r3 = __builtin_bit_cast(u32, d3);
  const u32 l0r = r0 + 0x7FFFu + ((r0 >> 16) & 1u);
  const u32 l1r = r1 + 0x7FFFu + ((r1 >> 16) & 1u);
  const u32 l2r = r2 + 0x7FFFu + ((r2 >> 16) & 1u);
  const u32 l3r = r3 + 0x7FFFu + ((r3 >> 16) & 1u);
  *hi = (gw_u2){__builtin_amdgcn_perm(h1r, h0r, 0x07060302u), __builtin_amdgcn_perm(h3r, h2r, 0x07060302u)};
  *lo = (gw_u2){__builtin_amdgcn_perm(l1r, l0r, 0x07060302u), __builtin_amdgcn_perm(l3r, l2r, 0x07060302u)};
}

static inline __attribute__((always_inline)) void r8_gw_split_f8(gw_f8 x, gw_bf8 *hi, gw_bf8 *lo) {
  u32x4 hp, lp;
#pragma unroll
  for (u32 i = 0; i < 4u; i++) {
    const float a = x[2u * i], b = x[2u * i + 1u];
    const u32 ua = __builtin_bit_cast(u32, a), ub = __builtin_bit_cast(u32, b);
    const u32 ahr = ua + 0x7FFFu + ((ua >> 16) & 1u);
    const u32 bhr = ub + 0x7FFFu + ((ub >> 16) & 1u);
    hp[i] = __builtin_amdgcn_perm(bhr, ahr, 0x07060302u);
    const u32 ra = __builtin_bit_cast(u32, a - __builtin_bit_cast(float, ahr & 0xFFFF0000u));
    const u32 rb = __builtin_bit_cast(u32, b - __builtin_bit_cast(float, bhr & 0xFFFF0000u));
    const u32 alr = ra + 0x7FFFu + ((ra >> 16) & 1u);
    const u32 blr = rb + 0x7FFFu + ((rb >> 16) & 1u);
    lp[i] = __builtin_amdgcn_perm(blr, alr, 0x07060302u);
  }
  *hi = __builtin_bit_cast(gw_bf8, hp);
  *lo = __builtin_bit_cast(gw_bf8, lp);
}

static inline __attribute__((always_inline)) void r8_gdn_gnorm_row(const LDSP float *orow_lds, const G float *in_z,
                                                                   const float nw_q[4], G float *out_r, G u8 *oq_r,
                                                                   u32 h, u32 l, u32 nb_tot, float eps) {
  float zv[4], ov[4];
#pragma unroll
  for (u32 q = 0; q < 4u; q++) zv[q] = in_z[32u * q + l];
  float s2 = 0.0f;
#pragma unroll
  for (u32 q = 0; q < 4u; q++) {
    ov[q] = orow_lds[32u * q + l];
    s2 += ov[q] * ov[q];
  }
  s2 = wave_sum_all(s2);
  const float r = 1.0f / __builtin_sqrtf(s2 / 128.0f + eps);
  float yv[4];
#pragma unroll
  for (u32 q = 0; q < 4u; q++) {
    yv[q] = ov[q] * r * nw_q[q] * (zv[q] / (1.0f + __builtin_expf(-zv[q])));
    if (out_r) out_r[32u * q + l] = yv[q];
  }
  if (oq_r) {
    float am[4], dv[4];
#pragma unroll
    for (u32 q = 0; q < 4u; q++) am[q] = wave_max_pos(__builtin_fabsf(yv[q]));
#pragma unroll
    for (u32 q = 0; q < 4u; q++) dv[q] = am[q] / 127.0f;
    int qi[4];
#pragma unroll
    for (u32 q = 0; q < 4u; q++) qi[q] = dv[q] != 0.0f ? (int)__builtin_roundf(yv[q] / dv[q]) : 0;
    const u32 b0 = (h * 128u) >> 5;
#pragma unroll
    for (u32 q = 0; q < 4u; q++) ((G i8 *)oq_r)[(b0 + q) * 32u + l] = (i8)qi[q];
    u32 sv[4];
#pragma unroll
    for (u32 q = 0; q < 4u; q++) sv[q] = wave_tree_u((u32)qi[q]);
    if (l == 0u) {
      G float *dp = (G float *)(oq_r + 32u * nb_tot) + b0;
      G u32 *sp = (G u32 *)(oq_r + 36u * nb_tot) + b0;
#pragma unroll
      for (u32 q = 0; q < 4u; q++) {
        dp[q] = dv[q];
        sp[q] = sv[q];
      }
    }
  }
}

typedef union {
  struct {
    __bf16 wh[GC][136];
    __bf16 wl[GC][136];
  } w;
  struct {
    __bf16 kth[128][40];
    __bf16 ktl[128][40];
  } kt;
  float o[GC][132];
} r8_gdn_u1_t;

static LDS r8_gdn_u1_t r8_gdn1_u1 __attribute__((aligned(16)));
static LDS __bf16 r8_gdn1_qh[GC][136] __attribute__((aligned(16)));
static LDS __bf16 r8_gdn1_ql[GC][136] __attribute__((aligned(16)));
static LDS __bf16 r8_gdn1_mh[GC][40] __attribute__((aligned(16)));
static LDS __bf16 r8_gdn1_ml[GC][40] __attribute__((aligned(16)));
static LDS float r8_gdn1_expG[GC] __attribute__((aligned(16)));
static LDS float r8_gdn1_expD[GC] __attribute__((aligned(16)));

KERNEL ie_saluki_gdn_seq_wmma_gnorm(const G float *qk, const G float *U, const G float *W, const G float *Mb,
                                    const G float *Gc, const G float *in, const G float *nw, G float *out, G u8 *oq,
                                    u32 cd, float eps, u32 is, u32 os, u32 qs,
                                    G float *S, u32 c, u32 ns, u32 pos, u32 T, u32 nk, u32 nv) {
  const u32 h = wgid();
  if (h >= nv || T == 0u) return;
  const u32 kh = h % nk, th = tid(), w = __builtin_amdgcn_readfirstlane(th >> 5), l = lane(), hl = l >> 4, rl = l & 15u;
  const u32 col = 16u * w + rl;
  const unsigned long slot = (unsigned long)nv * 16384u;
  const unsigned long wrs = (unsigned long)nv * 128u, qrs = (unsigned long)nk * 256u;
  const G float *Sp = S + (unsigned long)c * slot + (unsigned long)h * 16384u;

  gw_f8 st[8];
  if (pos) {
#pragma unroll
    for (u32 kd = 0; kd < 8u; kd++)
#pragma unroll
      for (u32 e = 0; e < 8u; e++)
        st[kd][e] = Sp[(16u * kd + 8u * hl + e) * 128u + col];
  } else {
    const gw_f8 z8 = (gw_f8)(0.0f);
#pragma unroll
    for (u32 kd = 0; kd < 8u; kd++) st[kd] = z8;
  }

  float nw_q[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (nw) {
#pragma unroll
    for (u32 q = 0; q < 4u; q++) nw_q[q] = nw[32u * q + l];
  }

  const u32 nb_tot = (nv * 128u) >> 5;
  const u32 mr = 4u * w + (l >> 3), mc4 = 4u * (l & 7u), c4 = 4u * l;

  for (u32 b = 0, t0 = 0; t0 < T; b++, t0 += GC) {
    const u32 n = T - t0 < GC ? T - t0 : GC;
    const int full = (n == GC);

    {
      f4 w4[4], q4[4];
      const G float *Wb = W + (unsigned long)(t0 + w) * wrs + h * 128u;
      const G float *Qb = qk + (unsigned long)(t0 + w) * qrs + kh * 256u;
      if (full) {
#pragma unroll
        for (u32 m = 0; m < 4u; m++) {
          w4[m] = *(const G f4 *)(Wb + (unsigned long)(8u * m) * wrs + c4);
          q4[m] = *(const G f4 *)(Qb + (unsigned long)(8u * m) * qrs + c4);
        }
      } else {
#pragma unroll
        for (u32 m = 0; m < 4u; m++) {
          const u32 r = w + 8u * m;
          w4[m] = r < n ? *(const G f4 *)(Wb + (unsigned long)(8u * m) * wrs + c4) : (f4){0.0f, 0.0f, 0.0f, 0.0f};
          q4[m] = r < n ? *(const G f4 *)(Qb + (unsigned long)(8u * m) * qrs + c4) : (f4){0.0f, 0.0f, 0.0f, 0.0f};
        }
      }
      const f4 mv = *(const G f4 *)(Mb + (((unsigned long)b * nv + h) * GC + mr) * GC + mc4);
      if (w == 0u) {
        const float gv = l < n ? Gc[(unsigned long)(t0 + l) * nv + h] : 0.0f;
        const float Gn_w0 = I2F(__builtin_amdgcn_ds_bpermute((int)((n - 1u) << 2), F2I(gv)));
        r8_gdn1_expG[l] = l < n ? __builtin_expf(gv) : 0.0f;
        r8_gdn1_expD[l] = l < n ? __builtin_expf(Gn_w0 - gv) : 0.0f;
      }

#pragma unroll
      for (u32 m = 0; m < 4u; m++) {
        const u32 r = w + 8u * m;
        gw_u2 whi, wlo, qhi, qlo;
        r8_gw_split_f4(w4[m], &whi, &wlo);
        r8_gw_split_f4(q4[m], &qhi, &qlo);
        *(LDSP gw_u2 *)&r8_gdn1_u1.w.wh[r][c4] = whi;
        *(LDSP gw_u2 *)&r8_gdn1_u1.w.wl[r][c4] = wlo;
        *(LDSP gw_u2 *)&r8_gdn1_qh[r][c4] = qhi;
        *(LDSP gw_u2 *)&r8_gdn1_ql[r][c4] = qlo;
      }
      {
        gw_u2 mhi, mlo;
        r8_gw_split_f4(mv, &mhi, &mlo);
        *(LDSP gw_u2 *)&r8_gdn1_mh[mr][mc4] = mhi;
        *(LDSP gw_u2 *)&r8_gdn1_ml[mr][mc4] = mlo;
      }
    }

    gw_f8 u_reg[2];
    {
      const G float *Ub = U + (unsigned long)(t0 + 8u * hl) * wrs + h * 128u + col;
      if (full) {
#pragma unroll
        for (u32 v = 0; v < 8u; v++) u_reg[0][v] = Ub[(unsigned long)v * wrs];
        const G float *Ub1 = Ub + 16ul * wrs;
#pragma unroll
        for (u32 v = 0; v < 8u; v++) u_reg[1][v] = Ub1[(unsigned long)v * wrs];
      } else {
        u32 hl8 = 8u * hl;
        __asm__("" : "+v"(hl8));
#pragma unroll
        for (u32 mt = 0; mt < 2u; mt++)
#pragma unroll
          for (u32 v = 0; v < 8u; v++) {
            const u32 rt = 16u * mt + hl8 + v;
            u_reg[mt][v] = rt < n ? U[(unsigned long)(t0 + rt) * wrs + h * 128u + col] : 0.0f;
          }
      }
    }

    lbarrier();

    gw_f8 ws[2] = {(gw_f8)(0.0f), (gw_f8)(0.0f)};
    gw_f8 qs_acc[2] = {(gw_f8)(0.0f), (gw_f8)(0.0f)};
    if (n > 16u) {
#pragma unroll
      for (u32 kd = 0; kd < 8u; kd++) {
        gw_bf8 sh, sl;
        r8_gw_split_f8(st[kd], &sh, &sl);
        const gw_bf8 wh0 = *(const LDSP gw_bf8 *)&r8_gdn1_u1.w.wh[rl][16u * kd + 8u * hl];
        const gw_bf8 wl0 = *(const LDSP gw_bf8 *)&r8_gdn1_u1.w.wl[rl][16u * kd + 8u * hl];
        const gw_bf8 qh0 = *(const LDSP gw_bf8 *)&r8_gdn1_qh[rl][16u * kd + 8u * hl];
        const gw_bf8 ql0 = *(const LDSP gw_bf8 *)&r8_gdn1_ql[rl][16u * kd + 8u * hl];
        ws[0] = gw_wmma_pair(wh0, wl0, sh, sl, ws[0]);
        qs_acc[0] = gw_wmma_pair(qh0, ql0, sh, sl, qs_acc[0]);
        const gw_bf8 wh1 = *(const LDSP gw_bf8 *)&r8_gdn1_u1.w.wh[16u + rl][16u * kd + 8u * hl];
        const gw_bf8 wl1 = *(const LDSP gw_bf8 *)&r8_gdn1_u1.w.wl[16u + rl][16u * kd + 8u * hl];
        const gw_bf8 qh1 = *(const LDSP gw_bf8 *)&r8_gdn1_qh[16u + rl][16u * kd + 8u * hl];
        const gw_bf8 ql1 = *(const LDSP gw_bf8 *)&r8_gdn1_ql[16u + rl][16u * kd + 8u * hl];
        ws[1] = gw_wmma_pair(wh1, wl1, sh, sl, ws[1]);
        qs_acc[1] = gw_wmma_pair(qh1, ql1, sh, sl, qs_acc[1]);
      }
    } else {
#pragma unroll
      for (u32 kd = 0; kd < 8u; kd++) {
        gw_bf8 sh, sl;
        r8_gw_split_f8(st[kd], &sh, &sl);
        const gw_bf8 wh0 = *(const LDSP gw_bf8 *)&r8_gdn1_u1.w.wh[rl][16u * kd + 8u * hl];
        const gw_bf8 wl0 = *(const LDSP gw_bf8 *)&r8_gdn1_u1.w.wl[rl][16u * kd + 8u * hl];
        const gw_bf8 qh0 = *(const LDSP gw_bf8 *)&r8_gdn1_qh[rl][16u * kd + 8u * hl];
        const gw_bf8 ql0 = *(const LDSP gw_bf8 *)&r8_gdn1_ql[rl][16u * kd + 8u * hl];
        ws[0] = gw_wmma_pair(wh0, wl0, sh, sl, ws[0]);
        qs_acc[0] = gw_wmma_pair(qh0, ql0, sh, sl, qs_acc[0]);
      }
    }

    gw_f8 d_reg[2], o_acc[2];
    gw_bf8 dh[2], dl[2];
    {
      const f32x4 eg0a = *(const LDSP f32x4 *)(r8_gdn1_expG + 8u * hl);
      const f32x4 eg0b = *(const LDSP f32x4 *)(r8_gdn1_expG + 8u * hl + 4u);
      const f32x4 eg1a = *(const LDSP f32x4 *)(r8_gdn1_expG + 16u + 8u * hl);
      const f32x4 eg1b = *(const LDSP f32x4 *)(r8_gdn1_expG + 16u + 8u * hl + 4u);
      const gw_f8 eg0 = {eg0a[0], eg0a[1], eg0a[2], eg0a[3], eg0b[0], eg0b[1], eg0b[2], eg0b[3]};
      const gw_f8 eg1 = {eg1a[0], eg1a[1], eg1a[2], eg1a[3], eg1b[0], eg1b[1], eg1b[2], eg1b[3]};
      if (full) {
        d_reg[0] = u_reg[0] - ws[0];
        d_reg[1] = u_reg[1] - ws[1];
        o_acc[0] = eg0 * qs_acc[0];
        o_acc[1] = eg1 * qs_acc[1];
      } else {
        u32 hl8 = 8u * hl;
        __asm__("" : "+v"(hl8));
#pragma unroll
        for (u32 v = 0; v < 8u; v++) {
          const u32 rt0 = hl8 + v, rt1 = 16u + rt0;
          d_reg[0][v] = rt0 < n ? (u_reg[0][v] - ws[0][v]) : 0.0f;
          o_acc[0][v] = rt0 < n ? (eg0[v] * qs_acc[0][v]) : 0.0f;
          d_reg[1][v] = rt1 < n ? (u_reg[1][v] - ws[1][v]) : 0.0f;
          o_acc[1][v] = rt1 < n ? (eg1[v] * qs_acc[1][v]) : 0.0f;
        }
      }
      r8_gw_split_f8(d_reg[0], &dh[0], &dl[0]);
      r8_gw_split_f8(d_reg[1], &dh[1], &dl[1]);
    }

    {
      const gw_bf8 mh00 = *(const LDSP gw_bf8 *)&r8_gdn1_mh[rl][8u * hl];
      const gw_bf8 ml00 = *(const LDSP gw_bf8 *)&r8_gdn1_ml[rl][8u * hl];
      o_acc[0] = gw_wmma_pair(mh00, ml00, dh[0], dl[0], o_acc[0]);
      if (n > 16u) {
        const gw_bf8 mh10 = *(const LDSP gw_bf8 *)&r8_gdn1_mh[16u + rl][8u * hl];
        const gw_bf8 ml10 = *(const LDSP gw_bf8 *)&r8_gdn1_ml[16u + rl][8u * hl];
        const gw_bf8 mh11 = *(const LDSP gw_bf8 *)&r8_gdn1_mh[16u + rl][16u + 8u * hl];
        const gw_bf8 ml11 = *(const LDSP gw_bf8 *)&r8_gdn1_ml[16u + rl][16u + 8u * hl];
        o_acc[1] = gw_wmma_pair(mh10, ml10, dh[0], dl[0], o_acc[1]);
        o_acc[1] = gw_wmma_pair(mh11, ml11, dh[1], dl[1], o_acc[1]);
      }
    }

    const float eg = r8_gdn1_expG[n - 1u];
    {
      const f32x4 ed0a = *(const LDSP f32x4 *)(r8_gdn1_expD + 8u * hl);
      const f32x4 ed0b = *(const LDSP f32x4 *)(r8_gdn1_expD + 8u * hl + 4u);
      const f32x4 ed1a = *(const LDSP f32x4 *)(r8_gdn1_expD + 16u + 8u * hl);
      const f32x4 ed1b = *(const LDSP f32x4 *)(r8_gdn1_expD + 16u + 8u * hl + 4u);
      const gw_f8 ed0 = {ed0a[0], ed0a[1], ed0a[2], ed0a[3], ed0b[0], ed0b[1], ed0b[2], ed0b[3]};
      const gw_f8 ed1 = {ed1a[0], ed1a[1], ed1a[2], ed1a[3], ed1b[0], ed1b[1], ed1b[2], ed1b[3]};
      gw_f8 ds0, ds1;
      if (full) {
        ds0 = ed0 * d_reg[0];
        ds1 = ed1 * d_reg[1];
      } else {
        u32 hl8 = 8u * hl;
        __asm__("" : "+v"(hl8));
#pragma unroll
        for (u32 v = 0; v < 8u; v++) {
          const u32 rt0 = hl8 + v, rt1 = 16u + rt0;
          ds0[v] = rt0 < n ? (ed0[v] * d_reg[0][v]) : 0.0f;
          ds1[v] = rt1 < n ? (ed1[v] * d_reg[1][v]) : 0.0f;
        }
      }
      r8_gw_split_f8(ds0, &dh[0], &dl[0]);
      r8_gw_split_f8(ds1, &dh[1], &dl[1]);
    }

    lbarrier();

    {
      const u32 r0 = 4u * w;
      const G float *Kb = qk + (unsigned long)(t0 + r0) * qrs + kh * 256u + 128u + l;
      f4 kv[4];
      if (full) {
#pragma unroll
        for (u32 j = 0; j < 4u; j++) {
          const G float *Kc = Kb + 32u * j;
          kv[j] = (f4){Kc[0ul * qrs], Kc[1ul * qrs], Kc[2ul * qrs], Kc[3ul * qrs]};
        }
      } else {
#pragma unroll
        for (u32 j = 0; j < 4u; j++) {
          const G float *Kc = Kb + 32u * j;
          kv[j] = (f4){(r0 + 0u < n) ? Kc[0ul * qrs] : 0.0f,
                       (r0 + 1u < n) ? Kc[1ul * qrs] : 0.0f,
                       (r0 + 2u < n) ? Kc[2ul * qrs] : 0.0f,
                       (r0 + 3u < n) ? Kc[3ul * qrs] : 0.0f};
        }
      }
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
        gw_u2 khi, klo;
        r8_gw_split_f4(kv[j], &khi, &klo);
        *(LDSP gw_u2 *)&r8_gdn1_u1.kt.kth[l + 32u * j][r0] = khi;
        *(LDSP gw_u2 *)&r8_gdn1_u1.kt.ktl[l + 32u * j][r0] = klo;
      }
    }
    lbarrier();

    if (n > 16u) {
#pragma unroll
      for (u32 kd = 0; kd < 8u; kd++) {
        st[kd] *= eg;
        const gw_bf8 kh0 = *(const LDSP gw_bf8 *)&r8_gdn1_u1.kt.kth[16u * kd + rl][8u * hl];
        const gw_bf8 kl0 = *(const LDSP gw_bf8 *)&r8_gdn1_u1.kt.ktl[16u * kd + rl][8u * hl];
        st[kd] = gw_wmma_pair(kh0, kl0, dh[0], dl[0], st[kd]);
        const gw_bf8 kh1 = *(const LDSP gw_bf8 *)&r8_gdn1_u1.kt.kth[16u * kd + rl][16u + 8u * hl];
        const gw_bf8 kl1 = *(const LDSP gw_bf8 *)&r8_gdn1_u1.kt.ktl[16u * kd + rl][16u + 8u * hl];
        st[kd] = gw_wmma_pair(kh1, kl1, dh[1], dl[1], st[kd]);
      }
    } else {
#pragma unroll
      for (u32 kd = 0; kd < 8u; kd++) {
        st[kd] *= eg;
        const gw_bf8 kh0 = *(const LDSP gw_bf8 *)&r8_gdn1_u1.kt.kth[16u * kd + rl][8u * hl];
        const gw_bf8 kl0 = *(const LDSP gw_bf8 *)&r8_gdn1_u1.kt.ktl[16u * kd + rl][8u * hl];
        st[kd] = gw_wmma_pair(kh0, kl0, dh[0], dl[0], st[kd]);
      }
    }

    if (!nw) {
      if (out) {
        u32 hl8 = 8u * hl;
        __asm__("" : "+v"(hl8));
#pragma unroll
        for (u32 mt = 0; mt < 2u; mt++)
#pragma unroll
          for (u32 v = 0; v < 8u; v++) {
            const u32 rt = 16u * mt + hl8 + v;
            if (rt < n) out[(unsigned long)(t0 + rt) * os + h * 128u + col] = o_acc[mt][v];
          }
      }
      lbarrier();
    } else {
      lbarrier();
#pragma unroll
      for (u32 mt = 0; mt < 2u; mt++)
#pragma unroll
        for (u32 v = 0; v < 8u; v++)
          r8_gdn1_u1.o[16u * mt + 8u * hl + v][col] = o_acc[mt][v];
      lbarrier();

      for (u32 rt = w; rt < n; rt += 8u) {
        const G float *in_z = in + (unsigned long)(t0 + rt) * is + cd + h * 128u;
        G float *out_r = out ? (out + (unsigned long)(t0 + rt) * os + h * 128u) : (G float *)0;
        G u8 *oq_r = oq ? (oq + (unsigned long)(t0 + rt) * qs) : (G u8 *)0;
        r8_gdn_gnorm_row(r8_gdn1_u1.o[rt], in_z, nw_q, out_r, oq_r, h, l, nb_tot, eps);
      }
      lbarrier();
    }
  }

  G float *Sw = S + (unsigned long)((c + T - 1u) % ns) * slot + (unsigned long)h * 16384u;
#pragma unroll
  for (u32 kd = 0; kd < 8u; kd++)
#pragma unroll
    for (u32 e = 0; e < 8u; e++)
      Sw[(16u * kd + 8u * hl + e) * 128u + col] = st[kd][e];
}

/* --- R8-4 & R8-5: ie_saluki_gemm_rq128_iq2xxs_sb & ie_saluki_gemm_rq128_q2k_pipe --- */
typedef _Float16 r8_f16;
typedef _Float16 r8_f16x2 __attribute__((ext_vector_type(2)));

static inline __attribute__((always_inline)) u32 r8_bcast16_u(const u32 v, const u32 idx) {
  const u32 sel = 0x11111111u * idx;
  return (u32)__builtin_amdgcn_permlane16(0, (int)v, (int)sel, (int)sel, false, false);
}

static inline __attribute__((always_inline)) float r8_q2k_sub_f16x2(float t, const u32 pair, const float nm0, const float nm1) {
  const r8_f16x2 s = __builtin_bit_cast(r8_f16x2, pair);
  t += (float)s.x * nm0;
  t += (float)s.y * nm1;
  return t;
}

static inline __attribute__((always_inline)) void r8_q2k_store_lsa(
    LDSP u32 *lsa, const u32 ntok, const u32 tk, const u32 c8, const u32x4 rx) {
  const r8_f16 hf = (r8_f16)rqc_sum16i(rx);
  const u32 bits = (u32)__builtin_bit_cast(u16, hf);
  const u32 other = (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ 1u) * 4u), (int)bits);
  if ((c8 & 1u) == 0u) lsa[(c8 >> 1u) * ntok + tk] = bits | (other << 16u);
}

static inline __attribute__((always_inline)) v2i r8_iq2_ksw(const u32 cp, const u32 s7) {
  const u32 p_ = s7 * 0x00810204u;
  const u32 lo = RQ_P2((p_ & 0x04040404u) | (cp & 0x03030303u));
  const u32 hi = RQ_P2(((((u32)__builtin_popcount(s7) << 26u) | (p_ >> 4u)) & 0x04040404u) | ((cp >> 4u) & 0x03030303u));
  const v2i v = {(int)lo, (int)hi};
  return v;
}

static inline __attribute__((always_inline)) void r8_rq_store_out(
    G u8 *y, const G float *bias, const G u8 *res, const u32 rows, const u32 T,
    const u32 ys, const u32 rs, const u32 row, const u32 tbase, const u32 t_wave,
    const u32 wj, const u32 h, const float acc[][8]) {
#pragma unroll
  for (u32 j = 0; j < wj; j++) {
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      const u32 t = tbase + t_wave + 16u * j + 8u * h + v;
      if (t < T && row < rows) {
        const G float *rp = res ? (const G float *)(res + (unsigned long)t * rs) : (const G float *)0;
        ((G float *)(y + (unsigned long)t * ys))[row] = epilogue(acc[j][v], row, bias, rp);
      }
    }
  }
}

#define R8_PIN_ACC8(a) __asm__ volatile("" : "+v"((a)[0]), "+v"((a)[1]), "+v"((a)[2]), "+v"((a)[3]), \
                                             "+v"((a)[4]), "+v"((a)[5]), "+v"((a)[6]), "+v"((a)[7]))

static inline __attribute__((always_inline)) void r8_xxs_math_words(
    const u32 r[9], const u32 qsh, const float d4, const LDSP u32 *lut,
    const LDSP u8 *lx, const LDSP float *lsx, const u32 ntok, const u32 t_wave,
    const u32 wj, const u32 chunk_j, const u32 l, const u32 h, float acc[][8]) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (u32 k = 0; k < RQ_KB; k++) {
    const u32 a0 = rq_align(r[2u * k], r[1u + 2u * k], qsh);
    const u32 a1 = rq_align(r[1u + 2u * k], r[2u + 2u * k], qsh);
    const float s0 = d4 * (0.5f + (float)(a1 >> 28u));
    const u32 g0 = h, g1 = 2u + h;
    const u32 cp0 = lut[(a0 >> (8u * g0)) & 255u];
    const u32 cp1 = lut[(a0 >> (8u * g1)) & 255u];
    const v2i w0 = r8_iq2_ksw(cp0, (a1 >> (7u * g0)) & 127u);
    const v2i w1 = r8_iq2_ksw(cp1, (a1 >> (7u * g1)) & 127u);
#pragma unroll
    for (u32 j0 = 0; j0 < wj; j0 += chunk_j) {
      v2i a0v[4], a1v[4];
      v8i c0[4];
#pragma unroll
      for (u32 j = 0; j < chunk_j; j++) {
        const u32 tk = t_wave + 16u * (j0 + j) + (l & 15u);
        a0v[j] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k) + 8u * h);
        a1v[j] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k + 1u) + 8u * h);
      }
#pragma unroll
      for (u32 j = 0; j < chunk_j; j++) {
        c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0v[j], 1, w0, z, 0);
        c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1v[j], 1, w1, c0[j], 0);
      }
#pragma unroll
      for (u32 j = 0; j < chunk_j; j++) {
        const u32 ts = t_wave + 16u * (j0 + j) + 8u * h;
        const f8 dx = *(const LDSP f8 *)(lsx + k * ntok + ts);
        const f8 t = __builtin_convertvector(c0[j], f8) * s0;
#pragma unroll
        for (u32 v = 0; v < 8u; v++) acc[j0 + j][v] += t[v] * dx[v];
        R8_PIN_ACC8(acc[j0 + j]);
      }
    }
  }
}

static inline __attribute__((always_inline)) void r8_q2k_math_half(
    const rq_hblk_t *cur, const LDSP u8 *lx, const LDSP float *lsx, const LDSP u32 *lsa,
    const u32 ntok, const u32 wj, const u32 chunk_j, const u32 l, const u32 h, float acc[][8]) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  float d = ie_rq_h2f(cur->r[0] & 0xFFFFu);
  float mn = ie_rq_h2f(cur->r[0] >> 16u);
  __asm__ volatile("" : "+v"(d), "+v"(mn));
  const u32 r16 = l & 15u;
#pragma unroll
  for (u32 k = 0; k < RQ_KB; k++) {
    const u32 sw = cur->r[1u + (k >> 1u)] >> (16u * (k & 1u));
    const u32 sc0 = sw & 255u, sc1 = (sw >> 8u) & 255u, ksh = 2u * k;
    v2i w0, w1;
    w0.x = (int)(((cur->r[3] >> ksh) & 0x03030303u) * (sc0 & 15u));
    w0.y = (int)(((cur->r[4] >> ksh) & 0x03030303u) * (sc0 & 15u));
    w1.x = (int)(((cur->r[5] >> ksh) & 0x03030303u) * (sc1 & 15u));
    w1.y = (int)(((cur->r[6] >> ksh) & 0x03030303u) * (sc1 & 15u));
    float nm0 = -(mn * (float)(sc0 >> 4u));
    float nm1 = -(mn * (float)(sc1 >> 4u));
    __asm__ volatile("" : "+v"(nm0), "+v"(nm1));
#pragma unroll
    for (u32 j0 = 0; j0 < wj; j0 += chunk_j) {
      v2i a0[4], a1[4];
      v8i c0[4];
#pragma unroll
      for (u32 j = 0; j < chunk_j; j++) {
        const u32 tk = 16u * (j0 + j) + r16;
        a0[j] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k) + 8u * h);
        a1[j] = *(const LDSP v2i *)((const LDSP u8 *)rqc_lx_cslot(lx, tk, 2u * k + 1u) + 8u * h);
      }
#pragma unroll
      for (u32 j = 0; j < chunk_j; j++) {
        c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0[j], 1, w0, z, 0);
        c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1[j], 1, w1, c0[j], 0);
      }
      const u32 perm_t = 16u * (j0 + (r16 >> 3u)) + 8u * h + (r16 & 7u);
      const u32 sa_p0 = lsa[k * ntok + perm_t];
      const u32 sa_p1 = (chunk_j == 4u) ? lsa[k * ntok + perm_t + 32u] : 0u;
#pragma unroll
      for (u32 j = 0; j < chunk_j; j++) {
        const u32 ts = 16u * (j0 + j) + 8u * h;
        const f8 dx = *(const LDSP f8 *)(lsx + k * ntok + ts);
        f8 t = __builtin_convertvector(c0[j], f8) * d;
        const u32 src = (j < 2u) ? sa_p0 : sa_p1;
        const u32 base = (j & 1u) * 8u;
#pragma unroll
        for (u32 v = 0; v < 8u; v++) {
          t[v] = r8_q2k_sub_f16x2(t[v], r8_bcast16_u(src, base + v), nm0, nm1);
          acc[j0 + j][v] += t[v] * dx[v];
        }
        R8_PIN_ACC8(acc[j0 + j]);
      }
    }
  }
}

static inline __attribute__((always_inline)) void r8_xxs_load_hb0(
    const G u8 *p, u32 *d_word, u32 w[9], u32 *sh_out) {
  *sh_out = (u32)((unsigned long)p & 2ul) << 3u;
  *d_word = *rq_ap(p, 0u);
  const G u32 *qp = rq_ap(p, 2u);
  const u32x4 v0 = *(const G u32x4 *)(qp + 0u);
  const u32x4 v1 = *(const G u32x4 *)(qp + 4u);
  w[0] = v0.x; w[1] = v0.y; w[2] = v0.z; w[3] = v0.w;
  w[4] = v1.x; w[5] = v1.y; w[6] = v1.z; w[7] = v1.w;
  w[8] = qp[8];
}

static inline __attribute__((always_inline)) void r8_xxs_load_hb1_tail(const G u8 *p, u32 tail[8]) {
  const G u32 *qp = rq_ap(p, 2u);
  const u32x4 v0 = *(const G u32x4 *)(qp + 9u);
  const u32x4 v1 = *(const G u32x4 *)(qp + 13u);
  tail[0] = v0.x; tail[1] = v0.y; tail[2] = v0.z; tail[3] = v0.w;
  tail[4] = v1.x; tail[5] = v1.y; tail[6] = v1.z; tail[7] = v1.w;
}

KERNEL ie_saluki_gemm_rq128_iq2xxs_sb(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T,
                                      const G float *bias, const G u8 *res, u32 xs, u32 ys, u32 rs) {
  static LDS u8 lx[128u * 128u] __attribute__((aligned(16)));
  static LDS float lsx[4u * 128u] __attribute__((aligned(32)));
  static LDS u32 lut[256] __attribute__((aligned(16)));
  if (T == 0u || rows == 0u || nb < 8u || (nb & 7u) != 0u || !q || !xq || !y) return;
  const u32 ntt = (T + 127u) / 128u, tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 rbase = rt * 128u, tbase = tt * 128u;
  if (rbase >= rows || tbase >= T) return;
  const u32 t0 = tid(), w = t0 >> 5u, l = lane(), h = l >> 4u;
  const u32 r_loc = 16u * w + (l & 15u), row = rbase + r_loc, rr = row < rows ? row : rows - 1u;
  const u32 mb = nb >> 3u;
  const G u8 *bp = q + (unsigned long)rr * mb * 66u;
  float acc[8][8];
#pragma unroll
  for (u32 j = 0; j < 8u; j++)
#pragma unroll
    for (u32 v = 0; v < 8u; v++) acc[j][v] = 0.0f;
  rq_stage_lut_raw(IE_RQ_IQ2_XXS, (LDSP ie_rq_u64 *)0, lut, 0);

  const G u8 *xp[4];
#pragma unroll
  for (u32 i = 0; i < 4u; i++) {
    const u32 p = t0 + 256u * i, tk = p >> 3u, gt = tbase + tk < T ? tbase + tk : T - 1u;
    xp[i] = xq + (unsigned long)gt * xs + (p & 7u) * 16u;
  }
  const u32 gs = tbase + (t0 & 127u) < T ? tbase + (t0 & 127u) : T - 1u;
  const G float *sxp = (const G float *)(xq + (unsigned long)gs * xs + 32ul * nb);

  u32x4 rx[4];
  f32x4 rsx = (f32x4)(0.0f);
#pragma unroll
  for (u32 i = 0; i < 4u; i++) rx[i] = *(const G u32x4 *)xp[i];
  if (t0 < 128u) rsx = rq_load_f32x4(sxp);
  u32 cur_dw, cur_w[9], cur_sh;
  r8_xxs_load_hb0(bp, &cur_dw, cur_w, &cur_sh);
#pragma unroll
  for (u32 i = 0; i < 4u; i++) {
    const u32 p = t0 + 256u * i, tk = p >> 3u, c8 = p & 7u;
    *rqc_lx_slot(lx, tk, c8) = rx[i];
  }
  if (t0 < 128u) { lsx[t0] = rsx.x; lsx[128u + t0] = rsx.y; lsx[256u + t0] = rsx.z; lsx[384u + t0] = rsx.w; }
  lbarrier();
  __asm__ volatile("" : "+v"(cur_dw), "+v"(cur_w[0]), "+v"(cur_w[4]), "+v"(cur_w[8]));

#pragma unroll 1
  for (u32 sb = 0; sb < mb; sb++) {
    const u32 kb1 = 8u * sb + 4u;
    u32 hb1_t[8];
    r8_xxs_load_hb1_tail(bp, hb1_t);
#pragma unroll
    for (u32 i = 0; i < 4u; i++) rx[i] = *(const G u32x4 *)(xp[i] + kb1 * 32u);
    if (t0 < 128u) rsx = rq_load_f32x4(sxp + kb1);
    __builtin_amdgcn_sched_barrier(0);
    float d4 = ie_rq_h2f((cur_dw >> cur_sh) & 0xFFFFu) * 0.25f;
    __asm__ volatile("" : "+v"(d4));
    const u32 qsh = cur_sh ^ 16u;
    r8_xxs_math_words(cur_w, qsh, d4, lut, lx, lsx, 128u, 0u, 8u, 2u, l, h, acc);
    __builtin_amdgcn_sched_barrier(0);
    lbarrier();
#pragma unroll
    for (u32 i = 0; i < 4u; i++) {
      const u32 p = t0 + 256u * i, tk = p >> 3u, c8 = p & 7u;
      *rqc_lx_slot(lx, tk, c8) = rx[i];
    }
    if (t0 < 128u) { lsx[t0] = rsx.x; lsx[128u + t0] = rsx.y; lsx[256u + t0] = rsx.z; lsx[384u + t0] = rsx.w; }
    lbarrier();
    u32 cur1[9];
    cur1[0] = cur_w[8];
#pragma unroll
    for (u32 i = 0; i < 8u; i++) cur1[1u + i] = hb1_t[i];
    __asm__ volatile("" : "+v"(cur1[1]), "+v"(cur1[5]));

    bp += 66u;
    const int more = sb + 1u < mb;
    u32 nxt_dw = 0u, nxt_w[9], nxt_sh = 0u;
    if (more) {
      r8_xxs_load_hb0(bp, &nxt_dw, nxt_w, &nxt_sh);
      const u32 kb0 = 8u * (sb + 1u);
#pragma unroll
      for (u32 i = 0; i < 4u; i++) rx[i] = *(const G u32x4 *)(xp[i] + kb0 * 32u);
      if (t0 < 128u) rsx = rq_load_f32x4(sxp + kb0);
    }
    __builtin_amdgcn_sched_barrier(0);
    r8_xxs_math_words(cur1, qsh, d4, lut, lx, lsx, 128u, 0u, 8u, 2u, l, h, acc);
    __builtin_amdgcn_sched_barrier(0);
    if (more) {
      lbarrier();
#pragma unroll
      for (u32 i = 0; i < 4u; i++) {
        const u32 p = t0 + 256u * i, tk = p >> 3u, c8 = p & 7u;
        *rqc_lx_slot(lx, tk, c8) = rx[i];
      }
      if (t0 < 128u) { lsx[t0] = rsx.x; lsx[128u + t0] = rsx.y; lsx[256u + t0] = rsx.z; lsx[384u + t0] = rsx.w; }
      lbarrier();
      cur_dw = nxt_dw; cur_sh = nxt_sh;
#pragma unroll
      for (u32 i = 0; i < 9u; i++) cur_w[i] = nxt_w[i];
      __asm__ volatile("" : "+v"(cur_dw), "+v"(cur_w[0]), "+v"(cur_w[4]), "+v"(cur_w[8]));
    }
  }
  r8_rq_store_out(y, bias, res, rows, T, ys, rs, row, tbase, 0u, 8u, h, acc);
}

KERNEL ie_saluki_gemm_rq128_q2k_pipe(const G u8 *q, const G u8 *xq, G u8 *y, u32 rows, u32 nb, u32 T,
                                     const G float *bias, const G u8 *res, u32 xs, u32 ys, u32 rs) {
  static LDS u8 lx[128u * 128u] __attribute__((aligned(16)));
  static LDS float lsx[4u * 128u] __attribute__((aligned(32)));
  static LDS u32 lsa[4u * 128u] __attribute__((aligned(16)));
  if (T == 0u || rows == 0u || nb < 8u || (nb & 7u) != 0u || !q || !xq || !y) return;
  const u32 ntt = (T + 127u) / 128u, tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 rbase = rt * 128u, tbase = tt * 128u;
  if (rbase >= rows || tbase >= T) return;
  const u32 t0 = tid(), w = t0 >> 5u, l = lane(), h = l >> 4u;
  const u32 r_loc = 16u * w + (l & 15u), row = rbase + r_loc, rr = row < rows ? row : rows - 1u;
  const u32 mb = nb >> 3u, bs = ie_rq_bs(IE_RQ_Q2_K);
  const G u8 *rowp = q + (unsigned long)rr * mb * bs;
  float acc[8][8];
#pragma unroll
  for (u32 j = 0; j < 8u; j++)
#pragma unroll
    for (u32 v = 0; v < 8u; v++) acc[j][v] = 0.0f;

  const G u8 *xp[4];
#pragma unroll
  for (u32 i = 0; i < 4u; i++) {
    const u32 p = t0 + 256u * i, tk = p >> 3u, gt = tbase + tk < T ? tbase + tk : T - 1u;
    xp[i] = xq + (unsigned long)gt * xs + (p & 7u) * 16u;
  }
  const u32 gs = tbase + (t0 & 127u) < T ? tbase + (t0 & 127u) : T - 1u;
  const G float *sxp = (const G float *)(xq + (unsigned long)gs * xs + 32ul * nb);
  u32x4 rx[4];
  f32x4 rsx = (f32x4)(0.0f);
#pragma unroll
  for (u32 i = 0; i < 4u; i++) rx[i] = *(const G u32x4 *)xp[i];
  if (t0 < 128u) rsx = rq_load_f32x4(sxp);
  rq_hblk_t cur = rq_hload(IE_RQ_Q2_K, rowp, 0u, h);
#pragma unroll
  for (u32 i = 0; i < 4u; i++) {
    const u32 p = t0 + 256u * i, tk = p >> 3u, c8 = p & 7u;
    *rqc_lx_slot(lx, tk, c8) = rx[i];
    r8_q2k_store_lsa(lsa, 128u, tk, c8, rx[i]);
  }
  if (t0 < 128u) { lsx[t0] = rsx.x; lsx[128u + t0] = rsx.y; lsx[256u + t0] = rsx.z; lsx[384u + t0] = rsx.w; }
  lbarrier();
  __asm__ volatile("" : "+v"(cur.r[0]), "+v"(cur.r[1]), "+v"(cur.r[2]), "+v"(cur.r[3]), "+v"(cur.r[4]), "+v"(cur.r[5]), "+v"(cur.r[6]));

  const u32 nc = 2u * mb;
#pragma unroll 1
  for (u32 c = 0; c < nc; c++) {
    const int more = c + 1u < nc;
    rq_hblk_t nxt;
    if (more) {
      const u32 cn = c + 1u;
      nxt = rq_hload(IE_RQ_Q2_K, rowp + (unsigned long)(cn >> 1u) * bs, cn & 1u, h);
      const u32 kb = 4u * cn;
#pragma unroll
      for (u32 i = 0; i < 4u; i++) rx[i] = *(const G u32x4 *)(xp[i] + kb * 32u);
      if (t0 < 128u) rsx = rq_load_f32x4(sxp + kb);
    }
    __builtin_amdgcn_sched_barrier(0);
    r8_q2k_math_half(&cur, lx, lsx, lsa, 128u, 8u, 2u, l, h, acc);
    __builtin_amdgcn_sched_barrier(0);
    if (more) {
      lbarrier();
#pragma unroll
      for (u32 i = 0; i < 4u; i++) {
        const u32 p = t0 + 256u * i, tk = p >> 3u, c8 = p & 7u;
        *rqc_lx_slot(lx, tk, c8) = rx[i];
        r8_q2k_store_lsa(lsa, 128u, tk, c8, rx[i]);
      }
      if (t0 < 128u) { lsx[t0] = rsx.x; lsx[128u + t0] = rsx.y; lsx[256u + t0] = rsx.z; lsx[384u + t0] = rsx.w; }
      lbarrier();
      cur = nxt;
      __asm__ volatile("" : "+v"(cur.r[0]), "+v"(cur.r[1]), "+v"(cur.r[2]), "+v"(cur.r[3]), "+v"(cur.r[4]), "+v"(cur.r[5]), "+v"(cur.r[6]));
    }
  }
  r8_rq_store_out(y, bias, res, rows, T, ys, rs, row, tbase, 0u, 8u, h, acc);
}

/* --- R8-6: ie_saluki_iq2xxs_tr9_pipe1 (T=8 verify IQ2_XXS, nb=160 shapes) --- */
static inline __attribute__((always_inline)) u32 r8_pack_iq2_code(ie_rq_u64 g) {
  const ie_rq_u64 c = TR8_G2C(g);
  return (u32)c | ((u32)(c >> 32u) << 4u);
}

static inline __attribute__((always_inline)) void r8_tr9_dot8_sload(
    const u32 wl[4], const u32 wh[4], const float dw,
    const G u8 *xq, const u32 xs, const u32 xo, const u32 dao, float aj[8]) {
#pragma clang fp contract(off)
#pragma unroll
  for (u32 t = 0; t < 8u; t++) {
    const TR8_CSP u8 *xb = (const TR8_CSP u8 *)tr8_uniform(xq + (unsigned long)t * xs);
    const float dav = *(const TR8_CSP float *)(xb + dao);
    const u32x4 x0 = *(const TR8_CSP u32x4 *)(xb + xo);
    const u32x4 x1 = *(const TR8_CSP u32x4 *)(xb + xo + 16u);
    u32 idot = ie_dot4_ss(0u, wl[0], x0.x);
    idot = ie_dot4_ss(idot, wh[0], x0.y);
    idot = ie_dot4_ss(idot, wl[1], x0.z);
    idot = ie_dot4_ss(idot, wh[1], x0.w);
    idot = ie_dot4_ss(idot, wl[2], x1.x);
    idot = ie_dot4_ss(idot, wh[2], x1.y);
    idot = ie_dot4_ss(idot, wl[3], x1.z);
    idot = ie_dot4_ss(idot, wh[3], x1.w);
    aj[t] += dav * (dw * (float)(int)idot);
    if ((t & 3u) == 3u) __asm__ volatile("" : "+v"(aj[t - 3u]), "+v"(aj[t - 2u]), "+v"(aj[t - 1u]), "+v"(aj[t]));
  }
}

static inline __attribute__((always_inline)) void r8_tr9_decode_xxs(
    const LDSP u32 *ulut, const u32 d16, const u32 a0, const u32 a1,
    u32 wl[4], u32 wh[4], float *dw_out) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d16);
  *dw_out = (d * 0.25f) * (0.5f + (float)(a1 >> 28u));
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 cp = ulut[ie_byte(a0, g)];
    TR8_KSW(cp & 0x0F0F0F0Fu, (cp >> 4u) & 0x0F0F0F0Fu, (a1 >> (7u * g)) & 127u, wl[g], wh[g]);
  }
}

static inline __attribute__((always_inline)) void r8_tr9_reduce32(
    LDSP float *red, const u32 w, const u32 l, const u32 r0, const u32 rows,
    float a[4][8], const G float *bias, const G float *res, G float *y, const u32 ys, const u32 rs) {
#pragma clang fp contract(off)
#pragma unroll
  for (u32 t = 0; t < 8u; t++) red[(w * 8u + t) * 32u + l] = (a[0][t] + a[2][t]) + (a[1][t] + a[3][t]);
  lbarrier();
  const u32 i = tid();
  const u32 rr = i & 31u, t = i >> 5u, row = r0 + rr;
  float v[8];
#pragma unroll
  for (u32 x = 0; x < 8u; x++) v[x] = red[(x * 8u + t) * 32u + rr];
  const float x0 = v[0] + v[4], x1 = v[1] + v[5], x2 = v[2] + v[6], x3 = v[3] + v[7];
  float out = (x0 + x2) + (x1 + x3);
  if (row < rows) {
    if (bias) out = out + bias[row];
    if (res) out = ((const G float *)((const G u8 *)res + (unsigned long)t * rs))[row] + out;
    ((G float *)((G u8 *)y + (unsigned long)t * ys))[row] = out;
  }
}

KERNEL ie_saluki_iq2xxs_tr9_pipe1(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                                  const G float *bias, const G float *res, const G float *nw, G float *ny,
                                  G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  const u32 r0 = 32u * wgid();
  if (T != 8u || r0 >= rows) return;
  static LDS u32 ulut[256];
  static LDS float red[8u * 8u * 32u];
  ulut[tid()] = r8_pack_iq2_code(ie_rq_iq2xxs_grid[tid()]);
  lbarrier();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 r = r0 + l;
  const u32 mb = nb >> 3u;
  const G u8 *rowp = q + (unsigned long)(r < rows ? r : r0) * mb * 66u;
  float a[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[j][t] = 0.0f;

  const u32 o = 2u + 8u * w;
  const u32 nb32 = nb & ~31u;
  if (nb32 > 0u) {
    const G u8 *bp = rowp;
    u32 d_c = ie_rq_a16(bp, 0u);
    u32 a0_c = ie_rq_a32(bp, o), a1_c = ie_rq_a32(bp, o + 4u);
    __asm__ volatile("" : "+v"(d_c), "+v"(a0_c), "+v"(a1_c));
    for (u32 k0 = 0; k0 < nb32; k0 += 32u) {
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
        const u32 b = k0 + w + 8u * j;
        const int has_nxt = (j < 3u) || (k0 + 32u < nb32);
        bp += 66u;
        u32 d_n = 0u, a0_n = 0u, a1_n = 0u;
        if (has_nxt) {
          d_n = ie_rq_a16(bp, 0u);
          a0_n = ie_rq_a32(bp, o);
          a1_n = ie_rq_a32(bp, o + 4u);
        }
        __builtin_amdgcn_sched_barrier(0);
        u32 wl[4], wh[4]; float dw;
        r8_tr9_decode_xxs(ulut, d_c, a0_c, a1_c, wl, wh, &dw);
        r8_tr9_dot8_sload(wl, wh, dw, xq, xs, 32u * b, 32u * nb + 4u * b, a[j]);
        __builtin_amdgcn_sched_barrier(0);
        d_c = d_n; a0_c = a0_n; a1_c = a1_n;
      }
    }
  }
  for (u32 j = 0; j < 4u; j++) {
    const u32 b = nb32 + w + 8u * j;
    if (b < nb) {
      const G u8 *blk = rowp + (unsigned long)(b >> 3u) * 66u;
      u32 wl[4], wh[4]; float dw;
      r8_tr9_decode_xxs(ulut, ie_rq_a16(blk, 0u), ie_rq_a32(blk, o), ie_rq_a32(blk, o + 4u), wl, wh, &dw);
      r8_tr9_dot8_sload(wl, wh, dw, xq, xs, 32u * b, 32u * nb + 4u * b, a[j]);
    }
  }
  r8_tr9_reduce32(red, w, l, r0, rows, a, bias, res, y, ys, rs);
}

/* --- R8-7 & R8-8: ie_saluki_attn_long_mp_pipe1 & ie_saluki_attn_long_mp_tr_nospill --- */
#define R8_TR_DEPTH(d03, d47, j) ((((j) < 4u ? (d03) : (d47)) >> (8u * ((j) & 3u))) & 255u)

static inline __attribute__((always_inline)) v8f r8_q8_kqt_klds(
    const u32x4 kq[8], const LDSP float *ksc_lds, const LDSP f16 *qh, const LDSP f16 *ql, u32 hl, u32 rl) {
  const v8f zero = (v8f)(0.0f);
  float sv[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (u32 blk = 0; blk < 8u; blk++) {
    const u32 qoff = rl * AL_QS + 32u * blk + 16u * hl;
    const h8 qh0 = *(const LDSP h8 *)(qh + qoff);
    const h8 ql0 = *(const LDSP h8 *)(ql + qoff);
    const h8 qh1 = *(const LDSP h8 *)(qh + qoff + 8u);
    const h8 ql1 = *(const LDSP h8 *)(ql + qoff + 8u);
    const h8 kb0 = al_i8_to_h8(kq[blk][0], kq[blk][1]);
    const h8 kb1 = al_i8_to_h8(kq[blk][2], kq[blk][3]);
    v8f t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb0, qh0, zero);
    t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb0, ql0, t);
    t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb1, qh1, t);
    t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb1, ql1, t);
    const v8f ks = *(const LDSP v8f *)(ksc_lds + blk * 16u + 8u * hl);
#pragma unroll
    for (u32 v = 0; v < 8u; v++) sv[v] = __builtin_fmaf(t[v], ks[v], sv[v]);
    R8_PIN_ACC8(sv);
  }
  const v8f out = {sv[0], sv[1], sv[2], sv[3], sv[4], sv[5], sv[6], sv[7]};
  return out;
}

static inline __attribute__((always_inline)) v8f r8_q8_kqt_stream(
    const G kvt *kc, unsigned long row, u32 n_kv, u32 kh, const LDSP f16 *qh, const LDSP f16 *ql, u32 hl, u32 rl) {
  const G u8 *rec = (const G u8 *)kc + (row * n_kv + kh) * 272ul;
  const u32x4 ksc4 = hl == 0u ? *(const G u32x4 *)(rec + 256ul) : (u32x4)(0u);
  const v8f zero = (v8f)(0.0f);
  v8f s = zero;
#pragma nounroll
  for (u32 blk = 0; blk < 8u; blk++) {
    const u32x4 kb = *(const G u32x4 *)(rec + 32ul * blk + 16ul * hl);
    const u32 qoff = rl * AL_QS + 32u * blk + 16u * hl;
    const h8 qh0 = *(const LDSP h8 *)(qh + qoff);
    const h8 ql0 = *(const LDSP h8 *)(ql + qoff);
    const h8 qh1 = *(const LDSP h8 *)(qh + qoff + 8u);
    const h8 ql1 = *(const LDSP h8 *)(ql + qoff + 8u);
    const h8 kb0 = al_i8_to_h8(kb[0], kb[1]);
    const h8 kb1 = al_i8_to_h8(kb[2], kb[3]);
    v8f t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb0, qh0, zero);
    t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb0, ql0, t);
    t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb1, qh1, t);
    t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb1, ql1, t);
    const u32 kw = ksc4[blk >> 1u];
    const float my_ks = (float)__builtin_bit_cast(f16, (u16)((blk & 1u) ? (kw >> 16u) : kw));
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      const float s0 = I2F(__builtin_amdgcn_readlane(F2I(my_ks), (int)v));
      const float s1 = I2F(__builtin_amdgcn_readlane(F2I(my_ks), (int)(8u + v)));
      s[v] = __builtin_fmaf(t[v], hl ? s1 : s0, s[v]);
    }
  }
  return s;
}

static inline __attribute__((always_inline)) void r8_attn_mp_body(
    const int is_tree,
    const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos,
    G u32 *count, G u32 *gcount, u32 gmax, G u8 *oq, const G float *gate, u32 gstride,
    u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride, const G u8 *tr) {
  const u32 n_head = 24u, n_kv = 4u, T = 8u, npg = 3u, grp = 6u, npair = 48u;
  const u32 d03 = is_tree ? __builtin_amdgcn_readfirstlane(*(const G u32 *)(tr + TR_DEP)) : 0u;
  const u32 d47 = is_tree ? __builtin_amdgcn_readfirstlane(*(const G u32 *)(tr + TR_DEP + 4u)) : 0u;
  const u32 kh = __builtin_amdgcn_readfirstlane(wgid() & 3u);
  const u32 sg = __builtin_amdgcn_readfirstlane(wgid() >> 2u);
  const u32 w = tid() >> 5, l = lane(), hl = l >> 4, rl = l & 15u;
  const u32 kend = __builtin_amdgcn_readfirstlane(pos + (is_tree ? (u32)tr[368] : 8u));
  const u32 c0 = __builtin_amdgcn_readfirstlane(sg * AL_CH);
  if (c0 >= kend) return;
  const int is_tail = is_tree ? __builtin_amdgcn_readfirstlane(c0 + AL_CH > pos) : 0;
  const v8f zero = (v8f)(0.0f);
  const unsigned long pst = pstride;
  const u32 kb = c0 + 16u * w;
  const int kon = kb < kend;

  v8f s0 = zero, s1 = zero, s2 = zero;
  if (__builtin_expect(!is_tail, 1)) {
    u32x4 kq[8];
    LDSP float *ksc_lds = (LDSP float *)(mp_lds + MP_OFF_SC) + w * 128u;
    if (kon) {
      const unsigned long row = kb + rl < kend ? kb + rl : kend - 1u;
      u32x4 ksc4 = (u32x4)(0u);
      al_q8_kload(kc, row, n_kv, kh, hl, kq, &ksc4);
      if (hl == 0u) {
#pragma unroll
        for (u32 blk = 0; blk < 8u; blk++) {
          const u32 kw = ksc4[blk >> 1u];
          ksc_lds[blk * 16u + rl] = (float)__builtin_bit_cast(f16, (u16)((blk & 1u) ? (kw >> 16u) : kw));
        }
      }
    }
#pragma unroll
    for (u32 pg = 0; pg < 3u; pg++) {
      mp_load_q_lds(q, pg, npair, grp, kh, qst);
      lbarrier();
      if (kon) {
        const LDSP f16 *qh = (const LDSP f16 *)mp_lds, *ql = qh + 16u * AL_QS;
        const v8f sv = r8_q8_kqt_klds(kq, ksc_lds, qh, ql, hl, rl);
        if (pg == 0u) s0 = sv;
        else if (pg == 1u) s1 = sv;
        else s2 = sv;
      }
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
      lbarrier();
    }
  } else {
#pragma nounroll
    for (u32 pg = 0; pg < 3u; pg++) {
      mp_load_q_lds(q, pg, npair, grp, kh, qst);
      lbarrier();
      if (kon) {
        const LDSP f16 *qh = (const LDSP f16 *)mp_lds, *ql = qh + 16u * AL_QS;
        const u32 ci = 16u * pg + rl, cj = ci / grp;
        v8f stmp = zero;
        if (kb + 15u >= pos) {
          const u32 jlo = 16u * pg / grp, jhi = (16u * pg + 15u) / grp;
#pragma nounroll
          for (u32 jj = jlo; jj <= jhi; jj++) {
            const unsigned long trow = al_q8_trow(tr, jj, kb + rl < kend ? kb + rl : kend - 1u, pos);
            const v8f sj = r8_q8_kqt_stream(kc, trow, n_kv, kh, qh, ql, hl, rl);
            if (cj == jj) stmp = sj;
          }
        } else {
          const unsigned long row = kb + rl < kend ? kb + rl : kend - 1u;
          stmp = r8_q8_kqt_stream(kc, row, n_kv, kh, qh, ql, hl, rl);
        }
        if (pg == 0u) s0 = stmp;
        else if (pg == 1u) s1 = stmp;
        else s2 = stmp;
      }
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
      lbarrier();
    }
  }

  u32x4 vr0 = (u32x4)(0u);
  f16 vsc0 = (f16)0.0f;
  if (!is_tail && c0 < kend) {
    const unsigned long row0 = c0 + rl < kend ? c0 + rl : kend - 1u;
    const G u8 *rec0 = (const G u8 *)vc + (row0 * n_kv + kh) * 272ul;
    vr0 = *(const G u32x4 *)(rec0 + 32ul * w + 16ul * hl);
    vsc0 = ((const G f16 *)(rec0 + 256ul))[w];
  }

#pragma unroll
  for (u32 pg = 0; pg < 3u; pg++) {
    const v8f spg = (pg == 0u) ? s0 : ((pg == 1u) ? s1 : s2);
    const u32 ci = 16u * pg + rl, cj = ci / grp;
    const u32 cpj = pos + (is_tree ? R8_TR_DEPTH(d03, d47, cj) : cj);
    float sv[8], mx = -__builtin_inff();
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      sv[v] = kb + 8u * hl + v <= cpj ? spg[v] : -__builtin_inff();
      mx = __builtin_fmaxf(mx, sv[v]);
    }
    mx = __builtin_fmaxf(mx, I2F(xrow_i(F2I(mx))));
    if (hl == 0u) al_red[0][16u * w + rl] = mx;
    lbarrier();
    float M = al_red[0][rl];
#pragma unroll
    for (u32 x = 1; x < 8u; x++) M = __builtin_fmaxf(M, al_red[0][16u * x + rl]);
    h8 ph8, pl8; float ls = 0.0f;
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      const float p = kb + 8u * hl + v <= cpj ? __builtin_expf(sv[v] - M) : 0.0f;
      ls += p; const f16 hv = (f16)p; ph8[v] = hv; pl8[v] = (f16)(p - (float)hv);
    }
    ls += I2F(xrow_i(F2I(ls)));
    if (hl == 0u) al_red[1][16u * w + rl] = ls;
    LDSP f16 *pp = (LDSP f16 *)mp_lds + pg * MP_PG_HALVES;
    *(LDSP h8 *)(pp + rl * AL_PS + 16u * w + 8u * hl) = ph8;
    *(LDSP h8 *)(pp + 16u * AL_PS + rl * AL_PS + 16u * w + 8u * hl) = pl8;
    lbarrier();
    float L = al_red[1][rl];
#pragma unroll
    for (u32 x = 1; x < 8u; x++) L += al_red[1][16u * x + rl];
    if (w == 0u && hl == 0u && c0 <= cpj) {
      const u32 h = kh * grp + ci % grp, ns = al_ns(cpj);
      G float *pp_ml = part + (unsigned long)cj * pst + ((unsigned long)h * (ns + al_ng(ns)) + sg) * AL_PSZ;
      pp_ml[0] = M; pp_ml[1] = L;
    }
  }

  v8f acc[3][2] = {{zero, zero}, {zero, zero}, {zero, zero}};
  {
    LDSP f16 *sc = (LDSP f16 *)(mp_lds + MP_OFF_SC) + w * (16u * AL_SS);
    if (__builtin_expect(!is_tail, 1)) {
      __asm__ volatile("" : "+v"(vr0), "+v"(vsc0));
#pragma nounroll
      for (u32 kt = 0; kt < 8u; kt++) {
        if (c0 + 16u * kt >= kend) break;
        u32x4 vr_nxt = (u32x4)(0u); f16 vsc_nxt = (f16)0.0f;
        if (kt + 1u < 8u && c0 + 16u * (kt + 1u) < kend) {
          const unsigned long nrow = c0 + 16u * (kt + 1u) + rl < kend ? c0 + 16u * (kt + 1u) + rl : kend - 1u;
          const G u8 *nrec = (const G u8 *)vc + (nrow * n_kv + kh) * 272ul;
          vr_nxt = *(const G u32x4 *)(nrec + 32ul * w + 16ul * hl);
          vsc_nxt = ((const G f16 *)(nrec + 256ul))[w];
        }
        __builtin_amdgcn_sched_barrier(0);
        h8 b0, b1; v8f vs; float w_sc;
        mp_q8_vprep(vr0, vsc0, sc, w, hl, rl, &b0, &b1, &vs, &w_sc);
#pragma unroll
        for (u32 pg = 0; pg < 3u; pg++)
          mp_q8_pv_accum(b0, b1, vs, w_sc, (const LDSP f16 *)mp_lds + pg * MP_PG_HALVES, kt, hl, rl, 0xffu,
                         &acc[pg][0], &acc[pg][1]);
        __builtin_amdgcn_sched_barrier(0);
        vr0 = vr_nxt; vsc0 = vsc_nxt;
        __asm__ volatile("" : "+v"(vr0), "+v"(vsc0));
      }
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
    } else {
#pragma nounroll
      for (u32 kt = 0; kt < 8u; kt++) {
        if (c0 + 16u * kt >= kend) break;
        if (c0 + 16u * kt + 15u >= pos) {
#pragma nounroll
          for (u32 jj = 0; jj < T; jj++) {
            const unsigned long key = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;
            const unsigned long trow = al_q8_trow(tr, jj, (u32)key, pos);
            const G u8 *trec = (const G u8 *)vc + (trow * n_kv + kh) * 272ul;
            const u32x4 vrj = *(const G u32x4 *)(trec + 32ul * w + 16ul * hl);
            const f16 vsj = ((const G f16 *)(trec + 256ul))[w];
            h8 b0, b1; v8f vs; float w_sc;
            mp_q8_vprep(vrj, vsj, sc, w, hl, rl, &b0, &b1, &vs, &w_sc);
#pragma unroll
            for (u32 pg = 0; pg < 3u; pg++) {
              if (16u * pg / grp <= jj && jj <= (16u * pg + 15u) / grp) {
                u32 vmask = 0u;
#pragma unroll
                for (u32 v = 0; v < 8u; v++)
                  if ((16u * pg + 8u * hl + v) / grp == jj) vmask |= 1u << v;
                mp_q8_pv_accum(b0, b1, vs, w_sc, (const LDSP f16 *)mp_lds + pg * MP_PG_HALVES, kt, hl, rl, vmask,
                               &acc[pg][0], &acc[pg][1]);
              }
            }
            __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
          }
        } else {
          const unsigned long nrow = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;
          const G u8 *nrec = (const G u8 *)vc + (nrow * n_kv + kh) * 272ul;
          const u32x4 vrc = *(const G u32x4 *)(nrec + 32ul * w + 16ul * hl);
          const f16 vscc = ((const G f16 *)(nrec + 256ul))[w];
          h8 b0, b1; v8f vs; float w_sc;
          mp_q8_vprep(vrc, vscc, sc, w, hl, rl, &b0, &b1, &vs, &w_sc);
#pragma unroll
          for (u32 pg = 0; pg < 3u; pg++)
            mp_q8_pv_accum(b0, b1, vs, w_sc, (const LDSP f16 *)mp_lds + pg * MP_PG_HALVES, kt, hl, rl, 0xffu,
                           &acc[pg][0], &acc[pg][1]);
          __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        }
      }
    }
  }

  const u32 ns_min = al_ns(pos);
  const u32 ns_max = al_ns(kend - 1u);
  if (__builtin_expect(c0 <= pos && ns_min == ns_max, 1)) {
    const unsigned long h_stride = (unsigned long)(ns_min + al_ng(ns_min)) * AL_PSZ;
    const unsigned long sg_base = (unsigned long)sg * AL_PSZ + 4u + 32ul * w + rl;
    const unsigned long kh_off = (unsigned long)(kh * grp) * h_stride;
#pragma unroll
    for (u32 pg = 0; pg < 3u; pg++) {
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const u32 pi0 = 16u * pg + v, pi1 = 16u * pg + 8u + v;
        const unsigned long off0 = (unsigned long)(pi0 / grp) * pst + (unsigned long)(pi0 % grp) * h_stride;
        const unsigned long off1 = (unsigned long)(pi1 / grp) * pst + (unsigned long)(pi1 % grp) * h_stride;
        G float *pp = part + kh_off + (hl ? off1 : off0) + sg_base;
        pp[0] = acc[pg][0][v];
        pp[16] = acc[pg][1][v];
      }
    }
  } else {
#pragma unroll
    for (u32 pg = 0; pg < 3u; pg++) {
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const u32 pi = 16u * pg + 8u * hl + v, j = pi / grp, h = kh * grp + pi % grp;
        const u32 cpj = pos + (is_tree ? R8_TR_DEPTH(d03, d47, j) : j);
        if (c0 <= cpj) {
          const u32 ns = al_ns(cpj);
          G float *pp = part + (unsigned long)j * pst + ((unsigned long)h * (ns + al_ng(ns)) + sg) * AL_PSZ + 4u;
          pp[32u * w + rl] = acc[pg][0][v]; pp[32u * w + 16u + rl] = acc[pg][1][v];
        }
      }
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (w < npg && hl == 0u) {
    const u32 ci = 16u * w + rl, cj = ci / grp;
    const u32 cpj = pos + (is_tree ? R8_TR_DEPTH(d03, d47, cj) : cj);
    u32 last = 0u;
    if (c0 <= cpj) {
      const u32 ns = al_ns(cpj), g = sg / AL_G, in_g = ns - g * AL_G < AL_G ? ns - g * AL_G : AL_G;
      const unsigned long gi = ((unsigned long)cj * n_head + kh * grp + ci % grp) * gmax + g;
      last = __scoped_atomic_fetch_add(&gcount[gi], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == in_g;
    }
    mp_last[ci] = last;
  }
  barrier();
  for (u32 pi = w; pi < 16u * npg; pi += 8u) {
    if (!mp_last[pi]) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const u32 j = pi / grp, h = kh * grp + pi % grp;
    const u32 ns = al_ns(pos + (is_tree ? R8_TR_DEPTH(d03, d47, j) : j)), ng = al_ng(ns), g = sg / AL_G;
    const u32 in_g = ns - g * AL_G < AL_G ? ns - g * AL_G : AL_G;
    G float *pb = part + (unsigned long)j * pst + (unsigned long)h * (ns + ng) * AL_PSZ;
    G float *ox = out ? (G float *)((G u8 *)out + (unsigned long)j * ost) + (unsigned long)h * 256ul : (G float *)0;
    G u8 *oqx = oq ? oq + (unsigned long)j * oqst : oq;
    const G float *gxh = gate ? (const G float *)((const G u8 *)gate + (unsigned long)j * gst) + (unsigned long)h * gstride : gate;
    float M2, L2; f32x4 a0, a1;
    al_msum(pb + (unsigned long)g * AL_G * AL_PSZ, in_g, &M2, &L2, &a0, &a1);
    if (l == 0u) gcount[((unsigned long)j * n_head + h) * gmax + g] = 0u;
    if (ng == 1u) { al_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_head * 8u, (LDSP float *)mp_lds + w * 256u); continue; }
    G float *gp = pb + (unsigned long)(ns + g) * AL_PSZ;
    if (l == 0u) gp[0] = M2, gp[1] = L2;
    *(G f32x4 *)(gp + 4u + 4u * l) = a0; *(G f32x4 *)(gp + 132u + 4u * l) = a1;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    u32 done = 0u;
    if (l == 0u) done = __scoped_atomic_fetch_add(&count[j * n_head + h], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == ng;
    if (!__builtin_amdgcn_readfirstlane(done)) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    al_msum(pb + (unsigned long)ns * AL_PSZ, ng, &M2, &L2, &a0, &a1);
    if (l == 0u) count[j * n_head + h] = 0u;
    al_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_head * 8u, (LDSP float *)mp_lds + w * 256u);
  }
}

KERNEL ie_saluki_attn_long_mp_pipe1(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos,
                                    u32 n_head, u32 n_kv, u32 T, u32 npg, G u32 *count, G u32 *gcount, u32 gmax, G u8 *oq,
                                    const G float *gate, u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride,
                                    const G u8 *tr) {
  (void)n_head; (void)n_kv; (void)T; (void)npg; (void)tr;
  r8_attn_mp_body(0, q, kc, vc, part, out, pos, count, gcount, gmax, oq, gate, gstride, qst, ost, oqst, gst, pstride, (const G u8 *)0);
}

KERNEL ie_saluki_attn_long_mp_tr_nospill(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos,
                                         u32 n_head, u32 n_kv, u32 T, u32 npg, G u32 *count, G u32 *gcount, u32 gmax, G u8 *oq,
                                         const G float *gate, u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride,
                                         const G u8 *tr) {
  (void)n_head; (void)n_kv; (void)T; (void)npg;
  r8_attn_mp_body(1, q, kc, vc, part, out, pos, count, gcount, gmax, oq, gate, gstride, qst, ost, oqst, gst, pstride, tr);
}
#undef R8_TR_DEPTH
#undef R8_PIN_ACC8
#undef R8_PL_BSTR
#undef R8_PL_RSTR
#undef R8_PL_ROWB
#undef R8_PL_VS
#undef R8_PL_KS
#undef R8_PL_NB
#include "ie_saluki_r9_rq128.inc"
#include "ie_saluki_r9_verify.inc"
#include "ie_saluki_r10_prefill.inc"
#include "ie_saluki_r10_attn.inc"
#include "ie_saluki_r10_verify.inc"
#include "ie_saluki_r11_g01.inc"
#include "ie_saluki_r11_g02.inc"
#include "ie_saluki_r11_g03.inc"
#include "ie_saluki_r11_g04.inc"
#include "ie_saluki_r11_g05.inc"
#include "ie_saluki_r11_g06.inc"
#include "ie_saluki_r11_g07.inc"
#include "ie_saluki_r11_g08.inc"
#include "ie_saluki_r11_g09.inc"
#include "ie_saluki_r11_g10.inc"

