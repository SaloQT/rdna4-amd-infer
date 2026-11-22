/* cpu.c -- the CPU backend. It runs the same algorithm as the GPU kernels
 * (kernels/ie_kernels.c), with the same work split, so that the two agree up to float
 * rounding:
 *  - activations are quantized per block of 32: d = amax / 127,
 *    q = round(x / d), stored as words + d + asum (the sum of the 32 q);
 *  - GEMV row r: lane l (l < 32) adds, for blocks b = ie_lane_blk(l, t) < nb,
 *    d_w * d_a * (float)(int32)ie_q4q8_block(...); then the 32 lane sums are
 *    added in the order of ie_tree (shuffle down by 16, 8, 4, 2, 1).
 * All index and integer arithmetic comes from core/ie_core.h. */
#include <math.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <malloc.h>
#endif

#include "../core/ie_core.h"
#include "../core/ie_ext.h"
#include "backend.h"
#include "common.h"

/* Raw GGUF IQ/K blocks (MAT_RAWQ), the same source as the GPU kernels. */
#define IE_RAWQ_HOST
#define IE_RQP const uint8_t
#define IE_RQF const float
#define IE_RQU const uint32_t
#include "../kernels/ie_iqraw.inc"

typedef struct {
  backend base;
  const model *m;
  const graph *g;
  uint8_t *arena;
  float *kc, *vc; /* KV cache: [attention layer][pos][n_kv * hd] */
  /* qwen35 linear attention state per linear layer: the last 4 conv inputs
   * (ring[slot][conv_dim], slot = pos % 4) and S[head][i (key)][j (value)] */
  float *ring, *st;
  float *sc_ring; /* inkling2 sconv ring state: sc_tot * 4 floats ((scoff + ch) * 4 + (pos & 3)) */
  float *snap_ring[IE_SNAP_MAX], *snap_st[IE_SNAP_MAX]; /* the prompt cache snapshots (cpu_snap) */
  float *snap_sc[IE_SNAP_MAX];
  float *snap_kv[IE_SNAP_MAX];                /* and the KV rings of the sliding window layers (flint) */
  int dump; /* IE_DUMP_OPS=1: print every op's output (debug only) */
} cpu_backend;

#define BUF(b, id) ((void *)((b)->arena + (b)->g->bufs[id].off))

/* ---------------------------------------------------------------- kernels */

void cpu_quant_q8(const float *x, uint8_t *q, uint32_t nb) {
  int8_t *qv = (int8_t *)q;
  float *d = (float *)(q + 32u * nb);
  u32 *asum = (u32 *)(q + 36u * nb);
  for (uint32_t b = 0; b < nb; b++) {
    float amax = 0.0f;
    for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[32 * b + j]));
    float db = amax / 127.0f;
    u32 s = 0;
    for (int j = 0; j < 32; j++) {
      int v = db != 0.0f ? (int)roundf(x[32 * b + j] / db) : 0;
      qv[32 * b + j] = (int8_t)v;
      s += (u32)v;
    }
    d[b] = db;
    asum[b] = s;
  }
}

/* The wave tree reduction of ie_tree on 32 lane values (lane 0's result). */
static float wave_tree(float v[32]) {
  for (u32 m = 1; m <= 5; m++) {
    u32 off = 32u >> m; /* 16, 8, 4, 2, 1 */
    for (u32 i = 0; i < off; i++) v[i] = v[i] + v[i + off];
  }
  return v[0];
}

void cpu_gemv_q4(const mat *w, const uint8_t *xq, float *y) {
  const u32 nb = w->nb;
  const Mem qw = {w->qw}, aw = {(const u32 *)xq};
  const float *da = (const float *)(xq + 32u * nb);
  const u32 *asum = (const u32 *)(xq + 36u * nb);
  const u32 ng = ie_gemv_ngroups(w->rows);
  for (u32 g = 0; g < ng; g++)
    for (u32 v = 0; v < ie_rows_per_wg(); v++) {
      const u32 r = ie_gemv_row(g, v);
      if (r >= w->rows) continue;
      float lane[32];
      for (u32 l = 0; l < ie_wave(); l++) {
        float acc = 0.0f;
        for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) {
          const u32 b = ie_lane_blk(l, t);
          const int32_t dot = (int32_t)ie_q4q8_block(qw, ie_q4_dst_word(r, b, 0, nb), aw, b * 8u, asum[b]);
          const float dw = ie_f16_to_f32(w->qs[ie_q4_dst_scale(r, b, nb)]);
          acc += (dw * da[b]) * (float)dot;
        }
        lane[l] = acc;
      }
      y[r] = wave_tree(lane);
    }
}

/* MAT_Q4K matrix x Q8 activations: the same grid, lanes and reduction as
 * Q4. Per sub-block: dq = sum of q * a (q = 0..15, from the Q4_0 block dot
 * plus 8 asum), then acc += da * ((d sc) dq - (dmin mn) asum), in the order
 * of the GPU kernels (ie_gemv_q4kq8). */
void cpu_gemv_q4k(const mat *w, const uint8_t *xq, float *y) {
  const u32 nb = w->nb;
  const Mem qw = {w->qw}, aw = {(const u32 *)xq};
  const float *da = (const float *)(xq + 32u * nb);
  const u32 *asum = (const u32 *)(xq + 36u * nb);
  const u32 ng = ie_gemv_ngroups(w->rows);
  for (u32 g = 0; g < ng; g++)
    for (u32 v = 0; v < ie_rows_per_wg(); v++) {
      const u32 r = ie_gemv_row(g, v);
      if (r >= w->rows) continue;
      float lane[32];
      for (u32 l = 0; l < ie_wave(); l++) {
        float acc = 0.0f;
        for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) {
          const u32 b = ie_lane_blk(l, t);
          const int dq = (int)ie_q4k_dot(qw, ie_q4_dst_word(r, b, 0, nb), aw, b * 8u); /* law q4k_dot */
          const uint16_t sm = w->qs[ie_q4k_sm(r, b, nb)];
          const uint32_t dd = ie_q4k_dd(w->rows, r, b, nb);
          const float s = ie_f16_to_f32(w->qs[dd]) * (float)(sm & 63u), mm = ie_f16_to_f32(w->qs[dd + 1]) * (float)(sm >> 8);
          acc = fmaf(-mm, da[b] * (float)(int)asum[b], fmaf(s * (float)dq, da[b], acc)); /* as the GPU kernels */
        }
        lane[l] = acc;
      }
      y[r] = wave_tree(lane);
    }
}

/* MAT_Q6K matrix x Q8 activations: per sub-block, the two half dot
 * products (ie_q6k_dot, law q6k_dot), then
 * acc = fma(fma(d sc1, d1, (d sc0) d0), da, acc), as the GPU kernels. */
void cpu_gemv_q6k(const mat *w, const uint8_t *xq, float *y) {
  const u32 nb = w->nb, rows = w->rows;
  const Mem qw = {w->qw}, aw = {(const u32 *)xq};
  const float *da = (const float *)(xq + 32u * nb);
  const u32 ng = ie_gemv_ngroups(rows);
  for (u32 g = 0; g < ng; g++)
    for (u32 v = 0; v < ie_rows_per_wg(); v++) {
      const u32 r = ie_gemv_row(g, v);
      if (r >= rows) continue;
      float lane[32];
      for (u32 l = 0; l < ie_wave(); l++) {
        float acc = 0.0f;
        for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) {
          const u32 b = ie_lane_blk(l, t), qb = ie_q4_dst_word(r, b, 0, nb), hb = ie_q6k_hw(rows, r, b, 0, nb);
          const int d0 = (int)ie_q6k_dots(qw, qb, qw, hb, aw, b * 8u, 0u);
          const int d1 = (int)ie_q6k_dots(qw, qb, qw, hb + 1u, aw, b * 8u, 1u);
          const uint16_t sc = w->qs[ie_q6k_sc(r, b, nb)];
          const float d = ie_f16_to_f32(w->qs[ie_q6k_d(rows, r, b, nb)]);
          const float s0 = d * (float)(int8_t)(sc & 255u), s1 = d * (float)(int8_t)(sc >> 8);
          acc = fmaf(fmaf(s1, (float)d1, s0 * (float)d0), da[b], acc);
        }
        lane[l] = acc;
      }
      y[r] = wave_tree(lane);
    }
}

/* Q8_0 matrix x Q8 activations: the same grid, lanes and reduction as Q4;
 * the block dot product is ie_q8q8_block (law q8q8_block). */
void cpu_gemv_q8(const mat *w, const uint8_t *xq, float *y) {
  const u32 nb = w->nb;
  const Mem qw = {w->qw}, aw = {(const u32 *)xq};
  const float *da = (const float *)(xq + 32u * nb);
  const u32 ng = ie_gemv_ngroups(w->rows);
  for (u32 g = 0; g < ng; g++)
    for (u32 v = 0; v < ie_rows_per_wg(); v++) {
      const u32 r = ie_gemv_row(g, v);
      if (r >= w->rows) continue;
      float lane[32];
      for (u32 l = 0; l < ie_wave(); l++) {
        float acc = 0.0f;
        for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) {
          const u32 b = ie_lane_blk(l, t);
          const int32_t dot = (int32_t)ie_q8q8_block(qw, ie_q8_dst_word(r, b, 0, nb), aw, b * 8u);
          const float dw = ie_f16_to_f32(w->qs[ie_q8_dst_scale(r, b, nb)]);
          acc += (dw * da[b]) * (float)dot;
        }
        lane[l] = acc;
      }
      y[r] = wave_tree(lane);
    }
}

/* MAT_Q5K matrix x Q8 activations: as Q4K, with dq = ie_q5k_dot (q = 0..31). */
void cpu_gemv_q5k(const mat *w, const uint8_t *xq, float *y) {
  const u32 nb = w->nb, rows = w->rows;
  const Mem qw = {w->qw}, aw = {(const u32 *)xq};
  const float *da = (const float *)(xq + 32u * nb);
  const u32 *asum = (const u32 *)(xq + 36u * nb);
  for (u32 r = 0; r < rows; r++) {
    float lane[32];
    for (u32 l = 0; l < ie_wave(); l++) {
      float acc = 0.0f;
      for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) {
        const u32 b = ie_lane_blk(l, t);
        const int dq = (int)ie_q5k_dot(qw, ie_q4_dst_word(r, b, 0, nb), w->qw[ie_q5k_hw(rows, r, b, nb)], aw, b * 8u);
        const uint16_t sm = w->qs[ie_q4k_sm(r, b, nb)];
        const uint32_t dd = ie_q4k_dd(rows, r, b, nb);
        const float s = ie_f16_to_f32(w->qs[dd]) * (float)(sm & 63u), mm = ie_f16_to_f32(w->qs[dd + 1]) * (float)(sm >> 8);
        acc = fmaf(-mm, da[b] * (float)(int)asum[b], fmaf(s * (float)dq, da[b], acc));
      }
      lane[l] = acc;
    }
    y[r] = wave_tree(lane);
  }
}

/* MAT_IQ4XS matrix x Q8 activations: dq = ie_iq4_dot (the table values),
 * acc = fma(d (ls - 32) dq, da, acc). */
void cpu_gemv_iq4xs(const mat *w, const uint8_t *xq, float *y) {
  const u32 nb = w->nb, rows = w->rows;
  const Mem qw = {w->qw}, aw = {(const u32 *)xq};
  const float *da = (const float *)(xq + 32u * nb);
  for (u32 r = 0; r < rows; r++) {
    float lane[32];
    for (u32 l = 0; l < ie_wave(); l++) {
      float acc = 0.0f;
      for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) {
        const u32 b = ie_lane_blk(l, t);
        const int dq = (int)ie_iq4_dot(qw, ie_q4_dst_word(r, b, 0, nb), aw, b * 8u);
        const float s = ie_f16_to_f32(w->qs[ie_q4k_dd(rows, r, b, nb)]) * (float)((int)(w->qs[ie_q4k_sm(r, b, nb)] & 63u) - 32);
        acc = fmaf(s * (float)dq, da[b], acc);
      }
      lane[l] = acc;
    }
    y[r] = wave_tree(lane);
  }
}

static void cpu_gemv_f32(const mat *w, const float *x, float *y);

/* MAT_RAWQ matrix x Q8 activations: the raw GGUF 256-weight blocks are read
 * directly (kernels/ie_iqraw.inc); lane l takes the blocks l, l + 32, ... of
 * the row, as the decode kernel does. */
void cpu_gemv_rawq(const mat *w, const uint8_t *xq, float *y) {
  const u32 nb = w->nb, rows = w->rows, mb = nb / 8u, bs = ie_rq_bs(w->sub);
  for (u32 r = 0; r < rows; r++) {
    float lane[32];
    const uint8_t *row = (const uint8_t *)w->qs + (size_t)r * mb * bs;
    for (u32 l = 0; l < 32; l++) {
      /* the GPU kernel's sub-block mapping: lane l takes the 32-weight blocks
       * l, l + 32, ... of the row */
      float acc = 0.0f;
      for (u32 b = l; b < nb; b += 32) {
        const u32 qb = b >> 3u, sb = b & 7u;
        acc += ie_rq_dot_sub(w->sub, row + (size_t)qb * bs, xq, nb, qb, sb);
      }
      lane[l] = acc;
    }
    y[r] = wave_tree(lane);
  }
}

static void cpu_gemv_iq4nl(const mat *w, const uint8_t *xq, float *y) {
  const u32 nb = w->nb;
  const Mem qw = {w->qw}, aw = {(const u32 *)xq};
  const float *da = (const float *)(xq + 32u * nb);
  const u32 ng = ie_gemv_ngroups(w->rows);
  for (u32 g = 0; g < ng; g++)
    for (u32 v = 0; v < ie_rows_per_wg(); v++) {
      const u32 r = ie_gemv_row(g, v);
      if (r >= w->rows) continue;
      float lane[32];
      for (u32 l = 0; l < ie_wave(); l++) {
        float acc = 0.0f;
        for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) {
          const u32 b = ie_lane_blk(l, t);
          const int32_t dot = (int32_t)ie_iq4_dot(qw, ie_q4_dst_word(r, b, 0, nb), aw, b * 8u);
          const float dw = ie_f16_to_f32(w->qs[ie_q4_dst_scale(r, b, nb)]);
          acc += (dw * da[b]) * (float)dot;
        }
        lane[l] = acc;
      }
      y[r] = wave_tree(lane);
    }
}

/* y = W x for any kind (x: Q8 bytes, or floats for MAT_F32). */
static void cpu_gemv_any(const mat *w, const void *x, float *y) {
  switch (w->kind) {
    case MAT_Q4: cpu_gemv_q4(w, x, y); break;
    case MAT_IQ4NL: cpu_gemv_iq4nl(w, x, y); break;
    case MAT_Q4K: cpu_gemv_q4k(w, x, y); break;
    case MAT_Q5K: cpu_gemv_q5k(w, x, y); break;
    case MAT_Q6K: cpu_gemv_q6k(w, x, y); break;
    case MAT_Q8: cpu_gemv_q8(w, x, y); break;
    case MAT_IQ4XS: cpu_gemv_iq4xs(w, x, y); break;
    case MAT_RAWQ: cpu_gemv_rawq(w, x, y); break;
    default: cpu_gemv_f32(w, x, y); break;
  }
}

/* f32 matrix: the same grid and lane split over elements. */
static void cpu_gemv_f32(const mat *w, const float *x, float *y) {
  for (u32 r = 0; r < w->rows; r++) {
    float lane[32];
    const float *row = w->f + (size_t)r * w->cols;
    for (u32 l = 0; l < 32; l++) {
      float acc = 0.0f;
      for (u32 t = 0; ie_lane_blk(l, t) < w->cols; t++) acc += row[ie_lane_blk(l, t)] * x[ie_lane_blk(l, t)];
      lane[l] = acc;
    }
    y[r] = wave_tree(lane);
  }
}

/* Row r of a matrix as floats (the embedding). */
static void mat_row(const mat *w, u32 r, float *out) {
  if (w->kind == MAT_F32) {
    memcpy(out, w->f + (size_t)r * w->cols, w->cols * 4);
    return;
  }
  if (w->kind == MAT_Q8) {
    const Mem qw = {w->qw};
    for (u32 b = 0; b < w->nb; b++) {
      const float d = ie_f16_to_f32(w->qs[ie_q8_dst_scale(r, b, w->nb)]);
      for (u32 j = 0; j < 32; j++)
        out[32 * b + j] = d * (float)(int8_t)ie_byte(IE_LOAD(qw, ie_q8_dst_word(r, b, j >> 2, w->nb)), j & 3u);
    }
    return;
  }
  const Mem qw = {w->qw};
  for (u32 b = 0; b < w->nb; b++) {
    const float d = ie_f16_to_f32(w->qs[ie_q4_dst_scale(r, b, w->nb)]);
    for (u32 j = 0; j < 32; j++) {
      const u32 nib = ie_q4_nib(IE_LOAD(qw, ie_q4_dst_word(r, b, ie_q4_word_of(j), w->nb)), j);
      const int q = w->kind == MAT_IQ4NL ? (int)(int8_t)ie_kv_byte(nib) : (int)nib - 8;
      out[32 * b + j] = d * (float)q;
    }
  }
}

static void rmsnorm(const float *x, const float *wt, float *y, u32 n, float eps) {
  float ss = 0.0f;
  for (u32 i = 0; i < n; i++) ss += x[i] * x[i];
  const float s = 1.0f / sqrtf(ss / (float)n + eps);
  for (u32 i = 0; i < n; i++) y[i] = x[i] * s * wt[i];
}

/* RoPE of the first nrot dims of each of nh heads of hd (stride hd). */
static void rope(float *x, u32 nh, u32 hd, u32 nrot, int neox, const float *c, const float *s) {
  const u32 h2 = nrot / 2;
  for (u32 h = 0; h < nh; h++) {
    float *v = x + h * hd;
    for (u32 i = 0; i < h2; i++) {
      const u32 i0 = neox ? i : 2 * i, i1 = neox ? i + h2 : 2 * i + 1;
      const float a = v[i0], b = v[i1];
      v[i0] = a * c[i] - b * s[i];
      v[i1] = a * s[i] + b * c[i];
    }
  }
}

/* Attention of one decode token over the positions p0 .. pos (win: the
 * last win positions; 0: all), position p in slot p % len of the cache.
 * sinks (or NULL): a logit per head in the softmax denominator only.
 * Scores buffer: [n_head][n_ctx]. */
static void attention(const model *m, const float *q, const float *kc, const float *vc, u32 pos, u32 len, u32 win, const float *sinks,
                      u32 n_ctx, float *sc, float *out, const float *gate, u32 gstride, const float *r, const float *rel_bias) {
  const u32 hd = m->hd, kvd = m->n_kv * hd, grp = m->n_head / m->n_kv;
  const float scale = 1.0f / sqrtf((float)hd);
  const u32 p0 = win && pos + 1 > win ? pos + 1 - win : 0, np = pos + 1 - p0;
  for (u32 h = 0; h < m->n_head; h++) {
    const float *qh = q + h * hd;
    const u32 kh = h / grp; /* GQA: the KV head of query head h */
    float *s = sc + (size_t)h * n_ctx;
    float mx = sinks ? sinks[h] : -INFINITY;
    for (u32 i = 0; i < np; i++) {
      const u32 kp = p0 + i;
      const float *k = kc + (size_t)(kp % len) * kvd + kh * hd;
      float d = 0.0f;
      for (u32 j = 0; j < hd; j++) d += qh[j] * k[j];
      float sb = d * scale;
      if (r && rel_bias) {
        const u32 dist = pos - kp;
        if (dist < m->rel_extent) {
          const float *rh = r + (size_t)h * m->d_rel;
          const float *rb = rel_bias + (size_t)dist * m->d_rel;
          float rd = 0.0f;
          for (u32 j = 0; j < m->d_rel; j++) rd += rh[j] * rb[j];
          sb += rd;
        }
      }
      s[i] = sb;
      mx = fmaxf(mx, sb);
    }
    float sum = sinks ? expf(sinks[h] - mx) : 0.0f;
    for (u32 i = 0; i < np; i++) sum += (s[i] = expf(s[i] - mx));
    const float inv = 1.0f / sum;
    float *o = out + h * hd;
    for (u32 i = 0; i < hd; i++) {
      float acc = 0.0f;
      for (u32 j = 0; j < np; j++) acc += s[j] * vc[(size_t)((p0 + j) % len) * kvd + kh * hd + i];
      o[i] = acc * inv;
      if (gate) o[i] = o[i] * (1.0f / (1.0f + expf(-gate[h * gstride + i])));
    }
  }
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

/* qwen35, flint: per head RMSNorm of hd values at src (weight w), RoPE of
 * the first nrot dims (NEOX pairs; not if norope), into dst. */
static void qk_head(const model *m, const float *src, const float *w, const float *cs, const float *sn, float *dst, int norope) {
  rmsnorm(src, w, dst, m->hd, m->eps);
  if (!norope) rope(dst, 1, m->hd, m->n_rot, 1, cs, sn);
}

/* flint routing: top K of sigmoid(logit) + bias (ties: the smaller index),
 * weights = sigmoid / (their sum + 1e-20) * scale. out: [K ids | K weights]. */
static void moe_route(const float *lg, u32 n, u32 K, const float *bias, float scale, u32 *out) {
  float sg[1024], wsel[16];
  if (n > 1024 || K > 16) ie_die("moe_route: too many experts");
  for (u32 e = 0; e < n; e++) sg[e] = 1.0f / (1.0f + expf(-lg[e]));
  uint8_t used[1024] = {0};
  float sum = 0.0f;
  for (u32 k = 0; k < K; k++) {
    u32 best = n;
    float bv = -INFINITY;
    for (u32 e = 0; e < n; e++) {
      const float c = bias ? sg[e] + bias[e] : sg[e];
      if (!used[e] && (best == n || c > bv)) best = e, bv = c;
    }
    used[best] = 1;
    out[k] = best;
    wsel[k] = sg[best];
    sum += sg[best];
  }
  const float inv = 1.0f / (sum + 1e-20f);
  for (u32 k = 0; k < K; k++) {
    const float wk = wsel[k] * inv * scale;
    memcpy(&out[K + k], &wk, 4);
  }
}

/* qwen35 linear attention (Gated DeltaNet) of one token, as ie_gdn. in =
 * [q k v | z | beta | alpha]; ring, S: the state of this layer. */
static void gdn(const model *m, const layer *L, const float *in, float *out, float *ring, float *S, u32 pos) {
  const u32 sd = m->sd, cd = m->conv_dim, nv = m->n_vh, nk = m->n_kh, inner = nv * sd;
  float *cv = ie_alloc((size_t)cd * 4);
  /* causal conv over [x(pos-3) .. x(pos)] (0 before position 0), then SiLU */
  for (u32 c = 0; c < cd; c++) {
    float acc = 0.0f;
    for (u32 j = 0; j < 4; j++) {
      const u32 d = 3 - j;
      const float x = d == 0 ? in[c] : pos >= d ? ring[((pos - d) & 3u) * cd + c] : 0.0f;
      acc += x * L->conv.f[c * 4 + j];
    }
    cv[c] = silu(acc);
  }
  for (u32 c = 0; c < cd; c++) ring[(pos & 3u) * cd + c] = in[c];
  /* L2 norm of q and k per key head */
  for (u32 h = 0; h < 2 * nk; h++) {
    float ss = 0.0f;
    for (u32 i = 0; i < sd; i++) ss += cv[h * sd + i] * cv[h * sd + i];
    const float r = 1.0f / sqrtf(ss + m->eps);
    for (u32 i = 0; i < sd; i++) cv[h * sd + i] *= r;
  }
  const float scale = 1.0f / sqrtf((float)sd);
  float *o = ie_alloc(sd * 4), *delta = ie_alloc(sd * 4);
  for (u32 h = 0; h < nv; h++) {
    const float *q = cv + (h % nk) * sd, *k = cv + nk * sd + (h % nk) * sd, *v = cv + 2 * nk * sd + h * sd;
    const float beta = 1.0f / (1.0f + expf(-in[cd + inner + h]));
    const float a = in[cd + inner + nv + h] + L->dt_bias.f[h];
    const float sp = a > 20.0f ? a : logf(1.0f + expf(a)); /* as ggml_compute_softplus_f32 */
    const float decay = expf(sp * L->ssm_a.f[h]);
    float *Sh = S + (size_t)h * sd * sd; /* Sh[i * sd + j] = S[i][j] */
    for (u32 j = 0; j < sd; j++) {
      float acc = 0.0f;
      for (u32 i = 0; i < sd; i++) {
        const float x = pos ? Sh[i * sd + j] * decay : 0.0f;
        Sh[i * sd + j] = x;
        acc += x * k[i];
      }
      delta[j] = (v[j] - acc) * beta;
    }
    for (u32 j = 0; j < sd; j++) {
      float acc = 0.0f;
      for (u32 i = 0; i < sd; i++) {
        Sh[i * sd + j] += k[i] * delta[j];
        acc += Sh[i * sd + j] * q[i];
      }
      o[j] = acc * scale;
    }
    /* gated RMSNorm: rmsnorm(o) * ssm_norm * silu(z) */
    float ss = 0.0f;
    for (u32 j = 0; j < sd; j++) ss += o[j] * o[j];
    const float r = 1.0f / sqrtf(ss / (float)sd + m->eps);
    for (u32 j = 0; j < sd; j++) out[h * sd + j] = o[j] * r * L->ssm_norm.f[j] * silu(in[cd + h * sd + j]);
  }
  free(cv), free(o), free(delta);
}

/* ---------------------------------------------------------------- backend */

static uint32_t cpu_step(backend *bk, uint32_t tok, uint32_t pos, float *logits, double *ms);
static void cpu_close(backend *bk);

/* The prompt cache snapshot (backend.snap): the linear attention state and the conv ring. */
static void cpu_snap(backend *bk, int save, int k, int hbuf, uint32_t pos) {
  cpu_backend *b = (cpu_backend *)bk;
  const model *m = b->m;
  const size_t rb = (size_t)m->n_rec * 4 * m->conv_dim * 4, sb = (size_t)m->n_rec * m->n_vh * m->sd * m->sd * 4;
  const size_t scb = (size_t)m->sc_tot * 4u * 4u;
  (void)hbuf; /* no MTP on the CPU */
  (void)pos;
  if (!b->snap_ring[k]) b->snap_ring[k] = ie_alloc(rb + 4), b->snap_st[k] = ie_alloc(sb + 4), b->snap_sc[k] = ie_alloc(scb + 4);
  memcpy(save ? b->snap_ring[k] : b->ring, save ? b->ring : b->snap_ring[k], rb);
  memcpy(save ? b->snap_st[k] : b->st, save ? b->st : b->snap_st[k], sb);
  memcpy(save ? b->snap_sc[k] : b->sc_ring, save ? b->sc_ring : b->snap_sc[k], scb);
  /* a ring slot is reused by a later position: keep the rings (K and V) */
  const graph *g = b->g;
  const size_t kvd = (size_t)m->n_kv * m->hd;
  size_t nr = 0;
  for (uint32_t i = 0; i < m->n_kvl; i++)
    if (g->kv_len[i] < g->n_ctx) nr += g->kv_len[i];
  if (!nr) return;
  if (!b->snap_kv[k]) b->snap_kv[k] = ie_alloc(2 * nr * kvd * 4);
  float *sp = b->snap_kv[k];
  for (uint32_t i = 0; i < m->n_kvl; i++) {
    if (g->kv_len[i] >= g->n_ctx) continue;
    const size_t n = g->kv_len[i] * kvd, o = g->kv_off[i] * kvd;
    memcpy(save ? sp : b->kc + o, save ? b->kc + o : sp, n * 4), sp += n;
    memcpy(save ? sp : b->vc + o, save ? b->vc + o : sp, n * 4), sp += n;
  }
}

/* backend.snap_io / kv_io (the disk prompt cache; tests on the tiny models). The snapshot holds the conv ring slots of
 * positions pos - 3 .. pos - 1 only: gdn() reads the ring at those positions and writes slot pos & 3 (the fourth slot)
 * after reading, and every later read is of a slot written since. The fourth slot reads as zero after a restore. */
static size_t cpu_snap_io(backend *bk, int k, int out, void *buf, uint32_t pos) {
  cpu_backend *b = (cpu_backend *)bk;
  const model *m = b->m;
  for (uint32_t i = 0; i < m->n_kvl; i++)
    if (b->g->kv_len[i] < b->g->n_ctx) return SIZE_MAX;
  const size_t rowb = (size_t)m->conv_dim * 4, rb = (size_t)m->n_rec * 3 * rowb, rall = (size_t)m->n_rec * 4 * rowb,
               sb = (size_t)m->n_rec * m->n_vh * m->sd * m->sd * 4;
  if (!buf) return rb + sb;
  if (!b->snap_ring[k]) {
    if (out) ie_die("cpu_snap_io: no snapshot %d to write out", k);
    b->snap_ring[k] = ie_alloc(rall + 4), b->snap_st[k] = ie_alloc(sb + 4);
  }
  char *p = buf, *ring = (char *)b->snap_ring[k];
  memcpy(out ? p : (char *)b->snap_st[k], out ? (char *)b->snap_st[k] : p, sb);
  p += sb;
  if (!out) memset(ring, 0, rall);
  for (uint32_t l = 0; l < m->n_rec; l++)
    for (uint32_t j = 0; j < 3; j++, p += rowb) {
      char *slot = ring + ((size_t)l * 4 + ((pos + 1u + j) & 3u)) * rowb;
      memcpy(out ? p : slot, out ? slot : p, rowb);
    }
  return rb + sb;
}
static void cpu_kv_io(backend *bk, int out, uint32_t p0, uint32_t n, void *buf) {
  if (!n) return;
  cpu_backend *b = (cpu_backend *)bk;
  const model *m = b->m;
  const size_t kvd = (size_t)m->n_kv * m->hd, len = (size_t)n * kvd;
  float *p = buf;
  for (uint32_t i = 0; i < m->n_kvl; i++)
    for (int v = 0; v < 2; v++, p += len) {
      float *rows = (v ? b->vc : b->kc) + ((size_t)b->g->kv_off[i] + p0) * kvd;
      memcpy(out ? p : rows, out ? rows : p, len * 4);
    }
}

/* buffers for the disk cache: aligned to 4096 (as the GPU's pinned ones) */
static void *cpu_host_alloc(backend *bk, size_t n) {
  (void)bk;
#ifdef _WIN32
  return _aligned_malloc(n ? n : 1, 4096);
#else
  void *p = NULL;
  return posix_memalign(&p, 4096, n ? n : 1) ? NULL : p;
#endif
}
static void cpu_host_free(backend *bk, void *p) {
  (void)bk;
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif
}

static void cpu_set_rope(backend *bk, uint32_t pos0, uint32_t n, const float *cos_tab, const float *sin_tab) {
  cpu_backend *b = (cpu_backend *)bk;
  const uint32_t h2 = b->m->n_rot / 2u;
  if (!n || pos0 + n > b->g->n_ctx) return;
  memcpy(b->g->rope_cos + (size_t)pos0 * h2, cos_tab, (size_t)n * h2 * 4u);
  memcpy(b->g->rope_sin + (size_t)pos0 * h2, sin_tab, (size_t)n * h2 * 4u);
}

backend *cpu_open(const model *m, const graph *g) {
  cpu_backend *b = calloc(1, sizeof *b);
  b->base.step = cpu_step;
  b->base.close = cpu_close;
  b->base.snap = cpu_snap;
  b->base.snap_io = cpu_snap_io;
  b->base.kv_io = cpu_kv_io;
  b->base.set_rope = cpu_set_rope;
  b->base.snap_f32b = (size_t)m->n_rec * m->n_vh * m->sd * m->sd * 4;
  b->base.host_alloc = cpu_host_alloc;
  b->base.host_free = cpu_host_free;
  b->base.kv_rowb = (size_t)m->n_kv * m->hd * 4 * 2 * m->n_kvl;
  b->m = m;
  b->g = g;
  {
    const char *dv = getenv("IE_DUMP_OPS");
    b->dump = dv && *dv && strcmp(dv, "0") != 0;
  }
  b->arena = ie_alloc(g->arena);
  const size_t kv = (size_t)g->kv_total * m->n_kv * m->hd;
  b->kc = ie_alloc(kv * 4 + 4);
  b->vc = ie_alloc(kv * 4 + 4);
  b->ring = ie_alloc((size_t)m->n_rec * 4 * m->conv_dim * 4 + 4);
  b->st = ie_alloc((size_t)m->n_rec * m->n_vh * m->sd * m->sd * 4 + 4);
  b->sc_ring = ie_alloc((size_t)m->sc_tot * 4u * 4u + 4u);
  return &b->base;
}

static void cpu_close(backend *bk) {
  cpu_backend *b = (cpu_backend *)bk;
  free(b->arena);
  free(b->kc);
  free(b->vc);
  free(b->ring);
  free(b->st);
  free(b->sc_ring);
  for (int k = 0; k < IE_SNAP_MAX; k++) free(b->snap_ring[k]), free(b->snap_st[k]), free(b->snap_sc[k]), free(b->snap_kv[k]);
  free(b);
}

static uint32_t cpu_step(backend *bk, uint32_t tok, uint32_t pos, float *logits, double *ms) {
  cpu_backend *B = (cpu_backend *)bk;
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  const model *m = B->m;
  const graph *g = B->g;
  const u32 h2 = m->n_rot / 2, kvd = m->n_kv * m->hd;
  if (tok >= m->vocab) ie_die("token %u >= vocab %u", tok, m->vocab);
  if (pos >= g->n_ctx) ie_die("position %u >= context %u", pos, g->n_ctx);
  for (u32 i = 0; i < g->n_ops; i++) {
    const op *o = &g->ops[i];
    float *a = o->a >= 0 ? (float *)((char *)BUF(B, o->a) + o->aoff) : NULL;
    float *bb = o->b >= 0 ? (float *)((char *)BUF(B, o->b) + o->boff) : NULL;
    float *cc = o->c >= 0 ? (float *)((char *)BUF(B, o->c) + o->coff) : NULL;
    uint8_t *qo = o->qo >= 0 ? (uint8_t *)BUF(B, o->qo) : NULL;
    const float *res = o->res >= 0 ? (const float *)BUF(B, o->res) : NULL;
    switch (o->kind) {
      case OP_EMBED:
        if (B->base.ext_rows && B->base.ext_rows[0]) {
          memcpy(a, B->base.ext_rows[0], (size_t)m->dim * 4u);
        } else {
          mat_row(o->w ? o->w : &m->tok, tok, a);
          if (o->escale != 0.0f) for (u32 j = 0; j < o->n; j++) a[j] *= o->escale;
        }
        break;
      case OP_RMSNORM:
        rmsnorm(a, o->v->f, bb, o->n, m->eps);
        if (qo) cpu_quant_q8(bb, qo, o->n / 32);
        break;
      case OP_QUANT: cpu_quant_q8(a, (uint8_t *)bb, o->n / 32); break;
      case OP_GEMV_Q4:
      case OP_GEMV_IQ4NL:
      case OP_GEMV_Q4K:
      case OP_GEMV_Q5K:
      case OP_GEMV_Q6K:
      case OP_GEMV_Q8:
      case OP_GEMV_IQ4XS:
      case OP_GEMV_RAW:
      case OP_GEMV_F32:
        cpu_gemv_any(o->w, a, bb);
        /* the fused epilogue, in the order of the GPU kernels */
        for (u32 j = 0; j < o->w->rows; j++) {
          float y = bb[j];
          if (o->v) y = y + o->v->f[j];
          if (res) y = res[j] + y;
          bb[j] = y;
        }
        if (o->nv) { /* the fused RMSNorm of the result */
          float *ny = (float *)BUF(B, o->nout);
          rmsnorm(bb, o->nv->f, ny, o->w->rows, m->eps);
          if (o->nq >= 0) cpu_quant_q8(ny, (uint8_t *)BUF(B, o->nq), o->w->rows / 32);
        }
        break;
      case OP_BIAS:
        for (u32 j = 0; j < o->n; j++) a[j] += o->v->f[j];
        break;
      case OP_ADD: {
        for (u32 j = 0; j < o->n; j++) cc[j] = a[j] + bb[j];
        break;
      }
      case OP_COPY:
        memcpy(bb, a, (size_t)o->n * 4u);
        break;
      case OP_ROPE:
        rope(a, o->nh, m->hd, m->n_rot, m->rope == ROPE_NEOX, g->rope_cos + (size_t)pos * h2, g->rope_sin + (size_t)pos * h2);
        break;
      case OP_ROPE_KV: {
        const float *cs = g->rope_cos + (size_t)pos * h2, *sn = g->rope_sin + (size_t)pos * h2;
        rope(a, o->nh, m->hd, m->n_rot, m->rope == ROPE_NEOX, cs, sn);
        rope(bb, m->n_kv, m->hd, m->n_rot, m->rope == ROPE_NEOX, cs, sn);
        const u32 kvi = o->kvi;
        const size_t base = (g->kv_off[kvi] + pos % g->kv_len[kvi]) * kvd;
        memcpy(B->kc + base, bb, kvd * 4);
        memcpy(B->vc + base, cc, kvd * 4);
        break;
      }
      case OP_QKN_ROPE_KV: {
        const float *cs = g->rope_cos + (size_t)pos * h2, *sn = g->rope_sin + (size_t)pos * h2;
        const u32 kvi = o->kvi;
        const size_t base = (g->kv_off[kvi] + pos % g->kv_len[kvi]) * kvd;
        const u32 hd = m->hd, qr = o->nh * o->qstr;
        for (u32 h = 0; h < o->nh; h++) qk_head(m, a + h * o->qstr, o->v->f, cs, sn, bb + h * hd, o->norope);
        for (u32 h = 0; h < m->n_kv; h++) qk_head(m, a + qr + h * hd, o->nv->f, cs, sn, B->kc + base + h * hd, o->norope);
        memcpy(B->vc + base, a + qr + kvd, kvd * 4);
        break;
      }
      case OP_GDN: {
        const layer *L = &m->l[o->layer];
        gdn(m, L, a, bb, B->ring + (size_t)L->sti * 4 * m->conv_dim, B->st + (size_t)L->sti * m->n_vh * m->sd * m->sd, pos);
        if (qo) cpu_quant_q8(bb, qo, o->n / 32);
        break;
      }
      case OP_KV: {
        const u32 kvi = o->kvi;
        const size_t base = (g->kv_off[kvi] + pos % g->kv_len[kvi]) * kvd;
        memcpy(B->kc + base, a, kvd * 4);
        memcpy(B->vc + base, bb, kvd * 4);
        break;
      }
      case OP_ATTN: {
        const u32 kvi = o->kvi;
        const size_t base = g->kv_off[kvi] * kvd;
        const float *gate = o->gt >= 0 ? (const float *)((char *)BUF(B, o->gt) + o->gtoff) : NULL;
        attention(m, a, B->kc + base, B->vc + base, pos, g->kv_len[kvi], o->win, o->sk ? o->sk->f : NULL, g->n_ctx, cc, bb, gate,
                  o->gstride, NULL, NULL);
        if (qo) cpu_quant_q8(bb, qo, o->n / 32);
        break;
      }
      case OP_VCLIP: {
        const u32 hdim = o->qstr;
        const float *wt = o->v ? o->v->f : NULL;
        const float tau = o->norope ? 1.0f + o->escale * logf(fmaxf((float)(pos + 1u) / (float)o->win, 1.0f)) : 1.0f;
        for (u32 h = 0; h < o->nh; h++) {
          const float *in = a + (size_t)h * hdim;
          float *out = bb + (size_t)h * hdim;
          float ss = 0.0f;
          for (u32 j = 0; j < hdim; j++) {
            float val = in[j];
            if (wt && o->clamp > 0.0f) val *= wt[h * hdim + j];
            out[j] = val;
            ss += val * val;
          }
          if (o->clamp > 0.0f) {
            const float norm = sqrtf(ss);
            const float sc = fminf(1.0f, o->clamp / (norm + 1e-6f)) * tau;
            for (u32 j = 0; j < hdim; j++) out[j] *= sc;
          } else {
            const float s = 1.0f / sqrtf(ss / (float)hdim + o->escale);
            for (u32 j = 0; j < hdim; j++) out[j] *= wt ? s * wt[h * hdim + j] : s;
          }
        }
        break;
      }
      case OP_ATTN_RES: {
        const u32 M = o->nh, K = o->qstr, D = o->n;
        if (M == 0 || M > 64u) ie_die("OP_ATTN_RES: bad slot count %u", M);
        const float *slots[64];
        float irms[64], logits_s[64];
        for (u32 s = 0; s < M; s++) {
          const float *p = (cc && s == M - 1u) ? cc : (a + (size_t)s * D);
          slots[s] = p;
          float ss = 0.0f;
          for (u32 j = 0; j < D; j++) ss += p[j] * p[j];
          irms[s] = 1.0f / sqrtf(ss / (float)D + m->eps);
        }
        const float inv_sqrt_d = 1.0f / sqrtf((float)D);
        for (u32 k = 0; k < K; k++) {
          const float *qk = o->v->f + o->roff + (size_t)k * D;
          float mx = -INFINITY;
          for (u32 s = 0; s < M; s++) {
            float dot = 0.0f;
            const float *p = slots[s];
            const float ir = irms[s];
            for (u32 j = 0; j < D; j++) dot += (p[j] * ir) * qk[j];
            logits_s[s] = dot * inv_sqrt_d;
            if (logits_s[s] > mx) mx = logits_s[s];
          }
          float sum = 0.0f;
          for (u32 s = 0; s < M; s++) sum += (logits_s[s] = expf(logits_s[s] - mx));
          const float inv = 1.0f / sum;
          float *out = bb + (size_t)k * D;
          for (u32 j = 0; j < D; j++) {
            float acc = 0.0f;
            for (u32 s = 0; s < M; s++) acc += (logits_s[s] * inv) * (slots[s][j] * irms[s]);
            out[j] = acc;
          }
        }
        break;
      }
      case OP_LOOP_GATE: {
        const float *fg = (const float *)((const char *)BUF(B, o->gt) + o->gtoff);
        const float *f = fg, *g_ = fg + o->n;
        for (u32 j = 0; j < o->n; j++) bb[j] = (1.0f - tanhf(f[j])) * a[j] + tanhf(g_[j]) * cc[j];
        break;
      }
      case OP_SCONV: {
        const u32 hdim = cc && o->nh ? o->n / o->nh : 0u;
        const float *wstat = o->v->f;
        for (u32 i_c = 0; i_c < o->n; i_c++) {
          float W[4];
          W[3] = a[i_c];
          for (u32 lag = 1u; lag <= 3u; lag++)
            W[3u - lag] = pos >= lag ? B->sc_ring[((size_t)o->scoff + i_c) * 4u + ((pos - lag) & 3u)] : 0.0f;
          const float *w = wstat + (size_t)i_c * 4u;
          float y = w[0] * W[0] + w[1] * W[1] + w[2] * W[2] + w[3] * W[3];
          if (cc) {
            const float *d = cc + (size_t)(i_c / hdim) * 4u;
            y += d[0] * W[3] + d[1] * W[2] + d[2] * W[1] + d[3] * W[0];
          }
          bb[i_c] = (W[3] + y) * o->escale;
        }
        for (u32 i_c = 0; i_c < o->n; i_c++) B->sc_ring[((size_t)o->scoff + i_c) * 4u + (pos & 3u)] = a[i_c];
        break;
      }
      case OP_REL_ATTN: {
        const u32 kvi = o->kvi;
        const size_t base = g->kv_off[kvi] * kvd;
        const float *r = (const float *)BUF(B, o->nq);
        const float *sk = o->res >= 0 ? (const float *)BUF(B, o->res) : NULL;
        attention(m, a, B->kc + base, B->vc + base, pos, g->kv_len[kvi], o->win, sk, g->n_ctx, cc, bb, NULL, 0, r, o->v->f);
        if (qo) cpu_quant_q8(bb, qo, o->n / 32);
        break;
      }
      case OP_SWIGLU:
        /* ricursive: gate clamped above, up on both sides (o->clamp 0: no clamp) */
        for (u32 j = 0; j < o->n; j++) {
          float g = a[j], u = bb[j];
          if (o->clamp > 0.0f) {
            if (g > o->clamp) g = o->clamp;
            if (u > o->clamp) u = o->clamp;
            else if (u < -o->clamp) u = -o->clamp;
          }
          a[j] = g / (1.0f + expf(-g)) * u;
        }
        if (qo) cpu_quant_q8(a, qo, o->n / 32);
        break;
      case OP_GATE: /* ricursive: a *= sigmoid(b) */
        for (u32 j = 0; j < o->n; j++) a[j] *= 1.0f / (1.0f + expf(-bb[j]));
        if (qo) cpu_quant_q8(a, qo, o->n / 32);
        break;
      case OP_VRES: { /* b = c + (a - c) * sigmoid(lam) */
        const float *x0 = cc;
        const float s = 1.0f / (1.0f + expf(-o->lam));
        for (u32 j = 0; j < o->n; j++) bb[j] = x0[j] + (a[j] - x0[j]) * s;
        if (qo) cpu_quant_q8(bb, qo, o->n / 32);
        break;
      }
      case OP_MOE_ROUTE: moe_route(a, o->n, o->nh, o->v ? o->v->f : NULL, m->escale, (u32 *)bb); break;
      case OP_GEMV_EXP: {
        const u32 *ids = (const u32 *)cc;
        for (u32 sl = 0; sl < o->nh; sl++) {
          const mat e = mat_expert(o->w, ids[sl]);
          cpu_gemv_any(&e, (const char *)a + (size_t)sl * o->astr, (float *)((char *)bb + (size_t)sl * o->bstr));
        }
        break;
      }
      case OP_MOE_SWIGLU:
        for (u32 sl = 0; sl < o->nh; sl++) {
          float *gs = (float *)((char *)a + (size_t)sl * o->astr);
          for (u32 j = 0; j < o->n; j++) gs[j] = gs[j] / (1.0f + expf(-gs[j])) * gs[o->n + j];
          cpu_quant_q8(gs, qo + (size_t)sl * o->bstr, o->n / 32);
        }
        break;
      case OP_SITU:
        for (u32 sl = 0; sl < o->nh; sl++) {
          float *gs = (float *)((char *)a + (size_t)sl * o->astr);
          for (u32 j = 0; j < o->n; j++) {
            const float g0 = gs[j], u0 = gs[o->n + j];
            const float sg = o->clamp * tanhf(g0 / o->clamp) * (1.0f / (1.0f + expf(-g0)));
            const float su = o->lam * tanhf(u0 / o->lam);
            gs[j] = sg * su;
          }
          if (qo) cpu_quant_q8(gs, qo + (size_t)sl * o->bstr, o->n / 32);
        }
        break;
      case OP_MOE_SUM: {
        const float *wt = cc + o->nh;
        for (u32 j = 0; j < o->n; j++) {
          float acc = 0.0f;
          for (u32 sl = 0; sl < o->nh; sl++) acc = fmaf(wt[sl], a[(size_t)sl * o->n + j], acc);
          bb[j] = res ? res[j] + acc : acc;
        }
        break;
      }
      case OP_LOGIT_CAP:
        for (u32 j = 0; j < o->nh; j++) bb[j] = o->clamp > 0.0f ? o->clamp * tanhf(a[j] / o->clamp) : a[j];
        for (u32 j = o->nh; j < o->n; j++) bb[j] = -1e30f;
        break;
      case OP_ARGMAX: {
        u32 best = 0;
        for (u32 j = 1; j < o->n; j++)
          if (a[j] > a[best]) best = j;
        *(u32 *)bb = best;
        break;
      }
    }
    /* IE_DUMP_OPS=1: the first floats of each op's output buffer, to compare
     * against a reference implementation op by op (debug only). */
    if (B->dump) {
      /* the op's own output: OP_ADD writes c, OP_GATE rewrites a, the rest write b */
      const int to_c = o->kind == OP_ADD, to_a = o->kind == OP_GATE;
      const int ob = to_c ? o->c : to_a ? o->a : (o->b >= 0 ? o->b : o->a);
      const uint32_t oo = to_c ? o->coff : to_a ? o->aoff : (o->b >= 0 ? o->boff : o->aoff);
      const float *ov = ob >= 0 ? (const float *)((char *)BUF(B, ob) + oo) : NULL;
      double ss = 0.0;
      /* a fixed window: the GPU dump must print the same floats to be comparable */
      u32 nn = o->n > 0 ? o->n : 0;
      if (nn > 1024u) nn = 1024u;
      if (ob >= 0 && B->g->bufs[ob].size > oo && nn > (B->g->bufs[ob].size - oo) / 4u) nn = (B->g->bufs[ob].size - oo) / 4u;
      for (u32 j = 0; ov && j < nn; j++) ss += (double)ov[j] * ov[j];
      fprintf(stderr, "op %3u kind %3u layer %2d n %5u |out| %.6g [", i, o->kind, o->layer, nn, sqrt(ss));
      for (u32 j = 0; ov && j < nn && j < 4; j++) fprintf(stderr, " %.6g", ov[j]);
      fprintf(stderr, "]");
      /* fingerprint of the fused Q8 copy: same bytes on both backends means the
       * GEMV dot is the only place they can differ */
      if (o->qo >= 0 && o->n % 32u == 0u) {
        const u32 nb = o->n / 32u, *as = (const u32 *)((const uint8_t *)BUF(B, o->qo) + 36u * nb);
        const uint8_t *qp = (const uint8_t *)BUF(B, o->qo);
        u32 h = 2166136261u; double ds = 0.0; unsigned long long sa = 0;
        for (u32 j = 0; j < 32u * nb; j++) h = (h ^ qp[j]) * 16777619u;
        for (u32 bb2 = 0; bb2 < nb; bb2++) { float d2; memcpy(&d2, qp + 32u * nb + 4u * bb2, 4); ds += d2; sa += as[bb2]; }
        fprintf(stderr, " q8 %u %08x %.9g %llu", nb, h, ds, sa);
      }
      fprintf(stderr, "\n");
    }
  }
  if (logits) memcpy(logits, BUF(B, g->logits), m->vocab * 4);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  if (ms) *ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
  return *(u32 *)BUF(B, g->argmax);
}

