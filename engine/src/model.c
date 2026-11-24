/* model.c -- load Llama / Qwen2 from GGUF, repack Q4_0 into the verified GPU
 * layout, build the op list of one decode step. */
#include "model.h"

#include <math.h>
#include <string.h>

#include "../core/ie_core.h"
#include "../core/ie_ext.h"
#include "common.h"
#include "iq3s_grid.h"

/* Raw GGUF IQ/K blocks (MAT_RAWQ): the float dequantizers are host-only. */
#define IE_RAWQ_HOST
#define IE_RQP const uint8_t
#define IE_RQF const float
#define IE_RQU const uint32_t
#include "../kernels/ie_iqraw.inc"

/* Bytes of the raw block array of a MAT_RAWQ matrix. */
uint64_t ie_rawq_bytes(const mat *w) {
  return (uint64_t)w->rows * (w->cols / 256u) * ie_rq_bs(w->sub);
}
/* Bytes of one 256-weight block of a ggml type (0 if not a MAT_RAWQ kind). */
uint32_t ie_rawq_bs(uint32_t type) { return ie_rq_bs(type); }

/* ---------------------------------------------------------------- loading */

static const gguf_tensor *need_tensor(gguf_file *g, const char *name) {
  const gguf_tensor *t = gguf_tensor_find(g, name);
  if (!t) ie_die("missing tensor %s", name);
  if (!t->data) ie_die("tensor %s: unsupported type %u", name, t->type);
  return t;
}

/* The 6-bit scale and min of sub-block j (< 8) of a Q4_K / Q5_K block (as
 * ggml's get_scale_min_k4). */
static void q4k_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
  if (j < 4) {
    *sc = q[j] & 63, *m = q[j + 4] & 63;
  } else {
    *sc = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

/* One 256-weight block of IQ4_XS (136 bytes) or IQ3_S (110 bytes) to float
 * (as ggml's dequantize_row_iq4_xs / _iq3_s). */
static void dequant_iq_block(uint32_t type, const uint8_t *blk, float *y) {
  uint16_t h;
  memcpy(&h, blk, 2);
  const float d = ie_f16_to_f32(h);
  if (type == GGML_IQ4_XS) {
    uint16_t sh;
    memcpy(&sh, blk + 2, 2);
    const uint8_t *sl = blk + 4, *qs = blk + 8;
    for (int ib = 0; ib < 8; ib++) {
      const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
      const float dl = d * (float)(ls - 32);
      for (int j = 0; j < 16; j++) {
        y[32 * ib + j] = dl * (float)(int8_t)ie_kv_byte(qs[16 * ib + j] & 15u);
        y[32 * ib + j + 16] = dl * (float)(int8_t)ie_kv_byte(qs[16 * ib + j] >> 4);
      }
    }
    return;
  }
  /* IQ3_S: d, qs[64], qh[8], signs[32], scales[4] */
  const uint8_t *qs = blk + 2, *qh = blk + 66, *signs = blk + 74, *scales = blk + 106;
  for (int ib32 = 0; ib32 < 8; ib32++) {
    const float db = d * (float)(1 + 2 * ((scales[ib32 / 2] >> (4 * (ib32 % 2))) & 0xF));
    for (int l = 0; l < 4; l++) {
      const uint8_t *g1 = (const uint8_t *)(iq3s_grid + (qs[2 * l] | ((qh[ib32] << (8 - 2 * l)) & 256)));
      const uint8_t *g2 = (const uint8_t *)(iq3s_grid + (qs[2 * l + 1] | ((qh[ib32] << (7 - 2 * l)) & 256)));
      for (int j = 0; j < 4; j++) {
        y[j] = db * (float)g1[j] * ((signs[l] >> j) & 1 ? -1.0f : 1.0f);
        y[j + 4] = db * (float)g2[j] * ((signs[l] >> (j + 4)) & 1 ? -1.0f : 1.0f);
      }
      y += 8;
    }
    qs += 8, signs += 4;
  }
}

/* Dequantize n elements of tensor data (F32 / F16 / BF16 / Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K / IQ4_XS / IQ3_S) to float. */
static void dequant_raw(uint32_t type, const uint8_t *p, const char *name, uint64_t first, uint64_t n, float *out) {
  switch (type) {
    case GGML_F32: memcpy(out, p + first * 4, n * 4); break;
    case GGML_F16:
      for (uint64_t i = 0; i < n; i++) {
        uint16_t h;
        memcpy(&h, p + (first + i) * 2, 2);
        out[i] = ie_f16_to_f32(h);
      }
      break;
    case GGML_BF16: /* the high 16 bits of an f32: exact */
      for (uint64_t i = 0; i < n; i++) {
        uint16_t h;
        memcpy(&h, p + (first + i) * 2, 2);
        const uint32_t u = (uint32_t)h << 16;
        memcpy(&out[i], &u, 4);
      }
      break;
    case GGML_Q8_0:
      for (uint64_t i = 0; i < n; i++) {
        uint64_t e = first + i;
        const uint8_t *blk = p + e / 32 * 34;
        uint16_t h;
        memcpy(&h, blk, 2);
        out[i] = ie_f16_to_f32(h) * (float)(int8_t)blk[2 + e % 32];
      }
      break;
    case GGML_Q4_0:
      for (uint64_t i = 0; i < n; i++) {
        uint64_t e = first + i;
        const uint8_t *blk = p + e / 32 * 18;
        uint16_t h;
        memcpy(&h, blk, 2);
        uint32_t j = (uint32_t)(e % 32), q = j < 16 ? (blk[2 + j] & 15u) : (blk[2 + j - 16] >> 4);
        out[i] = ie_f16_to_f32(h) * (float)((int)q - 8);
      }
      break;
    case GGML_IQ4_NL:
      for (uint64_t i = 0; i < n; i++) {
        uint64_t e = first + i;
        const uint8_t *blk = p + e / 32 * 18;
        uint16_t h;
        memcpy(&h, blk, 2);
        uint32_t j = (uint32_t)(e % 32), q = j < 16 ? (blk[2 + j] & 15u) : (blk[2 + j - 16] >> 4);
        out[i] = ie_f16_to_f32(h) * (float)(int)(int8_t)ie_kv_byte(q);
      }
      break;
    case GGML_Q6_K: {
      /* blocks of 256: ql[128], qh[64], int8 scales[16], f16 d (as ggml's
       * dequantize_row_q6_K) */
      const uint64_t b0 = first / 256, b1 = (first + n + 255) / 256;
      float tmp[256];
      for (uint64_t bk = b0; bk < b1; bk++) {
        const uint8_t *blk = p + bk * 210, *ql = blk, *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        uint16_t h;
        memcpy(&h, blk + 208, 2);
        const float d = ie_f16_to_f32(h);
        for (int half = 0; half < 2; half++) {
          const uint8_t *L = ql + 64 * half, *Hh = qh + 32 * half;
          const int8_t *S = sc + 8 * half;
          float *y = tmp + 128 * half;
          for (int l = 0; l < 32; l++) {
            const int is = l / 16;
            const int q1 = (int)((L[l] & 0xF) | (((Hh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((L[l + 32] & 0xF) | (((Hh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((L[l] >> 4) | (((Hh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((L[l + 32] >> 4) | (((Hh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * S[is] * q1;
            y[l + 32] = d * S[is + 2] * q2;
            y[l + 64] = d * S[is + 4] * q3;
            y[l + 96] = d * S[is + 6] * q4;
          }
        }
        for (int j = 0; j < 256; j++) {
          const uint64_t e = bk * 256 + j;
          if (e >= first && e < first + n) out[e - first] = tmp[j];
        }
      }
      break;
    }
    case GGML_Q4_K:
    case GGML_Q5_K: {
      const int ttype = (int)type;
      /* blocks of 256 (as ggml's dequantize_row_q4_K / _q5_K): f16 d, f16
       * dmin, 12 bytes of 6-bit scales and mins, (Q5_K: qh[32]), qs[128] */
      const int five = ttype == GGML_Q5_K;
      const uint64_t bs = five ? 176 : 144, b0 = first / 256, b1 = (first + n + 255) / 256;
      float tmp[256];
      for (uint64_t bk = b0; bk < b1; bk++) {
        const uint8_t *blk = p + bk * bs, *sc = blk + 4, *qh = blk + 16, *q = blk + (five ? 48 : 16);
        uint16_t h0, h1;
        memcpy(&h0, blk, 2), memcpy(&h1, blk + 2, 2);
        const float d = ie_f16_to_f32(h0), dmin = ie_f16_to_f32(h1);
        for (int j = 0, is = 0; j < 256; j += 64, is += 2, q += 32) {
          uint8_t s0, m0, s1, m1;
          q4k_scale_min(is, sc, &s0, &m0), q4k_scale_min(is + 1, sc, &s1, &m1);
          const float d1 = d * s0, mm1 = dmin * m0, d2 = d * s1, mm2 = dmin * m1;
          for (int l = 0; l < 32; l++) {
            const int h1b = five ? ((qh[l] >> (j / 32)) & 1) << 4 : 0, h2b = five ? ((qh[l] >> (j / 32 + 1)) & 1) << 4 : 0;
            tmp[j + l] = d1 * (float)((q[l] & 0xF) + h1b) - mm1;
            tmp[j + 32 + l] = d2 * (float)((q[l] >> 4) + h2b) - mm2;
          }
        }
        for (int j = 0; j < 256; j++) {
          const uint64_t e = bk * 256 + j;
          if (e >= first && e < first + n) out[e - first] = tmp[j];
        }
      }
      break;
    }
    case GGML_IQ4_XS:
    case GGML_IQ3_S: {
      const uint64_t bs = type == GGML_IQ4_XS ? 136 : 110, b0 = first / 256, b1 = (first + n + 255) / 256;
      float tmp[256];
      for (uint64_t bk = b0; bk < b1; bk++) {
        dequant_iq_block(type, p + bk * bs, tmp);
        for (int j = 0; j < 256; j++) {
          const uint64_t e = bk * 256 + j;
          if (e >= first && e < first + n) out[e - first] = tmp[j];
        }
      }
      break;
    }
    case GGML_Q2_K:
    case GGML_IQ2_XXS:
    case GGML_IQ2_XS:
    case GGML_IQ3_XXS:
    case GGML_IQ1_S:
    case GGML_IQ2_S:
    case GGML_IQ1_M: {
      const uint32_t bs = ie_rq_bs(type);
      const uint64_t b0 = first / 256, b1 = (first + n + 255) / 256;
      float tmp[256];
      for (uint64_t bk = b0; bk < b1; bk++) {
        ie_rq_dequant(type, p + bk * bs, tmp);
        for (int j = 0; j < 256; j++) {
          const uint64_t e = bk * 256 + j;
          if (e >= first && e < first + n) out[e - first] = tmp[j];
        }
      }
      break;
    }
    default: ie_die("tensor %s: type %s is not supported here", name, ggml_type_name(type));
  }
}
static void dequant(const gguf_tensor *t, uint64_t first, uint64_t n, float *out) { dequant_raw(t->type, t->data, t->name, first, n, out); }

/* ---- the tensor registry (ie-lab; model.h) ---- */
ie_tref *ie_trefs;
uint32_t ie_n_trefs;
static uint32_t cap_trefs;
int ie_mtp_keep;

static void tref_add(const char *name, mat *w, vec *v, const gguf_tensor *t, uint32_t rows, uint32_t cols, int q8_ok, int emb,
                     int q4q8) {
  if (ie_n_trefs == cap_trefs) {
    cap_trefs = cap_trefs ? 2 * cap_trefs : 512;
    ie_trefs = realloc(ie_trefs, cap_trefs * sizeof *ie_trefs);
    if (!ie_trefs) ie_die("out of memory");
  }
  ie_tref *r = &ie_trefs[ie_n_trefs++];
  memset(r, 0, sizeof *r);
  snprintf(r->name, sizeof r->name, "%s", name);
  r->w = w, r->v = v, r->src = t, r->rows = rows, r->cols = cols, r->q8_ok = q8_ok, r->emb = emb, r->q4q8 = q4q8;
}

/* The rows of matrix from now are rows r0 .. of to (concat_mats, or a struct copy: r0 = 0). */
static void tref_moved(const mat *from, mat *to, uint32_t r0) {
  for (uint32_t i = 0; i < ie_n_trefs; i++)
    if (ie_trefs[i].w == from) ie_trefs[i].w = to, ie_trefs[i].r0 += r0;
}

const ie_tref *ie_tref_find(const char *name) {
  for (uint32_t i = 0; i < ie_n_trefs; i++)
    if (!strcmp(ie_trefs[i].name, name)) return &ie_trefs[i];
  return NULL;
}

uint64_t ie_tref_bytes(const ie_tref *r) { return r->src->nbytes; }

void ie_tref_set_vec(const ie_tref *r, const uint8_t *data) {
  gguf_tensor t = *r->src;
  t.data = data;
  dequant(&t, 0, r->cols, r->v->f);
}

static void load_vec(vec *v, gguf_file *g, const char *name, uint32_t n, int optional) {
  const gguf_tensor *t = gguf_tensor_find(g, name);
  memset(v, 0, sizeof *v);
  if (!t && optional) return;
  t = need_tensor(g, name);
  if (t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] != n) ie_die("tensor %s: expected %u elements", name, n);
  v->n = n;
  v->f = ie_alloc((size_t)n * 4);
  dequant(t, 0, n, v->f);
  tref_add(name, NULL, v, t, 1, n, 0, 0, 0);
}

/* Repack a GGUF Q4_0 matrix into the verified GPU layout. The index maps
 * are those of core/ie_core.h (ie_q4_src_*, ie_q4_dst_*, ie_pack4); the
 * laws q4_addr / q4_inverse / q4_onto / q4_nibble prove that this writes
 * every word once, reads inside the source, and keeps each nibble. */
void repack_q4(mat *w, const uint8_t *src) {
  uint32_t rows = w->rows, nb = w->nb;
  if (!ie_sizes_ok(rows, nb)) ie_die("Q4_0 matrix %ux%u: outside ie_sizes_ok (rows < 2^18, blocks < 512)", rows, w->cols);
  w->qw = ie_alloc((size_t)rows * nb * 16);
  w->qs = ie_alloc((size_t)rows * nb * 2);
  for (u32 r = 0; r < rows; r++)
    for (u32 b = 0; b < nb; b++) {
      u32 s = ie_q4_src_blk(r, b, nb);
      w->qs[ie_q4_dst_scale(r, b, nb)] = (uint16_t)(src[s] | (src[s + 1] << 8));
      for (u32 k = 0; k < 4; k++)
        w->qw[ie_q4_dst_word(r, b, k, nb)] =
            ie_pack4(src[ie_q4_src_qbyte(r, b, 4 * k + 0, nb)], src[ie_q4_src_qbyte(r, b, 4 * k + 1, nb)],
                     src[ie_q4_src_qbyte(r, b, 4 * k + 2, nb)], src[ie_q4_src_qbyte(r, b, 4 * k + 3, nb)]);
    }
}

/* Repack a GGUF Q8_0 matrix into the verified GPU layout (ie_q8_src_*,
 * ie_q8_dst_*, ie_pack4); the laws q8_addr / q8_inverse / q8_onto / q8_byte
 * prove that this writes every word once, reads inside the source, and keeps
 * each int8 weight. */
void repack_q8(mat *w, const uint8_t *src) {
  uint32_t rows = w->rows, nb = w->nb;
  if (!ie_q8_sizes_ok(rows, nb)) ie_die("Q8_0 matrix %ux%u: outside ie_q8_sizes_ok (rows < 2^18, blocks < 2048, rows * blocks < 2^26)", rows, w->cols);
  w->qw = ie_alloc((size_t)rows * nb * 32);
  w->qs = ie_alloc((size_t)rows * nb * 2);
  for (u32 r = 0; r < rows; r++)
    for (u32 b = 0; b < nb; b++) {
      u32 s = ie_q8_src_blk(r, b, nb);
      w->qs[ie_q8_dst_scale(r, b, nb)] = (uint16_t)(src[s] | (src[s + 1] << 8));
      for (u32 k = 0; k < 8; k++)
        w->qw[ie_q8_dst_word(r, b, k, nb)] =
            ie_pack4(src[ie_q8_src_qbyte(r, b, 4 * k + 0, nb)], src[ie_q8_src_qbyte(r, b, 4 * k + 1, nb)],
                     src[ie_q8_src_qbyte(r, b, 4 * k + 2, nb)], src[ie_q8_src_qbyte(r, b, 4 * k + 3, nb)]);
    }
}

/* Repack a GGUF Q4_K matrix into MAT_Q4K (model.h). Sub-block 2k of a
 * 64-weight group holds the low nibbles of its 32 bytes, sub-block 2k + 1
 * the high nibbles; each sub-block is stored as a Q4_0 block (byte j: low
 * nibble = weight j, high nibble = weight j + 16), so the word maps and the
 * nibble order are those of MAT_Q4. */
static void repack_q4k(mat *w, const uint8_t *src) {
  const uint32_t rows = w->rows, nb = w->nb, ns = nb / 8;
  w->qw = ie_alloc((size_t)rows * nb * 16);
  w->qs = ie_alloc((size_t)ie_q4k_qs_n(rows, nb) * 2);
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t sb = 0; sb < ns; sb++) {
      const uint8_t *blk = src + ie_q4k_src_blk(r, sb * 8u, nb); /* the super-block header: d, dmin, scales */
      memcpy(&w->qs[ie_q4k_dd(rows, r, sb * 8u, nb)], blk, 2);
      memcpy(&w->qs[ie_q4k_dd(rows, r, sb * 8u, nb) + 1u], blk + 2, 2);
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t b = sb * 8 + k;
        uint8_t sc, mn;
        q4k_scale_min((int)k, blk + 4, &sc, &mn);
        w->qs[ie_q4k_sm(r, b, nb)] = (uint16_t)(sc | (mn << 8));
        /* byte j of the Q4_0-order block: source bytes j and j + 16 (law q4k_nibble, q4k_addr) */
        uint32_t by[16];
        for (uint32_t j = 0; j < 16; j++)
          by[j] = ie_q4k_pack(src[ie_q4k_src_qbyte(r, b, j, nb)], src[ie_q4k_src_qbyte(r, b, j + 16u, nb)], b);
        for (uint32_t wd = 0; wd < 4; wd++) w->qw[ie_q4_dst_word(r, b, wd, nb)] = ie_pack4(by[4 * wd], by[4 * wd + 1], by[4 * wd + 2], by[4 * wd + 3]);
      }
    }
}

/* Repack a GGUF Q6_K matrix into MAT_Q6K (model.h; index maps and the
 * packing: core/ie_core.h, laws q6k_addr, q6k_unpack). */
static void repack_q6k(mat *w, const uint8_t *src) {
  const uint32_t rows = w->rows, nb = w->nb;
  w->qw = ie_alloc((size_t)ie_q6k_qw_n(rows, nb) * 4);
  w->qs = ie_alloc((size_t)ie_q6k_qs_n(rows, nb) * 2);
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t b = 0; b < nb; b++) {
      uint32_t q[32];
      for (uint32_t j = 0; j < 32; j++) q[j] = ie_q6k_src_q(src[ie_q6k_src_ql(r, b, j, nb)], src[ie_q6k_src_qh(r, b, j, nb)], b);
      for (uint32_t wd = 0; wd < 4; wd++) {
        uint32_t by[4];
        for (uint32_t i = 0; i < 4; i++) by[i] = (q[4 * wd + i] & 15u) | ((q[4 * wd + i + 16] & 15u) << 4);
        w->qw[ie_q4_dst_word(r, b, wd, nb)] = ie_pack4(by[0], by[1], by[2], by[3]);
      }
      for (uint32_t h = 0; h < 2; h++) {
        const uint32_t *p = q + 16 * h;
        w->qw[ie_q6k_hw(rows, r, b, h, nb)] =
            ie_q6k_hpack(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
      }
      w->qs[ie_q6k_sc(r, b, nb)] = (uint16_t)(src[ie_q6k_src_sc(r, b, 0, nb)] | (src[ie_q6k_src_sc(r, b, 1, nb)] << 8));
      if (b % 8 == 0) memcpy(&w->qs[ie_q6k_d(rows, r, b, nb)], src + ie_q6k_src_d(r, b, nb), 2);
    }
}

/* Repack a GGUF Q5_K matrix into MAT_Q5K (core/ie_ext.h): the low nibbles
 * and the scales as MAT_Q4K, bit 4 of each weight in one word per sub-block. */
static void repack_q5k(mat *w, const uint8_t *src) {
  const uint32_t rows = w->rows, nb = w->nb, ns = nb / 8;
  w->qw = ie_alloc((size_t)ie_q5k_qw_n(rows, nb) * 4);
  w->qs = ie_alloc((size_t)ie_q4k_qs_n(rows, nb) * 2);
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t sb = 0; sb < ns; sb++) {
      const uint8_t *blk = src + ((size_t)r * ns + sb) * 176, *qh = blk + 16;
      memcpy(&w->qs[ie_q4k_dd(rows, r, sb * 8u, nb)], blk, 2);
      memcpy(&w->qs[ie_q4k_dd(rows, r, sb * 8u, nb) + 1u], blk + 2, 2);
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t b = sb * 8 + k;
        const uint8_t *q = blk + 48 + 32 * (k / 2);
        uint8_t sc, mn;
        q4k_scale_min((int)k, blk + 4, &sc, &mn);
        w->qs[ie_q4k_sm(r, b, nb)] = (uint16_t)(sc | (mn << 8));
        uint32_t v[32], hw = 0;
        for (uint32_t l = 0; l < 32; l++) {
          v[l] = k % 2 ? (uint32_t)(q[l] >> 4) : q[l] & 15u;
          hw |= (uint32_t)((qh[l] >> k) & 1u) << l;
        }
        for (uint32_t wd = 0; wd < 4; wd++)
          w->qw[ie_q4_dst_word(r, b, wd, nb)] = ie_pack4(v[4 * wd] | v[4 * wd + 16] << 4, v[4 * wd + 1] | v[4 * wd + 17] << 4,
                                                         v[4 * wd + 2] | v[4 * wd + 18] << 4, v[4 * wd + 3] | v[4 * wd + 19] << 4);
        w->qw[ie_q5k_hw(rows, r, b, nb)] = hw;
      }
    }
}

/* Repack a GGUF IQ4_XS matrix into MAT_IQ4XS (core/ie_ext.h). */
static void repack_iq4xs(mat *w, const uint8_t *src) {
  const uint32_t rows = w->rows, nb = w->nb, ns = nb / 8;
  w->qw = ie_alloc((size_t)rows * nb * 16);
  w->qs = ie_alloc((size_t)ie_q4k_qs_n(rows, nb) * 2);
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t sb = 0; sb < ns; sb++) {
      const uint8_t *blk = src + ((size_t)r * ns + sb) * 136, *sl = blk + 4;
      uint16_t sh;
      memcpy(&sh, blk + 2, 2);
      memcpy(&w->qs[ie_q4k_dd(rows, r, sb * 8u, nb)], blk, 2);
      w->qs[ie_q4k_dd(rows, r, sb * 8u, nb) + 1u] = 0;
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t b = sb * 8 + k;
        w->qs[ie_q4k_sm(r, b, nb)] = (uint16_t)(((sl[k / 2] >> (4 * (k % 2))) & 0xF) | (((sh >> (2 * k)) & 3) << 4));
        const uint8_t *q = blk + 8 + 16 * k;
        for (uint32_t wd = 0; wd < 4; wd++) w->qw[ie_q4_dst_word(r, b, wd, nb)] = ie_pack4(q[4 * wd], q[4 * wd + 1], q[4 * wd + 2], q[4 * wd + 3]);
      }
    }
}

uint64_t mat_qw_words(int kind, uint32_t rows, uint32_t cols) {
  const uint64_t nb = cols / 32;
  switch (kind) {
    case MAT_Q4: case MAT_IQ4NL: case MAT_Q4K: case MAT_IQ4XS: return rows * nb * 4;
    case MAT_Q8: return rows * nb * 8;
    case MAT_Q5K: return rows * nb * 5;
    case MAT_Q6K: return rows * nb * 6;
    case MAT_RAWQ: return 0; /* the raw bytes live in qs (ie_rawq_bytes) */
    default: return 0;
  }
}
uint64_t mat_qs_n(int kind, uint32_t rows, uint32_t cols) {
  const uint64_t nb = cols / 32;
  switch (kind) {
    case MAT_Q4: case MAT_IQ4NL: case MAT_Q8: return rows * nb;
    case MAT_Q4K: case MAT_Q5K: case MAT_IQ4XS: return rows * nb + rows * (nb / 8) * 2;
    case MAT_Q6K: return rows * nb + rows * (nb / 8);
    case MAT_RAWQ: return 0; /* raw bytes: ie_rawq_bytes */
    default: return 0;
  }
}
mat mat_expert(const mat *w, uint32_t e) {
  mat v = *w;
  v.n_exp = 1;
  if (w->kind == MAT_F32) {
    v.f = w->f + (size_t)e * w->rows * w->cols;
  } else if (w->kind == MAT_RAWQ) {
    v.qw = NULL;
    v.qs = (uint16_t *)((uint8_t *)w->qs + (size_t)e * ie_rawq_bytes(w));
  } else {
    v.qw = w->qw + (size_t)e * mat_qw_words(w->kind, w->rows, w->cols);
    v.qs = w->qs + (size_t)e * mat_qs_n(w->kind, w->rows, w->cols);
  }
  return v;
}

/* GGUF Q8_0 bytes of n floats (n % 32 == 0), as ggml's quantize_row_q8_0:
 * d = amax / 127 (stored as f16), q = round(x / d). */
static void quant_q8_0_bytes(const float *x, uint64_t n, uint8_t *out) {
  for (uint64_t b = 0; b < n / 32; b++) {
    float amax = 0.0f;
    for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[32 * b + j]));
    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    const uint16_t h = ie_f32_to_f16(d);
    memcpy(out + 34 * b, &h, 2);
    for (int j = 0; j < 32; j++) out[34 * b + 2 + j] = (uint8_t)(int8_t)roundf(x[32 * b + j] * id);
  }
}

/* qwen35: Q4_0 matrices of the main layers are loaded as Q8_0 (the Q8_0
 * kernels have the multi-token paths). The MTP layer keeps Q4_0 (MAT_Q4,
 * half the bytes): it runs one token at a time (drafts, and the last token
 * of a verify), and its results only choose the drafts, not the output. */
static int q4_as_q8;
/* Q5_K as MAT_Q5K (1), or requantized to Q8_0 at load (0: the models before
 * flint keep their measured path; IE_Q5K_NATIVE=0/1 overrides) */
static int native_q5k;

/* Make matrix w of rows x cols from the GGUF bytes src of type ttype. A
 * Q8_0 matrix keeps its integer form (MAT_Q8) when q8_ok, else it becomes
 * f32; emb: 1 for the token embedding (no K or IQ kinds: OP_EMBED reads Q8,
 * Q4 or f32), 2 for MoE experts (repacked kinds + MAT_RAWQ). */
static void load_mat_src(mat *w, uint32_t ttype, const uint8_t *data, const char *name, uint32_t rows, uint32_t cols,
                         uint64_t *bytes, int q8_ok, int emb) {
  memset(w, 0, sizeof *w);
  w->rows = rows;
  w->cols = cols;
  const int kq = cols % 256 == 0 && ie_q4k_sizes_ok(rows, cols / 32) && emb != 1;
  if (ttype == GGML_Q4_0 && ie_sizes_ok(rows, cols / 32) && !q4_as_q8) {
    w->kind = MAT_Q4;
    w->nb = cols / 32;
    repack_q4(w, data);
    *bytes += (uint64_t)rows * w->nb * 18;
  } else if (ttype == GGML_IQ4_NL && ie_sizes_ok(rows, cols / 32) && !q4_as_q8) {
    w->kind = MAT_IQ4NL;
    w->nb = cols / 32;
    repack_q4(w, data);
    *bytes += (uint64_t)rows * w->nb * 18;
  } else if (ttype == GGML_IQ4_NL && q8_ok && ie_q8_sizes_ok(rows, cols / 32)) {
    w->kind = MAT_Q8;
    w->nb = cols / 32;
    uint8_t *q8 = ie_alloc((size_t)rows * w->nb * 34);
    for (size_t k = 0; k < (size_t)rows * w->nb; k++) {
      memcpy(q8 + 34 * k, data + 18 * k, 2);
      for (int j = 0; j < 16; j++) {
        q8[34 * k + 2 + j] = (uint8_t)ie_kv_byte(data[18 * k + 2 + j] & 15u);
        q8[34 * k + 18 + j] = (uint8_t)ie_kv_byte(data[18 * k + 2 + j] >> 4);
      }
    }
    repack_q8(w, q8);
    free(q8);
    *bytes += (uint64_t)rows * w->nb * 34;
  } else if (ttype == GGML_Q4_0 && q8_ok && ie_q8_sizes_ok(rows, cols / 32)) {
    /* Q4_0 past the Q4 size limit (K > 16352), or q4_as_q8 (qwen35: the
     * Q8_0 kernels have the multi-token paths): the same weights as Q8_0,
     * exactly (int8 q - 8, the same scale) */
    w->kind = MAT_Q8;
    w->nb = cols / 32;
    const uint8_t *src = data;
    uint8_t *q8 = ie_alloc((size_t)rows * w->nb * 34);
    for (size_t k = 0; k < (size_t)rows * w->nb; k++) {
      memcpy(q8 + 34 * k, src + 18 * k, 2);
      for (int j = 0; j < 16; j++) {
        q8[34 * k + 2 + j] = (uint8_t)(int8_t)((src[18 * k + 2 + j] & 15) - 8);
        q8[34 * k + 18 + j] = (uint8_t)(int8_t)((src[18 * k + 2 + j] >> 4) - 8);
      }
    }
    repack_q8(w, q8);
    free(q8);
    *bytes += (uint64_t)rows * w->nb * 34;
  } else if (ttype == GGML_Q4_K && kq) {
    w->kind = MAT_Q4K;
    w->nb = cols / 32;
    repack_q4k(w, data);
    *bytes += (uint64_t)rows * w->nb * 18;
  } else if (ttype == GGML_Q5_K && kq && native_q5k) {
    w->kind = MAT_Q5K;
    w->nb = cols / 32;
    repack_q5k(w, data);
    *bytes += (uint64_t)rows * w->nb * 22;
  } else if (ttype == GGML_IQ4_XS && kq) {
    w->kind = MAT_IQ4XS;
    w->nb = cols / 32;
    repack_iq4xs(w, data);
    *bytes += (uint64_t)rows * w->nb * 17;
  } else if (ie_rq_bs(ttype) != 0u && kq && emb != 1 && !(ttype == GGML_IQ3_S && getenv("IE_IQ3S_Q8"))) {
    /* IE_IQ3S_Q8=1: keep the old load-time expansion of IQ3_S to Q8_0, so the
     * native dot can be A/B'd against the path it replaced without a rebuild. */
    /* Q2_K and the IQ2/IQ1/IQ3_XXS blocks: kept raw (MAT_RAWQ), the kernels
     * read the GGUF blocks directly (kernels/ie_iqraw.inc) */
    const uint64_t n = (uint64_t)rows * (cols / 256u) * ie_rq_bs(ttype);
    if (n > 0xFFFFFFFFull) ie_die("%s: raw block array too large (%llu bytes)", name, (unsigned long long)n);
    w->kind = MAT_RAWQ;
    w->sub = ttype;
    w->nb = cols / 32;
    w->qs = ie_alloc((size_t)n);
    memcpy(w->qs, data, (size_t)n);
    *bytes += n;
  } else if (ie_rq_bs(ttype) != 0u && q8_ok && ie_q8_sizes_ok(rows, cols / 32)) {
    /* the same blocks where the raw path does not apply (an embedding):
     * dequantize and requantize to Q8_0 */
    w->kind = MAT_Q8;
    w->nb = cols / 32;
    float *row = ie_alloc((size_t)cols * 4);
    uint8_t *q8 = ie_alloc((size_t)rows * w->nb * 34);
    for (uint32_t r = 0; r < rows; r++) {
      dequant_raw(ttype, data, name, (uint64_t)r * cols, cols, row);
      quant_q8_0_bytes(row, cols, q8 + (size_t)r * w->nb * 34);
    }
    repack_q8(w, q8);
    free(row), free(q8);
    *bytes += (uint64_t)rows * w->nb * 34;
  } else if (ttype == GGML_Q6_K && kq && !getenv("IE_Q6K_AS_Q8")) {
    w->kind = MAT_Q6K;
    w->nb = cols / 32;
    repack_q6k(w, data);
    *bytes += (uint64_t)rows * w->nb * 210 / 8;
  } else if ((ttype == GGML_Q6_K || ttype == GGML_Q5_K || ttype == GGML_Q4_K || ttype == GGML_IQ3_S || ttype == GGML_IQ4_XS) && q8_ok &&
             ie_q8_sizes_ok(rows, cols / 32)) {
    /* Q6_K, Q5_K (not native), IQ3_S (and a K or IQ embedding): no integer
     * kernel; requantize to Q8_0 at load (an approximation: Q8_0 has one
     * scale per 32 weights, Q6_K one per 16) */
    w->kind = MAT_Q8;
    w->nb = cols / 32;
    float *row = ie_alloc((size_t)cols * 4);
    uint8_t *q8 = ie_alloc((size_t)rows * w->nb * 34);
    for (uint32_t r = 0; r < rows; r++) {
      dequant_raw(ttype, data, name, (uint64_t)r * cols, cols, row);
      quant_q8_0_bytes(row, cols, q8 + (size_t)r * w->nb * 34);
    }
    repack_q8(w, q8);
    free(row), free(q8);
    *bytes += (uint64_t)rows * w->nb * 34;
  } else if (ttype == GGML_Q8_0 && q8_ok && ie_q8_sizes_ok(rows, cols / 32)) {
    w->kind = MAT_Q8;
    w->nb = cols / 32;
    repack_q8(w, data);
    *bytes += (uint64_t)rows * w->nb * 34;
  } else {
    w->kind = MAT_F32;
    w->f = ie_alloc((size_t)rows * cols * 4);
    dequant_raw(ttype, data, name, 0, (uint64_t)rows * cols, w->f);
    *bytes += (uint64_t)rows * cols * 4;
  }
}

void model_mat_from_gguf(mat *w, uint32_t ttype, const uint8_t *src, uint32_t rows, uint32_t cols) {
  uint64_t b = 0;
  const int n5 = native_q5k;
  native_q5k = 1;
  load_mat_src(w, ttype, src, "test", rows, cols, &b, 1, 0);
  native_q5k = n5;
}

/* Load matrix `name` of rows x cols (GGUF ne = [cols, rows]). */
static void load_mat(mat *w, gguf_file *g, const char *name, uint32_t rows, uint32_t cols, uint64_t *bytes, int q8_ok) {
  const gguf_tensor *t = need_tensor(g, name);
  if (t->ne[0] != cols || t->ne[1] != rows || t->ne[2] != 1 || t->ne[3] != 1)
    ie_die("tensor %s: shape [%llu, %llu], expected [%u, %u]", name, (unsigned long long)t->ne[0],
           (unsigned long long)t->ne[1], cols, rows);
  const int emb = strcmp(name, "token_embd.weight") == 0;
  load_mat_src(w, t->type, t->data, name, rows, cols, bytes, q8_ok, emb);
  tref_add(name, w, NULL, t, rows, cols, q8_ok, emb, q4_as_q8);
}

typedef struct {
  const gguf_file *file;
  double alpha;
  uint64_t matched;
} lora_load;

static void load_lora_mat(mat *w, gguf_file *g, const char *name, uint32_t rows, uint32_t cols,
                          uint64_t *bytes, lora_load *lora) {
  load_mat(w, g, name, rows, cols, bytes, 1);
  if (!lora->file) return;
  char key[128];
  snprintf(key, sizeof key, "%s.lora_a", name);
  const gguf_tensor *a = gguf_tensor_find(lora->file, key);
  snprintf(key, sizeof key, "%s.lora_b", name);
  const gguf_tensor *b = gguf_tensor_find(lora->file, key);
  if (!a && !b) return;
  if (!a || !b || !a->data || !b->data || a->n_dims != 2 || b->n_dims != 2 ||
      (a->type != GGML_F16 && a->type != GGML_F32) || (b->type != GGML_F16 && b->type != GGML_F32) ||
      a->ne[0] != cols || !a->ne[1] || a->ne[1] > cols || a->ne[1] != b->ne[0] || b->ne[1] != rows)
    ie_die("LoRA: incomplete pair, unsupported type or incompatible shape for %s", name);
  const uint32_t rank = (uint32_t)a->ne[1];
  const float scale = (float)(lora->alpha / rank);
  if (!isfinite(scale) || scale <= 0) ie_die("LoRA: invalid scale for %s", name);
  w->lora = ie_alloc(sizeof *w->lora);
  load_mat_src(&w->lora->a, a->type, a->data, a->name, rank, cols, bytes, 0, 0);
  load_mat_src(&w->lora->b, b->type, b->data, b->name, rows, rank, bytes, 0, 0);
  for (size_t i = 0; i < (size_t)rows * rank; i++) w->lora->b.f[i] *= scale;
  lora->matched += 2;
}

int ie_tref_decode(const ie_tref *r, const uint8_t *data, mat *part) {
  uint64_t b = 0;
  const int keep = q4_as_q8;
  q4_as_q8 = r->q4q8;
  load_mat_src(part, r->src->type, data, r->name, r->rows, r->cols, &b, r->q8_ok, r->emb);
  q4_as_q8 = keep;
  return part->kind == r->w->kind && part->sub == r->w->sub && part->nb == r->w->nb && part->cols == r->w->cols;
}

/* Load the expert tensors names[0 .. n) (GGUF ne = [cols, rows[i], n_exp])
 * as one expert matrix: expert e = [part 0 of e; part 1 of e; ...] (rows:
 * the sum). Returns 0 (and loads nothing) if the parts' types differ.
 * bytes: += the bytes of one expert. */
static int load_experts(mat *w, gguf_file *g, const char *const *names, const uint32_t *rows, int n, uint32_t n_exp, uint32_t cols,
                        uint64_t *bytes) {
  const gguf_tensor *t[4];
  uint32_t R = 0;
  for (int i = 0; i < n; i++) {
    t[i] = need_tensor(g, names[i]);
    if (t[i]->ne[0] != cols || t[i]->ne[1] != rows[i] || t[i]->ne[2] != n_exp || t[i]->ne[3] != 1)
      ie_die("tensor %s: shape [%llu, %llu, %llu], expected [%u, %u, %u]", names[i], (unsigned long long)t[i]->ne[0],
             (unsigned long long)t[i]->ne[1], (unsigned long long)t[i]->ne[2], cols, rows[i], n_exp);
    if (t[i]->type != t[0]->type) return 0;
    R += rows[i];
  }
  const uint64_t rb = ggml_nbytes(t[0]->type, cols);
  uint8_t *src = n > 1 ? ie_alloc((size_t)R * rb) : NULL;
  memset(w, 0, sizeof *w);
  uint64_t qwn = 0, qsn = 0, rawb = 0;
  for (uint32_t e = 0; e < n_exp; e++) {
    const uint8_t *se = (const uint8_t *)t[0]->data + (size_t)e * rows[0] * rb;
    if (n > 1) {
      size_t o = 0;
      for (int i = 0; i < n; i++) {
        memcpy(src + o, (const uint8_t *)t[i]->data + (size_t)e * rows[i] * rb, (size_t)rows[i] * rb);
        o += (size_t)rows[i] * rb;
      }
      se = src;
    }
    mat x;
    uint64_t xb = 0;
    load_mat_src(&x, t[0]->type, se, names[0], R, cols, &xb, 1, 2);
    if (e == 0) {
      *w = x;
      w->n_exp = n_exp;
      *bytes += xb;
      if (x.kind == MAT_F32) {
        w->f = ie_alloc((size_t)n_exp * R * cols * 4);
      } else if (x.kind == MAT_RAWQ) {
        rawb = ie_rawq_bytes(&x);
        w->qw = NULL;
        w->qs = ie_alloc((size_t)n_exp * rawb);
      } else {
        qwn = mat_qw_words(x.kind, R, cols), qsn = mat_qs_n(x.kind, R, cols);
        w->qw = ie_alloc((size_t)n_exp * qwn * 4);
        w->qs = ie_alloc((size_t)n_exp * qsn * 2);
      }
    }
    if (x.kind != w->kind || (x.kind == MAT_RAWQ && x.sub != w->sub)) ie_die("tensor %s: experts of different kinds", names[0]);
    if (x.kind == MAT_F32) {
      memcpy(w->f + (size_t)e * R * cols, x.f, (size_t)R * cols * 4);
    } else if (x.kind == MAT_RAWQ) {
      memcpy((uint8_t *)w->qs + (size_t)e * rawb, x.qs, (size_t)rawb);
    } else {
      memcpy(w->qw + (size_t)e * qwn, x.qw, qwn * 4);
      memcpy(w->qs + (size_t)e * qsn, x.qs, qsn * 2);
    }
    free(x.qw), free(x.qs), free(x.f);
  }
  free(src);
  return 1;
}

/* Concatenate the rows of n matrices of one kind and width into dst. The
 * GPU layouts are row major (ie_q4_dst_word / ie_q8_dst_word are
 * (r * nb + b) * W + w), so row r of part i becomes row off_i + r and the
 * index maps stay those of the verified core. The parts are freed. Returns
 * 0 (and changes nothing) if the kinds or widths differ or the sizes are
 * outside the verified limits. */
static int concat_mats(mat *dst, mat *const *src, int n) {
  uint32_t rows = 0;
  for (int i = 0; i < n; i++) {
    if (src[i]->lora) return 0; /* each projection has its own low-rank update */
    if (src[i]->kind != src[0]->kind || src[i]->cols != src[0]->cols) return 0;
    rows += src[i]->rows;
  }
  const int kind = src[0]->kind;
  const uint32_t cols = src[0]->cols, nb = src[0]->nb;
  if ((kind == MAT_Q4 || kind == MAT_IQ4NL) && !ie_sizes_ok(rows, nb)) return 0;
  if (kind == MAT_Q6K) { /* two sections in qw and in qs: concatenate each */
    if (!ie_q4k_sizes_ok(rows, nb)) return 0;
    memset(dst, 0, sizeof *dst);
    dst->kind = kind, dst->rows = rows, dst->cols = src[0]->cols, dst->nb = nb;
    dst->qw = ie_alloc((size_t)ie_q6k_qw_n(rows, nb) * 4);
    dst->qs = ie_alloc((size_t)ie_q6k_qs_n(rows, nb) * 2);
    size_t r0 = 0;
    for (int i = 0; i < n; i++) {
      mat *w = src[i];
      memcpy(dst->qw + ie_q4_dst_word((uint32_t)r0, 0, 0, nb), w->qw, (size_t)w->rows * nb * 16);
      memcpy(dst->qw + ie_q6k_hw(rows, (uint32_t)r0, 0, 0, nb), w->qw + ie_q6k_hw(w->rows, 0, 0, 0, nb), (size_t)w->rows * nb * 8);
      memcpy(dst->qs + ie_q6k_sc((uint32_t)r0, 0, nb), w->qs, (size_t)w->rows * nb * 2);
      memcpy(dst->qs + ie_q6k_d(rows, (uint32_t)r0, 0, nb), w->qs + ie_q6k_d(w->rows, 0, 0, nb), (size_t)w->rows * (nb / 8) * 2);
      tref_moved(w, dst, (uint32_t)r0);
      r0 += w->rows;
      free(w->qw), free(w->qs);
      w->qw = NULL, w->qs = NULL;
    }
    return 1;
  }
  if (kind == MAT_Q4K || kind == MAT_IQ4XS || kind == MAT_Q5K) { /* two sections in qs (Q5K: also in qw): concatenate each */
    if (!ie_q4k_sizes_ok(rows, nb)) return 0;
    memset(dst, 0, sizeof *dst);
    dst->kind = kind, dst->rows = rows, dst->cols = src[0]->cols, dst->nb = nb;
    dst->qw = ie_alloc((size_t)mat_qw_words(kind, rows, src[0]->cols) * 4);
    dst->qs = ie_alloc((size_t)ie_q4k_qs_n(rows, nb) * 2);
    size_t r0 = 0;
    for (int i = 0; i < n; i++) {
      mat *w = src[i];
      memcpy(dst->qw + r0 * nb * 4, w->qw, (size_t)w->rows * nb * 16);
      if (kind == MAT_Q5K) memcpy(dst->qw + ie_q5k_hw(rows, (uint32_t)r0, 0, nb), w->qw + ie_q5k_hw(w->rows, 0, 0, nb), (size_t)w->rows * nb * 4);
      memcpy(dst->qs + ie_q4k_sm((uint32_t)r0, 0, nb), w->qs, (size_t)w->rows * nb * 2);
      memcpy(dst->qs + ie_q4k_dd(rows, (uint32_t)r0, 0, nb), w->qs + ie_q4k_dd(w->rows, 0, 0, nb), (size_t)w->rows * (nb / 8) * 4);
      tref_moved(w, dst, (uint32_t)r0);
      r0 += w->rows;
      free(w->qw), free(w->qs);
      w->qw = NULL, w->qs = NULL;
    }
    return 1;
  }
  if (kind == MAT_RAWQ) {
    const uint32_t sub = src[0]->sub, bs = ie_rawq_bs(sub), mb = cols / 256u;
    for (int i = 1; i < n; i++)
      if (src[i]->sub != sub) return 0;
    const uint64_t tot = (uint64_t)rows * mb * bs;
    if (!bs || cols % 256u != 0u || !ie_q4k_sizes_ok(rows, nb) || tot > 0xFFFFFFFFull) return 0;
    memset(dst, 0, sizeof *dst);
    dst->kind = MAT_RAWQ, dst->sub = sub, dst->rows = rows, dst->cols = cols, dst->nb = nb;
    dst->qs = ie_alloc((size_t)tot);
    size_t r0 = 0;
    for (int i = 0; i < n; i++) {
      mat *w = src[i];
      memcpy((uint8_t *)dst->qs + r0 * mb * bs, w->qs, (size_t)w->rows * mb * bs);
      tref_moved(w, dst, (uint32_t)r0);
      r0 += w->rows;
      free(w->qs);
      w->qs = NULL;
    }
    return 1;
  }
  if (kind != MAT_Q4 && kind != MAT_IQ4NL && kind != MAT_Q8 && kind != MAT_F32) return 0; /* a layout this function does not know */
  if (kind == MAT_Q8 && !ie_q8_sizes_ok(rows, nb)) return 0;
  memset(dst, 0, sizeof *dst);
  dst->kind = kind, dst->rows = rows, dst->cols = cols, dst->nb = nb;
  const size_t words = (kind == MAT_Q4 || kind == MAT_IQ4NL) ? 4 : 8;
  if (kind == MAT_F32) {
    dst->f = ie_alloc((size_t)rows * cols * 4);
  } else {
    dst->qw = ie_alloc((size_t)rows * nb * words * 4);
    dst->qs = ie_alloc((size_t)rows * nb * 2);
  }
  size_t r0 = 0;
  for (int i = 0; i < n; i++) {
    mat *w = src[i];
    if (kind == MAT_F32) {
      memcpy(dst->f + r0 * cols, w->f, (size_t)w->rows * cols * 4);
    } else {
      memcpy(dst->qw + r0 * nb * words, w->qw, (size_t)w->rows * nb * words * 4);
      memcpy(dst->qs + r0 * nb, w->qs, (size_t)w->rows * nb * 2);
    }
    tref_moved(w, dst, (uint32_t)r0);
    r0 += w->rows;
    free(w->qw), free(w->qs), free(w->f);
    w->qw = NULL, w->qs = NULL, w->f = NULL;
  }
  return 1;
}

/* The bias of [q; k; v]: the parts, or zeros for a missing part. n = 0 if
 * no part has a bias. */
static void concat_bias(vec *dst, const vec *const *src, const uint32_t *len, int n) {
  int any = 0;
  uint32_t tot = 0;
  for (int i = 0; i < n; i++) any |= src[i]->n != 0, tot += len[i];
  memset(dst, 0, sizeof *dst);
  if (!any) return;
  dst->n = tot;
  dst->f = calloc(tot, 4);
  if (!dst->f) ie_die("out of memory");
  for (uint32_t i = 0, o = 0; i < (uint32_t)n; o += len[i], i++)
    if (src[i]->n) memcpy(dst->f + o, src[i]->f, len[i] * 4);
}

static void free_mat(mat *w);
void (*ie_mat_sink)(mat *w);

/* Give the finished matrices of a layer to the sink (merged parts have no
 * host arrays left and are skipped by the sink). */
/* IE_HOST_LAYERS=N: the last N blocks keep their weights in pinned host memory and the
 * GEMV reads them over PCIe on every token. That is the deliberate version of what the
 * OS does by itself when a model overflows the VRAM budget, and it is far cheaper: the
 * OS evicts whole allocations and Ripple-65B ends up reading ~6 GB/token, while N blocks
 * read exactly N * (per-layer bytes). 0 or unset: everything in VRAM. */
static int host_layers(void) {
  static int n = -1;
  if (n < 0) {
    const char *e = getenv("IE_HOST_LAYERS");
    n = (e && e[0]) ? atoi(e) : 0;
    if (n < 0) n = 0;
  }
  return n;
}

/* IE_HOST_SET=a,b,c-d: exactly those blocks are host layers (instead of the last IE_HOST_LAYERS). With the GPU
 * backend's IE_HOST_STREAM=1 they are copied into VRAM by DMA while the model runs (otherwise read over PCIe), so the choice decides
 * how the copies overlap the work: blocks spread over the model hide their copies best. */
static int host_set(uint32_t i, int *given) {
  static uint8_t set[1024];
  static int init = 0, any = 0;
  if (!init) {
    init = 1;
    const char *s = getenv("IE_HOST_SET");
    any = s && s[0];
    while (s && *s) {
      char *e;
      long a = strtol(s, &e, 10), b = a;
      if (e == s) break;
      if (*e == '-') b = strtol(e + 1, &e, 10);
      for (long k = a; k <= b && k < 1024; k++)
        if (k >= 0) set[k] = 1;
      s = *e == ',' ? e + 1 : e;
    }
  }
  *given = any;
  return i < 1024u && set[i];
}

static int ink_moe_cache_env(void) {
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("IE_MOE_CACHE_SLOTS");
    v = (e && *e && (strcmp(e, "auto") == 0 || strcmp(e, "AUTO") == 0 || atoi(e) > 0)) ? 1 : 0;
  }
  return v;
}

static void sink_layer(const model *m, layer *L, uint32_t i) {
  if (!ie_mat_sink) return;
  mat *all[] = {&L->wq, &L->wk, &L->wv, &L->wqkv, &L->wo, &L->wgate, &L->wup, &L->wdown, &L->wgu,
                &L->wlqkv, &L->wz, &L->wlz, &L->wbeta, &L->walpha, &L->win, &L->wagt,
                &L->wrel, &L->wsink, &L->k_conv_proj, &L->v_conv_proj,
                &L->wr, &L->egu, &L->eg, &L->eu, &L->ed,
                &L->loop[0].down, &L->loop[0].gain, &L->loop[0].forget, &L->loop[0].wfg,
                &L->loop[1].down, &L->loop[1].gain, &L->loop[1].forget, &L->loop[1].wfg,
                &L->loop[2].down, &L->loop[2].gain, &L->loop[2].forget, &L->loop[2].wfg,
                &L->loop[3].down, &L->loop[3].gain, &L->loop[3].forget, &L->loop[3].wfg};
  const int nl = host_layers();
  int given;
  const int inset = host_set(i, &given);
  const int host = given ? inset : nl > 0 && i + (uint32_t)nl >= m->n_layer;
  const int moe_cache = (m->arch == ARCH_INKLING2 && ink_moe_cache_env() &&
                         (i < m->loop_start || i >= m->loop_end));
  for (unsigned k = 0; k < sizeof all / sizeof all[0]; k++) {
    if (host || (moe_cache && (all[k] == &L->egu || all[k] == &L->ed))) all[k]->host = 1;
    ie_mat_sink(all[k]);
    if (all[k]->lora) {
      all[k]->lora->a.host = all[k]->lora->b.host = host;
      ie_mat_sink(&all[k]->lora->a);
      ie_mat_sink(&all[k]->lora->b);
    }
  }
}

static uint32_t opt_u32(gguf_file *g, const char *arch, const char *key, uint32_t def) {
  char k[256];
  int64_t v;
  snprintf(k, sizeof k, "%s.%s", arch, key);
  if (!gguf_get_int(g, k, &v)) return def;
  if (v < 0 || v > 0x7FFFFFFF) ie_die("bad metadata %s", k);
  return (uint32_t)v;
}

static uint32_t need_u32(gguf_file *g, const char *arch, const char *key) {
  char k[256];
  int64_t v;
  snprintf(k, sizeof k, "%s.%s", arch, key);
  if (!gguf_get_int(g, k, &v) || v <= 0 || v > 0x7FFFFFFF) ie_die("missing or bad metadata %s", k);
  return (uint32_t)v;
}

static uint32_t need_u32_first(gguf_file *g, const char *arch, const char *key) {
  char k[256];
  int64_t v;
  snprintf(k, sizeof k, "%s.%s", arch, key);
  if (gguf_get_int(g, k, &v) && v > 0 && v <= 0x7FFFFFFF) return (uint32_t)v;
  const gguf_kv *kv = gguf_find(g, k);
  if (kv && kv->type == GGUF_ARR && kv->arr_n > 0) {
    if (kv->arr_type == GGUF_U32 || kv->arr_type == GGUF_I32) {
      uint32_t u = 0;
      memcpy(&u, kv->arr_data, 4);
      if (kv->arr_type == GGUF_U32 && u > 0 && u <= 0x7FFFFFFFu) return u;
      if (kv->arr_type == GGUF_I32 && (int32_t)u > 0) return u;
    }
  }
  ie_die("missing or bad metadata %s", k);
  return 0;
}

static float opt_f32(gguf_file *g, const char *arch, const char *key, float def) {
  char k[256];
  double d;
  snprintf(k, sizeof k, "%s.%s", arch, key);
  return gguf_get_float(g, k, &d) ? (float)d : def;
}

static int has_tensor2(const gguf_file *g, const char *a, const char *b) {
  return gguf_tensor_find(g, a) != NULL || (b && gguf_tensor_find(g, b) != NULL);
}

static void load_vec_alias(vec *v, gguf_file *g, const char *a, const char *b, uint32_t n, int optional) {
  if (gguf_tensor_find(g, a) || !b || !gguf_tensor_find(g, b)) load_vec(v, g, a, n, optional);
  else load_vec(v, g, b, n, optional);
}

static void load_mat_alias(mat *w, gguf_file *g, const char *a, const char *b, uint32_t rows, uint32_t cols, uint64_t *bytes, int q8_ok) {
  if (gguf_tensor_find(g, a) || !b || !gguf_tensor_find(g, b)) load_mat(w, g, a, rows, cols, bytes, q8_ok);
  else load_mat(w, g, b, rows, cols, bytes, q8_ok);
}

static int ink_phys(const model *m, uint32_t il, uint32_t *ip_out) {
  const uint32_t cs = m->loop_start, ce = m->loop_end, nl = m->n_loops;
  const uint32_t cl = ce > cs ? ce - cs : 0u;
  if (nl > 1u && cl > 0u) {
    const uint32_t loop_tot = nl * cl;
    if (il >= cs && il < cs + loop_tot) {
      const uint32_t off = il - cs;
      *ip_out = cs + off % cl;
      return (int)(off / cl);
    }
    if (il >= cs + loop_tot) {
      *ip_out = il - (nl - 1u) * cl;
      return -1;
    }
  }
  *ip_out = il;
  return -1;
}

/* A flint layer: attention (QK norm, sinks; window win, RoPE unless
 * norope), then a dense FFN or (moe) the router, the experts and the shared
 * expert. */
static void flint_layer(model *m, layer *L, gguf_file *g, uint32_t i, uint32_t win, int norope, int moe) {
  const uint32_t qd = m->n_head * m->hd, kvd = m->n_kv * m->hd;
  char n[128], n2[128];
#define NAME(s) (snprintf(n, sizeof n, "blk.%u." s, i), n)
  L->swin = win, L->norope = norope, L->moe = moe;
  L->kvi = m->n_kvl++;
  load_vec(&L->attn_norm, g, NAME("attn_norm.weight"), m->dim, 0);
  load_vec(&L->ffn_norm, g, NAME("ffn_norm.weight"), m->dim, 0);
  load_vec(&L->q_norm, g, NAME("attn_q_norm.weight"), m->hd, 0);
  load_vec(&L->k_norm, g, NAME("attn_k_norm.weight"), m->hd, 0);
  load_vec(&L->sinks, g, NAME("attn_sinks.weight"), m->n_head, 1);
  load_mat(&L->wq, g, NAME("attn_q.weight"), qd, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wk, g, NAME("attn_k.weight"), kvd, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wv, g, NAME("attn_v.weight"), kvd, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wo, g, NAME("attn_output.weight"), m->dim, qd, &m->weight_bytes, 1);
  mat *qkv[3] = {&L->wq, &L->wk, &L->wv};
  concat_mats(&L->wqkv, qkv, 3);
  if (!moe) {
    load_mat(&L->wgate, g, NAME("ffn_gate.weight"), m->ffn, m->dim, &m->weight_bytes, 1);
    load_mat(&L->wup, g, NAME("ffn_up.weight"), m->ffn, m->dim, &m->weight_bytes, 1);
    load_mat(&L->wdown, g, NAME("ffn_down.weight"), m->dim, m->ffn, &m->weight_bytes, 1);
  } else {
    load_mat(&L->wr, g, NAME("ffn_gate_inp.weight"), m->n_exp, m->dim, &m->weight_bytes, 0);
    load_vec(&L->ebias, g, NAME("exp_probs_b.bias"), m->n_exp, 1);
    snprintf(n2, sizeof n2, "blk.%u.ffn_up_exps.weight", i);
    const char *gu[2] = {NAME("ffn_gate_exps.weight"), n2};
    const uint32_t r2[2] = {m->eff, m->eff};
    uint64_t eb = 0; /* the bytes of one expert: n_used of them per token */
    if (!load_experts(&L->egu, g, gu, r2, 2, m->n_exp, m->dim, &eb)) {
      load_experts(&L->eg, g, gu, r2, 1, m->n_exp, m->dim, &eb);
      load_experts(&L->eu, g, gu + 1, r2, 1, m->n_exp, m->dim, &eb);
    }
    const char *dn[1] = {NAME("ffn_down_exps.weight")};
    const uint32_t rd[1] = {m->dim};
    load_experts(&L->ed, g, dn, rd, 1, m->n_exp, m->eff, &eb);
    m->weight_bytes += eb * m->n_used;
    if (m->sff) {
      load_mat(&L->wgate, g, NAME("ffn_gate_shexp.weight"), m->sff, m->dim, &m->weight_bytes, 1);
      load_mat(&L->wup, g, NAME("ffn_up_shexp.weight"), m->sff, m->dim, &m->weight_bytes, 1);
      load_mat(&L->wdown, g, NAME("ffn_down_shexp.weight"), m->dim, m->sff, &m->weight_bytes, 1);
    }
  }
#undef NAME
  if (L->wgate.rows) {
    mat *gu2[2] = {&L->wgate, &L->wup};
    concat_mats(&L->wgu, gu2, 2);
  }
  sink_layer(m, L, i);
}

/* Ripple (ricursive): PeriNorm block with a value residual and a gated
 * attention output. Sliding-window layers rotate q and k (RoPE, NeoX pairs);
 * global layers do not (norope). The value residual mixes the scaled layer-0
 * embeddings (0) into the normed attention input, so v_proj reads a different
 * vector than q_proj and k_proj; the gate is a full matrix on that same normed
 * input. */
static void ricursive_layer(model *m, layer *L, gguf_file *g, uint32_t i, uint32_t win, int norope) {
  const uint32_t qd = m->n_head * m->hd, kvd = m->n_kv * m->hd;
  char n[128];
#define NAME(s) (snprintf(n, sizeof n, "blk.%u." s, i), n)
  L->swin = win, L->norope = norope;
  L->kvi = m->n_kvl++;
  L->fclamp = m->swiglu_clamp;
  load_vec(&L->attn_norm, g, NAME("attn_norm.weight"), m->dim, 0); /* input_layernorm */
  load_vec(&L->ffn_norm, g, NAME("ffn_norm.weight"), m->dim, 0);   /* pre_feedforward_layernorm */
  load_vec(&L->pan, g, NAME("post_attention_norm.weight"), m->dim, 0);
  load_vec(&L->pffn, g, NAME("post_ffw_norm.weight"), m->dim, 0);
  load_vec(&L->q_norm, g, NAME("attn_q_norm.weight"), m->hd, 0);
  load_vec(&L->k_norm, g, NAME("attn_k_norm.weight"), m->hd, 0);
  {
    vec lam;
    load_vec(&lam, g, NAME("attn_vres_lambda.weight"), 1, 0);
    L->lam = lam.f[0];
    free(lam.f);
  }
  load_mat(&L->wq, g, NAME("attn_q.weight"), qd, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wk, g, NAME("attn_k.weight"), kvd, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wv, g, NAME("attn_v.weight"), kvd, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wo, g, NAME("attn_output.weight"), m->dim, qd, &m->weight_bytes, 1);
  load_mat(&L->wagt, g, NAME("attn_gate.weight"), qd, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wgate, g, NAME("ffn_gate.weight"), m->ffn, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wup, g, NAME("ffn_up.weight"), m->ffn, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wdown, g, NAME("ffn_down.weight"), m->dim, m->ffn, &m->weight_bytes, 1);
#undef NAME
  {
    mat *gu[2] = {&L->wgate, &L->wup};
    concat_mats(&L->wgu, gu, 2);
  }
  sink_layer(m, L, i);
}

static void inkling_layer(model *m, layer *L, gguf_file *g, uint32_t i, uint32_t sw) {
  const uint32_t qd = m->n_head * m->hd, kvd = m->n_kv * m->hd;
  char n[128], n2[128];
#define NAME(s) (snprintf(n, sizeof n, "blk.%u." s, i), n)
#define NAME2(s) (snprintf(n2, sizeof n2, "blk.%u." s, i), n2)
  const int is_full = has_tensor2(g, NAME("attn_q_inp_norm.weight"), NAME2("attn_q_input_norm.weight"));
  L->is_full = is_full;
  L->swin = is_full ? 0u : sw;
  L->moe = 1;
  if (is_full) {
    load_vec_alias(&L->q_inp_norm, g, NAME("attn_q_inp_norm.weight"), NAME2("attn_q_input_norm.weight"), m->dim, 0);
    load_vec_alias(&L->k_inp_norm, g, NAME("attn_k_inp_norm.weight"), NAME2("attn_k_input_norm.weight"), m->dim, 0);
    load_vec_alias(&L->v_inp_norm, g, NAME("attn_v_inp_norm.weight"), NAME2("attn_v_input_norm.weight"), m->dim, 0);
    load_mat(&L->k_conv_proj, g, NAME("attn_k_conv_proj.weight"), m->n_kv * m->d_sconv, m->dim, &m->weight_bytes, 1);
    load_mat(&L->v_conv_proj, g, NAME("attn_v_conv_proj.weight"), m->n_kv * m->d_sconv, m->dim, &m->weight_bytes, 1);
  } else {
    load_vec(&L->attn_norm, g, NAME("attn_norm.weight"), m->dim, 0);
  }
  load_vec(&L->ffn_norm, g, NAME("ffn_norm.weight"), m->dim, 0);
  load_mat(&L->wq, g, NAME("attn_q.weight"), qd, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wk, g, NAME("attn_k.weight"), kvd, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wv, g, NAME("attn_v.weight"), kvd, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wo, g, NAME("attn_output.weight"), m->dim, qd, &m->weight_bytes, 1);
  load_mat(&L->wagt, g, NAME("attn_gate.weight"), qd, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wrel, g, NAME("attn_rel.weight"), m->n_head * m->d_rel, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wsink, g, NAME("attn_sink_proj.weight"), m->n_head, m->dim, &m->weight_bytes, 1);
  if (!is_full && !(i >= m->loop_start && i < m->loop_end)) {
    mat *qkv[3] = {&L->wq, &L->wk, &L->wv};
    concat_mats(&L->wqkv, qkv, 3);
  }
  load_vec(&L->k_conv, g, NAME("attn_k_conv.weight"), m->d_sconv * kvd, 0);
  load_vec(&L->v_conv, g, NAME("attn_v_conv.weight"), m->d_sconv * kvd, 0);
  load_vec(&L->attn_pconv, g, NAME("attn_post_conv.weight"), m->d_sconv * m->dim, 0);
  load_vec(&L->ffn_pconv, g, NAME("ffn_post_conv.weight"), m->d_sconv * m->dim, 0);
  load_vec(&L->q_clip, g, NAME("attn_q_clip.weight"), qd, 0);
  load_vec(&L->rel_clip, g, NAME("attn_rel_clip.weight"), m->n_head * m->d_rel, 0);
  load_vec(&L->rel_bias, g, NAME("attn_rel_bias.weight"), m->rel_extent * m->d_rel, 0);
  load_vec(&L->attn_res_q, g, NAME("attn_res_query.weight"), is_full ? 3u * m->dim : m->dim, i == 0 ? 1 : 0);
  load_vec(&L->ffn_res_q, g, NAME("ffn_res_query.weight"), m->dim, 0);

  load_mat(&L->wr, g, NAME("ffn_gate_inp.weight"), m->n_exp, m->dim, &m->weight_bytes, 0);
  load_vec(&L->ebias, g, NAME("exp_probs_b.bias"), m->n_exp, 1);
  uint64_t eb = 0;
  if (gguf_tensor_find(g, NAME("ffn_gate_up_exps.weight"))) {
    const char *gu[1] = {n};
    const uint32_t r1[1] = {2u * m->eff};
    load_experts(&L->egu, g, gu, r1, 1, m->n_exp, m->dim, &eb);
  } else {
    snprintf(n, sizeof n, "blk.%u.ffn_gate_exps.weight", i);
    snprintf(n2, sizeof n2, "blk.%u.ffn_up_exps.weight", i);
    const char *gu[2] = {n, n2};
    const uint32_t r2[2] = {m->eff, m->eff};
    if (!load_experts(&L->egu, g, gu, r2, 2, m->n_exp, m->dim, &eb)) {
      load_experts(&L->eg, g, gu, r2, 1, m->n_exp, m->dim, &eb);
      load_experts(&L->eu, g, gu + 1, r2, 1, m->n_exp, m->dim, &eb);
    }
  }
  snprintf(n, sizeof n, "blk.%u.ffn_down_exps.weight", i);
  const char *dn[1] = {n};
  const uint32_t rd[1] = {m->dim};
  load_experts(&L->ed, g, dn, rd, 1, m->n_exp, m->eff, &eb);
  m->weight_bytes += eb * m->n_used;

  if (i >= m->loop_start && i < m->loop_end) {
    static const char *const br[4] = {"attn", "k", "v", "mlp"};
    static const char *const br2[4] = {"q_ctrl", "k_ctrl", "v_ctrl", "ffn_ctrl"};
    for (int b = 0; b < 4; b++) {
      snprintf(n, sizeof n, "blk.%u.loop_%s_down.weight", i, br[b]);
      snprintf(n2, sizeof n2, "blk.%u.loop_%s_down.weight", i, br2[b]);
      load_mat_alias(&L->loop[b].down, g, n, n2, m->loop_rank, 2u * m->dim, &m->weight_bytes, 1);
      snprintf(n, sizeof n, "blk.%u.loop_%s_gain.weight", i, br[b]);
      snprintf(n2, sizeof n2, "blk.%u.loop_%s_gain.weight", i, br2[b]);
      load_mat_alias(&L->loop[b].gain, g, n, n2, m->dim, m->loop_rank, &m->weight_bytes, 1);
      snprintf(n, sizeof n, "blk.%u.loop_%s_forget.weight", i, br[b]);
      snprintf(n2, sizeof n2, "blk.%u.loop_%s_forget.weight", i, br2[b]);
      load_mat_alias(&L->loop[b].forget, g, n, n2, m->dim, m->loop_rank, &m->weight_bytes, 1);
      mat *fg[2] = {&L->loop[b].forget, &L->loop[b].gain};
      concat_mats(&L->loop[b].wfg, fg, 2);
    }
  }
#undef NAME2
#undef NAME
  sink_layer(m, L, i);
}

void model_load(model *m, gguf_file *g) {
  model_load_lora(m, g, NULL);
}

void model_load_lora(model *m, gguf_file *g, const gguf_file *adapter) {
  memset(m, 0, sizeof *m);
  char *arch = gguf_get_str(g, "general.architecture");
  if (!arch) ie_die("missing general.architecture");
  if (strcmp(arch, "llama") == 0) m->arch = ARCH_LLAMA, m->rope = ROPE_NORM;
  else if (strcmp(arch, "qwen2") == 0) m->arch = ARCH_QWEN2, m->rope = ROPE_NEOX;
  else if (strcmp(arch, "qwen3") == 0) m->arch = ARCH_QWEN2, m->rope = ROPE_NEOX, m->qk_norm = 1; /* qwen2 + per head QK RMSNorm */
  else if (strcmp(arch, "qwen35") == 0) m->arch = ARCH_QWEN35, m->rope = ROPE_NEOX;
  else if (strcmp(arch, "flint") == 0) m->arch = ARCH_FLINT, m->rope = ROPE_NEOX;
  else if (strcmp(arch, "ricursive") == 0) m->arch = ARCH_RICURSIVE, m->rope = ROPE_NEOX; /* Ripple: NeoX-style RoPE */
  else if (strcmp(arch, "inkling2") == 0 || strcmp(arch, "inkling2_loop") == 0) m->arch = ARCH_INKLING2, m->rope = ROPE_NEOX;
  else ie_die("architecture %s is not supported (llama, qwen2, qwen3, qwen35, flint, ricursive, inkling2)", arch);
  const int q35 = m->arch == ARCH_QWEN35, fl = m->arch == ARCH_FLINT, ric = m->arch == ARCH_RICURSIVE, ink = m->arch == ARCH_INKLING2;
  lora_load lora = {.file = adapter};
  if (adapter) {
    char *arch_lora = gguf_get_str(adapter, "general.architecture");
    char *type = gguf_get_str(adapter, "general.type");
    char *kind = gguf_get_str(adapter, "adapter.type");
    if (!q35 || !arch_lora || strcmp(arch_lora, arch) || !type || strcmp(type, "adapter") ||
        !kind || strcmp(kind, "lora") || gguf_find(adapter, "adapter.alora.invocation_tokens") ||
        !gguf_get_float(adapter, "adapter.lora.alpha", &lora.alpha) || !isfinite(lora.alpha) || lora.alpha <= 0)
      ie_die("LoRA: expected a qwen35 GGUF LoRA adapter with positive alpha");
    free(arch_lora), free(type), free(kind);
  }
  q4_as_q8 = q35;
  native_q5k = fl || ric; /* ricursive: attn_output and the output matrix are Q5_K */
  if (getenv("IE_Q5K_NATIVE")) native_q5k = getenv("IE_Q5K_NATIVE")[0] == '1';

  m->n_layer = need_u32(g, arch, "block_count");
  /* qwen35: the last nextn_predict_layers blocks are multi-token prediction
   * layers, not used for decoding one token at a time */
  if (q35) m->n_layer -= opt_u32(g, arch, "nextn_predict_layers", 0);
  m->dim = need_u32(g, arch, "embedding_length");
  m->ffn = ink ? opt_u32(g, arch, "feed_forward_length", 0) : need_u32(g, arch, "feed_forward_length");
  m->n_head = ink ? need_u32_first(g, arch, "attention.head_count") : need_u32(g, arch, "attention.head_count");
  m->n_kv = ink ? need_u32_first(g, arch, "attention.head_count_kv") : need_u32(g, arch, "attention.head_count_kv");
  m->ctx_train = need_u32(g, arch, "context_length");
  char k[256];
  double d;
  snprintf(k, sizeof k, "%s.attention.layer_norm_rms_epsilon", arch);
  m->eps = gguf_get_float(g, k, &d) ? (float)d : 1e-5f;
  snprintf(k, sizeof k, "%s.rope.freq_base", arch);
  m->rope_base = gguf_get_float(g, k, &d) ? (float)d : 10000.0f;
  if (m->n_head % m->n_kv) ie_die("bad head counts");
  uint32_t interval = 1;
  if (q35) {
    m->hd = need_u32(g, arch, "attention.key_length");
    if (opt_u32(g, arch, "attention.value_length", m->hd) != m->hd) ie_die("key and value head dims differ");
    m->n_rot = opt_u32(g, arch, "rope.dimension_count", m->hd);
    snprintf(k, sizeof k, "%s.rope.dimension_sections", arch);
    const gguf_kv *ms = gguf_find(g, k);
    if (ms && ms->type == GGUF_ARR && (ms->arr_type == GGUF_I32 || ms->arr_type == GGUF_U32) && ms->arr_n >= 3) {
      for (int j = 0; j < 3; j++) {
        uint32_t v;
        memcpy(&v, ms->arr_data + 4 * j, 4);
        m->mrope_sec[j] = v;
      }
    }
    m->d_conv = need_u32(g, arch, "ssm.conv_kernel");
    m->sd = need_u32(g, arch, "ssm.state_size");
    m->n_kh = need_u32(g, arch, "ssm.group_count");
    m->n_vh = need_u32(g, arch, "ssm.time_step_rank");
    interval = need_u32(g, arch, "full_attention_interval");
    if (need_u32(g, arch, "ssm.inner_size") != m->n_vh * m->sd) ie_die("ssm.inner_size != time_step_rank * state_size");
    /* the kernels (ie_gdn) are written for these sizes */
    if (m->d_conv != 4 || m->sd != 128 || m->n_vh % m->n_kh) ie_die("linear attention: unsupported sizes");
    m->conv_dim = 2 * m->n_kh * m->sd + m->n_vh * m->sd;
  } else if (fl) {
    m->hd = need_u32(g, arch, "attention.key_length");
    if (opt_u32(g, arch, "attention.value_length", m->hd) != m->hd) ie_die("key and value head dims differ");
    m->n_rot = m->hd;
    m->n_exp = need_u32(g, arch, "expert_count");
    m->n_used = need_u32(g, arch, "expert_used_count");
    m->eff = need_u32(g, arch, "expert_feed_forward_length");
    m->sff = opt_u32(g, arch, "expert_shared_feed_forward_length", 0);
    snprintf(k, sizeof k, "%s.expert_weights_scale", arch);
    m->escale = gguf_get_float(g, k, &d) ? (float)d : 1.0f;
    if (opt_u32(g, arch, "expert_gating_func", 2) != 2) ie_die("flint: only sigmoid routing (expert_gating_func 2)");
    snprintf(k, sizeof k, "%s.expert_weights_norm", arch);
    int64_t wn = 1;
    if (gguf_get_int(g, k, &wn) && !wn) ie_die("flint: unnormalized routing weights are not supported");
    if (m->n_used == 0 || m->n_used > 16 || m->n_used > m->n_exp || m->eff % 256 || m->sff % 32) ie_die("flint: unsupported MoE sizes");
  } else if (ric) { /* Ripple: 80 heads of 128, 8 kv heads, full-head RoPE on the sliding layers only */
    m->hd = need_u32(g, arch, "attention.key_length");
    if (opt_u32(g, arch, "attention.value_length", m->hd) != m->hd) ie_die("key and value head dims differ");
    m->n_rot = opt_u32(g, arch, "rope.dimension_count", m->hd);
    snprintf(k, sizeof k, "%s.rope.dimension_sections", arch);
    const gguf_kv *ms = gguf_find(g, k);
    if (ms && ms->type == GGUF_ARR && (ms->arr_type == GGUF_I32 || ms->arr_type == GGUF_U32) && ms->arr_n >= 3) {
      for (int j = 0; j < 3; j++) {
        uint32_t v;
        memcpy(&v, ms->arr_data + 4 * j, 4);
        m->mrope_sec[j] = v;
      }
    }
    if (m->n_rot != m->hd) ie_die("ricursive: partial RoPE is not supported");
  } else if (m->qk_norm) { /* qwen3: the head dim is its own (0.6B: 1024 wide, 16 heads of 128) */
    m->hd = opt_u32(g, arch, "attention.key_length", m->dim / m->n_head);
    if (opt_u32(g, arch, "attention.value_length", m->hd) != m->hd) ie_die("key and value head dims differ");
    m->n_rot = m->hd;
  } else if (ink) {
    m->hd = need_u32(g, arch, "attention.key_length");
    if (opt_u32(g, arch, "attention.value_length", m->hd) != m->hd) ie_die("key and value head dims differ");
    m->n_rot = opt_u32(g, arch, "rope.dimension_count", 0);
    m->d_rel = opt_u32(g, arch, "attention.relative_length", 16);
    m->rel_extent = opt_u32(g, arch, "attention.relative_extent", 1024);
    m->d_sconv = opt_u32(g, arch, "shortconv.l_cache", 4);
    if (m->d_sconv != 4u) ie_die("inkling2: shortconv.l_cache %u != 4", m->d_sconv);
    m->attn_res_bs = opt_u32(g, arch, "attn_res.block_size", 6);
    m->loop_start = opt_u32(g, arch, "loop.core_start", 0);
    m->loop_end = opt_u32(g, arch, "loop.core_end", 0);
    m->n_loops = opt_u32(g, arch, "num_loops", 1);
    m->loop_rank = opt_u32(g, arch, "loop.gate_rank", 64);
    m->n_exp = need_u32(g, arch, "expert_count");
    m->n_used = need_u32(g, arch, "expert_used_count");
    m->eff = need_u32(g, arch, "expert_feed_forward_length");
    m->escale = opt_f32(g, arch, "expert_weights_scale", 8.0f);
    m->qk_clip = opt_f32(g, arch, "attention.qk_clip", 22.627417f);
    m->rel_clip = opt_f32(g, arch, "attention.relative_clip", 8.0f);
    m->log_scale_floor = opt_f32(g, arch, "attention.log_scale_floor", 8192.0f);
    m->log_scale_alpha = opt_f32(g, arch, "attention.log_scale_alpha", 0.1f);
    m->logit_cap = opt_f32(g, arch, "final_logit_softcapping", 30.0f);
    m->situ_beta = opt_f32(g, arch, "activation.situ_beta", 4.0f);
    m->situ_lbeta = opt_f32(g, arch, "activation.situ_linear_beta", 25.0f);
    m->vocab_unpadded = opt_u32(g, arch, "vocab_size_unpadded", 0);
    uint32_t n_phys = 0;
    for (;;) {
      char tn[64];
      snprintf(tn, sizeof tn, "blk.%u.ffn_norm.weight", n_phys);
      if (!gguf_tensor_find(g, tn)) break;
      n_phys++;
    }
    if (!n_phys) ie_die("inkling2: no physical layers found");
    const uint32_t cl = m->loop_end > m->loop_start ? m->loop_end - m->loop_start : 0u;
    const uint32_t visits = n_phys + (m->n_loops > 1u ? m->n_loops - 1u : 0u) * cl;
    if (m->n_layer != n_phys && m->n_layer != visits)
      ie_die("inkling2: block_count %u does not match physical %u / visits %u", m->n_layer, n_phys, visits);
    m->n_layer = n_phys;
    m->n_visits = visits;
    if (m->n_used == 0 || m->n_used > 16 || m->n_used > m->n_exp || m->eff % 32 || (m->loop_rank % 32))
      ie_die("inkling2: unsupported MoE or loop sizes");
  } else {
    if (m->dim % m->n_head) ie_die("bad head counts");
    m->hd = m->dim / m->n_head;
    m->n_rot = m->hd;
  }
  if (m->hd % 2 || m->n_rot % 2 || m->n_rot > m->hd) ie_die("bad head dim");
  if (m->dim % 32 || (m->ffn && m->ffn % 32)) ie_die("dims must be multiples of 32");
  if (ric) {
    /* embed_scale: a non-folded checkpoint carries it (Ripple 65B: 71.55) and
     * applies it to the embeddings before the value-residual seed; 0 when the
     * scale is baked into token_embd. */
    snprintf(k, sizeof k, "%s.embedding_scale", arch);
    m->emb_scale = gguf_get_float(g, k, &d) ? (float)d : 0.0f;
    snprintf(k, sizeof k, "%s.swiglu_clamp", arch);
    m->swiglu_clamp = gguf_get_float(g, k, &d) ? (float)d : 0.0f;
    fprintf(stderr, "ricursive: embed_scale %g, FFN clamp %g, window %u\n", m->emb_scale, m->swiglu_clamp,
            opt_u32(g, arch, "attention.sliding_window", 0));
  }

  const gguf_tensor *te = need_tensor(g, "token_embd.weight");
  m->vocab = (uint32_t)te->ne[1];
  uint64_t wb = 0;
  /* qwen35, flint, ricursive, inkling2: a Q8_0 embedding stays Q8_0 */
  load_mat(&m->tok, g, "token_embd.weight", m->vocab, m->dim, &wb, q35 || fl || ric || ink);
  /* an untied embedding is read one row per token: it can stay in host
   * memory (IE_EMBED_HOST=0: in VRAM); a tied one is also the output matrix */
  const char *eh = getenv("IE_EMBED_HOST");
  /* tied K-quant embedding (Qwen3.5-4B: Q6_K): the lookup copy above is Q8_0
   * (embeddings never take the native K paths), so the output GEMV gets its
   * own native copy of the same tensor: exact weights and 26.25 instead of 34
   * bytes per 32 (-154 MB read per token for the 4B). The Q8 lookup copy then
   * moves to host memory like an untied one. IE_TIED_NATIVE_HEAD=0: share the
   * Q8 copy (the previous behaviour). */
  const char *tn = getenv("IE_TIED_NATIVE_HEAD");
  int tied_native = 0;
  if (!gguf_tensor_find(g, "output.weight") && !(tn && tn[0] == '0') && m->tok.kind == MAT_Q8 &&
      (te->type == GGML_Q6_K || te->type == GGML_Q4_K)) {
    uint64_t ob = 0;
    load_mat_src(&m->out, te->type, te->data, "token_embd.weight(head)", m->vocab, m->dim, &ob, 1, 0);
    if (m->out.kind == MAT_Q6K || m->out.kind == MAT_Q4K) {
      tied_native = 1;
      m->weight_bytes += ob;
      fprintf(stderr, "tied output head: native %s copy (%.0f MB), Q8_0 embedding lookup\n",
              m->out.kind == MAT_Q6K ? "Q6_K" : "Q4_K", ob / 1e6);
    } else {
      free_mat(&m->out);
      memset(&m->out, 0, sizeof m->out);
    }
  }
  m->tok.host = (gguf_tensor_find(g, "output.weight") || tied_native) && !(eh && eh[0] == '0');
  if (ie_mat_sink) ie_mat_sink(&m->tok);
  if (tied_native) {
    if (ie_mat_sink) ie_mat_sink(&m->out);
  } else if (gguf_tensor_find(g, "output.weight")) {
    load_mat(&m->out, g, "output.weight", m->vocab, m->dim, &m->weight_bytes, 1);
    if (ie_mat_sink) ie_mat_sink(&m->out);
  } else {
    m->out = m->tok;
    m->tied = 1;
    m->weight_bytes += wb; /* the output GEMV reads the whole matrix */
  }
  load_vec(&m->out_norm, g, "output_norm.weight", m->dim, 0);
  if (ink) {
    load_vec(&m->tok_norm, g, "token_embd_norm.weight", m->dim, 0);
    load_vec(&m->out_res_q, g, "output_res_query.weight", m->dim, 0);
    if (m->n_loops > 1u && m->loop_end > m->loop_start) {
      const uint32_t nc = m->loop_end - m->loop_start;
      load_vec_alias(&m->loop_res_q, g, "loop_res_query.weight", "loop_msg_query.weight", 2u * nc * m->dim, 0);
      load_vec_alias(&m->loop_kv_res_q, g, "loop_kv_res_query.weight", "loop_kv_msg_query.weight", 2u * nc * m->dim, 0);
    }
    m->ones_dim.n = m->dim;
    m->ones_dim.f = ie_alloc((size_t)m->dim * 4);
    for (uint32_t j = 0; j < m->dim; j++) m->ones_dim.f[j] = 1.0f;
    m->ones_hd.n = m->hd;
    m->ones_hd.f = ie_alloc((size_t)m->hd * 4);
    for (uint32_t j = 0; j < m->hd; j++) m->ones_hd.f[j] = 1.0f;
  }

  uint32_t qd = m->n_head * m->hd, kvd = m->n_kv * m->hd;
  m->l = calloc(m->n_layer + 1, sizeof(layer)); /* + the MTP layer */
  if (!m->l) ie_die("out of memory");
  /* flint: per layer sliding window and RoPE (bool arrays), dense layers first.
   * ricursive: only the sliding window pattern; a global layer has no RoPE. */
  const gguf_kv *swp = NULL, *rpp = NULL;
  uint32_t sw = 0, n_dense = 0;
  if (fl || ric) {
    snprintf(k, sizeof k, "%s.attention.sliding_window_pattern", arch);
    swp = gguf_find(g, k);
    if (!swp || swp->type != GGUF_ARR || swp->arr_type != GGUF_BOOL || swp->arr_n != m->n_layer)
      ie_die("%s: missing or bad attention.sliding_window_pattern", arch);
    sw = need_u32(g, arch, "attention.sliding_window");
  }
  if (ink) sw = opt_u32(g, arch, "attention.sliding_window", 1024);
  if (fl) {
    snprintf(k, sizeof k, "%s.attention.rope_pattern", arch);
    rpp = gguf_find(g, k);
    if (!rpp || rpp->type != GGUF_ARR || rpp->arr_type != GGUF_BOOL || rpp->arr_n != m->n_layer)
      ie_die("flint: missing or bad attention.rope_pattern");
    n_dense = opt_u32(g, arch, "leading_dense_block_count", 0);
  }
  for (uint32_t i = 0; i < m->n_layer; i++) {
    layer *L = &m->l[i];
    char n[128];
#define NAME(s) (snprintf(n, sizeof n, "blk.%u." s, i), n)
    if (ink) {
      inkling_layer(m, L, g, i, sw);
      continue;
    }
    if (fl) {
      flint_layer(m, L, g, i, swp->arr_data[i] ? sw : 0, !rpp->arr_data[i], i >= n_dense);
      continue;
    }
    if (ric) {
      const int slide = swp->arr_data[i] != 0;
      ricursive_layer(m, L, g, i, slide ? sw : 0, !slide);
      continue;
    }
    load_vec(&L->attn_norm, g, NAME("attn_norm.weight"), m->dim, 0);
    load_vec(&L->ffn_norm, g, q35 ? NAME("post_attention_norm.weight") : NAME("ffn_norm.weight"), m->dim, 0);
    load_lora_mat(&L->wgate, g, NAME("ffn_gate.weight"), m->ffn, m->dim, &m->weight_bytes, &lora);
    load_lora_mat(&L->wup, g, NAME("ffn_up.weight"), m->ffn, m->dim, &m->weight_bytes, &lora);
    load_lora_mat(&L->wdown, g, NAME("ffn_down.weight"), m->dim, m->ffn, &m->weight_bytes, &lora);
    mat *gu[2] = {&L->wgate, &L->wup};
    concat_mats(&L->wgu, gu, 2);
    if (q35 && (i + 1) % interval != 0) { /* linear attention (Gated DeltaNet) */
      const uint32_t inner = m->n_vh * m->sd;
      L->rec = 1;
      L->sti = m->n_rec++;
      load_mat(&L->wlqkv, g, NAME("attn_qkv.weight"), m->conv_dim, m->dim, &m->weight_bytes, 1);
      load_mat(&L->wz, g, NAME("attn_gate.weight"), inner, m->dim, &m->weight_bytes, 1);
      load_mat(&L->wbeta, g, NAME("ssm_beta.weight"), m->n_vh, m->dim, &m->weight_bytes, 1);
      load_mat(&L->walpha, g, NAME("ssm_alpha.weight"), m->n_vh, m->dim, &m->weight_bytes, 1);
      mat *in4[4] = {&L->wlqkv, &L->wz, &L->wbeta, &L->walpha};
      if (!concat_mats(&L->win, in4, 4)) { /* kinds differ (Q4_K_M / IQ: beta, alpha differ): merge [wlqkv; wz] and [beta; alpha] where possible */
        mat *lz2[2] = {&L->wlqkv, &L->wz};
        concat_mats(&L->wlz, lz2, 2);
        mat ba, *in2[2] = {&L->wbeta, &L->walpha};
        if (concat_mats(&ba, in2, 2)) {
          L->wbeta = ba; /* walpha: rows 0 (merged into wbeta) */
          tref_moved(&ba, &L->wbeta, 0);
          L->walpha.rows = 0;
        }
      }
      load_vec(&L->conv, g, NAME("ssm_conv1d.weight"), m->conv_dim * m->d_conv, 0);
      load_vec(&L->dt_bias, g, NAME("ssm_dt.bias"), m->n_vh, 0);
      load_vec(&L->ssm_a, g, NAME("ssm_a"), m->n_vh, 0);
      load_vec(&L->ssm_norm, g, NAME("ssm_norm.weight"), m->sd, 0);
      load_mat(&L->wo, g, NAME("ssm_out.weight"), m->dim, inner, &m->weight_bytes, 1);
      sink_layer(m, L, i);
      continue;
    }
    L->kvi = m->n_kvl++;
    if (q35 || m->qk_norm) {
      load_vec(&L->q_norm, g, NAME("attn_q_norm.weight"), m->hd, 0);
      load_vec(&L->k_norm, g, NAME("attn_k_norm.weight"), m->hd, 0);
    }
    load_lora_mat(&L->wq, g, NAME("attn_q.weight"), q35 ? 2 * qd : qd, m->dim, &m->weight_bytes, &lora);
    load_lora_mat(&L->wk, g, NAME("attn_k.weight"), kvd, m->dim, &m->weight_bytes, &lora);
    load_lora_mat(&L->wv, g, NAME("attn_v.weight"), kvd, m->dim, &m->weight_bytes, &lora);
    load_lora_mat(&L->wo, g, NAME("attn_output.weight"), m->dim, qd, &m->weight_bytes, &lora);
    load_vec(&L->bq, g, NAME("attn_q.bias"), qd, 1);
    load_vec(&L->bk, g, NAME("attn_k.bias"), kvd, 1);
    load_vec(&L->bv, g, NAME("attn_v.bias"), kvd, 1);
    mat *qkv[3] = {&L->wq, &L->wk, &L->wv};
    if (concat_mats(&L->wqkv, qkv, 3)) {
      const vec *b3[3] = {&L->bq, &L->bk, &L->bv};
      const uint32_t n3[3] = {L->wq.rows, kvd, kvd};
      concat_bias(&L->bqkv, b3, n3, 3);
    }
    sink_layer(m, L, i);
#undef NAME
  }
  if (ink) {
    m->n_kvl = m->n_visits;
    m->kv_win = ie_alloc((size_t)(m->n_kvl ? m->n_kvl : 1u) * sizeof(uint32_t));
    for (uint32_t il = 0; il < m->n_visits; il++) {
      uint32_t ip;
      ink_phys(m, il, &ip);
      m->kv_win[il] = m->l[ip].swin;
    }
    m->sc_tot = m->n_visits * (2u * kvd + 2u * m->dim);
  }

  if (adapter) {
    if (!lora.matched || lora.matched != adapter->n_tensors)
      ie_die("LoRA: matched %llu of %llu tensors (unsupported, duplicate or missing target)",
             (unsigned long long)lora.matched, (unsigned long long)adapter->n_tensors);
    fprintf(stderr, "LoRA: loaded %llu projection pairs, alpha %g\n", (unsigned long long)(lora.matched / 2), lora.alpha);
  }

  /* tokenizer */
  char *tm = gguf_get_str(g, "tokenizer.ggml.model");
  m->tok_gpt2 = tm && strcmp(tm, "gpt2") == 0;
  free(tm);
  const gguf_kv *tk = gguf_find(g, "tokenizer.ggml.tokens");
  if (tk && tk->type == GGUF_ARR && tk->arr_type == GGUF_STR) {
    m->n_tokens = (uint32_t)tk->arr_n;
    m->tokens = calloc(m->n_tokens ? m->n_tokens : 1, sizeof(char *));
    for (uint32_t i = 0; i < m->n_tokens; i++) {
      gguf_str s = tk->arr_str[i];
      m->tokens[i] = malloc(s.n + 1);
      memcpy(m->tokens[i], s.p, s.n);
      m->tokens[i][s.n] = 0;
    }
  }
  free(arch);
}

static void free_mat(mat *w) {
  if (w->lora) {
    free_mat(&w->lora->a);
    free_mat(&w->lora->b);
    free(w->lora);
    w->lora = NULL;
  }
  free(w->qw);
  free(w->qs);
  free(w->f);
}
void ie_mat_free(mat *w) {
  free_mat(w);
  w->qw = NULL, w->qs = NULL, w->f = NULL;
}

int gguf_has_mtp(const gguf_file *g) {
  for (uint64_t i = 0; i < g->n_tensors; i++) {
    unsigned n;
    char tail[64];
    if (sscanf(g->t[i].name, "blk.%u.%63s", &n, tail) == 2 && !strcmp(tail, "nextn.eh_proj.weight")) return 1;
  }
  return 0;
}

int model_load_mtp(model *m, gguf_file *g) {
  if (m->arch != ARCH_QWEN35) return 0;
  int il = -1;
  for (uint64_t i = 0; i < g->n_tensors; i++) {
    unsigned n;
    char tail[64];
    if (sscanf(g->t[i].name, "blk.%u.%63s", &n, tail) == 2 && !strcmp(tail, "nextn.eh_proj.weight")) il = (int)n;
  }
  if (il < 0) return 0;
  const uint32_t qd = m->n_head * m->hd, kvd = m->n_kv * m->hd;
  layer *L = &m->l[m->n_layer];
  memset(L, 0, sizeof *L);
  char n[128];
#define NAME(s) (snprintf(n, sizeof n, "blk.%d." s, il), n)
  load_vec(&m->enorm, g, NAME("nextn.enorm.weight"), m->dim, 0);
  load_vec(&m->hnorm, g, NAME("nextn.hnorm.weight"), m->dim, 0);
  load_vec(&m->head_norm, g, NAME("nextn.shared_head_norm.weight"), m->dim, 1);
  if (!m->head_norm.n) load_vec(&m->head_norm, g, "output_norm.weight", m->dim, 0);
  uint64_t wb = 0; /* the MTP weights are not in weight_bytes (the bytes of one model step) */
  const int q4q8 = q4_as_q8;
  q4_as_q8 = getenv("IE_MTP_Q8") != NULL; /* IE_MTP_Q8=1: Q4_0 as Q8_0 here too (as before) */
  load_mat(&m->eh, g, NAME("nextn.eh_proj.weight"), m->dim, 2 * m->dim, &wb, 1);
  if (ie_mat_sink) ie_mat_sink(&m->eh);
  load_vec(&L->attn_norm, g, NAME("attn_norm.weight"), m->dim, 0);
  load_vec(&L->ffn_norm, g, NAME("post_attention_norm.weight"), m->dim, 0);
  load_vec(&L->q_norm, g, NAME("attn_q_norm.weight"), m->hd, 0);
  load_vec(&L->k_norm, g, NAME("attn_k_norm.weight"), m->hd, 0);
  load_mat(&L->wq, g, NAME("attn_q.weight"), 2 * qd, m->dim, &wb, 1);
  load_mat(&L->wk, g, NAME("attn_k.weight"), kvd, m->dim, &wb, 1);
  load_mat(&L->wv, g, NAME("attn_v.weight"), kvd, m->dim, &wb, 1);
  load_mat(&L->wo, g, NAME("attn_output.weight"), m->dim, qd, &wb, 1);
  load_mat(&L->wgate, g, NAME("ffn_gate.weight"), m->ffn, m->dim, &wb, 1);
  load_mat(&L->wup, g, NAME("ffn_up.weight"), m->ffn, m->dim, &wb, 1);
  load_mat(&L->wdown, g, NAME("ffn_down.weight"), m->dim, m->ffn, &wb, 1);
#undef NAME
  q4_as_q8 = q4q8;
  mat *qkv[3] = {&L->wq, &L->wk, &L->wv};
  concat_mats(&L->wqkv, qkv, 3);
  mat *gu[2] = {&L->wgate, &L->wup};
  concat_mats(&L->wgu, gu, 2);
  sink_layer(m, L, 0u);   /* the MTP head: a single layer, never a host layer */
  L->kvi = m->n_kvl++;
  const char *me = getenv("IE_MTP_EMBED");
  if (me && me[0] == '1') {
    const gguf_tensor *t = need_tensor(g, "token_embd.weight");
    if (t->ne[0] != m->dim || t->ne[1] != m->vocab) ie_die("IE_MTP_EMBED: the head file's token_embd is not %u x %u", m->vocab, m->dim);
    uint64_t eb = 0;
    load_mat_src(&m->mtp_tok, t->type, t->data, "token_embd.weight(mtp)", m->vocab, m->dim, &eb, 1, 1);
    /* pinned host memory like the model's embedding, or VRAM with it (IE_EMBED_HOST=0) */
    const char *eh = getenv("IE_EMBED_HOST");
    m->mtp_tok.host = !(eh && eh[0] == '0');
    if (ie_mat_sink) ie_mat_sink(&m->mtp_tok);
    fprintf(stderr, "mtp: input embedding from the head file (%.0f MB, %s)\n", eb / 1e6, m->mtp_tok.host ? "host memory" : "VRAM");
  }
  m->has_mtp = 1;
  fprintf(stderr, "mtp: head blk.%d loaded (%.2f MB)\n", il, wb / 1e6);
  return 1;
}

int model_load_dflash(model *m, gguf_file *g) {
  char *arch = gguf_get_str(g, "general.architecture");
  const int ok = arch && !strcmp(arch, "dflash");
  free(arch);
  if (!ok) return 0;
  if (m->arch != ARCH_QWEN35 && !m->qk_norm) ie_die("dflash: the target must be a qwen35 or qwen3 model");
  dflash *d = calloc(1, sizeof *d);
  if (!d) ie_die("out of memory");
  int64_t v;
  double f;
#define GI(k, dst) (gguf_get_int(g, "dflash." k, &v) ? (void)((dst) = (uint32_t)v) : ie_die("dflash: no dflash." k))
  GI("block_count", d->n_layer), GI("embedding_length", d->dim), GI("feed_forward_length", d->ffn);
  GI("attention.head_count", d->n_head), GI("attention.head_count_kv", d->n_kv), GI("attention.key_length", d->hd);
  GI("block_size", d->block);
  /* a full-attention drafter (z-lab Qwen3-8B: sliding_window None) has no window key: the last 4096 positions, the
   * most ie_df_attn holds (DFA_MAX 4112 with a block of 16) */
  d->window = gguf_get_int(g, "dflash.attention.sliding_window", &v) ? (uint32_t)v : 4096u;
  /* DFlash2 has the selector and the convolutions; a v1 drafter has neither */
  d->has_sel = gguf_find(g, "dflash.selector_top_k") != NULL;
  d->has_conv = gguf_find(g, "dflash.conv_kernel_size") != NULL;
  if (d->has_sel) GI("selector_top_k", d->top_k), GI("selector_rank", d->rank);
  else d->top_k = 16, d->rank = 0; /* the top K for pmin (softmax over it) and the argmax */
  if (d->has_conv) GI("conv_kernel_size", d->kernel), GI("conv_group_size", d->group);
  if (gguf_get_int(g, "dflash.attention.value_length", &v) && (uint32_t)v != d->hd) ie_die("dflash: value_length != key_length");
#undef GI
  d->eps = gguf_get_float(g, "dflash.attention.layer_norm_rms_epsilon", &f) ? (float)f : 1e-6f;
  d->rope_base = gguf_get_float(g, "dflash.rope.freq_base", &f) ? (float)f : 1e7f;
  /* config.json dflash_config.mask_token_id: Qwen3.8 DFlash2 248070, z-lab Qwen3.5 DFlash (v1) 248077;
   * the GGUF's own key wins, IE_DFLASH_MASK over all */
  d->mask = d->has_sel ? 248070u : 248077u;
  if (gguf_get_int(g, "dflash.mask_token_id", &v) || gguf_get_int(g, "tokenizer.ggml.mask_token_id", &v)) d->mask = (uint32_t)v;
  if (getenv("IE_DFLASH_MASK")) d->mask = (uint32_t)strtoul(getenv("IE_DFLASH_MASK"), NULL, 10);
  const gguf_kv *tl = gguf_find(g, "dflash.target_layers");
  if (!tl || tl->type != GGUF_ARR || tl->arr_n == 0 || tl->arr_n > 8) ie_die("dflash: no dflash.target_layers");
  d->n_tl = (uint32_t)tl->arr_n;
  for (uint32_t i = 0; i < d->n_tl; i++) { /* the GGUF ids count the embedding as 0: layer id - 1 */
    int32_t x;
    memcpy(&x, tl->arr_data + 4 * (size_t)i, 4);
    if (x < 1 || (uint32_t)x > m->n_layer) ie_die("dflash: target layer %d", x);
    d->tl[i] = (uint32_t)x - 1u;
  }
  if (d->dim != m->dim || (d->has_conv && (d->kernel != 2 || !d->group || d->dim % d->group)) || d->hd % 32 || d->hd > 256 ||
      d->top_k > 64 || (d->has_sel && d->rank % 32) || d->block > 16 || d->window + d->block > 4112u /* ie_df_attn DFA_MAX */)
    ie_die("dflash: unsupported shape");
  const uint32_t qd = d->n_head * d->hd, kvd = d->n_kv * d->hd, cp = d->has_conv ? 2u * d->kernel * (d->dim / d->group) : 0u;
  uint64_t wb = 0;
  load_vec(&d->enc_norm, g, "enc.output_norm.weight", d->dim, 0);
  load_vec(&d->out_norm, g, "output_norm.weight", d->dim, 0);
  load_mat(&d->fc, g, "fc.weight", d->dim, d->n_tl * d->dim, &wb, 1);
  if (ie_mat_sink) ie_mat_sink(&d->fc);
  if (d->has_sel) {
    load_mat(&d->sel_h, g, "selector_hidden.weight", d->rank, d->dim, &wb, 1);
    if (ie_mat_sink) ie_mat_sink(&d->sel_h);
    d->sel_prev = need_tensor(g, "selector_predecessor.weight");
    d->sel_next = need_tensor(g, "selector_successor.weight");
  }
  d->l = calloc(d->n_layer, sizeof *d->l);
  char n[128];
  for (uint32_t il = 0; il < d->n_layer; il++) {
    df_layer *L = &d->l[il];
#define NAME(s) (snprintf(n, sizeof n, "blk.%u." s, il), n)
    load_vec(&L->attn_norm, g, NAME("attn_norm.weight"), d->dim, 0);
    load_vec(&L->ffn_norm, g, NAME("ffn_norm.weight"), d->dim, 0);
    load_vec(&L->q_norm, g, NAME("attn_q_norm.weight"), d->hd, 0);
    load_vec(&L->k_norm, g, NAME("attn_k_norm.weight"), d->hd, 0);
    if (d->has_conv) {
      load_vec(&L->aconv_base, g, NAME("attn_conv_base"), 2 * d->kernel * d->dim, 0);
      load_vec(&L->fconv_base, g, NAME("ffn_conv_base"), 2 * d->kernel * d->dim, 0);
    }
    load_mat(&L->wq, g, NAME("attn_q.weight"), qd, d->dim, &wb, 1);
    load_mat(&L->wk, g, NAME("attn_k.weight"), kvd, d->dim, &wb, 1);
    load_mat(&L->wv, g, NAME("attn_v.weight"), kvd, d->dim, &wb, 1);
    load_mat(&L->wo, g, NAME("attn_output.weight"), d->dim, qd, &wb, 1);
    load_mat(&L->wgate, g, NAME("ffn_gate.weight"), d->ffn, d->dim, &wb, 1);
    load_mat(&L->wup, g, NAME("ffn_up.weight"), d->ffn, d->dim, &wb, 1);
    load_mat(&L->wdown, g, NAME("ffn_down.weight"), d->dim, d->ffn, &wb, 1);
    if (d->has_conv) {
      load_mat(&L->aconv, g, NAME("attn_conv_proj.weight"), cp, d->dim, &wb, 1);
      load_mat(&L->fconv, g, NAME("ffn_conv_proj.weight"), cp, d->dim, &wb, 1);
    }
#undef NAME
    mat *gu[2] = {&L->wgate, &L->wup};
    concat_mats(&L->wgu, gu, 2);
    if (ie_mat_sink) {
      mat *all[] = {&L->wq, &L->wk, &L->wv, &L->wo, &L->wgate, &L->wup, &L->wdown, &L->wgu, &L->aconv, &L->fconv};
      const unsigned na = d->has_conv ? sizeof all / sizeof all[0] : sizeof all / sizeof all[0] - 2u;
      for (unsigned i = 0; i < na; i++) ie_mat_sink(all[i]);
    }
  }
  m->df = d;
  fprintf(stderr, "dflash: %s, %u layers, block %u, window %u, mask %u, target layers", d->has_conv || d->has_sel ? "DFlash2" : "v1",
          d->n_layer, d->block, d->window, d->mask);
  for (uint32_t i = 0; i < d->n_tl; i++) fprintf(stderr, " %u", d->tl[i]);
  fprintf(stderr, " (%.2f MB)\n", wb / 1e6);
  return 1;
}

void model_df_code(const model *m, int which, uint32_t tok, float *out) {
  const gguf_tensor *t = which ? m->df->sel_next : m->df->sel_prev;
  dequant(t, (uint64_t)tok * m->df->rank, m->df->rank, out);
}

int model_load_dspark(model *m, gguf_file *g) {
  char *arch = gguf_get_str(g, "general.architecture");
  const int ok = arch && !strcmp(arch, "dspark");
  free(arch);
  if (!ok) return 0;
  if (m->arch != ARCH_INKLING2) ie_die("dspark: the target must be an inkling2 model");
  dspark *d = calloc(1, sizeof *d);
  if (!d) ie_die("out of memory");
  int64_t v;
  double f;
#define GI(k, dst) (gguf_get_int(g, "dspark." k, &v) ? (void)((dst) = (uint32_t)v) : ie_die("dspark: no dspark." k))
  GI("block_count", d->n_layer), GI("embedding_length", d->dim), GI("feed_forward_length", d->ffn);
  GI("attention.head_count", d->n_head), GI("attention.head_count_kv", d->n_kv), GI("attention.key_length", d->hd);
  GI("block_size", d->block), GI("markov_rank", d->markov_rank);
#undef GI
  d->window = gguf_get_int(g, "dspark.attention.sliding_window", &v) ? (uint32_t)v : 1024u;
  d->vocab = gguf_get_int(g, "dspark.vocab_size", &v) ? (uint32_t)v : m->vocab;
  d->vocab_unpadded = gguf_get_int(g, "dspark.vocab_size_unpadded", &v) ? (uint32_t)v : (m->vocab_unpadded ? m->vocab_unpadded : d->vocab);
  d->eps = gguf_get_float(g, "dspark.attention.layer_norm_rms_epsilon", &f) ? (float)f : 1e-6f;
  d->rope_base = gguf_get_float(g, "dspark.rope.freq_base", &f) ? (float)f : 1e6f;
  d->logit_cap = gguf_get_float(g, "dspark.final_logit_softcapping", &f) ? (float)f : 30.0f;
  const gguf_kv *tl = gguf_find(g, "dspark.target_layers");
  if (!tl || tl->type != GGUF_ARR || tl->arr_n != 4) ie_die("dspark: need 4 dspark.target_layers");
  d->n_tl = (uint32_t)tl->arr_n;
  for (uint32_t i = 0; i < d->n_tl; i++) {
    int32_t x;
    memcpy(&x, tl->arr_data + 4 * (size_t)i, 4);
    if (x < 0 || (uint32_t)x > m->n_visits) ie_die("dspark: target layer %d out of range [0..%u]", x, m->n_visits);
    d->tl[i] = (uint32_t)x;
  }
  if (d->dim != m->dim || (d->dim & 31u) || (d->markov_rank & 31u) || d->hd > 256u || d->block < 2u || d->block > 16u)
    ie_die("dspark: unsupported shape");
  const uint32_t qd = d->n_head * d->hd, kvd = d->n_kv * d->hd, r = d->markov_rank;
  uint64_t wb = 0;
  load_vec(&d->fuse_weight, g, "fuse_weight", 16u, 0);
  load_vec(&d->mask_embed, g, "mask_embed", d->dim, 0);
  load_vec(&d->out_norm, g, "output_norm.weight", d->dim, 0);
  load_vec(&d->mk_dflare_norm, g, "markov.dflare_norm.weight", d->dim, 0);
  load_vec(&d->mk_rnn_bias, g, "markov.rnn_proj.bias", 3u * r, 0);
  load_vec(&d->mk_hidden_bias, g, "markov.hidden_out.bias", d->dim, 0);
  load_mat(&d->mk_rnn_proj, g, "markov.rnn_proj.weight", 3u * r, 2u * r + 2u * d->dim, &wb, 1);
  load_mat(&d->mk_hidden_out, g, "markov.hidden_out.weight", d->dim, r, &wb, 1);
  load_mat(&d->mk_w1, g, "markov.w1.weight", d->vocab, r, &wb, 1);
  d->mk_w1.host = 1; /* 54.7 MB embedding table (1 row/step): pin in host memory to save VRAM */
  if (ie_mat_sink) {
    ie_mat_sink(&d->mk_rnn_proj);
    ie_mat_sink(&d->mk_hidden_out);
    ie_mat_sink(&d->mk_w1);
  }
  d->l = calloc(d->n_layer, sizeof *d->l);
  if (!d->l) ie_die("out of memory");
  const gguf_kv *ll = gguf_find(g, "dspark.attention.local_layer_ids");
  if (ll && ll->type == GGUF_ARR) {
    for (uint32_t i = 0; i < (uint32_t)ll->arr_n; i++) {
      int32_t id;
      memcpy(&id, ll->arr_data + 4 * (size_t)i, 4);
      if (id >= 0 && (uint32_t)id < d->n_layer) d->l[id].win = d->window;
    }
  }
  char n[128];
  for (uint32_t il = 0; il < d->n_layer; il++) {
    ds_layer *L = &d->l[il];
#define NAME(s) (snprintf(n, sizeof n, "blk.%u." s, il), n)
    load_vec(&L->dflare_norm, g, NAME("dflare_norm.weight"), d->dim, 0);
    load_vec(&L->attn_norm, g, NAME("attn_norm.weight"), d->dim, 0);
    load_vec(&L->ffn_norm, g, NAME("ffn_norm.weight"), d->dim, 0);
    load_vec(&L->q_norm, g, NAME("attn_q_norm.weight"), d->hd, 0);
    load_vec(&L->k_norm, g, NAME("attn_k_norm.weight"), d->hd, 0);
    load_vec(&L->k_ctx_norm, g, NAME("attn_k_ctx_norm.weight"), d->hd, 0);
    load_mat(&L->wq, g, NAME("attn_q.weight"), qd, d->dim, &wb, 1);
    load_mat(&L->wk, g, NAME("attn_k.weight"), kvd, d->dim, &wb, 1);
    load_mat(&L->wv, g, NAME("attn_v.weight"), kvd, d->dim, &wb, 1);
    load_mat(&L->wk_ctx, g, NAME("attn_k_ctx.weight"), kvd, d->dim, &wb, 1);
    load_mat(&L->wv_ctx, g, NAME("attn_v_ctx.weight"), kvd, d->dim, &wb, 1);
    load_mat(&L->wo, g, NAME("attn_output.weight"), d->dim, qd, &wb, 1);
    load_mat(&L->wgate, g, NAME("ffn_gate.weight"), d->ffn, d->dim, &wb, 1);
    load_mat(&L->wup, g, NAME("ffn_up.weight"), d->ffn, d->dim, &wb, 1);
    load_mat(&L->wdown, g, NAME("ffn_down.weight"), d->dim, d->ffn, &wb, 1);
#undef NAME
    mat *gu[2] = {&L->wgate, &L->wup};
    concat_mats(&L->wgu, gu, 2);
    if (ie_mat_sink) {
      mat *all[] = {&L->wq, &L->wk, &L->wv, &L->wk_ctx, &L->wv_ctx, &L->wo, &L->wgate, &L->wup, &L->wdown, &L->wgu};
      for (unsigned i = 0; i < sizeof all / sizeof all[0]; i++) ie_mat_sink(all[i]);
    }
  }
  m->ds = d;
  fprintf(stderr, "dspark: %u layers, block %u, rank %u, window %u, target layers %u %u %u %u (%.2f MB)\n",
          d->n_layer, d->block, d->markov_rank, d->window, d->tl[0], d->tl[1], d->tl[2], d->tl[3], wb / 1e6);
  return 1;
}

/* GGUF Q4_0 bytes of n floats, as ggml's quantize_row_q4_0_ref: d =
 * (the value of largest magnitude) / -8, q = min(15, (int8)(x / d + 8.5)). */
static void quant_q4_0_bytes(const float *x, uint64_t n, uint8_t *out) {
  for (uint64_t b = 0; b < n / 32; b++) {
    float amax = 0.0f, mx = 0.0f;
    for (int j = 0; j < 32; j++) {
      const float v = x[32 * b + j];
      if (fabsf(v) > amax) amax = fabsf(v), mx = v;
    }
    const float d = mx / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    const uint16_t h = ie_f32_to_f16(d);
    memcpy(out + 18 * b, &h, 2);
    for (int j = 0; j < 16; j++) {
      int q0 = (int)(int8_t)(x[32 * b + j] * id + 8.5f), q1 = (int)(int8_t)(x[32 * b + 16 + j] * id + 8.5f);
      q0 = q0 > 15 ? 15 : q0, q1 = q1 > 15 ? 15 : q1;
      out[18 * b + 2 + j] = (uint8_t)(q0 | (q1 << 4));
    }
  }
}

/* Raw GGUF Q2_K blocks (84 bytes per 256 weights, matching kernels/ie_iqraw.inc ie_rq_dequant):
 *   bytes 0..15 : scales[16] = (ls | (lm << 4)) for 16-weight sub-block is = 0..15
 *   bytes 16..79: qs[64], 2 bits per weight at p[16 + 32*(is>>3) + 16*(is&1) + l] >> (2*((is>>1)&3))
 *   bytes 80..81: f16 d; bytes 82..83: f16 dmin; dequantized y = (d * ls) * q - (dmin * lm), q in 0..3. */
static void quant_q2_k_bytes(const float *x, uint64_t n, uint8_t *out) {
  for (uint64_t b = 0; b < n / 256u; b++) {
    uint8_t *p = out + 84u * b;
    const float *xb = x + 256u * b;
    float scales[16], mins[16], max_s = 0.0f, max_m = 0.0f;
    for (int is = 0; is < 16; is++) {
      const float *xs = xb + 16 * is;
      float xmin = 0.0f, xmax = 0.0f, sumx2 = 0.0f;
      for (int l = 0; l < 16; l++) {
        const float v = xs[l];
        if (v < xmin) xmin = v;
        if (v > xmax) xmax = v;
        sumx2 += v * v;
      }
      float m = -xmin, s = (xmax + m) / 3.0f;
      if (s > 0.0f) {
        const float avx = sqrtf(sumx2 * (1.0f / 16.0f));
        for (int it = 0; it < 3; it++) {
          const float isc = s > 0.0f ? 1.0f / s : 0.0f;
          float sw = 0.0f, swq = 0.0f, swq2 = 0.0f, swx = 0.0f, swqx = 0.0f;
          for (int l = 0; l < 16; l++) {
            int q = (int)lrintf((xs[l] + m) * isc);
            q = q < 0 ? 0 : q > 3 ? 3 : q;
            const float w = avx + fabsf(xs[l]), fq = (float)q;
            sw += w, swq += w * fq, swq2 += w * fq * fq;
            swx += w * xs[l], swqx += w * fq * xs[l];
          }
          const float det = sw * swq2 - swq * swq;
          if (det > 0.0f) {
            float ns = (sw * swqx - swq * swx) / det;
            float nm = (swq * swqx - swq2 * swx) / det;
            if (nm < 0.0f) nm = 0.0f, ns = swq2 > 0.0f ? swqx / swq2 : 0.0f;
            if (ns < 0.0f) ns = 0.0f;
            s = ns, m = nm;
          }
        }
      } else {
        s = 0.0f, m = 0.0f;
      }
      scales[is] = s, mins[is] = m;
      if (s > max_s) max_s = s;
      if (m > max_m) max_m = m;
    }
    const uint16_t hd = ie_f32_to_f16(max_s * (1.0f / 15.0f)), hm = ie_f32_to_f16(max_m * (1.0f / 15.0f));
    const float d = ie_f16_to_f32(hd), dmin = ie_f16_to_f32(hm);
    const float id = d > 0.0f ? 1.0f / d : 0.0f, idm = dmin > 0.0f ? 1.0f / dmin : 0.0f;
    memset(p + 16, 0, 64);
    memcpy(p + 80, &hd, 2);
    memcpy(p + 82, &hm, 2);
    for (int is = 0; is < 16; is++) {
      int ls = (int)lrintf(scales[is] * id), lm = (int)lrintf(mins[is] * idm);
      ls = ls < 0 ? 0 : ls > 15 ? 15 : ls;
      lm = lm < 0 ? 0 : lm > 15 ? 15 : lm;
      p[is] = (uint8_t)(ls | (lm << 4));
      const float dl = d * (float)ls, ml = dmin * (float)lm, idl = dl > 0.0f ? 1.0f / dl : 0.0f;
      const float *xs = xb + 16 * is;
      uint8_t *qb = p + 16 + 32 * (is >> 3) + 16 * (is & 1);
      const int sh = 2 * ((is >> 1) & 3);
      for (int l = 0; l < 16; l++) {
        int q = dl > 0.0f ? (int)lrintf((xs[l] + ml) * idl) : 0;
        q = q < 0 ? 0 : q > 3 ? 3 : q;
        qb[l] |= (uint8_t)(q << sh);
      }
    }
  }
}

/* ---- vocabulary shortlists (IE_DRAFT_VOCAB=64k or a manifest) ---- */

static const uint32_t SHA_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

static void sha256_block(uint32_t h[8], const uint8_t *p) {
#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
  uint32_t w[64], a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
  for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
  for (int i = 16; i < 64; i++) {
    const uint32_t s0 = ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ (w[i - 15] >> 3), s1 = ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  for (int i = 0; i < 64; i++) {
    const uint32_t t1 = hh + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) + ((e & f) ^ (~e & g)) + SHA_K[i] + w[i];
    const uint32_t t2 = (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
  }
  h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
#undef ROR
}

void ie_sha256_hex(const void *p, size_t n, char out[65]) {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  const uint8_t *s = p;
  size_t full = n / 64, rem = n % 64;
  for (size_t i = 0; i < full; i++) sha256_block(h, s + 64 * i);
  uint8_t tail[128];
  memset(tail, 0, sizeof tail);
  if (rem) memcpy(tail, s + 64 * full, rem);
  tail[rem] = 0x80;
  const size_t tl = rem < 56 ? 64 : 128;
  const uint64_t bits = (uint64_t)n * 8u;
  for (int j = 0; j < 8; j++) tail[tl - 1 - j] = (uint8_t)(bits >> (8 * j));
  sha256_block(h, tail);
  if (tl == 128) sha256_block(h, tail + 64);
  for (int i = 0; i < 8; i++) snprintf(out + 8 * i, 9, "%08x", h[i]);
}

/* The text after `"key" :` in a JSON object (the first such key anywhere in s), or NULL. */
static const char *mj_find(const char *s, const char *key) {
  char pat[96];
  snprintf(pat, sizeof pat, "\"%s\"", key);
  const size_t pl = strlen(pat);
  for (const char *p = s; (p = strstr(p, pat)); p += pl) {
    const char *q = p + pl;
    while (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n') q++;
    if (*q != ':') continue;
    for (q++; *q == ' ' || *q == '\t' || *q == '\r' || *q == '\n';) q++;
    return q;
  }
  return NULL;
}
/* The string value v (v[0] == '"'; escapes \\ \" \/ only) into out; 0 if it is not one or does not fit. */
static int mj_str(const char *v, char *out, size_t cap) {
  if (!v || *v != '"') return 0;
  size_t n = 0;
  for (v++; *v != '"'; v++) {
    if (!*v) return 0;
    if (*v == '\\') {
      v++;
      if (*v != '\\' && *v != '"' && *v != '/') return 0;
    }
    if (n + 1 >= cap) return 0;
    out[n++] = *v;
  }
  out[n] = 0;
  return 1;
}
static int mj_uint(const char *v, uint32_t *out) {
  if (!v || *v < '0' || *v > '9') return 0;
  char *end;
  const unsigned long long x = strtoull(v, &end, 10);
  if (x > 0xffffffffull) return 0;
  *out = (uint32_t)x;
  return 1;
}

uint32_t *ie_load_vocab_shortlist(const char *manifest, uint32_t vocab, uint32_t *n_out, char *err, size_t errn) {
  uint32_t *ids = NULL;
  char *mt = NULL, *it = NULL;
  *n_out = 0;
#define FAIL(...) do { snprintf(err, errn, __VA_ARGS__); free(mt), free(it), free(ids); return NULL; } while (0)
  /* a whole file (at most cap bytes) into a malloc'ed, 0-terminated buffer; NULL if unreadable or larger */
#define SLURP(dst, path, cap, nn) do { \
    FILE *f_ = fopen(path, "rb"); \
    if (f_) { \
      (dst) = malloc((size_t)(cap) + 1); \
      if (!(dst)) ie_die("out of memory"); \
      (nn) = fread(dst, 1, (size_t)(cap) + 1, f_); \
      fclose(f_); \
      if ((nn) > (size_t)(cap)) free(dst), (dst) = NULL; else (dst)[nn] = 0; \
    } } while (0)
  size_t mn = 0, in = 0;
  SLURP(mt, manifest, 1 << 20, mn);
  if (!mt) FAIL("cannot read the manifest %s (missing, or over 1 MiB)", manifest);
  char schema[96], ipath[1024], want[128], tmp[1024];
  if (!mj_str(mj_find(mt, "schema"), schema, sizeof schema) || strcmp(schema, "rdna4.mtp.vocab-shortlist.v1"))
    FAIL("%s: schema is not rdna4.mtp.vocab-shortlist.v1", manifest);
  uint32_t mvocab = 0, size = 0;
  if (!mj_uint(mj_find(mt, "vocabulary"), &mvocab) || mvocab != vocab) FAIL("%s: vocabulary %u, the model has %u tokens", manifest, mvocab, vocab);
  if (!mj_uint(mj_find(mt, "size"), &size) || size < 1 || size > vocab) FAIL("%s: bad size %u", manifest, size);
  if (!mj_str(mj_find(mt, "ids_text"), ipath, sizeof ipath)) FAIL("%s: no ids_text path", manifest);
  if (!mj_str(mj_find(mt, "ids_text_sha256"), want, sizeof want) || strlen(want) != 64) FAIL("%s: no ids_text_sha256", manifest);
  uint32_t eog[64], n_eog = 0;
  const char *ev = mj_find(mt, "required_eog_ids");
  if (!ev || *ev != '[') FAIL("%s: no required_eog_ids", manifest);
  for (ev++;;) {
    while (*ev == ' ' || *ev == '\t' || *ev == '\r' || *ev == '\n' || *ev == ',') ev++;
    if (*ev == ']') break;
    if (n_eog == 64 || !mj_uint(ev, &eog[n_eog])) FAIL("%s: bad required_eog_ids", manifest);
    n_eog++;
    while (*ev >= '0' && *ev <= '9') ev++;
  }
  /* the ids file: the path as written; else the same file name next to the manifest (the lab may have moved) */
  const uint64_t cap = (uint64_t)size * 12u + 4096u; /* a decimal id and a newline: at most 11 bytes */
  SLURP(it, ipath, cap, in);
  if (!it) {
    const char *base = ipath;
    for (const char *c = ipath; *c; c++)
      if (*c == '/' || *c == '\\') base = c + 1;
    const char *sl = manifest, *cut = manifest;
    for (; *sl; sl++)
      if (*sl == '/' || *sl == '\\') cut = sl + 1;
    if ((size_t)(cut - manifest) + strlen(base) + 1 > sizeof tmp) FAIL("%s: path too long", manifest);
    snprintf(tmp, sizeof tmp, "%.*s%s", (int)(cut - manifest), manifest, base);
    SLURP(it, tmp, cap, in);
  }
  if (!it) FAIL("%s: cannot read the ids file %s (or its name next to the manifest)", manifest, ipath);
  char got[65];
  ie_sha256_hex(it, in, got);
  for (char *c = want; *c; c++) *c = (char)(*c >= 'A' && *c <= 'F' ? *c + 32 : *c);
  if (strcmp(got, want)) FAIL("%s: ids file sha256 %s, the manifest says %s", manifest, got, want);
  ids = malloc((size_t)size * 4);
  if (!ids) ie_die("out of memory");
  uint32_t n = 0;
  for (const char *p = it; *p;) {
    if (*p == '\n' || *p == '\r' || *p == ' ' || *p == '\t') { p++; continue; }
    char *end;
    if (*p < '0' || *p > '9') FAIL("%s: ids file: unexpected character 0x%02x", manifest, (unsigned)(uint8_t)*p);
    const unsigned long long x = strtoull(p, &end, 10);
    if (x >= vocab) FAIL("%s: ids file: id %llu >= vocabulary %u", manifest, x, vocab);
    if (n == size) FAIL("%s: ids file holds more than size = %u ids", manifest, size);
    ids[n++] = (uint32_t)x;
    p = end;
  }
  if (n != size) FAIL("%s: ids file holds %u ids, size says %u", manifest, n, size);
  /* ascending ids: the draft head's argmax then breaks ties toward the lowest token, as the full vocabulary does.
   * A sorted file is the usual case; else a bitmap sorts it and finds a repeated id. */
  uint32_t asc = 1;
  for (uint32_t i = 1; i < n; i++) asc &= ids[i] > ids[i - 1];
  if (!asc) {
    char *seen = calloc(vocab, 1);
    if (!seen) ie_die("out of memory");
    for (uint32_t j = 0; j < n; j++) {
      if (seen[ids[j]]) {
        const uint32_t dup = ids[j];
        free(seen);
        FAIL("%s: ids file: id %u twice", manifest, dup);
      }
      seen[ids[j]] = 1;
    }
    uint32_t k = 0;
    for (uint32_t v = 0; v < vocab; v++)
      if (seen[v]) ids[k++] = v;
    free(seen);
  }
  for (uint32_t e = 0; e < n_eog; e++) { /* binary search: the ids are ascending now */
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (ids[mid] < eog[e]) lo = mid + 1; else hi = mid;
    }
    if (lo == n || ids[lo] != eog[e]) FAIL("%s: required end-of-generation id %u is not in the shortlist", manifest, eog[e]);
  }
  free(mt), free(it);
  *n_out = n;
  return ids;
#undef SLURP
#undef FAIL
}

/* IE_DRAFT_VOCAB as a shortlist: "64k" (the map of 64 * 1024 tokens in data/mtp-vocab-pilot/maps/default, found from
 * IE_DRAFT_VOCAB_DIR, or from the working directory up) or a path to a manifest (.json / with a directory). 1 and the
 * manifest path in mp if so; 0: a token count. */
static int draft_vocab_manifest(const char *dv, char *mp, size_t cap) {
  const size_t l = strlen(dv);
  if (strchr(dv, '/') || strchr(dv, '\\') || (l > 5 && !strcmp(dv + l - 5, ".json"))) {
    snprintf(mp, cap, "%s", dv);
    return 1;
  }
  char *end;
  const unsigned long k = strtoul(dv, &end, 10);
  if (end == dv || (*end != 'k' && *end != 'K') || end[1]) return 0;
  char name[64];
  snprintf(name, sizeof name, "map-%lu.manifest.json", k * 1024ul);
  const char *dir = getenv("IE_DRAFT_VOCAB_DIR");
  if (dir && *dir) {
    snprintf(mp, cap, "%s/%s", dir, name);
    return 1;
  }
  char up[64] = "";
  for (int i = 0; i < 7; i++, strcat(up, "../")) {
    snprintf(mp, cap, "%sdata/mtp-vocab-pilot/maps/default/%s", up, name);
    FILE *f = fopen(mp, "rb");
    if (f) { fclose(f); return 1; }
  }
  snprintf(mp, cap, "data/mtp-vocab-pilot/maps/default/%s", name); /* not found: the loader says so */
  return 1;
}

uint32_t model_draft_tok_map(const model *m, uint32_t i) {
  const uint32_t lim = m->out_draft.rows ? m->out_draft.rows : m->ds_head.rows;
  if (i >= lim) ie_die("draft head row %u >= %u rows", i, lim);
  return m->dv_map[i];
}

/* The DFlash block's logits over the draft vocabulary (the rows of the MTP draft head, IE_DRAFT_VOCAB) as Q8_0: the
 * Q8_0 GEMV has multi-token kernels (the draft head's Q4_0 has not: a block read it once per row). Only the drafts
 * change (the verify uses out). Default for DFlash v1 drafters: MiMo-9B, 33044 rows (144 MB read per block instead
 * of the 834 MB Q6_K output matrix), drafts 5.55 -> 4.37 ms per step, slightly fewer accepted, +0.7 to +3.4 % on
 * four prompts. Not for DFlash 2 (Saluki 27B, 65812 rows: -4 %). IE_DFLASH_HEAD=full / draft: either one. */
void model_make_df_head(model *m, gguf_file *g) {
  mat *w = &m->df_head;
  memset(w, 0, sizeof *w);
  const char *e = getenv("IE_DFLASH_HEAD");
  if (!m->df || !m->out_draft.rows || (e && !strcmp(e, "full")) || (m->df->has_sel && !(e && !strcmp(e, "draft")))) return;
  const gguf_tensor *t = gguf_tensor_find(g, "output.weight");
  if (!t) t = need_tensor(g, "token_embd.weight"); /* tied */
  w->kind = MAT_Q8, w->rows = m->out_draft.rows, w->cols = m->dim, w->nb = m->dim / 32;
  if (!ie_q8_sizes_ok(w->rows, w->nb)) {
    memset(w, 0, sizeof *w);
    return;
  }
  float *row = ie_alloc((size_t)m->dim * 4);
  uint8_t *q8 = ie_alloc((size_t)w->rows * w->nb * 34);
  for (uint32_t r = 0; r < w->rows; r++) {
    dequant(t, (uint64_t)model_draft_tok(m, r) * m->dim, m->dim, row);
    quant_q8_0_bytes(row, m->dim, q8 + (size_t)r * w->nb * 34);
  }
  repack_q8(w, q8);
  free(row), free(q8);
  if (ie_mat_sink) ie_mat_sink(w);
  fprintf(stderr, "dflash: head Q8_0 over the draft vocabulary (%u rows, %.2f MB)\n", w->rows, (double)w->rows * w->nb * 34 / 1e6);
}

static uint32_t *build_draft_shortlist(const model *m, gguf_file *g, uint32_t target, uint32_t *n_out);

void model_make_ds_head(model *m, gguf_file *g) {
  mat *w = &m->ds_head;
  memset(w, 0, sizeof *w);
  const char *he = getenv("IE_DSPARK_HEAD");
  const char *dv = getenv("IE_DRAFT_VOCAB");
  /* Default to full target output.weight (0 extra VRAM, avoids missing high-ID BPE tokens);
   * set IE_DRAFT_VOCAB=N (e.g. 32768) to build a shortlist Q8_0 draft head. */
  if (!m->ds || (he && !strcmp(he, "full")) || !dv || !dv[0]) return;
  uint32_t N = (uint32_t)atoi(dv);
  if (!N || N >= m->vocab) return;
  const gguf_tensor *t = gguf_tensor_find(g, "output.weight");
  if (!t) t = need_tensor(g, "token_embd.weight");
  free(m->dv_map);
  m->dv_map = NULL, m->dv_n = m->dv_s = 0;
  uint32_t n = 0;
  m->dv_map = build_draft_shortlist(m, g, N, &n);
  if (!m->dv_map || !n) return;
  w->kind = MAT_Q8, w->rows = n, w->cols = m->dim, w->nb = m->dim / 32;
  if (!ie_q8_sizes_ok(w->rows, w->nb)) {
    memset(w, 0, sizeof *w);
    free(m->dv_map);
    m->dv_map = NULL;
    return;
  }
  float *row = ie_alloc((size_t)m->dim * 4);
  uint8_t *q8 = ie_alloc((size_t)w->rows * w->nb * 34);
  for (uint32_t r = 0; r < w->rows; r++) {
    dequant(t, (uint64_t)m->dv_map[r] * m->dim, m->dim, row);
    quant_q8_0_bytes(row, m->dim, q8 + (size_t)r * w->nb * 34);
  }
  repack_q8(w, q8);
  free(row), free(q8);
  if (ie_mat_sink) ie_mat_sink(w);
  fprintf(stderr, "dspark: head Q8_0 over draft vocabulary (%u of %u rows, %.2f MB)\n", w->rows, m->vocab, (double)w->rows * w->nb * 34 / 1e6);
}

/* 1 if token v decodes exclusively to 7-bit ASCII bytes (0x00 .. 0x7F).
 * In GPT-2 byte-level BPE (m->tok_gpt2), printable ASCII 33..126 are literal single bytes and
 * ASCII 0..32, 127 (\t, \n, \r, ' ', etc.) map to U+0100..U+0121 (UTF-8 0xC4 0x80..0xA1). */
static int tok_is_ascii(const model *m, uint32_t v) {
  if (!m->tokens || v >= m->n_tokens || !m->tokens[v] || !m->tokens[v][0]) return 0;
  const unsigned char *s = (const unsigned char *)m->tokens[v];
  for (size_t i = 0; s[i];) {
    if (s[i] < 0x80u) i++;
    else if (m->tok_gpt2 && s[i] == 0xC4u && s[i + 1] >= 0x80u && s[i + 1] <= 0xA1u) i += 2;
    else return 0;
  }
  return 1;
}

/* Build an ascending draft-vocab shortlist of `target` token IDs (< m->vocab) that always retains:
 *   (1) all non-normal / special / chat / thinking / EOG / control tokens (<think>, </think>, <|im_end|>, etc.);
 *   (2) the universal top-frequency BPE core 0 .. target/2 - 1 (byte tokens, punctuation, common UTF-8 symbols);
 *   (3) pure-ASCII / code / English / whitespace tokens in BPE merge-rank order;
 *   (4) lowest remaining BPE token IDs up to `target`. */
static uint32_t *build_draft_shortlist(const model *m, gguf_file *g, uint32_t target, uint32_t *n_out) {
  *n_out = 0;
  if (!target || target >= m->vocab) return NULL;
  uint8_t *keep = calloc(m->vocab, 1);
  if (!keep) ie_die("out of memory");
  uint32_t cnt = 0;
#define KEEP_ID(id_)                                       \
  do {                                                     \
    const uint32_t u_ = (uint32_t)(id_);                   \
    if (u_ < m->vocab && !keep[u_] && cnt < target) keep[u_] = 1, cnt++; \
  } while (0)
  int64_t sp_id;
  if (gguf_get_int(g, "tokenizer.ggml.eos_token_id", &sp_id) && sp_id >= 0) KEEP_ID(sp_id);
  if (gguf_get_int(g, "tokenizer.ggml.bos_token_id", &sp_id) && sp_id >= 0) KEEP_ID(sp_id);
  if (gguf_get_int(g, "tokenizer.ggml.eot_token_id", &sp_id) && sp_id >= 0) KEEP_ID(sp_id);
  const gguf_kv *tt = gguf_find(g, "tokenizer.ggml.token_type");
  if (tt && tt->type == GGUF_ARR && tt->arr_type == GGUF_I32) {
    const uint32_t tn = (uint32_t)tt->arr_n < m->vocab ? (uint32_t)tt->arr_n : m->vocab;
    for (uint32_t v = 0; v < tn; v++) {
      int32_t ty;
      memcpy(&ty, tt->arr_data + (size_t)v * 4, 4);
      if (ty != 1) KEEP_ID(v);
    }
  }
  if (m->tokens) {
    static const char *const must[] = {"<think>", "</think>", "<|im_start|>", "<|im_end|>",
                                       "<|endoftext|>", "<tool_call>", "</tool_call>"};
    const uint32_t tn = m->n_tokens < m->vocab ? m->n_tokens : m->vocab;
    for (uint32_t v = 0; v < tn; v++) {
      const char *s = m->tokens[v];
      if (!s || (s[0] != '<' && s[0] != '[')) continue;
      for (unsigned k = 0; k < sizeof must / sizeof must[0]; k++)
        if (!strcmp(s, must[k])) { KEEP_ID(v); break; }
    }
  }
  const int raw_pfx = getenv("IE_DRAFT_VOCAB_PREFIX") && getenv("IE_DRAFT_VOCAB_PREFIX")[0] == '1';
  const uint32_t core = raw_pfx ? target : target / 2u;
  for (uint32_t v = 0; v < core && v < m->vocab && cnt < target; v++) KEEP_ID(v);
  if (!raw_pfx) {
    for (uint32_t v = core; v < m->vocab && cnt < target; v++)
      if (tok_is_ascii(m, v)) KEEP_ID(v);
  }
  for (uint32_t v = 0; v < m->vocab && cnt < target; v++) KEEP_ID(v);
#undef KEEP_ID
  uint32_t *ids = ie_alloc((size_t)cnt * 4);
  uint32_t k = 0;
  for (uint32_t v = 0; v < m->vocab; v++)
    if (keep[v]) ids[k++] = v;
  free(keep);
  *n_out = k;
  return ids;
}

void model_make_draft_head(model *m, gguf_file *g) {
  const gguf_tensor *t = gguf_tensor_find(g, "output.weight");
  if (!t) t = need_tensor(g, "token_embd.weight"); /* tied */
  mat *w = &m->out_draft;
  memset(w, 0, sizeof *w);
  w->rows = m->vocab, w->cols = m->dim, w->nb = m->dim / 32;
  /* the rows come from the model file's output matrix (the MTP file's own copy is identical): vocab rows of dim */
  if (t->ne[0] != m->dim || t->ne[1] * t->ne[2] * t->ne[3] < m->vocab) ie_die("draft head: %s is %llu x %llu, expected %u x %u", t->name, (unsigned long long)t->ne[0], (unsigned long long)t->ne[1], m->dim, m->vocab);
  /* IE_DRAFT_QTYPE: raw (default when output.weight is already a MAT_RAWQ type and dim % 256 == 0: exact raw GGUF row
   * copy, zero requantization loss, full _tr/_tr4/_tr8/t16 speed), q2_k (MAT_RAWQ GGML_Q2_K), q4_0 (MAT_Q4), or q8_0 (MAT_Q8).
   * IE_DRAFT_VOCAB: default 64k corpus-ranked manifest when present for 248k Qwen vocab (else 65536 shortlist); 0 = full vocab. */
  const char *qt = getenv("IE_DRAFT_QTYPE");
  const uint32_t src_rbs = (m->dim % 256u == 0u) ? ie_rq_bs(t->type) : 0u;
  int want_q = src_rbs ? 1 : (m->dim % 256u == 0u ? 2 : 4); /* 1: exact raw MAT_RAWQ, 2: Q2_K, 4: Q4_0, 8: Q8_0 */
  if (qt && (!strcmp(qt, "q2_k") || !strcmp(qt, "Q2_K") || !strcmp(qt, "q2"))) want_q = 2;
  else if (qt && (!strcmp(qt, "q4_0") || !strcmp(qt, "Q4_0") || !strcmp(qt, "q4"))) want_q = 4;
  else if (qt && (!strcmp(qt, "q8_0") || !strcmp(qt, "Q8_0") || !strcmp(qt, "q8"))) want_q = 8;
  if ((want_q == 1 && !src_rbs) || (want_q == 2 && m->dim % 256u != 0u)) want_q = 4;

  m->dv_n = m->dv_s = 0;
  free(m->dv_map);
  m->dv_map = NULL;
  const char *dv = getenv("IE_DRAFT_VOCAB");
  char mpath[1024];
  const int sl = dv && draft_vocab_manifest(dv, mpath, sizeof mpath);
  uint32_t N = dv ? (uint32_t)atoi(dv) : 65536u;
  if (sl) {
    char err[600];
    uint32_t n = 0;
    m->dv_map = ie_load_vocab_shortlist(mpath, m->vocab, &n, err, sizeof err);
    if (!m->dv_map) ie_die("IE_DRAFT_VOCAB=%s: %s", dv, err);
    w->rows = n;
    fprintf(stderr, "mtp: draft vocabulary: %u of %u tokens from %s (ids sha256 checked)\n", n, m->vocab, mpath);
  } else if (!dv && m->vocab == 248320u && draft_vocab_manifest("64k", mpath, sizeof mpath)) {
    char err[600];
    uint32_t n = 0;
    m->dv_map = ie_load_vocab_shortlist(mpath, m->vocab, &n, err, sizeof err);
    if (m->dv_map) {
      w->rows = n;
      fprintf(stderr, "mtp: draft vocabulary: %u of %u tokens from %s (ids sha256 checked)\n", n, m->vocab, mpath);
    }
  }
  if (!m->dv_map && !sl && N && N < m->vocab) {
    uint32_t n = 0;
    m->dv_map = build_draft_shortlist(m, g, N, &n);
    if (m->dv_map) {
      w->rows = n;
      fprintf(stderr, "mtp: draft vocabulary: %u of %u tokens (special + BPE core + ASCII/code shortlist)\n", n, m->vocab);
    }
  }
  if ((want_q == 1 || want_q == 2) && (w->rows == 0u || w->nb < 8u || w->nb % 8u != 0u)) want_q = 4;
  if ((want_q == 4 && !ie_sizes_ok(w->rows, w->nb)) || (want_q == 8 && !ie_q8_sizes_ok(w->rows, w->nb))) {
    memset(w, 0, sizeof *w);
    free(m->dv_map);
    m->dv_map = NULL, m->dv_n = m->dv_s = 0;
    return;
  }
  if (want_q == 1) {
    const size_t rb = (size_t)(m->dim / 256u) * src_rbs, tot = (size_t)w->rows * rb;
    w->kind = MAT_RAWQ, w->sub = t->type;
    w->qs = ie_alloc(tot);
    uint8_t *dst = (uint8_t *)w->qs;
    const uint8_t *src = (const uint8_t *)t->data;
    for (uint32_t r = 0; r < w->rows; r++)
      memcpy(dst + (size_t)r * rb, src + (size_t)model_draft_tok(m, r) * rb, rb);
    if (ie_mat_sink) ie_mat_sink(w);
    fprintf(stderr, "mtp: draft head %s (%u rows, %.2f MB)\n", ggml_type_name(t->type), w->rows, (double)tot / 1e6);
    return;
  }
  float *row = ie_alloc((size_t)m->dim * 4);
  if (want_q == 2) {
    const size_t rb = (size_t)(m->dim / 256u) * 84u, tot = (size_t)w->rows * rb;
    w->kind = MAT_RAWQ, w->sub = GGML_Q2_K;
    w->qs = ie_alloc(tot);
    uint8_t *dst = (uint8_t *)w->qs;
    for (uint32_t r = 0; r < w->rows; r++) {
      dequant(t, (uint64_t)model_draft_tok(m, r) * m->dim, m->dim, row);
      quant_q2_k_bytes(row, m->dim, dst + (size_t)r * rb);
    }
    free(row);
    if (ie_mat_sink) ie_mat_sink(w);
    fprintf(stderr, "mtp: draft head Q2_K (%u rows, %.2f MB)\n", w->rows, (double)tot / 1e6);
    return;
  }
  if (want_q == 8) {
    w->kind = MAT_Q8;
    uint8_t *q8 = ie_alloc((size_t)w->rows * w->nb * 34);
    for (uint32_t r = 0; r < w->rows; r++) {
      dequant(t, (uint64_t)model_draft_tok(m, r) * m->dim, m->dim, row);
      quant_q8_0_bytes(row, m->dim, q8 + (size_t)r * w->nb * 34);
    }
    repack_q8(w, q8);
    free(row), free(q8);
    if (ie_mat_sink) ie_mat_sink(w);
    fprintf(stderr, "mtp: draft head Q8_0 (%u rows, %.2f MB)\n", w->rows, (double)w->rows * w->nb * 34 / 1e6);
    return;
  }
  w->kind = MAT_Q4;
  uint8_t *q4 = ie_alloc((size_t)w->rows * w->nb * 18);
  for (uint32_t r = 0; r < w->rows; r++) {
    dequant(t, (uint64_t)model_draft_tok(m, r) * m->dim, m->dim, row);
    quant_q4_0_bytes(row, m->dim, q4 + (size_t)r * w->nb * 18);
  }
  repack_q4(w, q4);
  free(row), free(q4);
  if (ie_mat_sink) ie_mat_sink(w);
  fprintf(stderr, "mtp: draft head Q4_0 (%u rows, %.2f MB)\n", w->rows, (double)w->rows * w->nb * 18 / 1e6);
}

void model_free(model *m) {
  free_mat(&m->out_draft);
  free_mat(&m->df_head);
  free_mat(&m->ds_head);
  if (m->ds) {
    dspark *d = m->ds;
    free(d->fuse_weight.f), free(d->mask_embed.f), free(d->out_norm.f);
    free(d->mk_dflare_norm.f), free(d->mk_rnn_bias.f), free(d->mk_hidden_bias.f);
    free_mat(&d->mk_rnn_proj), free_mat(&d->mk_hidden_out), free_mat(&d->mk_w1);
    for (uint32_t i = 0; i < d->n_layer; i++) {
      ds_layer *L = &d->l[i];
      free(L->dflare_norm.f), free(L->attn_norm.f), free(L->ffn_norm.f);
      free(L->q_norm.f), free(L->k_norm.f), free(L->k_ctx_norm.f);
      free_mat(&L->wq), free_mat(&L->wk), free_mat(&L->wv), free_mat(&L->wk_ctx), free_mat(&L->wv_ctx);
      free_mat(&L->wo), free_mat(&L->wgate), free_mat(&L->wup), free_mat(&L->wdown), free_mat(&L->wgu);
    }
    free(d->l), free(d);
    m->ds = NULL;
  }
  free(m->dv_map);
  if (m->has_mtp) {
    m->n_layer++; /* free the MTP layer with the others */
    free_mat(&m->eh), free(m->enorm.f), free(m->hnorm.f), free(m->head_norm.f);
  }
  free_mat(&m->tok);
  free_mat(&m->mtp_tok);
  if (!m->tied) free_mat(&m->out);
  free(m->out_norm.f);
  free(m->tok_norm.f), free(m->out_res_q.f), free(m->loop_res_q.f), free(m->loop_kv_res_q.f);
  free(m->ones_dim.f), free(m->ones_hd.f), free(m->kv_win);
  for (uint32_t i = 0; i < m->n_layer; i++) {
    layer *L = &m->l[i];
    free(L->attn_norm.f), free(L->ffn_norm.f), free(L->bq.f), free(L->bk.f), free(L->bv.f), free(L->bqkv.f);
    free_mat(&L->wqkv), free_mat(&L->wgu);
    free_mat(&L->wq), free_mat(&L->wk), free_mat(&L->wv), free_mat(&L->wo);
    free_mat(&L->wgate), free_mat(&L->wup), free_mat(&L->wdown);
    free_mat(&L->wlqkv), free_mat(&L->wz), free_mat(&L->wlz), free_mat(&L->wbeta), free_mat(&L->walpha), free_mat(&L->win);
    free(L->q_norm.f), free(L->k_norm.f), free(L->conv.f), free(L->dt_bias.f), free(L->ssm_a.f), free(L->ssm_norm.f);
    free(L->sinks.f), free(L->ebias.f), free(L->pan.f), free(L->pffn.f);
    free_mat(&L->wagt), free_mat(&L->wrel), free_mat(&L->wsink), free_mat(&L->k_conv_proj), free_mat(&L->v_conv_proj);
    free(L->q_inp_norm.f), free(L->k_inp_norm.f), free(L->v_inp_norm.f);
    free(L->k_conv.f), free(L->v_conv.f), free(L->attn_pconv.f), free(L->ffn_pconv.f);
    free(L->q_clip.f), free(L->rel_clip.f), free(L->rel_bias.f), free(L->attn_res_q.f), free(L->ffn_res_q.f);
    for (int b = 0; b < 4; b++)
      free_mat(&L->loop[b].down), free_mat(&L->loop[b].gain), free_mat(&L->loop[b].forget), free_mat(&L->loop[b].wfg);
    free_mat(&L->wr), free_mat(&L->egu), free_mat(&L->eg), free_mat(&L->eu), free_mat(&L->ed);
  }
  free(m->l);
  for (uint32_t i = 0; i < m->n_tokens; i++) free(m->tokens[i]);
  free(m->tokens);
}

/* ------------------------------------------------------------ detokenize */

/* GPT2 byte-level: the inverse of bytes_to_unicode(). */
static int gpt2_byte(uint32_t cp) {
  if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) return (int)cp;
  /* the other 68 bytes map to 256, 257, ... in byte order */
  uint32_t n = 0;
  for (uint32_t b = 0; b < 256; b++) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) continue;
    if (cp == 256 + n) return (int)b;
    n++;
  }
  return -1;
}

static uint32_t utf8_next(const unsigned char **s) {
  const unsigned char *p = *s;
  uint32_t c = *p++;
  int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
  if (extra) c &= 0x3Fu >> extra;
  while (extra-- > 0 && (*p & 0xC0) == 0x80) c = (c << 6) | (*p++ & 0x3Fu);
  *s = p;
  return c;
}

char *model_detok(const model *m, uint32_t t, uint32_t *len) {
  const char *s = t < m->n_tokens ? m->tokens[t] : "";
  size_t n = strlen(s);
  char *o = malloc(n + 4);
  uint32_t k = 0;
  if (m->tok_gpt2) {
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
      int b = gpt2_byte(utf8_next(&p));
      if (b >= 0) o[k++] = (char)b;
    }
  } else {
    unsigned hx;
    if (n == 6 && sscanf(s, "<0x%2X>", &hx) == 1) {
      o[k++] = (char)hx;
    } else {
      for (size_t i = 0; i < n; i++) {
        if ((unsigned char)s[i] == 0xE2 && (unsigned char)s[i + 1] == 0x96 && (unsigned char)s[i + 2] == 0x81) {
          o[k++] = ' '; /* U+2581 */
          i += 2;
        } else {
          o[k++] = s[i];
        }
      }
    }
  }
  o[k] = 0;
  *len = k;
  return o;
}

/* --------------------------------------------------------------- the graph */

typedef struct {
  graph *g;
  uint32_t cap_ops;
  int layer; /* the layer of the ops emitted now (-1: outside the layers) */
} gb;

static int new_buf(graph *g, const char *name, uint32_t bytes) {
  if (g->n_bufs >= MAX_BUFS) ie_die("too many buffers");
  buf *b = &g->bufs[g->n_bufs];
  b->name = name;
  b->stride = (bytes + 255u) & ~255u; /* each token's part 256-aligned */
  b->size = b->stride * g->T;
  b->first = UINT32_MAX;
  b->last = 0;
  return (int)g->n_bufs++;
}

void graph_cap_logits(graph *g) {
  if (g->T <= 16u) return;
  const int ids[2] = {g->logits, g->mtp_logits};
  for (int k = 0; k < 2; k++) {
    if (ids[k] < 0) continue;
    buf *b = &g->bufs[ids[k]];
    b->rows = 16u;
    b->size = b->stride * 16u;
  }
}

void ie_ink_parse_mask(uint64_t *lo, uint64_t *hi, uint64_t *hi3) {
  uint64_t m_lo = IE_INK_DEFAULT_MASK, m_hi = IE_INK_DEFAULT_MASK2, m_h3 = IE_INK_DEFAULT_MASK3;
  const char *e1 = getenv("IE_INKLING_KERNEL_MASK");
  const char *e2 = getenv("IE_INKLING_KERNEL_MASK2");
  const char *e3 = getenv("IE_INKLING_KERNEL_MASK3");
  int hex128 = 0, hex192 = 0;
  if (e1 && *e1) {
    const char *p = e1;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
      p += 2;
      m_lo = 0ULL;
      m_hi = 0ULL;
      m_h3 = 0ULL;
      uint32_t digits = 0u;
      for (; *p; p++) {
        uint64_t nib;
        if (*p >= '0' && *p <= '9') nib = (uint64_t)(*p - '0');
        else if (*p >= 'a' && *p <= 'f') nib = (uint64_t)(*p - 'a' + 10);
        else if (*p >= 'A' && *p <= 'F') nib = (uint64_t)(*p - 'A' + 10);
        else break;
        m_h3 = (m_h3 << 4) | (m_hi >> 60);
        m_hi = (m_hi << 4) | (m_lo >> 60);
        m_lo = (m_lo << 4) | nib;
        digits++;
      }
      hex128 = (digits > 16u);
      hex192 = (digits > 32u);
    } else {
      m_lo = (uint64_t)strtoull(e1, NULL, 0);
      m_hi = 0ULL;
      m_h3 = 0ULL;
    }
  }
  if (!hex128 && e2 && *e2) m_hi = (uint64_t)strtoull(e2, NULL, 0);
  if (!hex192 && e3 && *e3) m_h3 = (uint64_t)strtoull(e3, NULL, 0);
  if (lo) *lo = m_lo;
  if (hi) *hi = m_hi;
  if (hi3) *hi3 = m_h3;
}

static int buf_used_outside(const graph *g, int id, uint32_t lo, uint32_t hi) {
  if (id < 0) return 0;
  for (uint32_t j = 0; j < g->n_ops; j++) {
    if (j >= lo && j <= hi) continue;
    const op *q = &g->ops[j];
    if (q->a == id || q->b == id || q->c == id || q->res == id || q->gt == id || q->qo == id || q->nout == id || q->nq == id)
      return 1;
  }
  return 0;
}

void graph_cap_exp_gu(graph *g, const model *m) {
  if (!m || m->arch != ARCH_INKLING2) return;
  const char *cap_env = getenv("IE_INK_CAP_EXP_GU");
  if (cap_env && cap_env[0] == '0') return;
  /* R41 (H40): On GPU, OP_REL_ATTN T=1 caps max_sp <= 64 (T>1 uses b->at_part), so shared scores needs at most
   * n_head * 64 * (hd + 2) * 4 B (0.76 MiB vs 3.00 MiB at 32K / 9.00 MiB at 98K), safe even with IE_INKLING_KERNELS=0. */
  {
    const uint32_t gpu_rel_sc = m->n_head * 64u * (m->hd + 2u) * 4u;
    for (uint32_t i = 0; i < g->n_ops; i++) {
      if (g->ops[i].kind != OP_REL_ATTN || g->ops[i].c < 0) continue;
      buf *sb = &g->bufs[g->ops[i].c];
      if (sb->stride == 0u && sb->size > gpu_rel_sc) sb->size = gpu_rel_sc;
    }
  }
  if (g->T <= 16u) return;
  const char *ink_env = getenv("IE_INKLING_KERNELS");
  if (ink_env && strcmp(ink_env, "0") == 0) return;
  const char *dmp = getenv("IE_DUMP_GPU_OPS");
  if (dmp && dmp[0] != '0') return;
  uint64_t mask = 0ULL, mask2 = 0ULL, mask3 = 0ULL;
  ie_ink_parse_mask(&mask, &mask2, &mask3);
  /* (1) Cap exp_gu when any fused prefill IQ3_XXS+SITU WMMA kernel is active (bits 57, 70, 102, 104). */
  const int has_iq3_situ_pf =
      !getenv("IE_GEMV_NOBATCH") && !getenv("IE_GEMM_RQ_EXP_NO") &&
      ((((mask  >> 57) & 1ULL) != 0ULL) ||
       (((mask2 >> (70u - 64u)) & 1ULL) != 0ULL) ||
       (((mask2 >> (102u - 64u)) & 1ULL) != 0ULL) ||
       (((mask2 >> (104u - 64u)) & 1ULL) != 0ULL));
  if (has_iq3_situ_pf) {
    for (uint32_t i = 0; i + 1u < g->n_ops; i++) {
      const op *o0 = &g->ops[i], *o1 = &g->ops[i + 1u];
      if (o0->kind != OP_GEMV_EXP || o1->kind != OP_SITU) continue;
      if (!o0->w || o0->w->kind != MAT_RAWQ || o0->w->sub != 18u) continue;
      if (o1->a != o0->b || o0->b < 0 || o1->aoff != 0u || o0->boff != 0u) continue;
      if (o1->nh != o0->nh || o1->b >= 0 || o1->qo < 0) continue;
      if (o1->n == 0u || (o1->n & 31u) != 0u || o0->w->rows != 2u * o1->n) continue;
      if (o0->w->nb < 8u || (o0->w->nb & 7u) != 0u) continue;
      if (g->bufs[o1->qo].stride != o1->nh * o1->bstr || (o1->bstr & 3u) != 0u) continue;
      if (o0->astr != 0u && g->bufs[o0->a].stride != o0->nh * o0->astr) continue;
      const int ye = o0->b;
      if (buf_used_outside(g, ye, i, i + 1u)) continue;
      buf *b = &g->bufs[ye];
      b->rows = 16u;
      b->size = b->stride * 16u;
    }
  }
  /* (2) Cap k_conv/k_clip/v_conv/v_norm to 16 rows when fused KV pipeline (bit 25 R2 or bit 75 R8) handles T >= 17. */
  const int kv_q_env = (getenv("IE_KV_Q8") && getenv("IE_KV_Q8")[0] != '0') ||
                       (getenv("IE_KV_Q4") && getenv("IE_KV_Q4")[0] != '0') ||
                       (getenv("IE_MEGAKERNEL") && getenv("IE_MEGAKERNEL")[0] != '0');
  const char *gkve = getenv("IE_INK_GLOB_KV");
  const int glob_kv_on = gkve && *gkve && strcmp(gkve, "f16") != 0 && strcmp(gkve, "0") != 0;
  const int has_kv_fuse = !kv_q_env && m->n_kv == 8u && m->hd == 128u &&
                          ((((mask >> 25) & 1ULL) != 0ULL) || (((mask2 >> (75u - 64u)) & 1ULL) != 0ULL));
  if (has_kv_fuse) {
    for (uint32_t i = 0; i + 4u < g->n_ops; i++) {
      const op *o0 = &g->ops[i];
      if (o0->kind != OP_SCONV || o0->n != 1024u) continue;
      uint32_t n_span = 0u;
      const op *okc = NULL, *ov = NULL, *ovc = NULL, *okv = NULL;
      if (g->ops[i + 1u].kind == OP_VCLIP && g->ops[i + 2u].kind == OP_SCONV &&
          g->ops[i + 3u].kind == OP_VCLIP && g->ops[i + 4u].kind == OP_KV) {
        okc = &g->ops[i + 1u]; ov = &g->ops[i + 2u]; ovc = &g->ops[i + 3u]; okv = &g->ops[i + 4u];
        n_span = 5u;
      } else if (i + 5u < g->n_ops && g->ops[i + 1u].kind == OP_VCLIP && g->ops[i + 2u].kind == OP_GEMV_Q8 &&
                 g->ops[i + 3u].kind == OP_SCONV && g->ops[i + 4u].kind == OP_VCLIP && g->ops[i + 5u].kind == OP_KV) {
        const op *ovd = &g->ops[i + 2u];
        okc = &g->ops[i + 1u]; ov = &g->ops[i + 3u]; ovc = &g->ops[i + 4u]; okv = &g->ops[i + 5u];
        if (ovd->b == ov->c && ovd->a != o0->b && ovd->a != okc->b) n_span = 6u;
      }
      if (!n_span || okc->a != o0->b || okc->aoff != 0u || ov->n != 1024u || ovc->a != ov->b || ovc->aoff != 0u) continue;
      if (okv->a != okc->b || okv->b != ovc->b || o0->escale != ov->escale) continue;
      if (!(okc->clamp > 0.0f) || okc->v || okc->norope || ovc->v || ovc->clamp != 0.0f) continue;
      if (g->kv_len[okv->kvi] == 0u) continue;
      if (glob_kv_on && g->kv_len[okv->kvi] >= g->n_ctx) continue;
      const int ids[4] = {o0->b, okc->b, ov->b, ovc->b};
      int ok = 1;
      for (int t = 0; t < 4; t++)
        if (ids[t] < 0 || buf_used_outside(g, ids[t], i, i + n_span - 1u)) { ok = 0; break; }
      if (!ok) continue;
      for (int t = 0; t < 4; t++) {
        buf *b = &g->bufs[ids[t]];
        b->rows = 16u;
        b->size = b->stride * 16u;
      }
    }
  }
  /* (3) Prepared Wave 10 H02 hook (inactive until bit 113 atomic-scatter IQ4_NL prefill is enabled): cap exp_down to 16 rows. */
  const int has_iq4_scatter =
      !getenv("IE_GEMV_NOBATCH") && !getenv("IE_GEMM_RQ_EXP_NO") &&
      (((mask2 >> (113u - 64u)) & 1ULL) != 0ULL);
  if (has_iq4_scatter) {
    for (uint32_t i = 0; i + 1u < g->n_ops; i++) {
      const op *o0 = &g->ops[i], *o1 = &g->ops[i + 1u];
      if (o0->kind != OP_GEMV_EXP || o1->kind != OP_MOE_SUM) continue;
      if (!o0->w || o0->w->kind != MAT_RAWQ || o0->w->sub != 20u) continue;
      if (o1->a != o0->b || o0->b < 0 || o1->aoff != 0u || o0->boff != 0u) continue;
      if (buf_used_outside(g, o0->b, i, i + 1u)) continue;
      buf *b = &g->bufs[o0->b];
      b->rows = 16u;
      b->size = b->stride * 16u;
    }
  }
}

/* A buffer that the tokens of a run share (stride 0): only for an op that
 * is launched once per token, one token after the other (attention scores). */
static int new_buf_shared(graph *g, const char *name, uint32_t bytes) {
  int id = new_buf(g, name, bytes);
  buf *b = &g->bufs[id];
  b->size = b->stride, b->stride = 0;
  return id;
}

static void touch(graph *g, int id, uint32_t t) {
  if (id < 0) return;
  buf *b = &g->bufs[id];
  if (t < b->first) b->first = t;
  if (t > b->last) b->last = t;
}

static op *emit(gb *B, int kind, int a, int b, int c) {
  graph *g = B->g;
  if (g->n_ops == B->cap_ops) {
    B->cap_ops = B->cap_ops ? 2 * B->cap_ops : 256;
    g->ops = realloc(g->ops, B->cap_ops * sizeof(op));
    if (!g->ops) ie_die("out of memory");
  }
  op *o = &g->ops[g->n_ops];
  memset(o, 0, sizeof *o);
  o->kind = kind, o->a = a, o->b = b, o->c = c, o->layer = B->layer, o->qo = -1, o->res = -1, o->nout = -1, o->nq = -1;
  o->gt = -1;
  touch(g, a, g->n_ops), touch(g, b, g->n_ops), touch(g, c, g->n_ops);
  g->n_ops++;
  return o;
}

/* The last emitted op also uses buffer id (as an input or output). */
static void use_last(gb *B, int id) {
  if (id >= 0) touch(B->g, id, B->g->n_ops - 1);
}

/* y = W x (+ bias) (+ res), where xq is the Q8 copy of x (or -1) and xf the
 * f32 x. bias may be NULL or empty; res is a buffer id or -1. The result goes
 * to buffer `into` at byte offset `off`, or (into < 0) to a new buffer. */
static int matvec_into(gb *B, const mat *w, int xf, int xq, int into, uint32_t off, const char *name, const vec *bias,
                       int res) {
  int y = into >= 0 ? into : new_buf(B->g, name, w->rows * 4);
  int base = w->lora ? new_buf(B->g, "lora_base", w->rows * 4) : y;
  const int quant_in = w->kind != MAT_F32;
  op *o = emit(B, w->kind == MAT_Q4 ? OP_GEMV_Q4 : w->kind == MAT_IQ4NL ? OP_GEMV_IQ4NL : w->kind == MAT_Q8 ? OP_GEMV_Q8
                   : w->kind == MAT_Q4K ? OP_GEMV_Q4K : w->kind == MAT_Q6K ? OP_GEMV_Q6K : w->kind == MAT_Q5K ? OP_GEMV_Q5K
                   : w->kind == MAT_IQ4XS ? OP_GEMV_IQ4XS : w->kind == MAT_RAWQ ? OP_GEMV_RAW
                   : OP_GEMV_F32,
               quant_in ? xq : xf, base, -1);
  o->w = w;
  o->n = w->cols;
  if (bias && bias->n) o->v = bias;
  o->res = res;
  o->boff = w->lora || into < 0 ? 0 : off;
  use_last(B, res);
  if (w->lora) {
    /* Apply the update to the unquantized activation, then add it to Wx. */
    int low = matvec_into(B, &w->lora->a, xf, -1, -1, 0, "lora_rank", NULL, -1);
    matvec_into(B, &w->lora->b, low, -1, y, off, name, NULL, base);
  }
  return y;
}
static int matvec(gb *B, const mat *w, int xf, int xq, const char *name, const vec *bias, int res) {
  return matvec_into(B, w, xf, xq, -1, 0, name, bias, res);
}

static int fuse_quant(gb *B, uint32_t n, const char *name);

/* The last op (a GEMV) also writes out = rmsnorm(its result) * nv and, if
 * quantized, its Q8 copy (a new buffer, returned in *q). */
static int last_out(const graph *g) {
  const op *p = &g->ops[g->n_ops - 1];
  /* OP_ADD writes its result to c (c = a + b), every other op to b (or a, in
   * place). Norming p->b after an add would norm one of the addends. */
  if (p->kind == OP_ADD) return p->c;
  return p->b;
}

static int fuse_norm(gb *B, const vec *nv, uint32_t n, const char *name, int quantized, const char *qname, int *q) {
  graph *g = B->g;
  /* By default the norm is a separate RMSNORM op: measured on the R9700, the
   * fused form (the last workgroup of the GEMV does the norm) is slower
   * (Qwen2.5-1.5B: 2.94 vs 2.81 ms/token). IE_NORM_FUSE=1 selects it. */
  const char *nf = getenv("IE_NORM_FUSE");
  const int k = g->ops[g->n_ops - 1].kind;
  const int gemv = k == OP_GEMV_Q4 || k == OP_GEMV_Q4K || k == OP_GEMV_Q6K || k == OP_GEMV_F32 || k == OP_GEMV_Q8;
  if (!(nf && nf[0] == '1') || !gemv) {
    int x = last_out(g), out = new_buf(g, name, n * 4);
    op *o = emit(B, OP_RMSNORM, x, out, -1);
    o->n = n, o->v = nv;
    *q = quantized ? fuse_quant(B, n, qname) : -1;
    return out;
  }
  op *o = &g->ops[g->n_ops - 1];
  int out = new_buf(g, name, n * 4);
  o->nv = nv;
  o->nout = out;
  use_last(B, out);
  *q = -1;
  if (quantized) {
    *q = new_buf(g, qname, (uint32_t)q8_bytes(n / 32));
    g->ops[g->n_ops - 1].nq = *q;
    use_last(B, *q);
  }
  return out;
}

/* A Q8 copy of the f32 result of the last op, written by that op (qo). */
static int fuse_quant(gb *B, uint32_t n, const char *name) {
  int q = new_buf(B->g, name, (uint32_t)q8_bytes(n / 32));
  B->g->ops[B->g->n_ops - 1].qo = q;
  use_last(B, q);
  return q;
}

static int quant(gb *B, int x, uint32_t n, const char *name) {
  int q = new_buf(B->g, name, (uint32_t)q8_bytes(n / 32));
  emit(B, OP_QUANT, x, q, -1)->n = n;
  return q;
}

static int any_q4(const mat *a, const mat *b, const mat *c) {
  return a->kind != MAT_F32 || (b && b->kind != MAT_F32) || (c && c->kind != MAT_F32);
}

/* Does the first GEMV of layer L read a Q8 copy of its input? */
static int layer_in_q8(const layer *L) {
  if (L->rec) return any_q4(&L->wlqkv, &L->wz, &L->wbeta) || L->walpha.kind != MAT_F32;
  return any_q4(&L->wq, &L->wk, &L->wv);
}

uint32_t ie_att_long_min(void) { return getenv("IE_ATTN_LONG_MIN") ? (uint32_t)atoi(getenv("IE_ATTN_LONG_MIN")) : 1u; }

/* Shape and environment of ie_attn_long: hd 256, up to 8 query heads per KV head, not the megakernel (IE_MK=1: its
 * attention is the older kernels), not an IE_ATTN debug mode, not IE_ATTN_LONG=0. */
int ie_att_long_ok(const model *m) {
  const char *al = getenv("IE_ATTN_LONG"), *mk = getenv("IE_MK");
  const uint32_t grp = m->n_kv ? m->n_head / m->n_kv : 0u;
  return !(al && al[0] == '0') && !getenv("IE_ATTN") && !(mk && mk[0] == '1') && m->arch != ARCH_FLINT && (m->hd == 256u || m->hd == 128u /* ie_attn_long128 */) && grp >= 1u &&
         grp <= 8u && m->n_head % m->n_kv == 0u;
}

uint32_t ie_att_part_floats(const model *m, uint32_t n_ctx) {
  uint32_t old_ctx = n_ctx, lng = 0;
  if (ie_att_long_ok(m)) { /* ie_attn_long takes the positions from ie_att_long_min() */
    if (ie_att_long_min() < n_ctx) old_ctx = ie_att_long_min();
    lng = m->n_head * ie_atl_parts(n_ctx) * IE_ATL_PSZ;
  }
  const uint32_t old = m->n_head * ie_att_max_split(old_ctx) * (m->hd + 2);
  return old > lng ? old : lng;
}

/* A full attention block: x + Wo attention(rmsnorm'd input xn / its Q8
 * copy xq). Returns the new x. */
static int attn_block(gb *B, const model *m, const layer *L, int x, int xn, int xq, uint32_t n_ctx) {
  graph *g = B->g;
  const uint32_t qd = m->n_head * m->hd, kvd = m->n_kv * m->hd;
  const int q35 = m->arch == ARCH_QWEN35;
  op *o;
    /* q, k, v in one buffer: [q | k | v] (qwen35: q is [q | gate] per head) */
    const uint32_t qrows = L->wq.rows;
    int qkv;
    if (L->wqkv.rows) {
      qkv = matvec(B, &L->wqkv, xn, xq, "qkv", &L->bqkv, -1);
    } else {
      qkv = new_buf(g, "qkv", (qrows + 2 * kvd) * 4);
      matvec_into(B, &L->wq, xn, xq, qkv, 0, "q", &L->bq, -1);
      matvec_into(B, &L->wk, xn, xq, qkv, qrows * 4, "k", &L->bk, -1);
      matvec_into(B, &L->wv, xn, xq, qkv, (qrows + kvd) * 4, "v", &L->bv, -1);
    }
    int qsrc = qkv;
    const int fl = m->arch == ARCH_FLINT;
    if (q35 || fl || m->qk_norm) { /* per head QK RMSNorm, RoPE (not for a flint norope layer); qwen35: q heads are [q | gate] */
      qsrc = new_buf(g, "q", qd * 4);
      o = emit(B, OP_QKN_ROPE_KV, qkv, qsrc, -1), o->n = qd, o->nh = m->n_head, o->kvi = L->kvi;
      o->v = &L->q_norm, o->nv = &L->k_norm, o->qstr = q35 ? 2 * m->hd : m->hd, o->norope = L->norope;
    } else {
      o = emit(B, OP_ROPE_KV, qkv, qkv, qkv), o->n = qd, o->nh = m->n_head, o->kvi = L->kvi;
      o->boff = qd * 4, o->coff = (qd + kvd) * 4;
    }
    int att = new_buf(g, "attn", qd * 4);
    const uint32_t sc_cpu = m->n_head * n_ctx, sc_gpu = ie_att_part_floats(m, n_ctx);
    int sc = new_buf_shared(g, "scores", (sc_cpu > sc_gpu ? sc_cpu : sc_gpu) * 4);
    o = emit(B, OP_ATTN, qsrc, att, sc), o->n = qd, o->kvi = L->kvi;
    o->win = L->swin, o->sk = L->sinks.n ? &L->sinks : NULL;
    if (q35) { /* out *= sigmoid(gate): the gate of head h is at h * 2 hd + hd */
      o->gt = qkv, o->gtoff = m->hd * 4, o->gstride = 2 * m->hd;
      use_last(B, qkv);
    }
    int aq = -1;
    if (L->wo.kind != MAT_F32) aq = m->hd % 32 == 0 && m->hd <= 256 ? fuse_quant(B, qd, "attn_q8") : quant(B, att, qd, "attn_q8");
    return matvec(B, &L->wo, att, aq, "x", NULL, x); /* x = x + Wo att */
}

/* The FFN block: x + Wdown (silu(Wgate h) * Wup h), h = rmsnorm(x) * ffn_norm. */
static int ffn_block(gb *B, const model *m, const layer *L, int x) {
  graph *g = B->g;
  op *o;
  int hq;
  int hn = fuse_norm(B, &L->ffn_norm, m->dim, "ffn_in", any_q4(&L->wgate, &L->wup, NULL), "ffn_in_q8", &hq);
  /* gate and up in one buffer: [gate | up] */
  int gu;
  if (L->wgu.rows) {
    gu = matvec(B, &L->wgu, hn, hq, "gate_up", NULL, -1);
  } else {
    gu = new_buf(g, "gate_up", 2 * m->ffn * 4);
    matvec_into(B, &L->wgate, hn, hq, gu, 0, "gate", NULL, -1);
    matvec_into(B, &L->wup, hn, hq, gu, m->ffn * 4, "up", NULL, -1);
  }
  o = emit(B, OP_SWIGLU, gu, gu, -1); /* gate = silu(gate) * up */
  o->n = m->ffn, o->boff = m->ffn * 4;
  int gq = L->wdown.kind != MAT_F32 ? fuse_quant(B, m->ffn, "ffn_mid_q8") : -1;
  use_last(B, hq); /* raw_swiglu_fuse reads hq and writes gq in one launch */
  return matvec(B, &L->wdown, gu, gq, "x", NULL, x); /* x = x + Wdown gate */
}

/* flint: the MoE block. x + shared(h) + sum over the n_used chosen experts
 * e of weight_e down_e(silu(gate_e h) * up_e h), h = rmsnorm(x) * ffn_norm. */
static int moe_block(gb *B, const model *m, const layer *L, int x) {
  graph *g = B->g;
  const uint32_t K = m->n_used, eff = m->eff, dim = m->dim;
  op *o;
  int hq;
  int hn = fuse_norm(B, &L->ffn_norm, dim, "ffn_in", 1, "ffn_in_q8", &hq);
  /* The order lets independent GEMVs share a phase of the megakernel
   * (hip.c mk_join): the router with the shared gate/up, the shared down with
   * the experts' gate/up. The router: logits (f32 weights). */
  int rl = matvec(B, &L->wr, hn, hq, "router", NULL, -1);
  int gu = -1, gq = -1, xs = x;
  if (m->sff) { /* the shared expert's gate and up */
    if (L->wgu.rows) {
      gu = matvec(B, &L->wgu, hn, hq, "sh_gate_up", NULL, -1);
    } else {
      gu = new_buf(g, "sh_gate_up", 2 * m->sff * 4);
      matvec_into(B, &L->wgate, hn, hq, gu, 0, "sh_gate", NULL, -1);
      matvec_into(B, &L->wup, hn, hq, gu, m->sff * 4, "sh_up", NULL, -1);
    }
    use_last(B, hn), use_last(B, rl); /* the router's input and output stay apart from gu: one megakernel phase */
    o = emit(B, OP_SWIGLU, gu, gu, -1), o->n = m->sff, o->boff = m->sff * 4;
    gq = L->wdown.kind != MAT_F32 ? fuse_quant(B, m->sff, "sh_mid_q8") : -1;
    use_last(B, hq);
  }
  /* the top K experts and their weights */
  int rt = new_buf(g, "route", 2 * K * 4);
  o = emit(B, OP_MOE_ROUTE, rl, rt, -1), o->n = m->n_exp, o->nh = K, o->v = L->ebias.n ? &L->ebias : NULL;
  /* the shared expert's down, with the residual: xs = x + down_s(silu(gate_s h) up_s h) */
  if (m->sff) xs = matvec(B, &L->wdown, gu, gq, "xs", NULL, x);
  /* the experts: gate and up of each chosen expert, [gate | up] per slot */
  const uint32_t gstr = 2 * eff * 4;
  int ye = new_buf(g, "exp_gu", K * gstr);
  const mat *parts[2] = {L->egu.rows ? &L->egu : &L->eg, L->egu.rows ? NULL : &L->eu};
  for (int i = 0; i < 2 && parts[i]; i++) {
    o = emit(B, OP_GEMV_EXP, parts[i]->kind == MAT_F32 ? hn : hq, ye, rt);
    o->w = parts[i], o->n = dim, o->nh = K, o->astr = 0, o->bstr = gstr, o->boff = i * eff * 4;
    /* the shared down's input, residual and output stay apart from ye: one megakernel phase */
    use_last(B, gq), use_last(B, x), use_last(B, xs);
  }
  /* a slot 256-aligned: a token's K slots fill its buffer stride (prompt chunks: pair p = t K + k at p qstr) */
  const uint32_t qstr = ((uint32_t)q8_bytes(eff / 32) + 255u) & ~255u;
  int mq = new_buf(g, "exp_mid_q8", K * qstr);
  o = emit(B, OP_MOE_SWIGLU, ye, -1, -1), o->n = eff, o->nh = K, o->astr = gstr, o->bstr = qstr, o->qo = mq;
  use_last(B, mq);
  int yd = new_buf(g, "exp_down", K * dim * 4);
  o = emit(B, OP_GEMV_EXP, mq, yd, rt), o->w = &L->ed, o->n = eff, o->nh = K, o->astr = qstr, o->bstr = dim * 4;
  if (L->ed.kind == MAT_F32) ie_die("flint: f32 down experts are not supported");
  int xo = new_buf(g, "x", dim * 4);
  o = emit(B, OP_MOE_SUM, yd, xo, rt), o->n = dim, o->nh = K, o->res = xs;
  use_last(B, xs);
  return xo;
}

/* The DFlash ops (see model.h): the context section, then the block. */
static void df_build(gb *B, const model *m) {
  graph *g = B->g;
  const dflash *d = m->df;
  const uint32_t dim = d->dim, qd = d->n_head * d->hd, kvd = d->n_kv * d->hd;
  op *o;
  B->layer = -1;
  g->i_dfc = g->n_ops;
  int fq = quant(B, g->df_feat, d->n_tl * dim, "df_feat_q8");
  int ch = matvec(B, &d->fc, g->df_feat, fq, "df_ctx", NULL, -1);
  int cn = new_buf(g, "df_ctxn", dim * 4);
  o = emit(B, OP_RMSNORM, ch, cn, -1), o->n = dim, o->v = &d->enc_norm;
  /* IE_DF_CAPTURE: the whole context input of the drafter, f32, row per
   * position: what a drafter trained outside the engine needs (gen.c:
   * df_capture). It costs no work when the variable is unset. */
  if (getenv("IE_DF_CAPTURE") && getenv("IE_DF_CAPTURE")[0]) {
    g->df_cap = new_buf(g, "df_cap", dim * 4);
    o = emit(B, OP_COPY, cn, g->df_cap, -1), o->n = dim;
  }
  int cq = fuse_quant(B, dim, "df_ctxn_q8");
  for (uint32_t il = 0; il < d->n_layer; il++) {
    const df_layer *L = &d->l[il];
    int kv = new_buf(g, "df_kv", 2 * kvd * 4);
    matvec_into(B, &L->wk, cn, cq, kv, 0, "df_k", NULL, -1);
    matvec_into(B, &L->wv, cn, cq, kv, kvd * 4, "df_v", NULL, -1);
    use_last(B, cn), use_last(B, cq);
    o = emit(B, OP_DF_QKV, kv, -1, -1), o->n = kvd, o->nh = 0, o->nv = &L->k_norm, o->layer = (int)il;
  }
  g->i_dfb = g->n_ops;
  int h = new_buf(g, "df_h", dim * 4);
  emit(B, OP_EMBED, h, -1, -1)->n = dim;
  for (uint32_t il = 0; il < d->n_layer; il++) {
    const df_layer *L = &d->l[il];
    /* attention */
    int xn = new_buf(g, "df_xn", dim * 4);
    o = emit(B, OP_RMSNORM, h, xn, -1), o->n = dim, o->v = &L->attn_norm;
    int xq = fuse_quant(B, dim, "df_xn_q8");
    int dyn = -1, xc = xn, xcq = xq;
    if (d->has_conv) {
      dyn = matvec(B, &L->aconv, xn, xq, "df_dyn", NULL, -1);
      xc = new_buf(g, "df_xc", dim * 4);
      o = emit(B, OP_DF_CONV, xn, xc, dyn), o->n = dim, o->nh = 0, o->v = &L->aconv_base;
      xcq = quant(B, xc, dim, "df_xc_q8");
    }
    int qkv = new_buf(g, "df_qkv", (qd + 2 * kvd) * 4);
    matvec_into(B, &L->wq, xc, xcq, qkv, 0, "df_q", NULL, -1);
    matvec_into(B, &L->wk, xc, xcq, qkv, qd * 4, "df_k", NULL, -1);
    matvec_into(B, &L->wv, xc, xcq, qkv, (qd + kvd) * 4, "df_v", NULL, -1);
    use_last(B, xc), use_last(B, xcq);
    o = emit(B, OP_DF_QKV, qkv, -1, -1), o->n = qd, o->nh = d->n_head, o->v = &L->q_norm, o->nv = &L->k_norm, o->layer = (int)il;
    int att = new_buf(g, "df_att", qd * 4);
    o = emit(B, OP_DF_ATTN, qkv, att, -1), o->n = qd, o->nh = d->n_head, o->layer = (int)il;
    int aq = quant(B, att, qd, "df_att_q8");
    int h2;
    if (d->has_conv) {
      int ob = matvec(B, &L->wo, att, aq, "df_o", NULL, -1);
      h2 = new_buf(g, "df_h", dim * 4);
      o = emit(B, OP_DF_CONV, ob, h2, dyn), o->n = dim, o->nh = 1, o->v = &L->aconv_base, o->res = h;
    } else
      h2 = matvec(B, &L->wo, att, aq, "df_h", NULL, h); /* h + wo att */
    use_last(B, h);
    h = h2;
    /* FFN */
    xn = new_buf(g, "df_xn", dim * 4);
    o = emit(B, OP_RMSNORM, h, xn, -1), o->n = dim, o->v = &L->ffn_norm;
    xq = fuse_quant(B, dim, "df_xn_q8");
    xc = xn, xcq = xq;
    if (d->has_conv) {
      dyn = matvec(B, &L->fconv, xn, xq, "df_dyn", NULL, -1);
      xc = new_buf(g, "df_xc", dim * 4);
      o = emit(B, OP_DF_CONV, xn, xc, dyn), o->n = dim, o->nh = 0, o->v = &L->fconv_base;
      xcq = quant(B, xc, dim, "df_xc_q8");
    }
    int gu;
    if (L->wgu.rows) {
      gu = matvec(B, &L->wgu, xc, xcq, "df_gu", NULL, -1);
    } else {
      gu = new_buf(g, "df_gu", 2 * d->ffn * 4);
      matvec_into(B, &L->wgate, xc, xcq, gu, 0, "df_gate", NULL, -1);
      matvec_into(B, &L->wup, xc, xcq, gu, d->ffn * 4, "df_up", NULL, -1);
      use_last(B, xc), use_last(B, xcq);
    }
    o = emit(B, OP_SWIGLU, gu, gu, -1), o->n = d->ffn, o->boff = d->ffn * 4;
    int gq = fuse_quant(B, d->ffn, "df_mid_q8");
    use_last(B, xcq);
    if (d->has_conv) {
      int dn = matvec(B, &L->wdown, gu, gq, "df_down", NULL, -1);
      h2 = new_buf(g, "df_h", dim * 4);
      o = emit(B, OP_DF_CONV, dn, h2, dyn), o->n = dim, o->nh = 1, o->v = &L->fconv_base, o->res = h;
    } else
      h2 = matvec(B, &L->wdown, gu, gq, "df_h", NULL, h); /* h + down(swiglu) */
    use_last(B, h);
    h = h2;
  }
  int hn = new_buf(g, "df_out", dim * 4);
  o = emit(B, OP_RMSNORM, h, hn, -1), o->n = dim, o->v = &d->out_norm;
  int hq = fuse_quant(B, dim, "df_out_q8");
  /* the output matrix (all tokens; one launch for the block): df_head (Q8_0, the draft vocabulary) or the target's.
   * Not the MTP draft head itself: MAT_Q4 has no multi-token GEMV, so a block read it once per row (slower) */
  g->df_logits = matvec(B, m->df_head.rows ? &m->df_head : &m->out, hn, hq, "df_logits", NULL, -1);
  g->df_sel = d->has_sel ? matvec(B, &d->sel_h, hn, hq, "df_sel", NULL, -1) : -1;
  use_last(B, hn), use_last(B, hq);
}

/* Ripple block (ricursive), see ricursive_layer: PeriNorm, value residual,
 * gated attention output. x0 = the scaled layer-0 embeddings, held for the
 * whole pass and read by every layer. The gate is a separate multiply (not
 * the attention kernels' gt, which the windowed path does not take). */
/* IE_SKIP_LAYERS=a,b,c-d (Ripple only, off unless set): those blocks are left out of the graph and the residual
 * passes through unchanged. For measuring a layer-skipping self-speculative draft (how often the shortened model
 * predicts the full model's token), not for generation. */
static int skip_layer(uint32_t l) {
  static uint8_t set[1024];
  static int init = 0;
  if (!init) {
    init = 1;
    const char *s = getenv("IE_SKIP_LAYERS");
    while (s && *s) {
      char *e;
      long a = strtol(s, &e, 10), b = a;
      if (e == s) break;
      if (*e == '-') b = strtol(e + 1, &e, 10);
      for (long i = a; i <= b && i < 1024; i++)
        if (i >= 0) set[i] = 1;
      s = *e == ',' ? e + 1 : e;
    }
  }
  return l < 1024u && set[l];
}

static int ric_block(gb *B, const model *m, const layer *L, uint32_t n_ctx, int x, int x0) {
  graph *g = B->g;
  const uint32_t dim = m->dim, qd = m->n_head * m->hd, kvd = m->n_kv * m->hd;
  op *o;
  /* h = rmsnorm(x) * attn_norm */
  int xn = new_buf(g, "attn_in", dim * 4);
  o = emit(B, OP_RMSNORM, x, xn, -1), o->n = dim, o->v = &L->attn_norm;
  int xq = layer_in_q8(L) ? fuse_quant(B, dim, "attn_in_q8") : -1;
  /* v_in = x0 + (xn - x0) * sigmoid(lam), with its Q8 copy for the wv GEMV */
  int vin = new_buf(g, "vres", dim * 4);
  o = emit(B, OP_VRES, xn, vin, x0), o->n = dim, o->lam = L->lam;
  int vq = L->wv.kind != MAT_F32 ? fuse_quant(B, dim, "vres_q8") : -1;
  /* q and k from the normed input, v from the mix */
  int qkv = new_buf(g, "qkv", (qd + 2 * kvd) * 4);
  matvec_into(B, &L->wq, xn, xq, qkv, 0, "q", NULL, -1);
  matvec_into(B, &L->wk, xn, xq, qkv, qd * 4, "k", NULL, -1);
  matvec_into(B, &L->wv, vin, vq, qkv, (qd + kvd) * 4, "v", NULL, -1);
  int qsrc = new_buf(g, "q", qd * 4);
  o = emit(B, OP_QKN_ROPE_KV, qkv, qsrc, -1), o->n = qd, o->nh = m->n_head, o->kvi = L->kvi;
  o->v = &L->q_norm, o->nv = &L->k_norm, o->qstr = m->hd, o->norope = L->norope;
  int gt = matvec(B, &L->wagt, xn, xq, "agate", NULL, -1);
  int att = new_buf(g, "attn", qd * 4);
  const uint32_t sc_cpu = m->n_head * n_ctx, sc_gpu = ie_att_part_floats(m, n_ctx);
  int sc = new_buf_shared(g, "scores", (sc_cpu > sc_gpu ? sc_cpu : sc_gpu) * 4);
  o = emit(B, OP_ATTN, qsrc, att, sc), o->n = qd, o->kvi = L->kvi, o->win = L->swin;
  o = emit(B, OP_GATE, att, gt, -1), o->n = qd; /* att *= sigmoid(g) */
  int aq = fuse_quant(B, qd, "attn_q8");
  matvec(B, &L->wo, att, aq, "attn_out", NULL, -1);
  /* x += post_attention_norm(y) */
  int pq;
  int pn = fuse_norm(B, &L->pan, dim, "post_attn", 0, NULL, &pq);
  int nx = new_buf(g, "x", dim * 4);
  o = emit(B, OP_ADD, x, pn, nx), o->n = dim;
  /* the FFN, with the clamp of gate and up */
  int hq;
  int hn = fuse_norm(B, &L->ffn_norm, dim, "ffn_in", 1, "ffn_in_q8", &hq);
  int gu = L->wgu.rows ? matvec(B, &L->wgu, hn, hq, "gate_up", NULL, -1) : 0;
  if (!L->wgu.rows) {
    gu = new_buf(g, "gate_up", 2 * m->ffn * 4);
    matvec_into(B, &L->wgate, hn, hq, gu, 0, "gate", NULL, -1);
    matvec_into(B, &L->wup, hn, hq, gu, m->ffn * 4, "up", NULL, -1);
  }
  o = emit(B, OP_SWIGLU, gu, gu, -1), o->n = m->ffn, o->boff = m->ffn * 4, o->clamp = L->fclamp;
  int mq = fuse_quant(B, m->ffn, "ffn_mid_q8");
  use_last(B, hq);
  matvec(B, &L->wdown, gu, mq, "ffn_out", NULL, -1);
  int dq;
  int dn = fuse_norm(B, &L->pffn, dim, "post_ffn", 0, NULL, &dq);
  int x2 = new_buf(g, "x", dim * 4);
  o = emit(B, OP_ADD, nx, dn, x2), o->n = dim;
  return x2;
}

static int ink_loop_ctrl_op(gb *B, const model *m, const ink_loop_ctrl *lc, int inp, uint32_t in_off, int msg,
                              uint32_t msg_off, const char *tag, int *h_out) {
  graph *g = B->g;
  const uint32_t dim = m->dim;
  op *o;
  int cat = new_buf(g, tag, 2u * dim * 4u);
  o = emit(B, OP_RMSNORM, inp, cat, -1), o->n = dim, o->aoff = in_off, o->v = &m->ones_dim;
  o = emit(B, OP_RMSNORM, msg, cat, -1), o->n = dim, o->aoff = msg_off, o->boff = dim * 4u, o->v = &m->ones_dim;
  int cq = lc->down.kind != MAT_F32 ? quant(B, cat, 2u * dim, "lc_q8") : -1;
  int h = matvec(B, &lc->down, cat, cq, "lc_h", NULL, -1);
  if (h_out) *h_out = h;
  const int need_hq = lc->wfg.rows ? (lc->wfg.kind != MAT_F32) : (lc->forget.kind != MAT_F32 || lc->gain.kind != MAT_F32);
  int hq = need_hq ? quant(B, h, m->loop_rank, "lc_hq") : -1;
  int fg;
  if (lc->wfg.rows) {
    fg = matvec(B, &lc->wfg, h, hq, "lc_fg", NULL, -1);
  } else {
    fg = new_buf(g, "lc_fg", 2u * dim * 4u);
    matvec_into(B, &lc->forget, h, lc->forget.kind != MAT_F32 ? hq : -1, fg, 0, "lc_f", NULL, -1);
    int gtmp = matvec(B, &lc->gain, h, lc->gain.kind != MAT_F32 ? hq : -1, "lc_g", NULL, -1);
    o = emit(B, OP_COPY, gtmp, fg, -1), o->n = dim, o->boff = dim * 4u;
  }
  int out = new_buf(g, "lc_out", dim * 4u);
  o = emit(B, OP_LOOP_GATE, inp, out, msg);
  o->n = dim, o->aoff = in_off, o->coff = msg_off, o->gt = fg, o->gtoff = 0;
  use_last(B, fg);
  return out;
}

static int ink_norm(gb *B, uint32_t dim, int src, uint32_t off, const vec *nv, int need_q8, const char *name, const char *qname,
                    int *q_out) {
  int out = new_buf(B->g, name, dim * 4u);
  op *o = emit(B, OP_RMSNORM, src, out, -1);
  o->n = dim, o->aoff = off, o->v = nv;
  *q_out = need_q8 ? fuse_quant(B, dim, qname) : -1;
  return out;
}

static int inkling_build(gb *B, const model *m, uint32_t n_ctx, int x, int *blocks_out, uint32_t *n_blk_out) {
  graph *g = B->g;
  const uint32_t dim = m->dim, qd = m->n_head * m->hd, kvd = m->n_kv * m->hd;
  const uint32_t bs = m->attn_res_bs ? m->attn_res_bs : 6u;
  const uint32_t max_blk = (m->n_layer + bs - 1u) / bs + 2u;
  if (max_blk > 64u) ie_die("inkling2: too many attn_res blocks (%u)", max_blk);
  op *o;

  int xt = new_buf(g, "tok_n", dim * 4u);
  o = emit(B, OP_RMSNORM, x, xt, -1), o->n = dim, o->v = &m->tok_norm;
  x = xt;

  int blocks_buf = new_buf(g, "blocks", max_blk * dim * 4u);
  int blk_bufs[64];
  uint32_t n_blk = 0, pre_n_blk = 0, n_bank = 0;
  int pre_cur = -1, bank_buf = -1;
  const float inv_s5 = 1.0f / sqrtf(5.0f), inv_s9 = 1.0f / 3.0f;

  for (uint32_t il = 0; il < m->n_visits; il++) {
    if (il < 64u) g->visit_start[il] = g->n_ops;
    uint32_t ip = 0;
    const int pass = ink_phys(m, il, &ip);
    const layer *L = &m->l[ip];
    B->layer = (int)ip;

    if (m->n_loops > 1u && ip == m->loop_start) {
      if (pass == 0) {
        pre_n_blk = n_blk;
        pre_cur = x;
      } else if (pass > 0) {
        const uint32_t bank_start = pre_n_blk + 1u;
        const uint32_t bank_prev = n_blk > bank_start ? n_blk - bank_start : 0u;
        n_bank = bank_prev + 1u;
        bank_buf = new_buf(g, "loop_bank", n_bank * dim * 4u);
        for (uint32_t k = 0; k < bank_prev; k++) {
          o = emit(B, OP_COPY, blk_bufs[bank_start + k], bank_buf, -1);
          o->n = dim, o->boff = k * dim * 4u;
        }
        o = emit(B, OP_COPY, x, bank_buf, -1);
        o->n = dim, o->boff = bank_prev * dim * 4u;
        n_blk = pre_n_blk;
        x = pre_cur;
      }
    }

    if (g->ds_feat >= 0 && m->ds) {
      for (uint32_t j = 0; j < m->ds->n_tl; j++)
        if (m->ds->tl[j] == il) {
          o = emit(B, OP_COPY, x, g->ds_feat, -1);
          o->n = dim, o->boff = j * dim * 4u;
        }
    }

    int msg_qm = -1, msg_kv = -1;
    if (pass > 0) {
      const uint32_t li = ip - m->loop_start;
      msg_qm = new_buf(g, "msg_qm", 2u * dim * 4u);
      o = emit(B, OP_ATTN_RES, bank_buf, msg_qm, -1);
      o->n = dim, o->nh = n_bank, o->qstr = 2u, o->v = &m->loop_res_q, o->roff = 2u * li * dim;
      msg_kv = new_buf(g, "msg_kv", 2u * dim * 4u);
      o = emit(B, OP_ATTN_RES, bank_buf, msg_kv, -1);
      o->n = dim, o->nh = n_bank, o->qstr = 2u, o->v = &m->loop_kv_res_q, o->roff = 2u * li * dim;
    }

    const int block_res = x;
    int qi = x, ki = x, vi = x;
    uint32_t qoff = 0, koff = 0, voff = 0;
    if (ip > 0) {
      if (!L->is_full) {
        int ar = new_buf(g, "attn_res", dim * 4u);
        o = emit(B, OP_ATTN_RES, n_blk ? blocks_buf : -1, ar, x);
        o->n = dim, o->nh = n_blk + 1u, o->qstr = 1u, o->v = &L->attn_res_q;
        qi = ki = vi = ar;
      } else {
        int ar = new_buf(g, "attn_res3", 3u * dim * 4u);
        o = emit(B, OP_ATTN_RES, n_blk ? blocks_buf : -1, ar, x);
        o->n = dim, o->nh = n_blk + 1u, o->qstr = 3u, o->v = &L->attn_res_q;
        qi = ki = vi = ar;
        koff = dim * 4u, voff = 2u * dim * 4u;
      }
    }

    int lc_h[3] = {-1, -1, -1}, lc_inps[3] = {-1, -1, -1}, lc_msgs[3] = {-1, -1, -1};
    int lc_phased = 0;
    if (pass > 0) {
      const int can_phase = L->loop[0].down.kind != MAT_F32 && L->loop[1].down.kind != MAT_F32 && L->loop[2].down.kind != MAT_F32 &&
                            L->loop[0].wfg.rows && L->loop[1].wfg.rows && L->loop[2].wfg.rows &&
                            L->loop[0].wfg.kind != MAT_F32 && L->loop[1].wfg.kind != MAT_F32 && L->loop[2].wfg.kind != MAT_F32;
      if (can_phase) {
        const int inps[3] = {qi, ki, vi};
        const uint32_t in_offs[3] = {qoff, koff, voff};
        const int msgs[3] = {msg_qm, msg_kv, msg_kv};
        const uint32_t msg_offs[3] = {0u, 0u, dim * 4u};
        const char *const tags[3] = {"lc_q", "lc_k", "lc_v"};
        int cats[3], cqs[3], outs[3];
        /* Phase 1: all 3 [RMSNORM(ones), RMSNORM(ones), QUANT(2*dim)] -> cqs[0..2] */
        for (int c = 0; c < 3; c++) {
          cats[c] = new_buf(g, tags[c], 2u * dim * 4u);
          o = emit(B, OP_RMSNORM, inps[c], cats[c], -1), o->n = dim, o->aoff = in_offs[c], o->v = &m->ones_dim;
          o = emit(B, OP_RMSNORM, msgs[c], cats[c], -1), o->n = dim, o->aoff = msg_offs[c], o->boff = dim * 4u, o->v = &m->ones_dim;
          cqs[c] = quant(B, cats[c], 2u * dim, "lc_q8");
        }
        /* Phase 2: all 3 down GEMV_Q8 -> lc_h[0..2] (keep cqs[0..2] disjoint across 3-way down launch) */
        for (int c = 0; c < 3; c++)
          lc_h[c] = matvec(B, &L->loop[c].down, cats[c], cqs[c], "lc_h", NULL, -1);
        for (int c = 0; c < 3; c++) use_last(B, cqs[c]);
        /* Phase 3: all 3 [QUANT(64), GEMV_Q8(wfg), LOOP_GATE] -> outs[0..2] */
        for (int c = 0; c < 3; c++) {
          int hq_c = quant(B, lc_h[c], m->loop_rank, "lc_hq");
          int fg_c = matvec(B, &L->loop[c].wfg, lc_h[c], hq_c, "lc_fg", NULL, -1);
          outs[c] = new_buf(g, "lc_out", dim * 4u);
          o = emit(B, OP_LOOP_GATE, inps[c], outs[c], msgs[c]);
          o->n = dim, o->aoff = in_offs[c], o->coff = msg_offs[c], o->gt = fg_c, o->gtoff = 0;
          use_last(B, fg_c);
        }
        for (int c = 0; c < 3; c++) lc_inps[c] = inps[c], lc_msgs[c] = msgs[c];
        qi = outs[0], ki = outs[1], vi = outs[2];
        qoff = koff = voff = 0u;
        lc_phased = 1;
      } else {
        qi = ink_loop_ctrl_op(B, m, &L->loop[0], qi, qoff, msg_qm, 0, "lc_q", NULL), qoff = 0;
        ki = ink_loop_ctrl_op(B, m, &L->loop[1], ki, koff, msg_kv, 0, "lc_k", NULL), koff = 0;
        vi = ink_loop_ctrl_op(B, m, &L->loop[2], vi, voff, msg_kv, dim * 4u, "lc_v", NULL), voff = 0;
      }
    }

    const int need_qq = (L->wqkv.rows ? L->wqkv.kind != MAT_F32 : L->wq.kind != MAT_F32) || L->wagt.kind != MAT_F32 ||
                        L->wrel.kind != MAT_F32 || L->wsink.kind != MAT_F32;
    const int need_kq = (L->wqkv.rows ? L->wqkv.kind != MAT_F32 : L->wk.kind != MAT_F32) ||
                        (L->is_full && L->k_conv_proj.kind != MAT_F32);
    const int need_vq = (L->wqkv.rows ? L->wqkv.kind != MAT_F32 : L->wv.kind != MAT_F32) ||
                        (L->is_full && L->v_conv_proj.kind != MAT_F32);

    int qn, qq, kn, kq, vn, vq;
    if (!L->is_full && pass <= 0) {
      qn = ink_norm(B, dim, qi, qoff, &L->attn_norm, need_qq || need_kq || need_vq, "attn_in", "attn_in_q8", &qq);
      kn = qn, kq = qq, vn = qn, vq = qq;
    } else {
      const vec *qn_w = L->is_full ? &L->q_inp_norm : &L->attn_norm;
      const vec *kn_w = L->is_full ? &L->k_inp_norm : &L->attn_norm;
      const vec *vn_w = L->is_full ? &L->v_inp_norm : &L->attn_norm;
      qn = ink_norm(B, dim, qi, qoff, qn_w, need_qq, "q_in", "q_in_q8", &qq);
      kn = ink_norm(B, dim, ki, koff, kn_w, need_kq, "k_in", "k_in_q8", &kq);
      vn = ink_norm(B, dim, vi, voff, vn_w, need_vq, "v_in", "v_in_q8", &vq);
      if (lc_phased) {
        /* Keep inputs & tmp outputs disjoint from q/k/v_in[_q8] for fused 3-way wfg_gate_norm_r6 */
        for (int c = 0; c < 3; c++) use_last(B, lc_h[c]), use_last(B, lc_inps[c]), use_last(B, lc_msgs[c]);
        use_last(B, qi), use_last(B, ki), use_last(B, vi);
      }
    }

    int q_raw, k_raw, v_raw;
    uint32_t qr_off = 0, kr_off = 0, vr_off = 0;
    if (L->wqkv.rows && pass <= 0 && !L->is_full) {
      int qkv = matvec(B, &L->wqkv, qn, qq, "qkv", NULL, -1);
      q_raw = k_raw = v_raw = qkv;
      kr_off = qd * 4u, vr_off = (qd + kvd) * 4u;
    } else {
      q_raw = matvec(B, &L->wq, qn, qq, "q", NULL, -1);
      k_raw = matvec(B, &L->wk, kn, kq, "k", NULL, -1);
      v_raw = matvec(B, &L->wv, vn, vq, "v", NULL, -1);
    }

    const uint32_t sc_base = il * (2u * kvd + 2u * dim);
    int q_clip = new_buf(g, "q_clip", qd * 4u);
    o = emit(B, OP_VCLIP, q_raw, q_clip, -1);
    o->n = qd, o->nh = m->n_head, o->qstr = m->hd, o->aoff = qr_off, o->v = &L->q_clip;
    o->clamp = m->qk_clip, o->norope = L->is_full, o->escale = m->log_scale_alpha, o->win = (uint32_t)m->log_scale_floor;

    int k_dyn = L->is_full ? matvec(B, &L->k_conv_proj, kn, kq, "k_dyn", NULL, -1) : -1;
    int v_dyn = L->is_full ? matvec(B, &L->v_conv_proj, vn, vq, "v_dyn", NULL, -1) : -1;
    if (kq >= 0) use_last(B, kq);
    if (vq >= 0) use_last(B, vq);
    if (L->is_full) use_last(B, q_raw);
    int k_conv = new_buf(g, "k_conv", kvd * 4u);
    o = emit(B, OP_SCONV, k_raw, k_conv, k_dyn);
    o->n = kvd, o->nh = m->n_kv, o->aoff = kr_off, o->v = &L->k_conv, o->scoff = sc_base;
    o->escale = L->is_full ? inv_s9 : inv_s5;
    int k_clip = new_buf(g, "k_clip", kvd * 4u);
    o = emit(B, OP_VCLIP, k_conv, k_clip, -1);
    o->n = kvd, o->nh = m->n_kv, o->qstr = m->hd, o->clamp = m->qk_clip;

    int v_conv = new_buf(g, "v_conv", kvd * 4u);
    o = emit(B, OP_SCONV, v_raw, v_conv, v_dyn);
    o->n = kvd, o->nh = m->n_kv, o->aoff = vr_off, o->v = &L->v_conv, o->scoff = sc_base + kvd;
    o->escale = L->is_full ? inv_s9 : inv_s5;
    int v_norm = new_buf(g, "v_norm", kvd * 4u);
    o = emit(B, OP_VCLIP, v_conv, v_norm, -1);
    o->n = kvd, o->nh = m->n_kv, o->qstr = m->hd, o->clamp = 0.0f, o->escale = m->eps;

    o = emit(B, OP_KV, k_clip, v_norm, -1), o->n = kvd, o->kvi = il;

    int r_raw = matvec(B, &L->wrel, qn, qq, "r_raw", NULL, -1);
    int r_clip = new_buf(g, "r_clip", m->n_head * m->d_rel * 4u);
    o = emit(B, OP_VCLIP, r_raw, r_clip, -1);
    o->n = m->n_head * m->d_rel, o->nh = m->n_head, o->qstr = m->d_rel, o->v = &L->rel_clip;
    o->clamp = m->rel_clip, o->norope = L->is_full, o->escale = m->log_scale_alpha, o->win = (uint32_t)m->log_scale_floor;

    int sink = matvec(B, &L->wsink, qn, qq, "sink", NULL, -1);
    int agate = matvec(B, &L->wagt, qn, qq, "agate", NULL, -1);

    int att = new_buf(g, "attn", qd * 4u);
    const uint32_t sc_cpu = m->n_head * n_ctx, sc_gpu = ie_att_part_floats(m, n_ctx);
    int sc = new_buf_shared(g, "scores", (sc_cpu > sc_gpu ? sc_cpu : sc_gpu) * 4u);
    o = emit(B, OP_REL_ATTN, q_clip, att, sc);
    o->n = qd, o->nh = m->n_head, o->qstr = m->d_rel, o->v = &L->rel_bias;
    o->kvi = il, o->win = L->swin, o->nq = r_clip, o->res = sink;
    use_last(B, r_clip), use_last(B, sink);

    o = emit(B, OP_GATE, att, agate, -1), o->n = qd;
    int aq = L->wo.kind != MAT_F32 ? fuse_quant(B, qd, "attn_q8") : -1;
    if (aq >= 0) {
      /* Wave 11 H15 (bit 110): keep REL_ATTN inputs & scores disjoint from attn_q8 for fused [REL_ATTN, GATE(qo)] */
      use_last(B, q_clip), use_last(B, r_clip), use_last(B, sink), use_last(B, sc), use_last(B, agate);
    }
    int aout = matvec(B, &L->wo, att, aq, "attn_out", NULL, -1);
    int apc = new_buf(g, "attn_pc", dim * 4u);
    o = emit(B, OP_SCONV, aout, apc, -1);
    o->n = dim, o->v = &L->attn_pconv, o->scoff = sc_base + 2u * kvd, o->escale = inv_s5;

    if (ip % bs == 0u) {
      o = emit(B, OP_COPY, block_res, blocks_buf, -1);
      o->n = dim, o->boff = n_blk * dim * 4u;
      blk_bufs[n_blk++] = block_res;
      x = apc;
    } else {
      int x1 = new_buf(g, "x_attn", dim * 4u);
      o = emit(B, OP_ADD, block_res, apc, x1), o->n = dim;
      use_last(B, aout);
      x = x1;
    }

    const int ffn_inp = x;
    int fres = new_buf(g, "ffn_res", dim * 4u);
    o = emit(B, OP_ATTN_RES, n_blk ? blocks_buf : -1, fres, ffn_inp);
    o->n = dim, o->nh = n_blk + 1u, o->qstr = 1u, o->v = &L->ffn_res_q;
    int mlp_h = -1;
    int minp = pass > 0 ? ink_loop_ctrl_op(B, m, &L->loop[3], fres, 0, msg_qm, dim * 4u, "lc_mlp", &mlp_h) : fres;

    int hq = -1;
    const mat *parts[2] = {L->egu.rows ? &L->egu : &L->eg, L->egu.rows ? NULL : &L->eu};
    const int need_ffn_q8 = parts[0]->kind != MAT_F32 || (parts[1] && parts[1]->kind != MAT_F32);
    int hffn = ink_norm(B, dim, minp, 0, &L->ffn_norm, need_ffn_q8, "ffn_in", "ffn_in_q8", &hq);
    if (pass > 0 && mlp_h >= 0) use_last(B, fres), use_last(B, msg_qm), use_last(B, mlp_h), use_last(B, minp);
    int rl = matvec(B, &L->wr, hffn, hq, "router", NULL, -1);
    int rt = new_buf(g, "route", 2u * m->n_used * 4u);
    o = emit(B, OP_MOE_ROUTE, rl, rt, -1), o->n = m->n_exp, o->nh = m->n_used, o->v = L->ebias.n ? &L->ebias : NULL;

    const uint32_t eff = m->eff, K = m->n_used, gstr = 2u * eff * 4u;
    int ye = new_buf(g, "exp_gu", K * gstr);
    for (int i = 0; i < 2 && parts[i]; i++) {
      o = emit(B, OP_GEMV_EXP, parts[i]->kind == MAT_F32 ? hffn : hq, ye, rt);
      o->w = parts[i], o->n = dim, o->nh = K, o->astr = 0, o->bstr = gstr, o->boff = (uint32_t)i * eff * 4u;
    }
    const uint32_t qstr = ((uint32_t)q8_bytes(eff / 32u) + 255u) & ~255u;
    int mq = L->ed.kind != MAT_F32 ? new_buf(g, "exp_mid_q8", K * qstr) : -1;
    o = emit(B, OP_SITU, ye, -1, -1);
    o->n = eff, o->nh = K, o->astr = gstr, o->bstr = qstr, o->qo = mq, o->clamp = m->situ_beta, o->lam = m->situ_lbeta;
    use_last(B, mq);
    use_last(B, hq); /* keep ffn_in_q8 disjoint from exp_mid_q8 for fused GEMV_EXP+SITU */

    int yd = new_buf(g, "exp_down", K * dim * 4u);
    o = emit(B, OP_GEMV_EXP, L->ed.kind == MAT_F32 ? ye : mq, yd, rt);
    o->w = &L->ed, o->n = eff, o->nh = K, o->astr = L->ed.kind == MAT_F32 ? gstr : qstr, o->bstr = dim * 4u;

    int ysum = new_buf(g, "exp_sum", dim * 4u);
    o = emit(B, OP_MOE_SUM, yd, ysum, rt), o->n = dim, o->nh = K, o->res = -1;

    int fpc = new_buf(g, "ffn_pc", dim * 4u);
    o = emit(B, OP_SCONV, ysum, fpc, -1);
    o->n = dim, o->v = &L->ffn_pconv, o->scoff = sc_base + 2u * kvd + dim, o->escale = inv_s5;

    int x2 = new_buf(g, "x", dim * 4u);
    o = emit(B, OP_ADD, ffn_inp, fpc, x2), o->n = dim;
    if (ink_moe_cache_env() && (ip < m->loop_start || ip >= m->loop_end)) use_last(B, hq);
    use_last(B, mq); use_last(B, yd); use_last(B, rt); use_last(B, ysum);
    x = x2;
  }
  g->n_visits_rec = m->n_visits < 64u ? m->n_visits : 64u;

  *blocks_out = blocks_buf;
  *n_blk_out = n_blk;
  return x;
}

void graph_build(graph *g, const model *m, uint32_t n_ctx, uint32_t T, int mtp) {
  memset(g, 0, sizeof *g);
  g->bufs = calloc(MAX_BUFS, sizeof(buf));
  g->n_ctx = n_ctx;
  g->T = T ? T : 1;
  g->h_out = g->mtp_h = g->mtp_g = g->mtp_logits = g->mtp_argmax = -1;
  g->mtp_r = g->mtp_a = g->mtp_y = -1;
  g->i_mtp_wq = -1;
  g->df_feat = g->df_logits = g->df_sel = g->df_cap = -1;
  g->ds_feat = -1;
  gb B = {g, 0, -1};
  uint32_t dim = m->dim;
  if (mtp && !m->has_mtp && !m->df && !m->ds) ie_die("the model has no MTP head");
  const int df = mtp && m->df;
  const int ds = mtp && m->ds;
  if (df) g->df_feat = new_buf(g, "df_feat", m->df->n_tl * dim * 4);
  if (ds) g->ds_feat = new_buf(g, "ds_feat", m->ds->n_tl * dim * 4);

  int x = new_buf(g, "x", dim * 4);
  op *o = emit(&B, OP_EMBED, x, -1, -1);
  o->n = dim;
  const int ric = m->arch == ARCH_RICURSIVE, ink = m->arch == ARCH_INKLING2;
  if (ric) o->escale = m->emb_scale; /* the value residual reads the scaled embeddings */
  const int x0 = x;                  /* Ripple: the embedding row every layer mixes into v */
  /* the norm of layer 0; the later norms are fused into the residual GEMVs */
  B.layer = 0;
  int xn = -1, xq = -1;
  if (ink) {
    int blocks_buf = -1;
    uint32_t n_blk = 0;
    x = inkling_build(&B, m, n_ctx, x, &blocks_buf, &n_blk);
    B.layer = -1;
    int hres = new_buf(g, "out_res", dim * 4u);
    o = emit(&B, OP_ATTN_RES, n_blk ? blocks_buf : -1, hres, x);
    o->n = dim, o->nh = n_blk + 1u, o->qstr = 1u, o->v = &m->out_res_q;
    xn = fuse_norm(&B, &m->out_norm, dim, "out_in", m->out.kind != MAT_F32, "out_in_q8", &xq);
    use_last(&B, x);
    if (ds) {
      for (uint32_t j = 0; j < m->ds->n_tl; j++)
        if (m->ds->tl[j] == m->n_visits) {
          o = emit(&B, OP_COPY, xn, g->ds_feat, -1);
          o->n = dim, o->boff = j * dim * 4u;
        }
    }
  } else {
  if (!ric) {
    xn = new_buf(g, "attn_in", dim * 4);
    o = emit(&B, OP_RMSNORM, x, xn, -1);
    o->n = dim, o->v = &m->l[0].attn_norm;
    xq = layer_in_q8(&m->l[0]) ? fuse_quant(&B, dim, "attn_in_q8") : -1;
  }
  for (uint32_t l = 0; l < m->n_layer; l++) {
    const layer *L = &m->l[l];
    B.layer = (int)l;
    if (ric) {
      if (!skip_layer(l)) x = ric_block(&B, m, L, n_ctx, x, x0);
      continue;
    }
    if (L->rec) {
      /* linear attention: [q k v (conv_dim) | z | beta | alpha] in one buffer */
      const uint32_t inner = m->n_vh * m->sd, cd = m->conv_dim;
      int lin;
      if (L->win.rows) {
        lin = matvec(&B, &L->win, xn, xq, "lin", NULL, -1);
      } else {
        lin = new_buf(g, "lin", (cd + inner + 2 * m->n_vh) * 4);
        if (L->wlz.rows) {
          matvec_into(&B, &L->wlz, xn, xq, lin, 0, "lz", NULL, -1);
        } else {
          matvec_into(&B, &L->wlqkv, xn, xq, lin, 0, "lqkv", NULL, -1);
          matvec_into(&B, &L->wz, xn, xq, lin, cd * 4, "z", NULL, -1);
        }
        matvec_into(&B, &L->wbeta, xn, xq, lin, (cd + inner) * 4, "beta", NULL, -1);
        if (L->walpha.rows) matvec_into(&B, &L->walpha, xn, xq, lin, (cd + inner + m->n_vh) * 4, "alpha", NULL, -1);
      }
      int go = new_buf(g, "gdn", inner * 4);
      o = emit(&B, OP_GDN, lin, go, -1), o->n = inner;
      int gq = L->wo.kind != MAT_F32 ? fuse_quant(&B, inner, "gdn_q8") : -1;
      x = matvec(&B, &L->wo, go, gq, "x", NULL, x); /* x = x + Wout gdn */
    } else {
    x = attn_block(&B, m, L, x, xn, xq, n_ctx);
    }
    x = L->moe ? moe_block(&B, m, L, x) : ffn_block(&B, m, L, x);
    if (l + 1 < m->n_layer) {
      const layer *N = &m->l[l + 1];
      xn = fuse_norm(&B, &N->attn_norm, dim, "attn_in", layer_in_q8(N), "attn_in_q8", &xq);
    } else {
      xn = fuse_norm(&B, &m->out_norm, dim, "out_in", m->out.kind != MAT_F32, "out_in_q8", &xq);
    }
    if (df) /* DFlash: the outputs of the target layers, side by side */
      for (uint32_t j = 0; j < m->df->n_tl; j++)
        if (m->df->tl[j] == l) {
          o = emit(&B, OP_COPY, x, g->df_feat, -1);
          o->n = dim, o->boff = j * dim * 4;
        }
  }
  B.layer = -1;
  if (ric) xn = fuse_norm(&B, &m->out_norm, dim, "out_in", m->out.kind != MAT_F32, "out_in_q8", &xq);
  }
  g->i_head = g->n_ops; /* the output matrix and the argmax: ops i_head .. n_main - 1 */
  g->logits = matvec(&B, &m->out, xn, xq, "logits", NULL, -1);
  const uint32_t v_unpad = ink && m->vocab_unpadded ? m->vocab_unpadded : m->vocab;
  if (ink && (m->logit_cap > 0.0f || v_unpad < m->vocab)) {
    o = emit(&B, OP_LOGIT_CAP, g->logits, g->logits, -1);
    o->n = m->vocab, o->nh = v_unpad, o->clamp = m->logit_cap;
  }
  g->argmax = new_buf(g, "argmax", 8); /* [index, probability (float bits)] */
  emit(&B, OP_ARGMAX, g->logits, g->argmax, -1)->n = v_unpad;
  g->h_out = xn;
  g->n_main = g->n_ops;

  if (mtp && m->has_mtp) { /* the MTP head (see model.h); a DFlash drafter alone has none */
    const layer *L = &m->l[m->n_layer];
    B.layer = (int)m->n_layer;
    g->mtp_h = new_buf(g, "mtp_h", dim * 4);
    int e = new_buf(g, "mtp_e", dim * 4);
    o = emit(&B, OP_EMBED, e, -1, -1), o->n = dim;
    if (m->mtp_tok.rows) o->w = &m->mtp_tok; /* IE_MTP_EMBED */
    int cat = new_buf(g, "mtp_cat", 2 * dim * 4); /* [enorm(e) | hnorm(h)] */
    o = emit(&B, OP_RMSNORM, e, cat, -1), o->n = dim, o->v = &m->enorm;
    o = emit(&B, OP_RMSNORM, g->mtp_h, cat, -1), o->n = dim, o->v = &m->hnorm, o->boff = dim * 4;
    int catq = m->eh.kind != MAT_F32 ? quant(&B, cat, 2 * dim, "mtp_cat_q8") : -1;
    int x2 = matvec(&B, &m->eh, cat, catq, "mtp_x", NULL, -1);
    int xn2 = new_buf(g, "mtp_attn_in", dim * 4);
    o = emit(&B, OP_RMSNORM, x2, xn2, -1), o->n = dim, o->v = &L->attn_norm;
    int xq2 = layer_in_q8(L) ? fuse_quant(&B, dim, "mtp_attn_in_q8") : -1;
    x2 = attn_block(&B, m, L, x2, xn2, xq2, n_ctx);
    const uint32_t i_ffn = g->n_ops;
    const int x_ffn = x2;
    x2 = ffn_block(&B, m, L, x2);
    if (getenv("IE_MTP_CAPTURE") || ie_mtp_keep) { /* the FFN input and the ffn_down input, read by prefill */
      g->mtp_r = x_ffn, g->mtp_y = x2;
      for (uint32_t i = i_ffn; i < g->n_ops; i++)
        if (g->ops[i].kind == OP_SWIGLU) g->mtp_a = g->ops[i].a;
      if (g->mtp_a < 0) ie_die("IE_MTP_CAPTURE: no SWIGLU in the MTP FFN");
    }
    g->mtp_g = new_buf(g, "mtp_g", dim * 4);
    o = emit(&B, OP_RMSNORM, x2, g->mtp_g, -1), o->n = dim, o->v = &m->head_norm;
    const mat *md = m->out_draft.rows ? &m->out_draft : &m->out;
    int gq2 = md->kind != MAT_F32 ? fuse_quant(&B, dim, "mtp_g_q8") : -1;
    B.layer = -1;
    g->i_mtp_head = g->n_ops;
    g->i_mtp_kv = g->i_mtp_head;
    if (g->mtp_a < 0) {
      for (uint32_t i = g->n_main; i < g->i_mtp_head; i++) {
        const int k = g->ops[i].kind;
        if (k == OP_QKN_ROPE_KV || k == OP_ROPE_KV || k == OP_KV) {
          g->i_mtp_kv = i + 1u;
          if (k == OP_QKN_ROPE_KV && !L->wqkv.rows && m->arch != ARCH_FLINT && m->arch != ARCH_RICURSIVE) {
            for (uint32_t j = g->n_main; j < i; j++)
              if (g->ops[j].w == &L->wq) {
                g->i_mtp_wq = (int32_t)j;
                break;
              }
          }
          break;
        }
      }
    }
    g->mtp_logits = matvec(&B, md, g->mtp_g, gq2, "mtp_logits", NULL, -1);
    g->mtp_argmax = new_buf(g, "mtp_argmax", 8);
    emit(&B, OP_ARGMAX, g->mtp_logits, g->mtp_argmax, -1)->n = md->rows;
  }
  g->n_mtp = g->i_dfc = g->i_dfb = g->n_ops;
  if (df) df_build(&B, m);
  /* The host reads (or copies) these buffers between runs: they live
   * through the whole op list. */
  const int keep[] = {g->logits, g->argmax, g->h_out, g->mtp_h, g->mtp_g, g->mtp_logits, g->mtp_argmax, g->df_feat, g->df_logits,
                      g->df_sel, g->df_cap, g->ds_feat, g->mtp_r, g->mtp_a, g->mtp_y};
  for (unsigned k = 0; k < sizeof keep / sizeof keep[0]; k++) touch(g, keep[k], 0), touch(g, keep[k], g->n_ops);

  /* Mark f32 output buffers b that have a fused Q8 output (qo >= 0), are not
   * host-visible, and are never referenced by any subsequent op (dead_b = 1). */
  const char *db = getenv("IE_DEAD_B"), *dmp = getenv("IE_DUMP_GPU_OPS");
  if (!(db && db[0] == '0') && !(dmp && dmp[0] != '0')) {
    for (uint32_t i = 0; i < g->n_ops; i++) {
      op *p = &g->ops[i];
      if (p->kind != OP_RMSNORM && p->kind != OP_SWIGLU && p->kind != OP_GDN && p->kind != OP_ATTN) continue;
      if (p->qo < 0 || p->b < 0 || p->nv != NULL) continue;
      if (p->kind != OP_SWIGLU && p->boff != 0) continue;
      const int b = p->b;
      if (b == x0 || b == g->x) continue;
      int kept = 0;
      for (unsigned k = 0; k < sizeof keep / sizeof keep[0]; k++)
        if (b == keep[k]) { kept = 1; break; }
      if (kept) continue;
      int used = 0;
      for (uint32_t j = i + 1; j < g->n_ops; j++) {
        const op *q = &g->ops[j];
        if (q->a == b || q->b == b || q->c == b || q->res == b || q->gt == b || q->qo == b || q->nout == b || q->nq == b) {
          used = 1;
          break;
        }
      }
      if (!used) p->dead_b = 1;
    }
  }

  /* bytes of weights read per token (GEMV + one embedding row) */
  uint64_t wb = 0;
  for (uint32_t i = 0; i < g->n_main; i++) {
    const op *p = &g->ops[i];
    if (p->kind == OP_GEMV_Q4 || p->kind == OP_GEMV_IQ4NL) wb += (uint64_t)p->w->rows * p->w->nb * 18;
    if (p->kind == OP_GEMV_Q4K) wb += (uint64_t)p->w->rows * p->w->nb * 18;
    if (p->kind == OP_GEMV_Q6K) wb += (uint64_t)p->w->rows * p->w->nb * 210 / 8;
    if (p->kind == OP_GEMV_F32) wb += (uint64_t)p->w->rows * p->w->cols * 4;
    if (p->kind == OP_GEMV_Q8) wb += (uint64_t)p->w->rows * p->w->nb * 34;
    if (p->kind == OP_GEMV_Q5K) wb += (uint64_t)p->w->rows * p->w->nb * 22;
    if (p->kind == OP_GEMV_IQ4XS) wb += (uint64_t)p->w->rows * p->w->nb * 17;
    if (p->kind == OP_GEMV_RAW) wb += ie_rawq_bytes(p->w);
    if (p->kind == OP_GEMV_EXP) {
      const int k = p->w->kind;
      const uint64_t per = k == MAT_F32 ? (uint64_t)p->w->rows * p->w->cols * 4
                           : k == MAT_RAWQ ? ie_rawq_bytes(p->w)
                           : (uint64_t)p->w->rows * p->w->nb * (k == MAT_Q8 ? 34 : k == MAT_Q5K ? 22 : k == MAT_IQ4XS ? 17 : k == MAT_Q6K ? 26 : 18);
      wb += per * p->nh;
    }
  }
  g->weight_bytes_per_token = wb;

  /* the KV cache of each attention layer: n_ctx positions, or a ring for a
   * sliding window layer */
  const uint32_t nkv = m->n_kvl;
  g->kv_len = ie_alloc((size_t)(nkv ? nkv : 1) * 4);
  g->kv_off = ie_alloc((size_t)(nkv ? nkv : 1) * 8);
  if (m->kv_win) {
    for (uint32_t kvi = 0; kvi < nkv; kvi++) {
      uint32_t len = n_ctx;
      const uint32_t sw = m->kv_win[kvi];
      if (sw) {
        uint32_t r = 1;
        while (r < sw + (g->T > IE_KV_SLACK ? g->T : IE_KV_SLACK)) r *= 2;
        if (r < len) len = r;
      }
      g->kv_len[kvi] = len;
    }
  } else {
  for (uint32_t l = 0; l < m->n_layer + (m->has_mtp ? 1u : 0u); l++) {
    const layer *L = &m->l[l];
    if (L->rec) continue;
    uint32_t len = n_ctx;
    if (L->swin) {
      uint32_t r = 1;
      while (r < L->swin + (g->T > IE_KV_SLACK ? g->T : IE_KV_SLACK)) r *= 2; /* room for a run of T tokens */
      if (r < len) len = r;
    }
    g->kv_len[L->kvi] = len;
  }
  }
  for (uint32_t i = 0; i < nkv; i++) g->kv_off[i] = g->kv_total, g->kv_total += g->kv_len[i];

  /* RoPE tables: theta_i = pos * base^(-2i/n_rot), i < n_rot / 2 */
  uint32_t h2 = m->n_rot / 2;
  g->rope_cos = ie_alloc((size_t)n_ctx * h2 * 4);
  g->rope_sin = ie_alloc((size_t)n_ctx * h2 * 4);
  for (uint32_t p = 0; p < n_ctx; p++)
    for (uint32_t i = 0; i < h2; i++) {
      double th = (double)p * pow((double)m->rope_base, -2.0 * i / (double)m->n_rot);
      g->rope_cos[p * h2 + i] = (float)cos(th);
      g->rope_sin[p * h2 + i] = (float)sin(th);
    }
}

void graph_free(graph *g) {
  free(g->ops);
  free(g->bufs);
  free(g->rope_cos);
  free(g->rope_sin);
  free(g->kv_len);
  free(g->kv_off);
  memset(g, 0, sizeof *g);
}
