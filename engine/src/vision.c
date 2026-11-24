/* vision.c -- ViT vision projector for Qwen3.5/3.8 (qwen3vl) and Inkling-2-Mini (inkling2v). */
#define COBJMACROS
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#include <wincodec.h>
#endif

#include "backend.h"
#include "common.h"
#include "gen.h"
#include "gguf.h"
#include "vision.h"

#define VIS_MAX_BLK 64
#define VIS_CACHE_N 8

enum { VIS_ARCH_QWEN35 = 0, VIS_ARCH_INKLING2V = 1 };

typedef struct {
  const gguf_tensor *ln1_w, *ln1_b;
  const gguf_tensor *qkv_w, *qkv_b;                         /* Qwen3.5 fused QKV */
  const gguf_tensor *q_w, *q_b, *k_w, *k_b, *v_w, *v_b;     /* Inkling2V separate Q/K/V */
  const gguf_tensor *out_w, *out_b;
  const gguf_tensor *ls1_w;                                 /* Inkling2V LayerScale 1 */
  const gguf_tensor *ln2_w, *ln2_b;
  const gguf_tensor *up_w, *up_b;
  const gguf_tensor *dn_w, *dn_b;
  const gguf_tensor *ls2_w;                                 /* Inkling2V LayerScale 2 */
} vis_blk;

typedef struct {
  uint64_t hash;
  size_t url_len;
  uint32_t n_tok, nx, ny, proj_dim;
  uint64_t used;
  float *embd;
} vis_cache_ent;

typedef struct {
  size_t off_qkv, off_out, off_up, off_dn;
  size_t off_ln1_w, off_ln1_b, off_qkv_b, off_out_b, off_ls1;
  size_t off_ln2_w, off_ln2_b, off_up_b, off_dn_b, off_ls2;
  size_t total_bytes;
} ink_layer_pack;

struct vision_ctx {
  gguf_file g;
  int arch;            /* VIS_ARCH_QWEN35 or VIS_ARCH_INKLING2V */
  uint32_t patch_size; /* 16 (qwen35) or 20 (inkling2v) */
  uint32_t merge_size; /* 2 */
  uint32_t dim;        /* 1152 (qwen35) or 1024 (inkling2v) */
  uint32_t ffn;        /* 4304 (qwen35) or 4096 (inkling2v) */
  uint32_t n_blk;      /* 27 (qwen35) or 23 (inkling2v) */
  uint32_t n_head;     /* 16 */
  uint32_t hd;         /* 72 (qwen35) or 64 (inkling2v) */
  uint32_t proj_dim;   /* 5120 (qwen35) or 2304 (inkling2v) */
  uint32_t pos_grid;   /* 48 (qwen35) or 32 (inkling2v) */
  float eps;           /* 1e-6 (qwen35) or 1e-5 (inkling2v) */
  float mean[3], std[3];
  float *w_patch;      /* [dim * 3 * patch_size * patch_size] f32 */
  uint16_t *w_patch_f16; /* [dim * 3 * patch_size * patch_size] f16 (inkling2v WMMA) */
  float *b_patch;      /* [dim] (qwen35 only; NULL for inkling2v) */
  float *pos_embd;     /* [pos_grid * pos_grid * dim] */
  float *cls_embd;     /* [dim] (inkling2v) */
  float *cls_pos_embd; /* [dim] (inkling2v) */
  const gguf_tensor *pre_ln_w, *pre_ln_b;   /* inkling2v */
  vis_blk blk[VIS_MAX_BLK];
  const gguf_tensor *post_ln_w, *post_ln_b; /* qwen35 */
  const gguf_tensor *mm0_w, *mm0_b;         /* qwen35 */
  const gguf_tensor *mm1_w, *mm1_b;         /* inkling2v */
  const gguf_tensor *mm2_w, *mm2_b;         /* both */
  size_t max_wbytes;
  size_t ink_slot0_bytes, ink_slot1_bytes; /* asymmetric ping-pong slots (slot1 holds larger mm.1) */
  /* GPU kernels (NULL when CPU-only) */
  void *mod;
  void *k_ln, *k_gq8, *k_gf16, *k_gf32, *k_rope, *k_attn;
  /* R14 gfx1201 WMMA vision kernels */
  void *k_wmma_f16, *k_wmma_q8, *k_attn_hd64, *k_ln_f16, *k_patch_prep, *k_merge2x2;
  int has_wmma;
  /* Async double-buffered H2D staging (host-pinned RAM only, 0.0 MiB persistent VRAM) */
  void *vis_cs;
  void *ev_h2d[2];
  void *ev_done[2];
  uint8_t *pin_slot[2];
  size_t pin_slot_cap[2];
  /* LRU cache of encoded images */
  vis_cache_ent cache[VIS_CACHE_N];
  uint64_t tick;
};

static size_t align256(size_t x) { return (x + 255u) & ~(size_t)255u; }

static float F16_LUT[65536];
static int F16_LUT_READY = 0;

static float f16_to_f32(uint16_t h) {
  const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x03FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) bits = sign;
    else {
      uint32_t m = mant, e = 127 - 14;
      while (!(m & 0x0400u)) m <<= 1, e--;
      bits = sign | (e << 23) | ((m & 0x03FFu) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

static void init_f16_lut(void) {
  if (F16_LUT_READY) return;
  for (uint32_t i = 0; i < 65536u; i++) F16_LUT[i] = f16_to_f32((uint16_t)i);
  F16_LUT_READY = 1;
}

static uint16_t f32_to_f16(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int exp = (int)((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x007FFFFFu;
  if (((x >> 23) & 0xFFu) == 0xFFu) {
    if (mant) return (uint16_t)(sign | 0x7E00u | (mant >> 13));
    return (uint16_t)(sign | 0x7C00u);
  }
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;
    const uint32_t shift = (uint32_t)(1 - exp);
    const uint32_t m_full = mant | 0x00800000u;
    const uint32_t m_sub = m_full >> shift;
    const uint32_t lost = m_full & ((1u << shift) - 1u);
    uint32_t hm = m_sub >> 13;
    const uint32_t rb = m_sub & 0x00001FFFu;
    if (rb > 0x00001000u || (rb == 0x00001000u && (lost != 0u || (hm & 1u)))) hm++;
    return (uint16_t)(sign | hm);
  }
  if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
  uint32_t rb = mant & 0x00001FFFu;
  uint32_t hm = mant >> 13;
  if (rb > 0x00001000u || (rb == 0x00001000u && (hm & 1u))) {
    hm++;
    if (hm == 0x0400u) {
      hm = 0;
      if (exp + 1 >= 31) return (uint16_t)(sign | 0x7C00u);
      return (uint16_t)(sign | ((uint32_t)(exp + 1) << 10));
    }
  }
  return (uint16_t)(sign | ((uint32_t)exp << 10) | hm);
}

static uint64_t fnv1a64(const void *p, size_t n) {
  const uint8_t *s = p;
  uint64_t h = 14695981039346656037ull;
  for (size_t i = 0; i < n; i++) h = (h ^ s[i]) * 1099511628211ull;
  return h;
}

static const gguf_tensor *need_t(const gguf_file *g, const char *name) {
  const gguf_tensor *t = gguf_tensor_find(g, name);
  if (!t) ie_die("mmproj: missing tensor %s", name);
  return t;
}

static void tensor_to_f32(const gguf_tensor *t, float *dst, size_t n) {
  if (t->type == 0) {
    memcpy(dst, t->data, n * 4u);
  } else if (t->type == 1) {
    const uint16_t *s = t->data;
    for (size_t i = 0; i < n; i++) dst[i] = f16_to_f32(s[i]);
  } else {
    ie_die("mmproj: unsupported f32 conversion for tensor %s (type %u)", t->name, t->type);
  }
}

static uint32_t kv_u32(const gguf_file *g, const char *k, uint32_t def) {
  int64_t v = 0;
  return gguf_get_int(g, k, &v) && v > 0 ? (uint32_t)v : def;
}

static void resolve_vis_hsaco(const char *hint, char *out, size_t outn) {
  const char *env = getenv("IE_VISION_HSACO");
  if (env && env[0]) {
    snprintf(out, outn, "%s", env);
    return;
  }
  if (hint && hint[0]) {
    const char *s1 = strrchr(hint, '/'), *s2 = strrchr(hint, '\\');
    const char *sl = s1 && s2 ? (s1 > s2 ? s1 : s2) : (s1 ? s1 : s2);
    if (sl) {
      const size_t dlen = (size_t)(sl - hint);
      snprintf(out, outn, "%.*s/ie_vision_kernels.hsaco", (int)dlen, hint);
      return;
    }
  }
  snprintf(out, outn, "build/ie_vision_kernels.hsaco");
}

static ink_layer_pack ink_calc_pack(const vision_ctx *vc, const vis_blk *b) {
  ink_layer_pack p;
  const size_t D4 = (size_t)vc->dim * 4u;
  const size_t F4 = (size_t)vc->ffn * 4u;
  const size_t qkv_b = (size_t)b->q_w->nbytes * 3u;
  p.off_qkv = 0;
  p.off_out = align256(p.off_qkv + qkv_b);
  p.off_up  = align256(p.off_out + (size_t)b->out_w->nbytes);
  p.off_dn  = align256(p.off_up  + (size_t)b->up_w->nbytes);
  size_t off = align256(p.off_dn + (size_t)b->dn_w->nbytes);
  p.off_ln1_w = off; off += D4;
  p.off_ln1_b = off; off += D4;
  p.off_qkv_b = off; off += 3u * D4;
  p.off_out_b = off; off += D4;
  p.off_ls1   = off; off += D4;
  p.off_ln2_w = off; off += D4;
  p.off_ln2_b = off; off += D4;
  p.off_up_b  = off; off += F4;
  p.off_dn_b  = off; off += D4;
  p.off_ls2   = off; off += D4;
  p.total_bytes = align256(off);
  return p;
}

static void ink_fill_pack(const vision_ctx *vc, const vis_blk *b, const ink_layer_pack *p, uint8_t *dst) {
  const size_t qb = (size_t)b->q_w->nbytes;
  const size_t D = vc->dim, F = vc->ffn;
  memcpy(dst + p->off_qkv + 0u * qb, b->q_w->data, qb);
  memcpy(dst + p->off_qkv + 1u * qb, b->k_w->data, qb);
  memcpy(dst + p->off_qkv + 2u * qb, b->v_w->data, qb);
  memcpy(dst + p->off_out, b->out_w->data, (size_t)b->out_w->nbytes);
  memcpy(dst + p->off_up,  b->up_w->data,  (size_t)b->up_w->nbytes);
  memcpy(dst + p->off_dn,  b->dn_w->data,  (size_t)b->dn_w->nbytes);
  tensor_to_f32(b->ln1_w, (float *)(dst + p->off_ln1_w), D);
  tensor_to_f32(b->ln1_b, (float *)(dst + p->off_ln1_b), D);
  tensor_to_f32(b->q_b,   (float *)(dst + p->off_qkv_b) + 0u * D, D);
  tensor_to_f32(b->k_b,   (float *)(dst + p->off_qkv_b) + 1u * D, D);
  tensor_to_f32(b->v_b,   (float *)(dst + p->off_qkv_b) + 2u * D, D);
  tensor_to_f32(b->out_b, (float *)(dst + p->off_out_b), D);
  tensor_to_f32(b->ls1_w, (float *)(dst + p->off_ls1),   D);
  tensor_to_f32(b->ln2_w, (float *)(dst + p->off_ln2_w), D);
  tensor_to_f32(b->ln2_b, (float *)(dst + p->off_ln2_b), D);
  tensor_to_f32(b->up_b,  (float *)(dst + p->off_up_b),  F);
  tensor_to_f32(b->dn_b,  (float *)(dst + p->off_dn_b),  D);
  tensor_to_f32(b->ls2_w, (float *)(dst + p->off_ls2),   D);
}

vision_ctx *vision_open(const char *mmproj_path, const char *hsaco_hint, uint32_t expected_proj_dim) {
  init_f16_lut();
  vision_ctx *vc = calloc(1, sizeof *vc);
  if (!vc) ie_die("vision: out of memory");
  gguf_open(&vc->g, mmproj_path);

  char *ptype = gguf_get_str(&vc->g, "clip.vision.projector_type");
  const int is_ink = (ptype && !strcmp(ptype, "inkling2v"));
  free(ptype);
  vc->arch = is_ink ? VIS_ARCH_INKLING2V : VIS_ARCH_QWEN35;

  vc->patch_size = kv_u32(&vc->g, "clip.vision.patch_size", is_ink ? 20 : 16);
  vc->merge_size = kv_u32(&vc->g, "clip.vision.spatial_merge_size", 2);
  vc->dim = kv_u32(&vc->g, "clip.vision.embedding_length", is_ink ? 1024 : 1152);
  vc->ffn = kv_u32(&vc->g, "clip.vision.feed_forward_length", is_ink ? 4096 : 4304);
  vc->n_blk = kv_u32(&vc->g, "clip.vision.block_count", is_ink ? 23 : 27);
  vc->n_head = kv_u32(&vc->g, "clip.vision.attention.head_count", 16);
  vc->proj_dim = kv_u32(&vc->g, "clip.vision.projection_dim", is_ink ? 2304 : 5120);
  double eps = is_ink ? 1e-5 : 1e-6;
  gguf_get_float(&vc->g, "clip.vision.attention.layer_norm_epsilon", &eps);
  vc->eps = (float)eps;
  vc->hd = vc->dim / vc->n_head;
  if (vc->n_blk > VIS_MAX_BLK || vc->dim % vc->n_head != 0 || vc->hd > 96 || (vc->hd & 3u) != 0)
    ie_die("mmproj: unsupported ViT geometry (blk=%u dim=%u head=%u hd=%u)", vc->n_blk, vc->dim, vc->n_head, vc->hd);
  if (expected_proj_dim && vc->proj_dim != expected_proj_dim)
    ie_die("mmproj: projection_dim %u != model dim %u", vc->proj_dim, expected_proj_dim);

  for (int c = 0; c < 3; c++) vc->mean[c] = 0.5f, vc->std[c] = 0.5f;
  const gguf_kv *km = gguf_find(&vc->g, "clip.vision.image_mean");
  if (km && km->type == GGUF_ARR && km->arr_type == GGUF_F32 && km->arr_n >= 3)
    memcpy(vc->mean, km->arr_data, 12);
  const gguf_kv *ks = gguf_find(&vc->g, "clip.vision.image_std");
  if (ks && ks->type == GGUF_ARR && ks->arr_type == GGUF_F32 && ks->arr_n >= 3)
    memcpy(vc->std, ks->arr_data, 12);

  const gguf_tensor *pw0 = need_t(&vc->g, "v.patch_embd.weight");
  const size_t patch_in = (size_t)3u * vc->patch_size * vc->patch_size;
  const size_t pnumel = (size_t)vc->dim * patch_in;
  vc->w_patch = malloc(pnumel * 4u);
  if (!vc->w_patch) ie_die("vision: out of memory");
  tensor_to_f32(pw0, vc->w_patch, pnumel);

  if (is_ink) {
    vc->w_patch_f16 = malloc(pnumel * 2u);
    if (!vc->w_patch_f16) ie_die("vision: out of memory");
    if (pw0->type == 1) {
      memcpy(vc->w_patch_f16, pw0->data, pnumel * 2u);
    } else {
      for (size_t i = 0; i < pnumel; i++) vc->w_patch_f16[i] = f32_to_f16(vc->w_patch[i]);
    }
    const gguf_tensor *ce = need_t(&vc->g, "v.class_embd");
    const gguf_tensor *cpe = need_t(&vc->g, "v.class_position_embd.weight");
    vc->cls_embd = malloc((size_t)vc->dim * 4u);
    vc->cls_pos_embd = malloc((size_t)vc->dim * 4u);
    if (!vc->cls_embd || !vc->cls_pos_embd) ie_die("vision: out of memory");
    tensor_to_f32(ce, vc->cls_embd, vc->dim);
    tensor_to_f32(cpe, vc->cls_pos_embd, vc->dim);
    vc->pre_ln_w = need_t(&vc->g, "v.pre_ln.weight");
    vc->pre_ln_b = need_t(&vc->g, "v.pre_ln.bias");
  } else {
    const gguf_tensor *pw1 = gguf_tensor_find(&vc->g, "v.patch_embd.weight.1");
    const gguf_tensor *pb = need_t(&vc->g, "v.patch_embd.bias");
    vc->b_patch = malloc((size_t)vc->dim * 4u);
    if (!vc->b_patch) ie_die("vision: out of memory");
    if (pw1) {
      float *tmp = malloc(pnumel * 4u);
      if (!tmp) ie_die("vision: out of memory");
      tensor_to_f32(pw1, tmp, pnumel);
      for (size_t i = 0; i < pnumel; i++) vc->w_patch[i] += tmp[i];
      free(tmp);
    }
    tensor_to_f32(pb, vc->b_patch, vc->dim);
  }

  const gguf_tensor *pe = need_t(&vc->g, "v.position_embd.weight");
  const uint32_t g2 = (uint32_t)pe->ne[1];
  vc->pos_grid = (uint32_t)(sqrt((double)g2) + 0.5);
  if (vc->pos_grid * vc->pos_grid != g2 || pe->ne[0] != vc->dim)
    ie_die("mmproj: unexpected position_embd shape [%llu, %llu]", (unsigned long long)pe->ne[0], (unsigned long long)pe->ne[1]);
  vc->pos_embd = malloc((size_t)g2 * vc->dim * 4u);
  if (!vc->pos_embd) ie_die("vision: out of memory");
  tensor_to_f32(pe, vc->pos_embd, (size_t)g2 * vc->dim);

  char nm[128];
  for (uint32_t i = 0; i < vc->n_blk; i++) {
    vis_blk *b = &vc->blk[i];
#define GETB(f, s) (snprintf(nm, sizeof nm, "v.blk.%u.%s", i, s), b->f = need_t(&vc->g, nm))
    GETB(ln1_w, "ln1.weight");
    GETB(ln1_b, "ln1.bias");
    if (is_ink) {
      GETB(q_w, "attn_q.weight");
      GETB(q_b, "attn_q.bias");
      GETB(k_w, "attn_k.weight");
      GETB(k_b, "attn_k.bias");
      GETB(v_w, "attn_v.weight");
      GETB(v_b, "attn_v.bias");
      if (b->q_w->type != b->k_w->type || b->q_w->type != b->v_w->type ||
          b->q_w->nbytes != b->k_w->nbytes || b->q_w->nbytes != b->v_w->nbytes)
        ie_die("mmproj: mismatched Q/K/V tensor types or sizes at blk.%u", i);
      GETB(out_w, "attn_out.weight");
      GETB(out_b, "attn_out.bias");
      GETB(ls1_w, "ls1.weight");
      GETB(ln2_w, "ln2.weight");
      GETB(ln2_b, "ln2.bias");
      GETB(up_w, "ffn_up.weight");
      GETB(up_b, "ffn_up.bias");
      GETB(dn_w, "ffn_down.weight");
      GETB(dn_b, "ffn_down.bias");
      GETB(ls2_w, "ls2.weight");
      const ink_layer_pack pk = ink_calc_pack(vc, b);
      if (pk.total_bytes > vc->ink_slot0_bytes) vc->ink_slot0_bytes = pk.total_bytes;
      if (pk.total_bytes > vc->ink_slot1_bytes) vc->ink_slot1_bytes = pk.total_bytes;
    } else {
      GETB(qkv_w, "attn_qkv.weight");
      GETB(qkv_b, "attn_qkv.bias");
      GETB(out_w, "attn_out.weight");
      GETB(out_b, "attn_out.bias");
      GETB(ln2_w, "ln2.weight");
      GETB(ln2_b, "ln2.bias");
      GETB(up_w, "ffn_up.weight");
      GETB(up_b, "ffn_up.bias");
      GETB(dn_w, "ffn_down.weight");
      GETB(dn_b, "ffn_down.bias");
      const gguf_tensor *ws[4] = {b->qkv_w, b->out_w, b->up_w, b->dn_w};
      for (int k = 0; k < 4; k++)
        if (ws[k]->nbytes > vc->max_wbytes) vc->max_wbytes = (size_t)ws[k]->nbytes;
    }
#undef GETB
  }
  if (is_ink) {
    vc->mm1_w = need_t(&vc->g, "mm.1.weight");
    vc->mm1_b = gguf_tensor_find(&vc->g, "mm.1.bias");
    vc->mm2_w = need_t(&vc->g, "mm.2.weight");
    vc->mm2_b = gguf_tensor_find(&vc->g, "mm.2.bias");
    const int last_s = (int)((vc->n_blk - 1u) & 1u);
    const size_t mm1_b = align256((size_t)vc->mm1_w->nbytes);
    const size_t mm2_b = align256((size_t)vc->mm2_w->nbytes);
    if (last_s == 0) {
      if (mm1_b > vc->ink_slot1_bytes) vc->ink_slot1_bytes = mm1_b;
      if (mm2_b > vc->ink_slot0_bytes) vc->ink_slot0_bytes = mm2_b;
    } else {
      if (mm1_b > vc->ink_slot0_bytes) vc->ink_slot0_bytes = mm1_b;
      if (mm2_b > vc->ink_slot1_bytes) vc->ink_slot1_bytes = mm2_b;
    }
  } else {
    vc->post_ln_w = need_t(&vc->g, "v.post_ln.weight");
    vc->post_ln_b = need_t(&vc->g, "v.post_ln.bias");
    vc->mm0_w = need_t(&vc->g, "mm.0.weight");
    vc->mm0_b = need_t(&vc->g, "mm.0.bias");
    vc->mm2_w = need_t(&vc->g, "mm.2.weight");
    vc->mm2_b = need_t(&vc->g, "mm.2.bias");
    if (vc->mm0_w->nbytes > vc->max_wbytes) vc->max_wbytes = (size_t)vc->mm0_w->nbytes;
    if (vc->mm2_w->nbytes > vc->max_wbytes) vc->max_wbytes = (size_t)vc->mm2_w->nbytes;
  }

  if (gpu_vis_ready()) {
    char hp[512];
    resolve_vis_hsaco(hsaco_hint, hp, sizeof hp);
    vc->mod = gpu_vis_load_mod(hp);
    if (vc->mod) {
      vc->k_ln = gpu_vis_get_fn(vc->mod, "ie_vis_layernorm");
      vc->k_gq8 = gpu_vis_get_fn(vc->mod, "ie_vis_gemm_q8");
      vc->k_gf16 = gpu_vis_get_fn(vc->mod, "ie_vis_gemm_f16");
      vc->k_gf32 = gpu_vis_get_fn(vc->mod, "ie_vis_gemm_f32");
      vc->k_rope = gpu_vis_get_fn(vc->mod, "ie_vis_rope2d");
      vc->k_attn = gpu_vis_get_fn(vc->mod, "ie_vis_attn");
      vc->k_wmma_f16 = gpu_vis_get_fn(vc->mod, "ie_vis_gemm_wmma_f16");
      vc->k_wmma_q8 = gpu_vis_get_fn(vc->mod, "ie_vis_gemm_wmma_q8");
      vc->k_attn_hd64 = gpu_vis_get_fn(vc->mod, "ie_vis_attn_wmma_hd64");
      vc->k_ln_f16 = gpu_vis_get_fn(vc->mod, "ie_vis_layernorm_f16");
      vc->k_patch_prep = gpu_vis_get_fn(vc->mod, "ie_vis_patch_prep_f16");
      vc->k_merge2x2 = gpu_vis_get_fn(vc->mod, "ie_vis_merge_2x2_f16");
      vc->has_wmma = (vc->k_wmma_f16 && vc->k_wmma_q8 && vc->k_attn_hd64 &&
                      vc->k_ln_f16 && vc->k_patch_prep && vc->k_merge2x2);
      if (is_ink) {
        if (!vc->has_wmma) {
          gpu_vis_unload_mod(vc->mod);
          vc->mod = NULL;
        }
      } else {
        if (!vc->k_ln || !vc->k_gq8 || !vc->k_gf16 || !vc->k_gf32 || !vc->k_rope || !vc->k_attn) {
          gpu_vis_unload_mod(vc->mod);
          vc->mod = NULL;
        }
      }
    }
    fprintf(stderr, "vision: loaded %s (%s, %u blk, dim %u -> %u, %s)\n", mmproj_path,
            is_ink ? "inkling2v" : "qwen3vl", vc->n_blk, vc->dim, vc->proj_dim,
            vc->mod ? (is_ink ? "GPU WMMA zero-persistent-VRAM" : "GPU zero-persistent-VRAM") : "CPU fallback");
  } else {
    fprintf(stderr, "vision: loaded %s (%s, %u blk, dim %u -> %u, CPU)\n", mmproj_path,
            is_ink ? "inkling2v" : "qwen3vl", vc->n_blk, vc->dim, vc->proj_dim);
  }
  return vc;
}

void vision_close(vision_ctx *vc) {
  if (!vc) return;
  for (int i = 0; i < VIS_CACHE_N; i++) free(vc->cache[i].embd);
  for (int i = 0; i < 2; i++) {
    if (vc->ev_h2d[i]) gpu_vis_event_destroy(vc->ev_h2d[i]);
    if (vc->ev_done[i]) gpu_vis_event_destroy(vc->ev_done[i]);
    if (vc->pin_slot[i]) gpu_vis_host_free(vc->pin_slot[i]);
  }
  if (vc->vis_cs) gpu_vis_stream_destroy(vc->vis_cs);
  if (vc->mod) gpu_vis_unload_mod(vc->mod);
  free(vc->w_patch);
  free(vc->w_patch_f16);
  free(vc->b_patch);
  free(vc->pos_embd);
  free(vc->cls_embd);
  free(vc->cls_pos_embd);
  gguf_close(&vc->g);
  free(vc);
}

/* ---- base64 + Windows WIC image decoding ---- */
static int b64_val(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+' || c == '-') return 62;
  if (c == '/' || c == '_') return 63;
  return -1;
}

static uint8_t *decode_base64(const char *s, size_t n, size_t *out_n) {
  uint8_t *buf = malloc((n * 3u) / 4u + 8u);
  if (!buf) return NULL;
  size_t w = 0;
  uint32_t acc = 0;
  int bits = 0;
  for (size_t i = 0; i < n; i++) {
    const unsigned char c = (unsigned char)s[i];
    if (c == '=' || c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
    const int v = b64_val(c);
    if (v < 0) { free(buf); return NULL; }
    acc = (acc << 6) | (uint32_t)v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      buf[w++] = (uint8_t)((acc >> bits) & 0xFFu);
    }
  }
  *out_n = w;
  return buf;
}

static uint8_t *url_to_bytes(const char *url, size_t len, size_t *out_n, char *err, size_t errn) {
  if (!url || !len) {
    snprintf(err, errn, "empty image URL");
    return NULL;
  }
  if (len > 5 && !strncmp(url, "data:", 5)) {
    const char *comma = memchr(url, ',', len);
    if (!comma) {
      snprintf(err, errn, "malformed data: image URI (missing comma)");
      return NULL;
    }
    const size_t hdr_len = (size_t)(comma - url);
    int is_b64 = 0;
    for (size_t i = 0; i + 7 <= hdr_len; i++)
      if (!strncmp(url + i, ";base64", 7)) { is_b64 = 1; break; }
    if (!is_b64) {
      snprintf(err, errn, "data: image URI must be base64-encoded");
      return NULL;
    }
    const char *body = comma + 1;
    const size_t blen = len - (size_t)(body - url);
    uint8_t *b = decode_base64(body, blen, out_n);
    if (!b || !*out_n) {
      free(b);
      snprintf(err, errn, "invalid base64 data in image URI");
      return NULL;
    }
    return b;
  }
  if (len > 7 && !strncmp(url, "file://", 7)) {
    const char *p = url + 7;
    size_t plen = len - 7;
    if (plen >= 3 && p[0] == '/' && p[2] == ':') p++, plen--; /* file:///C:/... */
    char path[1024];
    if (plen >= sizeof path) { snprintf(err, errn, "file:// image path too long"); return NULL; }
    memcpy(path, p, plen);
    path[plen] = 0;
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errn, "cannot open image file %s", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > (64l << 20)) { fclose(f); snprintf(err, errn, "image file empty or > 64 MB"); return NULL; }
    uint8_t *b = malloc((size_t)sz);
    if (!b || fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); snprintf(err, errn, "cannot read image file"); return NULL; }
    fclose(f);
    *out_n = (size_t)sz;
    return b;
  }
  snprintf(err, errn, "unsupported image URL scheme (expected data:image/...;base64,... or file://)");
  return NULL;
}

int vision_validate_url(const char *url, size_t len, char *err, size_t errn) {
  if (!url || !len) { snprintf(err, errn, "empty image URL"); return 0; }
  if (len > 5 && !strncmp(url, "data:", 5)) {
    const char *comma = memchr(url, ',', len);
    if (!comma) { snprintf(err, errn, "malformed data: image URI"); return 0; }
    return 1;
  }
  if (len > 7 && !strncmp(url, "file://", 7)) return 1;
  snprintf(err, errn, "unsupported image URL (expected data:image/...;base64,... or file://)");
  return 0;
}

#ifdef _WIN32
static uint8_t *decode_image_rgb(const uint8_t *data, size_t nbytes, uint32_t *ow, uint32_t *oh, char *err, size_t errn) {
  HRESULT hr_co = CoInitializeEx(NULL, COINIT_MULTITHREADED);
  const int did_co = (hr_co == S_OK || hr_co == S_FALSE);
  IWICImagingFactory *fac = NULL;
  IWICBitmapDecoder *dec = NULL;
  IWICBitmapFrameDecode *frm = NULL;
  IWICFormatConverter *conv = NULL;
  IStream *stm = NULL;
  HGLOBAL hmem = NULL;
  uint8_t *rgb = NULL;

  HRESULT hr = CoCreateInstance(&CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, &IID_IWICImagingFactory, (void **)&fac);
  if (FAILED(hr)) { snprintf(err, errn, "WIC init failed (0x%lx)", (unsigned long)hr); goto done; }

  hmem = GlobalAlloc(GMEM_MOVEABLE, nbytes);
  if (!hmem) { snprintf(err, errn, "out of memory decoding image"); goto done; }
  void *mp = GlobalLock(hmem);
  memcpy(mp, data, nbytes);
  GlobalUnlock(hmem);
  hr = CreateStreamOnHGlobal(hmem, TRUE, &stm);
  if (FAILED(hr)) { GlobalFree(hmem); snprintf(err, errn, "WIC stream failed"); goto done; }
  hmem = NULL; /* owned by stm */

  hr = IWICImagingFactory_CreateDecoderFromStream(fac, stm, NULL, WICDecodeMetadataCacheOnDemand, &dec);
  if (FAILED(hr)) { snprintf(err, errn, "WIC cannot decode image (0x%lx)", (unsigned long)hr); goto done; }
  hr = IWICBitmapDecoder_GetFrame(dec, 0, &frm);
  if (FAILED(hr)) { snprintf(err, errn, "WIC GetFrame failed"); goto done; }
  UINT w = 0, h = 0;
  IWICBitmapFrameDecode_GetSize(frm, &w, &h);
  if (!w || !h || w > 16384u || h > 16384u) { snprintf(err, errn, "invalid image dimensions %ux%u", (unsigned)w, (unsigned)h); goto done; }

  hr = IWICImagingFactory_CreateFormatConverter(fac, &conv);
  if (FAILED(hr)) { snprintf(err, errn, "WIC FormatConverter failed"); goto done; }
  hr = IWICFormatConverter_Initialize(conv, (IWICBitmapSource *)frm, &GUID_WICPixelFormat24bppRGB,
                                      WICBitmapDitherTypeNone, NULL, 0.0, WICBitmapPaletteTypeCustom);
  if (FAILED(hr)) { snprintf(err, errn, "WIC RGB24 conversion failed"); goto done; }

  const UINT stride = w * 3u;
  rgb = malloc((size_t)stride * h);
  if (!rgb) { snprintf(err, errn, "out of memory for RGB image"); goto done; }
  hr = IWICFormatConverter_CopyPixels(conv, NULL, stride, (UINT)((size_t)stride * h), rgb);
  if (FAILED(hr)) { free(rgb); rgb = NULL; snprintf(err, errn, "WIC CopyPixels failed"); goto done; }
  *ow = (uint32_t)w, *oh = (uint32_t)h;
done:
  if (conv) IWICFormatConverter_Release(conv);
  if (frm) IWICBitmapFrameDecode_Release(frm);
  if (dec) IWICBitmapDecoder_Release(dec);
  if (stm) IStream_Release(stm);
  if (fac) IWICImagingFactory_Release(fac);
  if (did_co) CoUninitialize();
  return rgb;
}
#else
static uint8_t *decode_image_rgb(const uint8_t *data, size_t nbytes, uint32_t *ow, uint32_t *oh, char *err, size_t errn) {
  (void)data; (void)nbytes; (void)ow; (void)oh;
  snprintf(err, errn, "image decoding requires Windows WIC");
  return NULL;
}
#endif

/* Qwen3.5 smart_resize: snap (w, h) to multiples of factor = patch_size * merge_size (32)
 * while keeping total merged tokens within [min_tok, max_tok]. */
static void smart_resize(uint32_t w, uint32_t h, uint32_t factor, uint32_t min_tok, uint32_t max_tok, uint32_t *ow, uint32_t *oh) {
  const double min_px = (double)min_tok * factor * factor;
  const double max_px = (double)max_tok * factor * factor;
  uint32_t nw = (uint32_t)llround((double)w / factor) * factor;
  uint32_t nh = (uint32_t)llround((double)h / factor) * factor;
  if (nw < factor) nw = factor;
  if (nh < factor) nh = factor;
  if ((double)nw * nh > max_px) {
    const double beta = sqrt(((double)w * h) / max_px);
    nw = (uint32_t)floor((double)w / beta / factor) * factor;
    nh = (uint32_t)floor((double)h / beta / factor) * factor;
  } else if ((double)nw * nh < min_px) {
    const double beta = sqrt(min_px / ((double)w * h));
    nw = (uint32_t)ceil((double)w * beta / factor) * factor;
    nh = (uint32_t)ceil((double)h * beta / factor) * factor;
  }
  if (nw < factor) nw = factor;
  if (nh < factor) nh = factor;
  *ow = nw, *oh = nh;
}

/* Catmull-Rom bicubic weight (a = -0.5, matching PIL/PyTorch bicubic) */
static float cubic_w(float x) {
  x = fabsf(x);
  if (x <= 1.0f) return (1.5f * x - 2.5f) * x * x + 1.0f;
  if (x < 2.0f) return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
  return 0.0f;
}

/* Separable bicubic resize RGB8 [ih, iw, 3] -> RGB8 [oh, ow, 3] */
static uint8_t *resize_rgb_bicubic(const uint8_t *rgb, uint32_t iw, uint32_t ih, uint32_t ow, uint32_t oh) {
  uint8_t *out = malloc((size_t)ow * oh * 3u);
  if (!out) ie_die("vision: out of memory");
  if (iw == ow && ih == oh) {
    memcpy(out, rgb, (size_t)ow * oh * 3u);
    return out;
  }
  float *tmp = malloc((size_t)ih * ow * 3u * 4u);
  if (!tmp) ie_die("vision: out of memory");
  const float sx = (float)iw / (float)ow;
  const float sy = (float)ih / (float)oh;
  for (uint32_t y = 0; y < ih; y++) {
    const uint8_t *row = rgb + (size_t)y * iw * 3u;
    float *dst = tmp + (size_t)y * ow * 3u;
    for (uint32_t x = 0; x < ow; x++) {
      const float cx = ((float)x + 0.5f) * sx - 0.5f;
      const int ix = (int)floorf(cx);
      float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, ws = 0.0f;
      for (int k = -1; k <= 2; k++) {
        int px = ix + k;
        if (px < 0) px = 0;
        if (px >= (int)iw) px = (int)iw - 1;
        const float w = cubic_w(cx - (float)(ix + k));
        ws += w;
        s0 += w * (float)row[px * 3 + 0];
        s1 += w * (float)row[px * 3 + 1];
        s2 += w * (float)row[px * 3 + 2];
      }
      const float inv = ws != 0.0f ? 1.0f / ws : 1.0f;
      dst[x * 3u + 0] = s0 * inv;
      dst[x * 3u + 1] = s1 * inv;
      dst[x * 3u + 2] = s2 * inv;
    }
  }
  for (uint32_t y = 0; y < oh; y++) {
    const float cy = ((float)y + 0.5f) * sy - 0.5f;
    const int iy = (int)floorf(cy);
    float wy[4], ws = 0.0f;
    int py[4];
    for (int k = 0; k < 4; k++) {
      int p = iy + k - 1;
      if (p < 0) p = 0;
      if (p >= (int)ih) p = (int)ih - 1;
      py[k] = p;
      wy[k] = cubic_w(cy - (float)(iy + k - 1));
      ws += wy[k];
    }
    const float inv = ws != 0.0f ? 1.0f / ws : 1.0f;
    for (int k = 0; k < 4; k++) wy[k] *= inv;
    for (uint32_t x = 0; x < ow; x++) {
      float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;
      for (int k = 0; k < 4; k++) {
        const float *s = tmp + ((size_t)py[k] * ow + x) * 3u;
        c0 += wy[k] * s[0], c1 += wy[k] * s[1], c2 += wy[k] * s[2];
      }
      if (c0 < 0.0f) c0 = 0.0f; else if (c0 > 255.0f) c0 = 255.0f;
      if (c1 < 0.0f) c1 = 0.0f; else if (c1 > 255.0f) c1 = 255.0f;
      if (c2 < 0.0f) c2 = 0.0f; else if (c2 > 255.0f) c2 = 255.0f;
      uint8_t *d = out + ((size_t)y * ow + x) * 3u;
      d[0] = (uint8_t)llroundf(c0);
      d[1] = (uint8_t)llroundf(c1);
      d[2] = (uint8_t)llroundf(c2);
    }
  }
  free(tmp);
  return out;
}

/* Separable bicubic resize RGB8 [ih, iw, 3] -> normalized f32 CHW [3, oh, ow] (Qwen3.5) */
static float *resize_and_norm(const uint8_t *rgb, uint32_t iw, uint32_t ih, uint32_t ow, uint32_t oh,
                              const float mean[3], const float std[3]) {
  float *out = malloc((size_t)3u * ow * oh * 4u);
  if (!out) ie_die("vision: out of memory");
  if (iw == ow && ih == oh) {
    const size_t plane = (size_t)ow * oh;
    for (size_t i = 0; i < plane; i++) {
      out[0 * plane + i] = ((float)rgb[i * 3u + 0] / 255.0f - mean[0]) / std[0];
      out[1 * plane + i] = ((float)rgb[i * 3u + 1] / 255.0f - mean[1]) / std[1];
      out[2 * plane + i] = ((float)rgb[i * 3u + 2] / 255.0f - mean[2]) / std[2];
    }
    return out;
  }
  /* Horizontal pass: [ih, ow, 3] */
  float *tmp = malloc((size_t)ih * ow * 3u * 4u);
  if (!tmp) ie_die("vision: out of memory");
  const float sx = (float)iw / (float)ow;
  const float sy = (float)ih / (float)oh;
  for (uint32_t y = 0; y < ih; y++) {
    const uint8_t *row = rgb + (size_t)y * iw * 3u;
    float *dst = tmp + (size_t)y * ow * 3u;
    for (uint32_t x = 0; x < ow; x++) {
      const float cx = ((float)x + 0.5f) * sx - 0.5f;
      const int ix = (int)floorf(cx);
      float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, ws = 0.0f;
      for (int k = -1; k <= 2; k++) {
        int px = ix + k;
        if (px < 0) px = 0;
        if (px >= (int)iw) px = (int)iw - 1;
        const float w = cubic_w(cx - (float)(ix + k));
        ws += w;
        s0 += w * (float)row[px * 3 + 0];
        s1 += w * (float)row[px * 3 + 1];
        s2 += w * (float)row[px * 3 + 2];
      }
      const float inv = ws != 0.0f ? 1.0f / ws : 1.0f;
      dst[x * 3u + 0] = s0 * inv;
      dst[x * 3u + 1] = s1 * inv;
      dst[x * 3u + 2] = s2 * inv;
    }
  }
  /* Vertical pass -> clamped [0, 255] -> normalized CHW */
  const size_t plane = (size_t)ow * oh;
  for (uint32_t y = 0; y < oh; y++) {
    const float cy = ((float)y + 0.5f) * sy - 0.5f;
    const int iy = (int)floorf(cy);
    float wy[4], ws = 0.0f;
    int py[4];
    for (int k = 0; k < 4; k++) {
      int p = iy + k - 1;
      if (p < 0) p = 0;
      if (p >= (int)ih) p = (int)ih - 1;
      py[k] = p;
      wy[k] = cubic_w(cy - (float)(iy + k - 1));
      ws += wy[k];
    }
    const float inv = ws != 0.0f ? 1.0f / ws : 1.0f;
    for (int k = 0; k < 4; k++) wy[k] *= inv;
    for (uint32_t x = 0; x < ow; x++) {
      float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;
      for (int k = 0; k < 4; k++) {
        const float *s = tmp + ((size_t)py[k] * ow + x) * 3u;
        c0 += wy[k] * s[0], c1 += wy[k] * s[1], c2 += wy[k] * s[2];
      }
      if (c0 < 0.0f) c0 = 0.0f; else if (c0 > 255.0f) c0 = 255.0f;
      if (c1 < 0.0f) c1 = 0.0f; else if (c1 > 255.0f) c1 = 255.0f;
      if (c2 < 0.0f) c2 = 0.0f; else if (c2 > 255.0f) c2 = 255.0f;
      const size_t idx = (size_t)y * ow + x;
      out[0 * plane + idx] = (c0 / 255.0f - mean[0]) / std[0];
      out[1 * plane + idx] = (c1 / 255.0f - mean[1]) / std[1];
      out[2 * plane + idx] = (c2 / 255.0f - mean[2]) / std[2];
    }
  }
  free(tmp);
  return out;
}

/* Inkling-2-Mini preprocess (matching llama.cpp PR #32217 mtmd-image.cpp):
 * - align = patch_size * merge_size (40)
 * - Only downscale if align_up(iw, 40) * align_up(ih, 40) > max_tok * align * align
 * - Pad bottom-right with -1.0f BEFORE mean/std normalization -> -1.007843137f for mean=std=0.5! */
static float *inkling2_resize_pad_norm(const uint8_t *rgb, uint32_t iw, uint32_t ih,
                                       uint32_t align, uint32_t max_tok,
                                       const float mean[3], const float std[3],
                                       uint32_t *ow_out, uint32_t *oh_out) {
  const uint64_t max_px = (uint64_t)max_tok * (uint64_t)align * (uint64_t)align;
  uint32_t sw = iw, sh = ih;
  uint32_t ow = ((sw + align - 1u) / align) * align;
  uint32_t oh = ((sh + align - 1u) / align) * align;
  if ((uint64_t)ow * (uint64_t)oh > max_px) {
    double beta = sqrt((double)max_px / ((double)iw * (double)ih));
    sw = (uint32_t)floor((double)iw * beta);
    sh = (uint32_t)floor((double)ih * beta);
    if (sw < 1u) sw = 1u;
    if (sh < 1u) sh = 1u;
    ow = ((sw + align - 1u) / align) * align;
    oh = ((sh + align - 1u) / align) * align;
    for (int it = 0; it < 16 && (uint64_t)ow * (uint64_t)oh > max_px; it++) {
      const double shrink = sqrt((double)max_px / ((double)ow * (double)oh)) * 0.999;
      uint32_t nw = (uint32_t)floor((double)sw * shrink);
      uint32_t nh = (uint32_t)floor((double)sh * shrink);
      if (nw >= sw && sw > 1u) nw = sw - 1u;
      if (nh >= sh && sh > 1u) nh = sh - 1u;
      if (nw < 1u) nw = 1u;
      if (nh < 1u) nh = 1u;
      sw = nw, sh = nh;
      ow = ((sw + align - 1u) / align) * align;
      oh = ((sh + align - 1u) / align) * align;
      if (sw == 1u && sh == 1u) break;
    }
  }
  uint8_t *resized = (sw != iw || sh != ih) ? resize_rgb_bicubic(rgb, iw, ih, sw, sh) : NULL;
  const uint8_t *src = resized ? resized : rgb;

  const size_t plane = (size_t)ow * oh;
  float *out = malloc((size_t)3u * plane * 4u);
  if (!out) ie_die("vision: out of memory");
  const float pad_v[3] = {
      (-1.0f / 255.0f - mean[0]) / std[0],
      (-1.0f / 255.0f - mean[1]) / std[1],
      (-1.0f / 255.0f - mean[2]) / std[2],
  };
  const float inv255_s[3] = {1.0f / (255.0f * std[0]), 1.0f / (255.0f * std[1]), 1.0f / (255.0f * std[2])};
  const float neg_ms[3] = {-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]};

  for (uint32_t y = 0; y < oh; y++) {
    const size_t row_off = (size_t)y * ow;
    if (y < sh) {
      const uint8_t *sr = src + (size_t)y * sw * 3u;
      for (uint32_t x = 0; x < sw; x++) {
        out[0 * plane + row_off + x] = (float)sr[x * 3u + 0] * inv255_s[0] + neg_ms[0];
        out[1 * plane + row_off + x] = (float)sr[x * 3u + 1] * inv255_s[1] + neg_ms[1];
        out[2 * plane + row_off + x] = (float)sr[x * 3u + 2] * inv255_s[2] + neg_ms[2];
      }
      for (uint32_t x = sw; x < ow; x++) {
        out[0 * plane + row_off + x] = pad_v[0];
        out[1 * plane + row_off + x] = pad_v[1];
        out[2 * plane + row_off + x] = pad_v[2];
      }
    } else {
      for (uint32_t x = 0; x < ow; x++) {
        out[0 * plane + row_off + x] = pad_v[0];
        out[1 * plane + row_off + x] = pad_v[1];
        out[2 * plane + row_off + x] = pad_v[2];
      }
    }
  }
  free(resized);
  *ow_out = ow, *oh_out = oh;
  return out;
}

/* ---- Qwen3.5 Host preprocessing: 2x2-merged patches, align_corners pos_emb, 2D RoPE ---- */
static void prep_inputs(const vision_ctx *vc, const float *rgb_chw, uint32_t w, uint32_t h,
                        float *patches_out, float *pos_out, float *rcos_out, float *rsin_out) {
  const uint32_t P = vc->patch_size, M = vc->merge_size, D = vc->dim;
  const uint32_t gh = h / P, gw = w / P;
  const uint32_t ny = gh / M, nx = gw / M;
  const uint32_t patch_in = 3u * P * P;
  const size_t plane = (size_t)w * h;
  const uint32_t og = vc->pos_grid;
  const uint32_t half = vc->hd / 2u, qtr = half / 2u;

  double inv_freq[32];
  for (uint32_t j = 0; j < qtr; j++) inv_freq[j] = pow(10000.0, -2.0 * (double)j / (double)half);

  uint32_t tok = 0;
  for (uint32_t my = 0; my < ny; my++) {
    for (uint32_t mx = 0; mx < nx; mx++) {
      for (uint32_t sy = 0; sy < M; sy++) {
        for (uint32_t sx = 0; sx < M; sx++) {
          const uint32_t py = my * M + sy;
          const uint32_t px = mx * M + sx;
          float *pbuf = patches_out + (size_t)tok * patch_in;
          uint32_t pi = 0;
          for (uint32_t c = 0; c < 3u; c++) {
            const float *cp = rgb_chw + c * plane;
            for (uint32_t dy = 0; dy < P; dy++) {
              const float *rp = cp + (size_t)(py * P + dy) * w + px * P;
              memcpy(pbuf + pi, rp, (size_t)P * 4u);
              pi += P;
            }
          }
          /* Bilinear align_corners pos_embd at (py, px) from [og, og, D] */
          const double fy = gh > 1u ? (double)py * (og - 1u) / (double)(gh - 1u) : 0.0;
          const double fx = gw > 1u ? (double)px * (og - 1u) / (double)(gw - 1u) : 0.0;
          const uint32_t y0 = (uint32_t)floor(fy), y1 = y0 + 1u < og ? y0 + 1u : og - 1u;
          const uint32_t x0 = (uint32_t)floor(fx), x1 = x0 + 1u < og ? x0 + 1u : og - 1u;
          const float wy = (float)(fy - y0), wx = (float)(fx - x0);
          const float w00 = (1.0f - wy) * (1.0f - wx), w01 = (1.0f - wy) * wx;
          const float w10 = wy * (1.0f - wx), w11 = wy * wx;
          const float *pe00 = vc->pos_embd + ((size_t)y0 * og + x0) * D;
          const float *pe01 = vc->pos_embd + ((size_t)y0 * og + x1) * D;
          const float *pe10 = vc->pos_embd + ((size_t)y1 * og + x0) * D;
          const float *pe11 = vc->pos_embd + ((size_t)y1 * og + x1) * D;
          float *pd = pos_out + (size_t)tok * D;
          for (uint32_t d = 0; d < D; d++)
            pd[d] = w00 * pe00[d] + w01 * pe01[d] + w10 * pe10[d] + w11 * pe11[d];

          /* 2D Vision RoPE: first qtr = py, second qtr = px */
          float *rc = rcos_out + (size_t)tok * half;
          float *rs = rsin_out + (size_t)tok * half;
          for (uint32_t j = 0; j < qtr; j++) {
            const double ay = (double)py * inv_freq[j];
            const double ax = (double)px * inv_freq[j];
            rc[j] = (float)cos(ay);
            rs[j] = (float)sin(ay);
            rc[qtr + j] = (float)cos(ax);
            rs[qtr + j] = (float)sin(ax);
          }
          tok++;
        }
      }
    }
  }
}

/* ---- Inkling-2-Mini Host preprocessing: row-major patches, align_corners=False pos_emb, 1-indexed Axial RoPE ---- */
static void prep_inputs_inkling2(const vision_ctx *vc, const float *rgb_chw, uint32_t w, uint32_t h,
                                 float *patches_out, float *pos_out, float *rcos_out, float *rsin_out) {
  const uint32_t P = vc->patch_size, D = vc->dim;
  const uint32_t gh = h / P, gw = w / P;
  const uint32_t patch_in = 3u * P * P;
  const size_t plane = (size_t)w * h;
  const uint32_t og = vc->pos_grid;
  const uint32_t half = vc->hd / 2u, qtr = half / 2u; /* 32, 16 */

  double inv_freq[32];
  for (uint32_t j = 0; j < qtr; j++) inv_freq[j] = pow(10000.0, -2.0 * (double)j / (double)half);

  /* Row 0 (CLS): pos_w = 0, pos_h = 0 -> cos = 1, sin = 0 */
  for (uint32_t j = 0; j < half; j++) {
    rcos_out[j] = 1.0f;
    rsin_out[j] = 0.0f;
  }

  for (uint32_t py = 0; py < gh; py++) {
    double fy = ((double)py + 0.5) * ((double)og / (double)gh) - 0.5;
    if (fy < 0.0) fy = 0.0; else if (fy > (double)(og - 1u)) fy = (double)(og - 1u);
    const uint32_t y0 = (uint32_t)floor(fy), y1 = y0 + 1u < og ? y0 + 1u : og - 1u;
    const float wy = (float)(fy - (double)y0);

    for (uint32_t px = 0; px < gw; px++) {
      const uint32_t idx = py * gw + px;
      if (patches_out) {
        float *pbuf = patches_out + (size_t)idx * patch_in;
        uint32_t pi = 0;
        for (uint32_t c = 0; c < 3u; c++) {
          const float *cp = rgb_chw + c * plane;
          for (uint32_t dy = 0; dy < P; dy++) {
            const float *rp = cp + (size_t)(py * P + dy) * w + px * P;
            memcpy(pbuf + pi, rp, (size_t)P * 4u);
            pi += P;
          }
        }
      }
      if (pos_out) {
        float *pd = pos_out + (size_t)idx * D;
        if (gh == og && gw == og) {
          memcpy(pd, vc->pos_embd + (size_t)idx * D, (size_t)D * 4u);
        } else {
          double fx = ((double)px + 0.5) * ((double)og / (double)gw) - 0.5;
          if (fx < 0.0) fx = 0.0; else if (fx > (double)(og - 1u)) fx = (double)(og - 1u);
          const uint32_t x0 = (uint32_t)floor(fx), x1 = x0 + 1u < og ? x0 + 1u : og - 1u;
          const float wx = (float)(fx - (double)x0);
          const float w00 = (1.0f - wy) * (1.0f - wx), w01 = (1.0f - wy) * wx;
          const float w10 = wy * (1.0f - wx), w11 = wy * wx;
          const float *pe00 = vc->pos_embd + ((size_t)y0 * og + x0) * D;
          const float *pe01 = vc->pos_embd + ((size_t)y0 * og + x1) * D;
          const float *pe10 = vc->pos_embd + ((size_t)y1 * og + x0) * D;
          const float *pe11 = vc->pos_embd + ((size_t)y1 * og + x1) * D;
          for (uint32_t d = 0; d < D; d++)
            pd[d] = w00 * pe00[d] + w01 * pe01[d] + w10 * pe10[d] + w11 * pe11[d];
        }
      }
      /* 1-indexed Axial RoPE: first qtr = pos_w (px + 1), second qtr = pos_h (py + 1) */
      const uint32_t t = 1u + idx;
      float *rc = rcos_out + (size_t)t * half;
      float *rs = rsin_out + (size_t)t * half;
      const double pw = (double)(px + 1u);
      const double ph = (double)(py + 1u);
      for (uint32_t j = 0; j < qtr; j++) {
        const double aw = pw * inv_freq[j];
        const double ah = ph * inv_freq[j];
        rc[j] = (float)cos(aw);
        rs[j] = (float)sin(aw);
        rc[qtr + j] = (float)cos(ah);
        rs[qtr + j] = (float)sin(ah);
      }
    }
  }
}

static void prep_inputs_inkling2_gpu(const vision_ctx *vc, const float *rgb_chw, uint32_t w, uint32_t h,
                                     uint16_t *patches_f16, float *pos_out) {
  const uint32_t P = vc->patch_size, D = vc->dim;
  const uint32_t gh = h / P, gw = w / P;
  const uint32_t patch_in = 3u * P * P;
  const size_t plane = (size_t)w * h;
  const uint32_t og = vc->pos_grid;

  if (gh == og && gw == og) {
    memcpy(pos_out, vc->pos_embd, (size_t)gh * gw * D * 4u);
  }
  for (uint32_t py = 0; py < gh; py++) {
    double fy = ((double)py + 0.5) * ((double)og / (double)gh) - 0.5;
    if (fy < 0.0) fy = 0.0; else if (fy > (double)(og - 1u)) fy = (double)(og - 1u);
    const uint32_t y0 = (uint32_t)floor(fy), y1 = y0 + 1u < og ? y0 + 1u : og - 1u;
    const float wy = (float)(fy - (double)y0);

    for (uint32_t px = 0; px < gw; px++) {
      const uint32_t idx = py * gw + px;
      uint16_t *pbuf = patches_f16 + (size_t)idx * patch_in;
      uint32_t pi = 0;
      for (uint32_t c = 0; c < 3u; c++) {
        const float *cp = rgb_chw + c * plane;
        for (uint32_t dy = 0; dy < P; dy++) {
          const float *rp = cp + (size_t)(py * P + dy) * w + px * P;
          for (uint32_t dx = 0; dx < P; dx++) pbuf[pi++] = f32_to_f16(rp[dx]);
        }
      }
      if (!(gh == og && gw == og)) {
        double fx = ((double)px + 0.5) * ((double)og / (double)gw) - 0.5;
        if (fx < 0.0) fx = 0.0; else if (fx > (double)(og - 1u)) fx = (double)(og - 1u);
        const uint32_t x0 = (uint32_t)floor(fx), x1 = x0 + 1u < og ? x0 + 1u : og - 1u;
        const float wx = (float)(fx - (double)x0);
        const float w00 = (1.0f - wy) * (1.0f - wx), w01 = (1.0f - wy) * wx;
        const float w10 = wy * (1.0f - wx), w11 = wy * wx;
        const float *pe00 = vc->pos_embd + ((size_t)y0 * og + x0) * D;
        const float *pe01 = vc->pos_embd + ((size_t)y0 * og + x1) * D;
        const float *pe10 = vc->pos_embd + ((size_t)y1 * og + x0) * D;
        const float *pe11 = vc->pos_embd + ((size_t)y1 * og + x1) * D;
        float *pd = pos_out + (size_t)idx * D;
        for (uint32_t d = 0; d < D; d++)
          pd[d] = w00 * pe00[d] + w01 * pe01[d] + w10 * pe10[d] + w11 * pe11[d];
      }
    }
  }
}

/* ---- Inkling-2-Mini RDNA4 WMMA GPU ViT forward pass (1 temporary arena, 0.0 MiB persistent VRAM) ---- */
static void ink_wmma_gemm(const vision_ctx *vc, uint32_t wtype,
                          void *d_A, void *d_W, void *d_bias, void *d_ls, void *d_res, void *d_C,
                          uint32_t N, uint32_t M, uint32_t K, uint32_t act, uint32_t out_f16) {
  void *args[] = {&d_A, &d_W, &d_bias, &d_ls, &d_res, &d_C, &N, &M, &K, &act, &out_f16};
  const unsigned gx = (M + 127u) / 128u, gy = (N + 63u) / 64u;
  if (wtype == 1u) {
    gpu_vis_launch(vc->k_wmma_f16, gx, gy, 1, 256, 1, 1, args);
  } else if (wtype == 8u) {
    gpu_vis_launch(vc->k_wmma_q8, gx, gy, 1, 256, 1, 1, args);
  } else {
    ie_die("vision GPU WMMA: unsupported weight type %u", wtype);
  }
}

static void vit_run_inkling2_gpu(vision_ctx *vc, const uint16_t *patches_f16, const float *pos_host,
                                 uint32_t gw, uint32_t gh, float *out_host) {
  const uint32_t D = vc->dim, F = vc->ffn, nh = vc->n_head, hd = vc->hd;
  const uint32_t patch_in = 3u * vc->patch_size * vc->patch_size;
  const uint32_t N_patches = gw * gh;
  const uint32_t N_pos = N_patches + 1u;
  const uint32_t ny = gh / vc->merge_size, nx = gw / vc->merge_size;
  const uint32_t n_tok = nx * ny, mdim = 4u * D;

  /* Lazy host-pinned double-buffered async staging (0.0 MiB VRAM!) */
  if (!vc->vis_cs) {
    const size_t pw_pack = align256((size_t)D * patch_in * 2u) + align256((size_t)4u * D * 4u);
    vc->pin_slot_cap[0] = vc->ink_slot0_bytes;
    vc->pin_slot_cap[1] = vc->ink_slot1_bytes > pw_pack ? vc->ink_slot1_bytes : pw_pack;
    vc->vis_cs = gpu_vis_stream_create();
    for (int s = 0; s < 2; s++) {
      vc->ev_h2d[s] = gpu_vis_event_create();
      vc->ev_done[s] = gpu_vis_event_create();
      vc->pin_slot[s] = gpu_vis_host_alloc(vc->pin_slot_cap[s]);
    }
  }

  /* Single temporary VRAM arena (borrows idle LLM b->arena in ie-serve -> 0.0 MiB extra VRAM!) */
  const size_t sz_x = align256((size_t)N_pos * D * 4u);
  const size_t sz_h = align256((size_t)N_pos * D * 2u);
  const size_t sz_att = align256((size_t)N_pos * D * 2u);
  const size_t pf_bytes = align256((size_t)N_patches * patch_in * 2u);
  const size_t step0_qf_req = pf_bytes + align256((size_t)N_patches * D * 4u);
  size_t qf_req = (size_t)N_pos * (F > 3u * D ? F : 3u * D) * 2u;
  if ((size_t)n_tok * mdim * 2u > qf_req) qf_req = (size_t)n_tok * mdim * 2u;
  if (step0_qf_req > qf_req) qf_req = step0_qf_req;
  const size_t sz_qf = align256(qf_req);
  const size_t sz_out = align256((size_t)n_tok * vc->proj_dim * 4u);
  const size_t sz_slot0 = align256(vc->pin_slot_cap[0]);
  const size_t sz_slot1 = align256(vc->pin_slot_cap[1]);

  size_t off = 0;
  const size_t off_x = off; off += sz_x;
  const size_t off_h = off; off += sz_h;
  const size_t off_att = off; off += sz_att;
  const size_t off_qf = off; off += sz_qf;
  const size_t off_out = off; off += sz_out;
  const size_t off_w0 = off; off += sz_slot0;
  const size_t off_w1 = off; off += sz_slot1;

  uint8_t *d_arena = gpu_vis_alloc(off);
  uint8_t *d_x = d_arena + off_x;
  uint8_t *d_h = d_arena + off_h;      /* Step 0: contiguous d_h + d_att holds pos_emb f32 (size >= N_pos*D*4) */
  uint8_t *d_att = d_arena + off_att;
  uint8_t *d_qf = d_arena + off_qf;    /* Step 0: [0..pf_bytes) holds patches_f16; [pf_bytes..) holds patch_out f32 */
  uint8_t *d_patch_out = d_qf + pf_bytes;
  uint8_t *d_out = d_arena + off_out;
  uint8_t *d_wslot[2] = {d_arena + off_w0, d_arena + off_w1};

  /* Step 0: pack w_patch_f16 + cls/pre_ln vectors into pin_slot[1] -> d_wslot[1] on compute stream */
  const size_t pw_bytes = (size_t)D * patch_in * 2u;
  const size_t o_cls = align256(pw_bytes);
  const size_t o_clsp = o_cls + (size_t)D * 4u;
  const size_t o_plw = o_clsp + (size_t)D * 4u;
  const size_t o_plb = o_plw + (size_t)D * 4u;
  const size_t init_bytes = align256(o_plb + (size_t)D * 4u);
  memcpy(vc->pin_slot[1], vc->w_patch_f16, pw_bytes);
  memcpy(vc->pin_slot[1] + o_cls, vc->cls_embd, (size_t)D * 4u);
  memcpy(vc->pin_slot[1] + o_clsp, vc->cls_pos_embd, (size_t)D * 4u);
  tensor_to_f32(vc->pre_ln_w, (float *)(vc->pin_slot[1] + o_plw), D);
  tensor_to_f32(vc->pre_ln_b, (float *)(vc->pin_slot[1] + o_plb), D);

  gpu_vis_h2d_async(d_wslot[1], vc->pin_slot[1], init_bytes, NULL);
  gpu_vis_h2d(d_qf, patches_f16, (size_t)N_patches * patch_in * 2u);
  gpu_vis_h2d(d_h, pos_host, (size_t)N_patches * D * 4u);

  /* Prefetch blk.0 asynchronously into d_wslot[0] on dedicated copy stream vis_cs */
  ink_layer_pack pk_cur = ink_calc_pack(vc, &vc->blk[0]);
  ink_fill_pack(vc, &vc->blk[0], &pk_cur, vc->pin_slot[0]);
  gpu_vis_h2d_async(d_wslot[0], vc->pin_slot[0], pk_cur.total_bytes, vc->vis_cs);
  gpu_vis_event_record(vc->ev_h2d[0], vc->vis_cs);

  /* Launch Step 0 patch projection GEMM + CLS prepend/pos_add/pre_ln on compute stream */
  ink_wmma_gemm(vc, 1u, d_qf, d_wslot[1], NULL, NULL, NULL, d_patch_out, N_patches, D, patch_in, 0u, 0u);
  {
    void *po = d_patch_out, *pe = d_h;
    void *ce = d_wslot[1] + o_cls, *cpe = d_wslot[1] + o_clsp;
    void *plw = d_wslot[1] + o_plw, *plb = d_wslot[1] + o_plb;
    void *o16 = NULL, *o32 = d_x;
    uint32_t np = N_patches, dd = D, in16 = 0u;
    float eps = vc->eps;
    void *args[] = {&po, &pe, &ce, &cpe, &plw, &plb, &o16, &o32, &np, &dd, &eps, &in16};
    gpu_vis_launch(vc->k_patch_prep, N_pos, 1, 1, 256, 1, 1, args);
  }
  gpu_vis_event_record(vc->ev_done[1], NULL);

  /* 23 ViT layers with double-buffered async H2D overlap */
  const float scale = 1.0f / sqrtf((float)hd);
  for (uint32_t i = 0; i < vc->n_blk; i++) {
    const int s = (int)(i & 1u);
    const int ns = s ^ 1;
    const vis_blk *b = &vc->blk[i];
    uint8_t *ws = d_wslot[s];

    gpu_vis_stream_wait(NULL, vc->ev_h2d[s]);

    /* 1. LN1 (f32 -> f16) */
    {
      void *xp = d_x, *lw = ws + pk_cur.off_ln1_w, *lb = ws + pk_cur.off_ln1_b;
      void *y16 = d_h, *y32 = NULL;
      uint32_t nn = N_pos, dd = D, in16 = 0u;
      float eps = vc->eps;
      void *args[] = {&xp, &lw, &lb, &y16, &y32, &nn, &dd, &eps, &in16};
      gpu_vis_launch(vc->k_ln_f16, N_pos, 1, 1, 256, 1, 1, args);
    }
    /* 2. QKV WMMA GEMM (f16 -> f16 [N_pos, 3*D]) */
    ink_wmma_gemm(vc, b->q_w->type, d_h, ws + pk_cur.off_qkv, ws + pk_cur.off_qkv_b, NULL, NULL,
                  d_qf, N_pos, 3u * D, D, 0u, 1u);
    /* 3. WMMA FlashAttention (hd=64) with inline fused 2D Axial RoPE */
    {
      void *qp = d_qf, *op = d_att;
      uint32_t nn = N_pos, nhl = nh, gwl = gw;
      float sc = scale;
      void *args[] = {&qp, &op, &nn, &nhl, &gwl, &sc};
      gpu_vis_launch(vc->k_attn_hd64, (N_pos + 127u) / 128u, nh, 1, 256, 1, 1, args);
    }
    /* 4. Attn Out + LayerScale 1 + Residual in-place into d_x (f32) */
    ink_wmma_gemm(vc, b->out_w->type, d_att, ws + pk_cur.off_out, ws + pk_cur.off_out_b,
                  ws + pk_cur.off_ls1, d_x, d_x, N_pos, D, D, 0u, 0u);
    /* 5. LN2 (f32 -> f16) */
    {
      void *xp = d_x, *lw = ws + pk_cur.off_ln2_w, *lb = ws + pk_cur.off_ln2_b;
      void *y16 = d_h, *y32 = NULL;
      uint32_t nn = N_pos, dd = D, in16 = 0u;
      float eps = vc->eps;
      void *args[] = {&xp, &lw, &lb, &y16, &y32, &nn, &dd, &eps, &in16};
      gpu_vis_launch(vc->k_ln_f16, N_pos, 1, 1, 256, 1, 1, args);
    }
    /* 6. FFN Up + Exact erf GELU (act=2, f16 out) */
    ink_wmma_gemm(vc, b->up_w->type, d_h, ws + pk_cur.off_up, ws + pk_cur.off_up_b, NULL, NULL,
                  d_qf, N_pos, F, D, 2u, 1u);
    /* 7. FFN Down + LayerScale 2 + Residual in-place into d_x (f32) */
    ink_wmma_gemm(vc, b->dn_w->type, d_qf, ws + pk_cur.off_dn, ws + pk_cur.off_dn_b,
                  ws + pk_cur.off_ls2, d_x, d_x, N_pos, D, F, 0u, 0u);

    gpu_vis_event_record(vc->ev_done[s], NULL);

    /* While GPU executes layer i, prepare and stream next payload into slot ns */
    gpu_vis_event_sync(vc->ev_done[ns]);
    if (i + 1u < vc->n_blk) {
      ink_layer_pack pk_next = ink_calc_pack(vc, &vc->blk[i + 1u]);
      ink_fill_pack(vc, &vc->blk[i + 1u], &pk_next, vc->pin_slot[ns]);
      gpu_vis_h2d_async(d_wslot[ns], vc->pin_slot[ns], pk_next.total_bytes, vc->vis_cs);
      gpu_vis_event_record(vc->ev_h2d[ns], vc->vis_cs);
      pk_cur = pk_next;
    } else {
      /* Prefetch mm.1 into slot ns */
      memcpy(vc->pin_slot[ns], vc->mm1_w->data, (size_t)vc->mm1_w->nbytes);
      gpu_vis_h2d_async(d_wslot[ns], vc->pin_slot[ns], (size_t)vc->mm1_w->nbytes, vc->vis_cs);
      gpu_vis_event_record(vc->ev_h2d[ns], vc->vis_cs);
    }
  }

  /* CLS strip + raster 2x2 merge (d_x f32 -> d_qf f16 [n_tok, 4*D]), overlapped with mm.1 H2D */
  {
    void *xp = d_x, *op = d_qf;
    uint32_t ghl = gh, gwl = gw, dd = D, skip_cls = 1u, in16 = 0u;
    void *args[] = {&xp, &op, &ghl, &gwl, &dd, &skip_cls, &in16};
    gpu_vis_launch(vc->k_merge2x2, n_tok, 1, 1, 256, 1, 1, args);
  }

  const int last_s = (int)((vc->n_blk - 1u) & 1u);
  const int mm1_s = last_s ^ 1;
  const int mm2_s = last_s;

  /* mm.1 (exact erf GELU, no bias): d_qf [n_tok, 4096] -> d_x [n_tok, 4096] (f16, d_x now dead!) */
  gpu_vis_stream_wait(NULL, vc->ev_h2d[mm1_s]);
  ink_wmma_gemm(vc, vc->mm1_w->type, d_qf, d_wslot[mm1_s], NULL, NULL, NULL,
                d_x, n_tok, mdim, mdim, 2u, 1u);

  /* Prefetch mm.2 into slot mm2_s while mm.1 runs */
  gpu_vis_event_sync(vc->ev_done[mm2_s]);
  memcpy(vc->pin_slot[mm2_s], vc->mm2_w->data, (size_t)vc->mm2_w->nbytes);
  gpu_vis_h2d_async(d_wslot[mm2_s], vc->pin_slot[mm2_s], (size_t)vc->mm2_w->nbytes, vc->vis_cs);
  gpu_vis_event_record(vc->ev_h2d[mm2_s], vc->vis_cs);

  /* mm.2 (no bias): d_x [n_tok, 4096] (f16) -> d_out [n_tok, 2304] (f32) */
  gpu_vis_stream_wait(NULL, vc->ev_h2d[mm2_s]);
  ink_wmma_gemm(vc, vc->mm2_w->type, d_x, d_wslot[mm2_s], NULL, NULL, NULL,
                d_out, n_tok, vc->proj_dim, mdim, 0u, 0u);

  gpu_vis_stream_sync(vc->vis_cs);
  gpu_vis_sync();
  gpu_vis_d2h(out_host, d_out, (size_t)n_tok * vc->proj_dim * 4u);
  gpu_vis_free(d_arena);
}

/* ---- Qwen3.5 GPU ViT forward pass (temporary workspace freed right after encode) ---- */
static void gpu_gemm_t(const vision_ctx *vc, const gguf_tensor *w, const gguf_tensor *bias,
                       void *d_w, void *d_b, const void *d_A, const void *d_res, void *d_C,
                       uint32_t N, uint32_t M, uint32_t K, uint32_t act) {
  gpu_vis_h2d(d_w, w->data, (size_t)w->nbytes);
  void *bp = NULL;
  if (bias) {
    gpu_vis_h2d(d_b, bias->data, (size_t)M * 4u);
    bp = d_b;
  }
  void *A = (void *)d_A, *W = d_w, *R = (void *)d_res, *C = d_C;
  void *args[] = {&A, &W, &bp, &R, &C, &N, &M, &K, &act};
  const unsigned gx = (M + 63u) / 64u, gy = (N + 63u) / 64u;
  if (w->type == 8) { /* Q8_0 */
    gpu_vis_launch(vc->k_gq8, gx, gy, 1, 256, 1, 1, args);
  } else if (w->type == 1) { /* F16 */
    gpu_vis_launch(vc->k_gf16, gx, gy, 1, 256, 1, 1, args);
  } else {
    ie_die("vision GPU: unsupported weight type %u (%s)", w->type, w->name);
  }
}

static void gpu_ln(const vision_ctx *vc, const gguf_tensor *w, const gguf_tensor *b,
                   void *d_b, const void *d_x, void *d_y, uint32_t N, uint32_t D) {
  char *bw = (char *)d_b;
  char *bb = bw + (size_t)D * 4u;
  gpu_vis_h2d(bw, w->data, (size_t)D * 4u);
  gpu_vis_h2d(bb, b->data, (size_t)D * 4u);
  void *X = (void *)d_x, *W = bw, *B = bb, *Y = d_y;
  float eps = vc->eps;
  void *args[] = {&X, &W, &B, &Y, &N, &D, &eps};
  gpu_vis_launch(vc->k_ln, N, 1, 1, 256, 1, 1, args);
}

static void vit_run_gpu(const vision_ctx *vc, const float *patches_host, const float *pos_host,
                        const float *rcos_host, const float *rsin_host, uint32_t N, float *out_host) {
  const uint32_t D = vc->dim, F = vc->ffn, nh = vc->n_head, hd = vc->hd;
  const uint32_t patch_in = 3u * vc->patch_size * vc->patch_size;
  const uint32_t m2 = vc->merge_size * vc->merge_size;
  const uint32_t n_tok = N / m2;
  const uint32_t mdim = D * m2; /* 4608 */
  const uint32_t qf_cols = F > 3u * D ? F : 3u * D;
  const size_t qf_bytes = (size_t)N * qf_cols * 4u > (size_t)n_tok * mdim * 4u
                              ? (size_t)N * qf_cols * 4u : (size_t)n_tok * mdim * 4u;

  void *d_x = gpu_vis_alloc((size_t)N * D * 4u);
  void *d_h = gpu_vis_alloc((size_t)N * D * 4u);
  void *d_qf = gpu_vis_alloc(qf_bytes);
  void *d_att = gpu_vis_alloc((size_t)N * D * 4u);
  void *d_rcos = gpu_vis_alloc((size_t)N * (hd / 2u) * 4u);
  void *d_rsin = gpu_vis_alloc((size_t)N * (hd / 2u) * 4u);
  const size_t pw_bytes = (size_t)D * patch_in * 4u;
  void *d_w = gpu_vis_alloc(vc->max_wbytes > pw_bytes ? vc->max_wbytes : pw_bytes);
  const size_t bmax = (size_t)(vc->proj_dim > 2u * D ? vc->proj_dim : 2u * D) * 4u;
  void *d_b = gpu_vis_alloc(bmax > (size_t)mdim * 4u ? bmax : (size_t)mdim * 4u);
  void *d_out = gpu_vis_alloc((size_t)n_tok * vc->proj_dim * 4u);

  /* Patch embedding on GPU: d_x[N, D] = patches[N, patch_in] @ w_patch[D, patch_in]^T + b_patch + pos_emb[N, D] */
  gpu_vis_h2d(d_qf, patches_host, (size_t)N * patch_in * 4u);
  gpu_vis_h2d(d_h, pos_host, (size_t)N * D * 4u);
  gpu_vis_h2d(d_w, vc->w_patch, pw_bytes);
  gpu_vis_h2d(d_b, vc->b_patch, (size_t)D * 4u);
  {
    uint32_t nn = N, mm = D, kk = patch_in;
    void *args[] = {&d_qf, &d_w, &d_b, &d_h, &d_x, &nn, &mm, &kk};
    gpu_vis_launch(vc->k_gf32, (D + 63u) / 64u, (N + 63u) / 64u, 1, 256, 1, 1, args);
  }
  gpu_vis_h2d(d_rcos, rcos_host, (size_t)N * (hd / 2u) * 4u);
  gpu_vis_h2d(d_rsin, rsin_host, (size_t)N * (hd / 2u) * 4u);

  const float scale = 1.0f / sqrtf((float)hd);
  for (uint32_t i = 0; i < vc->n_blk; i++) {
    const vis_blk *b = &vc->blk[i];
    gpu_ln(vc, b->ln1_w, b->ln1_b, d_b, d_x, d_h, N, D);
    gpu_gemm_t(vc, b->qkv_w, b->qkv_b, d_w, d_b, d_h, NULL, d_qf, N, 3u * D, D, 0);
    {
      uint32_t nn = N, nhl = nh, hdl = hd;
      void *args[] = {&d_qf, &d_rcos, &d_rsin, &nn, &nhl, &hdl};
      gpu_vis_launch(vc->k_rope, N, 1, 1, 256, 1, 1, args);
    }
    {
      uint32_t nn = N, nhl = nh, hdl = hd;
      float sc = scale;
      void *args[] = {&d_qf, &d_att, &nn, &nhl, &hdl, &sc};
      gpu_vis_launch(vc->k_attn, (N + 7u) / 8u, nh, 1, 256, 1, 1, args);
    }
    gpu_gemm_t(vc, b->out_w, b->out_b, d_w, d_b, d_att, d_x, d_x, N, D, D, 0);
    gpu_ln(vc, b->ln2_w, b->ln2_b, d_b, d_x, d_h, N, D);
    gpu_gemm_t(vc, b->up_w, b->up_b, d_w, d_b, d_h, NULL, d_qf, N, F, D, 1);
    gpu_gemm_t(vc, b->dn_w, b->dn_b, d_w, d_b, d_qf, d_x, d_x, N, D, F, 0);
  }
  /* Post-LN + 2x2 merger: d_h [N, D] viewed as [n_tok, 4*D] -> mm0 + GELU -> mm2 */
  gpu_ln(vc, vc->post_ln_w, vc->post_ln_b, d_b, d_x, d_h, N, D);
  gpu_gemm_t(vc, vc->mm0_w, vc->mm0_b, d_w, d_b, d_h, NULL, d_qf, n_tok, mdim, mdim, 1);
  gpu_gemm_t(vc, vc->mm2_w, vc->mm2_b, d_w, d_b, d_qf, NULL, d_out, n_tok, vc->proj_dim, mdim, 0);

  gpu_vis_sync();
  gpu_vis_d2h(out_host, d_out, (size_t)n_tok * vc->proj_dim * 4u);

  gpu_vis_free(d_out);
  gpu_vis_free(d_b);
  gpu_vis_free(d_w);
  gpu_vis_free(d_rsin);
  gpu_vis_free(d_rcos);
  gpu_vis_free(d_att);
  gpu_vis_free(d_qf);
  gpu_vis_free(d_h);
  gpu_vis_free(d_x);
}

/* ---- CPU fallback (for --backend cpu, no-GPU environments, and exact parity reference) ---- */
static void cpu_ln(const float *x, const float *w, const float *b, float *y, uint32_t N, uint32_t D, float eps) {
  for (uint32_t n = 0; n < N; n++) {
    const float *xn = x + (size_t)n * D;
    float *yn = y + (size_t)n * D;
    float mu = 0.0f;
    for (uint32_t i = 0; i < D; i++) mu += xn[i];
    mu /= (float)D;
    float v = 0.0f;
    for (uint32_t i = 0; i < D; i++) { float d = xn[i] - mu; v += d * d; }
    const float inv = 1.0f / sqrtf(v / (float)D + eps);
    for (uint32_t i = 0; i < D; i++) yn[i] = (xn[i] - mu) * inv * w[i] + b[i];
  }
}

static float cpu_gelu(float x) {
  const float z = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + tanhf(z));
}

static float cpu_gelu_erf(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f));
}

typedef struct {
  const gguf_tensor *w, *bias, *ls;
  const float *A, *res;
  float *C;
  uint32_t n0, n1, M, K;
  int act;
} cpu_gemm_job;

static void cpu_gemm_range(const cpu_gemm_job *j) {
  const gguf_tensor *w = j->w;
  const float *bp = j->bias ? (const float *)j->bias->data : NULL;
  const float *lp = j->ls ? (const float *)j->ls->data : NULL;
  const uint32_t M = j->M, K = j->K;
  const int act = j->act;
  uint32_t n = j->n0;
  for (; n + 4u <= j->n1; n += 4u) {
    const float *a0 = j->A + (size_t)(n + 0u) * K;
    const float *a1 = j->A + (size_t)(n + 1u) * K;
    const float *a2 = j->A + (size_t)(n + 2u) * K;
    const float *a3 = j->A + (size_t)(n + 3u) * K;
    float *c0 = j->C + (size_t)(n + 0u) * M;
    float *c1 = j->C + (size_t)(n + 1u) * M;
    float *c2 = j->C + (size_t)(n + 2u) * M;
    float *c3 = j->C + (size_t)(n + 3u) * M;
    const float *r0 = j->res ? j->res + (size_t)(n + 0u) * M : NULL;
    const float *r1 = j->res ? j->res + (size_t)(n + 1u) * M : NULL;
    const float *r2 = j->res ? j->res + (size_t)(n + 2u) * M : NULL;
    const float *r3 = j->res ? j->res + (size_t)(n + 3u) * M : NULL;
    for (uint32_t m = 0; m < M; m++) {
      float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
      if (w->type == 8) {
        const uint32_t nb = K / 32u;
        const uint8_t *row = (const uint8_t *)w->data + ((size_t)m * nb) * 34u;
        for (uint32_t b = 0; b < nb; b++) {
          const uint8_t *blk = row + (size_t)b * 34u;
          uint16_t dh;
          memcpy(&dh, blk, 2);
          const float d = F16_LUT[dh];
          const int8_t *qs = (const int8_t *)(blk + 2);
          const uint32_t bo = b * 32u;
          float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
          for (int k = 0; k < 32; k++) {
            const float qf = (float)qs[k];
            b0 += a0[bo + k] * qf;
            b1 += a1[bo + k] * qf;
            b2 += a2[bo + k] * qf;
            b3 += a3[bo + k] * qf;
          }
          s0 += d * b0; s1 += d * b1; s2 += d * b2; s3 += d * b3;
        }
      } else if (w->type == 1) {
        const uint16_t *row = (const uint16_t *)w->data + (size_t)m * K;
        for (uint32_t k = 0; k < K; k++) {
          const float wk = F16_LUT[row[k]];
          s0 += a0[k] * wk;
          s1 += a1[k] * wk;
          s2 += a2[k] * wk;
          s3 += a3[k] * wk;
        }
      } else {
        ie_die("vision CPU: unsupported weight type %u", w->type);
      }
      if (bp) { const float bv = bp[m]; s0 += bv; s1 += bv; s2 += bv; s3 += bv; }
      if (act == 1) { s0 = cpu_gelu(s0); s1 = cpu_gelu(s1); s2 = cpu_gelu(s2); s3 = cpu_gelu(s3); }
      else if (act == 2) { s0 = cpu_gelu_erf(s0); s1 = cpu_gelu_erf(s1); s2 = cpu_gelu_erf(s2); s3 = cpu_gelu_erf(s3); }
      if (lp) { const float lv = lp[m]; s0 *= lv; s1 *= lv; s2 *= lv; s3 *= lv; }
      if (r0) { s0 += r0[m]; s1 += r1[m]; s2 += r2[m]; s3 += r3[m]; }
      c0[m] = s0; c1[m] = s1; c2[m] = s2; c3[m] = s3;
    }
  }
  for (; n < j->n1; n++) {
    const float *an = j->A + (size_t)n * K;
    float *cn = j->C + (size_t)n * M;
    const float *rn = j->res ? j->res + (size_t)n * M : NULL;
    for (uint32_t m = 0; m < M; m++) {
      float s = 0.0f;
      if (w->type == 8) {
        const uint32_t nb = K / 32u;
        const uint8_t *row = (const uint8_t *)w->data + ((size_t)m * nb) * 34u;
        for (uint32_t b = 0; b < nb; b++) {
          const uint8_t *blk = row + (size_t)b * 34u;
          uint16_t dh;
          memcpy(&dh, blk, 2);
          const float d = F16_LUT[dh];
          const int8_t *qs = (const int8_t *)(blk + 2);
          const float *ak = an + b * 32u;
          float bs = 0.0f;
          for (int k = 0; k < 32; k++) bs += ak[k] * (float)qs[k];
          s += d * bs;
        }
      } else if (w->type == 1) {
        const uint16_t *row = (const uint16_t *)w->data + (size_t)m * K;
        for (uint32_t k = 0; k < K; k++) s += an[k] * F16_LUT[row[k]];
      } else {
        ie_die("vision CPU: unsupported weight type %u", w->type);
      }
      if (bp) s += bp[m];
      if (act == 1) s = cpu_gelu(s);
      else if (act == 2) s = cpu_gelu_erf(s);
      if (lp) s *= lp[m];
      if (rn) s += rn[m];
      cn[m] = s;
    }
  }
}

static void *cpu_gemm_worker(void *arg) {
  cpu_gemm_range((const cpu_gemm_job *)arg);
  return NULL;
}

static void cpu_gemm_t_ex(const gguf_tensor *w, const gguf_tensor *bias, const gguf_tensor *ls,
                          const float *A, const float *res, float *C,
                          uint32_t N, uint32_t M, uint32_t K, int act) {
  enum { MAX_TH = 16 };
  if (N < 32u) {
    const cpu_gemm_job j = {w, bias, ls, A, res, C, 0, N, M, K, act};
    cpu_gemm_range(&j);
    return;
  }
  uint32_t nt = N < MAX_TH ? N : MAX_TH;
  pthread_t th[MAX_TH];
  cpu_gemm_job jobs[MAX_TH];
  for (uint32_t t = 0; t < nt; t++) {
    uint32_t n0 = (t * N) / nt, n1 = ((t + 1u) * N) / nt;
    jobs[t] = (cpu_gemm_job){w, bias, ls, A, res, C, n0, n1, M, K, act};
    if (t > 0u) pthread_create(&th[t], NULL, cpu_gemm_worker, &jobs[t]);
  }
  cpu_gemm_range(&jobs[0]);
  for (uint32_t t = 1u; t < nt; t++) pthread_join(th[t], NULL);
}

static void cpu_gemm_t(const gguf_tensor *w, const gguf_tensor *bias, const float *A, const float *res, float *C,
                       uint32_t N, uint32_t M, uint32_t K, int act) {
  cpu_gemm_t_ex(w, bias, NULL, A, res, C, N, M, K, act);
}

static void vit_run_cpu(const vision_ctx *vc, const float *x_in, const float *rcos, const float *rsin,
                        uint32_t N, float *out) {
  const uint32_t D = vc->dim, F = vc->ffn, nh = vc->n_head, hd = vc->hd, half = hd / 2u;
  const uint32_t m2 = vc->merge_size * vc->merge_size, n_tok = N / m2, mdim = D * m2;
  const uint32_t qfc = F > 3u * D ? F : 3u * D;
  float *x = malloc((size_t)N * D * 4u);
  float *h = malloc((size_t)N * D * 4u);
  float *qf = malloc(((size_t)N * qfc > (size_t)n_tok * mdim ? (size_t)N * qfc : (size_t)n_tok * mdim) * 4u);
  float *att = malloc((size_t)N * D * 4u);
  float *sc = malloc((size_t)N * 4u);
  if (!x || !h || !qf || !att || !sc) ie_die("vision: out of memory");
  memcpy(x, x_in, (size_t)N * D * 4u);
  const float scale = 1.0f / sqrtf((float)hd);

  for (uint32_t i = 0; i < vc->n_blk; i++) {
    const vis_blk *b = &vc->blk[i];
    cpu_ln(x, b->ln1_w->data, b->ln1_b->data, h, N, D, vc->eps);
    cpu_gemm_t(b->qkv_w, b->qkv_b, h, NULL, qf, N, 3u * D, D, 0);
    for (uint32_t n = 0; n < N; n++) {
      float *qn = qf + (size_t)n * 3u * D, *kn = qn + D;
      const float *cs = rcos + (size_t)n * half, *sn = rsin + (size_t)n * half;
      for (uint32_t hh = 0; hh < nh; hh++) {
        for (uint32_t d = 0; d < half; d++) {
          const uint32_t o0 = hh * hd + d, o1 = o0 + half;
          const float q0 = qn[o0], q1 = qn[o1], k0 = kn[o0], k1 = kn[o1];
          qn[o0] = q0 * cs[d] - q1 * sn[d], qn[o1] = q0 * sn[d] + q1 * cs[d];
          kn[o0] = k0 * cs[d] - k1 * sn[d], kn[o1] = k0 * sn[d] + k1 * cs[d];
        }
      }
    }
    for (uint32_t hh = 0; hh < nh; hh++) {
      for (uint32_t qi = 0; qi < N; qi++) {
        const float *qp = qf + (size_t)qi * 3u * D + hh * hd;
        float mx = -INFINITY;
        for (uint32_t ki = 0; ki < N; ki++) {
          const float *kp = qf + (size_t)ki * 3u * D + D + hh * hd;
          float dot = 0.0f;
          for (uint32_t d = 0; d < hd; d++) dot += qp[d] * kp[d];
          sc[ki] = dot * scale;
          if (sc[ki] > mx) mx = sc[ki];
        }
        float z = 0.0f;
        for (uint32_t ki = 0; ki < N; ki++) { sc[ki] = expf(sc[ki] - mx); z += sc[ki]; }
        float *op = att + (size_t)qi * D + hh * hd;
        for (uint32_t d = 0; d < hd; d++) op[d] = 0.0f;
        const float inv = 1.0f / z;
        for (uint32_t ki = 0; ki < N; ki++) {
          const float wv = sc[ki] * inv;
          const float *vp = qf + (size_t)ki * 3u * D + 2u * D + hh * hd;
          for (uint32_t d = 0; d < hd; d++) op[d] += wv * vp[d];
        }
      }
    }
    cpu_gemm_t(b->out_w, b->out_b, att, x, x, N, D, D, 0);
    cpu_ln(x, b->ln2_w->data, b->ln2_b->data, h, N, D, vc->eps);
    cpu_gemm_t(b->up_w, b->up_b, h, NULL, qf, N, F, D, 1);
    cpu_gemm_t(b->dn_w, b->dn_b, qf, x, x, N, D, F, 0);
  }
  cpu_ln(x, vc->post_ln_w->data, vc->post_ln_b->data, h, N, D, vc->eps);
  cpu_gemm_t(vc->mm0_w, vc->mm0_b, h, NULL, qf, n_tok, mdim, mdim, 1);
  cpu_gemm_t(vc->mm2_w, vc->mm2_b, qf, NULL, out, n_tok, vc->proj_dim, mdim, 0);
  free(sc); free(att); free(qf); free(h); free(x);
}

typedef struct {
  const float *q, *k, *v;
  float *att;
  uint32_t N_pos, D, hd, h0, h1;
  float scale;
} cpu_attn_job;

static void *cpu_attn_worker(void *arg) {
  const cpu_attn_job *j = (const cpu_attn_job *)arg;
  const uint32_t N_pos = j->N_pos, D = j->D, hd = j->hd;
  const float scale = j->scale;
  float *sc = malloc((size_t)N_pos * 4u);
  if (!sc) ie_die("vision: out of memory");
  for (uint32_t hh = j->h0; hh < j->h1; hh++) {
    for (uint32_t qi = 0; qi < N_pos; qi++) {
      const float *qp = j->q + (size_t)qi * D + hh * hd;
      float mx = -INFINITY;
      for (uint32_t ki = 0; ki < N_pos; ki++) {
        const float *kp = j->k + (size_t)ki * D + hh * hd;
        float dot = 0.0f;
        for (uint32_t d = 0; d < hd; d++) dot += qp[d] * kp[d];
        sc[ki] = dot * scale;
        if (sc[ki] > mx) mx = sc[ki];
      }
      float z = 0.0f;
      for (uint32_t ki = 0; ki < N_pos; ki++) { sc[ki] = expf(sc[ki] - mx); z += sc[ki]; }
      float *op = j->att + (size_t)qi * D + hh * hd;
      for (uint32_t d = 0; d < hd; d++) op[d] = 0.0f;
      const float inv = 1.0f / z;
      for (uint32_t ki = 0; ki < N_pos; ki++) {
        const float wv = sc[ki] * inv;
        const float *vp = j->v + (size_t)ki * D + hh * hd;
        for (uint32_t d = 0; d < hd; d++) op[d] += wv * vp[d];
      }
    }
  }
  free(sc);
  return NULL;
}

/* Exact CPU reference for Inkling-2-Mini (inkling2v) matching llama.cpp PR #32217:
 * - Row 0 = CLS (class_embd + class_position_embd), rows 1..N = patch_embd + pos_embd
 * - v.pre_ln before blk.0, NO v.post_ln
 * - 2D Axial RoPE (interleaved pairs mode=0, pos_w on [0..31], pos_h on [32..63], CLS=(0,0))
 * - LayerScale ls1/ls2 + exact erff GELU in FFN and mm.1
 * - Strip CLS row 0 -> raster 2x2 merge [gh, gw, D] -> [n_tok, 4*D] -> mm.1 -> mm.2 */
static void vit_run_inkling2_cpu(const vision_ctx *vc, const float *patches, const float *pos_emb,
                                 const float *rcos, const float *rsin, uint32_t gw, uint32_t gh, float *out) {
  const uint32_t D = vc->dim, F = vc->ffn, nh = vc->n_head, hd = vc->hd;
  const uint32_t half = hd / 2u, qtr = half / 2u; /* 32, 16 */
  const uint32_t N_patches = gw * gh;
  const uint32_t N_pos = N_patches + 1u;
  const uint32_t patch_in = 3u * vc->patch_size * vc->patch_size;
  const uint32_t ny = gh / vc->merge_size, nx = gw / vc->merge_size;
  const uint32_t n_tok = nx * ny, mdim = 4u * D;

  float *x = malloc((size_t)N_pos * D * 4u);
  float *h = malloc((size_t)N_pos * D * 4u);
  float *q = malloc((size_t)N_pos * D * 4u);
  float *k = malloc((size_t)N_pos * D * 4u);
  float *v = malloc((size_t)N_pos * D * 4u);
  float *att = malloc((size_t)N_pos * D * 4u);
  float *ff = malloc(((size_t)N_pos * F > (size_t)n_tok * mdim ? (size_t)N_pos * F : (size_t)n_tok * mdim) * 4u);
  float *merged = malloc((size_t)n_tok * mdim * 4u);
  if (!x || !h || !q || !k || !v || !att || !ff || !merged) ie_die("vision: out of memory");

  /* CLS token (row 0) + patch embedding + pos_emb (rows 1..N_patches) */
  for (uint32_t d = 0; d < D; d++) x[d] = vc->cls_embd[d] + vc->cls_pos_embd[d];
  for (uint32_t idx = 0; idx < N_patches; idx++) {
    const float *pt = patches + (size_t)idx * patch_in;
    const float *pe = pos_emb + (size_t)idx * D;
    float *xd = x + (size_t)(1u + idx) * D;
    for (uint32_t d = 0; d < D; d++) {
      const float *wd = vc->w_patch + (size_t)d * patch_in;
      float s = pe[d];
      for (uint32_t j = 0; j < patch_in; j++) s += wd[j] * pt[j];
      xd[d] = s;
    }
  }
  /* Pre-LayerNorm before blk.0 */
  cpu_ln(x, vc->pre_ln_w->data, vc->pre_ln_b->data, x, N_pos, D, vc->eps);

  const float scale = 1.0f / sqrtf((float)hd);
  for (uint32_t i = 0; i < vc->n_blk; i++) {
    const vis_blk *b = &vc->blk[i];
    cpu_ln(x, b->ln1_w->data, b->ln1_b->data, h, N_pos, D, vc->eps);
    cpu_gemm_t_ex(b->q_w, b->q_b, NULL, h, NULL, q, N_pos, D, D, 0);
    cpu_gemm_t_ex(b->k_w, b->k_b, NULL, h, NULL, k, N_pos, D, D, 0);
    cpu_gemm_t_ex(b->v_w, b->v_b, NULL, h, NULL, v, N_pos, D, D, 0);

    /* Interleaved 2D Axial RoPE (mode=0): [0..31] uses pos_w (rcos[0..15]), [32..63] uses pos_h (rcos[16..31]) */
    for (uint32_t n = 0; n < N_pos; n++) {
      float *qn = q + (size_t)n * D;
      float *kn = k + (size_t)n * D;
      const float *cs = rcos + (size_t)n * half;
      const float *sn = rsin + (size_t)n * half;
      for (uint32_t hh = 0; hh < nh; hh++) {
        float *qh = qn + hh * hd;
        float *kh = kn + hh * hd;
        for (uint32_t j = 0; j < qtr; j++) {
          const float cw = cs[j], sw = sn[j];
          const float ch = cs[qtr + j], sh = sn[qtr + j];
          const uint32_t i0 = 2u * j, i1 = i0 + 1u;
          const uint32_t j0 = half + 2u * j, j1 = j0 + 1u;
          const float q0 = qh[i0], q1 = qh[i1], k0 = kh[i0], k1 = kh[i1];
          qh[i0] = q0 * cw - q1 * sw; qh[i1] = q0 * sw + q1 * cw;
          kh[i0] = k0 * cw - k1 * sw; kh[i1] = k0 * sw + k1 * cw;
          const float q2 = qh[j0], q3 = qh[j1], k2 = kh[j0], k3 = kh[j1];
          qh[j0] = q2 * ch - q3 * sh; qh[j1] = q2 * sh + q3 * ch;
          kh[j0] = k2 * ch - k3 * sh; kh[j1] = k2 * sh + k3 * ch;
        }
      }
    }

    {
      enum { MAX_ATH = 16 };
      const uint32_t nt = nh < MAX_ATH ? nh : MAX_ATH;
      pthread_t ath[MAX_ATH];
      cpu_attn_job aj[MAX_ATH];
      for (uint32_t t = 0; t < nt; t++) {
        uint32_t h0 = (t * nh) / nt, h1 = ((t + 1u) * nh) / nt;
        aj[t] = (cpu_attn_job){q, k, v, att, N_pos, D, hd, h0, h1, scale};
        if (t > 0u) pthread_create(&ath[t], NULL, cpu_attn_worker, &aj[t]);
      }
      cpu_attn_worker(&aj[0]);
      for (uint32_t t = 1u; t < nt; t++) pthread_join(ath[t], NULL);
    }

    cpu_gemm_t_ex(b->out_w, b->out_b, b->ls1_w, att, x, x, N_pos, D, D, 0);
    cpu_ln(x, b->ln2_w->data, b->ln2_b->data, h, N_pos, D, vc->eps);
    cpu_gemm_t_ex(b->up_w, b->up_b, NULL, h, NULL, ff, N_pos, F, D, 2);
    cpu_gemm_t_ex(b->dn_w, b->dn_b, b->ls2_w, ff, x, x, N_pos, D, F, 0);
  }

  /* Strip CLS row 0 and merge raster 2x2 patches [gh, gw, D] -> [n_tok, 4*D] */
  const float *xp = x + D;
  for (uint32_t my = 0; my < ny; my++) {
    for (uint32_t mx = 0; mx < nx; mx++) {
      float *dst = merged + (size_t)(my * nx + mx) * mdim;
      const uint32_t r00 = (2u * my + 0u) * gw + (2u * mx + 0u);
      const uint32_t r01 = (2u * my + 0u) * gw + (2u * mx + 1u);
      const uint32_t r10 = (2u * my + 1u) * gw + (2u * mx + 0u);
      const uint32_t r11 = (2u * my + 1u) * gw + (2u * mx + 1u);
      memcpy(dst + 0u * D, xp + (size_t)r00 * D, (size_t)D * 4u);
      memcpy(dst + 1u * D, xp + (size_t)r01 * D, (size_t)D * 4u);
      memcpy(dst + 2u * D, xp + (size_t)r10 * D, (size_t)D * 4u);
      memcpy(dst + 3u * D, xp + (size_t)r11 * D, (size_t)D * 4u);
    }
  }
  /* Projector: mm.1 (exact erf GELU, no bias) -> mm.2 (no bias) */
  cpu_gemm_t_ex(vc->mm1_w, vc->mm1_b, NULL, merged, NULL, ff, n_tok, mdim, mdim, 2);
  cpu_gemm_t_ex(vc->mm2_w, vc->mm2_b, NULL, ff, NULL, out, n_tok, vc->proj_dim, mdim, 0);

  free(merged); free(ff); free(att); free(v); free(k); free(q); free(h); free(x);
}

int vision_encode_f32(vision_ctx *vc, const float *rgb_chw, uint32_t w, uint32_t h, vision_emb *out) {
  const uint32_t factor = vc->patch_size * vc->merge_size;
  if (!w || !h || (w % factor) != 0 || (h % factor) != 0) return 0;
  const uint32_t gh = h / vc->patch_size, gw = w / vc->patch_size;
  const uint32_t N = gh * gw;
  const uint32_t ny = gh / vc->merge_size, nx = gw / vc->merge_size;
  const uint32_t n_tok = nx * ny;
  const uint32_t patch_in = 3u * vc->patch_size * vc->patch_size;
  const int force_cpu = getenv("IE_VISION_CPU") && getenv("IE_VISION_CPU")[0] == '1';

  float *embd = malloc((size_t)n_tok * vc->proj_dim * 4u);
  if (!embd) ie_die("vision: out of memory");

  if (vc->arch == VIS_ARCH_INKLING2V) {
    if (vc->mod && vc->has_wmma && !force_cpu) {
      uint16_t *patches_f16 = malloc((size_t)N * patch_in * 2u);
      float *pos_emb = malloc((size_t)N * vc->dim * 4u);
      if (!patches_f16 || !pos_emb) ie_die("vision: out of memory");
      prep_inputs_inkling2_gpu(vc, rgb_chw, w, h, patches_f16, pos_emb);
      vit_run_inkling2_gpu(vc, patches_f16, pos_emb, gw, gh, embd);
      free(pos_emb); free(patches_f16);
    } else {
      const uint32_t N_pos = N + 1u;
      float *patches = malloc((size_t)N * patch_in * 4u);
      float *pos_emb = malloc((size_t)N * vc->dim * 4u);
      float *rcos = malloc((size_t)N_pos * (vc->hd / 2u) * 4u);
      float *rsin = malloc((size_t)N_pos * (vc->hd / 2u) * 4u);
      if (!patches || !pos_emb || !rcos || !rsin) ie_die("vision: out of memory");
      prep_inputs_inkling2(vc, rgb_chw, w, h, patches, pos_emb, rcos, rsin);
      vit_run_inkling2_cpu(vc, patches, pos_emb, rcos, rsin, gw, gh, embd);
      free(rsin); free(rcos); free(pos_emb); free(patches);
    }
  } else {
    float *patches = malloc((size_t)N * patch_in * 4u);
    float *pos_emb = malloc((size_t)N * vc->dim * 4u);
    float *rcos = malloc((size_t)N * (vc->hd / 2u) * 4u);
    float *rsin = malloc((size_t)N * (vc->hd / 2u) * 4u);
    if (!patches || !pos_emb || !rcos || !rsin) ie_die("vision: out of memory");

    prep_inputs(vc, rgb_chw, w, h, patches, pos_emb, rcos, rsin);
    if (vc->mod && !force_cpu) {
      vit_run_gpu(vc, patches, pos_emb, rcos, rsin, N, embd);
    } else {
      float *x0 = malloc((size_t)N * vc->dim * 4u);
      if (!x0) ie_die("vision: out of memory");
      for (uint32_t t = 0; t < N; t++) {
        const float *pt = patches + (size_t)t * patch_in;
        const float *pe = pos_emb + (size_t)t * vc->dim;
        float *xd = x0 + (size_t)t * vc->dim;
        for (uint32_t d = 0; d < vc->dim; d++) {
          const float *wd = vc->w_patch + (size_t)d * patch_in;
          float s = vc->b_patch[d] + pe[d];
          for (uint32_t k = 0; k < patch_in; k++) s += wd[k] * pt[k];
          xd[d] = s;
        }
      }
      vit_run_cpu(vc, x0, rcos, rsin, N, embd);
      free(x0);
    }
    free(rsin); free(rcos); free(pos_emb); free(patches);
  }

  out->embd = embd;
  out->n_tok = n_tok;
  out->nx = nx;
  out->ny = ny;
  out->proj_dim = vc->proj_dim;
  out->hash = fnv1a64(embd, (size_t)n_tok * vc->proj_dim * 4u);
  out->owned = 1;
  return 1;
}

int vision_encode_url(vision_ctx *vc, const char *url, size_t len, vision_emb *out, char *err, size_t errn) {
  const uint64_t h = fnv1a64(url, len);
  for (int i = 0; i < VIS_CACHE_N; i++) {
    vis_cache_ent *e = &vc->cache[i];
    if (e->embd && e->hash == h && e->url_len == len) {
      e->used = ++vc->tick;
      const size_t bytes = (size_t)e->n_tok * e->proj_dim * 4u;
      float *cp = malloc(bytes);
      if (!cp) ie_die("vision: out of memory");
      memcpy(cp, e->embd, bytes);
      out->embd = cp;
      out->n_tok = e->n_tok;
      out->nx = e->nx;
      out->ny = e->ny;
      out->proj_dim = e->proj_dim;
      out->hash = e->hash;
      out->owned = 1;
      fprintf(stderr, "vision: cache hit (%ux%u grid, %u tokens)\n", e->nx, e->ny, e->n_tok);
      return 1;
    }
  }
  size_t raw_n = 0;
  uint8_t *raw = url_to_bytes(url, len, &raw_n, err, errn);
  if (!raw) return 0;
  uint32_t iw = 0, ih = 0;
  uint8_t *rgb = decode_image_rgb(raw, raw_n, &iw, &ih, err, errn);
  free(raw);
  if (!rgb) return 0;

  const char *emin = getenv("IE_IMAGE_MIN_TOKENS"), *emax = getenv("IE_IMAGE_MAX_TOKENS");
  uint32_t min_tok = emin && atoi(emin) > 0 ? (uint32_t)atoi(emin) : 64u;
  uint32_t max_tok = emax && atoi(emax) > 0 ? (uint32_t)atoi(emax) : 1024u;
  if (max_tok > 4096u) max_tok = 4096u;
  if (min_tok > max_tok) min_tok = max_tok;

  uint32_t ow = 0, oh = 0;
  float *chw = NULL;
  if (vc->arch == VIS_ARCH_INKLING2V) {
    chw = inkling2_resize_pad_norm(rgb, iw, ih, vc->patch_size * vc->merge_size, max_tok, vc->mean, vc->std, &ow, &oh);
  } else {
    smart_resize(iw, ih, vc->patch_size * vc->merge_size, min_tok, max_tok, &ow, &oh);
    chw = resize_and_norm(rgb, iw, ih, ow, oh, vc->mean, vc->std);
  }
  free(rgb);

  vision_emb tmp = {0};
  const double t0 = now_ms();
  const int ok = vision_encode_f32(vc, chw, ow, oh, &tmp);
  const double dt = now_ms() - t0;
  free(chw);
  if (!ok) {
    snprintf(err, errn, "vision encode failed for %ux%u", ow, oh);
    return 0;
  }
  fprintf(stderr, "vision: encoded %ux%u -> %ux%u (%ux%u grid, %u tokens) in %.1f ms\n",
          iw, ih, ow, oh, tmp.nx, tmp.ny, tmp.n_tok, dt);

  /* Insert independent copy into LRU cache so caller always owns out->embd safely */
  int slot = 0;
  for (int i = 0; i < VIS_CACHE_N; i++) {
    if (!vc->cache[i].embd) { slot = i; break; }
    if (vc->cache[i].used < vc->cache[slot].used) slot = i;
  }
  const size_t bytes = (size_t)tmp.n_tok * tmp.proj_dim * 4u;
  float *cached_copy = malloc(bytes);
  if (cached_copy) {
    memcpy(cached_copy, tmp.embd, bytes);
    free(vc->cache[slot].embd);
    vc->cache[slot] = (vis_cache_ent){
        .hash = h, .url_len = len, .n_tok = tmp.n_tok, .nx = tmp.nx, .ny = tmp.ny,
        .proj_dim = tmp.proj_dim, .used = ++vc->tick, .embd = cached_copy,
    };
  }
  *out = tmp;
  out->hash = h;
  out->owned = 1;
  return 1;
}

void vision_emb_free(vision_emb *e) {
  if (e && e->owned) free(e->embd);
  if (e) memset(e, 0, sizeof *e);
}
