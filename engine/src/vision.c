/* vision.c -- Qwen3.5/3.8 ViT vision projector (see vision.h). */
#define COBJMACROS
#include <math.h>
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

typedef struct {
  const gguf_tensor *ln1_w, *ln1_b;
  const gguf_tensor *qkv_w, *qkv_b;
  const gguf_tensor *out_w, *out_b;
  const gguf_tensor *ln2_w, *ln2_b;
  const gguf_tensor *up_w, *up_b;
  const gguf_tensor *dn_w, *dn_b;
} vis_blk;

typedef struct {
  uint64_t hash;
  size_t url_len;
  uint32_t n_tok, nx, ny, proj_dim;
  uint64_t used;
  float *embd;
} vis_cache_ent;

struct vision_ctx {
  gguf_file g;
  uint32_t patch_size; /* 16 */
  uint32_t merge_size; /* 2 */
  uint32_t dim;        /* 1152 */
  uint32_t ffn;        /* 4304 */
  uint32_t n_blk;      /* 27 */
  uint32_t n_head;     /* 16 */
  uint32_t hd;         /* 72 */
  uint32_t proj_dim;   /* 5120 */
  uint32_t pos_grid;   /* 48 */
  float eps;           /* 1e-6 */
  float mean[3], std[3];
  float *w_patch;      /* [dim * 3 * patch_size * patch_size] = patch_embd.weight + weight.1 */
  float *b_patch;      /* [dim] */
  float *pos_embd;     /* [pos_grid * pos_grid * dim] */
  vis_blk blk[VIS_MAX_BLK];
  const gguf_tensor *post_ln_w, *post_ln_b;
  const gguf_tensor *mm0_w, *mm0_b;
  const gguf_tensor *mm2_w, *mm2_b;
  size_t max_wbytes;   /* largest single weight matrix byte size */
  /* GPU kernels (NULL when CPU-only) */
  void *mod;
  void *k_ln, *k_gq8, *k_gf16, *k_gf32, *k_rope, *k_attn;
  /* LRU cache of encoded images */
  vis_cache_ent cache[VIS_CACHE_N];
  uint64_t tick;
};

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

vision_ctx *vision_open(const char *mmproj_path, const char *hsaco_hint, uint32_t expected_proj_dim) {
  vision_ctx *vc = calloc(1, sizeof *vc);
  if (!vc) ie_die("vision: out of memory");
  gguf_open(&vc->g, mmproj_path);
  vc->patch_size = kv_u32(&vc->g, "clip.vision.patch_size", 16);
  vc->merge_size = kv_u32(&vc->g, "clip.vision.spatial_merge_size", 2);
  vc->dim = kv_u32(&vc->g, "clip.vision.embedding_length", 1152);
  vc->ffn = kv_u32(&vc->g, "clip.vision.feed_forward_length", 4304);
  vc->n_blk = kv_u32(&vc->g, "clip.vision.block_count", 27);
  vc->n_head = kv_u32(&vc->g, "clip.vision.attention.head_count", 16);
  vc->proj_dim = kv_u32(&vc->g, "clip.vision.projection_dim", 5120);
  double eps = 1e-6;
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

  /* Patch embedding: sum temporal frames weight + weight.1 for still images */
  const gguf_tensor *pw0 = need_t(&vc->g, "v.patch_embd.weight");
  const gguf_tensor *pw1 = gguf_tensor_find(&vc->g, "v.patch_embd.weight.1");
  const gguf_tensor *pb = need_t(&vc->g, "v.patch_embd.bias");
  const size_t patch_in = (size_t)3u * vc->patch_size * vc->patch_size;
  const size_t pnumel = (size_t)vc->dim * patch_in;
  vc->w_patch = malloc(pnumel * 4u);
  vc->b_patch = malloc((size_t)vc->dim * 4u);
  if (!vc->w_patch || !vc->b_patch) ie_die("vision: out of memory");
  tensor_to_f32(pw0, vc->w_patch, pnumel);
  if (pw1) {
    float *tmp = malloc(pnumel * 4u);
    if (!tmp) ie_die("vision: out of memory");
    tensor_to_f32(pw1, tmp, pnumel);
    for (size_t i = 0; i < pnumel; i++) vc->w_patch[i] += tmp[i];
    free(tmp);
  }
  tensor_to_f32(pb, vc->b_patch, vc->dim);

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
#undef GETB
    const gguf_tensor *ws[4] = {b->qkv_w, b->out_w, b->up_w, b->dn_w};
    for (int k = 0; k < 4; k++)
      if (ws[k]->nbytes > vc->max_wbytes) vc->max_wbytes = (size_t)ws[k]->nbytes;
  }
  vc->post_ln_w = need_t(&vc->g, "v.post_ln.weight");
  vc->post_ln_b = need_t(&vc->g, "v.post_ln.bias");
  vc->mm0_w = need_t(&vc->g, "mm.0.weight");
  vc->mm0_b = need_t(&vc->g, "mm.0.bias");
  vc->mm2_w = need_t(&vc->g, "mm.2.weight");
  vc->mm2_b = need_t(&vc->g, "mm.2.bias");
  if (vc->mm0_w->nbytes > vc->max_wbytes) vc->max_wbytes = (size_t)vc->mm0_w->nbytes;
  if (vc->mm2_w->nbytes > vc->max_wbytes) vc->max_wbytes = (size_t)vc->mm2_w->nbytes;

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
      if (!vc->k_ln || !vc->k_gq8 || !vc->k_gf16 || !vc->k_gf32 || !vc->k_rope || !vc->k_attn) {
        gpu_vis_unload_mod(vc->mod);
        vc->mod = NULL;
      }
    }
    fprintf(stderr, "vision: loaded %s (%u blk, dim %u -> %u, %s)\n", mmproj_path, vc->n_blk, vc->dim, vc->proj_dim,
            vc->mod ? "GPU zero-persistent-VRAM" : "CPU fallback");
  } else {
    fprintf(stderr, "vision: loaded %s (%u blk, dim %u -> %u, CPU)\n", mmproj_path, vc->n_blk, vc->dim, vc->proj_dim);
  }
  return vc;
}

void vision_close(vision_ctx *vc) {
  if (!vc) return;
  for (int i = 0; i < VIS_CACHE_N; i++) free(vc->cache[i].embd);
  if (vc->mod) gpu_vis_unload_mod(vc->mod);
  free(vc->w_patch);
  free(vc->b_patch);
  free(vc->pos_embd);
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

/* Separable bicubic resize RGB8 [ih, iw, 3] -> normalized f32 CHW [3, oh, ow] */
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

/* ---- Host preprocessing: extract 2x2-merged patches [N, patch_in], bilinear pos_emb [N, D], and 2D RoPE ---- */
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
          /* Extract patch [3, P, P] into patches_out[tok] */
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

/* ---- GPU ViT forward pass (temporary workspace freed right after encode) ---- */
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

/* ---- CPU fallback (for --backend cpu or no-GPU environments) ---- */
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

static void cpu_gemm_t(const gguf_tensor *w, const gguf_tensor *bias, const float *A, const float *res, float *C,
                       uint32_t N, uint32_t M, uint32_t K, int act) {
  const float *bp = bias ? (const float *)bias->data : NULL;
  for (uint32_t n = 0; n < N; n++) {
    const float *an = A + (size_t)n * K;
    float *cn = C + (size_t)n * M;
    const float *rn = res ? res + (size_t)n * M : NULL;
    for (uint32_t m = 0; m < M; m++) {
      float s = 0.0f;
      if (w->type == 8) {
        const uint32_t nb = K / 32u;
        const uint8_t *row = (const uint8_t *)w->data + ((size_t)m * nb) * 34u;
        for (uint32_t b = 0; b < nb; b++) {
          const uint8_t *blk = row + (size_t)b * 34u;
          uint16_t dh;
          memcpy(&dh, blk, 2);
          const float d = f16_to_f32(dh);
          const int8_t *qs = (const int8_t *)(blk + 2);
          const float *ak = an + b * 32u;
          float bs = 0.0f;
          for (int j = 0; j < 32; j++) bs += ak[j] * (float)qs[j];
          s += d * bs;
        }
      } else if (w->type == 1) {
        const uint16_t *row = (const uint16_t *)w->data + (size_t)m * K;
        for (uint32_t k = 0; k < K; k++) s += an[k] * f16_to_f32(row[k]);
      } else {
        ie_die("vision CPU: unsupported weight type %u", w->type);
      }
      if (bp) s += bp[m];
      if (act) s = cpu_gelu(s);
      if (rn) s += rn[m];
      cn[m] = s;
    }
  }
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

int vision_encode_f32(vision_ctx *vc, const float *rgb_chw, uint32_t w, uint32_t h, vision_emb *out) {
  const uint32_t factor = vc->patch_size * vc->merge_size;
  if (!w || !h || (w % factor) != 0 || (h % factor) != 0) return 0;
  const uint32_t gh = h / vc->patch_size, gw = w / vc->patch_size;
  const uint32_t N = gh * gw;
  const uint32_t ny = gh / vc->merge_size, nx = gw / vc->merge_size;
  const uint32_t n_tok = nx * ny;
  const uint32_t patch_in = 3u * vc->patch_size * vc->patch_size;
  float *patches = malloc((size_t)N * patch_in * 4u);
  float *pos_emb = malloc((size_t)N * vc->dim * 4u);
  float *rcos = malloc((size_t)N * (vc->hd / 2u) * 4u);
  float *rsin = malloc((size_t)N * (vc->hd / 2u) * 4u);
  float *embd = malloc((size_t)n_tok * vc->proj_dim * 4u);
  if (!patches || !pos_emb || !rcos || !rsin || !embd) ie_die("vision: out of memory");

  prep_inputs(vc, rgb_chw, w, h, patches, pos_emb, rcos, rsin);
  if (vc->mod) {
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
  smart_resize(iw, ih, vc->patch_size * vc->merge_size, min_tok, max_tok, &ow, &oh);
  float *chw = resize_and_norm(rgb, iw, ih, ow, oh, vc->mean, vc->std);
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
  fprintf(stderr, "vision: encoded %ux%u -> %ux%u (%ux%u grid, %u tokens) in %.0f ms\n",
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
