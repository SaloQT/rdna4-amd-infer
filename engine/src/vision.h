/* vision.h -- Qwen3.5/3.8 ViT vision projector (--mmproj) for amd-infer.
 * Loads an mmproj GGUF (Q8_0/F16/F32), decodes data:image/...;base64,... URLs
 * via Windows WIC, resizes to Qwen3.5's multiple-of-32 grid, runs the 27-layer
 * ViT + 2x2 spatial merger on the GPU with zero persistent VRAM overhead, and
 * caches recent encoded images in host RAM. */
#ifndef IE_VISION_H
#define IE_VISION_H

#include <stddef.h>
#include <stdint.h>

typedef struct vision_ctx vision_ctx;

typedef struct {
  float *embd;     /* [n_tok * proj_dim] f32 embeddings (owned by caller if !cached, or borrowed via vision_emb_free) */
  uint32_t n_tok;  /* nx * ny merged tokens */
  uint32_t nx, ny; /* merged spatial grid (width, height) for 3D IMRoPE */
  uint32_t proj_dim;
  uint64_t hash;   /* content hash of image + grid (for prompt cache matching) */
  int owned;       /* 1: free(embd) in vision_emb_free; 0: borrowed from cache */
} vision_emb;

/* Open an mmproj GGUF file. hsaco_hint (e.g. build/ie_kernels.hsaco, or NULL) is used
 * to locate sibling ie_vision_kernels.hsaco when GPU backend is active. Exits on fatal error. */
vision_ctx *vision_open(const char *mmproj_path, const char *hsaco_hint, uint32_t expected_proj_dim);
void vision_close(vision_ctx *vc);

/* Validate an image URL (data:image/...;base64,... or file://...) on the connection thread
 * without touching the GPU. Returns 1 if valid, 0 with message in err otherwise. */
int vision_validate_url(const char *url, size_t len, char *err, size_t errn);

/* Decode, smart-resize, normalize, and encode an image URL into [n_tok * proj_dim] f32 embeddings.
 * Uses LRU cache (8 entries) and temporary GPU workspace (freed immediately after encode).
 * Must be called from the serialized engine thread. Returns 1 on success, 0 with err on failure. */
int vision_encode_url(vision_ctx *vc, const char *url, size_t len, vision_emb *out, char *err, size_t errn);

/* Encode an already-normalized [3, h, w] f32 image (w, h multiples of patch_size * merge_size).
 * Used directly by vision_test to verify exact parity against llama-mtmd-debug. */
int vision_encode_f32(vision_ctx *vc, const float *rgb_chw, uint32_t w, uint32_t h, vision_emb *out);

void vision_emb_free(vision_emb *e);

#endif
