/* vision_test.c -- verify GPU ViT vision projector against CPU and llama-mtmd-debug oracle. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/backend.h"
#include "../src/gen.h"
#include "../src/vision.h"

int main(int argc, char **argv) {
  const char *mmproj = argc > 1 ? argv[1] : "../../models/mmproj-Underdog-Saluki-27B-1.0-Q8_0.gguf";
  const char *hsaco = argc > 2 ? argv[2] : "build/ie_kernels.hsaco";

  /* 1. Build exact 64x64 checkerboard matching llama-mtmd-debug -p encode -n 64 --image cb */
  const uint32_t W = 64, H = 64;
  float *cb = calloc((size_t)3u * W * H, 4u);
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++) {
      const float v = ((x + y) % 2 == 0) ? 1.0f : 0.0f;
      cb[0 * W * H + y * W + x] = v;
      cb[1 * W * H + y * W + x] = v;
      cb[2 * W * H + y * W + x] = v;
    }

  /* Run CPU first (before GPU init) */
  vision_ctx *v_cpu = vision_open(mmproj, NULL, 5120);
  vision_emb e_cpu = {0};
  double t0 = now_ms();
  if (!vision_encode_f32(v_cpu, cb, W, H, &e_cpu)) {
    fprintf(stderr, "FAIL cpu vision_encode_f32\n");
    return 1;
  }
  printf("CPU cb64: n_tok=%u (%ux%u) in %.1f ms\n", e_cpu.n_tok, e_cpu.nx, e_cpu.ny, now_ms() - t0);
  vision_close(v_cpu);

  /* Now init GPU and run GPU ViT */
  gpu_stream_init();
  vision_ctx *v_gpu = vision_open(mmproj, hsaco, 5120);
  vision_emb e_gpu = {0};
  t0 = now_ms();
  if (!vision_encode_f32(v_gpu, cb, W, H, &e_gpu)) {
    fprintf(stderr, "FAIL gpu vision_encode_f32\n");
    return 1;
  }
  printf("GPU cb64: n_tok=%u (%ux%u) in %.1f ms\n", e_gpu.n_tok, e_gpu.nx, e_gpu.ny, now_ms() - t0);

  /* Compare GPU vs CPU element-by-element */
  double max_diff = 0.0, dot = 0.0, n1 = 0.0, n2 = 0.0, sum_g = 0.0;
  const size_t ntot = (size_t)e_gpu.n_tok * e_gpu.proj_dim;
  for (size_t i = 0; i < ntot; i++) {
    const double a = e_gpu.embd[i], b = e_cpu.embd[i];
    const double d = fabs(a - b);
    if (d > max_diff) max_diff = d;
    dot += a * b, n1 += a * a, n2 += b * b, sum_g += a;
  }
  const double cos_sim = dot / (sqrt(n1) * sqrt(n2));
  printf("GPU vs CPU: max_abs_diff=%.6g cos_sim=%.9f sum=%.6f\n", max_diff, cos_sim, sum_g);
  printf("GPU row0 [:3] = [%.4f, %.4f, %.4f]  [-3:] = [%.4f, %.4f, %.4f]\n",
         e_gpu.embd[0], e_gpu.embd[1], e_gpu.embd[2],
         e_gpu.embd[5117], e_gpu.embd[5118], e_gpu.embd[5119]);
  printf("REF row0 [:3] = [-0.2808, -0.6624, -0.3748]  [-3:] = [0.0420, 0.2329, -0.0807]\n");

  int fail = 0;
  if (max_diff > 1e-3 || cos_sim < 0.999999) {
    fprintf(stderr, "FAIL GPU vs CPU mismatch!\n");
    fail = 1;
  }
  if (fabsf(e_gpu.embd[0] - (-0.2859f)) > 0.01f || fabsf(e_gpu.embd[1] - (-0.6618f)) > 0.01f) {
    fprintf(stderr, "FAIL GPU vs mtmd-debug reference!\n");
    fail = 1;
  }

  /* 2. Time GPU on a full-HD 1344x768 screenshot (1008 merged image tokens = 4032 patches!) */
  const uint32_t W2 = 1344, H2 = 768;
  float *hd = calloc((size_t)3u * W2 * H2, 4u);
  for (size_t i = 0; i < (size_t)3u * W2 * H2; i++) hd[i] = ((float)(i % 97u) / 96.0f) * 2.0f - 1.0f;
  vision_emb e_hd = {0};
  t0 = now_ms();
  vision_encode_f32(v_gpu, hd, W2, H2, &e_hd);
  printf("GPU 1344x768 (%u merged tokens, 4032 patches): %.1f ms (row0[0]=%.4f)\n",
         e_hd.n_tok, now_ms() - t0, e_hd.embd[0]);

  vision_emb_free(&e_hd);
  vision_emb_free(&e_gpu);
  vision_emb_free(&e_cpu);
  vision_close(v_gpu);
  free(hd);
  free(cb);
  printf(fail ? "vision_test: FAILED\n" : "vision_test: ALL PASSED\n");
  return fail;
}
