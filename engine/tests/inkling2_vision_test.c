/* inkling2_vision_test.c -- CPU vs GPU WMMA cosine parity, speed, and 0-persistent-VRAM verifier for inkling2v. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/backend.h"
#include "../src/gen.h"
#include "../src/vision.h"

static float *make_synth_chw(uint32_t w, uint32_t h) {
  const size_t plane = (size_t)w * h;
  float *buf = malloc(3u * plane * 4u);
  if (!buf) { fprintf(stderr, "OOM\n"); exit(1); }
  for (uint32_t y = 0; y < h; y++) {
    for (uint32_t x = 0; x < w; x++) {
      const size_t i = (size_t)y * w + x;
      const float fx = (float)x / (float)w;
      const float fy = (float)y / (float)h;
      const float checker = (((x / 20u) + (y / 20u)) & 1u) ? 0.65f : -0.65f;
      buf[0 * plane + i] = fx * 1.8f - 0.9f + 0.1f * checker;
      buf[1 * plane + i] = fy * 1.8f - 0.9f - 0.1f * checker;
      buf[2 * plane + i] = sinf(fx * 6.283185f) * cosf(fy * 6.283185f);
    }
  }
  return buf;
}

static void compare_embs(const vision_emb *a, const vision_emb *b,
                         double *cos_out, double *max_diff_out, double *rms_out) {
  if (a->n_tok != b->n_tok || a->proj_dim != b->proj_dim) {
    *cos_out = -1.0; *max_diff_out = 1e9; *rms_out = 1e9;
    return;
  }
  const size_t n = (size_t)a->n_tok * a->proj_dim;
  double dot = 0.0, na = 0.0, nb = 0.0, md = 0.0, s2 = 0.0;
  for (size_t i = 0; i < n; i++) {
    const double va = a->embd[i], vb = b->embd[i];
    const double d = fabs(va - vb);
    if (d > md) md = d;
    s2 += d * d;
    dot += va * vb;
    na += va * va;
    nb += vb * vb;
  }
  *cos_out = (na > 0.0 && nb > 0.0) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
  *max_diff_out = md;
  *rms_out = sqrt(s2 / (double)n);
}

static int run_parity_case(vision_ctx *v_cpu, vision_ctx *v_gpu, const char *tag, uint32_t w, uint32_t h) {
  float *img = make_synth_chw(w, h);
  vision_emb ec = {0}, eg = {0};
  double t0 = now_ms();
  int okc = vision_encode_f32(v_cpu, img, w, h, &ec);
  double ms_cpu = now_ms() - t0;
  t0 = now_ms();
  int okg = vision_encode_f32(v_gpu, img, w, h, &eg);
  double ms_gpu = now_ms() - t0;
  free(img);
  if (!okc || !okg) {
    fprintf(stderr, "FAIL %s (%ux%u): encode returned 0 (cpu=%d gpu=%d)\n", tag, w, h, okc, okg);
    vision_emb_free(&ec);
    vision_emb_free(&eg);
    return 1;
  }
  double cs = 0.0, md = 0.0, rms = 0.0;
  compare_embs(&eg, &ec, &cs, &md, &rms);
  const int pass = (cs > 0.9999 && !isnan(cs));
  printf("PARITY %-12s %4ux%-4u n_tok=%4u (%2ux%-2u) cos=%.7f max_diff=%.6g rms=%.6g cpu_ms=%.1f gpu_ms=%.2f %s\n",
         tag, w, h, eg.n_tok, eg.nx, eg.ny, cs, md, rms, ms_cpu, ms_gpu, pass ? "PASS" : "FAIL");
  fflush(stdout);
  vision_emb_free(&ec);
  vision_emb_free(&eg);
  return pass ? 0 : 1;
}

static void run_speed_case(vision_ctx *v_gpu, const char *tag, uint32_t w, uint32_t h, int reps) {
  float *img = make_synth_chw(w, h);
  vision_emb e = {0};
  double t0 = now_ms();
  if (!vision_encode_f32(v_gpu, img, w, h, &e)) {
    fprintf(stderr, "FAIL speed %s (%ux%u)\n", tag, w, h);
    exit(1);
  }
  double cold_ms = now_ms() - t0;
  uint32_t n_tok = e.n_tok, nx = e.nx, ny = e.ny;
  vision_emb_free(&e);

  double warm_min = 1e9, warm_sum = 0.0, warm_ms[16];
  if (reps > 16) reps = 16;
  for (int r = 0; r < reps; r++) {
    t0 = now_ms();
    if (!vision_encode_f32(v_gpu, img, w, h, &e)) {
      fprintf(stderr, "FAIL speed warm %s\n", tag);
      exit(1);
    }
    double dt = now_ms() - t0;
    warm_ms[r] = dt;
    if (dt < warm_min) warm_min = dt;
    warm_sum += dt;
    vision_emb_free(&e);
  }
  /* Simple sort for median */
  for (int i = 0; i < reps; i++)
    for (int j = i + 1; j < reps; j++)
      if (warm_ms[j] < warm_ms[i]) { double tmp = warm_ms[i]; warm_ms[i] = warm_ms[j]; warm_ms[j] = tmp; }
  double med = warm_ms[reps / 2];
  printf("SPEED  %-12s %4ux%-4u n_tok=%4u (%2ux%-2u) cold_ms=%.2f warm_med_ms=%.2f warm_min_ms=%.2f warm_avg_ms=%.2f\n",
         tag, w, h, n_tok, nx, ny, cold_ms, med, warm_min, warm_sum / (double)reps);
  fflush(stdout);
  free(img);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <mmproj.gguf> [hsaco_hint] [--speed-only] [--png PATH]\n", argv[0]);
    return 2;
  }
  const char *mmproj = argv[1];
  const char *hsaco = (argc > 2 && argv[2][0] != '-') ? argv[2] : "build/ie_kernels.hsaco";
  int speed_only = 0;
  const char *png_path = NULL;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--speed-only")) speed_only = 1;
    else if (!strcmp(argv[i], "--png") && i + 1 < argc) png_path = argv[++i];
  }

  /* CPU ctx opened BEFORE GPU init so gpu_vis_ready() == 0 */
  vision_ctx *v_cpu = speed_only ? NULL : vision_open(mmproj, NULL, 2304);

  /* Initialize GPU & open GPU ctx */
  gpu_stream_init();
  size_t vram_free_0 = 0, vram_tot = 0;
  uint64_t dx_u0 = 0, dx_b0 = 0, dx_u_open = 0, dx_u_warm = 0, dx_u_end = 0, dx_u_close = 0;
  gpu_vis_mem_info(&vram_free_0, &vram_tot);
  gpu_vis_dxgi_info(&dx_u0, &dx_b0);
  vision_ctx *v_gpu = vision_open(mmproj, hsaco, 2304);
  size_t vram_free_open = 0;
  gpu_vis_mem_info(&vram_free_open, &vram_tot);
  gpu_vis_dxgi_info(&dx_u_open, &dx_b0);
  printf("VRAM_OPEN  dxgi_before=%.2f MiB dxgi_open=%.2f MiB dxgi_delta=%.2f MiB budget=%.2f GiB hip_free=%.2f MiB\n",
         dx_u0 / 1048576.0, dx_u_open / 1048576.0, ((double)dx_u_open - (double)dx_u0) / 1048576.0,
         dx_b0 / 1073741824.0, vram_free_open / 1048576.0);
  fflush(stdout);

  int fail = 0;
  if (!speed_only) {
    fail |= run_parity_case(v_cpu, v_gpu, "sq_640", 640, 640);
    gpu_vis_dxgi_info(&dx_u_warm, &dx_b0);
    fail |= run_parity_case(v_cpu, v_gpu, "sq_1040", 1040, 1040);
    fail |= run_parity_case(v_cpu, v_gpu, "rect_1280x720", 1280, 720);
    if (png_path) {
      char url[1100], err[256] = "";
      snprintf(url, sizeof url, "file:///%s", png_path);
      vision_emb ec = {0}, eg = {0};
      int okc = vision_encode_url(v_cpu, url, strlen(url), &ec, err, sizeof err);
      int okg = vision_encode_url(v_gpu, url, strlen(url), &eg, err, sizeof err);
      if (!okc || !okg) {
        fprintf(stderr, "FAIL png_url %s: %s\n", png_path, err);
        fail = 1;
      } else {
        double cs = 0.0, md = 0.0, rms = 0.0;
        compare_embs(&eg, &ec, &cs, &md, &rms);
        int pass = (cs > 0.9999 && !isnan(cs));
        printf("PARITY %-12s png n_tok=%4u (%2ux%-2u) cos=%.7f max_diff=%.6g rms=%.6g %s\n",
               "png_url", eg.n_tok, eg.nx, eg.ny, cs, md, rms, pass ? "PASS" : "FAIL");
        if (!pass) fail = 1;
      }
      vision_emb_free(&ec);
      vision_emb_free(&eg);
    }
    vision_close(v_cpu);
  }

  /* Speed suite (cold + 5 warm reps) */
  run_speed_case(v_gpu, "640x640", 640, 640, 5);
  run_speed_case(v_gpu, "1040x1040", 1040, 1040, 5);
  run_speed_case(v_gpu, "1280x1280", 1280, 1280, 5);
  gpu_vis_dxgi_info(&dx_u_warm, &dx_b0);
  /* Repeat 1280x1280 3 more times to prove 0.00 MiB VRAM leak across repeated encodes */
  run_speed_case(v_gpu, "1280_repeat", 1280, 1280, 3);

  size_t vram_free_end = 0;
  gpu_vis_mem_info(&vram_free_end, &vram_tot);
  gpu_vis_dxgi_info(&dx_u_end, &dx_b0);
  const double dx_open_delta = ((double)dx_u_end - (double)dx_u_open) / 1048576.0;
  const double dx_repeat_delta = ((double)dx_u_end - (double)dx_u_warm) / 1048576.0;
  const double hip_delta = ((double)vram_free_open - (double)vram_free_end) / 1048576.0;
  printf("VRAM_END   dxgi_open=%.2f MiB dxgi_peak_pool=%.2f MiB dxgi_end=%.2f MiB repeat_delta=%.2f MiB hip_delta=%.2f MiB\n",
         dx_u_open / 1048576.0, dx_u_warm / 1048576.0, dx_u_end / 1048576.0, dx_repeat_delta, hip_delta);
  vision_close(v_gpu);
  gpu_vis_dxgi_info(&dx_u_close, &dx_b0);
  printf("VRAM_CLOSE dxgi_close=%.2f MiB delta_vs_init=%.2f MiB (open_delta=%.2f MiB)\n",
         dx_u_close / 1048576.0, ((double)dx_u_close - (double)dx_u0) / 1048576.0, dx_open_delta);
  if (fabs(dx_repeat_delta) > 1.0) {
    fprintf(stderr, "FAIL persistent VRAM leak across repeated encodes: %.2f MiB > 1.0 MiB!\n", dx_repeat_delta);
    fail = 1;
  }
  printf(fail ? "inkling2_vision_test: FAILED\n" : "inkling2_vision_test: ALL PASSED\n");
  return fail;
}
