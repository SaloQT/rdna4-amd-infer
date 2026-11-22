/* Byte-plane split / merge speed in C (the transform a lossless disk-cache codec would need around zstd/Huffman).
 * Reads the state of one checkpoint file (offset and size from argv), times split and merge, single thread and N threads.
 *   clang -O3 -march=native -o dcache_compress_shuffle_bench.exe dcache_compress_shuffle_bench.c
 *   dcache_compress_shuffle_bench.exe FILE OFFSET BYTES
 * Read-only on FILE; no GPU. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

static double now(void) {
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f), QueryPerformanceCounter(&c);
  return (double)c.QuadPart / (double)f.QuadPart;
}

/* 4 planes (each n bytes, low byte first) from n u32 */
static void split4(const uint32_t *restrict s, uint8_t *restrict p0, uint8_t *restrict p1, uint8_t *restrict p2, uint8_t *restrict p3, size_t n) {
  for (size_t i = 0; i < n; i++) {
    const uint32_t v = s[i];
    p0[i] = (uint8_t)v, p1[i] = (uint8_t)(v >> 8), p2[i] = (uint8_t)(v >> 16), p3[i] = (uint8_t)(v >> 24);
  }
}
static void merge4(uint32_t *restrict d, const uint8_t *restrict p0, const uint8_t *restrict p1, const uint8_t *restrict p2, const uint8_t *restrict p3, size_t n) {
  for (size_t i = 0; i < n; i++)
    d[i] = (uint32_t)p0[i] | (uint32_t)p1[i] << 8 | (uint32_t)p2[i] << 16 | (uint32_t)p3[i] << 24;
}
typedef struct {
  uint32_t *s, *d;
  uint8_t *p;
  size_t n;
  int merge;
} job;
static DWORD WINAPI run(LPVOID a) {
  job *j = a;
  if (j->merge) merge4(j->d, j->p, j->p + j->n, j->p + 2 * j->n, j->p + 3 * j->n, j->n);
  else split4(j->s, j->p, j->p + j->n, j->p + 2 * j->n, j->p + 3 * j->n, j->n);
  return 0;
}
static double bench(uint32_t *s, uint32_t *d, uint8_t *p, size_t n, int threads, int merge) {
  double best = 1e9;
  for (int rep = 0; rep < 5; rep++) {
    HANDLE h[64];
    job jb[64];
    const size_t per = n / threads;
    double t = now();
    for (int k = 0; k < threads; k++) {
      /* each thread works on its own slice with its own 4 planes (slice-local layout, as one frame per slice) */
      jb[k] = (job){s + k * per, d + k * per, p + (size_t)k * per * 4, per, merge};
      h[k] = CreateThread(NULL, 0, run, &jb[k], 0, NULL);
    }
    WaitForMultipleObjects(threads, h, TRUE, INFINITE);
    t = now() - t;
    for (int k = 0; k < threads; k++) CloseHandle(h[k]);
    if (t < best) best = t;
  }
  return best;
}
int main(int argc, char **argv) {
  if (argc < 4) return 1;
  FILE *f = fopen(argv[1], "rb");
  const long long off = atoll(argv[2]);
  const size_t bytes = (size_t)atoll(argv[3]), n = bytes / 4;
  uint32_t *s = malloc(bytes), *d = malloc(bytes);
  uint8_t *p = malloc(bytes);
  _fseeki64(f, off, SEEK_SET);
  if (!f || fread(s, 1, bytes, f) != bytes) return 2;
  fclose(f);
  memset(p, 1, bytes), memset(d, 1, bytes);
  printf("state %.1f MB\n", bytes / 1e6);
  for (int th = 1; th <= 8; th *= 2) {
    const double ts = bench(s, d, p, n, th, 0), tm = bench(s, d, p, n, th, 1);
    printf("threads %d: split %6.0f MB/s (%.1f ms)  merge %6.0f MB/s (%.1f ms)\n", th, bytes / ts / 1e6, ts * 1e3, bytes / tm / 1e6, tm * 1e3);
  }
  /* check: merge(split(x)) == x on the last configuration */
  bench(s, d, p, n, 8, 0), bench(s, d, p, n, 8, 1);
  printf("roundtrip %s\n", memcmp(s, d, bytes) ? "FAIL" : "ok");
  /* memcpy for scale */
  double best = 1e9;
  for (int r = 0; r < 5; r++) {
    double t = now();
    memcpy(d, s, bytes);
    t = now() - t;
    if (t < best) best = t;
  }
  printf("memcpy single thread %.0f MB/s\n", bytes / best / 1e6);
  return 0;
}
