/* dcache_io_bench.c -- CPU-side file I/O speeds for the disk prompt cache (src/gen.c, dc_*), no GPU.
 *   dcache_io_bench DIR [MB]
 * Writes and reads a temporary file of MB megabytes (default 1536) in DIR and deletes it. Windows: buffered stdio
 * against unbuffered (FILE_FLAG_NO_BUFFERING) overlapped reads of several chunk sizes and queue depths, on a file
 * that is cold (never in the system cache) and warm (just written or read through the cache). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

static double now_s(void) {
#ifdef _WIN32
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f), QueryPerformanceCounter(&c);
  return (double)c.QuadPart / (double)f.QuadPart;
#else
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
#endif
}
#define MB (1024.0 * 1024.0)

#ifdef _WIN32
static void *abuf(size_t n) { return _aligned_malloc(n, 4096); }
static HANDLE open_w(const char *p, int direct) {
  return CreateFileA(p, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | (direct ? FILE_FLAG_NO_BUFFERING : 0), NULL);
}
/* write n bytes of src in chunk-sized WriteFile calls; prealloc: set the file size first */
static double write_win(const char *p, const char *src, size_t n, size_t chunk, int direct, int prealloc) {
  const double t0 = now_s();
  HANDLE h = open_w(p, direct);
  if (h == INVALID_HANDLE_VALUE) return -1;
  if (prealloc) {
    LARGE_INTEGER li;
    li.QuadPart = (LONGLONG)n;
    SetFilePointerEx(h, li, NULL, FILE_BEGIN), SetEndOfFile(h);
    li.QuadPart = 0;
    SetFilePointerEx(h, li, NULL, FILE_BEGIN);
  }
  for (size_t o = 0; o < n; o += chunk) {
    DWORD w, c = (DWORD)(n - o < chunk ? n - o : chunk);
    if (!WriteFile(h, src + o, c, &w, NULL) || w != c) return -1;
  }
  CloseHandle(h);
  return now_s() - t0;
}
/* read the whole file into dst: chunk-sized overlapped reads, depth in flight */
static double read_ov(const char *p, char *dst, size_t n, size_t chunk, int depth, int direct) {
  const double t0 = now_s();
  HANDLE h = CreateFileA(p, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING,
                         FILE_FLAG_OVERLAPPED | (direct ? FILE_FLAG_NO_BUFFERING : 0), NULL);
  if (h == INVALID_HANDLE_VALUE) return -1;
  const size_t nc = (n + chunk - 1) / chunk;
  OVERLAPPED *ov = calloc(nc, sizeof *ov);
  for (size_t i = 0; i < nc; i++) ov[i].hEvent = CreateEventA(NULL, TRUE, FALSE, NULL);
  size_t next = 0, done = 0;
  int bad = 0;
  while (done < nc) {
    while (next < nc && next - done < (size_t)depth) {
      const size_t o = next * chunk, c = n - o < chunk ? n - o : chunk;
      ov[next].Offset = (DWORD)(o & 0xffffffffu), ov[next].OffsetHigh = (DWORD)(o >> 32);
      if (!ReadFile(h, dst + o, (DWORD)c, NULL, &ov[next]) && GetLastError() != ERROR_IO_PENDING) bad = 1;
      next++;
    }
    DWORD got = 0;
    if (!GetOverlappedResult(h, &ov[done], &got, TRUE)) bad = 1;
    done++;
  }
  for (size_t i = 0; i < nc; i++) CloseHandle(ov[i].hEvent);
  free(ov);
  CloseHandle(h);
  return bad ? -1 : now_s() - t0;
}
#endif

static double read_stdio(const char *p, char *dst, size_t n, size_t chunk) {
  const double t0 = now_s();
  FILE *fp = fopen(p, "rb");
  if (!fp) return -1;
  for (size_t o = 0; o < n; o += chunk) {
    const size_t c = n - o < chunk ? n - o : chunk;
    if (fread(dst + o, 1, c, fp) != c) return -1;
  }
  fclose(fp);
  return now_s() - t0;
}
static double write_stdio(const char *p, const char *src, size_t n, size_t chunk) {
  const double t0 = now_s();
  FILE *fp = fopen(p, "wb");
  if (!fp) return -1;
  for (size_t o = 0; o < n; o += chunk) {
    const size_t c = n - o < chunk ? n - o : chunk;
    if (fwrite(src + o, 1, c, fp) != c) return -1;
  }
  fclose(fp);
  return now_s() - t0;
}
static void line(const char *what, double s, size_t n) {
  if (s < 0) printf("  %-58s failed\n", what);
  else printf("  %-58s %7.0f ms  %6.2f GB/s\n", what, s * 1e3, n / s / 1e9);
  fflush(stdout);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: dcache_io_bench DIR [MB]\n");
    return 2;
  }
  const size_t n = (size_t)(argc > 2 ? atof(argv[2]) : 1536.0) * 1024 * 1024;
  char path[1024], what[200];
  snprintf(path, sizeof path, "%s/dcache_io_bench.tmp", argv[1]);
#ifdef _WIN32
  char *src = abuf(n), *dst = abuf(n);
#else
  char *src = malloc(n), *dst = malloc(n);
#endif
  if (!src || !dst) return 1;
  for (size_t i = 0; i < n; i += 8) *(uint64_t *)(src + i) = i * 0x9E3779B97F4A7C15ull;
  const double t0 = now_s();
  memcpy(dst, src, n);
  line("memcpy (RAM to RAM)", now_s() - t0, n);
  printf("file %.0f MB in %s\n", n / MB, argv[1]);
  const size_t chunks[] = {1u << 20, 2u << 20, 4u << 20, 8u << 20, 16u << 20};
  const int depths[] = {1, 2, 4, 8};
#ifdef _WIN32
  printf("write\n");
  line("stdio fwrite, 8 MB chunks (buffered)", write_stdio(path, src, n, 8u << 20), n);
  line("WriteFile unbuffered, 8 MB chunks", write_win(path, src, n, 8u << 20, 1, 0), n);
  line("WriteFile unbuffered, 8 MB chunks, size preset", write_win(path, src, n, 8u << 20, 1, 1), n);
  line("WriteFile unbuffered, 32 MB chunks", write_win(path, src, n, 32u << 20, 1, 0), n);
  line("WriteFile buffered, 8 MB chunks", write_win(path, src, n, 8u << 20, 0, 0), n);
  printf("read, file just written through the cache (warm, dirty pages)\n");
  line("stdio fread 8 MB (buffered)", read_stdio(path, dst, n, 8u << 20), n);
  line("overlapped unbuffered 8 MB x 4", read_ov(path, dst, n, 8u << 20, 4, 1), n);
  printf("read, file written unbuffered (cold: not in the system cache)\n");
  write_win(path, src, n, 8u << 20, 1, 0);
  for (size_t ci = 0; ci < sizeof chunks / sizeof *chunks; ci++)
    for (size_t di = 0; di < sizeof depths / sizeof *depths; di++) {
      snprintf(what, sizeof what, "overlapped unbuffered %zu MB x %d in flight", chunks[ci] >> 20, depths[di]);
      line(what, read_ov(path, dst, n, chunks[ci], depths[di], 1), n);
    }
  line("overlapped buffered 8 MB x 4 (also warms the cache)", read_ov(path, dst, n, 8u << 20, 4, 0), n);
  printf("read, file now in the system cache (warm)\n");
  line("stdio fread 8 MB (buffered)", read_stdio(path, dst, n, 8u << 20), n);
  line("overlapped buffered 8 MB x 4", read_ov(path, dst, n, 8u << 20, 4, 0), n);
  line("overlapped unbuffered 8 MB x 4", read_ov(path, dst, n, 8u << 20, 4, 1), n);
  printf("read, cold, stdio\n");
  write_win(path, src, n, 8u << 20, 1, 0);
  line("stdio fread 8 MB (buffered), cold", read_stdio(path, dst, n, 8u << 20), n);
  write_win(path, src, n, 8u << 20, 1, 0);
  line("stdio fread 1 MB (buffered), cold", read_stdio(path, dst, n, 1u << 20), n);
#else
  line("stdio fwrite", write_stdio(path, src, n, 8u << 20), n);
  line("stdio fread", read_stdio(path, dst, n, 8u << 20), n);
  (void)chunks, (void)depths, (void)what;
#endif
  remove(path);
  return 0;
}
