/* kbench.c -- replay one captured kernel launch (IE_KCAP, src/hip.c) from one or two code objects: check the outputs
 * and time the kernel alone, with the real arguments and the real input bytes of the engine run it came from.
 *   kbench CAPTURE REF.hsaco [CAND.hsaco] [-n LAUNCHES] [-r ROUNDS] [-k CAND_KERNEL_NAME] [--cold MB]
 * 1. REF is launched once on the captured inputs and every allocation is hashed: "replay matches engine" says the
 *    replay reproduced the bytes the engine produced (it checks the capture, not the kernel).
 * 2. CAND (if given) is launched once on the same inputs and its outputs are compared with REF's byte for byte.
 * 3. Timing: ROUNDS rounds (default 7), REF and CAND alternating (the order flips every round); each block restores
 *    the inputs, runs 3 warm-up launches and then LAUNCHES launches (default 50) back to back between two events.
 *    It prints the median and the minimum microseconds per launch. Back-to-back launches find the weights in the
 *    64 MB cache when they fit; --cold MB clears it before every launch with a memset of MB megabytes, and the time of
 *    the memsets alone (measured the same way) is subtracted.
 * Build: gcc -std=c11 -O2 -o build/kbench.exe tools/kbench.c (Windows, the ROCm wheel bin directory on PATH),
 *        cc -std=c11 -O2 -o build/kbench tools/kbench.c -ldl (Linux). */
#ifdef _WIN32
#include "../src/win_compat.h"
#else
#include <dlfcn.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int hipError_t;
typedef void *hipModule_t, *hipFunction_t, *hipStream_t, *hipEvent_t;
enum { H2D = 1, D2H = 2, MAX_ARGS = 64, MAX_REG = 64, SLACK = 1u << 20 };

static struct {
  hipError_t (*SetDevice)(int);
  hipError_t (*Malloc)(void **, size_t);
  hipError_t (*Memcpy)(void *, const void *, size_t, int);
  hipError_t (*Memset)(void *, int, size_t);
  hipError_t (*MemsetAsync)(void *, int, size_t, hipStream_t);
  hipError_t (*ModuleLoadData)(hipModule_t *, const void *);
  hipError_t (*ModuleGetFunction)(hipFunction_t *, hipModule_t, const char *);
  hipError_t (*ModuleLaunchKernel)(hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                   hipStream_t, void **, void **);
  hipError_t (*DeviceSynchronize)(void);
  hipError_t (*EventCreate)(hipEvent_t *);
  hipError_t (*EventRecord)(hipEvent_t, hipStream_t);
  hipError_t (*EventSynchronize)(hipEvent_t);
  hipError_t (*EventElapsedTime)(float *, hipEvent_t, hipEvent_t);
  const char *(*GetErrorString)(hipError_t);
} H;

#define CK(x)                                                                                              \
  do {                                                                                                     \
    hipError_t e_ = (x);                                                                                   \
    if (e_) {                                                                                              \
      fprintf(stderr, "HIP error %d (%s) at line %d: %s\n", e_, H.GetErrorString(e_), __LINE__, #x);      \
      exit(1);                                                                                             \
    }                                                                                                      \
  } while (0)

static void die(const char *m) {
  fprintf(stderr, "kbench: %s\n", m);
  exit(1);
}

/* the capture */
static char kname[256];
static unsigned geo[7], nargs, nreg;
static unsigned asz[MAX_ARGS], areg[MAX_ARGS];
static uint64_t aoff[MAX_ARGS], aval[MAX_ARGS], want_hash[MAX_REG];
static uint8_t *init[MAX_REG]; /* input bytes */
static size_t rlen[MAX_REG];
static uint8_t *dev[MAX_REG];

static void rd(FILE *f, void *p, size_t n) {
  if (fread(p, 1, n, f) != n) die("capture file is truncated");
}
static void load_capture(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot open the capture");
  char mg[8];
  unsigned nl;
  rd(f, mg, 8);
  if (memcmp(mg, "IEKCAP01", 8)) die("not an IE_KCAP file");
  rd(f, &nl, 4);
  if (nl >= sizeof kname) die("kernel name too long");
  rd(f, kname, nl);
  rd(f, geo, sizeof geo);
  rd(f, &nargs, 4);
  if (nargs > MAX_ARGS) die("too many arguments");
  for (unsigned i = 0; i < nargs; i++) {
    rd(f, &asz[i], 4);
    rd(f, &areg[i], 4);
    rd(f, &aoff[i], 8);
    rd(f, &aval[i], 8);
  }
  rd(f, &nreg, 4);
  if (nreg > MAX_REG) die("too many allocations");
  for (unsigned r = 0; r < nreg; r++) {
    size_t cap = 0, n = 0;
    uint8_t *p = NULL;
    for (;;) {
      unsigned pl;
      rd(f, &pl, 4);
      if (!pl) break;
      if (n + pl > cap) {
        cap = (n + pl) * 2;
        if (!(p = realloc(p, cap))) die("out of memory");
      }
      rd(f, p + n, pl);
      n += pl;
    }
    init[r] = p, rlen[r] = n;
  }
  for (unsigned r = 0; r < nreg; r++) rd(f, &want_hash[r], 8);
  fclose(f);
}

static uint64_t fnv(uint64_t h, const uint8_t *p, size_t n) {
  for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}
static void restore(void) {
  for (unsigned r = 0; r < nreg; r++)
    if (rlen[r]) CK(H.Memcpy(dev[r], init[r], rlen[r], H2D));
  CK(H.DeviceSynchronize());
}
static void *slots[MAX_ARGS];
static uint64_t vals[MAX_ARGS];
static void launch(hipFunction_t fn) {
  CK(H.ModuleLaunchKernel(fn, geo[0], geo[1], geo[2], geo[3], geo[4], geo[5], geo[6], NULL, slots, NULL));
}
static hipFunction_t load_fn(const char *path, const char *name) {
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot open a code object");
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *img = malloc((size_t)sz);
  if (!img) die("out of memory");
  rd(f, img, (size_t)sz);
  fclose(f);
  hipModule_t mod;
  hipFunction_t fn;
  CK(H.ModuleLoadData(&mod, img));
  if (H.ModuleGetFunction(&fn, mod, name)) {
    fprintf(stderr, "kbench: %s has no kernel %s\n", path, name);
    exit(1);
  }
  return fn;
}
static uint8_t **download(void) {
  uint8_t **o = calloc(nreg, sizeof *o);
  for (unsigned r = 0; r < nreg; r++) {
    if (!(o[r] = malloc(rlen[r] ? rlen[r] : 1))) die("out of memory");
    if (rlen[r]) CK(H.Memcpy(o[r], dev[r], rlen[r], D2H));
  }
  return o;
}

static hipEvent_t e0, e1;
static void *flush;
static size_t flush_bytes;
/* microseconds per launch of a block of n launches (fn NULL: the cache-clearing memsets alone) */
static double block(hipFunction_t fn, int n) {
  restore();
  for (int i = 0; i < 3; i++) {
    if (flush) CK(H.MemsetAsync(flush, i, flush_bytes, NULL));
    if (fn) launch(fn);
  }
  CK(H.EventRecord(e0, NULL));
  for (int i = 0; i < n; i++) {
    if (flush) CK(H.MemsetAsync(flush, i & 255, flush_bytes, NULL));
    if (fn) launch(fn);
  }
  CK(H.EventRecord(e1, NULL));
  CK(H.EventSynchronize(e1));
  float ms = 0;
  CK(H.EventElapsedTime(&ms, e0, e1));
  return ms * 1000.0 / n;
}
static int cmpd(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

int main(int argc, char **argv) {
  const char *cap = NULL, *refp = NULL, *candp = NULL, *cname = NULL;
  int n = 50, rounds = 7;
  double cold_mb = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-r") && i + 1 < argc) rounds = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-k") && i + 1 < argc) cname = argv[++i];
    else if (!strcmp(argv[i], "--cold") && i + 1 < argc) cold_mb = atof(argv[++i]);
    else if (!cap) cap = argv[i];
    else if (!refp) refp = argv[i];
    else if (!candp) candp = argv[i];
    else die("too many arguments");
  }
  if (!cap || !refp || n < 1 || n > 10000 || rounds < 1 || rounds > 100 || cold_mb < 0 || cold_mb > 1024) {
    fprintf(stderr, "usage: kbench CAPTURE REF.hsaco [CAND.hsaco] [-n 1..10000] [-r 1..100] [-k CAND_KERNEL] "
                    "[--cold MB<=1024]\n");
    return 2;
  }
  load_capture(cap);
  if (!cname) cname = kname;

#ifdef _WIN32
  const char *libs[] = {"amdhip64_7.dll", "amdhip64.dll", "amdhip64_6.dll"};
#else
  const char *libs[] = {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6", "/opt/rocm/lib/libamdhip64.so"};
#endif
  void *lib = NULL;
  for (unsigned i = 0; i < sizeof libs / sizeof libs[0] && !lib; i++) lib = dlopen(libs[i], RTLD_NOW);
  if (!lib) die("cannot load the HIP runtime (put the ROCm bin directory on PATH)");
#define SYM(f, s)                                     \
  do {                                                \
    *(void **)&H.f = dlsym(lib, s);                   \
    if (!H.f) die("HIP runtime is missing " s);       \
  } while (0)
  SYM(SetDevice, "hipSetDevice");
  SYM(Malloc, "hipMalloc");
  SYM(Memcpy, "hipMemcpy");
  SYM(Memset, "hipMemset");
  SYM(MemsetAsync, "hipMemsetAsync");
  SYM(ModuleLoadData, "hipModuleLoadData");
  SYM(ModuleGetFunction, "hipModuleGetFunction");
  SYM(ModuleLaunchKernel, "hipModuleLaunchKernel");
  SYM(DeviceSynchronize, "hipDeviceSynchronize");
  SYM(EventCreate, "hipEventCreate");
  SYM(EventRecord, "hipEventRecord");
  SYM(EventSynchronize, "hipEventSynchronize");
  SYM(EventElapsedTime, "hipEventElapsedTime");
  SYM(GetErrorString, "hipGetErrorString");
#undef SYM
  CK(H.SetDevice(0));

  double mb = 0;
  for (unsigned r = 0; r < nreg; r++) {
    void *p;
    CK(H.Malloc(&p, rlen[r] + SLACK)); /* zeroed slack: a kernel may read a tile past the captured end */
    CK(H.Memset(p, 0, rlen[r] + SLACK));
    dev[r] = p;
    mb += rlen[r] / 1048576.0;
  }
  for (unsigned i = 0; i < nargs; i++) {
    vals[i] = areg[i] == ~0u ? aval[i] : (uint64_t)(uintptr_t)(dev[areg[i]] + aoff[i]);
    slots[i] = &vals[i];
  }
  printf("kernel %s: grid %u x %u x %u, block %u, LDS %u, %u arguments, %u allocations (%.1f MB)\n", kname, geo[0],
         geo[1], geo[2], geo[3], geo[6], nargs, nreg, mb);

  hipFunction_t fr = load_fn(refp, kname), fc = candp ? load_fn(candp, cname) : NULL;
  restore();
  launch(fr);
  CK(H.DeviceSynchronize());
  uint8_t **out_ref = download();
  int match = 1;
  for (unsigned r = 0; r < nreg; r++)
    if (fnv(1469598103934665603ull, out_ref[r], rlen[r]) != want_hash[r]) {
      printf("  allocation %u differs from the engine's output\n", r);
      match = 0;
    }
  printf("replay matches engine: %s\n", match ? "yes" : "NO (the capture or replay is not faithful)");

  if (fc) {
    restore();
    launch(fc);
    CK(H.DeviceSynchronize());
    uint8_t **out_c = download();
    int same = 1;
    for (unsigned r = 0; r < nreg; r++) {
      size_t nd = 0, first = 0;
      for (size_t i = 0; i < rlen[r]; i++)
        if (out_ref[r][i] != out_c[r][i] && !nd++) first = i;
      if (!nd) continue;
      same = 0;
      size_t w = first & ~(size_t)3;
      float a = 0, c = 0;
      if (w + 4 <= rlen[r]) memcpy(&a, out_ref[r] + w, 4), memcpy(&c, out_c[r] + w, 4);
      printf("  allocation %u: %zu bytes differ, first at byte %zu (as f32: ref %.9g cand %.9g)\n", r, nd, first, a,
             c);
    }
    printf("candidate output: %s\n", same ? "bit-identical to REF" : "DIFFERS from REF");
  }

  CK(H.EventCreate(&e0));
  CK(H.EventCreate(&e1));
  double base = 0;
  if (cold_mb > 0) {
    flush_bytes = (size_t)(cold_mb * 1048576.0);
    CK(H.Malloc(&flush, flush_bytes));
    double t[100];
    for (int i = 0; i < rounds; i++) t[i] = block(NULL, n);
    qsort(t, rounds, sizeof t[0], cmpd);
    base = t[rounds / 2];
    printf("cache clear: %.0f MB memset per launch, %.2f us each (subtracted)\n", cold_mb, base);
  }
  double tr[100], tc[100];
  for (int i = 0; i < rounds; i++) {
    if (fc && (i & 1)) tc[i] = block(fc, n) - base, tr[i] = block(fr, n) - base;
    else {
      tr[i] = block(fr, n) - base;
      if (fc) tc[i] = block(fc, n) - base;
    }
  }
  qsort(tr, rounds, sizeof tr[0], cmpd);
  printf("ref : median %8.2f us  min %8.2f us  (%d rounds x %d launches)\n", tr[rounds / 2], tr[0], rounds, n);
  if (fc) {
    qsort(tc, rounds, sizeof tc[0], cmpd);
    printf("cand: median %8.2f us  min %8.2f us  (%+.1f%% median)\n", tc[rounds / 2], tc[0],
           100.0 * (tc[rounds / 2] - tr[rounds / 2]) / tr[rounds / 2]);
  }
  return 0;
}
