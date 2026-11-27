/* kbench.c -- replay one captured kernel launch (IE_KCAP, src/hip.c) from one or two code objects: check the outputs
 * and time the kernel alone, with the real arguments and the real input bytes of the engine run it came from.
 *   kbench CAPTURE REF.hsaco [CAND.hsaco] [-n LAUNCHES] [-r ROUNDS] [-k CAND_KERNEL_NAME] [--cold COPIES]
 * 1. REF is launched once on the captured inputs and every allocation is hashed: "replay matches engine" says the
 *    replay reproduced the bytes the engine produced (it checks the capture, not the kernel).
 * 2. CAND (if given) is launched once on the same inputs and its outputs are compared with REF's byte for byte.
 * 3. Timing: ROUNDS rounds (default 7), REF and CAND alternating (the order flips every round); each block restores
 *    the inputs, runs 3 warm-up launches and then LAUNCHES launches (default 50) back to back between two events,
 *    and gives microseconds per launch. It prints the median and the minimum over the rounds. A block whose events
 *    give no valid time (WDDM does that now and then) is repeated; one under a quarter of the median is left out and
 *    counted.
 *    Back-to-back launches find their data in the 64 MB cache when it fits there, which the engine (where the other
 *    layers run in between) does not. --cold COPIES keeps that many copies of the captured memory and cycles the
 *    launches through them: with COPIES x (the bytes one launch touches) well above 64 MB, each launch reads from
 *    DRAM, as in the engine.
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
enum { H2D = 1, D2H = 2, MAX_ARGS = 64, MAX_REG = 64, MAX_COPY = 16, SLACK = 1u << 20 };

static struct {
  hipError_t (*SetDevice)(int);
  hipError_t (*Malloc)(void **, size_t);
  hipError_t (*Memcpy)(void *, const void *, size_t, int);
  hipError_t (*Memset)(void *, int, size_t);
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
/* the device copies */
static unsigned ncopy = 1;
static uint8_t *dev[MAX_COPY][MAX_REG];
static uint64_t vals[MAX_COPY][MAX_ARGS];
static void *slots[MAX_COPY][MAX_ARGS];

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
  for (unsigned c = 0; c < ncopy; c++)
    for (unsigned r = 0; r < nreg; r++)
      if (rlen[r]) CK(H.Memcpy(dev[c][r], init[r], rlen[r], H2D));
  CK(H.DeviceSynchronize());
}
static void launch(hipFunction_t fn, unsigned c) {
  CK(H.ModuleLaunchKernel(fn, geo[0], geo[1], geo[2], geo[3], geo[4], geo[5], geo[6], NULL, slots[c], NULL));
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
/* the bytes of copy 0 after a launch */
static uint8_t **download(void) {
  uint8_t **o = calloc(nreg, sizeof *o);
  for (unsigned r = 0; r < nreg; r++) {
    if (!(o[r] = malloc(rlen[r] ? rlen[r] : 1))) die("out of memory");
    if (rlen[r]) CK(H.Memcpy(o[r], dev[0][r], rlen[r], D2H));
  }
  return o;
}

static hipEvent_t e0, e1;
static int n_retry;
/* microseconds per launch of a block of n launches; <= 0 when the events gave no valid time */
static double block_once(hipFunction_t fn, int n) {
  restore();
  for (int i = 0; i < 3; i++) launch(fn, (unsigned)i % ncopy);
  CK(H.EventRecord(e0, NULL));
  for (int i = 0; i < n; i++) launch(fn, (unsigned)i % ncopy);
  CK(H.EventRecord(e1, NULL));
  CK(H.EventSynchronize(e1));
  float ms = 0;
  if (H.EventElapsedTime(&ms, e0, e1)) return -1.0;
  return ms * 1000.0 / n;
}
static double block(hipFunction_t fn, int n) {
  for (int i = 0; i < 5; i++) {
    double t = block_once(fn, n);
    if (t > 0) return t;
    n_retry++;
  }
  die("the events gave no valid time in 5 tries");
  return 0;
}
static int cmpd(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}
/* median and minimum of the rounds, leaving out any below a quarter of the median (an event glitch: a block cannot
 * be that much faster than its own median); returns how many were left out */
static int summarize(double *t, int n, double *med, double *mn) {
  qsort(t, n, sizeof t[0], cmpd);
  double m = t[n / 2];
  int k = 0;
  while (k < n && t[k] < 0.25 * m) k++;
  *med = t[k + (n - k) / 2], *mn = t[k];
  return k;
}

int main(int argc, char **argv) {
  const char *cap = NULL, *refp = NULL, *candp = NULL, *cname = NULL;
  int n = 50, rounds = 7, cold = 1;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-r") && i + 1 < argc) rounds = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-k") && i + 1 < argc) cname = argv[++i];
    else if (!strcmp(argv[i], "--cold") && i + 1 < argc) cold = atoi(argv[++i]);
    else if (!cap) cap = argv[i];
    else if (!refp) refp = argv[i];
    else if (!candp) candp = argv[i];
    else die("too many arguments");
  }
  if (!cap || !refp || n < 1 || n > 10000 || rounds < 1 || rounds > 100 || cold < 1 || cold > MAX_COPY) {
    fprintf(stderr, "usage: kbench CAPTURE REF.hsaco [CAND.hsaco] [-n 1..10000] [-r 1..100] [-k CAND_KERNEL] "
                    "[--cold 1..%d]\n", MAX_COPY);
    return 2;
  }
  ncopy = (unsigned)cold;
  load_capture(cap);
  if (!cname) cname = kname;
  double mb = 0;
  for (unsigned r = 0; r < nreg; r++) mb += rlen[r] / 1048576.0;
  if (mb * ncopy > 8192.0) die("the copies would take more than 8 GB of VRAM; use fewer --cold copies");

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

  for (unsigned c = 0; c < ncopy; c++) {
    for (unsigned r = 0; r < nreg; r++) {
      void *p;
      CK(H.Malloc(&p, rlen[r] + SLACK)); /* zeroed slack: a kernel may read a tile past the captured end */
      CK(H.Memset(p, 0, rlen[r] + SLACK));
      dev[c][r] = p;
    }
    for (unsigned i = 0; i < nargs; i++) {
      vals[c][i] = areg[i] == ~0u ? aval[i] : (uint64_t)(uintptr_t)(dev[c][areg[i]] + aoff[i]);
      slots[c][i] = &vals[c][i];
    }
  }
  printf("kernel %s: grid %u x %u x %u, block %u, LDS %u, %u arguments, %u allocations (%.1f MB", kname, geo[0],
         geo[1], geo[2], geo[3], geo[6], nargs, nreg, mb);
  printf(ncopy > 1 ? ", %u copies)\n" : ")\n", ncopy);

  hipFunction_t fr = load_fn(refp, kname), fc = candp ? load_fn(candp, cname) : NULL;
  restore();
  launch(fr, 0);
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
    launch(fc, 0);
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
  double tr[100], tc[100];
  for (int i = 0; i < rounds; i++) {
    if (fc && (i & 1)) tc[i] = block(fc, n), tr[i] = block(fr, n);
    else {
      tr[i] = block(fr, n);
      if (fc) tc[i] = block(fc, n);
    }
  }
  if (n_retry) printf("(%d blocks repeated: the events gave no valid time)\n", n_retry);
  double mr, lr, mc = 0, lc = 0;
  int dr = summarize(tr, rounds, &mr, &lr), dc = fc ? summarize(tc, rounds, &mc, &lc) : 0;
  printf("ref : median %8.2f us  min %8.2f us  (%d rounds x %d launches", mr, lr, rounds, n);
  printf(dr ? ", %d dropped)\n" : ")\n", dr);
  if (fc) {
    printf("cand: median %8.2f us  min %8.2f us  (%+.1f%% median", mc, lc, 100.0 * (mc - mr) / mr);
    printf(dc ? ", %d dropped)\n" : ")\n", dc);
  }
  return 0;
}
