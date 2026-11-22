/* dcache_io_check.c -- check the storage side of the disk prompt cache (src/gen.c, dc_*) on a hand-made CPU backend
 * with linear attention layers (no model file, no GPU).
 *   dcache_io_check DIR
 * It includes cpu.c and gen.c to reach their static parts. DIR is emptied of cache files first. Checks: the conv ring
 * slots in a snapshot, the v2 file format (4096-byte sections, the header), the RAM tier (a load after the files were
 * deleted), the hit counter in the header, the int16 form of the state (error bound), the eviction order, the buffer
 * pool, and a writer thread running against the engine thread. */
#include <dirent.h>
#include <pthread.h>

#include "../src/cpu.c"
#include "../src/gen.c"

static int fails;
static void want(int ok, const char *what) {
  printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
  fails += !ok;
}
static void set_env(const char *kv) {
#ifdef _WIN32
  _putenv(kv);
#else
  putenv((char *)kv);
#endif
}

#define NCTX 512u
static model M;
static graph G;
static backend *B;
static gen_ctx C;
static uint32_t toks[32768];
static const char *DIR_;

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd(void) {
  rs ^= rs << 13, rs ^= rs >> 7, rs ^= rs << 17;
  return (uint32_t)(rs >> 11);
}
static float frnd(void) { return ((float)(rnd() % 2000001u) / 1000000.0f - 1.0f); }

static void clear_dir(void) {
  DIR *dd = opendir(DIR_);
  if (!dd) return;
  struct dirent *de;
  char path[1200];
  while ((de = readdir(dd)))
    if (strstr(de->d_name, ".iec") || strstr(de->d_name, ".tmp") || strstr(de->d_name, ".other"))
      snprintf(path, sizeof path, "%s/%s", DIR_, de->d_name), remove(path);
  closedir(dd);
}
static void fixture(uint32_t n_kvl, uint32_t n_kv, uint32_t hd, uint32_t n_rec, uint32_t n_vh, uint32_t conv_dim, uint32_t n_ctx) {
  memset(&M, 0, sizeof M), memset(&G, 0, sizeof G);
  M.n_kvl = n_kvl, M.n_kv = n_kv, M.hd = hd, M.n_rec = n_rec, M.n_vh = n_vh, M.n_kh = 1, M.sd = 128, M.d_conv = 4, M.conv_dim = conv_dim, M.vocab = 1000;
  G.n_ctx = n_ctx, G.arena = 4096, G.kv_len = calloc(n_kvl, 4), G.kv_off = calloc(n_kvl, 8), G.kv_total = (uint64_t)n_kvl * n_ctx, G.df_feat = -1;
  for (uint32_t l = 0; l < n_kvl; l++) G.kv_len[l] = n_ctx, G.kv_off[l] = (uint64_t)l * n_ctx;
  B = cpu_open(&M, &G);
}
static void open_cache(void) {
  memset(&C, 0, sizeof C);
  C.b = B, C.m = &M, C.g = &G;
  C.kv_toks = toks;
  if (!gen_disk_open(&C, DIR_, "dcache_io_check", 0.0, NULL)) ie_die("dcache_io_check: no disk cache");
}
/* make a conversation of pos tokens (seed): the KV rows, the live state, snapshot 0 */
static void make_state(uint32_t seed, uint32_t pos) {
  cpu_backend *b = (cpu_backend *)B;
  rs = seed * 2654435761ull + 12345u;
  for (uint32_t i = 0; i < pos; i++) toks[i] = rnd() % 1000u;
  C.kv_n = pos;
  const size_t kvd = (size_t)M.n_kv * M.hd, nk = (size_t)G.kv_total * kvd;
  for (size_t i = 0; i < nk; i++) b->kc[i] = frnd(), b->vc[i] = frnd();
  const size_t ns = (size_t)M.n_rec * M.n_vh * M.sd * M.sd;
  for (size_t i = 0; i < ns; i++) { /* blocks of different size; one block of zeros, one with an outlier */
    const size_t blk = i / 128;
    float v = frnd() * (blk % 5 == 0 ? 1e-3f : blk % 7 == 0 ? 50.0f : 1.0f);
    if (blk == 3) v = 0.0f;
    if (blk == 11 && i % 128 == 5) v = 3e4f;
    b->st[i] = v;
  }
  for (size_t i = 0; i < (size_t)M.n_rec * 4 * M.conv_dim; i++) b->ring[i] = frnd();
  B->snap(B, 1, 0, -1);
}
static void stage(uint32_t pos, int kind) {
  C.ck[0].pos = pos, C.ck[0].slot = 0, C.ck[0].kind = kind, C.ck[0].h = 0, C.ck[0].disk = 0, C.n_ck = 1;
  dc_stage(&C, 0);
}
static void stage_write(uint32_t seed, uint32_t pos, int kind) {
  make_state(seed, pos);
  stage(pos, kind);
  gen_disk_write(&C);
}
typedef struct {
  float *kc, *vc, *st, *ring;
} copy_t;
static copy_t save_live(void) {
  cpu_backend *b = (cpu_backend *)B;
  copy_t o;
  const size_t nk = (size_t)G.kv_total * M.n_kv * M.hd, ns = (size_t)M.n_rec * M.n_vh * M.sd * M.sd, nr = (size_t)M.n_rec * 4 * M.conv_dim;
  o.kc = malloc(nk * 4), o.vc = malloc(nk * 4), o.st = malloc(ns * 4), o.ring = malloc(nr * 4);
  memcpy(o.kc, b->kc, nk * 4), memcpy(o.vc, b->vc, nk * 4), memcpy(o.st, b->st, ns * 4), memcpy(o.ring, b->ring, nr * 4);
  return o;
}
static void free_copy(copy_t *o) { free(o->kc), free(o->vc), free(o->st), free(o->ring); }
static void wipe_live(void) {
  cpu_backend *b = (cpu_backend *)B;
  const size_t nk = (size_t)G.kv_total * M.n_kv * M.hd, ns = (size_t)M.n_rec * M.n_vh * M.sd * M.sd, nr = (size_t)M.n_rec * 4 * M.conv_dim;
  memset(b->kc, 0, nk * 4), memset(b->vc, 0, nk * 4), memset(b->st, 0, ns * 4), memset(b->ring, 0, nr * 4);
}
/* the KV rows of positions [0, pos) equal; the state within a bound of ref (0: exactly); the ring slots of positions
 * pos - 3 .. pos - 1 equal and the other one zero */
static int same_rows(const copy_t *ref, uint32_t pos) {
  cpu_backend *b = (cpu_backend *)B;
  const size_t kvd = (size_t)M.n_kv * M.hd;
  for (uint32_t l = 0; l < M.n_kvl; l++) {
    const size_t o = (size_t)G.kv_off[l] * kvd;
    if (memcmp(b->kc + o, ref->kc + o, (size_t)pos * kvd * 4) || memcmp(b->vc + o, ref->vc + o, (size_t)pos * kvd * 4)) return 0;
  }
  return 1;
}
static double state_err(const copy_t *ref, double *rel_worst) {
  cpu_backend *b = (cpu_backend *)B;
  const size_t ns = (size_t)M.n_rec * M.n_vh * M.sd * M.sd;
  double worst = 0;
  *rel_worst = 0;
  for (size_t blk = 0; blk < ns / 128; blk++) {
    double amax = 0, e = 0;
    for (size_t i = 0; i < 128; i++) amax = fmax(amax, fabs(ref->st[blk * 128 + i]));
    for (size_t i = 0; i < 128; i++) e = fmax(e, fabs((double)b->st[blk * 128 + i] - ref->st[blk * 128 + i]));
    worst = fmax(worst, e);
    if (amax > 0) *rel_worst = fmax(*rel_worst, e / (amax / 65534.0)); /* in units of the largest allowed error */
    else if (e != 0) *rel_worst = 1e9;
  }
  return worst;
}
static int ring_ok(const copy_t *ref, uint32_t pos) {
  cpu_backend *b = (cpu_backend *)B;
  for (uint32_t l = 0; l < M.n_rec; l++)
    for (uint32_t s = 0; s < 4; s++) {
      const float *got = b->ring + ((size_t)l * 4 + s) * M.conv_dim, *want_ = ref->ring + ((size_t)l * 4 + s) * M.conv_dim;
      const int needed = s != (pos & 3u);
      for (uint32_t c = 0; c < M.conv_dim; c++)
        if (needed ? got[c] != want_[c] : got[c] != 0.0f) return 0;
    }
  return 1;
}
static void close_cache(void) {
  dc_free(C.dc);
  C.dc = NULL;
}
static int load(uint32_t pos, int slot) {
  const uint64_t eh = dc_ehash(C.dc, toks, pos);
  return dc_load(&C, eh, pos, toks, slot);
}
static int count_files(char type) {
  int n = 0;
  for (uint32_t i = 0; i < C.dc->n; i++) n += C.dc->f[i].type == type;
  return n;
}
/* trim until one more checkpoint or other-sig file is gone (a trim first deletes the blocks nothing needs) */
static void evict_one(void) {
  const int before = count_files('e') + count_files('x');
  for (int t = 0; t < 4 && count_files('e') + count_files('x') == before; t++) C.dc->limit = C.dc->bytes - 1, dc_trim(C.dc);
}
static int read_head(uint64_t hash, char type, dc_head *h, size_t *size) {
  char path[1200];
  dc_path(C.dc, path, sizeof path, type, hash, NULL);
  FILE *fp = fopen(path, "rb");
  if (!fp) return 0;
  int ok = fread(h, sizeof *h, 1, fp) == 1;
  fseek(fp, 0, SEEK_END);
  *size = (size_t)ftell(fp);
  fclose(fp);
  return ok;
}

static pthread_mutex_t LK = PTHREAD_MUTEX_INITIALIZER;
static void lk(int on) { on ? pthread_mutex_lock(&LK) : pthread_mutex_unlock(&LK); }
static void nap(void) { /* about a millisecond (nanosleep returns at once on this C library) */
#ifdef _WIN32
  Sleep(1);
#else
  struct timespec ts = {0, 1000000};
  nanosleep(&ts, NULL);
#endif
}
static volatile int stop_writer;
static void *writer_main(void *arg) {
  (void)arg;
  while (!stop_writer) {
    gen_disk_write(&C);
    nap();
  }
  gen_disk_write(&C);
  return NULL;
}

/* Saluki-sized files (21760-byte rows, 151 MB of state, blocks of 1024): how long writing and loading one checkpoint of
 * 26731 tokens takes through the real code, the copies to and from the CPU backend's memory standing in for the GPU's */
static int big_run(void) {
  const uint32_t pos = 26731;
  fixture(16, 1, 170, 48, 48, 10240, 32768);
  cpu_backend *b = (cpu_backend *)B;
  set_env("IE_DISK_BLOCK=1024"), set_env("IE_DISK_MIN=1024");
  const size_t kvd = 170, nk = (size_t)G.kv_total * kvd, ns = (size_t)M.n_rec * M.n_vh * M.sd * M.sd;
  for (size_t i = 0; i < nk; i++) b->kc[i] = (float)(i % 977) * 0.01f, b->vc[i] = (float)(i % 631) * 0.02f;
  for (size_t i = 0; i < ns; i++) b->st[i] = (float)(i % 1013) * 0.001f;
  for (size_t i = 0; i < (size_t)M.n_rec * 4 * M.conv_dim; i++) b->ring[i] = (float)(i % 97);
  rs = 7;
  for (uint32_t i = 0; i < pos; i++) toks[i] = rnd() % 1000u;
  B->snap(B, 1, 0, -1);
  const char *modes[4] = {"IE_DISK_DIRECT=1", "IE_DISK_DIRECT=0", "IE_DISK_DIRECT=1", "IE_DISK_DIRECT=1"};
  for (int mode = 0; mode < 4; mode++) {
    set_env(modes[mode]);
    set_env(mode == 3 ? "IE_DISK_RAM_GB=8" : "IE_DISK_RAM_GB=0");
    clear_dir();
    open_cache();
    C.kv_n = pos;
    double t0 = now_ms();
    stage(pos, GEN_CK_EVERY);
    const double t_stage = now_ms() - t0;
    t0 = now_ms();
    gen_disk_write(&C);
    const double t_write = now_ms() - t0;
    const double mb = (C.dc->bytes + 0.0) / 1e6;
    if (mode == 3) { /* RAM tier on: the next load reads memory */
    }
    double t_load[3];
    for (int rep = 0; rep < 3; rep++) {
      memset(b->kc, 0, nk * 4), memset(b->vc, 0, nk * 4);
      t0 = now_ms();
      const int r = load(pos, 1);
      t_load[rep] = now_ms() - t0;
      if (r) printf("   load failed (%d)\n", r);
    }
    printf("%s, RAM tier %s: %.0f MB; stage %.0f ms, write %.0f ms (%.2f GB/s); load %.0f / %.0f / %.0f ms (%.2f GB/s first)\n", modes[mode],
           mode == 3 ? "on " : "off", mb, t_stage, t_write, mb / t_write, t_load[0], t_load[1], t_load[2], mb / t_load[0]);
    close_cache();
  }
  clear_dir();
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: dcache_io_check DIR\n");
    return 2;
  }
  DIR_ = argv[1];
  set_env("IE_DISK_BLOCK=16"), set_env("IE_DISK_MIN=16");
  clear_dir();
  if (argc > 2 && !strcmp(argv[2], "big")) return big_run();
  fixture(2, 2, 16, 2, 2, 64, NCTX);
  cpu_backend *cb = (cpu_backend *)B;

  printf("conv ring slots in a snapshot\n");
  {
    const size_t rowb = (size_t)M.conv_dim * 4, state = (size_t)M.n_rec * M.n_vh * M.sd * M.sd * 4;
    want(B->snap_io(B, 0, 1, NULL, 0) == state + (size_t)M.n_rec * 3 * rowb, "snapshot size: the state and 3 of the 4 ring slots per layer");
    make_state(1, 100);
    copy_t ref = save_live();
    uint8_t *buf = malloc(B->snap_io(B, 0, 1, NULL, 0));
    for (uint32_t pos = 1; pos < 12; pos += 5) {
      B->snap_io(B, 0, 1, buf, pos);
      wipe_live();
      B->snap_io(B, 2, 0, buf, pos);
      B->snap(B, 0, 2, -1);
      double rel;
      state_err(&ref, &rel);
      char what[100];
      snprintf(what, sizeof what, "pos %u: state exact, ring slots of positions pos-3..pos-1 exact, the other zero", pos);
      want(rel == 0 && ring_ok(&ref, pos), what);
    }
    free(buf), free_copy(&ref);
  }

  printf("file format v2, hit counter, RAM tier\n");
  set_env("IE_DISK_RAM_GB=0"); /* no RAM tier: loads read the disk */
  open_cache();
  const uint32_t pos1 = 100; /* 6 blocks of 16 and a tail of 4 */
  stage_write(2, pos1, GEN_CK_EVERY);
  copy_t ref = save_live();
  {
    dc_head h;
    size_t sz;
    const uint64_t eh = dc_ehash(C.dc, toks, pos1);
    want(read_head(eh, 'e', &h, &sz) && h.version == 2 && h.pos == pos1 && h.hits == 0 && h.flags == 0 && sizeof(dc_head) == 64, "checkpoint header: version 2, pos, hits 0, flags 0, 64 bytes");
    dc_lay L;
    dc_layout(C.dc, 'e', pos1, 0, &L);
    want(sz == L.total && sz % 4096 == 0 && L.snap_off % 4096 == 0 && L.rows_off % 4096 == 0 && L.tok_off == 4096, "checkpoint file: size and section offsets are multiples of 4096");
    char path[1200];
    dc_path(C.dc, path, sizeof path, 'e', eh, NULL);
    FILE *fp = fopen(path, "rb");
    uint32_t *tk = malloc((size_t)pos1 * 4);
    fseek(fp, (long)L.tok_off, SEEK_SET);
    const size_t got = fread(tk, 4, pos1, fp);
    want(got == pos1 && !memcmp(tk, toks, (size_t)pos1 * 4), "tokens at offset 4096");
    free(tk), fclose(fp);
    uint64_t ch[8];
    dc_chain(C.dc, toks, pos1 / 16, ch);
    want(read_head(ch[0], 'k', &h, &sz) && sz == (size_t)4096 + 4096 + 16 * 512 && h.magic == 0x4B435049u, "block file: header page, token page, 16 rows of 512 bytes");
  }
  {
    wipe_live();
    int r = load(pos1, 1);
    B->snap(B, 0, 1, -1);
    double rel;
    state_err(&ref, &rel);
    want(r == 0 && same_rows(&ref, pos1) && rel == 0 && ring_ok(&ref, pos1), "load from the disk: KV rows, state and ring slots as saved");
    dc_head h;
    size_t sz;
    const uint64_t eh = dc_ehash(C.dc, toks, pos1);
    read_head(eh, 'e', &h, &sz);
    want(h.hits == 1 && C.dc->f[dc_find(C.dc, 'e', eh)].hits == 1, "hits is 1 in the file and the list after one load");
    wipe_live();
    load(pos1, 1);
    read_head(eh, 'e', &h, &sz);
    want(h.hits == 2, "hits is 2 after the second load");
    want(C.dc->ram_bytes == 0, "RAM tier off: nothing kept");
    close_cache();
    open_cache();
    want(C.dc->f[dc_find(C.dc, 'e', eh)].hits == 2, "hits survive a restart (read from the header)");
  }
  close_cache(), clear_dir();
  set_env("IE_DISK_RAM_GB=8");
  open_cache();
  stage_write(2, pos1, GEN_CK_EVERY);
  {
    const uint64_t eh = dc_ehash(C.dc, toks, pos1);
    want(C.dc->ram_bytes > 0 && C.dc->f[dc_find(C.dc, 'e', eh)].img.p != NULL, "written files are in the RAM tier");
    clear_dir(); /* the files are gone; the list still has them */
    wipe_live();
    int r = load(pos1, 1);
    B->snap(B, 0, 1, -1);
    double rel;
    state_err(&ref, &rel);
    want(r == 0 && same_rows(&ref, pos1) && rel == 0 && ring_ok(&ref, pos1), "load from RAM with the files deleted: everything as saved");
    want(C.dc->f[dc_find(C.dc, 'e', eh)].hits == 1, "hits counted in the list");
    C.dc->ram_limit = 0;
    for (uint32_t i = 0; i < C.dc->n; i++) /* drop the tier */
      if (C.dc->f[i].img.p) {
        C.dc->ram_bytes -= C.dc->f[i].img.cap, C.dc->ram_pinned -= C.dc->f[i].img.pinned ? C.dc->f[i].img.cap : 0;
        dc_put_l(C.dc, C.dc->f[i].img), memset(&C.dc->f[i].img, 0, sizeof C.dc->f[i].img);
      }
    want(load(pos1, 1) == -1, "load with the files gone and no RAM copy: -1, nothing changed");
  }
  close_cache(), clear_dir();

  printf("RAM tier eviction and the pinned pool\n");
  {
    set_env("IE_DISK_RAM_GB=0.0008"); /* room for a checkpoint and a few blocks of this size */
    open_cache();
    uint64_t eh[4];
    for (int i = 0; i < 4; i++) {
      stage_write(10 + (uint32_t)i, 40, GEN_CK_EVERY);
      eh[i] = dc_ehash(C.dc, toks, 40);
    }
    int inram = 0;
    for (int i = 0; i < 4; i++) inram += C.dc->f[dc_find(C.dc, 'e', eh[i])].img.p != NULL;
    want(C.dc->ram_bytes <= C.dc->ram_limit && inram >= 1 && inram < 4, "RAM tier stays within its limit (the oldest images are dropped)");
    want(C.dc->f[dc_find(C.dc, 'e', eh[3])].img.p != NULL, "the newest image is kept");
    dc_mem a = dc_get(C.dc, 5u << 20), b2 = dc_get(C.dc, 3u << 20);
    want(((uintptr_t)a.p & 4095u) == 0 && ((uintptr_t)b2.p & 4095u) == 0 && a.pinned && b2.pinned, "buffers are 4096-aligned and counted as pinned");
    char *pa = a.p;
    dc_put(C.dc, a);
    dc_mem a2 = dc_get(C.dc, 4600000u);
    want(a2.p == pa, "a returned buffer is reused (no new allocation)");
    dc_put(C.dc, a2), dc_put(C.dc, b2);
    C.dc->pinned_limit = 0;
    dc_mem z = dc_get(C.dc, 40u << 20);
    want(!z.pinned && ((uintptr_t)z.p & 4095u) == 0 && C.dc->pinned <= 20u << 20, "past the pinned limit: plain aligned memory, free pinned buffers released");
    dc_put(C.dc, z);
    close_cache(), clear_dir();
    set_env("IE_DISK_RAM_GB=8");
  }

  printf("state as int16\n");
  {
    set_env("IE_DISK_STATE16=1"), set_env("IE_DISK_RAM_GB=0");
    open_cache();
    stage_write(2, pos1, GEN_CK_EVERY);
    dc_head h;
    size_t sz;
    const uint64_t eh = dc_ehash(C.dc, toks, pos1);
    dc_lay L32, L16;
    dc_layout(C.dc, 'e', pos1, 0, &L32), dc_layout(C.dc, 'e', pos1, DC_F_STATE16, &L16);
    want(read_head(eh, 'e', &h, &sz) && h.flags == 1 && sz == L16.total && sz < L32.total && sz % 4096 == 0, "file has flag 1 and the smaller layout");
    printf("    snapshot %zu bytes as f32, %zu as int16 (file %zu against %zu)\n", C.dc->snapb, C.dc->snap16b, L16.total, L32.total);
    wipe_live();
    int r = load(pos1, 1);
    B->snap(B, 0, 1, -1);
    double rel, worst = state_err(&ref, &rel);
    want(r == 0 && same_rows(&ref, pos1) && ring_ok(&ref, pos1), "load: KV rows, ring slots exact");
    printf("    worst state error %.3g; relative to the allowed amax/65534 per 128 values: %.3f\n", worst, rel);
    want(rel <= 1.02 && rel > 0, "state error within amax/65534 per block of 128 (blocks of zeros stay zero)");
    close_cache();
    open_cache(); /* the flag-1 layout is accepted when the files are listed again */
    want(count_files('e') == 1 && C.dc->f[dc_find(C.dc, 'e', eh)].flags == 1 && count_files('k') == 6, "reopened: the int16 checkpoint and its 6 blocks are listed");
    wipe_live();
    r = load(pos1, 1);
    B->snap(B, 0, 1, -1);
    double rel2;
    state_err(&ref, &rel2);
    want(r == 0 && rel2 == rel, "and loads the same state");
    close_cache(), clear_dir();
    set_env("IE_DISK_STATE16=0");
    open_cache();
    stage_write(2, pos1, GEN_CK_EVERY);
    want(read_head(eh, 'e', &h, &sz) && h.flags == 0, "without IE_DISK_STATE16 the flag is 0");
    set_env("IE_DISK_STATE16=1");
    close_cache();
    open_cache(); /* a flag-0 file is read by a cache that writes flag 1 */
    wipe_live();
    r = load(pos1, 1);
    B->snap(B, 0, 1, -1);
    state_err(&ref, &rel);
    want(r == 0 && rel == 0, "a file without the flag loads exactly, also with IE_DISK_STATE16=1");
    close_cache(), clear_dir();
    set_env("IE_DISK_STATE16=0"), set_env("IE_DISK_RAM_GB=8");
  }
  free_copy(&ref);

  printf("eviction order\n");
  {
    set_env("IE_DISK_RAM_GB=0");
    open_cache();
    const uint32_t seeds[6] = {21, 22, 23, 24, 25, 26};
    const int kinds[6] = {GEN_CK_EVERY, GEN_CK_EVERY, GEN_CK_EVERY, GEN_CK_SPLIT, GEN_CK_MARK, GEN_CK_ANSWER};
    uint64_t eh[6];
    for (int i = 0; i < 6; i++) {
      stage_write(seeds[i], 40, kinds[i]);
      eh[i] = dc_ehash(C.dc, toks, 40);
    }
    /* 0: probationary, used 100; 1: probationary, 200; 2: loaded before (hits 1), 50; 3: split, 60; 4: mark, 70; 5: answer, 300 */
    const int64_t used[6] = {100, 200, 50, 60, 70, 300};
    for (int i = 0; i < 6; i++) C.dc->f[dc_find(C.dc, 'e', eh[i])].used = used[i];
    C.dc->f[dc_find(C.dc, 'e', eh[2])].hits = 1;
    dc_file *x = dc_add(C.dc);
    x->type = 'x', x->name = dc_alloc(16), snprintf(x->name, 16, "old.other.iec"), x->bytes = 4096, x->used = 1000000;
    char path[1200];
    snprintf(path, sizeof path, "%s/old.other.iec", DIR_);
    FILE *fp = fopen(path, "wb");
    fwrite(C.dc->f, 1, 4096, fp);
    fclose(fp);
    C.dc->bytes += x->bytes;
    const int order[6] = {0, 1, 5, 2, 3, 4}; /* the answer (300) is probationary and newer than 0 and 1 */
    int gone_x = 0, bad = 0;
    for (int round = 0; round < 7; round++) {
      evict_one();
      if (round == 0) {
        gone_x = dc_find(C.dc, 'x', 0) < 0 && count_files('e') == 6;
        continue;
      }
      for (int i = 0; i < 6; i++) {
        const int alive = dc_find(C.dc, 'e', eh[i]) >= 0;
        int expect_alive = 1;
        for (int k = 0; k < round; k++) expect_alive &= order[k] != i;
        bad += alive != expect_alive;
      }
    }
    C.dc->limit = C.dc->bytes - 1, dc_trim(C.dc);
    want(gone_x, "the other sig's file goes first, before any checkpoint");
    want(!bad && count_files('e') == 0, "then probationary by age (0, 1, 5), then loaded (2), split (3), mark (4)");
    want(count_files('k') == 0, "blocks no checkpoint needs are gone with them");
    close_cache(), clear_dir();
    set_env("IE_DISK_RAM_GB=8");
  }

  printf("a writer thread against the engine thread\n");
  {
    set_env("IE_DISK_RAM_GB=0.05");
    memset(&C, 0, sizeof C);
    C.b = B, C.m = &M, C.g = &G, C.kv_toks = toks;
    if (!gen_disk_open(&C, DIR_, "dcache_io_check", 0.0, lk)) ie_die("no disk cache");
    pthread_t th;
    stop_writer = 0;
    pthread_create(&th, NULL, writer_main, NULL);
    uint32_t okc = 0, n = 24;
    uint32_t seedpos[24];
    copy_t keep[24];
    for (uint32_t i = 0; i < n; i++) {
      seedpos[i] = 48 + (i % 5) * 16 + 3;
      make_state(100 + i, seedpos[i]);
      keep[i] = save_live();
      stage(seedpos[i], GEN_CK_EVERY);
      if (i % 3 == 2) { /* meanwhile load an earlier one */
        const uint32_t j = i - 2;
        rs = (100 + j) * 2654435761ull + 12345u;
        for (uint32_t t = 0; t < seedpos[j]; t++) toks[t] = rnd() % 1000u;
        wipe_live();
        for (int tries = 0; tries < 5000 && !dc_best(C.dc, toks, seedpos[j], 0, 0, &(uint64_t){0}, &(uint32_t){0}, &(uint32_t){0}, &(uint32_t){0}); tries++) nap();
        const int r = load(seedpos[j], 1);
        B->snap(B, 0, 1, -1);
        double rel;
        state_err(&keep[j], &rel);
        okc += r == 0 && same_rows(&keep[j], seedpos[j]) && rel == 0;
      }
    }
    stop_writer = 1;
    pthread_join(th, NULL);
    want(okc == n / 3, "loads while the writer thread runs: all correct");
    want(C.dc->qbytes == 0 && count_files('e') == (int)n, "all checkpoints written, queue empty");
    uint32_t pend = 0;
    for (uint32_t i = 0; i < C.dc->n; i++) pend += C.dc->f[i].pending;
    want(pend == 0 && C.dc->ram_bytes <= C.dc->ram_limit, "nothing pending; the RAM tier within its limit");
    for (uint32_t i = 0; i < n; i++) free_copy(&keep[i]);
    close_cache(), clear_dir();
  }
  (void)cb;
  printf("dcache_io_check: %s\n", fails ? "FAIL" : "pass");
  return fails ? 1 : 0;
}
