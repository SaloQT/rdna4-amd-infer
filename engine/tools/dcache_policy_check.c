/* dcache_policy_check.c -- check the disk prompt cache of gen_run (ie-serve's IE_DISK_CACHE) and its policy (which
 * checkpoints are written, and when) on the CPU with a tiny model.
 *   dcache_policy_check MODEL.gguf DIR
 * DIR is emptied of cache files first. Blocks of 64 positions and checkpoints from 64 tokens (IE_DISK_BLOCK,
 * IE_DISK_MIN). Sampled with one seed: on the CPU (one token per step) an answer is the same whether its prompt
 * continued from memory, from disk, or ran without the cache.
 *   1. the files and the storage (every checkpoint admitted, IE_DISK_ADMIT=all, a checkpoint every 64 positions):
 *      A, A again (memory), B (another prompt), A (disk), a new gen_ctx on DIR (a restarted server): A (disk),
 *      A with another tail (a disk checkpoint before the tail).
 *   2. write-back: the turns of one conversation write nothing; the switch to another conversation writes the
 *      admitted checkpoints (not the periodic ones) and the first conversation is read back; idle writes the rest.
 *   3. marks (IE_CACHE_MARKS=32,96): a system prompt of 200 tokens whose end changes from session to session; after a
 *      "restart" the second session (changed from 170) resumes from the checkpoint at 160, a third one (changed from
 *      130) from 100 and learns the distance of where it differed; a fresh system prompt gets a checkpoint at that
 *      distance from its end.
 *   4. eviction: a full cache (2 checkpoints) writes the admitted checkpoint it drops.
 *   5. learned prefix: detect a changed token from a checkpoint on disk and use
 *      the new boundary for an unseen variant after another restart. */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/backend.h"
#include "../src/common.h"
#include "../src/gen.h"
#include "../src/gguf.h"
#include "../src/model.h"

typedef struct {
  uint32_t v[256], n;
} toks;
static int collect(void *ctx, const uint32_t *t, uint32_t n) {
  toks *o = ctx;
  for (uint32_t i = 0; i < n && o->n < 256; i++) o->v[o->n++] = t[i];
  return 0;
}
static int same(const toks *a, const toks *b) { return a->n == b->n && !memcmp(a->v, b->v, a->n * 4); }
static void set(const char *kv) {
#ifdef _WIN32
  _putenv(kv);
#else
  putenv((char *)kv);
#endif
}
static int fails;
static const char *CDIR; /* the cache directory (the log lines count its files) */
static void want(int ok, const char *what) {
  printf("  %s: %s\n", what, ok ? "ok" : "FAIL");
  fails += !ok;
}
/* the files in dir whose name holds sub (".iec": all; "-e-": checkpoints); remove them with del */
static int files(const char *dir, const char *sub, int del) {
  int n = 0;
  DIR *dd = opendir(dir);
  if (!dd) return 0;
  struct dirent *de;
  char path[1200];
  while ((de = readdir(dd)))
    if (strstr(de->d_name, sub)) {
      n++;
      if (del) snprintf(path, sizeof path, "%s/%s", dir, de->d_name), remove(path);
    }
  closedir(dd);
  return n;
}
static void clean(const char *dir) {
  files(dir, ".iec", 1);
  files(dir, "policy-splits", 1);
}
static void run(gen_ctx *c, gen_params *p, const uint32_t *pr, uint32_t n, toks *o, gen_stats *st, const char *what) {
  memset(o, 0, sizeof *o);
  p->ctx = o;
  gen_run(c, pr, n, p, st);
  printf("%-34s prompt %4u, cached %4u (from %d), answer %u %u %u ...; %d checkpoint files\n", what, st->n_prompt, st->cached,
         st->cache_hit, o->v[0], o->v[1], o->v[2], files(CDIR, "-e-", 0));
}
static int has_ck(const gen_ctx *c, uint32_t pos, int kind) {
  for (uint32_t i = 0; i < c->n_ck; i++)
    if (c->ck[i].pos == pos && c->ck[i].kind == kind) return 1;
  return 0;
}
static void reopen(gen_ctx *c, backend *b, const model *m, const graph *g, const char *dir) {
  gen_free(c);
  gen_init(c, b, m, g);
  if (!gen_disk_open(c, dir, "dcache_policy_check", 0.0, NULL)) ie_die("dcache_policy_check: no disk cache with this model");
}
/* n tokens of a made-up prompt (salt picks which), without newlines except one at each position i % 20 == 19 from nl0 on */
static void fill(const gen_ctx *c, uint32_t *t, uint32_t from, uint32_t to, uint32_t salt, uint32_t V, uint32_t nlt, uint32_t nl0, uint32_t nl1) {
  for (uint32_t i = from; i < to; i++) {
    uint32_t x = (i * 7919u + salt * 104729u + 13u) % V;
    while (c->nl[x]) x = (x + 1u) % V;
    t[i] = i >= nl0 && i < nl1 && i % 20u == 19u ? nlt : x;
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: dcache_policy_check MODEL.gguf DIR\n");
    return 2;
  }
  const char *dir = argv[2];
  CDIR = dir;
  set("IE_DISK_BLOCK=64"), set("IE_DISK_MIN=64"), set("IE_CACHE_EVERY=64"), set("IE_CACHE_MARKS=0"), set("IE_CACHE_MARK_GAP=8");
  set("IE_DISK_ADMIT=all");
  clean(dir);
  gguf_file g;
  gguf_open(&g, argv[1]);
  model m;
  model_load(&m, &g);
  graph gr;
  graph_build(&gr, &m, 1024, 1, 0);
  plan_arena(&gr, 0);
  backend *b = cpu_open(&m, &gr);
  gen_ctx c;
  gen_init(&c, b, &m, &gr);
  if (!gen_disk_open(&c, dir, "dcache_policy_check", 0.0, NULL)) ie_die("dcache_policy_check: no disk cache with this model");
  gen_params p = {0};
  p.max_new = 16, p.chunk = 1, p.cache = 1, p.emit = collect, p.temp = 1.0f, p.top_p = 1.0f, p.seed = 7;

  const uint32_t V = m.vocab < 5000 ? m.vocab : 5000;
  uint32_t nlt = 0;
  while (nlt < m.vocab && !c.nl[nlt]) nlt++;
  if (nlt == m.vocab) ie_die("dcache_policy_check: no token holds a newline");
  static uint32_t A[400], A2[400], Bp[300];
  const uint32_t na = 300, nb = 200;
  for (uint32_t i = 0; i < na; i++) A[i] = (i * 7919u + 13u) % V, A2[i] = i < 250 ? A[i] : (i * 211u + 17u) % V;
  for (uint32_t i = 0; i < nb; i++) Bp[i] = (i * 104729u + 7u) % V;
  toks ref = {0}, ref2 = {0}, o = {0};
  gen_stats st;

  printf("1. storage, every checkpoint admitted\n");
  p.cache = 0;
  run(&c, &p, A, na, &ref, &st, "A without the cache");
  run(&c, &p, A2, na, &ref2, &st, "A' without the cache");
  p.cache = 1;
  run(&c, &p, A, na, &o, &st, "1. A");
  want(same(&o, &ref), "answer as without the cache");
  want(files(dir, ".iec", 0) == 0, "nothing written after the request");
  run(&c, &p, A, na, &o, &st, "2. A again (memory)");
  want(st.cache_hit >= 0 && st.cache_hit != GEN_CK_DISK && st.cached == na - 1 && same(&o, &ref), "from memory, same answer");
  want(files(dir, ".iec", 0) == 0, "still nothing written");
  run(&c, &p, Bp, nb, &o, &st, "3. B");
  want(files(dir, "-e-", 0) >= 4, "the switch to B wrote A's checkpoints");
  run(&c, &p, A, na, &o, &st, "4. A after B (disk)");
  want(st.cache_hit == GEN_CK_DISK && st.cached == na - 1 && same(&o, &ref), "from disk, same answer");
  gen_disk_idle(&c); /* what idle does: B's were written at the switch */
  p.cache = 0;
  run(&c, &p, Bp, nb, &o, &st, "   (B without the cache)");
  reopen(&c, b, &m, &gr, dir); /* a restarted server */
  p.cache = 1;
  run(&c, &p, A, na, &o, &st, "5. A after a restart (disk)");
  want(st.cache_hit == GEN_CK_DISK && st.cached == na - 1 && same(&o, &ref), "from disk, same answer");
  run(&c, &p, A2, na, &o, &st, "6. A' (another tail)");
  want(st.cached >= 64 && st.cached <= 250 && same(&o, &ref2), "from a checkpoint before the tail, same answer as A'");
  const int ne_all = files(dir, "-e-", 0);

  /* 2. write-back and the admission: the conversation C (three turns), then D, then C again */
  printf("2. write-back\n");
  set("IE_DISK_ADMIT=policy");
  clean(dir);
  reopen(&c, b, &m, &gr, dir);
  static uint32_t C1[400], C2[400], C3[400], C4[400], Dp[300];
  uint32_t n1 = 200, n2, n3, n4;
  fill(&c, C1, 0, n1, 3, V, nlt, 0, 0);
  fill(&c, Dp, 0, nb, 5, V, nlt, 0, 0);
  toks r1 = {0}, r2 = {0}, r3 = {0}, r4 = {0};
  p.cache = 0;
  run(&c, &p, C1, n1, &r1, &st, "C turn 1 without the cache");
  memcpy(C2, C1, n1 * 4), n2 = n1;
  for (uint32_t i = 0; i + 1 < r1.n; i++) C2[n2++] = r1.v[i];
  fill(&c, C2, n2, n2 + 7, 9, V, nlt, 0, 0), n2 += 7;
  run(&c, &p, C2, n2, &r2, &st, "C turn 2 without the cache");
  memcpy(C3, C2, n2 * 4), n3 = n2;
  for (uint32_t i = 0; i + 1 < r2.n; i++) C3[n3++] = r2.v[i];
  fill(&c, C3, n3, n3 + 7, 11, V, nlt, 0, 0), n3 += 7;
  run(&c, &p, C3, n3, &r3, &st, "C turn 3 without the cache");
  memcpy(C4, C3, n3 * 4), n4 = n3;
  for (uint32_t i = 0; i + 1 < r3.n; i++) C4[n4++] = r3.v[i];
  fill(&c, C4, n4, n4 + 7, 13, V, nlt, 0, 0), n4 += 7;
  toks rD = {0};
  run(&c, &p, Dp, nb, &rD, &st, "D without the cache");
  run(&c, &p, C4, n4, &r4, &st, "C turn 4 without the cache");
  p.cache = 1;
  run(&c, &p, C1, n1, &o, &st, "C turn 1");
  want(same(&o, &r1), "answer as without the cache");
  run(&c, &p, C2, n2, &o, &st, "C turn 2");
  want(st.cache_hit == GEN_CK_ANSWER && same(&o, &r2), "from the last answer, same answer");
  run(&c, &p, C3, n3, &o, &st, "C turn 3");
  want(st.cache_hit == GEN_CK_ANSWER && same(&o, &r3), "from the last answer, same answer");
  want(files(dir, ".iec", 0) == 0, "the turns of one conversation wrote nothing");
  want(gen_disk_pending(&c) >= 2, "the heads wait to be written");
  run(&c, &p, Dp, nb, &o, &st, "D (another conversation)");
  want(same(&o, &rD), "answer as without the cache");
  const int ne = files(dir, "-e-", 0);
  printf("  checkpoint files after the switch: %d (with every checkpoint admitted: %d)\n", ne, ne_all);
  want(ne >= 2 && ne <= 3, "the switch wrote the admitted checkpoints only (two heads and a fallback)");
  run(&c, &p, C3, n3, &o, &st, "C turn 3 again (disk)");
  want(st.cache_hit == GEN_CK_DISK && st.cached == n3 - 1 && same(&o, &r3), "read back from disk, same answer");
  const int nf = files(dir, ".iec", 0);
  run(&c, &p, C4, n4, &o, &st, "C turn 4");
  want(same(&o, &r4) && files(dir, ".iec", 0) == nf, "a follow-up turn: same answer, no files");
  want(gen_disk_pending(&c) > 0, "something waits for idle");
  want(gen_disk_idle(&c) > 0 && files(dir, ".iec", 0) > nf, "idle writes it");
  want(gen_disk_pending(&c) == 0 && gen_disk_idle(&c) == 0, "and then nothing more");

  /* 3. marks near the end of a system prompt */
  printf("3. marks\n");
  set("IE_CACHE_MARKS=32,96"), set("IE_CACHE_EVERY=0");
  clean(dir);
  reopen(&c, b, &m, &gr, dir);
  static uint32_t S1[400], S2[400], S3[400], S4[400];
  const uint32_t ns = 300;
  /* the first 170 tokens are the same in S1 and S2, the first 130 in S3; newlines at 19, 39, ... of the system part */
  fill(&c, S1, 0, 170, 1, V, nlt, 0, 200), fill(&c, S1, 170, ns, 21, V, nlt, 0, 200);
  memcpy(S2, S1, 170 * 4), fill(&c, S2, 170, ns, 22, V, nlt, 0, 200);
  memcpy(S3, S1, 130 * 4), fill(&c, S3, 130, ns, 23, V, nlt, 0, 200);
  toks e2 = {0}, e3 = {0};
  p.cache = 0, p.sys_end = 200, p.last_user = 203;
  run(&c, &p, S2, ns, &e2, &st, "session 2 without the cache");
  run(&c, &p, S3, ns, &e3, &st, "session 3 without the cache");
  p.cache = 1;
  run(&c, &p, S1, ns, &o, &st, "session 1");
  want(has_ck(&c, 100, GEN_CK_MARK) && has_ck(&c, 160, GEN_CK_MARK) && has_ck(&c, 203, GEN_CK_MARK),
       "marks at 100 and 160 (system end 200 - 96, - 32, to line starts) and at the last user message");
  want(gen_disk_idle(&c) >= 5 && files(dir, "-e-", 0) == 5, "idle wrote the three marks and both heads");
  reopen(&c, b, &m, &gr, dir); /* a restart */
  run(&c, &p, S2, ns, &o, &st, "session 2 after a restart");
  want(st.cache_hit == GEN_CK_DISK && st.cached == 160 && same(&o, &e2), "resumed from the mark at 160 (changed from 170), same answer");
  run(&c, &p, S3, ns, &o, &st, "session 3 (changed from 130)");
  want(st.cached == 100 && same(&o, &e3), "resumed from the mark at 100, same answer");
  want(has_ck(&c, 120, GEN_CK_SPLIT), "and a split at the line start 120");
  char sp[1200];
  snprintf(sp, sizeof sp, "%s/policy-splits.txt", dir);
  FILE *fp = fopen(sp, "r");
  unsigned dist = 0;
  want(fp && fscanf(fp, "%u", &dist) == 1 && dist == 80, "the distance 80 (200 - 120) is learned");
  if (fp) fclose(fp);
  /* a fresh system prompt of 260 tokens in a new context: marks at 260 - 32 (220), and 260 - 80 (180) */
  reopen(&c, b, &m, &gr, dir);
  set("IE_CACHE_MARKS=32");
  fill(&c, S4, 0, ns, 31, V, nlt, 0, 260);
  p.sys_end = 260, p.last_user = 0;
  toks e4 = {0};
  p.cache = 0;
  run(&c, &p, S4, ns, &e4, &st, "fresh system prompt without the cache");
  p.cache = 1;
  run(&c, &p, S4, ns, &o, &st, "fresh system prompt");
  want(same(&o, &e4) && has_ck(&c, 220, GEN_CK_MARK) && has_ck(&c, 180, GEN_CK_MARK) && !has_ck(&c, 160, GEN_CK_MARK),
       "marks at 220 (fixed) and 180 (learned), none at 164 -> 160");
  p.sys_end = 0, p.last_user = 0;

  /* 4. a full cache writes what it drops */
  printf("4. eviction\n");
  set("IE_CACHE_MARKS=0"), set("IE_CACHE_EVERY=64"), set("IE_CACHE_CKPT=2");
  clean(dir);
  reopen(&c, b, &m, &gr, dir);
  run(&c, &p, A, na, &o, &st, "A (2 checkpoints at most)");
  want(same(&o, &ref), "answer as without the cache");
  want(files(dir, "-e-", 0) == 1 && gen_disk_pending(&c) == 2, "the periodic checkpoint was written when its slot was reused");
  gen_disk_idle(&c);
  want(files(dir, "-e-", 0) == 3, "idle wrote both heads");
  set("IE_CACHE_CKPT=16");

  printf("5. learned prefix\n");
  set("IE_CACHE_EVERY=0"), set("IE_DISK_MIN=1024");
  clean(dir);
  reopen(&c, b, &m, &gr, dir);
  const uint32_t split = 173;
  for (uint32_t i = 0; i < ns; i++) {
    S1[i] = A[i];
    S2[i] = i < split ? A[i] : (A[i] + 1) % V;
    S3[i] = i < split ? A[i] : (A[i] + 2) % V;
  }
  p.tools_end = 213;
  p.prefix_client = "OMP";
  p.sys_end = p.prefix[2] = 256;
  run(&c, &p, S1, ns, &o, &st, "original agent prompt");
  want(files(dir, "-e-", 0) == 2, "tool and system prefixes saved below the ordinary disk minimum");
  reopen(&c, b, &m, &gr, dir);
  memcpy(S4, S1, ns * sizeof *S4);
  for (uint32_t i = 229; i < ns; i++) S4[i] = (S1[i] + 3) % V;
  run(&c, &p, S4, ns, &o, &st, "same tools, changed system after restart");
  want(st.cache_hit == GEN_CK_DISK && st.cached == p.tools_end, "reused the standalone tool prefix");
  const int before_tool_change = files(dir, "-e-", 0);
  reopen(&c, b, &m, &gr, dir);
  run(&c, &p, S2, ns, &o, &st, "changed tool schema after restart");
  want(st.cached == 0 && has_ck(&c, split, GEN_CK_PREFIX), "learned exact divergence inside the tools from disk");
  want(files(dir, "-e-", 0) == before_tool_change + 3, "learned checkpoint and new tool and system prefixes saved after generation");
  reopen(&c, b, &m, &gr, dir);
  p.cache = 0;
  run(&c, &p, S3, ns, &ref, &st, "third variant without cache");
  p.cache = 1;
  run(&c, &p, S3, ns, &o, &st, "third variant after restart");
  want(st.cache_hit == GEN_CK_DISK && st.cached == split && same(&o, &ref), "reused learned prefix, same answer");

  gen_free(&c);
  b->close(b);
  printf("dcache_policy_check: %s\n", fails ? "FAIL" : "pass");
  return fails ? 1 : 0;
}
