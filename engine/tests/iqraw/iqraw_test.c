/* iqraw_test.c -- the harness for kernels/ie_iqraw.inc (MAT_RAWQ blocks).
 *
 * The oracle is ggml's own ggml-quants.c, compiled on its own (see run.sh),
 * on real blocks sliced out of models/Underdog-Saluki-27B-1.0-IQ2-mix.gguf by
 * slice_iqraw.py:
 *
 *  (a) ie_rq_dequant(type, block, y) is bit-identical to ggml's
 *      dequantize_row_<type>(block, z, 256) on every sliced block, and
 *      ie_rq_dequant_row over two blocks equals the two block dequantizers.
 *  (b) with random int8 activations and per-32 scales da, ie_rq_dot(type,
 *      block, xq, nb = 16, qb) equals the reference
 *        sum over the 8 sub-blocks of sum_j w[32s+j] * q8[256qb+32s+j] * da[8qb+s]
 *      computed in double, for every block, at qb = 0 and qb = 1.
 *
 * Usage: iqraw_test <slice-dir>   (the dir holding iqraw_<type>.bin and
 *          iqraw_manifest.txt; default ".")
 */
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* the host build of the include, as src/cpu.c and src/model.c do it */
#define IE_RAWQ_HOST
#define IE_RQP const uint8_t
#define IE_RQF const float
#define IE_RQU const uint32_t
#include "../../kernels/ie_iqraw.inc"

/* ggml's dequantizers (ggml/src/ggml-quants.c): one 256-weight block in, 256
 * floats out. The block pointer is opaque here, and the block structs are
 * ggml's (ggml-common.h) -- the first arguments are those pointers. */
void dequantize_row_q2_K(const void *, float *, int64_t);
void dequantize_row_iq2_xxs(const void *, float *, int64_t);
void dequantize_row_iq2_xs(const void *, float *, int64_t);
void dequantize_row_iq2_s(const void *, float *, int64_t);
void dequantize_row_iq3_xxs(const void *, float *, int64_t);
void dequantize_row_iq1_s(const void *, float *, int64_t);
void dequantize_row_iq1_m(const void *, float *, int64_t);
void dequantize_row_iq3_s(const void *, float *, int64_t);

/* The four ggml symbols ggml-quants.c leaves undefined: only its quantizers
 * (and ggml_validate_row_data) call them, which this harness never runs. */
void ggml_abort(const char *file, int line, const char *fmt, ...) {
  (void)file;
  (void)line;
  fputs("ggml_abort: ", stderr);
  (void)fmt;
  abort();
}
size_t ggml_row_size(int type, int64_t ne) {
  (void)type;
  (void)ne;
  abort();
}
size_t ggml_type_size(int type) {
  (void)type;
  abort();
}
const char *ggml_type_name(int type) {
  (void)type;
  return "?";
}

typedef struct {
  uint32_t type;
  const char *name;
  uint32_t bs;
  void (*gq)(const void *, float *, int64_t);
} qinfo;

static const qinfo QT[] = {
    {IE_RQ_Q2_K, "Q2_K", 84u, dequantize_row_q2_K},   {IE_RQ_IQ2_XXS, "IQ2_XXS", 66u, dequantize_row_iq2_xxs},
    {IE_RQ_IQ2_XS, "IQ2_XS", 74u, dequantize_row_iq2_xs}, {IE_RQ_IQ2_S, "IQ2_S", 82u, dequantize_row_iq2_s},
    {IE_RQ_IQ3_XXS, "IQ3_XXS", 98u, dequantize_row_iq3_xxs}, {IE_RQ_IQ1_S, "IQ1_S", 50u, dequantize_row_iq1_s},
    {IE_RQ_IQ1_M, "IQ1_M", 56u, dequantize_row_iq1_m}, {IE_RQ_IQ3_S, "IQ3_S", 110u, dequantize_row_iq3_s},
};
#define NQT (sizeof(QT) / sizeof(QT[0]))
#define MAXB 16u   /* blocks read per type (the slicer writes 8) */
#define NB 16u     /* Q8 blocks of 32 per row: two 256-weight blocks (qb 0, 1) */
#define NROUND 3u  /* independent activation sets per block */

static uint32_t rng = 0x6d2b79f5u;
static uint32_t rnd(void) {
  rng ^= rng << 13;
  rng ^= rng >> 17;
  rng ^= rng << 5;
  return rng;
}

static int load(const char *dir, const char *fn, uint8_t *dst, size_t n) {
  char path[1024];
  snprintf(path, sizeof path, "%s/%s", dir, fn);
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "iqraw_test: cannot open %s\n", path);
    return 0;
  }
  const size_t got = fread(dst, 1, n, f);
  fclose(f);
  if (got != n) {
    fprintf(stderr, "iqraw_test: %s: %zu of %zu bytes\n", path, got, n);
    return 0;
  }
  return 1;
}

static unsigned fbits(float f) {
  unsigned u;
  memcpy(&u, &f, sizeof u);
  return u;
}
static int same_bits(float a, float b) { return fbits(a) == fbits(b); }

/* NROUND independent Q8 activation sets: int8 values, the per-32 scale and
 * the exact per-32 int8 sum (as the engine's quantize kernel writes them). */
static void fill_q8(uint32_t round, int8_t *qv, float *da, uint32_t *as) {
  rng = 0x6d2b79f5u + 0x9e3779b9u * round;
  for (uint32_t b = 0; b < NB; b++) {
    int32_t sum = 0;
    for (uint32_t j = 0; j < 32u; j++) {
      const int32_t v = (int32_t)(rnd() % 255u) - 127;
      qv[32u * b + j] = (int8_t)v;
      sum += v;
    }
    da[b] = 0.25f + (float)(rnd() & 1023u) / 512.0f; /* 0.25 .. 2.25 */
    as[b] = (uint32_t)sum;
  }
}

int main(int argc, char **argv) {
  const char *dir = argc > 1 ? argv[1] : ".";
  char mpath[1024];
  snprintf(mpath, sizeof mpath, "%s/iqraw_manifest.txt", dir);
  FILE *mf = fopen(mpath, "rb");
  if (!mf) {
    fprintf(stderr, "iqraw_test: cannot open %s (run slice_iqraw.py first)\n", mpath);
    return 2;
  }

  /* the Q8 activation buffer of one row: int8 at xq[0, 32 nb), the per-32
   * scales at 32 nb, the per-32 int8 sums at 36 nb (src/common.h q8_bytes) */
  _Alignas(64) static uint8_t xq[40u * NB + 64u];
  int8_t *qv = (int8_t *)xq;
  float *da = (float *)(xq + 32u * NB);
  uint32_t *as = (uint32_t *)(xq + 36u * NB);
  for (uint32_t round = 0; round < NROUND; round++) fill_q8(round, qv, da, as);   /* the last round stays for the report */

  static uint8_t blkbuf[MAXB * 128]; /* the sliced blocks, back to back */
  uint32_t n_a_blocks = 0, n_a_ok = 0, n_a_row = 0, n_a_row_ok = 0;
  uint32_t n_b_blocks = 0, n_b_ok = 0, n_b_absok = 0, n_b_cancel = 0, n_b_ulp = 0;
  double max_abs = 0.0, max_rel = 0.0;
  int bad = 0;

  printf("== (a) ie_rq_dequant vs ggml dequantize_row_*: 256 floats, bit for bit\n");
  char line[4096];
  while (fgets(line, sizeof line, mf)) {
    unsigned t = 0, bs = 0, n = 0;
    char fn[256];
    if (sscanf(line, "%u %u %u %255s", &t, &bs, &n, fn) != 4) continue;
    if (n > MAXB) n = MAXB;
    const qinfo *q = NULL;
    for (size_t i = 0; i < NQT; i++)
      if (QT[i].type == t) q = &QT[i];
    if (!q) {
      fprintf(stderr, "iqraw_test: manifest type %u is not handled here\n", t);
      bad = 1;
      continue;
    }
    if (q->bs != bs || ie_rq_bs(t) != bs) {
      fprintf(stderr, "iqraw_test: %s: block size %u, manifest %u, ie_rq_bs %u\n", q->name, q->bs, bs, ie_rq_bs(t));
      bad = 1;
      continue;
    }
    if (n < 2) {
      fprintf(stderr, "iqraw_test: %s: need 2 blocks for the row check, manifest has %u\n", q->name, n);
      return 2;
    }
    if (!load(dir, fn, blkbuf, (size_t)n * bs)) return 2;

    uint32_t ok = 0;
    double dmax = 0.0;
    for (uint32_t k = 0; k < n; k++) {
      float ref[256], got[256];
      q->gq(blkbuf + k * bs, ref, 256);
      ie_rq_dequant(t, blkbuf + k * bs, got);
      bool seen = false;
      for (uint32_t i = 0; i < 256u; i++) {
        if (!same_bits(ref[i], got[i])) {
          const double d = fabs((double)ref[i] - (double)got[i]);
          if (d > dmax) dmax = d;
          if (!seen) {
            seen = true;
            printf("   %s block %u: first mismatch at %u: ggml %.9g (0x%08x) / ie %.9g (0x%08x)\n", q->name, k, i, ref[i],
                   fbits(ref[i]), got[i], fbits(got[i]));
          }
        }
      }
      if (!seen) ok++;
    }
    /* the row form: two blocks back to back must give the two block results */
    float row[512], one[256], two[256];
    q->gq(blkbuf, one, 256);
    q->gq(blkbuf + bs, two, 256);
    ie_rq_dequant_row(t, blkbuf, 512u, row);
    if (memcmp(row, one, sizeof one) == 0 && memcmp(row + 256, two, sizeof two) == 0) n_a_row_ok++;
    n_a_row++;
    printf("   %-8s %2u B  blocks=%2u  bit-identical=%2u  row-of-2=%s%s\n", q->name, bs, n, ok,
           n_a_row_ok == n_a_row ? "same" : "DIFF", ok == n ? "" : "   <-- FAIL");
    if (ok != n) {
      bad = 1;
      printf("   %s: %u of %u blocks differ (max |diff| %.3g)\n", q->name, n - ok, n, dmax);
    }
    n_a_blocks += n;
    n_a_ok += ok;
  }

  /* (b) the dot product of every block against the Q8 activations, at qb = 0
   * and qb = 1 (nb = 16 blocks of 32, i.e. two 256-weight blocks in a row),
   * for NROUND independent activation sets */
  printf("== (b) ie_rq_dot vs the double reference (%u activation sets, da 0.25..2.25)\n", NROUND);
  rewind(mf);
  while (fgets(line, sizeof line, mf)) {
    unsigned t = 0, bs = 0, n = 0;
    char fn[256];
    if (sscanf(line, "%u %u %u %255s", &t, &bs, &n, fn) != 4) continue;
    if (n > MAXB) n = MAXB;
    const qinfo *q = NULL;
    for (size_t i = 0; i < NQT; i++)
      if (QT[i].type == t) q = &QT[i];
    if (!q) continue;
    if (!load(dir, fn, blkbuf, (size_t)n * bs)) return 2;

    uint32_t ok = 0, absok = 0, cancel = 0, nblk = 0, ulp = 0;
    double tmax = 0.0, trmax = 0.0;
    for (uint32_t round = 0; round < NROUND; round++) {
      fill_q8(round, qv, da, as);
      for (uint32_t k = 0; k < n; k++) {
        const uint32_t qb = k & 1u;
        float w[256];
        q->gq(blkbuf + k * bs, w, 256);
        double ref = 0.0, absref = 0.0;
        for (uint32_t s = 0; s < 8u; s++) {
          double part = 0.0;
          for (uint32_t j = 0; j < 32u; j++) {
            const uint32_t i = 32u * s + j;
            const double term = (double)w[i] * (double)qv[256u * qb + i] * (double)da[8u * qb + s];
            part += term;
            absref += fabs(term);
          }
          ref += part;
        }
        const float got = ie_rq_dot(t, blkbuf + k * bs, xq, NB, qb);
        const double err = fabs((double)got - ref), aref = fabs(ref);
        const double rel = aref > 0.0 ? err / aref : (err == 0.0 ? 0.0 : 1e9);
        const int ok_rel = err <= 1e-6 * aref;        /* the acceptance criterion */
        const int ok_abs = err <= 1e-6 * absref;      /* 1e-6 of the summed magnitudes */
        const int ok_ulp = err <= absref * 0x1p-24;    /* inside one ulp of the weights */
        nblk++;
        if (ok_rel) ok++;
        if (ok_abs) absok++;
        if (ok_ulp) ulp++;
        if (!ok_rel && ok_abs) {
          cancel++; /* |ref| far below sum|terms|: only the norm-wise bound applies */
          printf("   %s block %u qb %u round %u: cancelled sum, ref %.9g err %.3g (%.1fx, sum|terms| %.3g)\n", q->name, k,
                 qb, round, ref, err, absref / (aref > 0.0 ? aref : 1.0), absref);
        }
        if (!ok_rel && !ok_abs) printf("   %s block %u qb %u round %u: got %.9g ref %.9g err %.3g (sum|terms| %.3g)\n",
                                       q->name, k, qb, round, got, ref, err, absref);
        if (ok_rel && rel > trmax) trmax = rel;
        if (err > tmax) tmax = err;
      }
    }
    n_b_blocks += nblk;
    n_b_ok += ok;
    n_b_absok += absok;
    n_b_cancel += cancel;
    n_b_ulp += ulp;
    if (trmax > max_rel) max_rel = trmax;
    if (tmax > max_abs) max_abs = tmax;
    if (ok + cancel != nblk) bad = 1;
    printf("   %-8s blocks=%2u  pass(rel<=1e-6)=%2u  norm-wise=%2u  within-1ulp=%2u  max_abs=%.3g  max_rel=%.3g\n", q->name,
           nblk, ok, absok, ulp, tmax, trmax);
  }
  fclose(mf);

  printf("== totals: dequant blocks=%u bit-identical=%u ; dequant_row rows=%u ok=%u ;\n"
         "   dot blocks=%u  pass(rel<=1e-6)=%u  of which %u cancelled sums within the norm-wise 1e-6  max_abs=%.3g  max_rel=%.3g\n",
         n_a_blocks, n_a_ok, n_a_row, n_a_row_ok, n_b_blocks, n_b_ok, n_b_cancel, max_abs, max_rel);
  printf("   normalised: pass(err<=1e-6*sum|terms|)=%u/%u  and pass(err<=1 ulp of the weights)=%u/%u\n", n_b_absok,
         n_b_blocks, n_b_ulp, n_b_blocks);
  if (n_a_row_ok != n_a_row) bad = 1;
  if (n_b_ok + n_b_cancel != n_b_blocks) bad = 1;
  const int fail = bad != 0 || n_a_ok != n_a_blocks;
  printf("%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
  return fail ? 1 : 0;
}
