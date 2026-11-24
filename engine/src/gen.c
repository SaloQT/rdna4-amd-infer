/* gen.c -- generation on a backend (see gen.h). The sampling, prefill and
 * spec_generate code came from main.c unchanged, except the emit callback. */
#include <dirent.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#include <sys/utime.h>
#define dc_mkdir(p) _mkdir(p)
#define dc_touch(p) _utime(p, NULL)
#else
#include <utime.h>
#define dc_mkdir(p) mkdir(p, 0755)
#define dc_touch(p) utime(p, NULL)
#endif

#include "common.h"
#include "gen.h"

double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}


static uint32_t model_vocab(const model *m) {
  return m && m->vocab_unpadded ? m->vocab_unpadded : m->vocab;
}

static double rng_u01(sampler *sp) { /* xorshift64*, 53 bits */
  sp->rng ^= sp->rng >> 12, sp->rng ^= sp->rng << 25, sp->rng ^= sp->rng >> 27;
  return (double)((sp->rng * 2685821657736338717ull) >> 11) / 9007199254740992.0;
}
static uint32_t sample_argmax(const float *lg, uint32_t V) {
  uint32_t best = 0;
  for (uint32_t i = 1; i < V; i++)
    if (lg[i] > lg[best]) best = i;
  return best;
}
/* The number of candidates: top_k, or (top_p < 1) the 1000 largest, or all. */
uint32_t sample_k(const sampler *sp, uint32_t V) {
  uint32_t K = sp->top_k ? sp->top_k : (sp->top_p < 1.0f ? 1000u : V);
  return K > V ? V : K;
}
/* ---- the loop penalty (gen.h: sampler.dry_*) ----
 * A bit set of hashes of every dry_allowed-token span of the output (the
 * current span set) makes the common case cheap: when the last dry_allowed
 * tokens are not in it, nothing repeats that long and no candidate is
 * penalized. Only then is the output scanned for the earlier occurrences
 * and their match lengths. */
#define DRY_BITS (1u << 21)
#define DRY_HTAB (1u << 15)
#define DRY_MAX_MATCH 512u /* match lengths are counted up to this */
static uint32_t dry_tok(const sampler *sp, uint32_t i) { return i < sp->dry_n ? sp->dry_h[i] : sp->dry_x[i - sp->dry_n]; }
static uint32_t dry_hash(const sampler *sp, uint32_t end) { /* the dry_allowed tokens before end */
  uint64_t h = 1469598103934665603ull;
  for (uint32_t i = end - sp->dry_allowed; i < end; i++) h = (h ^ dry_tok(sp, i)) * 1099511628211ull;
  return (uint32_t)((h ^ (h >> 29)) & (DRY_BITS - 1u));
}
void dry_begin(sampler *sp) {
  sp->dry_n = sp->dry_seg = sp->dry_xn = sp->dry_hits = 0;
  sp->dry_paused = 0;
  sp->dry_pause_wait_reset = sp->dry_pause_after_reset && sp->dry_reset >= 0;
  sp->dry_x = NULL;
  if (sp->dry_mult <= 0.0f || !sp->dry_allowed) return;
  if (!sp->dry_bits && !(sp->dry_bits = malloc(DRY_BITS / 8))) ie_die("dry: out of memory");
  memset(sp->dry_bits, 0, DRY_BITS / 8);
  if (!sp->dry_head && !(sp->dry_head = malloc((size_t)DRY_HTAB * sizeof(int32_t)))) ie_die("dry: out of memory");
  memset(sp->dry_head, 0xFF, (size_t)DRY_HTAB * sizeof(int32_t));
}
void dry_commit(sampler *sp, const uint32_t *t, uint32_t n) {
  if (sp->dry_mult <= 0.0f || !sp->dry_allowed || !sp->dry_bits) return;
  if (sp->dry_n + n > sp->dry_cap) {
    sp->dry_cap = (sp->dry_n + n) * 2 + 1024;
    if (!(sp->dry_h = realloc(sp->dry_h, (size_t)sp->dry_cap * 4))) ie_die("dry: out of memory");
    if (!(sp->dry_prev = realloc(sp->dry_prev, (size_t)sp->dry_cap * sizeof(int32_t)))) ie_die("dry: out of memory");
  }
  const uint32_t xn = sp->dry_xn;
  sp->dry_xn = 0;
  for (uint32_t k = 0; k < n; k++) {
    const uint32_t p = sp->dry_n++;
    sp->dry_h[p] = t[k];
    if (sp->dry_reset >= 0 && (int64_t)t[k] == sp->dry_reset) { /* a new span set */
      sp->dry_seg = p + 1;
      sp->dry_paused = 0;
      sp->dry_pause_wait_reset = 0;
      memset(sp->dry_bits, 0, DRY_BITS / 8);
      if (sp->dry_head) memset(sp->dry_head, 0xFF, (size_t)DRY_HTAB * sizeof(int32_t));
    } else if (!sp->dry_pause_wait_reset && sp->dry_pause_open_plus_one && t[k] == sp->dry_pause_open_plus_one - 1u) {
      sp->dry_paused = 1;
    } else if (!sp->dry_pause_wait_reset && sp->dry_paused && sp->dry_pause_close_plus_one &&
               t[k] == sp->dry_pause_close_plus_one - 1u) {
      sp->dry_paused = 0;
      sp->dry_seg = p + 1;
      memset(sp->dry_bits, 0, DRY_BITS / 8);
      if (sp->dry_head) memset(sp->dry_head, 0xFF, (size_t)DRY_HTAB * sizeof(int32_t));
    } else if (!sp->dry_paused && p + 1 - sp->dry_seg >= sp->dry_allowed) {
      const uint32_t b = dry_hash(sp, p + 1);
      sp->dry_bits[b >> 6] |= 1ull << (b & 63u);
      if (sp->dry_head && sp->dry_prev) {
        const uint32_t hb = b & (DRY_HTAB - 1u);
        sp->dry_prev[p] = sp->dry_head[hb];
        sp->dry_head[hb] = (int32_t)p;
      }
    }
  }
  sp->dry_xn = xn;
}

static int reasoning_exhausted(const sampler *sp) {
  return sp->reasoning_end_plus_one && !sp->reasoning_left;
}

static void commit_output_token(sampler *sp, uint32_t token) {
  if (sp->reasoning_end_plus_one) {
    if (token == sp->reasoning_end_plus_one - 1u) {
      sp->reasoning_limited = !sp->reasoning_left;
      sp->reasoning_end_plus_one = 0;
    } else if (sp->reasoning_left) sp->reasoning_left--;
  }
  dry_commit(sp, &token, 1);
}
/* Lower the candidates that continue a repeat; sort again (value descending, equal values smaller id first). */
static void dry_apply(sampler *sp, float *v, uint32_t *id, uint32_t n) {
  const uint32_t A = sp->dry_allowed, L = sp->dry_n + sp->dry_xn;
  uint32_t S = sp->dry_seg;
  int paused = sp->dry_paused;
  int wait_reset = sp->dry_pause_wait_reset;
  for (uint32_t k = 0; k < sp->dry_xn; k++) {
    const uint32_t tok = sp->dry_x[k];
    const uint32_t pos = sp->dry_n + k;
    if (sp->dry_reset >= 0 && (int64_t)tok == sp->dry_reset) {
      S = pos + 1u;
      paused = 0;
      wait_reset = 0;
    } else if (!wait_reset && sp->dry_pause_open_plus_one && tok == sp->dry_pause_open_plus_one - 1u) {
      paused = 1;
    } else if (!wait_reset && paused && sp->dry_pause_close_plus_one && tok == sp->dry_pause_close_plus_one - 1u) {
      paused = 0;
      S = pos + 1u;
    }
  }
  if (paused || L < S + A + 1u) return;
  const uint32_t b = dry_hash(sp, L);
  if (!(sp->dry_bits[b >> 6] >> (b & 63u) & 1u)) return; /* the last A tokens are new: no repeat that long */
  /* best[c]: the longest match that candidate c would continue (n <= the top K, at most 1000) */
  uint32_t best[1024];
  if (n > 1024u) n = 1024u;
  memset(best, 0, (size_t)n * 4);
  int any = 0;
  const uint32_t last = dry_tok(sp, L - 1);
  static int dry_hash_on = -1;
  if (dry_hash_on < 0) dry_hash_on = !(getenv("IE_DRY_HASH") && getenv("IE_DRY_HASH")[0] == '0');
  /* i: the end of an earlier occurrence of the last tokens; dry_h[i + 1] would continue the repeat */
  if (dry_hash_on && sp->dry_head && sp->dry_prev) {
    for (int32_t ii = sp->dry_head[b & (DRY_HTAB - 1u)]; ii >= 0; ii = sp->dry_prev[ii]) {
      const uint32_t i = (uint32_t)ii;
      if (i < S + A - 1u) break;
      if (i + 1u >= sp->dry_n || i + 1u >= L) continue;
      if (sp->dry_h[i] != last) continue;
      uint32_t m = 0;
      while (m < DRY_MAX_MATCH && m <= i - S && sp->dry_h[i - m] == dry_tok(sp, L - 1 - m)) m++;
      if (m < A) continue;
      const uint32_t f = sp->dry_h[i + 1];
      for (uint32_t c = 0; c < n; c++)
        if (id[c] == f) {
          if (m > best[c]) best[c] = m, any = 1;
          break;
        }
    }
  } else {
    for (uint32_t i = S + A - 1u; i + 1u < sp->dry_n && i + 1u < L; i++) {
      if (sp->dry_h[i] != last) continue;
      uint32_t m = 0;
      while (m < DRY_MAX_MATCH && m <= i - S && sp->dry_h[i - m] == dry_tok(sp, L - 1 - m)) m++;
      if (m < A) continue;
      const uint32_t f = sp->dry_h[i + 1];
      for (uint32_t c = 0; c < n; c++)
        if (id[c] == f) {
          if (m > best[c]) best[c] = m, any = 1;
          break;
        }
    }
  }
  if (!any) return;
  for (uint32_t c = 0; c < n; c++)
    if (best[c]) {
      const uint32_t e = best[c] - A < 40u ? best[c] - A : 40u;
      v[c] -= sp->dry_mult * powf(sp->dry_base, (float)e);
      sp->dry_hits++;
    }
  for (uint32_t i = 1; i < n; i++) { /* insertion sort: n <= K */
    const float x = v[i];
    const uint32_t t = id[i];
    uint32_t j = i;
    while (j && (v[j - 1] < x || (v[j - 1] == x && id[j - 1] > t))) v[j] = v[j - 1], id[j] = id[j - 1], j--;
    v[j] = x, id[j] = t;
  }
}

/* Sample from n candidates (values v, ids id) in the order value
 * descending, equal values smaller id first: weights exp((v - v[0]) /
 * temp), the nucleus top_p, one draw. v may alias sp->pr. */
uint32_t sample_cands(sampler *sp, float *v, uint32_t *id, uint32_t n) {
  const uint32_t lim = sp->vocab ? sp->vocab : 0xFFFFFFFFu;
  if (!n || !v || !id) {
    fprintf(stderr, "sample_cands: empty candidate set (n=%u)\n", n);
    return 0;
  }
  uint32_t vn = 0;
  for (uint32_t i = 0; i < n; i++) {
    if (id[i] < lim && isfinite(v[i])) {
      if (vn != i) v[vn] = v[i], id[vn] = id[i];
      vn++;
    }
  }
  if (!vn) {
    static unsigned warn_n;
    if (warn_n < 8u) {
      warn_n++;
      fprintf(stderr, "sample_cands: invalid/NaN candidates (n=%u, v[0]=%g, id[0]=0x%x, vocab=%u); falling back to 0\n",
              n, v[0], id[0], sp->vocab);
    }
    return 0;
  }
  n = vn;
  if (sp->dry_mult > 0.0f && sp->dry_allowed && sp->dry_bits) dry_apply(sp, v, id, n);
  if (sp->temp <= 0.0f) return id[0];
  const float mx = v[0];
  double z = 0;
  for (uint32_t i = 0; i < n; i++) {
    const float w = expf((v[i] - mx) / sp->temp);
    sp->pr[i] = isfinite(w) ? w : 0.0f;
    z += sp->pr[i];
  }
  if (!(z > 0.0) || !isfinite(z)) return id[0];
  if (sp->top_p < 1.0f && n > 1) { /* the weights are already descending: keep the nucleus */
    double c = 0;
    uint32_t m = 0;
    while (m < n && c < sp->top_p * z) c += sp->pr[m++];
    if (!m) m = 1, c = sp->pr[0];
    n = m, z = c;
  }
  double u = rng_u01(sp) * z;
  for (uint32_t i = 0; i < n; i++) {
    u -= sp->pr[i];
    if (u <= 0) return id[i];
  }
  return id[n - 1];
}
/* The candidates of rows r0 .. r0 + n - 1 of the logits on the GPU (top K):
 * 1 if done (cv, ci: n * K entries), 0 if the backend cannot (then the
 * host reads the rows and selects). */
int gpu_cands(const sampler *sp, backend *b, const graph *g, uint32_t r0, uint32_t n, uint32_t V, float *cv, uint32_t *ci) {
  const uint32_t K = sample_k(sp, V);
  if (!b->topk || K >= V || getenv("IE_HOST_TOPK")) return 0;
  return b->topk(b, g->logits, r0, n, V, K, cv, ci);
}
uint32_t sample(sampler *sp, const float *lg, uint32_t V) {
  if (sp->temp <= 0.0f) return sample_argmax(lg, V);
  const uint32_t K = sample_k(sp, V);
  uint32_t n = 0;
  if (K == V && !(sp->top_p < 1.0f)) {
    if (!V) return 0;
    float mx = -INFINITY;
    for (uint32_t i = 0; i < V; i++)
      if (isfinite(lg[i]) && lg[i] > mx) mx = lg[i];
    if (!isfinite(mx)) {
      static unsigned warn_s0;
      if (warn_s0 < 8u) {
        warn_s0++;
        fprintf(stderr, "sample: invalid/NaN logits (V=%u, lg[0]=%g); falling back to 0\n", V, lg[0]);
      }
      return 0;
    }
    const float th = mx - 104.0f * sp->temp;
    double z = 0;
    for (uint32_t i = 0; i < V; i++) {
      const float x = lg[i];
      float w = 0.0f;
      if (x >= th && isfinite(x)) {
        w = expf((x - mx) / sp->temp);
        if (!isfinite(w)) w = 0.0f;
      }
      sp->pr[i] = w;
      z += w;
    }
    if (!(z > 0.0) || !isfinite(z)) return 0;
    double u = rng_u01(sp) * z;
    for (uint32_t i = 0; i < V; i++) {
      u -= sp->pr[i];
      if (u <= 0) return i;
    }
    return V - 1u;
  }
  if (K == V) {
    for (uint32_t i = 0; i < V; i++) sp->idx[n++] = i;
  } else { /* the K largest: a min-heap of K ids */
    for (uint32_t i = 0; i < V; i++) {
      if (n < K) {
        uint32_t c = n++;
        sp->idx[c] = i;
        while (c && lg[sp->idx[(c - 1) / 2]] > lg[sp->idx[c]]) {
          uint32_t pp = (c - 1) / 2, t = sp->idx[pp];
          sp->idx[pp] = sp->idx[c], sp->idx[c] = t, c = pp;
        }
      } else if (lg[i] > lg[sp->idx[0]]) {
        sp->idx[0] = i;
        uint32_t c = 0;
        for (;;) {
          uint32_t l = 2 * c + 1, r = l + 1, m = c;
          if (l < n && lg[sp->idx[l]] < lg[sp->idx[m]]) m = l;
          if (r < n && lg[sp->idx[r]] < lg[sp->idx[m]]) m = r;
          if (m == c) break;
          uint32_t t = sp->idx[m];
          sp->idx[m] = sp->idx[c], sp->idx[c] = t, c = m;
        }
      }
    }
  }
  if (K < V) { /* the candidates in the order of the GPU top K: value descending, equal values smaller index first */
    for (uint32_t i = 0; i < n; i++) sp->pr[i] = lg[sp->idx[i]];
    for (uint32_t i = 1; i < n; i++) { /* insertion sort: n <= K */
      const float v = sp->pr[i];
      const uint32_t id = sp->idx[i];
      uint32_t j = i;
      while (j && (sp->pr[j - 1] < v || (sp->pr[j - 1] == v && sp->idx[j - 1] > id))) sp->pr[j] = sp->pr[j - 1], sp->idx[j] = sp->idx[j - 1], j--;
      sp->pr[j] = v, sp->idx[j] = id;
    }
    return sample_cands(sp, sp->pr, sp->idx, n);
  }
  if (!n) return 0;
  float mx = -INFINITY;
  for (uint32_t i = 0; i < n; i++)
    if (isfinite(lg[sp->idx[i]]) && lg[sp->idx[i]] > mx) mx = lg[sp->idx[i]];
  if (!isfinite(mx)) {
    static unsigned warn_s;
    if (warn_s < 8u) {
      warn_s++;
      fprintf(stderr, "sample: invalid/NaN logits (V=%u, lg[0]=%g); falling back to 0\n", V, lg[0]);
    }
    return 0;
  }
  double z = 0;
  for (uint32_t i = 0; i < n; i++) {
    const float x = lg[sp->idx[i]];
    const float w = isfinite(x) ? expf((x - mx) / sp->temp) : 0.0f;
    sp->pr[i] = isfinite(w) ? w : 0.0f;
    z += sp->pr[i];
  }
  if (!(z > 0.0) || !isfinite(z)) return sp->idx[0] < V ? sp->idx[0] : 0;
  if (sp->top_p < 1.0f && n > 1) { /* sort by weight, keep the nucleus */
    for (uint32_t i = 1; i < n; i++) { /* insertion sort: n <= top_k (small) */
      float w = sp->pr[i];
      uint32_t id = sp->idx[i], j = i;
      while (j && sp->pr[j - 1] < w) sp->pr[j] = sp->pr[j - 1], sp->idx[j] = sp->idx[j - 1], j--;
      sp->pr[j] = w, sp->idx[j] = id;
    }
    double c = 0;
    uint32_t m = 0;
    while (m < n && c < sp->top_p * z) c += sp->pr[m++];
    if (!m) m = 1, c = sp->pr[0];
    n = m, z = c;
  }
  double u = rng_u01(sp) * z;
  for (uint32_t i = 0; i < n; i++) {
    u -= sp->pr[i];
    if (u <= 0) return sp->idx[i];
  }
  return sp->idx[n - 1];
}


static void ppl_add(ppl_acc *pa, const float *lg, uint32_t vocab, uint32_t next) {
  double mx = lg[0], z = 0;
  for (uint32_t i = 1; i < vocab; i++) mx = lg[i] > mx ? lg[i] : mx;
  for (uint32_t i = 0; i < vocab; i++) z += exp((double)lg[i] - mx);
  const double lse = mx + log(z);
  pa->nll += lse - lg[next];
  if (pa->ksave || pa->kbase) {
    if (!pa->klp) pa->klp = malloc((size_t)vocab * 4), pa->kref = malloc((size_t)vocab * 4);
    for (uint32_t i = 0; i < vocab; i++) pa->klp[i] = (float)((double)lg[i] - lse);
    if (pa->ksave && fwrite(pa->klp, 4, vocab, pa->ksave) != vocab) ie_die("--kld-save: write failed");
    if (pa->kbase) {
      if (fread(pa->kref, 4, vocab, pa->kbase) != vocab) ie_die("--kld: the reference file has fewer scored positions");
      double kl = 0;
      uint32_t ar = 0, at = 0;
      for (uint32_t i = 0; i < vocab; i++) {
        const double r = pa->kref[i];
        kl += exp(r) * (r - (double)pa->klp[i]);
        if (pa->kref[i] > pa->kref[ar]) ar = i;
        if (pa->klp[i] > pa->klp[at]) at = i;
      }
      pa->kl = realloc(pa->kl, (size_t)(pa->n + 1) * 4);
      pa->kl[pa->n] = (float)kl;
      pa->ktop += ar == at;
      pa->knll -= pa->kref[next];
    }
  }
  pa->n++;
}

/* IE_MTP_CAPTURE=FILE (graph built with mtp_r / mtp_a): after each prompt
 * chunk's MTP run, append one record per position q: u32 q, u32 token q,
 * then f32 rows: the MTP FFN input x (dim), the ffn_down input (ffn), the
 * model's h(q) (dim, the normed hidden state the MTP output at q stands in
 * for) and the MTP normed output g(q) (dim). Header: "MTPC", u32 dim, u32 ffn. */
FILE *ie_mtp_cap_file;
uint32_t *ie_spec_nk;

static void mtp_capture(backend *b, const graph *g, const uint32_t *toks, uint32_t T, uint32_t p0) {
  static FILE *fe; /* IE_MTP_CAPTURE */
  const uint32_t dim = g->bufs[g->mtp_g].stride / 4u, ffn = g->bufs[g->mtp_a].stride / 8u;
  if (!ie_mtp_cap_file && !fe) {
    fe = fopen(getenv("IE_MTP_CAPTURE"), "wb");
    if (!fe) ie_die("IE_MTP_CAPTURE: cannot write %s", getenv("IE_MTP_CAPTURE"));
  }
  FILE *f = ie_mtp_cap_file ? ie_mtp_cap_file : fe;
  if (ftell(f) == 0) {
    const uint32_t hd[3] = {0x4350544Du, dim, ffn};
    fwrite(hd, 4, 3, f);
  }
  float *r = malloc((size_t)T * dim * 4), *a = malloc((size_t)T * ffn * 8), *h = malloc((size_t)T * dim * 4),
        *o = malloc((size_t)T * dim * 4);
  b->read_rows(b, g->mtp_r, 0, T, r);
  b->read_rows(b, g->mtp_a, 0, T, a);
  b->read_rows(b, g->h_out, 0, T, h);
  b->read_rows(b, g->mtp_g, 0, T, o);
  for (uint32_t t = 0; t < T; t++) {
    const uint32_t id[2] = {p0 + t, toks[t]};
    fwrite(id, 4, 2, f);
    fwrite(r + (size_t)t * dim, 4, dim, f);
    fwrite(a + (size_t)t * ffn * 2, 4, ffn, f);
    fwrite(h + (size_t)t * dim, 4, dim, f);
    fwrite(o + (size_t)t * dim, 4, dim, f);
  }
  fflush(f);
  free(r), free(a), free(h), free(o);
}

/* IE_DF_CAPTURE: append the drafter's context input (cn) for T positions (below). */
static void df_capture(backend *b, const graph *g, const uint32_t *toks, uint32_t T, uint32_t p0);

/* The prompt in chunks of up to `chunk` tokens (GPU): one model run per
 * chunk reads the weights once for all its tokens. With mtp, the MTP head
 * also fills its KV cache for every prompt position q (the pair (h(q-1),
 * token q), h(-1) = 0, as llama.cpp). pa: perplexity of the prompt (needs
 * every row of logits). Returns the greedy token after the prompt; *hrow:
 * the row of g->h_out that holds h(n-1). */
static int bal_chunk_on(void) {
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("IE_BALANCED_CHUNK");
    v = (e && *e) ? (atoi(e) != 0) : 0; /* R50 (H49): rejected on odd_tail (+17 ms at chunk=1024); opt-in via IE_BALANCED_CHUNK=1 */
  }
  return v;
}
static uint32_t next_chunk_len(uint32_t rem, uint32_t chunk) {
  if (rem <= chunk) return rem;
  if (bal_chunk_on() && chunk >= 128u && rem <= chunk + 64u) {
    uint32_t t = ((rem + 1u) / 2u + 15u) & ~15u;
    if (t > chunk) t = chunk;
    if (t < 1u) t = 1u;
    return t;
  }
  return chunk;
}
uint32_t prefill(backend *b, const graph *g, uint32_t vocab, const uint32_t *toks, uint32_t n, uint32_t chunk,
                        int mtp, ppl_acc *pa, uint32_t *hrow) {
  uint32_t *a = malloc(chunk * 4), last = 0, T = 1;
  float *lg = pa && pa->on ? malloc((size_t)chunk * g->bufs[g->logits].stride) : NULL;
  b->all_logits = lg != NULL;
  mtp = mtp && g->mtp_h >= 0; /* a DFlash drafter alone: no MTP head to keep current */
  if (b->kv_io && n) b->kv_io(b, 0, n, 0, NULL); /* pre-map exact prompt KV capacity upfront in 1 idle VMM batch */
  if (mtp) b->copy_rows(b, g->mtp_h, 0, -1, 0, 1); /* h(-1) = 0 */
  for (uint32_t p0 = 0; p0 < n; p0 += T) {
    T = next_chunk_len(n - p0, chunk);
    const int is_last = p0 + T >= n;
    b->all_logits = lg && p0 + T > pa->first; /* the model head for every row only where positions are scored */
    b->run(b, 0, toks + p0, T, p0, 1, (is_last || b->all_logits) ? a : NULL, NULL);
    b->accept(b, T - 1);
    if (is_last) last = a[T - 1];
    if (g->df_feat >= 0 || g->ds_feat >= 0) b->run(b, 2, toks + p0, T, p0, 0, NULL, NULL); /* DFlash/DSpark context */
    if (g->df_cap >= 0) df_capture(b, g, toks + p0, T, p0);              /* IE_DF_CAPTURE */
    if (lg && p0 + T > pa->first) {
      b->read_rows(b, g->logits, 0, T, lg);
      const uint32_t stride = g->bufs[g->logits].stride / 4u;
      for (uint32_t t = 0; t < T; t++)
        if (p0 + t >= pa->first && p0 + t + 1 < n) ppl_add(pa, lg + (size_t)t * stride, vocab, toks[p0 + t + 1]);
    }
    if (mtp) { /* MTP positions p0 .. p0+T-1: h rows [h(p0-1) (row 0, from before), h(p0) .. h(p0+T-2)] */
      if (T > 1) b->copy_rows(b, g->mtp_h, 1, g->h_out, 0, T - 1);
      b->run(b, 4, toks + p0, T, p0, 1, NULL, NULL); /* async KV-only catch-up: skips out_draft + host sync */
      if (g->mtp_a >= 0 && (ie_mtp_cap_file || getenv("IE_MTP_CAPTURE"))) mtp_capture(b, g, toks + p0, T, p0);
      b->copy_rows(b, g->mtp_h, 0, g->h_out, T - 1, 1);
    }
  }
  free(lg), free(a);
  *hrow = T - 1;
  return last;
}

/* IE_DF_CAPTURE: the drafter's context input (enc_norm(fc(the target
 * layers' outputs))) for every prefill position, the rows a drafter trained
 * outside the engine needs. f32 rows: u32 position, u32 token, dim floats.
 * Header: u32 magic "IDPC", u32 dim, u32 reserved. */
static void df_capture(backend *b, const graph *g, const uint32_t *toks, uint32_t T, uint32_t p0) {
  static FILE *fe;
  const uint32_t dim = g->bufs[g->df_cap].stride / 4u;
  if (!fe) {
    fe = fopen(getenv("IE_DF_CAPTURE"), "wb");
    if (!fe) ie_die("IE_DF_CAPTURE: cannot write %s", getenv("IE_DF_CAPTURE"));
  }
  if (ftell(fe) == 0) {
    const uint32_t hd[3] = {0x43504449u, dim, 0};
    fwrite(hd, 4, 3, fe);
  }
  float *r = malloc((size_t)T * dim * 4);
  b->read_rows(b, g->df_cap, 0, T, r);
  for (uint32_t t = 0; t < T; t++) {
    const uint32_t id[2] = {p0 + t, toks[t]};
    fwrite(id, 4, 2, fe);
    fwrite(r + (size_t)t * dim, 4, dim, fe);
  }
  fflush(fe);
  free(r);
}

/* Prompt lookup: the drafts from the text so far. t[0 .. n) are the tokens
 * (t[n-1]: the token at the step's position). Finds the earlier place whose
 * tokens before it match the longest suffix of t (up to LOOKUP_NMAX tokens,
 * the latest place on a tie) and copies up to D tokens after that place to
 * d[1 ..]. Returns the number of drafts; *mlen: the match length (0: none). */
enum { LOOKUP_NMAX = 16 };

/* The most tokens of a verify run: 8 (7 MTP drafts), or 1 + IE_LOOKUP_MAX
 * (default 7, up to 15) with the prompt lookup on; the GPU backend keeps
 * this many linear attention states. Default 7: 8 states (1.2 GB for Saluki
 * 27B) fit in VRAM next to the desktop's; with 16 (2.5 GB) part of the
 * model moved to system memory and a 16-token verify ran at PCIe speed
 * (RX 9070 XT, 2026-10-29: repeated text 5.6 vs 9.4 ms/token). At least 1 +
 * the drafts of spec_set_draft (8 drafts: 9 tokens). */
/* The most tokens of a verify run: 16 by default (supporting up to 15 tree or
 * lookup drafts + root in one target verify pass; IE_GDN_RECORDS=1 stores one
 * GDN state + 15 compact replay records in ~107 MB instead of 16 full states).
 * IE_VERIFY_MAX=N (2 .. 16) overrides. */
static uint32_t spec_dmax;
void spec_set_draft(uint32_t D) { spec_dmax = D; }
uint32_t spec_verify_max(void) {
  const uint32_t dn = spec_dmax + 1u > 8u ? spec_dmax + 1u : 8u;
  const char *vm = getenv("IE_VERIFY_MAX");
  if (vm && atoi(vm) >= 2) {
    uint32_t v = (uint32_t)atoi(vm);
    if (v > 16u) v = 16u;
    return v > dn ? v : dn;
  }
  const char *mt = getenv("IE_MTP_TREE");
  if (!(mt && mt[0] == '0')) return 16u > dn ? 16u : dn;
  const char *e = getenv("IE_LOOKUP");
  if (e && e[0] == '0') return dn;
  e = getenv("IE_LOOKUP_MAX");
  uint32_t n = e ? (uint32_t)atoi(e) : 15u;
  if (n > 15u) n = 15u;
  return n + 1u > dn ? n + 1u : dn;
}

/* The positions past the generated tokens that a verify run may use. */
uint32_t spec_room(uint32_t draft) {
  const uint32_t r = spec_verify_max() - 1u;
  return draft ? (r > draft ? r : draft) : 0u;
}
/* R31: Compact O(1) online Suffix Automaton (SAM) for prompt/history lookup with strict endpos <= n - 2 invariant. */
typedef struct {
  int32_t link;
  uint32_t len;
  uint32_t first_pos;
  uint32_t last_pos;
  int32_t head_edge;
} sam_state;
typedef struct {
  uint32_t tok;
  int32_t to;
  int32_t next;
} sam_edge;
typedef struct {
  uint64_t key;
  int32_t edge;
} sam_hslot;
typedef struct {
  sam_state *st;
  sam_edge *ed;
  sam_hslot *ht;
  uint32_t n_st, cap_st;
  uint32_t n_ed, cap_ed;
  uint32_t ht_mask;
  int32_t last;
  uint32_t built;
  uint32_t lp_done;
} ie_sam;

static inline uint64_t sam_key(int32_t st, uint32_t tok) {
  return ((uint64_t)(uint32_t)st << 24) | (uint64_t)(tok & 0xFFFFFFu);
}
static inline uint32_t sam_hash(uint64_t k, uint32_t mask) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  return (uint32_t)k & mask;
}
static int sam_init(ie_sam *S, uint32_t max_n) {
  memset(S, 0, sizeof *S);
  if (max_n < 64u) max_n = 64u;
  S->cap_st = 2u * max_n + 8u;
  S->cap_ed = 3u * max_n + 16u;
  uint32_t hsz = 64u;
  const uint64_t need = (uint64_t)S->cap_ed * 2u + 16u;
  while ((uint64_t)hsz < need && hsz < (1u << 24)) hsz <<= 1u;
  S->st = (sam_state *)malloc((size_t)S->cap_st * sizeof(sam_state));
  S->ed = (sam_edge *)malloc((size_t)S->cap_ed * sizeof(sam_edge));
  S->ht = (sam_hslot *)malloc((size_t)hsz * sizeof(sam_hslot));
  if (!S->st || !S->ed || !S->ht) {
    free(S->st), free(S->ed), free(S->ht);
    memset(S, 0, sizeof *S);
    return 0;
  }
  memset(S->ht, 0xFF, (size_t)hsz * sizeof(sam_hslot));
  S->ht_mask = hsz - 1u;
  S->n_st = 1u;
  S->st[0].link = -1;
  S->st[0].len = 0u;
  S->st[0].first_pos = 0u;
  S->st[0].last_pos = 0u;
  S->st[0].head_edge = -1;
  S->last = 0;
  return 1;
}
static void sam_free(ie_sam *S) {
  free(S->st), free(S->ed), free(S->ht);
  memset(S, 0, sizeof *S);
}
static inline int32_t sam_find_edge(const ie_sam *S, int32_t st, uint32_t tok) {
  const uint64_t k = sam_key(st, tok);
  uint32_t h = sam_hash(k, S->ht_mask);
  for (;;) {
    const uint64_t ck = S->ht[h].key;
    if (ck == 0xFFFFFFFFFFFFFFFFULL) return -1;
    if (ck == k) return S->ht[h].edge;
    h = (h + 1u) & S->ht_mask;
  }
}
static inline void sam_add_edge(ie_sam *S, int32_t st, uint32_t tok, int32_t to) {
  if (S->n_ed >= S->cap_ed) return;
  const int32_t ei = (int32_t)S->n_ed++;
  S->ed[ei].tok = tok;
  S->ed[ei].to = to;
  S->ed[ei].next = S->st[st].head_edge;
  S->st[st].head_edge = ei;
  const uint64_t k = sam_key(st, tok);
  uint32_t h = sam_hash(k, S->ht_mask);
  while (S->ht[h].key != 0xFFFFFFFFFFFFFFFFULL) h = (h + 1u) & S->ht_mask;
  S->ht[h].key = k;
  S->ht[h].edge = ei;
}
static void sam_extend(ie_sam *S, uint32_t tok, uint32_t pos) {
  if (!S->st || S->n_st + 2u > S->cap_st || S->n_ed + 8u > S->cap_ed) return;
  const int32_t cur = (int32_t)S->n_st++;
  S->st[cur].len = S->st[S->last].len + 1u;
  S->st[cur].first_pos = pos;
  S->st[cur].last_pos = pos;
  S->st[cur].head_edge = -1;
  int32_t p = S->last;
  while (p >= 0) {
    const int32_t ei = sam_find_edge(S, p, tok);
    if (ei >= 0) break;
    sam_add_edge(S, p, tok, cur);
    p = S->st[p].link;
  }
  if (p < 0) {
    S->st[cur].link = 0;
  } else {
    const int32_t q = S->ed[sam_find_edge(S, p, tok)].to;
    if (S->st[p].len + 1u == S->st[q].len) {
      S->st[cur].link = q;
    } else {
      const int32_t cl = (int32_t)S->n_st++;
      S->st[cl].len = S->st[p].len + 1u;
      S->st[cl].link = S->st[q].link;
      S->st[cl].first_pos = S->st[q].first_pos;
      S->st[cl].last_pos = S->st[q].last_pos;
      S->st[cl].head_edge = -1;
      for (int32_t e = S->st[q].head_edge; e >= 0; e = S->ed[e].next) {
        if (S->n_ed < S->cap_ed) sam_add_edge(S, cl, S->ed[e].tok, S->ed[e].to);
      }
      while (p >= 0) {
        const int32_t ei = sam_find_edge(S, p, tok);
        if (ei < 0 || S->ed[ei].to != q) break;
        S->ed[ei].to = cl;
        p = S->st[p].link;
      }
      S->st[q].link = cl;
      S->st[cur].link = cl;
    }
  }
  S->last = cur;
  S->built = pos + 1u;
}
static inline void sam_mark_lastpos(ie_sam *S, int32_t st_id, uint32_t pos) {
  int32_t v = S->st[st_id].link;
  for (int h = 0; h < 6 && v > 0; h++, v = S->st[v].link) S->st[v].last_pos = pos;
}
static uint32_t sam_query(ie_sam *S, const uint32_t *out, uint32_t n, uint32_t cap, uint32_t *ld, uint32_t *mlen) {
  *mlen = 0u;
  if (!S->st || n < 2u || cap == 0u) return 0u;
  if (S->built > 0u && S->lp_done < S->built && S->built < n) {
    sam_mark_lastpos(S, S->last, S->built - 1u);
    S->lp_done = S->built;
  }
  for (uint32_t p = S->built; p + 1u < n; p++) {
    sam_extend(S, out[p], p);
    sam_mark_lastpos(S, S->last, p);
    S->lp_done = p + 1u;
  }
  if (S->built < n) sam_extend(S, out[n - 1u], n - 1u);
  const int32_t v = S->st[S->last].link;
  if (v <= 0) return 0u;
  const uint32_t ml = S->st[v].len;
  if (ml == 0u) return 0u;
  uint32_t bi = S->st[v].last_pos;
  if (bi >= n - 1u || (n - 1u - bi < 4u && S->st[v].first_pos + 4u <= n - 1u)) bi = S->st[v].first_pos;
  if (bi >= n - 1u || out[bi] != out[n - 1u]) return 0u;
  const uint32_t avail = n - 1u - bi;
  const uint32_t c = avail < cap ? avail : cap;
  for (uint32_t j = 1u; j <= c; j++) ld[j] = out[bi + j];
  *mlen = ml;
  return c;
}

static uint32_t lookup_draft(const uint32_t *t, uint32_t n, uint32_t D, uint32_t *d, uint32_t *mlen) {
  uint32_t best = 0, bi = 0;
  *mlen = 0;
  if (n < 2) return 0;
  const uint32_t last = t[n - 1];
  for (uint32_t i = n - 1; i-- > 0;) {
    if (t[i] != last) continue;
    uint32_t L = 1;
    while (L < LOOKUP_NMAX && L <= i && t[i - L] == t[n - 1 - L]) L++;
    if (L > best) {
      best = L, bi = i;
      if (best == LOOKUP_NMAX) break;
    }
  }
  if (!best) return 0;
  uint32_t c = n - 1 - bi; /* tokens after the place */
  if (c > D) c = D;
  for (uint32_t j = 0; j < c; j++) d[j + 1] = t[bi + 1 + j];
  *mlen = best;
  return c;
}

/* DFlash: drafts for the block [t(P), mask...] at P (the drafter context
 * is up to date): the block run, the top-k candidates of rows 1 .. n, and
 * the selector's path from the anchor t(P) (greedy). Returns n. */
static uint32_t df_draft(backend *b, const model *m, const graph *g, sampler *sp, uint32_t tP, uint32_t P, uint32_t n, uint32_t *d,
                         float pmin) {
  const dflash *df = m->df;
  const uint32_t K = df->top_k, R = df->rank, B = n + 1;
  uint32_t bt[17];
  bt[0] = tP;
  for (uint32_t j = 1; j < B; j++) bt[j] = df->mask;
  float *uv = sp->cv;
  uint32_t *ci = sp->ci;
  static int nosel_env = -1; /* IE_DFLASH_SEL=0: the top candidate of each position (no selector) */
  if (nosel_env < 0) nosel_env = getenv("IE_DFLASH_SEL") && getenv("IE_DFLASH_SEL")[0] == '0';
  const int nosel = nosel_env || !df->has_sel; /* DFlash v1: no selector */
  const uint32_t ss = nosel ? 0u : g->bufs[g->df_sel].stride / 4u;
  /* IE_DFLASH_PASSES=N (default 1): run the block N times, each pass seeded with the previous
   * pass's chain (coarse first pass, then refinement against it), the last pass's top k the
   * drafts. IE_DFLASH_FILL=k: only the chain's first k rows are seeded (default: all n).
   * IE_DFLASH_STRIDE=s (default 1): seed the rows j with (j - 1) % s == 0, a skeleton at every
   * s-th position (s = 2: half the block known, the rest interpolated). */
  static int npass = -1, nfill = -1, nstride = -1;
  if (npass < 0) {
    const char *e = getenv("IE_DFLASH_PASSES");
    const char *f = getenv("IE_DFLASH_FILL");
    const char *s = getenv("IE_DFLASH_STRIDE");
    npass = e ? atoi(e) : 1;
    if (npass < 1) npass = 1;
    if (npass > 4) npass = 4;
    nfill = f ? atoi(f) : (int)n;
    if (nfill < 0) nfill = 0;
    if (nfill > (int)n) nfill = (int)n;
    nstride = s ? atoi(s) : 1;
    if (nstride < 1) nstride = 1;
  }
  float *hp = nosel ? NULL : malloc((size_t)n * ss * 4), *pc = nosel ? NULL : malloc((size_t)R * 4),
        *sc = nosel ? NULL : malloc((size_t)R * 4);
  for (int pass = 0; pass < npass; pass++) {
    b->run(b, 3, bt, B, P, 0, NULL, NULL);
    if (!b->topk(b, g->df_logits, 1, n, m->df_head.rows ? m->df_head.rows : m->vocab, K, uv, ci))
      ie_die("dflash: top k on the GPU failed");
    if (m->df_head.rows) /* draft vocabulary rows -> token ids */
      for (uint32_t j = 0; j < n * K; j++) ci[j] = model_draft_tok(m, ci[j]);
    if (!nosel) b->read_rows(b, g->df_sel, 1, n, hp);
    uint32_t prev = tP;
    for (uint32_t j = 0; j < n && nosel; j++) {
      uint32_t bi = 0;
      for (uint32_t c = 1; c < K; c++) if (uv[j * K + c] > uv[j * K + bi]) bi = c;
      d[j + 1] = ci[j * K + bi];
    }
    for (uint32_t j = 0; j < n && !nosel; j++) {
      model_df_code(m, 0, prev, pc);
      const float *h = hp + (size_t)j * ss;
      for (uint32_t r = 0; r < R; r++) pc[r] *= h[r];
      float best = -INFINITY;
      uint32_t bi = 0;
      for (uint32_t c = 0; c < K; c++) {
        const uint32_t tok = ci[j * K + c];
        model_df_code(m, 1, tok, sc);
        float s = 0.0f;
        for (uint32_t r = 0; r < R; r++) s += pc[r] * sc[r];
        s += uv[j * K + c];
        if (s > best) best = s, bi = tok;
      }
      d[j + 1] = prev = bi;
    }
    if (pass + 1 < npass) /* the next pass reads the chain: the coarse plan, then the refinement */
      for (uint32_t j = 1; j <= (uint32_t)nfill; j++)
        if ((int)((j - 1) % (uint32_t)nstride) == 0) bt[j] = d[j];
  }
  free(hp), free(pc), free(sc);
  /* Keep the drafts while the drafter's probability of each (softmax over
   * its top K logits) is at least pmin, as the MTP drafts. */
  uint32_t nd = 0;
  for (uint32_t j = 0; j < n; j++) {
    const float *u = uv + j * K;
    float mx = u[0], z = 0.0f, pd = 0.0f;
    for (uint32_t c = 1; c < K; c++) mx = u[c] > mx ? u[c] : mx;
    for (uint32_t c = 0; c < K; c++) {
      const float e = expf(u[c] - mx);
      z += e;
      if (ci[j * K + c] == d[j + 1]) pd = e;
    }
    if (pd < pmin * z) break;
    nd = j + 1;
  }
  return nd;
}

/* DDTree (arXiv 2604.12989) from one DFlash block: the most likely prefixes of the block's per-row candidates (row
 * j: the top K of position P + j, softmax over them), best first by the product of their probabilities, while that
 * product is at least pmin, at most B nodes; node 0 the root tP. Laid out depth first, the most likely child first
 * (so the usual path, the top candidate of each row, is nodes 1, 2, ..: no rows to move at accept). Returns the
 * number of nodes (1: none); tok / par get them. */
static uint32_t dd_tree(const float *uv, const uint32_t *ci, uint32_t K, uint32_t rows, uint32_t B, float pmin, uint32_t tP,
                        uint32_t *tok, uint8_t *par, const float *vc, float td) {
  float pr[16][64];
  for (uint32_t j = 0; j < rows && j < 16u; j++) { /* softmax over the row's top K (values descending) */
    const float *u = uv + (size_t)j * K;
    float z = 0.0f;
    for (uint32_t c = 0; c < K; c++) z += expf(u[c] - u[0]);
    for (uint32_t c = 0; c < K; c++) pr[j][c] = expf(u[c] - u[0]) / z;
  }
  /* best first: candidates (prob, parent node, depth, rank); nodes in pop order */
  typedef struct { float p; uint32_t par, dep, rank; } cand;
  cand cq[64];
  uint32_t nc = 0, n = 1, ndep[16], nrank[16], npar[16];
  float np[16];
  np[0] = 1.0f, ndep[0] = 0, nrank[0] = 0, npar[0] = 0;
  if (rows) cq[nc++] = (cand){pr[0][0], 0, 1, 0};
  while (nc && n <= B && n < 16u) {
    uint32_t bi = 0;
    for (uint32_t i = 1; i < nc; i++) if (cq[i].p > cq[bi].p) bi = i;
    const cand c = cq[bi];
    cq[bi] = cq[--nc];
    if (c.p < pmin) break;
    np[n] = c.p, npar[n] = c.par, ndep[n] = c.dep, nrank[n] = c.rank;
    if (c.rank + 1u < K && nc < 64u) cq[nc++] = (cand){np[c.par] * pr[c.dep - 1u][c.rank + 1u], c.par, c.dep, c.rank + 1u};
    if (c.dep < rows && c.dep < 15u && nc < 64u) cq[nc++] = (cand){c.p * pr[c.dep][0], n, c.dep + 1u, 0};
    n++;
  }
  /* the cost cut (vc: the verify time of 1 .. 16 tokens, td: the draft time, both relative): the nodes come in order
   * of falling path probability, so each first n is a tree; keep the n with the most expected tokens per time,
   * (1 + the sum of the path probabilities) / (td + vc[n - 1]) */
  if (vc && n > 1u) {
    float e = 0.0f, best = 1.0f / (td + vc[0]);
    uint32_t bn = 1;
    for (uint32_t i = 1; i < n; i++) {
      e += np[i];
      const float r = (1.0f + e) / (td + vc[i]);
      if (r > best) best = r, bn = i + 1u;
    }
    n = bn;
  }
  /* depth first, children by rank (= probability) */
  uint32_t order[16], no = 0, stk[16], ns = 0, idx[16];
  stk[ns++] = 0;
  while (ns) {
    const uint32_t x = stk[--ns];
    idx[x] = no, order[no++] = x;
    for (uint32_t r = 16; r-- > 0;) /* push the children, the largest rank first: rank 0 on top */
      for (uint32_t y = 1; y < n; y++)
        if (npar[y] == x && nrank[y] == r && ns < 16u) stk[ns++] = y;
  }
  for (uint32_t i = 0; i < no; i++) {
    const uint32_t x = order[i];
    tok[i] = x ? ci[(size_t)(ndep[x] - 1u) * K + nrank[x]] : tP;
    par[i] = (uint8_t)(x ? idx[npar[x]] : 0u);
  }
  return no;
}

/* ---- Batched Probability-Guided MTP Tree Speculation + Free Suffix Lookup ----
 *
 * Builds a multi-branch draft tree of up to W drafts (IE_MTP_TREE_W, default 12,
 * max 15 -> T <= 16 target verify tokens) in ~2.5-3 MTP passes instead of 5-7
 * serial 1-token passes:
 *   - Catch-up pass (mandatory): updates blk.64 KV for positions P-k .. P and
 *     fetches root top-K (K = 8, fitting ie_topk_max + ie_topk_sel fast path
 *     for 40,960-row out_draft). Selects up to 3 root branches (c0, c1, c2).
 *   - Batched MTP Pass 1 (T = 1..3 in ONE blk.64 pass): expands (c0, c1[, c2])
 *     together via graph.mtp_tr -> picks top depth-2 children (c00, c01, c10, ..).
 *   - Batched MTP Pass 2 (T = 1..3 in ONE blk.64 pass): expands depth-2 frontier
 *     -> picks top depth-3 children (c000, c001, c010, ..).
 *   - Gated MTP Pass 3 (T = 1..2 when path prob >= 0.25) [+ optional T = 1 Pass 4
 *     when path prob >= 0.40]: extends deepest high-confidence branch to c0000[0].
 *   - Free 0-ms CPU suffix-lookup graft (lookup_draft): grafts 2..4 tokens from
 *     the longest prompt/answer suffix match into remaining node budget up to W,
 *     deduplicating shared prefix edges against already-drafted MTP nodes. */
enum { MTP_TK = 8, QDIST_MAX = 32 };

typedef struct {
  uint32_t tok;
  uint8_t par, dep, slot, stoch, from_lk;
  uint8_t path[16]; /* blk.64 KV slot offsets relative to P along root .. self */
  float p;
} mtp_tnode;

typedef struct {
  uint8_t n; /* > 0 when this node's primary child was sampled from q[0..n-1] */
  uint32_t id[QDIST_MAX];
  double q[QDIST_MAX];
} mtp_qdist;

static double drng_u01(uint64_t *r) {
  *r ^= *r >> 12, *r ^= *r << 25, *r ^= *r >> 27;
  return (double)((*r * 2685821657736338717ull) >> 11) / 9007199254740992.0;
}

static void mtp_qdist_make(const sampler *sp, uint32_t V, int rej_on, const float *uv, const uint32_t *ci,
                           const float *up_in, float *ep_out, mtp_qdist *qd) {
  qd->n = 0;
  for (uint32_t c = 0; c < MTP_TK; c++) ep_out[c] = up_in[c];
  if (!rej_on || !sp || sp->temp <= 0.0f) return;
  const uint32_t lim = sp->vocab ? sp->vocab : V;
  const uint32_t kcap = sample_k(sp, V);
  uint32_t id_t[MTP_TK], nq = 0;
  float v_t[MTP_TK];
  for (uint32_t c = 0; c < MTP_TK && nq < kcap; c++) {
    if (ci[c] >= lim || !isfinite(uv[c]) || !(up_in[c] > 0.0f)) continue;
    int dup = 0;
    for (uint32_t j = 0; j < nq; j++)
      if (id_t[j] == ci[c]) { dup = 1; break; }
    if (dup) continue;
    id_t[nq] = ci[c], v_t[nq] = uv[c], nq++;
  }
  if (!nq) return;
  double w[MTP_TK], z8 = 0.0;
  for (uint32_t c = 0; c < nq; c++) {
    const double e = exp((double)(v_t[c] - v_t[0]) / (double)sp->temp);
    w[c] = isfinite(e) ? e : 0.0;
    z8 += w[c];
  }
  if (!(z8 > 0.0) || !isfinite(z8)) return;
  double mass8 = 0.0;
  for (uint32_t c = 0; c < MTP_TK; c++)
    if (up_in[c] > 0.0f && isfinite(up_in[c])) mass8 += (double)up_in[c];
  double z_tail = 0.0;
  if (mass8 < 0.999999 && up_in[0] > 1e-12f) {
    const double tail1 = (1.0 - mass8) / (double)up_in[0];
    const double v_edge = (double)(v_t[nq - 1] - v_t[0]);
    z_tail = sp->temp <= 1.0f ? tail1 * exp(v_edge * (1.0 / (double)sp->temp - 1.0)) : tail1;
    if (!isfinite(z_tail) || z_tail < 0.0) z_tail = 0.0;
  }
  const double inv_tot = 1.0 / (z8 + z_tail);
  for (uint32_t c = 0; c < MTP_TK; c++) ep_out[c] = 0.0f;
  for (uint32_t c = 0; c < nq; c++) ep_out[c] = (float)(w[c] * inv_tot);

  uint32_t mq = nq;
  double zq = z8;
  if (sp->top_p < 1.0f && nq > 1u) {
    double cum = 0.0;
    uint32_t m = 0;
    while (m < nq && cum < (double)sp->top_p * z8) cum += w[m++];
    if (!m) m = 1u, cum = w[0];
    mq = m, zq = cum;
  }
  if (!(zq > 0.0) || !isfinite(zq)) return;
  qd->n = (uint8_t)mq;
  const double inv_q = 1.0 / zq;
  for (uint32_t c = 0; c < mq; c++) qd->id[c] = id_t[c], qd->q[c] = w[c] * inv_q;
}

static uint32_t mtp_qdist_draw(const mtp_qdist *qd, uint64_t *drng) {
  double u = drng_u01(drng);
  uint32_t last = 0;
  for (uint32_t c = 0; c < qd->n; c++) {
    if (qd->q[c] > 0.0) last = c;
    u -= qd->q[c];
    if (u <= 0.0 && qd->q[c] > 0.0) return c;
  }
  return last;
}

static void mtp_probs(const float *uv, uint32_t K, float *pr) {
  if (!K || !isfinite(uv[0])) {
    for (uint32_t c = 0; c < K; c++) pr[c] = 0.0f;
    return;
  }
  float z = 0.0f;
  for (uint32_t c = 0; c < K; c++) {
    const float e = isfinite(uv[c]) ? expf(uv[c] - uv[0]) : 0.0f;
    pr[c] = e, z += e;
  }
  const float inv = z > 0.0f ? 1.0f / z : 0.0f;
  for (uint32_t c = 0; c < K; c++) pr[c] *= inv;
}

static void mtp_run_topk(backend *b, const model *m, const graph *g, const uint32_t *toks, uint32_t T, uint32_t pos,
                         int last_only, uint32_t r0, uint32_t nrows, float *uv, uint32_t *ci, float *up) {
  struct ie_mtp_tree *mtr = graph_mtp_tr(g);
  const uint32_t nv = m->out_draft.rows ? m->out_draft.rows : m->vocab;
  const uint32_t K = nv < MTP_TK ? nv : MTP_TK;
  uint32_t da[16];
  float dp[16], lse[16];
  for (uint32_t r = 0; r < 16u; r++) lse[r] = NAN;
  mtr->topk_k = K, mtr->topk_row = r0, mtr->topk_n = nrows;
  mtr->topk_v = uv, mtr->topk_i = ci, mtr->topk_lse = lse, mtr->topk_done = 0;
  b->run(b, 1, toks, T, pos, last_only, da, dp);
  const int done = mtr->topk_done;
  mtr->on = 0, mtr->topk_k = 0, mtr->topk_v = NULL, mtr->topk_i = NULL, mtr->topk_lse = NULL, mtr->topk_done = 0;
  if (!done && !b->topk(b, g->mtp_logits, r0, nrows, nv, K, uv, ci))
    ie_die("mtp tree: top k on the GPU failed");
  if (m->out_draft.rows)
    for (uint32_t j = 0; j < nrows * K; j++) ci[j] = model_draft_tok(m, ci[j]);
  for (uint32_t r = nrows; r-- > 0;) {
    float rp[MTP_TK];
    uint32_t ri[MTP_TK];
    if (done && isfinite(lse[r])) {
      for (uint32_t c = 0; c < K; c++) {
        const float v = uv[(size_t)r * K + c];
        rp[c] = isfinite(v) ? expf(v - lse[r]) : 0.0f;
      }
    } else {
      mtp_probs(uv + (size_t)r * K, K, rp);
    }
    for (uint32_t c = 0; c < K; c++) ri[c] = ci[(size_t)r * K + c];
    for (uint32_t c = K; c < MTP_TK; c++) rp[c] = 0.0f, ri[c] = 0u;
    memcpy(up + (size_t)r * MTP_TK, rp, sizeof rp);
    memcpy(ci + (size_t)r * MTP_TK, ri, sizeof ri);
  }
}

static void mtp_run_frontier(backend *b, const model *m, const graph *g, uint32_t P, mtp_tnode *tr, const uint8_t *fidx,
                             uint32_t nf, uint32_t *slot_next, uint8_t *pgrow, float *uv, uint32_t *ci, float *up) {
  struct ie_mtp_tree *mtr = graph_mtp_tr(g);
  uint8_t ngrow[16] = {0};
  int contig = nf == 1u && *slot_next == tr[fidx[0]].dep;
  if (contig) {
    const uint8_t u = fidx[0], pu = tr[u].par;
    for (uint32_t e = 0; e <= tr[pu].dep; e++)
      if (tr[pu].path[e] != e) contig = 0;
  }
  if (contig) {
    const uint8_t u = fidx[0], pu = tr[u].par;
    tr[u].slot = (uint8_t)(*slot_next)++;
    memcpy(tr[u].path, tr[pu].path, tr[pu].dep + 1u);
    tr[u].path[tr[u].dep] = tr[u].slot;
    b->copy_rows(b, g->mtp_h, 0, g->mtp_g, pgrow[pu], 1);
    mtr->on = 0;
    mtp_run_topk(b, m, g, &tr[u].tok, 1u, P + tr[u].dep, 1, 0u, 1u, uv, ci, up);
    ngrow[u] = 0;
    memcpy(pgrow, ngrow, sizeof ngrow);
    return;
  }
  uint32_t ftok[16];
  for (uint32_t f = 0; f < nf; f++) {
    const uint8_t u = fidx[f], pu = tr[u].par;
    tr[u].slot = (uint8_t)(*slot_next + f);
    memcpy(tr[u].path, tr[pu].path, tr[pu].dep + 1u);
    tr[u].path[tr[u].dep] = tr[u].slot;
    mtr->dep[f] = tr[u].dep;
    memcpy(mtr->path[f], tr[u].path, tr[u].dep + 1u);
    ftok[f] = tr[u].tok;
    b->copy_rows(b, g->mtp_h, f, g->mtp_g, pgrow[pu], 1);
    ngrow[u] = (uint8_t)f;
  }
  mtr->on = 1, mtr->slot_base = *slot_next;
  *slot_next += nf;
  mtp_run_topk(b, m, g, ftok, nf, P, 0, 0u, nf, uv, ci, up);
  memcpy(pgrow, ngrow, sizeof ngrow);
}

typedef struct {
  uint32_t tok;
  uint8_t par;
  float p;
} mtp_cand;

static void mtp_sort_cands(mtp_cand *c, uint32_t n) {
  for (uint32_t i = 1; i < n; i++) {
    const mtp_cand x = c[i];
    uint32_t j = i;
    while (j && c[j - 1].p < x.p) c[j] = c[j - 1], j--;
    c[j] = x;
  }
}

static int mtp_ch_gt(const mtp_tnode *tr, uint32_t a, uint32_t b) {
  if (tr[a].stoch != tr[b].stoch) return tr[a].stoch > tr[b].stoch;
  return tr[a].p > tr[b].p;
}

static void mtp_draft_tree(backend *b, const model *m, const graph *g, const sampler *sp, uint32_t V, int rej_on,
                           uint64_t *drng, const uint32_t *out, uint32_t n, uint32_t P, uint32_t k, uint32_t hrow,
                           uint32_t tP, uint32_t D, uint32_t max_tn, int lk_on, uint32_t lk_min, uint32_t lk_long,
                           uint32_t lk_agree, uint32_t lk_max, float pmin, float p_branch, float p_child, float p_deep,
                           float *uv, uint32_t *ci, uint32_t *d, uint32_t *nd_out, uint32_t *tk, uint8_t *tpar,
                           uint32_t *tree_n_out, mtp_qdist *q_out, uint32_t *lk_np) {
  static int max_d_env = -1;
  static float p_unc = -1.0f, p_par1 = -1.0f, p_t9 = -1.0f, p_t16 = -1.0f;
  if (max_d_env < 0) max_d_env = getenv("IE_MTP_MAX_D") ? atoi(getenv("IE_MTP_MAX_D")) : 0;
  if (p_unc < 0.0f) p_unc = getenv("IE_MTP_UNC_P") ? (float)atof(getenv("IE_MTP_UNC_P")) : 0.06f;
  if (p_par1 < 0.0f) p_par1 = getenv("IE_MTP_PAR1_P") ? (float)atof(getenv("IE_MTP_PAR1_P")) : 0.18f;
  if (p_t9 < 0.0f) p_t9 = getenv("IE_MTP_TIER9_P") ? (float)atof(getenv("IE_MTP_TIER9_P")) : 0.10f;
  if (p_t16 < 0.0f) p_t16 = getenv("IE_MTP_TIER16_P") ? (float)atof(getenv("IE_MTP_TIER16_P")) : 0.42f;

  static int mp_env = -1;
  if (mp_env < 0) mp_env = !(getenv("IE_ATTN_LONG_MP") && getenv("IE_ATTN_LONG_MP")[0] == '0');
  const int mp_q8 = mp_env && m->hd == 256u && (m->n_kv ? m->n_head / m->n_kv : 6u) == 6u &&
                    b->kv_rowb == (uint64_t)m->n_kvl * m->n_kv * (m->hd + (m->hd >> 4)) * 2u;
  /* When ie_attn_long_mp is active (hd=256, grp=6, Q8 KV), nt <= 8 runs _tr8 + 1-pass ie_attn_long_mp
   * whereas nt >= 9 jumps to _tr16 + 4..6-pass ie_attn_long; keep pure MTP trees inside nt <= 8. */
  const uint32_t mtp_cap = (mp_q8 && max_tn > 8u && !getenv("IE_MTP_TIER16_P") && max_d_env <= 0) ? 8u : max_tn;

  const int samp = (sp && sp->temp > 0.0f);
  const float p_keep = samp ? pmin * (P >= 8192u ? 0.68f : 0.80f) : pmin;
  const float p_deep_eff = samp ? (P >= 8192u ? 0.24f : 0.32f) : p_deep;
  /* At short context (< 8K) nf=1 avoids WDDM H2D table copies unless root is genuinely split;
   * at long context (>= 8K), verify KV cost dominates so nf=2 2-branch expansion wins */
  const float p_par_eff = getenv("IE_MTP_PAR1_P") ? p_par1
                                                  : (samp ? (P >= 8192u ? 0.16f : 2.0f) : (P >= 8192u ? p_par1 : 0.28f));
  const float p_br_eff = samp ? (p_branch < 0.05f ? p_branch : 0.05f) : p_branch;
  const float p_ch_eff = samp ? (p_child < 0.04f ? p_child : 0.04f) : p_child;

  float up[16 * MTP_TK], ep0_arr[MTP_TK], ep1_arr[MTP_TK];
  mtp_qdist tr_q[16];
  memset(tr_q, 0, sizeof tr_q);
  memset(q_out, 0, 16u * sizeof *q_out);

  struct ie_mtp_tree *mtr = graph_mtp_tr(g);
  mtr->on = 0;
  b->copy_rows(b, g->mtp_h, 0, g->h_out, hrow, k + 1u);
  mtp_run_topk(b, m, g, &out[P - k], k + 1u, P - k, 0, k, 1u, uv, ci, up);

  mtp_qdist qd0;
  mtp_qdist_make(sp, V, rej_on, uv, ci, up, ep0_arr, &qd0);
  tr_q[0] = qd0;

  uint32_t ml = 0, ld[16], nl = lk_on ? lookup_draft(out, n, lk_max, ld, &ml) : 0u;
  if (nl && ml < lk_min) nl = 0u;

  mtp_tnode tr[16];
  memset(tr, 0, sizeof tr);
  tr[0].tok = tP, tr[0].par = 0, tr[0].dep = 0, tr[0].slot = 0, tr[0].path[0] = 0, tr[0].p = 1.0f;
  uint32_t nt = 1u, slot_next = 1u;
  uint8_t pgrow[16] = {0};
  pgrow[0] = (uint8_t)k;

  /* 1. Uncertain root: run ZERO extra MTP passes (0 ms draft cost),
   *    verifying up to 3 deterministic top root candidates inside ultra-fast _tr4 (nt <= 4). */
  const float r_u0 = P >= 8192u ? 0.60f : 0.78f, r_u1 = P >= 8192u ? 1.05f : 1.30f;
  const int root_unc = samp ? (ep0_arr[0] < pmin * r_u0 && (ep0_arr[0] + ep0_arr[1]) < pmin * r_u1)
                            : (ep0_arr[0] < pmin);
  if (root_unc) {
    const uint32_t cap4 = max_tn < 4u ? max_tn : 4u;
    const uint32_t nr0 = qd0.n > 0u ? qd0.n : MTP_TK;
    for (uint32_t r = 0; r < 3u && r < nr0 && nt < cap4; r++) {
      const uint32_t rtok = qd0.n > 0u ? qd0.id[r] : ci[r];
      if (ep0_arr[r] < p_unc) break;
      const uint8_t idx = (uint8_t)nt++;
      tr[idx].tok = rtok, tr[idx].par = 0, tr[idx].dep = 1, tr[idx].p = ep0_arr[r];
    }
    if (nl >= 1u && ml >= 3u && nt < cap4) {
      int dup = 0;
      for (uint32_t y = 1; y < nt; y++)
        if (tr[y].par == 0u && tr[y].tok == ld[1]) dup = 1;
      if (!dup) {
        const uint8_t idx = (uint8_t)nt++;
        tr[idx].tok = ld[1], tr[idx].par = 0, tr[idx].dep = 1, tr[idx].p = 0.05f, tr[idx].from_lk = 1;
        (*lk_np)++;
      }
    }
    goto emit_tree;
  }

  /* 2. Deterministic Top-K multi-branch frontier (nf <= 2) + free sibling leaves + lookup rules (a) & (b).
   *    Keeping ci[0] deterministic on the primary trunk prevents random draws from truncating the trunk, while
   *    spec_rej_step's without-replacement conditional verification preserves the exact target distribution and
   *    accepts any drafted child at a node with full sum_r p_target(child_r) probability. */
  uint32_t max_d = max_d_env > 0 ? (uint32_t)max_d_env
                                 : (samp ? (P >= 16384u ? (ep0_arr[0] >= 0.32f ? D : (D > 5u ? 5u : D))
                                                        : (ep0_arr[0] >= 0.58f ? D : (D > 4u ? 4u : D)))
                                         : (D >= 5u ? D + 1u : D));
  if (max_d + 1u > mtp_cap) max_d = mtp_cap - 1u;
  const uint32_t lk_cap = (mp_q8 && (P >= 32768u || nl < 11u) && max_tn > 8u) ? 8u : max_tn;

  const uint32_t r0_tok = qd0.n > 0u ? qd0.id[0] : ci[0];
  const uint32_t r0_cnt = qd0.n > 0u ? qd0.n : MTP_TK;
  int lk_match_top = (nl > lk_agree && r0_tok == ld[1]);
  if (lk_match_top && ml >= lk_long && (lk_agree <= 1u || D < 2u || P + slot_next + 1u > g->n_ctx)) {
    for (uint32_t j = 1; j <= nl; j++) d[j] = ld[j];
    *nd_out = nl, *tree_n_out = 0u, (*lk_np)++;
    return;
  }

  mtp_cand sibs[48];
  uint32_t nsib = 0;
  uint8_t fcur[2] = {1, 0};
  uint32_t nf = 1u;
  int lk_match = lk_match_top;

  tr[1].tok = r0_tok, tr[1].par = 0, tr[1].dep = 1, tr[1].p = ep0_arr[0], nt = 2u;
  if (!lk_match && r0_cnt > 1u && ep0_arr[1] >= p_par_eff && nt + 1u < mtp_cap && D >= 2u &&
      slot_next + 2u <= 16u && P + slot_next + 2u <= g->n_ctx) {
    const uint8_t idx = (uint8_t)nt++;
    tr[idx].tok = qd0.n > 0u ? qd0.id[1] : ci[1];
    tr[idx].par = 0, tr[idx].dep = 1, tr[idx].p = ep0_arr[1];
    fcur[nf++] = idx;
  }
  for (uint32_t r = nf; r < 4u && r < r0_cnt && nsib < 48u; r++) {
    const uint32_t rtok = qd0.n > 0u ? qd0.id[r] : ci[r];
    if (ep0_arr[r] >= p_br_eff) sibs[nsib++] = (mtp_cand){rtok, 0, ep0_arr[r]};
  }

  uint8_t trunk_end = 1u;
  uint32_t trunk_len = 1u;

  for (uint32_t dep = 1u; dep < max_d && nt < mtp_cap && slot_next + nf <= 16u && P + slot_next + nf <= g->n_ctx; dep++) {
    mtp_run_frontier(b, m, g, P, tr, fcur, nf, &slot_next, pgrow, uv, ci, up);

    const uint8_t u0 = fcur[0];
    mtp_qdist qd_u0;
    mtp_qdist_make(sp, V, rej_on, uv, ci, up, ep0_arr, &qd_u0);
    tr_q[u0] = qd_u0;
    const uint32_t cnt0 = qd_u0.n > 0u ? qd_u0.n : MTP_TK;
    const uint32_t tok0 = qd_u0.n > 0u ? qd_u0.id[0] : ci[0];
    const float ep0 = ep0_arr[0], cp0 = tr[u0].p * ep0;

    uint8_t u1 = 0;
    uint32_t cnt1 = 0, tok1_top = 0;
    float ep1_top = 0.0f, cp1_top = 0.0f;
    if (nf == 2u) {
      u1 = fcur[1];
      mtp_qdist qd_u1;
      mtp_qdist_make(sp, V, rej_on, uv + MTP_TK, ci + MTP_TK, up + MTP_TK, ep1_arr, &qd_u1);
      tr_q[u1] = qd_u1;
      cnt1 = qd_u1.n > 0u ? qd_u1.n : MTP_TK;
      tok1_top = qd_u1.n > 0u ? qd_u1.id[0] : ci[MTP_TK];
      ep1_top = ep1_arr[0], cp1_top = tr[u1].p * ep1_top;
    }

    /* Rule (a): long lookup match agreed on lk_agree drafts -> take full lookup chain immediately! */
    if (lk_match && ep0 >= p_keep && tok0 == ld[dep + 1u] && dep + 1u >= lk_agree && ml >= lk_long) {
      for (uint32_t r = 1u; r < 3u && r < cnt0 && nsib < 48u; r++) {
        const uint32_t rtok = qd_u0.n > 0u ? qd_u0.id[r] : ci[r];
        const float ep = ep0_arr[r], cp = tr[u0].p * ep;
        if (ep >= p_br_eff && cp >= p_ch_eff) sibs[nsib++] = (mtp_cand){rtok, u0, cp};
      }
      if (nf == 2u) {
        for (uint32_t r = 0u; r < 2u && r < cnt1 && nsib < 48u; r++) {
          const uint32_t rtok = tr_q[u1].n > 0u ? tr_q[u1].id[r] : ci[MTP_TK + r];
          const float ep = ep1_arr[r], cp = tr[u1].p * ep;
          if (ep >= p_br_eff && cp >= p_ch_eff) sibs[nsib++] = (mtp_cand){rtok, u1, cp};
        }
      }
      for (uint32_t j = dep + 1u; j <= nl && nt < lk_cap; j++) {
        const uint8_t idx = (uint8_t)nt++;
        tr[idx].tok = ld[j], tr[idx].par = trunk_end, tr[idx].dep = (uint8_t)j, tr[idx].p = cp0, tr[idx].from_lk = 1;
        trunk_end = idx, trunk_len = j;
      }
      (*lk_np)++;
      lk_match = 0;
      break;
    }

    const int keep_trunk = (dep < D) ? (ep0 >= p_keep) : (ep0 >= p_keep && cp0 >= p_deep_eff);
    if (!keep_trunk) {
      for (uint32_t r = 0u; r < 3u && r < cnt0 && nsib < 48u; r++) {
        const uint32_t rtok = qd_u0.n > 0u ? qd_u0.id[r] : ci[r];
        const float ep = ep0_arr[r], cp = tr[u0].p * ep;
        if (ep >= p_br_eff && cp >= p_ch_eff) sibs[nsib++] = (mtp_cand){rtok, u0, cp};
      }
      if (nf == 2u) {
        for (uint32_t r = 0u; r < 2u && r < cnt1 && nsib < 48u; r++) {
          const uint32_t rtok = tr_q[u1].n > 0u ? tr_q[u1].id[r] : ci[MTP_TK + r];
          const float ep = ep1_arr[r], cp = tr[u1].p * ep;
          if (ep >= p_br_eff && cp >= p_ch_eff) sibs[nsib++] = (mtp_cand){rtok, u1, cp};
        }
      }
      break;
    }

    const uint8_t idx0 = (uint8_t)nt++;
    tr[idx0].tok = tok0, tr[idx0].par = u0, tr[idx0].dep = (uint8_t)(dep + 1u), tr[idx0].p = cp0;
    trunk_end = idx0, trunk_len = dep + 1u;
    if (lk_match && (dep + 1u > nl || tok0 != ld[dep + 1u])) lk_match = 0;

    /* Dynamic 2nd frontier node at dep == 1 only (keeping dep >= 2 on fast contiguous nf=1 path) */
    const uint32_t tok0_alt = cnt0 > 1u ? (qd_u0.n > 0u ? qd_u0.id[1] : ci[1]) : 0u;
    const float ep0_alt = cnt0 > 1u ? ep0_arr[1] : 0.0f, cp0_alt = tr[u0].p * ep0_alt;
    const int can_par = !lk_match && dep == 1u && dep + 1u < max_d && nt + 1u < mtp_cap &&
                        slot_next + 2u <= 16u && P + slot_next + 2u <= g->n_ctx;
    const int ok_b = (nf == 2u && cnt1 > 0u && ep1_top >= p_keep && cp1_top >= p_par_eff);
    const int ok_a = (cnt0 > 1u && ep0_alt >= p_par_eff && cp0_alt >= p_par_eff);

    fcur[0] = idx0, nf = 1u;
    uint32_t r0_start = 1u, r1_start = 0u;
    if (can_par && (ok_b || ok_a)) {
      if (ok_b && (!ok_a || cp1_top >= cp0_alt)) {
        const uint8_t idx1 = (uint8_t)nt++;
        tr[idx1].tok = tok1_top, tr[idx1].par = u1, tr[idx1].dep = (uint8_t)(dep + 1u), tr[idx1].p = cp1_top;
        fcur[nf++] = idx1, r1_start = 1u;
      } else {
        const uint8_t idx1 = (uint8_t)nt++;
        tr[idx1].tok = tok0_alt, tr[idx1].par = u0, tr[idx1].dep = (uint8_t)(dep + 1u), tr[idx1].p = cp0_alt;
        fcur[nf++] = idx1, r0_start = 2u;
      }
    }

    for (uint32_t r = r0_start; r < 4u && r < cnt0 && nsib < 48u; r++) {
      const uint32_t rtok = qd_u0.n > 0u ? qd_u0.id[r] : ci[r];
      const float ep = ep0_arr[r], cp = tr[u0].p * ep;
      if (ep >= p_br_eff && cp >= p_ch_eff) sibs[nsib++] = (mtp_cand){rtok, u0, cp};
    }
    if (u1) {
      for (uint32_t r = r1_start; r < 3u && r < cnt1 && nsib < 48u; r++) {
        const uint32_t rtok = tr_q[u1].n > 0u ? tr_q[u1].id[r] : ci[MTP_TK + r];
        const float ep = ep1_arr[r], cp = tr[u1].p * ep;
        if (ep >= p_br_eff && cp >= p_ch_eff) sibs[nsib++] = (mtp_cand){rtok, u1, cp};
      }
    }
  }

  /* Rule (b): after MTP trunk stops, if all trunk_len >= lk_agree drafts matched lookup, extend trunk with lookup! */
  if (lk_match && trunk_len >= lk_agree && nl > trunk_len) {
    for (uint32_t j = trunk_len + 1u; j <= nl && nt < lk_cap; j++) {
      const uint8_t idx = (uint8_t)nt++;
      tr[idx].tok = ld[j], tr[idx].par = trunk_end, tr[idx].dep = (uint8_t)j, tr[idx].p = 0.25f, tr[idx].from_lk = 1;
      trunk_end = idx, trunk_len = j;
    }
    (*lk_np)++;
    lk_match = 0;
  }

  /* Context-aware verify-tier & KV pair-group (npg) knapsack packing for free sibling leaves */
  mtp_sort_cands(sibs, nsib);
  {
    const uint32_t kv_pgrp = mp_q8 ? 48u : 16u; /* ie_attn_long_mp handles up to 48 pairs (npg <= 3, nt <= 8) in 1 KV pass */
    const uint32_t grp = m->n_kv ? m->n_head / m->n_kv : 6u;
    const uint32_t sib_cap = nt > mtp_cap ? nt : mtp_cap;
    float ctx_pen = P > 2048u ? (float)(P - 2048u) * (0.16f / 32768.0f) : 0.0f;
    if (ctx_pen > 0.24f) ctx_pen = 0.24f;
    for (uint32_t i = 0; i < nsib && nt < sib_cap; i++) {
      int dup = 0;
      for (uint32_t y = 1; y < nt; y++)
        if (tr[y].par == sibs[i].par && tr[y].tok == sibs[i].tok) { dup = 1; break; }
      if (dup || tr[sibs[i].par].dep >= 15u) continue;

      const uint32_t npg_cur = (grp * nt + kv_pgrp - 1u) / kv_pgrp;
      const uint32_t npg_nxt = (grp * (nt + 1u) + kv_pgrp - 1u) / kv_pgrp;
      float p_req = p_ch_eff;
      if (nt == 4u) {
        const float g8 = samp ? 0.09f : p_t9;
        if (p_req < g8) p_req = g8;
      }
      if (nt == 8u || nt == 9u) {
        /* _tr8 (<= 8 nodes, 2-4 rows/wave GEMV + 1-pass ie_attn_long_mp) -> _tr16 (9..16 nodes + 4-pass KV at 80K):
         * only cross past nt = 8 when cumulative path probability covers both weight and long-KV step cost */
        const float g16 = (samp ? 0.30f : p_t16) + ctx_pen + (P >= 49152u ? 0.08f : 0.0f);
        if (p_req < g16) p_req = g16;
      } else if (npg_nxt > npg_cur) {
        /* 1 extra KV pair-group pass (only when kv_pgrp == 16): break-even across the bucket */
        const float gnpg = 0.05f + ctx_pen * 0.30f;
        if (p_req < gnpg) p_req = gnpg;
      }

      if (sibs[i].p < p_req) {
        if (nt == 8u || nt == 9u) break;
        /* Bucket lookahead: if multiple remaining candidates fit into the new (tier, npg) bucket,
         * enter the bucket when their combined acceptance gain covers the step's crossing cost. */
        uint32_t tier_end = (nt + 1u) <= 4u ? 4u : (nt + 1u) <= 8u ? 8u : sib_cap;
        if (tier_end > sib_cap) tier_end = sib_cap;
        uint32_t b_end = nt + 1u;
        while (b_end < tier_end && (grp * (b_end + 1u) + kv_pgrp - 1u) / kv_pgrp == npg_nxt) b_end++;
        float b_sum = 0.0f;
        uint32_t b_cnt = 0, cap_cnt = b_end - nt;
        for (uint32_t j = i; j < nsib && b_cnt < cap_cnt; j++) {
          if (sibs[j].p < p_ch_eff || tr[sibs[j].par].dep >= 15u) continue;
          int d2 = 0;
          for (uint32_t y = 1; y < nt; y++)
            if (tr[y].par == sibs[j].par && tr[y].tok == sibs[j].tok) { d2 = 1; break; }
          if (d2) continue;
          b_sum += sibs[j].p, b_cnt++;
        }
        if (b_sum < p_req) break;
      }

      const uint8_t idx = (uint8_t)nt++;
      tr[idx].tok = sibs[i].tok, tr[idx].par = sibs[i].par;
      tr[idx].dep = (uint8_t)(tr[sibs[i].par].dep + 1u), tr[idx].p = sibs[i].p;
    }

    /* Optional 0-ms lookup graft when lookup did not match trunk, staying inside the current (tier, npg) bucket */
    if (nl >= 2u && ml >= 3u) {
      const uint32_t npg_cur = (grp * nt + kv_pgrp - 1u) / kv_pgrp;
      uint32_t b_cap = nt <= 4u ? (P < 8192u && ml >= 4u ? 5u : 4u) : nt <= 8u ? 8u : max_tn;
      while (P >= 16384u && b_cap > nt && (grp * b_cap + kv_pgrp - 1u) / kv_pgrp > npg_cur) b_cap--;
      const uint32_t cap = b_cap < max_tn ? b_cap : max_tn;
      if (nt < cap) {
        const uint32_t glim = nl < 3u ? nl : 3u;
        uint8_t cur = 0;
        int added = 0;
        float gp = 0.045f;
        for (uint32_t j = 1; j <= glim; j++, gp *= 0.85f) {
          uint8_t hit = 0;
          for (uint32_t y = 1; y < nt; y++)
            if (tr[y].par == cur && tr[y].tok == ld[j]) { hit = (uint8_t)y; break; }
          if (hit) { cur = hit; continue; }
          if (nt >= cap || tr[cur].dep >= 15u) break;
          const uint8_t idx = (uint8_t)nt++;
          tr[idx].tok = ld[j], tr[idx].par = cur, tr[idx].dep = (uint8_t)(tr[cur].dep + 1u);
          tr[idx].p = gp, tr[idx].from_lk = 1;
          cur = idx, added = 1;
        }
        if (added) (*lk_np)++;
      }
    }
  }

emit_tree:
  /* Finalize per-node q_out:
   * - Nodes with >= 2 children or whose child has grandchildren use q_out.n = 0 (exact sequential conditional
   *   multi-draft verification in spec_rej_step, giving full sum_r p_target(child_r) acceptance and preserving trunk).
   * - A node whose ONLY child is an unexpanded MTP leaf upgrades that leaf to a Leviathan q-draw (sum min(p, q)). */
  if (rej_on && samp) {
    uint8_t nch[16] = {0}, only_ch[16] = {0};
    for (uint32_t y = 1; y < nt; y++) {
      const uint8_t p = tr[y].par;
      nch[p]++, only_ch[p] = (uint8_t)y;
    }
    for (uint32_t u = 0; u < nt; u++) {
      if (nch[u] == 1u) {
        const uint8_t v = only_ch[u];
        if (nch[v] == 0u && !tr[v].from_lk && tr_q[u].n > 0u) {
          const uint32_t pick = mtp_qdist_draw(&tr_q[u], drng);
          tr[v].tok = tr_q[u].id[pick], tr[v].stoch = 1;
          continue;
        }
      }
      tr_q[u].n = 0;
    }
  } else {
    for (uint32_t u = 0; u < nt; u++) tr_q[u].n = 0;
  }

  /* Emit DFS order visiting stochastic child first, then highest-probability child */
  {
    uint32_t order[16], no = 0, stk[16], ns = 0, map[16] = {0};
    stk[ns++] = 0;
    while (ns) {
      const uint32_t x = stk[--ns];
      map[x] = no, order[no++] = x;
      uint32_t ch[16], nc = 0;
      for (uint32_t y = 1; y < nt; y++)
        if (tr[y].par == x) ch[nc++] = y;
      for (uint32_t i = 1; i < nc; i++) { /* ascending priority -> pushed lowest-first -> popped highest-first */
        const uint32_t v = ch[i];
        uint32_t j = i;
        while (j && mtp_ch_gt(tr, ch[j - 1], v)) ch[j] = ch[j - 1], j--;
        ch[j] = v;
      }
      for (uint32_t i = 0; i < nc && ns < 16u; i++) stk[ns++] = ch[i];
    }
    int is_chain = 1;
    for (uint32_t i = 0; i < no; i++) {
      const uint32_t x = order[i];
      tk[i] = tr[x].tok;
      tpar[i] = (uint8_t)(x ? map[tr[x].par] : 0u);
      q_out[i] = tr_q[x];
      if (i && tpar[i] != i - 1u) is_chain = 0;
    }
    if (is_chain) {
      *tree_n_out = 0u, *nd_out = no - 1u;
      for (uint32_t j = 1; j < no; j++) d[j] = tk[j];
    } else {
      *tree_n_out = no, *nd_out = 0u;
    }
  }
}

/* Exact multi-draft speculative rejection step (Leviathan et al. 2023 / SpecInfer):
 * builds the exact target candidate distribution (matching sample_cands / sample with dry_apply, temp, top_k, top_p),
 * tests stochastic primary child (if qd->n > 0) via min(1, p(c)/q(c)) + max(0, p - q) residual, then tests any
 * remaining deterministic sibling children via point-mass rejection, and samples the bonus token from residual p. */
static uint32_t spec_rej_step(sampler *sp, float *cv_row, uint32_t *ci_row, uint32_t KS, int cands, const float *lg_row,
                              uint32_t V, const mtp_qdist *qd, const uint32_t *ctok, const uint32_t *cid_node,
                              uint32_t n_child, uint32_t *acc_node) {
  *acc_node = 0u;
  const uint32_t lim = sp->vocab ? sp->vocab : 0xFFFFFFFFu;
  float v_buf[1024];
  uint32_t id_buf[1024];
  uint32_t m = 0;

  if (cands) {
    for (uint32_t i = 0; i < KS && m < 1024u; i++) {
      if (ci_row[i] < lim && isfinite(cv_row[i])) v_buf[m] = cv_row[i], id_buf[m] = ci_row[i], m++;
    }
  } else {
    const uint32_t K = sample_k(sp, V);
    if (K >= V || K > 1024u) return sample(sp, lg_row, V);
    uint32_t hn = 0;
    for (uint32_t i = 0; i < V; i++) {
      if (hn < K) {
        uint32_t c = hn++;
        id_buf[c] = i;
        while (c && lg_row[id_buf[(c - 1u) / 2u]] > lg_row[id_buf[c]]) {
          uint32_t pp = (c - 1u) / 2u, t = id_buf[pp];
          id_buf[pp] = id_buf[c], id_buf[c] = t, c = pp;
        }
      } else if (lg_row[i] > lg_row[id_buf[0]]) {
        id_buf[0] = i;
        uint32_t c = 0;
        for (;;) {
          uint32_t l = 2u * c + 1u, r = l + 1u, mn = c;
          if (l < hn && lg_row[id_buf[l]] < lg_row[id_buf[mn]]) mn = l;
          if (r < hn && lg_row[id_buf[r]] < lg_row[id_buf[mn]]) mn = r;
          if (mn == c) break;
          uint32_t t = id_buf[mn];
          id_buf[mn] = id_buf[c], id_buf[c] = t, c = mn;
        }
      }
    }
    for (uint32_t i = 0; i < hn; i++) v_buf[i] = lg_row[id_buf[i]];
    for (uint32_t i = 1; i < hn; i++) {
      const float vv = v_buf[i];
      const uint32_t ii = id_buf[i];
      uint32_t j = i;
      while (j && (v_buf[j - 1] < vv || (v_buf[j - 1] == vv && id_buf[j - 1] > ii)))
        v_buf[j] = v_buf[j - 1], id_buf[j] = id_buf[j - 1], j--;
      v_buf[j] = vv, id_buf[j] = ii;
    }
    for (uint32_t i = 0; i < hn; i++) {
      if (id_buf[i] < lim && isfinite(v_buf[i])) {
        if (m != i) v_buf[m] = v_buf[i], id_buf[m] = id_buf[i];
        m++;
      }
    }
  }
  if (!m) return 0u;
  if (sp->dry_mult > 0.0f && sp->dry_allowed && sp->dry_bits) dry_apply(sp, v_buf, id_buf, m);
  if (sp->temp <= 0.0f) return id_buf[0];

  double pd[1024], z = 0.0;
  const float mx = v_buf[0];
  for (uint32_t i = 0; i < m; i++) {
    const float w = expf((v_buf[i] - mx) / sp->temp);
    pd[i] = isfinite(w) ? (double)w : 0.0;
    z += pd[i];
  }
  if (!(z > 0.0) || !isfinite(z)) return id_buf[0];
  if (sp->top_p < 1.0f && m > 1u) {
    double c = 0.0;
    uint32_t keep = 0;
    while (keep < m && c < (double)sp->top_p * z) c += pd[keep++];
    if (!keep) keep = 1u, c = pd[0];
    m = keep, z = c;
  }
  const double inv_z = 1.0 / z;
  for (uint32_t i = 0; i < m; i++) pd[i] *= inv_z;

  uint32_t r0 = 0u;
  if (qd && qd->n > 0u && n_child > 0u) {
    const uint32_t ztok = ctok[0];
    double qz = 0.0, pz = 0.0;
    for (uint32_t c = 0; c < qd->n; c++)
      if (qd->id[c] == ztok) { qz = qd->q[c]; break; }
    for (uint32_t i = 0; i < m; i++)
      if (id_buf[i] == ztok) { pz = pd[i]; break; }
    if (qz > 0.0) {
      if (pz >= qz || (pz > 0.0 && rng_u01(sp) * qz < pz)) {
        *acc_node = cid_node[0];
        return ztok;
      }
      double rsum = 0.0;
      for (uint32_t i = 0; i < m; i++) {
        double qi = 0.0;
        for (uint32_t c = 0; c < qd->n; c++)
          if (qd->id[c] == id_buf[i]) { qi = qd->q[c]; break; }
        pd[i] = pd[i] > qi ? pd[i] - qi : 0.0;
        rsum += pd[i];
      }
      if (rsum > 1e-15) {
        const double inv = 1.0 / rsum;
        for (uint32_t i = 0; i < m; i++) pd[i] *= inv;
      } else {
        for (uint32_t i = 0; i < m; i++) pd[i] = 0.0;
        pd[0] = 1.0;
      }
      r0 = 1u;
    }
  }

  for (uint32_t r = r0; r < n_child; r++) {
    const uint32_t ztok = ctok[r];
    int32_t hit = -1;
    for (uint32_t i = 0; i < m; i++)
      if (id_buf[i] == ztok) { hit = (int32_t)i; break; }
    if (hit < 0 || !(pd[hit] > 0.0)) continue;
    if (pd[hit] >= 1.0 - 1e-12 || rng_u01(sp) < pd[hit]) {
      *acc_node = cid_node[r];
      return ztok;
    }
    pd[hit] = 0.0;
    double rsum = 0.0;
    for (uint32_t i = 0; i < m; i++) rsum += pd[i];
    if (rsum > 1e-15) {
      const double inv = 1.0 / rsum;
      for (uint32_t i = 0; i < m; i++) pd[i] *= inv;
    } else {
      pd[0] = 1.0;
      break;
    }
  }

  double u = rng_u01(sp);
  uint32_t last_nz = 0;
  for (uint32_t i = 0; i < m; i++) {
    if (pd[i] > 0.0) last_nz = i;
    u -= pd[i];
    if (u <= 0.0 && pd[i] > 0.0) return id_buf[i];
  }
  return id_buf[last_nz];
}

/* Greedy decoding with the MTP head drafting D tokens per step (GPU).
 * Positions < P are done in the model and in the MTP KV cache (MTP position
 * q holds the pair (h(q-1), token q), h = the model's normed final hidden
 * state). A step: (1) the MTP head catches up the tokens accepted in the last
 * step with their true h and drafts d1 from (h(P-1), t(P)); (2) for D >= 2 it
 * drafts d2 from (its own h, d1), and so on; (3) the model runs [t(P), d1, .., dD] at
 * P .. P + D in one pass (the weights read once) and gives the greedy token
 * a_j after each; (4) d_j is accepted while d_j == a_(j-1). So the output is
 * exactly the plain greedy output. The linear attention state after the
 * last accepted token is in slot (slot + k) % 4 (ie_gdn). out[0 .. n_prompt)
 * is the prompt; returns the number of tokens in out. */
uint32_t spec_generate(backend *b, const model *m, const graph *g, uint32_t *out, uint32_t n_prompt, uint32_t total, uint32_t D,
                       int64_t eos, int stop, uint32_t first, uint32_t hrow, sampler *sp, uint32_t V, double *gen_ms,
                       uint32_t *n_steps, uint32_t *n_acc, float pmin, uint32_t *n_drafted, gen_emit emit, void *ectx,
                       const int64_t *stops, uint32_t n_stops, uint32_t *p_done, uint32_t *h_row) {
  /* With sampling (temp > 0): the model samples its token a_j after each
   * of t(P), d1, .., dD from its own logits, and d_(j+1) is accepted while
   * it equals a_j ("sample and match"). The drafts are fixed, so each emitted
   * token is drawn once from the model's exact distribution given the tokens
   * before it (with incremental dry_commit at every depth). Allocate 16 rows
   * so tree and lookup verify runs (up to 16 nodes) never overflow lg. */
  if (m && m->vocab_unpadded && V > m->vocab_unpadded) V = m->vocab_unpadded;
  if (stop && eos < 0 && m && m->arch == ARCH_INKLING2 && V > 200006u) eos = 200006;
  const uint32_t lg_rows = D + 1u > 16u ? D + 1u : 16u;
  float *lg = sp->temp > 0.0f ? malloc((size_t)lg_rows * g->bufs[g->logits].stride) : NULL;
  double sample_ms = 0; /* host time of the sampling (with the reads of the logits or of the top K) */
  uint32_t a[16], d[16], tk[16];
  /* the prompt is done (prefill): the model and the MTP KV cache for
   * positions < P; first: the greedy token at P; h(P-1) in h_out row hrow */
  uint32_t P = n_prompt, n = n_prompt, k = 0, tP = first;
  /* IE_SPEC_TIME=1: the host time of the drafts and of the verify runs (each run ends with a read of its argmax, so it waits for the GPU) */
  const int stime = getenv("IE_SPEC_TIME") != NULL;
  double t_draft = 0, t_verify = 0, tq;
  /* prompt lookup: IE_LOOKUP=0 off; IE_LOOKUP_MIN: the shortest match used
   * (default 2); IE_LOOKUP_LONG: the match that skips the rest of the MTP
   * chain (default 16); IE_LOOKUP_AGREE: the MTP drafts that must agree
   * (default 2); IE_LOOKUP_MAX: lookup drafts per step (default 15) */
  const char *ev = getenv("IE_LOOKUP");
  const int lk_on = !(ev && ev[0] == '0');
  ev = getenv("IE_LOOKUP_MIN");
  const uint32_t lk_min = ev ? (uint32_t)atoi(ev) : 2u;
  ev = getenv("IE_LOOKUP_LONG");
  const uint32_t lk_long = ev ? (uint32_t)atoi(ev) : 16u;
  ev = getenv("IE_LOOKUP_AGREE");
  const uint32_t lk_agree = ev ? (uint32_t)atoi(ev) : 2u;
  uint32_t lk_n = 0;
  /* Multi-branch batched MTP tree speculation (default on when D >= 3 and b->tree_ok; IE_MTP_TREE=0 reverts to chain) */
  const char *mte = getenv("IE_MTP_TREE");
  const int mtp_tr_allow = !(mte && mte[0] == '0') && D >= 3u && b->tree_ok && b->accept_tree && b->topk;
  uint32_t mtp_W = getenv("IE_MTP_TREE_W") ? (uint32_t)atoi(getenv("IE_MTP_TREE_W")) : 15u;
  if (mtp_W < 1u) mtp_W = 1u;
  if (mtp_W > 15u) mtp_W = 15u;
  const float mtp_pb = getenv("IE_MTP_BRANCH_P") ? (float)atof(getenv("IE_MTP_BRANCH_P")) : 0.08f;
  const float mtp_pc = getenv("IE_MTP_CHILD_P") ? (float)atof(getenv("IE_MTP_CHILD_P")) : 0.06f;
  const float mtp_pd = getenv("IE_MTP_DEEP_P") ? (float)atof(getenv("IE_MTP_DEEP_P")) : 0.25f;
  /* DFlash (m->df, the graph's DFlash ops): IE_DFLASH=0 uses the MTP head */
  const char *dfe = getenv("IE_DFLASH");
  const int dfm = g->df_feat >= 0 && !(dfe && dfe[0] == '0');
  const int dsm = g->ds_feat >= 0 && m->ds != NULL;
  /* drafts per DFlash block: 15 (twice the trained block, as lucebox; measured faster than 7 on code and prose) */
  uint32_t ing = 0, df_n = dfm ? 15u : 0u;
  if (dfm && df_n > spec_verify_max() - 1u) df_n = spec_verify_max() - 1u;
  if (dfm && b->verify_max && df_n > b->verify_max - 1u) df_n = b->verify_max - 1u;
  if (dfm && df_n + 1u > g->T) df_n = g->T - 1u;
  if (dfm && getenv("IE_DFLASH_N")) df_n = (uint32_t)atoi(getenv("IE_DFLASH_N"));
  const float df_pmin = getenv("IE_DFLASH_PMIN") ? (float)atof(getenv("IE_DFLASH_PMIN")) : 0.3f;
  /* IE_DFLASH_AUTO=0: DFlash at every step (the MTP head not run) */
  const int df_auto = dfm && g->mtp_h >= 0 && !(getenv("IE_DFLASH_AUTO") && getenv("IE_DFLASH_AUTO")[0] == '0'); /* no MTP head: DFlash alone */
  int df_mode = 1, use_df = 0, df_seen[2] = {0, 0};
  /* a DFlash step verifies a DDTree (dd_tree) where the backend can (IE_DFLASH_TREE=0: the chain): at most
   * IE_TREE_B nodes (default 15) of path probability at least IE_TREE_PMIN (default 0.05) */
  const int tr_on = dfm && b->tree_ok && !(getenv("IE_DFLASH_TREE") && getenv("IE_DFLASH_TREE")[0] == '0');
  uint32_t tr_B = getenv("IE_TREE_B") ? (uint32_t)atoi(getenv("IE_TREE_B")) : 15u;
  if (tr_B > 15u) tr_B = 15u;
  if (b->verify_max && tr_B > b->verify_max - 1u) tr_B = b->verify_max - 1u;
  const float tr_pmin = getenv("IE_TREE_PMIN") ? (float)atof(getenv("IE_TREE_PMIN")) : 0.05f;
  /* the tree's size by cost (dd_tree; IE_TREE_COST=0: all nodes down to IE_TREE_PMIN): the verify time of 1 .. 16
   * tokens (ms, MiMo-9B at 55 tokens, IE_VERIFY_BENCH; the shape matters, not the scale) and IE_TREE_TD, the time of
   * a DFlash draft in the same units (default 4.9) */
  static const float tr_vc_ms[16] = {9.8f, 10.9f, 11.4f, 11.6f, 12.0f, 12.2f, 12.7f, 12.9f,
                                     14.1f, 14.5f, 14.5f, 14.7f, 16.2f, 17.1f, 17.3f, 17.6f};
  const float *tr_vc = getenv("IE_TREE_COST") && getenv("IE_TREE_COST")[0] == '0' ? NULL : tr_vc_ms;
  /* IE_TREE_VC=t1,t2,..,t16: another model's verify times (IE_VERIFY_BENCH=1 prints them), e.g. a dense 8B */
  static float tr_vc_env[16];
  if (tr_vc && getenv("IE_TREE_VC")) {
    const char *s = getenv("IE_TREE_VC");
    uint32_t i = 0;
    for (; i < 16u && *s; i++) {
      char *e;
      tr_vc_env[i] = strtof(s, &e);
      if (e == s) break;
      s = *e == ',' ? e + 1 : e;
    }
    if (i == 16u) tr_vc = tr_vc_env;
    else fprintf(stderr, "IE_TREE_VC: 16 comma separated times needed; the built-in ones are used\n");
  }
  const float tr_td = getenv("IE_TREE_TD") ? (float)atof(getenv("IE_TREE_TD")) : 4.9f;
  uint32_t tree_n = 0, tr_steps = 0;
  uint8_t tpar[16], tpath[16];
  mtp_qdist qnode[16];
  const char *reje = getenv("IE_SPEC_REJ");
  const int rej_on = lg != NULL && !(reje && reje[0] == '0') && sample_k(sp, V) <= 1024u;
  uint64_t drng = (sp->rng ^ 0x9E3779B97F4A7C15ull) | 1ull;
  /* IE_TREE_DUMP=FILE (debug, with --dflash and IE_DFLASH_AUTO=0): each step also runs the whole MTP chain (D drafts,
   * no pmin cut) and writes both drafters' candidates: "S P tP", "M d p ..." (the MTP chain and its probabilities),
   * "F j id logit ..." (DFlash's top K at each block row j); "I ids" at the end (the output). The verify is the
   * DFlash chain as without it, so the output is unchanged (tools: research/mimo-9b/tools/tree_oracle.py). */
  FILE *tdump = dfm && getenv("IE_TREE_DUMP") ? fopen(getenv("IE_TREE_DUMP"), "w") : NULL;
  double df_rate[2] = {0, 0}, df_tok[2] = {0, 0}, df_ms[2] = {0, 0};
  uint32_t df_used[2] = {0, 0};
  if (dfm && getenv("IE_DFLASH_N") && b->verify_max && df_n > b->verify_max - 1u) df_n = b->verify_max - 1u;
  if (dfm && (df_n < 1 || df_n > 15 || df_n + 1 > g->T || df_n > spec_verify_max() - 1u)) ie_die("IE_DFLASH_N: 1 to 15 (the verify room)");
  if (b->verify_max && D + 1u > b->verify_max) { /* fewer linear attention states than asked for (VRAM) */
    fprintf(stderr, "spec: %u drafts, but a verify run holds %u tokens: %u drafts\n", D, b->verify_max, b->verify_max - 1u);
    D = b->verify_max - 1u;
  }
  uint32_t lk_max = spec_verify_max() - 1u; /* lookup drafts per step: the graph's room and the backend's states */
  if (b->verify_max && lk_max > b->verify_max - 1u) lk_max = b->verify_max - 1u;
  if (lk_max > g->T - 1u) lk_max = g->T - 1u;
  if (lk_max < D) lk_max = D;
  /* R30: register prompt tokens into DSpark MicroSpec shortlist (no-op if inactive) */
  if (dsm && b->ds_set_shortlist) b->ds_set_shortlist(b, out, n_prompt);
  /* R29 ForeSpark online entropy-calibrated survival stopping (IE_DS_FORESPARK=1; default 0 until A/B-proven) */
  static int fs_env = -1;
  static float fs_gain = 1.0f, fs_tau_fix = 0.0f, fs_tau_min = 0.18f, fs_tau_max = 0.62f;
  if (fs_env < 0) {
    const char *e = getenv("IE_DS_FORESPARK");
    fs_env = (e && e[0] && e[0] != '0') ? 1 : 0;
    if (getenv("IE_DS_FORESPARK_GAIN")) fs_gain = (float)atof(getenv("IE_DS_FORESPARK_GAIN"));
    if (getenv("IE_DS_SURV_TAU")) fs_tau_fix = (float)atof(getenv("IE_DS_SURV_TAU"));
    if (getenv("IE_DS_FORESPARK_MIN")) fs_tau_min = (float)atof(getenv("IE_DS_FORESPARK_MIN"));
    if (getenv("IE_DS_FORESPARK_MAX")) fs_tau_max = (float)atof(getenv("IE_DS_FORESPARK_MAX"));
  }
  static const float fs_prior[8] = {0.92f, 0.85f, 0.78f, 0.70f, 0.62f, 0.54f, 0.45f, 0.35f};
  float fs_num[8], fs_den[8], ds_last_ent[16];
  for (int i = 0; i < 8; i++) fs_num[i] = 4.0f * fs_prior[i], fs_den[i] = 4.0f;
  double fs_step_tok = 0.0, fs_step_ms = 0.0;
  uint32_t ds_prob_nd = 0u;
  /* R31 O(1) CPU Suffix Automaton pre-check (default 1 on Inkling-2 DSpark; IE_SAM_LOOKUP=0 opts out) */
  static int sam_env = -1, sam_pre_env = -1;
  static uint32_t sam_min_ml = 12u, sam_hi_temp_ml = 14u, sam_min_draft = 4u, sam_max_draft = 7u;
  static float sam_max_temp = 0.3f;
  if (sam_env < 0) {
    const char *e = getenv("IE_SAM_LOOKUP");
    sam_env = !(e && e[0] == '0');
    const char *pe = getenv("IE_SAM_PRECHECK");
    sam_pre_env = (pe && pe[0] == '0') ? 0 : 1;
    if (getenv("IE_SAM_MIN_ML")) sam_min_ml = (uint32_t)atoi(getenv("IE_SAM_MIN_ML"));
    if (getenv("IE_SAM_HI_TEMP_ML")) sam_hi_temp_ml = (uint32_t)atoi(getenv("IE_SAM_HI_TEMP_ML"));
    if (getenv("IE_SAM_MIN_DRAFT")) sam_min_draft = (uint32_t)atoi(getenv("IE_SAM_MIN_DRAFT"));
    if (getenv("IE_SAM_MAX_DRAFT")) sam_max_draft = (uint32_t)atoi(getenv("IE_SAM_MAX_DRAFT"));
    if (getenv("IE_SAM_MAX_TEMP")) sam_max_temp = (float)atof(getenv("IE_SAM_MAX_TEMP"));
  }
  ie_sam sam;
  memset(&sam, 0, sizeof sam);
  const int sam_on = dsm && sam_env && sam_init(&sam, total < g->n_ctx ? total : g->n_ctx);
  if (sam_on && n_prompt > 0u) {
    /* Pre-extend prompt tokens [0..n_prompt-1] before decode timer t0 (all strictly <= n-2 since out[n_prompt]=tP). */
    for (uint32_t p = 0u; p < n_prompt; p++) {
      sam_extend(&sam, out[p], p);
      sam_mark_lastpos(&sam, sam.last, p);
      sam.lp_done = p + 1u;
    }
  }
  uint32_t sam_skip_n = 0u;
  /* R32 Tiny 8-node DSpark tree verify (IE_DS_TREE=1 adaptive 2-branch, 2 linear-chain parity check) */
  static int ds_tree_env = -1;
  static float ds_tree_pmax = 0.78f, ds_tree_p2 = 0.12f;
  if (ds_tree_env < 0) {
    const char *e = getenv("IE_DS_TREE");
    ds_tree_env = e ? atoi(e) : 0;
    if (getenv("IE_DS_TREE_PMAX")) ds_tree_pmax = (float)atof(getenv("IE_DS_TREE_PMAX"));
    if (getenv("IE_DS_TREE_P2")) ds_tree_p2 = (float)atof(getenv("IE_DS_TREE_P2"));
  }
  out[n++] = tP;
  commit_output_token(sp, tP);
  /* out[n_prompt .. sent) went to emit; a nonzero return ends the loop */
  uint32_t sent = n_prompt;
  int ended = 0;
#define SPEC_EMIT() (emit && !ended && n > sent ? (emit(ectx, out + sent, n - sent) ? (ended = 1) : (sent = n, 0)) : 0)
  const double t0 = now_ms();
  *n_steps = 0, *n_acc = 0, *n_drafted = 0;
  if (getenv("IE_VERIFY_BENCH")) { /* debug: the time of a verify run of T tokens at P (T = 1 .. 16), a tree of 16 nodes
                                    * (a chain), and the one-token step; then exit */
    const int reps = 20;
    uint8_t cpar[16];
    /* IE_VERIFY_BENCH_TEXT=1: the run holds the last 16 tokens of the text instead of 16 copies of one token (distinct
     * tokens route to more MoE experts, as real drafts do) */
    const int vb_text = getenv("IE_VERIFY_BENCH_TEXT") && n >= 16u;
    for (uint32_t j = 0; j < 16u; j++) tk[j] = vb_text ? out[n - 16u + j] : tP, cpar[j] = (uint8_t)(j ? j - 1u : 0u);
    const uint32_t vb_t = (uint32_t)atoi(getenv("IE_VERIFY_BENCH")); /* 1: all T; T > 1: that T only (IE_PROFILE=2) */
    for (uint32_t T = vb_t > 1u ? vb_t : 1u; T <= (vb_t > 1u ? vb_t : 16u); T++) {
      if (T + 1u > g->T) break;
      for (int r = -3; r < reps; r++) {
        if (r == 0) tq = now_ms();
        b->run(b, 0, tk, T, P, 0, a, NULL);
      }
      fprintf(stderr, "verify bench: T %2u %.3f ms\n", T, (now_ms() - tq) / reps);
    }
    if (vb_t <= 1u && b->tree_ok && b->tree_ok(b, P, 16u)) {
      for (int r = -3; r < reps; r++) {
        if (r == 0) tq = now_ms();
        b->tree_par = cpar;
        b->run(b, 0, tk, 16u, P, 0, a, NULL);
        b->tree_par = NULL;
      }
      fprintf(stderr, "verify bench: tree 16 %.3f ms\n", (now_ms() - tq) / reps);
    }
    for (int r = -3; r < reps; r++) { /* last: the step changes the state */
      if (r == 0) tq = now_ms();
      b->step(b, tP, P, NULL, NULL);
    }
    fprintf(stderr, "verify bench: step %.3f ms\n", (now_ms() - tq) / reps);
    if (vb_t > 1u) b->close(b); /* the IE_PROFILE report */
    exit(0);
  }
  while (n < total && !(stop && (int64_t)out[n - 1] == eos)) {
    if (SPEC_EMIT()) break;
    /* (1) catch up positions P-k .. P-1 and draft at P: tokens out[P-k .. P].
     * A draft is kept only while the MTP head's probability of it is at
     * least pmin: nd drafts (0 .. D) this step. */
    float pr[16];
    uint32_t nd = 0;
    tq = stime ? now_ms() : 0;
    const double t_step = now_ms();
    /* DFlash: the context of the last accepted tokens (every step, so the
     * drafter can be used at any step). With the MTP head kept current too
     * (df_auto), each step uses the drafter with the better measured rate
     * (tokens per ms) and tries the other one every 16th step. */
    if ((dfm || dsm) && ing) b->run(b, 2, &out[P - ing], ing, P - ing, 0, NULL, NULL);
    if (dfm && ing && g->df_cap >= 0) df_capture(b, g, &out[P - ing], ing, P - ing); /* IE_DF_CAPTURE */
    if (dsm) {
      static int ds_prob_env = -1;
      if (ds_prob_env < 0) ds_prob_env = !(getenv("IE_DS_PROB") && getenv("IE_DS_PROB")[0] == '0');
      const int use_ds_prob = ds_prob_env && rej_on && sp && sp->temp > 0.0f;
      ds_prob_nd = 0u;
      /* R31: O(1) SAM Pre-Check - skip ds_draft (0.00 ms GPU!) on strong exact suffix match */
      uint32_t sam_ml = 0u, sam_ld[16];
      uint32_t sam_nl = 0u;
      if (sam_on) {
        const uint32_t room = g->n_ctx > P + 1u ? g->n_ctx - P - 1u : 0u;
        const uint32_t sam_cap = lk_max < room ? lk_max : room;
        if (sam_cap > 0u) sam_nl = sam_query(&sam, out, n, sam_cap, sam_ld, &sam_ml);
        const uint32_t min_ml = (sp && sp->temp > sam_max_temp) ? sam_hi_temp_ml : sam_min_ml;
        if (sam_pre_env && sam_nl >= sam_min_draft && sam_ml >= min_ml) {
          nd = sam_nl < sam_max_draft ? sam_nl : sam_max_draft;
          for (uint32_t j = 1u; j <= nd; j++) d[j] = sam_ld[j];
          tree_n = 0u;
          memset(qnode, 0, sizeof qnode);
          lk_n++, sam_skip_n++;
          goto verify;
        }
      }
      uint32_t ds_n = D;
      if (ds_n > m->ds->block - 1u) ds_n = m->ds->block - 1u;
      if (b->verify_max && ds_n > b->verify_max - 1u) ds_n = b->verify_max - 1u;
      if (ds_n + 1u > g->T) ds_n = g->T - 1u;
      if (P + ds_n + 1u > g->n_ctx) ds_n = g->n_ctx > P + 1u ? g->n_ctx - P - 1u : 0u;
      memset(qnode, 0, sizeof qnode);
      if (ds_n > 0u) {
        ie_ds_samp dss;
        if (use_ds_prob) {
          memset(&dss, 0, sizeof dss);
          dss.on = 1;
          dss.temp = sp->temp;
          dss.top_p = sp->top_p;
          const uint32_t sk = sample_k(sp, V);
          dss.top_k = (sk > 0u && sk < QDIST_MAX) ? sk : QDIST_MAX;
          dss.rng = drng;
          if (fs_env) {
            /* Weighted linear regression of 8-bin calibrated EMA alpha_lut[b] vs H_b = 0.4*b + 0.2 */
            double sw = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
            for (int bi = 0; bi < 8; bi++) {
              const double w = (double)fs_den[bi];
              const double x = 0.4 * (double)bi + 0.2;
              const double y = (double)(fs_num[bi] / fs_den[bi]);
              sw += w; sx += w * x; sy += w * y; sxx += w * x * x; sxy += w * x * y;
            }
            const double den = sw * sxx - sx * sx;
            double slope = den > 1e-9 ? (sw * sxy - sx * sy) / den : -0.30;
            double icpt = sw > 1e-9 ? (sy - slope * sx) / sw : 0.92;
            float sa = (float)icpt;
            float sb = (float)(-slope);
            if (sa < 0.55f) sa = 0.55f; else if (sa > 1.00f) sa = 1.00f;
            if (sb < 0.05f) sb = 0.05f; else if (sb > 0.65f) sb = 0.65f;
            const float ms_tok = (fs_step_tok > 1.0) ? (float)(fs_step_ms / fs_step_tok) : 5.8f;
            const float dt_dr = b->ds_set_shortlist ? 0.10f : 0.18f;
            const float d_vc = 1.05f;
            float tau = fs_tau_fix > 0.0f ? fs_tau_fix : (fs_gain * (dt_dr + d_vc) / (ms_tok > 1.0f ? ms_tok : 1.0f));
            if (tau < fs_tau_min) tau = fs_tau_min; else if (tau > fs_tau_max) tau = fs_tau_max;
            float d0h = (sb > 1e-4f) ? (sa - 0.5f * tau) / sb : 2.5f;
            if (d0h < 1.2f) d0h = 1.2f; else if (d0h > 2.8f) d0h = 2.8f;
            dss.fs_on = 1;
            dss.surv_a = sa;
            dss.surv_b = sb;
            dss.surv_pmin = tau;
            dss.drop0_h = d0h;
          }
          b->ds_samp = &dss;
        }
        pr[0] = pmin;
        b->run(b, 3, &tP, ds_n, P, 0, d + 1, pr);
        const int prob_used = use_ds_prob && dss.used;
        if (use_ds_prob) {
          if (dss.used) drng = dss.rng;
          b->ds_samp = NULL;
        }
        for (uint32_t j = 0; j < ds_n; j++) {
          if (prob_used) {
            if (!dss.out_n[j]) break;
            nd = j + 1u;
            ds_last_ent[j] = dss.out_entropy[j];
            const uint32_t qn = dss.out_n[j] <= QDIST_MAX ? dss.out_n[j] : QDIST_MAX;
            double qsum = 0.0;
            for (uint32_t c = 0; c < qn; c++) qsum += (double)dss.out_q[j][c];
            if (qsum > 0.0 && isfinite(qsum)) {
              qnode[j].n = (uint8_t)qn;
              const double inv = 1.0 / qsum;
              for (uint32_t c = 0; c < qn; c++) {
                qnode[j].id[c] = dss.out_id[j][c];
                qnode[j].q[c] = (double)dss.out_q[j][c] * inv;
              }
            }
          } else {
            if (pr[j] < pmin) break;
            nd = j + 1u;
          }
        }
        if (prob_used) ds_prob_nd = nd;
        if (lk_on && nd >= lk_agree) {
          uint32_t ml = sam_on ? sam_ml : 0u, ld[16];
          const uint32_t nl = sam_on ? (memcpy(ld, sam_ld, sizeof ld), sam_nl)
                                     : lookup_draft(out, n, lk_max, ld, &ml);
          if (nl > lk_agree && ml >= lk_min && !memcmp(d + 1, ld + 1, lk_agree * sizeof *d)) {
            if ((ml >= lk_long && nl >= nd) || (nl > nd && !memcmp(d + 1, ld + 1, nd * sizeof *d))) {
              for (uint32_t j = 1; j <= nl; j++) {
                if (j > nd || d[j] != ld[j]) qnode[j - 1u].n = 0; /* lookup-grafted token: exact point-mass rej */
                d[j] = ld[j];
              }
              nd = nl, lk_n++;
            }
          }
        }
        /* R32: Tiny 8-node speculative tree verify (IE_DS_TREE=2 linear-chain parity, =1 adaptive 2-branch depth-1 split) */
        if (ds_tree_env && b->tree_ok && b->accept_tree && nd >= 1u) {
          if (ds_tree_env == 2 && nd + 1u <= 8u && b->tree_ok(b, P, nd + 1u)) {
            tree_n = nd + 1u;
            tk[0] = tP; tpar[0] = 0u;
            for (uint32_t j = 1u; j <= nd; j++) tk[j] = d[j], tpar[j] = (uint8_t)(j - 1u);
            goto verify;
          }
          if (ds_tree_env == 1 && prob_used && qnode[0].n >= 2u && nd <= 6u && b->tree_ok(b, P, nd + 2u)) {
            double p_prim = 0.0, p_alt = 0.0;
            uint32_t alt_tok = 0u;
            int have_alt = 0;
            for (uint32_t c = 0u; c < qnode[0].n; c++) {
              if (qnode[0].id[c] == d[1]) p_prim = qnode[0].q[c];
              else if (!have_alt || qnode[0].q[c] > p_alt) { p_alt = qnode[0].q[c]; alt_tok = qnode[0].id[c]; have_alt = 1; }
            }
            if (have_alt && p_prim < (double)ds_tree_pmax && p_alt >= (double)ds_tree_p2) {
              tree_n = nd + 2u;
              tk[0] = tP; tpar[0] = 0u;
              for (uint32_t j = 1u; j <= nd; j++) tk[j] = d[j], tpar[j] = (uint8_t)(j - 1u);
              tk[nd + 1u] = alt_tok; tpar[nd + 1u] = 0u;
              qnode[nd + 1u].n = 0u;
              goto verify;
            }
          }
        }
      }
      tree_n = 0;
      goto verify;
    }

    /* the other drafter at steps 1 and 16, 32, ... (a probe) */
    const int probe = *n_steps == 1u || (*n_steps && *n_steps % 16u == 0u);
    /* a long match in the text: the MTP path, whose chain stops after the
     * first lk_agree drafts agree with the lookup (cheaper than a DFlash
     * block); such steps do not count in the rates */
    uint32_t ml0 = 0, ld0[16];
    const int lkl = df_auto && lk_on && lookup_draft(out, n, lk_max, ld0, &ml0) > lk_agree && ml0 >= lk_long;
    use_df = dfm && !lkl && (!df_auto || (probe ? !df_mode : df_mode));
    tree_n = 0;
    memset(qnode, 0, sizeof qnode);
    if (!use_df && !tdump && mtp_tr_allow) {
      uint32_t max_tn = mtp_W + 1u;
      if (max_tn > spec_verify_max()) max_tn = spec_verify_max();
      if (b->verify_max && max_tn > b->verify_max) max_tn = b->verify_max;
      if (max_tn > g->T) max_tn = g->T;
      if (P + max_tn > g->n_ctx) max_tn = g->n_ctx > P ? g->n_ctx - P : 1u;
      while (max_tn > 2u && !b->tree_ok(b, P, max_tn)) max_tn--;
      if (max_tn >= 2u && b->tree_ok(b, P, max_tn)) {
        const uint32_t hr0 = hrow;
        hrow = 0;
        mtp_draft_tree(b, m, g, sp, V, rej_on, &drng, out, n, P, k, hr0, tP, D, max_tn, lk_on, lk_min, lk_long,
                       lk_agree, lk_max, pmin, mtp_pb, mtp_pc, mtp_pd, sp->cv, sp->ci, d, &nd, tk, tpar, &tree_n,
                       qnode, &lk_n);
        goto verify;
      }
    }
    if (!use_df || df_auto || tdump) { /* (1) the MTP catch-up and its first draft (DFlash: the KV cache only) */
      b->copy_rows(b, g->mtp_h, 0, g->h_out, hrow, k + 1);
      hrow = 0;
      b->run(b, use_df && !tdump ? 4 : 1, &out[P - k], k + 1, P - k, 0, tk, pr);
    }
    if (use_df && tdump) { /* the MTP chain at this step too (its KV rows past P are written again later) */
      uint32_t md[16];
      float mp[16];
      md[1] = model_draft_tok(m, tk[k]), mp[1] = pr[k];
      for (uint32_t j = 2; j <= D; j++) {
        b->copy_rows(b, g->mtp_h, 0, g->mtp_g, j == 2 ? k : 0, 1);
        b->run(b, 1, &md[j - 1], 1, P + j - 1, 0, tk, pr);
        md[j] = model_draft_tok(m, tk[0]), mp[j] = pr[0];
      }
      fprintf(tdump, "S %u %u\nM", P, tP);
      for (uint32_t j = 1; j <= D; j++) fprintf(tdump, " %u %.5g", md[j], mp[j]);
      fprintf(tdump, "\n");
    }
    if (use_df) {
      nd = df_draft(b, m, g, sp, tP, P, df_n, d, df_pmin);
      const uint32_t lk0 = lk_n;
      if (tdump) {
        if (getenv("IE_DF_RAW") && getenv("IE_DF_RAW")[0]) { /* debug: the drafter's full logits, rows 1 .. df_n */
          static FILE *fr;
          const uint32_t nvb = m->df_head.rows ? m->df_head.rows : m->vocab;
          if (!fr) fr = fopen(getenv("IE_DF_RAW"), "wb");
          float *rw = malloc((size_t)nvb * 4);
          const uint32_t hd_raw[3] = {P, df_n, nvb};
          fwrite(hd_raw, 4, 3, fr);
          for (uint32_t j = 1; j <= df_n; j++) {
            b->read_rows(b, g->df_logits, j, 1, rw);
            fwrite(rw, 4, nvb, fr);
          }
          free(rw);
          fflush(fr);
        }
        for (uint32_t j = 0; j < df_n; j++) {
          fprintf(tdump, "F %u", j + 1);
          for (uint32_t c = 0; c < m->df->top_k; c++)
            fprintf(tdump, " %u %.6g", sp->ci[j * m->df->top_k + c], sp->cv[j * m->df->top_k + c]);
          fprintf(tdump, "\n");
        }
      }
      if (lk_on && nd >= lk_agree) { /* the prompt lookup, as with the MTP chain: (a) replaces, (b) extends the drafts */
        uint32_t ml = 0, ld[16];
        const uint32_t nl = lookup_draft(out, n, lk_max, ld, &ml);
        if (nl > lk_agree && ml >= lk_min && !memcmp(d + 1, ld + 1, lk_agree * sizeof *d)) {
          if ((ml >= lk_long && nl >= nd) || (nl > nd && !memcmp(d + 1, ld + 1, nd * sizeof *d))) {
            for (uint32_t j = 1; j <= nl; j++) d[j] = ld[j];
            nd = nl, lk_n++;
          }
        }
      }
      if (tr_on && lk_n == lk0) { /* the DDTree of the block (more than the chain), if the backend can verify it */
        uint32_t ttok[16];
        const uint32_t tn = dd_tree(sp->cv, sp->ci, m->df->top_k, df_n, tr_B, tr_pmin, tP, ttok, tpar, tr_vc, tr_td);
        if (b->tree_ok(b, P, tn > 2u ? tn : 2u)) {
          if (tn > (tr_vc ? 1u : nd + 1u)) { /* with the cost cut, the tree (it may be smaller than the chain) */
            tree_n = tn;
            for (uint32_t j = 0; j < tn; j++) tk[j] = ttok[j];
          } else if (tr_vc)
            nd = 0; /* no draft is worth its verify time */
        }
      }
      goto verify;
    }
    if (pr[k] >= pmin) d[1] = model_draft_tok(m, tk[k]), nd = 1;
    /* Prompt lookup (IE_LOOKUP, default on): an earlier place in the text
     * that ends like the text now (a match of ml >= lk_min tokens) gives up
     * to lk_max drafts: its next tokens. A verify of 16 tokens costs about
     * 1.6 times one of 8, so they are used only where the MTP chain agrees:
     * (a) ml >= lk_long: when the chain's first lk_agree drafts are the
     *     lookup's, the lookup's drafts are used and the rest of the chain is
     *     not run; (b) else, after the chain: when its nd >= lk_agree drafts
     *     are all the lookup's, the lookup's next tokens are added. The verify
     *     step is the same, so the output is the same; only the drafts change. */
    uint32_t ml = 0, ld[16], nl = lk_on ? lookup_draft(out, n, lk_max, ld, &ml) : 0;
    if (nl && (ml < lk_min || nl <= lk_agree)) nl = 0;
    /* the chain so far differs from the lookup: no lookup this step */
#define LK_DIFF() (nd && d[nd] != ld[nd])
    if (nl && LK_DIFF()) nl = 0;
    for (uint32_t j = 2; j <= D && nd == j - 1; j++) { /* (2) */
      if (nl && ml >= lk_long && nd >= lk_agree) { /* (a) */
        for (uint32_t i = nd + 1; i <= nl; i++) d[i] = ld[i];
        nd = nl, lk_n++;
        goto verify;
      }
      b->copy_rows(b, g->mtp_h, 0, g->mtp_g, j == 2 ? k : 0, 1);
      b->run(b, 1, &d[j - 1], 1, P + j - 1, 0, tk, pr);
      if (pr[0] >= pmin) d[j] = model_draft_tok(m, tk[0]), nd = j;
      if (nl && LK_DIFF()) nl = 0;
    }
#undef LK_DIFF
    if (nl > nd && nd >= lk_agree) { /* (b) */
      for (uint32_t i = nd + 1; i <= nl; i++) d[i] = ld[i];
      nd = nl, lk_n++;
    }
  verify: /* (3) lossless target verification + incremental DRY history commit */
    if (stime) t_draft += now_ms() - tq, tq = now_ms();
    sp->dry_xn = 0;
    if (tree_n) { /* a tree: one run of its nodes, then walk from root drawing once per visited node and committing
                   * each accepted token into DRY history BEFORE evaluating depth + 1 (100% draft=0 equivalence) */
      static int host_topk_tr = -1;
      if (host_topk_tr < 0) host_topk_tr = getenv("IE_HOST_TOPK") != NULL;
      b->sample_k = lg && b->topk && sample_k(sp, V) < V && !host_topk_tr ? sample_k(sp, V) : 0;
      b->tree_par = tpar;
      b->run(b, 0, tk, tree_n, P, 0, a, NULL);
      b->sample_k = 0;
      if (stime) t_verify += now_ms() - tq;
      const double ts1 = now_ms();
      const uint32_t KS = lg ? sample_k(sp, V) : 0u;
      const int cands = lg && gpu_cands(sp, b, g, 0, tree_n, V, sp->cv, sp->ci);
      if (lg && !cands) b->read_rows(b, g->logits, 0, tree_n, lg);
      uint32_t cur = 0, kk = 0, pt[17];
      const uint32_t lst = g->bufs[g->logits].stride / 4u;
      tpath[0] = 0;
      for (;;) {
        uint32_t tok = a[cur], c = 0;
        const int force_end = reasoning_exhausted(sp);
        if (lg && rej_on && !force_end) {
          uint32_t ctok[16], cnode[16], nc = 0;
          for (uint32_t y = cur + 1u; y < tree_n; y++)
            if (tpar[y] == cur) ctok[nc] = tk[y], cnode[nc] = y, nc++;
          tok = spec_rej_step(sp, sp->cv + (size_t)cur * KS, sp->ci + (size_t)cur * KS, KS, cands,
                              lg ? lg + (size_t)cur * lst : NULL, V, &qnode[cur], ctok, cnode, nc, &c);
          commit_output_token(sp, tok);
          int st = (stop && (int64_t)tok == eos) || (n + kk + 1u >= total);
          for (uint32_t s = 0; s < n_stops; s++) st |= stops[s] >= 0 && (int64_t)tok == stops[s];
          if (st) c = 0;
        } else {
          if (force_end) tok = sp->reasoning_end_plus_one - 1u;
          else if (lg) {
            tok = cands ? sample_cands(sp, sp->cv + (size_t)cur * KS, sp->ci + (size_t)cur * KS, KS)
                        : sample(sp, lg + (size_t)cur * lst, V);
          }
          commit_output_token(sp, tok);
          int st = (stop && (int64_t)tok == eos) || (n + kk + 1u >= total);
          for (uint32_t s = 0; s < n_stops; s++) st |= stops[s] >= 0 && (int64_t)tok == stops[s];
          for (uint32_t y = cur + 1u; !st && y < tree_n; y++)
            if (tpar[y] == cur && tk[y] == tok) { c = y; break; }
        }
        if (!c) { a[kk] = tok; break; } /* bonus / terminal token of this step */
        tpath[++kk] = (uint8_t)c, pt[kk] = tok, cur = c;
      }
      sample_ms += now_ms() - ts1;
      for (uint32_t j = 1; j <= kk; j++) d[j] = pt[j];
      for (uint32_t j = 0; j < kk; j++) a[j] = d[j + 1];
      *n_drafted += tree_n - 1u;
      k = kk;
      tr_steps++;
      goto commit_step;
    }
    tk[0] = tP;
    for (uint32_t j = 1; j <= nd; j++) tk[j] = d[j];
    /* sampling from the top K on the GPU (gpu_cands below): the verify run takes that top K itself, in its one wait */
    static int host_topk_ver = -1;
    if (host_topk_ver < 0) host_topk_ver = getenv("IE_HOST_TOPK") != NULL;
    b->sample_k = lg && b->topk && sample_k(sp, V) < V && !host_topk_ver ? sample_k(sp, V) : 0;
    b->run(b, 0, tk, nd + 1, P, 0, a, NULL);
    b->sample_k = 0;
    if (stime) t_verify += now_ms() - tq;
    const double ts0 = now_ms();
    {
      const uint32_t KS = lg ? sample_k(sp, V) : 0u;
      const int cands = lg && gpu_cands(sp, b, g, 0, nd + 1, V, sp->cv, sp->ci);
      const uint32_t lst = g->bufs[g->logits].stride / 4u;
      if (lg && !cands) b->read_rows(b, g->logits, 0, nd + 1, lg);
      k = 0;
      for (uint32_t j = 0; j <= nd; j++) {
        uint32_t acc_c = 0;
        if (reasoning_exhausted(sp)) {
          a[j] = sp->reasoning_end_plus_one - 1u;
          acc_c = (j < nd && a[j] == d[j + 1]) ? (j + 1u) : 0u;
        } else if (lg && rej_on) {
          const uint32_t ctok = j < nd ? d[j + 1] : 0u;
          const uint32_t cid1 = j + 1u;
          a[j] = spec_rej_step(sp, sp->cv + (size_t)j * KS, sp->ci + (size_t)j * KS, KS, cands,
                               lg ? lg + (size_t)j * lst : NULL, V, &qnode[j], &ctok, &cid1, j < nd ? 1u : 0u, &acc_c);
        } else if (lg) {
          a[j] = cands ? sample_cands(sp, sp->cv + (size_t)j * KS, sp->ci + (size_t)j * KS, KS)
                       : sample(sp, lg + (size_t)j * lst, V);
          acc_c = (j < nd && a[j] == d[j + 1]) ? (j + 1u) : 0u;
        } else {
          acc_c = (j < nd && a[j] == d[j + 1]) ? (j + 1u) : 0u;
        }
        const uint32_t tok = a[j];
        commit_output_token(sp, tok);
        int st = (stop && (int64_t)tok == eos) || (n + j + 1u >= total);
        for (uint32_t s = 0; s < n_stops; s++) st |= stops[s] >= 0 && (int64_t)tok == stops[s];
        if (st || j == nd || !acc_c) { k = j; break; }
      }
    }
    sample_ms += now_ms() - ts0;
    *n_drafted += nd;
    if (ie_spec_nk && nd <= 8u && k <= 8u) ie_spec_nk[nd * 9u + k]++;
  commit_step: /* (4) accept */
    if (getenv("IE_SPEC_TRACE")) {
      const uint32_t show = tree_n ? k : nd;
      fprintf(stderr, "step P=%u t=%u%s drafts", P, tP, tree_n ? " (tree)" : "");
      for (uint32_t j = 1; j <= show; j++) fprintf(stderr, " %u", d[j]);
      fprintf(stderr, " model");
      for (uint32_t j = 0; j <= k; j++) fprintf(stderr, " %u", a[j]);
      fprintf(stderr, " -> k=%u\n", k);
    }
    for (uint32_t j = 1; j <= k && n < total; j++) out[n++] = d[j];
    if (n < total) out[n++] = a[k];
    if (dsm && b->ds_note_tokens) b->ds_note_tokens(b, &a[k], 1u);
    if (dsm && fs_env) {
      if (ds_prob_nd > 0u) {
        const uint32_t lim = (k < ds_prob_nd) ? (k + 1u) : ds_prob_nd;
        for (uint32_t j = 0u; j < lim; j++) {
          if (qnode[j].n == 0u) continue;
          int bi = (int)(2.5f * ds_last_ent[j]);
          if (bi < 0) bi = 0; else if (bi > 7) bi = 7;
          const float obs = (j < k) ? 1.0f : 0.0f;
          fs_num[bi] = 0.95f * fs_num[bi] + obs;
          fs_den[bi] = 0.95f * fs_den[bi] + 1.0f;
        }
      }
      fs_step_tok = 0.9 * fs_step_tok + (double)(k + 1u);
      fs_step_ms = 0.9 * fs_step_ms + (now_ms() - t_step);
    }
    if (tree_n) b->accept_tree(b, tpath, k); /* the path's rows into place, the state after its last node */
    else b->accept(b, k); /* the linear attention state after token k */
    ing = k + 1; /* DFlash: positions P .. P + k into the drafter context (their features: verify rows 0 .. k) */
    if (df_auto && !lkl) { /* the rate of the drafter of this step */
      /* the rate: decayed sums of tokens and of time (about the last 10 steps of that drafter) */
      const int u = use_df;
      df_tok[u] = 0.9 * df_tok[u] + (k + 1), df_ms[u] = 0.9 * df_ms[u] + (now_ms() - t_step);
      df_rate[u] = df_tok[u] / df_ms[u];
      df_seen[u] = 1, df_used[u]++;
      /* switch only when the other drafter's rate is 10 % higher */
      if (df_seen[0] && df_seen[1] && df_rate[!df_mode] > 1.1 * df_rate[df_mode]) df_mode = !df_mode;
    }
    P += k + 1;
    tP = a[k];
    (*n_steps)++, *n_acc += k;
    if (stop)
      for (uint32_t j = n_prompt; j < n; j++)
        if ((int64_t)out[j] == eos) { n = j + 1; break; }
  }
  SPEC_EMIT(); /* the last step's tokens (the result does not matter now) */
  if (tdump) {
    fprintf(tdump, "I");
    for (uint32_t j = 0; j < n; j++) fprintf(tdump, " %u", out[j]);
    fprintf(tdump, "\n");
    fclose(tdump);
  }
#undef SPEC_EMIT
  /* If the caller caches this answer (ie-serve multi-turn), catch up the last step's k accepted tokens in blk.64 KV
   * with their true target h_out[0..k-1] so positions < P have exact teacher-forced MTP KV on the next turn. */
  if (p_done && *n_steps && k > 0 && g->mtp_h >= 0 && (!dfm || df_auto)) {
    b->copy_rows(b, g->mtp_h, 0, g->h_out, 0, k);
    b->run(b, 4, &out[P - k], k, P - k, 1, NULL, NULL);
  }
  if (p_done && *n_steps && dsm && ing) {
    b->run(b, 2, &out[P - ing], ing, P - ing, 0, NULL, NULL);
  }
  if (p_done) *p_done = P, *h_row = hrow ? hrow : k; /* the last verify's row k holds h(P - 1) */
  if (lk_on && !emit && *n_steps) fprintf(stderr, "lookup: %u of %u steps drafted from the text\n", lk_n, *n_steps);
  if (sam_skip_n && !emit && *n_steps) fprintf(stderr, "sam: %u of %u steps skipped ds_draft\n", sam_skip_n, *n_steps);
  if (!dfm && tr_steps && !emit) fprintf(stderr, "%s tree: %u of %u steps\n", dsm ? "dspark" : "mtp", tr_steps, *n_steps);
  if (df_auto && !emit) fprintf(stderr, "dflash: %u steps DFlash, %u steps MTP (rates %.3f / %.3f tokens per ms)\n", df_used[1], df_used[0], df_rate[1], df_rate[0]);
  if (dfm && !emit) fprintf(stderr, "dflash: %u tree steps\n", tr_steps);
  if (stime && *n_steps)
    fprintf(stderr, "spec time: drafts %.2f ms/step, verify %.2f ms/step (%u steps)\n", t_draft / *n_steps, t_verify / *n_steps, *n_steps);
  *gen_ms = now_ms() - t0;
  if (lg) fprintf(stderr, "sampling: %.1f ms in all (%.3f ms per step)\n", sample_ms, *n_steps ? sample_ms / *n_steps : 0.0);
  sam_free(&sam);
  free(lg);
  return n;
}
/* ---- one request (ie-serve) ---- */

static void dc_free(struct dcache *d);
void gen_init(gen_ctx *c, backend *b, const model *m, const graph *g) {
  memset(c, 0, sizeof *c);
  c->b = b, c->m = m, c->g = g;
  const uint32_t V = model_vocab(m);
  c->sp.vocab = V;
  c->sp.idx = malloc((size_t)m->vocab * 4);
  c->sp.pr = malloc((size_t)m->vocab * 4);
  c->sp.cv = malloc(16 * 64 * 4);
  c->sp.ci = malloc(16 * 64 * 4);
  /* read_rows writes a full row (buf.stride bytes, 256-aligned), step writes vocab floats */
  size_t n = g->bufs[g->logits].stride;
  if (n < (size_t)m->vocab * 4) n = (size_t)m->vocab * 4;
  c->logits = malloc(n);
  c->nl = calloc(m->vocab, 1);
  if (!c->sp.idx || !c->sp.pr || !c->sp.cv || !c->sp.ci || !c->logits || !c->nl) ie_die("gen_init: out of memory");
  for (uint32_t t = 0; t < V && t < m->n_tokens; t++) {
    uint32_t len;
    char *s = model_detok(m, t, &len);
    c->nl[t] = memchr(s, '\n', len) != NULL;
    free(s);
  }
}

void gen_free(gen_ctx *c) {
  free(c->sp.idx), free(c->sp.pr), free(c->sp.cv), free(c->sp.ci), free(c->logits), free(c->kv_toks), free(c->nl);
  free(c->sp.dry_h), free(c->sp.dry_bits), free(c->sp.dry_head), free(c->sp.dry_prev), free(c->kv_imgs);
  dc_free(c->dc);
  memset(c, 0, sizeof *c);
}

static const float *img_row_at(const gen_img *imgs, uint32_t n_imgs, uint32_t dim, uint32_t pos) {
  for (uint32_t k = 0; k < n_imgs; k++) {
    if (pos >= imgs[k].pos && pos < imgs[k].pos + imgs[k].n_tok)
      return imgs[k].embd + (size_t)(pos - imgs[k].pos) * dim;
  }
  return NULL;
}

/* The prompt tokens toks[start .. end) at their positions, in chunks (GPU),
 * as prefill does but from any start: the MTP h input for start (row 0 of
 * mtp_h) is already set. When need_last == 0 (intermediate checkpoint spans)
 * or on non-final chunks, passes out = NULL so output.weight + D2H sync are
 * skipped, and runs MTP catch-up via sec = 4 (async KV-only). Returns the
 * greedy token after the last one when need_last != 0; *hrow: row T - 1. */
static uint32_t prefill_range(backend *b, const graph *g, const uint32_t *toks, uint32_t start, uint32_t end, uint32_t chunk,
                              int mtp, int need_last, uint32_t *hrow, const gen_img *imgs, uint32_t n_imgs, uint32_t dim) {
  if (chunk > g->T) chunk = g->T;
  if (chunk < 1u) chunk = 1u;
  uint32_t *a = need_last ? malloc((size_t)chunk * 4) : NULL, last = 0, T = 1;
  const float **erows = n_imgs ? malloc((size_t)chunk * sizeof *erows) : NULL;
  if ((need_last && !a) || (n_imgs && !erows)) ie_die("prefill_range: out of memory");
  mtp = mtp && g->mtp_h >= 0; /* a DFlash drafter alone: no MTP head to keep current */
  b->all_logits = 0;
  if (b->kv_io && end > start) b->kv_io(b, 0, end, 0, NULL); /* pre-map up to end in 1 idle VMM batch */
  for (uint32_t p0 = start; p0 < end; p0 += T) {
    T = next_chunk_len(end - p0, chunk);
    const int is_last = need_last && p0 + T >= end;
    int has_img = 0;
    if (n_imgs) {
      for (uint32_t t = 0; t < T; t++) {
        erows[t] = img_row_at(imgs, n_imgs, dim, p0 + t);
        if (erows[t]) has_img = 1;
      }
    }
    b->ext_rows = has_img ? erows : NULL;
    b->run(b, 0, toks + p0, T, p0, 1, is_last ? a : NULL, NULL);
    b->accept(b, T - 1);
    if (is_last) last = a[T - 1];
    if (g->df_feat >= 0 || g->ds_feat >= 0)
      b->run(b, 2, toks + p0, T, p0, (is_last && need_last == 2) ? 2 : 0, NULL, NULL); /* DFlash/DSpark context */
    if (g->df_cap >= 0) df_capture(b, g, toks + p0, T, p0);              /* IE_DF_CAPTURE */
    if (mtp) {
      if (T > 1) b->copy_rows(b, g->mtp_h, 1, g->h_out, 0, T - 1);
      b->run(b, 4, toks + p0, T, p0, 1, NULL, NULL); /* async KV-only catch-up: skips out_draft + host sync */
      b->copy_rows(b, g->mtp_h, 0, g->h_out, T - 1, 1);
    }
    b->ext_rows = NULL;
  }
  free(erows);
  free(a);
  *hrow = T - 1;
  return last;
}

/* ---- the prompt cache (gen_ctx: kv_toks and the checkpoints) ---- */

/* IE_CACHE_CKPT: the most checkpoints (2 .. GEN_CK_MAX, default 16). Models with sliding window layers keep 2 in VRAM,
 * or up to 8 (IE_CACHE_CKPT) in host RAM for Inkling-2 when IE_SNAP_HOST != 0 (0 extra VRAM, compact SWA). */
static uint32_t ck_limit(const gen_ctx *c) {
  const char *ce = getenv("IE_CACHE_CKPT");
  const int v = ce ? atoi(ce) : 16; /* read each time: the tests change it */
  for (uint32_t i = 0; i < c->m->n_kvl; i++)
    if (c->g->kv_len[i] < c->g->n_ctx) {
      if (c->m->arch == ARCH_INKLING2 && !(getenv("IE_SNAP_HOST") && getenv("IE_SNAP_HOST")[0] == '0')) {
        const int iv = ce ? atoi(ce) : 8;
        return iv < 2 ? 2u : iv > GEN_CK_MAX ? (uint32_t)GEN_CK_MAX : (uint32_t)iv;
      }
      return 2u;
    }
  return v < 2 ? 2u : v > GEN_CK_MAX ? (uint32_t)GEN_CK_MAX : (uint32_t)v;
}
/* IE_CACHE_EVERY: a checkpoint every this many positions of a prompt (default 4096; 0: none) */
static uint32_t ck_every(void) {
  const int v = getenv("IE_CACHE_EVERY") ? atoi(getenv("IE_CACHE_EVERY")) : 4096;
  return v > 0 ? (uint32_t)v : 0u;
}
/* kv_toks[pos .. pos + n) = t[0 .. n), and the cache holds pos + n tokens */
static void kv_put(gen_ctx *c, uint32_t pos, const uint32_t *t, uint32_t n) {
  if (pos + n > c->kv_cap) {
    c->kv_cap = pos + n + 4096;
    c->kv_toks = realloc(c->kv_toks, (size_t)c->kv_cap * 4);
    if (!c->kv_toks) ie_die("gen: out of memory");
  }
  if (n) memcpy(c->kv_toks + pos, t, (size_t)n * 4);
  c->kv_n = pos + n;
}
/* the order of dropping when the cache is full: periodic ones first, then the ends of the last prompt and answer, then the places
 * where a prompt first differed (they mark where a prompt changes: the next one with the same start uses them) */
static int ck_rank(int kind) {
  if (kind == GEN_CK_PREFIX) return 3;
  return kind == GEN_CK_SPLIT || kind == GEN_CK_MARK ? 2 : kind == GEN_CK_ANSWER || kind == GEN_CK_PROMPT ? 1 : 0;
}
static int ck_at(const gen_ctx *c, uint32_t pos) {
  for (uint32_t i = 0; i < c->n_ck; i++)
    if (c->ck[i].pos == pos) return (int)i;
  return -1;
}
static void dp_evict(gen_ctx *c, uint32_t i); /* the disk cache policy, below */
/* the checkpoint (index) for position pos: the one there, a new one with a free snapshot slot, or, all slots used, the
 * one to drop (its slot is reused) */
static int ck_slot(gen_ctx *c, uint32_t pos) {
  int i = ck_at(c, pos);
  const int same = i >= 0;
  if (!same && c->n_ck < ck_limit(c)) { /* a free snapshot slot: the lowest one no checkpoint uses */
    int slot = 0;
    for (int used = 1; used; slot += used) {
      used = 0;
      for (uint32_t j = 0; j < c->n_ck; j++) used |= c->ck[j].slot == slot;
    }
    i = (int)c->n_ck++;
    c->ck[i].slot = slot;
  } else if (!same) { /* full: of those this request neither made nor used, the lowest rank, then the least recently
                       * used; if this request made or used them all, the least recently used */
    int old = 0;
    for (uint32_t j = 0; j < c->n_ck; j++) old |= c->ck[j].used < c->req_tick;
    i = -1;
    for (uint32_t j = 0; j < c->n_ck; j++) {
      if (old && c->ck[j].used >= c->req_tick) continue;
      if (i < 0) { i = (int)j; continue; }
      const int rj = ck_rank(c->ck[j].kind), ri = ck_rank(c->ck[i].kind);
      if (rj < ri || (rj == ri && c->ck[j].used < c->ck[i].used)) i = (int)j;
    }
    if (i >= 0) dp_evict(c, (uint32_t)i); /* an admitted one that is not on disk yet is queued before its slot is reused */
  }
  return i;
}
/* a checkpoint of the backend's state now, after the first pos tokens of kv_toks */
static void ck_save(gen_ctx *c, uint32_t pos, int kind, int hbuf) {
  const int same = ck_at(c, pos) >= 0, i = ck_slot(c, pos);
  /* Checkpoints past the first image hold non-reconstructible vision KV rows: keep in memory only, never write to disk. */
  c->ck[i].disk = (c->kv_n_imgs > 0 && pos > c->kv_imgs[0].pos) ? 1 : 0;
  c->ck[i].h = hbuf >= 0;
  /* the end of a newer prompt or answer: the older ends are checkpoints inside the cached tokens now */
  if (kind == GEN_CK_PROMPT || kind == GEN_CK_ANSWER)
    for (uint32_t j = 0; j < c->n_ck; j++)
      if ((int)j != i && c->ck[j].kind == kind) c->ck[j].kind = GEN_CK_EVERY;
  c->b->snap(c->b, 1, c->ck[i].slot, hbuf, pos);
  if (!same || ck_rank(kind) > ck_rank(c->ck[i].kind)) c->ck[i].kind = kind;
  c->ck[i].pos = pos, c->ck[i].used = ++c->tick;
}
/* where the line of position L starts in t: after the last token before L whose text holds a newline, at most 512
 * tokens back; L if there is none. A prompt whose changing part (omp's recalled memories) differs from the last one a
 * few tokens earlier than the last two did still finds the checkpoint there */
static uint32_t line_start(const gen_ctx *c, const uint32_t *t, uint32_t L) {
  for (uint32_t B = L; B > 0 && L - B < 512u; B--)
    if (t[B - 1] < c->m->vocab && c->nl[t[B - 1]]) return B;
  return L;
}
/* IE_CACHE_MARKS: the distances before the end of the system message at which a prompt gets a checkpoint (default
 * 256,1024,4096; 0 or off: no checkpoints at message positions at all). n distances (at most 8) in d, 0 if off */
static uint32_t ck_mark_dists(uint32_t *d) {
  const char *e = getenv("IE_CACHE_MARKS");
  if (!e) e = "256,1024,4096";
  uint32_t n = 0;
  for (; *e && n < 8; e++) {
    if (*e < '0' || *e > '9') continue;
    const uint32_t v = (uint32_t)atoi(e);
    while (*e >= '0' && *e <= '9') e++;
    e--;
    if (v) d[n++] = v;
  }
  return n;
}
/* IE_CACHE_MARK_GAP: a checkpoint within this many tokens of another is not made (default 128: that is 0.1 s of prefill) */
static uint32_t ck_mark_gap(void) {
  const int v = getenv("IE_CACHE_MARK_GAP") ? atoi(getenv("IE_CACHE_MARK_GAP")) : 128;
  return v > 0 ? (uint32_t)v : 1u;
}
static int near_pos(uint32_t a, uint32_t b, uint32_t gap) { return (a > b ? a - b : b - a) < gap; }
/* Tool lists can change within the first disk block. Prefix checkpoints are
 * worth keeping at smaller positions than ordinary conversation checkpoints. */
static uint32_t ck_prefix_min(uint32_t disk_min) {
  return disk_min < 128u ? disk_min : 128u;
}
/* the checkpoints of a prompt run from S to end, in order: L (the start of the line where it first differs; SPLIT),
 * the xm[0 .. nxm) positions (MARK: where prompts are likely to differ; not within the gap of a checkpoint that is
 * there or planned) and every IE_CACHE_EVERY positions (a multiple of it if more would fill over half the cache) */
static uint32_t ck_marks(const gen_ctx *c, uint32_t S, uint32_t L, uint32_t end, const uint32_t *xm, uint32_t nxm,
                         const gen_params *p, uint32_t learned, uint32_t prefix_min, uint32_t *mk, int *kind) {
  uint32_t ev = ck_every();
  const uint32_t half = ck_limit(c) / 2u, gap = ck_mark_gap();
  if (ev && end > S)
    while ((end - S) / ev > half) ev += ck_every();
  uint32_t n = 0;
  const uint32_t pins[] = {p->tools_end, p->prefix[0], p->prefix[1], p->prefix[2], learned};
  for (uint32_t i = 0; i < sizeof pins / sizeof pins[0]; i++) {
    const uint32_t pos = pins[i];
    int add = pos > S && pos < end && pos >= prefix_min;
    for (uint32_t j = 0; add && j < n; j++) add = mk[j] != pos;
    if (add) {
      mk[n] = pos, kind[n++] = GEN_CK_PREFIX;
      if (L == pos) L = 0;
    }
  }
  if (L > S && L < end) mk[n] = L, kind[n++] = GEN_CK_SPLIT;
  for (uint32_t k = 0; k < nxm && n < GEN_CK_MAX; k++) {
    int ok = xm[k] > S && xm[k] < end;
    for (uint32_t j = 0; ok && j < c->n_ck; j++) ok = !near_pos(c->ck[j].pos, xm[k], gap);
    for (uint32_t j = 0; ok && j < n; j++) ok = !near_pos(mk[j], xm[k], gap);
    if (ok) mk[n] = xm[k], kind[n++] = GEN_CK_MARK;
  }
  for (uint32_t e = ev ? (S / ev + 1u) * ev : UINT32_MAX; e < end && n < GEN_CK_MAX; e += ev) {
    int ok = 1;
    for (uint32_t j = 0; ok && j < n; j++) ok = mk[j] != e;
    if (ok) mk[n] = e, kind[n++] = GEN_CK_EVERY;
  }
  for (uint32_t i = 1; i < n; i++) { /* in order of position */
    const uint32_t p = mk[i];
    const int k = kind[i];
    uint32_t j = i;
    for (; j > 0 && mk[j - 1] > p; j--) mk[j] = mk[j - 1], kind[j] = kind[j - 1];
    mk[j] = p, kind[j] = k;
  }
  return n;
}

/* ==== BEGIN disk cache storage (dc_*): the files, their list, reading and writing ==== */
/* ---- the disk prompt cache (gen.h: gen_disk_open) ----
 * Files in the directory, named SIG-T-HASH.iec (SIG and HASH 16 hex digits). Every file is a multiple of 4096 bytes long
 * and its sections start at multiples of 4096 (for reads and writes that bypass the system cache):
 *   a 64-byte header (dc_head: its hits field counts the loads of the checkpoint, its flags field the stored form);
 *   the tokens; for a checkpoint the snapshot; the K/V rows.
 *   T = k, a block of K/V rows: the d->block tokens of positions i * block .. (i + 1) * block - 1, then their rows
 *     (backend.kv_io). Rows depend on every token before them, so HASH chains the hashes of the blocks before it; the
 *     checkpoints of a conversation share its blocks.
 *   T = e, a checkpoint after the first pos tokens: the pos tokens, the snapshot (backend.snap_io; with flag 1 its state
 *     as int16 with one f32 scale per 128 values), then the rows of the positions after its last whole block (pos % block
 *     of them).
 * A file is written under NAME.tmp, then renamed: a killed server leaves no part of a file under a real name.
 * gen_disk_save copies what is to be written to host buffers (pinned for the GPU; on the engine thread, the GPU's);
 * gen_disk_write writes the files, on another thread when the caller gave a lock (ie-serve), so the next request does not
 * wait for the disk. Files queued and not written yet are in the list as pending: not read, not deleted.
 * The RAM tier keeps the images of the files last written or read, up to IE_DISK_RAM_GB (default 8), so that the next
 * resume of a recent conversation reads memory. Reads go straight into the host buffers with several chunks in flight and
 * do not use the system cache (IE_DISK_DIRECT=0: they do). Over the size limit the files go in this order: another sig's
 * (oldest first), the checkpoints not used since they were written (least recently used first), then the protected ones
 * (loaded before, or placed at a split or a mark). */
#ifdef _WIN32
#include <malloc.h>
#endif
#define DC_QUEUE_MAX 4000000000ull /* the most bytes queued for writing: past it the engine thread writes too */
#define DCACHE_VERSION 3u
#define DC_ALIGN 4096u
#define DC_F_STATE16 1u         /* flags: the snapshot's f32 state is stored as int16 + a scale per 128 values */
typedef struct {
  uint32_t magic, version;
  uint64_t sig, hash;
  uint32_t pos, kind, has_h, block;
  uint64_t rowb, snapb; /* snapb: the bytes of the snapshot as backend.snap_io gives it (f32) */
  uint32_t hits, flags;
} dc_head;
_Static_assert(sizeof(dc_head) == 64, "dc_head is 64 bytes");
typedef struct { /* a host buffer: the bytes [0, len) are used */
  char *p;
  size_t cap, len;
  uint8_t pinned; /* allocated by backend.host_alloc */
} dc_mem;
typedef struct {
  char type; /* 'e', 'k', or 'x' (another sig's file) */
  uint8_t pending, busy; /* busy: being read now */
  uint64_t hash, bytes;
  int64_t used;  /* last write or use: microseconds since 1970 (from the file time at the start) */
  int64_t rused; /* the same for the RAM tier */
  char *name;    /* 'x' only */
  uint32_t pos, kind, has_h, hits, flags, *tok; /* 'e' only */
  dc_mem img;    /* the RAM tier: the whole file, or p NULL */
} dc_file;
typedef struct dc_job { /* a file to write: its whole image */
  struct dc_job *next;
  char type;
  uint64_t hash, qb; /* qb: the bytes counted in qbytes */
  dc_mem img;
  uint32_t pos;
  uint8_t pack; /* the image holds an f32 snapshot: convert it to int16 first */
} dc_job;
typedef struct { /* where the sections of a file are */
  size_t tok_off, snap_off, rows_off, total, tok_len, snap_len, rows_len;
} dc_lay;
typedef struct dcache {
  char dir[1024];
  uint64_t sig, limit, bytes; /* bytes: the files written */
  size_t rowb, snapb, f32b, snap16b;
  dc_file *f;
  uint32_t n, cap;
  backend *b;
  dc_mem *pool; /* free pinned buffers */
  uint32_t pool_n, pool_cap;
  uint64_t pinned, pinned_limit;                 /* pinned bytes held (free or not) and the most */
  uint64_t queue_limit;                         /* backpressure on staged disk writes */
  uint64_t ram_bytes, ram_pinned, ram_limit;     /* the RAM tier */
  uint32_t seq, max_nb; /* max_nb: the most blocks before a checkpoint, + 1 */
  uint32_t block, min; /* positions per block (IE_DISK_BLOCK, default 1024); checkpoints at fewer tokens are not
                       * written (IE_DISK_MIN, default 1024: that prefill is under a second) */
  uint32_t state16, direct, ahead; /* IE_DISK_STATE16, IE_DISK_DIRECT (default 1), blocks read ahead (IE_DISK_AHEAD) */
  size_t chunk; /* bytes per read (IE_DISK_CHUNK_MB) */
  void (*lock)(int on); /* NULL: one thread */
  dc_job *qh, *qt;
  uint64_t qbytes;
  uint32_t inflight;  /* jobs taken from the queue and not finished */
  uint32_t w_n;       /* checkpoints written since the queue was last empty (the log line) */
  uint64_t w_bytes;
  double w_t0;
} dcache;

static void dc_lk(const dcache *d, int on) {
  if (d->lock) d->lock(on);
}
static size_t dc_up(size_t n) { return (n + DC_ALIGN - 1) & ~(size_t)(DC_ALIGN - 1); }
static void *dc_amalloc(size_t n) {
#ifdef _WIN32
  return _aligned_malloc(n ? n : 1, DC_ALIGN);
#else
  void *p = NULL;
  return posix_memalign(&p, DC_ALIGN, n ? n : 1) ? NULL : p;
#endif
}
static void dc_afree(void *p) {
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif
}
static void dc_mem_free(const dcache *d, dc_mem *m) {
  if (m->p) m->pinned ? d->b->host_free(d->b, m->p) : dc_afree(m->p);
  memset(m, 0, sizeof *m);
}
static void dc_free(dcache *d) {
  if (!d) return;
  for (uint32_t i = 0; i < d->n; i++) free(d->f[i].tok), free(d->f[i].name), dc_mem_free(d, &d->f[i].img);
  for (dc_job *j = d->qh, *nx; j; j = nx) nx = j->next, dc_mem_free(d, &j->img), free(j);
  for (uint32_t i = 0; i < d->pool_n; i++) dc_mem_free(d, &d->pool[i]);
  free(d->pool), free(d->f), free(d);
}
static uint64_t fnv64(uint64_t h, const void *p, size_t n) {
  const unsigned char *s = p;
  for (size_t i = 0; i < n; i++) h = (h ^ s[i]) * 0x100000001b3ull;
  return h;
}
static int64_t dc_now(dcache *d) { return (int64_t)time(NULL) * 1000000 + d->seq++ % 1000000u; }
/* the chain hashes of the first nb blocks of t */
static void dc_chain(const dcache *d, const uint32_t *t, uint32_t nb, uint64_t *h) {
  for (uint32_t i = 0; i < nb; i++) h[i] = fnv64(i ? h[i - 1] : d->sig, t + (size_t)i * d->block, (size_t)d->block * 4);
}
static uint64_t dc_ehash(const dcache *d, const uint32_t *t, uint32_t pos) {
  return fnv64(fnv64(d->sig ^ 0x65ull, &pos, 4), t, (size_t)pos * 4);
}
static void dc_path(const dcache *d, char *out, size_t cap, char type, uint64_t hash, const char *name) {
  if (type == 'x') snprintf(out, cap, "%s/%s", d->dir, name);
  else snprintf(out, cap, "%s/%016" PRIx64 "-%c-%016" PRIx64 ".iec", d->dir, d->sig, type, hash);
}
static int dc_find(const dcache *d, char type, uint64_t hash) {
  for (uint32_t i = 0; i < d->n; i++)
    if (d->f[i].type == type && d->f[i].hash == hash) return (int)i;
  return -1;
}
static dc_file *dc_add(dcache *d) {
  if (d->n == d->cap) {
    d->cap = d->cap ? d->cap * 2 : 256;
    d->f = realloc(d->f, (size_t)d->cap * sizeof *d->f);
    if (!d->f) ie_die("disk cache: out of memory");
  }
  dc_file *f = &d->f[d->n++];
  memset(f, 0, sizeof *f);
  return f;
}
static void *dc_alloc(size_t n) {
  void *p = malloc(n ? n : 1);
  if (!p) ie_die("disk cache: out of memory (%zu bytes)", n);
  return p;
}

/* ---- host buffers: a pool of pinned ones (allocating them is slow), plain aligned memory past the pinned limit ---- */
/* give buffer m back (the lock is held) */
static void dc_put_l(dcache *d, dc_mem m) {
  if (!m.p) return;
  if (!m.pinned) {
    dc_afree(m.p);
    return;
  }
  if (d->pool_n == d->pool_cap) {
    d->pool_cap = d->pool_cap ? d->pool_cap * 2 : 32;
    d->pool = realloc(d->pool, (size_t)d->pool_cap * sizeof *d->pool);
    if (!d->pool) ie_die("disk cache: out of memory");
  }
  m.len = 0;
  d->pool[d->pool_n++] = m;
}
static void dc_put(dcache *d, dc_mem m) {
  if (!m.p) return;
  dc_lk(d, 1);
  dc_put_l(d, m);
  dc_lk(d, 0);
}
/* the capacity of a buffer for n bytes: a multiple of 8 MB, 1 MB or 64 KB by size */
static size_t dc_cap(size_t n) {
  const size_t s = n >= (64u << 20) ? (8u << 20) : n >= (1u << 20) ? (1u << 20) : 65536u;
  return (n + s - 1) / s * s;
}
/* IE_DISK_STREAM=1: opt-in plain-RAM e + pre-warmed pinned k pool + overlapped k-before-e streaming (default 0: measured 173 vs 153 ms at warm NVMe ceiling) */
static int dc_stream_on(void) {
  static int on = -1;
  if (on < 0) on = (getenv("IE_DISK_STREAM") && getenv("IE_DISK_STREAM")[0] == '1');
  return on;
}
/* plain 4K-aligned host buffer (no hipHostMalloc GART pinning; still valid for FILE_FLAG_NO_BUFFERING) */
static dc_mem dc_get_plain(size_t n) {
  dc_mem m = {0};
  m.cap = dc_up(n);
  m.p = dc_amalloc(m.cap);
  if (!m.p) ie_die("disk cache: out of memory (%zu bytes)", n);
  m.len = n;
  return m;
}
/* a buffer of n bytes, aligned to 4096; pinned while the pinned total stays under its limit (engine thread only: the
 * driver is called here) */
static dc_mem dc_get(dcache *d, size_t n) {
  dc_mem m = {0}, victim[16];
  uint32_t nv = 0;
  int reserved = 0;
  const size_t cap = dc_cap(n);
  dc_lk(d, 1);
  int best = -1;
  for (uint32_t i = 0; i < d->pool_n; i++)
    if (d->pool[i].cap >= n && d->pool[i].cap <= cap + cap / 4 && (best < 0 || d->pool[i].cap < d->pool[best].cap)) best = (int)i;
  if (best >= 0) m = d->pool[best], d->pool[best] = d->pool[--d->pool_n];
  else if (d->b->host_alloc) {
    while (d->pinned + cap > d->pinned_limit && d->pool_n && nv < 16u) {
      victim[nv] = d->pool[--d->pool_n], d->pinned -= victim[nv++].cap;
    }
    if (d->pinned + cap <= d->pinned_limit) d->pinned += cap, reserved = 1;
  }
  dc_lk(d, 0);
  for (uint32_t i = 0; i < nv; i++) dc_mem_free(d, &victim[i]);
  if (!m.p && reserved) {
    m.p = d->b->host_alloc(d->b, cap);
    if (m.p) m.cap = cap, m.pinned = 1;
    else {
      dc_lk(d, 1);
      d->pinned -= cap;
      dc_lk(d, 0);
    }
  }
  if (!m.p) {
    m.cap = dc_up(n);
    m.p = dc_amalloc(m.cap);
    if (!m.p) ie_die("disk cache: out of memory (%zu bytes)", n);
  }
  m.len = n;
  return m;
}
/* the same bytes in plain memory (the pinned buffer goes back to the pool) */
static void dc_unpin(dcache *d, dc_mem *m) {
  if (!m->pinned) return;
  dc_mem o = *m;
  m->p = dc_amalloc(o.cap);
  if (!m->p) {
    *m = o;
    return;
  }
  memcpy(m->p, o.p, o.len);
  m->pinned = 0;
  dc_put(d, o);
}

/* ---- the RAM tier (the lock is held) ---- */
/* keep image m of file f if there is room (evicting the least recently used images that are not being read); 1 and m is
 * the tier's now */
static int dc_ram_adopt_l(dcache *d, dc_file *f, dc_mem *m) {
  if (!m->p || f->img.p || m->cap > d->ram_limit) return 0;
  if (m->pinned && d->ram_pinned + m->cap > d->pinned_limit / 2u) return 0; /* the rest of the pinned buffers are for copies */
  while (d->ram_bytes + m->cap > d->ram_limit) {
    dc_file *v = NULL;
    for (uint32_t i = 0; i < d->n; i++)
      if (d->f[i].img.p && !d->f[i].busy && !d->f[i].pending && &d->f[i] != f && (!v || d->f[i].rused < v->rused)) v = &d->f[i];
    if (!v) return 0;
    d->ram_bytes -= v->img.cap, d->ram_pinned -= v->img.pinned ? v->img.cap : 0;
    dc_put_l(d, v->img);
    memset(&v->img, 0, sizeof v->img);
  }
  f->img = *m, f->rused = f->used;
  d->ram_bytes += m->cap, d->ram_pinned += m->pinned ? m->cap : 0;
  memset(m, 0, sizeof *m);
  return 1;
}
static void dc_remove(dcache *d, uint32_t i) {
  char path[1200];
  dc_path(d, path, sizeof path, d->f[i].type, d->f[i].hash, d->f[i].name);
  remove(path);
  d->bytes -= d->f[i].bytes;
  if (d->f[i].img.p) {
    d->ram_bytes -= d->f[i].img.cap, d->ram_pinned -= d->f[i].img.pinned ? d->f[i].img.cap : 0;
    dc_put_l(d, d->f[i].img);
  }
  free(d->f[i].tok), free(d->f[i].name);
  d->f[i] = d->f[--d->n];
}

/* ---- the layout of a file, the int16 form of a snapshot ---- */
static void dc_layout(const dcache *d, char type, uint32_t pos, uint32_t flags, dc_lay *L) {
  memset(L, 0, sizeof *L);
  L->tok_off = DC_ALIGN;
  if (type == 'k') L->tok_len = (size_t)d->block * 4, L->rows_len = (size_t)d->block * d->rowb;
  else
    L->tok_len = (size_t)pos * 4, L->snap_len = flags & DC_F_STATE16 ? d->snap16b : d->snapb,
    L->rows_len = (size_t)(pos % d->block) * d->rowb;
  L->snap_off = L->tok_off + dc_up(L->tok_len);
  L->rows_off = L->snap_off + dc_up(L->snap_len);
  L->total = L->rows_off + dc_up(L->rows_len);
}
/* the f32 state of the snapshot in img (laid out for f32) as int16 with a scale (absmax / 32767) per 128 values, in place;
 * the rest of the snapshot and the rows move down, img.len shrinks to the layout with flag 1 */
static void dc_pack16(const dcache *d, dc_job *j) {
  dc_lay L32, L16;
  dc_layout(d, 'e', j->pos, 0, &L32), dc_layout(d, 'e', j->pos, DC_F_STATE16, &L16);
  char *s = j->img.p + L32.snap_off;
  const size_t n = d->f32b / 4, ns = n / 128;
  float *x = (float *)s, *sc = dc_alloc(ns * 4);
  int16_t *q = (int16_t *)s; /* q[128 b ..] is written over x[64 b ..] of a block already read */
  for (size_t b = 0; b < ns; b++) {
    float t[128], amax = 0.0f;
    memcpy(t, x + b * 128, sizeof t);
    for (int i = 0; i < 128; i++) amax = fmaxf(amax, fabsf(t[i]));
    const float scale = amax > 0.0f && isfinite(amax) ? amax / 32767.0f : 0.0f, inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int i = 0; i < 128; i++) {
      float v = roundf(t[i] * inv);
      v = v > 32767.0f ? 32767.0f : v < -32767.0f ? -32767.0f : v == v ? v : 0.0f;
      q[b * 128 + (size_t)i] = (int16_t)v;
    }
    sc[b] = scale;
  }
  memcpy(s + n * 2, sc, ns * 4);
  free(sc);
  memmove(s + n * 2 + ns * 4, s + d->f32b, d->snapb - d->f32b);
  char *img = j->img.p;
  memset(img + L16.snap_off + n * 2 + ns * 4 + (d->snapb - d->f32b), 0,
         L16.rows_off - (L16.snap_off + n * 2 + ns * 4 + (d->snapb - d->f32b)));
  if (L32.rows_len) memmove(img + L16.rows_off, img + L32.rows_off, L32.rows_len);
  memset(img + L16.rows_off + L16.rows_len, 0, L16.total - L16.rows_off - L16.rows_len);
  j->img.len = L16.total;
}
/* the snapshot stored with flag 1 (src) back to the f32 form (dst, d->snapb bytes) */
static void dc_unpack16(const dcache *d, const char *src, char *dst) {
  const size_t n = d->f32b / 4, ns = n / 128;
  const int16_t *q = (const int16_t *)src;
  const float *sc = (const float *)(src + n * 2);
  float *x = (float *)dst;
  for (size_t b = 0; b < ns; b++)
    for (size_t i = 0; i < 128; i++) x[b * 128 + i] = (float)q[b * 128 + i] * sc[b];
  memcpy(dst + d->f32b, src + n * 2 + ns * 4, d->snapb - d->f32b);
}

/* ---- reading and writing files ---- */
typedef struct { /* reads of one file into a buffer, several chunks in flight */
  int ok;
#ifdef _WIN32
  HANDLE h;
  OVERLAPPED *ov;
  size_t len, chunk;
  uint32_t nch, issued;
#endif
} dc_rd;
#ifdef _WIN32
static void dc_rd_start(const dcache *d, dc_rd *r, const char *path, char *dst, size_t len) {
  memset(r, 0, sizeof *r);
  const DWORD sh = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  r->h = CreateFileA(path, GENERIC_READ, sh, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED | (d->direct ? FILE_FLAG_NO_BUFFERING : 0), NULL);
  if (r->h == INVALID_HANDLE_VALUE && d->direct)
    r->h = CreateFileA(path, GENERIC_READ, sh, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
  if (r->h == INVALID_HANDLE_VALUE) {
    r->h = NULL;
    return;
  }
  r->len = len, r->chunk = d->chunk, r->nch = (uint32_t)((len + d->chunk - 1) / d->chunk), r->ok = 1;
  r->ov = calloc(r->nch ? r->nch : 1, sizeof *r->ov);
  if (!r->ov) ie_die("disk cache: out of memory");
  for (; r->issued < r->nch; r->issued++) {
    OVERLAPPED *o = &r->ov[r->issued];
    const size_t off = (size_t)r->issued * r->chunk, c = len - off < r->chunk ? len - off : r->chunk;
    o->hEvent = CreateEventA(NULL, TRUE, FALSE, NULL);
    o->Offset = (DWORD)(off & 0xffffffffu), o->OffsetHigh = (DWORD)((uint64_t)off >> 32);
    if (!o->hEvent || (!ReadFile(r->h, dst + off, (DWORD)c, NULL, o) && GetLastError() != ERROR_IO_PENDING)) {
      r->ok = 0;
      break;
    }
  }
}
/* wait for all the chunks, close the file: 1 if every byte arrived */
static int dc_rd_wait(dc_rd *r) {
  if (!r->h) return 0;
  for (uint32_t i = 0; i < r->nch; i++) {
    if (i < r->issued) {
      DWORD got = 0;
      const size_t off = (size_t)i * r->chunk, c = r->len - off < r->chunk ? r->len - off : r->chunk;
      if (!GetOverlappedResult(r->h, &r->ov[i], &got, TRUE) || got != c) r->ok = 0;
    }
    if (r->ov[i].hEvent) CloseHandle(r->ov[i].hEvent);
  }
  r->ok = r->ok && r->issued == r->nch;
  free(r->ov), CloseHandle(r->h);
  r->h = NULL;
  return r->ok;
}
#else
static void dc_rd_start(const dcache *d, dc_rd *r, const char *path, char *dst, size_t len) {
  (void)d;
  FILE *fp = fopen(path, "rb");
  r->ok = fp && fread(dst, 1, len, fp) == len;
  if (fp) fclose(fp);
}
static int dc_rd_wait(dc_rd *r) { return r->ok; }
#endif
/* write len bytes (a multiple of 4096) of src to path */
static int dc_write(const dcache *d, const char *path, const char *src, size_t len) {
#ifdef _WIN32
  HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | (d->direct ? FILE_FLAG_NO_BUFFERING : 0), NULL);
  if (h == INVALID_HANDLE_VALUE) return 0;
  LARGE_INTEGER li;
  li.QuadPart = (LONGLONG)len; /* the size first: the file system allocates once */
  int ok = SetFilePointerEx(h, li, NULL, FILE_BEGIN) && SetEndOfFile(h);
  li.QuadPart = 0;
  ok = ok && SetFilePointerEx(h, li, NULL, FILE_BEGIN);
  for (size_t o = 0; ok && o < len;) {
    const DWORD c = (DWORD)(len - o < (8u << 20) ? len - o : (8u << 20));
    DWORD w = 0;
    ok = WriteFile(h, src + o, c, &w, NULL) && w == c;
    o += c;
  }
  return CloseHandle(h) && ok;
#else
  (void)d;
  FILE *fp = fopen(path, "wb");
  int ok = fp && fwrite(src, 1, len, fp) == len;
  if (fp) ok = (fclose(fp) == 0) && ok;
  return ok;
#endif
}
/* count a load in the file's header */
static void dc_hit_write(const char *path, uint32_t hits) {
  FILE *fp = fopen(path, "r+b");
  if (!fp) return;
  if (fseek(fp, (long)offsetof(dc_head, hits), SEEK_SET) == 0) fwrite(&hits, 4, 1, fp);
  fclose(fp);
}

/* delete the blocks no checkpoint needs, then, while over the limit, the least useful file (another sig's, then a
 * checkpoint not loaded since it was written and not at a split or a mark, then the others; least recently used first)
 * and its blocks if no other needs them. Pending and busy files stay. (Locked.) */
static void dc_trim(dcache *d) {
  while (d->bytes > d->limit) {
    uint8_t *need = calloc(d->n ? d->n : 1, 1);
    uint64_t *ch = dc_alloc((size_t)d->max_nb * 8);
    if (!need) ie_die("disk cache: out of memory");
    for (uint32_t i = 0; i < d->n; i++) {
      if (d->f[i].type != 'e') continue;
      const uint32_t nb = d->f[i].pos / d->block;
      dc_chain(d, d->f[i].tok, nb, ch);
      for (uint32_t k = 0; k < nb; k++) {
        const int j = dc_find(d, 'k', ch[k]);
        if (j >= 0) need[j] = 1;
      }
    }
    for (uint32_t i = d->n; i-- > 0;) /* from the end: dc_remove moves the last file (one already seen) to i */
      if (d->f[i].type == 'k' && !need[i] && !d->f[i].pending && !d->f[i].busy) dc_remove(d, i);
    free(need), free(ch);
    if (d->bytes <= d->limit) return;
    int v = -1, vr = 3;
    for (uint32_t i = 0; i < d->n; i++) {
      const dc_file *f = &d->f[i];
      if (f->type == 'k' || f->pending || f->busy) continue;
      const int rank = f->type == 'x' ? 0 : f->kind == GEN_CK_PREFIX ? 3 : f->hits || ck_rank(f->kind) == 2 ? 2 : 1;
      if (v < 0 || rank < vr || (rank == vr && f->used < d->f[v].used)) v = (int)i, vr = rank;
    }
    if (v < 0) return;
    dc_remove(d, (uint32_t)v);
  }
}

static double dc_env(const char *name, double def) {
  const char *e = getenv(name);
  return e && *e ? atof(e) : def;
}
int gen_disk_open(gen_ctx *c, const char *dir, const char *sig, double limit_gb, void (*lock)(int on)) {
  backend *b = c->b;
  if (!b->snap || !b->snap_io || !b->kv_io || !b->kv_rowb || c->g->df_feat >= 0 || c->g->n_ctx > (1u << 27))
    return 0;
  const size_t snapb = b->snap_io(b, 0, 1, NULL, 0);
  if (snapb == SIZE_MAX) return 0;
  dcache *d = calloc(1, sizeof *d);
  if (!d) ie_die("disk cache: out of memory");
  const char *eb = getenv("IE_DISK_BLOCK"), *em = getenv("IE_DISK_MIN");
  d->block = eb && atoi(eb) >= 16 ? (uint32_t)atoi(eb) : 1024u, d->min = em && atoi(em) > 0 ? (uint32_t)atoi(em) : 1024u;
  snprintf(d->dir, sizeof d->dir, "%s", dir);
  dc_mkdir(dir);
  DIR *dd = opendir(dir);
  if (!dd) {
    fprintf(stderr, "disk cache: cannot open the directory %s: off\n", dir);
    free(d);
    return 0;
  }
  d->b = b, d->rowb = b->kv_rowb, d->snapb = snapb, d->max_nb = c->g->n_ctx / d->block + 1u, d->lock = lock;
  d->f32b = b->snap_f32b <= snapb && b->snap_f32b % 512u == 0 ? b->snap_f32b : 0;
  if (d->f32b) d->snap16b = snapb - d->f32b + d->f32b / 2u + d->f32b / 128u;
  d->state16 = d->f32b && dc_env("IE_DISK_STATE16", 0) == 1.0;
  d->ram_limit = (uint64_t)(dc_env("IE_DISK_RAM_GB", 8.0) * 1e9), d->pinned_limit = (uint64_t)(dc_env("IE_DISK_PINNED_GB", 4.0) * 1e9);
  d->queue_limit = (uint64_t)(fmax(0.125, fmin(4.0, dc_env("IE_DISK_QUEUE_GB", 4.0))) * DC_QUEUE_MAX / 4.0);
  d->direct = dc_env("IE_DISK_DIRECT", 1.0) != 0.0;
  d->chunk = (size_t)(dc_env("IE_DISK_CHUNK_MB", 8.0) >= 1.0 ? dc_env("IE_DISK_CHUNK_MB", 8.0) : 8.0) << 20;
  d->ahead = (uint32_t)(dc_env("IE_DISK_AHEAD", 4.0) >= 1.0 ? dc_env("IE_DISK_AHEAD", 4.0) : 4.0);
  d->limit = limit_gb > 0 ? (uint64_t)(limit_gb * 1e9) : UINT64_MAX; /* 0: no limit */
  const uint64_t shape[5] = {DCACHE_VERSION, d->block, d->rowb, d->snapb, c->m->n_kvl};
  d->sig = fnv64(fnv64(0xcbf29ce484222325ull, sig, strlen(sig)), shape, sizeof shape);
  if (b->kv_fmt_tag) {
    const uint64_t tag = (uint64_t)b->kv_fmt_tag;
    d->sig = fnv64(d->sig, &tag, sizeof tag);
  }
  uint64_t other = 0;
  dc_lay Lk;
  dc_layout(d, 'k', 0, 0, &Lk);
  struct dirent *de;
  while ((de = readdir(dd))) {
    const char *nm = de->d_name;
    const size_t ln = strlen(nm);
    char path[1200];
    snprintf(path, sizeof path, "%s/%s", dir, nm);
    if (ln > 8 && !strcmp(nm + ln - 8, ".iec.tmp")) { /* a write the server did not finish */
      remove(path);
      continue;
    }
    if (ln < 4 || strcmp(nm + ln - 4, ".iec")) continue;
    struct stat sb;
    if (stat(path, &sb)) continue;
    uint64_t fs, fh;
    char ft = 0;
    dc_file *f = dc_add(d);
    f->bytes = (uint64_t)sb.st_size, f->used = (int64_t)sb.st_mtime * 1000000;
    d->bytes += f->bytes;
    if (ln == 39 && sscanf(nm, "%16" SCNx64 "-%c-%16" SCNx64 ".iec", &fs, &ft, &fh) == 3 && fs == d->sig &&
        (ft == 'e' || ft == 'k')) {
      f->type = ft, f->hash = fh;
      if (ft == 'k') {
        if (f->bytes != Lk.total) dc_remove(d, d->n - 1); /* damaged */
        continue;
      }
      /* a checkpoint: its header and tokens (matched against each prompt) */
      FILE *fp = fopen(path, "rb");
      dc_head h;
      int ok = fp && fread(&h, sizeof h, 1, fp) == 1 && h.magic == 0x45435049u && h.version == DCACHE_VERSION && h.sig == d->sig &&
               h.hash == fh && h.rowb == d->rowb && h.snapb == d->snapb && h.block == d->block && h.pos && h.pos <= c->g->n_ctx &&
               !(h.flags & ~DC_F_STATE16) && (!(h.flags & DC_F_STATE16) || d->snap16b);
      if (ok) {
        dc_lay L;
        dc_layout(d, 'e', h.pos, h.flags, &L);
        f->tok = malloc((size_t)h.pos * 4);
        ok = L.total == f->bytes && f->tok && fseek(fp, (long)L.tok_off, SEEK_SET) == 0 && fread(f->tok, 4, h.pos, fp) == h.pos &&
             dc_ehash(d, f->tok, h.pos) == fh;
        f->pos = h.pos, f->kind = h.kind, f->has_h = h.has_h, f->hits = h.hits, f->flags = h.flags;
      }
      if (fp) fclose(fp);
      if (!ok) dc_remove(d, d->n - 1); /* damaged */
      continue;
    }
    f->type = 'x', f->name = dc_alloc(ln + 1), other += f->bytes;
    memcpy(f->name, nm, ln + 1);
  }
  closedir(dd);
  uint32_t ne = 0, nk = 0;
  for (uint32_t i = 0; i < d->n; i++) ne += d->f[i].type == 'e', nk += d->f[i].type == 'k';
  fprintf(stderr, "disk cache: %s: %u checkpoints and %u blocks of %u positions (%.1f GB; %.1f GB of other models), limit %.0f GB; RAM tier %.1f GB, pinned %.1f GB, %s reads%s\n",
          dir, ne, nk, d->block, (d->bytes - other) / 1e9, other / 1e9, d->limit / 1e9, d->ram_limit / 1e9, d->pinned_limit / 1e9,
          d->direct ? "unbuffered" : "buffered", d->state16 ? ", state as int16" : "");
  c->dc = d;
  dc_trim(d);
  if (dc_stream_on() && b->host_alloc && Lk.total > 0) {
    uint32_t pre = d->ahead + 1u;
    if (pre > 4u) pre = 4u;
    const size_t kcap = dc_cap(Lk.total);
    for (uint32_t p = 0; p < pre && d->pinned + kcap <= d->pinned_limit; p++) {
      dc_mem pm = {0};
      pm.p = b->host_alloc(b, kcap);
      if (!pm.p) break;
      pm.cap = kcap, pm.pinned = 1;
      d->pinned += kcap;
      dc_put_l(d, pm);
    }
  }
  return 1;
}

static void dc_queue(dc_job **head, dc_job **tail, char type, uint64_t hash, dc_mem m, uint32_t pos, int pack) {
  dc_job *j = calloc(1, sizeof *j);
  if (!j) ie_die("disk cache: out of memory");
  j->type = type, j->hash = hash, j->img = m, j->pos = pos, j->pack = (uint8_t)pack, j->qb = m.len;
  if (*tail) (*tail)->next = j;
  else *head = j;
  *tail = j;
}
/* the header page, the tokens and the zero padding of a file image (the snapshot and rows are written by the caller) */
static void dc_fill(char *img, const dc_head *h, const uint32_t *toks, const dc_lay *L) {
  memset(img, 0, L->tok_off);
  memcpy(img, h, sizeof *h);
  memcpy(img + L->tok_off, toks, L->tok_len);
  memset(img + L->tok_off + L->tok_len, 0, L->snap_off - L->tok_off - L->tok_len);
  memset(img + L->snap_off + L->snap_len, 0, L->rows_off - L->snap_off - L->snap_len);
  memset(img + L->rows_off + L->rows_len, 0, L->total - L->rows_off - L->rows_len);
}
/* queue checkpoint i (its tokens are kv_toks[0 .. pos)): the blocks not on disk, then the checkpoint; the bytes queued.
 * The rows come from the GPU straight into the (pinned) file images. */
static uint64_t dc_stage(gen_ctx *c, uint32_t i) {
  dcache *d = c->dc;
  backend *b = c->b;
  const uint32_t pos = c->ck[i].pos, nb = pos / d->block, tail = pos - nb * d->block;
  const uint32_t *t = c->kv_toks;
  const uint64_t eh = dc_ehash(d, t, pos);
  uint64_t *ch = dc_alloc((size_t)(nb + 1) * 8);
  uint8_t *miss = calloc(nb + 1, 1);
  if (!miss) ie_die("disk cache: out of memory");
  dc_chain(d, t, nb, ch);
  dc_lk(d, 1);
  const int have = dc_find(d, 'e', eh);
  if (have >= 0) { /* the same tokens are there: it counts as used */
    char path[1200];
    dc_path(d, path, sizeof path, 'e', eh, NULL);
    if (!d->f[have].pending) dc_touch(path);
    d->f[have].used = d->f[have].rused = dc_now(d);
    dc_lk(d, 0);
    free(ch), free(miss);
    return 0;
  }
  for (uint32_t k = 0; k < nb; k++)
    if (dc_find(d, 'k', ch[k]) < 0) {
      dc_file *f = dc_add(d);
      f->type = 'k', f->hash = ch[k], f->pending = 1, f->used = dc_now(d), miss[k] = 1;
    }
  const uint32_t flags = d->state16 ? DC_F_STATE16 : 0;
  dc_file *f = dc_add(d);
  f->type = 'e', f->hash = eh, f->pending = 1, f->used = dc_now(d);
  f->pos = pos, f->kind = (uint32_t)c->ck[i].kind, f->has_h = c->ck[i].h, f->flags = flags, f->tok = dc_alloc((size_t)pos * 4);
  memcpy(f->tok, t, (size_t)pos * 4);
  dc_lk(d, 0);
  /* the copies (the GPU's rows, the snapshot), outside the lock */
  dc_job *qh = NULL, *qt = NULL;
  uint64_t staged = 0;
  dc_lay Lk, Le;
  dc_layout(d, 'k', 0, 0, &Lk), dc_layout(d, 'e', pos, 0, &Le);
  for (uint32_t k = 0; k < nb; k++) {
    if (!miss[k]) continue;
    dc_mem m = dc_get(d, Lk.total);
    const dc_head h = {0x4B435049u, DCACHE_VERSION, d->sig, ch[k], (k + 1) * d->block, 0, 0, d->block, d->rowb, d->snapb, 0, 0};
    dc_fill(m.p, &h, t + (size_t)k * d->block, &Lk);
    b->kv_io(b, 1, k * d->block, d->block, m.p + Lk.rows_off);
    dc_queue(&qh, &qt, 'k', ch[k], m, h.pos, 0);
    staged += m.len;
  }
  dc_mem m = dc_get(d, Le.total);
  const dc_head h = {0x45435049u, DCACHE_VERSION, d->sig, eh, pos, (uint32_t)c->ck[i].kind, c->ck[i].h, d->block, d->rowb, d->snapb, 0, flags};
  dc_fill(m.p, &h, t, &Le);
  b->snap_io(b, c->ck[i].slot, 1, m.p + Le.snap_off, pos);
  b->kv_io(b, 1, nb * d->block, tail, m.p + Le.rows_off);
  dc_queue(&qh, &qt, 'e', eh, m, pos, flags != 0);
  staged += m.len;
  free(ch), free(miss);
  dc_lk(d, 1);
  if (d->qt) d->qt->next = qh;
  else d->qh = qh;
  d->qt = qt, d->qbytes += staged;
  dc_lk(d, 0);
  return staged;
}

/* write the first queued file: 1, or 0 if the queue is empty (the disk thread, or the engine thread when it waits) */
static int dc_write_one(dcache *d) {
  {
    dc_lk(d, 1);
    dc_job *j = d->qh;
    if (j) {
      d->qh = j->next, d->inflight++;
      if (!d->qh) d->qt = NULL;
      if (!d->w_n && !d->w_bytes) d->w_t0 = now_ms();
    }
    dc_lk(d, 0);
    if (!j) return 0;
    if (j->pack) dc_pack16(d, j);
    char path[1200], tmp[1210];
    dc_path(d, path, sizeof path, j->type, j->hash, NULL);
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    int ok = dc_write(d, tmp, j->img.p, j->img.len);
    if (!ok || rename(tmp, path)) remove(tmp), ok = 0;
    const uint64_t sz = j->img.len;
    if (ok && j->img.pinned && d->ram_limit) { /* the tier keeps plain memory once it holds its share of the pinned buffers */
      dc_lk(d, 1);
      const int full = d->ram_pinned + j->img.cap > d->pinned_limit / 2u;
      dc_lk(d, 0);
      if (full) dc_unpin(d, &j->img);
    }
    dc_lk(d, 1);
    const int i = dc_find(d, j->type, j->hash);
    if (i >= 0 && ok) {
      dc_file *f = &d->f[i];
      f->pending = 0, f->bytes = sz, d->bytes += sz, f->used = dc_now(d);
      if (j->type == 'e') f->hits = 0, f->flags = j->pack ? DC_F_STATE16 : 0;
      dc_ram_adopt_l(d, f, &j->img);
    } else if (i >= 0) dc_remove(d, (uint32_t)i);
    d->qbytes -= j->qb, d->w_n += ok && j->type == 'e', d->w_bytes += ok ? sz : 0, d->inflight--;
    if (!ok) fprintf(stderr, "disk cache: cannot write %s\n", path);
    if (!d->qh && !d->inflight) {
      dc_trim(d);
      if (d->w_n)
        fprintf(stderr, "disk cache: %u checkpoints written (%.0f MB) in %.0f ms; %.1f of %.0f GB used, %.1f GB in RAM\n", d->w_n,
                d->w_bytes / 1e6, now_ms() - d->w_t0, d->bytes / 1e9, d->limit / 1e9, d->ram_bytes / 1e9);
      d->w_n = 0, d->w_bytes = 0;
    }
    dc_put_l(d, j->img);
    dc_lk(d, 0);
    free(j);
  }
  return 1;
}
void gen_disk_write(gen_ctx *c) {
  if (c->dc)
    while (dc_write_one(c->dc)) {}
}
static void dc_nap(void) {
#ifdef _WIN32
  Sleep(1);
#else
  const struct timespec ts = {0, 1000000};
  nanosleep(&ts, NULL);
#endif
}


/* the checkpoint on disk to continue a prompt from: the furthest one past S, at most at lim, whose tokens the prompt
 * starts with and whose blocks are all there (with need_h, one that holds the MTP h input): 1 and its hash, position,
 * kind and has_h; 0 if none */
/* checkpoint i (whose tokens the prompt starts with) and the blocks before it: 1 all written, 0 one is still queued or
 * being written, -1 one is missing (locked) */
static int dc_state_l(const dcache *d, uint32_t i, const uint32_t *prompt, uint64_t *ch) {
  const uint32_t nb = d->f[i].pos / d->block;
  int ready = !d->f[i].pending;
  dc_chain(d, prompt, nb, ch);
  for (uint32_t k = 0; k < nb; k++) {
    const int j = dc_find(d, 'k', ch[k]);
    if (j < 0) return -1;
    ready &= !d->f[j].pending;
  }
  return ready;
}
/* the furthest written checkpoint past S, at most at lim, whose tokens the prompt starts with; *later: a further one
 * still being written, or -1 (locked) */
static int dc_scan_l(const dcache *d, const uint32_t *prompt, uint32_t lim, uint32_t S, int need_h, uint64_t *ch, int *later) {
  int best = -1;
  *later = -1;
  for (uint32_t i = 0; i < d->n; i++) {
    const dc_file *f = &d->f[i];
    const uint32_t bp = best >= 0 ? d->f[best].pos : 0, lp = *later >= 0 ? d->f[*later].pos : 0;
    if (f->type != 'e' || f->pos <= S || f->pos > lim || (need_h && !f->has_h) || f->pos <= (bp > lp ? bp : lp)) continue;
    if (memcmp(f->tok, prompt, (size_t)f->pos * 4)) continue;
    const int s = dc_state_l(d, i, prompt, ch);
    if (s > 0) best = (int)i;
    else if (!s) *later = (int)i;
  }
  return best;
}

/* Find an observed change within the system prompt, using the token IDs
 * already indexed for reusable prefixes and earlier prompt ends. A shorter all-matching checkpoint
 * alone tells us nothing about where the next token changes. (Locked.) */
static uint32_t dc_prefix_divergence_l(const dcache *d, const uint32_t *prompt, uint32_t sys_end, uint32_t start) {
  const uint32_t min = ck_prefix_min(d->min);
  uint32_t best = start >= min ? start : min - 1;
  const uint32_t floor = best;
  for (uint32_t i = 0; i < d->n; i++) {
    const dc_file *f = &d->f[i];
    if (f->type != 'e' || (f->kind != GEN_CK_PREFIX && f->kind != GEN_CK_PROMPT)) continue;
    uint32_t hi = f->pos < sys_end ? f->pos : sys_end;
    uint32_t lo = best + 1;
    if (lo >= hi || memcmp(f->tok, prompt, (size_t)lo * sizeof *prompt) ||
        !memcmp(f->tok + lo, prompt + lo, (size_t)(hi - lo) * sizeof *prompt)) continue;
    while (hi - lo > 1) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (memcmp(f->tok + lo, prompt + lo, (size_t)(mid - lo) * sizeof *prompt)) hi = mid;
      else lo = mid;
    }
    best = lo;
  }
  return best > floor ? best : 0;
}

/* the checkpoint on disk to continue a prompt from: the furthest one past S, at most at lim, whose tokens the prompt
 * starts with and whose blocks are all there (with need_h, one that holds the MTP h input): 1 and its hash, position,
 * kind and has_h; 0 if none. When the furthest is still being written (a quick return to the conversation the last
 * request left), the engine thread writes the queue too until its files are on disk. */
static int dc_best(dcache *d, const uint32_t *prompt, uint32_t lim, uint32_t S, int need_h, uint64_t *eh, uint32_t *pos,
                   uint32_t *kind, uint32_t *has_h, uint32_t sys_end, uint32_t *learned) {
  uint64_t *ch = dc_alloc((size_t)d->max_nb * 8);
  int best = -1, later = -1;
  uint64_t lh = 0;
  uint32_t lpos = 0;
  for (int round = 0; round < 2; round++) {
    dc_lk(d, 1);
    best = dc_scan_l(d, prompt, lim, S, need_h, ch, &later);
    if (!round && sys_end) {
      uint32_t matched = best >= 0 ? d->f[best].pos : S;
      if (later >= 0 && d->f[later].pos > matched) matched = d->f[later].pos;
      *learned = dc_prefix_divergence_l(d, prompt, sys_end, matched);
    }
    if (best >= 0) *eh = d->f[best].hash, *pos = d->f[best].pos, *kind = d->f[best].kind, *has_h = d->f[best].has_h;
    if (later >= 0) lh = d->f[later].hash, lpos = d->f[later].pos;
    dc_lk(d, 0);
    if (later < 0 || round) break;
    const double t0 = now_ms();
    for (int s = 0; !s && now_ms() - t0 < 60000.0;) {
      dc_lk(d, 1);
      const int i = dc_find(d, 'e', lh);
      s = i < 0 ? -1 : dc_state_l(d, (uint32_t)i, prompt, ch);
      dc_lk(d, 0);
      if (!s && !dc_write_one(d)) dc_nap(); /* the queue is empty: the disk thread is writing the last of them */
    }
    fprintf(stderr, "disk cache: waited %.0f ms for the files of the checkpoint at %u\n", now_ms() - t0, lpos);
  }
  free(ch);
  return best >= 0;
}
static void dc_drop(dcache *d, uint64_t eh) {
  dc_lk(d, 1);
  const int i = dc_find(d, 'e', eh);
  if (i >= 0 && !d->f[i].pending) dc_remove(d, (uint32_t)i);
  dc_lk(d, 0);
}

/* ---- reading a checkpoint: its file and its blocks, each into one buffer, a few reads ahead of the copies to the GPU ---- */
typedef struct { /* a file to read */
  char type;
  uint64_t hash;
  size_t bytes;
  uint8_t ram, issued, waited, ok; /* ram: the image is in the RAM tier (m is borrowed) */
  dc_mem m;
  dc_rd rd;
} dc_item;
static void dc_issue(dcache *d, dc_item *it) {
  if (it->issued) return;
  it->issued = 1;
  if (it->ram) return;
  char path[1200];
  dc_path(d, path, sizeof path, it->type, it->hash, NULL);
  const int plain_e = (it->type == 'e') && d->b->snap_in_ram && dc_stream_on();
  it->m = plain_e ? dc_get_plain(it->bytes) : dc_get(d, it->bytes);
  dc_rd_start(d, &it->rd, path, it->m.p, it->bytes);
}
static int dc_item_wait(dc_item *it) {
  if (it->ram) return 1;
  if (!it->waited) it->waited = 1, it->ok = dc_rd_wait(&it->rd);
  return it->ok;
}
/* done with the image: the RAM tier takes it (adopt) or it goes back to the pool (a failed read first waits) */
static void dc_item_release(dcache *d, dc_item *it, int adopt) {
  if (it->ram || !it->issued || !it->m.p) return;
  dc_item_wait(it);
  if (adopt && it->ok) {
    dc_lk(d, 1);
    const int j = dc_find(d, it->type, it->hash);
    if (j >= 0) dc_ram_adopt_l(d, &d->f[j], &it->m);
    dc_lk(d, 0);
  }
  dc_put(d, it->m);
  memset(&it->m, 0, sizeof it->m);
}
/* the header of a file image is that of the file wanted */
static int dc_head_ok(const dcache *d, const dc_mem *m, char type, uint64_t hash) {
  if (m->len < DC_ALIGN) return 0;
  const dc_head *h = (const dc_head *)m->p;
  return h->magic == (type == 'e' ? 0x45435049u : 0x4B435049u) && h->version == DCACHE_VERSION && h->sig == d->sig &&
         h->hash == hash && h->rowb == d->rowb && h->snapb == d->snapb && h->block == d->block && !(h->flags & ~DC_F_STATE16) &&
         (!(h->flags & DC_F_STATE16) || d->snap16b);
}
/* read checkpoint eh (at pos) into snapshot slot and the KV rows before it: 0, or -1 (nothing changed) or -2 (the KV
 * rows were partly overwritten) if a file is missing or damaged */
static int dc_load(gen_ctx *c, uint64_t eh, uint32_t pos, const uint32_t *prompt, int slot) {
  dcache *d = c->dc;
  backend *b = c->b;
  const double t0 = now_ms();
  const uint32_t nb = pos / d->block, tail = pos - nb * d->block;
  uint64_t *ch = dc_alloc((size_t)(nb + 1) * 8);
  dc_item *it = calloc(nb + 1, sizeof *it);
  if (!it) ie_die("disk cache: out of memory");
  dc_chain(d, prompt, nb, ch);
  int r = 0;
  uint32_t nbusy = 0, from_ram = 0;
  uint64_t read_bytes = 0;
  dc_lk(d, 1); /* in use now: the last to be deleted, and not deleted while they are read */
  const int64_t now = dc_now(d);
  for (uint32_t x = 0; x <= nb && !r; x++) {
    dc_item *t = &it[x];
    t->type = x ? 'k' : 'e', t->hash = x ? ch[x - 1] : eh;
    const int j = dc_find(d, t->type, t->hash);
    if (j < 0 || d->f[j].pending) {
      r = -1;
      break;
    }
    dc_file *f = &d->f[j];
    f->busy = 1, f->used = f->rused = now, t->bytes = f->bytes, nbusy = x + 1;
    if (f->img.p) t->ram = 1, t->m = f->img, from_ram++;
    else read_bytes += f->bytes;
  }
  dc_lk(d, 0);
  dc_lay Lk, Le = {0};
  dc_layout(d, 'k', 0, 0, &Lk);
  dc_head hd = {0};
  uint32_t hits = 0;
  if (!r) {
    const int stream = dc_stream_on();
    if (stream && pos) b->kv_io(b, 0, pos, 0, NULL); /* pre-map VMM once before overlapped reads/H2D */
    if (stream) {
      /* Issue k[0..ahead-1] and e in parallel; stream k blocks to GPU WHILE large e finishes reading from NVMe! */
      for (uint32_t x = 1; x <= nb && x <= d->ahead; x++) dc_issue(d, &it[x]);
      dc_issue(d, &it[0]);
      for (uint32_t k = 0; k < nb && !r; k++) {
        dc_item *t = &it[1 + k];
        dc_issue(d, t);
        const int ok = dc_item_wait(t) && dc_head_ok(d, &t->m, 'k', ch[k]) && t->m.len == Lk.total &&
                       !memcmp(t->m.p + Lk.tok_off, prompt + (size_t)k * d->block, (size_t)d->block * 4);
        if (!ok) r = k ? -2 : -1;
        else b->kv_io(b, 0, k * d->block, d->block, t->m.p + Lk.rows_off);
        dc_item_release(d, t, ok);
        if (k + 1 + d->ahead <= nb) dc_issue(d, &it[k + 1 + d->ahead]);
      }
      if (!r) {
        dc_item *e = &it[0];
        if (!dc_item_wait(e) || !dc_head_ok(d, &e->m, 'e', eh)) r = nb ? -2 : -1;
        else {
          memcpy(&hd, e->m.p, sizeof hd);
          dc_layout(d, 'e', pos, hd.flags, &Le);
          if (hd.pos != pos || e->m.len != Le.total) r = nb ? -2 : -1;
        }
      }
    } else {
      for (uint32_t x = 0; x <= nb && x <= d->ahead; x++) dc_issue(d, &it[x]); /* the checkpoint, then the first blocks */
      dc_item *e = &it[0];
      if (!dc_item_wait(e) || !dc_head_ok(d, &e->m, 'e', eh)) r = -1;
      else {
        memcpy(&hd, e->m.p, sizeof hd);
        dc_layout(d, 'e', pos, hd.flags, &Le);
        if (hd.pos != pos || e->m.len != Le.total) r = -1;
      }
      for (uint32_t k = 0; k < nb && !r; k++) {
        dc_item *t = &it[1 + k];
        dc_issue(d, t);
        const int ok = dc_item_wait(t) && dc_head_ok(d, &t->m, 'k', ch[k]) && t->m.len == Lk.total &&
                       !memcmp(t->m.p + Lk.tok_off, prompt + (size_t)k * d->block, (size_t)d->block * 4);
        if (!ok) r = k ? -2 : -1;
        else b->kv_io(b, 0, k * d->block, d->block, t->m.p + Lk.rows_off);
        dc_item_release(d, t, ok);
        if (k + 1 + d->ahead <= nb) dc_issue(d, &it[k + 1 + d->ahead]);
      }
    }
  }
  if (!r) {
    const dc_item *e = &it[0];
    if (tail > 0) {
      if (!e->m.pinned && dc_stream_on() && d->b->host_alloc && Le.rows_len <= Lk.total) {
        dc_mem tm = dc_get(d, Lk.total); /* hits pre-warmed k-block pinned pool (0 hipHostMalloc!) */
        memcpy(tm.p, e->m.p + Le.rows_off, Le.rows_len);
        b->kv_io(b, 0, nb * d->block, tail, tm.p);
        dc_put(d, tm);
      } else {
        b->kv_io(b, 0, nb * d->block, tail, e->m.p + Le.rows_off);
      }
    }
    if (hd.flags & DC_F_STATE16) {
      dc_mem tmp = d->b->snap_in_ram && dc_stream_on() ? dc_get_plain(d->snapb) : dc_get(d, d->snapb);
      dc_unpack16(d, e->m.p + Le.snap_off, tmp.p);
      b->snap_io(b, slot, 0, tmp.p, pos);
      dc_put(d, tmp);
    } else b->snap_io(b, slot, 0, e->m.p + Le.snap_off, pos);
  }
  for (uint32_t x = 0; x <= nb; x++) dc_item_release(d, &it[x], !r); /* reads still in flight are waited for */
  dc_lk(d, 1);
  for (uint32_t x = 0; x < nbusy; x++) {
    const int j = dc_find(d, it[x].type, it[x].hash);
    if (j >= 0) d->f[j].busy = 0;
  }
  if (!r) {
    const int j = dc_find(d, 'e', eh);
    if (j >= 0) hits = ++d->f[j].hits;
  }
  dc_lk(d, 0);
  if (!r) {
    char path[1200];
    dc_path(d, path, sizeof path, 'e', eh, NULL);
    dc_hit_write(path, hits);
    dc_touch(path);
    fprintf(stderr, "disk cache: %u of %u files from RAM, %.0f MB from the disk, %.0f ms\n", from_ram, nb + 1, read_bytes / 1e6, now_ms() - t0);
  }
  free(ch), free(it);
  return r;
}
/* ==== END disk cache storage ==== */

/* ---- the disk cache policy: which checkpoints go to disk, and when ----
 * Not after every request: a checkpoint is written (staged: copied to host memory for the disk thread) when
 *   (a) a request leaves the cached conversation: the checkpoints past the tokens it shares are about to be dropped,
 *   (b) nothing has come for a while (ie-serve: IE_DISK_IDLE_S), or
 *   (c) a full cache drops it to reuse its slot.
 * Most of a long conversation's follow-up requests then write nothing: their ends supersede each other in memory.
 * Admitted for a file: the places where prompts differ (SPLIT, MARK: what later prompts resume from), the latest
 * PROMPT and ANSWER ends (the same conversation continues from them), and the deepest periodic checkpoint below the
 * first of those (what a prompt that differs earlier than the marks resumes from). The other periodic ones are
 * the write-volume of the uniform grid and are rarely the nearest resume point, so they stay in memory only. */
static const char *const DP_KIND[] = {"prompt end", "answer end", "split", "periodic", "disk", "mark", "prefix"};
/* IE_DISK_ADMIT=all: every checkpoint of IE_DISK_MIN tokens or more (as when each request was written) */
static int dp_all(void) {
  const char *e = getenv("IE_DISK_ADMIT");
  return e && !strcmp(e, "all");
}
/* does checkpoint i get a file? */
static int dp_admit(const gen_ctx *c, uint32_t i) {
  const dcache *d = c->dc;
  const uint32_t pos = c->ck[i].pos;
  if (!d || pos > c->kv_n) return 0;
  if (c->ck[i].kind == GEN_CK_PREFIX) return pos >= ck_prefix_min(d->min);
  if (pos < d->min) return 0;
  if (dp_all() || c->ck[i].kind != GEN_CK_EVERY) return 1;
  uint32_t place = UINT32_MAX, head = UINT32_MAX, deep = 0; /* the first SPLIT/MARK, the first head, the deepest periodic below */
  for (uint32_t j = 0; j < c->n_ck; j++) {
    const int k = c->ck[j].kind;
    if (ck_rank(k) >= 2) place = c->ck[j].pos < place ? c->ck[j].pos : place;
    else if (k == GEN_CK_PROMPT || k == GEN_CK_ANSWER) head = c->ck[j].pos < head ? c->ck[j].pos : head;
  }
  if (place == UINT32_MAX) place = head;
  for (uint32_t j = 0; j < c->n_ck; j++)
    if (c->ck[j].kind == GEN_CK_EVERY && c->ck[j].pos < place && c->ck[j].pos > deep) deep = c->ck[j].pos;
  return pos == deep;
}
/* queue checkpoint i; the bytes (0: it was on disk already) */
static uint64_t dp_stage1(gen_ctx *c, uint32_t i) {
  dcache *d = c->dc;
  dc_lk(d, 1);
  const uint64_t q = d->qbytes;
  dc_lk(d, 0);
  if (q > d->queue_limit) gen_disk_write(c); /* the writer is behind: help */
  c->ck[i].disk = 1;
  return dc_stage(c, i);
}

/* Copy newly computed prefixes after generation. They are already usable in
 * memory during this request, so staging to host memory can wait past TTFT. */
static void dp_save_prefixes(gen_ctx *c, const char *client, const uint32_t *mk, const int *kind,
                             uint32_t n, uint32_t learned) {
  if (!c->dc || !client) return;
  for (uint32_t j = 0; j < n; j++) {
    if (kind[j] != GEN_CK_PREFIX) continue;
    const int i = ck_at(c, mk[j]);
    if (i < 0 || c->ck[i].disk || !dp_admit(c, (uint32_t)i) || !dp_stage1(c, (uint32_t)i)) continue;
    fprintf(stderr, "disk cache: queued %s%s prefix checkpoint at %u tokens\n", mk[j] == learned ? "learned " : "", client, mk[j]);
    if (c->disk_kick) c->disk_kick();
    else gen_disk_write(c);
  }
}
/* queue the admitted checkpoints past position lo that are not queued yet, in order of position (a conversation's blocks
 * are copied once); why: the line in the log */
static uint32_t dp_flush(gen_ctx *c, const char *why, uint32_t lo) {
  dcache *d = c->dc;
  if (!d) return 0;
  const double t0 = now_ms();
  uint64_t staged = 0;
  uint32_t n = 0;
  for (;;) {
    int i = -1;
    for (uint32_t j = 0; j < c->n_ck; j++)
      if (!c->ck[j].disk && c->ck[j].pos > lo && dp_admit(c, j) && (i < 0 || c->ck[j].pos < c->ck[i].pos)) i = (int)j;
    if (i < 0) break;
    const uint64_t s = dp_stage1(c, (uint32_t)i);
    staged += s, n += s > 0;
  }
  if (!n) return 0;
  fprintf(stderr, "disk cache: %s: %u checkpoints copied for writing (%.0f MB) in %.0f ms\n", why, n, staged / 1e6, now_ms() - t0);
  if (!d->lock) gen_disk_write(c); /* one thread: write them now */
  else if (c->disk_kick) c->disk_kick(); /* the disk thread starts now, while this request runs */
  return n;
}
uint32_t gen_disk_save(gen_ctx *c) { return dp_flush(c, "after the request", 0); }
uint32_t gen_disk_idle(gen_ctx *c) { return dp_flush(c, "idle", 0); }
uint32_t gen_disk_pending(const gen_ctx *c) {
  uint32_t n = 0;
  for (uint32_t j = 0; c->dc && j < c->n_ck; j++) n += !c->ck[j].disk && dp_admit(c, j);
  return n;
}
/* ck_slot is about to reuse the slot of checkpoint i */
static void dp_evict(gen_ctx *c, uint32_t i) {
  dcache *d = c->dc;
  if (!d || c->ck[i].disk || !dp_admit(c, i)) return;
  const double t0 = now_ms();
  const uint32_t pos = c->ck[i].pos;
  const char *kn = DP_KIND[c->ck[i].kind];
  const uint64_t s = dp_stage1(c, i);
  if (!s) return;
  fprintf(stderr, "disk cache: evict: 1 checkpoint (%s at %u) copied for writing (%.0f MB) in %.0f ms\n", kn, pos, s / 1e6,
          now_ms() - t0);
  if (!d->lock) gen_disk_write(c);
  else if (c->disk_kick) c->disk_kick();
}
/* the request continues the cached conversation when a PROMPT or ANSWER end is inside the tokens it shares (an edited
 * or regenerated last message, the earlier reasoning stripped from the history: the checkpoints past L belong to the
 * branch that was dropped); otherwise it is another conversation and the cached one is written first */
static int dp_continues(const gen_ctx *c, uint32_t L) {
  for (uint32_t j = 0; j < c->n_ck; j++)
    if ((c->ck[j].kind == GEN_CK_PROMPT || c->ck[j].kind == GEN_CK_ANSWER) && c->ck[j].pos && c->ck[j].pos <= L) return 1;
  return 0;
}

/* The distances of the places where system prompts first differed (SPLIT checkpoints): sys_end minus the position,
 * the last 16 in policy-splits.txt in the disk cache directory, so a fresh system prompt gets checkpoints there. */
#define DP_FILE "policy-splits.txt"
static void dp_learn_load(gen_ctx *c) {
  if (c->pol_loaded || !c->dc) return;
  c->pol_loaded = 1;
  char path[1100];
  snprintf(path, sizeof path, "%s/" DP_FILE, c->dc->dir);
  FILE *fp = fopen(path, "r");
  if (!fp) return;
  unsigned v;
  while (fscanf(fp, "%u", &v) == 1) {
    if (!v || v > 16384u) continue;
    if (c->pol_n == 16) memmove(c->pol_dist, c->pol_dist + 1, 15 * sizeof c->pol_dist[0]), c->pol_n--;
    c->pol_dist[c->pol_n++] = v;
  }
  fclose(fp);
}
static void dp_learn_add(gen_ctx *c, uint32_t dist) {
  if (!c->dc || !dist || dist > 16384u) return;
  uint32_t w = 0;
  for (uint32_t i = 0; i < c->pol_n; i++) /* about the same distance: the newer one replaces it */
    if (!near_pos(c->pol_dist[i], dist, 32u)) c->pol_dist[w++] = c->pol_dist[i];
  if (w == 16) memmove(c->pol_dist, c->pol_dist + 1, 15 * sizeof c->pol_dist[0]), w--;
  c->pol_dist[w++] = dist, c->pol_n = w;
  char path[1100];
  snprintf(path, sizeof path, "%s/" DP_FILE, c->dc->dir);
  FILE *fp = fopen(path, "w");
  if (!fp) return;
  for (uint32_t i = 0; i < c->pol_n; i++) fprintf(fp, "%u\n", c->pol_dist[i]);
  fclose(fp);
}
/* the positions of this prompt where a later one is likely to differ and that are not checkpoints yet: the system
 * message's end minus the distances (the fixed list, then the 3 latest learned; snapped to line starts), and the start of
 * the last user message. At most 16 in xm. */
static uint32_t dp_places(const gen_ctx *c, const gen_params *p, const uint32_t *prompt, uint32_t n_prompt, uint32_t *xm) {
  uint32_t dist[16], nd = ck_mark_dists(dist), n = 0;
  if (!nd) return 0;
  const uint32_t mn = c->dc ? c->dc->min : 1024u;
  for (uint32_t i = c->pol_n; i-- > 0 && nd < 11u;) {
    uint32_t k = 0;
    while (k < nd && dist[k] != c->pol_dist[i]) k++;
    if (k == nd) dist[nd++] = c->pol_dist[i];
  }
  if (p->sys_end && p->sys_end < n_prompt)
    for (uint32_t i = 0; i < nd; i++) {
      if (dist[i] >= p->sys_end) continue;
      const uint32_t a = line_start(c, prompt, p->sys_end - dist[i]);
      if (a >= mn && a < p->sys_end && n < 15u) xm[n++] = a;
    }
  if (p->last_user >= mn && p->last_user < n_prompt) xm[n++] = p->last_user;
  for (uint32_t i = 1; i < n; i++) { /* in order of position */
    const uint32_t v = xm[i];
    uint32_t j = i;
    for (; j > 0 && xm[j - 1] > v; j--) xm[j] = xm[j - 1];
    xm[j] = v;
  }
  return n;
}

static void gen_setup_mrope(gen_ctx *c, const gen_params *p, uint32_t n_prompt) {
  backend *b = c->b;
  const model *m = c->m;
  const graph *g = c->g;
  if (!b->set_rope) return;
  const uint32_t h2 = m->n_rot / 2u;
  if (!p->n_imgs) {
    if (c->rope_custom_n) {
      b->set_rope(b, 0, c->rope_custom_n, g->rope_cos, g->rope_sin);
      c->rope_custom_n = 0;
    }
    return;
  }
  uint32_t n_rope = n_prompt + p->max_new + spec_room(p->draft) + 16u;
  if (n_rope > g->n_ctx) n_rope = g->n_ctx;
  float *cs = malloc((size_t)n_rope * h2 * 4u);
  float *sn = malloc((size_t)n_rope * h2 * 4u);
  double *inv_f = malloc((size_t)h2 * sizeof(double));
  if (!cs || !sn || !inv_f) ie_die("mrope: out of memory");
  for (uint32_t i = 0; i < h2; i++) inv_f[i] = pow((double)m->rope_base, -2.0 * (double)i / (double)m->n_rot);
  uint32_t s_h = m->mrope_sec[1], s_w = m->mrope_sec[2];
  if (!s_h && !s_w && h2 == 32u) s_h = 11u, s_w = 10u; /* Qwen3.5/3.8 default [11, 11, 10] */

  uint32_t plog = 0, ik = 0;
  for (uint32_t kv = 0; kv < n_prompt;) {
    if (ik < p->n_imgs && kv == p->imgs[ik].pos) {
      const gen_img *im = &p->imgs[ik++];
      for (uint32_t my = 0; my < im->ny && kv < n_prompt; my++) {
        for (uint32_t mx = 0; mx < im->nx && kv < n_prompt; mx++) {
          const uint32_t pt = plog, ph = plog + my, pw = plog + mx;
          float *rc = cs + (size_t)kv * h2, *rs = sn + (size_t)kv * h2;
          for (uint32_t i = 0; i < h2; i++) {
            const uint32_t pi = (i % 3u == 1u && i < 3u * s_h) ? ph
                              : (i % 3u == 2u && i < 3u * s_w) ? pw : pt;
            const double th = (double)pi * inv_f[i];
            rc[i] = (float)cos(th);
            rs[i] = (float)sin(th);
          }
          kv++;
        }
      }
      plog += im->ny > im->nx ? im->ny : im->nx;
    } else {
      memcpy(cs + (size_t)kv * h2, g->rope_cos + (size_t)plog * h2, (size_t)h2 * 4u);
      memcpy(sn + (size_t)kv * h2, g->rope_sin + (size_t)plog * h2, (size_t)h2 * 4u);
      kv++, plog++;
    }
  }
  for (uint32_t kv = n_prompt; kv < n_rope; kv++, plog++) {
    memcpy(cs + (size_t)kv * h2, g->rope_cos + (size_t)plog * h2, (size_t)h2 * 4u);
    memcpy(sn + (size_t)kv * h2, g->rope_sin + (size_t)plog * h2, (size_t)h2 * 4u);
  }
  b->set_rope(b, 0, n_rope, cs, sn);
  if (c->rope_custom_n > n_rope)
    b->set_rope(b, n_rope, c->rope_custom_n - n_rope, g->rope_cos + (size_t)n_rope * h2, g->rope_sin + (size_t)n_rope * h2);
  c->rope_custom_n = n_rope;
  free(inv_f), free(sn), free(cs);
}

static uint32_t img_prefix_limit(const gen_img *a, uint32_t na, const gen_img *b, uint32_t nb, uint32_t L) {
  uint32_t ia = 0, ib = 0;
  while (ia < na || ib < nb) {
    const uint32_t pa = ia < na ? a[ia].pos : UINT32_MAX;
    const uint32_t pb = ib < nb ? b[ib].pos : UINT32_MAX;
    const uint32_t pmin = pa < pb ? pa : pb;
    if (pmin >= L) break;
    if (ia >= na || ib >= nb || a[ia].pos != b[ib].pos || a[ia].n_tok != b[ib].n_tok ||
        a[ia].nx != b[ib].nx || a[ia].ny != b[ib].ny || a[ia].hash != b[ib].hash) {
      L = pmin;
      break;
    }
    ia++, ib++;
  }
  return L;
}

static int ie_defer_cksave_mode(void) {
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("IE_DEFER_CKSAVE");
    v = (e && *e) ? atoi(e) : 1; /* R38 (H42): default 1 on Inkling-2; 0 opts out */
  }
  return v;
}

typedef struct {
  gen_ctx *c;
  gen_emit inner_emit;
  void *inner_ctx;
  uint32_t ck_pos;
  int ck_hbuf;
  int pending;
} defer_ck_ctx;

static int defer_ck_emit(void *vctx, const uint32_t *toks, uint32_t n) {
  defer_ck_ctx *d = (defer_ck_ctx *)vctx;
  const int stop = d->inner_emit ? d->inner_emit(d->inner_ctx, toks, n) : 0;
  if (d->pending) {
    d->pending = 0;
    ck_save(d->c, d->ck_pos, GEN_CK_PROMPT, d->ck_hbuf);
  }
  return stop;
}

void gen_run(gen_ctx *c, const uint32_t *prompt, uint32_t n_prompt, const gen_params *p, gen_stats *st) {
  backend *b = c->b;
  const graph *g = c->g;
  sampler *sp = &c->sp;
  const uint32_t V = model_vocab(c->m);
  sp->temp = p->temp, sp->top_p = p->top_p, sp->top_k = p->top_k;
  sp->dry_mult = p->temp > 0.0f ? p->dry_mult : 0.0f, sp->dry_base = p->dry_base, sp->dry_allowed = p->dry_allowed;
  sp->dry_reset = p->dry_reset;
  sp->dry_pause_open_plus_one = p->dry_pause_open_plus_one && p->dry_pause_open_plus_one - 1u < V
                                    ? p->dry_pause_open_plus_one : 0u;
  sp->dry_pause_close_plus_one = p->dry_pause_close_plus_one && p->dry_pause_close_plus_one - 1u < V
                                     ? p->dry_pause_close_plus_one : 0u;
  sp->dry_pause_after_reset = p->dry_pause_after_reset && sp->dry_reset >= 0;
  sp->reasoning_end_plus_one = p->reasoning_budget && p->dry_reset >= 0 && p->dry_reset < V
                                  ? (uint32_t)p->dry_reset + 1u : 0u;
  sp->reasoning_left = p->reasoning_budget;
  sp->reasoning_limited = 0;
  dry_begin(sp);
  sp->rng = p->seed * 0x9E3779B97F4A7C15ull + 1; /* as ie-run --seed */
  memset(st, 0, sizeof *st);
  st->n_prompt = n_prompt;
  c->req_tick = c->tick + 1u; /* the checkpoints this request makes or uses get ticks from here on */

  /* the prompt: positions 0 .. n_prompt - 1 (position 0 starts a new
   * sequence: the linear attention state and the conv ring read as zero).
   * The prompt cache: L = the prompt's first tokens that equal the cached ones; continue from the latest
   * checkpoint at most L (and short of the last prompt token, which is always run) */
  const int mtp = p->draft > 0 && g->mtp_h >= 0;
  const int cache = p->cache && b->snap;
  const int hbuf = mtp ? g->mtp_h : -1;
  gen_setup_mrope(c, p, n_prompt);
  uint32_t S = 0, L = 0;
  int use = -1;
  st->cache_hit = -1;
  const double t0 = now_ms();
  /* Ephemeral side-request cache immunity (Inkling-2, IE_EPHEMERAL_CACHE=0 off): when a short divergent request
   * arrives while a long conversation (>= 2048 tokens) is cached, stash ONLY the overwritten global KV prefix
   * [0 .. stash_n) in host RAM (36 KiB/pos, 0 extra VRAM) and restore it afterward so the long prefix survives. */
  int ephem = 0;
  uint32_t stash_n = 0;
  void *stash_buf = NULL;
  double stash_ms = 0.0;
  if (cache) {
    while (L < c->kv_n && L < n_prompt && prompt[L] == c->kv_toks[L]) L++;
    if (c->kv_n_imgs || p->n_imgs) L = img_prefix_limit(c->kv_imgs, c->kv_n_imgs, p->imgs, p->n_imgs, L);
    uint32_t best_ck_pos = 0;
    for (uint32_t i = 0; i < c->n_ck; i++)
      if (c->ck[i].pos > best_ck_pos) best_ck_pos = c->ck[i].pos;
    const char *ee = getenv("IE_EPHEMERAL_CACHE");
    const int ephem_on = c->m->arch == ARCH_INKLING2 && !(ee && ee[0] == '0') && b->run && b->kv_io && b->kv_rowb > 0 &&
                         !p->n_imgs && !c->kv_n_imgs;
    const char *eme = getenv("IE_EPHEMERAL_MAX");
    const uint32_t ephem_max = eme && atoi(eme) > 0 ? (uint32_t)atoi(eme) : 2048u;
    const uint32_t ephem_cap = ephem_max > 4096u ? ephem_max : 4096u;
    if (ephem_on && best_ck_pos >= 2048u && L < best_ck_pos / 4u && n_prompt <= ephem_max && n_prompt < best_ck_pos / 2u && p->max_new <= 1024u) {
      uint64_t req_max = (uint64_t)n_prompt + (uint64_t)p->max_new + (uint64_t)spec_room(p->draft) + 16ull;
      stash_n = req_max < (uint64_t)ephem_cap ? (uint32_t)req_max : ephem_cap;
      if (stash_n > c->kv_n) stash_n = c->kv_n;
      if (stash_n > 0u) {
        stash_buf = malloc((size_t)stash_n * b->kv_rowb);
        if (stash_buf) {
          const double ts = now_ms();
          b->kv_io(b, 1, 0, stash_n, stash_buf);
          stash_ms = now_ms() - ts;
          ephem = 1;
          st->common = L;
        }
      }
    }
    if (!ephem) {
      const uint32_t lim = L < n_prompt - 1 ? L : n_prompt - 1;
      for (uint32_t i = 0; i < c->n_ck; i++)
        if (c->ck[i].pos && c->ck[i].pos <= lim && (use < 0 || c->ck[i].pos > c->ck[use].pos)) use = (int)i;
      if (use >= 0) S = c->ck[use].pos, st->cache_hit = c->ck[use].kind, c->ck[use].used = ++c->tick;
      st->common = L;
      if (c->dc) { /* another conversation: write the one in memory before its checkpoints past L are dropped */
        dp_learn_load(c);
        if (!dp_continues(c, L)) dp_flush(c, "switch", L);
      }
    }
  }
  const int cache_run = cache && !ephem;
  uint64_t deh = 0;
  uint32_t P = 0, kind = 0, has_h = 0, learned = 0;
  const double t_sw = now_ms();
  const uint32_t learn_end = p->prefix_client ? p->sys_end : 0;
  const uint32_t dc_lim = p->n_imgs && p->imgs[0].pos < n_prompt - 1 ? p->imgs[0].pos : n_prompt - 1;
  const int dfound = cache_run && c->dc && dc_lim > S &&
                     dc_best(c->dc, prompt, dc_lim, S, mtp, &deh, &P, &kind, &has_h, learn_end, &learned);
  if (L > learned && L < learn_end && L < c->kv_n) learned = L;
  const double t_best = now_ms();
  if (dfound) {
    /* the disk cache holds more of this prompt: its rows replace those in the KV cache */
    uint32_t w = 0; /* the checkpoints past L are of other tokens */
    for (uint32_t r = 0; r < c->n_ck; r++)
      if (c->ck[r].pos <= L) c->ck[w++] = c->ck[r];
    c->n_ck = w;
    const int i = ck_slot(c, P), added = c->n_ck > w;
    const int r = dc_load(c, deh, P, prompt, c->ck[i].slot);
    if (!r) {
      c->ck[i].pos = P, c->ck[i].kind = (int)kind, c->ck[i].used = ++c->tick, c->ck[i].disk = 1, c->ck[i].h = (uint8_t)has_h;
      use = i, S = P, st->cache_hit = GEN_CK_DISK, st->disk_ms = now_ms() - t0;
      if (L < P) L = P;
      fprintf(stderr, "disk cache: continued from %u tokens (%.0f MB) after %.0f ms (the last conversation's checkpoints %.0f ms, "
              "finding this one %.0f ms, reading it %.0f ms)\n", P, (c->dc->snapb + (double)P * c->dc->rowb) / 1e6, st->disk_ms,
              t_sw - t0, t_best - t_sw, now_ms() - t_best);
    } else {
      if (added) c->n_ck--;
      if (r == -2) c->n_ck = 0, c->kv_n = 0, L = 0, S = 0, use = -1, st->cache_hit = -1; /* rows overwritten: start over */
      fprintf(stderr, "disk cache: a file of the checkpoint at %u is missing or damaged (the checkpoint is deleted)\n", P);
      dc_drop(c->dc, deh);
    }
  }
  st->cached = S;
  if (cache_run && L < c->kv_n && L < n_prompt && getenv("IE_SERVE_DEBUG")) { /* where the prompt leaves the cached tokens */
    fprintf(stderr, "cache: the prompt (%u tokens) differs from the cached tokens (%u) at %u:", n_prompt, c->kv_n, L);
    for (uint32_t k = L > 3 ? L - 3 : 0; k < L + 5 && k < c->kv_n; k++) fprintf(stderr, " %u", c->kv_toks[k]);
    fprintf(stderr, " | prompt:");
    for (uint32_t k = L > 3 ? L - 3 : 0; k < L + 5 && k < n_prompt; k++) fprintf(stderr, " %u", prompt[k]);
    fprintf(stderr, "\n");
  }
  uint32_t first, hrow = 0;
  if (use >= 0) b->snap(b, 0, c->ck[use].slot, hbuf, c->ck[use].pos);
  /* this request writes the positions from S on: the checkpoints past L are of other tokens */
  uint32_t mk[GEN_CK_MAX], nm = 0;
  int mkind[GEN_CK_MAX];
  if (cache_run) {
    uint32_t w = 0;
    for (uint32_t r = 0; r < c->n_ck; r++)
      if (c->ck[r].pos <= L) c->ck[w++] = c->ck[r];
    c->n_ck = w;
    kv_put(c, 0, prompt, n_prompt);
    free(c->kv_imgs);
    c->kv_imgs = NULL, c->kv_n_imgs = 0;
    if (p->n_imgs) {
      c->kv_imgs = malloc((size_t)p->n_imgs * sizeof *c->kv_imgs);
      if (!c->kv_imgs) ie_die("gen_run: out of memory");
      for (uint32_t k = 0; k < p->n_imgs; k++) {
        c->kv_imgs[k] = p->imgs[k];
        c->kv_imgs[k].embd = NULL;
      }
      c->kv_n_imgs = p->n_imgs;
    }
    const uint32_t A = L > S ? line_start(c, prompt, L) : L; /* the checkpoint for where the prompt changed */
    uint32_t xm[16];
    const uint32_t nxm = dp_places(c, p, prompt, n_prompt, xm);
    if (learned <= S || learned - S < ck_mark_gap()) learned = 0;
    const uint32_t prefix_min = c->dc ? ck_prefix_min(c->dc->min) : 128u;
    nm = ck_marks(c, S, A > S ? A : L, n_prompt - 1, xm, nxm, p, learned, prefix_min, mk, mkind);
    for (uint32_t k = 0; k < nm; k++) /* the system prompt differed from the last one here: learn the distance */
      if (mkind[k] == GEN_CK_SPLIT && p->sys_end && mk[k] < p->sys_end) dp_learn_add(c, p->sys_end - mk[k]);
  } else if (!ephem) c->n_ck = 0, c->kv_n = 0, free(c->kv_imgs), c->kv_imgs = NULL, c->kv_n_imgs = 0; /* a run without the cache changes all positions */
  defer_ck_ctx dck = {c, p->emit, p->ctx, 0u, hbuf, 0};
  if (b->run) { /* GPU: chunks */
    if (!ephem && b->kv_trim) b->kv_trim(b, n_prompt > S ? n_prompt : S);
    if (b->kv_io && n_prompt) b->kv_io(b, 0, n_prompt, 0, NULL); /* pre-map full prompt KV capacity once across all checkpoint segments */
    if (!S && mtp) b->copy_rows(b, g->mtp_h, 0, -1, 0, 1); /* h(-1) = 0 */
    if (!cache_run) first = prefill_range(b, g, prompt, 0, n_prompt, p->chunk, mtp, 1, &hrow, p->imgs, p->n_imgs, c->m->dim); /* as ie-run */
    else { /* up to each checkpoint, then all but the last token, its checkpoint, then the last token */
      uint32_t from = S;
      for (uint32_t i = 0; i < nm; i++) {
        prefill_range(b, g, prompt, from, mk[i], p->chunk, mtp, 0, &hrow, p->imgs, p->n_imgs, c->m->dim);
        ck_save(c, mk[i], mkind[i], hbuf);
        from = mk[i];
      }
      const int defer_prompt_ck = c->m->arch == ARCH_INKLING2 && !mtp && !p->n_imgs &&
                                  ie_defer_cksave_mode() >= 1 && n_prompt >= 2u &&
                                  from < n_prompt - 1u && ck_at(c, n_prompt - 1u) < 0;
      if (defer_prompt_ck) {
        /* R38 (H42): fold final token into prefill chunk (-8 ms T=1 pass) and defer GEN_CK_PROMPT post-token-1 */
        first = prefill_range(b, g, prompt, from, n_prompt, p->chunk, mtp, 2, &hrow, p->imgs, p->n_imgs, c->m->dim);
        dck.ck_pos = n_prompt - 1u;
        dck.ck_hbuf = hbuf;
        dck.pending = 1;
      } else {
        if (n_prompt - 1 > from) prefill_range(b, g, prompt, from, n_prompt - 1, p->chunk, mtp, 0, &hrow, p->imgs, p->n_imgs, c->m->dim);
        if (ck_at(c, n_prompt - 1) < 0) ck_save(c, n_prompt - 1, GEN_CK_PROMPT, hbuf);
        first = prefill_range(b, g, prompt, n_prompt - 1, n_prompt, p->chunk, mtp, 1, &hrow, p->imgs, p->n_imgs, c->m->dim);
      }
    }
    static int cache_dbg = -1;
    if (cache_dbg < 0) cache_dbg = getenv("IE_CACHE_DEBUG") != NULL;
    if (cache_dbg) { /* the first token's 5 largest logits (a cached and an uncached run compared) */
      b->read_rows(b, g->logits, hrow, 1, c->logits);
      uint32_t top[5] = {0, 0, 0, 0, 0};
      float tv[5] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
      for (uint32_t v = 0; v < V; v++) {
        const float x = c->logits[v];
        if (!(x > tv[4])) continue;
        uint32_t k = 4;
        for (; k > 0 && x > tv[k - 1]; k--) tv[k] = tv[k - 1], top[k] = top[k - 1];
        tv[k] = x, top[k] = v;
      }
      fprintf(stderr, "cache: first token logits (from %u):", S);
      for (uint32_t k = 0; k < 5u; k++) fprintf(stderr, " %u:%.4f", top[k], tv[k]);
      fprintf(stderr, "\n");
    }
    if (sp->temp > 0.0f) {
      if (gpu_cands(sp, b, g, hrow, 1, V, sp->cv, sp->ci)) first = sample_cands(sp, sp->cv, sp->ci, sample_k(sp, V));
      else {
        b->read_rows(b, g->logits, hrow, 1, c->logits);
        first = sample(sp, c->logits, V);
      }
    }
  } else { /* CPU: one token per step */
    double ms;
    uint32_t mi = 0;
    for (uint32_t pos = S; pos + 1 < n_prompt; pos++) {
      const float *er = p->n_imgs ? img_row_at(p->imgs, p->n_imgs, c->m->dim, pos) : NULL;
      b->ext_rows = er ? &er : NULL;
      b->step(b, prompt[pos], pos, NULL, &ms);
      b->ext_rows = NULL;
      if (mi < nm && pos + 1 == mk[mi]) {
        ck_save(c, mk[mi], mkind[mi], -1);
        mi++;
      }
    }
    if (cache && ck_at(c, n_prompt - 1) < 0) ck_save(c, n_prompt - 1, GEN_CK_PROMPT, -1);
    const float *er_last = p->n_imgs ? img_row_at(p->imgs, p->n_imgs, c->m->dim, n_prompt - 1) : NULL;
    b->ext_rows = er_last ? &er_last : NULL;
    first = b->step(b, prompt[n_prompt - 1], n_prompt - 1, sp->temp > 0.0f ? c->logits : NULL, &ms);
    b->ext_rows = NULL;
    if (sp->temp > 0.0f) first = sample(sp, c->logits, V);
  }
  st->prefill_ms = now_ms() - t0;

  if (p->draft) { /* MTP: the same loop as ie-run --draft */
    const uint32_t total = n_prompt + p->max_new;
    uint32_t *out = malloc((size_t)total * 4);
    if (!out) ie_die("gen_run: out of memory");
    memcpy(out, prompt, (size_t)n_prompt * 4);
    uint32_t done = 0, hr = 0;
    const uint32_t n = spec_generate(b, c->m, g, out, n_prompt, total, p->draft, -1, 0, first, hrow, sp, V, &st->gen_ms,
                                     &st->steps, &st->acc, p->pmin, &st->drafted,
                                     dck.pending ? defer_ck_emit : p->emit, dck.pending ? (void *)&dck : p->ctx,
                                     p->stops, p->n_stops, &done, &hr);
    if (dck.pending) {
      dck.pending = 0;
      ck_save(c, dck.ck_pos, GEN_CK_PROMPT, dck.ck_hbuf);
    }
    st->n_out = n - n_prompt;
    st->dry_hits = sp->dry_hits;
    st->reasoning_limited = sp->reasoning_limited;
    if (ephem) {
      uint32_t max_w = (done > n_prompt ? done : n_prompt) + spec_room(p->draft) + 2u;
      if (max_w <= stash_n) {
        const double tr = now_ms();
        b->kv_io(b, 0, 0, stash_n, stash_buf);
        fprintf(stderr, "cache: ephemeral side request (%u+%u tok) preserved %u-token prefix (%u ckpts, %u rows in %.1f ms)\n",
                n_prompt, st->n_out, c->kv_n, c->n_ck, stash_n, stash_ms + (now_ms() - tr));
      } else {
        c->n_ck = 0, c->kv_n = 0;
        if (done > n_prompt) {
          if (mtp) b->copy_rows(b, g->mtp_h, 0, g->h_out, hr, 1);
          kv_put(c, 0, out, done);
          ck_save(c, done, GEN_CK_ANSWER, hbuf);
        }
        if (b->kv_trim) b->kv_trim(b, c->kv_n);
      }
      free(stash_buf);
      st->n_ck = c->n_ck;
      free(out);
      return;
    }
    if (cache_run && done > n_prompt) { /* a checkpoint after the prompt and the answer tokens that were run */
      if (mtp) b->copy_rows(b, g->mtp_h, 0, g->h_out, hr, 1); /* the MTP h input of the next position */
      kv_put(c, n_prompt, out + n_prompt, done - n_prompt);
      ck_save(c, done, GEN_CK_ANSWER, hbuf);
    }
    dp_save_prefixes(c, p->prefix_client, mk, mkind, nm, learned);
    st->n_ck = c->n_ck;
    free(out);
    return;
  }

  /* one token per step, as the ie-run loop */
  const double t1 = now_ms();
  uint32_t tok = first, pos = n_prompt, *ov = cache_run ? malloc((size_t)p->max_new * 4) : NULL;
  if (ov) ov[0] = first;
  st->n_out = 1;
  commit_output_token(sp, first);
  const int stop0 = p->emit ? p->emit(p->ctx, &tok, 1) : 0;
  if (dck.pending) {
    dck.pending = 0;
    ck_save(c, dck.ck_pos, GEN_CK_PROMPT, dck.ck_hbuf);
  }
  if (!stop0) {
    static int host_topk_step = -1;
    if (host_topk_step < 0) host_topk_step = getenv("IE_HOST_TOPK") != NULL;
    for (; st->n_out < p->max_new; st->n_out++, pos++) {
      const int gk = sp->temp > 0.0f && b->topk && sample_k(sp, V) <= 64u && !host_topk_step;
      uint32_t next = b->step(b, tok, pos, (sp->temp > 0.0f && !gk) ? c->logits : NULL, NULL);
      if (reasoning_exhausted(sp)) next = sp->reasoning_end_plus_one - 1u;
      else if (sp->temp > 0.0f) {
        if (gk && gpu_cands(sp, b, g, 0, 1, V, sp->cv, sp->ci)) next = sample_cands(sp, sp->cv, sp->ci, sample_k(sp, V));
        else {
          if (gk) b->read_rows(b, g->logits, 0, 1, c->logits);
          next = sample(sp, c->logits, V);
        }
      }
      tok = next;
      commit_output_token(sp, tok);
      if (ov) ov[st->n_out] = tok;
      if (p->emit(p->ctx, &tok, 1)) { st->n_out++; break; }
    }
  }
  st->gen_ms = now_ms() - t1;
  st->dry_hits = sp->dry_hits;
  st->reasoning_limited = sp->reasoning_limited;
  if (ephem) {
    uint32_t max_w = n_prompt + st->n_out + 2u;
    if (max_w <= stash_n) {
      const double tr = now_ms();
      b->kv_io(b, 0, 0, stash_n, stash_buf);
      fprintf(stderr, "cache: ephemeral side request (%u+%u tok) preserved %u-token prefix (%u ckpts, %u rows in %.1f ms)\n",
              n_prompt, st->n_out, c->kv_n, c->n_ck, stash_n, stash_ms + (now_ms() - tr));
    } else {
      c->n_ck = 0, c->kv_n = 0;
    }
    free(stash_buf);
    st->n_ck = c->n_ck;
    free(ov);
    return;
  }
  /* a checkpoint after the answer: the last token given out was not run; all before it were */
  if (ov && st->n_out > 1) {
    kv_put(c, n_prompt, ov, st->n_out - 1);
    ck_save(c, n_prompt + st->n_out - 1, GEN_CK_ANSWER, hbuf);
  }
  dp_save_prefixes(c, p->prefix_client, mk, mkind, nm, learned);
  st->n_ck = c->n_ck;
  free(ov);
}
