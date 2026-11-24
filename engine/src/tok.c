/* tok.c -- text to token ids (see tok.h).
 *
 * Steps for one text:
 *   1. Split on special tokens (longest match first).
 *   2. Split each other part with the pre-tokenizer regex (hand-written
 *      matcher, below), on code points.
 *   3. Map each piece's bytes to the GPT2 byte-level characters and apply
 *      the BPE merges (lowest rank first, leftmost first on equal rank).
 *   4. Look up each final symbol; a symbol not in the vocabulary becomes one
 *      token per byte (as llama.cpp).
 * Checked against llama.cpp's test vocabularies (models/ggml-vocab-qwen2.gguf
 * and ggml-vocab-qwen35.gguf with their .inp/.out files): tools/tok_check.c. */
#include <string.h>

#include "common.h"
#ifdef _WIN32
#include <process.h>
#else
#include <pthread.h>
#endif
#include "tok.h"
#include "tok_unicode.h"

/* ---- hash map: byte string -> u32 ---- */
typedef struct {
  const char *s; /* NULL: empty slot */
  uint32_t n, v;
} hent;
typedef struct {
  hent *t;
  uint32_t cap; /* power of 2 */
} hmap;

static uint64_t fnv1a(const char *s, uint32_t n) {
  uint64_t h = 1469598103934665603ull;
  for (uint32_t i = 0; i < n; i++) h = (h ^ (unsigned char)s[i]) * 1099511628211ull;
  return h;
}
static void hmap_init(hmap *h, uint32_t n) {
  h->cap = 16;
  while (h->cap < 2 * n + 16) h->cap *= 2;
  h->t = calloc(h->cap, sizeof(hent));
  if (!h->t) ie_die("tok: out of memory");
}
/* s must live as long as the map. The first value for a key is kept. */
static void hmap_put(hmap *h, const char *s, uint32_t n, uint32_t v) {
  for (uint64_t i = fnv1a(s, n) & (h->cap - 1);; i = (i + 1) & (h->cap - 1)) {
    hent *e = &h->t[i];
    if (!e->s) { e->s = s, e->n = n, e->v = v; return; }
    if (e->n == n && !memcmp(e->s, s, n)) return;
  }
}
static int64_t hmap_get(const hmap *h, const char *s, uint32_t n) {
  for (uint64_t i = fnv1a(s, n) & (h->cap - 1);; i = (i + 1) & (h->cap - 1)) {
    const hent *e = &h->t[i];
    if (!e->s) return -1;
    if (e->n == n && !memcmp(e->s, s, n)) return e->v;
  }
}

/* ---- the tokenizer ---- */
enum { PRE_QWEN2 = 0, PRE_QWEN35 = 1, PRE_FLINT = 2, PRE_DBRX = 3, PRE_INKLING2 = 4, PRE_GPT4O = 5 };
typedef struct {
  const char *s;
  uint32_t n, id;
  int control; /* 1: type 3 (control), 0: type 4 (user-defined) */
} special;

struct tokenizer {
  int pre;
  int ignore_merges;
  char *pre_name;
  hmap vocab;  /* token string -> id (strings: the model's) */
  hmap merges; /* "left right" -> rank */
  char *mbuf;  /* the merge strings */
  special *sp; /* longest first */
  uint32_t n_sp;
  uint8_t *ctrl;         /* 1: token_type == 3 (control token, never matched by ignore_merges) */
  uint8_t sp_first[256]; /* 1: some special token starts with this byte */
  uint32_t byte_id[256]; /* the token of each single byte (UINT32_MAX: none) */
  char bchar[256][3];    /* GPT2 byte -> character (UTF-8, 1 or 2 bytes) */
  uint8_t blen[256];
};

void tok_ids_push(tok_ids *a, uint32_t id) {
  if (a->n == a->cap) {
    a->cap = a->cap ? 2 * a->cap : 64;
    a->v = realloc(a->v, (size_t)a->cap * 4);
    if (!a->v) ie_die("tok: out of memory");
  }
  a->v[a->n++] = id;
}
void tok_ids_free(tok_ids *a) {
  free(a->v);
  memset(a, 0, sizeof *a);
}

/* GPT2 bytes_to_unicode(): printable bytes map to themselves, the other 68 to 256, 257, ... */
static void byte_chars(tokenizer *t) {
  uint32_t n = 0;
  for (uint32_t b = 0; b < 256; b++) {
    const int keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    const uint32_t cp = keep ? b : 256 + n++;
    if (cp < 0x80) t->bchar[b][0] = (char)cp, t->blen[b] = 1;
    else t->bchar[b][0] = (char)(0xC0 | (cp >> 6)), t->bchar[b][1] = (char)(0x80 | (cp & 0x3F)), t->blen[b] = 2;
  }
}

static int cmp_special(const void *a, const void *b) {
  const special *x = a, *y = b;
  return x->n != y->n ? (x->n > y->n ? -1 : 1) : (x->id < y->id ? -1 : x->id > y->id);
}

tokenizer *tok_open(const model *m, const gguf_file *g, char *err, size_t errn) {
  if (!m->tok_gpt2 || !m->n_tokens) { snprintf(err, errn, "the vocabulary is not GPT2 byte-level BPE"); return NULL; }
  char *pre = gguf_get_str(g, "tokenizer.ggml.pre");
  int pt;
  if (pre && !strcmp(pre, "qwen35")) pt = PRE_QWEN35;
  else if (pre && (!strcmp(pre, "qwen2") || !strcmp(pre, "deepseek-r1-qwen"))) pt = PRE_QWEN2;
  else if (pre && !strcmp(pre, "flint")) pt = PRE_FLINT;
  else if (pre && !strcmp(pre, "dbrx")) pt = PRE_DBRX; /* cl100k / OLMo-2 (Ripple-65B): Qwen2's split, numbers in runs of 1-3 */
  else if (pre && !strcmp(pre, "inkling2")) pt = PRE_INKLING2;
  else if (pre && !strcmp(pre, "gpt-4o")) pt = PRE_GPT4O;
  else {
    snprintf(err, errn, "pre-tokenizer \"%s\" is not supported (qwen2, qwen35, flint, dbrx, inkling2)", pre ? pre : "(none)");
    free(pre);
    return NULL;
  }
  const gguf_kv *mk = gguf_find(g, "tokenizer.ggml.merges");
  if (!mk || mk->type != GGUF_ARR || mk->arr_type != GGUF_STR) {
    snprintf(err, errn, "no tokenizer.ggml.merges");
    free(pre);
    return NULL;
  }
  tokenizer *t = calloc(1, sizeof *t);
  if (!t) ie_die("tok: out of memory");
  t->pre = pt, t->pre_name = pre;
  t->ignore_merges = (pt == PRE_INKLING2);
  {
    int64_t ig = 0;
    if (gguf_get_int(g, "tokenizer.ggml.ignore_merges", &ig)) t->ignore_merges = (ig != 0);
  }
  byte_chars(t);

  hmap_init(&t->vocab, m->n_tokens);
  for (uint32_t i = 0; i < m->n_tokens; i++) hmap_put(&t->vocab, m->tokens[i], (uint32_t)strlen(m->tokens[i]), i);

  /* merges: one buffer for all strings */
  size_t tot = 0;
  for (uint64_t i = 0; i < mk->arr_n; i++) tot += mk->arr_str[i].n;
  t->mbuf = malloc(tot ? tot : 1);
  if (!t->mbuf) ie_die("tok: out of memory");
  hmap_init(&t->merges, (uint32_t)mk->arr_n);
  size_t o = 0;
  for (uint64_t i = 0; i < mk->arr_n; i++) {
    memcpy(t->mbuf + o, mk->arr_str[i].p, mk->arr_str[i].n);
    hmap_put(&t->merges, t->mbuf + o, (uint32_t)mk->arr_str[i].n, (uint32_t)i);
    o += mk->arr_str[i].n;
  }

  /* special tokens: type 3 (control) and 4 (user-defined) */
  const gguf_kv *tt = gguf_find(g, "tokenizer.ggml.token_type");
  if (tt && tt->type == GGUF_ARR && tt->arr_type == GGUF_I32) {
    t->sp = calloc(tt->arr_n + 1, sizeof(special));
    t->ctrl = calloc(m->n_tokens ? m->n_tokens : 1, 1);
    if (!t->sp || !t->ctrl) ie_die("tok: out of memory");
    for (uint64_t i = 0; i < tt->arr_n && i < m->n_tokens; i++) {
      int32_t ty;
      memcpy(&ty, tt->arr_data + i * 4, 4); /* the file data may be unaligned */
      if (ty == 3) t->ctrl[i] = 1;
      const uint32_t n = (uint32_t)strlen(m->tokens[i]);
      if ((ty == 3 || ty == 4) && n) {
        t->sp[t->n_sp++] = (special){m->tokens[i], n, (uint32_t)i, ty == 3};
        t->sp_first[(unsigned char)m->tokens[i][0]] = 1;
      }
    }
    qsort(t->sp, t->n_sp, sizeof(special), cmp_special);
  }

  for (uint32_t b = 0; b < 256; b++) {
    const int64_t id = hmap_get(&t->vocab, t->bchar[b], t->blen[b]);
    t->byte_id[b] = id < 0 ? UINT32_MAX : (uint32_t)id;
  }
  return t;
}

void tok_close(tokenizer *t) {
  if (!t) return;
  free(t->vocab.t), free(t->merges.t), free(t->mbuf), free(t->sp), free(t->ctrl), free(t->pre_name);
  free(t);
}

int64_t tok_find(const tokenizer *t, const char *s) { return hmap_get(&t->vocab, s, (uint32_t)strlen(s)); }
const char *tok_pre(const tokenizer *t) { return t->pre_name; }

/* ---- code point classes ---- */
static int uc_flags(uint32_t cp) {
  uint32_t lo = 0, hi = UC_NRANGES;
  while (lo < hi) {
    const uint32_t mid = (lo + hi) / 2;
    if (cp < uc_ranges[mid][0]) hi = mid;
    else if (cp > uc_ranges[mid][1]) lo = mid + 1;
    else return (int)uc_ranges[mid][2];
  }
  return 0;
}
/* \s: the Unicode White_Space property */
static int uc_space(uint32_t cp) {
  return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
         cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

/* A part of the text as code points: cp[i] starts at byte off[i]. An
 * invalid byte becomes one code point of class "other" (cp 0xFFFFFFFF). */
typedef struct {
  uint32_t *cp, *off;
  uint16_t *cl; /* bits: UC_L, UC_M, UC_N, 8 = space, 16 = CR or LF; UC_UP .. UC_S (tok_unicode.h) */
  uint32_t n;
} cps;

static void decode(const char *s, size_t len, cps *c) {
  c->cp = malloc((len + 1) * 4), c->off = malloc((len + 1) * 4), c->cl = malloc((len + 1) * 2);
  if (!c->cp || !c->off || !c->cl) ie_die("tok: out of memory");
  uint32_t n = 0;
  const unsigned char *p = (const unsigned char *)s;
  size_t i = 0;
  while (i < len) {
    const uint32_t b0 = p[i];
    uint32_t cp, need, k = 1;
    if (b0 < 0x80) cp = b0, need = 0;
    else if ((b0 & 0xE0) == 0xC0) cp = b0 & 31, need = 1;
    else if ((b0 & 0xF0) == 0xE0) cp = b0 & 15, need = 2;
    else if ((b0 & 0xF8) == 0xF0) cp = b0 & 7, need = 3;
    else cp = 0xFFFFFFFFu, need = 0; /* a continuation byte alone, or 0xF8 .. 0xFF */
    if (need && i + need >= len) cp = 0xFFFFFFFFu, need = 0; /* cut off at the end */
    for (uint32_t j = 1; j <= need; j++) {
      if ((p[i + j] & 0xC0) != 0x80) { cp = 0xFFFFFFFFu; break; }
      cp = (cp << 6) | (p[i + j] & 0x3F), k++;
    }
    if (cp == 0xFFFFFFFFu) k = 1;
    c->cp[n] = cp, c->off[n] = (uint32_t)i;
    uint16_t cl = cp == 0xFFFFFFFFu ? 0 : (uint16_t)uc_flags(cp);
    if (cp != 0xFFFFFFFFu && uc_space(cp)) cl |= 8;
    if (cp == '\r' || cp == '\n') cl |= 16;
    c->cl[n++] = cl;
    i += k;
  }
  c->off[n] = (uint32_t)len;
  c->n = n;
}

static uint32_t lower_ascii(uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }

/* Match \s*[\r\n]+|\s+(?!\S)|\s+ starting at i (cl[i] & 8). */
static uint32_t match_ws(const cps *c, uint32_t i, uint32_t n) {
  const uint16_t *cl = c->cl;
  uint32_t j = i;
  while (j < n && (cl[j] & 8)) j++;
  /* \s*[\r\n]+ ends after the last CR or LF of the space run */
  for (uint32_t k = j; k > i; k--)
    if (cl[k - 1] & 16) return k;
  /* \s+(?!\S) */
  if (j == n) return j;
  if (j - 1 > i) return j - 1;
  /* \s+ */
  return j;
}

/* The end of the regex match at i (> i). Qwen2:
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 * Qwen3.5: \p{L} -> [\p{L}\p{M}] in the second alternative, and \p{M} also
 * excluded in the fourth. */
static uint32_t match(const cps *c, uint32_t i, int pre) {
  const uint32_t n = c->n;
  const uint16_t *cl = c->cl;
  const uint16_t LET = pre == PRE_QWEN35 ? (UC_L | UC_M) : UC_L, OTH = LET | UC_N | 8;
  /* 1: contractions */
  if (c->cp[i] == '\'' && i + 1 < n) {
    const uint32_t a = lower_ascii(c->cp[i + 1]);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
    if (i + 2 < n) {
      const uint32_t b = lower_ascii(c->cp[i + 2]);
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
    }
  }
  /* 2: one optional char that is not CR, LF, L or N; then letters */
  {
    uint32_t j = i;
    if (!(cl[j] & (LET | 16 | UC_L | UC_N)) && j + 1 < n && (cl[j + 1] & LET)) j++;
    if (cl[j] & LET) {
      while (j < n && (cl[j] & LET)) j++;
      return j;
    }
  }
  /* 3: one number (dbrx, cl100k: \p{N}{1,3}, up to three) */
  if (cl[i] & UC_N) {
    uint32_t j = i + 1;
    if (pre == PRE_DBRX)
      while (j < n && j < i + 3 && (cl[j] & UC_N)) j++;
    return j;
  }
  /* 4: optional space, chars that are not space, L or N, then CR and LF */
  {
    uint32_t j = i + (c->cp[i] == ' ' ? 1 : 0), k = j;
    while (k < n && !(cl[k] & OTH)) k++;
    if (k > j) {
      while (k < n && (cl[k] & 16)) k++;
      return k;
    }
  }
  if (cl[i] & 8) return match_ws(c, i, n);
  return i + 1;
}

/* A contraction (?i:'s|'t|'re|'ve|'m|'ll|'d) at j (before n), or nothing: the end. */
static uint32_t contraction(const cps *c, uint32_t j, uint32_t n) {
  if (j + 1 < n && c->cp[j] == '\'') {
    const uint32_t a = lower_ascii(c->cp[j + 1]);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return j + 2;
    if (j + 2 < n) {
      const uint32_t b = lower_ascii(c->cp[j + 2]);
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return j + 3;
    }
  }
  return j;
}

/* Case-split word + optional contraction (shared by flint and gpt4o/inkling2):
 *   [^\r\n\p{L}\p{N}]?[U]*[W]+(?i:'s|..)?
 *   [^\r\n\p{L}\p{N}]?[U]+[W]*(?i:'s|..)?
 * Returns end index > i, or i if neither matches. */
static uint32_t match_case_word(const cps *c, uint32_t i, uint32_t n) {
  const uint16_t *cl = c->cl;
#define FU(k) (cl[k] & (UC_UP | UC_LMO | UC_M))
#define FW(k) (cl[k] & (UC_LO | UC_LMO | UC_M))
#define FPRE(k) (!(cl[k] & (16 | UC_L | UC_N)))
  for (int pf = 1; pf >= 0; pf--) { /* 1: U* W+ */
    if (pf && !FPRE(i)) continue;
    const uint32_t j = i + (uint32_t)pf;
    uint32_t ue = j;
    while (ue < n && FU(ue)) ue++;
    for (uint32_t k = ue;; k--) {
      uint32_t we = k;
      while (we < n && FW(we)) we++;
      if (we > k) return contraction(c, we, n);
      if (k == j) break;
    }
  }
  for (int pf = 1; pf >= 0; pf--) { /* 2: U+ W* */
    if (pf && !FPRE(i)) continue;
    const uint32_t j = i + (uint32_t)pf;
    uint32_t ue = j;
    while (ue < n && FU(ue)) ue++;
    if (ue > j) {
      while (ue < n && FW(ue)) ue++;
      return contraction(c, ue, n);
    }
  }
#undef FU
#undef FW
#undef FPRE
  return i;
}

/* flint: the end of the match of the third split's regex at i in [i, n), or
 * i (no match: i is in a gap):
 *   [^\r\n\p{L}\p{N}]?[U]*[W]+(?i:'s|..)?   U = [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
 *   [^\r\n\p{L}\p{N}]?[U]+[W]*(?i:'s|..)?   W = [\p{Ll}\p{Lm}\p{Lo}\p{M}]
 *    ?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
 * tried in order, each with its backtracking (the optional first char taken
 * first; U* longest first). */
static uint32_t match_flint(const cps *c, uint32_t i, uint32_t n) {
  const uint16_t *cl = c->cl;
  const uint32_t w = match_case_word(c, i, n);
  if (w > i) return w;
  { /* 3: optional space, punctuation and symbols, then CR and LF */
    const uint32_t j = i + (c->cp[i] == ' ' ? 1 : 0);
    uint32_t k = j;
    while (k < n && (cl[k] & (UC_P | UC_S))) k++;
    if (k > j) {
      while (k < n && (cl[k] & 16)) k++;
      return k;
    }
  }
  if (cl[i] & 8) return match_ws(c, i, n);
  return i;
}

/* inkling2 / GPT-4o pre-tokenizer regex:
 *   [^\r\n\p{L}\p{N}]?[U]*[W]+(?i:'s|..)?
 * | [^\r\n\p{L}\p{N}]?[U]+[W]*(?i:'s|..)?
 * | \p{N}{1,3}
 * |  ?[^\s\p{L}\p{N}]+[\r\n/]*
 * | \s*[\r\n]+ | \s+(?!\S) | \s+ */
static uint32_t match_gpt4o(const cps *c, uint32_t i) {
  const uint32_t n = c->n;
  const uint16_t *cl = c->cl;
  const uint32_t w = match_case_word(c, i, n);
  if (w > i) return w;
  if (cl[i] & UC_N) { /* 3: \p{N}{1,3} */
    uint32_t j = i + 1;
    while (j < n && j < i + 3 && (cl[j] & UC_N)) j++;
    return j;
  }
  { /* 4: optional space, non-space/letter/number run, then CR, LF or '/' */
    const uint32_t j = i + (c->cp[i] == ' ' ? 1 : 0);
    uint32_t k = j;
    while (k < n && !(cl[k] & (8 | UC_L | UC_N))) k++;
    if (k > j) {
      while (k < n && ((cl[k] & 16) || c->cp[k] == '/')) k++;
      return k;
    }
  }
  if (cl[i] & 8) return match_ws(c, i, n);
  return i + 1;
}

/* flint: [\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff] (CJK ideographs, hiragana, katakana) */
static int cjk_kana(uint32_t cp) { return (cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x30FF); }

static void bpe(const tokenizer *t, const char *s, uint32_t len, tok_ids *out, int fast_bpe);
/* flint: the pieces of code points [a, b) of the third split (regex
 * matches; the unmatched characters between them form one piece each run) */
static void flint_split3(const tokenizer *t, const char *s, const cps *c, uint32_t a, uint32_t b, tok_ids *out, int fast_bpe) {
  uint32_t gap = a;
  for (uint32_t i = a; i < b;) {
    const uint32_t j = match_flint(c, i, b);
    if (j == i) { i++; continue; }
    if (gap < i) bpe(t, s + c->off[gap], c->off[i] - c->off[gap], out, fast_bpe);
    bpe(t, s + c->off[i], c->off[j] - c->off[i], out, fast_bpe);
    i = gap = j;
  }
  if (gap < b) bpe(t, s + c->off[gap], c->off[b] - c->off[gap], out, fast_bpe);
}
/* flint: the second split (CJK and kana runs apart), then the third */
static void flint_split2(const tokenizer *t, const char *s, const cps *c, uint32_t a, uint32_t b, tok_ids *out, int fast_bpe) {
  for (uint32_t i = a; i < b;) {
    uint32_t j = i;
    const int k = cjk_kana(c->cp[i]);
    while (j < b && cjk_kana(c->cp[j]) == k) j++;
    if (k) bpe(t, s + c->off[i], c->off[j] - c->off[i], out, fast_bpe);
    else flint_split3(t, s, c, i, j, out, fast_bpe);
    i = j;
  }
}

static void tok_ids_append(tok_ids *a, const uint32_t *src, uint32_t n) {
  if (!n) return;
  if (a->n + n > a->cap) {
    uint32_t cap = a->cap ? a->cap : 64u;
    while (cap < a->n + n) {
      if (cap > UINT32_MAX / 2u) { cap = a->n + n; break; }
      cap *= 2u;
    }
    a->v = realloc(a->v, (size_t)cap * 4u);
    if (!a->v) ie_die("tok: out of memory");
    a->cap = cap;
  }
  memcpy(a->v + a->n, src, (size_t)n * 4u);
  a->n += n;
}

static int tok_fast_bpe_env(void) {
  const char *e = getenv("IE_TOK_FAST_BPE");
  return !(e && e[0] == '0'); /* R82 promoted default ON (=0 opts out) */
}

static int tok_threads_env(void) {
  const char *e = getenv("IE_TOK_THREADS");
  if (!e || !e[0]) return 4; /* R82 promoted default 4T (=1 opts out) */
  int n = atoi(e);
  if (n < 1) n = 1;
  if (n > 8) n = 8;
  return n;
}

/* BPE on one piece (bytes s[0 .. len)).
 * R82 (H47b, IE_TOK_FAST_BPE=1 opt-in): use stack arrays for len <= 256 to eliminate per-piece heap malloc/free. */
static void bpe(const tokenizer *t, const char *s, uint32_t len, tok_ids *out, int fast_bpe) {
  /* the GPT2 characters of the bytes */
  char s_buf[513], s_key[516];
  uint32_t s_off[257], s_ln[257], s_rk[257];
  const int use_stack = fast_bpe && len <= 256u;
  char *buf = use_stack ? s_buf : malloc((size_t)len * 2u + 1u);
  if (!buf) ie_die("tok: out of memory");
  uint32_t bn = 0;
  for (uint32_t i = 0; i < len; i++) {
    const unsigned char b = (unsigned char)s[i];
    memcpy(buf + bn, t->bchar[b], t->blen[b]);
    bn += t->blen[b];
  }
  if (t->ignore_merges) {
    const int64_t id = hmap_get(&t->vocab, buf, bn);
    if (id >= 0 && (!t->ctrl || !t->ctrl[id])) {
      tok_ids_push(out, (uint32_t)id);
      if (!use_stack) free(buf);
      return;
    }
  }
  uint32_t *off = use_stack ? s_off : malloc((size_t)(len + 1u) * 4u);
  uint32_t *ln = use_stack ? s_ln : malloc((size_t)(len + 1u) * 4u);
  uint32_t *rk = use_stack ? s_rk : malloc((size_t)(len + 1u) * 4u);
  if (!off || !ln || !rk) ie_die("tok: out of memory");
  uint32_t ns = 0;
  for (uint32_t i = 0, bo = 0; i < len; i++) {
    const unsigned char b = (unsigned char)s[i];
    off[ns] = bo, ln[ns] = t->blen[b], ns++;
    bo += t->blen[b];
  }
  char *key = use_stack ? s_key : malloc((size_t)bn + 2u);
  if (!key) ie_die("tok: out of memory");
#define RANK(k)                                                                       \
  ((k) + 1 < ns ? (memcpy(key, buf + off[k], ln[k]), key[ln[k]] = ' ',               \
                   memcpy(key + ln[k] + 1, buf + off[(k) + 1], ln[(k) + 1]),          \
                   (uint32_t)hmap_get(&t->merges, key, ln[k] + 1 + ln[(k) + 1]))      \
                : UINT32_MAX)
  for (uint32_t k = 0; k < ns; k++) rk[k] = RANK(k); /* -1 (no merge) is UINT32_MAX */
  for (;;) {
    uint32_t best = UINT32_MAX, bk = 0;
    for (uint32_t k = 0; k + 1 < ns; k++)
      if (rk[k] < best) best = rk[k], bk = k;
    if (best == UINT32_MAX) break;
    ln[bk] += ln[bk + 1]; /* the symbols are adjacent in buf */
    memmove(off + bk + 1, off + bk + 2, (ns - bk - 2) * 4);
    memmove(ln + bk + 1, ln + bk + 2, (ns - bk - 2) * 4);
    memmove(rk + bk + 1, rk + bk + 2, (ns - bk - 2) * 4);
    ns--;
    rk[bk] = RANK(bk);
    if (bk) rk[bk - 1] = RANK(bk - 1);
  }
#undef RANK
  for (uint32_t k = 0; k < ns; k++) {
    const int64_t id = hmap_get(&t->vocab, buf + off[k], ln[k]);
    if (id >= 0) { tok_ids_push(out, (uint32_t)id); continue; }
    /* not in the vocabulary: one token per byte (find the bytes of this symbol) */
    for (uint32_t p = off[k]; p < off[k] + ln[k];) {
      const unsigned char c0 = (unsigned char)buf[p];
      const uint32_t cl = c0 < 0x80 ? 1 : 2;
      for (uint32_t b = 0; b < 256; b++)
        if (t->blen[b] == cl && !memcmp(t->bchar[b], buf + p, cl)) {
          if (t->byte_id[b] != UINT32_MAX) tok_ids_push(out, t->byte_id[b]);
          break;
        }
      p += cl;
    }
  }
  if (!use_stack) free(buf), free(off), free(ln), free(rk), free(key);
}

/* Single-threaded encode of a part without special tokens. */
static void encode_plain_st(const tokenizer *t, const char *s, size_t len, tok_ids *out, int fast_bpe) {
  if (!len) return;
  cps c;
  decode(s, len, &c);
  if (t->pre == PRE_FLINT) { /* \p{N}{1,3} apart, then the second and third splits */
    for (uint32_t i = 0; i < c.n;) {
      uint32_t j = i;
      if (c.cl[i] & UC_N) {
        while (j < c.n && j < i + 3 && (c.cl[j] & UC_N)) j++;
        bpe(t, s + c.off[i], c.off[j] - c.off[i], out, fast_bpe);
      } else {
        while (j < c.n && !(c.cl[j] & UC_N)) j++;
        flint_split2(t, s, &c, i, j, out, fast_bpe);
      }
      i = j;
    }
    free(c.cp), free(c.off), free(c.cl);
    return;
  }
  for (uint32_t i = 0; i < c.n;) {
    const uint32_t j = (t->pre == PRE_INKLING2 || t->pre == PRE_GPT4O) ? match_gpt4o(&c, i) : match(&c, i, t->pre);
    bpe(t, s + c.off[i], c.off[j] - c.off[i], out, fast_bpe);
    i = j;
  }
  free(c.cp), free(c.off), free(c.cl);
}

/* R82 (H47b, IE_TOK_THREADS=N opt-in): lossless multi-threaded chunked BPE.
 * Safe split invariant: a byte offset p with s[p - 1] == '\n' and s[p] in [A-Za-z0-9]
 * is an exact UTF-8 and pre-tokenizer regex boundary across PRE_INKLING2, PRE_GPT4O,
 * PRE_QWEN2, PRE_QWEN35, PRE_DBRX, and PRE_FLINT:
 *   - match_case_word excludes CR/LF from FPRE, FU, FW;
 *   - punctuation/symbol + [\r\n/]* stops before [A-Za-z0-9];
 *   - match_ws (\s*[\r\n]+) terminates right after '\n' at p in both full text and chunk;
 *   - BPE merges never cross regex piece boundaries.
 * If fewer than 2 safe newline cuts exist (e.g. single-line minified JSON), fall back
 * to exact piece-boundary fan-out over pre-tokenized pieces (non-PRE_FLINT). */
static inline int tok_is_ascii_alnum(unsigned char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
}

static size_t find_safe_split(const char *s, size_t lo, size_t hi, size_t target) {
  if (lo < 1u) lo = 1u;
  if (lo >= hi) return 0u;
  if (target < lo) target = lo;
  if (target >= hi) target = hi - 1u;
  const size_t max_d = (target - lo > hi - 1u - target) ? (target - lo) : (hi - 1u - target);
  for (size_t d = 0u; d <= max_d; d++) {
    if (target + d < hi) {
      const size_t p = target + d;
      if (s[p - 1u] == '\n' && tok_is_ascii_alnum((unsigned char)s[p])) return p;
    }
    if (d > 0u && target >= lo + d) {
      const size_t p = target - d;
      if (s[p - 1u] == '\n' && tok_is_ascii_alnum((unsigned char)s[p])) return p;
    }
  }
  return 0u;
}

typedef struct {
  const tokenizer *t;
  const char *s;
  size_t len;
  const uint32_t *poff; /* non-NULL in piece-range mode */
  uint32_t p0, p1;
  int fast_bpe;
  tok_ids out;
} tok_job;

static void tok_job_run(tok_job *j) {
  if (j->poff) {
    for (uint32_t k = j->p0; k < j->p1; k++)
      bpe(j->t, j->s + j->poff[k], j->poff[k + 1u] - j->poff[k], &j->out, j->fast_bpe);
  } else {
    encode_plain_st(j->t, j->s, j->len, &j->out, j->fast_bpe);
  }
}

#ifdef _WIN32
static unsigned __stdcall tok_worker_entry(void *arg) {
  tok_job_run((tok_job *)arg);
  return 0u;
}
#else
static void *tok_worker_entry(void *arg) {
  tok_job_run((tok_job *)arg);
  return NULL;
}
#endif

static void tok_run_jobs(tok_job *jobs, uint32_t nc, tok_ids *out) {
#ifdef _WIN32
  HANDLE hs[8];
  DWORD nh = 0u;
  for (uint32_t k = 1u; k < nc; k++) {
    uintptr_t h = _beginthreadex(NULL, 0u, tok_worker_entry, &jobs[k], 0u, NULL);
    if (h) hs[nh++] = (HANDLE)h;
    else tok_job_run(&jobs[k]); /* thread spawn failure: execute inline safely */
  }
  tok_job_run(&jobs[0]);
  if (nh > 0u) {
    WaitForMultipleObjects(nh, hs, TRUE, INFINITE);
    for (DWORD i = 0u; i < nh; i++) CloseHandle(hs[i]);
  }
#else
  pthread_t th[8];
  uint32_t spawned[8];
  for (uint32_t k = 1u; k < nc; k++) {
    if (pthread_create(&th[k], NULL, tok_worker_entry, &jobs[k]) == 0) spawned[k] = 1u;
    else { spawned[k] = 0u; tok_job_run(&jobs[k]); }
  }
  tok_job_run(&jobs[0]);
  for (uint32_t k = 1u; k < nc; k++)
    if (spawned[k]) pthread_join(th[k], NULL);
#endif
  uint32_t tot_add = 0u;
  for (uint32_t k = 0u; k < nc; k++) tot_add += jobs[k].out.n;
  if (tot_add > 0u && out->n + tot_add > out->cap) {
    uint32_t cap = out->cap ? out->cap : 64u;
    while (cap < out->n + tot_add) {
      if (cap > UINT32_MAX / 2u) { cap = out->n + tot_add; break; }
      cap *= 2u;
    }
    out->v = realloc(out->v, (size_t)cap * 4u);
    if (!out->v) ie_die("tok: out of memory");
    out->cap = cap;
  }
  for (uint32_t k = 0u; k < nc; k++) {
    tok_ids_append(out, jobs[k].out.v, jobs[k].out.n);
    tok_ids_free(&jobs[k].out);
  }
}

/* A part without special tokens. */
static void encode_plain(const tokenizer *t, const char *s, size_t len, tok_ids *out, int fast_bpe, int n_threads) {
  if (!len) return;
  if (n_threads <= 1 || len < 32768u) {
    encode_plain_st(t, s, len, out, fast_bpe);
    return;
  }
  /* Fast Path A: split at lossless '\n' + [A-Za-z0-9] boundaries (parallelizes decode + regex + BPE). */
  size_t cuts[9];
  uint32_t ncuts = 0u;
  cuts[ncuts++] = 0u;
  size_t prev = 0u;
  for (int k = 1; k < n_threads; k++) {
    const size_t target = (len * (size_t)k) / (size_t)n_threads;
    const size_t lo = prev + 4096u;
    const size_t hi = (len > 4096u) ? (len - 4096u) : 0u;
    if (lo >= hi) break;
    const size_t win = len / (size_t)(2 * n_threads);
    const size_t wlo = (target > win && target - win > lo) ? (target - win) : lo;
    const size_t whi = (target + win < hi) ? (target + win) : hi;
    const size_t p = find_safe_split(s, wlo, whi, target);
    if (p > prev && p < len) {
      cuts[ncuts++] = p;
      prev = p;
    }
  }
  cuts[ncuts++] = len;
  const uint32_t nc = ncuts - 1u;
  if (nc >= 2u) {
    tok_job jobs[8];
    memset(jobs, 0, sizeof jobs);
    for (uint32_t k = 0u; k < nc; k++) {
      jobs[k].t = t;
      jobs[k].s = s + cuts[k];
      jobs[k].len = cuts[k + 1u] - cuts[k];
      jobs[k].fast_bpe = fast_bpe;
    }
    tok_run_jobs(jobs, nc, out);
    return;
  }
  /* Fallback Path B: exact piece-boundary fan-out for single-line inputs (e.g. minified JSON). */
  if (t->pre == PRE_FLINT) {
    encode_plain_st(t, s, len, out, fast_bpe);
    return;
  }
  cps c;
  decode(s, len, &c);
  uint32_t *poff = malloc((size_t)(c.n + 1u) * 4u);
  if (!poff) ie_die("tok: out of memory");
  uint32_t np = 0u;
  for (uint32_t i = 0u; i < c.n;) {
    const uint32_t j = (t->pre == PRE_INKLING2 || t->pre == PRE_GPT4O) ? match_gpt4o(&c, i) : match(&c, i, t->pre);
    poff[np++] = c.off[i];
    i = j;
  }
  poff[np] = c.off[c.n];
  free(c.cp), free(c.off), free(c.cl);
  if (np < (uint32_t)n_threads * 64u) {
    for (uint32_t k = 0u; k < np; k++)
      bpe(t, s + poff[k], poff[k + 1u] - poff[k], out, fast_bpe);
    free(poff);
    return;
  }
  tok_job jobs[8];
  memset(jobs, 0, sizeof jobs);
  const uint32_t nt = (uint32_t)n_threads;
  for (uint32_t k = 0u; k < nt; k++) {
    jobs[k].t = t;
    jobs[k].s = s;
    jobs[k].poff = poff;
    jobs[k].p0 = (uint32_t)(((uint64_t)np * k) / nt);
    jobs[k].p1 = (uint32_t)(((uint64_t)np * (k + 1u)) / nt);
    jobs[k].fast_bpe = fast_bpe;
  }
  tok_run_jobs(jobs, nt, out);
  free(poff);
}

void tok_encode(const tokenizer *t, const char *text, size_t len, int parse_control, tok_ids *out) {
  const int fast_bpe = tok_fast_bpe_env();
  const int n_threads = tok_threads_env();
  size_t a = 0; /* start of the current plain part */
  for (size_t i = 0; i < len;) {
    const special *hit = NULL;
    if (t->sp_first[(unsigned char)text[i]])
      for (uint32_t k = 0; k < t->n_sp; k++) { /* longest first */
        const special *sp = &t->sp[k];
        if ((parse_control || !sp->control) && sp->n <= len - i && !memcmp(text + i, sp->s, sp->n)) { hit = sp; break; }
      }
    if (!hit) { i++; continue; }
    encode_plain(t, text + a, i - a, out, fast_bpe, n_threads);
    tok_ids_push(out, hit->id);
    i += hit->n, a = i;
  }
  encode_plain(t, text + a, len - a, out, fast_bpe, n_threads);
}
