/* lab.c -- ie-lab: one model and its MTP head resident on the GPU for weight-editing experiments. A tensor is
 * changed in place (no reload), measured with teacher-forced captures and greedy speculative runs, and the edits
 * leave as a delta file (a GGUF holding only the edited tensors).
 *   ie-lab MODEL.gguf [--mtp FILE] [--port 8090] [--ctx 8192] [--chunk 256] [--hsaco FILE]
 *   ie-lab --apply BASE.gguf DELTA.gguf OUT.gguf   (no GPU) BASE with DELTA's tensors written in, after checking
 *                                                   that each replaced tensor is the one the delta was made from
 * HTTP on 127.0.0.1, one request at a time on the thread that owns the GPU. Tensors are named as in the GGUF and
 * sent as GGUF bytes of their own type and size (Q8_0, F32, ...):
 *   GET  /health, /info                  the model, its dimensions, its tensors (name, type, shape, bytes, edited)
 *   GET  /tensor?name=N                   the tensor's current bytes (the edit, or the file's)
 *   POST /tensor?name=N  (body: bytes)    replace it in place; bytes equal to the file's drop the edit
 *   POST /reset[?name=N]                  back to the file's bytes (every edited tensor without name)
 *   POST /capture {"tokens_file": F | "tokens": [...], "out": FILE}
 *        a teacher-forced prompt run that writes the MTP capture records (gen.c mtp_capture) to FILE
 *   POST /spec {"files": [F, ...] | "tokens": [...], "n": 768, "draft": 7, "pmin": 0.35, "ids": false, "ab": false}
 *        greedy speculative decoding per prompt, as ie-run --draft D --stop: steps, drafts made and accepted (also
 *        per draft depth), ms; "ab": the same prompts again with the file's weights (swapped in, then back), and
 *        whether the output ids agree
 *   POST /chain {"files": [...], "from": [...], "out": FILE}   the MTP draft chain at every answer position (h_chain)
 *   POST /delta/save {"path": P}          the edited tensors as a GGUF (types and shapes of the file; the key
 *        ie_lab.base_hash.<name> holds the FNV-1a 64 hash of the file's bytes of that tensor)
 *   POST /delta/load {"path": P}          patch a delta in (each hash must match the loaded file)
 *   POST /quit
 * The token embedding and the output matrix are refused: the MTP draft head is a copy of the output matrix. */
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include <strings.h>

#include "backend.h"
#include "common.h"
#include "gen.h"
#include "gguf.h"
#include "json.h"
#include "model.h"
#include "win_net.h" /* after the other headers */

static struct {
  const char *path, *mtp, *hsaco;
  int port;
  uint32_t ctx, chunk;
} O = {.hsaco = "build/ie_kernels.hsaco", .port = 8090, .ctx = 8192, .chunk = 256};

static gguf_file G, G2;
static model M;
static graph GR;
static backend *B;
static gen_ctx GC;
static int64_t EOS = -1;

/* ---- edits: the current bytes of each edited tensor (by name) ---- */
typedef struct {
  char name[96];
  uint8_t *bytes;
  uint64_t n;
} edit;
static edit *ED;
static uint32_t N_ED, CAP_ED;

static edit *edit_find(const char *name) {
  for (uint32_t i = 0; i < N_ED; i++)
    if (!strcmp(ED[i].name, name)) return &ED[i];
  return NULL;
}

static uint64_t fnv64(const uint8_t *p, uint64_t n) {
  uint64_t h = 1469598103934665603ull;
  for (uint64_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

/* Write bytes into every registry entry named name (the device copies). err: why not (then nothing changed if
 * the failure is in the checks; a partial write is reported as such). */
static int patch_in(const char *name, const uint8_t *bytes, uint64_t n, char *err, size_t errn) {
  if (!strcmp(name, "token_embd.weight") || !strcmp(name, "output.weight")) {
    snprintf(err, errn, "%s is refused: the MTP draft head is a copy of the output matrix", name);
    return 0;
  }
  const ie_tref *r0 = ie_tref_find(name);
  if (!r0) { snprintf(err, errn, "no tensor %s", name); return 0; }
  if (n != ie_tref_bytes(r0)) {
    snprintf(err, errn, "%s: %llu bytes, expected %llu (%s)", name, (unsigned long long)n, (unsigned long long)ie_tref_bytes(r0),
             ggml_type_name(r0->src->type));
    return 0;
  }
  /* decode every matrix entry first: a kind the device layout cannot take changes nothing */
  mat parts[4];
  const ie_tref *rs[4];
  int np = 0;
  for (uint32_t i = 0; i < ie_n_trefs; i++) {
    const ie_tref *r = &ie_trefs[i];
    if (strcmp(r->name, name)) continue;
    if (np == 4) { snprintf(err, errn, "%s: too many entries", name); return 0; }
    rs[np] = r;
    memset(&parts[np], 0, sizeof parts[np]);
    if (r->w && !ie_tref_decode(r, bytes, &parts[np])) {
      for (int k = 0; k <= np; k++) ie_mat_free(&parts[k]);
      snprintf(err, errn, "%s: decodes to another matrix kind than the loaded one", name);
      return 0;
    }
    np++;
  }
  for (int k = 0; k < np; k++) {
    const ie_tref *r = rs[k];
    if (r->w) {
      const int ok = gpu_patch_mat(B, r->w, r->r0, &parts[k]);
      ie_mat_free(&parts[k]);
      if (!ok) {
        for (int j = k + 1; j < np; j++) ie_mat_free(&parts[j]);
        snprintf(err, errn, "%s: its matrix kind has no in-place layout (Q4_0, Q8_0, F32 have)%s", name, k ? " (partly written)" : "");
        return 0;
      }
    } else {
      ie_tref_set_vec(r, bytes);
      gpu_patch_vec(B, r->v);
    }
  }
  return 1;
}

/* Patch and record. */
static int edit_set(const char *name, const uint8_t *bytes, uint64_t n, char *err, size_t errn) {
  if (!patch_in(name, bytes, n, err, errn)) return 0;
  const ie_tref *r = ie_tref_find(name);
  edit *e = edit_find(name);
  if (!memcmp(bytes, r->src->data, n)) { /* the file's bytes: no edit */
    if (e) {
      free(e->bytes);
      *e = ED[--N_ED];
    }
    return 1;
  }
  if (!e) {
    if (N_ED == CAP_ED) {
      CAP_ED = CAP_ED ? 2 * CAP_ED : 64;
      ED = realloc(ED, CAP_ED * sizeof *ED);
      if (!ED) ie_die("out of memory");
    }
    e = &ED[N_ED++];
    memset(e, 0, sizeof *e);
    snprintf(e->name, sizeof e->name, "%s", name);
  }
  free(e->bytes);
  e->bytes = ie_alloc(n);
  memcpy(e->bytes, bytes, n);
  e->n = n;
  return 1;
}

/* With file = 1 the file's bytes of every edited tensor go in (the edits stay recorded); 0: the edits again. */
static void swap_all(int file) {
  char err[256];
  for (uint32_t i = 0; i < N_ED; i++) {
    const ie_tref *r = ie_tref_find(ED[i].name);
    if (!patch_in(ED[i].name, file ? (const uint8_t *)r->src->data : ED[i].bytes, ED[i].n, err, sizeof err)) ie_die("swap: %s", err);
  }
}

/* ---- tokens ---- */
typedef struct {
  uint32_t *v;
  uint32_t n, cap;
} ids;
static void ids_push(ids *t, uint32_t x) {
  if (t->n == t->cap) {
    t->cap = t->cap ? 2 * t->cap : 1024;
    t->v = realloc(t->v, t->cap * 4);
    if (!t->v) ie_die("out of memory");
  }
  t->v[t->n++] = x;
}
/* token ids separated by commas or white space (as ie-run --tokens-file) */
static int ids_file(const char *path, ids *t) {
  FILE *f = fopen(path, "rb");
  if (!f) return 0;
  int c;
  uint64_t x = 0;
  int have = 0;
  while ((c = fgetc(f)) != EOF) {
    if (c >= '0' && c <= '9') x = x * 10 + (uint64_t)(c - '0'), have = 1;
    else if (have) ids_push(t, (uint32_t)x), x = 0, have = 0;
  }
  if (have) ids_push(t, (uint32_t)x);
  fclose(f);
  return 1;
}
/* the prompt of a request object: "tokens_file" or "tokens" */
static int req_tokens(const jval *o, ids *t, char *err, size_t errn) {
  const jval *f = json_get(o, "tokens_file"), *a = json_get(o, "tokens");
  if (f && f->t == J_STR) {
    if (!ids_file(f->str, t)) { snprintf(err, errn, "cannot read %s", f->str); return 0; }
  } else if (a && a->t == J_ARR) {
    for (const jval *x = a->kid; x; x = x->next) ids_push(t, (uint32_t)x->num);
  } else {
    snprintf(err, errn, "give tokens_file or tokens");
    return 0;
  }
  for (uint32_t i = 0; i < t->n; i++)
    if (t->v[i] >= M.vocab) { snprintf(err, errn, "token %u >= vocab %u", t->v[i], M.vocab); return 0; }
  if (!t->n) { snprintf(err, errn, "no tokens"); return 0; }
  return 1;
}

/* ---- runs ---- */
typedef struct {
  uint32_t n_prompt, n_out, gen, steps, acc, drafted; /* gen: the output tokens after the first (made by the prompt run) */
  uint32_t at_drafted[9], at_acc[9]; /* per draft depth 1 .. 8 */
  double prefill_ms, gen_ms;
  uint64_t ids_hash;
  ids out;
} run_res;

static void run_spec(const ids *p, uint32_t n_new, uint32_t D, float pmin, run_res *r, int keep_ids) {
  memset(r, 0, sizeof *r);
  sampler *sp = &GC.sp;
  sp->temp = 0.0f, sp->top_p = 1.0f, sp->top_k = 0, sp->dry_mult = 0.0f, sp->dry_reset = -1;
  dry_begin(sp);
  uint32_t hrow = 0;
  const double t0 = now_ms();
  uint32_t first = prefill(B, &GR, M.vocab, p->v, p->n, O.chunk, 1, NULL, &hrow);
  r->prefill_ms = now_ms() - t0;
  uint32_t total = p->n + n_new, *out = ie_alloc((size_t)total * 4);
  memcpy(out, p->v, (size_t)p->n * 4);
  uint32_t nk[81] = {0};
  if ((int64_t)first == EOS) { /* as ie-run --stop: the answer is the eos token */
    out[p->n] = first, total = p->n + 1;
  } else {
    ie_spec_nk = nk;
    total = spec_generate(B, &M, &GR, out, p->n, total, D, EOS, 1, first, hrow, sp, M.vocab, &r->gen_ms, &r->steps, &r->acc, pmin,
                          &r->drafted, NULL, NULL, &EOS, 1, NULL, NULL);
    ie_spec_nk = NULL;
  }
  r->n_prompt = p->n, r->n_out = total - p->n, r->gen = r->n_out > 0 ? r->n_out - 1 : 0;
  for (uint32_t nd = 0; nd <= 8; nd++)
    for (uint32_t k = 0; k <= 8; k++)
      for (uint32_t j = 1; j <= 8; j++) {
        if (nd >= j) r->at_drafted[j] += nk[nd * 9 + k];
        if (k >= j) r->at_acc[j] += nk[nd * 9 + k];
      }
  r->ids_hash = fnv64((const uint8_t *)out, (uint64_t)total * 4);
  if (keep_ids)
    for (uint32_t i = p->n; i < total; i++) ids_push(&r->out, out[i]);
  free(out);
}

static void res_json(sbuf *b, const run_res *r, int with_ids) {
  sb_printf(b, "{\"prompt\":%u,\"out\":%u,\"steps\":%u,\"drafted\":%u,\"accepted\":%u,\"tok_per_step\":%.4f,\"accept_pct\":%.3f,"
               "\"prefill_ms\":%.2f,\"gen_ms\":%.2f,\"ms_per_tok\":%.4f,\"ids_hash\":\"%016llx\",\"depth_drafted\":[",
            r->n_prompt, r->n_out, r->steps, r->drafted, r->acc, r->steps ? (double)r->gen / r->steps : 0.0,
            r->drafted ? 100.0 * r->acc / r->drafted : 0.0, r->prefill_ms, r->gen_ms,
            r->gen ? r->gen_ms / r->gen : 0.0, (unsigned long long)r->ids_hash);
  for (int j = 1; j <= 8; j++) sb_printf(b, "%s%u", j > 1 ? "," : "", r->at_drafted[j]);
  sb_puts(b, "],\"depth_accepted\":[");
  for (int j = 1; j <= 8; j++) sb_printf(b, "%s%u", j > 1 ? "," : "", r->at_acc[j]);
  sb_puts(b, "]");
  if (with_ids) {
    sb_puts(b, ",\"ids\":[");
    for (uint32_t i = 0; i < r->out.n; i++) sb_printf(b, "%s%u", i ? "," : "", r->out.v[i]);
    sb_puts(b, "]");
  }
  sb_puts(b, "}");
}

static void tot_add(run_res *t, const run_res *r) {
  t->n_prompt += r->n_prompt, t->n_out += r->n_out, t->gen += r->gen, t->steps += r->steps, t->acc += r->acc, t->drafted += r->drafted;
  t->prefill_ms += r->prefill_ms, t->gen_ms += r->gen_ms;
  for (int j = 0; j < 9; j++) t->at_drafted[j] += r->at_drafted[j], t->at_acc[j] += r->at_acc[j];
  t->ids_hash ^= r->ids_hash;
}

/* ---- delta files ---- */
static void w_u32(FILE *f, uint32_t x) { fwrite(&x, 4, 1, f); }
static void w_u64(FILE *f, uint64_t x) { fwrite(&x, 8, 1, f); }
static void w_str(FILE *f, const char *s) { w_u64(f, strlen(s)), fwrite(s, 1, strlen(s), f); }
static void w_pad(FILE *f) {
  static const uint8_t z[32] = {0};
  const long p = ftell(f);
  if (p % 32) fwrite(z, 1, 32 - (size_t)(p % 32), f);
}

static int delta_save(const char *path, char *err, size_t errn) {
  FILE *f = fopen(path, "wb");
  if (!f) { snprintf(err, errn, "cannot write %s", path); return 0; }
  fwrite("GGUF", 1, 4, f);
  w_u32(f, 3), w_u64(f, N_ED), w_u64(f, 3 + N_ED);
  w_str(f, "general.architecture"), w_u32(f, GGUF_STR), w_str(f, "ie_lab_delta");
  w_str(f, "ie_lab.base"), w_u32(f, GGUF_STR), w_str(f, O.path);
  w_str(f, "ie_lab.mtp"), w_u32(f, GGUF_STR), w_str(f, O.mtp ? O.mtp : "");
  for (uint32_t i = 0; i < N_ED; i++) {
    const ie_tref *r = ie_tref_find(ED[i].name);
    char k[160];
    snprintf(k, sizeof k, "ie_lab.base_hash.%s", ED[i].name);
    w_str(f, k), w_u32(f, GGUF_U64), w_u64(f, fnv64(r->src->data, r->src->nbytes));
  }
  uint64_t off = 0;
  for (uint32_t i = 0; i < N_ED; i++) {
    const gguf_tensor *t = ie_tref_find(ED[i].name)->src;
    w_str(f, ED[i].name), w_u32(f, t->n_dims);
    for (uint32_t d = 0; d < t->n_dims; d++) w_u64(f, t->ne[d]);
    w_u32(f, t->type), w_u64(f, off);
    off += (ED[i].n + 31) / 32 * 32;
  }
  w_pad(f);
  for (uint32_t i = 0; i < N_ED; i++) fwrite(ED[i].bytes, 1, ED[i].n, f), w_pad(f);
  const int ok = !ferror(f);
  fclose(f);
  if (!ok) snprintf(err, errn, "write error on %s", path);
  return ok;
}

/* The expected hash of a delta tensor's original (0: none recorded). */
static uint64_t delta_hash(const gguf_file *d, const char *name) {
  char k[160];
  snprintf(k, sizeof k, "ie_lab.base_hash.%s", name);
  const gguf_kv *kv = gguf_find(d, k);
  return kv && kv->type == GGUF_U64 ? kv->v.u : 0;
}

static int delta_load(const char *path, char *err, size_t errn, uint32_t *n) {
  gguf_file d;
  gguf_open(&d, path);
  *n = 0;
  for (uint64_t i = 0; i < d.n_tensors; i++) { /* check all first */
    const gguf_tensor *t = &d.t[i];
    const ie_tref *r = ie_tref_find(t->name);
    if (!r) { snprintf(err, errn, "%s: no such tensor in the model", t->name); gguf_close(&d); return 0; }
    if (t->type != r->src->type || t->nbytes != r->src->nbytes) { snprintf(err, errn, "%s: type or size differ", t->name); gguf_close(&d); return 0; }
    const uint64_t h = delta_hash(&d, t->name);
    if (h && h != fnv64(r->src->data, r->src->nbytes)) {
      snprintf(err, errn, "%s: the delta was made from another original tensor", t->name);
      gguf_close(&d);
      return 0;
    }
  }
  for (uint64_t i = 0; i < d.n_tensors; i++) {
    if (!edit_set(d.t[i].name, d.t[i].data, d.t[i].nbytes, err, errn)) { gguf_close(&d); return 0; }
    (*n)++;
  }
  gguf_close(&d);
  return 1;
}

/* ie-lab --apply BASE DELTA OUT */
static int apply_main(const char *base, const char *delta, const char *out) {
  gguf_file b, d;
  gguf_open(&b, base);
  gguf_open(&d, delta);
  for (uint64_t i = 0; i < d.n_tensors; i++) {
    const gguf_tensor *t = &d.t[i], *s = gguf_tensor_find(&b, t->name);
    if (!s) ie_die("%s: no such tensor in %s", t->name, base);
    if (s->type != t->type || s->nbytes != t->nbytes) ie_die("%s: type or size differ", t->name);
    const uint64_t h = delta_hash(&d, t->name);
    if (h && h != fnv64(s->data, s->nbytes)) ie_die("%s: the delta was made from another original tensor", t->name);
  }
  FILE *src = fopen(base, "rb"), *dst = fopen(out, "wb");
  if (!src || !dst) ie_die("cannot open %s or %s", base, out);
  static uint8_t buf[1 << 20];
  size_t k;
  while ((k = fread(buf, 1, sizeof buf, src)) > 0)
    if (fwrite(buf, 1, k, dst) != k) ie_die("write error on %s", out);
  fclose(src);
  for (uint64_t i = 0; i < d.n_tensors; i++) {
    const gguf_tensor *t = &d.t[i], *s = gguf_tensor_find(&b, t->name);
    if (fseeko(dst, (off_t)(b.data_off + s->offset), SEEK_SET) || fwrite(t->data, 1, t->nbytes, dst) != t->nbytes)
      ie_die("write error on %s", out);
  }
  if (fclose(dst)) ie_die("write error on %s", out);
  fprintf(stderr, "ie-lab: %s = %s with %llu tensors of %s\n", out, base, (unsigned long long)d.n_tensors, delta);
  gguf_close(&d), gguf_close(&b);
  return 0;
}

/* ---- HTTP ---- */
typedef struct {
  char method[16], path[256], query[512];
  char *body;
  size_t body_n;
} request;

static int send_all(int fd, const char *p, size_t n) {
  while (n) {
    ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return -1;
    p += k, n -= (size_t)k;
  }
  return 0;
}
static void reply(int fd, int code, const char *ctype, const char *body, size_t n) {
  sbuf h = {0};
  sb_printf(&h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", code,
            code == 200 ? "OK" : code == 404 ? "Not Found" : "Error", ctype, n);
  send_all(fd, h.p, h.n);
  if (n) send_all(fd, body, n);
  sb_free(&h);
}
static void reply_json(int fd, sbuf *b) { reply(fd, 200, "application/json", b->p ? b->p : "{}", b->n); }
static void reply_err(int fd, int code, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
static void reply_err(int fd, int code, const char *fmt, ...) {
  char msg[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof msg, fmt, ap);
  va_end(ap);
  sbuf b = {0};
  sb_puts(&b, "{\"error\":");
  sb_json_str(&b, msg, strlen(msg));
  sb_puts(&b, "}");
  reply(fd, code, "application/json", b.p, b.n);
  sb_free(&b);
  fprintf(stderr, "ie-lab: error %d: %s\n", code, msg);
}

static int read_request(int fd, request *r) {
  memset(r, 0, sizeof *r);
  char *buf = malloc(65536 + 1);
  size_t n = 0;
  char *he = NULL;
  while (!he) {
    if (n == 65536) { free(buf); return -1; }
    ssize_t k = recv(fd, buf + n, 65536 - n, 0);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) { free(buf); return -1; }
    n += (size_t)k;
    buf[n] = 0;
    he = strstr(buf, "\r\n\r\n");
  }
  *he = 0;
  const size_t head_n = (size_t)(he - buf) + 4;
  char target[1024];
  if (sscanf(buf, "%15s %1023s", r->method, target) != 2) { free(buf); return -1; }
  char *q = strchr(target, '?');
  if (q) *q++ = 0;
  snprintf(r->path, sizeof r->path, "%s", target);
  snprintf(r->query, sizeof r->query, "%s", q ? q : "");
  size_t len = 0;
  for (char *line = strstr(buf, "\r\n"); line; line = strstr(line, "\r\n")) {
    line += 2;
    if (!strncasecmp(line, "Content-Length:", 15)) len = strtoull(line + 15, NULL, 10);
  }
  if (len > ((size_t)1 << 30)) { free(buf); return -1; }
  r->body = malloc(len + 1);
  size_t have = n - head_n < len ? n - head_n : len;
  memcpy(r->body, buf + head_n, have);
  while (have < len) {
    ssize_t k = recv(fd, r->body + have, len - have, 0);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) { free(buf), free(r->body); return -1; }
    have += (size_t)k;
  }
  r->body[len] = 0;
  r->body_n = len;
  free(buf);
  return 0;
}

/* the value of key in a query string (%XX decoded), or "" */
static void query_get(const char *q, const char *key, char *out, size_t on) {
  out[0] = 0;
  const size_t kn = strlen(key);
  for (const char *p = q; p && *p;) {
    const char *e = strchr(p, '&');
    if (!strncmp(p, key, kn) && p[kn] == '=') {
      const char *v = p + kn + 1, *ve = e ? e : v + strlen(v);
      size_t o = 0;
      for (; v < ve && o + 1 < on; v++) {
        if (*v == '%' && ve - v >= 3) {
          char hx[3] = {v[1], v[2], 0};
          out[o++] = (char)strtol(hx, NULL, 16);
          v += 2;
        } else out[o++] = *v == '+' ? ' ' : *v;
      }
      out[o] = 0;
      return;
    }
    p = e ? e + 1 : NULL;
  }
}

static double num_or(const jval *o, const char *k, double d) {
  const jval *x = json_get(o, k);
  return x && x->t == J_NUM ? x->num : d;
}
static int bool_or(const jval *o, const char *k, int d) {
  const jval *x = json_get(o, k);
  return x ? x->t == J_TRUE : d;
}

static void h_info(int fd) {
  sbuf b = {0};
  sb_puts(&b, "{\"model\":");
  sb_json_str(&b, O.path, strlen(O.path));
  sb_puts(&b, ",\"mtp\":");
  sb_json_str(&b, O.mtp ? O.mtp : "", O.mtp ? strlen(O.mtp) : 0);
  sb_printf(&b, ",\"dim\":%u,\"layers\":%u,\"ffn\":%u,\"heads\":%u,\"kv_heads\":%u,\"head_dim\":%u,\"vocab\":%u,\"ctx\":%u,"
                "\"edited\":%u,\"tensors\":[",
            M.dim, M.n_layer, M.ffn, M.n_head, M.n_kv, M.hd, M.vocab, GR.n_ctx, N_ED);
  int first = 1;
  for (uint32_t i = 0; i < ie_n_trefs; i++) {
    const ie_tref *r = &ie_trefs[i];
    if (ie_tref_find(r->name) != r) continue; /* a name once */
    const gguf_tensor *t = r->src;
    sb_printf(&b, "%s{\"name\":", first ? "" : ",");
    sb_json_str(&b, r->name, strlen(r->name));
    sb_printf(&b, ",\"type\":\"%s\",\"shape\":[%llu,%llu],\"bytes\":%llu,\"edited\":%s,\"fused\":%s}", ggml_type_name(t->type),
              (unsigned long long)t->ne[1], (unsigned long long)t->ne[0], (unsigned long long)t->nbytes, edit_find(r->name) ? "true" : "false",
              r->w && r->rows != r->w->rows ? "true" : "false");
    first = 0;
  }
  sb_puts(&b, "]}");
  reply_json(fd, &b);
  sb_free(&b);
}

static void h_spec(int fd, const jval *o) {
  char err[512];
  const uint32_t n_new = (uint32_t)num_or(o, "n", 768), D = (uint32_t)num_or(o, "draft", 7);
  const float pmin = (float)num_or(o, "pmin", 0.35);
  const int with_ids = bool_or(o, "ids", 0), ab = bool_or(o, "ab", 0) && N_ED;
  if (D < 1 || D > MTP_DRAFT_MAX) { reply_err(fd, 400, "draft: 1 to 8"); return; }
  /* the prompts: "files" or one prompt */
  const jval *files = json_get(o, "files");
  uint32_t np = 0;
  ids *ps = NULL;
  char **names = NULL;
  if (files && files->t == J_ARR) {
    for (const jval *x = files->kid; x; x = x->next) np++;
    ps = calloc(np, sizeof *ps), names = calloc(np, sizeof *names);
    uint32_t i = 0;
    for (const jval *x = files->kid; x; x = x->next, i++) {
      if (x->t != J_STR || !ids_file(x->str, &ps[i])) { reply_err(fd, 400, "cannot read a prompt file"); goto done; }
      names[i] = x->str;
    }
  } else {
    np = 1, ps = calloc(1, sizeof *ps), names = calloc(1, sizeof *names);
    if (!req_tokens(o, &ps[0], err, sizeof err)) { reply_err(fd, 400, "%s", err); goto done; }
    names[0] = "tokens";
  }
  for (uint32_t i = 0; i < np; i++)
    if (ps[i].n + n_new + spec_room(D) > GR.n_ctx) { reply_err(fd, 400, "%s: prompt + n is past the context %u", names[i], GR.n_ctx); goto done; }
  {
    run_res *re = calloc(np, sizeof *re), *ro = ab ? calloc(np, sizeof *ro) : NULL, te = {0}, to = {0};
    for (uint32_t i = 0; i < np; i++) run_spec(&ps[i], n_new, D, pmin, &re[i], with_ids), tot_add(&te, &re[i]);
    uint32_t agree = 0;
    if (ab) {
      swap_all(1);
      for (uint32_t i = 0; i < np; i++) run_spec(&ps[i], n_new, D, pmin, &ro[i], with_ids), tot_add(&to, &ro[i]), agree += ro[i].ids_hash == re[i].ids_hash;
      swap_all(0);
    }
    sbuf b = {0};
    sb_printf(&b, "{\"edits\":%u,\"total\":", N_ED);
    res_json(&b, &te, 0);
    if (ab) sb_puts(&b, ",\"original_total\":"), res_json(&b, &to, 0), sb_printf(&b, ",\"ids_agree\":%u,\"prompts\":%u", agree, np);
    sb_puts(&b, ",\"per\":[");
    for (uint32_t i = 0; i < np; i++) {
      sb_printf(&b, "%s{\"name\":", i ? "," : "");
      sb_json_str(&b, names[i], strlen(names[i]));
      sb_puts(&b, ",\"edited\":"), res_json(&b, &re[i], with_ids);
      if (ab) sb_puts(&b, ",\"original\":"), res_json(&b, &ro[i], with_ids);
      sb_puts(&b, "}");
    }
    sb_puts(&b, "]}");
    reply_json(fd, &b);
    sb_free(&b);
    for (uint32_t i = 0; i < np; i++) free(re[i].out.v), free(ro ? ro[i].out.v : NULL);
    free(re), free(ro);
  }
done:
  for (uint32_t i = 0; i < np; i++) free(ps[i].v);
  free(ps), free(names);
}

static void h_capture(int fd, const jval *o) {
  char err[512];
  ids t = {0};
  const jval *out = json_get(o, "out");
  if (!out || out->t != J_STR) { reply_err(fd, 400, "give out (a file)"); return; }
  if (!req_tokens(o, &t, err, sizeof err)) { reply_err(fd, 400, "%s", err); free(t.v); return; }
  if (t.n + 1 > GR.n_ctx) { reply_err(fd, 400, "%u tokens: past the context %u", t.n, GR.n_ctx); free(t.v); return; }
  FILE *f = fopen(out->str, "wb");
  if (!f) { reply_err(fd, 400, "cannot write %s", out->str); free(t.v); return; }
  uint32_t hrow;
  const double t0 = now_ms();
  ie_mtp_cap_file = f;
  prefill(B, &GR, M.vocab, t.v, t.n, O.chunk, 1, NULL, &hrow);
  ie_mtp_cap_file = NULL;
  const long bytes = ftell(f);
  fclose(f);
  sbuf b = {0};
  sb_printf(&b, "{\"positions\":%u,\"bytes\":%ld,\"ms\":%.2f,\"edits\":%u}", t.n, bytes, now_ms() - t0, N_ED);
  reply_json(fd, &b);
  sb_free(&b);
  free(t.v);
}

/* POST /chain {"files": [F, ...], "from": [n, ...], "draft": 7, "topk": 8, "out": FILE}: each file holds a whole
 * text (a prompt and its greedy answer from n on). At every position P >= n the MTP chain runs as spec_generate
 * drafts it, without the pmin cut: draft 1 from (h(P-1), token P), draft j from (the head's own g, draft j-1), with
 * the head's KV cache teacher-forced below P. Records (40 bytes) after the header {"MTCH", D, K}: text, P, depth j,
 * the text's token at P + j (0xFFFFFFFF past the end), the draft, its probability, the rank of the text's token in
 * the head's top K (-1: not there), its probability (0 when not in the top K), the head's second token and its
 * probability. Every pmin can be simulated from
 * one call: a step at P keeps drafts while their probability is at least pmin and accepts while they match.
 * "cap": FILE also writes, for each draft whose earlier drafts all match the text (the states a real chain reaches),
 * the terms of a fit of the head's output toward the model's: header {"MTCC", dim, ffn}, then per record text, P,
 * depth j, rms of the layer output y (before shared_head_norm), the ffn_down input a (ffn), the head's g (dim) and
 * the model's own normed hidden state h(P + j - 1) (dim), the h whose logits g should give. A record whose h is not
 * in the same chunk of model rows (the last j - 1 positions of a chunk) is left out. */
static void h_chain(int fd, const jval *o) {
  const jval *files = json_get(o, "files"), *from = json_get(o, "from"), *out = json_get(o, "out"), *cap = json_get(o, "cap");
  const uint32_t D = (uint32_t)num_or(o, "draft", 7), K = (uint32_t)num_or(o, "topk", 8);
  if (!files || files->t != J_ARR || !from || from->t != J_ARR || !out || out->t != J_STR) {
    reply_err(fd, 400, "give files, from (the answer start of each) and out");
    return;
  }
  if (D < 1 || D > MTP_DRAFT_MAX || K < 1 || K > 16) { reply_err(fd, 400, "draft: 1 to 8, topk: 1 to 16"); return; }
  FILE *f = fopen(out->str, "wb");
  if (!f) { reply_err(fd, 400, "cannot write %s", out->str); return; }
  const uint32_t hd[3] = {0x4843544Du, D, K}, nv = M.out_draft.rows ? M.out_draft.rows : M.vocab;
  fwrite(hd, 4, 3, f);
  FILE *cf = NULL;
  const uint32_t dim = GR.bufs[GR.mtp_g].stride / 4u, ffn = GR.bufs[GR.mtp_a].stride / 8u;
  float *cb = NULL;
  if (cap && cap->t == J_STR) {
    if (GR.mtp_y < 0 || !(cf = fopen(cap->str, "wb"))) { reply_err(fd, 400, "cannot write %s", cap->str); fclose(f); return; }
    const uint32_t ch[3] = {0x4343544Du, dim, ffn};
    fwrite(ch, 4, 3, cf);
    cb = malloc((size_t)(2 * ffn + 3 * dim + 4096) * 4); /* + room for a padded row stride */
  }
  const double t0 = now_ms();
  uint32_t *a = malloc(O.chunk * 4), seq = 0, recs = 0, crecs = 0;
  float val[16];
  uint32_t idx[16];
  const jval *x = files->kid, *y = from->kid;
  for (; x && y; x = x->next, y = y->next, seq++) {
    ids t = {0};
    if (x->t != J_STR || y->t != J_NUM || !ids_file(x->str, &t)) { reply_err(fd, 400, "cannot read %s", x->t == J_STR ? x->str : "?"); goto fail; }
    const uint32_t n0 = (uint32_t)y->num, n = t.n;
    if (n0 < 1 || n0 >= n || n + D + 1 > GR.n_ctx) { reply_err(fd, 400, "%s: from %u of %u tokens (context %u)", x->str, n0, n, GR.n_ctx); free(t.v); goto fail; }
    for (uint32_t i = 0; i < n; i++)
      if (t.v[i] >= M.vocab) { reply_err(fd, 400, "%s: token %u >= vocab", x->str, t.v[i]); free(t.v); goto fail; }
    uint32_t hrow, T;
    prefill(B, &GR, M.vocab, t.v, n0, O.chunk, 1, NULL, &hrow); /* the prompt: h(n0 - 1) in h_out row hrow */
    for (uint32_t p0 = n0; p0 < n; p0 += T) {
      T = n - p0 < O.chunk ? n - p0 : O.chunk;
      B->copy_rows(B, GR.mtp_h, 1, GR.h_out, hrow, 1); /* h(p0 - 1), kept in row 1 (the chain uses row 0) */
      B->run(B, 0, t.v + p0, T, p0, 1, a, NULL);
      B->accept(B, T - 1);
      hrow = T - 1;
      for (uint32_t u = 0; u < T; u++) {
        const uint32_t P = p0 + u;
        if (u == 0) B->copy_rows(B, GR.mtp_h, 0, GR.mtp_h, 1, 1);
        else B->copy_rows(B, GR.mtp_h, 0, GR.h_out, u - 1, 1);
        uint32_t tin = t.v[P], tk;
        float pr;
        int on = 1; /* drafts 1 .. j-1 match the text */
        for (uint32_t j = 1; j <= D; j++) {
          if (j > 1) B->copy_rows(B, GR.mtp_h, 0, GR.mtp_g, 0, 1);
          B->run(B, 1, &tin, 1, P + j - 1, 0, &tk, &pr);
          if (!B->topk(B, GR.mtp_logits, 0, 1, nv, K, val, idx)) { reply_err(fd, 500, "top K does not fit"); free(t.v); goto fail; }
          const uint32_t d = model_draft_tok(&M, tk), lab = P + j < n ? t.v[P + j] : 0xFFFFFFFFu;
          const float lz = val[0] - logf(pr > 1e-30f ? pr : 1e-30f);
          int32_t rank = -1;
          float pt = 0.0f;
          for (uint32_t c = 0; c < K; c++)
            if (model_draft_tok(&M, idx[c]) == lab) { rank = (int32_t)c, pt = expf(val[c] - lz); break; }
          const uint32_t rec[5] = {seq, P, j, lab, d}, d2 = K > 1 ? model_draft_tok(&M, idx[1]) : 0xFFFFFFFFu;
          const float p2 = K > 1 ? expf(val[1] - lz) : 0.0f;
          fwrite(rec, 4, 5, f), fwrite(&pr, 4, 1, f), fwrite(&rank, 4, 1, f), fwrite(&pt, 4, 1, f), fwrite(&d2, 4, 1, f), fwrite(&p2, 4, 1, f);
          recs++;
          if (cf && on && u + j - 1 < T) {
            float *ab = cb, *gb = ab + 2 * ffn, *yb = gb + dim, *hb = yb + dim;
            B->read_rows(B, GR.mtp_a, 0, 1, ab);
            B->read_rows(B, GR.mtp_g, 0, 1, gb);
            B->read_rows(B, GR.mtp_y, 0, 1, yb);
            B->read_rows(B, GR.h_out, u + j - 1, 1, hb);
            double ss = 0;
            for (uint32_t i = 0; i < dim; i++) ss += (double)yb[i] * yb[i];
            const float ry = (float)sqrt(ss / dim);
            const uint32_t id[3] = {seq, P, j};
            fwrite(id, 4, 3, cf), fwrite(&ry, 4, 1, cf), fwrite(ab, 4, ffn, cf), fwrite(gb, 4, dim, cf), fwrite(hb, 4, dim, cf);
            crecs++;
          }
          on = on && d == lab;
          tin = d;
        }
      }
    }
    free(t.v);
  }
  fclose(f);
  if (cf) fclose(cf);
  free(a), free(cb);
  sbuf b = {0};
  sb_printf(&b, "{\"texts\":%u,\"records\":%u,\"cap_records\":%u,\"ms\":%.2f,\"edits\":%u}", seq, recs, crecs, now_ms() - t0, N_ED);
  reply_json(fd, &b);
  sb_free(&b);
  return;
fail:
  fclose(f);
  if (cf) fclose(cf);
  free(a), free(cb);
}

static void handle(int fd, const request *r) {
  char err[512], name[256];
  const int post = !strcmp(r->method, "POST");
  jval *o = NULL;
  if (post && r->body_n && r->body[0] == '{') {
    o = json_parse(r->body, r->body_n, err, sizeof err);
    if (!o) { reply_err(fd, 400, "%s", err); return; }
  }
  const jval *body = o ? o : &(jval){.t = J_OBJ};
  if (!strcmp(r->path, "/health")) {
    reply(fd, 200, "application/json", "{\"status\":\"ok\"}", 15);
  } else if (!strcmp(r->path, "/info")) {
    h_info(fd);
  } else if (!strcmp(r->path, "/tensor")) {
    query_get(r->query, "name", name, sizeof name);
    const ie_tref *t = ie_tref_find(name);
    if (!t) reply_err(fd, 404, "no tensor %s", name);
    else if (!post) {
      const edit *e = edit_find(name);
      reply(fd, 200, "application/octet-stream", e ? (const char *)e->bytes : (const char *)t->src->data, t->src->nbytes);
    } else if (!edit_set(name, (const uint8_t *)r->body, r->body_n, err, sizeof err)) {
      reply_err(fd, 400, "%s", err);
    } else {
      sbuf b = {0};
      sb_printf(&b, "{\"name\":\"%s\",\"edited\":%s,\"edits\":%u}", name, edit_find(name) ? "true" : "false", N_ED);
      reply_json(fd, &b);
      sb_free(&b);
    }
  } else if (!strcmp(r->path, "/reset") && post) {
    query_get(r->query, "name", name, sizeof name);
    uint32_t n = 0;
    for (uint32_t i = N_ED; i-- > 0;) {
      if (name[0] && strcmp(ED[i].name, name)) continue;
      const ie_tref *t = ie_tref_find(ED[i].name);
      char nm[96];
      snprintf(nm, sizeof nm, "%s", ED[i].name);
      if (!edit_set(nm, t->src->data, t->src->nbytes, err, sizeof err)) ie_die("reset %s: %s", nm, err);
      n++;
    }
    sbuf b = {0};
    sb_printf(&b, "{\"reset\":%u,\"edits\":%u}", n, N_ED);
    reply_json(fd, &b);
    sb_free(&b);
  } else if (!strcmp(r->path, "/spec") && post) {
    h_spec(fd, body);
  } else if (!strcmp(r->path, "/capture") && post) {
    h_capture(fd, body);
  } else if (!strcmp(r->path, "/chain") && post) {
    h_chain(fd, body);
  } else if ((!strcmp(r->path, "/delta/save") || !strcmp(r->path, "/delta/load")) && post) {
    const jval *p = json_get(body, "path");
    uint32_t n = N_ED;
    if (!p || p->t != J_STR) reply_err(fd, 400, "give path");
    else if (r->path[7] == 's' ? !delta_save(p->str, err, sizeof err) : !delta_load(p->str, err, sizeof err, &n)) reply_err(fd, 400, "%s", err);
    else {
      sbuf b = {0};
      sb_printf(&b, "{\"tensors\":%u,\"edits\":%u}", n, N_ED);
      reply_json(fd, &b);
      sb_free(&b);
    }
  } else if (!strcmp(r->path, "/quit") && post) {
    reply(fd, 200, "application/json", "{\"bye\":true}", 12);
    close(fd);
    fprintf(stderr, "ie-lab: quit\n");
    exit(0);
  } else {
    reply_err(fd, 404, "no endpoint %s %s", r->method, r->path);
  }
  json_free(o);
}

static void usage(void) {
  fprintf(stderr, "usage: ie-lab MODEL.gguf [--mtp FILE] [--port 8090] [--ctx 8192] [--chunk 256] [--hsaco FILE]\n"
                  "       ie-lab --apply BASE.gguf DELTA.gguf OUT.gguf\n");
  exit(2);
}

int main(int argc, char **argv) {
  if (argc == 5 && !strcmp(argv[1], "--apply")) return apply_main(argv[2], argv[3], argv[4]);
  for (int i = 1; i < argc; i++) {
#define ARG (i + 1 < argc ? argv[++i] : (usage(), ""))
    if (!strcmp(argv[i], "--mtp")) O.mtp = ARG;
    else if (!strcmp(argv[i], "--port")) O.port = atoi(ARG);
    else if (!strcmp(argv[i], "--ctx")) O.ctx = (uint32_t)atoi(ARG);
    else if (!strcmp(argv[i], "--chunk")) O.chunk = (uint32_t)atoi(ARG);
    else if (!strcmp(argv[i], "--hsaco")) O.hsaco = ARG;
    else if (argv[i][0] != '-' && !O.path) O.path = argv[i];
    else usage();
#undef ARG
  }
  if (!O.path || O.chunk < 8 || O.chunk > 512 || O.ctx < 64) usage();
  ie_net_init();
  struct sockaddr_in sa = {0};
  sa.sin_family = AF_INET;
  sa.sin_port = htons((uint16_t)O.port);
  inet_pton(AF_INET, "127.0.0.1", &sa.sin_addr);
  const int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
  ie_setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  if (bind(ls, (struct sockaddr *)&sa, sizeof sa)) ie_die("bind 127.0.0.1:%d: %s", O.port, strerror(errno));

  const double t0 = now_ms();
  gguf_open(&G, O.path);
  gpu_stream_init();
  model_load(&M, &G);
  if (O.mtp) gguf_open(&G2, O.mtp);
  if (!model_load_mtp(&M, O.mtp ? &G2 : &G)) ie_die("no MTP head (blk.N.nextn.*) in %s", O.mtp ? O.mtp : O.path);
  model_make_draft_head(&M, &G);
  if (!gguf_get_int(&G, "tokenizer.ggml.eos_token_id", &EOS)) EOS = -1;
  ie_mtp_keep = 1;
  spec_set_draft(MTP_DRAFT_MAX); /* the verify room of any draft count a request may ask for */
  graph_build(&GR, &M, O.ctx, O.chunk > 9 ? O.chunk : 9, 1);
  graph_cap_logits(&GR);
  plan_arena(&GR, 1);
  B = gpu_open(&M, &GR, O.hsaco);
  gen_init(&GC, B, &M, &GR);
  if (listen(ls, 16)) ie_die("listen: %s", strerror(errno));
  fprintf(stderr, "ie-lab: %s%s%s ready in %.1f s: http://127.0.0.1:%d (%u tensors)\n", O.path, O.mtp ? " + " : "", O.mtp ? O.mtp : "",
          (now_ms() - t0) / 1e3, O.port, ie_n_trefs);
  for (;;) {
    const int fd = accept(ls, NULL, NULL);
    if (fd < 0) continue;
    ie_sock_timeout(fd, 120);
    request r;
    if (!read_request(fd, &r)) {
      const double h0 = now_ms();
      handle(fd, &r);
      fprintf(stderr, "ie-lab: %s %s%s%s %.1f ms\n", r.method, r.path, r.query[0] ? "?" : "", r.query, now_ms() - h0);
    }
    free(r.body);
    close(fd);
  }
}
