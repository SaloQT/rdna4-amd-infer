/* serve.c -- ie-serve: an OpenAI-compatible HTTP server for one model.
 *   ie-serve MODEL.gguf [--backend gpu|cpu] [--host 127.0.0.1] [--port 8000]
 *            [--api-key KEY] [--ctx 8192] [--draft D] ...   (--help: all options)
 * Endpoints:
 *   GET  /health                  {"status":"ok"}
 *   GET  /v1/models               the one model
 *   POST /v1/chat/completions     messages -> the Qwen chat template -> tokens;
 *                                 "stream": true gives server-sent events (SSE)
 * Threads: one engine thread owns the model and the backend (all HIP calls
 * are on it) and runs one request at a time from a queue; one short thread
 * per connection reads and checks the request, makes the prompt tokens and
 * puts the request in the queue. The engine thread writes the response.
 * A request whose client is gone (closed connection) stops at the next
 * check (every 8 tokens) or at the next failed write. */
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>

#include "backend.h"
#include "common.h"
#include "gen.h"
#include "gguf.h"
#include "chat.h"
#include "json.h"
#include "model.h"
#include "tok.h"
#include "vision.h"
#include "win_net.h" /* the sockets (POSIX, or Winsock on Windows); after the other headers */

int chat_render_inkling2(const jval *msgs, const jval *tools, const jval *effort_jv, const char *effort_str,
                         int allow_images, chat_prompt *p, char *err, size_t errn);
int chat_parse_inkling2_tool(const char *s, size_t n, sbuf *name_out, sbuf *args_out);

/* ---- options ---- */
typedef struct {
  const char *path, *backend, *hsaco, *host, *api_key, *mtp, *name, *effort, *dflash, *dspark, *lora, *mmproj;
  int port, think, queue_max, cache, raw_think;
  uint32_t ctx, draft, chunk, top_k, max_body, reasoning_budget;
  float pmin, temp, top_p, think_temp;
  float dry_mult, dry_base; /* the loop penalty (gen.h); dry_mult 0: off */
  uint32_t dry_allowed;
  float tool_temp; /* a request with tools that sets no temperature (< 0: off, the server default) */
} options;

static options O = {
    .effort = "xhigh", .backend = "gpu", .hsaco = "build/ie_kernels.hsaco", .host = "127.0.0.1", .port = 8000, .think = 1, .cache = 1, .queue_max = 8,
    .ctx = 8192, .draft = DRAFT_AUTO, .chunk = 0, .top_k = UINT32_MAX, .max_body = 32u << 20, .pmin = 0.6f, .temp = -1.0f, .top_p = -1.0f,
    .think_temp = -1.0f, .dry_mult = 0.0f, .dry_base = 1.75f, .dry_allowed = 24, .tool_temp = -1.0f,
}; /* temp/top_p/think_temp/tool_temp < 0, top_k UINT32_MAX: not given, see SAMPLING */

/* default sampling for a request that sets none, by thinking mode: [0] no thinking, [1] thinking */
typedef struct {
  float temp, top_p;
  uint32_t top_k;
} sampling;
static sampling SAMPLING[2];

static void usage(void) {
  fprintf(stderr,
          "usage: ie-serve MODEL.gguf [options]\n"
          "  --backend gpu|cpu   (default gpu; cpu: for tests with small models)\n"
          "  --hsaco FILE        GPU kernels (default build/ie_kernels.hsaco)\n"
          "  --host ADDR         listen address (default 127.0.0.1; a Tailscale address, or 0.0.0.0)\n"
          "  --port N            (default 8000)\n"
          "  --api-key KEY       require \"Authorization: Bearer KEY\" (or the environment variable IE_API_KEY)\n"
          "  --name ID           the model id in /v1/models (default: general.name of the file; spaces become -)\n"
          "  --ctx N             context: prompt + output tokens (default 8192)\n"
          "  --draft D           MTP drafts per step, 1 to 8 (GPU; default 3 when there is an MTP head:\n"
          "                      --mtp FILE or the model's own, else off; 0: off)\n"
          "  --draft-pmin P      (default 0.6)   --mtp FILE   the MTP head from FILE\n"
          "  --lora FILE         Qwen3.5/3.8 GGUF LoRA for the target's FFN and full-attention projections\n"
          "  --mmproj FILE       Qwen3.5/3.8 GGUF vision projector (enables image_url inputs)\n"
          "  --dflash FILE       with --draft: also a DFlash 2 drafter (GGUF); each step uses the faster drafter\n"
          "  --dspark FILE       with --draft: a DSpark drafter (GGUF, Inkling-2)\n"
          "  --chunk N           prompt chunk, 1 to 1024 (GPU; default 512; IE_CHUNK overrides)\n"
          "  --temp T --top-k K --top-p P   defaults when a request does not set them (thinking: the model file's\n"
          "                      general.sampling.*, else 0.6, 20, 0.95; no thinking: 0.7, 20, 0.8)\n"
          "  --think-temp T      the default temperature when thinking only (after --temp)\n"
          "  --dry-multiplier M  the loop penalty (default 0: off): a candidate that would continue a span of\n"
          "                      at least --dry-allowed-length N (24) tokens already in the answer loses\n"
          "                      M * --dry-base B (1.75) ^ (match - N) from its logit; sampling only; output\n"
          "                      only, new spans after </think>, paused inside <tool_call>...</tool_call>\n"
          "                      (IE_DRY_TOOL_PAUSE=0 disables). A request may set dry_multiplier, dry_base,\n"
          "                      dry_allowed_length\n"
          "  --tool-temp T       a non-thinking request with tools and no temperature uses T (default -1:\n"
          "                      off, the normal default for its thinking mode); the loop penalty needs\n"
          "                      temperature > 0, so T 0 (greedy) has none\n"
          "  --no-think          end the prompt with an empty <think></think> block (no reasoning text)\n"
          "  --effort E          reasoning effort when thinking: xhigh (default), medium, low\n"
          "  --reasoning-budget N   cap an open Qwen reasoning block at N tokens (0: unlimited)\n"
          "  --no-cache          compute every prompt from the start (no prompt cache)\n"
          "  --think-in-content  send the reasoning in content as <think>...</think> (default: in reasoning_content)\n"
          "  --queue N           the most waiting requests (default 8; more: HTTP 503)\n"
          "  --show-template     print the chat template of the model file and exit\n");
  exit(2);
}

/* ---- the engine (only the engine thread uses these after the start) ---- */
static gguf_file G, G2, G3; /* G3: the DFlash drafter (open while serving: the selector reads its codebooks) */
static model M;
static graph GR;
static backend *B;
static gen_ctx GC;
static tokenizer *TK;          /* read-only after the start: the connection threads use it too */
static int64_t STOP_IDS[8];    /* eos, <|im_end|>, <|endoftext|> (flint: tokenizer.ggml.eog_token_ids); -1: none */
static int64_t IM_START, IM_END;
/* flint: its own chat template (chat_render_flint), the prompt tokenized with
 * the control tokens; the model decides whether it reasons: an answer that
 * starts with <think> (THINK_ID) is reasoning up to </think> */
static int FLINT;
/* MiMo v2.6 (its own template on qwen35, picked from tokenizer.chat_template): chat_render_mimo;
 * with thinking the model opens <think> itself (as flint), without it the prompt ends <think></think> */
static int MIMO;
/* Inkling-2 (ARCH_INKLING2): chat_render_inkling2; model emits <|message_model|> + <|content_*|> blocks
 * ending with <|end_message|>, turn terminated by <|content_model_end_sampling|> (200006). */
static int INKLING2;
static int EFFORT_SET;
static int64_t THINK_ID = -1;
static int64_t THINK_END_ID = -1; /* </think>: the loop penalty starts new spans after it */
static int64_t TOOL_CALL_OPEN_ID = -1, TOOL_CALL_CLOSE_ID = -1; /* <tool_call>..</tool_call>: pause DRY inside tool calls */
static int64_t VISION_START_ID = -1, VISION_END_ID = -1, IMAGE_PAD_ID = -1;
static vision_ctx *VIS;
static char MODEL_ID[256];
static time_t T_START;

/* ---- the queue ---- */
typedef struct job {
  int fd;
  unsigned long rid;
  int stream, usage;
  tok_ids prompt;
  gen_params gp;
  chat_img *img_urls; /* image URLs pointing into req */
  uint32_t *img_slot; /* placeholder token index of each image in prompt before expansion */
  size_t n_img_urls;
  uint32_t req_max_tok; /* explicit max_tokens / max_completion_tokens from request (0: unlimited) */
  char id[48];
  long created;
  jval *req;           /* the request (the tools' schemas are used for the answer) */
  const jval *tools;   /* its "tools" array, or NULL */
  struct job *next;
  /* while running */
  sbuf all;            /* the whole answer text (with "<think>\n" first when thinking) */
  size_t sent;         /* streaming: bytes of all sent; else the end of the content */
  size_t scan;         /* all is searched for </think> and <tool_call> from about here */
  long think_end;      /* the end of "</think>" in all (0 without thinking; -1: not yet) */
  long tc_at;          /* the start of the first "<tool_call>" (-1: none) */
  uint32_t n_tok;      /* output tokens, without the stop token */
  int finish;          /* 0: length, 1: stop, 2: client gone */
  int think;           /* the answer text starts with the opening of a reasoning block (rp bytes) */
  int auto_think;      /* flint: think is set by the first token (<think>) */
  size_t rp;           /* the bytes of that opening: 8 ("<think>\n", Qwen: the prompt's), 7 ("<think>", flint) */
  int content_started; /* split reasoning: a content character after </think> was sent */
  volatile int dead;   /* a write failed (the engine thread's or, paced, the sender thread's) */
  double pace_last;    /* paced streaming: when the last step's tokens came (0: none yet) */
  /* Inkling-2 response block extraction (shared by streaming and non-streaming) */
  int ink_st;          /* 0: header/between blocks, 1: thinking, 2: text, 3: tool_json, 4: plain fallback */
  int ink_seen_tag;    /* 1 once any Inkling control/content block tag has been matched */
  size_t ink_pos;      /* current scan/emit offset in all */
  size_t ink_body;     /* start of current tool_json body */
  int ink_n_calls;     /* number of Inkling-2 tool calls extracted */
  size_t ink_arg_pos, ink_arg_end;
  int ink_arg_depth, ink_arg_string, ink_arg_escape, ink_stream_invalid;
  sbuf ink_reason, ink_content, ink_calls;
  /* streaming tool calls (tcs_*, stream with tools, not flint): the calls go out as they are written, as OpenAI
   * tool_calls deltas (id and name, then pieces of the arguments); at the end chat_tool_calls's arguments decide the
   * rest, so the joined pieces equal the arguments a client got before in one piece */
  char tcs_idp[64];    /* the id prefix (call k: tcs_idp k) */
  int tcs_st;          /* 0: before <tool_call>, 1: before <function=..>, 2: in a function, 3: a string value, 4: another value */
  int tcs_k;           /* calls started */
  int tcs_np;          /* parameters of the current call */
  int tcs_off;         /* streaming stopped (too many calls) */
  size_t tcs_pos;      /* parse position in all */
  size_t tcs_name, tcs_nn; /* the current call's name in all */
  size_t tcs_v;        /* the current value's first byte in all */
  size_t tcs_vs;       /* its text start (after a first newline; 0: not known yet) */
  size_t tcs_ve;       /* string value: sent up to here */
  sbuf tcs_sent[16];   /* the arguments text sent per call */
  char sse_pfx[384];   /* pre-formatted constant SSE chunk prefix */
  size_t sse_pfx_n;
} job;

static pthread_mutex_t QM = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t QC = PTHREAD_COND_INITIALIZER, READY = PTHREAD_COND_INITIALIZER;
static job *QH, *QT;
static int QN, BUSY, IS_READY;
static unsigned long NEXT_RID;

/* ---- socket helpers ---- */
static int send_all(int fd, const char *p, size_t n) {
  while (n) {
    ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return -1;
    p += k, n -= (size_t)k;
  }
  return 0;
}
static const char *status_text(int code) {
  switch (code) {
  case 200: return "OK";
  case 204: return "No Content";
  case 400: return "Bad Request";
  case 401: return "Unauthorized";
  case 404: return "Not Found";
  case 405: return "Method Not Allowed";
  case 411: return "Length Required";
  case 413: return "Payload Too Large";
  case 503: return "Service Unavailable";
  default: return "Error";
  }
}
static void http_reply(int fd, int code, const char *ctype, const char *body, size_t n) {
  sbuf h = {0};
  sb_printf(&h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", code, status_text(code),
            ctype, n);
  send_all(fd, h.p, h.n);
  if (n) send_all(fd, body, n);
  sb_free(&h);
}
static void http_error(int fd, int code, const char *type, const char *fmt, ...) __attribute__((format(printf, 4, 5)));
static void http_error(int fd, int code, const char *type, const char *fmt, ...) {
  char msg[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof msg, fmt, ap);
  va_end(ap);
  sbuf b = {0};
  sb_puts(&b, "{\"error\":{\"message\":");
  sb_json_str(&b, msg, strlen(msg));
  sb_printf(&b, ",\"type\":\"%s\",\"code\":%d}}", type, code);
  http_reply(fd, code, "application/json", b.p, b.n);
  sb_free(&b);
}
/* 1 if the client closed its side (a check that does not wait) */
static int peer_gone(int fd) {
  struct pollfd p = {fd, POLLIN | POLLRDHUP, 0};
  if (poll(&p, 1, 0) <= 0) return 0;
  if (p.revents & (POLLRDHUP | POLLHUP | POLLERR)) return 1;
  char c;
  return recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT) == 0;
}

/* ---- the response, written by the engine thread ---- */
static int is_stop(uint32_t t) {
  for (int i = 0; i < 8; i++)
    if (STOP_IDS[i] >= 0 && (uint32_t)STOP_IDS[i] == t) return 1;
  return 0;
}
/* every chunk also carries n_tokens, the output tokens so far (other clients ignore it; the lab's Open WebUI speed
 * filter turns it into tokens per second while the answer streams) */
static void sse_begin(sbuf *b, job *j) {
  if (!j->sse_pfx_n) {
    sbuf tmp = {0};
    sb_printf(&tmp, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", j->id, j->created);
    sb_json_str(&tmp, MODEL_ID, strlen(MODEL_ID));
    sb_puts(&tmp, ",\"n_tokens\":");
    if (tmp.n <= sizeof j->sse_pfx) {
      memcpy(j->sse_pfx, tmp.p, tmp.n);
      j->sse_pfx_n = tmp.n;
    }
    sb_free(&tmp);
  }
  if (j->sse_pfx_n) {
    sb_add(b, j->sse_pfx, j->sse_pfx_n);
    char nbuf[16];
    const int nn = snprintf(nbuf, sizeof nbuf, "%u", j->n_tok);
    if (nn > 0) sb_add(b, nbuf, (size_t)nn);
  } else {
    sb_printf(b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", j->id, j->created);
    sb_json_str(b, MODEL_ID, strlen(MODEL_ID));
    sb_printf(b, ",\"n_tokens\":%u", j->n_tok);
  }
}
/* ---- paced streaming (IE_STREAM_PACE, default 1): each output token is its own chunk, and a sender thread spreads
 * the tokens of one decoding step (MTP: several come at once) over the time a step takes, so the text arrives one
 * token at a time at the decoding rate instead of in bursts. A token waits at most about one step (tens of ms): when
 * the next step's tokens come, what is left of the previous step goes at once. The engine thread only queues the
 * chunks; at the end of a request it waits until they are sent. IE_STREAM_PACE=0: one chunk per step, written by the
 * engine thread as soon as the step ends. ---- */
typedef struct pitem {
  char *p;
  size_t n;
  int fd;
  volatile int *dead;
  double due;          /* monotonic seconds; 0: at once */
  unsigned long batch; /* the step it belongs to */
  struct pitem *next;
} pitem;
static int PACE = 1;
static pthread_mutex_t PM = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t PC = PTHREAD_COND_INITIALIZER, PDONE = PTHREAD_COND_INITIALIZER;
static pitem *PH, *PT;
static int PBUSY;               /* the sender is writing an item */
static unsigned long PBATCH;    /* the newest step (written by the engine thread under PM) */
static double PACE_DUE;         /* engine thread: when the chunk being made goes out (0: at once) */
static double PACE_STEP = 0.03; /* engine thread: the time between steps, a moving average (s) */

static double mono_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
static void *pace_main(void *arg) {
  (void)arg;
  pthread_mutex_lock(&PM);
  for (;;) {
    while (!PH) pthread_cond_wait(&PC, &PM);
    pitem *it = PH;
    for (;;) { /* until it is due, or the next step's tokens are here (then what is left goes at once) */
      const double w = it->due - mono_s();
      if (w <= 0 || PBATCH > it->batch || *it->dead) break;
      struct timespec dl;
      timespec_get(&dl, TIME_UTC);
      const long long ns = (long long)dl.tv_nsec + (long long)(w * 1e9);
      dl.tv_sec += (time_t)(ns / 1000000000LL), dl.tv_nsec = (long)(ns % 1000000000LL);
      pthread_cond_timedwait(&PC, &PM, &dl);
    }
    PH = it->next;
    if (!PH) PT = NULL;
    PBUSY = 1;
    pthread_mutex_unlock(&PM);
    if (!*it->dead && send_all(it->fd, it->p, it->n)) *it->dead = 1;
    free(it->p), free(it);
    pthread_mutex_lock(&PM);
    PBUSY = 0;
    if (!PH) pthread_cond_broadcast(&PDONE);
  }
  return NULL;
}
/* engine thread: a new step's tokens (wakes the sender, which sends what is left of the last step at once) */
static void pace_step(void) {
  pthread_mutex_lock(&PM);
  PBATCH++;
  pthread_cond_signal(&PC);
  pthread_mutex_unlock(&PM);
}
/* engine thread: wait until every queued chunk is sent (or dropped: the client is gone) */
static void pace_drain(void) {
  pthread_mutex_lock(&PM);
  while (PH || PBUSY) pthread_cond_wait(&PDONE, &PM);
  pthread_mutex_unlock(&PM);
}

static int pace_first_fast(void) {
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("IE_PACE_FIRST_FAST");
    v = (e && *e) ? (atoi(e) != 0) : 1; /* R51 (H48): promoted default; IE_PACE_FIRST_FAST=0 opts out */
  }
  return v;
}
static void sse_send(job *j, sbuf *b) {
  if (PACE) { /* the sender thread writes it (takes the buffer) */
    if (pace_first_fast() && PACE_DUE == 0.0 && j->n_tok <= 1u) {
      pthread_mutex_lock(&PM);
      const int empty = (!PH && !PBUSY);
      pthread_mutex_unlock(&PM);
      if (empty) {
        if (send_all(j->fd, b->p, b->n)) j->dead = 1;
        sb_free(b);
        return;
      }
    }
    pitem *it = malloc(sizeof *it);
    if (!it) ie_die("ie-serve: out of memory");
    *it = (pitem){b->p, b->n, j->fd, &j->dead, PACE_DUE, PBATCH, NULL};
    b->p = NULL, b->n = b->cap = 0;
    pthread_mutex_lock(&PM);
    if (PT) PT->next = it;
    else PH = it;
    PT = it;
    pthread_cond_signal(&PC);
    pthread_mutex_unlock(&PM);
    return;
  }
  if (send_all(j->fd, b->p, b->n)) j->dead = 1;
  sb_free(b);
}
/* text (NULL: the role chunk) or finish */
static void sse_chunk(job *j, const char *text, size_t n, const char *finish) {
  sbuf b = {0};
  sse_begin(&b, j);
  sb_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{");
  if (!text) sb_puts(&b, "\"role\":\"assistant\",\"content\":\"\"");
  else if (n) sb_puts(&b, "\"content\":"), sb_json_str(&b, text, n);
  sb_puts(&b, "},\"finish_reason\":");
  if (finish) sb_printf(&b, "\"%s\"", finish);
  else sb_puts(&b, "null");
  sb_puts(&b, "}]}\n\n");
  sse_send(j, &b);
}
/* one delta with text in field ("content" or "reasoning_content") */
static void sse_field(job *j, const char *field, const char *text, size_t n) {
  if (!n) return;
  sbuf b = {0};
  sse_begin(&b, j);
  sb_printf(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"%s\":", field);
  sb_json_str(&b, text, n);
  sb_puts(&b, "},\"finish_reason\":null}]}\n\n");
  sse_send(j, &b);
}
/* the tool calls (a JSON array from chat_tool_calls) as one delta, each with its index */
static void sse_tool_calls(job *j, const char *arr, size_t n) {
  char err[64];
  jval *a = json_parse(arr, n, err, sizeof err);
  sbuf b = {0};
  sse_begin(&b, j);
  sb_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[");
  int k = 0;
  for (const jval *c = a ? a->kid : NULL; c; c = c->next, k++) {
    sbuf one = {0};
    py_json(&one, c);
    sb_printf(&b, "%s{\"index\":%d, ", k ? "," : "", k);
    sb_add(&b, one.p + 1, one.n - 1); /* the object without its "{" */
    sb_free(&one);
  }
  sb_puts(&b, "]},\"finish_reason\":null}]}\n\n");
  sse_send(j, &b);
  json_free(a);
}

static const char *find_in(const char *s, size_t from, size_t n, const char *pat) {
  const size_t k = strlen(pat);
  for (size_t i = from; i + k <= n; i++)
    if (!memcmp(s + i, pat, k)) return s + i;
  return NULL;
}
/* ---- streaming tool calls (the <tool_call><function=NAME><parameter=K>V</parameter>..</function></tool_call>
 * form that chat_tool_calls reads, parsed as it is written) ---- */
#define TCS_MAX 16
/* the head of call k: its id, type and name, empty arguments */
static void tcs_head(job *j, int k, const char *name, size_t nn) {
  sbuf b = {0};
  sse_begin(&b, j);
  sb_printf(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":%d,\"id\":\"%s%d\",\"type\":\"function\",\"function\":{\"name\":",
            k, j->tcs_idp, k);
  sb_json_str(&b, name, nn);
  sb_puts(&b, ",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n");
  sse_send(j, &b);
}
/* a piece of the arguments text of call k */
static void tcs_piece(job *j, int k, const char *t, size_t n) {
  if (!n) return;
  if (k < TCS_MAX) sb_add(&j->tcs_sent[k], t, n);
  sbuf b = {0};
  sse_begin(&b, j);
  sb_printf(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":%d,\"function\":{\"arguments\":", k);
  sb_json_str(&b, t, n);
  sb_puts(&b, "}}]},\"finish_reason\":null}]}\n\n");
  sse_send(j, &b);
}
/* the escaped text of s[0 .. n) (sb_json_str without its quotes) as a piece of call k */
static void tcs_text(job *j, int k, const char *s, size_t n) {
  if (!n) return;
  sbuf e = {0};
  sb_json_str(&e, s, n);
  tcs_piece(j, k, e.p + 1, e.n - 2);
  sb_free(&e);
}
/* the bytes at the end of s[a .. n) that could start pat */
static size_t tcs_partial(const char *s, size_t a, size_t n, const char *pat) {
  for (size_t k = strlen(pat) - 1; k >= 1; k--)
    if (n >= a + k && !memcmp(s + n - k, pat, k)) return k;
  return 0;
}
/* the value of the current parameter, s[vs .. ve) after chat_tool_calls's trim (a first and a last newline) */
static void tcs_value_end(job *j, const char *s, size_t ve) {
  const int k = j->tcs_k - 1;
  size_t vs = j->tcs_v, vn = ve - vs;
  if (vn && s[vs] == '\n') vs++, vn--;
  if (vn && s[vs + vn - 1] == '\n') vn--;
  if (j->tcs_st == 3) { /* the rest of the string, then its closing quote */
    const size_t from = j->tcs_ve > vs ? j->tcs_ve : vs;
    if (vs + vn > from) tcs_text(j, k, s + from, vs + vn - from);
    tcs_piece(j, k, "\"", 1);
  } else { /* a typed value: JSON if it parses, else a string (as chat_tool_calls) */
    char e2[64];
    jval *jv = json_parse(s + vs, vn, e2, sizeof e2);
    sbuf b = {0};
    if (jv) py_json(&b, jv);
    else sb_json_str(&b, s + vs, vn);
    json_free(jv);
    tcs_piece(j, k, b.p, b.n);
    sb_free(&b);
  }
}
static void tcs_run(job *j) {
  const char *s = j->all.p;
  const size_t n = j->all.n;
  while (!j->tcs_off && !j->dead) {
    const size_t p = j->tcs_pos;
    if (j->tcs_st == 0) { /* the next <tool_call> */
      const char *c = find_in(s, p, n, "<tool_call>");
      if (!c) { j->tcs_pos = n > p + 10 ? n - 10 : p; return; }
      j->tcs_st = 1, j->tcs_pos = (size_t)(c - s);
    } else if (j->tcs_st == 1) { /* <function=NAME> */
      const char *f = find_in(s, p, n, "<function="), *fe = f ? find_in(s, (size_t)(f - s) + 10, n, ">") : NULL;
      if (!fe) return;
      if (j->tcs_k >= TCS_MAX) { j->tcs_off = 1; return; }
      j->tcs_name = (size_t)(f - s) + 10, j->tcs_nn = (size_t)(fe - f) - 10;
      tcs_head(j, j->tcs_k, s + j->tcs_name, j->tcs_nn);
      tcs_piece(j, j->tcs_k, "{", 1);
      j->tcs_k++, j->tcs_np = 0, j->tcs_st = 2, j->tcs_pos = (size_t)(fe - s) + 1;
    } else if (j->tcs_st == 2) { /* <parameter=K> or </function> */
      const char *pp = find_in(s, p, n, "<parameter="), *fn = find_in(s, p, n, "</function>");
      if (fn && (!pp || fn < pp)) { j->tcs_st = 0, j->tcs_pos = (size_t)(fn - s) + 11; continue; }
      if (!pp) { j->tcs_pos = n > p + 11 ? n - 11 : p; return; }
      const char *ke = find_in(s, (size_t)(pp - s) + 11, n, ">");
      if (!ke) { j->tcs_pos = (size_t)(pp - s); return; }
      const char *key = pp + 11;
      const size_t kn = (size_t)(ke - key);
      const int k = j->tcs_k - 1;
      if (j->tcs_np++) tcs_piece(j, k, ", ", 2);
      sbuf kb = {0};
      sb_json_str(&kb, key, kn);
      sb_puts(&kb, ": ");
      tcs_piece(j, k, kb.p, kb.n);
      sb_free(&kb);
      const char *ty = chat_param_type(j->tools, s + j->tcs_name, j->tcs_nn, key, kn);
      const int str = !strcmp(ty, "string") || !*ty;
      if (str) tcs_piece(j, k, "\"", 1);
      j->tcs_st = str ? 3 : 4, j->tcs_v = (size_t)(ke - s) + 1, j->tcs_vs = 0, j->tcs_ve = 0, j->tcs_pos = j->tcs_v;
    } else { /* in a value: up to </parameter> or </function> */
      const char *pe = find_in(s, p, n, "</parameter>"), *fn = find_in(s, p, n, "</function>");
      const char *ve = pe && (!fn || pe < fn) ? pe : fn;
      if (ve) {
        tcs_value_end(j, s, (size_t)(ve - s));
        if (ve == pe) j->tcs_st = 2, j->tcs_pos = (size_t)(pe - s) + 12;
        else j->tcs_st = 0, j->tcs_pos = (size_t)(fn - s) + 11;
        continue;
      }
      const size_t hold = tcs_partial(s, j->tcs_v, n, "</parameter>") > tcs_partial(s, j->tcs_v, n, "</function>")
                              ? tcs_partial(s, j->tcs_v, n, "</parameter>") : tcs_partial(s, j->tcs_v, n, "</function>");
      j->tcs_pos = n - hold > j->tcs_v ? n - hold : j->tcs_v;
      if (j->tcs_st == 3) { /* send the string's text that cannot change: not a last newline, nor a cut character */
        if (!j->tcs_vs) {
          if (n <= j->tcs_v) return;
          j->tcs_vs = j->tcs_v + (s[j->tcs_v] == '\n');
          j->tcs_ve = j->tcs_vs;
        }
        size_t e = n - hold;
        if (e > j->tcs_ve && s[e - 1] == '\n') e--;
        if (e > j->tcs_ve) e = j->tcs_ve + utf8_whole(s + j->tcs_ve, e - j->tcs_ve);
        if (e > j->tcs_ve) tcs_text(j, j->tcs_k - 1, s + j->tcs_ve, e - j->tcs_ve), j->tcs_ve = e;
      }
      return;
    }
  }
}
/* the end: chat_tool_calls's calls (n_calls in calls) after what went out; 0 if some call does not continue what
 * was sent (logged) */
static int tcs_finish(job *j, const char *arr, size_t an) {
  char err[64];
  jval *a = json_parse(arr, an, err, sizeof err);
  int k = 0, ok = 1;
  for (const jval *c = a ? a->kid : NULL; c; c = c->next, k++) {
    const jval *fn = json_get(c, "function"), *nm = json_get(fn, "name"), *ar = json_get(fn, "arguments");
    if (!nm || nm->t != J_STR || !ar || ar->t != J_STR) { ok = 0; continue; }
    if (k < j->tcs_k && k < TCS_MAX) {
      const sbuf *t = &j->tcs_sent[k];
      if (ar->slen < t->n || memcmp(ar->str, t->p, t->n)) {
        fprintf(stderr, "[%lu] tool call %d: the streamed arguments do not begin the parsed ones\n", j->rid, k);
        ok = 0;
        continue;
      }
      tcs_piece(j, k, ar->str + t->n, ar->slen - t->n);
    } else {
      tcs_head(j, k, nm->str, nm->slen);
      tcs_piece(j, k, ar->str, ar->slen);
    }
  }
  json_free(a);
  return ok;
}

/* How much of all can go to the client now. The text from the first
 * <tool_call> after the reasoning is held back (with tools it becomes
 * tool_calls), and so is an end of the text that could be the start of
 * "</think>" (while thinking) or "<tool_call>". */
static size_t safe_end(job *j, int final) {
  const char *s = j->all.p;
  const size_t n = j->all.n;
  if (!s) return 0;
  const size_t from = j->scan > 16 ? j->scan - 16 : 0;
  if (j->think_end < 0) {
    const char *e = find_in(s, from, n, "</think>");
    if (e) j->think_end = (long)(e - s) + 8;
  }
  if (j->tools && j->tc_at < 0 && j->think_end >= 0) {
    const size_t f2 = from > (size_t)j->think_end ? from : (size_t)j->think_end;
    const char *c = find_in(s, f2, n, "<tool_call>");
    if (c) j->tc_at = (long)(c - s);
  }
  j->scan = n;
  if (j->tc_at >= 0) return (size_t)j->tc_at;
  if (final) return n;
  size_t lim = n;
  const char *pat = j->think_end < 0 ? "</think>" : j->tools ? "<tool_call>" : NULL;
  const size_t base = j->think_end < 0 ? 0 : (size_t)j->think_end;
  if (pat)
    for (size_t k = strlen(pat) - 1; k >= 1; k--)
      if (n >= base + k && !memcmp(s + n - k, pat, k)) { lim = n - k; break; }
  return utf8_whole(s, lim);
}
static void ink_out_text(job *j, int is_reason, const char *s, size_t n) {
  if (!n) return;
  if (O.raw_think) {
    if (j->stream) sse_field(j, "content", s, n);
    else sb_add(&j->ink_content, s, n);
    return;
  }
  if (j->stream) sse_field(j, is_reason ? "reasoning_content" : "content", s, n);
  else sb_add(is_reason ? &j->ink_reason : &j->ink_content, s, n);
}

static void ink_emit_tool_sse(job *j, int k, const char *name, size_t nn, const char *args, size_t an) {
  sbuf b = {0};
  sse_begin(&b, j);
  sb_printf(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":%d,\"id\":\"%s%d\",\"type\":\"function\",\"function\":{\"name\":",
            k, j->tcs_idp, k);
  sb_json_str(&b, name, nn);
  sb_puts(&b, ",\"arguments\":");
  sb_json_str(&b, args, an);
  sb_puts(&b, "}}]},\"finish_reason\":null}]}\n\n");
  sse_send(j, &b);
}

static const char *ink_skip_space(const char *p, const char *end) {
  while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) p++;
  return p;
}

static const char *ink_after(const char *p, const char *end, const char *literal) {
  p = ink_skip_space(p, end);
  const size_t n = strlen(literal);
  return (size_t)(end - p) >= n && !memcmp(p, literal, n) ? p + n : NULL;
}

static void ink_stream_file_args(job *j, const char *s, size_t limit) {
  if (!j->stream || !j->tools || j->ink_arg_end || j->ink_stream_invalid) return;
  const int k = j->ink_n_calls;
  if (!j->ink_arg_pos) {
    const char *end = s + limit;
    const char *p = ink_after(s + j->ink_body, end, "{");
    if (!p || !(p = ink_after(p, end, "\"name\"")) || !(p = ink_after(p, end, ":"))) return;
    const char *name = ink_skip_space(p, end);
    const char *tail = ink_after(name, end, "\"write\"");
    if (!tail) tail = ink_after(name, end, "\"edit\"");
    if (!tail) tail = ink_after(name, end, "\"str_replace_editor\"");
    if (!tail || !(p = ink_after(tail, end, ","))) return;
    const char *key = ink_after(p, end, "\"args\"");
    if (!key) key = ink_after(p, end, "\"arguments\"");
    if (!key || !(p = ink_after(key, end, ":")) || !(p = ink_after(p, end, "{"))) return;
    j->ink_arg_pos = (size_t)(p - s) - 1;
    tcs_head(j, k, name + 1, (size_t)(tail - name) - 2);
  }
  const size_t from = j->ink_arg_pos;
  limit = from + utf8_whole(s + from, limit - from);
  size_t pos = from;
  for (; pos < limit; pos++) {
    const char c = s[pos];
    if (j->ink_arg_string) {
      if (j->ink_arg_escape) j->ink_arg_escape = 0;
      else if (c == '\\') j->ink_arg_escape = 1;
      else if (c == '"') j->ink_arg_string = 0;
    } else if (c == '"') {
      j->ink_arg_string = 1;
    } else if (c == '{') {
      j->ink_arg_depth++;
    } else if (c == '}' && --j->ink_arg_depth == 0) {
      j->ink_arg_end = pos;
      break;
    }
  }
  tcs_piece(j, k, s + from, pos - from);
  j->ink_arg_pos = pos;
}

static int ink_prefix_of_hdr(const char *s, size_t n) {
  static const char *const h[] = {"<|message_model|>", "<|content_"};
  for (int i = 0; i < 2; i++) {
    const size_t hl = strlen(h[i]), m = n < hl ? n : hl;
    if (!memcmp(s, h[i], m)) return 1;
  }
  return 0;
}

static int ink_tag_eq(const char *tag, size_t tlen, const char *lit) {
  const size_t l = strlen(lit);
  return tlen == l && !memcmp(tag, lit, l);
}

static void ink_flush(job *j, int final) {
  const char *s = j->all.p;
  const size_t n = j->all.n;
  if (!s) return;
  while (j->ink_pos < n && !j->dead) {
    if (j->ink_st == 0) {
      if (!j->ink_seen_tag && j->ink_pos == 0 && !ink_prefix_of_hdr(s, n)) {
        j->ink_st = 4; /* plain-text fallback when model doesn't emit Inkling tags */
        continue;
      }
      size_t p = j->ink_pos;
      int matched = 0;
      while (p < n) {
        const char *tag = find_in(s, p, n, "<|");
        if (!tag) break;
        const size_t toff = (size_t)(tag - s);
        const char *gt = find_in(s, toff + 2, n, "|>");
        if (!gt) {
          if (!final) return; /* partial control tag */
          break;
        }
        const size_t tend = (size_t)(gt - s) + 2;
        const size_t tlen = tend - toff;
        if (ink_tag_eq(tag, tlen, "<|message_model|>")) {
          p = tend;
          continue;
        }
        if (ink_tag_eq(tag, tlen, "<|content_thinking|>")) {
          j->ink_seen_tag = 1, j->ink_st = 1, j->ink_pos = tend;
          if (O.raw_think) ink_out_text(j, 0, "<think>\n", 8);
          matched = 1;
          break;
        }
        if (ink_tag_eq(tag, tlen, "<|content_text|>") || ink_tag_eq(tag, tlen, "<|content_tool_error|>") ||
            ink_tag_eq(tag, tlen, "<|content_invoke_tool_text|>")) {
          j->ink_seen_tag = 1, j->ink_st = 2, j->ink_pos = tend;
          matched = 1;
          break;
        }
        if (ink_tag_eq(tag, tlen, "<|content_invoke_tool_json|>")) {
          j->ink_seen_tag = 1, j->ink_st = 3, j->ink_pos = tend, j->ink_body = tend;
          j->ink_arg_pos = j->ink_arg_end = 0;
          j->ink_arg_depth = j->ink_arg_string = j->ink_arg_escape = 0;
          matched = 1;
          break;
        }
        if (ink_tag_eq(tag, tlen, "<|end_message|>") || ink_tag_eq(tag, tlen, "<|content_model_end_sampling|>")) {
          j->ink_seen_tag = 1, j->ink_pos = tend;
          p = tend;
          continue;
        }
        p = tend;
      }
      if (!matched) {
        if (final) j->ink_pos = n;
        return;
      }
    } else if (j->ink_st == 1 || j->ink_st == 2) {
      const int is_r = j->ink_st == 1;
      const char *end = find_in(s, j->ink_pos, n, "<|end_message|>");
      if (end) {
        const size_t eoff = (size_t)(end - s);
        if (eoff > j->ink_pos) ink_out_text(j, is_r, s + j->ink_pos, eoff - j->ink_pos);
        if (is_r && O.raw_think) ink_out_text(j, 0, "\n</think>\n\n", 11);
        j->ink_pos = eoff + 15, j->ink_st = 0;
        continue;
      }
      size_t lim = n;
      if (!final) {
        const size_t hold = tcs_partial(s, j->ink_pos, n, "<|end_message|>");
        if (hold) lim = n - hold;
        lim = j->ink_pos + utf8_whole(s + j->ink_pos, lim - j->ink_pos);
      }
      if (lim > j->ink_pos) {
        ink_out_text(j, is_r, s + j->ink_pos, lim - j->ink_pos);
        j->ink_pos = lim;
      }
      if (final && is_r && O.raw_think) ink_out_text(j, 0, "\n</think>\n\n", 11), j->ink_st = 0;
      return;
    } else if (j->ink_st == 3) {
      const char *end = find_in(s, j->ink_pos, n, "<|end_message|>");
      const size_t eoff = end ? (size_t)(end - s) : n;
      const size_t limit = end ? eoff : n - tcs_partial(s, j->ink_pos, n, "<|end_message|>");
      ink_stream_file_args(j, s, limit);
      if (!end) {
        j->ink_pos = limit;
        if (final && n > j->ink_body) {
          if (j->ink_arg_pos) j->ink_stream_invalid = 1;
          else ink_out_text(j, 0, s + j->ink_body, n - j->ink_body);
          j->ink_pos = n;
        }
        return;
      }
      const size_t blen = eoff - j->ink_body;
      sbuf nm = {0}, args = {0};
      const int valid = j->tools && chat_parse_inkling2_tool(s + j->ink_body, blen, &nm, &args);
      int match = valid && j->ink_arg_end;
      if (match && !ink_after(s + j->ink_arg_end + 1, end, "}")) {
        sbuf sent = {0}, sent_name = {0}, sent_args = {0};
        sb_add(&sent, s + j->ink_body, j->ink_arg_end + 1 - j->ink_body);
        sb_puts(&sent, "}");
        match = chat_parse_inkling2_tool(sent.p, sent.n, &sent_name, &sent_args) &&
                sent_name.n == nm.n && !memcmp(sent_name.p, nm.p, nm.n) &&
                sent_args.n == args.n && !memcmp(sent_args.p, args.p, args.n);
        sb_free(&sent), sb_free(&sent_name), sb_free(&sent_args);
      }
      if (j->ink_arg_pos && !match) {
        j->ink_stream_invalid = 1;
      } else if (valid) {
        const int k = j->ink_n_calls++;
        if (k == 0) sb_add(&j->ink_calls, "[", 1);
        else sb_add(&j->ink_calls, ",", 1);
        sb_printf(&j->ink_calls, "{\"id\":\"%s%d\",\"type\":\"function\",\"function\":{\"name\":", j->tcs_idp, k);
        sb_json_str(&j->ink_calls, nm.p, nm.n);
        sb_puts(&j->ink_calls, ",\"arguments\":");
        sb_json_str(&j->ink_calls, args.p, args.n);
        sb_puts(&j->ink_calls, "}}");
        if (j->ink_arg_end) tcs_piece(j, k, "}", 1);
        else if (j->stream) ink_emit_tool_sse(j, k, nm.p, nm.n, args.p, args.n);
      } else if (blen) {
        ink_out_text(j, 0, s + j->ink_body, blen);
      }
      sb_free(&nm), sb_free(&args);
      j->ink_pos = eoff + 15, j->ink_st = 0;
    } else { /* ink_st == 4: plain fallback */
      const size_t lim = final ? n : j->ink_pos + utf8_whole(s + j->ink_pos, n - j->ink_pos);
      if (lim > j->ink_pos) {
        ink_out_text(j, 0, s + j->ink_pos, lim - j->ink_pos);
        j->ink_pos = lim;
      }
      return;
    }
  }
  if (final && j->ink_st == 3 && j->ink_arg_pos) j->ink_stream_invalid = 1;
  if (final && j->ink_st == 1 && O.raw_think) ink_out_text(j, 0, "\n</think>\n\n", 11), j->ink_st = 0;
}

/* split reasoning: all = "<think>\n" reasoning "</think>" white space content */
static int split_think(const job *j) { return j->think && !O.raw_think; }
static size_t reason_end(const job *j, size_t e) { return j->think_end >= 0 && (size_t)j->think_end - 8 < e ? (size_t)j->think_end - 8 : e; }
static void flush_text(job *j, int final) {
  if (INKLING2) {
    if (j->stream || final) ink_flush(j, final);
    return;
  }
  const size_t e = safe_end(j, final);
  if (e <= j->sent) return;
  if (j->stream && !split_think(j)) sse_chunk(j, j->all.p + j->sent, e - j->sent, NULL);
  else if (j->stream) {
    const char *s = j->all.p;
    size_t a = j->sent > j->rp ? j->sent : j->rp, re = reason_end(j, e);
    if (a < re) sse_field(j, "reasoning_content", s + a, re - a);
    if (j->think_end >= 0) {
      size_t c = j->sent > (size_t)j->think_end ? j->sent : (size_t)j->think_end;
      if (!j->content_started)
        while (c < e && strchr(" \t\r\n", s[c])) c++;
      if (c < e) sse_field(j, "content", s + c, e - c), j->content_started = 1;
    }
  }
  j->sent = e;
}
static int on_tokens(void *ctx, const uint32_t *t, uint32_t n) {
  job *j = ctx;
  const int pace = PACE && j->stream;
  double t0 = 0, gap = 0;
  if (pace) { /* this step's tokens go out evenly over about the time a step takes */
    t0 = mono_s();
    if (j->pace_last > 0) {
      const double dt = t0 - j->pace_last;
      PACE_STEP = 0.7 * PACE_STEP + 0.3 * (dt < 0.25 ? dt : 0.25);
    }
    j->pace_last = t0;
    gap = n ? PACE_STEP / n : 0;
    pace_step();
  }
  for (uint32_t i = 0; i < n; i++) {
    if (is_stop(t[i])) {
      j->finish = 1;
      PACE_DUE = 0;
      return 1;
    }
    if (j->auto_think && j->n_tok == 0 && THINK_ID >= 0 && t[i] == (uint32_t)THINK_ID) /* flint: a reasoning block */
      j->think = 1, j->think_end = -1, j->rp = 7;
    uint32_t len;
    char *s = model_detok(&M, t[i], &len);
    sb_add(&j->all, s, len);
    free(s);
    j->n_tok++;
    if (pace) PACE_DUE = (pace_first_fast() && j->n_tok == 1u) ? 0.0 : (t0 + gap * i), flush_text(j, 0); /* one chunk per token */
  }
  PACE_DUE = 0;
  if (!pace) flush_text(j, 0);
  if (j->stream && j->tools && !FLINT && !INKLING2 && j->tc_at >= 0 && !j->dead) { /* the tool calls as they are written */
    if (j->tcs_pos == (size_t)-1) j->tcs_pos = (size_t)j->tc_at;
    tcs_run(j);
  }
  if (j->dead || (j->n_tok % 8 == 0 && peer_gone(j->fd))) {
    j->finish = 2;
    return 1;
  }
  return 0;
}

/* ---- the hit rate: how many prompt tokens the cache supplied (a log line every IE_HIT_LOG_N requests, default 10, and
 * when idle) ---- */
static uint64_t HIT_WIN[4], HIT_ALL[3]; /* requests, prompt tokens, cached tokens, (window only:) cached from the disk cache */
static void hit_log(void) {
  if (!HIT_WIN[0]) return;
  fprintf(stderr, "cache: %lu requests: %lu prompt tokens, %lu cached (%.1f%%), %lu of them from the disk cache; since the start %lu requests, %.1f%% of %lu tokens\n",
          (unsigned long)HIT_WIN[0], (unsigned long)HIT_WIN[1], (unsigned long)HIT_WIN[2],
          HIT_WIN[1] ? 100.0 * (double)HIT_WIN[2] / (double)HIT_WIN[1] : 0.0, (unsigned long)HIT_WIN[3], (unsigned long)HIT_ALL[0],
          HIT_ALL[1] ? 100.0 * (double)HIT_ALL[2] / (double)HIT_ALL[1] : 0.0, (unsigned long)HIT_ALL[1]);
  memset(HIT_WIN, 0, sizeof HIT_WIN);
}
static void hit_note(const gen_stats *st) {
  static int every = -1;
  if (every < 0) every = getenv("IE_HIT_LOG_N") ? atoi(getenv("IE_HIT_LOG_N")) : 10;
  HIT_WIN[0]++, HIT_WIN[1] += st->n_prompt, HIT_WIN[2] += st->cached, HIT_WIN[3] += st->cache_hit == GEN_CK_DISK ? st->cached : 0;
  HIT_ALL[0]++, HIT_ALL[1] += st->n_prompt, HIT_ALL[2] += st->cached;
  if (every > 0 && HIT_WIN[0] >= (uint64_t)every) hit_log();
}

static void run_job(job *j) {
  if (peer_gone(j->fd)) { /* gone while it waited in the queue */
    fprintf(stderr, "[%lu] client gone before the start\n", j->rid);
    return;
  }
  vision_emb *vembs = NULL;
  gen_img *gimgs = NULL;
  if (j->n_img_urls) {
    char verr[256] = "";
    vembs = calloc(j->n_img_urls, sizeof *vembs);
    gimgs = calloc(j->n_img_urls, sizeof *gimgs);
    if (!vembs || !gimgs) ie_die("vision: out of memory");
    for (size_t k = 0; k < j->n_img_urls; k++) {
      if (!vision_encode_url(VIS, j->img_urls[k].url, j->img_urls[k].len, &vembs[k], verr, sizeof verr)) {
        for (size_t z = 0; z < k; z++) vision_emb_free(&vembs[z]);
        free(vembs), free(gimgs);
        http_error(j->fd, 400, "invalid_request_error", "image %zu: %s", k + 1, verr);
        return;
      }
    }
    /* Expand each 1-token IMAGE_PAD_ID placeholder into vembs[k].n_tok tokens */
    tok_ids exp = {0};
    size_t ik = 0;
    uint32_t new_last_user = j->gp.last_user;
    for (uint32_t p = 0; p < j->prompt.n; p++) {
      if (ik < j->n_img_urls && p == j->img_slot[ik]) {
        const uint32_t at = exp.n;
        const uint32_t nt = vembs[ik].n_tok;
        for (uint32_t z = 0; z < nt; z++) tok_ids_push(&exp, (uint32_t)IMAGE_PAD_ID);
        gimgs[ik] = (gen_img){at, nt, vembs[ik].nx, vembs[ik].ny, vembs[ik].hash, vembs[ik].embd};
        if (p < j->gp.last_user && nt > 1u) new_last_user += nt - 1u;
        ik++;
      } else {
        tok_ids_push(&exp, j->prompt.v[p]);
      }
    }
    tok_ids_free(&j->prompt);
    j->prompt = exp;
    j->gp.last_user = new_last_user;
    const uint32_t used = j->prompt.n + spec_room(O.draft);
    if (used >= O.ctx) {
      for (size_t z = 0; z < j->n_img_urls; z++) vision_emb_free(&vembs[z]);
      free(vembs), free(gimgs);
      http_error(j->fd, 400, "invalid_request_error",
                 "the prompt with %zu image(s) has %u tokens; the context is %u (--ctx)", j->n_img_urls, j->prompt.n, O.ctx);
      return;
    }
    uint32_t room = O.ctx - used;
    if (j->req_max_tok && j->req_max_tok < room) room = j->req_max_tok;
    j->gp.max_new = room;
    j->gp.imgs = gimgs;
    j->gp.n_imgs = (uint32_t)j->n_img_urls;
  }
  if (j->stream) {
    const char *h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if (pace_first_fast()) {
      sbuf b = {0};
      sb_puts(&b, h);
      sse_begin(&b, j);
      sb_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n");
      const int serr = send_all(j->fd, b.p, b.n);
      sb_free(&b);
      if (serr) {
        for (size_t z = 0; z < j->n_img_urls; z++) vision_emb_free(&vembs[z]);
        free(vembs), free(gimgs);
        j->gp.imgs = NULL, j->gp.n_imgs = 0;
        return;
      }
    } else {
      if (send_all(j->fd, h, strlen(h))) {
        for (size_t z = 0; z < j->n_img_urls; z++) vision_emb_free(&vembs[z]);
        free(vembs), free(gimgs);
        j->gp.imgs = NULL, j->gp.n_imgs = 0;
        return;
      }
      sse_chunk(j, NULL, 0, NULL); /* the role */
    }
  }
  j->think_end = j->think ? -1 : 0, j->tc_at = -1, j->rp = 8;
  j->tcs_pos = (size_t)-1, j->tcs_st = 0, j->tcs_k = 0, j->tcs_off = 0;
  snprintf(j->tcs_idp, sizeof j->tcs_idp, "call_%lx%lu_", (unsigned long)T_START, j->rid);
  sb_add(&j->all, "", 0);
  if (j->think) sb_add(&j->all, "<think>\n", 8); /* the prompt opened <think>: the client sees the whole block (Open WebUI folds it) */
  flush_text(j, 0);
  j->gp.emit = on_tokens, j->gp.ctx = j;
  gen_stats st;
  gen_run(&GC, j->prompt.v, j->prompt.n, &j->gp, &st);
  if (vembs) {
    for (size_t z = 0; z < j->n_img_urls; z++) vision_emb_free(&vembs[z]);
    free(vembs), free(gimgs);
    j->gp.imgs = NULL, j->gp.n_imgs = 0;
  }

  /* tool calls: the text from the first <tool_call> */
  sbuf calls = {0};
  const char *idp = j->tcs_idp;
  int n_calls = 0;
  if (!INKLING2 && j->tc_at >= 0 && j->finish != 2) {
    n_calls = FLINT ? chat_tool_calls_flint(j->all.p + j->tc_at, j->all.n - (size_t)j->tc_at, j->tools, idp, &calls)
                    : chat_tool_calls(j->all.p + j->tc_at, j->all.n - (size_t)j->tc_at, j->tools, idp, &calls);
    if (!n_calls) j->tc_at = -1; /* not valid: send it as text */
  }
  flush_text(j, 1); /* the rest of the text (an invalid UTF-8 sequence becomes U+FFFD) */
  if (INKLING2) {
    n_calls = j->ink_n_calls;
    if (n_calls) sb_add(&j->ink_calls, "]", 1);
  }
  const char *fin = j->ink_stream_invalid ? "network_error" : n_calls ? "tool_calls" : j->finish == 1 ? "stop" : "length";

  const uint32_t n_gen = j->n_tok;
  /* usage: the OpenAI counts, then the speed (Open WebUI shows all of usage
   * in the (i) tooltip under an answer) */
  sbuf us = {0};
  {
    const double gs = st.gen_ms / 1e3, ps = st.prefill_ms / 1e3, nc = (double)(st.n_prompt - st.cached);
    sb_printf(&us, "{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u", j->prompt.n, n_gen, j->prompt.n + n_gen);
    sb_printf(&us, ",\"prompt_tokens_details\":{\"cached_tokens\":%u}", st.cached);
    sb_printf(&us, ",\"response_token/s\":%.1f,\"prompt_token/s\":%.0f", gs > 0 ? n_gen / gs : 0.0, ps > 0 ? nc / ps : 0.0);
    sb_printf(&us, ",\"prompt_ms\":%.0f,\"response_ms\":%.0f", st.prefill_ms, st.gen_ms);
    if (st.steps) sb_printf(&us, ",\"mtp_tokens_per_step\":%.2f", (double)st.n_out / st.steps);
    sb_puts(&us, "}");
  }
  if (j->stream && j->finish != 2 && !j->dead) {
    if (n_calls && !INKLING2 && j->tcs_k) tcs_finish(j, calls.p, calls.n); /* the rest of the streamed calls */
    else if (n_calls && !INKLING2) sse_tool_calls(j, calls.p, calls.n);
    { /* the last chunk carries usage too (as llama.cpp does), so clients that do not ask still see it */
      sbuf b = {0};
      sse_begin(&b, j);
      sb_printf(&b, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],\"usage\":%s}\n\n", fin, us.p);
      sse_send(j, &b);
    }
    if (j->usage) {
      sbuf b = {0};
      sse_begin(&b, j);
      sb_printf(&b, ",\"choices\":[],\"usage\":%s}\n\n", us.p);
      sse_send(j, &b);
    }
    sbuf b = {0};
    sb_puts(&b, "data: [DONE]\n\n");
    sse_send(j, &b);
  } else if (!j->stream && j->finish != 2) {
    sbuf b = {0};
    sb_printf(&b, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", j->id, j->created);
    sb_json_str(&b, MODEL_ID, strlen(MODEL_ID));
    sb_puts(&b, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    if (INKLING2) {
      sb_json_str(&b, j->ink_content.p ? j->ink_content.p : "", j->ink_content.n);
      if (!O.raw_think && j->ink_reason.n) {
        sb_puts(&b, ",\"reasoning_content\":");
        sb_json_str(&b, j->ink_reason.p, j->ink_reason.n);
      }
      if (n_calls) sb_puts(&b, ",\"tool_calls\":"), sb_add(&b, j->ink_calls.p, j->ink_calls.n);
    } else {
      size_t cn = j->sent; /* the content: the text before the tool calls, without the white space before them */
      if (n_calls)
        while (cn && strchr(" \t\r\n", j->all.p[cn - 1])) cn--;
      if (!split_think(j)) sb_json_str(&b, j->all.p ? j->all.p : "", cn);
      else { /* content after </think> (without the white space), then reasoning_content */
        const char *s = j->all.p;
        size_t c = j->think_end >= 0 ? (size_t)j->think_end : cn, re = reason_end(j, cn);
        while (c < cn && strchr(" \t\r\n", s[c])) c++;
        while (re > j->rp && strchr(" \t\r\n", s[re - 1])) re--;
        sb_json_str(&b, c < cn ? s + c : "", c < cn ? cn - c : 0);
        sb_puts(&b, ",\"reasoning_content\":");
        sb_json_str(&b, re > j->rp ? s + j->rp : "", re > j->rp ? re - j->rp : 0);
      }
      if (n_calls) sb_puts(&b, ",\"tool_calls\":"), sb_add(&b, calls.p, calls.n);
    }
    sb_printf(&b, "},\"finish_reason\":\"%s\"}],\"usage\":%s}", fin, us.p);
    http_reply(j->fd, 200, "application/json", b.p, b.n);
    sb_free(&b);
  }
  sb_free(&calls), sb_free(&us);
  if (PACE && j->stream) pace_drain(); /* job_free closes the connection */
  for (int i = 0; i < TCS_MAX; i++) sb_free(&j->tcs_sent[i]);
  const double gs = st.gen_ms / 1e3;
  static const char *const ckn[] = {" with the last prompt", " with the last answer", " from where a prompt changed", " from a checkpoint",
                                    " from the disk cache", " from where a prompt is likely to change", " from an agent prefix"};
  fprintf(stderr, "[%lu] prompt %u tokens (%u cached%s, %.0f ms; %u in common, %u checkpoints), output %u tokens in %.2f s (%.1f tokens/s)", j->rid, st.n_prompt,
          st.cached, st.cache_hit >= 0 && st.cache_hit <= GEN_CK_PREFIX ? ckn[st.cache_hit] : "", st.prefill_ms, st.common, st.n_ck, n_gen, gs,
          gs > 0 ? n_gen / gs : 0.0);
  if (st.steps) fprintf(stderr, ", mtp %.2f tokens per step", (double)st.n_out / st.steps);
  if (st.dry_hits) fprintf(stderr, ", loop penalty on %u candidates", st.dry_hits);
  if (st.reasoning_limited) fprintf(stderr, ", closed reasoning at %u tokens", j->gp.reasoning_budget);
  if (n_calls) fprintf(stderr, ", %d tool calls", n_calls);
  fprintf(stderr, ", %s\n", j->finish == 2 ? "client gone" : fin);
  if (j->finish != 2 && st.n_prompt) hit_note(&st);
}

static void job_free(job *j) {
  if (j->fd >= 0) close(j->fd);
  tok_ids_free(&j->prompt);
  free(j->img_urls);
  free(j->img_slot);
  sb_free(&j->all);
  sb_free(&j->ink_reason);
  sb_free(&j->ink_content);
  sb_free(&j->ink_calls);
  json_free(j->req);
  free(j);
}

/* ---- disk thread: writes the disk cache's files (gen_disk_write), so the next request does not wait for them ---- */
static pthread_mutex_t DCM = PTHREAD_MUTEX_INITIALIZER, DKM = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t DKC = PTHREAD_COND_INITIALIZER;
static int DK_KICK, DK_PREWARM;
static void disk_lock(int on) {
  if (on) pthread_mutex_lock(&DCM);
  else pthread_mutex_unlock(&DCM);
}
static void *disk_main(void *arg) {
  (void)arg;
  for (;;) {
    pthread_mutex_lock(&DKM);
    while (!DK_KICK && !DK_PREWARM) pthread_cond_wait(&DKC, &DKM);
    const int kick = DK_KICK;
    const int prewarm = DK_PREWARM;
    DK_KICK = 0;
    pthread_mutex_unlock(&DKM);
    if (kick) gen_disk_write(&GC);
    if (prewarm && GC.dc) {
      pthread_mutex_lock(&QM);
      const int active = BUSY || QN > 0;
      pthread_mutex_unlock(&QM);
      if (active) {
#ifdef _WIN32
        Sleep(20);
#else
        struct timespec ts = {0, 20000000};
        nanosleep(&ts, NULL);
#endif
      } else {
        const int more = dc_prewarm_step(&GC, 16u << 20);
        if (more <= 0) {
          pthread_mutex_lock(&DKM);
          DK_PREWARM = 0;
          pthread_mutex_unlock(&DKM);
        }
      }
    }
  }
  return NULL;
}

static void disk_kick(void) { /* the disk thread writes what was queued (also what a request queued by itself: a switch, an eviction) */
  pthread_mutex_lock(&DKM);
  DK_KICK = 1;
  pthread_cond_signal(&DKC);
  pthread_mutex_unlock(&DKM);
}

/* ---- engine thread: load, then run the queue ---- */
static void *engine_main(void *arg) {
  (void)arg;
  const int gpu = !strcmp(O.backend, "gpu");
  gguf_open(&G, O.path);
  const char *se = getenv("IE_STREAM");
  if (gpu && !(se && se[0] == '0')) gpu_stream_init();
  if (O.lora) {
    gguf_file adapter;
    gguf_open(&adapter, O.lora);
    model_load_lora(&M, &G, &adapter);
    gguf_close(&adapter);
  } else model_load(&M, &G);
  if (O.draft == DRAFT_AUTO) { /* no --draft: drafts if there is a head */
    O.draft = gpu && (O.mtp || gguf_has_mtp(&G) || O.dflash || O.dspark) ? MTP_DRAFT_DEFAULT : 0u;
    if (!O.draft && gpu) fprintf(stderr, "ie-serve: no MTP head in %s (--mtp FILE): no speculative decoding\n", O.path);
  }
  if (O.draft && (O.mtp || gguf_has_mtp(&G) || (!O.dflash && !O.dspark))) { /* a DFlash/DSpark drafter alone needs no MTP head */
    spec_set_draft(O.draft);
    if (O.mtp) gguf_open(&G2, O.mtp);
    if (!model_load_mtp(&M, O.mtp ? &G2 : &G)) ie_die("no MTP head (blk.N.nextn.*) in %s", O.mtp ? O.mtp : O.path);
    const char *dh = getenv("IE_DRAFT_HEAD");
    if (!(dh && !strcmp(dh, "q8"))) model_make_draft_head(&M, &G);
  } else if (O.draft && O.dflash) {
    spec_set_draft(O.draft);
    /* the draft vocabulary (IE_DRAFT_VOCAB, default the first 65536 ids): model_make_df_head builds the drafter's Q8
     * head over it instead of the full output matrix (Qwen3-8B: 285 instead of 510 MB read per draft block) */
    model_make_draft_head(&M, &G);
    fprintf(stderr, "ie-serve: no MTP head: every step drafts with DFlash\n");
  } else if (O.draft && O.dspark) {
    spec_set_draft(O.draft);
    fprintf(stderr, "ie-serve: every step drafts with DSpark\n");
  }
  if (O.dflash) {
    gguf_open(&G3, O.dflash);
    if (!model_load_dflash(&M, &G3)) ie_die("%s is not a DFlash drafter", O.dflash);
    model_make_df_head(&M, &G);
  }
  if (O.dspark && O.draft) {
    gguf_file Gds;
    gguf_open(&Gds, O.dspark);
    if (!model_load_dspark(&M, &Gds)) ie_die("%s is not a DSpark drafter", O.dspark);
    gguf_close(&Gds);
    model_make_ds_head(&M, &G);
  }
  char err[256];
  if (!(TK = tok_open(&M, &G, err, sizeof err))) ie_die("tokenizer: %s", err);
  THINK_END_ID = tok_find(TK, "</think>");
  TOOL_CALL_OPEN_ID = tok_find(TK, "<tool_call>");
  TOOL_CALL_CLOSE_ID = tok_find(TK, "</tool_call>");
  VISION_START_ID = tok_find(TK, "<|vision_start|>");
  VISION_END_ID = tok_find(TK, "<|vision_end|>");
  IMAGE_PAD_ID = tok_find(TK, "<|image_pad|>");
  for (int i = 0; i < 8; i++) STOP_IDS[i] = -1;
  int64_t eos = -1;
  if (!gguf_get_int(&G, "tokenizer.ggml.eos_token_id", &eos)) eos = -1;
  FLINT = M.arch == ARCH_FLINT;
  INKLING2 = M.arch == ARCH_INKLING2;
  if (!O.chunk && getenv("IE_CHUNK")) O.chunk = (uint32_t)atoi(getenv("IE_CHUNK"));
  if (!O.chunk) O.chunk = INKLING2 ? 1024u : 512u;
  if (O.chunk < 1u || O.chunk > 1024u) ie_die("--chunk: 1 to 1024");
  if (FLINT) { /* the end-of-generation tokens: <|end_of_text|>, <|user|>, <|observation|>, <|developer|> */
    THINK_ID = tok_find(TK, "<think>");
    const gguf_kv *eg = gguf_find(&G, "tokenizer.ggml.eog_token_ids");
    int ns = 0;
    if (eg && eg->type == GGUF_ARR && (eg->arr_type == GGUF_I32 || eg->arr_type == GGUF_U32))
      for (uint64_t i = 0; i < eg->arr_n && ns < 7; i++) {
        int32_t x;
        memcpy(&x, eg->arr_data + 4 * i, 4);
        STOP_IDS[ns++] = x;
      }
    if (!ns) STOP_IDS[ns++] = eos, STOP_IDS[ns++] = tok_find(TK, "<|user|>"), STOP_IDS[ns++] = tok_find(TK, "<|observation|>");
  } else if (INKLING2) {
    TOOL_CALL_OPEN_ID = tok_find(TK, "<|content_invoke_tool_json|>");
    TOOL_CALL_CLOSE_ID = tok_find(TK, "<|end_message|>");
    THINK_END_ID = tok_find(TK, "<|content_text|>");
    IMAGE_PAD_ID = tok_find(TK, "<|unused_200054|>");
    const int64_t end_sampling = tok_find(TK, "<|content_model_end_sampling|>");
    STOP_IDS[0] = eos >= 0 ? eos : end_sampling;
    STOP_IDS[1] = tok_find(TK, "<|endoftext|>");
    if (end_sampling >= 0 && end_sampling != STOP_IDS[0] && end_sampling != STOP_IDS[1]) STOP_IDS[2] = end_sampling;
  } else {
    char *tpl = gguf_get_str(&G, "tokenizer.chat_template");
    const char *ct = getenv("IE_CHAT_TEMPLATE"); /* qwen or mimo: overrides the detection */
    MIMO = ct ? !strcmp(ct, "mimo") : tpl && strstr(tpl, "mimo_audio_start") && strstr(tpl, "render_assistant_message");
    free(tpl);
    if (MIMO) THINK_ID = tok_find(TK, "<think>"), fprintf(stderr, "ie-serve: chat template MiMo v2.6\n");
    IM_START = tok_find(TK, "<|im_start|>"), IM_END = tok_find(TK, "<|im_end|>");
    if (IM_START < 0 || IM_END < 0) ie_die("the vocabulary has no <|im_start|> or <|im_end|> (ChatML)");
    STOP_IDS[0] = eos, STOP_IDS[1] = IM_END, STOP_IDS[2] = tok_find(TK, "<|endoftext|>");
  }
  const char *xs = getenv("IE_EXTRA_STOP"); /* tests: one more stop token id */
  if (xs) STOP_IDS[7] = atoll(xs);
  if (!O.name) {
    char *gn = gguf_get_str(&G, "general.name");
    snprintf(MODEL_ID, sizeof MODEL_ID, "%s", gn ? gn : "amd-infer");
    free(gn);
  } else snprintf(MODEL_ID, sizeof MODEL_ID, "%s", O.name);
  /* no white space in the id (Open WebUI refuses a model id with spaces): "Qwen3.8 27B" -> "Qwen3.8-27B" */
  for (char *c = MODEL_ID; *c; c++)
    if (*c == ' ' || *c == '\t' || *c == '\n' || *c == '\r') *c = '-';
  /* default sampling: thinking takes the model file's general.sampling.* (copied from its generation_config,
     which holds the thinking values; Qwen3.8 27B: 1.0, 0.95, 20), else Qwen's thinking 0.6, 0.95, 20; no
     thinking takes Qwen's 0.7, 0.8, 20. --temp/--top-p/--top-k replace both; --think-temp then replaces the
     thinking temperature alone. */
  double sv;
  int64_t sk;
  SAMPLING[1] = (sampling){gguf_get_float(&G, "general.sampling.temp", &sv) ? (float)sv : 0.6f,
                           gguf_get_float(&G, "general.sampling.top_p", &sv) ? (float)sv : 0.95f,
                           gguf_get_int(&G, "general.sampling.top_k", &sk) && sk >= 0 ? (uint32_t)sk : 20u};
  SAMPLING[0] = (sampling){0.7f, 0.8f, 20u};
  for (int m = 0; m < 2; m++) {
    if (O.temp >= 0) SAMPLING[m].temp = O.temp;
    if (O.top_p >= 0) SAMPLING[m].top_p = O.top_p;
    if (O.top_k != UINT32_MAX) SAMPLING[m].top_k = O.top_k;
  }
  if (O.think_temp >= 0) SAMPLING[1].temp = O.think_temp;
  fprintf(stderr, "ie-serve: default sampling: thinking temp %g, top_p %g, top_k %u; no thinking temp %g, top_p %g, top_k %u\n",
          SAMPLING[1].temp, SAMPLING[1].top_p, SAMPLING[1].top_k, SAMPLING[0].temp, SAMPLING[0].top_p, SAMPLING[0].top_k);

  const uint32_t chunk = gpu ? (O.chunk = gpu_chunk_clamp(O.chunk, &M, O.ctx)) : 1u;
  uint32_t gT = O.draft ? spec_verify_max() : 1u;
  if (O.draft + 1u > gT) gT = O.draft + 1u;
  if (chunk > gT) gT = chunk;
  graph_build(&GR, &M, O.ctx, gT, O.draft > 0);
  graph_cap_logits(&GR); /* 16 rows of logits, not one per prompt chunk token */
  if (gpu) graph_cap_exp_gu(&GR, &M);
  plan_arena(&GR, 1);
  B = gpu ? gpu_open(&M, &GR, O.hsaco) : cpu_open(&M, &GR);
  gen_init(&GC, B, &M, &GR);
  if (O.mmproj) {
    if (INKLING2) {
      if (IMAGE_PAD_ID < 0) ie_die("--mmproj: Inkling-2 model vocabulary lacks <|unused_200054|>");
    } else if (VISION_START_ID < 0 || VISION_END_ID < 0 || IMAGE_PAD_ID < 0) {
      ie_die("--mmproj: model vocabulary lacks <|vision_start|>/<|image_pad|>/<|vision_end|>");
    }
    VIS = vision_open(O.mmproj, gpu ? O.hsaco : NULL, M.dim);
  }
  const char *dcd = getenv("IE_DISK_CACHE");
  const char *rce = getenv("IE_RAM_CACHE_GB");
  const int ram_l2_req = rce && rce[0] && atof(rce) > 0.0;
  if (((dcd && dcd[0]) || ram_l2_req) && O.cache) { /* what the K/V rows and states depend on: the files (size, time) and the cache format */
    sbuf sig = {0};
    const char *files[4] = {O.path, O.draft ? O.mtp : NULL, gpu ? O.hsaco : NULL, O.dflash ? O.dflash : O.dspark};
    for (int i = 0; i < 4; i++) {
      struct stat fs;
      if (files[i] && !stat(files[i], &fs)) sb_printf(&sig, "%d:%lld:%lld|", i, (long long)fs.st_size, (long long)fs.st_mtime);
    }
    if (O.lora) {
      struct stat fs;
      if (stat(O.lora, &fs)) ie_die("cannot stat LoRA for disk cache: %s", O.lora);
      sb_printf(&sig, "lora=%s:%lld:%lld|", O.lora, (long long)fs.st_size, (long long)fs.st_mtime);
    }
    static const char *const env[] = {"IE_KV_Q4", "IE_KV_Q8", "IE_KV_ROT", "IE_MTP_EMBED", "IE_MK"};
    for (size_t i = 0; i < sizeof env / sizeof *env; i++) sb_printf(&sig, "%s=%s|", env[i], getenv(env[i]) ? getenv(env[i]) : "");
    sb_printf(&sig, "gpu=%d|draft=%d", gpu, O.draft > 0);
    const char *gb = getenv("IE_DISK_CACHE_GB");
    if (!gen_disk_open(&GC, (dcd && dcd[0]) ? dcd : NULL, sig.p, gb ? atof(gb) : 100.0, disk_lock)) fprintf(stderr, "disk cache: not with this backend or model\n");
    else {
      pthread_t dt;
      pthread_create(&dt, NULL, disk_main, NULL);
      GC.disk_kick = disk_kick;
    }
    sb_free(&sig);
  }
  fprintf(stderr, "model \"%s\": vocab %u, pre-tokenizer %s, context %u, drafts %u%s, stop tokens", MODEL_ID,
          M.vocab_unpadded ? M.vocab_unpadded : M.vocab, tok_pre(TK), O.ctx, O.draft, M.df ? " (+ DFlash)" : "");
  for (int i = 0; i < 8; i++)
    if (STOP_IDS[i] >= 0) fprintf(stderr, " %lld", (long long)STOP_IDS[i]);
  fprintf(stderr, "%s\n", FLINT ? " (flint chat template)" : INKLING2 ? " (inkling2 chat template)" : "");

  pthread_mutex_lock(&QM);
  IS_READY = 1;
  pthread_cond_broadcast(&READY);
  pthread_mutex_unlock(&QM);
  /* IE_DISK_IDLE_S (default 10): the disk cache gets the checkpoints that are not in it yet when no request has come for
   * this long (and when a request leaves the cached conversation, or a full cache drops one: gen.h). 0: right after
   * each request */
  const char *ie = getenv("IE_DISK_IDLE_S");
  const double idle_s = ie && ie[0] ? atof(ie) : 10.0;
  /* R83 (H50, IE_DISK_PREWARM promoted default ON; =0 opts out): warm <= 16 MiB per tick on disk_main while idle. */
  const char *pwe = getenv("IE_DISK_PREWARM");
  const int prewarm_on = !(pwe && pwe[0] == '0');
  if (prewarm_on && GC.dc != NULL) {
    pthread_mutex_lock(&DKM);
    DK_PREWARM = 1;
    pthread_cond_signal(&DKC);
    pthread_mutex_unlock(&DKM);
  }
  struct timespec dl = {0, 0}; /* the end of the idle time; tv_sec 0: not running */
  for (;;) {
    pthread_mutex_lock(&QM);
    while (!QH) {
      if (!((GC.dc && idle_s > 0 && gen_disk_pending(&GC) > 0) || HIT_WIN[0] > 0)) { /* nothing to do when idle */
        dl.tv_sec = 0;
        pthread_cond_wait(&QC, &QM);
        continue;
      }
      if (!dl.tv_sec) { /* the idle time starts */
        const double w = idle_s > 0 ? idle_s : 10.0;
        timespec_get(&dl, TIME_UTC);
        dl.tv_sec += (time_t)w, dl.tv_nsec += (long)((w - (double)(time_t)w) * 1e9);
        if (dl.tv_nsec >= 1000000000L) dl.tv_sec++, dl.tv_nsec -= 1000000000L;
      }
      if (pthread_cond_timedwait(&QC, &QM, &dl) == ETIMEDOUT && !QH) { /* idle: the engine thread makes the copies */
        dl.tv_sec = 0;
        pthread_mutex_unlock(&QM);
        if (GC.dc && idle_s > 0 && gen_disk_idle(&GC)) disk_kick();
        hit_log();
        pthread_mutex_lock(&QM);
      }
    }
    job *j = QH;
    QH = j->next;
    if (!QH) QT = NULL;
    QN--, BUSY = 1, dl.tv_sec = 0;
    pthread_mutex_unlock(&QM);
    run_job(j);
    job_free(j);         /* the client has the whole answer */
    gpu_trace_flush();
    if (GC.dc) { /* the disk thread writes what the request queued (a switch, an eviction) and, with no idle time, its new checkpoints */
      if (idle_s <= 0) gen_disk_save(&GC);
      disk_kick();
    }
    pthread_mutex_lock(&QM);
    BUSY = 0;
    pthread_mutex_unlock(&QM);
  }
  return NULL;
}

/* ---- the request: HTTP ---- */
enum { AGENT_NONE, AGENT_OPENCODE, AGENT_OMP };
typedef struct {
  char method[16], path[256];
  char *body;
  size_t body_n;
  int has_len, chunked, expect100, agent;
  size_t len;
  char auth[512];
} request;

/* Read the head and the body. 0: ok; otherwise the HTTP code to send. */
static int read_request(int fd, request *r) {
  memset(r, 0, sizeof *r);
  char *buf = malloc(65536 + 1);
  size_t n = 0;
  char *he = NULL;
  while (!he) {
    if (n == 65536) { free(buf); return 413; }
    ssize_t k = recv(fd, buf + n, 65536 - n, 0);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) { free(buf); return -1; }
    n += (size_t)k;
    buf[n] = 0;
    he = strstr(buf, "\r\n\r\n");
  }
  *he = 0;
  const size_t head_n = (size_t)(he - buf) + 4;
  if (sscanf(buf, "%15s %255s", r->method, r->path) != 2) { free(buf); return 400; }
  char *q = strchr(r->path, '?');
  if (q) *q = 0;
  for (char *line = strstr(buf, "\r\n"); line; line = strstr(line, "\r\n")) {
    line += 2;
    char *colon = strchr(line, ':'), *eol = strstr(line, "\r\n");
    if (!colon || (eol && colon > eol)) continue;
    char *v = colon + 1;
    while (*v == ' ' || *v == '\t') v++;
    const size_t vn = eol ? (size_t)(eol - v) : strlen(v);
    const size_t kn = (size_t)(colon - line);
    if (kn == 14 && !strncasecmp(line, "Content-Length", 14)) r->len = strtoul(v, NULL, 10), r->has_len = 1;
    else if (kn == 17 && !strncasecmp(line, "Transfer-Encoding", 17) && vn >= 7 && !strncasecmp(v, "chunked", 7)) r->chunked = 1;
    else if (kn == 6 && !strncasecmp(line, "Expect", 6) && vn >= 12 && !strncasecmp(v, "100-continue", 12)) r->expect100 = 1;
    else if (kn == 13 && !strncasecmp(line, "Authorization", 13)) snprintf(r->auth, sizeof r->auth, "%.*s", (int)vn, v);
    else if (kn == 10 && !strncasecmp(line, "User-Agent", 10)) {
      if (vn >= 9 && !strncasecmp(v, "opencode/", 9)) r->agent = AGENT_OPENCODE;
      else if (vn >= 4 && !strncasecmp(v, "omp/", 4)) r->agent = AGENT_OMP;
    }
  }
  if (!strcmp(r->method, "POST")) {
    if (r->chunked) { free(buf); return 411; }
    if (!r->has_len) { free(buf); return 411; }
    if (r->len > O.max_body) { free(buf); return 413; }
    if (r->expect100) send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n", 25);
    r->body = malloc(r->len + 1);
    size_t have = n - head_n < r->len ? n - head_n : r->len;
    memcpy(r->body, buf + head_n, have);
    while (have < r->len) {
      ssize_t k = recv(fd, r->body + have, r->len - have, 0);
      if (k < 0 && errno == EINTR) continue;
      if (k <= 0) { free(buf); free(r->body); r->body = NULL; return -1; }
      have += (size_t)k;
    }
    r->body[r->len] = 0;
    r->body_n = r->len;
  }
  free(buf);
  return 0;
}

/* The compare time does not depend on where the strings differ. */
static int same_secret(const char *a, const char *b) {
  const size_t na = strlen(a), nb = strlen(b);
  unsigned char d = na != nb;
  for (size_t i = 0; i < na; i++) d |= (unsigned char)(a[i] ^ b[i % (nb ? nb : 1)]);
  return d == 0;
}

/* ---- R82 (H47a): opt-in segment-level FNV-64 BPE cache (IE_TOK_SEG_CACHE=1, default OFF) ----
 * Bounded to <= 256 entries and <= 4 MiB of cached token IDs (<= 16 MiB exact text copies).
 * Every hit verifies exact (hash, len, parse_ctrl) plus full byte memcmp -> 0% collision risk. */
#define SEG_CACHE_MAX_ENT 256u
#define SEG_CACHE_MAX_TOK_BYTES (4u << 20)
#define SEG_CACHE_MAX_TXT_BYTES (16u << 20)
#define SEG_CACHE_MIN_LEN 32u

typedef struct {
  uint64_t hash;
  size_t len;
  int parse_ctrl;
  char *text;
  uint32_t *ids;
  uint32_t n_ids;
  uint64_t used;
} seg_cache_ent;

static pthread_mutex_t SEGM = PTHREAD_MUTEX_INITIALIZER;
static seg_cache_ent SEG_ENTS[SEG_CACHE_MAX_ENT];
static uint32_t SEG_N;
static size_t SEG_TOK_BYTES, SEG_TXT_BYTES;
static uint64_t SEG_TICK;

static int tok_seg_cache_on(void) {
  const char *e = getenv("IE_TOK_SEG_CACHE");
  return !(e && e[0] == '0'); /* R82 promoted default ON (=0 opts out) */
}

static uint64_t seg_fnv64(const char *s, size_t n, int parse_ctrl) {
  uint64_t h = 14695981039346656037ull ^ (uint64_t)(parse_ctrl ? 0x5Au : 0xA5u);
  for (size_t i = 0u; i < n; i++) h = (h ^ (unsigned char)s[i]) * 1099511628211ull;
  return h;
}

static void serve_tok_ids_append(tok_ids *a, const uint32_t *src, uint32_t n) {
  if (!n) return;
  if (a->n + n > a->cap) {
    uint32_t cap = a->cap ? a->cap : 64u;
    while (cap < a->n + n) {
      if (cap > UINT32_MAX / 2u) { cap = a->n + n; break; }
      cap *= 2u;
    }
    a->v = realloc(a->v, (size_t)cap * 4u);
    if (!a->v) ie_die("ie-serve: out of memory");
    a->cap = cap;
  }
  memcpy(a->v + a->n, src, (size_t)n * 4u);
  a->n += n;
}

static void tok_encode_seg(const tokenizer *t, const char *text, size_t len, int parse_ctrl, tok_ids *out) {
  if (!tok_seg_cache_on() || len < SEG_CACHE_MIN_LEN || len > SEG_CACHE_MAX_TXT_BYTES / 2u) {
    tok_encode(t, text, len, parse_ctrl, out);
    return;
  }
  const uint64_t h = seg_fnv64(text, len, parse_ctrl);
  pthread_mutex_lock(&SEGM);
  for (uint32_t i = 0u; i < SEG_N; i++) {
    seg_cache_ent *e = &SEG_ENTS[i];
    if (e->hash == h && e->len == len && e->parse_ctrl == parse_ctrl && !memcmp(e->text, text, len)) {
      e->used = ++SEG_TICK;
      serve_tok_ids_append(out, e->ids, e->n_ids);
      pthread_mutex_unlock(&SEGM);
      return;
    }
  }
  pthread_mutex_unlock(&SEGM);

  const uint32_t start = out->n;
  tok_encode(t, text, len, parse_ctrl, out);
  const uint32_t n_ids = out->n - start;
  const size_t id_bytes = (size_t)n_ids * 4u;
  if (!n_ids || id_bytes > SEG_CACHE_MAX_TOK_BYTES / 2u) return;

  char *tc = malloc(len);
  uint32_t *ic = malloc(id_bytes);
  if (!tc || !ic) {
    free(tc), free(ic);
    return;
  }
  memcpy(tc, text, len);
  memcpy(ic, out->v + start, id_bytes);

  pthread_mutex_lock(&SEGM);
  for (uint32_t i = 0u; i < SEG_N; i++) {
    seg_cache_ent *e = &SEG_ENTS[i];
    if (e->hash == h && e->len == len && e->parse_ctrl == parse_ctrl && !memcmp(e->text, tc, len)) {
      e->used = ++SEG_TICK;
      pthread_mutex_unlock(&SEGM);
      free(tc), free(ic);
      return;
    }
  }
  while (SEG_N > 0u && (SEG_N >= SEG_CACHE_MAX_ENT || SEG_TOK_BYTES + id_bytes > SEG_CACHE_MAX_TOK_BYTES ||
                        SEG_TXT_BYTES + len > SEG_CACHE_MAX_TXT_BYTES)) {
    uint32_t v = 0u;
    for (uint32_t i = 1u; i < SEG_N; i++)
      if (SEG_ENTS[i].used < SEG_ENTS[v].used) v = i;
    SEG_TOK_BYTES -= (size_t)SEG_ENTS[v].n_ids * 4u;
    SEG_TXT_BYTES -= SEG_ENTS[v].len;
    free(SEG_ENTS[v].text), free(SEG_ENTS[v].ids);
    SEG_ENTS[v] = SEG_ENTS[--SEG_N];
  }
  SEG_ENTS[SEG_N++] = (seg_cache_ent){h, len, parse_ctrl, tc, ic, n_ids, ++SEG_TICK};
  SEG_TOK_BYTES += id_bytes;
  SEG_TXT_BYTES += len;
  pthread_mutex_unlock(&SEGM);
}

/* ---- the request: chat ---- */
/* The prompt tokens: control tokens as ids, text without control tokens
 * (a user cannot write <|im_start|> into the prompt). Also where the messages are (token positions, 0: none or not
 * known): *sys_end, the closing token of the first message when that is the system message; *last_user, the start of
 * the last message with the role user (a tool result is one). gen_run puts checkpoints there. */
/* Decoded byte length of one vocabulary token (O(len(token)) without malloc for GPT2 byte-level BPE). */
static size_t tok_byte_len(uint32_t t) {
  if (t >= M.n_tokens) return 0;
  const char *s = M.tokens[t];
  if (M.tok_gpt2) {
    size_t n = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++)
      if ((*p & 0xC0u) != 0x80u) n++;
    return n;
  }
  uint32_t len = 0;
  char *d = model_detok(&M, t, &len);
  free(d);
  return len;
}

static size_t prefix_boundary_off(const char *text, size_t len, const char *marker, int after) {
  const char *at = strstr(text, marker);
  if (!at || at >= text + len) return 0;
  const size_t n = (size_t)(at - text) + (after ? strlen(marker) : 0);
  return n <= len ? n : 0;
}

/* Resolve multiple byte-offset boundaries in one linear pass over out->v[start .. out->n),
 * avoiding 4x redundant BPE re-tokenization of the system prompt. */
static void prefix_resolve_all(const tok_ids *out, uint32_t start, const size_t *offs, uint32_t **dsts, int n_marks) {
  if (n_marks <= 0) return;
  int left = n_marks;
  size_t bpos = 0;
  for (uint32_t k = start; k < out->n && left > 0; k++) {
    const size_t tlen = tok_byte_len(out->v[k]);
    for (int m = 0; m < n_marks; m++) {
      if (!dsts[m]) continue;
      if (bpos == offs[m] || (bpos < offs[m] && bpos + tlen > offs[m])) {
        *dsts[m] = k;
        dsts[m] = NULL;
        left--;
      }
    }
    bpos += tlen;
  }
  for (int m = 0; m < n_marks; m++)
    if (dsts[m] && bpos >= offs[m]) *dsts[m] = out->n;
}

static void prompt_tokens(const chat_prompt *p, tok_ids *out, gen_params *gp, int agent, uint32_t *img_slots) {
  uint32_t *prefix = agent ? gp->prefix : NULL;
  if (FLINT) { /* the whole text with the control tokens (see chat_render_flint) */
    tok_encode_seg(TK, p->text.p, p->text.n, 1, out);
    return;
  }
  if (INKLING2) { /* gaps hold template control tags; SEG_TEXT holds untrusted message text; SEG_IMG holds image slot */
    size_t pos = 0;
    for (size_t i = 0; i < p->n_seg; i++) {
      const chat_seg *s = &p->seg[i];
      if (s->off > pos) tok_encode_seg(TK, p->text.p + pos, s->off - pos, 1, out);
      if (s->kind == SEG_IMG) {
        if (img_slots) img_slots[s->n] = out->n;
        tok_ids_push(out, (uint32_t)IMAGE_PAD_ID);
        pos = s->off;
      } else {
        tok_encode_seg(TK, p->text.p + s->off, s->n, 0, out);
        pos = s->off + s->n;
      }
    }
    if (p->text.n > pos) tok_encode_seg(TK, p->text.p + pos, p->text.n - pos, 1, out);
    return;
  }
  int first = 1, in_sys = 0; /* in_sys: inside the first message, which is the system message */
  for (size_t i = 0; i < p->n_seg; i++) {
    const chat_seg *s = &p->seg[i];
    if (s->kind == SEG_IM_START) {
      const chat_seg *nx = i + 1 < p->n_seg ? &p->seg[i + 1] : NULL;
      const char *role = nx && nx->kind == SEG_TEXT ? p->text.p + nx->off : "";
      const size_t rn = nx && nx->kind == SEG_TEXT ? nx->n : 0;
      if (first && rn >= 7 && !strncmp(role, "system\n", 7)) in_sys = 1;
      if (rn >= 5 && !strncmp(role, "user\n", 5)) gp->last_user = out->n;
      first = 0;
      tok_ids_push(out, (uint32_t)IM_START);
    } else if (s->kind == SEG_IM_END) {
      if (in_sys) {
        gp->sys_end = out->n, in_sys = 0;
        if (agent == AGENT_OPENCODE) prefix[2] = out->n;
      }
      tok_ids_push(out, (uint32_t)IM_END);
    } else if (s->kind == SEG_IMG) {
      tok_ids_push(out, (uint32_t)VISION_START_ID);
      if (img_slots) img_slots[s->off] = out->n;
      tok_ids_push(out, (uint32_t)IMAGE_PAD_ID);
      tok_ids_push(out, (uint32_t)VISION_END_ID);
    } else {
      const uint32_t start = out->n;
      const char *text = p->text.p + s->off;
      tok_encode_seg(TK, text, s->n, 0, out);
      if (prefix && in_sys) {
        size_t offs[4];
        uint32_t *dsts[4];
        int nm = 0;
        if (p->tools_end > s->off && p->tools_end <= s->off + s->n) {
          offs[nm] = p->tools_end - s->off;
          dsts[nm++] = &gp->tools_end;
        }
        if (s->n > 1024) {
          if (agent == AGENT_OPENCODE) {
            size_t o;
            if (!prefix[0] && (o = prefix_boundary_off(text, s->n, "</available_skills>", 1))) offs[nm] = o, dsts[nm++] = &prefix[0];
            if (!prefix[1] && (o = prefix_boundary_off(text, s->n, "\n<env>", 0))) offs[nm] = o, dsts[nm++] = &prefix[1];
          } else {
            size_t o;
            if (!prefix[0] && (o = prefix_boundary_off(text, s->n, "\n<skills>", 0))) offs[nm] = o, dsts[nm++] = &prefix[0];
            if (!prefix[1] && (o = prefix_boundary_off(text, s->n, "\n<project-context>", 0))) offs[nm] = o, dsts[nm++] = &prefix[1];
            if (!prefix[2] && (o = prefix_boundary_off(text, s->n, "\n<memories>", 0))) offs[nm] = o, dsts[nm++] = &prefix[2];
          }
        }
        if (nm) prefix_resolve_all(out, start, offs, dsts, nm);
      }
    }
  }
}

static int num_field(const jval *o, const char *k, double *v) {
  const jval *x = json_get(o, k);
  if (!x || x->t != J_NUM) return 0;
  *v = x->num;
  return 1;
}

static int str_is_zero_effort(const char *s, size_t n) {
  while (n && (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r')) s++, n--;
  while (n && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\n' || s[n - 1] == '\r')) n--;
  if (n == 4 && !memcmp(s, "none", 4)) return 1;
  if (!n || n >= 32) return 0;
  char buf[32], *end = NULL;
  memcpy(buf, s, n), buf[n] = 0;
  const double v = strtod(buf, &end);
  return end == buf + n && v == 0.0;
}

static void handle_chat(int fd, const request *r) {
  char err[512];
  jval *root = json_parse(r->body, r->body_n, err, sizeof err);
  if (!root || root->t != J_OBJ) {
    if (!root) http_error(fd, 400, "invalid_request_error", "%s", err);
    else http_error(fd, 400, "invalid_request_error", "the body must be a JSON object");
    json_free(root);
    close(fd);
    return;
  }
  job *j = calloc(1, sizeof *j);
  j->fd = fd;
  j->req = root; /* the job frees it */
  double v;
  j->gp.seed = num_field(root, "seed", &v) ? (uint64_t)(int64_t)v : (uint64_t)time(NULL) ^ ((uint64_t)fd << 32);
  j->gp.draft = O.draft, j->gp.pmin = O.pmin, j->gp.chunk = O.chunk, j->gp.cache = O.cache;
  j->gp.stops = STOP_IDS, j->gp.n_stops = 8;
  j->gp.dry_mult = O.dry_mult, j->gp.dry_base = O.dry_base, j->gp.dry_allowed = O.dry_allowed, j->gp.dry_reset = THINK_END_ID;
  if (num_field(root, "dry_multiplier", &v)) j->gp.dry_mult = v > 0 ? (float)v : 0.0f;
  if (num_field(root, "dry_base", &v) && v >= 1.0) j->gp.dry_base = (float)v;
  if (num_field(root, "dry_allowed_length", &v) && v >= 1.0) j->gp.dry_allowed = v > 4096.0 ? 4096u : (uint32_t)v;
  if (num_field(root, "n", &v) && v != 1) {
    http_error(fd, 400, "invalid_request_error", "only n = 1 is supported");
    goto bad;
  }
  const jval *st = json_get(root, "stream");
  j->stream = st && st->t == J_TRUE;
  const jval *so = json_get(root, "stream_options"), *iu = json_get(so, "include_usage");
  j->usage = iu && iu->t == J_TRUE;
  int think = O.think;
  const jval *kw = json_get(root, "chat_template_kwargs"), *et = json_get(kw, "enable_thinking");
  if (et && (et->t == J_TRUE || et->t == J_FALSE)) think = et->t == J_TRUE;
  const char *effort = O.effort;
  const jval *ef = json_get(root, "reasoning_effort");
  if (!ef || ef->t == J_NULL) ef = json_get(kw, "reasoning_effort");
  if (ef && ef->t == J_STR) effort = ef->str;
  const char *ink_eff_str = NULL;
  int ink_think = think;
  if (INKLING2) {
    if (ef && ef->t == J_STR) ink_think = !str_is_zero_effort(ef->str, ef->slen);
    else if (ef && ef->t == J_NUM) ink_think = ef->num != 0.0;
    else if (ef && ef->t != J_NULL) ink_think = 1;
    else if (!think) ink_eff_str = "none", ink_think = 0;
    else if (EFFORT_SET) ink_eff_str = O.effort, ink_think = !str_is_zero_effort(O.effort, strlen(O.effort));
    else ink_think = 1;
  }
  j->think = FLINT || MIMO || INKLING2 ? 0 : think, j->auto_think = FLINT || (MIMO && think); /* MiMo: the model opens <think> */
  if (j->think && THINK_END_ID >= 0) {
    j->gp.reasoning_budget = O.reasoning_budget;
    if (num_field(root, "reasoning_budget", &v) && v >= 0 && v <= O.ctx) j->gp.reasoning_budget = (uint32_t)v;
  }
  /* sampling: the request's values, else the defaults for its thinking mode (flint decides itself: thinking) */
  const sampling *sd = &SAMPLING[INKLING2 ? ink_think : (FLINT || think)];
  j->gp.temp = sd->temp, j->gp.top_p = sd->top_p, j->gp.top_k = sd->top_k;
  int req_temp = num_field(root, "temperature", &v);
  if (req_temp) j->gp.temp = (float)v;
  if (num_field(root, "top_p", &v)) j->gp.top_p = (float)v;
  if (num_field(root, "top_k", &v)) j->gp.top_k = v < 0 ? 0 : (uint32_t)v;
  if (j->gp.top_p <= 0.0f || j->gp.top_p > 1.0f) j->gp.top_p = 1.0f;

  /* tools: rendered into the prompt, and the answer's <tool_call> blocks become tool_calls */
  const jval *tools = json_get(root, "tools"), *tc = json_get(root, "tool_choice");
  if (!tools || tools->t != J_ARR || !tools->kid || (tc && tc->t == J_STR && !strcmp(tc->str, "none"))) tools = NULL;
  j->tools = tools;
  const char *dtp = getenv("IE_DRY_TOOL_PAUSE");
  if (tools && (!dtp || strcmp(dtp, "0") != 0) && TOOL_CALL_OPEN_ID >= 0 && TOOL_CALL_CLOSE_ID >= 0) {
    j->gp.dry_pause_open_plus_one = (uint32_t)TOOL_CALL_OPEN_ID + 1u;
    j->gp.dry_pause_close_plus_one = (uint32_t)TOOL_CALL_CLOSE_ID + 1u;
    j->gp.dry_pause_after_reset = !INKLING2 && j->think && THINK_END_ID >= 0;
  }
  /* a direct (non-thinking) tool request's own temperature (--tool-temp) when the request does not set one;
   * when thinking is on (e.g. coding agents like omp), keep the thinking temperature and DRY loop penalty */
  if (tools && !(INKLING2 ? ink_think : think) && !FLINT && !req_temp && O.tool_temp >= 0.0f) j->gp.temp = O.tool_temp;
  chat_prompt cp;
  if (FLINT      ? !chat_render_flint(json_get(root, "messages"), tools, &cp, err, sizeof err)
      : MIMO     ? !chat_render_mimo(json_get(root, "messages"), tools, think, &cp, err, sizeof err)
      : INKLING2 ? !chat_render_inkling2(json_get(root, "messages"), tools, ef, ink_eff_str, VIS != NULL, &cp, err, sizeof err)
                 : !chat_render_vis(json_get(root, "messages"), tools, think, effort, VIS != NULL, &cp, err, sizeof err)) {
    http_error(fd, 400, "invalid_request_error", "%s", err);
    goto bad;
  }
  if (cp.n_imgs) {
    for (size_t k = 0; k < cp.n_imgs; k++) {
      if (!vision_validate_url(cp.imgs[k].url, cp.imgs[k].len, err, sizeof err)) {
        chat_prompt_free(&cp);
        http_error(fd, 400, "invalid_request_error", "image %zu: %s", k + 1, err);
        goto bad;
      }
    }
    j->img_urls = malloc(cp.n_imgs * sizeof *j->img_urls);
    j->img_slot = calloc(cp.n_imgs, sizeof *j->img_slot);
    if (!j->img_urls || !j->img_slot) ie_die("serve: out of memory");
    memcpy(j->img_urls, cp.imgs, cp.n_imgs * sizeof *j->img_urls);
    j->n_img_urls = cp.n_imgs;
  }
  const char *oc_prefix = getenv("IE_OPENCODE_PREFIX");
  const char *omp_prefix = getenv("IE_OMP_PREFIX");
  int agent = AGENT_NONE;
  if (O.cache && !MIMO && !FLINT && !INKLING2) {
    if (r->agent == AGENT_OPENCODE && tools && oc_prefix && !strcmp(oc_prefix, "1")) {
      agent = AGENT_OPENCODE, j->gp.prefix_client = "OpenCode";
    } else if (r->agent == AGENT_OMP && (!omp_prefix || !strcmp(omp_prefix, "1"))) {
      agent = AGENT_OMP, j->gp.prefix_client = "OMP";
    }
  }
  prompt_tokens(&cp, &j->prompt, &j->gp, agent, j->img_slot);
  chat_prompt_free(&cp);
  /* the room in the context: n_prompt + max_new + spec_room(draft) <= ctx */
  if (getenv("IE_SERVE_DEBUG")) { /* the prompt ids, to compare with llama-tokenize */
    sbuf b = {0};
    for (uint32_t i = 0; i < j->prompt.n; i++) sb_printf(&b, "%s%u", i ? " " : "", j->prompt.v[i]);
    fprintf(stderr, "prompt ids: %s\n", b.p ? b.p : "");
    sb_free(&b);
  }
  const uint32_t used = j->prompt.n + spec_room(O.draft);
  if (used >= O.ctx) {
    http_error(fd, 400, "invalid_request_error", "the prompt has %u tokens; the context is %u (--ctx)", j->prompt.n, O.ctx);
    goto bad;
  }
  uint32_t room = O.ctx - used;
  if ((num_field(root, "max_completion_tokens", &v) || num_field(root, "max_tokens", &v)) && v >= 1) {
    j->req_max_tok = v > (double)O.ctx ? O.ctx : (uint32_t)v;
    if (j->req_max_tok < room) room = j->req_max_tok;
  }
  j->gp.max_new = room;

  pthread_mutex_lock(&QM);
  if (QN >= O.queue_max) {
    pthread_mutex_unlock(&QM);
    http_error(fd, 503, "server_busy", "%d requests are waiting; try again later", O.queue_max);
    j->fd = -1;
    job_free(j);
    close(fd);
    return;
  }
  j->rid = ++NEXT_RID;
  snprintf(j->id, sizeof j->id, "chatcmpl-%lx%lu", (unsigned long)T_START, j->rid);
  j->created = (long)time(NULL);
  if (QT) QT->next = j;
  else QH = j;
  QT = j, QN++;
  const int waiting = QN - 1 + BUSY;
  pthread_cond_signal(&QC);
  pthread_mutex_unlock(&QM);
  char dry[96] = "";
  if (j->gp.dry_mult > 0.0f && j->gp.temp > 0.0f)
    snprintf(dry, sizeof dry, ", loop penalty %g/%g/%u%s", j->gp.dry_mult, j->gp.dry_base, j->gp.dry_allowed,
             j->gp.dry_pause_open_plus_one ? " (paused in tool_call)" : "");
  fprintf(stderr, "[%lu] queued: %u prompt tokens, max %u output tokens, %s%s, temp %g top_p %g top_k %u%s, %d before it\n", j->rid,
          j->prompt.n, j->gp.max_new, j->stream ? "stream" : "no stream", j->tools ? ", tools" : "", j->gp.temp, j->gp.top_p,
          j->gp.top_k, dry, waiting);
  return; /* the engine thread writes the response and closes fd */
bad:
  j->fd = -1;
  job_free(j);
  close(fd);
}

static void *conn_main(void *arg) {
  const int fd = (int)(intptr_t)arg;
  request r;
  const int rc = read_request(fd, &r);
  if (rc) {
    if (rc > 0) http_error(fd, rc, "invalid_request_error", "%s", status_text(rc));
    close(fd);
    return NULL;
  }
  const int is_v1 = !strncmp(r.path, "/v1/", 4);
  const char *p = is_v1 ? r.path + 3 : r.path; /* "/models" etc. */
  if (!strcmp(r.method, "OPTIONS")) {
    http_reply(fd, 204, "text/plain", "", 0);
  } else if (!strcmp(p, "/health")) {
    const char *b = "{\"status\":\"ok\"}";
    http_reply(fd, 200, "application/json", b, strlen(b));
  } else if (O.api_key && (strncmp(r.auth, "Bearer ", 7) || !same_secret(r.auth + 7, O.api_key))) {
    http_error(fd, 401, "authentication_error", "missing or wrong API key (Authorization: Bearer KEY)");
  } else if (!strcmp(p, "/models")) {
    if (strcmp(r.method, "GET")) http_error(fd, 405, "invalid_request_error", "use GET");
    else {
      sbuf b = {0};
      sb_puts(&b, "{\"object\":\"list\",\"data\":[{\"id\":");
      sb_json_str(&b, MODEL_ID, strlen(MODEL_ID));
      sb_printf(&b, ",\"object\":\"model\",\"created\":%ld,\"owned_by\":\"amd-infer\",\"context_length\":%u}]}", (long)T_START, O.ctx);
      http_reply(fd, 200, "application/json", b.p, b.n);
      sb_free(&b);
    }
  } else if (!strcmp(p, "/chat/completions")) {
    if (strcmp(r.method, "POST")) http_error(fd, 405, "invalid_request_error", "use POST");
    else {
      handle_chat(fd, &r); /* owns fd from here */
      free(r.body);
      return NULL;
    }
  } else {
    http_error(fd, 404, "invalid_request_error", "no endpoint %s %s", r.method, r.path);
  }
  free(r.body);
  close(fd);
  return NULL;
}

int main(int argc, char **argv) {
  int show_template = 0;
  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
#define ARG (i + 1 < argc ? argv[++i] : (usage(), ""))
    if (!strcmp(a, "--backend")) O.backend = ARG;
    else if (!strcmp(a, "--hsaco")) O.hsaco = ARG;
    else if (!strcmp(a, "--host")) O.host = ARG;
    else if (!strcmp(a, "--port")) O.port = atoi(ARG);
    else if (!strcmp(a, "--api-key")) O.api_key = ARG;
    else if (!strcmp(a, "--name")) O.name = ARG;
    else if (!strcmp(a, "--lora")) O.lora = ARG;
    else if (!strcmp(a, "--mmproj")) O.mmproj = ARG;
    else if (!strcmp(a, "--ctx")) O.ctx = (uint32_t)atoi(ARG);
    else if (!strcmp(a, "--draft")) O.draft = (uint32_t)atoi(ARG);
    else if (!strcmp(a, "--draft-pmin")) O.pmin = (float)atof(ARG);
    else if (!strcmp(a, "--mtp")) O.mtp = ARG;
    else if (!strcmp(a, "--dflash")) O.dflash = ARG;
    else if (!strcmp(a, "--dspark")) O.dspark = ARG;
    else if (!strcmp(a, "--chunk")) O.chunk = (uint32_t)atoi(ARG);
    else if (!strcmp(a, "--temp")) O.temp = (float)atof(ARG);
    else if (!strcmp(a, "--think-temp")) O.think_temp = (float)atof(ARG);
    else if (!strcmp(a, "--dry-multiplier")) O.dry_mult = (float)atof(ARG);
    else if (!strcmp(a, "--dry-base")) O.dry_base = (float)atof(ARG);
    else if (!strcmp(a, "--dry-allowed-length")) O.dry_allowed = (uint32_t)atoi(ARG);
    else if (!strcmp(a, "--tool-temp")) O.tool_temp = (float)atof(ARG);
    else if (!strcmp(a, "--top-k")) O.top_k = (uint32_t)atoi(ARG);
    else if (!strcmp(a, "--top-p")) O.top_p = (float)atof(ARG);
    else if (!strcmp(a, "--no-think")) O.think = 0;
    else if (!strcmp(a, "--effort")) O.effort = ARG, EFFORT_SET = 1;
    else if (!strcmp(a, "--reasoning-budget")) O.reasoning_budget = (uint32_t)strtoul(ARG, NULL, 10);
    else if (!strcmp(a, "--no-cache")) O.cache = 0;
    else if (!strcmp(a, "--think-in-content")) O.raw_think = 1;
    else if (!strcmp(a, "--queue")) O.queue_max = atoi(ARG);
    else if (!strcmp(a, "--show-template")) show_template = 1;
    else if (a[0] != '-' && !O.path) O.path = a;
    else usage();
#undef ARG
  }
  if (!O.path) usage();
  if (!O.api_key) O.api_key = getenv("IE_API_KEY");
  if (O.api_key && !O.api_key[0]) O.api_key = NULL;
  if (show_template) {
    gguf_open(&G, O.path);
    char *t = gguf_get_str(&G, "tokenizer.chat_template");
    printf("%s\n", t ? t : "(no tokenizer.chat_template)");
    return 0;
  }
  const int gpu = !strcmp(O.backend, "gpu");
  if (!gpu && strcmp(O.backend, "cpu")) usage();
  if (O.draft != DRAFT_AUTO && (O.draft > MTP_DRAFT_MAX || (O.draft && !gpu))) ie_die("--draft: 1 to 8, GPU only");
  if (O.reasoning_budget > O.ctx) ie_die("--reasoning-budget must fit the context");
  if (O.dflash && !O.draft) ie_die("--dflash needs --draft");
  if (O.dspark && O.dflash) ie_die("--dspark and --dflash are mutually exclusive");
  if (O.chunk > 1024) ie_die("--chunk: 1 to 1024 (0: by the model)");
  if (O.ctx < 16) ie_die("--ctx: at least 16");
  if (O.queue_max < 1) O.queue_max = 1;
  ie_net_init();
  T_START = time(NULL);
  { /* paced streaming: the sender thread; on Windows 1 ms timers (the default 15.6 ms is coarser than a token) */
    const char *e = getenv("IE_STREAM_PACE");
    PACE = !(e && *e) || atoi(e) != 0;
    if (PACE) {
#ifdef _WIN32
      HMODULE wm = LoadLibraryA("winmm.dll");
      typedef unsigned(__stdcall * tbp_fn)(unsigned);
      tbp_fn tbp = wm ? (tbp_fn)(void (*)(void))GetProcAddress(wm, "timeBeginPeriod") : NULL;
      if (tbp) tbp(1);
#endif
      pthread_t pt;
      pthread_create(&pt, NULL, pace_main, NULL);
      fprintf(stderr, "ie-serve: streaming one token per chunk, spread over each step (IE_STREAM_PACE=0: one chunk per step)\n");
    }
  }

  /* bind first: a busy port is an error before the long model load */
  struct sockaddr_in sa = {0};
  sa.sin_family = AF_INET;
  sa.sin_port = htons((uint16_t)O.port);
  if (inet_pton(AF_INET, O.host, &sa.sin_addr) != 1) ie_die("--host: not an IPv4 address: %s", O.host);
  const int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
  ie_setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  if (bind(ls, (struct sockaddr *)&sa, sizeof sa)) ie_die("bind %s:%d: %s", O.host, O.port, strerror(errno));
  if (!O.api_key && strcmp(O.host, "127.0.0.1"))
    fprintf(stderr, "warning: listening on %s without --api-key: every device that can reach this address can use the model\n", O.host);

  pthread_t et;
  pthread_create(&et, NULL, engine_main, NULL);
  pthread_mutex_lock(&QM);
  while (!IS_READY) pthread_cond_wait(&READY, &QM);
  pthread_mutex_unlock(&QM);
  if (listen(ls, 64)) ie_die("listen: %s", strerror(errno));
  fprintf(stderr, "ie-serve: http://%s:%d/v1 (%s)\n", O.host, O.port, O.api_key ? "API key required" : "no API key");

  for (;;) {
    const int fd = accept(ls, NULL, NULL);
    if (fd < 0) {
      if (errno != EINTR) perror("accept");
      continue;
    }
    ie_sock_timeout(fd, 30); /* a client that stops reading or writing for 30 s is dropped */
    ie_setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    pthread_t t;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    if (pthread_create(&t, &at, conn_main, (void *)(intptr_t)fd)) close(fd);
    pthread_attr_destroy(&at);
  }
}
