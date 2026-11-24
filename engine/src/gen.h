/* gen.h -- generation on a backend: sampling, the prompt in chunks
 * (prefill), speculative decoding with the MTP head, and gen_run (one
 * request, tokens given to a callback as they are made). ie-run (main.c)
 * and ie-serve (serve.c) use it. */
#ifndef IE_GEN_H
#define IE_GEN_H

#include "backend.h"
#include "model.h"

/* ---- sampling (temperature, top-k, top-p) ----
 * temp = 0: greedy (argmax). Otherwise p ~ softmax(logits / temp) over the
 * top_k largest logits (top_k = 0: all; but top_p < 1 with top_k = 0 keeps
 * the 1000 largest: the nucleus is cut from those), then the smallest set
 * of the largest whose probability reaches top_p. */
typedef struct {
  float temp, top_p;
  uint32_t top_k;
  uint64_t rng;
  uint32_t *idx; /* scratch: candidate ids */
  float *pr;     /* scratch: their weights */
  float *cv;     /* the GPU top K: values (16 rows x 64) */
  uint32_t *ci;  /* and ids */
  /* The loop penalty (DRY style; dry_mult 0: off, the default): a candidate
   * that would continue a span of at least dry_allowed tokens that already
   * occurred in this answer gets dry_mult * dry_base^(match - dry_allowed)
   * subtracted from its logit. Only the output counts (not the prompt), and
   * the token dry_reset (</think>) starts a new span set, so the answer can
   * repeat code from the reasoning. Sampling only (temp > 0); it acts on the
   * top K candidates. dry_begin / dry_commit (gen.c) keep the state. */
  float dry_mult, dry_base;
  uint32_t dry_allowed;
  int64_t dry_reset;
  uint32_t dry_pause_open_plus_one, dry_pause_close_plus_one; /* pause DRY inside [open, close] (e.g. <tool_call>..</tool_call>); 0: off */
  int dry_pause_after_reset;       /* ignore open/close until dry_reset (</think>) has been seen */
  int dry_paused, dry_pause_wait_reset;
  uint32_t *dry_h, dry_n, dry_cap; /* the output tokens so far */
  uint32_t dry_seg;                /* the current span set starts at dry_h[dry_seg] */
  uint64_t *dry_bits;              /* the hashes of the dry_allowed-token spans of the set */
  int32_t *dry_head, *dry_prev;    /* O(1) hash-chain index of span end positions */
  const uint32_t *dry_x;           /* tokens after dry_h (a verify row's accepted drafts), dry_xn of them */
  uint32_t dry_xn;
  uint32_t dry_hits;               /* candidates penalized in this request */
  uint32_t vocab;                  /* upper bound on valid candidate ids (0: 0xFFFFFFFFu sentinel only) */
  uint32_t reasoning_end_plus_one, reasoning_left; /* active only while a budgeted reasoning block is open */
  int reasoning_limited;
} sampler;

/* Perplexity: the sum of -log p(next prompt token) at positions >= first. With ksave (ie-run --kld-save) each scored
 * position's log probabilities (vocab floats) go to that file; with kbase (--kld) they are read back as the reference
 * and kl[i] gets KL(reference || this run) of scored position i, ktop counts the positions whose most likely token is
 * the reference's, knll sums the reference's -log p(next). */
typedef struct {
  int on;
  uint32_t first, n;
  double nll;
  FILE *ksave, *kbase;
  float *klp, *kref, *kl;
  uint32_t ktop;
  double knll;
} ppl_acc;

/* Called with new output tokens, in order. Returns nonzero to end the
 * generation (a stop token, or the client is gone). */
typedef int (*gen_emit)(void *ctx, const uint32_t *toks, uint32_t n);

double now_ms(void);
/* The number of candidates: top_k, or (top_p < 1) the 1000 largest, or all. */
uint32_t sample_k(const sampler *sp, uint32_t V);
/* Sample from n candidates (values v, ids id), value descending. With the
 * loop penalty on, v and id are scratch: penalized and sorted again. */
uint32_t sample_cands(sampler *sp, float *v, uint32_t *id, uint32_t n);
/* The loop penalty: start a request (forget the output), and add output tokens. */
void dry_begin(sampler *sp);
void dry_commit(sampler *sp, const uint32_t *t, uint32_t n);
/* The top K candidates of rows r0 .. r0 + n - 1 of the logits on the GPU:
 * 1 if done, 0 if the backend cannot (then the host selects). */
int gpu_cands(const sampler *sp, backend *b, const graph *g, uint32_t r0, uint32_t n, uint32_t V, float *cv, uint32_t *ci);
/* Sample one token from vocab logits lg (greedy if temp <= 0). */
uint32_t sample(sampler *sp, const float *lg, uint32_t V);
/* The prompt in chunks (GPU). Returns the greedy token after the prompt;
 * *hrow: the row of g->h_out that holds h(n-1). pa may be NULL. */
uint32_t prefill(backend *b, const graph *g, uint32_t vocab, const uint32_t *toks, uint32_t n, uint32_t chunk, int mtp,
                 ppl_acc *pa, uint32_t *hrow);
/* Same as prefill(), starting at KV position pos0 (used by IE_KV_SYNTH_POS). */
uint32_t prefill_at(backend *b, const graph *g, uint32_t vocab, const uint32_t *toks, uint32_t n, uint32_t pos0,
                    uint32_t chunk, int mtp, ppl_acc *pa, uint32_t *hrow);
/* Speculative decoding with the MTP head (GPU). emit (or NULL) gets each
 * output token after the prompt; when it returns nonzero, the generation
 * ends. Returns the number of tokens in out. */
uint32_t spec_verify_max(void);
/* The most MTP drafts per step (before the backend is made): the verify room holds 1 + D tokens. */
void spec_set_draft(uint32_t D);
uint32_t spec_room(uint32_t draft);
uint32_t spec_generate(backend *b, const model *m, const graph *g, uint32_t *out, uint32_t n_prompt, uint32_t total, uint32_t D,
                       int64_t eos, int stop, uint32_t first, uint32_t hrow, sampler *sp, uint32_t V, double *gen_ms,
                       uint32_t *n_steps, uint32_t *n_acc, float pmin, uint32_t *n_drafted, gen_emit emit, void *ectx,
                       const int64_t *stops, uint32_t n_stops, uint32_t *p_done, uint32_t *h_row);
/* ie-lab: the file prefill writes the MTP capture records to (see gen.c mtp_capture; NULL: IE_MTP_CAPTURE or none;
 * the graph needs mtp_r / mtp_a), and a 9 x 9 histogram spec_generate counts its chain steps in (NULL: none):
 * [drafts made * 9 + drafts accepted]++. */
#include <stdio.h>
extern FILE *ie_mtp_cap_file;
extern uint32_t *ie_spec_nk;
/* stops (n_stops ids, or NULL): a draft that is a stop token is not accepted,
 * so the model does not run past the stop token (as in plain decoding); the
 * output is the same. p_done (or NULL): the positions done at the end (the
 * state is after out[0 .. *p_done)); h_row: the row of h_out that holds h(*p_done - 1). */

/* ---- one request (ie-serve) ---- */
typedef struct {
  uint32_t pos;      /* first prompt token index of this image */
  uint32_t n_tok;    /* nx * ny merged vision tokens */
  uint32_t nx, ny;   /* merged spatial grid (width, height) for 3D IMRoPE */
  uint64_t hash;     /* content hash for in-memory KV cache matching */
  const float *embd; /* [n_tok * m->dim] f32 embeddings */
} gen_img;

typedef struct {
  float temp, top_p; /* temp <= 0: greedy */
  uint32_t top_k;
  uint64_t seed;
  uint32_t max_new; /* the most output tokens (>= 1) */
  uint32_t draft;   /* 0: no MTP; 1 .. 7: drafts per step (GPU, the graph built with the MTP head) */
  float pmin;       /* --draft-pmin */
  float dry_mult, dry_base; /* the loop penalty (sampler.dry_*); dry_mult 0: off */
  uint32_t dry_allowed;
  int64_t dry_reset;        /* </think>, or -1 */
  uint32_t dry_pause_open_plus_one, dry_pause_close_plus_one; /* pause DRY inside [open, close] (0: off) */
  int dry_pause_after_reset; /* ignore open/close until dry_reset has been seen */
  uint32_t reasoning_budget; /* 0: no cap on the reasoning block */
  uint32_t chunk;   /* prompt chunk (GPU) */
  int cache;        /* use and update the prompt cache (needs backend.snap) */
  /* where the messages are in the prompt (token positions; 0: unknown): sys_end, the end of the first (system) message
   * (the position of its closing token); last_user, the start of the last user message. gen_run puts checkpoints
   * before the end of a long system message (where agents put per-session text) and at the last user message. */
  uint32_t sys_end, last_user;
  const char *prefix_client; /* enable exact agent-prefix learning, NULL for ordinary chat */
  uint32_t prefix[3]; /* reusable boundaries supplied by the caller, zero when absent */
  uint32_t tools_end; /* optional reusable prefix ending at the rendered tool declarations */
  const int64_t *stops; /* the stop token ids (n_stops; -1 entries are ignored) */
  uint32_t n_stops;
  const gen_img *imgs; /* optional vision embeddings sorted by pos (NULL / 0: text-only) */
  uint32_t n_imgs;
  gen_emit emit;
  void *ctx;
} gen_params;

typedef struct {
  uint32_t n_prompt, n_out; /* n_out: tokens made (also those after a stop) */
  double prefill_ms, gen_ms;
  uint32_t steps, acc, drafted; /* MTP */
  uint32_t cached; /* prompt tokens taken from the prompt cache (not computed again) */
  uint32_t dry_hits; /* candidates the loop penalty lowered */
  int reasoning_limited; /* the engine closed a block that exhausted its budget */
  /* the checkpoint used (GEN_CK_*): -1 none, 0 the end of a prompt (but its last token), 1 the end of an answer,
   * 2 the start of the line where an earlier prompt first differed from the cached tokens, 3 a periodic one, 4 one
   * read from the disk cache, 5 a place where prompts are likely to differ */
  int cache_hit;
  double disk_ms; /* reading it */
  uint32_t common; /* the prompt's first tokens that equal the cached tokens (with the cache) */
  uint32_t n_ck;   /* checkpoints kept after this request */
} gen_stats;

/* GEN_CK_DISK is a cache_hit value only; GEN_CK_MARK: a checkpoint placed where a later prompt is likely to differ (the
 * end of the stable part of a system prompt, the start of the last user message) */
enum { GEN_CK_PROMPT = 0, GEN_CK_ANSWER = 1, GEN_CK_SPLIT = 2, GEN_CK_EVERY = 3, GEN_CK_DISK = 4, GEN_CK_MARK = 5, GEN_CK_PREFIX = 6 };
struct dcache;
struct ie_sam_persist;
#define GEN_CK_MAX IE_SNAP_MAX /* backend snapshot slots (backend.snap k = 0 .. GEN_CK_MAX - 1) */

typedef struct {
  backend *b;
  const model *m;
  const graph *g;
  sampler sp;   /* scratch arrays allocated by gen_init */
  float *logits; /* vocab floats */
  /* The prompt cache. kv_toks[0 .. kv_n): the tokens whose K/V rows the KV caches hold at those positions. A
   * checkpoint is a backend snapshot (slot) of the state that is not kept per position (the linear attention
   * state, the conv ring, the MTP h input) after the first pos tokens of kv_toks. A prompt continues from the
   * latest checkpoint at or before its first token that differs from kv_toks; the checkpoints past that token are
   * dropped. A request adds checkpoints at the start of the line where its prompt first differs (the next prompt
   * with the same start continues from there, also when it differs a few tokens earlier in that line), every
   * IE_CACHE_EVERY tokens, at the end of the prompt (but its last token) and at the end of the answer, and (MARK) where
   * a later prompt is likely to differ: before the end of a long system message (IE_CACHE_MARKS distances, default
   * 256,1024,4096, and the distances where earlier system prompts differed; IE_CACHE_MARK_GAP, default 128, is the least
   * distance between two checkpoints made this way) and at the start of the last user message. At most IE_CACHE_CKPT are kept (default 16; about 160 MB of host memory each for
   * Saluki 27B); a full cache drops the least useful (periodic ones first, then the least recently used). */
  uint32_t *kv_toks, kv_n, kv_cap;
  /* disk: the checkpoint is in the disk cache; h: its snapshot holds the MTP h input */
  struct { uint32_t pos, used; int slot, kind; uint8_t disk, h; } ck[GEN_CK_MAX];
  uint32_t n_ck, tick, req_tick; /* req_tick: the first tick of the request now running */
  uint8_t *nl; /* vocab entries: 1 if the token's text holds a newline (where lines start) */
  struct dcache *dc; /* the disk prompt cache, or NULL */
  void (*disk_kick)(void); /* (or NULL) called when checkpoints were queued for the disk thread (ie-serve: it starts writing) */
  /* the distances (sys_end minus the position) of the places where a system prompt first differed from the one before;
   * checkpoints go there for later system prompts (persisted in the disk cache directory) */
  uint32_t pol_dist[16], pol_n;
  uint8_t pol_loaded;
  /* Vision / 3D IMRoPE state: custom RoPE table rows uploaded for an image prompt, and cached image spans. */
  uint32_t rope_custom_n;
  gen_img *kv_imgs;
  uint32_t kv_n_imgs;
  /* R71 (H29): persistent 2-tier 256K prompt-suffix SAM (<= 22 MiB host RAM, 0 B VRAM) */
  struct ie_sam_persist *sam_p;
} gen_ctx;

void gen_init(gen_ctx *c, backend *b, const model *m, const graph *g);
void gen_free(gen_ctx *c);
/* The disk prompt cache (ie-serve, IE_DISK_CACHE; as DeepSeek's context caching on disk): the prompt cache's
 * checkpoints go to files in dir, each with the K/V rows of the positions before it (in blocks of DC_BLOCK positions
 * that the checkpoints of one conversation share). A prompt continues from the longest one that holds its first
 * tokens when that is further than the checkpoints in memory: after a restart, or after other prompts replaced its
 * tokens in the KV cache. sig names what the files depend on (model files, cache format); files of another sig are
 * not read. Over limit_gb (0: no limit) the least recently used checkpoints (or other sigs' files) are deleted, with
 * the blocks no other checkpoint needs. 0 (and no disk cache) if the backend cannot (CPU, sliding window layers,
 * DFlash). lock (or NULL: one thread): lock(1) / lock(0) around the disk cache's lists, when gen_disk_write runs on
 * another thread. */
int gen_disk_open(gen_ctx *c, const char *dir, const char *sig, double limit_gb, void (*lock)(int on));
/* The policy: which checkpoints go to the disk cache and when. Checkpoints are written back, not after each request:
 * (a) when a request leaves the cached conversation (a different one: its checkpoints past the common start are about
 * to be dropped), (b) when idle (gen_disk_idle), (c) when a full cache drops one to reuse its slot. Admitted: the
 * places where prompts differ (kinds SPLIT, MARK), the latest PROMPT and ANSWER ends, and the deepest periodic one
 * below the first of those (a fallback); not the other periodic ones. IE_DISK_ADMIT=all admits every checkpoint of
 * at least IE_DISK_MIN tokens (the write-after-each-request policy of before). */
/* Queue the admitted checkpoints that are not in the disk cache yet: copies of their K/V rows and snapshots in host
 * memory (the engine thread). Without a lock it writes them too. Returns the number queued (the caller kicks the
 * disk thread when it is not 0). gen_disk_save: right after a request (ie-serve with IE_DISK_IDLE_S=0); gen_disk_idle:
 * when no request has come for a while. */
uint32_t gen_disk_save(gen_ctx *c);
uint32_t gen_disk_idle(gen_ctx *c);
/* The admitted checkpoints not queued yet (0 without a disk cache): ie-serve waits for the idle time while it is not 0. */
uint32_t gen_disk_pending(const gen_ctx *c);
/* Write the queued files (ie-serve: its disk thread, after each gen_disk_save). */
void gen_disk_write(gen_ctx *c);
/* Wave 15 R83 (H50): warm unwarmed .iec blocks from disk into Host-RAM Radix L2 up to max_bytes.
 * Returns > 0 if work was done or more unwarmed blocks remain, 0 when done, disabled, or c == NULL. */
int dc_prewarm_step(gen_ctx *c, size_t max_bytes);
/* Run the prompt (positions 0 .. n_prompt - 1) and generate up to
 * p->max_new tokens. With p->cache, the prompt continues from the latest
 * checkpoint inside its first tokens that equal the cached ones (gen_ctx;
 * the result is close to, but not bitwise the same as, the full prompt:
 * the chunks differ). The caller checks that n_prompt + max_new
 * + spec_room(draft) <= the graph's n_ctx. */
void gen_run(gen_ctx *c, const uint32_t *prompt, uint32_t n_prompt, const gen_params *p, gen_stats *st);

#endif
