/* backend.h -- the two backends run the op list of graph.h per token. */
#ifndef IE_BACKEND_H
#define IE_BACKEND_H

#include "model.h"

/* snapshot slots of backend.snap (the prompt cache checkpoints, src/gen.h) */
#define IE_SNAP_MAX 64
typedef struct backend backend;
struct backend {
  /* One decode step: token tok at position pos. Writes vocab logits (if
   * not NULL) and returns the argmax token. ms: time of the step. */
  uint32_t (*step)(backend *b, uint32_t tok, uint32_t pos, float *logits, double *ms);
  void (*close)(backend *b);
  /* GPU only (NULL on the CPU). Run section sec of the op list (0: the
   * model, 1: the MTP head, 4: the MTP layer without its output head;
   * DFlash: 2 the drafter context, 3 the drafter block; out unused for
   * 2 to 4) for T tokens toks at positions pos .. pos + T - 1. The linear attention state after each token t is kept (or, if
   * last_only, after the last token only) until accept. out: the T argmax
   * tokens. */
  void (*run)(backend *b, int sec, const uint32_t *toks, uint32_t T, uint32_t pos, int last_only, uint32_t *out,
              float *prob); /* prob (or NULL): the softmax probability of each argmax token */
  /* The tokens 0 .. k of the last model run are final: later runs continue
   * from the linear attention state after token k. (A one-token step needs
   * no accept.) */
  void (*accept)(backend *b, uint32_t k);
  /* Copy n token rows of buffer id (from row r0) to the host: dst gets n *
   * buf.stride bytes (rows buf.stride apart). */
  void (*read_rows)(backend *b, int id, uint32_t r0, uint32_t n, float *dst);
  /* Copy n token rows of buffer src (from row r0) to buffer dst (from row
   * d0); src < 0: write zeros. */
  void (*copy_rows)(backend *b, int dst, uint32_t d0, int src, uint32_t r0, uint32_t n);
  /* GPU only (NULL on the CPU): the K largest values of n rows of buffer
   * id (from row r0), each row's first nv values (the vocabulary): val and
   * idx get n * K entries, per row value descending, equal values smaller
   * index first. Returns 0 (and does nothing) if K is too large. */
  int (*topk)(backend *b, int id, uint32_t r0, uint32_t n, uint32_t nv, uint32_t K, float *val, uint32_t *idx);
  /* GPU: set by the caller around a verify run (run, sec 0) whose rows it will sample from the top K of K logits
   * (topk, buffer g->logits, rows 0 .. T - 1): the run then takes that top K itself, behind the output matrix, and
   * the next topk call for those rows returns it without another wait. The run's argmax tokens (out) are not
   * computed then (out is zero). 0 (default): the run computes the argmax and nothing else. */
  uint32_t sample_k;
  /* GPU: the most tokens of one verify run (the linear attention states
   * kept: 8 or 16); 0 on the CPU */
  uint32_t verify_max;
  /* The prompt cache (ie-serve). save = 1: copy the state that is not kept
   * per position (the linear attention state of the current slot and the
   * conv ring) and row 0 of buffer hbuf (-1: none; the MTP h input) to
   * snapshot k (0 .. IE_SNAP_MAX - 1); save = 0: copy snapshot k back. The KV caches are
   * not copied: runs after the snapshot write only higher positions. NULL if
   * the backend has no snapshots. */
  void (*snap)(backend *b, int save, int k, int hbuf, uint32_t pos);
  /* The disk prompt cache (gen.c, IE_DISK_CACHE). snap_io: out = 1: copy snapshot k (saved before, a checkpoint after the
   * first pos tokens) to buf; out = 0: fill snapshot k from buf. Returns the bytes of a snapshot (buf NULL: only that; 0 for
   * a model without linear attention); SIZE_MAX if the backend cannot (the sliding window layers' rings are not written
   * out). The snapshot in buf is the linear attention states (snap_f32b bytes of f32, first), then the d_conv - 1 conv
   * ring slots of the positions pos - d_conv + 1 .. pos - 1 (the only ones a run from position pos reads before it writes
   * them; the other slots read as zero after an out = 0), then the h row. kv_io: out = 1: copy the K and V rows of
   * positions p0 .. p0 + n - 1 of every attention layer to buf (per layer its K rows, then its V rows); out = 0: write them
   * from buf. kv_rowb: the bytes of one position's rows (all layers, K and V). NULL / 0 on the CPU.
   * Layers with a ring cache (inkling2's sliding window visits) are not in kv_io / kv_rowb: snap_io carries their whole
   * rings, as gpu_snap does. */
  size_t (*snap_io)(backend *b, int k, int out, void *buf, uint32_t pos);
  size_t snap_f32b;
  void (*kv_io)(backend *b, int out, uint32_t p0, uint32_t n, void *buf);
  void (*kv_trim)(backend *b, uint32_t keep_pos); /* GPU: unmap high VMM KV chunks past keep_pos (IE_KV_SHRINK=0: off) */
  size_t kv_rowb;
  uint32_t kv_fmt_tag; /* non-zero when global KV uses a specialized format/rotation (hashed into disk cache sig) */
  int snap_in_ram; /* 1 when snapshots live in host RAM (IE_SNAP_HOST!=0): disk 'e' reads need no hipHostMalloc */
  /* Probabilistic DSpark proposal hook (NULL when greedy temp <= 0 -> 0-bit change). */
  struct ie_ds_samp *ds_samp;
  /* R30 MicroSpec dynamic prompt union + online fallback certificate for DSpark shortlist (NULL if inactive). */
  void (*ds_set_shortlist)(backend *b, const uint32_t *prompt, uint32_t n_prompt);
  void (*ds_note_tokens)(backend *b, const uint32_t *toks, uint32_t n_toks);
  /* Host buffers for the disk cache: host_alloc returns n bytes aligned to 4096 (pinned for the GPU, so copies to and from
   * the device run at full speed), or NULL; host_free frees one. NULL members: use plain aligned memory. */
  void *(*host_alloc)(backend *b, size_t n);
  void (*host_free)(backend *b, void *p);
  /* 1: prompt chunks compute the logits of every token (perplexity); 0
   * (default): only of the last token */
  int all_logits;
  /* GPU (NULL on the CPU): a tree verify. tree_ok: 1 if a run (sec 0) of T nodes at position pos can be a tree.
   * The caller sets tree_par (par[t] < t: the parent of node t, par[0] unused) before that run; its tokens are the
   * nodes, each at position pos + its depth, seeing its ancestors only. accept_tree then keeps the nodes path[0] = 0,
   * path[1 .. k] (each the parent of the next), as accept(k) for a chain, and clears tree_par. */
  const uint8_t *tree_par;
  int (*tree_ok)(backend *b, uint32_t pos, uint32_t T);
  void (*accept_tree)(backend *b, const uint8_t *path, uint32_t k);
  /* Vision / external prefill embeddings (NULL for text-only chunks): T host pointers;
   * non-NULL entry t supplies m->dim f32 values directly into OP_EMBED's output buffer for token t. */
  const float *const *ext_rows;
  /* Override precomputed RoPE cos/sin rows [pos0 .. pos0 + n) (e.g. 3D IMRoPE for images). */
  void (*set_rope)(backend *b, uint32_t pos0, uint32_t n, const float *cos_tab, const float *sin_tab);
};

typedef struct ie_ds_samp {
  int on, used;
  float temp, top_p;
  uint32_t top_k;          /* <= 32 */
  uint64_t rng;            /* xorshift64* state, advanced per draft step */
  /* R29 ForeSpark online-calibrated survival controls (active when fs_on != 0) */
  int fs_on;
  float surv_a, surv_b, surv_pmin, drop0_h;
  uint32_t out_n[16];      /* support size of q_j (0 if stopped) */
  uint32_t out_id[16][32]; /* token ids */
  float out_q[16][32];     /* normalized probabilities */
  float out_entropy[16];   /* nucleus entropy H_j written by device kernel */
} ie_ds_samp;

/* Low-level HIP helpers shared by src/vision.c (implemented in src/hip.c). */
int gpu_vis_ready(void);
void *gpu_vis_alloc(size_t bytes);
void gpu_vis_free(void *p);
void gpu_vis_h2d(void *dst, const void *src, size_t bytes);
void gpu_vis_d2h(void *dst, const void *src, size_t bytes);
void *gpu_vis_host_alloc(size_t bytes);
void gpu_vis_host_free(void *p);
void *gpu_vis_stream_create(void);
void gpu_vis_stream_destroy(void *st);
void *gpu_vis_event_create(void);
void gpu_vis_event_destroy(void *ev);
void gpu_vis_h2d_async(void *dst, const void *src, size_t bytes, void *st);
void gpu_vis_d2h_async(void *dst, const void *src, size_t bytes, void *st);
void gpu_vis_event_record(void *ev, void *st);
void gpu_vis_event_sync(void *ev);
void gpu_vis_stream_wait(void *st, void *ev);
void gpu_vis_stream_sync(void *st);
void *gpu_vis_load_mod(const char *path);
void *gpu_vis_get_fn(void *mod, const char *name);
void gpu_vis_unload_mod(void *mod);
void gpu_vis_launch(void *fn, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, void **args);
void gpu_vis_sync(void);
void gpu_vis_mem_info(size_t *free_b, size_t *total_b);
int gpu_vis_dxgi_info(uint64_t *usage_out, uint64_t *budget_out);

/* CPU: the same algorithm and the same work split as the GPU kernels. */
backend *cpu_open(const model *m, const graph *g);
/* GPU (HIP, gfx1201): kernels from the hsaco file; exits with a message
 * if libamdhip64.so or a device is missing. */
backend *gpu_open(const model *m, const graph *g, const char *hsaco);
void gpu_trace_flush(void);
/* Before model_load: start HIP and set ie_mat_sink, so that each matrix
 * goes to the GPU as soon as it is repacked (host memory: one layer). */
void gpu_stream_init(void);
/* R50 (H09): clamp --chunk > 512 to 768/512 if free VRAM lacks headroom for larger arena. */
uint32_t gpu_chunk_clamp(uint32_t chunk, const model *m, uint32_t n_ctx);
/* ie-lab (GPU): write part (a host matrix of w's kind holding rows r0 .. r0 + part->rows - 1 of w) into w's device
 * arrays. 0 (nothing written) if w was not streamed or its kind is not row-contiguous (Q4_0, Q8_0 and F32 are). */
int gpu_patch_mat(backend *b, const mat *w, uint32_t r0, const mat *part);
/* ie-lab (GPU): copy v->f into every device copy the ops use; returns the copies written. */
int gpu_patch_vec(backend *b, const vec *v);


/* The CPU kernels, also used by tests. */
void cpu_quant_q8(const float *x, uint8_t *q, uint32_t nb);
void cpu_gemv_q4(const mat *w, const uint8_t *xq, float *y);
void cpu_gemv_q4k(const mat *w, const uint8_t *xq, float *y);
void cpu_gemv_q6k(const mat *w, const uint8_t *xq, float *y);
void cpu_gemv_q8(const mat *w, const uint8_t *xq, float *y);
void cpu_gemv_q5k(const mat *w, const uint8_t *xq, float *y);
void cpu_gemv_iq4xs(const mat *w, const uint8_t *xq, float *y);

#endif
