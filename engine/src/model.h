/* model.h -- Llama / Qwen2 models from GGUF, and the op list of one decode
 * step. Both backends (cpu.c, hip.c) run the same op list on the same
 * arena offsets. */
#ifndef IE_MODEL_H
#define IE_MODEL_H

#include <stdint.h>

#include "gguf.h"

enum { ARCH_LLAMA = 0, ARCH_QWEN2 = 1, ARCH_QWEN35 = 2, ARCH_FLINT = 3, ARCH_RICURSIVE = 4, ARCH_INKLING2 = 5 };
enum { ROPE_NORM = 0, ROPE_NEOX = 1 }; /* NORM: pairs (2i, 2i+1); NEOX: (i, i + hd/2) */

/* A weight matrix: rows x cols, y = W x. */
enum { MAT_Q4 = 0, MAT_F32 = 1, MAT_Q8 = 2, MAT_Q4K = 3, MAT_Q6K = 4, MAT_Q5K = 5, MAT_IQ4XS = 6,
       MAT_RAWQ = 7, /* raw GGUF IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ1_S / IQ1_M / Q2_K, see kernels/ie_iqraw.inc */
       MAT_IQ4NL = 8 }; /* Q4_0 nibble layout with kvalues_iq4nl lookup table */
typedef struct {
  int kind;
  uint32_t rows, cols, nb; /* nb = cols / 32 (Q4 and Q8) */
  /* MAT_RAWQ: the ggml type tag of the raw blocks (sub) and the block bytes
   * in qs (rows * (cols / 256) * ie_rq_bs(sub)); qw is NULL. */
  uint32_t sub;
  /* MAT_Q4: the verified GPU layout (core/ie_core.h): rows*nb*4 nibble words
   * and rows*nb f16 scales. */
  uint32_t *qw;
  uint16_t *qs;
  /* MAT_Q8: the verified Q8_0 GPU layout (ie_q8_dst_word, ie_q8_dst_scale):
   * rows*nb*8 int8 words in qw and rows*nb f16 scales in qs. */
  /* MAT_Q4K (GGUF Q4_K, sub-blocks of 32 in super-blocks of 256): qw as
   * MAT_Q4 (rows*nb*4 nibble words, the nibbles of each sub-block in the
   * Q4_0 order, values 0..15); qs: rows*nb u16 (6-bit scale | 6-bit min << 8)
   * of each sub-block, then rows*(nb/8)*2 u16 (f16 d, f16 dmin) of each
   * super-block. Weight = d*sc*q - dmin*mn (as ggml). Index maps: ie_q4k_* in core/ie_core.h. */
  /* MAT_Q6K (GGUF Q6_K): qw = the low 4 bits as MAT_Q4 nibble words, then
   * 2 words of high bits per sub-block (ie_q6k_hw); qs = (sc0 | sc1 << 8)
   * (int8) per sub-block, then f16 d per super-block (ie_q6k_sc, ie_q6k_d). */
  /* MAT_Q5K, MAT_IQ4XS: core/ie_ext.h. */
  /* MAT_F32: rows*cols floats (dequantized from F32/F16, or Q8_0 for the embedding). */
  float *f;
  /* Experts (flint MoE): n_exp > 1 matrices of rows x cols, one after the
   * other in qw, qs (or f); expert e starts at e * mat_qw_words(), e *
   * mat_qs_n() (or e * rows * cols floats). n_exp 0 or 1: one matrix. */
  uint32_t n_exp;
  /* Streaming load (ie_mat_sink set): the device copies of qw and qs (or
   * f in d0); the host arrays are then freed (NULL). */
  void *d0, *d1;
  /* 1: the streaming load puts d0, d1 in pinned host memory the device reads
   * (the token embedding: one row per token, 1.3 GB of VRAM saved) */
  int host;
  struct mat_lora *lora;
} mat;

typedef struct mat_lora {
  mat a, b;
} mat_lora;

typedef struct {
  uint32_t n;
  float *f;
} vec;


typedef struct {
  mat down, gain, forget, wfg;
} ink_loop_ctrl;

typedef struct {
  vec attn_norm, ffn_norm;
  mat wq, wk, wv, wo, wgate, wup, wdown;
  vec bq, bk, bv; /* n == 0 if absent */
  /* Merged matrices (rows == 0 if not merged: the parts differ in kind).
   * wqkv = [wq; wk; wv] with bias bqkv; wgu = [wgate; wup]. When merged, the
   * parts are freed (their rows stay set, their arrays are NULL). */
  mat wqkv, wgu;
  vec bqkv;
  /* qwen35 (Qwen3.5 / 3.8): rec = 1 for a linear attention layer (Gated
   * DeltaNet), 0 for a full attention layer. ffn_norm holds
   * post_attention_norm. Full attention: wq has 2 * n_head * hd rows, per
   * head [q (hd) | gate (hd)]; q_norm and k_norm are per head RMSNorm
   * weights (hd). Linear attention: wlqkv (conv_dim rows: q | k | v), wz
   * (inner), wbeta, walpha (n_vh rows each), merged into win = [wlqkv; wz;
   * wbeta; walpha] when they have one kind (or wlz = [wlqkv; wz] when only
   * those two match); conv = ssm_conv1d [conv_dim][4]; wo = ssm_out. */
  int rec;
  uint32_t kvi, sti; /* index among the attention layers (KV cache) / the linear layers (state) */
  vec q_norm, k_norm;
  mat wlqkv, wz, wbeta, walpha, wlz, win;
  vec conv, dt_bias, ssm_a, ssm_norm;
  /* flint: win = sliding window (tokens, the current one included; 0: full
   * attention), norope = no RoPE in this layer, sinks = the learned logit
   * per head added to the softmax denominator. moe: wr = router (n_exp rows,
   * f32), egu = [gate_e; up_e] per expert (or eg, eu when their kinds
   * differ), ed = down per expert, ebias = the router bias (selection only;
   * n == 0: none); the shared expert is wgate, wup, wdown (wgu). */
  uint32_t swin;
  int norope, moe;
  vec sinks, ebias;
  /* ricursive (Ripple): PeriNorm adds the *normed* sublayer output to the
   * residual stream, so each block has two post-norms (pan: after
   * attention, pffn: after the FFN), and no bias vectors. lam = the
   * value-residual weight (pre-sigmoid), fclamp = the FFN clamp bound (0:
   * none). */
  vec pan, pffn;
  float lam, fclamp;
  mat wagt; /* ricursive: the attention gate g_proj (attn_gate) */
  mat wr, egu, eg, eu, ed;
  /* inkling2 / inkling2_loop: 24 physical layers shared across 48 logical
   * visits. is_full: global attention (0: SWA with swin). Loop core layers
   * hold loop[0..3] controllers for (q/attn, k, v, mlp). */
  int is_full;
  mat wrel, wsink, k_conv_proj, v_conv_proj;
  vec q_inp_norm, k_inp_norm, v_inp_norm;
  vec k_conv, v_conv, attn_pconv, ffn_pconv;
  vec q_clip, rel_clip, rel_bias;
  vec attn_res_q, ffn_res_q;
  ink_loop_ctrl loop[4]; /* 0: q/attn, 1: k, 2: v, 3: mlp */
} layer;

/* DFlash 2 drafter (z-lab/dflash, a separate GGUF, architecture "dflash"):
 * a block of B tokens [t(P), mask x (B-1)] at positions P .. P+B-1 runs
 * through n_layer Qwen3 layers whose attention also sees the context: the
 * keys and values of fc(the hidden states of n_tl target layers) of the
 * earlier positions (non-causal, sliding window). Two-tap dynamic
 * convolutions along the block wrap the attention and the FFN. The token
 * embedding and the output matrix are the target's. A selector (host)
 * picks one path through the top-k candidates of each position. */
typedef struct {
  vec attn_norm, ffn_norm, q_norm, k_norm;
  vec aconv_base, fconv_base; /* [2][kernel][dim]: [0] before, [1] after the sublayer */
  mat wq, wk, wv, wo, wgate, wup, wdown, wgu, aconv, fconv; /* aconv, fconv: dim -> 2*kernel*groups */
} df_layer;
typedef struct {
  uint32_t n_layer, dim, ffn, n_head, n_kv, hd, block, window, mask, top_k, rank, kernel, group;
  /* DFlash2: the two-tap convolutions and the selector; DFlash v1 (z-lab Qwen3.5 drafters) has
   * neither: plain pre-norm layers, the top candidate of each position */
  int has_conv, has_sel;
  uint32_t n_tl, tl[8]; /* target layers (0-based): their outputs, concatenated, are fc's input */
  float eps, rope_base;
  vec enc_norm, out_norm;
  mat fc, sel_h;
  df_layer *l;
  const void *sel_prev, *sel_next; /* gguf_tensor of the selector codebooks (host, the file mapped) */
} dflash;

/* DSpark drafter (Inkling-2-Mini, separate GGUF, architecture "dspark"):
 * 3 dense pre-norm GQA layers with DFlare 4-feature mix context KV (layers 0,2
 * SWA win=1024; layer 1 global), bidirectional block attention over block_size=8,
 * and sequential rnn_hidden Markov head (rank=256) sharing target embed + lm_head. */
typedef struct {
  vec dflare_norm, attn_norm, ffn_norm;
  vec q_norm, k_norm, k_ctx_norm;
  mat wq, wk, wv, wk_ctx, wv_ctx, wo, wgate, wup, wdown, wgu;
  uint32_t win; /* sliding window (1024 for local layers; 0 for global) */
} ds_layer;
typedef struct {
  uint32_t n_layer, dim, ffn, n_head, n_kv, hd, block, window, markov_rank, vocab, vocab_unpadded;
  uint32_t n_tl, tl[8]; /* target_layer_ids (e.g. [5, 23, 41, 48]) */
  float eps, rope_base, logit_cap;
  vec fuse_weight, mask_embed, out_norm;
  vec mk_dflare_norm, mk_rnn_bias, mk_hidden_bias;
  mat mk_rnn_proj, mk_hidden_out, mk_w1;
  ds_layer *l;
} dspark;

typedef struct {
  int arch, rope;
  int qk_norm; /* qwen3: per head RMSNorm of q and k before RoPE (arch ARCH_QWEN2) */
  uint32_t n_layer, dim, ffn, n_head, n_kv, hd, vocab, ctx_train;
  float eps, rope_base;
  uint32_t n_rot;        /* rotated dims per head (hd, or fewer: qwen35 partial RoPE) */
  uint32_t mrope_sec[3]; /* qwen35 interleaved M-RoPE sections [T, H, W] (sum == n_rot / 2, or 0) */
  uint32_t n_kvl, n_rec; /* layers with a KV cache / with a linear attention state */
  /* qwen35 linear attention: key heads, value heads, head dim (keys and
   * values), conv window; conv_dim = 2 * n_kh * sd + n_vh * sd. */
  uint32_t n_kh, n_vh, sd, d_conv, conv_dim;
  /* flint MoE: experts, experts per token, expert FFN, shared expert FFN;
   * the routing weights are normalized and scaled by escale */
  uint32_t n_exp, n_used, eff, sff;
  float escale;
  /* ricursive: the embedding row scale (0: none, a folded checkpoint) and the
   * FFN clamp bound (0: none) */
  float emb_scale, swiglu_clamp;
  /* inkling2 / inkling2_loop */
  uint32_t n_visits, loop_start, loop_end, n_loops, loop_rank, attn_res_bs;
  uint32_t d_rel, rel_extent, d_sconv, vocab_unpadded;
  uint32_t sc_tot;   /* total sconv channels across all visits (sc_ring = sc_tot * 4 floats) */
  uint32_t *kv_win;  /* [n_kvl]: sliding window per logical KV cache (0: full) */
  float qk_clip, rel_clip, log_scale_floor, log_scale_alpha, logit_cap, situ_beta, situ_lbeta;
  vec tok_norm, out_res_q, loop_res_q, loop_kv_res_q;
  vec ones_dim, ones_hd; /* constant 1.0f vectors for unweighted RMSNorm via OP_RMSNORM */
  mat tok;   /* token_embd (Q4 repacked or F32), rows = vocab, cols = dim */
  mat out;   /* output; may share the arrays of tok (tied) */
  int tied;
  vec out_norm;
  layer *l; /* n_layer layers, then (has_mtp) the MTP layer l[n_layer] */
  /* qwen35 multi-token prediction (MTP, "nextn") head: one full attention
   * layer l[n_layer] (its own KV cache, kvi = n_kvl - 1) after
   * x = eh_proj [rmsnorm(embed(token)) * enorm | rmsnorm(h) * hnorm]; then
   * rmsnorm(.) * head_norm (this is also the h of the next draft) and the
   * output matrix. */
  int has_mtp;
  mat eh;
  /* IE_MTP_EMBED=1: the MTP head reads its input token from the head file's own token_embd (Q8_0 or Q4_0, in host
   * memory) instead of tok; rows == 0: tok */
  mat mtp_tok;
  /* the output matrix requantized to Q4_0 for the MTP drafts only (rows ==
   * 0: the drafts use out). Half the bytes of Q8_0; the verify step still
   * uses out, so the output is unchanged. */
  mat out_draft;
  /* the DFlash block's output matrix: the rows of out_draft as Q8_0 (multi-token GEMVs; rows == 0: out) */
  mat df_head;
  /* the DSpark Markov head's output matrix: Q8_0 over draft vocab (rows == 0: out) */
  mat ds_head;
  /* the draft head's rows: tokens 0 .. dv_n - 1, then dv_s .. (IE_DRAFT_VOCAB; 0: all rows, row = token) */
  uint32_t dv_n, dv_s;
  /* IE_DRAFT_VOCAB=64k or a manifest: the token of each draft head row (out_draft.rows ascending ids; NULL: dv_n / dv_s) */
  uint32_t *dv_map;
  vec enorm, hnorm, head_norm;
  dflash *df; /* the DFlash 2 drafter (NULL: none) */
  dspark *ds; /* the DSpark drafter (NULL: none) */
  /* tokenizer */
  int tok_gpt2; /* 1: GPT2 byte-level, 0: SentencePiece */
  uint32_t n_tokens;
  char **tokens;
  uint64_t weight_bytes; /* bytes of all repacked/dequantized weights */
} model;

/* The tensor registry (ie-lab): where the loader put each GGUF tensor it read (load_mat, load_vec). A matrix
 * tensor holds rows r0 .. r0 + rows - 1 of w, which may be a fused matrix (wqkv, wgu, win: concat_mats moves the
 * entries); a vector tensor is v. src: the tensor in its (open, mapped) GGUF file: the original bytes. A name can
 * appear twice (output_norm.weight is also the MTP head norm of a head without its own). */
typedef struct {
  char name[96];
  mat *w;
  vec *v;
  uint32_t r0, rows, cols; /* a vector: rows 1, cols its length */
  const gguf_tensor *src;
  int q8_ok, emb, q4q8;    /* load_mat_src's arguments and q4_as_q8 at the load */
} ie_tref;
extern ie_tref *ie_trefs;
extern uint32_t ie_n_trefs;
/* The first entry named name, or NULL. */
const ie_tref *ie_tref_find(const char *name);
/* The bytes of the tensor in its GGUF type. */
uint64_t ie_tref_bytes(const ie_tref *r);
/* A matrix entry: data (GGUF bytes of its type, ie_tref_bytes) decoded by the loader's own path into part, a host
 * matrix of r->rows rows (free with ie_mat_free). 0 if the result's kind or block count differs from r->w (the
 * caller then cannot patch the device copy). */
int ie_tref_decode(const ie_tref *r, const uint8_t *data, mat *part);
/* A vector entry: r->v->f replaced by data decoded. */
void ie_tref_set_vec(const ie_tref *r, const uint8_t *data);
void ie_mat_free(mat *w);
/* 1: graph_build keeps the MTP FFN buffers for capture (graph.mtp_r, mtp_a), as IE_MTP_CAPTURE does */
extern int ie_mtp_keep;

/* Load the MTP head (blk.N.nextn.*) from g (the model file or a separate
 * file); returns 0 if g has none. */
int model_load_mtp(model *m, gguf_file *g);
/* 1 if g has an MTP head (blk.N.nextn.*) of a model model_load_mtp takes. */
int gguf_has_mtp(const gguf_file *g);
/* Without --draft, ie-run and ie-serve draft this many tokens per step when
 * there is an MTP head (--mtp FILE or the model's own); --draft 0: off.
 * DRAFT_AUTO: no --draft given. */
#define MTP_DRAFT_DEFAULT 3u
/* The most drafts per step: a verify of 9 tokens (the _tr8 GEMV kernels take up to 9) */
#define MTP_DRAFT_MAX 8u
#define DRAFT_AUTO 0xffffffffu
/* Load a DFlash 2 drafter from g (kept open: the selector reads its
 * codebooks); returns 0 if g is not one. */
int model_load_dflash(model *m, gguf_file *g);
/* Row tok of a selector codebook (which 0: predecessor, 1: successor). */
void model_df_code(const model *m, int which, uint32_t tok, float *out);
/* Make m->out_draft from the output matrix of g (the model file). */
void model_make_draft_head(model *m, gguf_file *g);
/* The token of row i of the draft head (a shortlist row at or past the end of the table is a bug: exits). */
uint32_t model_draft_tok_map(const model *m, uint32_t i);
/* after model_load_dflash: the DFlash head (df_head) from the target file g */
void model_make_df_head(model *m, gguf_file *g);
/* Load a DSpark drafter (Inkling-2) from g; returns 0 if g is not one. */
int model_load_dspark(model *m, gguf_file *g);
/* after model_load_dspark: the DSpark Q8_0 head (ds_head) over the draft vocabulary */
void model_make_ds_head(model *m, gguf_file *g);
static inline uint32_t model_draft_tok(const model *m, uint32_t i) {
  return m->dv_map ? model_draft_tok_map(m, i) : m->dv_n && i >= m->dv_n ? m->dv_s + (i - m->dv_n) : i;
}
/* SHA-256 of p[0 .. n) as 64 lowercase hex characters and a 0. */
void ie_sha256_hex(const void *p, size_t n, char out[65]);
/* Load a vocabulary shortlist manifest (schema rdna4.mtp.vocab-shortlist.v1): the ids file it names must have the
 * manifest's sha256, `size` distinct ids below `vocab`, and hold every required_eog_ids entry. Returns the ids
 * (malloc'ed, ascending; *n of them), or NULL with a message in err. */
uint32_t *ie_load_vocab_shortlist(const char *manifest, uint32_t vocab, uint32_t *n, char *err, size_t errn);
/* If set, model_load gives every finished matrix to this function (for
 * example: copy it to the GPU and free the host arrays), so that the host
 * never holds all the weights at once. */
extern void (*ie_mat_sink)(mat *w);
void model_load(model *m, gguf_file *g);
void model_load_lora(model *m, gguf_file *g, const gguf_file *adapter);
void model_free(model *m);
/* Repack GGUF Q4_0 bytes (w->rows, w->cols, w->nb set) into w->qw, w->qs. */
void repack_q4(mat *w, const uint8_t *src);
/* A matrix of rows x cols from GGUF bytes of type ttype, as model_load
 * makes it (Q8_0 kept as Q8; K and IQ kinds native where supported). */
void model_mat_from_gguf(mat *w, uint32_t ttype, const uint8_t *src, uint32_t rows, uint32_t cols);
/* Repack GGUF Q8_0 bytes into w->qw, w->qs (the verified Q8_0 layout). */
void repack_q8(mat *w, const uint8_t *src);
/* The text of token id t (malloc'ed, decoded bytes; *len set). */
char *model_detok(const model *m, uint32_t t, uint32_t *len);

/* ---- the op list of one decode step ---- */
enum {
  OP_EMBED,   /* a = x <- row tok of token_embd */
  OP_RMSNORM, /* b = rmsnorm(a) * w */
  OP_QUANT,   /* b = Q8(a) */
  OP_GEMV_Q4, /* b = W a (a is a Q8 buffer) */
  OP_GEMV_Q4K, /* the same for a MAT_Q4K matrix */
  OP_GEMV_Q6K, /* the same for a MAT_Q6K matrix */
  OP_GEMV_F32,/* b = W a (a is f32) */
  OP_GEMV_Q8, /* b = W a, W is Q8_0 (a is a Q8 buffer) */
  OP_BIAS,    /* a += w */
  OP_ADD,     /* c = a + b */
  OP_ROPE,    /* rotate a in place: nh heads of hd */
  OP_KV,      /* store a (k) and b (v) at the step position of layer */
  OP_ROPE_KV, /* rotate a (q, nh heads) and b (k, n_kv heads); store b and c (v) in the KV cache of layer */
  OP_ATTN,    /* b = attention(q = a, the KV cache of layer), scores in c */
  OP_SWIGLU,  /* a = silu(a) * b */
  OP_ARGMAX,  /* b (u32) = argmax(a) */
  /* qwen35 full attention: a = [q|gate per head (2 hd) x nh | k | v]; b = q
   * (nh x hd): per head RMSNorm (v = q_norm, nv = k_norm), RoPE of the
   * first n_rot dims (NEOX pairs), k and v into the KV cache of layer. */
  OP_QKN_ROPE_KV,
  /* qwen35 linear attention (Gated DeltaNet) of one token: a = [q k v
   * (conv_dim) | z (inner) | beta (n_vh) | alpha (n_vh)], b = the output
   * (inner, after the gated RMSNorm), Q8 copy in qo. State of layer. */
  OP_GDN,
  /* b (at boff) = a: n floats per token */
  OP_COPY,
  /* DFlash: b = conv(a) along the tokens of the run: b[t] = sum_{o<2}
   * (base[o] + dyn[t][o]) * a[t-o] per channel (a[-1] = 0); v = base
   * [2][2][dim], c = dyn ([2][2][groups] per token), nh = which half (0 or
   * 1); res >= 0: b = res + conv(a) */
  OP_DF_CONV,
  /* DFlash: a = [q (n_head hd) | k | v] per token (q absent: nh == 0):
   * per head RMSNorm (v: q_norm, nv: k_norm), RoPE (NEOX, all hd dims),
   * k and v into the drafter KV cache of layer at pos + t */
  OP_DF_QKV,
  /* DFlash: b = attention(q = a) over the drafter KV cache of layer: query
   * at pos + t sees the positions within the window up to pos + T - 1 */
  OP_DF_ATTN,
  OP_GEMV_Q5K,   /* b = W a, MAT_Q5K (a is a Q8 buffer) */
  OP_GEMV_IQ4XS, /* b = W a, MAT_IQ4XS */
  OP_GEMV_RAW,   /* b = W a, MAT_RAWQ (raw GGUF IQ2/IQ1/IQ3_XXS/Q2_K blocks) */
  /* flint MoE. ROUTE: b = [nh expert ids (u32) | nh weights (f32)] from the
   * router logits a (n experts): sigmoid, top nh by sigmoid + bias (v, or
   * none), weights = sigmoid / sum * m->escale. */
  OP_MOE_ROUTE,
  /* b + s bstr = W_e (a + s astr) for slot s < nh, e = expert id s of the
   * route buffer c (astr 0: one input for all slots); w->kind: Q8, Q4K, Q5K,
   * Q6K, IQ4XS (a is Q8) or F32 */
  OP_GEMV_EXP,
  /* per slot s < nh: g = a + s astr (floats [gate n | up n]): g = silu(g) * up;
   * the Q8 copy of slot s at qo + s q8_bytes(n / 32) */
  OP_MOE_SWIGLU,
  /* b = res + sum over slots s of weight_s (a + s n floats) (c: the route) */
  OP_MOE_SUM,
  /* ricursive value residual: b = c + (a - c) * sigmoid(lam), the mix of the
   * scaled layer-0 embeddings (c, held for the whole pass) into the normed
   * attention input a; qo: the Q8 copy of b (the input of the wv GEMV) */
  OP_VRES,
  /* ricursive: a *= sigmoid(b), n floats per token (the attention gate);
   * qo: the Q8 copy of a */
  OP_GATE,
  /* b = W a, MAT_IQ4NL (a is a Q8 buffer, Q4_0 nibble layout with kvalues_iq4nl) */
  OP_GEMV_IQ4NL,
  /* inkling2 per-head scale/clip (clamp > 0) or unweighted RMSNorm (clamp == 0):
   * nh heads of qstr floats (a -> b, in-place a == b allowed).
   * clamp > 0: x *= (v ? v : 1); x *= min(1, clamp / (||x||_2 + 1e-6));
   *            if (norope) x *= 1 + escale * logf(max((pos + 1) / (float)win, 1.0f)).
   * clamp == 0: x *= rsqrtf(mean(x^2) + escale) per head (v optional). */
  OP_VCLIP,
  /* inkling2 depth-wise AttnRes mix: S = nh vectors of n floats per token:
   * slots 0 .. nh-2 (or 0 .. nh-1 if c < 0) contiguous in a (at aoff),
   * plus optional last slot in c (at coff). Sn[s] = rmsnorm_unweighted(S[s], eps).
   * For k = 0 .. qstr - 1 queries q_k at v->f + roff + k * n:
   * a_k = softmax_s(dot(Sn[s], q_k) / sqrt(n)), b[boff + k*n ..] = sum_s a_k[s] * Sn[s]. */
  OP_ATTN_RES,
  /* inkling2 loop controller gate: a (inp, n floats at aoff), c (msg, n floats at coff),
   * gt (f at gtoff, g at gtoff + n * 4): b (at boff) = (1 - tanh(f)) * a + tanh(g) * c. */
  OP_LOOP_GATE,
  /* inkling2 4-tap short conv: a (n floats) -> b (n floats), static weights v (4*n floats
   * in [t-3, t-2, t-1, t] order), optional dynamic proj c (nh * 4 floats in forward lag
   * order [lag=0..3], applied to nh heads of n/nh floats), channel base scoff in sc_ring
   * ([sc_tot][4] floats indexed by pos & 3). out = (u + y) * escale; updates ring[pos & 3] = u. */
  OP_SCONV,
  /* inkling2 attention: a = Q (nh x n), b = out (nh x n), c = scores/parts scratch,
   * nq = R (nh x m->d_rel), res = dynamic sink logits (nh floats), v = rel_bias
   * ([m->rel_extent][m->d_rel]), kvi = logical KV index, win = SWA window (0: full).
   * Score = dot(Q,K)/sqrt(hd) + (d < rel_extent ? dot(R[h], rel_bias[d]) : 0);
   * softmax includes exp(sink[h]); sink prob omitted from V sum. */
  OP_REL_ATTN,
  /* inkling2 MoE SITU activation: nh slots at a + sl * astr ([g (n) | u (n)]),
   * g = (clamp * tanh(g / clamp)) * sigmoid(g) * (lam * tanh(u / lam));
   * quantizes slot sl to qo + sl * bstr if qo >= 0. */
  OP_SITU,
  /* inkling2 logit softcap + unpadded vocab mask: a -> b (in-place a == b allowed):
   * for j < nh: b[j] = clamp > 0 ? clamp * tanh(a[j] / clamp) : a[j];
   * for nh <= j < n: b[j] = -1e30f. */
  OP_LOGIT_CAP,
};

typedef struct {
  int kind;
  int a, b, c;     /* buffer ids (-1: none) */
  uint32_t n;      /* elements (of a) */
  uint32_t nh;     /* rope: heads */
  int layer;
  const mat *w;
  const vec *v;
  /* Byte offsets into the buffers a, b, c (a buffer can hold several
   * vectors, for example q, k and v). */
  uint32_t aoff, boff, coff;
  /* GEMV only: after the result (with bias and residual) is complete,
   * nout = rmsnorm(result) * nv and, if nq >= 0, its Q8 copy in nq. The GPU
   * does this in the last workgroup to finish. */
  const vec *nv;
  int nout, nq;
  /* Fused work (-1: none):
   *  qo:  also write the Q8 copy of this op's f32 result (RMSNORM: b,
   *       ATTN: b, SWIGLU: a), as OP_QUANT would.
   *  res: GEMV only: add this f32 buffer to the result (residual). The GEMV
   *       also adds v (the bias) when v is set. */
  int qo, res;
  /* ATTN only: if gt >= 0, out *= sigmoid(gate), gate of head h at float
   * gtoff / 4 + h * gstride of buffer gt (qwen35). */
  int gt;
  uint32_t gtoff, gstride;
  /* QKN_ROPE_KV: qstr = floats from one q head to the next (qwen35: 2 hd,
   * flint: hd), norope: no RoPE. ATTN: win (0: all positions), sk = the
   * sinks (NULL: none). GEMV_EXP, MOE_SWIGLU: astr, bstr = byte strides of
   * the slots. */
  uint32_t qstr, win, astr, bstr;
  int norope;
  const vec *sk;
  /* SWIGLU, MOE_SWIGLU: bound of the gate and the up half (ricursive: 20;
   * 0: no clamp; the gate is bounded above only, up on both sides).
   * EMBED: the row is multiplied by escale (ricursive embed_scale; 0: none).
   * VRES: lam, the pre-sigmoid value-residual weight of the layer. */
  float clamp, escale, lam;
  /* 1: f32 output b (or a for SWIGLU, where a == b) has no consumer and is not
   * host-visible; qo >= 0 holds the live Q8 output. GPU kernels may skip writing b. */
  uint8_t dead_b;
  /* Logical KV index (0 .. m->n_kvl - 1) for OP_KV / OP_ATTN / OP_ROPE_KV / OP_QKN_ROPE_KV / OP_REL_ATTN;
   * sconv channel offset (scoff) in sc_ring; float offset (roff) into v->f for OP_ATTN_RES. */
  uint32_t kvi, scoff, roff;
} op;

/* Activation buffers. Each has a size in bytes and, after planning, an
 * arena offset and a lifetime [first, last] in op indices. */
#define MAX_BUFS 8192
/* GPU attention (ie_attn_split): positions per workgroup at decode position
 * pos (positions 0 .. pos). Measured on the R9700: chunks of 8 or 16 for
 * short contexts were not faster at 64 tokens and slower at 512 (the last
 * workgroup merges more partial results), so the chunk is 32. */
static inline uint32_t ie_att_ch(uint32_t pos) { (void)pos; return 32u; }
static inline uint32_t ie_att_nsplit(uint32_t pos) { return (pos + ie_att_ch(pos)) / ie_att_ch(pos); }
/* The most splits for any position below n_ctx. The scores buffer of OP_ATTN
 * also holds the partial results: n_head * ie_att_max_split(n_ctx) * (hd + 2)
 * floats. ie_attn_split needs hd <= IE_ATT_MAX_HD. */
static inline uint32_t ie_att_max_split(uint32_t n_ctx) {
  uint32_t mx = 1;
  for (uint32_t p = 0; p < n_ctx; p++)
    if (ie_att_nsplit(p) > mx) mx = ie_att_nsplit(p);
  return mx;
}
#define IE_ATT_MAX_HD 256u
/* ie_attn_split keeps the weights of up to this many splits in LDS. */
#define IE_ATT_MAX_SPLIT 4096u
/* ie_attn_long (GPU, hd 256, kernels/ie_kernels.c): the tokens at positions >= ie_att_long_min() use splits of
 * IE_ATL_CH positions, merged in groups of IE_ATL_G splits; a split or group leaves IE_ATL_PSZ floats (M, L, o) per
 * query head. The older split kernels (ie_att_ch, partial results of hd + 2 floats) then only cover the positions below
 * ie_att_long_min(), so the part area of the older kernels is small and its limit IE_ATT_MAX_SPLIT never applies. */
#define IE_ATL_CH 128u
#define IE_ATL_G 8u
#define IE_ATL_PSZ 260u
/* partial results per query head for the positions below n_ctx: the splits, the groups and one spare */
static inline uint32_t ie_atl_parts(uint32_t n_ctx) {
  const uint32_t ns = (n_ctx + IE_ATL_CH - 1u) / IE_ATL_CH;
  return ns + (ns + IE_ATL_G - 1u) / IE_ATL_G + 1u;
}
uint32_t ie_att_long_min(void);       /* IE_ATTN_LONG_MIN, default 256 (src/model.c) */
int ie_att_long_ok(const model *m);   /* the model and the environment allow ie_attn_long (the GPU backend also needs the kernel) */
/* floats of the part area (OP_ATTN's scores buffer) that one token of the GPU split attentions needs */
uint32_t ie_att_part_floats(const model *m, uint32_t n_ctx);
typedef struct {
  const char *name;
  /* stride: bytes per token (256-aligned); size = stride * graph.T; or
   * stride 0: one part that all tokens share (new_buf_shared) */
  uint32_t stride, size, off, first, last;
  /* rows allocated, if fewer than graph.T (graph_cap_logits); 0: graph.T. A token past the last row uses the last row. */
  uint32_t rows;
} buf;

typedef struct {
  op *ops;
  uint32_t n_ops;
  buf *bufs;
  uint32_t n_bufs;
  uint32_t arena; /* bytes */
  int x, logits, argmax; /* buffer ids that the host reads */
  uint32_t n_ctx;
  /* T: the most tokens one run of the op list does (speculative decoding:
   * 1 + drafts). Each buffer holds T tokens (buf.stride apart). */
  uint32_t T;
  /* The ops [0, n_main) are the model; [n_main, n_ops) the MTP head (if
   * built). h_out: the final normed hidden state of the model (the MTP h
   * input of the next position); mtp_h: the MTP h input (the host copies
   * rows into it); mtp_g: the MTP normed output (the h of the next draft). */
  uint32_t n_main;
  /* first op of the output head (output matrix, argmax) of the model and of
   * the MTP head: a prompt chunk runs the model head for its last token
   * only, and the MTP head not at all; i_mtp_kv is the end of the MTP KV-write
   * prefix (sec == 4 stops here and skips i_mtp_wq when IE_MTP_CAPTURE is off) */
  uint32_t i_head, i_mtp_head, i_mtp_kv;
  int32_t i_mtp_wq;
  int h_out, mtp_h, mtp_g, mtp_logits, mtp_argmax;
  /* IE_MTP_CAPTURE only (else -1): the MTP layer's FFN input x (the residual
   * after attention) and its gate/up buffer (after SWIGLU, the first ffn
   * floats of a row are the input of ffn_down); kept for the whole op list.
   * mtp_y: the layer's output before shared_head_norm (ie-lab /chain) */
  int mtp_r, mtp_a, mtp_y;
  /* the MTP ops end at n_mtp; DFlash (m->df set): the context ops [i_dfc,
   * i_dfb) (fc of df_feat rows, their keys and values into the drafter KV
   * cache), the block ops [i_dfb, n_ops) (df_logits: the draft head's
   * logits, df_sel: the selector's hidden projection). df_feat: the outputs
   * of the target layers df->tl, written by the model ops. */
  uint32_t n_mtp, i_dfc, i_dfb;
  int df_feat, df_logits, df_sel;
  int ds_feat; /* DSpark: 4 target layer hidden states [4 * dim] per token (-1: none) */
  uint32_t visit_start[64]; /* R35 inkling2: op index where logical visit il (0..n_visits_rec-1) begins */
  uint32_t n_visits_rec;
  /* IE_DF_CAPTURE only (else -1): the drafter's context input for every
   * prefill position, enc_norm(fc(df_feat)), read by prefill */
  int df_cap;
  float *rope_cos, *rope_sin; /* [n_ctx][n_rot/2] */
  /* The KV cache of attention layer i (kvi): kv_len[i] positions (position
   * p in slot p % kv_len[i]; n_ctx, or a ring for a sliding window layer)
   * from entry kv_off[i] (in units of n_kv * hd values); kv_total entries */
  uint32_t *kv_len;
  uint64_t *kv_off, kv_total;
  uint64_t weight_bytes_per_token;
  /* Batched MTP tree frontier & folded top-K control for sec == 1 (b->run(b, 1, ...)):
   *   - on == 0 (normal linear catch-up at pos .. pos + T - 1):
   *     when topk_k > 0 (<= 64) and topk_v && topk_i, backend may skip OP_ARGMAX + out D2H,
   *     run i_mtp_head from row topk_row, run gpu_topk(K = topk_k) on mtp_logits rows
   *     [topk_row .. topk_row + topk_n - 1] into (topk_v, topk_i), and set topk_done = 1.
   *   - on == 1 (batched MTP tree frontier pass of T = 1..15 nodes, root at `pos`):
   *     caller has placed node t's parent hidden state into mtp_h row t and passes `pos = P` (root).
   *     OP_QKN_ROPE_KV writes blk.64 KV slots (pos + slot_base) .. (pos + slot_base + T - 1) using
   *     RoPE positions (pos + dep[t]). OP_ATTN (ie_attn_long) runs at p0 = pos, Tl = T with tr_dev:
   *       TR_DEP[t] = dep[t], TR_PATH[16*t + e] = path[t][e] (0 <= e <= dep[t]; path[t][0] = 0,
   *       path[t][dep[t]] = slot_base + t), tr_tab[IE_TR_KEND] = max_{t<T}(dep[t] + 1).
   *     Evaluates i_mtp_head on ALL T frontier rows (wfrom = 0), and if topk_k > 0 folds gpu_topk
   *     over mtp_logits rows 0 .. T - 1 into (topk_v, topk_i) in one D2H sync (topk_done = 1). */
  struct ie_mtp_tree {
    int on;
    uint32_t slot_base;
    uint8_t dep[16];
    uint8_t path[16][16];
    uint32_t topk_k, topk_row, topk_n;
    float *topk_v;
    uint32_t *topk_i;
    float *topk_lse;
    int topk_done;
  } mtp_tr;
} graph;

/* Byte offset in the 384-byte tree table (tr_tab / tr_dev) storing max_{t<T}(dep[t] + 1) so
 * ie_attn_long sets kend = pos + tr[IE_TR_KEND] even on partial frontier passes where T <= dep[t]. */
#define IE_TR_KEND 368u
static inline struct ie_mtp_tree *graph_mtp_tr(const graph *g) {
  return (struct ie_mtp_tree *)(uintptr_t)&g->mtp_tr;
}

/* T: tokens per run (1, or 1 + drafts); mtp: also build the MTP ops (and
 * the DFlash ops when m->df is set). */
void graph_build(graph *gr, const model *m, uint32_t n_ctx, uint32_t T, int mtp);
/* Default IE_INKLING_KERNEL_MASK shared between model.c (exp_gu cap eligibility) and hip.c (HSACO load):
 * R1 + R2 prefill + R4 verify/sconv/dspark + R5 zsync (bit 48) + R6 winners:
 *   bits 50..53 (4-ctrl loop fusion; bit 53 T<=16), bit 54 (compact MoE), bit 57 (prefill IQ3+SITU), bit 58 (QT=4 global verify attn).
 * Opt-in (0): bit 49 (small_ver_r5), bit 55 (slot_r6 MoE). */
#define IE_INK_DEFAULT_MASK (0x369c3ffffULL | (0x7fULL << 42) | (0x1fULL << 50) | (0x3ULL << 57) | (0x3ULL << 59) | (0x3ULL << 61) | (1ULL << 63))
/* R05+ defaults in ink_mask2 (bits 64..127):
 *   bit 64: ie_inkling_qkv_qclip_r7 (R09 Fusion D1, T<=4)
 *   bit 65: ie_inkling_rel_clip_sink_gate_r7 (R10 Fusion D2, T<=4)
 *   bit 67: ie_inkling_attn_res_norm_q8_r7 (R11 Fusion E2; bit 66 E1 & bit 68 E3 opt-in)
 *   bit 69: ie_inkling_moe_pf_worklist_r7 (R13 Wave 3 Prefill flat worklist)
 *   bit 70: ie_inkling_gemm_rq_exp_iq3xxs_situ_r7 (R14 Wave 3 3-WG/CU IQ3+SITU WMMA; bit 71 IQ4_NL r7 opt-in)
 *   bit 72: ie_inkling_gemm_q8_med_r7 (R16 Wave 3 3-WG/CU Dense Q8_0 Med WMMA)
 *   bit 73: ie_inkling_gemm_q8_64x64_r7 (R17 Wave 3 64x64 Q8_0 WMMA for wk/wv; small_r7, b74, b75 opt-in)
 *   bit 76: ie_inkling_gemv_q4k_head_ver_r7 (R19 Wave 4 1-Pass Q4_K Head + Logit-Cap; bit 77 SWA QT=4 opt-in)
 *   bit 78: ie_inkling_rel_attn_ver_glob_wmma_r7 (R21 Wave 4 Single-Pass T<=8 Global Verify Attention)
 *   bit 79: ie_inkling_rel_attn_pf_r7 (R22 Wave 4 Zero-Division Far/Near Prefill Relative Attention)
 *   bit 80: ie_inkling_rel_attn_pf_split_r7 (R23 Wave 4 Incremental Prefill Split-KV Attention, T=17..96)
 *   bit 88: ie_inkling_ds_{base_q4k,gemv_p_q4,cap_sample}_idmap*_r7 (R30 Wave 6 0-VRAM 60K+4K Certified Shortlist)
 *   bit 92: ie_inkling_argmax_batched_r7
 *   bit 93: ie_inkling_topk_batched_r7
 *   bit 94: ie_inkling_ds_cap_sample_zsync_r7
 *   bit 95: ie_inkling_ds_attn_long_r7
 *   bit 96: ie_inkling_qkv_qclip_t8_r9 (R25 Wave 5 Single-Pass CT=8 D1, T=1..8)
 *   bit 97: ie_inkling_rel_clip_sink_gate_t8_r9 (R25 Wave 5 Single-Pass CT=8 D2, T=1..8)
 *   bit 98: ie_inkling_wo_sconv_post_t1_r9 (R25 Wave 5 Lean 58-VGPR E3, T=1)
 *   bit 102: ie_inkling_gemm_rq_exp_iq3xxs_situ_r9 (R36 Wave 7 Turbo 113-VGPR IQ3_XXS+SITU Prefill WMMA; bit 99 IQ4_NL, bits 100-101 Q8 R9, bit 90 subspan glue opt-in)
 *   bits 81, 103, 104: R42 (H34+H01) 0-VRAM FLUTE SoA IQ3_XXS load-time repack + R7 T=1/verify + R10 110-VGPR prefill WMMA
 *   bit 121: R55 (H20) Single-Launch QKV+sconv+KV (-48 L FP16 / -40 L 256K Q4) */
#define IE_INK_DEFAULT_MASK2 ((1ULL << (64u - 64u)) | (1ULL << (65u - 64u)) | (1ULL << (67u - 64u)) | (1ULL << (69u - 64u)) | (1ULL << (70u - 64u)) | (1ULL << (72u - 64u)) | (1ULL << (73u - 64u)) | (1ULL << (76u - 64u)) | (1ULL << (78u - 64u)) | (1ULL << (79u - 64u)) | (1ULL << (80u - 64u)) | (1ULL << (81u - 64u)) | (1ULL << (88u - 64u)) | (1ULL << (92u - 64u)) | (1ULL << (93u - 64u)) | (1ULL << (94u - 64u)) | (1ULL << (95u - 64u)) | (1ULL << (96u - 64u)) | (1ULL << (97u - 64u)) | (1ULL << (98u - 64u)) | (1ULL << (102u - 64u)) | (1ULL << (103u - 64u)) | (1ULL << (104u - 64u)) | (1ULL << (121u - 64u)))
/* R66+ defaults in ink_mask3 (bits 128..191):
 *   bit 128: H24 fused Markov RNN + batched ADD_Q8 (R66 Wave 13, 100% exact SHA)
 *   bit 129: H25b WMMA L1 ring attn + H25c fused proj+res+norm+Q8 (R66 Wave 13, H25a opt-in off, 100% exact SHA) */
#define IE_INK_DEFAULT_MASK3 ((1ULL << (128u - 128u)) | (1ULL << (129u - 128u)))
/* Parse 192-bit Inkling-2 kernel mask from IE_INKLING_KERNEL_MASK (up to 192-bit hex or low 64 bits),
 * IE_INKLING_KERNEL_MASK2 (bits 64..127), and IE_INKLING_KERNEL_MASK3 (bits 128..191),
 * falling back to IE_INK_DEFAULT_MASK/MASK2/MASK3. Any output pointer may be NULL. */
void ie_ink_parse_mask(uint64_t *lo, uint64_t *hi, uint64_t *hi3);

/* Before plan_arena: the logits buffers (main and MTP head) keep 16 rows instead of T when the graph has more than 16
 * tokens. A prompt chunk computes its head for the last token only and a verify run has at most 16 tokens, so the
 * 256-token chunks of the server save 320 MB of the arena. Not for --ppl (it reads every row of a chunk). */
void graph_cap_logits(graph *g);
/* Before plan_arena (GPU only): when Inkling-2 R6 fused prefill IQ3_XXS+SITU (bit 57) is active, cap exp_gu to 16 rows
 * (36 MiB -> 1.125 MiB at C512) while keeping T=1..16 unfused/verify intact. */
void graph_cap_exp_gu(graph *g, const model *m);
void graph_free(graph *gr);
/* The ring of a sliding window layer: the smallest power of two >= win +
 * max(IE_KV_SLACK, graph T) (room for the tokens of one run), at most n_ctx. */
#define IE_KV_SLACK 16u
/* u32 words of qw and u16 entries of qs of one rows x cols matrix (one expert). */
uint64_t mat_qw_words(int kind, uint32_t rows, uint32_t cols);
uint64_t mat_qs_n(int kind, uint32_t rows, uint32_t cols);
/* Bytes of the raw block array of a MAT_RAWQ matrix (model.c). */
uint64_t ie_rawq_bytes(const mat *w);
/* Bytes of one 256-weight block of a ggml type (0 if not a MAT_RAWQ kind). */
uint32_t ie_rawq_bs(uint32_t type);
/* Expert e of w as a matrix of its own (the arrays point into w's). */
mat mat_expert(const mat *w, uint32_t e);

/* plan.c: give each buffer an arena offset; checks with ie_plan_check and
 * exits if the check fails. */
void plan_arena(graph *gr, int verbose);

#endif
