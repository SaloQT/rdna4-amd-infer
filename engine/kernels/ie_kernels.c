/* ie_kernels.c -- the GPU kernels for gfx1201 (RDNA4, wave32), in plain C.
 *
 * Build (see Makefile):
 *   clang -x c -std=c11 --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -nogpulib -O3
 *   ld.lld -shared -> build/ie_kernels.hsaco
 *
 * Index, layout, integer dot product, grid, lane split and the order of
 * the wave reduction come from the verified core (core/ie_core.h). In this
 * build ie_dot4_us is the v_dot4_i32_iu8 instruction.
 *
 * Every kernel runs with 256 threads per workgroup (8 waves of 32). The
 * flat work group size 256 is written into the IR by the Makefile (clang 18
 * does not accept amdgpu_flat_work_group_size in C). */
#include "../core/ie_core.h"
#include "../core/ie_ext.h"

#define G __attribute__((address_space(1))) /* global memory */
#define LDS __attribute__((address_space(3), loader_uninitialized))
#define LDSP __attribute__((address_space(3))) /* LDS pointer targets */
#ifndef KERNEL
#define KERNEL __attribute__((amdgpu_kernel, visibility("protected"))) void
#endif
/* the multi-token GEMVs (measured: as functions of their own in the
 * megakernel (noinline) MTP drops 94 -> 71 tokens/s; they stay inlined) */
#ifndef KERNEL_HEAVY
#define KERNEL_HEAVY KERNEL
#endif
#ifndef KERNEL_CALL
#define KERNEL_CALL KERNEL /* the megakernel: a call, not inlined (its registers apart) */
#endif
#ifndef KERNEL_LOOP
#define KERNEL_LOOP KERNEL /* the megakernel: inlined into a per-phase loop function (tools/mk_gen.py) */
#endif
#define NT 256u
/* A tree verify (src/hip.c tree_table): the table of its nodes (node 0: the root, nodes in an order with parent <
 * child), bytes: TR_DEP depth of node t; TR_PATH + 16 j + d: the ancestor of node j at depth d (d <= depth of j; the
 * node itself at its depth); TR_PAR parent of node t; TR_A1 + 16 (d - 1) + t: the ancestor of node t d = 1 .. 3 levels
 * up (0xFF: before the root, from the conv ring); TR_ANC (u16 at 2 t): the ancestors of node t and t itself, without
 * the root, as bits. */
#define TR_DEP 0u
#define TR_PATH 16u
#define TR_PAR 272u
#define TR_A1 288u
#define TR_ANC 336u
static inline unsigned tr_anc(const G unsigned char *tr, unsigned t) { return (u32)((const G unsigned short *)(tr + TR_ANC))[t]; }

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef unsigned char u8;
typedef unsigned short u16;
typedef signed char i8;
typedef _Float16 f16;

/* Raw GGUF IQ/K blocks (MAT_RAWQ): one source with the CPU backend
 * (kernels/ie_iqraw.inc). The block pointer is a global pointer. */
#define IE_RQP const G u8
#define IE_RQF const G float
#define IE_RQU const G u32
#include "ie_iqraw.inc"
/* The KV cache element: fp16 (half the memory of f32: 128K positions of
 * Qwen3.8-27B in 9.1 GB), or f32 with -DIE_KV_F32. The attention kernels
 * read it as f32 (exact), so only the store rounds. The host finds the type
 * by the kernel ie_kv_f16 (present only in the fp16 build). */
#ifdef IE_KV_F32
typedef float kvt;
typedef float kv4 __attribute__((ext_vector_type(4)));
#else
typedef _Float16 kvt;
typedef _Float16 kv4 __attribute__((ext_vector_type(4)));
#endif
#define KV4(p, i) __builtin_convertvector(((const G kv4 *)(p))[i], f32x4) /* 4 elements as f32 */

/* The q8_0 KV cache (-DIE_KV_Q8, build/ie_kernels_q8.hsaco; the host picks it
 * with IE_KV_Q8=1): about half the memory of fp16. For each position and KV
 * head one record of hd bytes (int8 values) and hd / 32 fp16 scales (one per
 * block of 32 values along the head dimension): value = q * scale, with
 * scale = max|x| / 127 rounded to fp16 and q = round(x / scale). The value is
 * exact in f32, so every attention kernel reads the same numbers. A record is
 * hd * 17 / 16 bytes (272 for hd = 256: 16-byte aligned). Only the qwen35
 * kernels (hd % 32 == 0) are converted; the host refuses q8 for the others.
 * The kernels pass kc/vc as before (typed pointers to the layer's rows) and
 * the macros below use their hd, n_kv and kvd:
 *   kvrow_t r = KVROWP(kc, p, kh)   the row (record) of position p, KV head kh
 *   KVAT(r, i)                      element i of that row as a float
 *   KVLD4(kc, p, kh, e4)            elements 4 e4 .. 4 e4 + 3 of the row, f32x4
 *   KVLD4O(kc, p, kh, eo, e4)       the same from element eo (a multiple of 4) on */
#if defined(IE_KV_Q8) && defined(IE_KV_Q4)
#error "IE_KV_Q8 and IE_KV_Q4"
#endif
#if defined(IE_KV_Q8) || defined(IE_KV_Q4)
#define IE_KV_QUANT 1 /* the cache is quantized: the records of KVROWP */
#endif
/* The 4-bit KV cache (-DIE_KV_Q4, build/ie_kernels_q4.hsaco; the host picks it with IE_KV_Q4=1): for each position and
 * KV head one record of hd / 2 bytes (value i in byte i / 2, an even i in the low nibble) and then hd / 32 pairs of
 * fp16 (d, m), one pair per block of 32 values along the head dimension: value = fmaf(q, d, m) with q = 0 .. 15,
 * m = the block's minimum rounded to fp16, d = (max - min) / 15 rounded to fp16, q = round((x - m) / d) limited to
 * 0 .. 15. The value is exact in f32, so every attention kernel reads the same numbers. A record is hd * 5 / 8 bytes
 * (160 for hd = 256: 16-byte aligned). The same macros as for q8_0. */
#ifdef IE_KV_Q4
#define KVR(hd) ((hd) / 2 + (hd) / 8) /* bytes of a record */
typedef const G u8 *kvrow_t;
#define KVROWP(c, p, kh) ((void)kvd, ((const G u8 *)(c)) + ((unsigned long)(p) * n_kv + (kh)) * KVR(hd))
static inline float kvq_1(kvrow_t r, u32 hd, u32 i) {
  const G f16 *dm = (const G f16 *)(r + hd / 2) + 2u * (i >> 5);
  return __builtin_fmaf((float)(((u32)r[i >> 1] >> (4u * (i & 1u))) & 15u), (float)dm[0], (float)dm[1]);
}
static inline f32x4 kvq_4(kvrow_t r, u32 hd, u32 e4) {
  const u32 w = ((const G unsigned short *)r)[e4];
  const G f16 *dm = (const G f16 *)(r + hd / 2) + 2u * (e4 >> 3);
  const float d = (float)dm[0], m = (float)dm[1];
  return (f32x4){__builtin_fmaf((float)(w & 15u), d, m), __builtin_fmaf((float)((w >> 4) & 15u), d, m),
                 __builtin_fmaf((float)((w >> 8) & 15u), d, m), __builtin_fmaf((float)(w >> 12), d, m)};
}
#define KVAT(r, i) kvq_1(r, hd, i)
#define KVLD4(c, p, kh, e4) kvq_4(KVROWP(c, p, kh), hd, e4)
#define KVLD4O(c, p, kh, eo, e4) kvq_4(KVROWP(c, p, kh), hd, (eo) / 4u + (e4))
#elif defined(IE_KV_Q8)
#define KVR(hd) ((hd) + ((hd) >> 4)) /* bytes of a record */
typedef const G u8 *kvrow_t;
#define KVROWP(c, p, kh) ((void)kvd, ((const G u8 *)(c)) + ((unsigned long)(p) * n_kv + (kh)) * KVR(hd))
static inline float kvq_1(kvrow_t r, u32 hd, u32 i) {
  return (float)((const G i8 *)r)[i] * (float)((const G f16 *)(r + hd))[i >> 5];
}
static inline f32x4 kvq_4(kvrow_t r, u32 hd, u32 e4) {
  const u32 w = ((const G u32 *)r)[e4];
  const float s = (float)((const G f16 *)(r + hd))[e4 >> 3];
  return (f32x4){s * (float)(i8)w, s * (float)(i8)(w >> 8), s * (float)(i8)(w >> 16), s * (float)(i8)(w >> 24)};
}
#define KVAT(r, i) kvq_1(r, hd, i)
#define KVLD4(c, p, kh, e4) kvq_4(KVROWP(c, p, kh), hd, e4)
#define KVLD4O(c, p, kh, eo, e4) kvq_4(KVROWP(c, p, kh), hd, (eo) / 4u + (e4))
#else
typedef const G kvt *kvrow_t;
#define KVROWP(c, p, kh) ((c) + (unsigned long)(p) * kvd + (kh) * hd)
#define KVAT(r, i) ((r)[i])
#define KVLD4(c, p, kh, e4) KV4(c, ((unsigned long)(p) * kvd + (kh) * hd) / 4u + (e4))
#define KVLD4O(c, p, kh, eo, e4) KV4(c, ((unsigned long)(p) * kvd + (kh) * hd + (eo)) / 4u + (e4))
#endif

static inline u32 tid(void) { return __builtin_amdgcn_workitem_id_x(); }
#ifdef IE_MEGA
/* The megakernel (kernels/ie_mk.c): a workgroup runs the workgroups of the
 * kernels one after another; the interpreter writes the current one here
 * (x, y, the number of workgroups in x) before it calls the kernel body. */
/* one copy per wave (lane 0 writes it, the wave reads it): the waves need
 * not meet to pass it, so the waves of a GEMV run their rows apart */
static LDS u32 mk_vid[8][4];
static inline u32 wgid(void) { return __builtin_amdgcn_readfirstlane(mk_vid[__builtin_amdgcn_workitem_id_x() >> 5][0]); }
static inline u32 tokid(void) { return __builtin_amdgcn_readfirstlane(mk_vid[__builtin_amdgcn_workitem_id_x() >> 5][1]); }
static inline u32 ngroups_x(void) { return __builtin_amdgcn_readfirstlane(mk_vid[__builtin_amdgcn_workitem_id_x() >> 5][2]); }
#else
static inline u32 wgid(void) { return __builtin_amdgcn_workgroup_id_x(); }
/* Token index of a multi-token launch (grid dimension y; 0 for one token).
 * The kernels that take byte strides move their pointers by y strides. */
static inline u32 tokid(void) { return __builtin_amdgcn_workgroup_id_y(); }
/* the number of launched workgroups in x (the host may add idle ones): the
 * dispatch packet's grid_size_x (byte 12) over the workgroup size */
static inline u32 ngroups_x(void) {
  return ((const __attribute__((address_space(4))) u32 *)__builtin_amdgcn_dispatch_ptr())[3] / 256u;
}
#endif
static inline u32 lane(void) { return tid() & 31u; }
#define TOK(p, s) ((p) ? (__typeof__(p))((G u8 *)(p) + (unsigned long)tokid() * (s)) : (p))
#define TOKC(p, s) ((__typeof__(p))((const G u8 *)(p) + (unsigned long)tokid() * (s)))

/* Workgroup barrier with the memory ordering of LDS and global memory. */
static inline void barrier(void) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

/* Workgroup barrier for LDS only: global loads still in flight stay in
 * flight (barrier() waits for them and invalidates the L0 cache). For
 * kernels whose threads do not exchange data through global memory. */
static inline void lbarrier(void) {
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  __builtin_amdgcn_s_barrier();
  __asm__ volatile("" ::: "memory");
}

/* Value of lane (this lane + off) of the wave: ds_bpermute (byte address). */
static inline float shfl_xor_f(float v, u32 m) {
  return __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((int)((lane() ^ m) * 4u), __builtin_bit_cast(int, v)));
}

/* The wave reduction of ie_tree: step m (1..5) adds the value of lane
 * i + (32 >> m), i.e. shuffle down by 16, 8, 4, 2, 1. Lane 0 holds the sum
 * (law wave_sum). Other lanes hold partial values. */
/* The same reduction with DPP (register moves, no LDS). Step 1: lane i of
 * row 0 gets lane i + 16 (v_permlanex16 with the identity selects). Steps
 * 2-5: lane i gets lane i + 8, 4, 2, 1 inside its row of 16 (DPP row_shl;
 * a source past the row reads 0). Lane 0 adds exactly the same values in
 * the same order as with the shuffles, so its result is the same. */
static inline int xrow_i(int v) { return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false, false); }
static inline int shl_i(int v, int n) {
  switch (n) { /* the DPP control must be a constant: row_shl:n = 0x100 + n */
    case 8: return __builtin_amdgcn_update_dpp(0, v, 0x108, 0xF, 0xF, true);
    case 4: return __builtin_amdgcn_update_dpp(0, v, 0x104, 0xF, 0xF, true);
    case 2: return __builtin_amdgcn_update_dpp(0, v, 0x102, 0xF, 0xF, true);
    default: return __builtin_amdgcn_update_dpp(0, v, 0x101, 0xF, 0xF, true);
  }
}
#define F2I(x) __builtin_bit_cast(int, (x))
#define I2F(x) __builtin_bit_cast(float, (x))
static inline float wave_tree_f(float v) {
  v += I2F(xrow_i(F2I(v)));
  v += I2F(shl_i(F2I(v), 8));
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}
static inline u32 wave_tree_u(u32 v) {
  v += (u32)xrow_i((int)v);
  v += (u32)shl_i((int)v, 8);
  v += (u32)shl_i((int)v, 4);
  v += (u32)shl_i((int)v, 2);
  v += (u32)shl_i((int)v, 1);
  return v;
}
/* The maximum of a wave (v >= 0: a source past the row reads 0), in every lane. */
static inline float wave_max_pos(float v) {
  v = __builtin_fmaxf(v, I2F(xrow_i(F2I(v))));
  v = __builtin_fmaxf(v, I2F(shl_i(F2I(v), 8)));
  v = __builtin_fmaxf(v, I2F(shl_i(F2I(v), 4)));
  v = __builtin_fmaxf(v, I2F(shl_i(F2I(v), 2)));
  v = __builtin_fmaxf(v, I2F(shl_i(F2I(v), 1)));
  return I2F(__builtin_amdgcn_readlane(F2I(v), 0));
}
/* The sum of a wave, in every lane. */
static inline float wave_sum_all(float v) { return I2F(__builtin_amdgcn_readlane(F2I(wave_tree_f(v)), 0)); }

/* Workgroup sum (all 256 threads); every thread gets the result. */
static LDS float red_sum[8];
static inline float wg_sum(float v) {
  v = wave_tree_f(v);
  if (lane() == 0u) red_sum[tid() >> 5] = v;
  barrier();
  float s = 0.0f;
  for (u32 w = 0; w < 8u; w++) s += red_sum[w];
  barrier();
  return s;
}
static LDS float red_max[8];
static inline float wg_max(float v) {
  for (u32 m = 16u; m >= 1u; m >>= 1) v = __builtin_fmaxf(v, shfl_xor_f(v, m));
  if (lane() == 0u) red_max[tid() >> 5] = v;
  barrier();
  float s = red_max[0];
  for (u32 w = 1; w < 8u; w++) s = __builtin_fmaxf(s, red_max[w]);
  barrier();
  return s;
}

/* ------------------------------------------------------------------ embed */

/* out[0..cols) = row tok of a Q4 matrix (GPU layout). One workgroup. */
KERNEL ie_embed_q4(G float *out, const G u32 *qw, const G f16 *qs, u32 nb, u32 tok, float sc) {
  for (u32 e = tid(); e < nb * 32u; e += NT) {
    const u32 b = e >> 5, j = e & 31u;
    const u32 w = qw[ie_q4_dst_word(tok, b, ie_q4_word_of(j), nb)];
    out[e] = (float)qs[ie_q4_dst_scale(tok, b, nb)] * (float)((int)ie_q4_nib(w, j) - 8) * sc;
  }
}

/* out[0..nb*32) = row tok of a Q8_0 matrix (GPU layout). One workgroup. */
KERNEL ie_embed_q8(G float *out, const G u32 *qw, const G f16 *qs, u32 nb, u32 tok, float sc) {
  for (u32 e = tid(); e < nb * 32u; e += NT) {
    const u32 b = e >> 5, j = e & 31u;
    const u32 w = qw[ie_q8_dst_word(tok, b, j >> 2, nb)];
    out[e] = (float)qs[ie_q8_dst_scale(tok, b, nb)] * (float)(i8)ie_byte(w, j & 3u) * sc;
  }
}

/* out = row tok of an f32 matrix. One workgroup. */
KERNEL ie_embed_f32(G float *out, const G float *f, u32 cols, u32 tok, float sc) {
  for (u32 e = tid(); e < cols; e += NT) out[e] = f[(unsigned long)tok * cols + e] * sc;
}

/* ---------------------------------------------------------------- rmsnorm */


/* ------------------------------------------------------------- quantize */

/* Q8 activations. q buffer: int8 values [32 nb], then float d [nb], then
 * u32 asum [nb]. quant_wave: lane j of a wave holds element j of block b;
 * every lane of the wave must call it (it shuffles). The fused kernels
 * (rmsnorm, attention, swiglu) call the same function, so their Q8 copy is
 * the one ie_quant_q8 would write. */
static inline void quant_wave(float v, u32 b, G u8 *q, u32 nb) {
  const u32 j = lane();
  const float amax = wave_max_pos(__builtin_fabsf(v));
  const float d = amax / 127.0f;
  const int qi = d != 0.0f ? (int)__builtin_roundf(v / d) : 0;
  ((G i8 *)q)[b * 32u + j] = (i8)qi;
  const u32 s = wave_tree_u((u32)qi); /* asum (ie_q8_sum semantics) */
  if (j == 0u) {
    ((G float *)(q + 32u * nb))[b] = d;
    ((G u32 *)(q + 36u * nb))[b] = s;
  }
}

#ifdef IE_KV_Q4
/* Store value i of a KV head (v: this thread's value, 0 past hd) into the 4-bit record rec: wave j holds block j
 * (hd % 32 == 0), every lane of the wave must call it. d and m are rounded to fp16 first and the values use the
 * rounded numbers, so the stored numbers are exactly what the attention kernels read. */
static inline float kvq_wmax(float v) { /* the maximum of a wave (any sign), in every lane */
  for (u32 k = 16u; k >= 1u; k >>= 1) v = __builtin_fmaxf(v, shfl_xor_f(v, k));
  return v;
}
static inline void kvq_store(float v, G u8 *rec, u32 hd, u32 i) {
  const float mx = kvq_wmax(v), mn = -kvq_wmax(-v);
  const f16 dh = (f16)((mx - mn) / 15.0f), mh = (f16)mn;
  const float d = (float)dh, m = (float)mh;
  int qi = d != 0.0f ? (int)__builtin_roundf((v - m) / d) : 0;
  qi = qi > 15 ? 15 : qi < 0 ? 0 : qi;
  const int qn = (int)shfl_xor_f((float)qi, 1u); /* the value of the next lane (an even lane writes the byte) */
  if (i < hd && (lane() & 1u) == 0u) rec[i >> 1] = (u8)(qi | (qn << 4));
  if (lane() == 0u && i < hd) {
    ((G f16 *)(rec + hd / 2))[2u * (i >> 5)] = dh;
    ((G f16 *)(rec + hd / 2))[2u * (i >> 5) + 1u] = mh;
  }
}
#endif
#ifdef IE_KV_Q8
/* Store value i of a KV head (v: this thread's value, 0 past hd) into the q8
 * record rec: wave j holds block j (hd % 32 == 0), every lane of the wave must
 * call it. The scale is rounded to fp16 first and the values use the rounded
 * scale, so the stored numbers are exactly what the attention kernels read. */
static inline void kvq_store(float v, G u8 *rec, u32 hd, u32 i) {
  const float amax = wave_max_pos(__builtin_fabsf(v));
  const f16 sh = (f16)(amax / 127.0f);
  const float s = (float)sh;
  int qi = s != 0.0f ? (int)__builtin_roundf(v / s) : 0;
  qi = qi > 127 ? 127 : qi < -127 ? -127 : qi;
  if (i < hd) ((G i8 *)rec)[i] = (i8)qi;
  if (lane() == 0u && i < hd) ((G f16 *)(rec + hd))[i >> 5] = sh;
}
#endif

/* One wave per block of 32, 8 blocks per workgroup. */
KERNEL ie_quant_q8(const G float *x, G u8 *q, u32 nb, u32 xs, u32 qs) {
  x = TOKC(x, xs), q = TOK(q, qs);
  const u32 b = wgid() * 8u + (tid() >> 5);
  if (b >= nb) return; /* whole waves leave together */
  quant_wave(x[b * 32u + lane()], b, q, nb);
}

/* y = x / sqrt(mean(x^2) + eps) * w. One workgroup. n is a multiple of 32.
 * If q is not NULL, also write the Q8 copy of y (fused quantize): wave w
 * does the blocks w, w + 8, ...; each lane stores and quantizes its own y. */
static inline void rmsnorm_wg(const G float *x, const G float *w, G float *y, u32 n, float eps, G u8 *q) {
  float ss = 0.0f;
  for (u32 i = tid(); i < n; i += NT) ss += x[i] * x[i];
  ss = wg_sum(ss);
  const float s = 1.0f / __builtin_sqrtf(ss / (float)n + eps);
  const u32 nb = n >> 5;
  for (u32 b = tid() >> 5; b < nb; b += 8u) {
    const u32 i = b * 32u + lane();
    const float v = x[i] * s * w[i];
#ifdef IE_MEGA
    y[i] = v;
#else
    if (y) y[i] = v;
#endif
    if (q) quant_wave(v, b, q, nb);
  }
}
/* The RMSNorm op: ceil(n / 256) workgroups. Every workgroup computes the
 * whole sum of squares (the same code, so the same value in every
 * workgroup); then wave v of workgroup g normalizes and quantizes block
 * g * 8 + v only. n is a multiple of 32. */
KERNEL ie_rmsnorm(const G float *x, const G float *w, G float *y, u32 n, float eps, G u8 *q, u32 xs, u32 ys, u32 qs) {
  x = TOKC(x, xs), y = TOK(y, ys), q = TOK(q, qs);
  float ss = 0.0f;
  u32 i0 = tid();
  for (; i0 + 7u * NT < n; i0 += 8u * NT) { /* 8 loads in flight; the same order of the sums */
    float v[8];
#pragma unroll
    for (u32 k = 0; k < 8u; k++) v[k] = x[i0 + k * NT];
#pragma unroll
    for (u32 k = 0; k < 8u; k++) ss += v[k] * v[k];
  }
  for (; i0 < n; i0 += NT) ss += x[i0] * x[i0];
  ss = wg_sum(ss);
  const float s = 1.0f / __builtin_sqrtf(ss / (float)n + eps);
  const u32 nb = n >> 5, b = wgid() * 8u + (tid() >> 5);
  if (b < nb) { /* whole waves */
    const u32 i = b * 32u + lane();
    const float v = x[i] * s * w[i];
#ifdef IE_MEGA
    y[i] = v;
#else
    if (y) y[i] = v;
#endif
    if (q) quant_wave(v, b, q, nb);
  }
}

/* ie_rmsnorm with every load in flight at once: the loads of the sum of squares (n / 256 per thread, up to 24)
 * and this thread's x and w for the output (element wgid * 256 + tid, read here too) are issued before the
 * first add, so the kernel waits for memory once, not 3 to 5 times. The same sums in the same order, and the
 * same output arithmetic, as ie_rmsnorm. An n that is not a multiple of 256 takes ie_rmsnorm's loops. */
KERNEL ie_rmsnorm_f(const G float *x, const G float *w, G float *y, u32 n, float eps, G u8 *q, u32 xs, u32 ys, u32 qs) {
  x = TOKC(x, xs), y = TOK(y, ys), q = TOK(q, qs);
  const u32 nb = n >> 5, b = wgid() * 8u + (tid() >> 5), i = b * 32u + lane();
  float xi = 0.0f, wi = 0.0f;
  float ss = 0.0f;
  if ((n & (NT - 1u)) == 0u && n <= 24u * NT) {
    float v[24];
#pragma unroll
    for (u32 k = 0; k < 24u; k++) v[k] = k * NT < n ? x[tid() + k * NT] : 0.0f;
    if (b < nb) xi = x[i], wi = w[i];
#pragma unroll
    for (u32 k = 0; k < 24u; k++)
      if (k * NT < n) ss += v[k] * v[k];
  } else {
    u32 i0 = tid();
    for (; i0 + 7u * NT < n; i0 += 8u * NT) {
      float v[8];
#pragma unroll
      for (u32 k = 0; k < 8u; k++) v[k] = x[i0 + k * NT];
#pragma unroll
      for (u32 k = 0; k < 8u; k++) ss += v[k] * v[k];
    }
    for (; i0 < n; i0 += NT) ss += x[i0] * x[i0];
    if (b < nb) xi = x[i], wi = w[i];
  }
  ss = wg_sum(ss);
  const float s = 1.0f / __builtin_sqrtf(ss / (float)n + eps);
  if (b < nb) { /* whole waves */
    const float v = xi * s * wi;
#ifdef IE_MEGA
    y[i] = v;
#else
    if (y) y[i] = v;
#endif
    if (q) quant_wave(v, b, q, nb);
  }
}

/* ------------------------------------------------------------------ GEMV */

/* The fused RMSNorm of a GEMV result y (rows elements): every workgroup
 * makes its rows visible and counts itself; the last one (count[0] reaches
 * ngroups) computes ny = rmsnorm(y) * nw and its Q8 copy nq, then sets the
 * count back to 0 for the next GEMV. Every thread of the workgroup calls it. */
static LDS u32 gm_last;
static inline void gemv_norm_tail(const G float *y, u32 rows, const G float *nw, G float *ny, G u8 *nq, float eps,
                                  G u32 *count) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  const u32 ngroups = ngroups_x(); /* the launched workgroups (the host may add idle ones) */
  /* the counters between workgroups (here, argmax and the split attentions) are device-scope atomics between
   * the agent-scope fences: a system-scope one also wrote back and invalidated the caches per workgroup */
  if (tid() == 0u) gm_last = __scoped_atomic_fetch_add(count, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ngroups - 1u;
  barrier();
  if (!gm_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  rmsnorm_wg(y, nw, ny, rows, eps, nq);
  if (tid() == 0u) *count = 0u;
}

/* The fused epilogue of a GEMV row (the CPU backend does the same, in the
 * same order): y = acc (+ bias[r]) then (res[r] +) y. */
static inline float epilogue(float acc, u32 r, const G float *bias, const G float *res) {
  float y = acc;
  if (bias) y = y + bias[r];
  if (res) y = res[r] + y;
  return y;
}

/* GEMV_U: blocks per lane per loop step. The loads of the GEMV_U blocks are
 * all issued before the first dot product, so a lane waits for memory once
 * per step, not once per block. A block past nb loads block t (valid) again
 * and adds 0.0f. The terms are added in block order, as before (and as
 * the CPU backend does). */
#ifndef GEMV_U
#define GEMV_U 2u
#endif


/* y = W x: W is Q4 (GPU layout: words qw, f16 scales qs), x is Q8 (as
 * written by ie_quant_q8). Grid: ie_gemv_ngroups(rows) workgroups of 8
 * waves; wave v of workgroup g does row ie_gemv_row(g, v); lane l does the
 * blocks ie_lane_blk(l, t) < nb; then the ie_tree wave reduction. */
static inline void gemv_q4q8_rows(u32 g, const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows,
                                  u32 nb, const G float *bias, const G float *res) {
  const u32 r = ie_gemv_row(g, tid() >> 5), l = lane();
  if (r < rows) {
    const G u32 *aw = (const G u32 *)xq;
    const G float *da = (const G float *)(xq + 32u * nb);
    const G u32 *asum = (const G u32 *)(xq + 36u * nb);
    float acc = 0.0f;
    for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
      u32x4 w4[GEMV_U], a0[GEMV_U], a1[GEMV_U];
      float sc[GEMV_U];
      u32 as[GEMV_U];
      int ok[GEMV_U];
      for (u32 k = 0; k < GEMV_U; k++) {
        ok[k] = ie_lane_blk(l, t + k) < nb;
        const u32 b = ok[k] ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
        /* one 128-bit load of the 4 nibble words, two of the 8 activation words */
        w4[k] = *(const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb));
        a0[k] = *(const G u32x4 *)(aw + b * 8u);
        a1[k] = *(const G u32x4 *)(aw + b * 8u + 4u);
        sc[k] = (float)qs[ie_q4_dst_scale(r, b, nb)] * da[b];
        as[k] = asum[b];
      }
      for (u32 k = 0; k < GEMV_U; k++) {
        /* no branch: a branch lets the compiler sink the loads into it */
        const u32 wq[4] = {w4[k].x, w4[k].y, w4[k].z, w4[k].w};
        const u32 av[8] = {a0[k].x, a0[k].y, a0[k].z, a0[k].w, a1[k].x, a1[k].y, a1[k].z, a1[k].w};
        const Mem qm = {wq}, am = {av};
        const int dot = (int)ie_q4q8_block(qm, 0u, am, 0u, as[k]);
        acc += ok[k] ? sc[k] * (float)dot : 0.0f; /* + 0.0f leaves acc unchanged */
      }
    }
    acc = wave_tree_f(acc);
    if (l == 0u) y[r] = epilogue(acc, r, bias, res);
  }
}

KERNEL ie_gemv_q4q8(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                  const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps,
                  G u32 *count) {
  gemv_q4q8_rows(wgid(), qw, qs, xq, y, rows, nb, bias, res);
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

/* y = W x: W is MAT_Q4K (GGUF Q4_K repacked, src/model.h: the nibbles as
 * Q4_0 blocks, values 0..15; qs: (scale | min << 8) per sub-block of 32,
 * then (f16 d, f16 dmin) per super-block of 256), x is Q8. The same grid,
 * lanes and reduction as ie_gemv_q4q8. Per sub-block b:
 *   dq = sum q a = ie_q4k_dot (law q4k_dot),
 *   acc += da * ((d sc) dq - (dmin mn) asum)   (no fused multiply-add).
 * ie_gemv_q4kq8_tr does the same arithmetic in the same order per token. */
/* The rows of workgroup g (a task of the megakernel: g need not be wgid()). */
static inline void gemv_q4kq8_rows(u32 g, const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows,
                                   u32 nb, const G float *bias, const G float *res) {
#pragma clang fp contract(off)
  const u32 r = ie_gemv_row(g, tid() >> 5), l = lane();
  if (r < rows) {
    const G u32 *aw = (const G u32 *)xq;
    const G float *da = (const G float *)(xq + 32u * nb);
    const G u32 *asum = (const G u32 *)(xq + 36u * nb);
    const G u16 *smr = qs + ie_q4k_sm(r, 0u, nb);
    const G f16 *ddr = (const G f16 *)qs + ie_q4k_dd(rows, r, 0u, nb);
    float acc = 0.0f;
    for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
      u32x4 w4[GEMV_U], a0[GEMV_U], a1[GEMV_U];
      u32 bk[GEMV_U], sm[GEMV_U], as[GEMV_U];
      float dv[GEMV_U], mv[GEMV_U], dav[GEMV_U];
      int ok[GEMV_U];
      for (u32 k = 0; k < GEMV_U; k++) { /* all loads of the step first */
        ok[k] = ie_lane_blk(l, t + k) < nb;
        bk[k] = ok[k] ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
        w4[k] = *(const G u32x4 *)(qw + ie_q4_dst_word(r, bk[k], 0u, nb));
        a0[k] = *(const G u32x4 *)(aw + bk[k] * 8u);
        a1[k] = *(const G u32x4 *)(aw + bk[k] * 8u + 4u);
        sm[k] = smr[bk[k]];
        dv[k] = (float)ddr[(bk[k] >> 3) * 2u], mv[k] = (float)ddr[(bk[k] >> 3) * 2u + 1u];
        as[k] = asum[bk[k]], dav[k] = da[bk[k]];
      }
      for (u32 k = 0; k < GEMV_U; k++) {
        const u32 wq[4] = {w4[k].x, w4[k].y, w4[k].z, w4[k].w};
        const u32 av[8] = {a0[k].x, a0[k].y, a0[k].z, a0[k].w, a1[k].x, a1[k].y, a1[k].z, a1[k].w};
        const Mem qm = {wq}, am = {av};
        const int dq = (int)ie_q4k_dot(qm, 0u, am, 0u); /* sum q a (law q4k_dot) */
        /* the arithmetic of q4k_term */
        const float sc = dv[k] * (float)(sm[k] & 63u), mm = mv[k] * (float)(sm[k] >> 8);
        /* acc = fma(sc dq, da, acc); acc = fma(-mm, da asum, acc) (explicit fma: the same in every Q4K kernel) */
        const float x = dav[k] * (float)(int)as[k];
        const float a1 = __builtin_fmaf(sc * (float)dq, dav[k], acc), a2 = __builtin_fmaf(-mm, x, a1);
        acc = ok[k] ? a2 : acc;
      }
    }
    acc = wave_tree_f(acc);
    if (l == 0u) y[r] = epilogue(acc, r, bias, res);
  }
}

KERNEL ie_gemv_q4kq8(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                     const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps,
                     G u32 *count) {
  gemv_q4kq8_rows(wgid(), qw, qs, xq, y, rows, nb, bias, res);
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

/* Probe for the megakernel: a persistent grid (any size) loops over the
 * workgroups of ie_gemv_q4kq8 (no fused norm). The same rows, the same
 * arithmetic: the output is bitwise equal. */
KERNEL ie_gemv_q4kq8_p(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                       const G float *bias, const G float *res) {
  const u32 ng = ie_gemv_ngroups(rows), np = __builtin_amdgcn_grid_size_x() / 256u;
  for (u32 g = wgid(); g < ng; g += np) gemv_q4kq8_rows(g, qw, qs, xq, y, rows, nb, bias, res);
}

/* Grid barrier of the persistent kernels (the megakernel, kernels/ie_mk.c).
 * ws[0]: arrival counter, never reset: each workgroup adds 1 and waits until
 * it reaches target (wrapping u32 compare). ws[1]: abort flag. Every wait is
 * bounded by a 250 ms real GPU wall-clock guard (s_sendmsg_rtnl 0x83, 100 MHz
 * clock, checked every 256 spins): on timeout the workgroup sets the abort flag
 * and every workgroup leaves. Returns 0 on abort. Every thread calls it. */
#ifndef MK_SPIN
#define MK_SPIN 400000u
#endif
#ifndef MK_TIMEOUT_TICKS
#define MK_TIMEOUT_TICKS 25000000ull /* 250 ms on 100 MHz GPU wall clock */
#endif
static LDS u32 mk_ok;
#ifndef MK_PFCHUNK
#define MK_PFCHUNK 16384u
#endif
/* While it waits, the workgroup prefetches (s_prefetch_data: no registers,
 * no wait) the next chunks of [pa, pa + cap) into the caches: chunks of
 * MK_PFCHUNK bytes claimed with an atomic add on *cur (cap 0: no prefetch). */
static inline int mk_grid_sync_pf(G u32 *ws, u32 target, G u32 *cur, const G u8 *pa, u32 cap) {
  /* every wave makes its stores visible to the device, then the workgroup
   * meets, then one thread counts it (a release by thread 0 alone does not
   * cover the stores of the other waves) */
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    __scoped_atomic_fetch_add(&ws[0], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    const unsigned long t0 = __builtin_amdgcn_s_sendmsg_rtnl(0x83);
    u32 r = 0u, pf = cap != 0u;
    for (u32 i = 0u;; i++) {
      if ((int)(__scoped_atomic_load_n(&ws[0], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) - target) >= 0) { r = 1u; break; }
      if ((i & 31u) == 31u && __scoped_atomic_load_n(&ws[1], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)) break;
      if ((i & 255u) == 255u && (__builtin_amdgcn_s_sendmsg_rtnl(0x83) - t0) > MK_TIMEOUT_TICKS) break;
      if (pf) {
        const u32 c = __builtin_amdgcn_readfirstlane(
            __scoped_atomic_fetch_add(cur, MK_PFCHUNK, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE));
        if (c >= cap) pf = 0u;
        else
#if __has_builtin(__builtin_amdgcn_s_prefetch_data) /* clang 19+ */
          for (u32 o = c; o < c + MK_PFCHUNK && o < cap; o += 256u) __builtin_amdgcn_s_prefetch_data(pa + o, 31);
#else
          (void)pa;
#endif
      }
    }
    if (!r) __scoped_atomic_store_n(&ws[1], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    mk_ok = r;
  }
  barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  return (int)mk_ok;
}
static inline int mk_grid_sync(G u32 *ws, u32 target) { return mk_grid_sync_pf(ws, target, ws, 0, 0u); }

/* Probe: n GEMVs (MAT_Q4K) one after another in one launch, a grid barrier
 * between two. d: n descriptors of 8 words {qw lo, qw hi, qs lo, qs hi,
 * y lo, y hi, rows, nb}; all read the activations xq. */
KERNEL ie_mk_gemv_chain(const G u32 *d, u32 n, const G u8 *xq, G u32 *ws) {
  const u32 np = __builtin_amdgcn_grid_size_x() / 256u;
  for (u32 i = 0; i < n; i++) {
    const G u32 *e = d + 8u * i;
    const G u32 *qw = (const G u32 *)(((unsigned long)e[1] << 32) | e[0]);
    const G u16 *qs = (const G u16 *)(((unsigned long)e[3] << 32) | e[2]);
    G float *y = (G float *)(((unsigned long)e[5] << 32) | e[4]);
    const u32 rows = e[6], nb = e[7], ng = ie_gemv_ngroups(rows);
    for (u32 g = wgid(); g < ng; g += np) gemv_q4kq8_rows(g, qw, qs, xq, y, rows, nb, 0, 0);
    if (i + 1u < n && !mk_grid_sync(ws, np * (i + 1u))) return;
  }
}

/* MAT_Q4K for T <= 16 tokens: 4 rows per wave (as ie_gemv_q8q8_tr); per
 * (row, token) the arithmetic and order of ie_gemv_q4kq8 (bitwise equal). */
#define Q4K_TR_BODY(QR, QT, QU) \
_Pragma("clang fp contract(off)") \
  const u32 r0 = (wgid() * 8u + (tid() >> 5)) * QR, l = lane(); \
  if (r0 >= rows) return; \
  u32 xso_ = xs, yso_ = ys, rso_ = rs; \
  if ((QT) > 8u) __asm__ volatile("" : "+v"(xso_), "+v"(yso_), "+v"(rso_)); \
  float acc[QR][QT]; \
_Pragma("unroll") \
  for (u32 q = 0; q < QR; q++) \
_Pragma("unroll") \
    for (u32 j = 0; j < QT; j++) acc[q][j] = 0.0f; \
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t += QU) { \
_Pragma("unroll") \
  for (u32 u = 0; u < QU; u++) { /* QU blocks: the loads of all go out first */ \
    const int okb = ie_lane_blk(l, t + u) < nb; \
    const u32 b = okb ? ie_lane_blk(l, t + u) : ie_lane_blk(l, t); \
    /* per row, once per block: the 32 weights as int8 words (weight j = \
     * byte j % 4 of word j / 4, as the activations) and the two scales */ \
    u32 wv[QR][8]; \
    float sc[QR], mm[QR]; \
    if (!okb) continue; /* past nb (only some lanes, at the end) */ \
_Pragma("unroll") \
    for (u32 q = 0; q < QR; q++) { \
      const u32 r = r0 + q < rows ? r0 + q : r0; /* a row past the end: row r0 again, not stored */ \
      const u32x4 w4 = __builtin_nontemporal_load((const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb))); \
      const u32 x4[4] = {w4.x, w4.y, w4.z, w4.w}; \
_Pragma("unroll") \
      for (u32 k = 0; k < 4u; k++) wv[q][2u * k] = ie_q4k_lo(x4[k]), wv[q][2u * k + 1u] = ie_q4k_hi(x4[k]); /* once per block */ \
      const u32 sm = qs[ie_q4k_sm(r, b, nb)]; \
      const u32 dd = ie_q4k_dd(rows, r, b, nb); \
      sc[q] = (float)((const G f16 *)qs)[dd] * (float)(sm & 63u); \
      mm[q] = (float)((const G f16 *)qs)[dd + 1u] * (float)(sm >> 8); \
    } \
_Pragma("unroll") \
    for (u32 j = 0; j < QT; j++) { \
      if (j < T) { /* uniform */ \
        const G u8 *xj = xq + (unsigned long)j * xso_; \
        const G u32 *aw = (const G u32 *)xj; \
        const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u); \
        const float da = ((const G float *)(xj + 32u * nb))[b]; \
        const int as = (int)((const G u32 *)(xj + 36u * nb))[b]; \
        const float xa = da * (float)as; /* per token: shared by the rows */ \
        const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}; \
_Pragma("unroll") \
        for (u32 q = 0; q < QR; q++) { \
          const Mem um = {wv[q]}, am = {av}; \
          const int dq = (int)ie_q4k_dot_u(um, 0u, am, 0u); /* sum q a (laws q4k_dot_u, q4k_dot) */ \
          acc[q][j] = __builtin_fmaf(-mm[q], xa, __builtin_fmaf(sc[q] * (float)dq, da, acc[q][j])); \
        } \
      } \
    } \
  } \
  } \
_Pragma("unroll") \
  for (u32 q = 0; q < QR; q++) { \
    const u32 r = r0 + q; \
_Pragma("unroll") \
    for (u32 j = 0; j < QT; j++) { \
      if (j < T) { \
        const float v = wave_tree_f(acc[q][j]); \
        if (l == 0u && r < rows) y[(unsigned long)j * yso_ + r] = epilogue(v, r, bias, res ? res + (unsigned long)j * rso_ : res); \
      } \
    } \
  }
#define Q4K_TS_BODY(QR, QT, QU, TS) \
_Pragma("clang fp contract(off)") \
  const u32 r0 = (wgid() * (8u / TS) + (tid() >> 5) / TS) * QR, l = lane(); \
  if (r0 >= rows) return; \
  u32 SK_; \
  { /* every wave computes qn_ = ceil(T / TS) tokens (the waves of a row group then read the same weights in step, \
     * the second from the cache: 4 + 2 tokens measured slower than 4 + 4); it owns tokens own_ .. own_ + qn_ - 1; the \
     * last wave starts at T - qn_ instead (it computes some tokens of the wave before again: the same arithmetic) \
     * and stores only its own (SK_: the ones it skips) */ \
    const u32 qn_ = (T + TS - 1u) / TS, own_ = ((tid() >> 5) % TS) * qn_; \
    if (own_ >= T) return; \
    const u32 j0_ = own_ < T - qn_ ? own_ : T - qn_; \
    SK_ = own_ - j0_; \
    xq += j0_ * xs, y += j0_ * ys, res = res ? res + j0_ * rs : res, T = qn_; } \
  if (T == 0u) return; \
  float acc[QR][QT]; \
_Pragma("unroll") \
  for (u32 q = 0; q < QR; q++) \
_Pragma("unroll") \
    for (u32 j = 0; j < QT; j++) acc[q][j] = 0.0f; \
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t += QU) { \
_Pragma("unroll") \
  for (u32 u = 0; u < QU; u++) { /* QU blocks: the loads of all go out first */ \
    const int okb = ie_lane_blk(l, t + u) < nb; \
    const u32 b = okb ? ie_lane_blk(l, t + u) : ie_lane_blk(l, t); \
    /* per row, once per block: the 32 weights as int8 words (weight j = \
     * byte j % 4 of word j / 4, as the activations) and the two scales */ \
    u32 wv[QR][8]; \
    float sc[QR], mm[QR]; \
    if (!okb) continue; /* past nb (only some lanes, at the end) */ \
_Pragma("unroll") \
    for (u32 q = 0; q < QR; q++) { \
      const u32 r = r0 + q < rows ? r0 + q : r0; /* a row past the end: row r0 again, not stored */ \
      const u32x4 w4 = __builtin_nontemporal_load((const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb))); \
      const u32 x4[4] = {w4.x, w4.y, w4.z, w4.w}; \
_Pragma("unroll") \
      for (u32 k = 0; k < 4u; k++) wv[q][2u * k] = ie_q4k_lo(x4[k]), wv[q][2u * k + 1u] = ie_q4k_hi(x4[k]); /* once per block */ \
      const u32 sm = qs[ie_q4k_sm(r, b, nb)]; \
      const u32 dd = ie_q4k_dd(rows, r, b, nb); \
      sc[q] = (float)((const G f16 *)qs)[dd] * (float)(sm & 63u); \
      mm[q] = (float)((const G f16 *)qs)[dd + 1u] * (float)(sm >> 8); \
    } \
_Pragma("unroll") \
    for (u32 j = 0; j < QT; j++) { \
      if (j < T) { /* uniform */ \
        const G u8 *xj = xq + j * xs; \
        const G u32 *aw = (const G u32 *)xj; \
        const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u); \
        const float da = ((const G float *)(xj + 32u * nb))[b]; \
        const int as = (int)((const G u32 *)(xj + 36u * nb))[b]; \
        const float xa = da * (float)as; /* per token: shared by the rows */ \
        const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}; \
_Pragma("unroll") \
        for (u32 q = 0; q < QR; q++) { \
          const Mem um = {wv[q]}, am = {av}; \
          const int dq = (int)ie_q4k_dot_u(um, 0u, am, 0u); /* sum q a (laws q4k_dot_u, q4k_dot) */ \
          acc[q][j] = __builtin_fmaf(-mm[q], xa, __builtin_fmaf(sc[q] * (float)dq, da, acc[q][j])); \
        } \
      } \
    } \
  } \
  } \
_Pragma("unroll") \
  for (u32 q = 0; q < QR; q++) { \
    const u32 r = r0 + q; \
_Pragma("unroll") \
    for (u32 j = 0; j < QT; j++) { \
      if (j < T && j >= SK_) { \
        const float v = wave_tree_f(acc[q][j]); \
        if (l == 0u && r < rows) y[j * ys + r] = epilogue(v, r, bias, res ? res + j * rs : res); \
      } \
    } \
  }
/* 6 <= T <= 8 tokens split over 2 waves of the workgroup (4 tokens per
 * wave): 8 rows per wave, 32 rows per workgroup, twice the workgroups of
 * ie_gemv_q4kq8_tr8 (more waves in flight: 6 to 14% faster at T = 8 on the
 * shapes of Qwen3.8-27B). The two waves of a row group read the same
 * weights (the second from the cache). Per (row, token) the order of
 * ie_gemv_q4kq8 (bitwise equal). */
KERNEL_HEAVY ie_gemv_q4kq8_ts2(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                         const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  Q4K_TS_BODY(8u, 4u, 1u, 2u)
}
KERNEL_HEAVY ie_gemv_q4kq8_tr(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                        const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
#ifdef MKS_INKLING_VER
  for (u32 T_rem_ = T; T_rem_ > 0u; ) {
    T = T_rem_ < 8u ? T_rem_ : 8u;
    {
      Q4K_TR_BODY(4u, 8u, 1u)
    }
    if (T_rem_ <= 8u) break;
    T_rem_ -= 8u;
    xq += 8ul * xs;
    y += 8ul * ys;
    if (res) res += 8ul * rs;
  }
#else
  Q4K_TR_BODY(4u, 16u, 2u)
#endif
}
/* 9 <= T <= 16 (a verify of up to 15 drafts): 8 rows per wave as ie_gemv_q4kq8_ts2, the tokens split over 2
 * waves of 8 (ts16) or 4 waves of 4 (ts4: 16 rows per workgroup). ie_gemv_q4kq8_tr (4 rows per wave) loads the
 * activations of all 16 tokens for every 4 rows: at 16 tokens those loads, not the weights, set its speed.
 * Per (row, token) the order of ie_gemv_q4kq8 (bitwise equal). */
KERNEL_HEAVY ie_gemv_q4kq8_ts16(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                          const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  Q4K_TS_BODY(8u, 8u, 1u, 2u)
}
KERNEL_HEAVY ie_gemv_q4kq8_ts4(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                         const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  Q4K_TS_BODY(8u, 4u, 1u, 4u)
}
/* T <= 8 (speculative verify): 8 rows per wave, half the activation loads per weight byte */
KERNEL_HEAVY ie_gemv_q4kq8_tr8(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                         const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  Q4K_TR_BODY(8u, 8u, 1u)
}

/* y = W x: W is MAT_Q6K (GGUF Q6_K repacked, src/model.h), x is Q8. The
 * same grid, lanes and reduction as ie_gemv_q4q8. Per sub-block b: the half
 * dot products d0, d1 (ie_q6k_dots, law q6k_dots: the q - 32 bytes, signed dot4), then
 *   acc = fma(fma(d sc1, d1, (d sc0) d0), da, acc)
 * (explicit fma: the same in ie_gemv_q6kq8_tr and the CPU). */
static inline void gemv_q6kq8_rows(u32 g, const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                                   const G float *bias, const G float *res) {
#pragma clang fp contract(off)
  const u32 r = ie_gemv_row(g, tid() >> 5), l = lane();
  if (r < rows) {
    const G u32 *aw = (const G u32 *)xq;
    const G float *da = (const G float *)(xq + 32u * nb);
    float acc = 0.0f;
    for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
      u32x4 w4[GEMV_U], a0[GEMV_U], a1[GEMV_U];
      u32 h0[GEMV_U], h1[GEMV_U], sc[GEMV_U];
      float dv[GEMV_U], dav[GEMV_U];
      int ok[GEMV_U];
      for (u32 k = 0; k < GEMV_U; k++) { /* all loads of the step first */
        ok[k] = ie_lane_blk(l, t + k) < nb;
        const u32 b = ok[k] ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
        w4[k] = *(const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb));
        const u32 hb = ie_q6k_hw(rows, r, b, 0u, nb);
        h0[k] = qw[hb], h1[k] = qw[hb + 1u];
        a0[k] = *(const G u32x4 *)(aw + b * 8u);
        a1[k] = *(const G u32x4 *)(aw + b * 8u + 4u);
        sc[k] = qs[ie_q6k_sc(r, b, nb)];
        dv[k] = (float)((const G f16 *)qs)[ie_q6k_d(rows, r, b, nb)];
        dav[k] = da[b];
      }
      for (u32 k = 0; k < GEMV_U; k++) {
        const u32 wq[4] = {w4[k].x, w4[k].y, w4[k].z, w4[k].w}, hq[2] = {h0[k], h1[k]};
        const u32 av[8] = {a0[k].x, a0[k].y, a0[k].z, a0[k].w, a1[k].x, a1[k].y, a1[k].z, a1[k].w};
        const Mem qm = {wq}, hm = {hq}, am = {av};
        const int d0 = (int)ie_q6k_dots(qm, 0u, hm, 0u, am, 0u, 0u); /* law q6k_dots */
        const int d1 = (int)ie_q6k_dots(qm, 0u, hm, 1u, am, 0u, 1u);
        const float s0 = dv[k] * (float)(int)(signed char)(sc[k] & 255u), s1 = dv[k] * (float)(int)(signed char)(sc[k] >> 8);
        const float a2 = __builtin_fmaf(__builtin_fmaf(s1, (float)d1, s0 * (float)d0), dav[k], acc);
        acc = ok[k] ? a2 : acc;
      }
    }
    acc = wave_tree_f(acc);
    if (l == 0u) y[r] = epilogue(acc, r, bias, res);
  }
}
KERNEL ie_gemv_q6kq8(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                     const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps,
                     G u32 *count) {
  gemv_q6kq8_rows(wgid(), qw, qs, xq, y, rows, nb, bias, res);
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

/* MAT_Q6K for T <= QT tokens, QR rows per wave (as the Q4K _tr kernels):
 * per (row, token) the arithmetic and order of ie_gemv_q6kq8 (bitwise
 * equal). The q - 32 bytes of a row are common to the tokens: computed once. */
#define Q6K_TR_BODY(QR, QT) \
  _Pragma("clang fp contract(off)") \
  const u32 r0 = (wgid() * 8u + (tid() >> 5)) * QR, l = lane(); \
  if (r0 >= rows) return; \
  float acc[QR][QT]; \
  _Pragma("unroll") for (u32 q = 0; q < QR; q++) \
    _Pragma("unroll") for (u32 j = 0; j < QT; j++) acc[q][j] = 0.0f; \
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) { \
    const u32 b = ie_lane_blk(l, t); \
    u32 sv[QR][2][4]; /* the q - 32 words of each half, made once per block */ \
    float s0[QR], s1[QR]; \
    _Pragma("unroll") for (u32 q = 0; q < QR; q++) { \
      const u32 r = r0 + q < rows ? r0 + q : r0; /* a row past the end: row r0 again, not stored */ \
      const u32x4 w4 = __builtin_nontemporal_load((const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb))); \
      const u32 x4[4] = {w4.x, w4.y, w4.z, w4.w}; \
      const u32 hb = ie_q6k_hw(rows, r, b, 0u, nb); \
      const u32 hq[2] = {__builtin_nontemporal_load(qw + hb), __builtin_nontemporal_load(qw + hb + 1u)}; \
      _Pragma("unroll") for (u32 hh = 0; hh < 2u; hh++) \
        _Pragma("unroll") for (u32 k = 0; k < 4u; k++) sv[q][hh][k] = ie_q6k_sw(x4[k], hq[hh], hh, k); \
      const u32 sc = qs[ie_q6k_sc(r, b, nb)]; \
      const float d = (float)((const G f16 *)qs)[ie_q6k_d(rows, r, b, nb)]; \
      s0[q] = d * (float)(int)(signed char)(sc & 255u), s1[q] = d * (float)(int)(signed char)(sc >> 8); \
    } \
    _Pragma("unroll") for (u32 j = 0; j < QT; j++) { \
      if (j < T) { /* uniform */ \
        const G u8 *xj = xq + j * xs; \
        const G u32 *aw = (const G u32 *)xj; \
        const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u); \
        const float da = ((const G float *)(xj + 32u * nb))[b]; \
        const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}; \
        const Mem am = {av}; \
        _Pragma("unroll") for (u32 q = 0; q < QR; q++) { \
          const Mem s0m = {sv[q][0]}, s1m = {sv[q][1]}; \
          const int d0 = (int)ie_q6k_dots_s(s0m, 0u, am, 0u, 0u); /* laws q6k_dots_s, q6k_dots */ \
          const int d1 = (int)ie_q6k_dots_s(s1m, 0u, am, 0u, 1u); \
          acc[q][j] = __builtin_fmaf(__builtin_fmaf(s1[q], (float)d1, s0[q] * (float)d0), da, acc[q][j]); \
        } \
      } \
    } \
  } \
  _Pragma("unroll") for (u32 q = 0; q < QR; q++) { \
    const u32 r = r0 + q; \
    _Pragma("unroll") for (u32 j = 0; j < QT; j++) { \
      if (j < T) { \
        const float v = wave_tree_f(acc[q][j]); \
        if (l == 0u && r < rows) y[j * ys + r] = epilogue(v, r, bias, res ? res + j * rs : res); \
      } \
    } \
  }
KERNEL_HEAVY ie_gemv_q6kq8_tr(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                        const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
#ifdef MKS_INKLING_VER
  for (u32 T_rem_ = T; T_rem_ > 0u; ) {
    T = T_rem_ < 8u ? T_rem_ : 8u;
    {
      Q6K_TR_BODY(4u, 8u)
    }
    if (T_rem_ <= 8u) break;
    T_rem_ -= 8u;
    xq += 8ul * xs;
    y += 8ul * ys;
    if (res) res += 8ul * rs;
  }
#else
  Q6K_TR_BODY(4u, 16u)
#endif
}
KERNEL_HEAVY ie_gemv_q6kq8_tr8(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                         const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  Q6K_TR_BODY(8u, 8u)
}
#define Q6K_TS_BODY(QR, QT, TS) \
  _Pragma("clang fp contract(off)") \
  const u32 r0 = (wgid() * (8u / TS) + (tid() >> 5) / TS) * QR, l = lane(); \
  if (r0 >= rows) return; \
  u32 SK_; \
  { /* every wave computes qn_ = ceil(T / TS) tokens (the waves of a row group then read the same weights in step, \
     * the second from the cache: 4 + 2 tokens measured slower than 4 + 4); it owns tokens own_ .. own_ + qn_ - 1; the \
     * last wave starts at T - qn_ instead (it computes some tokens of the wave before again: the same arithmetic) \
     * and stores only its own (SK_: the ones it skips) */ \
    const u32 qn_ = (T + TS - 1u) / TS, own_ = ((tid() >> 5) % TS) * qn_; \
    if (own_ >= T) return; \
    const u32 j0_ = own_ < T - qn_ ? own_ : T - qn_; \
    SK_ = own_ - j0_; \
    xq += j0_ * xs, y += j0_ * ys, res = res ? res + j0_ * rs : res, T = qn_; } \
  if (T == 0u) return; \
  float acc[QR][QT]; \
  _Pragma("unroll") for (u32 q = 0; q < QR; q++) \
    _Pragma("unroll") for (u32 j = 0; j < QT; j++) acc[q][j] = 0.0f; \
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) { \
    const u32 b = ie_lane_blk(l, t); \
    u32 sv[QR][2][4]; /* the q - 32 words of each half, made once per block */ \
    float s0[QR], s1[QR]; \
    _Pragma("unroll") for (u32 q = 0; q < QR; q++) { \
      const u32 r = r0 + q < rows ? r0 + q : r0; /* a row past the end: row r0 again, not stored */ \
      const u32x4 w4 = __builtin_nontemporal_load((const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb))); \
      const u32 x4[4] = {w4.x, w4.y, w4.z, w4.w}; \
      const u32 hb = ie_q6k_hw(rows, r, b, 0u, nb); \
      const u32 hq[2] = {__builtin_nontemporal_load(qw + hb), __builtin_nontemporal_load(qw + hb + 1u)}; \
      _Pragma("unroll") for (u32 hh = 0; hh < 2u; hh++) \
        _Pragma("unroll") for (u32 k = 0; k < 4u; k++) sv[q][hh][k] = ie_q6k_sw(x4[k], hq[hh], hh, k); \
      const u32 sc = qs[ie_q6k_sc(r, b, nb)]; \
      const float d = (float)((const G f16 *)qs)[ie_q6k_d(rows, r, b, nb)]; \
      s0[q] = d * (float)(int)(signed char)(sc & 255u), s1[q] = d * (float)(int)(signed char)(sc >> 8); \
    } \
    _Pragma("unroll") for (u32 j = 0; j < QT; j++) { \
      if (j < T) { /* uniform */ \
        const G u8 *xj = xq + j * xs; \
        const G u32 *aw = (const G u32 *)xj; \
        const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u); \
        const float da = ((const G float *)(xj + 32u * nb))[b]; \
        const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}; \
        const Mem am = {av}; \
        _Pragma("unroll") for (u32 q = 0; q < QR; q++) { \
          const Mem s0m = {sv[q][0]}, s1m = {sv[q][1]}; \
          const int d0 = (int)ie_q6k_dots_s(s0m, 0u, am, 0u, 0u); /* laws q6k_dots_s, q6k_dots */ \
          const int d1 = (int)ie_q6k_dots_s(s1m, 0u, am, 0u, 1u); \
          acc[q][j] = __builtin_fmaf(__builtin_fmaf(s1[q], (float)d1, s0[q] * (float)d0), da, acc[q][j]); \
        } \
      } \
    } \
  } \
  _Pragma("unroll") for (u32 q = 0; q < QR; q++) { \
    const u32 r = r0 + q; \
    _Pragma("unroll") for (u32 j = 0; j < QT; j++) { \
      if (j < T && j >= SK_) { \
        const float v = wave_tree_f(acc[q][j]); \
        if (l == 0u && r < rows) y[j * ys + r] = epilogue(v, r, bias, res ? res + j * rs : res); \
      } \
    } \
  }
/* 6 <= T <= 8: the tokens split over 2 waves (as ie_gemv_q4kq8_ts2); bitwise equal to ie_gemv_q6kq8 */
KERNEL_HEAVY ie_gemv_q6kq8_ts2(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                         const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  Q6K_TS_BODY(8u, 4u, 2u)
}
/* 9 <= T <= 16: as ie_gemv_q4kq8_ts16 / _ts4; bitwise equal to ie_gemv_q6kq8 */
KERNEL_HEAVY ie_gemv_q6kq8_ts16(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                          const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  Q6K_TS_BODY(8u, 8u, 2u)
}
KERNEL_HEAVY ie_gemv_q6kq8_ts4(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                         const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  Q6K_TS_BODY(8u, 4u, 4u)
}

/* y = W x: W is Q8_0 (GPU layout: 8 int8 words per block in qw, f16 scales
 * qs), x is Q8. The same grid, lane split and reduction as ie_gemv_q4q8;
 * the block dot product is ie_q8q8_block (8 v_dot4_i32_iu8, both signed). */
static inline void gemv_q8q8_rows(u32 g, const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                                   const G float *bias, const G float *res) {
  const u32 r = ie_gemv_row(g, tid() >> 5), l = lane();
  if (r < rows) {
    const G u32 *aw = (const G u32 *)xq;
    const G float *da = (const G float *)(xq + 32u * nb);
    float acc = 0.0f;
    for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
      u32x4 w0[GEMV_U], w1[GEMV_U], a0[GEMV_U], a1[GEMV_U];
      float sc[GEMV_U];
      int ok[GEMV_U];
      for (u32 k = 0; k < GEMV_U; k++) {
        ok[k] = ie_lane_blk(l, t + k) < nb;
        const u32 b = ok[k] ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
        /* two 128-bit loads of the 8 weight words, two of the 8 activation words */
        const G u32 *wp = qw + ie_q8_dst_word(r, b, 0u, nb);
        w0[k] = *(const G u32x4 *)wp;
        w1[k] = *(const G u32x4 *)(wp + 4u);
        a0[k] = *(const G u32x4 *)(aw + b * 8u);
        a1[k] = *(const G u32x4 *)(aw + b * 8u + 4u);
        sc[k] = (float)qs[ie_q8_dst_scale(r, b, nb)] * da[b];
      }
      for (u32 k = 0; k < GEMV_U; k++) {
        const u32 wq[8] = {w0[k].x, w0[k].y, w0[k].z, w0[k].w, w1[k].x, w1[k].y, w1[k].z, w1[k].w};
        const u32 av[8] = {a0[k].x, a0[k].y, a0[k].z, a0[k].w, a1[k].x, a1[k].y, a1[k].z, a1[k].w};
        const Mem qm = {wq}, am = {av};
        const int dot = (int)ie_q8q8_block(qm, 0u, am, 0u);
        acc += ok[k] ? sc[k] * (float)dot : 0.0f;
      }
    }
    acc = wave_tree_f(acc);
    if (l == 0u) y[r] = epilogue(acc, r, bias, res);
  }
}
KERNEL ie_gemv_q8q8(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                     const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps,
                     G u32 *count) {
  gemv_q8q8_rows(wgid(), qw, qs, xq, y, rows, nb, bias, res);
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

/* The same GEMV for T <= 16 tokens (speculative decoding: verify several
 * tokens in one pass; prompt processing in chunks). Each weight block is loaded once and used for every
 * token. For each token the arithmetic and its order are those of
 * ie_gemv_q8q8, so token t's result is bitwise the result of a one-token
 * launch. xs: bytes between the Q8 activations of two tokens; ys: floats
 * between the outputs (and the residuals) of two tokens. */
#define GEMV_T 16u
KERNEL ie_gemv_q8q8_t(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                      const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  const u32 r = ie_gemv_row(wgid(), tid() >> 5), l = lane();
  if (r >= rows) return;
  float acc[GEMV_T];
  for (u32 j = 0; j < GEMV_T; j++) acc[j] = 0.0f;
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
    u32x4 w0[GEMV_U], w1[GEMV_U];
    float sw[GEMV_U];
    int ok[GEMV_U];
    u32 bk[GEMV_U];
    for (u32 k = 0; k < GEMV_U; k++) {
      ok[k] = ie_lane_blk(l, t + k) < nb;
      const u32 b = ok[k] ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
      bk[k] = b;
      const G u32 *wp = qw + ie_q8_dst_word(r, b, 0u, nb);
      w0[k] = *(const G u32x4 *)wp;
      w1[k] = *(const G u32x4 *)(wp + 4u);
      sw[k] = (float)qs[ie_q8_dst_scale(r, b, nb)];
    }
    for (u32 j = 0; j < GEMV_T; j++) {
      if (j >= T) break; /* uniform */
      const G u8 *xj = xq + j * xs;
      const G u32 *aw = (const G u32 *)xj;
      const G float *da = (const G float *)(xj + 32u * nb);
      u32x4 a0[GEMV_U], a1[GEMV_U];
      float sc[GEMV_U];
      for (u32 k = 0; k < GEMV_U; k++) {
        a0[k] = *(const G u32x4 *)(aw + bk[k] * 8u);
        a1[k] = *(const G u32x4 *)(aw + bk[k] * 8u + 4u);
        sc[k] = sw[k] * da[bk[k]];
      }
      for (u32 k = 0; k < GEMV_U; k++) {
        const u32 wq[8] = {w0[k].x, w0[k].y, w0[k].z, w0[k].w, w1[k].x, w1[k].y, w1[k].z, w1[k].w};
        const u32 av[8] = {a0[k].x, a0[k].y, a0[k].z, a0[k].w, a1[k].x, a1[k].y, a1[k].z, a1[k].w};
        const Mem qm = {wq}, am = {av};
        const int dot = (int)ie_q8q8_block(qm, 0u, am, 0u);
        acc[j] += ok[k] ? sc[k] * (float)dot : 0.0f;
      }
    }
  }
  for (u32 j = 0; j < GEMV_T; j++) {
    if (j >= T) break;
    const float a = wave_tree_f(acc[j]);
    if (l == 0u) y[j * ys + r] = epilogue(a, r, bias, res ? res + j * rs : res);
  }
}

/* ie_gemv_q8q8_t for few rows (Qwen3.5's 96-row beta/alpha) spread over more workgroups: workgroups
 * part * gr .. part * gr + gr - 1 (gr = the row groups, ie_gemv_ngroups) do the tokens part * tpw ..
 * part * tpw + tpw - 1 (tpw <= Q8TS_T) of the same rows, each wave one row. Every (row, token) runs the
 * code of ie_gemv_q8q8_t, so outputs are bitwise the same; the weights are read once per part (they are small
 * and sit in cache), the waves in flight go up and each wave's chain of loads gets shorter. */
#define Q8TS_T 4u
KERNEL ie_gemv_q8q8_ts(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                       const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs, u32 gr, u32 tpw) {
  const u32 part = wgid() / gr, g = wgid() - part * gr, j0 = part * tpw;
#ifdef IE_DBG_SKIP_Q8
  return; /* debug: time a verify without this kernel's work (wrong outputs) */
#endif
  if (j0 >= T) return; /* uniform: grid padding */
  const u32 jn = T - j0 < tpw ? T - j0 : tpw;
  const u32 r = ie_gemv_row(g, tid() >> 5), l = lane();
  if (r >= rows) return;
  float acc[Q8TS_T];
  for (u32 j = 0; j < Q8TS_T; j++) acc[j] = 0.0f;
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
    u32x4 w0[GEMV_U], w1[GEMV_U];
    float sw[GEMV_U];
    int ok[GEMV_U];
    u32 bk[GEMV_U];
    for (u32 k = 0; k < GEMV_U; k++) {
      ok[k] = ie_lane_blk(l, t + k) < nb;
      const u32 b = ok[k] ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
      bk[k] = b;
      const G u32 *wp = qw + ie_q8_dst_word(r, b, 0u, nb);
      w0[k] = *(const G u32x4 *)wp;
      w1[k] = *(const G u32x4 *)(wp + 4u);
      sw[k] = (float)qs[ie_q8_dst_scale(r, b, nb)];
    }
    for (u32 j = 0; j < Q8TS_T; j++) {
      if (j >= jn) break; /* uniform */
      const G u8 *xj = xq + (j0 + j) * xs;
      const G u32 *aw = (const G u32 *)xj;
      const G float *da = (const G float *)(xj + 32u * nb);
      u32x4 a0[GEMV_U], a1[GEMV_U];
      float sc[GEMV_U];
      for (u32 k = 0; k < GEMV_U; k++) {
        a0[k] = *(const G u32x4 *)(aw + bk[k] * 8u);
        a1[k] = *(const G u32x4 *)(aw + bk[k] * 8u + 4u);
        sc[k] = sw[k] * da[bk[k]];
      }
      for (u32 k = 0; k < GEMV_U; k++) {
        const u32 wq[8] = {w0[k].x, w0[k].y, w0[k].z, w0[k].w, w1[k].x, w1[k].y, w1[k].z, w1[k].w};
        const u32 av[8] = {a0[k].x, a0[k].y, a0[k].z, a0[k].w, a1[k].x, a1[k].y, a1[k].z, a1[k].w};
        const Mem qm = {wq}, am = {av};
        const int dot = (int)ie_q8q8_block(qm, 0u, am, 0u);
        acc[j] += ok[k] ? sc[k] * (float)dot : 0.0f;
      }
    }
  }
  for (u32 j = 0; j < Q8TS_T; j++) {
    if (j >= jn) break;
    const float a = wave_tree_f(acc[j]);
    if (l == 0u) y[(j0 + j) * ys + r] = epilogue(a, r, bias, res ? res + (j0 + j) * rs : res);
  }
}


/* The multi-token GEMV for many tokens (prompt chunks): wave v of
 * workgroup g does the 4 rows 4 (8 g + v) .. + 3 (32 rows per workgroup),
 * so each activation load serves 4 rows (4 times less cache traffic than
 * ie_gemv_q8q8_t). For each (row, token) the arithmetic and its order are
 * those of ie_gemv_q8q8: the same lane blocks, the same block dot product,
 * the same reduction. */
#ifndef GEMV_R
#define GEMV_R 4u
#endif
#ifndef TR_NT
#define TR_NT 1
#endif
KERNEL_HEAVY ie_gemv_q8q8_tr(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                       const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off) /* no fused multiply-add: the same rounding as ie_gemv_q8q8 */
  const u32 r0 = (wgid() * 8u + (tid() >> 5)) * GEMV_R, l = lane();
  if (r0 >= rows) return;
#ifdef MKS_INKLING_VER
#define Q8TR_CT 8u
  for (u32 T_rem_ = T; T_rem_ > 0u; ) {
    T = T_rem_ < Q8TR_CT ? T_rem_ : Q8TR_CT;
#else
#define Q8TR_CT GEMV_T
#endif
  float acc[GEMV_R][Q8TR_CT];
#pragma unroll
  for (u32 q = 0; q < GEMV_R; q++)
#pragma unroll
    for (u32 j = 0; j < Q8TR_CT; j++) acc[q][j] = 0.0f;
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) {
    const u32 b = ie_lane_blk(l, t);
    u32x4 w0[GEMV_R], w1[GEMV_R];
    float sw[GEMV_R];
#pragma unroll
    for (u32 q = 0; q < GEMV_R; q++) {
      const u32 r = r0 + q < rows ? r0 + q : r0; /* a row past the end: row r0 again, not stored */
      const G u32 *wp = qw + ie_q8_dst_word(r, b, 0u, nb);
      if (TR_NT) { /* weights are read once: keep them out of the caches the activations live in */
        w0[q] = __builtin_nontemporal_load((const G u32x4 *)wp);
        w1[q] = __builtin_nontemporal_load((const G u32x4 *)(wp + 4u));
      } else {
        w0[q] = *(const G u32x4 *)wp;
        w1[q] = *(const G u32x4 *)(wp + 4u);
      }
      sw[q] = (float)qs[ie_q8_dst_scale(r, b, nb)];
    }
#pragma unroll
    for (u32 j = 0; j < Q8TR_CT; j++) {
      if (j < T) { /* uniform */
        const G u8 *xj = xq + j * xs;
        const G u32 *aw = (const G u32 *)xj;
        const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u);
        const float da = ((const G float *)(xj + 32u * nb))[b];
        const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (u32 q = 0; q < GEMV_R; q++) {
          const u32 wq[8] = {w0[q].x, w0[q].y, w0[q].z, w0[q].w, w1[q].x, w1[q].y, w1[q].z, w1[q].w};
          const Mem qm = {wq}, am = {av};
          const int dot = (int)ie_q8q8_block(qm, 0u, am, 0u);
          const float sc = sw[q] * da;
          acc[q][j] = acc[q][j] + sc * (float)dot;
        }
      }
    }
  }
#pragma unroll
  for (u32 q = 0; q < GEMV_R; q++) {
    const u32 r = r0 + q;
#pragma unroll
    for (u32 j = 0; j < Q8TR_CT; j++) {
      if (j < T) {
        const float v = wave_tree_f(acc[q][j]);
        if (l == 0u && r < rows) y[j * ys + r] = epilogue(v, r, bias, res ? res + j * rs : res);
      }
    }
  }
#ifdef MKS_INKLING_VER
    if (T_rem_ <= Q8TR_CT) break;
    T_rem_ -= Q8TR_CT;
    xq += (unsigned long)Q8TR_CT * xs;
    y += (unsigned long)Q8TR_CT * ys;
    if (res) res += (unsigned long)Q8TR_CT * rs;
  }
#endif
#undef Q8TR_CT
}


/* Prompt chunks of many tokens: Y = W X with the matrix instruction
 * v_wmma_i32_16x16x16_iu8 (int8 x int8 -> int32, 16 x 16 x 16 per wave).
 * Lane layout (gfx12, checked by tools/wmma_test.c): A: lane l holds row
 * l % 16, K bytes 8 (l / 16) .. + 7; B: lane l holds token l % 16, the same
 * K bytes; D: lane l, element v = row 8 (l / 16) + v, token l % 16.
 *
 * Workgroup tile: 128 rows x 64 tokens. K goes in chunks of GM_KB blocks of
 * 32: the workgroup copies the weight tile (128 rows x GM_KB x 32 bytes,
 * 16-byte loads, each row's bytes are contiguous), the activation tile and
 * both scale tiles into LDS; then wave w does rows 32 (w % 4) .. + 31 and
 * tokens 32 (w / 4) .. + 31: 2 x 2 WMMA tiles, two WMMA (K = 16 each) per
 * block give the exact int32 block dot product of each (row, token), then
 * acc += (scale_w * scale_x) * dot, the blocks in order. The token tiles of
 * one row tile are next to each other in the grid, so they read W from the
 * cache: W comes from memory about once. The float sums are in a different
 * order than in ie_gemv_q8q8, so the results are close to (about 1e-7
 * relative), not bitwise equal to, the one-token results. */
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
#ifndef GM_KB
#define GM_KB 4u /* blocks of 32 per K chunk */
#endif
/* Workgroup tile GM_R rows x GM_T tokens; each wave GM_WI x GM_WJ WMMA tiles
 * (16 x 16 each); the 8 waves form a (GM_R / 16 GM_WI) x (GM_T / 16 GM_WJ)
 * grid. src/hip.c uses the same GM_R and GM_T for the launch grid. */
#ifndef GM_R
#define GM_R 128u
#endif
#ifndef GM_T
#define GM_T 64u
#endif
#ifndef GM_WI
#define GM_WI 2u
#endif
#ifndef GM_WJ
#define GM_WJ 2u
#endif
#define GM_WROWS (GM_R / (16u * GM_WI)) /* waves along the rows */
#define GM_LS (GM_KB * 32u + 16u) /* LDS row stride, bytes (padded: fewer bank conflicts, 16-byte aligned) */
#ifndef GM_NBUF
#define GM_NBUF 1u /* LDS stages: 2 = double buffer (one barrier per K chunk; measured slower: half the workgroups per CU) */
#endif
static LDS u8 gm_w[GM_NBUF][GM_R * GM_LS];
static LDS u8 gm_x[GM_NBUF][GM_T * GM_LS];
static LDS float gm_sw[GM_NBUF][GM_KB * GM_R], gm_sx[GM_NBUF][GM_T * GM_KB]; /* gm_sw[k][row]: 8 rows in 2 16-byte reads */
typedef float f8 __attribute__((ext_vector_type(8)));
#define GM_NSX ((GM_T * GM_KB + NT - 1u) / NT)
KERNEL ie_gemm_q8(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, const G float *bias,
                  const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  const u32 ntt = (T + GM_T - 1u) / GM_T, tt = wgid() % ntt, rt = wgid() / ntt, t0 = tid(), w = t0 >> 5, l = lane();
  const u32 rbase = rt * GM_R, tbase = tt * GM_T, h = l >> 4;
  const u32 wr = 16u * GM_WI * (w % GM_WROWS), wt = 16u * GM_WJ * (w / GM_WROWS); /* this wave's rows and tokens */
  f8 acc[GM_WI][GM_WJ];
#pragma unroll
  for (u32 i = 0; i < GM_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GM_WJ; j++) acc[i][j] = (f8)(0.0f);
  /* Software pipeline: the global loads of K chunk n + 1 go out (into
   * registers) before the WMMA of chunk n, and (GM_NBUF = 2) are stored into
   * the other LDS buffer after it: one barrier per chunk. Thread t0 copies
   * the pieces t0 + 256 i. */
  u32x4 rw[GM_R * 8u / NT], rx[GM_T * 8u / NT];
  float rsw[GM_R * GM_KB / NT], rsx[GM_NSX];
#define GM_FETCH(kb0_)                                                                                          \
  do {                                                                                                          \
    const u32 kbf = (kb0_), nkf = nb - kbf < GM_KB ? nb - kbf : GM_KB;                                          \
    _Pragma("unroll") for (u32 i = 0; i < GM_R * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i, r = p >> 3, pc = p & 7u, gr = rbase + r < rows ? rbase + r : rows - 1u;        \
      if (pc < 2u * nkf) rw[i] = *(const G u32x4 *)(qw + ie_q8_dst_word(gr, kbf, 0u, nb) + pc * 4u);            \
    }                                                                                                           \
    _Pragma("unroll") for (u32 i = 0; i < GM_T * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i, tk = p >> 3, pc = p & 7u, gt = tbase + tk < T ? tbase + tk : T - 1u;           \
      if (pc < 2u * nkf) rx[i] = *(const G u32x4 *)(xq + (unsigned long)gt * xs + kbf * 32u + pc * 16u);       \
    }                                                                                                           \
    _Pragma("unroll") for (u32 i = 0; i < GM_R * GM_KB / NT; i++) {                                             \
      const u32 p = t0 + NT * i, k = p / GM_R, r = p % GM_R, gr = rbase + r < rows ? rbase + r : rows - 1u;     \
      if (k < nkf) rsw[i] = (float)qs[ie_q8_dst_scale(gr, kbf + k, nb)];                                       \
    }                                                                                                           \
    _Pragma("unroll") for (u32 i = 0; i < GM_NSX; i++) {                                                        \
      const u32 p = t0 + NT * i, tk = p / GM_KB, k = p % GM_KB, gt = tbase + tk < T ? tbase + tk : T - 1u;      \
      if (p < GM_T * GM_KB && k < nkf) rsx[i] = ((const G float *)(xq + (unsigned long)gt * xs + 32u * nb))[kbf + k]; \
    }                                                                                                           \
  } while (0)
#define GM_STORE(bf_, kb0_)                                                                                     \
  do {                                                                                                          \
    const u32 nks = nb - (kb0_) < GM_KB ? nb - (kb0_) : GM_KB;                                                  \
    _Pragma("unroll") for (u32 i = 0; i < GM_R * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i;                                                                                \
      if ((p & 7u) < 2u * nks) *(LDSP u32x4 *)(gm_w[bf_] + (p >> 3) * GM_LS + (p & 7u) * 16u) = rw[i];          \
    }                                                                                                           \
    _Pragma("unroll") for (u32 i = 0; i < GM_T * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i;                                                                                \
      if ((p & 7u) < 2u * nks) *(LDSP u32x4 *)(gm_x[bf_] + (p >> 3) * GM_LS + (p & 7u) * 16u) = rx[i];          \
    }                                                                                                           \
    _Pragma("unroll") for (u32 i = 0; i < GM_R * GM_KB / NT; i++) gm_sw[bf_][t0 + NT * i] = rsw[i];            \
    _Pragma("unroll") for (u32 i = 0; i < GM_NSX; i++)                                                          \
      if (t0 + NT * i < GM_T * GM_KB) gm_sx[bf_][t0 + NT * i] = rsx[i];                                         \
  } while (0)
  GM_FETCH(0u);
  GM_STORE(0u, 0u);
  barrier();
  u32 bf = 0;
  for (u32 kb0 = 0; kb0 < nb; kb0 += GM_KB) {
    const u32 nk = nb - kb0 < GM_KB ? nb - kb0 : GM_KB;
    const int more = kb0 + GM_KB < nb;
    if (more) GM_FETCH(kb0 + GM_KB); /* in flight during the math below */
    const LDSP u8 *lw = gm_w[bf], *lx = gm_x[bf];
    const LDSP float *lsw = gm_sw[bf], *lsx = gm_sx[bf];
    for (u32 k = 0; k < nk; k++) {
      v2i a0[GM_WI], a1[GM_WI], b0[GM_WJ], b1[GM_WJ];
#pragma unroll
      for (u32 i = 0; i < GM_WI; i++) {
        const LDSP u8 *pa = lw + (wr + 16u * i + (l & 15u)) * GM_LS + k * 32u + 8u * h;
        a0[i] = *(const LDSP v2i *)pa, a1[i] = *(const LDSP v2i *)(pa + 16u);
      }
#pragma unroll
      for (u32 j = 0; j < GM_WJ; j++) {
        const LDSP u8 *pb = lx + (wt + 16u * j + (l & 15u)) * GM_LS + k * 32u + 8u * h;
        b0[j] = *(const LDSP v2i *)pb, b1[j] = *(const LDSP v2i *)(pb + 16u);
      }
      /* all WMMA of the block first (independent tiles), then the scaling.
       * The int32 sums start at the bits of 12582912.0f (0x4B400000): the
       * result read as float is exactly 12582912 + dot (|dot| <= 32 * 127 *
       * 127 < 2^22), so one exact subtraction gives the dot as a float. */
      const v8i zm = (v8i)(0x4B400000);
      v8i c[GM_WI][GM_WJ];
#pragma unroll
      for (u32 i = 0; i < GM_WI; i++)
#pragma unroll
        for (u32 j = 0; j < GM_WJ; j++) c[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0[i], 1, b0[j], zm, 0);
#pragma unroll
      for (u32 i = 0; i < GM_WI; i++)
#pragma unroll
        for (u32 j = 0; j < GM_WJ; j++) c[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1[i], 1, b1[j], c[i][j], 0);
      f8 sw[GM_WI];
#pragma unroll
      for (u32 i = 0; i < GM_WI; i++) sw[i] = *(const LDSP f8 *)(lsw + k * GM_R + wr + 16u * i + 8u * h);
#pragma unroll
      for (u32 j = 0; j < GM_WJ; j++) {
        const float dx = lsx[(wt + 16u * j + (l & 15u)) * GM_KB + k];
#pragma unroll
        for (u32 i = 0; i < GM_WI; i++) acc[i][j] += (sw[i] * dx) * (__builtin_bit_cast(f8, c[i][j]) - 12582912.0f);
      }
    }
    if (GM_NBUF == 2u) {
      if (more) GM_STORE(bf ^ 1u, kb0 + GM_KB); /* the other buffer: nobody reads it in this chunk */
      barrier();
      bf ^= 1u;
    } else {
      barrier(); /* the tiles are read before the next chunk overwrites them */
      if (more) GM_STORE(0u, kb0 + GM_KB);
      barrier();
    }
  }
#pragma unroll
  for (u32 i = 0; i < GM_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GM_WJ; j++) {
      const u32 t = tbase + wt + 16u * j + (l & 15u);
      if (t >= T) continue;
      for (u32 v = 0; v < 8u; v++) {
        const u32 r = rbase + wr + 16u * i + 8u * h + v;
        if (r < rows) y[t * ys + r] = epilogue(acc[i][j][v], r, bias, res ? res + t * rs : res);
      }
    }
#undef GM_FETCH
#undef GM_STORE
}

/* ie_gemm_q8r: the same product, weights straight from memory into the
 * registers of the wave that uses them (no LDS for W). Each wave owns
 * GR_WI x 16 rows and all GR_T tokens of the workgroup tile, so no other
 * wave needs its weights; only the activations (used by all waves) and the
 * scales go through LDS (double buffered: one barrier per K chunk). The
 * weights of the next K chunk are loaded into registers during the math of
 * the current one.
 * K order inside a 32-byte block: lane half h takes bytes 16 h .. 16 h + 7
 * for the first WMMA and 16 h + 8 .. + 15 for the second (one 16-byte read
 * per lane and block). A and B use the same order and each byte is used
 * once, so the block dot product is the same exact int32. */
#ifndef GR_WI
#define GR_WI 2u /* row tiles (16 rows) per wave */
#endif
#ifndef GR_T
#define GR_T 64u /* tokens per workgroup tile (all waves) */
#endif
#ifndef GR_KB
#define GR_KB 4u
#endif
#define GR_R (8u * 16u * GR_WI) /* rows per workgroup */
#define GR_WJ (GR_T / 16u)
#define GR_SPT (GR_R >= NT ? GR_R / NT : 1u) /* weight scale rows per thread */
#define GR_LS (GR_KB * 32u + 16u) /* LDS row stride, bytes */
static LDS u8 gr_x[2][GR_T * GR_LS] __attribute__((aligned(16)));
static LDS float gr_sw[2][GR_KB * GR_R] __attribute__((aligned(32))), gr_sx[2][GR_T * GR_KB];
KERNEL ie_gemm_q8r(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, const G float *bias,
                   const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  /* nb % GR_KB == 0 (src/hip.c checks): no partial K chunk */
  const u32 ntt = (T + GR_T - 1u) / GR_T, tt = wgid() % ntt, rt = wgid() / ntt, t0 = tid(), w = t0 >> 5, l = lane();
  const u32 rbase = rt * GR_R, tbase = tt * GR_T, h = l >> 4, wr = 16u * GR_WI * w;
  f8 acc[GR_WI][GR_WJ];
#pragma unroll
  for (u32 i = 0; i < GR_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GR_WJ; j++) acc[i][j] = (f8)(0.0f);
  /* the addresses of chunk 0 (rows and tokens past the end are clamped:
   * computed, not stored); chunk kb0 adds kb0 * 32 bytes (scales: kb0) */
  const G u8 *wp[GR_WI];
#pragma unroll
  for (u32 i = 0; i < GR_WI; i++) {
    const u32 r = rbase + wr + 16u * i + (l & 15u);
    wp[i] = (const G u8 *)qw + (unsigned long)(r < rows ? r : rows - 1u) * nb * 32u + 16u * h;
  }
  const G u8 *xp[GR_T * 8u / NT];
#pragma unroll
  for (u32 i = 0; i < GR_T * 8u / NT; i++) {
    const u32 p = t0 + NT * i, tk = p >> 3, gt = tbase + tk < T ? tbase + tk : T - 1u;
    xp[i] = xq + (unsigned long)gt * xs + (p & 7u) * 16u;
  }
  const u32 sr = rbase + t0 * GR_SPT; /* scale rows of this thread: GR_SPT consecutive (threads t0 * GR_SPT < GR_R) */
  const G f16 *swp = qs + (unsigned long)(sr < rows ? sr : rows - 1u) * nb; /* idle threads: a valid row */
  const u32 sxt = tbase + t0 / GR_KB;
  const G float *sxp = (const G float *)(xq + (unsigned long)(sxt < T ? sxt : T - 1u) * xs + 32u * nb) + t0 % GR_KB;
  u32x4 wa[GR_KB][GR_WI], wn[GR_KB][GR_WI]; /* weights: current chunk, next chunk */
  u32x4 rx[GR_T * 8u / NT];
  f16 rsw[GR_SPT][GR_KB];
  float rsx;
#define GR_FETCH_W(dst_, kb0_)                                                                                  \
  _Pragma("unroll") for (u32 k = 0; k < GR_KB; k++)                                                             \
    _Pragma("unroll") for (u32 i = 0; i < GR_WI; i++) dst_[k][i] = *(const G u32x4 *)(wp[i] + ((kb0_) + k) * 32u);
#define GR_FETCH_X(kb0_)                                                                                        \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GR_T * 8u / NT; i++) rx[i] = *(const G u32x4 *)(xp[i] + (kb0_) * 32u); \
    _Pragma("unroll") for (u32 q = 0; q < GR_SPT; q++) {                                                        \
      const u32 r = sr + q < rows ? q : 0u;                                                                     \
      _Pragma("unroll") for (u32 k = 0; k < GR_KB; k++) rsw[q][k] = swp[(unsigned long)r * nb + (kb0_) + k];   \
    }                                                                                                           \
    rsx = t0 < GR_T * GR_KB ? sxp[(kb0_)] : 0.0f;                                                               \
  } while (0)
#define GR_STORE_X(bf_)                                                                                         \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GR_T * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i;                                                                                \
      *(LDSP u32x4 *)(gr_x[bf_] + (p >> 3) * GR_LS + (p & 7u) * 16u) = rx[i];                                   \
    }                                                                                                           \
    if (t0 * GR_SPT < GR_R) _Pragma("unroll") for (u32 q = 0; q < GR_SPT; q++)                                   \
      _Pragma("unroll") for (u32 k = 0; k < GR_KB; k++) gr_sw[bf_][k * GR_R + t0 * GR_SPT + q] = (float)rsw[q][k]; \
    if (t0 < GR_T * GR_KB) gr_sx[bf_][t0] = rsx;                                                                \
  } while (0)
  GR_FETCH_W(wa, 0u);
  GR_FETCH_X(0u);
  GR_STORE_X(0u);
  barrier();
  u32 bf = 0;
#pragma unroll 1
  for (u32 kb0 = 0; kb0 < nb; kb0 += GR_KB) {
    const int more = kb0 + GR_KB < nb;
    if (more) { /* in flight during the math below */
      GR_FETCH_W(wn, kb0 + GR_KB);
      GR_FETCH_X(kb0 + GR_KB);
    }
    const LDSP u8 *lx = gr_x[bf];
    const LDSP float *lsw = gr_sw[bf], *lsx = gr_sx[bf];
    const u32 nk = nb - kb0 < GR_KB ? nb - kb0 : GR_KB; /* = GR_KB; the branch keeps the blocks apart for the scheduler */
#pragma unroll
    for (u32 k = 0; k < GR_KB; k++) {
      if (k >= nk) break;
      u32x4 b[GR_WJ];
#pragma unroll
      for (u32 j = 0; j < GR_WJ; j++) b[j] = *(const LDSP u32x4 *)(lx + (16u * j + (l & 15u)) * GR_LS + k * 32u + 16u * h);
      const v8i zm = (v8i)(0x4B400000); /* see ie_gemm_q8 */
      float dx[GR_WJ];
#pragma unroll
      for (u32 j = 0; j < GR_WJ; j++) dx[j] = lsx[(16u * j + (l & 15u)) * GR_KB + k];
#pragma unroll
      for (u32 i = 0; i < GR_WI; i++) { /* one row tile at a time (fewer registers) */
        v8i c[GR_WJ];
        const v2i a0 = {(int)wa[k][i].x, (int)wa[k][i].y}, a1 = {(int)wa[k][i].z, (int)wa[k][i].w};
#pragma unroll
        for (u32 j = 0; j < GR_WJ; j++) {
          const v2i b0 = {(int)b[j].x, (int)b[j].y};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0, zm, 0);
        }
#pragma unroll
        for (u32 j = 0; j < GR_WJ; j++) {
          const v2i b1 = {(int)b[j].z, (int)b[j].w};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1, c[j], 0);
        }
        const f8 sw = *(const LDSP f8 *)(lsw + k * GR_R + wr + 16u * i + 8u * h);
#pragma unroll
        for (u32 j = 0; j < GR_WJ; j++) acc[i][j] += (sw * dx[j]) * (__builtin_bit_cast(f8, c[j]) - 12582912.0f);
      }
      __builtin_amdgcn_sched_barrier(0); /* keep the blocks apart: the scheduler otherwise overlaps them and spills */
    }
    if (more) GR_STORE_X(bf ^ 1u); /* nobody reads that buffer in this chunk */
    lbarrier();
    bf ^= 1u;
#pragma unroll
    for (u32 k = 0; k < GR_KB; k++)
#pragma unroll
      for (u32 i = 0; i < GR_WI; i++) wa[k][i] = wn[k][i];
  }
#pragma unroll
  for (u32 i = 0; i < GR_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GR_WJ; j++) {
      const u32 t = tbase + 16u * j + (l & 15u);
      if (t >= T) continue;
      for (u32 v = 0; v < 8u; v++) {
        const u32 r = rbase + wr + 16u * i + 8u * h + v;
        if (r < rows) y[t * ys + r] = epilogue(acc[i][j][v], r, bias, res ? res + t * rs : res);
      }
    }
#undef GR_FETCH_W
#undef GR_FETCH_X
#undef GR_STORE_X
}

/* ie_gemm_q4kr: ie_gemm_q8r for MAT_Q4K weights (nb % 8 == 0). Each lane
 * reads the whole 16-byte nibble block and takes the low (lane half 0:
 * weights 0..15) or high nibbles (weights 16..31), values 0..15: the WMMA
 * gives dq = sum q a exactly; then acc += da ((d sc) dq - (dmin mn) asum).
 * Not bitwise equal to ie_gemv_q4kq8 (another order of the float sums). */
#ifndef GK_WI
#define GK_WI 1u /* row tiles per wave (fewer registers than ie_gemm_q8r: 4 more VALU terms) */
#endif
#ifndef GK_T
#define GK_T 64u
#endif
#define GK_KB 4u
#define GK_R (8u * 16u * GK_WI)
#define GK_WJ (GK_T / 16u)
#define GK_LS (GK_KB * 32u + 16u)
static LDS u8 gk_x[2][GK_T * GK_LS] __attribute__((aligned(16)));
static LDS float gk_sw[2][GK_KB * GK_R] __attribute__((aligned(32))), gk_sx[2][GK_T * GK_KB];
static LDS float gk_mw[2][GK_KB * GK_R] __attribute__((aligned(32)));
/* the min term - sum over blocks of (dmin mn) (da asum) is a small matrix
 * product (K = the blocks): fp16 operands per chunk (row r: -dmin mn of
 * its GK_KB blocks; token t: da asum), one fp16 WMMA per chunk adds it to
 * the f32 sums. fp16 rounding: relative 2^-11 on each of the two factors
 * (far below the 4-bit weight and 8-bit activation rounding). */
typedef _Float16 hk8 __attribute__((ext_vector_type(8)));
typedef _Float16 hk4 __attribute__((ext_vector_type(4)));
static LDS hk4 gk_mh[2][GK_R], gk_sh[2][GK_T];
KERNEL ie_gemm_q4kr(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, const G float *bias,
                   const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  /* nb % GK_KB == 0 (src/hip.c checks): no partial K chunk */
  const u32 ntt = (T + GK_T - 1u) / GK_T, tt = wgid() % ntt, rt = wgid() / ntt, t0 = tid(), w = t0 >> 5, l = lane();
  const u32 rbase = rt * GK_R, tbase = tt * GK_T, h = l >> 4, wr = 16u * GK_WI * w;
  f8 acc[GK_WI][GK_WJ];
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GK_WJ; j++) acc[i][j] = (f8)(0.0f);
  /* the addresses of chunk 0 (rows and tokens past the end are clamped:
   * computed, not stored); chunk kb0 adds kb0 * 32 bytes (scales: kb0) */
  const G u8 *wp[GK_WI];
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++) {
    const u32 r = rbase + wr + 16u * i + (l & 15u);
    wp[i] = (const G u8 *)qw + (unsigned long)(r < rows ? r : rows - 1u) * nb * 16u; /* the whole 16-byte nibble block */
  }
  const G u8 *xp[GK_T * 8u / NT];
#pragma unroll
  for (u32 i = 0; i < GK_T * 8u / NT; i++) {
    const u32 p = t0 + NT * i, tk = p >> 3, gt = tbase + tk < T ? tbase + tk : T - 1u;
    xp[i] = xq + (unsigned long)gt * xs + (p & 7u) * 16u;
  }
  const u32 sr = rbase + t0 * 1u; /* scale rows of this thread: 1u consecutive (threads t0 * 1u < GK_R) */
  const u32 srr = sr < rows ? sr : rows - 1u; /* idle threads: a valid row */
  const G u16 *swp = qs + (unsigned long)srr * nb;
  const G f16 *ddp = (const G f16 *)qs + (unsigned long)rows * nb + (unsigned long)srr * (nb >> 3) * 2u;
  const u32 sxt = tbase + t0 / GK_KB;
  const G float *sxp = (const G float *)(xq + (unsigned long)(sxt < T ? sxt : T - 1u) * xs + 32u * nb) + t0 % GK_KB;
  const G int *sap = (const G int *)(xq + (unsigned long)(sxt < T ? sxt : T - 1u) * xs + 36u * nb) + t0 % GK_KB;
  u32x4 wa[GK_KB][GK_WI], wn[GK_KB][GK_WI]; /* weights: current chunk, next chunk */
  u32x4 rx[GK_T * 8u / NT];
  u16 rsw[GK_KB];
  f16 rdd[2];
  float rsx;
  int rsa; /* asum as loaded: converted at the store, so the fetch does not wait for its loads */
#define GK_FETCH_W(dst_, kb0_)                                                                                  \
  _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++)                                                             \
    _Pragma("unroll") for (u32 i = 0; i < GK_WI; i++) dst_[k][i] = *(const G u32x4 *)(wp[i] + ((kb0_) + k) * 16u);
#define GK_FETCH_X(kb0_)                                                                                        \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GK_T * 8u / NT; i++) rx[i] = *(const G u32x4 *)(xp[i] + (kb0_) * 32u); \
    _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++) rsw[k] = swp[(kb0_) + k];                                 \
    rdd[0] = ddp[((kb0_) >> 3) * 2u], rdd[1] = ddp[((kb0_) >> 3) * 2u + 1u]; /* GK_KB divides 8: one super-block */ \
    rsa = t0 < GK_T * GK_KB ? sap[(kb0_)] : 0;                                                                 \
    rsx = t0 < GK_T * GK_KB ? sxp[(kb0_)] : 0.0f;                                                               \
  } while (0)
#define GK_STORE_X(bf_)                                                                                         \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GK_T * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i;                                                                                \
      *(LDSP u32x4 *)(gk_x[bf_] + (p >> 3) * GK_LS + (p & 7u) * 16u) = rx[i];                                   \
    }                                                                                                           \
    if (t0 < GK_R) {                                                                                            \
      hk4 mh_;                                                                                                  \
      _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++) {                                                       \
        gk_sw[bf_][k * GK_R + t0] = (float)rdd[0] * (float)(rsw[k] & 63u);                                     \
        mh_[k] = (f16)(-((float)rdd[1] * (float)(rsw[k] >> 8)));                                                \
      }                                                                                                         \
      gk_mh[bf_][t0] = mh_;                                                                                     \
    }                                                                                                           \
    if (t0 < GK_T * GK_KB) ((LDSP f16 *)gk_sh[bf_])[t0] = (f16)((float)rsa * rsx);                                    \
    if (t0 < GK_T * GK_KB) gk_sx[bf_][t0] = rsx;                                                                \
  } while (0)
  GK_FETCH_W(wa, 0u);
  GK_FETCH_X(0u);
  GK_STORE_X(0u);
  barrier();
  u32 bf = 0;
#pragma unroll 1
  for (u32 kb0 = 0; kb0 < nb; kb0 += GK_KB) {
    const int more = kb0 + GK_KB < nb;
    if (more) { /* in flight during the math below */
      GK_FETCH_W(wn, kb0 + GK_KB);
      GK_FETCH_X(kb0 + GK_KB);
    }
    const LDSP u8 *lx = gk_x[bf];
    const LDSP float *lsw = gk_sw[bf], *lsx = gk_sx[bf];
    const u32 nk = nb - kb0 < GK_KB ? nb - kb0 : GK_KB; /* = GK_KB; the branch keeps the blocks apart for the scheduler */
#pragma unroll
    for (u32 k = 0; k < GK_KB; k++) {
      if (k >= nk) break;
      u32x4 b[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) b[j] = *(const LDSP u32x4 *)(lx + (16u * j + (l & 15u)) * GK_LS + k * 32u + 16u * h);
      const v8i zm = (v8i)(0x4B400000); /* see ie_gemm_q8 */
      float dx[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) dx[j] = lsx[(16u * j + (l & 15u)) * GK_KB + k];
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) { /* one row tile at a time (fewer registers) */
        v8i c[GK_WJ];
        /* lane half h: weights 16 h .. 16 h + 15 = the low (h = 0) or high nibbles of the 16 bytes */
        const u32x4 nb4 = h ? (wa[k][i] >> 4) & 0x0F0F0F0Fu : wa[k][i] & 0x0F0F0F0Fu;
        const v2i a0 = {(int)nb4.x, (int)nb4.y}, a1 = {(int)nb4.z, (int)nb4.w};
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) {
          const v2i b0 = {(int)b[j].x, (int)b[j].y};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0, zm, 0);
        }
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) {
          const v2i b1 = {(int)b[j].z, (int)b[j].w};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1, c[j], 0);
        }
        const f8 sw = *(const LDSP f8 *)(lsw + k * GK_R + wr + 16u * i + 8u * h);
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) acc[i][j] += (sw * dx[j]) * (__builtin_bit_cast(f8, c[j]) - 12582912.0f); /* (da d sc) dq */
      }
      __builtin_amdgcn_sched_barrier(0); /* keep the blocks apart: the scheduler otherwise overlaps them and spills */
    }
    { /* the min term of the chunk: lane half 0 holds K = blocks 0 .. GK_KB - 1, half 1 zeros */
      const hk4 z4 = (hk4)(0.0f);
      hk8 bm[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) {
        const hk4 v = h ? z4 : gk_sh[bf][16u * j + (l & 15u)];
        bm[j] = (hk8){v[0], v[1], v[2], v[3], 0, 0, 0, 0};
      }
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) {
        const hk4 v = h ? z4 : gk_mh[bf][wr + 16u * i + (l & 15u)];
        const hk8 am = {v[0], v[1], v[2], v[3], 0, 0, 0, 0};
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(am, bm[j], acc[i][j]);
      }
    }
    if (more) GK_STORE_X(bf ^ 1u); /* nobody reads that buffer in this chunk */
    lbarrier();
    bf ^= 1u;
#pragma unroll
    for (u32 k = 0; k < GK_KB; k++)
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) wa[k][i] = wn[k][i];
  }
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GK_WJ; j++) {
      const u32 t = tbase + 16u * j + (l & 15u);
      if (t >= T) continue;
      for (u32 v = 0; v < 8u; v++) {
        const u32 r = rbase + wr + 16u * i + 8u * h + v;
        if (r < rows) y[t * ys + r] = epilogue(acc[i][j][v], r, bias, res ? res + t * rs : res);
      }
    }
#undef GK_FETCH_W
#undef GK_FETCH_X
#undef GK_STORE_X
}

static LDS float kx_sa[2][GK_T * GK_KB] __attribute__((aligned(32))); /* ie_gemm_kx: da asum per (token, block) */
/* ie_gemm_q4kr for MAT_Q4K (kind 3), MAT_Q5K (5: the high bit added to the
 * nibble bytes) and MAT_IQ4XS (6: the nibbles mapped to the int8 table, the
 * scale d (ls - 32), no min term), by groups like ie_gexp_t: group e = grid
 * y (the pairs list[offs[e] .. offs[e + 1]), or 0 .. np - 1), workgroup =
 * GK_R rows, the pairs GK_T at a time (pair p: Q8 row X + (p / xdiv) xs,
 * output row Y + p ys, residual res + p rs, bytes). The min term in f32
 * (not the fp16 WMMA of ie_gemm_q4kr: da asum overflows fp16 for the large
 * activations of some flint layers). Not bitwise equal to decode (another
 * order of the float sums). */
KERNEL ie_gemm_kx(const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y, u32 rows, u32 nb, const G u32 *list,
                  const G u32 *offs, u32 np, u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv, u32 kind, const G u8 *res, u32 rs) {
  /* nb % GK_KB == 0 (src/hip.c checks): no partial K chunk */
  const u32 e = list ? tokid() : 0u;
  const u32 p0 = list ? offs[e] : tokid() * GK_T;
  const u32 p1 = list ? offs[e + 1u] : (p0 + GK_T < np ? p0 + GK_T : np);
  const u32 T = p1 > p0 ? p1 - p0 : 0u;
  const u32 rt = wgid(), t0 = tid(), w = t0 >> 5, l = lane();
  const u32 rbase = rt * GK_R, h = l >> 4, wr = 16u * GK_WI * w;
  if (T == 0u || rbase >= rows) return; /* the whole workgroup */
  const G u32 *qw = qw0 + (unsigned long)e * qwn;
  const G u16 *qs = qs0 + (unsigned long)e * qsn;
  for (u32 tbase = 0; tbase < T; tbase += GK_T) {
  /* the pair of token slot tk of this tile: its Q8 row and its output row */
#define KX_PID(tk_) (list ? list[p0 + (tk_)] : p0 + (tk_))
  f8 acc[GK_WI][GK_WJ];
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GK_WJ; j++) acc[i][j] = (f8)(0.0f);
  /* the addresses of chunk 0 (rows and tokens past the end are clamped:
   * computed, not stored); chunk kb0 adds kb0 * 32 bytes (scales: kb0) */
  const G u8 *wp[GK_WI];
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++) {
    const u32 r = rbase + wr + 16u * i + (l & 15u);
    wp[i] = (const G u8 *)qw + (unsigned long)(r < rows ? r : rows - 1u) * nb * 16u; /* the whole 16-byte nibble block */
  }
  const G u32 *hwp[GK_WI]; /* Q5_K: the high-bit word of each block of the lane's row */
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++) {
    const u32 r = rbase + wr + 16u * i + (l & 15u);
    hwp[i] = qw + ie_q5k_hw(rows, r < rows ? r : rows - 1u, 0u, nb);
  }
  const G u8 *xp[GK_T * 8u / NT];
#pragma unroll
  for (u32 i = 0; i < GK_T * 8u / NT; i++) {
    const u32 p = t0 + NT * i, tk = p >> 3, gt = tbase + tk < T ? tbase + tk : T - 1u;
    xp[i] = X + (unsigned long)(KX_PID(gt) / xdiv) * xs + (p & 7u) * 16u;
  }
  const u32 sr = rbase + t0 * 1u; /* scale rows of this thread: 1u consecutive (threads t0 * 1u < GK_R) */
  const u32 srr = sr < rows ? sr : rows - 1u; /* idle threads: a valid row */
  const G u16 *swp = qs + (unsigned long)srr * nb;
  const G f16 *ddp = (kind == 0u || kind == 8u) ? 0 : (const G f16 *)qs + (unsigned long)rows * nb + (unsigned long)srr * (nb >> 3) * 2u;
  const u32 sxt = tbase + t0 / GK_KB;
  const G u8 *sxr = X + (unsigned long)(KX_PID(sxt < T ? sxt : T - 1u) / xdiv) * xs;
  const G float *sxp = (const G float *)(sxr + 32u * nb) + t0 % GK_KB;
  const G int *sap = (const G int *)(sxr + 36u * nb) + t0 % GK_KB;
  u32x4 wa[GK_KB][GK_WI], wn[GK_KB][GK_WI]; /* weights: current chunk, next chunk */
  u32 ha[GK_KB][GK_WI], hn[GK_KB][GK_WI];    /* Q5_K high bits: current, next */
  u32x4 rx[GK_T * 8u / NT];
  u16 rsw[GK_KB];
  f16 rdd[2];
  float rsx;
  int rsa; /* asum as loaded: converted at the store, so the fetch does not wait for its loads */
#define GK_FETCH_W(dst_, hdst_, kb0_)                                                                           \
  _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++)                                                             \
    _Pragma("unroll") for (u32 i = 0; i < GK_WI; i++) {                                                         \
      dst_[k][i] = *(const G u32x4 *)(wp[i] + ((kb0_) + k) * 16u);                                              \
      hdst_[k][i] = kind == 5u ? hwp[i][(kb0_) + k] : 0u;                                                       \
    }
#define GK_FETCH_X(kb0_)                                                                                        \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GK_T * 8u / NT; i++) rx[i] = *(const G u32x4 *)(xp[i] + (kb0_) * 32u); \
    _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++) rsw[k] = swp[(kb0_) + k];                                 \
    if (ddp) rdd[0] = ddp[((kb0_) >> 3) * 2u], rdd[1] = ddp[((kb0_) >> 3) * 2u + 1u]; /* GK_KB divides 8: one super-block */ \
    rsa = t0 < GK_T * GK_KB ? sap[(kb0_)] : 0;                                                                 \
    rsx = t0 < GK_T * GK_KB ? sxp[(kb0_)] : 0.0f;                                                               \
  } while (0)
#define GK_STORE_X(bf_)                                                                                         \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GK_T * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i;                                                                                \
      *(LDSP u32x4 *)(gk_x[bf_] + (p >> 3) * GK_LS + (p & 7u) * 16u) = rx[i];                                   \
    }                                                                                                           \
    if (t0 < GK_R) {                                                                                            \
      hk4 mh_;                                                                                                  \
      _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++) {                                                       \
        if (kind == 0u || kind == 8u) {                                                                         \
          const float df = (float)__builtin_bit_cast(f16, rsw[k]);                                              \
          gk_sw[bf_][k * GK_R + t0] = df;                                                                       \
          gk_mw[bf_][k * GK_R + t0] = kind == 0u ? -8.0f * df : 0.0f;                                           \
        } else {                                                                                                \
          gk_sw[bf_][k * GK_R + t0] = kind == 6u ? (float)rdd[0] * (float)((int)(rsw[k] & 63u) - 32)           \
                                                 : (float)rdd[0] * (float)(rsw[k] & 63u);                       \
          gk_mw[bf_][k * GK_R + t0] = kind == 6u ? 0.0f : -((float)rdd[1] * (float)(rsw[k] >> 8));             \
        }                                                                                                       \
      }                                                                                                         \
      (void)mh_;                                                                                                \
    }                                                                                                           \
    if (t0 < GK_T * GK_KB) kx_sa[bf_][t0] = (float)rsa * rsx;                                                  \
    if (t0 < GK_T * GK_KB) gk_sx[bf_][t0] = rsx;                                                                \
  } while (0)
  GK_FETCH_W(wa, ha, 0u);
  GK_FETCH_X(0u);
  GK_STORE_X(0u);
  barrier();
  u32 bf = 0;
#pragma unroll 1
  for (u32 kb0 = 0; kb0 < nb; kb0 += GK_KB) {
    const int more = kb0 + GK_KB < nb;
    if (more) { /* in flight during the math below */
      GK_FETCH_W(wn, hn, kb0 + GK_KB);
      GK_FETCH_X(kb0 + GK_KB);
    }
    const LDSP u8 *lx = gk_x[bf];
    const LDSP float *lsw = gk_sw[bf], *lsx = gk_sx[bf];
    const u32 nk = nb - kb0 < GK_KB ? nb - kb0 : GK_KB; /* = GK_KB; the branch keeps the blocks apart for the scheduler */
#pragma unroll
    for (u32 k = 0; k < GK_KB; k++) {
      if (k >= nk) break;
      u32x4 b[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) b[j] = *(const LDSP u32x4 *)(lx + (16u * j + (l & 15u)) * GK_LS + k * 32u + 16u * h);
      const v8i zm = (v8i)(0x4B400000); /* see ie_gemm_q8 */
      float dx[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) dx[j] = lsx[(16u * j + (l & 15u)) * GK_KB + k];
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) { /* one row tile at a time (fewer registers) */
        v8i c[GK_WJ];
        /* lane half h: weights 16 h .. 16 h + 15 = the low (h = 0) or high nibbles of the 16 bytes */
        u32x4 nb4 = h ? (wa[k][i] >> 4) & 0x0F0F0F0Fu : wa[k][i] & 0x0F0F0F0Fu;
        if (kind == 5u) { /* Q5_K: bit 4 of weights 16 h + 4 c .. + 3 */
          const u32 hb = ha[k][i] >> (16u * h);
          nb4.x |= ie_spread4(hb) << 4u, nb4.y |= ie_spread4(hb >> 4u) << 4u;
          nb4.z |= ie_spread4(hb >> 8u) << 4u, nb4.w |= ie_spread4(hb >> 12u) << 4u;
        } else if (kind == 6u || kind == 8u) { /* IQ4_XS / IQ4_NL: the int8 table values */
          nb4.x = ie_iq4_map(nb4.x), nb4.y = ie_iq4_map(nb4.y), nb4.z = ie_iq4_map(nb4.z), nb4.w = ie_iq4_map(nb4.w);
        }
        const v2i a0 = {(int)nb4.x, (int)nb4.y}, a1 = {(int)nb4.z, (int)nb4.w};
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) {
          const v2i b0 = {(int)b[j].x, (int)b[j].y};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0, zm, 0);
        }
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) {
          const v2i b1 = {(int)b[j].z, (int)b[j].w};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1, c[j], 0);
        }
        const f8 sw = *(const LDSP f8 *)(lsw + k * GK_R + wr + 16u * i + 8u * h);
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) acc[i][j] += (sw * dx[j]) * (__builtin_bit_cast(f8, c[j]) - 12582912.0f); /* (da d sc) dq */
        if (kind != 6u && kind != 8u) { /* the min term in f32: - (dmin mn) (da asum) (fp16 overflows for large activations) */
          const f8 mw = *(const LDSP f8 *)(gk_mw[bf] + k * GK_R + wr + 16u * i + 8u * h);
#pragma unroll
          for (u32 j = 0; j < GK_WJ; j++) acc[i][j] += mw * kx_sa[bf][(16u * j + (l & 15u)) * GK_KB + k];
        }
      }
      __builtin_amdgcn_sched_barrier(0); /* keep the blocks apart: the scheduler otherwise overlaps them and spills */
    }
    if (more) GK_STORE_X(bf ^ 1u); /* nobody reads that buffer in this chunk */
    lbarrier();
    bf ^= 1u;
#pragma unroll
    for (u32 k = 0; k < GK_KB; k++)
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) wa[k][i] = wn[k][i], ha[k][i] = hn[k][i];
  }
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GK_WJ; j++) {
      const u32 t = tbase + 16u * j + (l & 15u);
      if (t >= T) continue;
      const u32 pid = KX_PID(t);
      G float *yr = (G float *)(Y + (unsigned long)pid * ys);
      const G float *rr = res ? (const G float *)(res + (unsigned long)pid * rs) : 0;
      for (u32 v = 0; v < 8u; v++) {
        const u32 r = rbase + wr + 16u * i + 8u * h + v;
        if (r < rows) yr[r] = epilogue(acc[i][j][v], r, 0, rr);
      }
    }
  barrier(); /* the next token tile reuses the LDS buffers */
  }
#undef KX_PID
#undef GK_FETCH_W
#undef GK_FETCH_X
#undef GK_STORE_X
}

/* Q4_K x fp16 activations for prompt chunks (ie_gemm_q4k_h). The Q8
 * activations of the chunk first become fp16 (ie_q8_f16: q d, exact up to
 * the fp16 rounding); the weights are dequantized to fp16 in LDS (w = d sc q
 * - dmin m, once per weight and workgroup) and fp16 WMMA sums in f32 over
 * all of K: no per-block scale math on the sums (ie_gemm_q4kr: 3 VALU per
 * output element and block). Workgroup = 64 rows x 512 tokens, so a chunk
 * of up to 512 tokens reads each weight once; wave w: the 64 rows x tokens
 * 64 w .. + 63 (4 x 4 tiles of 16 x 16, 128 f32 registers). Per block of 32
 * (K step): the activations (512 x 32 fp16) and the dequantized weights
 * (64 x 32) to LDS, the next block's loads in registers during the math.
 * Rows % 64 == 0, nb % 8 == 0 (the host checks). Measured against
 * ie_gemm_q4kr (one harness with warm-up, tools/tilelang/ab_q4k.py, 512
 * tokens, without the prefetch): 34816 x 5120 1.982 vs 1.957 ms, 5120 x 17408
 * 1.269 vs 1.006, 10240 x 5120 0.661 vs 0.586; TileLang (same tile) 1.988,
 * 1.178, 0.610: not faster than int8, off by default (IE_GEMM_H=1). */
KERNEL ie_q8_f16(const G u8 *q, G f16 *xh, u32 nb, u32 qs, u32 hs) {
  q = TOKC(q, qs), xh = TOK(xh, hs);
  const u32 i = wgid() * NT + tid();
  if (i >= nb * 32u) return;
  const float d = ((const G float *)(q + 32u * nb))[i >> 5];
  xh[i] = (f16)((float)((const G i8 *)q)[i] * d);
}
#define QH_R 64u
#define QH_T 512u
#define QH_S 40u /* LDS row stride, halves (80 bytes: the 16 rows of a lane group hit distinct banks) */
static LDS f16 qh_x[QH_T * QH_S] __attribute__((aligned(16)));
static LDS f16 qh_w[QH_R * QH_S] __attribute__((aligned(16)));
typedef unsigned int u32x2v __attribute__((ext_vector_type(2)));
static inline __attribute__((always_inline)) void gemm_q4k_hx(const G u32 *qw, const G u16 *qs, const G f16 *xh, G float *y, u32 rows,
                                                            u32 nb, const G float *bias, const G float *res, u32 T, u32 hs, u32 ys,
                                                            u32 rs, const u32 QT, const int PF, const int BATCH) {
  const u32 WT = QT / 8u, NJ = WT / 16u, NL = QT * 4u / NT; /* tokens per wave, its token tiles, 16-byte loads per thread */
  const u32 ntt = (T + QT - 1u) / QT, tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 r0 = rt * QH_R, t0 = tt * QT, w = __builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 hl = l >> 4, rl = l & 15u, wt0 = t0 + WT * w;
  const int won = wt0 < T; /* wave-uniform: this wave has tokens */
  f8 acc[4][4]; /* [row tile][token tile < NJ] */
#pragma unroll
  for (u32 i = 0; i < 4u; i++)
#pragma unroll
    for (u32 j = 0; j < 4u; j++) acc[i][j] = (f8)(0.0f);
  /* loaders: activations, 16 bytes (8 halves) per chunk c = tid + 256 n: token c / 4, part c % 4 */
  const G f16 *xp[8];
#pragma unroll
  for (u32 n = 0; n < NL; n++) {
    const u32 c = tid() + NT * n, tk = t0 + (c >> 2);
    xp[n] = xh + (unsigned long)(tk < T ? tk : T - 1u) * hs + (c & 3u) * 8u;
  }
  /* weights: thread = row tid / 4, quarter q4 = tid % 4: weights 8 q4 .. + 7 of each block
   * (q4 < 2: low nibbles of bytes 8 q4 .. ; q4 >= 2: high nibbles of bytes 8 (q4 - 2) ..) */
  const u32 wr = tid() >> 2, q4 = tid() & 3u;
  const G u8 *wp = (const G u8 *)qw + (unsigned long)(r0 + wr) * nb * 16u + (q4 & 1u) * 8u;
  const G u16 *sp = qs + (unsigned long)(r0 + wr) * nb;
  const G f16 *dp = (const G f16 *)qs + (unsigned long)rows * nb + (unsigned long)(r0 + wr) * (nb >> 3) * 2u;
  u32x4 xr[8];
  u32x2v wb;
  u32 sm;
  float dd, dm;
#define QH_LOAD(kb_)                                                                            \
  do {                                                                                          \
    _Pragma("unroll") for (u32 n = 0; n < NL; n++) xr[n] = *(const G u32x4 *)(xp[n] + (kb_) * 32u); \
    wb = *(const G u32x2v *)(wp + (kb_) * 16u);                                                 \
    sm = sp[(kb_)];                                                                             \
    dd = (float)dp[((kb_) >> 3) * 2u], dm = (float)dp[((kb_) >> 3) * 2u + 1u];                  \
  } while (0)
  if (PF) QH_LOAD(0u);
#pragma unroll 1
  for (u32 kb = 0; kb < nb; kb++) {
    if (!PF) QH_LOAD(kb);
#pragma unroll
    for (u32 n = 0; n < NL; n++) {
      const u32 c = tid() + NT * n;
      *(LDSP u32x4 *)(qh_x + (c >> 2) * QH_S + (c & 3u) * 8u) = xr[n];
    }
    {
      const float a = dd * (float)(sm & 63u), mb = dm * (float)((sm >> 8) & 255u);
      hk8 wv;
#pragma unroll
      for (u32 e = 0; e < 8u; e++) {
        const u32 byte = ((e < 4u ? wb.x : wb.y) >> (8u * (e & 3u))) & 255u;
        const u32 qv = q4 >= 2u ? byte >> 4 : byte & 15u;
        wv[e] = (f16)(a * (float)qv - mb);
      }
      *(LDSP hk8 *)(qh_w + wr * QH_S + q4 * 8u) = wv;
    }
    barrier();
    if (PF && kb + 1u < nb) QH_LOAD(kb + 1u); /* in flight during the math */
    if (won && BATCH) { /* all fragments of a K half first, then its 16 WMMA */
#pragma unroll
      for (u32 kk = 0; kk < 2u; kk++) {
        hk8 a1[4], b1[4];
#pragma unroll
        for (u32 i = 0; i < 4u; i++) a1[i] = *(const LDSP hk8 *)(qh_w + (16u * i + rl) * QH_S + 16u * kk + 8u * hl);
#pragma unroll
        for (u32 j = 0; j < NJ; j++) b1[j] = *(const LDSP hk8 *)(qh_x + (WT * w + 16u * j + rl) * QH_S + 16u * kk + 8u * hl);
#pragma unroll
        for (u32 j = 0; j < NJ; j++)
#pragma unroll
          for (u32 i = 0; i < 4u; i++) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1[i], b1[j], acc[i][j]);
      }
    } else if (won) {
      hk8 a[4][2];
#pragma unroll
      for (u32 i = 0; i < 4u; i++)
#pragma unroll
        for (u32 kk = 0; kk < 2u; kk++) a[i][kk] = *(const LDSP hk8 *)(qh_w + (16u * i + rl) * QH_S + 16u * kk + 8u * hl);
#pragma unroll
      for (u32 j = 0; j < NJ; j++) {
        hk8 bx[2];
#pragma unroll
        for (u32 kk = 0; kk < 2u; kk++) bx[kk] = *(const LDSP hk8 *)(qh_x + (WT * w + 16u * j + rl) * QH_S + 16u * kk + 8u * hl);
#pragma unroll
        for (u32 i = 0; i < 4u; i++)
#pragma unroll
          for (u32 kk = 0; kk < 2u; kk++) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i][kk], bx[kk], acc[i][j]);
      }
    }
    barrier();
  }
#undef QH_LOAD
  if (!won) return;
#pragma unroll
  for (u32 j = 0; j < NJ; j++) {
    const u32 t = wt0 + 16u * j + rl;
    if (t >= T) continue;
#pragma unroll
    for (u32 i = 0; i < 4u; i++)
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const u32 r = r0 + 16u * i + 8u * hl + v;
        y[(unsigned long)t * ys + r] = epilogue(acc[i][j][v], r, bias, res ? res + (unsigned long)t * rs : res);
      }
  }
}

/* PF = 0: the loads of a block at its start, not during the last block's
 * math. Measured (512 tokens, one harness with warm-up): 34816 x 5120 1.982
 * ms without vs 2.226 with the register prefetch; loading all fragments of a
 * K half before its WMMA (BATCH = 1) 2.154 ms. */
KERNEL ie_gemm_q4k_h(const G u32 *qw, const G u16 *qs, const G f16 *xh, G float *y, u32 rows, u32 nb,
                     const G float *bias, const G float *res, u32 T, u32 hs, u32 ys, u32 rs) {
  gemm_q4k_hx(qw, qs, xh, y, rows, nb, bias, res, T, hs, ys, rs, 512u, 0, 0);
}
KERNEL ie_gemm_q4k_h256(const G u32 *qw, const G u16 *qs, const G f16 *xh, G float *y, u32 rows, u32 nb,
                        const G float *bias, const G float *res, u32 T, u32 hs, u32 ys, u32 rs) {
  gemm_q4k_hx(qw, qs, xh, y, rows, nb, bias, res, T, hs, ys, rs, 256u, 0, 0);
}

/* ie_gemm_q4kr_sb: ie_gemm_q4kr for activations whose 8 blocks of a
 * super-block of 256 share one scale da (ie_requant_sb). Then the sum over
 * the blocks of a chunk is exact in int32: sum sc dq (sc: the 6-bit block
 * scales, dq <= 60960 in 24 bits, v_mad_i32_i24), and one float step per
 * chunk acc += (d da) sum: about 1.25 VALU per element and block instead of
 * 3. The min term as ie_gemm_q4kr. Measured (8K prefill, 2026-10-10): 5904
 * -> 5785 ms (2%), perplexity 4K 5.7357 -> 5.7354; off by default
 * (IE_Q4K_SB=1): the GEMM is also bound by its LDS loads and latency (without
 * any scale math it is only 16% faster). */
static LDS int gk_sc[2][GK_KB * GK_R] __attribute__((aligned(32)));
static LDS float gk_dd[2][GK_R] __attribute__((aligned(32)));
static inline v8i mul24v(v8i a, v8i b) { /* the operands fit in 24 bits (signed): v_mul_i32_i24 */
  return ((a << 8) >> 8) * ((b << 8) >> 8);
}
/* Requantize Q8 activations so that the 8 blocks of each super-block of 256
 * share one scale: dequantize (q d), then quantize with D = max |v| / 127
 * of the 256 values (as llama.cpp's Q8_K). One wave per super-block (lane
 * l: block l / 4, elements 8 (l % 4) ..), grid y = tokens. Same layout:
 * int8 [32 nb], float d [nb] (all D), u32 asum [nb]. nb % 8 == 0. */
KERNEL ie_requant_sb(const G u8 *qin, G u8 *qout, u32 nb, u32 is, u32 os) {
  qin = TOKC(qin, is), qout = TOK(qout, os);
  const u32 sb = wgid() * 8u + (tid() >> 5), l = lane();
  if (8u * sb >= nb) return; /* whole waves */
  const u32 blk = 8u * sb + (l >> 2), e0 = 8u * (l & 3u);
  const float d = ((const G float *)(qin + 32u * nb))[blk];
  const G i8 *src = (const G i8 *)qin + blk * 32u + e0;
  float v[8], m = 0.0f;
#pragma unroll
  for (u32 e = 0; e < 8u; e++) v[e] = (float)src[e] * d, m = __builtin_fmaxf(m, __builtin_fabsf(v[e]));
  m = wave_max_pos(m);
  const float D = m / 127.0f;
  G i8 *dst = (G i8 *)qout + blk * 32u + e0;
  float sum = 0.0f;
#pragma unroll
  for (u32 e = 0; e < 8u; e++) {
    const int q = D != 0.0f ? (int)__builtin_roundf(v[e] / D) : 0;
    dst[e] = (i8)q;
    sum += (float)q;
  }
  sum += shfl_xor_f(sum, 1u);
  sum += shfl_xor_f(sum, 2u);
  if ((l & 3u) == 0u) {
    ((G float *)(qout + 32u * nb))[blk] = D;
    ((G u32 *)(qout + 36u * nb))[blk] = (u32)(int)sum;
  }
}
KERNEL ie_gemm_q4kr_sb(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, const G float *bias,
                   const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  /* nb % GK_KB == 0 (src/hip.c checks): no partial K chunk */
  const u32 ntt = (T + GK_T - 1u) / GK_T, tt = wgid() % ntt, rt = wgid() / ntt, t0 = tid(), w = t0 >> 5, l = lane();
  const u32 rbase = rt * GK_R, tbase = tt * GK_T, h = l >> 4, wr = 16u * GK_WI * w;
  f8 acc[GK_WI][GK_WJ];
  v8i acc_i[GK_WI][GK_WJ];
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GK_WJ; j++) acc[i][j] = (f8)(0.0f), acc_i[i][j] = (v8i)(0);
  /* the addresses of chunk 0 (rows and tokens past the end are clamped:
   * computed, not stored); chunk kb0 adds kb0 * 32 bytes (scales: kb0) */
  const G u8 *wp[GK_WI];
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++) {
    const u32 r = rbase + wr + 16u * i + (l & 15u);
    wp[i] = (const G u8 *)qw + (unsigned long)(r < rows ? r : rows - 1u) * nb * 16u; /* the whole 16-byte nibble block */
  }
  const G u8 *xp[GK_T * 8u / NT];
#pragma unroll
  for (u32 i = 0; i < GK_T * 8u / NT; i++) {
    const u32 p = t0 + NT * i, tk = p >> 3, gt = tbase + tk < T ? tbase + tk : T - 1u;
    xp[i] = xq + (unsigned long)gt * xs + (p & 7u) * 16u;
  }
  const u32 sr = rbase + t0 * 1u; /* scale rows of this thread: 1u consecutive (threads t0 * 1u < GK_R) */
  const u32 srr = sr < rows ? sr : rows - 1u; /* idle threads: a valid row */
  const G u16 *swp = qs + (unsigned long)srr * nb;
  const G f16 *ddp = (const G f16 *)qs + (unsigned long)rows * nb + (unsigned long)srr * (nb >> 3) * 2u;
  const u32 sxt = tbase + t0 / GK_KB;
  const G float *sxp = (const G float *)(xq + (unsigned long)(sxt < T ? sxt : T - 1u) * xs + 32u * nb) + t0 % GK_KB;
  const G int *sap = (const G int *)(xq + (unsigned long)(sxt < T ? sxt : T - 1u) * xs + 36u * nb) + t0 % GK_KB;
  u32x4 wa[GK_KB][GK_WI], wn[GK_KB][GK_WI]; /* weights: current chunk, next chunk */
  u32x4 rx[GK_T * 8u / NT];
  u16 rsw[GK_KB];
  f16 rdd[2];
  float rsx;
  int rsa; /* asum as loaded: converted at the store, so the fetch does not wait for its loads */
#define GK_FETCH_W(dst_, kb0_)                                                                                  \
  _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++)                                                             \
    _Pragma("unroll") for (u32 i = 0; i < GK_WI; i++) dst_[k][i] = *(const G u32x4 *)(wp[i] + ((kb0_) + k) * 16u);
#define GK_FETCH_X(kb0_)                                                                                        \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GK_T * 8u / NT; i++) rx[i] = *(const G u32x4 *)(xp[i] + (kb0_) * 32u); \
    _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++) rsw[k] = swp[(kb0_) + k];                                 \
    rdd[0] = ddp[((kb0_) >> 3) * 2u], rdd[1] = ddp[((kb0_) >> 3) * 2u + 1u]; /* GK_KB divides 8: one super-block */ \
    rsa = t0 < GK_T * GK_KB ? sap[(kb0_)] : 0;                                                                 \
    rsx = t0 < GK_T * GK_KB ? sxp[(kb0_)] : 0.0f;                                                               \
  } while (0)
#define GK_STORE_X(bf_)                                                                                         \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GK_T * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i;                                                                                \
      *(LDSP u32x4 *)(gk_x[bf_] + (p >> 3) * GK_LS + (p & 7u) * 16u) = rx[i];                                   \
    }                                                                                                           \
    if (t0 < GK_R) {                                                                                            \
      hk4 mh_;                                                                                                  \
      _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++) {                                                       \
        gk_sc[bf_][k * GK_R + t0] = (int)(rsw[k] & 63u);                                                        \
        mh_[k] = (f16)(-((float)rdd[1] * (float)(rsw[k] >> 8)));                                                \
      }                                                                                                         \
      gk_mh[bf_][t0] = mh_;                                                                                     \
      gk_dd[bf_][t0] = (float)rdd[0];                                                                           \
    }                                                                                                           \
    if (t0 < GK_T * GK_KB) ((LDSP f16 *)gk_sh[bf_])[t0] = (f16)((float)rsa * rsx);                                    \
    if (t0 < GK_T * GK_KB) gk_sx[bf_][t0] = rsx;                                                                \
  } while (0)
  GK_FETCH_W(wa, 0u);
  GK_FETCH_X(0u);
  GK_STORE_X(0u);
  barrier();
  u32 bf = 0;
#pragma unroll 1
  for (u32 kb0 = 0; kb0 < nb; kb0 += GK_KB) {
    const int more = kb0 + GK_KB < nb;
    if (more) { /* in flight during the math below */
      GK_FETCH_W(wn, kb0 + GK_KB);
      GK_FETCH_X(kb0 + GK_KB);
    }
    const LDSP u8 *lx = gk_x[bf];
    const LDSP float *lsx = gk_sx[bf];
    const LDSP int *lsc = gk_sc[bf];
    const u32 nk = nb - kb0 < GK_KB ? nb - kb0 : GK_KB; /* = GK_KB; the branch keeps the blocks apart for the scheduler */
#pragma unroll
    for (u32 k = 0; k < GK_KB; k++) {
      if (k >= nk) break;
      u32x4 b[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) b[j] = *(const LDSP u32x4 *)(lx + (16u * j + (l & 15u)) * GK_LS + k * 32u + 16u * h);
      const v8i zm = (v8i)(0);
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) { /* one row tile at a time (fewer registers) */
        v8i c[GK_WJ];
        /* lane half h: weights 16 h .. 16 h + 15 = the low (h = 0) or high nibbles of the 16 bytes */
        const u32x4 nb4 = h ? (wa[k][i] >> 4) & 0x0F0F0F0Fu : wa[k][i] & 0x0F0F0F0Fu;
        const v2i a0 = {(int)nb4.x, (int)nb4.y}, a1 = {(int)nb4.z, (int)nb4.w};
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) {
          const v2i b0 = {(int)b[j].x, (int)b[j].y};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0, zm, 0);
        }
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) {
          const v2i b1 = {(int)b[j].z, (int)b[j].w};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1, c[j], 0);
        }
        const v8i scv = *(const LDSP v8i *)(lsc + k * GK_R + wr + 16u * i + 8u * h);
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) acc_i[i][j] += mul24v(c[j], scv); /* sc dq, exact */
      }
    }
#pragma unroll
    for (u32 i = 0; i < GK_WI; i++) { /* the chunk's scale: d of the rows x the shared da of the tokens */
      const f8 dd = *(const LDSP f8 *)(gk_dd[bf] + wr + 16u * i + 8u * h);
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) {
        const float dx0 = lsx[(16u * j + (l & 15u)) * GK_KB];
        acc[i][j] += (dd * dx0) * __builtin_convertvector(acc_i[i][j], f8);
        acc_i[i][j] = (v8i)(0);
      }
    }
    { /* the min term of the chunk: lane half 0 holds K = blocks 0 .. GK_KB - 1, half 1 zeros */
      const hk4 z4 = (hk4)(0.0f);
      hk8 bm[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) {
        const hk4 v = h ? z4 : gk_sh[bf][16u * j + (l & 15u)];
        bm[j] = (hk8){v[0], v[1], v[2], v[3], 0, 0, 0, 0};
      }
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) {
        const hk4 v = h ? z4 : gk_mh[bf][wr + 16u * i + (l & 15u)];
        const hk8 am = {v[0], v[1], v[2], v[3], 0, 0, 0, 0};
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(am, bm[j], acc[i][j]);
      }
    }
    if (more) GK_STORE_X(bf ^ 1u); /* nobody reads that buffer in this chunk */
    lbarrier();
    bf ^= 1u;
#pragma unroll
    for (u32 k = 0; k < GK_KB; k++)
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) wa[k][i] = wn[k][i];
  }
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GK_WJ; j++) {
      const u32 t = tbase + 16u * j + (l & 15u);
      if (t >= T) continue;
      for (u32 v = 0; v < 8u; v++) {
        const u32 r = rbase + wr + 16u * i + 8u * h + v;
        if (r < rows) y[t * ys + r] = epilogue(acc[i][j][v], r, bias, res ? res + t * rs : res);
      }
    }
#undef GK_FETCH_W
#undef GK_FETCH_X
#undef GK_STORE_X
}

/* ie_gemm_q6kr: ie_gemm_q4kr for MAT_Q6K weights. A operands: the q - 32
 * bytes (ie_q6k_sw), in the standard K order so that each WMMA covers one
 * half (one int8 scale): c0 = sum over weights 0..15, c1 = over 16..31;
 * then acc += da ((d sc0) c0 + (d sc1) c1). */
#ifndef G6_PAD
#define G6_PAD 16u /* bytes after each token row of the activation tile in LDS */
#endif
#define G6_LS (GK_KB * 32u + G6_PAD)
static LDS u8 g6_x[2][GK_T * G6_LS] __attribute__((aligned(16)));
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));
KERNEL ie_gemm_q6kr(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, const G float *bias,
                   const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  /* nb % GK_KB == 0 (src/hip.c checks): no partial K chunk */
  const u32 ntt = (T + GK_T - 1u) / GK_T, tt = wgid() % ntt, rt = wgid() / ntt, t0 = tid(), w = t0 >> 5, l = lane();
  const u32 rbase = rt * GK_R, tbase = tt * GK_T, h = l >> 4, wr = 16u * GK_WI * w;
  f8 acc[GK_WI][GK_WJ];
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GK_WJ; j++) acc[i][j] = (f8)(0.0f);
  /* the addresses of chunk 0 (rows and tokens past the end are clamped:
   * computed, not stored); chunk kb0 adds kb0 * 32 bytes (scales: kb0) */
  const G u8 *wp[GK_WI];
  const G u32 *hp[GK_WI];
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++) {
    const u32 r = rbase + wr + 16u * i + (l & 15u), rc = r < rows ? r : rows - 1u;
    wp[i] = (const G u8 *)qw + (unsigned long)rc * nb * 16u; /* the 16-byte nibble block */
    hp[i] = qw + ie_q6k_hw(rows, rc, 0u, 0u, nb);             /* the 2 high-bits words */
  }
  const G u8 *xp[GK_T * 8u / NT];
#pragma unroll
  for (u32 i = 0; i < GK_T * 8u / NT; i++) {
    const u32 p = t0 + NT * i, tk = p >> 3, gt = tbase + tk < T ? tbase + tk : T - 1u;
    xp[i] = xq + (unsigned long)gt * xs + (p & 7u) * 16u;
  }
  const u32 sr = rbase + t0 * 1u; /* scale rows of this thread: 1u consecutive (threads t0 * 1u < GK_R) */
  const u32 srr = sr < rows ? sr : rows - 1u; /* idle threads: a valid row */
  const G u16 *swp = qs + ie_q6k_sc(srr, 0u, nb);
  const G f16 *ddp = (const G f16 *)qs + ie_q6k_d(rows, srr, 0u, nb);
  const u32 sxt = tbase + t0 / GK_KB;
  const G float *sxp = (const G float *)(xq + (unsigned long)(sxt < T ? sxt : T - 1u) * xs + 32u * nb) + t0 % GK_KB;
  u32x4 wa[GK_KB][GK_WI], wn[GK_KB][GK_WI]; /* weights: current chunk, next chunk */
  u32x2 ha[GK_KB][GK_WI], hn[GK_KB][GK_WI]; /* their high bits */
  u32x4 rx[GK_T * 8u / NT];
  u16 rsw[GK_KB];
  f16 rdd;
  float rsx;
#define GK_FETCH_W(dst_, kb0_)                                                                                  \
  _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++)                                                             \
    _Pragma("unroll") for (u32 i = 0; i < GK_WI; i++) {                                                        \
      dst_[k][i] = *(const G u32x4 *)(wp[i] + ((kb0_) + k) * 16u);                                              \
      GK_H(dst_)[k][i] = *(const G u32x2 *)(hp[i] + ((kb0_) + k) * 2u);                                            \
    }
#define GK_FETCH_X(kb0_)                                                                                        \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GK_T * 8u / NT; i++) rx[i] = *(const G u32x4 *)(xp[i] + (kb0_) * 32u); \
    _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++) rsw[k] = swp[(kb0_) + k];                                 \
    rdd = ddp[(kb0_) >> 3]; /* GK_KB divides 8: one super-block */                                            \
    rsx = t0 < GK_T * GK_KB ? sxp[(kb0_)] : 0.0f;                                                               \
  } while (0)
#define GK_STORE_X(bf_)                                                                                         \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GK_T * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i;                                                                                \
      *(LDSP u32x4 *)(g6_x[bf_] + (p >> 3) * G6_LS + (p & 7u) * 16u) = rx[i];                                   \
    }                                                                                                           \
    if (t0 < GK_R) _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++) {                                       \
      gk_sw[bf_][k * GK_R + t0] = (float)rdd * (float)(int)(signed char)(rsw[k] & 255u); /* d sc0 */           \
      gk_mw[bf_][k * GK_R + t0] = (float)rdd * (float)(int)(signed char)(rsw[k] >> 8);   /* d sc1 */           \
    }                                                                                                           \
    if (t0 < GK_T * GK_KB) gk_sx[bf_][t0] = rsx;                                                                \
  } while (0)
#define GK_H(d_) GK_H_##d_
#define GK_H_wa ha
#define GK_H_wn hn
  GK_FETCH_W(wa, 0u);
  GK_FETCH_X(0u);
  GK_STORE_X(0u);
  barrier();
  u32 bf = 0;
#pragma unroll 1
  for (u32 kb0 = 0; kb0 < nb; kb0 += GK_KB) {
    const int more = kb0 + GK_KB < nb;
    if (more) { /* in flight during the math below */
      GK_FETCH_W(wn, kb0 + GK_KB);
      GK_FETCH_X(kb0 + GK_KB);
    }
    const LDSP u8 *lx = g6_x[bf];
    const LDSP float *lsw = gk_sw[bf], *lsx = gk_sx[bf], *lmw = gk_mw[bf];
    const u32 nk = nb - kb0 < GK_KB ? nb - kb0 : GK_KB; /* = GK_KB; the branch keeps the blocks apart for the scheduler */
#pragma unroll
    for (u32 k = 0; k < GK_KB; k++) {
      if (k >= nk) break;
      /* the standard K order: WMMA 1 = weights 0..15 (half 0), lane half h
       * bytes 8h .. 8h + 7; WMMA 2 = weights 16..31 (half 1), bytes 16 + 8h .. */
      v2i b0[GK_WJ], b1[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) {
        const LDSP u8 *pb = lx + (16u * j + (l & 15u)) * G6_LS + k * 32u + 8u * h;
        b0[j] = *(const LDSP v2i *)pb, b1[j] = *(const LDSP v2i *)(pb + 16u);
      }
      const v8i zm = (v8i)(0x4B400000); /* see ie_gemm_q8 */
      float dx[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) dx[j] = lsx[(16u * j + (l & 15u)) * GK_KB + k];
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) { /* one row tile at a time (fewer registers) */
        v8i c0[GK_WJ], c1[GK_WJ];
        /* lane half h: the q - 32 bytes (ie_q6k_sw) of words 2h, 2h + 1 of each half */
        const u32 wk[4] = {wa[k][i].x, wa[k][i].y, wa[k][i].z, wa[k][i].w};
        const v2i a0 = {(int)ie_q6k_sw(wk[2u * h], ha[k][i].x, 0u, 2u * h), (int)ie_q6k_sw(wk[2u * h + 1u], ha[k][i].x, 0u, 2u * h + 1u)};
        const v2i a1 = {(int)ie_q6k_sw(wk[2u * h], ha[k][i].y, 1u, 2u * h), (int)ie_q6k_sw(wk[2u * h + 1u], ha[k][i].y, 1u, 2u * h + 1u)};
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0[j], zm, 0);
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) c1[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1[j], zm, 0);
        const f8 s0 = *(const LDSP f8 *)(lsw + k * GK_R + wr + 16u * i + 8u * h);
        const f8 s1 = *(const LDSP f8 *)(lmw + k * GK_R + wr + 16u * i + 8u * h);
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) { /* da ((d sc0) d0 + (d sc1) d1) */
          f8 in = s0 * (__builtin_bit_cast(f8, c0[j]) - 12582912.0f);
          in += s1 * (__builtin_bit_cast(f8, c1[j]) - 12582912.0f);
          acc[i][j] += dx[j] * in;
        }
      }
      __builtin_amdgcn_sched_barrier(0); /* keep the blocks apart: the scheduler otherwise overlaps them and spills */
    }
    if (more) GK_STORE_X(bf ^ 1u); /* nobody reads that buffer in this chunk */
    lbarrier();
    bf ^= 1u;
#pragma unroll
    for (u32 k = 0; k < GK_KB; k++)
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) wa[k][i] = wn[k][i], ha[k][i] = hn[k][i];
  }
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GK_WJ; j++) {
      const u32 t = tbase + 16u * j + (l & 15u);
      if (t >= T) continue;
      for (u32 v = 0; v < 8u; v++) {
        const u32 r = rbase + wr + 16u * i + 8u * h + v;
        if (r < rows) y[t * ys + r] = epilogue(acc[i][j][v], r, bias, res ? res + t * rs : res);
      }
    }
#undef GK_FETCH_W
#undef GK_FETCH_X
#undef GK_STORE_X
#undef GK_H
#undef GK_H_wa
#undef GK_H_wn
}

/* The same product with fp16 WMMA (v_wmma_f32_16x16x16_f16, f32 sums):
 * while staging the tiles in LDS, each weight becomes fp16 (int8 x its
 * block scale) and each activation too (int8 x its block scale), so the
 * WMMA sums need no per-block scaling. Tile: 128 rows x 64 tokens, K in
 * chunks of GH_K = 64; waves as in ie_gemm_q8. Lane layout as the int8
 * WMMA (8 values per lane). fp16 keeps 11 significant bits: each product
 * is rounded once to fp16 (relative error <= 2^-11) before the f32 sums. */
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
#define GH_K 64u
#define GH_LS (GH_K + 8u) /* LDS row stride in halves (16-byte aligned rows, fewer bank conflicts) */
static LDS f16 gh_w[GM_R * GH_LS];
static LDS f16 gh_x[GM_T * GH_LS];
KERNEL ie_gemm_h(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, const G float *bias,
                 const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  const u32 ntt = (T + GM_T - 1u) / GM_T, tt = wgid() % ntt, rt = wgid() / ntt, t0 = tid(), w = t0 >> 5, l = lane();
  const u32 rbase = rt * GM_R, tbase = tt * GM_T, h = l >> 4;
  const u32 wr = 32u * (w & 3u), wt = 32u * (w >> 2);
  const u32 K = nb * 32u;
  v8f acc[2][2];
  for (u32 i = 0; i < 2u; i++)
    for (u32 j = 0; j < 2u; j++) acc[i][j] = (v8f)(0.0f);
  for (u32 k0 = 0; k0 < K; k0 += GH_K) { /* K is a multiple of 32; a last chunk of 32 is padded with zeros */
    /* weights: 128 rows x 4 pieces of 16 int8 (one piece: half a block) */
    for (u32 p = t0; p < GM_R * 4u; p += NT) {
      const u32 r = p >> 2, pc = p & 3u, gr = rbase + r < rows ? rbase + r : rows - 1u, kk = k0 + pc * 16u;
      h8 lo = (h8)(0.0f), hi = (h8)(0.0f);
      if (kk < K) {
        const u32x4 q = *(const G u32x4 *)(qw + ie_q8_dst_word(gr, kk >> 5, (kk & 31u) >> 2, nb));
        const float sc = (float)qs[ie_q8_dst_scale(gr, kk >> 5, nb)];
        const u32 wd[4] = {q.x, q.y, q.z, q.w};
        for (u32 e = 0; e < 8u; e++) lo[e] = (f16)(sc * (float)(i8)ie_byte(wd[e >> 2], e & 3u));
        for (u32 e = 0; e < 8u; e++) hi[e] = (f16)(sc * (float)(i8)ie_byte(wd[2u + (e >> 2)], e & 3u));
      }
      *(LDSP h8 *)(gh_w + r * GH_LS + pc * 16u) = lo;
      *(LDSP h8 *)(gh_w + r * GH_LS + pc * 16u + 8u) = hi;
    }
    /* activations: 64 tokens x 4 pieces */
    for (u32 p = t0; p < GM_T * 4u; p += NT) {
      const u32 tk = p >> 2, pc = p & 3u, gt = tbase + tk < T ? tbase + tk : T - 1u, kk = k0 + pc * 16u;
      h8 lo = (h8)(0.0f), hi = (h8)(0.0f);
      if (kk < K) {
        const G u8 *xt = xq + (unsigned long)gt * xs;
        const u32x4 q = *(const G u32x4 *)(xt + kk);
        const float sc = ((const G float *)(xt + 32u * nb))[kk >> 5];
        const u32 wd[4] = {q.x, q.y, q.z, q.w};
        for (u32 e = 0; e < 8u; e++) lo[e] = (f16)(sc * (float)(i8)ie_byte(wd[e >> 2], e & 3u));
        for (u32 e = 0; e < 8u; e++) hi[e] = (f16)(sc * (float)(i8)ie_byte(wd[2u + (e >> 2)], e & 3u));
      }
      *(LDSP h8 *)(gh_x + tk * GH_LS + pc * 16u) = lo;
      *(LDSP h8 *)(gh_x + tk * GH_LS + pc * 16u + 8u) = hi;
    }
    barrier();
    for (u32 ks = 0; ks < GH_K; ks += 16u) {
      h8 a[2], bb[2];
      for (u32 i = 0; i < 2u; i++) {
        a[i] = *(const LDSP h8 *)(gh_w + (wr + 16u * i + (l & 15u)) * GH_LS + ks + 8u * h);
        bb[i] = *(const LDSP h8 *)(gh_x + (wt + 16u * i + (l & 15u)) * GH_LS + ks + 8u * h);
      }
      for (u32 i = 0; i < 2u; i++)
        for (u32 j = 0; j < 2u; j++) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], bb[j], acc[i][j]);
    }
    barrier();
  }
  for (u32 i = 0; i < 2u; i++)
    for (u32 j = 0; j < 2u; j++) {
      const u32 t = tbase + wt + 16u * j + (l & 15u);
      if (t >= T) continue;
      for (u32 v = 0; v < 8u; v++) {
        const u32 r = rbase + wr + 16u * i + 8u * h + v;
        if (r < rows) y[t * ys + r] = epilogue(acc[i][j][v], r, bias, res ? res + t * rs : res);
      }
    }
}

/* y = W x for an f32 matrix (F16 weights dequantized at load):
 * the same grid; lane l does the elements ie_lane_blk(l, t) < cols. */
/* One row per workgroup (small matrices, e.g. 48 x 5120: latency bound;
 * one row per wave gave 6 workgroups, 20 dependent load rounds each).
 * Thread t takes columns t, t + 256, ..., 8 loads of each in flight; then
 * the wave tree and the workgroup sum (wg_sum: waves 0..7 in order). */
static inline float gemv_f32_row(const G float *row, const G float *x, u32 cols) {
  float acc = 0.0f;
  u32 c = tid();
  for (; c + 7u * NT < cols; c += 8u * NT) {
    float wv[8], xv[8];
#pragma unroll
    for (u32 k = 0; k < 8u; k++) wv[k] = row[c + k * NT], xv[k] = x[c + k * NT];
#pragma unroll
    for (u32 k = 0; k < 8u; k++) acc += wv[k] * xv[k];
  }
  for (; c < cols; c += NT) acc += row[c] * x[c];
  return wg_sum(acc);
}
KERNEL ie_gemv_f32(const G float *w, const G float *x, G float *y, u32 rows, u32 cols, const G float *bias,
                   const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps, G u32 *count) {
  const u32 r = wgid();
  if (r < rows) {
    const float acc = gemv_f32_row(w + (unsigned long)r * cols, x, cols);
    if (tid() == 0u) y[r] = epilogue(acc, r, bias, res);
  }
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

/* ie_gemv_f32 for T tokens in one launch (grid y = token; byte strides xs,
 * ys, rs between tokens): the same arithmetic per token. */
KERNEL ie_gemv_f32_t(const G float *w, const G float *x, G float *y, u32 rows, u32 cols, const G float *bias,
                     const G float *res, u32 xs, u32 ys, u32 rs) {
  x = TOKC(x, xs), y = TOK(y, ys), res = TOK(res, rs);
  const u32 r = wgid();
  if (r >= rows) return;
  const float acc = gemv_f32_row(w + (unsigned long)r * cols, x, cols);
  if (tid() == 0u) y[r] = epilogue(acc, r, bias, res);
}

/* ie_gemv_f32_t for prompt chunks (many tokens, not bitwise equal to
 * ie_gemv_f32): one row per wave, 8 rows per workgroup. */
KERNEL ie_gemv_f32_tw(const G float *w, const G float *x, G float *y, u32 rows, u32 cols, const G float *bias,
                      const G float *res, u32 xs, u32 ys, u32 rs) {
  x = TOKC(x, xs), y = TOK(y, ys), res = TOK(res, rs);
  const u32 r = ie_gemv_row(wgid(), tid() >> 5), l = lane();
  if (r >= rows) return;
  const G float *row = w + (unsigned long)r * cols;
  float acc = 0.0f;
  u32 t = 0;
  for (; ie_lane_blk(l, t + 7u) < cols; t += 8u) {
    float wv[8], xv[8];
#pragma unroll
    for (u32 k = 0; k < 8u; k++) wv[k] = row[ie_lane_blk(l, t + k)], xv[k] = x[ie_lane_blk(l, t + k)];
#pragma unroll
    for (u32 k = 0; k < 8u; k++) acc += wv[k] * xv[k];
  }
  for (; ie_lane_blk(l, t) < cols; t++) acc += row[ie_lane_blk(l, t)] * x[ie_lane_blk(l, t)];
  acc = wave_tree_f(acc);
  if (l == 0u) y[r] = epilogue(acc, r, bias, res);
}

/* f32 matrix x prompt chunk, for small matrices (the 96 x 5120 projections
 * of qwen35): workgroup (g, tt) = rows 8 g .. + 7 (one per wave) and tokens
 * 8 tt .. + 7. Per chunk of 256 columns the 8 tokens' x go to LDS once for
 * the 8 waves; lane l takes the columns l, l + 32, .. of its row, each
 * weight used for the 8 tokens. Then a wave sum per token. Not bitwise equal
 * to ie_gemv_f32. */
#define GS_T 8u
#define GS_C 256u
static LDS float gs_x[GS_T][GS_C];
KERNEL ie_gemm_f32_s(const G float *w, const G float *x, G float *y, u32 rows, u32 cols, const G float *bias,
                     const G float *res, u32 xs, u32 ys, u32 rs, u32 T) {
  const u32 ntt = (T + GS_T - 1u) / GS_T, tt = wgid() % ntt, g = wgid() / ntt, wv = tid() >> 5, l = lane();
  const u32 r = 8u * g + wv, t0 = GS_T * tt;
  const G float *row = w + (unsigned long)(r < rows ? r : rows - 1u) * cols;
  float acc[GS_T];
#pragma unroll
  for (u32 t = 0; t < GS_T; t++) acc[t] = 0.0f;
  for (u32 c0 = 0; c0 < cols; c0 += GS_C) {
    const u32 nc = cols - c0 < GS_C ? cols - c0 : GS_C;
#pragma unroll
    for (u32 n = 0; n < GS_T * GS_C / NT; n++) { /* 8 tokens x 256 columns */
      const u32 i = tid() + NT * n, t = i / GS_C, c = i % GS_C, tk = t0 + t < T ? t0 + t : T - 1u;
      gs_x[t][c] = c < nc ? ((const G float *)((const G u8 *)x + (unsigned long)tk * xs))[c0 + c] : 0.0f;
    }
    barrier();
#pragma unroll
    for (u32 k = 0; k < GS_C / 32u; k++) {
      const u32 c = l + 32u * k;
      const float wk = c < nc ? row[c0 + c] : 0.0f;
#pragma unroll
      for (u32 t = 0; t < GS_T; t++) acc[t] += wk * gs_x[t][c];
    }
    barrier();
  }
#pragma unroll
  for (u32 t = 0; t < GS_T; t++) {
    const float a = wave_tree_f(acc[t]);
    const u32 tk = t0 + t;
    if (l == 0u && r < rows && tk < T)
      ((G float *)((G u8 *)y + (unsigned long)tk * ys))[r] = epilogue(a, r, bias, res ? (const G float *)((const G u8 *)res + (unsigned long)tk * rs) : res);
  }
}

/* ---------------------------------------------------------- elementwise */

KERNEL ie_bias(G float *a, const G float *w, u32 n) {
  const u32 i = wgid() * NT + tid();
  if (i < n) a[i] += w[i];
}

KERNEL ie_add(const G float *a, const G float *b, G float *c, u32 n) {
  const u32 i = wgid() * NT + tid();
  if (i < n) c[i] = a[i] + b[i];
}

/* b[t] = a[t]: n floats per token (grid y: token; strides in floats) */
KERNEL ie_copy_rows(const G float *a, G float *b, u32 n, u32 as, u32 bs) {
  const u32 i = wgid() * NT + tid(), t = tokid();
  if (i < n) b[t * bs + i] = a[t * as + i];
}

/* ------------------------------------------------------------- DFlash 2 */

/* The two-tap dynamic convolution along the tokens of the run (grid y:
 * token t): b[t][c] = res[t][c] + sum_{o < 2, o <= t} (base[w][o][c] +
 * dyn[t][w][o][c / gs]) * a[t - o][c], w = which (base [2][2][n], dyn per
 * token [2][2][n / gs]; strides in floats). */
KERNEL ie_df_conv(const G float *a, G float *b, const G float *dyn, const G float *base, const G float *res, u32 n,
                  u32 which, u32 gs, u32 as, u32 bs, u32 ds, u32 rs) {
  const u32 c = wgid() * NT + tid(), t = tokid();
  if (c >= n) return;
  const u32 ng = n / gs;
  float acc = 0.0f;
  for (u32 o = 0; o < 2u && o <= t; o++) {
    const float x = a[(t - o) * as + c];
    acc += base[(which * 2u + o) * n + c] * x;
    acc += dyn[t * ds + (which * 2u + o) * ng + c / gs] * x;
  }
  b[t * bs + c] = res ? res[t * rs + c] + acc : acc;
}

/* DFlash per head RMSNorm, RoPE (NEOX: pairs (i, i + hd/2), all hd dims)
 * and the KV store, for token t (grid y) at position pos + t. x: [q (nh
 * heads) | k (nkv heads) | v (nkv heads)] of hd each (stride xs floats).
 * Workgroup h < nh: q head h (in place); nh <= h < nh + nkv: k head (to kc);
 * then nkv workgroups copy the v heads (to vc). kc, vc: [ring][nkv*hd], the
 * row of position p is p % ring (ring > the window + the block). */
static LDS float dfq_x[256];
KERNEL ie_df_qkv(G float *x, u32 xs, const G float *qw, const G float *kw, G float *kc, G float *vc, u32 nh, u32 nkv,
                 u32 hd, u32 pos, float eps, float base, u32 ring) {
  const u32 h = wgid(), t = tokid(), i = tid(), kvd = nkv * hd, p = pos + t, row = p % ring; /* the cache row of position p */
  G float *xt = x + (unsigned long)t * xs;
  if (h >= nh + nkv) { /* v */
    const u32 kh = h - nh - nkv;
    if (i < hd) vc[(unsigned long)row * kvd + kh * hd + i] = xt[nh * hd + kvd + kh * hd + i];
    return;
  }
  G float *v = xt + h * hd; /* q heads, then k heads: contiguous */
  const float xv = i < hd ? v[i] : 0.0f;
  const float ss = wg_sum(xv * xv);
  const float r = 1.0f / __builtin_sqrtf(ss / (float)hd + eps);
  const G float *w = h < nh ? qw : kw;
  const float y = i < hd ? xv * r * w[i] : 0.0f;
  if (i < hd) dfq_x[i] = y;
  barrier();
  if (i < hd) {
    const u32 half = hd / 2u, j = i < half ? i : i - half;
    const float th = (float)p * __builtin_powf(base, -2.0f * (float)j / (float)hd);
    const float cs = __builtin_cosf(th), sn = __builtin_sinf(th);
    const float o = i < half ? y * cs - dfq_x[i + half] * sn : y * cs + dfq_x[i - half] * sn;
    if (h < nh) v[i] = o;
    else kc[(unsigned long)row * kvd + (h - nh) * hd + i] = o;
  }
}

/* DFlash attention of the query heads of token t (grid y) at position
 * pos + t over the drafter KV cache: the positions lo .. pos + T - 1, lo =
 * max(0, pos + t - (win - 1)) (non-causal inside the block, a sliding
 * window). Workgroup: query head h (GQA: KV head h / (nh / nkv)). q: [q |
 * k | v] rows of stride qs floats; out rows of stride os floats. */
#define DFA_MAX 4112u /* window + block: DFlash2 2048 + 16, the Qwen3.5 DFlash v1 drafters 4096 + 16 */
static LDS float dfa_s[DFA_MAX];
static LDS float dfa_q[256];
KERNEL ie_df_attn(const G float *q, u32 qs, G float *out, u32 os, const G float *kc, const G float *vc, u32 pos, u32 T, u32 nh,
                  u32 nkv, u32 hd, u32 win, u32 ring) {
  const u32 h = wgid(), t = tokid(), i = tid(), kh = h / (nh / nkv), kvd = nkv * hd;
  const u32 p = pos + t, hi = pos + T;
  u32 lo = p + 1u > win ? p + 1u - win : 0u; /* keys lo .. hi - 1 */
  if (hi - lo > DFA_MAX) lo = hi - DFA_MAX;  /* (the loader keeps win + block <= DFA_MAX) the newest keys */
  const u32 nk = hi - lo;
  const G float *qh = q + (unsigned long)t * qs + h * hd;
  if (i < hd) dfa_q[i] = qh[i];
  barrier();
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  const u32 w = i >> 5, l = lane();
  for (u32 j = w; j < nk; j += 8u) { /* a wave per key */
    const G float *kr = kc + (unsigned long)((lo + j) % ring) * kvd + kh * hd;
    float d = 0.0f;
    for (u32 e = l; e < hd; e += 32u) d += dfa_q[e] * kr[e];
    d = wave_sum_all(d);
    if (l == 0u) dfa_s[j] = d * scale;
  }
  barrier();
  float mv = -__builtin_inff();
  for (u32 j = i; j < nk; j += NT) mv = __builtin_fmaxf(mv, dfa_s[j]);
  const float M = wg_max(mv);
  float sv = 0.0f;
  for (u32 j = i; j < nk; j += NT) {
    const float e = __builtin_expf(dfa_s[j] - M);
    dfa_s[j] = e;
    sv += e;
  }
  const float L = wg_sum(sv); /* its barrier: dfa_s complete */
  if (i < hd) {
    float o = 0.0f;
    for (u32 j = 0; j < nk; j++) o += dfa_s[j] * vc[(unsigned long)((lo + j) % ring) * kvd + kh * hd + i];
    out[(unsigned long)t * os + h * hd + i] = o / L;
  }
}

/* The same attention with the reads shared (GQA, 4 query heads per KV head, hd 128): workgroup = (KV head kh, token
 * t), the keys of ie_df_attn's range; each key and value row is read once for the 4 heads. The 8 waves take keys
 * j = w, w + 8, ... with an online softmax per head (lane: dims 4 l .. 4 l + 3), merged in LDS at the end. Not
 * bitwise ie_df_attn (another summation order): the drafter's output only picks the drafts. */
#define DFG_G 4u
static LDS float dfg_o[8][DFG_G][128];
static LDS float dfg_m[8][DFG_G], dfg_l[8][DFG_G];
KERNEL ie_df_attn_g(const G float *q, u32 qs, G float *out, u32 os, const G float *kc, const G float *vc, u32 pos, u32 T, u32 nh,
                    u32 nkv, u32 hd, u32 win, u32 ring) {
  (void)nh, (void)hd;
  const u32 kh = wgid(), t = tokid(), w = tid() >> 5, l = lane(), kvd = nkv * 128u;
  const u32 p = pos + t, hi = pos + T;
  u32 lo = p + 1u > win ? p + 1u - win : 0u;
  if (hi - lo > DFA_MAX) lo = hi - DFA_MAX;
  const u32 nk = hi - lo;
  const float scale = 1.0f / __builtin_sqrtf(128.0f);
  f32x4 qv[DFG_G], o[DFG_G];
  float m[DFG_G], s[DFG_G];
_Pragma("unroll")
  for (u32 g = 0; g < DFG_G; g++) {
    qv[g] = *(const G f32x4 *)(q + (unsigned long)t * qs + (kh * DFG_G + g) * 128u + 4u * l) * scale;
    o[g] = (f32x4){0.0f, 0.0f, 0.0f, 0.0f}, m[g] = -__builtin_inff(), s[g] = 0.0f;
  }
  const G float *kb = kc + kh * 128u + 4u * l, *vb = vc + kh * 128u + 4u * l;
  for (u32 j = w; j < nk; j += 16u) { /* two keys per pass: j and j + 8 (their loads go out together) */
    const int two = j + 8u < nk;
    const unsigned long r0 = (unsigned long)((lo + j) % ring) * kvd, r1 = two ? (unsigned long)((lo + j + 8u) % ring) * kvd : r0;
    const f32x4 k0 = *(const G f32x4 *)(kb + r0), v0 = *(const G f32x4 *)(vb + r0);
    const f32x4 k1 = *(const G f32x4 *)(kb + r1), v1 = *(const G f32x4 *)(vb + r1);
_Pragma("unroll")
    for (u32 g = 0; g < DFG_G; g++) {
      const f32x4 a = qv[g] * k0, b = qv[g] * k1;
      const float d0 = wave_sum_all(a.x + a.y + a.z + a.w);
      const float d1 = two ? wave_sum_all(b.x + b.y + b.z + b.w) : -__builtin_inff();
      const float mn = __builtin_fmaxf(m[g], __builtin_fmaxf(d0, d1));
      const float c = __builtin_expf(m[g] - mn), e0 = __builtin_expf(d0 - mn), e1 = two ? __builtin_expf(d1 - mn) : 0.0f;
      s[g] = s[g] * c + e0 + e1;
      o[g] = o[g] * c + v0 * e0 + v1 * e1;
      m[g] = mn;
    }
  }
_Pragma("unroll")
  for (u32 g = 0; g < DFG_G; g++) {
    dfg_o[w][g][4u * l] = o[g].x, dfg_o[w][g][4u * l + 1u] = o[g].y;
    dfg_o[w][g][4u * l + 2u] = o[g].z, dfg_o[w][g][4u * l + 3u] = o[g].w;
    if (l == 0u) dfg_m[w][g] = m[g], dfg_l[w][g] = s[g];
  }
  barrier();
  for (u32 x = tid(); x < DFG_G * 128u; x += NT) { /* output (head g, dim i) */
    const u32 g = x / 128u, i = x % 128u;
    float M = -__builtin_inff();
    for (u32 v = 0; v < 8u; v++) M = __builtin_fmaxf(M, dfg_m[v][g]);
    float L = 0.0f, O = 0.0f;
    for (u32 v = 0; v < 8u; v++) {
      const float f = dfg_m[v][g] == -__builtin_inff() ? 0.0f : __builtin_expf(dfg_m[v][g] - M); /* a wave with no keys */
      L += dfg_l[v][g] * f, O += dfg_o[v][g][i] * f;
    }
    out[(unsigned long)t * os + (kh * DFG_G + g) * 128u + i] = O / L;
  }
}

/* Copies in one launch (a tree verify's accept, src/hip.c): n entries of 3 u64 {src, dst, bytes} (bytes a multiple
 * of 4), a workgroup per entry. The entries must not overlap one another. */
KERNEL ie_copy_list(const G unsigned long *list, u32 n) {
  const u32 e = wgid();
  if (e >= n) return;
  const G u32 *src = (const G u32 *)list[3u * e];
  G u32 *dst = (G u32 *)list[3u * e + 1u];
  const unsigned long nw = list[3u * e + 2u] / 4u;
  for (unsigned long i = tid(); i < nw; i += NT) dst[i] = src[i];
}

/* a = silu(a) * b */
/* a = silu(a) * b. n is a multiple of 32, so a wave is one whole block (or
 * none). If q is not NULL, also write the Q8 copy of the result. */
/* cl > 0 (ricursive): the gate is clamped above, up on both sides */
KERNEL ie_swiglu(G float *a, const G float *b, u32 n, G u8 *q, u32 as, u32 qs, float cl) {
  a = TOK(a, as), b = TOKC(b, as), q = TOK(q, qs);
#ifndef IE_MEGA
  const int skip_f32 = (n >> 31) != 0u;
  n &= 0x7FFFFFFFu;
#endif
  const u32 i = wgid() * NT + tid();
#ifdef IE_DBG_SKIP_SWIGLU
  return; /* debug: time a verify without this kernel's work (wrong outputs) */
#endif
  if (i >= n) return; /* whole waves leave together */
  float g = a[i], u = b[i];
  if (cl > 0.0f) {
    if (g > cl) g = cl;
    if (u > cl) u = cl;
    else if (u < -cl) u = -cl;
  }
  const float v = g / (1.0f + __builtin_expf(-g)) * u;
#ifdef IE_MEGA
  a[i] = v;
#else
  if (!skip_f32 || !q) a[i] = v;
#endif
  if (q) quant_wave(v, i >> 5, q, n >> 5);
}

/* ricursive value residual: b = c + (a - c) * sigmoid(lam), q: the Q8 copy */
KERNEL ie_vres(G float *b, const G float *a, const G float *c, u32 n, G u8 *q, u32 bs, u32 as, u32 cs, u32 qs,
               float lam) {
  b = TOK(b, bs), a = TOKC(a, as), c = TOKC(c, cs), q = TOK(q, qs);
  const u32 i = wgid() * NT + tid();
  if (i >= n) return;
  const float s = 1.0f / (1.0f + __builtin_expf(-lam));
  const float v = c[i] + (a[i] - c[i]) * s;
  b[i] = v;
  if (q) quant_wave(v, i >> 5, q, n >> 5);
}

/* ricursive attention gate: a *= sigmoid(b), q: the Q8 copy of a */
KERNEL ie_gate(G float *a, const G float *b, u32 n, G u8 *q, u32 as, u32 bs, u32 qs) {
  a = TOK(a, as), b = TOKC(b, bs), q = TOK(q, qs);
  const u32 i = wgid() * NT + tid();
  if (i >= n) return;
  const float v = a[i] * (1.0f / (1.0f + __builtin_expf(-b[i])));
  a[i] = v;
  if (q) quant_wave(v, i >> 5, q, n >> 5);
}

/* RoPE in place: nh heads of hd; cs/sn: the tables of this position.
 * neox = 0: pairs (2i, 2i+1) (llama); 1: pairs (i, i + hd/2) (qwen2). */
KERNEL ie_rope(G float *x, const G float *cs, const G float *sn, u32 nh, u32 hd, u32 neox) {
  const u32 h2 = hd >> 1, k = wgid() * NT + tid();
  if (k >= nh * h2) return;
  const u32 h = k / h2, i = k % h2;
  G float *v = x + h * hd;
  const u32 i0 = neox ? i : 2u * i, i1 = neox ? i + h2 : 2u * i + 1u;
  const float a = v[i0], b = v[i1];
  v[i0] = a * cs[i] - b * sn[i];
  v[i1] = a * sn[i] + b * cs[i];
}

/* RoPE of q (nq heads) and k (nkv heads) in place, then k and v into the
 * caches kc/vc (already offset to this layer and position). Threads
 * 0 .. (nq+nkv)*h2 - 1 rotate one pair each (a k pair also writes kc); the
 * next kvd threads copy v. */
KERNEL ie_rope_kv(G float *q, G float *k, const G float *v, G kvt *kc, G kvt *vc, const G float *cs,
                  const G float *sn, u32 nq, u32 nkv, u32 hd, u32 neox) {
  const u32 h2 = hd >> 1, t = wgid() * NT + tid(), nrot = (nq + nkv) * h2;
  if (t < nrot) {
    const int is_k = t >= nq * h2;
    const u32 u = is_k ? t - nq * h2 : t, h = u / h2, i = u % h2;
    G float *x = (is_k ? k : q) + h * hd;
    const u32 i0 = neox ? i : 2u * i, i1 = neox ? i + h2 : 2u * i + 1u;
    const float a = x[i0], b = x[i1];
    const float r0 = a * cs[i] - b * sn[i], r1 = a * sn[i] + b * cs[i];
    x[i0] = r0;
    x[i1] = r1;
    if (is_k) {
      kc[h * hd + i0] = r0;
      kc[h * hd + i1] = r1;
    }
  } else if (t < nrot + nkv * hd) {
    vc[t - nrot] = v[t - nrot];
  }
}

/* Store k and v (n floats each) into the caches at kc/vc (already offset to
 * the row of this layer and position). */
KERNEL ie_kv_store(const G float *k, const G float *v, G kvt *kc, G kvt *vc, u32 n) {
  const u32 i = wgid() * NT + tid();
  if (i < n) {
    kc[i] = k[i];
    vc[i] = v[i];
  }
}

/* --------------------------------------------------------------- attention */

#if !defined(IE_KV_F32) && !defined(IE_KV_QUANT)
KERNEL ie_kv_f16(void) {} /* marker: the KV cache is fp16 */
#endif
#ifdef IE_KV_Q8
KERNEL ie_kv_q8(void) {} /* marker: the KV cache is q8_0 (see KVROWP) */
#endif
#ifdef IE_KV_Q4
KERNEL ie_kv_q4(void) {} /* marker: the KV cache is 4 bit (see KVROWP) */
#endif

/* One decode token. One workgroup per query head h (GQA: KV head h / grp).
 * kc, vc: the caches of this layer, [n_ctx][n_kv * hd]. sc: scores
 * [n_head][n_ctx]. Positions 0..pos. */
static inline float sigmoidf(float x) { return 1.0f / (1.0f + __builtin_expf(-x)); }

KERNEL ie_attn(const G float *q, const G kvt *kc, const G kvt *vc, G float *sc, G float *out, u32 pos,
               u32 n_ctx, u32 hd, u32 n_head, u32 n_kv, G u8 *oq, const G float *gate, u32 gstride) {
  const u32 h = wgid(), kh = h / (n_head / n_kv), kvd = n_kv * hd;
  const G float *qh = q + h * hd;
  G float *s = sc + h * n_ctx;
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  float mx = -__builtin_inff();
  for (u32 p = tid(); p <= pos; p += NT) {
    kvrow_t k = KVROWP(kc, p, kh);
    float d = 0.0f;
    for (u32 i = 0; i < hd; i++) d += qh[i] * KVAT(k, i);
    d *= scale;
    s[p] = d;
    mx = __builtin_fmaxf(mx, d);
  }
  mx = wg_max(mx);
  float sum = 0.0f;
  for (u32 p = tid(); p <= pos; p += NT) {
    const float e = __builtin_expf(s[p] - mx);
    s[p] = e;
    sum += e;
  }
  sum = wg_sum(sum); /* has a barrier: the scores are visible to all threads */
  const float inv = 1.0f / sum;
  for (u32 i = tid(); i < hd; i += NT) {
    float acc = 0.0f;
    for (u32 p = 0; p <= pos; p++) acc += s[p] * KVAT(KVROWP(vc, p, kh), i);
    float o = acc * inv;
    if (gate) o = o * sigmoidf(gate[h * gstride + i]);
#ifdef IE_MEGA
    out[h * hd + i] = o;
#else
    if (out) out[h * hd + i] = o;
#endif
    /* fused quantize: only when hd is a multiple of 32 and hd <= 256 (the
     * graph checks it), so the waves with i < hd hold whole blocks */
    if (oq) quant_wave(o, (h * hd + i) >> 5, oq, (n_head * hd) >> 5);
  }
}

/* Attention for one decode token, split over positions (flash decoding).
 * Workgroup g = h * nsplit + s: query head h (KV head h / (n_head / n_kv)),
 * positions s*ch .. min(s*ch + ch, pos + 1) - 1. Wave w does the positions
 * s*ch + w, + 8, ...: the 32 lanes compute q.k (lane l holds the elements
 * l, l + 32, ... of hd), then an online softmax update of (m, l, o[hd]).
 * The 8 waves merge in LDS and the workgroup writes (m, l, o) to part. The
 * last workgroup of head h (atomic count[h]) merges the nsplit partial
 * results, writes out = o / l and, if oq, its Q8 copy (hd % 32 == 0), and
 * sets count[h] back to 0 for the next layer. hd <= 256. */
static LDS float at_m[8], at_l[8];
static LDS float at_o[8][256];
static LDS u32 at_last;
static inline void attn_split_body(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out,
                                   u32 pos, u32 hd, u32 n_head, u32 n_kv, u32 nsplit, u32 ch, G u32 *count, G u8 *oq,
                                   const G float *gate, u32 gstride, u32 h, u32 s) {
  const u32 kh = h / (n_head / n_kv), kvd = n_kv * hd;
  const u32 w = tid() >> 5, l = lane();
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  float qv[8], acc[8];
  for (u32 j = 0; j < 8u; j++) {
    const u32 i = l + 32u * j;
    qv[j] = i < hd ? q[h * hd + i] : 0.0f;
    acc[j] = 0.0f;
  }
  float m = -__builtin_inff(), sum = 0.0f;
  const u32 p0 = s * ch, p1 = p0 + ch < pos + 1u ? p0 + ch : pos + 1u;
  for (u32 p = p0 + w; p < p1; p += 8u) {
    kvrow_t k = KVROWP(kc, p, kh);
    kvrow_t v = KVROWP(vc, p, kh);
    float d = 0.0f;
    for (u32 j = 0; j < 8u; j++)
      if (l + 32u * j < hd) d += qv[j] * KVAT(k, l + 32u * j);
    d = wave_sum_all(d);
    d *= scale;
    const float mn = __builtin_fmaxf(m, d), c = __builtin_expf(m - mn), e = __builtin_expf(d - mn);
    sum = sum * c + e;
    for (u32 j = 0; j < 8u; j++)
      if (l + 32u * j < hd) acc[j] = acc[j] * c + e * KVAT(v, l + 32u * j);
    m = mn;
  }
  /* merge the 8 waves (a wave without positions has m = -inf, sum = 0) */
  if (l == 0u) at_m[w] = m, at_l[w] = sum;
  for (u32 j = 0; j < 8u; j++)
    if (l + 32u * j < hd) at_o[w][l + 32u * j] = acc[j];
  barrier();
  float M = at_m[0];
  for (u32 x = 1; x < 8u; x++) M = __builtin_fmaxf(M, at_m[x]);
  G float *pp = part + (unsigned long)(h * nsplit + s) * (hd + 2u);
  if (tid() < hd) {
    float o = 0.0f;
    for (u32 x = 0; x < 8u; x++) o += at_o[x][tid()] * __builtin_expf(at_m[x] - M);
    pp[2u + tid()] = o;
  }
  if (tid() == 0u) {
    float L = 0.0f;
    for (u32 x = 0; x < 8u; x++) L += at_l[x] * __builtin_expf(at_m[x] - M);
    pp[0] = M;
    pp[1] = L;
  }
  /* every wave makes its stores visible to the device before the count */
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) at_last = __scoped_atomic_fetch_add(&count[h], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u;
  barrier();
  if (!at_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  /* merge: the whole workgroup finds M2 = max m_t and L2 = sum l_t w_t, with
   * the weight w_t = exp(m_t - M2) of split t computed once and written over
   * m_t in part (not in LDS: the megakernel needs little LDS) */
  G float *ph = part + (unsigned long)h * nsplit * (hd + 2u);
  float mv = -__builtin_inff();
  for (u32 t = tid(); t < nsplit; t += NT) mv = __builtin_fmaxf(mv, ph[t * (hd + 2u)]);
  const float M2 = wg_max(mv); /* its barrier: every m_t is read before a w_t is written */
  float lv = 0.0f;
  for (u32 t = tid(); t < nsplit; t += NT) {
    const float wt = __builtin_expf(ph[t * (hd + 2u)] - M2);
    ph[t * (hd + 2u)] = wt;
    lv += ph[t * (hd + 2u) + 1u] * wt;
  }
  const float L2 = wg_sum(lv); /* its barrier also makes the w_t visible to the workgroup */
  if (tid() < hd) { /* hd % 32 == 0 when oq is set: whole waves */
    float o = 0.0f;
    for (u32 t = 0; t < nsplit; t++) o += ph[t * (hd + 2u) + 2u + tid()] * ph[t * (hd + 2u)];
    o = o / L2;
    if (gate) o = o * sigmoidf(gate[h * gstride + tid()]); /* qwen35: the output gate */
#ifdef IE_MEGA
    out[h * hd + tid()] = o;
#else
    if (out) out[h * hd + tid()] = o;
#endif
    if (oq) quant_wave(o, (h * hd + tid()) >> 5, oq, (n_head * hd) >> 5);
  }
  if (tid() == 0u) count[h] = 0u;
}
KERNEL ie_attn_split(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos,
                     u32 hd, u32 n_head, u32 n_kv, u32 nsplit, u32 ch, G u32 *count, G u8 *oq, const G float *gate,
                     u32 gstride) {
  attn_split_body(q, kc, vc, part, out, pos, hd, n_head, n_kv, nsplit, ch, count, oq, gate, gstride, wgid() / nsplit,
                  wgid() % nsplit);
}

/* ie_attn_split for T tokens in one launch (speculative verify; grid y =
 * token j at position pos + j): per token the splits, the arithmetic and the
 * merge of ie_attn_split at that position (bitwise equal). ch: ie_att_ch. Grid x: n_head *
 * smax (smax: the splits of the last token); a workgroup past the splits of
 * its token returns at once. Per token: its part area (pstride floats), its
 * counters (n_head) and the byte strides of q, out, oq and gate. */
KERNEL ie_attn_split_t(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos,
                       u32 hd, u32 n_head, u32 n_kv, u32 smax, u32 ch, G u32 *count, G u8 *oq, const G float *gate,
                       u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride) {
  const u32 j = tokid(), pj = pos + j;
  const u32 nsplit = (pj + ch) / ch; /* ie_att_nsplit (src/model.h): ch does not depend on the position */
  const u32 h = wgid() / smax, s = wgid() % smax;
  if (s >= nsplit) return;
  attn_split_body((const G float *)((const G u8 *)q + (unsigned long)j * qst), kc, vc, part + (unsigned long)j * pstride,
#ifdef IE_MEGA
                  (G float *)((G u8 *)out + (unsigned long)j * ost),
#else
                  out ? (G float *)((G u8 *)out + (unsigned long)j * ost) : out,
#endif
                  pj, hd, n_head, n_kv, nsplit, ch, count + j * n_head,
                  oq ? oq + (unsigned long)j * oqst : oq, gate ? (const G float *)((const G u8 *)gate + (unsigned long)j * gst) : gate,
                  gstride, h, s);
}

#if defined(IE_KV_Q8) && !defined(IE_MEGA)
/* ie_attn_split_t with the loads issued early (ie_attn_split_tp, q8_0 KV, ch <= 32): attn_split_body_pf is
 * attn_split_body with the K and V bytes (and their scales) of all of the wave's positions (4 for ch = 32) loaded
 * before the first dot product, and the final merge reading 8 splits ahead. The arithmetic is the same: the same
 * products and sums in the same order (a value is (float)int8 * (float)scale, the dot adds the terms j = 0 .. 7 in
 * order, then the wave sum, the online softmax update per position), only the loads come earlier, so each of the 4
 * positions no longer waits for the one before it to finish. */
#define ASP_IT 4u
static inline float asp_q(i8 b, f16 sc) { return (float)b * (float)sc; } /* as kvq_1 */
static inline void attn_split_body_pf(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out,
                                      u32 pos, u32 hd, u32 n_head, u32 n_kv, u32 nsplit, u32 ch, G u32 *count, G u8 *oq,
                                      const G float *gate, u32 gstride, u32 h, u32 s) {
  const u32 kh = h / (n_head / n_kv), kvd = n_kv * hd;
  const u32 w = tid() >> 5, l = lane();
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  float qv[8], acc[8];
  for (u32 j = 0; j < 8u; j++) {
    const u32 i = l + 32u * j;
    qv[j] = i < hd ? q[h * hd + i] : 0.0f;
    acc[j] = 0.0f;
  }
  float m = -__builtin_inff(), sum = 0.0f;
  const u32 p0 = s * ch, p1 = p0 + ch < pos + 1u ? p0 + ch : pos + 1u;
  i8 kb[ASP_IT][8], vb[ASP_IT][8];
  f16 ks[ASP_IT][8], vs[ASP_IT][8];
#pragma unroll
  for (u32 it = 0; it < ASP_IT; it++) {
    const u32 p = p0 + w + 8u * it;
    if (p < p1) {
      kvrow_t k = KVROWP(kc, p, kh);
      kvrow_t v = KVROWP(vc, p, kh);
#pragma unroll
      for (u32 j = 0; j < 8u; j++)
        if (l + 32u * j < hd) {
          kb[it][j] = ((const G i8 *)k)[l + 32u * j];
          ks[it][j] = ((const G f16 *)(k + hd))[j];
          vb[it][j] = ((const G i8 *)v)[l + 32u * j];
          vs[it][j] = ((const G f16 *)(v + hd))[j];
        }
    }
  }
#pragma unroll
  for (u32 it = 0; it < ASP_IT; it++) {
    const u32 p = p0 + w + 8u * it;
    if (p < p1) {
      float d = 0.0f;
#pragma unroll
      for (u32 j = 0; j < 8u; j++)
        if (l + 32u * j < hd) d += qv[j] * asp_q(kb[it][j], ks[it][j]);
      d = wave_sum_all(d);
      d *= scale;
      const float mn = __builtin_fmaxf(m, d), c = __builtin_expf(m - mn), e = __builtin_expf(d - mn);
      sum = sum * c + e;
#pragma unroll
      for (u32 j = 0; j < 8u; j++)
        if (l + 32u * j < hd) acc[j] = acc[j] * c + e * asp_q(vb[it][j], vs[it][j]);
      m = mn;
    }
  }
  /* merge the 8 waves (a wave without positions has m = -inf, sum = 0) */
  if (l == 0u) at_m[w] = m, at_l[w] = sum;
  for (u32 j = 0; j < 8u; j++)
    if (l + 32u * j < hd) at_o[w][l + 32u * j] = acc[j];
  barrier();
  float M = at_m[0];
  for (u32 x = 1; x < 8u; x++) M = __builtin_fmaxf(M, at_m[x]);
  G float *pp = part + (unsigned long)(h * nsplit + s) * (hd + 2u);
  if (tid() < hd) {
    float o = 0.0f;
    for (u32 x = 0; x < 8u; x++) o += at_o[x][tid()] * __builtin_expf(at_m[x] - M);
    pp[2u + tid()] = o;
  }
  if (tid() == 0u) {
    float L = 0.0f;
    for (u32 x = 0; x < 8u; x++) L += at_l[x] * __builtin_expf(at_m[x] - M);
    pp[0] = M;
    pp[1] = L;
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) at_last = __scoped_atomic_fetch_add(&count[h], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u;
  barrier();
  if (!at_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  G float *ph = part + (unsigned long)h * nsplit * (hd + 2u);
  float mv = -__builtin_inff();
  for (u32 t = tid(); t < nsplit; t += NT) mv = __builtin_fmaxf(mv, ph[t * (hd + 2u)]);
  const float M2 = wg_max(mv);
  float lv = 0.0f;
  for (u32 t = tid(); t < nsplit; t += NT) {
    const float wt = __builtin_expf(ph[t * (hd + 2u)] - M2);
    ph[t * (hd + 2u)] = wt;
    lv += ph[t * (hd + 2u) + 1u] * wt;
  }
  const float L2 = wg_sum(lv);
  if (tid() < hd) {
    float o = 0.0f;
    for (u32 t0 = 0; t0 < nsplit; t0 += 8u) { /* the splits t0 .. t0 + 7 loaded together, summed in the order t */
      float xo[8], xw[8];
#pragma unroll
      for (u32 k = 0; k < 8u; k++)
        if (t0 + k < nsplit) xo[k] = ph[(t0 + k) * (hd + 2u) + 2u + tid()], xw[k] = ph[(t0 + k) * (hd + 2u)];
#pragma unroll
      for (u32 k = 0; k < 8u; k++)
        if (t0 + k < nsplit) o += xo[k] * xw[k];
    }
    o = o / L2;
    if (gate) o = o * sigmoidf(gate[h * gstride + tid()]);
    if (out) out[h * hd + tid()] = o;
    if (oq) quant_wave(o, (h * hd + tid()) >> 5, oq, (n_head * hd) >> 5);
  }
  if (tid() == 0u) count[h] = 0u;
}
/* ie_attn_split_t with attn_split_body_pf (IE_ATTN_SPLIT_P=0: ie_attn_split_t); ch <= 32 */
KERNEL ie_attn_split_tp(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos,
                        u32 hd, u32 n_head, u32 n_kv, u32 smax, u32 ch, G u32 *count, G u8 *oq, const G float *gate,
                        u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride) {
  const u32 j = tokid(), pj = pos + j;
  const u32 nsplit = (pj + ch) / ch;
  const u32 h = wgid() / smax, s = wgid() % smax;
  if (s >= nsplit) return;
  attn_split_body_pf((const G float *)((const G u8 *)q + (unsigned long)j * qst), kc, vc, part + (unsigned long)j * pstride,
                     out ? (G float *)((G u8 *)out + (unsigned long)j * ost) : out, pj, hd, n_head, n_kv, nsplit, ch, count + j * n_head,
                     oq ? oq + (unsigned long)j * oqst : oq, gate ? (const G float *)((const G u8 *)gate + (unsigned long)j * gst) : gate,
                     gstride, h, s);
}
#endif

/* Split attention with the K/V read once per KV head (ie_attn_kv): for T
 * tokens at positions pos .. pos + T - 1 and the G = n_head / n_kv query
 * heads of one KV head. Workgroup g = (kh * ngs + sg) * nb + b: KV head kh,
 * the splits sg * spw .. sg * spw + spw - 1 (split s: positions s*ch ..
 * s*ch + ch - 1, as ie_attn_split), pairs 8b .. 8b + 7 of the G * T
 * (query head, token) pairs of kh (pair i: token i / G, query head
 * kh * G + i % G). K and V go to LDS, 8 positions at a time (16-byte loads,
 * the next 8 positions in flight during the math); wave w does pair 8b + w.
 * Bitwise equal per (head, token) to attn_split_body at that position: wave
 * x of ie_attn_split does the positions s*ch + x, + 8, ... of split s; here
 * one wave keeps 8 streams x = 0 .. 7 with the same positions in the same
 * order and the same arithmetic, merges them in the order of the LDS merge
 * (x = 0 .. 7) and writes the split's (M, L, o) to the same place. A
 * workgroup does spw splits one after the other (the splits are not
 * changed; spw only sets the work per workgroup), then adds its number of
 * splits to count[j * n_head + h]; the workgroup that completes the count
 * merges the splits with all 256 threads, as attn_split_body (the same sums
 * in the same order, its loads issued 8 at a time). LDS: the K/V buffer also
 * holds the merge weights (16 KB in all). hd <= 256, hd % 4 == 0,
 * ch % 8 == 0, kc and vc 8-byte aligned. */
#define AKV_P 8u
#if defined(IE_MEGA) && !defined(IE_KV_F32)
/* The megakernel (16 KB of LDS for all its kernels): K and V stay fp16 in
 * LDS (8 KB, the same bytes as at_o: one kernel at a time per workgroup) and
 * become f32 when read (exact, so the arithmetic is the same); the weights
 * of the merge go over the m_t in part, as attn_split_body. */
#define akv_kv (*(LDSP kvt(*)[2][AKV_P][256]) & at_o[0][0])
typedef kv4 akv_r;
#define AKV_RAW(p, i) (((const G kv4 *)(p))[i])
#define AKV_ZERO ((kv4){0, 0, 0, 0})
#define AKV_AT(k, r, e) ((float)akv_kv[k][r][e])
#define AKV_WG 1
_Static_assert(sizeof(kvt[2][AKV_P][256]) <= sizeof(at_o), "akv_kv over at_o");
#else
static LDS float akv_kv[2][AKV_P][256]; /* K, V; in the merge: the weights of the splits (<= 4096) */
typedef f32x4 akv_r;
#define AKV_RAW(p, i) KV4(p, i)
#define AKV_ZERO ((f32x4){0.0f, 0.0f, 0.0f, 0.0f})
#define AKV_AT(k, r, e) (akv_kv[k][r][e])
#define AKV_WG 0
#endif
#ifdef IE_KV_QUANT
#define AKV_LD(cc, p, e4) KVLD4(cc, p, kh, e4)
#else
#define AKV_LD(cc, p, e4) AKV_RAW(cc, ((unsigned long)(p) * kvd + kh * hd) / 4u + (e4))
#endif
static LDS u32 akv_last[8];
KERNEL_CALL ie_attn_kv(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos, u32 hd,
                  u32 n_head, u32 n_kv, u32 smax, u32 ch, G u32 *count, G u8 *oq, const G float *gate, u32 gstride, u32 qst,
                  u32 ost, u32 oqst, u32 gst, u32 pstride, u32 T, u32 nb, u32 spw) {
  const u32 grp = n_head / n_kv, kvd = n_kv * hd, ngs = (smax + spw - 1u) / spw;
  const u32 b = wgid() % nb, sg = (wgid() / nb) % ngs, kh = wgid() / (nb * ngs);
  const u32 w = tid() >> 5, l = lane();
  const u32 pi = 8u * b + w;                  /* this wave's pair */
  const u32 j = pi / grp, h = kh * grp + pi % grp;
  const u32 pj = pos + j, nsplit = (pj + ch) / ch; /* ie_att_nsplit for token j */
  const u32 s0 = sg * spw, s1 = s0 + spw < smax ? s0 + spw : smax;
  const int on = pi < grp * T && s0 < nsplit; /* wave-uniform: this pair has splits here */
  const u32 ndone = on ? (s1 < nsplit ? s1 : nsplit) - s0 : 0u; /* its splits in this workgroup */
  /* the workgroup's positions: up to the last token's */
  const u32 P0 = s0 * ch, PE = s1 * ch < pos + T ? s1 * ch : pos + T;
  if (P0 >= PE) return; /* no token has these splits (workgroup-uniform) */
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  const G float *qh = on ? (const G float *)((const G u8 *)q + (unsigned long)j * qst) + h * hd : q;
  float qv[8];
  for (u32 jj = 0; jj < 8u; jj++) {
    const u32 i = l + 32u * jj;
    qv[jj] = on && i < hd ? qh[i] : 0.0f;
  }
  float sm[8], ss[8], sa[8][8]; /* the 8 streams (the waves of ie_attn_split) of the current split */
  const u32 h4 = hd >> 2, n4 = AKV_P * h4; /* float4 per row, per pass (<= 512) */
  akv_r kr[2], vr[2];
#define AKV_LOAD(c)                                                                          \
  for (u32 u = 0; u < 2u; u++) {                                                             \
    const u32 i4 = tid() + NT * u, r = i4 / h4, e4 = i4 - r * h4;                           \
    kr[u] = vr[u] = AKV_ZERO;                                                                \
    if (i4 < n4 && (c) + r < PE) {                                                           \
      kr[u] = AKV_LD(kc, (c) + r, e4);                                                       \
      vr[u] = AKV_LD(vc, (c) + r, e4);                                                       \
    }                                                                                        \
  }
  AKV_LOAD(P0)
  for (u32 c = P0; c < PE; c += AKV_P) {
    const u32 s = c / ch, sp0 = s * ch, sp1 = sp0 + ch; /* the split of these 8 positions */
    if (c == sp0) /* a new split: new streams */
      for (u32 x = 0; x < 8u; x++) {
        sm[x] = -__builtin_inff(), ss[x] = 0.0f;
        for (u32 jj = 0; jj < 8u; jj++) sa[x][jj] = 0.0f;
      }
    for (u32 u = 0; u < 2u; u++) {
      const u32 i4 = tid() + NT * u, r = i4 / h4, e4 = i4 - r * h4;
      if (i4 < n4) {
        *(LDSP akv_r *)&akv_kv[0][r][4u * e4] = kr[u];
        *(LDSP akv_r *)&akv_kv[1][r][4u * e4] = vr[u];
      }
    }
    barrier();
    if (c + AKV_P < PE) { AKV_LOAD(c + AKV_P) }
    const int live = on && s < nsplit; /* this pair has split s */
    const u32 p1 = sp1 < pj + 1u ? sp1 : pj + 1u; /* this pair's end in split s */
    if (live)
#pragma unroll
      for (u32 r = 0; r < AKV_P; r++) {
        /* no branch: the 8 streams are independent, so their math can
         * interleave; a position past this pair's end keeps the old state */
        const u32 p = c + r, x = r; /* stream (p - sp0) % 8; c - sp0 is a multiple of 8 */
        const int ok = p < p1;
        float d = 0.0f;
        for (u32 jj = 0; jj < 8u; jj++)
          if (l + 32u * jj < hd) d += qv[jj] * AKV_AT(0, r, l + 32u * jj);
        d = wave_sum_all(d);
        d *= scale;
        const float mn = __builtin_fmaxf(sm[x], d), cc = __builtin_expf(sm[x] - mn), e = __builtin_expf(d - mn);
        const float s2 = ss[x] * cc + e;
        ss[x] = ok ? s2 : ss[x];
        for (u32 jj = 0; jj < 8u; jj++)
          if (l + 32u * jj < hd) {
            const float a2 = sa[x][jj] * cc + e * AKV_AT(1, r, l + 32u * jj);
            sa[x][jj] = ok ? a2 : sa[x][jj];
          }
        sm[x] = ok ? mn : sm[x];
      }
    barrier();
    /* the end of split s (for this workgroup): merge the 8 streams as the LDS
     * merge of attn_split_body and write the split's (M, L, o) */
    if (live && (c + AKV_P >= sp1 || c + AKV_P >= PE)) {
      float M = sm[0];
      for (u32 x = 1; x < 8u; x++) M = __builtin_fmaxf(M, sm[x]);
      G float *pp = part + (unsigned long)j * pstride + (unsigned long)(h * nsplit + s) * (hd + 2u);
      for (u32 jj = 0; jj < 8u; jj++) {
        const u32 i = l + 32u * jj;
        if (i < hd) {
          float o = 0.0f;
          for (u32 x = 0; x < 8u; x++) o += sa[x][jj] * __builtin_expf(sm[x] - M);
          pp[2u + i] = o;
        }
      }
      if (l == 0u) {
        float L = 0.0f;
        for (u32 x = 0; x < 8u; x++) L += ss[x] * __builtin_expf(sm[x] - M);
        pp[0] = M;
        pp[1] = L;
      }
    }
  }
#undef AKV_LOAD
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (l == 0u)
    akv_last[w] = ndone ? __scoped_atomic_fetch_add(&count[j * n_head + h], ndone, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + ndone == nsplit : 0u;
  barrier();
  /* the merges this workgroup completed (all 256 threads, as attn_split_body) */
#if AKV_WG
#define WV(t) ph[(t) * (hd + 2u)] /* the weights over the m_t (the megakernel) */
#else
  LDSP float *wv = &akv_kv[0][0][0]; /* the weights of the splits (the K/V buffer is free now) */
#define WV(t) wv[t]
#endif
  for (u32 x = 0; x < 8u; x++) {
    if (!akv_last[x]) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const u32 px = 8u * b + x, jx = px / grp, hx = kh * grp + px % grp, ns = (pos + jx + ch) / ch;
    G float *ph = part + (unsigned long)jx * pstride + (unsigned long)hx * ns * (hd + 2u);
    float mv = -__builtin_inff();
    for (u32 t = tid(); t < ns; t += NT) mv = __builtin_fmaxf(mv, ph[t * (hd + 2u)]);
    const float M2 = wg_max(mv);
    float lv = 0.0f;
    for (u32 t = tid(); t < ns; t += NT) {
      const float wt = __builtin_expf(ph[t * (hd + 2u)] - M2);
      WV(t) = wt;
      lv += ph[t * (hd + 2u) + 1u] * wt;
    }
    const float L2 = wg_sum(lv); /* its barrier also makes wv visible */
#ifdef IE_MEGA
    G float *ox = (G float *)((G u8 *)out + (unsigned long)jx * ost);
#else
    G float *ox = out ? (G float *)((G u8 *)out + (unsigned long)jx * ost) : out;
#endif
    G u8 *oqx = oq ? oq + (unsigned long)jx * oqst : oq;
    const G float *gx = gate ? (const G float *)((const G u8 *)gate + (unsigned long)jx * gst) : gate;
    if (tid() < hd) {
      float o = 0.0f;
      u32 t = 0;
      for (; t + 8u <= ns; t += 8u) { /* 8 loads in flight; the sum in the order t = 0, 1, ... */
        float pv[8];
        for (u32 u = 0; u < 8u; u++) pv[u] = ph[(t + u) * (hd + 2u) + 2u + tid()];
        for (u32 u = 0; u < 8u; u++) o += pv[u] * WV(t + u);
      }
      for (; t < ns; t++) o += ph[t * (hd + 2u) + 2u + tid()] * WV(t);
      o = o / L2;
      if (gx) o = o * sigmoidf(gx[hx * gstride + tid()]);
#ifdef IE_MEGA
      ox[hx * hd + tid()] = o;
#else
      if (ox) ox[hx * hd + tid()] = o;
#endif
      if (oqx) quant_wave(o, (hx * hd + tid()) >> 5, oqx, (n_head * hd) >> 5);
    }
    if (tid() == 0u) count[jx * n_head + hx] = 0u;
    barrier(); /* wv is used again by the next merge */
  }
}
#undef WV

/* Split attention for long contexts with fp16 WMMA (ie_attn_long), hd == 256.
 * Workgroup (sg, kh, pg) = split sg (positions 128 sg .. 128 sg + 127), KV head
 * kh, pairs 16 pg .. 16 pg + 15 of the grp * T (token, query head) pairs of kh
 * (pair i: token i / grp at position pos + i / grp, query head kh * grp + i %
 * grp). Wave w owns the 16 keys 128 sg + 16 w .. + 15:
 *   S^T = K Q^T      keys x pairs, 16 WMMA steps (K from global memory straight into the
 *                    A operand, Q from LDS), Q = Qhi + Qlo (fp16 pair of the scaled f32 q)
 *   softmax          max and sum over the 128 keys of the split through LDS, P = exp(S - M)
 *   O = P V          the wave's 32 columns of the output (V slice through a private
 *                    LDS transpose), P = Phi + Plo
 * K and V are exact fp16 and every product is exact in f32, so the sums are f32-accurate. All the
 * arithmetic of a pair (token, head) depends only on that pair's data and position: a token gives the
 * same result alone (T = 1) or inside a verify chunk. The split's (M, L, o) go to part (AL_PSZ floats per
 * split, o 16-byte aligned). The splits are merged in two levels: groups of AL_G consecutive splits, then
 * the groups (the weights exp(M_t - M2), the sums over t in the order t = 0, 1, ...). The workgroup that
 * completes a group (atomic gcount) merges it and writes the group's (M, L, o) after the splits of the
 * pair; the one that completes the groups (atomic count) merges them into the output. One wave per pair.
 * Needs T <= 16, grp <= 8, pstride a multiple of 4, part 16-byte aligned, and room for
 * n_head * (ns + ng + 1) * AL_PSZ floats per token (ns splits, ng groups). */
#if !defined(IE_MEGA) && !defined(IE_KV_F32)
/* q8_0 KV: the V^T hi and lo planes of a tile go through one plane of scratch in turn (AL_V1P, default 1), the same
 * WMMAs in the same order: 17.6 KB of LDS instead of 26, 3 workgroups per WGP instead of 2 */
#ifndef AL_V1P
#ifdef IE_KV_Q8
#define AL_V1P 1
#else
#define AL_V1P 0
#endif
#endif
#if defined(IE_KV_QUANT) && !AL_V1P
#define AL_SCP 2u /* q8_0, 4 bit: the V values are dequantized to f32 and split into an fp16 hi and lo plane */
#else
#define AL_SCP 1u
#endif
#ifndef AL_DBG
#define AL_DBG 0 /* bench experiments: 1 no merge, 2 no V and PV, 4 no QK */
#endif
#define AL_CH 128u  /* positions per split */
#ifndef AL_G
#define AL_G 8u     /* splits per group of the first merge level */
#endif
#ifndef AL_QS_PAD
#define AL_QS_PAD 8u
#endif
#define AL_QS (256u + AL_QS_PAD)  /* LDS row stride of Q hi and lo (halves) */
#define AL_PS 136u  /* LDS row stride of P hi and lo (halves) */
/* the quantized caches (two V^T planes): 16, unpadded, so that the LDS (26 KB, was 34) lets 2 workgroups share a WGP:
 * q8 KV at 27K positions 3-4% faster for 1 to 6 tokens, outputs bitwise the same (tools/attn_long_bench.py) */
#ifndef AL_SS_V
#ifdef IE_KV_QUANT
#define AL_SS_V 16u
#else
#define AL_SS_V 24u
#endif
#endif
#define AL_SS AL_SS_V   /* LDS row stride of the V^T scratch (halves) */
#define AL_PSZ 260u /* floats per partial: M, L, 2 unused, o[256] */
#define AL_OFF_PL (16u * AL_PS * 2u)         /* P lo, after P hi (bytes) */
#define AL_OFF_SC (2u * AL_OFF_PL)           /* the V^T scratch, 8 waves x 32 x AL_SS halves */
#define AL_BYTES (AL_OFF_SC + 8u * AL_SCP * 32u * AL_SS * 2u)
_Static_assert(AL_BYTES >= 2u * 16u * AL_QS * 2u, "al_lds holds Q hi and lo");
_Static_assert(AL_BYTES >= 8u * 1024u, "al_lds holds the merge staging");
static LDS u8 al_lds[AL_BYTES] __attribute__((aligned(16)));
static LDS float al_red[2][8 * 16];
static LDS u32 al_last[16];
#ifdef IE_KV_Q4
static inline float al_lo(u32 x) { return (float)__builtin_bit_cast(f16, (unsigned short)x); }         /* the fp16 d of a (d, m) pair */
static inline float al_hi(u32 x) { return (float)__builtin_bit_cast(f16, (unsigned short)(x >> 16)); } /* the fp16 m */
#endif
static inline u32 al_ns(u32 pj) { return pj / AL_CH + 1u; }          /* splits of the token at position pj */
static inline u32 al_ng(u32 ns) { return (ns + AL_G - 1u) / AL_G; }  /* groups of ns splits */
static inline float al_wave_max(float v) {
  for (u32 m = 16u; m >= 1u; m >>= 1) v = __builtin_fmaxf(v, shfl_xor_f(v, m));
  return v;
}
/* One wave sums n partials at ph (AL_PSZ floats apart): M2 = max M_t, w_t = exp(M_t - M2), L2 = sum l_t w_t, and
 * lane l holds the columns 4 l .. 4 l + 3 (a0) and 128 + 4 l .. (a1) of o2 = sum o_t w_t, all in the order t. */
static inline void al_msum(const G float *ph, u32 n, float *M2o, float *L2o, f32x4 *a0o, f32x4 *a1o) {
  const u32 l = lane();
  float mv = -__builtin_inff();
  for (u32 t = l; t < n; t += 32u) mv = __builtin_fmaxf(mv, ph[(unsigned long)t * AL_PSZ]);
  const float M2 = al_wave_max(mv);
  f32x4 a0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f}, a1 = a0;
  float L2 = 0.0f;
  for (u32 t0 = 0; t0 < n; t0 += 32u) {
    const u32 tl = t0 + l, nu = n - t0 < 32u ? n - t0 : 32u;
    const float wl = tl < n ? __builtin_expf(ph[(unsigned long)tl * AL_PSZ] - M2) : 0.0f;
    const float ll = tl < n ? ph[(unsigned long)tl * AL_PSZ + 1u] : 0.0f;
    u32 u = 0;
    for (; u + 8u <= nu; u += 8u) { /* 16 loads in flight, then the sums in the order t */
      f32x4 x0[8], x1[8];
#pragma unroll
      for (u32 k = 0; k < 8u; k++) {
        const G float *pt = ph + (unsigned long)(t0 + u + k) * AL_PSZ + 4u + 4u * l;
        x0[k] = *(const G f32x4 *)pt;
        x1[k] = *(const G f32x4 *)(pt + 128);
      }
#pragma unroll
      for (u32 k = 0; k < 8u; k++) {
        const float wt = I2F(__builtin_amdgcn_readlane(F2I(wl), (int)(u + k)));
        L2 += I2F(__builtin_amdgcn_readlane(F2I(ll), (int)(u + k))) * wt;
        a0 += x0[k] * wt;
        a1 += x1[k] * wt;
      }
    }
    for (; u < nu; u++) {
      const G float *pt = ph + (unsigned long)(t0 + u) * AL_PSZ + 4u + 4u * l;
      const f32x4 x0 = *(const G f32x4 *)pt, x1 = *(const G f32x4 *)(pt + 128);
      const float wt = I2F(__builtin_amdgcn_readlane(F2I(wl), (int)u));
      L2 += I2F(__builtin_amdgcn_readlane(F2I(ll), (int)u)) * wt;
      a0 += x0 * wt;
      a1 += x1 * wt;
    }
  }
  *M2o = M2, *L2o = L2, *a0o = a0, *a1o = a1;
}
/* out = o / L, times the gate, and its Q8 copy. ox: the head's 256 outputs, gxh: its gate (or NULL), blk0: its first
 * Q8 block. */
static inline void al_final(f32x4 a0, f32x4 a1, float L2, G float *ox, G u8 *oqx, const G float *gxh, u32 blk0, u32 nbq,
                            LDSP float *ob) {
  const u32 l = lane();
  /* lane l holds the columns 4 l .. 4 l + 3 and 128 + 4 l ..; the Q8 blocks need lane j = column 32 b + j */
  *(LDSP f32x4 *)(ob + 4u * l) = a0 / L2;
  *(LDSP f32x4 *)(ob + 128u + 4u * l) = a1 / L2;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 i = 0; i < 8u; i++) {
    float o = ob[32u * i + l];
    if (gxh) o = o * sigmoidf(gxh[32u * i + l]);
    if (ox) ox[32u * i + l] = o;
    if (oqx) quant_wave(o, blk0 + i, oqx, nbq);
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}
#ifdef IE_KV_Q8
/* 8 int8 (the words w0, w1) times the fp16 scale s as an fp16 hi plane and an fp16 lo plane (hi + lo = the exact
 * product): the byte ^ 0x80 into the mantissa of 1024 (exact), minus 1152 (exact: the int8 value), hi = q s (one
 * rounding of the exact product, the same as the f32 product rounded to fp16), lo = q s - hi (one fma, exact: the same as
 * the f32 difference rounded to fp16). Bitwise the values of the scalar f32 route, in four packed ops per two values
 * instead of seven scalar ones (ie_attn_pl does the same for the prompt). */
typedef _Float16 al_h2 __attribute__((ext_vector_type(2)));
static inline void al_deq8(u32 w0, u32 w1, f16 s, h8 *hi, h8 *lo) {
  const u32 mg = 0x64646464u, x0 = w0 ^ 0x80808080u, x1 = w1 ^ 0x80808080u;
  const al_h2 sv = {s, s}, bias = {(f16)1152.0f, (f16)1152.0f};
  const al_h2 q0 = __builtin_bit_cast(al_h2, __builtin_amdgcn_perm(mg, x0, 0x04010400u)) - bias;
  const al_h2 q1 = __builtin_bit_cast(al_h2, __builtin_amdgcn_perm(mg, x0, 0x04030402u)) - bias;
  const al_h2 q2 = __builtin_bit_cast(al_h2, __builtin_amdgcn_perm(mg, x1, 0x04010400u)) - bias;
  const al_h2 q3 = __builtin_bit_cast(al_h2, __builtin_amdgcn_perm(mg, x1, 0x04030402u)) - bias;
  const al_h2 h0 = q0 * sv, h1 = q1 * sv, h2 = q2 * sv, h3 = q3 * sv;
  const al_h2 l0 = __builtin_elementwise_fma(q0, sv, -h0), l1 = __builtin_elementwise_fma(q1, sv, -h1),
              l2 = __builtin_elementwise_fma(q2, sv, -h2), l3 = __builtin_elementwise_fma(q3, sv, -h3);
  *hi = (h8){h0[0], h0[1], h1[0], h1[1], h2[0], h2[1], h3[0], h3[1]};
  *lo = (h8){l0[0], l0[1], l1[0], l1[1], l2[0], l2[1], l3[0], l3[1]};
}
#endif
#include "ie_attn_long_q8.inc"
#if !defined(IE_KV_QUANT)
#define AL_VDECL h8 vr[8][2];
#define AL_VLOAD                                                                                        \
  _Pragma("unroll") for (u32 kt = 0; kt < 8u; kt++) if (c0 + 16u * kt < kend && !(AL_DBG & 2)) {          \
    const unsigned long row = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;               \
    const G h8 *vp = (const G h8 *)(vc + (row * n_kv + kh) * 256ul + 32ul * w + 16ul * hl);             \
    vr[kt][0] = vp[0];                                                                                  \
    vr[kt][1] = vp[1];                                                                                  \
  }
#elif defined(IE_KV_Q8)
#define AL_VDECL                                                                                        \
  u32x4 vr[8];                                                                                          \
  f16 vsc[8];
#define AL_VLOAD                                                                                        \
  _Pragma("unroll") for (u32 kt = 0; kt < 8u; kt++) if (c0 + 16u * kt < kend && !(AL_DBG & 2)) {          \
    const unsigned long row = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;               \
    const G u8 *rec = (const G u8 *)vc + (row * n_kv + kh) * 272ul;                                     \
    vr[kt] = *(const G u32x4 *)(rec + 32ul * w + 16ul * hl);                                            \
    vsc[kt] = ((const G f16 *)(rec + 256))[w];                                                          \
  }
#else
/* 4 bit: the 16 values (8 bytes) of columns 32 w + 16 hl .. of key row rl, and the (d, m) pair of the block w */
#define AL_VDECL                                                                                        \
  u32x2 vr[8];                                                                                          \
  u32 vdm[8];
#define AL_VLOAD                                                                                        \
  _Pragma("unroll") for (u32 kt = 0; kt < 8u; kt++) if (c0 + 16u * kt < kend && !(AL_DBG & 2)) {          \
    const unsigned long row = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;               \
    const G u8 *rec = (const G u8 *)vc + (row * n_kv + kh) * 160ul;                                     \
    vr[kt] = *(const G u32x2 *)(rec + 16ul * w + 8ul * hl);                                              \
    vdm[kt] = *(const G u32 *)(rec + 128ul + 4ul * w);                                                   \
  }
#endif
/* early: load the V slice before the scores (more registers, but the V loads overlap the K loads and the scores:
 * for the launches with fewer workgroups than CUs); otherwise after the first barrier (fewer registers). Only the
 * time of the loads differs. */
/* tr (or NULL; f16 KV only): a tree verify (TR_*): token j is at position pos + its depth and sees the keys 0 .. pos
 * - 1 and, at the positions pos + d, the rows pos + (its ancestor at depth d). The tiles of 16 keys that reach pos are
 * computed once per token of the workgroup (S^T: that token's K rows; P V: its V rows) and each pair keeps its own
 * token's results, so every pair gets exactly the bits of a lone token at its position (ie_attn_long per token). */
#if !defined(IE_KV_QUANT)
static inline u32 al_trow(const G u8 *tr, u32 j, u32 key, u32 pos) {
  return key < pos || key - pos > tr[TR_DEP + j] ? key : pos + tr[TR_PATH + 16u * j + (key - pos)];
}
#endif
static inline __attribute__((always_inline)) void al_body(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos, u32 n_head,
                    u32 n_kv, u32 T, u32 npg, G u32 *count, G u32 *gcount, u32 gmax, G u8 *oq, const G float *gate,
                    u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride, const int early, const G u8 *tr) {
#ifdef IE_KV_Q4
  tr = (const G u8 *)0; /* (the host does not ask for a tree with a 4-bit KV cache) */
#endif
#define AL_DJ(j) (tr ? (u32)tr[TR_DEP + (j)] : (j))
  const u32 grp = n_head / n_kv;
  /* One pair group per workgroup. Tried 2026-11-04: one workgroup looping over every pair group with its K and V kept
   * in registers (a verify of 6 tokens has 3 groups and reads the KV 3 times): 231-256 registers (spills without
   * them), 309 against 270 us at 27K positions. */
  const u32 wbeg = __builtin_amdgcn_readfirstlane(npg >> 5);
  npg = __builtin_amdgcn_readfirstlane(npg & 31u);
  const u32 wskip = __builtin_amdgcn_readfirstlane(wbeg >= 256u ? wbeg / AL_CH - 1u : 0u);
  const u32 pg = __builtin_amdgcn_readfirstlane(wgid() % npg);
  const u32 kh = __builtin_amdgcn_readfirstlane((wgid() / npg) % n_kv);
  const u32 sg = __builtin_amdgcn_readfirstlane(wgid() / (npg * n_kv));
  const u32 psg = __builtin_amdgcn_readfirstlane(sg ? sg + wskip : 0u);
  const u32 w = tid() >> 5, l = lane(), hl = l >> 4, rl = l & 15u;
  const u32 npair = __builtin_amdgcn_readfirstlane(grp * T);
  const u32 kend = __builtin_amdgcn_readfirstlane(pos + (tr ? (u32)tr[368] : T)); /* keys 0 .. kend - 1 exist */
  const u32 c0 = __builtin_amdgcn_readfirstlane(psg * AL_CH);
  if (c0 >= kend) return; /* workgroup-uniform */
  const v8f zero = (v8f)(0.0f);
  const unsigned long pst = pstride;
  /* K (the A operand: key rl, dims 128 hl + 8 ks .. + 7 for the step ks) of the wave: loads first. Rows past
   * kend - 1 are replaced by the last row (their weights are 0) */
  const u32 kb = c0 + 16u * w;
  const int kon = kb < kend; /* wave-uniform */
#if !defined(IE_KV_QUANT)
  h8 kr[16];
#elif defined(IE_KV_Q8)
  u32x4 kq[8]; /* the int8 values of the 8 q8_0 blocks (dims 32 blk + 16 hl .. + 15) and the 8 fp16 block scales */
  u32x4 ksc4 = (u32x4)(0u);
#else
  u32x4 kq[4]; /* the nibbles of the dims 128 hl .. 128 hl + 127 of the key (one u32x4 per block of 32), and the (d, m) pairs */
  u32 kdm[4];
#endif
  if (kon) {
    const unsigned long row = kb + rl < kend ? kb + rl : kend - 1u;
#if !defined(IE_KV_QUANT)
    const G h8 *kp = (const G h8 *)(kc + (row * n_kv + kh) * 256ul + 128ul * hl);
#pragma unroll
    for (u32 ks = 0; ks < 16u; ks++) kr[ks] = kp[ks];
#elif defined(IE_KV_Q8)
    al_q8_kload(kc, row, n_kv, kh, hl, kq, &ksc4);
#else
    const G u8 *rec = (const G u8 *)kc + (row * n_kv + kh) * 160ul;
#pragma unroll
    for (u32 i = 0; i < 4u; i++) kq[i] = ((const G u32x4 *)(rec + 64ul * hl))[i];
#pragma unroll
    for (u32 i = 0; i < 4u; i++) kdm[i] = ((const G u32 *)(rec + 128ul + 16ul * hl))[i];
#endif
  }
  AL_VDECL
  if (early) {
    AL_VLOAD
  }
  /* this lane's column of S^T (pair 16 pg + rl) */
  const u32 ci = 16u * pg + rl;
  const int clive = ci < npair;
  const u32 cj = clive ? ci / grp : 0u, cpj = pos + AL_DJ(cj);
  const int cval = clive && c0 <= cpj; /* the pair has keys in this split */
  /* Q * (1 / 16) (exact) as fp16 hi + lo, rows of 256 halves (AL_QS apart), 16 pairs; dead pairs 0 */
  {
    const u32 r = tid() >> 4, d0 = 16u * (tid() & 15u), i = 16u * pg + r;
    h8 hi0 = (h8)(0.0f), hi1 = hi0, lo0 = hi0, lo1 = hi0;
    if (i < npair) {
      const u32 j = i / grp, h = kh * grp + i % grp;
      const G float *qp = (const G float *)((const G u8 *)q + (unsigned long)j * qst) + (unsigned long)h * 256ul + d0;
      f32x4 x[4];
#pragma unroll
      for (u32 u = 0; u < 4u; u++) x[u] = ((const G f32x4 *)qp)[u];
#pragma unroll
      for (u32 u = 0; u < 4u; u++)
#pragma unroll
        for (u32 e = 0; e < 4u; e++) {
          const float v = x[u][e] * 0.0625f;
          const f16 hv = (f16)v, lv = (f16)(v - (float)hv);
          if (u < 2u) hi0[4u * u + e] = hv, lo0[4u * u + e] = lv;
          else hi1[4u * (u - 2u) + e] = hv, lo1[4u * (u - 2u) + e] = lv;
        }
    }
    LDSP f16 *qh = (LDSP f16 *)al_lds, *ql = qh + 16u * AL_QS;
    *(LDSP h8 *)(qh + r * AL_QS + d0) = hi0;
    *(LDSP h8 *)(qh + r * AL_QS + d0 + 8u) = hi1;
    *(LDSP h8 *)(ql + r * AL_QS + d0) = lo0;
    *(LDSP h8 *)(ql + r * AL_QS + d0 + 8u) = lo1;
  }
  lbarrier();
  /* S^T: rows keys 8 hl + v of the tile, column pair rl */
  v8f s = zero;
#ifdef IE_KV_Q8
  if (kon && !(AL_DBG & 4)) {
    const LDSP f16 *qh = (const LDSP f16 *)al_lds, *ql = qh + 16u * AL_QS;
    s = al_q8_kqt(kc, kq, ksc4, qh, ql, tr, pos, kb, kend, pg, grp, T, cj, n_kv, kh, hl, rl);
  }
#else
#if !defined(IE_KV_QUANT)
  if (tr && kon && kb + 15u >= pos && !(AL_DBG & 4)) { /* a tile of the tree: once per token of the workgroup */
    const LDSP f16 *qh = (const LDSP f16 *)al_lds, *ql = qh + 16u * AL_QS;
    const u32 jlo = 16u * pg / grp, jhi = (16u * pg + 15u) / grp < T - 1u ? (16u * pg + 15u) / grp : T - 1u;
    for (u32 jj = jlo; jj <= jhi; jj++) {
      const unsigned long row = al_trow(tr, jj, kb + rl < kend ? kb + rl : kend - 1u, pos);
      const G h8 *kp = (const G h8 *)(kc + (row * n_kv + kh) * 256ul + 128ul * hl);
#pragma unroll
      for (u32 ks = 0; ks < 16u; ks++) kr[ks] = kp[ks];
      v8f sj = zero;
#pragma unroll
      for (u32 ks = 0; ks < 16u; ks++) {
        const h8 bh = *(const LDSP h8 *)(qh + rl * AL_QS + 128u * hl + 8u * ks);
        const h8 bl = *(const LDSP h8 *)(ql + rl * AL_QS + 128u * hl + 8u * ks);
        sj = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kr[ks], bh, sj);
        sj = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kr[ks], bl, sj);
      }
      if (cj == jj) s = sj; /* this lane's pair is of token jj */
    }
  } else
#endif
  if (kon && !(AL_DBG & 4)) {
    const LDSP f16 *qh = (const LDSP f16 *)al_lds, *ql = qh + 16u * AL_QS;
#pragma unroll
    for (u32 ks = 0; ks < 16u; ks++) {
      const h8 bh = *(const LDSP h8 *)(qh + rl * AL_QS + 128u * hl + 8u * ks);
      const h8 bl = *(const LDSP h8 *)(ql + rl * AL_QS + 128u * hl + 8u * ks);
#if !defined(IE_KV_QUANT)
      s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kr[ks], bh, s);
      s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kr[ks], bl, s);
#else
      h8 kh8, kl8; /* dequantized (exact in f32) and split into fp16 hi + lo */
      {
        const u32 kw = kq[ks >> 2][ks & 3u]; /* the 8 nibbles of the dims 8 ks .. 8 ks + 7 */
        const float kd = al_lo(kdm[ks >> 2]), km = al_hi(kdm[ks >> 2]);
#pragma unroll
        for (u32 e = 0; e < 8u; e++) {
          const float x = __builtin_fmaf((float)((kw >> (4u * e)) & 15u), kd, km);
          kh8[e] = (f16)x;
          kl8[e] = (f16)(x - (float)kh8[e]);
        }
      }
      s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kh8, bh, s);
      s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kh8, bl, s);
      if (!(AL_DBG & 8)) s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kl8, bh, s);
#endif
    }
  }
#endif
  const u32 cmax = (wbeg && !sg) ? 63u : cpj;
  float sv[8], mx = -__builtin_inff();
#pragma unroll
  for (u32 v = 0; v < 8u; v++) {
    sv[v] = kb + 8u * hl + v <= cmax ? s[v] : -__builtin_inff();
    mx = __builtin_fmaxf(mx, sv[v]);
  }
  mx = __builtin_fmaxf(mx, I2F(xrow_i(F2I(mx)))); /* the 16 keys of the tile */
  if (hl == 0u) al_red[0][16u * w + rl] = mx;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory"); /* all the Q reads of this wave are done */
  lbarrier();
  float M = al_red[0][rl];
#pragma unroll
  for (u32 x = 1; x < 8u; x++) M = __builtin_fmaxf(M, al_red[0][16u * x + rl]);
  /* the V slice of the wave (key rl, columns 32 w + 16 hl ..), loaded only now so that K and V are not in registers
   * together (fewer registers) */
  if (!early) {
    AL_VLOAD
  }
  /* P = exp(S - M) (0 for masked keys), the sum, P as fp16 hi + lo (A layout: pair rl, keys 8 hl .. + 7 of tile w) */
  h8 ph8, pl8;
  float ls = 0.0f;
#pragma unroll
  for (u32 v = 0; v < 8u; v++) {
    const float p = kb + 8u * hl + v <= cmax ? __builtin_expf(sv[v] - M) : 0.0f;
    ls += p;
    const f16 hv = (f16)p;
    ph8[v] = hv;
    pl8[v] = (f16)(p - (float)hv);
  }
  ls += I2F(xrow_i(F2I(ls)));
  if (hl == 0u) al_red[1][16u * w + rl] = ls;
  {
    LDSP f16 *pp = (LDSP f16 *)al_lds;
    *(LDSP h8 *)(pp + rl * AL_PS + 16u * w + 8u * hl) = ph8;
    *(LDSP h8 *)(pp + 16u * AL_PS + rl * AL_PS + 16u * w + 8u * hl) = pl8;
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  lbarrier();
  float L = al_red[1][rl];
#pragma unroll
  for (u32 x = 1; x < 8u; x++) L += al_red[1][16u * x + rl];
  /* O = P V for the wave's columns 32 w + 16 dt + rl: the V slice of each key tile through a private transpose */
  v8f acc[2] = {zero, zero};
  {
    LDSP f16 *sc = (LDSP f16 *)(al_lds + AL_OFF_SC) + w * (AL_SCP * 32u * AL_SS);
    const LDSP f16 *pp = (const LDSP f16 *)al_lds;
#pragma unroll
    for (u32 kt = 0; kt < 8u; kt++)
      if (c0 + 16u * kt < kend && !(AL_DBG & 2)) {
#if !defined(IE_KV_QUANT)
        if (tr && c0 + 16u * kt + 15u >= pos) { /* a tile of the tree: per token, its V rows; each row keeps its token's */
          const u32 jlo = 16u * pg / grp, jhi = (16u * pg + 15u) / grp < T - 1u ? (16u * pg + 15u) / grp : T - 1u;
          const v8f a0in = acc[0], a1in = acc[1];
          const h8 ah = *(const LDSP h8 *)(pp + rl * AL_PS + 16u * kt + 8u * hl);
          const h8 al = *(const LDSP h8 *)(pp + 16u * AL_PS + rl * AL_PS + 16u * kt + 8u * hl);
          for (u32 jj = jlo; jj <= jhi; jj++) {
            const unsigned long key = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;
            const unsigned long row = al_trow(tr, jj, (u32)key, pos);
            const G h8 *vp = (const G h8 *)(vc + (row * n_kv + kh) * 256ul + 32ul * w + 16ul * hl);
            const h8 v0 = vp[0], v1 = vp[1];
#pragma unroll
            for (u32 e = 0; e < 8u; e++) {
              sc[(16u * hl + e) * AL_SS + rl] = v0[e];
              sc[(16u * hl + 8u + e) * AL_SS + rl] = v1[e];
            }
            __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
            const h8 b0 = *(const LDSP h8 *)(sc + rl * AL_SS + 8u * hl);
            const h8 b1 = *(const LDSP h8 *)(sc + (16u + rl) * AL_SS + 8u * hl);
            v8f t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, b0, a0in);
            t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(al, b0, t0);
            v8f t1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, b1, a1in);
            t1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(al, b1, t1);
#pragma unroll
            for (u32 v = 0; v < 8u; v++)
              if ((16u * pg + 8u * hl + v) / grp == jj) acc[0][v] = t0[v], acc[1][v] = t1[v];
            __asm__ volatile("s_wait_dscnt 0x0" ::: "memory"); /* the reads before the next pass's writes */
          }
          continue;
        }
#pragma unroll
        for (u32 e = 0; e < 8u; e++) {
          sc[(16u * hl + e) * AL_SS + rl] = vr[kt][0][e];
          sc[(16u * hl + 8u + e) * AL_SS + rl] = vr[kt][1][e];
        }
#elif defined(IE_KV_Q8)
        al_q8_pv_tile(vc, vr[kt], vsc[kt], pp, sc, tr, pos, c0, kt, kend, pg, grp, T, n_kv, kh, w, hl, rl, acc);
        continue;
#else
        LDSP f16 *scl = sc + 32u * AL_SS;
        const float vd = al_lo(vdm[kt]), vm = al_hi(vdm[kt]);
#pragma unroll
        for (u32 e = 0; e < 16u; e++) {
          const float x = __builtin_fmaf((float)((vr[kt][e >> 3] >> (4u * (e & 7u))) & 15u), vd, vm);
          const f16 hv = (f16)x;
          sc[(16u * hl + e) * AL_SS + rl] = hv;
          scl[(16u * hl + e) * AL_SS + rl] = (f16)(x - (float)hv);
        }
#endif
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        const h8 b0 = *(const LDSP h8 *)(sc + rl * AL_SS + 8u * hl);
        const h8 b1 = *(const LDSP h8 *)(sc + (16u + rl) * AL_SS + 8u * hl);
        const h8 ah = *(const LDSP h8 *)(pp + rl * AL_PS + 16u * kt + 8u * hl);
        const h8 al = *(const LDSP h8 *)(pp + 16u * AL_PS + rl * AL_PS + 16u * kt + 8u * hl);
        acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, b0, acc[0]);
        acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(al, b0, acc[0]);
        acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, b1, acc[1]);
        acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(al, b1, acc[1]);
#ifdef IE_KV_Q4
        const h8 c0l = *(const LDSP h8 *)(scl + rl * AL_SS + 8u * hl);
        const h8 c1l = *(const LDSP h8 *)(scl + (16u + rl) * AL_SS + 8u * hl);
        acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, c0l, acc[0]);
        acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, c1l, acc[1]);
#endif
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory"); /* the reads before the next tile's writes */
      }
  }
  /* the split's partial results: o rows 8 hl + v (pair 16 pg + 8 hl + v), columns 32 w + 16 dt + rl */
#pragma unroll
  for (u32 v = 0; v < 8u; v++) {
    const u32 pi = 16u * pg + 8u * hl + v;
    if (pi < npair) {
      const u32 j = pi / grp, h = kh * grp + pi % grp;
      if (c0 <= pos + AL_DJ(j)) {
        const u32 ns = al_ns(pos + AL_DJ(j)) - wskip;
        G float *pp = part + j * pst + ((unsigned long)h * (ns + al_ng(ns)) + sg) * AL_PSZ + 4u;
        pp[32u * w + rl] = acc[0][v];
        pp[32u * w + 16u + rl] = acc[1][v];
      }
    }
  }
  if (w == 0u && hl == 0u && cval) {
    const u32 h = kh * grp + ci % grp, ns = al_ns(cpj) - wskip;
    G float *pp = part + cj * pst + ((unsigned long)h * (ns + al_ng(ns)) + sg) * AL_PSZ;
    pp[0] = M;
    pp[1] = L;
  }
  /* make the stores visible, count the split into its group; the last one of a group merges the group. The counters
   * are device-scope relaxed atomics between agent-scope fences (a plain __atomic_ acq_rel is system scope: a cache
   * write-back before it and a cache invalidate after it, for every workgroup) */
  if (AL_DBG & 1) return;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (w == 0u && hl == 0u) {
    u32 last = 0u;
    if (cval) {
      const u32 ns = al_ns(cpj) - wskip, g = sg / AL_G, in_g = ns - g * AL_G < AL_G ? ns - g * AL_G : AL_G;
      const unsigned long gi = ((unsigned long)cj * n_head + kh * grp + ci % grp) * gmax + g;
      last = __scoped_atomic_fetch_add(&gcount[gi], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == in_g;
    }
    al_last[rl] = last;
  }
  barrier();
  for (u32 x = w; x < 16u; x += 8u) {
    if (!al_last[x]) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const u32 pi = 16u * pg + x, j = pi / grp, h = kh * grp + pi % grp, ns = al_ns(pos + AL_DJ(j)) - wskip, ng = al_ng(ns), g = sg / AL_G;
    const u32 in_g = ns - g * AL_G < AL_G ? ns - g * AL_G : AL_G;
    G float *pb = part + j * pst + (unsigned long)h * (ns + ng) * AL_PSZ; /* this pair's splits, then its groups */
    G float *ox = out ? (G float *)((G u8 *)out + (unsigned long)j * ost) + (unsigned long)h * 256ul : (G float *)0;
    G u8 *oqx = oq ? oq + (unsigned long)j * oqst : oq;
    const G float *gxh = gate ? (const G float *)((const G u8 *)gate + (unsigned long)j * gst) + (unsigned long)h * gstride : gate;
    float M2, L2;
    f32x4 a0, a1;
    al_msum(pb + (unsigned long)g * AL_G * AL_PSZ, in_g, &M2, &L2, &a0, &a1);
    if (l == 0u) gcount[((unsigned long)j * n_head + h) * gmax + g] = 0u;
    if (ng == 1u) {
      al_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_head * 8u, (LDSP float *)al_lds + w * 256u);
      continue;
    }
    G float *gp = pb + (unsigned long)(ns + g) * AL_PSZ; /* the group's (M, L, o) */
    if (l == 0u) gp[0] = M2, gp[1] = L2;
    *(G f32x4 *)(gp + 4u + 4u * l) = a0;
    *(G f32x4 *)(gp + 132u + 4u * l) = a1;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    u32 done = 0u;
    if (l == 0u) done = __scoped_atomic_fetch_add(&count[j * n_head + h], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == ng;
    if (!__builtin_amdgcn_readfirstlane(done)) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    al_msum(pb + (unsigned long)ns * AL_PSZ, ng, &M2, &L2, &a0, &a1);
    if (l == 0u) count[j * n_head + h] = 0u;
    al_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_head * 8u, (LDSP float *)al_lds + w * 256u);
  }
}
#ifdef AL_EARLYV
#define AL_E0 1
#else
#define AL_E0 0
#endif
KERNEL ie_attn_long(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos, u32 n_head,
                    u32 n_kv, u32 T, u32 npg, G u32 *count, G u32 *gcount, u32 gmax, G u8 *oq, const G float *gate,
                    u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride, const G u8 *tr) {
  al_body(q, kc, vc, part, out, pos, n_head, n_kv, T, npg, count, gcount, gmax, oq, gate, gstride, qst, ost, oqst, gst, pstride, AL_E0, tr);
}
/* The same with the V loads first (launches of fewer than 64 workgroups, ie_attn_long_e, IE_ATTN_LONG_E=0: off). */
KERNEL ie_attn_long_e(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos, u32 n_head,
                      u32 n_kv, u32 T, u32 npg, G u32 *count, G u32 *gcount, u32 gmax, G u8 *oq, const G float *gate,
                      u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride, const G u8 *tr) {
  al_body(q, kc, vc, part, out, pos, n_head, n_kv, T, npg, count, gcount, gmax, oq, gate, gstride, qst, ost, oqst, gst, pstride, 1, tr);
}
#include "ie_attn_long128.inc" /* the same for heads of 128 (Qwen3) */
#include "ie_attn_long_mp_q8.inc" /* single-KV-pass multi-pair-group Q8 long attention (npg = 2 .. 3) */
#endif

/* ----------------------------------------------------------------- qwen35 */

/* Full attention, before the attention: workgroup g < nq: query head g, at
 * qkv + g * 2 hd (q of [q | gate]); nq <= g < nq + nkv: key head g - nq,
 * at qkv + 2 nq hd + (g - nq) hd; then nkv workgroups copy the value heads.
 * A q or k head: y = rmsnorm(x) * w (hd <= 256 values, thread t holds value
 * t), then RoPE of the first nrot values (pairs (i, i + nrot/2)); q goes to
 * qo + g * hd, k to the cache kc. */
static LDS float qk_buf[256];
KERNEL ie_qkn_rope_kv(const G float *qkv, G float *qo, G kvt *kc, G kvt *vc, const G float *qw, const G float *kw,
                      const G float *cs, const G float *sn, u32 nq, u32 nkv, u32 hd, u32 nrot, float eps, u32 ins,
                      u32 qos, u32 qstr) {
  /* token y of a multi-token launch: position + y (KV rows and RoPE tables) */
  qkv = TOKC(qkv, ins), qo = TOK(qo, qos);
#ifdef IE_KV_QUANT
  G u8 *kc8 = TOK((G u8 *)kc, nkv * KVR(hd)), *vc8 = TOK((G u8 *)vc, nkv * KVR(hd));
#else
  kc = TOK(kc, nkv * hd * (u32)sizeof(kvt)), vc = TOK(vc, nkv * hd * (u32)sizeof(kvt));
#endif
  cs = TOKC(cs, (nrot >> 1) * 4u), sn = TOKC(sn, (nrot >> 1) * 4u);
  const u32 g = wgid(), t = tid();
  if (g >= nq + nkv) { /* value head */
    const u32 h = g - nq - nkv;
#ifdef IE_KV_QUANT
    kvq_store(t < hd ? qkv[nq * qstr + nkv * hd + h * hd + t] : 0.0f, vc8 + h * KVR(hd), hd, t);
#else
    if (t < hd) vc[h * hd + t] = qkv[nq * qstr + nkv * hd + h * hd + t];
#endif
    return;
  }
  const int is_k = g >= nq;
  const u32 h = is_k ? g - nq : g;
  const G float *src = is_k ? qkv + nq * qstr + h * hd : qkv + h * qstr;
  const G float *w = is_k ? kw : qw;
  const float x = t < hd ? src[t] : 0.0f;
  const float ss = wg_sum(x * x);
  const float y = x * (1.0f / __builtin_sqrtf(ss / (float)hd + eps)) * (t < hd ? w[t] : 0.0f);
  if (t < hd) qk_buf[t] = y;
  barrier();
  float r = y;
  const u32 h2 = nrot >> 1;
  if (t < nrot) {
    const u32 i = t < h2 ? t : t - h2;
    const float a = qk_buf[i], b = qk_buf[i + h2];
    r = t < h2 ? a * cs[i] - b * sn[i] : a * sn[i] + b * cs[i];
  }
#ifdef IE_KV_QUANT
  if (is_k) kvq_store(t < hd ? r : 0.0f, kc8 + h * KVR(hd), hd, t);
  else if (t < hd) qo[h * hd + t] = r;
#else
  if (t < hd) {
    if (is_k) kc[h * hd + t] = (kvt)r;
    else qo[h * hd + t] = r;
  }
#endif
}

/* The rotated KV cache (IE_KV_ROT=1, any cache format): q, k and v of a head go through the normalized Walsh-Hadamard
 * transform H (hd = 32 .. 256, a power of two; H is orthogonal and its own inverse) before the cache and the attention,
 * so q . k is unchanged and the outliers of a head spread over all its values before quantization. The attention then
 * yields H o; ie_attn_unrot applies H again, then the gate and the Q8 copy. Thread t holds value t; every thread of the
 * workgroup calls it (barriers). buf: hd floats of LDS. */
static inline float wht_wg(float r, u32 t, u32 hd, LDSP float *buf) {
  for (u32 s = 1u; s < 32u; s <<= 1) { /* within a wave */
    const float o = shfl_xor_f(r, s);
    r = (t & s) ? o - r : r + o;
  }
  for (u32 s = 32u; s < hd; s <<= 1) { /* across waves */
    barrier();
    if (t < hd) buf[t] = r;
    barrier();
    const float o = t < hd ? buf[t ^ s] : 0.0f;
    r = (t & s) ? o - r : r + o;
  }
  return r * (1.0f / __builtin_sqrtf((float)hd));
}

/* -DIE_KV_FAKE_V4 (experiment, q8 build only): IE_KV_ROT=1 rotates v alone, rounds it to the 4-bit cache's grid (the
 * same d and m as kvq_store of the 4-bit build) and stores that in the q8 cache; q and k stay as without IE_KV_ROT.
 * The quality of 8-bit K with rotated 4-bit V, before a real mixed format exists. */
#if defined(IE_KV_FAKE_V4) && defined(IE_KV_Q8)
#define ROT_QK 0
static inline float fake_q4(float v) { /* every lane of the wave calls it */
  float mx = v, mn = v;
  for (u32 k = 16u; k >= 1u; k >>= 1) mx = __builtin_fmaxf(mx, shfl_xor_f(mx, k)), mn = __builtin_fminf(mn, shfl_xor_f(mn, k));
  const f16 dh = (f16)((mx - mn) / 15.0f), mh = (f16)mn;
  const float d = (float)dh, m = (float)mh;
  int qi = d != 0.0f ? (int)__builtin_roundf((v - m) / d) : 0;
  qi = qi > 15 ? 15 : qi < 0 ? 0 : qi;
  return __builtin_fmaf((float)qi, d, m);
}
#else
#define ROT_QK 1
#define fake_q4(v) (v)
#endif

/* ie_qkn_rope_kv with q, k and v rotated (wht_wg) after RoPE. */
KERNEL ie_qkn_rope_kv_rot(const G float *qkv, G float *qo, G kvt *kc, G kvt *vc, const G float *qw, const G float *kw,
                          const G float *cs, const G float *sn, u32 nq, u32 nkv, u32 hd, u32 nrot, float eps, u32 ins,
                          u32 qos, u32 qstr) {
  qkv = TOKC(qkv, ins), qo = TOK(qo, qos);
#ifdef IE_KV_QUANT
  G u8 *kc8 = TOK((G u8 *)kc, nkv * KVR(hd)), *vc8 = TOK((G u8 *)vc, nkv * KVR(hd));
#else
  kc = TOK(kc, nkv * hd * (u32)sizeof(kvt)), vc = TOK(vc, nkv * hd * (u32)sizeof(kvt));
#endif
  cs = TOKC(cs, (nrot >> 1) * 4u), sn = TOKC(sn, (nrot >> 1) * 4u);
  const u32 g = wgid(), t = tid();
  if (g >= nq + nkv) { /* value head */
    const u32 h = g - nq - nkv;
    const float v = fake_q4(wht_wg(t < hd ? qkv[nq * qstr + nkv * hd + h * hd + t] : 0.0f, t, hd, qk_buf));
#ifdef IE_KV_QUANT
    kvq_store(t < hd ? v : 0.0f, vc8 + h * KVR(hd), hd, t);
#else
    if (t < hd) vc[h * hd + t] = (kvt)v;
#endif
    return;
  }
  const int is_k = g >= nq;
  const u32 h = is_k ? g - nq : g;
  const G float *src = is_k ? qkv + nq * qstr + h * hd : qkv + h * qstr;
  const G float *w = is_k ? kw : qw;
  const float x = t < hd ? src[t] : 0.0f;
  const float ss = wg_sum(x * x);
  const float y = x * (1.0f / __builtin_sqrtf(ss / (float)hd + eps)) * (t < hd ? w[t] : 0.0f);
  if (t < hd) qk_buf[t] = y;
  barrier();
  float r = y;
  const u32 h2 = nrot >> 1;
  if (t < nrot) {
    const u32 i = t < h2 ? t : t - h2;
    const float a = qk_buf[i], b = qk_buf[i + h2];
    r = t < h2 ? a * cs[i] - b * sn[i] : a * sn[i] + b * cs[i];
  }
  if (ROT_QK) r = wht_wg(r, t, hd, qk_buf); /* its first barrier: every thread has read qk_buf */
#ifdef IE_KV_QUANT
  if (is_k) kvq_store(t < hd ? r : 0.0f, kc8 + h * KVR(hd), hd, t);
  else if (t < hd) qo[h * hd + t] = r;
#else
  if (t < hd) {
    if (is_k) kc[h * hd + t] = (kvt)r;
    else qo[h * hd + t] = r;
  }
#endif
}

/* After an attention of the rotated cache (run without gate and Q8 copy): workgroup h (grid y: token) takes head h of
 * out (H o), applies H, the gate (gate: NULL for none) and writes out and its Q8 copy (oq: NULL for none). os, gs,
 * oqs: the byte strides of a token. */
KERNEL ie_attn_unrot(G float *out, G u8 *oq, const G float *gate, u32 hd, u32 nh, u32 gstride, u32 os, u32 oqs, u32 gs) {
  out = TOK(out, os), oq = oq ? TOK(oq, oqs) : oq;
  if (gate) gate = TOKC(gate, gs);
  const u32 h = wgid(), t = tid();
  float o = wht_wg(t < hd ? out[h * hd + t] : 0.0f, t, hd, qk_buf);
  if (gate && t < hd) o = o * sigmoidf(gate[h * gstride + t]);
  if (t < hd) out[h * hd + t] = o;
  if (oq && t < hd) quant_wave(o, (h * hd + t) >> 5, oq, nh * hd / 32u);
}

/* Linear attention (Gated DeltaNet) of T <= 16 tokens at positions pos ..
 * pos + T - 1, one after the other; one workgroup per value head h (key
 * head kh = h % nk), state dims 128 x 128. Token t: in + t * is = [q k v (cd)
 * | z (inner) | beta (nv) | alpha (nv)], inner = nv * 128; output at out +
 * t * os, its Q8 copy at oq + t * qs. ring: the conv inputs of the last 32
 * positions [slot][cd] (slot = position % 32). S: ns (4 or 8) state slots [slot][h][i][j]
 * (i: key dim, j: value dim); the state is read from slot c and the state
 * after token t is written to slot (c + t) % ns (only for t >= wfrom: a prompt chunk keeps only its last state), so that a speculative step
 * can go back to the state after any of its tokens. Each thread reads and
 * writes only its own state values, so slot c can be overwritten. Thread t:
 * value dim j = t % 128, key dims i of half t / 128. The CPU backend (cpu.c,
 * gdn) does the same arithmetic for one token (slot c = 0).
 * rec (or NULL): instead of a state per token, the verify run keeps one slot
 * (the state after token 0 is written to it) and, for each token t >= 1 and
 * each head, a record of what the state update used: rec + ((t - 1) * nv + h)
 * * GDN_RECF floats = [decay, 3 unused | k (128) | delta (128)]. ie_gdn_replay
 * applies the records of the accepted tokens to the slot afterwards. */
#define GDN_RING 32u
#define GDN_RECF 260u
#define GDN_RECK 4u
#define GDN_RECD 132u
static LDS float gd_q[128], gd_k[128], gd_r[8];
/* conv input of channel c at token u of this launch, or (before the launch)
 * from the ring; 0 before position 0 */
static inline float conv_in(const G float *in, u32 is, const G float *ring, u32 c, u32 t, u32 d, u32 pos, u32 cd) {
  if (d <= t) return in[(t - d) * is + c];
  const u32 p = pos + t;
  return p >= d ? ring[((p - d) & (GDN_RING - 1u)) * cd + c] : 0.0f;
}
/* the same for node t of a tree verify: the input d levels up its path (d <= its depth) or, before the root, the
 * ring at the position d before it */
static inline float conv_in_tr(const G float *in, u32 is, const G float *ring, u32 c, u32 t, u32 d, u32 pos, u32 cd,
                               const G u8 *tr) {
  if (d == 0u) return in[t * is + c];
  const u32 a = tr[TR_A1 + 16u * (d - 1u) + t];
  if (a != 0xFFu) return in[a * is + c];
  const u32 p = pos + tr[TR_DEP + t];
  return p >= d ? ring[((p - d) & (GDN_RING - 1u)) * cd + c] : 0.0f;
}
static inline float conv_ch_tr(const G float *in, u32 is, const G float *ring, const G float *cw, u32 c, u32 t, u32 pos,
                               u32 cd, const G u8 *tr) {
  float acc = 0.0f;
  for (u32 j = 0; j < 4u; j++) acc += conv_in_tr(in, is, ring, c, t, 3u - j, pos, cd, tr) * cw[c * 4u + j];
  return acc / (1.0f + __builtin_expf(-acc));
}
static inline float conv_ch(const G float *in, u32 is, G float *ring, const G float *cw, u32 c, u32 t, u32 pos, u32 cd,
                            int writer) { /* writer = 0: the host stores the ring after the launch (ie_ring_store) */
  float acc = 0.0f;
  for (u32 j = 0; j < 4u; j++) /* oldest first, as ggml_ssm_conv */
    acc += conv_in(in, is, ring, c, t, 3u - j, pos, cd) * cw[c * 4u + j];
  /* slot (pos + t) % GDN_RING: no workgroup reads it in this launch (reads
   * of the ring are of positions pos - 3 .. pos - 1, writes of pos .. pos +
   * 15: 19 slots < GDN_RING) */
  if (writer) ring[((pos + t) & (GDN_RING - 1u)) * cd + c] = in[t * is + c];
  return acc / (1.0f + __builtin_expf(-acc));
}
/* Linear attention for 1 .. 4 tokens (decode, speculative verify): one
 * workgroup per value head, the gated RMSNorm and the Q8 copy fused (no
 * second launch). Thread t: value dim j = t % 128, key dims i of half t /
 * 128. Same state slots and ring as ie_gdn. */
static LDS float gd1_v[128], gd1_p[2][128];
/* GDN_WRSPLIT (default 1): the state sweep that may store the state is compiled twice, with the store and without.
 * With `if (wr)` inside the unrolled sweep every element got its own branch, and no LDS load could move above one:
 * each of the 64 k (and q) loads waited alone (~5 us per token at Saluki's shapes). */
#ifndef GDN_WRSPLIT
#define GDN_WRSPLIT 1
#endif
#define GDN_SWEEP_UPD(K, Q, WR)                                                                                               \
  for (u32 ii = 0; ii < 64u; ii++) {                                                                                      \
    st[ii] += K[half * 64u + ii] * delta;                                                                                 \
    if (WR) Sw[ii * 128u] = st[ii];                                                                                       \
    po += st[ii] * Q[half * 64u + ii];                                                                                    \
  }
/* Inside the token loop the threads share only LDS, so its barriers are LDS barriers (lbarrier: no wait for the
 * global stores of the records and outputs). */
KERNEL ie_gdn1(const G float *in, G float *out, G u8 *oq, G float *ring, G float *S, const G float *cw, const G float *dtb,
               const G float *sa, const G float *nw, u32 pos, u32 cd, u32 nk, u32 nv, float eps, u32 T, u32 is, u32 os,
               u32 qs, u32 c, u32 wfrom, u32 rw, u32 ns, G float *rec) {
  const u32 h = wgid(), t0 = tid(), j = t0 & 127u, half = t0 >> 7, kh = h % nk, w = t0 >> 5, inner = nv * 128u;
  const unsigned long slot = (unsigned long)nv * 16384u, own = (unsigned long)h * 16384u + half * 64u * 128u + j;
  const u32 ca = half ? nk * 128u + kh * 128u + j : kh * 128u + j, cv = 2u * nk * 128u + h * 128u + j;
  float st[64];
  for (u32 ii = 0; ii < 64u; ii++) st[ii] = pos ? S[c * slot + own + ii * 128u] : 0.0f;
  const float dth = dtb[h], sah = sa[h], nwj = nw[j];
  for (u32 t = 0; t < T; t++) {
    const G float *it = in + t * is;
    const float a = conv_ch(in, is, ring, cw, ca, t, pos, cd, rw && h == kh);
    const float vv = half ? 0.0f : conv_ch(in, is, ring, cw, cv, t, pos, cd, rw);
    const float ss = wave_sum_all(a * a);
    if (lane() == 0u) gd_r[w] = ss;
    lbarrier();
    const float n2 = half ? gd_r[4] + gd_r[5] + gd_r[6] + gd_r[7] : gd_r[0] + gd_r[1] + gd_r[2] + gd_r[3];
    const float an = a * (1.0f / __builtin_sqrtf(n2 + eps));
    G float *rt = rec && t ? rec + ((unsigned long)(t - 1u) * nv + h) * GDN_RECF : (G float *)0; /* this token's record */
    if (half) {
      gd_k[j] = an;
      if (rt) rt[GDN_RECK + j] = an;
    } else gd_q[j] = an, gd1_v[j] = vv;
    const float beta = sigmoidf(it[cd + inner + h]);
    const float al = it[cd + inner + nv + h] + dth;
    const float sp = al > 20.0f ? al : __builtin_logf(1.0f + __builtin_expf(al));
    const float decay = __builtin_expf(sp * sah);
    if (rt && t0 == 0u) rt[0] = decay;
    lbarrier();
    float pk = 0.0f;
    for (u32 ii = 0; ii < 64u; ii++) {
      st[ii] = st[ii] * decay;
      pk += st[ii] * gd_k[half * 64u + ii];
    }
    gd1_p[half][j] = pk;
    lbarrier();
    const float delta = (gd1_v[j] - (gd1_p[0][j] + gd1_p[1][j])) * beta;
    if (rt && !half) rt[GDN_RECD + j] = delta;
    lbarrier();
    float po = 0.0f;
    G float *Sw = S + ((c + t) % ns) * slot + own;
    const int wr = t >= wfrom && (!rec || t == 0u); /* uniform; with records only token 0's state is kept */
    if (GDN_WRSPLIT && wr) GDN_SWEEP_UPD(gd_k, gd_q, 1) else GDN_SWEEP_UPD(gd_k, gd_q, GDN_WRSPLIT ? 0 : wr)
    gd1_p[half][j] = po;
    lbarrier();
    const float o = (gd1_p[0][j] + gd1_p[1][j]) * (1.0f / __builtin_sqrtf(128.0f));
    const float s2 = wave_sum_all(o * o);
    if (lane() == 0u) gd_r[w] = s2;
    lbarrier();
    if (!half) { /* whole waves */
      const float n3 = gd_r[0] + gd_r[1] + gd_r[2] + gd_r[3];
      const float z = it[cd + h * 128u + j];
      const float y = o * (1.0f / __builtin_sqrtf(n3 / 128.0f + eps)) * nwj * (z / (1.0f + __builtin_expf(-z)));
#ifdef IE_MEGA
      out[t * os + h * 128u + j] = y;
#else
      if (out) out[t * os + h * 128u + j] = y;
#endif
      if (oq) quant_wave(y, (h * 128u + j) >> 5, oq + t * qs, inner >> 5);
    }
    lbarrier();
  }
}

#ifndef IE_MEGA
/* ie_gdn1 for 2 .. 16 tokens with the state-free work of all tokens done ahead of the state loop: the conv, the L2
 * norms of q and k, beta and the decay of every token are computed together (their loads in flight at once, the
 * ring stores after them) and kept in LDS, so the loop over the tokens holds only the two state sweeps and 2
 * barriers (ie_gdn1: global load waits and 5 barriers per token); the gated norm and the Q8 copy of all tokens run
 * after the loop, a token per wave. Same arguments, thread mapping, records and arithmetic as ie_gdn1 (each value
 * is computed by the same operations in the same order, the norm sums over the same 4 wave sums). */
/* tr (or NULL): a tree verify (TR_*; records mode): node t's conv inputs are its ancestors', and a node whose parent
 * is not node t - 1 starts from the state after the root (the slot) with the records of its parent's path applied
 * (as ie_gdn_replay: the state the parent had). The ring gets node t's inputs at position pos + t (src/hip.c moves
 * the accepted path's into place). */
#ifdef IE_SALUKI_GDN1P
#define GDN1P_KERNEL __attribute__((amdgpu_kernel, visibility("protected"))) void ie_saluki_gdn1p
#define IE_GDN1P_TCAP 8u
#else
#define GDN1P_KERNEL KERNEL ie_gdn1p
#define IE_GDN1P_TCAP 16u
#endif
static LDS float gp_q[IE_GDN1P_TCAP][128], gp_k[IE_GDN1P_TCAP][128], gp_v[IE_GDN1P_TCAP][128], gp_r[IE_GDN1P_TCAP][8],
    gp_bd[IE_GDN1P_TCAP][2], gp_px[2][128], gp_po[2][128], gp_d[IE_GDN1P_TCAP][128];
/* GDN1P_FUSE (default 1; not for a tree verify): one sweep over the state per token instead of two. Token t + 1's decay
 * and k sum need only the state after token t's update, so the sweep that updates the state for token t and sums its
 * output also applies token t + 1's decay and sums its k product: two independent chains per sweep and one barrier per
 * token (gp_px2 / gp_po2 alternate between tokens). Each value takes the same operations in the same order. */
#ifndef GDN1P_FUSE
#define GDN1P_FUSE 1
#endif
static LDS float gp_px2[2][2][128], gp_po2[2][2][128];
#include "ie_gdn_lazy.inc"
GDN1P_KERNEL(const G float *in, G float *out, G u8 *oq, G float *ring, G float *S, const G float *cw, const G float *dtb,
                const G float *sa, const G float *nw, u32 pos, u32 cd, u32 nk, u32 nv, float eps, u32 T, u32 is, u32 os,
                u32 qs, u32 c, u32 wfrom, u32 rw, u32 ns, G float *rec, const G u8 *tr,
                const G float *lazy_rec, u32 lazy_mask, unsigned long rls, u32 r_idx) {
#ifdef IE_SALUKI_GDN1P
  /* The host checks the eight-token linear Saluki records geometry. Keep
   * deferred replay and the token-zero store identical to the base path. */
  if (!rec) return;
  T = 8u;
  tr = (const G u8 *)0;
  nk = 16u;
  nv = 48u;
  cd = 10240u;
  c = wfrom = 0u;
  rw = ns = 1u;
#endif
  const u32 h = wgid(), t0 = tid(), j = t0 & 127u, half = t0 >> 7, kh = h % nk, w = t0 >> 5, inner = nv * 128u;
  const unsigned long slot = (unsigned long)nv * 16384u, own = (unsigned long)h * 16384u + half * 64u * 128u + j;
  const u32 ca = half ? nk * 128u + kh * 128u + j : kh * 128u + j, cv = 2u * nk * 128u + h * 128u + j;
  G float *S_slot_own = S + c * slot + own;
  float st[64];
  for (u32 ii = 0; ii < 64u; ii++) st[ii] = (pos || lazy_mask) ? S_slot_own[ii * 128u] : 0.0f;
  ie_gdn_lazy_apply(st, S_slot_own, lazy_rec, lazy_mask, rls, r_idx, nv, h, half, j, T > 0u && wfrom == 0u);
  /* conv + SiLU of every token; the sums of squares per wave */
#pragma unroll 4
  for (u32 t = 0; t < T; t++) {
    const float a = tr ? conv_ch_tr(in, is, ring, cw, ca, t, pos, cd, tr) : conv_ch(in, is, ring, cw, ca, t, pos, cd, 0);
    const float vv = half ? 0.0f : tr ? conv_ch_tr(in, is, ring, cw, cv, t, pos, cd, tr) : conv_ch(in, is, ring, cw, cv, t, pos, cd, 0);
    const float ss = wave_sum_all(a * a);
    if (lane() == 0u) gp_r[t][w] = ss;
    if (half) gp_k[t][j] = a; else gp_q[t][j] = a;
    if (!half) gp_v[t][j] = vv;
  }
  if (t0 < T) { /* thread t: beta and decay of token t */
    const G float *it = in + t0 * is;
    const float beta = sigmoidf(it[cd + inner + h]);
    const float al = it[cd + inner + nv + h] + dtb[h];
    const float sp = al > 20.0f ? al : __builtin_logf(1.0f + __builtin_expf(al));
    const float decay = __builtin_expf(sp * sa[h]);
    gp_bd[t0][0] = beta, gp_bd[t0][1] = decay;
    if (rec && t0) rec[((unsigned long)(t0 - 1u) * nv + h) * GDN_RECF] = decay;
  }
  if (rw) /* the conv inputs of this launch into the ring (slots no workgroup reads in this launch, as in ie_gdn1) */
    for (u32 t = 0; t < T; t++) {
      if (h == kh) ring[((pos + t) & (GDN_RING - 1u)) * cd + ca] = in[t * is + ca];
      if (!half) ring[((pos + t) & (GDN_RING - 1u)) * cd + cv] = in[t * is + cv];
    }
  lbarrier();
  for (u32 t = 0; t < T; t++) { /* the L2 norm of q and k */
    const float n2 = half ? gp_r[t][4] + gp_r[t][5] + gp_r[t][6] + gp_r[t][7] : gp_r[t][0] + gp_r[t][1] + gp_r[t][2] + gp_r[t][3];
    const float an = (half ? gp_k[t][j] : gp_q[t][j]) * (1.0f / __builtin_sqrtf(n2 + eps));
    if (half) {
      gp_k[t][j] = an;
      if (rec && t) rec[((unsigned long)(t - 1u) * nv + h) * GDN_RECF + GDN_RECK + j] = an;
    } else gp_q[t][j] = an;
  }
  lbarrier();
#if GDN1P_FUSE && !defined(IE_SALUKI_GDN1P)
  if (!tr) { /* see GDN1P_FUSE */
    float pk = 0.0f;
    {
      const float decay = gp_bd[0][1];
      for (u32 ii = 0; ii < 64u; ii++) {
        st[ii] = st[ii] * decay;
        pk += st[ii] * gp_k[0][half * 64u + ii];
      }
    }
    gp_px2[0][half][j] = pk;
    lbarrier();
    for (u32 t = 0; t < T; t++) {
      const u32 b = t & 1u;
      G float *rt = rec && t ? rec + ((unsigned long)(t - 1u) * nv + h) * GDN_RECF : (G float *)0; /* this token's record */
      const float delta = (gp_v[t][j] - (gp_px2[b][0][j] + gp_px2[b][1][j])) * gp_bd[t][0];
      if (rt && !half) rt[GDN_RECD + j] = delta;
      float po = 0.0f, pn = 0.0f;
      G float *Sw = S + ((c + t) % ns) * slot + own;
      const int wr = t >= wfrom && (!rec || t == 0u); /* uniform; with records only token 0's state is kept */
      if (t + 1u < T) { /* uniform */
        const float dn = gp_bd[t + 1u][1];
#define GDN1P_SWEEP2(WR)                                                                                                  \
  for (u32 ii = 0; ii < 64u; ii++) {                                                                                      \
    st[ii] += gp_k[t][half * 64u + ii] * delta;                                                                           \
    if (WR) Sw[ii * 128u] = st[ii];                                                                                       \
    po += st[ii] * gp_q[t][half * 64u + ii];                                                                              \
    st[ii] = st[ii] * dn;                                                                                                 \
    pn += st[ii] * gp_k[t + 1u][half * 64u + ii];                                                                         \
  }
        if (GDN_WRSPLIT && wr) GDN1P_SWEEP2(1) else GDN1P_SWEEP2(GDN_WRSPLIT ? 0 : wr)
#undef GDN1P_SWEEP2
        gp_px2[b ^ 1u][half][j] = pn;
      } else if (GDN_WRSPLIT && wr) GDN_SWEEP_UPD(gp_k[t], gp_q[t], 1) else GDN_SWEEP_UPD(gp_k[t], gp_q[t], GDN_WRSPLIT ? 0 : wr)
      gp_po2[b][half][j] = po;
      lbarrier();
      if (!half) gp_q[t][j] = (gp_po2[b][0][j] + gp_po2[b][1][j]) * (1.0f / __builtin_sqrtf(128.0f));
    }
  } else
#endif
  /* The records-only specialization uses these shorter state sweeps. On
   * gfx1201 they trade extra barriers for lower register pressure. */
  for (u32 t = 0; t < T; t++) {
    const float beta = gp_bd[t][0], decay = gp_bd[t][1];
    G float *rt = rec && t ? rec + ((unsigned long)(t - 1u) * nv + h) * GDN_RECF : (G float *)0; /* this token's record */
    if (tr && t && tr[TR_PAR + t] != t - 1u) { /* uniform: the state of the parent, from the root's */
      const u32 pa = tr[TR_PAR + t];
      for (u32 ii = 0; ii < 64u; ii++) st[ii] = S[c * slot + own + ii * 128u];
      for (u32 m = pa ? tr_anc(tr, pa) : 0u; m; m &= m - 1u) {
        const u32 u = (u32)__builtin_ctz(m);
        const float du = gp_bd[u][1], dl = gp_d[u][j];
        for (u32 ii = 0; ii < 64u; ii++) {
          st[ii] = st[ii] * du;
          st[ii] = __builtin_fmaf(gp_k[u][half * 64u + ii], dl, st[ii]);
        }
      }
    }
    float pk = 0.0f;
    for (u32 ii = 0; ii < 64u; ii++) {
      st[ii] = st[ii] * decay;
      pk += st[ii] * gp_k[t][half * 64u + ii];
    }
    gp_px[half][j] = pk;
    lbarrier();
    const float delta = (gp_v[t][j] - (gp_px[0][j] + gp_px[1][j])) * beta;
    if (rt && !half) rt[GDN_RECD + j] = delta;
    if (tr && !half) gp_d[t][j] = delta; /* the later nodes' replays */
    float po = 0.0f;
    G float *Sw = S + ((c + t) % ns) * slot + own;
    const int wr = t >= wfrom && (!rec || t == 0u); /* uniform; with records only token 0's state is kept */
    if (GDN_WRSPLIT && wr) GDN_SWEEP_UPD(gp_k[t], gp_q[t], 1) else GDN_SWEEP_UPD(gp_k[t], gp_q[t], GDN_WRSPLIT ? 0 : wr)
    gp_po[half][j] = po;
    lbarrier(); /* also: gp_px of this token is read before the next token writes it */
    if (!half) gp_q[t][j] = (gp_po[0][j] + gp_po[1][j]) * (1.0f / __builtin_sqrtf(128.0f));
  }
  lbarrier();
  /* the gated RMSNorm and the Q8 copy: wave w takes tokens w, w + 8 */
  for (u32 t = w; t < T; t += 8u) {
    const u32 l = lane();
    float o[4], s[4];
    for (u32 q = 0; q < 4u; q++) o[q] = gp_q[t][32u * q + l], s[q] = wave_sum_all(o[q] * o[q]);
    const float n3 = s[0] + s[1] + s[2] + s[3];
    for (u32 q = 0; q < 4u; q++) {
      const u32 jj = 32u * q + l;
      const float z = in[t * is + cd + h * 128u + jj];
      const float y = o[q] * (1.0f / __builtin_sqrtf(n3 / 128.0f + eps)) * nw[jj] * (z / (1.0f + __builtin_expf(-z)));
      if (out) out[t * os + h * 128u + jj] = y;
      if (oq) quant_wave(y, (h * 128u + 32u * q) >> 5, oq + t * qs, inner >> 5);
    }
  }
}
#undef GDN1P_KERNEL
#undef IE_GDN1P_TCAP
#endif /* !IE_MEGA */

/* The state after the accepted tokens of a verify run: the n records (tokens
 * 1 .. n) of ie_gdn1 / ie_gdn applied to the state slot of each linear layer
 * (which holds the state after token 0). One workgroup per (layer, value
 * head), the thread mapping of ie_gdn1; the update is its update, token by
 * token: st = st * decay, then st += k * delta as one fused multiply-add (as
 * the compiler fuses it there, where st * decay has another user). S: layer l
 * at S + l * sls floats, rec: layer l at rec + l * rls floats. */
KERNEL ie_gdn_replay(G float *S, const G float *rec, u32 nv, u32 mask, unsigned long sls, unsigned long rls) {
  const u32 l = wgid() / nv, h = wgid() % nv, t0 = tid(), j = t0 & 127u, half = t0 >> 7;
  G float *Sp = S + l * sls + (unsigned long)h * 16384u + half * 64u * 128u + j;
  const G float *rp = rec + l * rls;
  float st[64];
  for (u32 ii = 0; ii < 64u; ii++) st[ii] = Sp[ii * 128u];
  for (u32 mk_ = mask; mk_; mk_ &= mk_ - 1u) { /* the nodes t of the bits, ascending: record t - 1 (a chain: 1 .. k) */
    const u32 t = (u32)__builtin_ctz(mk_) - 1u;
    const G float *r = rp + ((unsigned long)t * nv + h) * GDN_RECF;
    const float decay = r[0], delta = r[GDN_RECD + j];
    const G float *k = r + GDN_RECK + half * 64u;
    for (u32 ii = 0; ii < 64u; ii++) {
      st[ii] = st[ii] * decay;
      st[ii] = __builtin_fmaf(k[ii], delta, st[ii]);
    }
  }
  for (u32 ii = 0; ii < 64u; ii++) Sp[ii * 128u] = st[ii];
}

/* (ie_gdn) Workgroup (h, cq): value head h, value columns 32 cq .. 32 cq + 31
 * (4 workgroups per head: more workgroups than heads keeps the GPU busy).
 * Wave w holds the state rows (key dims) 16 w .. 16 w + 15 of these columns:
 * lane l, column j = 32 cq + l, 16 state values in registers. The conv, L2
 * norm and gates are computed by every workgroup of the head (cheap); the
 * output o (before the gated norm) goes to out, and ie_gnorm then does the
 * gated RMSNorm, which needs the whole head. */
#define GD_SPLIT 4u
static LDS float gd_p8[8][32], gd_v32[32];
KERNEL ie_gdn(const G float *in, G float *out, G float *ring, G float *S, const G float *cw, const G float *dtb,
              const G float *sa, u32 pos, u32 cd, u32 nk, u32 nv, float eps, u32 T, u32 is, u32 os, u32 c, u32 wfrom,
              u32 rw, u32 ns, G float *rec) {
  const u32 h = wgid() / GD_SPLIT, cq = wgid() % GD_SPLIT, t0 = tid(), w = t0 >> 5, l = lane(), kh = h % nk;
  const u32 j128 = t0 & 127u, half = t0 >> 7, inner = nv * 128u;
  const unsigned long slot = (unsigned long)nv * 16384u, own = (unsigned long)h * 16384u + (16u * w) * 128u + 32u * cq + l;
  /* the state loads first: they do not depend on the conv, so their
   * memory latency overlaps it */
  float st[16];
  for (u32 ii = 0; ii < 16u; ii++) st[ii] = pos ? S[c * slot + own + ii * 128u] : 0.0f;
  for (u32 t = 0; t < T; t++) {
    const G float *it = in + t * is;
    /* conv + SiLU: threads < 128 do q (channel kh*128 + j128), threads >=
     * 128 do k; threads < 32 also v (this workgroup's 32 columns). Ring
     * writers: q and k by workgroup (kh, 0), v by its workgroup. */
    const float a = conv_ch(in, is, ring, cw, half ? nk * 128u + kh * 128u + j128 : kh * 128u + j128, t, pos, cd,
                            rw && h == kh && cq == 0u);
    const float vv = t0 < 32u ? conv_ch(in, is, ring, cw, 2u * nk * 128u + h * 128u + 32u * cq + t0, t, pos, cd, rw) : 0.0f;
    /* L2 norm of q (waves 0-3) and k (waves 4-7): x / sqrt(sum x^2 + eps) */
    const float ss = wave_sum_all(a * a);
    if (lane() == 0u) gd_r[w] = ss;
    barrier();
    const float n2 = half ? gd_r[4] + gd_r[5] + gd_r[6] + gd_r[7] : gd_r[0] + gd_r[1] + gd_r[2] + gd_r[3];
    const float an = a * (1.0f / __builtin_sqrtf(n2 + eps));
    G float *rt = rec && t ? rec + ((unsigned long)(t - 1u) * nv + h) * GDN_RECF : (G float *)0; /* this token's record (k and decay: by workgroup cq 0) */
    if (half) {
      gd_k[j128] = an;
      if (rt && cq == 0u) rt[GDN_RECK + j128] = an;
    } else gd_q[j128] = an;
    if (t0 < 32u) gd_v32[t0] = vv;
    const float beta = sigmoidf(it[cd + inner + h]);
    const float al = it[cd + inner + nv + h] + dtb[h];
    const float sp = al > 20.0f ? al : __builtin_logf(1.0f + __builtin_expf(al));
    const float decay = __builtin_expf(sp * sa[h]);
    if (rt && cq == 0u && t0 == 0u) rt[0] = decay;
    barrier();
    float pk = 0.0f;
    for (u32 ii = 0; ii < 16u; ii++) {
      st[ii] = st[ii] * decay;
      pk += st[ii] * gd_k[16u * w + ii];
    }
    gd_p8[w][l] = pk;
    barrier();
    float kv = 0.0f;
    for (u32 x = 0; x < 8u; x++) kv += gd_p8[x][l];
    const float delta = (gd_v32[l] - kv) * beta;
    if (rt && w == 0u) rt[GDN_RECD + 32u * cq + l] = delta;
    barrier();
    float po = 0.0f;
    G float *Sw = S + ((c + t) % ns) * slot + own;
    const int wr = t >= wfrom && (!rec || t == 0u); /* uniform; with records only token 0's state is kept */
    for (u32 ii = 0; ii < 16u; ii++) {
      st[ii] += gd_k[16u * w + ii] * delta;
      if (wr) Sw[ii * 128u] = st[ii];
      po += st[ii] * gd_q[16u * w + ii];
    }
    gd_p8[w][l] = po;
    barrier();
    if (w == 0u) {
      float o = 0.0f;
      for (u32 x = 0; x < 8u; x++) o += gd_p8[x][l];
      out[t * os + h * 128u + 32u * cq + l] = o * (1.0f / __builtin_sqrtf(128.0f));
    }
    barrier(); /* the LDS of this token is read before the next token writes it */
  }
}

/* The gated RMSNorm of the linear attention output, in place: per value
 * head (128 values) y = rmsnorm(o) * nw * silu(z), z from the input row (in
 * + cd); and the Q8 copy. Wave w of workgroup (g, token y) does head 8 g +
 * w; lane l holds the values l, l + 32, l + 64, l + 96 (one per Q8 block). */
KERNEL ie_gnorm(G float *out, const G float *in, const G float *nw, G u8 *oq, u32 nv, u32 cd, float eps, u32 is, u32 os,
                u32 qs) {
#ifndef IE_MEGA
  const int skip_f32 = (nv >> 31) != 0u;
  nv &= 0x7FFFFFFFu;
#endif
  out = TOK(out, os), in = TOKC(in, is), oq = TOK(oq, qs);
  const u32 h = wgid() * 8u + (tid() >> 5), l = lane();
  if (h >= nv) return; /* whole waves */
  float o[4], s2 = 0.0f;
  for (u32 q = 0; q < 4u; q++) o[q] = out[h * 128u + 32u * q + l], s2 += o[q] * o[q];
  s2 = wave_sum_all(s2);
  const float r = 1.0f / __builtin_sqrtf(s2 / 128.0f + eps);
  for (u32 q = 0; q < 4u; q++) {
    const u32 jj = 32u * q + l;
    const float z = in[cd + h * 128u + jj];
    const float y = o[q] * r * nw[jj] * (z / (1.0f + __builtin_expf(-z)));
#ifdef IE_MEGA
    out[h * 128u + jj] = y;
#else
    if (!skip_f32 || !oq) out[h * 128u + jj] = y;
#endif
    if (oq) quant_wave(y, (h * 128u + 32u * q) >> 5, oq, (nv * 128u) >> 5);
  }
}

/* ---- Linear attention of a prompt chunk, the chunked (parallel) form ----
 * For a block of GC tokens with start state S0 (S[i][j], i key, j value),
 * gamma_t = the product of the decays g of tokens 1 .. t of the block:
 *   delta_t = beta_t (v_t - gamma_t S0^T k_t - sum_{s<t} (gamma_t/gamma_s) (k_t.k_s) delta_s)
 * so Delta = U - W S0 with U = Tm diag(beta) V, W = Tm diag(beta gamma) K,
 * Tm = (I + L)^-1, L[t][s] = beta_t (gamma_t/gamma_s) k_t.k_s (s < t): U and
 * W do not depend on S0. Then
 *   O = diag(gamma) Q S0 + M Delta,  M[t][s] = (gamma_t/gamma_s) q_t.k_s (s <= t)
 *   S_new = gamma_n S0 + K^T diag(gamma_n / gamma) Delta.
 * (tools/gdn_chunk.py checks the identity against the token recurrence.)
 * Scratch (global, per run): qk[t][nk][256] (q (times 1/sqrt(128)) | k, L2
 * normed), vv[t][nv][128], bg[t][nv][2] (beta, log g), U, W [t][nv][128],
 * Mb[block][nv][GC][GC], Gc[t][nv] (cumulative log g in the block). */
#define GC 32u
typedef float f4 __attribute__((ext_vector_type(4)));
/* 1: conv + SiLU of all channels, L2 norms, beta and log g; workgroup y = token y */
KERNEL ie_gdn_prep(const G float *in, const G float *ring, const G float *cw, const G float *dtb, const G float *sa,
                   u32 pos, u32 cd, u32 nk, u32 nv, float eps, u32 is, G float *qk, G float *vv, G float *bg) {
  const u32 t = tokid(), inner = nv * 128u, nq = nk * 128u;
  const G float *it = in + t * is;
  for (u32 c = tid(); c < cd; c += NT) {
    float acc = 0.0f;
    for (u32 j = 0; j < 4u; j++) acc += conv_in(in, is, ring, c, t, 3u - j, pos, cd) * cw[c * 4u + j];
    const float y = acc / (1.0f + __builtin_expf(-acc));
    if (c < 2u * nq) {
      const u32 isk = c >= nq, cc = isk ? c - nq : c;
      qk[((unsigned long)t * nk + cc / 128u) * 256u + isk * 128u + cc % 128u] = y;
    } else {
      vv[(unsigned long)t * inner + (c - 2u * nq)] = y;
    }
  }
  for (u32 h = tid(); h < nv; h += NT) {
    const float al = it[cd + inner + nv + h] + dtb[h];
    const float sp = al > 20.0f ? al : __builtin_logf(1.0f + __builtin_expf(al));
    bg[((unsigned long)t * nv + h) * 2u] = 1.0f / (1.0f + __builtin_expf(-it[cd + inner + h]));
    bg[((unsigned long)t * nv + h) * 2u + 1u] = sp * sa[h];
  }
  barrier(); /* the conv outputs of this workgroup are visible to it */
  const u32 w = tid() >> 5, l = lane();
  for (u32 hh = w; hh < 2u * nk; hh += 8u) { /* q of key head hh / 2 (even) or its k (odd) */
    G float *x = qk + ((unsigned long)t * nk + hh / 2u) * 256u + (hh & 1u) * 128u;
    float v4[4], ss = 0.0f;
    for (u32 q = 0; q < 4u; q++) v4[q] = x[l + 32u * q], ss += v4[q] * v4[q];
    ss = wave_sum_all(ss);
    const float r = (1.0f / __builtin_sqrtf(ss + eps)) * ((hh & 1u) ? 1.0f : 1.0f / __builtin_sqrtf(128.0f));
    for (u32 q = 0; q < 4u; q++) x[l + 32u * q] = v4[q] * r;
  }
}

/* 2: U, W and M of block b for value head h (workgroup h + nv b) */
static LDS float wy_k[GC][132] __attribute__((aligned(16))), wy_q[GC][132] __attribute__((aligned(16))), wy_L[GC][GC] __attribute__((aligned(16))), wy_G[GC], wy_b[GC], wy_lg[GC];
/* rows of 132: 16-byte reads of 8 lanes (8 rows) fall on different banks */
KERNEL ie_gdn_wy(const G float *qk, const G float *vv, const G float *bg, G float *U, G float *W, G float *Mb, G float *Gc,
                 u32 T, u32 nk, u32 nv) {
  const u32 h = wgid() % nv, b = wgid() / nv, t0 = b * GC, n = T - t0 < GC ? T - t0 : GC, kh = h % nk, th = tid();
  {
    f4 q4[4], k4[4];
#pragma unroll
    for (u32 k = 0; k < 4u; k++) { /* float4 e: row e / 32, dims 4 (e % 32) .. + 3 */
      const u32 e = th + NT * k, r = e >> 5;
      const G float *src = qk + ((unsigned long)(t0 + r) * nk + kh) * 256u + 4u * (e & 31u);
      q4[k] = r < n ? *(const G f4 *)src : (f4){0.0f, 0.0f, 0.0f, 0.0f};
      k4[k] = r < n ? *(const G f4 *)(src + 128u) : (f4){0.0f, 0.0f, 0.0f, 0.0f};
    }
#pragma unroll
    for (u32 k = 0; k < 4u; k++) {
      const u32 e = th + NT * k;
      *(LDSP f4 *)&wy_q[e >> 5][4u * (e & 31u)] = q4[k];
      *(LDSP f4 *)&wy_k[e >> 5][4u * (e & 31u)] = k4[k];
    }
  }
  if (th < GC) {
    wy_b[th] = th < n ? bg[((unsigned long)(t0 + th) * nv + h) * 2u] : 0.0f;
    wy_lg[th] = th < n ? bg[((unsigned long)(t0 + th) * nv + h) * 2u + 1u] : 0.0f;
  }
  lbarrier();
  if (th < GC) { /* cumulative log decay in the block */
    float G0 = 0.0f;
    for (u32 r = 0; r <= th; r++) G0 += wy_lg[r];
    wy_G[th] = G0;
    if (th < n) Gc[(unsigned long)(t0 + th) * nv + h] = G0;
  }
  lbarrier();
  /* L (strictly lower) and M (lower with the diagonal): thread e does pairs e, e + 256, .. */
  for (u32 e = th; e < GC * GC; e += NT) {
    const u32 r = e / GC, c = e % GC;
    float kk = 0.0f, qq = 0.0f;
    if (r < n && c <= r) {
      float kk2 = 0.0f, qq2 = 0.0f;
#pragma unroll 4
      for (u32 i = 0; i < 128u; i += 8u) {
        const f4 a0 = *(const LDSP f4 *)&wy_k[r][i], a1 = *(const LDSP f4 *)&wy_k[r][i + 4u];
        const f4 q0 = *(const LDSP f4 *)&wy_q[r][i], q1 = *(const LDSP f4 *)&wy_q[r][i + 4u];
        const f4 c0 = *(const LDSP f4 *)&wy_k[c][i], c1 = *(const LDSP f4 *)&wy_k[c][i + 4u];
        kk += a0.x * c0.x + a0.y * c0.y + a0.z * c0.z + a0.w * c0.w;
        kk2 += a1.x * c1.x + a1.y * c1.y + a1.z * c1.z + a1.w * c1.w;
        qq += q0.x * c0.x + q0.y * c0.y + q0.z * c0.z + q0.w * c0.w;
        qq2 += q1.x * c1.x + q1.y * c1.y + q1.z * c1.z + q1.w * c1.w;
      }
      kk += kk2, qq += qq2;
    }
    const float dec = r < n && c <= r ? __builtin_expf(wy_G[r] - wy_G[c]) : 0.0f;
    wy_L[r][c] = c < r ? wy_b[r] * dec * kk : 0.0f;
    Mb[(((unsigned long)b * nv + h) * GC + r) * GC + c] = dec * qq;
  }
  lbarrier();
  /* forward substitution, one column per thread: column th of [beta V | beta gamma K] */
  float x[GC];
  const int isw = th >= 128u;
  const u32 j = th & 127u;
#pragma unroll
  for (u32 r = 0; r < GC; r++) {
    x[r] = 0.0f;
    if (r < n) { /* uniform */
      float acc = isw ? wy_b[r] * __builtin_expf(wy_G[r]) * wy_k[r][j] : wy_b[r] * vv[((unsigned long)(t0 + r) * nv + h) * 128u + j];
#pragma unroll
      for (u32 c = 0; c < r; c++) acc -= wy_L[r][c] * x[c];
      x[r] = acc;
      (isw ? W : U)[((unsigned long)(t0 + r) * nv + h) * 128u + j] = acc;
    }
  }
}

/* 3: the blocks in order, for value head h and its value columns 32 cq ..
 * 32 cq + 31 (workgroup 4 h + cq): the columns of the state do not depend on
 * each other, so 4 workgroups share a head. Wave w owns the 4 columns
 * 32 cq + 4 w + a (a < 4); lane l holds the state of the key dims 4 l + k
 * (k < 4): st[4 k + a] = S[4 l + k][32 cq + 4 w + a]. A product X S0 (X:
 * 32 rows of 128) is then a sum over the 32 lanes, done as a reduce-scatter
 * with DPP: no exchange between waves, and the LDS reads of X rows are
 * 16 bytes per lane, consecutive (no broadcast). Writes the output o (before
 * the gated norm) of every token and the state after the last token to
 * slot (c + T - 1) % ns. */
static LDS float sq_w[GC][128] __attribute__((aligned(16))), sq_q[GC][128] __attribute__((aligned(16))),
    sq_k[GC][128] __attribute__((aligned(16))), sq_d[8][GC][4] __attribute__((aligned(16))), sq_m[GC][GC + 1u], sq_G[GC];
static inline int xor_i(int v, u32 m) { /* the value of lane (this lane ^ m) */
  switch (m) { /* DPP row_xmask:m = 0x160 + m */
    case 16: return xrow_i(v);
    case 8: return __builtin_amdgcn_update_dpp(0, v, 0x168, 0xF, 0xF, true);
    case 4: return __builtin_amdgcn_update_dpp(0, v, 0x164, 0xF, 0xF, true);
    case 2: return __builtin_amdgcn_update_dpp(0, v, 0x162, 0xF, 0xF, true);
    default: return __builtin_amdgcn_update_dpp(0, v, 0x161, 0xF, 0xF, true);
  }
}
/* p[a] = sum over the 128 key dims of x[r][i] S[i][col a] for the row r =
 * rho(l) = 16 (l & 1) + (l >> 1) of this lane (a permutation of the rows):
 * per lane, partial sums over its 4 key dims, for 16 rows at a time (fewer
 * registers); 4 halving steps (step m: the lanes with bit m keep the upper
 * half of the rows and send the lower half to lane ^ m) leave row l >> 1 in
 * the lanes l and l ^ 1, which then add their values. */
static inline void sq_prod(const LDSP float (*x)[128], const float *st, u32 l, float *p) {
#pragma unroll
  for (u32 hf = 0; hf < 2u; hf++) {
    float acc[64];
#pragma unroll
    for (u32 r = 0; r < 16u; r++) {
      const f4 v = *(const LDSP f4 *)&x[16u * hf + r][4u * l];
#pragma unroll
      for (u32 a = 0; a < 4u; a++) acc[4u * r + a] = v.x * st[a] + v.y * st[4u + a] + v.z * st[8u + a] + v.w * st[12u + a];
    }
#pragma unroll
    for (u32 m = 16u, R = 16u; m >= 2u; m >>= 1, R >>= 1) {
      const int up = (l & m) != 0u;
#pragma unroll
      for (u32 r = 0; r < R / 2u; r++)
#pragma unroll
        for (u32 a = 0; a < 4u; a++) {
          const float lo = acc[4u * r + a], hi = acc[4u * (r + R / 2u) + a];
          acc[4u * r + a] = (up ? hi : lo) + I2F(xor_i(F2I(up ? lo : hi), m));
        }
    }
#pragma unroll
    for (u32 a = 0; a < 4u; a++) {
      const float v = acc[a] + I2F(xor_i(F2I(acc[a]), 1u));
      if ((l & 1u) == hf) p[a] = v;
    }
  }
}
/* 32 rows of 128 floats (rows r < n; row r at src + r * rs) into registers:
 * thread th holds the float4 number th + NT k (row (th + NT k) / 32) */
static inline void sq_ld(f4 *x, const G float *src, unsigned long rs, u32 n, u32 th) {
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 e = th + NT * k, r = e >> 5;
    x[k] = r < n ? *(const G f4 *)(src + r * rs + 4u * (e & 31u)) : (f4){0.0f, 0.0f, 0.0f, 0.0f};
  }
}
static inline void sq_st(LDSP float (*dst)[128], const f4 *x, u32 th) {
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 e = th + NT * k;
    *(LDSP f4 *)&dst[e >> 5][4u * (e & 31u)] = x[k];
  }
}
KERNEL ie_gdn_seq(const G float *qk, const G float *U, const G float *W, const G float *Mb, const G float *Gc, G float *out,
                  u32 os, G float *S, u32 c, u32 ns, u32 pos, u32 T, u32 nk, u32 nv) {
  const u32 h = wgid() / GD_SPLIT, cq = wgid() % GD_SPLIT, kh = h % nk, th = tid(), w = th >> 5, l = lane();
  const u32 col = 32u * cq + 4u * w, rl = 16u * (l & 1u) + (l >> 1); /* this wave's first column; this lane's row (sq_prod) */
  const unsigned long slot = (unsigned long)nv * 16384u, own = (unsigned long)h * 16384u + (4u * l) * 128u + col;
  const unsigned long wrs = (unsigned long)nv * 128u, qrs = (unsigned long)nk * 256u;
  float st[16];
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
#pragma unroll
    for (u32 a = 0; a < 4u; a++) st[4u * k + a] = pos ? S[c * slot + own + k * 128u + a] : 0.0f;
  /* the next block's inputs are loaded into registers one block ahead */
  f4 rw[4], rq[4], rk[4];
  float ru[4], rm[4], rg;
  {
    const u32 n = T < GC ? T : GC;
    sq_ld(rw, W + (unsigned long)h * 128u, wrs, n, th);
    sq_ld(rq, qk + (unsigned long)kh * 256u, qrs, n, th);
    sq_ld(rk, qk + (unsigned long)kh * 256u + 128u, qrs, n, th);
#pragma unroll
    for (u32 a = 0; a < 4u; a++) ru[a] = rl < n ? U[(unsigned long)rl * wrs + h * 128u + col + a] : 0.0f;
#pragma unroll
    for (u32 k = 0; k < 4u; k++) rm[k] = Mb[(unsigned long)h * GC * GC + th + NT * k];
    rg = th < n ? Gc[(unsigned long)th * nv + h] : 0.0f;
  }
  for (u32 b = 0, t0 = 0; t0 < T; b++, t0 += GC) {
    const u32 n = T - t0 < GC ? T - t0 : GC, t1 = t0 + GC, n1 = t1 < T ? (T - t1 < GC ? T - t1 : GC) : 0u;
    sq_st(sq_w, rw, th);
    sq_st(sq_q, rq, th);
    sq_st(sq_k, rk, th);
#pragma unroll
    for (u32 k = 0; k < 4u; k++) {
      const u32 e = th + NT * k;
      sq_m[e >> 5][e & 31u] = rm[k];
    }
    if (th < GC) sq_G[th] = rg;
    float u[4];
#pragma unroll
    for (u32 a = 0; a < 4u; a++) u[a] = ru[a];
    lbarrier();
    if (n1) { /* uniform: prefetch the next block */
      sq_ld(rw, W + (unsigned long)t1 * wrs + h * 128u, wrs, n1, th);
      sq_ld(rq, qk + (unsigned long)t1 * qrs + kh * 256u, qrs, n1, th);
      sq_ld(rk, qk + (unsigned long)t1 * qrs + kh * 256u + 128u, qrs, n1, th);
#pragma unroll
      for (u32 a = 0; a < 4u; a++) ru[a] = rl < n1 ? U[(unsigned long)(t1 + rl) * wrs + h * 128u + col + a] : 0.0f;
#pragma unroll
      for (u32 k = 0; k < 4u; k++) rm[k] = Mb[(((unsigned long)(b + 1u) * nv + h) * GC) * GC + th + NT * k];
      rg = th < n1 ? Gc[(unsigned long)(t1 + th) * nv + h] : 0.0f;
    }
    /* Delta = U - W S0, row rl of the block (0 for rows >= n) */
    float p[4], d[4];
    sq_prod(sq_w, st, l, p);
#pragma unroll
    for (u32 a = 0; a < 4u; a++) d[a] = rl < n ? u[a] - p[a] : 0.0f;
    *(LDSP f4 *)&sq_d[w][rl][0] = (f4){d[0], d[1], d[2], d[3]}; /* this wave's rows of Delta */
    /* O = diag(gamma) Q S0 + M Delta (M is 0 above the diagonal) */
    sq_prod(sq_q, st, l, p);
    const float gl = sq_G[rl];
    float o[4];
#pragma unroll
    for (u32 a = 0; a < 4u; a++) o[a] = __builtin_expf(gl) * p[a];
#pragma unroll 8
    for (u32 s2 = 0; s2 < GC; s2++) {
      const float mm = sq_m[rl][s2];
      const f4 dv = *(const LDSP f4 *)&sq_d[w][s2][0];
      o[0] += mm * dv.x, o[1] += mm * dv.y, o[2] += mm * dv.z, o[3] += mm * dv.w;
    }
    if (rl < n) {
      G float *op = out + (t0 + rl) * os + h * 128u + col;
#pragma unroll
      for (u32 a = 0; a < 4u; a++) op[a] = o[a];
    }
    /* S_new = gamma_n S0 + K^T diag(gamma_n / gamma) Delta */
    const float Gn = sq_G[n - 1u], eg = __builtin_expf(Gn), sc = __builtin_expf(Gn - gl);
    /* the reads of sq_d above are done (one wave: LDS operations in order) */
    *(LDSP f4 *)&sq_d[w][rl][0] = (f4){d[0] * sc, d[1] * sc, d[2] * sc, d[3] * sc};
#pragma unroll
    for (u32 i = 0; i < 16u; i++) st[i] *= eg;
#pragma unroll 4
    for (u32 r = 0; r < GC; r++) {
      const f4 kv = *(const LDSP f4 *)&sq_k[r][4u * l];
      const f4 dv = *(const LDSP f4 *)&sq_d[w][r][0];
      const float kk[4] = {kv.x, kv.y, kv.z, kv.w}, dd[4] = {dv.x, dv.y, dv.z, dv.w};
#pragma unroll
      for (u32 k = 0; k < 4u; k++)
#pragma unroll
        for (u32 a = 0; a < 4u; a++) st[4u * k + a] += kk[k] * dd[a];
    }
    lbarrier(); /* the LDS of this block is read before the next block writes it */
  }
  G float *Sw = S + ((c + T - 1u) % ns) * slot + own;
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
#pragma unroll
    for (u32 a = 0; a < 4u; a++) Sw[k * 128u + a] = st[4u * k + a];
}

/* The conv inputs of the last 3 of T tokens (positions pos + T - 3 ..) into
 * the ring, after an ie_gdn launch with rw = 0 (a prompt chunk: more tokens
 * than the ring has slots, so ie_gdn cannot write the ring safely while
 * other workgroups still read it). Thread i: channel i % cd of token T - 3
 * + i / cd. */
KERNEL ie_ring_store(const G float *in, G float *ring, u32 pos, u32 T, u32 cd, u32 is) {
  const u32 i = wgid() * NT + tid(), n = T < 3u ? T : 3u;
  if (i >= n * cd) return;
  const u32 t = T - n + i / cd, c = i % cd;
  ring[((pos + t) & (GDN_RING - 1u)) * cd + c] = in[t * is + c];
}

/* Attention of a prompt chunk in one launch: workgroup (h, y) = query head h
 * of token y (position pos0 + y), all positions 0 .. pos0 + y; wave w does
 * the positions w, w + 8, .. with an online softmax, then the 8 waves merge
 * in LDS (as ie_attn_split, with one split). Output gate and Q8 copy as
 * ie_attn_split. Strides (bytes) between the tokens: qs (q), os (out), oqs
 * (oq), gs (gate). hd <= 256, hd % 32 == 0 when oq is set. */
KERNEL ie_attn_pf(const G float *q, const G kvt *kc, const G kvt *vc, G float *out, u32 pos0, u32 hd, u32 n_head,
                  u32 n_kv, G u8 *oq, const G float *gate, u32 gstride, u32 qs, u32 os, u32 oqs, u32 gs) {
  q = TOKC(q, qs), out = TOK(out, os), oq = TOK(oq, oqs), gate = TOK(gate, gs);
  const u32 h = wgid(), pos = pos0 + tokid(), kh = h / (n_head / n_kv), kvd = n_kv * hd;
  const u32 w = tid() >> 5, l = lane();
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  float qv[8], acc[8];
  for (u32 j = 0; j < 8u; j++) {
    const u32 i = l + 32u * j;
    qv[j] = i < hd ? q[h * hd + i] : 0.0f;
    acc[j] = 0.0f;
  }
  float m = -__builtin_inff(), sum = 0.0f;
  for (u32 p = w; p <= pos; p += 8u) {
    kvrow_t k = KVROWP(kc, p, kh);
    kvrow_t v = KVROWP(vc, p, kh);
    float d = 0.0f;
    for (u32 j = 0; j < 8u; j++)
      if (l + 32u * j < hd) d += qv[j] * KVAT(k, l + 32u * j);
    d = wave_sum_all(d) * scale;
    const float mn = __builtin_fmaxf(m, d), c = __builtin_expf(m - mn), e = __builtin_expf(d - mn);
    sum = sum * c + e;
    for (u32 j = 0; j < 8u; j++)
      if (l + 32u * j < hd) acc[j] = acc[j] * c + e * KVAT(v, l + 32u * j);
    m = mn;
  }
  if (l == 0u) at_m[w] = m, at_l[w] = sum;
  for (u32 j = 0; j < 8u; j++)
    if (l + 32u * j < hd) at_o[w][l + 32u * j] = acc[j];
  barrier();
  float M = at_m[0];
  for (u32 x = 1; x < 8u; x++) M = __builtin_fmaxf(M, at_m[x]);
  float L = 0.0f;
  for (u32 x = 0; x < 8u; x++) L += at_l[x] * (at_m[x] == -__builtin_inff() ? 0.0f : __builtin_expf(at_m[x] - M));
  if (tid() < hd) { /* whole waves when oq is set */
    float o = 0.0f;
    for (u32 x = 0; x < 8u; x++)
      if (at_m[x] != -__builtin_inff()) o += at_o[x][tid()] * __builtin_expf(at_m[x] - M);
    o = o / L;
    if (gate) o = o * sigmoidf(gate[h * gstride + tid()]);
#ifdef IE_MEGA
    out[h * hd + tid()] = o;
#else
    if (out) out[h * hd + tid()] = o;
#endif
    if (oq) quant_wave(o, (h * hd + tid()) >> 5, oq, (n_head * hd) >> 5);
  }
}

/* RMSNorm of many tokens (prompt chunks): workgroup y = token y does the
 * whole row (each element read once for the sum of squares, once for the
 * output), wave w the blocks w, w + 8, .. . Same arithmetic as rmsnorm_wg. */
KERNEL ie_rmsnorm_t(const G float *x, const G float *w, G float *y, u32 n, float eps, G u8 *q, u32 xs, u32 ys, u32 qs) {
  x = TOKC(x, xs), y = TOK(y, ys), q = TOK(q, qs);
  rmsnorm_wg(x, w, y, n, eps, q);
}

/* Attention of a prompt chunk, grouped: workgroup (g, y) = KV head g of
 * token y (position pos0 + y) and its GA = n_head / n_kv query heads (<= 8).
 * Each K and V row is read once for all GA heads (6 for Qwen3.8-27B), and
 * the GA dot products are independent (more work in flight per wave). Wave
 * w does the positions w, w + 8, ..; then the waves merge one head at a
 * time in LDS. As ie_attn_pf otherwise. */
#define GA_MAX 8u
static LDS float ag_m[8][GA_MAX], ag_l[8][GA_MAX];
KERNEL ie_attn_pfg(const G float *q, const G kvt *kc, const G kvt *vc, G float *out, u32 pos0, u32 hd, u32 n_head,
                   u32 n_kv, G u8 *oq, const G float *gate, u32 gstride, u32 qs, u32 os, u32 oqs, u32 gs) {
  q = TOKC(q, qs), out = TOK(out, os), oq = TOK(oq, oqs), gate = TOK(gate, gs);
  const u32 kh = wgid(), pos = pos0 + tokid(), ga = n_head / n_kv, kvd = n_kv * hd;
  const u32 w = tid() >> 5, l = lane();
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  float qv[GA_MAX][8], acc[GA_MAX][8], m[GA_MAX], sum[GA_MAX];
#pragma unroll
  for (u32 g = 0; g < GA_MAX; g++) {
    m[g] = -__builtin_inff(), sum[g] = 0.0f;
    for (u32 j = 0; j < 8u; j++) {
      const u32 i = l + 32u * j;
      qv[g][j] = g < ga && i < hd ? q[(kh * ga + g) * hd + i] : 0.0f;
      acc[g][j] = 0.0f;
    }
  }
  for (u32 p = w; p <= pos; p += 8u) {
    kvrow_t k = KVROWP(kc, p, kh);
    kvrow_t v = KVROWP(vc, p, kh);
    float kr[8], vr[8];
    for (u32 j = 0; j < 8u; j++) {
      kr[j] = l + 32u * j < hd ? KVAT(k, l + 32u * j) : 0.0f;
      vr[j] = l + 32u * j < hd ? KVAT(v, l + 32u * j) : 0.0f;
    }
#pragma unroll
    for (u32 g = 0; g < GA_MAX; g++) {
      if (g < ga) { /* uniform */
      float d = 0.0f;
      for (u32 j = 0; j < 8u; j++) d += qv[g][j] * kr[j];
      d = wave_sum_all(d) * scale;
      const float mn = __builtin_fmaxf(m[g], d), c = __builtin_expf(m[g] - mn), e = __builtin_expf(d - mn);
      sum[g] = sum[g] * c + e;
      for (u32 j = 0; j < 8u; j++) acc[g][j] = acc[g][j] * c + e * vr[j];
      m[g] = mn;
      }
    }
  }
  if (l == 0u) {
#pragma unroll
    for (u32 g = 0; g < GA_MAX; g++) ag_m[w][g] = m[g], ag_l[w][g] = sum[g];
  }
#pragma unroll
  for (u32 g = 0; g < GA_MAX; g++) {
    if (g >= ga) continue; /* uniform */
    for (u32 j = 0; j < 8u; j++)
      if (l + 32u * j < hd) at_o[w][l + 32u * j] = acc[g][j];
    barrier();
    float M = ag_m[0][g];
    for (u32 x = 1; x < 8u; x++) M = __builtin_fmaxf(M, ag_m[x][g]);
    float L = 0.0f;
    for (u32 x = 0; x < 8u; x++) L += ag_m[x][g] == -__builtin_inff() ? 0.0f : ag_l[x][g] * __builtin_expf(ag_m[x][g] - M);
    if (tid() < hd) {
      float o = 0.0f;
      for (u32 x = 0; x < 8u; x++)
        if (ag_m[x][g] != -__builtin_inff()) o += at_o[x][tid()] * __builtin_expf(ag_m[x][g] - M);
      o = o / L;
      const u32 h = kh * ga + g;
      if (gate) o = o * sigmoidf(gate[h * gstride + tid()]);
#ifdef IE_MEGA
      out[h * hd + tid()] = o;
#else
      if (out) out[h * hd + tid()] = o;
#endif
      if (oq) quant_wave(o, (h * hd + tid()) >> 5, oq, (n_head * hd) >> 5);
    }
    barrier(); /* at_o is read before the next head writes it */
  }
}

/* Attention of a prompt chunk with fp16 WMMA (flash attention): workgroup
 * (kh, part, tile) = KV head kh, output columns FA_DC part .. + FA_DC - 1
 * of each head, and the 16 tokens 16 tile .. + 15 of the chunk (positions
 * pos0 + t). Wave w < GA = n_head / n_kv (<= 8) does query head kh * GA + w
 * for those 16 tokens; all waves load. Per block of FA_NB = 32 key
 * positions: K (fp16, key rows, all of hd) and the part's columns of V
 * (fp16, transposed: one row per column) go to LDS once for the GA heads and
 * 16 tokens; then
 *   S = (Q * scale) K^T     16 x 32, two WMMA tiles, K-dim = hd (fp16 x fp16 -> f32)
 *   online softmax per token row in f32 (causal mask: key > token position -> -inf)
 *   O = O * c + P V         16 x FA_DC, P as fp16 through LDS (D layout -> A layout)
 * The output is split in parts so that a wave holds Q (64 registers) and
 * its part of O (64): S is computed once per part. WMMA lane layout (gfx12,
 * tools/wmma_test.c): A lane l = row l % 16, K 8 (l / 16) .. + 7; B (given
 * as N x K) lane l = column l % 16, the same K; D lane l, element v = row
 * 8 (l / 16) + v, column l % 16. The row sums stay per lane until the end
 * (c is the same in every lane of a row). Not bitwise equal to ie_attn_pfg:
 * Q, K, V and P are rounded to fp16 (relative error <= 2^-11 each), the sums
 * are f32 in another order. hd % 32 == 0, hd <= 256. Output and its fused Q8
 * copy as ie_attn_pfg (the Q8 copy by the workgroup of the last part). */
#define FA_NB 32u
#define FA_DC 128u
#ifndef FA_KPAD
#define FA_KPAD 8u
#endif
#ifndef FA_PPAD
#define FA_PPAD 8u
#endif
#ifndef FA2_VPAD
#define FA2_VPAD 8u
#endif
#define FA_KS (256u + FA_KPAD) /* LDS row stride of K, halves */
#define FA_VS (FA_NB + FA_PPAD) /* LDS row stride of V^T and of P, halves */
static LDS f16 fa_k[FA_NB * FA_KS];
static LDS f16 fa_vt[FA_DC * FA_VS];
static LDS f16 fa_p[8u][16u * FA_VS];
static inline __attribute__((always_inline)) void attn_fa_body(const G float *q, const G kvt *kc, const G kvt *vc, G float *out,
                                                              u32 pos0, u32 hd, u32 n_head, u32 n_kv, const G float *gate,
                                                              u32 gstride, u32 qs, u32 os, u32 gs, u32 T, const int sw, u32 win,
                                                              u32 len, const G float *sinks) {
  const u32 ga = n_head / n_kv, ngroups = (ga + 7u) / 8u; /* 8 waves: heads 8 gsub .. 8 gsub + 7 */
  const u32 ntile = (T + 15u) / 16u, nparts = (hd + FA_DC - 1u) / FA_DC;
  const u32 tile = wgid() % ntile, part = (wgid() / ntile) % nparts;
  const u32 gsub = (wgid() / (ntile * nparts)) % ngroups, kh = wgid() / (ntile * nparts * ngroups), t0 = 16u * tile;
  const u32 kvd = n_kv * hd, w = tid() >> 5, l = lane(), hl = l >> 4, rl = l & 15u;
  const u32 nks = hd / 16u, h = kh * ga + 8u * gsub + w, c0 = part * FA_DC, nc = hd - c0 < FA_DC ? hd - c0 : FA_DC;
  const int on = 8u * gsub + w < ga; /* wave-uniform */
  const u32 tlast = t0 + 15u < T ? t0 + 15u : T - 1u, kend = pos0 + tlast + 1u; /* keys 0 .. kend - 1 */
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  /* Q (scaled, fp16) in the A layout: token t0 + rl, elements 16 ks + 8 hl .. + 7 */
  h8 qa[16];
  {
    const u32 tq = t0 + rl < T ? t0 + rl : T - 1u;
    const G float *qr = (const G float *)((const G u8 *)q + (unsigned long)tq * qs) + h * hd;
#pragma unroll
    for (u32 ks = 0; ks < 16u; ks++)
      for (u32 e = 0; e < 8u; e++) qa[ks][e] = on && ks < nks ? (f16)(qr[16u * ks + 8u * hl + e] * scale) : (f16)0.0f;
  }
  v8f o[FA_DC / 16u], zero = (v8f)(0.0f);
  float m[8], ls[8];
#pragma unroll
  for (u32 nt = 0; nt < FA_DC / 16u; nt++) o[nt] = zero;
  for (u32 v = 0; v < 8u; v++) m[v] = -__builtin_inff(), ls[v] = 0.0f;
  const u32 h4 = hd >> 2, n4 = FA_NB * h4, c4 = nc >> 2; /* float4 per key row, per block; of the part */
  /* sw (flint): a window of win keys (win 0: all), key p in row p % len, sinks */
  const u32 kbeg = sw && win && pos0 + t0 + 1u > win ? pos0 + t0 + 1u - win : 0u;
  for (u32 kb = kbeg; kb < kend; kb += FA_NB) {
    /* K (all of hd) and V (the part's columns) of keys kb .. kb + 31 to LDS (fp16); keys >= kend as zeros */
    for (u32 i4 = tid(); i4 < n4; i4 += NT) {
      const u32 r = i4 / h4, e4 = i4 - r * h4, p = kb + r;
      f32x4 kv = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      if (p < kend) kv = KVLD4(kc, sw ? p % len : p, kh, e4);
      LDSP f16 *kd = fa_k + r * FA_KS + 4u * e4;
      kd[0] = (f16)kv.x, kd[1] = (f16)kv.y, kd[2] = (f16)kv.z, kd[3] = (f16)kv.w;
    }
    for (u32 i4 = tid(); i4 < FA_NB * c4; i4 += NT) {
      const u32 r = i4 / c4, e4 = i4 - r * c4, p = kb + r;
      f32x4 vv = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      if (p < kend) vv = KVLD4O(vc, sw ? p % len : p, kh, c0, e4);
      fa_vt[(4u * e4 + 0u) * FA_VS + r] = (f16)vv.x;
      fa_vt[(4u * e4 + 1u) * FA_VS + r] = (f16)vv.y;
      fa_vt[(4u * e4 + 2u) * FA_VS + r] = (f16)vv.z;
      fa_vt[(4u * e4 + 3u) * FA_VS + r] = (f16)vv.w;
    }
    barrier();
    if (on) {
      /* S = Q K^T: two 16 x 16 tiles (keys kb + 16 j + rl) */
      v8f s[2];
#pragma unroll
      for (u32 j = 0; j < 2u; j++) {
        s[j] = zero;
#pragma unroll
        for (u32 ks = 0; ks < 16u; ks++)
          if (ks < nks) {
            const h8 kb8 = *(const LDSP h8 *)(fa_k + (16u * j + rl) * FA_KS + 16u * ks + 8u * hl);
            s[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qa[ks], kb8, s[j]);
          }
      }
      /* online softmax of rows 8 hl + v (tokens t0 + 8 hl + v) */
      float c[8];
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const u32 tp = pos0 + t0 + 8u * hl + v; /* this row's position */
        const u32 k0 = kb + rl, k1 = kb + 16u + rl;
        const int v0 = k0 <= tp && (!sw || !win || k0 + win > tp), v1 = k1 <= tp && (!sw || !win || k1 + win > tp);
        const float s0 = v0 ? s[0][v] : -__builtin_inff(), s1 = v1 ? s[1][v] : -__builtin_inff();
        float rm = __builtin_fmaxf(s0, s1);
        for (u32 x = 1u; x < 16u; x <<= 1) rm = __builtin_fmaxf(rm, shfl_xor_f(rm, x)); /* the 16 lanes of the row */
        const float mn = __builtin_fmaxf(m[v], rm);
        /* sw: a row may have no key yet (its window starts later): then 0, not exp(-inf + inf) */
        const float mz = sw && mn == -__builtin_inff() ? 0.0f : mn;
        c[v] = __builtin_expf(m[v] - mz);
        const float p0 = __builtin_expf(s0 - mz), p1 = __builtin_expf(s1 - mz);
        ls[v] = ls[v] * c[v] + p0 + p1;
        m[v] = mn;
        LDSP f16 *pr = fa_p[w] + (8u * hl + v) * FA_VS;
        pr[rl] = (f16)p0;
        pr[16u + rl] = (f16)p1;
      }
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory"); /* this wave's P stores before its loads */
      h8 pa[2];
      for (u32 kk = 0; kk < 2u; kk++) pa[kk] = *(const LDSP h8 *)(fa_p[w] + rl * FA_VS + 16u * kk + 8u * hl);
#pragma unroll
      for (u32 nt = 0; nt < FA_DC / 16u; nt++) {
        if (16u * nt >= nc) continue;
        for (u32 v = 0; v < 8u; v++) o[nt][v] *= c[v];
#pragma unroll
        for (u32 kk = 0; kk < 2u; kk++) {
          const h8 vb = *(const LDSP h8 *)(fa_vt + (16u * nt + rl) * FA_VS + 16u * kk + 8u * hl);
          o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pa[kk], vb, o[nt]);
        }
      }
    }
    barrier();
  }
  if (on) {
    float inv[8];
    for (u32 v = 0; v < 8u; v++) {
      float x = ls[v];
      for (u32 s = 1u; s < 16u; s <<= 1) x += shfl_xor_f(x, s);
      if (sw && sinks) x += __builtin_expf(sinks[h] - m[v]); /* the sink: in the denominator only */
      inv[v] = 1.0f / x;
    }
#pragma unroll
    for (u32 nt = 0; nt < FA_DC / 16u; nt++) {
      if (16u * nt >= nc) continue;
      for (u32 v = 0; v < 8u; v++) {
        const u32 tk = t0 + 8u * hl + v, d = c0 + 16u * nt + rl;
        if (tk >= T) continue;
        float x = o[nt][v] * inv[v];
        if (gate) x = x * sigmoidf(((const G float *)((const G u8 *)gate + (unsigned long)tk * gs))[h * gstride + d]);
        ((G float *)((G u8 *)out + (unsigned long)tk * os))[h * hd + d] = x;
      }
    }
  }
}
KERNEL ie_attn_fa(const G float *q, const G kvt *kc, const G kvt *vc, G float *out, u32 pos0, u32 hd, u32 n_head,
                  u32 n_kv, const G float *gate, u32 gstride, u32 qs, u32 os, u32 gs, u32 T) {
  attn_fa_body(q, kc, vc, out, pos0, hd, n_head, n_kv, gate, gstride, qs, os, gs, T, 0, 0u, 1u, 0);
}
/* ie_attn_fa for flint prompt chunks: keys in a window of win (0: all), key
 * p in row p % len of kc, vc (a ring), sinks[h] in the denominator (or NULL). */
KERNEL ie_attn_fa_sw(const G float *q, const G kvt *kc, const G kvt *vc, G float *out, u32 pos0, u32 hd, u32 n_head,
                     u32 n_kv, u32 qs, u32 os, u32 T, u32 win, u32 len, const G float *sinks) {
  attn_fa_body(q, kc, vc, out, pos0, hd, n_head, n_kv, 0, 0u, qs, os, 0u, T, 1, win, len, sinks);
}

/* ie_attn_fa with S computed once (hd == 256): workgroup (kh, tile) = KV
 * head kh and the 16 tokens 16 tile .. + 15, all 256 output columns. Per
 * block of FA_NB keys: K and all of V^T to LDS; phase A: wave w < GA
 * computes S = Q K^T of head kh * GA + w (Q in its registers), the online
 * softmax of its rows, and writes P (fp16) and the rescale c of each row to
 * LDS; phase B: the GA * 16 output tiles (head, 16 columns) are spread
 * over all 8 waves, in registers: FA2_OA per wave w < GA (it also holds Q),
 * the rest on the other 8 - GA waves (<= FA2_OB each). Each does O = O * c
 * + P V for its tiles. At the end wave w < GA writes 1 / l of its rows to
 * LDS for the owners of the tiles. The two kinds of waves run separate
 * copies of the loop (fa2_run, the same barriers), so that their registers
 * are allocated apart. Per block, all K and V loads are issued before the
 * LDS stores (one wait for the memory, not one per load). The same arithmetic
 * per output element as ie_attn_fa
 * (S, P, the sums in the same order), so the same result. Needs GA <= 6. */
/* 4 KV elements as fp16 (the fp16 cache: as stored) */
#ifdef IE_KV_QUANT
#define KVH4(c, p, kh, e4) __builtin_convertvector(KVLD4(c, p, kh, e4), hk4)
#else
#define KVH4(c, p, kh, e4) kv_h4(c, ((unsigned long)(p) * kvd + (kh) * hd) / 4u + (e4))
#endif
#ifndef IE_KV_QUANT
static inline hk4 kv_h4(const G kvt *p, unsigned long i4) {
#ifdef IE_KV_F32
  return __builtin_convertvector(((const G kv4 *)p)[i4], hk4);
#else
  return ((const G kv4 *)p)[i4];
#endif
}
#endif
#define FA2_OA 9u
#define FA2_OB 21u
#define FA2_VS (FA_NB + FA2_VPAD)
struct fa2_ep_t {
  G float *out;
  const G float *gate;
  u32 gstride, os, gs, T, ntile;
};
static LDS f16 fa2_vt[256u * FA2_VS];
static LDS float fa2_c[8u][16u];
static LDS struct fa2_ep_t fa2_ep;
static inline __attribute__((always_inline)) void fa2_run(const G float *q, const G kvt *kc, const G kvt *vc,
                                                         u32 pos0, u32 n_head, u32 n_kv, u32 qs, u32 T,
                                                         const int dos, const u32 OT, u32 tb, u32 tn, v8f *o) {
  const u32 ntile = (T + 15u) / 16u, tile = wgid() % ntile, kh = wgid() / ntile, t0 = 16u * tile;
  const u32 hd = 256u, ga = n_head / n_kv, kvd = n_kv * hd, w = __builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 tlast = t0 + 15u < T ? t0 + 15u : T - 1u, kend = pos0 + tlast + 1u; /* keys 0 .. kend - 1 */
  h8 qa[16];
  if (dos) {
    const u32 l = lane(), hl = l >> 4, rl = l & 15u;
    const float scale = 1.0f / 16.0f; /* 1 / sqrt(256) */
    const u32 tq = t0 + rl < T ? t0 + rl : T - 1u;
    const G float *qr = (const G float *)((const G u8 *)q + (unsigned long)tq * qs) + (kh * ga + w) * hd;
#pragma unroll
    for (u32 ks = 0; ks < 16u; ks++)
#pragma unroll
      for (u32 e = 0; e < 8u; e++) qa[ks][e] = (f16)(qr[16u * ks + 8u * hl + e] * scale);
  }
  const v8f zero = (v8f)(0.0f);
  float m[8], ls[8];
#pragma unroll
  for (u32 t = 0; t < OT; t++) o[t] = zero;
  if (dos) {
#pragma unroll
    for (u32 v = 0; v < 8u; v++) m[v] = -__builtin_inff(), ls[v] = 0.0f;
  }
  for (u32 kb = 0; kb < kend; kb += FA_NB) {
    /* opaque copies of the thread/lane numbers: the LDS addresses are computed in the loop, not kept in registers */
    u32 to = tid(), lo = lane();
    __asm__ volatile("" : "+v"(to), "+v"(lo));
    const u32 hlo = lo >> 4, rlo = lo & 15u;
    { /* K loads -> LDS first, then V loads -> transposed LDS (keeps peak VGPR <= 240 under IE_KV_Q8) */
      hk4 xk[FA_NB * 64u / NT];
#pragma unroll
      for (u32 n = 0; n < FA_NB * 64u / NT; n++) {
        const u32 i4 = to + NT * n, r = i4 >> 6, e4 = i4 & 63u, p = kb + r;
        xk[n] = (hk4)(0.0f);
        if (p < kend) xk[n] = KVH4(kc, p, kh, e4);
      }
#pragma unroll
      for (u32 n = 0; n < FA_NB * 64u / NT; n++) {
        const u32 i4 = to + NT * n, r = i4 >> 6, e4 = i4 & 63u;
        *(LDSP hk4 *)(fa_k + r * FA_KS + 4u * e4) = xk[n];
      }
      __asm__ volatile("" ::: "memory");
      hk4 xv[(FA_NB / 4u) * 64u / NT][4];
#pragma unroll
      for (u32 n = 0; n < (FA_NB / 4u) * 64u / NT; n++) {
        const u32 j = to + NT * n, kq = j >> 6, e4 = j & 63u;
#pragma unroll
        for (u32 u = 0; u < 4u; u++) {
          const u32 p = kb + 4u * kq + u;
          xv[n][u] = (hk4)(0.0f);
          if (p < kend) xv[n][u] = KVH4(vc, p, kh, e4);
        }
      }
#pragma unroll
      for (u32 n = 0; n < (FA_NB / 4u) * 64u / NT; n++) {
        const u32 j = to + NT * n, kq = j >> 6, e4 = j & 63u;
#pragma unroll
        for (u32 c = 0; c < 4u; c++)
          *(LDSP hk4 *)(fa2_vt + (4u * e4 + c) * FA2_VS + 4u * kq) = (hk4){xv[n][0][c], xv[n][1][c], xv[n][2][c], xv[n][3][c]};
      }
    }
    barrier();
    if (dos) { /* phase A */
      v8f s[2];
#pragma unroll
      for (u32 j = 0; j < 2u; j++) {
        s[j] = zero;
#pragma unroll
        for (u32 ks = 0; ks < 16u; ks++) {
          const h8 kb8 = *(const LDSP h8 *)(fa_k + (16u * j + rlo) * FA_KS + 16u * ks + 8u * hlo);
          s[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qa[ks], kb8, s[j]);
        }
      }
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const u32 tp = pos0 + t0 + 8u * hlo + v;
        const u32 k0 = kb + rlo, k1 = kb + 16u + rlo;
        const float s0 = k0 <= tp ? s[0][v] : -__builtin_inff(), s1 = k1 <= tp ? s[1][v] : -__builtin_inff();
        float rm = __builtin_fmaxf(s0, s1);
#pragma unroll
        for (u32 x = 1u; x < 16u; x <<= 1) rm = __builtin_fmaxf(rm, shfl_xor_f(rm, x));
        const float mn = __builtin_fmaxf(m[v], rm);
        const float c = __builtin_expf(m[v] - mn);
        const float p0 = __builtin_expf(s0 - mn), p1 = __builtin_expf(s1 - mn);
        ls[v] = ls[v] * c + p0 + p1;
        m[v] = mn;
        LDSP f16 *pr = fa_p[w] + (8u * hlo + v) * FA_VS;
        pr[rlo] = (f16)p0;
        pr[16u + rlo] = (f16)p1;
        if (rlo == 0u) fa2_c[w][8u * hlo + v] = c;
      }
    }
    barrier();
    h8 pa[2];
    float cr[8];
#pragma unroll
    for (u32 t = 0; t < OT; t++) { /* phase B: tiles tb .. tb + tn - 1 (head i / 16, columns 16 (i % 16) ..) */
      if (t < tn) {
        const u32 i = tb + t, hh = i >> 4, nt = i & 15u;
        if (t == 0u || nt == 0u) { /* a new head (uniform): its P and row rescales */
#pragma unroll
          for (u32 kk = 0; kk < 2u; kk++) pa[kk] = *(const LDSP h8 *)(fa_p[hh] + rlo * FA_VS + 16u * kk + 8u * hlo);
#pragma unroll
          for (u32 v = 0; v < 8u; v++) cr[v] = fa2_c[hh][8u * hlo + v];
        }
#pragma unroll
        for (u32 v = 0; v < 8u; v++) o[t][v] *= cr[v];
#pragma unroll
        for (u32 kk = 0; kk < 2u; kk++) {
          const h8 vb = *(const LDSP h8 *)(fa2_vt + (16u * nt + rlo) * FA2_VS + 16u * kk + 8u * hlo);
          o[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pa[kk], vb, o[t]);
        }
      }
    }
    barrier();
  }
  u32 le = lane(), ge = wgid(); /* opaque: the epilogue's addresses are not computed before the loop */
  __asm__ volatile("" : "+v"(le), "+v"(ge));
  if (dos)
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      float x = ls[v];
#pragma unroll
      for (u32 s = 1u; s < 16u; s <<= 1) x += shfl_xor_f(x, s);
      if ((le & 15u) == 0u) fa2_c[w][8u * (le >> 4) + v] = 1.0f / x;
    }
  barrier();
  G float *const out = fa2_ep.out;
  const G float *const gate = fa2_ep.gate;
  const u32 gstride = fa2_ep.gstride, os = fa2_ep.os, gs = fa2_ep.gs, Te = fa2_ep.T, nte = fa2_ep.ntile;
  const u32 hle = le >> 4, rle = le & 15u, t0e = 16u * (ge % nte), khe = ge / nte;
#pragma unroll
  for (u32 t = 0; t < OT; t++)
    if (t < tn) {
      const u32 i = tb + t, hh = i >> 4, nt = i & 15u, h = khe * ga + hh;
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const u32 tk = t0e + 8u * hle + v, d = 16u * nt + rle;
        if (tk >= Te) continue;
        float x = o[t][v] * fa2_c[hh][8u * hle + v];
        if (gate) x = x * sigmoidf(((const G float *)((const G u8 *)gate + (unsigned long)tk * gs))[h * gstride + d]);
        ((G float *)((G u8 *)out + (unsigned long)tk * os))[h * hd + d] = x;
      }
    }
}
KERNEL ie_attn_fa2(const G float *q, const G kvt *kc, const G kvt *vc, G float *out, u32 pos0, u32 hd, u32 n_head,
                   u32 n_kv, const G float *gate, u32 gstride, u32 qs, u32 os, u32 gs, u32 T) {
  (void)hd;
  if (T == 0u) return;
  if (tid() == 0u) {
    fa2_ep.out = out;
    fa2_ep.gate = gate;
    fa2_ep.gstride = gstride;
    fa2_ep.os = os;
    fa2_ep.gs = gs;
    fa2_ep.T = T;
    fa2_ep.ntile = (T + 15u) / 16u;
  }
  const u32 ga = n_head / n_kv, w = __builtin_amdgcn_readfirstlane(tid() >> 5), ntot = 16u * ga;
  const u32 na = FA2_OA * ga < ntot ? FA2_OA * ga : ntot; /* the tiles of the S waves */
  const u32 nbw = 8u - ga, per = (ntot - na + nbw - 1u) / nbw; /* the others: per wave (<= FA2_OB) */
  if (w < ga) {
    v8f oa[FA2_OA];
    const u32 tb = w * FA2_OA, tn = tb < na ? (na - tb < FA2_OA ? na - tb : FA2_OA) : 0u;
    fa2_run(q, kc, vc, pos0, n_head, n_kv, qs, T, 1, FA2_OA, tb, tn, oa);
  } else {
    v8f ob[FA2_OB];
    const u32 tb = na + (w - ga) * per, tn = tb < ntot ? (ntot - tb < per ? ntot - tb : per) : 0u;
    fa2_run(q, kc, vc, pos0, n_head, n_kv, qs, T, 0, FA2_OB, tb, tn, ob);
  }
}

/* The Q8 copy of ie_attn_fa's output (fused into ie_attn_pfg, a separate
 * pass here: the parts of a head are in different workgroups). Workgroup y
 * = token, wave = block of 32, as ie_quant_q8. */
KERNEL ie_attn_fa_q8(const G float *out, G u8 *oq, u32 nb, u32 os, u32 oqs) {
  out = TOKC(out, os), oq = TOK(oq, oqs);
  const u32 b = wgid() * 8u + (tid() >> 5);
  if (b >= nb) return;
  quant_wave(out[b * 32u + lane()], b, oq, nb);
}

/* ------------------------------------------------------------------ argmax */

/* out[0] = the first index of the largest x. ng workgroups: workgroup g
 * scans i = g*NT + tid, step ng*NT, and writes its best (value, index) to
 * part[g]. The last workgroup to finish (atomic counter) combines the ng
 * partial results and sets the counter back to 0 for the next step.
 * Ties: the smaller index wins, as in the CPU backend. */
static LDS float am_v[8];
static LDS u32 am_i[8];
static LDS u32 am_last;
static inline int better(float v, u32 i, float bv, u32 bi) {
  return i != 0xFFFFFFFFu && (bi == 0xFFFFFFFFu || v > bv || (v == bv && i < bi));
}
static inline void wg_best(float *bv, u32 *bi) {
  float v = *bv;
  u32 i = *bi;
#pragma unroll
  for (u32 m = 16u; m > 0u; m >>= 1) {
    const float ov = shfl_xor_f(v, m);
    const u32 oi = (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ m) * 4u), (int)i);
    if (better(ov, oi, v, i)) v = ov, i = oi;
  }
  if (lane() == 0u) am_v[tid() >> 5] = v, am_i[tid() >> 5] = i;
  lbarrier();
  v = am_v[0], i = am_i[0];
#pragma unroll
  for (u32 w = 1u; w < 8u; w++)
    if (better(am_v[w], am_i[w], v, i)) v = am_v[w], i = am_i[w];
  *bv = v, *bi = i;
  lbarrier();
}
/* out[0] = the first index of the largest x; out[1] (float bits) = its
 * softmax probability 1 / sum_i exp(x_i - max) (the MTP draft confidence).
 * part: 3 words per workgroup (max, index, sum of exp(x - max) over its
 * elements). The argmax itself is as before (the sums do not change it). */
KERNEL ie_argmax(const G float *x, u32 n, G u32 *out, G u32 *part, G u32 *count, u32 ng) {
  float bv = -__builtin_inff();
  u32 bi = 0xFFFFFFFFu;
  for (u32 i = wgid() * NT + tid(); i < n; i += ng * NT)
    if (better(x[i], i, bv, bi)) bv = x[i], bi = i;
  wg_best(&bv, &bi);
  float se = 0.0f;
  for (u32 i = wgid() * NT + tid(); i < n; i += ng * NT) se += __builtin_expf(x[i] - bv);
  se = wg_sum(se);
  if (tid() == 0u) {
    part[3u * wgid()] = __builtin_bit_cast(u32, bv);
    part[3u * wgid() + 1u] = bi;
    part[3u * wgid() + 2u] = __builtin_bit_cast(u32, se);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    am_last = __scoped_atomic_fetch_add(count, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u;
  }
  barrier();
  if (!am_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  bv = -__builtin_inff(), bi = 0xFFFFFFFFu;
  for (u32 g = tid(); g < ng; g += NT) {
    const float v = __builtin_bit_cast(float, __scoped_atomic_load_n(&part[3u * g], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE));
    const u32 i = __scoped_atomic_load_n(&part[3u * g + 1u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    if (better(v, i, bv, bi)) bv = v, bi = i;
  }
  wg_best(&bv, &bi);
  float S = 0.0f;
  for (u32 g = tid(); g < ng; g += NT) {
    const float v = __builtin_bit_cast(float, __scoped_atomic_load_n(&part[3u * g], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE));
    const float sg = __builtin_bit_cast(float, __scoped_atomic_load_n(&part[3u * g + 2u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE));
    if (sg > 0.0f) S += sg * __builtin_expf(v - bv);
  }
  S = wg_sum(S);
  if (tid() == 0u) {
    out[0] = bi;
    out[1] = __builtin_bit_cast(u32, 1.0f / S);
    *count = 0u;
  }
}

/* ------------------------------------------------------------ top K (sampling) */

/* The K largest logits of each row (K <= TK_MAX), for sampling: the host
 * then needs K (value, index) pairs instead of the whole row. Order: value
 * descending; equal values: the smaller index first (as the host's
 * selection). Two launches:
 *   ie_topk_part (grid ng x rows): workgroup g takes the indices
 *     [g cs, (g + 1) cs) of its row (cs <= TK_CS), keeps them in LDS and
 *     takes its K best one at a time (best of the workgroup, then removed);
 *   ie_topk_merge (grid rows): the ng K candidates of a row, the same way. */
#define TK_MAX 64u
#define TK_CS 4096u
static LDS float tk_v[TK_CS];
static LDS u32 tk_i[TK_CS];
static LDS float tk_wv[8];
static LDS u32 tk_wi[8], tk_wp[8];
/* (v, i) better than (bv, bi): larger value, or equal value and smaller index; i = ~0: none */
static inline int tk_better(float v, u32 i, float bv, u32 bi) {
  return i != 0xFFFFFFFFu && (bi == 0xFFFFFFFFu || v > bv || (v == bv && i < bi));
}
/* the best (value, index, LDS position) of the workgroup, in every thread */
static inline void tk_wg_best(float *bv, u32 *bi, u32 *bp) {
#pragma unroll
  for (u32 m = 16u; m > 0u; m >>= 1) {
    const float ov = shfl_xor_f(*bv, m);
    const u32 oi = (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ m) * 4u), (int)*bi);
    const u32 op = (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ m) * 4u), (int)*bp);
    if (tk_better(ov, oi, *bv, *bi)) *bv = ov, *bi = oi, *bp = op;
  }
  if (lane() == 0u) tk_wv[tid() >> 5] = *bv, tk_wi[tid() >> 5] = *bi, tk_wp[tid() >> 5] = *bp;
  lbarrier();
  float v = tk_wv[0];
  u32 i = tk_wi[0], p = tk_wp[0];
  for (u32 w = 1; w < 8u; w++)
    if (tk_better(tk_wv[w], tk_wi[w], v, i)) v = tk_wv[w], i = tk_wi[w], p = tk_wp[w];
  *bv = v, *bi = i, *bp = p;
}
/* the K best of the n (value, index) pairs in tk_v / tk_i, to ov / oi */
static inline void tk_select(u32 n, u32 K, G float *ov, G u32 *oi) {
  for (u32 k = 0; k < K; k++) {
    float bv = -__builtin_inff();
    u32 bi = 0xFFFFFFFFu, bp = 0u;
    for (u32 p = tid(); p < n; p += NT)
      if (tk_better(tk_v[p], tk_i[p], bv, bi)) bv = tk_v[p], bi = tk_i[p], bp = p;
    tk_wg_best(&bv, &bi, &bp);
    if (tid() == 0u) {
      ov[k] = bv, oi[k] = bi;
      if (bi != 0xFFFFFFFFu) tk_i[bp] = 0xFFFFFFFFu; /* taken */
    }
    lbarrier(); /* the removal is visible; tk_wv free again */
  }
}
KERNEL ie_topk_part(const G float *x, u32 n, u32 xs, u32 K, u32 cs, u32 ng, G float *cv, G u32 *ci) {
  const u32 r = tokid(), g = wgid();
  const G float *row = x + (unsigned long)r * xs;
  const u32 i0 = g * cs;
  for (u32 p = tid(); p < cs; p += NT) {
    const u32 i = i0 + p;
    tk_v[p] = i < n ? row[i] : -__builtin_inff();
    tk_i[p] = i < n ? i : 0xFFFFFFFFu;
  }
  lbarrier();
  tk_select(cs, K, cv + ((unsigned long)r * ng + g) * K, ci + ((unsigned long)r * ng + g) * K);
}
KERNEL ie_topk_merge(const G float *cv, const G u32 *ci, u32 nc, u32 K, G float *ov, G u32 *oi) {
  const u32 r = wgid();
  for (u32 p = tid(); p < nc; p += NT) tk_v[p] = cv[(unsigned long)r * nc + p], tk_i[p] = ci[(unsigned long)r * nc + p];
  lbarrier();
  tk_select(nc, K, ov + (unsigned long)r * K, oi + (unsigned long)r * K);
}

/* Top K with a threshold (the sampling path; same result and order as ie_topk_part + ie_topk_merge above, which take
 * K full passes over their segment). The row is cut into ng segments of cs values.
 *   ie_topk_max (grid ng x rows): the maximum of each segment, and resets the row's counters.
 *   ie_topk_sel (grid ng x rows): tau = the K-th largest of the ng segment maxima. At least K values are >= tau, so the
 *     K-th largest value of the row is >= tau, and every value of the answer is >= tau. Each workgroup appends the
 *     values >= tau of its segment (value, index) to the row's candidate list (a few dozen in practice). The last
 *     workgroup of the row to finish ranks the candidates (value descending, equal values smaller index first) and
 *     writes the first K. If more than TF_CAP values are >= tau (e.g. many equal logits) it writes the index 0xFFFFFFFF
 *     as the row's first index instead and the host takes the slow path for the run.
 * Candidate lists: cv / ci, TF_CAP entries per row; segment maxima smax and counters cnt / fin: 64 / 1 / 1 per row. */
#define TF_CAP 1024u
static LDS float tf_w[8];
static LDS float tf_wl[8];
static LDS float tf_m[64];
static LDS float tf_t;
static LDS u32 tf_last;
static LDS float tf_cv[TF_CAP];
static LDS u32 tf_ci[TF_CAP];
KERNEL ie_topk_max(const G float *x, u32 n, u32 xs, u32 cs, G float *smax, G u32 *cnt, G u32 *fin) {
  const u32 r = tokid(), g = wgid();
  const G float *row = x + (unsigned long)r * xs;
  const u32 i0 = g * cs;
  float m = -__builtin_inff(), l = 0.0f;
  for (u32 p = tid(); p < cs; p += NT) {
    const u32 i = i0 + p;
    if (i < n) {
      const float v = row[i];
      if (v > m) {
        l = l * __builtin_expf(m - v) + 1.0f;
        m = v;
      } else if (v > -__builtin_inff()) {
        l += __builtin_expf(v - m);
      }
    }
  }
  for (u32 s = 16u; s > 0u; s >>= 1) {
    const float om = shfl_xor_f(m, s), ol = shfl_xor_f(l, s);
    if (om > m) {
      l = l * __builtin_expf(m - om) + ol;
      m = om;
    } else if (m > -__builtin_inff()) {
      l += ol * __builtin_expf(om - m);
    }
  }
  if (lane() == 0u) tf_w[tid() >> 5] = m, tf_wl[tid() >> 5] = l;
  lbarrier();
  if (tid() == 0u) {
    float t = tf_w[0], tl = tf_wl[0];
    for (u32 w = 1; w < 8u; w++) {
      const float wm = tf_w[w], wl = tf_wl[w];
      if (wm > t) {
        tl = tl * __builtin_expf(t - wm) + wl;
        t = wm;
      } else if (t > -__builtin_inff()) {
        tl += wl * __builtin_expf(wm - t);
      }
    }
    smax[r * 64u + g] = t;
    smax[1024u + r * 64u + g] = tl;
    if (g == 0u) cnt[r] = 0u, fin[r] = 0u;
  }
}
KERNEL ie_topk_sel(const G float *x, u32 n, u32 xs, u32 K, u32 cs, u32 ng, const G float *smax, G float *cv, G u32 *ci,
                   G u32 *cnt, G u32 *fin, G float *ov, G u32 *oi) {
  const u32 r = tokid(), g = wgid();
  if (tid() < 64u) tf_m[tid()] = tid() < ng ? smax[r * 64u + tid()] : -__builtin_inff();
  if (tid() == 0u) tf_t = -__builtin_inff(); /* K > ng: everything is a candidate (the host does not launch this) */
  lbarrier();
  if (tid() < ng) {
    const float v = tf_m[tid()];
    u32 rank = 0;
    for (u32 u = 0; u < ng; u++) rank += (tf_m[u] > v) | ((tf_m[u] == v) & (u < tid()));
    if (rank + 1u == K) tf_t = v;
  }
  lbarrier();
  const float tau = tf_t;
  const G float *row = x + (unsigned long)r * xs;
  const u32 i0 = g * cs;
  float v[16]; /* cs <= 4096 */
#pragma unroll
  for (u32 k = 0; k < 16u; k++) {
    const u32 p = tid() + k * NT;
    v[k] = p < cs && i0 + p < n ? row[i0 + p] : -__builtin_inff();
  }
#pragma unroll
  for (u32 k = 0; k < 16u; k++) {
    const u32 p = tid() + k * NT;
    if (v[k] >= tau && p < cs && i0 + p < n) {
      const u32 pos = __scoped_atomic_fetch_add(cnt + r, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      if (pos < TF_CAP) cv[(unsigned long)r * TF_CAP + pos] = v[k], ci[(unsigned long)r * TF_CAP + pos] = i0 + p;
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __builtin_amdgcn_s_barrier();
  if (tid() == 0u) tf_last = __scoped_atomic_fetch_add(fin + r, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u;
  barrier();
  if (!tf_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  if (tid() == 0u) {
    float rm = -__builtin_inff(), rl = 0.0f;
    for (u32 u = 0; u < ng; u++) {
      const float um = tf_m[u], ul = smax[1024u + r * 64u + u];
      if (um > rm) {
        rl = rl * __builtin_expf(rm - um) + ul;
        rm = um;
      } else if (rm > -__builtin_inff()) {
        rl += ul * __builtin_expf(um - rm);
      }
    }
    ((G float *)smax)[2048u + r] = rl > 0.0f ? rm + __builtin_logf(rl) : -__builtin_inff();
  }
  const u32 c = __scoped_atomic_load_n(cnt + r, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  if (c > TF_CAP) {
    if (tid() == 0u) oi[(unsigned long)r * K] = 0xFFFFFFFFu;
    return;
  }
  for (u32 j = tid(); j < c; j += NT) tf_cv[j] = cv[(unsigned long)r * TF_CAP + j], tf_ci[j] = ci[(unsigned long)r * TF_CAP + j];
  lbarrier();
  for (u32 j = tid(); j < c; j += NT) {
    const float a = tf_cv[j];
    const u32 ai = tf_ci[j];
    u32 rank = 0;
    for (u32 u = 0; u < c; u++) rank += (tf_cv[u] > a) | ((tf_cv[u] == a) & (tf_ci[u] < ai));
    if (rank < K) ov[(unsigned long)r * K + rank] = a, oi[(unsigned long)r * K + rank] = ai;
  }
  if (c < K && tid() < K - c) ov[(unsigned long)r * K + c + tid()] = -__builtin_inff(), oi[(unsigned long)r * K + c + tid()] = 0xFFFFFFFFu;
}

/* ------------------------------------------------------------------ flint */
/* Recursive flint (src/model.h ARCH_FLINT): the weight kinds MAT_Q5K and
 * MAT_IQ4XS (core/ie_ext.h: tested, not proved), the MoE ops, and attention
 * with a sliding window (a KV ring) and learned sinks. One token per launch
 * (decode); the CPU backend (cpu.c) does the same arithmetic in the same
 * order (expf and the reductions round differently on the two). */

/* MAT_Q5K: as gemv_q4kq8_rows, with dq = ie_q5k_dot (q = 0..31). */
static inline void gemv_q5kq8_rows(u32 g, const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows,
                                   u32 nb, const G float *bias, const G float *res) {
#pragma clang fp contract(off)
  const u32 r = ie_gemv_row(g, tid() >> 5), l = lane();
  if (r < rows) {
    const G u32 *aw = (const G u32 *)xq;
    const G float *da = (const G float *)(xq + 32u * nb);
    const G u32 *asum = (const G u32 *)(xq + 36u * nb);
    const G u16 *smr = qs + ie_q4k_sm(r, 0u, nb);
    const G f16 *ddr = (const G f16 *)qs + ie_q4k_dd(rows, r, 0u, nb);
    const G u32 *hwr = qw + ie_q5k_hw(rows, r, 0u, nb);
    float acc = 0.0f;
    for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
      u32x4 w4[GEMV_U], a0[GEMV_U], a1[GEMV_U];
      u32 hb[GEMV_U], sm[GEMV_U], as[GEMV_U];
      float dv[GEMV_U], mv[GEMV_U], dav[GEMV_U];
      int ok[GEMV_U];
      for (u32 k = 0; k < GEMV_U; k++) {
        ok[k] = ie_lane_blk(l, t + k) < nb;
        const u32 b = ok[k] ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
        w4[k] = *(const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb));
        hb[k] = hwr[b];
        a0[k] = *(const G u32x4 *)(aw + b * 8u);
        a1[k] = *(const G u32x4 *)(aw + b * 8u + 4u);
        sm[k] = smr[b];
        dv[k] = (float)ddr[(b >> 3) * 2u], mv[k] = (float)ddr[(b >> 3) * 2u + 1u];
        as[k] = asum[b], dav[k] = da[b];
      }
      for (u32 k = 0; k < GEMV_U; k++) {
        const u32 wq[4] = {w4[k].x, w4[k].y, w4[k].z, w4[k].w};
        const u32 av[8] = {a0[k].x, a0[k].y, a0[k].z, a0[k].w, a1[k].x, a1[k].y, a1[k].z, a1[k].w};
        const Mem qm = {wq}, am = {av};
        const int dq = (int)ie_q5k_dot(qm, 0u, hb[k], am, 0u);
        const float sc = dv[k] * (float)(sm[k] & 63u), mm = mv[k] * (float)(sm[k] >> 8);
        const float x = dav[k] * (float)(int)as[k];
        const float a1v = __builtin_fmaf(sc * (float)dq, dav[k], acc), a2 = __builtin_fmaf(-mm, x, a1v);
        acc = ok[k] ? a2 : acc;
      }
    }
    acc = wave_tree_f(acc);
    if (l == 0u) y[r] = epilogue(acc, r, bias, res);
  }
}
KERNEL ie_gemv_q5kq8(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                     const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps,
                     G u32 *count) {
  gemv_q5kq8_rows(wgid(), qw, qs, xq, y, rows, nb, bias, res);
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

/* MAT_IQ4XS: dq = ie_iq4_dot (the table values), acc = fma(d (ls - 32) dq, da, acc). */
static inline void gemv_iq4xsq8_rows(u32 g, const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows,
                                     u32 nb, const G float *bias, const G float *res) {
#pragma clang fp contract(off)
  const u32 r = ie_gemv_row(g, tid() >> 5), l = lane();
  if (r < rows) {
    const G u32 *aw = (const G u32 *)xq;
    const G float *da = (const G float *)(xq + 32u * nb);
    const G u16 *smr = qs + ie_q4k_sm(r, 0u, nb);
    const G f16 *ddr = (const G f16 *)qs + ie_q4k_dd(rows, r, 0u, nb);
    float acc = 0.0f;
    for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
      u32x4 w4[GEMV_U], a0[GEMV_U], a1[GEMV_U];
      u32 sm[GEMV_U];
      float dv[GEMV_U], dav[GEMV_U];
      int ok[GEMV_U];
      for (u32 k = 0; k < GEMV_U; k++) {
        ok[k] = ie_lane_blk(l, t + k) < nb;
        const u32 b = ok[k] ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
        w4[k] = *(const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb));
        a0[k] = *(const G u32x4 *)(aw + b * 8u);
        a1[k] = *(const G u32x4 *)(aw + b * 8u + 4u);
        sm[k] = smr[b];
        dv[k] = (float)ddr[(b >> 3) * 2u];
        dav[k] = da[b];
      }
      for (u32 k = 0; k < GEMV_U; k++) {
        const u32 wq[4] = {w4[k].x, w4[k].y, w4[k].z, w4[k].w};
        const u32 av[8] = {a0[k].x, a0[k].y, a0[k].z, a0[k].w, a1[k].x, a1[k].y, a1[k].z, a1[k].w};
        const Mem qm = {wq}, am = {av};
        const int dq = (int)ie_iq4_dot(qm, 0u, am, 0u);
        const float sc = dv[k] * (float)((int)(sm[k] & 63u) - 32);
        const float a1v = __builtin_fmaf(sc * (float)dq, dav[k], acc);
        acc = ok[k] ? a1v : acc;
      }
    }
    acc = wave_tree_f(acc);
    if (l == 0u) y[r] = epilogue(acc, r, bias, res);
  }
}
KERNEL ie_gemv_iq4xsq8(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                       const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps,
                       G u32 *count) {
  gemv_iq4xsq8_rows(wgid(), qw, qs, xq, y, rows, nb, bias, res);
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

/* IQ4_XS for up to 8 tokens (verify runs): each weight block is read once;
 * token j's sum is formed as gemv_iq4xsq8_rows forms it (the same blocks per
 * lane in the same order, the same fma), so the result is bitwise the same.
 * Token j reads xq + j*xs and writes y + j*ys (bytes); res + j*rs. */
KERNEL ie_gemv_iq4xsq8_tr(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                          const G float *bias, const G float *res, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  const u32 r = ie_gemv_row(wgid(), (u32)__builtin_amdgcn_readfirstlane(tid() >> 5)), l = lane();
  if (r >= rows || T == 0u) return;
  const G u16 *smr = qs + ie_q4k_sm(r, 0u, nb);
  const G f16 *ddr = (const G f16 *)qs + ie_q4k_dd(rows, r, 0u, nb);
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
    for (u32 k = 0; k < GEMV_U; k++) {
      const int ok = ie_lane_blk(l, t + k) < nb;
      const u32 b = ok ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
      const u32x4 w4 = *(const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb));
      const u32 wq[4] = {w4.x, w4.y, w4.z, w4.w};
      const Mem qm = {wq};
      const float sc = (float)ddr[(b >> 3) * 2u] * (float)((int)(smr[b] & 63u) - 32);
#pragma clang loop unroll(full)
      for (u32 j = 0; j < 8u; j++) {
        if (j < T) {
          const G u8 *xj = xq + (unsigned long)j * xs;
          const G u32 *aw = (const G u32 *)xj;
          const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u);
          const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
          const Mem am = {av};
          const int dq = (int)ie_iq4_dot(qm, 0u, am, 0u);
          const float a1v = __builtin_fmaf(sc * (float)dq, ((const G float *)(xj + 32u * nb))[b], acc[j]);
          acc[j] = ok ? a1v : acc[j];
        }
      }
    }
  }
#pragma clang loop unroll(full)
  for (u32 j = 0; j < 8u; j++) {
    if (j < T) {
      const float v = wave_tree_f(acc[j]);
      if (l == 0u)
        ((G float *)((G u8 *)y + (unsigned long)j * ys))[r] =
            epilogue(v, r, bias, res ? (const G float *)((const G u8 *)res + (unsigned long)j * rs) : res);
    }
  }
}

/* MAT_IQ4NL: Q4_0 nibble packing + f16 block scale qs[r * nb + b], signed weights from kvalues_iq4nl (ie_iq4_dot). */
static inline void gemv_iq4nlq8_rows(u32 g, const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows,
                                     u32 nb, const G float *bias, const G float *res) {
#pragma clang fp contract(off)
  const u32 r = ie_gemv_row(g, tid() >> 5), l = lane();
  if (r < rows) {
    const G u32 *aw = (const G u32 *)xq;
    const G float *da = (const G float *)(xq + 32u * nb);
    float acc = 0.0f;
    for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
      u32x4 w4[GEMV_U], a0[GEMV_U], a1[GEMV_U];
      float sc[GEMV_U], dav[GEMV_U];
      int ok[GEMV_U];
      for (u32 k = 0; k < GEMV_U; k++) {
        ok[k] = ie_lane_blk(l, t + k) < nb;
        const u32 b = ok[k] ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
        w4[k] = *(const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb));
        a0[k] = *(const G u32x4 *)(aw + b * 8u);
        a1[k] = *(const G u32x4 *)(aw + b * 8u + 4u);
        sc[k] = (float)qs[ie_q4_dst_scale(r, b, nb)];
        dav[k] = da[b];
      }
      for (u32 k = 0; k < GEMV_U; k++) {
        const u32 wq[4] = {w4[k].x, w4[k].y, w4[k].z, w4[k].w};
        const u32 av[8] = {a0[k].x, a0[k].y, a0[k].z, a0[k].w, a1[k].x, a1[k].y, a1[k].z, a1[k].w};
        const Mem qm = {wq}, am = {av};
        const int dq = (int)ie_iq4_dot(qm, 0u, am, 0u);
        const float a1v = __builtin_fmaf(sc[k] * (float)dq, dav[k], acc);
        acc = ok[k] ? a1v : acc;
      }
    }
    acc = wave_tree_f(acc);
    if (l == 0u) y[r] = epilogue(acc, r, bias, res);
  }
}
KERNEL ie_gemv_iq4nlq8(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                       const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps,
                       G u32 *count) {
  gemv_iq4nlq8_rows(wgid(), qw, qs, xq, y, rows, nb, bias, res);
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}
KERNEL ie_gemv_iq4nlq8_tr(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                          const G float *bias, const G float *res, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  const u32 r = ie_gemv_row(wgid(), (u32)__builtin_amdgcn_readfirstlane(tid() >> 5)), l = lane();
  if (r >= rows || T == 0u) return;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
    for (u32 k = 0; k < GEMV_U; k++) {
      const int ok = ie_lane_blk(l, t + k) < nb;
      const u32 b = ok ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
      const u32x4 w4 = *(const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb));
      const u32 wq[4] = {w4.x, w4.y, w4.z, w4.w};
      const Mem qm = {wq};
      const float sc = (float)qs[ie_q4_dst_scale(r, b, nb)];
#pragma clang loop unroll(full)
      for (u32 j = 0; j < 8u; j++) {
        if (j < T) {
          const G u8 *xj = xq + (unsigned long)j * xs;
          const G u32 *aw = (const G u32 *)xj;
          const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u);
          const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
          const Mem am = {av};
          const int dq = (int)ie_iq4_dot(qm, 0u, am, 0u);
          const float a1v = __builtin_fmaf(sc * (float)dq, ((const G float *)(xj + 32u * nb))[b], acc[j]);
          acc[j] = ok ? a1v : acc[j];
        }
      }
    }
  }
#pragma clang loop unroll(full)
  for (u32 j = 0; j < 8u; j++) {
    if (j < T) {
      const float v = wave_tree_f(acc[j]);
      if (l == 0u)
        ((G float *)((G u8 *)y + (unsigned long)j * ys))[r] =
            epilogue(v, r, bias, res ? (const G float *)((const G u8 *)res + (unsigned long)j * rs) : res);
    }
  }
}

/* MAT_RAWQ: the raw GGUF 256-weight blocks (kernels/ie_iqraw.inc) against
 * the Q8 activations. Lane l of the row's wave takes the 32-weight sub-blocks
 * l, l + 32, ... of the row, so all 32 lanes are busy at any row width (a
 * 5120-column row has 160 sub-blocks: five per lane, against twenty blocks on
 * twenty lanes before); then the wave tree sums the lanes. */
/* The sub-block loop unrolled per type, so a lane's next sub-blocks are
 * loaded while it works on this one (the order of the sum is unchanged: the
 * output is bitwise the same). Separate launches (full occupancy, measured
 * with tools/raw_gemv_bench.py on Saluki's shapes): 4 helps iq2xxs/iq2xs by
 * 15-30%, 2-3 the others by a few percent; q2k and iq1m are ALU bound. In the
 * megakernel (6 waves per SIMD, one 192-VGPR frame; measured with its
 * IE_MK_TRACE time per kernel) only iq2xxs at 4 (-17%) and iq1s at 5 (-12%)
 * gain: the other types got 9-45% slower unrolled there. */
static inline u32 gemv_rawq_unroll(u32 type) {
  switch (type) {
#ifdef IE_MEGA
    case IE_RQ_IQ2_XXS: return 4u;
    case IE_RQ_IQ1_S: return 5u;
#else
    case IE_RQ_Q2_K: return 3u;
    case IE_RQ_IQ2_XXS: return 4u;
    case IE_RQ_IQ2_XS: return 4u;
    case IE_RQ_IQ3_XXS: return 3u;
    case IE_RQ_IQ1_S: return 4u;
    case IE_RQ_IQ1_M: return 4u;
    case IE_RQ_IQ3_S: return 4u;
    case IE_RQ_IQ2_S: return 3u;
#endif
    default: return 1u;
  }
}
#define IE_PRAGMA(x) _Pragma(#x)
#ifdef IE_MEGA
#define RAWQ_LOOP(U)                                                     \
  IE_PRAGMA(clang loop unroll_count(U))                                  \
  for (u32 b = l; b < nb; b += 32u) {                                    \
    acc += ie_rq_dot_row(type, row, xq, nb, b);                          \
  }
static inline void gemv_rawq_rows(u32 g, const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 type,
                                  const G float *bias, const G float *res) {
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 r = ie_gemv_row(g, wv), l = lane();
  if (r < rows) {
    const u32 mb = nb / 8u, bs = ie_rq_bs(type);
    const G u8 *row = q + (unsigned long)r * mb * bs;
    float acc = 0.0f;
    switch (gemv_rawq_unroll(type)) { /* a constant in each typed kernel */
      case 2u: RAWQ_LOOP(2) break;
      case 3u: RAWQ_LOOP(3) break;
      case 4u: RAWQ_LOOP(4) break;
      case 5u: RAWQ_LOOP(5) break;
      default:
        for (u32 b = l; b < nb; b += 32u) acc += ie_rq_dot_row(type, row, xq, nb, b);
    }
    acc = wave_tree_f(acc);
    if (l == 0u) y[r] = epilogue(acc, r, bias, res);
  }
}
#else
LDS static ie_rq_u64 rq1_lds[2048];
static inline __attribute__((always_inline)) const G u8 *rq_uniform(const G u8 *p) {
  const unsigned long v = (unsigned long)p;
  const u32 lo = (u32)__builtin_amdgcn_readfirstlane((u32)v), hi = (u32)__builtin_amdgcn_readfirstlane((u32)(v >> 32u));
  return (const G u8 *)(((unsigned long)hi << 32u) | (unsigned long)lo);
}
#define RAWQ_LOOP(U)                                                     \
  IE_PRAGMA(clang loop unroll_count(U))                                  \
  for (u32 b = l; b < nb; b += 32u) {                                    \
    acc += ie_rq_dot_row_lds(type, row, xq, nb, b, rq1_lds);             \
  }
static inline void gemv_rawq_rows(u32 g, const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 type,
                                  const G float *bias, const G float *res) {
  switch (type) {
    case IE_RQ_IQ2_XXS:
      rq1_lds[tid()] = ie_rq_iq2xxs_grid[tid()];
      if (tid() < 128u) rq1_lds[256u + tid()] = ie_rq_ksigns64[tid()];
      lbarrier();
      break;
    case IE_RQ_IQ2_XS:
      for (u32 i = tid(); i < 512u; i += NT) rq1_lds[i] = ie_rq_iq2xs_grid[i];
      if (tid() < 128u) rq1_lds[512u + tid()] = ie_rq_ksigns64[tid()];
      lbarrier();
      break;
    case IE_RQ_IQ2_S:
      for (u32 i = tid(); i < 1024u; i += NT) rq1_lds[i] = ie_rq_iq2s_grid[i];
      lbarrier();
      break;
    case IE_RQ_IQ3_XXS:
      ((LDSP u32 *)rq1_lds)[tid()] = ie_rq_iq3xxs_grid[tid()];
      if (tid() < 128u) rq1_lds[128u + tid()] = ie_rq_ksigns64[tid()];
      lbarrier();
      break;
    case IE_RQ_IQ3_S:
      for (u32 i = tid(); i < 512u; i += NT) ((LDSP u32 *)rq1_lds)[i] = ie_rq_iq3s_grid[i];
      lbarrier();
      break;
    case IE_RQ_IQ1_S:
    case IE_RQ_IQ1_M:
      for (u32 i = tid(); i < 2048u; i += NT) rq1_lds[i] = ie_rq_iq1s_grid[i];
      lbarrier();
      break;
    default: break;
  }
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 r = ie_gemv_row(g, wv), l = lane();
  if (r < rows) {
    const u32 mb = nb / 8u, bs = ie_rq_bs(type);
    const G u8 *row = rq_uniform(q + (unsigned long)r * mb * bs);
    xq = rq_uniform(xq);
    float acc = 0.0f;
    switch (gemv_rawq_unroll(type)) { /* a constant in each typed kernel */
      case 2u: RAWQ_LOOP(2) break;
      case 3u: RAWQ_LOOP(3) break;
      case 4u: RAWQ_LOOP(4) break;
      case 5u: RAWQ_LOOP(5) break;
      default:
        for (u32 b = l; b < nb; b += 32u) acc += ie_rq_dot_row_lds(type, row, xq, nb, b, rq1_lds);
    }
    acc = wave_tree_f(acc);
    if (l == 0u) y[r] = epilogue(acc, r, bias, res);
  }
}
#endif
/* One kernel per ggml type: with the type a compile-time constant the switch
 * in ie_rq_dot folds away, so each kernel carries only its own format's code.
 * That matters for the megakernel, where every body sits in one launch: a
 * single generic kernel costs 136 VGPR and pushed ie_mk to 248 VGPR with
 * spills; seven typed ones cost whatever the largest format needs. The type
 * argument stays (and is ignored) so the host's launch layout is unchanged. */
KERNEL_LOOP ie_gemv_raw_q2k(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 type,
                            const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                            float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)type;
  const G u8 *x = TOKC(xq, xs);
  G float *yy = TOK(y, ys);
  const G float *rr = res ? (const G float *)((const G u8 *)res + (unsigned long)tokid() * rs) : res;
  gemv_rawq_rows(wgid(), q, x, yy, rows, nb, IE_RQ_Q2_K, bias, rr);
  if (nw) gemv_norm_tail(yy, rows, nw, ny, nq, eps, count);
}
KERNEL_LOOP ie_gemv_raw_iq2xxs(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 type,
                            const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                            float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)type;
  const G u8 *x = TOKC(xq, xs);
  G float *yy = TOK(y, ys);
  const G float *rr = res ? (const G float *)((const G u8 *)res + (unsigned long)tokid() * rs) : res;
  gemv_rawq_rows(wgid(), q, x, yy, rows, nb, IE_RQ_IQ2_XXS, bias, rr);
  if (nw) gemv_norm_tail(yy, rows, nw, ny, nq, eps, count);
}
KERNEL_LOOP ie_gemv_raw_iq2xs(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 type,
                            const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                            float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)type;
  const G u8 *x = TOKC(xq, xs);
  G float *yy = TOK(y, ys);
  const G float *rr = res ? (const G float *)((const G u8 *)res + (unsigned long)tokid() * rs) : res;
  gemv_rawq_rows(wgid(), q, x, yy, rows, nb, IE_RQ_IQ2_XS, bias, rr);
  if (nw) gemv_norm_tail(yy, rows, nw, ny, nq, eps, count);
}
KERNEL_LOOP ie_gemv_raw_iq3xxs(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 type,
                            const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                            float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)type;
  const G u8 *x = TOKC(xq, xs);
  G float *yy = TOK(y, ys);
  const G float *rr = res ? (const G float *)((const G u8 *)res + (unsigned long)tokid() * rs) : res;
  gemv_rawq_rows(wgid(), q, x, yy, rows, nb, IE_RQ_IQ3_XXS, bias, rr);
  if (nw) gemv_norm_tail(yy, rows, nw, ny, nq, eps, count);
}
KERNEL_LOOP ie_gemv_raw_iq1s(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 type,
                            const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                            float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)type;
  const G u8 *x = TOKC(xq, xs);
  G float *yy = TOK(y, ys);
  const G float *rr = res ? (const G float *)((const G u8 *)res + (unsigned long)tokid() * rs) : res;
  gemv_rawq_rows(wgid(), q, x, yy, rows, nb, IE_RQ_IQ1_S, bias, rr);
  if (nw) gemv_norm_tail(yy, rows, nw, ny, nq, eps, count);
}
KERNEL_LOOP ie_gemv_raw_iq2s(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 type,
                            const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                            float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)type;
  const G u8 *x = TOKC(xq, xs);
  G float *yy = TOK(y, ys);
  const G float *rr = res ? (const G float *)((const G u8 *)res + (unsigned long)tokid() * rs) : res;
  gemv_rawq_rows(wgid(), q, x, yy, rows, nb, IE_RQ_IQ2_S, bias, rr);
  if (nw) gemv_norm_tail(yy, rows, nw, ny, nq, eps, count);
}
KERNEL_LOOP ie_gemv_raw_iq1m(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 type,
                            const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                            float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)type;
  const G u8 *x = TOKC(xq, xs);
  G float *yy = TOK(y, ys);
  const G float *rr = res ? (const G float *)((const G u8 *)res + (unsigned long)tokid() * rs) : res;
  gemv_rawq_rows(wgid(), q, x, yy, rows, nb, IE_RQ_IQ1_M, bias, rr);
  if (nw) gemv_norm_tail(yy, rows, nw, ny, nq, eps, count);
}

/* The GEMVs of the chosen experts, one launch: grid x = the row groups of
 * one expert, grid y = slot s < K. Expert e = route[s]; its arrays start at
 * e qwn words, e qsn entries; input xq + s astr, output y + s bstr (bytes).
 * kind: MAT_Q8 (2), MAT_Q4K (3), MAT_Q6K (4), MAT_Q5K (5), MAT_IQ4XS (6) (src/model.h). */
/* The sum of the 16 lanes of each half of a wave (lanes 0 and 16 hold it):
 * the steps 2 to 5 of wave_tree_f (shuffle down 8, 4, 2, 1 inside each row
 * of 16). With the other half zero, wave_tree_f gives the same value. */
static inline float half_tree_f(float v) {
  v += I2F(shl_i(F2I(v), 8));
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}
/* gexp_block's arithmetic with the activation words av, scale da and sum asb of block b */
static inline float gexp_dot(float acc, u32 kind, const G u32 *qw, const G u16 *qs, u32 rows, u32 nb, u32 r, u32 b,
                             const u32 av[8], float da, u32 asb) {
#pragma clang fp contract(off)
  const Mem am = {av};
  if (kind == 2u) {
    const G u32 *wp = qw + ie_q8_dst_word(r, b, 0u, nb);
    const u32x4 w0 = *(const G u32x4 *)wp, w1 = *(const G u32x4 *)(wp + 4u);
    const u32 wq[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
    const Mem qm = {wq};
    const float sc = (float)((const G f16 *)qs)[ie_q8_dst_scale(r, b, nb)] * da;
    return acc + sc * (float)(int)ie_q8q8_block(qm, 0u, am, 0u);
  }
  const u32x4 w4 = *(const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb));
  const u32 wq[4] = {w4.x, w4.y, w4.z, w4.w};
  const Mem qm = {wq};
  if (kind == 0u) { /* MAT_Q4 */
    const int dot = (int)ie_q4q8_block(qm, 0u, am, 0u, asb);
    const float sc = (float)((const G f16 *)qs)[ie_q4_dst_scale(r, b, nb)] * da;
    return acc + sc * (float)dot;
  }
  if (kind == 8u) { /* MAT_IQ4NL */
    const int dq = (int)ie_iq4_dot(qm, 0u, am, 0u);
    const float sc = (float)((const G f16 *)qs)[ie_q4_dst_scale(r, b, nb)];
    return __builtin_fmaf(sc * (float)dq, da, acc);
  }
  if (kind == 4u) {
    const u32 hb = ie_q6k_hw(rows, r, b, 0u, nb), hq[2] = {qw[hb], qw[hb + 1u]};
    const Mem hm = {hq};
    const int d0 = (int)ie_q6k_dots(qm, 0u, hm, 0u, am, 0u, 0u), d1 = (int)ie_q6k_dots(qm, 0u, hm, 1u, am, 0u, 1u);
    const u32 sc = qs[ie_q6k_sc(r, b, nb)];
    const float dv = (float)((const G f16 *)qs)[ie_q6k_d(rows, r, b, nb)];
    const float s0 = dv * (float)(int)(signed char)(sc & 255u), s1 = dv * (float)(int)(signed char)(sc >> 8);
    return __builtin_fmaf(__builtin_fmaf(s1, (float)d1, s0 * (float)d0), da, acc);
  }
  const u32 sm = qs[ie_q4k_sm(r, b, nb)];
  const float dv = (float)((const G f16 *)qs)[ie_q4k_dd(rows, r, b, nb)];
  if (kind == 6u) { /* IQ4_XS */
    const int dq = (int)ie_iq4_dot(qm, 0u, am, 0u);
    const float sc = dv * (float)((int)(sm & 63u) - 32);
    return __builtin_fmaf(sc * (float)dq, da, acc);
  }
  /* Q4_K (3), Q5_K (5) */
  const int dq = kind == 5u ? (int)ie_q5k_dot(qm, 0u, qw[ie_q5k_hw(rows, r, b, nb)], am, 0u) : (int)ie_q4k_dot(qm, 0u, am, 0u);
  const float mv = (float)((const G f16 *)qs)[ie_q4k_dd(rows, r, b, nb) + 1u];
  const float sc = dv * (float)(sm & 63u), mm = mv * (float)(sm >> 8);
  const float xa = da * (float)(int)asb;
  return __builtin_fmaf(-mm, xa, __builtin_fmaf(sc * (float)dq, da, acc));
}
/* acc after block b of row r of an expert matrix of the given kind against
 * the Q8 activations x, in the arithmetic and order of the row functions. */
static inline float gexp_block(float acc, u32 kind, const G u32 *qw, const G u16 *qs, const G u8 *x, u32 rows, u32 nb, u32 r, u32 b) {
  const G u32 *aw = (const G u32 *)x;
  const float da = ((const G float *)(x + 32u * nb))[b];
  const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u);
  const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
  const u32 asb = (kind == 0u || kind == 3u || kind == 5u) ? ((const G u32 *)(x + 36u * nb))[b] : 0u;
  return gexp_dot(acc, kind, qw, qs, rows, nb, r, b, av, da, asb);
}
/* the same, the activations in LDS */
static inline float gexp_block_l(float acc, u32 kind, const G u32 *qw, const G u16 *qs, const LDSP u8 *x, u32 rows, u32 nb, u32 r, u32 b) {
  const LDSP u32 *aw = (const LDSP u32 *)x;
  const float da = ((const LDSP float *)(x + 32u * nb))[b];
  const u32x4 a0 = *(const LDSP u32x4 *)(aw + b * 8u), a1 = *(const LDSP u32x4 *)(aw + b * 8u + 4u);
  const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
  const u32 asb = (kind == 0u || kind == 3u || kind == 5u) ? ((const LDSP u32 *)(x + 36u * nb))[b] : 0u;
  return gexp_dot(acc, kind, qw, qs, rows, nb, r, b, av, da, asb);
}

/* nb <= 16 (the expert down matrices: 512 columns): two rows per wave, one
 * per half (lane & 15: the block); 16 rows per workgroup. Each row's value
 * is that of the row functions (one block per lane, the same tree). */
static inline void gexp_half_rows(u32 g, u32 kind, const G u32 *qw, const G u16 *qs, const G u8 *x, G float *y, u32 rows, u32 nb) {
  const u32 l = lane(), r = (g * 8u + (tid() >> 5)) * 2u + (l >> 4), b = l & 15u;
  float v = 0.0f;
  if (r < rows && b < nb) v = gexp_block(0.0f, kind, qw, qs, x, rows, nb, r, b);
  v = half_tree_f(v);
  if ((l & 15u) == 0u && r < rows) y[r] = v;
}

/* 16 < nb <= 64 (the expert gate/up matrices: 2048 columns): two rows per
 * wave (16 per workgroup), lane l the blocks l, l + 32 of both; each row's
 * value as the row functions (the same terms, order and tree). */
static inline void gexp_two_rows(u32 g, u32 kind, const G u32 *qw, const G u16 *qs, const G u8 *x, G float *y, u32 rows, u32 nb) {
  const u32 l = lane(), r0 = (g * 8u + (tid() >> 5)) * 2u;
  if (r0 >= rows) return;
  const u32 r1 = r0 + 1u < rows ? r0 + 1u : r0;
  float a0 = 0.0f, a1 = 0.0f;
  for (u32 t = 0; t < 2u; t++) {
    const u32 b = ie_lane_blk(l, t);
    if (b < nb) {
      a0 = gexp_block(a0, kind, qw, qs, x, rows, nb, r0, b);
      a1 = gexp_block(a1, kind, qw, qs, x, rows, nb, r1, b);
    }
  }
  a0 = wave_tree_f(a0), a1 = wave_tree_f(a1);
  if (l == 0u) {
    y[r0] = a0;
    if (r0 + 1u < rows) y[r0 + 1u] = a1;
  }
}

KERNEL ie_gexp(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, const G u32 *route, u32 qwn,
               u32 qsn, u32 astr, u32 bstr, u32 kind, u32 two_off) {
  const u32 s = tokid(), e = route[s];
  const G u32 *w = qw + (unsigned long)e * qwn;
  const G u16 *q = qs + (unsigned long)e * qsn;
  const G u8 *x = xq + (unsigned long)s * astr;
  G float *o = (G float *)((G u8 *)y + (unsigned long)s * bstr);
  if (nb <= 16u) gexp_half_rows(wgid(), kind, w, q, x, o, rows, nb);
  else if (nb <= 64u && !two_off) gexp_two_rows(wgid(), kind, w, q, x, o, rows, nb);
  else if (kind == 0u) gemv_q4q8_rows(wgid(), w, (const G f16 *)q, x, o, rows, nb, 0, 0);
  else if (kind == 2u) gemv_q8q8_rows(wgid(), w, (const G f16 *)q, x, o, rows, nb, 0, 0);
  else if (kind == 3u) gemv_q4kq8_rows(wgid(), w, q, x, o, rows, nb, 0, 0);
  else if (kind == 4u) gemv_q6kq8_rows(wgid(), w, q, x, o, rows, nb, 0, 0);
  else if (kind == 5u) gemv_q5kq8_rows(wgid(), w, q, x, o, rows, nb, 0, 0);
  else if (kind == 8u) gemv_iq4nlq8_rows(wgid(), w, (const G f16 *)q, x, o, rows, nb, 0, 0);
  else gemv_iq4xsq8_rows(wgid(), w, q, x, o, rows, nb, 0, 0);
}

/* Prompt chunks: Y = W X for many tokens, by groups (the experts) or one
 * matrix. Group e = grid y: the pairs list[offs[e] .. offs[e + 1]) (list
 * NULL: one group, the pairs 0 .. np - 1); pair p reads the Q8 row
 * X + (p / xdiv) xs and writes the row Y + p ys (bytes), plus the row
 * res + p rs when res. Workgroup (grid x): GXT_RB rows; the pairs GXT_P at a
 * time, their Q8 rows copied to LDS once (each row of the workgroup reads
 * them from there). Per (row, pair) the arithmetic and order of ie_gexp
 * (nb <= 16: the half-wave rows) or of the row functions (else: one row per
 * wave, blocks l + 32 t): the values of decode. nb <= GXT_NBMAX. */
#define GXT_P 8u
#define GXT_RB 64u
#ifndef GXT_W
#define GXT_W 4u /* rows of a wave at a time */
#endif
#define GXT_NBMAX 128u
static LDS u8 gxt_x[GXT_P * GXT_NBMAX * 40u];
KERNEL ie_gexp_t(const G u32 *qw, const G u16 *qs, const G u8 *X, G u8 *Y, u32 rows, u32 nb, const G u32 *list,
                 const G u32 *offs, u32 np, u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv, u32 kind, const G u8 *res, u32 rs) {
  const u32 e = tokid(), l = lane(), wv = tid() >> 5, half = nb <= 16u, xb = 40u * nb, rb = wgid() * GXT_RB;
  const u32 p0 = list ? offs[e] : 0u, p1 = list ? offs[e + 1u] : np;
  if (p0 >= p1 || rb >= rows) return; /* the whole workgroup */
  u32 xso_ = xs, yso_ = ys, rso_ = rs;
  __asm__ volatile("" : "+v"(xso_), "+v"(yso_), "+v"(rso_));
  const G u32 *w = qw + (unsigned long)e * qwn;
  const G u16 *q = qs + (unsigned long)e * qsn;
  for (u32 pb = p0; pb < p1; pb += GXT_P) {
    const u32 nj = p1 - pb < GXT_P ? p1 - pb : GXT_P;
    u32 pid[GXT_P];
#pragma unroll
    for (u32 j = 0; j < GXT_P; j++) {
      const u32 pj = pb + (j < nj ? j : 0u);
      pid[j] = list ? list[pj] : pj;
    }
    barrier(); /* the previous pairs are done with gxt_x */
    for (u32 i = tid() * 16u; i < nj * xb; i += NT * 16u) {
      const u32 j = i / xb, o = i - j * xb;
      u32 pj = pid[0];
#pragma unroll
      for (u32 k = 1; k < GXT_P; k++) pj = j == k ? pid[k] : pj;
      *(LDSP u32x4 *)(gxt_x + i) = *(const G u32x4 *)(X + (unsigned long)(pj / xdiv) * xso_ + o);
    }
    barrier();
    /* GXT_W rows of a wave at a time (their weight loads in flight together) */
    const u32 nit = half ? GXT_RB / 16u : GXT_RB / 8u;
    for (u32 it = 0; it < nit; it += GXT_W) {
      u32 r[GXT_W];
#pragma unroll
      for (u32 k = 0; k < GXT_W; k++)
        r[k] = half ? rb + (wv + 8u * (it + k)) * 2u + (l >> 4) : rb + wv + 8u * (it + k);
      if ((half ? r[0] & ~1u : r[0]) >= rows) break; /* the whole wave */
      float acc[GXT_W][GXT_P];
#pragma unroll
      for (u32 k = 0; k < GXT_W; k++)
#pragma unroll
        for (u32 j = 0; j < GXT_P; j++) acc[k][j] = 0.0f;
      if (half) {
        const u32 b = l & 15u;
        if (b < nb)
#pragma unroll
          for (u32 k = 0; k < GXT_W; k++)
            if (r[k] < rows)
#pragma unroll
              for (u32 j = 0; j < GXT_P; j++)
                if (j < nj) acc[k][j] = gexp_block_l(0.0f, kind, w, q, gxt_x + j * xb, rows, nb, r[k], b);
#pragma unroll
        for (u32 k = 0; k < GXT_W; k++)
#pragma unroll
          for (u32 j = 0; j < GXT_P; j++) acc[k][j] = half_tree_f(acc[k][j]);
      } else {
        for (u32 t = 0; ie_lane_blk(l, t) < nb; t++)
#pragma unroll
          for (u32 k = 0; k < GXT_W; k++)
            if (r[k] < rows)
#pragma unroll
              for (u32 j = 0; j < GXT_P; j++)
                if (j < nj) acc[k][j] = gexp_block_l(acc[k][j], kind, w, q, gxt_x + j * xb, rows, nb, r[k], ie_lane_blk(l, t));
#pragma unroll
        for (u32 k = 0; k < GXT_W; k++)
#pragma unroll
          for (u32 j = 0; j < GXT_P; j++) acc[k][j] = wave_tree_f(acc[k][j]);
      }
#pragma unroll
      for (u32 k = 0; k < GXT_W; k++)
        if ((l & (half ? 15u : 31u)) == 0u && r[k] < rows)
#pragma unroll
          for (u32 j = 0; j < GXT_P; j++)
            if (j < nj) {
              const G float *rr = res ? (const G float *)(res + (unsigned long)pid[j] * rso_) : 0;
              ((G float *)(Y + (unsigned long)pid[j] * yso_))[r[k]] = epilogue(acc[k][j], r[k], 0, rr);
            }
    }
  }
}

/* The pairs of a prompt chunk by expert: pair p = t K + k (token t, choice
 * k) with expert route_t[k] (route_t = route + t rst bytes). offs[e] .. offs[e
 * + 1]: the list entries of expert e, in pair order (one workgroup: the
 * counts in LDS, then thread 0 places the pairs in order from LDS). */
#define MS_MAXP 8192u /* pairs per chunk (src/hip.c checks) */
static LDS u32 ms_cnt[1024];
static LDS unsigned short ms_id[MS_MAXP];
KERNEL ie_moe_sort(const G u32 *route, u32 rst, u32 T, u32 K, u32 n_exp, G u32 *offs, G u32 *list) {
  const u32 t = tid(), np = T * K;
  for (u32 e = t; e < n_exp; e += NT) ms_cnt[e] = 0u;
  barrier();
  for (u32 p = t; p < np; p += NT) {
    const u32 e = ((const G u32 *)((const G u8 *)route + (unsigned long)(p / K) * rst))[p % K];
    ms_id[p] = (unsigned short)e;
    __atomic_fetch_add(&ms_cnt[e], 1u, __ATOMIC_RELAXED);
  }
  barrier();
  if (t == 0u) { /* exclusive prefix sum, then the pairs in order (stable), from LDS */
    u32 a = 0u;
    for (u32 e = 0; e < n_exp; e++) {
      const u32 c = ms_cnt[e];
      offs[e] = a, ms_cnt[e] = a, a += c;
    }
    offs[n_exp] = a;
    for (u32 p = 0; p < np; p++) list[ms_cnt[ms_id[p]]++] = p;
  }
}

/* Routing (wave 0 alone, no LDS, few registers: the megakernel; n <= 1024
 * experts, K <= 16 <= 32): s = sigmoid(logit); the K largest s + bias, one
 * at a time (ties: the smaller index); out = [K ids | K weights], weight k =
 * s_k / (sum of the K s + 1e-20) * scale, the sum in the order of the choice
 * (as moe_route in cpu.c). Lane l owns the experts l + 32 j (bit j of its
 * used mask); each round computes their sigmoids again; lane k keeps choice k. */
/* The wave maximum (any sign; v of every lane), in every lane: DPP inside
 * each row of 16 (a lane past the row keeps its own value), then the two rows. */
static inline float wave_max_any(float v) {
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(F2I(v), F2I(v), 0x108, 0xF, 0xF, false)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(F2I(v), F2I(v), 0x104, 0xF, 0xF, false)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(F2I(v), F2I(v), 0x102, 0xF, 0xF, false)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(F2I(v), F2I(v), 0x101, 0xF, 0xF, false)));
  return __builtin_fmaxf(I2F(__builtin_amdgcn_readlane(F2I(v), 0)), I2F(__builtin_amdgcn_readlane(F2I(v), 16)));
}
static inline u32 wave_min_u(u32 v) {
  u32 t;
  t = (u32)__builtin_amdgcn_update_dpp((int)v, (int)v, 0x108, 0xF, 0xF, false), v = t < v ? t : v;
  t = (u32)__builtin_amdgcn_update_dpp((int)v, (int)v, 0x104, 0xF, 0xF, false), v = t < v ? t : v;
  t = (u32)__builtin_amdgcn_update_dpp((int)v, (int)v, 0x102, 0xF, 0xF, false), v = t < v ? t : v;
  t = (u32)__builtin_amdgcn_update_dpp((int)v, (int)v, 0x101, 0xF, 0xF, false), v = t < v ? t : v;
  const u32 a = (u32)__builtin_amdgcn_readlane((int)v, 0), b = (u32)__builtin_amdgcn_readlane((int)v, 16);
  return a < b ? a : b;
}
/* ie_moe_route for n <= 256: lane l keeps the sigmoids (and choice values) of
 * its 8 experts l + 32 j; per round the largest choice value (wave_max_any),
 * then the smallest index that has it (wave_min_u): the same choice and the
 * same sum order as the general form. */
static inline void moe_route8(const G float *lg, u32 n, u32 K, const G float *bias, float scale, G u32 *out) {
  const u32 l = lane();
  float sg[8], c[8];
  for (u32 j = 0; j < 8u; j++) {
    const u32 i = l + 32u * j;
    sg[j] = i < n ? 1.0f / (1.0f + __builtin_expf(-lg[i])) : 0.0f;
    c[j] = i < n ? (bias ? sg[j] + bias[i] : sg[j]) : -__builtin_inff();
  }
  u32 myid = 0u;
  float myw = 0.0f, sum = 0.0f;
  for (u32 k = 0; k < K; k++) {
    float bv = -__builtin_inff();
    for (u32 j = 0; j < 8u; j++) bv = __builtin_fmaxf(bv, c[j]);
    const float M = wave_max_any(bv);
    u32 bi = 0xFFFFFFFFu;
    for (u32 j = 8u; j-- > 0u;) /* the smallest index of this lane with the maximum */
      if (c[j] == M) bi = l + 32u * j;
    bi = wave_min_u(bi);
    const u32 own = bi & 31u, jj = bi >> 5;
    float sb = 0.0f;
    for (u32 j = 0; j < 8u; j++)
      if (j == jj) {
        sb = sg[j];
        if (l == own) c[j] = -__builtin_inff();
      }
    sb = I2F(__builtin_amdgcn_readlane(F2I(sb), __builtin_amdgcn_readfirstlane(own)));
    if (l == k) myid = bi, myw = sb;
    sum += sb;
  }
  const float inv = 1.0f / (sum + 1e-20f);
  if (l < K) out[l] = myid, ((G float *)out)[K + l] = myw * inv * scale;
}
KERNEL ie_moe_route(const G float *lg, u32 n, u32 K, const G float *bias, float scale, G u32 *out, u32 ls, u32 ost) {
  if (tid() >= 32u) return;
  lg = TOKC(lg, ls), out = TOK(out, ost); /* token tokid() (grid y) */
  if (n <= 256u) {
    moe_route8(lg, n, K, bias, scale, out);
    return;
  }
  const u32 l = lane(), nj = (n + 31u) >> 5;
  u32 used = 0u, myid = 0u;
  float myw = 0.0f, sum = 0.0f;
  for (u32 k = 0; k < K; k++) {
    float bv = -__builtin_inff(), bs = 0.0f;
    u32 bi = 0xFFFFFFFFu;
    for (u32 j = 0; j < nj; j++) { /* the lane's best: its smallest index first */
      const u32 i = l + 32u * j;
      if (i < n && !((used >> j) & 1u)) {
        const float sg = 1.0f / (1.0f + __builtin_expf(-lg[i])), c = bias ? sg + bias[i] : sg;
        if (better(c, i, bv, bi)) bv = c, bi = i, bs = sg;
      }
    }
    for (u32 m = 16u; m >= 1u; m >>= 1) { /* the wave's best (ties: the smaller index) */
      const float ov = shfl_xor_f(bv, m), os = shfl_xor_f(bs, m);
      const u32 oi = (u32)__builtin_amdgcn_ds_bpermute((int)((l ^ m) << 2), (int)bi);
      if (better(ov, oi, bv, bi)) bv = ov, bi = oi, bs = os;
    }
    if ((bi & 31u) == l) used |= 1u << (bi >> 5);
    if (l == k) myid = bi, myw = bs;
    sum += bs;
  }
  const float inv = 1.0f / (sum + 1e-20f);
  if (l < K) out[l] = myid, ((G float *)out)[K + l] = myw * inv * scale;
}

/* out = res + sum over slots k of weight_k y_k (fma, slot order). */
/* Decode attention, grouped (flint): workgroup kh * nsplit + s does the
 * G = n_head / n_kv q heads of kv head kh (G <= 8, hd <= 128, hd % 8 == 0)
 * over the positions p0 + s ch .. (n <= ch <= 256 of them; K/V of position p
 * in row p % len). One thread per position for the scores (its K row read
 * once for the G heads, 64 dims of loads together), one wave per head for
 * the softmax; P V: one thread per dim and position parity, all G heads. One split: the output directly (sinks as ie_attn_sw);
 * else partial results (M, L, o) per q head and the last workgroup of kv
 * head kh merges, with the arithmetic of the ie_attn_sw merge. */
static LDS float ag_q[8][128]; /* q, then the P V sums of the odd positions */
static LDS float ag_c[8];      /* per pass: the rescale of each head's O and sum */
typedef kvt ag_kv8 __attribute__((ext_vector_type(8)));
/* The value of v in a register here: the loads before it, one wait for all
 * of them (the compiler would wait for each load before its use). */
#define AG_HOLD(v) __asm__ volatile("" : "+v"(v))
#define AG_QH (*(LDSP f16(*)[8][128]) & ag_q[0][0])       /* q in f16: hi (fp16 KV) */
#define AG_QL (*(LDSP f16(*)[8][128]) & ag_q[4][0])       /* and lo */
_Static_assert(NT * 4u == 8u * 128u, "ie_attn_gq: 4 q values per thread");
#define AG_SC at_o /* the scores, then the weights: [head][position] */
KERNEL_CALL ie_attn_gq(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 p0, u32 np,
                  u32 len, u32 hd, u32 n_head, u32 n_kv, u32 nsplit, u32 ch, G u32 *count, G u8 *oq, const G float *sinks) {
  const u32 kh = wgid() / nsplit, s = wgid() % nsplit, ng = n_head / n_kv, kvd = n_kv * hd, t = tid();
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
#ifndef IE_KV_F32
  /* the scores with WMMA (f16 x f16 -> f32): S^T = Q K^T, Q rows = the heads
   * (q = hi + lo, two f16, so the products are those of q in f32 to about
   * 2^-22), K columns = 16 positions. Lane layout as ie_attn_fa: A lane l
   * head l % 16, dims 8 (l / 16) .. + 7 of the 16 of a step; B lane l the
   * position l % 16, the same dims; D lane l, v: head 8 (l / 16) + v. */
#pragma unroll
  for (u32 x = 0; x < 4u; x++) {
    const u32 i = t + NT * x, ok = (i >> 7) < ng && (i & 127u) < hd;
    const float qv = q[ok ? (kh * ng + (i >> 7)) * hd + (i & 127u) : 0u];
    const f16 hi = ok ? (f16)qv : (f16)0.0f;
    AG_QH[i >> 7][i & 127u] = hi, AG_QL[i >> 7][i & 127u] = ok ? (f16)(qv - (float)hi) : (f16)0.0f;
  }
  barrier();
  /* the chunk of this split (na <= ch positions) in passes of 256: the
   * running maximum and sum per head (wave w < ng), O rescaled per pass */
  const u32 a0 = s * ch, na = a0 + ch < np ? ch : np - a0;
  float o[8], Mr = -__builtin_inff(), Lr = 0.0f;
  for (u32 g = 0; g < 8u; g++) o[g] = 0.0f;
  for (u32 sb = 0; sb < na; sb += 256u) {
  const u32 n = na - sb < 256u ? na - sb : 256u, r0 = (p0 + a0 + sb) % len;
  {
    const u32 l = lane(), rl = l & 15u, hl = l >> 4;
    const v8f zero = (v8f){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const h8 hz = (h8){0, 0, 0, 0, 0, 0, 0, 0};
    for (u32 tb = 16u * (t >> 5); tb < n; tb += 128u) { /* 16 positions per wave */
      const u32 pi = tb + rl < n ? tb + rl : 0u, r = r0 + pi, slot = r < len ? r : r - len;
      const G kvt *k = kc + (unsigned long)slot * kvd + kh * hd;
      h8 kb[8];
#pragma unroll
      for (u32 st = 0; st < 8u; st++) { /* past hd: dim 0 (q is 0 there) */
        const u32 c = 16u * st + 8u * hl;
        kb[st] = *(const G h8 *)(k + (c < hd ? c : 0u));
      }
#pragma unroll
      for (u32 st = 0; st < 8u; st++) AG_HOLD(kb[st]);
      v8f acc = zero;
#pragma unroll
      for (u32 st = 0; st < 8u; st++) {
        if (16u * st >= hd) break;
        const u32 c = 16u * st + 8u * hl;
        const h8 ah = rl < 8u ? *(const LDSP h8 *)(&AG_QH[rl & 7u][c & 127u]) : hz;
        const h8 al = rl < 8u ? *(const LDSP h8 *)(&AG_QL[rl & 7u][c & 127u]) : hz;
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, kb[st], acc);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(al, kb[st], acc);
      }
      if (hl == 0u && tb + rl < n)
#pragma unroll
        for (u32 v = 0; v < 8u; v++)
          if (v < ng) AG_SC[v][tb + rl] = acc[v] * scale;
    }
  }
#else
  { /* zero past the heads and dims: the score loop has constant bounds */
    float qq[4];
#pragma unroll
    for (u32 x = 0; x < 4u; x++) {
      const u32 i = t + NT * x, ok = (i >> 7) < ng && (i & 127u) < hd;
      qq[x] = q[ok ? (kh * ng + (i >> 7)) * hd + (i & 127u) : 0u];
    }
#pragma unroll
    for (u32 x = 0; x < 4u; x++) AG_HOLD(qq[x]);
#pragma unroll
    for (u32 x = 0; x < 4u; x++) {
      const u32 i = t + NT * x;
      ag_q[i >> 7][i & 127u] = (i >> 7) < ng && (i & 127u) < hd ? qq[x] : 0.0f;
    }
  }
  barrier();
  /* the chunk of this split (na <= ch positions) in passes of 256: the
   * running maximum and sum per head (wave w < ng), O rescaled per pass */
  const u32 a0 = s * ch, na = a0 + ch < np ? ch : np - a0;
  float o[8], Mr = -__builtin_inff(), Lr = 0.0f;
  for (u32 g = 0; g < 8u; g++) o[g] = 0.0f;
  for (u32 sb = 0; sb < na; sb += 256u) {
  const u32 n = na - sb < 256u ? na - sb : 256u, r0 = (p0 + a0 + sb) % len;
  if (t < n) {
    const u32 slot = r0 + t < len ? r0 + t : r0 + t - len;
    const G kvt *k = kc + (unsigned long)slot * kvd + kh * hd;
    float d[8];
    for (u32 g = 0; g < 8u; g++) d[g] = 0.0f;
    for (u32 c0 = 0; c0 < 128u; c0 += 64u) { /* 64 dims of the K row in registers: the loads together */
      ag_kv8 kk[8];
#pragma unroll
      for (u32 x = 0; x < 8u; x++) /* past hd: a valid row (dim 0), q is 0 there */
        kk[x] = *(const G ag_kv8 *)(k + (c0 + 8u * x < hd ? c0 + 8u * x : 0u));
#pragma unroll
      for (u32 x = 0; x < 8u; x++) AG_HOLD(kk[x]);
#pragma unroll
      for (u32 x = 0; x < 8u; x++)
#pragma unroll
        for (u32 g = 0; g < 8u; g++)
#pragma unroll
          for (u32 e = 0; e < 8u; e++) d[g] += ag_q[g][c0 + 8u * x + e] * (float)kk[x][e];
      if (c0 + 64u >= hd) break;
    }
    for (u32 g = 0; g < 8u; g++)
      if (g < ng) AG_SC[g][t] = d[g] * scale;
  }
#endif
  barrier();
  const u32 w = t >> 5, l = lane();
  if (w < ng) { /* one pass (na <= 256): M = the maximum, c = 0, Lr = the sum, as before */
    float mx = -__builtin_inff();
    for (u32 i = l; i < n; i += 32u) mx = __builtin_fmaxf(mx, AG_SC[w][i]);
    mx = wave_max_any(mx);
    const float M = __builtin_fmaxf(Mr, mx), c = __builtin_expf(Mr - M);
    float sm = 0.0f;
    for (u32 i = l; i < n; i += 32u) {
      const float e = __builtin_expf(AG_SC[w][i] - M);
      AG_SC[w][i] = e;
      sm += e;
    }
    sm = wave_sum_all(sm);
    Lr = Lr * c + sm, Mr = M;
    if (l == 0u) ag_c[w] = c;
  }
  barrier();
  /* P V: thread t the dim t & 127 of all heads, positions of parity t >> 7,
   * 16 V loads together; then the odd sums through ag_q */
  const u32 dd = t & 127u, par = t >> 7;
  if (sb)
    for (u32 g = 0; g < 8u; g++)
      if (g < ng) o[g] *= ag_c[g];
  if (dd < hd) {
    const G kvt *v = vc + kh * hd + dd;
    for (u32 i0 = par; i0 < n; i0 += 32u) {
      kvt vh[16];
#pragma unroll
      for (u32 x = 0; x < 16u; x++) { /* past n: a valid row (r0), weight 0 */
        const u32 i = i0 + 2u * x, r = r0 + (i < n ? i : 0u), slot = r < len ? r : r - len;
        vh[x] = v[(unsigned long)slot * kvd];
      }
#pragma unroll
      for (u32 x = 0; x < 16u; x++) AG_HOLD(vh[x]);
#pragma unroll
      for (u32 x = 0; x < 16u; x++) {
        const u32 i = i0 + 2u * x, ic = i < n ? i : 0u;
        const float vv = i < n ? (float)vh[x] : 0.0f;
#pragma unroll
        for (u32 g = 0; g < 8u; g++)
          if (g < ng) o[g] += AG_SC[g][ic] * vv;
      }
    }
  }
  if (sb + 256u < na) barrier(); /* AG_SC and ag_c: the next pass (the last: the barrier below) */
  }
  if (t < 8u * 32u && (t >> 5) < ng && lane() == 0u) at_m[t >> 5] = Mr, at_l[t >> 5] = Lr;
  const u32 dd = t & 127u, par = t >> 7;
  barrier(); /* every thread is done with ag_q (the scores) */
  if (par == 1u && dd < hd)
    for (u32 g = 0; g < 8u; g++)
      if (g < ng) ag_q[g][dd] = o[g];
  barrier();
  if (par == 1u) { /* the merge below: every thread meets at its barriers */
    if (nsplit == 1u) return;
  } else if (dd < hd)
    for (u32 g = 0; g < 8u; g++)
      if (g < ng) o[g] += ag_q[g][dd];
  if (nsplit == 1u) {
    for (u32 g = 0; g < 8u; g++) { /* constant bounds: o in registers */
      if (g >= ng) break;
      const u32 h = kh * ng + g;
      const float M = at_m[g], sk = sinks ? sinks[h] : -__builtin_inff(), M2 = __builtin_fmaxf(sk, M);
      const float wt = __builtin_expf(M - M2);
      float L2 = 0.0f + at_l[g] * wt;
      if (sinks) L2 += __builtin_expf(sk - M2);
      if (dd < hd) {
        const float r = (0.0f + o[g] * wt) / L2;
        out[h * hd + dd] = r;
        if (oq) quant_wave(r, (h * hd + dd) >> 5, oq, (n_head * hd) >> 5);
      }
    }
    return;
  }
  if (par == 0u)
    for (u32 g = 0; g < 8u; g++) { /* constant bounds: o in registers */
      if (g >= ng) break;
      const u32 h = kh * ng + g;
      G float *pp = part + (unsigned long)(h * nsplit + s) * (hd + 2u);
      if (dd < hd) pp[2u + dd] = o[g];
      if (dd == 0u) pp[0] = at_m[g], pp[1] = at_l[g];
    }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (t == 0u) at_last = __scoped_atomic_fetch_add(&count[kh], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u;
  barrier();
  if (!at_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  for (u32 g = 0; g < ng; g++) {
    const u32 h = kh * ng + g;
    G float *ph = part + (unsigned long)h * nsplit * (hd + 2u);
    const float sk = sinks ? sinks[h] : -__builtin_inff();
    float mv = sk;
    for (u32 x = t; x < nsplit; x += NT) mv = __builtin_fmaxf(mv, ph[x * (hd + 2u)]);
    const float M2 = wg_max(mv);
    float lv = 0.0f;
    for (u32 x = t; x < nsplit; x += NT) {
      const float wt = __builtin_expf(ph[x * (hd + 2u)] - M2);
      ph[x * (hd + 2u)] = wt;
      lv += ph[x * (hd + 2u) + 1u] * wt;
    }
    float L2 = wg_sum(lv);
    if (sinks) L2 += __builtin_expf(sk - M2);
    if (t < hd) {
      float r = 0.0f;
      for (u32 x = 0; x < nsplit; x++) r += ph[x * (hd + 2u) + 2u + t] * ph[x * (hd + 2u)];
      r = r / L2;
      out[h * hd + t] = r;
      if (oq) quant_wave(r, (h * hd + t) >> 5, oq, (n_head * hd) >> 5);
    }
  }
  if (t == 0u) count[kh] = 0u;
}

KERNEL ie_moe_sum(const G float *y, G float *out, const G u32 *route, const G float *res, u32 n, u32 K, u32 ys, u32 os,
                  u32 rts, u32 rs) {
  const u32 j = wgid() * NT + tid();
  if (j >= n) return;
  /* token tokid(): the strides in bytes (grid y; decode: one token) */
  y = TOKC(y, ys), out = TOK(out, os), route = TOKC(route, rts), res = res ? TOKC(res, rs) : res;
  const G float *w = (const G float *)(route + K);
  float acc = 0.0f;
  for (u32 k = 0; k < K; k++) acc = __builtin_fmaf(w[k], y[(unsigned long)k * n + j], acc);
  out[j] = res ? res[j] + acc : acc;
}

/* As ie_qkn_rope_kv, for heads of qstr floats (flint: q heads of hd, no
 * gate): q head h at qkv + h qstr, key head h at qkv + nq qstr + h hd, value
 * head h at qkv + nq qstr + nkv hd + h hd. nrot = 0: no RoPE. kc, vc: the
 * cache rows of this position. One token. */
KERNEL ie_qkn_rope_kv2(const G float *qkv, G float *qo, G kvt *kc, G kvt *vc, const G float *qw, const G float *kw,
                       const G float *cs, const G float *sn, u32 nq, u32 nkv, u32 hd, u32 nrot, float eps, u32 qstr,
                       u32 ins, u32 qos, u32 pos0, u32 len, u32 h2r) {
  /* token tokid() of a chunk (grid y): position pos0 + tokid(); kc, vc: row 0
   * of the layer (its K/V in row p % len); cs, sn: the tables (h2r per position) */
  const u32 g = wgid(), t = tid(), pp = pos0 + tokid();
  qkv = TOKC(qkv, ins), qo = TOK(qo, qos);
  kc += (unsigned long)(pp % len) * nkv * hd, vc += (unsigned long)(pp % len) * nkv * hd;
  cs += (unsigned long)pp * h2r, sn += (unsigned long)pp * h2r;
  const G float *kb = qkv + nq * qstr;
  if (g >= nq + nkv) { /* value head */
    const u32 h = g - nq - nkv;
    if (t < hd) vc[h * hd + t] = kb[nkv * hd + h * hd + t];
    return;
  }
  const int is_k = g >= nq;
  const u32 h = is_k ? g - nq : g;
  const G float *src = is_k ? kb + h * hd : qkv + h * qstr;
  const G float *w = is_k ? kw : qw;
  const float x = t < hd ? src[t] : 0.0f;
  const float ss = wg_sum(x * x);
  const float y = x * (1.0f / __builtin_sqrtf(ss / (float)hd + eps)) * (t < hd ? w[t] : 0.0f);
  if (t < hd) qk_buf[t] = y;
  barrier();
  float r = y;
  const u32 h2 = nrot >> 1;
  if (t < nrot) {
    const u32 i = t < h2 ? t : t - h2;
    const float a = qk_buf[i], b = qk_buf[i + h2];
    r = t < h2 ? a * cs[i] - b * sn[i] : a * sn[i] + b * cs[i];
  }
  if (t < hd) {
    if (is_k) kc[h * hd + t] = (kvt)r;
    else qo[h * hd + t] = r;
  }
}

/* Decode attention over the np positions p0 .. p0 + np - 1 (a window), the
 * K/V of position p in row p % len of kc, vc (a ring, or len >= all
 * positions). As ie_attn_split: workgroup h * nsplit + s does the positions
 * p0 + s ch .. (ch of them) with 8 waves, the last workgroup of head h
 * merges. sinks (or NULL): logit sinks[h] in the softmax denominator only. */
KERNEL ie_attn_sw(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 p0, u32 np,
                  u32 len, u32 hd, u32 n_head, u32 n_kv, u32 nsplit, u32 ch, G u32 *count, G u8 *oq, const G float *sinks) {
  const u32 h = wgid() / nsplit, s = wgid() % nsplit;
  const u32 kh = h / (n_head / n_kv), kvd = n_kv * hd;
  const u32 w = tid() >> 5, l = lane();
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  float qv[8], acc[8];
  for (u32 j = 0; j < 8u; j++) {
    const u32 i = l + 32u * j;
    qv[j] = i < hd ? q[h * hd + i] : 0.0f;
    acc[j] = 0.0f;
  }
  float m = -__builtin_inff(), sum = 0.0f;
  const u32 a0 = s * ch, a1 = a0 + ch < np ? a0 + ch : np;
  for (u32 a = a0 + w; a < a1; a += 8u) {
    const u32 slot = (p0 + a) % len;
    const G kvt *k = kc + (unsigned long)slot * kvd + kh * hd;
    const G kvt *v = vc + (unsigned long)slot * kvd + kh * hd;
    float d = 0.0f;
    for (u32 j = 0; j < 8u; j++)
      if (l + 32u * j < hd) d += qv[j] * (float)k[l + 32u * j];
    d = wave_sum_all(d);
    d *= scale;
    const float mn = __builtin_fmaxf(m, d), c = __builtin_expf(m - mn), e = __builtin_expf(d - mn);
    sum = sum * c + e;
    for (u32 j = 0; j < 8u; j++)
      if (l + 32u * j < hd) acc[j] = acc[j] * c + e * (float)v[l + 32u * j];
    m = mn;
  }
  if (l == 0u) at_m[w] = m, at_l[w] = sum;
  for (u32 j = 0; j < 8u; j++)
    if (l + 32u * j < hd) at_o[w][l + 32u * j] = acc[j];
  barrier();
  float M = at_m[0];
  for (u32 x = 1; x < 8u; x++) M = __builtin_fmaxf(M, at_m[x]);
  if (nsplit == 1u) { /* one split: the output directly (no partial results,
                       * no count), with the same arithmetic as the merge below */
    const float sk = sinks ? sinks[h] : -__builtin_inff(), M2 = __builtin_fmaxf(sk, M);
    const float wt = __builtin_expf(M - M2);
    float L = 0.0f;
    for (u32 x = 0; x < 8u; x++) L += at_m[x] == -__builtin_inff() ? 0.0f : at_l[x] * __builtin_expf(at_m[x] - M);
    float L2 = 0.0f + L * wt;
    if (sinks) L2 += __builtin_expf(sk - M2);
    if (tid() < hd) {
      float o = 0.0f;
      for (u32 x = 0; x < 8u; x++) o += at_m[x] == -__builtin_inff() ? 0.0f : at_o[x][tid()] * __builtin_expf(at_m[x] - M);
      o = (0.0f + o * wt) / L2;
      out[h * hd + tid()] = o;
      if (oq) quant_wave(o, (h * hd + tid()) >> 5, oq, (n_head * hd) >> 5);
    }
    return;
  }
  G float *pp = part + (unsigned long)(h * nsplit + s) * (hd + 2u);
  if (tid() < hd) {
    float o = 0.0f;
    for (u32 x = 0; x < 8u; x++) o += at_m[x] == -__builtin_inff() ? 0.0f : at_o[x][tid()] * __builtin_expf(at_m[x] - M);
    pp[2u + tid()] = o;
  }
  if (tid() == 0u) {
    float L = 0.0f;
    for (u32 x = 0; x < 8u; x++) L += at_m[x] == -__builtin_inff() ? 0.0f : at_l[x] * __builtin_expf(at_m[x] - M);
    pp[0] = M;
    pp[1] = L;
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) at_last = __scoped_atomic_fetch_add(&count[h], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u;
  barrier();
  if (!at_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  G float *ph = part + (unsigned long)h * nsplit * (hd + 2u);
  const float sk = sinks ? sinks[h] : -__builtin_inff();
  float mv = sk;
  for (u32 t = tid(); t < nsplit; t += NT) mv = __builtin_fmaxf(mv, ph[t * (hd + 2u)]);
  const float M2 = wg_max(mv);
  float lv = 0.0f;
  for (u32 t = tid(); t < nsplit; t += NT) {
    const float wt = __builtin_expf(ph[t * (hd + 2u)] - M2);
    ph[t * (hd + 2u)] = wt;
    lv += ph[t * (hd + 2u) + 1u] * wt;
  }
  float L2 = wg_sum(lv);
  if (sinks) L2 += __builtin_expf(sk - M2);
  if (tid() < hd) {
    float o = 0.0f;
    for (u32 t = 0; t < nsplit; t++) o += ph[t * (hd + 2u) + 2u + tid()] * ph[t * (hd + 2u)];
    o = o / L2;
    out[h * hd + tid()] = o;
    if (oq) quant_wave(o, (h * hd + tid()) >> 5, oq, (n_head * hd) >> 5);
  }
  if (tid() == 0u) count[h] = 0u;
}

/* Multi-token raw GEMV: the same row per wave, but each 256-weight block is
 * read once and dotted against up to 4 tokens' Q8 activations (weight bytes
 * are the prefill bound: without this every prompt token re-reads the whole
 * matrix). The integer work repeats per token - one grid expansion per token
 * next to one weight stream - and token t's result lands at y + t*ys (bytes).
 * No fused RMSNorm here: the caller keeps that as its own op. */
KERNEL_CALL ie_gemv_raw_q2k_tr(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                               const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                               float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  const u32 r = ie_gemv_row(wgid(), (u32)__builtin_amdgcn_readfirstlane(tid() >> 5)), l = lane();
  if (r >= rows || T == 0u) return;
  const u32 mb = nb / 8u, bs = ie_rq_bs(IE_RQ_Q2_K);
  const G u8 *row = q + (unsigned long)r * mb * bs;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
  for (u32 b = l; b < nb; b += 32u) { /* the row kernel's sub-block map: all 32 lanes busy */
    const u32 qb = b >> 3u, sb = b & 7u;
    const G u8 *blk = row + (unsigned long)qb * bs;
    if (T > 0u) a0 += ie_rq_dot_sub(IE_RQ_Q2_K, blk, xq, nb, qb, sb);
    if (T > 1u) a1 += ie_rq_dot_sub(IE_RQ_Q2_K, blk, xq + (unsigned long)xs, nb, qb, sb);
    if (T > 2u) a2 += ie_rq_dot_sub(IE_RQ_Q2_K, blk, xq + (unsigned long)2u * xs, nb, qb, sb);
    if (T > 3u) a3 += ie_rq_dot_sub(IE_RQ_Q2_K, blk, xq + (unsigned long)3u * xs, nb, qb, sb);
    if (T > 4u) a4 += ie_rq_dot_sub(IE_RQ_Q2_K, blk, xq + (unsigned long)4u * xs, nb, qb, sb);
    if (T > 5u) a5 += ie_rq_dot_sub(IE_RQ_Q2_K, blk, xq + (unsigned long)5u * xs, nb, qb, sb);
    if (T > 6u) a6 += ie_rq_dot_sub(IE_RQ_Q2_K, blk, xq + (unsigned long)6u * xs, nb, qb, sb);
    if (T > 7u) a7 += ie_rq_dot_sub(IE_RQ_Q2_K, blk, xq + (unsigned long)7u * xs, nb, qb, sb);
  }
  const float v0 = T > 0u ? wave_tree_f(a0) : 0.0f;
  const float v1 = T > 1u ? wave_tree_f(a1) : 0.0f;
  const float v2 = T > 2u ? wave_tree_f(a2) : 0.0f;
  const float v3 = T > 3u ? wave_tree_f(a3) : 0.0f;
  const float v4 = T > 4u ? wave_tree_f(a4) : 0.0f;
  const float v5 = T > 5u ? wave_tree_f(a5) : 0.0f;
  const float v6 = T > 6u ? wave_tree_f(a6) : 0.0f;
  const float v7 = T > 7u ? wave_tree_f(a7) : 0.0f;
  if (l == 0u) {
    G float *y0 = (G float *)((G u8 *)y);
    if (T > 0u) ((G float *)((G u8 *)y0 + 0ul * ys))[r] = epilogue(v0, r, bias, res ? (const G float *)((const G u8 *)res + 0ul * rs) : res);
    if (T > 1u) ((G float *)((G u8 *)y0 + ys))[r] = epilogue(v1, r, bias, res ? (const G float *)((const G u8 *)res + rs) : res);
    if (T > 2u) ((G float *)((G u8 *)y0 + 2ul * ys))[r] = epilogue(v2, r, bias, res ? (const G float *)((const G u8 *)res + 2ul * rs) : res);
    if (T > 3u) ((G float *)((G u8 *)y0 + 3ul * ys))[r] = epilogue(v3, r, bias, res ? (const G float *)((const G u8 *)res + 3ul * rs) : res);
    if (T > 4u) ((G float *)((G u8 *)y0 + 4ul * ys))[r] = epilogue(v4, r, bias, res ? (const G float *)((const G u8 *)res + 4ul * rs) : res);
    if (T > 5u) ((G float *)((G u8 *)y0 + 5ul * ys))[r] = epilogue(v5, r, bias, res ? (const G float *)((const G u8 *)res + 5ul * rs) : res);
    if (T > 6u) ((G float *)((G u8 *)y0 + 6ul * ys))[r] = epilogue(v6, r, bias, res ? (const G float *)((const G u8 *)res + 6ul * rs) : res);
    if (T > 7u) ((G float *)((G u8 *)y0 + 7ul * ys))[r] = epilogue(v7, r, bias, res ? (const G float *)((const G u8 *)res + 7ul * rs) : res);
  }
}
KERNEL_CALL ie_gemv_raw_iq2xxs_tr(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                               const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                               float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  const u32 r = ie_gemv_row(wgid(), (u32)__builtin_amdgcn_readfirstlane(tid() >> 5)), l = lane();
  if (r >= rows || T == 0u) return;
  const u32 mb = nb / 8u, bs = ie_rq_bs(IE_RQ_IQ2_XXS);
  const G u8 *row = q + (unsigned long)r * mb * bs;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
  for (u32 b = l; b < nb; b += 32u) { /* the row kernel's sub-block map: all 32 lanes busy */
    const u32 qb = b >> 3u, sb = b & 7u;
    const G u8 *blk = row + (unsigned long)qb * bs;
    if (T > 0u) a0 += ie_rq_dot_sub(IE_RQ_IQ2_XXS, blk, xq, nb, qb, sb);
    if (T > 1u) a1 += ie_rq_dot_sub(IE_RQ_IQ2_XXS, blk, xq + (unsigned long)xs, nb, qb, sb);
    if (T > 2u) a2 += ie_rq_dot_sub(IE_RQ_IQ2_XXS, blk, xq + (unsigned long)2u * xs, nb, qb, sb);
    if (T > 3u) a3 += ie_rq_dot_sub(IE_RQ_IQ2_XXS, blk, xq + (unsigned long)3u * xs, nb, qb, sb);
    if (T > 4u) a4 += ie_rq_dot_sub(IE_RQ_IQ2_XXS, blk, xq + (unsigned long)4u * xs, nb, qb, sb);
    if (T > 5u) a5 += ie_rq_dot_sub(IE_RQ_IQ2_XXS, blk, xq + (unsigned long)5u * xs, nb, qb, sb);
    if (T > 6u) a6 += ie_rq_dot_sub(IE_RQ_IQ2_XXS, blk, xq + (unsigned long)6u * xs, nb, qb, sb);
    if (T > 7u) a7 += ie_rq_dot_sub(IE_RQ_IQ2_XXS, blk, xq + (unsigned long)7u * xs, nb, qb, sb);
  }
  const float v0 = T > 0u ? wave_tree_f(a0) : 0.0f;
  const float v1 = T > 1u ? wave_tree_f(a1) : 0.0f;
  const float v2 = T > 2u ? wave_tree_f(a2) : 0.0f;
  const float v3 = T > 3u ? wave_tree_f(a3) : 0.0f;
  const float v4 = T > 4u ? wave_tree_f(a4) : 0.0f;
  const float v5 = T > 5u ? wave_tree_f(a5) : 0.0f;
  const float v6 = T > 6u ? wave_tree_f(a6) : 0.0f;
  const float v7 = T > 7u ? wave_tree_f(a7) : 0.0f;
  if (l == 0u) {
    G float *y0 = (G float *)((G u8 *)y);
    if (T > 0u) ((G float *)((G u8 *)y0 + 0ul * ys))[r] = epilogue(v0, r, bias, res ? (const G float *)((const G u8 *)res + 0ul * rs) : res);
    if (T > 1u) ((G float *)((G u8 *)y0 + ys))[r] = epilogue(v1, r, bias, res ? (const G float *)((const G u8 *)res + rs) : res);
    if (T > 2u) ((G float *)((G u8 *)y0 + 2ul * ys))[r] = epilogue(v2, r, bias, res ? (const G float *)((const G u8 *)res + 2ul * rs) : res);
    if (T > 3u) ((G float *)((G u8 *)y0 + 3ul * ys))[r] = epilogue(v3, r, bias, res ? (const G float *)((const G u8 *)res + 3ul * rs) : res);
    if (T > 4u) ((G float *)((G u8 *)y0 + 4ul * ys))[r] = epilogue(v4, r, bias, res ? (const G float *)((const G u8 *)res + 4ul * rs) : res);
    if (T > 5u) ((G float *)((G u8 *)y0 + 5ul * ys))[r] = epilogue(v5, r, bias, res ? (const G float *)((const G u8 *)res + 5ul * rs) : res);
    if (T > 6u) ((G float *)((G u8 *)y0 + 6ul * ys))[r] = epilogue(v6, r, bias, res ? (const G float *)((const G u8 *)res + 6ul * rs) : res);
    if (T > 7u) ((G float *)((G u8 *)y0 + 7ul * ys))[r] = epilogue(v7, r, bias, res ? (const G float *)((const G u8 *)res + 7ul * rs) : res);
  }
}
KERNEL_CALL ie_gemv_raw_iq2xs_tr(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                               const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                               float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  const u32 r = ie_gemv_row(wgid(), (u32)__builtin_amdgcn_readfirstlane(tid() >> 5)), l = lane();
  if (r >= rows || T == 0u) return;
  const u32 mb = nb / 8u, bs = ie_rq_bs(IE_RQ_IQ2_XS);
  const G u8 *row = q + (unsigned long)r * mb * bs;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
  for (u32 b = l; b < nb; b += 32u) { /* the row kernel's sub-block map: all 32 lanes busy */
    const u32 qb = b >> 3u, sb = b & 7u;
    const G u8 *blk = row + (unsigned long)qb * bs;
    if (T > 0u) a0 += ie_rq_dot_sub(IE_RQ_IQ2_XS, blk, xq, nb, qb, sb);
    if (T > 1u) a1 += ie_rq_dot_sub(IE_RQ_IQ2_XS, blk, xq + (unsigned long)xs, nb, qb, sb);
    if (T > 2u) a2 += ie_rq_dot_sub(IE_RQ_IQ2_XS, blk, xq + (unsigned long)2u * xs, nb, qb, sb);
    if (T > 3u) a3 += ie_rq_dot_sub(IE_RQ_IQ2_XS, blk, xq + (unsigned long)3u * xs, nb, qb, sb);
    if (T > 4u) a4 += ie_rq_dot_sub(IE_RQ_IQ2_XS, blk, xq + (unsigned long)4u * xs, nb, qb, sb);
    if (T > 5u) a5 += ie_rq_dot_sub(IE_RQ_IQ2_XS, blk, xq + (unsigned long)5u * xs, nb, qb, sb);
    if (T > 6u) a6 += ie_rq_dot_sub(IE_RQ_IQ2_XS, blk, xq + (unsigned long)6u * xs, nb, qb, sb);
    if (T > 7u) a7 += ie_rq_dot_sub(IE_RQ_IQ2_XS, blk, xq + (unsigned long)7u * xs, nb, qb, sb);
  }
  const float v0 = T > 0u ? wave_tree_f(a0) : 0.0f;
  const float v1 = T > 1u ? wave_tree_f(a1) : 0.0f;
  const float v2 = T > 2u ? wave_tree_f(a2) : 0.0f;
  const float v3 = T > 3u ? wave_tree_f(a3) : 0.0f;
  const float v4 = T > 4u ? wave_tree_f(a4) : 0.0f;
  const float v5 = T > 5u ? wave_tree_f(a5) : 0.0f;
  const float v6 = T > 6u ? wave_tree_f(a6) : 0.0f;
  const float v7 = T > 7u ? wave_tree_f(a7) : 0.0f;
  if (l == 0u) {
    G float *y0 = (G float *)((G u8 *)y);
    if (T > 0u) ((G float *)((G u8 *)y0 + 0ul * ys))[r] = epilogue(v0, r, bias, res ? (const G float *)((const G u8 *)res + 0ul * rs) : res);
    if (T > 1u) ((G float *)((G u8 *)y0 + ys))[r] = epilogue(v1, r, bias, res ? (const G float *)((const G u8 *)res + rs) : res);
    if (T > 2u) ((G float *)((G u8 *)y0 + 2ul * ys))[r] = epilogue(v2, r, bias, res ? (const G float *)((const G u8 *)res + 2ul * rs) : res);
    if (T > 3u) ((G float *)((G u8 *)y0 + 3ul * ys))[r] = epilogue(v3, r, bias, res ? (const G float *)((const G u8 *)res + 3ul * rs) : res);
    if (T > 4u) ((G float *)((G u8 *)y0 + 4ul * ys))[r] = epilogue(v4, r, bias, res ? (const G float *)((const G u8 *)res + 4ul * rs) : res);
    if (T > 5u) ((G float *)((G u8 *)y0 + 5ul * ys))[r] = epilogue(v5, r, bias, res ? (const G float *)((const G u8 *)res + 5ul * rs) : res);
    if (T > 6u) ((G float *)((G u8 *)y0 + 6ul * ys))[r] = epilogue(v6, r, bias, res ? (const G float *)((const G u8 *)res + 6ul * rs) : res);
    if (T > 7u) ((G float *)((G u8 *)y0 + 7ul * ys))[r] = epilogue(v7, r, bias, res ? (const G float *)((const G u8 *)res + 7ul * rs) : res);
  }
}
KERNEL_CALL ie_gemv_raw_iq3xxs_tr(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                               const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                               float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  const u32 r = ie_gemv_row(wgid(), (u32)__builtin_amdgcn_readfirstlane(tid() >> 5)), l = lane();
  if (r >= rows || T == 0u) return;
  const u32 mb = nb / 8u, bs = ie_rq_bs(IE_RQ_IQ3_XXS);
  const G u8 *row = q + (unsigned long)r * mb * bs;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
  for (u32 b = l; b < nb; b += 32u) { /* the row kernel's sub-block map: all 32 lanes busy */
    const u32 qb = b >> 3u, sb = b & 7u;
    const G u8 *blk = row + (unsigned long)qb * bs;
    if (T > 0u) a0 += ie_rq_dot_sub(IE_RQ_IQ3_XXS, blk, xq, nb, qb, sb);
    if (T > 1u) a1 += ie_rq_dot_sub(IE_RQ_IQ3_XXS, blk, xq + (unsigned long)xs, nb, qb, sb);
    if (T > 2u) a2 += ie_rq_dot_sub(IE_RQ_IQ3_XXS, blk, xq + (unsigned long)2u * xs, nb, qb, sb);
    if (T > 3u) a3 += ie_rq_dot_sub(IE_RQ_IQ3_XXS, blk, xq + (unsigned long)3u * xs, nb, qb, sb);
    if (T > 4u) a4 += ie_rq_dot_sub(IE_RQ_IQ3_XXS, blk, xq + (unsigned long)4u * xs, nb, qb, sb);
    if (T > 5u) a5 += ie_rq_dot_sub(IE_RQ_IQ3_XXS, blk, xq + (unsigned long)5u * xs, nb, qb, sb);
    if (T > 6u) a6 += ie_rq_dot_sub(IE_RQ_IQ3_XXS, blk, xq + (unsigned long)6u * xs, nb, qb, sb);
    if (T > 7u) a7 += ie_rq_dot_sub(IE_RQ_IQ3_XXS, blk, xq + (unsigned long)7u * xs, nb, qb, sb);
  }
  const float v0 = T > 0u ? wave_tree_f(a0) : 0.0f;
  const float v1 = T > 1u ? wave_tree_f(a1) : 0.0f;
  const float v2 = T > 2u ? wave_tree_f(a2) : 0.0f;
  const float v3 = T > 3u ? wave_tree_f(a3) : 0.0f;
  const float v4 = T > 4u ? wave_tree_f(a4) : 0.0f;
  const float v5 = T > 5u ? wave_tree_f(a5) : 0.0f;
  const float v6 = T > 6u ? wave_tree_f(a6) : 0.0f;
  const float v7 = T > 7u ? wave_tree_f(a7) : 0.0f;
  if (l == 0u) {
    G float *y0 = (G float *)((G u8 *)y);
    if (T > 0u) ((G float *)((G u8 *)y0 + 0ul * ys))[r] = epilogue(v0, r, bias, res ? (const G float *)((const G u8 *)res + 0ul * rs) : res);
    if (T > 1u) ((G float *)((G u8 *)y0 + ys))[r] = epilogue(v1, r, bias, res ? (const G float *)((const G u8 *)res + rs) : res);
    if (T > 2u) ((G float *)((G u8 *)y0 + 2ul * ys))[r] = epilogue(v2, r, bias, res ? (const G float *)((const G u8 *)res + 2ul * rs) : res);
    if (T > 3u) ((G float *)((G u8 *)y0 + 3ul * ys))[r] = epilogue(v3, r, bias, res ? (const G float *)((const G u8 *)res + 3ul * rs) : res);
    if (T > 4u) ((G float *)((G u8 *)y0 + 4ul * ys))[r] = epilogue(v4, r, bias, res ? (const G float *)((const G u8 *)res + 4ul * rs) : res);
    if (T > 5u) ((G float *)((G u8 *)y0 + 5ul * ys))[r] = epilogue(v5, r, bias, res ? (const G float *)((const G u8 *)res + 5ul * rs) : res);
    if (T > 6u) ((G float *)((G u8 *)y0 + 6ul * ys))[r] = epilogue(v6, r, bias, res ? (const G float *)((const G u8 *)res + 6ul * rs) : res);
    if (T > 7u) ((G float *)((G u8 *)y0 + 7ul * ys))[r] = epilogue(v7, r, bias, res ? (const G float *)((const G u8 *)res + 7ul * rs) : res);
  }
}
KERNEL_CALL ie_gemv_raw_iq1s_tr(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                               const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                               float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  const u32 r = ie_gemv_row(wgid(), (u32)__builtin_amdgcn_readfirstlane(tid() >> 5)), l = lane();
  if (r >= rows || T == 0u) return;
  const u32 mb = nb / 8u, bs = ie_rq_bs(IE_RQ_IQ1_S);
  const G u8 *row = q + (unsigned long)r * mb * bs;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
  for (u32 b = l; b < nb; b += 32u) { /* the row kernel's sub-block map: all 32 lanes busy */
    const u32 qb = b >> 3u, sb = b & 7u;
    const G u8 *blk = row + (unsigned long)qb * bs;
    if (T > 0u) a0 += ie_rq_dot_sub(IE_RQ_IQ1_S, blk, xq, nb, qb, sb);
    if (T > 1u) a1 += ie_rq_dot_sub(IE_RQ_IQ1_S, blk, xq + (unsigned long)xs, nb, qb, sb);
    if (T > 2u) a2 += ie_rq_dot_sub(IE_RQ_IQ1_S, blk, xq + (unsigned long)2u * xs, nb, qb, sb);
    if (T > 3u) a3 += ie_rq_dot_sub(IE_RQ_IQ1_S, blk, xq + (unsigned long)3u * xs, nb, qb, sb);
    if (T > 4u) a4 += ie_rq_dot_sub(IE_RQ_IQ1_S, blk, xq + (unsigned long)4u * xs, nb, qb, sb);
    if (T > 5u) a5 += ie_rq_dot_sub(IE_RQ_IQ1_S, blk, xq + (unsigned long)5u * xs, nb, qb, sb);
    if (T > 6u) a6 += ie_rq_dot_sub(IE_RQ_IQ1_S, blk, xq + (unsigned long)6u * xs, nb, qb, sb);
    if (T > 7u) a7 += ie_rq_dot_sub(IE_RQ_IQ1_S, blk, xq + (unsigned long)7u * xs, nb, qb, sb);
  }
  const float v0 = T > 0u ? wave_tree_f(a0) : 0.0f;
  const float v1 = T > 1u ? wave_tree_f(a1) : 0.0f;
  const float v2 = T > 2u ? wave_tree_f(a2) : 0.0f;
  const float v3 = T > 3u ? wave_tree_f(a3) : 0.0f;
  const float v4 = T > 4u ? wave_tree_f(a4) : 0.0f;
  const float v5 = T > 5u ? wave_tree_f(a5) : 0.0f;
  const float v6 = T > 6u ? wave_tree_f(a6) : 0.0f;
  const float v7 = T > 7u ? wave_tree_f(a7) : 0.0f;
  if (l == 0u) {
    G float *y0 = (G float *)((G u8 *)y);
    if (T > 0u) ((G float *)((G u8 *)y0 + 0ul * ys))[r] = epilogue(v0, r, bias, res ? (const G float *)((const G u8 *)res + 0ul * rs) : res);
    if (T > 1u) ((G float *)((G u8 *)y0 + ys))[r] = epilogue(v1, r, bias, res ? (const G float *)((const G u8 *)res + rs) : res);
    if (T > 2u) ((G float *)((G u8 *)y0 + 2ul * ys))[r] = epilogue(v2, r, bias, res ? (const G float *)((const G u8 *)res + 2ul * rs) : res);
    if (T > 3u) ((G float *)((G u8 *)y0 + 3ul * ys))[r] = epilogue(v3, r, bias, res ? (const G float *)((const G u8 *)res + 3ul * rs) : res);
    if (T > 4u) ((G float *)((G u8 *)y0 + 4ul * ys))[r] = epilogue(v4, r, bias, res ? (const G float *)((const G u8 *)res + 4ul * rs) : res);
    if (T > 5u) ((G float *)((G u8 *)y0 + 5ul * ys))[r] = epilogue(v5, r, bias, res ? (const G float *)((const G u8 *)res + 5ul * rs) : res);
    if (T > 6u) ((G float *)((G u8 *)y0 + 6ul * ys))[r] = epilogue(v6, r, bias, res ? (const G float *)((const G u8 *)res + 6ul * rs) : res);
    if (T > 7u) ((G float *)((G u8 *)y0 + 7ul * ys))[r] = epilogue(v7, r, bias, res ? (const G float *)((const G u8 *)res + 7ul * rs) : res);
  }
}
KERNEL_CALL ie_gemv_raw_iq2s_tr(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                               const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                               float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  const u32 r = ie_gemv_row(wgid(), (u32)__builtin_amdgcn_readfirstlane(tid() >> 5)), l = lane();
  if (r >= rows || T == 0u) return;
  const u32 mb = nb / 8u, bs = ie_rq_bs(IE_RQ_IQ2_S);
  const G u8 *row = q + (unsigned long)r * mb * bs;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
  for (u32 b = l; b < nb; b += 32u) { /* the row kernel's sub-block map: all 32 lanes busy */
    const u32 qb = b >> 3u, sb = b & 7u;
    const G u8 *blk = row + (unsigned long)qb * bs;
    if (T > 0u) a0 += ie_rq_dot_sub(IE_RQ_IQ2_S, blk, xq, nb, qb, sb);
    if (T > 1u) a1 += ie_rq_dot_sub(IE_RQ_IQ2_S, blk, xq + (unsigned long)xs, nb, qb, sb);
    if (T > 2u) a2 += ie_rq_dot_sub(IE_RQ_IQ2_S, blk, xq + (unsigned long)2u * xs, nb, qb, sb);
    if (T > 3u) a3 += ie_rq_dot_sub(IE_RQ_IQ2_S, blk, xq + (unsigned long)3u * xs, nb, qb, sb);
    if (T > 4u) a4 += ie_rq_dot_sub(IE_RQ_IQ2_S, blk, xq + (unsigned long)4u * xs, nb, qb, sb);
    if (T > 5u) a5 += ie_rq_dot_sub(IE_RQ_IQ2_S, blk, xq + (unsigned long)5u * xs, nb, qb, sb);
    if (T > 6u) a6 += ie_rq_dot_sub(IE_RQ_IQ2_S, blk, xq + (unsigned long)6u * xs, nb, qb, sb);
    if (T > 7u) a7 += ie_rq_dot_sub(IE_RQ_IQ2_S, blk, xq + (unsigned long)7u * xs, nb, qb, sb);
  }
  const float v0 = T > 0u ? wave_tree_f(a0) : 0.0f;
  const float v1 = T > 1u ? wave_tree_f(a1) : 0.0f;
  const float v2 = T > 2u ? wave_tree_f(a2) : 0.0f;
  const float v3 = T > 3u ? wave_tree_f(a3) : 0.0f;
  const float v4 = T > 4u ? wave_tree_f(a4) : 0.0f;
  const float v5 = T > 5u ? wave_tree_f(a5) : 0.0f;
  const float v6 = T > 6u ? wave_tree_f(a6) : 0.0f;
  const float v7 = T > 7u ? wave_tree_f(a7) : 0.0f;
  if (l == 0u) {
    G float *y0 = (G float *)((G u8 *)y);
    if (T > 0u) ((G float *)((G u8 *)y0 + 0ul * ys))[r] = epilogue(v0, r, bias, res ? (const G float *)((const G u8 *)res + 0ul * rs) : res);
    if (T > 1u) ((G float *)((G u8 *)y0 + ys))[r] = epilogue(v1, r, bias, res ? (const G float *)((const G u8 *)res + rs) : res);
    if (T > 2u) ((G float *)((G u8 *)y0 + 2ul * ys))[r] = epilogue(v2, r, bias, res ? (const G float *)((const G u8 *)res + 2ul * rs) : res);
    if (T > 3u) ((G float *)((G u8 *)y0 + 3ul * ys))[r] = epilogue(v3, r, bias, res ? (const G float *)((const G u8 *)res + 3ul * rs) : res);
    if (T > 4u) ((G float *)((G u8 *)y0 + 4ul * ys))[r] = epilogue(v4, r, bias, res ? (const G float *)((const G u8 *)res + 4ul * rs) : res);
    if (T > 5u) ((G float *)((G u8 *)y0 + 5ul * ys))[r] = epilogue(v5, r, bias, res ? (const G float *)((const G u8 *)res + 5ul * rs) : res);
    if (T > 6u) ((G float *)((G u8 *)y0 + 6ul * ys))[r] = epilogue(v6, r, bias, res ? (const G float *)((const G u8 *)res + 6ul * rs) : res);
    if (T > 7u) ((G float *)((G u8 *)y0 + 7ul * ys))[r] = epilogue(v7, r, bias, res ? (const G float *)((const G u8 *)res + 7ul * rs) : res);
  }
}
KERNEL_CALL ie_gemv_raw_iq1m_tr(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                               const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                               float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  const u32 r = ie_gemv_row(wgid(), (u32)__builtin_amdgcn_readfirstlane(tid() >> 5)), l = lane();
  if (r >= rows || T == 0u) return;
  const u32 mb = nb / 8u, bs = ie_rq_bs(IE_RQ_IQ1_M);
  const G u8 *row = q + (unsigned long)r * mb * bs;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
  for (u32 b = l; b < nb; b += 32u) { /* the row kernel's sub-block map: all 32 lanes busy */
    const u32 qb = b >> 3u, sb = b & 7u;
    const G u8 *blk = row + (unsigned long)qb * bs;
    if (T > 0u) a0 += ie_rq_dot_sub(IE_RQ_IQ1_M, blk, xq, nb, qb, sb);
    if (T > 1u) a1 += ie_rq_dot_sub(IE_RQ_IQ1_M, blk, xq + (unsigned long)xs, nb, qb, sb);
    if (T > 2u) a2 += ie_rq_dot_sub(IE_RQ_IQ1_M, blk, xq + (unsigned long)2u * xs, nb, qb, sb);
    if (T > 3u) a3 += ie_rq_dot_sub(IE_RQ_IQ1_M, blk, xq + (unsigned long)3u * xs, nb, qb, sb);
    if (T > 4u) a4 += ie_rq_dot_sub(IE_RQ_IQ1_M, blk, xq + (unsigned long)4u * xs, nb, qb, sb);
    if (T > 5u) a5 += ie_rq_dot_sub(IE_RQ_IQ1_M, blk, xq + (unsigned long)5u * xs, nb, qb, sb);
    if (T > 6u) a6 += ie_rq_dot_sub(IE_RQ_IQ1_M, blk, xq + (unsigned long)6u * xs, nb, qb, sb);
    if (T > 7u) a7 += ie_rq_dot_sub(IE_RQ_IQ1_M, blk, xq + (unsigned long)7u * xs, nb, qb, sb);
  }
  const float v0 = T > 0u ? wave_tree_f(a0) : 0.0f;
  const float v1 = T > 1u ? wave_tree_f(a1) : 0.0f;
  const float v2 = T > 2u ? wave_tree_f(a2) : 0.0f;
  const float v3 = T > 3u ? wave_tree_f(a3) : 0.0f;
  const float v4 = T > 4u ? wave_tree_f(a4) : 0.0f;
  const float v5 = T > 5u ? wave_tree_f(a5) : 0.0f;
  const float v6 = T > 6u ? wave_tree_f(a6) : 0.0f;
  const float v7 = T > 7u ? wave_tree_f(a7) : 0.0f;
  if (l == 0u) {
    G float *y0 = (G float *)((G u8 *)y);
    if (T > 0u) ((G float *)((G u8 *)y0 + 0ul * ys))[r] = epilogue(v0, r, bias, res ? (const G float *)((const G u8 *)res + 0ul * rs) : res);
    if (T > 1u) ((G float *)((G u8 *)y0 + ys))[r] = epilogue(v1, r, bias, res ? (const G float *)((const G u8 *)res + rs) : res);
    if (T > 2u) ((G float *)((G u8 *)y0 + 2ul * ys))[r] = epilogue(v2, r, bias, res ? (const G float *)((const G u8 *)res + 2ul * rs) : res);
    if (T > 3u) ((G float *)((G u8 *)y0 + 3ul * ys))[r] = epilogue(v3, r, bias, res ? (const G float *)((const G u8 *)res + 3ul * rs) : res);
    if (T > 4u) ((G float *)((G u8 *)y0 + 4ul * ys))[r] = epilogue(v4, r, bias, res ? (const G float *)((const G u8 *)res + 4ul * rs) : res);
    if (T > 5u) ((G float *)((G u8 *)y0 + 5ul * ys))[r] = epilogue(v5, r, bias, res ? (const G float *)((const G u8 *)res + 5ul * rs) : res);
    if (T > 6u) ((G float *)((G u8 *)y0 + 6ul * ys))[r] = epilogue(v6, r, bias, res ? (const G float *)((const G u8 *)res + 6ul * rs) : res);
    if (T > 7u) ((G float *)((G u8 *)y0 + 7ul * ys))[r] = epilogue(v7, r, bias, res ? (const G float *)((const G u8 *)res + 7ul * rs) : res);
  }
}









/* IQ3_S (3.44 bpw) natively: the model carries 44 of these tensors (2.9 B
 * parameters). Dequantised to Q8 at load they cost 3.1 GB of the 9.4 GB a
 * token streams; read as raw blocks they are 1.25 GB, which is the largest
 * single win left on this build. */
KERNEL_LOOP ie_gemv_raw_iq3s(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 type,
                             const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                             float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)type;
  const G u8 *x = TOKC(xq, xs);
  G float *yy = TOK(y, ys);
  const G float *rr = res ? (const G float *)((const G u8 *)res + (unsigned long)tokid() * rs) : res;
  gemv_rawq_rows(wgid(), q, x, yy, rows, nb, IE_RQ_IQ3_S, bias, rr);
  if (nw) gemv_norm_tail(yy, rows, nw, ny, nq, eps, count);
}


KERNEL_CALL ie_gemv_raw_iq3s_tr(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                               const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                               float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  const u32 r = ie_gemv_row(wgid(), (u32)__builtin_amdgcn_readfirstlane(tid() >> 5)), l = lane();
  if (r >= rows || T == 0u) return;
  const u32 mb = nb / 8u, bs = ie_rq_bs(IE_RQ_IQ3_S);
  const G u8 *row = q + (unsigned long)r * mb * bs;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
  for (u32 b = l; b < nb; b += 32u) { /* the row kernel's sub-block map: all 32 lanes busy */
    const u32 qb = b >> 3u, sb = b & 7u;
    const G u8 *blk = row + (unsigned long)qb * bs;
    if (T > 0u) a0 += ie_rq_dot_sub(IE_RQ_IQ3_S, blk, xq, nb, qb, sb);
    if (T > 1u) a1 += ie_rq_dot_sub(IE_RQ_IQ3_S, blk, xq + (unsigned long)xs, nb, qb, sb);
    if (T > 2u) a2 += ie_rq_dot_sub(IE_RQ_IQ3_S, blk, xq + (unsigned long)2u * xs, nb, qb, sb);
    if (T > 3u) a3 += ie_rq_dot_sub(IE_RQ_IQ3_S, blk, xq + (unsigned long)3u * xs, nb, qb, sb);
    if (T > 4u) a4 += ie_rq_dot_sub(IE_RQ_IQ3_S, blk, xq + (unsigned long)4u * xs, nb, qb, sb);
    if (T > 5u) a5 += ie_rq_dot_sub(IE_RQ_IQ3_S, blk, xq + (unsigned long)5u * xs, nb, qb, sb);
    if (T > 6u) a6 += ie_rq_dot_sub(IE_RQ_IQ3_S, blk, xq + (unsigned long)6u * xs, nb, qb, sb);
    if (T > 7u) a7 += ie_rq_dot_sub(IE_RQ_IQ3_S, blk, xq + (unsigned long)7u * xs, nb, qb, sb);
  }
  const float v0 = T > 0u ? wave_tree_f(a0) : 0.0f;
  const float v1 = T > 1u ? wave_tree_f(a1) : 0.0f;
  const float v2 = T > 2u ? wave_tree_f(a2) : 0.0f;
  const float v3 = T > 3u ? wave_tree_f(a3) : 0.0f;
  const float v4 = T > 4u ? wave_tree_f(a4) : 0.0f;
  const float v5 = T > 5u ? wave_tree_f(a5) : 0.0f;
  const float v6 = T > 6u ? wave_tree_f(a6) : 0.0f;
  const float v7 = T > 7u ? wave_tree_f(a7) : 0.0f;
  if (l == 0u) {
    G float *y0 = (G float *)((G u8 *)y);
    if (T > 0u) ((G float *)((G u8 *)y0 + 0ul * ys))[r] = epilogue(v0, r, bias, res ? (const G float *)((const G u8 *)res + 0ul * rs) : res);
    if (T > 1u) ((G float *)((G u8 *)y0 + ys))[r] = epilogue(v1, r, bias, res ? (const G float *)((const G u8 *)res + rs) : res);
    if (T > 2u) ((G float *)((G u8 *)y0 + 2ul * ys))[r] = epilogue(v2, r, bias, res ? (const G float *)((const G u8 *)res + 2ul * rs) : res);
    if (T > 3u) ((G float *)((G u8 *)y0 + 3ul * ys))[r] = epilogue(v3, r, bias, res ? (const G float *)((const G u8 *)res + 3ul * rs) : res);
    if (T > 4u) ((G float *)((G u8 *)y0 + 4ul * ys))[r] = epilogue(v4, r, bias, res ? (const G float *)((const G u8 *)res + 4ul * rs) : res);
    if (T > 5u) ((G float *)((G u8 *)y0 + 5ul * ys))[r] = epilogue(v5, r, bias, res ? (const G float *)((const G u8 *)res + 5ul * rs) : res);
    if (T > 6u) ((G float *)((G u8 *)y0 + 6ul * ys))[r] = epilogue(v6, r, bias, res ? (const G float *)((const G u8 *)res + 6ul * rs) : res);
    if (T > 7u) ((G float *)((G u8 *)y0 + 7ul * ys))[r] = epilogue(v7, r, bias, res ? (const G float *)((const G u8 *)res + 7ul * rs) : res);
  }
}

#ifndef IE_MEGA /* not in the megakernel (its sets list the _tr kernels; src/hip.c keeps those for megakernel runs) */
#ifndef IE_RQ_Q8_0
#define IE_RQ_Q8_0 8u
#endif
static inline __attribute__((always_inline)) void gemv_q8q8_mseg(u32 wg, const G u8 *q, const G u8 *xq, G float *y,
                                                                 u32 rows, u32 nb, u32 T, const G float *bias,
                                                                 const G float *res, u32 xs, u32 ys, u32 rs) {
  const u32 gr = ie_gemv_ngroups(rows), t = wg / gr, g = wg - t * gr;
  if (t >= T) return;
  gemv_q8q8_rows(g, (const G u32 *)bias, (const G f16 *)q, xq + (unsigned long)t * xs,
                 (G float *)((G u8 *)y + (unsigned long)t * ys), rows, nb, (const G float *)0,
                 res ? (const G float *)((const G u8 *)res + (unsigned long)t * rs) : (const G float *)0);
}
#include "ie_raw_tr4.inc"
/* Verify of a longer speculative step: 5 to 9 tokens through one pass over the weights (ie_gemv_raw_*_tr8).
 *
 * The same idea as the _tr4 kernels: the token count is a compile-time constant in each case of the switch, and a
 * wave may do two rows that share the activation loads of the tokens. Same sums as ie_gemv_raw_*_tr (each token's
 * lane adds the sub-blocks l, l + 32, ... in ascending order, then the wave tree), so every output is bitwise equal.
 * IE_TR8_Rn is the number of rows per wave at n tokens (1, 2 or 4; src/hip.c tr8_rpw must agree: the grid is
 * rows / (8 R) workgroups). IE_TR8_GRP > 0 and < n works through the tokens in groups of that many, with a
 * scheduling barrier between the groups, so that the activation registers of all tokens are not live at once. */
#ifndef IE_TR8_R5
#define IE_TR8_R5 2u
#endif
#ifndef IE_TR8_R6
#define IE_TR8_R6 2u
#endif
#ifndef IE_TR8_R7
#define IE_TR8_R7 2u /* 2 rows: 7 to 31% faster than 1 at 7 tokens, 9 to 27% at 8 (tools/raw_gemv_bench.py -tr8) */
#endif
#ifndef IE_TR8_R8
#define IE_TR8_R8 2u
#endif
#ifndef IE_TR8_R9 /* 9 tokens: a verify of 8 drafts */
#define IE_TR8_R9 2u
#endif
#ifndef IE_TR8_GRP
#define IE_TR8_GRP 8u
#endif
/* the token group at 9 tokens (IE_TR8_GRP9; 9: one group, the weights decoded once for all 9) */
#ifndef IE_TR8_GRP9
#define IE_TR8_GRP9 9u
#endif
#ifndef IE_TR16_R
#define IE_TR16_R 2u
#endif
#define TR8_GRPN(n) ((n) > 9u ? 8u : (n) > 8u ? IE_TR8_GRP9 : IE_TR8_GRP)
/* 4 rows per wave for the types that were faster so at 8 tokens (IQ1_S, IQ1_M: their activations are a larger share
 * of the work per weight byte); IE_TR8_R8X=0: IE_TR8_R8 for every type. src/hip.c tr8_rpw must agree. */
#ifndef IE_TR8_R8X
#define IE_TR8_R8X 1
#endif
#define TR8_R8(type) (IE_TR8_R8X && ((type) == IE_RQ_IQ1_S || (type) == IE_RQ_IQ1_M) ? 4u : IE_TR8_R8)
/* The other IQ grids from LDS too (IE_TR8_LDS=0: from global memory): the lookups have a different address in every
 * lane, which LDS serves far faster than the vector cache. Each kernel copies only its own type's tables.
 * The kernel of several matrices (ie_gemv_raw_tr8_m, mt = 1) keeps the tables of whichever type a workgroup has in
 * one array, tr8_tab: the sign masks at 0 .. 127, the grid from TR8_TG on (2048 entries: IQ1_S / IQ1_M, the largest;
 * the IQ3 grids are 32-bit). */
#ifndef IE_TR8_LDS
#define IE_TR8_LDS 1
#endif
#ifndef LDSP
#define LDSP __attribute__((address_space(3)))
#endif
#define TR8_TG 128u
LDS static ie_rq_u64 tr8_tab[TR8_TG + 2048u];
/* The IQ2 grids (IQ2_XXS, IQ2_XS, IQ2_S: every byte 8, 25 or 43) as codes 0, 1, 2 in LDS (IE_TR8_PERM, default 1):
 * one byte permute turns a code with the weight's sign at bit 2 into the signed byte (selector c: 8, 25, 43 from the
 * low word; c + 4: -8, -25, -43 from the high one), so a signed word is an and-or and a permute where it was an xor,
 * an and and an add (IQ2_S: its byte masks, 4 more instructions per word, are not built at all). The same int8
 * values, so bitwise the same sums. */
#ifndef IE_TR8_PERM
#define IE_TR8_PERM 1
#endif
#if IE_TR8_PERM
#if !IE_TR8_LDS
#error IE_TR8_PERM needs IE_TR8_LDS
#endif
#define TR8_P2(sel) __builtin_amdgcn_perm(0x00D5E7F8u, 0x002B1908u, (sel))
/* the code word of a grid entry: byte b -> (b > 8) + (b > 25) */
static inline __attribute__((always_inline)) ie_rq_u64 tr8_code8(ie_rq_u64 g) {
  ie_rq_u64 c = 0u;
  for (ie_rq_u32 i = 0; i < 8u; i++) {
    const ie_rq_u32 b = (ie_rq_u32)(g >> (8u * i)) & 255u;
    c |= (ie_rq_u64)((b > 8u ? 1u : 0u) + (b > 25u ? 1u : 0u)) << (8u * i);
  }
  return c;
}
/* the signed word of 4 codes c and a sign mask m (0x00 / 0xff bytes), or of 4 sign bits b (bit i: weight i;
 * b * 0x00810204 holds bit i at bit 2 of byte i with no carries, as ie_rq_mask4) */
#define TR8_SGN_M(c, m) TR8_P2(((m) & 0x04040404u) | (c))
#define TR8_SGN_B(c, b) TR8_P2((((b) * 0x00810204u) & 0x04040404u) | (c))
#define TR8_G2C(g) tr8_code8(g)
/* IE_TR8_KSALU: the IQ2_XXS / IQ2_XS sign masks from their 7-bit index s (ie_rq_ksigns64: bit 7 = the parity of s)
 * without their LDS table: one LDS load per group of 8 instead of two. p = s * 0x00810204 holds s at bits 2, 9, 16, 23
 * (7 bits apart, s < 128: no carries), so bit 2 of byte i of p is bit i of s (weights 0-3) and of p >> 4 bit i + 4
 * (weights 4-6); byte 3 of the high word takes the parity (popcount bit 0 at bit 26; its higher bits fall outside the
 * mask). These kernels are about as limited by their LDS loads as by their ALU: this removes half the loads for 3
 * more instructions per group. */
#ifndef IE_TR8_KSALU
#define IE_TR8_KSALU 0
#endif
#define TR8_KSW(c0, c1, s, lo, hi)                                                                       \
  do {                                                                                                   \
    const ie_rq_u32 p_ = (s) * 0x00810204u;                                                              \
    (lo) = TR8_P2((p_ & 0x04040404u) | (c0));                                                            \
    (hi) = TR8_P2((((ie_rq_u32)__builtin_popcount(s) << 26u | (p_ >> 4u)) & 0x04040404u) | (c1));        \
  } while (0)
#else
#define TR8_G2C(g) (g)
#endif
#define TR8_TAB32(i) ((LDSP ie_rq_u32 *)(tr8_tab + TR8_TG))[(i)]
#define TR8_GRID_XXS(i) (mt ? tr8_tab[TR8_TG + (i)] : tr4_xxs_grid[(i)])
#define TR8_KS(i) (mt ? tr8_tab[(i)] : tr4_ks[(i)])
/* IQ1_M in _tr8 / _tr8_m: each group of 8 weights as 8 g +- 1 folded via v_perm_b32 on ie_rq_iq1s_grid (16 KiB LDS),
 * so a token's dot is 2 dot4 per group, without the activations' sum and without a 32 KiB tr8_g1m table (bitwise
 * identical outputs). IE_TR8_IQ1M_FOLD=0: the grid and the sum. */
#ifndef IE_TR8_IQ1M_FOLD
#define IE_TR8_IQ1M_FOLD 1
#endif
#if IE_TR8_LDS
LDS static ie_rq_u64 tr8_g2xs[512], tr8_g2s[1024], tr8_g1s[2048];
LDS static ie_rq_u32 tr8_g3xxs[256], tr8_g3s[512];
#define TR8_G2XS(i) (mt ? tr8_tab[TR8_TG + (i)] : tr8_g2xs[(i)])
#define TR8_G2S(i) (mt ? tr8_tab[TR8_TG + (i)] : tr8_g2s[(i)])
#define TR8_G1S(i) (mt ? tr8_tab[TR8_TG + (i)] : tr8_g1s[(i)])
#define TR8_G3XXS(i) (mt ? TR8_TAB32(i) : tr8_g3xxs[(i)])
#define TR8_G3S(i) (mt ? TR8_TAB32(i) : tr8_g3s[(i)])
#define TR8_KSG(i) (mt ? tr8_tab[(i)] : tr4_ks[(i)])
#else
#define TR8_G2XS(i) (mt ? tr8_tab[TR8_TG + (i)] : ie_rq_iq2xs_grid[(i)])
#define TR8_G2S(i) (mt ? tr8_tab[TR8_TG + (i)] : ie_rq_iq2s_grid[(i)])
#define TR8_G1S(i) (mt ? tr8_tab[TR8_TG + (i)] : ie_rq_iq1s_grid[(i)])
#define TR8_G3XXS(i) (mt ? TR8_TAB32(i) : ie_rq_iq3xxs_grid[(i)])
#define TR8_G3S(i) (mt ? TR8_TAB32(i) : ie_rq_iq3s_grid[(i)])
#define TR8_KSG(i) (mt ? tr8_tab[(i)] : ie_rq_ksigns64[(i)])
#endif
/* The contribution of sub-block b (32 weights) of one token to a lane's sum: ie_rq_dot_sub (kernels/ie_iqraw.inc) with
 * the same sums, written so that the activation loads are a wave-uniform base (xb, the token's buffer) plus a 32-bit
 * lane offset: every load of every token then reads the same three lane offsets, and the compiler does not build one
 * 64-bit vector address per token and per load. p is the raw 256-weight block of the row that holds sub-block b. The
 * IQ2_XXS tables come from LDS (TR8_GRID_XXS, TR8_KS). IE_TR8_KEEPZERO: add the term to a 0.0f first, as the original
 * does; without it the term is returned as it is (the lane sums start at +0.0f, so a term of -0.0f added to one or to
 * 0.0f + the term makes the same sum). */
/* sx (the _tr9 kernels: b the same in every lane): the activations through a constant-memory pointer, which lets the
 * compiler use scalar loads for them (it cannot prove that a global pointer's memory is not written) */
#define TR8_CSP __attribute__((address_space(4)))
#define TR8_LD(T, a) (sx ? *(const TR8_CSP T *)(a) : *(const G T *)(a))
#define TR8_X(w) TR8_LD(ie_rq_u32, xb + xo + 4u * (w))
static inline __attribute__((always_inline)) float tr8_dot(const ie_rq_u32 type, const int mt, IE_RQP *p, IE_RQP *xb, ie_rq_u32 nb,
                                                           ie_rq_u32 b, const int sx) {
#pragma clang fp contract(off)
  const ie_rq_u32 s = b & 7u, xo = 32u * b;
  const float dav = TR8_LD(float, xb + (32u * nb + 4u * b)); /* the scale of this 32-block of the token */
  float acc = 0.0f;
#ifdef IE_TR8_KEEPZERO
#define TR8_RET(term) do { acc += (term); return acc; } while (0)
#else
#define TR8_RET(term) return (term)
#endif
  switch (type) {
    case IE_RQ_Q2_K: {
      IE_RQP *ps = p + 2u * s, *pq = p + (16u + (s >> 2u) * 32u);
      const float d = ie_rq_h2f(ie_rq_a16(p, 80u)), mn = ie_rq_h2f(ie_rq_a16(p, 82u));
      const ie_rq_u32 sc16 = ie_rq_a16(ps, 0u);
      const ie_rq_u32 sh = 2u * (s & 3u), qmask = 0x03030303u << sh;
      for (ie_rq_u32 h = 0; h < 2u; h++) {
        const ie_rq_u32 sc = (sc16 >> (8u * h)) & 255u;
        ie_rq_u32 idots = 0u;
        for (ie_rq_u32 w = 0; w < 4u; w++) idots = ie_dot4_us(idots, ie_rq_a32(pq, 16u * h + 4u * w) & qmask, TR8_X(4u * h + w));
        const ie_rq_u32 idot = (ie_rq_u32)((int)idots >> sh);
        const float dw = d * (float)(sc & 15u), dm = mn * (float)(sc >> 4u);
        const ie_rq_u32 xs = ie_dot4_ss(ie_dot4_ss(0u, TR8_X(4u * h), 0x01010101u), TR8_X(4u * h + 1u), 0x01010101u) +
                             ie_dot4_ss(ie_dot4_ss(0u, TR8_X(4u * h + 2u), 0x01010101u), TR8_X(4u * h + 3u), 0x01010101u);
        const float term = dav * (dw * (float)(int)idot - dm * (float)(int)xs);
#ifdef IE_TR8_KEEPZERO
        acc += term;
#else
        acc = h == 0u ? term : acc + term;
#endif
      }
      return acc;
    }
    case IE_RQ_IQ2_XXS: {
      const ie_rq_u32 o = 2u + 8u * s;
      const float d = ie_rq_h2f(ie_rq_a16(p, 0u));
      const ie_rq_u32 a0 = ie_rq_a32(p, o), a1 = ie_rq_a32(p, o + 4u);
      const float dw = d * (0.5f + (float)(a1 >> 28u)) * 0.25f;
      ie_rq_u32 idot = 0u;
      for (ie_rq_u32 l = 0; l < 4u; l++) {
#if IE_TR8_PERM && IE_TR8_KSALU
        {
          const ie_rq_u64 g = TR8_GRID_XXS(ie_byte(a0, l));
          ie_rq_u32 wl, wh;
          TR8_KSW((ie_rq_u32)g, (ie_rq_u32)(g >> 32u), (a1 >> (7u * l)) & 127u, wl, wh);
          idot = ie_dot4_ss(ie_dot4_ss(idot, wl, TR8_X(2u * l)), wh, TR8_X(2u * l + 1u));
          continue;
        }
#endif
        const ie_rq_u64 m = TR8_KS((a1 >> (7u * l)) & 127u), g = TR8_GRID_XXS(ie_byte(a0, l));
#if IE_TR8_PERM
        idot = ie_dot4_ss(ie_dot4_ss(idot, TR8_SGN_M((ie_rq_u32)g, (ie_rq_u32)m), TR8_X(2u * l)),
                          TR8_SGN_M((ie_rq_u32)(g >> 32u), (ie_rq_u32)(m >> 32u)), TR8_X(2u * l + 1u));
#else
        idot = ie_dot4_ss(ie_dot4_ss(idot, ie_rq_ssign((ie_rq_u32)g, (ie_rq_u32)m), TR8_X(2u * l)),
                          ie_rq_ssign((ie_rq_u32)(g >> 32u), (ie_rq_u32)(m >> 32u)), TR8_X(2u * l + 1u));
#endif
      }
      TR8_RET(dav * (dw * (float)(int)idot));
    }
    case IE_RQ_IQ2_XS: {
      const float d = ie_rq_h2f(ie_rq_a16(p, 0u));
      const ie_rq_u32 sc = (ie_rq_a16(p, 66u + 2u * (s >> 1u)) >> (8u * (s & 1u))) & 255u;
      const float dw0 = d * (0.5f + (float)(sc & 15u)) * 0.25f, dw1 = d * (0.5f + (float)(sc >> 4u)) * 0.25f;
      ie_rq_u32 id0 = 0u, id1 = 0u;
      for (ie_rq_u32 l = 0; l < 4u; l++) {
        const ie_rq_u32 q2 = ie_rq_a16(p, 2u + 8u * s + 2u * l);
#if IE_TR8_PERM && IE_TR8_KSALU
        {
          const ie_rq_u64 g = TR8_G2XS(q2 & 511u);
          ie_rq_u32 wl, wh;
          TR8_KSW((ie_rq_u32)g, (ie_rq_u32)(g >> 32u), q2 >> 9u, wl, wh);
          const ie_rq_u32 id = ie_dot4_ss(ie_dot4_ss(0u, wl, TR8_X(2u * l)), wh, TR8_X(2u * l + 1u));
          if (l < 2u) id0 += id; else id1 += id;
          continue;
        }
#endif
        const ie_rq_u64 m = TR8_KSG(q2 >> 9u), g = TR8_G2XS(q2 & 511u);
#if IE_TR8_PERM
        const ie_rq_u32 id = ie_dot4_ss(ie_dot4_ss(0u, TR8_SGN_M((ie_rq_u32)g, (ie_rq_u32)m), TR8_X(2u * l)),
                                        TR8_SGN_M((ie_rq_u32)(g >> 32u), (ie_rq_u32)(m >> 32u)), TR8_X(2u * l + 1u));
#else
        const ie_rq_u32 id = ie_dot4_ss(ie_dot4_ss(0u, ie_rq_ssign((ie_rq_u32)g, (ie_rq_u32)m), TR8_X(2u * l)),
                                        ie_rq_ssign((ie_rq_u32)(g >> 32u), (ie_rq_u32)(m >> 32u)), TR8_X(2u * l + 1u));
#endif
        if (l < 2u) id0 += id; else id1 += id;
      }
      TR8_RET(dav * (dw0 * (float)(int)id0 + dw1 * (float)(int)id1));
    }
    case IE_RQ_IQ2_S: {
      const float d = ie_rq_h2f(ie_rq_a16(p, 0u));
      const ie_rq_u32 qh = (ie_rq_a16(p, 66u + 2u * (s >> 1u)) >> (8u * (s & 1u))) & 255u;
      const ie_rq_u32 sc = (ie_rq_a16(p, 74u + 2u * (s >> 1u)) >> (8u * (s & 1u))) & 255u;
      const ie_rq_u32 qlo = ie_rq_a32(p, 2u + 4u * s), sgn = ie_rq_a32(p, 34u + 4u * s);
      const float dw0 = d * (0.5f + (float)(sc & 15u)) * 0.25f, dw1 = d * (0.5f + (float)(sc >> 4u)) * 0.25f;
      ie_rq_u32 id0 = 0u, id1 = 0u;
      for (ie_rq_u32 l = 0; l < 4u; l++) {
        const ie_rq_u32 idx = ie_byte(qlo, l) | ((qh << (8u - 2u * l)) & 0x300u);
#if IE_TR8_PERM
        const ie_rq_u32 sb = ie_byte(sgn, l);
        const ie_rq_u64 g = TR8_G2S(idx);
        const ie_rq_u32 id = ie_dot4_ss(ie_dot4_ss(0u, TR8_SGN_B((ie_rq_u32)g, sb & 15u), TR8_X(2u * l)),
                                        TR8_SGN_B((ie_rq_u32)(g >> 32u), sb >> 4u), TR8_X(2u * l + 1u));
#else
        const ie_rq_u64 m = ie_rq_mask8(ie_byte(sgn, l)), g = TR8_G2S(idx);
        const ie_rq_u32 id = ie_dot4_ss(ie_dot4_ss(0u, ie_rq_ssign((ie_rq_u32)g, (ie_rq_u32)m), TR8_X(2u * l)),
                                        ie_rq_ssign((ie_rq_u32)(g >> 32u), (ie_rq_u32)(m >> 32u)), TR8_X(2u * l + 1u));
#endif
        if (l < 2u) id0 += id; else id1 += id;
      }
      TR8_RET(dav * (dw0 * (float)(int)id0 + dw1 * (float)(int)id1));
    }
    case IE_RQ_IQ3_XXS: {
      const float d = ie_rq_h2f(ie_rq_a16(p, 0u));
      const ie_rq_u32 aux = ie_rq_a32(p, 66u + 4u * s);
      const float dw = d * (0.5f + (float)(aux >> 28u)) * 0.5f;
      ie_rq_u32 idot = 0u;
      for (ie_rq_u32 l = 0; l < 4u; l++) {
        const ie_rq_u32 qp = ie_rq_a16(p, 2u + 8u * s + 2u * l);
        const ie_rq_u32 g1 = TR8_G3XXS(qp & 255u);
        const ie_rq_u32 g2 = TR8_G3XXS(qp >> 8u);
        const ie_rq_u64 m = TR8_KSG((aux >> (7u * l)) & 127u);
        idot = ie_dot4_ss(ie_dot4_ss(idot, ie_rq_ssign(g1, (ie_rq_u32)m), TR8_X(2u * l)),
                          ie_rq_ssign(g2, (ie_rq_u32)(m >> 32u)), TR8_X(2u * l + 1u));
      }
      TR8_RET(dav * (dw * (float)(int)idot));
    }
    case IE_RQ_IQ3_S: {
      const float d = ie_rq_h2f(ie_rq_a16(p, 0u));
      const ie_rq_u32 qh = (ie_rq_a16(p, 66u + 2u * (s >> 1u)) >> (8u * (s & 1u))) & 255u;
      const ie_rq_u32 sc = (ie_rq_a16(p, 106u + 2u * (s >> 2u)) >> (4u * (s & 3u))) & 15u;
      const ie_rq_u32 sgn = ie_rq_a32(p, 74u + 4u * s);
      const float dw = d * (float)(int)(1u + 2u * sc);
      ie_rq_u32 idot = 0u;
      for (ie_rq_u32 l = 0; l < 4u; l++) {
        const ie_rq_u32 qp = ie_rq_a16(p, 2u + 8u * s + 2u * l);
        const ie_rq_u32 i1 = (qp & 255u) | (((qh >> (2u * l)) & 1u) << 8u);
        const ie_rq_u32 i2 = (qp >> 8u) | (((qh >> (2u * l + 1u)) & 1u) << 8u);
        const ie_rq_u32 g1 = TR8_G3S(i1), g2 = TR8_G3S(i2);
        const ie_rq_u64 m = ie_rq_mask8(ie_byte(sgn, l));
        idot = ie_dot4_ss(ie_dot4_ss(idot, ie_rq_ssign(g1, (ie_rq_u32)m), TR8_X(2u * l)),
                          ie_rq_ssign(g2, (ie_rq_u32)(m >> 32u)), TR8_X(2u * l + 1u));
      }
      TR8_RET(dav * (dw * (float)(int)idot));
    }
    case IE_RQ_IQ1_S: {
      IE_RQP *pi = p + 4u * s, *ph = p + 2u * s;
      const float d = ie_rq_h2f(ie_rq_a16(p, 0u));
      const ie_rq_u32 qh = ie_rq_a16(ph, 34u), qlo = ie_rq_a32(pi, 2u);
      const float dw = d * (float)(int)(2u * ((qh >> 12u) & 7u) + 1u);
      /* the folded 8 g +- 1 table (as IQ1_M) gives bitwise the same sums but was no faster here (375 GB/s both, 6
       * tokens): this type's time is its grid lookups, not the arithmetic */
      const float del = (qh & 0x8000u) != 0u ? -0.125f : 0.125f; /* IQ1S_DELTA */
      ie_rq_u32 idot = 0u;
      for (ie_rq_u32 l = 0; l < 4u; l++) {
        const ie_rq_u32 idx = ie_byte(qlo, l) | (((qh >> (3u * l)) & 7u) << 8u);
        const ie_rq_u64 g = TR8_G1S(idx);
        idot = ie_dot4_ss(ie_dot4_ss(idot, (ie_rq_u32)g, TR8_X(2u * l)), (ie_rq_u32)(g >> 32u), TR8_X(2u * l + 1u));
      }
      const ie_rq_u32 as = TR8_LD(ie_rq_u32, xb + (36u * nb + 4u * b)); /* the int8 sum of this 32-block */
      TR8_RET(dav * (dw * ((float)(int)idot + del * (float)(int)as)));
    }
    case IE_RQ_IQ1_M: {
      IE_RQP *pi = p + 4u * s, *ph = p + 2u * s;
      const ie_rq_u32 slo = ie_rq_a32(p, 48u), shi = ie_rq_a32(p, 52u);
      const ie_rq_u32 s0 = slo & 0xFFFFu, s1 = slo >> 16u, s2 = shi & 0xFFFFu, s3 = shi >> 16u;
      const float d = ie_rq_h2f((s0 >> 12u) | ((s1 >> 8u) & 0x00F0u) | ((s2 >> 4u) & 0x0F00u) | (s3 & 0xF000u));
      const ie_rq_u32 pair = (s & 4u) != 0u ? shi : slo;
      const ie_rq_u32 scv = pair >> (16u * ((s >> 1u) & 1u)), qhb = ie_rq_a16(ph, 32u), qlo = ie_rq_a32(pi, 0u), sh = 6u * (s & 1u);
      const float dq0 = (d * (float)(int)(2u * ((scv >> sh) & 7u) + 1u)) * 0.125f;
      const float dq1 = (d * (float)(int)(2u * ((scv >> (sh + 3u)) & 7u) + 1u)) * 0.125f;
#if IE_TR8_IQ1M_FOLD
      ie_rq_u32 c0 = 0u, c1 = 0u;
      for (ie_rq_u32 l = 0; l < 4u; l++) {
        const ie_rq_u32 qh = l < 2u ? (qhb & 255u) : (qhb >> 8u);
        const ie_rq_u32 idx = ie_byte(qlo, l) | ((qh << (8u - 4u * (l & 1u))) & 0x700u);
        const ie_rq_u64 g = TR8_G1S(idx);
        const ie_rq_u32 tbl = (qh & ((l & 1u) != 0u ? 0x80u : 0x08u)) != 0u ? 0xF70007FFu : 0xF9000901u;
        const ie_rq_u32 wl = __builtin_amdgcn_perm(0u, tbl, (ie_rq_u32)g & 0x03030303u);
        const ie_rq_u32 wh = __builtin_amdgcn_perm(0u, tbl, (ie_rq_u32)(g >> 32u) & 0x03030303u);
        const ie_rq_u32 xl = TR8_X(2u * l), xh = TR8_X(2u * l + 1u);
        if (l < 2u) c0 = ie_dot4_ss(ie_dot4_ss(c0, wl, xl), wh, xh);
        else c1 = ie_dot4_ss(ie_dot4_ss(c1, wl, xl), wh, xh);
      }
      TR8_RET(dav * (dq0 * (float)(int)c0 + dq1 * (float)(int)c1));
#else
      int t0 = 0, t1 = 0;
      for (ie_rq_u32 l = 0; l < 4u; l++) {
        const ie_rq_u32 qh = l < 2u ? (qhb & 255u) : (qhb >> 8u);
        const ie_rq_u32 idx = ie_byte(qlo, l) | ((qh << (8u - 4u * (l & 1u))) & 0x700u);
        const ie_rq_u64 g = TR8_G1S(idx);
        const ie_rq_u32 xl = TR8_X(2u * l), xh = TR8_X(2u * l + 1u);
        const ie_rq_u32 id = ie_dot4_ss(ie_dot4_ss(0u, (ie_rq_u32)g, xl), (ie_rq_u32)(g >> 32u), xh);
        /* the delta term: the 8 activations' sum, or its negative when the row's sign bit is set; neither depends on
         * the row, so the rows of a wave share both and each row only selects */
        const ie_rq_u32 s8 = ie_dot4_ss(ie_dot4_ss(0u, 0x01010101u, xl), 0x01010101u, xh);
        const ie_rq_u32 n8 = 0u - s8; /* the dot with bytes of -1: the same int, without two more dots */
        const ie_rq_u32 dl = (qh & ((l & 1u) != 0u ? 0x80u : 0x08u)) != 0u ? n8 : s8;
        const int tg = (int)((id << 3u) + dl);
        if (l < 2u) t0 += tg; else t1 += tg;
      }
      TR8_RET(dav * (dq0 * (float)t0 + dq1 * (float)t1));
#endif
    }
    default: return 0.0f;
  }
}
#undef TR8_RET
#undef TR8_X
/* A pointer that is the same in every lane, said so (readfirstlane of both halves): the compiler then keeps it in
 * scalar registers and every load is that base plus one 32-bit lane offset, instead of chains of 64-bit vector adds
 * (one per token) in each trip of the loop. The builtin returns a signed int: each half goes into a u32 before it is
 * widened, or a low half with bit 31 set would fill the high half with ones. */
static inline __attribute__((always_inline)) const G u8 *tr8_uniform(const G u8 *p) {
  const unsigned long v = (unsigned long)p;
  const u32 lo = (u32)__builtin_amdgcn_readfirstlane((u32)v), hi = (u32)__builtin_amdgcn_readfirstlane((u32)(v >> 32u));
  return (const G u8 *)(((unsigned long)hi << 32u) | (unsigned long)lo);
}
static inline __attribute__((always_inline)) void gemv_raw_tr8_run(const u32 type, const int mt, const u32 NTOK, const u32 R, u32 wg,
                                                                   const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb,
                                                                   u32 xs, u32 ys, u32 rs, const G float *bias, const G float *res) {
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5); /* the wave's number in the workgroup, said to be the same in every lane */
  const u32 r0 = ie_gemv_row(wg, wv) * R, l = lane();
  if (r0 >= rows) return;
  const u32 mb = nb / 8u, bs = ie_rq_bs(type);
  const G u8 *row[4];
  _Pragma("unroll") for (u32 k = 0; k < R; k++) row[k] = tr8_uniform(q + (unsigned long)(r0 + k < rows ? r0 + k : r0) * mb * bs); /* past the last row: repeat the first */
  const G u8 *xt[16];
  _Pragma("unroll") for (u32 t = 0; t < NTOK; t++) xt[t] = tr8_uniform(xq + (unsigned long)t * xs);
  float a[4][16];
  _Pragma("unroll") for (u32 k = 0; k < R; k++)
    _Pragma("unroll") for (u32 t = 0; t < NTOK; t++) a[k][t] = 0.0f;
#ifdef IE_DBG_EXTRAX /* debug: IE_DBG_EXTRAX more 16-byte activation loads per trip (cache hits, one xor each) */
  u32 dbgx = 0u;
#endif
  for (u32 b = l; b < nb; b += 32u) { /* the row kernel's sub-block map: all 32 lanes busy */
    const u32 qb = b >> 3u;
#ifdef IE_DBG_EXTRAX
    _Pragma("unroll") for (u32 e = 0; e < IE_DBG_EXTRAX; e++) {
      const u32 bn = b + 1u + e / NTOK < nb ? b + 1u + e / NTOK : b - 1u - e / NTOK;
      const u32x4 v = *(const G u32x4 *)(xt[e % NTOK] + 32u * bn + 16u * ((e / NTOK) & 1u));
      dbgx ^= v.x ^ v.w;
    }
#endif
    _Pragma("unroll") for (u32 g0 = 0; g0 < NTOK; g0 += TR8_GRPN(NTOK)) {
      _Pragma("unroll") for (u32 k = 0; k < R; k++) {
        const G u8 *blk = row[k] + (unsigned long)qb * bs;
        _Pragma("unroll") for (u32 t = g0; t < g0 + TR8_GRPN(NTOK) && t < NTOK; t++)
            a[k][t] += tr8_dot(type, mt, blk, xt[t], nb, b, 0);
      }
      if (TR8_GRPN(NTOK) < NTOK) __builtin_amdgcn_sched_barrier(0);
    }
  }
  /* Lane t (< NTOK) finishes token t with the sums of lane 0 (readlane), the same epilogue: the residual loads of
   * all rows and tokens are in flight together, where lane 0 alone waited for each before the next store. */
  float o[4];
  _Pragma("unroll") for (u32 k = 0; k < R; k++) {
    o[k] = 0.0f;
    _Pragma("unroll") for (u32 t = 0; t < NTOK; t++) {
      const float s = I2F(__builtin_amdgcn_readlane(F2I(wave_tree_f(a[k][t])), 0));
      if (l == t) o[k] = s;
    }
  }
#ifdef IE_DBG_EXTRAX
  if (dbgx == 0x9E3779B1u) o[0] = 0.0f;
#endif
  if (l < NTOK) {
    const G float *rt = res ? (const G float *)((const G u8 *)res + (unsigned long)l * rs) : res;
    G float *yt = (G float *)((G u8 *)y + (unsigned long)l * ys);
    _Pragma("unroll") for (u32 k = 0; k < R; k++)
      if (bias && r0 + k < rows) o[k] = o[k] + bias[r0 + k];
    _Pragma("unroll") for (u32 k = 0; k < R; k++)
      if (rt && r0 + k < rows) o[k] = rt[r0 + k] + o[k];
    _Pragma("unroll") for (u32 k = 0; k < R; k++)
      if (r0 + k < rows) yt[r0 + k] = o[k];
  }
}
/* ie_gemv_raw_<type>_tr9: the same sums with one row per lane. Workgroup wg = rows 32 wg .. 32 wg + 31 (lane l: row
 * 32 wg + l); wave w does the sub-blocks b of the _tr8 lanes v = w, w + 8, w + 16, w + 24 (b = v, v + 32, ...), in
 * the same order, so each (row, token) gets the same 32 lane sums p_v. Every lane of a wave works on the same
 * sub-block, so the activations are the same for all lanes (scalar loads) and only the weights take the vector memory
 * path, where the _tr8 kernels load each token's activations per lane. The sums then go through wave_tree_f's tree:
 * steps 1 and 2 (v + 16, v + 8) in the wave's registers, steps 3 to 5 (+ 4, + 2, + 1) over the 8 waves through LDS;
 * the same additions in the same order, so bitwise the outputs of the _tr8 kernels. */
LDS static float tr9_red[8u * 9u * 32u];
static inline __attribute__((always_inline)) void gemv_raw_tr9_run(const u32 type, const u32 NTOK, u32 wg, const G u8 *q,
                                                                   const G u8 *xq, G float *y, u32 rows, u32 nb, u32 xs,
                                                                   u32 ys, u32 rs, const G float *bias, const G float *res) {
  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 r0 = 32u * wg, r = r0 + l;
  const u32 mb = nb / 8u, bs = ie_rq_bs(type);
  const G u8 *rowp = q + (unsigned long)(r < rows ? r : r0) * mb * bs; /* past the last row: the first row again */
  const G u8 *xt[9];
  _Pragma("unroll") for (u32 t = 0; t < NTOK; t++) xt[t] = tr8_uniform(xq + (unsigned long)t * xs);
  float a[4][9];
  _Pragma("unroll") for (u32 j = 0; j < 4u; j++)
    _Pragma("unroll") for (u32 t = 0; t < NTOK; t++) a[j][t] = 0.0f;
  for (u32 k0 = 0; k0 < nb; k0 += 32u) {
    _Pragma("unroll") for (u32 j = 0; j < 4u; j++) {
      const u32 b = k0 + w + 8u * j; /* the same in every lane */
      if (b < nb) {
        const G u8 *blk = rowp + (unsigned long)(b >> 3u) * bs;
        _Pragma("unroll") for (u32 t = 0; t < NTOK; t++) a[j][t] += tr8_dot(type, 0, blk, xt[t], nb, b, 1);
      }
    }
  }
  /* IQ1_M / IQ1_S: the partial sums in their grid's LDS (16 KB) once every wave is done with it */
  LDSP float *red = (type == IE_RQ_IQ1_M || type == IE_RQ_IQ1_S) ? (LDSP float *)tr8_g1s : tr9_red;
  if (type == IE_RQ_IQ1_M || type == IE_RQ_IQ1_S) lbarrier();
  /* wave_tree_f, lane i: u_i = p_i + p_(i+16), w_i = u_i + u_(i+8); here i = w */
  _Pragma("unroll") for (u32 t = 0; t < NTOK; t++) red[(w * 9u + t) * 32u + l] = (a[0][t] + a[2][t]) + (a[1][t] + a[3][t]);
  lbarrier();
  for (u32 i = tid(); i < 32u * NTOK; i += NT) { /* x_i = w_i + w_(i+4), y_i = x_i + x_(i+2), the sum y_0 + y_1 */
    const u32 rr = i & 31u, t = i >> 5u, row = r0 + rr;
    float v[8];
    _Pragma("unroll") for (u32 x = 0; x < 8u; x++) v[x] = red[(x * 9u + t) * 32u + rr];
    const float x0 = v[0] + v[4], x1 = v[1] + v[5], x2 = v[2] + v[6], x3 = v[3] + v[7];
    float o = (x0 + x2) + (x1 + x3);
    if (row < rows) {
      if (bias) o = o + bias[row];
      if (res) o = ((const G float *)((const G u8 *)res + (unsigned long)t * rs))[row] + o;
      ((G float *)((G u8 *)y + (unsigned long)t * ys))[row] = o;
    }
  }
}
static inline __attribute__((always_inline)) void gemv_raw_tr8(const u32 type, const G u8 *q, const G u8 *xq, G float *y, u32 rows,
                                                               u32 nb, u32 T, const G float *bias, const G float *res, u32 xs,
                                                               u32 ys, u32 rs, const int nine) {
  if (type == IE_RQ_IQ2_XXS) { /* every thread of the workgroup, before anything returns */
    tr4_xxs_grid[tid()] = TR8_G2C(ie_rq_iq2xxs_grid[tid()]);
    if (tid() < 128u) tr4_ks[tid()] = ie_rq_ksigns64[tid()];
    lbarrier();
  }
#if IE_TR8_LDS
  if (type == IE_RQ_IQ2_XS || type == IE_RQ_IQ3_XXS) {
    if (tid() < 128u) tr4_ks[tid()] = ie_rq_ksigns64[tid()];
  }
  if (type == IE_RQ_IQ2_XS) for (u32 i = tid(); i < 512u; i += NT) tr8_g2xs[i] = TR8_G2C(ie_rq_iq2xs_grid[i]);
  if (type == IE_RQ_IQ2_S) for (u32 i = tid(); i < 1024u; i += NT) tr8_g2s[i] = TR8_G2C(ie_rq_iq2s_grid[i]);
  if (type == IE_RQ_IQ1_S || type == IE_RQ_IQ1_M) for (u32 i = tid(); i < 2048u; i += NT) tr8_g1s[i] = ie_rq_iq1s_grid[i];
  if (type == IE_RQ_IQ3_XXS) tr8_g3xxs[tid()] = ie_rq_iq3xxs_grid[tid()];
  if (type == IE_RQ_IQ3_S) for (u32 i = tid(); i < 512u; i += NT) tr8_g3s[i] = ie_rq_iq3s_grid[i];
  if (type != IE_RQ_IQ2_XXS && type != IE_RQ_Q2_K) lbarrier();
#endif
  const u32 wg = wgid();
#ifdef IE_TR8_ONLY /* debug: only the code of that many tokens (the registers of that case alone) */
  if (T == IE_TR8_ONLY && nine) gemv_raw_tr9_run(type, IE_TR8_ONLY, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res);
  else if (T == IE_TR8_ONLY) gemv_raw_tr8_run(type, 0, IE_TR8_ONLY, IE_TR8_ONLY == 8u ? TR8_R8(type) : IE_TR8_R6, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res);
  return;
#endif
  if (nine) switch (T) {
    case 5u: gemv_raw_tr9_run(type, 5u, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 6u: gemv_raw_tr9_run(type, 6u, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 7u: gemv_raw_tr9_run(type, 7u, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 8u: gemv_raw_tr9_run(type, 8u, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 9u: gemv_raw_tr9_run(type, 9u, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    default: return;
  }
  switch (T) {
    case 5u: gemv_raw_tr8_run(type, 0, 5u, IE_TR8_R5, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 6u: gemv_raw_tr8_run(type, 0, 6u, IE_TR8_R6, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 7u: gemv_raw_tr8_run(type, 0, 7u, IE_TR8_R7, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 8u: gemv_raw_tr8_run(type, 0, 8u, TR8_R8(type), wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 9u: gemv_raw_tr8_run(type, 0, 9u, IE_TR8_R9, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    default: return; /* the host sends 5 to 9 tokens to these kernels; fewer go to the _tr4 / _tr kernels */
  }
}
/* ie_gemv_raw_tr8_m holds the code of every type it takes, and takes the registers of the most demanding. IQ1_M
 * (174 registers at 2 rows per wave, 193 at 4) left fewer waves for all, and merged launches came out no faster
 * than separate ones; without it the kernel takes 124, as many as the largest single-type kernel, and IQ1_M
 * multiplies keep their own launch. IE_TR8M_TYPES: the types it takes (bit n: GGML type n); src/hip.c tr8m_type
 * offers no others. At 8 tokens IQ1_S takes 2 rows per wave here, as fast as its own kernel's 4 (the rows per wave
 * do not change a row's result); src/hip.c tr8m_rpw must agree. */
#ifndef IE_TR8M_TYPES
#define IE_TR8M_TYPES (1u << IE_RQ_Q2_K | 1u << IE_RQ_IQ2_XXS | 1u << IE_RQ_IQ2_XS | 1u << IE_RQ_IQ3_XXS | 1u << IE_RQ_IQ1_S | \
                       1u << IE_RQ_IQ2_S | 1u << IE_RQ_IQ1_M | 1u << IE_RQ_IQ3_S)
#endif
/* One matrix of ie_gemv_raw_tr8_m: its type's tables into tr8_tab (every thread, before anything returns), then the
 * rows of workgroup wg of that matrix, as ie_gemv_raw_<type>_tr8 does them. */
static inline __attribute__((always_inline)) void gemv_raw_tr8_mseg(const u32 type, u32 wg, const G u8 *q, const G u8 *xq, G float *y,
                                                                    u32 rows, u32 nb, u32 T, const G float *bias, const G float *res,
                                                                    u32 xs, u32 ys, u32 rs) {
  if (type == IE_RQ_IQ2_XXS || type == IE_RQ_IQ2_XS || type == IE_RQ_IQ3_XXS)
    if (tid() < 128u) tr8_tab[tid()] = ie_rq_ksigns64[tid()];
  if (type == IE_RQ_IQ2_XXS) tr8_tab[TR8_TG + tid()] = TR8_G2C(ie_rq_iq2xxs_grid[tid()]);
  if (type == IE_RQ_IQ2_XS) for (u32 i = tid(); i < 512u; i += NT) tr8_tab[TR8_TG + i] = TR8_G2C(ie_rq_iq2xs_grid[i]);
  if (type == IE_RQ_IQ2_S) for (u32 i = tid(); i < 1024u; i += NT) tr8_tab[TR8_TG + i] = TR8_G2C(ie_rq_iq2s_grid[i]);
  if (type == IE_RQ_IQ1_S || type == IE_RQ_IQ1_M) for (u32 i = tid(); i < 2048u; i += NT) tr8_tab[TR8_TG + i] = ie_rq_iq1s_grid[i];
  if (type == IE_RQ_IQ3_XXS) TR8_TAB32(tid()) = ie_rq_iq3xxs_grid[tid()];
  if (type == IE_RQ_IQ3_S) for (u32 i = tid(); i < 512u; i += NT) TR8_TAB32(i) = ie_rq_iq3s_grid[i];
  if (type != IE_RQ_Q2_K) lbarrier();
  switch (T) {
    case 5u: gemv_raw_tr8_run(type, 1, 5u, IE_TR8_R5, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 6u: gemv_raw_tr8_run(type, 1, 6u, IE_TR8_R6, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 7u: gemv_raw_tr8_run(type, 1, 7u, IE_TR8_R7, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 8u: gemv_raw_tr8_run(type, 1, 8u, IE_TR8_R8, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 9u: gemv_raw_tr8_run(type, 1, 9u, IE_TR8_R9, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    default: return;
  }
}
/* Up to 3 weight multiplies of the same input in one launch (the gate and up of a layer, its q / k / v or qkv / z
 * projections), each matrix of any raw type: workgroups 0 .. g1 - 1 do matrix 0, g1 .. g2 - 1 matrix 1, g2 on matrix
 * 2 (two matrices: rows2 = 0 and type2 = 0, the workgroups past g2 do nothing). Every row goes through the code of
 * ie_gemv_raw_<type>_tr8, so each output is bitwise the same; one launch instead of several saves the GPU's gap
 * between kernels and the tail of each. The host gives each matrix ceil(rows / (8 * tr8m_rpw)) workgroups. */
#define TR8M_SEG(t) \
  case t: \
    if ((IE_TR8M_TYPES) >> t & 1u) gemv_raw_tr8_mseg(t, wg, q, xq, y, rows, nb, T, bias, res, xs, ys, rs); \
    return;
KERNEL ie_gemv_raw_tr8_m(const G u8 *xq, u32 nb, u32 T, u32 xs, u32 g1, u32 g2,
                         const G u8 *q0, G float *y0, u32 rows0, u32 type0, u32 ys0, const G float *res0, u32 rs0, const G float *bias0,
                         const G u8 *q1, G float *y1, u32 rows1, u32 type1, u32 ys1, const G float *res1, u32 rs1, const G float *bias1,
                         const G u8 *q2, G float *y2, u32 rows2, u32 type2, u32 ys2, const G float *res2, u32 rs2, const G float *bias2) {
  const u32 w = wgid(), s = w < g1 ? 0u : w < g2 ? 1u : 2u, wg = w - (s == 0u ? 0u : s == 1u ? g1 : g2);
  const G u8 *q = s == 0u ? q0 : s == 1u ? q1 : q2;
  G float *y = s == 0u ? y0 : s == 1u ? y1 : y2;
  const G float *res = s == 0u ? res0 : s == 1u ? res1 : res2, *bias = s == 0u ? bias0 : s == 1u ? bias1 : bias2;
  const u32 rows = s == 0u ? rows0 : s == 1u ? rows1 : rows2, type = s == 0u ? type0 : s == 1u ? type1 : type2;
  const u32 ys = s == 0u ? ys0 : s == 1u ? ys1 : ys2, rs = s == 0u ? rs0 : s == 1u ? rs1 : rs2;
  switch (type) {
    TR8M_SEG(IE_RQ_Q2_K)
    TR8M_SEG(IE_RQ_IQ2_XXS)
    TR8M_SEG(IE_RQ_IQ2_XS)
    TR8M_SEG(IE_RQ_IQ3_XXS)
    TR8M_SEG(IE_RQ_IQ1_S)
    TR8M_SEG(IE_RQ_IQ2_S)
    TR8M_SEG(IE_RQ_IQ1_M)
    TR8M_SEG(IE_RQ_IQ3_S)
    case IE_RQ_Q8_0:
      gemv_q8q8_mseg(wg, q, xq, y, rows, nb, T, bias, res, xs, ys, rs);
      return;
    default: return;
  }
}
KERNEL ie_gemv_raw_q2k_tr8(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                           const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                           float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  gemv_raw_tr8(IE_RQ_Q2_K, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, 0);
}
KERNEL ie_gemv_raw_iq2xxs_tr8(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                           const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                           float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  gemv_raw_tr8(IE_RQ_IQ2_XXS, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, 0);
}
KERNEL ie_gemv_raw_iq2xs_tr8(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                           const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                           float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  gemv_raw_tr8(IE_RQ_IQ2_XS, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, 0);
}
KERNEL ie_gemv_raw_iq3xxs_tr8(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                           const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                           float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  gemv_raw_tr8(IE_RQ_IQ3_XXS, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, 0);
}
KERNEL ie_gemv_raw_iq1s_tr8(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                           const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                           float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  gemv_raw_tr8(IE_RQ_IQ1_S, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, 0);
}
KERNEL ie_gemv_raw_iq2s_tr8(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                           const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                           float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  gemv_raw_tr8(IE_RQ_IQ2_S, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, 0);
}
KERNEL ie_gemv_raw_iq1m_tr8(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                           const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                           float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  gemv_raw_tr8(IE_RQ_IQ1_M, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, 0);
}
KERNEL ie_gemv_raw_iq3s_tr8(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                           const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq,
                           float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
  (void)nw, (void)ny, (void)nq, (void)eps, (void)count; /* no fused norm */
  gemv_raw_tr8(IE_RQ_IQ3_S, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, 0);
}
static inline __attribute__((always_inline)) void gemv_raw_tr16(const u32 type, const G u8 *q, const G u8 *xq, G float *y, u32 rows,
                                                                u32 nb, u32 T, const G float *bias, const G float *res, u32 xs,
                                                                u32 ys, u32 rs) {
  if (type == IE_RQ_IQ2_XXS) {
    tr4_xxs_grid[tid()] = TR8_G2C(ie_rq_iq2xxs_grid[tid()]);
    if (tid() < 128u) tr4_ks[tid()] = ie_rq_ksigns64[tid()];
    lbarrier();
  }
#if IE_TR8_LDS
  if (type == IE_RQ_IQ2_XS || type == IE_RQ_IQ3_XXS) {
    if (tid() < 128u) tr4_ks[tid()] = ie_rq_ksigns64[tid()];
  }
  if (type == IE_RQ_IQ2_XS) for (u32 i = tid(); i < 512u; i += NT) tr8_g2xs[i] = TR8_G2C(ie_rq_iq2xs_grid[i]);
  if (type == IE_RQ_IQ2_S) for (u32 i = tid(); i < 1024u; i += NT) tr8_g2s[i] = TR8_G2C(ie_rq_iq2s_grid[i]);
  if (type == IE_RQ_IQ1_S || type == IE_RQ_IQ1_M) for (u32 i = tid(); i < 2048u; i += NT) tr8_g1s[i] = ie_rq_iq1s_grid[i];
  if (type == IE_RQ_IQ3_XXS) tr8_g3xxs[tid()] = ie_rq_iq3xxs_grid[tid()];
  if (type == IE_RQ_IQ3_S) for (u32 i = tid(); i < 512u; i += NT) tr8_g3s[i] = ie_rq_iq3s_grid[i];
  if (type != IE_RQ_IQ2_XXS && type != IE_RQ_Q2_K) lbarrier();
#endif
  const u32 wg = wgid();
  switch (T) {
    case 10u: gemv_raw_tr8_run(type, 0, 10u, IE_TR16_R, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 11u: gemv_raw_tr8_run(type, 0, 11u, IE_TR16_R, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 12u: gemv_raw_tr8_run(type, 0, 12u, IE_TR16_R, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 13u: gemv_raw_tr8_run(type, 0, 13u, IE_TR16_R, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 14u: gemv_raw_tr8_run(type, 0, 14u, IE_TR16_R, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 15u: gemv_raw_tr8_run(type, 0, 15u, IE_TR16_R, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 16u: gemv_raw_tr8_run(type, 0, 16u, IE_TR16_R, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    default: return;
  }
}
static inline __attribute__((always_inline)) void gemv_raw_tr16_mseg(const u32 type, u32 wg, const G u8 *q, const G u8 *xq, G float *y,
                                                                     u32 rows, u32 nb, u32 T, const G float *bias, const G float *res,
                                                                     u32 xs, u32 ys, u32 rs) {
  if (type == IE_RQ_IQ2_XXS || type == IE_RQ_IQ2_XS || type == IE_RQ_IQ3_XXS)
    if (tid() < 128u) tr8_tab[tid()] = ie_rq_ksigns64[tid()];
  if (type == IE_RQ_IQ2_XXS) tr8_tab[TR8_TG + tid()] = TR8_G2C(ie_rq_iq2xxs_grid[tid()]);
  if (type == IE_RQ_IQ2_XS) for (u32 i = tid(); i < 512u; i += NT) tr8_tab[TR8_TG + i] = TR8_G2C(ie_rq_iq2xs_grid[i]);
  if (type == IE_RQ_IQ2_S) for (u32 i = tid(); i < 1024u; i += NT) tr8_tab[TR8_TG + i] = TR8_G2C(ie_rq_iq2s_grid[i]);
  if (type == IE_RQ_IQ1_S || type == IE_RQ_IQ1_M) for (u32 i = tid(); i < 2048u; i += NT) tr8_tab[TR8_TG + i] = ie_rq_iq1s_grid[i];
  if (type == IE_RQ_IQ3_XXS) TR8_TAB32(tid()) = ie_rq_iq3xxs_grid[tid()];
  if (type == IE_RQ_IQ3_S) for (u32 i = tid(); i < 512u; i += NT) TR8_TAB32(i) = ie_rq_iq3s_grid[i];
  if (type != IE_RQ_Q2_K) lbarrier();
  switch (T) {
    case 10u: gemv_raw_tr8_run(type, 1, 10u, IE_TR16_R, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 11u: gemv_raw_tr8_run(type, 1, 11u, IE_TR16_R, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 12u: gemv_raw_tr8_run(type, 1, 12u, IE_TR16_R, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 13u: gemv_raw_tr8_run(type, 1, 13u, IE_TR16_R, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 14u: gemv_raw_tr8_run(type, 1, 14u, IE_TR16_R, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 15u: gemv_raw_tr8_run(type, 1, 15u, IE_TR16_R, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    case 16u: gemv_raw_tr8_run(type, 1, 16u, IE_TR16_R, wg, q, xq, y, rows, nb, xs, ys, rs, bias, res); return;
    default: return;
  }
}
#define TR16M_SEG(t) \
  case t: \
    if ((IE_TR8M_TYPES) >> t & 1u) gemv_raw_tr16_mseg(t, wg, q, xq, y, rows, nb, T, bias, res, xs, ys, rs); \
    return;
KERNEL ie_gemv_raw_tr16_m(const G u8 *xq, u32 nb, u32 T, u32 xs, u32 g1, u32 g2,
                          const G u8 *q0, G float *y0, u32 rows0, u32 type0, u32 ys0, const G float *res0, u32 rs0, const G float *bias0,
                          const G u8 *q1, G float *y1, u32 rows1, u32 type1, u32 ys1, const G float *res1, u32 rs1, const G float *bias1,
                          const G u8 *q2, G float *y2, u32 rows2, u32 type2, u32 ys2, const G float *res2, u32 rs2, const G float *bias2) {
  const u32 w = wgid(), s = w < g1 ? 0u : w < g2 ? 1u : 2u, wg = w - (s == 0u ? 0u : s == 1u ? g1 : g2);
  const G u8 *q = s == 0u ? q0 : s == 1u ? q1 : q2;
  G float *y = s == 0u ? y0 : s == 1u ? y1 : y2;
  const G float *res = s == 0u ? res0 : s == 1u ? res1 : res2, *bias = s == 0u ? bias0 : s == 1u ? bias1 : bias2;
  const u32 rows = s == 0u ? rows0 : s == 1u ? rows1 : rows2, type = s == 0u ? type0 : s == 1u ? type1 : type2;
  const u32 ys = s == 0u ? ys0 : s == 1u ? ys1 : ys2, rs = s == 0u ? rs0 : s == 1u ? rs1 : rs2;
  switch (type) {
    TR16M_SEG(IE_RQ_Q2_K)
    TR16M_SEG(IE_RQ_IQ2_XXS)
    TR16M_SEG(IE_RQ_IQ2_XS)
    TR16M_SEG(IE_RQ_IQ3_XXS)
    TR16M_SEG(IE_RQ_IQ1_S)
    TR16M_SEG(IE_RQ_IQ2_S)
    TR16M_SEG(IE_RQ_IQ1_M)
    TR16M_SEG(IE_RQ_IQ3_S)
    case IE_RQ_Q8_0:
      gemv_q8q8_mseg(wg, q, xq, y, rows, nb, T, bias, res, xs, ys, rs);
      return;
    default: return;
  }
}
#undef TR16M_SEG
#define TR16_KERNEL(name, t)                                                                                        \
  KERNEL name(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T, const G float *bias,              \
              const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys,\
              u32 rs) {                                                                                             \
    (void)nw, (void)ny, (void)nq, (void)eps, (void)count;                                                           \
    gemv_raw_tr16(t, q, xq, y, rows, nb, T, bias, res, xs, ys, rs);                                                 \
  }
TR16_KERNEL(ie_gemv_raw_q2k_tr16, IE_RQ_Q2_K)
TR16_KERNEL(ie_gemv_raw_iq2xxs_tr16, IE_RQ_IQ2_XXS)
TR16_KERNEL(ie_gemv_raw_iq2xs_tr16, IE_RQ_IQ2_XS)
TR16_KERNEL(ie_gemv_raw_iq3xxs_tr16, IE_RQ_IQ3_XXS)
TR16_KERNEL(ie_gemv_raw_iq1s_tr16, IE_RQ_IQ1_S)
TR16_KERNEL(ie_gemv_raw_iq2s_tr16, IE_RQ_IQ2_S)
TR16_KERNEL(ie_gemv_raw_iq1m_tr16, IE_RQ_IQ1_M)
TR16_KERNEL(ie_gemv_raw_iq3s_tr16, IE_RQ_IQ3_S)
#undef TR16_KERNEL
/* The _tr9 kernels (gemv_raw_tr9_run): the same arguments and outputs, ceil(rows / 32) workgroups. Built only with
 * IE_TR9=1 (tools/raw_gemv_bench.py -tr9); the engine does not use them. Measured 2026-11-04 against the _tr8
 * kernels: bitwise the same everywhere; 6 tokens -17% to +10% by type and shape (iq2s, iq2xs, iq1m with many rows
 * ahead; iq2xxs, iq3xxs, q2k behind), 8 tokens 6-94% slower (the activations of 8 tokens overflow the scalar
 * registers), so no gain worth a per-shape choice. */
#ifndef IE_TR9
#define IE_TR9 0
#endif
#if IE_TR9
#define TR9_KERNEL(name, t)                                                                                        \
  KERNEL name(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T, const G float *bias,             \
              const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, \
              u32 rs) {                                                                                              \
    (void)nw, (void)ny, (void)nq, (void)eps, (void)count;                                                           \
    gemv_raw_tr8(t, q, xq, y, rows, nb, T, bias, res, xs, ys, rs, 1);                                               \
  }
TR9_KERNEL(ie_gemv_raw_q2k_tr9, IE_RQ_Q2_K)
TR9_KERNEL(ie_gemv_raw_iq2xxs_tr9, IE_RQ_IQ2_XXS)
TR9_KERNEL(ie_gemv_raw_iq2xs_tr9, IE_RQ_IQ2_XS)
TR9_KERNEL(ie_gemv_raw_iq3xxs_tr9, IE_RQ_IQ3_XXS)
TR9_KERNEL(ie_gemv_raw_iq1s_tr9, IE_RQ_IQ1_S)
TR9_KERNEL(ie_gemv_raw_iq2s_tr9, IE_RQ_IQ2_S)
TR9_KERNEL(ie_gemv_raw_iq1m_tr9, IE_RQ_IQ1_M)
TR9_KERNEL(ie_gemv_raw_iq3s_tr9, IE_RQ_IQ3_S)
#endif
/* MAT_IQ4XS for 5 to 16 tokens, 2 rows per wave: as ie_gemv_iq4xsq8_tr (the same blocks per lane in the same order,
 * the same fma per row and token, so the sums are bitwise the same), with each token's activation loads shared by the
 * wave's two rows, the table-mapped weight words made once per row and block, and lane t finishing token t (as
 * gemv_raw_tr8_run). rows / 16 workgroups; src/hip.c launches it from the _tr arguments (IE_IQ4_TR8=0: ie_gemv_iq4xsq8_tr). */
#define IQ4_TR8_R 2u
#define IQ4_TR8_T 9u
#define IQ4_TR16_T 16u
static inline __attribute__((always_inline)) void gemv_iq4xs_tr_run(const u32 MAXT, const G u32 *qw, const G u16 *qs,
                                                                    const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                                                                    const G float *bias, const G float *res, u32 xs,
                                                                    u32 ys, u32 rs) {
#pragma clang fp contract(off)
  const u32 r0 = ie_gemv_row(wgid(), (u32)__builtin_amdgcn_readfirstlane(tid() >> 5)) * IQ4_TR8_R, l = lane();
  if (r0 >= rows || T == 0u) return;
  const G u16 *smr[IQ4_TR8_R];
  const G f16 *ddr[IQ4_TR8_R];
  u32 rr[IQ4_TR8_R];
  float acc[IQ4_TR8_R][IQ4_TR16_T];
#pragma clang loop unroll(full)
  for (u32 q = 0; q < IQ4_TR8_R; q++) {
    rr[q] = r0 + q < rows ? r0 + q : r0; /* a row past the end: row r0 again, not stored */
    smr[q] = qs + ie_q4k_sm(rr[q], 0u, nb);
    ddr[q] = (const G f16 *)qs + ie_q4k_dd(rows, rr[q], 0u, nb);
#pragma clang loop unroll(full)
    for (u32 j = 0; j < MAXT; j++) acc[q][j] = 0.0f;
  }
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
    u32x4 w4[GEMV_U][IQ4_TR8_R];
    u32 sm[GEMV_U][IQ4_TR8_R], bk[GEMV_U];
    float dv[GEMV_U][IQ4_TR8_R];
    int ok[GEMV_U];
    for (u32 k = 0; k < GEMV_U; k++) { /* the weight loads of the step first */
      ok[k] = ie_lane_blk(l, t + k) < nb;
      bk[k] = ok[k] ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
      for (u32 q = 0; q < IQ4_TR8_R; q++) {
        w4[k][q] = *(const G u32x4 *)(qw + ie_q4_dst_word(rr[q], bk[k], 0u, nb));
        sm[k][q] = smr[q][bk[k]];
        dv[k][q] = (float)ddr[q][(bk[k] >> 3) * 2u];
      }
    }
    for (u32 k = 0; k < GEMV_U; k++) {
      if (!ok[k]) continue; /* past nb (only some lanes, at the end): the sums stay */
      const u32 b = bk[k];
      u32 wm[IQ4_TR8_R][8];
      float sc[IQ4_TR8_R];
      for (u32 q = 0; q < IQ4_TR8_R; q++) {
        const u32 wq[4] = {w4[k][q].x, w4[k][q].y, w4[k][q].z, w4[k][q].w};
        for (u32 w = 0; w < 4u; w++) {
          wm[q][2u * w] = ie_iq4_map(wq[w] & 0x0F0F0F0Fu);
          wm[q][2u * w + 1u] = ie_iq4_map((wq[w] >> 4u) & 0x0F0F0F0Fu);
        }
        sc[q] = dv[k][q] * (float)((int)(sm[k][q] & 63u) - 32);
      }
#pragma clang loop unroll(full)
      for (u32 j = 0; j < MAXT; j++) {
        if (j < T) { /* uniform */
          const G u8 *xj = xq + (unsigned long)j * xs;
          const G u32 *aw = (const G u32 *)xj;
          const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u);
          const float da = ((const G float *)(xj + 32u * nb))[b];
          const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
          for (u32 q = 0; q < IQ4_TR8_R; q++) {
            u32 d = 0u; /* as ie_iq4_dot */
            for (u32 w = 0; w < 4u; w++) d = ie_dot4_ss(ie_dot4_ss(d, wm[q][2u * w], av[w]), wm[q][2u * w + 1u], av[4u + w]);
            acc[q][j] = __builtin_fmaf(sc[q] * (float)(int)d, da, acc[q][j]);
          }
        }
      }
    }
  }
  /* lane t (< T) finishes token t with the sums of lane 0, the epilogue as epilogue(): bias, then the residual */
  float o[IQ4_TR8_R];
#pragma clang loop unroll(full)
  for (u32 q = 0; q < IQ4_TR8_R; q++) {
    o[q] = 0.0f;
#pragma clang loop unroll(full)
    for (u32 j = 0; j < MAXT; j++) {
      if (j < T) { /* uniform */
        const float s = I2F(__builtin_amdgcn_readlane(F2I(wave_tree_f(acc[q][j])), 0));
        if (l == j) o[q] = s;
      }
    }
  }
  if (l < T) {
    const G float *rt = res ? (const G float *)((const G u8 *)res + (unsigned long)l * rs) : res;
    G float *yt = (G float *)((G u8 *)y + (unsigned long)l * ys);
    for (u32 q = 0; q < IQ4_TR8_R; q++) {
      if (r0 + q >= rows) continue;
      float v = o[q];
      if (bias) v = v + bias[r0 + q];
      if (rt) v = rt[r0 + q] + v;
      yt[r0 + q] = v;
    }
  }
}
KERNEL ie_gemv_iq4xsq8_tr8(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                           const G float *bias, const G float *res, u32 xs, u32 ys, u32 rs) {
  gemv_iq4xs_tr_run(IQ4_TR8_T, qw, qs, xq, y, rows, nb, T, bias, res, xs, ys, rs);
}
KERNEL ie_gemv_iq4xsq8_tr16(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                            const G float *bias, const G float *res, u32 xs, u32 ys, u32 rs) {
  gemv_iq4xs_tr_run(IQ4_TR16_T, qw, qs, xq, y, rows, nb, T, bias, res, xs, ys, rs);
}
#include "ie_raw_swiglu.inc"
#endif /* !IE_MEGA */

/* Prompt chunks of raw blocks on the matrix cores (not in the megakernel). */
#include "ie_gemm_raw.inc"
#include "ie_raw_t16.inc"
/* Prompt-chunk attention at long positions (not in the megakernel; it reads fp16 or q8 KV, so the 4-bit
 * build leaves it out and prompt chunks use ie_attn_fa2 there). */
#if !defined(IE_KV_Q4) && !defined(IE_SALUKI_ATTN_MP)
#include "ie_attn_pl.inc"
#endif
#include "ie_gdn_wmma.inc"
#include "ie_norm_fuse.inc"

/* Overflow-free tanhf for inkling2 gates, SITU softcap, and logit softcapping (no libcall under -nogpulib). */
static inline float ie_tanhf_s(float x) {
  const float ax = x >= 0.0f ? x : -x;
  if (ax > 20.0f) return x >= 0.0f ? 1.0f : -1.0f;
  if (ax < 1e-3f) return x * (1.0f - (1.0f / 3.0f) * x * x);
  const float e = __builtin_expf(-2.0f * ax);
  const float t = (1.0f - e) / (1.0f + e);
  return x >= 0.0f ? t : -t;
}

static inline const G u8 *rq_ptr_u(const G u8 *p) {
  const unsigned long v = (unsigned long)p;
  const u32 lo = (u32)__builtin_amdgcn_readfirstlane((u32)v), hi = (u32)__builtin_amdgcn_readfirstlane((u32)(v >> 32u));
  return (const G u8 *)(((unsigned long)hi << 32u) | (unsigned long)lo);
}

/* Batched residual add over T tokens (grid y). Safe when c aliases a or b. */
KERNEL ie_add_t(const G float *a, const G float *b, G float *c, u32 n, u32 as, u32 bs, u32 cs) {
  const u32 i = wgid() * NT + tid();
  if (i >= n) return;
  a = TOKC(a, as), b = TOKC(b, bs), c = TOK(c, cs);
  c[i] = a[i] + b[i];
}

/* Batched KV cache store over T tokens (grid y), supporting ring kv_len and fp16/f32/q8/q4 KV. */
KERNEL ie_kv_store_t(const G float *k, const G float *v, G kvt *kc, G kvt *vc, u32 hd, u32 n_kv, u32 pos0, u32 kv_len,
                     u32 ks, u32 vs) {
  const u32 t = tokid(), p = (pos0 + t) % kv_len;
  k = TOKC(k, ks), v = TOKC(v, vs);
#ifdef IE_KV_QUANT
  const u32 b = wgid() * 8u + (tid() >> 5), nb_h = hd >> 5;
  const u32 kh = b / nb_h, j = (b % nb_h) * 32u + lane();
  if (kh < n_kv) {
    const u32 kvd = n_kv * hd;
    (void)kvd;
    kvq_store(k[kh * hd + j], (G u8 *)KVROWP(kc, p, kh), hd, j);
    kvq_store(v[kh * hd + j], (G u8 *)KVROWP(vc, p, kh), hd, j);
  }
#else
  const u32 i = wgid() * NT + tid(), n = n_kv * hd;
  if (i < n) {
    const unsigned long off = (unsigned long)p * n + i;
    kc[off] = k[i];
    vc[off] = v[i];
  }
#endif
}

KERNEL ie_sconv_ring_store(const G float *a, G float *ring, u32 n, u32 pos0, u32 T, u32 as) {
  const u32 i = wgid() * NT + tid();
  if (i >= n) return;
  const u32 k0 = T > 4u ? T - 4u : 0u;
  for (u32 t = k0; t < T; t++) {
    const G float *at = (const G float *)((const G u8 *)a + (unsigned long)t * as);
    ring[i * 4u + ((pos0 + t) & 3u)] = at[i];
  }
}

/* Inkling2 final logit softcap + unpadded mask: j < unpadded ? cap * tanh(a[j] / cap) : -1e30f (grid y = T). */
KERNEL ie_logit_cap(const G float *a, G float *b, u32 n, u32 unpadded, float cap, u32 as, u32 bs) {
  const u32 i = wgid() * NT + tid();
  if (i >= n) return;
  a = TOKC(a, as), b = TOK(b, bs);
  if (i < unpadded) {
    const float x = a[i];
    b[i] = cap > 0.0f ? cap * ie_tanhf_s(x / cap) : x;
  } else {
    b[i] = -1e30f;
  }
}

#ifndef IE_MEGA
/* Stage the lookup table for a raw GGUF IQ/K type into a 16 KiB LDS buffer (caller runs lbarrier). */
static inline void raw_stage_lut_lds(u32 type, LDSP ie_rq_u64 *lut) {
  switch (type) {
    case IE_RQ_IQ2_XXS:
      lut[tid()] = ie_rq_iq2xxs_grid[tid()];
      if (tid() < 128u) lut[256u + tid()] = ie_rq_ksigns64[tid()];
      break;
    case IE_RQ_IQ2_XS:
      for (u32 i = tid(); i < 512u; i += NT) lut[i] = ie_rq_iq2xs_grid[i];
      if (tid() < 128u) lut[512u + tid()] = ie_rq_ksigns64[tid()];
      break;
    case IE_RQ_IQ2_S:
      for (u32 i = tid(); i < 1024u; i += NT) lut[i] = ie_rq_iq2s_grid[i];
      break;
    case IE_RQ_IQ3_XXS:
      ((LDSP u32 *)lut)[tid()] = ie_rq_iq3xxs_grid[tid()];
      if (tid() < 128u) lut[128u + tid()] = ie_rq_ksigns64[tid()];
      break;
    case IE_RQ_IQ3_S:
      for (u32 i = tid(); i < 512u; i += NT) ((LDSP u32 *)lut)[i] = ie_rq_iq3s_grid[i];
      break;
    case IE_RQ_IQ1_S:
    case IE_RQ_IQ1_M:
      for (u32 i = tid(); i < 2048u; i += NT) lut[i] = ie_rq_iq1s_grid[i];
      break;
    default: break;
  }
}

/* One decode token (K active expert slots in grid.y) for MAT_RAWQ expert weights. */
KERNEL ie_gexp_raw(const G u8 *qw, const G u8 *xq, G float *y, u32 rows, u32 nb, const G u32 *route, u32 q_bytes,
                   u32 astr, u32 bstr, u32 type) {
  static LDS ie_rq_u64 lut[2048];
  type = (u32)__builtin_amdgcn_readfirstlane(type);
  raw_stage_lut_lds(type, lut);
  lbarrier();
  const u32 s = tokid(), e = route[s];
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 r = ie_gemv_row(wgid(), wv), l = lane();
  if (r < rows) {
    const u32 mb = nb / 8u, bs = ie_rq_bs(type);
    const G u8 *row = rq_ptr_u(qw + (unsigned long)e * q_bytes + (unsigned long)r * mb * bs);
    const G u8 *xp = rq_ptr_u(xq + (unsigned long)s * astr);
    float acc = 0.0f;
    if (type == IE_RQ_IQ3_XXS) {
      for (u32 b = l; b < nb; b += 32u) acc += ie_rq_dot_row_lds(IE_RQ_IQ3_XXS, row, xp, nb, b, lut);
    } else {
      for (u32 b = l; b < nb; b += 32u) acc += ie_rq_dot_row_lds(type, row, xp, nb, b, lut);
    }
    acc = wave_tree_f(acc);
    if (l == 0u) ((G float *)((G u8 *)y + (unsigned long)s * bstr))[r] = acc;
  }
}

/* Grouped multi-token MAT_RAWQ expert GEMV fallback (any nb % 8 == 0, low VGPRs). 16 rows/WG, 4 pairs/step. */
KERNEL ie_gexp_raw_t(const G u8 *qw, const G u8 *X, G float *Y, u32 rows, u32 nb, const G u32 *list,
                     const G u32 *offs, u32 np, u32 q_bytes, u32 xs, u32 ys, u32 xdiv, u32 type,
                     const G float *res, u32 rs) {
  static LDS ie_rq_u64 lut[2048];
  type = (u32)__builtin_amdgcn_readfirstlane(type);
  const u32 e = tokid(), p0 = list ? offs[e] : 0u, p1 = list ? offs[e + 1u] : np, rb = wgid() * 16u;
  if (p0 >= p1 || rb >= rows) return;
  raw_stage_lut_lds(type, lut);
  lbarrier();
  const u32 l = lane(), wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 r0 = rb + 2u * wv, r1 = r0 + 1u;
  if (r0 >= rows) return;
  const u32 mb = nb / 8u, bs = ie_rq_bs(type);
  const G u8 *base = qw + (unsigned long)e * q_bytes;
  const G u8 *row0 = rq_ptr_u(base + (unsigned long)r0 * mb * bs);
  const G u8 *row1 = rq_ptr_u(base + (unsigned long)(r1 < rows ? r1 : r0) * mb * bs);
  for (u32 pb = p0; pb < p1; pb += 4u) {
    const u32 nj = p1 - pb < 4u ? p1 - pb : 4u;
    u32 pid[4];
    const G u8 *xp[4];
#pragma unroll
    for (u32 j = 0; j < 4u; j++) {
      const u32 pj = pb + (j < nj ? j : 0u);
      pid[j] = list ? list[pj] : pj;
      xp[j] = rq_ptr_u(X + (unsigned long)(pid[j] / xdiv) * xs);
    }
    float a0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, a1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (type == IE_RQ_IQ3_XXS) {
      for (u32 b = l; b < nb; b += 32u) {
#pragma unroll
        for (u32 j = 0; j < 4u; j++) {
          if (j < nj) {
            a0[j] += ie_rq_dot_row_lds(IE_RQ_IQ3_XXS, row0, xp[j], nb, b, lut);
            if (r1 < rows) a1[j] += ie_rq_dot_row_lds(IE_RQ_IQ3_XXS, row1, xp[j], nb, b, lut);
          }
        }
      }
    } else {
      for (u32 b = l; b < nb; b += 32u) {
#pragma unroll
        for (u32 j = 0; j < 4u; j++) {
          if (j < nj) {
            a0[j] += ie_rq_dot_row_lds(type, row0, xp[j], nb, b, lut);
            if (r1 < rows) a1[j] += ie_rq_dot_row_lds(type, row1, xp[j], nb, b, lut);
          }
        }
      }
    }
#pragma unroll
    for (u32 j = 0; j < 4u; j++) {
      if (j < nj) {
        const float v0 = wave_tree_f(a0[j]), v1 = r1 < rows ? wave_tree_f(a1[j]) : 0.0f;
        if (l == 0u) {
          G float *yp = (G float *)((G u8 *)Y + (unsigned long)pid[j] * ys);
          const G float *rp = res ? (const G float *)((const G u8 *)res + (unsigned long)pid[j] * rs) : 0;
          yp[r0] = rp ? rp[r0] + v0 : v0;
          if (r1 < rows) yp[r1] = rp ? rp[r1] + v1 : v1;
        }
      }
    }
  }
}

/* Grouped WMMA matrix-core prefill GEMM for MAT_RAWQ experts (rows % 32 == 0, nb >= 8 && nb % 8 == 0).
 * Uses 128 rows x 32 pairs per WG tile, exact per-32 Q8_0 activation scales. */
typedef struct {
  u8 x[2][32u * RQ_LS] __attribute__((aligned(16)));
  float sx[2][RQ_KB * 32u] __attribute__((aligned(32)));
  u32 lut[2048] __attribute__((aligned(16)));
  float lsa[2][2u * RQ_KB * 32u] __attribute__((aligned(16)));
  u32 pid[32];
} rq_exp_lds_t;

KERNEL ie_gemm_rq_exp_iq3xxs(const G u8 *qw, const G u8 *X, G float *Y, u32 rows, u32 nb, const G u32 *list,
                              const G u32 *offs, u32 np, u32 q_bytes, u32 xs, u32 ys, u32 xdiv,
                              const G float *res, u32 rs) {
  static LDS rq_exp_lds_t lds;
  const u32 type = IE_RQ_IQ3_XXS;
  if (rows == 0u || (rows & 31u) != 0u || nb < 8u || (nb & 7u) != 0u || !qw || !X) return;
  const u32 bs = ie_rq_bs(type);
  const u32 e = tokid(), p0 = list ? offs[e] : 0u, p1 = list ? offs[e + 1u] : np;
  const u32 rbase = wgid() * 128u;
  if (p0 >= p1 || rbase >= rows) return;

  const u32 t0 = tid(), w = t0 >> 5u, l = lane(), h = l >> 4u;
  const u32 row = rbase + 16u * w + (l & 15u), rr = row < rows ? row : rows - 1u;
  const u32 mb = nb >> 3u, nc = 2u * mb;
  const G u8 *rowp = qw + (unsigned long)e * q_bytes + (unsigned long)rr * mb * bs;
  rq_stage_lut_raw(type, (LDSP ie_rq_u64 *)0, lds.lut, 0);

  u32 xso_ = xs, yso_ = ys, rso_ = rs;
  __asm__ volatile("" : "+v"(xso_), "+v"(yso_), "+v"(rso_));
  const u32 tk_ld = t0 >> 3u, pc_ld = t0 & 7u, ts_ld = t0 & 31u;

  for (u32 pb = p0; pb < p1; pb += 32u) {
    if (pb > p0) barrier();
    const u32 nj = p1 - pb < 32u ? p1 - pb : 32u;
    if (t0 < 32u) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      lds.pid[t0] = list ? list[pj] : pj;
    }
    barrier();
    const G u8 *xp = X + (unsigned long)(lds.pid[tk_ld] / xdiv) * xso_ + pc_ld * 16u;
    const G float *sxp = (const G float *)(X + (unsigned long)(lds.pid[ts_ld] / xdiv) * xso_ + 32ul * nb);

    u32x4 rx = *(const G u32x4 *)xp;
    f32x4 rsx = t0 < 32u ? rq_load_f32x4(sxp) : (f32x4)(0.0f);
    *(LDSP u32x4 *)(lds.x[0] + tk_ld * RQ_LS + pc_ld * 16u) = rx;
    if (t0 < 32u) {
      lds.sx[0][0u * 32u + t0] = rsx.x;
      lds.sx[0][1u * 32u + t0] = rsx.y;
      lds.sx[0][2u * 32u + t0] = rsx.z;
      lds.sx[0][3u * 32u + t0] = rsx.w;
    }

    rq_hblk_t cur = rq_hload(type, rowp, 0u, h);
    barrier();

    f8 acc[2];
    acc[0] = (f8)(0.0f), acc[1] = (f8)(0.0f);
#pragma unroll 1
    for (u32 c = 0; c < nc; c++) {
      const u32 bf = c & 1u, hb = c & 1u;
      const int more = c + 1u < nc;
      const u32 cn = more ? c + 1u : c;
      const rq_hblk_t nxt = rq_hload(type, rowp + (unsigned long)(cn >> 1u) * bs, cn & 1u, h);
      if (more) {
        const u32 kb = 4u * (c + 1u);
        rx = *(const G u32x4 *)(xp + kb * 32u);
        if (t0 < 32u) rsx = rq_load_f32x4(sxp + kb);
      }
      __builtin_amdgcn_sched_barrier(0);

      const LDSP u8 *lx = lds.x[bf];
      const LDSP float *lsx = lds.sx[bf];
      const LDSP float *lsa = lds.lsa[bf];
#pragma unroll
      for (u32 k = 0; k < RQ_KB; k++) {
        v2i a0[2], a1[2];
        f8 dx[2];
#pragma unroll
        for (u32 j = 0; j < 2u; j++) {
          const LDSP u8 *pa = lx + (16u * j + (l & 15u)) * RQ_LS + k * 32u + 8u * h;
          a0[j] = *(const LDSP v2i *)pa;
          a1[j] = *(const LDSP v2i *)(pa + 16u);
          dx[j] = *(const LDSP f8 *)(lsx + k * 32u + 16u * j + 8u * h);
        }
        const rq_sub_t dc = rq_hdec(type, &cur, (const LDSP ie_rq_u64 *)0, lds.lut, hb, k, h);
        rq_swg_step(type, &dc, a0, a1, dx, lsa + (2u * k) * 32u, h, acc);
      }

      if (more) {
        const u32 nbf = bf ^ 1u;
        *(LDSP u32x4 *)(lds.x[nbf] + tk_ld * RQ_LS + pc_ld * 16u) = rx;
        if (t0 < 32u) {
          lds.sx[nbf][0u * 32u + t0] = rsx.x;
          lds.sx[nbf][1u * 32u + t0] = rsx.y;
          lds.sx[nbf][2u * 32u + t0] = rsx.z;
          lds.sx[nbf][3u * 32u + t0] = rsx.w;
        }
        lbarrier();
      }
      cur = nxt;
    }

    if (row < rows) {
#pragma unroll
      for (u32 j = 0; j < 2u; j++) {
#pragma unroll
        for (u32 v = 0; v < 8u; v++) {
          const u32 slot = 16u * j + 8u * h + v;
          if (slot < nj) {
            const u32 pid = lds.pid[slot];
            G float *yp = (G float *)((G u8 *)Y + (unsigned long)pid * yso_);
            const G float *rp = res ? (const G float *)((const G u8 *)res + (unsigned long)pid * rso_) : 0;
            const float val = acc[j][v];
            yp[row] = rp ? rp[row] + val : val;
          }
        }
      }
    }
  }
}

/* Inkling2 per-head L2 clip (+ optional scale v + optional tau log-scaling) when clamp > 0,
 * or unweighted per-head RMSNorm (+ optional scale v) when clamp == 0. One wave per head, 8 heads/WG, grid y = T. */
KERNEL ie_vclip(const G float *a, G float *b, const G float *v, u32 nh, u32 hd, float clamp, u32 norope, float escale,
                u32 win, u32 pos0, u32 as, u32 bs) {
  const u32 t = tokid(), h = wgid() * 8u + (tid() >> 5), l = lane();
  if (h >= nh) return;
  a = TOKC(a, as) + (unsigned long)h * hd;
  b = TOK(b, bs) + (unsigned long)h * hd;
  const G float *vp = v ? v + (unsigned long)h * hd : 0;

  if (hd <= 256u) {
    float xs[8];
    float ss = 0.0f;
    for (u32 j = 0; j < 8u; j++) {
      const u32 i = l + 32u * j;
      float val = i < hd ? a[i] : 0.0f;
      if (clamp > 0.0f && vp && i < hd) val *= vp[i];
      xs[j] = val;
      ss += val * val;
    }
    ss = wave_sum_all(ss);
    float f;
    if (clamp > 0.0f) {
      f = clamp / (__builtin_sqrtf(ss) + 1e-6f);
      if (f > 1.0f) f = 1.0f;
      if (norope) {
        const float ratio = (float)(pos0 + t + 1u) / (float)win;
        if (ratio > 1.0f) f *= 1.0f + escale * __builtin_logf(ratio);
      }
    } else {
      f = 1.0f / __builtin_sqrtf(ss / (float)hd + escale);
    }
    for (u32 j = 0; j < 8u; j++) {
      const u32 i = l + 32u * j;
      if (i < hd) {
        float outv = xs[j] * f;
        if (clamp == 0.0f && vp) outv *= vp[i];
        b[i] = outv;
      }
    }
  } else {
    float ss = 0.0f;
    for (u32 i = l; i < hd; i += 32u) {
      float val = a[i];
      if (clamp > 0.0f && vp) val *= vp[i];
      ss += val * val;
    }
    ss = wave_sum_all(ss);
    float f;
    if (clamp > 0.0f) {
      f = clamp / (__builtin_sqrtf(ss) + 1e-6f);
      if (f > 1.0f) f = 1.0f;
      if (norope) {
        const float ratio = (float)(pos0 + t + 1u) / (float)win;
        if (ratio > 1.0f) f *= 1.0f + escale * __builtin_logf(ratio);
      }
    } else {
      f = 1.0f / __builtin_sqrtf(ss / (float)hd + escale);
    }
    for (u32 i = l; i < hd; i += 32u) {
      float outv = a[i] * f;
      if (vp) outv *= vp[i];
      b[i] = outv;
    }
  }
}

/* Inkling2 depth-attention mixing over M slots of size n floats (0..M-2 or 0..M-1 in a, optional last in c),
 * with K queries in q[0..K-1]. One workgroup per token (grid y = T). */
#define AR_MAXM 64u
#define AR_MAXK 8u
KERNEL ie_attn_res(const G float *a, const G float *c, const G float *q, G float *b, u32 M, u32 K, u32 n, float eps,
                   u32 as, u32 cs, u32 bs, int has_c) {
  static LDS float ar_red[8][AR_MAXK + 1u];
  static LDS float ar_inv[AR_MAXM];
  static LDS float ar_prob[AR_MAXK][AR_MAXM];
  if (M == 0u || M > AR_MAXM || K == 0u || K > AR_MAXK) return;
  const u32 w = tid() >> 5, l = lane();
  a = TOKC(a, as), b = TOK(b, bs);
  if (has_c) c = TOKC(c, cs);
  const u32 n_a = has_c ? (M - 1u) : M;
  const float inv_sqn = 1.0f / __builtin_sqrtf((float)n);

  for (u32 s = 0; s < M; s++) {
    const G float *sp = s < n_a ? (a + (unsigned long)s * n) : c;
    float sq = 0.0f;
    float dt[AR_MAXK];
    for (u32 k = 0; k < K; k++) dt[k] = 0.0f;
    for (u32 i = tid(); i < n; i += NT) {
      const float x = sp[i];
      sq += x * x;
      for (u32 k = 0; k < K; k++) dt[k] += x * q[(unsigned long)k * n + i];
    }
    sq = wave_sum_all(sq);
    if (l == 0u) ar_red[w][0] = sq;
    for (u32 k = 0; k < K; k++) {
      const float dk = wave_sum_all(dt[k]);
      if (l == 0u) ar_red[w][1u + k] = dk;
    }
    barrier();
    if (tid() == 0u) {
      float sum_sq = 0.0f;
      for (u32 x = 0; x < 8u; x++) sum_sq += ar_red[x][0];
      const float inv_s = 1.0f / __builtin_sqrtf(sum_sq / (float)n + eps);
      ar_inv[s] = inv_s;
      for (u32 k = 0; k < K; k++) {
        float sum_d = 0.0f;
        for (u32 x = 0; x < 8u; x++) sum_d += ar_red[x][1u + k];
        ar_prob[k][s] = sum_d * inv_s * inv_sqn;
      }
    }
    barrier();
  }
  if (tid() == 0u) {
    for (u32 k = 0; k < K; k++) {
      float mx = ar_prob[k][0];
      for (u32 s = 1u; s < M; s++) mx = __builtin_fmaxf(mx, ar_prob[k][s]);
      float ls = 0.0f;
      for (u32 s = 0; s < M; s++) {
        const float e = __builtin_expf(ar_prob[k][s] - mx);
        ar_prob[k][s] = e;
        ls += e;
      }
      const float inv_l = 1.0f / ls;
      for (u32 s = 0; s < M; s++) ar_prob[k][s] *= inv_l;
    }
  }
  barrier();
  for (u32 i = tid(); i < n; i += NT) {
    for (u32 k = 0; k < K; k++) {
      float acc = 0.0f;
      for (u32 s = 0; s < M; s++) {
        const G float *sp = s < n_a ? (a + (unsigned long)s * n) : c;
        acc += ar_prob[k][s] * (sp[i] * ar_inv[s]);
      }
      b[(unsigned long)k * n + i] = acc;
    }
  }
}

/* Inkling2 loop controller gate: b = (1 - tanh(f)) * a + tanh(g) * c, where gt holds [f(n) | g(n)]. */
KERNEL ie_loop_gate(const G float *a, const G float *c, const G float *gt, G float *b, u32 n, u32 as, u32 cs, u32 gs,
                    u32 bs) {
  const u32 i = wgid() * NT + tid();
  if (i >= n) return;
  a = TOKC(a, as), c = TOKC(c, cs), gt = TOKC(gt, gs), b = TOK(b, bs);
  const float f = ie_tanhf_s(gt[i]), g = ie_tanhf_s(gt[n + i]);
  b[i] = (1.0f - f) * a[i] + g * c[i];
}

/* Inkling2 4-tap causal short convolution: static v[n][4] ([t-3, t-2, t-1, t]) + optional dynamic c[nh][4]
 * ([lag 0..3]), ring[n][4]. When T == 1, writes ring in-place; when T > 1, ie_sconv_ring_store updates ring after. */
KERNEL ie_sconv(const G float *a, const G float *v, const G float *c, G float *ring, G float *b, u32 n, u32 nh,
                float escale, u32 pos0, u32 T, u32 as, u32 cs, u32 bs) {
  const u32 i = wgid() * NT + tid(), t = tokid();
  if (i >= n) return;
  const u32 pos = pos0 + t;
  const G float *at = (const G float *)((const G u8 *)a + (unsigned long)t * as);
  G float *bt = (G float *)((G u8 *)b + (unsigned long)t * bs);
  const float x0 = at[i];
  const float x1 = pos >= 1u ? (t >= 1u ? ((const G float *)((const G u8 *)a + (unsigned long)(t - 1u) * as))[i]
                                        : ring[i * 4u + ((pos - 1u) & 3u)])
                             : 0.0f;
  const float x2 = pos >= 2u ? (t >= 2u ? ((const G float *)((const G u8 *)a + (unsigned long)(t - 2u) * as))[i]
                                        : ring[i * 4u + ((pos - 2u) & 3u)])
                             : 0.0f;
  const float x3 = pos >= 3u ? (t >= 3u ? ((const G float *)((const G u8 *)a + (unsigned long)(t - 3u) * as))[i]
                                        : ring[i * 4u + ((pos - 3u) & 3u)])
                             : 0.0f;
  const G float *vw = v + (unsigned long)i * 4u;
  float w0 = vw[3], w1 = vw[2], w2 = vw[1], w3 = vw[0];
  if (c) {
    const u32 hd = n / nh, h = i / hd;
    const G float *ct = (const G float *)((const G u8 *)c + (unsigned long)t * cs) + (unsigned long)h * 4u;
    w0 += ct[0], w1 += ct[1], w2 += ct[2], w3 += ct[3];
  }
  const float y = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
  bt[i] = (x0 + y) * escale;
  if (T == 1u) ring[i * 4u + (pos & 3u)] = x0;
}

/* Inkling2 decode relative-position attention with per-head dynamic sink.
 * Handles both nsplit == 1 (zero part/count overhead) and nsplit > 1 flash-decoding over KV ring kv_len. */
KERNEL ie_rel_attn(const G float *q, const G kvt *kc, const G kvt *vc, const G float *R, const G float *rel_bias,
                   const G float *sink, G float *part, G float *out, u32 pos, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
                   u32 d_rel, u32 rel_extent, u32 win, u32 nsplit, u32 ch, G u32 *count) {
  static LDS float rel_m[8], rel_l[8], rel_o[8][256];
  static LDS u32 rel_last;
  const u32 h = wgid() / nsplit, s = wgid() % nsplit;
  const u32 kh = h / (n_head / n_kv), kvd = n_kv * hd;
  const u32 w = tid() >> 5, l = lane();
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  const float sink_val = sink[h];

  float qv[8], acc[8];
  for (u32 j = 0; j < 8u; j++) {
    const u32 i = l + 32u * j;
    qv[j] = i < hd ? q[(unsigned long)h * hd + i] : 0.0f;
    acc[j] = 0.0f;
  }
  const float rv = (d_rel <= 32u && l < d_rel) ? R[(unsigned long)h * d_rel + l] : 0.0f;

  float m = -__builtin_inff(), sum = 0.0f;
  const u32 w0 = (win > 0u && pos + 1u > win) ? (pos + 1u - win) : 0u;
  const u32 p0 = w0 + s * ch;
  const u32 pend = pos + 1u;
  const u32 p1 = w0 + (s + 1u) * ch < pend ? w0 + (s + 1u) * ch : pend;
  for (u32 kp = p0 + w; kp < p1; kp += 8u) {
    const u32 kvp = kp % kv_len;
    kvrow_t k = KVROWP(kc, kvp, kh), v = KVROWP(vc, kvp, kh);
    float d = 0.0f;
    for (u32 j = 0; j < 8u; j++)
      if (l + 32u * j < hd) d += qv[j] * KVAT(k, l + 32u * j);
    d = wave_sum_all(d) * scale;
    const u32 dist = pos - kp;
    if (dist < extent_lim) {
      float rd = 0.0f;
      if (d_rel <= 32u) {
        const float rb = l < d_rel ? rel_bias[(unsigned long)dist * d_rel + l] : 0.0f;
        rd = wave_sum_all(rv * rb);
      } else {
        for (u32 j = l; j < d_rel; j += 32u) rd += R[(unsigned long)h * d_rel + j] * rel_bias[(unsigned long)dist * d_rel + j];
        rd = wave_sum_all(rd);
      }
      d += rd;
    }
    const float mn = __builtin_fmaxf(m, d), c = sum > 0.0f ? __builtin_expf(m - mn) : 0.0f, e = __builtin_expf(d - mn);
    sum = sum * c + e;
    for (u32 j = 0; j < 8u; j++)
      if (l + 32u * j < hd) acc[j] = acc[j] * c + e * KVAT(v, l + 32u * j);
    m = mn;
  }

  if (l == 0u) rel_m[w] = m, rel_l[w] = sum;
  for (u32 j = 0; j < 8u; j++)
    if (l + 32u * j < hd) rel_o[w][l + 32u * j] = acc[j];
  barrier();

  if (nsplit == 1u) {
    float M = sink_val;
    for (u32 x = 0; x < 8u; x++)
      if (rel_l[x] > 0.0f) M = __builtin_fmaxf(M, rel_m[x]);
    float L = __builtin_expf(sink_val - M);
    for (u32 x = 0; x < 8u; x++)
      if (rel_l[x] > 0.0f) L += rel_l[x] * __builtin_expf(rel_m[x] - M);
    if (tid() < hd) {
      float o = 0.0f;
      for (u32 x = 0; x < 8u; x++)
        if (rel_l[x] > 0.0f) o += rel_o[x][tid()] * __builtin_expf(rel_m[x] - M);
      out[(unsigned long)h * hd + tid()] = o / L;
    }
    return;
  }

  float M = -__builtin_inff();
  for (u32 x = 0; x < 8u; x++)
    if (rel_l[x] > 0.0f) M = __builtin_fmaxf(M, rel_m[x]);
  G float *pp = part + (unsigned long)(h * nsplit + s) * (hd + 2u);
  if (tid() < hd) {
    float o = 0.0f;
    for (u32 x = 0; x < 8u; x++)
      if (rel_l[x] > 0.0f) o += rel_o[x][tid()] * __builtin_expf(rel_m[x] - M);
    pp[2u + tid()] = o;
  }
  if (tid() == 0u) {
    float L = 0.0f;
    for (u32 x = 0; x < 8u; x++)
      if (rel_l[x] > 0.0f) L += rel_l[x] * __builtin_expf(rel_m[x] - M);
    pp[0] = M;
    pp[1] = L;
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) rel_last = __scoped_atomic_fetch_add(&count[h], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u;
  barrier();
  if (!rel_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  G float *ph = part + (unsigned long)h * nsplit * (hd + 2u);
  float mv = tid() == 0u ? sink_val : -__builtin_inff();
  for (u32 t = tid(); t < nsplit; t += NT)
    if (ph[t * (hd + 2u) + 1u] > 0.0f) mv = __builtin_fmaxf(mv, ph[t * (hd + 2u)]);
  const float M2 = wg_max(mv);
  float lv = 0.0f;
  for (u32 t = tid(); t < nsplit; t += NT) {
    const float lp = ph[t * (hd + 2u) + 1u];
    const float wt = lp > 0.0f ? __builtin_expf(ph[t * (hd + 2u)] - M2) : 0.0f;
    ph[t * (hd + 2u)] = wt;
    lv += lp * wt;
  }
  const float L2 = wg_sum(lv) + __builtin_expf(sink_val - M2);
  if (tid() < hd) {
    float o = 0.0f;
    for (u32 t = 0; t < nsplit; t++) o += ph[t * (hd + 2u) + 2u + tid()] * ph[t * (hd + 2u)];
    out[(unsigned long)h * hd + tid()] = o / L2;
  }
  if (tid() == 0u) count[h] = 0u;
}

/* Inkling2 tiled prefill relative-position attention (T > 1): grid ((T+7)/8, n_head, 1).
 * Cooperatively stages 12 K/V keys + up to 20 rel_bias rows into < 28 KiB LDS; zero extra global scratch. */
KERNEL ie_rel_attn_pf(const G float *q, const G kvt *kc, const G kvt *vc, const G float *R, const G float *rel_bias,
                      const G float *sink, G float *out, u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
                      u32 d_rel, u32 rel_extent, u32 win, u32 qs, u32 rs, u32 ss, u32 os) {
  static LDS float pf_k[12][256];
  static LDS float pf_v[12][256];
  static LDS float pf_rb[20][32];
  if (hd > 256u) return;
  const u32 tb = wgid() * 8u, h = tokid(), kh = h / (n_head / n_kv), kvd = n_kv * hd;
  const u32 w = tid() >> 5, l = lane(), qt = tb + w;
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const float scale = 1.0f / __builtin_sqrtf((float)hd);

  float qv[8], acc[8];
  const G float *rp = 0;
  float rv = 0.0f, m = -__builtin_inff(), sum = 1.0f;
  u32 qpos = 0u;
  if (qt < T) {
    const G float *qp = (const G float *)((const G u8 *)q + (unsigned long)qt * qs) + (unsigned long)h * hd;
    rp = (const G float *)((const G u8 *)R + (unsigned long)qt * rs) + (unsigned long)h * d_rel;
    const G float *sp = (const G float *)((const G u8 *)sink + (unsigned long)qt * ss);
    m = sp[h];
    qpos = pos0 + qt;
    if (d_rel <= 32u && l < d_rel) rv = rp[l];
    for (u32 j = 0; j < 8u; j++) {
      const u32 i = l + 32u * j;
      qv[j] = i < hd ? qp[i] : 0.0f;
      acc[j] = 0.0f;
    }
  } else {
    for (u32 j = 0; j < 8u; j++) qv[j] = 0.0f, acc[j] = 0.0f;
  }

  const u32 qpos_min = pos0 + tb;
  const u32 qpos_max = pos0 + (tb + 7u < T ? tb + 7u : T - 1u);
  const u32 k_start = (win > 0u && qpos_min + 1u > win) ? (qpos_min + 1u - win) : 0u;
  const u32 k_end = qpos_max;

  for (u32 kb = k_start; kb <= k_end; kb += 12u) {
    u32 len_t = k_end - kb + 1u;
    if (len_t > 12u) len_t = 12u;
    for (u32 kk = w; kk < len_t; kk += 8u) {
      const u32 kvp = (kb + kk) % kv_len;
      kvrow_t kr = KVROWP(kc, kvp, kh), vr = KVROWP(vc, kvp, kh);
      for (u32 j = 0; j < 8u; j++) {
        const u32 i = l + 32u * j;
        if (i < hd) pf_k[kk][i] = KVAT(kr, i), pf_v[kk][i] = KVAT(vr, i);
      }
    }
    const int d_min_s = (int)qpos_min - (int)(kb + len_t - 1u);
    const u32 d_base = d_min_s > 0 ? (u32)d_min_s : 0u;
    const u32 d_max = qpos_max - kb;
    if (d_rel <= 32u && d_base < extent_lim) {
      const u32 d_hi = d_max < extent_lim ? d_max : (extent_lim - 1u);
      const u32 nd = d_hi >= d_base ? (d_hi - d_base + 1u) : 0u;
      for (u32 dd = w; dd < nd; dd += 8u)
        if (l < d_rel) pf_rb[dd][l] = rel_bias[(unsigned long)(d_base + dd) * d_rel + l];
    }
    barrier();

    if (qt < T) {
      for (u32 kk = 0; kk < len_t; kk++) {
        const u32 kp = kb + kk;
        if (kp <= qpos && (win == 0u || qpos - kp < win)) {
          float dk = 0.0f;
          for (u32 j = 0; j < 8u; j++)
            if (l + 32u * j < hd) dk += qv[j] * pf_k[kk][l + 32u * j];
          dk = wave_sum_all(dk) * scale;
          const u32 dist = qpos - kp;
          if (dist < extent_lim) {
            float rd = 0.0f;
            if (d_rel <= 32u) rd = wave_sum_all(rv * (l < d_rel ? pf_rb[dist - d_base][l] : 0.0f));
            else {
              for (u32 j = l; j < d_rel; j += 32u) rd += rp[j] * rel_bias[(unsigned long)dist * d_rel + j];
              rd = wave_sum_all(rd);
            }
            dk += rd;
          }
          const float mn = __builtin_fmaxf(m, dk), c = __builtin_expf(m - mn), e = __builtin_expf(dk - mn);
          sum = sum * c + e;
          for (u32 j = 0; j < 8u; j++)
            if (l + 32u * j < hd) acc[j] = acc[j] * c + e * pf_v[kk][l + 32u * j];
          m = mn;
        }
      }
    }
    barrier();
  }

  if (qt < T) {
    G float *op = (G float *)((G u8 *)out + (unsigned long)qt * os) + (unsigned long)h * hd;
    const float inv = 1.0f / sum;
    for (u32 j = 0; j < 8u; j++)
      if (l + 32u * j < hd) op[l + 32u * j] = acc[j] * inv;
  }
}

/* Inkling2 SITU MoE activation: per slot (grid y = nh * T), input [g(n) | u(n)] at astr ->
 * out = (beta * tanh(g / beta)) * sigmoid(g) * (lbeta * tanh(u / lbeta)), quantized to q (qstr) and optional b (bstr). */
KERNEL ie_situ(const G float *a, G float *b, G u8 *q, u32 n, u32 astr, u32 bstr, u32 qstr, float beta, float lbeta) {
  const u32 i = wgid() * NT + tid();
  if (i >= n) return;
  a = TOKC(a, astr);
  const float g = a[i], u = a[n + i];
  const float gs = beta > 0.0f ? beta * ie_tanhf_s(g / beta) : g;
  const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(u / lbeta) : u;
  const float v = gs * sigmoidf(g) * us;
  if (b) TOK(b, bstr)[i] = v;
  if (q) quant_wave(v, i >> 5, TOK(q, qstr), n >> 5);
}
#endif /* !IE_MEGA */
