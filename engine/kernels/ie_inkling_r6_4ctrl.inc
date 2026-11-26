/* ============================================================================
 * Inkling-2-Mini Round 6 (p7-4ctrl-fuse): Cross-Controller Loop Fusion Kernels
 * Target: RX 9070 XT (gfx1201, wave32, NT=256), Clang 21 bit_cast-safe, 0 spills.
 *
 * Audit Exp #2 (24 repeated core visits: ip = loop_start..loop_end-1, pass = 1..2):
 *   Replaces per-controller loop bank reads, prenorms, rank-64 down projections,
 *   wfg+loop_gate, and downstream RMSNorm+Q8 across controllers (q, k, v, mlp).
 *
 * Graph dataflow note (src/model.c:3052-3190):
 *   - Loop bank reads (msg_qm=[msg_q,msg_mlp], msg_kv=[msg_k,msg_v]) are adjacent
 *     at the top of every repeated visit and share identical bank_buf -> K=4 single read.
 *   - Attention controllers loop[0..2] (q, k, v) execute back-to-back before attention:
 *       * 20/24 SWA visits (!L->is_full): inp0==inp1==inp2==ar (shared!), downstream
 *         norm weight nw0==nw1==nw2==&L->attn_norm, ny0==ny1==ny2==NULL (dead_b=1).
 *       * 4/24 global visits (L->is_full, ip=11,17): distinct inp0=ar, inp1=ar+dim,
 *         inp2=ar+2*dim, distinct nw=(q_inp_norm, k_inp_norm, v_inp_norm), ny==NULL (dead_b=1).
 *   - MLP controller loop[3] (mlp) depends on post-attention fres (model.c:3181-3190)
 *     and runs after attention with n_ctrl=1, nw=&L->ffn_norm, ny=ffn_in (NON-NULL!
 *     dead_b=0 because F32 router L->wr reads ffn_in), nq=ffn_in_q8.
 *   - Kernels 2, 3, 4 support n_ctrl = 1..4 uniformly (n_ctrl=3 for q/k/v, n_ctrl=1
 *     for mlp, and n_ctrl=4 for full 4-way stacked tests/variants).
 *   - Launch count on Inkling-2-Mini (24 repeated visits):
 *       * T=1 default (IE_INKLING_T1_FUSE=0): 816 -> 168 launches/step (-648, -29.2% total)
 *       * T>1 (vs R2 fused):                  432 -> 168 launches/step (-264, -61.1% LC)
 *
 * All stride parameters across all 4 kernels are per-token BYTE strides.
 *
 * Exported Kernels, Grids, Guards, and Parity:
 *
 * 1. KERNEL ie_inkling_attn_res_4out_r6(
 *        const G float *a, const G float *c,
 *        const G float *q0, const G float *q1,
 *        G float *b0, G float *b1,
 *        u32 M, u32 n, float eps,
 *        u32 as, u32 cs, u32 bs0, u32 bs1)
 *    - Fuses:  Two K=2 OP_ATTN_RES bank reads (msg_qm=[q,mlp], msg_kv=[k,v]) into
 *              ONE K=4 bank read (reads all M slots of bank_buf ONCE; caches slots
 *              0..1 in VGPRs -> 0 second global reads for Inkling-2's n_bank=2).
 *              If q1==NULL, reads queries 2..3 from q0 + 2*n.
 *              If b1==NULL, writes outputs 2..3 to b0 + 2*n (stride bs0).
 *    - Grid:   (1u, T, 1u) via launch_t(b, K_INK_ATTN_RES_4OUT_R6, 1u, T, args)
 *    - Guard:  M >= 1u && M <= 64u && n >= 1u && q0 && b0 && (c || a) &&
 *              ((n & 3u) != 0u || (
 *                ((!c || M > 1u) ? (((uintptr_t)a & 15u) == 0u && (T == 1u || (as & 15u) == 0u)) : 1) &&
 *                (!c || (((uintptr_t)c & 15u) == 0u && (T == 1u || (cs & 15u) == 0u))) &&
 *                ((uintptr_t)q0 & 15u) == 0u && (!q1 || ((uintptr_t)q1 & 15u) == 0u) &&
 *                ((uintptr_t)b0 & 15u) == 0u && (T == 1u || (bs0 & 15u) == 0u) &&
 *                (!b1 || (((uintptr_t)b1 & 15u) == 0u && (T == 1u || (bs1 & 15u) == 0u)))))
 *    - Res:    11,520 B LDS (<= 12 KiB), <= 84 VGPR, 0 spills, 4 WGs/CU.
 *    - Parity: bitwise-exact (0-bit-diff) vs two calls to ie_inkling_attn_res_r1 (K=2).
 *
 * 2. KERNEL ie_inkling_lc_prenorm_q8_4x_r6(
 *        const G float *inp0, const G float *inp1, const G float *inp2, const G float *inp3,
 *        const G float *msg0, const G float *msg1, const G float *msg2, const G float *msg3,
 *        G u8 *q0, G u8 *q1, G u8 *q2, G u8 *q3,
 *        u32 dim, float eps, u32 n_ctrl, u32 flags,
 *        u32 is0, u32 is1, u32 is2, u32 is3,
 *        u32 ms0, u32 ms1, u32 ms2, u32 ms3, u32 qs)
 *    - Flags:  bit 0 (1u) = SHARED_INP: all n_ctrl controllers share inp0 (read &
 *                           RMSNorm+Q8-quantized ONCE in bx==0!).
 *              bit 1 (2u) = NO_REP_INP (with bit 0): write inp_q ONLY to q0[0..nb-1]
 *                           (pairs with ie_inkling_lc_down_4x_r6 shared_inp=1; saves
 *                           2x/3x global inp_q stores). Clear bit 1 to replicate
 *                           inp_q to q0..q_{n_ctrl-1} (standalone 2*dim Q8_0 for prefill).
 *    - Grid:   (flags & 1u) ? (1u + n_ctrl, T, 1u) : (2u * n_ctrl, T, 1u)
 *              (SWA n_ctrl=3: (4, T, 1); 4-ctrl shared: (5, T, 1); global n_ctrl=3: (6, T, 1))
 *    - Guard:  T >= 1u && n_ctrl >= 1u && n_ctrl <= 4u && dim >= 32u && (dim & 31u) == 0u &&
 *              all active inp_c, msg_c 16-byte aligned (and is_c, ms_c 16-byte aligned for T>1),
 *              all active q_c non-NULL (4-byte aligned, qs 4-byte aligned for T>1).
 *    - Res:    4,128 B LDS, <= 36 VGPR, 0 spills, 4 WGs/CU.
 *    - Parity: bitwise-exact (0-bit-diff) vs n_ctrl calls to ie_inkling_lc_prenorm_q8_r2
 *              (and vs ie_inkling_rmsnorm_mt_r1 + ie_quant_q8).
 *
 * 3. KERNEL ie_inkling_lc_down_4x_r6(
 *        const G u32 *qw0, const G f16 *qs0, const G u8 *xq0, G float *y0,
 *        const G u32 *qw1, const G f16 *qs1, const G u8 *xq1, G float *y1,
 *        const G u32 *qw2, const G f16 *qs2, const G u8 *xq2, G float *y2,
 *        const G u32 *qw3, const G f16 *qs3, const G u8 *xq3, G float *y3,
 *        u32 rows, u32 nb, u32 T, u32 n_ctrl, u32 shared_inp,
 *        u32 xs, u32 ys)
 *    - Fuses:  Up to 4 rank-64 Q8_0 down projections [rows=64, 2*dim=4608] (nb=144)
 *              into ONE launch for T=1 (decode) and T=2..16 (verify; safe for any T>=1).
 *              When shared_inp != 0u && (nb & 1u) == 0u, blocks [0..nb/2-1] of ALL
 *              controllers read xq0 (zero duplicate inp_q cache lines; works with
 *              prenorm NO_REP_INP=2u!).
 *    - Grid:   (n_ctrl * rows, 1u, 1u)  [n_ctrl=3: 192 WGs; n_ctrl=4: 256 -> pad 257 ok]
 *              Note: on gfx1201, if gx in [253..256], host may pad gx=257 (safe early-exit).
 *    - Guard:  T >= 1u && T <= 16u && n_ctrl >= 1u && n_ctrl <= 4u && rows > 0u && nb > 0u &&
 *              all active qw_c, qs_c, xq_c, y_c non-NULL and 4-byte aligned
 *              (and xs, ys 4-byte aligned for T>1). (T >= 17 prefill: use WMMA small_r1.)
 *    - Res:    8,192 B LDS, <= 48 VGPR, 0 spills, 4 WGs/CU.
 *    - Parity: bitwise-exact (0-bit-diff) vs ie_gemv_q8q8 / ie_inkling_gemv_q8_small_r1 (T=1)
 *              and ie_gemv_q8q8_t / ie_inkling_gemv_q8_small_ver_r5 (T=2..16).
 *
 * 4. KERNEL ie_inkling_lc_wfg_gate_norm_r6(
 *        const G u32 *qw0, const G f16 *qs0, const G float *h0, const G float *inp0, const G float *msg0,
 *        G float *tmp0, const G float *nw0, G float *ny0, G u8 *nq0,
 *        const G u32 *qw1, const G f16 *qs1, const G float *h1, const G float *inp1, const G float *msg1,
 *        G float *tmp1, const G float *nw1, G float *ny1, G u8 *nq1,
 *        const G u32 *qw2, const G f16 *qs2, const G float *h2, const G float *inp2, const G float *msg2,
 *        G float *tmp2, const G float *nw2, G float *ny2, G u8 *nq2,
 *        const G u32 *qw3, const G f16 *qs3, const G float *h3, const G float *inp3, const G float *msg3,
 *        G float *tmp3, const G float *nw3, G float *ny3, G u8 *nq3,
 *        G u32 *count, u32 dim, u32 nb, u32 T, u32 n_ctrl, float eps,
 *        u32 hs, u32 is0, u32 is1, u32 is2, u32 is3,
 *        u32 ms0, u32 ms1, u32 ms2, u32 ms3,
 *        u32 ts, u32 ys, u32 qs_out)
 *    - Fuses:  [OP_QUANT(h[64]) + OP_GEMV_Q8/GEMM(wfg[2*dim,64]) + OP_LOOP_GATE(inp,msg,fg -> tmp)
 *               + OP_RMSNORM(tmp, nw -> ny [NULL if dead_b], nq [Q8_0])] across n_ctrl = 1..4!
 *              Pass nw_c == NULL (or count == NULL) to skip the downstream RMSNorm tail.
 *              Pass ny_c == NULL when dead_b == 1 (all Q/K/V with Q8 dense weights);
 *              pass ny_c = ffn_in (NON-NULL!) for MLP (F32 router reads ffn_in).
 *    - Grid:   T <= 16u: ((dim + 31u) / 32u, n_ctrl, 1u)
 *                        [dim=2304: (72, 3, 1)=216 WGs for QKV; (72, 1, 1)=72 WGs for MLP]
 *              T >= 17u: ((dim + 127u) / 128u, n_ctrl * ((T + 63u) / 64u), 1u)
 *                        [dim=2304, T=512, n_ctrl=3: (18, 24, 1)=432 WGs]
 *    - Count:  count must have >= n_ctrl u32s for T <= 16u, and >= n_ctrl * ((T + 63u) / 64u)
 *              u32s for T >= 17u, all zero-initialized once at init; each counter resets to 0u.
 *    - Guard:  T >= 1u && n_ctrl >= 1u && n_ctrl <= 4u && dim >= 32u && (dim & 31u) == 0u && nb == 2u &&
 *              all active qw_c, qs_c, h_c, inp_c, msg_c, tmp_c non-NULL;
 *              qw_c, qs_c, h_c, inp_c, msg_c 4-byte aligned;
 *              if nw_c && count: tmp_c, nw_c, (ny_c if non-NULL) 16-byte aligned
 *              (and ts, (ys if ny_c) 16-byte aligned for T>1; hs, is_c, ms_c, qs_out 4-byte aligned).
 *    - Res:    5,636 B LDS (union wfg|rmsnorm + last flag), <= 84 VGPR, 0 spills, 4 WGs/CU.
 *    - Parity: T <= 16u: bitwise-exact (0-bit-diff) vs ie_quant_q8 + ie_gemv_q8q8[_t] +
 *                        ie_loop_gate + ie_inkling_rmsnorm_mt_r1.
 *              T >= 17u: bitwise-exact (0-bit-diff) vs ie_inkling_lc_wfg_gate_r2 +
 *                        ie_inkling_rmsnorm_mt_r1.
 *
 * Integration / Wiring Notes (src/model.c, src/hip.c, kernels/ie_inkling_kernels.c):
 *   1. kernels/ie_inkling_kernels.c: append #include "ie_inkling_r6_4ctrl.inc".
 *   2. src/hip.c:
 *      - Add K_INK_ATTN_RES_4OUT_R6, K_INK_LC_PRENORM_Q8_4X_R6, K_INK_LC_DOWN_4X_R6,
 *        K_INK_LC_WFG_GATE_NORM_R6 to enum, kname[], and ink_bit[] (bits 50..53; bump
 *        ink_nbits from 50u -> 54u in hip.c:1702).
 *      - Allocate device u32 *ink_lc_count = max(16u, 4u * ((g->T + 63u) / 64u) * 4u)
 *        zeroed once at init and freed in gpu_free (separate from 4-byte b->gm_count!).
 *      - Zero-model.c-change peepholes in ink_r2_fuse (work immediately!):
 *        (a) 2 adjacent OP_ATTN_RES (msg_qm, msg_kv, same bank_buf, c<0, qstr==2)
 *            -> K_INK_ATTN_RES_4OUT_R6 (returns 2u).
 *        (b) [OP_QUANT(64), OP_GEMV_Q8(wfg,nb=2), OP_LOOP_GATE, OP_RMSNORM(out->ny,qo=nq)]
 *            -> K_INK_LC_WFG_GATE_NORM_R6 with n_ctrl=1 (returns 4u; hits MLP and any
 *            unreordered controller! Pass ny0 = (o3->dead_b && nq0 && deadb_on()) ? NULL : B3).
 *      - Full 3-way stacked (q, k, v) fusion (15-line phased emit in src/model.c:3081-3085):
 *        In current model.c, sequential ink_loop_ctrl_op() calls let cq/h of q, k, v alias
 *        in the arena (while lc_q/lc_k/lc_v and qq/kq/vq are already disjoint). Emit the
 *        3 attention controllers in 3 phases (all 3 prenorm+quant -> all 3 down -> all 3
 *        wfg+loop_gate, with use_last(B, cq[c]) after down and use_last(B, h[c]) after
 *        loop_gate; identical topological semantics for cpu.c!). Then ink_r2_fuse matches:
 *        (c) 9 ops (3x [RMSNORM(ones), RMSNORM(ones), QUANT(2*dim)])
 *            -> K_INK_LC_PRENORM_Q8_4X_R6 (n_ctrl=3, flags = same_inp ? (T<=16 ? 3u : 1u) : 0u).
 *        (d) 3 ops (3x OP_GEMV_Q8 down) for T<=16 -> K_INK_LC_DOWN_4X_R6 (n_ctrl=3,
 *            shared_inp = same_inp ? 1u : 0u). For T>=17 falls through to K_INK_GEMM_Q8_SMALL.
 *        (e) 12 ops (3x [QUANT(64), GEMV_Q8(wfg), LOOP_GATE] + 3x OP_RMSNORM q/k/v_in)
 *            -> K_INK_LC_WFG_GATE_NORM_R6 (n_ctrl=3, ny0=ny1=ny2=NULL when dead_b==1).
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(f16) == 2u, "ink_r6_4c f16 must be 2 bytes");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r6_4c float/u32 must be 4 bytes");

#define INK_R6_4C_AR_MAXM    64u
#define INK_R6_4C_AR_SLOTS   (INK_R6_4C_AR_MAXM * 5u) /* 320 * 8 floats = 10,240 B */
#define INK_R6_4C_PN_SHARED  1u
#define INK_R6_4C_PN_NOREP   2u
#define INK_R6_4C_DN_CHUNK   8u
#define INK_R6_4C_NB2_R      128u
#define INK_R6_4C_NB2_T      64u
#define INK_R6_4C_NB2_XLS    80u /* 64 B (nb=2) + 16 B pad -> 0 bank conflicts on b128 reads */

typedef _Float16 ink_r6_4c_f16x2 __attribute__((ext_vector_type(2)));

struct __attribute__((packed, aligned(4))) ink_r6_4c_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r6_4c_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r6_4c_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) float ink_r6_4c_wave_max(float v) {
  v = __builtin_fmaxf(v, shfl_xor_f(v, 16u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 8u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 4u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 2u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 1u));
  return v;
}

/* Wave-parallel softmax over lanes 0..M-1 (M <= 32, called uniformly by all 32 lanes).
 * Exact arithmetic match to ink_r1_g06_wave_softmax. */
static inline __attribute__((always_inline)) float ink_r6_4c_wave_softmax(float logit, u32 l, u32 M) {
  const float v = (l < M) ? logit : -__builtin_inff();
  const float mx = ink_r6_4c_wave_max(v);
  const float e = (l < M) ? __builtin_expf(logit - mx) : 0.0f;
  const float ls = wave_sum_all(e);
  return (l < M) ? (e * (1.0f / ls)) : 0.0f;
}

/* Exact Q8_0 x Q8_0 single-block signed integer dot product (matches ie_q8q8_block). */
static inline __attribute__((always_inline)) int ink_r6_4c_dot8_ss(u32x4 w0, u32x4 w1, u32x4 a0, u32x4 a1) {
  u32 d = ie_dot4_ss(0u, w0.x, a0.x);
  d = ie_dot4_ss(d, w0.y, a0.y);
  d = ie_dot4_ss(d, w0.z, a0.z);
  d = ie_dot4_ss(d, w0.w, a0.w);
  d = ie_dot4_ss(d, w1.x, a1.x);
  d = ie_dot4_ss(d, w1.y, a1.y);
  d = ie_dot4_ss(d, w1.z, a1.z);
  d = ie_dot4_ss(d, w1.w, a1.w);
  return (int)d;
}

/* Exact non-FMA loop gate: (1 - tanh(f)) * a + tanh(g) * c (matches ie_loop_gate). */
static inline __attribute__((always_inline)) float ink_r6_4c_loop_gate1(float a, float c, float f_raw, float g_raw) {
#pragma clang fp contract(off)
  const float f = ie_tanhf_s(f_raw);
  const float g = ie_tanhf_s(g_raw);
  return (1.0f - f) * a + g * c;
}

/* Pack 32 per-lane int8 values in a wave into 8 conflict-free u32 LDS stores (32 bytes). */
static inline __attribute__((always_inline)) void ink_r6_4c_quant_pack_lds(float v, u32 l, LDSP u8 *dst_q,
                                                                           LDSP float *dst_d) {
  const float amax = wave_max_pos(__builtin_fabsf(v));
  const float d = amax / 127.0f;
  const int qi = d != 0.0f ? (int)__builtin_roundf(v / d) : 0;
  const u32 b0 = (u32)(u8)(i8)qi;
  const u32 p2 = b0 | ((u32)shl_i((int)b0, 1) << 8u);
  const u32 p4 = p2 | ((u32)shl_i((int)p2, 2) << 16u);
  if ((l & 3u) == 0u) ((LDSP u32 *)dst_q)[l >> 2u] = p4;
  if (l == 0u) *dst_d = d;
}

/* ============================================================================
 * Kernel 1: ie_inkling_attn_res_4out_r6
 * Fuses the two K=2 OP_ATTN_RES loop-bank reads into ONE K=4 pass over bank_buf.
 * Caches slots 0..1 in named VGPRs (100% VGPR-cached for Inkling-2's n_bank=2;
 * zero second global reads, 1 lbarrier(), <= 84 VGPR, 0 spills, 11.25 KiB LDS).
 * Exact FP32 reduction & rounding order matching ie_inkling_attn_res_r1.
 * ========================================================================== */

static LDS float ink_r6_4c_ar_red[INK_R6_4C_AR_SLOTS * 8u];
static LDS float ink_r6_4c_ar_inv[INK_R6_4C_AR_MAXM];
static LDS float ink_r6_4c_ar_prob[4u][INK_R6_4C_AR_MAXM];

KERNEL ie_inkling_attn_res_4out_r6(const G float *a, const G float *c,
                                   const G float *q0, const G float *q1,
                                   G float *b0, G float *b1,
                                   u32 M, u32 n, float eps,
                                   u32 as, u32 cs, u32 bs0, u32 bs1) {
  _Static_assert(sizeof(ink_r6_4c_ar_red) + sizeof(ink_r6_4c_ar_inv) + sizeof(ink_r6_4c_ar_prob) == 11520u,
                 "ink_r6_4c attn_res LDS must be 11520 B");
  if (M == 0u || M > INK_R6_4C_AR_MAXM || n == 0u || !q0 || !b0) return;
  const u32 n_a = c ? (M - 1u) : M;
  if (n_a > 0u && !a) return;

  const u32 th = tid(), wv = th >> 5u, l = lane();
  if (a) a = TOKC(a, as);
  if (c) c = TOKC(c, cs);
  G float *out0 = TOK(b0, bs0);
  G float *out2 = b1 ? TOK(b1, bs1) : (out0 + (unsigned long)n * 2ul);
  const G float *q2_base = q1 ? q1 : (q0 + (unsigned long)n * 2ul);
  const float inv_sqn = 1.0f / __builtin_sqrtf((float)n);

  /* Fast path: Inkling-2 dim == 2304, M <= 32 (100% of repeated loop visits).
   * Loads all 4 queries once (36 VGPRs), caches slots 0..1 (18 VGPRs), 1 lbarrier(). */
  if (n == 2304u && M <= 32u) {
    const f32x4 z4 = {0.0f, 0.0f, 0.0f, 0.0f};
    const G float *qk0 = q0;
    const G float *qk1 = q0 + 2304u;
    const G float *qk2 = q2_base;
    const G float *qk3 = q2_base + 2304u;

    const f32x4 q0_0 = *(const G f32x4 *)(qk0 + th * 4u);
    const f32x4 q1_0 = *(const G f32x4 *)(qk0 + 1024u + th * 4u);
    const float q2_0 = qk0[2048u + th];

    const f32x4 q0_1 = *(const G f32x4 *)(qk1 + th * 4u);
    const f32x4 q1_1 = *(const G f32x4 *)(qk1 + 1024u + th * 4u);
    const float q2_1 = qk1[2048u + th];

    const f32x4 q0_2 = *(const G f32x4 *)(qk2 + th * 4u);
    const f32x4 q1_2 = *(const G f32x4 *)(qk2 + 1024u + th * 4u);
    const float q2_2 = qk2[2048u + th];

    const f32x4 q0_3 = *(const G f32x4 *)(qk3 + th * 4u);
    const f32x4 q1_3 = *(const G f32x4 *)(qk3 + 1024u + th * 4u);
    const float q2_3 = qk3[2048u + th];

    /* Named VGPR cache for slots 0..1 (covers 100% of Inkling-2 n_bank=2, keeps VGPRs <= 84). */
    f32x4 c0_0 = z4, c1_0 = z4, c0_1 = z4, c1_1 = z4;
    float c2_0 = 0.0f, c2_1 = 0.0f;

    /* Pass 1: compute norm^2 and all 4 dot products per slot with zero in-loop barriers. */
    for (u32 s = 0u; s < M; s++) {
      const G float *sp = s < n_a ? (a + (unsigned long)s * 2304ul) : c;
      const f32x4 x0 = *(const G f32x4 *)(sp + th * 4u);
      const f32x4 x1 = *(const G f32x4 *)(sp + 1024u + th * 4u);
      const float x2 = sp[2048u + th];
      if (s == 0u) {
        c0_0 = x0; c1_0 = x1; c2_0 = x2;
      } else if (s == 1u) {
        c0_1 = x0; c1_1 = x1; c2_1 = x2;
      }

      const u32 rbase = s * 40u; /* 5 * 8 floats per slot */
      float sq = ((x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w)) +
                 ((x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w)) +
                 x2 * x2;
      sq = wave_tree_f(sq);
      if (l == 0u) ink_r6_4c_ar_red[rbase + wv] = sq;

      {
        const f32x4 p0 = x0 * q0_0, p1 = x1 * q1_0;
        float d0 = ((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w)) + x2 * q2_0;
        d0 = wave_tree_f(d0);
        if (l == 0u) ink_r6_4c_ar_red[rbase + 8u + wv] = d0;
      }
      {
        const f32x4 p0 = x0 * q0_1, p1 = x1 * q1_1;
        float d1 = ((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w)) + x2 * q2_1;
        d1 = wave_tree_f(d1);
        if (l == 0u) ink_r6_4c_ar_red[rbase + 16u + wv] = d1;
      }
      {
        const f32x4 p0 = x0 * q0_2, p1 = x1 * q1_2;
        float d2 = ((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w)) + x2 * q2_2;
        d2 = wave_tree_f(d2);
        if (l == 0u) ink_r6_4c_ar_red[rbase + 24u + wv] = d2;
      }
      {
        const f32x4 p0 = x0 * q0_3, p1 = x1 * q1_3;
        float d3 = ((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w)) + x2 * q2_3;
        d3 = wave_tree_f(d3);
        if (l == 0u) ink_r6_4c_ar_red[rbase + 32u + wv] = d3;
      }
    }

    /* Single LDS-only barrier for the entire fast path. */
    lbarrier();

    /* Wave-parallel norm & 4-query softmax across lanes 0..M-1 (matches ie_inkling_attn_res_r1). */
    float my_inv = 0.0f;
    float lg0 = 0.0f, lg1 = 0.0f, lg2 = 0.0f, lg3 = 0.0f;
    if (l < M) {
      const u32 rbase = l * 40u;
      float sum_sq = 0.0f;
#pragma unroll
      for (u32 x = 0u; x < 8u; x++) sum_sq += ink_r6_4c_ar_red[rbase + x];
      my_inv = 1.0f / __builtin_sqrtf(sum_sq / 2304.0f + eps);
      const float scale = my_inv * inv_sqn;

      float sd0 = 0.0f, sd1 = 0.0f, sd2 = 0.0f, sd3 = 0.0f;
#pragma unroll
      for (u32 x = 0u; x < 8u; x++) sd0 += ink_r6_4c_ar_red[rbase + 8u + x];
#pragma unroll
      for (u32 x = 0u; x < 8u; x++) sd1 += ink_r6_4c_ar_red[rbase + 16u + x];
#pragma unroll
      for (u32 x = 0u; x < 8u; x++) sd2 += ink_r6_4c_ar_red[rbase + 24u + x];
#pragma unroll
      for (u32 x = 0u; x < 8u; x++) sd3 += ink_r6_4c_ar_red[rbase + 32u + x];

      lg0 = sd0 * scale;
      lg1 = sd1 * scale;
      lg2 = sd2 * scale;
      lg3 = sd3 * scale;
    }
    const float my_prob0 = ink_r6_4c_wave_softmax(lg0, l, M);
    const float my_prob1 = ink_r6_4c_wave_softmax(lg1, l, M);
    const float my_prob2 = ink_r6_4c_wave_softmax(lg2, l, M);
    const float my_prob3 = ink_r6_4c_wave_softmax(lg3, l, M);

    /* Pass 2: weighted sum across all 4 messages (q0..q3 are dead -> accumulators reuse their VGPRs). */
    f32x4 a0_0 = z4, a1_0 = z4, a0_1 = z4, a1_1 = z4;
    f32x4 a0_2 = z4, a1_2 = z4, a0_3 = z4, a1_3 = z4;
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
    for (u32 s = 0u; s < M; s++) {
      f32x4 x0, x1;
      float x2;
      if (s == 0u) {
        x0 = c0_0; x1 = c1_0; x2 = c2_0;
      } else if (s == 1u) {
        x0 = c0_1; x1 = c1_1; x2 = c2_1;
      } else {
        const G float *sp = s < n_a ? (a + (unsigned long)s * 2304ul) : c;
        x0 = *(const G f32x4 *)(sp + th * 4u);
        x1 = *(const G f32x4 *)(sp + 1024u + th * 4u);
        x2 = sp[2048u + th];
      }
      const float inv_s = I2F(__builtin_amdgcn_readlane(F2I(my_inv), (int)s));
      const f32x4 xn0 = x0 * inv_s;
      const f32x4 xn1 = x1 * inv_s;
      const float xn2 = x2 * inv_s;

      const float p0 = I2F(__builtin_amdgcn_readlane(F2I(my_prob0), (int)s));
      a0_0 += xn0 * p0; a1_0 += xn1 * p0; a2_0 += xn2 * p0;

      const float p1 = I2F(__builtin_amdgcn_readlane(F2I(my_prob1), (int)s));
      a0_1 += xn0 * p1; a1_1 += xn1 * p1; a2_1 += xn2 * p1;

      const float p2 = I2F(__builtin_amdgcn_readlane(F2I(my_prob2), (int)s));
      a0_2 += xn0 * p2; a1_2 += xn1 * p2; a2_2 += xn2 * p2;

      const float p3 = I2F(__builtin_amdgcn_readlane(F2I(my_prob3), (int)s));
      a0_3 += xn0 * p3; a1_3 += xn1 * p3; a2_3 += xn2 * p3;
    }

    *(G f32x4 *)(out0 + th * 4u) = a0_0;
    *(G f32x4 *)(out0 + 1024u + th * 4u) = a1_0;
    out0[2048u + th] = a2_0;

    G float *out1 = out0 + 2304u;
    *(G f32x4 *)(out1 + th * 4u) = a0_1;
    *(G f32x4 *)(out1 + 1024u + th * 4u) = a1_1;
    out1[2048u + th] = a2_1;

    *(G f32x4 *)(out2 + th * 4u) = a0_2;
    *(G f32x4 *)(out2 + 1024u + th * 4u) = a1_2;
    out2[2048u + th] = a2_2;

    G float *out3 = out2 + 2304u;
    *(G f32x4 *)(out3 + th * 4u) = a0_3;
    *(G f32x4 *)(out3 + 1024u + th * 4u) = a1_3;
    out3[2048u + th] = a2_3;
    return;
  }

  /* General fallback for any n >= 1, M <= 64 (M * 5 <= 320 always fits ink_r6_4c_ar_red). */
  const int vec4_ok = ((n & 3u) == 0u);
  const G float *qptr[4u] = {q0, q0 + (unsigned long)n, q2_base, q2_base + (unsigned long)n};
  G float *bptr[4u] = {out0, out0 + (unsigned long)n, out2, out2 + (unsigned long)n};

  for (u32 s = 0u; s < M; s++) {
    const G float *sp = s < n_a ? (a + (unsigned long)s * (unsigned long)n) : c;
    float sq = 0.0f;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    if (vec4_ok) {
      for (u32 i = th * 4u; i < n; i += NT * 4u) {
        const f32x4 x4 = *(const G f32x4 *)(sp + i);
        sq += (x4.x * x4.x + x4.y * x4.y) + (x4.z * x4.z + x4.w * x4.w);
        const f32x4 p0 = x4 * (*(const G f32x4 *)(qptr[0] + i));
        d0 += (p0.x + p0.y) + (p0.z + p0.w);
        const f32x4 p1 = x4 * (*(const G f32x4 *)(qptr[1] + i));
        d1 += (p1.x + p1.y) + (p1.z + p1.w);
        const f32x4 p2 = x4 * (*(const G f32x4 *)(qptr[2] + i));
        d2 += (p2.x + p2.y) + (p2.z + p2.w);
        const f32x4 p3 = x4 * (*(const G f32x4 *)(qptr[3] + i));
        d3 += (p3.x + p3.y) + (p3.z + p3.w);
      }
    } else {
      for (u32 i = th; i < n; i += NT) {
        const float x = sp[i];
        sq += x * x;
        d0 += x * qptr[0][i];
        d1 += x * qptr[1][i];
        d2 += x * qptr[2][i];
        d3 += x * qptr[3][i];
      }
    }
    const u32 rbase = s * 40u;
    sq = wave_tree_f(sq);
    d0 = wave_tree_f(d0);
    d1 = wave_tree_f(d1);
    d2 = wave_tree_f(d2);
    d3 = wave_tree_f(d3);
    if (l == 0u) {
      ink_r6_4c_ar_red[rbase + wv] = sq;
      ink_r6_4c_ar_red[rbase + 8u + wv] = d0;
      ink_r6_4c_ar_red[rbase + 16u + wv] = d1;
      ink_r6_4c_ar_red[rbase + 24u + wv] = d2;
      ink_r6_4c_ar_red[rbase + 32u + wv] = d3;
    }
  }
  lbarrier();

  if (wv == 0u) {
    for (u32 s = l; s < M; s += 32u) {
      const u32 rbase = s * 40u;
      float sum_sq = 0.0f;
#pragma unroll
      for (u32 x = 0u; x < 8u; x++) sum_sq += ink_r6_4c_ar_red[rbase + x];
      const float inv_s = 1.0f / __builtin_sqrtf(sum_sq / (float)n + eps);
      ink_r6_4c_ar_inv[s] = inv_s;
      for (u32 k = 0u; k < 4u; k++) {
        float sum_d = 0.0f;
#pragma unroll
        for (u32 x = 0u; x < 8u; x++) sum_d += ink_r6_4c_ar_red[rbase + (1u + k) * 8u + x];
        ink_r6_4c_ar_prob[k][s] = sum_d * inv_s * inv_sqn;
      }
    }
    __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
    if (l == 0u) {
      for (u32 k = 0u; k < 4u; k++) {
        float mx = ink_r6_4c_ar_prob[k][0];
        for (u32 s = 1u; s < M; s++) mx = __builtin_fmaxf(mx, ink_r6_4c_ar_prob[k][s]);
        float ls = 0.0f;
        for (u32 s = 0u; s < M; s++) {
          const float e = __builtin_expf(ink_r6_4c_ar_prob[k][s] - mx);
          ink_r6_4c_ar_prob[k][s] = e;
          ls += e;
        }
        const float inv_l = 1.0f / ls;
        for (u32 s = 0u; s < M; s++) ink_r6_4c_ar_prob[k][s] *= inv_l;
      }
    }
  }
  lbarrier();

  if (vec4_ok) {
    const f32x4 z4 = {0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 i = th * 4u; i < n; i += NT * 4u) {
      f32x4 a0 = z4, a1 = z4, a2 = z4, a3 = z4;
      for (u32 s = 0u; s < M; s++) {
        const G float *sp = s < n_a ? (a + (unsigned long)s * (unsigned long)n) : c;
        const f32x4 xn = (*(const G f32x4 *)(sp + i)) * ink_r6_4c_ar_inv[s];
        a0 += xn * ink_r6_4c_ar_prob[0u][s];
        a1 += xn * ink_r6_4c_ar_prob[1u][s];
        a2 += xn * ink_r6_4c_ar_prob[2u][s];
        a3 += xn * ink_r6_4c_ar_prob[3u][s];
      }
      *(G f32x4 *)(bptr[0] + i) = a0;
      *(G f32x4 *)(bptr[1] + i) = a1;
      *(G f32x4 *)(bptr[2] + i) = a2;
      *(G f32x4 *)(bptr[3] + i) = a3;
    }
  } else {
    for (u32 i = th; i < n; i += NT) {
      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
      for (u32 s = 0u; s < M; s++) {
        const G float *sp = s < n_a ? (a + (unsigned long)s * (unsigned long)n) : c;
        const float xn = sp[i] * ink_r6_4c_ar_inv[s];
        a0 += xn * ink_r6_4c_ar_prob[0u][s];
        a1 += xn * ink_r6_4c_ar_prob[1u][s];
        a2 += xn * ink_r6_4c_ar_prob[2u][s];
        a3 += xn * ink_r6_4c_ar_prob[3u][s];
      }
      bptr[0][i] = a0;
      bptr[1][i] = a1;
      bptr[2][i] = a2;
      bptr[3][i] = a3;
    }
  }
}

/* ============================================================================
 * Kernel 2: ie_inkling_lc_prenorm_q8_4x_r6
 * Fuses unweighted RMSNorm + Q8_0 quant across n_ctrl = 1..4 controllers in
 * ONE launch. When SHARED_INP (flags & 1u) is set, reads/normalizes/quantizes
 * inp0 ONCE in bx==0 (saving 3x/4x inp reads and RMSNorm work!).
 * ========================================================================== */

static LDS float ink_r6_4c_pn_red[8u];
static LDS float ink_r6_4c_pn_sh[8u][128u] __attribute__((aligned(16)));

static inline __attribute__((always_inline)) float ink_r6_4c_pn_wg_sum_1b(float v, u32 wv, u32 l) {
  v = wave_tree_f(v);
  if (l == 0u) ink_r6_4c_pn_red[wv] = v;
  lbarrier();
  float s = 0.0f;
#pragma unroll
  for (u32 w = 0u; w < 8u; w++) s += ink_r6_4c_pn_red[w];
  return s;
}

/* Exact quant_wave that broadcasts the already-quantized block to 1..4 Q8_0 destinations
 * using packed 32-bit global_store_b32 writes (4x fewer store ops vs byte stores). */
static inline __attribute__((always_inline)) void ink_r6_4c_quant_wave_n(
    float v, u32 b, G u8 *d0, G u8 *d1, G u8 *d2, G u8 *d3, u32 n_dst, u32 nb_tot) {
  const u32 j = lane();
  const float amax = wave_max_pos(__builtin_fabsf(v));
  const float d = amax / 127.0f;
  const int qi = d != 0.0f ? (int)__builtin_roundf(v / d) : 0;
  const u32 b0 = (u32)(u8)(i8)qi;
  const u32 p2 = b0 | ((u32)shl_i((int)b0, 1) << 8u);
  const u32 p4 = p2 | ((u32)shl_i((int)p2, 2) << 16u);
  const u32 s = wave_tree_u((u32)qi);
  const u32 q_off4 = b * 8u + (j >> 2u);
  const u32 d_off = 32u * nb_tot;
  const u32 s_off = 36u * nb_tot;

  if ((j & 3u) == 0u) ((G u32 *)d0)[q_off4] = p4;
  if (j == 0u) {
    ((G float *)(d0 + d_off))[b] = d;
    ((G u32 *)(d0 + s_off))[b] = s;
  }
  if (n_dst > 1u && d1) {
    if ((j & 3u) == 0u) ((G u32 *)d1)[q_off4] = p4;
    if (j == 0u) {
      ((G float *)(d1 + d_off))[b] = d;
      ((G u32 *)(d1 + s_off))[b] = s;
    }
  }
  if (n_dst > 2u && d2) {
    if ((j & 3u) == 0u) ((G u32 *)d2)[q_off4] = p4;
    if (j == 0u) {
      ((G float *)(d2 + d_off))[b] = d;
      ((G u32 *)(d2 + s_off))[b] = s;
    }
  }
  if (n_dst > 3u && d3) {
    if ((j & 3u) == 0u) ((G u32 *)d3)[q_off4] = p4;
    if (j == 0u) {
      ((G float *)(d3 + d_off))[b] = d;
      ((G u32 *)(d3 + s_off))[b] = s;
    }
  }
}

static inline __attribute__((always_inline)) void ink_r6_4c_pn_quant_chunk4(
    f32x4 y4, u32 wv, u32 l, u32 b0, G u8 *d0, G u8 *d1, G u8 *d2, G u8 *d3, u32 n_dst, u32 nb_tot) {
  *(LDSP f32x4 *)&ink_r6_4c_pn_sh[wv][l * 4u] = y4;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 m = 0u; m < 4u; m++) {
    ink_r6_4c_quant_wave_n(ink_r6_4c_pn_sh[wv][32u * m + l], b0 + m, d0, d1, d2, d3, n_dst, nb_tot);
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

KERNEL ie_inkling_lc_prenorm_q8_4x_r6(
    const G float *inp0, const G float *inp1, const G float *inp2, const G float *inp3,
    const G float *msg0, const G float *msg1, const G float *msg2, const G float *msg3,
    G u8 *q0, G u8 *q1, G u8 *q2, G u8 *q3,
    u32 dim, float eps, u32 n_ctrl, u32 flags,
    u32 is0, u32 is1, u32 is2, u32 is3,
    u32 ms0, u32 ms1, u32 ms2, u32 ms3, u32 qs) {
  if (n_ctrl == 0u || n_ctrl > 4u || dim == 0u || (dim & 31u) != 0u) return;

  const u32 bx = wgid();
  const u32 shared = flags & INK_R6_4C_PN_SHARED;
  const u32 no_rep = flags & INK_R6_4C_PN_NOREP;
  const u32 n_wgx = shared ? (1u + n_ctrl) : (2u * n_ctrl);
  if (bx >= n_wgx) return;

  const G float *x = (const G float *)0;
  G u8 *d0 = (G u8 *)0;
  G u8 *d1 = (G u8 *)0;
  G u8 *d2 = (G u8 *)0;
  G u8 *d3 = (G u8 *)0;
  u32 n_dst = 1u;
  u32 is_msg = 0u;

  if (shared) {
    if (bx == 0u) {
      x = inp0 ? TOKC(inp0, is0) : (const G float *)0;
      d0 = TOK(q0, qs);
      if (!no_rep) {
        d1 = (n_ctrl > 1u) ? TOK(q1, qs) : (G u8 *)0;
        d2 = (n_ctrl > 2u) ? TOK(q2, qs) : (G u8 *)0;
        d3 = (n_ctrl > 3u) ? TOK(q3, qs) : (G u8 *)0;
        n_dst = n_ctrl;
      }
      is_msg = 0u;
    } else {
      const u32 c = bx - 1u;
      is_msg = 1u;
      if (c == 0u) {
        x = msg0 ? TOKC(msg0, ms0) : (const G float *)0;
        d0 = TOK(q0, qs);
      } else if (c == 1u) {
        x = msg1 ? TOKC(msg1, ms1) : (const G float *)0;
        d0 = TOK(q1, qs);
      } else if (c == 2u) {
        x = msg2 ? TOKC(msg2, ms2) : (const G float *)0;
        d0 = TOK(q2, qs);
      } else {
        x = msg3 ? TOKC(msg3, ms3) : (const G float *)0;
        d0 = TOK(q3, qs);
      }
    }
  } else {
    is_msg = (bx >= n_ctrl) ? 1u : 0u;
    const u32 c = is_msg ? (bx - n_ctrl) : bx;
    if (!is_msg) {
      if (c == 0u) {
        x = inp0 ? TOKC(inp0, is0) : (const G float *)0;
        d0 = TOK(q0, qs);
      } else if (c == 1u) {
        x = inp1 ? TOKC(inp1, is1) : (const G float *)0;
        d0 = TOK(q1, qs);
      } else if (c == 2u) {
        x = inp2 ? TOKC(inp2, is2) : (const G float *)0;
        d0 = TOK(q2, qs);
      } else {
        x = inp3 ? TOKC(inp3, is3) : (const G float *)0;
        d0 = TOK(q3, qs);
      }
    } else {
      if (c == 0u) {
        x = msg0 ? TOKC(msg0, ms0) : (const G float *)0;
        d0 = TOK(q0, qs);
      } else if (c == 1u) {
        x = msg1 ? TOKC(msg1, ms1) : (const G float *)0;
        d0 = TOK(q1, qs);
      } else if (c == 2u) {
        x = msg2 ? TOKC(msg2, ms2) : (const G float *)0;
        d0 = TOK(q2, qs);
      } else {
        x = msg3 ? TOKC(msg3, ms3) : (const G float *)0;
        d0 = TOK(q3, qs);
      }
    }
  }

  if (!x || !d0) return;

  const u32 th = tid(), wv = th >> 5u, l = lane();
  const u32 nb = dim >> 5u;
  const u32 nb_tot = nb << 1u;
  const u32 q_blk_off = is_msg ? nb : 0u;

  /* Fast path: Inkling-2 dim == 2304 (100% of loop controller calls; 1 lbarrier, 0 spills). */
  if (dim == 2304u) {
    const f32x4 x0 = *(const G f32x4 *)(x + th * 4u);
    const f32x4 x1 = *(const G f32x4 *)(x + 1024u + th * 4u);
    const float x2 = x[2048u + th];

    float ss = (x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w);
    ss += (x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w);
    ss += x2 * x2;
    ss = ink_r6_4c_pn_wg_sum_1b(ss, wv, l);
    const float s = 1.0f / __builtin_sqrtf(ss / 2304.0f + eps);

    const f32x4 y0 = x0 * s;
    const f32x4 y1 = x1 * s;
    const float y2 = x2 * s;
    ink_r6_4c_pn_quant_chunk4(y0, wv, l, q_blk_off + wv * 4u, d0, d1, d2, d3, n_dst, 144u);
    ink_r6_4c_pn_quant_chunk4(y1, wv, l, q_blk_off + 32u + wv * 4u, d0, d1, d2, d3, n_dst, 144u);
    ink_r6_4c_quant_wave_n(y2, q_blk_off + 64u + wv, d0, d1, d2, d3, n_dst, 144u);
    return;
  }

  /* Small-dim path (dim <= 256). */
  if (dim <= NT) {
    const float xi = (th < dim) ? x[th] : 0.0f;
    const float ss = ink_r6_4c_pn_wg_sum_1b(xi * xi, wv, l);
    const float s = 1.0f / __builtin_sqrtf(ss / (float)dim + eps);
    if (th < dim) {
      ink_r6_4c_quant_wave_n(xi * s, q_blk_off + wv, d0, d1, d2, d3, n_dst, nb_tot);
    }
    return;
  }

  /* General path for any (dim & 31u) == 0u. */
  float ss = 0.0f;
  u32 k = 0u;
  for (; (k + 1u) * 1024u <= dim; k++) {
    const f32x4 vk = *(const G f32x4 *)(x + k * 1024u + th * 4u);
    ss += (vk.x * vk.x + vk.y * vk.y) + (vk.z * vk.z + vk.w * vk.w);
  }
  for (u32 i = k * 1024u + th; i < dim; i += NT) {
    const float xi = x[i];
    ss += xi * xi;
  }
  ss = ink_r6_4c_pn_wg_sum_1b(ss, wv, l);
  const float s = 1.0f / __builtin_sqrtf(ss / (float)dim + eps);

  for (u32 c = 0u; c < k; c++) {
    const u32 base = c * 1024u + th * 4u;
    const f32x4 y4 = (*(const G f32x4 *)(x + base)) * s;
    ink_r6_4c_pn_quant_chunk4(y4, wv, l, q_blk_off + c * 32u + wv * 4u, d0, d1, d2, d3, n_dst, nb_tot);
  }
  for (u32 b = k * 32u + wv; b < nb; b += 8u) {
    const u32 i = b * 32u + l;
    ink_r6_4c_quant_wave_n(x[i] * s, q_blk_off + b, d0, d1, d2, d3, n_dst, nb_tot);
  }
}

/* ============================================================================
 * Kernel 3: ie_inkling_lc_down_4x_r6
 * Batches n_ctrl = 1..4 rank-64 Q8_0 down projections [rows, nb*32] into ONE
 * launch (grid = (n_ctrl * rows, 1, 1); 192 WGs for QKV, 256 WGs for 4-ctrl).
 *   - T == 1: single-RTT 1 WG/row (for nb=144 <= 256: 0 loop iters, 1 lbarrier).
 *   - T >= 2: 8-token chunked weight reuse (loads each weight block ONCE per 8
 *             tokens; all <= 8 waves reduce distinct tokens in parallel via 8 KiB LDS).
 *   - shared_inp != 0u: blocks [0..nb/2-1] of ALL controllers read xq0, keeping
 *     the shared inp_q hot in L1/L2 and pairing directly with prenorm NO_REP_INP.
 * Bitwise-exact (0-bit-diff) vs ie_gemv_q8q8 / ie_gemv_q8q8_t.
 * ========================================================================== */

static LDS float ink_r6_4c_dn_lds[INK_R6_4C_DN_CHUNK][8u][32u] __attribute__((aligned(16)));

KERNEL ie_inkling_lc_down_4x_r6(
    const G u32 *qw0, const G f16 *qs0, const G u8 *xq0, G float *y0,
    const G u32 *qw1, const G f16 *qs1, const G u8 *xq1, G float *y1,
    const G u32 *qw2, const G f16 *qs2, const G u8 *xq2, G float *y2,
    const G u32 *qw3, const G f16 *qs3, const G u8 *xq3, G float *y3,
    u32 rows, u32 nb, u32 T, u32 n_ctrl, u32 shared_inp,
    u32 xs, u32 ys) {
#pragma clang fp contract(off)
  _Static_assert(sizeof(ink_r6_4c_dn_lds) == 8192u, "ink_r6_4c down LDS must be 8 KiB");
  if (T == 0u || rows == 0u || nb == 0u || n_ctrl == 0u || n_ctrl > 4u) return;

  const u32 g = wgid();
  if (g >= n_ctrl * rows) return;

  const u32 c = g / rows;
  const u32 r = g - c * rows;

  const G u32 *qw = qw0;
  const G f16 *qs = qs0;
  const G u8 *xq = xq0;
  G float *y = y0;
  if (c == 1u) {
    qw = qw1; qs = qs1; xq = xq1; y = y1;
  } else if (c == 2u) {
    qw = qw2; qs = qs2; xq = xq2; y = y2;
  } else if (c == 3u) {
    qw = qw3; qs = qs3; xq = xq3; y = y3;
  }
  if (!qw || !qs || !xq || !y) return;

  const G u8 *xq_lo = (shared_inp && ((nb & 1u) == 0u) && xq0) ? xq0 : xq;
  const u32 nb_half = nb >> 1u;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane();
  const u32 nchunks = (nb + 255u) >> 8u;
  const G u32 *row_qw = qw + (unsigned long)r * (unsigned long)nb * 8ul;
  const G f16 *row_qs = qs + (unsigned long)r * (unsigned long)nb;

  /* Fast path: T == 1 decode (zero token-loop overhead; 1 RTT for nb <= 256). */
  if (T == 1u) {
    float acc = 0.0f;
    for (u32 ch = 0u; ch < nchunks; ch++) {
      const u32 base = ch << 8u;
      const u32 b = base + t0;
      float term = 0.0f;
      if (b < nb) {
        const G u32 *wp = row_qw + (unsigned long)b * 8ul;
        const u32x4 w0 = ink_r6_4c_ld4(wp);
        const u32x4 w1 = ink_r6_4c_ld4(wp + 4u);
        const float sw = (float)row_qs[b];

        const G u8 *xsrc = (b < nb_half) ? xq_lo : xq;
        const G u32 *aw = (const G u32 *)xsrc;
        const u32x4 a0 = ink_r6_4c_ld4(aw + (unsigned long)b * 8ul);
        const u32x4 a1 = ink_r6_4c_ld4(aw + (unsigned long)b * 8ul + 4ul);
        const float da = ((const G float *)(xsrc + (unsigned long)nb * 32ul))[b];

        const float sc = sw * da;
        const int dot = ink_r6_4c_dot8_ss(w0, w1, a0, a1);
        term = sc * (float)dot;
      }
      ink_r6_4c_dn_lds[0u][w][l] = term;
      lbarrier();

      if (w == 0u) {
        const u32 rem = nb - base;
        const u32 nw_act = rem >= 256u ? 8u : ((rem + 31u) >> 5u);
        for (u32 k = 0u; k < nw_act; k++) {
          if (base + (k << 5u) + l < nb) acc = acc + ink_r6_4c_dn_lds[0u][k][l];
        }
      }
      if (ch + 1u < nchunks) lbarrier();
    }
    if (w == 0u) {
      acc = wave_tree_f(acc);
      if (l == 0u) y[r] = acc;
    }
    return;
  }

  /* Multi-token verify path (T >= 2, optimal for T = 2..16): 8 tokens per weight read. */
  for (u32 t_base = 0u; t_base < T; t_base += INK_R6_4C_DN_CHUNK) {
    const u32 tn = (T - t_base) < INK_R6_4C_DN_CHUNK ? (T - t_base) : INK_R6_4C_DN_CHUNK;
    float acc = 0.0f;

    for (u32 ch = 0u; ch < nchunks; ch++) {
      const u32 base = ch << 8u;
      const u32 b = base + t0;
      if (b < nb) {
        const G u32 *wp = row_qw + (unsigned long)b * 8ul;
        const u32x4 w0 = ink_r6_4c_ld4(wp);
        const u32x4 w1 = ink_r6_4c_ld4(wp + 4u);
        const float sw = (float)row_qs[b];
        const G u8 *xbase = (b < nb_half) ? xq_lo : xq;

        u32 j = 0u;
        for (; j + 1u < tn; j += 2u) {
          const G u8 *xj0 = xbase + (unsigned long)(t_base + j) * (unsigned long)xs;
          const G u8 *xj1 = xbase + (unsigned long)(t_base + j + 1u) * (unsigned long)xs;
          const G u32 *aw0 = (const G u32 *)xj0;
          const G u32 *aw1 = (const G u32 *)xj1;
          const u32x4 a0_0 = ink_r6_4c_ld4(aw0 + (unsigned long)b * 8ul);
          const u32x4 a1_0 = ink_r6_4c_ld4(aw0 + (unsigned long)b * 8ul + 4ul);
          const float da0 = ((const G float *)(xj0 + (unsigned long)nb * 32ul))[b];
          const u32x4 a0_1 = ink_r6_4c_ld4(aw1 + (unsigned long)b * 8ul);
          const u32x4 a1_1 = ink_r6_4c_ld4(aw1 + (unsigned long)b * 8ul + 4ul);
          const float da1 = ((const G float *)(xj1 + (unsigned long)nb * 32ul))[b];
          const float sc0 = sw * da0;
          const float sc1 = sw * da1;
          const int d0 = ink_r6_4c_dot8_ss(w0, w1, a0_0, a1_0);
          const int d1 = ink_r6_4c_dot8_ss(w0, w1, a0_1, a1_1);
          ink_r6_4c_dn_lds[j][w][l] = sc0 * (float)d0;
          ink_r6_4c_dn_lds[j + 1u][w][l] = sc1 * (float)d1;
        }
        if (j < tn) {
          const G u8 *xj0 = xbase + (unsigned long)(t_base + j) * (unsigned long)xs;
          const G u32 *aw0 = (const G u32 *)xj0;
          const u32x4 a0_0 = ink_r6_4c_ld4(aw0 + (unsigned long)b * 8ul);
          const u32x4 a1_0 = ink_r6_4c_ld4(aw0 + (unsigned long)b * 8ul + 4ul);
          const float da0 = ((const G float *)(xj0 + (unsigned long)nb * 32ul))[b];
          const float sc0 = sw * da0;
          const int d0 = ink_r6_4c_dot8_ss(w0, w1, a0_0, a1_0);
          ink_r6_4c_dn_lds[j][w][l] = sc0 * (float)d0;
        }
      }

      lbarrier();

      /* Wave w reduces token (t_base + w) in exact ascending block order. */
      if (w < tn) {
        const u32 rem = nb - base;
        const u32 nw_act = rem >= 256u ? 8u : ((rem + 31u) >> 5u);
        for (u32 k = 0u; k < nw_act; k++) {
          if (base + (k << 5u) + l < nb) acc = acc + ink_r6_4c_dn_lds[w][k][l];
        }
      }

      if (ch + 1u < nchunks) lbarrier();
    }

    if (w < tn) {
      const u32 tok = t_base + w;
      acc = wave_tree_f(acc);
      if (l == 0u) {
        G float *yt = (G float *)((G u8 *)y + (unsigned long)tok * (unsigned long)ys);
        yt[r] = acc;
      }
    }

    if (t_base + INK_R6_4C_DN_CHUNK < T) lbarrier();
  }
}

/* ============================================================================
 * Kernel 4: ie_inkling_lc_wfg_gate_norm_r6
 * Fuses quant(h[64]) + wfg[2*dim,64] + loop_gate(inp,msg,fg -> tmp) +
 * downstream RMSNorm(tmp, nw -> ny [NULL if dead_b], nq [Q8_0]) across
 * n_ctrl = 1..4 controllers in ONE launch!
 *   - T == 1:    grid = ((dim+31)/32, n_ctrl, 1), 100% bitwise vs GEMV+gate+RMSNorm
 *   - T = 2..16: grid = ((dim+31)/32, n_ctrl, 1), weights read ONCE for all T tokens,
 *                100% bitwise vs GEMV_T+gate+RMSNorm
 *   - T >= 17:   grid = ((dim+127)/128, n_ctrl*((T+63)/64), 1), 128x64 swapped WMMA
 *                + per-tile L2-hot RMSNorm tail, 100% bitwise vs R2+RMSNorm
 * ========================================================================== */

typedef union {
  struct {
    u8 x[INK_R6_4C_NB2_T * INK_R6_4C_NB2_XLS] __attribute__((aligned(16)));
    float dx[2u][INK_R6_4C_NB2_T] __attribute__((aligned(32)));
  } wfg; /* 5,632 B */
  struct {
    float red[8u];
    float sh[8u][128u] __attribute__((aligned(16)));
  } rn; /* 4,128 B */
} ink_r6_4c_wfg_lds_u;

static LDS ink_r6_4c_wfg_lds_u ink_r6_4c_wfg_lds __attribute__((aligned(32)));
static LDS u32 ink_r6_4c_wfg_last;

static inline __attribute__((always_inline)) float ink_r6_4c_rn_wg_sum_1b(float v, u32 wv, u32 l) {
  v = wave_tree_f(v);
  if (l == 0u) ink_r6_4c_wfg_lds.rn.red[wv] = v;
  lbarrier();
  float s = 0.0f;
#pragma unroll
  for (u32 w = 0u; w < 8u; w++) s += ink_r6_4c_wfg_lds.rn.red[w];
  return s;
}

static inline __attribute__((always_inline)) void ink_r6_4c_rn_quant_chunk4(
    f32x4 y4, u32 wv, u32 l, u32 b0, G u8 *q, u32 nb) {
  *(LDSP f32x4 *)&ink_r6_4c_wfg_lds.rn.sh[wv][l * 4u] = y4;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 m = 0u; m < 4u; m++) quant_wave(ink_r6_4c_wfg_lds.rn.sh[wv][32u * m + l], b0 + m, q, nb);
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

/* Single-token 1-WG RMSNorm + optional F32 write (ny) + optional Q8_0 write (nq).
 * Exact arithmetic & reduction order of ie_inkling_rmsnorm_mt_r1 (0-bit-diff).
 * Caller must issue lbarrier() between consecutive calls on different tokens. */
static inline __attribute__((always_inline)) void ink_r6_4c_rmsnorm_1tok(
    const G float *x, const G float *nw, G float *ny, G u8 *nq, u32 n, float eps, u32 th, u32 wv, u32 l) {
  const u32 nb = n >> 5u;

  if (n == 2304u) {
    const f32x4 x0 = *(const G f32x4 *)(x + th * 4u);
    const f32x4 x1 = *(const G f32x4 *)(x + 1024u + th * 4u);
    const float x2 = x[2048u + th];
    const f32x4 w0 = *(const G f32x4 *)(nw + th * 4u);
    const f32x4 w1 = *(const G f32x4 *)(nw + 1024u + th * 4u);
    const float w2 = nw[2048u + th];

    float ss = (x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w);
    ss += (x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w);
    ss += x2 * x2;
    ss = ink_r6_4c_rn_wg_sum_1b(ss, wv, l);
    const float s = 1.0f / __builtin_sqrtf(ss / 2304.0f + eps);

    const f32x4 y0 = x0 * s * w0;
    const f32x4 y1 = x1 * s * w1;
    const float y2 = x2 * s * w2;
    if (ny) {
      *(G f32x4 *)(ny + th * 4u) = y0;
      *(G f32x4 *)(ny + 1024u + th * 4u) = y1;
      ny[2048u + th] = y2;
    }
    if (nq) {
      ink_r6_4c_rn_quant_chunk4(y0, wv, l, wv * 4u, nq, 72u);
      ink_r6_4c_rn_quant_chunk4(y1, wv, l, 32u + wv * 4u, nq, 72u);
      quant_wave(y2, 64u + wv, nq, 72u);
    }
    return;
  }

  if (n <= NT) {
    const float xi = (th < n) ? x[th] : 0.0f;
    const float wi = (th < n) ? nw[th] : 0.0f;
    const float ss = ink_r6_4c_rn_wg_sum_1b(xi * xi, wv, l);
    const float s = 1.0f / __builtin_sqrtf(ss / (float)n + eps);
    if (th < n) {
      const float v = xi * s * wi;
      if (ny) ny[th] = v;
      if (nq) quant_wave(v, wv, nq, nb);
    }
    return;
  }

  float ss = 0.0f;
  u32 k = 0u;
  for (; (k + 1u) * 1024u <= n; k++) {
    const f32x4 vk = *(const G f32x4 *)(x + k * 1024u + th * 4u);
    ss += (vk.x * vk.x + vk.y * vk.y) + (vk.z * vk.z + vk.w * vk.w);
  }
  for (u32 i = k * 1024u + th; i < n; i += NT) {
    const float xi = x[i];
    ss += xi * xi;
  }
  ss = ink_r6_4c_rn_wg_sum_1b(ss, wv, l);
  const float s = 1.0f / __builtin_sqrtf(ss / (float)n + eps);

  for (u32 ci = 0u; ci < k; ci++) {
    const u32 base = ci * 1024u + th * 4u;
    const f32x4 y4 = (*(const G f32x4 *)(x + base)) * s * (*(const G f32x4 *)(nw + base));
    if (ny) *(G f32x4 *)(ny + base) = y4;
    if (nq) ink_r6_4c_rn_quant_chunk4(y4, wv, l, ci * 32u + wv * 4u, nq, nb);
  }
  for (u32 b = k * 32u + wv; b < nb; b += 8u) {
    const u32 i = b * 32u + l;
    const float v = x[i] * s * nw[i];
    if (ny) ny[i] = v;
    if (nq) quant_wave(v, b, nq, nb);
  }
}

/* Cross-WG release + last-WG atomic barrier + in-L2 RMSNorm/Q8 tail over [t0, t1). */
static inline __attribute__((always_inline)) void ink_r6_4c_wfg_norm_tail(
    const G float *tmp, const G float *nw, G float *ny, G u8 *nq, G u32 *cnt_ptr,
    u32 n_wgx, u32 dim, float eps, u32 t0, u32 t1, u32 ts, u32 ys, u32 qs_out, u32 th, u32 wv, u32 l) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (th == 0u) {
    ink_r6_4c_wfg_last =
        (__scoped_atomic_fetch_add(cnt_ptr, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == n_wgx - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r6_4c_wfg_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  for (u32 t = t0; t < t1; t++) {
    const G float *xt = (const G float *)((const G u8 *)tmp + (unsigned long)t * (unsigned long)ts);
    G float *nyt = ny ? (G float *)((G u8 *)ny + (unsigned long)t * (unsigned long)ys) : (G float *)0;
    G u8 *nqt = nq ? (nq + (unsigned long)t * (unsigned long)qs_out) : (G u8 *)0;
    ink_r6_4c_rmsnorm_1tok(xt, nw, nyt, nqt, dim, eps, th, wv, l);
    if (t + 1u < t1) lbarrier();
  }
  if (th == 0u) __scoped_atomic_store_n(cnt_ptr, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
}

/* Decode path (T == 1): 32 outputs/WG (4/wave, 8 lanes/output), 100% lane util, 0-bit-diff. */
static inline __attribute__((always_inline)) void ink_r6_4c_wfg_gate_dec1(
    const G u32 *qw, const G f16 *qs, const G float *h, const G float *inp, const G float *msg,
    G float *tmp, u32 dim, u32 bx, u32 w, u32 l) {
#pragma clang fp contract(off)
  if (w < 2u) {
    const float hv = h[(w << 5u) + l];
    ink_r6_4c_quant_pack_lds(hv, l, ink_r6_4c_wfg_lds.wfg.x + (w << 5u), &ink_r6_4c_wfg_lds.wfg.dx[w][0u]);
  }

  const u32 grp = l >> 3u;
  const u32 is_gain = (l >> 2u) & 1u;
  const u32 sub = l & 3u;
  const u32 blk = sub >> 1u;
  const u32 i = bx * 32u + (w << 2u) + grp;

  u32x4 wv = (u32x4)(0u);
  float sw = 0.0f;
  float inp_val = 0.0f, msg_val = 0.0f;
  if (i < dim) {
    const u32 row = is_gain ? (dim + i) : i;
    wv = ink_r6_4c_ld4((const G u8 *)qw + (unsigned long)row * 64ul + (unsigned long)(sub << 4u));
    sw = (float)qs[(unsigned long)row * 2ul + blk];
    if ((l & 7u) == 0u) {
      inp_val = inp[i];
      msg_val = msg[i];
    }
  }

  lbarrier();

  if (i < dim) {
    const u32x4 av = *(const LDSP u32x4 *)(ink_r6_4c_wfg_lds.wfg.x + (sub << 4u));
    const u32 d_half =
        ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(0u, wv.x, av.x), wv.y, av.y), wv.z, av.z), wv.w, av.w);
    const int dot = (int)d_half + shl_i((int)d_half, 1);
    const float sc = sw * ink_r6_4c_wfg_lds.wfg.dx[blk][0u];
    const float term = sc * (float)dot;
    const float sum = term + I2F(shl_i(F2I(term), 2));
    const float gain_val = I2F(shl_i(F2I(sum), 4));
    if ((l & 7u) == 0u) tmp[i] = ink_r6_4c_loop_gate1(inp_val, msg_val, sum, gain_val);
  }
}

/* Verify path (2 <= T <= 16): reads wfg ONCE into VGPRs for all T tokens (Tx weight reuse),
 * exact T=1 arithmetic per token (0-bit-diff vs ie_quant_q8 + ie_gemv_q8q8_t + ie_loop_gate). */
static inline __attribute__((always_inline)) void ink_r6_4c_wfg_gate_ver16(
    const G u32 *qw, const G f16 *qs, const G float *h, const G float *inp, const G float *msg,
    G float *tmp, u32 dim, u32 T, u32 hs, u32 is, u32 ms, u32 ts, u32 bx, u32 w, u32 l) {
#pragma clang fp contract(off)
  for (u32 tk = w; tk < T; tk += 8u) {
    const G float *ht = (const G float *)((const G u8 *)h + (unsigned long)tk * (unsigned long)hs);
    LDSP u8 *dst = ink_r6_4c_wfg_lds.wfg.x + tk * INK_R6_4C_NB2_XLS;
    ink_r6_4c_quant_pack_lds(ht[l], l, dst, &ink_r6_4c_wfg_lds.wfg.dx[0u][tk]);
    ink_r6_4c_quant_pack_lds(ht[32u + l], l, dst + 32u, &ink_r6_4c_wfg_lds.wfg.dx[1u][tk]);
  }

  const u32 grp = l >> 3u;
  const u32 is_gain = (l >> 2u) & 1u;
  const u32 sub = l & 3u;
  const u32 blk = sub >> 1u;
  const u32 i = bx * 32u + (w << 2u) + grp;

  u32x4 wv = (u32x4)(0u);
  float sw = 0.0f;
  if (i < dim) {
    const u32 row = is_gain ? (dim + i) : i;
    wv = ink_r6_4c_ld4((const G u8 *)qw + (unsigned long)row * 64ul + (unsigned long)(sub << 4u));
    sw = (float)qs[(unsigned long)row * 2ul + blk];
  }

  lbarrier();

  if (i < dim) {
    for (u32 t = 0u; t < T; t++) {
      const u32x4 av = *(const LDSP u32x4 *)(ink_r6_4c_wfg_lds.wfg.x + t * INK_R6_4C_NB2_XLS + (sub << 4u));
      const u32 d_half =
          ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(0u, wv.x, av.x), wv.y, av.y), wv.z, av.z), wv.w, av.w);
      const int dot = (int)d_half + shl_i((int)d_half, 1);
      const float sc = sw * ink_r6_4c_wfg_lds.wfg.dx[blk][t];
      const float term = sc * (float)dot;
      const float sum = term + I2F(shl_i(F2I(term), 2));
      const float gain_val = I2F(shl_i(F2I(sum), 4));
      if ((l & 7u) == 0u) {
        const float a_val = ((const G float *)((const G u8 *)inp + (unsigned long)t * (unsigned long)is))[i];
        const float c_val = ((const G float *)((const G u8 *)msg + (unsigned long)t * (unsigned long)ms))[i];
        ((G float *)((G u8 *)tmp + (unsigned long)t * (unsigned long)ts))[i] =
            ink_r6_4c_loop_gate1(a_val, c_val, sum, gain_val);
      }
    }
  }
}

/* Prefill path (T >= 17): 128 rows x 64 tokens swapped WMMA (A=X, B=W).
 * Sequential forget-then-gain WMMA saves 16 VGPRs vs R2 -> <= 84 VGPR, 0 spills,
 * bitwise-exact (0-bit-diff) vs ie_inkling_lc_wfg_gate_r2. */
static inline __attribute__((always_inline)) void ink_r6_4c_wfg_gate_pf(
    const G u32 *qw, const G f16 *qs, const G float *h, const G float *inp, const G float *msg,
    G float *tmp, u32 dim, u32 T, u32 hs, u32 is, u32 ms, u32 ts, u32 rbase, u32 tbase, u32 w, u32 l) {
  const u32 r16 = l & 15u, hh = l >> 4u;

#pragma unroll
  for (u32 k = 0u; k < 8u; k++) {
    const u32 tk = (w << 3u) + k;
    const u32 gt = (tbase + tk < T) ? (tbase + tk) : (T - 1u);
    const G float *ht = (const G float *)((const G u8 *)h + (unsigned long)gt * (unsigned long)hs);
    LDSP u8 *dst = ink_r6_4c_wfg_lds.wfg.x + tk * INK_R6_4C_NB2_XLS;
    ink_r6_4c_quant_pack_lds(ht[l], l, dst, &ink_r6_4c_wfg_lds.wfg.dx[0u][tk]);
    ink_r6_4c_quant_pack_lds(ht[32u + l], l, dst + 32u, &ink_r6_4c_wfg_lds.wfg.dx[1u][tk]);
  }

  const u32 r = rbase + (w << 4u) + r16;
  const u32 gr = r < dim ? r : (dim - 1u);
  const u32 gr_g = dim + gr;

  const G u8 *wpf = (const G u8 *)qw + (unsigned long)gr * 64ul + (unsigned long)(hh << 4u);
  const u32x4 wf0 = ink_r6_4c_ld4(wpf);
  const u32x4 wf1 = ink_r6_4c_ld4(wpf + 32ul);
  const u32 swf_raw = *(const G u32 *)(qs + (unsigned long)gr * 2ul);
  const ink_r6_4c_f16x2 swf_p = __builtin_bit_cast(ink_r6_4c_f16x2, swf_raw);
  const float swf0 = (float)swf_p.x;
  const float swf1 = (float)swf_p.y;

  const G u8 *wpg = (const G u8 *)qw + (unsigned long)gr_g * 64ul + (unsigned long)(hh << 4u);
  const u32x4 wg0 = ink_r6_4c_ld4(wpg);
  const u32x4 wg1 = ink_r6_4c_ld4(wpg + 32ul);
  const u32 swg_raw = *(const G u32 *)(qs + (unsigned long)gr_g * 2ul);
  const ink_r6_4c_f16x2 swg_p = __builtin_bit_cast(ink_r6_4c_f16x2, swg_raw);
  const float swg0 = (float)swg_p.x;
  const float swg1 = (float)swg_p.y;

  lbarrier();

  const v8i zm = (v8i)(0x4B400000);
  const v2i bf0_0 = {(int)wf0.x, (int)wf0.y}, bf0_1 = {(int)wf0.z, (int)wf0.w};
  const v2i bf1_0 = {(int)wf1.x, (int)wf1.y}, bf1_1 = {(int)wf1.z, (int)wf1.w};
  const v2i bg0_0 = {(int)wg0.x, (int)wg0.y}, bg0_1 = {(int)wg0.z, (int)wg0.w};
  const v2i bg1_0 = {(int)wg1.x, (int)wg1.y}, bg1_1 = {(int)wg1.z, (int)wg1.w};

#pragma unroll 1
  for (u32 j = 0u; j < 4u; j++) {
    const u32 t_j0 = tbase + (j << 4u);
    if (t_j0 >= T) break;

    const LDSP u8 *xp = ink_r6_4c_wfg_lds.wfg.x + ((j << 4u) + r16) * INK_R6_4C_NB2_XLS + (hh << 4u);
    const u32x4 x0 = *(const LDSP u32x4 *)(xp);
    const u32x4 x1 = *(const LDSP u32x4 *)(xp + 32u);
    const f8 dx0 = *(const LDSP f8 *)(ink_r6_4c_wfg_lds.wfg.dx[0u] + (j << 4u) + (hh << 3u));
    const f8 dx1 = *(const LDSP f8 *)(ink_r6_4c_wfg_lds.wfg.dx[1u] + (j << 4u) + (hh << 3u));

    const v2i a0_0 = {(int)x0.x, (int)x0.y}, a0_1 = {(int)x0.z, (int)x0.w};
    const v2i a1_0 = {(int)x1.x, (int)x1.y}, a1_1 = {(int)x1.z, (int)x1.w};

    /* Stage 1: forget WMMA -> 8 f_raw scalars (releases df0/df1 VGPRs before gain WMMA). */
    v8i cf0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_0, 1, bf0_0, zm, 0);
    cf0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_1, 1, bf0_1, cf0, 0);
    const f8 df0 = __builtin_bit_cast(f8, cf0) - 12582912.0f;

    v8i cf1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_0, 1, bf1_0, zm, 0);
    cf1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_1, 1, bf1_1, cf1, 0);
    const f8 df1 = __builtin_bit_cast(f8, cf1) - 12582912.0f;

    const f8 scf0 = (f8)(swf0) * dx0;
    const f8 scf1 = (f8)(swf1) * dx1;
    const f8 f_vec = {
        __builtin_fmaf(scf1[0], df1[0], __builtin_fmaf(scf0[0], df0[0], 0.0f)),
        __builtin_fmaf(scf1[1], df1[1], __builtin_fmaf(scf0[1], df0[1], 0.0f)),
        __builtin_fmaf(scf1[2], df1[2], __builtin_fmaf(scf0[2], df0[2], 0.0f)),
        __builtin_fmaf(scf1[3], df1[3], __builtin_fmaf(scf0[3], df0[3], 0.0f)),
        __builtin_fmaf(scf1[4], df1[4], __builtin_fmaf(scf0[4], df0[4], 0.0f)),
        __builtin_fmaf(scf1[5], df1[5], __builtin_fmaf(scf0[5], df0[5], 0.0f)),
        __builtin_fmaf(scf1[6], df1[6], __builtin_fmaf(scf0[6], df0[6], 0.0f)),
        __builtin_fmaf(scf1[7], df1[7], __builtin_fmaf(scf0[7], df0[7], 0.0f)),
    };

    /* Stage 2: gain WMMA -> 8 g_raw scalars, fused with non-FMA loop_gate store. */
    v8i cg0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_0, 1, bg0_0, zm, 0);
    cg0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_1, 1, bg0_1, cg0, 0);
    const f8 dg0 = __builtin_bit_cast(f8, cg0) - 12582912.0f;

    v8i cg1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_0, 1, bg1_0, zm, 0);
    cg1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_1, 1, bg1_1, cg1, 0);
    const f8 dg1 = __builtin_bit_cast(f8, cg1) - 12582912.0f;

    const f8 scg0 = (f8)(swg0) * dx0;
    const f8 scg1 = (f8)(swg1) * dx1;

    const u32 t_base_v = t_j0 + (hh << 3u);
    if (r < dim) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 t = t_base_v + v;
        if (t < T) {
          float g_raw = __builtin_fmaf(scg0[v], dg0[v], 0.0f);
          g_raw = __builtin_fmaf(scg1[v], dg1[v], g_raw);
          const float a_val = *(const G float *)((const G u8 *)inp + (unsigned long)t * (unsigned long)is + (unsigned long)r * 4ul);
          const float c_val = *(const G float *)((const G u8 *)msg + (unsigned long)t * (unsigned long)ms + (unsigned long)r * 4ul);
          *(G float *)((G u8 *)tmp + (unsigned long)t * (unsigned long)ts + (unsigned long)r * 4ul) =
              ink_r6_4c_loop_gate1(a_val, c_val, f_vec[v], g_raw);
        }
      }
    }
  }
}

KERNEL ie_inkling_lc_wfg_gate_norm_r6(
    const G u32 *qw0, const G f16 *qs0, const G float *h0, const G float *inp0, const G float *msg0,
    G float *tmp0, const G float *nw0, G float *ny0, G u8 *nq0,
    const G u32 *qw1, const G f16 *qs1, const G float *h1, const G float *inp1, const G float *msg1,
    G float *tmp1, const G float *nw1, G float *ny1, G u8 *nq1,
    const G u32 *qw2, const G f16 *qs2, const G float *h2, const G float *inp2, const G float *msg2,
    G float *tmp2, const G float *nw2, G float *ny2, G u8 *nq2,
    const G u32 *qw3, const G f16 *qs3, const G float *h3, const G float *inp3, const G float *msg3,
    G float *tmp3, const G float *nw3, G float *ny3, G u8 *nq3,
    G u32 *count, u32 dim, u32 nb, u32 T, u32 n_ctrl, float eps,
    u32 hs, u32 is0, u32 is1, u32 is2, u32 is3,
    u32 ms0, u32 ms1, u32 ms2, u32 ms3,
    u32 ts, u32 ys, u32 qs_out) {
  _Static_assert(sizeof(ink_r6_4c_wfg_lds) == 5632u, "ink_r6_4c wfg/norm LDS union must be 5632 B");
  if (T == 0u || dim == 0u || (dim & 31u) != 0u || nb != 2u || n_ctrl == 0u || n_ctrl > 4u) return;

  const u32 bx = wgid();
  const u32 gy = tokid();
  const u32 is_pf = (T >= 17u) ? 1u : 0u;
  const u32 n_wgx = is_pf ? ((dim + 127u) >> 7u) : ((dim + 31u) >> 5u);
  const u32 n_tt = is_pf ? ((T + 63u) >> 6u) : 1u;
  const u32 c = is_pf ? (gy / n_tt) : gy;
  const u32 tt = is_pf ? (gy - c * n_tt) : 0u;
  if (c >= n_ctrl || bx >= n_wgx) return;

  /* Uniform per-WG selection of controller c's arguments (stays in SGPRs). */
  const G u32 *qw = qw0;
  const G f16 *qs = qs0;
  const G float *h = h0;
  const G float *inp = inp0;
  const G float *msg = msg0;
  G float *tmp = tmp0;
  const G float *nw = nw0;
  G float *ny = ny0;
  G u8 *nq = nq0;
  u32 is = is0;
  u32 ms = ms0;
  if (c == 1u) {
    qw = qw1; qs = qs1; h = h1; inp = inp1; msg = msg1;
    tmp = tmp1; nw = nw1; ny = ny1; nq = nq1; is = is1; ms = ms1;
  } else if (c == 2u) {
    qw = qw2; qs = qs2; h = h2; inp = inp2; msg = msg2;
    tmp = tmp2; nw = nw2; ny = ny2; nq = nq2; is = is2; ms = ms2;
  } else if (c == 3u) {
    qw = qw3; qs = qs3; h = h3; inp = inp3; msg = msg3;
    tmp = tmp3; nw = nw3; ny = ny3; nq = nq3; is = is3; ms = ms3;
  }
  if (!qw || !qs || !h || !inp || !msg || !tmp) return;

  const u32 th = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(th >> 5u));
  const u32 l = lane();

  u32 t_beg = 0u;
  u32 t_end = T;
  G u32 *cnt_ptr = count ? (count + c) : (G u32 *)0;
  if (T == 1u) {
    ink_r6_4c_wfg_gate_dec1(qw, qs, h, inp, msg, tmp, dim, bx, w, l);
  } else if (!is_pf) {
    ink_r6_4c_wfg_gate_ver16(qw, qs, h, inp, msg, tmp, dim, T, hs, is, ms, ts, bx, w, l);
  } else {
    t_beg = tt * INK_R6_4C_NB2_T;
    t_end = (t_beg + INK_R6_4C_NB2_T < T) ? (t_beg + INK_R6_4C_NB2_T) : T;
    cnt_ptr = count ? (count + (unsigned long)(c * n_tt + tt)) : (G u32 *)0;
    ink_r6_4c_wfg_gate_pf(qw, qs, h, inp, msg, tmp, dim, T, hs, is, ms, ts,
                          bx * INK_R6_4C_NB2_R, t_beg, w, l);
  }
  if (nw && cnt_ptr && (ny || nq)) {
    ink_r6_4c_wfg_norm_tail(tmp, nw, ny, nq, cnt_ptr, n_wgx, dim, eps,
                            t_beg, t_end, ts, ys, qs_out, th, w, l);
  }
}

#undef INK_R6_4C_NB2_XLS
#undef INK_R6_4C_NB2_T
#undef INK_R6_4C_NB2_R
#undef INK_R6_4C_DN_CHUNK
#undef INK_R6_4C_PN_NOREP
#undef INK_R6_4C_PN_SHARED
#undef INK_R6_4C_AR_SLOTS
#undef INK_R6_4C_AR_MAXM

#endif /* !IE_MEGA */
