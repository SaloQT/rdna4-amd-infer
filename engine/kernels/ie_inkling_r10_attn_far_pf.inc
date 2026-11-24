/* ============================================================================
 * Inkling-2 Round 10 (Wave 9/10, H04 + H05):
 *   1. Dedicated Zero-rb NB=32 3-WGs/CU Far-Field Global Prefill Attention
 *      (ie_inkling_rel_attn_pf_far_r10): 16,384 B LDS, <= 128 VGPR (3 WGs/CU!),
 *      ping-pong K/VT split-barrier overlap, exact FA4 conditional rescaling.
 *   2. Near-Field Merge Companion (ie_inkling_rel_attn_pf_near_merge_r10):
 *      consumes (O_far, M_far, L_far) from b->at_part (stride 130 floats, 0
 *      extra VRAM!) and runs ONLY the <= 66 near-field blocks (exact rel_bias
 *      + causal mask + split barriers + conditional rescaling) to emit final O.
 *   3. Split-Barrier + Conditional-Rescale SWA/All-Phase Upgrade
 *      (ie_inkling_rel_attn_pf_sb_r10): 100% exact-SHA drop-in upgrade of
 *      ie_inkling_rel_attn_pf_r7 for all 40 SWA layers and Global fallback!
 *
 * Target: RX 9070 XT (gfx1201, wave32, hd=128, d_rel=16, f16 KV, NT=256).
 *
 * Exported Kernels, Grids, Guards, and Parity:
 *
 * 1. KERNEL ie_inkling_rel_attn_pf_far_r10(
 *        const G float *q, const G kvt *kc, const G kvt *vc,
 *        const G float *sink, G float *part,
 *        u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
 *        u32 rel_extent, u32 qs, u32 ss)
 *    - Grid:      ((T + 31u) / 32u, n_kv, 1u), NT = 256.
 *    - Scratch:   part = b->at_part (stride 130 floats per (tk, h):
 *                 pp[0]=m_log2, pp[1]=l_tot, pp[2..129]=o_far[0..127];
 *                 requires b->at_part_floats >= (size_t)T * n_head * 130u;
 *                 at T=512, n_head=24: 1,597,440 <= 3,194,880 floats -> 0 VRAM!).
 *    - Guard:     !b->kvq8 && !b->kvq4 && !mk.rec && win == 0u && pos0 >= 2048u &&
 *                 T > 1u && hd == 128u && n_kv > 0u && n_head == n_kv * 3u &&
 *                 kv_len > 0u && part && (size_t)T * n_head * 130u <= at_part_floats.
 *    - Resources: <= 120 VGPR (true 12w/SIMD = 3 WGs/CU!), 16,384 B LDS, 0/0 spills.
 *
 * 2. KERNEL ie_inkling_rel_attn_pf_near_merge_r10(
 *        const G float *q, const G kvt *kc, const G kvt *vc,
 *        const G float *R, const G float *rel_bias, const G float *sink,
 *        const G float *part, G float *out,
 *        u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
 *        u32 d_rel, u32 rel_extent, u32 win,
 *        u32 qs, u32 rs, u32 ss, u32 os)
 *    - Grid:      ((T + 31u) / 32u, n_kv, 1u), NT = 256 (launched immediately
 *                 after far_r10 on the same stream!).
 *    - Parity:    0-1 ULP / KLD ~ 0 vs pf_r7 (exact 16+16 two-step online
 *                 softmax + 0-ULP conditional rescale skip).
 *
 * 3. KERNEL ie_inkling_rel_attn_pf_sb_r10(
 *        const G float *q, const G kvt *kc, const G kvt *vc,
 *        const G float *R, const G float *rel_bias, const G float *sink, G float *out,
 *        u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
 *        u32 d_rel, u32 rel_extent, u32 win,
 *        u32 qs, u32 rs, u32 ss, u32 os)
 *    - Exact drop-in signature & grid `((T + 31u) / 32u, n_kv, 1u)` of pf_r7!
 *    - Parity:    100% bitwise-exact (0-ULP, exact-SHA) vs ie_inkling_rel_attn_pf_r7!
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(kvt) == 2u, "ie_inkling_r10_attn_far_pf requires f16 KV cache");

#define INK_R10_ATT_NB16  16u
#define INK_R10_ATT_NB32  32u
#define INK_R10_ATT_D     3u
#define INK_R10_ATT_LOG2E 1.44269504088896340736f
#define INK_R10_ATT_QSC   (0.08838834764831844299f * 1.44269504088896340736f)
#define INK_R10_ATT_ROWB  256u
#define INK_R10_ATT_PSTR  130u

typedef unsigned int ink_r10_att_u2 __attribute__((ext_vector_type(2)));
typedef float        ink_r10_att_f2 __attribute__((ext_vector_type(2)));

/* ============================================================================
 * Dedicated Far-Field LDS (16,384 B ONLY -> 3 WGs/CU at <= 128 VGPR!).
 * Referenced ONLY by ie_inkling_rel_attn_pf_far_r10!
 * ========================================================================== */
typedef struct {
  f16 k[INK_R10_ATT_NB32 * 128u];      /*  8,192 B: single-stage NB=32 XOR-swizzled K */
  f16 vt[2][128u * INK_R10_ATT_NB16];  /*  8,192 B: two 128x16 XOR-swizzled V^T halves */
} ink_r10_att_far_lds_t;

_Static_assert(sizeof(ink_r10_att_far_lds_t) == 16384u, "ink_r10_att_far_lds_t must be 16,384 B");

/* ============================================================================
 * Near-Field / SWA LDS (32,128 B).
 * Referenced ONLY by near_merge_r10 and sb_r10 (never by far_r10!).
 * ========================================================================== */
typedef struct {
  f16   k[2][INK_R10_ATT_NB16 * 128u];  /*  8,192 B */
  f16   vt[2][128u * INK_R10_ATT_NB16]; /*  8,192 B */
  f16   rb[2][48u * 16u];               /*  3,072 B */
  float rbsc[6][16][33];                /* 12,672 B */
} ink_r10_att_nf_lds_t;

_Static_assert(sizeof(ink_r10_att_nf_lds_t) <= 32768u, "ink_r10_att_nf_lds_t must fit in 32 KiB");

/* gfx1201 decoupled LDS split-barrier primitives. */
static inline __attribute__((always_inline)) void ink_r10_att_lsignal(void) {
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  __builtin_amdgcn_s_barrier_signal(-1);
  __asm__ volatile("" ::: "memory");
}

static inline __attribute__((always_inline)) void ink_r10_att_lwait(void) {
  __builtin_amdgcn_s_barrier_wait(-1);
  __asm__ volatile("" ::: "memory");
}

static inline __attribute__((always_inline)) u32 ink_r10_att_k_off(const u32 r, const u32 c) {
  return r * 128u + 8u * (c ^ (r & 15u));
}

static inline __attribute__((always_inline)) u32 ink_r10_att_vt_off(const u32 col, const u32 h) {
  return 8u * ((2u * col + h) ^ ((col >> 2u) & 1u) ^ ((col >> 3u) & 7u));
}

static inline __attribute__((always_inline)) float ink_r10_att_xrow(const float v) {
  return I2F(xrow_i(F2I(v)));
}

/* Shared exact NB=32 far-field block count (guaranteed no-wrap & zero rel_bias). */
static inline __attribute__((always_inline)) u32 ink_r10_att_far_nblk32(
    const u32 qpos_min, const u32 rel_extent, const u32 kv_len) {
  if (qpos_min + 1u <= rel_extent) return 0u;
  const u32 lim = qpos_min + 1u - rel_extent;
  const u32 lim_nw = (lim <= kv_len) ? lim : kv_len;
  return lim_nw >> 5u;
}

static inline __attribute__((always_inline)) void ink_r10_att_transpose8(const u32 *k, u32x4 *ev, u32x4 *od) {
  u32 e[4], o[4];
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    e[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x05040100u);
    o[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x07060302u);
  }
  *ev = (u32x4){e[0], e[1], e[2], e[3]};
  *od = (u32x4){o[0], o[1], o[2], o[3]};
}

/* ============================================================================
 * Part A Helpers: Dedicated Zero-rb NB=32 Far-Field Attention (<= 128 VGPR!)
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r10_att_far_load_k32(
    u32x4 kx[8], const G u8 *kb_kh, const unsigned long row_step, const u32 kb, const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
  const G u8 *k_w = kb_kh + (unsigned long)(kb + 16u * wi) * row_step;
  const unsigned long k_off = (unsigned long)sub_r * row_step + 16ul * c;
#pragma unroll
  for (u32 i = 0u; i < 8u; i++) {
    kx[i] = *(const G u32x4 *)(k_w + (unsigned long)(2u * i) * row_step + k_off);
  }
}

static inline __attribute__((always_inline)) void ink_r10_att_far_pub_k32(
    LDSP f16 *k_lds, const u32x4 kx[8], const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
#pragma unroll
  for (u32 i = 0u; i < 8u; i++) {
    const u32 r = 16u * wi + 2u * i + sub_r;
    *(LDSP u32x4 *)(k_lds + ink_r10_att_k_off(r, c)) = kx[i];
  }
}

static inline __attribute__((always_inline)) void ink_r10_att_far_load_v32(
    ink_r10_att_u2 vx0[8], ink_r10_att_u2 vx1[8], const G u8 *vb_kh,
    const unsigned long row_step, const u32 kb, const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const unsigned long v_off = 8ul * L;
  const G u8 *v_w0 = vb_kh + (unsigned long)(kb + 8u * wi) * row_step;
  const G u8 *v_w1 = vb_kh + (unsigned long)(kb + 16u + 8u * wi) * row_step;
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    vx0[u] = *(const G ink_r10_att_u2 *)(v_w0 + (unsigned long)u * row_step + v_off);
    vx1[u] = *(const G ink_r10_att_u2 *)(v_w1 + (unsigned long)u * row_step + v_off);
  }
}

static inline __attribute__((always_inline)) void ink_r10_att_far_pub_vt16(
    LDSP f16 *vt_lds, const ink_r10_att_u2 vx[8], const u32 wi, const u32 L) {
  u32 vx0[8], vx1[8];
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    vx0[u] = vx[u].x;
    vx1[u] = vx[u].y;
  }
  u32x4 ev0, od0, ev1, od1;
  ink_r10_att_transpose8(vx0, &ev0, &od0);
  ink_r10_att_transpose8(vx1, &ev1, &od1);
  const u32 c0 = 4u * L;
  *(LDSP u32x4 *)(vt_lds + ink_r10_att_vt_off(c0 + 0u, wi)) = ev0;
  *(LDSP u32x4 *)(vt_lds + ink_r10_att_vt_off(c0 + 1u, wi)) = od0;
  *(LDSP u32x4 *)(vt_lds + ink_r10_att_vt_off(c0 + 2u, wi)) = ev1;
  *(LDSP u32x4 *)(vt_lds + ink_r10_att_vt_off(c0 + 3u, wi)) = od1;
}

/* Uniform full-exec lane ID (0..31) via volatile v_mbcnt_lo_u32_b32:
 * Volatile prevents LLVM LICM from hoisting kb_b / vt_lane out of the K-loop
 * (saving 9 loop-carried VGPRs -> <= 120 VGPR with 0/0 spills!).
 * ONLY call where all 32 lanes are guaranteed active (exec_lo == 0xffffffff). */
static inline __attribute__((always_inline)) u32 ink_r10_att_far_lane(void) {
  u32 l;
  __asm__ volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(l));
  return l;
}

/* Low-VGPR 1-chain 2-deep pipelined QK^T for first 16 keys (s0, peak <= 114 VGPR). */
static inline __attribute__((always_inline)) v8f ink_r10_att_far_qk16(
    const LDSP f16 *k_lds, const h8 qa[8]) {
  const u32 l_loc = ink_r10_att_far_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 kb_b = (rl << 8u) | ((rl ^ hl) << 4u);
  const LDSP u8 *kp = (const LDSP u8 *)k_lds;

  v8f sa = (v8f)(0.0f);
  h8 k0 = *(const LDSP h8 *)(kp + (kb_b ^ 0u));
  h8 k1 = *(const LDSP h8 *)(kp + (kb_b ^ 32u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[0], sa);
  k0 = *(const LDSP h8 *)(kp + (kb_b ^ 64u));
  __builtin_amdgcn_sched_barrier(0);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[1], sa);
  k1 = *(const LDSP h8 *)(kp + (kb_b ^ 96u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[2], sa);
  k0 = *(const LDSP h8 *)(kp + (kb_b ^ 128u));
  __builtin_amdgcn_sched_barrier(0);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[3], sa);
  k1 = *(const LDSP h8 *)(kp + (kb_b ^ 160u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[4], sa);
  k0 = *(const LDSP h8 *)(kp + (kb_b ^ 192u));
  __builtin_amdgcn_sched_barrier(0);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[5], sa);
  k1 = *(const LDSP h8 *)(kp + (kb_b ^ 224u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[6], sa);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[7], sa);
  __builtin_amdgcn_sched_barrier(0);
  return sa;
}

/* Ultra-low-VGPR 1-deep in-place-XOR QK^T for second 16 keys (s1 while s0 live):
 * Uses 4 fewer VGPRs (1x k0 + single mutating kb_b) so total peak stays <= 120 VGPR
 * (true gfx1201 hardware 12 waves/SIMD = 3 WGs/CU with 0/0 spills!). */
static inline __attribute__((always_inline)) v8f ink_r10_att_far_qk16_s1(
    const LDSP f16 *k_lds, const h8 qa[8]) {
  const u32 l_loc = ink_r10_att_far_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  u32 kb_b = (rl << 8u) | ((rl ^ hl) << 4u);
  const LDSP u8 *kp = (const LDSP u8 *)k_lds;

  v8f sa = (v8f)(0.0f);
  h8 k0;

  k0 = *(const LDSP h8 *)(kp + kb_b);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[0], sa);
  kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);

  k0 = *(const LDSP h8 *)(kp + kb_b);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[1], sa);
  kb_b ^= 96u;
  __builtin_amdgcn_sched_barrier(0);

  k0 = *(const LDSP h8 *)(kp + kb_b);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[2], sa);
  kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);

  k0 = *(const LDSP h8 *)(kp + kb_b);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[3], sa);
  kb_b ^= 224u;
  __builtin_amdgcn_sched_barrier(0);

  k0 = *(const LDSP h8 *)(kp + kb_b);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[4], sa);
  kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);

  k0 = *(const LDSP h8 *)(kp + kb_b);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[5], sa);
  kb_b ^= 96u;
  __builtin_amdgcn_sched_barrier(0);

  k0 = *(const LDSP h8 *)(kp + kb_b);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[6], sa);
  kb_b ^= 32u;
  __builtin_amdgcn_sched_barrier(0);

  k0 = *(const LDSP h8 *)(kp + kb_b);
  /* All K LDS loads issued: flush dscnt & signal barrier BEFORE final QK WMMA! */
  ink_r10_att_lsignal();
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[7], sa);
  __builtin_amdgcn_sched_barrier(0);
  return sa;
}

/* Low-VGPR 2-deep pipelined PV for 16 keys (single-VGPR vtb + mbcnt_lo -> peak <= 117 VGPR!). */
static inline __attribute__((always_inline)) void ink_r10_att_far_pv16(
    const LDSP f16 *vt_lds, const h8 pb, v8f o[8], const int do_signal) {
  const u32 l_loc = ink_r10_att_far_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2u) & 1u) ^ (rl >> 3u);
  const u32 vtb = 16u * vt_lane;
  const LDSP u8 *vp = (const LDSP u8 *)vt_lds;

  h8 v0 = *(const LDSP h8 *)(vp + 0u + vtb);
  h8 v1 = *(const LDSP h8 *)(vp + 512u + (vtb ^ 32u));
  __builtin_amdgcn_sched_barrier(0);

  o[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[0]);
  v0 = *(const LDSP h8 *)(vp + 1024u + (vtb ^ 64u));
  __builtin_amdgcn_sched_barrier(0);
  o[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[1]);
  v1 = *(const LDSP h8 *)(vp + 1536u + (vtb ^ 96u));
  __builtin_amdgcn_sched_barrier(0);

  o[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[2]);
  v0 = *(const LDSP h8 *)(vp + 2048u + vtb);
  __builtin_amdgcn_sched_barrier(0);
  o[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[3]);
  v1 = *(const LDSP h8 *)(vp + 2560u + (vtb ^ 32u));
  __builtin_amdgcn_sched_barrier(0);

  o[4] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[4]);
  v0 = *(const LDSP h8 *)(vp + 3072u + (vtb ^ 64u));
  __builtin_amdgcn_sched_barrier(0);
  o[5] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[5]);
  v1 = *(const LDSP h8 *)(vp + 3584u + (vtb ^ 96u));
  if (do_signal) {
    /* Last VT LDS loads issued: flush dscnt & signal BEFORE final 2 PV WMMAs! */
    ink_r10_att_lsignal();
  } else {
    __builtin_amdgcn_sched_barrier(0);
  }

  o[6] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[6]);
  o[7] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[7]);
  __builtin_amdgcn_sched_barrier(0);
}

/* Exact two-step NB=32 (16+16) QK + conditional rescale + split-barrier PV step:
 * Frees s0 (8 VGPRs) into pb0 (4 VGPRs) BEFORE QK_1, and frees s1 (8 VGPRs) into
 * pb1 (4 VGPRs) BEFORE lwait -> peak <= 117 VGPR (true 12 waves/SIMD = 3 WGs/CU
 * with 0/0 spills!) while matching NB=16 online softmax math (pf_r7) exactly! */
static inline __attribute__((always_inline)) void ink_r10_att_far_blk32_step(
    const LDSP f16 *k_lds, const LDSP f16 *vt0, const LDSP f16 *vt1,
    const h8 qa[8], const int has_next, float *m_p, float *ls_p, v8f o[8]) {
  float m = *m_p;
  float ls = *ls_p;

  /* Half 0: QK_0 (2-deep k0/k1) + exact softmax0 -> frees s0 before QK_1! */
  const v8f s0 = ink_r10_att_far_qk16(k_lds, qa);
  h8 pb0 = (h8)(0.0f);
  int active0 = 0;
  {
    float bm0 = s0[0];
#pragma unroll
    for (u32 v = 1u; v < 8u; v++) bm0 = __builtin_fmaxf(bm0, s0[v]);
    bm0 = __builtin_fmaxf(bm0, ink_r10_att_xrow(bm0));
    active0 = (__builtin_amdgcn_ballot_w32(bm0 > -1e30f && !(bm0 - m < -30.0f)) != 0u);
    if (active0) {
      const int nr0 = (__builtin_amdgcn_ballot_w32(bm0 > m || m <= -1e30f) != 0u);
      const float mn0 = nr0 ? __builtin_fmaxf(m, bm0) : m;
      float ps0 = 0.0f;
      if (mn0 > -1e30f) {
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) {
          const float p0 = __builtin_amdgcn_exp2f(s0[v] - mn0);
          ps0 += p0;
          pb0[v] = (f16)p0;
        }
      }
      float c0 = 1.0f;
      if (nr0) {
        c0 = (m > -1e30f) ? __builtin_amdgcn_exp2f(m - mn0) : 0.0f;
#pragma unroll
        for (u32 nt = 0u; nt < 8u; nt++)
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c0;
      }
      ls = ls * c0 + ps0;
      m = mn0;
    }
  }
  __builtin_amdgcn_sched_barrier(0);

  /* Half 1: QK_1 (1-deep k0, signals K consumed) + exact softmax1 -> frees s1 before lwait! */
  const v8f s1 = ink_r10_att_far_qk16_s1(k_lds + 2048u, qa);
  h8 pb1 = (h8)(0.0f);
  float c1 = 1.0f;
  int active1 = 0, nr1 = 0;
  {
    float bm1 = s1[0];
#pragma unroll
    for (u32 v = 1u; v < 8u; v++) bm1 = __builtin_fmaxf(bm1, s1[v]);
    bm1 = __builtin_fmaxf(bm1, ink_r10_att_xrow(bm1));
    active1 = (__builtin_amdgcn_ballot_w32(bm1 > -1e30f && !(bm1 - m < -30.0f)) != 0u);
    if (active1) {
      nr1 = (__builtin_amdgcn_ballot_w32(bm1 > m || m <= -1e30f) != 0u);
      const float mn1 = nr1 ? __builtin_fmaxf(m, bm1) : m;
      float ps1 = 0.0f;
      if (mn1 > -1e30f) {
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) {
          const float p1 = __builtin_amdgcn_exp2f(s1[v] - mn1);
          ps1 += p1;
          pb1[v] = (f16)p1;
        }
      }
      if (nr1) c1 = (m > -1e30f) ? __builtin_amdgcn_exp2f(m - mn1) : 0.0f;
      ls = ls * c1 + ps1;
      m = mn1;
    }
  }
  *m_p = m;
  *ls_p = ls;

  /* Wait for VT[0..1] ready (both QKs + both softmaxes overlapped behind this barrier!). */
  ink_r10_att_lwait();

  if (active0) ink_r10_att_far_pv16(vt0, pb0, o, 0);
  if (active1) {
    if (nr1) {
#pragma unroll
      for (u32 nt = 0u; nt < 8u; nt++)
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c1;
    }
    ink_r10_att_far_pv16(vt1, pb1, o, has_next);
  }
  if (has_next) {
    if (!active1) ink_r10_att_lsignal();
    ink_r10_att_lwait();
  }
}

/* ============================================================================
 * Kernel 1: ie_inkling_rel_attn_pf_far_r10 (Zero-rb, NB=32, 16 KiB, 3 WGs/CU!)
 * ========================================================================== */
__attribute__((amdgpu_waves_per_eu(12, 16)))
KERNEL ie_inkling_rel_attn_pf_far_r10(
    const G float *q, const G kvt *kc, const G kvt *vc,
    const G float *sink, G float *part,
    u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
    u32 rel_extent, u32 qs, u32 ss) {
  static LDS ink_r10_att_far_lds_t ink_r10_att_far_lds __attribute__((aligned(16)));
  if (hd != 128u || n_kv == 0u || n_head != n_kv * 3u || T == 0u || kv_len == 0u || !part) return;

  const u32 ntile = (T + 31u) >> 5u;
  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  if (wgid() >= ntile || kh >= n_kv) return;

  const u32 tile = (ntile == 16u && n_kv == 8u && kh < 4u) ? wgid() : ((ntile - 1u) - wgid());
  const u32 t0 = tile << 5u;
  const u32 qpos_min = pos0 + t0;
  const u32 nblk32 = (u32)__builtin_amdgcn_readfirstlane((int)ink_r10_att_far_nblk32(qpos_min, rel_extent, kv_len));
  if (nblk32 == 0u) return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));

  if (w >= 6u) {
    /* 2 Producer waves: 1 full NB=32 block latency hiding with ping-pong K/VT split barriers! */
    const u32 pt = tid() - 192u;
    const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
    const u32 L = pt & 31u;
    const unsigned long row_step = (unsigned long)n_kv * INK_R10_ATT_ROWB;
    const G u8 *kb_kh = (const G u8 *)kc + (unsigned long)kh * INK_R10_ATT_ROWB;
    const G u8 *vb_kh = (const G u8 *)vc + (unsigned long)kh * INK_R10_ATT_ROWB;

    u32x4 kx[8];
    ink_r10_att_u2 vx0[8], vx1[8];

    ink_r10_att_far_load_k32(kx, kb_kh, row_step, 0u, pt);
    __builtin_amdgcn_sched_barrier(0);
    ink_r10_att_far_pub_k32(ink_r10_att_far_lds.k, kx, pt);
    ink_r10_att_lsignal();
    ink_r10_att_far_load_v32(vx0, vx1, vb_kh, row_step, 0u, pt);
    if (1u < nblk32) ink_r10_att_far_load_k32(kx, kb_kh, row_step, 32u, pt);
    ink_r10_att_lwait();

#pragma clang loop unroll(disable)
    for (u32 b = 0u; b < nblk32; b++) {
      const u32 b_u = (u32)__builtin_amdgcn_readfirstlane((int)b);
      ink_r10_att_far_pub_vt16(ink_r10_att_far_lds.vt[0], vx0, wi, L);
      ink_r10_att_far_pub_vt16(ink_r10_att_far_lds.vt[1], vx1, wi, L);
      ink_r10_att_lsignal();
      if (b_u + 1u < nblk32) {
        ink_r10_att_far_load_v32(vx0, vx1, vb_kh, row_step, (b_u + 1u) << 5u, pt);
      }
      ink_r10_att_lwait();

      if (b_u + 1u < nblk32) {
        ink_r10_att_far_pub_k32(ink_r10_att_far_lds.k, kx, pt);
        ink_r10_att_lsignal();
        if (b_u + 2u < nblk32) {
          ink_r10_att_far_load_k32(kx, kb_kh, row_step, (b_u + 2u) << 5u, pt);
        }
        ink_r10_att_lwait();
      }
    }
    return;
  }

  /* 6 Consumer waves (2 Q-subtiles x 3 GQA heads). */
  const u32 tsub = w >= 3u ? 1u : 0u;
  const u32 hsub = w - 3u * tsub;
  const u32 t_sub0 = t0 + 16u * tsub;
  if (t_sub0 >= T) {
    const u32 nbar = 2u * nblk32;
    for (u32 i = 0u; i < nbar; i++) {
      ink_r10_att_lsignal();
      ink_r10_att_lwait();
    }
    return;
  }

  /* Signal Sync 0 BEFORE Q/sink loads (hides initial barrier latency behind Q fetch!). */
  ink_r10_att_lsignal();

  const u32 h = (u32)__builtin_amdgcn_readfirstlane((int)(kh * 3u + hsub));
  h8 qa[8];
  float m, ls;
  {
    const u32 l = lane(), hl = l >> 4u, rl = l & 15u;
    const u32 tk = t_sub0 + rl;
    const u32 tq = tk < T ? tk : (T - 1u);
    const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (unsigned long)tq * qs) +
                                           (unsigned long)h * 128ul);
#pragma unroll
    for (u32 ks = 0u; ks < 8u; ks++) {
      const f32x4 v0 = qp4[4u * ks + 2u * hl + 0u];
      const f32x4 v1 = qp4[4u * ks + 2u * hl + 1u];
      qa[ks] = (h8){(f16)(v0.x * INK_R10_ATT_QSC), (f16)(v0.y * INK_R10_ATT_QSC),
                    (f16)(v0.z * INK_R10_ATT_QSC), (f16)(v0.w * INK_R10_ATT_QSC),
                    (f16)(v1.x * INK_R10_ATT_QSC), (f16)(v1.y * INK_R10_ATT_QSC),
                    (f16)(v1.z * INK_R10_ATT_QSC), (f16)(v1.w * INK_R10_ATT_QSC)};
      __builtin_amdgcn_sched_barrier(0);
    }
    const float sink_val = ((const G float *)((const G u8 *)sink + (unsigned long)tq * ss))[h];
    m = sink_val * INK_R10_ATT_LOG2E;
    ls = (hl == 0u) ? 1.0f : 0.0f;
  }
  __builtin_amdgcn_sched_barrier(0);

  v8f o[8];
#pragma unroll
  for (u32 nt = 0u; nt < 8u; nt++) o[nt] = (v8f)(0.0f);
  __builtin_amdgcn_sched_barrier(0);

  ink_r10_att_lwait();

#pragma clang loop unroll(disable)
  for (u32 b = 0u; b < nblk32; b++) {
    const u32 b_u = (u32)__builtin_amdgcn_readfirstlane((int)b);
    const int has_next = (b_u + 1u < nblk32);
    ink_r10_att_far_blk32_step(ink_r10_att_far_lds.k, ink_r10_att_far_lds.vt[0],
                               ink_r10_att_far_lds.vt[1], qa, has_next, &m, &ls, o);
  }

  /* Emit (M_far [log2], L_far, O_far[0..127]) into part (stride 130 floats, 8-B aligned f32x2). */
  const float l_tot = ls + ink_r10_att_xrow(ls);
  const u32 l_ep = ink_r10_att_far_lane();
  const u32 rl_ep = l_ep & 15u, hl_ep = l_ep >> 4u;
  const u32 tk_ep = t_sub0 + rl_ep;
  if (tk_ep < T) {
    G float *pp = part + (unsigned long)(tk_ep * n_head + h) * INK_R10_ATT_PSTR;
    if (hl_ep == 0u) {
      pp[0] = m;
      pp[1] = l_tot;
    }
    G ink_r10_att_f2 *po2 = (G ink_r10_att_f2 *)(pp + 2u);
#pragma unroll
    for (u32 nt = 0u; nt < 8u; nt++) {
      const u32 base2 = 8u * nt + 4u * hl_ep;
      po2[base2 + 0u] = (ink_r10_att_f2){o[nt][0], o[nt][1]};
      po2[base2 + 1u] = (ink_r10_att_f2){o[nt][2], o[nt][3]};
      po2[base2 + 2u] = (ink_r10_att_f2){o[nt][4], o[nt][5]};
      po2[base2 + 3u] = (ink_r10_att_f2){o[nt][6], o[nt][7]};
    }
  }
}

/* ============================================================================
 * Part B & C Helpers: Split-Barrier + Exact Conditional-Rescale NB=16 Engine
 * Used by ie_inkling_rel_attn_pf_near_merge_r10 & ie_inkling_rel_attn_pf_sb_r10.
 * ========================================================================== */

static LDS ink_r10_att_nf_lds_t ink_r10_att_nf_lds __attribute__((aligned(16)));

typedef struct {
  u32x4           kx[4];
  ink_r10_att_u2  vx[8];
} ink_r10_att_slot16_t;

static inline __attribute__((always_inline)) u32 ink_r10_att_wrap_s(u32 v, const u32 kv_len) {
  if (kv_len >= 16u) return (v >= kv_len) ? (v - kv_len) : v;
  while (v >= kv_len) v -= kv_len;
  return v;
}

static inline __attribute__((always_inline)) u32 ink_r10_att_kv_adv(const u32 kv_b, const u32 kv_len) {
  return ink_r10_att_wrap_s(kv_b + INK_R10_ATT_NB16, kv_len);
}

static inline __attribute__((always_inline)) void ink_r10_att_load16_nowrap(
    ink_r10_att_slot16_t *s, const G u8 *kb_kh, const G u8 *vb_kh,
    const unsigned long row_step, const u32 kb, const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
  const G u8 *k_w = kb_kh + (unsigned long)(kb + 8u * wi) * row_step;
  const unsigned long k_off = (unsigned long)sub_r * row_step + 16ul * c;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    s->kx[i] = *(const G u32x4 *)(k_w + (unsigned long)(2u * i) * row_step + k_off);
  }
  const G u8 *v_w = vb_kh + (unsigned long)(kb + 8u * wi) * row_step;
  const unsigned long v_off = 8ul * L;
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    s->vx[u] = *(const G ink_r10_att_u2 *)(v_w + (unsigned long)u * row_step + v_off);
  }
}

static inline __attribute__((always_inline)) void ink_r10_att_load16_ring(
    ink_r10_att_slot16_t *s, const G u8 *kb_kh, const G u8 *vb_kh,
    const unsigned long row_step, const u32 kv_len, const u32 kb, const u32 kv_b,
    const u32 kend, const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
  const u32 r_base = 8u * wi;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r0 = r_base + 2u * i;
    const u32 p0 = kb + r0;
    const u32 p1 = p0 + 1u;
    const u32 kvp0 = ink_r10_att_wrap_s(kv_b + r0, kv_len);
    const u32 kvp1 = ink_r10_att_wrap_s(kv_b + r0 + 1u, kv_len);
    const u32 p = sub_r ? p1 : p0;
    const u32 kvp = sub_r ? kvp1 : kvp0;
    s->kx[i] = (u32x4)(0u);
    if (p < kend) {
      s->kx[i] = *(const G u32x4 *)(kb_kh + (unsigned long)kvp * row_step + 16ul * c);
    }
  }
  const unsigned long v_off = 8ul * L;
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const u32 r = r_base + u;
    const u32 p = kb + r;
    s->vx[u] = (ink_r10_att_u2)(0u);
    if (p < kend) {
      const u32 kvp = ink_r10_att_wrap_s(kv_b + r, kv_len);
      const G u8 *v_row = vb_kh + (unsigned long)kvp * row_step;
      s->vx[u] = *(const G ink_r10_att_u2 *)(v_row + v_off);
    }
  }
}

static inline __attribute__((always_inline)) void ink_r10_att_pub16_kv(
    const ink_r10_att_slot16_t *s, const u32 bf, const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r = 8u * wi + 2u * i + sub_r;
    *(LDSP u32x4 *)(ink_r10_att_nf_lds.k[bf] + ink_r10_att_k_off(r, c)) = s->kx[i];
  }
  u32 vx0[8], vx1[8];
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    vx0[u] = s->vx[u].x;
    vx1[u] = s->vx[u].y;
  }
  u32x4 ev0, od0, ev1, od1;
  ink_r10_att_transpose8(vx0, &ev0, &od0);
  ink_r10_att_transpose8(vx1, &ev1, &od1);
  const u32 c0 = 4u * L;
  *(LDSP u32x4 *)(ink_r10_att_nf_lds.vt[bf] + ink_r10_att_vt_off(c0 + 0u, wi)) = ev0;
  *(LDSP u32x4 *)(ink_r10_att_nf_lds.vt[bf] + ink_r10_att_vt_off(c0 + 1u, wi)) = od0;
  *(LDSP u32x4 *)(ink_r10_att_nf_lds.vt[bf] + ink_r10_att_vt_off(c0 + 2u, wi)) = ev1;
  *(LDSP u32x4 *)(ink_r10_att_nf_lds.vt[bf] + ink_r10_att_vt_off(c0 + 3u, wi)) = od1;
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r10_att_pub16_rb(
    const G float *rel_bias, const u32 qpos_min, const u32 extent_lim,
    const u32 kb, const u32 bf, const u32 pt) {
  const int d_min_s = (int)qpos_min - (int)(kb + (INK_R10_ATT_NB16 - 1u));
  const u32 d_base_wg = d_min_s > 0 ? (u32)d_min_s : 0u;
  if (d_base_wg < extent_lim && pt < 48u) {
    const u32 d = d_base_wg + pt;
    h8 r0 = (h8)(0.0f), r1 = (h8)(0.0f);
    if (d < extent_lim) {
      const G f32x4 *rp = (const G f32x4 *)(rel_bias + (unsigned long)d * 16ul);
      const f32x4 v0 = rp[0], v1 = rp[1], v2 = rp[2], v3 = rp[3];
      r0 = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
      r1 = (h8){(f16)v2.x, (f16)v2.y, (f16)v2.z, (f16)v2.w, (f16)v3.x, (f16)v3.y, (f16)v3.z, (f16)v3.w};
    }
    *(LDSP h8 *)(ink_r10_att_nf_lds.rb[bf] + pt * 16u + 0u) = r0;
    *(LDSP h8 *)(ink_r10_att_nf_lds.rb[bf] + pt * 16u + 8u) = r1;
  }
}

/* Split-barrier D=3 Producer for NB=16:
 * Emits EXACTLY `nblk` split barriers (Sync 0 after publishing bf=0, then 1 per k+1 < nblk),
 * hiding the next VMEM prefetch behind EVERY split barrier! */
static inline __attribute__((always_inline)) void ink_r10_att_producer16_sb(
    const G u8 *kb_kh, const G u8 *vb_kh, const G float *rel_bias,
    const unsigned long row_step, const u32 kv_len, const u32 k_start,
    const u32 kend, const u32 qpos_min, const u32 extent_lim,
    const u32 nblk, const u32 blk_far_end, const int no_wrap, const u32 pt) {
  if (nblk == 0u) return;

  const u32 nblk_full = (kend > k_start) ? ((kend - k_start) >> 4u) : 0u;
  const u32 fast_lim = no_wrap ? (nblk_full < nblk ? nblk_full : nblk) : 0u;

  u32 kv_load = no_wrap ? 0u : (k_start % kv_len);
  u32 k0 = 0u;
  ink_r10_att_slot16_t ring[INK_R10_ATT_D];

  if (no_wrap && fast_lim >= 6u && blk_far_end >= 6u) {
    const u32 far_loc = blk_far_end < fast_lim ? blk_far_end : fast_lim;
    ink_r10_att_load16_nowrap(&ring[0], kb_kh, vb_kh, row_step, k_start + 0u, pt);
    ink_r10_att_load16_nowrap(&ring[1], kb_kh, vb_kh, row_step, k_start + 16u, pt);
    ink_r10_att_load16_nowrap(&ring[2], kb_kh, vb_kh, row_step, k_start + 32u, pt);
    __builtin_amdgcn_sched_barrier(0);
    ink_r10_att_pub16_kv(&ring[0], 0u, pt);
    ink_r10_att_lsignal();
    ink_r10_att_load16_nowrap(&ring[0], kb_kh, vb_kh, row_step, k_start + 48u, pt);
    ink_r10_att_lwait();

#pragma clang loop unroll(disable)
    for (; k0 + 6u < far_loc; k0 += 3u) {
      ink_r10_att_pub16_kv(&ring[1], (k0 + 1u) & 1u, pt);
      ink_r10_att_lsignal();
      ink_r10_att_load16_nowrap(&ring[1], kb_kh, vb_kh, row_step, k_start + 16u * (k0 + 4u), pt);
      ink_r10_att_lwait();

      ink_r10_att_pub16_kv(&ring[2], (k0 + 2u) & 1u, pt);
      ink_r10_att_lsignal();
      ink_r10_att_load16_nowrap(&ring[2], kb_kh, vb_kh, row_step, k_start + 16u * (k0 + 5u), pt);
      ink_r10_att_lwait();

      ink_r10_att_pub16_kv(&ring[0], (k0 + 3u) & 1u, pt);
      ink_r10_att_lsignal();
      ink_r10_att_load16_nowrap(&ring[0], kb_kh, vb_kh, row_step, k_start + 16u * (k0 + 6u), pt);
      ink_r10_att_lwait();
    }
  } else {
#pragma unroll
    for (u32 s = 0u; s < INK_R10_ATT_D; s++) {
      if (s < nblk) {
        const u32 kbs = k_start + 16u * s;
        if (s < fast_lim) {
          ink_r10_att_load16_nowrap(&ring[s], kb_kh, vb_kh, row_step, kbs, pt);
        } else {
          ink_r10_att_load16_ring(&ring[s], kb_kh, vb_kh, row_step, kv_len, kbs, no_wrap ? kbs : kv_load, kend, pt);
        }
        if (!no_wrap) kv_load = ink_r10_att_kv_adv(kv_load, kv_len);
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r10_att_pub16_kv(&ring[0], 0u, pt);
    if (0u >= blk_far_end) {
      ink_r10_att_pub16_rb(rel_bias, qpos_min, extent_lim, k_start, 0u, pt);
    }
    ink_r10_att_lsignal();
    if (INK_R10_ATT_D < nblk) {
      const u32 kbd = k_start + 16u * INK_R10_ATT_D;
      if (INK_R10_ATT_D < fast_lim) {
        ink_r10_att_load16_nowrap(&ring[0], kb_kh, vb_kh, row_step, kbd, pt);
      } else {
        ink_r10_att_load16_ring(&ring[0], kb_kh, vb_kh, row_step, kv_len, kbd, no_wrap ? kbd : kv_load, kend, pt);
      }
      if (!no_wrap) kv_load = ink_r10_att_kv_adv(kv_load, kv_len);
    }
    ink_r10_att_lwait();
  }

#pragma clang loop unroll(disable)
  for (; k0 < nblk; k0 += INK_R10_ATT_D) {
#pragma unroll
    for (u32 j = 0u; j < INK_R10_ATT_D; j++) {
      const u32 k = k0 + j;
      if (k + 1u < nblk) {
        const u32 nxt = (j + 1u) % INK_R10_ATT_D;
        const u32 kn = k + 1u;
        ink_r10_att_pub16_kv(&ring[nxt], kn & 1u, pt);
        if (kn >= blk_far_end) {
          ink_r10_att_pub16_rb(rel_bias, qpos_min, extent_lim, k_start + 16u * kn, kn & 1u, pt);
        }
        ink_r10_att_lsignal();
        const u32 kl = k + 1u + INK_R10_ATT_D;
        if (kl < nblk) {
          const u32 kbl = k_start + 16u * kl;
          if (kl < fast_lim) {
            ink_r10_att_load16_nowrap(&ring[nxt], kb_kh, vb_kh, row_step, kbl, pt);
          } else {
            ink_r10_att_load16_ring(&ring[nxt], kb_kh, vb_kh, row_step, kv_len, kbl, no_wrap ? kbl : kv_load, kend, pt);
          }
          if (!no_wrap) kv_load = ink_r10_att_kv_adv(kv_load, kv_len);
        }
        ink_r10_att_lwait();
      }
    }
  }
}

/* 2-chain 2-deep QK^T (single kb_b addr VGPR, exact bitwise match to pf_r7!). */
static inline __attribute__((always_inline)) void ink_r10_att_qk16_pipe(
    const u32 bf, const h8 qa[8], float s_out[8]) {
  u32 l_loc = lane();
  __asm__ volatile("" : "+v"(l_loc));
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 kb_b = (rl << 8u) | ((rl ^ hl) << 4u);
  const LDSP u8 *kp = (const LDSP u8 *)ink_r10_att_nf_lds.k[bf];
  const v8f zero = (v8f)(0.0f);
  v8f sa = zero, sb = zero;

  h8 k0 = *(const LDSP h8 *)(kp + (kb_b ^ 0u));
  h8 k1 = *(const LDSP h8 *)(kp + (kb_b ^ 32u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[0], sa);
  k0 = *(const LDSP h8 *)(kp + (kb_b ^ 64u));
  sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[1], sb);
  k1 = *(const LDSP h8 *)(kp + (kb_b ^ 96u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[2], sa);
  k0 = *(const LDSP h8 *)(kp + (kb_b ^ 128u));
  sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[3], sb);
  k1 = *(const LDSP h8 *)(kp + (kb_b ^ 160u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[4], sa);
  k0 = *(const LDSP h8 *)(kp + (kb_b ^ 192u));
  sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[5], sb);
  k1 = *(const LDSP h8 *)(kp + (kb_b ^ 224u));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[6], sa);
  sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[7], sb);

#pragma unroll
  for (u32 v = 0u; v < 8u; v++) s_out[v] = sa[v] + sb[v];
}

/* Pipelined PV with gfx1201 split-barrier signal BEFORE final 2 PV WMMAs (0 hoisted addrs!). */
static inline __attribute__((always_inline)) void ink_r10_att_pv16_pipe_sb(
    const u32 bf, const h8 pb, v8f o[8], const int has_next) {
  u32 l_loc = lane();
  __asm__ volatile("" : "+v"(l_loc));
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2u) & 1u) ^ (rl >> 3u);

  const LDSP f16 *vp0 = ink_r10_att_nf_lds.vt[bf] + 8u * vt_lane;
  const LDSP f16 *vp1 = ink_r10_att_nf_lds.vt[bf] + 256u + 8u * (vt_lane ^ 2u);
  const LDSP f16 *vp2 = ink_r10_att_nf_lds.vt[bf] + 512u + 8u * (vt_lane ^ 4u);
  const LDSP f16 *vp3 = ink_r10_att_nf_lds.vt[bf] + 768u + 8u * (vt_lane ^ 6u);

  h8 v0 = *(const LDSP h8 *)(vp0);
  h8 v1 = *(const LDSP h8 *)(vp1);
  __builtin_amdgcn_sched_barrier(0);

  o[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[0]);
  v0 = *(const LDSP h8 *)(vp2);
  __builtin_amdgcn_sched_barrier(0);
  o[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[1]);
  v1 = *(const LDSP h8 *)(vp3);
  __builtin_amdgcn_sched_barrier(0);

  o[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[2]);
  v0 = *(const LDSP h8 *)(vp0 + 1024u);
  __builtin_amdgcn_sched_barrier(0);
  o[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[3]);
  v1 = *(const LDSP h8 *)(vp1 + 1024u);
  __builtin_amdgcn_sched_barrier(0);

  o[4] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[4]);
  v0 = *(const LDSP h8 *)(vp2 + 1024u);
  __builtin_amdgcn_sched_barrier(0);
  o[5] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[5]);
  v1 = *(const LDSP h8 *)(vp3 + 1024u);
  if (has_next) {
    /* All LDS loads of buffer `bf` issued: flush dscnt & signal BEFORE final 2 PV WMMAs! */
    ink_r10_att_lsignal();
  } else {
    __builtin_amdgcn_sched_barrier(0);
  }

  o[6] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[6]);
  o[7] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[7]);
}

/* One-sided exact conditional-rescaling softmax + split-barrier PV (0-ULP exact vs pf_r7!).
 * Keeping barriers outside `if (active)` prevents +64-VGPR o[0..7] duplication! */
static inline __attribute__((always_inline)) void ink_r10_att_softmax_pv16_sb(
    const float s[8], const u32 bf, const int has_next,
    float *m_p, float *ls_p, v8f o[8]) {
  float bm = s[0];
#pragma unroll
  for (u32 v = 1u; v < 8u; v++) bm = __builtin_fmaxf(bm, s[v]);
  bm = __builtin_fmaxf(bm, ink_r10_att_xrow(bm));

  const float m = *m_p;
  const int active = (__builtin_amdgcn_ballot_w32(bm > -1e30f && !(bm - m < -30.0f)) != 0u);
  if (active) {
    const int need_rescale = (__builtin_amdgcn_ballot_w32(bm > m || m <= -1e30f) != 0u);
    float mn = m;
    float c = 1.0f;
    if (need_rescale) {
      mn = __builtin_fmaxf(m, bm);
      c = (m > -1e30f) ? __builtin_amdgcn_exp2f(m - mn) : 0.0f;
    }

    h8 pb;
    float ps = 0.0f;
    if (mn > -1e30f) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const float p = __builtin_amdgcn_exp2f(s[v] - mn);
        ps += p;
        pb[v] = (f16)p;
      }
    } else {
      pb = (h8)(0.0f);
    }
    *ls_p = (*ls_p) * c + ps;
    *m_p = mn;

    if (need_rescale) {
#pragma unroll
      for (u32 nt = 0u; nt < 8u; nt++)
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c;
    }

    ink_r10_att_pv16_pipe_sb(bf, pb, o, has_next);
  }
  if (has_next) {
    if (!active) ink_r10_att_lsignal();
    ink_r10_att_lwait();
  }
}

/* Two-phase split-barrier consumer loop (Sync 0 ALREADY completed before entry!). */
static inline __attribute__((always_inline)) void ink_r10_att_consumer16_sb(
    const u32 w, const u32 k_start, const u32 qpos_min, const u32 sub_qpos_min, const u32 sub_qpos_max,
    const u32 win, const u32 extent_lim, const u32 nblk, const u32 blk_far_end,
    const h8 qa[8], const h8 ra, float *m_p, float *ls_p, v8f o[8]) {
  const v8f zero = (v8f)(0.0f);
  const u32 nfar = blk_far_end < nblk ? blk_far_end : nblk;

  /* Phase A: Far-field unmasked blocks. */
#pragma clang loop unroll(disable)
  for (u32 k = 0u; k < nfar; k++) {
    const u32 k_u = (u32)__builtin_amdgcn_readfirstlane((int)k);
    const u32 bf = k_u & 1u;
    float s[8];
    ink_r10_att_qk16_pipe(bf, qa, s);
    ink_r10_att_softmax_pv16_sb(s, bf, (k_u + 1u < nblk), m_p, ls_p, o);
  }

  /* Phase B: Near-field (exact rel_bias + causal/SWA masking). */
#pragma clang loop unroll(disable)
  for (u32 k = nfar; k < nblk; k++) {
    const u32 k_u = (u32)__builtin_amdgcn_readfirstlane((int)k);
    const u32 kb = k_start + INK_R10_ATT_NB16 * k_u;
    const u32 bf = k_u & 1u;
    float s[8];
    ink_r10_att_qk16_pipe(bf, qa, s);

    u32 l_loc = lane();
    __asm__ volatile("" : "+v"(l_loc));
    const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
    const u32 tp = sub_qpos_min + rl;

    const int d_min_sub_s = (int)sub_qpos_min - (int)(kb + (INK_R10_ATT_NB16 - 1u));
    const u32 d_base_sub = d_min_sub_s > 0 ? (u32)d_min_sub_s : 0u;
    if (d_base_sub < extent_lim) {
      const int d_min_wg_s = (int)qpos_min - (int)(kb + (INK_R10_ATT_NB16 - 1u));
      const u32 d_base_wg = d_min_wg_s > 0 ? (u32)d_min_wg_s : 0u;
      const u32 off = d_base_sub - d_base_wg;
      const h8 b0 = *(const LDSP h8 *)(ink_r10_att_nf_lds.rb[bf] + (off + rl) * 16u + 8u * hl);
      const h8 b1 = *(const LDSP h8 *)(ink_r10_att_nf_lds.rb[bf] + (off + 16u + rl) * 16u + 8u * hl);
      const v8f rb0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0, ra, zero);
      const v8f rb1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1, ra, zero);
      LDSP float *sc_row = ink_r10_att_nf_lds.rbsc[w][rl];
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        sc_row[8u * hl + v] = rb0[v];
        sc_row[16u + 8u * hl + v] = rb1[v];
      }
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 kp = kb + 8u * hl + v;
        if (kp <= tp) {
          const u32 dist = tp - kp;
          if (dist < extent_lim) s[v] += sc_row[dist - d_base_sub];
        }
      }
    }

    if (kb + (INK_R10_ATT_NB16 - 1u) > sub_qpos_min || (win > 0u && sub_qpos_max >= kb + win)) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 kp = kb + 8u * hl + v;
        if (kp > tp || (win > 0u && (tp - kp) >= win)) s[v] = -__builtin_inff();
      }
    }

    ink_r10_att_softmax_pv16_sb(s, bf, (k_u + 1u < nblk), m_p, ls_p, o);
  }
}

static inline __attribute__((always_inline)) u32 ink_r10_att_calc_far_end16(
    const u32 win, const u32 qpos_min, const u32 extent_lim, const u32 k_start, const u32 nblk) {
  if (win > 0u) return 0u;
  const u32 guard = extent_lim + (INK_R10_ATT_NB16 - 1u);
  if (qpos_min < guard || qpos_min - guard < k_start) return 0u;
  const u32 b_far = ((qpos_min - guard - k_start) >> 4u) + 1u;
  return b_far < nblk ? b_far : nblk;
}

/* ============================================================================
 * Kernel 2: ie_inkling_rel_attn_pf_near_merge_r10
 *   Consumes (O_far, M_far, L_far) from far_r10 and runs ONLY the <= 66
 *   near-field blocks (exact rel_bias + causal mask + split barriers +
 *   conditional rescaling) to produce final O!
 * ========================================================================== */
KERNEL ie_inkling_rel_attn_pf_near_merge_r10(
    const G float *q, const G kvt *kc, const G kvt *vc,
    const G float *R, const G float *rel_bias, const G float *sink,
    const G float *part, G float *out,
    u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
    u32 d_rel, u32 rel_extent, u32 win,
    u32 qs, u32 rs, u32 ss, u32 os) {
  if (hd != 128u || d_rel != 16u || n_kv == 0u || n_head != n_kv * 3u || T == 0u || kv_len == 0u) return;

  const u32 ntile = (T + 31u) >> 5u;
  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  if (wgid() >= ntile || kh >= n_kv) return;

  const u32 tile = (ntile == 16u && n_kv == 8u && win == 0u && kh < 4u) ? wgid() : ((ntile - 1u) - wgid());
  const u32 t0 = tile << 5u;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));

  const u32 tlast = (t0 + 31u < T) ? (t0 + 31u) : (T - 1u);
  const u32 qpos_min = pos0 + t0;
  const u32 qpos_max = pos0 + tlast;
  const u32 nblk32_far = (win == 0u && part) ? ink_r10_att_far_nblk32(qpos_min, rel_extent, kv_len) : 0u;
  const u32 k_start = (nblk32_far > 0u)
                          ? (nblk32_far << 5u)
                          : ((win > 0u && qpos_min + 1u > win) ? (qpos_min + 1u - win) : 0u);
  const u32 kend = qpos_max + 1u;
  const u32 nblk = (kend > k_start) ? ((kend - k_start + INK_R10_ATT_NB16 - 1u) >> 4u) : 0u;
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const u32 blk_far_end = ink_r10_att_calc_far_end16(win, qpos_min, extent_lim, k_start, nblk);

  if (w >= 6u) {
    const unsigned long row_step = (unsigned long)n_kv * INK_R10_ATT_ROWB;
    const G u8 *kb_kh = (const G u8 *)kc + (unsigned long)kh * INK_R10_ATT_ROWB;
    const G u8 *vb_kh = (const G u8 *)vc + (unsigned long)kh * INK_R10_ATT_ROWB;
    const int no_wrap = (kend <= kv_len);
    ink_r10_att_producer16_sb(kb_kh, vb_kh, rel_bias, row_step, kv_len, k_start, kend,
                              qpos_min, extent_lim, nblk, blk_far_end, no_wrap, tid() - 192u);
    return;
  }

  const u32 tsub = w >= 3u ? 1u : 0u;
  const u32 hsub = w - 3u * tsub;
  const u32 t_sub0 = t0 + 16u * tsub;
  if (t_sub0 >= T) {
    for (u32 k = 0u; k < nblk; k++) {
      ink_r10_att_lsignal();
      ink_r10_att_lwait();
    }
    return;
  }

  if (nblk > 0u) ink_r10_att_lsignal();

  const u32 h = (u32)__builtin_amdgcn_readfirstlane((int)(kh * 3u + hsub));
  const u32 sub_qpos_min = pos0 + t_sub0;
  const u32 sub_qpos_max = pos0 + (t_sub0 + 15u);

  h8 qa[8];
  h8 ra;
  float m, ls;
  v8f o[8];
  {
    const u32 l = lane(), hl = l >> 4u, rl = l & 15u;
    const u32 tk = t_sub0 + rl;
    const u32 tq = tk < T ? tk : (T - 1u);
    const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (unsigned long)tq * qs) +
                                           (unsigned long)h * 128ul);
#pragma unroll
    for (u32 ks = 0u; ks < 8u; ks++) {
      const f32x4 v0 = qp4[4u * ks + 2u * hl + 0u];
      const f32x4 v1 = qp4[4u * ks + 2u * hl + 1u];
      qa[ks] = (h8){(f16)(v0.x * INK_R10_ATT_QSC), (f16)(v0.y * INK_R10_ATT_QSC),
                    (f16)(v0.z * INK_R10_ATT_QSC), (f16)(v0.w * INK_R10_ATT_QSC),
                    (f16)(v1.x * INK_R10_ATT_QSC), (f16)(v1.y * INK_R10_ATT_QSC),
                    (f16)(v1.z * INK_R10_ATT_QSC), (f16)(v1.w * INK_R10_ATT_QSC)};
    }
    const G f32x4 *rp4 = (const G f32x4 *)((const G float *)((const G u8 *)R + (unsigned long)tq * rs) +
                                           (unsigned long)h * 16ul);
    const f32x4 r0 = rp4[2u * hl + 0u];
    const f32x4 r1 = rp4[2u * hl + 1u];
    ra = (h8){(f16)(r0.x * INK_R10_ATT_LOG2E), (f16)(r0.y * INK_R10_ATT_LOG2E),
              (f16)(r0.z * INK_R10_ATT_LOG2E), (f16)(r0.w * INK_R10_ATT_LOG2E),
              (f16)(r1.x * INK_R10_ATT_LOG2E), (f16)(r1.y * INK_R10_ATT_LOG2E),
              (f16)(r1.z * INK_R10_ATT_LOG2E), (f16)(r1.w * INK_R10_ATT_LOG2E)};

    if (nblk32_far > 0u) {
      const G float *pp = part + (unsigned long)(tq * n_head + h) * INK_R10_ATT_PSTR;
      m = pp[0];
      ls = (hl == 0u) ? pp[1] : 0.0f;
      const G ink_r10_att_f2 *po2 = (const G ink_r10_att_f2 *)(pp + 2u);
#pragma unroll
      for (u32 nt = 0u; nt < 8u; nt++) {
        const u32 base2 = 8u * nt + 4u * hl;
        const ink_r10_att_f2 a0 = po2[base2 + 0u];
        const ink_r10_att_f2 a1 = po2[base2 + 1u];
        const ink_r10_att_f2 a2 = po2[base2 + 2u];
        const ink_r10_att_f2 a3 = po2[base2 + 3u];
        o[nt] = (v8f){a0.x, a0.y, a1.x, a1.y, a2.x, a2.y, a3.x, a3.y};
      }
    } else {
      const float sink_val = ((const G float *)((const G u8 *)sink + (unsigned long)tq * ss))[h];
      m = sink_val * INK_R10_ATT_LOG2E;
      ls = (hl == 0u) ? 1.0f : 0.0f;
#pragma unroll
      for (u32 nt = 0u; nt < 8u; nt++) o[nt] = (v8f)(0.0f);
    }
  }

  if (nblk > 0u) {
    ink_r10_att_lwait();
    ink_r10_att_consumer16_sb(w, k_start, qpos_min, sub_qpos_min, sub_qpos_max,
                              win, extent_lim, nblk, blk_far_end, qa, ra, &m, &ls, o);
  }

  const float inv = 1.0f / (ls + ink_r10_att_xrow(ls));
  u32 l_ep = lane();
  __asm__ volatile("" : "+v"(l_ep));
  const u32 rl_ep = l_ep & 15u, hl_ep = l_ep >> 4u;
  const u32 tk_ep = t_sub0 + rl_ep;
  if (tk_ep < T) {
    G f32x4 *op4 = (G f32x4 *)((G float *)((G u8 *)out + (unsigned long)tk_ep * os) + (unsigned long)h * 128ul);
#pragma unroll
    for (u32 nt = 0u; nt < 8u; nt++) {
      const f32x4 a0 = {o[nt][0] * inv, o[nt][1] * inv, o[nt][2] * inv, o[nt][3] * inv};
      const f32x4 a1 = {o[nt][4] * inv, o[nt][5] * inv, o[nt][6] * inv, o[nt][7] * inv};
      op4[4u * nt + 2u * hl_ep + 0u] = a0;
      op4[4u * nt + 2u * hl_ep + 1u] = a1;
    }
  }
}

/* ============================================================================
 * Kernel 3: ie_inkling_rel_attn_pf_sb_r10
 *   100% Exact-SHA drop-in upgrade of ie_inkling_rel_attn_pf_r7 adding
 *   exact m_new == m_old rescale skip + gfx1201 split barriers for ALL 40 SWA
 *   layers (and Global layers when far/near split is inactive)!
 * ========================================================================== */
KERNEL ie_inkling_rel_attn_pf_sb_r10(
    const G float *q, const G kvt *kc, const G kvt *vc,
    const G float *R, const G float *rel_bias, const G float *sink, G float *out,
    u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
    u32 d_rel, u32 rel_extent, u32 win,
    u32 qs, u32 rs, u32 ss, u32 os) {
  if (hd != 128u || d_rel != 16u || n_kv == 0u || n_head != n_kv * 3u || T == 0u || kv_len == 0u) return;

  const u32 ntile = (T + 31u) >> 5u;
  const u32 kh = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  if (wgid() >= ntile || kh >= n_kv) return;

  const u32 tile = (ntile == 16u && n_kv == 8u && win == 0u && kh < 4u) ? wgid() : ((ntile - 1u) - wgid());
  const u32 t0 = tile << 5u;
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));

  const u32 tlast = (t0 + 31u < T) ? (t0 + 31u) : (T - 1u);
  const u32 qpos_min = pos0 + t0;
  const u32 qpos_max = pos0 + tlast;
  const u32 k_start = (win > 0u && qpos_min + 1u > win) ? (qpos_min + 1u - win) : 0u;
  const u32 kend = qpos_max + 1u;
  const u32 nblk = (kend > k_start) ? ((kend - k_start + INK_R10_ATT_NB16 - 1u) >> 4u) : 0u;
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const u32 blk_far_end = ink_r10_att_calc_far_end16(win, qpos_min, extent_lim, k_start, nblk);

  if (w >= 6u) {
    const unsigned long row_step = (unsigned long)n_kv * INK_R10_ATT_ROWB;
    const G u8 *kb_kh = (const G u8 *)kc + (unsigned long)kh * INK_R10_ATT_ROWB;
    const G u8 *vb_kh = (const G u8 *)vc + (unsigned long)kh * INK_R10_ATT_ROWB;
    const int no_wrap = (kend <= kv_len);
    ink_r10_att_producer16_sb(kb_kh, vb_kh, rel_bias, row_step, kv_len, k_start, kend,
                              qpos_min, extent_lim, nblk, blk_far_end, no_wrap, tid() - 192u);
    return;
  }

  const u32 tsub = w >= 3u ? 1u : 0u;
  const u32 hsub = w - 3u * tsub;
  const u32 t_sub0 = t0 + 16u * tsub;
  if (t_sub0 >= T) {
    for (u32 k = 0u; k < nblk; k++) {
      ink_r10_att_lsignal();
      ink_r10_att_lwait();
    }
    return;
  }

  if (nblk > 0u) ink_r10_att_lsignal();

  const u32 h = (u32)__builtin_amdgcn_readfirstlane((int)(kh * 3u + hsub));
  const u32 sub_qpos_min = pos0 + t_sub0;
  const u32 sub_qpos_max = pos0 + (t_sub0 + 15u);

  h8 qa[8];
  h8 ra;
  float m, ls;
  {
    const u32 l = lane(), hl = l >> 4u, rl = l & 15u;
    const u32 tk = t_sub0 + rl;
    const u32 tq = tk < T ? tk : (T - 1u);
    const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (unsigned long)tq * qs) +
                                           (unsigned long)h * 128ul);
#pragma unroll
    for (u32 ks = 0u; ks < 8u; ks++) {
      const f32x4 v0 = qp4[4u * ks + 2u * hl + 0u];
      const f32x4 v1 = qp4[4u * ks + 2u * hl + 1u];
      qa[ks] = (h8){(f16)(v0.x * INK_R10_ATT_QSC), (f16)(v0.y * INK_R10_ATT_QSC),
                    (f16)(v0.z * INK_R10_ATT_QSC), (f16)(v0.w * INK_R10_ATT_QSC),
                    (f16)(v1.x * INK_R10_ATT_QSC), (f16)(v1.y * INK_R10_ATT_QSC),
                    (f16)(v1.z * INK_R10_ATT_QSC), (f16)(v1.w * INK_R10_ATT_QSC)};
    }
    const G f32x4 *rp4 = (const G f32x4 *)((const G float *)((const G u8 *)R + (unsigned long)tq * rs) +
                                           (unsigned long)h * 16ul);
    const f32x4 r0 = rp4[2u * hl + 0u];
    const f32x4 r1 = rp4[2u * hl + 1u];
    ra = (h8){(f16)(r0.x * INK_R10_ATT_LOG2E), (f16)(r0.y * INK_R10_ATT_LOG2E),
              (f16)(r0.z * INK_R10_ATT_LOG2E), (f16)(r0.w * INK_R10_ATT_LOG2E),
              (f16)(r1.x * INK_R10_ATT_LOG2E), (f16)(r1.y * INK_R10_ATT_LOG2E),
              (f16)(r1.z * INK_R10_ATT_LOG2E), (f16)(r1.w * INK_R10_ATT_LOG2E)};

    const float sink_val = ((const G float *)((const G u8 *)sink + (unsigned long)tq * ss))[h];
    m = sink_val * INK_R10_ATT_LOG2E;
    ls = (hl == 0u) ? 1.0f : 0.0f;
  }

  v8f o[8];
#pragma unroll
  for (u32 nt = 0u; nt < 8u; nt++) o[nt] = (v8f)(0.0f);

  if (nblk > 0u) {
    ink_r10_att_lwait();
    ink_r10_att_consumer16_sb(w, k_start, qpos_min, sub_qpos_min, sub_qpos_max,
                              win, extent_lim, nblk, blk_far_end, qa, ra, &m, &ls, o);
  }

  const float inv = 1.0f / (ls + ink_r10_att_xrow(ls));
  u32 l_ep = lane();
  __asm__ volatile("" : "+v"(l_ep));
  const u32 rl_ep = l_ep & 15u, hl_ep = l_ep >> 4u;
  const u32 tk_ep = t_sub0 + rl_ep;
  if (tk_ep < T) {
    G f32x4 *op4 = (G f32x4 *)((G float *)((G u8 *)out + (unsigned long)tk_ep * os) + (unsigned long)h * 128ul);
#pragma unroll
    for (u32 nt = 0u; nt < 8u; nt++) {
      const f32x4 a0 = {o[nt][0] * inv, o[nt][1] * inv, o[nt][2] * inv, o[nt][3] * inv};
      const f32x4 a1 = {o[nt][4] * inv, o[nt][5] * inv, o[nt][6] * inv, o[nt][7] * inv};
      op4[4u * nt + 2u * hl_ep + 0u] = a0;
      op4[4u * nt + 2u * hl_ep + 1u] = a1;
    }
  }
}

#undef INK_R10_ATT_PSTR
#undef INK_R10_ATT_ROWB
#undef INK_R10_ATT_QSC
#undef INK_R10_ATT_LOG2E
#undef INK_R10_ATT_D
#undef INK_R10_ATT_NB32
#undef INK_R10_ATT_NB16

#endif /* !IE_MEGA */
