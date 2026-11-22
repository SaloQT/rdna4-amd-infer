/* ============================================================================
 * Inkling-2 Round 6: Fused Prefill IQ3_XXS Gate/Up WMMA + Exact SITU + Q8_0
 *   Target: RX 9070 XT (gfx1201, wave32, NT=256, ROCm/Clang AMDGPU)
 *   Audit Exp #4: eliminates 36 MiB FP32 exp_gu materialization at C512 and
 *   ~27 GiB of logical exp_gu write+reread traffic across a 4K prefill.
 *
 * Exported Kernel (included into kernels/ie_inkling_kernels.c after ie_kernels.c):
 *   KERNEL ie_inkling_gemm_rq_exp_iq3xxs_situ_r6(
 *       const G u8 *qw, const G u8 *X, G u8 *Qout,
 *       u32 eff, u32 nb,
 *       const G u32 *list, const G u32 *offs, u32 np,
 *       u32 q_bytes, u32 xs, u32 qstr, u32 xdiv,
 *       float beta, float lbeta)
 *
 * Replaces:
 *   Prefill [OP_GEMV_EXP(IQ3_XXS egu [2*eff, nb*32]), OP_SITU(Q8_0 only)]
 *   (Inkling-2-Mini: 48 MoE visits/chunk, eff=1152, nb=72, n_exp=128, K=8,
 *    beta=4.0f, lbeta=25.0f).
 *
 * Grid:
 *   ((eff + 63u) / 64u, list ? n_exp : 1u, 1u)
 *   - For Inkling-2-Mini (eff=1152, n_exp=128): (18, 128, 1) = 2,304 WGs
 *     (exact same 18 WGs/expert grid as R1's 2304/128, zero occupancy loss).
 *
 * Dispatch Guards (src/hip.c ink_r2_fuse / prefill fusion hook):
 *   - k0 == OP_GEMV_EXP && !mk.rec && T >= T_min (e.g. 17u or rq_exp_min=32u)
 *     && !getenv("IE_GEMM_RQ_EXP_NO") && i + 1u < i1
 *   - o0->w->kind == MAT_RAWQ && o0->w->sub == 18u (GGML_IQ3_XXS)
 *   - o1->kind == OP_SITU && o1->a == o0->b && o1->aoff == 0u && o0->boff == 0u
 *   - o1->nh == o0->nh && o1->b < 0 && o1->qo >= 0
 *   - o1->n > 0u && (o1->n & 31u) == 0u && o0->w->rows == 2u * o1->n
 *   - o0->w->nb >= 8u && (o0->w->nb & 7u) == 0u
 *   - g->bufs[o1->qo].stride == o1->nh * o1->bstr && (o1->bstr & 3u) == 0u
 *   - o0->astr == 0u || g->bufs[o0->a].stride == o0->nh * o0->astr
 *   - buf_disjoint(g, o0->a, o1->qo)
 *   - Arguments:
 *       qw      = b->d[i].w1
 *       X       = b->arena + g->bufs[o0->a].off + o0->aoff
 *       Qout    = b->arena + g->bufs[o1->qo].off
 *       eff     = o1->n
 *       nb      = o0->w->nb
 *       list    = b->ms_list, offs = b->ms_offs, np = 0u
 *       q_bytes = (u32)ie_rawq_bytes(o0->w)
 *       xs      = o0->astr ? o0->astr : g->bufs[o0->a].stride
 *       qstr    = o1->bstr   (e.g. 1536 B for eff=1152; SoA Q8_0: 40*(eff/32)=1440 B)
 *       xdiv    = o0->astr ? 1u : o0->nh
 *       beta    = o1->clamp, lbeta = o1->lam
 *
 * Parity:
 *   100% bitwise-exact (0 bit diffs in Qout codes, FP32 scales, and u32 sums)
 *   vs ie_inkling_gemm_rq_exp_iq3xxs_r1 + ie_inkling_situ_r1 (and base
 *   ie_gemm_rq_exp_iq3xxs + ie_situ):
 *   - Exact per-32 INT8 WMMA + FP32 scale*dx FMA K-loop (default fp contract,
 *     matching ie_gemm_rq_exp_iq3xxs / r1 / r3 bit-for-bit).
 *   - Exact SITU in dedicated #pragma clang fp contract(off) epilogue helper
 *     (beta * ie_tanhf_s(g / beta), lbeta * ie_tanhf_s(u / lbeta),
 *      gs * sigmoidf(g) * us) + exact SoA Q8_0 via quant_wave:
 *       codes : ((G i8 *)qp)[blk * 32u + l]          [bytes 0 .. 32*nb_out)
 *       scale : ((G float *)(qp + 32u * nb_out))[blk] [bytes 32*nb_out .. 36*nb_out)
 *       asum  : ((G u32 *)(qp + 36u * nb_out))[blk]   [bytes 36*nb_out .. 40*nb_out)
 *     where qp = Qout + (unsigned long)pid * qstr and nb_out = eff >> 5u.
 *
 * Row Pairing & Zero-Extra-LDS Epilogue Layout (19,712 B total, 0 B extra LDS):
 *   - 8 waves cover 128 rows = 64 paired (gate, up) channels:
 *       waves 0..3 (is_up=0): gate rows gbase + 16*w + r16       (0 .. eff-1)
 *       waves 4..7 (is_up=1): up   rows eff + gbase + 16*(w-4) + r16 (eff .. 2*eff-1)
 *     where gbase = wgid() * 64u, r16 = lane() & 15u, h = lane() >> 4u.
 *     Handles eff % 64 == 32 cleanly via nblk = (eff - gbase >= 64u) ? 2u : 1u
 *     and safe row clamping on waves 2,3,6,7.
 *   - Adaptive 128x64 (rem > 32, WJ=4) / 128x32 (rem <= 32, WJ=2) token tiling.
 *   - Union aliases dead activation LDS (x[2][64*128] + sx[2][4*64] = 18,432 B)
 *     with float ep[2][32][66] (16,896 B <= 18,432 B); lut[256] and pid[64] sit
 *     outside the union (bytes 18,432..19,711) and remain live across tiles.
 *   - Stride 66 floats gives 8*66 = 528 == 16 (mod 32) -> 0 bank conflicts on
 *     both WMMA-to-LDS write (h=0 lanes 0..15 vs h=1 lanes 16..31) and
 *     contiguous 32-lane LDS-to-SITU read (ep[0/1][tk][32*b + l]).
 *   - For 64-pair tiles, epilogue runs as two 32-slot strips (acc[0..1] for
 *     slots 0..31, then acc[2..3] for slots 32..nj-1) with lbarrier() fencing
 *     every producer/consumer handoff.
 *
 * Low-VGPR Design (target <= 160 VGPR / 2 WGs/CU, 0 spills):
 *   - Wave-uniform SGPR bases (ebase, X, X_s, w, is_up, p0, p1) + 32-bit VGPR
 *     offsets (row_off, xoff0, xoff1, sxoff) eliminate 64-bit VGPR pointers.
 *   - Early staging of rx0/rx1/rsx into the dead double-buffer right after k=1
 *     frees 12 VGPRs BEFORE k=2..3 2-k decode + in-place dead-weight reload,
 *     so activation prefetch never overlaps xdc[2] or weight tail reloads.
 *   - Single-mul IQ3_XXS sign decode + split-half dlo/dhi FMA (no t0[8] array)
 *     + SGPR-pinned nk basic-block guards prevent LLVM hoisting/spilling.
 *
 * ============================================================================
 * REQUIRED src/model.c ARENA & ALIASING CHECK FOR OMITTING / CAPPING exp_gu
 * ============================================================================
 * In src/model.c (inkling_build, lines 3194-3209):
 *   1. Aliasing / use_last invariant (MUST ALWAYS BE PRESERVED):
 *      - Currently line 3204-3205 emits:
 *          use_last(B, mq);
 *          use_last(B, hq); // keeps ffn_in_q8 disjoint from exp_mid_q8
 *        Even if exp_gu (ye) is shrunk or omitted, hq (ffn_in_q8) MUST remain
 *        live through the op that produces mq (exp_mid_q8) via use_last(B, hq)
 *        (and rt remains live through OP_MOE_SUM), so plan_arena() + verified
 *        ie_plan_check() guarantees buf_disjoint(g, hq, mq) and
 *        buf_disjoint(g, rt, mq).
 *      - Runtime hip.c fusion guard MUST still check buf_disjoint(g, o0->a, o1->qo).
 *   2. Exact eligibility check before shrinking/omitting exp_gu (36 MiB at C512):
 *      All 48 Inkling-2 visits (or each visit with a qualifying layer L) satisfy:
 *        - L->egu.rows > 0 (merged [gate; up], parts[1] == NULL)
 *        - L->egu.kind == MAT_RAWQ && L->egu.sub == GGML_IQ3_XXS (18u)
 *        - L->egu.rows == 2u * m->eff && m->eff > 0u && (m->eff & 31u) == 0u
 *        - L->egu.nb >= 8u && (L->egu.nb & 7u) == 0u
 *        - L->ed.kind != MAT_F32 (so mq >= 0 and OP_GEMV_EXP down reads mq,
 *          never ye, and OP_SITU has o->b == -1)
 *      AND the host build/runtime has NO unfused path that can run with T > T_cap:
 *        - GPU backend only (NOT --backend cpu, NOT IE_DUMP_GPU_OPS=1, NOT
 *          IE_GEMV_NOBATCH, NOT IE_GEMM_RQ_EXP_NO, NOT IE_INKLING_KERNELS=0,
 *          and the R6 kernel mask bit is enabled in IE_INKLING_KERNEL_MASK).
 *   3. Multi-regime T safety (CRITICAL: one graph serves T=1, T=2..16 verify,
 *      AND T>16 prefill chunks!):
 *      - Because main.c / serve.c build ONE graph at g->T (e.g. 512) and run
 *        T=1 decode and T=2..16 DSpark verify on the same arena:
 *        (a) Safe partial-cap path (recommended, parallels graph_cap_logits()
 *            at model.c:2497): if hip.c dispatches this fused R6 kernel for
 *            ALL T > T_cap (e.g. T_cap = 16u so T >= 17u fuses while T=1 uses
 *            R1/R3 decode and T=2..16 uses R4 verify), cap each qualifying
 *            exp_gu buffer BEFORE plan_arena() to:
 *              b->rows = T_cap; b->size = b->stride * T_cap;
 *            At T_cap = 16 (eff=1152, K=8), exp_gu shrinks 36.0 MiB -> 1.125 MiB
 *            (-34.875 MiB), and hip.c asserts (b->rows == 0 || T <= b->rows)
 *            before any unfused OP_GEMV_EXP/OP_SITU fallback.
 *        (b) Full 36.0 MiB omission (ye removed / size 0): safe ONLY if EVERY
 *            T in 1..g->T executes a fused gate/up+SITU kernel (T=1 fused
 *            gexp_iq3xxs_situ_r3/r2 + T=2..16 fused verify + T>=17 fused R6)
 *            with zero unfused fallback paths (including megakernel rec).
 *      - Arena high-water caveat: because exp_down (yd, also 36 MiB at C512)
 *        does not overlap exp_gu in lifetime, verify actual net g->arena drop
 *        via IE_PLAN_DUMP=1 (peak may shift to exp_down + exp_mid_q8 + blocks).
 * ========================================================================== */

#ifndef IE_MEGA

struct __attribute__((packed, aligned(4))) ink_r6_pf_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r6_pf_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r6_pf_u32x4p *)p)->v;
}

typedef struct {
  v2i w0, w1;
  float s0;
} ink_r6_pf_dc_t;

typedef struct {
  union {
    struct {
      u8 x[2][64u * 128u] __attribute__((aligned(16)));
      float sx[2][4u * 64u] __attribute__((aligned(32)));
    } k;
    float ep[2][32u][66u] __attribute__((aligned(32)));
  } u;
  u32 lut[256] __attribute__((aligned(16)));
  u32 pid[64] __attribute__((aligned(16)));
} ink_r6_pf_lds_t;

_Static_assert(sizeof(ink_r6_pf_lds_t) == 19712u, "ink_r6_pf_lds_t must be 19,712 B");

static inline __attribute__((always_inline)) u32 ink_r6_pf_perm_al(const u32 lo, const u32 hi, const u32 psel) {
  return __builtin_amdgcn_perm(hi, lo, psel);
}

static inline __attribute__((always_inline)) u32 ink_r6_pf_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

/* Single-mul IQ3_XXS 7-bit+parity sign decode: bitwise identical to
 * rq_ksign7(s7) + 2x ie_rq_ssign(g, ie_rq_mask4(...)) for all s7 in [0, 127]. */
static inline __attribute__((always_inline)) v2i ink_r6_pf_iq3xxs_ksw(const u32 g0, const u32 g1, const u32 s7) {
  const u32 p_ = s7 * 0x00204081u;
  const u32 t0 = p_ & 0x01010101u;
  const u32 t1 = ((p_ >> 4u) | ((u32)__builtin_popcount(s7) << 24u)) & 0x01010101u;
  const u32 m0 = __builtin_amdgcn_perm(0u, 0u, t0 | 0x0C0C0C0Cu);
  const u32 m1 = __builtin_amdgcn_perm(0u, 0u, t1 | 0x0C0C0C0Cu);
  const v2i v = {(int)((g0 ^ m0) + t0), (int)((g1 ^ m1) + t1)};
  return v;
}

/* Load IQ3_XXS hb0 (base[0..8] = d+qs_hb0, base[16..20] = aux_hb0) in 5 global loads.
 * base[8] and base[20] are carried forward into cur_q[0]/cur_a[0] for hb1. */
static inline __attribute__((always_inline)) void ink_r6_pf_load_hb0(const G u8 *p, u32 qw[9], u32 aw[5]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v0 = ink_r6_pf_ld4(base + 0u);
  const u32x4 v1 = ink_r6_pf_ld4(base + 4u);
  qw[0] = v0.x; qw[1] = v0.y; qw[2] = v0.z; qw[3] = v0.w;
  qw[4] = v1.x; qw[5] = v1.y; qw[6] = v1.z; qw[7] = v1.w;
  qw[8] = base[8];
  const u32x4 a0 = ink_r6_pf_ld4(base + 16u);
  aw[0] = a0.x; aw[1] = a0.y; aw[2] = a0.z; aw[3] = a0.w;
  aw[4] = base[20];
}

/* Load IQ3_XXS hb1 tail (base[9..16] = qs_hb1, base[21..24] = aux_hb1) in 3x global_load_b128. */
static inline __attribute__((always_inline)) void ink_r6_pf_load_hb1_tail(const G u8 *p, u32 qt[8], u32 at[4]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v0 = ink_r6_pf_ld4(base + 9u);
  const u32x4 v1 = ink_r6_pf_ld4(base + 13u);
  qt[0] = v0.x; qt[1] = v0.y; qt[2] = v0.z; qt[3] = v0.w;
  qt[4] = v1.x; qt[5] = v1.y; qt[6] = v1.z; qt[7] = v1.w;
  const u32x4 a0 = ink_r6_pf_ld4(base + 21u);
  at[0] = a0.x; at[1] = a0.y; at[2] = a0.z; at[3] = a0.w;
}

/* 1-k IQ3_XXS decode (used for k=0, 1 while activation prefetch is live -> -5 VGPRs). */
static inline __attribute__((always_inline)) void ink_r6_pf_decode_one(
    const u32 *qw, const u32 *aw, const u32 psel, const float d2, const LDSP u32 *lut,
    const u32 h, ink_r6_pf_dc_t *dc) {
  const u32 psel_q = h ? 0x07060302u : 0x05040100u;
  const u32 auxv = ink_r6_pf_perm_al(aw[0], aw[1], psel);
  const u32 w0 = ink_r6_pf_perm_al(qw[0], qw[1], psel);
  const u32 w1 = ink_r6_pf_perm_al(qw[1], qw[2], psel);
  const u32 qp = __builtin_amdgcn_perm(w1, w0, psel_q);
  const u32 g00 = lut[qp & 255u];
  const u32 g01 = lut[(qp >> 8u) & 255u];
  const u32 g10 = lut[(qp >> 16u) & 255u];
  const u32 g11 = lut[qp >> 24u];
  __builtin_amdgcn_sched_barrier(0);
  dc->s0 = d2 * (0.5f + (float)(auxv >> 28u));
  dc->w0 = ink_r6_pf_iq3xxs_ksw(g00, g01, (auxv >> (7u * h)) & 127u);
  dc->w1 = ink_r6_pf_iq3xxs_ksw(g10, g11, (auxv >> (7u * (2u + h))) & 127u);
}

/* 2-k batched IQ3_XXS decode (used for k=2, 3 after rx0/rx1/rsx are staged & dead,
 * allowing in-place hb1_tail / next-hb0 reload to overlap both k=2 and k=3 WMMAs). */
static inline __attribute__((always_inline)) void ink_r6_pf_decode_pair(
    const u32 *qw, const u32 *aw, const u32 psel, const float d2, const LDSP u32 *lut,
    const u32 h, ink_r6_pf_dc_t dc[2]) {
  const u32 psel_q = h ? 0x07060302u : 0x05040100u;
  u32 auxv[2], g00[2], g01[2], g10[2], g11[2];
#pragma unroll
  for (u32 dk = 0u; dk < 2u; dk++) {
    auxv[dk] = ink_r6_pf_perm_al(aw[dk], aw[dk + 1u], psel);
    const u32 w0 = ink_r6_pf_perm_al(qw[2u * dk], qw[2u * dk + 1u], psel);
    const u32 w1 = ink_r6_pf_perm_al(qw[2u * dk + 1u], qw[2u * dk + 2u], psel);
    const u32 qp = __builtin_amdgcn_perm(w1, w0, psel_q);
    g00[dk] = lut[qp & 255u];
    g01[dk] = lut[(qp >> 8u) & 255u];
    g10[dk] = lut[(qp >> 16u) & 255u];
    g11[dk] = lut[qp >> 24u];
  }
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for (u32 dk = 0u; dk < 2u; dk++) {
    dc[dk].s0 = d2 * (0.5f + (float)(auxv[dk] >> 28u));
  }
#pragma unroll
  for (u32 dk = 0u; dk < 2u; dk++) {
    dc[dk].w0 = ink_r6_pf_iq3xxs_ksw(g00[dk], g01[dk], (auxv[dk] >> (7u * h)) & 127u);
    dc[dk].w1 = ink_r6_pf_iq3xxs_ksw(g10[dk], g11[dk], (auxv[dk] >> (7u * (2u + h))) & 127u);
  }
}

/* Strictly sequenced 2-tile (j0, j0+1) WMMA + exact FMA epilogue.
 * Splits dlo/dhi per 4 elements (reusing 4+4 VGPRs instead of 8+8 for dx+t0[8])
 * while preserving exact (float)r[v] * s0 -> v_fma_f32(..., dx, acc) order. */
static inline __attribute__((always_inline)) void ink_r6_pf_apply_pair_j(
    const ink_r6_pf_dc_t dck, const LDSP u8 *lx, const LDSP float *lsx_k,
    const u32 x_r_off, const u32 c0, const u32 c1, const u32 j0, float acc[][8]) {
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  const u32 j_off = j0 * 2048u;
  const LDSP u8 *lx_r = lx + x_r_off;
  const v2i a0_0 = *(const LDSP v2i *)(lx_r + c0 + j_off);
  const v2i a0_1 = *(const LDSP v2i *)(lx_r + c0 + j_off + 2048u);
  const v2i a1_0 = *(const LDSP v2i *)(lx_r + c1 + j_off);
  const v2i a1_1 = *(const LDSP v2i *)(lx_r + c1 + j_off + 2048u);
  __builtin_amdgcn_sched_barrier(0);

  v8i r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_0, 1, dck.w0, z, 0);
  v8i r1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_1, 1, dck.w0, z, 0);
  r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_0, 1, dck.w1, r0, 0);
  r1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_1, 1, dck.w1, r1, 0);
  __builtin_amdgcn_sched_barrier(0);

  const float s0 = dck.s0;
  const LDSP float *sx0 = lsx_k + 16u * j0;
  {
    const f32x4 dlo = *(const LDSP f32x4 *)(sx0);
    __builtin_amdgcn_sched_barrier(0);
    const float t0 = (float)r0[0] * s0;
    const float t1 = (float)r0[1] * s0;
    const float t2 = (float)r0[2] * s0;
    const float t3 = (float)r0[3] * s0;
    const f32x4 dhi = *(const LDSP f32x4 *)(sx0 + 4u);
    __builtin_amdgcn_sched_barrier(0);
    acc[j0][0] += t0 * dlo[0];
    acc[j0][1] += t1 * dlo[1];
    acc[j0][2] += t2 * dlo[2];
    acc[j0][3] += t3 * dlo[3];
    const float t4 = (float)r0[4] * s0;
    const float t5 = (float)r0[5] * s0;
    const float t6 = (float)r0[6] * s0;
    const float t7 = (float)r0[7] * s0;
    __builtin_amdgcn_sched_barrier(0);
    acc[j0][4] += t4 * dhi[0];
    acc[j0][5] += t5 * dhi[1];
    acc[j0][6] += t6 * dhi[2];
    acc[j0][7] += t7 * dhi[3];
    __builtin_amdgcn_sched_barrier(0);
  }
  {
    const f32x4 dlo = *(const LDSP f32x4 *)(sx0 + 16u);
    __builtin_amdgcn_sched_barrier(0);
    const float t0 = (float)r1[0] * s0;
    const float t1 = (float)r1[1] * s0;
    const float t2 = (float)r1[2] * s0;
    const float t3 = (float)r1[3] * s0;
    const f32x4 dhi = *(const LDSP f32x4 *)(sx0 + 20u);
    __builtin_amdgcn_sched_barrier(0);
    acc[j0 + 1u][0] += t0 * dlo[0];
    acc[j0 + 1u][1] += t1 * dlo[1];
    acc[j0 + 1u][2] += t2 * dlo[2];
    acc[j0 + 1u][3] += t3 * dlo[3];
    const float t4 = (float)r1[4] * s0;
    const float t5 = (float)r1[5] * s0;
    const float t6 = (float)r1[6] * s0;
    const float t7 = (float)r1[7] * s0;
    __builtin_amdgcn_sched_barrier(0);
    acc[j0 + 1u][4] += t4 * dhi[0];
    acc[j0 + 1u][5] += t5 * dhi[1];
    acc[j0 + 1u][6] += t6 * dhi[2];
    acc[j0 + 1u][7] += t7 * dhi[3];
    __builtin_amdgcn_sched_barrier(0);
  }
}

static inline __attribute__((always_inline)) void ink_r6_pf_apply_k64(
    const ink_r6_pf_dc_t dck, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 k, const u32 r16, const u32 h, const u32 x_r_off, float acc[4][8]) {
  const u32 c0 = ((2u * k) ^ (r16 & 7u)) << 4u;
  const u32 c1 = c0 ^ 16u;
  const LDSP float *lsx_k = lsx + k * 64u + 8u * h;
  ink_r6_pf_apply_pair_j(dck, lx, lsx_k, x_r_off, c0, c1, 0u, acc);
  ink_r6_pf_apply_pair_j(dck, lx, lsx_k, x_r_off, c0, c1, 2u, acc);
}

static inline __attribute__((always_inline)) void ink_r6_pf_apply_k32(
    const ink_r6_pf_dc_t dck, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 k, const u32 r16, const u32 h, const u32 x_r_off, float acc[2][8]) {
  const u32 c0 = ((2u * k) ^ (r16 & 7u)) << 4u;
  const u32 c1 = c0 ^ 16u;
  const LDSP float *lsx_k = lsx + k * 32u + 8u * h;
  ink_r6_pf_apply_pair_j(dck, lx, lsx_k, x_r_off, c0, c1, 0u, acc);
}

static inline __attribute__((always_inline)) void ink_r6_pf_stage_x64(
    LDSP u8 *lx, LDSP float *lsx, const u32 st_x, const u32 t0, const int is_s,
    const u32x4 rx0, const u32x4 rx1, const f32x4 rsx) {
  *(LDSP u32x4 *)(lx + st_x) = rx0;
  *(LDSP u32x4 *)(lx + st_x + 4096u) = rx1;
  if (is_s) {
    lsx[t0] = rsx.x;
    lsx[64u + t0] = rsx.y;
    lsx[128u + t0] = rsx.z;
    lsx[192u + t0] = rsx.w;
  }
}

static inline __attribute__((always_inline)) void ink_r6_pf_stage_x32(
    LDSP u8 *lx, LDSP float *lsx, const u32 st_x, const u32 t0, const int is_s,
    const u32x4 rx0, const f32x4 rsx) {
  *(LDSP u32x4 *)(lx + st_x) = rx0;
  if (is_s) {
    lsx[t0] = rsx.x;
    lsx[32u + t0] = rsx.y;
    lsx[64u + t0] = rsx.z;
    lsx[96u + t0] = rsx.w;
  }
}

/* Stage one 32-slot strip (acc[j_base + 0..1]) into dead activation LDS ep[2][32][66]
 * (0 bank conflicts: h=0 lanes 0..15 vs h=1 lanes 16..31 differ by 8*66 = 528 == 16 mod 32),
 * then compute exact SITU (#pragma clang fp contract(off)) + exact SoA Q8_0 (quant_wave)
 * directly into Qout[pid * qstr]. Caller guarantees lbarrier() before entry. */
static inline __attribute__((always_inline)) void ink_r6_pf_epilogue_strip(
    LDSP float *ep, const LDSP u32 *lpid, G u8 *Qout,
    const u32 is_up, const u32 r_loc, const u32 h, const u32 w, const u32 l,
    const u32 strip_base, const u32 nj_strip, const u32 blk0, const u32 nblk,
    const u32 nb_out, const u32 qstr, const float beta, const float lbeta,
    const u32 j_base, float acc[][8]) {
#pragma clang fp contract(off)
  LDSP float *ep_wr = ep + (is_up * 32u + 8u * h) * 66u + r_loc;
#pragma unroll
  for (u32 j_loc = 0u; j_loc < 2u; j_loc++) {
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      ep_wr[(16u * j_loc + v) * 66u] = acc[j_base + j_loc][v];
    }
  }
  lbarrier();

  const int use_beta = beta > 0.0f;
  const int use_lbeta = lbeta > 0.0f;
  for (u32 tk = w; tk < nj_strip; tk += 8u) {
    const u32 pid = (u32)__builtin_amdgcn_readfirstlane((int)lpid[strip_base + tk]);
    G u8 *qp = Qout + (unsigned long)pid * qstr;
    const LDSP float *ep_g = ep + tk * 66u + l;
#pragma unroll
    for (u32 b = 0u; b < 2u; b++) {
      if (b < nblk) {
        const float g = ep_g[32u * b];
        const float u = ep_g[2112u + 32u * b];
        const float gs = use_beta ? beta * ie_tanhf_s(g / beta) : g;
        const float us = use_lbeta ? lbeta * ie_tanhf_s(u / lbeta) : u;
        const float val = gs * sigmoidf(g) * us;
        quant_wave(val, blk0 + b, qp, nb_out);
      }
    }
  }
}

KERNEL ie_inkling_gemm_rq_exp_iq3xxs_situ_r6(const G u8 *qw, const G u8 *X, G u8 *Qout, u32 eff, u32 nb,
                                             const G u32 *list, const G u32 *offs, u32 np, u32 q_bytes,
                                             u32 xs, u32 qstr, u32 xdiv, float beta, float lbeta) {
  static LDS ink_r6_pf_lds_t ink_r6_pf_lds;
  if (eff == 0u || (eff & 31u) != 0u || nb < 8u || (nb & 7u) != 0u || xdiv == 0u || !qw || !X || !Qout) return;

  const u32 e = tokid();
  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)(list ? offs[e] : 0u));
  const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)(list ? offs[e + 1u] : np));
  const u32 gbase = wgid() * 64u;
  if (p0 >= p1 || gbase >= eff) return;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;

  /* Waves 0..3 compute 64 gate rows [gbase .. gbase+63];
   * waves 4..7 compute matching 64 up rows [eff+gbase .. eff+gbase+63]. */
  const u32 is_up = w >> 2u;
  const u32 r_loc = 16u * (w & 3u) + r16;
  const u32 ch = gbase + r_loc;
  const u32 ch_c = ch < eff ? ch : eff - 1u;
  const u32 w_row = is_up ? (eff + ch_c) : ch_c;

  const u32 mb = nb >> 3u;
  const u32 nb_out = eff >> 5u;
  const u32 blk0 = gbase >> 5u;
  const u32 nblk = (eff - gbase >= 64u) ? 2u : 1u;

  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * q_bytes);
  const G u8 *X_s = rq_ptr_u(X + 32ul * nb);
  const u32 row_off = w_row * mb * 98u;

  /* Stage 1 KiB IQ3_XXS grid LUT (synchronized by first tile's pid lbarrier). */
  ink_r6_pf_lds.lut[t0] = ie_rq_iq3xxs_grid[t0];

  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  const u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);
  const u32 x_r_off = r16 * 128u + 8u * h;

  for (u32 pb = p0; pb < p1; ) {
    if (pb > p0) lbarrier();
    const u32 rem = p1 - pb;

    if (rem > 32u) {
      /* ---------------- Adaptive 64-pair tile (128x64, WJ=4) ---------------- */
      const u32 nj = rem < 64u ? rem : 64u;
      const int is_s = (w < 2u);
      if (is_s) {
        const u32 pj = pb + (t0 < nj ? t0 : 0u);
        ink_r6_pf_lds.pid[t0] = list ? list[pj] : pj;
      }
      lbarrier();

      const u32 xoff0 = ink_r6_pf_div_x(ink_r6_pf_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
      const u32 xoff1 = ink_r6_pf_div_x(ink_r6_pf_lds.pid[tk_ld + 32u], xdiv) * xs + c8_ld * 16u;
      const u32 sxoff = is_s ? ink_r6_pf_div_x(ink_r6_pf_lds.pid[t0 & 63u], xdiv) * xs : 0u;

      float acc[4][8];
#pragma unroll
      for (u32 j = 0u; j < 4u; j++) {
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) acc[j][v] = 0.0f;
      }

      u32x4 rx0 = ink_r6_pf_ld4(X + (unsigned long)xoff0);
      u32x4 rx1 = ink_r6_pf_ld4(X + (unsigned long)xoff1);
      f32x4 rsx = is_s ? rq_load_f32x4((const G float *)(X_s + (unsigned long)sxoff)) : (f32x4)(0.0f);
      u32 cur_psel = (((unsigned long)ebase + (unsigned long)row_off) & 2ul) ? 0x07060504u : 0x05040302u;
      u32 cur_q[9], cur_a[5];
      ink_r6_pf_load_hb0(ebase + (unsigned long)row_off, cur_q, cur_a);
      ink_r6_pf_stage_x64(ink_r6_pf_lds.u.k.x[0], ink_r6_pf_lds.u.k.sx[0], st_x, t0, is_s, rx0, rx1, rsx);
      lbarrier();

#pragma unroll 1
      for (u32 sb = 0u; ; sb++) {
        const u32 kb1 = 8u * sb + 4u;
        u32 nk = nb - 8u * sb < 4u ? nb - 8u * sb : 4u;
        __asm__ volatile("" : "+s"(nk));

        /* hb0: prefetch hb1 activations FIRST; stage them after k=1 to free 12 VGPRs. */
        const G u8 *X_hb1 = X + (unsigned long)kb1 * 32ul;
        rx0 = ink_r6_pf_ld4(X_hb1 + (unsigned long)xoff0);
        rx1 = ink_r6_pf_ld4(X_hb1 + (unsigned long)xoff1);
        if (is_s) rsx = rq_load_f32x4((const G float *)(X_s + (unsigned long)kb1 * 4ul + (unsigned long)sxoff));
        __builtin_amdgcn_sched_barrier(0);

        const float d2 = ie_rq_h2f((cur_q[0] >> ((cur_psel & 4u) << 2u)) & 0xFFFFu) * 0.5f;
        ink_r6_pf_dc_t xdc0;

        /* hb0 k = 0, 1 (1-k decode keeps xdc = 5 VGPRs while rx0/rx1/rsx are live) */
        ink_r6_pf_decode_one(cur_q + 0u, cur_a + 0u, cur_psel, d2, ink_r6_pf_lds.lut, h, &xdc0);
        __builtin_amdgcn_sched_barrier(0);
        if (0u < nk) ink_r6_pf_apply_k64(xdc0, ink_r6_pf_lds.u.k.x[0], ink_r6_pf_lds.u.k.sx[0], 0u, r16, h, x_r_off, acc);

        ink_r6_pf_decode_one(cur_q + 2u, cur_a + 1u, cur_psel, d2, ink_r6_pf_lds.lut, h, &xdc0);
        __builtin_amdgcn_sched_barrier(0);
        if (1u < nk) ink_r6_pf_apply_k64(xdc0, ink_r6_pf_lds.u.k.x[0], ink_r6_pf_lds.u.k.sx[0], 1u, r16, h, x_r_off, acc);

        /* x[1]/sx[1] are dead until next lbarrier: stage hb1 activations now (frees 12 VGPRs). */
        ink_r6_pf_stage_x64(ink_r6_pf_lds.u.k.x[1], ink_r6_pf_lds.u.k.sx[1], st_x, t0, is_s, rx0, rx1, rsx);
        __builtin_amdgcn_sched_barrier(0);

        /* hb0 k = 2, 3: 2-k decode -> cur_q/cur_a dead -> in-place hb1_tail reload overlaps k=2,3. */
        ink_r6_pf_dc_t xdc2[2];
        ink_r6_pf_decode_pair(cur_q + 4u, cur_a + 2u, cur_psel, d2, ink_r6_pf_lds.lut, h, xdc2);
        __builtin_amdgcn_sched_barrier(0);
        cur_q[0] = cur_q[8];
        cur_a[0] = cur_a[4];
        ink_r6_pf_load_hb1_tail((ebase + (unsigned long)sb * 98ul) + (unsigned long)row_off, cur_q + 1u, cur_a + 1u);
        __builtin_amdgcn_sched_barrier(0);
        if (2u < nk) ink_r6_pf_apply_k64(xdc2[0], ink_r6_pf_lds.u.k.x[0], ink_r6_pf_lds.u.k.sx[0], 2u, r16, h, x_r_off, acc);
        if (3u < nk) ink_r6_pf_apply_k64(xdc2[1], ink_r6_pf_lds.u.k.x[0], ink_r6_pf_lds.u.k.sx[0], 3u, r16, h, x_r_off, acc);

        lbarrier();

        /* hb1: cur_q/cur_a hold carried base[8]/base[20] + hb1_tail; prefetch next-SB activations. */
        const int more = (sb + 1u < mb);
        if (more) {
          const u32 kbn = 8u * (sb + 1u);
          const G u8 *X_nxt = X + (unsigned long)kbn * 32ul;
          rx0 = ink_r6_pf_ld4(X_nxt + (unsigned long)xoff0);
          rx1 = ink_r6_pf_ld4(X_nxt + (unsigned long)xoff1);
          if (is_s) rsx = rq_load_f32x4((const G float *)(X_s + (unsigned long)kbn * 4ul + (unsigned long)sxoff));
        }
        __builtin_amdgcn_sched_barrier(0);

        /* hb1 k = 0, 1 */
        ink_r6_pf_decode_one(cur_q + 0u, cur_a + 0u, cur_psel, d2, ink_r6_pf_lds.lut, h, &xdc0);
        __builtin_amdgcn_sched_barrier(0);
        if (0u < nk) ink_r6_pf_apply_k64(xdc0, ink_r6_pf_lds.u.k.x[1], ink_r6_pf_lds.u.k.sx[1], 0u, r16, h, x_r_off, acc);

        ink_r6_pf_decode_one(cur_q + 2u, cur_a + 1u, cur_psel, d2, ink_r6_pf_lds.lut, h, &xdc0);
        __builtin_amdgcn_sched_barrier(0);
        if (1u < nk) ink_r6_pf_apply_k64(xdc0, ink_r6_pf_lds.u.k.x[1], ink_r6_pf_lds.u.k.sx[1], 1u, r16, h, x_r_off, acc);

        /* x[0]/sx[0] are dead until next lbarrier: stage next-SB activations now (frees 12 VGPRs). */
        if (more) {
          ink_r6_pf_stage_x64(ink_r6_pf_lds.u.k.x[0], ink_r6_pf_lds.u.k.sx[0], st_x, t0, is_s, rx0, rx1, rsx);
        }
        __builtin_amdgcn_sched_barrier(0);

        /* hb1 k = 2, 3: 2-k decode -> in-place next-SB hb0 reload overlaps k=2,3. */
        ink_r6_pf_decode_pair(cur_q + 4u, cur_a + 2u, cur_psel, d2, ink_r6_pf_lds.lut, h, xdc2);
        __builtin_amdgcn_sched_barrier(0);
        if (more) {
          ink_r6_pf_load_hb0((ebase + (unsigned long)(sb + 1u) * 98ul) + (unsigned long)row_off, cur_q, cur_a);
        }
        __builtin_amdgcn_sched_barrier(0);
        if (2u < nk) ink_r6_pf_apply_k64(xdc2[0], ink_r6_pf_lds.u.k.x[1], ink_r6_pf_lds.u.k.sx[1], 2u, r16, h, x_r_off, acc);
        if (3u < nk) ink_r6_pf_apply_k64(xdc2[1], ink_r6_pf_lds.u.k.x[1], ink_r6_pf_lds.u.k.sx[1], 3u, r16, h, x_r_off, acc);

        if (!more) break;
        cur_psel ^= 0x02020606u;
        lbarrier();
      }

      /* Two-strip zero-extra-LDS exact SITU + Q8_0 epilogue over slots 0..31 then 32..nj-1. */
      __builtin_amdgcn_sched_barrier(0);
      lbarrier();
      ink_r6_pf_epilogue_strip(&ink_r6_pf_lds.u.ep[0][0][0], ink_r6_pf_lds.pid, Qout,
                               is_up, r_loc, h, w, l, 0u, 32u, blk0, nblk, nb_out, qstr,
                               beta, lbeta, 0u, acc);
      lbarrier();
      ink_r6_pf_epilogue_strip(&ink_r6_pf_lds.u.ep[0][0][0], ink_r6_pf_lds.pid, Qout,
                               is_up, r_loc, h, w, l, 32u, nj - 32u, blk0, nblk, nb_out, qstr,
                               beta, lbeta, 2u, acc);
      pb += 64u;
    } else {
      /* ---------------- Adaptive 32-pair tile (128x32, WJ=2) ---------------- */
      const u32 nj = rem;
      const int is_s = (w == 0u);
      if (is_s) {
        const u32 pj = pb + (t0 < nj ? t0 : 0u);
        ink_r6_pf_lds.pid[t0] = list ? list[pj] : pj;
      }
      lbarrier();

      const u32 xoff0 = ink_r6_pf_div_x(ink_r6_pf_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
      const u32 sxoff = is_s ? ink_r6_pf_div_x(ink_r6_pf_lds.pid[t0 & 31u], xdiv) * xs : 0u;

      float acc[2][8];
#pragma unroll
      for (u32 j = 0u; j < 2u; j++) {
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) acc[j][v] = 0.0f;
      }

      u32x4 rx0 = ink_r6_pf_ld4(X + (unsigned long)xoff0);
      f32x4 rsx = is_s ? rq_load_f32x4((const G float *)(X_s + (unsigned long)sxoff)) : (f32x4)(0.0f);
      u32 cur_psel = (((unsigned long)ebase + (unsigned long)row_off) & 2ul) ? 0x07060504u : 0x05040302u;
      u32 cur_q[9], cur_a[5];
      ink_r6_pf_load_hb0(ebase + (unsigned long)row_off, cur_q, cur_a);
      ink_r6_pf_stage_x32(ink_r6_pf_lds.u.k.x[0], ink_r6_pf_lds.u.k.sx[0], st_x, t0, is_s, rx0, rsx);
      lbarrier();

#pragma unroll 1
      for (u32 sb = 0u; ; sb++) {
        const u32 kb1 = 8u * sb + 4u;
        u32 nk = nb - 8u * sb < 4u ? nb - 8u * sb : 4u;
        __asm__ volatile("" : "+s"(nk));

        /* hb0: prefetch hb1 activations FIRST; stage after k=1. */
        rx0 = ink_r6_pf_ld4((X + (unsigned long)kb1 * 32ul) + (unsigned long)xoff0);
        if (is_s) rsx = rq_load_f32x4((const G float *)(X_s + (unsigned long)kb1 * 4ul + (unsigned long)sxoff));
        __builtin_amdgcn_sched_barrier(0);

        const float d2 = ie_rq_h2f((cur_q[0] >> ((cur_psel & 4u) << 2u)) & 0xFFFFu) * 0.5f;
        ink_r6_pf_dc_t xdc0;

        /* hb0 k = 0, 1 */
        ink_r6_pf_decode_one(cur_q + 0u, cur_a + 0u, cur_psel, d2, ink_r6_pf_lds.lut, h, &xdc0);
        __builtin_amdgcn_sched_barrier(0);
        if (0u < nk) ink_r6_pf_apply_k32(xdc0, ink_r6_pf_lds.u.k.x[0], ink_r6_pf_lds.u.k.sx[0], 0u, r16, h, x_r_off, acc);

        ink_r6_pf_decode_one(cur_q + 2u, cur_a + 1u, cur_psel, d2, ink_r6_pf_lds.lut, h, &xdc0);
        __builtin_amdgcn_sched_barrier(0);
        if (1u < nk) ink_r6_pf_apply_k32(xdc0, ink_r6_pf_lds.u.k.x[0], ink_r6_pf_lds.u.k.sx[0], 1u, r16, h, x_r_off, acc);

        ink_r6_pf_stage_x32(ink_r6_pf_lds.u.k.x[1], ink_r6_pf_lds.u.k.sx[1], st_x, t0, is_s, rx0, rsx);
        __builtin_amdgcn_sched_barrier(0);

        /* hb0 k = 2, 3 */
        ink_r6_pf_dc_t xdc2[2];
        ink_r6_pf_decode_pair(cur_q + 4u, cur_a + 2u, cur_psel, d2, ink_r6_pf_lds.lut, h, xdc2);
        __builtin_amdgcn_sched_barrier(0);
        cur_q[0] = cur_q[8];
        cur_a[0] = cur_a[4];
        ink_r6_pf_load_hb1_tail((ebase + (unsigned long)sb * 98ul) + (unsigned long)row_off, cur_q + 1u, cur_a + 1u);
        __builtin_amdgcn_sched_barrier(0);
        if (2u < nk) ink_r6_pf_apply_k32(xdc2[0], ink_r6_pf_lds.u.k.x[0], ink_r6_pf_lds.u.k.sx[0], 2u, r16, h, x_r_off, acc);
        if (3u < nk) ink_r6_pf_apply_k32(xdc2[1], ink_r6_pf_lds.u.k.x[0], ink_r6_pf_lds.u.k.sx[0], 3u, r16, h, x_r_off, acc);

        lbarrier();

        /* hb1 */
        const int more = (sb + 1u < mb);
        if (more) {
          const u32 kbn = 8u * (sb + 1u);
          rx0 = ink_r6_pf_ld4((X + (unsigned long)kbn * 32ul) + (unsigned long)xoff0);
          if (is_s) rsx = rq_load_f32x4((const G float *)(X_s + (unsigned long)kbn * 4ul + (unsigned long)sxoff));
        }
        __builtin_amdgcn_sched_barrier(0);

        /* hb1 k = 0, 1 */
        ink_r6_pf_decode_one(cur_q + 0u, cur_a + 0u, cur_psel, d2, ink_r6_pf_lds.lut, h, &xdc0);
        __builtin_amdgcn_sched_barrier(0);
        if (0u < nk) ink_r6_pf_apply_k32(xdc0, ink_r6_pf_lds.u.k.x[1], ink_r6_pf_lds.u.k.sx[1], 0u, r16, h, x_r_off, acc);

        ink_r6_pf_decode_one(cur_q + 2u, cur_a + 1u, cur_psel, d2, ink_r6_pf_lds.lut, h, &xdc0);
        __builtin_amdgcn_sched_barrier(0);
        if (1u < nk) ink_r6_pf_apply_k32(xdc0, ink_r6_pf_lds.u.k.x[1], ink_r6_pf_lds.u.k.sx[1], 1u, r16, h, x_r_off, acc);

        if (more) {
          ink_r6_pf_stage_x32(ink_r6_pf_lds.u.k.x[0], ink_r6_pf_lds.u.k.sx[0], st_x, t0, is_s, rx0, rsx);
        }
        __builtin_amdgcn_sched_barrier(0);

        /* hb1 k = 2, 3 */
        ink_r6_pf_decode_pair(cur_q + 4u, cur_a + 2u, cur_psel, d2, ink_r6_pf_lds.lut, h, xdc2);
        __builtin_amdgcn_sched_barrier(0);
        if (more) {
          ink_r6_pf_load_hb0((ebase + (unsigned long)(sb + 1u) * 98ul) + (unsigned long)row_off, cur_q, cur_a);
        }
        __builtin_amdgcn_sched_barrier(0);
        if (2u < nk) ink_r6_pf_apply_k32(xdc2[0], ink_r6_pf_lds.u.k.x[1], ink_r6_pf_lds.u.k.sx[1], 2u, r16, h, x_r_off, acc);
        if (3u < nk) ink_r6_pf_apply_k32(xdc2[1], ink_r6_pf_lds.u.k.x[1], ink_r6_pf_lds.u.k.sx[1], 3u, r16, h, x_r_off, acc);

        if (!more) break;
        cur_psel ^= 0x02020606u;
        lbarrier();
      }

      /* Single-strip zero-extra-LDS exact SITU + Q8_0 epilogue over slots 0..nj-1. */
      __builtin_amdgcn_sched_barrier(0);
      lbarrier();
      ink_r6_pf_epilogue_strip(&ink_r6_pf_lds.u.ep[0][0][0], ink_r6_pf_lds.pid, Qout,
                               is_up, r_loc, h, w, l, 0u, nj, blk0, nblk, nb_out, qstr,
                               beta, lbeta, 0u, acc);
      pb += 32u;
    }
  }
}

#endif /* !IE_MEGA */
