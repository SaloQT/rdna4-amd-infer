/* ============================================================================
 * Round 11 Group 5: Unified Bitwise-Exact Decode & Tree-Verify Q8 Long Attention
 *                   for Saluki 27B on RX 9070 XT (gfx1201, wave32, hd=256,
 *                   n_q=24, n_kv=4, T=8, npg=3)
 *
 * Included at the end of kernels/ie_saluki_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed r11_g05_ /
 * R11_G05_ and all macros are #undef'd at EOF.
 *
 * Kernel exported (100% bitwise identical to ie_attn_long / mp_r10 / mp_tr_r10):
 *   ie_saluki_attn_long_mp_r11  (unified tr == NULL & tr != NULL, replaces both
 *                                ie_saluki_attn_long_mp_r10 [523.3 ms] and
 *                                ie_saluki_attn_long_mp_tr_r10 [355.5 ms] =
 *                                878.8 ms combined!)
 *
 * Key SOTA improvements over R10:
 *   1. Zero-spill, 0-private-stack unified tree + non-tree body (<= ~160 VGPRs,
 *      priv = 0 B, 16,384 B LDS -> full 4 WGs/WGP = 128 concurrent WGs on BOTH
 *      non-tree and tree-verify; fixes mp_tr_r10's VGPR=208 / priv=256 B /
 *      40-scratch-inst occupancy cliff via explicit SSA sp0/sp1/sp2 registers,
 *      streaming is_tail KQT, scalar-constant tree vmasks, and non-pipelined
 *      cold is_tail V loop).
 *   2. LeanAttention longest-tile-first scheduling for tree verify:
 *      sg = is_tree ? (smx - 1 - sg_raw) : sg_raw launches the 4 slow is_tail
 *      WGs at wgid 0..3 (t=0) instead of at the end of the grid, hiding tail
 *      latency behind interior splits while preserving per-WGP kh Q-L1 reuse
 *      and consecutive per-group L2 locality.
 *   3. Phase 1: overlaps K global loads with pg=0 Q loads/converts/barrier by
 *      deferring r11_g05_kload_sc until after lbarrier(); pairs consecutive
 *      q8_0 blocks (2p, 2p+1) across two independent WMMA accumulators (t0, t1)
 *      to eliminate back-to-back WMMA RAW stalls while keeping exact blk 0..7
 *      FMA order; eliminates ~190 v_mov 0 insts via persistent permlane16 dst.
 *   4. Phase 2: full_valid (c0 + 127 <= pos, 99.5% of splits) branchless fast
 *      path removes all 24 sequential s_and_saveexec/s_cbranch_execz blocks
 *      around expf (and branchless masking on !full_valid), unlocking full
 *      v_dual_* and v_exp_f32 pipelining (~3x faster Phase 2); Wave 0 writes
 *      all 48 (M, L) pairs directly via 3 coalesced global_store_b64s overlapped
 *      with Phase 3, freeing ml[48][2] from LDS and removing 48 scalar stores.
 *   5. Phase 3: constant += 17408 B pointer stepping on full_valid (-192 insts),
 *      4-op 16-lane integer vsc max (-112 insts, exact for all 65,536 f16 bits),
 *      top-of-loop p_tile[next] publish, pre-wait pg=0 p_tile prefetch, and
 *      per-pg acc pinning (prevents 48 VGPRs of WMMA temps leaking across
 *      lbarrier() and lets vr_nxt fire immediately).
 *   6. Phase 4: full 16-row conflict-free o_stg[16][256] unioned LDS transpose
 *      cuts Phase-4 barriers from 11 -> 5 with 100% 32-lane utilization;
 *      internal group-packed 256B-aligned part layout (48*(smx+ng_max)*258 <
 *      48*(smx+ng_max)*260 floats per kh slab) eliminates partial-line RFOs,
 *      TLB scatter across 41 MB, and false sharing while making L1/L2 msum
 *      contiguous sequential streams; specialized msum_l1 (n<=8) + compact
 *      nounroll r11_g05_final cuts code size from ~75-88 KiB -> ~26 KiB (fits
 *      inside the 32 KiB L0 I-cache).
 * ========================================================================== */

#define R11_G05_TR_DEPTH(d03, d47, j) ((((j) < 4u ? (d03) : (d47)) >> (8u * ((j) & 3u))) & 255u)
#define R11_G05_DIV6(i) (((i) * 43u) >> 8u) /* exact i / 6 for all i in 0..47 */
#define R11_G05_PIN_ACC8(a) \
  __asm__ volatile("" : "+v"((a)[0]), "+v"((a)[1]), "+v"((a)[2]), "+v"((a)[3]), \
                        "+v"((a)[4]), "+v"((a)[5]), "+v"((a)[6]), "+v"((a)[7]))

typedef unsigned int r11_g05_u2 __attribute__((ext_vector_type(2)));
typedef float r11_g05_f2 __attribute__((ext_vector_type(2)));

/* Exactly 16,384 B LDS union -> 4 WGs/WGP (128 concurrent WGs on 32 WGPs).
 * Disjoint phase lifetimes (every transition guarded by workgroup barrier):
 *   Phase 1:        q.{qh, ql}                      (16,384 B)
 *   Phases 2 & 3:   ph23 (p_tile, v_sc, vs, red_*)  (13,824 B)
 *   Phase 4 store:  o_stg[16][256]                  (16,384 B)
 *   Phase 4 merge:  merge.{ob, last, wg_last/done}  ( 8,392 B) */
typedef union {
  struct {
    f16 qh[16u * 256u]; /*  8,192 B */
    f16 ql[16u * 256u]; /*  8,192 B */
  } q;                  /* 16,384 B */
  struct {
    float p_tile[2][768]; /*  6,144 B: double-buffered 3x16x16 P tiles */
    u32   v_sc[8][128];   /*  4,096 B: 8 waves x 512 B int8 V^T scratch */
    float vs[8][16];      /*    512 B: 8 waves x 16 normalized V scales */
    float red_mx[3][128]; /*  1,536 B: Phase-2 max reductions (3 pg) */
    float red_ls[3][128]; /*  1,536 B: Phase-2 sum reductions (3 pg) */
  } ph23;                 /* 13,824 B */
  float o_stg[16][256];   /* 16,384 B: Phase-4 full-16-row coalesced transpose */
  struct {
    float ob[8][256];     /*  8,192 B: Phase-4 r11_g05_final staging */
    u32   last[48];       /*    192 B: fallback per-pair merge flags */
    u32   wg_last;        /*      4 B: uniform Level-1 merge flag */
    u32   wg_done;        /*      4 B: uniform Level-2 merge flag */
  } merge;                /*  8,392 B */
} r11_g05_lds_t;

static LDS r11_g05_lds_t r11_g05_lds __attribute__((aligned(32)));
_Static_assert(sizeof(r11_g05_lds_t) == 16384u, "r11_g05_lds must be 16384 B for 4 WGs/WGP");

/* Bijective bank-conflict-free Q LDS offset (in f16) for row r in 0..15, chunk c in 0..31 (8 f16 each). */
static inline __attribute__((always_inline)) u32 r11_g05_q_off(u32 r, u32 c) {
  return r * 256u + 8u * (c ^ ((c >> 3u) & 1u) ^ r);
}

/* Bijective bank-conflict-free p_tile offset (in float) for row r in 0..15, f32x4 chunk c in 0..3. */
static inline __attribute__((always_inline)) u32 r11_g05_p_off(u32 r, u32 c) {
  return r * 16u + 4u * (c ^ ((r >> 1u) & 3u));
}

/* Load 16 query pairs for pair group pg (0..2) into XOR-swizzled qh/ql LDS (T=8, grp=6 -> all 48 live). */
static inline __attribute__((always_inline)) void r11_g05_load_q_lds(
    const G float *q, u32 pg, u32 kh, u32 qst) {
  const u32 t = tid(), r = t >> 4u, t16 = t & 15u, d0 = 16u * t16;
  const u32 i = 16u * pg + r;
  const u32 j = R11_G05_DIV6(i);
  const u32 h = kh * 6u + (i - 6u * j);
  const G float *qp = (const G float *)((const G u8 *)q + (unsigned long)j * qst) + (unsigned long)h * 256ul + d0;
  f32x4 x[4];
#pragma unroll
  for (u32 u = 0u; u < 4u; u++) x[u] = ((const G f32x4 *)qp)[u];
  h8 hi0, hi1, lo0, lo1;
#pragma unroll
  for (u32 u = 0u; u < 4u; u++) {
#pragma unroll
    for (u32 e = 0u; e < 4u; e++) {
      const float v = x[u][e] * 0.0625f;
      const f16 hv = (f16)v, lv = (f16)(v - (float)hv);
      if (u < 2u) { hi0[4u * u + e] = hv; lo0[4u * u + e] = lv; }
      else        { hi1[4u * (u - 2u) + e] = hv; lo1[4u * (u - 2u) + e] = lv; }
    }
  }
  const u32 o0 = r11_g05_q_off(r, 2u * t16);
  const u32 o1 = r11_g05_q_off(r, 2u * t16 + 1u);
  LDSP f16 *qh = r11_g05_lds.q.qh, *ql = r11_g05_lds.q.ql;
  *(LDSP h8 *)(qh + o0) = hi0;
  *(LDSP h8 *)(qh + o1) = hi1;
  *(LDSP h8 *)(ql + o0) = lo0;
  *(LDSP h8 *)(ql + o1) = lo1;
}

/* Convert hl==0 lanes' 8 fp16 key scales (ksc4) into per-half-wave register array ks_own[8].
 * Deferred until after pg=0 Q load + lbarrier() so all 9 K global loads overlap Q loading. */
static inline __attribute__((always_inline)) void r11_g05_kload_sc(
    u32x4 ksc4, u32 hl, int ks_own[8]) {
#pragma unroll
  for (u32 blk = 0u; blk < 8u; blk++) {
    const u32 kw = ksc4[blk >> 1u];
    const float my_ks = (float)__builtin_bit_cast(f16, (u16)((blk & 1u) ? (kw >> 16u) : kw));
    const int my_bits = F2I(my_ks);
    const int hi_bits = __builtin_amdgcn_permlanex16(my_bits, my_bits, (int)0xFEDCBA98u, (int)0x76543210u, false, false);
    ks_own[blk] = hl ? hi_bits : my_bits;
  }
}

/* Paired 2-accumulator S^T = K Q^T: interleaves blocks (2p, 2p+1) on independent WMMA
 * accumulators (t0, t1) to eliminate WMMA RAW stalls while accumulating sv[0..7] in exact
 * blk = 0, 1, ..., 7 FMA order. Reuses persistent kt0/kt1 for permlane16 to kill v_mov 0. */
static inline __attribute__((always_inline)) v8f r11_g05_kqt(
    const u32x4 kq[8], const int ks_own[8], const LDSP f16 *qh, const LDSP f16 *ql, u32 hl, u32 rl) {
  const v8f zero = (v8f)(0.0f);
  const u32 rbase = rl * 256u, rl_hl = rl ^ (2u * hl);
  float sv[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  int kt0 = 0, kt1 = 0;
#pragma unroll
  for (u32 p = 0u; p < 4u; p++) {
    const u32 b0 = 2u * p, b1 = 2u * p + 1u;
    const u32 cm0a = (4u * b0) ^ ((b0 >> 1u) & 1u), cm1a = cm0a ^ 1u;
    const u32 cm0b = (4u * b1) ^ ((b1 >> 1u) & 1u), cm1b = cm0b ^ 1u;
    const u32 o0a = rbase + 8u * (rl_hl ^ cm0a), o1a = rbase + 8u * (rl_hl ^ cm1a);
    const u32 o0b = rbase + 8u * (rl_hl ^ cm0b), o1b = rbase + 8u * (rl_hl ^ cm1b);

    const h8 qh0a = *(const LDSP h8 *)(qh + o0a);
    const h8 qh0b = *(const LDSP h8 *)(qh + o0b);
    const h8 ql0a = *(const LDSP h8 *)(ql + o0a);
    const h8 ql0b = *(const LDSP h8 *)(ql + o0b);
    const h8 kb0a = al_i8_to_h8(kq[b0][0], kq[b0][1]);
    const h8 kb0b = al_i8_to_h8(kq[b1][0], kq[b1][1]);

    v8f t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb0a, qh0a, zero);
    v8f t1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb0b, qh0b, zero);
    t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb0a, ql0a, t0);
    t1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb0b, ql0b, t1);

    const h8 qh1a = *(const LDSP h8 *)(qh + o1a);
    const h8 qh1b = *(const LDSP h8 *)(qh + o1b);
    const h8 ql1a = *(const LDSP h8 *)(ql + o1a);
    const h8 ql1b = *(const LDSP h8 *)(ql + o1b);
    const h8 kb1a = al_i8_to_h8(kq[b0][2], kq[b0][3]);
    const h8 kb1b = al_i8_to_h8(kq[b1][2], kq[b1][3]);

    t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb1a, qh1a, t0);
    t1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb1b, qh1b, t1);
    t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb1a, ql1a, t0);
    t1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb1b, ql1b, t1);

    const int ksb0 = ks_own[b0];
#pragma unroll
    for (u32 v = 0u; v < 8u; v += 2u) {
      const int s0 = (int)(0x11111111u * v);
      const int s1 = (int)(0x11111111u * (v + 1u));
      kt0 = __builtin_amdgcn_permlane16(kt0, ksb0, s0, s0, false, false);
      kt1 = __builtin_amdgcn_permlane16(kt1, ksb0, s1, s1, false, false);
      sv[v]      = __builtin_fmaf(t0[v],      I2F(kt0), sv[v]);
      sv[v + 1u] = __builtin_fmaf(t0[v + 1u], I2F(kt1), sv[v + 1u]);
    }
    R11_G05_PIN_ACC8(sv);

    const int ksb1 = ks_own[b1];
#pragma unroll
    for (u32 v = 0u; v < 8u; v += 2u) {
      const int s0 = (int)(0x11111111u * v);
      const int s1 = (int)(0x11111111u * (v + 1u));
      kt0 = __builtin_amdgcn_permlane16(kt0, ksb1, s0, s0, false, false);
      kt1 = __builtin_amdgcn_permlane16(kt1, ksb1, s1, s1, false, false);
      sv[v]      = __builtin_fmaf(t1[v],      I2F(kt0), sv[v]);
      sv[v + 1u] = __builtin_fmaf(t1[v + 1u], I2F(kt1), sv[v + 1u]);
    }
    R11_G05_PIN_ACC8(sv);
  }
  return (v8f){sv[0], sv[1], sv[2], sv[3], sv[4], sv[5], sv[6], sv[7]};
}

/* Low-VGPR streaming S^T = K Q^T for cold tree-verify is_tail splits (0 extra live K VGPRs). */
static inline __attribute__((always_inline)) v8f r11_g05_kqt_stream(
    const G kvt *kc, unsigned long row, u32 kh, const LDSP f16 *qh, const LDSP f16 *ql, u32 hl, u32 rl) {
  const G u8 *rec = (const G u8 *)kc + (row * 4ul + kh) * 272ul;
  const u32x4 ksc4 = hl == 0u ? *(const G u32x4 *)(rec + 256ul) : (u32x4)(0u);
  const v8f zero = (v8f)(0.0f);
  const u32 rbase = rl * 256u, rl_hl = rl ^ (2u * hl);
  float sv[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  int kt0 = 0;
#pragma nounroll
  for (u32 blk = 0u; blk < 8u; blk++) {
    const u32x4 kb = *(const G u32x4 *)(rec + 32ul * blk + 16ul * hl);
    const u32 cm0 = (4u * blk) ^ ((blk >> 1u) & 1u);
    const u32 cm1 = cm0 ^ 1u;
    const u32 o0 = rbase + 8u * (rl_hl ^ cm0);
    const u32 o1 = rbase + 8u * (rl_hl ^ cm1);
    const h8 qh0 = *(const LDSP h8 *)(qh + o0);
    const h8 ql0 = *(const LDSP h8 *)(ql + o0);
    const h8 qh1 = *(const LDSP h8 *)(qh + o1);
    const h8 ql1 = *(const LDSP h8 *)(ql + o1);
    const h8 kb0 = al_i8_to_h8(kb[0], kb[1]);
    const h8 kb1 = al_i8_to_h8(kb[2], kb[3]);
    v8f t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb0, qh0, zero);
    t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb0, ql0, t);
    t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb1, qh1, t);
    t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb1, ql1, t);
    const u32 kw = ksc4[blk >> 1u];
    const float my_ks = (float)__builtin_bit_cast(f16, (u16)((blk & 1u) ? (kw >> 16u) : kw));
    const int my_bits = F2I(my_ks);
    const int hi_bits = __builtin_amdgcn_permlanex16(my_bits, my_bits, (int)0xFEDCBA98u, (int)0x76543210u, false, false);
    const int ksb = hl ? hi_bits : my_bits;
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      const int sel = (int)(0x11111111u * v);
      kt0 = __builtin_amdgcn_permlane16(kt0, ksb, sel, sel, false, false);
      sv[v] = __builtin_fmaf(t[v], I2F(kt0), sv[v]);
    }
    R11_G05_PIN_ACC8(sv);
  }
  return (v8f){sv[0], sv[1], sv[2], sv[3], sv[4], sv[5], sv[6], sv[7]};
}

/* Publish one wave's 16-key (sp0, sp1, sp2) into XOR-swizzled p_tile[buf] (6 conflict-free b128 stores). */
static inline __attribute__((always_inline)) void r11_g05_pub_p(
    v8f sp0, v8f sp1, v8f sp2, LDSP float *p_tile, u32 o0, u32 o1) {
  *(LDSP f32x4 *)(p_tile + o0)        = (f32x4){sp0[0], sp0[1], sp0[2], sp0[3]};
  *(LDSP f32x4 *)(p_tile + o1)        = (f32x4){sp0[4], sp0[5], sp0[6], sp0[7]};
  *(LDSP f32x4 *)(p_tile + 256u + o0) = (f32x4){sp1[0], sp1[1], sp1[2], sp1[3]};
  *(LDSP f32x4 *)(p_tile + 256u + o1) = (f32x4){sp1[4], sp1[5], sp1[6], sp1[7]};
  *(LDSP f32x4 *)(p_tile + 512u + o0) = (f32x4){sp2[0], sp2[1], sp2[2], sp2[3]};
  *(LDSP f32x4 *)(p_tile + 512u + o1) = (f32x4){sp2[4], sp2[5], sp2[6], sp2[7]};
}

/* 1-pass bank-conflict-free int8 V^T transpose + 4-op 16-lane integer wave-exponent max.
 * Bitwise exact against wave_max_pos(hl==0 ? fabsf((float)vsck) : 0.0f) for all 65,536 f16 patterns. */
static inline __attribute__((always_inline)) void r11_g05_vprep(
    u32x4 vrk, f16 vsck, LDSP u32 *v_sc_w, LDSP float *vs_w, u32 hl, u32 rl, u32 sel1, u32 sel2,
    h8 *b0_out, h8 *b1_out, v8f *vs_out, float *w_sc_out) {
  const float vsc_f = (float)vsck;
  u32 vm = (hl == 0u && vsc_f == vsc_f) ? ((u32)F2I(vsc_f) & 0x7fffffffu) : 0u;
  u32 vn;
  vn = (u32)shl_i((int)vm, 8); vm = vm > vn ? vm : vn;
  vn = (u32)shl_i((int)vm, 4); vm = vm > vn ? vm : vn;
  vn = (u32)shl_i((int)vm, 2); vm = vm > vn ? vm : vn;
  vn = (u32)shl_i((int)vm, 1); vm = vm > vn ? vm : vn;
  const u32 vbits = (u32)__builtin_amdgcn_readlane((int)vm, 0) & 0x7f800000u;
  const float inv_w = vbits ? I2F((264u << 23) - vbits) : 0.0f;
  *w_sc_out = vbits ? I2F(vbits - (10u << 23)) : 0.0f;
  if (hl == 0u) vs_w[rl] = vsc_f * inv_w;

  const u32 g = rl >> 2u, s = rl & 3u;
#pragma unroll
  for (u32 q = 0u; q < 4u; q++) {
    const u32 x = vrk[q];
    const u32 y = (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0xB1, 0xF, 0xF, true);
    const u32 u = __builtin_amdgcn_perm(y, x, sel1);
    const u32 z = (u32)__builtin_amdgcn_update_dpp(0, (int)u, 0x4E, 0xF, 0xF, true);
    const u32 t = __builtin_amdgcn_perm(z, u, sel2);
    const u32 col_sw = 16u * hl + 4u * (q ^ hl) + s;
    v_sc_w[4u * col_sw + (g ^ (q & 2u))] = t;
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");

  const u32 kh_sw = (2u * hl) ^ ((rl >> 2u) & 2u);
  const r11_g05_u2 p0 = *(const LDSP r11_g05_u2 *)(v_sc_w + 4u * rl + kh_sw);
  const r11_g05_u2 p1 = *(const LDSP r11_g05_u2 *)(v_sc_w + 4u * (16u + (rl ^ 4u)) + kh_sw);
  *vs_out = *(const LDSP v8f *)(vs_w + 8u * hl);
  *b0_out = al_i8_to_h8(p0.x, p0.y);
  *b1_out = al_i8_to_h8(p1.x, p1.y);
}

/* Pipelined 3-pg PV accumulation with pre-fetched pg=0 p_tile and per-pg acc pinning
 * (reuses 16 VGPRs for t0/t1 instead of leaking 48 VGPRs across lbarrier()). */
static inline __attribute__((always_inline)) void r11_g05_pv_3pg(
    h8 b0, h8 b1, v8f vs, float w_sc, const LDSP float *p_tile, u32 o0, u32 o1,
    f32x4 pl0, f32x4 pl1, float acc[3][2][8]) {
  const v8f zero = (v8f)(0.0f);
#pragma unroll
  for (u32 pg = 0u; pg < 3u; pg++) {
    h8 ahs, als;
#pragma unroll
    for (u32 e = 0u; e < 4u; e++) {
      const float p0 = pl0[e] * vs[e];
      const f16 h0 = (f16)p0;
      ahs[e] = h0;
      als[e] = (f16)(p0 - (float)h0);
      const float p1 = pl1[e] * vs[4u + e];
      const f16 h1 = (f16)p1;
      ahs[4u + e] = h1;
      als[4u + e] = (f16)(p1 - (float)h1);
    }
    if (pg + 1u < 3u) {
      pl0 = *(const LDSP f32x4 *)(p_tile + 256u * (pg + 1u) + o0);
      pl1 = *(const LDSP f32x4 *)(p_tile + 256u * (pg + 1u) + o1);
    }
    v8f t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ahs, b0, zero);
    v8f t1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ahs, b1, zero);
    t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(als, b0, t0);
    t1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(als, b1, t1);
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      acc[pg][0][v] = __builtin_fmaf(t0[v], w_sc, acc[pg][0][v]);
      acc[pg][1][v] = __builtin_fmaf(t1[v], w_sc, acc[pg][1][v]);
    }
    R11_G05_PIN_ACC8(acc[pg][0]);
    R11_G05_PIN_ACC8(acc[pg][1]);
  }
}

/* Zero-VGPR scalar-constant vmask for tree-verify masked PV: bit v set iff (16*pg + 8*hl + v)/6 == jj. */
static inline __attribute__((always_inline)) u32 r11_g05_tree_vmask(u32 pg, u32 hl, u32 jj) {
  if (pg == 0u) {
    const u32 m0 = jj == 0u ? 0x3fu : (jj == 1u ? 0xc0u : 0u);
    const u32 m1 = jj == 1u ? 0x0fu : (jj == 2u ? 0xf0u : 0u);
    return hl ? m1 : m0;
  }
  if (pg == 1u) {
    const u32 m0 = jj == 2u ? 0x03u : (jj == 3u ? 0xfcu : 0u);
    const u32 m1 = jj == 4u ? 0x3fu : (jj == 5u ? 0xc0u : 0u);
    return hl ? m1 : m0;
  }
  const u32 m0 = jj == 5u ? 0x0fu : (jj == 6u ? 0xf0u : 0u);
  const u32 m1 = jj == 6u ? 0x03u : (jj == 7u ? 0xfcu : 0u);
  return hl ? m1 : m0;
}

/* Masked single-pg PV accumulation for cold tree-verify boundary tiles (c0 + 16*kt + 15 >= pos). */
static inline __attribute__((always_inline)) void r11_g05_pv_accum_masked(
    h8 b0, h8 b1, v8f vs, float w_sc, const LDSP float *p_pg, u32 o0, u32 o1, u32 vmask, float acc0[8], float acc1[8]) {
  const v8f zero = (v8f)(0.0f);
  const f32x4 pl0 = *(const LDSP f32x4 *)(p_pg + o0);
  const f32x4 pl1 = *(const LDSP f32x4 *)(p_pg + o1);
  h8 ahs, als;
#pragma unroll
  for (u32 e = 0u; e < 4u; e++) {
    const float p0 = pl0[e] * vs[e];
    const f16 h0 = (f16)p0;
    ahs[e] = h0;
    als[e] = (f16)(p0 - (float)h0);
    const float p1 = pl1[e] * vs[4u + e];
    const f16 h1 = (f16)p1;
    ahs[4u + e] = h1;
    als[4u + e] = (f16)(p1 - (float)h1);
  }
  v8f t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ahs, b0, zero);
  v8f t1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ahs, b1, zero);
  t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(als, b0, t0);
  t1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(als, b1, t1);
#pragma unroll
  for (u32 v = 0u; v < 8u; v++) {
    if (vmask & (1u << v)) {
      acc0[v] = __builtin_fmaf(t0[v], w_sc, acc0[v]);
      acc1[v] = __builtin_fmaf(t1[v], w_sc, acc1[v]);
    }
  }
}

/* Specialized Level-1 merge (n in 1..8) over contiguous 256B-aligned O_L0[0..n-1] (stride 256 f32)
 * and packed L1-hot ML_L0[t][pi] (stride 48 f32x2). Exact t = 0..n-1 reduction order. */
static inline __attribute__((always_inline)) void r11_g05_msum_l1(
    const G float *o_pi, const G r11_g05_f2 *ml_pi, u32 n, float *M2o, float *L2o, f32x4 *a0o, f32x4 *a1o) {
  const u32 l = lane();
  float mv = -__builtin_inff(), ll = 0.0f;
  if (l < n) {
    const r11_g05_f2 ml = ml_pi[(unsigned long)l * 48u];
    mv = ml.x;
    ll = ml.y;
  }
  const float M2 = al_wave_max(mv);
  const float wl = l < n ? __builtin_expf(mv - M2) : 0.0f;
  f32x4 a0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f}, a1 = a0;
  float L2 = 0.0f;
  if (__builtin_expect(n == 8u, 1)) {
    f32x4 x0[8], x1[8];
#pragma unroll
    for (u32 k = 0u; k < 8u; k++) {
      const G float *pt = o_pi + (unsigned long)k * 256ul + 4u * l;
      x0[k] = *(const G f32x4 *)pt;
      x1[k] = *(const G f32x4 *)(pt + 128u);
    }
#pragma unroll
    for (u32 k = 0u; k < 8u; k++) {
      const float wt = I2F(__builtin_amdgcn_readlane(F2I(wl), (int)k));
      L2 += I2F(__builtin_amdgcn_readlane(F2I(ll), (int)k)) * wt;
      a0 += x0[k] * wt;
      a1 += x1[k] * wt;
    }
  } else {
    const u32 nr = __builtin_amdgcn_readfirstlane(n);
    f32x4 x0[7], x1[7];
#pragma unroll
    for (u32 k = 0u; k < 7u; k++) {
      if (k < nr) {
        const G float *pt = o_pi + (unsigned long)k * 256ul + 4u * l;
        x0[k] = *(const G f32x4 *)pt;
        x1[k] = *(const G f32x4 *)(pt + 128u);
      }
    }
#pragma unroll
    for (u32 k = 0u; k < 7u; k++) {
      if (k < nr) {
        const float wt = I2F(__builtin_amdgcn_readlane(F2I(wl), (int)k));
        L2 += I2F(__builtin_amdgcn_readlane(F2I(ll), (int)k)) * wt;
        a0 += x0[k] * wt;
        a1 += x1[k] * wt;
      }
    }
  }
  *M2o = M2; *L2o = L2; *a0o = a0; *a1o = a1;
}

/* Chunk helper for Level-2 merge (arbitrary ng >= 1) over contiguous O_L1 (stride 256 f32). */
static inline __attribute__((always_inline)) void r11_g05_msum_chunk(
    const G float *o_pi, u32 t0, u32 nu, float wl, float ll, u32 l,
    float *L2p, f32x4 *a0p, f32x4 *a1p) {
  float L2 = *L2p;
  f32x4 a0 = *a0p, a1 = *a1p;
  u32 u = 0u;
  for (; u + 8u <= nu; u += 8u) {
    f32x4 x0[8], x1[8];
#pragma unroll
    for (u32 k = 0u; k < 8u; k++) {
      const G float *pt = o_pi + (unsigned long)(t0 + u + k) * 256ul + 4u * l;
      x0[k] = *(const G f32x4 *)pt;
      x1[k] = *(const G f32x4 *)(pt + 128u);
    }
#pragma unroll
    for (u32 k = 0u; k < 8u; k++) {
      const float wt = I2F(__builtin_amdgcn_readlane(F2I(wl), (int)(u + k)));
      L2 += I2F(__builtin_amdgcn_readlane(F2I(ll), (int)(u + k))) * wt;
      a0 += x0[k] * wt;
      a1 += x1[k] * wt;
    }
  }
  const u32 rem = __builtin_amdgcn_readfirstlane(nu - u);
  if (rem > 0u) {
    f32x4 x0[7], x1[7];
#pragma unroll
    for (u32 k = 0u; k < 7u; k++) {
      if (k < rem) {
        const G float *pt = o_pi + (unsigned long)(t0 + u + k) * 256ul + 4u * l;
        x0[k] = *(const G f32x4 *)pt;
        x1[k] = *(const G f32x4 *)(pt + 128u);
      }
    }
#pragma unroll
    for (u32 k = 0u; k < 7u; k++) {
      if (k < rem) {
        const float wt = I2F(__builtin_amdgcn_readlane(F2I(wl), (int)(u + k)));
        L2 += I2F(__builtin_amdgcn_readlane(F2I(ll), (int)(u + k))) * wt;
        a0 += x0[k] * wt;
        a1 += x1[k] * wt;
      }
    }
  }
  *L2p = L2; *a0p = a0; *a1p = a1;
}

/* Level-2 merge over contiguous O_L1[0..n-1] (stride 256 f32) and ML_L1[g][pi] (stride 48 f32x2). */
static inline __attribute__((always_inline)) void r11_g05_msum_l2(
    const G float *o_pi, const G r11_g05_f2 *ml_pi, u32 n, float *M2o, float *L2o, f32x4 *a0o, f32x4 *a1o) {
  const u32 l = lane();
  f32x4 a0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f}, a1 = a0;
  float L2 = 0.0f;
  if (__builtin_expect(n <= 32u, 1)) {
    float mv = -__builtin_inff(), ll = 0.0f;
    if (l < n) {
      const r11_g05_f2 ml = ml_pi[(unsigned long)l * 48u];
      mv = ml.x;
      ll = ml.y;
    }
    const float M2 = al_wave_max(mv);
    const float wl = l < n ? __builtin_expf(mv - M2) : 0.0f;
    r11_g05_msum_chunk(o_pi, 0u, n, wl, ll, l, &L2, &a0, &a1);
    *M2o = M2; *L2o = L2; *a0o = a0; *a1o = a1;
    return;
  }
  float mv = -__builtin_inff();
  for (u32 t = l; t < n; t += 32u) mv = __builtin_fmaxf(mv, ml_pi[(unsigned long)t * 48u].x);
  const float M2 = al_wave_max(mv);
  for (u32 t0 = 0u; t0 < n; t0 += 32u) {
    const u32 tl = t0 + l, nu = n - t0 < 32u ? n - t0 : 32u;
    float wl = 0.0f, ll = 0.0f;
    if (tl < n) {
      const r11_g05_f2 ml = ml_pi[(unsigned long)tl * 48u];
      wl = __builtin_expf(ml.x - M2);
      ll = ml.y;
    }
    r11_g05_msum_chunk(o_pi, t0, nu, wl, ll, l, &L2, &a0, &a1);
  }
  *M2o = M2; *L2o = L2; *a0o = a0; *a1o = a1;
}

/* Bitwise-exact compact al_final: #pragma nounroll keeps code size inside the 32 KiB L0 I-cache
 * with zero local arrays (0 private scratch). Runs on only 1 winning WG per kh per launch. */
static inline __attribute__((always_inline)) void r11_g05_final(
    f32x4 a0, f32x4 a1, float L2, G float *ox, G u8 *oqx, const G float *gxh,
    u32 blk0, u32 nbq, LDSP float *ob) {
  const u32 l = lane();
  *(LDSP f32x4 *)(ob + 4u * l) = a0 / L2;
  *(LDSP f32x4 *)(ob + 128u + 4u * l) = a1 / L2;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma nounroll
  for (u32 i = 0u; i < 8u; i++) {
    float o = ob[32u * i + l];
    if (gxh) o = o * sigmoidf(gxh[32u * i + l]);
    if (ox) ox[32u * i + l] = o;
    if (oqx) quant_wave(o, blk0 + i, oqx, nbq);
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

KERNEL ie_saluki_attn_long_mp_r11(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos,
                                  u32 n_head, u32 n_kv, u32 T, u32 npg, G u32 *count, G u32 *gcount, u32 gmax, G u8 *oq,
                                  const G float *gate, u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride,
                                  const G u8 *tr) {
  (void)n_head; (void)n_kv; (void)T; (void)npg; (void)pstride;
  const u32 n_h = 24u, n_k = 4u, grp = 6u;
  const int is_tree = __builtin_amdgcn_readfirstlane(tr != (const G u8 *)0);
  const u32 d03 = is_tree ? __builtin_amdgcn_readfirstlane(*(const G u32 *)(tr + TR_DEP)) : 0x03020100u;
  const u32 d47 = is_tree ? __builtin_amdgcn_readfirstlane(*(const G u32 *)(tr + TR_DEP + 4u)) : 0x07060504u;
  const u32 kend = __builtin_amdgcn_readfirstlane(pos + (is_tree ? (u32)tr[368] : 8u));
  const u32 smx = __builtin_amdgcn_readfirstlane(al_ns(kend - 1u));
  const u32 sg_raw = __builtin_amdgcn_readfirstlane(wgid() >> 2u);
  if (sg_raw >= smx) return; /* handles 253..256 -> 257 grid bump cleanly */
  const u32 kh = __builtin_amdgcn_readfirstlane(wgid() & 3u);
  /* Longest-tile-first for tree verify: launch slow is_tail WGs at wgid 0..3 (t=0). */
  const u32 sg = is_tree ? (smx - 1u - sg_raw) : sg_raw;
  const u32 c0 = __builtin_amdgcn_readfirstlane(sg * AL_CH);
  const u32 w = __builtin_amdgcn_readfirstlane(tid() >> 5u);
  const u32 l = lane(), hl = l >> 4u, rl = l & 15u;
  const int is_tail = __builtin_amdgcn_readfirstlane(is_tree && (c0 + AL_CH > pos));
  const int full_valid = __builtin_amdgcn_readfirstlane(c0 + 127u <= pos);
  const v8f zero = (v8f)(0.0f);
  const u32 kb = c0 + 16u * w;
  const int kon = kb < kend;

  /* Internal group-packed 256B-aligned part layout (48*(smx+ng_max)*258 < 48*(smx+ng_max)*260 <= 2*pst per kh). */
  const u32 ng_max = __builtin_amdgcn_readfirstlane(al_ng(smx));
  const unsigned long slab_f = (unsigned long)(smx + ng_max) * 12480ul; /* 48 * 260 floats = 49,920 B (256B-aligned) */
  G float *const o0_base = part + (unsigned long)kh * slab_f;
  G float *const o1_base = o0_base + (unsigned long)smx * 12288ul;       /* 48 * 256 floats per split */
  G r11_g05_f2 *const ml0_base = (G r11_g05_f2 *)(o1_base + (unsigned long)ng_max * 12288ul);
  G r11_g05_f2 *const ml1_base = ml0_base + (unsigned long)smx * 48ul;

  /* ---- Phase 1: Async K/Q overlap + paired 2-acc KQT (hot) / zero-scratch streaming KQT (cold is_tail) ---- */
  v8f sp0 = zero, sp1 = zero, sp2 = zero;
  if (__builtin_expect(!is_tail, 1)) {
    u32x4 kq[8];
    u32x4 ksc4 = (u32x4)(0u);
    int ks_own[8];
    if (kon) {
      const unsigned long row = full_valid ? (unsigned long)(kb + rl)
                                           : (unsigned long)(kb + rl < kend ? kb + rl : kend - 1u);
      al_q8_kload(kc, row, n_k, kh, hl, kq, &ksc4);
    }
    /* Load pg=0 Q into LDS while all 9 K global loads remain in flight across lbarrier(). */
    r11_g05_load_q_lds(q, 0u, kh, qst);
    lbarrier();
    if (kon) {
      r11_g05_kload_sc(ksc4, hl, ks_own);
      sp0 = r11_g05_kqt(kq, ks_own, r11_g05_lds.q.qh, r11_g05_lds.q.ql, hl, rl);
    }
    lbarrier();

    r11_g05_load_q_lds(q, 1u, kh, qst);
    lbarrier();
    if (kon) sp1 = r11_g05_kqt(kq, ks_own, r11_g05_lds.q.qh, r11_g05_lds.q.ql, hl, rl);
    lbarrier();

    r11_g05_load_q_lds(q, 2u, kh, qst);
    lbarrier();
    if (kon) sp2 = r11_g05_kqt(kq, ks_own, r11_g05_lds.q.qh, r11_g05_lds.q.ql, hl, rl);
    lbarrier();
  } else {
    const int k_int = kon && (kb + 15u < pos);
#pragma nounroll
    for (u32 pg = 0u; pg < 3u; pg++) {
      r11_g05_load_q_lds(q, pg, kh, qst);
      lbarrier();
      if (kon) {
        const LDSP f16 *qh = r11_g05_lds.q.qh, *ql = r11_g05_lds.q.ql;
        v8f res = zero;
        if (k_int) {
          const unsigned long row = kb + rl < kend ? kb + rl : kend - 1u;
          res = r11_g05_kqt_stream(kc, row, kh, qh, ql, hl, rl);
        } else {
          const u32 ci = 16u * pg + rl, cj = R11_G05_DIV6(ci);
          const u32 jlo = R11_G05_DIV6(16u * pg), jhi = R11_G05_DIV6(16u * pg + 15u);
#pragma nounroll
          for (u32 jj = jlo; jj <= jhi; jj++) {
            const unsigned long trow = al_q8_trow(tr, jj, kb + rl < kend ? kb + rl : kend - 1u, pos);
            const v8f sj = r11_g05_kqt_stream(kc, trow, kh, qh, ql, hl, rl);
            if (cj == jj) res = sj;
          }
        }
        if (pg == 0u) sp0 = res;
        else if (pg == 1u) sp1 = res;
        else sp2 = res;
      }
      lbarrier();
    }
  }

  /* Prefetch V tile 0 ahead of Phase 2 on !is_tail (and keep base pointers for full_valid stepping). */
  u32x4 vr0 = (u32x4)(0u);
  f16 vsc0 = (f16)0.0f;
  const G u8 *vp_cur = (const G u8 *)0;
  const G u8 *vs_cur = (const G u8 *)0;
  if (__builtin_expect(!is_tail, 1)) {
    const unsigned long row0 = full_valid ? (unsigned long)(c0 + rl)
                                          : (unsigned long)(c0 + rl < kend ? c0 + rl : kend - 1u);
    const G u8 *rec0 = (const G u8 *)vc + (row0 * n_k + kh) * 272ul;
    vp_cur = rec0 + 32ul * w + 16ul * hl;
    vs_cur = rec0 + 256ul + 2ul * w;
    vr0 = *(const G u32x4 *)vp_cur;
    vsc0 = *(const G f16 *)vs_cur;
  }

  /* ---- Phase 2: 2-barrier branchless fused softmax + direct Wave-0 coalesced ML_L0 global write ---- */
  const u32 p_o0 = r11_g05_p_off(rl, 2u * hl);
  const u32 p_o1 = r11_g05_p_off(rl, 2u * hl + 1u);
  float M_w0_0 = 0.0f, M_w0_1 = 0.0f, M_w0_2 = 0.0f;
  u32 cpj0 = 0u, cpj1 = 0u, cpj2 = 0u;

  if (__builtin_expect(full_valid, 1)) {
    /* 99.5% interior splits: every key c0..c0+127 <= pos <= cpj -> zero masking, zero expf branches! */
#pragma unroll
    for (u32 pg = 0u; pg < 3u; pg++) {
      const v8f spv = (pg == 0u) ? sp0 : ((pg == 1u) ? sp1 : sp2);
      float mx = -__builtin_inff();
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) mx = __builtin_fmaxf(mx, spv[v]);
      mx = __builtin_fmaxf(mx, I2F(xrow_i(F2I(mx))));
      if (hl == 0u) r11_g05_lds.ph23.red_mx[pg][16u * w + rl] = mx;
    }
    lbarrier();

#pragma unroll
    for (u32 pg = 0u; pg < 3u; pg++) {
      float M = r11_g05_lds.ph23.red_mx[pg][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) M = __builtin_fmaxf(M, r11_g05_lds.ph23.red_mx[pg][16u * x + rl]);
      if (pg == 0u) M_w0_0 = M;
      else if (pg == 1u) M_w0_1 = M;
      else M_w0_2 = M;
      const v8f spv = (pg == 0u) ? sp0 : ((pg == 1u) ? sp1 : sp2);
      v8f phl;
      float ls = 0.0f;
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const float p = __builtin_expf(spv[v] - M);
        ls += p;
        const f16 hv = (f16)p;
        const f16 lv = (f16)(p - (float)hv);
        phl[v] = (float)hv + (float)lv;
      }
      if (pg == 0u) sp0 = phl;
      else if (pg == 1u) sp1 = phl;
      else sp2 = phl;
      ls += I2F(xrow_i(F2I(ls)));
      if (hl == 0u) r11_g05_lds.ph23.red_ls[pg][16u * w + rl] = ls;
    }
  } else {
    /* Boundary/tail split (0.5%): masked scores, branchless expf (0 s_cbranch_execz!). */
    cpj0 = pos + R11_G05_TR_DEPTH(d03, d47, R11_G05_DIV6(rl));
    cpj1 = pos + R11_G05_TR_DEPTH(d03, d47, R11_G05_DIV6(16u + rl));
    cpj2 = pos + R11_G05_TR_DEPTH(d03, d47, R11_G05_DIV6(32u + rl));
#pragma unroll
    for (u32 pg = 0u; pg < 3u; pg++) {
      const u32 cpj = (pg == 0u) ? cpj0 : ((pg == 1u) ? cpj1 : cpj2);
      v8f spv = (pg == 0u) ? sp0 : ((pg == 1u) ? sp1 : sp2);
      float mx = -__builtin_inff();
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const float sv = (kb + 8u * hl + v <= cpj) ? spv[v] : -__builtin_inff();
        spv[v] = sv;
        mx = __builtin_fmaxf(mx, sv);
      }
      if (pg == 0u) sp0 = spv;
      else if (pg == 1u) sp1 = spv;
      else sp2 = spv;
      mx = __builtin_fmaxf(mx, I2F(xrow_i(F2I(mx))));
      if (hl == 0u) r11_g05_lds.ph23.red_mx[pg][16u * w + rl] = mx;
    }
    lbarrier();

#pragma unroll
    for (u32 pg = 0u; pg < 3u; pg++) {
      const u32 cpj = (pg == 0u) ? cpj0 : ((pg == 1u) ? cpj1 : cpj2);
      float M = r11_g05_lds.ph23.red_mx[pg][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) M = __builtin_fmaxf(M, r11_g05_lds.ph23.red_mx[pg][16u * x + rl]);
      if (pg == 0u) M_w0_0 = M;
      else if (pg == 1u) M_w0_1 = M;
      else M_w0_2 = M;
      const v8f spv = (pg == 0u) ? sp0 : ((pg == 1u) ? sp1 : sp2);
      v8f phl;
      float ls = 0.0f;
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const float e = __builtin_expf(spv[v] - M);
        const float p = (kb + 8u * hl + v <= cpj) ? e : 0.0f;
        ls += p;
        const f16 hv = (f16)p;
        const f16 lv = (f16)(p - (float)hv);
        phl[v] = (float)hv + (float)lv;
      }
      if (pg == 0u) sp0 = phl;
      else if (pg == 1u) sp1 = phl;
      else sp2 = phl;
      ls += I2F(xrow_i(F2I(ls)));
      if (hl == 0u) r11_g05_lds.ph23.red_ls[pg][16u * w + rl] = ls;
    }
  }
  if (w == 0u) r11_g05_pub_p(sp0, sp1, sp2, r11_g05_lds.ph23.p_tile[0], p_o0, p_o1);
  lbarrier();

  /* Wave 0 writes all 48 (M, L) pairs directly to contiguous ML_L0[sg][0..47] (overlapped with Phase 3). */
  if (w == 0u && hl == 0u) {
    G r11_g05_f2 *const ml_sg = ml0_base + (unsigned long)sg * 48ul;
#pragma unroll
    for (u32 pg = 0u; pg < 3u; pg++) {
      float L = r11_g05_lds.ph23.red_ls[pg][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) L += r11_g05_lds.ph23.red_ls[pg][16u * x + rl];
      const float M = (pg == 0u) ? M_w0_0 : ((pg == 1u) ? M_w0_1 : M_w0_2);
      const u32 cpj = (pg == 0u) ? cpj0 : ((pg == 1u) ? cpj1 : cpj2);
      if (full_valid || c0 <= cpj) ml_sg[16u * pg + rl] = (r11_g05_f2){M, L};
    }
  }

  /* ---- Phase 3: Double-buffered p_tile + fast pointer stepping + 1-pass V^T + per-pg pinned PV ---- */
  const u32 sel1 = (rl & 1u) ? 0x03070105u : 0x06020400u;
  const u32 sel2 = (rl & 2u) ? 0x03020706u : 0x05040100u;
  LDSP u32 *v_sc_w = r11_g05_lds.ph23.v_sc[w];
  LDSP float *vs_w = r11_g05_lds.ph23.vs[w];
  float acc[3][2][8];
#pragma unroll
  for (u32 pg = 0u; pg < 3u; pg++) {
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      acc[pg][0][v] = 0.0f;
      acc[pg][1][v] = 0.0f;
    }
  }

  if (__builtin_expect(!is_tail, 1)) {
    __asm__ volatile("" : "+v"(vr0), "+v"(vsc0));
    if (__builtin_expect(full_valid, 1)) {
      /* 99.5% interior splits: all 8 kt tiles valid, constant += 17408 B V pointer stepping. */
#pragma nounroll
      for (u32 kt = 0u; kt < 8u; kt++) {
        const int has_nxt = kt < 7u;
        u32x4 vr_nxt = (u32x4)(0u);
        f16 vsc_nxt = (f16)0.0f;
        if (has_nxt) {
          vp_cur += 17408ul; /* 16 rows * 4 kv_heads * 272 B */
          vs_cur += 17408ul;
          vr_nxt = *(const G u32x4 *)vp_cur;
          vsc_nxt = *(const G f16 *)vs_cur;
        }
        /* Publish next tile's P at TOP of kt so its 6 ds_store_b128s overlap vprep + pv_3pg! */
        if (has_nxt && w == kt + 1u)
          r11_g05_pub_p(sp0, sp1, sp2, r11_g05_lds.ph23.p_tile[(kt + 1u) & 1u], p_o0, p_o1);
        /* Pre-fetch pg=0 p_tile ahead of vprep's s_wait_dscnt 0x0 to hide LDS read latency. */
        const LDSP float *const pt_cur = r11_g05_lds.ph23.p_tile[kt & 1u];
        const f32x4 pl0_0 = *(const LDSP f32x4 *)(pt_cur + p_o0);
        const f32x4 pl1_0 = *(const LDSP f32x4 *)(pt_cur + p_o1);
        __builtin_amdgcn_sched_barrier(0);
        h8 b0, b1; v8f vs; float w_sc;
        r11_g05_vprep(vr0, vsc0, v_sc_w, vs_w, hl, rl, sel1, sel2, &b0, &b1, &vs, &w_sc);
        r11_g05_pv_3pg(b0, b1, vs, w_sc, pt_cur, p_o0, p_o1, pl0_0, pl1_0, acc);
        lbarrier();
        __builtin_amdgcn_sched_barrier(0);
        vr0 = vr_nxt; vsc0 = vsc_nxt;
        __asm__ volatile("" : "+v"(vr0), "+v"(vsc0));
      }
    } else {
      /* Non-tree tail split (c0 + 127 > pos, e.g. last split of tr == NULL). */
#pragma nounroll
      for (u32 kt = 0u; kt < 8u; kt++) {
        if (c0 + 16u * kt >= kend) break;
        const int has_nxt = (kt + 1u < 8u) && (c0 + 16u * (kt + 1u) < kend);
        u32x4 vr_nxt = (u32x4)(0u);
        f16 vsc_nxt = (f16)0.0f;
        if (has_nxt) {
          const unsigned long nrow = c0 + 16u * (kt + 1u) + rl < kend ? c0 + 16u * (kt + 1u) + rl : kend - 1u;
          const G u8 *nrec = (const G u8 *)vc + (nrow * n_k + kh) * 272ul;
          vr_nxt = *(const G u32x4 *)(nrec + 32ul * w + 16ul * hl);
          vsc_nxt = ((const G f16 *)(nrec + 256ul))[w];
        }
        if (has_nxt && w == kt + 1u)
          r11_g05_pub_p(sp0, sp1, sp2, r11_g05_lds.ph23.p_tile[(kt + 1u) & 1u], p_o0, p_o1);
        const LDSP float *const pt_cur = r11_g05_lds.ph23.p_tile[kt & 1u];
        const f32x4 pl0_0 = *(const LDSP f32x4 *)(pt_cur + p_o0);
        const f32x4 pl1_0 = *(const LDSP f32x4 *)(pt_cur + p_o1);
        __builtin_amdgcn_sched_barrier(0);
        h8 b0, b1; v8f vs; float w_sc;
        r11_g05_vprep(vr0, vsc0, v_sc_w, vs_w, hl, rl, sel1, sel2, &b0, &b1, &vs, &w_sc);
        r11_g05_pv_3pg(b0, b1, vs, w_sc, pt_cur, p_o0, p_o1, pl0_0, pl1_0, acc);
        lbarrier();
        __builtin_amdgcn_sched_barrier(0);
        vr0 = vr_nxt; vsc0 = vsc_nxt;
        __asm__ volatile("" : "+v"(vr0), "+v"(vsc0));
      }
    }
  } else {
    /* Cold tree-verify is_tail (1 split per launch, launched first at wgid 0..3):
     * No vr_nxt pipelining + scalar-constant vmask -> saves ~34 VGPRs (fixes v207 cliff!). */
#pragma nounroll
    for (u32 kt = 0u; kt < 8u; kt++) {
      if (c0 + 16u * kt >= kend) break;
      const int has_nxt = (kt + 1u < 8u) && (c0 + 16u * (kt + 1u) < kend);
      if (has_nxt && w == kt + 1u)
        r11_g05_pub_p(sp0, sp1, sp2, r11_g05_lds.ph23.p_tile[(kt + 1u) & 1u], p_o0, p_o1);
      const LDSP float *const pt_cur = r11_g05_lds.ph23.p_tile[kt & 1u];
      if (c0 + 16u * kt + 15u >= pos) {
#pragma nounroll
        for (u32 jj = 0u; jj < 8u; jj++) {
          const unsigned long key = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;
          const unsigned long trow = al_q8_trow(tr, jj, (u32)key, pos);
          const G u8 *trec = (const G u8 *)vc + (trow * n_k + kh) * 272ul;
          const u32x4 vrj = *(const G u32x4 *)(trec + 32ul * w + 16ul * hl);
          const f16 vsj = ((const G f16 *)(trec + 256ul))[w];
          h8 b0, b1; v8f vs; float w_sc;
          r11_g05_vprep(vrj, vsj, v_sc_w, vs_w, hl, rl, sel1, sel2, &b0, &b1, &vs, &w_sc);
#pragma unroll
          for (u32 pg = 0u; pg < 3u; pg++) {
            if (__builtin_amdgcn_readfirstlane((pg == 0u ? (jj <= 2u) : (pg == 1u ? (jj >= 2u && jj <= 5u) : (jj >= 5u))))) {
              const u32 vmask = r11_g05_tree_vmask(pg, hl, jj);
              r11_g05_pv_accum_masked(b0, b1, vs, w_sc, pt_cur + 256u * pg, p_o0, p_o1, vmask, acc[pg][0], acc[pg][1]);
            }
          }
          __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        }
      } else {
        const unsigned long nrow = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;
        const G u8 *nrec = (const G u8 *)vc + (nrow * n_k + kh) * 272ul;
        const u32x4 vrc = *(const G u32x4 *)(nrec + 32ul * w + 16ul * hl);
        const f16 vscc = ((const G f16 *)(nrec + 256ul))[w];
        const f32x4 pl0_0 = *(const LDSP f32x4 *)(pt_cur + p_o0);
        const f32x4 pl1_0 = *(const LDSP f32x4 *)(pt_cur + p_o1);
        h8 b0, b1; v8f vs; float w_sc;
        r11_g05_vprep(vrc, vscc, v_sc_w, vs_w, hl, rl, sel1, sel2, &b0, &b1, &vs, &w_sc);
        r11_g05_pv_3pg(b0, b1, vs, w_sc, pt_cur, p_o0, p_o1, pl0_0, pl1_0, acc);
      }
      lbarrier();
    }
  }

  /* ---- Phase 4: Full-16-row conflict-free o_stg[16][256] (5 barriers vs 11) + 256B-aligned O_L0 ---- */
  const u32 ns_min = __builtin_amdgcn_readfirstlane(al_ns(pos));
  const int uniform_merge = __builtin_amdgcn_readfirstlane(ns_min == smx);
  const u32 g_cur = __builtin_amdgcn_readfirstlane(sg >> 3u);
  const u32 t_cur = __builtin_amdgcn_readfirstlane(sg & 7u);
  const u32 g_rem = smx - 8u * g_cur;
  const u32 in_g_max = __builtin_amdgcn_readfirstlane(g_rem < 8u ? g_rem : 8u);
  G float *const o0_g = o0_base + (unsigned long)(8u * g_cur) * 12288ul;

  const u32 c_a = 32u * w + 16u * hl + rl;
  const u32 c_b = c_a ^ 16u;
#pragma unroll
  for (u32 pg = 0u; pg < 3u; pg++) {
    /* All 32 lanes active, 0 bank conflicts: hl=0 writes col 0..15 then 16..31; hl=1 writes 16..31 then 0..15.
     * Constant-indexed acc[pg][0/1][v] (no ternary pointer) guarantees 100% SROA and priv == 0 B. */
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      const float a0v = acc[pg][0][v], a1v = acc[pg][1][v];
      r11_g05_lds.o_stg[8u * hl + v][c_a] = hl ? a1v : a0v;
      r11_g05_lds.o_stg[8u * hl + v][c_b] = hl ? a0v : a1v;
    }
    lbarrier();
#pragma unroll
    for (u32 hb = 0u; hb < 2u; hb++) {
      const u32 row = 8u * hb + w;
      const u32 pi = 16u * pg + row;
      const u32 j = R11_G05_DIV6(pi);
      const u32 cpj = pos + R11_G05_TR_DEPTH(d03, d47, j);
      if (uniform_merge || c0 <= cpj) {
        G float *pp = o0_g + ((unsigned long)pi * in_g_max + t_cur) * 256ul;
        const f32x4 v0 = *(const LDSP f32x4 *)(r11_g05_lds.o_stg[row] + 4u * l);
        const f32x4 v1 = *(const LDSP f32x4 *)(r11_g05_lds.o_stg[row] + 128u + 4u * l);
        *(G f32x4 *)(pp + 4u * l) = v0;
        *(G f32x4 *)(pp + 128u + 4u * l) = v1;
      }
    }
    if (pg < 2u) lbarrier();
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();

  if (__builtin_expect(uniform_merge, 1)) {
    const u32 in_g = in_g_max;
    const unsigned long gi = (unsigned long)(kh * grp) * gmax + g_cur;
    if (tid() == 0u)
      r11_g05_lds.merge.wg_last =
          (__scoped_atomic_fetch_add(&gcount[gi], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == in_g);
    barrier();
    if (!r11_g05_lds.merge.wg_last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

    const G r11_g05_f2 *const ml0_g = ml0_base + (unsigned long)(8u * g_cur) * 48ul;
    G r11_g05_f2 *const ml1_g = ml1_base + (unsigned long)g_cur * 48ul;
    for (u32 pi = w; pi < 48u; pi += 8u) {
      const u32 j = R11_G05_DIV6(pi);
      const u32 h = kh * grp + (pi - 6u * j);
      float M2, L2; f32x4 a0, a1;
      r11_g05_msum_l1(o0_g + (unsigned long)pi * in_g * 256ul, ml0_g + pi, in_g, &M2, &L2, &a0, &a1);
      if (ng_max == 1u) {
        G float *ox = out ? (G float *)((G u8 *)out + (unsigned long)j * ost) + (unsigned long)h * 256ul : (G float *)0;
        G u8 *oqx = oq ? oq + (unsigned long)j * oqst : oq;
        const G float *gxh = gate ? (const G float *)((const G u8 *)gate + (unsigned long)j * gst) + (unsigned long)h * gstride : gate;
        r11_g05_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_h * 8u, r11_g05_lds.merge.ob[w]);
      } else {
        G float *gp = o1_base + ((unsigned long)pi * ng_max + g_cur) * 256ul;
        if (l == 0u) ml1_g[pi] = (r11_g05_f2){M2, L2};
        *(G f32x4 *)(gp + 4u * l) = a0;
        *(G f32x4 *)(gp + 128u + 4u * l) = a1;
      }
    }
    if (tid() == 0u) gcount[gi] = 0u;
    if (ng_max == 1u) return;

    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (tid() == 0u)
      r11_g05_lds.merge.wg_done =
          (__scoped_atomic_fetch_add(&count[kh * grp], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == ng_max);
    barrier();
    if (!r11_g05_lds.merge.wg_done) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

    for (u32 pi = w; pi < 48u; pi += 8u) {
      const u32 j = R11_G05_DIV6(pi);
      const u32 h = kh * grp + (pi - 6u * j);
      G float *ox = out ? (G float *)((G u8 *)out + (unsigned long)j * ost) + (unsigned long)h * 256ul : (G float *)0;
      G u8 *oqx = oq ? oq + (unsigned long)j * oqst : oq;
      const G float *gxh = gate ? (const G float *)((const G u8 *)gate + (unsigned long)j * gst) + (unsigned long)h * gstride : gate;
      float M2, L2; f32x4 a0, a1;
      r11_g05_msum_l2(o1_base + (unsigned long)pi * ng_max * 256ul, ml1_base + pi, ng_max, &M2, &L2, &a0, &a1);
      r11_g05_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_h * 8u, r11_g05_lds.merge.ob[w]);
    }
    if (tid() == 0u) count[kh * grp] = 0u;
    return;
  }

  /* Boundary-straddling fallback path (ns_min != smx, ~6% of steps). */
  if (w < 3u && hl == 0u) {
    const u32 ci = 16u * w + rl;
    const u32 cj = R11_G05_DIV6(ci);
    const u32 cpj = pos + R11_G05_TR_DEPTH(d03, d47, cj);
    u32 last = 0u;
    if (c0 <= cpj) {
      const u32 ns = al_ns(cpj);
      const u32 in_g = ns - g_cur * AL_G < AL_G ? ns - g_cur * AL_G : AL_G;
      const unsigned long gi = ((unsigned long)cj * n_h + kh * grp + (ci - 6u * cj)) * gmax + g_cur;
      last = __scoped_atomic_fetch_add(&gcount[gi], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == in_g;
    }
    r11_g05_lds.merge.last[ci] = last;
  }
  barrier();
  const G r11_g05_f2 *const ml0_g = ml0_base + (unsigned long)(8u * g_cur) * 48ul;
  G r11_g05_f2 *const ml1_g = ml1_base + (unsigned long)g_cur * 48ul;
  for (u32 pi = w; pi < 48u; pi += 8u) {
    if (!r11_g05_lds.merge.last[pi]) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const u32 j = R11_G05_DIV6(pi);
    const u32 h = kh * grp + (pi - 6u * j);
    const u32 ns = al_ns(pos + R11_G05_TR_DEPTH(d03, d47, j)), ng = al_ng(ns);
    const u32 in_g = ns - g_cur * AL_G < AL_G ? ns - g_cur * AL_G : AL_G;
    G float *ox = out ? (G float *)((G u8 *)out + (unsigned long)j * ost) + (unsigned long)h * 256ul : (G float *)0;
    G u8 *oqx = oq ? oq + (unsigned long)j * oqst : oq;
    const G float *gxh = gate ? (const G float *)((const G u8 *)gate + (unsigned long)j * gst) + (unsigned long)h * gstride : gate;
    float M2, L2; f32x4 a0, a1;
    r11_g05_msum_l1(o0_g + (unsigned long)pi * in_g_max * 256ul, ml0_g + pi, in_g, &M2, &L2, &a0, &a1);
    if (l == 0u) gcount[((unsigned long)j * n_h + h) * gmax + g_cur] = 0u;
    if (ng == 1u) {
      r11_g05_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_h * 8u, r11_g05_lds.merge.ob[w]);
      continue;
    }
    G float *gp = o1_base + ((unsigned long)pi * ng_max + g_cur) * 256ul;
    if (l == 0u) ml1_g[pi] = (r11_g05_f2){M2, L2};
    *(G f32x4 *)(gp + 4u * l) = a0;
    *(G f32x4 *)(gp + 128u + 4u * l) = a1;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    u32 done = 0u;
    if (l == 0u) done = __scoped_atomic_fetch_add(&count[j * n_h + h], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == ng;
    if (!__builtin_amdgcn_readfirstlane(done)) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    r11_g05_msum_l2(o1_base + (unsigned long)pi * ng_max * 256ul, ml1_base + pi, ng, &M2, &L2, &a0, &a1);
    if (l == 0u) count[j * n_h + h] = 0u;
    r11_g05_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_h * 8u, r11_g05_lds.merge.ob[w]);
  }
}

#undef R11_G05_TR_DEPTH
#undef R11_G05_DIV6
#undef R11_G05_PIN_ACC8
