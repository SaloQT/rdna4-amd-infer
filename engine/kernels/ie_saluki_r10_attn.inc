/* ============================================================================
 * Round 10: Bitwise-exact Q8 Long-Attention & Prefill Attention Kernels
 *           for Saluki 27B on RX 9070 XT (gfx1201, wave32, hd=256, n_q=24, n_kv=4)
 *
 * Included at the end of kernels/ie_saluki_kernels.c (after R8 & R9 .inc files).
 * All local types, helpers, LDS symbols, and macros are prefixed r10_attn_ /
 * R10_ATTN_ and all macros are #undef'd at EOF.
 *
 * Kernels exported (all 100% bitwise identical to R8/base):
 *   1. ie_saluki_attn_long_mp_r10     (T=8 decode/verify Q8 long attention)
 *   2. ie_saluki_attn_long_mp_tr_r10  (T=8 tree-verify Q8 long attention)
 *      - 16,384 B unioned LDS (halved from 31,936 B -> 4 WGs/WGP = 128 WGs,
 *        2x occupancy, 0 spills, ~145-165 VGPRs <= 192).
 *      - XOR-swizzled bank-conflict-free Q LDS (0 padding) + register-only
 *        permlanex16/permlane16 key-scale broadcast (0 LDS, 0 readlane).
 *      - 2-barrier fused Phase-2 softmax (was 6) + in-register p_hl[3] +
 *        double-buffered XOR-swizzled 16-key p_tile[2][3][256] (0 conflicts).
 *      - 1-pass bank-conflict-free int8 V^T transpose via dpp(0xB1/0x4E) +
 *        v_perm_b32 + v_sc[8][128] (4 -> 1 s_wait_dscnt, 32 b16 -> 4 b32).
 *      - Zero-extra-VGPR pg=0..2 p_tile prefetch overlap in PV.
 *      - Phase-4 coalesced f32x4 part writes via o_stg[8][256] (144 -> 18
 *        stores, 0 RFOs) + single-atomic (48 -> 1) batched-fence (18 -> 2)
 *        merge when al_ns(pos) == al_ns(kend - 1) + pipelined r10_attn_msum
 *        remainder & single-load M/L + preloaded gxh[8] in r10_attn_final.
 *      - Tree-verify is_tail keeps interior waves (kb + 15 < pos) on 1-pass
 *        K load + permlane16 and interior kt tiles pipelined on 1-pass vprep.
 *   3. ie_saluki_attn_pl_r10          (Prefill Q8 attention, 32 KiB LDS)
 *      - Causal Wave-2 min+max WG pairing for ntile == 32 (T=497..512, exact
 *        64c + 33 blocks/CU vs 64c + 64 while retaining 32x per-KV-head L2/MALL
 *        sharing) and descending longest-tile-first for all other ntile.
 * ========================================================================== */

#define R10_ATTN_TR_DEPTH(d03, d47, j) ((((j) < 4u ? (d03) : (d47)) >> (8u * ((j) & 3u))) & 255u)
#define R10_ATTN_PIN_ACC8(a) \
  __asm__ volatile("" : "+v"((a)[0]), "+v"((a)[1]), "+v"((a)[2]), "+v"((a)[3]), \
                        "+v"((a)[4]), "+v"((a)[5]), "+v"((a)[6]), "+v"((a)[7]))
#define R10_ATTN_PL_ROWB 272u

typedef unsigned int r10_attn_u2 __attribute__((ext_vector_type(2)));
typedef float r10_attn_f2 __attribute__((ext_vector_type(2)));

/* Exactly 16,384 B LDS union -> 4 WGs/WGP (128 concurrent WGs on 32 WGPs).
 * Disjoint phase lifetimes:
 *   Phase 1:       q.{qh, ql}                          (16,384 B)
 *   Phases 2 & 3:  post.u.ph23 + post.ml               (14,208 B)
 *   Phase 4:       post.u.o_stg + post.{ml,last,wg_*}  ( 8,776 B) */
typedef union {
  struct {
    f16 qh[16u * 256u]; /*  8,192 B */
    f16 ql[16u * 256u]; /*  8,192 B */
  } q;                  /* 16,384 B */
  struct {
    union {
      struct {
        float p_tile[2][768];    /*  6,144 B: double-buffered 3x16x16 P tiles */
        u32   v_sc[8][128];      /*  4,096 B: 8 waves x 512 B int8 V^T scratch */
        float vs[8][16];         /*    512 B: 8 waves x 16 normalized V scales */
        float red_mx[3][128];    /*  1,536 B: Phase-2 max reductions (3 pg) */
        float red_ls[3][128];    /*  1,536 B: Phase-2 sum reductions (3 pg) */
      } ph23;                    /* 13,824 B */
      float o_stg[8][256];       /*  8,192 B: Phase-4 coalesced store & al_final */
    } u;                         /* 13,824 B */
    float ml[48][2];             /*    384 B: per-pair (M, L), Phase 2 -> Phase 4 */
    u32   last[48];              /*    192 B: fallback per-pair merge flags */
    u32   wg_last;               /*      4 B: uniform Level-1 merge flag */
    u32   wg_done;               /*      4 B: uniform Level-2 merge flag */
  } post;                        /* 14,408 B <= 16,384 B */
} r10_attn_mp_lds_t;

static LDS r10_attn_mp_lds_t r10_attn_mp_lds __attribute__((aligned(32)));
_Static_assert(sizeof(r10_attn_mp_lds_t) == 16384u, "r10_attn_mp_lds must be 16384 B for 4 WGs/WGP");

/* Bijective bank-conflict-free Q LDS offset (in f16) for row r in 0..15, chunk c in 0..31 (8 f16 each).
 * Both strided-2 writer (r10_attn_load_q_lds) and fixed-c reader (r10_attn_kqt) hit all 8 4-bank groups. */
static inline __attribute__((always_inline)) u32 r10_attn_q_off(u32 r, u32 c) {
  return r * 256u + 8u * (c ^ ((c >> 3u) & 1u) ^ r);
}

/* Bijective bank-conflict-free p_tile offset (in float) for row r in 0..15, f32x4 chunk c in 0..3.
 * Lanes r=0..7 and r=8..15 each hit all 8 4-bank groups (0 bank conflicts on b128 store & load). */
static inline __attribute__((always_inline)) u32 r10_attn_p_off(u32 r, u32 c) {
  return r * 16u + 4u * (c ^ ((r >> 1u) & 3u));
}

/* Load 16 query pairs for pair group pg (0..2) into XOR-swizzled qh/ql LDS (T=8, grp=6 -> all 48 live). */
static inline __attribute__((always_inline)) void r10_attn_load_q_lds(
    const G float *q, u32 pg, u32 kh, u32 qst) {
  const u32 t = tid(), r = t >> 4u, t16 = t & 15u, d0 = 16u * t16;
  const u32 i = 16u * pg + r;
  const u32 j = i / 6u, h = kh * 6u + i % 6u;
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
  const u32 o0 = r10_attn_q_off(r, 2u * t16);
  const u32 o1 = r10_attn_q_off(r, 2u * t16 + 1u);
  LDSP f16 *qh = r10_attn_mp_lds.q.qh, *ql = r10_attn_mp_lds.q.ql;
  *(LDSP h8 *)(qh + o0) = hi0;
  *(LDSP h8 *)(qh + o1) = hi1;
  *(LDSP h8 *)(ql + o0) = lo0;
  *(LDSP h8 *)(ql + o1) = lo1;
}

/* Convert hl==0 lanes' 8 fp16 key scales (ksc4) into per-half-wave register array ks_own[8]:
 * Row 0 (hl=0) lanes 0..7 hold ks(kb + 0..7); Row 1 (hl=1) lanes 16..23 hold ks(kb + 8..15).
 * Must be called with all 32 lanes active (wave-uniform). */
static inline __attribute__((always_inline)) void r10_attn_kload_sc(
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

/* S^T = K Q^T using register-resident kq[8] & ks_own[8] (permlane16 broadcast, 0 LDS scales, 0 readlane). */
static inline __attribute__((always_inline)) v8f r10_attn_kqt(
    const u32x4 kq[8], const int ks_own[8], const LDSP f16 *qh, const LDSP f16 *ql, u32 hl, u32 rl) {
  const v8f zero = (v8f)(0.0f);
  const u32 rbase = rl * 256u, rl_hl = rl ^ (2u * hl);
  float sv[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (u32 blk = 0u; blk < 8u; blk++) {
    const u32 cm0 = (4u * blk) ^ ((blk >> 1u) & 1u);
    const u32 cm1 = cm0 ^ 1u;
    const u32 o0 = rbase + 8u * (rl_hl ^ cm0);
    const u32 o1 = rbase + 8u * (rl_hl ^ cm1);
    const h8 qh0 = *(const LDSP h8 *)(qh + o0);
    const h8 ql0 = *(const LDSP h8 *)(ql + o0);
    const h8 qh1 = *(const LDSP h8 *)(qh + o1);
    const h8 ql1 = *(const LDSP h8 *)(ql + o1);
    const h8 kb0 = al_i8_to_h8(kq[blk][0], kq[blk][1]);
    const h8 kb1 = al_i8_to_h8(kq[blk][2], kq[blk][3]);
    v8f t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb0, qh0, zero);
    t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb0, ql0, t);
    t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb1, qh1, t);
    t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb1, ql1, t);
    const int ksb = ks_own[blk];
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      const int sel = (int)(0x11111111u * v);
      const float ks_v = I2F(__builtin_amdgcn_permlane16(0, ksb, sel, sel, false, false));
      sv[v] = __builtin_fmaf(t[v], ks_v, sv[v]);
    }
    R10_ATTN_PIN_ACC8(sv);
  }
  return (v8f){sv[0], sv[1], sv[2], sv[3], sv[4], sv[5], sv[6], sv[7]};
}

/* Streaming S^T = K Q^T for tree-verify boundary waves (kb + 15 >= pos): 0 extra VGPRs, permlane16 scales. */
static inline __attribute__((always_inline)) v8f r10_attn_kqt_stream(
    const G kvt *kc, unsigned long row, u32 kh, const LDSP f16 *qh, const LDSP f16 *ql, u32 hl, u32 rl) {
  const G u8 *rec = (const G u8 *)kc + (row * 4ul + kh) * 272ul;
  const u32x4 ksc4 = hl == 0u ? *(const G u32x4 *)(rec + 256ul) : (u32x4)(0u);
  const v8f zero = (v8f)(0.0f);
  const u32 rbase = rl * 256u, rl_hl = rl ^ (2u * hl);
  float sv[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
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
      const float ks_v = I2F(__builtin_amdgcn_permlane16(0, ksb, sel, sel, false, false));
      sv[v] = __builtin_fmaf(t[v], ks_v, sv[v]);
    }
    R10_ATTN_PIN_ACC8(sv);
  }
  return (v8f){sv[0], sv[1], sv[2], sv[3], sv[4], sv[5], sv[6], sv[7]};
}

/* Publish one wave's 16-key p_hl[3] slice into XOR-swizzled p_tile[buf] (6 bank-conflict-free b128 stores). */
static inline __attribute__((always_inline)) void r10_attn_pub_p(
    const v8f sp[3], LDSP float *p_tile, u32 o0, u32 o1) {
#pragma unroll
  for (u32 pg = 0u; pg < 3u; pg++) {
    *(LDSP f32x4 *)(p_tile + 256u * pg + o0) = (f32x4){sp[pg][0], sp[pg][1], sp[pg][2], sp[pg][3]};
    *(LDSP f32x4 *)(p_tile + 256u * pg + o1) = (f32x4){sp[pg][4], sp[pg][5], sp[pg][6], sp[pg][7]};
  }
}

/* 1-pass bank-conflict-free int8 V^T transpose + wave-exponent normalization (1 s_wait_dscnt vs 4 in R8).
 * Produces (b0, b1, vs, w_sc) bitwise identical to mp_q8_vprep for all 32 lanes. */
static inline __attribute__((always_inline)) void r10_attn_vprep(
    u32x4 vrk, f16 vsck, LDSP u32 *v_sc_w, LDSP float *vs_w, u32 hl, u32 rl, u32 sel1, u32 sel2,
    h8 *b0_out, h8 *b1_out, v8f *vs_out, float *w_sc_out) {
  const float vsm = wave_max_pos(hl == 0u ? __builtin_fabsf((float)vsck) : 0.0f);
  const u32 vbits = F2I(vsm) & 0x7f800000u;
  const float inv_w = vbits ? I2F((264u << 23) - vbits) : 0.0f;
  *w_sc_out = vbits ? I2F(vbits - (10u << 23)) : 0.0f;
  if (hl == 0u) vs_w[rl] = (float)vsck * inv_w;

  /* In-register 4x4 byte transpose across 4-lane key quads (g=rl>>2, s=rl&3), then 4 b32 stores
   * hitting all 32 distinct LDS banks (0 bank conflicts). */
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

  /* Two bank-conflict-free b64 loads + broadcast vs load, then unpack int8 -> exact fp16. */
  const u32 kh_sw = (2u * hl) ^ ((rl >> 2u) & 2u);
  const r10_attn_u2 p0 = *(const LDSP r10_attn_u2 *)(v_sc_w + 4u * rl + kh_sw);
  const r10_attn_u2 p1 = *(const LDSP r10_attn_u2 *)(v_sc_w + 4u * (16u + (rl ^ 4u)) + kh_sw);
  *vs_out = *(const LDSP v8f *)(vs_w + 8u * hl);
  *b0_out = al_i8_to_h8(p0.x, p0.y);
  *b1_out = al_i8_to_h8(p1.x, p1.y);
}

/* Pipelined 3-pg PV accumulation (zero extra VGPRs: next pg's p_tile loads issue after ahs/als consume pl0/pl1). */
static inline __attribute__((always_inline)) void r10_attn_pv_3pg(
    h8 b0, h8 b1, v8f vs, float w_sc, const LDSP float *p_tile, u32 o0, u32 o1, v8f acc[3][2]) {
  const v8f zero = (v8f)(0.0f);
  f32x4 pl0 = *(const LDSP f32x4 *)(p_tile + o0);
  f32x4 pl1 = *(const LDSP f32x4 *)(p_tile + o1);
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
  }
}

/* Masked single-pg PV accumulation for tree-verify boundary tiles (c0 + 16*kt + 15 >= pos). */
static inline __attribute__((always_inline)) void r10_attn_pv_accum_masked(
    h8 b0, h8 b1, v8f vs, float w_sc, const LDSP float *p_pg, u32 o0, u32 o1, u32 vmask, v8f *acc0, v8f *acc1) {
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
      (*acc0)[v] = __builtin_fmaf(t0[v], w_sc, (*acc0)[v]);
      (*acc1)[v] = __builtin_fmaf(t1[v], w_sc, (*acc1)[v]);
    }
  }
}

/* Pipelined al_msum chunk: issues all remainder (1..7) loads in flight before summing in exact t order. */
static inline __attribute__((always_inline)) void r10_attn_msum_chunk(
    const G float *ph, u32 t0, u32 nu, float wl, float ll, u32 l,
    float *L2p, f32x4 *a0p, f32x4 *a1p) {
  float L2 = *L2p;
  f32x4 a0 = *a0p, a1 = *a1p;
  u32 u = 0u;
  for (; u + 8u <= nu; u += 8u) {
    f32x4 x0[8], x1[8];
#pragma unroll
    for (u32 k = 0u; k < 8u; k++) {
      const G float *pt = ph + (unsigned long)(t0 + u + k) * AL_PSZ + 4u + 4u * l;
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
        const G float *pt = ph + (unsigned long)(t0 + u + k) * AL_PSZ + 4u + 4u * l;
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

/* Bitwise-exact al_msum with single 64-bit (M, L) load when n <= 32 and pipelined remainder loads. */
static inline __attribute__((always_inline)) void r10_attn_msum(
    const G float *ph, u32 n, float *M2o, float *L2o, f32x4 *a0o, f32x4 *a1o) {
  const u32 l = lane();
  f32x4 a0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f}, a1 = a0;
  float L2 = 0.0f;
  if (__builtin_expect(n <= 32u, 1)) {
    float mv = -__builtin_inff(), ll = 0.0f;
    if (l < n) {
      const r10_attn_f2 ml = *(const G r10_attn_f2 *)(ph + (unsigned long)l * AL_PSZ);
      mv = ml.x;
      ll = ml.y;
    }
    const float M2 = al_wave_max(mv);
    const float wl = l < n ? __builtin_expf(mv - M2) : 0.0f;
    r10_attn_msum_chunk(ph, 0u, n, wl, ll, l, &L2, &a0, &a1);
    *M2o = M2; *L2o = L2; *a0o = a0; *a1o = a1;
    return;
  }
  float mv = -__builtin_inff();
  for (u32 t = l; t < n; t += 32u) mv = __builtin_fmaxf(mv, ph[(unsigned long)t * AL_PSZ]);
  const float M2 = al_wave_max(mv);
  for (u32 t0 = 0u; t0 < n; t0 += 32u) {
    const u32 tl = t0 + l, nu = n - t0 < 32u ? n - t0 : 32u;
    float wl = 0.0f, ll = 0.0f;
    if (tl < n) {
      const r10_attn_f2 ml = *(const G r10_attn_f2 *)(ph + (unsigned long)tl * AL_PSZ);
      wl = __builtin_expf(ml.x - M2);
      ll = ml.y;
    }
    r10_attn_msum_chunk(ph, t0, nu, wl, ll, l, &L2, &a0, &a1);
  }
  *M2o = M2; *L2o = L2; *a0o = a0; *a1o = a1;
}

/* Bitwise-exact al_final: preloads all 8 gate elements ahead of a0/L2 and LDS transpose. */
static inline __attribute__((always_inline)) void r10_attn_final(
    f32x4 a0, f32x4 a1, float L2, G float *ox, G u8 *oqx, const G float *gxh,
    u32 blk0, u32 nbq, LDSP float *ob) {
  const u32 l = lane();
  float gv[8];
  if (gxh) {
#pragma unroll
    for (u32 i = 0u; i < 8u; i++) gv[i] = gxh[32u * i + l];
  }
  *(LDSP f32x4 *)(ob + 4u * l) = a0 / L2;
  *(LDSP f32x4 *)(ob + 128u + 4u * l) = a1 / L2;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 i = 0u; i < 8u; i++) {
    float o = ob[32u * i + l];
    if (gxh) o = o * sigmoidf(gv[i]);
    if (ox) ox[32u * i + l] = o;
    if (oqx) quant_wave(o, blk0 + i, oqx, nbq);
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

/* Shared body for ie_saluki_attn_long_mp_r10 (is_tree=0) and ie_saluki_attn_long_mp_tr_r10 (is_tree=1). */
static inline __attribute__((always_inline)) void r10_attn_mp_body(
    const int is_tree,
    const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos,
    G u32 *count, G u32 *gcount, u32 gmax, G u8 *oq, const G float *gate, u32 gstride,
    u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride, const G u8 *tr) {
  const u32 n_head = 24u, n_kv = 4u, T = 8u, grp = 6u;
  const u32 d03 = is_tree ? __builtin_amdgcn_readfirstlane(*(const G u32 *)(tr + TR_DEP)) : 0u;
  const u32 d47 = is_tree ? __builtin_amdgcn_readfirstlane(*(const G u32 *)(tr + TR_DEP + 4u)) : 0u;
  const u32 kh = __builtin_amdgcn_readfirstlane(wgid() & 3u);
  const u32 sg = __builtin_amdgcn_readfirstlane(wgid() >> 2u);
  const u32 w = __builtin_amdgcn_readfirstlane(tid() >> 5u);
  const u32 l = lane(), hl = l >> 4u, rl = l & 15u;
  const u32 kend = __builtin_amdgcn_readfirstlane(pos + (is_tree ? (u32)tr[368] : 8u));
  const u32 c0 = __builtin_amdgcn_readfirstlane(sg * AL_CH);
  if (c0 >= kend) return;
  const int is_tail = is_tree ? __builtin_amdgcn_readfirstlane(c0 + AL_CH > pos) : 0;
  const v8f zero = (v8f)(0.0f);
  const unsigned long pst = pstride;
  const u32 kb = c0 + 16u * w;
  const int kon = kb < kend;

  /* ---- Phase 1: Single-pass K load + XOR-swizzled Q LDS + register-only key scales ---- */
  v8f sp[3] = {zero, zero, zero};
  if (__builtin_expect(!is_tail, 1)) {
    u32x4 kq[8];
    int ks_own[8];
    if (kon) {
      const unsigned long row = kb + rl < kend ? kb + rl : kend - 1u;
      u32x4 ksc4 = (u32x4)(0u);
      al_q8_kload(kc, row, n_kv, kh, hl, kq, &ksc4);
      r10_attn_kload_sc(ksc4, hl, ks_own);
    }
#pragma unroll
    for (u32 pg = 0u; pg < 3u; pg++) {
      r10_attn_load_q_lds(q, pg, kh, qst);
      lbarrier();
      if (kon) sp[pg] = r10_attn_kqt(kq, ks_own, r10_attn_mp_lds.q.qh, r10_attn_mp_lds.q.ql, hl, rl);
      lbarrier();
    }
  } else {
    /* Tree-verify tail split: interior waves (kb + 15 < pos) STILL reuse a single K load across pg=0..2! */
    const int k_int = kon && (kb + 15u < pos);
    u32x4 kq[8];
    int ks_own[8];
    if (k_int) {
      const unsigned long row = kb + rl < kend ? kb + rl : kend - 1u;
      u32x4 ksc4 = (u32x4)(0u);
      al_q8_kload(kc, row, n_kv, kh, hl, kq, &ksc4);
      r10_attn_kload_sc(ksc4, hl, ks_own);
    }
#pragma nounroll
    for (u32 pg = 0u; pg < 3u; pg++) {
      r10_attn_load_q_lds(q, pg, kh, qst);
      lbarrier();
      if (kon) {
        const LDSP f16 *qh = r10_attn_mp_lds.q.qh, *ql = r10_attn_mp_lds.q.ql;
        if (k_int) {
          sp[pg] = r10_attn_kqt(kq, ks_own, qh, ql, hl, rl);
        } else {
          const u32 ci = 16u * pg + rl, cj = ci / grp;
          const u32 jlo = 16u * pg / grp, jhi = (16u * pg + 15u) / grp;
          v8f stmp = zero;
#pragma nounroll
          for (u32 jj = jlo; jj <= jhi; jj++) {
            const unsigned long trow = al_q8_trow(tr, jj, kb + rl < kend ? kb + rl : kend - 1u, pos);
            const v8f sj = r10_attn_kqt_stream(kc, trow, kh, qh, ql, hl, rl);
            if (cj == jj) stmp = sj;
          }
          sp[pg] = stmp;
        }
      }
      lbarrier();
    }
  }

  /* Issue V tile-0 prefetch ahead of Phase 2 whenever tile 0 is interior (all non-tree & most tail splits). */
  u32x4 vr0 = (u32x4)(0u);
  f16 vsc0 = (f16)0.0f;
  if (!is_tail || c0 + 15u < pos) {
    const unsigned long row0 = c0 + rl < kend ? c0 + rl : kend - 1u;
    const G u8 *rec0 = (const G u8 *)vc + (row0 * n_kv + kh) * 272ul;
    vr0 = *(const G u32x4 *)(rec0 + 32ul * w + 16ul * hl);
    vsc0 = ((const G f16 *)(rec0 + 256ul))[w];
  }

  /* ---- Phase 2: 2-barrier fused softmax across pg=0..2 + in-place p_hl + wave-0 p_tile[0] publish ---- */
  u32 cpj_pg[3];
#pragma unroll
  for (u32 pg = 0u; pg < 3u; pg++) {
    const u32 ci = 16u * pg + rl, cj = ci / grp;
    const u32 cpj = pos + (is_tree ? R10_ATTN_TR_DEPTH(d03, d47, cj) : cj);
    cpj_pg[pg] = cpj;
    float mx = -__builtin_inff();
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      const float sv = (kb + 8u * hl + v <= cpj) ? sp[pg][v] : -__builtin_inff();
      sp[pg][v] = sv;
      mx = __builtin_fmaxf(mx, sv);
    }
    mx = __builtin_fmaxf(mx, I2F(xrow_i(F2I(mx))));
    if (hl == 0u) r10_attn_mp_lds.post.u.ph23.red_mx[pg][16u * w + rl] = mx;
  }
  lbarrier();

  const u32 p_o0 = r10_attn_p_off(rl, 2u * hl);
  const u32 p_o1 = r10_attn_p_off(rl, 2u * hl + 1u);
#pragma unroll
  for (u32 pg = 0u; pg < 3u; pg++) {
    const u32 cpj = cpj_pg[pg];
    float M = r10_attn_mp_lds.post.u.ph23.red_mx[pg][rl];
#pragma unroll
    for (u32 x = 1u; x < 8u; x++) M = __builtin_fmaxf(M, r10_attn_mp_lds.post.u.ph23.red_mx[pg][16u * x + rl]);
    if (w == 0u && hl == 0u) r10_attn_mp_lds.post.ml[16u * pg + rl][0] = M;
    float ls = 0.0f;
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      const float p = (kb + 8u * hl + v <= cpj) ? __builtin_expf(sp[pg][v] - M) : 0.0f;
      ls += p;
      const f16 hv = (f16)p;
      const f16 lv = (f16)(p - (float)hv);
      sp[pg][v] = (float)hv + (float)lv;
    }
    ls += I2F(xrow_i(F2I(ls)));
    if (hl == 0u) r10_attn_mp_lds.post.u.ph23.red_ls[pg][16u * w + rl] = ls;
  }
  if (w == 0u) r10_attn_pub_p(sp, r10_attn_mp_lds.post.u.ph23.p_tile[0], p_o0, p_o1);
  lbarrier();

  if (w == 0u && hl == 0u) {
#pragma unroll
    for (u32 pg = 0u; pg < 3u; pg++) {
      float L = r10_attn_mp_lds.post.u.ph23.red_ls[pg][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) L += r10_attn_mp_lds.post.u.ph23.red_ls[pg][16u * x + rl];
      r10_attn_mp_lds.post.ml[16u * pg + rl][1] = L;
    }
  }

  /* ---- Phase 3: Double-buffered p_tile + 1-pass int8 V^T + zero-extra-VGPR pipelined PV ---- */
  const u32 sel1 = (rl & 1u) ? 0x03070105u : 0x06020400u;
  const u32 sel2 = (rl & 2u) ? 0x03020706u : 0x05040100u;
  LDSP u32 *v_sc_w = r10_attn_mp_lds.post.u.ph23.v_sc[w];
  LDSP float *vs_w = r10_attn_mp_lds.post.u.ph23.vs[w];
  v8f acc[3][2] = {{zero, zero}, {zero, zero}, {zero, zero}};

  if (__builtin_expect(!is_tail, 1)) {
    __asm__ volatile("" : "+v"(vr0), "+v"(vsc0));
#pragma nounroll
    for (u32 kt = 0u; kt < 8u; kt++) {
      if (c0 + 16u * kt >= kend) break;
      const int has_nxt = (kt + 1u < 8u) && (c0 + 16u * (kt + 1u) < kend);
      u32x4 vr_nxt = (u32x4)(0u);
      f16 vsc_nxt = (f16)0.0f;
      if (has_nxt) {
        const unsigned long nrow = c0 + 16u * (kt + 1u) + rl < kend ? c0 + 16u * (kt + 1u) + rl : kend - 1u;
        const G u8 *nrec = (const G u8 *)vc + (nrow * n_kv + kh) * 272ul;
        vr_nxt = *(const G u32x4 *)(nrec + 32ul * w + 16ul * hl);
        vsc_nxt = ((const G f16 *)(nrec + 256ul))[w];
      }
      __builtin_amdgcn_sched_barrier(0);
      h8 b0, b1; v8f vs; float w_sc;
      r10_attn_vprep(vr0, vsc0, v_sc_w, vs_w, hl, rl, sel1, sel2, &b0, &b1, &vs, &w_sc);
      r10_attn_pv_3pg(b0, b1, vs, w_sc, r10_attn_mp_lds.post.u.ph23.p_tile[kt & 1u], p_o0, p_o1, acc);
      if (has_nxt && w == kt + 1u)
        r10_attn_pub_p(sp, r10_attn_mp_lds.post.u.ph23.p_tile[(kt + 1u) & 1u], p_o0, p_o1);
      /* lbarrier() waits on dscnt only (leaves vr_nxt in flight) and synchronizes p_tile & Phase-4 o_stg. */
      lbarrier();
      __builtin_amdgcn_sched_barrier(0);
      vr0 = vr_nxt; vsc0 = vsc_nxt;
      __asm__ volatile("" : "+v"(vr0), "+v"(vsc0));
    }
  } else {
#pragma nounroll
    for (u32 kt = 0u; kt < 8u; kt++) {
      if (c0 + 16u * kt >= kend) break;
      const int has_nxt = (kt + 1u < 8u) && (c0 + 16u * (kt + 1u) < kend);
      const int nxt_int = has_nxt && (c0 + 16u * (kt + 1u) + 15u < pos);
      u32x4 vr_nxt = (u32x4)(0u);
      f16 vsc_nxt = (f16)0.0f;
      if (nxt_int) {
        const unsigned long nrow = c0 + 16u * (kt + 1u) + rl < kend ? c0 + 16u * (kt + 1u) + rl : kend - 1u;
        const G u8 *nrec = (const G u8 *)vc + (nrow * n_kv + kh) * 272ul;
        vr_nxt = *(const G u32x4 *)(nrec + 32ul * w + 16ul * hl);
        vsc_nxt = ((const G f16 *)(nrec + 256ul))[w];
      }
      __builtin_amdgcn_sched_barrier(0);
      if (c0 + 16u * kt + 15u >= pos) {
#pragma nounroll
        for (u32 jj = 0u; jj < T; jj++) {
          const unsigned long key = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;
          const unsigned long trow = al_q8_trow(tr, jj, (u32)key, pos);
          const G u8 *trec = (const G u8 *)vc + (trow * n_kv + kh) * 272ul;
          const u32x4 vrj = *(const G u32x4 *)(trec + 32ul * w + 16ul * hl);
          const f16 vsj = ((const G f16 *)(trec + 256ul))[w];
          h8 b0, b1; v8f vs; float w_sc;
          r10_attn_vprep(vrj, vsj, v_sc_w, vs_w, hl, rl, sel1, sel2, &b0, &b1, &vs, &w_sc);
#pragma unroll
          for (u32 pg = 0u; pg < 3u; pg++) {
            if (16u * pg / grp <= jj && jj <= (16u * pg + 15u) / grp) {
              u32 vmask = 0u;
#pragma unroll
              for (u32 v = 0u; v < 8u; v++)
                if ((16u * pg + 8u * hl + v) / grp == jj) vmask |= 1u << v;
              r10_attn_pv_accum_masked(b0, b1, vs, w_sc, r10_attn_mp_lds.post.u.ph23.p_tile[kt & 1u] + 256u * pg,
                                       p_o0, p_o1, vmask, &acc[pg][0], &acc[pg][1]);
            }
          }
          __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        }
      } else {
        h8 b0, b1; v8f vs; float w_sc;
        r10_attn_vprep(vr0, vsc0, v_sc_w, vs_w, hl, rl, sel1, sel2, &b0, &b1, &vs, &w_sc);
        r10_attn_pv_3pg(b0, b1, vs, w_sc, r10_attn_mp_lds.post.u.ph23.p_tile[kt & 1u], p_o0, p_o1, acc);
      }
      if (has_nxt && w == kt + 1u)
        r10_attn_pub_p(sp, r10_attn_mp_lds.post.u.ph23.p_tile[(kt + 1u) & 1u], p_o0, p_o1);
      lbarrier();
      if (nxt_int) { vr0 = vr_nxt; vsc0 = vsc_nxt; }
    }
  }

  /* ---- Phase 4: Coalesced f32x4 part writes via o_stg[8][256] + single-atomic batched-fence merge ---- */
  const u32 ns_min = al_ns(pos);
  const u32 ns_max = al_ns(kend - 1u);
  const int uniform_merge = (ns_min == ns_max);
  const u32 ng_min = al_ng(ns_min);

#pragma unroll
  for (u32 pg = 0u; pg < 3u; pg++) {
#pragma unroll
    for (u32 hb = 0u; hb < 2u; hb++) {
      if (hl == hb) {
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) {
          r10_attn_mp_lds.post.u.o_stg[v][32u * w + rl] = acc[pg][0][v];
          r10_attn_mp_lds.post.u.o_stg[v][32u * w + 16u + rl] = acc[pg][1][v];
        }
      }
      lbarrier();
      const u32 pi = 16u * pg + 8u * hb + w;
      const u32 j = pi / grp, h = kh * grp + pi % grp;
      const u32 cpj = pos + (is_tree ? R10_ATTN_TR_DEPTH(d03, d47, j) : j);
      if (uniform_merge || c0 <= cpj) {
        const u32 ns = uniform_merge ? ns_min : al_ns(cpj);
        const u32 ng = uniform_merge ? ng_min : al_ng(ns);
        G float *pp = part + (unsigned long)j * pst + ((unsigned long)h * (ns + ng) + sg) * AL_PSZ;
        const f32x4 v0 = *(const LDSP f32x4 *)(r10_attn_mp_lds.post.u.o_stg[w] + 4u * l);
        const f32x4 v1 = *(const LDSP f32x4 *)(r10_attn_mp_lds.post.u.o_stg[w] + 128u + 4u * l);
        if (l == 0u) {
          pp[0] = r10_attn_mp_lds.post.ml[pi][0];
          pp[1] = r10_attn_mp_lds.post.ml[pi][1];
        }
        *(G f32x4 *)(pp + 4u + 4u * l) = v0;
        *(G f32x4 *)(pp + 132u + 4u * l) = v1;
      }
      if (pg < 2u || hb == 0u) lbarrier();
    }
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();

  if (__builtin_expect(uniform_merge, 1)) {
    /* Kernel-wide uniform fast path (~94% of steps): all 48 pairs share exact (ns, ng, g, in_g).
     * Reduces 48 -> 1 atomic per level and 18 -> 2 agent-scope cache invalidations per winning WG. */
    const u32 ns = ns_min, ng = ng_min, g = sg / AL_G;
    const u32 in_g = ns - g * AL_G < AL_G ? ns - g * AL_G : AL_G;
    const unsigned long gi = (unsigned long)(kh * grp) * gmax + g;
    if (tid() == 0u)
      r10_attn_mp_lds.post.wg_last =
          (__scoped_atomic_fetch_add(&gcount[gi], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == in_g);
    barrier();
    if (!r10_attn_mp_lds.post.wg_last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

    for (u32 pi = w; pi < 48u; pi += 8u) {
      const u32 j = pi / grp, h = kh * grp + pi % grp;
      G float *pb = part + (unsigned long)j * pst + (unsigned long)h * (ns + ng) * AL_PSZ;
      float M2, L2; f32x4 a0, a1;
      r10_attn_msum(pb + (unsigned long)g * AL_G * AL_PSZ, in_g, &M2, &L2, &a0, &a1);
      if (ng == 1u) {
        G float *ox = out ? (G float *)((G u8 *)out + (unsigned long)j * ost) + (unsigned long)h * 256ul : (G float *)0;
        G u8 *oqx = oq ? oq + (unsigned long)j * oqst : oq;
        const G float *gxh = gate ? (const G float *)((const G u8 *)gate + (unsigned long)j * gst) + (unsigned long)h * gstride : gate;
        r10_attn_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_head * 8u, r10_attn_mp_lds.post.u.o_stg[w]);
      } else {
        G float *gp = pb + (unsigned long)(ns + g) * AL_PSZ;
        if (l == 0u) { gp[0] = M2; gp[1] = L2; }
        *(G f32x4 *)(gp + 4u + 4u * l) = a0;
        *(G f32x4 *)(gp + 132u + 4u * l) = a1;
      }
    }
    if (tid() == 0u) gcount[gi] = 0u;
    if (ng == 1u) return;

    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (tid() == 0u)
      r10_attn_mp_lds.post.wg_done =
          (__scoped_atomic_fetch_add(&count[kh * grp], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == ng);
    barrier();
    if (!r10_attn_mp_lds.post.wg_done) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

    for (u32 pi = w; pi < 48u; pi += 8u) {
      const u32 j = pi / grp, h = kh * grp + pi % grp;
      G float *pb = part + (unsigned long)j * pst + (unsigned long)h * (ns + ng) * AL_PSZ;
      G float *ox = out ? (G float *)((G u8 *)out + (unsigned long)j * ost) + (unsigned long)h * 256ul : (G float *)0;
      G u8 *oqx = oq ? oq + (unsigned long)j * oqst : oq;
      const G float *gxh = gate ? (const G float *)((const G u8 *)gate + (unsigned long)j * gst) + (unsigned long)h * gstride : gate;
      float M2, L2; f32x4 a0, a1;
      r10_attn_msum(pb + (unsigned long)ns * AL_PSZ, ng, &M2, &L2, &a0, &a1);
      r10_attn_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_head * 8u, r10_attn_mp_lds.post.u.o_stg[w]);
    }
    if (tid() == 0u) count[kh * grp] = 0u;
    return;
  }

  /* Boundary-straddling fallback path (al_ns(pos) != al_ns(kend - 1), ~6% of steps). */
  if (w < 3u && hl == 0u) {
    const u32 ci = 16u * w + rl, cj = ci / grp;
    const u32 cpj = pos + (is_tree ? R10_ATTN_TR_DEPTH(d03, d47, cj) : cj);
    u32 last = 0u;
    if (c0 <= cpj) {
      const u32 ns = al_ns(cpj), g = sg / AL_G;
      const u32 in_g = ns - g * AL_G < AL_G ? ns - g * AL_G : AL_G;
      const unsigned long gi = ((unsigned long)cj * n_head + kh * grp + ci % grp) * gmax + g;
      last = __scoped_atomic_fetch_add(&gcount[gi], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == in_g;
    }
    r10_attn_mp_lds.post.last[ci] = last;
  }
  barrier();
  for (u32 pi = w; pi < 48u; pi += 8u) {
    if (!r10_attn_mp_lds.post.last[pi]) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const u32 j = pi / grp, h = kh * grp + pi % grp;
    const u32 ns = al_ns(pos + (is_tree ? R10_ATTN_TR_DEPTH(d03, d47, j) : j)), ng = al_ng(ns), g = sg / AL_G;
    const u32 in_g = ns - g * AL_G < AL_G ? ns - g * AL_G : AL_G;
    G float *pb = part + (unsigned long)j * pst + (unsigned long)h * (ns + ng) * AL_PSZ;
    G float *ox = out ? (G float *)((G u8 *)out + (unsigned long)j * ost) + (unsigned long)h * 256ul : (G float *)0;
    G u8 *oqx = oq ? oq + (unsigned long)j * oqst : oq;
    const G float *gxh = gate ? (const G float *)((const G u8 *)gate + (unsigned long)j * gst) + (unsigned long)h * gstride : gate;
    float M2, L2; f32x4 a0, a1;
    r10_attn_msum(pb + (unsigned long)g * AL_G * AL_PSZ, in_g, &M2, &L2, &a0, &a1);
    if (l == 0u) gcount[((unsigned long)j * n_head + h) * gmax + g] = 0u;
    if (ng == 1u) { r10_attn_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_head * 8u, r10_attn_mp_lds.post.u.o_stg[w]); continue; }
    G float *gp = pb + (unsigned long)(ns + g) * AL_PSZ;
    if (l == 0u) { gp[0] = M2; gp[1] = L2; }
    *(G f32x4 *)(gp + 4u + 4u * l) = a0;
    *(G f32x4 *)(gp + 132u + 4u * l) = a1;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    u32 done = 0u;
    if (l == 0u) done = __scoped_atomic_fetch_add(&count[j * n_head + h], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == ng;
    if (!__builtin_amdgcn_readfirstlane(done)) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    r10_attn_msum(pb + (unsigned long)ns * AL_PSZ, ng, &M2, &L2, &a0, &a1);
    if (l == 0u) count[j * n_head + h] = 0u;
    r10_attn_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_head * 8u, r10_attn_mp_lds.post.u.o_stg[w]);
  }
}

KERNEL ie_saluki_attn_long_mp_r10(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos,
                                  u32 n_head, u32 n_kv, u32 T, u32 npg, G u32 *count, G u32 *gcount, u32 gmax, G u8 *oq,
                                  const G float *gate, u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride,
                                  const G u8 *tr) {
  (void)n_head; (void)n_kv; (void)T; (void)npg; (void)tr;
  r10_attn_mp_body(0, q, kc, vc, part, out, pos, count, gcount, gmax, oq, gate, gstride, qst, ost, oqst, gst, pstride, (const G u8 *)0);
}

KERNEL ie_saluki_attn_long_mp_tr_r10(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos,
                                     u32 n_head, u32 n_kv, u32 T, u32 npg, G u32 *count, G u32 *gcount, u32 gmax, G u8 *oq,
                                     const G float *gate, u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride,
                                     const G u8 *tr) {
  (void)n_head; (void)n_kv; (void)T; (void)npg;
  r10_attn_mp_body(1, q, kc, vc, part, out, pos, count, gcount, gmax, oq, gate, gstride, qst, ost, oqst, gst, pstride, tr);
}

/* --- R10-3: ie_saluki_attn_pl_r10 (Causal Wave-2 min+max pairing + descending tile scheduling) --- */
__attribute__((amdgpu_waves_per_eu(4, 4)))
KERNEL ie_saluki_attn_pl_r10(const G float *q, const G kvt *kc, const G kvt *vc, G float *out, u32 pos0, u32 hd,
                             u32 n_head, u32 n_kv, const G float *gate, u32 gstride, u32 qs, u32 os, u32 gs, u32 T,
                             G u8 *oq, u32 oqs, const G f16 *q16) {
  if (hd != 256u || n_head != 24u || n_kv != 4u || T == 0u) return;
  const u32 ntile = (T + 15u) >> 4u, wg = wgid();
  u32 tile, kh;
  if (ntile == 32u) {
    /* T=497..512 (128 WGs = 2 full 64-WG waves): Wave 1 runs kh=0,1 tiles 0..31 ascending
     * (32x KV reuse in L2/MALL); Wave 2 runs kh=2,3 tiles 31..0 descending so every CU
     * pairs min+max causal length (64c + 33 blocks/CU vs 64c + 64 in R8). */
    if (wg >= 128u) return;
    if (wg < 64u) {
      kh = wg & 1u;
      tile = wg >> 1u;
    } else {
      const u32 u = wg - 64u;
      kh = 2u + (u & 1u);
      tile = 31u - (u >> 1u);
    }
  } else {
    /* All other T (including grid=257 cliff guard when ntile==64): longest-tile-first. */
    const u32 rtile = wg >> 2u;
    if (rtile >= ntile) return;
    kh = wg & 3u;
    tile = ntile - 1u - rtile;
  }
  const u32 t0 = 16u * tile;
  const u32 w = __builtin_amdgcn_readfirstlane(tid() >> 5u), l = lane(), hl = l >> 4u, rl = l & 15u;
  const u32 tlast = t0 + 15u < T ? t0 + 15u : T - 1u, kend = pos0 + tlast + 1u;
  const u32 nblk = (kend + 15u) >> 4u;
  if (w >= 6u) {
    const G u8 *kb_kh = (const G u8 *)kc + (unsigned long)kh * R10_ATTN_PL_ROWB;
    const G u8 *vb_kh = (const G u8 *)vc + (unsigned long)kh * R10_ATTN_PL_ROWB;
    const u32 pt = tid() - 192u;
    const r8_pl_poff po = r8_pl_make_poff(pt);
    const u32 nblk_full = kend >> 4u;
    r8_pl_slot ring[3];
    if (nblk_full >= 6u) {
      r8_pl_load_fast(&ring[0], kb_kh, vb_kh, 0u, po);
      r8_pl_load_fast(&ring[1], kb_kh, vb_kh, 1u, po);
      r8_pl_load_fast(&ring[2], kb_kh, vb_kh, 2u, po);
      r8_pl_publish(&ring[0], 0u, pt);
      r8_pl_load_fast(&ring[0], kb_kh, vb_kh, 3u, po);
      u32 k0 = 0u;
      for (; k0 + 6u < nblk_full; k0 += 3u) {
        lbarrier();
        r8_pl_publish(&ring[1], (k0 + 1u) & 1u, pt);
        r8_pl_load_fast(&ring[1], kb_kh, vb_kh, k0 + 4u, po);
        lbarrier();
        r8_pl_publish(&ring[2], (k0 + 2u) & 1u, pt);
        r8_pl_load_fast(&ring[2], kb_kh, vb_kh, k0 + 5u, po);
        lbarrier();
        r8_pl_publish(&ring[0], (k0 + 3u) & 1u, pt);
        r8_pl_load_fast(&ring[0], kb_kh, vb_kh, k0 + 6u, po);
      }
#pragma clang loop unroll(disable)
      for (; k0 < nblk; k0 += 3u) {
#pragma unroll
        for (u32 j = 0u; j < 3u; j++) {
          const u32 k = k0 + j;
          if (k < nblk) {
            lbarrier();
            if (k + 1u < nblk) {
              r8_pl_publish(&ring[(j + 1u) % 3u], (k + 1u) & 1u, pt);
              if (k + 4u < nblk) r8_pl_load(&ring[(j + 1u) % 3u], kb_kh, vb_kh, kend, k + 4u, pt, po);
            }
          }
        }
      }
    } else {
#pragma unroll
      for (u32 s = 0u; s < 3u; s++)
        if (s < nblk) r8_pl_load(&ring[s], kb_kh, vb_kh, kend, s, pt, po);
      r8_pl_publish(&ring[0], 0u, pt);
      if (3u < nblk) r8_pl_load(&ring[0], kb_kh, vb_kh, kend, 3u, pt, po);
#pragma clang loop unroll(disable)
      for (u32 k0 = 0u; k0 < nblk; k0 += 3u) {
#pragma unroll
        for (u32 j = 0u; j < 3u; j++) {
          const u32 k = k0 + j;
          if (k < nblk) {
            lbarrier();
            if (k + 1u < nblk) {
              r8_pl_publish(&ring[(j + 1u) % 3u], (k + 1u) & 1u, pt);
              if (k + 4u < nblk) r8_pl_load(&ring[(j + 1u) % 3u], kb_kh, vb_kh, kend, k + 4u, pt, po);
            }
          }
        }
      }
    }
    return;
  }
  const u32 h = kh * 6u + w;
  const u32 tq = t0 + rl < T ? t0 + rl : T - 1u;
  h8 qr_reg[16];
  r8_pl_load_qreg(qr_reg, q, q16, tq, h, qs, hl);

  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2u) & 1u) ^ (rl >> 3u);
  const v8f zero = (v8f)(0.0f);
  v8f o[16];
#pragma unroll
  for (u32 nt = 0u; nt < 16u; nt++) o[nt] = zero;
  float m = -__builtin_inff(), ls = 0.0f;
  const u32 tp = pos0 + t0 + rl;
  const u32 pos_t0 = pos0 + t0;

  for (u32 k = 0u; k < nblk; k++) {
    lbarrier();
    const u32 kb = 16u * k, bf = k & 1u;
    const u32 k_b0 = r8_pl_k_base_b(bf, rl, hl);
    v8f sa = zero, sb = zero;
    h8 k0 = r8_pl_ld_k(k_b0, 0u);
    h8 k1 = r8_pl_ld_k(k_b0, 1u);
    h8 k2 = r8_pl_ld_k(k_b0, 2u);
    h8 k3 = r8_pl_ld_k(k_b0, 3u);
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (u32 g = 0u; g < 3u; g++) {
      const u32 ks = 4u * g;
      sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qr_reg[ks + 0u], sa);
      sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qr_reg[ks + 1u], sb);
      k0 = r8_pl_ld_k(k_b0, ks + 4u);
      k1 = r8_pl_ld_k(k_b0, ks + 5u);
      __builtin_amdgcn_sched_barrier(0);
      sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k2, qr_reg[ks + 2u], sa);
      sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k3, qr_reg[ks + 3u], sb);
      k2 = r8_pl_ld_k(k_b0, ks + 6u);
      k3 = r8_pl_ld_k(k_b0, ks + 7u);
      __builtin_amdgcn_sched_barrier(0);
    }
    sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qr_reg[12u], sa);
    sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qr_reg[13u], sb);
    sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k2, qr_reg[14u], sa);
    sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k3, qr_reg[15u], sb);

    float s[8];
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) s[v] = sa[v] + sb[v];
    if (kb + 15u > pos_t0) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++)
        if (kb + 8u * hl + v > tp) s[v] = -__builtin_inff();
    }
    float bm = s[0];
#pragma unroll
    for (u32 v = 1u; v < 8u; v++) bm = __builtin_fmaxf(bm, s[v]);
    bm = __builtin_fmaxf(bm, r8_pl_xrow(bm));
    if (__builtin_amdgcn_ballot_w32(!(bm - m < -28.0f)) == 0u) continue;
    const float mn = __builtin_fmaxf(m, bm);
    const float c = __builtin_amdgcn_exp2f(m - mn);
    h8 pb;
    float ps = 0.0f;
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) {
      const float p = __builtin_amdgcn_exp2f(s[v] - mn);
      ps += p;
      pb[v] = (f16)p;
    }
    ls = ls * c + ps;
    m = mn;
    if (__builtin_amdgcn_ballot_w32(c != 1.0f) != 0u) {
#pragma unroll
      for (u32 nt = 0u; nt < 16u; nt++)
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c;
    }
    const u32 v_b0 = r8_pl_v_base_b(bf, vt_lane);
    h8 v0 = r8_pl_ld_vt(v_b0, 0u, 0u);
    h8 v1 = r8_pl_ld_vt(v_b0, 1u, 0u);
    h8 v2 = r8_pl_ld_vt(v_b0, 2u, 0u);
    h8 v3 = r8_pl_ld_vt(v_b0, 3u, 0u);
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (u32 g = 0u; g < 3u; g++) {
      o[4u * g + 0u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[4u * g + 0u]);
      o[4u * g + 1u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[4u * g + 1u]);
      v0 = r8_pl_ld_vt(v_b0, 0u, g + 1u);
      v1 = r8_pl_ld_vt(v_b0, 1u, g + 1u);
      __builtin_amdgcn_sched_barrier(0);
      o[4u * g + 2u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v2, pb, o[4u * g + 2u]);
      o[4u * g + 3u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v3, pb, o[4u * g + 3u]);
      v2 = r8_pl_ld_vt(v_b0, 2u, g + 1u);
      v3 = r8_pl_ld_vt(v_b0, 3u, g + 1u);
      __builtin_amdgcn_sched_barrier(0);
    }
    o[12u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[12u]);
    o[13u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[13u]);
    o[14u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v2, pb, o[14u]);
    o[15u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v3, pb, o[15u]);
  }
  r8_pl_epilogue(o, ls, t0, rl, hl, T, h, 24u, out, gate, gstride, os, gs, oq, oqs);
}

#undef R10_ATTN_PL_ROWB
#undef R10_ATTN_PIN_ACC8
#undef R10_ATTN_TR_DEPTH
