/* ie_attn_long_q8.inc -- single-plane exact int8->fp16 WMMA path for ie_attn_long under IE_KV_Q8.
 *
 * Cuts ie_attn_long WMMA count by 33.3% (96 -> 64 WMMAs per split: 48 -> 32 in K Q^T and
 * 48 -> 32 in P V) and halves P V LDS transpose stages/waits (2 planes -> 1 plane):
 *   - Every signed int8 (-128 .. 127) is 100% exact in a single fp16 (zero lo plane needed).
 *   - S^T = K Q^T: per q8_0 block blk = 0 .. 7 (32 dims), lane (hl, rl) loads the 16 int8 bytes
 *     at 32*blk + 16*hl so BOTH hl=0 and hl=1 belong to block blk and share the same per-key
 *     scale ksc[blk]. Four WMMAs (kb0*qh0, kb0*ql0, kb1*qh1, kb1*ql1) accumulate into fresh
 *     fp32 t = 0, then t[v] (key 8*hl + v) is scaled by that key's fp32 block scale via FMA.
 *   - O = P V: wave w (0 .. 7) computes output columns 32*w .. 32*w + 31 = ONE single q8_0
 *     block w, so vsc[kt] depends only on key rl (0 .. 15). V int8 bytes transpose once through
 *     sc as a single fp16 plane; per-key vsc[kt] is normalized by the wave's exact power-of-2
 *     exponent (preserving full 22-bit ahs + als fp16 pair dynamic range without subnormal
 *     underflow), folded into P in registers, and accumulated in 4 WMMAs instead of 6.
 *
 * Integration in kernels/ie_kernels.c (keep al_deq8 above intact -- ie_attn_long128.inc uses it!):
 *   1) Right after al_deq8's `#endif` (~L3532):
 *        #include "ie_attn_long_q8.inc"
 *   2) In al_body K decl / load (~L3601-3620), under `#elif defined(IE_KV_Q8)`:
 *        decl:  u32x4 kq[8]; u32x4 ksc4 = (u32x4)(0u);
 *        load:  al_q8_kload(kc, row, n_kv, kh, hl, kq, &ksc4);
 *   3) In al_body S^T = K Q^T (~L3687):
 *        #ifdef IE_KV_Q8
 *        if (kon && !(AL_DBG & 4)) {
 *          const LDSP f16 *qh = (const LDSP f16 *)al_lds, *ql = qh + 16u * AL_QS;
 *          s = al_q8_kqt(kc, kq, ksc4, qh, ql, tr, pos, kb, kend, pg, grp, T, cj, n_kv, kh, hl, rl);
 *        }
 *        #else ... (existing !IE_KV_QUANT and IE_KV_Q4 paths) #endif
 *   4) In al_body O = P V loop (~L3807), replace the IE_KV_Q8 branches with:
 *        #ifdef IE_KV_Q8
 *        al_q8_pv_tile(vc, vr[kt], vsc[kt], pp, sc, tr, pos, c0, kt, kend, pg, grp, T, n_kv, kh, w, hl, rl, acc);
 *        continue;
 *        #endif
 */
#ifdef IE_KV_Q8

static LDS float al_q8_vs[8][16] __attribute__((aligned(16)));

/* Unpack 8 signed int8 bytes in (w0, w1) directly to 8 exact fp16 integers in [-128, 127].
 * (byte ^ 0x80) placed into the mantissa of 1024 (0x6400) equals 1152 + byte (exact in fp16);
 * subtracting 1152 gives the exact signed integer in fp16 in 4 perm + 4 pk_sub_f16 ops. */
static inline h8 al_i8_to_h8(u32 w0, u32 w1) {
  const u32 mg = 0x64646464u, x0 = w0 ^ 0x80808080u, x1 = w1 ^ 0x80808080u;
  const al_h2 bias = {(f16)1152.0f, (f16)1152.0f};
  const al_h2 q0 = __builtin_bit_cast(al_h2, __builtin_amdgcn_perm(mg, x0, 0x04010400u)) - bias;
  const al_h2 q1 = __builtin_bit_cast(al_h2, __builtin_amdgcn_perm(mg, x0, 0x04030402u)) - bias;
  const al_h2 q2 = __builtin_bit_cast(al_h2, __builtin_amdgcn_perm(mg, x1, 0x04010400u)) - bias;
  const al_h2 q3 = __builtin_bit_cast(al_h2, __builtin_amdgcn_perm(mg, x1, 0x04030402u)) - bias;
  return (h8){q0[0], q0[1], q1[0], q1[1], q2[0], q2[1], q3[0], q3[1]};
}

static inline u32 al_q8_trow(const G u8 *tr, u32 j, u32 key, u32 pos) {
  return key < pos || key - pos > tr[TR_DEP + j] ? key : pos + tr[TR_PATH + 16u * j + (key - pos)];
}

/* Load one key row's 8 q8_0 blocks so lane (hl, rl) holds dims 32*blk + 16*hl .. + 15 of key rl,
 * and hl == 0 lanes load all 8 fp16 block scales (16 bytes) in one aligned u32x4 load. */
static inline __attribute__((always_inline)) void al_q8_kload(const G kvt *kc, unsigned long row, u32 n_kv, u32 kh,
                                                              u32 hl, u32x4 kq[8], u32x4 *ksc4) {
  const G u8 *rec = (const G u8 *)kc + (row * n_kv + kh) * 272ul;
#pragma unroll
  for (u32 blk = 0; blk < 8u; blk++) kq[blk] = *(const G u32x4 *)(rec + 32ul * blk + 16ul * hl);
  *ksc4 = hl == 0u ? *(const G u32x4 *)(rec + 256ul) : (u32x4)(0u);
}

static inline __attribute__((always_inline)) v8f al_q8_kqt_blocks(const u32x4 kq[8], u32x4 ksc4, const LDSP f16 *qh,
                                                                  const LDSP f16 *ql, u32 hl, u32 rl) {
  const v8f zero = (v8f)(0.0f);
  v8f s = zero;
#pragma unroll
  for (u32 blk = 0; blk < 8u; blk++) {
    const u32 qoff = rl * AL_QS + 32u * blk + 16u * hl;
    const h8 qh0 = *(const LDSP h8 *)(qh + qoff);
    const h8 ql0 = *(const LDSP h8 *)(ql + qoff);
    const h8 qh1 = *(const LDSP h8 *)(qh + qoff + 8u);
    const h8 ql1 = *(const LDSP h8 *)(ql + qoff + 8u);
    const h8 kb0 = al_i8_to_h8(kq[blk][0], kq[blk][1]);
    const h8 kb1 = al_i8_to_h8(kq[blk][2], kq[blk][3]);
    v8f t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb0, qh0, zero);
    t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb0, ql0, t);
    t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb1, qh1, t);
    t = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb1, ql1, t);
    const u32 kw = ksc4[blk >> 1];
    const float my_ks = (float)__builtin_bit_cast(f16, (u16)((blk & 1u) ? (kw >> 16) : kw));
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      const float s0 = I2F(__builtin_amdgcn_readlane(F2I(my_ks), (int)v));
      const float s1 = I2F(__builtin_amdgcn_readlane(F2I(my_ks), (int)(8u + v)));
      s[v] = __builtin_fmaf(t[v], hl ? s1 : s0, s[v]);
    }
  }
  return s;
}

static inline __attribute__((always_inline)) v8f al_q8_kqt(const G kvt *kc, u32x4 kq[8], u32x4 ksc4,
                                                           const LDSP f16 *qh, const LDSP f16 *ql, const G u8 *tr,
                                                           u32 pos, u32 kb, u32 kend, u32 pg, u32 grp, u32 T, u32 cj,
                                                           u32 n_kv, u32 kh, u32 hl, u32 rl) {
  if (tr && kb + 15u >= pos) {
    const u32 jlo = 16u * pg / grp, jhi = (16u * pg + 15u) / grp < T - 1u ? (16u * pg + 15u) / grp : T - 1u;
    v8f s = (v8f)(0.0f);
    for (u32 jj = jlo; jj <= jhi; jj++) {
      const unsigned long row = al_q8_trow(tr, jj, kb + rl < kend ? kb + rl : kend - 1u, pos);
      al_q8_kload(kc, row, n_kv, kh, hl, kq, &ksc4);
      const v8f sj = al_q8_kqt_blocks(kq, ksc4, qh, ql, hl, rl);
      if (cj == jj) s = sj;
    }
    return s;
  }
  return al_q8_kqt_blocks(kq, ksc4, qh, ql, hl, rl);
}

/* Single-plane V^T transpose + 4-WMMA P V step for one 16-key tile kt and wave w's q8_0 block w.
 * Power-of-2 wave-exponent scaling maps max |vsc| into [1024, 2048) bit-exactly (mantissa untouched,
 * inv_w * w_sc == 1.0f), keeping p_sc = P * (vsc * inv_w) safely below fp16 max (<= 2048 < 65504)
 * while lifting small vsc out of fp16 subnormals so (ahs, als) retains full 22-bit precision. */
static inline __attribute__((always_inline)) void al_q8_pv_step(u32x4 vrk, f16 vsck, const LDSP f16 *pp, LDSP f16 *sc,
                                                                u32 kt, u32 w, u32 hl, u32 rl, v8f *acc0, v8f *acc1) {
  const v8f zero = (v8f)(0.0f);
  const h8 vh0 = al_i8_to_h8(vrk[0], vrk[1]);
  const h8 vh1 = al_i8_to_h8(vrk[2], vrk[3]);
  const float vsm = wave_max_pos(hl == 0u ? __builtin_fabsf((float)vsck) : 0.0f);
  const u32 vbits = F2I(vsm) & 0x7f800000u;
  const float inv_w = vbits ? I2F((264u << 23) - vbits) : 0.0f;
  const float w_sc = vbits ? I2F(vbits - (10u << 23)) : 0.0f;
#pragma unroll
  for (u32 e = 0; e < 8u; e++) {
    sc[(16u * hl + e) * AL_SS + rl] = vh0[e];
    sc[(16u * hl + 8u + e) * AL_SS + rl] = vh1[e];
  }
  if (hl == 0u) al_q8_vs[w][rl] = (float)vsck * inv_w;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  const h8 b0 = *(const LDSP h8 *)(sc + rl * AL_SS + 8u * hl);
  const h8 b1 = *(const LDSP h8 *)(sc + (16u + rl) * AL_SS + 8u * hl);
  const h8 ah = *(const LDSP h8 *)(pp + rl * AL_PS + 16u * kt + 8u * hl);
  const h8 al = *(const LDSP h8 *)(pp + 16u * AL_PS + rl * AL_PS + 16u * kt + 8u * hl);
  const v8f vs = *(const LDSP v8f *)(&al_q8_vs[w][8u * hl]);
  h8 ahs, als;
#pragma unroll
  for (u32 e = 0; e < 8u; e++) {
    const float p_sc = ((float)ah[e] + (float)al[e]) * vs[e];
    const f16 hv = (f16)p_sc;
    ahs[e] = hv;
    als[e] = (f16)(p_sc - (float)hv);
  }
  v8f t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ahs, b0, zero);
  v8f t1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ahs, b1, zero);
  if (!(AL_DBG & 16)) {
    t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(als, b0, t0);
    t1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(als, b1, t1);
  }
#pragma unroll
  for (u32 v = 0; v < 8u; v++) {
    (*acc0)[v] = __builtin_fmaf(t0[v], w_sc, (*acc0)[v]);
    (*acc1)[v] = __builtin_fmaf(t1[v], w_sc, (*acc1)[v]);
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

static inline __attribute__((always_inline)) void al_q8_pv_tile(const G kvt *vc, u32x4 vrk, f16 vsck,
                                                                const LDSP f16 *pp, LDSP f16 *sc, const G u8 *tr,
                                                                u32 pos, u32 c0, u32 kt, u32 kend, u32 pg, u32 grp,
                                                                u32 T, u32 n_kv, u32 kh, u32 w, u32 hl, u32 rl,
                                                                v8f acc[2]) {
  if (tr && c0 + 16u * kt + 15u >= pos) {
    const u32 jlo = 16u * pg / grp, jhi = (16u * pg + 15u) / grp < T - 1u ? (16u * pg + 15u) / grp : T - 1u;
    const v8f a0in = acc[0], a1in = acc[1];
    for (u32 jj = jlo; jj <= jhi; jj++) {
      const unsigned long key = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;
      const unsigned long row = al_q8_trow(tr, jj, (u32)key, pos);
      const G u8 *rec = (const G u8 *)vc + (row * n_kv + kh) * 272ul;
      const u32x4 vrj = *(const G u32x4 *)(rec + 32ul * w + 16ul * hl);
      const f16 vsj = ((const G f16 *)(rec + 256ul))[w];
      v8f t0 = a0in, t1 = a1in;
      al_q8_pv_step(vrj, vsj, pp, sc, kt, w, hl, rl, &t0, &t1);
#pragma unroll
      for (u32 v = 0; v < 8u; v++)
        if ((16u * pg + 8u * hl + v) / grp == jj) acc[0][v] = t0[v], acc[1][v] = t1[v];
    }
    return;
  }
  al_q8_pv_step(vrk, vsck, pp, sc, kt, w, hl, rl, &acc[0], &acc[1]);
}

#endif /* IE_KV_Q8 */
