/* ie_attn_pl.inc -- attention of a prompt chunk at long positions: ie_attn_pl,
 * included at the end of ie_kernels.c (not in the megakernel builds, not for
 * the f32 KV cache). The same job as ie_attn_fa2 (hd 256, up to 6 query heads
 * per KV head, causal, fp16 WMMA flash attention, optional output gate), built
 * so that a deep chunk is not limited by the latency of its own K/V loads.
 *
 * Workgroup (kh, tile) = KV head kh and the 16 tokens 16 tile .. + 15 of the
 * chunk (positions pos0 + t), as ie_attn_fa2. The 8 waves have two jobs:
 *   waves 0 .. GA - 1 (consumers): wave w does query head kh GA + w for the 16
 *     tokens, end to end. S^T = K Q^T per block of PL_NB = 16 keys (the keys
 *     are the rows of the WMMA, the tokens the columns, so a lane's S values
 *     are 8 keys of one token), the online softmax of the lane's token (one
 *     exchange with the lane 16 away), and O^T += V^T P^T: the f16 P values
 *     are the right operand layout as they are, no trip through LDS. The
 *     output tile O^T (256 columns x 16 tokens) stays in 128 registers; the
 *     rescale by exp2(m_old - m_new) is skipped while no lane's maximum grows.
 *   waves 6, 7 (producers): all K/V staging. 64 threads load the next blocks
 *     into their registers (PL_D blocks in flight: the latency of the loads
 *     is covered by the compute of the earlier blocks), turn them into fp16
 *     (the q8 cache: int8 x scale with packed fp16 arithmetic, bitwise the
 *     same values as the f32 path of ie_attn_fa2) and store K (row major) and
 *     V (transposed: one row per column) to LDS, double buffered. One
 *     workgroup barrier per block.
 * The scores are in the log2 domain (Q is scaled by log2(e) / sqrt(hd)). Not
 * bitwise equal to ie_attn_fa2: P is rounded to fp16 against another running
 * maximum, the sums are f32 in another order. Needs hd == 256 and
 * n_head / n_kv <= 6 (src/hip.c checks); waves GA .. 5 only take part in the
 * barriers. */
#if !defined(IE_MEGA) && !defined(IE_KV_F32)

#define PL_NB 16u
#define PL_KS 256u  /* halves per K row in LDS (XOR-swizzled: 16 KiB total across 2 buffers) */
#define PL_VS PL_NB /* halves per V^T row in LDS (XOR-swizzled: 16 KiB total across 2 buffers; 32 KiB total = 4 WGs/WGP) */
#define PL_D 3u /* blocks in flight per producer thread (3 * 37 = 111 VGPRs -> 0 spills at 8 waves/SIMD = 4 WGs/WGP) */
#ifdef IE_KV_Q8
#define PL_ATTR __attribute__((amdgpu_waves_per_eu(8, 8)))
#else
#define PL_ATTR
#endif
static LDS f16 pl_k[2][PL_NB * PL_KS] __attribute__((aligned(16)));
static LDS f16 pl_vt[2][256u * PL_VS] __attribute__((aligned(16)));

typedef _Float16 pl_h2 __attribute__((ext_vector_type(2)));
typedef _Float16 pl_h4 __attribute__((ext_vector_type(4)));
typedef unsigned int pl_u2 __attribute__((ext_vector_type(2)));

/* Bank-conflict-free XOR swizzles for 32 KiB LDS (zero padding, 4 WGs/WGP):
 *  - pl_k_off(r, c)    : row r (0..15), 8-half chunk c (0..31); consumer rl=0..7 hits all 8 bank groups.
 *  - pl_vt_off(col, h) : column col (0..255), key-half h (0..1); bijection on 0..511 where BOTH producer
 *                        (col = 8*L + sub) and consumer (col = 16*nt + rl) hit all 8 bank groups! */
static inline u32 pl_k_off(u32 r, u32 c) { return r * 256u + 8u * (c ^ r); }
static inline u32 pl_vt_off(u32 col, u32 h) { return 8u * ((2u * col + h) ^ ((col >> 2) & 1u) ^ ((col >> 3) & 7u)); }

#ifdef IE_KV_Q8
/* 8 int8 (two words) times the fp16 scale (low 16 bits of sbits) as 8 fp16: byte + 128 into the mantissa of 1024
 * (exact), minus 1152 (exact: the int8 value), times the scale (one rounding of the exact product, the same as
 * the f32 product rounded to fp16). */
static inline h8 pl_deq8(u32 w0, u32 w1, u32 sbits) {
  const u32 mg = 0x64646464u, x0 = w0 ^ 0x80808080u, x1 = w1 ^ 0x80808080u;
  const f16 s = __builtin_bit_cast(f16, (u16)sbits);
  const pl_h2 sv = {s, s}, bias = {(f16)1152.0f, (f16)1152.0f};
  const pl_h2 r0 = (__builtin_bit_cast(pl_h2, __builtin_amdgcn_perm(mg, x0, 0x04010400u)) - bias) * sv;
  const pl_h2 r1 = (__builtin_bit_cast(pl_h2, __builtin_amdgcn_perm(mg, x0, 0x04030402u)) - bias) * sv;
  const pl_h2 r2 = (__builtin_bit_cast(pl_h2, __builtin_amdgcn_perm(mg, x1, 0x04010400u)) - bias) * sv;
  const pl_h2 r3 = (__builtin_bit_cast(pl_h2, __builtin_amdgcn_perm(mg, x1, 0x04030402u)) - bias) * sv;
  return (h8){r0[0], r0[1], r1[0], r1[1], r2[0], r2[1], r3[0], r3[1]};
}

/* 4 int8 (one word) times the fp16 scale: columns 0, 1 in *a, columns 2, 3 in *b (two halves each). */
static inline void pl_deq4(u32 w, u32 sbits, u32 *a, u32 *b) {
  const u32 mg = 0x64646464u, x = w ^ 0x80808080u;
  const f16 s = __builtin_bit_cast(f16, (u16)sbits);
  const pl_h2 sv = {s, s}, bias = {(f16)1152.0f, (f16)1152.0f};
  *a = __builtin_bit_cast(u32, (__builtin_bit_cast(pl_h2, __builtin_amdgcn_perm(mg, x, 0x04010400u)) - bias) * sv);
  *b = __builtin_bit_cast(u32, (__builtin_bit_cast(pl_h2, __builtin_amdgcn_perm(mg, x, 0x04030402u)) - bias) * sv);
}
#endif

/* The loaded K/V of one block: a producer thread's share. K: the thread's 8 pieces of 16 bytes (rows of keys, as the
 * cache has them). V: wave wi of the two producer waves has keys 8 wi .. + 7, lane L holds from each of those keys
 * the column pairs (fp16) or quads (q8) cp = L + 32 j: the keys of a column are then in the registers of one lane
 * and go to the transposed V^T in LDS as 16-byte stores (a store of 4 keys of a column per lane would put all 32
 * lanes of a wave on the same banks). */
typedef struct {
#ifdef IE_KV_Q8
  u32x4 kq[2][2]; /* K: 2 blocks of 32 int8 */
  u32 ks;         /* and their 2 fp16 scales packed (low16 = block 0, high16 = block 1) */
  pl_u2 vq[8];    /* V: 8 contiguous int8 (columns 8 L .. 8 L + 7) per key u = 0..7 */
  u32 vs[4];      /* and those 8 keys' fp16 scales packed 2 per u32 (low16 = key 2m, high16 = key 2m+1) */
#else
  u32x4 kx[8];    /* K: 8 pieces of 8 halves */
  u32 vx[4][8];   /* V: 4 column pairs x 8 keys (2 halves) */
#endif
} pl_slot;

/* The bytes of one (position, KV head) row of the cache. */
#ifdef IE_KV_Q8
#define PL_ROWB 272u
#else
#define PL_ROWB 512u
#endif

/* Issue the loads of block b (keys 16 b .. 16 b + 15; keys >= kend become zeros) for producer thread pt. */
static inline __attribute__((always_inline)) void pl_load(pl_slot *s, const G u8 *kc, const G u8 *vc, const u32 n_kv, const u32 kh,
                                                          const u32 kend, const u32 b, const u32 pt) {
  const u32 wi = pt >> 5, L = pt & 31u;
#ifdef IE_KV_Q8
  u32 ks_pack = 0u;
#pragma unroll
  for (u32 i = 0; i < 2u; i++) {
    const u32 blk = pt + 64u * i, r = blk >> 3, bi = blk & 7u, p = 16u * b + r;
    s->kq[i][0] = s->kq[i][1] = (u32x4)(0u);
    if (p < kend) {
      const G u8 *rec = kc + ((unsigned long)p * n_kv + kh) * PL_ROWB;
      s->kq[i][0] = *(const G u32x4 *)(rec + 32u * bi);
      s->kq[i][1] = *(const G u32x4 *)(rec + 32u * bi + 16u);
      ks_pack |= (u32)(*(const G u16 *)(rec + 256u + 2u * bi)) << (16u * i);
    }
  }
  s->ks = ks_pack;
  /* 1 coalesced u32 load per lane covers all 8 keys x 8 fp16 scales; ds_bpermute broadcasts sub-block bi = L >> 2 */
  u32 s_pair = 0u;
  {
    const u32 uk = L >> 2, wp = L & 3u, pk = 16u * b + 8u * wi + uk;
    if (pk < kend) s_pair = *(const G u32 *)(vc + ((unsigned long)pk * n_kv + kh) * PL_ROWB + 256u + 4u * wp);
  }
  const u32 bi = L >> 2, wp_need = bi >> 1, sh_need = 16u * (bi & 1u);
#pragma unroll
  for (u32 m = 0; m < 4u; m++) {
    const u32 s0 = ((u32)__builtin_amdgcn_ds_bpermute((int)((8u * m + wp_need) << 2), (int)s_pair) >> sh_need) & 0xFFFFu;
    const u32 s1 = ((u32)__builtin_amdgcn_ds_bpermute((int)((8u * m + 4u + wp_need) << 2), (int)s_pair) >> sh_need) & 0xFFFFu;
    s->vs[m] = s0 | (s1 << 16);
  }
#pragma unroll
  for (u32 u = 0; u < 8u; u++) {
    const u32 p = 16u * b + 8u * wi + u;
    s->vq[u] = p < kend ? *(const G pl_u2 *)(vc + ((unsigned long)p * n_kv + kh) * PL_ROWB + 8u * L) : (pl_u2)(0u);
  }
#else
#pragma unroll
  for (u32 i = 0; i < 8u; i++) {
    const u32 r = 2u * i + wi, p = 16u * b + r;
    s->kx[i] = (u32x4)(0u);
    if (p < kend) s->kx[i] = *(const G u32x4 *)(kc + ((unsigned long)p * n_kv + kh) * PL_ROWB + 16u * L);
  }
#pragma unroll
  for (u32 u = 0; u < 8u; u++) {
    const u32 p = 16u * b + 8u * wi + u;
#pragma unroll
    for (u32 j = 0; j < 4u; j++) {
      s->vx[j][u] = 0u;
      if (p < kend) s->vx[j][u] = *(const G u32 *)(vc + ((unsigned long)p * n_kv + kh) * PL_ROWB + 4u * (L + 32u * j));
    }
  }
#endif
}

/* 8 keys of two columns (word u: key u's two halves) as the two columns' 8 halves each: v_perm picks the low
 * (even column) or high (odd column) halves of the keys 2 i and 2 i + 1. */
static inline void pl_transpose8(const u32 *k, u32x4 *ev, u32x4 *od) {
  u32 e[4], o[4];
#pragma unroll
  for (u32 i = 0; i < 4u; i++) {
    e[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x05040100u);
    o[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x07060302u);
  }
  *ev = (u32x4){e[0], e[1], e[2], e[3]};
  *od = (u32x4){o[0], o[1], o[2], o[3]};
}

/* The loaded block to LDS buffer bf: K XOR-swizzled row major, V XOR-swizzled transposed ([column][key]). */
static inline __attribute__((always_inline)) void pl_publish(const pl_slot *s, const u32 bf, const u32 pt) {
  const u32 wi = pt >> 5, L = pt & 31u;
#ifdef IE_KV_Q8
#pragma unroll
  for (u32 i = 0; i < 2u; i++) {
    const u32 blk = pt + 64u * i, r = blk >> 3, bi = blk & 7u;
    const u32 ksi = i ? (s->ks >> 16) : s->ks;
#pragma unroll
    for (u32 q = 0; q < 2u; q++) {
      const u32x4 wv = s->kq[i][q];
      *(LDSP h8 *)(pl_k[bf] + pl_k_off(r, 4u * bi + 2u * q)) = pl_deq8(wv.x, wv.y, ksi);
      *(LDSP h8 *)(pl_k[bf] + pl_k_off(r, 4u * bi + 2u * q + 1u)) = pl_deq8(wv.z, wv.w, ksi);
    }
  }
#pragma unroll
  for (u32 g = 0; g < 2u; g++) {
    u32 lo[8], hi[8];
#pragma unroll
    for (u32 u = 0; u < 8u; u++) {
      u32 a, b2;
      const u32 w = g ? s->vq[u].y : s->vq[u].x;
      const u32 sb = (u & 1u) ? (s->vs[u >> 1] >> 16) : s->vs[u >> 1];
      pl_deq4(w, sb, &a, &b2);
      lo[u] = a, hi[u] = b2;
    }
    const u32 c0 = 8u * L + 4u * g;
    u32x4 e0, o0, e1, o1;
    pl_transpose8(lo, &e0, &o0);
    pl_transpose8(hi, &e1, &o1);
    *(LDSP u32x4 *)(pl_vt[bf] + pl_vt_off(c0 + 0u, wi)) = e0;
    *(LDSP u32x4 *)(pl_vt[bf] + pl_vt_off(c0 + 1u, wi)) = o0;
    *(LDSP u32x4 *)(pl_vt[bf] + pl_vt_off(c0 + 2u, wi)) = e1;
    *(LDSP u32x4 *)(pl_vt[bf] + pl_vt_off(c0 + 3u, wi)) = o1;
  }
#else
#pragma unroll
  for (u32 i = 0; i < 8u; i++) {
    const u32 r = 2u * i + wi;
    *(LDSP u32x4 *)(pl_k[bf] + pl_k_off(r, L)) = s->kx[i];
  }
#pragma unroll
  for (u32 j = 0; j < 4u; j++) {
    const u32 cp = L + 32u * j;
    u32x4 ev, od;
    pl_transpose8(s->vx[j], &ev, &od);
    *(LDSP u32x4 *)(pl_vt[bf] + pl_vt_off(2u * cp, wi)) = ev;
    *(LDSP u32x4 *)(pl_vt[bf] + pl_vt_off(2u * cp + 1u, wi)) = od;
  }
#endif
}

static inline float pl_xrow(float v) { return __builtin_bit_cast(float, xrow_i(__builtin_bit_cast(int, v))); }

KERNEL ie_q_scale_f16(const G u8 *q, G f16 *q16, u32 n, u32 qs) {
  const u32 tk = tokid(), i = (wgid() * NT + tid()) * 4u;
  if (i >= n) return;
  const float qscale = 0.0625f * 1.44269504088896341f;
  const f32x4 v = *(const G f32x4 *)(q + (unsigned long)tk * qs + 4ul * i);
  *(G pl_h4 *)(q16 + (unsigned long)tk * n + i) =
      (pl_h4){(f16)(v.x * qscale), (f16)(v.y * qscale), (f16)(v.z * qscale), (f16)(v.w * qscale)};
}

PL_ATTR KERNEL ie_attn_pl(const G float *q, const G kvt *kc, const G kvt *vc, G float *out, u32 pos0, u32 hd, u32 n_head, u32 n_kv,
                          const G float *gate, u32 gstride, u32 qs, u32 os, u32 gs, u32 T, G u8 *oq, u32 oqs, const G f16 *q16) {
  const u32 ga = n_head / n_kv;
  if (hd != 256u || ga == 0u || ga > 6u || T == 0u) return; /* uniform: before any barrier */
  const u32 ntile = (T + 15u) / 16u, tile = wgid() % ntile, kh = wgid() / ntile, t0 = 16u * tile;
  const u32 w = __builtin_amdgcn_readfirstlane(tid() >> 5), l = lane(), hl = l >> 4, rl = l & 15u;
  const u32 tlast = t0 + 15u < T ? t0 + 15u : T - 1u, kend = pos0 + tlast + 1u; /* keys 0 .. kend - 1 */
  const u32 nblk = (kend + PL_NB - 1u) / PL_NB;
  if (w >= 6u) { /* producers */
    const G u8 *kb8 = (const G u8 *)kc, *vb8 = (const G u8 *)vc;
    const u32 pt = tid() - 192u;
    pl_slot ring[PL_D];
#pragma unroll
    for (u32 s = 0; s < PL_D; s++)
      if (s < nblk) pl_load(&ring[s], kb8, vb8, n_kv, kh, kend, s, pt);
    pl_publish(&ring[0], 0u, pt); /* block 0 */
    if (PL_D < nblk) pl_load(&ring[0], kb8, vb8, n_kv, kh, kend, PL_D, pt);
    for (u32 k0 = 0; k0 < nblk; k0 += PL_D) {
#pragma unroll
      for (u32 j = 0; j < PL_D; j++) {
        const u32 k = k0 + j;
        if (k < nblk) {
          lbarrier(); /* block k is in LDS; the consumers have finished block k - 1 */
          if (k + 1u < nblk) {
            pl_publish(&ring[(j + 1u) % PL_D], (k + 1u) & 1u, pt);
            if (k + 1u + PL_D < nblk) pl_load(&ring[(j + 1u) % PL_D], kb8, vb8, n_kv, kh, kend, k + 1u + PL_D, pt);
          }
        }
      }
    }
    return;
  }
  if (w >= ga) { /* idle waves (fewer than 6 query heads per KV head) */
    for (u32 k = 0; k < nblk; k++) lbarrier();
    return;
  }
  /* consumer: query head kh ga + w, tokens t0 .. t0 + 15 */
  const u32 h = kh * ga + w;
  const float qscale = 0.0625f * 1.44269504088896341f; /* log2(e) / sqrt(256) */
  const u32 tq = t0 + rl < T ? t0 + rl : T - 1u;
  const G u8 *qbase = q16 ? (const G u8 *)(q16 + ((unsigned long)tq * n_head + h) * 256u)
                          : ((const G u8 *)q + (unsigned long)tq * qs + (unsigned long)h * 1024u);
  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2) & 1u) ^ (rl >> 3);
  const v8f zero = (v8f)(0.0f);
  v8f o[16];
#pragma unroll
  for (u32 nt = 0; nt < 16u; nt++) o[nt] = zero;
  float m = -__builtin_inff(), ls = 0.0f;
  const u32 tp = pos0 + t0 + rl; /* the position of this lane's token */
  for (u32 k = 0; k < nblk; k++) {
    lbarrier();
    const u32 kb = PL_NB * k, bf = k & 1u;
    v8f sa = zero, sb = zero; /* two chains: the WMMA of one chain wait for each other */
    if (q16) {
      const G f16 *q16r = (const G f16 *)qbase;
#pragma clang loop unroll_count(2)
      for (u32 ks = 0; ks < 16u; ks += 2u) {
        const h8 q0 = *(const G h8 *)(q16r + 16u * ks + 8u * hl);
        const h8 q1 = *(const G h8 *)(q16r + 16u * (ks + 1u) + 8u * hl);
        const h8 k0 = *(const LDSP h8 *)(pl_k[bf] + pl_k_off(rl, 2u * ks + hl));
        const h8 k1 = *(const LDSP h8 *)(pl_k[bf] + pl_k_off(rl, 2u * (ks + 1u) + hl));
        sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, q0, sa);
        sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, q1, sb);
      }
    } else {
      const G float *qr = (const G float *)qbase;
#pragma clang loop unroll_count(2)
      for (u32 ks = 0; ks < 16u; ks += 2u) {
        h8 q0, q1;
#pragma unroll
        for (u32 e = 0; e < 8u; e++) {
          q0[e] = (f16)(qr[16u * ks + 8u * hl + e] * qscale);
          q1[e] = (f16)(qr[16u * (ks + 1u) + 8u * hl + e] * qscale);
        }
        const h8 k0 = *(const LDSP h8 *)(pl_k[bf] + pl_k_off(rl, 2u * ks + hl));
        const h8 k1 = *(const LDSP h8 *)(pl_k[bf] + pl_k_off(rl, 2u * (ks + 1u) + hl));
        sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, q0, sa);
        sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, q1, sb);
      }
    }
    float s[8];
#pragma unroll
    for (u32 v = 0; v < 8u; v++) s[v] = sa[v] + sb[v];
    if (kb + (PL_NB - 1u) > pos0 + t0) { /* the block may reach past a token's position: causal mask */
#pragma unroll
      for (u32 v = 0; v < 8u; v++)
        if (kb + 8u * hl + v > tp) s[v] = -__builtin_inff();
    }
    float bm = s[0];
#pragma unroll
    for (u32 v = 1; v < 8u; v++) bm = __builtin_fmaxf(bm, s[v]);
    bm = __builtin_fmaxf(bm, pl_xrow(bm));
    /* Bitwise-exact zero-tile PV skip: when every lane has bm - m < -28 (never true on k == 0 where m = -inf),
     * mn == m, c == 1, every p = exp2f(s[v] - m) < 2^-28 rounds to +0.0h in fp16 (pb == 0), and ps < 8 * 2^-28 = 2^-25
     * < 0.5 ULP of ls >= 1.0f so ls + ps == ls in f32. Skips exp2f, rescale, 16 LDS V^T loads, and 16 WMMAs. */
    if (__builtin_amdgcn_ballot_w32(!(bm - m < -28.0f)) == 0u) continue;
    const float mn = __builtin_fmaxf(m, bm);
    const float c = __builtin_amdgcn_exp2f(m - mn);
    h8 pb;
    float ps = 0.0f;
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      const float p = __builtin_amdgcn_exp2f(s[v] - mn);
      ps += p;
      pb[v] = (f16)p;
    }
    ls = ls * c + ps;
    m = mn;
    if (__builtin_amdgcn_ballot_w32(c != 1.0f) != 0u) { /* some lane's maximum grew: rescale the sums (uniform) */
#pragma unroll
      for (u32 nt = 0; nt < 16u; nt++)
#pragma unroll
        for (u32 v = 0; v < 8u; v++) o[nt][v] *= c;
    }
    const LDSP f16 *vp0 = pl_vt[bf] + 8u * vt_lane;
    const LDSP f16 *vp1 = pl_vt[bf] + 256u + 8u * (vt_lane ^ 2u);
    const LDSP f16 *vp2 = pl_vt[bf] + 512u + 8u * (vt_lane ^ 4u);
    const LDSP f16 *vp3 = pl_vt[bf] + 768u + 8u * (vt_lane ^ 6u);
#pragma unroll
    for (u32 g = 0; g < 4u; g++) {
      const h8 v0 = *(const LDSP h8 *)(vp0 + 1024u * g);
      const h8 v1 = *(const LDSP h8 *)(vp1 + 1024u * g);
      const h8 v2 = *(const LDSP h8 *)(vp2 + 1024u * g);
      const h8 v3 = *(const LDSP h8 *)(vp3 + 1024u * g);
      o[4u * g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[4u * g]);
      o[4u * g + 1u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[4u * g + 1u]);
      o[4u * g + 2u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v2, pb, o[4u * g + 2u]);
      o[4u * g + 3u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v3, pb, o[4u * g + 3u]);
      if (g < 3u) __asm__ volatile("" ::: "memory");
    }
  }
  const float inv = 1.0f / (ls + pl_xrow(ls));
  const u32 tk = t0 + rl;
  const G float *gr = (gate && tk < T) ? (const G float *)((const G u8 *)gate + (unsigned long)tk * gs) : (const G float *)0;
  G float *orow = (out && tk < T) ? (G float *)((G u8 *)out + (unsigned long)tk * os) + h * 256u : (G float *)0;
  G u8 *qrow = (oq && tk < T) ? oq + (unsigned long)tk * oqs : (G u8 *)0;
  const u32 nbq = n_head * 8u;
#pragma unroll
  for (u32 blk = 0; blk < 8u; blk++) {
    const u32 nt0 = 2u * blk, nt1 = nt0 + 1u;
    float x0[8], x1[8];
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      const u32 d0 = 32u * blk + 8u * hl + v, d1 = d0 + 16u;
      float a0 = tk < T ? o[nt0][v] * inv : 0.0f;
      float a1 = tk < T ? o[nt1][v] * inv : 0.0f;
      if (gr) {
        a0 *= sigmoidf(gr[h * gstride + d0]);
        a1 *= sigmoidf(gr[h * gstride + d1]);
      }
      x0[v] = a0;
      x1[v] = a1;
      if (orow) {
        orow[d0] = a0;
        orow[d1] = a1;
      }
    }
    if (oq) {
      /* All 32 columns of Q8 block (tk, h * 8 + blk) live in partner lanes (hl=0, rl) and (hl=1, rl):
       * exact 32-way amax and u32 asum via v_permlanex16 (bitwise identical to quant_wave, 0 LDS, 0 barriers). */
      float am = 0.0f;
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        am = __builtin_fmaxf(am, __builtin_fabsf(x0[v]));
        am = __builtin_fmaxf(am, __builtin_fabsf(x1[v]));
      }
      const float amax = __builtin_fmaxf(am, pl_xrow(am));
      const float d = amax / 127.0f;
      u32 sloc = 0u, p00 = 0u, p01 = 0u, p10 = 0u, p11 = 0u;
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const int q0 = d != 0.0f ? (int)__builtin_roundf(x0[v] / d) : 0;
        const int q1 = d != 0.0f ? (int)__builtin_roundf(x1[v] / d) : 0;
        sloc += (u32)q0 + (u32)q1;
        const u32 sh = 8u * (v & 3u);
        if (v < 4u) {
          p00 |= ((u32)(u8)q0) << sh;
          p10 |= ((u32)(u8)q1) << sh;
        } else {
          p01 |= ((u32)(u8)q0) << sh;
          p11 |= ((u32)(u8)q1) << sh;
        }
      }
      const u32 asum = sloc + (u32)xrow_i((int)sloc);
      if (qrow) {
        const u32 qb = h * 8u + blk;
        G u32 *qp = (G u32 *)(qrow + qb * 32u + 8u * hl);
        qp[0] = p00;
        qp[1] = p01;
        qp[4] = p10;
        qp[5] = p11;
        if (hl == 0u) {
          ((G float *)(qrow + 32u * nbq))[qb] = d;
          ((G u32 *)(qrow + 36u * nbq))[qb] = asum;
        }
      }
    }
  }
}

#endif /* !IE_MEGA && !IE_KV_F32 */
