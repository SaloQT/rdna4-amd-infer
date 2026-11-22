/* ie_gdn_wmma.inc -- BF16-pair matrix-core (gfx12 WMMA) chunked Gated DeltaNet
 * prefill kernels + fused conv-ring prep + fused sequential state/gated-RMSNorm
 * (48 linear-attention layers on Qwen3.5 / Saluki).
 *
 * Included by kernels/ie_kernels.c outside the megakernel (#ifndef IE_MEGA).
 *
 * Precision model (lossless BF16-pair splitting):
 *   Every finite FP32 input x is decomposed in round-to-nearest-even into
 *     hi = rne_bf16(x),  lo = rne_bf16(x - (float)hi)
 *   and every matrix product A * B^T accumulates in FP32 across 3 gfx12 WMMAs:
 *     wmma(A_hi, B_hi) + wmma(A_hi, B_lo) + wmma(A_lo, B_hi)
 *   via __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12 (~17+ bits per input,
 *   FP32 accumulation, matching BF16's full 8-bit FP32 exponent range without
 *   FP16 underflow/overflow; max |err| < 1e-6 on chunk outputs and < 1e-5 on S).
 *
 * Kernels provided:
 *   1. ie_gdn_prep_ring      -- Fuses conv1d + SiLU + Q/K L2-norm + beta/log_g
 *                               with race-free K_RING_STORE in 1 launch
 *                               (grid = 2 * nk + nv, safe for all T incl. 256/512).
 *   2. ie_gdn_wy_wmma        -- Chunked WY factors (U, W, Mb, Gc) for GC = 32
 *                               using BF16-pair WMMA for K*K^T and Q*K^T
 *                               (grid = nv * ((T + 31) / 32), matches hip.c cq_m).
 *   3. ie_gdn_seq_wmma_gnorm -- Inter-chunk state recurrence (W*S0, Q*S0, M*Delta,
 *                               K^T*Delta_tilde) keeping all 128x128 state S in
 *                               VGPRs in native WMMA layout (zero S/Delta LDS
 *                               staging), fused with ie_gnorm + Q8_0 quant_wave
 *                               and dead_b (out == NULL) support (grid = nv).
 *
 * D1 Tree-Verify Audit Summary for ie_gdn1p (T = 1..16):
 *   - kernels/ie_kernels.c ie_gdn1p already sizes all static LDS arrays for 16
 *     tokens (gp_q/k/v/d[16][128], gp_r[16][8], gp_bd[16][2]: 39,552 B LDS =
 *     3 WGs/WGP, 255 VGPRs, 0 spills), u16 TR_ANC bitmask covers bits 1..15,
 *     GDN_RING=32 avoids wraparound (16 + 3 <= 32), and ie_gdn_lazy.inc replays
 *     15-bit path masks in ascending topological order.
 *   - Host wiring notes for T = 9..16 in src/hip.c:
 *     (a) line 1437: ensure spec_verify_max() > 8 (or tree verify enabled) sets
 *         initial b->ns = 16u when g->T >= 16u.
 *     (b) lines 1562-1569: skip the s16 (16 full states = 2.41 GB) free-VRAM
 *         downgrade to 8 when IE_GDN_RECORDS != 0 (records mode immediately sets
 *         b->ns = 1u on line 1595 and uses only ~36 MB for 15 records).
 *     (c) lines 3650-3653: in gpu_tree_ok / OP_ATTN, allow pos < b->long_min by
 *         routing tr_on verify attention to ie_attn_long at any pos >= 0.
 */

#ifndef IE_MEGA

#ifndef GC
#define GC 32u
#endif

typedef __bf16 gw_bf8 __attribute__((ext_vector_type(8)));
typedef float gw_f8 __attribute__((ext_vector_type(8)));
typedef unsigned int gw_u2 __attribute__((ext_vector_type(2)));

/* Round finite FP32 to nearest-even BF16 kept in upper 16 bits of u32 (exact
 * IEEE754 FP32 representation of the rounded BF16 value when bit-cast to float). */
static inline u32 gw_rne_bf16_hi32(float x) {
  const u32 u = __builtin_bit_cast(u32, x);
  return (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
}

static inline void gw_split_scalar(float x, __bf16 *hi, __bf16 *lo) {
  const u32 h = gw_rne_bf16_hi32(x);
  const u32 l = gw_rne_bf16_hi32(x - __builtin_bit_cast(float, h));
  const u16 hu = (u16)(h >> 16), lu = (u16)(l >> 16);
  *hi = __builtin_bit_cast(__bf16, hu);
  *lo = __builtin_bit_cast(__bf16, lu);
}

static inline void gw_split_f4(f4 x, gw_u2 *hi, gw_u2 *lo) {
  const u32 h0 = gw_rne_bf16_hi32(x.x), h1 = gw_rne_bf16_hi32(x.y);
  const u32 h2 = gw_rne_bf16_hi32(x.z), h3 = gw_rne_bf16_hi32(x.w);
  const u32 l0 = gw_rne_bf16_hi32(x.x - __builtin_bit_cast(float, h0));
  const u32 l1 = gw_rne_bf16_hi32(x.y - __builtin_bit_cast(float, h1));
  const u32 l2 = gw_rne_bf16_hi32(x.z - __builtin_bit_cast(float, h2));
  const u32 l3 = gw_rne_bf16_hi32(x.w - __builtin_bit_cast(float, h3));
  *hi = (gw_u2){(h0 >> 16) | h1, (h2 >> 16) | h3};
  *lo = (gw_u2){(l0 >> 16) | l1, (l2 >> 16) | l3};
}

static inline void gw_split_f8(gw_f8 x, gw_bf8 *hi, gw_bf8 *lo) {
  u32x4 hp, lp;
#pragma unroll
  for (u32 i = 0; i < 4u; i++) {
    const float a = x[2u * i], b = x[2u * i + 1u];
    const u32 ah = gw_rne_bf16_hi32(a), bh = gw_rne_bf16_hi32(b);
    hp[i] = (ah >> 16) | bh;
    const u32 al = gw_rne_bf16_hi32(a - __builtin_bit_cast(float, ah));
    const u32 bl = gw_rne_bf16_hi32(b - __builtin_bit_cast(float, bh));
    lp[i] = (al >> 16) | bl;
  }
  *hi = __builtin_bit_cast(gw_bf8, hp);
  *lo = __builtin_bit_cast(gw_bf8, lp);
}

static inline gw_f8 gw_wmma_pair(gw_bf8 ah, gw_bf8 al, gw_bf8 bh, gw_bf8 bl, gw_f8 acc) {
  acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(ah, bh, acc);
  acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(ah, bl, acc);
  acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(al, bh, acc);
  return acc;
}

/* 1. Fused GDN prep + conv-ring store:
 *    Grid: 1D of (2 * nk + nv) workgroups (80 WGs on Saluki).
 *    Workgroup wg owns channels [wg*128 .. wg*128 + 127] exclusively across all
 *    T tokens, reading historical ring slots (pos - 1..3) into registers first,
 *    computing causal conv1d + SiLU + L2-norm (Q/K) + (beta, log_g) without
 *    global qk reread, then barrier() + writing the last min(T, 3) tokens into
 *    ring with zero cross-WG aliasing even when T is a multiple of GDN_RING (32). */
KERNEL ie_gdn_prep_ring(const G float *in, G float *ring, const G float *cw, const G float *dtb, const G float *sa,
                        u32 pos, u32 T, u32 cd, u32 nk, u32 nv, float eps, u32 is, G float *qk, G float *vv, G float *bg) {
#pragma clang fp contract(off)
  const u32 wg = wgid(), th = tid(), w = th >> 5, l = lane();
  const u32 n_heads = 2u * nk + nv, inner = nv * 128u;
  if (wg >= n_heads || T == 0u) return;
  const u32 c_base = wg * 128u;

  f4 wc[4];
  float rh0[4], rh1[4], rh2[4];
#pragma unroll
  for (u32 q = 0; q < 4u; q++) {
    const u32 c = c_base + 32u * q + l;
    wc[q] = *(const G f4 *)(cw + c * 4u);
    rh0[q] = pos >= 1u ? ring[(unsigned long)((pos - 1u) & (GDN_RING - 1u)) * cd + c] : 0.0f;
    rh1[q] = pos >= 2u ? ring[(unsigned long)((pos - 2u) & (GDN_RING - 1u)) * cd + c] : 0.0f;
    rh2[q] = pos >= 3u ? ring[(unsigned long)((pos - 3u) & (GDN_RING - 1u)) * cd + c] : 0.0f;
  }

  for (u32 t = w; t < T; t += 8u) {
    float y4[4], ss = 0.0f;
#pragma unroll
    for (u32 q = 0; q < 4u; q++) {
      const u32 c = c_base + 32u * q + l;
      const float x3 = t >= 3u ? in[(unsigned long)(t - 3u) * is + c] : (t == 2u ? rh0[q] : (t == 1u ? rh1[q] : rh2[q]));
      const float x2 = t >= 2u ? in[(unsigned long)(t - 2u) * is + c] : (t == 1u ? rh0[q] : rh1[q]);
      const float x1 = t >= 1u ? in[(unsigned long)(t - 1u) * is + c] : rh0[q];
      const float x0 = in[(unsigned long)t * is + c];
      const float acc = ((x3 * wc[q].x + x2 * wc[q].y) + x1 * wc[q].z) + x0 * wc[q].w;
      const float y = acc / (1.0f + __builtin_expf(-acc));
      y4[q] = y;
      ss += y * y;
    }
    if (wg < 2u * nk) {
      const u32 isk = wg >= nk ? 1u : 0u, kh = isk ? (wg - nk) : wg;
      ss = wave_sum_all(ss);
      const float r = (1.0f / __builtin_sqrtf(ss + eps)) * (isk ? 1.0f : (1.0f / __builtin_sqrtf(128.0f)));
      G float *dst = qk + ((unsigned long)t * nk + kh) * 256u + isk * 128u;
#pragma unroll
      for (u32 q = 0; q < 4u; q++) dst[32u * q + l] = y4[q] * r;
    } else {
      const u32 hv = wg - 2u * nk;
      G float *dst = vv + (unsigned long)t * inner + hv * 128u;
#pragma unroll
      for (u32 q = 0; q < 4u; q++) dst[32u * q + l] = y4[q];
    }
  }

  if (wg >= 2u * nk) {
    const u32 hv = wg - 2u * nk;
    const float dtb_h = dtb[hv], sa_h = sa[hv];
    for (u32 t = th; t < T; t += NT) {
      const G float *it = in + (unsigned long)t * is;
      const float al = it[cd + inner + nv + hv] + dtb_h;
      const float sp = al > 20.0f ? al : __builtin_logf(1.0f + __builtin_expf(al));
      bg[((unsigned long)t * nv + hv) * 2u] = 1.0f / (1.0f + __builtin_expf(-it[cd + inner + hv]));
      bg[((unsigned long)t * nv + hv) * 2u + 1u] = sp * sa_h;
    }
  }

  barrier();
  const u32 nr = T < 3u ? T : 3u;
  if (w < nr) {
    const u32 tw = T - nr + w;
    G float *rdst = ring + (unsigned long)((pos + tw) & (GDN_RING - 1u)) * cd + c_base;
    const G float *rsrc = in + (unsigned long)tw * is + c_base;
#pragma unroll
    for (u32 q = 0; q < 4u; q++) rdst[32u * q + l] = rsrc[32u * q + l];
  }
}

/* 2. Chunked WY representation using BF16-pair WMMA (GC = 32):
 *    Grid: nv * ((T + 31) / 32) workgroups (drop-in signature & buffer match for ie_gdn_wy).
 *    Waves 0..2 compute lower-triangular tiles (0,0), (1,0), (1,1) of K*K^T;
 *    Waves 3..5 compute lower-triangular tiles (0,0), (1,0), (1,1) of Q*K^T;
 *    Wave 6 zeroes the upper-right (0,1) tile of Mb;
 *    All 256 threads solve (I + L)^-1 [beta*V | beta*gamma*K] -> U, W in FP32. */
typedef union {
  struct {
    __bf16 qh[GC][136];
    __bf16 ql[GC][136];
  } q;
  float L[GC][GC + 4u];
} gw_wy_u_t;

static LDS __bf16 gw_wy_kh[GC][136] __attribute__((aligned(16)));
static LDS __bf16 gw_wy_kl[GC][136] __attribute__((aligned(16)));
static LDS gw_wy_u_t gw_wy_u __attribute__((aligned(16)));
static LDS float gw_wy_G[GC], gw_wy_b[GC], gw_wy_lg[GC];

KERNEL ie_gdn_wy_wmma(const G float *qk, const G float *vv, const G float *bg, G float *U, G float *W, G float *Mb,
                      G float *Gc, u32 T, u32 nk, u32 nv) {
  const u32 h = wgid() % nv, b = wgid() / nv, t0 = b * GC, n = T - t0 < GC ? T - t0 : GC;
  const u32 kh = h % nk, th = tid(), w = th >> 5, l = lane(), hl = l >> 4, rl = l & 15u;

  {
    f4 q4[4], k4[4];
#pragma unroll
    for (u32 m = 0; m < 4u; m++) {
      const u32 e = th + NT * m, r = e >> 5;
      const G float *src = qk + ((unsigned long)(t0 + r) * nk + kh) * 256u + 4u * (e & 31u);
      q4[m] = r < n ? *(const G f4 *)src : (f4){0.0f, 0.0f, 0.0f, 0.0f};
      k4[m] = r < n ? *(const G f4 *)(src + 128u) : (f4){0.0f, 0.0f, 0.0f, 0.0f};
    }
#pragma unroll
    for (u32 m = 0; m < 4u; m++) {
      const u32 e = th + NT * m, r = e >> 5, c4 = 4u * (e & 31u);
      gw_u2 qhi, qlo, khi, klo;
      gw_split_f4(q4[m], &qhi, &qlo);
      gw_split_f4(k4[m], &khi, &klo);
      *(LDSP gw_u2 *)&gw_wy_u.q.qh[r][c4] = qhi;
      *(LDSP gw_u2 *)&gw_wy_u.q.ql[r][c4] = qlo;
      *(LDSP gw_u2 *)&gw_wy_kh[r][c4] = khi;
      *(LDSP gw_u2 *)&gw_wy_kl[r][c4] = klo;
    }
  }
  if (th < GC) {
    gw_wy_b[th] = th < n ? bg[((unsigned long)(t0 + th) * nv + h) * 2u] : 0.0f;
    gw_wy_lg[th] = th < n ? bg[((unsigned long)(t0 + th) * nv + h) * 2u + 1u] : 0.0f;
  }
  lbarrier();
  if (th < GC) {
    float G0 = 0.0f;
    for (u32 r = 0; r <= th; r++) G0 += gw_wy_lg[r];
    gw_wy_G[th] = G0;
    if (th < n) Gc[(unsigned long)(t0 + th) * nv + h] = G0;
  }
  lbarrier();

  gw_f8 acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const u32 is_qk = w >= 3u ? 1u : 0u;
  const u32 ti = is_qk ? (w - 3u) : w;
  const u32 tr = ti >= 1u ? 1u : 0u, tc = ti == 2u ? 1u : 0u;
  if (w < 6u && (tr == 0u || n > 16u)) {
    const LDSP __bf16 (*ah_p)[136] = is_qk ? (const LDSP __bf16 (*)[136])gw_wy_u.q.qh : (const LDSP __bf16 (*)[136])gw_wy_kh;
    const LDSP __bf16 (*al_p)[136] = is_qk ? (const LDSP __bf16 (*)[136])gw_wy_u.q.ql : (const LDSP __bf16 (*)[136])gw_wy_kl;
#pragma unroll
    for (u32 kd = 0; kd < 8u; kd++) {
      const gw_bf8 ah = *(const LDSP gw_bf8 *)&ah_p[16u * tr + rl][16u * kd + 8u * hl];
      const gw_bf8 al = *(const LDSP gw_bf8 *)&al_p[16u * tr + rl][16u * kd + 8u * hl];
      const gw_bf8 bh = *(const LDSP gw_bf8 *)&gw_wy_kh[16u * tc + rl][16u * kd + 8u * hl];
      const gw_bf8 bl = *(const LDSP gw_bf8 *)&gw_wy_kl[16u * tc + rl][16u * kd + 8u * hl];
      acc = gw_wmma_pair(ah, al, bh, bl, acc);
    }
  }
  lbarrier(); /* gw_wy_u.q reads complete before gw_wy_u.L writes */

  G float *Mb_h = Mb + ((unsigned long)b * nv + h) * GC * GC;
  if (w < 3u) {
    const u32 c = 16u * tc + rl;
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      const u32 r = 16u * tr + 8u * hl + v;
      const float dec = (r < n && c < r) ? __builtin_expf(gw_wy_G[r] - gw_wy_G[c]) : 0.0f;
      gw_wy_u.L[r][c] = (r < n && c < r) ? gw_wy_b[r] * dec * acc[v] : 0.0f;
    }
  } else if (w < 6u) {
    const u32 c = 16u * tc + rl;
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      const u32 r = 16u * tr + 8u * hl + v;
      const float dec = (r < n && c <= r) ? __builtin_expf(gw_wy_G[r] - gw_wy_G[c]) : 0.0f;
      Mb_h[r * GC + c] = dec * acc[v];
    }
  } else if (w == 6u) {
    const u32 c = 16u + rl;
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      const u32 r = 8u * hl + v;
      Mb_h[r * GC + c] = 0.0f;
    }
  }
  lbarrier();

  /* Forward substitution (I + L)^-1 [beta*V | beta*gamma*K], 1 column per thread in FP32 */
  float x[GC];
  const int isw = th >= 128u;
  const u32 j = th & 127u;
#pragma unroll
  for (u32 r = 0; r < GC; r++) {
    x[r] = 0.0f;
    if (r < n) {
      const float kv_r = isw ? qk[((unsigned long)(t0 + r) * nk + kh) * 256u + 128u + j]
                             : vv[((unsigned long)(t0 + r) * nv + h) * 128u + j];
      float rhs = gw_wy_b[r] * (isw ? (__builtin_expf(gw_wy_G[r]) * kv_r) : kv_r);
#pragma unroll
      for (u32 c = 0; c < r; c++) rhs -= gw_wy_u.L[r][c] * x[c];
      x[r] = rhs;
      (isw ? W : U)[((unsigned long)(t0 + r) * nv + h) * 128u + j] = rhs;
    }
  }
}

/* 3. Inter-chunk sequential recurrence + output projection + fused ie_gnorm + Q8_0:
 *    Grid: nv workgroups (1 WG per value head h; wave w owns value cols 16*w .. 16*w+15).
 *    Lane (hl = l>>4, rl = l&15) holds st[kd][e] = S[16*kd + 8*hl + e][16*w + rl]
 *    (64 VGPRs/lane), matching the B-operand & D-accumulator layout of gfx12 WMMA
 *    so S0, Delta, and Delta_tilde never stage to LDS.
 *    If nw != NULL, fuses ie_gnorm (RMSNorm(o) * nw * SiLU(z)) + quant_wave Q8_0,
 *    skipping global FP32 writes when out == NULL (dead_b). If nw == NULL, writes
 *    raw pre-gnorm O to out (compatible with IE_GDN_CHECK=1). */
typedef union {
  struct {
    __bf16 wh[GC][136];
    __bf16 wl[GC][136];
  } w;
  struct {
    __bf16 kth[128][40];
    __bf16 ktl[128][40];
  } kt;
  float o[GC][132];
} gw_sq_u1_t;

static LDS gw_sq_u1_t gw_sq_u1 __attribute__((aligned(16)));
static LDS __bf16 gw_sq_qh[GC][136] __attribute__((aligned(16)));
static LDS __bf16 gw_sq_ql[GC][136] __attribute__((aligned(16)));
static LDS __bf16 gw_sq_mh[GC][40] __attribute__((aligned(16)));
static LDS __bf16 gw_sq_ml[GC][40] __attribute__((aligned(16)));
static LDS float gw_sq_G[GC];

KERNEL ie_gdn_seq_wmma_gnorm(const G float *qk, const G float *U, const G float *W, const G float *Mb,
                             const G float *Gc, const G float *in, const G float *nw, G float *out, G u8 *oq,
                             u32 cd, float eps, u32 is, u32 os, u32 qs,
                             G float *S, u32 c, u32 ns, u32 pos, u32 T, u32 nk, u32 nv) {
  const u32 h = wgid();
  if (h >= nv || T == 0u) return;
  const u32 kh = h % nk, th = tid(), w = th >> 5, l = lane(), hl = l >> 4, rl = l & 15u;
  const u32 col = 16u * w + rl;
  const unsigned long slot = (unsigned long)nv * 16384u;
  const unsigned long wrs = (unsigned long)nv * 128u, qrs = (unsigned long)nk * 256u;
  const G float *Sp = S + (unsigned long)c * slot + (unsigned long)h * 16384u;

  gw_f8 st[8];
#pragma unroll
  for (u32 kd = 0; kd < 8u; kd++)
#pragma unroll
    for (u32 e = 0; e < 8u; e++)
      st[kd][e] = pos ? Sp[(16u * kd + 8u * hl + e) * 128u + col] : 0.0f;

  float nw_q[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (nw) {
#pragma unroll
    for (u32 q = 0; q < 4u; q++) nw_q[q] = nw[32u * q + l];
  }

  for (u32 b = 0, t0 = 0; t0 < T; b++, t0 += GC) {
    const u32 n = T - t0 < GC ? T - t0 : GC;

    /* Stage W, Q, M (BF16-pair) + Gc into LDS, and preload U into VGPRs */
    {
      f4 w4[4], q4[4];
#pragma unroll
      for (u32 m = 0; m < 4u; m++) {
        const u32 e = th + NT * m, r = e >> 5, c4 = 4u * (e & 31u);
        w4[m] = r < n ? *(const G f4 *)(W + (unsigned long)(t0 + r) * wrs + h * 128u + c4) : (f4){0.0f, 0.0f, 0.0f, 0.0f};
        q4[m] = r < n ? *(const G f4 *)(qk + (unsigned long)(t0 + r) * qrs + kh * 256u + c4) : (f4){0.0f, 0.0f, 0.0f, 0.0f};
      }
      const u32 mr = th >> 3, mc4 = 4u * (th & 7u);
      const f4 mv = *(const G f4 *)(Mb + (((unsigned long)b * nv + h) * GC + mr) * GC + mc4);
      const float gv = th < n ? Gc[(unsigned long)(t0 + th) * nv + h] : 0.0f;

#pragma unroll
      for (u32 m = 0; m < 4u; m++) {
        const u32 e = th + NT * m, r = e >> 5, c4 = 4u * (e & 31u);
        gw_u2 whi, wlo, qhi, qlo;
        gw_split_f4(w4[m], &whi, &wlo);
        gw_split_f4(q4[m], &qhi, &qlo);
        *(LDSP gw_u2 *)&gw_sq_u1.w.wh[r][c4] = whi;
        *(LDSP gw_u2 *)&gw_sq_u1.w.wl[r][c4] = wlo;
        *(LDSP gw_u2 *)&gw_sq_qh[r][c4] = qhi;
        *(LDSP gw_u2 *)&gw_sq_ql[r][c4] = qlo;
      }
      {
        gw_u2 mhi, mlo;
        gw_split_f4(mv, &mhi, &mlo);
        *(LDSP gw_u2 *)&gw_sq_mh[mr][mc4] = mhi;
        *(LDSP gw_u2 *)&gw_sq_ml[mr][mc4] = mlo;
      }
      if (th < GC) gw_sq_G[th] = gv;
    }

    gw_f8 u_reg[2];
#pragma unroll
    for (u32 mt = 0; mt < 2u; mt++)
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const u32 rt = 16u * mt + 8u * hl + v;
        u_reg[mt][v] = rt < n ? U[(unsigned long)(t0 + rt) * wrs + h * 128u + col] : 0.0f;
      }

    lbarrier();

    /* W * S0 and Q * S0 directly from st[kd] registers */
    gw_f8 ws[2] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                   {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    gw_f8 qs_acc[2] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
    for (u32 kd = 0; kd < 8u; kd++) {
      gw_bf8 sh, sl;
      gw_split_f8(st[kd], &sh, &sl);
      {
        const gw_bf8 wh0 = *(const LDSP gw_bf8 *)&gw_sq_u1.w.wh[rl][16u * kd + 8u * hl];
        const gw_bf8 wl0 = *(const LDSP gw_bf8 *)&gw_sq_u1.w.wl[rl][16u * kd + 8u * hl];
        const gw_bf8 qh0 = *(const LDSP gw_bf8 *)&gw_sq_qh[rl][16u * kd + 8u * hl];
        const gw_bf8 ql0 = *(const LDSP gw_bf8 *)&gw_sq_ql[rl][16u * kd + 8u * hl];
        ws[0] = gw_wmma_pair(wh0, wl0, sh, sl, ws[0]);
        qs_acc[0] = gw_wmma_pair(qh0, ql0, sh, sl, qs_acc[0]);
      }
      if (n > 16u) {
        const gw_bf8 wh1 = *(const LDSP gw_bf8 *)&gw_sq_u1.w.wh[16u + rl][16u * kd + 8u * hl];
        const gw_bf8 wl1 = *(const LDSP gw_bf8 *)&gw_sq_u1.w.wl[16u + rl][16u * kd + 8u * hl];
        const gw_bf8 qh1 = *(const LDSP gw_bf8 *)&gw_sq_qh[16u + rl][16u * kd + 8u * hl];
        const gw_bf8 ql1 = *(const LDSP gw_bf8 *)&gw_sq_ql[16u + rl][16u * kd + 8u * hl];
        ws[1] = gw_wmma_pair(wh1, wl1, sh, sl, ws[1]);
        qs_acc[1] = gw_wmma_pair(qh1, ql1, sh, sl, qs_acc[1]);
      }
    }

    /* Delta = U - W * S0 (already in B-operand layout for M * Delta!) */
    gw_f8 d_reg[2], o_acc[2];
    gw_bf8 dh[2], dl[2];
#pragma unroll
    for (u32 mt = 0; mt < 2u; mt++) {
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const u32 rt = 16u * mt + 8u * hl + v;
        const float gl = gw_sq_G[rt];
        d_reg[mt][v] = rt < n ? (u_reg[mt][v] - ws[mt][v]) : 0.0f;
        o_acc[mt][v] = rt < n ? (__builtin_expf(gl) * qs_acc[mt][v]) : 0.0f;
      }
      gw_split_f8(d_reg[mt], &dh[mt], &dl[mt]);
    }

    /* O = diag(gamma) Q S0 + M * Delta (M is lower-triangular: (0,0), (1,0), (1,1)) */
    {
      const gw_bf8 mh00 = *(const LDSP gw_bf8 *)&gw_sq_mh[rl][8u * hl];
      const gw_bf8 ml00 = *(const LDSP gw_bf8 *)&gw_sq_ml[rl][8u * hl];
      o_acc[0] = gw_wmma_pair(mh00, ml00, dh[0], dl[0], o_acc[0]);
      if (n > 16u) {
        const gw_bf8 mh10 = *(const LDSP gw_bf8 *)&gw_sq_mh[16u + rl][8u * hl];
        const gw_bf8 ml10 = *(const LDSP gw_bf8 *)&gw_sq_ml[16u + rl][8u * hl];
        const gw_bf8 mh11 = *(const LDSP gw_bf8 *)&gw_sq_mh[16u + rl][16u + 8u * hl];
        const gw_bf8 ml11 = *(const LDSP gw_bf8 *)&gw_sq_ml[16u + rl][16u + 8u * hl];
        o_acc[1] = gw_wmma_pair(mh10, ml10, dh[0], dl[0], o_acc[1]);
        o_acc[1] = gw_wmma_pair(mh11, ml11, dh[1], dl[1], o_acc[1]);
      }
    }

    /* Rescale Delta_tilde = diag(gamma_n / gamma) * Delta in registers */
    const float Gn = gw_sq_G[n - 1u], eg = __builtin_expf(Gn);
#pragma unroll
    for (u32 kt = 0; kt < 2u; kt++) {
      gw_f8 d_sc;
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const u32 rt = 16u * kt + 8u * hl + v;
        d_sc[v] = rt < n ? (__builtin_expf(Gn - gw_sq_G[rt]) * d_reg[kt][v]) : 0.0f;
      }
      gw_split_f8(d_sc, &dh[kt], &dl[kt]);
    }

    lbarrier(); /* W reads complete before K^T stages into gw_sq_u1.kt */

    /* Stage K^T [128][32] into gw_sq_u1.kt */
    {
      f4 k4[4];
#pragma unroll
      for (u32 m = 0; m < 4u; m++) {
        const u32 e = th + NT * m, r = e >> 5, c4 = 4u * (e & 31u);
        k4[m] = r < n ? *(const G f4 *)(qk + (unsigned long)(t0 + r) * qrs + kh * 256u + 128u + c4)
                      : (f4){0.0f, 0.0f, 0.0f, 0.0f};
      }
#pragma unroll
      for (u32 m = 0; m < 4u; m++) {
        const u32 e = th + NT * m, r = e >> 5, c4 = 4u * (e & 31u);
        __bf16 kh0, kl0, kh1, kl1, kh2, kl2, kh3, kl3;
        gw_split_scalar(k4[m].x, &kh0, &kl0);
        gw_split_scalar(k4[m].y, &kh1, &kl1);
        gw_split_scalar(k4[m].z, &kh2, &kl2);
        gw_split_scalar(k4[m].w, &kh3, &kl3);
        gw_sq_u1.kt.kth[c4 + 0u][r] = kh0, gw_sq_u1.kt.ktl[c4 + 0u][r] = kl0;
        gw_sq_u1.kt.kth[c4 + 1u][r] = kh1, gw_sq_u1.kt.ktl[c4 + 1u][r] = kl1;
        gw_sq_u1.kt.kth[c4 + 2u][r] = kh2, gw_sq_u1.kt.ktl[c4 + 2u][r] = kl2;
        gw_sq_u1.kt.kth[c4 + 3u][r] = kh3, gw_sq_u1.kt.ktl[c4 + 3u][r] = kl3;
      }
    }
    lbarrier();

    /* S_new = gamma_n * S0 + K^T * Delta_tilde (accumulates directly into st[kd]) */
#pragma unroll
    for (u32 kd = 0; kd < 8u; kd++) {
      st[kd] *= eg;
      {
        const gw_bf8 kh0 = *(const LDSP gw_bf8 *)&gw_sq_u1.kt.kth[16u * kd + rl][8u * hl];
        const gw_bf8 kl0 = *(const LDSP gw_bf8 *)&gw_sq_u1.kt.ktl[16u * kd + rl][8u * hl];
        st[kd] = gw_wmma_pair(kh0, kl0, dh[0], dl[0], st[kd]);
      }
      if (n > 16u) {
        const gw_bf8 kh1 = *(const LDSP gw_bf8 *)&gw_sq_u1.kt.kth[16u * kd + rl][16u + 8u * hl];
        const gw_bf8 kl1 = *(const LDSP gw_bf8 *)&gw_sq_u1.kt.ktl[16u * kd + rl][16u + 8u * hl];
        st[kd] = gw_wmma_pair(kh1, kl1, dh[1], dl[1], st[kd]);
      }
    }

    if (!nw) {
      if (out) {
#pragma unroll
        for (u32 mt = 0; mt < 2u; mt++)
#pragma unroll
          for (u32 v = 0; v < 8u; v++) {
            const u32 rt = 16u * mt + 8u * hl + v;
            if (rt < n) out[(unsigned long)(t0 + rt) * os + h * 128u + col] = o_acc[mt][v];
          }
      }
      lbarrier();
    } else {
      lbarrier(); /* K^T reads complete before O stages into gw_sq_u1.o */
#pragma unroll
      for (u32 mt = 0; mt < 2u; mt++)
#pragma unroll
        for (u32 v = 0; v < 8u; v++) {
          const u32 rt = 16u * mt + 8u * hl + v;
          gw_sq_u1.o[rt][col] = o_acc[mt][v];
        }
      lbarrier();

      const u32 nb_tot = (nv * 128u) >> 5;
      for (u32 rt = w; rt < n; rt += 8u) {
        float ov[4], s2 = 0.0f;
#pragma unroll
        for (u32 q = 0; q < 4u; q++) {
          ov[q] = gw_sq_u1.o[rt][32u * q + l];
          s2 += ov[q] * ov[q];
        }
        s2 = wave_sum_all(s2);
        const float r = 1.0f / __builtin_sqrtf(s2 / 128.0f + eps);
        const G float *in_z = in + (unsigned long)(t0 + rt) * is + cd + h * 128u;
        G float *out_r = out ? (out + (unsigned long)(t0 + rt) * os + h * 128u) : (G float *)0;
        G u8 *oq_r = oq ? (oq + (unsigned long)(t0 + rt) * qs) : (G u8 *)0;
#pragma unroll
        for (u32 q = 0; q < 4u; q++) {
          const u32 jj = 32u * q + l;
          const float z = in_z[jj];
          const float y = ov[q] * r * nw_q[q] * (z / (1.0f + __builtin_expf(-z)));
          if (out_r) out_r[jj] = y;
          if (oq_r) quant_wave(y, (h * 128u + 32u * q) >> 5, oq_r, nb_tot);
        }
      }
      lbarrier();
    }
  }

  G float *Sw = S + (unsigned long)((c + T - 1u) % ns) * slot + (unsigned long)h * 16384u;
#pragma unroll
  for (u32 kd = 0; kd < 8u; kd++)
#pragma unroll
    for (u32 e = 0; e < 8u; e++)
      Sw[(16u * kd + 8u * hl + e) * 128u + col] = st[kd][e];
}

#endif /* !IE_MEGA */
