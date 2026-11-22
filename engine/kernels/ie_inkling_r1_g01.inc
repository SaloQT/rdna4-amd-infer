/* ============================================================================
 * Inkling-2 R1 (g01): WMMA Producer-Consumer Prefill Relative-Position Attention
 *                     for RX 9070 XT (gfx1201, wave32, hd=128, d_rel=16, f16 KV)
 *
 * Signature (matches base ie_rel_attn_pf exactly):
 *   KERNEL ie_inkling_rel_attn_pf_r1(
 *       const G float *q, const G kvt *kc, const G kvt *vc,
 *       const G float *R, const G float *rel_bias, const G float *sink, G float *out,
 *       u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
 *       u32 d_rel, u32 rel_extent, u32 win, u32 qs, u32 rs, u32 ss, u32 os)
 *
 * Grid:   ((T + 31u) / 32u, n_kv, 1u), 256 threads/WG (NT=256, 8 waves).
 * Guards: !b->kvq8 && !b->kvq4 && T > 1u && hd == 128u && d_rel == 16u &&
 *         n_kv > 0u && n_head == n_kv * 3u && kv_len > 0u.
 * Parity: f32-tol (fp16 WMMA for Q*K^T, R*rel_bias, P*V; log2 online softmax
 *         with exact per-token/per-head dynamic sink and causal/SWA masking).
 * ========================================================================== */

_Static_assert(sizeof(kvt) == 2u, "ie_inkling_rel_attn_pf_r1 requires f16 KV cache");

#define INK_R1_G01_NB    16u
#define INK_R1_G01_D     3u
#define INK_R1_G01_LOG2E 1.44269504088896340736f
/* 1.0f / sqrtf(128.0f) * log2(e) */
#define INK_R1_G01_QSC   (0.08838834764831844299f * 1.44269504088896340736f)
#define INK_R1_G01_ROWB  256u /* hd=128 f16 elements = 256 bytes per (pos, kh) row */

typedef unsigned int ink_r1_g01_u2 __attribute__((ext_vector_type(2)));

/* Total LDS = 8192 + 8192 + 3072 + 12672 = 32,128 B <= 32 KiB (4 WGs/WGP). */
typedef struct {
  f16   k[2][INK_R1_G01_NB * 128u];   /*  8,192 B: double-buffered XOR-swizzled K */
  f16   vt[2][128u * INK_R1_G01_NB];  /*  8,192 B: double-buffered XOR-swizzled V^T */
  f16   rb[2][48u * 16u];             /*  3,072 B: double-buffered rel_bias (48 dists x 16) */
  float rbsc[6][16][33];              /* 12,672 B: per-consumer-wave R*rel_bias scratch (stride 33 -> low conflict) */
} ink_r1_g01_lds_t;

static LDS ink_r1_g01_lds_t ink_r1_g01_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r1_g01_lds_t) <= 32768u, "ink_r1_g01_lds must fit in 32 KiB for 4 WGs/WGP");

/* Bijective bank-conflict-free XOR swizzles for hd=128, NB=16:
 *  - k_off(r, c):     row r (0..15), 8-half chunk c (0..15) -> 0 bank conflicts on load & store.
 *  - vt_off(col, h):  col (0..127), key-half h (0..1) -> bijection on 0..255, 0 conflicts on consumer read. */
static inline __attribute__((always_inline)) u32 ink_r1_g01_k_off(u32 r, u32 c) {
  return r * 128u + 8u * (c ^ r);
}

static inline __attribute__((always_inline)) u32 ink_r1_g01_vt_off(u32 col, u32 h) {
  return 8u * ((2u * col + h) ^ ((col >> 2u) & 1u) ^ ((col >> 3u) & 7u));
}

static inline __attribute__((always_inline)) float ink_r1_g01_xrow(float v) {
  return __builtin_bit_cast(float, xrow_i(__builtin_bit_cast(int, v)));
}

typedef struct {
  u32x4        kx[4]; /* K: 4 chunks of 8 halves (16 B each) per producer thread */
  ink_r1_g01_u2 vx[8]; /* V: 8 keys x 4 contiguous halves (2 column pairs = 8 B each) */
} ink_r1_g01_slot_t;

/* Transpose 8 keys of 2 f16 columns (packed as 8 u32) into two u32x4 (8 keys per column). */
static inline __attribute__((always_inline)) void ink_r1_g01_transpose8(const u32 *k, u32x4 *ev, u32x4 *od) {
  u32 e[4], o[4];
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    e[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x05040100u);
    o[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x07060302u);
  }
  *ev = (u32x4){e[0], e[1], e[2], e[3]};
  *od = (u32x4){o[0], o[1], o[2], o[3]};
}

/* Issue coalesced global K/V loads for block b (keys k_start + 16*b .. +15; keys >= kend zeroed). */
static inline __attribute__((always_inline)) void ink_r1_g01_load(
    ink_r1_g01_slot_t *s, const G u8 *kc, const G u8 *vc,
    u32 n_kv, u32 kh, u32 kv_len, u32 k_start, u32 kend, u32 b, u32 pt) {
  const u32 wi = pt >> 5u, L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
  const u32 kb = k_start + INK_R1_G01_NB * b;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r = 8u * wi + 2u * i + sub_r;
    const u32 p = kb + r;
    s->kx[i] = (u32x4)(0u);
    if (p < kend) {
      const u32 kvp = p % kv_len;
      s->kx[i] = *(const G u32x4 *)(kc + ((unsigned long)kvp * n_kv + kh) * INK_R1_G01_ROWB + 16u * c);
    }
  }
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const u32 p = kb + 8u * wi + u;
    s->vx[u] = (ink_r1_g01_u2)(0u);
    if (p < kend) {
      const u32 kvp = p % kv_len;
      s->vx[u] = *(const G ink_r1_g01_u2 *)(vc + ((unsigned long)kvp * n_kv + kh) * INK_R1_G01_ROWB + 8u * L);
    }
  }
}

/* Publish loaded K/V block and L1/L2-hot rel_bias window into LDS buffer bf. */
static inline __attribute__((always_inline)) void ink_r1_g01_publish(
    const ink_r1_g01_slot_t *s, const G float *rel_bias,
    u32 qpos_min, u32 extent_lim, u32 k_start, u32 b, u32 bf, u32 pt) {
  const u32 wi = pt >> 5u, L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r = 8u * wi + 2u * i + sub_r;
    *(LDSP u32x4 *)(ink_r1_g01_lds.k[bf] + ink_r1_g01_k_off(r, c)) = s->kx[i];
  }
  u32 vx0[8], vx1[8];
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    vx0[u] = s->vx[u].x;
    vx1[u] = s->vx[u].y;
  }
  u32x4 ev0, od0, ev1, od1;
  ink_r1_g01_transpose8(vx0, &ev0, &od0);
  ink_r1_g01_transpose8(vx1, &ev1, &od1);
  const u32 c0 = 4u * L;
  *(LDSP u32x4 *)(ink_r1_g01_lds.vt[bf] + ink_r1_g01_vt_off(c0 + 0u, wi)) = ev0;
  *(LDSP u32x4 *)(ink_r1_g01_lds.vt[bf] + ink_r1_g01_vt_off(c0 + 1u, wi)) = od0;
  *(LDSP u32x4 *)(ink_r1_g01_lds.vt[bf] + ink_r1_g01_vt_off(c0 + 2u, wi)) = ev1;
  *(LDSP u32x4 *)(ink_r1_g01_lds.vt[bf] + ink_r1_g01_vt_off(c0 + 3u, wi)) = od1;

  const u32 kb = k_start + INK_R1_G01_NB * b;
  const int d_min_s = (int)qpos_min - (int)(kb + (INK_R1_G01_NB - 1u));
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
    *(LDSP h8 *)(ink_r1_g01_lds.rb[bf] + pt * 16u + 0u) = r0;
    *(LDSP h8 *)(ink_r1_g01_lds.rb[bf] + pt * 16u + 8u) = r1;
  }
}

KERNEL ie_inkling_rel_attn_pf_r1(const G float *q, const G kvt *kc, const G kvt *vc,
                                 const G float *R, const G float *rel_bias,
                                 const G float *sink, G float *out,
                                 u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
                                 u32 d_rel, u32 rel_extent, u32 win,
                                 u32 qs, u32 rs, u32 ss, u32 os) {
  if (hd != 128u || d_rel != 16u || n_kv == 0u || n_head != n_kv * 3u || T == 0u || kv_len == 0u) return;

  const u32 ntile = (T + 31u) >> 5u;
  /* Descending longest-tile-first scheduling reduces causal tail imbalance. */
  const u32 tile = (ntile - 1u) - wgid();
  const u32 kh = tokid();
  const u32 t0 = tile << 5u;
  const u32 w = __builtin_amdgcn_readfirstlane(tid() >> 5u);
  const u32 l = lane(), hl = l >> 4u, rl = l & 15u;

  const u32 tlast = (t0 + 31u < T) ? (t0 + 31u) : (T - 1u);
  const u32 qpos_min = pos0 + t0;
  const u32 qpos_max = pos0 + tlast;
  const u32 k_start = (win > 0u && qpos_min + 1u > win) ? (qpos_min + 1u - win) : 0u;
  const u32 kend = qpos_max + 1u;
  const u32 nblk = (kend > k_start) ? ((kend - k_start + INK_R1_G01_NB - 1u) >> 4u) : 0u;
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;

  if (w >= 6u) { /* Producer waves 6, 7: 3-deep async K/V pipeline + rel_bias staging */
    const G u8 *kb8 = (const G u8 *)kc;
    const G u8 *vb8 = (const G u8 *)vc;
    const u32 pt = tid() - 192u;
    ink_r1_g01_slot_t ring[INK_R1_G01_D];
#pragma unroll
    for (u32 s = 0u; s < INK_R1_G01_D; s++)
      if (s < nblk) ink_r1_g01_load(&ring[s], kb8, vb8, n_kv, kh, kv_len, k_start, kend, s, pt);
    if (nblk > 0u) {
      ink_r1_g01_publish(&ring[0], rel_bias, qpos_min, extent_lim, k_start, 0u, 0u, pt);
      if (INK_R1_G01_D < nblk)
        ink_r1_g01_load(&ring[0], kb8, vb8, n_kv, kh, kv_len, k_start, kend, INK_R1_G01_D, pt);
    }
    for (u32 k0 = 0u; k0 < nblk; k0 += INK_R1_G01_D) {
#pragma unroll
      for (u32 j = 0u; j < INK_R1_G01_D; j++) {
        const u32 k = k0 + j;
        if (k < nblk) {
          lbarrier();
          if (k + 1u < nblk) {
            const u32 nxt = (j + 1u) % INK_R1_G01_D;
            ink_r1_g01_publish(&ring[nxt], rel_bias, qpos_min, extent_lim, k_start, k + 1u, (k + 1u) & 1u, pt);
            if (k + 1u + INK_R1_G01_D < nblk)
              ink_r1_g01_load(&ring[nxt], kb8, vb8, n_kv, kh, kv_len, k_start, kend, k + 1u + INK_R1_G01_D, pt);
          }
        }
      }
    }
    return;
  }

  /* Consumer waves 0..5: 2 sub-tiles of 16 tokens x 3 GQA query heads */
  const u32 tsub = w >= 3u ? 1u : 0u;
  const u32 hsub = w - 3u * tsub;
  const u32 t_sub0 = t0 + 16u * tsub;
  if (t_sub0 >= T) { /* Idle tail sub-tile (uniform across wave): participate in barriers only */
    for (u32 k = 0u; k < nblk; k++) lbarrier();
    return;
  }

  const u32 h = kh * 3u + hsub;
  const u32 tk = t_sub0 + rl;
  const u32 tq = tk < T ? tk : (T - 1u);
  const u32 tp = pos0 + tk;
  const u32 sub_qpos_min = pos0 + t_sub0;
  const u32 sub_qpos_max = pos0 + (t_sub0 + 15u);

  /* Preload Q (scaled by log2(e)/sqrt(128)) and R (scaled by log2(e)) into VGPRs (36 VGPRs total). */
  h8 qa[8];
  {
    const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (unsigned long)tq * qs) +
                                           (unsigned long)h * 128ul);
#pragma unroll
    for (u32 ks = 0u; ks < 8u; ks++) {
      const f32x4 v0 = qp4[4u * ks + 2u * hl + 0u];
      const f32x4 v1 = qp4[4u * ks + 2u * hl + 1u];
      qa[ks] = (h8){(f16)(v0.x * INK_R1_G01_QSC), (f16)(v0.y * INK_R1_G01_QSC),
                    (f16)(v0.z * INK_R1_G01_QSC), (f16)(v0.w * INK_R1_G01_QSC),
                    (f16)(v1.x * INK_R1_G01_QSC), (f16)(v1.y * INK_R1_G01_QSC),
                    (f16)(v1.z * INK_R1_G01_QSC), (f16)(v1.w * INK_R1_G01_QSC)};
    }
  }
  h8 ra;
  {
    const G f32x4 *rp4 = (const G f32x4 *)((const G float *)((const G u8 *)R + (unsigned long)tq * rs) +
                                           (unsigned long)h * 16ul);
    const f32x4 r0 = rp4[2u * hl + 0u];
    const f32x4 r1 = rp4[2u * hl + 1u];
    ra = (h8){(f16)(r0.x * INK_R1_G01_LOG2E), (f16)(r0.y * INK_R1_G01_LOG2E),
              (f16)(r0.z * INK_R1_G01_LOG2E), (f16)(r0.w * INK_R1_G01_LOG2E),
              (f16)(r1.x * INK_R1_G01_LOG2E), (f16)(r1.y * INK_R1_G01_LOG2E),
              (f16)(r1.z * INK_R1_G01_LOG2E), (f16)(r1.w * INK_R1_G01_LOG2E)};
  }

  /* Dynamic per-token, per-head sink initializes log2 online softmax (m = sink*log2(e), sum = 1 across partner pair). */
  const float sink_val = ((const G float *)((const G u8 *)sink + (unsigned long)tq * ss))[h];
  float m = sink_val * INK_R1_G01_LOG2E;
  float ls = (hl == 0u) ? 1.0f : 0.0f;

  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2u) & 1u) ^ (rl >> 3u);
  const v8f zero = (v8f)(0.0f);
  v8f o[8];
#pragma unroll
  for (u32 nt = 0u; nt < 8u; nt++) o[nt] = zero;

  for (u32 k = 0u; k < nblk; k++) {
    lbarrier();
    const u32 kb = k_start + INK_R1_G01_NB * k;
    const u32 bf = k & 1u;

    /* S^T = K * Q^T (16 keys x 16 tokens, 8 WMMAs across hd=128 in 2 independent chains). */
    v8f sa = zero, sb = zero;
#pragma unroll
    for (u32 ks = 0u; ks < 8u; ks += 2u) {
      const h8 k0 = *(const LDSP h8 *)(ink_r1_g01_lds.k[bf] + ink_r1_g01_k_off(rl, 2u * ks + hl));
      const h8 k1 = *(const LDSP h8 *)(ink_r1_g01_lds.k[bf] + ink_r1_g01_k_off(rl, 2u * (ks + 1u) + hl));
      sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[ks], sa);
      sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[ks + 1u], sb);
    }
    float s[8];
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) s[v] = sa[v] + sb[v];

    /* Relative position bias: 2 WMMAs cover all <= 31 distinct distances for this 16-token sub-tile. */
    const int d_min_sub_s = (int)sub_qpos_min - (int)(kb + (INK_R1_G01_NB - 1u));
    const u32 d_base_sub = d_min_sub_s > 0 ? (u32)d_min_sub_s : 0u;
    if (d_base_sub < extent_lim) {
      const int d_min_wg_s = (int)qpos_min - (int)(kb + (INK_R1_G01_NB - 1u));
      const u32 d_base_wg = d_min_wg_s > 0 ? (u32)d_min_wg_s : 0u;
      const u32 off = d_base_sub - d_base_wg;
      const h8 b0 = *(const LDSP h8 *)(ink_r1_g01_lds.rb[bf] + (off + rl) * 16u + 8u * hl);
      const h8 b1 = *(const LDSP h8 *)(ink_r1_g01_lds.rb[bf] + (off + 16u + rl) * 16u + 8u * hl);
      const v8f rb0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0, ra, zero);
      const v8f rb1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1, ra, zero);
      LDSP float *sc_row = ink_r1_g01_lds.rbsc[w][rl];
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

    /* Causal + SWA window mask (uniform branch skips interior unmasked blocks). */
    if (kb + (INK_R1_G01_NB - 1u) > sub_qpos_min || (win > 0u && sub_qpos_max >= kb + win)) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 kp = kb + 8u * hl + v;
        if (kp > tp || (win > 0u && (tp - kp) >= win)) s[v] = -__builtin_inff();
      }
    }

    float bm = s[0];
#pragma unroll
    for (u32 v = 1u; v < 8u; v++) bm = __builtin_fmaxf(bm, s[v]);
    bm = __builtin_fmaxf(bm, ink_r1_g01_xrow(bm));

    /* Exact zero-tile PV skip: also skips blocks fully masked out for this sub-tile. */
    if (__builtin_amdgcn_ballot_w32(!(bm - m < -30.0f)) == 0u) continue;

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
      for (u32 nt = 0u; nt < 8u; nt++)
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c;
    }

    /* O^T += V^T * P^T (8 WMMAs across hd=128, P stays in VGPRs, 0 LDS round-trip). */
    const LDSP f16 *vp0 = ink_r1_g01_lds.vt[bf] + 8u * vt_lane;
    const LDSP f16 *vp1 = ink_r1_g01_lds.vt[bf] + 256u + 8u * (vt_lane ^ 2u);
    const LDSP f16 *vp2 = ink_r1_g01_lds.vt[bf] + 512u + 8u * (vt_lane ^ 4u);
    const LDSP f16 *vp3 = ink_r1_g01_lds.vt[bf] + 768u + 8u * (vt_lane ^ 6u);
#pragma unroll
    for (u32 g = 0u; g < 2u; g++) {
      const h8 v0 = *(const LDSP h8 *)(vp0 + 1024u * g);
      const h8 v1 = *(const LDSP h8 *)(vp1 + 1024u * g);
      const h8 v2 = *(const LDSP h8 *)(vp2 + 1024u * g);
      const h8 v3 = *(const LDSP h8 *)(vp3 + 1024u * g);
      o[4u * g + 0u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[4u * g + 0u]);
      o[4u * g + 1u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[4u * g + 1u]);
      o[4u * g + 2u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v2, pb, o[4u * g + 2u]);
      o[4u * g + 3u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v3, pb, o[4u * g + 3u]);
      if (g == 0u) __asm__ volatile("" ::: "memory");
    }
  }

  /* Final normalization (wave-uniform xrow before tail mask) and coalesced f32x4 output stores. */
  const float inv = 1.0f / (ls + ink_r1_g01_xrow(ls));
  if (tk < T) {
    G f32x4 *op4 = (G f32x4 *)((G float *)((G u8 *)out + (unsigned long)tk * os) + (unsigned long)h * 128ul);
#pragma unroll
    for (u32 nt = 0u; nt < 8u; nt++) {
      const f32x4 a0 = {o[nt][0] * inv, o[nt][1] * inv, o[nt][2] * inv, o[nt][3] * inv};
      const f32x4 a1 = {o[nt][4] * inv, o[nt][5] * inv, o[nt][6] * inv, o[nt][7] * inv};
      op4[4u * nt + 2u * hl + 0u] = a0;
      op4[4u * nt + 2u * hl + 1u] = a1;
    }
  }
}

#undef INK_R1_G01_NB
#undef INK_R1_G01_D
#undef INK_R1_G01_LOG2E
#undef INK_R1_G01_QSC
#undef INK_R1_G01_ROWB
