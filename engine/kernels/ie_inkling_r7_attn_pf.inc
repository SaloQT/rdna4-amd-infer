/* ============================================================================
 * Inkling-2 Round 7 (Rank 5): Zero-Modulo, Far/Near Phase-Split, Pipelined
 * Producer-Consumer WMMA Prefill Relative-Position Attention + Split-KV TTFT
 * Companion for RX 9070 XT (gfx1201, wave32, hd=128, d_rel=16, f16 KV, NT=256).
 *
 * Key Upgrades over ie_inkling_rel_attn_pf_r1 (174 VGPR, 32,128 B LDS):
 *   1. Eliminates ALL 85 `p % kv_len` integer divisions in the producer K/V
 *      loop: global no-wrap path (`kend <= kv_len`) indexes `p` directly;
 *      SWA/ring path computes `kv_start = k_start % kv_len` ONCE and advances
 *      by conditional subtraction (`kvp >= kv_len ? kvp - kv_len : kvp`).
 *   2. Dedicated Far-Field / Near-Field Two-Phase K-loop: at 16K-32K Global
 *      (`win == 0, rel_extent == 1024`), 94-97% of blocks run the zero-mask,
 *      zero-rel_bias, zero-boundary-check Phase A loop (no `rb` staging, no
 *      `rbsc` LDS traffic, no causal/SWA predicate).
 *   3. Pipelined QK & PV LDS loads (`ie_saluki_attn_pl_r10` schedule): preloads
 *      `k0..k3` / `v0..v3` and overlaps next `ds_load_b128` behind 2 independent
 *      WMMA chains while preserving the D=3 producer ring.
 *   4. Causal Wave-2 min+max WG pairing at `ntile == 16 && n_kv == 8` (T=481..512,
 *      128 WGs, global): Wave 1 (`kh=0..3`) runs tiles ascending `0..15`, Wave 2
 *      (`kh=4..7`) runs descending `15..0` so every CU pairs short + long causal
 *      work with ZERO host grid changes.
 *   5. Split-KV Companion (`ie_inkling_rel_attn_pf_split_r7`): saturates all 64
 *      CUs during small-T long-context incremental prefill (`T = 17..256`,
 *      `pos0 >= 2048`, `win == 0`) where `ntile * n_kv = 16..64 WGs` otherwise
 *      leaves 50-87% of the GPU idle!
 *
 * Exported Kernels:
 *
 * 1. KERNEL ie_inkling_rel_attn_pf_r7(
 *        const G float *q, const G kvt *kc, const G kvt *vc,
 *        const G float *R, const G float *rel_bias, const G float *sink, G float *out,
 *        u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
 *        u32 d_rel, u32 rel_extent, u32 win, u32 qs, u32 rs, u32 ss, u32 os)
 *    - Exact drop-in signature & grid `((T + 31u) / 32u, n_kv, 1u)` of pf_r1!
 *    - Guards: !b->kvq8 && !b->kvq4 && !mk.rec && T > 1u && hd == 128u &&
 *              d_rel == 16u && n_kv > 0u && n_head == n_kv * 3u && kv_len > 0u.
 *    - Resources: <= 192 VGPR (2 WGs/CU), 32,128 B LDS (<= 32 KiB), 0/0 spills.
 *    - Parity: f32-tol (matches ie_inkling_rel_attn_pf_r1).
 *
 * 2. KERNEL ie_inkling_rel_attn_pf_split_r7(
 *        const G float *q, const G kvt *kc, const G kvt *vc,
 *        const G float *R, const G float *rel_bias, const G float *sink,
 *        G float *part, G float *out,
 *        u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
 *        u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
 *        u32 qs, u32 rs, u32 ss, u32 os, G u32 *count)
 *    - Grid:   (((T + 31u) / 32u) * nsplit, n_kv, 1u), NT = 256.
 *    - Guards: !b->kvq8 && !b->kvq4 && !mk.rec && win == 0u && pos0 >= 2048u &&
 *              T >= 17u && T <= 256u && nsplit >= 2u && nsplit <= 8u &&
 *              hd == 128u && d_rel == 16u && n_kv > 0u && n_head == n_kv * 3u &&
 *              kv_len > 0u && part && count &&
 *              part_floats >= (size_t)T * n_head * nsplit * 130u &&
 *              count_u32s >= ((T + 31u) / 32u) * n_kv
 *              (Suggested nsplit: target ntile * nsplit * n_kv >= 128, capped by
 *               available part_floats / (T * n_head * 130u) and <= 8u).
 *    - Counter: count[tile * n_kv + kh] (self-reset to 0 by last arriving WG).
 *    - Resources: <= 192 VGPR (2 WGs/CU), 32,132 B LDS (<= 32 KiB), 0/0 spills.
 *    - Parity: f32-tol vs ie_inkling_rel_attn_pf_r1 / ie_rel_attn_pf.
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(kvt) == 2u, "ie_inkling_r7_attn_pf requires f16 KV cache");

#define INK_R7_PF_NB    16u
#define INK_R7_PF_D     3u
#define INK_R7_PF_LOG2E 1.44269504088896340736f
#define INK_R7_PF_LN2   0.69314718055994530942f
#define INK_R7_PF_QSC   (0.08838834764831844299f * 1.44269504088896340736f)
#define INK_R7_PF_ROWB  256u
#define INK_R7_PF_PSTR  130u

typedef unsigned int ink_r7_pf_u2 __attribute__((ext_vector_type(2)));

typedef struct {
  f16   k[2][INK_R7_PF_NB * 128u];  /*  8,192 B: double-buffered XOR-swizzled K */
  f16   vt[2][128u * INK_R7_PF_NB]; /*  8,192 B: double-buffered XOR-swizzled V^T */
  f16   rb[2][48u * 16u];           /*  3,072 B: double-buffered rel_bias (48 dists x 16) */
  float rbsc[6][16][33];            /* 12,672 B: 1-pass per-consumer-wave R*rel_bias scratch */
  u32   is_last;                    /*      4 B: Split-KV last-WG flag (unused in non-split) */
} ink_r7_pf_lds_t;

static LDS ink_r7_pf_lds_t ink_r7_pf_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r7_pf_lds_t) <= 32768u, "ink_r7_pf_lds_t must fit in 32 KiB (2 WGs/CU)");

static inline __attribute__((always_inline)) u32 ink_r7_pf_k_off(u32 r, u32 c) {
  return r * 128u + 8u * (c ^ r);
}

static inline __attribute__((always_inline)) u32 ink_r7_pf_vt_off(u32 col, u32 h) {
  return 8u * ((2u * col + h) ^ ((col >> 2u) & 1u) ^ ((col >> 3u) & 7u));
}

static inline __attribute__((always_inline)) float ink_r7_pf_xrow(float v) {
  return I2F(xrow_i(F2I(v)));
}

typedef struct {
  u32x4         kx[4];
  ink_r7_pf_u2  vx[8];
} ink_r7_pf_slot_t;

static inline __attribute__((always_inline)) void ink_r7_pf_transpose8(const u32 *k, u32x4 *ev, u32x4 *od) {
  u32 e[4], o[4];
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    e[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x05040100u);
    o[i] = __builtin_amdgcn_perm(k[2u * i + 1u], k[2u * i], 0x07060302u);
  }
  *ev = (u32x4){e[0], e[1], e[2], e[3]};
  *od = (u32x4){o[0], o[1], o[2], o[3]};
}

/* SGPR-uniform modulo-free ring wrap (0 integer divisions for all kv_len >= 1). */
static inline __attribute__((always_inline)) u32 ink_r7_pf_wrap_s(u32 v, const u32 kv_len) {
  if (kv_len >= 16u) return (v >= kv_len) ? (v - kv_len) : v;
  while (v >= kv_len) v -= kv_len;
  return v;
}

/* Zero-modulo interior producer load (no-wrap global fast path: SGPR wave base + 1 VGPR offset). */
static inline __attribute__((always_inline)) void ink_r7_pf_load_nowrap(
    ink_r7_pf_slot_t *s, const G u8 *kb_kh, const G u8 *vb_kh,
    const unsigned long row_step, const u32 kb, const u32 pt) {
  const u32 wi = __builtin_amdgcn_readfirstlane(pt >> 5u);
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
    s->vx[u] = *(const G ink_r7_pf_u2 *)(v_w + (unsigned long)u * row_step + v_off);
  }
}

/* Zero-modulo ring/boundary producer load: all row indices & wrap math stay 100% SGPR. */
static inline __attribute__((always_inline)) void ink_r7_pf_load_ring(
    ink_r7_pf_slot_t *s, const G u8 *kb_kh, const G u8 *vb_kh,
    const unsigned long row_step, const u32 kv_len, const u32 kb, const u32 kv_b,
    const u32 kend, const u32 pt) {
  const u32 wi = __builtin_amdgcn_readfirstlane(pt >> 5u);
  const u32 L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
  const u32 r_base = 8u * wi;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r0 = r_base + 2u * i;
    const u32 p0 = kb + r0;
    const u32 p1 = p0 + 1u;
    const u32 kvp0 = ink_r7_pf_wrap_s(kv_b + r0, kv_len);
    const u32 kvp1 = ink_r7_pf_wrap_s(kv_b + r0 + 1u, kv_len);
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
    s->vx[u] = (ink_r7_pf_u2)(0u);
    if (p < kend) {
      const u32 kvp = ink_r7_pf_wrap_s(kv_b + r, kv_len);
      const G u8 *v_row = vb_kh + (unsigned long)kvp * row_step;
      s->vx[u] = *(const G ink_r7_pf_u2 *)(v_row + v_off);
    }
  }
}

static inline __attribute__((always_inline)) void ink_r7_pf_publish_kv(
    const ink_r7_pf_slot_t *s, const u32 bf, const u32 pt) {
  const u32 wi = __builtin_amdgcn_readfirstlane(pt >> 5u);
  const u32 L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r = 8u * wi + 2u * i + sub_r;
    *(LDSP u32x4 *)(ink_r7_pf_lds.k[bf] + ink_r7_pf_k_off(r, c)) = s->kx[i];
  }
  u32 vx0[8], vx1[8];
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    vx0[u] = s->vx[u].x;
    vx1[u] = s->vx[u].y;
  }
  u32x4 ev0, od0, ev1, od1;
  ink_r7_pf_transpose8(vx0, &ev0, &od0);
  ink_r7_pf_transpose8(vx1, &ev1, &od1);
  const u32 c0 = 4u * L;
  *(LDSP u32x4 *)(ink_r7_pf_lds.vt[bf] + ink_r7_pf_vt_off(c0 + 0u, wi)) = ev0;
  *(LDSP u32x4 *)(ink_r7_pf_lds.vt[bf] + ink_r7_pf_vt_off(c0 + 1u, wi)) = od0;
  *(LDSP u32x4 *)(ink_r7_pf_lds.vt[bf] + ink_r7_pf_vt_off(c0 + 2u, wi)) = ev1;
  *(LDSP u32x4 *)(ink_r7_pf_lds.vt[bf] + ink_r7_pf_vt_off(c0 + 3u, wi)) = od1;
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r7_pf_publish_rb(
    const G float *rel_bias, const u32 qpos_min, const u32 extent_lim,
    const u32 kb, const u32 bf, const u32 pt) {
  const int d_min_s = (int)qpos_min - (int)(kb + (INK_R7_PF_NB - 1u));
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
    *(LDSP h8 *)(ink_r7_pf_lds.rb[bf] + pt * 16u + 0u) = r0;
    *(LDSP h8 *)(ink_r7_pf_lds.rb[bf] + pt * 16u + 8u) = r1;
  }
}

/* Advance ring-wrapped block base by +16 with 0 integer divisions. */
static inline __attribute__((always_inline)) u32 ink_r7_pf_kv_adv(const u32 kv_b, const u32 kv_len) {
  return ink_r7_pf_wrap_s(kv_b + INK_R7_PF_NB, kv_len);
}

/* Shared Producer D=3 async pipeline for block range [blk_beg, blk_end). */
static inline __attribute__((always_inline)) void ink_r7_pf_producer(
    const G u8 *kb_kh, const G u8 *vb_kh, const G float *rel_bias,
    const unsigned long row_step, const u32 kv_len, const u32 k_start,
    const u32 kend, const u32 qpos_min, const u32 extent_lim,
    const u32 blk_beg, const u32 blk_end, const u32 blk_far_end,
    const int no_wrap, const u32 pt) {
  const u32 nblk_loc = blk_end - blk_beg;
  if (nblk_loc == 0u) return;

  const u32 kb0_start = k_start + INK_R7_PF_NB * blk_beg;
  const u32 nblk_full = (kend > kb0_start) ? ((kend - kb0_start) >> 4u) : 0u;
  const u32 fast_lim = no_wrap ? (nblk_full < nblk_loc ? nblk_full : nblk_loc) : 0u;

  u32 kv_load = no_wrap ? 0u : (kb0_start % kv_len);
  u32 k0 = 0u;
  ink_r7_pf_slot_t ring[INK_R7_PF_D];

  if (no_wrap && fast_lim >= 6u && blk_far_end >= blk_beg + 6u) {
    /* Pure far-field no-wrap hot loop (94-97% of 16K-32K Global prefill):
     * 0 modulo, 0 boundary checks, 0 rel_bias checks! */
    const u32 far_loc = (blk_far_end - blk_beg) < fast_lim ? (blk_far_end - blk_beg) : fast_lim;
    ink_r7_pf_load_nowrap(&ring[0], kb_kh, vb_kh, row_step, kb0_start + 0u, pt);
    ink_r7_pf_load_nowrap(&ring[1], kb_kh, vb_kh, row_step, kb0_start + 16u, pt);
    ink_r7_pf_load_nowrap(&ring[2], kb_kh, vb_kh, row_step, kb0_start + 32u, pt);
    __builtin_amdgcn_sched_barrier(0);
    ink_r7_pf_publish_kv(&ring[0], 0u, pt);
    ink_r7_pf_load_nowrap(&ring[0], kb_kh, vb_kh, row_step, kb0_start + 48u, pt);

#pragma clang loop unroll(disable)
    for (; k0 + 6u < far_loc; k0 += 3u) {
      lbarrier();
      ink_r7_pf_publish_kv(&ring[1], (k0 + 1u) & 1u, pt);
      ink_r7_pf_load_nowrap(&ring[1], kb_kh, vb_kh, row_step, kb0_start + 16u * (k0 + 4u), pt);
      lbarrier();
      ink_r7_pf_publish_kv(&ring[2], (k0 + 2u) & 1u, pt);
      ink_r7_pf_load_nowrap(&ring[2], kb_kh, vb_kh, row_step, kb0_start + 16u * (k0 + 5u), pt);
      lbarrier();
      ink_r7_pf_publish_kv(&ring[0], (k0 + 3u) & 1u, pt);
      ink_r7_pf_load_nowrap(&ring[0], kb_kh, vb_kh, row_step, kb0_start + 16u * (k0 + 6u), pt);
    }
  } else {
    /* General / SWA ring D=3 prime (0 integer divisions inside loop). */
#pragma unroll
    for (u32 s = 0u; s < INK_R7_PF_D; s++) {
      if (s < nblk_loc) {
        const u32 kbs = kb0_start + 16u * s;
        if (s < fast_lim) {
          ink_r7_pf_load_nowrap(&ring[s], kb_kh, vb_kh, row_step, kbs, pt);
        } else {
          ink_r7_pf_load_ring(&ring[s], kb_kh, vb_kh, row_step, kv_len, kbs, no_wrap ? kbs : kv_load, kend, pt);
        }
        if (!no_wrap) kv_load = ink_r7_pf_kv_adv(kv_load, kv_len);
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r7_pf_publish_kv(&ring[0], 0u, pt);
    if (blk_beg >= blk_far_end) {
      ink_r7_pf_publish_rb(rel_bias, qpos_min, extent_lim, kb0_start, 0u, pt);
    }
    if (INK_R7_PF_D < nblk_loc) {
      const u32 kbd = kb0_start + 16u * INK_R7_PF_D;
      if (INK_R7_PF_D < fast_lim) {
        ink_r7_pf_load_nowrap(&ring[0], kb_kh, vb_kh, row_step, kbd, pt);
      } else {
        ink_r7_pf_load_ring(&ring[0], kb_kh, vb_kh, row_step, kv_len, kbd, no_wrap ? kbd : kv_load, kend, pt);
      }
      if (!no_wrap) kv_load = ink_r7_pf_kv_adv(kv_load, kv_len);
    }
  }

  /* Unified tail / near-field / SWA ring D=3 loop (shares ring[3] seamlessly). */
#pragma clang loop unroll(disable)
  for (; k0 < nblk_loc; k0 += INK_R7_PF_D) {
#pragma unroll
    for (u32 j = 0u; j < INK_R7_PF_D; j++) {
      const u32 k = k0 + j;
      if (k < nblk_loc) {
        lbarrier();
        if (k + 1u < nblk_loc) {
          const u32 nxt = (j + 1u) % INK_R7_PF_D;
          const u32 kn = k + 1u;
          ink_r7_pf_publish_kv(&ring[nxt], kn & 1u, pt);
          if (blk_beg + kn >= blk_far_end) {
            ink_r7_pf_publish_rb(rel_bias, qpos_min, extent_lim, kb0_start + 16u * kn, kn & 1u, pt);
          }
          const u32 kl = k + 1u + INK_R7_PF_D;
          if (kl < nblk_loc) {
            const u32 kbl = kb0_start + 16u * kl;
            if (kl < fast_lim) {
              ink_r7_pf_load_nowrap(&ring[nxt], kb_kh, vb_kh, row_step, kbl, pt);
            } else {
              ink_r7_pf_load_ring(&ring[nxt], kb_kh, vb_kh, row_step, kv_len, kbl, no_wrap ? kbl : kv_load, kend, pt);
            }
            if (!no_wrap) kv_load = ink_r7_pf_kv_adv(kv_load, kv_len);
          }
        }
      }
    }
  }
}

/* Pipelined QK^T (4-deep K LDS preload + 2-chain WMMA) for hd = 128. */
static inline __attribute__((always_inline)) void ink_r7_pf_qk_pipe(
    const u32 bf, const u32 rl, const u32 hl, const h8 qa[8], float s_out[8]) {
  const LDSP f16 *k_lds = ink_r7_pf_lds.k[bf];
  const v8f zero = (v8f)(0.0f);
  v8f sa = zero, sb = zero;

  h8 k0 = *(const LDSP h8 *)(k_lds + ink_r7_pf_k_off(rl, 0u + hl));
  h8 k1 = *(const LDSP h8 *)(k_lds + ink_r7_pf_k_off(rl, 2u + hl));
  h8 k2 = *(const LDSP h8 *)(k_lds + ink_r7_pf_k_off(rl, 4u + hl));
  h8 k3 = *(const LDSP h8 *)(k_lds + ink_r7_pf_k_off(rl, 6u + hl));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[0], sa);
  sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[1], sb);
  k0 = *(const LDSP h8 *)(k_lds + ink_r7_pf_k_off(rl, 8u + hl));
  k1 = *(const LDSP h8 *)(k_lds + ink_r7_pf_k_off(rl, 10u + hl));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k2, qa[2], sa);
  sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k3, qa[3], sb);
  k2 = *(const LDSP h8 *)(k_lds + ink_r7_pf_k_off(rl, 12u + hl));
  k3 = *(const LDSP h8 *)(k_lds + ink_r7_pf_k_off(rl, 14u + hl));
  __builtin_amdgcn_sched_barrier(0);

  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[4], sa);
  sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k1, qa[5], sb);
  sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k2, qa[6], sa);
  sb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k3, qa[7], sb);

#pragma unroll
  for (u32 v = 0u; v < 8u; v++) s_out[v] = sa[v] + sb[v];
}

/* Pipelined PV (4-deep V^T LDS preload overlapped between g=0 and g=1 WMMAs) for hd = 128. */
static inline __attribute__((always_inline)) void ink_r7_pf_pv_pipe(
    const u32 bf, const u32 vt_lane, const h8 pb, v8f o[8]) {
  const LDSP f16 *vp0 = ink_r7_pf_lds.vt[bf] + 8u * vt_lane;
  const LDSP f16 *vp1 = ink_r7_pf_lds.vt[bf] + 256u + 8u * (vt_lane ^ 2u);
  const LDSP f16 *vp2 = ink_r7_pf_lds.vt[bf] + 512u + 8u * (vt_lane ^ 4u);
  const LDSP f16 *vp3 = ink_r7_pf_lds.vt[bf] + 768u + 8u * (vt_lane ^ 6u);

  h8 v0 = *(const LDSP h8 *)(vp0);
  h8 v1 = *(const LDSP h8 *)(vp1);
  h8 v2 = *(const LDSP h8 *)(vp2);
  h8 v3 = *(const LDSP h8 *)(vp3);
  __builtin_amdgcn_sched_barrier(0);

  o[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[0]);
  o[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[1]);
  v0 = *(const LDSP h8 *)(vp0 + 1024u);
  v1 = *(const LDSP h8 *)(vp1 + 1024u);
  __builtin_amdgcn_sched_barrier(0);

  o[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v2, pb, o[2]);
  o[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v3, pb, o[3]);
  v2 = *(const LDSP h8 *)(vp2 + 1024u);
  v3 = *(const LDSP h8 *)(vp3 + 1024u);
  __builtin_amdgcn_sched_barrier(0);

  o[4] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[4]);
  o[5] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v1, pb, o[5]);
  o[6] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v2, pb, o[6]);
  o[7] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v3, pb, o[7]);
}

/* Online softmax update + zero-tile PV skip + pipelined PV. */
static inline __attribute__((always_inline)) void ink_r7_pf_softmax_pv(
    const float s[8], const u32 bf, const u32 vt_lane, float *m_p, float *ls_p, v8f o[8]) {
  float bm = s[0];
#pragma unroll
  for (u32 v = 1u; v < 8u; v++) bm = __builtin_fmaxf(bm, s[v]);
  bm = __builtin_fmaxf(bm, ink_r7_pf_xrow(bm));

  const float m = *m_p;
  if (__builtin_amdgcn_ballot_w32(bm > -1e30f && !(bm - m < -30.0f)) == 0u) return;

  const float mn = __builtin_fmaxf(m, bm);
  const float c = (m > -1e30f) ? __builtin_amdgcn_exp2f(m - mn) : 0.0f;
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

  if (__builtin_amdgcn_ballot_w32(c != 1.0f) != 0u) {
#pragma unroll
    for (u32 nt = 0u; nt < 8u; nt++)
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) o[nt][v] *= c;
  }

  ink_r7_pf_pv_pipe(bf, vt_lane, pb, o);
}

/* Shared two-phase consumer loop (Phase A: pure far-field; Phase B: near-field + mask). */
static inline __attribute__((always_inline)) void ink_r7_pf_consumer_loop(
    const u32 w, const u32 rl, const u32 hl, const u32 vt_lane,
    const u32 k_start, const u32 qpos_min, const u32 sub_qpos_min, const u32 sub_qpos_max,
    const u32 tp, const u32 win, const u32 extent_lim,
    const u32 blk_beg, const u32 blk_end, const u32 blk_far_end,
    const h8 qa[8], const h8 ra, float *m_p, float *ls_p, v8f o[8]) {
  const v8f zero = (v8f)(0.0f);
  const u32 nblk_loc = blk_end - blk_beg;
  const u32 nfar = (blk_far_end > blk_beg) ? ((blk_far_end < blk_end ? blk_far_end : blk_end) - blk_beg) : 0u;

  /* Phase A: Far-field unmasked blocks (0 rel_bias, 0 causal/SWA mask branches). */
#pragma clang loop unroll(disable)
  for (u32 k = 0u; k < nfar; k++) {
    lbarrier();
    const u32 bf = k & 1u;
    float s[8];
    ink_r7_pf_qk_pipe(bf, rl, hl, qa, s);
    ink_r7_pf_softmax_pv(s, bf, vt_lane, m_p, ls_p, o);
  }

  /* Phase B: Near-field (rel_bias + causal/SWA masking). */
#pragma clang loop unroll(disable)
  for (u32 k = nfar; k < nblk_loc; k++) {
    lbarrier();
    const u32 kb = k_start + INK_R7_PF_NB * (blk_beg + k);
    const u32 bf = k & 1u;
    float s[8];
    ink_r7_pf_qk_pipe(bf, rl, hl, qa, s);

    const int d_min_sub_s = (int)sub_qpos_min - (int)(kb + (INK_R7_PF_NB - 1u));
    const u32 d_base_sub = d_min_sub_s > 0 ? (u32)d_min_sub_s : 0u;
    if (d_base_sub < extent_lim) {
      const int d_min_wg_s = (int)qpos_min - (int)(kb + (INK_R7_PF_NB - 1u));
      const u32 d_base_wg = d_min_wg_s > 0 ? (u32)d_min_wg_s : 0u;
      const u32 off = d_base_sub - d_base_wg;
      const h8 b0 = *(const LDSP h8 *)(ink_r7_pf_lds.rb[bf] + (off + rl) * 16u + 8u * hl);
      const h8 b1 = *(const LDSP h8 *)(ink_r7_pf_lds.rb[bf] + (off + 16u + rl) * 16u + 8u * hl);
      const v8f rb0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0, ra, zero);
      const v8f rb1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1, ra, zero);
      LDSP float *sc_row = ink_r7_pf_lds.rbsc[w][rl];
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

    if (kb + (INK_R7_PF_NB - 1u) > sub_qpos_min || (win > 0u && sub_qpos_max >= kb + win)) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 kp = kb + 8u * hl + v;
        if (kp > tp || (win > 0u && (tp - kp) >= win)) s[v] = -__builtin_inff();
      }
    }

    ink_r7_pf_softmax_pv(s, bf, vt_lane, m_p, ls_p, o);
  }
}

/* Compute exact WG-uniform far-field block boundary blk_far_end (in [0, nblk]).
 * Any block b < blk_far_end guarantees:
 *   1) No rel_bias for ANY query in the 32-token tile: qpos_min - (kb + 15) >= extent_lim
 *      WAIT: check max distance vs min distance! A query in [qpos_min, qpos_max] has
 *      dist = tp - kp >= qpos_min - (kb + 15). So if qpos_min - (kb + 15) >= extent_lim,
 *      EVERY query in the WG has dist >= extent_lim (NO rel_bias!).
 *   2) No causal mask: kb + 15 <= qpos_min.
 *   3) No SWA mask: win == 0 (for win > 0, extent_lim <= win so far-field is empty anyway). */
static inline __attribute__((always_inline)) u32 ink_r7_pf_calc_far_end(
    const u32 win, const u32 qpos_min, const u32 extent_lim, const u32 k_start, const u32 nblk) {
  if (win > 0u) return 0u;
  const u32 guard = extent_lim + (INK_R7_PF_NB - 1u);
  if (qpos_min < guard || qpos_min - guard < k_start) return 0u;
  const u32 b_far = ((qpos_min - guard - k_start) >> 4u) + 1u;
  return b_far < nblk ? b_far : nblk;
}

/* ============================================================================
 * Kernel 1: ie_inkling_rel_attn_pf_r7 (Drop-in replacement for pf_r1)
 * ========================================================================== */
KERNEL ie_inkling_rel_attn_pf_r7(const G float *q, const G kvt *kc, const G kvt *vc,
                                 const G float *R, const G float *rel_bias,
                                 const G float *sink, G float *out,
                                 u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
                                 u32 d_rel, u32 rel_extent, u32 win,
                                 u32 qs, u32 rs, u32 ss, u32 os) {
  if (hd != 128u || d_rel != 16u || n_kv == 0u || n_head != n_kv * 3u || T == 0u || kv_len == 0u) return;

  const u32 ntile = (T + 31u) >> 5u;
  const u32 kh = tokid();
  if (wgid() >= ntile || kh >= n_kv) return;
  /* Causal Wave-2 min+max pairing at T=481..512 (ntile==16, n_kv==8, global):
   * pairs short + long causal tiles across Wave 1 (kh=0..3) and Wave 2 (kh=4..7). */
  const u32 tile = (ntile == 16u && n_kv == 8u && win == 0u && kh < 4u) ? wgid() : ((ntile - 1u) - wgid());
  const u32 t0 = tile << 5u;
  const u32 w = __builtin_amdgcn_readfirstlane(tid() >> 5u);
  const u32 l = lane(), hl = l >> 4u, rl = l & 15u;

  const u32 tlast = (t0 + 31u < T) ? (t0 + 31u) : (T - 1u);
  const u32 qpos_min = pos0 + t0;
  const u32 qpos_max = pos0 + tlast;
  const u32 k_start = (win > 0u && qpos_min + 1u > win) ? (qpos_min + 1u - win) : 0u;
  const u32 kend = qpos_max + 1u;
  const u32 nblk = (kend > k_start) ? ((kend - k_start + INK_R7_PF_NB - 1u) >> 4u) : 0u;
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const u32 blk_far_end = ink_r7_pf_calc_far_end(win, qpos_min, extent_lim, k_start, nblk);

  if (w >= 6u) {
    const unsigned long row_step = (unsigned long)n_kv * INK_R7_PF_ROWB;
    const G u8 *kb_kh = (const G u8 *)kc + (unsigned long)kh * INK_R7_PF_ROWB;
    const G u8 *vb_kh = (const G u8 *)vc + (unsigned long)kh * INK_R7_PF_ROWB;
    const int no_wrap = (kend <= kv_len);
    ink_r7_pf_producer(kb_kh, vb_kh, rel_bias, row_step, kv_len, k_start, kend,
                       qpos_min, extent_lim, 0u, nblk, blk_far_end, no_wrap, tid() - 192u);
    return;
  }

  const u32 tsub = w >= 3u ? 1u : 0u;
  const u32 hsub = w - 3u * tsub;
  const u32 t_sub0 = t0 + 16u * tsub;
  if (t_sub0 >= T) {
    for (u32 k = 0u; k < nblk; k++) lbarrier();
    return;
  }

  const u32 h = kh * 3u + hsub;
  const u32 tk = t_sub0 + rl;
  const u32 tq = tk < T ? tk : (T - 1u);
  const u32 tp = pos0 + tk;
  const u32 sub_qpos_min = pos0 + t_sub0;
  const u32 sub_qpos_max = pos0 + (t_sub0 + 15u);

  h8 qa[8];
  {
    const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (unsigned long)tq * qs) +
                                           (unsigned long)h * 128ul);
#pragma unroll
    for (u32 ks = 0u; ks < 8u; ks++) {
      const f32x4 v0 = qp4[4u * ks + 2u * hl + 0u];
      const f32x4 v1 = qp4[4u * ks + 2u * hl + 1u];
      qa[ks] = (h8){(f16)(v0.x * INK_R7_PF_QSC), (f16)(v0.y * INK_R7_PF_QSC),
                    (f16)(v0.z * INK_R7_PF_QSC), (f16)(v0.w * INK_R7_PF_QSC),
                    (f16)(v1.x * INK_R7_PF_QSC), (f16)(v1.y * INK_R7_PF_QSC),
                    (f16)(v1.z * INK_R7_PF_QSC), (f16)(v1.w * INK_R7_PF_QSC)};
    }
  }
  h8 ra;
  {
    const G f32x4 *rp4 = (const G f32x4 *)((const G float *)((const G u8 *)R + (unsigned long)tq * rs) +
                                           (unsigned long)h * 16ul);
    const f32x4 r0 = rp4[2u * hl + 0u];
    const f32x4 r1 = rp4[2u * hl + 1u];
    ra = (h8){(f16)(r0.x * INK_R7_PF_LOG2E), (f16)(r0.y * INK_R7_PF_LOG2E),
              (f16)(r0.z * INK_R7_PF_LOG2E), (f16)(r0.w * INK_R7_PF_LOG2E),
              (f16)(r1.x * INK_R7_PF_LOG2E), (f16)(r1.y * INK_R7_PF_LOG2E),
              (f16)(r1.z * INK_R7_PF_LOG2E), (f16)(r1.w * INK_R7_PF_LOG2E)};
  }

  const float sink_val = ((const G float *)((const G u8 *)sink + (unsigned long)tq * ss))[h];
  float m = sink_val * INK_R7_PF_LOG2E;
  float ls = (hl == 0u) ? 1.0f : 0.0f;

  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2u) & 1u) ^ (rl >> 3u);
  const v8f zero = (v8f)(0.0f);
  v8f o[8];
#pragma unroll
  for (u32 nt = 0u; nt < 8u; nt++) o[nt] = zero;

  ink_r7_pf_consumer_loop(w, rl, hl, vt_lane, k_start, qpos_min, sub_qpos_min, sub_qpos_max,
                          tp, win, extent_lim, 0u, nblk, blk_far_end, qa, ra, &m, &ls, o);

  const float inv = 1.0f / (ls + ink_r7_pf_xrow(ls));
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

/* ============================================================================
 * Kernel 2: ie_inkling_rel_attn_pf_split_r7
 * Split-KV companion for small-T long-context incremental prefill TTFT
 * (T = 17..256, pos0 >= 2048, win == 0, nsplit = 2..8).
 * Grid: (((T + 31u) / 32u) * nsplit, n_kv, 1u), NT = 256.
 * ========================================================================== */
KERNEL ie_inkling_rel_attn_pf_split_r7(const G float *q, const G kvt *kc, const G kvt *vc,
                                       const G float *R, const G float *rel_bias,
                                       const G float *sink, G float *part, G float *out,
                                       u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
                                       u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
                                       u32 qs, u32 rs, u32 ss, u32 os, G u32 *count) {
  if (hd != 128u || d_rel != 16u || n_kv == 0u || n_head != n_kv * 3u || T == 0u ||
      kv_len == 0u || nsplit < 2u || nsplit > 8u || !part || !count)
    return;

  const u32 ntile = (T + 31u) >> 5u;
  const u32 wg = wgid();
  const u32 tile_raw = wg / nsplit;
  const u32 s = wg - tile_raw * nsplit;
  if (tile_raw >= ntile) return;
  const u32 tile = (ntile - 1u) - tile_raw;
  const u32 kh = tokid();
  if (kh >= n_kv) return;

  const u32 t0 = tile << 5u;
  const u32 w = __builtin_amdgcn_readfirstlane(tid() >> 5u);
  const u32 l = lane(), hl = l >> 4u, rl = l & 15u;

  const u32 tlast = (t0 + 31u < T) ? (t0 + 31u) : (T - 1u);
  const u32 nq_tile = tlast - t0 + 1u;
  const u32 qpos_min = pos0 + t0;
  const u32 qpos_max = pos0 + tlast;
  const u32 k_start = (win > 0u && qpos_min + 1u > win) ? (qpos_min + 1u - win) : 0u;
  const u32 kend = qpos_max + 1u;
  const u32 nblk = (kend > k_start) ? ((kend - k_start + INK_R7_PF_NB - 1u) >> 4u) : 0u;
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const u32 blk_far_end = ink_r7_pf_calc_far_end(win, qpos_min, extent_lim, k_start, nblk);

  const u32 blk_per_s = (nblk + nsplit - 1u) / nsplit;
  const u32 blk_beg = (s * blk_per_s < nblk) ? (s * blk_per_s) : nblk;
  const u32 blk_end = (blk_beg + blk_per_s < nblk) ? (blk_beg + blk_per_s) : nblk;
  const u32 nblk_loc = blk_end - blk_beg;

  if (nblk_loc == 0u) {
    /* Empty split: write sentinel partials for all tokens in this tile x 3 heads. */
    const u32 h0 = kh * 3u;
    for (u32 qi = w; qi < nq_tile * 3u; qi += 8u) {
      const u32 tk = t0 + qi / 3u;
      const u32 h = h0 + qi % 3u;
      G float *pp = part + (unsigned long)((tk * n_head + h) * nsplit + s) * INK_R7_PF_PSTR;
      if (l == 0u) {
        pp[0] = -1e30f;
        pp[1] = 0.0f;
      }
      pp[2u + 4u * l + 0u] = 0.0f;
      pp[2u + 4u * l + 1u] = 0.0f;
      pp[2u + 4u * l + 2u] = 0.0f;
      pp[2u + 4u * l + 3u] = 0.0f;
    }
  } else if (w >= 6u) {
    const unsigned long row_step = (unsigned long)n_kv * INK_R7_PF_ROWB;
    const G u8 *kb_kh = (const G u8 *)kc + (unsigned long)kh * INK_R7_PF_ROWB;
    const G u8 *vb_kh = (const G u8 *)vc + (unsigned long)kh * INK_R7_PF_ROWB;
    const int no_wrap = (kend <= kv_len);
    ink_r7_pf_producer(kb_kh, vb_kh, rel_bias, row_step, kv_len, k_start, kend,
                       qpos_min, extent_lim, blk_beg, blk_end, blk_far_end, no_wrap, tid() - 192u);
  } else {
    const u32 tsub = w >= 3u ? 1u : 0u;
    const u32 hsub = w - 3u * tsub;
    const u32 t_sub0 = t0 + 16u * tsub;
    if (t_sub0 >= T) {
      for (u32 k = 0u; k < nblk_loc; k++) lbarrier();
    } else {
      const u32 h = kh * 3u + hsub;
      const u32 tk = t_sub0 + rl;
      const u32 tq = tk < T ? tk : (T - 1u);
      const u32 tp = pos0 + tk;
      const u32 sub_qpos_min = pos0 + t_sub0;
      const u32 sub_qpos_max = pos0 + (t_sub0 + 15u);

      h8 qa[8];
      {
        const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (unsigned long)tq * qs) +
                                               (unsigned long)h * 128ul);
#pragma unroll
        for (u32 ks = 0u; ks < 8u; ks++) {
          const f32x4 v0 = qp4[4u * ks + 2u * hl + 0u];
          const f32x4 v1 = qp4[4u * ks + 2u * hl + 1u];
          qa[ks] = (h8){(f16)(v0.x * INK_R7_PF_QSC), (f16)(v0.y * INK_R7_PF_QSC),
                        (f16)(v0.z * INK_R7_PF_QSC), (f16)(v0.w * INK_R7_PF_QSC),
                        (f16)(v1.x * INK_R7_PF_QSC), (f16)(v1.y * INK_R7_PF_QSC),
                        (f16)(v1.z * INK_R7_PF_QSC), (f16)(v1.w * INK_R7_PF_QSC)};
        }
      }
      h8 ra;
      {
        const G f32x4 *rp4 = (const G f32x4 *)((const G float *)((const G u8 *)R + (unsigned long)tq * rs) +
                                               (unsigned long)h * 16ul);
        const f32x4 r0 = rp4[2u * hl + 0u];
        const f32x4 r1 = rp4[2u * hl + 1u];
        ra = (h8){(f16)(r0.x * INK_R7_PF_LOG2E), (f16)(r0.y * INK_R7_PF_LOG2E),
                  (f16)(r0.z * INK_R7_PF_LOG2E), (f16)(r0.w * INK_R7_PF_LOG2E),
                  (f16)(r1.x * INK_R7_PF_LOG2E), (f16)(r1.y * INK_R7_PF_LOG2E),
                  (f16)(r1.z * INK_R7_PF_LOG2E), (f16)(r1.w * INK_R7_PF_LOG2E)};
      }

      /* Split-KV omits sink here; dynamic per-token sink is added once in the final merge. */
      float m = -__builtin_inff();
      float ls = 0.0f;
      const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2u) & 1u) ^ (rl >> 3u);
      const v8f zero = (v8f)(0.0f);
      v8f o[8];
#pragma unroll
      for (u32 nt = 0u; nt < 8u; nt++) o[nt] = zero;

      ink_r7_pf_consumer_loop(w, rl, hl, vt_lane, k_start, qpos_min, sub_qpos_min, sub_qpos_max,
                              tp, win, extent_lim, blk_beg, blk_end, blk_far_end, qa, ra, &m, &ls, o);

      const float l_tot = ls + ink_r7_pf_xrow(ls);
      if (tk < T) {
        G float *pp = part + (unsigned long)((tk * n_head + h) * nsplit + s) * INK_R7_PF_PSTR;
        if (hl == 0u) {
          pp[0] = (l_tot > 0.0f) ? (m * INK_R7_PF_LN2) : -1e30f;
          pp[1] = l_tot;
        }
        G float *po = pp + 2u;
#pragma unroll
        for (u32 nt = 0u; nt < 8u; nt++) {
          const u32 off = 16u * nt + 8u * hl;
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) po[off + v] = o[nt][v];
        }
      }
    }
  }

  /* Split-K arrival & last-WG parallel merge across nq_tile * 3 heads (nsplit <= 8). */
  const u32 cidx = __builtin_amdgcn_readfirstlane(tile * n_kv + kh);
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r7_pf_lds.is_last =
        (__scoped_atomic_fetch_add(&count[cidx], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r7_pf_lds.is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const u32 h0 = kh * 3u;
  const u32 total_qh = nq_tile * 3u;
  for (u32 qi = w; qi < total_qh; qi += 8u) {
    const u32 tk = t0 + qi / 3u;
    const u32 h = h0 + qi % 3u;
    const float sv = ((const G float *)((const G u8 *)sink + (unsigned long)tk * ss))[h];
    const G float *ph = part + (unsigned long)((tk * n_head + h) * nsplit) * INK_R7_PF_PSTR;

    float M2 = sv;
    float ml_m[8], ml_l[8];
#pragma unroll
    for (u32 sp = 0u; sp < 8u; sp++) {
      if (sp < nsplit) {
        const unsigned long off = (unsigned long)sp * INK_R7_PF_PSTR;
        ml_m[sp] = ph[off];
        ml_l[sp] = ph[off + 1u];
        if (ml_l[sp] > 0.0f) M2 = __builtin_fmaxf(M2, ml_m[sp]);
      } else {
        ml_m[sp] = -1e30f;
        ml_l[sp] = 0.0f;
      }
    }

    float L2 = __builtin_expf(sv - M2);
    f32x4 acc = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (u32 sp = 0u; sp < 8u; sp++) {
      if (sp < nsplit && ml_l[sp] > 0.0f) {
        const float wt = __builtin_expf(ml_m[sp] - M2);
        L2 += ml_l[sp] * wt;
        const G float *po = ph + (unsigned long)sp * INK_R7_PF_PSTR + 2u + 4u * l;
        const f32x4 ov = {po[0], po[1], po[2], po[3]};
        acc += ov * wt;
      }
    }
    const float inv = 1.0f / L2;
    G float *ot = (G float *)((G u8 *)out + (unsigned long)tk * os) + (unsigned long)h * 128ul;
    ((G f32x4 *)ot)[l] = acc * inv;
  }

  barrier();
  if (tid() == 0u) count[cidx] = 0u;
}

#endif /* !IE_MEGA */

#undef INK_R7_PF_PSTR
#undef INK_R7_PF_ROWB
#undef INK_R7_PF_QSC
#undef INK_R7_PF_LN2
#undef INK_R7_PF_LOG2E
#undef INK_R7_PF_D
#undef INK_R7_PF_NB
