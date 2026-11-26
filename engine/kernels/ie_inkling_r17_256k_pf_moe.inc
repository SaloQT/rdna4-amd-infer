/* ============================================================================
 * Inkling-2-Mini Round 17 (256K Cold Prefill SOTA Suite: SWA + BM=64 MoE):
 *   1. ie_inkling_rel_attn_pf_swa_fast_r17        (3-WG/CU f16 SWA/All-Phase PF)
 *   2. ie_inkling_gemm_rq_exp_iq3xxs_flute_m64_r17 (BM=64 FLUTE IQ3_XXS+SITU PF)
 *   3. ie_inkling_gemm_iq4nl_wave_m64_r17          (BM=64 Wave IQ4_NL Down PF)
 * Target: RX 9070 XT (gfx1201, RDNA4, wave32, NT=256, 64 CUs, 64 MiB MALL)
 *
 * Included at EOF of kernels/ie_inkling_kernels.c (after ie_kernels.c).
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r17_ /
 * INK_R17_ and all macros are #undef'd at EOF.
 * Designed for 0 warnings (-Wall -Wextra), 0/0 spills, priv=0, scratch=0,
 * TH_LOAD_NT=0, TH_LOAD_HT=0 (R43 default cached loads), f16_mix=0 in MoE
 * GEMM K-loops, scalar __builtin_bit_cast safety, and 3 WGs/CU (<= 120 VGPR,
 * <= 21,504 B LDS) across all 3 kernels!
 *
 * Key Upgrades Targeting #1 Remaining 256K Cold Prefill Bottlenecks (chunk=1024):
 *   1. ie_inkling_rel_attn_pf_swa_fast_r17 (40 SWA visits/chunk, win=1024, f16):
 *      - Exact 20-arg OP_REL_ATTN drop-in (replaces 2-WG/CU 32 KiB / 145-VGPR
 *        pf_sb_r10 with 3-WG/CU 19,456 B LDS / <= 120-VGPR async engine!).
 *      - Strips 12,672 B rbsc LDS via flipped (d4 outer, v inner) zero-LDS
 *        L1-cached R*rb (8x fewer global R loads than inner-d4; 100% bit-exact
 *        to pf_glob_q4_fast_r15!).
 *      - Contiguous SWA fast-path loader: 62 of 66 win=1024 ring-wrapped blocks
 *        satisfy (kb+16 <= kend && kv_b+16 <= kv_len) -> zero-branch, zero-modulo
 *        load16_cont! D=2 register prefetch ring (2x-unrolled -> 0 spills) +
 *        gfx1201 split barriers + ballot_w32 conditional rescale + zero-tile PV skip.
 *   2. ie_inkling_gemm_rq_exp_iq3xxs_flute_m64_r17 (BM=64 FLUTE IQ3_XXS + SITU):
 *      - Exact 14-arg drop-in for flute_situ_r10 / flute_l0lut_r13 supporting
 *        BOTH worklist mode (n_tiles != NULL, K_INK_MOE_WL_FILTER_R14 slot
 *        remapping for Core 128E + H38 Mode C) AND expert-major mode
 *        (n_tiles == NULL, tiles = offs_w = b->ms_offs + e_lo, grid.y = cnt,
 *        keeping FLUTE weights hot across consecutive BM=64 tiles for skewed
 *        chunk=1024 experts with > 64 tokens!).
 *      - L0$-cached LUT (18,688 B LDS, 0 LUT bank conflicts), gfx1201 decoupled
 *        split barriers hiding a_cur/d_cur + barrier latency behind final WMMAs,
 *        pinned h2f eliminating all v_fma_mix_f32 (f16_mix=0), exact-SHA!
 *   3. ie_inkling_gemm_iq4nl_wave_m64_r17 (BM=64 Wave-Interleaved IQ4_NL Down):
 *      - Exact 17-arg drop-in for expmaj_r10 / wave_expmaj_r13 supporting BOTH
 *        128E Core + H38 Mode C (offs_w = b->ms_offs + e_lo, grid.y = cnt).
 *      - Reads Wave-Interleaved IQ4_NL (H36a repack, 64 B contiguous qw + 8 B
 *        contiguous qs per 4-block step including nb=36 rem=4 tail) once per
 *        64-token tile, adds 0x4B400000 exact magic-float conversion (-32
 *        v_cvt_f32_i32/K-block), gfx1201 decoupled split barriers overlapping
 *        next wa[0..3] loads + barrier with k=3 WMMAs, 20,864 B LDS (3 WG/CU),
 *        pinned sw conversion (f16_mix=0), exact-SHA!
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(kvt) == 2u, "ie_inkling_r17_256k_pf_moe requires f16 kvt");
_Static_assert(sizeof(f16) == 2u, "ink_r17 f16 must be 2 bytes");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r17 float/u32 must be 4 bytes");

typedef unsigned long long ink_r17_u64;
typedef unsigned int       ink_r17_u2 __attribute__((ext_vector_type(2)));
typedef f16                ink_r17_h4 __attribute__((ext_vector_type(4)));

struct __attribute__((packed, aligned(4))) ink_r17_u32x4p {
  u32x4 v;
};

struct __attribute__((packed, aligned(4))) ink_r17_u2p {
  ink_r17_u2 v;
};

_Static_assert(sizeof(ink_r17_u64) == 8u, "ink_r17_u64 must be 8 bytes");

#define INK_R17_ATT_HD      128u
#define INK_R17_ATT_DREL    16u
#define INK_R17_ATT_GQA     3u
#define INK_R17_ATT_NB16    16u
#define INK_R17_ATT_ROWB    256ul
#define INK_R17_ATT_LOG2E   1.44269504088896340736f
#define INK_R17_ATT_QSC     (0.08838834764831844299f * 1.44269504088896340736f)

#define INK_R17_IQ4_KB      4u
#define INK_R17_IQ4_R       128u
#define INK_R17_IQ4_SW_STR  136u
#define INK_R17_IQ4_EP_STR  129u

/* ============================================================================
 * Shared gfx1201 Primitives (Split Barriers, Lane Pin, Default Cached Loads)
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r17_lsignal(void) {
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  __builtin_amdgcn_s_barrier_signal(-1);
  __asm__ volatile("" ::: "memory");
}

static inline __attribute__((always_inline)) void ink_r17_lwait(void) {
  __builtin_amdgcn_s_barrier_wait(-1);
  __asm__ volatile("" ::: "memory");
}

/* Full-exec lane ID (0..31) blocking LLVM LICM from hoisting LDS addresses
 * across K-loops. ONLY call where all 32 lanes are active. */
static inline __attribute__((always_inline)) u32 ink_r17_lane(void) {
  u32 l;
  __asm__ volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(l));
  return l;
}

static inline __attribute__((always_inline)) float ink_r17_xrow(const float v) {
  return I2F(xrow_i(F2I(v)));
}

static inline __attribute__((always_inline)) u32x4 ink_r17_ld4(const G void *p) {
  return ((const G struct ink_r17_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) ink_r17_u2 ink_r17_ld2(const G void *p) {
  return ((const G struct ink_r17_u2p *)p)->v;
}

static inline __attribute__((always_inline)) u32 ink_r17_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

/* ============================================================================
 * PART 1: ie_inkling_rel_attn_pf_swa_fast_r17
 *   3-WG/CU (19,456 B LDS, <= 120 VGPR) Async Double-Buffered f16 SWA /
 *   All-Phase Prefill Attention (exact 20-arg OP_REL_ATTN drop-in!).
 *   Grid: ((T + 31u) >> 5u, n_kv, 1u), NT = 256.
 * ========================================================================== */

typedef struct {
  f16 k[2u][INK_R17_ATT_NB16 * INK_R17_ATT_HD];  /*  8,192 B: double-buffered K */
  f16 vt[2u][INK_R17_ATT_HD * INK_R17_ATT_NB16]; /*  8,192 B: double-buffered V^T */
  f16 rb[2u][48u * INK_R17_ATT_DREL];            /*  3,072 B: double-buffered rel_bias */
} ink_r17_swa_lds_t;

_Static_assert(sizeof(ink_r17_swa_lds_t) == 19456u, "ink_r17_swa_lds_t must be 19,456 B (3 WG/CU)");

typedef struct {
  u32x4      kx[4];
  ink_r17_u2 vx[8];
} ink_r17_swa_slot16_t;

static inline __attribute__((always_inline)) u32 ink_r17_swa_k_off(const u32 r, const u32 c) {
  return r * 128u + 8u * (c ^ (r & 15u));
}

static inline __attribute__((always_inline)) u32 ink_r17_swa_vt_off(const u32 d, const u32 h) {
  return 8u * ((2u * d + h) ^ ((d >> 2u) & 1u) ^ ((d >> 3u) & 7u));
}

static inline __attribute__((always_inline)) u32 ink_r17_swa_wrap_s(u32 v, const u32 kv_len) {
  if (kv_len >= 16u) return (v >= kv_len) ? (v - kv_len) : v;
  while (v >= kv_len) v -= kv_len;
  return v;
}

static inline __attribute__((always_inline)) u32 ink_r17_swa_kv_adv16(const u32 kv_b, const u32 kv_len) {
  return ink_r17_swa_wrap_s(kv_b + INK_R17_ATT_NB16, kv_len);
}

static inline __attribute__((always_inline)) u32 ink_r17_swa_calc_far_end16(
    const u32 win, const u32 qpos_min, const u32 extent_lim, const u32 k_start, const u32 nblk) {
  if (win > 0u) return 0u;
  const u32 guard = extent_lim + (INK_R17_ATT_NB16 - 1u);
  if (qpos_min < guard || qpos_min - guard < k_start) return 0u;
  const u32 b_far = ((qpos_min - guard - k_start) >> 4u) + 1u;
  return (b_far < nblk) ? b_far : nblk;
}

static inline __attribute__((always_inline)) void ink_r17_swa_t8(
    const u32 x[8], u32x4 *ev, u32x4 *od) {
  const u32 e0 = __builtin_amdgcn_perm(x[1], x[0], 0x05040100u);
  const u32 e1 = __builtin_amdgcn_perm(x[3], x[2], 0x05040100u);
  const u32 e2 = __builtin_amdgcn_perm(x[5], x[4], 0x05040100u);
  const u32 e3 = __builtin_amdgcn_perm(x[7], x[6], 0x05040100u);
  const u32 o0 = __builtin_amdgcn_perm(x[1], x[0], 0x07060302u);
  const u32 o1 = __builtin_amdgcn_perm(x[3], x[2], 0x07060302u);
  const u32 o2 = __builtin_amdgcn_perm(x[5], x[4], 0x07060302u);
  const u32 o3 = __builtin_amdgcn_perm(x[7], x[6], 0x07060302u);
  *ev = (u32x4){e0, e1, e2, e3};
  *od = (u32x4){o0, o1, o2, o3};
}

/* Zero-branch contiguous f16 loader: valid for EVERY block with
 * (kb + 16 <= kend && kv_b + 16 <= kv_len), covering 62 of 66 win=1024 SWA blocks! */
static inline __attribute__((always_inline)) void ink_r17_swa_load16_cont(
    ink_r17_swa_slot16_t *s, const G u8 *kb_kh, const G u8 *vb_kh,
    const ink_r17_u64 row_step, const u32 kv_b, const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
  const G u8 *k_w = kb_kh + (ink_r17_u64)(kv_b + 8u * wi) * row_step;
  const ink_r17_u64 k_off = (ink_r17_u64)sub_r * row_step + 16ull * (ink_r17_u64)c;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    s->kx[i] = *(const G u32x4 *)(k_w + (ink_r17_u64)(2u * i) * row_step + k_off);
  }
  const G u8 *v_w = vb_kh + (ink_r17_u64)(kv_b + 8u * wi) * row_step;
  const ink_r17_u64 v_off = 8ull * (ink_r17_u64)L;
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    s->vx[u] = *(const G ink_r17_u2 *)(v_w + (ink_r17_u64)u * row_step + v_off);
  }
}

static inline __attribute__((always_inline)) void ink_r17_swa_load16_ring(
    ink_r17_swa_slot16_t *s, const G u8 *kb_kh, const G u8 *vb_kh,
    const ink_r17_u64 row_step, const u32 kv_len, const u32 kb, const u32 kv_b,
    const u32 kend, const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
  const u32 r_base = 8u * wi;
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r = r_base + 2u * i + sub_r;
    const u32 kp = kb + r;
    s->kx[i] = (u32x4)(0u);
    if (kp < kend) {
      const u32 kvp = ink_r17_swa_wrap_s(kv_b + r, kv_len);
      s->kx[i] = *(const G u32x4 *)(kb_kh + (ink_r17_u64)kvp * row_step + 16ull * (ink_r17_u64)c);
    }
  }
  const ink_r17_u64 v_off = 8ull * (ink_r17_u64)L;
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    const u32 r = r_base + u;
    const u32 kp = kb + r;
    s->vx[u] = (ink_r17_u2)(0u);
    if (kp < kend) {
      const u32 kvp = ink_r17_swa_wrap_s(kv_b + r, kv_len);
      s->vx[u] = *(const G ink_r17_u2 *)(vb_kh + (ink_r17_u64)kvp * row_step + v_off);
    }
  }
}

static inline __attribute__((always_inline)) void ink_r17_swa_load16_auto(
    ink_r17_swa_slot16_t *s, const G u8 *kb_kh, const G u8 *vb_kh,
    const ink_r17_u64 row_step, const u32 kv_len, const u32 kb, const u32 kv_b,
    const u32 kend, const u32 pt) {
  if (kb + INK_R17_ATT_NB16 <= kend && kv_b + INK_R17_ATT_NB16 <= kv_len) {
    ink_r17_swa_load16_cont(s, kb_kh, vb_kh, row_step, kv_b, pt);
  } else {
    ink_r17_swa_load16_ring(s, kb_kh, vb_kh, row_step, kv_len, kb, kv_b, kend, pt);
  }
}

static inline __attribute__((always_inline)) void ink_r17_swa_pub16_kv(
    LDSP ink_r17_swa_lds_t *lds, const ink_r17_swa_slot16_t *s, const u32 bf, const u32 pt) {
  const u32 wi = (u32)__builtin_amdgcn_readfirstlane((int)(pt >> 5u));
  const u32 L = pt & 31u;
  const u32 sub_r = L >> 4u, c = L & 15u;
  LDSP f16 *k_dst = lds->k[bf];
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) {
    const u32 r = 8u * wi + 2u * i + sub_r;
    *(LDSP u32x4 *)(k_dst + ink_r17_swa_k_off(r, c)) = s->kx[i];
  }
  u32 vx0[8], vx1[8];
#pragma unroll
  for (u32 u = 0u; u < 8u; u++) {
    vx0[u] = s->vx[u].x;
    vx1[u] = s->vx[u].y;
  }
  u32x4 ev0, od0, ev1, od1;
  ink_r17_swa_t8(vx0, &ev0, &od0);
  ink_r17_swa_t8(vx1, &ev1, &od1);
  const u32 c0 = 4u * L;
  LDSP f16 *vt_dst = lds->vt[bf];
  *(LDSP u32x4 *)(vt_dst + ink_r17_swa_vt_off(c0 + 0u, wi)) = ev0;
  *(LDSP u32x4 *)(vt_dst + ink_r17_swa_vt_off(c0 + 1u, wi)) = od0;
  *(LDSP u32x4 *)(vt_dst + ink_r17_swa_vt_off(c0 + 2u, wi)) = ev1;
  *(LDSP u32x4 *)(vt_dst + ink_r17_swa_vt_off(c0 + 3u, wi)) = od1;
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r17_swa_pub16_rb(
    LDSP ink_r17_swa_lds_t *lds, const G float *rel_bias, const u32 qpos_min,
    const u32 extent_lim, const u32 kb, const u32 bf, const u32 pt) {
  if (!rel_bias || extent_lim == 0u) return;
  const int d_min_s = (int)qpos_min - (int)(kb + (INK_R17_ATT_NB16 - 1u));
  const u32 d_base_wg = (d_min_s > 0) ? (u32)d_min_s : 0u;
  if (d_base_wg < extent_lim && pt < 48u) {
    const u32 d = d_base_wg + pt;
    h8 r0 = (h8)(0.0f), r1 = (h8)(0.0f);
    if (d < extent_lim) {
      const G f32x4 *rp = (const G f32x4 *)(rel_bias + (ink_r17_u64)d * 16ull);
      const f32x4 v0 = rp[0], v1 = rp[1], v2 = rp[2], v3 = rp[3];
      r0 = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
      r1 = (h8){(f16)v2.x, (f16)v2.y, (f16)v2.z, (f16)v2.w, (f16)v3.x, (f16)v3.y, (f16)v3.z, (f16)v3.w};
    }
    *(LDSP h8 *)(lds->rb[bf] + pt * 16u + 0u) = r0;
    *(LDSP h8 *)(lds->rb[bf] + pt * 16u + 8u) = r1;
  }
}

/* D=2 Async Double-Buffered Producer for f16 SWA/All-Phase (2x-unrolled so
 * ring[0]/ring[1] stay compile-time constant -> 0 private stack / 0 spills!).
 * Emits EXACTLY nblk split barriers matching consumer16. */
static inline __attribute__((always_inline)) void ink_r17_swa_producer16(
    LDSP ink_r17_swa_lds_t *lds, const G u8 *kb_kh, const G u8 *vb_kh,
    const G float *rel_bias, const ink_r17_u64 row_step, const u32 kv_len,
    const u32 k_start, const u32 kend, const u32 qpos_min, const u32 extent_lim,
    const u32 nblk, const u32 blk_far_end, const int no_wrap, const u32 pt) {
  if (nblk == 0u) return;

  const u32 nblk_full = (kend > k_start) ? ((kend - k_start) >> 4u) : 0u;
  const u32 fast_lim = no_wrap ? ((nblk_full < nblk) ? nblk_full : nblk) : 0u;
  const u32 far_loc = (blk_far_end < fast_lim) ? blk_far_end : fast_lim;

  u32 kv_load = no_wrap ? k_start : (k_start % kv_len);
  ink_r17_swa_slot16_t ring[2];

  /* Prime ring[0] (block 0) and ring[1] (block 1) while consumers load Q. */
  ink_r17_swa_load16_auto(&ring[0], kb_kh, vb_kh, row_step, kv_len, k_start, kv_load, kend, pt);
  kv_load = no_wrap ? (k_start + 16u) : ink_r17_swa_kv_adv16(kv_load, kv_len);

  if (1u < nblk) {
    ink_r17_swa_load16_auto(&ring[1], kb_kh, vb_kh, row_step, kv_len, k_start + 16u, kv_load, kend, pt);
    kv_load = no_wrap ? (k_start + 32u) : ink_r17_swa_kv_adv16(kv_load, kv_len);
  }
  __builtin_amdgcn_sched_barrier(0);

  /* Publish block 0 into bf=0, signal Sync 0, prefetch block 2 into ring[0], wait Sync 0. */
  ink_r17_swa_pub16_kv(lds, &ring[0], 0u, pt);
  if (0u >= blk_far_end) {
    ink_r17_swa_pub16_rb(lds, rel_bias, qpos_min, extent_lim, k_start, 0u, pt);
  }
  ink_r17_lsignal();
  if (2u < nblk) {
    ink_r17_swa_load16_auto(&ring[0], kb_kh, vb_kh, row_step, kv_len, k_start + 32u, kv_load, kend, pt);
    kv_load = no_wrap ? (k_start + 48u) : ink_r17_swa_kv_adv16(kv_load, kv_len);
  }
  ink_r17_lwait();

  u32 k = 0u;
  /* Pure far-field no-wrap 2x hot loop (for win==0 global FP16 layers). */
  if (no_wrap && far_loc >= 4u) {
#pragma clang loop unroll(disable)
    for (; k + 4u < far_loc; k += 2u) {
      ink_r17_swa_pub16_kv(lds, &ring[1], 1u, pt);
      ink_r17_lsignal();
      ink_r17_swa_load16_cont(&ring[1], kb_kh, vb_kh, row_step, k_start + 16u * (k + 3u), pt);
      ink_r17_lwait();

      ink_r17_swa_pub16_kv(lds, &ring[0], 0u, pt);
      ink_r17_lsignal();
      ink_r17_swa_load16_cont(&ring[0], kb_kh, vb_kh, row_step, k_start + 16u * (k + 4u), pt);
      ink_r17_lwait();
    }
    kv_load = k_start + 16u * (k + 3u);
  }

  /* Unified SWA + near-field + tail 2x-unrolled loop (constant ring[0]/ring[1]!). */
#pragma clang loop unroll(disable)
  for (; k + 1u < nblk; k += 2u) {
    {
      const u32 kn = k + 1u;
      ink_r17_swa_pub16_kv(lds, &ring[1], 1u, pt);
      if (kn >= blk_far_end) {
        ink_r17_swa_pub16_rb(lds, rel_bias, qpos_min, extent_lim, k_start + 16u * kn, 1u, pt);
      }
      ink_r17_lsignal();
      const u32 kl = kn + 2u;
      if (kl < nblk) {
        const u32 kbl = k_start + 16u * kl;
        ink_r17_swa_load16_auto(&ring[1], kb_kh, vb_kh, row_step, kv_len, kbl, no_wrap ? kbl : kv_load, kend, pt);
        if (!no_wrap) kv_load = ink_r17_swa_kv_adv16(kv_load, kv_len);
      }
      ink_r17_lwait();
    }
    if (k + 2u < nblk) {
      const u32 kn = k + 2u;
      ink_r17_swa_pub16_kv(lds, &ring[0], 0u, pt);
      if (kn >= blk_far_end) {
        ink_r17_swa_pub16_rb(lds, rel_bias, qpos_min, extent_lim, k_start + 16u * kn, 0u, pt);
      }
      ink_r17_lsignal();
      const u32 kl = kn + 2u;
      if (kl < nblk) {
        const u32 kbl = k_start + 16u * kl;
        ink_r17_swa_load16_auto(&ring[0], kb_kh, vb_kh, row_step, kv_len, kbl, no_wrap ? kbl : kv_load, kend, pt);
        if (!no_wrap) kv_load = ink_r17_swa_kv_adv16(kv_load, kv_len);
      }
      ink_r17_lwait();
    }
  }
}

/* 1-deep QK16 (<= 112 VGPR peak, 0 hoisted address VGPRs!). */
static inline __attribute__((always_inline)) v8f ink_r17_swa_qk16_1d(
    const LDSP f16 *k_lds, const h8 qa[8]) {
  const u32 l_loc = ink_r17_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  u32 kb_b = (rl << 8u) | ((rl ^ hl) << 4u);
  const LDSP u8 *kp = (const LDSP u8 *)k_lds;

  v8f sa = (v8f)(0.0f);
  h8 k0;
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[0], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[1], sa); kb_b ^= 96u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[2], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[3], sa); kb_b ^= 224u; __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[4], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[5], sa); kb_b ^= 96u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[6], sa); kb_b ^= 32u;  __builtin_amdgcn_sched_barrier(0);
  k0 = *(const LDSP h8 *)(kp + kb_b); sa = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k0, qa[7], sa); __builtin_amdgcn_sched_barrier(0);
  return sa;
}

/* 1-deep PV16 with split-barrier signal BEFORE final WMMA (<= 112 VGPR!). */
static inline __attribute__((always_inline)) void ink_r17_swa_pv16_1d(
    const LDSP f16 *vt_lds, const h8 pb, v8f o[8], const int do_signal) {
  const u32 l_loc = ink_r17_lane();
  const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
  const u32 vt_lane = (2u * rl + hl) ^ ((rl >> 2u) & 1u) ^ (rl >> 3u);
  const u32 vtb = 16u * vt_lane;
  const LDSP u8 *vp = (const LDSP u8 *)vt_lds;
  h8 v0;
  v0 = *(const LDSP h8 *)(vp +    0u + vtb);         o[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[0]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp +  512u + (vtb ^ 32u)); o[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[1]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 1024u + (vtb ^ 64u)); o[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[2]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 1536u + (vtb ^ 96u)); o[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[3]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 2048u + vtb);         o[4] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[4]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 2560u + (vtb ^ 32u)); o[5] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[5]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 3072u + (vtb ^ 64u)); o[6] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[6]); __builtin_amdgcn_sched_barrier(0);
  v0 = *(const LDSP h8 *)(vp + 3584u + (vtb ^ 96u));
  if (do_signal) {
    ink_r17_lsignal();
  } else {
    __builtin_amdgcn_sched_barrier(0);
  }
  o[7] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v0, pb, o[7]);
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r17_swa_softmax_pv16_sb(
    const LDSP f16 *vt_lds, const v8f sa, const int has_next,
    float *m_p, float *ls_p, v8f o[8]) {
  float bm = sa[0];
#pragma unroll
  for (u32 v = 1u; v < 8u; v++) bm = __builtin_fmaxf(bm, sa[v]);
  bm = __builtin_fmaxf(bm, ink_r17_xrow(bm));

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
        const float p = __builtin_amdgcn_exp2f(sa[v] - mn);
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

    ink_r17_swa_pv16_1d(vt_lds, pb, o, has_next);
  }
  if (has_next) {
    if (!active) ink_r17_lsignal();
    ink_r17_lwait();
  }
}

/* Flipped (d4 outer, v inner) zero-LDS L1-cached R*rel_bias + exact SWA/causal mask!
 * Loads R[tq, h, 0..15] ONCE per block (4x b128 instead of 32x -> 8x less L1 traffic!)
 * using only <= 6 live temp VGPRs (zero rbsc LDS -> true 3 WGs/CU!). */
static inline __attribute__((always_inline)) float ink_r17_swa_rb_acc4(
    float sv, const LDSP f16 *rb_lds, const u32 d4, const ink_r17_h4 rh,
    const u32 tp, const u32 kp, const u32 extent_lim, const u32 d_base_wg) {
  if (kp <= tp) {
    const u32 dist = tp - kp;
    if (dist < extent_lim) {
      const ink_r17_h4 bv = *(const LDSP ink_r17_h4 *)(rb_lds + (dist - d_base_wg) * 16u + d4);
      sv = __builtin_fmaf((float)rh.x, (float)bv.x, sv);
      sv = __builtin_fmaf((float)rh.y, (float)bv.y, sv);
      sv = __builtin_fmaf((float)rh.z, (float)bv.z, sv);
      sv = __builtin_fmaf((float)rh.w, (float)bv.w, sv);
    }
  }
  return sv;
}

static inline __attribute__((always_inline)) v8f ink_r17_swa_nf_apply(
    const v8f sa_v, const LDSP f16 *rb_lds, const G float *R, const u32 rs,
    const u32 t_sub0, const u32 T, const u32 h, const u32 kb,
    const u32 sub_qpos_min, const u32 win, const u32 extent_lim, const u32 d_base_wg,
    const int has_rb) {
  float s0 = sa_v[0], s1 = sa_v[1], s2 = sa_v[2], s3 = sa_v[3];
  float s4 = sa_v[4], s5 = sa_v[5], s6 = sa_v[6], s7 = sa_v[7];

  if (has_rb && d_base_wg < extent_lim) {
#pragma clang loop unroll(disable)
    for (u32 d4 = 0u; d4 < 16u; d4 += 4u) {
      const u32 l_loc = ink_r17_lane();
      const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
      const u32 tp = sub_qpos_min + rl;
      const u32 kp0 = kb + (hl << 3u);
      const u32 tq = (t_sub0 + rl < T) ? (t_sub0 + rl) : (T - 1u);
      const G float *rp = (const G float *)((const G u8 *)R + (ink_r17_u64)tq * (ink_r17_u64)rs) +
                          (ink_r17_u64)h * 16ull;
      const f32x4 rv = *(const G f32x4 *)(rp + d4);
      const ink_r17_h4 rh = {(f16)(rv.x * INK_R17_ATT_LOG2E), (f16)(rv.y * INK_R17_ATT_LOG2E),
                             (f16)(rv.z * INK_R17_ATT_LOG2E), (f16)(rv.w * INK_R17_ATT_LOG2E)};
      s0 = ink_r17_swa_rb_acc4(s0, rb_lds, d4, rh, tp, kp0 + 0u, extent_lim, d_base_wg);
      s1 = ink_r17_swa_rb_acc4(s1, rb_lds, d4, rh, tp, kp0 + 1u, extent_lim, d_base_wg);
      s2 = ink_r17_swa_rb_acc4(s2, rb_lds, d4, rh, tp, kp0 + 2u, extent_lim, d_base_wg);
      s3 = ink_r17_swa_rb_acc4(s3, rb_lds, d4, rh, tp, kp0 + 3u, extent_lim, d_base_wg);
      s4 = ink_r17_swa_rb_acc4(s4, rb_lds, d4, rh, tp, kp0 + 4u, extent_lim, d_base_wg);
      s5 = ink_r17_swa_rb_acc4(s5, rb_lds, d4, rh, tp, kp0 + 5u, extent_lim, d_base_wg);
      s6 = ink_r17_swa_rb_acc4(s6, rb_lds, d4, rh, tp, kp0 + 6u, extent_lim, d_base_wg);
      s7 = ink_r17_swa_rb_acc4(s7, rb_lds, d4, rh, tp, kp0 + 7u, extent_lim, d_base_wg);
      __builtin_amdgcn_sched_barrier(0);
    }
  }

  /* Exact causal + sliding-window mask (only enters on boundary blocks!). */
  const u32 sub_qpos_max = sub_qpos_min + 15u;
  if (kb + (INK_R17_ATT_NB16 - 1u) > sub_qpos_min || (win > 0u && sub_qpos_max >= kb + win)) {
    const u32 l_loc = ink_r17_lane();
    const u32 rl = l_loc & 15u, hl = l_loc >> 4u;
    const u32 tp = sub_qpos_min + rl;
    const u32 kp0 = kb + (hl << 3u);
    const float ninf = -__builtin_inff();
    if (kp0 + 0u > tp || (win > 0u && (tp - (kp0 + 0u)) >= win)) s0 = ninf;
    if (kp0 + 1u > tp || (win > 0u && (tp - (kp0 + 1u)) >= win)) s1 = ninf;
    if (kp0 + 2u > tp || (win > 0u && (tp - (kp0 + 2u)) >= win)) s2 = ninf;
    if (kp0 + 3u > tp || (win > 0u && (tp - (kp0 + 3u)) >= win)) s3 = ninf;
    if (kp0 + 4u > tp || (win > 0u && (tp - (kp0 + 4u)) >= win)) s4 = ninf;
    if (kp0 + 5u > tp || (win > 0u && (tp - (kp0 + 5u)) >= win)) s5 = ninf;
    if (kp0 + 6u > tp || (win > 0u && (tp - (kp0 + 6u)) >= win)) s6 = ninf;
    if (kp0 + 7u > tp || (win > 0u && (tp - (kp0 + 7u)) >= win)) s7 = ninf;
  }

  return (v8f){s0, s1, s2, s3, s4, s5, s6, s7};
}

__attribute__((amdgpu_waves_per_eu(12, 16)))
KERNEL ie_inkling_rel_attn_pf_swa_fast_r17(
    const G float *q, const G kvt *kc, const G kvt *vc,
    const G float *R, const G float *rel_bias, const G float *sink, G float *out,
    u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv,
    u32 d_rel, u32 rel_extent, u32 win,
    u32 qs, u32 rs, u32 ss, u32 os) {
  static LDS ink_r17_swa_lds_t ink_r17_swa_lds __attribute__((aligned(16)));
  if (hd != INK_R17_ATT_HD || d_rel != INK_R17_ATT_DREL || n_kv == 0u ||
      n_head != n_kv * INK_R17_ATT_GQA || T == 0u || kv_len == 0u ||
      !q || !kc || !vc || !sink || !out)
    return;

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
  const u32 nblk = (kend > k_start) ? ((kend - k_start + INK_R17_ATT_NB16 - 1u) >> 4u) : 0u;
  if (nblk == 0u) return;
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const u32 blk_far_end = ink_r17_swa_calc_far_end16(win, qpos_min, extent_lim, k_start, nblk);

  if (w >= 6u) {
    const ink_r17_u64 row_step = (ink_r17_u64)n_kv * (ink_r17_u64)INK_R17_ATT_ROWB;
    const G u8 *kb_kh = (const G u8 *)kc + (ink_r17_u64)kh * (ink_r17_u64)INK_R17_ATT_ROWB;
    const G u8 *vb_kh = (const G u8 *)vc + (ink_r17_u64)kh * (ink_r17_u64)INK_R17_ATT_ROWB;
    const int no_wrap = (kend <= kv_len);
    ink_r17_swa_producer16(&ink_r17_swa_lds, kb_kh, vb_kh, rel_bias, row_step, kv_len,
                           k_start, kend, qpos_min, extent_lim, nblk, blk_far_end,
                           no_wrap, tid() - 192u);
    return;
  }

  const u32 tsub = w >= 3u ? 1u : 0u;
  const u32 hsub = w - 3u * tsub;
  const u32 t_sub0 = (u32)__builtin_amdgcn_readfirstlane((int)(t0 + 16u * tsub));
  if (t_sub0 >= T) {
    for (u32 k = 0u; k < nblk; k++) {
      ink_r17_lsignal();
      ink_r17_lwait();
    }
    return;
  }

  /* Signal Sync 0 BEFORE Q/sink loads (hides initial barrier latency!). */
  ink_r17_lsignal();

  const u32 h = (u32)__builtin_amdgcn_readfirstlane((int)(kh * INK_R17_ATT_GQA + hsub));
  const u32 sub_qpos_min = (u32)__builtin_amdgcn_readfirstlane((int)(pos0 + t_sub0));

  h8 qa[8];
  float m, ls;
  {
    const u32 l_i = ink_r17_lane();
    const u32 rl_i = l_i & 15u, hl_i = l_i >> 4u;
    const u32 tk_i = t_sub0 + rl_i;
    const u32 tq_i = (tk_i < T) ? tk_i : (T - 1u);
    const G f32x4 *qp4 = (const G f32x4 *)((const G float *)((const G u8 *)q + (ink_r17_u64)tq_i * (ink_r17_u64)qs) +
                                           (ink_r17_u64)h * 128ull);
#pragma unroll
    for (u32 ks_i = 0u; ks_i < 8u; ks_i++) {
      const f32x4 v0 = qp4[4u * ks_i + 2u * hl_i + 0u];
      const f32x4 v1 = qp4[4u * ks_i + 2u * hl_i + 1u];
      qa[ks_i] = (h8){(f16)(v0.x * INK_R17_ATT_QSC), (f16)(v0.y * INK_R17_ATT_QSC),
                      (f16)(v0.z * INK_R17_ATT_QSC), (f16)(v0.w * INK_R17_ATT_QSC),
                      (f16)(v1.x * INK_R17_ATT_QSC), (f16)(v1.y * INK_R17_ATT_QSC),
                      (f16)(v1.z * INK_R17_ATT_QSC), (f16)(v1.w * INK_R17_ATT_QSC)};
      __builtin_amdgcn_sched_barrier(0);
    }
    const float sink_val = ((const G float *)((const G u8 *)sink + (ink_r17_u64)tq_i * (ink_r17_u64)ss))[h];
    m = sink_val * INK_R17_ATT_LOG2E;
    ls = (hl_i == 0u) ? 1.0f : 0.0f;
  }

  ink_r17_lwait();

  v8f o[8];
#pragma unroll
  for (u32 nt = 0u; nt < 8u; nt++) {
    v8f z = (v8f)(0.0f);
    __asm__ volatile("" : "+v"(z));
    o[nt] = z;
  }
  __builtin_amdgcn_sched_barrier(0);

  const u32 nfar = (u32)__builtin_amdgcn_readfirstlane((int)((blk_far_end < nblk) ? blk_far_end : nblk));

  /* Phase A: Pure Far-Field (win==0 global FP16 layers only; skipped when win>0). */
#pragma clang loop unroll(disable)
  for (u32 k = 0u; k < nfar; k++) {
    const u32 k_u = (u32)__builtin_amdgcn_readfirstlane((int)k);
    const u32 bf = k_u & 1u;
    const v8f sa = ink_r17_swa_qk16_1d(ink_r17_swa_lds.k[bf], qa);
    ink_r17_swa_softmax_pv16_sb(ink_r17_swa_lds.vt[bf], sa, (k_u + 1u < nblk), &m, &ls, o);
  }

  /* Phase B: SWA + Near-Field (exact L1-cached R*rb + causal/win mask). */
  if (nfar < nblk) {
#pragma clang loop unroll(disable)
    for (u32 k = nfar; k < nblk; k++) {
      const u32 k_u = (u32)__builtin_amdgcn_readfirstlane((int)k);
      const u32 kb = (u32)__builtin_amdgcn_readfirstlane((int)(k_start + INK_R17_ATT_NB16 * k_u));
      const u32 bf = k_u & 1u;
      const v8f sa_v = ink_r17_swa_qk16_1d(ink_r17_swa_lds.k[bf], qa);

      const int d_min_wg_s = (int)qpos_min - (int)(kb + (INK_R17_ATT_NB16 - 1u));
      const u32 d_base_wg = (u32)__builtin_amdgcn_readfirstlane((d_min_wg_s > 0) ? d_min_wg_s : 0);
      const int has_rb = (extent_lim > 0u && rel_bias != (const G float *)0 && R != (const G float *)0);
      const v8f sa_adj = ink_r17_swa_nf_apply(sa_v, ink_r17_swa_lds.rb[bf], R, rs, t_sub0, T, h, kb,
                                              sub_qpos_min, win, extent_lim, d_base_wg, has_rb);
      ink_r17_swa_softmax_pv16_sb(ink_r17_swa_lds.vt[bf], sa_adj, (k_u + 1u < nblk), &m, &ls, o);
    }
  }

  const float inv = 1.0f / (ls + ink_r17_xrow(ls));
  const u32 l_ep = ink_r17_lane();
  const u32 rl_ep = l_ep & 15u, hl_ep = l_ep >> 4u;
  const u32 tk_ep = t_sub0 + rl_ep;
  if (tk_ep < T) {
    G f32x4 *op4 = (G f32x4 *)((G float *)((G u8 *)out + (ink_r17_u64)tk_ep * (ink_r17_u64)os) +
                               (ink_r17_u64)h * 128ull);
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
 * PART 2: ie_inkling_gemm_rq_exp_iq3xxs_flute_m64_r17
 *   BM=64 (chunk=1024) FLUTE IQ3_XXS Dual-Proj (Gate+Up) + Inline SITU + Q8_0
 *   - Supports BOTH:
 *       * Worklist mode (n_tiles != NULL): exact 14-arg drop-in for
 *         flute_situ_r10 / flute_l0lut_r13 (K_INK_MOE_WL_FILTER_R14 remapped
 *         slot e = desc & 0xFFu, pb = (desc >> 8) & 0x1FFFu, nj <= 64).
 *       * Expert-major mode (n_tiles == NULL, tiles = offs_w = b->ms_offs+e_lo,
 *         grid.y = cnt): loops consecutive BM=64 tiles per expert inside 1 WG
 *         for L1$/L2$ weight reuse on skewed chunk=1024 experts (M > 64) and
 *         elides K_INK_MOE_WL_FILTER_R14 launches!
 *   - 18,688 B LDS (L0$-cached LUT), <= 116 VGPR (3 WG/CU), 0 v_fma_mix_f32,
 *     gfx1201 decoupled split barriers hiding a_cur/d_cur + barrier latency!
 * ========================================================================== */

typedef struct {
  v2i   w0, w1;
  float s0;
} ink_r17_fl_dc_t;

typedef struct {
  union {
    struct {
      u8    x[2u][64u * 128u] __attribute__((aligned(16)));
      float sx[2u][4u * 64u] __attribute__((aligned(32)));
    } k;
    float ep[2u][32u][66u] __attribute__((aligned(32)));
  } u;
  u32 pid[64] __attribute__((aligned(16)));
} ink_r17_fl_lds_t;

_Static_assert(sizeof(ink_r17_fl_lds_t) == 18688u, "ink_r17_fl_lds_t must be 18,688 B (3 WG/CU)");

/* Pinned f16 -> f32 conversion preventing quarter-rate v_fma_mix_f32! */
static inline __attribute__((always_inline)) float ink_r17_fl_h2f(const u32 d_bits) {
  const u16 h16 = (u16)(d_bits & 0xFFFFu);
  float f = (float)__builtin_bit_cast(f16, h16);
  __asm__ volatile("" : "+v"(f));
  return f;
}

static inline __attribute__((always_inline)) v2i ink_r17_fl_ksw(const u32 g0, const u32 g1, const u32 s7) {
  const u32 p_ = s7 * 0x00204081u;
  const u32 t0 = p_ & 0x01010101u;
  const u32 t1 = ((p_ >> 4u) | ((u32)__builtin_popcount(s7) << 24u)) & 0x01010101u;
  const u32 m0 = __builtin_amdgcn_perm(0u, 0u, t0 | 0x0C0C0C0Cu);
  const u32 m1 = __builtin_amdgcn_perm(0u, 0u, t1 | 0x0C0C0C0Cu);
  const v2i v = {(int)((g0 ^ m0) + t0), (int)((g1 ^ m1) + t1)};
  return v;
}

static inline __attribute__((always_inline)) void ink_r17_fl_decode_l0(
    const u32 qp, const u32 auxv, const float d2, const u32 h, ink_r17_fl_dc_t *dc) {
  const u32 g00 = ie_rq_iq3xxs_grid[qp & 255u];
  const u32 g01 = ie_rq_iq3xxs_grid[(qp >> 8u) & 255u];
  const u32 g10 = ie_rq_iq3xxs_grid[(qp >> 16u) & 255u];
  const u32 g11 = ie_rq_iq3xxs_grid[qp >> 24u];
  __builtin_amdgcn_sched_barrier(0);
  const u32 s_sh = h ? 14u : 0u;
  dc->s0 = d2 * (0.5f + (float)(auxv >> 28u));
  dc->w0 = ink_r17_fl_ksw(g00, g01, (auxv >> s_sh) & 127u);
  dc->w1 = ink_r17_fl_ksw(g10, g11, (auxv >> (s_sh + 7u)) & 127u);
}

static inline __attribute__((always_inline)) u32 ink_r17_fl_qp0(const u32x4 qv, const u32 h) {
  const u32 q0 = qv.x, q1 = qv.y;
  const u32 q1_x = (u32)xrow_i((int)q1);
  return h ? q1_x : q0;
}

static inline __attribute__((always_inline)) u32 ink_r17_fl_qp1(const u32x4 qv, const u32 h) {
  const u32 q2 = qv.z, q3 = qv.w;
  const u32 q3_x = (u32)xrow_i((int)q3);
  return h ? q3_x : q2;
}

static inline __attribute__((always_inline)) u32 ink_r17_fl_qp2(const u32x4 qv, const u32 h) {
  const u32 q0 = qv.x, q1 = qv.y;
  const u32 q0_x = (u32)xrow_i((int)q0);
  return h ? q1 : q0_x;
}

static inline __attribute__((always_inline)) u32 ink_r17_fl_qp3(const u32x4 qv, const u32 h) {
  const u32 q2 = qv.z, q3 = qv.w;
  const u32 q2_x = (u32)xrow_i((int)q2);
  return h ? q3 : q2_x;
}

static inline __attribute__((always_inline)) void ink_r17_fl_accum_tile(
    const v8i r, const float s0, const LDSP float *sx0, f8 *acc_j) {
  const f32x4 dlo = *(const LDSP f32x4 *)(sx0);
  const f32x4 dhi = *(const LDSP f32x4 *)(sx0 + 4u);
  const f8 rf = (__builtin_bit_cast(f8, r) - 12582912.0f) * s0;
  __builtin_amdgcn_sched_barrier(0);
  const f8 dx = {dlo[0], dlo[1], dlo[2], dlo[3], dhi[0], dhi[1], dhi[2], dhi[3]};
  *acc_j += rf * dx;
  __asm__ volatile("" : : "v"(*acc_j) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r17_fl_accum_pre(
    const v8i r, const float s0, const f32x4 dlo, const f32x4 dhi, f8 *acc_j) {
  const f8 rf = (__builtin_bit_cast(f8, r) - 12582912.0f) * s0;
  __builtin_amdgcn_sched_barrier(0);
  const f8 dx = {dlo[0], dlo[1], dlo[2], dlo[3], dhi[0], dhi[1], dhi[2], dhi[3]};
  *acc_j += rf * dx;
  __asm__ volatile("" : : "v"(*acc_j) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r17_fl_apply_one_j(
    const ink_r17_fl_dc_t dck, const LDSP u8 *lx_k, const LDSP float *lsx_k,
    const u32 j0, const int do_sig, f8 *acc) {
  const v8i zm = (v8i)(0x4B400000);
  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + j0 * 2048u);
  if (do_sig) {
    const f32x4 dlo = *(const LDSP f32x4 *)(lsx_k + 16u * j0);
    const f32x4 dhi = *(const LDSP f32x4 *)(lsx_k + 16u * j0 + 4u);
    ink_r17_lsignal();
    const v2i a0 = {(int)b0.x, (int)b0.y}, a1 = {(int)b0.z, (int)b0.w};
    v8i r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, dck.w0, zm, 0);
    r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, dck.w1, r0, 0);
    __builtin_amdgcn_sched_barrier(0);
    ink_r17_fl_accum_pre(r0, dck.s0, dlo, dhi, &acc[j0]);
    return;
  }
  __builtin_amdgcn_sched_barrier(0);
  const v2i a0 = {(int)b0.x, (int)b0.y}, a1 = {(int)b0.z, (int)b0.w};
  v8i r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, dck.w0, zm, 0);
  r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, dck.w1, r0, 0);
  __builtin_amdgcn_sched_barrier(0);
  ink_r17_fl_accum_tile(r0, dck.s0, lsx_k + 16u * j0, &acc[j0]);
}

static inline __attribute__((always_inline)) void ink_r17_fl_apply_pair_j(
    const ink_r17_fl_dc_t dck, const LDSP u8 *lx_k, const LDSP float *lsx_k,
    const u32 j0, const int do_sig, f8 *acc) {
  const v8i zm = (v8i)(0x4B400000);
  const u32 j_off = j0 * 2048u;
  if (do_sig) {
    /* Zero-extra-VGPR split barrier: finish j0 first (frees r0), then preload b1+d1 & signal! */
    const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + j_off);
    __builtin_amdgcn_sched_barrier(0);
    const v2i a0_0 = {(int)b0.x, (int)b0.y}, a0_1 = {(int)b0.z, (int)b0.w};
    v8i r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_0, 1, dck.w0, zm, 0);
    r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_1, 1, dck.w1, r0, 0);
    __builtin_amdgcn_sched_barrier(0);
    ink_r17_fl_accum_tile(r0, dck.s0, lsx_k + 16u * j0, &acc[j0]);
    const u32x4 b1 = *(const LDSP u32x4 *)(lx_k + j_off + 2048u);
    const f32x4 dlo1 = *(const LDSP f32x4 *)(lsx_k + 16u * (j0 + 1u));
    const f32x4 dhi1 = *(const LDSP f32x4 *)(lsx_k + 16u * (j0 + 1u) + 4u);
    ink_r17_lsignal();
    const v2i a1_0 = {(int)b1.x, (int)b1.y}, a1_1 = {(int)b1.z, (int)b1.w};
    v8i r1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_0, 1, dck.w0, zm, 0);
    r1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_1, 1, dck.w1, r1, 0);
    __builtin_amdgcn_sched_barrier(0);
    ink_r17_fl_accum_pre(r1, dck.s0, dlo1, dhi1, &acc[j0 + 1u]);
    return;
  }
  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + j_off);
  const u32x4 b1 = *(const LDSP u32x4 *)(lx_k + j_off + 2048u);
  __builtin_amdgcn_sched_barrier(0);
  const v2i a0_0 = {(int)b0.x, (int)b0.y}, a0_1 = {(int)b0.z, (int)b0.w};
  const v2i a1_0 = {(int)b1.x, (int)b1.y}, a1_1 = {(int)b1.z, (int)b1.w};
  v8i r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_0, 1, dck.w0, zm, 0);
  v8i r1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_0, 1, dck.w0, zm, 0);
  r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_1, 1, dck.w1, r0, 0);
  r1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_1, 1, dck.w1, r1, 0);
  __builtin_amdgcn_sched_barrier(0);
  ink_r17_fl_accum_tile(r0, dck.s0, lsx_k + 16u * j0, &acc[j0]);
  ink_r17_fl_accum_tile(r1, dck.s0, lsx_k + 16u * (j0 + 1u), &acc[j0 + 1u]);
}

static inline __attribute__((always_inline)) void ink_r17_fl_kstep(
    const ink_r17_fl_dc_t dck, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 k, const u32 lx_base0, const u32 wj, const int do_sig, f8 *acc) {
  const u32 l_loc = ink_r17_lane();
  const u32 h = l_loc >> 4u;
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const u32 sx_str = (wj >= 3u) ? 64u : 32u;
  const LDSP float *lsx_k = lsx + k * sx_str + 8u * h;
  if (wj == 1u) {
    ink_r17_fl_apply_one_j(dck, lx_k, lsx_k, 0u, do_sig, acc);
  } else if (wj == 2u) {
    ink_r17_fl_apply_pair_j(dck, lx_k, lsx_k, 0u, do_sig, acc);
  } else if (wj == 3u) {
    ink_r17_fl_apply_pair_j(dck, lx_k, lsx_k, 0u, 0, acc);
    ink_r17_fl_apply_one_j(dck, lx_k, lsx_k, 2u, do_sig, acc);
  } else {
    ink_r17_fl_apply_pair_j(dck, lx_k, lsx_k, 0u, 0, acc);
    ink_r17_fl_apply_pair_j(dck, lx_k, lsx_k, 2u, do_sig, acc);
  }
}

static inline __attribute__((always_inline)) void ink_r17_fl_stage_sx(
    LDSP float *lsx, const u32 stride, const u32 t0, const int is_s, const f32x4 rsx) {
  if (is_s) {
    u32 ti = t0;
    __asm__ volatile("" : "+v"(ti));
    lsx[0u * stride + ti] = rsx.x;
    lsx[1u * stride + ti] = rsx.y;
    lsx[2u * stride + ti] = rsx.z;
    lsx[3u * stride + ti] = rsx.w;
  }
}

static inline __attribute__((always_inline)) void ink_r17_fl_epilogue_strip(
    LDSP float *ep, const LDSP u32 *lpid, G u8 *Qout,
    const u32 strip_base, const u32 nj_strip, const u32 blk0, const u32 nblk,
    const u32 nb_out, const u32 qstr, const float beta, const float lbeta,
    const u32 j_base, const f8 *acc) {
#pragma clang fp contract(off)
  u32 t_ep = tid();
  __asm__ volatile("" : "+v"(t_ep));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t_ep >> 5u));
  const u32 l = t_ep & 31u, r16 = l & 15u, h = l >> 4u;
  const u32 is_up = w >> 2u;
  const u32 r_loc = 16u * (w & 3u) + r16;

  LDSP float *ep_wr = ep + (is_up * 32u + 8u * h) * 66u + r_loc;
#pragma unroll
  for (u32 j_loc = 0u; j_loc < 2u; j_loc++) {
    if (16u * j_loc < nj_strip) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        ep_wr[(16u * j_loc + v) * 66u] = acc[j_base + j_loc][v];
      }
    }
  }
  lbarrier();

  const int use_beta = beta > 0.0f;
  const int use_lbeta = lbeta > 0.0f;
  for (u32 tk = w; tk < nj_strip; tk += 8u) {
    const u32 pid = (u32)__builtin_amdgcn_readfirstlane((int)lpid[strip_base + tk]);
    G u8 *qp = Qout + (ink_r17_u64)pid * (ink_r17_u64)qstr;
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

/* Split-barrier L0$-LUT FLUTE K-loop body for wj in {1, 2, 3, 4} (BM=64 hot path
 * at wj=4!). Derives lane/address indices locally from pinned tid() so ZERO
 * tile-invariant address VGPRs leak across the outer expert-major BM=64 loop! */
static inline __attribute__((always_inline)) void ink_r17_fl_run_wj(
    LDSP ink_r17_fl_lds_t *lds, const G u8 *p_qs_e, const G u8 *p_aux_e, const G u8 *p_d_e,
    const G u8 *X_u, const G u8 *X_s, const G u32 *list,
    const u32 pb, const u32 nj, const u32 gbase, const u32 eff, const u32 mb,
    const u32 xs, const u32 xdiv, const u32 wj, f8 *acc) {
  u32 t0 = tid();
  __asm__ volatile("" : "+v"(t0));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = t0 & 31u, r16 = l & 15u, h = l >> 4u;
  const u32 is_up = w >> 2u;
  const u32 r_loc = 16u * (w & 3u) + r16;
  const u32 ch = gbase + r_loc;
  const u32 ch_c = ch < eff ? ch : (eff - 1u);
  const u32 w_row = is_up ? (eff + ch_c) : ch_c;

  const u32 r_idx = ((w_row >> 2u) * mb) * 4u + (w_row & 3u);
  const u32 q_off = (r_idx << 6u) + (h << 4u);
  const u32 a_off = r_idx << 5u;
  const u32 d_off = r_idx << 1u;

  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  const u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);
  const u32 lx_base0 = r16 * 128u + ((h ^ (r16 & 7u)) << 4u);

  const int ld_lo = (wj >= 2u) || (tk_ld < 16u);
  const int ld_hi = (wj == 4u) || ((wj == 3u) && (tk_ld < 16u));
  const int is_s = (wj == 4u) ? (w < 2u) : (wj == 3u ? (t0 < 48u) : (wj == 2u ? (w == 0u) : (t0 < 16u)));
  const u32 sx_str = (wj >= 3u) ? 64u : 32u;

  if (is_s) {
    const u32 pj = pb + (t0 < nj ? t0 : 0u);
    lds->pid[t0] = list[pj];
  }
  lbarrier();

  const u32 xoff0 = ld_lo ? (ink_r17_div_x(lds->pid[tk_ld], xdiv) * xs + (c8_ld << 4u)) : 0u;
  const u32 xoff1 = ld_hi ? (ink_r17_div_x(lds->pid[tk_ld + 32u], xdiv) * xs + (c8_ld << 4u)) : 0u;
  const u32 sxoff = is_s ? (ink_r17_div_x(lds->pid[t0 & 63u], xdiv) * xs) : 0u;

#pragma unroll
  for (u32 j = 0u; j < wj; j++) acc[j] = (f8)(0.0f);

  u32x4 q_cur = *(const G u32x4 *)(p_qs_e + q_off);
  u32x4 a_cur = *(const G u32x4 *)(p_aux_e + a_off);
  u32 d_cur = (u32)(*(const G u16 *)(p_d_e + d_off));
  {
    const u32x4 rx0_i = ld_lo ? ink_r17_ld4(X_u + xoff0) : (u32x4)(0u);
    const u32x4 rx1_i = ld_hi ? ink_r17_ld4(X_u + xoff1) : (u32x4)(0u);
    const f32x4 rsx_i = is_s ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
    if (ld_lo) *(LDSP u32x4 *)(lds->u.k.x[0] + st_x) = rx0_i;
    if (ld_hi) *(LDSP u32x4 *)(lds->u.k.x[0] + st_x + 4096u) = rx1_i;
    ink_r17_fl_stage_sx(lds->u.k.sx[0], sx_str, t0, is_s, rsx_i);
  }
  lbarrier();

  u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
  for (u32 sb = 0u; ; sb++) {
    const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
    const u32 kb1 = 8u * sb_u + 4u;
    const G u8 *p_qs_sb  = rq_ptr_u(p_qs_e  + (ink_r17_u64)sb_u * 256ull);
    const G u8 *p_aux_sb = rq_ptr_u(p_aux_e + (ink_r17_u64)sb_u * 128ull);

    /* -------- hb0 (k = 0..3): read x[0]/sx[0], fill x[1]/sx[1] -------- */
    const float d2 = ink_r17_fl_h2f(d_cur) * 0.5f;
    ink_r17_fl_dc_t xdc;
    ink_r17_fl_decode_l0(ink_r17_fl_qp0(q_cur, h), a_cur.x, d2, h, &xdc);
    __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
    __builtin_amdgcn_sched_barrier(0);
    {
      const G u8 *X_hb1 = rq_ptr_u(X_u + (ink_r17_u64)kb1 * 32ull);
      if (ld_lo) rx0 = ink_r17_ld4(X_hb1 + xoff0);
      if (ld_hi) rx1 = ink_r17_ld4(X_hb1 + xoff1);
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r17_fl_kstep(xdc, lds->u.k.x[0], lds->u.k.sx[0], 0u, lx_base0, wj, 0, acc);

    /* hb0 k=1: issue hb1 aligned 1x b128 qs into q_nxt. */
    ink_r17_fl_decode_l0(ink_r17_fl_qp1(q_cur, h), a_cur.y, d2, h, &xdc);
    const u32x4 q_nxt = *(const G u32x4 *)(p_qs_sb + q_off + 32u);
    __builtin_amdgcn_sched_barrier(0);
    ink_r17_fl_kstep(xdc, lds->u.k.x[0], lds->u.k.sx[0], 1u, lx_base0, wj, 0, acc);

    /* hb0 k=2: commit x[1] (frees rx0/rx1 BEFORE rsx load -> -8 VGPR!), load rsx. */
    if (ld_lo) *(LDSP u32x4 *)(lds->u.k.x[1] + st_x) = rx0;
    if (ld_hi) *(LDSP u32x4 *)(lds->u.k.x[1] + st_x + 4096u) = rx1;
    f32x4 rsx = (f32x4)(0.0f);
    if (is_s) {
      const G u8 *Xs_hb1 = rq_ptr_u(X_s + (ink_r17_u64)kb1 * 4ull);
      rsx = rq_load_f32x4((const G float *)(Xs_hb1 + sxoff));
    }
    ink_r17_fl_decode_l0(ink_r17_fl_qp2(q_cur, h), a_cur.z, d2, h, &xdc);
    __builtin_amdgcn_sched_barrier(0);
    ink_r17_fl_kstep(xdc, lds->u.k.x[0], lds->u.k.sx[0], 2u, lx_base0, wj, 0, acc);

    /* hb0 k=3: commit sx[1] BEFORE k=3 split barrier, decode, run kstep(3)+lsignal,
     * then issue a_cur(hb1) right before lwait (0 extra VGPR across kstep(3)!). */
    ink_r17_fl_stage_sx(lds->u.k.sx[1], sx_str, t0, is_s, rsx);
    ink_r17_fl_decode_l0(ink_r17_fl_qp3(q_cur, h), a_cur.w, d2, h, &xdc);
    __builtin_amdgcn_sched_barrier(0);
    ink_r17_fl_kstep(xdc, lds->u.k.x[0], lds->u.k.sx[0], 3u, lx_base0, wj, 1, acc);
    a_cur = *(const G u32x4 *)(p_aux_sb + a_off + 16u);
    ink_r17_lwait();

    /* -------- hb1 (k = 0..3): read x[1]/sx[1], fill x[0]/sx[0] if more -------- */
    const int more = (sb_u + 1u < mb);
    const u32 kbn = 8u * (sb_u + 1u);
    const G u8 *p_qs_nxt  = rq_ptr_u(p_qs_e  + (ink_r17_u64)(sb_u + 1u) * 256ull);
    const G u8 *p_aux_nxt = rq_ptr_u(p_aux_e + (ink_r17_u64)(sb_u + 1u) * 128ull);
    const G u8 *p_d_nxt   = rq_ptr_u(p_d_e   + (ink_r17_u64)(sb_u + 1u) * 8ull);

    ink_r17_fl_decode_l0(ink_r17_fl_qp0(q_nxt, h), a_cur.x, d2, h, &xdc);
    __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
    __builtin_amdgcn_sched_barrier(0);
    if (more) {
      const G u8 *X_nxt = rq_ptr_u(X_u + (ink_r17_u64)kbn * 32ull);
      if (ld_lo) rx0 = ink_r17_ld4(X_nxt + xoff0);
      if (ld_hi) rx1 = ink_r17_ld4(X_nxt + xoff1);
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r17_fl_kstep(xdc, lds->u.k.x[1], lds->u.k.sx[1], 0u, lx_base0, wj, 0, acc);

    /* hb1 k=1 */
    ink_r17_fl_decode_l0(ink_r17_fl_qp1(q_nxt, h), a_cur.y, d2, h, &xdc);
    if (more) q_cur = *(const G u32x4 *)(p_qs_nxt + q_off);
    __builtin_amdgcn_sched_barrier(0);
    ink_r17_fl_kstep(xdc, lds->u.k.x[1], lds->u.k.sx[1], 1u, lx_base0, wj, 0, acc);

    /* hb1 k=2: commit x[0] first (frees rx0/rx1!), then load rsx1. */
    f32x4 rsx1 = (f32x4)(0.0f);
    if (more) {
      if (ld_lo) *(LDSP u32x4 *)(lds->u.k.x[0] + st_x) = rx0;
      if (ld_hi) *(LDSP u32x4 *)(lds->u.k.x[0] + st_x + 4096u) = rx1;
      if (is_s) {
        const G u8 *Xs_nxt = rq_ptr_u(X_s + (ink_r17_u64)kbn * 4ull);
        rsx1 = rq_load_f32x4((const G float *)(Xs_nxt + sxoff));
      }
    }
    ink_r17_fl_decode_l0(ink_r17_fl_qp2(q_nxt, h), a_cur.z, d2, h, &xdc);
    __builtin_amdgcn_sched_barrier(0);
    ink_r17_fl_kstep(xdc, lds->u.k.x[1], lds->u.k.sx[1], 2u, lx_base0, wj, 0, acc);

    /* hb1 k=3: commit sx[0], decode, kstep(3)+lsignal, then prefetch a_cur/d_cur before lwait! */
    if (more) ink_r17_fl_stage_sx(lds->u.k.sx[0], sx_str, t0, is_s, rsx1);
    ink_r17_fl_decode_l0(ink_r17_fl_qp3(q_nxt, h), a_cur.w, d2, h, &xdc);
    __builtin_amdgcn_sched_barrier(0);
    ink_r17_fl_kstep(xdc, lds->u.k.x[1], lds->u.k.sx[1], 3u, lx_base0, wj, more, acc);

    if (!more) break;
    a_cur = *(const G u32x4 *)(p_aux_nxt + a_off);
    d_cur = (u32)(*(const G u16 *)(p_d_nxt + d_off));
    ink_r17_lwait();
  }
}

__attribute__((amdgpu_waves_per_eu(12, 16)))
KERNEL ie_inkling_gemm_rq_exp_iq3xxs_flute_m64_r17(
    const G u8 *qw, const G u8 *X, G u8 *Qout, u32 eff, u32 nb,
    const G u32 *list, const G u32 *tiles, const G u32 *n_tiles,
    u32 q_bytes, u32 xs, u32 qstr, u32 xdiv, float beta, float lbeta) {
  static LDS ink_r17_fl_lds_t ink_r17_fl_lds __attribute__((aligned(16)));
  if (!tiles || !list || !qw || !X || !Qout) return;

  const u32 gbase = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() * 64u));
  if (gbase >= eff || eff == 0u || (eff & 31u) != 0u || nb < 8u || (nb & 7u) != 0u || xdiv == 0u)
    return;

  u32 e, p0, p1;
  if (n_tiles) {
    /* Worklist mode (drop-in for flute_situ_r10 / flute_l0lut_r13). */
    const u32 ti = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
    const u32 nt_act = (u32)__builtin_amdgcn_readfirstlane((int)(*n_tiles));
    if (ti >= nt_act) return;
    const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)tiles[ti]);
    e = desc & 0xFFu;
    p0 = (desc >> 8u) & 0x1FFFu;
    const u32 nj_d = (desc >> 21u) & 0x7Fu;
    if (nj_d == 0u || nj_d > 64u) return;
    p1 = p0 + nj_d;
  } else {
    /* Expert-major mode: tiles == offs_w (b->ms_offs + e_lo), grid.y == cnt! */
    e = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
    p0 = (u32)__builtin_amdgcn_readfirstlane((int)tiles[e]);
    p1 = (u32)__builtin_amdgcn_readfirstlane((int)tiles[e + 1u]);
    if (p0 >= p1) return;
  }

  const u32 mb = nb >> 3u;
  const u32 nb_out = eff >> 5u;
  const u32 blk0 = gbase >> 5u;
  const u32 nblk = (eff - gbase >= 64u) ? 2u : 1u;

  /* FLUTE 3-plane SGPR base pointers (exact 98 B/row/sb, 128-B aligned!). */
  const ink_r17_u64 n_sb_tot = (ink_r17_u64)(eff >> 1u) * (ink_r17_u64)mb;
  const G u8 *p_qs_e  = rq_ptr_u(qw + (ink_r17_u64)e * (ink_r17_u64)q_bytes);
  const G u8 *p_aux_e = rq_ptr_u(p_qs_e + n_sb_tot * 256ull);
  const G u8 *p_d_e   = rq_ptr_u(p_qs_e + n_sb_tot * 384ull);
  const G u8 *X_u = rq_ptr_u(X);
  const G u8 *X_s = rq_ptr_u(X + (ink_r17_u64)nb * 32ull);

  u32 pb = p0;
  /* Primary BM=64 hot path (49..64 tokens per pass; loops back-to-back in expert-major mode!). */
#pragma unroll 1
  while (pb < p1 && p1 - pb > 48u) {
    const u32 pb_u = (u32)__builtin_amdgcn_readfirstlane((int)pb);
    const u32 rem = p1 - pb_u;
    const u32 nj = (rem < 64u) ? rem : 64u;
    f8 acc[4];
    ink_r17_fl_run_wj(&ink_r17_fl_lds, p_qs_e, p_aux_e, p_d_e, X_u, X_s, list,
                      pb_u, nj, gbase, eff, mb, xs, xdiv, 4u, acc);
    lbarrier();
    ink_r17_fl_epilogue_strip(&ink_r17_fl_lds.u.ep[0][0][0], ink_r17_fl_lds.pid, Qout,
                              0u, 32u, blk0, nblk, nb_out, qstr, beta, lbeta, 0u, acc);
    lbarrier();
    ink_r17_fl_epilogue_strip(&ink_r17_fl_lds.u.ep[0][0][0], ink_r17_fl_lds.pid, Qout,
                              32u, nj - 32u, blk0, nblk, nb_out, qstr, beta, lbeta, 2u, acc);
    pb = pb_u + nj;
    if (pb < p1) lbarrier();
  }

  /* Tail pass for 1..48 remaining tokens (or small-M tiles in worklist mode). */
  if (pb < p1) {
    const u32 pb_u = (u32)__builtin_amdgcn_readfirstlane((int)pb);
    const u32 nj = p1 - pb_u;
    f8 acc[3];
    if (nj > 32u) {
      ink_r17_fl_run_wj(&ink_r17_fl_lds, p_qs_e, p_aux_e, p_d_e, X_u, X_s, list,
                        pb_u, nj, gbase, eff, mb, xs, xdiv, 3u, acc);
      lbarrier();
      ink_r17_fl_epilogue_strip(&ink_r17_fl_lds.u.ep[0][0][0], ink_r17_fl_lds.pid, Qout,
                                0u, 32u, blk0, nblk, nb_out, qstr, beta, lbeta, 0u, acc);
      lbarrier();
      ink_r17_fl_epilogue_strip(&ink_r17_fl_lds.u.ep[0][0][0], ink_r17_fl_lds.pid, Qout,
                                32u, nj - 32u, blk0, nblk, nb_out, qstr, beta, lbeta, 2u, acc);
    } else if (nj > 16u) {
      ink_r17_fl_run_wj(&ink_r17_fl_lds, p_qs_e, p_aux_e, p_d_e, X_u, X_s, list,
                        pb_u, nj, gbase, eff, mb, xs, xdiv, 2u, acc);
      lbarrier();
      ink_r17_fl_epilogue_strip(&ink_r17_fl_lds.u.ep[0][0][0], ink_r17_fl_lds.pid, Qout,
                                0u, nj, blk0, nblk, nb_out, qstr, beta, lbeta, 0u, acc);
    } else {
      ink_r17_fl_run_wj(&ink_r17_fl_lds, p_qs_e, p_aux_e, p_d_e, X_u, X_s, list,
                        pb_u, nj, gbase, eff, mb, xs, xdiv, 1u, acc);
      lbarrier();
      ink_r17_fl_epilogue_strip(&ink_r17_fl_lds.u.ep[0][0][0], ink_r17_fl_lds.pid, Qout,
                                0u, nj, blk0, nblk, nb_out, qstr, beta, lbeta, 0u, acc);
    }
  }
}

/* ============================================================================
 * PART 3: ie_inkling_gemm_iq4nl_wave_m64_r17
 *   BM=64 (chunk=1024) Expert-Major Wave-Interleaved IQ4_NL MoE Down Prefill
 *   - Exact 17-arg drop-in for expmaj_r10 / wave_expmaj_r13 supporting BOTH
 *     128E Core + H38 Mode C (offs_w = b->ms_offs + e_lo, grid.y = cnt).
 *   - Grid: ((rows + 127u) / 128u, list ? cnt : (np + 63u) / 64u, 1u), NT=256.
 *   - 20,864 B LDS (3 WG/CU), <= 116 VGPR, 0x4B400000 exact magic-float (-32
 *     v_cvt_f32_i32/K-block), gfx1201 split barriers overlapping wa[0..3] loads
 *     + barrier behind k=3 WMMAs, f16_mix=0, exact-SHA!
 * ========================================================================== */

typedef struct {
  union {
    struct {
      u8    x[2u][64u * 128u] __attribute__((aligned(16)));
      f16   sw[2u][INK_R17_IQ4_KB * INK_R17_IQ4_SW_STR] __attribute__((aligned(16)));
      float sx[2u][INK_R17_IQ4_KB * 64u] __attribute__((aligned(32)));
    } k;
    float ep[32u][INK_R17_IQ4_EP_STR] __attribute__((aligned(32)));
  } u;
  u32 pid[64] __attribute__((aligned(16)));
} ink_r17_iq4_lds_t;

_Static_assert(sizeof(ink_r17_iq4_lds_t) == 20864u, "ink_r17_iq4_lds_t must be 20,864 B (3 WG/CU)");

/* Wave-Interleaved IQ4_NL flat block index (H36a exact layout, nb % 4 == 0). */
static inline __attribute__((always_inline)) u32 ink_r17_iq4_dblk0(
    const u32 r, const u32 nb, const u32 kb0) {
  const u32 rg = r >> 2u;
  const u32 rk = r & 3u;
  const u32 nfull8 = nb & ~7u;
  if (kb0 < nfull8) {
    return ((rg * nb + (kb0 & ~7u)) << 2u) + (rk << 3u) + (kb0 & 4u);
  }
  const u32 rem = nb & 7u;
  return ((rg * nb + nfull8) << 2u) + rk * rem;
}

static inline __attribute__((always_inline)) void ink_r17_iq4_stage_sw(
    LDSP f16 *lsw, const u32 t0, const int is_sw, const ink_r17_u2 rsw) {
  if (is_sw) {
    u32 t_loc = t0;
    __asm__ volatile("" : "+v"(t_loc));
    const u32 w0 = rsw.x, w1 = rsw.y;
    const u32 p0 = (u32)__builtin_amdgcn_update_dpp(0, (int)w0, 0xB1, 0xF, 0xF, true);
    const u32 p1 = (u32)__builtin_amdgcn_update_dpp(0, (int)w1, 0xB1, 0xF, 0xF, true);
    const u32 odd = t_loc & 1u;
    const u32 sel = odd ? 0x03020706u : 0x05040100u;
    const u32 v_a = __builtin_amdgcn_perm(p0, w0, sel);
    const u32 v_b = __builtin_amdgcn_perm(p1, w1, sel);
    const u32 r2 = t_loc & ~1u;
    LDSP u32 *sw32 = (LDSP u32 *)lsw;
    const u32 str32 = INK_R17_IQ4_SW_STR >> 1u;
    sw32[(0u + odd) * str32 + (r2 >> 1u)] = v_a;
    sw32[(2u + odd) * str32 + (r2 >> 1u)] = v_b;
  }
}

static inline __attribute__((always_inline)) void ink_r17_iq4_stage_scales(
    LDSP f16 *lsw, LDSP float *lsx, const u32 t0, const u32 sx_stride,
    const int is_sw, const int is_sx, const ink_r17_u2 rsw, const f32x4 rsx) {
  ink_r17_iq4_stage_sw(lsw, t0, is_sw, rsw);
  if (is_sx) {
    u32 t_loc = t0;
    __asm__ volatile("" : "+v"(t_loc));
    lsx[0u * sx_stride + t_loc] = rsx.x;
    lsx[1u * sx_stride + t_loc] = rsx.y;
    lsx[2u * sx_stride + t_loc] = rsx.z;
    lsx[3u * sx_stride + t_loc] = rsx.w;
  }
}

/* Pinned f16 -> f32 scale unpack: guarantees 0 v_fma_mix_f32 in K-loop! */
static inline __attribute__((always_inline)) f8 ink_r17_iq4_load_sw_f8(
    const LDSP f16 *lsw, const u32 k, const u32 wr, const u32 h) {
  const h8 sw_h = *(const LDSP h8 *)(lsw + k * INK_R17_IQ4_SW_STR + wr + 8u * h);
  f8 sw = __builtin_convertvector(sw_h, f8);
  __asm__ volatile("" : "+v"(sw));
  return sw;
}

static inline __attribute__((always_inline)) void ink_r17_iq4_decode_w(
    const u32x4 wk, const u32 h, v2i *a0_out, v2i *a1_out) {
  u32x4 nb4 = h ? ((wk >> 4u) & 0x0F0F0F0Fu) : (wk & 0x0F0F0F0Fu);
  nb4.x = ie_iq4_map(nb4.x);
  nb4.y = ie_iq4_map(nb4.y);
  nb4.z = ie_iq4_map(nb4.z);
  nb4.w = ie_iq4_map(nb4.w);
  *a0_out = (v2i){(int)nb4.x, (int)nb4.y};
  *a1_out = (v2i){(int)nb4.z, (int)nb4.w};
}

/* IQ4_NL WMMA K-step: exact 0x4B400000 magic-float on wj=1..3, inline-zero
 * zm=(v8i)(0) on wj=4 (saves 8 VGPRs -> 0/0 spills at 3 WGs/CU!). Optional
 * split-barrier signal on k == 3 overlaps WG barrier behind final WMMAs! */
static inline __attribute__((always_inline)) void ink_r17_iq4_k_step(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 lx_base0, const u32 wj, const u32 sx_str,
    const int do_sig, f8 *acc) {
  const v8i zm = (wj == 4u) ? (v8i)(0) : (v8i)(0x4B400000);
  const u32 l_loc = ink_r17_lane();
  const u32 l16 = l_loc & 15u, h = l_loc >> 4u;
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * sx_str + l16;

  v2i a0, a1;
  ink_r17_iq4_decode_w(wk, h, &a0, &a1);
  const f8 sw = ink_r17_iq4_load_sw_f8(lsw, k, wr, h);

  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + 0u * 2048u);
  const float dx0 = lsx_k[0u];
  if (wj == 1u) {
    if (do_sig) {
      ink_r17_lsignal();
    } else {
      __builtin_amdgcn_sched_barrier(0);
    }
    const v2i b0_lo = {(int)b0.x, (int)b0.y}, b0_hi = {(int)b0.z, (int)b0.w};
    v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_lo, zm, 0);
    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_hi, c0, 0);
    __builtin_amdgcn_sched_barrier(0);
    acc[0] += (sw * dx0) * (__builtin_bit_cast(f8, c0) - 12582912.0f);
    __asm__ volatile("" : : "v"(acc[0]) : "memory");
    __builtin_amdgcn_sched_barrier(0);
    return;
  }

  const u32x4 b1 = *(const LDSP u32x4 *)(lx_k + 1u * 2048u);
  const float dx1 = lsx_k[16u];
  if (wj == 2u && do_sig) {
    ink_r17_lsignal();
  } else {
    __builtin_amdgcn_sched_barrier(0);
  }
  const v2i b0_lo = {(int)b0.x, (int)b0.y}, b0_hi = {(int)b0.z, (int)b0.w};
  const v2i b1_lo = {(int)b1.x, (int)b1.y}, b1_hi = {(int)b1.z, (int)b1.w};
  v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_lo, zm, 0);
  v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b1_lo, zm, 0);
  c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_hi, c0, 0);
  c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1_hi, c1, 0);
  __builtin_amdgcn_sched_barrier(0);
  acc[0] += (sw * dx0) * ((wj == 4u) ? __builtin_convertvector(c0, f8) : (__builtin_bit_cast(f8, c0) - 12582912.0f));
  __asm__ volatile("" : : "v"(acc[0]));
  __builtin_amdgcn_sched_barrier(0);
  acc[1] += (sw * dx1) * ((wj == 4u) ? __builtin_convertvector(c1, f8) : (__builtin_bit_cast(f8, c1) - 12582912.0f));
  __asm__ volatile("" : : "v"(acc[1]) : "memory");
  __builtin_amdgcn_sched_barrier(0);
  if (wj == 2u) return;

  const u32x4 b2 = *(const LDSP u32x4 *)(lx_k + 2u * 2048u);
  const float dx2 = lsx_k[32u];
  if (wj == 3u) {
    if (do_sig) {
      ink_r17_lsignal();
    } else {
      __builtin_amdgcn_sched_barrier(0);
    }
    const v2i b2_lo = {(int)b2.x, (int)b2.y}, b2_hi = {(int)b2.z, (int)b2.w};
    v8i c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b2_lo, zm, 0);
    c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b2_hi, c2, 0);
    __builtin_amdgcn_sched_barrier(0);
    acc[2] += (sw * dx2) * (__builtin_bit_cast(f8, c2) - 12582912.0f);
    __asm__ volatile("" : : "v"(acc[2]) : "memory");
    __builtin_amdgcn_sched_barrier(0);
    return;
  }

  const u32x4 b3 = *(const LDSP u32x4 *)(lx_k + 3u * 2048u);
  const float dx3 = lsx_k[48u];
  if (do_sig) {
    ink_r17_lsignal();
  } else {
    __builtin_amdgcn_sched_barrier(0);
  }
  const v2i b2_lo = {(int)b2.x, (int)b2.y}, b2_hi = {(int)b2.z, (int)b2.w};
  const v2i b3_lo = {(int)b3.x, (int)b3.y}, b3_hi = {(int)b3.z, (int)b3.w};
  v8i c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b2_lo, zm, 0);
  v8i c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b3_lo, zm, 0);
  c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b2_hi, c2, 0);
  c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b3_hi, c3, 0);
  __builtin_amdgcn_sched_barrier(0);
  acc[2] += (sw * dx2) * __builtin_convertvector(c2, f8);
  __asm__ volatile("" : : "v"(acc[2]));
  __builtin_amdgcn_sched_barrier(0);
  acc[3] += (sw * dx3) * __builtin_convertvector(c3, f8);
  __asm__ volatile("" : : "v"(acc[3]) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r17_iq4_run_wj(
    LDSP ink_r17_iq4_lds_t *lds, const G u8 *qw_e, const G u8 *qs_e,
    const G u8 *X_u, const G u8 *X_s, const G u32 *list,
    const u32 pb, const u32 nj, const u32 rbase, const u32 rows, const u32 nb,
    const u32 xs, const u32 xdiv, const u32 wj, f8 *acc) {
  u32 t0 = tid();
  __asm__ volatile("" : "+v"(t0));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = t0 & 31u, l16 = l & 15u, h = l >> 4u;
  const u32 wr = 16u * w;
  const int is_sw = (w < 4u);
  const u32 n_sx = wj << 4u;
  const int is_sx = (t0 < n_sx);
  const u32 sx_str = (wj > 2u) ? 64u : 32u;
  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  const int ld_lo = (wj >= 2u) || (tk_ld < 16u);
  const int ld_hi = (wj == 4u) || ((wj == 3u) && (tk_ld < 16u));

  if (is_sx) {
    const u32 pj = pb + (t0 < nj ? t0 : 0u);
    lds->pid[t0] = list ? list[pj] : pj;
  }
  lbarrier();

  const u32 rw_idx = rbase + wr + l16;
  const u32 rw_cl = rw_idx < rows ? rw_idx : (rows - 1u);
  const u32 sr = rbase + (is_sw ? t0 : 0u);
  const u32 sr_cl = sr < rows ? sr : (rows - 1u);
  const u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);
  const u32 lx_base0 = l16 * 128u + ((h ^ (l16 & 7u)) << 4u);

  const u32 xoff0 = ld_lo ? (ink_r17_div_x(lds->pid[tk_ld], xdiv) * xs + (c8_ld << 4u)) : 0u;
  const u32 xoff1 = ld_hi ? (ink_r17_div_x(lds->pid[tk_ld + 32u], xdiv) * xs + (c8_ld << 4u)) : 0u;
  const u32 sxoff = is_sx ? (ink_r17_div_x(lds->pid[t0], xdiv) * xs) : 0u;

#pragma unroll
  for (u32 j = 0u; j < wj; j++) acc[j] = (f8)(0.0f);

  u32x4 wa[INK_R17_IQ4_KB];
  {
    const u32 db_w0 = ink_r17_iq4_dblk0(rw_cl, nb, 0u);
    const G u8 *qw0_p = qw_e + (ink_r17_u64)db_w0 * 16ull;
#pragma unroll
    for (u32 k = 0u; k < INK_R17_IQ4_KB; k++) wa[k] = ink_r17_ld4(qw0_p + k * 16u);
    const u32x4 rx0_i = ld_lo ? ink_r17_ld4(X_u + xoff0) : (u32x4)(0u);
    const u32x4 rx1_i = ld_hi ? ink_r17_ld4(X_u + xoff1) : (u32x4)(0u);
    const u32 db_s0 = is_sw ? ink_r17_iq4_dblk0(sr_cl, nb, 0u) : 0u;
    const ink_r17_u2 rsw_i = is_sw ? ink_r17_ld2(qs_e + (ink_r17_u64)db_s0 * 2ull) : (ink_r17_u2){0u, 0u};
    const f32x4 rsx_i = is_sx ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
    if (ld_lo) *(LDSP u32x4 *)(lds->u.k.x[0] + st_x) = rx0_i;
    if (ld_hi) *(LDSP u32x4 *)(lds->u.k.x[0] + st_x + 4096u) = rx1_i;
    ink_r17_iq4_stage_scales(lds->u.k.sw[0], lds->u.k.sx[0], t0, sx_str, is_sw, is_sx, rsw_i, rsx_i);
  }
  lbarrier();

  u32 bf = 0u;
  u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R17_IQ4_KB) {
    const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R17_IQ4_KB));
    const int more = (kb_n < nb);
    /* Force wa[0..1] ready before issuing rx0/rx1 so s_wait_loadcnt does not stall k=0 on rx0/rx1. */
    __asm__ volatile("" : : "v"(wa[0]), "v"(wa[1]));
    if (more) {
      const G u8 *X_n = rq_ptr_u(X_u + (ink_r17_u64)kb_n * 32ull);
      if (ld_lo) rx0 = ink_r17_ld4(X_n + xoff0);
      if (ld_hi) rx1 = ink_r17_ld4(X_n + xoff1);
    }
    __builtin_amdgcn_sched_barrier(0);
    const LDSP u8 *lx = lds->u.k.x[bf];
    const LDSP f16 *lsw = lds->u.k.sw[bf];
    const LDSP float *lsx = lds->u.k.sx[bf];

    ink_r17_iq4_k_step(wa[0], lx, lsw, lsx, 0u, wr, lx_base0, wj, sx_str, 0, acc);
    ink_r17_iq4_k_step(wa[1], lx, lsw, lsx, 1u, wr, lx_base0, wj, sx_str, 0, acc);

    /* Commit next X and prefetch next scales after k=1 (exact 0-spill wave_expmaj schedule!). */
    ink_r17_u2 rsw = {0u, 0u};
    f32x4 rsx = (f32x4)(0.0f);
    if (more) {
      const u32 nbf = bf ^ 1u;
      if (ld_lo) *(LDSP u32x4 *)(lds->u.k.x[nbf] + st_x) = rx0;
      if (ld_hi) *(LDSP u32x4 *)(lds->u.k.x[nbf] + st_x + 4096u) = rx1;
      const u32 db_sn = is_sw ? ink_r17_iq4_dblk0(sr_cl, nb, kb_n) : 0u;
      const G u8 *Xs_n = rq_ptr_u(X_s + (ink_r17_u64)kb_n * 4ull);
      if (is_sw) rsw = ink_r17_ld2(qs_e + (ink_r17_u64)db_sn * 2ull);
      if (is_sx) rsx = rq_load_f32x4((const G float *)(Xs_n + sxoff));
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r17_iq4_k_step(wa[2], lx, lsw, lsx, 2u, wr, lx_base0, wj, sx_str, 0, acc);

    u32 db_wn = 0u;
    if (more) {
      const u32 nbf = bf ^ 1u;
      ink_r17_iq4_stage_scales(lds->u.k.sw[nbf], lds->u.k.sx[nbf], t0, sx_str, is_sw, is_sx, rsw, rsx);
      __builtin_amdgcn_sched_barrier(0);
      db_wn = ink_r17_iq4_dblk0(rw_cl, nb, kb_n);
      const G u8 *qw_n = qw_e + (ink_r17_u64)db_wn * 16ull;
      wa[0] = ink_r17_ld4(qw_n + 0u);
      wa[1] = ink_r17_ld4(qw_n + 16u);
    }
    __builtin_amdgcn_sched_barrier(0);
    /* k=3: all writes to nbf issued; split-barrier signal fires inside k_step BEFORE final WMMAs! */
    ink_r17_iq4_k_step(wa[3], lx, lsw, lsx, 3u, wr, lx_base0, wj, sx_str, more, acc);
    if (more) {
      const G u8 *qw_n = qw_e + (ink_r17_u64)db_wn * 16ull;
      wa[2] = ink_r17_ld4(qw_n + 32u);
      wa[3] = ink_r17_ld4(qw_n + 48u);
      ink_r17_lwait();
      bf ^= 1u;
    }
  }
}

static inline __attribute__((always_inline)) void ink_r17_iq4_store_strip(
    LDSP float *ep, const LDSP u32 *lpid, G u8 *Y, const G u8 *res,
    const u32 rbase, const u32 rows, const u32 ys, const u32 rs,
    const u32 strip_base, const u32 nj_strip, const u32 j_base, const f8 *acc) {
  u32 t_ep = tid();
  __asm__ volatile("" : "+v"(t_ep));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t_ep >> 5u));
  const u32 l = t_ep & 31u, l16 = l & 15u, h = l >> 4u;
  const u32 r_wr = 16u * w + 8u * h;

#pragma unroll
  for (u32 j_loc = 0u; j_loc < 2u; j_loc++) {
    const u32 slot_loc = 16u * j_loc + l16;
    if (slot_loc < nj_strip) {
      LDSP float *dst = ep + slot_loc * INK_R17_IQ4_EP_STR + r_wr;
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) dst[v] = acc[j_base + j_loc][v];
    }
  }
  lbarrier();

  const int vec_ok = (rbase + 128u <= rows) && ((ys & 15u) == 0u) && ((((unsigned long)Y) & 15ul) == 0ul) &&
                     (!res || (((rs & 15u) == 0u) && ((((unsigned long)res) & 15ul) == 0ul)));
  if (__builtin_expect(vec_ok, 1)) {
    for (u32 tk = w; tk < nj_strip; tk += 8u) {
      const u32 pid = (u32)__builtin_amdgcn_readfirstlane((int)lpid[strip_base + tk]);
      const LDSP float *src = ep + tk * INK_R17_IQ4_EP_STR + 4u * l;
      f32x4 v = {src[0], src[1], src[2], src[3]};
      if (res) {
        const G f32x4 *rr4 = (const G f32x4 *)((const G float *)(res + (ink_r17_u64)pid * (ink_r17_u64)rs) + rbase);
        v += rr4[l];
      }
      G f32x4 *yr4 = (G f32x4 *)((G float *)(Y + (ink_r17_u64)pid * (ink_r17_u64)ys) + rbase);
      yr4[l] = v;
    }
  } else {
    for (u32 tk = w; tk < nj_strip; tk += 8u) {
      const u32 pid = (u32)__builtin_amdgcn_readfirstlane((int)lpid[strip_base + tk]);
      const LDSP float *src = ep + tk * INK_R17_IQ4_EP_STR + 4u * l;
      G float *yr = (G float *)(Y + (ink_r17_u64)pid * (ink_r17_u64)ys);
      const G float *rr = res ? (const G float *)(res + (ink_r17_u64)pid * (ink_r17_u64)rs) : (const G float *)0;
#pragma unroll 1
      for (u32 c = 0u; c < 4u; c++) {
        const u32 r = rbase + 4u * l + c;
        if (r < rows) yr[r] = epilogue(src[c], r, 0, rr);
      }
    }
  }
}

__attribute__((amdgpu_waves_per_eu(12, 16)))
KERNEL ie_inkling_gemm_iq4nl_wave_m64_r17(
    const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
    u32 rows, u32 nb, const G u32 *list, const G u32 *offs, u32 np,
    u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv, u32 kind,
    const G u8 *res, u32 rs) {
  static LDS ink_r17_iq4_lds_t ink_r17_iq4_lds __attribute__((aligned(16)));
  (void)kind;
  if (!qw0 || !qs0 || !X || !Y || (list && !offs) || rows == 0u || (rows & 3u) != 0u ||
      nb == 0u || (nb & 3u) != 0u || xdiv == 0u)
    return;

  const u32 e = list ? (u32)__builtin_amdgcn_readfirstlane((int)tokid()) : 0u;
  const u32 p0 = list ? (u32)__builtin_amdgcn_readfirstlane((int)offs[e])
                      : (u32)__builtin_amdgcn_readfirstlane((int)(tokid() * 64u));
  const u32 p1 = list ? (u32)__builtin_amdgcn_readfirstlane((int)offs[e + 1u])
                      : ((p0 + 64u < np) ? (p0 + 64u) : np);
  const u32 rbase = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() * INK_R17_IQ4_R));
  if (p0 >= p1 || rbase >= rows) return;

  const G u8 *qw_e = rq_ptr_u((const G u8 *)(qw0 + (ink_r17_u64)e * (ink_r17_u64)qwn));
  const G u8 *qs_e = rq_ptr_u((const G u8 *)(qs0 + (ink_r17_u64)e * (ink_r17_u64)qsn));
  const G u8 *X_u = rq_ptr_u(X);
  const G u8 *X_s = rq_ptr_u(X + (ink_r17_u64)nb * 32ull);

  u32 pb = p0;
  /* Primary BM=64 expert-major hot loop (reads IQ4_NL once per 64-token tile!). */
#pragma unroll 1
  while (pb < p1 && p1 - pb > 48u) {
    const u32 pb_u = (u32)__builtin_amdgcn_readfirstlane((int)pb);
    const u32 rem = p1 - pb_u;
    const u32 nj = (rem < 64u) ? rem : 64u;
    f8 acc[4];
    ink_r17_iq4_run_wj(&ink_r17_iq4_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                       rbase, rows, nb, xs, xdiv, 4u, acc);
    lbarrier();
    ink_r17_iq4_store_strip(&ink_r17_iq4_lds.u.ep[0][0], ink_r17_iq4_lds.pid, Y, res,
                            rbase, rows, ys, rs, 0u, 32u, 0u, acc);
    lbarrier();
    ink_r17_iq4_store_strip(&ink_r17_iq4_lds.u.ep[0][0], ink_r17_iq4_lds.pid, Y, res,
                            rbase, rows, ys, rs, 32u, nj - 32u, 2u, acc);
    pb = pb_u + nj;
    if (pb < p1) lbarrier();
  }

  /* Tail pass for 1..48 remaining tokens. */
  if (pb < p1) {
    const u32 pb_u = (u32)__builtin_amdgcn_readfirstlane((int)pb);
    const u32 nj = p1 - pb_u;
    f8 acc[3];
    if (nj > 32u) {
      ink_r17_iq4_run_wj(&ink_r17_iq4_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                         rbase, rows, nb, xs, xdiv, 3u, acc);
      lbarrier();
      ink_r17_iq4_store_strip(&ink_r17_iq4_lds.u.ep[0][0], ink_r17_iq4_lds.pid, Y, res,
                              rbase, rows, ys, rs, 0u, 32u, 0u, acc);
      lbarrier();
      ink_r17_iq4_store_strip(&ink_r17_iq4_lds.u.ep[0][0], ink_r17_iq4_lds.pid, Y, res,
                              rbase, rows, ys, rs, 32u, nj - 32u, 2u, acc);
    } else if (nj > 16u) {
      ink_r17_iq4_run_wj(&ink_r17_iq4_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                         rbase, rows, nb, xs, xdiv, 2u, acc);
      lbarrier();
      ink_r17_iq4_store_strip(&ink_r17_iq4_lds.u.ep[0][0], ink_r17_iq4_lds.pid, Y, res,
                              rbase, rows, ys, rs, 0u, nj, 0u, acc);
    } else {
      ink_r17_iq4_run_wj(&ink_r17_iq4_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                         rbase, rows, nb, xs, xdiv, 1u, acc);
      lbarrier();
      ink_r17_iq4_store_strip(&ink_r17_iq4_lds.u.ep[0][0], ink_r17_iq4_lds.pid, Y, res,
                              rbase, rows, ys, rs, 0u, nj, 0u, acc);
    }
  }
}

#undef INK_R17_IQ4_EP_STR
#undef INK_R17_IQ4_SW_STR
#undef INK_R17_IQ4_R
#undef INK_R17_IQ4_KB
#undef INK_R17_ATT_QSC
#undef INK_R17_ATT_LOG2E
#undef INK_R17_ATT_ROWB
#undef INK_R17_ATT_NB16
#undef INK_R17_ATT_GQA
#undef INK_R17_ATT_DREL
#undef INK_R17_ATT_HD

#endif /* !IE_MEGA */
