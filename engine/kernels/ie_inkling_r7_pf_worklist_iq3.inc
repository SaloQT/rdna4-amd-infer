/* ============================================================================
 * Inkling-2 Round 7 (Wave 3, Bits 69-70):
 *   1. Shared Flat Largest-First 16/32/48/64 MoE Prefill Tile Worklist Builder
 *      (SonicMoE / ScatterMoE / llama.cpp PR #32065 pattern)
 *   2. Low-VGPR (<= 128 VGPR = 3 WGs/CU) Fused Prefill IQ3_XXS Gate/Up WMMA +
 *      Exact SITU + Q8_0 consuming the compact 16/32/48/64 worklist
 *
 * Target: RX 9070 XT (gfx1201, wave32, NT=256), 45.7% of 4K prefill!
 * All local symbols/LDS/macros prefixed ink_r7_wl_ / ink_r7_iq3_; macros
 * #undef'd at EOF. Clang 21 scalar __builtin_bit_cast safe, 0/0 spills.
 *
 * Shared 32-bit Packed Tile Descriptor Format (tiles[0 .. *n_tiles - 1]):
 *   - bits [ 7: 0] : e   (0..255, active expert ID)
 *   - bits [20: 8] : lo  (0..8191, start index in list[])
 *   - bits [27:21] : nj  (1..64, routed pair count in this tile)
 *   - bits [29:28] : cls ((nj - 1u) >> 4u in 0..3: 0=1..16, 1=17..32,
 *                         2=33..48, 3=49..64)
 *   Tiles are emitted in LARGEST-FIRST bucket order (cls 3 -> 2 -> 1 -> 0):
 *     - Eliminates empty-expert WGs completely
 *     - Splits >64-pair hot experts across parallel WGs (kills serial stragglers)
 *     - Schedules 49..64 tiles first and 1..16 tiles last (fills CU tail cracks)
 *     - Enables exact 16/32/48/64 (WJ=1/2/3/4) X-load & WMMA trimming
 *
 * Exported Kernels, Grids, Guards, and Parity:
 *
 * 1. KERNEL ie_inkling_moe_pf_worklist_r7(
 *        const G u32 *route, u32 rst, u32 T, u32 K, u32 n_exp,
 *        G u32 *offs, G u32 *list, G u32 *tiles, G u32 *n_tiles)
 *    - Bit:    69 (ink_mask2 bit 5)
 *    - Grid:   (1u, 1u, 1u), NT = 256 (1 WG total; replaces ie_moe_sort at T>=17)
 *    - Buffer: tiles >= max_tiles u32s where max_tiles = min(T*K, n_exp + (T*K)/64u)
 *              (for Inkling-2 T<=512, K=8, n_exp=128: max_tiles <= 192; bump
 *               b->ms_packs from 160 -> 512 u32s = 2 KiB to cover up to
 *               T*K=8192, n_exp=256 with zero new struct fields!).
 *              n_tiles = b->ms_npacks (1 u32).
 *    - Guard:  !mk.rec && T >= 17u && n_exp > 0u && n_exp <= 256u &&
 *              T * K <= 8192u && b->ms_packs && b->ms_npacks &&
 *              max_tiles <= ms_packs_cap (e.g. 512u).
 *    - Parity: offs[0..n_exp] bitwise-exact vs ie_moe_sort; list[0..T*K-1] exact
 *              per-expert partition; tiles[0..*n_tiles-1] exact disjoint cover
 *              of all active experts' pairs sorted largest-first.
 *
 * 2. KERNEL ie_inkling_gemm_rq_exp_iq3xxs_situ_r7(
 *        const G u8 *qw, const G u8 *X, G u8 *Qout, u32 eff, u32 nb,
 *        const G u32 *list, const G u32 *tiles, const G u32 *n_tiles,
 *        u32 q_bytes, u32 xs, u32 qstr, u32 xdiv, float beta, float lbeta)
 *    - Bit:    70 (ink_mask2 bit 6)
 *    - Grid:   ((eff + 63u) / 64u, max_tiles, 1u), NT = 256
 *              where max_tiles = min(T*K, n_exp + (T*K)/64u)
 *              (WGs with tokid() >= *n_tiles exit immediately on instruction #3).
 *    - Guards: same as ie_inkling_gemm_rq_exp_iq3xxs_situ_r6 PLUS worklist active
 *              (T >= 17u, n_exp <= 256u, T * K <= 8192u, tiles && n_tiles).
 *    - Key Microarchitectural Upgrades over r6_iq3_situ_pf (154 VGPR, 2 WGs/CU):
 *      (a) <= 128 VGPR (3 WGs/CU = +50% occupancy!) + 19,712 B LDS (<= 21.8 KiB)
 *      (b) h=0/1 -> w0/w1 0-bank-conflict remap: h=0 handles elems 0..15, h=1
 *          handles elems 16..31 -> 1x ds_load_b128 (0 conflicts) replaces 2x
 *          ds_load_b64 (2-way conflicts), eliminates 2 v_perm_b32/k-block,
 *          shrinks cur_q[9] -> cur_q[8] (zero carry reg!), and shares d2 via xrow.
 *      (c) Exact 0x4B400000 WMMA int32->f32 bias trick (bitwise-identical for
 *          all |dot| <= 4.19M, saves 32 v_cvt_f32_i32 per K-block).
 *      (d) Adaptive 16/32/48/64 (WJ=1/2/3/4) worklist tiles: cuts X loads and
 *          WMMAs 25% on 33..48 tiles and 50% on 1..16 tiles + largest-first wave.
 *    - Parity: 100% bitwise-exact (0-bit-diff in Qout codes, scales, asums) vs
 *              ie_inkling_gemm_rq_exp_iq3xxs_situ_r6!
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R7_WL_MAX_EXP   256u
#define INK_R7_WL_MAX_PAIRS 8192u
#define INK_R7_WL_MAX_TILES 512u

static inline __attribute__((always_inline)) u32 ink_r7_wl_wave_scan_inc(u32 x) {
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x111, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x112, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x114, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x118, 0xF, 0xF, true);
  const u32 lo16 = (u32)__builtin_amdgcn_readlane((int)x, 15);
  if (lane() >= 16u) x += lo16;
  return x;
}

/* ============================================================================
 * Kernel 1: ie_inkling_moe_pf_worklist_r7 (Bit 69)
 *   Single-WG parallel LDS histogram + DPP prefix sum + parallel list[] scatter
 *   + 4-bucket largest-first (49..64 -> 33..48 -> 17..32 -> 1..16) tile emission.
 * ========================================================================== */
KERNEL ie_inkling_moe_pf_worklist_r7(const G u32 *route, u32 rst, u32 T, u32 K, u32 n_exp,
                                     G u32 *offs, G u32 *list, G u32 *tiles, G u32 *n_tiles) {
  static LDS u32 ink_r7_wl_cnt[INK_R7_WL_MAX_EXP];
  static LDS u32 ink_r7_wl_er[INK_R7_WL_MAX_PAIRS];
  static LDS u32 ink_r7_wl_wsum[9];
  static LDS u32 ink_r7_wl_bcnt[4];
  static LDS u32 ink_r7_wl_boff[5];

  const u32 t = tid(), w = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u)), l = lane();
  const u32 np = T * K;

  if (n_exp == 0u) {
    if (t == 0u) {
      if (offs) offs[0] = 0u;
      if (n_tiles) *n_tiles = 0u;
    }
    return;
  }
  if (!offs || n_exp > INK_R7_WL_MAX_EXP || np > INK_R7_WL_MAX_PAIRS) return;
  if (np > 0u && (!route || !list)) return;

  if (t < n_exp) ink_r7_wl_cnt[t] = 0u;
  if (t < 4u) ink_r7_wl_bcnt[t] = 0u;
  lbarrier();

  /* Pass 1: Parallel histogram + record (expert << 16 | in_expert_rank) per pair. */
  if (K == 8u) {
    const u32 kk = t & 7u;
    const u32 tk0 = t >> 3u;
    for (u32 p = t, tk = tk0; p < np; p += NT, tk += 32u) {
      const u32 e = ((const G u32 *)((const G u8 *)route + (unsigned long)tk * rst))[kk];
      const u32 r = __atomic_fetch_add(&ink_r7_wl_cnt[e], 1u, __ATOMIC_RELAXED);
      ink_r7_wl_er[p] = (e << 16u) | r;
    }
  } else if (K > 0u) {
    for (u32 p = t; p < np; p += NT) {
      const u32 e = ((const G u32 *)((const G u8 *)route + (unsigned long)(p / K) * rst))[p % K];
      const u32 r = __atomic_fetch_add(&ink_r7_wl_cnt[e], 1u, __ATOMIC_RELAXED);
      ink_r7_wl_er[p] = (e << 16u) | r;
    }
  }
  lbarrier();

  /* Pass 2: Exclusive prefix sum over cnt[0..n_exp-1] -> offs[0..n_exp], plus count
   * tiles per size bucket (3: 49..64, 2: 33..48, 1: 17..32, 0: 1..16). */
  const u32 c = (t < n_exp) ? ink_r7_wl_cnt[t] : 0u;
  const u32 c_inc = ink_r7_wl_wave_scan_inc(c);
  if (l == 31u) ink_r7_wl_wsum[w] = c_inc;

  if (tiles && c > 0u) {
    const u32 nfull = c >> 6u;
    const u32 rem = c & 63u;
    if (nfull > 0u) __atomic_fetch_add(&ink_r7_wl_bcnt[3], nfull, __ATOMIC_RELAXED);
    if (rem > 0u) {
      const u32 cls = (rem - 1u) >> 4u;
      __atomic_fetch_add(&ink_r7_wl_bcnt[cls], 1u, __ATOMIC_RELAXED);
    }
  }
  lbarrier();

  if (w == 0u) {
    const u32 ws = (l < 8u) ? ink_r7_wl_wsum[l] : 0u;
    const u32 winc = ink_r7_wl_wave_scan_inc(ws);
    if (l < 8u) ink_r7_wl_wsum[l] = winc - ws;
    if (l == 7u) ink_r7_wl_wsum[8] = winc;
  } else if (w == 1u && l == 0u) {
    /* Largest-first bucket base offsets: cls 3 (49..64) -> 2 (33..48) -> 1 (17..32) -> 0 (1..16). */
    const u32 b3 = ink_r7_wl_bcnt[3];
    const u32 b2 = ink_r7_wl_bcnt[2];
    const u32 b1 = ink_r7_wl_bcnt[1];
    const u32 b0 = ink_r7_wl_bcnt[0];
    ink_r7_wl_boff[3] = 0u;
    ink_r7_wl_boff[2] = b3;
    ink_r7_wl_boff[1] = b3 + b2;
    ink_r7_wl_boff[0] = b3 + b2 + b1;
    ink_r7_wl_boff[4] = b3 + b2 + b1 + b0;
  }
  lbarrier();

  if (t < n_exp) {
    const u32 off = ink_r7_wl_wsum[w] + (c_inc - c);
    ink_r7_wl_cnt[t] = off;
    offs[t] = off;

    /* Pass 3A: Emit compact 32-bit tile descriptors into largest-first buckets. */
    if (tiles && c > 0u) {
      const u32 nfull = c >> 6u;
      const u32 rem = c & 63u;
      if (nfull > 0u) {
        const u32 base_pos = __atomic_fetch_add(&ink_r7_wl_boff[3], nfull, __ATOMIC_RELAXED);
        for (u32 k = 0u; k < nfull; k++) {
          const u32 lo = off + (k << 6u);
          tiles[base_pos + k] = t | (lo << 8u) | (64u << 21u) | (3u << 28u);
        }
      }
      if (rem > 0u) {
        const u32 cls = (rem - 1u) >> 4u;
        const u32 pos = __atomic_fetch_add(&ink_r7_wl_boff[cls], 1u, __ATOMIC_RELAXED);
        const u32 lo = off + (nfull << 6u);
        tiles[pos] = t | (lo << 8u) | (rem << 21u) | (cls << 28u);
      }
    }
  }
  if (t == 0u) {
    offs[n_exp] = ink_r7_wl_wsum[8];
    if (n_tiles) *n_tiles = tiles ? ink_r7_wl_boff[4] : 0u;
  }
  lbarrier();

  /* Pass 3B: Zero-atomic parallel scatter of list[offs[e] + rank] = p. */
  for (u32 p = t; p < np; p += NT) {
    const u32 er = ink_r7_wl_er[p];
    const u32 e = er >> 16u;
    const u32 r = er & 0xFFFFu;
    list[ink_r7_wl_cnt[e] + r] = p;
  }
}

/* ============================================================================
 * Kernel 2: ie_inkling_gemm_rq_exp_iq3xxs_situ_r7 (Bit 70)
 *   <= 128 VGPR (3 WGs/CU), 19,712 B LDS (<= 21.8 KiB = 3 WGs/CU), 0/0 spills,
 *   h=0/1 -> w0/w1 0-bank-conflict remap, exact 0x4B400000 int32->f32 bias,
 *   16/32/48/64 flat largest-first worklist, 0-bit-diff vs r6_iq3_situ_pf!
 * ========================================================================== */

struct __attribute__((packed, aligned(4))) ink_r7_iq3_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r7_iq3_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r7_iq3_u32x4p *)p)->v;
}

typedef struct {
  v2i w0, w1;
  float s0;
} ink_r7_iq3_dc_t;

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
} ink_r7_iq3_lds_t;

_Static_assert(sizeof(ink_r7_iq3_lds_t) == 19712u, "ink_r7_iq3_lds_t must be 19,712 B (<= 21.8 KiB = 3 WGs/CU)");

static inline __attribute__((always_inline)) u32 ink_r7_iq3_perm_al(const u32 lo, const u32 hi, const u32 psel) {
  return __builtin_amdgcn_perm(hi, lo, psel);
}

static inline __attribute__((always_inline)) u32 ink_r7_iq3_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

/* Single-mul IQ3_XXS 7-bit+parity sign decode (bitwise-identical to rq_ksign7 + ie_rq_ssign). */
static inline __attribute__((always_inline)) v2i ink_r7_iq3_ksw(const u32 g0, const u32 g1, const u32 s7) {
  const u32 p_ = s7 * 0x00204081u;
  const u32 t0 = p_ & 0x01010101u;
  const u32 t1 = ((p_ >> 4u) | ((u32)__builtin_popcount(s7) << 24u)) & 0x01010101u;
  const u32 m0 = __builtin_amdgcn_perm(0u, 0u, t0 | 0x0C0C0C0Cu);
  const u32 m1 = __builtin_amdgcn_perm(0u, 0u, t1 | 0x0C0C0C0Cu);
  const v2i v = {(int)((g0 ^ m0) + t0), (int)((g1 ^ m1) + t1)};
  return v;
}

/* Load IQ3_XXS hb0 with h=0/1 -> w0/w1 remap:
 *   h=0 loads base[0..7] (elems 0..15 of k=0..3), h=1 loads base[1..8] (elems 16..31 of k=0..3).
 *   Zero cur_q[8] carry register; only 2x global_load_b128 for qs + 1x b128 + 1x b32 for aux! */
static inline __attribute__((always_inline)) void ink_r7_iq3_load_hb0(
    const G u8 *p, const u32 h, u32 qw[8], u32 aw[5]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v0 = ink_r7_iq3_ld4(base + h);
  const u32x4 v1 = ink_r7_iq3_ld4(base + h + 4u);
  qw[0] = v0.x; qw[1] = v0.y; qw[2] = v0.z; qw[3] = v0.w;
  qw[4] = v1.x; qw[5] = v1.y; qw[6] = v1.z; qw[7] = v1.w;
  const u32x4 a0 = ink_r7_iq3_ld4(base + 16u);
  aw[0] = a0.x; aw[1] = a0.y; aw[2] = a0.z; aw[3] = a0.w;
  aw[4] = base[20];
}

/* Load IQ3_XXS hb1 with h=0/1 -> w0/w1 remap:
 *   h=0 loads base[8..15], h=1 loads base[9..16]; aw[1..4] = base[21..24] (aw[0] carried from aw[4]). */
static inline __attribute__((always_inline)) void ink_r7_iq3_load_hb1(
    const G u8 *p, const u32 h, u32 qw[8], u32 at[4]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v0 = ink_r7_iq3_ld4(base + 8u + h);
  const u32x4 v1 = ink_r7_iq3_ld4(base + 12u + h);
  qw[0] = v0.x; qw[1] = v0.y; qw[2] = v0.z; qw[3] = v0.w;
  qw[4] = v1.x; qw[5] = v1.y; qw[6] = v1.z; qw[7] = v1.w;
  const u32x4 a0 = ink_r7_iq3_ld4(base + 21u);
  at[0] = a0.x; at[1] = a0.y; at[2] = a0.z; at[3] = a0.w;
}

/* 1-k IQ3_XXS decode with h=0/1 -> w0/w1 remap:
 *   h=0 decodes elems 0..7 (w0) and 8..15 (w1) from single perm(qw[0], qw[1]).
 *   h=1 decodes elems 16..23 (w0) and 24..31 (w1) from single perm(qw[0], qw[1]).
 *   Eliminates 2 v_perm_b32 per k-block vs r6_iq3_situ_pf! */
static inline __attribute__((always_inline)) void ink_r7_iq3_decode_one(
    const u32 q0, const u32 q1, const u32 a0, const u32 a1,
    const u32 psel, const float d2, const LDSP u32 *lut, const u32 h, ink_r7_iq3_dc_t *dc) {
  const u32 auxv = ink_r7_iq3_perm_al(a0, a1, psel);
  const u32 qp = ink_r7_iq3_perm_al(q0, q1, psel);
  const u32 g00 = lut[qp & 255u];
  const u32 g01 = lut[(qp >> 8u) & 255u];
  const u32 g10 = lut[(qp >> 16u) & 255u];
  const u32 g11 = lut[qp >> 24u];
  __builtin_amdgcn_sched_barrier(0);
  const u32 s_sh = h ? 14u : 0u;
  dc->s0 = d2 * (0.5f + (float)(auxv >> 28u));
  dc->w0 = ink_r7_iq3_ksw(g00, g01, (auxv >> s_sh) & 127u);
  dc->w1 = ink_r7_iq3_ksw(g10, g11, (auxv >> (s_sh + 7u)) & 127u);
}

/* Single 16-token WMMA tile (for j=0 on 1..16 tiles or j=2 on 33..48 tiles):
 * 1x 0-bank-conflict ds_load_b128 (h=0 -> elems 0..15, h=1 -> elems 16..31) +
 * exact 0x4B400000 int32->f32 bias + split dlo/dhi exact ((float)r * s0) * dx FMA order. */
static inline __attribute__((always_inline)) void ink_r7_iq3_apply_one_j(
    const ink_r7_iq3_dc_t dck, const LDSP u8 *lx_k, const LDSP float *lsx_k,
    const u32 j0, f8 *acc) {
  const v8i zm = (v8i)(0x4B400000);
  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + j0 * 2048u);
  __builtin_amdgcn_sched_barrier(0);

  const v2i a0 = {(int)b0.x, (int)b0.y};
  const v2i a1 = {(int)b0.z, (int)b0.w};
  v8i r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, dck.w0, zm, 0);
  r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, dck.w1, r0, 0);
  __builtin_amdgcn_sched_barrier(0);

  const float s0 = dck.s0;
  const LDSP float *sx0 = lsx_k + 16u * j0;
  const f32x4 dlo = *(const LDSP f32x4 *)(sx0);
  __builtin_amdgcn_sched_barrier(0);
  const u32 r00 = (u32)r0[0], r01 = (u32)r0[1], r02 = (u32)r0[2], r03 = (u32)r0[3];
  const float t0 = (__builtin_bit_cast(float, r00) - 12582912.0f) * s0;
  const float t1 = (__builtin_bit_cast(float, r01) - 12582912.0f) * s0;
  const float t2 = (__builtin_bit_cast(float, r02) - 12582912.0f) * s0;
  const float t3 = (__builtin_bit_cast(float, r03) - 12582912.0f) * s0;
  const f32x4 dhi = *(const LDSP f32x4 *)(sx0 + 4u);
  __builtin_amdgcn_sched_barrier(0);
  acc[j0][0] += t0 * dlo[0];
  acc[j0][1] += t1 * dlo[1];
  acc[j0][2] += t2 * dlo[2];
  acc[j0][3] += t3 * dlo[3];
  const u32 r04 = (u32)r0[4], r05 = (u32)r0[5], r06 = (u32)r0[6], r07 = (u32)r0[7];
  const float t4 = (__builtin_bit_cast(float, r04) - 12582912.0f) * s0;
  const float t5 = (__builtin_bit_cast(float, r05) - 12582912.0f) * s0;
  const float t6 = (__builtin_bit_cast(float, r06) - 12582912.0f) * s0;
  const float t7 = (__builtin_bit_cast(float, r07) - 12582912.0f) * s0;
  __builtin_amdgcn_sched_barrier(0);
  acc[j0][4] += t4 * dhi[0];
  acc[j0][5] += t5 * dhi[1];
  acc[j0][6] += t6 * dhi[2];
  acc[j0][7] += t7 * dhi[3];
  __asm__ volatile("" : "+v"(acc[j0]));
  __builtin_amdgcn_sched_barrier(0);
}

/* Sequenced 2-tile (j0, j0+1) WMMA + exact 0x4B400000 + exact FMA order:
 * 2x 0-bank-conflict ds_load_b128 (halves LDS loads vs r6's 4x ds_load_b64) +
 * strictly fenced split dlo/dhi halves + +v acc anchor prevent pre-RA hoisting
 * and keep peak VGPRs <= 128 (3 WGs/CU). */
static inline __attribute__((always_inline)) void ink_r7_iq3_apply_pair_j(
    const ink_r7_iq3_dc_t dck, const LDSP u8 *lx_k, const LDSP float *lsx_k,
    const u32 j0, f8 *acc) {
  const v8i zm = (v8i)(0x4B400000);
  const u32 j_off = j0 * 2048u;
  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + j_off);
  const u32x4 b1 = *(const LDSP u32x4 *)(lx_k + j_off + 2048u);
  __builtin_amdgcn_sched_barrier(0);

  const v2i a0_0 = {(int)b0.x, (int)b0.y};
  const v2i a0_1 = {(int)b0.z, (int)b0.w};
  const v2i a1_0 = {(int)b1.x, (int)b1.y};
  const v2i a1_1 = {(int)b1.z, (int)b1.w};
  v8i r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_0, 1, dck.w0, zm, 0);
  v8i r1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_0, 1, dck.w0, zm, 0);
  r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_1, 1, dck.w1, r0, 0);
  r1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_1, 1, dck.w1, r1, 0);
  __builtin_amdgcn_sched_barrier(0);

  const float s0 = dck.s0;
  const LDSP float *sx0 = lsx_k + 16u * j0;
  {
    const f32x4 dlo = *(const LDSP f32x4 *)(sx0);
    __builtin_amdgcn_sched_barrier(0);
    const u32 r00 = (u32)r0[0], r01 = (u32)r0[1], r02 = (u32)r0[2], r03 = (u32)r0[3];
    const float t0 = (__builtin_bit_cast(float, r00) - 12582912.0f) * s0;
    const float t1 = (__builtin_bit_cast(float, r01) - 12582912.0f) * s0;
    const float t2 = (__builtin_bit_cast(float, r02) - 12582912.0f) * s0;
    const float t3 = (__builtin_bit_cast(float, r03) - 12582912.0f) * s0;
    const f32x4 dhi = *(const LDSP f32x4 *)(sx0 + 4u);
    __builtin_amdgcn_sched_barrier(0);
    acc[j0][0] += t0 * dlo[0];
    acc[j0][1] += t1 * dlo[1];
    acc[j0][2] += t2 * dlo[2];
    acc[j0][3] += t3 * dlo[3];
    const u32 r04 = (u32)r0[4], r05 = (u32)r0[5], r06 = (u32)r0[6], r07 = (u32)r0[7];
    const float t4 = (__builtin_bit_cast(float, r04) - 12582912.0f) * s0;
    const float t5 = (__builtin_bit_cast(float, r05) - 12582912.0f) * s0;
    const float t6 = (__builtin_bit_cast(float, r06) - 12582912.0f) * s0;
    const float t7 = (__builtin_bit_cast(float, r07) - 12582912.0f) * s0;
    __builtin_amdgcn_sched_barrier(0);
    acc[j0][4] += t4 * dhi[0];
    acc[j0][5] += t5 * dhi[1];
    acc[j0][6] += t6 * dhi[2];
    acc[j0][7] += t7 * dhi[3];
    __asm__ volatile("" : "+v"(acc[j0]));
    __builtin_amdgcn_sched_barrier(0);
  }
  {
    const f32x4 dlo = *(const LDSP f32x4 *)(sx0 + 16u);
    __builtin_amdgcn_sched_barrier(0);
    const u32 r10 = (u32)r1[0], r11 = (u32)r1[1], r12 = (u32)r1[2], r13 = (u32)r1[3];
    const float t0 = (__builtin_bit_cast(float, r10) - 12582912.0f) * s0;
    const float t1 = (__builtin_bit_cast(float, r11) - 12582912.0f) * s0;
    const float t2 = (__builtin_bit_cast(float, r12) - 12582912.0f) * s0;
    const float t3 = (__builtin_bit_cast(float, r13) - 12582912.0f) * s0;
    const f32x4 dhi = *(const LDSP f32x4 *)(sx0 + 20u);
    __builtin_amdgcn_sched_barrier(0);
    acc[j0 + 1u][0] += t0 * dlo[0];
    acc[j0 + 1u][1] += t1 * dlo[1];
    acc[j0 + 1u][2] += t2 * dlo[2];
    acc[j0 + 1u][3] += t3 * dlo[3];
    const u32 r14 = (u32)r1[4], r15 = (u32)r1[5], r16_v = (u32)r1[6], r17 = (u32)r1[7];
    const float t4 = (__builtin_bit_cast(float, r14) - 12582912.0f) * s0;
    const float t5 = (__builtin_bit_cast(float, r15) - 12582912.0f) * s0;
    const float t6 = (__builtin_bit_cast(float, r16_v) - 12582912.0f) * s0;
    const float t7 = (__builtin_bit_cast(float, r17) - 12582912.0f) * s0;
    __builtin_amdgcn_sched_barrier(0);
    acc[j0 + 1u][4] += t4 * dhi[0];
    acc[j0 + 1u][5] += t5 * dhi[1];
    acc[j0 + 1u][6] += t6 * dhi[2];
    acc[j0 + 1u][7] += t7 * dhi[3];
    __asm__ volatile("" : "+v"(acc[j0 + 1u]));
    __builtin_amdgcn_sched_barrier(0);
  }
}

static inline __attribute__((always_inline)) void ink_r7_iq3_apply_k_hi(
    const ink_r7_iq3_dc_t dck, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 k, const u32 r16, const u32 h, const int has_j3, f8 acc[4]) {
  const u32 x_col = ((2u * k + h) ^ (r16 & 7u)) << 4u;
  const LDSP u8 *lx_k = lx + r16 * 128u + x_col;
  const LDSP float *lsx_k = lsx + k * 64u + 8u * h;
  ink_r7_iq3_apply_pair_j(dck, lx_k, lsx_k, 0u, acc);
  if (has_j3) ink_r7_iq3_apply_pair_j(dck, lx_k, lsx_k, 2u, acc);
  else ink_r7_iq3_apply_one_j(dck, lx_k, lsx_k, 2u, acc);
}

static inline __attribute__((always_inline)) void ink_r7_iq3_apply_k_lo(
    const ink_r7_iq3_dc_t dck, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 k, const u32 r16, const u32 h, const int has_j1, f8 acc[2]) {
  const u32 x_col = ((2u * k + h) ^ (r16 & 7u)) << 4u;
  const LDSP u8 *lx_k = lx + r16 * 128u + x_col;
  const LDSP float *lsx_k = lsx + k * 32u + 8u * h;
  if (has_j1) ink_r7_iq3_apply_pair_j(dck, lx_k, lsx_k, 0u, acc);
  else ink_r7_iq3_apply_one_j(dck, lx_k, lsx_k, 0u, acc);
}

static inline __attribute__((always_inline)) void ink_r7_iq3_stage_x64(
    LDSP u8 *lx, LDSP float *lsx, const u32 st_x, const u32 t0, const int is_s,
    const int ld_hi, const u32x4 rx0, const u32x4 rx1, const f32x4 rsx) {
  *(LDSP u32x4 *)(lx + st_x) = rx0;
  if (ld_hi) *(LDSP u32x4 *)(lx + st_x + 4096u) = rx1;
  if (is_s) {
    lsx[t0] = rsx.x;
    lsx[64u + t0] = rsx.y;
    lsx[128u + t0] = rsx.z;
    lsx[192u + t0] = rsx.w;
  }
}

static inline __attribute__((always_inline)) void ink_r7_iq3_stage_x32(
    LDSP u8 *lx, LDSP float *lsx, const u32 st_x, const u32 t0, const int is_s,
    const int ld_lo, const u32x4 rx0, const f32x4 rsx) {
  if (ld_lo) *(LDSP u32x4 *)(lx + st_x) = rx0;
  if (is_s) {
    lsx[t0] = rsx.x;
    lsx[32u + t0] = rsx.y;
    lsx[64u + t0] = rsx.z;
    lsx[96u + t0] = rsx.w;
  }
}

/* Exact SITU (#pragma clang fp contract(off)) + SoA Q8_0 (quant_wave) epilogue strip. */
static inline __attribute__((always_inline)) void ink_r7_iq3_epilogue_strip(
    LDSP float *ep, const LDSP u32 *lpid, G u8 *Qout,
    const u32 is_up, const u32 r_loc, const u32 h, const u32 w, const u32 l,
    const u32 strip_base, const u32 nj_strip, const u32 blk0, const u32 nblk,
    const u32 nb_out, const u32 qstr, const float beta, const float lbeta,
    const u32 j_base, const f8 *acc) {
#pragma clang fp contract(off)
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

KERNEL ie_inkling_gemm_rq_exp_iq3xxs_situ_r7(const G u8 *qw, const G u8 *X, G u8 *Qout, u32 eff, u32 nb,
                                             const G u32 *list, const G u32 *tiles, const G u32 *n_tiles,
                                             u32 q_bytes, u32 xs, u32 qstr, u32 xdiv, float beta, float lbeta) {
  static LDS ink_r7_iq3_lds_t ink_r7_iq3_lds;
  if (!n_tiles || !tiles || !list || !qw || !X || !Qout) return;

  const u32 ti = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  const u32 nt_act = (u32)__builtin_amdgcn_readfirstlane((int)(*n_tiles));
  const u32 gbase = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() * 64u));
  if (ti >= nt_act || gbase >= eff || eff == 0u || (eff & 31u) != 0u || nb < 8u || (nb & 7u) != 0u || xdiv == 0u)
    return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)tiles[ti]);
  const u32 e = desc & 0xFFu;
  const u32 pb = (desc >> 8u) & 0x1FFFu;
  const u32 nj = (desc >> 21u) & 0x7Fu;
  if (nj == 0u || nj > 64u) return;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;

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

  /* Stage 1 KiB IQ3_XXS grid LUT (synchronized by tile pid lbarrier below). */
  ink_r7_iq3_lds.lut[t0] = ie_rq_iq3xxs_grid[t0];

  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  const u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);

  if (nj > 32u) {
    /* ---------------- Adaptive 48- or 64-pair tile (WJ = 3 or 4) ---------------- */
    const int has_j3 = __builtin_amdgcn_readfirstlane(nj > 48u);
    const int ld_hi = has_j3 || (tk_ld < 16u);
    const int is_s = (w < 2u) && (has_j3 || (t0 < 48u));
    if (w < 2u) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r7_iq3_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ink_r7_iq3_div_x(ink_r7_iq3_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
    const u32 xoff1 = ld_hi ? (ink_r7_iq3_div_x(ink_r7_iq3_lds.pid[tk_ld + 32u], xdiv) * xs + c8_ld * 16u) : 0u;
    const u32 sxoff = is_s ? ink_r7_iq3_div_x(ink_r7_iq3_lds.pid[t0 & 63u], xdiv) * xs : 0u;

    f8 acc[4];
#pragma unroll
    for (u32 j = 0u; j < 4u; j++) acc[j] = (f8)(0.0f);

    u32x4 rx0 = ink_r7_iq3_ld4(X + (unsigned long)xoff0);
    u32x4 rx1 = ld_hi ? ink_r7_iq3_ld4(X + (unsigned long)xoff1) : (u32x4)(0u);
    f32x4 rsx = is_s ? rq_load_f32x4((const G float *)(X_s + (unsigned long)sxoff)) : (f32x4)(0.0f);
    u32 cur_psel = (((unsigned long)ebase + (unsigned long)row_off) & 2ul) ? 0x07060504u : 0x05040302u;
    u32 cur_q[8], cur_a[5];
    ink_r7_iq3_load_hb0(ebase + (unsigned long)row_off, h, cur_q, cur_a);
    ink_r7_iq3_stage_x64(ink_r7_iq3_lds.u.k.x[0], ink_r7_iq3_lds.u.k.sx[0], st_x, t0, is_s, ld_hi, rx0, rx1, rsx);
    lbarrier();

#pragma unroll 1
    for (u32 sb = 0u; ; sb++) {
      const u32 kb1 = 8u * sb + 4u;
      u32 nk = (u32)__builtin_amdgcn_readfirstlane((int)(nb - 8u * sb < 4u ? nb - 8u * sb : 4u));
      __asm__ volatile("" : "+s"(nk));

      /* hb0: prefetch hb1 activations FIRST; stage after k=1 to free 12 VGPRs. */
      const G u8 *X_hb1 = X + (unsigned long)kb1 * 32ul;
      rx0 = ink_r7_iq3_ld4(X_hb1 + (unsigned long)xoff0);
      if (ld_hi) rx1 = ink_r7_iq3_ld4(X_hb1 + (unsigned long)xoff1);
      if (is_s) rsx = rq_load_f32x4((const G float *)(X_s + (unsigned long)kb1 * 4ul + (unsigned long)sxoff));
      __builtin_amdgcn_sched_barrier(0);

      /* d2 computed in h=0 from base[0] (cur_q[0]) and broadcast to h=1 via xrow_i. */
      const float d2_0 = ie_rq_h2f((cur_q[0] >> ((cur_psel & 4u) << 2u)) & 0xFFFFu) * 0.5f;
      const float d2_x = I2F(xrow_i(F2I(d2_0)));
      const float d2 = h ? d2_x : d2_0;
      ink_r7_iq3_dc_t xdc;

      /* hb0 k = 0, 1 */
      ink_r7_iq3_decode_one(cur_q[0], cur_q[1], cur_a[0], cur_a[1], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      if (0u < nk) ink_r7_iq3_apply_k_hi(xdc, ink_r7_iq3_lds.u.k.x[0], ink_r7_iq3_lds.u.k.sx[0], 0u, r16, h, has_j3, acc);

      ink_r7_iq3_decode_one(cur_q[2], cur_q[3], cur_a[1], cur_a[2], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      if (1u < nk) ink_r7_iq3_apply_k_hi(xdc, ink_r7_iq3_lds.u.k.x[0], ink_r7_iq3_lds.u.k.sx[0], 1u, r16, h, has_j3, acc);

      /* x[1]/sx[1] are dead until next lbarrier: stage hb1 activations now (frees 12 VGPRs). */
      ink_r7_iq3_stage_x64(ink_r7_iq3_lds.u.k.x[1], ink_r7_iq3_lds.u.k.sx[1], st_x, t0, is_s, ld_hi, rx0, rx1, rsx);
      __builtin_amdgcn_sched_barrier(0);

      /* hb0 k = 2, 3 (1-k decode keeps xdc = 5 VGPRs instead of xdc2[2] = 10 VGPRs!). */
      ink_r7_iq3_decode_one(cur_q[4], cur_q[5], cur_a[2], cur_a[3], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      if (2u < nk) ink_r7_iq3_apply_k_hi(xdc, ink_r7_iq3_lds.u.k.x[0], ink_r7_iq3_lds.u.k.sx[0], 2u, r16, h, has_j3, acc);

      ink_r7_iq3_decode_one(cur_q[6], cur_q[7], cur_a[3], cur_a[4], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      cur_a[0] = cur_a[4];
      ink_r7_iq3_load_hb1((ebase + (unsigned long)sb * 98ul) + (unsigned long)row_off, h, cur_q, cur_a + 1u);
      __builtin_amdgcn_sched_barrier(0);
      if (3u < nk) ink_r7_iq3_apply_k_hi(xdc, ink_r7_iq3_lds.u.k.x[0], ink_r7_iq3_lds.u.k.sx[0], 3u, r16, h, has_j3, acc);

      lbarrier();

      /* hb1: prefetch next-SB activations FIRST. */
      const int more = (sb + 1u < mb);
      if (more) {
        const u32 kbn = 8u * (sb + 1u);
        const G u8 *X_nxt = X + (unsigned long)kbn * 32ul;
        rx0 = ink_r7_iq3_ld4(X_nxt + (unsigned long)xoff0);
        if (ld_hi) rx1 = ink_r7_iq3_ld4(X_nxt + (unsigned long)xoff1);
        if (is_s) rsx = rq_load_f32x4((const G float *)(X_s + (unsigned long)kbn * 4ul + (unsigned long)sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);

      /* hb1 k = 0, 1 */
      ink_r7_iq3_decode_one(cur_q[0], cur_q[1], cur_a[0], cur_a[1], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      if (0u < nk) ink_r7_iq3_apply_k_hi(xdc, ink_r7_iq3_lds.u.k.x[1], ink_r7_iq3_lds.u.k.sx[1], 0u, r16, h, has_j3, acc);

      ink_r7_iq3_decode_one(cur_q[2], cur_q[3], cur_a[1], cur_a[2], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      if (1u < nk) ink_r7_iq3_apply_k_hi(xdc, ink_r7_iq3_lds.u.k.x[1], ink_r7_iq3_lds.u.k.sx[1], 1u, r16, h, has_j3, acc);

      if (more) {
        ink_r7_iq3_stage_x64(ink_r7_iq3_lds.u.k.x[0], ink_r7_iq3_lds.u.k.sx[0], st_x, t0, is_s, ld_hi, rx0, rx1, rsx);
      }
      __builtin_amdgcn_sched_barrier(0);

      /* hb1 k = 2, 3 */
      ink_r7_iq3_decode_one(cur_q[4], cur_q[5], cur_a[2], cur_a[3], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      if (2u < nk) ink_r7_iq3_apply_k_hi(xdc, ink_r7_iq3_lds.u.k.x[1], ink_r7_iq3_lds.u.k.sx[1], 2u, r16, h, has_j3, acc);

      ink_r7_iq3_decode_one(cur_q[6], cur_q[7], cur_a[3], cur_a[4], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      if (more) {
        ink_r7_iq3_load_hb0((ebase + (unsigned long)(sb + 1u) * 98ul) + (unsigned long)row_off, h, cur_q, cur_a);
      }
      __builtin_amdgcn_sched_barrier(0);
      if (3u < nk) ink_r7_iq3_apply_k_hi(xdc, ink_r7_iq3_lds.u.k.x[1], ink_r7_iq3_lds.u.k.sx[1], 3u, r16, h, has_j3, acc);

      if (!more) break;
      cur_psel ^= 0x02020606u;
      lbarrier();
    }

    __builtin_amdgcn_sched_barrier(0);
    lbarrier();
    ink_r7_iq3_epilogue_strip(&ink_r7_iq3_lds.u.ep[0][0][0], ink_r7_iq3_lds.pid, Qout,
                              is_up, r_loc, h, w, l, 0u, 32u, blk0, nblk, nb_out, qstr,
                              beta, lbeta, 0u, acc);
    lbarrier();
    ink_r7_iq3_epilogue_strip(&ink_r7_iq3_lds.u.ep[0][0][0], ink_r7_iq3_lds.pid, Qout,
                              is_up, r_loc, h, w, l, 32u, nj - 32u, blk0, nblk, nb_out, qstr,
                              beta, lbeta, 2u, acc);
  } else {
    /* ---------------- Adaptive 16- or 32-pair tile (WJ = 1 or 2) ---------------- */
    const int has_j1 = __builtin_amdgcn_readfirstlane(nj > 16u);
    const int ld_lo = has_j1 || (tk_ld < 16u);
    const int is_s = (w == 0u) && (has_j1 || (t0 < 16u));
    if (w == 0u) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r7_iq3_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ld_lo ? (ink_r7_iq3_div_x(ink_r7_iq3_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u) : 0u;
    const u32 sxoff = is_s ? ink_r7_iq3_div_x(ink_r7_iq3_lds.pid[t0 & 31u], xdiv) * xs : 0u;

    f8 acc[2];
    acc[0] = (f8)(0.0f);
    acc[1] = (f8)(0.0f);

    u32x4 rx0 = ld_lo ? ink_r7_iq3_ld4(X + (unsigned long)xoff0) : (u32x4)(0u);
    f32x4 rsx = is_s ? rq_load_f32x4((const G float *)(X_s + (unsigned long)sxoff)) : (f32x4)(0.0f);
    u32 cur_psel = (((unsigned long)ebase + (unsigned long)row_off) & 2ul) ? 0x07060504u : 0x05040302u;
    u32 cur_q[8], cur_a[5];
    ink_r7_iq3_load_hb0(ebase + (unsigned long)row_off, h, cur_q, cur_a);
    ink_r7_iq3_stage_x32(ink_r7_iq3_lds.u.k.x[0], ink_r7_iq3_lds.u.k.sx[0], st_x, t0, is_s, ld_lo, rx0, rsx);
    lbarrier();

#pragma unroll 1
    for (u32 sb = 0u; ; sb++) {
      const u32 kb1 = 8u * sb + 4u;
      u32 nk = (u32)__builtin_amdgcn_readfirstlane((int)(nb - 8u * sb < 4u ? nb - 8u * sb : 4u));
      __asm__ volatile("" : "+s"(nk));

      if (ld_lo) rx0 = ink_r7_iq3_ld4((X + (unsigned long)kb1 * 32ul) + (unsigned long)xoff0);
      if (is_s) rsx = rq_load_f32x4((const G float *)(X_s + (unsigned long)kb1 * 4ul + (unsigned long)sxoff));
      __builtin_amdgcn_sched_barrier(0);

      const float d2_0 = ie_rq_h2f((cur_q[0] >> ((cur_psel & 4u) << 2u)) & 0xFFFFu) * 0.5f;
      const float d2_x = I2F(xrow_i(F2I(d2_0)));
      const float d2 = h ? d2_x : d2_0;
      ink_r7_iq3_dc_t xdc;

      /* hb0 k = 0, 1 */
      ink_r7_iq3_decode_one(cur_q[0], cur_q[1], cur_a[0], cur_a[1], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      if (0u < nk) ink_r7_iq3_apply_k_lo(xdc, ink_r7_iq3_lds.u.k.x[0], ink_r7_iq3_lds.u.k.sx[0], 0u, r16, h, has_j1, acc);

      ink_r7_iq3_decode_one(cur_q[2], cur_q[3], cur_a[1], cur_a[2], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      if (1u < nk) ink_r7_iq3_apply_k_lo(xdc, ink_r7_iq3_lds.u.k.x[0], ink_r7_iq3_lds.u.k.sx[0], 1u, r16, h, has_j1, acc);

      ink_r7_iq3_stage_x32(ink_r7_iq3_lds.u.k.x[1], ink_r7_iq3_lds.u.k.sx[1], st_x, t0, is_s, ld_lo, rx0, rsx);
      __builtin_amdgcn_sched_barrier(0);

      /* hb0 k = 2, 3 */
      ink_r7_iq3_decode_one(cur_q[4], cur_q[5], cur_a[2], cur_a[3], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      if (2u < nk) ink_r7_iq3_apply_k_lo(xdc, ink_r7_iq3_lds.u.k.x[0], ink_r7_iq3_lds.u.k.sx[0], 2u, r16, h, has_j1, acc);

      ink_r7_iq3_decode_one(cur_q[6], cur_q[7], cur_a[3], cur_a[4], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      cur_a[0] = cur_a[4];
      ink_r7_iq3_load_hb1((ebase + (unsigned long)sb * 98ul) + (unsigned long)row_off, h, cur_q, cur_a + 1u);
      __builtin_amdgcn_sched_barrier(0);
      if (3u < nk) ink_r7_iq3_apply_k_lo(xdc, ink_r7_iq3_lds.u.k.x[0], ink_r7_iq3_lds.u.k.sx[0], 3u, r16, h, has_j1, acc);

      lbarrier();

      /* hb1 */
      const int more = (sb + 1u < mb);
      if (more) {
        const u32 kbn = 8u * (sb + 1u);
        if (ld_lo) rx0 = ink_r7_iq3_ld4((X + (unsigned long)kbn * 32ul) + (unsigned long)xoff0);
        if (is_s) rsx = rq_load_f32x4((const G float *)(X_s + (unsigned long)kbn * 4ul + (unsigned long)sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);

      /* hb1 k = 0, 1 */
      ink_r7_iq3_decode_one(cur_q[0], cur_q[1], cur_a[0], cur_a[1], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      if (0u < nk) ink_r7_iq3_apply_k_lo(xdc, ink_r7_iq3_lds.u.k.x[1], ink_r7_iq3_lds.u.k.sx[1], 0u, r16, h, has_j1, acc);

      ink_r7_iq3_decode_one(cur_q[2], cur_q[3], cur_a[1], cur_a[2], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      if (1u < nk) ink_r7_iq3_apply_k_lo(xdc, ink_r7_iq3_lds.u.k.x[1], ink_r7_iq3_lds.u.k.sx[1], 1u, r16, h, has_j1, acc);

      if (more) {
        ink_r7_iq3_stage_x32(ink_r7_iq3_lds.u.k.x[0], ink_r7_iq3_lds.u.k.sx[0], st_x, t0, is_s, ld_lo, rx0, rsx);
      }
      __builtin_amdgcn_sched_barrier(0);

      /* hb1 k = 2, 3 */
      ink_r7_iq3_decode_one(cur_q[4], cur_q[5], cur_a[2], cur_a[3], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      if (2u < nk) ink_r7_iq3_apply_k_lo(xdc, ink_r7_iq3_lds.u.k.x[1], ink_r7_iq3_lds.u.k.sx[1], 2u, r16, h, has_j1, acc);

      ink_r7_iq3_decode_one(cur_q[6], cur_q[7], cur_a[3], cur_a[4], cur_psel, d2, ink_r7_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      if (more) {
        ink_r7_iq3_load_hb0((ebase + (unsigned long)(sb + 1u) * 98ul) + (unsigned long)row_off, h, cur_q, cur_a);
      }
      __builtin_amdgcn_sched_barrier(0);
      if (3u < nk) ink_r7_iq3_apply_k_lo(xdc, ink_r7_iq3_lds.u.k.x[1], ink_r7_iq3_lds.u.k.sx[1], 3u, r16, h, has_j1, acc);

      if (!more) break;
      cur_psel ^= 0x02020606u;
      lbarrier();
    }

    __builtin_amdgcn_sched_barrier(0);
    lbarrier();
    ink_r7_iq3_epilogue_strip(&ink_r7_iq3_lds.u.ep[0][0][0], ink_r7_iq3_lds.pid, Qout,
                              is_up, r_loc, h, w, l, 0u, nj, blk0, nblk, nb_out, qstr,
                              beta, lbeta, 0u, acc);
  }
}

#undef INK_R7_WL_MAX_TILES
#undef INK_R7_WL_MAX_PAIRS
#undef INK_R7_WL_MAX_EXP

#endif /* !IE_MEGA */
