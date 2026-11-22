/* ============================================================================
 * Inkling-2 R6 Compact Active-Expert MoE Verify & Direct Per-Slot Crossover
 * Kernels (RX 9070 XT, gfx1201, wave32, NT=256).
 *
 * Included at EOF of kernels/ie_inkling_kernels.c (after ie_kernels.c).
 * All local symbols/LDS/macros prefixed ink_r6_cm_ / INK_R6_CM_; all macros
 * #undef'd at EOF. Clang 21 __builtin_bit_cast & ext_vector safe, 0 spills.
 *
 * Buffer size notes (verify crossover T = 2, 4, 6, 8, 12, 16; K = 8; E = 128):
 *   - packs   : 160 u32s (640 B). Worst-case pack count is
 *               max_packs = min(T * K, n_exp + (T * K) / 4u)
 *               (<= 16/32/48/64/96/128 for T = 2/4/6/8/12/16, K = 8, E = 128;
 *               160 u32s also covers n_exp + (T * K) / 4u = 160 at T = 16).
 *   - n_packs : 1 u32 (4 B). Written by ie_inkling_moe_sort_compact_r6.
 *   - Pack descriptor word packs[p] (u32):
 *       bits [ 7: 0] = e   (0..255, active expert ID)
 *       bits [19: 8] = lo  (0..4095, start index in list[])
 *       bits [23:20] = cnt (1..4, routed pair count in this pack; experts with
 *                           c > 4 split into ceil(c / 4) packs of <= 4 pairs)
 *
 * Exported KERNELs, exact signatures, grids (gx, gy, 1), guards, and parity:
 *
 * 1. KERNEL ie_inkling_moe_sort_compact_r6(
 *        const G u32 *route, u32 rst, u32 T, u32 K, u32 n_exp,
 *        G u32 *offs, G u32 *list, G u32 *packs, G u32 *n_packs)
 *    - Grid:   (1u, 1u, 1u)
 *    - Guard:  n_exp <= 256u && T * K <= 4096u (with 160-u32 packs buffer:
 *              min(T * K, n_exp + (T * K) / 4u) <= 160u, e.g. T <= 16, K <= 8,
 *              n_exp <= 128). rst is route token stride in bytes.
 *    - LDS:    17,480 B (1 WG total).
 *    - Parity: offs[0..n_exp] & list[0..T*K-1] bitwise-exact vs
 *              ie_inkling_moe_sort_r1; packs[0..*n_packs-1] exact 1..4-pair
 *              partition of active experts.
 *
 * 2. KERNEL ie_inkling_gexp_raw_iq3xxs_compact_r6(
 *        const G u8 *qw, const G u8 *X, G float *Y, u32 rows, u32 nb,
 *        const G u32 *list, const G u32 *packs, const G u32 *n_packs,
 *        u32 q_bytes, u32 xs, u32 ys, u32 xdiv, const G float *res, u32 rs)
 *    - Grid:   ((rows + 31u) / 32u, max_packs, 1u)
 *              where max_packs = min(T * K, n_exp + (T * K) / 4u)
 *              [e.g. (72, 16/32/48/64/96/128, 1) vs (72, 128, 1) in R4].
 *    - Guard:  T >= 2u && T <= 16u && type == IE_RQ_IQ3_XXS (18u) &&
 *              rows > 0u && nb >= 8u && (nb & 7u) == 0u &&
 *              n_exp <= 256u && T * K <= 4096u.
 *    - Res:    1 KiB LDS LUT (untouched on by >= *n_packs early return),
 *              <= 80 VGPR, 0 spills.
 *    - Parity: 100% bitwise-exact (0 bit diffs) vs
 *              ie_inkling_gexp_raw_iq3xxs_ver_r4 and ie_gexp_raw_t.
 *
 * 3. KERNEL ie_inkling_gexp_iq4nl_compact_r6(
 *        const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
 *        u32 rows, u32 nb, const G u32 *list, const G u32 *packs,
 *        const G u32 *n_packs, u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv,
 *        u32 kind, const G u8 *res, u32 rs)
 *    - Grid:   ((rows + 31u) / 32u, max_packs, 1u)
 *              where max_packs = min(T * K, n_exp + (T * K) / 4u).
 *    - Guard:  T >= 2u && T <= 16u && kind == MAT_IQ4NL (8u) &&
 *              rows > 0u && nb > 0u && n_exp <= 256u && T * K <= 4096u.
 *    - Res:    0 LDS, <= 80 VGPR, 0 spills.
 *    - Parity: 100% bitwise-exact (0 bit diffs) vs
 *              ie_inkling_gexp_iq4nl_ver_r4 and ie_gexp_t.
 *
 * 4. KERNEL ie_inkling_gexp_raw_iq3xxs_slot_r6(
 *        const G u8 *qw, const G u8 *X, G float *Y, u32 rows, u32 nb,
 *        const G u32 *route, u32 rst, u32 K, u32 q_bytes,
 *        u32 xs, u32 ys, u32 xdiv, const G float *res, u32 rs)
 *    - Grid:   ((rows + 31u) / 32u, T * K, 1u)  [NO sort dependency!]
 *    - Guard:  T >= 1u && K >= 1u && type == IE_RQ_IQ3_XXS (18u) &&
 *              rows > 0u && nb >= 8u && (nb & 7u) == 0u.
 *              rst is route token byte stride (g->bufs[o->c].stride).
 *    - Res:    1 KiB LDS LUT, <= 64 VGPR, 0 spills.
 *    - Parity: 100% bitwise-exact (0 bit diffs) vs
 *              ie_inkling_gexp_raw_iq3xxs_ver_r4 and ie_gexp_raw_t.
 *
 * 5. KERNEL ie_inkling_gexp_iq4nl_slot_r6(
 *        const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
 *        u32 rows, u32 nb, const G u32 *route, u32 rst, u32 K,
 *        u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv, u32 kind,
 *        const G u8 *res, u32 rs)
 *    - Grid:   ((rows + 31u) / 32u, T * K, 1u)  [NO sort dependency!]
 *    - Guard:  T >= 1u && K >= 1u && kind == MAT_IQ4NL (8u) &&
 *              rows > 0u && nb > 0u.
 *              rst is route token byte stride (g->bufs[o->c].stride).
 *    - Res:    0 LDS, <= 64 VGPR, 0 spills.
 *    - Parity: 100% bitwise-exact (0 bit diffs) vs
 *              ie_inkling_gexp_iq4nl_ver_r4 and ie_gexp_t.
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(f16) == 2u, "ink_r6_cm f16 must be 2 bytes");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r6_cm float/u32 must be 4 bytes");

#define INK_R6_CM_MAX_EXP   256u
#define INK_R6_CM_MAX_PAIRS 4096u

#define INK_R6_CM_ACC_ADD(a0_, a1_, a2_, a3_, j_, v_) \
  do {                                                \
    if ((j_) == 0u) (a0_) += (v_);                    \
    else if ((j_) == 1u) (a1_) += (v_);               \
    else if ((j_) == 2u) (a2_) += (v_);               \
    else (a3_) += (v_);                               \
  } while (0)

#define INK_R6_CM_ACC_FMA(a0_, a1_, a2_, a3_, j_, v_, d_)          \
  do {                                                             \
    if ((j_) == 0u) (a0_) = __builtin_fmaf((v_), (d_), (a0_));     \
    else if ((j_) == 1u) (a1_) = __builtin_fmaf((v_), (d_), (a1_)); \
    else if ((j_) == 2u) (a2_) = __builtin_fmaf((v_), (d_), (a2_)); \
    else (a3_) = __builtin_fmaf((v_), (d_), (a3_));                \
  } while (0)

/* ----------------------------------------------------------------------------
 * Shared Helpers (Clang 21 __builtin_bit_cast & ext_vector safe)
 * -------------------------------------------------------------------------- */

static inline __attribute__((always_inline)) u32 ink_r6_cm_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

/* 8-lane subgroup reduction inside each 16-lane DPP row (lanes 0, 8, 16, 24
 * hold the exact IEEE-754 sum of their 8-lane subgroup). */
static inline __attribute__((always_inline)) float ink_r6_cm_oct_tree_f(float v) {
#pragma clang fp contract(off)
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}

static inline __attribute__((always_inline)) u32 ink_r6_cm_wave_scan_inc(u32 x) {
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x111, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x112, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x114, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x118, 0xF, 0xF, true);
  const u32 lo16 = (u32)__builtin_amdgcn_readlane((int)x, 15);
  if (lane() >= 16u) x += lo16;
  return x;
}

/* Shared-weight IQ3_XXS 32-element dot (weights pre-decoded once for nj >= 2). */
static inline __attribute__((always_inline)) float ink_r6_cm_iq3_dot(
    const u32 wl0, const u32 wh0, const u32 wl1, const u32 wh1,
    const u32 wl2, const u32 wh2, const u32 wl3, const u32 wh3,
    const float dw, const u32x4 x0, const u32x4 x1, const float dav) {
#pragma clang fp contract(off)
  const u32 x00 = x0.x, x01 = x0.y, x02 = x0.z, x03 = x0.w;
  const u32 x10 = x1.x, x11 = x1.y, x12 = x1.z, x13 = x1.w;
  u32 idot = ie_dot4_ss(0u, wl0, x00);
  idot = ie_dot4_ss(idot, wh0, x01);
  idot = ie_dot4_ss(idot, wl1, x02);
  idot = ie_dot4_ss(idot, wh1, x03);
  idot = ie_dot4_ss(idot, wl2, x10);
  idot = ie_dot4_ss(idot, wh2, x11);
  idot = ie_dot4_ss(idot, wl3, x12);
  idot = ie_dot4_ss(idot, wh3, x13);
  return dav * (dw * (float)(int)idot);
}

/* Streaming 1-pair IQ3_XXS 32-element dot from pre-fetched LDS grid words
 * (avoids holding all 8 wl/wh live simultaneously -> low VGPRs). */
static inline __attribute__((always_inline)) float ink_r6_cm_iq3_stream_dot(
    const u32 d16, const u32 aux,
    const u32 g_lo0, const u32 g_hi0, const u32 g_lo1, const u32 g_hi1,
    const u32 g_lo2, const u32 g_hi2, const u32 g_lo3, const u32 g_hi3,
    const u32x4 x0, const u32x4 x1, const float dav) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d16);
  const float dw = d * (0.5f + (float)(aux >> 28u)) * 0.5f;
  const u32 x00 = x0.x, x01 = x0.y, x02 = x0.z, x03 = x0.w;
  const u32 x10 = x1.x, x11 = x1.y, x12 = x1.z, x13 = x1.w;

  const u32 s8_0 = rq_ksign7(aux & 127u);
  u32 idot = ie_dot4_ss(0u, ie_rq_ssign(g_lo0, ie_rq_mask4(s8_0 & 15u)), x00);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_hi0, ie_rq_mask4(s8_0 >> 4u)), x01);

  const u32 s8_1 = rq_ksign7((aux >> 7u) & 127u);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_lo1, ie_rq_mask4(s8_1 & 15u)), x02);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_hi1, ie_rq_mask4(s8_1 >> 4u)), x03);

  const u32 s8_2 = rq_ksign7((aux >> 14u) & 127u);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_lo2, ie_rq_mask4(s8_2 & 15u)), x10);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_hi2, ie_rq_mask4(s8_2 >> 4u)), x11);

  const u32 s8_3 = rq_ksign7((aux >> 21u) & 127u);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_lo3, ie_rq_mask4(s8_3 & 15u)), x12);
  idot = ie_dot4_ss(idot, ie_rq_ssign(g_hi3, ie_rq_mask4(s8_3 >> 4u)), x13);

  return dav * (dw * (float)(int)idot);
}

/* Shared-weight IQ4_NL 32-element dot (nibbles pre-mapped once for nj >= 2). */
static inline __attribute__((always_inline)) int ink_r6_cm_iq4_dot(
    const u32 wl0, const u32 wh0, const u32 wl1, const u32 wh1,
    const u32 wl2, const u32 wh2, const u32 wl3, const u32 wh3,
    const u32x4 a0, const u32x4 a1) {
  const u32 a00 = a0.x, a01 = a0.y, a02 = a0.z, a03 = a0.w;
  const u32 a10 = a1.x, a11 = a1.y, a12 = a1.z, a13 = a1.w;
  u32 d = 0u;
  d = ie_dot4_ss(ie_dot4_ss(d, wl0, a00), wh0, a10);
  d = ie_dot4_ss(ie_dot4_ss(d, wl1, a01), wh1, a11);
  d = ie_dot4_ss(ie_dot4_ss(d, wl2, a02), wh2, a12);
  d = ie_dot4_ss(ie_dot4_ss(d, wl3, a03), wh3, a13);
  return (int)d;
}

/* Streaming 1-pair IQ4_NL 32-element dot (exact r1_g04 arithmetic). */
static inline __attribute__((always_inline)) int ink_r6_cm_iq4_dot4(
    const u32x4 w4, const u32x4 a0, const u32x4 a1) {
  const u32 wx = w4.x, wy = w4.y, wz = w4.z, ww = w4.w;
  const u32 a00 = a0.x, a01 = a0.y, a02 = a0.z, a03 = a0.w;
  const u32 a10 = a1.x, a11 = a1.y, a12 = a1.z, a13 = a1.w;
  u32 d = 0u;
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(wx & 0x0F0F0F0Fu), a00), ie_iq4_map((wx >> 4u) & 0x0F0F0F0Fu), a10);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(wy & 0x0F0F0F0Fu), a01), ie_iq4_map((wy >> 4u) & 0x0F0F0F0Fu), a11);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(wz & 0x0F0F0F0Fu), a02), ie_iq4_map((wz >> 4u) & 0x0F0F0F0Fu), a12);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(ww & 0x0F0F0F0Fu), a03), ie_iq4_map((ww >> 4u) & 0x0F0F0F0Fu), a13);
  return (int)d;
}

/* ============================================================================
 * Kernel 1: ie_inkling_moe_sort_compact_r6
 *   Single-WG parallel LDS histogram + dual DPP prefix scan (pair offs + pack
 *   offs with ZERO extra barriers vs ie_inkling_moe_sort_r1) -> parallel pack
 *   descriptor emission + zero-atomic parallel scatter of list[].
 * ========================================================================== */

KERNEL ie_inkling_moe_sort_compact_r6(const G u32 *route, u32 rst, u32 T, u32 K, u32 n_exp,
                                      G u32 *offs, G u32 *list, G u32 *packs, G u32 *n_packs) {
  static LDS u32 ink_r6_cm_sort_cnt[INK_R6_CM_MAX_EXP];
  static LDS u32 ink_r6_cm_sort_er[INK_R6_CM_MAX_PAIRS];
  static LDS u32 ink_r6_cm_sort_wsum[9];
  static LDS u32 ink_r6_cm_sort_psum[9];

  const u32 t = tid(), w = t >> 5u, l = lane();
  const u32 np = T * K;

  if (n_exp == 0u) {
    if (t == 0u) {
      if (offs) offs[0] = 0u;
      if (n_packs) *n_packs = 0u;
    }
    return;
  }
  if (!offs || n_exp > INK_R6_CM_MAX_EXP || np > INK_R6_CM_MAX_PAIRS) return;
  if (np > 0u && (!route || !list)) return;

  if (t < n_exp) ink_r6_cm_sort_cnt[t] = 0u;
  lbarrier();

  /* Pass 1: parallel histogram + record (expert << 16 | in_expert_rank) per pair. */
  if (K == 8u) {
    const u32 kk = t & 7u;
    const u32 tk0 = t >> 3u;
    for (u32 p = t, tk = tk0; p < np; p += NT, tk += 32u) {
      const u32 e = ((const G u32 *)((const G u8 *)route + (unsigned long)tk * rst))[kk];
      const u32 r = __atomic_fetch_add(&ink_r6_cm_sort_cnt[e], 1u, __ATOMIC_RELAXED);
      ink_r6_cm_sort_er[p] = (e << 16u) | r;
    }
  } else if (K > 0u) {
    for (u32 p = t; p < np; p += NT) {
      const u32 e = ((const G u32 *)((const G u8 *)route + (unsigned long)(p / K) * rst))[p % K];
      const u32 r = __atomic_fetch_add(&ink_r6_cm_sort_cnt[e], 1u, __ATOMIC_RELAXED);
      ink_r6_cm_sort_er[p] = (e << 16u) | r;
    }
  }
  lbarrier();

  /* Dual parallel exclusive prefix sum over e = 0..n_exp-1 (n_exp <= 256):
   *   c  = pair count for expert e -> offs[e]
   *   pc = ceil(c / 4) pack count  -> pack start index for expert e */
  const u32 c = t < n_exp ? ink_r6_cm_sort_cnt[t] : 0u;
  const u32 pc = (c + 3u) >> 2u;

  const u32 c_inc = ink_r6_cm_wave_scan_inc(c);
  const u32 pc_inc = ink_r6_cm_wave_scan_inc(pc);
  if (l == 31u) {
    ink_r6_cm_sort_wsum[w] = c_inc;
    ink_r6_cm_sort_psum[w] = pc_inc;
  }
  lbarrier();

  if (w == 0u) {
    const u32 ws = l < 8u ? ink_r6_cm_sort_wsum[l] : 0u;
    const u32 ps = l < 8u ? ink_r6_cm_sort_psum[l] : 0u;
    const u32 winc = ink_r6_cm_wave_scan_inc(ws);
    const u32 pinc = ink_r6_cm_wave_scan_inc(ps);
    if (l < 8u) {
      ink_r6_cm_sort_wsum[l] = winc - ws;
      ink_r6_cm_sort_psum[l] = pinc - ps;
    }
    if (l == 7u) {
      ink_r6_cm_sort_wsum[8] = winc;
      ink_r6_cm_sort_psum[8] = pinc;
    }
  }
  lbarrier();

  if (t < n_exp) {
    const u32 off = ink_r6_cm_sort_wsum[w] + (c_inc - c);
    const u32 poff = ink_r6_cm_sort_psum[w] + (pc_inc - pc);
    ink_r6_cm_sort_cnt[t] = off;
    offs[t] = off;

    /* Emit compact 1..4-route pack descriptors for active expert t. */
    if (packs && c > 0u) {
      u32 rem = c;
      u32 lo_p = off;
      u32 pi = poff;
      for (u32 k = 0u; k < pc; k++) {
        const u32 cnt_p = rem > 4u ? 4u : rem;
        packs[pi] = (t & 0xFFu) | ((lo_p & 0xFFFu) << 8u) | ((cnt_p & 0xFu) << 20u);
        lo_p += cnt_p;
        rem -= cnt_p;
        pi++;
      }
    }
  }
  if (t == 0u) {
    offs[n_exp] = ink_r6_cm_sort_wsum[8];
    if (n_packs) *n_packs = ink_r6_cm_sort_psum[8];
  }
  lbarrier();

  /* Pass 2: zero-atomic parallel scatter of all T*K pairs into list[]. */
  for (u32 p = t; p < np; p += NT) {
    const u32 er = ink_r6_cm_sort_er[p];
    const u32 e = er >> 16u;
    const u32 r = er & 0xFFFFu;
    list[ink_r6_cm_sort_cnt[e] + r] = p;
  }
}

/* ============================================================================
 * Kernel 2: ie_inkling_gexp_raw_iq3xxs_compact_r6
 *   Compact active-pack IQ3_XXS verify (4 rows/wave, 8 lanes/row, 32 rows/WG).
 *   - Immediate early return on by >= *n_packs BEFORE touching LDS or lbarrier.
 *   - Wave-uniform nj == 1 fast path (0 nj branches, streaming dot4, ~48 VGPR)
 *     + nj == 2..4 shared-weight path (unconditional pair 0+1, <= 76 VGPR).
 *   - X loads issued BEFORE unconditional next-SB DRAM weight loads (bp_nxt)
 *     so SIInsertWaitcnts keeps all 4 next-SB DRAM weight loads in flight.
 *   - 100% bitwise-exact vs ie_inkling_gexp_raw_iq3xxs_ver_r4 & ie_gexp_raw_t.
 * ========================================================================== */

KERNEL ie_inkling_gexp_raw_iq3xxs_compact_r6(const G u8 *qw, const G u8 *X, G float *Y, u32 rows, u32 nb,
                                             const G u32 *list, const G u32 *packs, const G u32 *n_packs,
                                             u32 q_bytes, u32 xs, u32 ys, u32 xdiv, const G float *res, u32 rs) {
#pragma clang fp contract(off)
  static LDS u32 ink_r6_cm_iq3_lut[256] __attribute__((aligned(16)));
  if (rows == 0u || nb < 8u || (nb & 7u) != 0u || !qw || !X || !Y || !list || !packs || !n_packs) return;

  const u32 by = tokid();
  const u32 npk = (u32)__builtin_amdgcn_readfirstlane((int)(*n_packs));
  const u32 rbase = wgid() * 32u;
  if (by >= npk || rbase >= rows) return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)packs[by]);
  const u32 e = desc & 0xFFu;
  const u32 lo = (desc >> 8u) & 0xFFFu;
  const u32 nj = (desc >> 20u) & 0xFu;
  if (nj == 0u) return;

  const u32 t0 = tid();
  ink_r6_cm_iq3_lut[t0] = ie_rq_iq3xxs_grid[t0];
  lbarrier();

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 r = rbase + wv * 4u + k;
  const u32 rr = r < rows ? r : rows - 1u;
  const u32 mb = nb >> 3u;
  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * q_bytes);
  const G u8 *bp = ebase + (unsigned long)rr * ((unsigned long)mb * 98ul);
  const u32 qo = 2u + 8u * s;
  const u32 auxo = 66u + 4u * s;
  const unsigned long x_scale_off = 32ul * (unsigned long)nb;

  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
  const G u8 *xp0 = rq_ptr_u(X + (unsigned long)ink_r6_cm_div_x(p0, xdiv) * xs);
  const G float *dap0 = (const G float *)(xp0 + x_scale_off);

  u32 d_c = ie_rq_a16(bp, 0u);
  u32 aux_c = ie_rq_a32(bp, auxo);
  u32 q01_c = ie_rq_a32(bp, qo);
  u32 q23_c = ie_rq_a32(bp, qo + 4u);
  __asm__ volatile("" : "+v"(d_c), "+v"(aux_c), "+v"(q01_c), "+v"(q23_c));

  if (nj == 1u) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

#pragma unroll 1
    for (u32 sb = 0u; sb < mb; sb++) {
      const u32 b = 8u * sb + s;
      const u32 j_acc = sb & 3u;
      const G u8 *bp_nxt = (sb + 1u < mb) ? (bp + 98u) : bp;

      /* L1/L2 X loads FIRST, unconditional next-SB DRAM weights SECOND, LDS LUT THIRD. */
      const u32x4 x0_a = *(const G u32x4 *)(xp0 + 32u * b);
      const u32x4 x1_a = *(const G u32x4 *)(xp0 + 32u * b + 16u);
      const float dav_a = dap0[b];

      const u32 d_n = ie_rq_a16(bp_nxt, 0u);
      const u32 aux_n = ie_rq_a32(bp_nxt, auxo);
      const u32 q01_n = ie_rq_a32(bp_nxt, qo);
      const u32 q23_n = ie_rq_a32(bp_nxt, qo + 4u);

      const u32 qp0 = q01_c & 0xFFFFu;
      const u32 qp1 = q01_c >> 16u;
      const u32 qp2 = q23_c & 0xFFFFu;
      const u32 qp3 = q23_c >> 16u;
      const u32 g_lo0 = ink_r6_cm_iq3_lut[qp0 & 255u];
      const u32 g_hi0 = ink_r6_cm_iq3_lut[qp0 >> 8u];
      const u32 g_lo1 = ink_r6_cm_iq3_lut[qp1 & 255u];
      const u32 g_hi1 = ink_r6_cm_iq3_lut[qp1 >> 8u];
      const u32 g_lo2 = ink_r6_cm_iq3_lut[qp2 & 255u];
      const u32 g_hi2 = ink_r6_cm_iq3_lut[qp2 >> 8u];
      const u32 g_lo3 = ink_r6_cm_iq3_lut[qp3 & 255u];
      const u32 g_hi3 = ink_r6_cm_iq3_lut[qp3 >> 8u];
      __builtin_amdgcn_sched_barrier(0);

      const float v0 = ink_r6_cm_iq3_stream_dot(
          d_c, aux_c, g_lo0, g_hi0, g_lo1, g_hi1, g_lo2, g_hi2, g_lo3, g_hi3,
          x0_a, x1_a, dav_a);
      INK_R6_CM_ACC_ADD(a0, a1, a2, a3, j_acc, v0);
      __builtin_amdgcn_sched_barrier(0);
      bp = bp_nxt;
      d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
    }

    const float acc0 = ink_r6_cm_oct_tree_f((a0 + a2) + (a1 + a3));
    if (s == 0u && r < rows) {
      G float *yp0 = (G float *)((G u8 *)Y + (unsigned long)p0 * ys);
      const G float *rp0 = res ? (const G float *)((const G u8 *)res + (unsigned long)p0 * rs) : 0;
      yp0[r] = rp0 ? rp0[r] + acc0 : acc0;
    }
    return;
  }

  /* Multi-pair path (nj in 2..4): decode weights ONCE across all nj pairs. */
  const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 1u]);
  const u32 p2 = nj > 2u ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 2u]) : p0;
  const u32 p3 = nj > 3u ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 3u]) : p0;

  const G u8 *xp1 = rq_ptr_u(X + (unsigned long)ink_r6_cm_div_x(p1, xdiv) * xs);
  const G u8 *xp2 = nj > 2u ? rq_ptr_u(X + (unsigned long)ink_r6_cm_div_x(p2, xdiv) * xs) : xp0;
  const G u8 *xp3 = nj > 3u ? rq_ptr_u(X + (unsigned long)ink_r6_cm_div_x(p3, xdiv) * xs) : xp0;

  const G float *dap1 = (const G float *)(xp1 + x_scale_off);
  const G float *dap2 = (const G float *)(xp2 + x_scale_off);
  const G float *dap3 = (const G float *)(xp3 + x_scale_off);

  float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
  float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
  float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
  float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;

#pragma unroll 1
  for (u32 sb = 0u; sb < mb; sb++) {
    const u32 b = 8u * sb + s;
    const u32 j_acc = sb & 3u;
    const G u8 *bp_nxt = (sb + 1u < mb) ? (bp + 98u) : bp;

    /* Issue pair 0 & pair 1 X loads FIRST, then unconditional next-SB DRAM weights,
     * then 8 LDS LUT reads -> s_wait_loadcnt keeps DRAM weights in flight through v0+v1. */
    u32x4 x0_a = *(const G u32x4 *)(xp0 + 32u * b);
    u32x4 x1_a = *(const G u32x4 *)(xp0 + 32u * b + 16u);
    float dav_a = dap0[b];

    u32x4 x0_b = *(const G u32x4 *)(xp1 + 32u * b);
    u32x4 x1_b = *(const G u32x4 *)(xp1 + 32u * b + 16u);
    float dav_b = dap1[b];

    const u32 d_n = ie_rq_a16(bp_nxt, 0u);
    const u32 aux_n = ie_rq_a32(bp_nxt, auxo);
    const u32 q01_n = ie_rq_a32(bp_nxt, qo);
    const u32 q23_n = ie_rq_a32(bp_nxt, qo + 4u);

    const u32 qp0 = q01_c & 0xFFFFu;
    const u32 qp1 = q01_c >> 16u;
    const u32 qp2 = q23_c & 0xFFFFu;
    const u32 qp3 = q23_c >> 16u;
    const u32 g_lo0 = ink_r6_cm_iq3_lut[qp0 & 255u];
    const u32 g_hi0 = ink_r6_cm_iq3_lut[qp0 >> 8u];
    const u32 g_lo1 = ink_r6_cm_iq3_lut[qp1 & 255u];
    const u32 g_hi1 = ink_r6_cm_iq3_lut[qp1 >> 8u];
    const u32 g_lo2 = ink_r6_cm_iq3_lut[qp2 & 255u];
    const u32 g_hi2 = ink_r6_cm_iq3_lut[qp2 >> 8u];
    const u32 g_lo3 = ink_r6_cm_iq3_lut[qp3 & 255u];
    const u32 g_hi3 = ink_r6_cm_iq3_lut[qp3 >> 8u];
    __builtin_amdgcn_sched_barrier(0);

    /* Decode IQ3_XXS scale & signed int8 weights ONCE (1-for-1 register reuse). */
    const float d = ie_rq_h2f(d_c);
    const float dw = d * (0.5f + (float)(aux_c >> 28u)) * 0.5f;
    const u32 s8_0 = rq_ksign7(aux_c & 127u);
    const u32 wl0 = ie_rq_ssign(g_lo0, ie_rq_mask4(s8_0 & 15u));
    const u32 wh0 = ie_rq_ssign(g_hi0, ie_rq_mask4(s8_0 >> 4u));
    const u32 s8_1 = rq_ksign7((aux_c >> 7u) & 127u);
    const u32 wl1 = ie_rq_ssign(g_lo1, ie_rq_mask4(s8_1 & 15u));
    const u32 wh1 = ie_rq_ssign(g_hi1, ie_rq_mask4(s8_1 >> 4u));
    const u32 s8_2 = rq_ksign7((aux_c >> 14u) & 127u);
    const u32 wl2 = ie_rq_ssign(g_lo2, ie_rq_mask4(s8_2 & 15u));
    const u32 wh2 = ie_rq_ssign(g_hi2, ie_rq_mask4(s8_2 >> 4u));
    const u32 s8_3 = rq_ksign7((aux_c >> 21u) & 127u);
    const u32 wl3 = ie_rq_ssign(g_lo3, ie_rq_mask4(s8_3 & 15u));
    const u32 wh3 = ie_rq_ssign(g_hi3, ie_rq_mask4(s8_3 >> 4u));

    const float v0 = ink_r6_cm_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_a, x1_a, dav_a);
    INK_R6_CM_ACC_ADD(a0_0, a0_1, a0_2, a0_3, j_acc, v0);

    if (nj > 2u) {
      x0_a = *(const G u32x4 *)(xp2 + 32u * b);
      x1_a = *(const G u32x4 *)(xp2 + 32u * b + 16u);
      dav_a = dap2[b];
    }
    __builtin_amdgcn_sched_barrier(0);

    const float v1 = ink_r6_cm_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_b, x1_b, dav_b);
    INK_R6_CM_ACC_ADD(a1_0, a1_1, a1_2, a1_3, j_acc, v1);

    if (nj > 2u) {
      if (nj > 3u) {
        x0_b = *(const G u32x4 *)(xp3 + 32u * b);
        x1_b = *(const G u32x4 *)(xp3 + 32u * b + 16u);
        dav_b = dap3[b];
      }
      __builtin_amdgcn_sched_barrier(0);
      const float v2 = ink_r6_cm_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_a, x1_a, dav_a);
      INK_R6_CM_ACC_ADD(a2_0, a2_1, a2_2, a2_3, j_acc, v2);

      if (nj > 3u) {
        __builtin_amdgcn_sched_barrier(0);
        const float v3 = ink_r6_cm_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_b, x1_b, dav_b);
        INK_R6_CM_ACC_ADD(a3_0, a3_1, a3_2, a3_3, j_acc, v3);
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    bp = bp_nxt;
    d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
  }

  /* Exact wave_tree_f reduction + store for each active pair in this pack. */
  const float acc0 = ink_r6_cm_oct_tree_f((a0_0 + a0_2) + (a0_1 + a0_3));
  if (s == 0u && r < rows) {
    G float *yp0 = (G float *)((G u8 *)Y + (unsigned long)p0 * ys);
    const G float *rp0 = res ? (const G float *)((const G u8 *)res + (unsigned long)p0 * rs) : 0;
    yp0[r] = rp0 ? rp0[r] + acc0 : acc0;
  }
  const float acc1 = ink_r6_cm_oct_tree_f((a1_0 + a1_2) + (a1_1 + a1_3));
  if (s == 0u && r < rows) {
    G float *yp1 = (G float *)((G u8 *)Y + (unsigned long)p1 * ys);
    const G float *rp1 = res ? (const G float *)((const G u8 *)res + (unsigned long)p1 * rs) : 0;
    yp1[r] = rp1 ? rp1[r] + acc1 : acc1;
  }
  if (nj > 2u) {
    const float acc2 = ink_r6_cm_oct_tree_f((a2_0 + a2_2) + (a2_1 + a2_3));
    if (s == 0u && r < rows) {
      G float *yp2 = (G float *)((G u8 *)Y + (unsigned long)p2 * ys);
      const G float *rp2 = res ? (const G float *)((const G u8 *)res + (unsigned long)p2 * rs) : 0;
      yp2[r] = rp2 ? rp2[r] + acc2 : acc2;
    }
  }
  if (nj > 3u) {
    const float acc3 = ink_r6_cm_oct_tree_f((a3_0 + a3_2) + (a3_1 + a3_3));
    if (s == 0u && r < rows) {
      G float *yp3 = (G float *)((G u8 *)Y + (unsigned long)p3 * ys);
      const G float *rp3 = res ? (const G float *)((const G u8 *)res + (unsigned long)p3 * rs) : 0;
      yp3[r] = rp3 ? rp3[r] + acc3 : acc3;
    }
  }
}

/* ============================================================================
 * Kernel 3: ie_inkling_gexp_iq4nl_compact_r6
 *   Compact active-pack IQ4_NL verify (4 rows/wave, 8 lanes/row, 32 rows/WG).
 *   - Immediate early return on by >= *n_packs; 0 LDS, <= 76 VGPR, 0 spills.
 *   - Wave-uniform nj == 1 fast path uses r1_g04's 2-block-overlapped streaming
 *     body; nj == 2..4 shares decoded IQ4_NL nibbles across all nj pairs with
 *     unconditional lookahead (bn_safe) keeping next-step DRAM loads in flight.
 *   - 100% bitwise-exact vs ie_inkling_gexp_iq4nl_ver_r4 & ie_gexp_t.
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq4nl_compact_r6(const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
                                        u32 rows, u32 nb, const G u32 *list, const G u32 *packs,
                                        const G u32 *n_packs, u32 qwn, u32 qsn, u32 xs, u32 ys,
                                        u32 xdiv, u32 kind, const G u8 *res, u32 rs) {
#pragma clang fp contract(off)
  (void)kind;
  if (rows == 0u || nb == 0u || !qw0 || !qs0 || !X || !Y || !list || !packs || !n_packs) return;

  const u32 by = tokid();
  const u32 npk = (u32)__builtin_amdgcn_readfirstlane((int)(*n_packs));
  const u32 rbase = wgid() * 32u;
  if (by >= npk || rbase >= rows) return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)packs[by]);
  const u32 e = desc & 0xFFu;
  const u32 lo = (desc >> 8u) & 0xFFFu;
  const u32 nj = (desc >> 20u) & 0xFu;
  if (nj == 0u) return;

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 r_wave = rbase + wv * 4u;
  if (r_wave >= rows) return;

  const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
  const u32 r = r_wave + sub;
  const u32 rr = r < rows ? r : rows - 1u;

  const G u8 *w_base = rq_ptr_u((const G u8 *)(qw0 + (unsigned long)e * qwn));
  const G u8 *q_base = rq_ptr_u((const G u8 *)(qs0 + (unsigned long)e * qsn));
  const G u8 *rw = w_base + (unsigned long)rr * ((unsigned long)nb * 16ul);
  const G f16 *rs_p = (const G f16 *)q_base + (unsigned long)rr * nb;
  const unsigned long x_scale_off = 32ul * (unsigned long)nb;
  const int full_wave = (nb > 16u);

  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
  const G u8 *xp0 = rq_ptr_u(X + (unsigned long)ink_r6_cm_div_x(p0, xdiv) * xs);
  const G u32 *aw0 = (const G u32 *)xp0;
  const G float *da0 = (const G float *)(xp0 + x_scale_off);

  if (nj == 1u) {
    if (r < rows) {
      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
      u32 t = 0u;
      for (; t + 32u <= nb; t += 32u) {
        const u32 b0 = t + sl, b1 = t + 8u + sl;
        const u32x4 w0 = *(const G u32x4 *)(rw + b0 * 16u);
        const float sc0 = (float)rs_p[b0];
        const u32x4 x00 = *(const G u32x4 *)(aw0 + b0 * 8u);
        const u32x4 x01 = *(const G u32x4 *)(aw0 + b0 * 8u + 4u);
        const float d0 = da0[b0];
        const u32x4 w1 = *(const G u32x4 *)(rw + b1 * 16u);
        const float sc1 = (float)rs_p[b1];
        const u32x4 x10 = *(const G u32x4 *)(aw0 + b1 * 8u);
        const u32x4 x11 = *(const G u32x4 *)(aw0 + b1 * 8u + 4u);
        const float d1 = da0[b1];
        a0 = __builtin_fmaf(sc0 * (float)ink_r6_cm_iq4_dot4(w0, x00, x01), d0, a0);
        a1 = __builtin_fmaf(sc1 * (float)ink_r6_cm_iq4_dot4(w1, x10, x11), d1, a1);

        const u32 b2 = t + 16u + sl, b3 = t + 24u + sl;
        const u32x4 w2 = *(const G u32x4 *)(rw + b2 * 16u);
        const float sc2 = (float)rs_p[b2];
        const u32x4 x20 = *(const G u32x4 *)(aw0 + b2 * 8u);
        const u32x4 x21 = *(const G u32x4 *)(aw0 + b2 * 8u + 4u);
        const float d2 = da0[b2];
        const u32x4 w3 = *(const G u32x4 *)(rw + b3 * 16u);
        const float sc3 = (float)rs_p[b3];
        const u32x4 x30 = *(const G u32x4 *)(aw0 + b3 * 8u);
        const u32x4 x31 = *(const G u32x4 *)(aw0 + b3 * 8u + 4u);
        const float d3 = da0[b3];
        a2 = __builtin_fmaf(sc2 * (float)ink_r6_cm_iq4_dot4(w2, x20, x21), d2, a2);
        a3 = __builtin_fmaf(sc3 * (float)ink_r6_cm_iq4_dot4(w3, x30, x31), d3, a3);
      }
      if (t + sl < nb) {
        const u32 b = t + sl;
        const u32x4 w0 = *(const G u32x4 *)(rw + b * 16u);
        const float sc0 = (float)rs_p[b];
        const u32x4 x00 = *(const G u32x4 *)(aw0 + b * 8u);
        const u32x4 x01 = *(const G u32x4 *)(aw0 + b * 8u + 4u);
        const float d0 = da0[b];
        a0 = __builtin_fmaf(sc0 * (float)ink_r6_cm_iq4_dot4(w0, x00, x01), d0, a0);
      }
      if (t + 8u + sl < nb) {
        const u32 b = t + 8u + sl;
        const u32x4 w1 = *(const G u32x4 *)(rw + b * 16u);
        const float sc1 = (float)rs_p[b];
        const u32x4 x10 = *(const G u32x4 *)(aw0 + b * 8u);
        const u32x4 x11 = *(const G u32x4 *)(aw0 + b * 8u + 4u);
        const float d1 = da0[b];
        a1 = __builtin_fmaf(sc1 * (float)ink_r6_cm_iq4_dot4(w1, x10, x11), d1, a1);
      }
      if (t + 16u + sl < nb) {
        const u32 b = t + 16u + sl;
        const u32x4 w2 = *(const G u32x4 *)(rw + b * 16u);
        const float sc2 = (float)rs_p[b];
        const u32x4 x20 = *(const G u32x4 *)(aw0 + b * 8u);
        const u32x4 x21 = *(const G u32x4 *)(aw0 + b * 8u + 4u);
        const float d2 = da0[b];
        a2 = __builtin_fmaf(sc2 * (float)ink_r6_cm_iq4_dot4(w2, x20, x21), d2, a2);
      }
      if (t + 24u + sl < nb) {
        const u32 b = t + 24u + sl;
        const u32x4 w3 = *(const G u32x4 *)(rw + b * 16u);
        const float sc3 = (float)rs_p[b];
        const u32x4 x30 = *(const G u32x4 *)(aw0 + b * 8u);
        const u32x4 x31 = *(const G u32x4 *)(aw0 + b * 8u + 4u);
        const float d3 = da0[b];
        a3 = __builtin_fmaf(sc3 * (float)ink_r6_cm_iq4_dot4(w3, x30, x31), d3, a3);
      }
      const float acc0 = ink_r6_cm_oct_tree_f(full_wave ? ((a0 + a2) + (a1 + a3)) : (a0 + a1));
      if (sl == 0u) {
        G float *yp0 = (G float *)(Y + (unsigned long)p0 * ys);
        const G float *rp0 = res ? (const G float *)(res + (unsigned long)p0 * rs) : 0;
        yp0[r] = epilogue(acc0, r, 0, rp0);
      }
    }
    return;
  }

  /* Multi-pair path (nj in 2..4): decode IQ4_NL weights ONCE across all nj pairs. */
  const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 1u]);
  const u32 p2 = nj > 2u ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 2u]) : p0;
  const u32 p3 = nj > 3u ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 3u]) : p0;

  const G u8 *xp1 = rq_ptr_u(X + (unsigned long)ink_r6_cm_div_x(p1, xdiv) * xs);
  const G u8 *xp2 = nj > 2u ? rq_ptr_u(X + (unsigned long)ink_r6_cm_div_x(p2, xdiv) * xs) : xp0;
  const G u8 *xp3 = nj > 3u ? rq_ptr_u(X + (unsigned long)ink_r6_cm_div_x(p3, xdiv) * xs) : xp0;

  const G u32 *aw1 = (const G u32 *)xp1;
  const G u32 *aw2 = (const G u32 *)xp2;
  const G u32 *aw3 = (const G u32 *)xp3;
  const G float *da1 = (const G float *)(xp1 + x_scale_off);
  const G float *da2 = (const G float *)(xp2 + x_scale_off);
  const G float *da3 = (const G float *)(xp3 + x_scale_off);

  float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
  float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
  float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
  float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;

  const u32 nstep = (nb + 7u) >> 3u;
  const u32 nb_last = nb - 1u;

  /* Prologue: prefetch step 0 weights (blocks 0..7). */
  u32x4 w_c = (u32x4)(0u);
  f16 sc_c = (f16)0.0f;
  if (sl < nb) {
    w_c = *(const G u32x4 *)(rw + sl * 16u);
    sc_c = rs_p[sl];
  }
  __builtin_amdgcn_sched_barrier(0);

#pragma unroll 1
  for (u32 step = 0u; step < nstep; step++) {
    const u32 b_c = step * 8u + sl;
    const u32 bn = b_c + 8u;
    const u32 j_acc = step & 3u;

    u32x4 w_n = (u32x4)(0u);
    f16 sc_n = (f16)0.0f;

    if (b_c < nb) {
      /* Issue pair 0 & pair 1 X loads FIRST, then unconditional next-step DRAM
       * weight prefetch (clamped to nb_last) so s_wait_loadcnt keeps w_n/sc_n in flight. */
      u32x4 x0_a = *(const G u32x4 *)(aw0 + b_c * 8u);
      u32x4 x1_a = *(const G u32x4 *)(aw0 + b_c * 8u + 4u);
      float d_a = da0[b_c];

      u32x4 x0_b = *(const G u32x4 *)(aw1 + b_c * 8u);
      u32x4 x1_b = *(const G u32x4 *)(aw1 + b_c * 8u + 4u);
      float d_b = da1[b_c];

      const u32 bn_safe = bn < nb ? bn : nb_last;
      w_n = *(const G u32x4 *)(rw + bn_safe * 16u);
      sc_n = rs_p[bn_safe];
      __builtin_amdgcn_sched_barrier(0);

      const float sc_f = (float)sc_c;
      const u32 wcx = w_c.x, wcy = w_c.y, wcz = w_c.z, wcw = w_c.w;
      const u32 wl0 = ie_iq4_map(wcx & 0x0F0F0F0Fu);
      const u32 wh0 = ie_iq4_map((wcx >> 4u) & 0x0F0F0F0Fu);
      const u32 wl1 = ie_iq4_map(wcy & 0x0F0F0F0Fu);
      const u32 wh1 = ie_iq4_map((wcy >> 4u) & 0x0F0F0F0Fu);
      const u32 wl2 = ie_iq4_map(wcz & 0x0F0F0F0Fu);
      const u32 wh2 = ie_iq4_map((wcz >> 4u) & 0x0F0F0F0Fu);
      const u32 wl3 = ie_iq4_map(wcw & 0x0F0F0F0Fu);
      const u32 wh3 = ie_iq4_map((wcw >> 4u) & 0x0F0F0F0Fu);

      const float v0 = sc_f * (float)ink_r6_cm_iq4_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_a, x1_a);
      INK_R6_CM_ACC_FMA(a0_0, a0_1, a0_2, a0_3, j_acc, v0, d_a);

      if (nj > 2u) {
        x0_a = *(const G u32x4 *)(aw2 + b_c * 8u);
        x1_a = *(const G u32x4 *)(aw2 + b_c * 8u + 4u);
        d_a = da2[b_c];
      }
      __builtin_amdgcn_sched_barrier(0);

      const float v1 = sc_f * (float)ink_r6_cm_iq4_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_b, x1_b);
      INK_R6_CM_ACC_FMA(a1_0, a1_1, a1_2, a1_3, j_acc, v1, d_b);

      if (nj > 2u) {
        if (nj > 3u) {
          x0_b = *(const G u32x4 *)(aw3 + b_c * 8u);
          x1_b = *(const G u32x4 *)(aw3 + b_c * 8u + 4u);
          d_b = da3[b_c];
        }
        __builtin_amdgcn_sched_barrier(0);
        const float v2 = sc_f * (float)ink_r6_cm_iq4_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_a, x1_a);
        INK_R6_CM_ACC_FMA(a2_0, a2_1, a2_2, a2_3, j_acc, v2, d_a);

        if (nj > 3u) {
          __builtin_amdgcn_sched_barrier(0);
          const float v3 = sc_f * (float)ink_r6_cm_iq4_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x0_b, x1_b);
          INK_R6_CM_ACC_FMA(a3_0, a3_1, a3_2, a3_3, j_acc, v3, d_b);
        }
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    w_c = w_n; sc_c = sc_n;
  }

  /* Exact wave_tree_f (nb > 16) / half_tree_f (nb <= 16) reduction + epilogue store. */
  const float acc0 = ink_r6_cm_oct_tree_f(full_wave ? ((a0_0 + a0_2) + (a0_1 + a0_3)) : (a0_0 + a0_1));
  if (sl == 0u && r < rows) {
    G float *yp0 = (G float *)(Y + (unsigned long)p0 * ys);
    const G float *rp0 = res ? (const G float *)(res + (unsigned long)p0 * rs) : 0;
    yp0[r] = epilogue(acc0, r, 0, rp0);
  }
  const float acc1 = ink_r6_cm_oct_tree_f(full_wave ? ((a1_0 + a1_2) + (a1_1 + a1_3)) : (a1_0 + a1_1));
  if (sl == 0u && r < rows) {
    G float *yp1 = (G float *)(Y + (unsigned long)p1 * ys);
    const G float *rp1 = res ? (const G float *)(res + (unsigned long)p1 * rs) : 0;
    yp1[r] = epilogue(acc1, r, 0, rp1);
  }
  if (nj > 2u) {
    const float acc2 = ink_r6_cm_oct_tree_f(full_wave ? ((a2_0 + a2_2) + (a2_1 + a2_3)) : (a2_0 + a2_1));
    if (sl == 0u && r < rows) {
      G float *yp2 = (G float *)(Y + (unsigned long)p2 * ys);
      const G float *rp2 = res ? (const G float *)(res + (unsigned long)p2 * rs) : 0;
      yp2[r] = epilogue(acc2, r, 0, rp2);
    }
  }
  if (nj > 3u) {
    const float acc3 = ink_r6_cm_oct_tree_f(full_wave ? ((a3_0 + a3_2) + (a3_1 + a3_3)) : (a3_0 + a3_1));
    if (sl == 0u && r < rows) {
      G float *yp3 = (G float *)(Y + (unsigned long)p3 * ys);
      const G float *rp3 = res ? (const G float *)(res + (unsigned long)p3 * rs) : 0;
      yp3[r] = epilogue(acc3, r, 0, rp3);
    }
  }
}

/* ============================================================================
 * Kernel 4: ie_inkling_gexp_raw_iq3xxs_slot_r6
 *   Direct per-slot IQ3_XXS verify baseline (NO sort dependency!).
 *   - Grid: ((rows + 31u) / 32u, T * K, 1u); by = tk * K + sk.
 *   - Reads e = route[tk * rst + sk] directly via readfirstlane; executes
 *     the high-BW r1_g03 4x-unrolled 4-row/wave 1-pair body with hoisted SGPR
 *     base pointers, FIFO loadcnt-ordered X/weight lookahead, streaming dot4
 *     (<= 64 VGPR, 1 KiB LDS LUT, 0 spills), 100% bitwise-exact.
 * ========================================================================== */

KERNEL ie_inkling_gexp_raw_iq3xxs_slot_r6(const G u8 *qw, const G u8 *X, G float *Y, u32 rows, u32 nb,
                                          const G u32 *route, u32 rst, u32 K, u32 q_bytes,
                                          u32 xs, u32 ys, u32 xdiv, const G float *res, u32 rs) {
#pragma clang fp contract(off)
  static LDS u32 ink_r6_cm_iq3_slot_lut[256] __attribute__((aligned(16)));
  if (rows == 0u || nb < 8u || (nb & 7u) != 0u || K == 0u || !qw || !X || !Y || !route) return;

  const u32 rbase = wgid() * 32u;
  if (rbase >= rows) return;

  const u32 t0 = tid();
  ink_r6_cm_iq3_slot_lut[t0] = ie_rq_iq3xxs_grid[t0];
  lbarrier();

  const u32 by = tokid();
  const u32 tk = (K == 8u) ? (by >> 3u) : (by / K);
  const u32 sk = (K == 8u) ? (by & 7u) : (by % K);
  const u32 e = (u32)__builtin_amdgcn_readfirstlane(
      (int)((const G u32 *)((const G u8 *)route + (unsigned long)tk * rst))[sk]);
  const u32 x_idx = (xdiv == 1u) ? by : (xdiv == K ? tk : ink_r6_cm_div_x(by, xdiv));

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 r = rbase + wv * 4u + k;
  const u32 rr = r < rows ? r : rows - 1u;
  const u32 mb = nb >> 3u;
  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * q_bytes);
  const G u8 *bp = ebase + (unsigned long)rr * ((unsigned long)mb * 98ul);
  const G u8 *xp = rq_ptr_u(X + (unsigned long)x_idx * xs);
  const G float *dap = (const G float *)(xp + 32ul * (unsigned long)nb);

  const u32 qo = 2u + 8u * s;
  const u32 auxo = 66u + 4u * s;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

  /* Prologue: prefetch superblock 0 weights. */
  u32 d_c = ie_rq_a16(bp, 0u);
  u32 aux_c = ie_rq_a32(bp, auxo);
  u32 q01_c = ie_rq_a32(bp, qo);
  u32 q23_c = ie_rq_a32(bp, qo + 4u);
  __asm__ volatile("" : "+v"(d_c), "+v"(aux_c), "+v"(q01_c), "+v"(q23_c));

  const u32 mb4 = mb & ~3u;
#pragma unroll 1
  for (u32 sb0 = 0u; sb0 < mb4; sb0 += 4u) {
#pragma unroll
    for (u32 j = 0u; j < 4u; j++) {
      const u32 sb = sb0 + j;
      const u32 b = 8u * sb + s;
      const G u8 *bp_nxt = (sb + 1u < mb) ? (bp + 98u) : bp;

      const u32x4 x0_a = *(const G u32x4 *)(xp + 32u * b);
      const u32x4 x1_a = *(const G u32x4 *)(xp + 32u * b + 16u);
      const float dav_a = dap[b];

      const u32 d_n = ie_rq_a16(bp_nxt, 0u);
      const u32 aux_n = ie_rq_a32(bp_nxt, auxo);
      const u32 q01_n = ie_rq_a32(bp_nxt, qo);
      const u32 q23_n = ie_rq_a32(bp_nxt, qo + 4u);

      const u32 qp0 = q01_c & 0xFFFFu;
      const u32 qp1 = q01_c >> 16u;
      const u32 qp2 = q23_c & 0xFFFFu;
      const u32 qp3 = q23_c >> 16u;
      const u32 g_lo0 = ink_r6_cm_iq3_slot_lut[qp0 & 255u];
      const u32 g_hi0 = ink_r6_cm_iq3_slot_lut[qp0 >> 8u];
      const u32 g_lo1 = ink_r6_cm_iq3_slot_lut[qp1 & 255u];
      const u32 g_hi1 = ink_r6_cm_iq3_slot_lut[qp1 >> 8u];
      const u32 g_lo2 = ink_r6_cm_iq3_slot_lut[qp2 & 255u];
      const u32 g_hi2 = ink_r6_cm_iq3_slot_lut[qp2 >> 8u];
      const u32 g_lo3 = ink_r6_cm_iq3_slot_lut[qp3 & 255u];
      const u32 g_hi3 = ink_r6_cm_iq3_slot_lut[qp3 >> 8u];
      __builtin_amdgcn_sched_barrier(0);

      const float v = ink_r6_cm_iq3_stream_dot(
          d_c, aux_c, g_lo0, g_hi0, g_lo1, g_hi1, g_lo2, g_hi2, g_lo3, g_hi3,
          x0_a, x1_a, dav_a);
      INK_R6_CM_ACC_ADD(a0, a1, a2, a3, j, v);
      __builtin_amdgcn_sched_barrier(0);
      bp = bp_nxt;
      d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
    }
  }

  if (mb4 < mb) {
#pragma unroll
    for (u32 j = 0u; j < 3u; j++) {
      const u32 sb = mb4 + j;
      if (sb < mb) {
        const u32 b = 8u * sb + s;
        const int has_nxt = (sb + 1u < mb);

        const u32x4 x0_a = *(const G u32x4 *)(xp + 32u * b);
        const u32x4 x1_a = *(const G u32x4 *)(xp + 32u * b + 16u);
        const float dav_a = dap[b];

        u32 d_n = 0u, aux_n = 0u, q01_n = 0u, q23_n = 0u;
        if (has_nxt) {
          bp += 98u;
          d_n = ie_rq_a16(bp, 0u);
          aux_n = ie_rq_a32(bp, auxo);
          q01_n = ie_rq_a32(bp, qo);
          q23_n = ie_rq_a32(bp, qo + 4u);
        }

        const u32 qp0 = q01_c & 0xFFFFu;
        const u32 qp1 = q01_c >> 16u;
        const u32 qp2 = q23_c & 0xFFFFu;
        const u32 qp3 = q23_c >> 16u;
        const u32 g_lo0 = ink_r6_cm_iq3_slot_lut[qp0 & 255u];
        const u32 g_hi0 = ink_r6_cm_iq3_slot_lut[qp0 >> 8u];
        const u32 g_lo1 = ink_r6_cm_iq3_slot_lut[qp1 & 255u];
        const u32 g_hi1 = ink_r6_cm_iq3_slot_lut[qp1 >> 8u];
        const u32 g_lo2 = ink_r6_cm_iq3_slot_lut[qp2 & 255u];
        const u32 g_hi2 = ink_r6_cm_iq3_slot_lut[qp2 >> 8u];
        const u32 g_lo3 = ink_r6_cm_iq3_slot_lut[qp3 & 255u];
        const u32 g_hi3 = ink_r6_cm_iq3_slot_lut[qp3 >> 8u];
        __builtin_amdgcn_sched_barrier(0);

        const float v = ink_r6_cm_iq3_stream_dot(
            d_c, aux_c, g_lo0, g_hi0, g_lo1, g_hi1, g_lo2, g_hi2, g_lo3, g_hi3,
            x0_a, x1_a, dav_a);
        INK_R6_CM_ACC_ADD(a0, a1, a2, a3, j, v);
        __builtin_amdgcn_sched_barrier(0);
        d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
      }
    }
  }

  /* Exact wave_tree_f parenthesization: (a0 + a2) + (a1 + a3), then oct_tree_f. */
  const float acc = ink_r6_cm_oct_tree_f((a0 + a2) + (a1 + a3));
  if (s == 0u && r < rows) {
    G float *yp = (G float *)((G u8 *)Y + (unsigned long)by * ys);
    const G float *rp = res ? (const G float *)((const G u8 *)res + (unsigned long)by * rs) : 0;
    yp[r] = rp ? rp[r] + acc : acc;
  }
}

/* ============================================================================
 * Kernel 5: ie_inkling_gexp_iq4nl_slot_r6
 *   Direct per-slot IQ4_NL verify baseline (NO sort dependency!).
 *   - Grid: ((rows + 31u) / 32u, T * K, 1u); by = tk * K + sk.
 *   - Reads e = route[tk * rst + sk] directly via readfirstlane; executes
 *     the high-BW r1_g04 2-block-overlapped 4-row/wave 1-pair body (0 LDS,
 *     <= 64 VGPR, 0 spills), 100% bitwise-exact vs ver_r4 & ie_gexp_t.
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq4nl_slot_r6(const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
                                     u32 rows, u32 nb, const G u32 *route, u32 rst, u32 K,
                                     u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv, u32 kind,
                                     const G u8 *res, u32 rs) {
#pragma clang fp contract(off)
  (void)kind;
  if (rows == 0u || nb == 0u || K == 0u || !qw0 || !qs0 || !X || !Y || !route) return;

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 r_wave = (wgid() * 8u + wv) * 4u;
  if (r_wave >= rows) return;

  const u32 by = tokid();
  const u32 tk = (K == 8u) ? (by >> 3u) : (by / K);
  const u32 sk = (K == 8u) ? (by & 7u) : (by % K);
  const u32 e = (u32)__builtin_amdgcn_readfirstlane(
      (int)((const G u32 *)((const G u8 *)route + (unsigned long)tk * rst))[sk]);
  const u32 x_idx = (xdiv == 1u) ? by : (xdiv == K ? tk : ink_r6_cm_div_x(by, xdiv));

  const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
  const u32 r = r_wave + sub;

  const G u8 *w_base = rq_ptr_u((const G u8 *)(qw0 + (unsigned long)e * qwn));
  const G u8 *q_base = rq_ptr_u((const G u8 *)(qs0 + (unsigned long)e * qsn));
  const G u8 *xp = rq_ptr_u(X + (unsigned long)x_idx * xs);

  if (r < rows) {
    const G u8 *rw = w_base + (unsigned long)r * ((unsigned long)nb * 16ul);
    const G f16 *rs_p = (const G f16 *)q_base + (unsigned long)r * nb;
    const G u32 *aw = (const G u32 *)xp;
    const G float *da = (const G float *)(xp + 32ul * (unsigned long)nb);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    u32 t = 0u;
    for (; t + 32u <= nb; t += 32u) {
      const u32 b0 = t + sl, b1 = t + 8u + sl;
      const u32x4 w0 = *(const G u32x4 *)(rw + b0 * 16u);
      const float sc0 = (float)rs_p[b0];
      const u32x4 x00 = *(const G u32x4 *)(aw + b0 * 8u);
      const u32x4 x01 = *(const G u32x4 *)(aw + b0 * 8u + 4u);
      const float d0 = da[b0];
      const u32x4 w1 = *(const G u32x4 *)(rw + b1 * 16u);
      const float sc1 = (float)rs_p[b1];
      const u32x4 x10 = *(const G u32x4 *)(aw + b1 * 8u);
      const u32x4 x11 = *(const G u32x4 *)(aw + b1 * 8u + 4u);
      const float d1 = da[b1];
      a0 = __builtin_fmaf(sc0 * (float)ink_r6_cm_iq4_dot4(w0, x00, x01), d0, a0);
      a1 = __builtin_fmaf(sc1 * (float)ink_r6_cm_iq4_dot4(w1, x10, x11), d1, a1);

      const u32 b2 = t + 16u + sl, b3 = t + 24u + sl;
      const u32x4 w2 = *(const G u32x4 *)(rw + b2 * 16u);
      const float sc2 = (float)rs_p[b2];
      const u32x4 x20 = *(const G u32x4 *)(aw + b2 * 8u);
      const u32x4 x21 = *(const G u32x4 *)(aw + b2 * 8u + 4u);
      const float d2 = da[b2];
      const u32x4 w3 = *(const G u32x4 *)(rw + b3 * 16u);
      const float sc3 = (float)rs_p[b3];
      const u32x4 x30 = *(const G u32x4 *)(aw + b3 * 8u);
      const u32x4 x31 = *(const G u32x4 *)(aw + b3 * 8u + 4u);
      const float d3 = da[b3];
      a2 = __builtin_fmaf(sc2 * (float)ink_r6_cm_iq4_dot4(w2, x20, x21), d2, a2);
      a3 = __builtin_fmaf(sc3 * (float)ink_r6_cm_iq4_dot4(w3, x30, x31), d3, a3);
    }
    if (t + sl < nb) {
      const u32 b = t + sl;
      const u32x4 w0 = *(const G u32x4 *)(rw + b * 16u);
      const float sc0 = (float)rs_p[b];
      const u32x4 x00 = *(const G u32x4 *)(aw + b * 8u);
      const u32x4 x01 = *(const G u32x4 *)(aw + b * 8u + 4u);
      const float d0 = da[b];
      a0 = __builtin_fmaf(sc0 * (float)ink_r6_cm_iq4_dot4(w0, x00, x01), d0, a0);
    }
    if (t + 8u + sl < nb) {
      const u32 b = t + 8u + sl;
      const u32x4 w1 = *(const G u32x4 *)(rw + b * 16u);
      const float sc1 = (float)rs_p[b];
      const u32x4 x10 = *(const G u32x4 *)(aw + b * 8u);
      const u32x4 x11 = *(const G u32x4 *)(aw + b * 8u + 4u);
      const float d1 = da[b];
      a1 = __builtin_fmaf(sc1 * (float)ink_r6_cm_iq4_dot4(w1, x10, x11), d1, a1);
    }
    if (t + 16u + sl < nb) {
      const u32 b = t + 16u + sl;
      const u32x4 w2 = *(const G u32x4 *)(rw + b * 16u);
      const float sc2 = (float)rs_p[b];
      const u32x4 x20 = *(const G u32x4 *)(aw + b * 8u);
      const u32x4 x21 = *(const G u32x4 *)(aw + b * 8u + 4u);
      const float d2 = da[b];
      a2 = __builtin_fmaf(sc2 * (float)ink_r6_cm_iq4_dot4(w2, x20, x21), d2, a2);
    }
    if (t + 24u + sl < nb) {
      const u32 b = t + 24u + sl;
      const u32x4 w3 = *(const G u32x4 *)(rw + b * 16u);
      const float sc3 = (float)rs_p[b];
      const u32x4 x30 = *(const G u32x4 *)(aw + b * 8u);
      const u32x4 x31 = *(const G u32x4 *)(aw + b * 8u + 4u);
      const float d3 = da[b];
      a3 = __builtin_fmaf(sc3 * (float)ink_r6_cm_iq4_dot4(w3, x30, x31), d3, a3);
    }
    const int full_wave = (nb > 16u);
    const float acc = ink_r6_cm_oct_tree_f(full_wave ? ((a0 + a2) + (a1 + a3)) : (a0 + a1));
    if (sl == 0u) {
      G float *yp = (G float *)(Y + (unsigned long)by * ys);
      const G float *rp = res ? (const G float *)(res + (unsigned long)by * rs) : 0;
      yp[r] = epilogue(acc, r, 0, rp);
    }
  }
}

#undef INK_R6_CM_ACC_FMA
#undef INK_R6_CM_ACC_ADD
#undef INK_R6_CM_MAX_PAIRS
#undef INK_R6_CM_MAX_EXP

#endif /* !IE_MEGA */
