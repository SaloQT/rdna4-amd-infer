/* ============================================================================
 * Inkling-2-Mini Round 7 (Fusion A): Exact Fused Router + Top-K Route + Compact Sort
 * Target: RX 9070 XT (gfx1201, wave32, NT=256), Clang 21 bit_cast-safe, 0 spills.
 *
 * Included at EOF of kernels/ie_inkling_kernels.c (after ie_kernels.c).
 * All local symbols/LDS/macros prefixed ink_r7_rf_ / INK_R7_RF_; macros #undef'd at EOF.
 *
 * Replaces across ALL 48 logical visits (dim=2304, n_exp=128, K=8, F32 router):
 *   - T == 1:     [OP_GEMV_F32(wr), OP_MOE_ROUTE]                 -> 1 launch (-48/step)
 *   - T == 2..16: [OP_GEMV_F32(wr), OP_MOE_ROUTE, moe_sort_compact_r6] -> 1 launch (-96/step)
 *
 * Exported KERNELs, Grids, Guards, Disjointness, and Parity:
 *
 * 1. KERNEL ie_inkling_router_route_exact_t1_r7(
 *        const G float *w, const G float *x, G float *logits,
 *        u32 rows, u32 cols, u32 K, const G float *ebias, float scale,
 *        G u32 *route, G u32 *count)
 *    - Grid:   ((rows >= 253u && rows <= 256u) ? 257u : rows, 1u, 1u)
 *              (128 WGs for rows=128; padded WGs r >= rows early-return safely.)
 *    - Guard:  T == 1u && !mk.rec && !o0->v && o0->res < 0 && !o0->nv &&
 *              o1->kind == OP_MOE_ROUTE && o1->a == o0->b && o1->aoff == 0u && o0->boff == 0u &&
 *              o0->w->rows > 0u && o0->w->rows <= 256u && o1->n == o0->w->rows &&
 *              o1->nh > 0u && o1->nh <= 32u && o0->w->cols > 0u && b->gm_count != NULL &&
 *              buf_disjoint(g, o0->a, o0->b) && buf_disjoint(g, o0->a, o1->b) &&
 *              buf_disjoint(g, o0->b, o1->b).
 *    - Model:  ZERO model.c edits required.
 *    - Count:  Reuses b->gm_count (1 u32, pre-zeroed; last WG resets *count = 0u).
 *    - Parity: 100% bitwise-exact (0-bit-diff) vs ie_gemv_f32 + ie_moe_route
 *              (calls exact gemv_f32_row + exact moe_route8; eliminates R2 bit 31's
 *               4-acc FMA tie-flip KLD=0.0033!).
 *
 * 2. KERNEL ie_inkling_router_route_sort_ver_r7(
 *        const G float *w, const G float *x, G float *logits,
 *        u32 rows, u32 cols, u32 T, u32 K, const G float *ebias, float scale,
 *        G u32 *route, u32 xs, u32 ls, u32 rst,
 *        G u32 *offs, G u32 *list, G u32 *packs, G u32 *n_packs, G u32 *count)
 *    - Grid:   (((rows + 3u) / 4u), 1u, 1u)  [32 WGs for rows=128; pad-safe]
 *    - Guard:  T >= 2u && T <= 16u && !mk.rec && !o0->v && o0->res < 0 && !o0->nv &&
 *              o1->kind == OP_MOE_ROUTE && o1->a == o0->b && o1->aoff == 0u && o0->boff == 0u &&
 *              o0->w->rows > 0u && o0->w->rows <= 256u && o1->n == o0->w->rows &&
 *              o1->nh > 0u && o1->nh <= 32u && T * o1->nh <= 512u && o0->w->cols > 0u &&
 *              b->gm_count != NULL &&
 *              buf_disjoint(g, o0->a, o0->b) && buf_disjoint(g, o0->a, o1->b) &&
 *              buf_disjoint(g, o0->b, o1->b).
 *              Pass offs=b->ms_offs, list=b->ms_list, packs=b->ms_packs, n_packs=b->ms_npacks
 *              (or offs=list=packs=n_packs=NULL if skip_sort is active).
 *    - Model:  ZERO model.c edits required.
 *    - Count:  Reuses b->gm_count (1 u32, pre-zeroed; last WG resets *count = 0u).
 *    - Parity: 100% bitwise-exact (0-bit-diff) vs ie_inkling_gemv_f32_router_ver_r4 +
 *              ie_moe_route + ie_inkling_moe_sort_compact_r6 (exact INK_R4_MOE_FMA4 +
 *              wave_tree_f + 2-wave sum + exact moe_route8 + exact offs/packs/n_packs
 *              and exact-set list[] permutation feeding downstream compact MoE GEMVs).
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r7_rf float/u32 must be 4 bytes");

#define INK_R7_RF_MAX_EXP   256u
#define INK_R7_RF_MAX_PAIRS 512u

struct __attribute__((packed, aligned(4))) ink_r7_rf_f32x4p {
  f32x4 v;
};

static inline __attribute__((always_inline)) f32x4 ink_r7_rf_ld_f4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r7_rf_f32x4p *)p)->v;
}

static inline __attribute__((always_inline)) u32 ink_r7_rf_wave_scan_inc(u32 x) {
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x111, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x112, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x114, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x118, 0xF, 0xF, true);
  const u32 lo16 = (u32)__builtin_amdgcn_readlane((int)x, 15);
  if (lane() >= 16u) x += lo16;
  return x;
}

#define INK_R7_RF_FMA4(acc_, w_, x_)                    \
  do {                                                  \
    (acc_) = __builtin_fmaf((w_).x, (x_).x, (acc_));    \
    (acc_) = __builtin_fmaf((w_).y, (x_).y, (acc_));    \
    (acc_) = __builtin_fmaf((w_).z, (x_).z, (acc_));    \
    (acc_) = __builtin_fmaf((w_).w, (x_).w, (acc_));    \
  } while (0)

#define INK_R7_RF_STORE_LOGIT(k_, acc_)                                           \
  do {                                                                            \
    const u32 tk_ = t0 + (k_);                                                    \
    const float sum_ = (acc_) + ink_r7_rf_ver_lds.rtr_red[row_local][(k_)];       \
    G float *yp_ = (G float *)((G u8 *)logits + (unsigned long)tk_ * ls);         \
    yp_[r] = sum_;                                                                \
  } while (0)

/* ============================================================================
 * Kernel 1: Exact T=1 Fused Router GEMV + Top-K Route (0-bit diff vs ie_gemv_f32 + ie_moe_route)
 * ========================================================================== */

KERNEL ie_inkling_router_route_exact_t1_r7(const G float *w, const G float *x, G float *logits,
                                           u32 rows, u32 cols, u32 K, const G float *ebias, float scale,
                                           G u32 *route, G u32 *count) {
  static LDS u32 ink_r7_rf_t1_last;
  if (rows == 0u || rows > INK_R7_RF_MAX_EXP || cols == 0u || K == 0u || K > 32u ||
      !w || !x || !logits || !route || !count)
    return;

  const u32 r = wgid();
  if (r >= rows) return;

  /* Exact ie_gemv_f32 reduction order via gemv_f32_row (256 threads/row, wg_sum). */
  const float acc = gemv_f32_row(w + (unsigned long)r * cols, x, cols);
  if (tid() == 0u) logits[r] = acc;

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r7_rf_t1_last =
        (__scoped_atomic_fetch_add(count, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == rows - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r7_rf_t1_last) return;

  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  if (tid() < 32u) moe_route8(logits, rows, K, ebias, scale, route);
  if (tid() == 0u) *count = 0u;
}

/* ============================================================================
 * Kernel 2: Exact T=2..16 Fused Router GEMV + 8-Wave Parallel Top-K Route + Compact Sort
 *   - Phase 1: exact ie_inkling_gemv_f32_router_ver_r4 (4 rows/WG, 2 waves/row, 32 WGs).
 *   - Atomic last-WG barrier on count (reuses b->gm_count, self-resets to 0).
 *   - Phase 2: 8 waves route all T<=16 tokens in parallel (exact moe_route8) AND directly
 *              populate LDS histogram + per-pair (expert << 16 | rank) with 0 route re-reads.
 *   - Phase 3: dual DPP prefix scan -> offs[], packs[], *n_packs, and zero-atomic list[] scatter.
 *   - Total LDS: 3,332 B (<= 4 KiB, 4 WGs/CU occupancy, 0 spills).
 * ========================================================================== */

static inline __attribute__((always_inline)) void ink_r7_rf_route8_and_hist(
    const G float *lg, u32 n, u32 K, const G float *bias, float scale, G u32 *out,
    u32 tk, int do_sort, LDSP u32 *sort_cnt, LDSP u32 *sort_er) {
  const u32 l = lane();
  float sg[8], c[8];
  for (u32 j = 0u; j < 8u; j++) {
    const u32 i = l + 32u * j;
    sg[j] = i < n ? 1.0f / (1.0f + __builtin_expf(-lg[i])) : 0.0f;
    c[j] = i < n ? (bias ? sg[j] + bias[i] : sg[j]) : -__builtin_inff();
  }
  u32 myid = 0u;
  float myw = 0.0f, sum = 0.0f;
  for (u32 k = 0u; k < K; k++) {
    float bv = -__builtin_inff();
    for (u32 j = 0u; j < 8u; j++) bv = __builtin_fmaxf(bv, c[j]);
    const float M = wave_max_any(bv);
    u32 bi = 0xFFFFFFFFu;
    for (u32 j = 8u; j-- > 0u;)
      if (c[j] == M) bi = l + 32u * j;
    bi = wave_min_u(bi);
    const u32 own = bi & 31u, jj = bi >> 5u;
    float sb = 0.0f;
    for (u32 j = 0u; j < 8u; j++) {
      if (j == jj) {
        sb = sg[j];
        if (l == own) c[j] = -__builtin_inff();
      }
    }
    sb = I2F(__builtin_amdgcn_readlane(F2I(sb), __builtin_amdgcn_readfirstlane((int)own)));
    if (l == k) myid = bi, myw = sb;
    sum += sb;
  }
  const float inv = 1.0f / (sum + 1e-20f);
  if (l < K) {
    out[l] = myid;
    ((G float *)out)[K + l] = myw * inv * scale;
    if (do_sort) {
      const u32 p = tk * K + l;
      const u32 rk = __atomic_fetch_add(&sort_cnt[myid], 1u, __ATOMIC_RELAXED);
      sort_er[p] = (myid << 16u) | rk;
    }
  }
}

KERNEL ie_inkling_router_route_sort_ver_r7(
    const G float *w, const G float *x, G float *logits,
    u32 rows, u32 cols, u32 T, u32 K, const G float *ebias, float scale,
    G u32 *route, u32 xs, u32 ls, u32 rst,
    G u32 *offs, G u32 *list, G u32 *packs, G u32 *n_packs, G u32 *count) {
  static LDS struct {
    float rtr_red[4][8];
    u32   sort_cnt[INK_R7_RF_MAX_EXP];
    u32   sort_er[INK_R7_RF_MAX_PAIRS];
    u32   wsum[9];
    u32   psum[9];
    u32   is_last;
  } ink_r7_rf_ver_lds __attribute__((aligned(16)));

  const u32 np = T * K;
  if (rows == 0u || rows > INK_R7_RF_MAX_EXP || cols == 0u || T == 0u || T > 16u ||
      K == 0u || K > 32u || np > INK_R7_RF_MAX_PAIRS ||
      !w || !x || !logits || !route || !count)
    return;

  const u32 n_wg = (rows + 3u) >> 2u;
  const u32 g = wgid();
  if (g >= n_wg) return;

  const u32 t = tid();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = lane();
  const u32 row_local = wv >> 1u;
  const u32 w_sub = wv & 1u;
  const u32 r = g * 4u + row_local;
  const u32 rr = r < rows ? r : rows - 1u;
  const u32 lane64 = (w_sub << 5u) | l;

  const G float *rowp = (const G float *)rq_ptr_u((const G u8 *)(w + (unsigned long)rr * cols));
  const u32 cols4 = cols >> 2u;

  /* Phase 1: exact ie_inkling_gemv_f32_router_ver_r4 (4 rows/WG, 2 waves/row, 8-tok chunks). */
  for (u32 t0 = 0u; t0 < T; t0 += 8u) {
    const u32 nt = (u32)__builtin_amdgcn_readfirstlane((int)(T - t0 < 8u ? T - t0 : 8u));
    const G float *xp0 = (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 0u) * xs);
    const G float *xp1 = nt > 1u ? (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 1u) * xs) : xp0;
    const G float *xp2 = nt > 2u ? (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 2u) * xs) : xp0;
    const G float *xp3 = nt > 3u ? (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 3u) * xs) : xp0;
    const G float *xp4 = nt > 4u ? (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 4u) * xs) : xp0;
    const G float *xp5 = nt > 5u ? (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 5u) * xs) : xp0;
    const G float *xp6 = nt > 6u ? (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 6u) * xs) : xp0;
    const G float *xp7 = nt > 7u ? (const G float *)rq_ptr_u((const G u8 *)x + (unsigned long)(t0 + 7u) * xs) : xp0;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

    if (lane64 < cols4) {
      f32x4 w_c = ink_r7_rf_ld_f4(rowp + (unsigned long)lane64 * 4ul);
      __builtin_amdgcn_sched_barrier(0);

#pragma unroll 1
      for (u32 c4 = lane64; c4 < cols4; c4 += 64u) {
        const u32 cn = c4 + 64u;
        f32x4 w_n = (f32x4)(0.0f);
        if (cn < cols4) w_n = ink_r7_rf_ld_f4(rowp + (unsigned long)cn * 4ul);

        const unsigned long off = (unsigned long)c4 * 4ul;
        f32x4 xv0 = ink_r7_rf_ld_f4(xp0 + off);
        f32x4 xv1 = nt > 1u ? ink_r7_rf_ld_f4(xp1 + off) : (f32x4)(0.0f);
        __builtin_amdgcn_sched_barrier(0);
        INK_R7_RF_FMA4(a0, w_c, xv0);

        if (nt > 1u) {
          if (nt > 2u) xv0 = ink_r7_rf_ld_f4(xp2 + off);
          __builtin_amdgcn_sched_barrier(0);
          INK_R7_RF_FMA4(a1, w_c, xv1);
          if (nt > 2u) {
            if (nt > 3u) xv1 = ink_r7_rf_ld_f4(xp3 + off);
            __builtin_amdgcn_sched_barrier(0);
            INK_R7_RF_FMA4(a2, w_c, xv0);
            if (nt > 3u) {
              if (nt > 4u) xv0 = ink_r7_rf_ld_f4(xp4 + off);
              __builtin_amdgcn_sched_barrier(0);
              INK_R7_RF_FMA4(a3, w_c, xv1);
              if (nt > 4u) {
                if (nt > 5u) xv1 = ink_r7_rf_ld_f4(xp5 + off);
                __builtin_amdgcn_sched_barrier(0);
                INK_R7_RF_FMA4(a4, w_c, xv0);
                if (nt > 5u) {
                  if (nt > 6u) xv0 = ink_r7_rf_ld_f4(xp6 + off);
                  __builtin_amdgcn_sched_barrier(0);
                  INK_R7_RF_FMA4(a5, w_c, xv1);
                  if (nt > 6u) {
                    if (nt > 7u) xv1 = ink_r7_rf_ld_f4(xp7 + off);
                    __builtin_amdgcn_sched_barrier(0);
                    INK_R7_RF_FMA4(a6, w_c, xv0);
                    if (nt > 7u) {
                      __builtin_amdgcn_sched_barrier(0);
                      INK_R7_RF_FMA4(a7, w_c, xv1);
                    }
                  }
                }
              }
            }
          }
        }
        __builtin_amdgcn_sched_barrier(0);
        w_c = w_n;
      }
    }

    if (cols & 3u) {
      const u32 ct = (cols4 << 2u) + lane64;
      if (ct < cols) {
        const float ws = rowp[ct];
        a0 = __builtin_fmaf(ws, xp0[ct], a0);
        if (nt > 1u) a1 = __builtin_fmaf(ws, xp1[ct], a1);
        if (nt > 2u) a2 = __builtin_fmaf(ws, xp2[ct], a2);
        if (nt > 3u) a3 = __builtin_fmaf(ws, xp3[ct], a3);
        if (nt > 4u) a4 = __builtin_fmaf(ws, xp4[ct], a4);
        if (nt > 5u) a5 = __builtin_fmaf(ws, xp5[ct], a5);
        if (nt > 6u) a6 = __builtin_fmaf(ws, xp6[ct], a6);
        if (nt > 7u) a7 = __builtin_fmaf(ws, xp7[ct], a7);
      }
    }

    a0 = wave_tree_f(a0);
    if (nt > 1u) a1 = wave_tree_f(a1);
    if (nt > 2u) a2 = wave_tree_f(a2);
    if (nt > 3u) a3 = wave_tree_f(a3);
    if (nt > 4u) a4 = wave_tree_f(a4);
    if (nt > 5u) a5 = wave_tree_f(a5);
    if (nt > 6u) a6 = wave_tree_f(a6);
    if (nt > 7u) a7 = wave_tree_f(a7);

    if (w_sub == 1u && l == 0u) {
      ink_r7_rf_ver_lds.rtr_red[row_local][0] = a0;
      if (nt > 1u) ink_r7_rf_ver_lds.rtr_red[row_local][1] = a1;
      if (nt > 2u) ink_r7_rf_ver_lds.rtr_red[row_local][2] = a2;
      if (nt > 3u) ink_r7_rf_ver_lds.rtr_red[row_local][3] = a3;
      if (nt > 4u) ink_r7_rf_ver_lds.rtr_red[row_local][4] = a4;
      if (nt > 5u) ink_r7_rf_ver_lds.rtr_red[row_local][5] = a5;
      if (nt > 6u) ink_r7_rf_ver_lds.rtr_red[row_local][6] = a6;
      if (nt > 7u) ink_r7_rf_ver_lds.rtr_red[row_local][7] = a7;
    }
    lbarrier();

    if (w_sub == 0u && l == 0u && r < rows) {
      INK_R7_RF_STORE_LOGIT(0u, a0);
      if (nt > 1u) INK_R7_RF_STORE_LOGIT(1u, a1);
      if (nt > 2u) INK_R7_RF_STORE_LOGIT(2u, a2);
      if (nt > 3u) INK_R7_RF_STORE_LOGIT(3u, a3);
      if (nt > 4u) INK_R7_RF_STORE_LOGIT(4u, a4);
      if (nt > 5u) INK_R7_RF_STORE_LOGIT(5u, a5);
      if (nt > 6u) INK_R7_RF_STORE_LOGIT(6u, a6);
      if (nt > 7u) INK_R7_RF_STORE_LOGIT(7u, a7);
    }
    if (t0 + 8u < T) lbarrier();
  }

  /* Inter-WG release/acquire barrier: last WG routes all T tokens & builds compact sort. */
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (t == 0u) {
    ink_r7_rf_ver_lds.is_last =
        (__scoped_atomic_fetch_add(count, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == n_wg - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r7_rf_ver_lds.is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const int do_sort = (offs && list) ? 1 : 0;
  if (do_sort) {
    if (t < rows) ink_r7_rf_ver_lds.sort_cnt[t] = 0u;
    lbarrier();
  }

  /* Phase 2: 8 waves route all T<=16 tokens in parallel (1 iter for T<=8, 2 for T=9..16). */
  for (u32 tk = wv; tk < T; tk += 8u) {
    const G float *lgt = (const G float *)((const G u8 *)logits + (unsigned long)tk * ls);
    G u32 *rtt = (G u32 *)((G u8 *)route + (unsigned long)tk * rst);
    ink_r7_rf_route8_and_hist(lgt, rows, K, ebias, scale, rtt, tk, do_sort,
                              ink_r7_rf_ver_lds.sort_cnt, ink_r7_rf_ver_lds.sort_er);
  }

  /* Phase 3: compact active-expert dual DPP prefix scan + pack emit + parallel list[] scatter. */
  if (do_sort) {
    lbarrier();
    const u32 c = t < rows ? ink_r7_rf_ver_lds.sort_cnt[t] : 0u;
    const u32 pc = (c + 3u) >> 2u;

    const u32 c_inc = ink_r7_rf_wave_scan_inc(c);
    const u32 pc_inc = ink_r7_rf_wave_scan_inc(pc);
    if (l == 31u) {
      ink_r7_rf_ver_lds.wsum[wv] = c_inc;
      ink_r7_rf_ver_lds.psum[wv] = pc_inc;
    }
    lbarrier();

    if (wv == 0u) {
      const u32 ws = l < 8u ? ink_r7_rf_ver_lds.wsum[l] : 0u;
      const u32 ps = l < 8u ? ink_r7_rf_ver_lds.psum[l] : 0u;
      const u32 winc = ink_r7_rf_wave_scan_inc(ws);
      const u32 pinc = ink_r7_rf_wave_scan_inc(ps);
      if (l < 8u) {
        ink_r7_rf_ver_lds.wsum[l] = winc - ws;
        ink_r7_rf_ver_lds.psum[l] = pinc - ps;
      }
      if (l == 7u) {
        ink_r7_rf_ver_lds.wsum[8] = winc;
        ink_r7_rf_ver_lds.psum[8] = pinc;
      }
    }
    lbarrier();

    if (t < rows) {
      const u32 off = ink_r7_rf_ver_lds.wsum[wv] + (c_inc - c);
      const u32 poff = ink_r7_rf_ver_lds.psum[wv] + (pc_inc - pc);
      ink_r7_rf_ver_lds.sort_cnt[t] = off;
      offs[t] = off;

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
      offs[rows] = ink_r7_rf_ver_lds.wsum[8];
      if (n_packs) *n_packs = ink_r7_rf_ver_lds.psum[8];
    }
    lbarrier();

    for (u32 p = t; p < np; p += NT) {
      const u32 er = ink_r7_rf_ver_lds.sort_er[p];
      const u32 e = er >> 16u;
      const u32 rk = er & 0xFFFFu;
      list[ink_r7_rf_ver_lds.sort_cnt[e] + rk] = p;
    }
  }

  if (t == 0u) *count = 0u;
}

#undef INK_R7_RF_STORE_LOGIT
#undef INK_R7_RF_FMA4
#undef INK_R7_RF_MAX_PAIRS
#undef INK_R7_RF_MAX_EXP

#endif /* !IE_MEGA */
