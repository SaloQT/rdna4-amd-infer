/* ============================================================================
 * Inkling-2 Round 1 Group 04 (g04-iq4nl-exp): Specialized IQ4_NL MoE Kernels
 * for RX 9070 XT (gfx1201, wave32, NT=256).
 *
 * Target: MoE ffn_down_exps MAT_IQ4NL (kind=8), shape 2304x1152 (nb=36), 128E/K=8.
 *
 * Exported Kernels, Grids, Guards, and Parity:
 * 1) KERNEL ie_inkling_gexp_iq4nl_r1(const G u32 *qw, const G u16 *qs,
 *      const G u8 *xq, G float *y, u32 rows, u32 nb, const G u32 *route,
 *      u32 qwn, u32 qsn, u32 astr, u32 bstr, u32 kind, u32 two_off)
 *    - Grid:   ((rows + 31u) / 32u, o->nh, 1)  [e.g. (72, 8, 1) = 576 WGs = 9/CU]
 *    - Guard:  T == 1u && kind == MAT_IQ4NL (8u)
 *    - Parity: bitwise-exact vs ie_gexp (4-acc wave_tree_f reconstruction)
 *
 * 2) KERNEL ie_inkling_gemm_iq4nl_exp_r1(const G u32 *qw0, const G u16 *qs0,
 *      const G u8 *X, G u8 *Y, u32 rows, u32 nb, const G u32 *list,
 *      const G u32 *offs, u32 np, u32 qwn, u32 qsn, u32 xs, u32 ys,
 *      u32 xdiv, u32 kind, const G u8 *res, u32 rs)
 *    - Grid:   ((rows + 127u) / 128u, list ? m->n_exp : (np + 31u) / 32u, 1)
 *              [e.g. (18, 128, 1) = 2304 WGs = 36/CU]
 *    - Guard:  T > 1u && kind == MAT_IQ4NL (8u) && nb > 0u && (nb & 3u) == 0u
 *    - Parity: f32-tol (~1e-5 vs ie_gemm_kx / ie_gexp_t; exact WMMA block math)
 * ============================================================================ */

typedef _Float16 ink_r1_g04_f16x4 __attribute__((ext_vector_type(4)));

/* 32-weight IQ4_NL x Q8_0 signed integer dot product using register-resident
 * kvalues_iq4nl permutation (ie_iq4_map) and 8x v_dot4_i32_i8 (ie_dot4_ss). */
static inline int ink_r1_g04_iq4_dot4(u32x4 w4, u32x4 a0, u32x4 a1) {
  u32 d = 0u;
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(w4.x & 0x0F0F0F0Fu), a0.x), ie_iq4_map((w4.x >> 4u) & 0x0F0F0F0Fu), a1.x);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(w4.y & 0x0F0F0F0Fu), a0.y), ie_iq4_map((w4.y >> 4u) & 0x0F0F0F0Fu), a1.y);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(w4.z & 0x0F0F0F0Fu), a0.z), ie_iq4_map((w4.z >> 4u) & 0x0F0F0F0Fu), a1.z);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(w4.w & 0x0F0F0F0Fu), a0.w), ie_iq4_map((w4.w >> 4u) & 0x0F0F0F0Fu), a1.w);
  return (int)d;
}

/* 8-lane subgroup reduction inside each 16-lane DPP row (lanes 0, 8, 16, 24
 * hold the exact sum of their 8-lane subgroup). */
static inline float ink_r1_g04_oct_tree_f(float v) {
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}

/* ----------------------------------------------------------------------------
 * Kernel 1: Decode (T=1) IQ4_NL Expert GEMV — BITWISE-EXACT vs ie_gexp
 *   - 4 rows/wave (8 lanes/row), 32 rows/WG -> 90% lane utilization at nb=36
 *     (4 full 8-block iters + 4-lane tail vs 56.25% in 1 row/wave).
 *   - 4 accumulators (a0..a3) per lane correspond to original 32-lane wave
 *     lanes (sl, sl+8, sl+16, sl+24); (a0 + a2) + (a1 + a3) + oct_tree_f
 *     reconstructs wave_tree_f in the exact IEEE-754 addition order.
 *   - 8 consecutive lanes x 16 B u32x4 = exact 128 B cache-line coalescing;
 *     2x overlapped block loads; zero generic kind branches.
 * ---------------------------------------------------------------------------- */
KERNEL ie_inkling_gexp_iq4nl_r1(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                                const G u32 *route, u32 qwn, u32 qsn, u32 astr, u32 bstr, u32 kind, u32 two_off) {
#pragma clang fp contract(off)
  (void)kind;
  (void)two_off;
  const u32 wv = tid() >> 5, l = lane();
  const u32 r_wave = (wgid() * 8u + wv) * 4u;
  if (r_wave >= rows) return;
  const u32 sub = l >> 3, sl = l & 7u;
  const u32 r = r_wave + sub;
  const u32 s = tokid(), e = route[s];
  const G u32 *w = qw + (unsigned long)e * qwn;
  const G f16 *q = (const G f16 *)(qs + (unsigned long)e * qsn);
  const G u8 *x = xq + (unsigned long)s * astr;
  G float *o = (G float *)((G u8 *)y + (unsigned long)s * bstr);
  if (r < rows) {
    const G u8 *rw = (const G u8 *)(w + (unsigned long)r * nb * 4u);
    const G f16 *rs = q + (unsigned long)r * nb;
    const G u32 *aw = (const G u32 *)x;
    const G float *da = (const G float *)(x + 32u * nb);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    u32 t = 0u;
    for (; t + 32u <= nb; t += 32u) {
      const u32 b0 = t + sl, b1 = t + 8u + sl;
      const u32x4 w0 = *(const G u32x4 *)(rw + b0 * 16u);
      const float sc0 = (float)rs[b0];
      const u32x4 x00 = *(const G u32x4 *)(aw + b0 * 8u);
      const u32x4 x01 = *(const G u32x4 *)(aw + b0 * 8u + 4u);
      const float d0 = da[b0];
      const u32x4 w1 = *(const G u32x4 *)(rw + b1 * 16u);
      const float sc1 = (float)rs[b1];
      const u32x4 x10 = *(const G u32x4 *)(aw + b1 * 8u);
      const u32x4 x11 = *(const G u32x4 *)(aw + b1 * 8u + 4u);
      const float d1 = da[b1];
      a0 = __builtin_fmaf(sc0 * (float)ink_r1_g04_iq4_dot4(w0, x00, x01), d0, a0);
      a1 = __builtin_fmaf(sc1 * (float)ink_r1_g04_iq4_dot4(w1, x10, x11), d1, a1);

      const u32 b2 = t + 16u + sl, b3 = t + 24u + sl;
      const u32x4 w2 = *(const G u32x4 *)(rw + b2 * 16u);
      const float sc2 = (float)rs[b2];
      const u32x4 x20 = *(const G u32x4 *)(aw + b2 * 8u);
      const u32x4 x21 = *(const G u32x4 *)(aw + b2 * 8u + 4u);
      const float d2 = da[b2];
      const u32x4 w3 = *(const G u32x4 *)(rw + b3 * 16u);
      const float sc3 = (float)rs[b3];
      const u32x4 x30 = *(const G u32x4 *)(aw + b3 * 8u);
      const u32x4 x31 = *(const G u32x4 *)(aw + b3 * 8u + 4u);
      const float d3 = da[b3];
      a2 = __builtin_fmaf(sc2 * (float)ink_r1_g04_iq4_dot4(w2, x20, x21), d2, a2);
      a3 = __builtin_fmaf(sc3 * (float)ink_r1_g04_iq4_dot4(w3, x30, x31), d3, a3);
    }
    if (t + sl < nb) {
      const u32 b = t + sl;
      const u32x4 w0 = *(const G u32x4 *)(rw + b * 16u);
      const float sc0 = (float)rs[b];
      const u32x4 x00 = *(const G u32x4 *)(aw + b * 8u);
      const u32x4 x01 = *(const G u32x4 *)(aw + b * 8u + 4u);
      const float d0 = da[b];
      a0 = __builtin_fmaf(sc0 * (float)ink_r1_g04_iq4_dot4(w0, x00, x01), d0, a0);
    }
    if (t + 8u + sl < nb) {
      const u32 b = t + 8u + sl;
      const u32x4 w1 = *(const G u32x4 *)(rw + b * 16u);
      const float sc1 = (float)rs[b];
      const u32x4 x10 = *(const G u32x4 *)(aw + b * 8u);
      const u32x4 x11 = *(const G u32x4 *)(aw + b * 8u + 4u);
      const float d1 = da[b];
      a1 = __builtin_fmaf(sc1 * (float)ink_r1_g04_iq4_dot4(w1, x10, x11), d1, a1);
    }
    if (t + 16u + sl < nb) {
      const u32 b = t + 16u + sl;
      const u32x4 w2 = *(const G u32x4 *)(rw + b * 16u);
      const float sc2 = (float)rs[b];
      const u32x4 x20 = *(const G u32x4 *)(aw + b * 8u);
      const u32x4 x21 = *(const G u32x4 *)(aw + b * 8u + 4u);
      const float d2 = da[b];
      a2 = __builtin_fmaf(sc2 * (float)ink_r1_g04_iq4_dot4(w2, x20, x21), d2, a2);
    }
    if (t + 24u + sl < nb) {
      const u32 b = t + 24u + sl;
      const u32x4 w3 = *(const G u32x4 *)(rw + b * 16u);
      const float sc3 = (float)rs[b];
      const u32x4 x30 = *(const G u32x4 *)(aw + b * 8u);
      const u32x4 x31 = *(const G u32x4 *)(aw + b * 8u + 4u);
      const float d3 = da[b];
      a3 = __builtin_fmaf(sc3 * (float)ink_r1_g04_iq4_dot4(w3, x30, x31), d3, a3);
    }
    const float u0 = a0 + a2;
    const float u1 = a1 + a3;
    const float acc = ink_r1_g04_oct_tree_f(u0 + u1);
    if (sl == 0u) o[r] = acc;
  }
}

/* ----------------------------------------------------------------------------
 * Kernel 2: Prefill (T>1) IQ4_NL MoE WMMA GEMM
 *   - 128x32 tile (WI=1, WJ=2, KB=4, NT=256): matches avg 32 pairs/expert at
 *     T=512, K=8, E=128 (vs GK_T=64 wasting half the tile), 18 WGs/expert x
 *     128E = 2304 WGs = 36/CU for strong MoE load balance across 64 CUs.
 *   - KB=4 keeps only 9 K-chunks / lbarrier()s for nb=36, 100% thread
 *     utilization on rx (256 threads x 16 B = 4096 B), 64-bit f16x4 scale
 *     loads, 14.0 KiB LDS, ~115 VGPR (3 WGs/CU, 0 spills).
 *   - Eliminates unused kx_sa / gk_mw min-term LDS buffers, Q5_K high-bit
 *     arrays, super-block scale loads, and all generic kind branches.
 * ---------------------------------------------------------------------------- */
#define INK_R1_G04_WI 1u
#define INK_R1_G04_T  32u
#define INK_R1_G04_KB 4u
#define INK_R1_G04_R  (8u * 16u * INK_R1_G04_WI) /* 128 rows/WG */
#define INK_R1_G04_WJ (INK_R1_G04_T / 16u)       /* 2 token tiles */
#define INK_R1_G04_LS (INK_R1_G04_KB * 32u + 16u) /* 144 B/row, bank-padded */

static LDS u8 ink_r1_g04_x[2][INK_R1_G04_T * INK_R1_G04_LS] __attribute__((aligned(16)));
static LDS float ink_r1_g04_sw[2][INK_R1_G04_KB * INK_R1_G04_R] __attribute__((aligned(32)));
static LDS float ink_r1_g04_sx[2][INK_R1_G04_T * INK_R1_G04_KB] __attribute__((aligned(16)));

KERNEL ie_inkling_gemm_iq4nl_exp_r1(const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y, u32 rows, u32 nb,
                                    const G u32 *list, const G u32 *offs, u32 np, u32 qwn, u32 qsn, u32 xs, u32 ys,
                                    u32 xdiv, u32 kind, const G u8 *res, u32 rs) {
  (void)kind;
  const u32 e = list ? tokid() : 0u;
  const u32 p0 = list ? offs[e] : tokid() * INK_R1_G04_T;
  const u32 p1 = list ? offs[e + 1u] : (p0 + INK_R1_G04_T < np ? p0 + INK_R1_G04_T : np);
  const u32 T = p1 > p0 ? p1 - p0 : 0u;
  const u32 rt = wgid(), t0 = tid(), w = t0 >> 5, l = lane();
  const u32 rbase = rt * INK_R1_G04_R, h = l >> 4, l16 = l & 15u, wr = 16u * INK_R1_G04_WI * w;
  if (T == 0u || rbase >= rows) return;

  const G u32 *qw = qw0 + (unsigned long)e * qwn;
  const G u16 *qs = qs0 + (unsigned long)e * qsn;

  /* Weight and scale pointers are invariant across token tiles (tbase). */
  const u32 rw_idx = rbase + wr + l16;
  const G u8 *wp = (const G u8 *)qw + (unsigned long)(rw_idx < rows ? rw_idx : rows - 1u) * nb * 16u;
  const u32 sr = rbase + (t0 < INK_R1_G04_R ? t0 : 0u);
  const u32 srr = sr < rows ? sr : rows - 1u;
  const G u16 *swp = qs + (unsigned long)srr * nb;

#define INK_R1_G04_PID(tk_) (list ? list[p0 + (tk_)] : p0 + (tk_))
#define INK_R1_G04_FETCH_W(dst_, kb0_)                                                                 \
  _Pragma("unroll") for (u32 k = 0u; k < INK_R1_G04_KB; k++)                                           \
    dst_[k] = *(const G u32x4 *)(wp + ((kb0_) + k) * 16u)
#define INK_R1_G04_FETCH_X(kb0_)                                                                       \
  do {                                                                                                 \
    rx = *(const G u32x4 *)(xp + (kb0_) * 32u);                                                        \
    if (t0 < INK_R1_G04_R) rsw = *(const G ink_r1_g04_f16x4 *)(swp + (kb0_));                          \
    if (t0 < INK_R1_G04_T * INK_R1_G04_KB) rsx = sxp[(kb0_)];                                          \
  } while (0)
#define INK_R1_G04_STORE_X(bf_)                                                                        \
  do {                                                                                                 \
    *(LDSP u32x4 *)(ink_r1_g04_x[bf_] + (t0 >> 3) * INK_R1_G04_LS + (t0 & 7u) * 16u) = rx;             \
    if (t0 < INK_R1_G04_R) {                                                                           \
      ink_r1_g04_sw[bf_][0u * INK_R1_G04_R + t0] = (float)rsw[0];                                      \
      ink_r1_g04_sw[bf_][1u * INK_R1_G04_R + t0] = (float)rsw[1];                                      \
      ink_r1_g04_sw[bf_][2u * INK_R1_G04_R + t0] = (float)rsw[2];                                      \
      ink_r1_g04_sw[bf_][3u * INK_R1_G04_R + t0] = (float)rsw[3];                                      \
    }                                                                                                  \
    if (t0 < INK_R1_G04_T * INK_R1_G04_KB) ink_r1_g04_sx[bf_][t0] = rsx;                               \
  } while (0)

  for (u32 tbase = 0u; tbase < T; tbase += INK_R1_G04_T) {
    f8 acc[INK_R1_G04_WJ];
#pragma unroll
    for (u32 j = 0u; j < INK_R1_G04_WJ; j++) acc[j] = (f8)(0.0f);

    /* 32 tokens x 4 blocks x 32 B = 4096 B = 256 threads x 16 B (1 load/thread). */
    const u32 tk_x = t0 >> 3;
    const u32 gt_x = tbase + tk_x < T ? tbase + tk_x : T - 1u;
    const G u8 *xp = X + (unsigned long)(INK_R1_G04_PID(gt_x) / xdiv) * xs + (t0 & 7u) * 16u;

    /* 32 tokens x 4 scales = 128 floats (threads t0 < 128). */
    const u32 sxt = tbase + (t0 >> 2);
    const u32 gt_s = sxt < T ? sxt : T - 1u;
    const G float *sxp = (const G float *)(X + (unsigned long)(INK_R1_G04_PID(gt_s) / xdiv) * xs + 32u * nb) + (t0 & 3u);

    u32x4 wa[INK_R1_G04_KB], wn[INK_R1_G04_KB];
    u32x4 rx;
    ink_r1_g04_f16x4 rsw = (ink_r1_g04_f16x4){0.0f, 0.0f, 0.0f, 0.0f};
    float rsx = 0.0f;

    INK_R1_G04_FETCH_W(wa, 0u);
    INK_R1_G04_FETCH_X(0u);
    INK_R1_G04_STORE_X(0u);
    lbarrier();

    u32 bf = 0u;
#pragma unroll 1
    for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R1_G04_KB) {
      const int more = kb0 + INK_R1_G04_KB < nb;
      if (more) {
        INK_R1_G04_FETCH_W(wn, kb0 + INK_R1_G04_KB);
        INK_R1_G04_FETCH_X(kb0 + INK_R1_G04_KB);
      }
      const LDSP u8 *lx = ink_r1_g04_x[bf];
      const LDSP float *lsw = ink_r1_g04_sw[bf], *lsx = ink_r1_g04_sx[bf];
      const u32 nk = nb - kb0 < INK_R1_G04_KB ? nb - kb0 : INK_R1_G04_KB; /* branch keeps k blocks apart for scheduler */
#pragma unroll
      for (u32 k = 0u; k < INK_R1_G04_KB; k++) {
        if (k >= nk) break;
        u32x4 b[INK_R1_G04_WJ];
#pragma unroll
        for (u32 j = 0u; j < INK_R1_G04_WJ; j++)
          b[j] = *(const LDSP u32x4 *)(lx + (16u * j + l16) * INK_R1_G04_LS + k * 32u + 16u * h);
        const v8i zm = (v8i)(0x4B400000);
        float dx[INK_R1_G04_WJ];
#pragma unroll
        for (u32 j = 0u; j < INK_R1_G04_WJ; j++) dx[j] = lsx[(16u * j + l16) * INK_R1_G04_KB + k];
        v8i c[INK_R1_G04_WJ];
        u32x4 nb4 = h ? (wa[k] >> 4u) & 0x0F0F0F0Fu : wa[k] & 0x0F0F0F0Fu;
        nb4.x = ie_iq4_map(nb4.x);
        nb4.y = ie_iq4_map(nb4.y);
        nb4.z = ie_iq4_map(nb4.z);
        nb4.w = ie_iq4_map(nb4.w);
        const v2i a0 = {(int)nb4.x, (int)nb4.y}, a1 = {(int)nb4.z, (int)nb4.w};
#pragma unroll
        for (u32 j = 0u; j < INK_R1_G04_WJ; j++) {
          const v2i b0 = {(int)b[j].x, (int)b[j].y};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0, zm, 0);
        }
#pragma unroll
        for (u32 j = 0u; j < INK_R1_G04_WJ; j++) {
          const v2i b1 = {(int)b[j].z, (int)b[j].w};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1, c[j], 0);
        }
        const f8 sw = *(const LDSP f8 *)(lsw + k * INK_R1_G04_R + wr + 8u * h);
#pragma unroll
        for (u32 j = 0u; j < INK_R1_G04_WJ; j++)
          acc[j] += (sw * dx[j]) * (__builtin_bit_cast(f8, c[j]) - 12582912.0f);
        __builtin_amdgcn_sched_barrier(0);
      }
      if (more) {
        INK_R1_G04_STORE_X(bf ^ 1u);
        lbarrier();
        bf ^= 1u;
#pragma unroll
        for (u32 k = 0u; k < INK_R1_G04_KB; k++) wa[k] = wn[k];
      }
    }

#pragma unroll
    for (u32 j = 0u; j < INK_R1_G04_WJ; j++) {
      const u32 t = tbase + 16u * j + l16;
      if (t >= T) continue;
      const u32 pid = INK_R1_G04_PID(t);
      G float *yr = (G float *)(Y + (unsigned long)pid * ys);
      const G float *rr = res ? (const G float *)(res + (unsigned long)pid * rs) : 0;
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 r = rbase + wr + 8u * h + v;
        if (r < rows) yr[r] = epilogue(acc[j][v], r, 0, rr);
      }
    }
    if (tbase + INK_R1_G04_T < T) lbarrier();
  }

#undef INK_R1_G04_PID
#undef INK_R1_G04_FETCH_W
#undef INK_R1_G04_FETCH_X
#undef INK_R1_G04_STORE_X
}

#undef INK_R1_G04_WI
#undef INK_R1_G04_T
#undef INK_R1_G04_KB
#undef INK_R1_G04_R
#undef INK_R1_G04_WJ
#undef INK_R1_G04_LS
