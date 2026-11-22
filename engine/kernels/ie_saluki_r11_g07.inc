/* ============================================================================
 * Round 11 Group 7: Bitwise-exact T=8 Multi-Output Raw-Quant + Q8_0 Verify GEMV
 *   Kernel: ie_saluki_gemv_raw_tr8_m_r11
 *
 *   Replaces base ie_gemv_raw_tr8_m at T=8 (1,237.75 ms trim / 1,285.90 ms med
 *   across 14,168 Saluki verify launches — the #1 single unspecialized decode
 *   kernel, covering all 46 non-IQ1_S-pair GDN input projections:
 *     - 10240x5120 + 6144x5120 + Q8_0:96x5120 (11,396 launches, 1,000.31 ms)
 *     - 16384x5120 + Q8_0:96x5120             ( 2,772 launches,   237.44 ms)).
 *
 *   Base ie_gemv_raw_tr8_m bottlenecks fixed:
 *     1. 17,408 B static LDS (u64 tr8_tab[128+2048]) capped occupancy at
 *        6 WGs/WGP and doubled LUT bank conflicts -> unioned 8,192 B LDS
 *        (packed u32 ulut[2048] | float red[8*8*32]) unlocks 8 WGs/WGP
 *        (4 WGs/CU) with 2x fewer LUT bank conflicts and zero ksigns64 LDS.
 *     2. Monolithic R=2 tr8 (16 rows/WG, T=5..9, unhoisted decode, per-lane
 *        VMEM X loads, ~174 VGPRs -> 2-3 WGs/CU) -> 1-row/lane tr9 (32 rows/WG)
 *        with hoisted single-mul ALU sign/code decode, wave-uniform SMEM
 *        s_load X (decoupled kmcnt vs VMEM loadcnt), single-SGPR base `q` +
 *        32-bit VGPR field offsets (pure SAddr global_load, zero 64-bit VGPR
 *        pointer arithmetic), branchless 1-superblock pipe1 hot loop (peeled
 *        last 32-subblock tile -> 0 j=3 s_wait_loadcnt stalls), once-at-entry
 *        segment selection + pre-switch hoisted xt[8] + 2-token s_load batching
 *        (~72-76 VGPRs <= 96, ~68-76 SGPRs <= 80, 0 VGPR/SGPR spills -> 8 WGs/WGP!).
 *     3. Q2_K uses 4-byte-aligned global_load_b128 (4 loads vs 11) + scalar X.
 *     4. IQ1_S / IQ1_M use exact folded 8*g +- 1 weights (zero `as` loads).
 *     5. Q8_0 dispatches directly to base gemv_q8q8_mseg (exact bit-for-bit).
 *
 *   Included at the end of kernels/ie_saluki_kernels.c.
 *   All local helpers/types/macros are prefixed r11_g07_ / R11_G07_; all
 *   macros are #undef'd at EOF.
 *   100% bitwise equal (0 bit diffs) to base ie_gemv_raw_tr8_m.
 * ========================================================================== */

/* 2-token s_load batching barrier: caps simultaneous live s_load_b256 tokens
 * to 2 (16 + 2 = 18 SGPRs vs 36 for 4-tok / 72 for 8-tok) and trims temporary
 * integer-dot VGPRs, guaranteeing SGPR <= 80, sspl = 0, VGPR <= 80, vspl = 0. */
#define R11_G07_PIN2(a0, a1)                                                                      \
  do {                                                                                            \
    __asm__ volatile("" : "+v"(a0), "+v"(a1));                                                    \
    __builtin_amdgcn_sched_barrier(0);                                                            \
  } while (0)

/* 4-byte-aligned u32x4 load helper (emits global_load_b128 on gfx1201 with
 * zero alignment UB for Q2_K's 84-byte superblock stride). */
struct __attribute__((packed, aligned(4))) r11_g07_u32x4_p4 { u32x4 v; };
static inline __attribute__((always_inline)) u32x4 r11_g07_ld_u32x4_a4(const G u8 *p, const u32 o) {
  return ((const G struct r11_g07_u32x4_p4 *)(p + o))->v;
}

/* ============================================================================
 * Shared tr9 scalar-activation 8-token dot helpers (<= 80 SGPRs via pre-switch
 * hoisted xt[8] + 2-tok s_load batching; exact FP32 order, fp contract off)
 * ========================================================================== */

static inline __attribute__((always_inline)) void r11_g07_tr9_dot8_sload(
    const u32 wl[4], const u32 wh[4], const float dw,
    const TR8_CSP u8 *const xt[8], const u32 xo, const u32 dao, float aj[8]) {
#pragma clang fp contract(off)
#pragma unroll
  for (u32 t = 0; t < 8u; t++) {
    const TR8_CSP u8 *xb = xt[t];
    const float dav = *(const TR8_CSP float *)(xb + dao);
    const u32x4 x0 = *(const TR8_CSP u32x4 *)(xb + xo);
    const u32x4 x1 = *(const TR8_CSP u32x4 *)(xb + xo + 16u);
    u32 idot = ie_dot4_ss(0u, wl[0], x0.x);
    idot = ie_dot4_ss(idot, wh[0], x0.y);
    idot = ie_dot4_ss(idot, wl[1], x0.z);
    idot = ie_dot4_ss(idot, wh[1], x0.w);
    idot = ie_dot4_ss(idot, wl[2], x1.x);
    idot = ie_dot4_ss(idot, wh[2], x1.y);
    idot = ie_dot4_ss(idot, wl[3], x1.z);
    idot = ie_dot4_ss(idot, wh[3], x1.w);
    aj[t] += dav * (dw * (float)(int)idot);
    if ((t & 1u) == 1u && t < 7u) R11_G07_PIN2(aj[t - 1u], aj[t]);
  }
}

static inline __attribute__((always_inline)) void r11_g07_tr9_dot8_sload_split16(
    const u32 wl[4], const u32 wh[4], const float dw0, const float dw1,
    const TR8_CSP u8 *const xt[8], const u32 xo, const u32 dao, float aj[8]) {
#pragma clang fp contract(off)
#pragma unroll
  for (u32 t = 0; t < 8u; t++) {
    const TR8_CSP u8 *xb = xt[t];
    const float dav = *(const TR8_CSP float *)(xb + dao);
    const u32x4 x0 = *(const TR8_CSP u32x4 *)(xb + xo);
    const u32x4 x1 = *(const TR8_CSP u32x4 *)(xb + xo + 16u);
    u32 id0 = ie_dot4_ss(0u, wl[0], x0.x);
    id0 = ie_dot4_ss(id0, wh[0], x0.y);
    id0 = ie_dot4_ss(id0, wl[1], x0.z);
    id0 = ie_dot4_ss(id0, wh[1], x0.w);
    u32 id1 = ie_dot4_ss(0u, wl[2], x1.x);
    id1 = ie_dot4_ss(id1, wh[2], x1.y);
    id1 = ie_dot4_ss(id1, wl[3], x1.z);
    id1 = ie_dot4_ss(id1, wh[3], x1.w);
    aj[t] += dav * (dw0 * (float)(int)id0 + dw1 * (float)(int)id1);
    if ((t & 1u) == 1u && t < 7u) R11_G07_PIN2(aj[t - 1u], aj[t]);
  }
}

static inline __attribute__((always_inline)) void r11_g07_tr9_dot8_sload_q2k(
    const u32 d_mn, const u32 sc16, const u32x4 q0_raw, const u32x4 q1_raw,
    const u32 sh, const u32 qmask,
    const TR8_CSP u8 *const xt[8], const u32 xo, const u32 dao, float aj[8]) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d_mn & 0xFFFFu);
  const float mn = ie_rq_h2f(d_mn >> 16u);
  const float dw0 = d * (float)(sc16 & 15u), dm0 = mn * (float)((sc16 >> 4u) & 15u);
  const float dw1 = d * (float)((sc16 >> 8u) & 15u), dm1 = mn * (float)(sc16 >> 12u);
  const u32 q00 = q0_raw.x & qmask, q01 = q0_raw.y & qmask;
  const u32 q02 = q0_raw.z & qmask, q03 = q0_raw.w & qmask;
  const u32 q10 = q1_raw.x & qmask, q11 = q1_raw.y & qmask;
  const u32 q12 = q1_raw.z & qmask, q13 = q1_raw.w & qmask;
#pragma unroll
  for (u32 t = 0; t < 8u; t++) {
    const TR8_CSP u8 *xb = xt[t];
    const float dav = *(const TR8_CSP float *)(xb + dao);
    const u32x4 x0 = *(const TR8_CSP u32x4 *)(xb + xo);
    const u32x4 x1 = *(const TR8_CSP u32x4 *)(xb + xo + 16u);
    u32 id0 = ie_dot4_us(0u, q00, x0.x);
    id0 = ie_dot4_us(id0, q01, x0.y);
    id0 = ie_dot4_us(id0, q02, x0.z);
    id0 = ie_dot4_us(id0, q03, x0.w);
    const u32 xs0 = ie_dot4_ss(ie_dot4_ss(0u, x0.x, 0x01010101u), x0.y, 0x01010101u) +
                    ie_dot4_ss(ie_dot4_ss(0u, x0.z, 0x01010101u), x0.w, 0x01010101u);
    const float t0 = dav * (dw0 * (float)((int)id0 >> sh) - dm0 * (float)(int)xs0);
    u32 id1 = ie_dot4_us(0u, q10, x1.x);
    id1 = ie_dot4_us(id1, q11, x1.y);
    id1 = ie_dot4_us(id1, q12, x1.z);
    id1 = ie_dot4_us(id1, q13, x1.w);
    const u32 xs1 = ie_dot4_ss(ie_dot4_ss(0u, x1.x, 0x01010101u), x1.y, 0x01010101u) +
                    ie_dot4_ss(ie_dot4_ss(0u, x1.z, 0x01010101u), x1.w, 0x01010101u);
    const float t1 = dav * (dw1 * (float)((int)id1 >> sh) - dm1 * (float)(int)xs1);
    aj[t] += (t0 + t1);
    if ((t & 1u) == 1u && t < 7u) R11_G07_PIN2(aj[t - 1u], aj[t]);
  }
}

/* ============================================================================
 * Per-subtype hoisted single-mul decoders (100% bitwise exact, fp contract off)
 * ========================================================================== */

static inline __attribute__((always_inline)) void r11_g07_decode_iq2xxs(
    const LDSP u32 *ulut, const u32 d16, const u32 a0, const u32 a1,
    u32 wl[4], u32 wh[4], float *dw_out) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d16);
  *dw_out = (d * 0.25f) * (0.5f + (float)(a1 >> 28u));
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 cp = ulut[ie_byte(a0, g)];
    TR8_KSW(cp & 0x03030303u, (cp >> 4u) & 0x03030303u, (a1 >> (7u * g)) & 127u, wl[g], wh[g]);
  }
}

static inline __attribute__((always_inline)) void r11_g07_decode_iq2xs(
    const LDSP u32 *ulut, const u32 d16, const u32 sc, const u32 q01, const u32 q23,
    u32 wl[4], u32 wh[4], float *dw0_out, float *dw1_out) {
#pragma clang fp contract(off)
  const float d4 = ie_rq_h2f(d16) * 0.25f;
  *dw0_out = d4 * (0.5f + (float)(sc & 15u));
  *dw1_out = d4 * (0.5f + (float)(sc >> 4u));
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 q2 = ((g < 2u ? q01 : q23) >> (16u * (g & 1u))) & 0xFFFFu;
    const u32 cp = ulut[q2 & 511u];
    TR8_KSW(cp & 0x03030303u, (cp >> 4u) & 0x03030303u, q2 >> 9u, wl[g], wh[g]);
  }
}

static inline __attribute__((always_inline)) void r11_g07_decode_iq2s(
    const LDSP u32 *ulut, const u32 d16, const u32 qlo, const u32 sgn,
    const u32 qh, const u32 sc, u32 wl[4], u32 wh[4], float *dw0_out, float *dw1_out) {
#pragma clang fp contract(off)
  const float d4 = ie_rq_h2f(d16) * 0.25f;
  *dw0_out = d4 * (0.5f + (float)(sc & 15u));
  *dw1_out = d4 * (0.5f + (float)(sc >> 4u));
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 idx = ie_byte(qlo, g) | ((qh << (8u - 2u * g)) & 0x300u);
    const u32 sb = ie_byte(sgn, g);
    const u32 cp = ulut[idx];
    const u32 p_ = (sb & 127u) * 0x00810204u;
    wl[g] = TR8_P2((p_ & 0x04040404u) | (cp & 0x03030303u));
    wh[g] = TR8_P2((((sb << 19u) | (p_ >> 4u)) & 0x04040404u) | ((cp >> 4u) & 0x03030303u));
  }
}

static inline __attribute__((always_inline)) void r11_g07_decode_iq3xxs(
    const LDSP u32 *ulut, const u32 d16, const u32 aux, const u32 q01, const u32 q23,
    u32 wl[4], u32 wh[4], float *dw_out) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d16);
  *dw_out = (d * 0.5f) * (0.5f + (float)(aux >> 28u));
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 qp = ((g < 2u ? q01 : q23) >> (16u * (g & 1u))) & 0xFFFFu;
    const u32 g1 = ulut[qp & 255u], g2 = ulut[qp >> 8u];
    const u32 s7 = (aux >> (7u * g)) & 127u;
    const u32 p_ = s7 * 0x00204081u;
    const u32 t0 = p_ & 0x01010101u;
    const u32 t1 = ((p_ >> 4u) | ((u32)__builtin_popcount(s7) << 24u)) & 0x01010101u;
    const u32 m0 = __builtin_amdgcn_perm(0u, 0u, t0 | 0x0C0C0C0Cu);
    const u32 m1 = __builtin_amdgcn_perm(0u, 0u, t1 | 0x0C0C0C0Cu);
    wl[g] = (g1 ^ m0) + t0;
    wh[g] = (g2 ^ m1) + t1;
  }
}

/* All 512 ie_rq_iq3s_grid entries hold strictly odd bytes ({1,3,5,7,9,11,13,15}),
 * so -g == g ^ 0xFE across all 4 bytes with zero carry. */
static inline __attribute__((always_inline)) void r11_g07_decode_iq3s(
    const LDSP u32 *ulut, const u32 d16, const u32 qh, const u32 sc_b, const u32 sgn,
    const u32 q01, const u32 q23, const u32 sc_sh,
    u32 wl[4], u32 wh[4], float *dw_out) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d16);
  const u32 sc = (sc_b >> sc_sh) & 15u;
  *dw_out = d * (float)(int)(1u + 2u * sc);
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 qp = ((g < 2u ? q01 : q23) >> (16u * (g & 1u))) & 0xFFFFu;
    const u32 i1 = (qp & 255u) | (((qh >> (2u * g)) & 1u) << 8u);
    const u32 i2 = (qp >> 8u) | (((qh >> (2u * g + 1u)) & 1u) << 8u);
    const u32 g1 = ulut[i1], g2 = ulut[i2];
    const u32 sb = ie_byte(sgn, g);
    const u32 p_ = (sb & 127u) * 0x00204081u;
    const u32 t0 = p_ & 0x01010101u;
    const u32 t1 = ((p_ >> 4u) | (sb << 17u)) & 0x01010101u;
    wl[g] = g1 ^ __builtin_amdgcn_perm(0u, 0x0000FE00u, t0);
    wh[g] = g2 ^ __builtin_amdgcn_perm(0u, 0x0000FE00u, t1);
  }
}

/* Folded IQ1_S: weights (8*g +- 1) in {-9,-7,-1,+1,+7,+9} via rq_pack_iq1 +
 * rq_iq1_map with exact power-of-2 scale (d * (2*ls+1)) * 0.125f. */
static inline __attribute__((always_inline)) void r11_g07_decode_iq1s(
    const LDSP u32 *ulut, const u32 d16, const u32 qlo, const u32 qh,
    u32 wl[4], u32 wh[4], float *dw8_out) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d16);
  *dw8_out = (d * (float)(int)(2u * ((qh >> 12u) & 7u) + 1u)) * 0.125f;
  const u32 tbl = (qh & 0x8000u) != 0u ? RQ_IQ1_TBL_NEG : RQ_IQ1_TBL_POS;
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 idx = ie_byte(qlo, g) | (((qh >> (3u * g)) & 7u) << 8u);
    const u32 cp = ulut[idx];
    wl[g] = rq_iq1_map(cp, tbl);
    wh[g] = rq_iq1_map(cp >> 4u, tbl);
  }
}

static inline __attribute__((always_inline)) void r11_g07_decode_iq1m(
    const LDSP u32 *ulut, const u32 slo, const u32 shi, const u32 qhb, const u32 qlo,
    const int use_shi, const u32 sc_sh16, const u32 sh,
    u32 wl[4], u32 wh[4], float *dq0_out, float *dq1_out) {
#pragma clang fp contract(off)
  const u32 s0 = slo & 0xFFFFu, s1 = slo >> 16u, s2 = shi & 0xFFFFu, s3 = shi >> 16u;
  const float d = ie_rq_h2f((s0 >> 12u) | ((s1 >> 8u) & 0x00F0u) | ((s2 >> 4u) & 0x0F00u) | (s3 & 0xF000u));
  const u32 scv = (use_shi ? shi : slo) >> sc_sh16;
  *dq0_out = (d * (float)(int)(2u * ((scv >> sh) & 7u) + 1u)) * 0.125f;
  *dq1_out = (d * (float)(int)(2u * ((scv >> (sh + 3u)) & 7u) + 1u)) * 0.125f;
#pragma unroll
  for (u32 g = 0; g < 4u; g++) {
    const u32 qh = g < 2u ? (qhb & 255u) : (qhb >> 8u);
    const u32 idx = ie_byte(qlo, g) | ((qh << (8u - 4u * (g & 1u))) & 0x700u);
    const u32 cp = ulut[idx];
    const u32 tbl = (qh & ((g & 1u) != 0u ? 0x80u : 0x08u)) != 0u ? RQ_IQ1_TBL_NEG : RQ_IQ1_TBL_POS;
    wl[g] = rq_iq1_map(cp, tbl);
    wh[g] = rq_iq1_map(cp >> 4u, tbl);
  }
}

/* ============================================================================
 * Shared exact 5-level binary reduction & epilogue (matches wave_tree_f)
 * Caller MUST execute lbarrier() before this so unioned sm.ulut reads finish.
 * ========================================================================== */

static inline __attribute__((always_inline)) void r11_g07_tr9_reduce32(
    LDSP float *red, const u32 w, const u32 l, const u32 r0, const u32 rows,
    float a[4][8], const G float *bias, const G float *res, G float *y, const u32 ys, const u32 rs) {
#pragma clang fp contract(off)
#pragma unroll
  for (u32 t = 0; t < 8u; t++) red[(w * 8u + t) * 32u + l] = (a[0][t] + a[2][t]) + (a[1][t] + a[3][t]);
  lbarrier();
  const u32 i = tid();
  const u32 rr = i & 31u, t = i >> 5u, row = r0 + rr;
  float v[8];
#pragma unroll
  for (u32 x = 0; x < 8u; x++) v[x] = red[(x * 8u + t) * 32u + rr];
  const float x0 = v[0] + v[4], x1 = v[1] + v[5], x2 = v[2] + v[6], x3 = v[3] + v[7];
  float out = (x0 + x2) + (x1 + x3);
  if (row < rows) {
    if (bias) out = out + bias[row];
    if (res) out = ((const G float *)((const G u8 *)res + (unsigned long)t * rs))[row] + out;
    ((G float *)((G u8 *)y + (unsigned long)t * ys))[row] = out;
  }
}

/* ============================================================================
 * R11-G07: ie_saluki_gemv_raw_tr8_m_r11
 *   Drop-in 30-arg multi-output T=8 verify GEMV covering all 8 raw types
 *   (32 rows/WG tr9) + Q8_0 (base gemv_q8q8_mseg).
 *   Grid: ng9[k] = (type[k] == Q8_0) ? ie_gemv_ngroups(rows[k]) * 8u
 *                                    : (rows[k] + 31u) / 32u;
 *         g1 = ng9[0], g2 = g1 + ng9[1], tot = g2 + ng9[2] (253..256 -> 257).
 *   Precondition (checked by host dispatch):
 *     T == 8u, nb > 0u, (nb & 7u) == 0u, and for every raw segment k,
 *     (uint64_t)rows[k] * (nb >> 3u) * ie_rawq_bs(type[k]) <= 0x7fffffffULL.
 * ========================================================================== */

KERNEL ie_saluki_gemv_raw_tr8_m_r11(
    const G u8 *xq, u32 nb, u32 T, u32 xs, u32 g1, u32 g2,
    const G u8 *q0, G float *y0, u32 rows0, u32 type0, u32 ys0, const G float *res0, u32 rs0, const G float *bias0,
    const G u8 *q1, G float *y1, u32 rows1, u32 type1, u32 ys1, const G float *res1, u32 rs1, const G float *bias1,
    const G u8 *q2, G float *y2, u32 rows2, u32 type2, u32 ys2, const G float *res2, u32 rs2, const G float *bias2) {
#pragma clang fp contract(off)
  if (T != 8u) return;
  const u32 wid = wgid();
  const u32 s = wid < g1 ? 0u : (wid < g2 ? 1u : 2u);
  const u32 wg = wid - (s == 0u ? 0u : (s == 1u ? g1 : g2));
  const u32 rows = s == 0u ? rows0 : (s == 1u ? rows1 : rows2);
  const u32 type = s == 0u ? type0 : (s == 1u ? type1 : type2);
  if (rows == 0u) return;

  /* Select all active-segment pointers/strides ONCE at entry so all 3-segment
   * kernargs (y0..bias2, s, g1, g2) die immediately (-20 SGPRs across switch). */
  const G u8 *q_raw = s == 0u ? q0 : (s == 1u ? q1 : q2);
  G float *y = s == 0u ? y0 : (s == 1u ? y1 : y2);
  const G float *res = s == 0u ? res0 : (s == 1u ? res1 : res2);
  const G float *bias = s == 0u ? bias0 : (s == 1u ? bias1 : bias2);
  const u32 ys = s == 0u ? ys0 : (s == 1u ? ys1 : ys2);
  const u32 rs = s == 0u ? rs0 : (s == 1u ? rs1 : rs2);

  /* Q8_0 segment (e.g., GDN ssm_beta+alpha): exact base gemv_q8q8_mseg path. */
  if (type == IE_RQ_Q8_0) {
    gemv_q8q8_mseg(wg, q_raw, xq, y, rows, nb, 8u, bias, res, xs, ys, rs);
    return;
  }

  /* Raw quant segments: 32 rows/WG (1 row/lane tr9), 8 KiB unioned LDS. */
  const u32 r0 = 32u * wg;
  if (r0 >= rows) return;

  static LDS union {
    u32 ulut[2048];
    float red[8u * 8u * 32u];
  } sm __attribute__((aligned(16)));

  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 l = lane();
  const u32 r = r0 + l;
  const u32 rr = r < rows ? r : r0;
  const G u8 *q = tr8_uniform(q_raw);
  const u32 xs_s = (u32)__builtin_amdgcn_readfirstlane(xs);
  const TR8_CSP u8 *xt[8];
#pragma unroll
  for (u32 t = 0; t < 8u; t++) xt[t] = (const TR8_CSP u8 *)tr8_uniform(xq + (unsigned long)t * xs_s);
  const u32 nb_s = (u32)__builtin_amdgcn_readfirstlane(nb);
  const u32 row_mb = rr * (nb_s >> 3u);
  const u32 nb32 = nb_s & ~31u;

  float a[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[j][t] = 0.0f;

  u32 xo = 32u * w, dao = 32u * nb_s + 4u * w;

  switch (type) {
    case IE_RQ_Q2_K: {
      const u32 sh = 2u * (w & 3u), qmask = 0x03030303u << sh;
      u32 roff = row_mb * 84u;
      u32 roff_ps = roff + 2u * w;
      u32 roff_pq = roff + (16u + (w >> 2u) * 32u);
      if (nb32 > 0u) {
        u32 d_mn_c = ie_rq_a32(q + (unsigned long)roff, 80u);
        u32 sc16_c = ie_rq_a16(q + (unsigned long)roff_ps, 0u);
        u32x4 q0_c = r11_g07_ld_u32x4_a4(q + (unsigned long)roff_pq, 0u);
        u32x4 q1_c = r11_g07_ld_u32x4_a4(q + (unsigned long)roff_pq, 16u);
        __asm__ volatile("" : "+v"(d_mn_c), "+v"(sc16_c), "+v"(q0_c), "+v"(q1_c));
        for (u32 k0 = 0; k0 + 32u < nb32; k0 += 32u) {
#pragma unroll
          for (u32 j = 0; j < 4u; j++) {
            const u32 noff = 84u * (j + 1u);
            const u32 d_mn_n = ie_rq_a32(q + (unsigned long)roff, noff + 80u);
            const u32 sc16_n = ie_rq_a16(q + (unsigned long)roff_ps, noff);
            const u32x4 q0_n = r11_g07_ld_u32x4_a4(q + (unsigned long)roff_pq, noff);
            const u32x4 q1_n = r11_g07_ld_u32x4_a4(q + (unsigned long)roff_pq, noff + 16u);
            __builtin_amdgcn_sched_barrier(0);
            r11_g07_tr9_dot8_sload_q2k(d_mn_c, sc16_c, q0_c, q1_c, sh, qmask,
                                       xt, xo + 256u * j, dao + 32u * j, a[j]);
            __builtin_amdgcn_sched_barrier(0);
            d_mn_c = d_mn_n; sc16_c = sc16_n; q0_c = q0_n; q1_c = q1_n;
          }
          roff += 336u; roff_ps += 336u; roff_pq += 336u; xo += 1024u; dao += 128u;
        }
#pragma unroll
        for (u32 j = 0; j < 3u; j++) {
          const u32 noff = 84u * (j + 1u);
          const u32 d_mn_n = ie_rq_a32(q + (unsigned long)roff, noff + 80u);
          const u32 sc16_n = ie_rq_a16(q + (unsigned long)roff_ps, noff);
          const u32x4 q0_n = r11_g07_ld_u32x4_a4(q + (unsigned long)roff_pq, noff);
          const u32x4 q1_n = r11_g07_ld_u32x4_a4(q + (unsigned long)roff_pq, noff + 16u);
          __builtin_amdgcn_sched_barrier(0);
          r11_g07_tr9_dot8_sload_q2k(d_mn_c, sc16_c, q0_c, q1_c, sh, qmask,
                                     xt, xo + 256u * j, dao + 32u * j, a[j]);
          __builtin_amdgcn_sched_barrier(0);
          d_mn_c = d_mn_n; sc16_c = sc16_n; q0_c = q0_n; q1_c = q1_n;
        }
        r11_g07_tr9_dot8_sload_q2k(d_mn_c, sc16_c, q0_c, q1_c, sh, qmask,
                                   xt, xo + 768u, dao + 96u, a[3]);
        roff += 336u; roff_ps += 336u; roff_pq += 336u; xo += 1024u; dao += 128u;
      }
      if (nb32 < nb_s) {
#pragma nounroll
        for (u32 j = 0; j < 4u; j++) {
          if (nb32 + w + 8u * j < nb_s) {
            const u32 off = 84u * j;
            r11_g07_tr9_dot8_sload_q2k(
                ie_rq_a32(q + (unsigned long)roff, off + 80u),
                ie_rq_a16(q + (unsigned long)roff_ps, off),
                r11_g07_ld_u32x4_a4(q + (unsigned long)roff_pq, off),
                r11_g07_ld_u32x4_a4(q + (unsigned long)roff_pq, off + 16u),
                sh, qmask, xt, xo + 256u * j, dao + 32u * j, a[j]);
          }
        }
      }
      break;
    }

    case IE_RQ_IQ2_XXS: {
      sm.ulut[tid()] = rq_pack_iq2(ie_rq_iq2xxs_grid[tid()]);
      lbarrier();
      u32 roff = row_mb * 66u;
      u32 roff_q = roff + (2u + 8u * w);
      if (nb32 > 0u) {
        u32 d_c = ie_rq_a16(q + (unsigned long)roff, 0u);
        u32 a0_c = ie_rq_a32(q + (unsigned long)roff_q, 0u);
        u32 a1_c = ie_rq_a32(q + (unsigned long)roff_q, 4u);
        __asm__ volatile("" : "+v"(d_c), "+v"(a0_c), "+v"(a1_c));
        for (u32 k0 = 0; k0 + 32u < nb32; k0 += 32u) {
#pragma unroll
          for (u32 j = 0; j < 4u; j++) {
            const u32 noff = 66u * (j + 1u);
            const u32 d_n = ie_rq_a16(q + (unsigned long)roff, noff);
            const u32 a0_n = ie_rq_a32(q + (unsigned long)roff_q, noff);
            const u32 a1_n = ie_rq_a32(q + (unsigned long)roff_q, noff + 4u);
            __builtin_amdgcn_sched_barrier(0);
            u32 wl[4], wh[4]; float dw;
            r11_g07_decode_iq2xxs(sm.ulut, d_c, a0_c, a1_c, wl, wh, &dw);
            r11_g07_tr9_dot8_sload(wl, wh, dw, xt, xo + 256u * j, dao + 32u * j, a[j]);
            __builtin_amdgcn_sched_barrier(0);
            d_c = d_n; a0_c = a0_n; a1_c = a1_n;
          }
          roff += 264u; roff_q += 264u; xo += 1024u; dao += 128u;
        }
#pragma unroll
        for (u32 j = 0; j < 3u; j++) {
          const u32 noff = 66u * (j + 1u);
          const u32 d_n = ie_rq_a16(q + (unsigned long)roff, noff);
          const u32 a0_n = ie_rq_a32(q + (unsigned long)roff_q, noff);
          const u32 a1_n = ie_rq_a32(q + (unsigned long)roff_q, noff + 4u);
          __builtin_amdgcn_sched_barrier(0);
          u32 wl[4], wh[4]; float dw;
          r11_g07_decode_iq2xxs(sm.ulut, d_c, a0_c, a1_c, wl, wh, &dw);
          r11_g07_tr9_dot8_sload(wl, wh, dw, xt, xo + 256u * j, dao + 32u * j, a[j]);
          __builtin_amdgcn_sched_barrier(0);
          d_c = d_n; a0_c = a0_n; a1_c = a1_n;
        }
        {
          u32 wl[4], wh[4]; float dw;
          r11_g07_decode_iq2xxs(sm.ulut, d_c, a0_c, a1_c, wl, wh, &dw);
          r11_g07_tr9_dot8_sload(wl, wh, dw, xt, xo + 768u, dao + 96u, a[3]);
        }
        roff += 264u; roff_q += 264u; xo += 1024u; dao += 128u;
      }
      if (nb32 < nb_s) {
#pragma nounroll
        for (u32 j = 0; j < 4u; j++) {
          if (nb32 + w + 8u * j < nb_s) {
            const u32 off = 66u * j;
            u32 wl[4], wh[4]; float dw;
            r11_g07_decode_iq2xxs(sm.ulut,
                                  ie_rq_a16(q + (unsigned long)roff, off),
                                  ie_rq_a32(q + (unsigned long)roff_q, off),
                                  ie_rq_a32(q + (unsigned long)roff_q, off + 4u),
                                  wl, wh, &dw);
            r11_g07_tr9_dot8_sload(wl, wh, dw, xt, xo + 256u * j, dao + 32u * j, a[j]);
          }
        }
      }
      break;
    }

    case IE_RQ_IQ2_XS: {
      sm.ulut[tid()] = rq_pack_iq2(ie_rq_iq2xs_grid[tid()]);
      sm.ulut[tid() + 256u] = rq_pack_iq2(ie_rq_iq2xs_grid[tid() + 256u]);
      lbarrier();
      u32 roff = row_mb * 74u;
      u32 roff_q = roff + (2u + 8u * w);
      u32 roff_sc = roff + (66u + w);
      if (nb32 > 0u) {
        u32 d_c = ie_rq_a16(q + (unsigned long)roff, 0u);
        u32 sc_c = (u32)(q + (unsigned long)roff_sc)[0];
        u32 q01_c = ie_rq_a32(q + (unsigned long)roff_q, 0u);
        u32 q23_c = ie_rq_a32(q + (unsigned long)roff_q, 4u);
        __asm__ volatile("" : "+v"(d_c), "+v"(sc_c), "+v"(q01_c), "+v"(q23_c));
        for (u32 k0 = 0; k0 + 32u < nb32; k0 += 32u) {
#pragma unroll
          for (u32 j = 0; j < 4u; j++) {
            const u32 noff = 74u * (j + 1u);
            const u32 d_n = ie_rq_a16(q + (unsigned long)roff, noff);
            const u32 sc_n = (u32)(q + (unsigned long)roff_sc)[noff];
            const u32 q01_n = ie_rq_a32(q + (unsigned long)roff_q, noff);
            const u32 q23_n = ie_rq_a32(q + (unsigned long)roff_q, noff + 4u);
            __builtin_amdgcn_sched_barrier(0);
            u32 wl[4], wh[4]; float dw0, dw1;
            r11_g07_decode_iq2xs(sm.ulut, d_c, sc_c, q01_c, q23_c, wl, wh, &dw0, &dw1);
            r11_g07_tr9_dot8_sload_split16(wl, wh, dw0, dw1, xt, xo + 256u * j, dao + 32u * j, a[j]);
            __builtin_amdgcn_sched_barrier(0);
            d_c = d_n; sc_c = sc_n; q01_c = q01_n; q23_c = q23_n;
          }
          roff += 296u; roff_q += 296u; roff_sc += 296u; xo += 1024u; dao += 128u;
        }
#pragma unroll
        for (u32 j = 0; j < 3u; j++) {
          const u32 noff = 74u * (j + 1u);
          const u32 d_n = ie_rq_a16(q + (unsigned long)roff, noff);
          const u32 sc_n = (u32)(q + (unsigned long)roff_sc)[noff];
          const u32 q01_n = ie_rq_a32(q + (unsigned long)roff_q, noff);
          const u32 q23_n = ie_rq_a32(q + (unsigned long)roff_q, noff + 4u);
          __builtin_amdgcn_sched_barrier(0);
          u32 wl[4], wh[4]; float dw0, dw1;
          r11_g07_decode_iq2xs(sm.ulut, d_c, sc_c, q01_c, q23_c, wl, wh, &dw0, &dw1);
          r11_g07_tr9_dot8_sload_split16(wl, wh, dw0, dw1, xt, xo + 256u * j, dao + 32u * j, a[j]);
          __builtin_amdgcn_sched_barrier(0);
          d_c = d_n; sc_c = sc_n; q01_c = q01_n; q23_c = q23_n;
        }
        {
          u32 wl[4], wh[4]; float dw0, dw1;
          r11_g07_decode_iq2xs(sm.ulut, d_c, sc_c, q01_c, q23_c, wl, wh, &dw0, &dw1);
          r11_g07_tr9_dot8_sload_split16(wl, wh, dw0, dw1, xt, xo + 768u, dao + 96u, a[3]);
        }
        roff += 296u; roff_q += 296u; roff_sc += 296u; xo += 1024u; dao += 128u;
      }
      if (nb32 < nb_s) {
#pragma nounroll
        for (u32 j = 0; j < 4u; j++) {
          if (nb32 + w + 8u * j < nb_s) {
            const u32 off = 74u * j;
            u32 wl[4], wh[4]; float dw0, dw1;
            r11_g07_decode_iq2xs(sm.ulut,
                                 ie_rq_a16(q + (unsigned long)roff, off),
                                 (u32)(q + (unsigned long)roff_sc)[off],
                                 ie_rq_a32(q + (unsigned long)roff_q, off),
                                 ie_rq_a32(q + (unsigned long)roff_q, off + 4u),
                                 wl, wh, &dw0, &dw1);
            r11_g07_tr9_dot8_sload_split16(wl, wh, dw0, dw1, xt, xo + 256u * j, dao + 32u * j, a[j]);
          }
        }
      }
      break;
    }

    case IE_RQ_IQ2_S: {
      for (u32 i = tid(); i < 1024u; i += NT) sm.ulut[i] = rq_pack_iq2(ie_rq_iq2s_grid[i]);
      lbarrier();
      u32 roff = row_mb * 82u;
      u32 roff_qlo = roff + (2u + 4u * w);
      u32 roff_qh = roff + (66u + w);
      if (nb32 > 0u) {
        u32 d_c = ie_rq_a16(q + (unsigned long)roff, 0u);
        u32 qlo_c = ie_rq_a32(q + (unsigned long)roff_qlo, 0u);
        u32 sgn_c = ie_rq_a32(q + (unsigned long)roff_qlo, 32u);
        u32 qh_c = (u32)(q + (unsigned long)roff_qh)[0];
        u32 sc_c = (u32)(q + (unsigned long)roff_qh)[8];
        __asm__ volatile("" : "+v"(d_c), "+v"(qlo_c), "+v"(sgn_c), "+v"(qh_c), "+v"(sc_c));
        for (u32 k0 = 0; k0 + 32u < nb32; k0 += 32u) {
#pragma unroll
          for (u32 j = 0; j < 4u; j++) {
            const u32 noff = 82u * (j + 1u);
            const u32 d_n = ie_rq_a16(q + (unsigned long)roff, noff);
            const u32 qlo_n = ie_rq_a32(q + (unsigned long)roff_qlo, noff);
            const u32 sgn_n = ie_rq_a32(q + (unsigned long)roff_qlo, noff + 32u);
            const u32 qh_n = (u32)(q + (unsigned long)roff_qh)[noff];
            const u32 sc_n = (u32)(q + (unsigned long)roff_qh)[noff + 8u];
            __builtin_amdgcn_sched_barrier(0);
            u32 wl[4], wh[4]; float dw0, dw1;
            r11_g07_decode_iq2s(sm.ulut, d_c, qlo_c, sgn_c, qh_c, sc_c, wl, wh, &dw0, &dw1);
            r11_g07_tr9_dot8_sload_split16(wl, wh, dw0, dw1, xt, xo + 256u * j, dao + 32u * j, a[j]);
            __builtin_amdgcn_sched_barrier(0);
            d_c = d_n; qlo_c = qlo_n; sgn_c = sgn_n; qh_c = qh_n; sc_c = sc_n;
          }
          roff += 328u; roff_qlo += 328u; roff_qh += 328u; xo += 1024u; dao += 128u;
        }
#pragma unroll
        for (u32 j = 0; j < 3u; j++) {
          const u32 noff = 82u * (j + 1u);
          const u32 d_n = ie_rq_a16(q + (unsigned long)roff, noff);
          const u32 qlo_n = ie_rq_a32(q + (unsigned long)roff_qlo, noff);
          const u32 sgn_n = ie_rq_a32(q + (unsigned long)roff_qlo, noff + 32u);
          const u32 qh_n = (u32)(q + (unsigned long)roff_qh)[noff];
          const u32 sc_n = (u32)(q + (unsigned long)roff_qh)[noff + 8u];
          __builtin_amdgcn_sched_barrier(0);
          u32 wl[4], wh[4]; float dw0, dw1;
          r11_g07_decode_iq2s(sm.ulut, d_c, qlo_c, sgn_c, qh_c, sc_c, wl, wh, &dw0, &dw1);
          r11_g07_tr9_dot8_sload_split16(wl, wh, dw0, dw1, xt, xo + 256u * j, dao + 32u * j, a[j]);
          __builtin_amdgcn_sched_barrier(0);
          d_c = d_n; qlo_c = qlo_n; sgn_c = sgn_n; qh_c = qh_n; sc_c = sc_n;
        }
        {
          u32 wl[4], wh[4]; float dw0, dw1;
          r11_g07_decode_iq2s(sm.ulut, d_c, qlo_c, sgn_c, qh_c, sc_c, wl, wh, &dw0, &dw1);
          r11_g07_tr9_dot8_sload_split16(wl, wh, dw0, dw1, xt, xo + 768u, dao + 96u, a[3]);
        }
        roff += 328u; roff_qlo += 328u; roff_qh += 328u; xo += 1024u; dao += 128u;
      }
      if (nb32 < nb_s) {
#pragma nounroll
        for (u32 j = 0; j < 4u; j++) {
          if (nb32 + w + 8u * j < nb_s) {
            const u32 off = 82u * j;
            u32 wl[4], wh[4]; float dw0, dw1;
            r11_g07_decode_iq2s(sm.ulut,
                                ie_rq_a16(q + (unsigned long)roff, off),
                                ie_rq_a32(q + (unsigned long)roff_qlo, off),
                                ie_rq_a32(q + (unsigned long)roff_qlo, off + 32u),
                                (u32)(q + (unsigned long)roff_qh)[off],
                                (u32)(q + (unsigned long)roff_qh)[off + 8u],
                                wl, wh, &dw0, &dw1);
            r11_g07_tr9_dot8_sload_split16(wl, wh, dw0, dw1, xt, xo + 256u * j, dao + 32u * j, a[j]);
          }
        }
      }
      break;
    }

    case IE_RQ_IQ3_XXS: {
      sm.ulut[tid()] = ie_rq_iq3xxs_grid[tid()];
      lbarrier();
      u32 roff = row_mb * 98u;
      u32 roff_q = roff + (2u + 8u * w);
      u32 roff_aux = roff + (66u + 4u * w);
      if (nb32 > 0u) {
        u32 d_c = ie_rq_a16(q + (unsigned long)roff, 0u);
        u32 aux_c = ie_rq_a32(q + (unsigned long)roff_aux, 0u);
        u32 q01_c = ie_rq_a32(q + (unsigned long)roff_q, 0u);
        u32 q23_c = ie_rq_a32(q + (unsigned long)roff_q, 4u);
        __asm__ volatile("" : "+v"(d_c), "+v"(aux_c), "+v"(q01_c), "+v"(q23_c));
        for (u32 k0 = 0; k0 + 32u < nb32; k0 += 32u) {
#pragma unroll
          for (u32 j = 0; j < 4u; j++) {
            const u32 noff = 98u * (j + 1u);
            const u32 d_n = ie_rq_a16(q + (unsigned long)roff, noff);
            const u32 aux_n = ie_rq_a32(q + (unsigned long)roff_aux, noff);
            const u32 q01_n = ie_rq_a32(q + (unsigned long)roff_q, noff);
            const u32 q23_n = ie_rq_a32(q + (unsigned long)roff_q, noff + 4u);
            __builtin_amdgcn_sched_barrier(0);
            u32 wl[4], wh[4]; float dw;
            r11_g07_decode_iq3xxs(sm.ulut, d_c, aux_c, q01_c, q23_c, wl, wh, &dw);
            r11_g07_tr9_dot8_sload(wl, wh, dw, xt, xo + 256u * j, dao + 32u * j, a[j]);
            __builtin_amdgcn_sched_barrier(0);
            d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
          }
          roff += 392u; roff_q += 392u; roff_aux += 392u; xo += 1024u; dao += 128u;
        }
#pragma unroll
        for (u32 j = 0; j < 3u; j++) {
          const u32 noff = 98u * (j + 1u);
          const u32 d_n = ie_rq_a16(q + (unsigned long)roff, noff);
          const u32 aux_n = ie_rq_a32(q + (unsigned long)roff_aux, noff);
          const u32 q01_n = ie_rq_a32(q + (unsigned long)roff_q, noff);
          const u32 q23_n = ie_rq_a32(q + (unsigned long)roff_q, noff + 4u);
          __builtin_amdgcn_sched_barrier(0);
          u32 wl[4], wh[4]; float dw;
          r11_g07_decode_iq3xxs(sm.ulut, d_c, aux_c, q01_c, q23_c, wl, wh, &dw);
          r11_g07_tr9_dot8_sload(wl, wh, dw, xt, xo + 256u * j, dao + 32u * j, a[j]);
          __builtin_amdgcn_sched_barrier(0);
          d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
        }
        {
          u32 wl[4], wh[4]; float dw;
          r11_g07_decode_iq3xxs(sm.ulut, d_c, aux_c, q01_c, q23_c, wl, wh, &dw);
          r11_g07_tr9_dot8_sload(wl, wh, dw, xt, xo + 768u, dao + 96u, a[3]);
        }
        roff += 392u; roff_q += 392u; roff_aux += 392u; xo += 1024u; dao += 128u;
      }
      if (nb32 < nb_s) {
#pragma nounroll
        for (u32 j = 0; j < 4u; j++) {
          if (nb32 + w + 8u * j < nb_s) {
            const u32 off = 98u * j;
            u32 wl[4], wh[4]; float dw;
            r11_g07_decode_iq3xxs(sm.ulut,
                                  ie_rq_a16(q + (unsigned long)roff, off),
                                  ie_rq_a32(q + (unsigned long)roff_aux, off),
                                  ie_rq_a32(q + (unsigned long)roff_q, off),
                                  ie_rq_a32(q + (unsigned long)roff_q, off + 4u),
                                  wl, wh, &dw);
            r11_g07_tr9_dot8_sload(wl, wh, dw, xt, xo + 256u * j, dao + 32u * j, a[j]);
          }
        }
      }
      break;
    }

    case IE_RQ_IQ3_S: {
      sm.ulut[tid()] = ie_rq_iq3s_grid[tid()];
      sm.ulut[tid() + 256u] = ie_rq_iq3s_grid[tid() + 256u];
      lbarrier();
      const u32 sc_sh = 4u * (w & 1u);
      u32 roff = row_mb * 110u;
      u32 roff_q = roff + (2u + 8u * w);
      u32 roff_qh = roff + (66u + w);
      u32 roff_sgn = roff + (74u + 4u * w);
      u32 roff_sc = roff + (106u + (w >> 1u));
      if (nb32 > 0u) {
        u32 d_c = ie_rq_a16(q + (unsigned long)roff, 0u);
        u32 qh_c = (u32)(q + (unsigned long)roff_qh)[0];
        u32 sc_c = (u32)(q + (unsigned long)roff_sc)[0];
        u32 sgn_c = ie_rq_a32(q + (unsigned long)roff_sgn, 0u);
        u32 q01_c = ie_rq_a32(q + (unsigned long)roff_q, 0u);
        u32 q23_c = ie_rq_a32(q + (unsigned long)roff_q, 4u);
        __asm__ volatile("" : "+v"(d_c), "+v"(qh_c), "+v"(sc_c), "+v"(sgn_c), "+v"(q01_c), "+v"(q23_c));
        for (u32 k0 = 0; k0 + 32u < nb32; k0 += 32u) {
#pragma unroll
          for (u32 j = 0; j < 4u; j++) {
            const u32 noff = 110u * (j + 1u);
            const u32 d_n = ie_rq_a16(q + (unsigned long)roff, noff);
            const u32 qh_n = (u32)(q + (unsigned long)roff_qh)[noff];
            const u32 sc_n = (u32)(q + (unsigned long)roff_sc)[noff];
            const u32 sgn_n = ie_rq_a32(q + (unsigned long)roff_sgn, noff);
            const u32 q01_n = ie_rq_a32(q + (unsigned long)roff_q, noff);
            const u32 q23_n = ie_rq_a32(q + (unsigned long)roff_q, noff + 4u);
            __builtin_amdgcn_sched_barrier(0);
            u32 wl[4], wh[4]; float dw;
            r11_g07_decode_iq3s(sm.ulut, d_c, qh_c, sc_c, sgn_c, q01_c, q23_c, sc_sh, wl, wh, &dw);
            r11_g07_tr9_dot8_sload(wl, wh, dw, xt, xo + 256u * j, dao + 32u * j, a[j]);
            __builtin_amdgcn_sched_barrier(0);
            d_c = d_n; qh_c = qh_n; sc_c = sc_n; sgn_c = sgn_n; q01_c = q01_n; q23_c = q23_n;
          }
          roff += 440u; roff_q += 440u; roff_qh += 440u; roff_sgn += 440u; roff_sc += 440u;
          xo += 1024u; dao += 128u;
        }
#pragma unroll
        for (u32 j = 0; j < 3u; j++) {
          const u32 noff = 110u * (j + 1u);
          const u32 d_n = ie_rq_a16(q + (unsigned long)roff, noff);
          const u32 qh_n = (u32)(q + (unsigned long)roff_qh)[noff];
          const u32 sc_n = (u32)(q + (unsigned long)roff_sc)[noff];
          const u32 sgn_n = ie_rq_a32(q + (unsigned long)roff_sgn, noff);
          const u32 q01_n = ie_rq_a32(q + (unsigned long)roff_q, noff);
          const u32 q23_n = ie_rq_a32(q + (unsigned long)roff_q, noff + 4u);
          __builtin_amdgcn_sched_barrier(0);
          u32 wl[4], wh[4]; float dw;
          r11_g07_decode_iq3s(sm.ulut, d_c, qh_c, sc_c, sgn_c, q01_c, q23_c, sc_sh, wl, wh, &dw);
          r11_g07_tr9_dot8_sload(wl, wh, dw, xt, xo + 256u * j, dao + 32u * j, a[j]);
          __builtin_amdgcn_sched_barrier(0);
          d_c = d_n; qh_c = qh_n; sc_c = sc_n; sgn_c = sgn_n; q01_c = q01_n; q23_c = q23_n;
        }
        {
          u32 wl[4], wh[4]; float dw;
          r11_g07_decode_iq3s(sm.ulut, d_c, qh_c, sc_c, sgn_c, q01_c, q23_c, sc_sh, wl, wh, &dw);
          r11_g07_tr9_dot8_sload(wl, wh, dw, xt, xo + 768u, dao + 96u, a[3]);
        }
        roff += 440u; roff_q += 440u; roff_qh += 440u; roff_sgn += 440u; roff_sc += 440u;
        xo += 1024u; dao += 128u;
      }
      if (nb32 < nb_s) {
#pragma nounroll
        for (u32 j = 0; j < 4u; j++) {
          if (nb32 + w + 8u * j < nb_s) {
            const u32 off = 110u * j;
            u32 wl[4], wh[4]; float dw;
            r11_g07_decode_iq3s(sm.ulut,
                                ie_rq_a16(q + (unsigned long)roff, off),
                                (u32)(q + (unsigned long)roff_qh)[off],
                                (u32)(q + (unsigned long)roff_sc)[off],
                                ie_rq_a32(q + (unsigned long)roff_sgn, off),
                                ie_rq_a32(q + (unsigned long)roff_q, off),
                                ie_rq_a32(q + (unsigned long)roff_q, off + 4u),
                                sc_sh, wl, wh, &dw);
            r11_g07_tr9_dot8_sload(wl, wh, dw, xt, xo + 256u * j, dao + 32u * j, a[j]);
          }
        }
      }
      break;
    }

    case IE_RQ_IQ1_S: {
      for (u32 i = tid(); i < 2048u; i += NT) sm.ulut[i] = rq_pack_iq1(ie_rq_iq1s_grid[i]);
      lbarrier();
      u32 roff = row_mb * 50u;
      u32 roff_qlo = roff + (2u + 4u * w);
      u32 roff_qh = roff + (34u + 2u * w);
      if (nb32 > 0u) {
        u32 d_c = ie_rq_a16(q + (unsigned long)roff, 0u);
        u32 qlo_c = ie_rq_a32(q + (unsigned long)roff_qlo, 0u);
        u32 qh_c = ie_rq_a16(q + (unsigned long)roff_qh, 0u);
        __asm__ volatile("" : "+v"(d_c), "+v"(qlo_c), "+v"(qh_c));
        for (u32 k0 = 0; k0 + 32u < nb32; k0 += 32u) {
#pragma unroll
          for (u32 j = 0; j < 4u; j++) {
            const u32 noff = 50u * (j + 1u);
            const u32 d_n = ie_rq_a16(q + (unsigned long)roff, noff);
            const u32 qlo_n = ie_rq_a32(q + (unsigned long)roff_qlo, noff);
            const u32 qh_n = ie_rq_a16(q + (unsigned long)roff_qh, noff);
            __builtin_amdgcn_sched_barrier(0);
            u32 wl[4], wh[4]; float dw8;
            r11_g07_decode_iq1s(sm.ulut, d_c, qlo_c, qh_c, wl, wh, &dw8);
            r11_g07_tr9_dot8_sload(wl, wh, dw8, xt, xo + 256u * j, dao + 32u * j, a[j]);
            __builtin_amdgcn_sched_barrier(0);
            d_c = d_n; qlo_c = qlo_n; qh_c = qh_n;
          }
          roff += 200u; roff_qlo += 200u; roff_qh += 200u; xo += 1024u; dao += 128u;
        }
#pragma unroll
        for (u32 j = 0; j < 3u; j++) {
          const u32 noff = 50u * (j + 1u);
          const u32 d_n = ie_rq_a16(q + (unsigned long)roff, noff);
          const u32 qlo_n = ie_rq_a32(q + (unsigned long)roff_qlo, noff);
          const u32 qh_n = ie_rq_a16(q + (unsigned long)roff_qh, noff);
          __builtin_amdgcn_sched_barrier(0);
          u32 wl[4], wh[4]; float dw8;
          r11_g07_decode_iq1s(sm.ulut, d_c, qlo_c, qh_c, wl, wh, &dw8);
          r11_g07_tr9_dot8_sload(wl, wh, dw8, xt, xo + 256u * j, dao + 32u * j, a[j]);
          __builtin_amdgcn_sched_barrier(0);
          d_c = d_n; qlo_c = qlo_n; qh_c = qh_n;
        }
        {
          u32 wl[4], wh[4]; float dw8;
          r11_g07_decode_iq1s(sm.ulut, d_c, qlo_c, qh_c, wl, wh, &dw8);
          r11_g07_tr9_dot8_sload(wl, wh, dw8, xt, xo + 768u, dao + 96u, a[3]);
        }
        roff += 200u; roff_qlo += 200u; roff_qh += 200u; xo += 1024u; dao += 128u;
      }
      if (nb32 < nb_s) {
#pragma nounroll
        for (u32 j = 0; j < 4u; j++) {
          if (nb32 + w + 8u * j < nb_s) {
            const u32 off = 50u * j;
            u32 wl[4], wh[4]; float dw8;
            r11_g07_decode_iq1s(sm.ulut,
                                ie_rq_a16(q + (unsigned long)roff, off),
                                ie_rq_a32(q + (unsigned long)roff_qlo, off),
                                ie_rq_a16(q + (unsigned long)roff_qh, off),
                                wl, wh, &dw8);
            r11_g07_tr9_dot8_sload(wl, wh, dw8, xt, xo + 256u * j, dao + 32u * j, a[j]);
          }
        }
      }
      break;
    }

    case IE_RQ_IQ1_M: {
      for (u32 i = tid(); i < 2048u; i += NT) sm.ulut[i] = rq_pack_iq1(ie_rq_iq1s_grid[i]);
      lbarrier();
      const int use_shi = (w & 4u) != 0u;
      const u32 sc_sh16 = 16u * ((w >> 1u) & 1u);
      const u32 sh = 6u * (w & 1u);
      u32 roff = row_mb * 56u;
      u32 roff_qlo = roff + 4u * w;
      u32 roff_qhb = roff + (32u + 2u * w);
      if (nb32 > 0u) {
        u32 slo_c = ie_rq_a32(q + (unsigned long)roff, 48u);
        u32 shi_c = ie_rq_a32(q + (unsigned long)roff, 52u);
        u32 qlo_c = ie_rq_a32(q + (unsigned long)roff_qlo, 0u);
        u32 qhb_c = ie_rq_a16(q + (unsigned long)roff_qhb, 0u);
        __asm__ volatile("" : "+v"(slo_c), "+v"(shi_c), "+v"(qlo_c), "+v"(qhb_c));
        for (u32 k0 = 0; k0 + 32u < nb32; k0 += 32u) {
#pragma unroll
          for (u32 j = 0; j < 4u; j++) {
            const u32 noff = 56u * (j + 1u);
            const u32 slo_n = ie_rq_a32(q + (unsigned long)roff, noff + 48u);
            const u32 shi_n = ie_rq_a32(q + (unsigned long)roff, noff + 52u);
            const u32 qlo_n = ie_rq_a32(q + (unsigned long)roff_qlo, noff);
            const u32 qhb_n = ie_rq_a16(q + (unsigned long)roff_qhb, noff);
            __builtin_amdgcn_sched_barrier(0);
            u32 wl[4], wh[4]; float dq0, dq1;
            r11_g07_decode_iq1m(sm.ulut, slo_c, shi_c, qhb_c, qlo_c, use_shi, sc_sh16, sh, wl, wh, &dq0, &dq1);
            r11_g07_tr9_dot8_sload_split16(wl, wh, dq0, dq1, xt, xo + 256u * j, dao + 32u * j, a[j]);
            __builtin_amdgcn_sched_barrier(0);
            slo_c = slo_n; shi_c = shi_n; qlo_c = qlo_n; qhb_c = qhb_n;
          }
          roff += 224u; roff_qlo += 224u; roff_qhb += 224u; xo += 1024u; dao += 128u;
        }
#pragma unroll
        for (u32 j = 0; j < 3u; j++) {
          const u32 noff = 56u * (j + 1u);
          const u32 slo_n = ie_rq_a32(q + (unsigned long)roff, noff + 48u);
          const u32 shi_n = ie_rq_a32(q + (unsigned long)roff, noff + 52u);
          const u32 qlo_n = ie_rq_a32(q + (unsigned long)roff_qlo, noff);
          const u32 qhb_n = ie_rq_a16(q + (unsigned long)roff_qhb, noff);
          __builtin_amdgcn_sched_barrier(0);
          u32 wl[4], wh[4]; float dq0, dq1;
          r11_g07_decode_iq1m(sm.ulut, slo_c, shi_c, qhb_c, qlo_c, use_shi, sc_sh16, sh, wl, wh, &dq0, &dq1);
          r11_g07_tr9_dot8_sload_split16(wl, wh, dq0, dq1, xt, xo + 256u * j, dao + 32u * j, a[j]);
          __builtin_amdgcn_sched_barrier(0);
          slo_c = slo_n; shi_c = shi_n; qlo_c = qlo_n; qhb_c = qhb_n;
        }
        {
          u32 wl[4], wh[4]; float dq0, dq1;
          r11_g07_decode_iq1m(sm.ulut, slo_c, shi_c, qhb_c, qlo_c, use_shi, sc_sh16, sh, wl, wh, &dq0, &dq1);
          r11_g07_tr9_dot8_sload_split16(wl, wh, dq0, dq1, xt, xo + 768u, dao + 96u, a[3]);
        }
        roff += 224u; roff_qlo += 224u; roff_qhb += 224u; xo += 1024u; dao += 128u;
      }
      if (nb32 < nb_s) {
#pragma nounroll
        for (u32 j = 0; j < 4u; j++) {
          if (nb32 + w + 8u * j < nb_s) {
            const u32 off = 56u * j;
            u32 wl[4], wh[4]; float dq0, dq1;
            r11_g07_decode_iq1m(sm.ulut,
                                ie_rq_a32(q + (unsigned long)roff, off + 48u),
                                ie_rq_a32(q + (unsigned long)roff, off + 52u),
                                ie_rq_a16(q + (unsigned long)roff_qhb, off),
                                ie_rq_a32(q + (unsigned long)roff_qlo, off),
                                use_shi, sc_sh16, sh, wl, wh, &dq0, &dq1);
            r11_g07_tr9_dot8_sload_split16(wl, wh, dq0, dq1, xt, xo + 256u * j, dao + 32u * j, a[j]);
          }
        }
      }
      break;
    }

    default:
      return;
  }

  /* Wait for all waves to finish reading unioned sm.ulut before sm.red write. */
  lbarrier();
  r11_g07_tr9_reduce32(sm.red, w, l, r0, rows, a, bias, res, y, ys, rs);
}

#undef R11_G07_PIN2
