/* ============================================================================
 * Inkling-2 R3 Group 19 (g19-decode-bw-r3): Low-VGPR High-Occupancy T=1 Decode
 * MoE Expert GEMVs for RX 9070 XT (gfx1201, wave32, NT=256).
 *
 * Target: Push in-VRAM decode bandwidth on the two MoE expert GEMVs
 *   (1,353.6 MB/tok = 40.6% of decode weights) by trimming VGPRs from
 *   107/126 -> <= 64 VGPR (2 -> 4 WGs/CU = 2x wave occupancy to hide GDDR6
 *   latency) while keeping 100% bitwise-exact parity.
 *
 * Exports (included into kernels/ie_inkling_kernels.c after ie_kernels.c):
 *   1. KERNEL ie_inkling_gexp_raw_iq3xxs_r3(const G u8 *qw, const G u8 *xq,
 *        G float *y, u32 rows, u32 nb, const G u32 *route, u32 q_bytes,
 *        u32 astr, u32 bstr, u32 type)                               [bit 38]
 *      - Replaces: ie_inkling_gexp_raw_iq3xxs_r1 (IQ3_XXS 2304x2304x128, K=8)
 *      - Grid:     ((rows + 31u) / 32u, K, 1u)   [e.g. (72, 8, 1) = 576 WGs]
 *      - Guard:    T == 1u && type == IE_RQ_IQ3_XXS && rows > 0u &&
 *                  nb >= 8u && (nb & 7u) == 0u
 *      - Parity:   100% bitwise-exact (0 bit diffs) vs ie_gexp_raw
 *                  (#pragma clang fp contract(off), 4-acc wave_tree_f order)
 *      - Opts:     Hoisted SGPR expert ID e = readfirstlane(route[slot]) and
 *                  ebase/xp pointers, 1 KiB LDS LUT, 4 rows/wave (8 lanes/row,
 *                  0 idle lanes on nb=72), single-SB #pragma unroll 1 loop
 *                  with 1-SB lookahead prefetch (eliminates R1's 4x inner
 *                  unroll + duplicated 4x tail), streaming dot4 without
 *                  wl[4]/wh[4] live arrays -> <= 64 VGPR (4 WGs/CU).
 *
 *   2. KERNEL ie_inkling_gexp_iq4nl_r3(const G u32 *qw, const G u16 *qs,
 *        const G u8 *xq, G float *y, u32 rows, u32 nb, const G u32 *route,
 *        u32 qwn, u32 qsn, u32 astr, u32 bstr, u32 kind, u32 two_off)[bit 39]
 *      - Replaces: ie_inkling_gexp_iq4nl_r1 (IQ4_NL 2304x1152x128, nb=36, K=8)
 *      - Grid:     ((rows + 31u) / 32u, o->nh, 1u) [e.g. (72, 8, 1) = 576 WGs]
 *      - Guard:    T == 1u && kind == MAT_IQ4NL (8u)
 *      - Parity:   100% bitwise-exact (0 bit diffs) vs ie_gexp
 *                  (#pragma clang fp contract(off), 4-acc wave_tree_f order)
 *      - Opts:     Hoisted SGPR wv, expert ID e = readfirstlane(route[s]), and
 *                  w/q/x base pointers; unified #pragma unroll 1 1-block
 *                  lookahead loop (overlaps nb=36 tail load with step-3 dot4,
 *                  keeps raw f16 scale load ahead of sched_barrier to avoid
 *                  premature vmcnt stall, eliminates R1's 4-block live-range
 *                  explosion + 4 duplicated tail blocks); 4 rows/wave (8
 *                  lanes/row, 90% util on nb=36), coalesced u32x4 loads ->
 *                  <= 64 VGPR (4 WGs/CU).
 * ========================================================================== */

/* 8-lane subgroup reduction inside each 16-lane DPP row (lanes 0, 8, 16, 24
 * hold the exact sum of their 8-lane subgroup). */
static inline __attribute__((always_inline)) float ink_r3_g19_oct_tree_f(float v) {
#pragma clang fp contract(off)
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}

/* ============================================================================
 * Kernel 1: T=1 Decode MoE IQ3_XXS (4 rows/wave, 8 lanes/row, 32 rows/WG)
 *   - 1 KiB LDS grid LUT + ALU parity sign masks -> <= 64 VGPR (4 WGs/CU).
 *   - 1-SB lookahead inside a clean #pragma unroll 1 loop; scalar j = sb & 3u
 *     routes each superblock to its exact wave_tree_f accumulator a0..a3.
 * ========================================================================== */

static inline __attribute__((always_inline)) float ink_r3_g19_iq3xxs_sub_dot(
    const LDSP u32 *ulut, const u32 d16, const u32 aux, const u32 q01, const u32 q23,
    const u32x4 x0, const u32x4 x1, const float dav) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d16);
  const float dw = d * (0.5f + (float)(aux >> 28u)) * 0.5f;
  u32 g_lo[4], g_hi[4];
#pragma unroll
  for (u32 g = 0u; g < 4u; g++) {
    const u32 qp = ((g < 2u ? q01 : q23) >> (16u * (g & 1u))) & 0xFFFFu;
    g_lo[g] = ulut[qp & 255u];
    g_hi[g] = ulut[qp >> 8u];
  }
  __builtin_amdgcn_sched_barrier(0);
  u32 idot = 0u;
#pragma unroll
  for (u32 g = 0u; g < 4u; g++) {
    const u32 s8 = rq_ksign7((aux >> (7u * g)) & 127u);
    const u32 wl = ie_rq_ssign(g_lo[g], ie_rq_mask4(s8 & 15u));
    const u32 wh = ie_rq_ssign(g_hi[g], ie_rq_mask4(s8 >> 4u));
    const u32 xv0 = (g == 0u) ? x0.x : ((g == 1u) ? x0.z : ((g == 2u) ? x1.x : x1.z));
    const u32 xv1 = (g == 0u) ? x0.y : ((g == 1u) ? x0.w : ((g == 2u) ? x1.y : x1.w));
    idot = ie_dot4_ss(idot, wl, xv0);
    idot = ie_dot4_ss(idot, wh, xv1);
  }
  return dav * (dw * (float)(int)idot);
}

KERNEL_LOOP ie_inkling_gexp_raw_iq3xxs_r3(const G u8 *qw, const G u8 *xq, G float *y, u32 rows, u32 nb,
                                          const G u32 *route, u32 q_bytes, u32 astr, u32 bstr, u32 type) {
#pragma clang fp contract(off)
#ifdef IE_MEGA
#define ink_r3_g19_dec_lut mk_iq3_lut
  if (type != IE_RQ_IQ3_XXS || rows == 0u || nb < 8u || (nb & 7u) != 0u) return;
  const u32 t0 = tid();
#else
  static LDS u32 ink_r3_g19_dec_lut[256] __attribute__((aligned(16)));
  if (type != IE_RQ_IQ3_XXS || rows == 0u || nb < 8u || (nb & 7u) != 0u) return;
  const u32 t0 = tid();
  ink_r3_g19_dec_lut[t0] = ie_rq_iq3xxs_grid[t0];
  lbarrier();
#endif

  const u32 slot = tokid();
  const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[slot]);
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 r = wgid() * 32u + wv * 4u + k;
  const u32 rr = r < rows ? r : rows - 1u;
  const u32 mb = nb >> 3u;
  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * q_bytes);
  const G u8 *bp = ebase + (unsigned long)rr * ((unsigned long)mb * 98ul);
  const G u8 *xp = rq_ptr_u(xq + (unsigned long)slot * astr);
  const G float *dap = (const G float *)(xp + 32ul * nb);

  const u32 qo = 2u + 8u * s;
  const u32 auxo = 66u + 4u * s;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

  /* Prologue: prefetch superblock 0 (weights + activations). */
  u32 d_c = ie_rq_a16(bp, 0u);
  u32 aux_c = ie_rq_a32(bp, auxo);
  u32 q01_c = ie_rq_a32(bp, qo);
  u32 q23_c = ie_rq_a32(bp, qo + 4u);
  u32x4 x0_c = *(const G u32x4 *)(xp + 32u * s);
  u32x4 x1_c = *(const G u32x4 *)(xp + 32u * s + 16u);
  float dav_c = dap[s];
  __asm__ volatile("" : "+v"(d_c), "+v"(aux_c), "+v"(q01_c), "+v"(q23_c));

#pragma unroll 1
  for (u32 sb = 0u; sb < mb; sb++) {
    const u32 j = sb & 3u;
    const int has_nxt = (sb + 1u < mb);
    bp += 98u;
    u32 d_n = 0u, aux_n = 0u, q01_n = 0u, q23_n = 0u;
    u32x4 x0_n = (u32x4)(0u), x1_n = (u32x4)(0u);
    float dav_n = 0.0f;
    if (has_nxt) {
      const u32 bn = 8u * (sb + 1u) + s;
      d_n = ie_rq_a16(bp, 0u);
      aux_n = ie_rq_a32(bp, auxo);
      q01_n = ie_rq_a32(bp, qo);
      q23_n = ie_rq_a32(bp, qo + 4u);
      x0_n = *(const G u32x4 *)(xp + 32u * bn);
      x1_n = *(const G u32x4 *)(xp + 32u * bn + 16u);
      dav_n = dap[bn];
    }
    __builtin_amdgcn_sched_barrier(0);
    const float v = ink_r3_g19_iq3xxs_sub_dot(ink_r3_g19_dec_lut, d_c, aux_c, q01_c, q23_c, x0_c, x1_c, dav_c);
    if (j == 0u) a0 += v;
    else if (j == 1u) a1 += v;
    else if (j == 2u) a2 += v;
    else a3 += v;
    __builtin_amdgcn_sched_barrier(0);
    d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
    x0_c = x0_n; x1_c = x1_n; dav_c = dav_n;
  }

  /* Exact wave_tree_f parenthesization: (lane s + s+16) + (s+8 + s+24), then shl 4, 2, 1. */
  const float acc = ink_r3_g19_oct_tree_f((a0 + a2) + (a1 + a3));
  if (s == 0u && r < rows) {
    ((G float *)((G u8 *)y + (unsigned long)slot * bstr))[r] = acc;
  }
}

/* ============================================================================
 * Kernel 2: T=1 Decode MoE IQ4_NL (4 rows/wave, 8 lanes/row, 32 rows/WG)
 *   - Hoisted SGPR wv, expert ID e, and base pointers (w, q, x, o).
 *   - Unified #pragma unroll 1 1-block lookahead loop: raw f16 scale load
 *     stays ahead of sched_barrier (f16->f32 conversion happens in compute
 *     phase), overlapping all loads including the nb=36 partial tail ->
 *     <= 64 VGPR (4 WGs/CU), 0 LDS, 100% bitwise-exact vs ie_gexp.
 * ========================================================================== */

static inline __attribute__((always_inline)) int ink_r3_g19_iq4_dot4(u32x4 w4, u32x4 a0, u32x4 a1) {
  u32 d = 0u;
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(w4.x & 0x0F0F0F0Fu), a0.x), ie_iq4_map((w4.x >> 4u) & 0x0F0F0F0Fu), a1.x);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(w4.y & 0x0F0F0F0Fu), a0.y), ie_iq4_map((w4.y >> 4u) & 0x0F0F0F0Fu), a1.y);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(w4.z & 0x0F0F0F0Fu), a0.z), ie_iq4_map((w4.z >> 4u) & 0x0F0F0F0Fu), a1.z);
  d = ie_dot4_ss(ie_dot4_ss(d, ie_iq4_map(w4.w & 0x0F0F0F0Fu), a0.w), ie_iq4_map((w4.w >> 4u) & 0x0F0F0F0Fu), a1.w);
  return (int)d;
}

KERNEL_LOOP ie_inkling_gexp_iq4nl_r3(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                                     const G u32 *route, u32 qwn, u32 qsn, u32 astr, u32 bstr, u32 kind, u32 two_off) {
#pragma clang fp contract(off)
  (void)kind;
  (void)two_off;
  if (rows == 0u || nb == 0u) return;
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 r_wave = (wgid() * 8u + wv) * 4u;
  if (r_wave >= rows) return;

  const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
  const u32 r = r_wave + sub;
  const u32 rr = r < rows ? r : rows - 1u;
  const u32 s = tokid();
  const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[s]);

  const G u8 *w_base = rq_ptr_u((const G u8 *)(qw + (unsigned long)e * qwn));
  const G u8 *q_base = rq_ptr_u((const G u8 *)(qs + (unsigned long)e * qsn));
  const G u8 *x = rq_ptr_u(xq + (unsigned long)s * astr);
  G float *o = (G float *)((G u8 *)y + (unsigned long)s * bstr);

  const G u8 *rw = w_base + (unsigned long)rr * ((unsigned long)nb * 16ul);
  const G f16 *rs = (const G f16 *)q_base + (unsigned long)rr * nb;
  const G u32 *aw = (const G u32 *)x;
  const G float *da = (const G float *)(x + 32ul * nb);

  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  const u32 nstep = (nb + 7u) >> 3u;

  /* Prologue: prefetch step 0 (blocks 0..7). */
  u32x4 w_c = (u32x4)(0u), x0_c = (u32x4)(0u), x1_c = (u32x4)(0u);
  f16 sc_c = (f16)0.0f;
  float d_c = 0.0f;
  if (sl < nb) {
    w_c = *(const G u32x4 *)(rw + sl * 16u);
    sc_c = rs[sl];
    x0_c = *(const G u32x4 *)(aw + sl * 8u);
    x1_c = *(const G u32x4 *)(aw + sl * 8u + 4u);
    d_c = da[sl];
  }
  __builtin_amdgcn_sched_barrier(0);

#pragma unroll 1
  for (u32 step = 0u; step < nstep; step++) {
    const u32 b_c = step * 8u + sl;
    const u32 bn = b_c + 8u;
    u32x4 w_n = (u32x4)(0u), x0_n = (u32x4)(0u), x1_n = (u32x4)(0u);
    f16 sc_n = (f16)0.0f;
    float d_n = 0.0f;
    if (bn < nb) {
      w_n = *(const G u32x4 *)(rw + bn * 16u);
      sc_n = rs[bn];
      x0_n = *(const G u32x4 *)(aw + bn * 8u);
      x1_n = *(const G u32x4 *)(aw + bn * 8u + 4u);
      d_n = da[bn];
    }
    __builtin_amdgcn_sched_barrier(0);
    if (b_c < nb) {
      const float v = (float)sc_c * (float)ink_r3_g19_iq4_dot4(w_c, x0_c, x1_c);
      const u32 j = step & 3u;
      if (j == 0u) a0 = __builtin_fmaf(v, d_c, a0);
      else if (j == 1u) a1 = __builtin_fmaf(v, d_c, a1);
      else if (j == 2u) a2 = __builtin_fmaf(v, d_c, a2);
      else a3 = __builtin_fmaf(v, d_c, a3);
    }
    __builtin_amdgcn_sched_barrier(0);
    w_c = w_n; sc_c = sc_n; x0_c = x0_n; x1_c = x1_n; d_c = d_n;
  }

  /* Exact wave_tree_f parenthesization: (lane sl + sl+16) + (sl+8 + sl+24), then shl 4, 2, 1. */
  const float u0 = a0 + a2;
  const float u1 = a1 + a3;
  const float acc = ink_r3_g19_oct_tree_f(u0 + u1);
  if (sl == 0u && r < rows) o[r] = acc;
}
#ifdef IE_MEGA
#undef ink_r3_g19_dec_lut
#endif
