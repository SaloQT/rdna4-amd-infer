/* ============================================================================
 * Inkling-2 R3 Group 18: Decode Fused IQ3_XXS Gate/Up + SITU + Q8_0 Quant
 *   Target: RX 9070 XT (gfx1201, wave32, NT=256, ROCm/Clang AMDGPU)
 *
 * Export (included into kernels/ie_inkling_kernels.c after ie_kernels.c):
 *   KERNEL ie_inkling_gexp_iq3xxs_situ_r3(
 *       const G u8 *qw, const G u8 *xq, G float *b, G u8 *q,
 *       u32 eff, u32 nb, const G u32 *route, u32 q_bytes,
 *       u32 astr, u32 bstr, u32 qstr, float beta, float lbeta)
 *
 *   - Bit:     37 (K_INK_GEXP_IQ3XXS_SITU_R3)
 *   - Replaces: T=1 decode [OP_GEMV_EXP(IQ3_XXS egu), OP_SITU(Q8)]
 *               (48 MoE passes/tok, 2304x2304x128 IQ3_XXS, eff=1152, K=8)
 *   - Grid:    ((eff + 31u) / 32u, K, 1u)   [e.g. (36, 8, 1) = 288 WGs]
 *   - Guards:  T == 1u && o0->w->kind == MAT_RAWQ && o0->w->sub == 18u &&
 *              o1->kind == OP_SITU && o1->a == o0->b &&
 *              o1->aoff == 0u && o0->boff == 0u && o1->nh == o0->nh &&
 *              o1->n > 0u && (o1->n & 31u) == 0u && o0->w->rows == 2u * o1->n &&
 *              o0->w->nb >= 8u && (o0->w->nb & 7u) == 0u &&
 *              o1->b < 0 && o1->qo >= 0
 *   - LDS:     1,152 B (1 KiB IQ3_XXS grid LUT + 128 B situ_v[32])
 *   - Target:  <= 64 VGPR (4 WGs/CU, 0 spills)
 *   - Parity:  100% bitwise-exact (0 bit diffs) vs
 *              ie_inkling_gexp_raw_iq3xxs_r1 + ie_inkling_situ_r1
 * ========================================================================== */

#define INK_R3_G18_SB_BYTES 98u

static inline __attribute__((always_inline)) float ink_r3_g18_iq3xxs_sub_dot(
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

  u32 wl[4], wh[4];
#pragma unroll
  for (u32 g = 0u; g < 4u; g++) {
    const u32 s8 = rq_ksign7((aux >> (7u * g)) & 127u);
    wl[g] = ie_rq_ssign(g_lo[g], ie_rq_mask4(s8 & 15u));
    wh[g] = ie_rq_ssign(g_hi[g], ie_rq_mask4(s8 >> 4u));
  }

  /* Extract x0/x1 lanes to scalar u32 (safe against Clang 21 ext_vec hazards). */
  const u32 x00 = x0.x, x01 = x0.y, x02 = x0.z, x03 = x0.w;
  const u32 x10 = x1.x, x11 = x1.y, x12 = x1.z, x13 = x1.w;

  u32 idot = ie_dot4_ss(0u, wl[0], x00);
  idot = ie_dot4_ss(idot, wh[0], x01);
  idot = ie_dot4_ss(idot, wl[1], x02);
  idot = ie_dot4_ss(idot, wh[1], x03);
  idot = ie_dot4_ss(idot, wl[2], x10);
  idot = ie_dot4_ss(idot, wh[2], x11);
  idot = ie_dot4_ss(idot, wl[3], x12);
  idot = ie_dot4_ss(idot, wh[3], x13);
  return dav * (dw * (float)(int)idot);
}

KERNEL ie_inkling_gexp_iq3xxs_situ_r3(const G u8 *qw, const G u8 *xq, G float *b, G u8 *q,
                                      u32 eff, u32 nb, const G u32 *route, u32 q_bytes,
                                      u32 astr, u32 bstr, u32 qstr, float beta, float lbeta) {
#pragma clang fp contract(off)
  static LDS u32 ink_r3_g18_dec_lut[256] __attribute__((aligned(16)));
  static LDS float ink_r3_g18_situ_v[32] __attribute__((aligned(16)));

  if (eff == 0u || (eff & 31u) != 0u || nb < 8u || (nb & 7u) != 0u || !qw || !xq || (!b && !q)) return;
  const u32 blk = wgid();
  const u32 nb_out = eff >> 5u;
  if (blk >= nb_out) return;

  const u32 t0 = tid();
  ink_r3_g18_dec_lut[t0] = ie_rq_iq3xxs_grid[t0];
  lbarrier();

  const u32 slot = tokid();
  const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[slot]);
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane();
  const u32 k = l >> 3u;
  const u32 s = l & 7u;
  const u32 r_in_blk = wv * 4u + k;
  const u32 r = blk * 32u + r_in_blk;

  const u32 mb = nb >> 3u;
  const u32 row_bytes = mb * INK_R3_G18_SB_BYTES;
  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * q_bytes);
  const G u8 *bpg = ebase + (unsigned long)r * row_bytes;
  const G u8 *bpu = ebase + (unsigned long)(eff + r) * row_bytes;
  const G u8 *xp = rq_ptr_u(xq + (unsigned long)slot * astr);
  const G float *dap = (const G float *)(xp + 32ul * nb);

  const u32 qo = 2u + 8u * s;
  const u32 auxo = 66u + 4u * s;

  /* 4-accumulator reduction matching exact wave_tree_f lane-group order. */
  float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
  float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;

  /* Prologue: prefetch superblock 0 activations + gate weights. */
  u32 dg_c = ie_rq_a16(bpg, 0u);
  u32 auxg_c = ie_rq_a32(bpg, auxo);
  u32 q01g_c = ie_rq_a32(bpg, qo);
  u32 q23g_c = ie_rq_a32(bpg, qo + 4u);
  u32x4 x0_c = *(const G u32x4 *)(xp + 32u * s);
  u32x4 x1_c = *(const G u32x4 *)(xp + 32u * s + 16u);
  float dav_c = dap[s];
  __asm__ volatile("" : "+v"(dg_c), "+v"(auxg_c), "+v"(q01g_c), "+v"(q23g_c));

#pragma unroll 1
  for (u32 sb = 0u; sb < mb; sb++) {
    const u32 j = sb & 3u;
    const int has_nxt = (sb + 1u < mb);

    /* Phase 1: issue current-SB up loads while computing current-SB gate dot. */
    const u32 du_c = ie_rq_a16(bpu, 0u);
    const u32 auxu_c = ie_rq_a32(bpu, auxo);
    const u32 q01u_c = ie_rq_a32(bpu, qo);
    const u32 q23u_c = ie_rq_a32(bpu, qo + 4u);
    __builtin_amdgcn_sched_barrier(0);

    const float vg = ink_r3_g18_iq3xxs_sub_dot(
        ink_r3_g18_dec_lut, dg_c, auxg_c, q01g_c, q23g_c, x0_c, x1_c, dav_c);
    if (j == 0u) ag0 += vg;
    else if (j == 1u) ag1 += vg;
    else if (j == 2u) ag2 += vg;
    else ag3 += vg;
    __builtin_amdgcn_sched_barrier(0);

    /* Phase 2: issue next-SB gate + activation loads while computing current-SB up dot. */
    bpg += INK_R3_G18_SB_BYTES;
    bpu += INK_R3_G18_SB_BYTES;
    u32 dg_n = 0u, auxg_n = 0u, q01g_n = 0u, q23g_n = 0u;
    u32x4 x0_n = (u32x4)(0u), x1_n = (u32x4)(0u);
    float dav_n = 0.0f;
    if (has_nxt) {
      const u32 bn = 8u * (sb + 1u) + s;
      dg_n = ie_rq_a16(bpg, 0u);
      auxg_n = ie_rq_a32(bpg, auxo);
      q01g_n = ie_rq_a32(bpg, qo);
      q23g_n = ie_rq_a32(bpg, qo + 4u);
      x0_n = *(const G u32x4 *)(xp + 32u * bn);
      x1_n = *(const G u32x4 *)(xp + 32u * bn + 16u);
      dav_n = dap[bn];
    }
    __builtin_amdgcn_sched_barrier(0);

    const float vu = ink_r3_g18_iq3xxs_sub_dot(
        ink_r3_g18_dec_lut, du_c, auxu_c, q01u_c, q23u_c, x0_c, x1_c, dav_c);
    if (j == 0u) au0 += vu;
    else if (j == 1u) au1 += vu;
    else if (j == 2u) au2 += vu;
    else au3 += vu;
    __builtin_amdgcn_sched_barrier(0);

    dg_c = dg_n;
    auxg_c = auxg_n;
    q01g_c = q01g_n;
    q23g_c = q23g_n;
    x0_c = x0_n;
    x1_c = x1_n;
    dav_c = dav_n;
  }

  /* Exact wave_tree_f parenthesization: (a0 + a2) + (a1 + a3), then DPP shl 4, 2, 1. */
  float g = (ag0 + ag2) + (ag1 + ag3);
  g += I2F(shl_i(F2I(g), 4));
  g += I2F(shl_i(F2I(g), 2));
  g += I2F(shl_i(F2I(g), 1));

  float u = (au0 + au2) + (au1 + au3);
  u += I2F(shl_i(F2I(u), 4));
  u += I2F(shl_i(F2I(u), 2));
  u += I2F(shl_i(F2I(u), 1));

  /* Exact SITU activation on the 32 block elements (s == 0u -> r_in_blk = 0..31). */
  if (s == 0u) {
    const float gs = beta > 0.0f ? beta * ie_tanhf_s(g / beta) : g;
    const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(u / lbeta) : u;
    ink_r3_g18_situ_v[r_in_blk] = gs * sigmoidf(g) * us;
  }
  lbarrier();

  /* Wave 0 writes optional f32 b and exact Q8_0 quantized output via quant_wave. */
  if (wv == 0u) {
    const float v = ink_r3_g18_situ_v[l];
    if (b) ((G float *)((G u8 *)b + (unsigned long)slot * bstr))[blk * 32u + l] = v;
    if (q) quant_wave(v, blk, q + (unsigned long)slot * qstr, nb_out);
  }
}

#undef INK_R3_G18_SB_BYTES
