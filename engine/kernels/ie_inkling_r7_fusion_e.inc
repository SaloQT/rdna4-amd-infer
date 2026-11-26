/* ============================================================================
 * Inkling-2 Round 7: Wave 2 Fusion E Super-Fusions (ink_mask2 bits 66, 67, 68)
 * Target: RX 9070 XT (gfx1201, wave32, NT=256)
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r7_fe_ /
 * INK_R7_FE_ and all macros are #undef'd at EOF.
 *
 * Exported kernels (100% bitwise-exact 0-bit-diff parity):
 *
 * 1. Bit 66 (E1): KERNEL ie_inkling_gexp_iq3xxs_situ_compact_r7(...)
 *    - Fuses verify (T=2..16) compact active-pack IQ3_XXS Gate/Up GEMV + SITU
 *      + Q8_0 quant into 1 launch (-48 launches/step, elides 28.3 MB/step of
 *      exp_gu f32 writes & reads!).
 *    - Consumes packs[] & n_packs from ie_inkling_moe_sort_compact_r6 /
 *      ie_inkling_router_route_sort_ver_r7.
 *    - Grid: ((eff + 31u) >> 5u, max_packs, 1u)  [(36, max_packs, 1) for eff=1152]
 *    - Guard: T >= 2u && T <= 16u && o0->w->kind == MAT_RAWQ && o0->w->sub == 18u &&
 *             o1->kind == OP_SITU && o1->a == o0->b && o1->aoff == 0u &&
 *             o0->boff == 0u && o1->nh == o0->nh && o1->n > 0u && (o1->n & 31u) == 0u &&
 *             o0->w->rows == 2u * o1->n && o0->w->nb >= 8u && (o0->w->nb & 7u) == 0u &&
 *             o1->b < 0 && o1->qo >= 0 && buf_disjoint(g, o0->a, o1->qo).
 *    - Parity: 0-bit-diff vs ie_inkling_gexp_raw_iq3xxs_compact_r6 + ie_inkling_situ_r1.
 *
 * 2. Bit 67 (E2): KERNEL ie_inkling_attn_res_norm_q8_r7(...)
 *    - Fuses non-loop OP_ATTN_RES (K=1 or K=3) + K x OP_RMSNORM [+ Q8_0 quant]
 *      into 1 launch (-57 launches/step: 24 ffn pass<=0 + 20 SWA attn pass<=0 +
 *      4*3 Global attn pass<=0 + 1 final out, eliding intermediate attn_res/ffn_res
 *      f32 writes & reads!).
 *    - Grid: (1u, T, 1u), NT = 256.
 *    - Guard: n == 2304u && M >= 1u && M <= 32u && K >= 1u && K <= 3u &&
 *             16-B aligned inputs/outputs & strides (T==1 || aligned) &&
 *             all output buffers (y0..y{K-1}, q0..q{K-1}) disjoint from inputs (a, c).
 *             (Note: intermediate attn_res/ffn_res buffer is completely dead and not written.)
 *    - Parity: 0-bit-diff vs ie_inkling_attn_res_r1 + K x ie_inkling_rmsnorm_mt_r1.
 *
 * 3. Bit 68 (E3): KERNEL ie_inkling_wo_sconv_post_r7(...)
 *    - Fuses OP_GEMV_Q8(wo, 2304x3072) + OP_SCONV(attn_pconv, static 4-tap) +
 *      (OP_ADD(block_res, apc -> x1) | OP_COPY(block_res -> blocks_buf)) [+ deferred
 *      sconv hist_out on verify] into 1 launch across ALL 48 visits!
 *      (-48 launches/step on T=1, -96 on T=2..16 verify; 0 atomics, 0 LDS, elides
 *      entire aout & apc global traffic!).
 *    - Grid: ((dim + 31u) >> 5u, 1u, 1u)  [(72, 1, 1) for dim = 2304]
 *    - Guard: T >= 1u && T <= 16u && dim > 0u && (dim & 31u) == 0u &&
 *             nb >= 8u && (nb & 7u) == 0u && c_dyn == NULL &&
 *             out disjoint from xq; if blk_dst != NULL, blk_dst disjoint from out & xq.
 *             (In-place out == block_res is safe when blk_dst == NULL.)
 *    - Parity: 0-bit-diff vs ie_gemv_q8q8[_tr] + ie_inkling_sconv_post_r7.
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R7_FE_SB_BYTES 98u
#define INK_R7_FE_AR_MAXM  32u
#define INK_R7_FE_AR_SLOTS 128u /* 32 * 4 = 128 -> 4096 B LDS */
#define INK_R7_FE_WO_TMAX  8u

typedef float ink_r7_fe_f8 __attribute__((ext_vector_type(8)));

struct __attribute__((packed, aligned(4))) ink_r7_fe_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r7_fe_ld4(const G void *p) {
  return ((const G struct ink_r7_fe_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) u32 ink_r7_fe_div_x(u32 pid, u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

static inline __attribute__((always_inline)) float ink_r7_fe_oct_tree_f(float v) {
#pragma clang fp contract(off)
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}

static inline __attribute__((always_inline)) float ink_r7_fe_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline __attribute__((always_inline)) int ink_r7_fe_dot8(u32x4 w0, u32x4 w1, u32x4 a0, u32x4 a1) {
  const u32 w00 = w0.x, w01 = w0.y, w02 = w0.z, w03 = w0.w;
  const u32 w10 = w1.x, w11 = w1.y, w12 = w1.z, w13 = w1.w;
  const u32 a00 = a0.x, a01 = a0.y, a02 = a0.z, a03 = a0.w;
  const u32 a10 = a1.x, a11 = a1.y, a12 = a1.z, a13 = a1.w;
  u32 d = ie_dot4_ss(0u, w00, a00);
  d = ie_dot4_ss(d, w01, a01);
  d = ie_dot4_ss(d, w02, a02);
  d = ie_dot4_ss(d, w03, a03);
  d = ie_dot4_ss(d, w10, a10);
  d = ie_dot4_ss(d, w11, a11);
  d = ie_dot4_ss(d, w12, a12);
  d = ie_dot4_ss(d, w13, a13);
  return (int)d;
}

static inline __attribute__((always_inline)) float ink_r7_fe_iq3_dot(
    u32 wl0, u32 wh0, u32 wl1, u32 wh1, u32 wl2, u32 wh2, u32 wl3, u32 wh3,
    float dw, u32x4 x0, u32x4 x1, float dav) {
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

static inline __attribute__((always_inline)) float ink_r7_fe_iq3_stream_dot(
    const LDSP u32 *ulut, u32 d16, u32 aux, u32 q01, u32 q23,
    u32x4 x0, u32x4 x1, float dav) {
#pragma clang fp contract(off)
  const float d = ie_rq_h2f(d16);
  const float dw = d * (0.5f + (float)(aux >> 28u)) * 0.5f;
  const u32 qp0 = q01 & 0xFFFFu, qp1 = q01 >> 16u;
  const u32 qp2 = q23 & 0xFFFFu, qp3 = q23 >> 16u;
  const u32 g_lo0 = ulut[qp0 & 255u], g_hi0 = ulut[qp0 >> 8u];
  const u32 g_lo1 = ulut[qp1 & 255u], g_hi1 = ulut[qp1 >> 8u];
  const u32 g_lo2 = ulut[qp2 & 255u], g_hi2 = ulut[qp2 >> 8u];
  const u32 g_lo3 = ulut[qp3 & 255u], g_hi3 = ulut[qp3 >> 8u];
  __builtin_amdgcn_sched_barrier(0);

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

#define INK_R7_FE_ADD4(a0_, a1_, a2_, a3_, j_, v_) \
  do {                                             \
    if ((j_) == 0u) (a0_) += (v_);                 \
    else if ((j_) == 1u) (a1_) += (v_);            \
    else if ((j_) == 2u) (a2_) += (v_);            \
    else (a3_) += (v_);                            \
  } while (0)

/* ============================================================================
 * Kernel E1 (Bit 66): ie_inkling_gexp_iq3xxs_situ_compact_r7
 * Compact active-pack IQ3_XXS Gate/Up + SITU + Q8_0 for verify (T=2..16).
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq3xxs_situ_compact_r7(
    const G u8 *qw, const G u8 *X, G float *b_out, G u8 *q_out,
    u32 eff, u32 nb, const G u32 *list, const G u32 *packs, const G u32 *n_packs,
    u32 q_bytes, u32 xs, u32 bstr, u32 qstr, u32 xdiv, float beta, float lbeta) {
#pragma clang fp contract(off)
  static LDS u32   ink_r7_fe_iq3_lut[256u] __attribute__((aligned(16)));
  static LDS float ink_r7_fe_situ_v[4u][32u] __attribute__((aligned(16)));

  if (eff == 0u || (eff & 31u) != 0u || nb < 8u || (nb & 7u) != 0u ||
      !qw || !X || (!b_out && !q_out) || !list || !packs || !n_packs) return;

  const u32 by = tokid();
  const u32 npk = (u32)__builtin_amdgcn_readfirstlane((int)(*n_packs));
  const u32 blk = wgid();
  const u32 nb_out = eff >> 5u;
  if (by >= npk || blk >= nb_out) return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)packs[by]);
  const u32 e = desc & 0xFFu;
  const u32 lo = (desc >> 8u) & 0xFFFu;
  const u32 nj = (desc >> 20u) & 0xFu;
  if (nj == 0u) return;

  const u32 t0 = tid();
  ink_r7_fe_iq3_lut[t0] = ie_rq_iq3xxs_grid[t0];
  lbarrier();

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 r_in_blk = (wv << 2u) + k;
  const u32 r = (blk << 5u) + r_in_blk;
  const u32 mb = nb >> 3u;
  const u32 row_bytes = mb * INK_R7_FE_SB_BYTES;
  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * (unsigned long)q_bytes);
  const G u8 *bpg = ebase + (unsigned long)r * (unsigned long)row_bytes;
  const G u8 *bpu = ebase + (unsigned long)(eff + r) * (unsigned long)row_bytes;
  const u32 qo = 2u + (s << 3u);
  const u32 auxo = 66u + (s << 2u);
  const unsigned long x_scale_off = 32ul * (unsigned long)nb;

  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
  const G u8 *xp0 = rq_ptr_u(X + (unsigned long)ink_r7_fe_div_x(p0, xdiv) * (unsigned long)xs);
  const G float *dap0 = (const G float *)(xp0 + x_scale_off);

  if (nj == 1u) {
    float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
    float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;

    u32 dg_c = ie_rq_a16(bpg, 0u);
    u32 auxg_c = ie_rq_a32(bpg, auxo);
    u32 q01g_c = ie_rq_a32(bpg, qo);
    u32 q23g_c = ie_rq_a32(bpg, qo + 4u);
    u32x4 x0_c = ink_r7_fe_ld4(xp0 + (s << 5u));
    u32x4 x1_c = ink_r7_fe_ld4(xp0 + (s << 5u) + 16u);
    float dav_c = dap0[s];
    __asm__ volatile("" : "+v"(dg_c), "+v"(auxg_c), "+v"(q01g_c), "+v"(q23g_c));

#pragma unroll 1
    for (u32 sb = 0u; sb < mb; sb++) {
      const u32 j_acc = sb & 3u;
      const u32 du_c = ie_rq_a16(bpu, 0u);
      const u32 auxu_c = ie_rq_a32(bpu, auxo);
      const u32 q01u_c = ie_rq_a32(bpu, qo);
      const u32 q23u_c = ie_rq_a32(bpu, qo + 4u);
      __builtin_amdgcn_sched_barrier(0);

      const float vg = ink_r7_fe_iq3_stream_dot(ink_r7_fe_iq3_lut, dg_c, auxg_c, q01g_c, q23g_c, x0_c, x1_c, dav_c);
      INK_R7_FE_ADD4(ag0, ag1, ag2, ag3, j_acc, vg);
      __builtin_amdgcn_sched_barrier(0);

      bpg += INK_R7_FE_SB_BYTES;
      bpu += INK_R7_FE_SB_BYTES;
      u32 dg_n = 0u, auxg_n = 0u, q01g_n = 0u, q23g_n = 0u;
      u32x4 x0_n = (u32x4)(0u), x1_n = (u32x4)(0u);
      float dav_n = 0.0f;
      if (sb + 1u < mb) {
        const u32 bn = ((sb + 1u) << 3u) + s;
        dg_n = ie_rq_a16(bpg, 0u);
        auxg_n = ie_rq_a32(bpg, auxo);
        q01g_n = ie_rq_a32(bpg, qo);
        q23g_n = ie_rq_a32(bpg, qo + 4u);
        x0_n = ink_r7_fe_ld4(xp0 + (bn << 5u));
        x1_n = ink_r7_fe_ld4(xp0 + (bn << 5u) + 16u);
        dav_n = dap0[bn];
      }
      __builtin_amdgcn_sched_barrier(0);

      const float vu = ink_r7_fe_iq3_stream_dot(ink_r7_fe_iq3_lut, du_c, auxu_c, q01u_c, q23u_c, x0_c, x1_c, dav_c);
      INK_R7_FE_ADD4(au0, au1, au2, au3, j_acc, vu);
      __builtin_amdgcn_sched_barrier(0);

      dg_c = dg_n; auxg_c = auxg_n; q01g_c = q01g_n; q23g_c = q23g_n;
      x0_c = x0_n; x1_c = x1_n; dav_c = dav_n;
    }

    const float gv = ink_r7_fe_oct_tree_f((ag0 + ag2) + (ag1 + ag3));
    const float uv = ink_r7_fe_oct_tree_f((au0 + au2) + (au1 + au3));
    if (s == 0u) {
      const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
      const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
      ink_r7_fe_situ_v[0u][r_in_blk] = gs * sigmoidf(gv) * us;
    }
    lbarrier();
    if (wv == 0u) {
      const float v = ink_r7_fe_situ_v[0u][l];
      if (b_out) ((G float *)((G u8 *)b_out + (unsigned long)p0 * (unsigned long)bstr))[(blk << 5u) + l] = v;
      if (q_out) quant_wave(v, blk, q_out + (unsigned long)p0 * (unsigned long)qstr, nb_out);
    }
    return;
  }

  /* Multi-pair path (nj in 2..4): decode gate & up weights ONCE across all nj pairs. */
  const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 1u]);
  const u32 p2 = (nj > 2u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 2u]) : p0;
  const u32 p3 = (nj > 3u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 3u]) : p0;

  const G u8 *xp1 = rq_ptr_u(X + (unsigned long)ink_r7_fe_div_x(p1, xdiv) * (unsigned long)xs);
  const G u8 *xp2 = (nj > 2u) ? rq_ptr_u(X + (unsigned long)ink_r7_fe_div_x(p2, xdiv) * (unsigned long)xs) : xp0;
  const G u8 *xp3 = (nj > 3u) ? rq_ptr_u(X + (unsigned long)ink_r7_fe_div_x(p3, xdiv) * (unsigned long)xs) : xp0;

  const G float *dap1 = (const G float *)(xp1 + x_scale_off);
  const G float *dap2 = (const G float *)(xp2 + x_scale_off);
  const G float *dap3 = (const G float *)(xp3 + x_scale_off);

  float ag0_0 = 0.0f, ag0_1 = 0.0f, ag0_2 = 0.0f, ag0_3 = 0.0f;
  float ag1_0 = 0.0f, ag1_1 = 0.0f, ag1_2 = 0.0f, ag1_3 = 0.0f;
  float ag2_0 = 0.0f, ag2_1 = 0.0f, ag2_2 = 0.0f, ag2_3 = 0.0f;
  float ag3_0 = 0.0f, ag3_1 = 0.0f, ag3_2 = 0.0f, ag3_3 = 0.0f;

  float au0_0 = 0.0f, au0_1 = 0.0f, au0_2 = 0.0f, au0_3 = 0.0f;
  float au1_0 = 0.0f, au1_1 = 0.0f, au1_2 = 0.0f, au1_3 = 0.0f;
  float au2_0 = 0.0f, au2_1 = 0.0f, au2_2 = 0.0f, au2_3 = 0.0f;
  float au3_0 = 0.0f, au3_1 = 0.0f, au3_2 = 0.0f, au3_3 = 0.0f;

#pragma unroll 1
  for (u32 sb = 0u; sb < mb; sb++) {
    const u32 b = (sb << 3u) + s;
    const u32 j_acc = sb & 3u;

    /* Pass A: Gate weights for superblock sb -> all nj pairs. */
    {
      const u32 d_c = ie_rq_a16(bpg, 0u);
      const u32 aux_c = ie_rq_a32(bpg, auxo);
      const u32 q01_c = ie_rq_a32(bpg, qo);
      const u32 q23_c = ie_rq_a32(bpg, qo + 4u);
      const u32x4 x0_a = ink_r7_fe_ld4(xp0 + (b << 5u));
      const u32x4 x1_a = ink_r7_fe_ld4(xp0 + (b << 5u) + 16u);
      const float dav_a = dap0[b];

      const u32 qp0 = q01_c & 0xFFFFu, qp1 = q01_c >> 16u;
      const u32 qp2 = q23_c & 0xFFFFu, qp3 = q23_c >> 16u;
      const u32 g_lo0 = ink_r7_fe_iq3_lut[qp0 & 255u], g_hi0 = ink_r7_fe_iq3_lut[qp0 >> 8u];
      const u32 g_lo1 = ink_r7_fe_iq3_lut[qp1 & 255u], g_hi1 = ink_r7_fe_iq3_lut[qp1 >> 8u];
      const u32 g_lo2 = ink_r7_fe_iq3_lut[qp2 & 255u], g_hi2 = ink_r7_fe_iq3_lut[qp2 >> 8u];
      const u32 g_lo3 = ink_r7_fe_iq3_lut[qp3 & 255u], g_hi3 = ink_r7_fe_iq3_lut[qp3 >> 8u];
      __builtin_amdgcn_sched_barrier(0);

      const float dw = ie_rq_h2f(d_c) * (0.5f + (float)(aux_c >> 28u)) * 0.5f;
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

      const float v0 = ink_r7_fe_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_a, x1_a, dav_a);
      INK_R7_FE_ADD4(ag0_0, ag0_1, ag0_2, ag0_3, j_acc, v0);
      __builtin_amdgcn_sched_barrier(0);

      const u32x4 x0_b = ink_r7_fe_ld4(xp1 + (b << 5u));
      const u32x4 x1_b = ink_r7_fe_ld4(xp1 + (b << 5u) + 16u);
      const float dav_b = dap1[b];
      const float v1 = ink_r7_fe_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_b, x1_b, dav_b);
      INK_R7_FE_ADD4(ag1_0, ag1_1, ag1_2, ag1_3, j_acc, v1);

      if (nj > 2u) {
        const u32x4 x0_c = ink_r7_fe_ld4(xp2 + (b << 5u));
        const u32x4 x1_c = ink_r7_fe_ld4(xp2 + (b << 5u) + 16u);
        const float dav_c = dap2[b];
        const float v2 = ink_r7_fe_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_c, x1_c, dav_c);
        INK_R7_FE_ADD4(ag2_0, ag2_1, ag2_2, ag2_3, j_acc, v2);
        if (nj > 3u) {
          const u32x4 x0_d = ink_r7_fe_ld4(xp3 + (b << 5u));
          const u32x4 x1_d = ink_r7_fe_ld4(xp3 + (b << 5u) + 16u);
          const float dav_d = dap3[b];
          const float v3 = ink_r7_fe_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_d, x1_d, dav_d);
          INK_R7_FE_ADD4(ag3_0, ag3_1, ag3_2, ag3_3, j_acc, v3);
        }
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    /* Pass B: Up weights for superblock sb -> all nj pairs (X is L1-hot from Pass A!). */
    {
      const u32 d_c = ie_rq_a16(bpu, 0u);
      const u32 aux_c = ie_rq_a32(bpu, auxo);
      const u32 q01_c = ie_rq_a32(bpu, qo);
      const u32 q23_c = ie_rq_a32(bpu, qo + 4u);
      const u32x4 x0_a = ink_r7_fe_ld4(xp0 + (b << 5u));
      const u32x4 x1_a = ink_r7_fe_ld4(xp0 + (b << 5u) + 16u);
      const float dav_a = dap0[b];

      const u32 qp0 = q01_c & 0xFFFFu, qp1 = q01_c >> 16u;
      const u32 qp2 = q23_c & 0xFFFFu, qp3 = q23_c >> 16u;
      const u32 g_lo0 = ink_r7_fe_iq3_lut[qp0 & 255u], g_hi0 = ink_r7_fe_iq3_lut[qp0 >> 8u];
      const u32 g_lo1 = ink_r7_fe_iq3_lut[qp1 & 255u], g_hi1 = ink_r7_fe_iq3_lut[qp1 >> 8u];
      const u32 g_lo2 = ink_r7_fe_iq3_lut[qp2 & 255u], g_hi2 = ink_r7_fe_iq3_lut[qp2 >> 8u];
      const u32 g_lo3 = ink_r7_fe_iq3_lut[qp3 & 255u], g_hi3 = ink_r7_fe_iq3_lut[qp3 >> 8u];
      __builtin_amdgcn_sched_barrier(0);

      const float dw = ie_rq_h2f(d_c) * (0.5f + (float)(aux_c >> 28u)) * 0.5f;
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

      const float v0 = ink_r7_fe_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_a, x1_a, dav_a);
      INK_R7_FE_ADD4(au0_0, au0_1, au0_2, au0_3, j_acc, v0);
      __builtin_amdgcn_sched_barrier(0);

      const u32x4 x0_b = ink_r7_fe_ld4(xp1 + (b << 5u));
      const u32x4 x1_b = ink_r7_fe_ld4(xp1 + (b << 5u) + 16u);
      const float dav_b = dap1[b];
      const float v1 = ink_r7_fe_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_b, x1_b, dav_b);
      INK_R7_FE_ADD4(au1_0, au1_1, au1_2, au1_3, j_acc, v1);

      if (nj > 2u) {
        const u32x4 x0_c = ink_r7_fe_ld4(xp2 + (b << 5u));
        const u32x4 x1_c = ink_r7_fe_ld4(xp2 + (b << 5u) + 16u);
        const float dav_c = dap2[b];
        const float v2 = ink_r7_fe_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_c, x1_c, dav_c);
        INK_R7_FE_ADD4(au2_0, au2_1, au2_2, au2_3, j_acc, v2);
        if (nj > 3u) {
          const u32x4 x0_d = ink_r7_fe_ld4(xp3 + (b << 5u));
          const u32x4 x1_d = ink_r7_fe_ld4(xp3 + (b << 5u) + 16u);
          const float dav_d = dap3[b];
          const float v3 = ink_r7_fe_iq3_dot(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_d, x1_d, dav_d);
          INK_R7_FE_ADD4(au3_0, au3_1, au3_2, au3_3, j_acc, v3);
        }
      }
    }
    bpg += INK_R7_FE_SB_BYTES;
    bpu += INK_R7_FE_SB_BYTES;
  }

  /* Exact SITU per pair -> 512 B LDS -> parallel wave quantization across wv = 0..nj-1! */
  {
    const float gv = ink_r7_fe_oct_tree_f((ag0_0 + ag0_2) + (ag0_1 + ag0_3));
    const float uv = ink_r7_fe_oct_tree_f((au0_0 + au0_2) + (au0_1 + au0_3));
    if (s == 0u) {
      const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
      const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
      ink_r7_fe_situ_v[0u][r_in_blk] = gs * sigmoidf(gv) * us;
    }
  }
  {
    const float gv = ink_r7_fe_oct_tree_f((ag1_0 + ag1_2) + (ag1_1 + ag1_3));
    const float uv = ink_r7_fe_oct_tree_f((au1_0 + au1_2) + (au1_1 + au1_3));
    if (s == 0u) {
      const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
      const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
      ink_r7_fe_situ_v[1u][r_in_blk] = gs * sigmoidf(gv) * us;
    }
  }
  if (nj > 2u) {
    const float gv = ink_r7_fe_oct_tree_f((ag2_0 + ag2_2) + (ag2_1 + ag2_3));
    const float uv = ink_r7_fe_oct_tree_f((au2_0 + au2_2) + (au2_1 + au2_3));
    if (s == 0u) {
      const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
      const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
      ink_r7_fe_situ_v[2u][r_in_blk] = gs * sigmoidf(gv) * us;
    }
  }
  if (nj > 3u) {
    const float gv = ink_r7_fe_oct_tree_f((ag3_0 + ag3_2) + (ag3_1 + ag3_3));
    const float uv = ink_r7_fe_oct_tree_f((au3_0 + au3_2) + (au3_1 + au3_3));
    if (s == 0u) {
      const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
      const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
      ink_r7_fe_situ_v[3u][r_in_blk] = gs * sigmoidf(gv) * us;
    }
  }
  lbarrier();

  if (wv < nj) {
    const u32 pw = (wv == 0u) ? p0 : ((wv == 1u) ? p1 : ((wv == 2u) ? p2 : p3));
    const float v = ink_r7_fe_situ_v[wv][l];
    if (b_out) ((G float *)((G u8 *)b_out + (unsigned long)pw * (unsigned long)bstr))[(blk << 5u) + l] = v;
    if (q_out) quant_wave(v, blk, q_out + (unsigned long)pw * (unsigned long)qstr, nb_out);
  }
}

/* ============================================================================
 * Kernel E2 (Bit 67): ie_inkling_attn_res_norm_q8_r7
 * Fuses non-loop OP_ATTN_RES (K=1..3, n=2304, M<=32) + K x OP_RMSNORM [+ Q8].
 * ========================================================================== */

typedef struct {
  union {
    float ar_red[INK_R7_FE_AR_SLOTS * 8u]; /* 4,096 B */
    struct {
      float rn_red[3u][8u];
      float rn_sh[8u][128u];               /* 4,096 B */
    } rn;
  } u;
} ink_r7_fe_ar_lds_t;

static LDS ink_r7_fe_ar_lds_t ink_r7_fe_ar_lds __attribute__((aligned(16)));

static inline __attribute__((always_inline)) float ink_r7_fe_wave_max(float v) {
  v = __builtin_fmaxf(v, shfl_xor_f(v, 16u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 8u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 4u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 2u));
  v = __builtin_fmaxf(v, shfl_xor_f(v, 1u));
  return v;
}

static inline __attribute__((always_inline)) float ink_r7_fe_wave_softmax(float logit, u32 l, u32 M) {
  const float v = (l < M) ? logit : -__builtin_inff();
  const float mx = ink_r7_fe_wave_max(v);
  const float e = (l < M) ? __builtin_expf(logit - mx) : 0.0f;
  const float ls = wave_sum_all(e);
  return (l < M) ? (e * (1.0f / ls)) : 0.0f;
}

static inline __attribute__((always_inline)) void ink_r7_fe_quant_chunk4(
    f32x4 y4, u32 wv, u32 l, u32 b0, G u8 *q, u32 nb) {
  *(LDSP f32x4 *)&ink_r7_fe_ar_lds.u.rn.rn_sh[wv][l << 2u] = y4;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 m = 0u; m < 4u; m++) quant_wave(ink_r7_fe_ar_lds.u.rn.rn_sh[wv][(m << 5u) + l], b0 + m, q, nb);
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

static inline __attribute__((always_inline)) float ink_r7_fe_ss_wave_2304(f32x4 x0, f32x4 x1, float x2) {
  float ss = (x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w);
  ss += (x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w);
  ss += x2 * x2;
  return wave_tree_f(ss);
}

static inline __attribute__((always_inline)) void ink_r7_fe_norm_finish_2304(
    f32x4 x0, f32x4 x1, float x2, const G float *nw, G float *y, G u8 *q,
    float eps, u32 head_idx, u32 th, u32 wv, u32 l) {
  float tot = 0.0f;
#pragma unroll
  for (u32 w = 0u; w < 8u; w++) tot += ink_r7_fe_ar_lds.u.rn.rn_red[head_idx][w];
  const float s = 1.0f / __builtin_sqrtf(tot / 2304.0f + eps);

  const f32x4 w0 = *(const G f32x4 *)(nw + (th << 2u));
  const f32x4 w1 = *(const G f32x4 *)(nw + 1024u + (th << 2u));
  const float w2 = nw[2048u + th];
  const f32x4 y0 = x0 * s * w0;
  const f32x4 y1 = x1 * s * w1;
  const float y2 = x2 * s * w2;
  if (y) {
    *(G f32x4 *)(y + (th << 2u)) = y0;
    *(G f32x4 *)(y + 1024u + (th << 2u)) = y1;
    y[2048u + th] = y2;
  }
  if (q) {
    ink_r7_fe_quant_chunk4(y0, wv, l, wv << 2u, q, 72u);
    ink_r7_fe_quant_chunk4(y1, wv, l, 32u + (wv << 2u), q, 72u);
    quant_wave(y2, 64u + wv, q, 72u);
  }
}

KERNEL ie_inkling_attn_res_norm_q8_r7(
    const G float *a, const G float *c, const G float *q,
    const G float *nw0, G float *y0, G u8 *q0,
    const G float *nw1, G float *y1, G u8 *q1,
    const G float *nw2, G float *y2, G u8 *q2,
    u32 M, u32 K, u32 n, float eps,
    u32 as, u32 cs, u32 y0s, u32 q0s, u32 y1s, u32 q1s, u32 y2s, u32 q2s,
    int has_c) {
  if (n != 2304u || M == 0u || M > INK_R7_FE_AR_MAXM || K == 0u || K > 3u) return;

  const u32 th = tid();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(th >> 5u));
  const u32 l = lane();
  if (a) a = TOKC(a, as);
  if (has_c && c) c = TOKC(c, cs);
  if (y0) y0 = TOK(y0, y0s);
  if (q0) q0 = TOK(q0, q0s);
  if (K > 1u) {
    if (y1) y1 = TOK(y1, y1s);
    if (q1) q1 = TOK(q1, q1s);
  }
  if (K > 2u) {
    if (y2) y2 = TOK(y2, y2s);
    if (q2) q2 = TOK(q2, q2s);
  }

  const u32 n_a = has_c ? (M - 1u) : M;
  const float inv_sqn = 1.0f / __builtin_sqrtf(2304.0f);
  const f32x4 z4 = {0.0f, 0.0f, 0.0f, 0.0f};

  /* Load queries q[0..K-1] into VGPRs. */
  const f32x4 q0_0 = *(const G f32x4 *)(q + (th << 2u));
  const f32x4 q1_0 = *(const G f32x4 *)(q + 1024u + (th << 2u));
  const float q2_0 = q[2048u + th];
  f32x4 q0_1 = z4, q1_1 = z4, q0_2 = z4, q1_2 = z4;
  float q2_1 = 0.0f, q2_2 = 0.0f;
  if (K > 1u) {
    const G float *qk1 = q + 2304u;
    q0_1 = *(const G f32x4 *)(qk1 + (th << 2u));
    q1_1 = *(const G f32x4 *)(qk1 + 1024u + (th << 2u));
    q2_1 = qk1[2048u + th];
  }
  if (K > 2u) {
    const G float *qk2 = q + 4608u;
    q0_2 = *(const G f32x4 *)(qk2 + (th << 2u));
    q1_2 = *(const G f32x4 *)(qk2 + 1024u + (th << 2u));
    q2_2 = qk2[2048u + th];
  }

  /* Pass 1: slot norms + K dot products (zero in-loop barriers). */
  const u32 stride_s = K + 1u;
  for (u32 s = 0u; s < M; s++) {
    const G float *sp = (s < n_a) ? (a + (unsigned long)s * 2304ul) : c;
    const f32x4 x0 = *(const G f32x4 *)(sp + (th << 2u));
    const f32x4 x1 = *(const G f32x4 *)(sp + 1024u + (th << 2u));
    const float x2 = sp[2048u + th];

    float sq = ((x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w)) +
               ((x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w)) + x2 * x2;
    sq = wave_tree_f(sq);
    const u32 rbase = s * stride_s * 8u;
    if (l == 0u) ink_r7_fe_ar_lds.u.ar_red[rbase + wv] = sq;

    {
      const f32x4 p0 = x0 * q0_0, p1 = x1 * q1_0;
      float d0 = ((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w)) + x2 * q2_0;
      d0 = wave_tree_f(d0);
      if (l == 0u) ink_r7_fe_ar_lds.u.ar_red[rbase + 8u + wv] = d0;
    }
    if (K > 1u) {
      const f32x4 p0 = x0 * q0_1, p1 = x1 * q1_1;
      float d1 = ((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w)) + x2 * q2_1;
      d1 = wave_tree_f(d1);
      if (l == 0u) ink_r7_fe_ar_lds.u.ar_red[rbase + 16u + wv] = d1;
    }
    if (K > 2u) {
      const f32x4 p0 = x0 * q0_2, p1 = x1 * q1_2;
      float d2 = ((p0.x + p0.y) + (p0.z + p0.w)) + ((p1.x + p1.y) + (p1.z + p1.w)) + x2 * q2_2;
      d2 = wave_tree_f(d2);
      if (l == 0u) ink_r7_fe_ar_lds.u.ar_red[rbase + 24u + wv] = d2;
    }
  }
  lbarrier();

  /* Wave-parallel softmax over lanes 0..M-1 (exact match to ie_inkling_attn_res_r1). */
  float my_inv = 0.0f, lg0 = 0.0f, lg1 = 0.0f, lg2 = 0.0f;
  if (l < M) {
    const u32 rbase = l * stride_s * 8u;
    float sum_sq = 0.0f;
#pragma unroll
    for (u32 x = 0u; x < 8u; x++) sum_sq += ink_r7_fe_ar_lds.u.ar_red[rbase + x];
    my_inv = 1.0f / __builtin_sqrtf(sum_sq / 2304.0f + eps);
    const float sc = my_inv * inv_sqn;

    float sd0 = 0.0f;
#pragma unroll
    for (u32 x = 0u; x < 8u; x++) sd0 += ink_r7_fe_ar_lds.u.ar_red[rbase + 8u + x];
    lg0 = sd0 * sc;

    if (K > 1u) {
      float sd1 = 0.0f;
#pragma unroll
      for (u32 x = 0u; x < 8u; x++) sd1 += ink_r7_fe_ar_lds.u.ar_red[rbase + 16u + x];
      lg1 = sd1 * sc;
    }
    if (K > 2u) {
      float sd2 = 0.0f;
#pragma unroll
      for (u32 x = 0u; x < 8u; x++) sd2 += ink_r7_fe_ar_lds.u.ar_red[rbase + 24u + x];
      lg2 = sd2 * sc;
    }
  }
  const float my_prob0 = ink_r7_fe_wave_softmax(lg0, l, M);
  const float my_prob1 = (K > 1u) ? ink_r7_fe_wave_softmax(lg1, l, M) : 0.0f;
  const float my_prob2 = (K > 2u) ? ink_r7_fe_wave_softmax(lg2, l, M) : 0.0f;
  lbarrier(); /* Retire ar_red reads so unioned rn_red/rn_sh is safe to write! */

  /* Pass 2: weighted sum directly into VGPRs (a0_k, a1_k, a2_k) -- 0 intermediate global stores! */
  f32x4 a0_0 = z4, a1_0 = z4, a0_1 = z4, a1_1 = z4, a0_2 = z4, a1_2 = z4;
  float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f;
  for (u32 s = 0u; s < M; s++) {
    const G float *sp = (s < n_a) ? (a + (unsigned long)s * 2304ul) : c;
    const f32x4 x0 = *(const G f32x4 *)(sp + (th << 2u));
    const f32x4 x1 = *(const G f32x4 *)(sp + 1024u + (th << 2u));
    const float x2 = sp[2048u + th];
    const float inv_s = I2F(__builtin_amdgcn_readlane(F2I(my_inv), (int)s));
    const f32x4 xn0 = x0 * inv_s;
    const f32x4 xn1 = x1 * inv_s;
    const float xn2 = x2 * inv_s;

    const float p0 = I2F(__builtin_amdgcn_readlane(F2I(my_prob0), (int)s));
    a0_0 += xn0 * p0; a1_0 += xn1 * p0; a2_0 += xn2 * p0;
    if (K > 1u) {
      const float p1 = I2F(__builtin_amdgcn_readlane(F2I(my_prob1), (int)s));
      a0_1 += xn0 * p1; a1_1 += xn1 * p1; a2_1 += xn2 * p1;
    }
    if (K > 2u) {
      const float p2 = I2F(__builtin_amdgcn_readlane(F2I(my_prob2), (int)s));
      a0_2 += xn0 * p2; a1_2 += xn1 * p2; a2_2 += xn2 * p2;
    }
  }

  /* Pass 3: fuse all K=1..3 RMSNorm sum-of-squares reductions behind 1 single lbarrier(). */
  const float ss0 = ink_r7_fe_ss_wave_2304(a0_0, a1_0, a2_0);
  const float ss1 = (K > 1u) ? ink_r7_fe_ss_wave_2304(a0_1, a1_1, a2_1) : 0.0f;
  const float ss2 = (K > 2u) ? ink_r7_fe_ss_wave_2304(a0_2, a1_2, a2_2) : 0.0f;
  if (l == 0u) {
    ink_r7_fe_ar_lds.u.rn.rn_red[0u][wv] = ss0;
    if (K > 1u) ink_r7_fe_ar_lds.u.rn.rn_red[1u][wv] = ss1;
    if (K > 2u) ink_r7_fe_ar_lds.u.rn.rn_red[2u][wv] = ss2;
  }
  lbarrier();

  ink_r7_fe_norm_finish_2304(a0_0, a1_0, a2_0, nw0, y0, q0, eps, 0u, th, wv, l);
  if (K > 1u) ink_r7_fe_norm_finish_2304(a0_1, a1_1, a2_1, nw1, y1, q1, eps, 1u, th, wv, l);
  if (K > 2u) ink_r7_fe_norm_finish_2304(a0_2, a1_2, a2_2, nw2, y2, q2, eps, 2u, th, wv, l);
}

/* ============================================================================
 * Kernel E3 (Bit 68): ie_inkling_wo_sconv_post_r7
 * Fuses OP_GEMV_Q8(wo) + OP_SCONV(attn_pconv) + (OP_ADD | OP_COPY) [+ hist_out].
 * 0 atomics, 0 LDS, 100% lane utilization on nb=96 (12 x 8), 0-bit-diff exact!
 * ========================================================================== */

KERNEL ie_inkling_wo_sconv_post_r7(
    const G u32 *qw, const G f16 *qs, const G u8 *xq,
    const G float *v_conv, G float *ring,
    const G float *block_res, G float *out, G float *blk_dst, G float *hist_out,
    u32 dim, u32 nb, float escale, u32 pos0, u32 T,
    u32 xs, u32 res_s, u32 out_s, u32 blk_s, u32 hist_row_stride) {
#pragma clang fp contract(off)
  if (T == 0u || T > 16u || dim == 0u || nb < 8u || (nb & 7u) != 0u) return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 r = (wgid() << 5u) + (w << 2u) + (l >> 3u);
  if (r >= dim) return;

  const G u32 *row_qw = qw + (unsigned long)r * (unsigned long)nb * 8ul;
  const G f16 *row_qs = qs + (unsigned long)r * (unsigned long)nb;

  /* Load 4-tap conv weights and seed causal ring history [x1, x2, x3] for channel r on sub == 0. */
  float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f;
  float x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
  if (sub == 0u) {
    const f32x4 vw = ((const G f32x4 *)v_conv)[r];
    w0 = vw.w; w1 = vw.z; w2 = vw.y; w3 = vw.x;
    if (pos0 > 0u) {
      const f32x4 rg = ((const G f32x4 *)ring)[r];
      x1 = ink_r7_fe_pick4(rg, (pos0 - 1u) & 3u);
      if (pos0 >= 2u) x2 = ink_r7_fe_pick4(rg, (pos0 - 2u) & 3u);
      if (pos0 >= 3u) x3 = ink_r7_fe_pick4(rg, (pos0 - 3u) & 3u);
    }
  }

  /* Process tokens in chunks of <= 8 using exact 8-lane/row Q8_0 GEMV (1x DRAM weight read
   * for all T <= 8 tokens, 100% lane utilization on nb=96 = 12x8!). */
  for (u32 t0 = 0u; t0 < T; t0 += INK_R7_FE_WO_TMAX) {
    const u32 tc = (T - t0 < INK_R7_FE_WO_TMAX) ? (T - t0) : INK_R7_FE_WO_TMAX;
    ink_r7_fe_f8 a0 = (ink_r7_fe_f8)(0.0f);
    ink_r7_fe_f8 a1 = (ink_r7_fe_f8)(0.0f);
    ink_r7_fe_f8 a2 = (ink_r7_fe_f8)(0.0f);
    ink_r7_fe_f8 a3 = (ink_r7_fe_f8)(0.0f);

    const u32 nb_main = nb >> 3u;
#pragma clang loop unroll(disable)
    for (u32 blk = 0u; blk < nb_main; blk++) {
      const u32 blk_u = (u32)__builtin_amdgcn_readfirstlane((int)blk);
      const u32 slot = (u32)__builtin_amdgcn_readfirstlane((int)(blk_u & 3u));
      const u32 b = (blk_u << 3u) + sub;

      const G u32 *wp = row_qw + (unsigned long)b * 8ul;
      const u32x4 wv0 = ink_r7_fe_ld4(wp);
      const u32x4 wv1 = ink_r7_fe_ld4(wp + 4u);
      const float sw = (float)row_qs[b];

#pragma unroll
      for (u32 j = 0u; j < INK_R7_FE_WO_TMAX; j++) {
        if (j < tc) {
          const G u8 *xj = xq + (unsigned long)(t0 + j) * (unsigned long)xs;
          const G u32 *aw = (const G u32 *)xj;
          const u32x4 av0 = ink_r7_fe_ld4(aw + (unsigned long)b * 8ul);
          const u32x4 av1 = ink_r7_fe_ld4(aw + (unsigned long)b * 8ul + 4ul);
          const float da = ((const G float *)(xj + (unsigned long)nb * 32ul))[b];
          const int dot = ink_r7_fe_dot8(wv0, wv1, av0, av1);
          const float term = (sw * da) * (float)dot;
          if (slot == 0u) a0[j] = a0[j] + term;
          else if (slot == 1u) a1[j] = a1[j] + term;
          else if (slot == 2u) a2[j] = a2[j] + term;
          else a3[j] = a3[j] + term;
          if ((j & 1u) == 1u) __builtin_amdgcn_sched_barrier(0);
        }
      }
      __builtin_amdgcn_sched_barrier(0);
    }

    /* Exact wave_tree_f + inline causal 4-tap sconv + residual add/block copy! */
#pragma unroll
    for (u32 j = 0u; j < INK_R7_FE_WO_TMAX; j++) {
      if (j < tc) {
        const float u0 = a0[j] + a2[j];
        const float u1 = a1[j] + a3[j];
        float x0 = ink_r7_fe_oct_tree_f(u0 + u1);
        if (sub == 0u) {
          const u32 tk = t0 + j;
          const float cv = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3;
          const float sc = (x0 + cv) * escale;
          const float rv = ((const G float *)((const G u8 *)block_res + (unsigned long)tk * (unsigned long)res_s))[r];
          G float *op = (G float *)((G u8 *)out + (unsigned long)tk * (unsigned long)out_s);
          if (blk_dst) {
            G float *bp = (G float *)((G u8 *)blk_dst + (unsigned long)tk * (unsigned long)blk_s);
            bp[r] = rv;
            op[r] = sc;
          } else {
            op[r] = rv + sc;
          }
          if (hist_out) {
            hist_out[(unsigned long)tk * (unsigned long)hist_row_stride + (unsigned long)r] = x0;
          } else if (tk + 4u >= T) {
            ring[(unsigned long)r * 4ul + ((pos0 + tk) & 3u)] = x0;
          }
          x3 = x2;
          x2 = x1;
          x1 = x0;
        }
      }
    }
  }
}

#undef INK_R7_FE_ADD4
#undef INK_R7_FE_WO_TMAX
#undef INK_R7_FE_AR_SLOTS
#undef INK_R7_FE_AR_MAXM
#undef INK_R7_FE_SB_BYTES

#endif /* IE_MEGA */
