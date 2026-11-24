/* ============================================================================
 * Inkling-2 Round 10: Decode & Verify Super-Fusions (H14 + H15)
 * Target: RX 9070 XT (gfx1201, RDNA4, wave32, NT=256)
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r10_df_ /
 * INK_R10_DF_ and all macros are #undef'd at EOF.
 *
 * Deliverables:
 *   (a) H14 Lean <= 80 VGPR (4 WGs/CU) Wave-Split AoS IQ3_XXS + SITU + Q8_0:
 *       Waves 0..3 compute 16 gate rows, Waves 4..7 compute 16 up rows per
 *       sub-pass (2 sub-passes = 1 exact 32-elem Q8_0 block), sharing ffn_in_q8
 *       in LDS, 1 epilogue lbarrier(), exact #pragma clang fp contract(off) SITU
 *       + wave32 absmax Q8_0 store directly into exp_mid_q8 (elides 48 OP_SITU
 *       launches & exp_gu!).
 *       1. ie_inkling_gexp_iq3xxs_situ_wsplit_t1_r10  (T=1, 13-arg drop-in)
 *       2. ie_inkling_gexp_iq3xxs_situ_wsplit_ver_r10 (T=2..16 compact, 16-arg drop-in)
 *
 *   (b) H15 Fused OP_REL_ATTN + OP_GATE + Q8_0 Epilogue (-48 launches + att elision):
 *       NOTE: OP_GATE (ie_gate / ie_inkling_gate_q8_r2 / model.c:3008,3283) is
 *       `out * sigmoid(agate) = out * (1.0f / (1.0f + __builtin_expf(-agate)))`
 *       (NOT silu(agate)!). Exact-order drop-in variants:
 *       3. ie_inkling_rel_attn_dec_gate_q8_r10           (T=1, rel_attn_dec_r1 + gate_q8)
 *       4. ie_inkling_rel_attn_ver_gate_q8_r10           (T=2..16 SWA, rel_attn_ver_r4 + gate_q8)
 *       5. ie_inkling_rel_attn_ver_glob_wmma_gate_q8_r10 (T=2..8 Global, glob_wmma_r7 + gate_q8)
 *       Passing out == NULL elides dead f32 att global writes completely!
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(kvt) == 2u, "ie_inkling_r10_dec_fusions requires f16 KV cache");
_Static_assert(sizeof(f16) == 2u, "ink_r10_df f16 must be 2 bytes");

#define INK_R10_DF_SB_BYTES  98u
#define INK_R10_DF_LDS_MAXNB 96u
#define INK_R10_DF_HD        128u
#define INK_R10_DF_DREL      16u
#define INK_R10_DF_GQA       3u
#define INK_R10_DF_PSTR      130u
#define INK_R10_DF_LOG2E     1.44269504088896340736f
#define INK_R10_DF_LN2       0.69314718055994530942f
#define INK_R10_DF_QSC       (0.08838834764831844299f * 1.44269504088896340736f)

struct __attribute__((packed, aligned(4))) ink_r10_df_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r10_df_ld4(const G void *p) {
  return ((const G struct ink_r10_df_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) u32 ink_r10_df_div_x(u32 pid, u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

static inline __attribute__((always_inline)) float ink_r10_df_oct_tree_f(float v) {
#pragma clang fp contract(off)
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}

static inline __attribute__((always_inline)) float ink_r10_df_sgpr_f(float v) {
  return I2F(__builtin_amdgcn_readfirstlane(F2I(v)));
}

static inline __attribute__((always_inline)) float ink_r10_df_halfwave_sum_all(float v) {
  v += I2F(shl_i(F2I(v), 8));
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return I2F(__builtin_amdgcn_readlane(F2I(v), 0));
}

static inline __attribute__((always_inline)) float ink_r10_df_dot4(f32x4 a, f32x4 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

static inline __attribute__((always_inline)) float ink_r10_df_xrow(float v) {
  return I2F(xrow_i(F2I(v)));
}

/* Exact OP_GATE math matching ie_gate / ie_inkling_gate_q8_r2: a * sigmoid(g). */
static inline __attribute__((always_inline)) float ink_r10_df_gate_f(float a, float g) {
#pragma clang fp contract(off)
  return a * (1.0f / (1.0f + __builtin_expf(-g)));
}

/* 8-lane subgroup reductions matching ie_inkling_gate_q8_r2 (0 extra LDS!). */
static inline __attribute__((always_inline)) float ink_r10_df_sub8_max_pos(float v) {
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0xB1, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0x4E, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_ds_swizzle(F2I(v), 0x101F)));
  return v;
}

static inline __attribute__((always_inline)) u32 ink_r10_df_sub8_sum_u32(u32 s) {
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0xB1, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0x4E, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_ds_swizzle((int)s, 0x101F);
  return s;
}

/* In-register f32x4 gate + Q8_0 epilogue for 1 full head (hd = 128 = 4 x 32):
 * Lane l (0..31) holds f32x4 for elements [4*l .. 4*l + 3]; subgroup sg = l >> 3
 * (8 lanes = 32 floats) quantizes Q8_0 block (h * 4 + sg) with 0 extra LDS! */
static inline __attribute__((always_inline)) void ink_r10_df_gate_q8_head_vec4(
    f32x4 ov, const G float *gt_h, G float *ot_h, G u8 *oqt, u32 h, u32 n_head, u32 l) {
#pragma clang fp contract(off)
  f32x4 vv = ov;
  if (gt_h) {
    const f32x4 gv = ((const G f32x4 *)gt_h)[l];
    vv = (f32x4){ink_r10_df_gate_f(ov.x, gv.x), ink_r10_df_gate_f(ov.y, gv.y),
                 ink_r10_df_gate_f(ov.z, gv.z), ink_r10_df_gate_f(ov.w, gv.w)};
  }
  if (ot_h) ((G f32x4 *)ot_h)[l] = vv;
  if (oqt) {
    const u32 sg = l >> 3u;
    const u32 sl = l & 7u;
    const u32 blk = (h << 2u) + sg;
    const u32 nb_q = n_head << 2u;
    float am = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(vv.x), __builtin_fabsf(vv.y)),
                               __builtin_fmaxf(__builtin_fabsf(vv.z), __builtin_fabsf(vv.w)));
    am = ink_r10_df_sub8_max_pos(am);
    const float d = am / 127.0f;
    int qi0 = 0, qi1 = 0, qi2 = 0, qi3 = 0;
    if (d != 0.0f) {
      qi0 = (int)__builtin_roundf(vv.x / d);
      qi1 = (int)__builtin_roundf(vv.y / d);
      qi2 = (int)__builtin_roundf(vv.z / d);
      qi3 = (int)__builtin_roundf(vv.w / d);
    }
    const u32 pk = ((u32)(u8)qi0) | (((u32)(u8)qi1) << 8u) |
                   (((u32)(u8)qi2) << 16u) | (((u32)(u8)qi3) << 24u);
    *(G u32 *)(oqt + (blk << 5u) + (sl << 2u)) = pk;
    const u32 s = ink_r10_df_sub8_sum_u32((u32)qi0 + (u32)qi1 + (u32)qi2 + (u32)qi3);
    if (sl == 0u) {
      ((G float *)(oqt + (nb_q << 5u)))[blk] = d;
      ((G u32 *)(oqt + nb_q * 36u))[blk] = s;
    }
  }
}

static inline __attribute__((always_inline)) float ink_r10_df_iq3_dot_pre(
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

static inline __attribute__((always_inline)) float ink_r10_df_iq3_stream_dot(
    const LDSP u32 *ulut, u32 d16, u32 aux, u32 q01, u32 q23,
    u32x4 x0, u32x4 x1, float dav) {
#pragma clang fp contract(off)
  const float dw = ie_rq_h2f(d16) * (0.5f + (float)(aux >> 28u)) * 0.5f;
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

#define INK_R10_DF_ADD4(a0_, a1_, a2_, a3_, j_, v_) \
  do {                                              \
    if ((j_) == 0u) (a0_) += (v_);                  \
    else if ((j_) == 1u) (a1_) += (v_);             \
    else if ((j_) == 2u) (a2_) += (v_);             \
    else (a3_) += (v_);                             \
  } while (0)

/* ============================================================================
 * Kernel 1 (H14-T1): ie_inkling_gexp_iq3xxs_situ_wsplit_t1_r10
 * Lean <= 64 VGPR (4 WGs/CU) Wave-Split AoS IQ3_XXS + SITU + Q8_0 for T=1.
 *   - Waves 0..3 compute 16 gate rows, Waves 4..7 compute 16 up rows per
 *     sub-pass (2 sub-passes -> full 32-elem Q8_0 block), sharing ffn_in_q8 in
 *     LDS (staged before the LUT barrier -> 0 extra barriers!).
 *   - 1 epilogue lbarrier() -> Wave 0 computes exact 32-lane SITU + quant_wave
 *     directly into exp_mid_q8 (0 exp_gu writes/reads, -48 OP_SITU launches!).
 *   - Grid: ((eff + 31u) / 32u, K, 1u) [(36, 8, 1)], exact 13-arg drop-in.
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq3xxs_situ_wsplit_t1_r10(
    const G u8 *qw, const G u8 *xq, G float *b, G u8 *q,
    u32 eff, u32 nb, const G u32 *route, u32 q_bytes,
    u32 astr, u32 bstr, u32 qstr, float beta, float lbeta) {
#pragma clang fp contract(off)
  static LDS u32   ink_r10_df_t1_lut[256u] __attribute__((aligned(16)));
  static LDS u32   ink_r10_df_t1_xq[INK_R10_DF_LDS_MAXNB * 8u] __attribute__((aligned(16)));
  static LDS float ink_r10_df_t1_xd[INK_R10_DF_LDS_MAXNB] __attribute__((aligned(16)));
  static LDS float ink_r10_df_t1_g[32u] __attribute__((aligned(16)));
  static LDS float ink_r10_df_t1_u[32u] __attribute__((aligned(16)));

  if (eff == 0u || (eff & 31u) != 0u || nb < 8u || (nb & 7u) != 0u || !qw || !xq || (!b && !q)) return;
  const u32 blk = wgid();
  const u32 nb_out = eff >> 5u;
  if (blk >= nb_out) return;

  const u32 t0 = tid();
  const u32 slot = tokid();
  const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[slot]);
  const G u8 *xp = rq_ptr_u(xq + (unsigned long)slot * (unsigned long)astr);
  const G float *dap = (const G float *)(xp + 32ul * (unsigned long)nb);

  /* Stage IQ3_XXS LUT + ffn_in_q8 cooperatively before 1st lbarrier(). */
  ink_r10_df_t1_lut[t0] = ie_rq_iq3xxs_grid[t0];
  const int use_lds_x = (nb <= INK_R10_DF_LDS_MAXNB);
  if (use_lds_x) {
    const u32 n_v4 = nb << 1u;
    if (t0 < n_v4) ((LDSP u32x4 *)ink_r10_df_t1_xq)[t0] = ink_r10_df_ld4(xp + (t0 << 4u));
    if (t0 < nb) ink_r10_df_t1_xd[t0] = dap[t0];
  }
  lbarrier();

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 is_up = wv >> 2u;
  const u32 wv4 = wv & 3u;
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 mb = nb >> 3u;
  const u32 row_bytes = mb * INK_R10_DF_SB_BYTES;
  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * (unsigned long)q_bytes);
  const u32 qo = 2u + (s << 3u);
  const u32 auxo = 66u + (s << 2u);

  /* 2 sub-passes of 16 gate (Waves 0..3) + 16 up (Waves 4..7) = 32 block rows. */
#pragma unroll 1
  for (u32 sp = 0u; sp < 2u; sp++) {
    const u32 r_loc = (sp << 4u) + (wv4 << 2u) + k;
    const u32 r_mat = (is_up ? eff : 0u) + (blk << 5u) + r_loc;
    const G u8 *bp = ebase + (unsigned long)r_mat * (unsigned long)row_bytes;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    u32 d_c = ie_rq_a16(bp, 0u);
    u32 aux_c = ie_rq_a32(bp, auxo);
    u32 q01_c = ie_rq_a32(bp, qo);
    u32 q23_c = ie_rq_a32(bp, qo + 4u);
    __asm__ volatile("" : "+v"(d_c), "+v"(aux_c), "+v"(q01_c), "+v"(q23_c));

#pragma unroll 1
    for (u32 sb = 0u; sb < mb; sb++) {
      const u32 b_idx = (sb << 3u) + s;
      const u32 j_acc = sb & 3u;
      const G u8 *bp_n = (sb + 1u < mb) ? (bp + INK_R10_DF_SB_BYTES) : bp;

      const u32x4 x0 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_df_t1_xq + (b_idx << 3u))
                                 : ink_r10_df_ld4(xp + (b_idx << 5u));
      const u32x4 x1 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_df_t1_xq + (b_idx << 3u) + 4u)
                                 : ink_r10_df_ld4(xp + (b_idx << 5u) + 16u);
      const float dav = use_lds_x ? ink_r10_df_t1_xd[b_idx] : dap[b_idx];

      const u32 d_n = ie_rq_a16(bp_n, 0u);
      const u32 aux_n = ie_rq_a32(bp_n, auxo);
      const u32 q01_n = ie_rq_a32(bp_n, qo);
      const u32 q23_n = ie_rq_a32(bp_n, qo + 4u);
      __builtin_amdgcn_sched_barrier(0);

      const float v = ink_r10_df_iq3_stream_dot(
          ink_r10_df_t1_lut, d_c, aux_c, q01_c, q23_c, x0, x1, dav);
      INK_R10_DF_ADD4(a0, a1, a2, a3, j_acc, v);
      __builtin_amdgcn_sched_barrier(0);

      bp = bp_n;
      d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
    }

    const float acc = ink_r10_df_oct_tree_f((a0 + a2) + (a1 + a3));
    if (s == 0u) {
      if (is_up) ink_r10_df_t1_u[r_loc] = acc;
      else ink_r10_df_t1_g[r_loc] = acc;
    }
  }
  lbarrier();

  /* Wave 0 computes exact 32-lane SITU + wave32 Q8_0 store into exp_mid_q8! */
  if (wv == 0u) {
    const float gv = ink_r10_df_t1_g[l];
    const float uv = ink_r10_df_t1_u[l];
    const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
    const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
    const float v = gs * sigmoidf(gv) * us;
    if (b) ((G float *)((G u8 *)b + (unsigned long)slot * (unsigned long)bstr))[(blk << 5u) + l] = v;
    if (q) quant_wave(v, blk, q + (unsigned long)slot * (unsigned long)qstr, nb_out);
  }
}

/* ============================================================================
 * Kernel 2 (H14-Ver): ie_inkling_gexp_iq3xxs_situ_wsplit_ver_r10
 * Lean <= 80 VGPR (4 WGs/CU) Wave-Split AoS IQ3_XXS + SITU + Q8_0 for compact
 * verify (T=2..16, nj=1 and nj=2..4).
 *   - Waves 0..3 compute gate rows, Waves 4..7 compute up rows (halves live
 *     accumulators from 32 -> 16 VGPRs vs bit-66 situ_compact_r7!).
 *   - Stages nj <= 4 activations into LDS once (eliminates 4x global X loads!).
 *   - 1-SB branch-free AoS weight lookahead prefetch + 1 epilogue lbarrier().
 *   - Grid: ((eff + 31u) / 32u, max_packs, 1u), 15,872 B LDS, exact 16-arg drop-in.
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq3xxs_situ_wsplit_ver_r10(
    const G u8 *qw, const G u8 *X, G float *b_out, G u8 *q_out,
    u32 eff, u32 nb, const G u32 *list, const G u32 *packs, const G u32 *n_packs,
    u32 q_bytes, u32 xs, u32 bstr, u32 qstr, u32 xdiv, float beta, float lbeta) {
#pragma clang fp contract(off)
  static LDS u32   ink_r10_df_v_lut[256u] __attribute__((aligned(16)));
  static LDS u32   ink_r10_df_v_xq[4u][INK_R10_DF_LDS_MAXNB * 8u] __attribute__((aligned(16)));
  static LDS float ink_r10_df_v_xd[4u][INK_R10_DF_LDS_MAXNB] __attribute__((aligned(16)));
  static LDS float ink_r10_df_v_g[4u][32u] __attribute__((aligned(16)));
  static LDS float ink_r10_df_v_u[4u][32u] __attribute__((aligned(16)));

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
  ink_r10_df_v_lut[t0] = ie_rq_iq3xxs_grid[t0];

  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
  const u32 p1 = (nj > 1u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 1u]) : p0;
  const u32 p2 = (nj > 2u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 2u]) : p0;
  const u32 p3 = (nj > 3u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 3u]) : p0;

  const G u8 *xp0 = rq_ptr_u(X + (unsigned long)ink_r10_df_div_x(p0, xdiv) * (unsigned long)xs);
  const G u8 *xp1 = (nj > 1u) ? rq_ptr_u(X + (unsigned long)ink_r10_df_div_x(p1, xdiv) * (unsigned long)xs) : xp0;
  const G u8 *xp2 = (nj > 2u) ? rq_ptr_u(X + (unsigned long)ink_r10_df_div_x(p2, xdiv) * (unsigned long)xs) : xp0;
  const G u8 *xp3 = (nj > 3u) ? rq_ptr_u(X + (unsigned long)ink_r10_df_div_x(p3, xdiv) * (unsigned long)xs) : xp0;

  const unsigned long x_sc_off = 32ul * (unsigned long)nb;
  const int use_lds_x = (nb <= INK_R10_DF_LDS_MAXNB);
  if (use_lds_x) {
    const u32 n_v4 = nb << 1u;
    for (u32 j = 0u; j < nj; j++) {
      const G u8 *xpj = (j == 0u) ? xp0 : ((j == 1u) ? xp1 : ((j == 2u) ? xp2 : xp3));
      const G float *dapj = (const G float *)(xpj + x_sc_off);
      for (u32 idx = t0; idx < n_v4; idx += NT) {
        ((LDSP u32x4 *)ink_r10_df_v_xq[j])[idx] = ink_r10_df_ld4(xpj + (idx << 4u));
      }
      for (u32 idx = t0; idx < nb; idx += NT) {
        ink_r10_df_v_xd[j][idx] = dapj[idx];
      }
    }
  }
  lbarrier();

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 is_up = wv >> 2u;
  const u32 wv4 = wv & 3u;
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 mb = nb >> 3u;
  const u32 row_bytes = mb * INK_R10_DF_SB_BYTES;
  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * (unsigned long)q_bytes);
  const u32 qo = 2u + (s << 3u);
  const u32 auxo = 66u + (s << 2u);

  const G float *dap0 = (const G float *)(xp0 + x_sc_off);

  if (nj == 1u) {
#pragma unroll 1
    for (u32 sp = 0u; sp < 2u; sp++) {
      const u32 r_loc = (sp << 4u) + (wv4 << 2u) + k;
      const u32 r_mat = (is_up ? eff : 0u) + (blk << 5u) + r_loc;
      const G u8 *bp = ebase + (unsigned long)r_mat * (unsigned long)row_bytes;

      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
      u32 d_c = ie_rq_a16(bp, 0u);
      u32 aux_c = ie_rq_a32(bp, auxo);
      u32 q01_c = ie_rq_a32(bp, qo);
      u32 q23_c = ie_rq_a32(bp, qo + 4u);
      __asm__ volatile("" : "+v"(d_c), "+v"(aux_c), "+v"(q01_c), "+v"(q23_c));

#pragma unroll 1
      for (u32 sb = 0u; sb < mb; sb++) {
        const u32 b_idx = (sb << 3u) + s;
        const u32 j_acc = sb & 3u;
        const G u8 *bp_n = (sb + 1u < mb) ? (bp + INK_R10_DF_SB_BYTES) : bp;

        const u32x4 x0 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_df_v_xq[0] + (b_idx << 3u))
                                   : ink_r10_df_ld4(xp0 + (b_idx << 5u));
        const u32x4 x1 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_df_v_xq[0] + (b_idx << 3u) + 4u)
                                   : ink_r10_df_ld4(xp0 + (b_idx << 5u) + 16u);
        const float dav = use_lds_x ? ink_r10_df_v_xd[0][b_idx] : dap0[b_idx];

        const u32 d_n = ie_rq_a16(bp_n, 0u);
        const u32 aux_n = ie_rq_a32(bp_n, auxo);
        const u32 q01_n = ie_rq_a32(bp_n, qo);
        const u32 q23_n = ie_rq_a32(bp_n, qo + 4u);
        __builtin_amdgcn_sched_barrier(0);

        const float v = ink_r10_df_iq3_stream_dot(
            ink_r10_df_v_lut, d_c, aux_c, q01_c, q23_c, x0, x1, dav);
        INK_R10_DF_ADD4(a0, a1, a2, a3, j_acc, v);
        __builtin_amdgcn_sched_barrier(0);
        bp = bp_n;
        d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
      }

      const float acc = ink_r10_df_oct_tree_f((a0 + a2) + (a1 + a3));
      if (s == 0u) {
        if (is_up) ink_r10_df_v_u[0][r_loc] = acc;
        else ink_r10_df_v_g[0][r_loc] = acc;
      }
    }
  } else {
    const G float *dap1 = (const G float *)(xp1 + x_sc_off);
    const G float *dap2 = (const G float *)(xp2 + x_sc_off);
    const G float *dap3 = (const G float *)(xp3 + x_sc_off);

#pragma unroll 1
    for (u32 sp = 0u; sp < 2u; sp++) {
      const u32 r_loc = (sp << 4u) + (wv4 << 2u) + k;
      const u32 r_mat = (is_up ? eff : 0u) + (blk << 5u) + r_loc;
      const G u8 *bp = ebase + (unsigned long)r_mat * (unsigned long)row_bytes;

      float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
      float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
      float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
      float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;

      u32 d_c = ie_rq_a16(bp, 0u);
      u32 aux_c = ie_rq_a32(bp, auxo);
      u32 q01_c = ie_rq_a32(bp, qo);
      u32 q23_c = ie_rq_a32(bp, qo + 4u);
      __asm__ volatile("" : "+v"(d_c), "+v"(aux_c), "+v"(q01_c), "+v"(q23_c));

#pragma unroll 1
      for (u32 sb = 0u; sb < mb; sb++) {
        const u32 b_idx = (sb << 3u) + s;
        const u32 j_acc = sb & 3u;
        const G u8 *bp_n = (sb + 1u < mb) ? (bp + INK_R10_DF_SB_BYTES) : bp;

        const u32 d_n = ie_rq_a16(bp_n, 0u);
        const u32 aux_n = ie_rq_a32(bp_n, auxo);
        const u32 q01_n = ie_rq_a32(bp_n, qo);
        const u32 q23_n = ie_rq_a32(bp_n, qo + 4u);

        const u32 qp0 = q01_c & 0xFFFFu, qp1 = q01_c >> 16u;
        const u32 qp2 = q23_c & 0xFFFFu, qp3 = q23_c >> 16u;
        const u32 g_lo0 = ink_r10_df_v_lut[qp0 & 255u], g_hi0 = ink_r10_df_v_lut[qp0 >> 8u];
        const u32 g_lo1 = ink_r10_df_v_lut[qp1 & 255u], g_hi1 = ink_r10_df_v_lut[qp1 >> 8u];
        const u32 g_lo2 = ink_r10_df_v_lut[qp2 & 255u], g_hi2 = ink_r10_df_v_lut[qp2 >> 8u];
        const u32 g_lo3 = ink_r10_df_v_lut[qp3 & 255u], g_hi3 = ink_r10_df_v_lut[qp3 >> 8u];
        __builtin_amdgcn_sched_barrier(0);

        const float dw = ie_rq_h2f(d_c) * (0.5f + (float)(aux_c >> 28u)) * 0.5f;
        const u32 s8_0 = rq_ksign7(aux_c & 127u);
        const u32 wl0 = ie_rq_ssign(g_lo0, ie_rq_mask4(s8_0 & 15u)), wh0 = ie_rq_ssign(g_hi0, ie_rq_mask4(s8_0 >> 4u));
        const u32 s8_1 = rq_ksign7((aux_c >> 7u) & 127u);
        const u32 wl1 = ie_rq_ssign(g_lo1, ie_rq_mask4(s8_1 & 15u)), wh1 = ie_rq_ssign(g_hi1, ie_rq_mask4(s8_1 >> 4u));
        const u32 s8_2 = rq_ksign7((aux_c >> 14u) & 127u);
        const u32 wl2 = ie_rq_ssign(g_lo2, ie_rq_mask4(s8_2 & 15u)), wh2 = ie_rq_ssign(g_hi2, ie_rq_mask4(s8_2 >> 4u));
        const u32 s8_3 = rq_ksign7((aux_c >> 21u) & 127u);
        const u32 wl3 = ie_rq_ssign(g_lo3, ie_rq_mask4(s8_3 & 15u)), wh3 = ie_rq_ssign(g_hi3, ie_rq_mask4(s8_3 >> 4u));

        const u32x4 x0_0 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_df_v_xq[0] + (b_idx << 3u))
                                     : ink_r10_df_ld4(xp0 + (b_idx << 5u));
        const u32x4 x1_0 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_df_v_xq[0] + (b_idx << 3u) + 4u)
                                     : ink_r10_df_ld4(xp0 + (b_idx << 5u) + 16u);
        const float d0 = use_lds_x ? ink_r10_df_v_xd[0][b_idx] : dap0[b_idx];
        INK_R10_DF_ADD4(a0_0, a0_1, a0_2, a0_3, j_acc,
                        ink_r10_df_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_0, x1_0, d0));

        const u32x4 x0_1 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_df_v_xq[1] + (b_idx << 3u))
                                     : ink_r10_df_ld4(xp1 + (b_idx << 5u));
        const u32x4 x1_1 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_df_v_xq[1] + (b_idx << 3u) + 4u)
                                     : ink_r10_df_ld4(xp1 + (b_idx << 5u) + 16u);
        const float d1 = use_lds_x ? ink_r10_df_v_xd[1][b_idx] : dap1[b_idx];
        INK_R10_DF_ADD4(a1_0, a1_1, a1_2, a1_3, j_acc,
                        ink_r10_df_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_1, x1_1, d1));

        if (nj > 2u) {
          const u32x4 x0_2 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_df_v_xq[2] + (b_idx << 3u))
                                       : ink_r10_df_ld4(xp2 + (b_idx << 5u));
          const u32x4 x1_2 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_df_v_xq[2] + (b_idx << 3u) + 4u)
                                       : ink_r10_df_ld4(xp2 + (b_idx << 5u) + 16u);
          const float d2 = use_lds_x ? ink_r10_df_v_xd[2][b_idx] : dap2[b_idx];
          INK_R10_DF_ADD4(a2_0, a2_1, a2_2, a2_3, j_acc,
                          ink_r10_df_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_2, x1_2, d2));

          if (nj > 3u) {
            const u32x4 x0_3 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_df_v_xq[3] + (b_idx << 3u))
                                         : ink_r10_df_ld4(xp3 + (b_idx << 5u));
            const u32x4 x1_3 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_df_v_xq[3] + (b_idx << 3u) + 4u)
                                         : ink_r10_df_ld4(xp3 + (b_idx << 5u) + 16u);
            const float d3 = use_lds_x ? ink_r10_df_v_xd[3][b_idx] : dap3[b_idx];
            INK_R10_DF_ADD4(a3_0, a3_1, a3_2, a3_3, j_acc,
                            ink_r10_df_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_3, x1_3, d3));
          }
        }
        __builtin_amdgcn_sched_barrier(0);
        bp = bp_n;
        d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
      }

      const float acc0 = ink_r10_df_oct_tree_f((a0_0 + a0_2) + (a0_1 + a0_3));
      const float acc1 = ink_r10_df_oct_tree_f((a1_0 + a1_2) + (a1_1 + a1_3));
      if (s == 0u) {
        if (is_up) {
          ink_r10_df_v_u[0][r_loc] = acc0;
          ink_r10_df_v_u[1][r_loc] = acc1;
        } else {
          ink_r10_df_v_g[0][r_loc] = acc0;
          ink_r10_df_v_g[1][r_loc] = acc1;
        }
      }
      if (nj > 2u) {
        const float acc2 = ink_r10_df_oct_tree_f((a2_0 + a2_2) + (a2_1 + a2_3));
        if (s == 0u) {
          if (is_up) ink_r10_df_v_u[2][r_loc] = acc2;
          else ink_r10_df_v_g[2][r_loc] = acc2;
        }
      }
      if (nj > 3u) {
        const float acc3 = ink_r10_df_oct_tree_f((a3_0 + a3_2) + (a3_1 + a3_3));
        if (s == 0u) {
          if (is_up) ink_r10_df_v_u[3][r_loc] = acc3;
          else ink_r10_df_v_g[3][r_loc] = acc3;
        }
      }
    }
  }
  lbarrier();

  if (wv < nj) {
    const u32 pw = (wv == 0u) ? p0 : ((wv == 1u) ? p1 : ((wv == 2u) ? p2 : p3));
    const float gv = ink_r10_df_v_g[wv][l];
    const float uv = ink_r10_df_v_u[wv][l];
    const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
    const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
    const float v = gs * sigmoidf(gv) * us;
    if (b_out) ((G float *)((G u8 *)b_out + (unsigned long)pw * (unsigned long)bstr))[(blk << 5u) + l] = v;
    if (q_out) quant_wave(v, blk, q_out + (unsigned long)pw * (unsigned long)qstr, nb_out);
  }
}

/* ============================================================================
 * Kernel 3 (H15-T1): ie_inkling_rel_attn_dec_gate_q8_r10
 * Exact-order drop-in upgrade of ie_inkling_rel_attn_dec_r1 fusing OP_GATE
 * (`out * sigmoid(agate)`) + Q8_0 quantization directly inside the final
 * reduction waves (0 extra LDS, 0 extra barriers, optional out == NULL).
 * Grid: (n_kv * nsplit, 1u, 1u), <= 72 VGPR (4 WGs/CU), 12,484 B LDS.
 * ========================================================================== */

KERNEL ie_inkling_rel_attn_dec_gate_q8_r10(
    const G float *q, const G kvt *kc, const G kvt *vc, const G float *R,
    const G float *rel_bias, const G float *sink, G float *part, G float *out,
    u32 pos, u32 kv_len, u32 hd, u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent,
    u32 win, u32 nsplit, u32 ch, G u32 *count,
    const G float *agate, G u8 *oq) {
  static LDS float ink_r10_df_dg_m[INK_R10_DF_GQA][8];
  static LDS float ink_r10_df_dg_l[INK_R10_DF_GQA][8];
  static LDS __attribute__((aligned(16))) float ink_r10_df_dg_o[INK_R10_DF_GQA][8][INK_R10_DF_HD];
  static LDS u32 ink_r10_df_dg_last;

  if (hd != INK_R10_DF_HD || d_rel != INK_R10_DF_DREL || n_head != INK_R10_DF_GQA * n_kv) return;

  const u32 kh = wgid() / nsplit;
  const u32 s = wgid() % nsplit;
  const u32 h0 = kh * INK_R10_DF_GQA;
  const u32 kvd = n_kv * INK_R10_DF_HD;
  const u32 w = tid() >> 5u;
  const u32 l = lane();
  const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
  const float scale = 1.0f / __builtin_sqrtf((float)INK_R10_DF_HD);

  const f32x4 q0 = ((const G f32x4 *)(q + (unsigned long)(h0 + 0u) * INK_R10_DF_HD))[l];
  const f32x4 q1 = ((const G f32x4 *)(q + (unsigned long)(h0 + 1u) * INK_R10_DF_HD))[l];
  const f32x4 q2 = ((const G f32x4 *)(q + (unsigned long)(h0 + 2u) * INK_R10_DF_HD))[l];

  const float rv0 = (l < INK_R10_DF_DREL) ? R[(unsigned long)(h0 + 0u) * INK_R10_DF_DREL + l] : 0.0f;
  const float rv1 = (l < INK_R10_DF_DREL) ? R[(unsigned long)(h0 + 1u) * INK_R10_DF_DREL + l] : 0.0f;
  const float rv2 = (l < INK_R10_DF_DREL) ? R[(unsigned long)(h0 + 2u) * INK_R10_DF_DREL + l] : 0.0f;

  f32x4 acc0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 acc1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 acc2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};

  float m0 = -__builtin_inff(), sum0 = 0.0f;
  float m1 = -__builtin_inff(), sum1 = 0.0f;
  float m2 = -__builtin_inff(), sum2 = 0.0f;

  const u32 w0 = (win > 0u && pos + 1u > win) ? (pos + 1u - win) : 0u;
  const u32 p0 = w0 + s * ch;
  const u32 pend = pos + 1u;
  const u32 p1 = (w0 + (s + 1u) * ch < pend) ? (w0 + (s + 1u) * ch) : pend;

  const G kvt *kc_h = kc + (unsigned long)kh * INK_R10_DF_HD;
  const G kvt *vc_h = vc + (unsigned long)kh * INK_R10_DF_HD;

  u32 kp = p0 + w;
  if (kp < p1) {
    u32 kvp = kp % kv_len;
    const G kv4 *krow = (const G kv4 *)(kc_h + (unsigned long)kvp * kvd);
    const G kv4 *vrow = (const G kv4 *)(vc_h + (unsigned long)kvp * kvd);
    kv4 kh4 = krow[l];
    kv4 vh4 = vrow[l];

    for (;;) {
      const u32 dist = pos - kp;
      const int has_rel = (dist < extent_lim);
      float rb = 0.0f;
      if (has_rel && l < INK_R10_DF_DREL) rb = rel_bias[(unsigned long)dist * INK_R10_DF_DREL + l];

      const u32 kp_nxt = kp + 8u;
      u32 kvp_nxt = kvp;
      kv4 kh4_nxt = kh4;
      kv4 vh4_nxt = vh4;
      if (kp_nxt < p1) {
        kvp_nxt = kvp + 8u;
        if (kvp_nxt >= kv_len) kvp_nxt = (kv_len >= 8u) ? (kvp_nxt - kv_len) : (kp_nxt % kv_len);
        const G kv4 *krow_n = (const G kv4 *)(kc_h + (unsigned long)kvp_nxt * kvd);
        const G kv4 *vrow_n = (const G kv4 *)(vc_h + (unsigned long)kvp_nxt * kvd);
        kh4_nxt = krow_n[l];
        vh4_nxt = vrow_n[l];
      }

      const f32x4 kf = __builtin_convertvector(kh4, f32x4);
      float d0 = wave_sum_all(ink_r10_df_dot4(q0, kf)) * scale;
      float d1 = wave_sum_all(ink_r10_df_dot4(q1, kf)) * scale;
      float d2 = wave_sum_all(ink_r10_df_dot4(q2, kf)) * scale;

      if (has_rel) {
        d0 += ink_r10_df_halfwave_sum_all(rv0 * rb);
        d1 += ink_r10_df_halfwave_sum_all(rv1 * rb);
        d2 += ink_r10_df_halfwave_sum_all(rv2 * rb);
      }

      const float mn0 = __builtin_fmaxf(m0, d0);
      const float c0 = (sum0 > 0.0f) ? __builtin_expf(m0 - mn0) : 0.0f;
      const float e0 = __builtin_expf(d0 - mn0);
      sum0 = sum0 * c0 + e0;
      m0 = mn0;

      const float mn1 = __builtin_fmaxf(m1, d1);
      const float c1 = (sum1 > 0.0f) ? __builtin_expf(m1 - mn1) : 0.0f;
      const float e1 = __builtin_expf(d1 - mn1);
      sum1 = sum1 * c1 + e1;
      m1 = mn1;

      const float mn2 = __builtin_fmaxf(m2, d2);
      const float c2 = (sum2 > 0.0f) ? __builtin_expf(m2 - mn2) : 0.0f;
      const float e2 = __builtin_expf(d2 - mn2);
      sum2 = sum2 * c2 + e2;
      m2 = mn2;

      const f32x4 vf = __builtin_convertvector(vh4, f32x4);
      acc0 = acc0 * c0 + e0 * vf;
      acc1 = acc1 * c1 + e1 * vf;
      acc2 = acc2 * c2 + e2 * vf;

      if (kp_nxt >= p1) break;
      kp = kp_nxt;
      kvp = kvp_nxt;
      kh4 = kh4_nxt;
      vh4 = vh4_nxt;
    }
  }

  if (l == 0u) {
    ink_r10_df_dg_m[0][w] = m0; ink_r10_df_dg_l[0][w] = sum0;
    ink_r10_df_dg_m[1][w] = m1; ink_r10_df_dg_l[1][w] = sum1;
    ink_r10_df_dg_m[2][w] = m2; ink_r10_df_dg_l[2][w] = sum2;
  }
  ((LDSP f32x4 *)&ink_r10_df_dg_o[0][w][0])[l] = acc0;
  ((LDSP f32x4 *)&ink_r10_df_dg_o[1][w][0])[l] = acc1;
  ((LDSP f32x4 *)&ink_r10_df_dg_o[2][w][0])[l] = acc2;
  lbarrier();

  const u32 nb_q = n_head << 2u;
  if (nsplit == 1u) {
    /* 384 elements = 12 exact wave32 Q8_0 blocks (0 cross-wave deps!). */
    for (u32 idx = tid(); idx < INK_R10_DF_GQA * INK_R10_DF_HD; idx += NT) {
      const u32 g = idx >> 7u;
      const u32 e = idx & 127u;
      const u32 h = h0 + g;
      const float sv = sink[h];
      float M = sv;
      for (u32 x = 0u; x < 8u; x++)
        if (ink_r10_df_dg_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r10_df_dg_m[g][x]);
      float L = __builtin_expf(sv - M);
      float o = 0.0f;
      for (u32 x = 0u; x < 8u; x++) {
        const float lx = ink_r10_df_dg_l[g][x];
        if (lx > 0.0f) {
          const float wx = __builtin_expf(ink_r10_df_dg_m[g][x] - M);
          L += lx * wx;
          o += ink_r10_df_dg_o[g][x][e] * wx;
        }
      }
      const u32 out_idx = h * INK_R10_DF_HD + e;
      float v = o / L;
      if (agate) v = ink_r10_df_gate_f(v, agate[out_idx]);
      if (out) out[out_idx] = v;
      if (oq) quant_wave(v, out_idx >> 5u, oq, nb_q);
    }
    return;
  }

  for (u32 idx = tid(); idx < INK_R10_DF_GQA * INK_R10_DF_HD; idx += NT) {
    const u32 g = idx >> 7u;
    const u32 e = idx & 127u;
    const u32 h = h0 + g;
    float M = -__builtin_inff();
    for (u32 x = 0u; x < 8u; x++)
      if (ink_r10_df_dg_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r10_df_dg_m[g][x]);
    G float *pp = part + (unsigned long)(h * nsplit + s) * INK_R10_DF_PSTR;
    float o = 0.0f;
    float L = 0.0f;
    for (u32 x = 0u; x < 8u; x++) {
      const float lx = ink_r10_df_dg_l[g][x];
      if (lx > 0.0f) {
        const float wx = __builtin_expf(ink_r10_df_dg_m[g][x] - M);
        o += ink_r10_df_dg_o[g][x][e] * wx;
        if (e == 0u) L += lx * wx;
      }
    }
    pp[2u + e] = o;
    if (e == 0u) {
      pp[0] = M;
      pp[1] = L;
    }
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r10_df_dg_last =
        (__scoped_atomic_fetch_add(&count[kh], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u);
  }
  barrier();
  if (!ink_r10_df_dg_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  G float *ph0 = part + (unsigned long)(h0 + 0u) * nsplit * INK_R10_DF_PSTR;
  G float *ph1 = part + (unsigned long)(h0 + 1u) * nsplit * INK_R10_DF_PSTR;
  G float *ph2 = part + (unsigned long)(h0 + 2u) * nsplit * INK_R10_DF_PSTR;

  float mv0 = (tid() == 0u) ? sink[h0 + 0u] : -__builtin_inff();
  float mv1 = (tid() == 0u) ? sink[h0 + 1u] : -__builtin_inff();
  float mv2 = (tid() == 0u) ? sink[h0 + 2u] : -__builtin_inff();
  for (u32 t = tid(); t < nsplit; t += NT) {
    const unsigned long off = (unsigned long)t * INK_R10_DF_PSTR;
    if (ph0[off + 1u] > 0.0f) mv0 = __builtin_fmaxf(mv0, ph0[off]);
    if (ph1[off + 1u] > 0.0f) mv1 = __builtin_fmaxf(mv1, ph1[off]);
    if (ph2[off + 1u] > 0.0f) mv2 = __builtin_fmaxf(mv2, ph2[off]);
  }
  for (u32 mask = 16u; mask >= 1u; mask >>= 1u) {
    mv0 = __builtin_fmaxf(mv0, shfl_xor_f(mv0, mask));
    mv1 = __builtin_fmaxf(mv1, shfl_xor_f(mv1, mask));
    mv2 = __builtin_fmaxf(mv2, shfl_xor_f(mv2, mask));
  }
  if (l == 0u) {
    ink_r10_df_dg_m[0][w] = mv0;
    ink_r10_df_dg_m[1][w] = mv1;
    ink_r10_df_dg_m[2][w] = mv2;
  }
  lbarrier();

  float M2[INK_R10_DF_GQA];
  for (u32 g = 0u; g < INK_R10_DF_GQA; g++) {
    float mx = ink_r10_df_dg_m[g][0];
    for (u32 x = 1u; x < 8u; x++) mx = __builtin_fmaxf(mx, ink_r10_df_dg_m[g][x]);
    M2[g] = mx;
  }

  const int wt_in_lds = (nsplit <= INK_R10_DF_HD);
  float lv0 = 0.0f, lv1 = 0.0f, lv2 = 0.0f;
  for (u32 t = tid(); t < nsplit; t += NT) {
    const unsigned long off = (unsigned long)t * INK_R10_DF_PSTR;
    const float lp0 = ph0[off + 1u];
    const float wt0 = (lp0 > 0.0f) ? __builtin_expf(ph0[off] - M2[0]) : 0.0f;
    if (wt_in_lds) ink_r10_df_dg_o[0][0][t] = wt0;
    else ph0[off] = wt0;
    lv0 += lp0 * wt0;

    const float lp1 = ph1[off + 1u];
    const float wt1 = (lp1 > 0.0f) ? __builtin_expf(ph1[off] - M2[1]) : 0.0f;
    if (wt_in_lds) ink_r10_df_dg_o[1][0][t] = wt1;
    else ph1[off] = wt1;
    lv1 += lp1 * wt1;

    const float lp2 = ph2[off + 1u];
    const float wt2 = (lp2 > 0.0f) ? __builtin_expf(ph2[off] - M2[2]) : 0.0f;
    if (wt_in_lds) ink_r10_df_dg_o[2][0][t] = wt2;
    else ph2[off] = wt2;
    lv2 += lp2 * wt2;
  }
  lv0 = wave_tree_f(lv0);
  lv1 = wave_tree_f(lv1);
  lv2 = wave_tree_f(lv2);
  if (l == 0u) {
    ink_r10_df_dg_l[0][w] = lv0;
    ink_r10_df_dg_l[1][w] = lv1;
    ink_r10_df_dg_l[2][w] = lv2;
  }
  if (wt_in_lds) lbarrier();
  else barrier();

  float L2[INK_R10_DF_GQA];
  for (u32 g = 0u; g < INK_R10_DF_GQA; g++) {
    float ls = 0.0f;
    for (u32 x = 0u; x < 8u; x++) ls += ink_r10_df_dg_l[g][x];
    L2[g] = ls + __builtin_expf(sink[h0 + g] - M2[g]);
  }

  for (u32 idx = tid(); idx < INK_R10_DF_GQA * INK_R10_DF_HD; idx += NT) {
    const u32 g = idx >> 7u;
    const u32 e = idx & 127u;
    const u32 h = h0 + g;
    const G float *ph = part + (unsigned long)h * nsplit * INK_R10_DF_PSTR;
    float o = 0.0f;
    if (wt_in_lds) {
      for (u32 t = 0u; t < nsplit; t++) o += ph[(unsigned long)t * INK_R10_DF_PSTR + 2u + e] * ink_r10_df_dg_o[g][0][t];
    } else {
      for (u32 t = 0u; t < nsplit; t++) o += ph[(unsigned long)t * INK_R10_DF_PSTR + 2u + e] * ph[(unsigned long)t * INK_R10_DF_PSTR];
    }
    const u32 out_idx = h * INK_R10_DF_HD + e;
    float v = o / L2[g];
    if (agate) v = ink_r10_df_gate_f(v, agate[out_idx]);
    if (out) out[out_idx] = v;
    if (oq) quant_wave(v, out_idx >> 5u, oq, nb_q);
  }

  if (tid() == 0u) count[kh] = 0u;
}

/* ============================================================================
 * Kernel 4 (H15-Ver-SWA): ie_inkling_rel_attn_ver_gate_q8_r10
 * Exact-order drop-in upgrade of ie_inkling_rel_attn_ver_r4 (T=2..16 SWA &
 * short Global) fusing OP_GATE (`out * sigmoid(agate)`) + Q8_0 quantization.
 * Grid: (n_kv * nsplit, T, 1u), <= 72 VGPR (4 WGs/CU), 12,484 B LDS.
 * ========================================================================== */

KERNEL ie_inkling_rel_attn_ver_gate_q8_r10(
    const G float *q, const G kvt *kc, const G kvt *vc, const G float *R,
    const G float *rel_bias, const G float *sink, G float *part, G float *out,
    u32 pos0, u32 T, u32 kv_len, u32 hd, u32 n_head, u32 n_kv, u32 d_rel,
    u32 rel_extent, u32 win, u32 nsplit, u32 qs, u32 rs, u32 ss, u32 os,
    G u32 *count,
    const G float *agate, G u8 *oq, u32 gs, u32 oqs) {
  static LDS float ink_r10_df_vg_m[INK_R10_DF_GQA][8];
  static LDS float ink_r10_df_vg_l[INK_R10_DF_GQA][8];
  static LDS __attribute__((aligned(16))) float ink_r10_df_vg_o[INK_R10_DF_GQA][8][INK_R10_DF_HD];
  static LDS u32 ink_r10_df_vg_last;

  const u32 t = tokid();
  if (t >= T || hd != INK_R10_DF_HD || d_rel != INK_R10_DF_DREL || n_kv == 0u ||
      n_head != INK_R10_DF_GQA * n_kv || kv_len == 0u || nsplit == 0u)
    return;

  const u32 kh = wgid() / nsplit;
  const u32 s = wgid() % nsplit;
  const u32 h0 = kh * INK_R10_DF_GQA;
  const u32 kvd = n_kv * INK_R10_DF_HD;
  const u32 w = tid() >> 5u;
  const u32 l = lane();

  const u32 pos = pos0 + t;
  const u32 pend = pos + 1u;
  const u32 w0 = (win > 0u && pend > win) ? (pend - win) : 0u;
  const u32 np = pend - w0;
  const u32 ch = (np + nsplit - 1u) / nsplit;
  const u32 off0 = s * ch;
  const u32 p0 = (off0 < np) ? (w0 + off0) : pend;
  const u32 p1 = (off0 < np && off0 + ch < np) ? (w0 + off0 + ch) : pend;

  const G float *qt = (const G float *)((const G u8 *)q + (unsigned long)t * qs);
  const G float *Rt = (const G float *)((const G u8 *)R + (unsigned long)t * rs);
  const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
  G float *ot = out ? (G float *)((G u8 *)out + (unsigned long)t * os) : (G float *)0;
  const G float *gt = agate ? (const G float *)((const G u8 *)agate + (unsigned long)t * gs) : (const G float *)0;
  G u8 *oqt = oq ? (oq + (unsigned long)t * oqs) : (G u8 *)0;
  const u32 nb_q = n_head << 2u;

  if (nsplit > 1u && p0 >= p1) {
    for (u32 idx = tid(); idx < INK_R10_DF_GQA * INK_R10_DF_HD; idx += NT) {
      const u32 g = idx >> 7u;
      const u32 e = idx & 127u;
      const u32 h = h0 + g;
      G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R10_DF_PSTR;
      pp[2u + e] = 0.0f;
      if (e == 0u) {
        pp[0] = -1e30f;
        pp[1] = 0.0f;
      }
    }
  } else {
    const u32 extent_lim = (win > 0u && win < rel_extent) ? win : rel_extent;
    const float scale = 1.0f / __builtin_sqrtf((float)INK_R10_DF_HD);

    const f32x4 q0 = ((const G f32x4 *)(qt + (unsigned long)(h0 + 0u) * INK_R10_DF_HD))[l];
    const f32x4 q1 = ((const G f32x4 *)(qt + (unsigned long)(h0 + 1u) * INK_R10_DF_HD))[l];
    const f32x4 q2 = ((const G f32x4 *)(qt + (unsigned long)(h0 + 2u) * INK_R10_DF_HD))[l];

    const float rv0 = (l < INK_R10_DF_DREL) ? Rt[(unsigned long)(h0 + 0u) * INK_R10_DF_DREL + l] : 0.0f;
    const float rv1 = (l < INK_R10_DF_DREL) ? Rt[(unsigned long)(h0 + 1u) * INK_R10_DF_DREL + l] : 0.0f;
    const float rv2 = (l < INK_R10_DF_DREL) ? Rt[(unsigned long)(h0 + 2u) * INK_R10_DF_DREL + l] : 0.0f;

    f32x4 acc0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 acc2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};

    float m0 = -__builtin_inff(), sum0 = 0.0f;
    float m1 = -__builtin_inff(), sum1 = 0.0f;
    float m2 = -__builtin_inff(), sum2 = 0.0f;

    const G kvt *kc_h = kc + (unsigned long)kh * INK_R10_DF_HD;
    const G kvt *vc_h = vc + (unsigned long)kh * INK_R10_DF_HD;

    u32 kp = p0 + w;
    if (kp < p1) {
      u32 kvp = kp % kv_len;
      const G kv4 *krow = (const G kv4 *)(kc_h + (unsigned long)kvp * kvd);
      const G kv4 *vrow = (const G kv4 *)(vc_h + (unsigned long)kvp * kvd);
      kv4 kh4 = krow[l];
      kv4 vh4 = vrow[l];

      for (;;) {
        const u32 dist = pos - kp;
        const int has_rel = (dist < extent_lim);
        float rb = 0.0f;
        if (has_rel && l < INK_R10_DF_DREL) rb = rel_bias[(unsigned long)dist * INK_R10_DF_DREL + l];

        const u32 kp_nxt = kp + 8u;
        u32 kvp_nxt = kvp;
        kv4 kh4_nxt = kh4;
        kv4 vh4_nxt = vh4;
        if (kp_nxt < p1) {
          kvp_nxt = kvp + 8u;
          if (kvp_nxt >= kv_len) kvp_nxt = (kv_len >= 8u) ? (kvp_nxt - kv_len) : (kp_nxt % kv_len);
          const G kv4 *krow_n = (const G kv4 *)(kc_h + (unsigned long)kvp_nxt * kvd);
          const G kv4 *vrow_n = (const G kv4 *)(vc_h + (unsigned long)kvp_nxt * kvd);
          kh4_nxt = krow_n[l];
          vh4_nxt = vrow_n[l];
        }

        const f32x4 kf = __builtin_convertvector(kh4, f32x4);
        float d0 = wave_sum_all(ink_r10_df_dot4(q0, kf)) * scale;
        float d1 = wave_sum_all(ink_r10_df_dot4(q1, kf)) * scale;
        float d2 = wave_sum_all(ink_r10_df_dot4(q2, kf)) * scale;

        if (has_rel) {
          d0 += ink_r10_df_halfwave_sum_all(rv0 * rb);
          d1 += ink_r10_df_halfwave_sum_all(rv1 * rb);
          d2 += ink_r10_df_halfwave_sum_all(rv2 * rb);
        }

        const float mn0 = __builtin_fmaxf(m0, d0);
        const float c0 = (sum0 > 0.0f) ? __builtin_expf(m0 - mn0) : 0.0f;
        const float e0 = __builtin_expf(d0 - mn0);
        sum0 = sum0 * c0 + e0;
        m0 = mn0;

        const float mn1 = __builtin_fmaxf(m1, d1);
        const float c1 = (sum1 > 0.0f) ? __builtin_expf(m1 - mn1) : 0.0f;
        const float e1 = __builtin_expf(d1 - mn1);
        sum1 = sum1 * c1 + e1;
        m1 = mn1;

        const float mn2 = __builtin_fmaxf(m2, d2);
        const float c2 = (sum2 > 0.0f) ? __builtin_expf(m2 - mn2) : 0.0f;
        const float e2 = __builtin_expf(d2 - mn2);
        sum2 = sum2 * c2 + e2;
        m2 = mn2;

        const f32x4 vf = __builtin_convertvector(vh4, f32x4);
        acc0 = acc0 * c0 + e0 * vf;
        acc1 = acc1 * c1 + e1 * vf;
        acc2 = acc2 * c2 + e2 * vf;

        if (kp_nxt >= p1) break;
        kp = kp_nxt;
        kvp = kvp_nxt;
        kh4 = kh4_nxt;
        vh4 = vh4_nxt;
      }
    }

    if (l == 0u) {
      ink_r10_df_vg_m[0][w] = m0; ink_r10_df_vg_l[0][w] = sum0;
      ink_r10_df_vg_m[1][w] = m1; ink_r10_df_vg_l[1][w] = sum1;
      ink_r10_df_vg_m[2][w] = m2; ink_r10_df_vg_l[2][w] = sum2;
    }
    ((LDSP f32x4 *)&ink_r10_df_vg_o[0][w][0])[l] = acc0;
    ((LDSP f32x4 *)&ink_r10_df_vg_o[1][w][0])[l] = acc1;
    ((LDSP f32x4 *)&ink_r10_df_vg_o[2][w][0])[l] = acc2;
    lbarrier();

    if (nsplit == 1u) {
      for (u32 idx = tid(); idx < INK_R10_DF_GQA * INK_R10_DF_HD; idx += NT) {
        const u32 g = idx >> 7u;
        const u32 e = idx & 127u;
        const u32 h = h0 + g;
        const float sv = st[h];
        float M = sv;
        for (u32 x = 0u; x < 8u; x++)
          if (ink_r10_df_vg_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r10_df_vg_m[g][x]);
        float L = __builtin_expf(sv - M);
        float o = 0.0f;
        for (u32 x = 0u; x < 8u; x++) {
          const float lx = ink_r10_df_vg_l[g][x];
          if (lx > 0.0f) {
            const float wx = __builtin_expf(ink_r10_df_vg_m[g][x] - M);
            L += lx * wx;
            o += ink_r10_df_vg_o[g][x][e] * wx;
          }
        }
        const u32 out_idx = h * INK_R10_DF_HD + e;
        float v = o / L;
        if (gt) v = ink_r10_df_gate_f(v, gt[out_idx]);
        if (ot) ot[out_idx] = v;
        if (oqt) quant_wave(v, out_idx >> 5u, oqt, nb_q);
      }
      return;
    }

    for (u32 idx = tid(); idx < INK_R10_DF_GQA * INK_R10_DF_HD; idx += NT) {
      const u32 g = idx >> 7u;
      const u32 e = idx & 127u;
      const u32 h = h0 + g;
      float M = -__builtin_inff();
      for (u32 x = 0u; x < 8u; x++)
        if (ink_r10_df_vg_l[g][x] > 0.0f) M = __builtin_fmaxf(M, ink_r10_df_vg_m[g][x]);
      G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R10_DF_PSTR;
      float o = 0.0f;
      float L = 0.0f;
      for (u32 x = 0u; x < 8u; x++) {
        const float lx = ink_r10_df_vg_l[g][x];
        if (lx > 0.0f) {
          const float wx = __builtin_expf(ink_r10_df_vg_m[g][x] - M);
          o += ink_r10_df_vg_o[g][x][e] * wx;
          if (e == 0u) L += lx * wx;
        }
      }
      pp[2u + e] = o;
      if (e == 0u) {
        pp[0] = M;
        pp[1] = L;
      }
    }
  }

  const u32 cidx = t * n_kv + kh;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r10_df_vg_last =
        (__scoped_atomic_fetch_add(&count[cidx], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u);
  }
  barrier();
  if (!ink_r10_df_vg_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  G float *ph0 = part + (unsigned long)((t * n_head + (h0 + 0u)) * nsplit) * INK_R10_DF_PSTR;
  G float *ph1 = part + (unsigned long)((t * n_head + (h0 + 1u)) * nsplit) * INK_R10_DF_PSTR;
  G float *ph2 = part + (unsigned long)((t * n_head + (h0 + 2u)) * nsplit) * INK_R10_DF_PSTR;

  float mv0 = (tid() == 0u) ? st[h0 + 0u] : -__builtin_inff();
  float mv1 = (tid() == 0u) ? st[h0 + 1u] : -__builtin_inff();
  float mv2 = (tid() == 0u) ? st[h0 + 2u] : -__builtin_inff();
  for (u32 sp = tid(); sp < nsplit; sp += NT) {
    const unsigned long off = (unsigned long)sp * INK_R10_DF_PSTR;
    if (ph0[off + 1u] > 0.0f) mv0 = __builtin_fmaxf(mv0, ph0[off]);
    if (ph1[off + 1u] > 0.0f) mv1 = __builtin_fmaxf(mv1, ph1[off]);
    if (ph2[off + 1u] > 0.0f) mv2 = __builtin_fmaxf(mv2, ph2[off]);
  }
  for (u32 mask = 16u; mask >= 1u; mask >>= 1u) {
    mv0 = __builtin_fmaxf(mv0, shfl_xor_f(mv0, mask));
    mv1 = __builtin_fmaxf(mv1, shfl_xor_f(mv1, mask));
    mv2 = __builtin_fmaxf(mv2, shfl_xor_f(mv2, mask));
  }
  if (l == 0u) {
    ink_r10_df_vg_m[0][w] = mv0;
    ink_r10_df_vg_m[1][w] = mv1;
    ink_r10_df_vg_m[2][w] = mv2;
  }
  lbarrier();

  float M2[INK_R10_DF_GQA];
  for (u32 g = 0u; g < INK_R10_DF_GQA; g++) {
    float mx = ink_r10_df_vg_m[g][0];
    for (u32 x = 1u; x < 8u; x++) mx = __builtin_fmaxf(mx, ink_r10_df_vg_m[g][x]);
    M2[g] = mx;
  }

  const int wt_in_lds = (nsplit <= INK_R10_DF_HD);
  float lv0 = 0.0f, lv1 = 0.0f, lv2 = 0.0f;
  for (u32 sp = tid(); sp < nsplit; sp += NT) {
    const unsigned long off = (unsigned long)sp * INK_R10_DF_PSTR;
    const float lp0 = ph0[off + 1u];
    const float wt0 = (lp0 > 0.0f) ? __builtin_expf(ph0[off] - M2[0]) : 0.0f;
    if (wt_in_lds) ink_r10_df_vg_o[0][0][sp] = wt0;
    else ph0[off] = wt0;
    lv0 += lp0 * wt0;

    const float lp1 = ph1[off + 1u];
    const float wt1 = (lp1 > 0.0f) ? __builtin_expf(ph1[off] - M2[1]) : 0.0f;
    if (wt_in_lds) ink_r10_df_vg_o[1][0][sp] = wt1;
    else ph1[off] = wt1;
    lv1 += lp1 * wt1;

    const float lp2 = ph2[off + 1u];
    const float wt2 = (lp2 > 0.0f) ? __builtin_expf(ph2[off] - M2[2]) : 0.0f;
    if (wt_in_lds) ink_r10_df_vg_o[2][0][sp] = wt2;
    else ph2[off] = wt2;
    lv2 += lp2 * wt2;
  }
  lv0 = wave_tree_f(lv0);
  lv1 = wave_tree_f(lv1);
  lv2 = wave_tree_f(lv2);
  if (l == 0u) {
    ink_r10_df_vg_l[0][w] = lv0;
    ink_r10_df_vg_l[1][w] = lv1;
    ink_r10_df_vg_l[2][w] = lv2;
  }
  if (wt_in_lds) lbarrier();
  else barrier();

  float L2[INK_R10_DF_GQA];
  for (u32 g = 0u; g < INK_R10_DF_GQA; g++) {
    float ls = 0.0f;
    for (u32 x = 0u; x < 8u; x++) ls += ink_r10_df_vg_l[g][x];
    L2[g] = ls + __builtin_expf(st[h0 + g] - M2[g]);
  }

  for (u32 idx = tid(); idx < INK_R10_DF_GQA * INK_R10_DF_HD; idx += NT) {
    const u32 g = idx >> 7u;
    const u32 e = idx & 127u;
    const u32 h = h0 + g;
    const G float *ph = part + (unsigned long)((t * n_head + h) * nsplit) * INK_R10_DF_PSTR;
    float o = 0.0f;
    if (wt_in_lds) {
      for (u32 sp = 0u; sp < nsplit; sp++) o += ph[(unsigned long)sp * INK_R10_DF_PSTR + 2u + e] * ink_r10_df_vg_o[g][0][sp];
    } else {
      for (u32 sp = 0u; sp < nsplit; sp++) o += ph[(unsigned long)sp * INK_R10_DF_PSTR + 2u + e] * ph[(unsigned long)sp * INK_R10_DF_PSTR];
    }
    const u32 out_idx = h * INK_R10_DF_HD + e;
    float v = o / L2[g];
    if (gt) v = ink_r10_df_gate_f(v, gt[out_idx]);
    if (ot) ot[out_idx] = v;
    if (oqt) quant_wave(v, out_idx >> 5u, oqt, nb_q);
  }

  if (tid() == 0u) count[cidx] = 0u;
}

/* ============================================================================
 * Kernel 5 (H15-Ver-Global-WMMA): ie_inkling_rel_attn_ver_glob_wmma_gate_q8_r10
 * Exact-order drop-in upgrade of ie_inkling_rel_attn_ver_glob_wmma_r7 (T=2..8
 * Global) fusing OP_GATE (`out * sigmoid(agate)`) + Q8_0 quantization right
 * inside the final f32x4 merge wave (0 extra LDS, exact 15,360 B <= 16 KiB,
 * <= 128 VGPR = 3 WGs/CU!).
 * Grid: (n_kv * nsplit, 1u, 1u), NT = 256.
 * ========================================================================== */

typedef union {
  struct {
    f16 q[32u * 128u] __attribute__((aligned(16)));
    f16 r[32u * 16u] __attribute__((aligned(16)));
    f16 p_tile[2u][2u][256u] __attribute__((aligned(16)));
    union {
      u32 v_sc[8u][128u] __attribute__((aligned(16)));
      struct {
        float mx[2u][128u];
        float ls[2u][128u];
      } red;
    } scratch;
  } kv;
  struct {
    float o_stg[8u][128u] __attribute__((aligned(16)));
    float m[32u];
    float l[32u];
    float wt[8u][128u] __attribute__((aligned(16)));
    u32   is_last;
  } ph4;
} ink_r10_df_gw_lds_t;

static LDS ink_r10_df_gw_lds_t ink_r10_df_gw_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r10_df_gw_lds_t) == 15360u, "ink_r10_df_gw_lds_t must be 15,360 B");

static inline __attribute__((always_inline)) u32 ink_r10_df_gw_q_off(u32 row, u32 c8) {
  return row * 128u + 8u * (c8 ^ (row & 15u));
}

static inline __attribute__((always_inline)) u32 ink_r10_df_gw_p_off(u32 rl, u32 hl) {
  return rl * 16u + 8u * (hl ^ ((rl >> 2u) & 1u));
}

static inline __attribute__((always_inline)) h8 ink_r10_df_gw_v_transpose(
    const u32x4 vr, LDSP u32 *v_sc_w, const u32 hl, const u32 rl) {
  const u32 sel = (rl & 1u) ? 0x03020706u : 0x05040100u;
  const u32 kh = rl >> 3u;
  const u32 k4 = (rl >> 1u) & 3u;
  const u32 w_arr[4] = {vr.x, vr.y, vr.z, vr.w};
#pragma unroll
  for (u32 q = 0u; q < 4u; q++) {
    const u32 x = w_arr[q];
    const u32 y = (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0xB1, 0xF, 0xF, true);
    const u32 t = __builtin_amdgcn_perm(y, x, sel);
    const u32 d = 8u * hl + 2u * q + (rl & 1u);
    const u32 s4 = ((d & 7u) ^ ((d >> 3u) << 1u) ^ (kh << 2u)) + ((d >> 3u) << 3u) + (kh << 4u);
    v_sc_w[s4 * 4u + k4] = t;
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  const u32 s4_r = ((rl & 7u) ^ ((rl >> 3u) << 1u) ^ (hl << 2u)) + ((rl >> 3u) << 3u) + (hl << 4u);
  return *(const LDSP h8 *)(v_sc_w + s4_r * 4u);
}

static inline __attribute__((always_inline)) void ink_r10_df_gw_apply_rb16(
    const v8f rb, const u32 base_off, const u32 hl, const u32 kb, const u32 pos_q,
    const u32 rel_extent, const u32 d_base, LDSP float *sc_row, v8f *sv) {
  if (hl == 0u) {
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) sc_row[v] = rb[v];
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 v = 0u; v < 8u; v++) {
    const u32 kp = kb + 8u * hl + v;
    if (kp <= pos_q) {
      const u32 dist = pos_q - kp;
      if (dist < rel_extent) {
        const u32 rel = (dist - d_base) - base_off;
        if (rel < 8u) (*sv)[v] += sc_row[rel];
      }
    }
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  if (hl == 1u) {
#pragma unroll
    for (u32 v = 0u; v < 8u; v++) sc_row[v] = rb[v];
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 v = 0u; v < 8u; v++) {
    const u32 kp = kb + 8u * hl + v;
    if (kp <= pos_q) {
      const u32 dist = pos_q - kp;
      if (dist < rel_extent) {
        const u32 rel = (dist - d_base) - (base_off + 8u);
        if (rel < 8u) (*sv)[v] += sc_row[rel];
      }
    }
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

static inline __attribute__((always_inline)) void ink_r10_df_gw_merge_head_gate_q8(
    const G float *ph, const float sink_val, const u32 nsplit, const u32 l,
    LDSP float *wt_w, const G float *gt_h, G float *ot_h, G u8 *oqt, u32 h, u32 n_head) {
  float mv = (l == 0u) ? sink_val : -__builtin_inff();
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const unsigned long off = (unsigned long)sp * INK_R10_DF_PSTR;
    if (ph[off + 1u] > 0.0f) mv = __builtin_fmaxf(mv, ph[off]);
  }
#pragma unroll
  for (u32 mask = 16u; mask >= 1u; mask >>= 1u) mv = __builtin_fmaxf(mv, shfl_xor_f(mv, mask));
  const float M2 = ink_r10_df_sgpr_f(mv);

  float lv = 0.0f;
  for (u32 sp = l; sp < nsplit; sp += 32u) {
    const unsigned long off = (unsigned long)sp * INK_R10_DF_PSTR;
    const float lp = ph[off + 1u];
    const float wt = (lp > 0.0f) ? __builtin_expf(ph[off] - M2) : 0.0f;
    if (sp < 128u) wt_w[sp] = wt;
    lv += lp * wt;
  }
  const float L2 = ink_r10_df_sgpr_f(wave_sum_all(lv) + __builtin_expf(sink_val - M2));
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");

  f32x4 acc = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  for (u32 sp = 0u; sp < nsplit; sp++) {
    const float wt = (sp < 128u) ? wt_w[sp]
                                 : ((ph[(unsigned long)sp * INK_R10_DF_PSTR + 1u] > 0.0f)
                                        ? __builtin_expf(ph[(unsigned long)sp * INK_R10_DF_PSTR] - M2)
                                        : 0.0f);
    if (wt > 0.0f) {
      const G float *po = ph + (unsigned long)sp * INK_R10_DF_PSTR + 2u + 4u * l;
      const f32x4 ov = {po[0], po[1], po[2], po[3]};
      acc += ov * wt;
    }
  }
  const float inv = 1.0f / L2;
  ink_r10_df_gate_q8_head_vec4(acc * inv, gt_h, ot_h, oqt, h, n_head, l);
}

KERNEL ie_inkling_rel_attn_ver_glob_wmma_gate_q8_r10(
    const G float *q, const G kvt *kc, const G kvt *vc,
    const G float *R, const G float *rel_bias, const G float *sink,
    G float *part, G float *out, u32 pos0, u32 T, u32 kv_len, u32 hd,
    u32 n_head, u32 n_kv, u32 d_rel, u32 rel_extent, u32 win, u32 nsplit,
    u32 qs, u32 rs, u32 ss, u32 os, G u32 *count,
    const G float *agate, G u8 *oq, u32 gs, u32 oqs) {
  if (T == 0u || T > 8u || win != 0u || hd != INK_R10_DF_HD || d_rel != INK_R10_DF_DREL || n_kv == 0u ||
      n_head != INK_R10_DF_GQA * n_kv || kv_len == 0u || nsplit == 0u)
    return;

  const u32 kh = __builtin_amdgcn_readfirstlane(wgid() / nsplit);
  const u32 s = __builtin_amdgcn_readfirstlane(wgid() % nsplit);
  if (kh >= n_kv) return;

  const u32 h0 = __builtin_amdgcn_readfirstlane(kh * INK_R10_DF_GQA);
  const u32 kvd = __builtin_amdgcn_readfirstlane(n_kv * INK_R10_DF_HD);
  const u32 w = __builtin_amdgcn_readfirstlane(tid() >> 5u);
  const u32 l = lane(), hl = l >> 4u, rl = l & 15u;

  const u32 nq_tot = __builtin_amdgcn_readfirstlane(T * INK_R10_DF_GQA);
  const u32 npg = __builtin_amdgcn_readfirstlane(nq_tot > 16u ? 2u : 1u);
  const u32 np = __builtin_amdgcn_readfirstlane(pos0 + T);
  const u32 ch = __builtin_amdgcn_readfirstlane((np + nsplit - 1u) / nsplit);
  const u32 off0 = __builtin_amdgcn_readfirstlane(s * ch);
  const u32 p0 = __builtin_amdgcn_readfirstlane((off0 < np) ? off0 : np);
  const u32 p1 = __builtin_amdgcn_readfirstlane((off0 < np && off0 + ch < np) ? (off0 + ch) : np);

  if (nsplit > 1u && p0 >= p1) {
    for (u32 qi = w; qi < nq_tot; qi += 8u) {
      const u32 t = qi / INK_R10_DF_GQA;
      const u32 h = h0 + (qi - t * INK_R10_DF_GQA);
      G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R10_DF_PSTR;
      if (l == 0u) {
        pp[0] = -1e30f;
        pp[1] = 0.0f;
      }
      pp[2u + 4u * l + 0u] = 0.0f;
      pp[2u + 4u * l + 1u] = 0.0f;
      pp[2u + 4u * l + 2u] = 0.0f;
      pp[2u + 4u * l + 3u] = 0.0f;
    }
  } else {
    const int far_split = __builtin_amdgcn_readfirstlane((p1 <= pos0 + 1u) && ((pos0 + 1u - p1) >= rel_extent));

    const u32 q_rows = npg * 16u;
    for (u32 idx = tid(); idx < q_rows * 16u; idx += NT) {
      const u32 qr = idx >> 4u;
      const u32 c8 = idx & 15u;
      h8 qh = (h8)(0.0f);
      if (qr < nq_tot) {
        const u32 tj = qr / INK_R10_DF_GQA;
        const u32 gi = qr - tj * INK_R10_DF_GQA;
        const G f32x4 *qp = (const G f32x4 *)((const G float *)((const G u8 *)q + (unsigned long)tj * qs) +
                                              (unsigned long)(h0 + gi) * INK_R10_DF_HD);
        const f32x4 v0 = qp[2u * c8 + 0u];
        const f32x4 v1 = qp[2u * c8 + 1u];
        qh = (h8){(f16)(v0.x * INK_R10_DF_QSC), (f16)(v0.y * INK_R10_DF_QSC),
                  (f16)(v0.z * INK_R10_DF_QSC), (f16)(v0.w * INK_R10_DF_QSC),
                  (f16)(v1.x * INK_R10_DF_QSC), (f16)(v1.y * INK_R10_DF_QSC),
                  (f16)(v1.z * INK_R10_DF_QSC), (f16)(v1.w * INK_R10_DF_QSC)};
      }
      *(LDSP h8 *)(ink_r10_df_gw_lds.kv.q + ink_r10_df_gw_q_off(qr, c8)) = qh;
    }
    if (!far_split && rel_extent > 0u) {
      if (tid() < q_rows * 2u) {
        const u32 qr = tid() >> 1u;
        const u32 h8i = tid() & 1u;
        h8 rh = (h8)(0.0f);
        if (qr < nq_tot) {
          const u32 tj = qr / INK_R10_DF_GQA;
          const u32 gi = qr - tj * INK_R10_DF_GQA;
          const G f32x4 *rp = (const G f32x4 *)((const G float *)((const G u8 *)R + (unsigned long)tj * rs) +
                                                (unsigned long)(h0 + gi) * INK_R10_DF_DREL);
          const f32x4 v0 = rp[2u * h8i + 0u];
          const f32x4 v1 = rp[2u * h8i + 1u];
          rh = (h8){(f16)(v0.x * INK_R10_DF_LOG2E), (f16)(v0.y * INK_R10_DF_LOG2E),
                    (f16)(v0.z * INK_R10_DF_LOG2E), (f16)(v0.w * INK_R10_DF_LOG2E),
                    (f16)(v1.x * INK_R10_DF_LOG2E), (f16)(v1.y * INK_R10_DF_LOG2E),
                    (f16)(v1.z * INK_R10_DF_LOG2E), (f16)(v1.w * INK_R10_DF_LOG2E)};
        }
        *(LDSP h8 *)(ink_r10_df_gw_lds.kv.r + qr * 16u + 8u * h8i) = rh;
      }
    }
    lbarrier();

    const G kvt *kc_h = kc + (unsigned long)kh * INK_R10_DF_HD;
    const G kvt *vc_h = vc + (unsigned long)kh * INK_R10_DF_HD;
    const int no_wrap = __builtin_amdgcn_readfirstlane(p1 <= kv_len);

    const v8f zero = (v8f)(0.0f);
    v8f acc0 = zero, acc1 = zero;
    float m0 = -__builtin_inff(), ls0 = 0.0f;
    float m1 = -__builtin_inff(), ls1 = 0.0f;

    const u32 p_off = ink_r10_df_gw_p_off(rl, hl);
    const u32 pos_q0 = pos0 + (rl / INK_R10_DF_GQA);
    const u32 pos_q1 = pos0 + ((16u + rl) / INK_R10_DF_GQA);

#pragma clang loop unroll(disable)
    for (u32 c0 = p0; c0 < p1; c0 += 128u) {
      const u32 kb = __builtin_amdgcn_readfirstlane(c0 + 16u * w);
      const u32 chunk_end = __builtin_amdgcn_readfirstlane((c0 + 128u < p1) ? (c0 + 128u) : p1);
      const int far_chunk = __builtin_amdgcn_readfirstlane(
          far_split || ((chunk_end <= pos0 + 1u) && ((pos0 + 1u - chunk_end) >= rel_extent)));

      v8f sp0 = (v8f)(-__builtin_inff());
      v8f sp1 = (v8f)(-__builtin_inff());

      if (kb < chunk_end) {
        const u32 kp_row = kb + rl;
        const int row_ok = (kp_row < chunk_end);
        const u32 kv_row = no_wrap ? (row_ok ? kp_row : (chunk_end - 1u))
                                   : ((row_ok ? kp_row : (chunk_end - 1u)) % kv_len);
        const G f16 *k_base = kc_h + (unsigned long)kv_row * kvd + 8u * hl;

        sp0 = zero;
        sp1 = zero;
        h8 kk0 = row_ok ? *(const G h8 *)(k_base + 0u) : (h8)(0.0f);
        h8 kk1 = row_ok ? *(const G h8 *)(k_base + 16u) : (h8)(0.0f);
#pragma unroll
        for (u32 ks = 0u; ks < 8u; ks += 2u) {
          const u32 c_a = 2u * ks + hl;
          const u32 c_b = 2u * (ks + 1u) + hl;
          const h8 q0_a = *(const LDSP h8 *)(ink_r10_df_gw_lds.kv.q + ink_r10_df_gw_q_off(rl, c_a));
          const h8 q0_b = *(const LDSP h8 *)(ink_r10_df_gw_lds.kv.q + ink_r10_df_gw_q_off(rl, c_b));
          sp0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk0, q0_a, sp0);
          sp0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk1, q0_b, sp0);
          if (npg > 1u) {
            const h8 q1_a = *(const LDSP h8 *)(ink_r10_df_gw_lds.kv.q + ink_r10_df_gw_q_off(16u + rl, c_a));
            const h8 q1_b = *(const LDSP h8 *)(ink_r10_df_gw_lds.kv.q + ink_r10_df_gw_q_off(16u + rl, c_b));
            sp1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk0, q1_a, sp1);
            sp1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kk1, q1_b, sp1);
          }
          if (ks + 2u < 8u) {
            kk0 = row_ok ? *(const G h8 *)(k_base + 16u * (ks + 2u)) : (h8)(0.0f);
            kk1 = row_ok ? *(const G h8 *)(k_base + 16u * (ks + 3u)) : (h8)(0.0f);
          }
          __builtin_amdgcn_sched_barrier(0);
        }

        if (!far_chunk) {
          const int d_min_s = (int)pos0 - (int)(kb + 15u);
          const u32 d_base = d_min_s > 0 ? (u32)d_min_s : 0u;
          if (d_base < rel_extent) {
            LDSP float *sc_row = ((LDSP float *)ink_r10_df_gw_lds.kv.scratch.v_sc[w]) + rl * 8u;
            const h8 ra0 = *(const LDSP h8 *)(ink_r10_df_gw_lds.kv.r + rl * 16u + 8u * hl);
            const u32 d0 = d_base + rl;
            h8 b0 = (h8)(0.0f);
            if (d0 < rel_extent) {
              const G f32x4 *rp = (const G f32x4 *)(rel_bias + (unsigned long)d0 * 16ul + 8ul * hl);
              const f32x4 v0 = rp[0], v1 = rp[1];
              b0 = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
            }
            const v8f rb0_0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0, ra0, zero);
            ink_r10_df_gw_apply_rb16(rb0_0, 0u, hl, kb, pos_q0, rel_extent, d_base, sc_row, &sp0);
            if (npg > 1u) {
              const h8 ra1 = *(const LDSP h8 *)(ink_r10_df_gw_lds.kv.r + (16u + rl) * 16u + 8u * hl);
              const v8f rb0_1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b0, ra1, zero);
              ink_r10_df_gw_apply_rb16(rb0_1, 0u, hl, kb, pos_q1, rel_extent, d_base, sc_row, &sp1);
            }
            if (d_base + 16u < rel_extent && (pos0 + T - 1u) >= kb + d_base + 16u) {
              const u32 d1 = d_base + 16u + rl;
              h8 b1 = (h8)(0.0f);
              if (d1 < rel_extent) {
                const G f32x4 *rp = (const G f32x4 *)(rel_bias + (unsigned long)d1 * 16ul + 8ul * hl);
                const f32x4 v0 = rp[0], v1 = rp[1];
                b1 = (h8){(f16)v0.x, (f16)v0.y, (f16)v0.z, (f16)v0.w, (f16)v1.x, (f16)v1.y, (f16)v1.z, (f16)v1.w};
              }
              const v8f rb1_0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1, ra0, zero);
              ink_r10_df_gw_apply_rb16(rb1_0, 16u, hl, kb, pos_q0, rel_extent, d_base, sc_row, &sp0);
              if (npg > 1u) {
                const h8 ra1 = *(const LDSP h8 *)(ink_r10_df_gw_lds.kv.r + (16u + rl) * 16u + 8u * hl);
                const v8f rb1_1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b1, ra1, zero);
                ink_r10_df_gw_apply_rb16(rb1_1, 16u, hl, kb, pos_q1, rel_extent, d_base, sc_row, &sp1);
              }
            }
          }
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) {
            const u32 kp = kb + 8u * hl + v;
            if (kp >= chunk_end || kp > pos_q0) sp0[v] = -__builtin_inff();
            if (npg > 1u && (kp >= chunk_end || kp > pos_q1)) sp1[v] = -__builtin_inff();
          }
        } else if (kb + 16u > chunk_end) {
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) {
            if (kb + 8u * hl + v >= chunk_end) {
              sp0[v] = -__builtin_inff();
              if (npg > 1u) sp1[v] = -__builtin_inff();
            }
          }
        }
      }

      u32x4 vr0 = (u32x4)(0u);
      {
        const u32 vp = c0 + rl;
        if (vp < chunk_end) {
          const u32 kv_v = no_wrap ? vp : (vp % kv_len);
          vr0 = *(const G u32x4 *)(vc_h + (unsigned long)kv_v * kvd + 16u * w + 8u * hl);
        }
      }

      if (!far_chunk) lbarrier();

      float mx0 = sp0[0];
#pragma unroll
      for (u32 v = 1u; v < 8u; v++) mx0 = __builtin_fmaxf(mx0, sp0[v]);
      mx0 = __builtin_fmaxf(mx0, ink_r10_df_xrow(mx0));
      if (hl == 0u) ink_r10_df_gw_lds.kv.scratch.red.mx[0][16u * w + rl] = mx0;

      if (npg > 1u) {
        float mx1 = sp1[0];
#pragma unroll
        for (u32 v = 1u; v < 8u; v++) mx1 = __builtin_fmaxf(mx1, sp1[v]);
        mx1 = __builtin_fmaxf(mx1, ink_r10_df_xrow(mx1));
        if (hl == 0u) ink_r10_df_gw_lds.kv.scratch.red.mx[1][16u * w + rl] = mx1;
      }
      lbarrier();

      float mc0 = ink_r10_df_gw_lds.kv.scratch.red.mx[0][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) mc0 = __builtin_fmaxf(mc0, ink_r10_df_gw_lds.kv.scratch.red.mx[0][16u * x + rl]);
      const float mn0 = __builtin_fmaxf(m0, mc0);
      const float rs0 = (ls0 > 0.0f && mn0 > -1e30f) ? __builtin_amdgcn_exp2f(m0 - mn0) : 0.0f;
      acc0 *= rs0;
      ls0 *= rs0;
      m0 = mn0;

      h8 ph0 = (h8)(0.0f);
      float lc0 = 0.0f;
      if (mn0 > -1e30f) {
#pragma unroll
        for (u32 v = 0u; v < 8u; v++) {
          const float p = __builtin_amdgcn_exp2f(sp0[v] - mn0);
          lc0 += p;
          ph0[v] = (f16)p;
        }
      }
      lc0 += ink_r10_df_xrow(lc0);
      if (hl == 0u) ink_r10_df_gw_lds.kv.scratch.red.ls[0][16u * w + rl] = lc0;

      h8 ph1 = (h8)(0.0f);
      if (npg > 1u) {
        float mc1 = ink_r10_df_gw_lds.kv.scratch.red.mx[1][rl];
#pragma unroll
        for (u32 x = 1u; x < 8u; x++) mc1 = __builtin_fmaxf(mc1, ink_r10_df_gw_lds.kv.scratch.red.mx[1][16u * x + rl]);
        const float mn1 = __builtin_fmaxf(m1, mc1);
        const float rs1 = (ls1 > 0.0f && mn1 > -1e30f) ? __builtin_amdgcn_exp2f(m1 - mn1) : 0.0f;
        acc1 *= rs1;
        ls1 *= rs1;
        m1 = mn1;

        float lc1 = 0.0f;
        if (mn1 > -1e30f) {
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) {
            const float p = __builtin_amdgcn_exp2f(sp1[v] - mn1);
            lc1 += p;
            ph1[v] = (f16)p;
          }
        }
        lc1 += ink_r10_df_xrow(lc1);
        if (hl == 0u) ink_r10_df_gw_lds.kv.scratch.red.ls[1][16u * w + rl] = lc1;
      }

      if (w == 0u) {
        *(LDSP h8 *)(ink_r10_df_gw_lds.kv.p_tile[0][0] + p_off) = ph0;
        if (npg > 1u) *(LDSP h8 *)(ink_r10_df_gw_lds.kv.p_tile[0][1] + p_off) = ph1;
      }
      lbarrier();

      float sum_l0 = ink_r10_df_gw_lds.kv.scratch.red.ls[0][rl];
#pragma unroll
      for (u32 x = 1u; x < 8u; x++) sum_l0 += ink_r10_df_gw_lds.kv.scratch.red.ls[0][16u * x + rl];
      ls0 += sum_l0;
      if (npg > 1u) {
        float sum_l1 = ink_r10_df_gw_lds.kv.scratch.red.ls[1][rl];
#pragma unroll
        for (u32 x = 1u; x < 8u; x++) sum_l1 += ink_r10_df_gw_lds.kv.scratch.red.ls[1][16u * x + rl];
        ls1 += sum_l1;
      }
      lbarrier();

      LDSP u32 *v_sc_w = ink_r10_df_gw_lds.kv.scratch.v_sc[w];
#pragma clang loop unroll(disable)
      for (u32 kt = 0u; kt < 8u; kt++) {
        if (c0 + 16u * kt >= chunk_end) break;
        const int has_nxt = (kt + 1u < 8u) && (c0 + 16u * (kt + 1u) < chunk_end);
        u32x4 vr_nxt = (u32x4)(0u);
        if (has_nxt) {
          const u32 vp_n = c0 + 16u * (kt + 1u) + rl;
          if (vp_n < chunk_end) {
            const u32 kv_vn = no_wrap ? vp_n : (vp_n % kv_len);
            vr_nxt = *(const G u32x4 *)(vc_h + (unsigned long)kv_vn * kvd + 16u * w + 8u * hl);
          }
        }
        __builtin_amdgcn_sched_barrier(0);

        const h8 vt_a = ink_r10_df_gw_v_transpose(vr0, v_sc_w, hl, rl);
        const h8 pt0 = *(const LDSP h8 *)(ink_r10_df_gw_lds.kv.p_tile[kt & 1u][0] + p_off);
        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt0, acc0);
        if (npg > 1u) {
          const h8 pt1 = *(const LDSP h8 *)(ink_r10_df_gw_lds.kv.p_tile[kt & 1u][1] + p_off);
          acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vt_a, pt1, acc1);
        }

        if (has_nxt && w == kt + 1u) {
          *(LDSP h8 *)(ink_r10_df_gw_lds.kv.p_tile[(kt + 1u) & 1u][0] + p_off) = ph0;
          if (npg > 1u) *(LDSP h8 *)(ink_r10_df_gw_lds.kv.p_tile[(kt + 1u) & 1u][1] + p_off) = ph1;
        }
        lbarrier();
        vr0 = vr_nxt;
      }
    }

    lbarrier();
    if (w == 0u && hl == 0u) {
      ink_r10_df_gw_lds.ph4.m[rl] = (ls0 > 0.0f) ? (m0 * INK_R10_DF_LN2) : -1e30f;
      ink_r10_df_gw_lds.ph4.l[rl] = ls0;
      if (npg > 1u) {
        ink_r10_df_gw_lds.ph4.m[16u + rl] = (ls1 > 0.0f) ? (m1 * INK_R10_DF_LN2) : -1e30f;
        ink_r10_df_gw_lds.ph4.l[16u + rl] = ls1;
      }
    }
    lbarrier();

#pragma clang loop unroll(disable)
    for (u32 pg = 0u; pg < npg; pg++) {
      const v8f a_cur = (pg == 0u) ? acc0 : acc1;
#pragma unroll
      for (u32 qb = 0u; qb < 2u; qb++) {
        const u32 q_base = 16u * pg + 8u * qb;
        if (q_base >= nq_tot) break;
        if ((rl >> 3u) == qb) {
          const u32 q_loc = rl & 7u;
          LDSP float *dst_w = ink_r10_df_gw_lds.ph4.o_stg[q_loc] + 16u * w + 8u * hl;
#pragma unroll
          for (u32 v = 0u; v < 8u; v++) dst_w[v] = a_cur[v];
        }
        lbarrier();

        const u32 qi = q_base + w;
        if (qi < nq_tot) {
          const u32 t = qi / INK_R10_DF_GQA;
          const u32 h = h0 + (qi - t * INK_R10_DF_GQA);
          const f32x4 ov = ((const LDSP f32x4 *)ink_r10_df_gw_lds.ph4.o_stg[w])[l];
          if (nsplit == 1u) {
            const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
            const float sv = st[h];
            const float mq = ink_r10_df_gw_lds.ph4.m[qi];
            const float lq = ink_r10_df_gw_lds.ph4.l[qi];
            const float M = (lq > 0.0f) ? __builtin_fmaxf(sv, mq) : sv;
            const float wx = (lq > 0.0f) ? __builtin_expf(mq - M) : 0.0f;
            const float L = __builtin_expf(sv - M) + lq * wx;
            const float sc_o = wx / L;
            const G float *gt_h = agate ? ((const G float *)((const G u8 *)agate + (unsigned long)t * gs) +
                                           (unsigned long)h * INK_R10_DF_HD)
                                        : (const G float *)0;
            G float *ot_h = out ? ((G float *)((G u8 *)out + (unsigned long)t * os) +
                                   (unsigned long)h * INK_R10_DF_HD)
                                : (G float *)0;
            G u8 *oqt = oq ? (oq + (unsigned long)t * oqs) : (G u8 *)0;
            ink_r10_df_gate_q8_head_vec4(ov * sc_o, gt_h, ot_h, oqt, h, n_head, l);
          } else {
            G float *pp = part + (unsigned long)((t * n_head + h) * nsplit + s) * INK_R10_DF_PSTR;
            if (l == 0u) {
              pp[0] = ink_r10_df_gw_lds.ph4.m[qi];
              pp[1] = ink_r10_df_gw_lds.ph4.l[qi];
            }
            G float *po = pp + 2u + 4u * l;
            po[0] = ov.x; po[1] = ov.y; po[2] = ov.z; po[3] = ov.w;
          }
        }
        lbarrier();
      }
    }

    if (nsplit == 1u) return;
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) {
    ink_r10_df_gw_lds.ph4.is_last =
        (__scoped_atomic_fetch_add(&count[kh], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == nsplit - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r10_df_gw_lds.ph4.is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  for (u32 qi = w; qi < nq_tot; qi += 8u) {
    const u32 t = qi / INK_R10_DF_GQA;
    const u32 h = h0 + (qi - t * INK_R10_DF_GQA);
    const G float *st = (const G float *)((const G u8 *)sink + (unsigned long)t * ss);
    const G float *gt_h = agate ? ((const G float *)((const G u8 *)agate + (unsigned long)t * gs) +
                                   (unsigned long)h * INK_R10_DF_HD)
                                : (const G float *)0;
    G float *ot_h = out ? ((G float *)((G u8 *)out + (unsigned long)t * os) +
                           (unsigned long)h * INK_R10_DF_HD)
                        : (G float *)0;
    G u8 *oqt = oq ? (oq + (unsigned long)t * oqs) : (G u8 *)0;
    const G float *ph = part + (unsigned long)((t * n_head + h) * nsplit) * INK_R10_DF_PSTR;
    ink_r10_df_gw_merge_head_gate_q8(ph, st[h], nsplit, l, ink_r10_df_gw_lds.ph4.wt[w],
                                     gt_h, ot_h, oqt, h, n_head);
  }
  barrier();
  if (tid() == 0u) count[kh] = 0u;
}

#undef INK_R10_DF_ADD4
#undef INK_R10_DF_QSC
#undef INK_R10_DF_LN2
#undef INK_R10_DF_LOG2E
#undef INK_R10_DF_PSTR
#undef INK_R10_DF_GQA
#undef INK_R10_DF_DREL
#undef INK_R10_DF_HD
#undef INK_R10_DF_LDS_MAXNB
#undef INK_R10_DF_SB_BYTES

#endif /* !IE_MEGA */
