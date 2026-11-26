/* ============================================================================
 * Inkling-2 Round 7: Wave 5 Decode Bandwidth Kernels (ink_mask2 bits 81, 82, 83)
 *   - Bit 81: FLUTE 0-Extra-VRAM SoA-Repacked IQ3_XXS (98 B -> 98 B exact!)
 *   - Bit 82: Multi-row Wave-Uniform IQ4_NL (v_perm_b32 LUT + double-buffer)
 *   - Bit 83: Multi-row Wave-Uniform Dense Q8_0 (8 lanes/row, 100% util on nb=72/96)
 * Target: RX 9070 XT (gfx1201, wave32, NT=256)
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r7_fl_ /
 * INK_R7_FL_ and all macros are #undef'd at EOF.
 *
 * ============================================================================
 * EXACT REVERSIBLE LOAD-TIME FLUTE SoA REPACK SPEC FOR gpu_sink() (0 EXTRA VRAM)
 * ============================================================================
 * For any IQ3_XXS expert matrix with rows % 4 == 0 (Inkling-2 egu: rows=2304,
 * nb=72 -> mb = nb/8 = 9 super-blocks; q_bytes = rows * mb * 98):
 * Within each expert slice `dst` of exact size `q_bytes = rows * mb * 98` bytes:
 *   const size_t n_rg   = rows / 4;                // 576 4-row wave groups
 *   const size_t n_sb   = n_rg * mb;               // 5184 wave-group super-blocks
 *   uint64_t *p_qs      = (uint64_t *)(dst);                  // Plane 0: n_sb * 256 B (64-B aligned!)
 *   uint32_t *p_aux     = (uint32_t *)(dst + n_sb * 256);     // Plane 1: n_sb * 128 B (32-B aligned!)
 *   uint16_t *p_d       = (uint16_t *)(dst + n_sb * 384);     // Plane 2: n_sb *   8 B (exact 98 B/row/sb!)
 *   // Total = n_sb * (256 + 128 + 8) = (rows/4) * mb * 392 = rows * mb * 98 == q_bytes (0 extra bytes!)
 *   for (size_t rg = 0; rg < n_rg; rg++) {
 *     for (size_t sb = 0; sb < mb; sb++) {
 *       const size_t gidx = rg * mb + sb;
 *       for (size_t k = 0; k < 4; k++) {
 *         const uint8_t *src = raw_exp + ((rg * 4 + k) * mb + sb) * 98;
 *         memcpy(&p_d[gidx * 4 + k], src, 2);
 *         for (size_t s = 0; s < 8; s++) {
 *           memcpy(&p_qs[gidx * 32 + k * 8 + s],  src + 2  + s * 8, 8);
 *           memcpy(&p_aux[gidx * 32 + k * 8 + s], src + 66 + s * 4, 4);
 *         }
 *       }
 *     }
 *   }
 * Every superblock step in the wave reads:
 *   - 256 B contiguous (2 exact 128-B cache lines) via global_load_b64 (p_qs)
 *   - 128 B contiguous (1 exact 128-B cache line)  via global_load_b32 (p_aux)
 *   -   8 B contiguous                             via global_load_u16 (p_d)
 * Zero unaligned 98-byte striding, 0 extra VRAM, 100% 0-bit-diff exact!
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R7_FL_Q8_TMAX 8u

typedef float ink_r7_fl_f8 __attribute__((ext_vector_type(8)));
typedef unsigned int ink_r7_fl_u32x2 __attribute__((ext_vector_type(2)));

struct __attribute__((packed, aligned(4))) ink_r7_fl_u32x4p {
  u32x4 v;
};

struct __attribute__((packed, aligned(4))) ink_r7_fl_u32x2p {
  ink_r7_fl_u32x2 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r7_fl_ld4(const G void *p) {
  return ((const G struct ink_r7_fl_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) ink_r7_fl_u32x2 ink_r7_fl_ld2(const G void *p) {
  return ((const G struct ink_r7_fl_u32x2p *)p)->v;
}

static inline __attribute__((always_inline)) u32 ink_r7_fl_div_x(u32 pid, u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

static inline __attribute__((always_inline)) float ink_r7_fl_oct_tree_f(float v) {
#pragma clang fp contract(off)
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}

static inline __attribute__((always_inline)) int ink_r7_fl_q8_dot8(u32x4 w0, u32x4 w1, u32x4 a0, u32x4 a1) {
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

static inline __attribute__((always_inline)) int ink_r7_fl_iq4_dot4(u32x4 w4, u32x4 a0, u32x4 a1) {
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

static inline __attribute__((always_inline)) int ink_r7_fl_iq4_dot_pre(
    u32 wl0, u32 wh0, u32 wl1, u32 wh1, u32 wl2, u32 wh2, u32 wl3, u32 wh3,
    u32x4 a0, u32x4 a1) {
  const u32 a00 = a0.x, a01 = a0.y, a02 = a0.z, a03 = a0.w;
  const u32 a10 = a1.x, a11 = a1.y, a12 = a1.z, a13 = a1.w;
  u32 d = 0u;
  d = ie_dot4_ss(ie_dot4_ss(d, wl0, a00), wh0, a10);
  d = ie_dot4_ss(ie_dot4_ss(d, wl1, a01), wh1, a11);
  d = ie_dot4_ss(ie_dot4_ss(d, wl2, a02), wh2, a12);
  d = ie_dot4_ss(ie_dot4_ss(d, wl3, a03), wh3, a13);
  return (int)d;
}

static inline __attribute__((always_inline)) float ink_r7_fl_iq3_dot_pre(
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

static inline __attribute__((always_inline)) float ink_r7_fl_iq3_stream_dot(
    const LDSP u32 *ulut, u32 d16, u32 aux, ink_r7_fl_u32x2 q0123,
    u32x4 x0, u32x4 x1, float dav) {
#pragma clang fp contract(off)
  const float dw = ie_rq_h2f(d16) * (0.5f + (float)(aux >> 28u)) * 0.5f;
  const u32 q01 = q0123.x, q23 = q0123.y;
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

#define INK_R7_FL_ADD4(a0_, a1_, a2_, a3_, j_, v_) \
  do {                                             \
    if ((j_) == 0u) (a0_) += (v_);                 \
    else if ((j_) == 1u) (a1_) += (v_);            \
    else if ((j_) == 2u) (a2_) += (v_);            \
    else (a3_) += (v_);                            \
  } while (0)

#define INK_R7_FL_FMA4(a0_, a1_, a2_, a3_, j_, v_, d_) \
  do {                                                 \
    if ((j_) == 0u) (a0_) = __builtin_fmaf((v_), (d_), (a0_)); \
    else if ((j_) == 1u) (a1_) = __builtin_fmaf((v_), (d_), (a1_)); \
    else if ((j_) == 2u) (a2_) = __builtin_fmaf((v_), (d_), (a2_)); \
    else (a3_) = __builtin_fmaf((v_), (d_), (a3_));    \
  } while (0)

/* ============================================================================
 * Kernel 1a (Bit 81): ie_inkling_gexp_iq3xxs_flute_situ_t1_r7
 * T=1 decode FLUTE SoA-repacked IQ3_XXS Gate/Up + SITU + Q8_0 (0-bit-diff!).
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq3xxs_flute_situ_t1_r7(
    const G u8 *qw, const G u8 *xq, G float *b, G u8 *q,
    u32 eff, u32 nb, const G u32 *route, u32 q_bytes,
    u32 astr, u32 bstr, u32 qstr, float beta, float lbeta) {
#pragma clang fp contract(off)
  static LDS u32   ink_r7_fl_t1_lut[256u] __attribute__((aligned(16)));
  static LDS float ink_r7_fl_t1_situ[32u] __attribute__((aligned(16)));

  if (eff == 0u || (eff & 31u) != 0u || nb < 8u || (nb & 7u) != 0u || !qw || !xq || (!b && !q)) return;
  const u32 blk = wgid();
  const u32 nb_out = eff >> 5u;
  if (blk >= nb_out) return;

  const u32 t0 = tid();
  ink_r7_fl_t1_lut[t0] = ie_rq_iq3xxs_grid[t0];
  lbarrier();

  const u32 slot = tokid();
  const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[slot]);
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 r_in_blk = (wv << 2u) + k;
  const u32 mb = nb >> 3u;

  /* FLUTE 3-plane pointers (exact q_bytes = 2*eff * mb * 98, 0 extra VRAM). */
  const unsigned long n_sb_tot = (unsigned long)((eff << 1u) >> 2u) * (unsigned long)mb;
  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * (unsigned long)q_bytes);
  const G ink_r7_fl_u32x2 *p_qs  = (const G ink_r7_fl_u32x2 *)ebase;
  const G u32             *p_aux = (const G u32 *)(ebase + n_sb_tot * 256ul);
  const G u16             *p_d   = (const G u16 *)(ebase + n_sb_tot * 384ul);

  const u32 rg_g = (blk << 3u) + wv;
  const u32 rg_u = ((eff >> 2u) + rg_g);
  const unsigned long gg0 = (unsigned long)rg_g * (unsigned long)mb;
  const unsigned long gu0 = (unsigned long)rg_u * (unsigned long)mb;

  const G u8 *xp = rq_ptr_u(xq + (unsigned long)slot * (unsigned long)astr);
  const G float *dap = (const G float *)(xp + 32ul * (unsigned long)nb);

  float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
  float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;

  /* Prologue: coalesced 256 B (qs) + 128 B (aux) + 8 B (d) load for gate sb=0! */
  ink_r7_fl_u32x2 qg_c = ink_r7_fl_ld2(p_qs + gg0 * 32ul + (unsigned long)l);
  u32 auxg_c = p_aux[gg0 * 32ul + (unsigned long)l];
  u32 dg_c = (u32)p_d[gg0 * 4ul + (unsigned long)k];
  u32x4 x0_c = ink_r7_fl_ld4(xp + (s << 5u));
  u32x4 x1_c = ink_r7_fl_ld4(xp + (s << 5u) + 16u);
  float dav_c = dap[s];
  __asm__ volatile("" : "+v"(dg_c), "+v"(auxg_c));

  /* Four super-blocks per pass, unrolled, so the accumulator index is a constant in each copy and
   * INK_R7_FL_ADD4 is one add instead of a scalar compare/branch ladder on sb & 3.
   * Every accumulator still takes the same super-blocks in the same order. */
#pragma clang loop unroll(disable)
  for (u32 sb4 = 0u; sb4 < mb; sb4 += 4u) {
#pragma unroll
    for (u32 j_acc = 0u; j_acc < 4u; j_acc++) {
      const u32 sb = sb4 + j_acc;
      if (sb < mb) {
        const unsigned long gu = gu0 + (unsigned long)sb;
        const ink_r7_fl_u32x2 qu_c = ink_r7_fl_ld2(p_qs + gu * 32ul + (unsigned long)l);
        const u32 auxu_c = p_aux[gu * 32ul + (unsigned long)l];
        const u32 du_c = (u32)p_d[gu * 4ul + (unsigned long)k];
        __builtin_amdgcn_sched_barrier(0);

        const float vg = ink_r7_fl_iq3_stream_dot(ink_r7_fl_t1_lut, dg_c, auxg_c, qg_c, x0_c, x1_c, dav_c);
        INK_R7_FL_ADD4(ag0, ag1, ag2, ag3, j_acc, vg);
        __builtin_amdgcn_sched_barrier(0);

        ink_r7_fl_u32x2 qg_n = (ink_r7_fl_u32x2){0u, 0u};
        u32 auxg_n = 0u, dg_n = 0u;
        u32x4 x0_n = (u32x4)(0u), x1_n = (u32x4)(0u);
        float dav_n = 0.0f;
        if (sb + 1u < mb) {
          const unsigned long ggn = gg0 + (unsigned long)(sb + 1u);
          const u32 bn = ((sb + 1u) << 3u) + s;
          qg_n = ink_r7_fl_ld2(p_qs + ggn * 32ul + (unsigned long)l);
          auxg_n = p_aux[ggn * 32ul + (unsigned long)l];
          dg_n = (u32)p_d[ggn * 4ul + (unsigned long)k];
          x0_n = ink_r7_fl_ld4(xp + (bn << 5u));
          x1_n = ink_r7_fl_ld4(xp + (bn << 5u) + 16u);
          dav_n = dap[bn];
        }
        __builtin_amdgcn_sched_barrier(0);

        const float vu = ink_r7_fl_iq3_stream_dot(ink_r7_fl_t1_lut, du_c, auxu_c, qu_c, x0_c, x1_c, dav_c);
        INK_R7_FL_ADD4(au0, au1, au2, au3, j_acc, vu);
        __builtin_amdgcn_sched_barrier(0);

        qg_c = qg_n; auxg_c = auxg_n; dg_c = dg_n;
        x0_c = x0_n; x1_c = x1_n; dav_c = dav_n;
      }
    }
  }

  const float gv = ink_r7_fl_oct_tree_f((ag0 + ag2) + (ag1 + ag3));
  const float uv = ink_r7_fl_oct_tree_f((au0 + au2) + (au1 + au3));
  if (s == 0u) {
    const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
    const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
    ink_r7_fl_t1_situ[r_in_blk] = gs * sigmoidf(gv) * us;
  }
  lbarrier();

  if (wv == 0u) {
    const float v = ink_r7_fl_t1_situ[l];
    if (b) ((G float *)((G u8 *)b + (unsigned long)slot * (unsigned long)bstr))[(blk << 5u) + l] = v;
    if (q) quant_wave(v, blk, q + (unsigned long)slot * (unsigned long)qstr, nb_out);
  }
}

/* ============================================================================
 * Kernel 1b (Bit 81): ie_inkling_gexp_iq3xxs_flute_situ_compact_r7
 * Compact active-pack verify (T=2..16) FLUTE SoA IQ3_XXS Gate/Up + SITU + Q8_0.
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq3xxs_flute_situ_compact_r7(
    const G u8 *qw, const G u8 *X, G float *b_out, G u8 *q_out,
    u32 eff, u32 nb, const G u32 *list, const G u32 *packs, const G u32 *n_packs,
    u32 q_bytes, u32 xs, u32 bstr, u32 qstr, u32 xdiv, float beta, float lbeta) {
#pragma clang fp contract(off)
  static LDS u32   ink_r7_fl_cm_lut[256u] __attribute__((aligned(16)));
  static LDS float ink_r7_fl_cm_situ[4u][32u] __attribute__((aligned(16)));

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
  ink_r7_fl_cm_lut[t0] = ie_rq_iq3xxs_grid[t0];
  lbarrier();

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 r_in_blk = (wv << 2u) + k;
  const u32 mb = nb >> 3u;

  const unsigned long n_sb_tot = (unsigned long)((eff << 1u) >> 2u) * (unsigned long)mb;
  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * (unsigned long)q_bytes);
  const G ink_r7_fl_u32x2 *p_qs  = (const G ink_r7_fl_u32x2 *)ebase;
  const G u32             *p_aux = (const G u32 *)(ebase + n_sb_tot * 256ul);
  const G u16             *p_d   = (const G u16 *)(ebase + n_sb_tot * 384ul);

  const u32 rg_g = (blk << 3u) + wv;
  const u32 rg_u = (eff >> 2u) + rg_g;
  const unsigned long gg0 = (unsigned long)rg_g * (unsigned long)mb;
  const unsigned long gu0 = (unsigned long)rg_u * (unsigned long)mb;
  const unsigned long x_scale_off = 32ul * (unsigned long)nb;

  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
  const G u8 *xp0 = rq_ptr_u(X + (unsigned long)ink_r7_fl_div_x(p0, xdiv) * (unsigned long)xs);
  const G float *dap0 = (const G float *)(xp0 + x_scale_off);

  if (nj == 1u) {
    float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
    float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;

    ink_r7_fl_u32x2 qg_c = ink_r7_fl_ld2(p_qs + gg0 * 32ul + (unsigned long)l);
    u32 auxg_c = p_aux[gg0 * 32ul + (unsigned long)l];
    u32 dg_c = (u32)p_d[gg0 * 4ul + (unsigned long)k];
    u32x4 x0_c = ink_r7_fl_ld4(xp0 + (s << 5u));
    u32x4 x1_c = ink_r7_fl_ld4(xp0 + (s << 5u) + 16u);
    float dav_c = dap0[s];
    __asm__ volatile("" : "+v"(dg_c), "+v"(auxg_c));

    /* Unrolled by four for a constant accumulator index, as in the T=1 kernel above. */
#pragma clang loop unroll(disable)
    for (u32 sb4 = 0u; sb4 < mb; sb4 += 4u) {
#pragma unroll
      for (u32 j_acc = 0u; j_acc < 4u; j_acc++) {
        const u32 sb = sb4 + j_acc;
        if (sb < mb) {
          const unsigned long gu = gu0 + (unsigned long)sb;
          const ink_r7_fl_u32x2 qu_c = ink_r7_fl_ld2(p_qs + gu * 32ul + (unsigned long)l);
          const u32 auxu_c = p_aux[gu * 32ul + (unsigned long)l];
          const u32 du_c = (u32)p_d[gu * 4ul + (unsigned long)k];
          __builtin_amdgcn_sched_barrier(0);

          const float vg = ink_r7_fl_iq3_stream_dot(ink_r7_fl_cm_lut, dg_c, auxg_c, qg_c, x0_c, x1_c, dav_c);
          INK_R7_FL_ADD4(ag0, ag1, ag2, ag3, j_acc, vg);
          __builtin_amdgcn_sched_barrier(0);

          ink_r7_fl_u32x2 qg_n = (ink_r7_fl_u32x2){0u, 0u};
          u32 auxg_n = 0u, dg_n = 0u;
          u32x4 x0_n = (u32x4)(0u), x1_n = (u32x4)(0u);
          float dav_n = 0.0f;
          if (sb + 1u < mb) {
            const unsigned long ggn = gg0 + (unsigned long)(sb + 1u);
            const u32 bn = ((sb + 1u) << 3u) + s;
            qg_n = ink_r7_fl_ld2(p_qs + ggn * 32ul + (unsigned long)l);
            auxg_n = p_aux[ggn * 32ul + (unsigned long)l];
            dg_n = (u32)p_d[ggn * 4ul + (unsigned long)k];
            x0_n = ink_r7_fl_ld4(xp0 + (bn << 5u));
            x1_n = ink_r7_fl_ld4(xp0 + (bn << 5u) + 16u);
            dav_n = dap0[bn];
          }
          __builtin_amdgcn_sched_barrier(0);

          const float vu = ink_r7_fl_iq3_stream_dot(ink_r7_fl_cm_lut, du_c, auxu_c, qu_c, x0_c, x1_c, dav_c);
          INK_R7_FL_ADD4(au0, au1, au2, au3, j_acc, vu);
          __builtin_amdgcn_sched_barrier(0);

          qg_c = qg_n; auxg_c = auxg_n; dg_c = dg_n;
          x0_c = x0_n; x1_c = x1_n; dav_c = dav_n;
        }
      }
    }

    const float gv = ink_r7_fl_oct_tree_f((ag0 + ag2) + (ag1 + ag3));
    const float uv = ink_r7_fl_oct_tree_f((au0 + au2) + (au1 + au3));
    if (s == 0u) {
      const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
      const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
      ink_r7_fl_cm_situ[0u][r_in_blk] = gs * sigmoidf(gv) * us;
    }
    lbarrier();
    if (wv == 0u) {
      const float v = ink_r7_fl_cm_situ[0u][l];
      if (b_out) ((G float *)((G u8 *)b_out + (unsigned long)p0 * (unsigned long)bstr))[(blk << 5u) + l] = v;
      if (q_out) quant_wave(v, blk, q_out + (unsigned long)p0 * (unsigned long)qstr, nb_out);
    }
    return;
  }

  /* Multi-pair path (nj in 2..4): coalesced FLUTE loads decoded ONCE across nj pairs. */
  const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 1u]);
  const u32 p2 = (nj > 2u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 2u]) : p0;
  const u32 p3 = (nj > 3u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 3u]) : p0;

  const G u8 *xp1 = rq_ptr_u(X + (unsigned long)ink_r7_fl_div_x(p1, xdiv) * (unsigned long)xs);
  const G u8 *xp2 = (nj > 2u) ? rq_ptr_u(X + (unsigned long)ink_r7_fl_div_x(p2, xdiv) * (unsigned long)xs) : xp0;
  const G u8 *xp3 = (nj > 3u) ? rq_ptr_u(X + (unsigned long)ink_r7_fl_div_x(p3, xdiv) * (unsigned long)xs) : xp0;
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
    const unsigned long gg = gg0 + (unsigned long)sb;
    const unsigned long gu = gu0 + (unsigned long)sb;

    {
      const ink_r7_fl_u32x2 q0123 = ink_r7_fl_ld2(p_qs + gg * 32ul + (unsigned long)l);
      const u32 aux_c = p_aux[gg * 32ul + (unsigned long)l];
      const u32 d_c = (u32)p_d[gg * 4ul + (unsigned long)k];
      const u32 q01_c = q0123.x, q23_c = q0123.y;

      const u32 qp0 = q01_c & 0xFFFFu, qp1 = q01_c >> 16u;
      const u32 qp2 = q23_c & 0xFFFFu, qp3 = q23_c >> 16u;
      const u32 g_lo0 = ink_r7_fl_cm_lut[qp0 & 255u], g_hi0 = ink_r7_fl_cm_lut[qp0 >> 8u];
      const u32 g_lo1 = ink_r7_fl_cm_lut[qp1 & 255u], g_hi1 = ink_r7_fl_cm_lut[qp1 >> 8u];
      const u32 g_lo2 = ink_r7_fl_cm_lut[qp2 & 255u], g_hi2 = ink_r7_fl_cm_lut[qp2 >> 8u];
      const u32 g_lo3 = ink_r7_fl_cm_lut[qp3 & 255u], g_hi3 = ink_r7_fl_cm_lut[qp3 >> 8u];
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

      const float v0 = ink_r7_fl_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                             ink_r7_fl_ld4(xp0 + (b << 5u)), ink_r7_fl_ld4(xp0 + (b << 5u) + 16u), dap0[b]);
      INK_R7_FL_ADD4(ag0_0, ag0_1, ag0_2, ag0_3, j_acc, v0);
      const float v1 = ink_r7_fl_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                             ink_r7_fl_ld4(xp1 + (b << 5u)), ink_r7_fl_ld4(xp1 + (b << 5u) + 16u), dap1[b]);
      INK_R7_FL_ADD4(ag1_0, ag1_1, ag1_2, ag1_3, j_acc, v1);
      if (nj > 2u) {
        const float v2 = ink_r7_fl_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                               ink_r7_fl_ld4(xp2 + (b << 5u)), ink_r7_fl_ld4(xp2 + (b << 5u) + 16u), dap2[b]);
        INK_R7_FL_ADD4(ag2_0, ag2_1, ag2_2, ag2_3, j_acc, v2);
        if (nj > 3u) {
          const float v3 = ink_r7_fl_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                                 ink_r7_fl_ld4(xp3 + (b << 5u)), ink_r7_fl_ld4(xp3 + (b << 5u) + 16u), dap3[b]);
          INK_R7_FL_ADD4(ag3_0, ag3_1, ag3_2, ag3_3, j_acc, v3);
        }
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    {
      const ink_r7_fl_u32x2 q0123 = ink_r7_fl_ld2(p_qs + gu * 32ul + (unsigned long)l);
      const u32 aux_c = p_aux[gu * 32ul + (unsigned long)l];
      const u32 d_c = (u32)p_d[gu * 4ul + (unsigned long)k];
      const u32 q01_c = q0123.x, q23_c = q0123.y;

      const u32 qp0 = q01_c & 0xFFFFu, qp1 = q01_c >> 16u;
      const u32 qp2 = q23_c & 0xFFFFu, qp3 = q23_c >> 16u;
      const u32 g_lo0 = ink_r7_fl_cm_lut[qp0 & 255u], g_hi0 = ink_r7_fl_cm_lut[qp0 >> 8u];
      const u32 g_lo1 = ink_r7_fl_cm_lut[qp1 & 255u], g_hi1 = ink_r7_fl_cm_lut[qp1 >> 8u];
      const u32 g_lo2 = ink_r7_fl_cm_lut[qp2 & 255u], g_hi2 = ink_r7_fl_cm_lut[qp2 >> 8u];
      const u32 g_lo3 = ink_r7_fl_cm_lut[qp3 & 255u], g_hi3 = ink_r7_fl_cm_lut[qp3 >> 8u];
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

      const float v0 = ink_r7_fl_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                             ink_r7_fl_ld4(xp0 + (b << 5u)), ink_r7_fl_ld4(xp0 + (b << 5u) + 16u), dap0[b]);
      INK_R7_FL_ADD4(au0_0, au0_1, au0_2, au0_3, j_acc, v0);
      const float v1 = ink_r7_fl_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                             ink_r7_fl_ld4(xp1 + (b << 5u)), ink_r7_fl_ld4(xp1 + (b << 5u) + 16u), dap1[b]);
      INK_R7_FL_ADD4(au1_0, au1_1, au1_2, au1_3, j_acc, v1);
      if (nj > 2u) {
        const float v2 = ink_r7_fl_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                               ink_r7_fl_ld4(xp2 + (b << 5u)), ink_r7_fl_ld4(xp2 + (b << 5u) + 16u), dap2[b]);
        INK_R7_FL_ADD4(au2_0, au2_1, au2_2, au2_3, j_acc, v2);
        if (nj > 3u) {
          const float v3 = ink_r7_fl_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw,
                                                 ink_r7_fl_ld4(xp3 + (b << 5u)), ink_r7_fl_ld4(xp3 + (b << 5u) + 16u), dap3[b]);
          INK_R7_FL_ADD4(au3_0, au3_1, au3_2, au3_3, j_acc, v3);
        }
      }
    }
  }

  {
    const float gv = ink_r7_fl_oct_tree_f((ag0_0 + ag0_2) + (ag0_1 + ag0_3));
    const float uv = ink_r7_fl_oct_tree_f((au0_0 + au0_2) + (au0_1 + au0_3));
    if (s == 0u) {
      const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
      const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
      ink_r7_fl_cm_situ[0u][r_in_blk] = gs * sigmoidf(gv) * us;
    }
  }
  {
    const float gv = ink_r7_fl_oct_tree_f((ag1_0 + ag1_2) + (ag1_1 + ag1_3));
    const float uv = ink_r7_fl_oct_tree_f((au1_0 + au1_2) + (au1_1 + au1_3));
    if (s == 0u) {
      const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
      const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
      ink_r7_fl_cm_situ[1u][r_in_blk] = gs * sigmoidf(gv) * us;
    }
  }
  if (nj > 2u) {
    const float gv = ink_r7_fl_oct_tree_f((ag2_0 + ag2_2) + (ag2_1 + ag2_3));
    const float uv = ink_r7_fl_oct_tree_f((au2_0 + au2_2) + (au2_1 + au2_3));
    if (s == 0u) {
      const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
      const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
      ink_r7_fl_cm_situ[2u][r_in_blk] = gs * sigmoidf(gv) * us;
    }
  }
  if (nj > 3u) {
    const float gv = ink_r7_fl_oct_tree_f((ag3_0 + ag3_2) + (ag3_1 + ag3_3));
    const float uv = ink_r7_fl_oct_tree_f((au3_0 + au3_2) + (au3_1 + au3_3));
    if (s == 0u) {
      const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
      const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
      ink_r7_fl_cm_situ[3u][r_in_blk] = gs * sigmoidf(gv) * us;
    }
  }
  lbarrier();

  if (wv < nj) {
    const u32 pw = (wv == 0u) ? p0 : ((wv == 1u) ? p1 : ((wv == 2u) ? p2 : p3));
    const float v = ink_r7_fl_cm_situ[wv][l];
    if (b_out) ((G float *)((G u8 *)b_out + (unsigned long)pw * (unsigned long)bstr))[(blk << 5u) + l] = v;
    if (q_out) quant_wave(v, blk, q_out + (unsigned long)pw * (unsigned long)qstr, nb_out);
  }
}

/* ============================================================================
 * Kernel 2a (Bit 82): ie_inkling_gexp_iq4nl_bw_t1_r7
 * Multi-row wave-uniform IQ4_NL T=1 GEMV with double-buffered u32x4 prefetch.
 * Drop-in signature for ie_inkling_gexp_iq4nl_r1 (0-bit-diff exact, 0 LDS).
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq4nl_bw_t1_r7(
    const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
    const G u32 *route, u32 qwn, u32 qsn, u32 astr, u32 bstr, u32 kind, u32 two_off) {
#pragma clang fp contract(off)
  (void)kind; (void)two_off;
  if (rows == 0u || nb == 0u) return;

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 r_wave = (wgid() << 5u) + (wv << 2u);
  if (r_wave >= rows) return;

  const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
  const u32 r = r_wave + sub;
  const u32 rr = (r < rows) ? r : (rows - 1u);
  const u32 s = tokid();
  const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[s]);

  const G u8 *rw = (const G u8 *)(qw + (unsigned long)e * (unsigned long)qwn + (unsigned long)rr * (unsigned long)nb * 4ul);
  const G f16 *rs = (const G f16 *)(qs + (unsigned long)e * (unsigned long)qsn) + (unsigned long)rr * (unsigned long)nb;
  const G u8 *xp = rq_ptr_u(xq + (unsigned long)s * (unsigned long)astr);
  const G u32 *aw = (const G u32 *)xp;
  const G float *da = (const G float *)(xp + 32ul * (unsigned long)nb);
  G float *o = (G float *)((G u8 *)y + (unsigned long)s * (unsigned long)bstr);

  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  const u32 n_steps = (nb + 7u) >> 3u;
  u32 b_c = sl;
  const int v0_ok = (b_c < nb);
  u32x4 w_c = v0_ok ? ink_r7_fl_ld4(rw + (unsigned long)b_c * 16ul) : (u32x4)(0u);
  float sc_c = v0_ok ? (float)rs[b_c] : 0.0f;
  u32x4 x0_c = v0_ok ? ink_r7_fl_ld4(aw + (unsigned long)b_c * 8ul) : (u32x4)(0u);
  u32x4 x1_c = v0_ok ? ink_r7_fl_ld4(aw + (unsigned long)b_c * 8ul + 4ul) : (u32x4)(0u);
  float d_c = v0_ok ? da[b_c] : 0.0f;

#pragma unroll 1
  for (u32 st = 0u; st < n_steps; st++) {
    const u32 b_n = b_c + 8u;
    const int ok_n = (b_n < nb);
    const u32x4 w_n = ok_n ? ink_r7_fl_ld4(rw + (unsigned long)b_n * 16ul) : (u32x4)(0u);
    const float sc_n = ok_n ? (float)rs[b_n] : 0.0f;
    const u32x4 x0_n = ok_n ? ink_r7_fl_ld4(aw + (unsigned long)b_n * 8ul) : (u32x4)(0u);
    const u32x4 x1_n = ok_n ? ink_r7_fl_ld4(aw + (unsigned long)b_n * 8ul + 4ul) : (u32x4)(0u);
    const float d_n = ok_n ? da[b_n] : 0.0f;
    __builtin_amdgcn_sched_barrier(0);

    if (b_c < nb) {
      const float prod = sc_c * (float)ink_r7_fl_iq4_dot4(w_c, x0_c, x1_c);
      INK_R7_FL_FMA4(a0, a1, a2, a3, st & 3u, prod, d_c);
    }
    __builtin_amdgcn_sched_barrier(0);
    b_c = b_n; w_c = w_n; sc_c = sc_n; x0_c = x0_n; x1_c = x1_n; d_c = d_n;
  }

  const float acc = ink_r7_fl_oct_tree_f((a0 + a2) + (a1 + a3));
  if (sl == 0u && r < rows) o[r] = acc;
}

/* ============================================================================
 * Kernel 2b (Bit 82): ie_inkling_gexp_iq4nl_bw_compact_r7
 * Multi-row wave-uniform IQ4_NL compact verify (T=2..16), double-buffered u32x4
 * prefetch + shared v_perm_b32 decoded nibbles across nj=2..4 pairs (0-bit-diff!).
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq4nl_bw_compact_r7(
    const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
    u32 rows, u32 nb, const G u32 *list, const G u32 *packs, const G u32 *n_packs,
    u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv, u32 kind, const G u8 *res, u32 rs) {
#pragma clang fp contract(off)
  (void)kind;
  if (rows == 0u || nb == 0u || !qw0 || !qs0 || !X || !Y || !list || !packs || !n_packs) return;

  const u32 by = tokid();
  const u32 npk = (u32)__builtin_amdgcn_readfirstlane((int)(*n_packs));
  const u32 rbase = wgid() << 5u;
  if (by >= npk || rbase >= rows) return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)packs[by]);
  const u32 e = desc & 0xFFu;
  const u32 lo = (desc >> 8u) & 0xFFFu;
  const u32 nj = (desc >> 20u) & 0xFu;
  if (nj == 0u) return;

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 r_wave = rbase + (wv << 2u);
  if (r_wave >= rows) return;

  const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
  const u32 r = r_wave + sub;
  const u32 rr = (r < rows) ? r : (rows - 1u);

  const G u8 *rw = (const G u8 *)(qw0 + (unsigned long)e * (unsigned long)qwn + (unsigned long)rr * (unsigned long)nb * 4ul);
  const G f16 *rs_p = (const G f16 *)(qs0 + (unsigned long)e * (unsigned long)qsn) + (unsigned long)rr * (unsigned long)nb;
  const unsigned long x_scale_off = 32ul * (unsigned long)nb;

  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
  const G u8 *xp0 = rq_ptr_u(X + (unsigned long)ink_r7_fl_div_x(p0, xdiv) * (unsigned long)xs);
  const G u32 *aw0 = (const G u32 *)xp0;
  const G float *da0 = (const G float *)(xp0 + x_scale_off);
  const u32 n_steps = (nb + 7u) >> 3u;

  if (nj == 1u) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    u32 b_c = sl;
    const int ok0 = (b_c < nb);
    u32x4 w_c = ok0 ? ink_r7_fl_ld4(rw + (unsigned long)b_c * 16ul) : (u32x4)(0u);
    float sc_c = ok0 ? (float)rs_p[b_c] : 0.0f;
    u32x4 x0_c = ok0 ? ink_r7_fl_ld4(aw0 + (unsigned long)b_c * 8ul) : (u32x4)(0u);
    u32x4 x1_c = ok0 ? ink_r7_fl_ld4(aw0 + (unsigned long)b_c * 8ul + 4ul) : (u32x4)(0u);
    float d_c = ok0 ? da0[b_c] : 0.0f;

#pragma unroll 1
    for (u32 st = 0u; st < n_steps; st++) {
      const u32 b_n = b_c + 8u;
      const int ok_n = (b_n < nb);
      const u32x4 w_n = ok_n ? ink_r7_fl_ld4(rw + (unsigned long)b_n * 16ul) : (u32x4)(0u);
      const float sc_n = ok_n ? (float)rs_p[b_n] : 0.0f;
      const u32x4 x0_n = ok_n ? ink_r7_fl_ld4(aw0 + (unsigned long)b_n * 8ul) : (u32x4)(0u);
      const u32x4 x1_n = ok_n ? ink_r7_fl_ld4(aw0 + (unsigned long)b_n * 8ul + 4ul) : (u32x4)(0u);
      const float d_n = ok_n ? da0[b_n] : 0.0f;
      __builtin_amdgcn_sched_barrier(0);

      if (b_c < nb) {
        const float prod = sc_c * (float)ink_r7_fl_iq4_dot4(w_c, x0_c, x1_c);
        INK_R7_FL_FMA4(a0, a1, a2, a3, st & 3u, prod, d_c);
      }
      __builtin_amdgcn_sched_barrier(0);
      b_c = b_n; w_c = w_n; sc_c = sc_n; x0_c = x0_n; x1_c = x1_n; d_c = d_n;
    }

    const float acc0 = ink_r7_fl_oct_tree_f((a0 + a2) + (a1 + a3));
    if (sl == 0u && r < rows) {
      G float *yp0 = (G float *)(Y + (unsigned long)p0 * (unsigned long)ys);
      const G float *rp0 = res ? (const G float *)(res + (unsigned long)p0 * (unsigned long)rs) : (const G float *)0;
      yp0[r] = rp0 ? (rp0[r] + acc0) : acc0;
    }
    return;
  }

  /* nj in 2..4: double-buffered u32x4 weights + shared v_perm_b32 LUT decode. */
  const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 1u]);
  const u32 p2 = (nj > 2u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 2u]) : p0;
  const u32 p3 = (nj > 3u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 3u]) : p0;
  const G u8 *xp1 = rq_ptr_u(X + (unsigned long)ink_r7_fl_div_x(p1, xdiv) * (unsigned long)xs);
  const G u8 *xp2 = (nj > 2u) ? rq_ptr_u(X + (unsigned long)ink_r7_fl_div_x(p2, xdiv) * (unsigned long)xs) : xp0;
  const G u8 *xp3 = (nj > 3u) ? rq_ptr_u(X + (unsigned long)ink_r7_fl_div_x(p3, xdiv) * (unsigned long)xs) : xp0;
  const G u32 *aw1 = (const G u32 *)xp1, *aw2 = (const G u32 *)xp2, *aw3 = (const G u32 *)xp3;
  const G float *da1 = (const G float *)(xp1 + x_scale_off);
  const G float *da2 = (const G float *)(xp2 + x_scale_off);
  const G float *da3 = (const G float *)(xp3 + x_scale_off);

  float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
  float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
  float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
  float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;

  u32 b_c = sl;
  const int ok0 = (b_c < nb);
  u32x4 w_c = ok0 ? ink_r7_fl_ld4(rw + (unsigned long)b_c * 16ul) : (u32x4)(0u);
  float sc_c = ok0 ? (float)rs_p[b_c] : 0.0f;

#pragma unroll 1
  for (u32 st = 0u; st < n_steps; st++) {
    const u32 b_n = b_c + 8u;
    const int ok_n = (b_n < nb);
    const u32x4 w_n = ok_n ? ink_r7_fl_ld4(rw + (unsigned long)b_n * 16ul) : (u32x4)(0u);
    const float sc_n = ok_n ? (float)rs_p[b_n] : 0.0f;
    __builtin_amdgcn_sched_barrier(0);

    if (b_c < nb) {
      const u32 wx = w_c.x, wy = w_c.y, wz = w_c.z, ww = w_c.w;
      const u32 wl0 = ie_iq4_map(wx & 0x0F0F0F0Fu), wh0 = ie_iq4_map((wx >> 4u) & 0x0F0F0F0Fu);
      const u32 wl1 = ie_iq4_map(wy & 0x0F0F0F0Fu), wh1 = ie_iq4_map((wy >> 4u) & 0x0F0F0F0Fu);
      const u32 wl2 = ie_iq4_map(wz & 0x0F0F0F0Fu), wh2 = ie_iq4_map((wz >> 4u) & 0x0F0F0F0Fu);
      const u32 wl3 = ie_iq4_map(ww & 0x0F0F0F0Fu), wh3 = ie_iq4_map((ww >> 4u) & 0x0F0F0F0Fu);
      const u32 j_acc = st & 3u;

      const u32x4 x00 = ink_r7_fl_ld4(aw0 + (unsigned long)b_c * 8ul);
      const u32x4 x01 = ink_r7_fl_ld4(aw0 + (unsigned long)b_c * 8ul + 4ul);
      const float prod0 = sc_c * (float)ink_r7_fl_iq4_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x00, x01);
      INK_R7_FL_FMA4(a0_0, a0_1, a0_2, a0_3, j_acc, prod0, da0[b_c]);

      const u32x4 x10 = ink_r7_fl_ld4(aw1 + (unsigned long)b_c * 8ul);
      const u32x4 x11 = ink_r7_fl_ld4(aw1 + (unsigned long)b_c * 8ul + 4ul);
      const float prod1 = sc_c * (float)ink_r7_fl_iq4_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x10, x11);
      INK_R7_FL_FMA4(a1_0, a1_1, a1_2, a1_3, j_acc, prod1, da1[b_c]);

      if (nj > 2u) {
        const u32x4 x20 = ink_r7_fl_ld4(aw2 + (unsigned long)b_c * 8ul);
        const u32x4 x21 = ink_r7_fl_ld4(aw2 + (unsigned long)b_c * 8ul + 4ul);
        const float prod2 = sc_c * (float)ink_r7_fl_iq4_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x20, x21);
        INK_R7_FL_FMA4(a2_0, a2_1, a2_2, a2_3, j_acc, prod2, da2[b_c]);
        if (nj > 3u) {
          const u32x4 x30 = ink_r7_fl_ld4(aw3 + (unsigned long)b_c * 8ul);
          const u32x4 x31 = ink_r7_fl_ld4(aw3 + (unsigned long)b_c * 8ul + 4ul);
          const float prod3 = sc_c * (float)ink_r7_fl_iq4_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, x30, x31);
          INK_R7_FL_FMA4(a3_0, a3_1, a3_2, a3_3, j_acc, prod3, da3[b_c]);
        }
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    b_c = b_n; w_c = w_n; sc_c = sc_n;
  }

  const float acc0 = ink_r7_fl_oct_tree_f((a0_0 + a0_2) + (a0_1 + a0_3));
  const float acc1 = ink_r7_fl_oct_tree_f((a1_0 + a1_2) + (a1_1 + a1_3));
  if (sl == 0u && r < rows) {
    G float *yp0 = (G float *)(Y + (unsigned long)p0 * (unsigned long)ys);
    const G float *rp0 = res ? (const G float *)(res + (unsigned long)p0 * (unsigned long)rs) : (const G float *)0;
    yp0[r] = rp0 ? (rp0[r] + acc0) : acc0;
    G float *yp1 = (G float *)(Y + (unsigned long)p1 * (unsigned long)ys);
    const G float *rp1 = res ? (const G float *)(res + (unsigned long)p1 * (unsigned long)rs) : (const G float *)0;
    yp1[r] = rp1 ? (rp1[r] + acc1) : acc1;
  }
  if (nj > 2u) {
    const float acc2 = ink_r7_fl_oct_tree_f((a2_0 + a2_2) + (a2_1 + a2_3));
    if (sl == 0u && r < rows) {
      G float *yp2 = (G float *)(Y + (unsigned long)p2 * (unsigned long)ys);
      const G float *rp2 = res ? (const G float *)(res + (unsigned long)p2 * (unsigned long)rs) : (const G float *)0;
      yp2[r] = rp2 ? (rp2[r] + acc2) : acc2;
    }
  }
  if (nj > 3u) {
    const float acc3 = ink_r7_fl_oct_tree_f((a3_0 + a3_2) + (a3_1 + a3_3));
    if (sl == 0u && r < rows) {
      G float *yp3 = (G float *)(Y + (unsigned long)p3 * (unsigned long)ys);
      const G float *rp3 = res ? (const G float *)(res + (unsigned long)p3 * (unsigned long)rs) : (const G float *)0;
      yp3[r] = rp3 ? (rp3[r] + acc3) : acc3;
    }
  }
}

/* ============================================================================
 * Kernel 3 (Bit 83): ie_inkling_gemv_q8_bw_r7
 * Multi-row wave-uniform dense Q8_0 GEMV for T=1 & verify T=2..16:
 *   - 4 rows/wave, 8 lanes/row -> 100% lane utilization on nb=72 (9x8) & nb=96 (12x8)
 *     (eliminates ie_gemv_q8q8_tr's 75%-idle tail & TH_LOAD_NT bypass!).
 *   - 1x DRAM weight read across T <= 8 tokens (SSA v8f a0..a3, 0 spills, 0 LDS).
 *   - Drop-in signature for ie_gemv_q8q8_tr (ys/rs in floats, xs in bytes).
 *   - 100% bitwise-exact (0-bit-diff) vs ie_gemv_q8q8 & ie_gemv_q8q8_tr.
 * ========================================================================== */

KERNEL ie_inkling_gemv_q8_bw_r7(
    const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
    u32 rows, u32 nb, const G float *bias, const G float *res,
    u32 T, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  if (T == 0u || rows == 0u || nb < 8u || (nb & 7u) != 0u) return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 r = (wgid() << 5u) + (w << 2u) + (l >> 3u);
  if (r >= rows) return;

  const G u32 *row_qw = qw + (unsigned long)r * (unsigned long)nb * 8ul;
  const G f16 *row_qs = qs + (unsigned long)r * (unsigned long)nb;
  const u32 n_blk = nb >> 3u;

  if (T == 1u) {
    /* Double-buffered u32x4 prefetch for single-token decode (<= 64 VGPR). */
    const G u32 *aw = (const G u32 *)xq;
    const G float *da = (const G float *)(xq + (unsigned long)nb * 32ul);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    u32 b_c = sub;
    u32x4 w0_c = ink_r7_fl_ld4(row_qw + (unsigned long)b_c * 8ul);
    u32x4 w1_c = ink_r7_fl_ld4(row_qw + (unsigned long)b_c * 8ul + 4ul);
    float sw_c = (float)row_qs[b_c];
    u32x4 x0_c = ink_r7_fl_ld4(aw + (unsigned long)b_c * 8ul);
    u32x4 x1_c = ink_r7_fl_ld4(aw + (unsigned long)b_c * 8ul + 4ul);
    float da_c = da[b_c];

#pragma unroll 1
    for (u32 st = 0u; st < n_blk; st++) {
      const u32 b_n = b_c + 8u;
      const int has_n = (st + 1u < n_blk);
      const u32x4 w0_n = has_n ? ink_r7_fl_ld4(row_qw + (unsigned long)b_n * 8ul) : (u32x4)(0u);
      const u32x4 w1_n = has_n ? ink_r7_fl_ld4(row_qw + (unsigned long)b_n * 8ul + 4ul) : (u32x4)(0u);
      const float sw_n = has_n ? (float)row_qs[b_n] : 0.0f;
      const u32x4 x0_n = has_n ? ink_r7_fl_ld4(aw + (unsigned long)b_n * 8ul) : (u32x4)(0u);
      const u32x4 x1_n = has_n ? ink_r7_fl_ld4(aw + (unsigned long)b_n * 8ul + 4ul) : (u32x4)(0u);
      const float da_n = has_n ? da[b_n] : 0.0f;
      __builtin_amdgcn_sched_barrier(0);

      const int dot = ink_r7_fl_q8_dot8(w0_c, w1_c, x0_c, x1_c);
      const float term = (sw_c * da_c) * (float)dot;
      INK_R7_FL_ADD4(a0, a1, a2, a3, st & 3u, term);
      __builtin_amdgcn_sched_barrier(0);

      b_c = b_n; w0_c = w0_n; w1_c = w1_n; sw_c = sw_n; x0_c = x0_n; x1_c = x1_n; da_c = da_n;
    }

    const float s = ink_r7_fl_oct_tree_f((a0 + a2) + (a1 + a3));
    if (sub == 0u) y[r] = epilogue(s, r, bias, res);
    return;
  }

  /* Verify T = 2..16: 8-token chunks, 8 lanes/row, 100% lane util on nb=72/96! */
  for (u32 t0 = 0u; t0 < T; t0 += INK_R7_FL_Q8_TMAX) {
    const u32 tc = (T - t0 < INK_R7_FL_Q8_TMAX) ? (T - t0) : INK_R7_FL_Q8_TMAX;
    ink_r7_fl_f8 a0 = (ink_r7_fl_f8)(0.0f);
    ink_r7_fl_f8 a1 = (ink_r7_fl_f8)(0.0f);
    ink_r7_fl_f8 a2 = (ink_r7_fl_f8)(0.0f);
    ink_r7_fl_f8 a3 = (ink_r7_fl_f8)(0.0f);

#pragma clang loop unroll(disable)
    for (u32 st = 0u; st < n_blk; st++) {
      const u32 st_u = (u32)__builtin_amdgcn_readfirstlane((int)st);
      const u32 slot = (u32)__builtin_amdgcn_readfirstlane((int)(st_u & 3u));
      const u32 b = (st_u << 3u) + sub;

      const u32x4 w0 = ink_r7_fl_ld4(row_qw + (unsigned long)b * 8ul);
      const u32x4 w1 = ink_r7_fl_ld4(row_qw + (unsigned long)b * 8ul + 4ul);
      const float sw = (float)row_qs[b];

#pragma unroll
      for (u32 j = 0u; j < INK_R7_FL_Q8_TMAX; j++) {
        if (j < tc) {
          const G u8 *xj = xq + (unsigned long)(t0 + j) * (unsigned long)xs;
          const G u32 *aw = (const G u32 *)xj;
          const u32x4 x0 = ink_r7_fl_ld4(aw + (unsigned long)b * 8ul);
          const u32x4 x1 = ink_r7_fl_ld4(aw + (unsigned long)b * 8ul + 4ul);
          const float da = ((const G float *)(xj + (unsigned long)nb * 32ul))[b];
          const int dot = ink_r7_fl_q8_dot8(w0, w1, x0, x1);
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

#pragma unroll
    for (u32 j = 0u; j < INK_R7_FL_Q8_TMAX; j++) {
      if (j < tc) {
        const float s = ink_r7_fl_oct_tree_f((a0[j] + a2[j]) + (a1[j] + a3[j]));
        if (sub == 0u) {
          const G float *rp = res ? (res + (unsigned long)(t0 + j) * (unsigned long)rs) : (const G float *)0;
          y[(unsigned long)(t0 + j) * (unsigned long)ys + (unsigned long)r] = epilogue(s, r, bias, rp);
        }
      }
    }
  }
}

#undef INK_R7_FL_FMA4
#undef INK_R7_FL_ADD4
#undef INK_R7_FL_Q8_TMAX

#endif /* IE_MEGA */
