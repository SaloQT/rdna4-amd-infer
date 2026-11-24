/* ============================================================================
 * Inkling-2 Round 10: FLUTE IQ3_XXS GPU Repacker & Surgical gfx1201 MALL
 * Temporal-Hint Kernels (H34 + H01 + H35)
 * Target: RX 9070 XT (gfx1201, RDNA4, wave32, NT=256, 64 MiB Infinity Cache)
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r10_fm_ /
 * INK_R10_FM_ and all macros are #undef'd at EOF.
 *
 * Temporal-Hint Mechanism (verified via llvm-objdump -d on gfx1201):
 *   - Streaming single-read weights (TH_LOAD_NT):
 *       __builtin_nontemporal_load(...) -> global_load_{b128,b64,b32,u16} ... th:TH_LOAD_NT
 *       (full SIInsertWaitcnts tracking, 0 extra SGPRs, 0 race hazards).
 *   - Reused activations & looped blk.6..17 dense Q8_0 weights (TH_LOAD_HT):
 *       __builtin_amdgcn_make_buffer_rsrc(..., 0x31004000u) +
 *       __builtin_amdgcn_raw_buffer_load_{b128,b64,b32,b16}(..., aux=2)
 *       -> buffer_load_{b128,b64,b32,u16} ... th:TH_LOAD_HT
 *       (official RDNA4 CK_TILE_BUFFER_RESOURCE_3RD_DWORD, full SIInsertWaitcnts
 *       tracking, 0 extra VGPRs, hardware bounds-checked, 0 race hazards).
 *
 * Audit of ie_inkling_r7_flute_gemv.inc (bit 81):
 *   - Correctness: 0 bit_cast bugs (all vector lanes extracted to scalars),
 *     exact 98 B -> 98 B SoA plane offsets and exact fp-contract-off SITU math.
 *   - Performance flaws fixed in _r10 below:
 *     1. r7 t1 used conditional `if (sb + 1u < mb)` + zero-init inside K-loop;
 *        r10 uses branch-free clamped prefetch + TH_LOAD_NT weights + TH_LOAD_HT X.
 *     2. r7 compact (nj=2..4) had 0 weight prefetching, 32 simultaneous acc VGPRs
 *        (ag + au), and reloaded X from global memory twice per superblock;
 *        r10 compact uses wave-split (Waves 0..3 gate, Waves 4..7 up), LDS-staged X
 *        loaded via TH_LOAD_HT, 1-SB lookahead, and TH_LOAD_NT strictly on nj==1.
 *
 * Exported Kernels (100% bitwise-exact 0-bit-diff parity, 0/0 spills, priv=0):
 *   1. ie_inkling_repack_iq3xxs_flute_r10          (H34 + H01 GPU load-time repacker)
 *   2. ie_inkling_gexp_iq3xxs_flute_situ_t1_nt_r10 (H34 + H35 FLUTE T=1 NT/HT)
 *   3. ie_inkling_gexp_iq3xxs_flute_situ_compact_nt_r10 (H34 + H35 FLUTE verify NT/HT)
 *   4. ie_inkling_gexp_iq3xxs_t1_nt_r10            (H35 AoS IQ3_XXS T=1 NT/HT)
 *   5. ie_inkling_gexp_iq4nl_tail_t1_nt_r10        (H35 IQ4_NL 4-op tail T=1 NT/HT)
 *   6. ie_inkling_gexp_iq4nl_t1_nt_r10             (H35 IQ4_NL standalone T=1 NT/HT)
 *   7. ie_inkling_q4k_head_ver_nt_r10              (H35 Q4_K single-pass T<=8 NT/HT)
 *   8. ie_inkling_gemv_q8_ht_r10                   (H35 looped blk.6..17 Q8_0 HT/HT)
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R10_FM_RSRC_W3   0x31004000u /* AMD gfx11/gfx12 raw buffer descriptor word 3 (OOB_SELECT=3) */
#define INK_R10_FM_TH_RT     0
#define INK_R10_FM_TH_NT     1           /* gfx1201 th:TH_LOAD_NT (single-read streaming) */
#define INK_R10_FM_TH_HT     2           /* gfx1201 th:TH_LOAD_HT (high-temporal MALL retention) */
#define INK_R10_FM_Q4K_TMAX  8u
#define INK_R10_FM_Q8_CT     8u
#define INK_R10_FM_LDS_MAXNB 96u
#define INK_R10_FM_SB_BYTES  98u

typedef unsigned int ink_r10_fm_u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned int ink_r10_fm_u32_a2 __attribute__((aligned(2)));
typedef u32x4 ink_r10_fm_u32x4_a4 __attribute__((aligned(4)));
typedef ink_r10_fm_u32x2 ink_r10_fm_u32x2_a4 __attribute__((aligned(4)));

struct __attribute__((packed, aligned(4))) ink_r10_fm_u32x4p {
  u32x4 v;
};

struct __attribute__((packed, aligned(4))) ink_r10_fm_u32x2p {
  ink_r10_fm_u32x2 v;
};

/* Wave-uniform RDNA4 buffer resource descriptor (128-bit V# in 4 SGPRs). */
static inline __attribute__((always_inline)) __amdgpu_buffer_rsrc_t ink_r10_fm_rsrc(
    const G void *base_u, u32 nbytes) {
  G void *p = (G void *)rq_ptr_u((const G u8 *)base_u);
  return __builtin_amdgcn_make_buffer_rsrc(p, 0, nbytes, INK_R10_FM_RSRC_W3);
}

/* Streaming non-temporal global loads (th:TH_LOAD_NT, full SIInsertWaitcnts). */
static inline __attribute__((always_inline)) u32x4 ink_r10_fm_nt_ld4(const G void *p) {
  return __builtin_nontemporal_load((const G ink_r10_fm_u32x4_a4 *)p);
}

static inline __attribute__((always_inline)) ink_r10_fm_u32x2 ink_r10_fm_nt_ld2(const G void *p) {
  return __builtin_nontemporal_load((const G ink_r10_fm_u32x2_a4 *)p);
}

static inline __attribute__((always_inline)) u32 ink_r10_fm_nt_ld1(const G void *p) {
  return __builtin_nontemporal_load((const G u32 *)p);
}

static inline __attribute__((always_inline)) u32 ink_r10_fm_nt_a32(const G u8 *p, u32 o) {
  return __builtin_nontemporal_load((const G ink_r10_fm_u32_a2 *)(p + o));
}

static inline __attribute__((always_inline)) u32 ink_r10_fm_nt_u16(const G void *p) {
  return (u32)__builtin_nontemporal_load((const G u16 *)p);
}

static inline __attribute__((always_inline)) f16 ink_r10_fm_nt_f16(const G void *p) {
  const u16 bits = __builtin_nontemporal_load((const G u16 *)p);
  return __builtin_bit_cast(f16, bits);
}

/* High-temporal RDNA4 buffer loads (th:TH_LOAD_HT, full SIInsertWaitcnts). */
static inline __attribute__((always_inline)) u32x4 ink_r10_fm_ht_ld4(
    __amdgpu_buffer_rsrc_t r, u32 voff) {
  return __builtin_amdgcn_raw_buffer_load_b128(r, voff, 0, INK_R10_FM_TH_HT);
}

static inline __attribute__((always_inline)) u32 ink_r10_fm_ht_u32(
    __amdgpu_buffer_rsrc_t r, u32 voff) {
  return __builtin_amdgcn_raw_buffer_load_b32(r, voff, 0, INK_R10_FM_TH_HT);
}

static inline __attribute__((always_inline)) float ink_r10_fm_ht_f32(
    __amdgpu_buffer_rsrc_t r, u32 voff) {
  const u32 bits = __builtin_amdgcn_raw_buffer_load_b32(r, voff, 0, INK_R10_FM_TH_HT);
  return I2F((int)bits);
}

static inline __attribute__((always_inline)) float ink_r10_fm_ht_f16f(
    __amdgpu_buffer_rsrc_t r, u32 voff) {
  const u16 bits = __builtin_amdgcn_raw_buffer_load_b16(r, voff, 0, INK_R10_FM_TH_HT);
  return (float)__builtin_bit_cast(f16, bits);
}

/* Regular-temporal helpers (used for shared nj>=2 verify weights). */
static inline __attribute__((always_inline)) u32x4 ink_r10_fm_rt_ld4(const G void *p) {
  return ((const G struct ink_r10_fm_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) ink_r10_fm_u32x2 ink_r10_fm_rt_ld2(const G void *p) {
  return ((const G struct ink_r10_fm_u32x2p *)p)->v;
}

static inline __attribute__((always_inline)) u32 ink_r10_fm_div_x(u32 pid, u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

static inline __attribute__((always_inline)) float ink_r10_fm_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline __attribute__((always_inline)) float ink_r10_fm_oct_tree_f(float v) {
#pragma clang fp contract(off)
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}

static inline __attribute__((always_inline)) int ink_r10_fm_q8_dot8(
    u32x4 w0, u32x4 w1, u32x4 a0, u32x4 a1) {
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

static inline __attribute__((always_inline)) int ink_r10_fm_iq4_dot4(
    u32x4 w4, u32x4 a0, u32x4 a1) {
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

static inline __attribute__((always_inline)) float ink_r10_fm_iq3_dot_pre(
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

static inline __attribute__((always_inline)) float ink_r10_fm_iq3_stream_dot(
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

#define INK_R10_FM_ADD4(a0_, a1_, a2_, a3_, j_, v_) \
  do {                                              \
    if ((j_) == 0u) (a0_) += (v_);                  \
    else if ((j_) == 1u) (a1_) += (v_);             \
    else if ((j_) == 2u) (a2_) += (v_);             \
    else (a3_) += (v_);                             \
  } while (0)

#define INK_R10_FM_FMA4(a0_, a1_, a2_, a3_, j_, v_, d_)       \
  do {                                                        \
    if ((j_) == 0u) (a0_) = __builtin_fmaf((v_), (d_), (a0_)); \
    else if ((j_) == 1u) (a1_) = __builtin_fmaf((v_), (d_), (a1_)); \
    else if ((j_) == 2u) (a2_) = __builtin_fmaf((v_), (d_), (a2_)); \
    else (a3_) = __builtin_fmaf((v_), (d_), (a3_));           \
  } while (0)

/* ============================================================================
 * Kernel 1 (H34 + H01): ie_inkling_repack_iq3xxs_flute_r10
 * GPU load-time AoS (98 B) -> FLUTE 4-row SoA (256 B qs | 128 B aux | 8 B d)
 * repacker matching ie_inkling_r7_flute_gemv.inc:15-41 byte-for-byte.
 *
 * Grid:  (((rows >> 2u) + 7u) / 8u, n_exp, 1u), NT = 256 (1 wave per 4-row rg).
 *        For Inkling-2 egu (rows=2304, nb=72, n_exp=128): grid = (72, n_b, 1).
 * Usage: Disjoint src -> dst (1..16-expert temp scratch + D2D inside gpu_open()
 *        after inkling_mod load, 0 permanent extra VRAM; matches wave9-host-specs.md
 *        param order {src, dst, rows, nb, q_bytes, n_exp}!).
 * ========================================================================== */

KERNEL ie_inkling_repack_iq3xxs_flute_r10(
    const G u8 *src, G u8 *dst, u32 rows, u32 nb, u32 q_bytes, u32 n_exp) {
  if (!src || !dst || src == dst || rows == 0u || (rows & 3u) != 0u ||
      nb < 8u || (nb & 7u) != 0u || n_exp == 0u)
    return;

  const u32 e = tokid();
  if (e >= n_exp) return;

  const u32 n_rg = rows >> 2u;
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 rg = (wgid() << 3u) + wv;
  if (rg >= n_rg) return;

  const u32 mb = nb >> 3u;
  const unsigned long n_sb = (unsigned long)n_rg * (unsigned long)mb;
  if ((unsigned long)q_bytes < n_sb * 392ul) return;

  const G u8 *src_e = rq_ptr_u(src + (unsigned long)e * (unsigned long)q_bytes);
  G u8 *dst_e = (G u8 *)rq_ptr_u(dst + (unsigned long)e * (unsigned long)q_bytes);

  G ink_r10_fm_u32x2 *p_qs  = (G ink_r10_fm_u32x2 *)dst_e;
  G u32              *p_aux = (G u32 *)(dst_e + n_sb * 256ul);
  G u16              *p_d   = (G u16 *)(dst_e + n_sb * 384ul);

  const u32 l = lane();
  const u32 k = l >> 3u;
  const u32 s = l & 7u;
  const u32 row = (rg << 2u) + k;
  const u32 qo = 2u + (s << 3u);
  const u32 auxo = 66u + (s << 2u);

  const G u8 *bp = src_e + (unsigned long)row * ((unsigned long)mb * INK_R10_FM_SB_BYTES);
  const unsigned long g0 = (unsigned long)rg * (unsigned long)mb;

#pragma unroll 1
  for (u32 sb = 0u; sb < mb; sb++) {
    const unsigned long gidx = g0 + (unsigned long)sb;
    const u32 q01 = ink_r10_fm_nt_a32(bp, qo);
    const u32 q23 = ink_r10_fm_nt_a32(bp, qo + 4u);
    const u32 aux = ink_r10_fm_nt_a32(bp, auxo);
    u32 d16 = 0u;
    if (s == 0u) d16 = ink_r10_fm_nt_u16(bp);

    /* Coalesced 256 B (qs) + 128 B (aux) + 8 B (d) stores matching r7 spec. */
    ((G struct ink_r10_fm_u32x2p *)(p_qs + gidx * 32ul + (unsigned long)l))->v =
        (ink_r10_fm_u32x2){q01, q23};
    p_aux[gidx * 32ul + (unsigned long)l] = aux;
    if (s == 0u) p_d[gidx * 4ul + (unsigned long)k] = (u16)d16;

    bp += INK_R10_FM_SB_BYTES;
  }
}

/* ============================================================================
 * Kernel 2 (H34 + H35): ie_inkling_gexp_iq3xxs_flute_situ_t1_nt_r10
 * T=1 FLUTE SoA IQ3_XXS Gate/Up + SITU + Q8_0:
 *   - TH_LOAD_NT on strictly single-read FLUTE planes (p_qs, p_aux, p_d)
 *   - TH_LOAD_HT on reused ffn_in_q8 (xq + dap)
 *   - Branch-free clamped 1-SB lookahead prefetch (fixes r7 inner branch)
 *   - Grid: ((eff + 31u) / 32u, K, 1u), <= 72 VGPR (4 WGs/CU), 1,152 B LDS.
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq3xxs_flute_situ_t1_nt_r10(
    const G u8 *qw, const G u8 *xq, G float *b, G u8 *q,
    u32 eff, u32 nb, const G u32 *route, u32 q_bytes,
    u32 astr, u32 bstr, u32 qstr, float beta, float lbeta) {
#pragma clang fp contract(off)
  static LDS u32   ink_r10_fm_ft1_lut[256u] __attribute__((aligned(16)));
  static LDS float ink_r10_fm_ft1_situ[32u] __attribute__((aligned(16)));

  if (eff == 0u || (eff & 31u) != 0u || nb < 8u || (nb & 7u) != 0u || !qw || !xq || (!b && !q)) return;
  const u32 blk = wgid();
  const u32 nb_out = eff >> 5u;
  if (blk >= nb_out) return;

  const u32 t0 = tid();
  ink_r10_fm_ft1_lut[t0] = ie_rq_iq3xxs_grid[t0];
  lbarrier();

  const u32 slot = tokid();
  const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[slot]);
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 r_in_blk = (wv << 2u) + k;
  const u32 mb = nb >> 3u;

  const unsigned long n_sb_tot = (unsigned long)(eff >> 1u) * (unsigned long)mb;
  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * (unsigned long)q_bytes);
  const G ink_r10_fm_u32x2 *p_qs  = (const G ink_r10_fm_u32x2 *)ebase;
  const G u32              *p_aux = (const G u32 *)(ebase + n_sb_tot * 256ul);
  const G u16              *p_d   = (const G u16 *)(ebase + n_sb_tot * 384ul);

  const u32 rg_g = (blk << 3u) + wv;
  const u32 rg_u = (eff >> 2u) + rg_g;
  const unsigned long gg0 = (unsigned long)rg_g * (unsigned long)mb;
  const unsigned long gu0 = (unsigned long)rg_u * (unsigned long)mb;

  const G u8 *xp = rq_ptr_u(xq + (unsigned long)slot * (unsigned long)astr);
  const __amdgpu_buffer_rsrc_t xr = ink_r10_fm_rsrc(xp, nb * 40u);
  const u32 da_base = nb << 5u;

  float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
  float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;

  /* Prologue: TH_LOAD_NT weights (sb=0 gate) + TH_LOAD_HT activations. */
  ink_r10_fm_u32x2 qg_c = ink_r10_fm_nt_ld2(p_qs + gg0 * 32ul + (unsigned long)l);
  u32 auxg_c = ink_r10_fm_nt_ld1(p_aux + gg0 * 32ul + (unsigned long)l);
  u32 dg_c = ink_r10_fm_nt_u16(p_d + gg0 * 4ul + (unsigned long)k);
  u32x4 x0_c = ink_r10_fm_ht_ld4(xr, s << 5u);
  u32x4 x1_c = ink_r10_fm_ht_ld4(xr, (s << 5u) + 16u);
  float dav_c = ink_r10_fm_ht_f32(xr, da_base + (s << 2u));
  __asm__ volatile("" : "+v"(dg_c), "+v"(auxg_c));

#pragma unroll 1
  for (u32 sb = 0u; sb < mb; sb++) {
    const u32 j_acc = sb & 3u;
    const unsigned long gu = gu0 + (unsigned long)sb;
    const ink_r10_fm_u32x2 qu_c = ink_r10_fm_nt_ld2(p_qs + gu * 32ul + (unsigned long)l);
    const u32 auxu_c = ink_r10_fm_nt_ld1(p_aux + gu * 32ul + (unsigned long)l);
    const u32 du_c = ink_r10_fm_nt_u16(p_d + gu * 4ul + (unsigned long)k);
    __builtin_amdgcn_sched_barrier(0);

    const float vg = ink_r10_fm_iq3_stream_dot(
        ink_r10_fm_ft1_lut, dg_c, auxg_c, qg_c.x, qg_c.y, x0_c, x1_c, dav_c);
    INK_R10_FM_ADD4(ag0, ag1, ag2, ag3, j_acc, vg);
    __builtin_amdgcn_sched_barrier(0);

    /* Branch-free clamped next-SB prefetch keeps SIInsertWaitcnts straight-line. */
    const u32 sb_n = (sb + 1u < mb) ? (sb + 1u) : sb;
    const unsigned long ggn = gg0 + (unsigned long)sb_n;
    const u32 bn = (sb_n << 3u) + s;
    const ink_r10_fm_u32x2 qg_n = ink_r10_fm_nt_ld2(p_qs + ggn * 32ul + (unsigned long)l);
    const u32 auxg_n = ink_r10_fm_nt_ld1(p_aux + ggn * 32ul + (unsigned long)l);
    const u32 dg_n = ink_r10_fm_nt_u16(p_d + ggn * 4ul + (unsigned long)k);
    const u32x4 x0_n = ink_r10_fm_ht_ld4(xr, bn << 5u);
    const u32x4 x1_n = ink_r10_fm_ht_ld4(xr, (bn << 5u) + 16u);
    const float dav_n = ink_r10_fm_ht_f32(xr, da_base + (bn << 2u));
    __builtin_amdgcn_sched_barrier(0);

    const float vu = ink_r10_fm_iq3_stream_dot(
        ink_r10_fm_ft1_lut, du_c, auxu_c, qu_c.x, qu_c.y, x0_c, x1_c, dav_c);
    INK_R10_FM_ADD4(au0, au1, au2, au3, j_acc, vu);
    __builtin_amdgcn_sched_barrier(0);

    qg_c = qg_n; auxg_c = auxg_n; dg_c = dg_n;
    x0_c = x0_n; x1_c = x1_n; dav_c = dav_n;
  }

  const float gv = ink_r10_fm_oct_tree_f((ag0 + ag2) + (ag1 + ag3));
  const float uv = ink_r10_fm_oct_tree_f((au0 + au2) + (au1 + au3));
  if (s == 0u) {
    const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
    const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
    ink_r10_fm_ft1_situ[r_in_blk] = gs * sigmoidf(gv) * us;
  }
  lbarrier();

  if (wv == 0u) {
    const float v = ink_r10_fm_ft1_situ[l];
    if (b) ((G float *)((G u8 *)b + (unsigned long)slot * (unsigned long)bstr))[(blk << 5u) + l] = v;
    if (q) quant_wave(v, blk, q + (unsigned long)slot * (unsigned long)qstr, nb_out);
  }
}

/* ============================================================================
 * Kernel 3 (H34 + H35 + H14): ie_inkling_gexp_iq3xxs_flute_situ_compact_nt_r10
 * Upgraded FLUTE SoA compact verify (T=2..16) fixing r7's missing prefetch &
 * 32-acc VGPR bloat via wave-split (Waves 0..3 gate, Waves 4..7 up) + LDS-staged
 * X (loaded via TH_LOAD_HT) + strict single-read TH_LOAD_NT on nj==1 weights
 * (TH_LOAD_RT on nj>=2 weights).
 * Grid: ((eff + 31u) / 32u, max_packs, 1u), <= 80 VGPR (4 WGs/CU), 15,872 B LDS.
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq3xxs_flute_situ_compact_nt_r10(
    const G u8 *qw, const G u8 *X, G float *b_out, G u8 *q_out,
    u32 eff, u32 nb, const G u32 *list, const G u32 *packs, const G u32 *n_packs,
    u32 q_bytes, u32 xs, u32 bstr, u32 qstr, u32 xdiv, float beta, float lbeta) {
#pragma clang fp contract(off)
  static LDS u32   ink_r10_fm_fc_lut[256u] __attribute__((aligned(16)));
  static LDS u32   ink_r10_fm_fc_xq[4u][INK_R10_FM_LDS_MAXNB * 8u] __attribute__((aligned(16)));
  static LDS float ink_r10_fm_fc_xd[4u][INK_R10_FM_LDS_MAXNB] __attribute__((aligned(16)));
  static LDS float ink_r10_fm_fc_g[4u][32u] __attribute__((aligned(16)));
  static LDS float ink_r10_fm_fc_u[4u][32u] __attribute__((aligned(16)));

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
  ink_r10_fm_fc_lut[t0] = ie_rq_iq3xxs_grid[t0];

  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
  const u32 p1 = (nj > 1u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 1u]) : p0;
  const u32 p2 = (nj > 2u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 2u]) : p0;
  const u32 p3 = (nj > 3u) ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 3u]) : p0;

  const G u8 *xp0 = rq_ptr_u(X + (unsigned long)ink_r10_fm_div_x(p0, xdiv) * (unsigned long)xs);
  const G u8 *xp1 = (nj > 1u) ? rq_ptr_u(X + (unsigned long)ink_r10_fm_div_x(p1, xdiv) * (unsigned long)xs) : xp0;
  const G u8 *xp2 = (nj > 2u) ? rq_ptr_u(X + (unsigned long)ink_r10_fm_div_x(p2, xdiv) * (unsigned long)xs) : xp0;
  const G u8 *xp3 = (nj > 3u) ? rq_ptr_u(X + (unsigned long)ink_r10_fm_div_x(p3, xdiv) * (unsigned long)xs) : xp0;

  const int use_lds_x = (nb <= INK_R10_FM_LDS_MAXNB);
  const u32 x_bytes = nb * 40u;
  const u32 da_off = nb << 5u;
  if (use_lds_x) {
    const u32 n_v4 = nb << 1u;
    for (u32 j = 0u; j < nj; j++) {
      const G u8 *xpj = (j == 0u) ? xp0 : ((j == 1u) ? xp1 : ((j == 2u) ? xp2 : xp3));
      const __amdgpu_buffer_rsrc_t xrj = ink_r10_fm_rsrc(xpj, x_bytes);
      for (u32 idx = t0; idx < n_v4; idx += NT) {
        ((LDSP u32x4 *)ink_r10_fm_fc_xq[j])[idx] = ink_r10_fm_ht_ld4(xrj, idx << 4u);
      }
      for (u32 idx = t0; idx < nb; idx += NT) {
        ink_r10_fm_fc_xd[j][idx] = ink_r10_fm_ht_f32(xrj, da_off + (idx << 2u));
      }
    }
  }
  lbarrier();

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 is_up = wv >> 2u;
  const u32 wv4 = wv & 3u;
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 mb = nb >> 3u;

  const unsigned long n_sb_tot = (unsigned long)(eff >> 1u) * (unsigned long)mb;
  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * (unsigned long)q_bytes);
  const G ink_r10_fm_u32x2 *p_qs  = (const G ink_r10_fm_u32x2 *)ebase;
  const G u32              *p_aux = (const G u32 *)(ebase + n_sb_tot * 256ul);
  const G u16              *p_d   = (const G u16 *)(ebase + n_sb_tot * 384ul);

  const u32 rg_half_base = (is_up ? (eff >> 2u) : 0u) + (blk << 3u) + wv4;

  if (nj == 1u) {
    /* Guaranteed single-read expert weights: TH_LOAD_NT + 1-SB lookahead, 2 sub-passes. */
    const __amdgpu_buffer_rsrc_t xr0 = ink_r10_fm_rsrc(xp0, x_bytes);
#pragma unroll 1
    for (u32 sp = 0u; sp < 2u; sp++) {
      const u32 rg = rg_half_base + (sp << 2u);
      const unsigned long g0 = (unsigned long)rg * (unsigned long)mb;
      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

      ink_r10_fm_u32x2 q_c = ink_r10_fm_nt_ld2(p_qs + g0 * 32ul + (unsigned long)l);
      u32 aux_c = ink_r10_fm_nt_ld1(p_aux + g0 * 32ul + (unsigned long)l);
      u32 d_c = ink_r10_fm_nt_u16(p_d + g0 * 4ul + (unsigned long)k);
      __asm__ volatile("" : "+v"(d_c), "+v"(aux_c));

#pragma unroll 1
      for (u32 sb = 0u; sb < mb; sb++) {
        const u32 b = (sb << 3u) + s;
        const u32 j_acc = sb & 3u;
        const u32 sb_n = (sb + 1u < mb) ? (sb + 1u) : sb;
        const unsigned long gn = g0 + (unsigned long)sb_n;

        const u32x4 x0 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_fm_fc_xq[0] + (b << 3u))
                                   : ink_r10_fm_ht_ld4(xr0, b << 5u);
        const u32x4 x1 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_fm_fc_xq[0] + (b << 3u) + 4u)
                                   : ink_r10_fm_ht_ld4(xr0, (b << 5u) + 16u);
        const float dav = use_lds_x ? ink_r10_fm_fc_xd[0][b]
                                    : ink_r10_fm_ht_f32(xr0, da_off + (b << 2u));

        const ink_r10_fm_u32x2 q_n = ink_r10_fm_nt_ld2(p_qs + gn * 32ul + (unsigned long)l);
        const u32 aux_n = ink_r10_fm_nt_ld1(p_aux + gn * 32ul + (unsigned long)l);
        const u32 d_n = ink_r10_fm_nt_u16(p_d + gn * 4ul + (unsigned long)k);
        __builtin_amdgcn_sched_barrier(0);

        const float v0 = ink_r10_fm_iq3_stream_dot(
            ink_r10_fm_fc_lut, d_c, aux_c, q_c.x, q_c.y, x0, x1, dav);
        INK_R10_FM_ADD4(a0, a1, a2, a3, j_acc, v0);
        __builtin_amdgcn_sched_barrier(0);
        q_c = q_n; aux_c = aux_n; d_c = d_n;
      }

      const float acc = ink_r10_fm_oct_tree_f((a0 + a2) + (a1 + a3));
      if (s == 0u) {
        const u32 r_loc = (sp << 4u) + (wv4 << 2u) + k;
        if (is_up) ink_r10_fm_fc_u[0][r_loc] = acc;
        else ink_r10_fm_fc_g[0][r_loc] = acc;
      }
    }
  } else {
    /* nj in 2..4: TH_LOAD_RT on weights (safe for multi-pack experts), LDS X,
     * 1-SB lookahead, wave-split cuts live acc VGPRs in half (16 vs 32)! */
    const unsigned long x_sc_off = 32ul * (unsigned long)nb;
    const G float *dap0 = (const G float *)(xp0 + x_sc_off);
    const G float *dap1 = (const G float *)(xp1 + x_sc_off);
    const G float *dap2 = (const G float *)(xp2 + x_sc_off);
    const G float *dap3 = (const G float *)(xp3 + x_sc_off);

#pragma unroll 1
    for (u32 sp = 0u; sp < 2u; sp++) {
      const u32 rg = rg_half_base + (sp << 2u);
      const unsigned long g0 = (unsigned long)rg * (unsigned long)mb;

      float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
      float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
      float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
      float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;

      ink_r10_fm_u32x2 q_c = ink_r10_fm_rt_ld2(p_qs + g0 * 32ul + (unsigned long)l);
      u32 aux_c = p_aux[g0 * 32ul + (unsigned long)l];
      u32 d_c = (u32)p_d[g0 * 4ul + (unsigned long)k];
      __asm__ volatile("" : "+v"(d_c), "+v"(aux_c));

#pragma unroll 1
      for (u32 sb = 0u; sb < mb; sb++) {
        const u32 b = (sb << 3u) + s;
        const u32 j_acc = sb & 3u;
        const u32 sb_n = (sb + 1u < mb) ? (sb + 1u) : sb;
        const unsigned long gn = g0 + (unsigned long)sb_n;

        const ink_r10_fm_u32x2 q_n = ink_r10_fm_rt_ld2(p_qs + gn * 32ul + (unsigned long)l);
        const u32 aux_n = p_aux[gn * 32ul + (unsigned long)l];
        const u32 d_n = (u32)p_d[gn * 4ul + (unsigned long)k];

        const u32 q01_c = q_c.x, q23_c = q_c.y;
        const u32 qp0 = q01_c & 0xFFFFu, qp1 = q01_c >> 16u;
        const u32 qp2 = q23_c & 0xFFFFu, qp3 = q23_c >> 16u;
        const u32 g_lo0 = ink_r10_fm_fc_lut[qp0 & 255u], g_hi0 = ink_r10_fm_fc_lut[qp0 >> 8u];
        const u32 g_lo1 = ink_r10_fm_fc_lut[qp1 & 255u], g_hi1 = ink_r10_fm_fc_lut[qp1 >> 8u];
        const u32 g_lo2 = ink_r10_fm_fc_lut[qp2 & 255u], g_hi2 = ink_r10_fm_fc_lut[qp2 >> 8u];
        const u32 g_lo3 = ink_r10_fm_fc_lut[qp3 & 255u], g_hi3 = ink_r10_fm_fc_lut[qp3 >> 8u];
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

        const u32x4 x0_0 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_fm_fc_xq[0] + (b << 3u))
                                     : ink_r10_fm_rt_ld4(xp0 + (b << 5u));
        const u32x4 x1_0 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_fm_fc_xq[0] + (b << 3u) + 4u)
                                     : ink_r10_fm_rt_ld4(xp0 + (b << 5u) + 16u);
        const float d0 = use_lds_x ? ink_r10_fm_fc_xd[0][b] : dap0[b];
        INK_R10_FM_ADD4(a0_0, a0_1, a0_2, a0_3, j_acc,
                        ink_r10_fm_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_0, x1_0, d0));

        const u32x4 x0_1 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_fm_fc_xq[1] + (b << 3u))
                                     : ink_r10_fm_rt_ld4(xp1 + (b << 5u));
        const u32x4 x1_1 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_fm_fc_xq[1] + (b << 3u) + 4u)
                                     : ink_r10_fm_rt_ld4(xp1 + (b << 5u) + 16u);
        const float d1 = use_lds_x ? ink_r10_fm_fc_xd[1][b] : dap1[b];
        INK_R10_FM_ADD4(a1_0, a1_1, a1_2, a1_3, j_acc,
                        ink_r10_fm_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_1, x1_1, d1));

        if (nj > 2u) {
          const u32x4 x0_2 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_fm_fc_xq[2] + (b << 3u))
                                       : ink_r10_fm_rt_ld4(xp2 + (b << 5u));
          const u32x4 x1_2 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_fm_fc_xq[2] + (b << 3u) + 4u)
                                       : ink_r10_fm_rt_ld4(xp2 + (b << 5u) + 16u);
          const float d2 = use_lds_x ? ink_r10_fm_fc_xd[2][b] : dap2[b];
          INK_R10_FM_ADD4(a2_0, a2_1, a2_2, a2_3, j_acc,
                          ink_r10_fm_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_2, x1_2, d2));

          if (nj > 3u) {
            const u32x4 x0_3 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_fm_fc_xq[3] + (b << 3u))
                                         : ink_r10_fm_rt_ld4(xp3 + (b << 5u));
            const u32x4 x1_3 = use_lds_x ? *(const LDSP u32x4 *)(ink_r10_fm_fc_xq[3] + (b << 3u) + 4u)
                                         : ink_r10_fm_rt_ld4(xp3 + (b << 5u) + 16u);
            const float d3 = use_lds_x ? ink_r10_fm_fc_xd[3][b] : dap3[b];
            INK_R10_FM_ADD4(a3_0, a3_1, a3_2, a3_3, j_acc,
                            ink_r10_fm_iq3_dot_pre(wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, dw, x0_3, x1_3, d3));
          }
        }
        __builtin_amdgcn_sched_barrier(0);
        q_c = q_n; aux_c = aux_n; d_c = d_n;
      }

      const u32 r_loc = (sp << 4u) + (wv4 << 2u) + k;
      const float acc0 = ink_r10_fm_oct_tree_f((a0_0 + a0_2) + (a0_1 + a0_3));
      const float acc1 = ink_r10_fm_oct_tree_f((a1_0 + a1_2) + (a1_1 + a1_3));
      if (s == 0u) {
        if (is_up) {
          ink_r10_fm_fc_u[0][r_loc] = acc0;
          ink_r10_fm_fc_u[1][r_loc] = acc1;
        } else {
          ink_r10_fm_fc_g[0][r_loc] = acc0;
          ink_r10_fm_fc_g[1][r_loc] = acc1;
        }
      }
      if (nj > 2u) {
        const float acc2 = ink_r10_fm_oct_tree_f((a2_0 + a2_2) + (a2_1 + a2_3));
        if (s == 0u) {
          if (is_up) ink_r10_fm_fc_u[2][r_loc] = acc2;
          else ink_r10_fm_fc_g[2][r_loc] = acc2;
        }
      }
      if (nj > 3u) {
        const float acc3 = ink_r10_fm_oct_tree_f((a3_0 + a3_2) + (a3_1 + a3_3));
        if (s == 0u) {
          if (is_up) ink_r10_fm_fc_u[3][r_loc] = acc3;
          else ink_r10_fm_fc_g[3][r_loc] = acc3;
        }
      }
    }
  }
  lbarrier();

  /* Waves 0..nj-1 run exact 32-lane SITU + wave32 Q8_0 quant in parallel! */
  if (wv < nj) {
    const u32 pw = (wv == 0u) ? p0 : ((wv == 1u) ? p1 : ((wv == 2u) ? p2 : p3));
    const float gv = ink_r10_fm_fc_g[wv][l];
    const float uv = ink_r10_fm_fc_u[wv][l];
    const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
    const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
    const float v = gs * sigmoidf(gv) * us;
    if (b_out) ((G float *)((G u8 *)b_out + (unsigned long)pw * (unsigned long)bstr))[(blk << 5u) + l] = v;
    if (q_out) quant_wave(v, blk, q_out + (unsigned long)pw * (unsigned long)qstr, nb_out);
  }
}

/* ============================================================================
 * Kernel 4 (H35): ie_inkling_gexp_iq3xxs_t1_nt_r10
 * Drop-in exact-SHA AoS T=1 IQ3_XXS GEMV (replaces ie_inkling_gexp_raw_iq3xxs_r3):
 *   - TH_LOAD_NT on strictly single-read AoS IQ3_XXS weights (780 MB/tok)
 *   - TH_LOAD_HT on reused ffn_in_q8 (xq + dap)
 *   - Branch-free clamped 1-SB lookahead prefetch
 *   - Grid: ((rows + 31u) / 32u, K, 1u), <= 64 VGPR (4 WGs/CU), 1,024 B LDS.
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq3xxs_t1_nt_r10(
    const G u8 *qw, const G u8 *xq, G float *y, u32 rows, u32 nb,
    const G u32 *route, u32 q_bytes, u32 astr, u32 bstr, u32 type) {
#pragma clang fp contract(off)
  static LDS u32 ink_r10_fm_aos_lut[256u] __attribute__((aligned(16)));
  if (type != IE_RQ_IQ3_XXS || rows == 0u || nb < 8u || (nb & 7u) != 0u) return;

  const u32 t0 = tid();
  ink_r10_fm_aos_lut[t0] = ie_rq_iq3xxs_grid[t0];
  lbarrier();

  const u32 slot = tokid();
  const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[slot]);
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), k = l >> 3u, s = l & 7u;
  const u32 r = (wgid() << 5u) + (wv << 2u) + k;
  const u32 rr = r < rows ? r : rows - 1u;
  const u32 mb = nb >> 3u;
  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * (unsigned long)q_bytes);
  const G u8 *bp = ebase + (unsigned long)rr * ((unsigned long)mb * INK_R10_FM_SB_BYTES);
  const G u8 *xp = rq_ptr_u(xq + (unsigned long)slot * (unsigned long)astr);
  const __amdgpu_buffer_rsrc_t xr = ink_r10_fm_rsrc(xp, nb * 40u);
  const u32 da_base = nb << 5u;

  const u32 qo = 2u + (s << 3u);
  const u32 auxo = 66u + (s << 2u);
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

  u32 d_c = ink_r10_fm_nt_u16(bp);
  u32 aux_c = ink_r10_fm_nt_a32(bp, auxo);
  u32 q01_c = ink_r10_fm_nt_a32(bp, qo);
  u32 q23_c = ink_r10_fm_nt_a32(bp, qo + 4u);
  u32x4 x0_c = ink_r10_fm_ht_ld4(xr, s << 5u);
  u32x4 x1_c = ink_r10_fm_ht_ld4(xr, (s << 5u) + 16u);
  float dav_c = ink_r10_fm_ht_f32(xr, da_base + (s << 2u));
  __asm__ volatile("" : "+v"(d_c), "+v"(aux_c), "+v"(q01_c), "+v"(q23_c));

#pragma unroll 1
  for (u32 sb = 0u; sb < mb; sb++) {
    const u32 j_acc = sb & 3u;
    const u32 sb_n = (sb + 1u < mb) ? (sb + 1u) : sb;
    const G u8 *bp_n = (sb + 1u < mb) ? (bp + INK_R10_FM_SB_BYTES) : bp;
    const u32 bn = (sb_n << 3u) + s;

    const u32 d_n = ink_r10_fm_nt_u16(bp_n);
    const u32 aux_n = ink_r10_fm_nt_a32(bp_n, auxo);
    const u32 q01_n = ink_r10_fm_nt_a32(bp_n, qo);
    const u32 q23_n = ink_r10_fm_nt_a32(bp_n, qo + 4u);
    const u32x4 x0_n = ink_r10_fm_ht_ld4(xr, bn << 5u);
    const u32x4 x1_n = ink_r10_fm_ht_ld4(xr, (bn << 5u) + 16u);
    const float dav_n = ink_r10_fm_ht_f32(xr, da_base + (bn << 2u));
    __builtin_amdgcn_sched_barrier(0);

    const float v = ink_r10_fm_iq3_stream_dot(
        ink_r10_fm_aos_lut, d_c, aux_c, q01_c, q23_c, x0_c, x1_c, dav_c);
    INK_R10_FM_ADD4(a0, a1, a2, a3, j_acc, v);
    __builtin_amdgcn_sched_barrier(0);

    bp = bp_n;
    d_c = d_n; aux_c = aux_n; q01_c = q01_n; q23_c = q23_n;
    x0_c = x0_n; x1_c = x1_n; dav_c = dav_n;
  }

  const float acc = ink_r10_fm_oct_tree_f((a0 + a2) + (a1 + a3));
  if (s == 0u && r < rows) {
    ((G float *)((G u8 *)y + (unsigned long)slot * (unsigned long)bstr))[r] = acc;
  }
}

/* ============================================================================
 * Kernel 5 (H35): ie_inkling_gexp_iq4nl_tail_t1_nt_r10
 * Drop-in exact-SHA replacement for ie_inkling_exp_down_tail_t1_r7 (19 args):
 *   - TH_LOAD_NT on strictly single-read IQ4_NL weights (qw, qs: 573 MB/tok)
 *   - TH_LOAD_HT on reused exp_mid_q8 (xq + da)
 *   - Exact ordered K-slot combine + 4-tap sconv + ring + residual add tail.
 *   - Grid: ((rows + 31u) / 32u, K, 1u), <= 84 VGPR (4 WGs/CU), 4 B LDS.
 * ========================================================================== */

static inline __attribute__((always_inline)) f32x4 ink_r10_fm_moe_combine4(
    const G f32x4 *y4, const G u32 *rt, u32 n4, u32 K, u32 i4) {
#pragma clang fp contract(off)
  const G float *w = (const G float *)(rt + K);
  f32x4 x0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (K == 8u) {
    const float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    const float w4 = w[4], w5 = w[5], w6 = w[6], w7 = w[7];
    const f32x4 y0 = y4[0ul * n4 + i4];
    const f32x4 y1 = y4[1ul * n4 + i4];
    const f32x4 y2 = y4[2ul * n4 + i4];
    const f32x4 y3 = y4[3ul * n4 + i4];
    const f32x4 y4v = y4[4ul * n4 + i4];
    const f32x4 y5 = y4[5ul * n4 + i4];
    const f32x4 y6 = y4[6ul * n4 + i4];
    const f32x4 y7 = y4[7ul * n4 + i4];
#pragma unroll
    for (u32 c = 0u; c < 4u; c++) {
      float acc = __builtin_fmaf(w0, y0[c], 0.0f);
      acc = __builtin_fmaf(w1, y1[c], acc);
      acc = __builtin_fmaf(w2, y2[c], acc);
      acc = __builtin_fmaf(w3, y3[c], acc);
      acc = __builtin_fmaf(w4, y4v[c], acc);
      acc = __builtin_fmaf(w5, y5[c], acc);
      acc = __builtin_fmaf(w6, y6[c], acc);
      acc = __builtin_fmaf(w7, y7[c], acc);
      x0[c] = acc;
    }
  } else {
    for (u32 k = 0u; k < K; k++) {
      const float wk = w[k];
      const f32x4 yk = y4[(unsigned long)k * n4 + i4];
#pragma unroll
      for (u32 c = 0u; c < 4u; c++) x0[c] = __builtin_fmaf(wk, yk[c], x0[c]);
    }
  }
  return x0;
}

KERNEL ie_inkling_gexp_iq4nl_tail_t1_nt_r10(
    const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *yd,
    u32 rows, u32 nb, const G u32 *route, u32 K, u32 qwn, u32 qsn,
    u32 astr, u32 bstr,
    const G float *v, G float *ring, const G float *res, G float *out,
    float escale, u32 pos0, G u32 *rcount) {
#pragma clang fp contract(off)
  static LDS u32 ink_r10_fm_i4t_last;
  if (rows == 0u || (rows & 3u) != 0u || nb == 0u || K == 0u ||
      !qw || !qs || !xq || !yd || !route || !v || !ring || !res || !out || !rcount)
    return;

  const u32 rb = wgid();
  const u32 rbase = rb << 5u;
  if (rbase >= rows) return;

  const u32 s = tokid();
  if (s >= K) return;

  const u32 t0 = tid();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 r_wave = rbase + (wv << 2u);
  const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
  const u32 r = r_wave + sub;

  const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[s]);
  const G u8 *w_base = rq_ptr_u((const G u8 *)(qw + (unsigned long)e * (unsigned long)qwn));
  const G u8 *q_base = rq_ptr_u((const G u8 *)(qs + (unsigned long)e * (unsigned long)qsn));
  const G u8 *xp = rq_ptr_u(xq + (unsigned long)s * (unsigned long)astr);
  const __amdgpu_buffer_rsrc_t xr = ink_r10_fm_rsrc(xp, nb * 40u);
  const u32 da_base = nb << 5u;
  G float *o = (G float *)((G u8 *)yd + (unsigned long)s * (unsigned long)bstr);

  if (r_wave < rows) {
    const u32 rr = r < rows ? r : rows - 1u;
    const G u8 *rw = w_base + (unsigned long)rr * ((unsigned long)nb * 16ul);
    const G f16 *rs_p = (const G f16 *)q_base + (unsigned long)rr * (unsigned long)nb;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const u32 nstep = (nb + 7u) >> 3u;

    u32x4 w_c = (u32x4)(0u), x0_c = (u32x4)(0u), x1_c = (u32x4)(0u);
    f16 sc_c = (f16)0.0f;
    float d_c = 0.0f;
    if (sl < nb) {
      w_c = ink_r10_fm_nt_ld4(rw + (unsigned long)sl * 16ul);
      sc_c = ink_r10_fm_nt_f16(rs_p + sl);
      x0_c = ink_r10_fm_ht_ld4(xr, sl << 5u);
      x1_c = ink_r10_fm_ht_ld4(xr, (sl << 5u) + 16u);
      d_c = ink_r10_fm_ht_f32(xr, da_base + (sl << 2u));
    }
    __builtin_amdgcn_sched_barrier(0);

#pragma unroll 1
    for (u32 step = 0u; step < nstep; step++) {
      const u32 b_c = (step << 3u) + sl;
      const u32 bn = b_c + 8u;
      u32x4 w_n = (u32x4)(0u), x0_n = (u32x4)(0u), x1_n = (u32x4)(0u);
      f16 sc_n = (f16)0.0f;
      float d_n = 0.0f;
      if (bn < nb) {
        w_n = ink_r10_fm_nt_ld4(rw + (unsigned long)bn * 16ul);
        sc_n = ink_r10_fm_nt_f16(rs_p + bn);
        x0_n = ink_r10_fm_ht_ld4(xr, bn << 5u);
        x1_n = ink_r10_fm_ht_ld4(xr, (bn << 5u) + 16u);
        d_n = ink_r10_fm_ht_f32(xr, da_base + (bn << 2u));
      }
      __builtin_amdgcn_sched_barrier(0);
      if (b_c < nb) {
        const float val = (float)sc_c * (float)ink_r10_fm_iq4_dot4(w_c, x0_c, x1_c);
        INK_R10_FM_FMA4(a0, a1, a2, a3, step & 3u, val, d_c);
      }
      __builtin_amdgcn_sched_barrier(0);
      w_c = w_n; sc_c = sc_n; x0_c = x0_n; x1_c = x1_n; d_c = d_n;
    }

    const int full_wave = (nb > 16u);
    const float acc = ink_r10_fm_oct_tree_f(full_wave ? ((a0 + a2) + (a1 + a3)) : (a0 + a1));
    if (sl == 0u && r < rows) o[r] = acc;
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (t0 == 0u) {
    ink_r10_fm_i4t_last =
        (__scoped_atomic_fetch_add(&rcount[rb], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == K - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r10_fm_i4t_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  const u32 n4 = rows >> 2u;
  if (t0 < 8u) {
    const u32 i4 = (rb << 3u) + t0;
    if (i4 < n4) {
      const u32 i0 = i4 << 2u;
      const G f32x4 *v4 = (const G f32x4 *)v + i0;
      const f32x4 vw0 = v4[0], vw1 = v4[1], vw2 = v4[2], vw3 = v4[3];
      const f32x4 sw0 = (f32x4){vw0.w, vw1.w, vw2.w, vw3.w};
      const f32x4 sw1 = (f32x4){vw0.z, vw1.z, vw2.z, vw3.z};
      const f32x4 sw2 = (f32x4){vw0.y, vw1.y, vw2.y, vw3.y};
      const f32x4 sw3 = (f32x4){vw0.x, vw1.x, vw2.x, vw3.x};

      f32x4 x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      f32x4 x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      f32x4 x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      if (pos0 > 0u) {
        const G f32x4 *rg = (const G f32x4 *)ring + i0;
        const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
        const u32 s1 = (pos0 - 1u) & 3u;
        x1 = (f32x4){ink_r10_fm_pick4(r0, s1), ink_r10_fm_pick4(r1, s1),
                     ink_r10_fm_pick4(r2, s1), ink_r10_fm_pick4(r3, s1)};
        if (pos0 >= 2u) {
          const u32 s2 = (pos0 - 2u) & 3u;
          x2 = (f32x4){ink_r10_fm_pick4(r0, s2), ink_r10_fm_pick4(r1, s2),
                       ink_r10_fm_pick4(r2, s2), ink_r10_fm_pick4(r3, s2)};
        }
        if (pos0 >= 3u) {
          const u32 s3 = (pos0 - 3u) & 3u;
          x3 = (f32x4){ink_r10_fm_pick4(r0, s3), ink_r10_fm_pick4(r1, s3),
                       ink_r10_fm_pick4(r2, s3), ink_r10_fm_pick4(r3, s3)};
        }
      }

      const f32x4 rv = ((const G f32x4 *)res)[i4];
      const f32x4 x0 = ink_r10_fm_moe_combine4((const G f32x4 *)yd, route, n4, K, i4);
      const f32x4 cv = ((x0 * sw0 + x1 * sw1) + x2 * sw2) + x3 * sw3;
      const f32x4 sc = (x0 + cv) * escale;
      ((G f32x4 *)out)[i4] = rv + sc;
      const u32 sl_r = pos0 & 3u;
      ring[(unsigned long)(i0 + 0u) * 4ul + sl_r] = x0.x;
      ring[(unsigned long)(i0 + 1u) * 4ul + sl_r] = x0.y;
      ring[(unsigned long)(i0 + 2u) * 4ul + sl_r] = x0.z;
      ring[(unsigned long)(i0 + 3u) * 4ul + sl_r] = x0.w;
    }
  }
  if (t0 == 0u) rcount[rb] = 0u;
}

/* ============================================================================
 * Kernel 6 (H35 companion): ie_inkling_gexp_iq4nl_t1_nt_r10
 * Standalone T=1 IQ4_NL drop-in for default hybrid B1 mode (replaces gexp_iq4nl_r3):
 *   - TH_LOAD_NT on IQ4_NL weights (qw, qs) + TH_LOAD_HT on exp_mid_q8 (xq, da)
 *   - Grid: ((rows + 31u) / 32u, K, 1u), <= 64 VGPR (4 WGs/CU), 0 B LDS.
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq4nl_t1_nt_r10(
    const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
    const G u32 *route, u32 qwn, u32 qsn, u32 astr, u32 bstr, u32 kind, u32 two_off) {
#pragma clang fp contract(off)
  (void)kind;
  (void)two_off;
  if (rows == 0u || nb == 0u) return;

  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 r_wave = (wgid() << 5u) + (wv << 2u);
  if (r_wave >= rows) return;

  const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
  const u32 r = r_wave + sub;
  const u32 rr = r < rows ? r : rows - 1u;
  const u32 s = tokid();
  const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[s]);

  const G u8 *w_base = rq_ptr_u((const G u8 *)(qw + (unsigned long)e * (unsigned long)qwn));
  const G u8 *q_base = rq_ptr_u((const G u8 *)(qs + (unsigned long)e * (unsigned long)qsn));
  const G u8 *xp = rq_ptr_u(xq + (unsigned long)s * (unsigned long)astr);
  const __amdgpu_buffer_rsrc_t xr = ink_r10_fm_rsrc(xp, nb * 40u);
  const u32 da_base = nb << 5u;
  G float *o = (G float *)((G u8 *)y + (unsigned long)s * (unsigned long)bstr);

  const G u8 *rw = w_base + (unsigned long)rr * ((unsigned long)nb * 16ul);
  const G f16 *rs_p = (const G f16 *)q_base + (unsigned long)rr * (unsigned long)nb;

  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  const u32 nstep = (nb + 7u) >> 3u;

  u32x4 w_c = (u32x4)(0u), x0_c = (u32x4)(0u), x1_c = (u32x4)(0u);
  f16 sc_c = (f16)0.0f;
  float d_c = 0.0f;
  if (sl < nb) {
    w_c = ink_r10_fm_nt_ld4(rw + (unsigned long)sl * 16ul);
    sc_c = ink_r10_fm_nt_f16(rs_p + sl);
    x0_c = ink_r10_fm_ht_ld4(xr, sl << 5u);
    x1_c = ink_r10_fm_ht_ld4(xr, (sl << 5u) + 16u);
    d_c = ink_r10_fm_ht_f32(xr, da_base + (sl << 2u));
  }
  __builtin_amdgcn_sched_barrier(0);

#pragma unroll 1
  for (u32 step = 0u; step < nstep; step++) {
    const u32 b_c = (step << 3u) + sl;
    const u32 bn = b_c + 8u;
    u32x4 w_n = (u32x4)(0u), x0_n = (u32x4)(0u), x1_n = (u32x4)(0u);
    f16 sc_n = (f16)0.0f;
    float d_n = 0.0f;
    if (bn < nb) {
      w_n = ink_r10_fm_nt_ld4(rw + (unsigned long)bn * 16ul);
      sc_n = ink_r10_fm_nt_f16(rs_p + bn);
      x0_n = ink_r10_fm_ht_ld4(xr, bn << 5u);
      x1_n = ink_r10_fm_ht_ld4(xr, (bn << 5u) + 16u);
      d_n = ink_r10_fm_ht_f32(xr, da_base + (bn << 2u));
    }
    __builtin_amdgcn_sched_barrier(0);
    if (b_c < nb) {
      const float val = (float)sc_c * (float)ink_r10_fm_iq4_dot4(w_c, x0_c, x1_c);
      INK_R10_FM_FMA4(a0, a1, a2, a3, step & 3u, val, d_c);
    }
    __builtin_amdgcn_sched_barrier(0);
    w_c = w_n; sc_c = sc_n; x0_c = x0_n; x1_c = x1_n; d_c = d_n;
  }

  const float acc = ink_r10_fm_oct_tree_f((a0 + a2) + (a1 + a3));
  if (sl == 0u && r < rows) o[r] = acc;
}

/* ============================================================================
 * Kernel 7 (H35): ie_inkling_q4k_head_ver_nt_r10
 * Single-pass T<=8 Q4_K Output Head + optional fused Logit-Cap:
 *   - Drop-in exact 14-arg signature of ie_inkling_gemv_q4k_head_ver_r7.
 *   - Strictly single-pass (avoids R19 ts2 2x DRAM read bug!).
 *   - TH_LOAD_NT on 260.5 MB output.weight (row_qw, row_sm, row_dd32) so head
 *     streaming does NOT evict looped blk.6..17 / SWA KV from 64 MiB MALL.
 *   - TH_LOAD_HT on reused xq (<= 23 KiB shared across all 6,282 WGs!).
 *   - Grid: ((rows + 31u) / 32u, 1u, 1u), <= 96 VGPR (4 WGs/CU), 0 B LDS.
 * ========================================================================== */

KERNEL ie_inkling_q4k_head_ver_nt_r10(
    const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y,
    u32 rows, u32 nb, const G float *bias, const G float *res,
    u32 T, u32 xs, u32 ys, u32 rs, u32 unpadded, float cap) {
#pragma clang fp contract(off)
  if (T == 0u || T > INK_R10_FM_Q4K_TMAX || rows == 0u || nb == 0u || (nb & 7u) != 0u ||
      !qw || !qs || !xq || !y)
    return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 r = (wgid() << 5u) + (w << 2u) + (l >> 3u);
  if (r >= rows) return;

  if (unpadded > 0u && r >= unpadded) {
    if (sub == 0u) {
#pragma unroll
      for (u32 j = 0u; j < INK_R10_FM_Q4K_TMAX; j++) {
        if (j < T) y[(unsigned long)j * (unsigned long)ys + (unsigned long)r] = -1e30f;
      }
    }
    return;
  }

  const u32 nsb = nb >> 3u;
  const G u32 *row_qw = qw + (unsigned long)r * (unsigned long)nb * 4ul;
  const G u16 *row_sm = qs + (unsigned long)r * (unsigned long)nb;
  const G u32 *row_dd32 = (const G u32 *)(qs + (unsigned long)rows * (unsigned long)nb +
                                          (unsigned long)r * (unsigned long)nsb * 2ul);
  const u32 sub32 = sub << 5u;
  const u32 sub4 = sub << 2u;
  const u32 x_tot = (T > 1u) ? (T * xs) : (nb * 40u);
  const __amdgpu_buffer_rsrc_t xr = ink_r10_fm_rsrc(xq, x_tot);

  v8f a0 = (v8f)(0.0f), a1 = (v8f)(0.0f), a2 = (v8f)(0.0f), a3 = (v8f)(0.0f);

#pragma clang loop unroll(disable)
  for (u32 sb = 0u; sb < nsb; sb++) {
    const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
    const u32 slot = (u32)__builtin_amdgcn_readfirstlane((int)(sb_u & 3u));
    const u32 b = (sb_u << 3u) + sub;

    /* Strictly single-read output.weight: TH_LOAD_NT (1x DRAM pass for all T<=8). */
    const u32x4 w4 = ink_r10_fm_nt_ld4(row_qw + (unsigned long)b * 4ul);
    const u32 sm = ink_r10_fm_nt_u16(row_sm + b);
    const u32 dd_raw = ink_r10_fm_nt_ld1(row_dd32 + sb_u);
    const u16 d_bits = (u16)(dd_raw & 0xFFFFu);
    const u16 dm_bits = (u16)(dd_raw >> 16u);
    const float d = (float)__builtin_bit_cast(f16, d_bits);
    const float dmin = (float)__builtin_bit_cast(f16, dm_bits);
    const float sc = d * (float)(sm & 63u);
    const float mm = dmin * (float)(sm >> 8u);

    const u32 lo0 = ie_q4k_lo(w4.x), hi0 = ie_q4k_hi(w4.x);
    const u32 lo1 = ie_q4k_lo(w4.y), hi1 = ie_q4k_hi(w4.y);
    const u32 lo2 = ie_q4k_lo(w4.z), hi2 = ie_q4k_hi(w4.z);
    const u32 lo3 = ie_q4k_lo(w4.w), hi3 = ie_q4k_hi(w4.w);

    const u32 aw_sb = sb_u << 8u;
    const u32 da_sb = (nb << 5u) + (sb_u << 5u);
    const u32 as_sb = nb * 36u + (sb_u << 5u);

#pragma unroll
    for (u32 j = 0u; j < INK_R10_FM_Q4K_TMAX; j++) {
      if (j < T) {
        const u32 joff = j * xs;
        const u32x4 act0 = ink_r10_fm_ht_ld4(xr, joff + aw_sb + sub32);
        const u32x4 act1 = ink_r10_fm_ht_ld4(xr, joff + aw_sb + sub32 + 16u);
        const float da = ink_r10_fm_ht_f32(xr, joff + da_sb + sub4);
        const int as = (int)ink_r10_fm_ht_u32(xr, joff + as_sb + sub4);
        const float xa = da * (float)as;

        u32 dq = ie_dot4_us(0u, lo0, act0.x);
        dq = ie_dot4_us(dq, hi0, act1.x);
        dq = ie_dot4_us(dq, lo1, act0.y);
        dq = ie_dot4_us(dq, hi1, act1.y);
        dq = ie_dot4_us(dq, lo2, act0.z);
        dq = ie_dot4_us(dq, hi2, act1.z);
        dq = ie_dot4_us(dq, lo3, act0.w);
        dq = ie_dot4_us(dq, hi3, act1.w);

        const float prod = sc * (float)(int)dq;
        if (slot == 0u) {
          const float a_mid = __builtin_fmaf(prod, da, a0[j]);
          a0[j] = __builtin_fmaf(-mm, xa, a_mid);
        } else if (slot == 1u) {
          const float a_mid = __builtin_fmaf(prod, da, a1[j]);
          a1[j] = __builtin_fmaf(-mm, xa, a_mid);
        } else if (slot == 2u) {
          const float a_mid = __builtin_fmaf(prod, da, a2[j]);
          a2[j] = __builtin_fmaf(-mm, xa, a_mid);
        } else {
          const float a_mid = __builtin_fmaf(prod, da, a3[j]);
          a3[j] = __builtin_fmaf(-mm, xa, a_mid);
        }
        if ((j & 1u) == 1u) __builtin_amdgcn_sched_barrier(0);
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  const int use_cap = (cap > 0.0f);
#pragma unroll
  for (u32 j = 0u; j < INK_R10_FM_Q4K_TMAX; j++) {
    if (j < T) {
      const float u0 = a0[j] + a2[j];
      const float u1 = a1[j] + a3[j];
      float s = u0 + u1;
      s = s + I2F(shl_i(F2I(s), 4));
      s = s + I2F(shl_i(F2I(s), 2));
      s = s + I2F(shl_i(F2I(s), 1));
      if (sub == 0u) {
        const G float *rp = res ? (res + (unsigned long)j * (unsigned long)rs) : (const G float *)0;
        float val = epilogue(s, r, bias, rp);
        if (use_cap) val = cap * ie_tanhf_s(val / cap);
        y[(unsigned long)j * (unsigned long)ys + (unsigned long)r] = val;
      }
    }
  }
}

/* ============================================================================
 * Kernel 8 (H35): ie_inkling_gemv_q8_ht_r10
 * Looped blk.6..17 Dense Q8_0 GEMV with TH_LOAD_HT on BOTH weights & activations
 * for 64 MiB Infinity Cache (MALL) residency across pass = 0, 1, 2!
 * Reverses ie_gemv_q8q8_tr's TH_LOAD_NT eviction on looped dense projections.
 *
 * Signature: exact 12-arg ie_gemv_q8q8_tr / ie_inkling_gemv_q8_bw_r7 contract.
 * Grid:      ((rows + 31u) / 32u, 1u, 1u) [32 rows/WG: 4 rows/wave].
 *   - T == 1:    8 lanes/row, 4-block 2x-pipelined exact r2_g10 loop (TH_LOAD_HT).
 *   - T = 2..16: 32 lanes/row, 4 rows/wave (CT=8 single-pass for T<=8) exact
 *                ie_gemv_q8q8_tr reduction order, TH_LOAD_HT on qw/qs/xq/da.
 * Parity:    100% bitwise-exact (0-bit-diff) vs ie_gemv_q8q8 / ie_gemv_q8q8_tr.
 * Resources: <= 96 VGPR (4 WGs/CU), 0 B LDS, 0/0 spills.
 * ========================================================================== */

KERNEL ie_inkling_gemv_q8_ht_r10(
    const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y,
    u32 rows, u32 nb, const G float *bias, const G float *res,
    u32 T, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  if (T == 0u || rows == 0u || nb == 0u || !qw || !qs || !xq || !y) return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 r_wave = (wgid() << 5u) + (w << 2u);
  if (r_wave >= rows) return;

  const u32 l = lane();
  const u32 row_wbytes = nb << 5u;
  const u32 row_sbytes = nb << 1u;
  const u32 r_rem = (rows - r_wave < 4u) ? (rows - r_wave) : 4u;
  const __amdgpu_buffer_rsrc_t wr = ink_r10_fm_rsrc(
      (const G u8 *)qw + (unsigned long)r_wave * (unsigned long)row_wbytes,
      r_rem * row_wbytes);
  const __amdgpu_buffer_rsrc_t sr = ink_r10_fm_rsrc(
      (const G u8 *)qs + (unsigned long)r_wave * (unsigned long)row_sbytes,
      r_rem * row_sbytes);

  if (T == 1u && nb >= 8u && (nb & 7u) == 0u) {
    /* T == 1 fast path: 4 rows/wave, 8 lanes/row, 2x-pipelined 4-block TH_LOAD_HT. */
    const u32 sub = l & 7u;
    const u32 q = l >> 3u;
    const u32 r = r_wave + q;
    if (r >= rows) return;

    const __amdgpu_buffer_rsrc_t xr = ink_r10_fm_rsrc(xq, nb * 40u);
    const u32 w_off0 = q * row_wbytes;
    const u32 s_off0 = q * row_sbytes;
    const u32 da_off0 = nb << 5u;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const u32 nb_main = nb & ~31u;

    for (u32 k0 = 0u; k0 < nb_main; k0 += 32u) {
      const u32 b0 = k0 + sub;
      const u32 b1 = b0 + 8u;
      const u32 bo0 = b0 << 5u, bo1 = b1 << 5u;
      const u32x4 w0_0 = ink_r10_fm_ht_ld4(wr, w_off0 + bo0);
      const u32x4 w0_1 = ink_r10_fm_ht_ld4(wr, w_off0 + bo0 + 16u);
      const u32x4 x0_0 = ink_r10_fm_ht_ld4(xr, bo0);
      const u32x4 x0_1 = ink_r10_fm_ht_ld4(xr, bo0 + 16u);
      const float sc0 = ink_r10_fm_ht_f16f(sr, s_off0 + (b0 << 1u)) * ink_r10_fm_ht_f32(xr, da_off0 + (b0 << 2u));

      const u32x4 w1_0 = ink_r10_fm_ht_ld4(wr, w_off0 + bo1);
      const u32x4 w1_1 = ink_r10_fm_ht_ld4(wr, w_off0 + bo1 + 16u);
      const u32x4 x1_0 = ink_r10_fm_ht_ld4(xr, bo1);
      const u32x4 x1_1 = ink_r10_fm_ht_ld4(xr, bo1 + 16u);
      const float sc1 = ink_r10_fm_ht_f16f(sr, s_off0 + (b1 << 1u)) * ink_r10_fm_ht_f32(xr, da_off0 + (b1 << 2u));

      a0 = a0 + sc0 * (float)ink_r10_fm_q8_dot8(w0_0, w0_1, x0_0, x0_1);
      a1 = a1 + sc1 * (float)ink_r10_fm_q8_dot8(w1_0, w1_1, x1_0, x1_1);
      __builtin_amdgcn_sched_barrier(0);

      const u32 b2 = b0 + 16u;
      const u32 b3 = b0 + 24u;
      const u32 bo2 = b2 << 5u, bo3 = b3 << 5u;
      const u32x4 w2_0 = ink_r10_fm_ht_ld4(wr, w_off0 + bo2);
      const u32x4 w2_1 = ink_r10_fm_ht_ld4(wr, w_off0 + bo2 + 16u);
      const u32x4 x2_0 = ink_r10_fm_ht_ld4(xr, bo2);
      const u32x4 x2_1 = ink_r10_fm_ht_ld4(xr, bo2 + 16u);
      const float sc2 = ink_r10_fm_ht_f16f(sr, s_off0 + (b2 << 1u)) * ink_r10_fm_ht_f32(xr, da_off0 + (b2 << 2u));

      const u32x4 w3_0 = ink_r10_fm_ht_ld4(wr, w_off0 + bo3);
      const u32x4 w3_1 = ink_r10_fm_ht_ld4(wr, w_off0 + bo3 + 16u);
      const u32x4 x3_0 = ink_r10_fm_ht_ld4(xr, bo3);
      const u32x4 x3_1 = ink_r10_fm_ht_ld4(xr, bo3 + 16u);
      const float sc3 = ink_r10_fm_ht_f16f(sr, s_off0 + (b3 << 1u)) * ink_r10_fm_ht_f32(xr, da_off0 + (b3 << 2u));

      a2 = a2 + sc2 * (float)ink_r10_fm_q8_dot8(w2_0, w2_1, x2_0, x2_1);
      a3 = a3 + sc3 * (float)ink_r10_fm_q8_dot8(w3_0, w3_1, x3_0, x3_1);
      __builtin_amdgcn_sched_barrier(0);
    }

    const u32 rem = nb - nb_main;
    if (rem > 0u) {
      const u32 b0 = nb_main + sub;
      if (b0 < nb) {
        const u32 bo0 = b0 << 5u;
        const int d0 = ink_r10_fm_q8_dot8(
            ink_r10_fm_ht_ld4(wr, w_off0 + bo0), ink_r10_fm_ht_ld4(wr, w_off0 + bo0 + 16u),
            ink_r10_fm_ht_ld4(xr, bo0), ink_r10_fm_ht_ld4(xr, bo0 + 16u));
        const float sc0 = ink_r10_fm_ht_f16f(sr, s_off0 + (b0 << 1u)) * ink_r10_fm_ht_f32(xr, da_off0 + (b0 << 2u));
        a0 = a0 + sc0 * (float)d0;
      }
      if (rem > 8u) {
        const u32 b1 = nb_main + 8u + sub;
        if (b1 < nb) {
          const u32 bo1 = b1 << 5u;
          const int d1 = ink_r10_fm_q8_dot8(
              ink_r10_fm_ht_ld4(wr, w_off0 + bo1), ink_r10_fm_ht_ld4(wr, w_off0 + bo1 + 16u),
              ink_r10_fm_ht_ld4(xr, bo1), ink_r10_fm_ht_ld4(xr, bo1 + 16u));
          const float sc1 = ink_r10_fm_ht_f16f(sr, s_off0 + (b1 << 1u)) * ink_r10_fm_ht_f32(xr, da_off0 + (b1 << 2u));
          a1 = a1 + sc1 * (float)d1;
        }
      }
      if (rem > 16u) {
        const u32 b2 = nb_main + 16u + sub;
        if (b2 < nb) {
          const u32 bo2 = b2 << 5u;
          const int d2 = ink_r10_fm_q8_dot8(
              ink_r10_fm_ht_ld4(wr, w_off0 + bo2), ink_r10_fm_ht_ld4(wr, w_off0 + bo2 + 16u),
              ink_r10_fm_ht_ld4(xr, bo2), ink_r10_fm_ht_ld4(xr, bo2 + 16u));
          const float sc2 = ink_r10_fm_ht_f16f(sr, s_off0 + (b2 << 1u)) * ink_r10_fm_ht_f32(xr, da_off0 + (b2 << 2u));
          a2 = a2 + sc2 * (float)d2;
        }
      }
    }

    const float s = ink_r10_fm_oct_tree_f((a0 + a2) + (a1 + a3));
    if (sub == 0u) y[r] = epilogue(s, r, bias, res);
    return;
  }

  /* Verify T = 2..16 (or nb % 8 != 0): exact 32-lane/row, 4 rows/wave, CT=8
   * single-pass for T<=8, TH_LOAD_HT on BOTH looped dense weights and xq!
   * Processing 2 rows at a time (pass p=0: rows 0..1, pass p=1: rows 2..3)
   * keeps peak VGPRs <= 80 (4 WGs/CU!) while reusing L0/L1-hot xq! */
  const u32 x_tot = (T > 1u) ? (T * xs) : (nb * 40u);
  const __amdgpu_buffer_rsrc_t xr = ink_r10_fm_rsrc(xq, x_tot);
  const u32 da_off0 = nb << 5u;

#pragma clang loop unroll(disable)
  for (u32 q0 = 0u; q0 < 4u; q0 += 2u) {
    if (r_wave + q0 >= rows) break;
    const u32 q1 = (r_wave + q0 + 1u < rows) ? (q0 + 1u) : q0;
    const u32 w0_off = q0 * row_wbytes;
    const u32 w1_off = q1 * row_wbytes;
    const u32 s0_off = q0 * row_sbytes;
    const u32 s1_off = q1 * row_sbytes;

    for (u32 t0 = 0u; t0 < T; t0 += INK_R10_FM_Q8_CT) {
      const u32 tc = (T - t0 < INK_R10_FM_Q8_CT) ? (T - t0) : INK_R10_FM_Q8_CT;
      v8f acc0 = (v8f)(0.0f);
      v8f acc1 = (v8f)(0.0f);

#pragma clang loop unroll(disable)
      for (u32 b = l; b < nb; b += 32u) {
        const u32 bo = b << 5u;
        const u32x4 w0a = ink_r10_fm_ht_ld4(wr, w0_off + bo);
        const u32x4 w0b = ink_r10_fm_ht_ld4(wr, w0_off + bo + 16u);
        const float sw0 = ink_r10_fm_ht_f16f(sr, s0_off + (b << 1u));

        const u32x4 w1a = ink_r10_fm_ht_ld4(wr, w1_off + bo);
        const u32x4 w1b = ink_r10_fm_ht_ld4(wr, w1_off + bo + 16u);
        const float sw1 = ink_r10_fm_ht_f16f(sr, s1_off + (b << 1u));
        __builtin_amdgcn_sched_barrier(0);

#pragma unroll
        for (u32 j = 0u; j < INK_R10_FM_Q8_CT; j++) {
          if (j < tc) {
            const u32 joff = (t0 + j) * xs;
            const u32x4 x0 = ink_r10_fm_ht_ld4(xr, joff + bo);
            const u32x4 x1 = ink_r10_fm_ht_ld4(xr, joff + bo + 16u);
            const float da = ink_r10_fm_ht_f32(xr, joff + da_off0 + (b << 2u));

            const int d0 = ink_r10_fm_q8_dot8(w0a, w0b, x0, x1);
            acc0[j] = acc0[j] + (sw0 * da) * (float)d0;
            const int d1 = ink_r10_fm_q8_dot8(w1a, w1b, x0, x1);
            acc1[j] = acc1[j] + (sw1 * da) * (float)d1;
            if ((j & 1u) == 1u) __builtin_amdgcn_sched_barrier(0);
          }
        }
        __builtin_amdgcn_sched_barrier(0);
      }

      const u32 r0 = r_wave + q0;
      const u32 r1 = r_wave + q0 + 1u;
#pragma unroll
      for (u32 j = 0u; j < INK_R10_FM_Q8_CT; j++) {
        if (j < tc) {
          const float v0 = wave_tree_f(acc0[j]);
          const float v1 = wave_tree_f(acc1[j]);
          if (l == 0u) {
            const G float *rp = res ? (res + (unsigned long)(t0 + j) * (unsigned long)rs) : (const G float *)0;
            G float *yp = y + (unsigned long)(t0 + j) * (unsigned long)ys;
            yp[r0] = epilogue(v0, r0, bias, rp);
            if (r1 < rows) yp[r1] = epilogue(v1, r1, bias, rp);
          }
        }
      }
    }
  }
}

#undef INK_R10_FM_FMA4
#undef INK_R10_FM_ADD4
#undef INK_R10_FM_SB_BYTES
#undef INK_R10_FM_LDS_MAXNB
#undef INK_R10_FM_Q8_CT
#undef INK_R10_FM_Q4K_TMAX
#undef INK_R10_FM_TH_HT
#undef INK_R10_FM_TH_NT
#undef INK_R10_FM_TH_RT
#undef INK_R10_FM_RSRC_W3

#endif /* !IE_MEGA */
