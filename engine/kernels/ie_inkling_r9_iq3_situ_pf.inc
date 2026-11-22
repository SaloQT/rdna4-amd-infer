/* ============================================================================
 * Inkling-2 Round 9 (Wave 8 / #1 Prefill Kernel Upgrade):
 *   3-WGs/CU Zero-Stall-Waitcnt, 3-K-Block Weight-Overlapped, Branch-Free
 *   4-Bucket (16/32/48/64) Fused Prefill IQ3_XXS Gate/Up WMMA + Exact SITU +
 *   Q8_0 for RX 9070 XT (gfx1201, wave32, NT=256).
 *
 * Target: MoE ffn_gate_exps + ffn_up_exps MAT_IQ3XXS + SITU + Q8_0 (~40% of
 *         total 4K/16K prefill time!). Drop-in upgrade for bit-70
 *         ie_inkling_gemm_rq_exp_iq3xxs_situ_r7 (same 14-arg signature,
 *         consumes bit-69 flat largest-first worklist b->ms_packs/ms_npacks).
 *
 * Root-Cause ISA Upgrades over Wave 3 promoted iq3xxs_situ_r7 (120 VGPR,
 * 6,811 insns, Loop 1 len=2,265, 38 cbranch, 163 wds, 77 mul_s, 64 fma_s):
 *   1. Killed hb1 s_wait_loadcnt 0x0 stall on rx0/rx1/rsx (50% of all K-chunks!):
 *      In r7, hb1 issued conditional rx0/rx1/rsx BEFORE hb1 k=0 decoded hb1
 *      weights, forcing s_wait_loadcnt 0x0 right after rx0/rx1/rsx (killing 100%
 *      of hb1 activation prefetch!). r9 decodes k=0 FIRST (draining only the
 *      already-in-flight weight block), THEN issues unconditional rx0/rx1
 *      (overlapped across k=0, k=1, k=2 = 12 WMMAs + 96 FMAs), stages rx0/rx1
 *      after k=2, and defers rsx (waves 0..1) to after k=2 (hidden across k=3,
 *      freeing 4 peak VGPRs during k=0..2!).
 *   2. 3x Deeper IQ3_XXS Weight-Load Latency Hiding (0 extra VGPRs!):
 *      r7 waited until after k=3 decode to load all 3 b128s of the next half-SB
 *      (only 1 K-block hidden). Since cur_q[0..3] are 100% dead right after k=1
 *      decode, r9 reloads next-half cur_q[0..3] right after k=1 (3 full K-blocks
 *      of DRAM/L2 hiding!) and cur_q[4..7] + aux after k=3 (cur_q[4..7] also
 *      hidden across 3 full K-blocks before next-half k=2!).
 *   3. Eliminated ALL 38 inner-loop branches/superblock (if(has_j3)/if(has_j1)
 *      + fake if(k < nk) since (nb & 7u) == 0u is guarded!):
 *      4 straight-line branch-free loops keyed by worklist cls = 3/2/1/0
 *      (WJ = 4/3/2/1), shrinking Loop 1 AND total binary size below r7!
 *   4. Restored 2-ops/cycle v_dual_mul_f32 + v_dual_fmac_f32 across all 8
 *      K-blocks: replaced "+v"(acc[j]) (inline-asm redef blocking v_fmac_f32
 *      shrinking into 64 scalar v_fma_f32 + 77 scalar v_mul_f32) with read-only
 *      "v"(acc[j]) barriers + collapsed 4 x_col VGPRs into 1 lx_base0!
 *   5. Halved scale s_wait_dscnt inside apply_pair_j/apply_one_j by issuing
 *      both f32x4 scale loads (dlo + dhi) together before bias subtraction.
 *
 * Exported Kernel, Grid, Guard, and Parity (drop-in replacement for b70):
 *   KERNEL ie_inkling_gemm_rq_exp_iq3xxs_situ_r9(
 *       const G u8 *qw, const G u8 *X, G u8 *Qout, u32 eff, u32 nb,
 *       const G u32 *list, const G u32 *tiles, const G u32 *n_tiles,
 *       u32 q_bytes, u32 xs, u32 qstr, u32 xdiv, float beta, float lbeta)
 *   - Grid:   ((eff + 63u) / 64u, max_tiles, 1u), NT = 256
 *             where max_tiles = min(T*K, n_exp + (T*K)/64u)
 *   - Guard:  identical to ie_inkling_gemm_rq_exp_iq3xxs_situ_r7
 *             (!mk.rec && T >= 17u && eff > 0u && (eff & 31u) == 0u &&
 *              nb >= 8u && (nb & 7u) == 0u && n_exp <= 256u && T * K <= 8192u)
 *   - Parity: 100% bitwise-exact (0-bit-diff in Qout codes, scales, asums) vs
 *             ie_inkling_gemm_rq_exp_iq3xxs_situ_r6 / r7!
 *   - All local symbols/LDS/macros prefixed ink_r9_iq3_; Clang 21 scalar
 *     __builtin_bit_cast safe, <= 128 VGPR + 19,712 B LDS (3 WGs/CU), 0/0 spills.
 * ========================================================================== */

#ifndef IE_MEGA

struct __attribute__((packed, aligned(4))) ink_r9_iq3_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r9_iq3_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r9_iq3_u32x4p *)p)->v;
}

typedef struct {
  v2i w0, w1;
  float s0;
} ink_r9_iq3_dc_t;

typedef struct {
  union {
    struct {
      u8 x[2][64u * 128u] __attribute__((aligned(16)));
      float sx[2][4u * 64u] __attribute__((aligned(32)));
    } k;
    float ep[2][32u][66u] __attribute__((aligned(32)));
  } u;
  u32 lut[256] __attribute__((aligned(16)));
  u32 pid[64] __attribute__((aligned(16)));
} ink_r9_iq3_lds_t;

_Static_assert(sizeof(ink_r9_iq3_lds_t) == 19712u, "ink_r9_iq3_lds_t must be 19,712 B (<= 21.8 KiB = 3 WGs/CU)");

static inline __attribute__((always_inline)) u32 ink_r9_iq3_perm_al(const u32 lo, const u32 hi, const u32 psel) {
  return __builtin_amdgcn_perm(hi, lo, psel);
}

static inline __attribute__((always_inline)) u32 ink_r9_iq3_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

/* Single-mul IQ3_XXS 7-bit+parity sign decode (bitwise-identical to rq_ksign7 + ie_rq_ssign). */
static inline __attribute__((always_inline)) v2i ink_r9_iq3_ksw(const u32 g0, const u32 g1, const u32 s7) {
  const u32 p_ = s7 * 0x00204081u;
  const u32 t0 = p_ & 0x01010101u;
  const u32 t1 = ((p_ >> 4u) | ((u32)__builtin_popcount(s7) << 24u)) & 0x01010101u;
  const u32 m0 = __builtin_amdgcn_perm(0u, 0u, t0 | 0x0C0C0C0Cu);
  const u32 m1 = __builtin_amdgcn_perm(0u, 0u, t1 | 0x0C0C0C0Cu);
  const v2i v = {(int)((g0 ^ m0) + t0), (int)((g1 ^ m1) + t1)};
  return v;
}

/* Prologue full hb0 load. */
static inline __attribute__((always_inline)) void ink_r9_iq3_load_hb0_all(
    const G u8 *p, const u32 h, u32 qw[8], u32 aw[5]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v0 = ink_r9_iq3_ld4(base + h);
  const u32x4 v1 = ink_r9_iq3_ld4(base + h + 4u);
  qw[0] = v0.x; qw[1] = v0.y; qw[2] = v0.z; qw[3] = v0.w;
  qw[4] = v1.x; qw[5] = v1.y; qw[6] = v1.z; qw[7] = v1.w;
  const u32x4 a0 = ink_r9_iq3_ld4(base + 16u);
  aw[0] = a0.x; aw[1] = a0.y; aw[2] = a0.z; aw[3] = a0.w;
  aw[4] = base[20];
}

/* Split hb1 early load (cur_q[0..3] reloaded right after hb0 k=1 decode -> 3 K-blocks hidden!). */
static inline __attribute__((always_inline)) void ink_r9_iq3_load_hb1_q0(
    const G u8 *p, const u32 h, u32 qw[8]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v0 = ink_r9_iq3_ld4(base + 8u + h);
  qw[0] = v0.x; qw[1] = v0.y; qw[2] = v0.z; qw[3] = v0.w;
}

/* Split hb1 tail load (cur_q[4..7] + cur_a[1..4] after hb0 k=3 decode). */
static inline __attribute__((always_inline)) void ink_r9_iq3_load_hb1_q1a(
    const G u8 *p, const u32 h, u32 qw[8], u32 at[4]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v1 = ink_r9_iq3_ld4(base + 12u + h);
  qw[4] = v1.x; qw[5] = v1.y; qw[6] = v1.z; qw[7] = v1.w;
  const u32x4 a0 = ink_r9_iq3_ld4(base + 21u);
  at[0] = a0.x; at[1] = a0.y; at[2] = a0.z; at[3] = a0.w;
}

/* Split next-SB hb0 early load (cur_q[0..3] right after hb1 k=1 decode -> 3 K-blocks hidden!). */
static inline __attribute__((always_inline)) void ink_r9_iq3_load_hb0_q0(
    const G u8 *p, const u32 h, u32 qw[8]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v0 = ink_r9_iq3_ld4(base + h);
  qw[0] = v0.x; qw[1] = v0.y; qw[2] = v0.z; qw[3] = v0.w;
}

/* Split next-SB hb0 tail load (cur_q[4..7] + cur_a[0..4] after hb1 k=3 decode). */
static inline __attribute__((always_inline)) void ink_r9_iq3_load_hb0_q1a(
    const G u8 *p, const u32 h, u32 qw[8], u32 aw[5]) {
  const G u32 *base = rq_ap(p, 0u);
  const u32x4 v1 = ink_r9_iq3_ld4(base + h + 4u);
  qw[4] = v1.x; qw[5] = v1.y; qw[6] = v1.z; qw[7] = v1.w;
  const u32x4 a0 = ink_r9_iq3_ld4(base + 16u);
  aw[0] = a0.x; aw[1] = a0.y; aw[2] = a0.z; aw[3] = a0.w;
  aw[4] = base[20];
}

static inline __attribute__((always_inline)) void ink_r9_iq3_decode_one(
    const u32 q0, const u32 q1, const u32 a0, const u32 a1,
    const u32 psel, const float d2, const LDSP u32 *lut, const u32 h, ink_r9_iq3_dc_t *dc) {
  const u32 auxv = ink_r9_iq3_perm_al(a0, a1, psel);
  const u32 qp = ink_r9_iq3_perm_al(q0, q1, psel);
  const u32 g00 = lut[qp & 255u];
  const u32 g01 = lut[(qp >> 8u) & 255u];
  const u32 g10 = lut[(qp >> 16u) & 255u];
  const u32 g11 = lut[qp >> 24u];
  __builtin_amdgcn_sched_barrier(0);
  const u32 s_sh = h ? 14u : 0u;
  dc->s0 = d2 * (0.5f + (float)(auxv >> 28u));
  dc->w0 = ink_r9_iq3_ksw(g00, g01, (auxv >> s_sh) & 127u);
  dc->w1 = ink_r9_iq3_ksw(g10, g11, (auxv >> (s_sh + 7u)) & 127u);
}

/* Exact ((float)r * s0) * dx update for one 16-token tile (read-only "v"(acc) preserves v_dual_fmac_f32!). */
static inline __attribute__((always_inline)) void ink_r9_iq3_accum_tile(
    const v8i r, const float s0, const LDSP float *sx0, f8 *acc_j) {
  const f32x4 dlo = *(const LDSP f32x4 *)(sx0);
  const f32x4 dhi = *(const LDSP f32x4 *)(sx0 + 4u);
  const f8 rf = (__builtin_bit_cast(f8, r) - 12582912.0f) * s0;
  __builtin_amdgcn_sched_barrier(0);
  const f8 dx = {dlo[0], dlo[1], dlo[2], dlo[3], dhi[0], dhi[1], dhi[2], dhi[3]};
  *acc_j += rf * dx;
  __asm__ volatile("" : : "v"(*acc_j) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r9_iq3_apply_one_j(
    const ink_r9_iq3_dc_t dck, const LDSP u8 *lx_k, const LDSP float *lsx_k,
    const u32 j0, f8 *acc) {
  const v8i zm = (v8i)(0x4B400000);
  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + j0 * 2048u);
  __builtin_amdgcn_sched_barrier(0);

  const v2i a0 = {(int)b0.x, (int)b0.y};
  const v2i a1 = {(int)b0.z, (int)b0.w};
  v8i r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, dck.w0, zm, 0);
  r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, dck.w1, r0, 0);
  __builtin_amdgcn_sched_barrier(0);

  ink_r9_iq3_accum_tile(r0, dck.s0, lsx_k + 16u * j0, &acc[j0]);
}

static inline __attribute__((always_inline)) void ink_r9_iq3_apply_pair_j(
    const ink_r9_iq3_dc_t dck, const LDSP u8 *lx_k, const LDSP float *lsx_k,
    const u32 j0, f8 *acc) {
  const v8i zm = (v8i)(0x4B400000);
  const u32 j_off = j0 * 2048u;
  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + j_off);
  const u32x4 b1 = *(const LDSP u32x4 *)(lx_k + j_off + 2048u);
  __builtin_amdgcn_sched_barrier(0);

  const v2i a0_0 = {(int)b0.x, (int)b0.y};
  const v2i a0_1 = {(int)b0.z, (int)b0.w};
  const v2i a1_0 = {(int)b1.x, (int)b1.y};
  const v2i a1_1 = {(int)b1.z, (int)b1.w};
  v8i r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_0, 1, dck.w0, zm, 0);
  v8i r1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_0, 1, dck.w0, zm, 0);
  r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0_1, 1, dck.w1, r0, 0);
  r1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1_1, 1, dck.w1, r1, 0);
  __builtin_amdgcn_sched_barrier(0);

  const LDSP float *sx0 = lsx_k + 16u * j0;
  ink_r9_iq3_accum_tile(r0, dck.s0, sx0, &acc[j0]);
  ink_r9_iq3_accum_tile(r1, dck.s0, sx0 + 16u, &acc[j0 + 1u]);
}

static inline __attribute__((always_inline)) void ink_r9_iq3_apply_wj4(
    const ink_r9_iq3_dc_t dck, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 k, const u32 h, const u32 lx_base0, f8 acc[4]) {
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * 64u + 8u * h;
  ink_r9_iq3_apply_pair_j(dck, lx_k, lsx_k, 0u, acc);
  ink_r9_iq3_apply_pair_j(dck, lx_k, lsx_k, 2u, acc);
}

static inline __attribute__((always_inline)) void ink_r9_iq3_apply_wj3(
    const ink_r9_iq3_dc_t dck, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 k, const u32 h, const u32 lx_base0, f8 acc[3]) {
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * 64u + 8u * h;
  ink_r9_iq3_apply_pair_j(dck, lx_k, lsx_k, 0u, acc);
  ink_r9_iq3_apply_one_j(dck, lx_k, lsx_k, 2u, acc);
}

static inline __attribute__((always_inline)) void ink_r9_iq3_apply_wj2(
    const ink_r9_iq3_dc_t dck, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 k, const u32 h, const u32 lx_base0, f8 acc[2]) {
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * 32u + 8u * h;
  ink_r9_iq3_apply_pair_j(dck, lx_k, lsx_k, 0u, acc);
}

static inline __attribute__((always_inline)) void ink_r9_iq3_apply_wj1(
    const ink_r9_iq3_dc_t dck, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 k, const u32 h, const u32 lx_base0, f8 *acc0) {
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * 32u + 8u * h;
  ink_r9_iq3_apply_one_j(dck, lx_k, lsx_k, 0u, acc0);
}

static inline __attribute__((always_inline)) void ink_r9_iq3_stage_sx(
    LDSP float *lsx, const u32 stride, const u32 t0, const int is_s, const f32x4 rsx) {
  if (is_s) {
    u32 ti = t0;
    __asm__ volatile("" : "+v"(ti));
    lsx[0u * stride + ti] = rsx.x;
    lsx[1u * stride + ti] = rsx.y;
    lsx[2u * stride + ti] = rsx.z;
    lsx[3u * stride + ti] = rsx.w;
  }
}

/* Exact SITU (#pragma clang fp contract(off)) + SoA Q8_0 (quant_wave) epilogue strip. */
static inline __attribute__((always_inline)) void ink_r9_iq3_epilogue_strip(
    LDSP float *ep, const LDSP u32 *lpid, G u8 *Qout,
    const u32 is_up, const u32 r_loc, const u32 h, const u32 w, const u32 l,
    const u32 strip_base, const u32 nj_strip, const u32 blk0, const u32 nblk,
    const u32 nb_out, const u32 qstr, const float beta, const float lbeta,
    const u32 j_base, const f8 *acc) {
#pragma clang fp contract(off)
  LDSP float *ep_wr = ep + (is_up * 32u + 8u * h) * 66u + r_loc;
#pragma unroll
  for (u32 j_loc = 0u; j_loc < 2u; j_loc++) {
    if (16u * j_loc < nj_strip) {
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        ep_wr[(16u * j_loc + v) * 66u] = acc[j_base + j_loc][v];
      }
    }
  }
  lbarrier();

  const int use_beta = beta > 0.0f;
  const int use_lbeta = lbeta > 0.0f;
  for (u32 tk = w; tk < nj_strip; tk += 8u) {
    const u32 pid = (u32)__builtin_amdgcn_readfirstlane((int)lpid[strip_base + tk]);
    G u8 *qp = Qout + (unsigned long)pid * qstr;
    const LDSP float *ep_g = ep + tk * 66u + l;
#pragma unroll
    for (u32 b = 0u; b < 2u; b++) {
      if (b < nblk) {
        const float g = ep_g[32u * b];
        const float u = ep_g[2112u + 32u * b];
        const float gs = use_beta ? beta * ie_tanhf_s(g / beta) : g;
        const float us = use_lbeta ? lbeta * ie_tanhf_s(u / lbeta) : u;
        const float val = gs * sigmoidf(g) * us;
        quant_wave(val, blk0 + b, qp, nb_out);
      }
    }
  }
}

KERNEL ie_inkling_gemm_rq_exp_iq3xxs_situ_r9(const G u8 *qw, const G u8 *X, G u8 *Qout, u32 eff, u32 nb,
                                             const G u32 *list, const G u32 *tiles, const G u32 *n_tiles,
                                             u32 q_bytes, u32 xs, u32 qstr, u32 xdiv, float beta, float lbeta) {
  static LDS ink_r9_iq3_lds_t ink_r9_iq3_lds;
  if (!n_tiles || !tiles || !list || !qw || !X || !Qout) return;

  const u32 ti = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  const u32 nt_act = (u32)__builtin_amdgcn_readfirstlane((int)(*n_tiles));
  const u32 gbase = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() * 64u));
  if (ti >= nt_act || gbase >= eff || eff == 0u || (eff & 31u) != 0u || nb < 8u || (nb & 7u) != 0u || xdiv == 0u)
    return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)tiles[ti]);
  const u32 e = desc & 0xFFu;
  const u32 pb = (desc >> 8u) & 0x1FFFu;
  const u32 nj = (desc >> 21u) & 0x7Fu;
  if (nj == 0u || nj > 64u) return;

  const u32 cls = (nj - 1u) >> 4u; /* 3: 49..64, 2: 33..48, 1: 17..32, 0: 1..16 (SGPR-uniform!) */
  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;

  const u32 is_up = w >> 2u;
  const u32 r_loc = 16u * (w & 3u) + r16;
  const u32 ch = gbase + r_loc;
  const u32 ch_c = ch < eff ? ch : eff - 1u;
  const u32 w_row = is_up ? (eff + ch_c) : ch_c;

  const u32 mb = nb >> 3u;
  const u32 nb_out = eff >> 5u;
  const u32 blk0 = gbase >> 5u;
  const u32 nblk = (eff - gbase >= 64u) ? 2u : 1u;

  const G u8 *ebase = rq_ptr_u(qw + (unsigned long)e * q_bytes);
  const G u8 *X_u = rq_ptr_u(X);
  const G u8 *X_s = rq_ptr_u(X + 32ul * nb);
  const u32 row_off = w_row * mb * 98u;

  /* Stage 1 KiB IQ3_XXS grid LUT (synchronized by tile pid lbarrier below). */
  ink_r9_iq3_lds.lut[t0] = ie_rq_iq3xxs_grid[t0];

  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  const u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);
  const u32 lx_base0 = r16 * 128u + ((h ^ (r16 & 7u)) << 4u);

  f8 acc[4];
#pragma unroll
  for (u32 j = 0u; j < 4u; j++) acc[j] = (f8)(0.0f);

  if (cls == 3u) {
    /* ================ Bucket 3: 49..64 pairs (WJ = 4, 0 K-loop branches) ================ */
    const int is_s = (w < 2u);
    if (is_s) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r9_iq3_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ink_r9_iq3_div_x(ink_r9_iq3_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
    const u32 xoff1 = ink_r9_iq3_div_x(ink_r9_iq3_lds.pid[tk_ld + 32u], xdiv) * xs + c8_ld * 16u;
    const u32 sxoff = is_s ? ink_r9_iq3_div_x(ink_r9_iq3_lds.pid[t0 & 63u], xdiv) * xs : 0u;

    u32 cur_psel = (((unsigned long)ebase + (unsigned long)row_off) & 2ul) ? 0x07060504u : 0x05040302u;
    u32 cur_q[8], cur_a[5];
    ink_r9_iq3_load_hb0_all(ebase + row_off, h, cur_q, cur_a);
    {
      const u32x4 rx0_i = ink_r9_iq3_ld4(X_u + xoff0);
      const u32x4 rx1_i = ink_r9_iq3_ld4(X_u + xoff1);
      const f32x4 rsx_i = is_s ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
      *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[0] + st_x) = rx0_i;
      *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[0] + st_x + 4096u) = rx1_i;
      ink_r9_iq3_stage_sx(ink_r9_iq3_lds.u.k.sx[0], 64u, t0, is_s, rsx_i);
    }
    lbarrier();

    u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
    for (u32 sb = 0u; ; sb++) {
      const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
      const u32 kb1 = 8u * sb_u + 4u;
      const G u8 *sb_p = rq_ptr_u(ebase + (unsigned long)sb_u * 98ul);

      /* Decode hb0 d2 & k=0 BEFORE issuing hb1 rx0/rx1 so k=0 never drains rx0/rx1! */
      const float d2_0 = ie_rq_h2f((cur_q[0] >> ((cur_psel & 4u) << 2u)) & 0xFFFFu) * 0.5f;
      const float d2_x = I2F(xrow_i(F2I(d2_0)));
      const float d2 = h ? d2_x : d2_0;
      ink_r9_iq3_dc_t xdc;
      ink_r9_iq3_decode_one(cur_q[0], cur_q[1], cur_a[0], cur_a[1], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      {
        const G u8 *X_hb1 = rq_ptr_u(X_u + (unsigned long)kb1 * 32ul);
        rx0 = ink_r9_iq3_ld4(X_hb1 + xoff0);
        rx1 = ink_r9_iq3_ld4(X_hb1 + xoff1);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj4(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 0u, h, lx_base0, acc);

      /* hb0 k=1: immediately reload hb1 cur_q[0..3] after k=1 decode (3 K-blocks hidden!). */
      ink_r9_iq3_decode_one(cur_q[2], cur_q[3], cur_a[1], cur_a[2], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      ink_r9_iq3_load_hb1_q0(sb_p + row_off, h, cur_q);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj4(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 1u, h, lx_base0, acc);

      /* hb0 k=2 */
      ink_r9_iq3_decode_one(cur_q[4], cur_q[5], cur_a[2], cur_a[3], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj4(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 2u, h, lx_base0, acc);

      /* Stage hb1 rx0/rx1 after k=2 (3 K-blocks hidden!) and prefetch rsx (hidden across k=3!). */
      *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[1] + st_x) = rx0;
      *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[1] + st_x + 4096u) = rx1;
      f32x4 rsx = (f32x4)(0.0f);
      if (is_s) {
        const G u8 *Xs_hb1 = rq_ptr_u(X_s + (unsigned long)kb1 * 4ul);
        rsx = rq_load_f32x4((const G float *)(Xs_hb1 + sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);

      /* hb0 k=3 */
      ink_r9_iq3_decode_one(cur_q[6], cur_q[7], cur_a[3], cur_a[4], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj4(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 3u, h, lx_base0, acc);

      /* Stage hb1 rsx + reload hb1 cur_q[4..7] & cur_a[1..4], then sync! */
      ink_r9_iq3_stage_sx(ink_r9_iq3_lds.u.k.sx[1], 64u, t0, is_s, rsx);
      cur_a[0] = cur_a[4];
      ink_r9_iq3_load_hb1_q1a(sb_p + row_off, h, cur_q, cur_a + 1u);
      lbarrier();

      /* -------- hb1 (k = 0..3) -------- */
      const int more = (sb_u + 1u < mb);
      const u32 kbn = 8u * (sb_u + 1u);
      const G u8 *nxt_p = rq_ptr_u(ebase + (unsigned long)(sb_u + 1u) * 98ul);

      /* Decode hb1 k=0 FIRST (drains only hb1 weights) BEFORE issuing next-SB rx0/rx1! */
      ink_r9_iq3_decode_one(cur_q[0], cur_q[1], cur_a[0], cur_a[1], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      if (more) {
        const G u8 *X_nxt = rq_ptr_u(X_u + (unsigned long)kbn * 32ul);
        rx0 = ink_r9_iq3_ld4(X_nxt + xoff0);
        rx1 = ink_r9_iq3_ld4(X_nxt + xoff1);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj4(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 0u, h, lx_base0, acc);

      /* hb1 k=1: immediately reload next-SB cur_q[0..3] after k=1 decode (3 K-blocks hidden!). */
      ink_r9_iq3_decode_one(cur_q[2], cur_q[3], cur_a[1], cur_a[2], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      if (more) ink_r9_iq3_load_hb0_q0(nxt_p + row_off, h, cur_q);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj4(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 1u, h, lx_base0, acc);

      /* hb1 k=2 */
      ink_r9_iq3_decode_one(cur_q[4], cur_q[5], cur_a[2], cur_a[3], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj4(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 2u, h, lx_base0, acc);

      f32x4 rsx1 = (f32x4)(0.0f);
      if (more) {
        *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[0] + st_x) = rx0;
        *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[0] + st_x + 4096u) = rx1;
        if (is_s) {
          const G u8 *Xs_nxt = rq_ptr_u(X_s + (unsigned long)kbn * 4ul);
          rsx1 = rq_load_f32x4((const G float *)(Xs_nxt + sxoff));
        }
      }
      __builtin_amdgcn_sched_barrier(0);

      /* hb1 k=3 */
      ink_r9_iq3_decode_one(cur_q[6], cur_q[7], cur_a[3], cur_a[4], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj4(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 3u, h, lx_base0, acc);

      if (!more) break;
      ink_r9_iq3_stage_sx(ink_r9_iq3_lds.u.k.sx[0], 64u, t0, is_s, rsx1);
      ink_r9_iq3_load_hb0_q1a(nxt_p + row_off, h, cur_q, cur_a);
      cur_psel ^= 0x02020606u;
      lbarrier();
    }
  } else if (cls == 2u) {
    /* ================ Bucket 2: 33..48 pairs (WJ = 3, -25% X & WMMA, 0 branches) ================ */
    const int ld_hi = (tk_ld < 16u);
    const int is_s = (t0 < 48u);
    if (is_s) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r9_iq3_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ink_r9_iq3_div_x(ink_r9_iq3_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
    const u32 xoff1 = ld_hi ? (ink_r9_iq3_div_x(ink_r9_iq3_lds.pid[tk_ld + 32u], xdiv) * xs + c8_ld * 16u) : 0u;
    const u32 sxoff = is_s ? ink_r9_iq3_div_x(ink_r9_iq3_lds.pid[t0], xdiv) * xs : 0u;

    u32 cur_psel = (((unsigned long)ebase + (unsigned long)row_off) & 2ul) ? 0x07060504u : 0x05040302u;
    u32 cur_q[8], cur_a[5];
    ink_r9_iq3_load_hb0_all(ebase + row_off, h, cur_q, cur_a);
    {
      const u32x4 rx0_i = ink_r9_iq3_ld4(X_u + xoff0);
      const u32x4 rx1_i = ld_hi ? ink_r9_iq3_ld4(X_u + xoff1) : (u32x4)(0u);
      const f32x4 rsx_i = is_s ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
      *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[0] + st_x) = rx0_i;
      if (ld_hi) *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[0] + st_x + 4096u) = rx1_i;
      ink_r9_iq3_stage_sx(ink_r9_iq3_lds.u.k.sx[0], 64u, t0, is_s, rsx_i);
    }
    lbarrier();

    u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
    for (u32 sb = 0u; ; sb++) {
      const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
      const u32 kb1 = 8u * sb_u + 4u;
      const G u8 *sb_p = rq_ptr_u(ebase + (unsigned long)sb_u * 98ul);

      const float d2_0 = ie_rq_h2f((cur_q[0] >> ((cur_psel & 4u) << 2u)) & 0xFFFFu) * 0.5f;
      const float d2_x = I2F(xrow_i(F2I(d2_0)));
      const float d2 = h ? d2_x : d2_0;
      ink_r9_iq3_dc_t xdc;
      ink_r9_iq3_decode_one(cur_q[0], cur_q[1], cur_a[0], cur_a[1], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      {
        const G u8 *X_hb1 = rq_ptr_u(X_u + (unsigned long)kb1 * 32ul);
        rx0 = ink_r9_iq3_ld4(X_hb1 + xoff0);
        if (ld_hi) rx1 = ink_r9_iq3_ld4(X_hb1 + xoff1);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj3(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 0u, h, lx_base0, acc);

      ink_r9_iq3_decode_one(cur_q[2], cur_q[3], cur_a[1], cur_a[2], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      ink_r9_iq3_load_hb1_q0(sb_p + row_off, h, cur_q);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj3(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 1u, h, lx_base0, acc);

      ink_r9_iq3_decode_one(cur_q[4], cur_q[5], cur_a[2], cur_a[3], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj3(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 2u, h, lx_base0, acc);

      *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[1] + st_x) = rx0;
      if (ld_hi) *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[1] + st_x + 4096u) = rx1;
      f32x4 rsx = (f32x4)(0.0f);
      if (is_s) {
        const G u8 *Xs_hb1 = rq_ptr_u(X_s + (unsigned long)kb1 * 4ul);
        rsx = rq_load_f32x4((const G float *)(Xs_hb1 + sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);

      ink_r9_iq3_decode_one(cur_q[6], cur_q[7], cur_a[3], cur_a[4], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj3(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 3u, h, lx_base0, acc);

      ink_r9_iq3_stage_sx(ink_r9_iq3_lds.u.k.sx[1], 64u, t0, is_s, rsx);
      cur_a[0] = cur_a[4];
      ink_r9_iq3_load_hb1_q1a(sb_p + row_off, h, cur_q, cur_a + 1u);
      lbarrier();

      const int more = (sb_u + 1u < mb);
      const u32 kbn = 8u * (sb_u + 1u);
      const G u8 *nxt_p = rq_ptr_u(ebase + (unsigned long)(sb_u + 1u) * 98ul);

      ink_r9_iq3_decode_one(cur_q[0], cur_q[1], cur_a[0], cur_a[1], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      if (more) {
        const G u8 *X_nxt = rq_ptr_u(X_u + (unsigned long)kbn * 32ul);
        rx0 = ink_r9_iq3_ld4(X_nxt + xoff0);
        if (ld_hi) rx1 = ink_r9_iq3_ld4(X_nxt + xoff1);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj3(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 0u, h, lx_base0, acc);

      ink_r9_iq3_decode_one(cur_q[2], cur_q[3], cur_a[1], cur_a[2], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      if (more) ink_r9_iq3_load_hb0_q0(nxt_p + row_off, h, cur_q);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj3(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 1u, h, lx_base0, acc);

      ink_r9_iq3_decode_one(cur_q[4], cur_q[5], cur_a[2], cur_a[3], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj3(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 2u, h, lx_base0, acc);

      f32x4 rsx1 = (f32x4)(0.0f);
      if (more) {
        *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[0] + st_x) = rx0;
        if (ld_hi) *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[0] + st_x + 4096u) = rx1;
        if (is_s) {
          const G u8 *Xs_nxt = rq_ptr_u(X_s + (unsigned long)kbn * 4ul);
          rsx1 = rq_load_f32x4((const G float *)(Xs_nxt + sxoff));
        }
      }
      __builtin_amdgcn_sched_barrier(0);

      ink_r9_iq3_decode_one(cur_q[6], cur_q[7], cur_a[3], cur_a[4], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj3(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 3u, h, lx_base0, acc);

      if (!more) break;
      ink_r9_iq3_stage_sx(ink_r9_iq3_lds.u.k.sx[0], 64u, t0, is_s, rsx1);
      ink_r9_iq3_load_hb0_q1a(nxt_p + row_off, h, cur_q, cur_a);
      cur_psel ^= 0x02020606u;
      lbarrier();
    }
  } else if (cls == 1u) {
    /* ================ Bucket 1: 17..32 pairs (WJ = 2, 0 K-loop branches) ================ */
    const int is_s = (w == 0u);
    if (is_s) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r9_iq3_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ink_r9_iq3_div_x(ink_r9_iq3_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
    const u32 sxoff = is_s ? ink_r9_iq3_div_x(ink_r9_iq3_lds.pid[t0], xdiv) * xs : 0u;

    u32 cur_psel = (((unsigned long)ebase + (unsigned long)row_off) & 2ul) ? 0x07060504u : 0x05040302u;
    u32 cur_q[8], cur_a[5];
    ink_r9_iq3_load_hb0_all(ebase + row_off, h, cur_q, cur_a);
    {
      const u32x4 rx0_i = ink_r9_iq3_ld4(X_u + xoff0);
      const f32x4 rsx_i = is_s ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
      *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[0] + st_x) = rx0_i;
      ink_r9_iq3_stage_sx(ink_r9_iq3_lds.u.k.sx[0], 32u, t0, is_s, rsx_i);
    }
    lbarrier();

    u32x4 rx0 = (u32x4)(0u);
#pragma unroll 1
    for (u32 sb = 0u; ; sb++) {
      const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
      const u32 kb1 = 8u * sb_u + 4u;
      const G u8 *sb_p = rq_ptr_u(ebase + (unsigned long)sb_u * 98ul);

      const float d2_0 = ie_rq_h2f((cur_q[0] >> ((cur_psel & 4u) << 2u)) & 0xFFFFu) * 0.5f;
      const float d2_x = I2F(xrow_i(F2I(d2_0)));
      const float d2 = h ? d2_x : d2_0;
      ink_r9_iq3_dc_t xdc;
      ink_r9_iq3_decode_one(cur_q[0], cur_q[1], cur_a[0], cur_a[1], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      {
        const G u8 *X_hb1 = rq_ptr_u(X_u + (unsigned long)kb1 * 32ul);
        rx0 = ink_r9_iq3_ld4(X_hb1 + xoff0);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj2(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 0u, h, lx_base0, acc);

      ink_r9_iq3_decode_one(cur_q[2], cur_q[3], cur_a[1], cur_a[2], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      ink_r9_iq3_load_hb1_q0(sb_p + row_off, h, cur_q);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj2(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 1u, h, lx_base0, acc);

      ink_r9_iq3_decode_one(cur_q[4], cur_q[5], cur_a[2], cur_a[3], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj2(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 2u, h, lx_base0, acc);

      *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[1] + st_x) = rx0;
      f32x4 rsx = (f32x4)(0.0f);
      if (is_s) {
        const G u8 *Xs_hb1 = rq_ptr_u(X_s + (unsigned long)kb1 * 4ul);
        rsx = rq_load_f32x4((const G float *)(Xs_hb1 + sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);

      ink_r9_iq3_decode_one(cur_q[6], cur_q[7], cur_a[3], cur_a[4], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj2(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 3u, h, lx_base0, acc);

      ink_r9_iq3_stage_sx(ink_r9_iq3_lds.u.k.sx[1], 32u, t0, is_s, rsx);
      cur_a[0] = cur_a[4];
      ink_r9_iq3_load_hb1_q1a(sb_p + row_off, h, cur_q, cur_a + 1u);
      lbarrier();

      const int more = (sb_u + 1u < mb);
      const u32 kbn = 8u * (sb_u + 1u);
      const G u8 *nxt_p = rq_ptr_u(ebase + (unsigned long)(sb_u + 1u) * 98ul);

      ink_r9_iq3_decode_one(cur_q[0], cur_q[1], cur_a[0], cur_a[1], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      if (more) {
        const G u8 *X_nxt = rq_ptr_u(X_u + (unsigned long)kbn * 32ul);
        rx0 = ink_r9_iq3_ld4(X_nxt + xoff0);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj2(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 0u, h, lx_base0, acc);

      ink_r9_iq3_decode_one(cur_q[2], cur_q[3], cur_a[1], cur_a[2], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      if (more) ink_r9_iq3_load_hb0_q0(nxt_p + row_off, h, cur_q);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj2(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 1u, h, lx_base0, acc);

      ink_r9_iq3_decode_one(cur_q[4], cur_q[5], cur_a[2], cur_a[3], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj2(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 2u, h, lx_base0, acc);

      f32x4 rsx1 = (f32x4)(0.0f);
      if (more) {
        *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[0] + st_x) = rx0;
        if (is_s) {
          const G u8 *Xs_nxt = rq_ptr_u(X_s + (unsigned long)kbn * 4ul);
          rsx1 = rq_load_f32x4((const G float *)(Xs_nxt + sxoff));
        }
      }
      __builtin_amdgcn_sched_barrier(0);

      ink_r9_iq3_decode_one(cur_q[6], cur_q[7], cur_a[3], cur_a[4], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj2(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 3u, h, lx_base0, acc);

      if (!more) break;
      ink_r9_iq3_stage_sx(ink_r9_iq3_lds.u.k.sx[0], 32u, t0, is_s, rsx1);
      ink_r9_iq3_load_hb0_q1a(nxt_p + row_off, h, cur_q, cur_a);
      cur_psel ^= 0x02020606u;
      lbarrier();
    }
  } else {
    /* ================ Bucket 0: 1..16 pairs (WJ = 1, -50% X & WMMA, 0 branches) ================ */
    const int ld_lo = (tk_ld < 16u);
    const int is_s = (t0 < 16u);
    if (is_s) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r9_iq3_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ld_lo ? (ink_r9_iq3_div_x(ink_r9_iq3_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u) : 0u;
    const u32 sxoff = is_s ? ink_r9_iq3_div_x(ink_r9_iq3_lds.pid[t0], xdiv) * xs : 0u;

    u32 cur_psel = (((unsigned long)ebase + (unsigned long)row_off) & 2ul) ? 0x07060504u : 0x05040302u;
    u32 cur_q[8], cur_a[5];
    ink_r9_iq3_load_hb0_all(ebase + row_off, h, cur_q, cur_a);
    {
      const u32x4 rx0_i = ld_lo ? ink_r9_iq3_ld4(X_u + xoff0) : (u32x4)(0u);
      const f32x4 rsx_i = is_s ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
      if (ld_lo) *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[0] + st_x) = rx0_i;
      ink_r9_iq3_stage_sx(ink_r9_iq3_lds.u.k.sx[0], 32u, t0, is_s, rsx_i);
    }
    lbarrier();

    u32x4 rx0 = (u32x4)(0u);
#pragma unroll 1
    for (u32 sb = 0u; ; sb++) {
      const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
      const u32 kb1 = 8u * sb_u + 4u;
      const G u8 *sb_p = rq_ptr_u(ebase + (unsigned long)sb_u * 98ul);

      const float d2_0 = ie_rq_h2f((cur_q[0] >> ((cur_psel & 4u) << 2u)) & 0xFFFFu) * 0.5f;
      const float d2_x = I2F(xrow_i(F2I(d2_0)));
      const float d2 = h ? d2_x : d2_0;
      ink_r9_iq3_dc_t xdc;
      ink_r9_iq3_decode_one(cur_q[0], cur_q[1], cur_a[0], cur_a[1], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      if (ld_lo) {
        const G u8 *X_hb1 = rq_ptr_u(X_u + (unsigned long)kb1 * 32ul);
        rx0 = ink_r9_iq3_ld4(X_hb1 + xoff0);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj1(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 0u, h, lx_base0, &acc[0]);

      ink_r9_iq3_decode_one(cur_q[2], cur_q[3], cur_a[1], cur_a[2], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      ink_r9_iq3_load_hb1_q0(sb_p + row_off, h, cur_q);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj1(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 1u, h, lx_base0, &acc[0]);

      ink_r9_iq3_decode_one(cur_q[4], cur_q[5], cur_a[2], cur_a[3], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj1(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 2u, h, lx_base0, &acc[0]);

      if (ld_lo) *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[1] + st_x) = rx0;
      f32x4 rsx = (f32x4)(0.0f);
      if (is_s) {
        const G u8 *Xs_hb1 = rq_ptr_u(X_s + (unsigned long)kb1 * 4ul);
        rsx = rq_load_f32x4((const G float *)(Xs_hb1 + sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);

      ink_r9_iq3_decode_one(cur_q[6], cur_q[7], cur_a[3], cur_a[4], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj1(xdc, ink_r9_iq3_lds.u.k.x[0], ink_r9_iq3_lds.u.k.sx[0], 3u, h, lx_base0, &acc[0]);

      ink_r9_iq3_stage_sx(ink_r9_iq3_lds.u.k.sx[1], 32u, t0, is_s, rsx);
      cur_a[0] = cur_a[4];
      ink_r9_iq3_load_hb1_q1a(sb_p + row_off, h, cur_q, cur_a + 1u);
      lbarrier();

      const int more = (sb_u + 1u < mb);
      const u32 kbn = 8u * (sb_u + 1u);
      const G u8 *nxt_p = rq_ptr_u(ebase + (unsigned long)(sb_u + 1u) * 98ul);

      ink_r9_iq3_decode_one(cur_q[0], cur_q[1], cur_a[0], cur_a[1], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      if (more && ld_lo) {
        const G u8 *X_nxt = rq_ptr_u(X_u + (unsigned long)kbn * 32ul);
        rx0 = ink_r9_iq3_ld4(X_nxt + xoff0);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj1(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 0u, h, lx_base0, &acc[0]);

      ink_r9_iq3_decode_one(cur_q[2], cur_q[3], cur_a[1], cur_a[2], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      if (more) ink_r9_iq3_load_hb0_q0(nxt_p + row_off, h, cur_q);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj1(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 1u, h, lx_base0, &acc[0]);

      ink_r9_iq3_decode_one(cur_q[4], cur_q[5], cur_a[2], cur_a[3], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj1(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 2u, h, lx_base0, &acc[0]);

      f32x4 rsx1 = (f32x4)(0.0f);
      if (more) {
        if (ld_lo) *(LDSP u32x4 *)(ink_r9_iq3_lds.u.k.x[0] + st_x) = rx0;
        if (is_s) {
          const G u8 *Xs_nxt = rq_ptr_u(X_s + (unsigned long)kbn * 4ul);
          rsx1 = rq_load_f32x4((const G float *)(Xs_nxt + sxoff));
        }
      }
      __builtin_amdgcn_sched_barrier(0);

      ink_r9_iq3_decode_one(cur_q[6], cur_q[7], cur_a[3], cur_a[4], cur_psel, d2, ink_r9_iq3_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r9_iq3_apply_wj1(xdc, ink_r9_iq3_lds.u.k.x[1], ink_r9_iq3_lds.u.k.sx[1], 3u, h, lx_base0, &acc[0]);

      if (!more) break;
      ink_r9_iq3_stage_sx(ink_r9_iq3_lds.u.k.sx[0], 32u, t0, is_s, rsx1);
      ink_r9_iq3_load_hb0_q1a(nxt_p + row_off, h, cur_q, cur_a);
      cur_psel ^= 0x02020606u;
      lbarrier();
    }
  }

  /* Single shared SITU + SoA Q8_0 epilogue (eliminates 4 duplicate copies for I-cache!). */
  __builtin_amdgcn_sched_barrier(0);
  lbarrier();
  ink_r9_iq3_epilogue_strip(&ink_r9_iq3_lds.u.ep[0][0][0], ink_r9_iq3_lds.pid, Qout,
                            is_up, r_loc, h, w, l, 0u, nj < 32u ? nj : 32u, blk0, nblk, nb_out, qstr,
                            beta, lbeta, 0u, acc);
  if (nj > 32u) {
    lbarrier();
    ink_r9_iq3_epilogue_strip(&ink_r9_iq3_lds.u.ep[0][0][0], ink_r9_iq3_lds.pid, Qout,
                              is_up, r_loc, h, w, l, 32u, nj - 32u, blk0, nblk, nb_out, qstr,
                              beta, lbeta, 2u, acc);
  }
}

#endif /* !IE_MEGA */
