/* ============================================================================
 * Inkling-2 Round 10 (Wave 9/10, H01):
 *   3-WGs/CU FLUTE 0-Extra-VRAM SoA-Layout, Zero-Realignment, 4-Bucket
 *   (16/32/48/64) Fused Prefill IQ3_XXS Gate/Up WMMA + Exact SITU + Q8_0
 *   for RX 9070 XT (gfx1201, wave32, NT=256).
 *
 * Target: MoE ffn_gate_exps + ffn_up_exps MAT_IQ3XXS + SITU + Q8_0 (~40% of
 *         4K/16K prefill). Consumes the exact 0-extra-VRAM (98 B -> 98 B)
 *         4-row wave-group SoA FLUTE layout specified in
 *         kernels/ie_inkling_r7_flute_gemv.inc:15-41 (shared with decode H34!):
 *           Plane 0 (p_qs):  n_sb_tot * 256 B (128-B aligned; 64 B/row/sb)
 *           Plane 1 (p_aux): n_sb_tot * 128 B (128-B aligned; 32 B/row/sb)
 *           Plane 2 (p_d):   n_sb_tot *   8 B (128-B aligned;  2 B/row/sb)
 *           where n_sb_tot = (rows / 4) * mb = (eff / 2) * (nb / 8).
 *
 * Key Upgrades over R36 promoted ie_inkling_gemm_rq_exp_iq3xxs_situ_r9 (113 VGPR):
 *   1. Eliminates 100% of unaligned 98-B weight loads (ink_r9_iq3_ld4) and
 *      ALL 16 v_perm_b32 (ink_r9_iq3_perm_al) realignment VALU ops per SB!
 *   2. Split-half coalesced qs loading: h=0 loads sub-blocks 0..1 (1x aligned
 *      global_load_b128), h=1 loads sub-blocks 2..3 (1x aligned b128) -> 50%
 *      fewer qs VMEM load instructions (4 -> 2 b128/SB) with 0 duplicate lane
 *      addresses and 3-K-block + barrier latency hiding via ping-pong q_cur/q_nxt!
 *   3. Aligned 1x global_load_b128 aux loads per half-SB (killing base[20]
 *      scalar load, cur_a[4], cur_psel, and all aux realignment perms) +
 *      direct aligned u16 d loads (killing d2 xrow/cndmask!).
 *   4. Preserves exact flat largest-first worklist (16/32/48/64, cls 3..0),
 *      v_dual_mul_f32 + v_dual_fmac_f32 dual-issue (0 v_fma_mix_f32),
 *      <= 116 VGPR + 19,712 B LDS (3 WGs/CU), 0/0 spills, 100% exact-SHA!
 *
 * Exported Kernel, Grid, Guard, and Parity:
 *   KERNEL ie_inkling_gemm_rq_exp_iq3xxs_flute_situ_r10(
 *       const G u8 *qw, const G u8 *X, G u8 *Qout, u32 eff, u32 nb,
 *       const G u32 *list, const G u32 *tiles, const G u32 *n_tiles,
 *       u32 q_bytes, u32 xs, u32 qstr, u32 xdiv, float beta, float lbeta)
 *   - Grid:   ((eff + 63u) / 64u, max_tiles, 1u), NT = 256
 *             where max_tiles = min(T*K, n_exp + (T*K)/64u)
 *   - Guard:  identical to ie_inkling_gemm_rq_exp_iq3xxs_situ_r9 PLUS L->egu
 *             repacked into FLUTE 4-row SoA layout in gpu_sink() (IE_INKLING_IQ3_REPACK=1).
 *   - Parity: 100% bitwise-exact (0-bit-diff in Qout codes, scales, asums) vs
 *             ie_inkling_gemm_rq_exp_iq3xxs_situ_r9!
 * ========================================================================== */

#ifndef IE_MEGA

struct __attribute__((packed, aligned(4))) ink_r10_fl_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 ink_r10_fl_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct ink_r10_fl_u32x4p *)p)->v;
}

typedef struct {
  v2i w0, w1;
  float s0;
} ink_r10_fl_dc_t;

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
} ink_r10_fl_lds_t;

_Static_assert(sizeof(ink_r10_fl_lds_t) == 19712u, "ink_r10_fl_lds_t must be 19,712 B (<= 20 KiB = 3 WGs/CU)");

static inline __attribute__((always_inline)) u32 ink_r10_fl_div_x(const u32 pid, const u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

/* Single-mul IQ3_XXS 7-bit+parity sign decode (bitwise-identical to rq_ksign7 + ie_rq_ssign). */
static inline __attribute__((always_inline)) v2i ink_r10_fl_ksw(const u32 g0, const u32 g1, const u32 s7) {
  const u32 p_ = s7 * 0x00204081u;
  const u32 t0 = p_ & 0x01010101u;
  const u32 t1 = ((p_ >> 4u) | ((u32)__builtin_popcount(s7) << 24u)) & 0x01010101u;
  const u32 m0 = __builtin_amdgcn_perm(0u, 0u, t0 | 0x0C0C0C0Cu);
  const u32 m1 = __builtin_amdgcn_perm(0u, 0u, t1 | 0x0C0C0C0Cu);
  const v2i v = {(int)((g0 ^ m0) + t0), (int)((g1 ^ m1) + t1)};
  return v;
}

/* Decode one aligned FLUTE sub-block (0 v_perm_b32 realignment!). */
static inline __attribute__((always_inline)) void ink_r10_fl_decode_qp(
    const u32 qp, const u32 auxv, const float d2, const LDSP u32 *lut, const u32 h, ink_r10_fl_dc_t *dc) {
  const u32 g00 = lut[qp & 255u];
  const u32 g01 = lut[(qp >> 8u) & 255u];
  const u32 g10 = lut[(qp >> 16u) & 255u];
  const u32 g11 = lut[qp >> 24u];
  __builtin_amdgcn_sched_barrier(0);
  const u32 s_sh = h ? 14u : 0u;
  dc->s0 = d2 * (0.5f + (float)(auxv >> 28u));
  dc->w0 = ink_r10_fl_ksw(g00, g01, (auxv >> s_sh) & 127u);
  dc->w1 = ink_r10_fl_ksw(g10, g11, (auxv >> (s_sh + 7u)) & 127u);
}

/* Split-half qp selectors (h=0 holds sub-blocks 0..1 {s0_h0,s0_h1,s1_h0,s1_h1};
 * h=1 holds sub-blocks 2..3 {s2_h0,s2_h1,s3_h0,s3_h1}; Clang 21 scalar-extract safe). */
static inline __attribute__((always_inline)) u32 ink_r10_fl_qp0(const u32x4 qv, const u32 h) {
  const u32 q0 = qv.x, q1 = qv.y;
  const u32 q1_x = (u32)xrow_i((int)q1);
  return h ? q1_x : q0;
}

static inline __attribute__((always_inline)) u32 ink_r10_fl_qp1(const u32x4 qv, const u32 h) {
  const u32 q2 = qv.z, q3 = qv.w;
  const u32 q3_x = (u32)xrow_i((int)q3);
  return h ? q3_x : q2;
}

static inline __attribute__((always_inline)) u32 ink_r10_fl_qp2(const u32x4 qv, const u32 h) {
  const u32 q0 = qv.x, q1 = qv.y;
  const u32 q0_x = (u32)xrow_i((int)q0);
  return h ? q1 : q0_x;
}

static inline __attribute__((always_inline)) u32 ink_r10_fl_qp3(const u32x4 qv, const u32 h) {
  const u32 q2 = qv.z, q3 = qv.w;
  const u32 q2_x = (u32)xrow_i((int)q2);
  return h ? q3 : q2_x;
}

/* Exact ((float)r * s0) * dx update for one 16-token tile (read-only "v"(acc) preserves v_dual_fmac_f32!). */
static inline __attribute__((always_inline)) void ink_r10_fl_accum_tile(
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

static inline __attribute__((always_inline)) void ink_r10_fl_apply_one_j(
    const ink_r10_fl_dc_t dck, const LDSP u8 *lx_k, const LDSP float *lsx_k,
    const u32 j0, f8 *acc) {
  const v8i zm = (v8i)(0x4B400000);
  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + j0 * 2048u);
  __builtin_amdgcn_sched_barrier(0);

  const v2i a0 = {(int)b0.x, (int)b0.y};
  const v2i a1 = {(int)b0.z, (int)b0.w};
  v8i r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, dck.w0, zm, 0);
  r0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, dck.w1, r0, 0);
  __builtin_amdgcn_sched_barrier(0);

  ink_r10_fl_accum_tile(r0, dck.s0, lsx_k + 16u * j0, &acc[j0]);
}

static inline __attribute__((always_inline)) void ink_r10_fl_apply_pair_j(
    const ink_r10_fl_dc_t dck, const LDSP u8 *lx_k, const LDSP float *lsx_k,
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
  ink_r10_fl_accum_tile(r0, dck.s0, sx0, &acc[j0]);
  ink_r10_fl_accum_tile(r1, dck.s0, sx0 + 16u, &acc[j0 + 1u]);
}

static inline __attribute__((always_inline)) void ink_r10_fl_apply_wj4(
    const ink_r10_fl_dc_t dck, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 k, const u32 h, const u32 lx_base0, f8 acc[4]) {
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * 64u + 8u * h;
  ink_r10_fl_apply_pair_j(dck, lx_k, lsx_k, 0u, acc);
  ink_r10_fl_apply_pair_j(dck, lx_k, lsx_k, 2u, acc);
}

static inline __attribute__((always_inline)) void ink_r10_fl_apply_wj3(
    const ink_r10_fl_dc_t dck, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 k, const u32 h, const u32 lx_base0, f8 acc[3]) {
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * 64u + 8u * h;
  ink_r10_fl_apply_pair_j(dck, lx_k, lsx_k, 0u, acc);
  ink_r10_fl_apply_one_j(dck, lx_k, lsx_k, 2u, acc);
}

static inline __attribute__((always_inline)) void ink_r10_fl_apply_wj2(
    const ink_r10_fl_dc_t dck, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 k, const u32 h, const u32 lx_base0, f8 acc[2]) {
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * 32u + 8u * h;
  ink_r10_fl_apply_pair_j(dck, lx_k, lsx_k, 0u, acc);
}

static inline __attribute__((always_inline)) void ink_r10_fl_apply_wj1(
    const ink_r10_fl_dc_t dck, const LDSP u8 *lx, const LDSP float *lsx,
    const u32 k, const u32 h, const u32 lx_base0, f8 *acc0) {
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * 32u + 8u * h;
  ink_r10_fl_apply_one_j(dck, lx_k, lsx_k, 0u, acc0);
}

static inline __attribute__((always_inline)) void ink_r10_fl_stage_sx(
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
static inline __attribute__((always_inline)) void ink_r10_fl_epilogue_strip(
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

KERNEL ie_inkling_gemm_rq_exp_iq3xxs_flute_situ_r10(
    const G u8 *qw, const G u8 *X, G u8 *Qout, u32 eff, u32 nb,
    const G u32 *list, const G u32 *tiles, const G u32 *n_tiles,
    u32 q_bytes, u32 xs, u32 qstr, u32 xdiv, float beta, float lbeta) {
  static LDS ink_r10_fl_lds_t ink_r10_fl_lds;
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

  /* FLUTE 3-plane SGPR base pointers (exact 98 B/row/sb, 128-B aligned!). */
  const unsigned long n_sb_tot = (unsigned long)(eff >> 1u) * (unsigned long)mb;
  const G u8 *p_qs_e  = rq_ptr_u(qw + (unsigned long)e * (unsigned long)q_bytes);
  const G u8 *p_aux_e = rq_ptr_u(p_qs_e + n_sb_tot * 256ul);
  const G u8 *p_d_e   = rq_ptr_u(p_qs_e + n_sb_tot * 384ul);

  const G u8 *X_u = rq_ptr_u(X);
  const G u8 *X_s = rq_ptr_u(X + 32ul * nb);

  /* Exact loop-invariant 32-bit VGPR byte offsets into Plane 0/1/2:
   *   r_idx = (w_row / 4) * mb * 4 + (w_row % 4)
   *   q_off = r_idx * 64 + h * 16   (hb0: +0, hb1: +32)
   *   a_off = r_idx * 32            (hb0: +0, hb1: +16)
   *   d_off = r_idx * 2             (u16 scale) */
  const u32 r_idx = ((w_row >> 2u) * mb) * 4u + (w_row & 3u);
  const u32 q_off = (r_idx << 6u) + (h << 4u);
  const u32 a_off = r_idx << 5u;
  const u32 d_off = r_idx << 1u;

  /* Stage 1 KiB IQ3_XXS grid LUT (synchronized by tile pid lbarrier below). */
  ink_r10_fl_lds.lut[t0] = ie_rq_iq3xxs_grid[t0];

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
      ink_r10_fl_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ink_r10_fl_div_x(ink_r10_fl_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
    const u32 xoff1 = ink_r10_fl_div_x(ink_r10_fl_lds.pid[tk_ld + 32u], xdiv) * xs + c8_ld * 16u;
    const u32 sxoff = is_s ? ink_r10_fl_div_x(ink_r10_fl_lds.pid[t0 & 63u], xdiv) * xs : 0u;

    u32x4 q_cur = *(const G u32x4 *)(p_qs_e + q_off);
    u32x4 a_cur = *(const G u32x4 *)(p_aux_e + a_off);
    u32 d_cur = (u32)(*(const G u16 *)(p_d_e + d_off));
    {
      const u32x4 rx0_i = ink_r10_fl_ld4(X_u + xoff0);
      const u32x4 rx1_i = ink_r10_fl_ld4(X_u + xoff1);
      const f32x4 rsx_i = is_s ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
      *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[0] + st_x) = rx0_i;
      *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[0] + st_x + 4096u) = rx1_i;
      ink_r10_fl_stage_sx(ink_r10_fl_lds.u.k.sx[0], 64u, t0, is_s, rsx_i);
    }
    lbarrier();

    u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
    for (u32 sb = 0u; ; sb++) {
      const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
      const u32 kb1 = 8u * sb_u + 4u;
      const G u8 *p_qs_sb  = rq_ptr_u(p_qs_e  + (unsigned long)sb_u * 256ul);
      const G u8 *p_aux_sb = rq_ptr_u(p_aux_e + (unsigned long)sb_u * 128ul);

      /* -------- hb0 (k = 0..3) -------- */
      const float d2 = ie_rq_h2f(d_cur) * 0.5f;
      ink_r10_fl_dc_t xdc;
      ink_r10_fl_decode_qp(ink_r10_fl_qp0(q_cur, h), a_cur.x, d2, ink_r10_fl_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      {
        const G u8 *X_hb1 = rq_ptr_u(X_u + (unsigned long)kb1 * 32ul);
        rx0 = ink_r10_fl_ld4(X_hb1 + xoff0);
        rx1 = ink_r10_fl_ld4(X_hb1 + xoff1);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj4(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 0u, h, lx_base0, acc);

      /* hb0 k=1: issue hb1 aligned 1x b128 qs into q_nxt (3 K-blocks + barrier hidden!). */
      ink_r10_fl_decode_qp(ink_r10_fl_qp1(q_cur, h), a_cur.y, d2, ink_r10_fl_lds.lut, h, &xdc);
      const u32x4 q_nxt = *(const G u32x4 *)(p_qs_sb + q_off + 32u);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj4(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 1u, h, lx_base0, acc);

      /* hb0 k=2 */
      ink_r10_fl_decode_qp(ink_r10_fl_qp2(q_cur, h), a_cur.z, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj4(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 2u, h, lx_base0, acc);

      /* Stage hb1 rx0/rx1 after k=2 (older in FIFO than q_nxt!) + prefetch rsx. */
      *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[1] + st_x) = rx0;
      *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[1] + st_x + 4096u) = rx1;
      f32x4 rsx = (f32x4)(0.0f);
      if (is_s) {
        const G u8 *Xs_hb1 = rq_ptr_u(X_s + (unsigned long)kb1 * 4ul);
        rsx = rq_load_f32x4((const G float *)(Xs_hb1 + sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);

      /* hb0 k=3 */
      ink_r10_fl_decode_qp(ink_r10_fl_qp3(q_cur, h), a_cur.w, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj4(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 3u, h, lx_base0, acc);

      /* Stage hb1 rsx + load hb1 aligned 1x b128 aux (L0$-hot), then sync! */
      ink_r10_fl_stage_sx(ink_r10_fl_lds.u.k.sx[1], 64u, t0, is_s, rsx);
      a_cur = *(const G u32x4 *)(p_aux_sb + a_off + 16u);
      lbarrier();

      /* -------- hb1 (k = 0..3) -------- */
      const int more = (sb_u + 1u < mb);
      const u32 kbn = 8u * (sb_u + 1u);
      const G u8 *p_qs_nxt  = rq_ptr_u(p_qs_e  + (unsigned long)(sb_u + 1u) * 256ul);
      const G u8 *p_aux_nxt = rq_ptr_u(p_aux_e + (unsigned long)(sb_u + 1u) * 128ul);
      const G u8 *p_d_nxt   = rq_ptr_u(p_d_e   + (unsigned long)(sb_u + 1u) * 8ul);

      ink_r10_fl_decode_qp(ink_r10_fl_qp0(q_nxt, h), a_cur.x, d2, ink_r10_fl_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      if (more) {
        const G u8 *X_nxt = rq_ptr_u(X_u + (unsigned long)kbn * 32ul);
        rx0 = ink_r10_fl_ld4(X_nxt + xoff0);
        rx1 = ink_r10_fl_ld4(X_nxt + xoff1);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj4(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 0u, h, lx_base0, acc);

      /* hb1 k=1: reload next-SB q_cur (3 K-blocks + barrier hidden!). */
      ink_r10_fl_decode_qp(ink_r10_fl_qp1(q_nxt, h), a_cur.y, d2, ink_r10_fl_lds.lut, h, &xdc);
      if (more) q_cur = *(const G u32x4 *)(p_qs_nxt + q_off);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj4(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 1u, h, lx_base0, acc);

      /* hb1 k=2 */
      ink_r10_fl_decode_qp(ink_r10_fl_qp2(q_nxt, h), a_cur.z, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj4(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 2u, h, lx_base0, acc);

      f32x4 rsx1 = (f32x4)(0.0f);
      if (more) {
        *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[0] + st_x) = rx0;
        *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[0] + st_x + 4096u) = rx1;
        if (is_s) {
          const G u8 *Xs_nxt = rq_ptr_u(X_s + (unsigned long)kbn * 4ul);
          rsx1 = rq_load_f32x4((const G float *)(Xs_nxt + sxoff));
        }
      }
      __builtin_amdgcn_sched_barrier(0);

      /* hb1 k=3 */
      ink_r10_fl_decode_qp(ink_r10_fl_qp3(q_nxt, h), a_cur.w, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj4(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 3u, h, lx_base0, acc);

      if (!more) break;
      ink_r10_fl_stage_sx(ink_r10_fl_lds.u.k.sx[0], 64u, t0, is_s, rsx1);
      a_cur = *(const G u32x4 *)(p_aux_nxt + a_off);
      d_cur = (u32)(*(const G u16 *)(p_d_nxt + d_off));
      lbarrier();
    }
  } else if (cls == 2u) {
    /* ================ Bucket 2: 33..48 pairs (WJ = 3, 0 K-loop branches) ================ */
    const int ld_hi = (tk_ld < 16u);
    const int is_s = (t0 < 48u);
    if (is_s) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r10_fl_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ink_r10_fl_div_x(ink_r10_fl_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
    const u32 xoff1 = ld_hi ? (ink_r10_fl_div_x(ink_r10_fl_lds.pid[tk_ld + 32u], xdiv) * xs + c8_ld * 16u) : 0u;
    const u32 sxoff = is_s ? ink_r10_fl_div_x(ink_r10_fl_lds.pid[t0], xdiv) * xs : 0u;

    u32x4 q_cur = *(const G u32x4 *)(p_qs_e + q_off);
    u32x4 a_cur = *(const G u32x4 *)(p_aux_e + a_off);
    u32 d_cur = (u32)(*(const G u16 *)(p_d_e + d_off));
    {
      const u32x4 rx0_i = ink_r10_fl_ld4(X_u + xoff0);
      const u32x4 rx1_i = ld_hi ? ink_r10_fl_ld4(X_u + xoff1) : (u32x4)(0u);
      const f32x4 rsx_i = is_s ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
      *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[0] + st_x) = rx0_i;
      if (ld_hi) *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[0] + st_x + 4096u) = rx1_i;
      ink_r10_fl_stage_sx(ink_r10_fl_lds.u.k.sx[0], 64u, t0, is_s, rsx_i);
    }
    lbarrier();

    u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
    for (u32 sb = 0u; ; sb++) {
      const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
      const u32 kb1 = 8u * sb_u + 4u;
      const G u8 *p_qs_sb  = rq_ptr_u(p_qs_e  + (unsigned long)sb_u * 256ul);
      const G u8 *p_aux_sb = rq_ptr_u(p_aux_e + (unsigned long)sb_u * 128ul);

      const float d2 = ie_rq_h2f(d_cur) * 0.5f;
      ink_r10_fl_dc_t xdc;
      ink_r10_fl_decode_qp(ink_r10_fl_qp0(q_cur, h), a_cur.x, d2, ink_r10_fl_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      {
        const G u8 *X_hb1 = rq_ptr_u(X_u + (unsigned long)kb1 * 32ul);
        rx0 = ink_r10_fl_ld4(X_hb1 + xoff0);
        if (ld_hi) rx1 = ink_r10_fl_ld4(X_hb1 + xoff1);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj3(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 0u, h, lx_base0, acc);

      ink_r10_fl_decode_qp(ink_r10_fl_qp1(q_cur, h), a_cur.y, d2, ink_r10_fl_lds.lut, h, &xdc);
      const u32x4 q_nxt = *(const G u32x4 *)(p_qs_sb + q_off + 32u);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj3(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 1u, h, lx_base0, acc);

      ink_r10_fl_decode_qp(ink_r10_fl_qp2(q_cur, h), a_cur.z, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj3(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 2u, h, lx_base0, acc);

      *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[1] + st_x) = rx0;
      if (ld_hi) *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[1] + st_x + 4096u) = rx1;
      f32x4 rsx = (f32x4)(0.0f);
      if (is_s) {
        const G u8 *Xs_hb1 = rq_ptr_u(X_s + (unsigned long)kb1 * 4ul);
        rsx = rq_load_f32x4((const G float *)(Xs_hb1 + sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);

      ink_r10_fl_decode_qp(ink_r10_fl_qp3(q_cur, h), a_cur.w, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj3(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 3u, h, lx_base0, acc);

      ink_r10_fl_stage_sx(ink_r10_fl_lds.u.k.sx[1], 64u, t0, is_s, rsx);
      a_cur = *(const G u32x4 *)(p_aux_sb + a_off + 16u);
      lbarrier();

      const int more = (sb_u + 1u < mb);
      const u32 kbn = 8u * (sb_u + 1u);
      const G u8 *p_qs_nxt  = rq_ptr_u(p_qs_e  + (unsigned long)(sb_u + 1u) * 256ul);
      const G u8 *p_aux_nxt = rq_ptr_u(p_aux_e + (unsigned long)(sb_u + 1u) * 128ul);
      const G u8 *p_d_nxt   = rq_ptr_u(p_d_e   + (unsigned long)(sb_u + 1u) * 8ul);

      ink_r10_fl_decode_qp(ink_r10_fl_qp0(q_nxt, h), a_cur.x, d2, ink_r10_fl_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      if (more) {
        const G u8 *X_nxt = rq_ptr_u(X_u + (unsigned long)kbn * 32ul);
        rx0 = ink_r10_fl_ld4(X_nxt + xoff0);
        if (ld_hi) rx1 = ink_r10_fl_ld4(X_nxt + xoff1);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj3(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 0u, h, lx_base0, acc);

      ink_r10_fl_decode_qp(ink_r10_fl_qp1(q_nxt, h), a_cur.y, d2, ink_r10_fl_lds.lut, h, &xdc);
      if (more) q_cur = *(const G u32x4 *)(p_qs_nxt + q_off);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj3(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 1u, h, lx_base0, acc);

      ink_r10_fl_decode_qp(ink_r10_fl_qp2(q_nxt, h), a_cur.z, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj3(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 2u, h, lx_base0, acc);

      f32x4 rsx1 = (f32x4)(0.0f);
      if (more) {
        *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[0] + st_x) = rx0;
        if (ld_hi) *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[0] + st_x + 4096u) = rx1;
        if (is_s) {
          const G u8 *Xs_nxt = rq_ptr_u(X_s + (unsigned long)kbn * 4ul);
          rsx1 = rq_load_f32x4((const G float *)(Xs_nxt + sxoff));
        }
      }
      __builtin_amdgcn_sched_barrier(0);

      ink_r10_fl_decode_qp(ink_r10_fl_qp3(q_nxt, h), a_cur.w, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj3(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 3u, h, lx_base0, acc);

      if (!more) break;
      ink_r10_fl_stage_sx(ink_r10_fl_lds.u.k.sx[0], 64u, t0, is_s, rsx1);
      a_cur = *(const G u32x4 *)(p_aux_nxt + a_off);
      d_cur = (u32)(*(const G u16 *)(p_d_nxt + d_off));
      lbarrier();
    }
  } else if (cls == 1u) {
    /* ================ Bucket 1: 17..32 pairs (WJ = 2, 0 K-loop branches) ================ */
    const int is_s = (w == 0u);
    if (is_s) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r10_fl_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ink_r10_fl_div_x(ink_r10_fl_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u;
    const u32 sxoff = is_s ? ink_r10_fl_div_x(ink_r10_fl_lds.pid[t0], xdiv) * xs : 0u;

    u32x4 q_cur = *(const G u32x4 *)(p_qs_e + q_off);
    u32x4 a_cur = *(const G u32x4 *)(p_aux_e + a_off);
    u32 d_cur = (u32)(*(const G u16 *)(p_d_e + d_off));
    {
      const u32x4 rx0_i = ink_r10_fl_ld4(X_u + xoff0);
      const f32x4 rsx_i = is_s ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
      *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[0] + st_x) = rx0_i;
      ink_r10_fl_stage_sx(ink_r10_fl_lds.u.k.sx[0], 32u, t0, is_s, rsx_i);
    }
    lbarrier();

    u32x4 rx0 = (u32x4)(0u);
#pragma unroll 1
    for (u32 sb = 0u; ; sb++) {
      const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
      const u32 kb1 = 8u * sb_u + 4u;
      const G u8 *p_qs_sb  = rq_ptr_u(p_qs_e  + (unsigned long)sb_u * 256ul);
      const G u8 *p_aux_sb = rq_ptr_u(p_aux_e + (unsigned long)sb_u * 128ul);

      const float d2 = ie_rq_h2f(d_cur) * 0.5f;
      ink_r10_fl_dc_t xdc;
      ink_r10_fl_decode_qp(ink_r10_fl_qp0(q_cur, h), a_cur.x, d2, ink_r10_fl_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      {
        const G u8 *X_hb1 = rq_ptr_u(X_u + (unsigned long)kb1 * 32ul);
        rx0 = ink_r10_fl_ld4(X_hb1 + xoff0);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj2(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 0u, h, lx_base0, acc);

      ink_r10_fl_decode_qp(ink_r10_fl_qp1(q_cur, h), a_cur.y, d2, ink_r10_fl_lds.lut, h, &xdc);
      const u32x4 q_nxt = *(const G u32x4 *)(p_qs_sb + q_off + 32u);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj2(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 1u, h, lx_base0, acc);

      ink_r10_fl_decode_qp(ink_r10_fl_qp2(q_cur, h), a_cur.z, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj2(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 2u, h, lx_base0, acc);

      *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[1] + st_x) = rx0;
      f32x4 rsx = (f32x4)(0.0f);
      if (is_s) {
        const G u8 *Xs_hb1 = rq_ptr_u(X_s + (unsigned long)kb1 * 4ul);
        rsx = rq_load_f32x4((const G float *)(Xs_hb1 + sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);

      ink_r10_fl_decode_qp(ink_r10_fl_qp3(q_cur, h), a_cur.w, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj2(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 3u, h, lx_base0, acc);

      ink_r10_fl_stage_sx(ink_r10_fl_lds.u.k.sx[1], 32u, t0, is_s, rsx);
      a_cur = *(const G u32x4 *)(p_aux_sb + a_off + 16u);
      lbarrier();

      const int more = (sb_u + 1u < mb);
      const u32 kbn = 8u * (sb_u + 1u);
      const G u8 *p_qs_nxt  = rq_ptr_u(p_qs_e  + (unsigned long)(sb_u + 1u) * 256ul);
      const G u8 *p_aux_nxt = rq_ptr_u(p_aux_e + (unsigned long)(sb_u + 1u) * 128ul);
      const G u8 *p_d_nxt   = rq_ptr_u(p_d_e   + (unsigned long)(sb_u + 1u) * 8ul);

      ink_r10_fl_decode_qp(ink_r10_fl_qp0(q_nxt, h), a_cur.x, d2, ink_r10_fl_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      if (more) {
        const G u8 *X_nxt = rq_ptr_u(X_u + (unsigned long)kbn * 32ul);
        rx0 = ink_r10_fl_ld4(X_nxt + xoff0);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj2(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 0u, h, lx_base0, acc);

      ink_r10_fl_decode_qp(ink_r10_fl_qp1(q_nxt, h), a_cur.y, d2, ink_r10_fl_lds.lut, h, &xdc);
      if (more) q_cur = *(const G u32x4 *)(p_qs_nxt + q_off);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj2(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 1u, h, lx_base0, acc);

      ink_r10_fl_decode_qp(ink_r10_fl_qp2(q_nxt, h), a_cur.z, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj2(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 2u, h, lx_base0, acc);

      f32x4 rsx1 = (f32x4)(0.0f);
      if (more) {
        *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[0] + st_x) = rx0;
        if (is_s) {
          const G u8 *Xs_nxt = rq_ptr_u(X_s + (unsigned long)kbn * 4ul);
          rsx1 = rq_load_f32x4((const G float *)(Xs_nxt + sxoff));
        }
      }
      __builtin_amdgcn_sched_barrier(0);

      ink_r10_fl_decode_qp(ink_r10_fl_qp3(q_nxt, h), a_cur.w, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj2(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 3u, h, lx_base0, acc);

      if (!more) break;
      ink_r10_fl_stage_sx(ink_r10_fl_lds.u.k.sx[0], 32u, t0, is_s, rsx1);
      a_cur = *(const G u32x4 *)(p_aux_nxt + a_off);
      d_cur = (u32)(*(const G u16 *)(p_d_nxt + d_off));
      lbarrier();
    }
  } else {
    /* ================ Bucket 0: 1..16 pairs (WJ = 1, 0 K-loop branches) ================ */
    const int ld_lo = (tk_ld < 16u);
    const int is_s = (t0 < 16u);
    if (is_s) {
      const u32 pj = pb + (t0 < nj ? t0 : 0u);
      ink_r10_fl_lds.pid[t0] = list[pj];
    }
    lbarrier();

    const u32 xoff0 = ld_lo ? (ink_r10_fl_div_x(ink_r10_fl_lds.pid[tk_ld], xdiv) * xs + c8_ld * 16u) : 0u;
    const u32 sxoff = is_s ? ink_r10_fl_div_x(ink_r10_fl_lds.pid[t0], xdiv) * xs : 0u;

    u32x4 q_cur = *(const G u32x4 *)(p_qs_e + q_off);
    u32x4 a_cur = *(const G u32x4 *)(p_aux_e + a_off);
    u32 d_cur = (u32)(*(const G u16 *)(p_d_e + d_off));
    {
      const u32x4 rx0_i = ld_lo ? ink_r10_fl_ld4(X_u + xoff0) : (u32x4)(0u);
      const f32x4 rsx_i = is_s ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
      if (ld_lo) *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[0] + st_x) = rx0_i;
      ink_r10_fl_stage_sx(ink_r10_fl_lds.u.k.sx[0], 32u, t0, is_s, rsx_i);
    }
    lbarrier();

    u32x4 rx0 = (u32x4)(0u);
#pragma unroll 1
    for (u32 sb = 0u; ; sb++) {
      const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
      const u32 kb1 = 8u * sb_u + 4u;
      const G u8 *p_qs_sb  = rq_ptr_u(p_qs_e  + (unsigned long)sb_u * 256ul);
      const G u8 *p_aux_sb = rq_ptr_u(p_aux_e + (unsigned long)sb_u * 128ul);

      const float d2 = ie_rq_h2f(d_cur) * 0.5f;
      ink_r10_fl_dc_t xdc;
      ink_r10_fl_decode_qp(ink_r10_fl_qp0(q_cur, h), a_cur.x, d2, ink_r10_fl_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      if (ld_lo) {
        const G u8 *X_hb1 = rq_ptr_u(X_u + (unsigned long)kb1 * 32ul);
        rx0 = ink_r10_fl_ld4(X_hb1 + xoff0);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj1(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 0u, h, lx_base0, &acc[0]);

      ink_r10_fl_decode_qp(ink_r10_fl_qp1(q_cur, h), a_cur.y, d2, ink_r10_fl_lds.lut, h, &xdc);
      const u32x4 q_nxt = *(const G u32x4 *)(p_qs_sb + q_off + 32u);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj1(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 1u, h, lx_base0, &acc[0]);

      ink_r10_fl_decode_qp(ink_r10_fl_qp2(q_cur, h), a_cur.z, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj1(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 2u, h, lx_base0, &acc[0]);

      if (ld_lo) *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[1] + st_x) = rx0;
      f32x4 rsx = (f32x4)(0.0f);
      if (is_s) {
        const G u8 *Xs_hb1 = rq_ptr_u(X_s + (unsigned long)kb1 * 4ul);
        rsx = rq_load_f32x4((const G float *)(Xs_hb1 + sxoff));
      }
      __builtin_amdgcn_sched_barrier(0);

      ink_r10_fl_decode_qp(ink_r10_fl_qp3(q_cur, h), a_cur.w, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj1(xdc, ink_r10_fl_lds.u.k.x[0], ink_r10_fl_lds.u.k.sx[0], 3u, h, lx_base0, &acc[0]);

      ink_r10_fl_stage_sx(ink_r10_fl_lds.u.k.sx[1], 32u, t0, is_s, rsx);
      a_cur = *(const G u32x4 *)(p_aux_sb + a_off + 16u);
      lbarrier();

      const int more = (sb_u + 1u < mb);
      const u32 kbn = 8u * (sb_u + 1u);
      const G u8 *p_qs_nxt  = rq_ptr_u(p_qs_e  + (unsigned long)(sb_u + 1u) * 256ul);
      const G u8 *p_aux_nxt = rq_ptr_u(p_aux_e + (unsigned long)(sb_u + 1u) * 128ul);
      const G u8 *p_d_nxt   = rq_ptr_u(p_d_e   + (unsigned long)(sb_u + 1u) * 8ul);

      ink_r10_fl_decode_qp(ink_r10_fl_qp0(q_nxt, h), a_cur.x, d2, ink_r10_fl_lds.lut, h, &xdc);
      __asm__ volatile("" : : "v"(xdc.w0), "v"(xdc.w1), "v"(xdc.s0));
      __builtin_amdgcn_sched_barrier(0);
      if (more && ld_lo) {
        const G u8 *X_nxt = rq_ptr_u(X_u + (unsigned long)kbn * 32ul);
        rx0 = ink_r10_fl_ld4(X_nxt + xoff0);
      }
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj1(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 0u, h, lx_base0, &acc[0]);

      ink_r10_fl_decode_qp(ink_r10_fl_qp1(q_nxt, h), a_cur.y, d2, ink_r10_fl_lds.lut, h, &xdc);
      if (more) q_cur = *(const G u32x4 *)(p_qs_nxt + q_off);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj1(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 1u, h, lx_base0, &acc[0]);

      ink_r10_fl_decode_qp(ink_r10_fl_qp2(q_nxt, h), a_cur.z, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj1(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 2u, h, lx_base0, &acc[0]);

      f32x4 rsx1 = (f32x4)(0.0f);
      if (more) {
        if (ld_lo) *(LDSP u32x4 *)(ink_r10_fl_lds.u.k.x[0] + st_x) = rx0;
        if (is_s) {
          const G u8 *Xs_nxt = rq_ptr_u(X_s + (unsigned long)kbn * 4ul);
          rsx1 = rq_load_f32x4((const G float *)(Xs_nxt + sxoff));
        }
      }
      __builtin_amdgcn_sched_barrier(0);

      ink_r10_fl_decode_qp(ink_r10_fl_qp3(q_nxt, h), a_cur.w, d2, ink_r10_fl_lds.lut, h, &xdc);
      __builtin_amdgcn_sched_barrier(0);
      ink_r10_fl_apply_wj1(xdc, ink_r10_fl_lds.u.k.x[1], ink_r10_fl_lds.u.k.sx[1], 3u, h, lx_base0, &acc[0]);

      if (!more) break;
      ink_r10_fl_stage_sx(ink_r10_fl_lds.u.k.sx[0], 32u, t0, is_s, rsx1);
      a_cur = *(const G u32x4 *)(p_aux_nxt + a_off);
      d_cur = (u32)(*(const G u16 *)(p_d_nxt + d_off));
      lbarrier();
    }
  }

  /* Single shared SITU + SoA Q8_0 epilogue (100% bitwise-exact vs r9!). */
  __builtin_amdgcn_sched_barrier(0);
  lbarrier();
  ink_r10_fl_epilogue_strip(&ink_r10_fl_lds.u.ep[0][0][0], ink_r10_fl_lds.pid, Qout,
                            is_up, r_loc, h, w, l, 0u, nj < 32u ? nj : 32u, blk0, nblk, nb_out, qstr,
                            beta, lbeta, 0u, acc);
  if (nj > 32u) {
    lbarrier();
    ink_r10_fl_epilogue_strip(&ink_r10_fl_lds.u.ep[0][0][0], ink_r10_fl_lds.pid, Qout,
                              is_up, r_loc, h, w, l, 32u, nj - 32u, blk0, nblk, nb_out, qstr,
                              beta, lbeta, 2u, acc);
  }
}

#endif /* !IE_MEGA */
