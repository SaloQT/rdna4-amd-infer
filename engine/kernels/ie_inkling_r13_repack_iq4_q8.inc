/* ============================================================================
 * Inkling-2-Mini Round 13 (Wave 13 - H36 + H37):
 *   0-VRAM Load-Time Wave-Interleaved Repacks & Decode/Verify GEMVs
 *   (IQ4_NL MoE Down, Q4_K Output Head, and Looped blk.6..17 Dense Q8_0)
 * Target: RX 9070 XT (gfx1201, wave32, NT=256, 64 CUs, 64 MiB MALL)
 *
 * Included at EOF of kernels/ie_inkling_kernels.c (after ie_kernels.c).
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r13_rp_ /
 * INK_R13_RP_ and all macros are #undef'd at EOF.
 * 0 warnings (-Wall -Wextra), 0/0 spills, priv=0, scratch=0, Clang bit_cast-safe.
 *
 * Byte-Exact 0-Extra-VRAM Layouts (repacked ONCE per physical tensor in gpu_open()):
 *
 * 1. H36a Wave-Interleaved IQ4_NL (EXACT rows*nb*16 B qw + rows*nb*2 B qs):
 *    - Requires (rows & 3u) == 0u (Inkling-2 ed: rows=2304, nb=36, n_exp=128).
 *    - 4-row groups rg = r >> 2, 8-block steps st = b >> 3 (width w_st = min(8, nb - st*8)).
 *    - Block (k = r & 3, sl = b & 7) in step st sits at flat block index:
 *        dst_blk = (rg * nb + st * 8u) * 4u + k * w_st + sl
 *      * Full 8-block steps (st < nb/8): dst_blk = rg * (nb * 4u) + st * 32u + l
 *        -> ALL 32 lanes read 1 contiguous 512 B qw strip + 64 B qs strip!
 *      * Tail step (nb % 8 != 0, e.g. last 4 blocks of nb=36): exact rem*4 packing
 *        -> 0 wasted padding bytes, 100.00% byte-conserved!
 *    - H38 256K COMPATIBLE (Core 128E + Outer Host-Pinned Mode A/B/B-overflow/C):
 *      Each expert e is independent with exact C->ed0_eb / C->ed1_eb bytes.
 *      Repack BOTH VRAM core layers (D2D) and H38 outer host-pinned C->ed0_host /
 *      C->ed1_host (batch_e=16 H2D->repack->D2H) in gpu_open() BEFORE
 *      ink_moe_cache_prepop(b) (identical to ink_iq3_flute at hip.c:3710-3734).
 *
 * 2. H36b Wave-Interleaved Q4_K Output Head (EXACT rows*nb*16 B qw + rows*(nb+nsb*2)*2 B qs):
 *    - Requires (rows & 3u) == 0u && (nb & 7u) == 0u (output.weight: 201024x2304, nb=72).
 *    - For 4-row group rg = r >> 2 and superblock sb in 0..nsb-1 (nsb = nb >> 3):
 *        qw_dst[(rg * nsb + sb) * 32 + l] = qw_src[(r * nb) + sb*8 + sub]  (512 B/wave!)
 *        sm_dst[(rg * nsb + sb) * 32 + l] = sm_src[(r * nb) + sb*8 + sub]  ( 64 B/wave!)
 *        dd_dst[(rg * nsb + sb) *  4 + k] = dd_src[r * nsb + sb]           ( 16 B/wave!)
 *    - CRITICAL HOST ORDERING:
 *      Run ie_inkling_repack_q4k_wave_r13 ONLY AFTER ds_split_setup() finishes!
 *      Chunked repack (e.g. nrows=16384) requires 3 D2D copies per chunk:
 *        (1) dst_qw -> out.d0 + row0*nb*16
 *        (2) dst_qs -> out.d1 + row0*nb*2 (nrows*nb*2 B)
 *        (3) dst_qs + nrows*nb*2 -> out.d1 + total_rows*nb*2 + row0*nsb*4 (nrows*nsb*4 B)
 *      Gate off when --ppl / T >= 17 OP_GEMV_Q4K (K_GEMM_Q4KR reads AoS).
 *
 * 3. H37 Wave-Striped Q8_0 with Inline Scales (EXACT rows*nb*34 B single buffer):
 *    - Requires (rows & 3u) == 0u && (nb & 7u) == 0u (blk.6..17 dense Q8_0: nb=72 or 96).
 *    - Single contiguous pointer stream q8p (eliminates 2nd TLB/L1 scale stream!).
 *    - Each 4-row x 8-block strip (rg, sb) occupies EXACTLY 1,088 B (64-B aligned):
 *        [   0 ..  511]: w0[l] (32 lanes x 16 B = 512 B coalesced)
 *        [ 512 .. 1023]: w1[l] (32 lanes x 16 B = 512 B coalesced)
 *        [1024 .. 1087]: s[l]  (32 lanes x  2 B =  64 B inline f16 scales)
 *    - HOST GATE: Requires contiguous d0+d1 placement (sdev_hot) for 0-VRAM in-place
 *      repack AND companion prefill/fused readers (keep opt-in IE_INKLING_Q8_REPACK=1).
 *
 * Exported Kernels (12, ALL R43-scrubbed TH_LOAD_RT, 0/0 spills, 100% exact-SHA):
 *   1.  ie_inkling_repack_iq4nl_wave_r13        (H36a GPU load-time repacker, Core+H38)
 *   2.  ie_inkling_gexp_iq4nl_wave_tail_t1_r13  (H36a T=1 4-op tail)
 *   3.  ie_inkling_gexp_iq4nl_wave_ver_r13      (H36a T=2..16 compact 4-op tail + opt tree)
 *   4.  ie_inkling_repack_q4k_wave_r13          (H36b GPU load-time chunked/full Q4_K repacker)
 *   5.  ie_inkling_q4k_head_wave_r13            (H36b T=1..8 single-pass Q4_K head + opt cap)
 *   6.  ie_inkling_repack_q8_wave_r13           (H37 GPU load-time Q8_0 inline-scale repacker)
 *   7.  ie_inkling_gemv_q8_wave_ht_r13          (H37 T=1..16 looped dense Q8_0, single stream)
 *   8.  ie_inkling_gexp_iq4nl_wave_t1_r13       (H36a T=1 standalone, H38 Mode A + Core)
 *   9.  ie_inkling_gexp_iq4nl_wave_compact_r13  (H36a T=2..16 standalone compact, H38 Mode B overflow)
 *   10. ie_inkling_gemm_iq4nl_wave_expmaj_r13   (H36a T>=17 WMMA prefill, H38 Mode C + Core)
 *   11. ie_inkling_q4k_head_wave_cap_topk_r13   (H36b T=1..8 fused head+cap+argmax/top-K, bit 84 drop-in)
 *   12. ie_inkling_ds_base_q4k_wave_idmap_r13   (H36b DSpark id_map base Q4_K gather, bit 88 drop-in)
 * ========================================================================== */

#ifndef IE_MEGA

_Static_assert(sizeof(f16) == 2u, "ink_r13_rp f16 must be 2 bytes");
_Static_assert(sizeof(float) == 4u && sizeof(u32) == 4u, "ink_r13_rp float/u32 must be 4 bytes");

#define INK_R13_RP_BUF_FLAGS 0x31004000u
#define INK_R13_RP_TMAX      8u
#define INK_R13_RP_Q8_STRIP  1088u /* 512 B w0 + 512 B w1 + 64 B inline f16 scales */

struct __attribute__((packed, aligned(4))) ink_r13_rp_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) __amdgpu_buffer_rsrc_t ink_r13_rp_mkbuf(
    const G void *base, u32 nbytes) {
  return __builtin_amdgcn_make_buffer_rsrc((G void *)rq_ptr_u((const G u8 *)base), 0, nbytes,
                                           INK_R13_RP_BUF_FLAGS);
}

/* R43 scrub: standard default cached loads (TH_LOAD_RT; aux=0). */
#define INK_R13_RP_RBL128(rsrc_, voff_, aux_) \
  __builtin_bit_cast(u32x4, __builtin_amdgcn_raw_buffer_load_b128((rsrc_), (int)(voff_), 0, 0))
#define INK_R13_RP_RBL32(rsrc_, voff_, aux_) \
  ((u32)__builtin_amdgcn_raw_buffer_load_b32((rsrc_), (int)(voff_), 0, 0))
#define INK_R13_RP_RBL16(rsrc_, voff_, aux_) \
  ((u16)__builtin_amdgcn_raw_buffer_load_b16((rsrc_), (int)(voff_), 0, 0))

static inline __attribute__((always_inline)) float ink_r13_rp_u32_f32(u32 bits) {
  return __builtin_bit_cast(float, bits);
}
static inline __attribute__((always_inline)) float ink_r13_rp_u16_f16f(u16 bits) {
  return (float)__builtin_bit_cast(f16, bits);
}

#define INK_R13_RP_RBLF32(rsrc_, voff_, aux_) \
  ink_r13_rp_u32_f32(INK_R13_RP_RBL32((rsrc_), (voff_), (aux_)))
#define INK_R13_RP_RBLF16F(rsrc_, voff_, aux_) \
  ink_r13_rp_u16_f16f(INK_R13_RP_RBL16((rsrc_), (voff_), (aux_)))

static inline __attribute__((always_inline)) u32x4 ink_r13_rp_nt4(const G void *p) {
  return *(const G u32x4 *)p;
}

static inline __attribute__((always_inline)) u32 ink_r13_rp_nt32(const G u32 *p) {
  return *p;
}

static inline __attribute__((always_inline)) u16 ink_r13_rp_nt16(const G u16 *p) {
  return *p;
}

static inline __attribute__((always_inline)) f16 ink_r13_rp_nt_f16(const G f16 *p) {
  const u16 bits = *(const G u16 *)p;
  return __builtin_bit_cast(f16, bits);
}

static inline __attribute__((always_inline)) u32x4 ink_r13_rp_rt4(const G void *p) {
  return ((const G struct ink_r13_rp_u32x4p *)p)->v;
}

static inline __attribute__((always_inline)) u32 ink_r13_rp_div_x(u32 pid, u32 xdiv) {
  return xdiv == 1u ? pid : (xdiv == 8u ? (pid >> 3u) : (pid / xdiv));
}

static inline __attribute__((always_inline)) float ink_r13_rp_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline __attribute__((always_inline)) float ink_r13_rp_oct_tree_f(float v) {
#pragma clang fp contract(off)
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}

#define INK_R13_RP_ACC_FMA(a0_, a1_, a2_, a3_, j_, v_, d_)          \
  do {                                                              \
    if ((j_) == 0u) (a0_) = __builtin_fmaf((v_), (d_), (a0_));      \
    else if ((j_) == 1u) (a1_) = __builtin_fmaf((v_), (d_), (a1_)); \
    else if ((j_) == 2u) (a2_) = __builtin_fmaf((v_), (d_), (a2_)); \
    else (a3_) = __builtin_fmaf((v_), (d_), (a3_));                 \
  } while (0)

static inline __attribute__((always_inline)) int ink_r13_rp_iq4_dot4(
    const u32x4 w4, const u32x4 a0, const u32x4 a1) {
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

static inline __attribute__((always_inline)) int ink_r13_rp_iq4_dot_pre(
    const u32 wl0, const u32 wh0, const u32 wl1, const u32 wh1,
    const u32 wl2, const u32 wh2, const u32 wl3, const u32 wh3,
    const u32x4 a0, const u32x4 a1) {
  const u32 a00 = a0.x, a01 = a0.y, a02 = a0.z, a03 = a0.w;
  const u32 a10 = a1.x, a11 = a1.y, a12 = a1.z, a13 = a1.w;
  u32 d = 0u;
  d = ie_dot4_ss(ie_dot4_ss(d, wl0, a00), wh0, a10);
  d = ie_dot4_ss(ie_dot4_ss(d, wl1, a01), wh1, a11);
  d = ie_dot4_ss(ie_dot4_ss(d, wl2, a02), wh2, a12);
  d = ie_dot4_ss(ie_dot4_ss(d, wl3, a03), wh3, a13);
  return (int)d;
}

static inline __attribute__((always_inline)) int ink_r13_rp_q8_dot8(
    const u32x4 w0, const u32x4 w1, const u32x4 a0, const u32x4 a1) {
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

static inline __attribute__((always_inline)) f32x4 ink_r13_rp_moe_combine4(
    const G f32x4 *y4, const G u32 *rt, u32 n4, u32 K, u32 i4) {
#pragma clang fp contract(off)
  const G float *w = (const G float *)(rt + K);
  f32x4 x0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  for (u32 k = 0u; k < K; k++) {
    const float wk = w[k];
    const f32x4 yk = y4[(unsigned long)k * (unsigned long)n4 + (unsigned long)i4];
#pragma unroll
    for (u32 c = 0u; c < 4u; c++) x0[c] = __builtin_fmaf(wk, yk[c], x0[c]);
  }
  return x0;
}

static inline __attribute__((always_inline)) f32x4 ink_r13_rp_sconv4(
    f32x4 x0, f32x4 x1, f32x4 x2, f32x4 x3,
    f32x4 w0, f32x4 w1, f32x4 w2, f32x4 w3, float escale) {
#pragma clang fp contract(off)
  const f32x4 y = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3;
  return (x0 + y) * escale;
}

static inline __attribute__((always_inline)) void ink_r13_rp_seed_ring(
    const G float *ring, u32 i0, u32 pos0, f32x4 *r1, f32x4 *r2, f32x4 *r3) {
  *r1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *r2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  *r3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (pos0 > 0u) {
    const G f32x4 *rg = (const G f32x4 *)ring + i0;
    const f32x4 v0 = rg[0], v1 = rg[1], v2 = rg[2], v3 = rg[3];
    const u32 s1 = (pos0 - 1u) & 3u;
    *r1 = (f32x4){ink_r13_rp_pick4(v0, s1), ink_r13_rp_pick4(v1, s1),
                  ink_r13_rp_pick4(v2, s1), ink_r13_rp_pick4(v3, s1)};
    if (pos0 >= 2u) {
      const u32 s2 = (pos0 - 2u) & 3u;
      *r2 = (f32x4){ink_r13_rp_pick4(v0, s2), ink_r13_rp_pick4(v1, s2),
                    ink_r13_rp_pick4(v2, s2), ink_r13_rp_pick4(v3, s2)};
    }
    if (pos0 >= 3u) {
      const u32 s3 = (pos0 - 3u) & 3u;
      *r3 = (f32x4){ink_r13_rp_pick4(v0, s3), ink_r13_rp_pick4(v1, s3),
                    ink_r13_rp_pick4(v2, s3), ink_r13_rp_pick4(v3, s3)};
    }
  }
}

static inline __attribute__((always_inline)) void ink_r13_rp_ring_commit(
    G float *ring, u32 i0, u32 pos0, u32 T, f32x4 x1, f32x4 x2, f32x4 x3, f32x4 x4) {
  if (T >= 4u) {
    const u32 s0 = (pos0 + T - 4u) & 3u;
    const f32x4 v0 = (s0 == 0u) ? x4 : ((s0 == 3u) ? x3 : ((s0 == 2u) ? x2 : x1));
    const f32x4 v1 = (s0 == 1u) ? x4 : ((s0 == 0u) ? x3 : ((s0 == 3u) ? x2 : x1));
    const f32x4 v2 = (s0 == 2u) ? x4 : ((s0 == 1u) ? x3 : ((s0 == 0u) ? x2 : x1));
    const f32x4 v3 = (s0 == 3u) ? x4 : ((s0 == 2u) ? x3 : ((s0 == 1u) ? x2 : x1));
    G f32x4 *rg = (G f32x4 *)ring + i0;
    rg[0] = (f32x4){v0.x, v1.x, v2.x, v3.x};
    rg[1] = (f32x4){v0.y, v1.y, v2.y, v3.y};
    rg[2] = (f32x4){v0.z, v1.z, v2.z, v3.z};
    rg[3] = (f32x4){v0.w, v1.w, v2.w, v3.w};
  } else {
    if (T >= 3u) {
      const u32 sl = (pos0 + T - 3u) & 3u;
      ring[(i0 + 0u) * 4u + sl] = x3.x; ring[(i0 + 1u) * 4u + sl] = x3.y;
      ring[(i0 + 2u) * 4u + sl] = x3.z; ring[(i0 + 3u) * 4u + sl] = x3.w;
    }
    if (T >= 2u) {
      const u32 sl = (pos0 + T - 2u) & 3u;
      ring[(i0 + 0u) * 4u + sl] = x2.x; ring[(i0 + 1u) * 4u + sl] = x2.y;
      ring[(i0 + 2u) * 4u + sl] = x2.z; ring[(i0 + 3u) * 4u + sl] = x2.w;
    }
    {
      const u32 sl = (pos0 + T - 1u) & 3u;
      ring[(i0 + 0u) * 4u + sl] = x1.x; ring[(i0 + 1u) * 4u + sl] = x1.y;
      ring[(i0 + 2u) * 4u + sl] = x1.z; ring[(i0 + 3u) * 4u + sl] = x1.w;
    }
  }
}

/* ============================================================================
 * 1. H36a: ie_inkling_repack_iq4nl_wave_r13
 *    Grid: (((rows >> 2) + 7) / 8, n_exp, 1)  [(72, n_b, 1) for rows=2304]
 * ========================================================================== */

KERNEL ie_inkling_repack_iq4nl_wave_r13(
    const G u32 *src_qw, const G u16 *src_qs,
    G u32 *dst_qw, G u16 *dst_qs,
    u32 rows, u32 nb, u32 qwn, u32 qsn, u32 n_exp) {
  if (!src_qw || !src_qs || !dst_qw || !dst_qs || src_qw == dst_qw || src_qs == dst_qs ||
      rows == 0u || (rows & 3u) != 0u || nb == 0u || n_exp == 0u)
    return;

  const u32 e = tokid();
  const u32 n_rg = rows >> 2u;
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 rg = (wgid() << 3u) + wv;
  if (e >= n_exp || rg >= n_rg) return;

  const u32 l = lane();
  const u32 k = l >> 3u;
  const u32 sl = l & 7u;
  const u32 r = (rg << 2u) + k;

  const G u32 *sqw = src_qw + (unsigned long)e * (unsigned long)qwn + (unsigned long)r * (unsigned long)nb * 4ul;
  const G u16 *sqs = src_qs + (unsigned long)e * (unsigned long)qsn + (unsigned long)r * (unsigned long)nb;
  G u32 *dqw = dst_qw + (unsigned long)e * (unsigned long)qwn;
  G u16 *dqs = dst_qs + (unsigned long)e * (unsigned long)qsn;

  const u32 nstep_full = nb >> 3u;
  const unsigned long grp_blk0 = (unsigned long)rg * (unsigned long)nb * 4ul;

#pragma clang loop unroll(disable)
  for (u32 st = 0u; st < nstep_full; st++) {
    const u32 b = (st << 3u) + sl;
    const u32x4 w4 = ink_r13_rp_nt4(sqw + (unsigned long)b * 4ul);
    const u16 s16 = ink_r13_rp_nt16(sqs + b);
    const unsigned long dblk = grp_blk0 + (unsigned long)(st << 5u) + (unsigned long)l;
    *(G u32x4 *)(dqw + dblk * 4ul) = w4;
    dqs[dblk] = s16;
  }

  const u32 rem = nb & 7u;
  if (rem > 0u && sl < rem) {
    const u32 b = (nstep_full << 3u) + sl;
    const u32x4 w4 = ink_r13_rp_nt4(sqw + (unsigned long)b * 4ul);
    const u16 s16 = ink_r13_rp_nt16(sqs + b);
    const unsigned long dblk = grp_blk0 + (unsigned long)(nstep_full << 5u) + (unsigned long)(k * rem + sl);
    *(G u32x4 *)(dqw + dblk * 4ul) = w4;
    dqs[dblk] = s16;
  }
}

/* ============================================================================
 * 2. H36a: ie_inkling_gexp_iq4nl_wave_tail_t1_r13
 *    T=1 4-Op Wave-Interleaved IQ4_NL + moe_sum + ffn_pc sconv + add
 *    Default cached TH_LOAD_RT on 512-B coalesced weight strips & exp_mid_q8 (R43).
 *    Grid: ((rows + 31) / 32, K, 1), 0-bit-diff exact-SHA, <= 72 VGPR (4 WG/CU).
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq4nl_wave_tail_t1_r13(
    const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *yd,
    u32 rows, u32 nb, const G u32 *route, u32 K, u32 qwn, u32 qsn,
    u32 astr, u32 bstr,
    const G float *v, G float *ring, const G float *res, G float *out,
    float escale, u32 pos0, G u32 *rcount) {
#pragma clang fp contract(off)
  static LDS u32 ink_r13_rp_t1_last;
  if (rows == 0u || (rows & 3u) != 0u || nb == 0u || K == 0u ||
      !qw || !qs || !xq || !yd || !route || !v || !ring || !res || !out || !rcount)
    return;

  const u32 k_slot = tokid();
  const u32 rb = wgid();
  const u32 rbase = rb << 5u;
  if (k_slot >= K || rbase >= rows) return;

  const u32 t0 = tid();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 r_wave = rbase + (wv << 2u);

  if (r_wave < rows) {
    const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[k_slot]);
    const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
    const u32 rg = r_wave >> 2u;
    const u32 r = r_wave + sub;

    const unsigned long grp_blk0 = (unsigned long)rg * (unsigned long)nb * 4ul;
    const G u32 *eqw = qw + (unsigned long)e * (unsigned long)qwn + grp_blk0 * 4ul;
    const G f16 *eqs = (const G f16 *)(qs + (unsigned long)e * (unsigned long)qsn) + grp_blk0;

    const G u8 *xp = xq + (unsigned long)k_slot * (unsigned long)astr;
    const __amdgpu_buffer_rsrc_t rx = ink_r13_rp_mkbuf(xp, nb * 36u);
    const u32 da_off = nb * 32u;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const u32 nstep_full = nb >> 3u;
    const u32 rem = nb & 7u;

#pragma clang loop unroll(disable)
    for (u32 st = 0u; st < nstep_full; st++) {
      const u32 b = (st << 3u) + sl;
      const unsigned long dblk = (unsigned long)(st << 5u) + (unsigned long)l;
      const u32x4 w4 = ink_r13_rp_nt4(eqw + dblk * 4ul);
      const float sc = (float)ink_r13_rp_nt_f16(eqs + dblk);
      const u32x4 x0 = INK_R13_RP_RBL128(rx, b * 32u, 2);
      const u32x4 x1 = INK_R13_RP_RBL128(rx, b * 32u + 16u, 2);
      const float dv = INK_R13_RP_RBLF32(rx, da_off + b * 4u, 2);
      const float val = sc * (float)ink_r13_rp_iq4_dot4(w4, x0, x1);
      const u32 j = st & 3u;
      INK_R13_RP_ACC_FMA(a0, a1, a2, a3, j, val, dv);
    }

    if (rem > 0u && sl < rem) {
      const u32 b = (nstep_full << 3u) + sl;
      const unsigned long dblk = (unsigned long)(nstep_full << 5u) + (unsigned long)(sub * rem + sl);
      const u32x4 w4 = ink_r13_rp_nt4(eqw + dblk * 4ul);
      const float sc = (float)ink_r13_rp_nt_f16(eqs + dblk);
      const u32x4 x0 = INK_R13_RP_RBL128(rx, b * 32u, 2);
      const u32x4 x1 = INK_R13_RP_RBL128(rx, b * 32u + 16u, 2);
      const float dv = INK_R13_RP_RBLF32(rx, da_off + b * 4u, 2);
      const float val = sc * (float)ink_r13_rp_iq4_dot4(w4, x0, x1);
      const u32 j = nstep_full & 3u;
      INK_R13_RP_ACC_FMA(a0, a1, a2, a3, j, val, dv);
    }

    const int full_wave = (nb > 16u);
    const float acc = ink_r13_rp_oct_tree_f(full_wave ? ((a0 + a2) + (a1 + a3)) : (a0 + a1));
    if (sl == 0u && r < rows) {
      G float *o = (G float *)((G u8 *)yd + (unsigned long)k_slot * (unsigned long)bstr);
      o[r] = acc;
    }
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (t0 == 0u) {
    ink_r13_rp_t1_last =
        (__scoped_atomic_fetch_add(&rcount[rb], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == K - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r13_rp_t1_last) return;
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
      f32x4 r1, r2, r3;
      ink_r13_rp_seed_ring(ring, i0, pos0, &r1, &r2, &r3);

      const f32x4 x0 = ink_r13_rp_moe_combine4((const G f32x4 *)yd, route, n4, K, i4);
      const f32x4 rv = ((const G f32x4 *)res)[i4];
      const f32x4 sc = ink_r13_rp_sconv4(x0, r1, r2, r3, sw0, sw1, sw2, sw3, escale);
      ((G f32x4 *)out)[i4] = rv + sc;

      const u32 sl = pos0 & 3u;
      ring[(i0 + 0u) * 4u + sl] = x0.x; ring[(i0 + 1u) * 4u + sl] = x0.y;
      ring[(i0 + 2u) * 4u + sl] = x0.z; ring[(i0 + 3u) * 4u + sl] = x0.w;
    }
  }
  if (t0 == 0u) rcount[rb] = 0u;
}

/* ============================================================================
 * 3. H36a: ie_inkling_gexp_iq4nl_wave_ver_r13
 *    T=2..16 Compact 4-Op Wave-Interleaved IQ4_NL + moe_sum + sconv + add + hist
 *    Default cached TH_LOAD_RT on 512-B coalesced weight strips & X (R43).
 *    Supports BOTH linear T=2..16 (tpar_pack == 0xFFFFFFFFu) AND arbitrary rooted
 *    trees T=2..8 (packed 4-bit parents in tpar_pack)!
 *    Grid: ((rows + 31) / 32, max_packs, 1), 1,040 B LDS, <= 80 VGPR (4 WG/CU).
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq4nl_wave_ver_r13(
    const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
    u32 rows, u32 nb,
    const G u32 *list, const G u32 *packs, const G u32 *n_packs,
    u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv,
    const G u32 *route, u32 K, u32 rts,
    const G float *v, G float *ring, const G float *res, G float *out, G float *hist,
    float escale, u32 pos0, u32 T, u32 yts, u32 rs, u32 os, u32 hrs,
    G u32 *rcount, u32 tpar_pack) {
#pragma clang fp contract(off)
  static LDS u32   ink_r13_rp_ver_last;
  static LDS f32x4 ink_r13_rp_ver_x[8u][8u] __attribute__((aligned(16)));

  if (rows == 0u || (rows & 3u) != 0u || nb == 0u || T == 0u || T > 16u || K == 0u ||
      !qw0 || !qs0 || !X || !Y || !list || !packs || !n_packs ||
      !route || !v || !ring || !res || !out || !rcount)
    return;

  const u32 by = tokid();
  const u32 npk = (u32)__builtin_amdgcn_readfirstlane((int)(*n_packs));
  const u32 rb = wgid();
  const u32 rbase = rb << 5u;
  if (by >= npk || rbase >= rows) return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)packs[by]);
  const u32 e = desc & 0xFFu;
  const u32 lo = (desc >> 8u) & 0xFFFu;
  const u32 nj = (desc >> 20u) & 0xFu;

  const u32 t0 = tid();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 r_wave = rbase + (wv << 2u);

  if (nj > 0u && r_wave < rows) {
    const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
    const u32 rg = r_wave >> 2u;
    const u32 r = r_wave + sub;
    const int full_wave = (nb > 16u);
    const u32 nstep_full = nb >> 3u;
    const u32 rem = nb & 7u;
    const u32 da_off = nb * 32u;

    const unsigned long grp_blk0 = (unsigned long)rg * (unsigned long)nb * 4ul;
    const G u32 *eqw = qw0 + (unsigned long)e * (unsigned long)qwn + grp_blk0 * 4ul;
    const G f16 *eqs = (const G f16 *)(qs0 + (unsigned long)e * (unsigned long)qsn) + grp_blk0;

    const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
    const G u8 *xp0 = X + (unsigned long)ink_r13_rp_div_x(p0, xdiv) * (unsigned long)xs;
    const __amdgpu_buffer_rsrc_t rx0 = ink_r13_rp_mkbuf(xp0, nb * 36u);

    if (nj == 1u) {
      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
      /* Four steps per pass, unrolled, so the accumulator index is a constant in each copy and
       * INK_R13_RP_ACC_FMA is one FMA instead of a scalar compare/branch ladder on st & 3.
       * Every accumulator still takes the same steps in the same order. */
#pragma clang loop unroll(disable)
      for (u32 st4 = 0u; st4 < nstep_full; st4 += 4u) {
#pragma unroll
        for (u32 j = 0u; j < 4u; j++) {
          const u32 st = st4 + j;
          if (st >= nstep_full) break;
          const u32 b = (st << 3u) + sl;
          const unsigned long dblk = (unsigned long)(st << 5u) + (unsigned long)l;
          const u32x4 w4 = ink_r13_rp_nt4(eqw + dblk * 4ul);
          const float sc = (float)ink_r13_rp_nt_f16(eqs + dblk);
          const u32x4 x0 = INK_R13_RP_RBL128(rx0, b * 32u, 2);
          const u32x4 x1 = INK_R13_RP_RBL128(rx0, b * 32u + 16u, 2);
          const float dv = INK_R13_RP_RBLF32(rx0, da_off + b * 4u, 2);
          const float val = sc * (float)ink_r13_rp_iq4_dot4(w4, x0, x1);
          INK_R13_RP_ACC_FMA(a0, a1, a2, a3, j, val, dv);
        }
      }
      if (rem > 0u && sl < rem) {
        const u32 b = (nstep_full << 3u) + sl;
        const unsigned long dblk = (unsigned long)(nstep_full << 5u) + (unsigned long)(sub * rem + sl);
        const u32x4 w4 = ink_r13_rp_nt4(eqw + dblk * 4ul);
        const float sc = (float)ink_r13_rp_nt_f16(eqs + dblk);
        const u32x4 x0 = INK_R13_RP_RBL128(rx0, b * 32u, 2);
        const u32x4 x1 = INK_R13_RP_RBL128(rx0, b * 32u + 16u, 2);
        const float dv = INK_R13_RP_RBLF32(rx0, da_off + b * 4u, 2);
        const float val = sc * (float)ink_r13_rp_iq4_dot4(w4, x0, x1);
        const u32 j = nstep_full & 3u;
        INK_R13_RP_ACC_FMA(a0, a1, a2, a3, j, val, dv);
      }
      const float acc0 = ink_r13_rp_oct_tree_f(full_wave ? ((a0 + a2) + (a1 + a3)) : (a0 + a1));
      if (sl == 0u && r < rows) ((G float *)(Y + (unsigned long)p0 * (unsigned long)ys))[r] = acc0;
    } else {
      const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 1u]);
      const u32 p2 = nj > 2u ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 2u]) : p0;
      const u32 p3 = nj > 3u ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 3u]) : p0;
      const __amdgpu_buffer_rsrc_t rx1 = ink_r13_rp_mkbuf(X + (unsigned long)ink_r13_rp_div_x(p1, xdiv) * (unsigned long)xs, nb * 36u);
      const __amdgpu_buffer_rsrc_t rx2 = ink_r13_rp_mkbuf(X + (unsigned long)ink_r13_rp_div_x(p2, xdiv) * (unsigned long)xs, nb * 36u);
      const __amdgpu_buffer_rsrc_t rx3 = ink_r13_rp_mkbuf(X + (unsigned long)ink_r13_rp_div_x(p3, xdiv) * (unsigned long)xs, nb * 36u);

      float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
      float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
      float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
      float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;

      const u32 nstep = nstep_full + (rem > 0u ? 1u : 0u);
      /* Unrolled by four for a constant accumulator index, as in the nj == 1 loop above. */
#pragma clang loop unroll(disable)
      for (u32 st4 = 0u; st4 < nstep; st4 += 4u) {
#pragma unroll
        for (u32 j_acc = 0u; j_acc < 4u; j_acc++) {
          const u32 st = st4 + j_acc;
          if (st >= nstep) break;
          const u32 b = (st << 3u) + sl;
          if (b < nb) {
            const unsigned long dblk = (st < nstep_full)
                                           ? ((unsigned long)(st << 5u) + (unsigned long)l)
                                           : ((unsigned long)(nstep_full << 5u) + (unsigned long)(sub * rem + sl));
            const u32x4 w4 = ink_r13_rp_rt4(eqw + dblk * 4ul);
            const float sc_f = (float)eqs[dblk];
            const u32 wx = w4.x, wy = w4.y, wz = w4.z, ww = w4.w;
            const u32 wl0 = ie_iq4_map(wx & 0x0F0F0F0Fu), wh0 = ie_iq4_map((wx >> 4u) & 0x0F0F0F0Fu);
            const u32 wl1 = ie_iq4_map(wy & 0x0F0F0F0Fu), wh1 = ie_iq4_map((wy >> 4u) & 0x0F0F0F0Fu);
            const u32 wl2 = ie_iq4_map(wz & 0x0F0F0F0Fu), wh2 = ie_iq4_map((wz >> 4u) & 0x0F0F0F0Fu);
            const u32 wl3 = ie_iq4_map(ww & 0x0F0F0F0Fu), wh3 = ie_iq4_map((ww >> 4u) & 0x0F0F0F0Fu);
            const u32 bo = b * 32u, doff = da_off + b * 4u;

            const float v0 = sc_f * (float)ink_r13_rp_iq4_dot_pre(
                wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3,
                INK_R13_RP_RBL128(rx0, bo, 2), INK_R13_RP_RBL128(rx0, bo + 16u, 2));
            INK_R13_RP_ACC_FMA(a0_0, a0_1, a0_2, a0_3, j_acc, v0, INK_R13_RP_RBLF32(rx0, doff, 2));

            const float v1 = sc_f * (float)ink_r13_rp_iq4_dot_pre(
                wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3,
                INK_R13_RP_RBL128(rx1, bo, 2), INK_R13_RP_RBL128(rx1, bo + 16u, 2));
            INK_R13_RP_ACC_FMA(a1_0, a1_1, a1_2, a1_3, j_acc, v1, INK_R13_RP_RBLF32(rx1, doff, 2));

            if (nj > 2u) {
              const float v2 = sc_f * (float)ink_r13_rp_iq4_dot_pre(
                  wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3,
                  INK_R13_RP_RBL128(rx2, bo, 2), INK_R13_RP_RBL128(rx2, bo + 16u, 2));
              INK_R13_RP_ACC_FMA(a2_0, a2_1, a2_2, a2_3, j_acc, v2, INK_R13_RP_RBLF32(rx2, doff, 2));
              if (nj > 3u) {
                const float v3 = sc_f * (float)ink_r13_rp_iq4_dot_pre(
                    wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3,
                    INK_R13_RP_RBL128(rx3, bo, 2), INK_R13_RP_RBL128(rx3, bo + 16u, 2));
                INK_R13_RP_ACC_FMA(a3_0, a3_1, a3_2, a3_3, j_acc, v3, INK_R13_RP_RBLF32(rx3, doff, 2));
              }
            }
          }
          __builtin_amdgcn_sched_barrier(0);
        }
      }

      const float acc0 = ink_r13_rp_oct_tree_f(full_wave ? ((a0_0 + a0_2) + (a0_1 + a0_3)) : (a0_0 + a0_1));
      if (sl == 0u && r < rows) ((G float *)(Y + (unsigned long)p0 * (unsigned long)ys))[r] = acc0;
      const float acc1 = ink_r13_rp_oct_tree_f(full_wave ? ((a1_0 + a1_2) + (a1_1 + a1_3)) : (a1_0 + a1_1));
      if (sl == 0u && r < rows) ((G float *)(Y + (unsigned long)p1 * (unsigned long)ys))[r] = acc1;
      if (nj > 2u) {
        const float acc2 = ink_r13_rp_oct_tree_f(full_wave ? ((a2_0 + a2_2) + (a2_1 + a2_3)) : (a2_0 + a2_1));
        if (sl == 0u && r < rows) ((G float *)(Y + (unsigned long)p2 * (unsigned long)ys))[r] = acc2;
      }
      if (nj > 3u) {
        const float acc3 = ink_r13_rp_oct_tree_f(full_wave ? ((a3_0 + a3_2) + (a3_1 + a3_3)) : (a3_0 + a3_1));
        if (sl == 0u && r < rows) ((G float *)(Y + (unsigned long)p3 * (unsigned long)ys))[r] = acc3;
      }
    }
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (t0 == 0u) {
    ink_r13_rp_ver_last =
        (__scoped_atomic_fetch_add(&rcount[rb], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == npk - 1u) ? 1u : 0u;
  }
  barrier();
  if (!ink_r13_rp_ver_last) return;
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
      f32x4 r1, r2, r3;
      ink_r13_rp_seed_ring(ring, i0, pos0, &r1, &r2, &r3);

      const int tree_on = (tpar_pack != 0xFFFFFFFFu && T <= 8u);
      f32x4 x1 = r1, x2 = r2, x3 = r3, prev_x0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
#pragma clang loop unroll(disable)
      for (u32 t = 0u; t < T; t++) {
        const G f32x4 *y4 = (const G f32x4 *)(Y + (unsigned long)t * (unsigned long)yts);
        const G u32 *rt = (const G u32 *)((const G u8 *)route + (unsigned long)t * (unsigned long)rts);
        const f32x4 x0 = ink_r13_rp_moe_combine4(y4, rt, n4, K, i4);
        if (tree_on) ink_r13_rp_ver_x[t0][t] = x0;
        if (hist) ((G f32x4 *)(hist + (unsigned long)t * (unsigned long)hrs))[i4] = x0;

        if (t == 0u) {
          x1 = r1; x2 = r2; x3 = r3;
        } else if (!tree_on) {
          x3 = x2; x2 = x1; x1 = prev_x0;
        } else {
          const u32 p1 = (tpar_pack >> (t << 2u)) & 0xFu;
          if (p1 == t - 1u) {
            x3 = x2; x2 = x1; x1 = prev_x0;
          } else {
            x1 = ink_r13_rp_ver_x[t0][p1];
            if (p1 == 0u) {
              x2 = r1; x3 = r2;
            } else {
              const u32 p2 = (tpar_pack >> (p1 << 2u)) & 0xFu;
              x2 = ink_r13_rp_ver_x[t0][p2];
              if (p2 == 0u) x3 = r1;
              else {
                const u32 p3 = (tpar_pack >> (p2 << 2u)) & 0xFu;
                x3 = ink_r13_rp_ver_x[t0][p3];
              }
            }
          }
        }

        const f32x4 rv = ((const G f32x4 *)((const G u8 *)res + (unsigned long)t * (unsigned long)rs))[i4];
        const f32x4 sc = ink_r13_rp_sconv4(x0, x1, x2, x3, sw0, sw1, sw2, sw3, escale);
        ((G f32x4 *)((G u8 *)out + (unsigned long)t * (unsigned long)os))[i4] = rv + sc;
        prev_x0 = x0;
      }

      if (!hist) ink_r13_rp_ring_commit(ring, i0, pos0, T, prev_x0, x1, x2, x3);
    }
  }
  if (t0 == 0u) rcount[rb] = 0u;
}

/* ============================================================================
 * 4. H36b: ie_inkling_repack_q4k_wave_r13
 *    Chunked or full-tensor 0-VRAM Q4_K wave-interleaved repacker.
 *    Grid: (((nrows >> 2) + 7) / 8, 1, 1)  [row0 & nrows multiples of 4, nb % 8 == 0]
 *    dst_qs layout for chunk nrows: sm[0..nrows*nb-1], dd[nrows*nb..nrows*(nb+nsb*2)-1].
 * ========================================================================== */

KERNEL ie_inkling_repack_q4k_wave_r13(
    const G u32 *src_qw, const G u16 *src_qs,
    G u32 *dst_qw, G u16 *dst_qs,
    u32 total_rows, u32 row0, u32 nrows, u32 nb) {
  if (!src_qw || !src_qs || !dst_qw || !dst_qs || src_qw == dst_qw || src_qs == dst_qs ||
      nrows == 0u || (row0 & 3u) != 0u || (nrows & 3u) != 0u || row0 + nrows > total_rows ||
      nb == 0u || (nb & 7u) != 0u)
    return;

  const u32 n_rg = nrows >> 2u;
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 rg = (wgid() << 3u) + wv;
  if (rg >= n_rg) return;

  const u32 l = lane(), k = l >> 3u, sub = l & 7u;
  const u32 r = row0 + (rg << 2u) + k;
  const u32 nsb = nb >> 3u;

  const G u32 *row_qw = src_qw + (unsigned long)r * (unsigned long)nb * 4ul;
  const G u16 *row_sm = src_qs + (unsigned long)r * (unsigned long)nb;
  const G u32 *row_dd = (const G u32 *)(src_qs + (unsigned long)total_rows * (unsigned long)nb +
                                        (unsigned long)r * (unsigned long)nsb * 2ul);

  G u32 *d_dd = (G u32 *)(dst_qs + (unsigned long)nrows * (unsigned long)nb);

#pragma clang loop unroll(disable)
  for (u32 sb = 0u; sb < nsb; sb++) {
    const u32 b = (sb << 3u) + sub;
    const u32x4 w4 = ink_r13_rp_nt4(row_qw + (unsigned long)b * 4ul);
    const u16 sm = ink_r13_rp_nt16(row_sm + b);
    const unsigned long strip = (unsigned long)rg * (unsigned long)nsb + (unsigned long)sb;
    *(G u32x4 *)(dst_qw + (strip * 32ul + (unsigned long)l) * 4ul) = w4;
    dst_qs[strip * 32ul + (unsigned long)l] = sm;
    if (sub == 0u) {
      d_dd[strip * 4ul + (unsigned long)k] = ink_r13_rp_nt32(row_dd + sb);
    }
  }
}

/* ============================================================================
 * 5. H36b: ie_inkling_q4k_head_wave_r13
 *    Single-pass T<=8 Wave-Interleaved Q4_K Output Head + Optional Logit-Cap.
 *    Default cached TH_LOAD_RT on 512-B qw + 64-B sm + 16-B dd strips & xq (R43).
 *    Grid: ((rows + 31) / 32, 1, 1), 0 B LDS, <= 80 VGPR (4 WG/CU), 100% exact-SHA.
 * ========================================================================== */

KERNEL ie_inkling_q4k_head_wave_r13(
    const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y,
    u32 rows, u32 nb, const G float *bias, const G float *res,
    u32 T, u32 xs, u32 ys, u32 rs, u32 unpadded, float cap) {
#pragma clang fp contract(off)
  if (T == 0u || T > INK_R13_RP_TMAX || rows == 0u || (rows & 3u) != 0u ||
      nb == 0u || (nb & 7u) != 0u || !qw || !qs || !xq || !y)
    return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 rg = (wgid() << 3u) + w;
  const u32 r_wave = rg << 2u;
  if (r_wave >= rows) return;

  const u32 l = lane(), k = l >> 3u, sub = l & 7u;
  const u32 r = r_wave + k;

  if (unpadded > 0u && r >= unpadded) {
    if (sub == 0u) {
#pragma unroll
      for (u32 j = 0u; j < INK_R13_RP_TMAX; j++) {
        if (j < T) y[(unsigned long)j * (unsigned long)ys + (unsigned long)r] = -1e30f;
      }
    }
    return;
  }

  const u32 nsb = nb >> 3u;
  const unsigned long strip0 = (unsigned long)rg * (unsigned long)nsb;
  const G u32 *w_dd = (const G u32 *)(qs + (unsigned long)rows * (unsigned long)nb);

  const u32 x_tot = (T > 1u) ? (T * xs) : (nb * 40u);
  const __amdgpu_buffer_rsrc_t rx = ink_r13_rp_mkbuf(xq, x_tot);
  const u32 da_base = nb * 32u;
  const u32 as_base = nb * 36u;

  float a0[INK_R13_RP_TMAX], a1[INK_R13_RP_TMAX], a2[INK_R13_RP_TMAX], a3[INK_R13_RP_TMAX];
#pragma unroll
  for (u32 j = 0u; j < INK_R13_RP_TMAX; j++) {
    a0[j] = 0.0f; a1[j] = 0.0f; a2[j] = 0.0f; a3[j] = 0.0f;
  }

#pragma clang loop unroll(disable)
  for (u32 sb = 0u; sb < nsb; sb++) {
    const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
    const u32 slot = (u32)__builtin_amdgcn_readfirstlane((int)(sb_u & 3u));
    const unsigned long strip = strip0 + (unsigned long)sb_u;

    const u32x4 w4 = ink_r13_rp_nt4(qw + (strip * 32ul + (unsigned long)l) * 4ul);
    const u32 sm = (u32)ink_r13_rp_nt16(qs + strip * 32ul + (unsigned long)l);
    const u32 dd_raw = ink_r13_rp_nt32(w_dd + strip * 4ul + (unsigned long)k);
    const u16 d_bits = (u16)(dd_raw & 0xFFFFu);
    const u16 dm_bits = (u16)(dd_raw >> 16u);
    const float dv = (float)__builtin_bit_cast(f16, d_bits);
    const float mv = (float)__builtin_bit_cast(f16, dm_bits);
    const float sc = dv * (float)(sm & 63u);
    const float mm = mv * (float)(sm >> 8u);

    const u32 lo0 = ie_q4k_lo(w4.x), hi0 = ie_q4k_hi(w4.x);
    const u32 lo1 = ie_q4k_lo(w4.y), hi1 = ie_q4k_hi(w4.y);
    const u32 lo2 = ie_q4k_lo(w4.z), hi2 = ie_q4k_hi(w4.z);
    const u32 lo3 = ie_q4k_lo(w4.w), hi3 = ie_q4k_hi(w4.w);

    const u32 b = (sb_u << 3u) + sub;
    const u32 aw_off = b * 32u;
    const u32 da_off = da_base + b * 4u;
    const u32 as_off = as_base + b * 4u;

#pragma unroll
    for (u32 j = 0u; j < INK_R13_RP_TMAX; j++) {
      if (j < T) {
        const u32 joff = (u32)__builtin_amdgcn_readfirstlane((int)(j * xs));
        const u32x4 act0 = INK_R13_RP_RBL128(rx, joff + aw_off, 2);
        const u32x4 act1 = INK_R13_RP_RBL128(rx, joff + aw_off + 16u, 2);
        const float da = INK_R13_RP_RBLF32(rx, joff + da_off, 2);
        const int as = (int)INK_R13_RP_RBL32(rx, joff + as_off, 2);
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
        if (slot == 0u) a0[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a0[j]));
        else if (slot == 1u) a1[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a1[j]));
        else if (slot == 2u) a2[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a2[j]));
        else a3[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a3[j]));
        __builtin_amdgcn_sched_barrier(0);
      }
    }
  }

#pragma unroll
  for (u32 j = 0u; j < INK_R13_RP_TMAX; j++) {
    if (j < T) {
      const float s = ink_r13_rp_oct_tree_f((a0[j] + a2[j]) + (a1[j] + a3[j]));
      if (sub == 0u) {
        const G float *rp = res ? (res + (unsigned long)j * (unsigned long)rs) : (const G float *)0;
        float val = epilogue(s, r, bias, rp);
        if (cap > 0.0f) val = cap * ie_tanhf_s(val / cap);
        y[(unsigned long)j * (unsigned long)ys + (unsigned long)r] = val;
      }
    }
  }
}

/* ============================================================================
 * 6. H37: ie_inkling_repack_q8_wave_r13
 *    0-VRAM Q8_0 4-row wave-striped repacker with inline scales (1,088 B/strip).
 *    Grid: (((rows >> 2) + 7) / 8, 1, 1)  [rows % 4 == 0, nb % 8 == 0]
 * ========================================================================== */

KERNEL ie_inkling_repack_q8_wave_r13(
    const G u32 *src_qw, const G f16 *src_qs, G u8 *dst_packed,
    u32 rows, u32 nb) {
  if (!src_qw || !src_qs || !dst_packed || rows == 0u || (rows & 3u) != 0u ||
      nb == 0u || (nb & 7u) != 0u)
    return;

  const u32 n_rg = rows >> 2u;
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 rg = (wgid() << 3u) + wv;
  if (rg >= n_rg) return;

  const u32 l = lane(), k = l >> 3u, sub = l & 7u;
  const u32 r = (rg << 2u) + k;
  const u32 nsb = nb >> 3u;

  const G u32 *row_qw = src_qw + (unsigned long)r * (unsigned long)nb * 8ul;
  const G f16 *row_qs = src_qs + (unsigned long)r * (unsigned long)nb;

#pragma clang loop unroll(disable)
  for (u32 sb = 0u; sb < nsb; sb++) {
    const u32 b = (sb << 3u) + sub;
    const u32x4 w0 = ink_r13_rp_nt4(row_qw + (unsigned long)b * 8ul);
    const u32x4 w1 = ink_r13_rp_nt4(row_qw + (unsigned long)b * 8ul + 4ul);
    const f16 sc = ink_r13_rp_nt_f16(row_qs + b);

    const unsigned long base = ((unsigned long)rg * (unsigned long)nsb + (unsigned long)sb) * INK_R13_RP_Q8_STRIP;
    G u8 *sp = dst_packed + base;
    *(G u32x4 *)(sp + (unsigned long)l * 16ul) = w0;
    *(G u32x4 *)(sp + 512ul + (unsigned long)l * 16ul) = w1;
    ((G f16 *)(sp + 1024ul))[l] = sc;
  }
}

/* ============================================================================
 * 7. H37: ie_inkling_gemv_q8_wave_ht_r13
 *    Looped blk.6..17 Dense Q8_0 Decode (T=1) & Verify (T=2..16) GEMV reading
 *    single-stream wave-striped q8p with inline scales (R43 default TH_LOAD_RT).
 *    Grid: ((rows + 31) / 32, 1, 1), 0 B LDS, <= 64 VGPR (4 WG/CU), exact-SHA.
 * ========================================================================== */

KERNEL ie_inkling_gemv_q8_wave_ht_r13(
    const G u8 *q8p, const G u8 *xq, G float *y,
    u32 rows, u32 nb, const G float *bias, const G float *res,
    u32 T, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  if (T == 0u || T > 16u || rows == 0u || (rows & 3u) != 0u ||
      nb == 0u || (nb & 7u) != 0u || !q8p || !xq || !y)
    return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 rg = (wgid() << 3u) + w;
  const u32 r_wave = rg << 2u;
  if (r_wave >= rows) return;

  const u32 l = lane(), k = l >> 3u, sub = l & 7u;
  const u32 r = r_wave + k;
  const u32 nsb = nb >> 3u;

  const unsigned long grp_bytes = (unsigned long)nsb * INK_R13_RP_Q8_STRIP;
  const __amdgpu_buffer_rsrc_t rw = ink_r13_rp_mkbuf(
      q8p + (unsigned long)rg * grp_bytes, (u32)grp_bytes);
  const u32 x_tot = (T > 1u) ? (T * xs) : (nb * 40u);
  const __amdgpu_buffer_rsrc_t rx = ink_r13_rp_mkbuf(xq, x_tot);
  const u32 da_base = nb * 32u;

  if (T == 1u) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma clang loop unroll(disable)
    for (u32 sb = 0u; sb < nsb; sb++) {
      const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
      const u32 slot = (u32)__builtin_amdgcn_readfirstlane((int)(sb_u & 3u));
      const u32 soff = sb_u * INK_R13_RP_Q8_STRIP;
      const u32 b = (sb_u << 3u) + sub;

      const u32x4 w0 = INK_R13_RP_RBL128(rw, soff + l * 16u, 2);
      const u32x4 w1 = INK_R13_RP_RBL128(rw, soff + 512u + l * 16u, 2);
      const float sw = INK_R13_RP_RBLF16F(rw, soff + 1024u + l * 2u, 2);
      const u32x4 x0 = INK_R13_RP_RBL128(rx, b * 32u, 2);
      const u32x4 x1 = INK_R13_RP_RBL128(rx, b * 32u + 16u, 2);
      const float da = INK_R13_RP_RBLF32(rx, da_base + b * 4u, 2);
      const float term = (sw * da) * (float)ink_r13_rp_q8_dot8(w0, w1, x0, x1);
      if (slot == 0u) a0 = a0 + term;
      else if (slot == 1u) a1 = a1 + term;
      else if (slot == 2u) a2 = a2 + term;
      else a3 = a3 + term;
    }
    const float s = ink_r13_rp_oct_tree_f((a0 + a2) + (a1 + a3));
    if (sub == 0u) y[r] = epilogue(s, r, bias, res);
    return;
  }

  /* Verify T = 2..16: single-pass for T<=8 (2 passes for 9..16), 4-row wave-striped HT/HT.
   * Two 4-token sub-batches reuse w0/w1/sw once per sb while keeping <= 64 VGPR (4 WG/CU)! */
#pragma clang loop unroll(disable)
  for (u32 t0 = 0u; t0 < T; t0 += INK_R13_RP_TMAX) {
    const u32 tc = (u32)__builtin_amdgcn_readfirstlane((int)((T - t0 < INK_R13_RP_TMAX) ? (T - t0) : INK_R13_RP_TMAX));
    float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f, a0_4 = 0.0f, a0_5 = 0.0f, a0_6 = 0.0f, a0_7 = 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f, a1_4 = 0.0f, a1_5 = 0.0f, a1_6 = 0.0f, a1_7 = 0.0f;
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f, a2_5 = 0.0f, a2_6 = 0.0f, a2_7 = 0.0f;
    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f, a3_7 = 0.0f;

#pragma clang loop unroll(disable)
    for (u32 sb = 0u; sb < nsb; sb++) {
      const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
      const u32 slot = (u32)__builtin_amdgcn_readfirstlane((int)(sb_u & 3u));
      const u32 soff = sb_u * INK_R13_RP_Q8_STRIP;
      const u32 b = (sb_u << 3u) + sub;

      const u32x4 w0 = INK_R13_RP_RBL128(rw, soff + (l << 4u), 2);
      const u32x4 w1 = INK_R13_RP_RBL128(rw, soff + 512u + (l << 4u), 2);
      const float sw = INK_R13_RP_RBLF16F(rw, soff + 1024u + (l << 1u), 2);
      const u32 aw_off = b << 5u;
      const u32 da_off = da_base + (b << 2u);
      __builtin_amdgcn_sched_barrier(0);

      float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
      {
        const u32 joff = (u32)__builtin_amdgcn_readfirstlane((int)((t0 + 0u) * xs));
        const u32x4 x0 = INK_R13_RP_RBL128(rx, joff + aw_off, 2);
        const u32x4 x1 = INK_R13_RP_RBL128(rx, joff + aw_off + 16u, 2);
        const float da = INK_R13_RP_RBLF32(rx, joff + da_off, 2);
        v0 = (sw * da) * (float)ink_r13_rp_q8_dot8(w0, w1, x0, x1);
        __asm__ volatile("" : "+v"(v0));
      }
      if (tc > 1u) {
        const u32 joff = (u32)__builtin_amdgcn_readfirstlane((int)((t0 + 1u) * xs));
        const u32x4 x0 = INK_R13_RP_RBL128(rx, joff + aw_off, 2);
        const u32x4 x1 = INK_R13_RP_RBL128(rx, joff + aw_off + 16u, 2);
        const float da = INK_R13_RP_RBLF32(rx, joff + da_off, 2);
        v1 = (sw * da) * (float)ink_r13_rp_q8_dot8(w0, w1, x0, x1);
        __asm__ volatile("" : "+v"(v1));
      }
      if (tc > 2u) {
        const u32 joff = (u32)__builtin_amdgcn_readfirstlane((int)((t0 + 2u) * xs));
        const u32x4 x0 = INK_R13_RP_RBL128(rx, joff + aw_off, 2);
        const u32x4 x1 = INK_R13_RP_RBL128(rx, joff + aw_off + 16u, 2);
        const float da = INK_R13_RP_RBLF32(rx, joff + da_off, 2);
        v2 = (sw * da) * (float)ink_r13_rp_q8_dot8(w0, w1, x0, x1);
        __asm__ volatile("" : "+v"(v2));
      }
      if (tc > 3u) {
        const u32 joff = (u32)__builtin_amdgcn_readfirstlane((int)((t0 + 3u) * xs));
        const u32x4 x0 = INK_R13_RP_RBL128(rx, joff + aw_off, 2);
        const u32x4 x1 = INK_R13_RP_RBL128(rx, joff + aw_off + 16u, 2);
        const float da = INK_R13_RP_RBLF32(rx, joff + da_off, 2);
        v3 = (sw * da) * (float)ink_r13_rp_q8_dot8(w0, w1, x0, x1);
        __asm__ volatile("" : "+v"(v3));
      }
      if (slot == 0u)      { a0_0 = a0_0 + v0; a0_1 = a0_1 + v1; a0_2 = a0_2 + v2; a0_3 = a0_3 + v3; }
      else if (slot == 1u) { a1_0 = a1_0 + v0; a1_1 = a1_1 + v1; a1_2 = a1_2 + v2; a1_3 = a1_3 + v3; }
      else if (slot == 2u) { a2_0 = a2_0 + v0; a2_1 = a2_1 + v1; a2_2 = a2_2 + v2; a2_3 = a2_3 + v3; }
      else                 { a3_0 = a3_0 + v0; a3_1 = a3_1 + v1; a3_2 = a3_2 + v2; a3_3 = a3_3 + v3; }

      if (tc > 4u) {
        v0 = 0.0f; v1 = 0.0f; v2 = 0.0f; v3 = 0.0f;
        {
          const u32 joff = (u32)__builtin_amdgcn_readfirstlane((int)((t0 + 4u) * xs));
          const u32x4 x0 = INK_R13_RP_RBL128(rx, joff + aw_off, 2);
          const u32x4 x1 = INK_R13_RP_RBL128(rx, joff + aw_off + 16u, 2);
          const float da = INK_R13_RP_RBLF32(rx, joff + da_off, 2);
          v0 = (sw * da) * (float)ink_r13_rp_q8_dot8(w0, w1, x0, x1);
          __asm__ volatile("" : "+v"(v0));
        }
        if (tc > 5u) {
          const u32 joff = (u32)__builtin_amdgcn_readfirstlane((int)((t0 + 5u) * xs));
          const u32x4 x0 = INK_R13_RP_RBL128(rx, joff + aw_off, 2);
          const u32x4 x1 = INK_R13_RP_RBL128(rx, joff + aw_off + 16u, 2);
          const float da = INK_R13_RP_RBLF32(rx, joff + da_off, 2);
          v1 = (sw * da) * (float)ink_r13_rp_q8_dot8(w0, w1, x0, x1);
          __asm__ volatile("" : "+v"(v1));
        }
        if (tc > 6u) {
          const u32 joff = (u32)__builtin_amdgcn_readfirstlane((int)((t0 + 6u) * xs));
          const u32x4 x0 = INK_R13_RP_RBL128(rx, joff + aw_off, 2);
          const u32x4 x1 = INK_R13_RP_RBL128(rx, joff + aw_off + 16u, 2);
          const float da = INK_R13_RP_RBLF32(rx, joff + da_off, 2);
          v2 = (sw * da) * (float)ink_r13_rp_q8_dot8(w0, w1, x0, x1);
          __asm__ volatile("" : "+v"(v2));
        }
        if (tc > 7u) {
          const u32 joff = (u32)__builtin_amdgcn_readfirstlane((int)((t0 + 7u) * xs));
          const u32x4 x0 = INK_R13_RP_RBL128(rx, joff + aw_off, 2);
          const u32x4 x1 = INK_R13_RP_RBL128(rx, joff + aw_off + 16u, 2);
          const float da = INK_R13_RP_RBLF32(rx, joff + da_off, 2);
          v3 = (sw * da) * (float)ink_r13_rp_q8_dot8(w0, w1, x0, x1);
          __asm__ volatile("" : "+v"(v3));
        }
        if (slot == 0u)      { a0_4 = a0_4 + v0; a0_5 = a0_5 + v1; a0_6 = a0_6 + v2; a0_7 = a0_7 + v3; }
        else if (slot == 1u) { a1_4 = a1_4 + v0; a1_5 = a1_5 + v1; a1_6 = a1_6 + v2; a1_7 = a1_7 + v3; }
        else if (slot == 2u) { a2_4 = a2_4 + v0; a2_5 = a2_5 + v1; a2_6 = a2_6 + v2; a2_7 = a2_7 + v3; }
        else                 { a3_4 = a3_4 + v0; a3_5 = a3_5 + v1; a3_6 = a3_6 + v2; a3_7 = a3_7 + v3; }
      }
    }

    const float ss[8] = {
      ink_r13_rp_oct_tree_f((a0_0 + a2_0) + (a1_0 + a3_0)),
      ink_r13_rp_oct_tree_f((a0_1 + a2_1) + (a1_1 + a3_1)),
      ink_r13_rp_oct_tree_f((a0_2 + a2_2) + (a1_2 + a3_2)),
      ink_r13_rp_oct_tree_f((a0_3 + a2_3) + (a1_3 + a3_3)),
      ink_r13_rp_oct_tree_f((a0_4 + a2_4) + (a1_4 + a3_4)),
      ink_r13_rp_oct_tree_f((a0_5 + a2_5) + (a1_5 + a3_5)),
      ink_r13_rp_oct_tree_f((a0_6 + a2_6) + (a1_6 + a3_6)),
      ink_r13_rp_oct_tree_f((a0_7 + a2_7) + (a1_7 + a3_7)),
    };
    u32 l_ep;
    __asm__ volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(l_ep));
    if ((l_ep & 7u) == 0u) {
      const u32 r_ep = r_wave + (l_ep >> 3u);
#pragma unroll
      for (u32 j = 0u; j < INK_R13_RP_TMAX; j++) {
        if (j < tc) {
          const G float *rp = res ? (res + (unsigned long)(t0 + j) * (unsigned long)rs) : (const G float *)0;
          y[(unsigned long)(t0 + j) * (unsigned long)ys + (unsigned long)r_ep] = epilogue(ss[j], r_ep, bias, rp);
        }
      }
    }
  }
}

/* ============================================================================
 * 8. H36a: ie_inkling_gexp_iq4nl_wave_t1_r13
 *    Standalone T=1 Wave-Interleaved IQ4_NL GEMV (drop-in for gexp_iq4nl_t1_nt_r10).
 *    Required by H38 Mode A (hip.c:6985, route = b->moe_rt_dev) & Core unfused T=1.
 *    Grid: ((rows + 31u) / 32u, K, 1u), NT=256, 0 B LDS, <= 40 VGPR (4 WG/CU).
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq4nl_wave_t1_r13(
    const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y,
    u32 rows, u32 nb, const G u32 *route, u32 qwn, u32 qsn,
    u32 astr, u32 bstr, u32 kind, u32 two_off) {
#pragma clang fp contract(off)
  (void)kind;
  (void)two_off;
  if (rows == 0u || (rows & 3u) != 0u || nb == 0u || !qw || !qs || !xq || !y || !route) return;

  const u32 k_slot = tokid();
  const u32 rb = wgid();
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 r_wave = (rb << 5u) + (wv << 2u);
  if (r_wave >= rows) return;

  const u32 e = (u32)__builtin_amdgcn_readfirstlane((int)route[k_slot]);
  const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
  const u32 rg = r_wave >> 2u;
  const u32 r = r_wave + sub;

  const unsigned long grp_blk0 = (unsigned long)rg * (unsigned long)nb * 4ul;
  const G u32 *eqw = qw + (unsigned long)e * (unsigned long)qwn + grp_blk0 * 4ul;
  const G f16 *eqs = (const G f16 *)(qs + (unsigned long)e * (unsigned long)qsn) + grp_blk0;

  const G u8 *xp = xq + (unsigned long)k_slot * (unsigned long)astr;
  const __amdgpu_buffer_rsrc_t rx = ink_r13_rp_mkbuf(xp, nb * 36u);
  const u32 da_off = nb * 32u;

  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  const u32 nstep_full = nb >> 3u;
  const u32 rem = nb & 7u;

  /* Four steps per pass, unrolled, so the accumulator index is a constant in each copy and
   * INK_R13_RP_ACC_FMA is one FMA instead of a scalar compare/branch ladder on st & 3.
   * Every accumulator still takes the same steps in the same order. */
#pragma clang loop unroll(disable)
  for (u32 st4 = 0u; st4 < nstep_full; st4 += 4u) {
#pragma unroll
    for (u32 j = 0u; j < 4u; j++) {
      const u32 st = st4 + j;
      if (st < nstep_full) {
        const u32 b = (st << 3u) + sl;
        const unsigned long dblk = (unsigned long)(st << 5u) + (unsigned long)l;
        const u32x4 w4 = ink_r13_rp_nt4(eqw + dblk * 4ul);
        const float sc = (float)ink_r13_rp_nt_f16(eqs + dblk);
        const u32x4 x0 = INK_R13_RP_RBL128(rx, b * 32u, 0);
        const u32x4 x1 = INK_R13_RP_RBL128(rx, b * 32u + 16u, 0);
        const float dv = INK_R13_RP_RBLF32(rx, da_off + b * 4u, 0);
        const float val = sc * (float)ink_r13_rp_iq4_dot4(w4, x0, x1);
        INK_R13_RP_ACC_FMA(a0, a1, a2, a3, j, val, dv);
      }
    }
  }
  if (rem > 0u && sl < rem) {
    const u32 b = (nstep_full << 3u) + sl;
    const unsigned long dblk = (unsigned long)(nstep_full << 5u) + (unsigned long)(sub * rem + sl);
    const u32x4 w4 = ink_r13_rp_nt4(eqw + dblk * 4ul);
    const float sc = (float)ink_r13_rp_nt_f16(eqs + dblk);
    const u32x4 x0 = INK_R13_RP_RBL128(rx, b * 32u, 0);
    const u32x4 x1 = INK_R13_RP_RBL128(rx, b * 32u + 16u, 0);
    const float dv = INK_R13_RP_RBLF32(rx, da_off + b * 4u, 0);
    const float val = sc * (float)ink_r13_rp_iq4_dot4(w4, x0, x1);
    const u32 j = nstep_full & 3u;
    INK_R13_RP_ACC_FMA(a0, a1, a2, a3, j, val, dv);
  }

  const int full_wave = (nb > 16u);
  const float acc = ink_r13_rp_oct_tree_f(full_wave ? ((a0 + a2) + (a1 + a3)) : (a0 + a1));
  if (sl == 0u && r < rows) {
    G float *o = (G float *)((G u8 *)y + (unsigned long)k_slot * (unsigned long)bstr);
    o[r] = acc;
  }
}

/* ============================================================================
 * 9. H36a: ie_inkling_gexp_iq4nl_wave_compact_r13
 *    Standalone T=2..16 Compact Wave-Interleaved IQ4_NL GEMV (drop-in for
 *    ie_inkling_gexp_iq4nl_bw_compact_r7). Required by H38 Mode B nu > ns
 *    sub-batch overflow (hip.c:7128) & non-B2 Mode B / Core verify (hip.c:7098).
 *    Grid: ((rows + 31u) / 32u, max_packs, 1u), NT=256, 0 B LDS, <= 64 VGPR.
 * ========================================================================== */

KERNEL ie_inkling_gexp_iq4nl_wave_compact_r13(
    const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
    u32 rows, u32 nb,
    const G u32 *list, const G u32 *packs, const G u32 *n_packs,
    u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv,
    u32 kind, const G u8 *res, u32 rs) {
#pragma clang fp contract(off)
  (void)kind;
  if (rows == 0u || (rows & 3u) != 0u || nb == 0u ||
      !qw0 || !qs0 || !X || !Y || !list || !packs || !n_packs)
    return;

  const u32 by = tokid();
  const u32 npk = (u32)__builtin_amdgcn_readfirstlane((int)(*n_packs));
  const u32 rb = wgid();
  const u32 rbase = rb << 5u;
  if (by >= npk || rbase >= rows) return;

  const u32 desc = (u32)__builtin_amdgcn_readfirstlane((int)packs[by]);
  const u32 e = desc & 0xFFu;
  const u32 lo = (desc >> 8u) & 0xFFFu;
  const u32 nj = (desc >> 20u) & 0xFu;
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 r_wave = rbase + (wv << 2u);
  if (nj == 0u || r_wave >= rows) return;

  const u32 l = lane(), sub = l >> 3u, sl = l & 7u;
  const u32 rg = r_wave >> 2u;
  const u32 r = r_wave + sub;
  const int full_wave = (nb > 16u);
  const u32 nstep_full = nb >> 3u;
  const u32 rem = nb & 7u;
  const u32 da_off = nb * 32u;

  const unsigned long grp_blk0 = (unsigned long)rg * (unsigned long)nb * 4ul;
  const G u32 *eqw = qw0 + (unsigned long)e * (unsigned long)qwn + grp_blk0 * 4ul;
  const G f16 *eqs = (const G f16 *)(qs0 + (unsigned long)e * (unsigned long)qsn) + grp_blk0;

  const u32 p0 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo]);
  const __amdgpu_buffer_rsrc_t rx0 = ink_r13_rp_mkbuf(
      X + (unsigned long)ink_r13_rp_div_x(p0, xdiv) * (unsigned long)xs, nb * 36u);

  if (nj == 1u) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma clang loop unroll(disable)
    for (u32 st = 0u; st < nstep_full; st++) {
      const u32 b = (st << 3u) + sl;
      const unsigned long dblk = (unsigned long)(st << 5u) + (unsigned long)l;
      const u32x4 w4 = ink_r13_rp_nt4(eqw + dblk * 4ul);
      const float sc = (float)ink_r13_rp_nt_f16(eqs + dblk);
      const u32x4 x0 = INK_R13_RP_RBL128(rx0, b * 32u, 0);
      const u32x4 x1 = INK_R13_RP_RBL128(rx0, b * 32u + 16u, 0);
      const float dv = INK_R13_RP_RBLF32(rx0, da_off + b * 4u, 0);
      const float val = sc * (float)ink_r13_rp_iq4_dot4(w4, x0, x1);
      const u32 j = st & 3u;
      INK_R13_RP_ACC_FMA(a0, a1, a2, a3, j, val, dv);
    }
    if (rem > 0u && sl < rem) {
      const u32 b = (nstep_full << 3u) + sl;
      const unsigned long dblk = (unsigned long)(nstep_full << 5u) + (unsigned long)(sub * rem + sl);
      const u32x4 w4 = ink_r13_rp_nt4(eqw + dblk * 4ul);
      const float sc = (float)ink_r13_rp_nt_f16(eqs + dblk);
      const u32x4 x0 = INK_R13_RP_RBL128(rx0, b * 32u, 0);
      const u32x4 x1 = INK_R13_RP_RBL128(rx0, b * 32u + 16u, 0);
      const float dv = INK_R13_RP_RBLF32(rx0, da_off + b * 4u, 0);
      const float val = sc * (float)ink_r13_rp_iq4_dot4(w4, x0, x1);
      const u32 j = nstep_full & 3u;
      INK_R13_RP_ACC_FMA(a0, a1, a2, a3, j, val, dv);
    }
    const float acc0 = ink_r13_rp_oct_tree_f(full_wave ? ((a0 + a2) + (a1 + a3)) : (a0 + a1));
    if (sl == 0u && r < rows) {
      const G float *rp0 = res ? (const G float *)(res + (unsigned long)p0 * (unsigned long)rs) : (const G float *)0;
      ((G float *)(Y + (unsigned long)p0 * (unsigned long)ys))[r] = epilogue(acc0, r, 0, rp0);
    }
    return;
  }

  const u32 p1 = (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 1u]);
  const u32 p2 = nj > 2u ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 2u]) : p0;
  const u32 p3 = nj > 3u ? (u32)__builtin_amdgcn_readfirstlane((int)list[lo + 3u]) : p0;
  const __amdgpu_buffer_rsrc_t rx1 = ink_r13_rp_mkbuf(X + (unsigned long)ink_r13_rp_div_x(p1, xdiv) * (unsigned long)xs, nb * 36u);
  const __amdgpu_buffer_rsrc_t rx2 = ink_r13_rp_mkbuf(X + (unsigned long)ink_r13_rp_div_x(p2, xdiv) * (unsigned long)xs, nb * 36u);
  const __amdgpu_buffer_rsrc_t rx3 = ink_r13_rp_mkbuf(X + (unsigned long)ink_r13_rp_div_x(p3, xdiv) * (unsigned long)xs, nb * 36u);

  float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
  float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
  float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
  float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;

  const u32 nstep = nstep_full + (rem > 0u ? 1u : 0u);
#pragma clang loop unroll(disable)
  for (u32 st = 0u; st < nstep; st++) {
    const u32 b = (st << 3u) + sl;
    if (b < nb) {
      const unsigned long dblk = (st < nstep_full)
                                     ? ((unsigned long)(st << 5u) + (unsigned long)l)
                                     : ((unsigned long)(nstep_full << 5u) + (unsigned long)(sub * rem + sl));
      const u32x4 w4 = ink_r13_rp_rt4(eqw + dblk * 4ul);
      const float sc_f = (float)eqs[dblk];
      const u32 wx = w4.x, wy = w4.y, wz = w4.z, ww = w4.w;
      const u32 wl0 = ie_iq4_map(wx & 0x0F0F0F0Fu), wh0 = ie_iq4_map((wx >> 4u) & 0x0F0F0F0Fu);
      const u32 wl1 = ie_iq4_map(wy & 0x0F0F0F0Fu), wh1 = ie_iq4_map((wy >> 4u) & 0x0F0F0F0Fu);
      const u32 wl2 = ie_iq4_map(wz & 0x0F0F0F0Fu), wh2 = ie_iq4_map((wz >> 4u) & 0x0F0F0F0Fu);
      const u32 wl3 = ie_iq4_map(ww & 0x0F0F0F0Fu), wh3 = ie_iq4_map((ww >> 4u) & 0x0F0F0F0Fu);
      const u32 j_acc = st & 3u;
      const u32 bo = b * 32u, doff = da_off + b * 4u;

      const float v0 = sc_f * (float)ink_r13_rp_iq4_dot_pre(
          wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3,
          INK_R13_RP_RBL128(rx0, bo, 0), INK_R13_RP_RBL128(rx0, bo + 16u, 0));
      INK_R13_RP_ACC_FMA(a0_0, a0_1, a0_2, a0_3, j_acc, v0, INK_R13_RP_RBLF32(rx0, doff, 0));

      const float v1 = sc_f * (float)ink_r13_rp_iq4_dot_pre(
          wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3,
          INK_R13_RP_RBL128(rx1, bo, 0), INK_R13_RP_RBL128(rx1, bo + 16u, 0));
      INK_R13_RP_ACC_FMA(a1_0, a1_1, a1_2, a1_3, j_acc, v1, INK_R13_RP_RBLF32(rx1, doff, 0));

      if (nj > 2u) {
        const float v2 = sc_f * (float)ink_r13_rp_iq4_dot_pre(
            wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3,
            INK_R13_RP_RBL128(rx2, bo, 0), INK_R13_RP_RBL128(rx2, bo + 16u, 0));
        INK_R13_RP_ACC_FMA(a2_0, a2_1, a2_2, a2_3, j_acc, v2, INK_R13_RP_RBLF32(rx2, doff, 0));
        if (nj > 3u) {
          const float v3 = sc_f * (float)ink_r13_rp_iq4_dot_pre(
              wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3,
              INK_R13_RP_RBL128(rx3, bo, 0), INK_R13_RP_RBL128(rx3, bo + 16u, 0));
          INK_R13_RP_ACC_FMA(a3_0, a3_1, a3_2, a3_3, j_acc, v3, INK_R13_RP_RBLF32(rx3, doff, 0));
        }
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  const float acc0 = ink_r13_rp_oct_tree_f(full_wave ? ((a0_0 + a0_2) + (a0_1 + a0_3)) : (a0_0 + a0_1));
  if (sl == 0u && r < rows) {
    const G float *rp = res ? (const G float *)(res + (unsigned long)p0 * (unsigned long)rs) : (const G float *)0;
    ((G float *)(Y + (unsigned long)p0 * (unsigned long)ys))[r] = epilogue(acc0, r, 0, rp);
  }
  const float acc1 = ink_r13_rp_oct_tree_f(full_wave ? ((a1_0 + a1_2) + (a1_1 + a1_3)) : (a1_0 + a1_1));
  if (sl == 0u && r < rows) {
    const G float *rp = res ? (const G float *)(res + (unsigned long)p1 * (unsigned long)rs) : (const G float *)0;
    ((G float *)(Y + (unsigned long)p1 * (unsigned long)ys))[r] = epilogue(acc1, r, 0, rp);
  }
  if (nj > 2u) {
    const float acc2 = ink_r13_rp_oct_tree_f(full_wave ? ((a2_0 + a2_2) + (a2_1 + a2_3)) : (a2_0 + a2_1));
    if (sl == 0u && r < rows) {
      const G float *rp = res ? (const G float *)(res + (unsigned long)p2 * (unsigned long)rs) : (const G float *)0;
      ((G float *)(Y + (unsigned long)p2 * (unsigned long)ys))[r] = epilogue(acc2, r, 0, rp);
    }
  }
  if (nj > 3u) {
    const float acc3 = ink_r13_rp_oct_tree_f(full_wave ? ((a3_0 + a3_2) + (a3_1 + a3_3)) : (a3_0 + a3_1));
    if (sl == 0u && r < rows) {
      const G float *rp = res ? (const G float *)(res + (unsigned long)p3 * (unsigned long)rs) : (const G float *)0;
      ((G float *)(Y + (unsigned long)p3 * (unsigned long)ys))[r] = epilogue(acc3, r, 0, rp);
    }
  }
}

/* ============================================================================
 * 10. H36a: ie_inkling_gemm_iq4nl_wave_expmaj_r13
 *     T>=17 Expert-Major 3-WG/CU Wave-Interleaved IQ4_NL Prefill WMMA GEMM
 *     (exact 17-arg drop-in for ie_inkling_gemm_iq4nl_expmaj_r10!).
 *     Works on BOTH 128E Core Prefill AND H38 Mode C Wave-Streaming
 *     (hip.c:7217-7223, passing offs_w = b->ms_offs + e_lo, grid.y = cnt).
 *     Grid: ((rows + 127u) / 128u, list ? cnt : (np + 63u) / 64u, 1u), NT=256.
 * ========================================================================== */

#define INK_R13_RP_PF_KB     4u
#define INK_R13_RP_PF_R      128u
#define INK_R13_RP_PF_SW_STR 160u
#define INK_R13_RP_PF_EP_STR 129u

typedef unsigned int ink_r13_rp_u32x2 __attribute__((ext_vector_type(2)));
struct __attribute__((packed, aligned(4))) ink_r13_rp_u32x2p {
  ink_r13_rp_u32x2 v;
};

static inline __attribute__((always_inline)) ink_r13_rp_u32x2 ink_r13_rp_ld2(const G void *p) {
  return ((const G struct ink_r13_rp_u32x2p *)p)->v;
}

typedef struct {
  union {
    struct {
      u8    x[2][64u * 128u] __attribute__((aligned(16)));
      f16   sw[2][INK_R13_RP_PF_KB * INK_R13_RP_PF_SW_STR] __attribute__((aligned(16)));
      float sx[2][INK_R13_RP_PF_KB * 64u] __attribute__((aligned(32)));
    } k;
    float ep[32u][INK_R13_RP_PF_EP_STR] __attribute__((aligned(32)));
  } u;
  u32 pid[64] __attribute__((aligned(16)));
} ink_r13_rp_pf_lds_t;

_Static_assert(sizeof(ink_r13_rp_pf_lds_t) == 21248u, "ink_r13_rp_pf_lds_t must be 21,248 B (3 WG/CU)");

/* Wave-Interleaved IQ4_NL flat block index for row r at 4-block step kb0 (nb % 4 == 0).
 * Guaranteed 4-block contiguous (64 B qw + 8 B qs) for both full 8-block steps & rem==4 tail! */
static inline __attribute__((always_inline)) u32 ink_r13_rp_iq4_dblk0(
    const u32 r, const u32 nb, const u32 kb0) {
  const u32 rg = r >> 2u;
  const u32 rk = r & 3u;
  const u32 nfull8 = nb & ~7u;
  if (kb0 < nfull8) {
    return ((rg * nb + (kb0 & ~7u)) << 2u) + (rk << 3u) + (kb0 & 4u);
  }
  const u32 rem = nb & 7u;
  return ((rg * nb + nfull8) << 2u) + rk * rem;
}

static inline __attribute__((always_inline)) void ink_r13_rp_pf_stage_sw(
    LDSP f16 *lsw, const u32 t0, const int is_sw, const ink_r13_rp_u32x2 rsw) {
  if (is_sw) {
    u32 t_loc = t0;
    __asm__ volatile("" : "+v"(t_loc));
    const u32 w0 = rsw.x, w1 = rsw.y;
    const u32 p0 = (u32)__builtin_amdgcn_update_dpp(0, (int)w0, 0xB1, 0xF, 0xF, true);
    const u32 p1 = (u32)__builtin_amdgcn_update_dpp(0, (int)w1, 0xB1, 0xF, 0xF, true);
    const u32 odd = t_loc & 1u;
    const u32 sel = odd ? 0x03020706u : 0x05040100u;
    const u32 v_a = __builtin_amdgcn_perm(p0, w0, sel);
    const u32 v_b = __builtin_amdgcn_perm(p1, w1, sel);
    const u32 r2 = t_loc & ~1u;
    LDSP u32 *sw32 = (LDSP u32 *)lsw;
    const u32 str32 = INK_R13_RP_PF_SW_STR >> 1u;
    sw32[(0u + odd) * str32 + (r2 >> 1u)] = v_a;
    sw32[(2u + odd) * str32 + (r2 >> 1u)] = v_b;
  }
}

static inline __attribute__((always_inline)) void ink_r13_rp_pf_stage_scales(
    LDSP f16 *lsw, LDSP float *lsx, const u32 t0, const u32 sx_stride,
    const int is_sw, const int is_sx, const ink_r13_rp_u32x2 rsw, const f32x4 rsx) {
  ink_r13_rp_pf_stage_sw(lsw, t0, is_sw, rsw);
  if (is_sx) {
    u32 t_loc = t0;
    __asm__ volatile("" : "+v"(t_loc));
    lsx[0u * sx_stride + t_loc] = rsx.x;
    lsx[1u * sx_stride + t_loc] = rsx.y;
    lsx[2u * sx_stride + t_loc] = rsx.z;
    lsx[3u * sx_stride + t_loc] = rsx.w;
  }
}

static inline __attribute__((always_inline)) f8 ink_r13_rp_pf_load_sw_f8(
    const LDSP f16 *lsw, const u32 k, const u32 wr, const u32 h) {
  const h8 sw_h = *(const LDSP h8 *)(lsw + k * INK_R13_RP_PF_SW_STR + wr + 8u * h);
  f8 sw = __builtin_convertvector(sw_h, f8);
  __asm__ volatile("" : "+v"(sw));
  return sw;
}

static inline __attribute__((always_inline)) void ink_r13_rp_pf_decode_w(
    const u32x4 wk, const u32 h, v2i *a0_out, v2i *a1_out) {
  u32x4 nb4 = h ? ((wk >> 4u) & 0x0F0F0F0Fu) : (wk & 0x0F0F0F0Fu);
  nb4.x = ie_iq4_map(nb4.x);
  nb4.y = ie_iq4_map(nb4.y);
  nb4.z = ie_iq4_map(nb4.z);
  nb4.w = ie_iq4_map(nb4.w);
  *a0_out = (v2i){(int)nb4.x, (int)nb4.y};
  *a1_out = (v2i){(int)nb4.z, (int)nb4.w};
}

static inline __attribute__((always_inline)) void ink_r13_rp_pf_k_step(
    const u32x4 wk, const LDSP u8 *lx, const LDSP f16 *lsw, const LDSP float *lsx,
    const u32 k, const u32 wr, const u32 l16, const u32 h, const u32 lx_base0,
    const u32 wj, const u32 sx_str, f8 *acc) {
  const v8i zm = (v8i)(0);
  u32 lxb = lx_base0;
  __asm__ volatile("" : "+v"(lxb));
  const LDSP u8 *lx_k = lx + (lxb ^ (k << 5u));
  const LDSP float *lsx_k = lsx + k * sx_str + l16;

  v2i a0, a1;
  ink_r13_rp_pf_decode_w(wk, h, &a0, &a1);
  const f8 sw = ink_r13_rp_pf_load_sw_f8(lsw, k, wr, h);

  const u32x4 b0 = *(const LDSP u32x4 *)(lx_k + 0u * 2048u);
  const float dx0 = lsx_k[0u];
  if (wj == 1u) {
    __builtin_amdgcn_sched_barrier(0);
    const v2i b0_lo = {(int)b0.x, (int)b0.y}, b0_hi = {(int)b0.z, (int)b0.w};
    v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_lo, zm, 0);
    c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_hi, c0, 0);
    __builtin_amdgcn_sched_barrier(0);
    acc[0] += (sw * dx0) * __builtin_convertvector(c0, f8);
    __asm__ volatile("" : : "v"(acc[0]) : "memory");
    __builtin_amdgcn_sched_barrier(0);
    return;
  }

  const u32x4 b1 = *(const LDSP u32x4 *)(lx_k + 1u * 2048u);
  const float dx1 = lsx_k[16u];
  __builtin_amdgcn_sched_barrier(0);
  const v2i b0_lo = {(int)b0.x, (int)b0.y}, b0_hi = {(int)b0.z, (int)b0.w};
  const v2i b1_lo = {(int)b1.x, (int)b1.y}, b1_hi = {(int)b1.z, (int)b1.w};
  v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0_lo, zm, 0);
  v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b1_lo, zm, 0);
  c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b0_hi, c0, 0);
  c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1_hi, c1, 0);
  __builtin_amdgcn_sched_barrier(0);
  acc[0] += (sw * dx0) * __builtin_convertvector(c0, f8);
  __asm__ volatile("" : : "v"(acc[0]));
  __builtin_amdgcn_sched_barrier(0);
  acc[1] += (sw * dx1) * __builtin_convertvector(c1, f8);
  __asm__ volatile("" : : "v"(acc[1]) : "memory");
  __builtin_amdgcn_sched_barrier(0);
  if (wj == 2u) return;

  const u32x4 b2 = *(const LDSP u32x4 *)(lx_k + 2u * 2048u);
  const float dx2 = lsx_k[32u];
  if (wj == 3u) {
    __builtin_amdgcn_sched_barrier(0);
    const v2i b2_lo = {(int)b2.x, (int)b2.y}, b2_hi = {(int)b2.z, (int)b2.w};
    v8i c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b2_lo, zm, 0);
    c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b2_hi, c2, 0);
    __builtin_amdgcn_sched_barrier(0);
    acc[2] += (sw * dx2) * __builtin_convertvector(c2, f8);
    __asm__ volatile("" : : "v"(acc[2]) : "memory");
    __builtin_amdgcn_sched_barrier(0);
    return;
  }

  const u32x4 b3 = *(const LDSP u32x4 *)(lx_k + 3u * 2048u);
  const float dx3 = lsx_k[48u];
  __builtin_amdgcn_sched_barrier(0);
  const v2i b2_lo = {(int)b2.x, (int)b2.y}, b2_hi = {(int)b2.z, (int)b2.w};
  const v2i b3_lo = {(int)b3.x, (int)b3.y}, b3_hi = {(int)b3.z, (int)b3.w};
  v8i c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b2_lo, zm, 0);
  v8i c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b3_lo, zm, 0);
  c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b2_hi, c2, 0);
  c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b3_hi, c3, 0);
  __builtin_amdgcn_sched_barrier(0);
  acc[2] += (sw * dx2) * __builtin_convertvector(c2, f8);
  __asm__ volatile("" : : "v"(acc[2]));
  __builtin_amdgcn_sched_barrier(0);
  acc[3] += (sw * dx3) * __builtin_convertvector(c3, f8);
  __asm__ volatile("" : : "v"(acc[3]) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

static inline __attribute__((always_inline)) void ink_r13_rp_pf_run_wj(
    LDSP ink_r13_rp_pf_lds_t *lds, const G u8 *qw_e, const G u8 *qs_e,
    const G u8 *X_u, const G u8 *X_s, const G u32 *list,
    const u32 pb, const u32 nj, const u32 rbase, const u32 rows, const u32 nb,
    const u32 xs, const u32 xdiv, const u32 wj, f8 *acc) {
  u32 t0 = tid();
  __asm__ volatile("" : "+v"(t0));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = t0 & 31u, l16 = l & 15u, h = l >> 4u;
  const u32 wr = 16u * w;
  const int is_sw = (w < 4u);
  const u32 n_sx = wj << 4u;
  const int is_sx = (t0 < n_sx);
  const u32 sx_str = (wj > 2u) ? 64u : 32u;
  const u32 tk_ld = t0 >> 3u, c8_ld = t0 & 7u;
  const int ld_lo = (wj >= 2u) || (tk_ld < 16u);
  const int ld_hi = (wj == 4u) || (wj == 3u && tk_ld < 16u);

  if (is_sx) {
    const u32 pj = pb + (t0 < nj ? t0 : 0u);
    lds->pid[t0] = list ? list[pj] : pj;
  }
  lbarrier();

  const u32 rw_idx = rbase + wr + l16;
  const u32 rw_cl = rw_idx < rows ? rw_idx : (rows - 1u);
  const u32 sr = rbase + (is_sw ? t0 : 0u);
  const u32 sr_cl = sr < rows ? sr : (rows - 1u);
  const u32 st_x = tk_ld * 128u + ((c8_ld ^ (tk_ld & 7u)) << 4u);
  const u32 lx_base0 = l16 * 128u + ((h ^ (l16 & 7u)) << 4u);

  const u32 xoff0 = ld_lo ? (ink_r13_rp_div_x(lds->pid[tk_ld], xdiv) * xs + c8_ld * 16u) : 0u;
  const u32 xoff1 = ld_hi ? (ink_r13_rp_div_x(lds->pid[tk_ld + 32u], xdiv) * xs + c8_ld * 16u) : 0u;
  const u32 sxoff = is_sx ? (ink_r13_rp_div_x(lds->pid[t0], xdiv) * xs) : 0u;

#pragma unroll
  for (u32 j = 0u; j < wj; j++) acc[j] = (f8)(0.0f);

  u32x4 wa[INK_R13_RP_PF_KB];
  {
    const u32 db_w0 = ink_r13_rp_iq4_dblk0(rw_cl, nb, 0u);
    const G u8 *qw0_p = qw_e + (unsigned long)db_w0 * 16ul;
#pragma unroll
    for (u32 k = 0u; k < INK_R13_RP_PF_KB; k++) wa[k] = *(const G u32x4 *)(qw0_p + k * 16u);
    const u32x4 rx0_i = ld_lo ? *(const G u32x4 *)(X_u + xoff0) : (u32x4)(0u);
    const u32x4 rx1_i = ld_hi ? *(const G u32x4 *)(X_u + xoff1) : (u32x4)(0u);
    const u32 db_s0 = is_sw ? ink_r13_rp_iq4_dblk0(sr_cl, nb, 0u) : 0u;
    const ink_r13_rp_u32x2 rsw_i = is_sw ? ink_r13_rp_ld2(qs_e + (unsigned long)db_s0 * 2ul)
                                         : (ink_r13_rp_u32x2){0u, 0u};
    const f32x4 rsx_i = is_sx ? rq_load_f32x4((const G float *)(X_s + sxoff)) : (f32x4)(0.0f);
    if (ld_lo) *(LDSP u32x4 *)(lds->u.k.x[0] + st_x) = rx0_i;
    if (ld_hi) *(LDSP u32x4 *)(lds->u.k.x[0] + st_x + 4096u) = rx1_i;
    ink_r13_rp_pf_stage_scales(lds->u.k.sw[0], lds->u.k.sx[0], t0, sx_str, is_sw, is_sx, rsw_i, rsx_i);
  }
  lbarrier();

  u32 bf = 0u;
  u32x4 rx0 = (u32x4)(0u), rx1 = (u32x4)(0u);
#pragma unroll 1
  for (u32 kb0 = 0u; kb0 < nb; kb0 += INK_R13_RP_PF_KB) {
    const u32 kb_n = (u32)__builtin_amdgcn_readfirstlane((int)(kb0 + INK_R13_RP_PF_KB));
    const int more = (kb_n < nb);
    if (more) {
      const G u8 *X_n = rq_ptr_u(X_u + (unsigned long)kb_n * 32ul);
      if (ld_lo) rx0 = *(const G u32x4 *)(X_n + xoff0);
      if (ld_hi) rx1 = *(const G u32x4 *)(X_n + xoff1);
    }
    __builtin_amdgcn_sched_barrier(0);
    const LDSP u8 *lx = lds->u.k.x[bf];
    const LDSP f16 *lsw = lds->u.k.sw[bf];
    const LDSP float *lsx = lds->u.k.sx[bf];

    ink_r13_rp_pf_k_step(wa[0], lx, lsw, lsx, 0u, wr, l16, h, lx_base0, wj, sx_str, acc);
    ink_r13_rp_pf_k_step(wa[1], lx, lsw, lsx, 1u, wr, l16, h, lx_base0, wj, sx_str, acc);

    ink_r13_rp_u32x2 rsw = {0u, 0u};
    f32x4 rsx = (f32x4)(0.0f);
    if (more) {
      const u32 nbf = bf ^ 1u;
      if (ld_lo) *(LDSP u32x4 *)(lds->u.k.x[nbf] + st_x) = rx0;
      if (ld_hi) *(LDSP u32x4 *)(lds->u.k.x[nbf] + st_x + 4096u) = rx1;
      const u32 db_sn = is_sw ? ink_r13_rp_iq4_dblk0(sr_cl, nb, kb_n) : 0u;
      const G u8 *Xs_n = rq_ptr_u(X_s + (unsigned long)kb_n * 4ul);
      if (is_sw) rsw = ink_r13_rp_ld2(qs_e + (unsigned long)db_sn * 2ul);
      if (is_sx) rsx = rq_load_f32x4((const G float *)(Xs_n + sxoff));
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r13_rp_pf_k_step(wa[2], lx, lsw, lsx, 2u, wr, l16, h, lx_base0, wj, sx_str, acc);

    u32 db_wn = 0u;
    if (more) {
      const u32 nbf = bf ^ 1u;
      ink_r13_rp_pf_stage_scales(lds->u.k.sw[nbf], lds->u.k.sx[nbf], t0, sx_str, is_sw, is_sx, rsw, rsx);
      __builtin_amdgcn_sched_barrier(0);
      db_wn = ink_r13_rp_iq4_dblk0(rw_cl, nb, kb_n);
      const G u8 *qw_n = qw_e + (unsigned long)db_wn * 16ul;
      wa[0] = *(const G u32x4 *)(qw_n + 0u);
      wa[1] = *(const G u32x4 *)(qw_n + 16u);
    }
    __builtin_amdgcn_sched_barrier(0);
    ink_r13_rp_pf_k_step(wa[3], lx, lsw, lsx, 3u, wr, l16, h, lx_base0, wj, sx_str, acc);
    if (more) {
      const u32 nbf = bf ^ 1u;
      const G u8 *qw_n = qw_e + (unsigned long)db_wn * 16ul;
      wa[2] = *(const G u32x4 *)(qw_n + 32u);
      wa[3] = *(const G u32x4 *)(qw_n + 48u);
      lbarrier();
      bf = nbf;
    }
  }
}

static inline __attribute__((always_inline)) void ink_r13_rp_pf_store_strip(
    LDSP float *ep, const LDSP u32 *lpid, G u8 *Y, const G u8 *res,
    const u32 rbase, const u32 rows, const u32 ys, const u32 rs,
    const u32 strip_base, const u32 nj_strip, const u32 j_base, const f8 *acc) {
  u32 t_ep = tid();
  __asm__ volatile("" : "+v"(t_ep));
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t_ep >> 5u));
  const u32 l = t_ep & 31u, l16 = l & 15u, h = l >> 4u;
  const u32 r_wr = 16u * w + 8u * h;

#pragma unroll
  for (u32 j_loc = 0u; j_loc < 2u; j_loc++) {
    const u32 slot_loc = 16u * j_loc + l16;
    if (slot_loc < nj_strip) {
      LDSP float *dst = ep + slot_loc * INK_R13_RP_PF_EP_STR + r_wr;
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) dst[v] = acc[j_base + j_loc][v];
    }
  }
  lbarrier();

  const int vec_ok = (rbase + 128u <= rows) && ((ys & 15u) == 0u) && ((((unsigned long)Y) & 15ul) == 0ul) &&
                     (!res || (((rs & 15u) == 0u) && ((((unsigned long)res) & 15ul) == 0ul)));
  if (__builtin_expect(vec_ok, 1)) {
    for (u32 tk = w; tk < nj_strip; tk += 8u) {
      const u32 pid = (u32)__builtin_amdgcn_readfirstlane((int)lpid[strip_base + tk]);
      const LDSP float *src = ep + tk * INK_R13_RP_PF_EP_STR + 4u * l;
      f32x4 v = {src[0], src[1], src[2], src[3]};
      if (res) {
        const G f32x4 *rr4 = (const G f32x4 *)((const G float *)(res + (unsigned long)pid * rs) + rbase);
        v += rr4[l];
      }
      G f32x4 *yr4 = (G f32x4 *)((G float *)(Y + (unsigned long)pid * ys) + rbase);
      yr4[l] = v;
    }
  } else {
    for (u32 tk = w; tk < nj_strip; tk += 8u) {
      const u32 pid = (u32)__builtin_amdgcn_readfirstlane((int)lpid[strip_base + tk]);
      const LDSP float *src = ep + tk * INK_R13_RP_PF_EP_STR + 4u * l;
      G float *yr = (G float *)(Y + (unsigned long)pid * ys);
      const G float *rr = res ? (const G float *)(res + (unsigned long)pid * rs) : (const G float *)0;
#pragma unroll 1
      for (u32 c = 0u; c < 4u; c++) {
        const u32 r = rbase + 4u * l + c;
        if (r < rows) yr[r] = epilogue(src[c], r, 0, rr);
      }
    }
  }
}

KERNEL ie_inkling_gemm_iq4nl_wave_expmaj_r13(
    const G u32 *qw0, const G u16 *qs0, const G u8 *X, G u8 *Y,
    u32 rows, u32 nb, const G u32 *list, const G u32 *offs, u32 np,
    u32 qwn, u32 qsn, u32 xs, u32 ys, u32 xdiv, u32 kind,
    const G u8 *res, u32 rs) {
  static LDS ink_r13_rp_pf_lds_t ink_r13_rp_pf_lds;
  (void)kind;
  if (!qw0 || !qs0 || !X || !Y || rows == 0u || (rows & 3u) != 0u ||
      nb == 0u || (nb & 3u) != 0u || xdiv == 0u)
    return;

  const u32 e = list ? (u32)__builtin_amdgcn_readfirstlane((int)tokid()) : 0u;
  const u32 p0 = list ? (u32)__builtin_amdgcn_readfirstlane((int)offs[e])
                      : (u32)__builtin_amdgcn_readfirstlane((int)(tokid() * 64u));
  const u32 p1 = list ? (u32)__builtin_amdgcn_readfirstlane((int)offs[e + 1u])
                      : (p0 + 64u < np ? p0 + 64u : np);
  const u32 rbase = (u32)__builtin_amdgcn_readfirstlane((int)(wgid() * INK_R13_RP_PF_R));
  if (p0 >= p1 || rbase >= rows) return;

  const G u8 *qw_e = rq_ptr_u((const G u8 *)(qw0 + (unsigned long)e * qwn));
  const G u8 *qs_e = rq_ptr_u((const G u8 *)(qs0 + (unsigned long)e * qsn));
  const G u8 *X_u = rq_ptr_u(X);
  const G u8 *X_s = rq_ptr_u(X + 32ul * nb);

  u32 pb = p0;
#pragma unroll 1
  while (pb < p1 && p1 - pb > 48u) {
    const u32 pb_u = (u32)__builtin_amdgcn_readfirstlane((int)pb);
    const u32 rem = p1 - pb_u;
    const u32 nj = rem < 64u ? rem : 64u;
    f8 acc[4];
    ink_r13_rp_pf_run_wj(&ink_r13_rp_pf_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                         rbase, rows, nb, xs, xdiv, 4u, acc);
    lbarrier();
    ink_r13_rp_pf_store_strip(&ink_r13_rp_pf_lds.u.ep[0][0], ink_r13_rp_pf_lds.pid, Y, res,
                              rbase, rows, ys, rs, 0u, 32u, 0u, acc);
    lbarrier();
    ink_r13_rp_pf_store_strip(&ink_r13_rp_pf_lds.u.ep[0][0], ink_r13_rp_pf_lds.pid, Y, res,
                              rbase, rows, ys, rs, 32u, nj - 32u, 2u, acc);
    pb = pb_u + nj;
    if (pb < p1) lbarrier();
  }

  if (pb < p1) {
    const u32 pb_u = (u32)__builtin_amdgcn_readfirstlane((int)pb);
    const u32 nj = p1 - pb_u;
    f8 acc[3];
    if (nj > 32u) {
      ink_r13_rp_pf_run_wj(&ink_r13_rp_pf_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                           rbase, rows, nb, xs, xdiv, 3u, acc);
      lbarrier();
      ink_r13_rp_pf_store_strip(&ink_r13_rp_pf_lds.u.ep[0][0], ink_r13_rp_pf_lds.pid, Y, res,
                                rbase, rows, ys, rs, 0u, 32u, 0u, acc);
      lbarrier();
      ink_r13_rp_pf_store_strip(&ink_r13_rp_pf_lds.u.ep[0][0], ink_r13_rp_pf_lds.pid, Y, res,
                                rbase, rows, ys, rs, 32u, nj - 32u, 2u, acc);
    } else if (nj > 16u) {
      ink_r13_rp_pf_run_wj(&ink_r13_rp_pf_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                           rbase, rows, nb, xs, xdiv, 2u, acc);
      lbarrier();
      ink_r13_rp_pf_store_strip(&ink_r13_rp_pf_lds.u.ep[0][0], ink_r13_rp_pf_lds.pid, Y, res,
                                rbase, rows, ys, rs, 0u, nj, 0u, acc);
    } else {
      ink_r13_rp_pf_run_wj(&ink_r13_rp_pf_lds, qw_e, qs_e, X_u, X_s, list, pb_u, nj,
                           rbase, rows, nb, xs, xdiv, 1u, &acc[0]);
      lbarrier();
      ink_r13_rp_pf_store_strip(&ink_r13_rp_pf_lds.u.ep[0][0], ink_r13_rp_pf_lds.pid, Y, res,
                                rbase, rows, ys, rs, 0u, nj, 0u, acc);
    }
  }
}

/* ============================================================================
 * 11. H36b: ie_inkling_q4k_head_wave_cap_topk_r13
 *     Exact 19-arg drop-in for ie_inkling_q4k_head_cap_topk_r7 (bit 84) reading
 *     Wave-Interleaved Q4_K output.weight! Coalesced 512-B qw + 64-B sm + 16-B dd.
 *     Grid: (ng, 1u, 1u) [ng=128], NT=256, 8,292 B LDS, <= 96 VGPR (4 WG/CU).
 * ========================================================================== */

#define INK_R13_RP_FH_KMAX  32u
#define INK_R13_RP_FH_NGMAX 128u
#define INK_R13_RP_FH_CAP   1024u

typedef float ink_r13_rp_f8 __attribute__((ext_vector_type(8)));

static inline __attribute__((always_inline)) float ink_r13_rp_fh_cap(float v, float cap) {
  return (cap > 0.0f) ? (cap * ie_tanhf_s(v / cap)) : v;
}

static inline __attribute__((always_inline)) int ink_r13_rp_fh_better(float v, u32 i, float bv, u32 bi) {
  return (v > bv) || (v == bv && i < bi);
}

static inline __attribute__((always_inline)) u32 ink_r13_rp_fh_f2u(float f) {
  u32 u = __builtin_bit_cast(u32, f);
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static inline __attribute__((always_inline)) float ink_r13_rp_fh_u2f(u32 k) {
  const u32 u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  return __builtin_bit_cast(float, u);
}

static inline __attribute__((always_inline)) u32 ink_r13_rp_fh_cswap(u32 v, u32 ov, u32 hi, u32 asc) {
  return (hi ^ asc) ? (v < ov ? v : ov) : (v > ov ? v : ov);
}

static inline __attribute__((always_inline)) u32 ink_r13_rp_fh_wave_step(u32 v, u32 t, u32 stride, u32 asc) {
  const u32 ov = (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ stride) * 4u), (int)v);
  return ink_r13_rp_fh_cswap(v, ov, (u32)((t & stride) != 0u), asc);
}

static inline __attribute__((always_inline)) u32 ink_r13_rp_fh_lds_step(
    u32 v, u32 t, u32 stride, u32 asc, LDSP u32 *buf) {
  buf[t] = v;
  lbarrier();
  const u32 ov = buf[t ^ stride];
  return ink_r13_rp_fh_cswap(v, ov, (u32)((t & stride) != 0u), asc);
}

/* Wave-Interleaved 8-lane/row Q4_K dot product for ANY row r (contiguous 512 B/wave
 * when r is sequential 4-row-aligned, or contiguous 128 B/row for id_map gather!). */
static inline __attribute__((always_inline)) void ink_r13_rp_q4k_wave_row(
    const G u32 *qw, const G u16 *qs, const G u8 *xq,
    u32 r, u32 rows, u32 nb, u32 T, u32 xs, u32 sub,
    ink_r13_rp_f8 *out_v) {
#pragma clang fp contract(off)
  const u32 nsb = nb >> 3u;
  const u32 rg = r >> 2u;
  const u32 k = r & 3u;
  const u32 lane_idx = (k << 3u) + sub;
  const unsigned long strip0 = (unsigned long)rg * (unsigned long)nsb;
  const G u32 *w_dd = (const G u32 *)(qs + (unsigned long)rows * (unsigned long)nb);
  const u32 sub32 = sub << 5u;
  const u32 sub4 = sub << 2u;

  ink_r13_rp_f8 a0 = (ink_r13_rp_f8)(0.0f);
  ink_r13_rp_f8 a1 = (ink_r13_rp_f8)(0.0f);
  ink_r13_rp_f8 a2 = (ink_r13_rp_f8)(0.0f);
  ink_r13_rp_f8 a3 = (ink_r13_rp_f8)(0.0f);

#pragma clang loop unroll(disable)
  for (u32 sb = 0u; sb < nsb; sb++) {
    const u32 sb_u = (u32)__builtin_amdgcn_readfirstlane((int)sb);
    const u32 slot = (u32)__builtin_amdgcn_readfirstlane((int)(sb_u & 3u));
    const unsigned long strip = strip0 + (unsigned long)sb_u;

    const u32x4 w4 = ink_r13_rp_nt4(qw + (strip * 32ul + (unsigned long)lane_idx) * 4ul);
    const u32 sm = (u32)ink_r13_rp_nt16(qs + strip * 32ul + (unsigned long)lane_idx);
    const u32 dd_raw = ink_r13_rp_nt32(w_dd + strip * 4ul + (unsigned long)k);
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

    const unsigned long aw_sb = (unsigned long)sb_u * 256ul;
    const unsigned long da_sb = (unsigned long)nb * 32ul + (unsigned long)sb_u * 32ul;
    const unsigned long as_sb = (unsigned long)nb * 36ul + (unsigned long)sb_u * 32ul;

#pragma unroll
    for (u32 j = 0u; j < INK_R13_RP_TMAX; j++) {
      if (j < T) {
        const G u8 *xj = xq + (unsigned long)j * (unsigned long)xs;
        const G u8 *xj_aw = xj + aw_sb;
        const u32x4 act0 = ink_r13_rp_rt4(xj_aw + sub32);
        const u32x4 act1 = ink_r13_rp_rt4(xj_aw + sub32 + 16u);
        const float da = *(const G float *)(xj + da_sb + sub4);
        const int as = (int)(*(const G u32 *)(xj + as_sb + sub4));
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
        if (slot == 0u)      a0[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a0[j]));
        else if (slot == 1u) a1[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a1[j]));
        else if (slot == 2u) a2[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a2[j]));
        else                 a3[j] = __builtin_fmaf(-mm, xa, __builtin_fmaf(prod, da, a3[j]));
        if ((j & 1u) == 1u) __builtin_amdgcn_sched_barrier(0);
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  }

  ink_r13_rp_f8 res = (ink_r13_rp_f8)(0.0f);
#pragma unroll
  for (u32 j = 0u; j < INK_R13_RP_TMAX; j++) {
    if (j < T) {
      res[j] = ink_r13_rp_oct_tree_f((a0[j] + a2[j]) + (a1[j] + a3[j]));
    }
  }
  *out_v = res;
}

typedef struct {
  union {
    struct {
      float kv[INK_R13_RP_TMAX][INK_R13_RP_FH_KMAX];
      u32   ki[INK_R13_RP_TMAX][INK_R13_RP_FH_KMAX];
      float min_v[INK_R13_RP_TMAX];
      u32   min_i[INK_R13_RP_TMAX];
      u32   min_pos[INK_R13_RP_TMAX];
      float ch_v[INK_R13_RP_TMAX][32u];
      u32   ch_i[32u];
      u32   any_pass[INK_R13_RP_TMAX];
    } st1;
    u32 sm[2u][NT];
    struct {
      float cv[INK_R13_RP_FH_CAP];
      u32   ci[INK_R13_RP_FH_CAP];
    } cand;
  } u;
  u32   wsum[2u][8u];
  float wv[8u];
  u32   wi[8u];
  u32   last;
} ink_r13_rp_fh_lds_t;

static LDS ink_r13_rp_fh_lds_t ink_r13_rp_fh_hlds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r13_rp_fh_lds_t) <= 16384u, "ink_r13_rp_fh_hlds must stay <= 16 KiB");

static inline __attribute__((always_inline)) u32 ink_r13_rp_fh_bitonic_kth_256(u32 v, u32 K) {
  const u32 t = tid();
#pragma clang loop unroll(disable)
  for (u32 size = 2u; size <= 32u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride > 0u; stride >>= 1u) {
      v = ink_r13_rp_fh_wave_step(v, t, stride, a);
    }
  }
  u32 buf_id = 0u;
#pragma clang loop unroll(disable)
  for (u32 size = 64u; size <= 256u; size <<= 1u) {
    const u32 a = (u32)((t & size) != 0u);
#pragma clang loop unroll(disable)
    for (u32 stride = size >> 1u; stride >= 32u; stride >>= 1u) {
      v = ink_r13_rp_fh_lds_step(v, t, stride, a, ink_r13_rp_fh_hlds.u.sm[buf_id]);
      buf_id ^= 1u;
    }
#pragma clang loop unroll(disable)
    for (u32 stride = 16u; stride > 0u; stride >>= 1u) {
      v = ink_r13_rp_fh_wave_step(v, t, stride, a);
    }
  }
  if (t == K - 1u) ink_r13_rp_fh_hlds.u.sm[0u][0u] = v;
  lbarrier();
  return ink_r13_rp_fh_hlds.u.sm[0u][0u];
}

static inline __attribute__((always_inline)) u32 ink_r13_rp_fh_wg_scan_u32(
    u32 v, LDSP u32 *wsum, u32 *total_out) {
  const u32 l = lane();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  u32 x = v;
#pragma clang loop unroll(disable)
  for (u32 d = 1u; d < 32u; d <<= 1u) {
    const u32 src = (l >= d) ? (l - d) : 0u;
    const u32 ov = (u32)__builtin_amdgcn_ds_bpermute((int)(src * 4u), (int)x);
    if (l >= d) x += ov;
  }
  if (l == 31u) wsum[w] = x;
  lbarrier();
  u32 wb = 0u, tot = 0u;
#pragma clang loop unroll(disable)
  for (u32 j = 0u; j < 8u; j++) {
    const u32 wv = wsum[j];
    if (j < w) wb += wv;
    tot += wv;
  }
  *total_out = tot;
  return wb + (x - v);
}

static inline __attribute__((always_inline)) void ink_r13_rp_fh_classify16(
    const float rv[16u], u32 vmask, u32 tmax_k, u32 tau_k, float tau,
    u32 *gt_mask_out, u32 *eq_mask_out) {
  u32 gm = 0u, em = 0u;
  if (tau_k == 0u) {
    gm = vmask;
  } else if (tmax_k == tau_k) {
#pragma unroll
    for (u32 k = 0u; k < 16u; k++) {
      if ((vmask & (1u << k)) && rv[k] == tau) em |= (1u << k);
    }
  } else if (tmax_k > tau_k) {
#pragma unroll
    for (u32 k = 0u; k < 16u; k++) {
      if (vmask & (1u << k)) {
        const float v = rv[k];
        if (v > tau) gm |= (1u << k);
        else if (v == tau) em |= (1u << k);
      }
    }
  }
  *gt_mask_out = gm;
  *eq_mask_out = em;
}

static inline __attribute__((always_inline)) void ink_r13_rp_fh_rank_write(
    u32 c, u32 K, G float *out_v, G u32 *out_i) {
  for (u32 j = tid(); j < c; j += NT) {
    const float a = ink_r13_rp_fh_hlds.u.cand.cv[j];
    const u32 ai = ink_r13_rp_fh_hlds.u.cand.ci[j];
    u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
    for (u32 u = 0u; u < c; u++) {
      const float b = ink_r13_rp_fh_hlds.u.cand.cv[u];
      rank += (u32)((b > a) | ((b == a) & (ink_r13_rp_fh_hlds.u.cand.ci[u] < ai)));
    }
    if (rank < K) {
      out_v[rank] = a;
      out_i[rank] = ai;
    }
  }
}

KERNEL ie_inkling_q4k_head_wave_cap_topk_r13(
    const G u32 *qw, const G u16 *qs, const G u8 *xq,
    const G float *bias,
    u32 rows, u32 unpadded, u32 nb, u32 T, u32 xs,
    float cap, u32 K, u32 ng,
    G float *part_v, G u32 *part_i, G u32 *done,
    G u32 *out_am, u32 am_stride,
    G float *ov, G u32 *oi) {
  const u32 g = wgid();
  const u32 t = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 row_in_wg = (w << 2u) + (l >> 3u);
  const u32 nv = (unpadded > 0u && unpadded < rows) ? unpadded : rows;

  if (g >= ng || ng == 0u || ng > INK_R13_RP_FH_NGMAX || T == 0u || T > INK_R13_RP_TMAX ||
      K == 0u || K > INK_R13_RP_FH_KMAX || rows == 0u || (rows & 3u) != 0u ||
      nb == 0u || (nb & 7u) != 0u || nv == 0u || !qw || !qs || !xq)
    return;

  if (t < T * K) {
    const u32 tj = t / K;
    const u32 tk = t - tj * K;
    ink_r13_rp_fh_hlds.u.st1.kv[tj][tk] = -__builtin_inff();
    ink_r13_rp_fh_hlds.u.st1.ki[tj][tk] = 0xFFFFFFFFu;
  }
  if (t < T) {
    ink_r13_rp_fh_hlds.u.st1.min_v[t] = -__builtin_inff();
    ink_r13_rp_fh_hlds.u.st1.min_i[t] = 0xFFFFFFFFu;
    ink_r13_rp_fh_hlds.u.st1.min_pos[t] = 0u;
    ink_r13_rp_fh_hlds.u.st1.any_pass[t] = 0u;
  }
  lbarrier();

  for (u32 rbase = g << 5u; rbase < nv; rbase += (ng << 5u)) {
    const u32 r = rbase + row_in_wg;
    const u32 r_valid = (r < nv) ? 1u : 0u;
    ink_r13_rp_f8 vals = (ink_r13_rp_f8)(-__builtin_inff());
    if (r_valid) {
      ink_r13_rp_q4k_wave_row(qw, qs, xq, r, rows, nb, T, xs, sub, &vals);
    }

    if (sub == 0u) {
      ink_r13_rp_fh_hlds.u.st1.ch_i[row_in_wg] = r_valid ? r : 0xFFFFFFFFu;
      const float bv_bias = (r_valid && bias) ? bias[r] : 0.0f;
#pragma unroll
      for (u32 j = 0u; j < INK_R13_RP_TMAX; j++) {
        if (j < T) {
          float v = r_valid ? ink_r13_rp_fh_cap(vals[j] + bv_bias, cap) : -__builtin_inff();
          ink_r13_rp_fh_hlds.u.st1.ch_v[j][row_in_wg] = v;
          if (r_valid && ink_r13_rp_fh_better(v, r, ink_r13_rp_fh_hlds.u.st1.min_v[j], ink_r13_rp_fh_hlds.u.st1.min_i[j])) {
            ink_r13_rp_fh_hlds.u.st1.any_pass[j] = 1u;
          }
        }
      }
    }
    lbarrier();

    if (w < T && ink_r13_rp_fh_hlds.u.st1.any_pass[w] != 0u) {
      if (l == 0u) {
        float cur_mv = ink_r13_rp_fh_hlds.u.st1.min_v[w];
        u32 cur_mi = ink_r13_rp_fh_hlds.u.st1.min_i[w];
        u32 cur_mp = ink_r13_rp_fh_hlds.u.st1.min_pos[w];
#pragma clang loop unroll(disable)
        for (u32 c = 0u; c < 32u; c++) {
          const u32 ci = ink_r13_rp_fh_hlds.u.st1.ch_i[c];
          if (ci == 0xFFFFFFFFu) continue;
          const float cv = ink_r13_rp_fh_hlds.u.st1.ch_v[w][c];
          if (ink_r13_rp_fh_better(cv, ci, cur_mv, cur_mi)) {
            ink_r13_rp_fh_hlds.u.st1.kv[w][cur_mp] = cv;
            ink_r13_rp_fh_hlds.u.st1.ki[w][cur_mp] = ci;
            if (K == 1u) {
              cur_mv = cv;
              cur_mi = ci;
            } else {
              float wv_min = ink_r13_rp_fh_hlds.u.st1.kv[w][0u];
              u32 wi_min = ink_r13_rp_fh_hlds.u.st1.ki[w][0u];
              u32 wp_min = 0u;
#pragma clang loop unroll(disable)
              for (u32 p = 1u; p < K; p++) {
                const float pv = ink_r13_rp_fh_hlds.u.st1.kv[w][p];
                const u32 pi = ink_r13_rp_fh_hlds.u.st1.ki[w][p];
                if (pi == 0xFFFFFFFFu) {
                  if (wi_min != 0xFFFFFFFFu) { wv_min = pv; wi_min = pi; wp_min = p; }
                } else if (wi_min != 0xFFFFFFFFu && (pv < wv_min || (pv == wv_min && pi > wi_min))) {
                  wv_min = pv; wi_min = pi; wp_min = p;
                }
              }
              cur_mv = wv_min; cur_mi = wi_min; cur_mp = wp_min;
            }
          }
        }
        ink_r13_rp_fh_hlds.u.st1.min_v[w] = cur_mv;
        ink_r13_rp_fh_hlds.u.st1.min_i[w] = cur_mi;
        ink_r13_rp_fh_hlds.u.st1.min_pos[w] = cur_mp;
        ink_r13_rp_fh_hlds.u.st1.any_pass[w] = 0u;
      }
    }
    lbarrier();
  }

  const u32 tk_total = T * K;
  for (u32 idx = t; idx < tk_total; idx += NT) {
    const u32 tj = idx / K;
    const u32 tk = idx - tj * K;
    const unsigned long dst = (((unsigned long)tj * (unsigned long)ng + (unsigned long)g) * (unsigned long)K) + tk;
    part_v[dst] = ink_r13_rp_fh_hlds.u.st1.kv[tj][tk];
    part_i[dst] = ink_r13_rp_fh_hlds.u.st1.ki[tj][tk];
  }

  if (ng > 1u) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    barrier();
    if (t == 0u) {
      ink_r13_rp_fh_hlds.last =
          (__scoped_atomic_fetch_add(done, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == ng - 1u) ? 1u : 0u;
    }
    barrier();
    if (!ink_r13_rp_fh_hlds.last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if (t == 0u) *done = 0u;
  } else {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  }

  const u32 nc = ng * K;
  for (u32 tj = 0u; tj < T; tj++) {
    const G float *cv_row = part_v + (unsigned long)tj * (unsigned long)nc;
    const G u32   *ci_row = part_i + (unsigned long)tj * (unsigned long)nc;

    if (K == 1u) {
      float bv = -__builtin_inff();
      u32 bi = 0xFFFFFFFFu;
      for (u32 u = t; u < ng; u += NT) {
        const float vu = cv_row[u];
        const u32 iu = ci_row[u];
        if (ink_r13_rp_fh_better(vu, iu, bv, bi)) { bv = vu; bi = iu; }
      }
#pragma unroll
      for (u32 m = 16u; m > 0u; m >>= 1u) {
        const float ov_m = shfl_xor_f(bv, m);
        const u32 oi_m = (u32)__builtin_amdgcn_ds_bpermute((int)((l ^ m) * 4u), (int)bi);
        if (ink_r13_rp_fh_better(ov_m, oi_m, bv, bi)) { bv = ov_m; bi = oi_m; }
      }
      if (l == 0u) { ink_r13_rp_fh_hlds.wv[w] = bv; ink_r13_rp_fh_hlds.wi[w] = bi; }
      lbarrier();
      if (t == 0u) {
        float fv = ink_r13_rp_fh_hlds.wv[0u];
        u32 fi = ink_r13_rp_fh_hlds.wi[0u];
#pragma clang loop unroll(disable)
        for (u32 j = 1u; j < 8u; j++) {
          if (ink_r13_rp_fh_better(ink_r13_rp_fh_hlds.wv[j], ink_r13_rp_fh_hlds.wi[j], fv, fi)) {
            fv = ink_r13_rp_fh_hlds.wv[j]; fi = ink_r13_rp_fh_hlds.wi[j];
          }
        }
        if (out_am) {
          G u32 *am_r = (G u32 *)((G u8 *)out_am + (unsigned long)tj * (unsigned long)am_stride);
          am_r[0u] = fi;
          am_r[1u] = 0u;
        }
        if (ov) ov[tj] = fv;
        if (oi) oi[tj] = fi;
      }
      lbarrier();
      continue;
    }

    G float *ov_row = ov + (unsigned long)tj * (unsigned long)K;
    G u32   *oi_row = oi + (unsigned long)tj * (unsigned long)K;

    if (nc <= NT) {
      float mv = -__builtin_inff();
      u32 mi = 0xFFFFFFFFu;
      if (t < nc) {
        mv = cv_row[t];
        mi = ci_row[t];
        ink_r13_rp_fh_hlds.u.cand.cv[t] = mv;
        ink_r13_rp_fh_hlds.u.cand.ci[t] = mi;
      }
      const u32 is_val = (t < nc && mi != 0xFFFFFFFFu) ? 1u : 0u;
      const u32 wcnt = (u32)__builtin_popcount(__builtin_amdgcn_ballot_w32(is_val != 0u));
      if (l == 0u) ink_r13_rp_fh_hlds.wsum[0u][w] = wcnt;
      lbarrier();
      u32 nvalid = 0u;
#pragma clang loop unroll(disable)
      for (u32 j = 0u; j < 8u; j++) nvalid += ink_r13_rp_fh_hlds.wsum[0u][j];
      if (is_val) {
        u32 rank = 0u;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
        for (u32 u = 0u; u < nc; u++) {
          const u32 ui = ink_r13_rp_fh_hlds.u.cand.ci[u];
          if (ui != 0xFFFFFFFFu) {
            const float b = ink_r13_rp_fh_hlds.u.cand.cv[u];
            rank += (u32)((b > mv) | ((b == mv) & (ui < mi)));
          }
        }
        if (rank < K) { ov_row[rank] = mv; oi_row[rank] = mi; }
      }
      if (nvalid < K && t < K - nvalid) {
        ov_row[nvalid + t] = -__builtin_inff();
        oi_row[nvalid + t] = 0xFFFFFFFFu;
      }
      lbarrier();
    } else {
      float mv[16u];
      u32 vm2 = 0u;
      float tm2 = -__builtin_inff();
#pragma unroll
      for (u32 k = 0u; k < 16u; k++) {
        const u32 p = t + k * NT;
        if (p < nc) {
          const u32 idx = ci_row[p];
          const float v = cv_row[p];
          mv[k] = v;
          if (idx != 0xFFFFFFFFu) {
            vm2 |= (1u << k);
            if (v > tm2) tm2 = v;
          }
        } else {
          mv[k] = -__builtin_inff();
        }
      }
      const u32 tm2_k = vm2 ? ink_r13_rp_fh_f2u(tm2) : 0u;
      const u32 tau2_k = ink_r13_rp_fh_bitonic_kth_256(tm2_k, K);
      const float tau2 = tau2_k ? ink_r13_rp_fh_u2f(tau2_k) : -__builtin_inff();

      u32 gm2 = 0u, em2 = 0u;
      ink_r13_rp_fh_classify16(mv, vm2, tm2_k, tau2_k, tau2, &gm2, &em2);
      const u32 my_gt2 = (u32)__builtin_popcount(gm2);
      const u32 my_eq2 = (u32)__builtin_popcount(em2);
      u32 ptot2 = 0u;
      const u32 poff2 = ink_r13_rp_fh_wg_scan_u32((my_eq2 << 16u) | my_gt2, ink_r13_rp_fh_hlds.wsum[1u], &ptot2);
      const u32 bgt2 = poff2 & 0xFFFFu;
      const u32 beq2 = poff2 >> 16u;
      const u32 tgt2 = ptot2 & 0xFFFFu;
      const u32 teq2 = ptot2 >> 16u;

      if (gm2) {
        u32 pos = bgt2;
#pragma unroll
        for (u32 k = 0u; k < 16u; k++) {
          if (gm2 & (1u << k)) {
            ink_r13_rp_fh_hlds.u.cand.cv[pos] = mv[k];
            ink_r13_rp_fh_hlds.u.cand.ci[pos] = ci_row[t + k * NT];
            pos++;
          }
        }
      }

      const u32 need2 = (tgt2 < K) ? (K - tgt2) : 0u;
      u32 c2 = tgt2;
      if (need2 > 0u && teq2 <= need2) {
        if (em2) {
          u32 pos = tgt2 + beq2;
#pragma unroll
          for (u32 k = 0u; k < 16u; k++) {
            if (em2 & (1u << k)) {
              ink_r13_rp_fh_hlds.u.cand.cv[pos] = mv[k];
              ink_r13_rp_fh_hlds.u.cand.ci[pos] = ci_row[t + k * NT];
              pos++;
            }
          }
        }
        c2 = tgt2 + teq2;
      }
      lbarrier();

      if (need2 > 0u && teq2 > need2) {
        u32 taken2 = 0u;
#pragma clang loop unroll(disable)
        for (u32 k = 0u; k < 16u; k++) {
          if (taken2 < need2) {
            const u32 hit = (em2 >> k) & 1u;
            u32 stot = 0u;
            const u32 soff = ink_r13_rp_fh_wg_scan_u32(hit, ink_r13_rp_fh_hlds.wsum[k & 1u], &stot);
            if (hit && taken2 + soff < need2) {
              const u32 p = t + k * NT;
              const u32 rk = tgt2 + taken2 + soff;
              ov_row[rk] = cv_row[p];
              oi_row[rk] = ci_row[p];
            }
            taken2 += stot;
          }
        }
      }
      ink_r13_rp_fh_rank_write(c2, K, ov_row, oi_row);
      if (tau2_k == 0u && c2 < K && t < K - c2) {
        ov_row[c2 + t] = -__builtin_inff();
        oi_row[c2 + t] = 0xFFFFFFFFu;
      }
      lbarrier();
    }
  }
}

/* ============================================================================
 * 12. H36b: ie_inkling_ds_base_q4k_wave_idmap_r13
 *     Exact 11-arg drop-in for ie_inkling_ds_base_q4k_idmap_r7 (bit 88) reading
 *     Wave-Interleaved Q4_K output.weight via id_map[s]!
 *     Grid: ((n_act + 31u) / 32u, 1u, 1u), NT=256, 0 B LDS, <= 80 VGPR (4 WG/CU).
 * ========================================================================== */

KERNEL ie_inkling_ds_base_q4k_wave_idmap_r13(
    const G u32 *qw, const G u16 *qs, const G u8 *xq,
    const G u32 *id_map, G float *base_out,
    u32 full_rows, u32 n_act, u32 nb, u32 T, u32 xs, u32 ys_f32) {
  if (T == 0u || T > INK_R13_RP_TMAX || n_act == 0u || full_rows == 0u ||
      (full_rows & 3u) != 0u || nb == 0u || (nb & 7u) != 0u ||
      !qw || !qs || !xq || !base_out)
    return;

  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(tid() >> 5u));
  const u32 l = lane();
  const u32 sub = l & 7u;
  const u32 s = (wgid() << 5u) + (w << 2u) + (l >> 3u);
  if (s >= n_act) return;

  const u32 tok = id_map ? id_map[s] : s;
  if (tok >= full_rows) return;

  ink_r13_rp_f8 vals;
  ink_r13_rp_q4k_wave_row(qw, qs, xq, tok, full_rows, nb, T, xs, sub, &vals);
  if (sub == 0u) {
#pragma unroll
    for (u32 j = 0u; j < INK_R13_RP_TMAX; j++) {
      if (j < T) base_out[(unsigned long)j * (unsigned long)ys_f32 + (unsigned long)s] = vals[j];
    }
  }
}

#undef INK_R13_RP_FH_CAP
#undef INK_R13_RP_FH_NGMAX
#undef INK_R13_RP_FH_KMAX
#undef INK_R13_RP_PF_EP_STR
#undef INK_R13_RP_PF_SW_STR
#undef INK_R13_RP_PF_R
#undef INK_R13_RP_PF_KB
#undef INK_R13_RP_ACC_FMA
#undef INK_R13_RP_RBLF16F
#undef INK_R13_RP_RBLF32
#undef INK_R13_RP_RBL16
#undef INK_R13_RP_RBL32
#undef INK_R13_RP_RBL128
#undef INK_R13_RP_Q8_STRIP
#undef INK_R13_RP_TMAX
#undef INK_R13_RP_BUF_FLAGS

#endif /* !IE_MEGA */
