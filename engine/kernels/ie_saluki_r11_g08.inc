/* ============================================================================
 * Round 11 Group 8: Bitwise-exact T=8 verify-decode Q2_K half-superblock GEMV
 *   Target: RX 9070 XT (gfx1201, RDNA 4, Wave32, 256 threads/WG, 32 rows/WG)
 *   Replaces ie_saluki_q2k_tr8_r4 (195 VGPR / 3 WGs/WGP / 24 VMEM X-loads /
 *   16 VMEM W-loads / 320 v_dot4 per tile; 566.4 ms across Q2_K:17408x5120
 *   and Q2_K:12288x5120 in R10-final trace).
 *
 *   Key SOTA design (`ie_saluki_q2k_tr9_r11`):
 *     1. Half-superblock-shared wave mapping (v = 4*w + j, j = 0..3):
 *        In Q2_K (84 B/SB = 16 B scales + 64 B qs + 4 B d/mn), subblocks
 *        s=0..3 share qs[0..31] + d/mn and s=4..7 share qs[32..63] + d/mn.
 *        Each wave w loads ONE half-superblock ONCE for all 4 subblocks
 *        (1x b32 d_mn + 1x b64 scales + 2x b128 qs = 44 B = 1.0 global_load
 *        per subblock, 1.05x L0 amplification vs 3.62x in naive tr9).
 *     2. Direct pre-shifted 2-bit unpacking ((q >> (2*j)) & 0x03030303u):
 *        Eliminates all 16 (int)id >> sh arithmetic shifts per subblock
 *        (bitwise exact: |idots| <= 16*192*128 = 393,216 << 2^31) without
 *        retaining extra shifted VGPR copies across j.
 *     3. Low-VGPR 1-tile software pipelining at j=3:
 *        d_c, mn_c, sc_c, q0_c, q1_c (12 VGPRs) die before issuing next
 *        tile's 4 global_loads (11 VGPRs) -> peak ~68-74 VGPRs (<= 96,
 *        0 spills) hiding full GDDR6 latency behind j=3 compute.
 *     4. 4-batch (2-token) wave-uniform scalar activation loads + Q8_0 asum:
 *        Hoisted xt[8], per-subblock opaque uniform offsets (r11_g08_opaque_u32:
 *        blocks LLVM LoadStoreVectorizer from merging contiguous j=0..3 loads
 *        into SGPR-spilling s_load_b512 / dav s_load_b128 while preserving
 *        per-(j,t) s_load_b256 + K$ caching), and 2-token batches (20 SGPRs)
 *        with sched_barrier(0) keep SGPRs ~62-68 <= 80 (0 SGPR spills) and
 *        halve xs v_dot4 work via xs1 = asum - xs0 (exact in Z/2^32Z).
 *     5. Exact 5-level wave_tree_f binary reduction in 8,192 B LDS:
 *        4 unrolled 2-token phases, 0 bank conflicts, 0 dynamic indexing
 *        (priv == 0), exact 8 WGs/WGP (16 waves/SIMD = 100% occupancy).
 *
 * Included at the end of kernels/ie_saluki_kernels.c (after Round 10).
 * All local helpers/types are prefixed r11_g08_; zero local macros defined.
 * Bitwise equal (0 bit diffs) to base ie_gemv_raw_q2k_tr8 & ie_saluki_q2k_tr8_r4.
 * ========================================================================== */

typedef u32 r11_g08_u32x2 __attribute__((ext_vector_type(2)));

/* 4-byte-aligned vector weight load helpers (Q2_K stride is 84 = 4 * 21 B, so
 * superblock offsets 0, 8, 16, 32, 48, 64, 80 are always 4-byte aligned). */
struct __attribute__((packed, aligned(4))) r11_g08_u32x2_p4 { r11_g08_u32x2 v; };
struct __attribute__((packed, aligned(4))) r11_g08_u32x4_p4 { u32x4 v; };

static inline __attribute__((always_inline)) r11_g08_u32x2 r11_g08_ld_u32x2_a4(const G u8 *p, const u32 o) {
  return ((const G struct r11_g08_u32x2_p4 *)(p + o))->v;
}

static inline __attribute__((always_inline)) u32x4 r11_g08_ld_u32x4_a4(const G u8 *p, const u32 o) {
  return ((const G struct r11_g08_u32x4_p4 *)(p + o))->v;
}

/* Break LLVM IR LoadStoreVectorizer SCEV chains across contiguous subblocks
 * j = 0..3 (preventing 160-SGPR s_load_b512 / dav s_load_b128 merges) while
 * keeping the offset 100% proven wave-uniform (isAlwaysUniform) for SMEM. */
static inline __attribute__((always_inline)) u32 r11_g08_opaque_u32(u32 v) {
  __asm__ volatile("" : "+s"(v));
  return (u32)__builtin_amdgcn_readfirstlane((int)v);
}

/* 8-token Q2_K subblock dot using 4x2-token scalar activation batches (20
 * SGPRs/batch -> ~62-68 total SGPRs, 0 spills) and Q8_0 asum (36*nb + 4*b). */
static inline __attribute__((always_inline)) void r11_g08_q2k_dot8_sload(
    const float dw0, const float dm0, const float dw1, const float dm1,
    const u32 q00, const u32 q01, const u32 q02, const u32 q03,
    const u32 q10, const u32 q11, const u32 q12, const u32 q13,
    const u32 one_v, const TR8_CSP u8 *const xt[8],
    const u32 xo, const u32 dao, const u32 aso, float aj[8]) {
#pragma clang fp contract(off)
  const u32 lxo = r11_g08_opaque_u32(xo);
  const u32 ldao = r11_g08_opaque_u32(dao);
  const u32 laso = r11_g08_opaque_u32(aso);
#pragma unroll
  for (u32 tb = 0; tb < 8u; tb += 2u) {
    u32x4 x0v[2], x1v[2];
    float dav_v[2];
    u32 as_v[2];
#pragma unroll
    for (u32 u = 0; u < 2u; u++) {
      const TR8_CSP u8 *xb = xt[tb + u];
      x0v[u] = *(const TR8_CSP u32x4 *)(xb + lxo);
      x1v[u] = *(const TR8_CSP u32x4 *)(xb + lxo + 16u);
      dav_v[u] = *(const TR8_CSP float *)(xb + ldao);
      as_v[u] = *(const TR8_CSP u32 *)(xb + laso);
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (u32 u = 0; u < 2u; u++) {
      u32 id0 = ie_dot4_us(0u, q00, x0v[u].x);
      id0 = ie_dot4_us(id0, q01, x0v[u].y);
      id0 = ie_dot4_us(id0, q02, x0v[u].z);
      id0 = ie_dot4_us(id0, q03, x0v[u].w);
      u32 xs0 = ie_dot4_ss(0u, one_v, x0v[u].x);
      xs0 = ie_dot4_ss(xs0, one_v, x0v[u].y);
      xs0 = ie_dot4_ss(xs0, one_v, x0v[u].z);
      xs0 = ie_dot4_ss(xs0, one_v, x0v[u].w);
      const float t0 = dav_v[u] * (dw0 * (float)(int)id0 - dm0 * (float)(int)xs0);

      u32 id1 = ie_dot4_us(0u, q10, x1v[u].x);
      id1 = ie_dot4_us(id1, q11, x1v[u].y);
      id1 = ie_dot4_us(id1, q12, x1v[u].z);
      id1 = ie_dot4_us(id1, q13, x1v[u].w);
      const u32 xs1 = as_v[u] - xs0;
      const float t1 = dav_v[u] * (dw1 * (float)(int)id1 - dm1 * (float)(int)xs1);
      aj[tb + u] += (t0 + t1);
    }
    __builtin_amdgcn_sched_barrier(0);
  }
}

/* Exact 5-level binary reduction matching wave_tree_f on v = 4*w + j across
 * 4 unrolled 2-token phases in 8,192 B LDS (0 bank conflicts, 8 WGs/WGP). */
static inline __attribute__((always_inline)) void r11_g08_q2k_reduce32(
    LDSP float *red, const u32 w, const u32 l, const u32 r0, const u32 rows,
    float a[4][8], const G float *bias, const G float *res, G float *y,
    const u32 ys, const u32 rs) {
#pragma clang fp contract(off)
  const u32 ti = tid();
  const u32 lt_r = ti >> 5u;
  const u32 rr = ti & 31u;
  const u32 row = r0 + rr;
#pragma unroll
  for (u32 p = 0; p < 4u; p++) {
#pragma unroll
    for (u32 lt = 0; lt < 2u; lt++) {
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
        red[((lt * 8u + w) * 4u + j) * 32u + l] = a[j][2u * p + lt];
      }
    }
    lbarrier();
    if (ti < 64u) {
      float x[4];
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
        const u32 jbase = j * 32u + rr;
        const float v0 = red[(lt_r * 8u + 0u) * 128u + jbase];
        const float v1 = red[(lt_r * 8u + 1u) * 128u + jbase];
        const float v2 = red[(lt_r * 8u + 2u) * 128u + jbase];
        const float v3 = red[(lt_r * 8u + 3u) * 128u + jbase];
        const float v4 = red[(lt_r * 8u + 4u) * 128u + jbase];
        const float v5 = red[(lt_r * 8u + 5u) * 128u + jbase];
        const float v6 = red[(lt_r * 8u + 6u) * 128u + jbase];
        const float v7 = red[(lt_r * 8u + 7u) * 128u + jbase];
        const float u0 = v0 + v4, u1 = v1 + v5, u2 = v2 + v6, u3 = v3 + v7;
        const float w0 = u0 + u2, w1 = u1 + u3;
        x[j] = w0 + w1;
      }
      float out = (x[0] + x[2]) + (x[1] + x[3]);
      if (row < rows) {
        const u32 tok = 2u * p + lt_r;
        if (bias) out = out + bias[row];
        if (res) out = ((const G float *)((const G u8 *)res + (unsigned long)tok * rs))[row] + out;
        ((G float *)((G u8 *)y + (unsigned long)tok * ys))[row] = out;
      }
    }
    if (p < 3u) lbarrier();
  }
}

KERNEL ie_saluki_q2k_tr9_r11(const G u8 *q, const G u8 *xq, G float *y, u32 rows, u32 nb, u32 T,
                             const G float *bias, const G float *res, const G float *nw, G float *ny,
                             G u8 *nq, float eps, G u32 *count, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off)
  (void)nw; (void)ny; (void)nq; (void)eps; (void)count;
  const u32 r0 = 32u * wgid();
  if (T != 8u || r0 >= rows) return;

  static LDS float red[2u * 8u * 4u * 32u] __attribute__((aligned(16)));

  const u32 w = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5u);
  const u32 l = lane();
  const u32 half = w & 1u;
  const u32 sb_off = 84u * (w >> 1u);
  const u32 r = r0 + l;
  const u32 mb = nb >> 3u;
  const G u8 *rowp = q + (unsigned long)(r < rows ? r : r0) * mb * 84u;

  const TR8_CSP u8 *xt[8];
#pragma unroll
  for (u32 t = 0; t < 8u; t++) xt[t] = (const TR8_CSP u8 *)tr8_uniform(xq + (unsigned long)t * xs);

  float a[4][8];
#pragma unroll
  for (u32 j = 0; j < 4u; j++)
#pragma unroll
    for (u32 t = 0; t < 8u; t++) a[j][t] = 0.0f;

  u32 one_v = 0x01010101u;
  __asm__("" : "+v"(one_v));

  const G u8 *bp_d = rowp + (sb_off + 80u);
  const G u8 *bp_sc = rowp + (sb_off + 8u * half);
  const G u8 *bp_q = rowp + (sb_off + 16u + 32u * half);
  u32 xo = 128u * w, dao = 32u * nb + 16u * w, aso = 36u * nb + 16u * w;
  const u32 nb32 = nb & ~31u;

  if (nb32 > 0u) {
    u32 d_mn_c = ie_rq_a32(bp_d, 0u);
    r11_g08_u32x2 sc_c = r11_g08_ld_u32x2_a4(bp_sc, 0u);
    u32x4 q0_c = r11_g08_ld_u32x4_a4(bp_q, 0u);
    u32x4 q1_c = r11_g08_ld_u32x4_a4(bp_q, 16u);
    __asm__ volatile("" : "+v"(d_mn_c), "+v"(sc_c), "+v"(q0_c), "+v"(q1_c));

    for (u32 k0 = 0; k0 + 32u < nb32; k0 += 32u) {
      const float d_c = ie_rq_h2f(d_mn_c & 0xFFFFu);
      const float mn_c = ie_rq_h2f(d_mn_c >> 16u);
#pragma unroll
      for (u32 j = 0; j < 3u; j++) {
        const u32 sh = 2u * j;
        const u32 sc16 = ((j < 2u ? sc_c.x : sc_c.y) >> (16u * (j & 1u))) & 0xFFFFu;
        const float dw0 = d_c * (float)(sc16 & 15u);
        const float dm0 = mn_c * (float)((sc16 >> 4u) & 15u);
        const float dw1 = d_c * (float)((sc16 >> 8u) & 15u);
        const float dm1 = mn_c * (float)((sc16 >> 12u) & 15u);
        const u32 q00 = (q0_c.x >> sh) & 0x03030303u, q01 = (q0_c.y >> sh) & 0x03030303u;
        const u32 q02 = (q0_c.z >> sh) & 0x03030303u, q03 = (q0_c.w >> sh) & 0x03030303u;
        const u32 q10 = (q1_c.x >> sh) & 0x03030303u, q11 = (q1_c.y >> sh) & 0x03030303u;
        const u32 q12 = (q1_c.z >> sh) & 0x03030303u, q13 = (q1_c.w >> sh) & 0x03030303u;
        r11_g08_q2k_dot8_sload(dw0, dm0, dw1, dm1, q00, q01, q02, q03, q10, q11, q12, q13,
                               one_v, xt, xo + 32u * j, dao + 4u * j, aso + 4u * j, a[j]);
      }
      {
        const u32 sc16 = sc_c.y >> 16u;
        const float dw0 = d_c * (float)(sc16 & 15u);
        const float dm0 = mn_c * (float)((sc16 >> 4u) & 15u);
        const float dw1 = d_c * (float)((sc16 >> 8u) & 15u);
        const float dm1 = mn_c * (float)((sc16 >> 12u) & 15u);
        const u32 q00 = (q0_c.x >> 6u) & 0x03030303u, q01 = (q0_c.y >> 6u) & 0x03030303u;
        const u32 q02 = (q0_c.z >> 6u) & 0x03030303u, q03 = (q0_c.w >> 6u) & 0x03030303u;
        const u32 q10 = (q1_c.x >> 6u) & 0x03030303u, q11 = (q1_c.y >> 6u) & 0x03030303u;
        const u32 q12 = (q1_c.z >> 6u) & 0x03030303u, q13 = (q1_c.w >> 6u) & 0x03030303u;
        const u32 d_mn_n = ie_rq_a32(bp_d, 336u);
        const r11_g08_u32x2 sc_n = r11_g08_ld_u32x2_a4(bp_sc, 336u);
        const u32x4 q0_n = r11_g08_ld_u32x4_a4(bp_q, 336u);
        const u32x4 q1_n = r11_g08_ld_u32x4_a4(bp_q, 352u);
        __builtin_amdgcn_sched_barrier(0);
        r11_g08_q2k_dot8_sload(dw0, dm0, dw1, dm1, q00, q01, q02, q03, q10, q11, q12, q13,
                               one_v, xt, xo + 96u, dao + 12u, aso + 12u, a[3]);
        d_mn_c = d_mn_n; sc_c = sc_n; q0_c = q0_n; q1_c = q1_n;
      }
      bp_d += 336u; bp_sc += 336u; bp_q += 336u;
      xo += 1024u; dao += 128u; aso += 128u;
    }

    /* Last full 32-subblock tile (branchless, no next-tile prefetch). */
    {
      const float d_c = ie_rq_h2f(d_mn_c & 0xFFFFu);
      const float mn_c = ie_rq_h2f(d_mn_c >> 16u);
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
        const u32 sh = 2u * j;
        const u32 sc16 = ((j < 2u ? sc_c.x : sc_c.y) >> (16u * (j & 1u))) & 0xFFFFu;
        const float dw0 = d_c * (float)(sc16 & 15u);
        const float dm0 = mn_c * (float)((sc16 >> 4u) & 15u);
        const float dw1 = d_c * (float)((sc16 >> 8u) & 15u);
        const float dm1 = mn_c * (float)((sc16 >> 12u) & 15u);
        const u32 q00 = (q0_c.x >> sh) & 0x03030303u, q01 = (q0_c.y >> sh) & 0x03030303u;
        const u32 q02 = (q0_c.z >> sh) & 0x03030303u, q03 = (q0_c.w >> sh) & 0x03030303u;
        const u32 q10 = (q1_c.x >> sh) & 0x03030303u, q11 = (q1_c.y >> sh) & 0x03030303u;
        const u32 q12 = (q1_c.z >> sh) & 0x03030303u, q13 = (q1_c.w >> sh) & 0x03030303u;
        r11_g08_q2k_dot8_sload(dw0, dm0, dw1, dm1, q00, q01, q02, q03, q10, q11, q12, q13,
                               one_v, xt, xo + 32u * j, dao + 4u * j, aso + 4u * j, a[j]);
      }
    }
    bp_d += 336u; bp_sc += 336u; bp_q += 336u;
    xo += 1024u; dao += 128u; aso += 128u;
  }

  /* Arbitrary nb % 32 != 0 tail (Q2_K superblocks are 8 subblocks, so nb is a
   * multiple of 8; single wave-uniform guard nb32 + 4*w + 3 < nb guarantees all
   * j = 0..3 are in-bounds with a branchless BB, avoiding CFG code-sinking SGPR
   * spills). */
  if (nb32 < nb) {
    if (nb32 + 4u * w + 3u < nb) {
      const u32 d_mn_c = ie_rq_a32(bp_d, 0u);
      const r11_g08_u32x2 sc_c = r11_g08_ld_u32x2_a4(bp_sc, 0u);
      const u32x4 q0_c = r11_g08_ld_u32x4_a4(bp_q, 0u);
      const u32x4 q1_c = r11_g08_ld_u32x4_a4(bp_q, 16u);
      const float d_c = ie_rq_h2f(d_mn_c & 0xFFFFu);
      const float mn_c = ie_rq_h2f(d_mn_c >> 16u);
#pragma unroll
      for (u32 j = 0; j < 4u; j++) {
        const u32 sh = 2u * j;
        const u32 sc16 = ((j < 2u ? sc_c.x : sc_c.y) >> (16u * (j & 1u))) & 0xFFFFu;
        const float dw0 = d_c * (float)(sc16 & 15u);
        const float dm0 = mn_c * (float)((sc16 >> 4u) & 15u);
        const float dw1 = d_c * (float)((sc16 >> 8u) & 15u);
        const float dm1 = mn_c * (float)((sc16 >> 12u) & 15u);
        const u32 q00 = (q0_c.x >> sh) & 0x03030303u, q01 = (q0_c.y >> sh) & 0x03030303u;
        const u32 q02 = (q0_c.z >> sh) & 0x03030303u, q03 = (q0_c.w >> sh) & 0x03030303u;
        const u32 q10 = (q1_c.x >> sh) & 0x03030303u, q11 = (q1_c.y >> sh) & 0x03030303u;
        const u32 q12 = (q1_c.z >> sh) & 0x03030303u, q13 = (q1_c.w >> sh) & 0x03030303u;
        r11_g08_q2k_dot8_sload(dw0, dm0, dw1, dm1, q00, q01, q02, q03, q10, q11, q12, q13,
                               one_v, xt, xo + 32u * j, dao + 4u * j, aso + 4u * j, a[j]);
      }
    }
  }

  r11_g08_q2k_reduce32(red, w, l, r0, rows, a, bias, res, y, ys, rs);
}
