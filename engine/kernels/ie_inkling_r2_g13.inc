/* ============================================================================
 * Inkling-2 Round 2 Group 13: Post-Conv Residual, MoE-Sum Tail & Gate+Q8 Fusions
 * for RX 9070 XT (gfx1201, wave32, NT = 256).
 *
 * Exported KERNELs, exact grids (gx, gy, 1), dispatch guards, and parity:
 *   1. KERNEL ie_inkling_sconv_add_r2(const G float *a, const G float *v, const G float *c, G float *ring,
 *                                     const G float *res, G float *out, u32 n, u32 nh, float escale, u32 pos0, u32 T,
 *                                     u32 as, u32 cs, u32 rs, u32 os)
 *      Bit:    27
 *      Fuses:  OP_SCONV(a, v, c, ring -> tmp) + OP_ADD(res, tmp -> out)  [attn_pc (36x) & ffn_pc (48x)]
 *      Grid:   (((n >> 2u) + 255u) / 256u, (T + 7u) / 8u, 1)  [TTILE = 8 tokens/WG; (3, 1, 1) at T=1, dim=2304]
 *      Guard:  (n & 3u) == 0u && (!c || (nh > 0u && ((n / nh) & 3u) == 0u)) &&
 *              ptrs a, v, ring, res, out (, c) 16-byte aligned &&
 *              (T == 1u || ((as | rs | os | (c ? cs : 0u)) & 15u) == 0u).
 *      Note:   When T == 1u updates ring in-place; when T > 1u host still runs ie_sconv_ring_store(a, ring, ...)
 *              after (avoids inter-WG ring race). out may alias res.
 *      Parity: 100% bitwise-exact vs OP_SCONV + OP_ADD (#pragma clang fp contract(off), exact left-to-right sum).
 *
 *   2. KERNEL ie_inkling_moe_sum_sconv_add_r2(const G float *y, const G u32 *route, const G float *v, G float *ring,
 *                                             const G float *res, G float *out, u32 n, u32 K, float escale, u32 pos0)
 *      Bit:    28
 *      Fuses:  OP_MOE_SUM(y, route, res=-1 -> tmp) + OP_SCONV(tmp, v, c=NULL, ring -> tmp2) + OP_ADD(res, tmp2 -> out)
 *              [decode T == 1 FFN tail, 48 visits/tok -> -96 launches/tok]
 *      Grid:   (((n >> 2u) + 255u) / 256u, 1, 1)  [(3, 1, 1) for dim = 2304]
 *      Guard:  T == 1u && (n & 3u) == 0u && K > 0u && ptrs y, v, ring, res, out 16-byte aligned.
 *      Note:   Updates ring in-place; out may alias res. Static 4-tap conv only (ffn_pconv has c == NULL).
 *      Parity: 100% bitwise-exact vs OP_MOE_SUM + OP_SCONV + OP_ADD (exact K=8 __builtin_fmaf order + exact sconv sum).
 *
 *   3. KERNEL ie_inkling_gate_q8_r2(const G float *a, const G float *g, G float *b, G u8 *q,
 *                                   u32 n, u32 as, u32 gs, u32 bs, u32 qs)
 *      Bit:    29
 *      Replaces: OP_GATE (att *= sigmoid(agate) + fused Q8_0 quant, 48 visits/step, BOTH T=1 and T>1).
 *      Grid:   ((n + 1023u) / 1024u, T, 1)  [1 WG per 32 Q8_0 blocks / 1024 floats; (3, T, 1) for qd = 3072]
 *              (accepts any gx >= 1u via grid-stride over 32-block chunks)
 *      Guard:  (n & 31u) == 0u && ptrs a, g (, b) 16-byte aligned && (!q || ptr q 4-byte aligned) &&
 *              (T == 1u || (((as | gs | (b ? bs : 0u)) & 15u) == 0u && (!q || (qs & 3u) == 0u))).
 *      Note:   b == NULL elides dead f32 writeback (in model.c:3007-3009, att is never read after OP_GATE when
 *              L->wo.kind != MAT_F32 because matvec reads aq). b == a is safe for in-place writeback.
 *      Parity: 100% bitwise-exact vs ie_gate (both f32 b and Q8_0 q).
 *
 * Included at EOF of kernels/ie_inkling_kernels.c.
 * All local helpers/macros prefixed ink_r2_g13_ / INK_R2_G13_; local macros #undef'd at EOF.
 * 0 LDS bytes used across all 3 kernels.
 * ========================================================================== */

#define INK_R2_G13_SCONV_TTILE 8u

static inline __attribute__((always_inline)) float ink_r2_g13_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline __attribute__((always_inline)) float ink_r2_g13_sub8_max_pos(float v) {
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0xB1, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0x4E, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_ds_swizzle(F2I(v), 0x101F)));
  return v;
}

static inline __attribute__((always_inline)) u32 ink_r2_g13_sub8_sum_u32(u32 s) {
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0xB1, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0x4E, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_ds_swizzle((int)s, 0x101F);
  return s;
}

/* ----------------------------------------------------------------------------
 * Kernel 1: ie_inkling_sconv_add_r2 (bit 27)
 *   Fused 4-tap causal short convolution + residual add, vectorized f32x4,
 *   TTILE=8 shift-register on T > 1, in-place ring update on T == 1.
 *   Bitwise-exact against OP_SCONV + OP_ADD.
 * -------------------------------------------------------------------------- */
KERNEL ie_inkling_sconv_add_r2(const G float *a, const G float *v, const G float *c, G float *ring, const G float *res,
                               G float *out, u32 n, u32 nh, float escale, u32 pos0, u32 T, u32 as, u32 cs, u32 rs,
                               u32 os) {
#pragma clang fp contract(off)
  const u32 n4 = n >> 2u;
  const u32 i4 = wgid() * NT + tid();
  if (i4 >= n4) return;
  const u32 i0 = i4 << 2u;

  /* Load 4-tap static conv weights once for all tokens in this WG tile. */
  const G f32x4 *v4 = (const G f32x4 *)v + i0;
  const f32x4 vw0 = v4[0], vw1 = v4[1], vw2 = v4[2], vw3 = v4[3];
  const f32x4 sw0 = (f32x4){vw0.w, vw1.w, vw2.w, vw3.w};
  const f32x4 sw1 = (f32x4){vw0.z, vw1.z, vw2.z, vw3.z};
  const f32x4 sw2 = (f32x4){vw0.y, vw1.y, vw2.y, vw3.y};
  const f32x4 sw3 = (f32x4){vw0.x, vw1.x, vw2.x, vw3.x};

  const u32 h = c ? (i0 / (n / nh)) : 0u;

  if (T == 1u) {
    const f32x4 x0 = ((const G f32x4 *)a)[i4];
    const f32x4 rv = ((const G f32x4 *)res)[i4];
    f32x4 x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    if (pos0 > 0u) {
      const G f32x4 *rg = (const G f32x4 *)ring + i0;
      const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
      const u32 s1 = (pos0 - 1u) & 3u;
      x1 = (f32x4){ink_r2_g13_pick4(r0, s1), ink_r2_g13_pick4(r1, s1), ink_r2_g13_pick4(r2, s1),
                   ink_r2_g13_pick4(r3, s1)};
      if (pos0 >= 2u) {
        const u32 s2 = (pos0 - 2u) & 3u;
        x2 = (f32x4){ink_r2_g13_pick4(r0, s2), ink_r2_g13_pick4(r1, s2), ink_r2_g13_pick4(r2, s2),
                     ink_r2_g13_pick4(r3, s2)};
      }
      if (pos0 >= 3u) {
        const u32 s3 = (pos0 - 3u) & 3u;
        x3 = (f32x4){ink_r2_g13_pick4(r0, s3), ink_r2_g13_pick4(r1, s3), ink_r2_g13_pick4(r2, s3),
                     ink_r2_g13_pick4(r3, s3)};
      }
    }
    f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
    if (c) {
      const f32x4 ct = ((const G f32x4 *)c)[h];
      w0 += ct.x, w1 += ct.y, w2 += ct.z, w3 += ct.w;
    }
    const f32x4 y = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3;
    const f32x4 sc = (x0 + y) * escale;
    ((G f32x4 *)out)[i4] = rv + sc;
    const u32 slot = pos0 & 3u;
    ring[(i0 + 0u) * 4u + slot] = x0.x;
    ring[(i0 + 1u) * 4u + slot] = x0.y;
    ring[(i0 + 2u) * 4u + slot] = x0.z;
    ring[(i0 + 3u) * 4u + slot] = x0.w;
    return;
  }

  const u32 t0 = tokid() * INK_R2_G13_SCONV_TTILE;
  if (t0 >= T) return;
  const u32 t_end = (t0 + INK_R2_G13_SCONV_TTILE < T) ? (t0 + INK_R2_G13_SCONV_TTILE) : T;

  /* Prime the 3-token sliding register window [x3, x2, x1] before stepping t = t0 .. t_end - 1. */
  f32x4 x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (t0 >= 3u) {
    x3 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)(t0 - 3u) * as))[i4];
    x2 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)(t0 - 2u) * as))[i4];
    x1 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)(t0 - 1u) * as))[i4];
  } else if (pos0 > 0u) {
    /* t0 == 0u (since TTILE == 8 >= 3): seed from ring for lags 1..3. */
    const G f32x4 *rg = (const G f32x4 *)ring + i0;
    const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
    const u32 s1 = (pos0 - 1u) & 3u;
    x1 = (f32x4){ink_r2_g13_pick4(r0, s1), ink_r2_g13_pick4(r1, s1), ink_r2_g13_pick4(r2, s1),
                 ink_r2_g13_pick4(r3, s1)};
    if (pos0 >= 2u) {
      const u32 s2 = (pos0 - 2u) & 3u;
      x2 = (f32x4){ink_r2_g13_pick4(r0, s2), ink_r2_g13_pick4(r1, s2), ink_r2_g13_pick4(r2, s2),
                   ink_r2_g13_pick4(r3, s2)};
    }
    if (pos0 >= 3u) {
      const u32 s3 = (pos0 - 3u) & 3u;
      x3 = (f32x4){ink_r2_g13_pick4(r0, s3), ink_r2_g13_pick4(r1, s3), ink_r2_g13_pick4(r2, s3),
                   ink_r2_g13_pick4(r3, s3)};
    }
  }

  for (u32 t = t0; t < t_end; t++) {
    const f32x4 x0 = ((const G f32x4 *)((const G u8 *)a + (unsigned long)t * as))[i4];
    const f32x4 rv = ((const G f32x4 *)((const G u8 *)res + (unsigned long)t * rs))[i4];
    f32x4 w0 = sw0, w1 = sw1, w2 = sw2, w3 = sw3;
    if (c) {
      const f32x4 ct = ((const G f32x4 *)((const G u8 *)c + (unsigned long)t * cs))[h];
      w0 += ct.x, w1 += ct.y, w2 += ct.z, w3 += ct.w;
    }
    const f32x4 y = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3;
    const f32x4 sc = (x0 + y) * escale;
    ((G f32x4 *)((G u8 *)out + (unsigned long)t * os))[i4] = rv + sc;
    x3 = x2;
    x2 = x1;
    x1 = x0;
  }
}

/* ----------------------------------------------------------------------------
 * Kernel 2: ie_inkling_moe_sum_sconv_add_r2 (bit 28)
 *   Decode (T == 1) FFN tail fusion: OP_MOE_SUM + OP_SCONV (static) + OP_ADD
 *   into 1 launch with in-place ring update.
 *   Bitwise-exact (exact K=8 __builtin_fmaf order + exact left-to-right sconv).
 * -------------------------------------------------------------------------- */
KERNEL ie_inkling_moe_sum_sconv_add_r2(const G float *y, const G u32 *route, const G float *v, G float *ring,
                                       const G float *res, G float *out, u32 n, u32 K, float escale, u32 pos0) {
#pragma clang fp contract(off)
  const u32 n4 = n >> 2u;
  const u32 i4 = wgid() * NT + tid();
  if (i4 >= n4) return;
  const u32 i0 = i4 << 2u;

  /* Issue independent weight, residual, and expert loads together. */
  const G f32x4 *v4 = (const G f32x4 *)v + i0;
  const f32x4 vw0 = v4[0], vw1 = v4[1], vw2 = v4[2], vw3 = v4[3];
  const f32x4 rv = ((const G f32x4 *)res)[i4];

  const G float *w = (const G float *)(route + K);
  const G f32x4 *y4 = (const G f32x4 *)y;
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

  f32x4 x1 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 x2 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  f32x4 x3 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  if (pos0 > 0u) {
    const G f32x4 *rg = (const G f32x4 *)ring + i0;
    const f32x4 r0 = rg[0], r1 = rg[1], r2 = rg[2], r3 = rg[3];
    const u32 s1 = (pos0 - 1u) & 3u;
    x1 = (f32x4){ink_r2_g13_pick4(r0, s1), ink_r2_g13_pick4(r1, s1), ink_r2_g13_pick4(r2, s1),
                 ink_r2_g13_pick4(r3, s1)};
    if (pos0 >= 2u) {
      const u32 s2 = (pos0 - 2u) & 3u;
      x2 = (f32x4){ink_r2_g13_pick4(r0, s2), ink_r2_g13_pick4(r1, s2), ink_r2_g13_pick4(r2, s2),
                   ink_r2_g13_pick4(r3, s2)};
    }
    if (pos0 >= 3u) {
      const u32 s3 = (pos0 - 3u) & 3u;
      x3 = (f32x4){ink_r2_g13_pick4(r0, s3), ink_r2_g13_pick4(r1, s3), ink_r2_g13_pick4(r2, s3),
                   ink_r2_g13_pick4(r3, s3)};
    }
  }

  const f32x4 sw0 = (f32x4){vw0.w, vw1.w, vw2.w, vw3.w};
  const f32x4 sw1 = (f32x4){vw0.z, vw1.z, vw2.z, vw3.z};
  const f32x4 sw2 = (f32x4){vw0.y, vw1.y, vw2.y, vw3.y};
  const f32x4 sw3 = (f32x4){vw0.x, vw1.x, vw2.x, vw3.x};

  const f32x4 cv = ((x0 * sw0 + x1 * sw1) + x2 * sw2) + x3 * sw3;
  const f32x4 sc = (x0 + cv) * escale;
  ((G f32x4 *)out)[i4] = rv + sc;

  const u32 slot = pos0 & 3u;
  ring[(i0 + 0u) * 4u + slot] = x0.x;
  ring[(i0 + 1u) * 4u + slot] = x0.y;
  ring[(i0 + 2u) * 4u + slot] = x0.z;
  ring[(i0 + 3u) * 4u + slot] = x0.w;
}

/* ----------------------------------------------------------------------------
 * Kernel 3: ie_inkling_gate_q8_r2 (bit 29)
 *   8-lane subgroup f32x4 attention output gate (a * sigmoid(g)) + fused Q8_0
 *   quantization (32 blocks / 1024 floats per WG, 0 LDS, packed u32 int8 stores).
 *   Supports b == NULL (dead f32 output elision) and b == a (in-place).
 *   100% bitwise-exact against ie_gate (both f32 b and Q8_0 q).
 * -------------------------------------------------------------------------- */
KERNEL ie_inkling_gate_q8_r2(const G float *a, const G float *g, G float *b, G u8 *q, u32 n, u32 as, u32 gs, u32 bs,
                             u32 qs) {
#pragma clang fp contract(off)
  const u32 nb = n >> 5u;
  const u32 sg = tid() >> 3u;
  const u32 sl = lane() & 7u;
  const u32 gx = ngroups_x();

  a = TOKC(a, as);
  g = TOKC(g, gs);
  if (b) b = TOK(b, bs);
  if (q) q = TOK(q, qs);

  for (u32 blk = wgid() * 32u + sg; blk < nb; blk += gx * 32u) {
    const u32 off = blk * 32u + sl * 4u;
    const f32x4 av = *(const G f32x4 *)(a + off);
    const f32x4 gv = *(const G f32x4 *)(g + off);

    f32x4 vv;
#pragma unroll
    for (u32 k = 0u; k < 4u; k++) {
      vv[k] = av[k] * (1.0f / (1.0f + __builtin_expf(-gv[k])));
    }

    if (b) *(G f32x4 *)(b + off) = vv;

    if (q) {
      float am = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(vv[0]), __builtin_fabsf(vv[1])),
                                 __builtin_fmaxf(__builtin_fabsf(vv[2]), __builtin_fabsf(vv[3])));
      am = ink_r2_g13_sub8_max_pos(am);
      const float d = am / 127.0f;
      int qi[4];
      if (d != 0.0f) {
#pragma unroll
        for (u32 k = 0u; k < 4u; k++) qi[k] = (int)__builtin_roundf(vv[k] / d);
      } else {
        qi[0] = qi[1] = qi[2] = qi[3] = 0;
      }
      const u32 pk = ((u32)(u8)qi[0]) | (((u32)(u8)qi[1]) << 8u) | (((u32)(u8)qi[2]) << 16u) |
                     (((u32)(u8)qi[3]) << 24u);
      *(G u32 *)(q + off) = pk;
      const u32 s = ink_r2_g13_sub8_sum_u32((u32)qi[0] + (u32)qi[1] + (u32)qi[2] + (u32)qi[3]);
      if (sl == 0u) {
        ((G float *)(q + 32u * nb))[blk] = d;
        ((G u32 *)(q + 36u * nb))[blk] = s;
      }
    }
  }
}

#undef INK_R2_G13_SCONV_TTILE
