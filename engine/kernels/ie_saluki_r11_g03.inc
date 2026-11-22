/* ============================================================================
 * Round 11 Group 3: Bitwise-exact tall-skinny Q8_0 x Q8_0 WMMA GEMM
 *                   (`ie_saluki_gemm_q8r_r11`) for RX 9070 XT (gfx1201).
 *
 * Targets `ie_gemm_q8r` (`178.7 ms` trim / `199.1 ms` med, `n=3,312` launches
 * in `logs/saluki-opt-r10-final-ab/traced_cand/summary.json`), where 100% of
 * Saluki prefill calls are `Q8_0: 96x5120` (GDN beta/alpha projection across
 * all 48 linear-attention layers):
 *   - Base `ie_gemm_q8r` uses a 256x64 tile (`GR_R=256, GR_T=64`), launching
 *     only `8` WGs at `T=512`, `2..5` WGs at `T=65..301`, and `1` WG at
 *     `T=17..64` on a 64-CU GPU (87.5%..98.4% of CUs idle), with 5 of 8 waves
 *     (62.5%) in every WG computing clamped dummy row 95 and discarding it.
 *   - `ie_saluki_gemm_q8r_r11` uses a 16x16 output tile with 8-way
 *     wave-cooperative K-step (`K_STEP = 8` blocks = 256 cols/iter):
 *       * Grid: `ceil(rows / 16) * ceil(T / 16)` -> `192` WGs at `T=512`
 *         (exact `3.0 WGs/CU = 6 WGs/WGP`, 100% single-wave GPU saturation,
 *         zero wave tail; `18..24` WGs on warm `T=44..49` vs `1` WG before).
 *       * Token-major `(rt = wgid / ntt, tt = wgid % ntt)`: at `T=512`
 *         (`ntt = 32 == 32 WGPs`), both round-robin (stride-32 -> X reuse) and
 *         contiguous (stride-1 -> W reuse) WG placement achieve L0 sharing.
 *       * Direct-VGPR weight & activation loads (`1x global_load_b128` each
 *         per wave): zero LDS staging for int8 W or X.
 *       * Wave-specialized coalesced scale loads: Wave 0 (`h==0`) loads 8
 *         consecutive `f16` weight scales in one `global_load_b128` per row
 *         and Wave 1 loads 4+4 consecutive `f32` activation scales in one
 *         `global_load_b128` per lane (`8x` less TA scale traffic).
 *       * Exact-order wave-cooperative reduction: all 256 threads own 1 output
 *         element `(r_out = t0 & 15, t_out = t0 >> 4)` each (`acc` in 1 VGPR)
 *         and accumulate `k = 0..7` in strict ascending block order via
 *         `__builtin_fmaf(sw * dx, dot_f, acc)` — 100% bitwise identical
 *         (0-bit-diff) to `ie_gemm_q8r`.
 *       * Double-buffered, zero-bank-conflict LDS (`19,456 B` total ->
 *         `6 WGs/WGP`): XOR-swizzled `lds_dot` (`2 * 2048` floats = `16,384 B`)
 *         + stride-12 padded `lds_sw` (`2 * 192` floats = `1,536 B`) +
 *         `lds_dx` (`2 * 192` floats = `1,536 B`) -> 1 `lbarrier()` per 8
 *         blocks (`20` barriers total for `nb=160` vs `40` in `ie_gemm_q8r`).
 *       * Double-buffered VGPR prefetch (`cur` + `nxt`) with uniform `saddr` +
 *         32-bit `voffset` loads and explicit `cur` drain before `nxt` issue:
 *         overlaps global loads across WMMA + LDS store + `lbarrier()` +
 *         8-block LDS reduction with ~48-56 VGPRs, 0 spills, 0 B private mem.
 *       * 100% coalesced output stores (16 consecutive rows per half-wave).
 *
 * Included at EOF of `kernels/ie_saluki_kernels.c` by r11-wiring.
 * All local helpers/types/macros are prefixed `r11_g03_` / `R11_G03_` and all
 * macros are `#undef`'d at EOF.
 * ========================================================================== */

#define R11_G03_TILE 16u
#define R11_G03_KSTEP 8u
#define R11_G03_SC_STRIDE 12u
#define R11_G03_DOT_BUF 2048u /* 8 waves * 16 tokens * 16 rows */
#define R11_G03_SC_BUF 192u   /* 16 rows/tokens * 12-float padded stride */

typedef _Float16 r11_g03_f16x2 __attribute__((ext_vector_type(2)));

struct __attribute__((packed, aligned(4))) r11_g03_u32x4p {
  u32x4 v;
};

static inline __attribute__((always_inline)) u32x4 r11_g03_ld4(const G void *p) {
  return ((const __attribute__((address_space(1))) struct r11_g03_u32x4p *)p)->v;
}

/* Compute exact int32 block dot product for block (kb0 + w) via 2x gfx12 int8
 * WMMA (same 0x4B400000 bias and K-byte ordering as ie_gemm_q8r), convert to
 * exact float dot, and store dot + wave-specialized scales into double-buffer
 * slot `dot_b`/`sw_b`/`dx_b` with mathematically zero LDS bank conflicts. */
static inline __attribute__((always_inline)) void r11_g03_compute_store(
    const u32x4 w_vec, const u32x4 x_vec, const u32x4 sc_vec, const u32 w,
    const u32 r16, const u32 h, const u32 st_col0, const u32 st_col1,
    LDSP float *dot_b, LDSP float *sw_b, LDSP float *dx_b) {
  const v8i zm = (v8i)(0x4B400000);
  const v2i a0 = {(int)w_vec.x, (int)w_vec.y};
  const v2i a1 = {(int)w_vec.z, (int)w_vec.w};
  const v2i b0 = {(int)x_vec.x, (int)x_vec.y};
  const v2i b1 = {(int)x_vec.z, (int)x_vec.w};
  v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0, zm, 0);
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1, c, 0);
  const f8 df = __builtin_bit_cast(f8, c) - 12582912.0f;
  LDSP float *dp = dot_b + (w << 8u) + (r16 << 4u);
  *(LDSP f32x4 *)(dp + st_col0) = (f32x4){df[0], df[1], df[2], df[3]};
  *(LDSP f32x4 *)(dp + st_col1) = (f32x4){df[4], df[5], df[6], df[7]};

  if (w == 0u) {
    if (h == 0u) {
      const u32 u0 = sc_vec.x, u1 = sc_vec.y, u2 = sc_vec.z, u3 = sc_vec.w;
      const r11_g03_f16x2 p0 = __builtin_bit_cast(r11_g03_f16x2, u0);
      const r11_g03_f16x2 p1 = __builtin_bit_cast(r11_g03_f16x2, u1);
      const r11_g03_f16x2 p2 = __builtin_bit_cast(r11_g03_f16x2, u2);
      const r11_g03_f16x2 p3 = __builtin_bit_cast(r11_g03_f16x2, u3);
      LDSP float *sp = sw_b + r16 * R11_G03_SC_STRIDE;
      *(LDSP f32x4 *)(sp + 0u) = (f32x4){(float)p0.x, (float)p0.y, (float)p1.x, (float)p1.y};
      *(LDSP f32x4 *)(sp + 4u) = (f32x4){(float)p2.x, (float)p2.y, (float)p3.x, (float)p3.y};
    }
  } else if (w == 1u) {
    const u32 u0 = sc_vec.x, u1 = sc_vec.y, u2 = sc_vec.z, u3 = sc_vec.w;
    const f32x4 dxv = {__builtin_bit_cast(float, u0), __builtin_bit_cast(float, u1),
                       __builtin_bit_cast(float, u2), __builtin_bit_cast(float, u3)};
    *(LDSP f32x4 *)(dx_b + r16 * R11_G03_SC_STRIDE + 4u * h) = dxv;
  }
}

/* Reduce 8 consecutive K-blocks (k = 0..7) for this thread's single output
 * element (r_out, t_out) in strict ascending order, matching ie_gemm_q8r's
 * exact FP32 arithmetic: acc = fma(sw * dx, dot_f, acc). */
static inline __attribute__((always_inline)) float r11_g03_reduce_buf(
    const LDSP float *dot_b, const LDSP float *sw_b, const LDSP float *dx_b,
    const u32 t_out, const u32 r_out, const u32 r_sw, float acc) {
  const LDSP float *dp = dot_b + (t_out << 4u) + r_sw;
  const LDSP float *sp = sw_b + r_out * R11_G03_SC_STRIDE;
  const LDSP float *xp = dx_b + t_out * R11_G03_SC_STRIDE;
  const f32x4 sw0 = *(const LDSP f32x4 *)(sp + 0u);
  const f32x4 dx0 = *(const LDSP f32x4 *)(xp + 0u);
  const float d0 = dp[0u << 8u];
  const float d1 = dp[1u << 8u];
  const float d2 = dp[2u << 8u];
  const float d3 = dp[3u << 8u];
  const f32x4 sw1 = *(const LDSP f32x4 *)(sp + 4u);
  const f32x4 dx1 = *(const LDSP f32x4 *)(xp + 4u);
  const float d4 = dp[4u << 8u];
  const float d5 = dp[5u << 8u];
  const float d6 = dp[6u << 8u];
  const float d7 = dp[7u << 8u];
  __builtin_amdgcn_sched_barrier(0);
  const f32x4 sc0 = sw0 * dx0;
  acc = __builtin_fmaf(sc0.x, d0, acc);
  acc = __builtin_fmaf(sc0.y, d1, acc);
  acc = __builtin_fmaf(sc0.z, d2, acc);
  acc = __builtin_fmaf(sc0.w, d3, acc);
  const f32x4 sc1 = sw1 * dx1;
  acc = __builtin_fmaf(sc1.x, d4, acc);
  acc = __builtin_fmaf(sc1.y, d5, acc);
  acc = __builtin_fmaf(sc1.z, d6, acc);
  acc = __builtin_fmaf(sc1.w, d7, acc);
  return acc;
}

KERNEL ie_saluki_gemm_q8r_r11(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                              const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  static LDS float lds_dot[2u * R11_G03_DOT_BUF] __attribute__((aligned(16)));
  static LDS float lds_sw[2u * R11_G03_SC_BUF] __attribute__((aligned(16)));
  static LDS float lds_dx[2u * R11_G03_SC_BUF] __attribute__((aligned(16)));
  _Static_assert(sizeof(f16) == 2u, "r11_g03 f16 must be 2 bytes");
  _Static_assert(sizeof(lds_dot) + sizeof(lds_sw) + sizeof(lds_dx) == 19456u, "r11_g03 LDS must be 19456 B");

  if (T == 0u || rows == 0u || nb < R11_G03_KSTEP || (nb & (R11_G03_KSTEP - 1u)) != 0u || !qw || !qs || !xq || !y)
    return;
  const u32 ntt = (T + R11_G03_TILE - 1u) >> 4u;
  const u32 tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 rbase = rt * R11_G03_TILE, tbase = tt * R11_G03_TILE;
  if (rbase >= rows || tbase >= T) return;

  const u32 t0 = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t0 >> 5u));
  const u32 l = lane(), r16 = l & 15u, h = l >> 4u;

  /* Output coordinate owned by thread t0 (1 thread per element of 16x16 tile). */
  const u32 r_out = t0 & 15u;
  const u32 t_out = t0 >> 4u;
  const u32 r_sw = r_out ^ ((t_out & 6u) << 1u);

  /* Zero-bank-conflict XOR store columns for WMMA output of lane l. */
  const u32 st_swiz = (r16 & 6u) << 1u;
  const u32 st_col0 = (8u * h) ^ st_swiz;
  const u32 st_col1 = (8u * h + 4u) ^ st_swiz;

  /* Uniform 64-bit tile bases + 32-bit lane offsets -> saddr + voffset global_load_b128. */
  const u32 dr_max = rows - 1u - rbase;
  const u32 dt_max = T - 1u - tbase;
  const u32 dr = r16 <= dr_max ? r16 : dr_max;
  const u32 dt = r16 <= dt_max ? r16 : dt_max;

  const G u8 *qw_tile = (const G u8 *)qw + (unsigned long)rbase * (unsigned long)nb * 32ul;
  const G u8 *xq_tile = xq + (unsigned long)tbase * (unsigned long)xs;
  const G u8 *sc_tile = (w == 0u) ? ((const G u8 *)qs + (unsigned long)rbase * (unsigned long)nb * 2ul)
                                  : (xq_tile + (unsigned long)nb * 32ul);
  const u32 sc_shift = (w == 0u) ? 1u : 2u;

  const u32 woff = dr * (nb * 32u) + w * 32u + 16u * h;
  const u32 xoff = dt * xs + w * 32u + 16u * h;
  const u32 scoff = (w == 0u) ? (dr * (nb * 2u)) : (dt * xs + 16u * h);

  float acc = 0.0f;
  __asm__ volatile("" : "+v"(acc));

  /* Prologue: load chunk 0 (kb0 = 0..7), drain it once, prefetch chunk 1 while
   * computing chunk 0 WMMA + storing into double-buffer slot 0. */
  u32x4 cur_w = r11_g03_ld4(qw_tile + (unsigned long)woff);
  u32x4 cur_x = r11_g03_ld4(xq_tile + (unsigned long)xoff);
  u32x4 cur_sc = (u32x4)(0u);
  if (w < 2u) {
    if (w == 1u || h == 0u) cur_sc = r11_g03_ld4(sc_tile + (unsigned long)scoff);
  }
  __asm__ volatile("" : "+v"(cur_w), "+v"(cur_x));
  if (w < 2u) __asm__ volatile("" : "+v"(cur_sc));

  u32x4 nxt_w = (u32x4)(0u), nxt_x = (u32x4)(0u), nxt_sc = (u32x4)(0u);
  if (nb > R11_G03_KSTEP) {
    const G u8 *qw_1 = qw_tile + (unsigned long)R11_G03_KSTEP * 32ul;
    const G u8 *xq_1 = xq_tile + (unsigned long)R11_G03_KSTEP * 32ul;
    const G u8 *sc_1 = sc_tile + ((unsigned long)R11_G03_KSTEP << sc_shift);
    nxt_w = r11_g03_ld4(qw_1 + (unsigned long)woff);
    nxt_x = r11_g03_ld4(xq_1 + (unsigned long)xoff);
    if (w < 2u) {
      if (w == 1u || h == 0u) nxt_sc = r11_g03_ld4(sc_1 + (unsigned long)scoff);
    }
  }
  __builtin_amdgcn_sched_barrier(0);

  r11_g03_compute_store(cur_w, cur_x, cur_sc, w, r16, h, st_col0, st_col1, lds_dot, lds_sw, lds_dx);
  lbarrier();

  u32 bf = 0u;
#pragma unroll 1
  for (u32 kb0 = R11_G03_KSTEP; kb0 < nb; kb0 += R11_G03_KSTEP) {
    /* Reduce chunk (kb0 - 8) from buffer `bf` while `nxt_*` (chunk kb0) finishes flying. */
    acc = r11_g03_reduce_buf(lds_dot + bf * R11_G03_DOT_BUF, lds_sw + bf * R11_G03_SC_BUF,
                             lds_dx + bf * R11_G03_SC_BUF, t_out, r_out, r_sw, acc);

    cur_w = nxt_w;
    cur_x = nxt_x;
    if (w < 2u) cur_sc = nxt_sc;
    __asm__ volatile("" : "+v"(cur_w), "+v"(cur_x));
    if (w < 2u) __asm__ volatile("" : "+v"(cur_sc));

    const u32 kbn = kb0 + R11_G03_KSTEP;
    if (kbn < nb) {
      const G u8 *qw_n = qw_tile + (unsigned long)kbn * 32ul;
      const G u8 *xq_n = xq_tile + (unsigned long)kbn * 32ul;
      const G u8 *sc_n = sc_tile + ((unsigned long)kbn << sc_shift);
      nxt_w = r11_g03_ld4(qw_n + (unsigned long)woff);
      nxt_x = r11_g03_ld4(xq_n + (unsigned long)xoff);
      if (w < 2u) {
        if (w == 1u || h == 0u) nxt_sc = r11_g03_ld4(sc_n + (unsigned long)scoff);
      }
    }
    __builtin_amdgcn_sched_barrier(0);

    const u32 nbf = bf ^ 1u;
    r11_g03_compute_store(cur_w, cur_x, cur_sc, w, r16, h, st_col0, st_col1,
                          lds_dot + nbf * R11_G03_DOT_BUF, lds_sw + nbf * R11_G03_SC_BUF,
                          lds_dx + nbf * R11_G03_SC_BUF);
    lbarrier();
    bf = nbf;
  }

  /* Epilogue: reduce final chunk (nb - 8 .. nb - 1) from buffer `bf` and store coalesced. */
  acc = r11_g03_reduce_buf(lds_dot + bf * R11_G03_DOT_BUF, lds_sw + bf * R11_G03_SC_BUF,
                           lds_dx + bf * R11_G03_SC_BUF, t_out, r_out, r_sw, acc);

  const u32 r = rbase + r_out;
  const u32 t = tbase + t_out;
  if (r < rows && t < T) {
    const G float *rp = res ? res + (unsigned long)t * (unsigned long)rs : (const G float *)0;
    y[(unsigned long)t * (unsigned long)ys + r] = epilogue(acc, r, bias, rp);
  }
}

#undef R11_G03_SC_BUF
#undef R11_G03_DOT_BUF
#undef R11_G03_SC_STRIDE
#undef R11_G03_KSTEP
#undef R11_G03_TILE
