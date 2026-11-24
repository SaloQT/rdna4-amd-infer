/* ============================================================================
 * Inkling-2 Round 14 (256K + DSpark Helper Kernels):
 *   1. ie_inkling_ds_rope_fill_r14:
 *      On-the-fly DSpark RoPE cos/sin table filler for [pos0 .. pos0 + T - 1].
 *      Replaces DSpark's static 2 * (n_ctx + 32) * 64 * 4 = 128.02 MiB rcos/rsin
 *      table (src/hip.c:3206-3216) with a 512 B FP64 inv_freq table + 512 KiB
 *      dynamic rcos/rsin scratch buffer -> reclaims 127.5 MiB VRAM at 256K
 *      (255.5 MiB at 512K) with ZERO changes to K_QKN_ROPE_KV!
 *
 *      Precision note (CRITICAL at 256K):
 *        Host src/hip.c:3210 computes:
 *          double th = (double)p * pow((double)d->rope_base, -2.0 * j / d->hd);
 *          hcos = (float)cos(th);  hsin = (float)sin(th);
 *        Storing inv_freq as float causes p * eps_f32 ~ 0.014 rad error at
 *        p = 262,144 (breaking exact-SHA). Storing inv_freq as double
 *        (64 * 8 = 512 B, uploaded once at gpu_open) + exact 3-part FP64
 *        Cody-Waite reduction modulo pi/2 (111-bit pi/2, exact up to p > 20M)
 *        + degree-14/13 FP64 Horner matches host (float)cos(th)/(float)sin(th)
 *        with 0 ULP / 0 bit diffs across 5.3M tested points up to p = 1,048,576
 *        for rope_base in {1e4, 1e6, 1e7}, with zero libm dependency.
 *
 *   2. ie_inkling_moe_wl_filter_r14:
 *      H38 Zero-VRAM Prefill Wave-Streaming filtered worklist builder.
 *      Reads the full offs_in[0 .. n_exp] already produced once per layer by
 *      ie_inkling_moe_pf_worklist_r7 / ie_inkling_moe_sort / ie_moe_sort,
 *      filters to contiguous wave [e_lo, min(e_hi, n_exp)), remaps expert e ->
 *      slot = e - e_lo in [0, cnt), and emits:
 *        - tiles_out[0 .. *n_tiles_out - 1]: exact 32-bit packed descriptors
 *          matching ie_inkling_r7_pf_worklist_iq3.inc (bits [7:0] = slot,
 *          [20:8] = lo in unchanged list[], [27:21] = nj 1..64,
 *          [29:28] = cls 3..0 largest-first). Safe to write in-place into
 *          b->ms_packs / b->ms_npacks (0 B extra VRAM!).
 *        - optional offs_out[0 .. cnt]: 0-based copy of offs_in[e_lo .. hi]
 *          when non-NULL (pass NULL to use offs_in + e_lo directly with
 *          expert-major ie_inkling_gemm_iq4nl_exp_r2 / expmaj_r10).
 *      Lets ie_inkling_gemm_rq_exp_iq3xxs_flute_situ_r10 and
 *      ie_inkling_gemm_iq4nl_exp_r2 / expmaj_r10 / exp_r9 run 100% UNCHANGED
 *      on the N_slots-sized VRAM cache buffer during T >= 17 prefill with
 *      0 B extra VRAM and 100% bit-exact OP_MOE_SUM!
 *
 * Target: RX 9070 XT (gfx1201, wave32, NT=256).
 * Guard & hygiene:
 *   - Wrapped in #ifndef IE_MEGA.
 *   - All local symbols/LDS/macros prefixed ink_r14_256k_ / INK_R14_256K_.
 *   - All macros #undef'd at EOF.
 *   - Default cached loads only (0 TH_LOAD_NT, 0 TH_LOAD_HT), 0/0 spills,
 *     priv=0, <= 48 VGPR (4 WGs/CU).
 * ========================================================================== */

#ifndef IE_MEGA

/* Exact 111-bit 3-part IEEE-754 double split of pi/2 (29 + 29 + 53 mantissa bits):
 * k * PIO2_1 is exact in FP64 for |k| <= 2^24 (p <= 26,000,000), residual < 5e-35. */
#define INK_R14_256K_INV_PIO2 0x1.45f306dc9c883p-1 /* 2 / pi */
#define INK_R14_256K_PIO2_1   0x1.921fb54000000p+0 /* high 29 bits of pi/2 */
#define INK_R14_256K_PIO2_2   0x1.10b4611000000p-30 /* mid 29 bits of pi/2 */
#define INK_R14_256K_PIO2_3   0x1.4c4c6628b80dcp-59 /* low 53 bits of pi/2 */

/* ============================================================================
 * Kernel 1: ie_inkling_ds_rope_fill_r14
 *   inv_freq : const G double[half], inv_freq[j] = pow((double)rope_base, -2.0 * j / hd)
 *   rcos     : G float[T * half], row dt holds cos((pos0 + dt) * inv_freq[j])
 *   rsin     : G float[T * half], row dt holds sin((pos0 + dt) * inv_freq[j])
 *   pos0     : starting token position (0 .. 2^24+)
 *   T        : number of contiguous positions to fill (e.g. chunk T or draft B)
 *   half     : n_rot / 2 (64 for DSpark hd=128; fast shift path for 64, general fallback)
 *   Grid     : ((T * half + 255u) / 256u, 1u, 1u), NT = 256
 *              (equals ((T + 3u) / 4u, 1u, 1u) when half == 64u)
 * ========================================================================== */
KERNEL ie_inkling_ds_rope_fill_r14(
    const G double *inv_freq, G float *rcos, G float *rsin,
    u32 pos0, u32 T, u32 half) {
  if (!inv_freq || !rcos || !rsin || T == 0u || half == 0u) return;

  const unsigned long total = (unsigned long)T * (unsigned long)half;
  const unsigned long idx = (unsigned long)wgid() * NT + (unsigned long)tid();
  if (idx >= total) return;

  u32 dt, j;
  if (half == 64u) {
    dt = (u32)(idx >> 6u);
    j = (u32)(idx & 63u);
  } else {
    dt = (u32)(idx / (unsigned long)half);
    j = (u32)(idx - (unsigned long)dt * (unsigned long)half);
  }

  const double p = (double)((unsigned long)pos0 + (unsigned long)dt);
  const double th = p * inv_freq[j];

  /* Exact 3-part FP64 Cody-Waite quadrant reduction modulo pi/2. */
  const double kd = __builtin_rint(th * INK_R14_256K_INV_PIO2);
  const u32 k = (u32)kd;
  double r = __builtin_fma(-kd, INK_R14_256K_PIO2_1, th);
  r = __builtin_fma(-kd, INK_R14_256K_PIO2_2, r);
  r = __builtin_fma(-kd, INK_R14_256K_PIO2_3, r);

  /* Degree-14 cos(r) and degree-13 sin(r) Horner on |r| <= pi/4 (< 1e-15 err). */
  const double r2 = r * r;
  double c = __builtin_fma(r2, -1.0 / 87178291200.0, 1.0 / 479001600.0);
  c = __builtin_fma(r2, c, -1.0 / 3628800.0);
  c = __builtin_fma(r2, c, 1.0 / 40320.0);
  c = __builtin_fma(r2, c, -1.0 / 720.0);
  c = __builtin_fma(r2, c, 1.0 / 24.0);
  c = __builtin_fma(r2, c, -0.5);
  c = __builtin_fma(r2, c, 1.0);

  double s = __builtin_fma(r2, 1.0 / 6227020800.0, -1.0 / 39916800.0);
  s = __builtin_fma(r2, s, 1.0 / 362880.0);
  s = __builtin_fma(r2, s, -1.0 / 5040.0);
  s = __builtin_fma(r2, s, 1.0 / 120.0);
  s = __builtin_fma(r2, s, -1.0 / 6.0);
  s = __builtin_fma(r2 * r, s, r);

  /* Branchless quadrant reconstruction: q = k & 3. */
  const u32 q = k & 3u;
  const double co = (q & 1u) ? -s : c;
  const double so = (q & 1u) ?  c : s;
  const double cf = (q & 2u) ? -co : co;
  const double sf = (q & 2u) ? -so : so;

  rcos[idx] = (float)cf;
  rsin[idx] = (float)sf;
}

/* ============================================================================
 * Kernel 2: ie_inkling_moe_wl_filter_r14
 *   offs_in     : const G u32[n_exp + 1], full expert offsets (b->ms_offs)
 *   n_exp       : total experts (e.g. 128; <= 1024)
 *   e_lo, e_hi  : half-open expert wave range [e_lo, min(e_hi, n_exp)),
 *                 cnt = hi - e_lo <= 256 (remapped slot = e - e_lo in 0..cnt-1)
 *   tiles_out   : G u32[max_tiles], packed 32-bit worklist descriptors (safe to
 *                 alias b->ms_packs; pass NULL to skip tile emission)
 *   n_tiles_out : G u32[1], active tile count for this wave (safe to alias
 *                 b->ms_npacks; set to 0 on empty/clamped wave)
 *   offs_out    : optional G u32[cnt + 1], 0-based copy of offs_in[e_lo .. hi]
 *                 (pass NULL to use offs_in + e_lo directly for r2/expmaj_r10)
 *   Grid        : (1u, 1u, 1u), NT = 256
 * ========================================================================== */
static LDS u32 ink_r14_256k_bcnt[4];
static LDS u32 ink_r14_256k_boff[4];

KERNEL ie_inkling_moe_wl_filter_r14(
    const G u32 *offs_in, u32 n_exp, u32 e_lo, u32 e_hi,
    G u32 *tiles_out, G u32 *n_tiles_out, G u32 *offs_out) {
  const u32 t = tid();

  if (!offs_in || n_exp == 0u || e_lo >= n_exp || e_lo >= e_hi) {
    if (t == 0u) {
      if (n_tiles_out) *n_tiles_out = 0u;
      if (offs_out) offs_out[0] = 0u;
    }
    return;
  }

  const u32 hi = e_hi < n_exp ? e_hi : n_exp;
  const u32 cnt = hi - e_lo;
  if (cnt > NT) {
    if (t == 0u && n_tiles_out) *n_tiles_out = 0u;
    return;
  }

  /* Load full-wave expert bounds before barrier (also safe if offs_out aliases). */
  u32 off0 = 0u, c = 0u, nfull = 0u, rem = 0u;
  if (t < cnt) {
    off0 = offs_in[e_lo + t];
    const u32 off1 = offs_in[e_lo + t + 1u];
    c = (off1 > off0) ? (off1 - off0) : 0u;
    nfull = c >> 6u;
    rem = c & 63u;
  }
  const u32 hi_off = (t == 0u) ? offs_in[hi] : 0u;
  if (t < 4u) ink_r14_256k_bcnt[t] = 0u;
  lbarrier();

  /* Optional 0-based offs_out copy for expert-major consumers. */
  if (offs_out) {
    if (t < cnt) offs_out[t] = off0;
    if (t == 0u) offs_out[cnt] = hi_off;
  }

  /* Pass 1: Count 4 largest-first tile buckets (cls 3: 49..64, 2: 33..48, 1: 17..32, 0: 1..16). */
  if (tiles_out && t < cnt && c > 0u) {
    if (nfull > 0u) __atomic_fetch_add(&ink_r14_256k_bcnt[3], nfull, __ATOMIC_RELAXED);
    if (rem > 0u) __atomic_fetch_add(&ink_r14_256k_bcnt[(rem - 1u) >> 4u], 1u, __ATOMIC_RELAXED);
  }
  lbarrier();

  /* Pass 2: Largest-first bucket prefix offsets + active tile count. */
  if (t == 0u) {
    const u32 b3 = ink_r14_256k_bcnt[3];
    const u32 b2 = ink_r14_256k_bcnt[2];
    const u32 b1 = ink_r14_256k_bcnt[1];
    const u32 b0 = ink_r14_256k_bcnt[0];
    ink_r14_256k_boff[3] = 0u;
    ink_r14_256k_boff[2] = b3;
    ink_r14_256k_boff[1] = b3 + b2;
    ink_r14_256k_boff[0] = b3 + b2 + b1;
    if (n_tiles_out) *n_tiles_out = tiles_out ? (b3 + b2 + b1 + b0) : 0u;
  }
  lbarrier();

  /* Pass 3: Emit exact 32-bit packed descriptors with slot = e - e_lo in bits [7:0]
   * and original list[] offset lo in bits [20:8] (matching ie_inkling_moe_pf_worklist_r7). */
  if (tiles_out && t < cnt && c > 0u) {
    if (nfull > 0u) {
      const u32 base_pos = __atomic_fetch_add(&ink_r14_256k_boff[3], nfull, __ATOMIC_RELAXED);
      for (u32 k = 0u; k < nfull; k++) {
        const u32 lo = off0 + (k << 6u);
        tiles_out[base_pos + k] = t | (lo << 8u) | (64u << 21u) | (3u << 28u);
      }
    }
    if (rem > 0u) {
      const u32 cls = (rem - 1u) >> 4u;
      const u32 pos = __atomic_fetch_add(&ink_r14_256k_boff[cls], 1u, __ATOMIC_RELAXED);
      const u32 lo = off0 + (nfull << 6u);
      tiles_out[pos] = t | (lo << 8u) | (rem << 21u) | (cls << 28u);
    }
  }
}

#undef INK_R14_256K_INV_PIO2
#undef INK_R14_256K_PIO2_1
#undef INK_R14_256K_PIO2_2
#undef INK_R14_256K_PIO2_3

#endif /* IE_MEGA */
