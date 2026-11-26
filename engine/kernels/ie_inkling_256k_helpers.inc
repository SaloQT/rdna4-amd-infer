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

  /* Cast reduced |r| <= pi/4 to FP32 and evaluate degree-8 cos / degree-9 sin
   * (4 FP32 FMAs each, < 0.05 ULP error vs host (float)cos/(float)sin, avoids 1/32-rate FP64). */
  const float rf = (float)r;
  const float r2 = rf * rf;
  float c = __builtin_fmaf(r2, 1.0f / 40320.0f, -1.0f / 720.0f);
  c = __builtin_fmaf(r2, c, 1.0f / 24.0f);
  c = __builtin_fmaf(r2, c, -0.5f);
  c = __builtin_fmaf(r2, c, 1.0f);

  float s = __builtin_fmaf(r2, 1.0f / 362880.0f, -1.0f / 5040.0f);
  s = __builtin_fmaf(r2, s, 1.0f / 120.0f);
  s = __builtin_fmaf(r2, s, -1.0f / 6.0f);
  s = __builtin_fmaf(r2 * rf, s, rf);

  /* Branchless quadrant reconstruction: q = k & 3. */
  const u32 q = k & 3u;
  const float co = (q & 1u) ? -s : c;
  const float so = (q & 1u) ?  c : s;
  const float cf = (q & 2u) ? -co : co;
  const float sf = (q & 2u) ? -so : so;

  rcos[idx] = cf;
  rsin[idx] = sf;
}

static inline __attribute__((always_inline)) u32 ink_r14_256k_wave_scan_inc(u32 x) {
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x111, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x112, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x114, 0xF, 0xF, true);
  x += (u32)__builtin_amdgcn_update_dpp(0, (int)x, 0x118, 0xF, 0xF, true);
  const u32 lo16 = (u32)__builtin_amdgcn_readlane((int)x, 15);
  if (lane() >= 16u) x += lo16;
  return x;
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
static LDS u32 ink_r14_256k_wcnt[4][8];

KERNEL ie_inkling_moe_wl_filter_r14(
    const G u32 *offs_in, u32 n_exp, u32 e_lo, u32 e_hi,
    G u32 *tiles_out, G u32 *n_tiles_out, G u32 *offs_out) {
  const u32 t = tid();
  const u32 w = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = lane();

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

  /* Zero-atomic wave ballot + scan across the 4 largest-first tile buckets. */
  const int active = (tiles_out != (G u32 *)0) && (t < cnt) && (c > 0u);
  const int has_rem = active && (rem > 0u);
  const u32 rem_cls = has_rem ? ((rem - 1u) >> 4u) : 4u;
  const u32 lt_mask = (1u << l) - 1u;

  const u32 m0 = __builtin_amdgcn_ballot_w32(has_rem && rem_cls == 0u);
  const u32 m1 = __builtin_amdgcn_ballot_w32(has_rem && rem_cls == 1u);
  const u32 m2 = __builtin_amdgcn_ballot_w32(has_rem && rem_cls == 2u);
  const u32 c3 = (active ? nfull : 0u) + ((has_rem && rem_cls == 3u) ? 1u : 0u);
  const u32 inc3 = ink_r14_256k_wave_scan_inc(c3);

  if (l == 0u) {
    ink_r14_256k_wcnt[0][w] = (u32)__builtin_popcount(m0);
    ink_r14_256k_wcnt[1][w] = (u32)__builtin_popcount(m1);
    ink_r14_256k_wcnt[2][w] = (u32)__builtin_popcount(m2);
  }
  if (l == 31u) ink_r14_256k_wcnt[3][w] = inc3;
  lbarrier();

  /* Optional 0-based offs_out copy for expert-major consumers. */
  if (offs_out) {
    if (t < cnt) offs_out[t] = off0;
    if (t == 0u) offs_out[cnt] = hi_off;
  }

  /* Wave prefix sums + largest-first global bucket offsets (cls 3 -> 2 -> 1 -> 0). */
  u32 woff[4] = {0u, 0u, 0u, 0u};
  u32 btot[4] = {0u, 0u, 0u, 0u};
#pragma unroll
  for (u32 b = 0u; b < 4u; b++) {
#pragma unroll
    for (u32 j = 0u; j < 8u; j++) {
      const u32 v = ink_r14_256k_wcnt[b][j];
      if (j < w) woff[b] += v;
      btot[b] += v;
    }
  }
  const u32 base3 = 0u;
  const u32 base2 = btot[3];
  const u32 base1 = base2 + btot[2];
  const u32 base0 = base1 + btot[1];
  if (t == 0u && n_tiles_out) *n_tiles_out = tiles_out ? (base0 + btot[0]) : 0u;

  /* Emit exact 32-bit packed descriptors with slot = e - e_lo in bits [7:0]
   * and original list[] offset lo in bits [20:8] (zero LDS atomics). */
  if (active) {
    const u32 off3 = base3 + woff[3] + (inc3 - c3);
    for (u32 k = 0u; k < nfull; k++) {
      const u32 lo = off0 + (k << 6u);
      tiles_out[off3 + k] = t | (lo << 8u) | (64u << 21u) | (3u << 28u);
    }
    if (has_rem) {
      const u32 lo = off0 + (nfull << 6u);
      u32 pos;
      if (rem_cls == 3u) pos = off3 + nfull;
      else if (rem_cls == 2u) pos = base2 + woff[2] + (u32)__builtin_popcount(m2 & lt_mask);
      else if (rem_cls == 1u) pos = base1 + woff[1] + (u32)__builtin_popcount(m1 & lt_mask);
      else pos = base0 + woff[0] + (u32)__builtin_popcount(m0 & lt_mask);
      tiles_out[pos] = t | (lo << 8u) | (rem << 21u) | (rem_cls << 28u);
    }
  }
}

#undef INK_R14_256K_INV_PIO2
#undef INK_R14_256K_PIO2_1
#undef INK_R14_256K_PIO2_2
#undef INK_R14_256K_PIO2_3

#endif /* IE_MEGA */
