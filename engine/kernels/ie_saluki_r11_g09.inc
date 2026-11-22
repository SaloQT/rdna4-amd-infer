/* ============================================================================
 * Round 11 Group 9: ie_saluki_lora_b_r11
 *   Bitwise-exact, 0-spill rank-32 LoRA-B projection for both prefill and
 *   verify/decode on RX 9070 XT (gfx1201, wave32, NT = 256, 32 rows/WG).
 *
 * Replaces ie_saluki_lora_b_t32 (1,148.9 ms in R10 trace: 743.6 ms prefill +
 * 405.3 ms verify across 101,888 launches) with five compounding wins:
 *   1. Adaptive t_tile (256 for rows >= 12288, 128 for rows >= 5120, 32 for
 *      rows < 5120): cuts redundant W loads 8x on 17408x32/12288x32 (35.65 MB
 *      -> 4.46 MB at T=512) and 4x on 5120x32, while aligning every Saluki
 *      prefill grid to an exact multiple of 64 CUs (1088, 768, 640, 512 WGs).
 *   2. Single branchless coalesced global_load_b128 per wave (4 rows x 32 cols
 *      = 512 B) -> 1x XOR-swizzled 0-bank-conflict ds_store_b128 into 4 KiB
 *      LDS -> lbarrier() -> 8x 0-bank-conflict ds_load_b128 into wr[32],
 *      eliminating ie_saluki_lora_b_t32's 4 branches and 4 serialized
 *      s_wait_loadcnt 0 weight-load stalls.
 *   3. Prologue overlap: wave's first-token x (2x s_load_b512 via TR8_CSP) and
 *      res (global_load_b32) are issued before lbarrier() so both stay in
 *      flight across lbarrier() and the 8x ds_load_b128 weight unpack (massive
 *      critical-path reduction on T=8 verify, 1 token/wave).
 *   4. IEEE-754 exact root-hoisted signed-zero canonicalization + 2-batch tree
 *      factorization: deletes 32 redundant (0.0f + ...) adds (16 wasted
 *      v_dual_add_f32 cycles/token, -33% inner-loop VALU) while preserving
 *      100% bitwise parity with ie_saluki_lora_b_t32 / ie_gemv_f32_tw under
 *      #pragma clang fp contract(off), and trims live temp VGPRs (<= 64 VGPR).
 *   5. Branchless 2x ping-pong software pipeline in the steady-state token
 *      loop: overlaps next token's 2x s_load_b512 + global_load_b32 (res)
 *      with current token's VOPD dot + store with zero CFG-merge wait stalls
 *      and zero SGPR copy moves.
 *
 * Included at the end of kernels/ie_saluki_kernels.c.
 * All local types/helpers/symbols are prefixed r11_g09_.
 * ========================================================================== */

typedef float r11_g09_f4 __attribute__((ext_vector_type(4)));

static inline __attribute__((always_inline)) void r11_g09_load_x32(const TR8_CSP u8 *xp, float dst[32]) {
  const TR8_CSP float *xf = (const TR8_CSP float *)xp;
#pragma unroll
  for (u32 c = 0u; c < 32u; c++) dst[c] = xf[c];
}

/* Exact 5-level binary reduction matching saluki_tree32_exact / wave_tree_f:
 *   a[i] = p[i] + p[i+16] (i=0..15)
 *   b[i] = a[i] + a[i+8]  (i=0..7)
 *   c[i] = b[i] + b[i+4]  (i=0..3)
 *   d[i] = c[i] + c[i+2]  (i=0..1)
 *   out  = 0.0f + (d[0] + d[1])
 * Under IEEE-754 RN with fp contract(off), 0.0f + tree32(w*x) is bitwise equal
 * to tree32(0.0f + w*x) for all inputs because (+0.0f) + z differs from z only
 * at z == -0.0f, and (-0.0f) survives an addition tree iff every leaf is -0.0f. */
static inline __attribute__((always_inline)) float r11_g09_dot32_exact(const float wr[32], const float xc[32]) {
#pragma clang fp contract(off)
  float b[8];
#pragma unroll
  for (u32 half = 0u; half < 8u; half += 4u) {
    float p0[4], p1[4], p2[4], p3[4];
#pragma unroll
    for (u32 k = 0u; k < 4u; k++) {
      const u32 i = half + k;
      p0[k] = wr[i] * xc[i];
      p2[k] = wr[i + 8u] * xc[i + 8u];
    }
#pragma unroll
    for (u32 k = 0u; k < 4u; k++) {
      const u32 i = half + k;
      p1[k] = wr[i + 16u] * xc[i + 16u];
      p3[k] = wr[i + 24u] * xc[i + 24u];
    }
    float a0[4], a1[4];
#pragma unroll
    for (u32 k = 0u; k < 4u; k++) {
      a0[k] = p0[k] + p1[k];
      a1[k] = p2[k] + p3[k];
    }
#pragma unroll
    for (u32 k = 0u; k < 4u; k++) b[half + k] = a0[k] + a1[k];
  }
  float c[4], d[2];
#pragma unroll
  for (u32 i = 0u; i < 4u; i++) c[i] = b[i] + b[i + 4u];
#pragma unroll
  for (u32 i = 0u; i < 2u; i++) d[i] = c[i] + c[i + 2u];
  return 0.0f + (d[0] + d[1]);
}

KERNEL ie_saluki_lora_b_r11(const G float *w, const G float *x, G float *y, const G float *res,
                            u32 rows, u32 T, u32 xs, u32 ys, u32 rs, u32 t_tile) {
#pragma clang fp contract(off)
  const u32 r0 = wgid() * 32u;
  const u32 tbase = tokid() * t_tile;
  if (t_tile == 0u || r0 >= rows || tbase >= T) return;

  static LDS r11_g09_f4 r11_g09_w[32][8] __attribute__((aligned(16)));

  const u32 tend = (t_tile < T - tbase) ? (tbase + t_tile) : T;
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane(tid() >> 5);
  const u32 l = lane();

  /* Stage 1: 1x coalesced global_load_b128 for 4 rows/wave of W + overlap
   * wave's token-0 scalar activation load (2x s_load_b512 on independent kmcnt). */
  const u32 rr = wv * 4u + (l >> 3);
  const u32 c4 = l & 7u;
  const u32 ri = (r0 + rr < rows) ? (r0 + rr) : (rows - 1u);
  const r11_g09_f4 w4 = *(const G r11_g09_f4 *)(w + (unsigned long)ri * 32u + c4 * 4u);

  const u32 p0 = tbase + wv;
  const u32 p0_safe = (p0 < tend) ? p0 : tbase;
  const TR8_CSP u8 *xptr = (const TR8_CSP u8 *)tr8_uniform((const G u8 *)x + (unsigned long)p0_safe * xs);
  float xa[32];
  r11_g09_load_x32(xptr, xa);

  __builtin_amdgcn_sched_barrier(0);
  /* 0-bank-conflict XOR-swizzled ds_store_b128 (4 KiB LDS, 16-byte aligned). */
  r11_g09_w[rr][c4 ^ (rr & 7u)] = w4;
  __builtin_amdgcn_sched_barrier(0);

  /* Issue token-0 residual load before lbarrier(): stays in flight across
   * lbarrier() (which only waits on dscnt) and the 8x ds_load_b128 unpack. */
  const u32 ro = r0 + l;
  const u32 ro_safe = (ro < rows) ? ro : (rows - 1u);
  const G u8 *rptr = (const G u8 *)res + (unsigned long)p0_safe * rs + (unsigned long)ro_safe * 4u;
  float rva = *(const G float *)rptr;

  lbarrier();
  if (p0 >= tend) return;
  if (ro >= rows) return;

  /* 8x 0-bank-conflict ds_load_b128 -> 32 transposed row weights per lane. */
  float wr[32];
  const u32 sw = l & 7u;
#pragma unroll
  for (u32 k = 0u; k < 8u; k++) {
    const r11_g09_f4 v = r11_g09_w[l][k ^ sw];
    wr[k * 4u + 0u] = v.x;
    wr[k * 4u + 1u] = v.y;
    wr[k * 4u + 2u] = v.z;
    wr[k * 4u + 3u] = v.w;
  }

  G u8 *yptr = (G u8 *)y + (unsigned long)p0 * ys + (unsigned long)ro * 4u;
  const unsigned long xs8 = (unsigned long)xs * 8ul;
  const unsigned long rs8 = (unsigned long)rs * 8ul;
  const unsigned long ys8 = (unsigned long)ys * 8ul;

  /* Steady-state 2x ping-pong software pipeline (zero branches inside body,
   * zero SGPR copy moves). For verify T <= 8, both checks are false and the
   * wave jumps directly to the single-token epilogue with xa/rva ready. */
  u32 p = p0;
  for (; tend - p > 16u; p += 16u) {
    xptr += xs8;
    rptr += rs8;
    float xb[32];
    r11_g09_load_x32(xptr, xb);
    const float rvb = *(const G float *)rptr;
    __builtin_amdgcn_sched_barrier(0);
    *(G float *)yptr = rva + r11_g09_dot32_exact(wr, xa);
    yptr += ys8;
    __builtin_amdgcn_sched_barrier(0);

    xptr += xs8;
    rptr += rs8;
    r11_g09_load_x32(xptr, xa);
    rva = *(const G float *)rptr;
    __builtin_amdgcn_sched_barrier(0);
    *(G float *)yptr = rvb + r11_g09_dot32_exact(wr, xb);
    yptr += ys8;
    __builtin_amdgcn_sched_barrier(0);
  }

  if (tend - p > 8u) {
    xptr += xs8;
    rptr += rs8;
    float xb[32];
    r11_g09_load_x32(xptr, xb);
    const float rvb = *(const G float *)rptr;
    __builtin_amdgcn_sched_barrier(0);
    *(G float *)yptr = rva + r11_g09_dot32_exact(wr, xa);
    yptr += ys8;
    __builtin_amdgcn_sched_barrier(0);
    *(G float *)yptr = rvb + r11_g09_dot32_exact(wr, xb);
  } else {
    *(G float *)yptr = rva + r11_g09_dot32_exact(wr, xa);
  }
}
