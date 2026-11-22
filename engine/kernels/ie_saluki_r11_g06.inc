/* ============================================================================
 * Round 11 Group 6 (`r11-g06-gdn-lin`): Bitwise-exact 3-Head GQA-Fused
 *   Chunked Gated DeltaNet WY-Representation Kernel for RX 9070 XT (gfx1201)
 *
 * Target: `ie_gdn_wy_wmma` (`347.2 ms` total / `326.4 ms` @ T=512, `n=3,312`
 *   in `logs/saluki-opt-r10-final-ab/traced_cand/summary.json`, completely
 *   unspecialized base kernel).
 *
 * Root causes fixed vs base `ie_gdn_wy_wmma` (`51 VGPR, 35,200 B LDS`):
 *   1. 3-Head GQA key-head sharing (`nk=16, nv=48`, `h = kh, kh+16, kh+32`):
 *      1 WG per `(chunk b, key head kh)` instead of 3 separate WGs. Cuts `qk`
 *      global reads, BF16-pair splitting, and all 24 `v_wmma` tiles (`K*K^T`
 *      and `Q*K^T`) by 3x; keeps `K` hot in CU L0/L1 across all 3 heads in
 *      Phase 8; eliminates 65-insn prologue software division via `wg >> 4`
 *      and `wg & 15`. (Grid `nb * 16`, bumped `253..256 -> 257` on host.)
 *   2. 4x occupancy via K_TILE=32 XOR-swizzled LDS union (`15,488 B <= 16 KiB`
 *      -> 4 WGs/WGP = 128 concurrent WGs vs base's `35,200 B` = 1 WG/WGP / 32
 *      WGs): zero padding, zero bank conflicts on both `ds_store_b64` and
 *      `ds_load_b128` (`c8_swz = c8 ^ (r & 6)`, `cc_swz = cc ^ ((r >> 1) & 3)`).
 *   3. Exact Wave 0..2 `readlane` prefix sum (`G[0..2]`) + single-pass
 *      `expG[g][l] = expf(G[g][l])` (96 vs 12,288 `expf` calls -> deletes all
 *      32 redundant 30-insn `expf` blocks from Phase 8).
 *   4. Zero-overhead Waves 6..7 overlap during WMMA: Waves 6 & 7 compute all
 *      496 strictly lower-triangular `dec = expf(G[g][r] - G[g][c])` per head
 *      in parallel with Waves 0..5's WMMA quarters `kq = 0..2` (stored in
 *      disjoint `dec0/dec1/dec2` LDS, eliminating duplicate `expf` across `L`
 *      and `Mb` and all serial `s_wait_dscnt 0` loops), and zero upper-right
 *      `Mb` tiles for all 3 heads via 3 coalesced `global_store_b128`s in `kq=3`.
 *   5. Phase 8 dual-column `(U_g, W_g)` forward substitution storing `-L` in
 *      LDS (`r11_g06_negf` exact bit-31 flip): halves `negL` LDS reads via
 *      16B-aligned `ds_load_b128`, unlocks `v_dual_fmac_f32` (2x FMA issue
 *      rate on gfx1201), uses 4-row ring prefetch (`pv[4]/pk[4]`) on the
 *      branchless `full` (`n==32`) path with `restrict` pointers + SGPR base +
 *      compile-time immediate offsets (`0x6000`/`0x4000` strides -> deletes
 *      ~900 64-bit address-math insns and 32 serial `s_wait_loadcnt 0` stalls).
 *
 * Included at the end of `kernels/ie_saluki_kernels.c` by `r11-wiring`.
 * All local types/helpers/LDS/macros are prefixed `r11_g06_` / `R11_G06_`;
 * all macros are `#undef`'d at EOF. 100% bitwise identical (0 bit diffs) to
 * base `ie_gdn_wy_wmma` across all `T >= 1` (`n = 1..32`).
 * ========================================================================== */

#define R11_G06_NK 16u
#define R11_G06_NV 48u
#define R11_G06_GC 32u
#define R11_G06_VSTR 6144u /* R11_G06_NV * 128u */
#define R11_G06_KSTR 4096u /* R11_G06_NK * 256u */

typedef float r11_g06_f2 __attribute__((ext_vector_type(2)));

/* Disjoint phase lifetimes inside 15,488 B LDS (<= 16,384 B -> 4 WGs/WGP):
 *   Phase 0..4: u.ph4.qk (0..8191), u.ph4.dec0 (8192..10239),
 *               u.ph4.dec1 (10240..12287), dec2 (12288..14335),
 *               G/b/expG (14336..15487).
 *   Phase 6a:   Waves 0..2 read dec0/dec1 & write u.ph8.negL[0..1] (0..8191);
 *               Waves 3..5 read dec0/dec1/dec2 & write global Mb[0..2].
 *   Phase 6b:   Waves 0..2 read dec2 (12288..14335) & write u.ph8.negL[2]
 *               (8192..12287).
 *   Phase 8:    All waves read u.ph8.negL[0..2] (0..12287) + b/expG. */
typedef union {
  struct {
    struct {
      __bf16 qh[R11_G06_GC][32]; /*  2,048 B */
      __bf16 ql[R11_G06_GC][32]; /*  2,048 B */
      __bf16 kh[R11_G06_GC][32]; /*  2,048 B */
      __bf16 kl[R11_G06_GC][32]; /*  2,048 B */
    } qk;                        /*  8,192 B (bytes     0 ..  8,191) */
    float dec0[512];             /*  2,048 B (bytes 8,192 .. 10,239) */
    float dec1[512];             /*  2,048 B (bytes 10,240 .. 12,287) */
  } ph4;                         /* 12,288 B */
  struct {
    float negL[3][R11_G06_GC][R11_G06_GC]; /* 12,288 B (bytes 0 .. 12,287) */
  } ph8;
} r11_g06_wy_u_t;

typedef struct {
  r11_g06_wy_u_t u;            /* 12,288 B (bytes      0 .. 12,287) */
  float dec2[512];             /*  2,048 B (bytes 12,288 .. 14,335) */
  float Gs[3][R11_G06_GC];     /*    384 B (bytes 14,336 .. 14,719) */
  float b[3][R11_G06_GC];      /*    384 B (bytes 14,720 .. 15,103) */
  float expG[3][R11_G06_GC];   /*    384 B (bytes 15,104 .. 15,487) */
} r11_g06_wy_lds_t;

static LDS r11_g06_wy_lds_t r11_g06_wy_lds __attribute__((aligned(32)));
_Static_assert(sizeof(r11_g06_wy_lds_t) == 15488u,
               "r11_g06_wy_lds_t must stay <= 16384 B for 4 WGs/WGP occupancy");

/* Exact bit-31 sign flip matching VOP3 `v_fma_f32 -L, x, rhs` source negation
 * for every IEEE-754 value (incl. +-0.0f) while enabling VOPD `v_dual_fmac_f32`. */
static inline __attribute__((always_inline)) float r11_g06_negf(float x) {
  return __builtin_bit_cast(float, __builtin_bit_cast(u32, x) ^ 0x80000000u);
}

/* Exact BF16-pair RNE splitter via v_perm_b32 (extracts scalar locals before
 * __builtin_bit_cast to avoid the Clang vector-element bitcast codegen bug). */
static inline __attribute__((always_inline)) void r11_g06_split_f4(f4 x, gw_u2 *hi, gw_u2 *lo) {
  const float x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
  const u32 u0 = __builtin_bit_cast(u32, x0), u1 = __builtin_bit_cast(u32, x1);
  const u32 u2 = __builtin_bit_cast(u32, x2), u3 = __builtin_bit_cast(u32, x3);
  const u32 h0r = u0 + 0x7FFFu + ((u0 >> 16u) & 1u);
  const u32 h1r = u1 + 0x7FFFu + ((u1 >> 16u) & 1u);
  const u32 h2r = u2 + 0x7FFFu + ((u2 >> 16u) & 1u);
  const u32 h3r = u3 + 0x7FFFu + ((u3 >> 16u) & 1u);
  const float d0 = x0 - __builtin_bit_cast(float, h0r & 0xFFFF0000u);
  const float d1 = x1 - __builtin_bit_cast(float, h1r & 0xFFFF0000u);
  const float d2 = x2 - __builtin_bit_cast(float, h2r & 0xFFFF0000u);
  const float d3 = x3 - __builtin_bit_cast(float, h3r & 0xFFFF0000u);
  const u32 r0 = __builtin_bit_cast(u32, d0), r1 = __builtin_bit_cast(u32, d1);
  const u32 r2 = __builtin_bit_cast(u32, d2), r3 = __builtin_bit_cast(u32, d3);
  const u32 l0r = r0 + 0x7FFFu + ((r0 >> 16u) & 1u);
  const u32 l1r = r1 + 0x7FFFu + ((r1 >> 16u) & 1u);
  const u32 l2r = r2 + 0x7FFFu + ((r2 >> 16u) & 1u);
  const u32 l3r = r3 + 0x7FFFu + ((r3 >> 16u) & 1u);
  *hi = (gw_u2){__builtin_amdgcn_perm(h1r, h0r, 0x07060302u), __builtin_amdgcn_perm(h3r, h2r, 0x07060302u)};
  *lo = (gw_u2){__builtin_amdgcn_perm(l1r, l0r, 0x07060302u), __builtin_amdgcn_perm(l3r, l2r, 0x07060302u)};
}

/* Populate negated strictly lower-triangular L[g] in LDS with exact `(b[r]*dec)*acc[v]` order. */
static inline __attribute__((always_inline)) void r11_g06_write_negL(
    const u32 g, const LDSP float *dec_p, const gw_f8 acc, const u32 r0, const u32 c, const u32 n) {
  const f32x4 b0 = *(const LDSP f32x4 *)(r11_g06_wy_lds.b[g] + r0);
  const f32x4 b1 = *(const LDSP f32x4 *)(r11_g06_wy_lds.b[g] + r0 + 4u);
  const float bv[8] = {b0[0], b0[1], b0[2], b0[3], b1[0], b1[1], b1[2], b1[3]};
#pragma unroll
  for (u32 v = 0u; v < 8u; v++) {
    const u32 r = r0 + v;
    if (r < n && c < r) {
      const float dec = dec_p[((r * (r - 1u)) >> 1u) + c];
      const float lv = (bv[v] * dec) * acc[v];
      r11_g06_wy_lds.u.ph8.negL[g][r][c] = r11_g06_negf(lv);
    }
  }
}

/* Dual-column forward substitution `(I + L)^-1 [beta*V | beta*exp(G)*K]` for
 * one value head `g`, computing `U[:, j]` and `W[:, j]` simultaneously while
 * sharing every `ds_load_b128` of `negL[g][r][0..r-1]`. Bitwise identical to
 * base `ie_gdn_wy_wmma`'s ascending `v_fma_f32 -L[r][c], x[c], rhs` chain. */
#define R11_G06_DUAL_FMA_ROW(r_idx, nl_row, rhsU, rhsW, xU, xW)                                                     \
  do {                                                                                                              \
    _Pragma("unroll") for (u32 c_ = 0u; c_ + 4u <= (r_idx); c_ += 4u) {                                             \
      const f32x4 nl4_ = *(const LDSP f32x4 *)((nl_row) + c_);                                                      \
      (rhsU) = __builtin_fmaf(nl4_[0], (xU)[c_ + 0u], (rhsU));                                                      \
      (rhsW) = __builtin_fmaf(nl4_[0], (xW)[c_ + 0u], (rhsW));                                                      \
      (rhsU) = __builtin_fmaf(nl4_[1], (xU)[c_ + 1u], (rhsU));                                                      \
      (rhsW) = __builtin_fmaf(nl4_[1], (xW)[c_ + 1u], (rhsW));                                                      \
      (rhsU) = __builtin_fmaf(nl4_[2], (xU)[c_ + 2u], (rhsU));                                                      \
      (rhsW) = __builtin_fmaf(nl4_[2], (xW)[c_ + 2u], (rhsW));                                                      \
      (rhsU) = __builtin_fmaf(nl4_[3], (xU)[c_ + 3u], (rhsU));                                                      \
      (rhsW) = __builtin_fmaf(nl4_[3], (xW)[c_ + 3u], (rhsW));                                                      \
    }                                                                                                               \
    _Pragma("unroll") for (u32 c_ = (r_idx) & ~3u; c_ < (r_idx); c_++) {                                            \
      const float nl_ = (nl_row)[c_];                                                                               \
      (rhsU) = __builtin_fmaf(nl_, (xU)[c_], (rhsU));                                                               \
      (rhsW) = __builtin_fmaf(nl_, (xW)[c_], (rhsW));                                                               \
    }                                                                                                               \
  } while (0)

KERNEL ie_saluki_gdn_wy_wmma_r11(const G float *restrict qk, const G float *restrict vv, const G float *restrict bg,
                                 G float *restrict U, G float *restrict W, G float *restrict Mb, G float *restrict Gc,
                                 u32 T, u32 nk, u32 nv) {
  (void)nk;
  (void)nv;
  const u32 wg = wgid();
  const u32 nb = (T + 31u) >> 5u;
  if (T == 0u || wg >= nb * R11_G06_NK) return;

  const u32 b = wg >> 4u;
  const u32 kh = wg & 15u;
  const u32 t0 = b * R11_G06_GC;
  const u32 n = (T - t0 < R11_G06_GC) ? (T - t0) : R11_G06_GC;
  const int full = (n == R11_G06_GC);

  const u32 th = tid();
  const u32 w = __builtin_amdgcn_readfirstlane(th >> 5u);
  const u32 l = lane();
  const u32 hl = l >> 4u;
  const u32 rl = l & 15u;

  /* Phase 0: Issue all 4 quarters of Q and K loads (8x global_load_b128) up front
   * so global memory latency overlaps Waves 0..2's bg load + prefix-sum + expG. */
  const u32 qr = th >> 3u;
  const u32 qc8 = th & 7u;
  const u32 qc4_swz = 4u * (qc8 ^ (qr & 6u));
  f4 q4[4], k4[4];
  {
    const G float *restrict q_row = qk + ((unsigned long)(t0 + qr) * R11_G06_NK + kh) * 256u + 4u * qc8;
    if (full) {
#pragma unroll
      for (u32 kq = 0u; kq < 4u; kq++) {
        q4[kq] = *(const G f4 *)(q_row + 32u * kq);
        k4[kq] = *(const G f4 *)(q_row + 128u + 32u * kq);
      }
    } else {
      const int rok = (qr < n);
#pragma unroll
      for (u32 kq = 0u; kq < 4u; kq++) {
        q4[kq] = rok ? *(const G f4 *)(q_row + 32u * kq) : (f4){0.0f, 0.0f, 0.0f, 0.0f};
        k4[kq] = rok ? *(const G f4 *)(q_row + 128u + 32u * kq) : (f4){0.0f, 0.0f, 0.0f, 0.0f};
      }
    }
  }

  /* Phase 0b: Waves 0..2 load (beta, log_g) for all 3 value heads h = kh + 16*w,
   * compute exact left-to-right sequential prefix sum G[w][0..31] in registers
   * via readlane, precompute expG[w][l] once, and write Gc. */
  if (w < 3u) {
    const u32 h = kh + R11_G06_NK * w;
    float b_l = 0.0f, lg_l = 0.0f;
    if (full || l < n) {
      const r11_g06_f2 bgv = *(const G r11_g06_f2 *)(bg + ((unsigned long)(t0 + l) * R11_G06_NV + h) * 2u);
      b_l = bgv[0];
      lg_l = bgv[1];
    }
    float G0 = 0.0f;
#pragma unroll
    for (u32 r = 0u; r < R11_G06_GC; r++) {
      const float lg_r = I2F(__builtin_amdgcn_readlane(F2I(lg_l), (int)r));
      if (r <= l) G0 += lg_r;
    }
    r11_g06_wy_lds.Gs[w][l] = G0;
    r11_g06_wy_lds.b[w][l] = b_l;
    r11_g06_wy_lds.expG[w][l] = (full || l < n) ? __builtin_expf(G0) : 0.0f;
    if (full || l < n) Gc[(unsigned long)(t0 + l) * R11_G06_NV + h] = G0;
  }

  /* Phase 4: Shared K*K^T (Waves 0..2) and Q*K^T (Waves 3..5) across 4 K_TILE=32
   * quarters (exact kd = 0..7 accumulation order), overlapped with Waves 6..7
   * computing all 496 strictly lower-tri expf(G[g][r]-G[g][c]) for g=0,1,2 and
   * zeroing upper-right (0,1) Mb tiles for all 3 heads in kq=3. */
  gw_f8 acc = (gw_f8)(0.0f);
  const u32 is_qk = (w >= 3u) ? 1u : 0u;
  const u32 ti = is_qk ? (w - 3u) : w;
  const u32 tr = (ti >= 1u) ? 1u : 0u;
  const u32 tc = (ti == 2u) ? 1u : 0u;
  const int active_wmma = (w < 6u) && (tr == 0u || n > 16u);

  const u32 rr_a = 16u * tr + rl;
  const u32 rr_b = 16u * tc + rl;
  const u32 swz_a = (rr_a >> 1u) & 3u;
  const u32 swz_b = (rr_b >> 1u) & 3u;

#pragma unroll
  for (u32 kq = 0u; kq < 4u; kq++) {
    {
      gw_u2 qhi, qlo, khi, klo;
      r11_g06_split_f4(q4[kq], &qhi, &qlo);
      r11_g06_split_f4(k4[kq], &khi, &klo);
      *(LDSP gw_u2 *)&r11_g06_wy_lds.u.ph4.qk.qh[qr][qc4_swz] = qhi;
      *(LDSP gw_u2 *)&r11_g06_wy_lds.u.ph4.qk.ql[qr][qc4_swz] = qlo;
      *(LDSP gw_u2 *)&r11_g06_wy_lds.u.ph4.qk.kh[qr][qc4_swz] = khi;
      *(LDSP gw_u2 *)&r11_g06_wy_lds.u.ph4.qk.kl[qr][qc4_swz] = klo;
    }
    lbarrier();

    if (active_wmma) {
      const LDSP __bf16 (*ah_p)[32] = is_qk ? (const LDSP __bf16 (*)[32])r11_g06_wy_lds.u.ph4.qk.qh
                                            : (const LDSP __bf16 (*)[32])r11_g06_wy_lds.u.ph4.qk.kh;
      const LDSP __bf16 (*al_p)[32] = is_qk ? (const LDSP __bf16 (*)[32])r11_g06_wy_lds.u.ph4.qk.ql
                                            : (const LDSP __bf16 (*)[32])r11_g06_wy_lds.u.ph4.qk.kl;
#pragma unroll
      for (u32 kd_l = 0u; kd_l < 2u; kd_l++) {
        const u32 cc = 2u * kd_l + hl;
        const u32 ca = 8u * (cc ^ swz_a);
        const u32 cb = 8u * (cc ^ swz_b);
        const gw_bf8 ah = *(const LDSP gw_bf8 *)&ah_p[rr_a][ca];
        const gw_bf8 al = *(const LDSP gw_bf8 *)&al_p[rr_a][ca];
        const gw_bf8 bh = *(const LDSP gw_bf8 *)&r11_g06_wy_lds.u.ph4.qk.kh[rr_b][cb];
        const gw_bf8 bl = *(const LDSP gw_bf8 *)&r11_g06_wy_lds.u.ph4.qk.kl[rr_b][cb];
        acc = gw_wmma_pair(ah, al, bh, bl, acc);
      }
    } else if (w >= 6u) {
      const u32 w67 = w - 6u;
      if (kq < 3u) {
        LDSP float *dec_dst = (kq == 0u) ? r11_g06_wy_lds.u.ph4.dec0
                            : (kq == 1u) ? r11_g06_wy_lds.u.ph4.dec1
                                         : r11_g06_wy_lds.dec2;
        if (l < 31u) {
#pragma unroll
          for (u32 it = 0u; it < 8u; it++) {
            const u32 row = 8u * w67 + it;
            const int low = (l < row);
            const u32 r = low ? row : (31u - row);
            const u32 c = low ? l : (30u - l);
            const u32 idx = ((r * (r - 1u)) >> 1u) + c;
            const float gr = low ? r11_g06_wy_lds.Gs[kq][row] : r11_g06_wy_lds.Gs[kq][31u - row];
            const float gc = r11_g06_wy_lds.Gs[kq][c];
            const float ev = __builtin_expf(gr - gc);
            dec_dst[idx] = (full || r < n) ? ev : 0.0f;
          }
        }
      } else {
        /* kq == 3: zero upper-right (0,1) tile of Mb for all 3 heads via f4 stores. */
        const u32 t67 = 32u * w67 + l;
        const u32 mr = t67 >> 2u;
        const u32 mc4 = 16u + 4u * (t67 & 3u);
        const f4 z4 = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (u32 g = 0u; g < 3u; g++) {
          G float *restrict Mb_h = Mb + ((unsigned long)b * R11_G06_NV + (kh + R11_G06_NK * g)) * 1024u;
          *(G f4 *)(Mb_h + mr * R11_G06_GC + mc4) = z4;
        }
      }
    }
    lbarrier();
  }

  /* Phase 6a: Waves 0..2 materialize negL[0..1] into bytes 0..8191 (reading
   * dec0/dec1 at 8192..12287); Waves 3..5 materialize global Mb[0..2] (reading
   * dec0/dec1/dec2 at 8192..14335). Exact `0.0f * acc[v]` preserves -0.0f. */
  if (w < 3u) {
    const u32 r0 = 16u * tr + 8u * hl;
    const u32 c = 16u * tc + rl;
    r11_g06_write_negL(0u, r11_g06_wy_lds.u.ph4.dec0, acc, r0, c, n);
    r11_g06_write_negL(1u, r11_g06_wy_lds.u.ph4.dec1, acc, r0, c, n);
  } else if (w < 6u) {
    const u32 r0 = 16u * tr + 8u * hl;
    const u32 c = 16u * tc + rl;
#pragma unroll
    for (u32 g = 0u; g < 3u; g++) {
      const LDSP float *dec_p = (g == 0u) ? r11_g06_wy_lds.u.ph4.dec0
                              : (g == 1u) ? r11_g06_wy_lds.u.ph4.dec1
                                          : r11_g06_wy_lds.dec2;
      G float *restrict Mb_r0 =
          Mb + ((unsigned long)b * R11_G06_NV + (kh + R11_G06_NK * g)) * 1024u + r0 * R11_G06_GC + c;
#pragma unroll
      for (u32 v = 0u; v < 8u; v++) {
        const u32 r = r0 + v;
        float dec = 0.0f;
        if ((full || r < n) && c <= r) {
          dec = (c == r) ? 1.0f : dec_p[((r * (r - 1u)) >> 1u) + c];
        }
        Mb_r0[v * R11_G06_GC] = dec * acc[v];
      }
    }
  }
  lbarrier();

  /* Phase 6b: Waves 0..2 materialize negL[2] into bytes 8192..12287 (reading
   * disjoint dec2 at 12288..14335). */
  if (w < 3u) {
    const u32 r0 = 16u * tr + 8u * hl;
    const u32 c = 16u * tc + rl;
    r11_g06_write_negL(2u, r11_g06_wy_lds.dec2, acc, r0, c, n);
  }
  lbarrier();

  /* Phase 8: Dual-column `(U_g, W_g)` forward substitution across all 3 heads.
   *   pass 0: Waves 0..3 solve (U_0, W_0), Waves 4..7 solve (U_1, W_1).
   *   pass 1: Waves 0..3 solve (U_2, W_2); Waves 4..7 exit early (no barriers). */
  const u32 j = 32u * (w & 3u) + l;
  const G float *restrict k_sgpr = qk + ((unsigned long)t0 * R11_G06_NK + kh) * 256u + 128u;

  if (full) {
#pragma nounroll
    for (u32 pass = 0u; pass < 2u; pass++) {
      if (pass == 1u && w >= 4u) break;
      const u32 g = (pass == 0u) ? (w >> 2u) : 2u;
      const u32 h = kh + R11_G06_NK * g;

      const G float *restrict v_sgpr = vv + ((unsigned long)t0 * R11_G06_NV + h) * 128u;
      G float *restrict u_sgpr = U + ((unsigned long)t0 * R11_G06_NV + h) * 128u;
      G float *restrict w_sgpr = W + ((unsigned long)t0 * R11_G06_NV + h) * 128u;
      const LDSP float *nl_base = &r11_g06_wy_lds.u.ph8.negL[g][0][0];
      const LDSP float *b_base = r11_g06_wy_lds.b[g];
      const LDSP float *eg_base = r11_g06_wy_lds.expG[g];

      /* Branchless 4-row ring-prefetched full-chunk fast path (0 CFG merges ->
       * FIFO `s_wait_loadcnt` overlap, compile-time immediate offsets, 0 dead
       * vector PHIs from the partial-chunk tail path). */
      float xU[R11_G06_GC], xW[R11_G06_GC];
      float pv[4], pk[4];
#pragma unroll
      for (u32 p = 0u; p < 4u; p++) {
        pv[p] = v_sgpr[j + R11_G06_VSTR * p];
        pk[p] = k_sgpr[j + R11_G06_KSTR * p];
      }
#pragma unroll
      for (u32 r = 0u; r < R11_G06_GC; r++) {
        const float br = b_base[r];
        const float egr = eg_base[r];
        float rhsU = br * pv[r & 3u];
        const float ek = egr * pk[r & 3u];
        float rhsW = br * ek;
        if (r < R11_G06_GC - 4u) {
          pv[r & 3u] = v_sgpr[j + R11_G06_VSTR * (r + 4u)];
          pk[r & 3u] = k_sgpr[j + R11_G06_KSTR * (r + 4u)];
        }
        const LDSP float *nl_row = nl_base + r * R11_G06_GC;
        R11_G06_DUAL_FMA_ROW(r, nl_row, rhsU, rhsW, xU, xW);
        xU[r] = rhsU;
        xW[r] = rhsW;
        u_sgpr[j + R11_G06_VSTR * r] = rhsU;
        w_sgpr[j + R11_G06_VSTR * r] = rhsW;
      }
    }
  } else {
    /* Partial-chunk tail path (`1 <= n <= 31`): clean rolled loops (0 unroll
     * warnings, 0 scratch via AMDGPU vector alloca promotion, isolated from full). */
#pragma nounroll
    for (u32 pass = 0u; pass < 2u; pass++) {
      if (pass == 1u && w >= 4u) break;
      const u32 g = (pass == 0u) ? (w >> 2u) : 2u;
      const u32 h = kh + R11_G06_NK * g;

      const G float *restrict v_sgpr = vv + ((unsigned long)t0 * R11_G06_NV + h) * 128u;
      G float *restrict u_sgpr = U + ((unsigned long)t0 * R11_G06_NV + h) * 128u;
      G float *restrict w_sgpr = W + ((unsigned long)t0 * R11_G06_NV + h) * 128u;
      const LDSP float *nl_base = &r11_g06_wy_lds.u.ph8.negL[g][0][0];
      const LDSP float *b_base = r11_g06_wy_lds.b[g];
      const LDSP float *eg_base = r11_g06_wy_lds.expG[g];

      float xU[R11_G06_GC], xW[R11_G06_GC];
      for (u32 r = 0u; r < n; r++) {
        const float vr = v_sgpr[j + R11_G06_VSTR * r];
        const float kr = k_sgpr[j + R11_G06_KSTR * r];
        const float br = b_base[r];
        const float egr = eg_base[r];
        float rhsU = br * vr;
        const float ek = egr * kr;
        float rhsW = br * ek;
        const LDSP float *nl_row = nl_base + r * R11_G06_GC;
        for (u32 c = 0u; c < r; c++) {
          const float nl = nl_row[c];
          rhsU = __builtin_fmaf(nl, xU[c], rhsU);
          rhsW = __builtin_fmaf(nl, xW[c], rhsW);
        }
        xU[r] = rhsU;
        xW[r] = rhsW;
        u_sgpr[j + R11_G06_VSTR * r] = rhsU;
        w_sgpr[j + R11_G06_VSTR * r] = rhsW;
      }
    }
  }
}

#undef R11_G06_DUAL_FMA_ROW
#undef R11_G06_KSTR
#undef R11_G06_VSTR
#undef R11_G06_GC
#undef R11_G06_NV
#undef R11_G06_NK
