/* ie_gdn_lazy.inc -- lazy GDN state replay fused into ie_gdn1p (#3 device half).
 *
 * When gpu_accept / gpu_accept_tree defers ie_gdn_replay (`lazy_mask != 0`), the next
 * ie_gdn1p (or ie_gdn1 if desired) launch applies the accepted tokens' records directly
 * to `st[0..63]` in registers right after loading `S[c * slot + own + ii * 128u]`, saving
 * the standalone ie_gdn_replay global read + write of all GDN states (18 MB per accept).
 *
 * Integration in kernels/ie_kernels.c:
 *   1) `#include "ie_gdn_lazy.inc"` right above `ie_gdn1p` (~L4286).
 *   2) Append lazy parameters to `ie_gdn1p`:
 *        const G float *lazy_rec, u32 lazy_mask, unsigned long rls, u32 r_idx
 *      (Windows LLP64 host ABI note: `unsigned long` on amdgcn64 is 64-bit; pass `uint64_t rls`
 *       from `src/hip.c`!).
 *   3) Right after loading `st[0..63]` in `ie_gdn1p` (~L4293) -- BEFORE Phase 1 writes new
 *      records into `rec` at L4311/L4324:
 *        G float *S_slot_own = S + c * slot + own;
 *        for (u32 ii = 0; ii < 64u; ii++) st[ii] = (pos || lazy_mask) ? S_slot_own[ii * 128u] : 0.0f;
 *        ie_gdn_lazy_apply(st, S_slot_own, lazy_rec, lazy_mask, rls, r_idx, nv, h, half, j,
 *                          T > 0u && wfrom == 0u);
 *      Note: `ie_gdn_lazy_apply` ends with `barrier()` when `lazy_mask != 0`, so `lazy_rec`
 *      may safely alias `b->rec` in-place (single-buffered) without any WAR race when L4311/L4324
 *      subsequently write the new verify step's records into `rec`!
 *      Also note for #7 (dead FP32 output): guard L4412 with `if (out) out[t * os + h * 128u + jj] = y;`.
 *
 * Optional LDS Aliasing Guide for `ie_gdn1p` (0 extra VGPRs, 0 scratch spills, saves 16 KiB LDS ->
 * drops LDS from 54.6 KiB to 38.6 KiB so 3 workgroups/WGP fit in 128 KiB instead of 2):
 *   - Why `gp_a[16][256]` (16 KiB) can be eliminated completely:
 *     (a) Phase 1 (L4301 -> lbarrier() L4318 -> L4321): thread `t0` (`half = t0 >> 7`, `j = t0 & 127`)
 *         writes `gp_a[t][t0] = a;` and after L4318 reads back ONLY its own `gp_a[t][t0]` before
 *         writing `gp_k[t][j]` (`half == 1`) or `gp_q[t][j]` (`half == 0`). Both `gp_q` and `gp_k`
 *         are otherwise untouched before L4321. Replacing L4301 with:
 *           if (half) gp_k[t][j] = a; else gp_q[t][j] = a;
 *         and L4321 with:
 *           const float an = (half ? gp_k[t][j] : gp_q[t][j]) * (1.0f / __builtin_sqrtf(n2 + eps));
 *         reuses `gp_q` + `gp_k` in-place with zero cross-thread conflict and zero extra VGPRs.
 *     (b) Phase 2/3 output staging (L4364, L4399 -> lbarrier() L4401 -> L4406):
 *         In both the `GDN1P_FUSE` loop (after `lbarrier()` at L4363) and the `tr`/unfused loop
 *         (after `lbarrier()` at L4398), token `t`'s `gp_q[t][0..127]` is 100% dead (`tr` ancestor
 *         replay at L4374-4381 reads only `gp_bd`, `gp_d`, `gp_k`, never `gp_q`). So L4364 & L4399
 *         can write `if (!half) gp_q[t][j] = ...;` (stride 128, exact same indexing!) and L4406
 *         reads `gp_q[t][32u * q + l]` after `lbarrier()` at L4401, removing `gp_a` entirely.
 *     (Alternatively, keeping `gp_a` out of Phase 2/3 only or aliasing `gp_a[16][128]` onto `gp_q`
 *      after L4401 saves 8 KiB if Phase 1 is changed as above.)
 */

static inline void ie_gdn_lazy_apply(float st[64], G float *S_slot_own, const G float *lazy_rec, u32 lazy_mask,
                                     unsigned long rls, u32 r_idx, u32 nv, u32 h, u32 half, u32 j, int will_write_t0) {
  if (!lazy_mask || !lazy_rec) return;
  const G float *rp = lazy_rec + (unsigned long)r_idx * rls;
  for (u32 mk_ = lazy_mask; mk_; mk_ &= mk_ - 1u) {
    const u32 t = (u32)__builtin_ctz(mk_) - 1u;
    const G float *r = rp + ((unsigned long)t * nv + h) * GDN_RECF;
    const float decay = r[0], delta = r[GDN_RECD + j];
    const G float *k = r + GDN_RECK + half * 64u;
    for (u32 ii = 0; ii < 64u; ii++) {
      st[ii] = st[ii] * decay;
      st[ii] = __builtin_fmaf(k[ii], delta, st[ii]);
    }
  }
  if (!will_write_t0) {
    for (u32 ii = 0; ii < 64u; ii++) S_slot_own[ii * 128u] = st[ii];
  }
  barrier();
}
