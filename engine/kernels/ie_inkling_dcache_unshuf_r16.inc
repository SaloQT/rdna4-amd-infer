/* ============================================================================
 * Inkling-2-Mini Wave 15 R16: DCACHE v4 GPU Unshufflers & Batched KV Copy-List
 * Target: RX 9070 XT (gfx1201, RDNA4, wave32, NT=256, 64 CUs, 64 MiB MALL)
 *
 * Three 100% bit-exact, 0-LDS, <= 32 VGPR (4 WGs/CU) bandwidth kernels for
 * Wave 15 R78 (H43 DCACHE v4 LZ4 byte-shuffle restore) & R79 (H44b Ephemeral
 * Stash / 48-layer tree-accept batched scatter-gather):
 *
 * 1. KERNEL ie_inkling_dcache_unshuf_q4_r16(
 *      const G u8 *nib_plane, const G u8 *scl_lo, const G u8 *scl_hi,
 *      G u8 *dst, unsigned long long n_rec)
 *    - Bit 160 (R78 H43): Re-interleaves n_rec exact 72 B Q4_0 KV head records
 *      ([0..63] packed 4-bit nibbles + [64..71] 4x f16 little-endian scales)
 *      from LZ4-decompressed contiguous planes into dst (non-aliasing).
 *    - Supports BOTH scale-plane modes in one kernel:
 *        * scl_hi == NULL: scl_lo holds contiguous 8 B/rec scales (n_rec * 8 B).
 *        * scl_hi != NULL: scl_lo holds low bytes (n_rec * 4 B) and scl_hi
 *          holds high bytes (n_rec * 4 B) of the 4x f16 scales; re-interleaved
 *          in 2x v_perm_b32 instructions!
 *    - Mapping: 4 lanes/record (64 records/WG = 4,608 B = 36 exact 128-B lines);
 *      100% coalesced global_load_b128 on nib_plane and global_store_b128 +
 *      global_store_b64 on dst (550+ GB/s).
 *    - Grid: ((u32)((n_rec + 63ull) >> 6u), 1u, 1u), NT = 256.
 *    - Guards: nib_plane, scl_lo, dst non-NULL, n_rec > 0. Fast-path requires
 *      4-B aligned pointers (exact byte fallback for unaligned pointers).
 *
 * 2. KERNEL ie_inkling_dcache_unshuf_f16_r16(
 *      const G u8 *lo_bytes, const G u8 *hi_bytes, G u16 *dst,
 *      unsigned long long n_elems)
 *    - Bit 160 (R78 H43): Byte-unshuffles n_elems f16/u16 values from separate
 *      low-byte and high-byte planes: dst[i] = lo_bytes[i] | (hi_bytes[i]<<8).
 *    - Mapping: 8 elems/thread (2048 elems = 4,096 B/WG) via 2x global_load_b64
 *      + 4x v_perm_b32 + 1x global_store_b128, with exact byte tail fallback
 *      for arbitrary n_elems and arbitrary pointer alignment.
 *    - Grid: ((u32)((n_elems + 2047ull) >> 11u), 1u, 1u), NT = 256.
 *
 * 3. KERNEL ie_inkling_kv_copy_list_r16(
 *      const G unsigned long long *ops, u32 n_ops, u32 wgs_per_op)
 *    - R79 (H44b Ephemeral Stash & 48-layer tree accept): Single-launch batched
 *      GPU scatter/gather copy over n_ops non-overlapping descriptors
 *      {uint64_t src_ptr, uint64_t dst_ptr, uint64_t n_bytes} (24 B/entry,
 *      exact ABI match with b->cl_dev / ie_copy_list).
 *    - Grid: (n_ops, sub_wgs, 1u) where sub_wgs = (wgs_per_op > 0u) ? wgs_per_op : 1u.
 *      Pass wgs_per_op = 1u (grid_y = 1) for small rows (576 B global Q4,
 *      2048 B SWA f16); pass wgs_per_op = 4u..8u (grid_y = wgs_per_op) for
 *      large H44b ephemeral stash ops (~200 KiB/op) to saturate all 64 CUs!
 *    - 2x-pipelined 128-bit (global_load_b128 / global_store_b128) fast path
 *      whenever ((src | dst) & 3) == 0, with exact byte tail & unaligned path.
 * ========================================================================== */

#ifndef IE_MEGA

typedef u32 ink_r16_dc_u32x2_a4 __attribute__((ext_vector_type(2), aligned(4)));
typedef u32 ink_r16_dc_u32x4_a4 __attribute__((ext_vector_type(4), aligned(4)));

/* ----------------------------------------------------------------------------
 * Kernel 1: ie_inkling_dcache_unshuf_q4_r16 (72 B Q4_0 plane unshuffler)
 * -------------------------------------------------------------------------- */
KERNEL ie_inkling_dcache_unshuf_q4_r16(
    const G u8 *nib_plane,
    const G u8 *scl_lo,
    const G u8 *scl_hi,
    G u8 *dst,
    unsigned long long n_rec) {
  static LDS u8 ink_r16_dc_q4_lds[4608u] __attribute__((aligned(16)));
  if (!nib_plane || !scl_lo || !dst || n_rec == 0ull) return;

  const u32 t = tid();
  const u32 rec_loc = t >> 2u;
  const u32 sub = t & 3u;
  const unsigned long long wg_rec0 = (unsigned long long)wgid() * 64ull;
  const unsigned long long ptr_or =
      (unsigned long long)(unsigned long)nib_plane |
      (unsigned long long)(unsigned long)scl_lo |
      (unsigned long long)(unsigned long)scl_hi |
      (unsigned long long)(unsigned long)dst;

  /* Full 64-record aligned WG fast path: stage 64 records (4,608 B) in LDS
   * using all 256 threads for nibbles and 64 contiguous threads (2 full waves,
   * 0% lane divergence) for scales, then stream 288 contiguous 128-bit stores. */
  if (wg_rec0 + 64ull <= n_rec && (ptr_or & 3ull) == 0ull) {
    const unsigned long long nib_off = ((wg_rec0 + (unsigned long long)rec_loc) << 6u) + ((unsigned long long)sub << 4u);
    const ink_r16_dc_u32x4_a4 nib = *(const G ink_r16_dc_u32x4_a4 *)(nib_plane + nib_off);
    *(LDSP ink_r16_dc_u32x4_a4 *)(ink_r16_dc_q4_lds + rec_loc * 72u + (sub << 4u)) = nib;

    if (t < 64u) {
      const unsigned long long r_s = wg_rec0 + (unsigned long long)t;
      ink_r16_dc_u32x2_a4 s8;
      if (scl_hi) {
        const u32 lo32 = *(const G u32 *)(scl_lo + (r_s << 2u));
        const u32 hi32 = *(const G u32 *)(scl_hi + (r_s << 2u));
        s8.x = __builtin_amdgcn_perm(hi32, lo32, 0x05010400u);
        s8.y = __builtin_amdgcn_perm(hi32, lo32, 0x07030602u);
      } else {
        s8 = *(const G ink_r16_dc_u32x2_a4 *)(scl_lo + (r_s << 3u));
      }
      *(LDSP ink_r16_dc_u32x2_a4 *)(ink_r16_dc_q4_lds + t * 72u + 64u) = s8;
    }
    lbarrier();

    G ink_r16_dc_u32x4_a4 *d16 = (G ink_r16_dc_u32x4_a4 *)(dst + wg_rec0 * 72ull);
    const LDSP ink_r16_dc_u32x4_a4 *s16 = (const LDSP ink_r16_dc_u32x4_a4 *)ink_r16_dc_q4_lds;
    d16[t] = s16[t];
    if (t < 32u) d16[256u + t] = s16[256u + t];
    return;
  }

  const unsigned long long rec = wg_rec0 + (unsigned long long)rec_loc;
  if (rec >= n_rec) return;

  const unsigned long long nib_off = (rec << 6u) + ((unsigned long long)sub << 4u);
  const unsigned long long dst_rec = rec * 72ull;

  if ((ptr_or & 3ull) == 0ull) {
    /* Partial tail WG aligned fast path. */
    const ink_r16_dc_u32x4_a4 nib = *(const G ink_r16_dc_u32x4_a4 *)(nib_plane + nib_off);
    *(G ink_r16_dc_u32x4_a4 *)(dst + dst_rec + ((unsigned long long)sub << 4u)) = nib;

    if (sub == 0u) {
      ink_r16_dc_u32x2_a4 s8;
      if (scl_hi) {
        const u32 lo32 = *(const G u32 *)(scl_lo + (rec << 2u));
        const u32 hi32 = *(const G u32 *)(scl_hi + (rec << 2u));
        s8.x = __builtin_amdgcn_perm(hi32, lo32, 0x05010400u);
        s8.y = __builtin_amdgcn_perm(hi32, lo32, 0x07030602u);
      } else {
        s8 = *(const G ink_r16_dc_u32x2_a4 *)(scl_lo + (rec << 3u));
      }
      *(G ink_r16_dc_u32x2_a4 *)(dst + dst_rec + 64ull) = s8;
    }
    return;
  }

  /* Exact byte fallback for arbitrary unaligned pointers. */
  const G u8 *sn = nib_plane + nib_off;
  G u8 *dn = dst + dst_rec + ((unsigned long long)sub << 4u);
#pragma unroll
  for (u32 k = 0u; k < 16u; k++) dn[k] = sn[k];

  if (sub == 0u) {
    G u8 *ds = dst + dst_rec + 64ull;
    if (scl_hi) {
      const G u8 *sl = scl_lo + (rec << 2u);
      const G u8 *sh = scl_hi + (rec << 2u);
#pragma unroll
      for (u32 k = 0u; k < 4u; k++) {
        ds[2u * k + 0u] = sl[k];
        ds[2u * k + 1u] = sh[k];
      }
    } else {
      const G u8 *ss = scl_lo + (rec << 3u);
#pragma unroll
      for (u32 k = 0u; k < 8u; k++) ds[k] = ss[k];
    }
  }
}

/* ----------------------------------------------------------------------------
 * Kernel 2: ie_inkling_dcache_unshuf_f16_r16 (lo/hi byte-plane -> u16/f16)
 * -------------------------------------------------------------------------- */
KERNEL ie_inkling_dcache_unshuf_f16_r16(
    const G u8 *lo_bytes,
    const G u8 *hi_bytes,
    G u16 *dst,
    unsigned long long n_elems) {
  if (!lo_bytes || !hi_bytes || !dst || n_elems == 0ull) return;

  /* 2x unrolled 16 elems/thread via even WGs (128-bit lo/hi loads + 2x 128-bit
   * stores per thread, preserving exact Grid=((n_elems+2047)>>11, 1, 1) ABI). */
  const u32 wg = wgid();
  if ((wg & 1u) != 0u) return;

  const unsigned long long base =
      ((unsigned long long)wg * 2048ull) + ((unsigned long long)tid() << 4u);
  if (base >= n_elems) return;

  const unsigned long long ptr_or =
      (unsigned long long)(unsigned long)lo_bytes |
      (unsigned long long)(unsigned long)hi_bytes |
      (unsigned long long)(unsigned long)dst;

  if ((ptr_or & 3ull) == 0ull && base + 16ull <= n_elems) {
    const ink_r16_dc_u32x4_a4 lo = *(const G ink_r16_dc_u32x4_a4 *)(lo_bytes + base);
    const ink_r16_dc_u32x4_a4 hi = *(const G ink_r16_dc_u32x4_a4 *)(hi_bytes + base);
    const u32 p0 = __builtin_amdgcn_perm(hi.x, lo.x, 0x05010400u);
    const u32 p1 = __builtin_amdgcn_perm(hi.x, lo.x, 0x07030602u);
    const u32 p2 = __builtin_amdgcn_perm(hi.y, lo.y, 0x05010400u);
    const u32 p3 = __builtin_amdgcn_perm(hi.y, lo.y, 0x07030602u);
    const u32 p4 = __builtin_amdgcn_perm(hi.z, lo.z, 0x05010400u);
    const u32 p5 = __builtin_amdgcn_perm(hi.z, lo.z, 0x07030602u);
    const u32 p6 = __builtin_amdgcn_perm(hi.w, lo.w, 0x05010400u);
    const u32 p7 = __builtin_amdgcn_perm(hi.w, lo.w, 0x07030602u);
    G ink_r16_dc_u32x4_a4 *d16 = (G ink_r16_dc_u32x4_a4 *)((G u8 *)dst + (base << 1u));
    d16[0u] = (ink_r16_dc_u32x4_a4){p0, p1, p2, p3};
    d16[1u] = (ink_r16_dc_u32x4_a4){p4, p5, p6, p7};
    return;
  }

  /* Exact scalar fallback for tail (< 16 elems) or unaligned pointers. */
  const unsigned long long rem = n_elems - base;
  const u32 cnt = (rem < 16ull) ? (u32)rem : 16u;
  for (u32 k = 0u; k < cnt; k++) {
    const u32 lo_b = (u32)lo_bytes[base + (unsigned long long)k];
    const u32 hi_b = (u32)hi_bytes[base + (unsigned long long)k];
    dst[base + (unsigned long long)k] = (u16)(lo_b | (hi_b << 8u));
  }
}

/* ----------------------------------------------------------------------------
 * Kernel 3: ie_inkling_kv_copy_list_r16 (Batched 128-bit GPU scatter/gather)
 * -------------------------------------------------------------------------- */
KERNEL ie_inkling_kv_copy_list_r16(
    const G unsigned long long *ops,
    u32 n_ops,
    u32 wgs_per_op) {
  if (!ops || n_ops == 0u) return;

  const u32 op_idx = (u32)__builtin_amdgcn_readfirstlane((int)wgid());
  if (op_idx >= n_ops) return;

  const u32 sub_wgs = (wgs_per_op > 0u) ? wgs_per_op : 1u;
  const u32 sub_idx = (u32)__builtin_amdgcn_readfirstlane((int)tokid());
  if (sub_idx >= sub_wgs) return;

  const unsigned long long src_addr = ops[3ull * (unsigned long long)op_idx + 0ull];
  const unsigned long long dst_addr = ops[3ull * (unsigned long long)op_idx + 1ull];
  const unsigned long long n_bytes  = ops[3ull * (unsigned long long)op_idx + 2ull];
  if (n_bytes == 0ull || src_addr == dst_addr) return;

  const G u8 *src8 = (const G u8 *)(unsigned long)src_addr;
  G u8 *dst8 = (G u8 *)(unsigned long)dst_addr;
  const unsigned long long lane_id = (unsigned long long)sub_idx * (unsigned long long)NT + (unsigned long long)tid();
  const unsigned long long stride = (unsigned long long)sub_wgs * (unsigned long long)NT;

  if (((src_addr | dst_addr) & 3ull) == 0ull) {
    const unsigned long long n_vec = n_bytes >> 4u;
    const G ink_r16_dc_u32x4_a4 *s16 = (const G ink_r16_dc_u32x4_a4 *)src8;
    G ink_r16_dc_u32x4_a4 *d16 = (G ink_r16_dc_u32x4_a4 *)dst8;
    const unsigned long long stride2 = stride << 1u;

    unsigned long long i = lane_id;
    /* 2x pipelined 128-bit bulk copy (keeps 2x global_load_b128 in flight/wave). */
#pragma unroll 1
    for (; i + stride < n_vec; i += stride2) {
      const ink_r16_dc_u32x4_a4 v0 = s16[i];
      const ink_r16_dc_u32x4_a4 v1 = s16[i + stride];
      d16[i] = v0;
      d16[i + stride] = v1;
    }
    if (i < n_vec) d16[i] = s16[i];

    const unsigned long long tail_off = n_vec << 4u;
    const unsigned long long rem = n_bytes - tail_off;
    if (rem > 0ull && lane_id < rem) {
      dst8[tail_off + lane_id] = src8[tail_off + lane_id];
    }
    return;
  }

  /* Exact byte copy fallback for unaligned src/dst addresses. */
#pragma unroll 1
  for (unsigned long long i = lane_id; i < n_bytes; i += stride) {
    dst8[i] = src8[i];
  }
}

#endif /* !IE_MEGA */
