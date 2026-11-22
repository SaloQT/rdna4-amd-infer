/* ie_attn_long_mp_q8.inc -- single-KV-pass multi-pair-group long attention for IE_KV_Q8 (hd 256).
 *
 * At T = 3 .. 8 verify tokens (e.g. mtp8(7): T = 8, grp = 6, npair = 48, npg = 3), ie_attn_long launches
 * one workgroup per pair group (nwg = n_kv * smx * npg), reading the entire 80K+ Q8 KV cache npg = 2 .. 3
 * separate times per verify step.
 *
 * ie_attn_long_mp launches nwg = n_kv * smx workgroups (all npg <= 3 pair groups, up to 48 (token, head)
 * pairs per KV head, inside one workgroup) while fitting in <= 32 KiB LDS (31,936 B -> 2 WGs/WGP = 16 waves/WGP)
 * without scratch spills:
 *   1) Loads the 128-key K slice ONCE per split, streams Q_pg through mp_lds for pg = 0 .. npg - 1, and lets
 *      the 36 K VGPRs die before V loads begin.
 *   2) Prefetches V tile 0 across Phase 2's softmax, stages P_pg (hi + lo fp16 planes, 3 x 8,704 = 26,112 B)
 *      into mp_lds, and lets s_pg die.
 *   3) Streams V tiles kt = 0 .. 7 ONCE with 1-tile software-pipelined global prefetch (10 V VGPRs instead of 40)
 *      and a 2-pass 16x16 wave-private LDS transpose (4,096 B across 8 waves), reusing (b0, b1, vs, w_sc)
 *      across all active pair groups pg = 0 .. npg - 1.
 *   4) Writes partials and runs the exact two-level gcount / count / al_msum / al_final merge across all
 *      16 * npg pairs, producing outputs bitwise identical to ie_attn_long.
 */
#if defined(IE_KV_Q8) && !defined(IE_MEGA) && !defined(IE_KV_F32)

#define MP_PG_HALVES (32u * AL_PS)                        /* 16 rows ph + 16 rows pl per pair group (4,352 halves = 8,704 B) */
#define MP_OFF_SC (3u * MP_PG_HALVES * 2u)                /* 26,112 B: P staging for pg = 0, 1, 2 */
#define MP_BYTES (MP_OFF_SC + 8u * 16u * AL_SS * 2u)      /* + 4,096 B (8 waves x 16 x 16 halves V^T scratch) = 30,208 B */
_Static_assert(MP_BYTES >= 2u * 16u * AL_QS * 2u, "mp_lds holds Q hi and lo");
_Static_assert(MP_BYTES >= 8u * 1024u, "mp_lds holds the merge staging");
_Static_assert(MP_BYTES + 48u * 4u + sizeof(al_red) + sizeof(al_q8_vs) <= 32768u,
               "ie_attn_long_mp fits <= 32 KiB LDS (2 WGs/WGP on RDNA4)");

static LDS u8 mp_lds[MP_BYTES] __attribute__((aligned(16)));
static LDS u32 mp_last[48];

/* 2-pass 16x16 wave-private V^T transpose + power-of-2 wave-exponent normalization for one 16-key tile.
 * Produces (b0, b1, vs, w_sc) bitwise identical to al_q8_pv_step while using half the LDS scratch (16x16 vs 32x16). */
static inline __attribute__((always_inline)) void mp_q8_vprep(u32x4 vrk, f16 vsck, LDSP f16 *sc, u32 w, u32 hl, u32 rl,
                                                              h8 *b0_out, h8 *b1_out, v8f *vs_out, float *w_sc_out) {
  const h8 vh0 = al_i8_to_h8(vrk[0], vrk[1]);
  const h8 vh1 = al_i8_to_h8(vrk[2], vrk[3]);
  const float vsm = wave_max_pos(hl == 0u ? __builtin_fabsf((float)vsck) : 0.0f);
  const u32 vbits = F2I(vsm) & 0x7f800000u;
  const float inv_w = vbits ? I2F((264u << 23) - vbits) : 0.0f;
  *w_sc_out = vbits ? I2F(vbits - (10u << 23)) : 0.0f;
  if (hl == 0u) {
#pragma unroll
    for (u32 e = 0; e < 8u; e++) {
      sc[e * AL_SS + rl] = vh0[e];
      sc[(8u + e) * AL_SS + rl] = vh1[e];
    }
    al_q8_vs[w][rl] = (float)vsck * inv_w;
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  *b0_out = *(const LDSP h8 *)(sc + rl * AL_SS + 8u * hl);
  *vs_out = *(const LDSP v8f *)(&al_q8_vs[w][8u * hl]);
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  if (hl == 1u) {
#pragma unroll
    for (u32 e = 0; e < 8u; e++) {
      sc[e * AL_SS + rl] = vh0[e];
      sc[(8u + e) * AL_SS + rl] = vh1[e];
    }
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  *b1_out = *(const LDSP h8 *)(sc + rl * AL_SS + 8u * hl);
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

static inline __attribute__((always_inline)) void mp_q8_pv_accum(h8 b0, h8 b1, v8f vs, float w_sc, const LDSP f16 *pp,
                                                                 u32 kt, u32 hl, u32 rl, u32 vmask, v8f *acc0, v8f *acc1) {
  const v8f zero = (v8f)(0.0f);
  const h8 ah = *(const LDSP h8 *)(pp + rl * AL_PS + 16u * kt + 8u * hl);
  const h8 al = *(const LDSP h8 *)(pp + 16u * AL_PS + rl * AL_PS + 16u * kt + 8u * hl);
  h8 ahs, als;
#pragma unroll
  for (u32 e = 0; e < 8u; e++) {
    const float p_sc = ((float)ah[e] + (float)al[e]) * vs[e];
    const f16 hv = (f16)p_sc;
    ahs[e] = hv;
    als[e] = (f16)(p_sc - (float)hv);
  }
  v8f t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ahs, b0, zero);
  v8f t1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ahs, b1, zero);
  if (!(AL_DBG & 16)) {
    t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(als, b0, t0);
    t1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(als, b1, t1);
  }
#pragma unroll
  for (u32 v = 0; v < 8u; v++) {
    if (vmask & (1u << v)) {
      (*acc0)[v] = __builtin_fmaf(t0[v], w_sc, (*acc0)[v]);
      (*acc1)[v] = __builtin_fmaf(t1[v], w_sc, (*acc1)[v]);
    }
  }
}

static inline __attribute__((always_inline)) void mp_load_q_lds(const G float *q, u32 pg, u32 npair, u32 grp, u32 kh,
                                                                u32 qst) {
  const u32 r = tid() >> 4, d0 = 16u * (tid() & 15u), i = 16u * pg + r;
  h8 hi0 = (h8)(0.0f), hi1 = hi0, lo0 = hi0, lo1 = hi0;
  if (i < npair) {
    const u32 j = i / grp, h = kh * grp + i % grp;
    const G float *qp = (const G float *)((const G u8 *)q + (unsigned long)j * qst) + (unsigned long)h * 256ul + d0;
    f32x4 x[4];
#pragma unroll
    for (u32 u = 0; u < 4u; u++) x[u] = ((const G f32x4 *)qp)[u];
#pragma unroll
    for (u32 u = 0; u < 4u; u++)
#pragma unroll
      for (u32 e = 0; e < 4u; e++) {
        const float v = x[u][e] * 0.0625f;
        const f16 hv = (f16)v, lv = (f16)(v - (float)hv);
        if (u < 2u) hi0[4u * u + e] = hv, lo0[4u * u + e] = lv;
        else hi1[4u * (u - 2u) + e] = hv, lo1[4u * (u - 2u) + e] = lv;
      }
  }
  LDSP f16 *qh = (LDSP f16 *)mp_lds, *ql = qh + 16u * AL_QS;
  *(LDSP h8 *)(qh + r * AL_QS + d0) = hi0;
  *(LDSP h8 *)(qh + r * AL_QS + d0 + 8u) = hi1;
  *(LDSP h8 *)(ql + r * AL_QS + d0) = lo0;
  *(LDSP h8 *)(ql + r * AL_QS + d0 + 8u) = lo1;
}

#ifdef IE_SALUKI_ATTN_MP
#define MP_KERNEL __attribute__((amdgpu_kernel, visibility("protected"))) void ie_saluki_attn_long_mp
#else
#define MP_KERNEL KERNEL ie_attn_long_mp
#endif
MP_KERNEL(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos,
                       u32 n_head, u32 n_kv, u32 T, u32 npg, G u32 *count, G u32 *gcount, u32 gmax, G u8 *oq,
                       const G float *gate, u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride,
                       const G u8 *tr) {
#ifdef IE_SALUKI_ATTN_MP
  /* The host checks this ABI geometry. Constant folding drops tree/window
   * handling and specializes all three query groups without changing the math. */
  n_head = 24u;
  n_kv = 4u;
  T = 8u;
  npg = 3u;
  tr = (const G u8 *)0;
#endif
#define MP_DJ(j) (tr ? (u32)tr[TR_DEP + (j)] : (j))
  const u32 grp = n_head / n_kv;
  const u32 wbeg = __builtin_amdgcn_readfirstlane(npg >> 5);
  npg = __builtin_amdgcn_readfirstlane(npg & 31u);
  if (npg == 0u || npg > 3u) return;
  const u32 wskip = __builtin_amdgcn_readfirstlane(wbeg >= 256u ? wbeg / AL_CH - 1u : 0u);
  const u32 kh = __builtin_amdgcn_readfirstlane(wgid() % n_kv);
  const u32 sg = __builtin_amdgcn_readfirstlane(wgid() / n_kv);
  const u32 psg = __builtin_amdgcn_readfirstlane(sg ? sg + wskip : 0u);
  const u32 w = tid() >> 5, l = lane(), hl = l >> 4, rl = l & 15u;
  const u32 npair = __builtin_amdgcn_readfirstlane(grp * T);
  const u32 kend = __builtin_amdgcn_readfirstlane(pos + (tr ? (u32)tr[368] : T));
  const u32 c0 = __builtin_amdgcn_readfirstlane(psg * AL_CH);
  if (c0 >= kend) return; /* workgroup-uniform */
  const int is_tail = __builtin_amdgcn_readfirstlane(tr != (const G u8 *)0 && c0 + AL_CH > pos);
  const v8f zero = (v8f)(0.0f);
  const unsigned long pst = pstride;

  /* Phase 1: load 16 keys per wave ONCE on interior splits; stream Q_pg through mp_lds and compute s_pg[0..npg-1]. */
  const u32 kb = c0 + 16u * w;
  const int kon = kb < kend;
  v8f s_pg[3] = {zero, zero, zero};
  if (!is_tail) {
    u32x4 kq[8];
    u32x4 ksc4 = (u32x4)(0u);
    if (kon) {
      const unsigned long row = kb + rl < kend ? kb + rl : kend - 1u;
      al_q8_kload(kc, row, n_kv, kh, hl, kq, &ksc4);
    }
#pragma unroll
    for (u32 pg = 0; pg < 3u; pg++) {
      if (pg < npg) {
        mp_load_q_lds(q, pg, npair, grp, kh, qst);
        lbarrier();
        if (kon && !(AL_DBG & 4)) {
          const LDSP f16 *qh = (const LDSP f16 *)mp_lds, *ql = qh + 16u * AL_QS;
          s_pg[pg] = al_q8_kqt_blocks(kq, ksc4, qh, ql, hl, rl);
        }
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        lbarrier();
      }
    }
  } else {
#pragma unroll
    for (u32 pg = 0; pg < 3u; pg++) {
      if (pg < npg) {
        mp_load_q_lds(q, pg, npair, grp, kh, qst);
        lbarrier();
        if (kon && !(AL_DBG & 4)) {
          const LDSP f16 *qh = (const LDSP f16 *)mp_lds, *ql = qh + 16u * AL_QS;
          const u32 ci = 16u * pg + rl;
          const u32 cj = ci < npair ? ci / grp : 0u;
          if (kb + 15u >= pos) {
            const u32 jlo = 16u * pg / grp, jhi = (16u * pg + 15u) / grp < T - 1u ? (16u * pg + 15u) / grp : T - 1u;
            v8f stmp = zero;
#pragma nounroll
            for (u32 jj = jlo; jj <= jhi; jj++) {
              u32x4 tkq[8];
              u32x4 tksc4 = (u32x4)(0u);
              const unsigned long trow = al_q8_trow(tr, jj, kb + rl < kend ? kb + rl : kend - 1u, pos);
              al_q8_kload(kc, trow, n_kv, kh, hl, tkq, &tksc4);
              const v8f sj = al_q8_kqt_blocks(tkq, tksc4, qh, ql, hl, rl);
              if (cj == jj) stmp = sj;
            }
            s_pg[pg] = stmp;
          } else {
            u32x4 tkq[8];
            u32x4 tksc4 = (u32x4)(0u);
            const unsigned long row = kb + rl < kend ? kb + rl : kend - 1u;
            al_q8_kload(kc, row, n_kv, kh, hl, tkq, &tksc4);
            s_pg[pg] = al_q8_kqt_blocks(tkq, tksc4, qh, ql, hl, rl);
          }
        }
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        lbarrier();
      }
    }
  }

  /* Phase 2: issue V tile-0 global prefetch ahead of softmax, write (M, L) directly to part, and stage P_pg into mp_lds. */
  u32x4 vr_cur = (u32x4)(0u);
  f16 vsc_cur = (f16)0.0f;
  if (c0 < kend && !(AL_DBG & 2)) {
    const unsigned long row0 = c0 + rl < kend ? c0 + rl : kend - 1u;
    const G u8 *rec0 = (const G u8 *)vc + (row0 * n_kv + kh) * 272ul;
    vr_cur = *(const G u32x4 *)(rec0 + 32ul * w + 16ul * hl);
    vsc_cur = ((const G f16 *)(rec0 + 256ul))[w];
  }

#pragma unroll
  for (u32 pg = 0; pg < 3u; pg++) {
    if (pg < npg) {
      const u32 ci = 16u * pg + rl;
      const u32 cj = ci < npair ? ci / grp : 0u;
      const u32 cpj = pos + MP_DJ(cj);
      const u32 cmax = (wbeg && !sg) ? 63u : cpj;
      float sv[8], mx = -__builtin_inff();
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        sv[v] = kb + 8u * hl + v <= cmax ? s_pg[pg][v] : -__builtin_inff();
        mx = __builtin_fmaxf(mx, sv[v]);
      }
      mx = __builtin_fmaxf(mx, I2F(xrow_i(F2I(mx))));
      if (hl == 0u) al_red[0][16u * w + rl] = mx;
      lbarrier();
      float M = al_red[0][rl];
#pragma unroll
      for (u32 x = 1; x < 8u; x++) M = __builtin_fmaxf(M, al_red[0][16u * x + rl]);
      h8 ph8, pl8;
      float ls = 0.0f;
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const float p = kb + 8u * hl + v <= cmax ? __builtin_expf(sv[v] - M) : 0.0f;
        ls += p;
        const f16 hv = (f16)p;
        ph8[v] = hv;
        pl8[v] = (f16)(p - (float)hv);
      }
      ls += I2F(xrow_i(F2I(ls)));
      if (hl == 0u) al_red[1][16u * w + rl] = ls;
      LDSP f16 *pp = (LDSP f16 *)mp_lds + pg * MP_PG_HALVES;
      *(LDSP h8 *)(pp + rl * AL_PS + 16u * w + 8u * hl) = ph8;
      *(LDSP h8 *)(pp + 16u * AL_PS + rl * AL_PS + 16u * w + 8u * hl) = pl8;
      lbarrier();
      float L = al_red[1][rl];
#pragma unroll
      for (u32 x = 1; x < 8u; x++) L += al_red[1][16u * x + rl];
      if (!(AL_DBG & 1) && w == 0u && hl == 0u && ci < npair && c0 <= cpj) {
        const u32 h = kh * grp + ci % grp, ns = al_ns(cpj) - wskip;
        G float *pp_ml = part + (unsigned long)cj * pst + ((unsigned long)h * (ns + al_ng(ns)) + sg) * AL_PSZ;
        pp_ml[0] = M;
        pp_ml[1] = L;
      }
    }
  }

  /* Phase 3: single-pass software-pipelined O = P V across kt = 0 .. 7, reusing transposed V across pg = 0 .. npg - 1. */
  v8f acc[3][2] = {{zero, zero}, {zero, zero}, {zero, zero}};
  {
    LDSP f16 *sc = (LDSP f16 *)(mp_lds + MP_OFF_SC) + w * (16u * AL_SS);
    if (!is_tail) {
#pragma nounroll
      for (u32 kt = 0; kt < 8u; kt++) {
        if (c0 + 16u * kt >= kend || (AL_DBG & 2)) break;
        u32x4 vr_nxt = (u32x4)(0u);
        f16 vsc_nxt = (f16)0.0f;
        if (kt + 1u < 8u && c0 + 16u * (kt + 1u) < kend) {
          const unsigned long nrow = c0 + 16u * (kt + 1u) + rl < kend ? c0 + 16u * (kt + 1u) + rl : kend - 1u;
          const G u8 *nrec = (const G u8 *)vc + (nrow * n_kv + kh) * 272ul;
          vr_nxt = *(const G u32x4 *)(nrec + 32ul * w + 16ul * hl);
          vsc_nxt = ((const G f16 *)(nrec + 256ul))[w];
        }
        h8 b0, b1;
        v8f vs;
        float w_sc;
        mp_q8_vprep(vr_cur, vsc_cur, sc, w, hl, rl, &b0, &b1, &vs, &w_sc);
#pragma unroll
        for (u32 pg = 0; pg < 3u; pg++) {
          if (pg < npg)
            mp_q8_pv_accum(b0, b1, vs, w_sc, (const LDSP f16 *)mp_lds + pg * MP_PG_HALVES, kt, hl, rl, 0xffu,
                           &acc[pg][0], &acc[pg][1]);
        }
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        vr_cur = vr_nxt;
        vsc_cur = vsc_nxt;
      }
    } else {
#pragma nounroll
      for (u32 kt = 0; kt < 8u; kt++) {
        if (c0 + 16u * kt >= kend || (AL_DBG & 2)) break;
        u32x4 vr_nxt = (u32x4)(0u);
        f16 vsc_nxt = (f16)0.0f;
        if (kt + 1u < 8u && c0 + 16u * (kt + 1u) < kend) {
          const unsigned long nrow = c0 + 16u * (kt + 1u) + rl < kend ? c0 + 16u * (kt + 1u) + rl : kend - 1u;
          const G u8 *nrec = (const G u8 *)vc + (nrow * n_kv + kh) * 272ul;
          vr_nxt = *(const G u32x4 *)(nrec + 32ul * w + 16ul * hl);
          vsc_nxt = ((const G f16 *)(nrec + 256ul))[w];
        }
        if (c0 + 16u * kt + 15u >= pos) {
#pragma nounroll
          for (u32 jj = 0; jj < T; jj++) {
            const unsigned long key = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;
            const unsigned long trow = al_q8_trow(tr, jj, (u32)key, pos);
            const G u8 *trec = (const G u8 *)vc + (trow * n_kv + kh) * 272ul;
            const u32x4 vrj = *(const G u32x4 *)(trec + 32ul * w + 16ul * hl);
            const f16 vsj = ((const G f16 *)(trec + 256ul))[w];
            h8 b0, b1;
            v8f vs;
            float w_sc;
            mp_q8_vprep(vrj, vsj, sc, w, hl, rl, &b0, &b1, &vs, &w_sc);
#pragma unroll
            for (u32 pg = 0; pg < 3u; pg++) {
              if (pg < npg && 16u * pg / grp <= jj && jj <= (16u * pg + 15u) / grp) {
                u32 vmask = 0u;
#pragma unroll
                for (u32 v = 0; v < 8u; v++)
                  if ((16u * pg + 8u * hl + v) / grp == jj) vmask |= 1u << v;
                mp_q8_pv_accum(b0, b1, vs, w_sc, (const LDSP f16 *)mp_lds + pg * MP_PG_HALVES, kt, hl, rl, vmask,
                               &acc[pg][0], &acc[pg][1]);
              }
            }
            __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
          }
        } else {
          h8 b0, b1;
          v8f vs;
          float w_sc;
          mp_q8_vprep(vr_cur, vsc_cur, sc, w, hl, rl, &b0, &b1, &vs, &w_sc);
#pragma unroll
          for (u32 pg = 0; pg < 3u; pg++) {
            if (pg < npg)
              mp_q8_pv_accum(b0, b1, vs, w_sc, (const LDSP f16 *)mp_lds + pg * MP_PG_HALVES, kt, hl, rl, 0xffu,
                             &acc[pg][0], &acc[pg][1]);
          }
          __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        }
        vr_cur = vr_nxt;
        vsc_cur = vsc_nxt;
      }
    }
  }

  /* Phase 4: write split partials for all active pair groups and run the two-level group/head merge. */
#pragma unroll
  for (u32 pg = 0; pg < 3u; pg++) {
    if (pg < npg) {
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const u32 pi = 16u * pg + 8u * hl + v;
        if (pi < npair) {
          const u32 j = pi / grp, h = kh * grp + pi % grp;
          if (c0 <= pos + MP_DJ(j)) {
            const u32 ns = al_ns(pos + MP_DJ(j)) - wskip;
            G float *pp = part + (unsigned long)j * pst + ((unsigned long)h * (ns + al_ng(ns)) + sg) * AL_PSZ + 4u;
            pp[32u * w + rl] = acc[pg][0][v];
            pp[32u * w + 16u + rl] = acc[pg][1][v];
          }
        }
      }
    }
  }
  if (AL_DBG & 1) return;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (w < npg && hl == 0u) {
    const u32 ci = 16u * w + rl;
    u32 last = 0u;
    if (ci < npair) {
      const u32 cj = ci / grp, cpj = pos + MP_DJ(cj);
      if (c0 <= cpj) {
        const u32 ns = al_ns(cpj) - wskip, g = sg / AL_G, in_g = ns - g * AL_G < AL_G ? ns - g * AL_G : AL_G;
        const unsigned long gi = ((unsigned long)cj * n_head + kh * grp + ci % grp) * gmax + g;
        last = __scoped_atomic_fetch_add(&gcount[gi], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == in_g;
      }
    }
    mp_last[ci] = last;
  }
  barrier();
  for (u32 pi = w; pi < 16u * npg; pi += 8u) {
    if (!mp_last[pi]) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const u32 j = pi / grp, h = kh * grp + pi % grp, ns = al_ns(pos + MP_DJ(j)) - wskip, ng = al_ng(ns), g = sg / AL_G;
    const u32 in_g = ns - g * AL_G < AL_G ? ns - g * AL_G : AL_G;
    G float *pb = part + (unsigned long)j * pst + (unsigned long)h * (ns + ng) * AL_PSZ;
    G float *ox = out ? (G float *)((G u8 *)out + (unsigned long)j * ost) + (unsigned long)h * 256ul : (G float *)0;
    G u8 *oqx = oq ? oq + (unsigned long)j * oqst : oq;
    const G float *gxh = gate ? (const G float *)((const G u8 *)gate + (unsigned long)j * gst) + (unsigned long)h * gstride : gate;
    float M2, L2;
    f32x4 a0, a1;
    al_msum(pb + (unsigned long)g * AL_G * AL_PSZ, in_g, &M2, &L2, &a0, &a1);
    if (l == 0u) gcount[((unsigned long)j * n_head + h) * gmax + g] = 0u;
    if (ng == 1u) {
      al_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_head * 8u, (LDSP float *)mp_lds + w * 256u);
      continue;
    }
    G float *gp = pb + (unsigned long)(ns + g) * AL_PSZ;
    if (l == 0u) gp[0] = M2, gp[1] = L2;
    *(G f32x4 *)(gp + 4u + 4u * l) = a0;
    *(G f32x4 *)(gp + 132u + 4u * l) = a1;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    u32 done = 0u;
    if (l == 0u) done = __scoped_atomic_fetch_add(&count[j * n_head + h], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == ng;
    if (!__builtin_amdgcn_readfirstlane(done)) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    al_msum(pb + (unsigned long)ns * AL_PSZ, ng, &M2, &L2, &a0, &a1);
    if (l == 0u) count[j * n_head + h] = 0u;
    al_final(a0, a1, L2, ox, oqx, gxh, h * 8u, n_head * 8u, (LDSP float *)mp_lds + w * 256u);
  }
#undef MP_DJ
}
#undef MP_KERNEL

#endif /* IE_KV_Q8 && !IE_MEGA && !IE_KV_F32 */
