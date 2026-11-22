/* ie_attn_long128: ie_attn_long for heads of 128 values (Qwen3: 32 query heads of 128, 8 KV heads). Included by
 * ie_kernels.c after ie_attn_long, inside the same #if (not the megakernel, not an f32 cache); the host loads it into
 * ie_attn_long's slot when the model's head size is 128, with the same arguments, grid and part area (AL_PSZ floats per
 * split, of which 4 + 128 are used). Everything follows ie_attn_long (al_body); the differences:
 *   S^T = K Q^T   8 WMMA steps (lane half hl holds the dims 64 hl + 8 ks .. + 7) instead of 16
 *   O = P V       wave w owns the 16 output columns 16 w .. 16 w + 15 (one 16 x 16 tile, lane half hl loads the 8
 *                 columns 16 w + 8 hl ..) instead of 32
 *   q scale       1 / sqrt(128) (not exact like ie_attn_long's 1/16; the hi + lo split keeps q at f32 accuracy)
 *   records       q8_0: 128 int8 + 4 fp16 scales (136 bytes); 4 bit: 64 bytes of nibbles + 4 (d, m) pairs (80)
 *   LDS           its own smaller buffer (Q: 16 rows of 136 halves), so more workgroups fit on a WGP. */
#define AL128_HD 128u
#define AL128_QS (128u + AL_QS_PAD) /* LDS row stride of Q hi and lo (halves) */
#define AL128_OFF_PL (16u * AL_PS * 2u)
#define AL128_OFF_SC (2u * AL128_OFF_PL)
#define AL128_BYTES (AL128_OFF_SC + 8u * AL_SCP * 16u * AL_SS * 2u)
#define AL128_MAX(a, b) ((a) > (b) ? (a) : (b))
#define AL128_LDS AL128_MAX(AL128_MAX(AL128_BYTES, 2u * 16u * AL128_QS * 2u), 8u * 256u * 4u)
static LDS u8 al128_lds[AL128_LDS] __attribute__((aligned(16)));
static LDS float al128_red[2][8 * 16];
static LDS u32 al128_last[16];
#if defined(IE_KV_Q8)
#define AL128_REC 136ul
#elif defined(IE_KV_QUANT)
#define AL128_REC 80ul
#endif

/* One wave sums n partials at ph (AL_PSZ floats apart): M2 = max M_t, w_t = exp(M_t - M2), L2 = sum l_t w_t, and lane l
 * holds the columns 4 l .. 4 l + 3 of o2 = sum o_t w_t, all in the order t. */
static inline void al128_msum(const G float *ph, u32 n, float *M2o, float *L2o, f32x4 *a0o) {
  const u32 l = lane();
  float mv = -__builtin_inff();
  for (u32 t = l; t < n; t += 32u) mv = __builtin_fmaxf(mv, ph[(unsigned long)t * AL_PSZ]);
  const float M2 = al_wave_max(mv);
  f32x4 a0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
  float L2 = 0.0f;
  for (u32 t0 = 0; t0 < n; t0 += 32u) {
    const u32 tl = t0 + l, nu = n - t0 < 32u ? n - t0 : 32u;
    const float wl = tl < n ? __builtin_expf(ph[(unsigned long)tl * AL_PSZ] - M2) : 0.0f;
    const float ll = tl < n ? ph[(unsigned long)tl * AL_PSZ + 1u] : 0.0f;
    u32 u = 0;
    for (; u + 8u <= nu; u += 8u) { /* 8 loads in flight, then the sums in the order t */
      f32x4 x0[8];
#pragma unroll
      for (u32 k = 0; k < 8u; k++) x0[k] = *(const G f32x4 *)(ph + (unsigned long)(t0 + u + k) * AL_PSZ + 4u + 4u * l);
#pragma unroll
      for (u32 k = 0; k < 8u; k++) {
        const float wt = I2F(__builtin_amdgcn_readlane(F2I(wl), (int)(u + k)));
        L2 += I2F(__builtin_amdgcn_readlane(F2I(ll), (int)(u + k))) * wt;
        a0 += x0[k] * wt;
      }
    }
    for (; u < nu; u++) {
      const f32x4 x0 = *(const G f32x4 *)(ph + (unsigned long)(t0 + u) * AL_PSZ + 4u + 4u * l);
      const float wt = I2F(__builtin_amdgcn_readlane(F2I(wl), (int)u));
      L2 += I2F(__builtin_amdgcn_readlane(F2I(ll), (int)u)) * wt;
      a0 += x0 * wt;
    }
  }
  *M2o = M2, *L2o = L2, *a0o = a0;
}
/* out = o / L, times the gate, and its Q8 copy (4 blocks of 32 per head) */
static inline void al128_final(f32x4 a0, float L2, G float *ox, G u8 *oqx, const G float *gxh, u32 blk0, u32 nbq, LDSP float *ob) {
  const u32 l = lane();
  *(LDSP f32x4 *)(ob + 4u * l) = a0 / L2;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
  for (u32 i = 0; i < 4u; i++) {
    float o = ob[32u * i + l];
    if (gxh) o = o * sigmoidf(gxh[32u * i + l]);
    ox[32u * i + l] = o;
    if (oqx) quant_wave(o, blk0 + i, oqx, nbq);
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
}

/* the V slice of the wave for tile kt: key rl, columns 16 w + 8 hl .. + 7 */
#if !defined(IE_KV_QUANT)
#define AL128_VDECL h8 vr[8];
#define AL128_VLOAD                                                                                     \
  _Pragma("unroll") for (u32 kt = 0; kt < 8u; kt++) if (c0 + 16u * kt < kend) {                         \
    const unsigned long row = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;               \
    vr[kt] = *(const G h8 *)(vc + (row * n_kv + kh) * 128ul + 16ul * w + 8ul * hl);                     \
  }
#elif defined(IE_KV_Q8)
#define AL128_VDECL                                                                                     \
  u32x2 vr[8];                                                                                          \
  f16 vsc[8];
#define AL128_VLOAD                                                                                     \
  _Pragma("unroll") for (u32 kt = 0; kt < 8u; kt++) if (c0 + 16u * kt < kend) {                         \
    const unsigned long row = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;               \
    const G u8 *rec = (const G u8 *)vc + (row * n_kv + kh) * AL128_REC;                                 \
    vr[kt] = *(const G u32x2 *)(rec + 16ul * w + 8ul * hl);                                             \
    vsc[kt] = ((const G f16 *)(rec + 128))[w >> 1];                                                     \
  }
#else
#define AL128_VDECL                                                                                     \
  u32 vr[8];                                                                                            \
  u32 vdm[8];
#define AL128_VLOAD                                                                                     \
  _Pragma("unroll") for (u32 kt = 0; kt < 8u; kt++) if (c0 + 16u * kt < kend) {                         \
    const unsigned long row = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;               \
    const G u8 *rec = (const G u8 *)vc + (row * n_kv + kh) * AL128_REC;                                 \
    vr[kt] = *(const G u32 *)(rec + 8ul * w + 4ul * hl);                                                \
    vdm[kt] = *(const G u32 *)(rec + 64ul + 4ul * (w >> 1));                                            \
  }
#endif

/* a tree verify (any cache format): token j's key/value row for the key position key */
static inline u32 al128_trow(const G u8 *tr, u32 j, u32 key, u32 pos) {
  return key < pos || key - pos > tr[TR_DEP + j] ? key : pos + tr[TR_PATH + 16u * j + (key - pos)];
}
/* S^T of one key row (this lane: key row's dims 64 hl + 8 ks ..) against the 16 pairs' Q in LDS: a tree tile's pass */
static inline v8f al128_kdot(const G kvt *kc, unsigned long row, u32 n_kv, u32 kh, u32 hl, u32 rl, const LDSP f16 *qh,
                             const LDSP f16 *ql) {
  v8f s = (v8f)(0.0f);
#if !defined(IE_KV_QUANT)
  const G h8 *kp = (const G h8 *)(kc + (row * n_kv + kh) * 128ul + 64ul * hl);
#pragma unroll
  for (u32 ks = 0; ks < 8u; ks++) {
    const h8 k8 = kp[ks];
    const h8 bh = *(const LDSP h8 *)(qh + rl * AL128_QS + 64u * hl + 8u * ks);
    const h8 bl = *(const LDSP h8 *)(ql + rl * AL128_QS + 64u * hl + 8u * ks);
    s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k8, bh, s);
    s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(k8, bl, s);
  }
#else
  const G u8 *rec = (const G u8 *)kc + (row * n_kv + kh) * AL128_REC;
#pragma unroll
  for (u32 ks = 0; ks < 8u; ks++) {
    h8 kh8, kl8;
#ifdef IE_KV_Q8
    const u32x4 kq = ((const G u32x4 *)(rec + 64ul * hl))[ks >> 1];
    al_deq8(kq[2u * (ks & 1u)], kq[2u * (ks & 1u) + 1u], ((const G f16 *)(rec + 128))[2u * hl + (ks >> 2)], &kh8, &kl8);
#else
    const u32 kw = ((const G u32 *)(rec + 32ul * hl))[ks];
    const u32 dm = ((const G u32 *)(rec + 64ul + 8ul * hl))[ks >> 2];
    const float kd = al_lo(dm), km = al_hi(dm);
#pragma unroll
    for (u32 e = 0; e < 8u; e++) {
      const float x = __builtin_fmaf((float)((kw >> (4u * e)) & 15u), kd, km);
      kh8[e] = (f16)x;
      kl8[e] = (f16)(x - (float)kh8[e]);
    }
#endif
    const h8 bh = *(const LDSP h8 *)(qh + rl * AL128_QS + 64u * hl + 8u * ks);
    const h8 bl = *(const LDSP h8 *)(ql + rl * AL128_QS + 64u * hl + 8u * ks);
    s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kh8, bh, s);
    s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kh8, bl, s);
    s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kl8, bh, s);
  }
#endif
  return s;
}
/* this lane's 8 values (columns 16 w + 8 hl ..) of one value row as fp16 hi + lo (f16 cache: lo 0) */
static inline void al128_vrow(const G kvt *vc, unsigned long row, u32 n_kv, u32 kh, u32 w, u32 hl, h8 *vh, h8 *vl) {
#if !defined(IE_KV_QUANT)
  *vh = *(const G h8 *)(vc + (row * n_kv + kh) * 128ul + 16ul * w + 8ul * hl);
  *vl = (h8)(0.0f);
#elif defined(IE_KV_Q8)
  const G u8 *rec = (const G u8 *)vc + (row * n_kv + kh) * AL128_REC;
  const u32x2 v = *(const G u32x2 *)(rec + 16ul * w + 8ul * hl);
  al_deq8(v[0], v[1], ((const G f16 *)(rec + 128))[w >> 1], vh, vl);
#else
  const G u8 *rec = (const G u8 *)vc + (row * n_kv + kh) * AL128_REC;
  const u32 v = *(const G u32 *)(rec + 8ul * w + 4ul * hl), dm = *(const G u32 *)(rec + 64ul + 4ul * (w >> 1));
  const float vd = al_lo(dm), vm = al_hi(dm);
#pragma unroll
  for (u32 e = 0; e < 8u; e++) {
    const float x = __builtin_fmaf((float)((v >> (4u * e)) & 15u), vd, vm);
    (*vh)[e] = (f16)x;
    (*vl)[e] = (f16)(x - (float)(*vh)[e]);
  }
#endif
}

static inline __attribute__((always_inline)) void al128_body(const G float *q, const G kvt *kc, const G kvt *vc, G float *part,
                                                             G float *out, u32 pos, u32 n_head, u32 n_kv, u32 T, u32 npg,
                                                             G u32 *count, G u32 *gcount, u32 gmax, G u8 *oq, const G float *gate,
                                                             u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride,
                                                             const int early, const G u8 *tr) {
#define AL128_DJ(j) (tr ? (u32)tr[TR_DEP + (j)] : (j))
  const u32 grp = n_head / n_kv;
  const u32 pg = wgid() % npg, kh = (wgid() / npg) % n_kv, sg = wgid() / (npg * n_kv);
  const u32 w = tid() >> 5, l = lane(), hl = l >> 4, rl = l & 15u;
  const u32 npair = grp * T, kend = pos + T, c0 = sg * AL_CH;
  if (c0 >= kend) return;
  const v8f zero = (v8f)(0.0f);
  const unsigned long pst = pstride;
  const u32 kb = c0 + 16u * w;
  const int kon = kb < kend;
#if !defined(IE_KV_QUANT)
  h8 kr[8];
#elif defined(IE_KV_Q8)
  u32x4 kq[4]; /* the int8 values of the dims 64 hl .. 64 hl + 63 of the key, and their 2 block scales */
  f16 ksc[2];
#else
  u32x4 kq[2]; /* the nibbles of the dims 64 hl .. 64 hl + 63 (one u32x4 per block of 32), and the (d, m) pairs */
  u32 kdm[2];
#endif
  if (kon) {
    const unsigned long row = kb + rl < kend ? kb + rl : kend - 1u;
#if !defined(IE_KV_QUANT)
    const G h8 *kp = (const G h8 *)(kc + (row * n_kv + kh) * 128ul + 64ul * hl);
#pragma unroll
    for (u32 ks = 0; ks < 8u; ks++) kr[ks] = kp[ks];
#elif defined(IE_KV_Q8)
    const G u8 *rec = (const G u8 *)kc + (row * n_kv + kh) * AL128_REC;
#pragma unroll
    for (u32 i = 0; i < 4u; i++) kq[i] = ((const G u32x4 *)(rec + 64ul * hl))[i];
#pragma unroll
    for (u32 i = 0; i < 2u; i++) ksc[i] = ((const G f16 *)(rec + 128))[2u * hl + i];
#else
    const G u8 *rec = (const G u8 *)kc + (row * n_kv + kh) * AL128_REC;
#pragma unroll
    for (u32 i = 0; i < 2u; i++) kq[i] = ((const G u32x4 *)(rec + 32ul * hl))[i];
#pragma unroll
    for (u32 i = 0; i < 2u; i++) kdm[i] = ((const G u32 *)(rec + 64ul + 8ul * hl))[i];
#endif
  }
  AL128_VDECL
  if (early) {
    AL128_VLOAD
  }
  const u32 ci = 16u * pg + rl;
  const int clive = ci < npair;
  const u32 cj = clive ? ci / grp : 0u, cpj = pos + AL128_DJ(cj);
  const int cval = clive && c0 <= cpj;
  /* Q / sqrt(128) as fp16 hi + lo, rows of 128 halves (AL128_QS apart), 16 pairs; dead pairs 0 */
  {
    const u32 r = tid() >> 4, d0 = 8u * (tid() & 15u), i = 16u * pg + r;
    h8 hi0 = (h8)(0.0f), lo0 = hi0;
    if (i < npair) {
      const u32 j = i / grp, h = kh * grp + i % grp;
      const G float *qp = (const G float *)((const G u8 *)q + (unsigned long)j * qst) + (unsigned long)h * 128ul + d0;
      const f32x4 x0 = ((const G f32x4 *)qp)[0], x1 = ((const G f32x4 *)qp)[1];
#pragma unroll
      for (u32 e = 0; e < 8u; e++) {
        const float v = (e < 4u ? x0[e] : x1[e - 4u]) * 0.08838834764831845f;
        const f16 hv = (f16)v;
        hi0[e] = hv, lo0[e] = (f16)(v - (float)hv);
      }
    }
    LDSP f16 *qh = (LDSP f16 *)al128_lds, *ql = qh + 16u * AL128_QS;
    *(LDSP h8 *)(qh + r * AL128_QS + d0) = hi0;
    *(LDSP h8 *)(ql + r * AL128_QS + d0) = lo0;
  }
  lbarrier();
  v8f s = zero;
  if (tr && kon && kb + 15u >= pos) { /* a tile of the tree: once per token of the workgroup, that token's key rows */
    const LDSP f16 *qh = (const LDSP f16 *)al128_lds, *ql = qh + 16u * AL128_QS;
    const u32 jlo = 16u * pg / grp, jhi = (16u * pg + 15u) / grp < T - 1u ? (16u * pg + 15u) / grp : T - 1u;
    for (u32 jj = jlo; jj <= jhi; jj++) {
      const unsigned long row = al128_trow(tr, jj, kb + rl < kend ? kb + rl : kend - 1u, pos);
      const v8f sj = al128_kdot(kc, row, n_kv, kh, hl, rl, qh, ql);
      if (cj == jj) s = sj; /* this lane's pair is of token jj */
    }
  } else if (kon) {
    const LDSP f16 *qh = (const LDSP f16 *)al128_lds, *ql = qh + 16u * AL128_QS;
#pragma unroll
    for (u32 ks = 0; ks < 8u; ks++) {
      const h8 bh = *(const LDSP h8 *)(qh + rl * AL128_QS + 64u * hl + 8u * ks);
      const h8 bl = *(const LDSP h8 *)(ql + rl * AL128_QS + 64u * hl + 8u * ks);
#if !defined(IE_KV_QUANT)
      s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kr[ks], bh, s);
      s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kr[ks], bl, s);
#else
      h8 kh8, kl8;
#ifdef IE_KV_Q8
      al_deq8(kq[ks >> 1][2u * (ks & 1u)], kq[ks >> 1][2u * (ks & 1u) + 1u], ksc[ks >> 2], &kh8, &kl8);
#else
      {
        const u32 kw = kq[ks >> 2][ks & 3u];
        const float kd = al_lo(kdm[ks >> 2]), km = al_hi(kdm[ks >> 2]);
#pragma unroll
        for (u32 e = 0; e < 8u; e++) {
          const float x = __builtin_fmaf((float)((kw >> (4u * e)) & 15u), kd, km);
          kh8[e] = (f16)x;
          kl8[e] = (f16)(x - (float)kh8[e]);
        }
      }
#endif
      s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kh8, bh, s);
      s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kh8, bl, s);
      s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kl8, bh, s);
#endif
    }
  }
  float sv[8], mx = -__builtin_inff();
#pragma unroll
  for (u32 v = 0; v < 8u; v++) {
    sv[v] = kb + 8u * hl + v <= cpj ? s[v] : -__builtin_inff();
    mx = __builtin_fmaxf(mx, sv[v]);
  }
  mx = __builtin_fmaxf(mx, I2F(xrow_i(F2I(mx))));
  if (hl == 0u) al128_red[0][16u * w + rl] = mx;
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  lbarrier();
  float M = al128_red[0][rl];
#pragma unroll
  for (u32 x = 1; x < 8u; x++) M = __builtin_fmaxf(M, al128_red[0][16u * x + rl]);
  if (!early) {
    AL128_VLOAD
  }
  h8 ph8, pl8;
  float ls = 0.0f;
#pragma unroll
  for (u32 v = 0; v < 8u; v++) {
    const float p = kb + 8u * hl + v <= cpj ? __builtin_expf(sv[v] - M) : 0.0f;
    ls += p;
    const f16 hv = (f16)p;
    ph8[v] = hv;
    pl8[v] = (f16)(p - (float)hv);
  }
  ls += I2F(xrow_i(F2I(ls)));
  if (hl == 0u) al128_red[1][16u * w + rl] = ls;
  {
    LDSP f16 *pp = (LDSP f16 *)al128_lds;
    *(LDSP h8 *)(pp + rl * AL_PS + 16u * w + 8u * hl) = ph8;
    *(LDSP h8 *)(pp + 16u * AL_PS + rl * AL_PS + 16u * w + 8u * hl) = pl8;
  }
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  lbarrier();
  float L = al128_red[1][rl];
#pragma unroll
  for (u32 x = 1; x < 8u; x++) L += al128_red[1][16u * x + rl];
  /* O = P V for the wave's columns 16 w + rl: the V slice of each key tile through a private transpose (rows: the
   * 16 columns, entries: the 16 keys) */
  v8f acc = zero;
  {
    LDSP f16 *sc = (LDSP f16 *)(al128_lds + AL128_OFF_SC) + w * (AL_SCP * 16u * AL_SS);
    const LDSP f16 *pp = (const LDSP f16 *)al128_lds;
#pragma unroll
    for (u32 kt = 0; kt < 8u; kt++)
      if (c0 + 16u * kt < kend) {
        const h8 ah = *(const LDSP h8 *)(pp + rl * AL_PS + 16u * kt + 8u * hl);
        const h8 al = *(const LDSP h8 *)(pp + 16u * AL_PS + rl * AL_PS + 16u * kt + 8u * hl);
        if (tr && c0 + 16u * kt + 15u >= pos) { /* a tile of the tree: per token, its V rows; each row keeps its token's */
          const u32 jlo = 16u * pg / grp, jhi = (16u * pg + 15u) / grp < T - 1u ? (16u * pg + 15u) / grp : T - 1u;
          const v8f ain = acc;
          for (u32 jj = jlo; jj <= jhi; jj++) {
            const unsigned long key = c0 + 16u * kt + rl < kend ? c0 + 16u * kt + rl : kend - 1u;
            h8 vh, vl;
            al128_vrow(vc, al128_trow(tr, jj, (u32)key, pos), n_kv, kh, w, hl, &vh, &vl);
#pragma unroll
            for (u32 e = 0; e < 8u; e++) sc[(8u * hl + e) * AL_SS + rl] = vh[e];
            __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
            const h8 b0 = *(const LDSP h8 *)(sc + rl * AL_SS + 8u * hl);
            __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
            v8f t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, b0, ain);
            t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(al, b0, t0);
#if defined(IE_KV_QUANT)
#pragma unroll
            for (u32 e = 0; e < 8u; e++) sc[(8u * hl + e) * AL_SS + rl] = vl[e];
            __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
            const h8 c0l = *(const LDSP h8 *)(sc + rl * AL_SS + 8u * hl);
            t0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, c0l, t0);
#endif
#pragma unroll
            for (u32 v = 0; v < 8u; v++)
              if ((16u * pg + 8u * hl + v) / grp == jj) acc[v] = t0[v];
            __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
          }
          continue;
        }
#if !defined(IE_KV_QUANT)
#pragma unroll
        for (u32 e = 0; e < 8u; e++) sc[(8u * hl + e) * AL_SS + rl] = vr[kt][e];
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        const h8 b0 = *(const LDSP h8 *)(sc + rl * AL_SS + 8u * hl);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, b0, acc);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(al, b0, acc);
#else
        h8 vh, vl; /* the columns 8 hl .. + 7 of key rl: hi and lo planes */
#ifdef IE_KV_Q8
        al_deq8(vr[kt][0], vr[kt][1], vsc[kt], &vh, &vl);
#else
        {
          const float vd = al_lo(vdm[kt]), vm = al_hi(vdm[kt]);
#pragma unroll
          for (u32 e = 0; e < 8u; e++) {
            const float x = __builtin_fmaf((float)((vr[kt] >> (4u * e)) & 15u), vd, vm);
            vh[e] = (f16)x;
            vl[e] = (f16)(x - (float)vh[e]);
          }
        }
#endif
        /* the hi plane, then the lo plane, through the same scratch */
#pragma unroll
        for (u32 e = 0; e < 8u; e++) sc[(8u * hl + e) * AL_SS + rl] = vh[e];
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        const h8 b0 = *(const LDSP h8 *)(sc + rl * AL_SS + 8u * hl);
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma unroll
        for (u32 e = 0; e < 8u; e++) sc[(8u * hl + e) * AL_SS + rl] = vl[e];
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, b0, acc);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(al, b0, acc);
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        const h8 c0l = *(const LDSP h8 *)(sc + rl * AL_SS + 8u * hl);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, c0l, acc);
#endif
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory"); /* the reads before the next tile's writes */
      }
  }
  /* the split's partial results: o rows 8 hl + v (pair 16 pg + 8 hl + v), column 16 w + rl */
#pragma unroll
  for (u32 v = 0; v < 8u; v++) {
    const u32 pi = 16u * pg + 8u * hl + v;
    if (pi < npair) {
      const u32 j = pi / grp, h = kh * grp + pi % grp;
      if (c0 <= pos + AL128_DJ(j)) {
        const u32 ns = al_ns(pos + AL128_DJ(j));
        G float *pp = part + j * pst + ((unsigned long)h * (ns + al_ng(ns)) + sg) * AL_PSZ + 4u;
        pp[16u * w + rl] = acc[v];
      }
    }
  }
  if (w == 0u && hl == 0u && cval) {
    const u32 h = kh * grp + ci % grp, ns = al_ns(cpj);
    G float *pp = part + cj * pst + ((unsigned long)h * (ns + al_ng(ns)) + sg) * AL_PSZ;
    pp[0] = M;
    pp[1] = L;
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (w == 0u && hl == 0u) {
    u32 last = 0u;
    if (cval) {
      const u32 ns = al_ns(cpj), g = sg / AL_G, in_g = ns - g * AL_G < AL_G ? ns - g * AL_G : AL_G;
      const unsigned long gi = ((unsigned long)cj * n_head + kh * grp + ci % grp) * gmax + g;
      last = __scoped_atomic_fetch_add(&gcount[gi], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == in_g;
    }
    al128_last[rl] = last;
  }
  barrier();
  for (u32 x = w; x < 16u; x += 8u) {
    if (!al128_last[x]) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const u32 pi = 16u * pg + x, j = pi / grp, h = kh * grp + pi % grp, ns = al_ns(pos + AL128_DJ(j)), ng = al_ng(ns), g = sg / AL_G;
    const u32 in_g = ns - g * AL_G < AL_G ? ns - g * AL_G : AL_G;
    G float *pb = part + j * pst + (unsigned long)h * (ns + ng) * AL_PSZ;
    G float *ox = (G float *)((G u8 *)out + (unsigned long)j * ost) + (unsigned long)h * 128ul;
    G u8 *oqx = oq ? oq + (unsigned long)j * oqst : oq;
    const G float *gxh = gate ? (const G float *)((const G u8 *)gate + (unsigned long)j * gst) + (unsigned long)h * gstride : gate;
    float M2, L2;
    f32x4 a0;
    al128_msum(pb + (unsigned long)g * AL_G * AL_PSZ, in_g, &M2, &L2, &a0);
    if (l == 0u) gcount[((unsigned long)j * n_head + h) * gmax + g] = 0u;
    if (ng == 1u) {
      al128_final(a0, L2, ox, oqx, gxh, h * 4u, n_head * 4u, (LDSP float *)al128_lds + w * 256u);
      continue;
    }
    G float *gp = pb + (unsigned long)(ns + g) * AL_PSZ;
    if (l == 0u) gp[0] = M2, gp[1] = L2;
    *(G f32x4 *)(gp + 4u + 4u * l) = a0;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    u32 done = 0u;
    if (l == 0u) done = __scoped_atomic_fetch_add(&count[j * n_head + h], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1u == ng;
    if (!__builtin_amdgcn_readfirstlane(done)) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    al128_msum(pb + (unsigned long)ns * AL_PSZ, ng, &M2, &L2, &a0);
    if (l == 0u) count[j * n_head + h] = 0u;
    al128_final(a0, L2, ox, oqx, gxh, h * 4u, n_head * 4u, (LDSP float *)al128_lds + w * 256u);
  }
#undef AL128_DJ
}
KERNEL ie_attn_long128(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos, u32 n_head,
                       u32 n_kv, u32 T, u32 npg, G u32 *count, G u32 *gcount, u32 gmax, G u8 *oq, const G float *gate,
                       u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride, const G u8 *tr) {
  al128_body(q, kc, vc, part, out, pos, n_head, n_kv, T, npg, count, gcount, gmax, oq, gate, gstride, qst, ost, oqst, gst, pstride, AL_E0, tr);
}
/* The same with the V loads first (launches of fewer than 64 workgroups). */
KERNEL ie_attn_long128_e(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos, u32 n_head,
                         u32 n_kv, u32 T, u32 npg, G u32 *count, G u32 *gcount, u32 gmax, G u8 *oq, const G float *gate,
                         u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride, const G u8 *tr) {
  al128_body(q, kc, vc, part, out, pos, n_head, n_kv, T, npg, count, gcount, gmax, oq, gate, gstride, qst, ost, oqst, gst, pstride, 1, tr);
}
