/* ============================================================================
 * Inkling-2 Round 8: Low-VGPR (<=64 VGPR) Sub-Span Persistent Glue Dispatcher
 *                    (ink_mask2 bit 90, R34 -- Mirage MPK / MonoMoE)
 * Target: RX 9070 XT (gfx1201, wave32, NT=256)
 *
 * Included at the end of kernels/ie_inkling_kernels.c.
 * All local types, helpers, LDS symbols, and macros are prefixed ink_r8_sp_ /
 * INK_R8_SP_ and all macros are #undef'd at EOF.
 *
 * SAFETY & OCCUPANCY INVARIANTS (per AGENTS.md WS4 / WDDM rules):
 *   1. Grid MUST stay <= 64 WGs (hard-enforced: if ng == 0 || ng > 64 -> return!).
 *   2. VGPR MUST stay <= 64 VGPR (4 WGs/CU, 0/0 spills, priv=0, LDS <= 4.5 KiB)
 *      so non-GEMV glue ops never drag down GEMV occupancy.
 *   3. Mandatory wall-clock poison guard on EVERY inter-WG barrier using
 *      gfx1201 `s_sendmsg_rtn_b64(0x83)` (100 MHz real-time wall clock):
 *        - Per-barrier timeout: default 25,000,000 ticks (250 ms)
 *        - Per-kernel  timeout: default 90,000,000 ticks (900 ms)
 *      On timeout, sets sync[2] |= 1u (poison_flag) and aborts cleanly so a
 *      WDDM scheduling stall can NEVER hang the GPU or TDR the machine.
 *      Host MUST check sync[2] != 0u and latch sub-span mode off if poisoned.
 *
 * Task Descriptor Layout (ink_r8_sp_task_t, 64 bytes = 16 u32s, 16-B aligned):
 *   u32 op;     // 0=ADD, 1=COPY, 2=GATE_Q8, 3=RMSNORM_Q8, 4=VCLIP,
 *               // 5=SCONV_ADD(T=1), 6=MOE_SUM_SCONV_ADD(T=1), 7=SITU_Q8
 *   u32 flags;  // bit 0: barrier AFTER this task; bit 1: norope (VCLIP)
 *   u32 n;      // element/dim count
 *   u32 a0;     // aux0 (T / nh / K / pos0)
 *   u32 a1;     // aux1 (hd / win / as_bytes)
 *   u32 a2;     // aux2 (bs_bytes / gs_bytes / pos0)
 *   u32 a3;     // aux3 (qs_bytes / T / bstr)
 *   u32 f0;     // float bits 0 (eps / clamp / escale / beta)
 *   u32 f1;     // float bits 1 (escale / lbeta)
 *   u32 rsv;    // reserved / stride extra
 *   u64 p0;     // ptr0 (a / x / y)
 *   u64 p1;     // ptr1 (b / w / g / v_conv / route)
 *   u64 p2;     // ptr2 (c / out / q / ring)
 *   u64 p3;     // ptr3 (q / res / out / extra)
 *   (Total = 10*4 + 4*8 = 72 -> pad to 80 B; using explicit 80-B struct with
 *    _Static_assert so host C & AMDGPU C11 match byte-for-byte!)
 *
 * Sync Control Block (`G u32 *sync`, 16 bytes = 4 u32s, zero-init once before first use):
 *   sync[0] = bar_arrived  (auto-reset to 0 inside sense barrier)
 *   sync[1] = bar_sense    (monotonically advancing sense counter; NEVER reset on exit -> 0 exit race!)
 *   sync[2] = poison_flag  (0 = clean, non-zero = timed out -> host latches off & re-zeros sync[0..3])
 *   sync[3] = max_wait_k   (diagnostic peak barrier wait in 1000-tick = 10 us units)
 * ========================================================================== */

#ifndef IE_MEGA

#define INK_R8_SP_MAX_GRID       64u
#define INK_R8_SP_DEF_BAR_TICKS  25000000ull /* 250 ms @ 100 MHz s_sendmsg_rtnl(0x83) */
#define INK_R8_SP_DEF_KRN_TICKS  90000000ull /* 900 ms @ 100 MHz s_sendmsg_rtnl(0x83) */

#define INK_R8_SP_OP_ADD           0u
#define INK_R8_SP_OP_COPY          1u
#define INK_R8_SP_OP_GATE_Q8       2u
#define INK_R8_SP_OP_RMSNORM_Q8    3u
#define INK_R8_SP_OP_VCLIP         4u
#define INK_R8_SP_OP_SCONV_ADD     5u
#define INK_R8_SP_OP_MOE_TAIL      6u
#define INK_R8_SP_OP_SITU_Q8       7u

#define INK_R8_SP_FLAG_BARRIER     1u
#define INK_R8_SP_FLAG_NOROPE      2u

typedef unsigned long long ink_r8_sp_u64;

typedef struct __attribute__((aligned(16))) {
  u32 op;
  u32 flags;
  u32 n;
  u32 a0;
  u32 a1;
  u32 a2;
  u32 a3;
  u32 f0;
  u32 f1;
  u32 a4;
  ink_r8_sp_u64 p0;
  ink_r8_sp_u64 p1;
  ink_r8_sp_u64 p2;
  ink_r8_sp_u64 p3;
  ink_r8_sp_u64 p4;
} ink_r8_sp_task_t;
_Static_assert(sizeof(ink_r8_sp_task_t) == 80u, "ink_r8_sp_task_t must be 80 bytes");

typedef struct {
  float rn_red[8u];
  float rn_sh[8u][128u]; /* 4,096 B for 2304-dim RMSNorm+Q8 */
  u32   poison;
} ink_r8_sp_lds_t;

static LDS ink_r8_sp_lds_t ink_r8_sp_lds __attribute__((aligned(16)));
_Static_assert(sizeof(ink_r8_sp_lds_t) <= 8192u, "ink_r8_sp_lds must stay <= 8 KiB");

static inline __attribute__((always_inline)) float ink_r8_sp_pick4(f32x4 v, u32 s) {
  return s == 0u ? v.x : (s == 1u ? v.y : (s == 2u ? v.z : v.w));
}

static inline __attribute__((always_inline)) float ink_r8_sp_sub8_max_pos(float v) {
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0xB1, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0x4E, 0xF, 0xF, true)));
  v = __builtin_fmaxf(v, I2F(__builtin_amdgcn_ds_swizzle(F2I(v), 0x101F)));
  return v;
}

static inline __attribute__((always_inline)) u32 ink_r8_sp_sub8_sum_u32(u32 s) {
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0xB1, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_update_dpp(0, (int)s, 0x4E, 0xF, 0xF, true);
  s += (u32)__builtin_amdgcn_ds_swizzle((int)s, 0x101F);
  return s;
}

static inline __attribute__((always_inline)) float ink_r8_sp_vclip_scale(
    float ss, float inv_hd, float clamp, u32 norope, float escale, u32 win, u32 pos0, u32 tk) {
  float f;
  if (clamp > 0.0f) {
    f = clamp / (__builtin_sqrtf(ss) + 1e-6f);
    if (f > 1.0f) f = 1.0f;
    if (norope) {
      const float ratio = (float)(pos0 + tk + 1u) / (float)win;
      if (ratio > 1.0f) f *= 1.0f + escale * __builtin_logf(ratio);
    }
  } else {
    f = 1.0f / __builtin_sqrtf(ss * inv_hd + escale);
  }
  return f;
}

/* Wall-clock-guarded sense-reversing inter-WG barrier (returns 1 if poisoned, 0 if clean). */
static inline __attribute__((always_inline)) u32 ink_r8_sp_grid_barrier(
    G u32 *sync, u32 ng, u32 *sense_io,
    ink_r8_sp_u64 k_t0, ink_r8_sp_u64 bar_lim, ink_r8_sp_u64 krn_lim) {
  if (ng <= 1u) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    return 0u;
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();

  const u32 next_sense = *sense_io + 1u;
  if (tid() == 0u) {
    u32 poisoned = __scoped_atomic_load_n(&sync[2u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    if (!poisoned) {
      const u32 prev = __scoped_atomic_fetch_add(&sync[0u], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      if (prev == ng - 1u) {
        __scoped_atomic_store_n(&sync[0u], 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __scoped_atomic_store_n(&sync[1u], next_sense, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      } else {
        const ink_r8_sp_u64 t0 = (ink_r8_sp_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
        for (;;) {
          const u32 cur = __scoped_atomic_load_n(&sync[1u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
          if ((int)(cur - next_sense) >= 0) break;
          if (__scoped_atomic_load_n(&sync[2u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) != 0u) {
            poisoned = 1u;
            break;
          }
          const ink_r8_sp_u64 now = (ink_r8_sp_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
          if ((now - t0) > bar_lim || (now - k_t0) > krn_lim) {
            __scoped_atomic_fetch_or(&sync[2u], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
            poisoned = 1u;
            break;
          }
          __builtin_amdgcn_s_sleep(2);
        }
        const ink_r8_sp_u64 dt_k = ((ink_r8_sp_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83) - t0) / 1000ull;
        __scoped_atomic_fetch_max(&sync[3u], (u32)dt_k, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      }
    }
    ink_r8_sp_lds.poison = poisoned;
  }
  barrier();
  *sense_io = next_sense;
  if (ink_r8_sp_lds.poison != 0u) return 1u;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  return 0u;
}

static inline __attribute__((always_inline)) void ink_r8_sp_do_add(
    const G ink_r8_sp_task_t *tk, u32 g, u32 ng, u32 t) {
#pragma clang fp contract(off)
  const G float *a = (const G float *)tk->p0;
  const G float *b = (const G float *)tk->p1;
  G float *c = (G float *)tk->p2;
  const u32 n4 = tk->n >> 2u;
  const u32 T = tk->a0 ? tk->a0 : 1u;
  const u32 as = tk->a1, bs = tk->a2, cs = tk->a3;
  for (u32 r = 0u; r < T; r++) {
    const G f32x4 *a4 = (const G f32x4 *)((const G u8 *)a + (unsigned long)r * (unsigned long)as);
    const G f32x4 *b4 = (const G f32x4 *)((const G u8 *)b + (unsigned long)r * (unsigned long)bs);
    G f32x4 *c4 = (G f32x4 *)((G u8 *)c + (unsigned long)r * (unsigned long)cs);
    for (u32 i = g * NT + t; i < n4; i += ng * NT) c4[i] = a4[i] + b4[i];
  }
}

static inline __attribute__((always_inline)) void ink_r8_sp_do_copy(
    const G ink_r8_sp_task_t *tk, u32 g, u32 ng, u32 t) {
  const G float *src = (const G float *)tk->p0;
  G float *dst = (G float *)tk->p1;
  const u32 n4 = tk->n >> 2u;
  const u32 T = tk->a0 ? tk->a0 : 1u;
  const u32 ss = tk->a1, ds = tk->a2;
  for (u32 r = 0u; r < T; r++) {
    const G f32x4 *s4 = (const G f32x4 *)((const G u8 *)src + (unsigned long)r * (unsigned long)ss);
    G f32x4 *d4 = (G f32x4 *)((G u8 *)dst + (unsigned long)r * (unsigned long)ds);
    for (u32 i = g * NT + t; i < n4; i += ng * NT) d4[i] = s4[i];
  }
}

static inline __attribute__((always_inline)) void ink_r8_sp_do_gate_q8(
    const G ink_r8_sp_task_t *tk, u32 g, u32 ng, u32 t) {
#pragma clang fp contract(off)
  const G float *a = (const G float *)tk->p0;
  const G float *gt = (const G float *)tk->p1;
  G float *b = (G float *)tk->p2;
  G u8 *q = (G u8 *)tk->p3;
  const u32 nb = tk->n >> 5u;
  const u32 T = tk->a0 ? tk->a0 : 1u;
  const u32 as = tk->a1, gs = tk->a2, bs = tk->a3, qs = tk->a4;
  const u32 sg = t >> 3u, sl = lane() & 7u;
  const u32 total_blks = T * nb;

  for (u32 idx = g * 32u + sg; idx < total_blks; idx += ng * 32u) {
    const u32 r = (T == 1u) ? 0u : (idx / nb);
    const u32 blk = (T == 1u) ? idx : (idx - r * nb);
    const u32 off = (blk << 5u) + (sl << 2u);
    const G float *ar = (const G float *)((const G u8 *)a + (unsigned long)r * (unsigned long)as);
    const G float *gr = (const G float *)((const G u8 *)gt + (unsigned long)r * (unsigned long)gs);
    const f32x4 av = *(const G f32x4 *)(ar + off);
    const f32x4 gv = *(const G f32x4 *)(gr + off);
    f32x4 vv = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
#pragma clang loop unroll(disable)
    for (u32 k = 0u; k < 4u; k++) {
      const float ak = (k == 0u) ? av.x : (k == 1u ? av.y : (k == 2u ? av.z : av.w));
      const float gk = (k == 0u) ? gv.x : (k == 1u ? gv.y : (k == 2u ? gv.z : gv.w));
      const float vk = ak * (1.0f / (1.0f + __builtin_expf(-gk)));
      if (k == 0u) vv.x = vk; else if (k == 1u) vv.y = vk; else if (k == 2u) vv.z = vk; else vv.w = vk;
    }
    if (b) {
      G float *br = (G float *)((G u8 *)b + (unsigned long)r * (unsigned long)bs);
      *(G f32x4 *)(br + off) = vv;
    }
    if (q) {
      G u8 *qr = q + (unsigned long)r * (unsigned long)qs;
      float am = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(vv[0]), __builtin_fabsf(vv[1])),
                                 __builtin_fmaxf(__builtin_fabsf(vv[2]), __builtin_fabsf(vv[3])));
      am = ink_r8_sp_sub8_max_pos(am);
      const float d = am / 127.0f;
      int qi[4u];
      if (d != 0.0f) {
#pragma unroll
        for (u32 k = 0u; k < 4u; k++) qi[k] = (int)__builtin_roundf(vv[k] / d);
      } else {
        qi[0] = qi[1] = qi[2] = qi[3] = 0;
      }
      const u32 pk = ((u32)(u8)qi[0]) | (((u32)(u8)qi[1]) << 8u) |
                     (((u32)(u8)qi[2]) << 16u) | (((u32)(u8)qi[3]) << 24u);
      *(G u32 *)(qr + off) = pk;
      const u32 s = ink_r8_sp_sub8_sum_u32((u32)qi[0] + (u32)qi[1] + (u32)qi[2] + (u32)qi[3]);
      if (sl == 0u) {
        ((G float *)(qr + 32ul * (unsigned long)nb))[blk] = d;
        ((G u32 *)(qr + 36ul * (unsigned long)nb))[blk] = s;
      }
    }
  }
}

static inline __attribute__((always_inline)) void ink_r8_sp_do_rmsnorm_q8(
    const G ink_r8_sp_task_t *tk, u32 g, u32 ng, u32 t) {
  const u32 T = tk->a0 ? tk->a0 : 1u;
  if (tk->n != 2304u) return;
  const G float *x_base = (const G float *)tk->p0;
  const G float *w = (const G float *)tk->p1;
  G float *y_base = (G float *)tk->p2;
  G u8 *q_base = (G u8 *)tk->p3;
  const u32 xs = tk->a1, ys = tk->a2, qs = tk->a3;
  const u32 eps_u = tk->f0;
  const float eps = __builtin_bit_cast(float, eps_u);
  const u32 wv = (u32)__builtin_amdgcn_readfirstlane((int)(t >> 5u));
  const u32 l = lane();

  for (u32 r = g; r < T; r += ng) {
    const G float *x = (const G float *)((const G u8 *)x_base + (unsigned long)r * (unsigned long)xs);
    const f32x4 x0 = *(const G f32x4 *)(x + (t << 2u));
    const f32x4 x1 = *(const G f32x4 *)(x + 1024u + (t << 2u));
    const float x2 = x[2048u + t];
    float ss = ((x0.x * x0.x + x0.y * x0.y) + (x0.z * x0.z + x0.w * x0.w)) +
               ((x1.x * x1.x + x1.y * x1.y) + (x1.z * x1.z + x1.w * x1.w)) + x2 * x2;
    ss = wave_tree_f(ss);
    if (l == 0u) ink_r8_sp_lds.rn_red[wv] = ss;
    lbarrier();
    float tot = 0.0f;
#pragma unroll
    for (u32 j = 0u; j < 8u; j++) tot += ink_r8_sp_lds.rn_red[j];
    const float s = 1.0f / __builtin_sqrtf(tot / 2304.0f + eps);
    G float *y = y_base ? (G float *)((G u8 *)y_base + (unsigned long)r * (unsigned long)ys) : (G float *)0;
    G u8 *q = q_base ? (q_base + (unsigned long)r * (unsigned long)qs) : (G u8 *)0;

    {
      const f32x4 y0 = x0 * s * (*(const G f32x4 *)(w + (t << 2u)));
      if (y) *(G f32x4 *)(y + (t << 2u)) = y0;
      if (q) {
        *(LDSP f32x4 *)&ink_r8_sp_lds.rn_sh[wv][l << 2u] = y0;
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma clang loop unroll(disable)
        for (u32 m = 0u; m < 4u; m++) quant_wave(ink_r8_sp_lds.rn_sh[wv][(m << 5u) + l], (wv << 2u) + m, q, 72u);
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
      }
    }
    {
      const f32x4 y1 = x1 * s * (*(const G f32x4 *)(w + 1024u + (t << 2u)));
      if (y) *(G f32x4 *)(y + 1024u + (t << 2u)) = y1;
      if (q) {
        *(LDSP f32x4 *)&ink_r8_sp_lds.rn_sh[wv][l << 2u] = y1;
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
#pragma clang loop unroll(disable)
        for (u32 m = 0u; m < 4u; m++) quant_wave(ink_r8_sp_lds.rn_sh[wv][(m << 5u) + l], 32u + (wv << 2u) + m, q, 72u);
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
      }
    }
    {
      const float y2 = x2 * s * w[2048u + t];
      if (y) y[2048u + t] = y2;
      if (q) quant_wave(y2, 64u + wv, q, 72u);
    }
    if (r + ng < T) lbarrier();
  }
}

static inline __attribute__((always_inline)) void ink_r8_sp_do_vclip(
    const G ink_r8_sp_task_t *tk, u32 g, u32 ng, u32 t) {
  const G float *a = (const G float *)tk->p0;
  G float *b = (G float *)tk->p1;
  const G float *v = (const G float *)tk->p2;
  const u32 nh = tk->n, hd = tk->a0, T = tk->a1 ? tk->a1 : 1u;
  const u32 win = tk->a2, pos0 = tk->a3, as = tk->a4;
  const u32 bs = (u32)tk->p3;
  const u32 clamp_u = tk->f0, escale_u = tk->f1;
  const float clamp = __builtin_bit_cast(float, clamp_u);
  const float escale = __builtin_bit_cast(float, escale_u);
  const u32 norope = (tk->flags & INK_R8_SP_FLAG_NOROPE) ? 1u : 0u;
  const u32 total_h = T * nh;

  if (hd == 128u) {
    const u32 wv = t >> 5u, l = lane();
    for (u32 idx = g * 8u + wv; idx < total_h; idx += ng * 8u) {
      const u32 r = (T == 1u) ? 0u : (idx / nh);
      const u32 h = (T == 1u) ? idx : (idx - r * nh);
      const G f32x4 *ah = (const G f32x4 *)((const G u8 *)a + (unsigned long)r * (unsigned long)as) + (unsigned long)h * 32ul;
      G f32x4 *bh = (G f32x4 *)((G u8 *)b + (unsigned long)r * (unsigned long)bs) + (unsigned long)h * 32ul;
      const G f32x4 *vp = v ? ((const G f32x4 *)v + (unsigned long)h * 32ul) : (const G f32x4 *)0;
      f32x4 xv = ah[l];
      const f32x4 vv = vp ? vp[l] : (f32x4){1.0f, 1.0f, 1.0f, 1.0f};
      if (clamp > 0.0f && vp) xv *= vv;
      float ss = wave_sum_all((xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w));
      const float f = ink_r8_sp_vclip_scale(ss, 1.0f / 128.0f, clamp, norope, escale, win, pos0, r);
      f32x4 outv = xv * f;
      if (clamp == 0.0f && vp) outv *= vv;
      bh[l] = outv;
    }
  } else if (hd == 16u) {
    const u32 sub = t >> 2u, sl = t & 3u;
    for (u32 idx = g * 64u + sub; idx < total_h; idx += ng * 64u) {
      const u32 r = (T == 1u) ? 0u : (idx / nh);
      const u32 h = (T == 1u) ? idx : (idx - r * nh);
      const G f32x4 *ah = (const G f32x4 *)((const G u8 *)a + (unsigned long)r * (unsigned long)as) + (unsigned long)h * 4ul;
      G f32x4 *bh = (G f32x4 *)((G u8 *)b + (unsigned long)r * (unsigned long)bs) + (unsigned long)h * 4ul;
      const G f32x4 *vp = v ? ((const G f32x4 *)v + (unsigned long)h * 4ul) : (const G f32x4 *)0;
      f32x4 xv = ah[sl];
      const f32x4 vv = vp ? vp[sl] : (f32x4){1.0f, 1.0f, 1.0f, 1.0f};
      if (clamp > 0.0f && vp) xv *= vv;
      float ss = (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
      ss += shfl_xor_f(ss, 2u);
      ss += shfl_xor_f(ss, 1u);
      const float f = ink_r8_sp_vclip_scale(ss, 1.0f / 16.0f, clamp, norope, escale, win, pos0, r);
      f32x4 outv = xv * f;
      if (clamp == 0.0f && vp) outv *= vv;
      bh[sl] = outv;
    }
  }
}

static inline __attribute__((always_inline)) void ink_r8_sp_sconv_finish4(
    f32x4 x0, const G float *v, G float *ring, const G float *res, G float *out,
    u32 i4, u32 pos0, float escale) {
#pragma clang fp contract(off)
  const u32 i0 = i4 << 2u;
  const u32 slot = pos0 & 3u;
  const u32 s1 = (pos0 - 1u) & 3u, s2 = (pos0 - 2u) & 3u, s3 = (pos0 - 3u) & 3u;
  const G f32x4 *rg = (const G f32x4 *)ring + i0;
  const G f32x4 *v4 = (const G f32x4 *)v + i0;
  f32x4 sc = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
#pragma clang loop unroll(disable)
  for (u32 c = 0u; c < 4u; c++) {
    const float xc = (c == 0u) ? x0.x : (c == 1u ? x0.y : (c == 2u ? x0.z : x0.w));
    float h1 = 0.0f, h2 = 0.0f, h3 = 0.0f;
    if (pos0 > 0u) {
      const f32x4 rc = rg[c];
      h1 = ink_r8_sp_pick4(rc, s1);
      if (pos0 >= 2u) h2 = ink_r8_sp_pick4(rc, s2);
      if (pos0 >= 3u) h3 = ink_r8_sp_pick4(rc, s3);
    }
    const f32x4 vc = v4[c];
    const float y = ((xc * vc.w + h1 * vc.z) + h2 * vc.y) + h3 * vc.x;
    const float o = (xc + y) * escale;
    if (c == 0u) sc.x = o; else if (c == 1u) sc.y = o; else if (c == 2u) sc.z = o; else sc.w = o;
    ring[(unsigned long)(i0 + c) * 4ul + slot] = xc;
  }
  const f32x4 rv = ((const G f32x4 *)res)[i4];
  ((G f32x4 *)out)[i4] = rv + sc;
}

static inline __attribute__((always_inline)) void ink_r8_sp_do_sconv_add_t1(
    const G ink_r8_sp_task_t *tk, u32 g, u32 ng, u32 t) {
  const G float *a = (const G float *)tk->p0;
  const G float *v = (const G float *)tk->p1;
  G float *ring = (G float *)tk->p2;
  const G float *res = (const G float *)tk->p3;
  G float *out = (G float *)tk->p4;
  const u32 n4 = tk->n >> 2u;
  const u32 pos0 = tk->a0;
  const u32 escale_u = tk->f0;
  const float escale = __builtin_bit_cast(float, escale_u);
  for (u32 i4 = g * NT + t; i4 < n4; i4 += ng * NT) {
    ink_r8_sp_sconv_finish4(((const G f32x4 *)a)[i4], v, ring, res, out, i4, pos0, escale);
  }
}

static inline __attribute__((always_inline)) void ink_r8_sp_do_situ_q8(
    const G ink_r8_sp_task_t *tk, u32 g, u32 ng, u32 t) {
#pragma clang fp contract(off)
  const G float *a = (const G float *)tk->p0;
  G u8 *q = (G u8 *)tk->p1;
  const u32 eff = tk->n, n_rows = tk->a0;
  const u32 astr = tk->a1, qstr = tk->a2;
  const u32 beta_u = tk->f0, lbeta_u = tk->f1;
  const float beta = __builtin_bit_cast(float, beta_u);
  const float lbeta = __builtin_bit_cast(float, lbeta_u);
  const u32 nb = eff >> 5u;
  const u32 wv = t >> 5u, l = lane();
  const u32 total_blks = n_rows * nb;

  for (u32 idx = g * 8u + wv; idx < total_blks; idx += ng * 8u) {
    const u32 row = idx / nb;
    const u32 blk = idx - row * nb;
    const G float *ar = (const G float *)((const G u8 *)a + (unsigned long)row * (unsigned long)astr);
    const u32 c = (blk << 5u) + l;
    const float gv = ar[c];
    const float uv = ar[eff + c];
    const float gs = beta > 0.0f ? beta * ie_tanhf_s(gv / beta) : gv;
    const float us = lbeta > 0.0f ? lbeta * ie_tanhf_s(uv / lbeta) : uv;
    quant_wave(gs * sigmoidf(gv) * us, blk, q + (unsigned long)row * (unsigned long)qstr, nb);
  }
}

static inline __attribute__((always_inline)) void ink_r8_sp_do_moe_tail_t1(
    const G ink_r8_sp_task_t *tk, u32 g, u32 ng, u32 t) {
#pragma clang fp contract(off)
  const G float *exp_down = (const G float *)tk->p0;
  const G float *rw = (const G float *)tk->p1;
  const G float *v = (const G float *)tk->p2;
  G float *ring = (G float *)tk->p3;
  G float *res_io = (G float *)tk->p4;
  const u32 n4 = tk->n >> 2u;
  const u32 K = tk->a0 ? tk->a0 : 8u;
  const u32 estr = tk->a1;
  const u32 pos0 = tk->a2;
  const u32 escale_u = tk->f0;
  const float escale = __builtin_bit_cast(float, escale_u);

  for (u32 i4 = g * NT + t; i4 < n4; i4 += ng * NT) {
    f32x4 x0 = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 k = 0u; k < K; k++) {
      const float wk = rw[k];
      const f32x4 ek = ((const G f32x4 *)((const G u8 *)exp_down + (unsigned long)k * (unsigned long)estr))[i4];
      x0 = x0 + ek * wk;
    }
    ink_r8_sp_sconv_finish4(x0, v, ring, res_io, res_io, i4, pos0, escale);
  }
}

KERNEL ie_inkling_subspan_glue_r8(
    const G ink_r8_sp_task_t *tasks, u32 n_tasks, u32 ng,
    G u32 *sync, ink_r8_sp_u64 bar_timeout_ticks, ink_r8_sp_u64 krn_timeout_ticks) {
  const u32 g = wgid();
  const u32 t = tid();
  /* Hard WDDM safety clamp: never execute persistent barriers above grid=64! */
  if (ng == 0u || ng > INK_R8_SP_MAX_GRID || g >= ng || n_tasks == 0u || !tasks || !sync) return;
  if (__scoped_atomic_load_n(&sync[2u], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) != 0u) return;

  const ink_r8_sp_u64 bar_lim = bar_timeout_ticks ? bar_timeout_ticks : INK_R8_SP_DEF_BAR_TICKS;
  const ink_r8_sp_u64 krn_lim = krn_timeout_ticks ? krn_timeout_ticks : INK_R8_SP_DEF_KRN_TICKS;
  const ink_r8_sp_u64 k_t0 = (ink_r8_sp_u64)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
  /* Monotonic sense counter: immutable across all ng WGs at entry (no exit reset race!). */
  u32 sense = __scoped_atomic_load_n(&sync[1u], __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);

  for (u32 ti = 0u; ti < n_tasks; ti++) {
    const G ink_r8_sp_task_t *tk = tasks + ti;
    const u32 op = tk->op;
    const u32 flags = tk->flags;
    if (op == INK_R8_SP_OP_ADD)             ink_r8_sp_do_add(tk, g, ng, t);
    else if (op == INK_R8_SP_OP_COPY)       ink_r8_sp_do_copy(tk, g, ng, t);
    else if (op == INK_R8_SP_OP_GATE_Q8)    ink_r8_sp_do_gate_q8(tk, g, ng, t);
    else if (op == INK_R8_SP_OP_RMSNORM_Q8) ink_r8_sp_do_rmsnorm_q8(tk, g, ng, t);
    else if (op == INK_R8_SP_OP_VCLIP)      ink_r8_sp_do_vclip(tk, g, ng, t);
    else if (op == INK_R8_SP_OP_SCONV_ADD)  ink_r8_sp_do_sconv_add_t1(tk, g, ng, t);
    else if (op == INK_R8_SP_OP_MOE_TAIL)   ink_r8_sp_do_moe_tail_t1(tk, g, ng, t);
    else if (op == INK_R8_SP_OP_SITU_Q8)    ink_r8_sp_do_situ_q8(tk, g, ng, t);
    if (flags & INK_R8_SP_FLAG_BARRIER) {
      if (ink_r8_sp_grid_barrier(sync, ng, &sense, k_t0, bar_lim, krn_lim)) return;
    }
  }
}

#undef INK_R8_SP_FLAG_NOROPE
#undef INK_R8_SP_FLAG_BARRIER
#undef INK_R8_SP_OP_SITU_Q8
#undef INK_R8_SP_OP_MOE_TAIL
#undef INK_R8_SP_OP_SCONV_ADD
#undef INK_R8_SP_OP_VCLIP
#undef INK_R8_SP_OP_RMSNORM_Q8
#undef INK_R8_SP_OP_GATE_Q8
#undef INK_R8_SP_OP_COPY
#undef INK_R8_SP_OP_ADD
#undef INK_R8_SP_DEF_KRN_TICKS
#undef INK_R8_SP_DEF_BAR_TICKS
#undef INK_R8_SP_MAX_GRID

#endif /* IE_MEGA */
