/* ie_mk.c -- the megakernel: one persistent launch runs a list of kernel
 * launches (the command list of a decode step, src/hip.c), in order.
 *
 * The kernels are those of ie_kernels.c, compiled here as functions
 * (KERNEL is a static inline function, IE_MEGA: wgid(), tokid() and
 * ngroups_x() read the current workgroup of the list from LDS). Command i:
 * 256 bytes, {kernel id, groups x, groups y, 0} then the kernel arguments in
 * launch layout (tools/mk_gen.py). Workgroup p of the np launched ones runs
 * the workgroups p, p + np, ... of command i (x fastest), then waits at a
 * grid barrier: command i + 1 starts when every workgroup of command i is
 * done, as with one launch after the other on a stream. So every kernel
 * runs the same code on the same data: the output is bitwise equal.
 *
 * The launch must be cooperative (all np workgroups resident: the grid
 * barrier needs them). ws: ws[0] the barrier counter (never reset; base:
 * its value at launch), ws[1] the abort flag (set when a wait times out:
 * then every workgroup leaves, and the host stops). prof (or NULL): per
 * command, the clock after its grid barrier (workgroup 0). trace (or
 * NULL): the per-workgroup timeline of prof/sm_profiler.h.
 * Bytes 232 .. 255 of a command: the prefetch of its barrier (the first
 * bytes of the weights of the next GEMV; src/hip.c mk_flush) and the claim
 * cursor (word 63, zero in the list the host writes). */
#define IE_MEGA 1
#pragma clang diagnostic ignored "-Wunused-function" /* the kernels a decode step does not use */
#define KERNEL_CALL static __attribute__((noinline)) void /* ie_attn_kv: inlined, it made the megakernel spill */
#define KERNEL static inline __attribute__((always_inline)) void /* (noinline: 192 VGPRs, but slower, and still 64 workgroups at most) */
/* the raw-block and heavy multi-token GEMVs: inlined into a loop function per
 * kernel (tools/mk_gen.py mkl_*), which runs a workgroup's bodies of one phase */
#define KERNEL_LOOP static inline __attribute__((always_inline)) void
#define KERNEL_HEAVY static inline __attribute__((always_inline)) void
#include "ie_kernels.c"
#if defined(MKS_INKLING) || defined(MKS_INKLING_VER)
/* Persistent IQ3_XXS grid LUT initialized once in ie_mk prologue (1,024 B).
 * Kept outside mk_ink_lds so IQ3_XXS expert kernels run with fr=1 (zero per-body
 * barriers and zero redundant LUT reloads across 48 layers x 96 WGs). */
static LDS u32 mk_iq3_lut[256] __attribute__((aligned(16)));

/* Shared scratch LDS union for Inkling-2 megakernels (<= 12,496 B).
 * Safe because every kernel using mk_ink_lds runs with fr=0 (ie_mk issues
 * barrier() before each body and mk_grid_sync_pf between phases). */
typedef union {
  struct {
    float m[3][8];
    float l[3][8];
    float o[3][8][128] __attribute__((aligned(16)));
    u32 last;
  } rel_attn; /* 12,484 B (max member) */
  struct {
    float ar_red[256u * 8u];
    float ar_inv[64u];
    float ar_prob[8u][64u];
  } attn_res; /* 10,496 B */
  struct {
    u8 x[64u * 80u] __attribute__((aligned(16)));
    float dx[2u][64u] __attribute__((aligned(32)));
  } lc_wfg; /* 5,632 B */
  struct {
    float red[8];
    float sh[8][128] __attribute__((aligned(16)));
  } rmsnorm; /* 4,128 B */
  struct {
    u32 cnt[256u];
    u32 er[512u];
    u32 wsum[9];
  } moe_sort; /* 3,108 B */
  struct {
    float gv_lds[8u][32u] __attribute__((aligned(16)));
  } q8_small; /* 1,024 B */
  struct {
    float red[4][8] __attribute__((aligned(16)));
  } router_ver; /* 128 B */
} mk_ink_lds_u;
_Static_assert(sizeof(mk_ink_lds_u) <= 12544u, "mk_ink_lds_u exceeds 12.25 KiB budget");
static LDS mk_ink_lds_u mk_ink_lds __attribute__((aligned(32)));

static inline __attribute__((always_inline)) u32 rq_ksign7(const u32 s7) {
  return s7 | (((u32)__builtin_popcount(s7) & 1u) << 7u);
}
static inline __attribute__((always_inline)) const G u32 *rq_ap(const G u8 *p, const u32 o) {
  return (const G u32 *)(((unsigned long)(p + o)) & ~3ul);
}

#include "ie_inkling_r1_g06.inc"
#include "ie_inkling_r1_g07.inc"
#include "ie_inkling_r1_g08.inc"
#include "ie_inkling_r2_g11.inc"
#include "ie_inkling_r2_g13.inc"
#ifdef MKS_INKLING
#include "ie_inkling_r1_g02.inc"
#include "ie_inkling_r1_g03.inc"
#include "ie_inkling_r1_g04.inc"
#include "ie_inkling_r1_g05.inc"
#include "ie_inkling_r3_g19.inc"
#endif
#ifdef MKS_INKLING_VER
#include "ie_inkling_r2_g16.inc"
#include "ie_inkling_r4_attn.inc"
#include "ie_inkling_r4_moe.inc"
#endif
#endif
#include "ie_mk_gen.h"
#include "../prof/sm_profiler.h"
/* the profiler has 32 event types: kernels 0 .. 29 their own, the later ones share 30, 31 the grid barrier */
#define MK_EV_BARRIER 31u
/* 32 event slots: the kernels 0 .. 29 have their own, the later ones share
 * 30 (MK_EV_BARRIER is 31). The raw-block GEMVs pushed the ids past 29, so
 * every build uses the mapping now. */
#define MK_EV(k) ((k) < 30u ? (k) : 30u)
#define MK_CMD 256u

/* IE_MK_VGPR=184 (build flag): cap the megakernel's VGPRs so two workgroups
 * fit per CU again. gfx1201/RDNA4: 192 KiB VGPR file per SIMD, 16 slots; a
 * wave of 32 lanes with V VGPRs costs V*128 B, so waves/SIMD = 1536/V. Every
 * body of this kernel is inlined into one switch, and with amdclang 24 the
 * frame lands at 248 VGPR (6 waves/SIMD) where clang 21 produced 192
 * (8 waves/SIMD, two workgroups per CU). Hence the flag: trade a few spills
 * for the occupancy the design assumes. Default 0: let the compiler decide. */
#ifndef IE_MK_VGPR
#define IE_MK_VGPR 0
#endif
#if IE_MK_VGPR
/* clang 24 marks amdgpu_num_vgpr deprecated in favour of amdgpu_waves_per_eu;
 * it is still the only *hard* cap on the frame, and the deprecation warning
 * would drown the build log. */
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#define IE_MK_ATTR __attribute__((amdgpu_num_vgpr(IE_MK_VGPR)))
#else
#define IE_MK_ATTR
#endif

IE_MK_ATTR __attribute__((amdgpu_kernel, visibility("protected"))) void ie_mk(G u8 *cmds, u32 n, G u32 *ws, u32 base, G unsigned long *prof, G uint64_t *trace) {
  const u32 np = __builtin_amdgcn_grid_size_x() / NT, p = __builtin_amdgcn_workgroup_id_x();
#if defined(MKS_INKLING) || defined(MKS_INKLING_VER)
  mk_iq3_lut[tid()] = ie_rq_iq3xxs_grid[tid()];
  lbarrier();
#endif
  /* IE_MK_TRACE: thread 0 records the timeline of this workgroup */
  const int t0 = trace && tid() == 0u;
  smp_rec tr = {0};
  if (t0) {
    const smp_info ti = smp_parse(trace);
    tr = smp_rec_open(&ti, p, 0u);
  }
  unsigned long ts = 0;
  u32 nbar = 0, off = 0; /* off: the workgroups of the phase before command i, mod np */
  for (u32 i = 0; i < n; i++) {
    const G u32 *h = (const G u32 *)(cmds + (unsigned long)i * MK_CMD);
    const u32 kid = h[0], gx = h[1], tot = gx * h[2];
    const G u8 *a = cmds + (unsigned long)i * MK_CMD + 16u;
    /* the workgroups of a phase are dealt out in one sequence: workgroup p
     * takes the phase indices p, p + np, ... */
    /* word 3, bit 1: the kernel uses no LDS and no workgroup barrier
     * (src/hip.c mk_free): its waves go on to the next workgroup body
     * without meeting; else the workgroup meets before each body (the
     * previous body is done with its LDS) */
    const u32 fr = h[3] & 2u, w = tid() >> 5;
    if (mk_loop_kernel(kid)) {
      /* all of this workgroup's bodies of the phase in one call: the call
       * saves and restores its registers once, not once per body */
      const u32 v0 = (p + np - off) % np;
      if (v0 < tot) {
        if (t0) ts = smp_time_ns();
        mk_dispatch_loop(kid, a, v0, np, tot, gx, fr);
        if (t0) smp_rec_range(&tr, MK_EV(kid), ts, smp_time_ns());
      }
    } else
    for (u32 v = (p + np - off) % np; v < tot; v += np) {
      if (!fr) barrier();
      if ((tid() & 31u) == 0u) mk_vid[w][0] = v % gx, mk_vid[w][1] = v / gx, mk_vid[w][2] = gx;
      /* IE_MK_TRACE: one range per workgroup body, on the timeline of
       * workgroup p (prof/sm_profiler.h; thread 0 records) */
      if (t0) ts = smp_time_ns();
      mk_dispatch(kid, a);
      if (t0) smp_rec_range(&tr, MK_EV(kid), ts, smp_time_ns());
    }
    /* word 3 of the next command: 1 = it joins this phase (independent
     * GEMVs, src/hip.c mk_join): no barrier */
    const u32 join = i + 1u < n ? ((const G u32 *)(cmds + (unsigned long)(i + 1u) * MK_CMD))[3] & 1u : 0u;
    if (i + 1u < n && !join) {
      nbar++, off = 0;
      /* the prefetch of this barrier: words 58..59 the address, 60 the bytes,
       * 61 the command whose word 63 is the claim cursor */
      const unsigned long pa = ((unsigned long)h[59] << 32) | h[58];
      G u32 *cur = (G u32 *)(cmds + (unsigned long)h[61] * MK_CMD) + 63;
      if (t0) ts = smp_time_ns();
      if (!mk_grid_sync_pf(ws, base + np * nbar, cur, (const G u8 *)pa, h[60])) {
        if (t0) smp_rec_close(&tr);
        return;
      }
      if (t0) smp_rec_range(&tr, MK_EV_BARRIER, ts, smp_time_ns());
    }
    if (join) off = (off + tot) % np;
    /* IE_MK_PROF: the time (100 MHz clock) when command i is done everywhere */
    if (prof && p == 0u && tid() == 0u) prof[i] = __builtin_amdgcn_s_sendmsg_rtnl(0x83);
  }
  if (t0) smp_rec_close(&tr);
}
