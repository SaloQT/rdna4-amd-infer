/* hip.c -- the GPU backend: HIP runtime loaded with dlopen at run time.
 * No ROCm headers are needed to build: the few HIP C API functions used are
 * declared here. The kernels are in build/ie_kernels.hsaco (kernels/ie_kernels.c).
 * Not tested here (no AMD GPU in the build machine). */
#ifndef _WIN32
#include <dlfcn.h>
#endif
#include <math.h>
#include <string.h>
#include <time.h>

#include "../core/ie_core.h"
#include "backend.h"
#include "common.h"
uint32_t spec_verify_max(void); /* src/gen.c */

typedef int hipError_t; /* 0 = hipSuccess */
typedef void *hipModule_t, *hipFunction_t, *hipStream_t, *hipEvent_t;
enum { hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2, hipMemcpyDeviceToDevice = 3 };

#define KV_SLACK 256u /* rows of the KV cache mapped beyond the last position (a kernel may read a tile past it) */
/* hipMemAllocationProp and hipMemAccessDesc of the virtual memory API (hip_runtime_api.h) */
typedef struct {
  int type, handle_type, loc_type, loc_id;
  void *win32;
  unsigned char compression, rdma;
  unsigned short usage;
} hipMemAllocProp;
typedef struct {
  int loc_type, loc_id, flags;
} hipMemAccDesc;
static struct {
  void *lib;
  hipError_t (*Init)(unsigned);
  /* virtual memory (optional: NULL if the runtime has none) */
  hipError_t (*MemAddressReserve)(void **, size_t, size_t, void *, unsigned long long);
  hipError_t (*MemAddressFree)(void *, size_t);
  hipError_t (*MemCreate)(void **, size_t, const hipMemAllocProp *, unsigned long long);
  hipError_t (*MemRelease)(void *);
  hipError_t (*MemMap)(void *, size_t, size_t, void *, unsigned long long);
  hipError_t (*MemUnmap)(void *, size_t);
  hipError_t (*MemSetAccess)(void *, size_t, const hipMemAccDesc *, size_t);
  hipError_t (*MemGetAllocationGranularity)(size_t *, const hipMemAllocProp *, int);
  hipError_t (*SetDevice)(int);
  hipError_t (*GetDeviceCount)(int *);
  hipError_t (*Malloc)(void **, size_t);
  hipError_t (*MemGetInfo)(size_t *, size_t *);
  hipError_t (*Free)(void *);
  hipError_t (*Memcpy)(void *, const void *, size_t, int);
  hipError_t (*MemcpyAsync)(void *, const void *, size_t, int, hipStream_t);
  hipError_t (*HostMalloc)(void **, size_t, unsigned);
  hipError_t (*HostFree)(void *);
  hipError_t (*Memset)(void *, int, size_t);
  hipError_t (*ModuleLoad)(hipModule_t *, const char *);
  hipError_t (*ModuleLoadData)(hipModule_t *, const void *);
  hipError_t (*ModuleUnload)(hipModule_t);
  hipError_t (*ModuleGetFunction)(hipFunction_t *, hipModule_t, const char *);
  hipError_t (*ModuleLaunchKernel)(hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                   unsigned, hipStream_t, void **, void **);
  hipError_t (*DeviceSynchronize)(void);
  hipError_t (*EventCreate)(hipEvent_t *);
  hipError_t (*EventRecord)(hipEvent_t, hipStream_t);
  hipError_t (*EventQuery)(hipEvent_t);
  hipError_t (*EventSynchronize)(hipEvent_t);
  hipError_t (*EventElapsedTime)(float *, hipEvent_t, hipEvent_t);
  hipError_t (*EventDestroy)(hipEvent_t);
  const char *(*GetErrorString)(hipError_t);
  hipError_t (*StreamCreate)(hipStream_t *);
  hipError_t (*StreamBeginCapture)(hipStream_t, int);
  hipError_t (*StreamEndCapture)(hipStream_t, void **);
  hipError_t (*GraphInstantiate)(void **, void *, void *, void *, size_t);
  hipError_t (*GraphLaunch)(void *, hipStream_t);
  hipError_t (*GraphDestroy)(void *);
  hipError_t (*GraphExecDestroy)(void *);
  hipError_t (*StreamSynchronize)(hipStream_t);
  hipError_t (*StreamWaitEvent)(hipStream_t, hipEvent_t, unsigned);
  hipError_t (*StreamCreateWithFlags)(hipStream_t *, unsigned);
  hipError_t (*StreamDestroy)(hipStream_t);
  hipError_t (*ModuleLaunchCooperativeKernel)(hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned,
                                              unsigned, unsigned, hipStream_t, void **);
  hipError_t (*DeviceGetAttribute)(int *, int, int);
  hipError_t (*ModuleOccupancy)(int *, hipFunction_t, int, size_t);
} H;
static hipStream_t cur_stream; /* the stream of all launches (NULL: the default stream) */

#define HIP(call)                                                                                            \
  do {                                                                                                       \
    hipError_t e_ = (call);                                                                                  \
    if (e_ != 0) ie_die("HIP error %d (%s) in %s", e_, H.GetErrorString ? H.GetErrorString(e_) : "?", #call); \
  } while (0)

#ifdef _WIN32
/* IE_VRAM_TRACE=1: after each allocation of 16 MiB or more (device, and pinned host arrays), this process's GPU memory
 * as Windows sees it (DXGI QueryVideoMemoryInfo of adapter 0: local = in VRAM, non-local = in system memory, and the
 * local budget), as tools/vram_budget.py prints it. */
typedef struct {
  unsigned long a;
  unsigned short b, c;
  unsigned char d[8];
} ie_guid;
typedef struct {
  unsigned long long budget, usage, avail, cur;
} ie_vmi;
typedef long(__stdcall *ie_mkfac)(const ie_guid *, void **);
typedef long(__stdcall *ie_enum1)(void *, unsigned, void **);
typedef long(__stdcall *ie_qi)(void *, const ie_guid *, void **);
typedef long(__stdcall *ie_qvmi)(void *, unsigned, int, ie_vmi *);
static void *vt_ad3;
static int vt_on = -1;
static void vram_trace(const char *what, size_t bytes) {
  if (vt_on < 0) {
    const char *e = getenv("IE_VRAM_TRACE");
    vt_on = e && e[0] == '1';
  }
  if (!vt_on || (bytes && bytes < ((size_t)16 << 20))) return; /* bytes 0: a point in time (gpu_run) */
  if (!vt_ad3) {
    static const ie_guid f1 = {0x770aae78, 0xf26f, 0x4dba, {0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}};
    static const ie_guid a3 = {0x645967a4, 0x1392, 0x4310, {0xa7, 0x98, 0x80, 0x53, 0xce, 0x3e, 0x93, 0xfd}};
    void *dx = dlopen("dxgi.dll", RTLD_NOW), *fac = NULL, *ad = NULL;
    ie_mkfac mk = dx ? (ie_mkfac)dlsym(dx, "CreateDXGIFactory1") : NULL;
    if (!mk || mk(&f1, &fac) || ((ie_enum1)(*(void ***)fac)[12])(fac, 0, &ad) || ((ie_qi)(*(void ***)ad)[0])(ad, &a3, &vt_ad3)) {
      vt_on = 0;
      return;
    }
  }
  ie_vmi lo = {0}, nl = {0};
  const ie_qvmi q = (ie_qvmi)(*(void ***)vt_ad3)[14];
  q(vt_ad3, 0, 0, &lo), q(vt_ad3, 0, 1, &nl);
  fprintf(stderr, "vram: %-6s %8.1f MiB -> VRAM %.2f GiB, system %.2f GiB, budget %.2f GiB\n", what, bytes / 1048576.0,
          lo.usage / 1073741824.0, nl.usage / 1073741824.0, lo.budget / 1073741824.0);
}
#else
static void vram_trace(const char *what, size_t bytes) { (void)what, (void)bytes; }
#endif

/* The live device allocations (hipMalloc, hipMemCreate), printed after loading: Windows lowers the VRAM budget of a
 * process with more than about 100 of them (slab_take). */
static uint32_t n_dalloc;
static hipError_t (*real_malloc)(void **, size_t), (*real_free)(void *);
static hipError_t (*real_mcreate)(void **, size_t, const hipMemAllocProp *, unsigned long long), (*real_mrelease)(void *);
static hipError_t count_malloc(void **p, size_t n) {
  const hipError_t e = real_malloc(p, n);
  if (e == 0) n_dalloc++, vram_trace("device", n);
  return e;
}
static hipError_t count_free(void *p) {
  if (p) n_dalloc--;
  return real_free(p);
}
static hipError_t count_mcreate(void **h, size_t n, const hipMemAllocProp *pr, unsigned long long f) {
  const hipError_t e = real_mcreate(h, n, pr, f);
  if (e == 0) n_dalloc++;
  return e;
}
static hipError_t count_mrelease(void *h) {
  n_dalloc--;
  return real_mrelease(h);
}
static void hip_load(void) {
  if (H.lib) return;
#ifdef _WIN32
  const char *names[] = {"amdhip64_7.dll", "amdhip64.dll", "amdhip64_6.dll"};
#else
  const char *names[] = {"libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.7", "/opt/rocm/lib/libamdhip64.so"};
#endif
  for (unsigned i = 0; i < sizeof names / sizeof names[0] && !H.lib; i++) H.lib = dlopen(names[i], RTLD_NOW);
  if (!H.lib)
#ifdef _WIN32
    ie_die("GPU backend: cannot load amdhip64_7.dll (the ROCm HIP runtime). Put the ROCm wheel "
           "bin directory on PATH.");
#else
    ie_die("GPU backend: cannot load libamdhip64.so (the ROCm HIP runtime). Install ROCm or set "
           "LD_LIBRARY_PATH=/opt/rocm/lib. (%s)", dlerror());
#endif
#define SYM(f, n)                                              \
  do {                                                         \
    *(void **)&H.f = dlsym(H.lib, n);                          \
    if (!H.f) ie_die("GPU backend: %s not found in libamdhip64", n); \
  } while (0)
  SYM(Init, "hipInit");
  SYM(SetDevice, "hipSetDevice");
  SYM(GetDeviceCount, "hipGetDeviceCount");
  SYM(Malloc, "hipMalloc");
  SYM(MemGetInfo, "hipMemGetInfo");
  SYM(Free, "hipFree");
  SYM(HostFree, "hipHostFree");
  SYM(Memcpy, "hipMemcpy");
  SYM(Memset, "hipMemset");
  SYM(ModuleLoad, "hipModuleLoad");
  SYM(ModuleLoadData, "hipModuleLoadData");
  SYM(ModuleUnload, "hipModuleUnload");
  SYM(ModuleGetFunction, "hipModuleGetFunction");
  SYM(ModuleLaunchKernel, "hipModuleLaunchKernel");
  SYM(DeviceSynchronize, "hipDeviceSynchronize");
  SYM(EventCreate, "hipEventCreate");
  SYM(EventRecord, "hipEventRecord");
  SYM(EventSynchronize, "hipEventSynchronize");
  SYM(EventElapsedTime, "hipEventElapsedTime");
  SYM(EventDestroy, "hipEventDestroy");
  SYM(GetErrorString, "hipGetErrorString");
  SYM(StreamCreate, "hipStreamCreate");
  SYM(StreamBeginCapture, "hipStreamBeginCapture");
  SYM(StreamEndCapture, "hipStreamEndCapture");
  SYM(GraphInstantiate, "hipGraphInstantiate");
  SYM(GraphLaunch, "hipGraphLaunch");
  SYM(StreamSynchronize, "hipStreamSynchronize");
  SYM(StreamWaitEvent, "hipStreamWaitEvent");
  SYM(StreamCreateWithFlags, "hipStreamCreateWithFlags");
  SYM(ModuleLaunchCooperativeKernel, "hipModuleLaunchCooperativeKernel");
  SYM(DeviceGetAttribute, "hipDeviceGetAttribute");
  SYM(ModuleOccupancy, "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
  SYM(MemcpyAsync, "hipMemcpyAsync");
  SYM(HostMalloc, "hipHostMalloc");
#undef SYM
  *(void **)&H.EventQuery = dlsym(H.lib, "hipEventQuery");
  *(void **)&H.MemAddressReserve = dlsym(H.lib, "hipMemAddressReserve");
  *(void **)&H.MemAddressFree = dlsym(H.lib, "hipMemAddressFree");
  *(void **)&H.MemCreate = dlsym(H.lib, "hipMemCreate");
  *(void **)&H.MemRelease = dlsym(H.lib, "hipMemRelease");
  *(void **)&H.MemMap = dlsym(H.lib, "hipMemMap");
  *(void **)&H.MemUnmap = dlsym(H.lib, "hipMemUnmap");
  *(void **)&H.MemSetAccess = dlsym(H.lib, "hipMemSetAccess");
  *(void **)&H.MemGetAllocationGranularity = dlsym(H.lib, "hipMemGetAllocationGranularity");
  *(void **)&H.StreamDestroy = dlsym(H.lib, "hipStreamDestroy");
  *(void **)&H.GraphDestroy = dlsym(H.lib, "hipGraphDestroy");
  *(void **)&H.GraphExecDestroy = dlsym(H.lib, "hipGraphExecDestroy");
  real_malloc = H.Malloc, H.Malloc = count_malloc, real_free = H.Free, H.Free = count_free;
  if (H.MemCreate && H.MemRelease) real_mcreate = H.MemCreate, H.MemCreate = count_mcreate, real_mrelease = H.MemRelease, H.MemRelease = count_mrelease;
}

/* Kernels of ie_kernels.hsaco. */
enum { K_EMBED_Q4, K_EMBED_F32, K_RMSNORM, K_QUANT, K_GEMV_Q4, K_GEMV_Q8, K_GEMV_F32, K_BIAS, K_ADD, K_SWIGLU, K_ROPE, K_KV,
       K_ATTN, K_ARGMAX, K_ROPE_KV, K_ATTN_SPLIT, K_EMBED_Q8, K_QKN_ROPE_KV, K_GDN, K_GEMV_Q8_T, K_GEMV_Q8_TR, K_GEMM_Q8, K_RING_STORE, K_ATTN_PF, K_GEMM_H, K_GNORM, K_GDN1, K_RMSNORM_T, K_ATTN_PFG, K_GDN_PREP, K_GDN_WY, K_GDN_SEQ, K_GEMM_Q8R, K_GEMV_Q4K, K_GEMV_Q4K_TR, K_GEMM_Q4KR, K_GEMV_F32_T, K_GEMV_Q4K_TR8, K_GEMV_Q6K, K_GEMV_Q6K_TR, K_GEMV_Q6K_TR8, K_GEMM_Q6KR, K_GEMV_F32_TW, K_GEMV_Q4K_TS2, K_GEMV_Q6K_TS2, K_ATTN_SPLIT_T, K_TOPK_PART, K_TOPK_MERGE, K_ATTN_KV, K_ATTN_FA, K_ATTN_FA_Q8, K_ATTN_FA2, K_GEMM_F32_S, K_GEMM_Q4KR_SB, K_REQUANT_SB, K_Q8_F16, K_GEMM_Q4K_H, K_GEMM_Q4K_H256, K_COPY_ROWS, K_DF_CONV, K_DF_QKV, K_DF_ATTN, K_GEMV_Q5K, K_GEMV_IQ4XS, K_GEXP, K_MOE_ROUTE, K_MOE_SUM, K_QKN_ROPE_KV2, K_ATTN_SW, K_ATTN_GQ, K_GEXP_T, K_MOE_SORT, K_ATTN_FA_SW, K_GEMM_KX, K_GEMV_RAW_Q2K, K_GEMV_RAW_IQ2XXS, K_GEMV_RAW_IQ2XS, K_GEMV_RAW_IQ3XXS, K_GEMV_RAW_IQ1S, K_GEMV_RAW_IQ2S, K_GEMV_RAW_IQ1M, K_GEMV_RAW_IQ3S,
       K_GEMV_RAW_Q2K_TR, K_GEMV_RAW_IQ2XXS_TR, K_GEMV_RAW_IQ2XS_TR, K_GEMV_RAW_IQ3XXS_TR, K_GEMV_RAW_IQ1S_TR, K_GEMV_RAW_IQ2S_TR, K_GEMV_RAW_IQ1M_TR, K_GEMV_RAW_IQ3S_TR,
       K_GEMV_IQ4XS_TR, K_GDN_REPLAY,
       K_GEMV_IQ4XS_TR8,
       K_GEMV_RAW_Q2K_TR4, K_GEMV_RAW_IQ2XXS_TR4, K_GEMV_RAW_IQ2XS_TR4, K_GEMV_RAW_IQ3XXS_TR4, K_GEMV_RAW_IQ1S_TR4, K_GEMV_RAW_IQ2S_TR4, K_GEMV_RAW_IQ1M_TR4, K_GEMV_RAW_IQ3S_TR4,
       K_GEMV_RAW_Q2K_TR8, K_GEMV_RAW_IQ2XXS_TR8, K_GEMV_RAW_IQ2XS_TR8, K_GEMV_RAW_IQ3XXS_TR8, K_GEMV_RAW_IQ1S_TR8, K_GEMV_RAW_IQ2S_TR8, K_GEMV_RAW_IQ1M_TR8, K_GEMV_RAW_IQ3S_TR8,
       K_GEMM_RAW_Q2K, K_GEMM_RAW_IQ2XXS, K_GEMM_RAW_IQ2XS, K_GEMM_RAW_IQ3XXS, K_GEMM_RAW_IQ1S, K_GEMM_RAW_IQ2S, K_GEMM_RAW_IQ1M, K_GEMM_RAW_IQ3S,
       K_GEMM_RAWI_Q2K, K_GEMM_RAWI_IQ2XXS, K_GEMM_RAWI_IQ2XS, K_GEMM_RAWI_IQ3XXS, K_GEMM_RAWI_IQ1S, K_GEMM_RAWI_IQ2S, K_GEMM_RAWI_IQ1M, K_GEMM_RAWI_IQ3S, K_ATTN_LONG, K_ATTN_PL, K_TOPK_MAX, K_TOPK_SEL, K_GEMV_RAW_TR8_M, K_RMSNORM_F, K_GDN1P, K_GEMV_Q8_TS, K_ATTN_LONG_E, K_ATTN_SPLIT_TP, K_DF_ATTN_G, K_GEMV_Q4K_TS16, K_GEMV_Q4K_TS4, K_GEMV_Q6K_TS16, K_GEMV_Q6K_TS4, K_COPY_LIST, K_QKN_ROPE_KV_ROT, K_ATTN_UNROT, K_VRES, K_GATE, K_GEMV_RAW_TR4_M, K_GEMV_RAW_SWIGLU, K_GEMV_RAW_SWIGLU_TR4, K_GEMV_RAW_SWIGLU_TR8,
       K_GEMV_IQ4XS_TR16,
       K_GEMM_RAW32_Q2K, K_GEMM_RAW32_IQ2XXS, K_GEMM_RAW32_IQ2XS, K_GEMM_RAW32_IQ3XXS, K_GEMM_RAW32_IQ1S, K_GEMM_RAW32_IQ2S, K_GEMM_RAW32_IQ1M, K_GEMM_RAW32_IQ3S,
       K_GEMM_RQ_SWIGLU,
       K_GEMM_RQ_T16_Q2K, K_GEMM_RQ_T16_IQ2XXS, K_GEMM_RQ_T16_IQ2XS, K_GEMM_RQ_T16_IQ3XXS, K_GEMM_RQ_T16_IQ1S, K_GEMM_RQ_T16_IQ2S, K_GEMM_RQ_T16_IQ1M, K_GEMM_RQ_T16_IQ3S,
       K_GEMM_RQ_T16_M, K_GEMM_RQ_T16_SWIGLU,
       K_GDN_PREP_RING, K_GDN_WY_WMMA, K_GDN_SEQ_WMMA_GNORM,
       K_RMSNORM_MT, K_MTP_PRENORM_Q8,
       K_GEMV_RAW_Q2K_TR16, K_GEMV_RAW_IQ2XXS_TR16, K_GEMV_RAW_IQ2XS_TR16, K_GEMV_RAW_IQ3XXS_TR16, K_GEMV_RAW_IQ1S_TR16, K_GEMV_RAW_IQ2S_TR16, K_GEMV_RAW_IQ1M_TR16, K_GEMV_RAW_IQ3S_TR16,
       K_GEMV_RAW_TR16_M, K_GEMV_RAW_SWIGLU_TR16, K_ATTN_LONG_MP, K_Q_SCALE_F16, K_GEMV_Q4K_SWIGLU,
       K_GEMV_IQ4NL, K_GEMV_IQ4NL_TR, K_GEXP_RAW, K_GEXP_RAW_T, K_GEMM_RQ_EXP_IQ3XXS, K_ADD_T, K_KV_T,
       K_VCLIP, K_ATTN_RES, K_LOOP_GATE, K_SCONV, K_SCONV_RING, K_REL_ATTN, K_REL_ATTN_PF, K_SITU, K_LOGIT_CAP,
       K_SALUKI_LORA_B, K_SALUKI_IQ2XXS_TR8, K_SALUKI_ATTN_LONG_MP, K_SALUKI_IQ1S_Q8_TR8, K_SALUKI_GDN1P,
       K_SALUKI_IQ2S_TR8, K_SALUKI_IQ1S_TR8, K_SALUKI_IQ1S_DOWN_TR8, K_SALUKI_LORA_B_T32, K_SALUKI_IQ2XXS_TR9,
       K_SALUKI_LORA_A_T4, K_SALUKI_LORA_A_T1, K_SALUKI_LORA_A_V4,
       K_SALUKI_Q2K_TR8_R4, K_SALUKI_IQ2XS_TR9, K_SALUKI_IQ1M_TR8_R4,
       K_SALUKI_IQ3S_TR8_R4, K_SALUKI_IQ3S_TR8_R2, K_SALUKI_IQ3XXS_TR9,
       K_SALUKI_ATTN_LONG_MP_TR, K_SALUKI_GDN1P_TR,
       K_SALUKI_RQ128_Q2K, K_SALUKI_RQ128_IQ2XXS, K_SALUKI_RQ128_IQ2XS, K_SALUKI_RQ128_IQ3XXS,
       K_SALUKI_RQ128_IQ1S, K_SALUKI_RQ128_IQ2S, K_SALUKI_RQ128_IQ1M, K_SALUKI_RQ128_IQ3S,
       K_SALUKI_RQ64_Q2K, K_SALUKI_RQ64_IQ2XXS, K_SALUKI_RQ64_IQ2XS, K_SALUKI_RQ64_IQ3XXS,
       K_SALUKI_RQ64_IQ1S, K_SALUKI_RQ64_IQ2S, K_SALUKI_RQ64_IQ1M, K_SALUKI_RQ64_IQ3S,
       K_SALUKI_ATTN_PL, K_SALUKI_SWIGLU_V4, K_SALUKI_GDN_SEQ_WMMA_GNORM,
       K_SALUKI_RQ128_IQ2XXS_SB, K_SALUKI_RQ128_Q2K_PIPE,
       K_SALUKI_IQ2XXS_TR9_PIPE1,
       K_SALUKI_ATTN_LONG_MP_PIPE1, K_SALUKI_ATTN_LONG_MP_TR_NOSPILL,
       K_SALUKI_RQ128_IQ2S_PIPE, K_SALUKI_RQ128_IQ3S_PIPE, K_SALUKI_RQ128_IQ2XS_PIPE,
       K_SALUKI_RQ128_IQ1M_PIPE, K_SALUKI_RQ128_IQ3XXS_PIPE,
       K_SALUKI_IQ2S_TR8_PIPE1, K_SALUKI_IQ2XS_TR9_PIPE1, K_SALUKI_IQ3XXS_TR9_PIPE1,
       K_SALUKI_IQ1S_TR8_PIPE1, K_SALUKI_Q2K_TR8_R4_PIPE1, K_SALUKI_IQ1M_TR8_R4_PIPE1,
       K_SALUKI_IQ3S_TR8_R2_PIPE1,
       K_SALUKI_RQ128_IQ2XXS_R10, K_SALUKI_RQ128_Q2K_R10, K_SALUKI_RQ128_IQ1S_R10,
       K_SALUKI_ATTN_PL_R10, K_SALUKI_ATTN_LONG_MP_R10, K_SALUKI_ATTN_LONG_MP_TR_R10,
       K_SALUKI_IQ2XXS_R4_LDS, K_SALUKI_IQ2XXS_TR9_R10, K_SALUKI_IQ1S_TR9_R10,
       K_SALUKI_Q2K_R4_LDS, K_SALUKI_IQ2S_R4_LDS, K_SALUKI_IQ2XS_R4_LDS,
       K_SALUKI_IQ1M_TR9_R10, K_SALUKI_IQ3S_R4_LDS,
       K_SALUKI_RQ128_IQ2XXS_R11, K_SALUKI_RQ128_IQ2S_R11, K_SALUKI_GEMM_Q8R_R11,
       K_SALUKI_QKN_ROPE_KV_R11, K_SALUKI_ATTN_LONG_MP_R11, K_SALUKI_GDN_WY_WMMA_R11,
       K_SALUKI_GEMV_RAW_TR8_M_R11, K_SALUKI_Q2K_TR9_R11, K_SALUKI_LORA_B_R11,
       K_SALUKI_TOPK_R11, K_SALUKI_END,
       K_INK_REL_ATTN_PF = K_SALUKI_END, K_INK_REL_ATTN_DEC,
       K_INK_GEXP_RAW_IQ3XXS, K_INK_GEMM_RQ_EXP_IQ3XXS,
       K_INK_GEXP_IQ4NL, K_INK_GEMM_IQ4NL_EXP,
       K_INK_GEMM_Q8_SMALL, K_INK_GEMM_Q8_NB2, K_INK_GEMV_Q8_NB2, K_INK_GEMV_Q8_SMALL,
       K_INK_ATTN_RES, K_INK_RMSNORM_MT,
       K_INK_MOE_SORT, K_INK_SITU, K_INK_MOE_SUM,
       K_INK_LOOP_GATE, K_INK_SCONV, K_INK_VCLIP,
       K_INK_GEMM_RQ_EXP_IQ3XXS_R2, K_INK_GEXP_IQ3XXS_SITU_R2,
       K_INK_GEMV_Q8_MED_R2, K_INK_GEMV_Q8_MULTI2_R2, K_INK_GEMM_Q8_MED_R2,
       K_INK_LC_PRENORM_Q8_R2, K_INK_LC_WFG_GATE_R2,
       K_INK_KV_SCONV_VCLIP_STORE_R2, K_INK_KV_RING_STORE_R2, K_INK_KV_SCONV_VCLIP_STORE_CTX_R2,
       K_INK_SCONV_ADD_R2, K_INK_MOE_SUM_SCONV_ADD_R2, K_INK_GATE_Q8_R2,
       K_INK_GEMM_F32_ROUTER_R2, K_INK_ROUTER_ROUTE_DEC_R2,
       K_INK_GEMM_IQ4NL_EXP_R2,
       K_INK_EMBED_Q8_CTX_R2, K_INK_EMBED_Q8_BATCH_R2,
       K_INK_VCLIP_CTX_R2, K_INK_SCONV_CTX_R2, K_INK_SCONV_ADD_CTX_R2,
       K_INK_MOE_SUM_SCONV_ADD_CTX_R2, K_INK_KV_STORE_CTX_R2,
       K_INK_REL_ATTN_DEC_CTX_R2,
       K_INK_GEMM_RQ_EXP_IQ3XXS_R3, K_INK_GEXP_IQ3XXS_SITU_R3,
       K_INK_GEXP_RAW_IQ3XXS_R3, K_INK_GEXP_IQ4NL_R3,
       K_INK_GEMM_F32_ROUTER_R3, K_INK_REL_ATTN_PF_R3,
       K_INK_REL_ATTN_VER_R4, K_INK_SCONV_HIST_R4, K_INK_SCONV_COMMIT_R4,
       K_INK_GEXP_RAW_IQ3XXS_VER_R4, K_INK_GEXP_IQ4NL_VER_R4, K_INK_GEMV_F32_ROUTER_VER_R4,
       K_DS_DFLARE_MIX_R4, K_DS_BLOCK_INIT_R4, K_DS_ATTN_R4,
       K_DS_CONCAT_Q8_R4, K_DS_RNN_STEP_R4, K_DS_ADD_Q8_R4,
       K_DS_BLOCK_INIT_ZSYNC_R5, K_DS_CONCAT_Q8_ZSYNC_R5, K_DS_RNN_STEP_ZSYNC_R5,
       K_DS_ADD_Q8_ZSYNC_R5, K_DS_GEMV_P_Q4_ZSYNC_R5, K_DS_CAP_ARGMAX_ZSYNC_R5,
       K_DS_GEMV_Q8_ZSYNC_R5, K_DS_GEMV_Q4K_ZSYNC_R5, K_DS_GEMV_Q6K_ZSYNC_R5,
       K_INK_GEMV_Q8_SMALL_VER_R5,
       K_INK_ATTN_RES_4OUT_R6, K_INK_LC_PRENORM_Q8_4X_R6, K_INK_LC_DOWN_4X_R6, K_INK_LC_WFG_GATE_NORM_R6,
       K_INK_MOE_SORT_COMPACT_R6, K_INK_GEXP_RAW_IQ3XXS_COMPACT_R6, K_INK_GEXP_IQ4NL_COMPACT_R6,
       K_INK_GEXP_RAW_IQ3XXS_SLOT_R6, K_INK_GEXP_IQ4NL_SLOT_R6,
       K_INK_GEMM_RQ_EXP_IQ3XXS_SITU_R6, K_INK_REL_ATTN_VER_QTILE_R6,
       K_INK_ROUTER_ROUTE_EXACT_T1_R7, K_INK_ROUTER_ROUTE_SORT_VER_R7,
       K_INK_KV_SCONV_VCLIP_STORE_R7, K_INK_SCONV_POST_R7,
       K_INK_MOE_SUM_SCONV_ADD_R7, K_INK_EXP_DOWN_TAIL_T1_R7, K_INK_EXP_DOWN_TAIL_VER_R7,
       K_INK_QKV_QCLIP_R7, K_INK_REL_CLIP_SINK_GATE_R7,
       K_INK_GEXP_IQ3XXS_SITU_COMPACT_R7, K_INK_ATTN_RES_NORM_Q8_R7, K_INK_WO_SCONV_POST_R7,
       K_INK_GEMV_Q4K_HEAD_VER_R7,
       K_INK_REL_ATTN_VER_SWA_QTILE_R7, K_INK_REL_ATTN_VER_GLOB_WMMA_R7,
       K_INK_REL_ATTN_PF_R7, K_INK_REL_ATTN_PF_SPLIT_R7,
       K_INK_GEXP_IQ3XXS_FLUTE_SITU_T1_R7, K_INK_GEXP_IQ3XXS_FLUTE_SITU_COMPACT_R7,
       K_INK_GEXP_IQ4NL_BW_T1_R7, K_INK_GEXP_IQ4NL_BW_COMPACT_R7, K_INK_GEMV_Q8_BW_R7,
       K_INK_Q4K_HEAD_CAP_TOPK_R7,
       K_INK_DS_BASE_Q4K_IDMAP_R7, K_INK_DS_GEMV_P_Q4_IDMAP_ZSYNC_R7, K_INK_DS_CAP_SAMPLE_IDMAP_ZSYNC_R7,
       K_INK_ARGMAX_BATCHED_R7, K_INK_TOPK_BATCHED_R7,
       K_DS_CAP_SAMPLE_ZSYNC_R7, K_DS_ATTN_LONG_R7,
       K_INK_MOE_PF_WORKLIST_R7, K_INK_GEMM_RQ_EXP_IQ3XXS_SITU_R7,
       K_INK_GEMM_IQ4NL_EXP_R7,
       K_INK_GEMM_Q8_MED_R7, K_INK_GEMM_Q8_SMALL_R7, K_INK_GEMM_Q8_64X64_R7, K_INK_ROUTER_ROUTE_PF_R7,
       K_INK_KV_SCONV_VCLIP_STORE_PF_R8, K_INK_SCONV_POST_PF_R8,
       K_DS_GEMV_P_CAP_SAMPLE_ZSYNC_R8,
       K_INK_MOE_SORT_ADAP_R8, K_INK_GEXP_IQ3XXS_SITU_NJ1_R8, K_INK_GEXP_IQ3XXS_SITU_NJM_R8,
       K_INK_GEXP_IQ4NL_NJ1_R8, K_INK_GEXP_IQ4NL_NJM_R8,
       K_INK_REL_ATTN_VER_TREE_R8, K_INK_SUBSPAN_GLUE_R8,
       K_INK_QKV_QCLIP_T8_R9, K_INK_REL_CLIP_SINK_GATE_T8_R9,
       K_INK_WO_SCONV_POST_T1_R9, K_INK_WO_SCONV_POST_T8_R9,
       K_INK_GEMM_IQ4NL_EXP_R9, K_INK_GEMM_Q8_MED_R9, K_INK_GEMM_Q8_64X64_R9,
       K_INK_GEMM_RQ_EXP_IQ3XXS_SITU_R9,
       K_N };
static const char *kname[K_N] = {"ie_embed_q4", "ie_embed_f32", "ie_rmsnorm", "ie_quant_q8", "ie_gemv_q4q8", "ie_gemv_q8q8",
                                 "ie_gemv_f32", "ie_bias",      "ie_add",     "ie_swiglu",   "ie_rope",
                                 "ie_kv_store", "ie_attn",      "ie_argmax",   "ie_rope_kv", "ie_attn_split",
                                 "ie_embed_q8", "ie_qkn_rope_kv", "ie_gdn", "ie_gemv_q8q8_t", "ie_gemv_q8q8_tr", "ie_gemm_q8", "ie_ring_store", "ie_attn_pf", "ie_gemm_h", "ie_gnorm", "ie_gdn1", "ie_rmsnorm_t", "ie_attn_pfg", "ie_gdn_prep", "ie_gdn_wy", "ie_gdn_seq", "ie_gemm_q8r", "ie_gemv_q4kq8", "ie_gemv_q4kq8_tr", "ie_gemm_q4kr", "ie_gemv_f32_t", "ie_gemv_q4kq8_tr8", "ie_gemv_q6kq8", "ie_gemv_q6kq8_tr", "ie_gemv_q6kq8_tr8", "ie_gemm_q6kr", "ie_gemv_f32_tw", "ie_gemv_q4kq8_ts2", "ie_gemv_q6kq8_ts2", "ie_attn_split_t", "ie_topk_part", "ie_topk_merge", "ie_attn_kv", "ie_attn_fa", "ie_attn_fa_q8", "ie_attn_fa2", "ie_gemm_f32_s", "ie_gemm_q4kr_sb", "ie_requant_sb", "ie_q8_f16", "ie_gemm_q4k_h", "ie_gemm_q4k_h256", "ie_copy_rows", "ie_df_conv", "ie_df_qkv", "ie_df_attn", "ie_gemv_q5kq8", "ie_gemv_iq4xsq8", "ie_gexp", "ie_moe_route", "ie_moe_sum", "ie_qkn_rope_kv2", "ie_attn_sw", "ie_attn_gq", "ie_gexp_t", "ie_moe_sort", "ie_attn_fa_sw", "ie_gemm_kx", "ie_gemv_raw_q2k", "ie_gemv_raw_iq2xxs", "ie_gemv_raw_iq2xs", "ie_gemv_raw_iq3xxs", "ie_gemv_raw_iq1s", "ie_gemv_raw_iq2s", "ie_gemv_raw_iq1m", "ie_gemv_raw_iq3s",
               "ie_gemv_raw_q2k_tr", "ie_gemv_raw_iq2xxs_tr", "ie_gemv_raw_iq2xs_tr", "ie_gemv_raw_iq3xxs_tr", "ie_gemv_raw_iq1s_tr", "ie_gemv_raw_iq2s_tr", "ie_gemv_raw_iq1m_tr", "ie_gemv_raw_iq3s_tr",
               "ie_gemv_iq4xsq8_tr", "ie_gdn_replay",
               "ie_gemv_iq4xsq8_tr8",
               "ie_gemv_raw_q2k_tr4", "ie_gemv_raw_iq2xxs_tr4", "ie_gemv_raw_iq2xs_tr4", "ie_gemv_raw_iq3xxs_tr4", "ie_gemv_raw_iq1s_tr4", "ie_gemv_raw_iq2s_tr4", "ie_gemv_raw_iq1m_tr4", "ie_gemv_raw_iq3s_tr4",
               "ie_gemv_raw_q2k_tr8", "ie_gemv_raw_iq2xxs_tr8", "ie_gemv_raw_iq2xs_tr8", "ie_gemv_raw_iq3xxs_tr8", "ie_gemv_raw_iq1s_tr8", "ie_gemv_raw_iq2s_tr8", "ie_gemv_raw_iq1m_tr8", "ie_gemv_raw_iq3s_tr8",
               "ie_gemm_rq_q2k", "ie_gemm_rq_iq2xxs", "ie_gemm_rq_iq2xs", "ie_gemm_rq_iq3xxs", "ie_gemm_rq_iq1s", "ie_gemm_rq_iq2s", "ie_gemm_rq_iq1m", "ie_gemm_rq_iq3s",
               "ie_gemm_ri_q2k", "ie_gemm_ri_iq2xxs", "ie_gemm_ri_iq2xs", "ie_gemm_ri_iq3xxs", "ie_gemm_ri_iq1s", "ie_gemm_ri_iq2s", "ie_gemm_ri_iq1m", "ie_gemm_ri_iq3s", "ie_attn_long", "ie_attn_pl", "ie_topk_max", "ie_topk_sel", "ie_gemv_raw_tr8_m", "ie_rmsnorm_f", "ie_gdn1p", "ie_gemv_q8q8_ts", "ie_attn_long_e", "ie_attn_split_tp", "ie_df_attn_g", "ie_gemv_q4kq8_ts16", "ie_gemv_q4kq8_ts4", "ie_gemv_q6kq8_ts16", "ie_gemv_q6kq8_ts4", "ie_copy_list", "ie_qkn_rope_kv_rot", "ie_attn_unrot", "ie_vres", "ie_gate", "ie_gemv_raw_tr4_m", "ie_gemv_raw_swiglu", "ie_gemv_raw_swiglu_tr4", "ie_gemv_raw_swiglu_tr8",
               "ie_gemv_iq4xsq8_tr16",
               "ie_gemm_rq32_q2k", "ie_gemm_rq32_iq2xxs", "ie_gemm_rq32_iq2xs", "ie_gemm_rq32_iq3xxs", "ie_gemm_rq32_iq1s", "ie_gemm_rq32_iq2s", "ie_gemm_rq32_iq1m", "ie_gemm_rq32_iq3s",
               "ie_gemm_rq_swiglu",
               "ie_gemm_rq_t16_q2k", "ie_gemm_rq_t16_iq2xxs", "ie_gemm_rq_t16_iq2xs", "ie_gemm_rq_t16_iq3xxs", "ie_gemm_rq_t16_iq1s", "ie_gemm_rq_t16_iq2s", "ie_gemm_rq_t16_iq1m", "ie_gemm_rq_t16_iq3s",
               "ie_gemm_rq_t16_m", "ie_gemm_rq_t16_swiglu",
               "ie_gdn_prep_ring", "ie_gdn_wy_wmma", "ie_gdn_seq_wmma_gnorm",
               "ie_rmsnorm_mt", "ie_mtp_prenorm_q8",
               "ie_gemv_raw_q2k_tr16", "ie_gemv_raw_iq2xxs_tr16", "ie_gemv_raw_iq2xs_tr16", "ie_gemv_raw_iq3xxs_tr16", "ie_gemv_raw_iq1s_tr16", "ie_gemv_raw_iq2s_tr16", "ie_gemv_raw_iq1m_tr16", "ie_gemv_raw_iq3s_tr16",
               "ie_gemv_raw_tr16_m", "ie_gemv_raw_swiglu_tr16", "ie_attn_long_mp", "ie_q_scale_f16", "ie_gemv_q4k_swiglu",
               "ie_gemv_iq4nlq8", "ie_gemv_iq4nlq8_tr", "ie_gexp_raw", "ie_gexp_raw_t", "ie_gemm_rq_exp_iq3xxs",
               "ie_add_t", "ie_kv_store_t", "ie_vclip", "ie_attn_res", "ie_loop_gate", "ie_sconv",
               "ie_sconv_ring_store", "ie_rel_attn", "ie_rel_attn_pf", "ie_situ", "ie_logit_cap",
               "ie_saluki_lora_b", "ie_saluki_iq2xxs_tr8_r4", "ie_saluki_attn_long_mp", "ie_saluki_iq1s_q8_tr8", "ie_saluki_gdn1p",
               "ie_saluki_iq2s_tr8", "ie_saluki_iq1s_tr8", "ie_saluki_iq1s_down_tr8", "ie_saluki_lora_b_t32", "ie_saluki_iq2xxs_tr9",
               "ie_saluki_lora_a_t4", "ie_saluki_lora_a_t1", "ie_saluki_lora_a_v4",
               "ie_saluki_q2k_tr8_r4", "ie_saluki_iq2xs_tr9", "ie_saluki_iq1m_tr8_r4",
               "ie_saluki_iq3s_tr8_r4", "ie_saluki_iq3s_tr8_r2", "ie_saluki_iq3xxs_tr9",
               "ie_saluki_attn_long_mp_tr", "ie_saluki_gdn1p_tr",
               "ie_saluki_gemm_rq128_q2k", "ie_saluki_gemm_rq128_iq2xxs", "ie_saluki_gemm_rq128_iq2xs", "ie_saluki_gemm_rq128_iq3xxs",
               "ie_saluki_gemm_rq128_iq1s", "ie_saluki_gemm_rq128_iq2s", "ie_saluki_gemm_rq128_iq1m", "ie_saluki_gemm_rq128_iq3s",
               "ie_saluki_gemm_rq64_q2k", "ie_saluki_gemm_rq64_iq2xxs", "ie_saluki_gemm_rq64_iq2xs", "ie_saluki_gemm_rq64_iq3xxs",
               "ie_saluki_gemm_rq64_iq1s", "ie_saluki_gemm_rq64_iq2s", "ie_saluki_gemm_rq64_iq1m", "ie_saluki_gemm_rq64_iq3s",
               "ie_saluki_attn_pl", "ie_saluki_swiglu_v4", "ie_saluki_gdn_seq_wmma_gnorm",
               "ie_saluki_gemm_rq128_iq2xxs_sb", "ie_saluki_gemm_rq128_q2k_pipe",
               "ie_saluki_iq2xxs_tr9_pipe1",
               "ie_saluki_attn_long_mp_pipe1", "ie_saluki_attn_long_mp_tr_nospill",
               "ie_saluki_gemm_rq128_iq2s_pipe", "ie_saluki_gemm_rq128_iq3s_pipe", "ie_saluki_gemm_rq128_iq2xs_pipe",
               "ie_saluki_gemm_rq128_iq1m_pipe", "ie_saluki_gemm_rq128_iq3xxs_pipe",
               "ie_saluki_iq2s_tr8_pipe1", "ie_saluki_iq2xs_tr9_pipe1", "ie_saluki_iq3xxs_tr9_pipe1",
               "ie_saluki_iq1s_tr8_pipe1", "ie_saluki_q2k_tr8_r4_pipe1", "ie_saluki_iq1m_tr8_r4_pipe1",
               "ie_saluki_iq3s_tr8_r2_pipe1",
               "ie_saluki_gemm_rq128_iq2xxs_r10", "ie_saluki_gemm_rq128_q2k_r10", "ie_saluki_gemm_rq128_iq1s_r10",
               "ie_saluki_attn_pl_r10", "ie_saluki_attn_long_mp_r10", "ie_saluki_attn_long_mp_tr_r10",
               "ie_saluki_iq2xxs_r4_lds", "ie_saluki_iq2xxs_tr9_r10", "ie_saluki_iq1s_tr9_r10",
               "ie_saluki_q2k_r4_lds", "ie_saluki_iq2s_r4_lds", "ie_saluki_iq2xs_r4_lds",
               "ie_saluki_iq1m_tr9_r10", "ie_saluki_iq3s_r4_lds",
               "ie_saluki_gemm_rq128_iq2xxs_r11", "ie_saluki_gemm_rq128_iq2s_r11", "ie_saluki_gemm_q8r_r11",
               "ie_saluki_qkn_rope_kv_r11", "ie_saluki_attn_long_mp_r11", "ie_saluki_gdn_wy_wmma_r11",
               "ie_saluki_gemv_raw_tr8_m_r11", "ie_saluki_q2k_tr9_r11", "ie_saluki_lora_b_r11",
               "ie_saluki_topk_r11",
               "ie_inkling_rel_attn_pf_r1", "ie_inkling_rel_attn_dec_r1",
               "ie_inkling_gexp_raw_iq3xxs_r1", "ie_inkling_gemm_rq_exp_iq3xxs_r1",
               "ie_inkling_gexp_iq4nl_r1", "ie_inkling_gemm_iq4nl_exp_r1",
               "ie_inkling_gemm_q8_small_r1", "ie_inkling_gemm_q8_nb2_r1",
               "ie_inkling_gemv_q8_nb2_r1", "ie_inkling_gemv_q8_small_r1",
               "ie_inkling_attn_res_r1", "ie_inkling_rmsnorm_mt_r1",
               "ie_inkling_moe_sort_r1", "ie_inkling_situ_r1", "ie_inkling_moe_sum_r1",
               "ie_inkling_loop_gate_r1", "ie_inkling_sconv_r1", "ie_inkling_vclip_r1",
               "ie_inkling_gemm_rq_exp_iq3xxs_r2", "ie_inkling_gexp_iq3xxs_situ_r2",
               "ie_inkling_gemv_q8_med_r2", "ie_inkling_gemv_q8_multi2_r2", "ie_inkling_gemm_q8_med_r2",
               "ie_inkling_lc_prenorm_q8_r2", "ie_inkling_lc_wfg_gate_r2",
               "ie_inkling_kv_sconv_vclip_store_r2", "ie_inkling_kv_ring_store_r2", "ie_inkling_kv_sconv_vclip_store_ctx_r2",
               "ie_inkling_sconv_add_r2", "ie_inkling_moe_sum_sconv_add_r2", "ie_inkling_gate_q8_r2",
               "ie_inkling_gemm_f32_router_r2", "ie_inkling_router_route_dec_r2",
               "ie_inkling_gemm_iq4nl_exp_r2",
               "ie_inkling_embed_q8_ctx_r2", "ie_inkling_embed_q8_batch_r2",
               "ie_inkling_vclip_ctx_r2", "ie_inkling_sconv_ctx_r2", "ie_inkling_sconv_add_ctx_r2",
               "ie_inkling_moe_sum_sconv_add_ctx_r2", "ie_inkling_kv_store_ctx_r2",
               "ie_inkling_rel_attn_dec_ctx_r2",
               "ie_inkling_gemm_rq_exp_iq3xxs_r3", "ie_inkling_gexp_iq3xxs_situ_r3",
               "ie_inkling_gexp_raw_iq3xxs_r3", "ie_inkling_gexp_iq4nl_r3",
               "ie_inkling_gemm_f32_router_r3", "ie_inkling_rel_attn_pf_r3",
               "ie_inkling_rel_attn_ver_r4", "ie_inkling_sconv_hist_r4", "ie_inkling_sconv_commit_r4",
               "ie_inkling_gexp_raw_iq3xxs_ver_r4", "ie_inkling_gexp_iq4nl_ver_r4", "ie_inkling_gemv_f32_router_ver_r4",
               "ie_dspark_dflare_mix_r4", "ie_dspark_block_init_r4", "ie_dspark_attn_r4",
               "ie_dspark_concat_q8_r4", "ie_dspark_rnn_step_r4", "ie_dspark_add_q8_r4",
               "ie_dspark_block_init_zsync_r5", "ie_dspark_concat_q8_zsync_r5", "ie_dspark_rnn_step_zsync_r5",
               "ie_dspark_add_q8_zsync_r5", "ie_dspark_gemv_p_q4_zsync_r5", "ie_dspark_cap_argmax_zsync_r5",
               "ie_dspark_gemv_q8_zsync_r5", "ie_dspark_gemv_q4k_zsync_r5", "ie_dspark_gemv_q6k_zsync_r5",
               "ie_inkling_gemv_q8_small_ver_r5",
               "ie_inkling_attn_res_4out_r6", "ie_inkling_lc_prenorm_q8_4x_r6",
               "ie_inkling_lc_down_4x_r6", "ie_inkling_lc_wfg_gate_norm_r6",
               "ie_inkling_moe_sort_compact_r6", "ie_inkling_gexp_raw_iq3xxs_compact_r6",
               "ie_inkling_gexp_iq4nl_compact_r6", "ie_inkling_gexp_raw_iq3xxs_slot_r6",
               "ie_inkling_gexp_iq4nl_slot_r6", "ie_inkling_gemm_rq_exp_iq3xxs_situ_r6",
               "ie_inkling_rel_attn_ver_qtile_r6",
               "ie_inkling_router_route_exact_t1_r7", "ie_inkling_router_route_sort_ver_r7",
               "ie_inkling_kv_sconv_vclip_store_r7", "ie_inkling_sconv_post_r7",
               "ie_inkling_moe_sum_sconv_add_r7", "ie_inkling_exp_down_tail_t1_r7", "ie_inkling_exp_down_tail_ver_r7",
               "ie_inkling_qkv_qclip_r7", "ie_inkling_rel_clip_sink_gate_r7",
               "ie_inkling_gexp_iq3xxs_situ_compact_r7", "ie_inkling_attn_res_norm_q8_r7", "ie_inkling_wo_sconv_post_r7",
               "ie_inkling_gemv_q4k_head_ver_r7",
               "ie_inkling_rel_attn_ver_swa_qtile_r7", "ie_inkling_rel_attn_ver_glob_wmma_r7",
               "ie_inkling_rel_attn_pf_r7", "ie_inkling_rel_attn_pf_split_r7",
               "ie_inkling_gexp_iq3xxs_flute_situ_t1_r7", "ie_inkling_gexp_iq3xxs_flute_situ_compact_r7",
               "ie_inkling_gexp_iq4nl_bw_t1_r7", "ie_inkling_gexp_iq4nl_bw_compact_r7", "ie_inkling_gemv_q8_bw_r7",
               "ie_inkling_q4k_head_cap_topk_r7",
               "ie_inkling_ds_base_q4k_idmap_r7", "ie_inkling_ds_gemv_p_q4_idmap_zsync_r7", "ie_inkling_ds_cap_sample_idmap_zsync_r7",
               "ie_inkling_argmax_batched_r7", "ie_inkling_topk_batched_r7",
               "ie_inkling_ds_cap_sample_zsync_r7", "ie_inkling_ds_attn_long_r7",
               "ie_inkling_moe_pf_worklist_r7", "ie_inkling_gemm_rq_exp_iq3xxs_situ_r7",
               "ie_inkling_gemm_iq4nl_exp_r7",
               "ie_inkling_gemm_q8_med_r7", "ie_inkling_gemm_q8_small_r7", "ie_inkling_gemm_q8_64x64_r7",
               "ie_inkling_router_route_pf_r7",
               "ie_inkling_kv_sconv_vclip_store_pf_r8", "ie_inkling_sconv_post_pf_r8",
               "ie_inkling_ds_gemv_p_cap_sample_zsync_r8",
               "ie_inkling_moe_sort_adap_r8", "ie_inkling_gexp_iq3xxs_situ_nj1_r8", "ie_inkling_gexp_iq3xxs_situ_njm_r8",
               "ie_inkling_gexp_iq4nl_nj1_r8", "ie_inkling_gexp_iq4nl_njm_r8",
               "ie_inkling_rel_attn_ver_tree_r8", "ie_inkling_subspan_glue_r8",
               "ie_inkling_qkv_qclip_t8_r9", "ie_inkling_rel_clip_sink_gate_t8_r9",
               "ie_inkling_wo_sconv_post_t1_r9", "ie_inkling_wo_sconv_post_t8_r9",
               "ie_inkling_gemm_iq4nl_exp_r9", "ie_inkling_gemm_q8_med_r9", "ie_inkling_gemm_q8_64x64_r9",
               "ie_inkling_gemm_rq_exp_iq3xxs_situ_r9"};
static const uint8_t ink_bit[K_N - K_INK_REL_ATTN_PF] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 33, 34, 34, 34, 34, 34, 35,
    36, 37, 38, 39, 40, 41,
    42, 43, 43, 44, 45, 46,
    47, 47, 47, 47, 47, 47,
    48, 48, 48, 48, 48, 48, 48, 48, 48, 49,
    50, 51, 52, 53, 54, 54, 54, 55, 55, 57, 58,
    59, 60, 61, 62, 63, 63, 63, 64, 65, 66, 67, 68, 76, 77, 78, 79, 80,
    81, 81, 82, 82, 83, 84, 88, 88, 88, 92, 93, 94, 95,
    69, 70, 71, 72, 73, 73, 74, 75, 75, 85, 86, 86, 86, 86, 86, 89, 90,
    96, 97, 98, 98, 99, 100, 101, 102};
_Static_assert(sizeof(ink_bit) == (size_t)(K_N - K_INK_REL_ATTN_PF), "ink_bit table length mismatch");
/* ie_gdn: conv input ring slots and state slots per linear layer */
/* ie_gdn: conv input ring slots (as in ie_kernels.c); the state slots per
 * linear layer are gpu_backend.ns: at least the tokens of a verify run */
enum { GDN_RING = 32, GDN_RECF = 260 }; /* GDN_RECF: floats of one head's record of a token (ie_kernels.c) */
/* The argmax kernel: workgroups, and its device scratch (partial results and
 * the counter of finished workgroups). */
enum { ARGMAX_GROUPS = 128 };

/* Device copies of the weights of one op. */
typedef struct {
  void *w0, *w1; /* Q4/Q8: words, scales; f32 matrix: floats; vec: floats */
  void *bias;    /* GEMV: the fused bias (o->v), or NULL */
  void *nw;      /* GEMV: the weight of the fused RMSNorm (o->nv), or NULL */
} dop;

/* Device arrays packed into a few large allocations (slab_take, next to sdev_put). */
typedef struct {
  void **a;     /* the allocations */
  char **cur;   /* per allocation: its free part (NULL: an array of its own) */
  size_t *left;
  uint32_t n, cap;
} slab_list;
static void *slab_take(slab_list *s, size_t bytes);

typedef struct {
  backend base;
  const model *m;
  const graph *g;
  hipModule_t mod;
  hipModule_t saluki_mod;
  hipModule_t inkling_mod;
  hipFunction_t k[K_N];
  char *arena; /* device */
  u32 *am_part, *am_count; /* argmax scratch (device) */
  int skip_argmax;         /* this run's argmax tokens are not wanted (the top K is taken instead): the model's argmax is not launched */
  u32 *at_count;           /* ie_attn_split: finished workgroups per head (device); ie_attn_split_t: per token too */
  float *at_part;          /* ie_attn_split_t: the partial results, at_pstride floats per token (ie_attn_long: at_lstride) */
  float *tk_buf;           /* top K: candidates (TK_ROWS x TK_NG x TK_MAX values, then indices), results */
  float *tk_pin;           /* top K: pinned host readback buffer (TK_ROWS * TK_MAX * 2 floats = 8 KiB) */
  size_t at_part_floats;   /* allocated capacity of at_part in floats */
  u32 at_pstride, at_tmax;
  u32 at_lstride, at_ltmax; /* ie_attn_long: floats of partial results per token, tokens per launch (at_part is shared) */
  int at_lpack;             /* IE_ATTN_LONG_PACK (default 1; 0: off): a launch's tokens at the stride of its own positions */
  int kvrot;                /* IE_KV_ROT=1: q, k and v rotated (ie_qkn_rope_kv_rot), the attention output back (ie_attn_unrot) */
  /* IE_KV_GROW (default; 0: off): the K and V cache of every attention layer is its own array of addresses with
   * physical memory mapped only for the positions in use (hipMemAddressReserve / hipMemCreate / hipMemMap; kv_grow 1),
   * instead of one allocation of n_ctx positions for all layers. At 131072 positions the cache is 4.6 GiB: a context that
   * is not full does not pay for it, and WDDM, when the VRAM budget is over, puts a whole allocation in shared memory (one
   * half of the cache read over PCIe at the speed of PCIe), which separate and smaller allocations avoid. Growing by
   * allocating a bigger array and copying does not work: the freed arrays stay allocated (about 2.2 GiB extra at 105K
   * positions, tools/kv_grow_test.py). kv_grow 2: separate arrays of the full size (no virtual memory in the runtime).
   * Not with the megakernel (it addresses one array) or ring layers (flint). */
  int kv_grow;
  void **kcl, **vcl;       /* the K and V array of each attention layer */
  u32 kv_cap, kv_step;     /* positions that may be used now (mapped, plus kv_slack), growth in steps of kv_step */
  size_t kv_chunk, kv_va, kv_mapped; /* mapping unit, bytes of addresses per array, bytes mapped per array */
  void **kv_h;             /* the physical chunks: [array 2 * layer + v][chunk] */
  size_t kv_nchunk;
  u32 *gm_count;           /* GEMV with a fused RMSNorm: finished workgroups (device) */
  u32 *ink_moe_rcount;     /* R7 Fusion B2: per-32-row-block completion counters ((dim+31)/32 u32s) */
  u32 *ink_qhead_cnt;      /* R7 Fusion D1: per-Q-head completion counters (n_head u32s) */
  u32 *ink_sconv_rcnt;     /* R8 Prefill sconv ring-commit completion counters (16 u32s) */
  u32 slot;                /* ie_gdn: the state slot that holds the current state */
  float *dfk, *dfv;        /* DFlash: the drafter KV cache [layer][df_ring][n_kv * hd] (f32), row = position % df_ring */
  u32 df_ring;
  float *sc_hist;          /* inkling2 verify sconv history: [vmax][sc_hist_stride] floats */
  u32 sc_hist_stride, sc_ver_pos, sc_ver_T;
  int sc_defer;
  struct {
    int ready, qkind;
    uint16_t *kc[8], *vc[8];
    u32 kv_cap[8];
    /* IE_DS_KV_GROW (default; 0: off): the K and V arrays of the global (unwindowed) layers are reserved addresses for
     * kv_cap positions with physical memory mapped only for the positions in use, like the main KV cache (kv_ensure).
     * The kernels still see one contiguous array and the same ring index (p % kv_cap). Windowed layers stay hipMalloc. */
    int kv_grow;
    uint8_t kv_vm[8];             /* layer il uses reserved addresses */
    u32 kv_rows, kv_step;    /* positions backed by memory now (all kv_vm layers), growth step */
    size_t kv_chunk, kv_va, kv_mapped, kv_nchunk, kv_rowb; /* mapping unit, bytes reserved / mapped per array, bytes per row */
    void **kv_h;             /* physical chunks: [2 * layer + v][chunk] */
    float *rcos, *rsin;
    float *fuse_w, *mask_embed, *out_norm, *tok_norm, *mk_dflare_norm, *mk_rnn_bias, *mk_hidden_bias;
    void *mk_rnn_w0, *mk_rnn_w1, *mk_ho_w0, *mk_ho_w1, *mk_w1_w0, *mk_w1_w1, *head_w0, *head_w1;
    u32 head_rows;
    int head_kind;           /* MAT_Q8, or MAT_Q4K / MAT_Q6K when output.weight is one (IE_TENSOR_OVERLAY, a requantized file) */
    void *p_w0, *p_w1;       /* IE_DS_SPLIT: P = head x mk_hidden_out, [head_rows, markov_rank] Q4_0/Q8_0 (NULL: unsplit) */
    int p_kind;              /* MAT_Q4 (29 MB, default) or MAT_Q8 (55 MB, IE_DS_SPLIT=q8) */
    float *base, *zero;      /* IE_DS_SPLIT: head x q8(hidden_i + bias) of a step's drafts [block][head_rows]; dim zeros */
    uint8_t *hq;             /* IE_DS_SPLIT: q8(hidden_i + bias) [block][xq_str] */
    void *zbuf;              /* R5/R7 zero-sync control + draft_ids/probs + qnode + partials (32 KiB) */
    void *zpin;              /* R7 pinned host readback buffer for probabilistic zsync (8 KiB) */
    /* R30: 0-VRAM 60K+4K FR-Spec / MicroSpec certified id_map shortlist (+0.256 MiB VRAM) */
    u32 *id_map_dev;         /* device u32[65536] */
    u32 *id_core_host;       /* host u32[n_core], ascending */
    u32 *id_dyn_pin;         /* pinned host u32[4096] for async tail H2D */
    uint8_t *core_bm, *dyn_bm;
    u32 n_core, n_dyn, n_act;
    int sl_bypass, id_dyn_pinned;
    float sl_miss_ema;
    /* R33: Multi-stream ds_ctx overlap behind target output head */
    hipStream_t ctx_stream;
    hipEvent_t ev_feat_ready, ev_ctx_done;
    int ctx_pending, ov_mode;
    u32 ctx_pos, ctx_T;
    float *dflare_norm[8], *attn_norm[8], *ffn_norm[8], *q_norm[8], *k_norm[8], *k_ctx_norm[8];
    void *wq0[8], *wq1[8], *wk0[8], *wk1[8], *wv0[8], *wv1[8], *wkc0[8], *wkc1[8], *wvc0[8], *wvc1[8];
    void *wo0[8], *wo1[8], *wgu0[8], *wgu1[8], *wd0[8], *wd1[8];
    uint8_t *ctx_q8, *xq, *aq, *gq, *cat_q8, *lat_q8, *h_q8;
    float *ctx_kv, *frnn, *h, *xn, *q, *k, *v, *att, *gu, *hidden, *state, *raw768, *delta, *logits;
    u32 *argmax;
    u32 ctx_q8_str, ctx_kv_str, h_str, xq_str, q_str, kv_str, aq_str, gu_str, gq_str;
  } ds;
  u32 ns;                  /* ie_gdn: state slots per linear layer (4, or 8 for more than 3 drafts; 1 with the records) */
  u32 vmax;                /* the most tokens of a verify run (8 or 16; 1 when there is no verify run) */
  float *rec;              /* ie_gdn records: [layer][vmax - 1 tokens][nv heads][GDN_RECF floats] of a verify run (NULL: a state slot per token) */
  float *rec_buf[2];       /* double-buffered record arrays so lazy replay reads the last run's records while writing the next */
  int rec_idx;
  float *lazy_rec;         /* pending GDN replay records from gpu_accept / gpu_accept_tree (NULL when none) */
  u32 lazy_mask;           /* pending node bitmask for lazy GDN replay (0 when none) */
  uint64_t lazy_rls;       /* per-layer float stride of lazy_rec */
  size_t rec_bytes;        /* its size in bytes */
  u32 rec_n;               /* the records of the last verify run (tokens 1 .. rec_n) not yet applied by gpu_accept */
  int attn_split;          /* use ie_attn_split (hd <= IE_ATT_MAX_HD, and not IE_ATTN=old) */
  int attn_gq;             /* flint: ie_attn_gq for decode (IE_ATTN_SW=1: ie_attn_sw) */
  uint32_t gq_ch;          /* its chunk: 256 positions (IE_ATTN_GQ_CH: 1 .. 256, for tests) */
  uint32_t gq_min;         /* from this position count on (IE_ATTN_GQ_MIN) */
  uint32_t gq_s;           /* at most this many splits per kv head (IE_ATTN_GQ_S, default 32) */
  int gq_ch_fixed;         /* IE_ATTN_GQ_CH set: that chunk always */
  int flint;               /* the model is flint (its megakernel builds) */
  u32 *ms_offs, *ms_list, *ms_packs, *ms_npacks; /* flint/inkling MoE sort + R6 compact packs */
  u32 *ms_packs1, *ms_npacks1, *ms_packsM, *ms_npacksM; /* R8 adaptive nj==1 vs nj==2..4 packs */
  int ink_moe_all_slot;                          /* 1 if every OP_GEMV_EXP supports R6 slot kernels */
  int attn_kv;             /* use ie_attn_kv for decode and verify (the K/V of a split read once; not IE_ATTN=split) */
  int has_long;            /* the code object has ie_attn_long */
  int has_split_tp;        /* the code object has ie_attn_split_tp (ie_attn_split_t with early loads) */
  int has_long_e;          /* and ie_attn_long_e (the same with the V loads first) */
  int has_long_mp;         /* and ie_attn_long_mp (single-KV-pass multi-pair-group Q8 long attention) */
  int kv_only;             /* sec == 4: run only the MTP KV-write prefix (skipping wq + q-heads + post-KV ops) */
  int has_topk_fast;       /* the code object has ie_topk_max and ie_topk_sel */
  int has_pl;              /* the code object has ie_attn_pl (prompt-chunk attention at long positions) */
  int has_gdn1p;           /* the code object has ie_gdn1p (linear attention of a verify run, state-free work ahead of the loop) */
  int has_q8ts;            /* the code object has ie_gemv_q8q8_ts (few-row Q8 GEMV over more workgroups) */
  int has_dfg;             /* the code object has ie_df_attn_g (DFlash attention, the reads shared per KV head) */
  int has_ts16;            /* the code object has ie_gemv_q4kq8_ts16 / _ts4 and ie_gemv_q6kq8_ts16 / _ts4 */
  int has_cl;              /* the code object has ie_copy_list (and the tree kernels: a tree verify can run) */
  /* a tree verify (tree_table): the node table and the nodes' RoPE rows (device), the copy list and its scratch */
  unsigned char *tr_dev;
  float *tr_rope;
  unsigned long long *cl_dev;
  unsigned char *cl_scr;
  size_t cl_scr_bytes;
  unsigned char *tr_pin;         /* pinned host ring for async tr_tab H2D uploads */
  unsigned long long *cl_pin;    /* pinned host ring for async copy_list H2D uploads */
  unsigned long long *cl_host;   /* persistent host entry scratch for gpu_accept_tree */
  u32 tr_ring, cl_ring;
  int tr_on;               /* the ops of this run are a tree's */
  u32 tr_pos;              /* its position */
  unsigned char tr_tab[384];
  uint64_t ink_tree_mask;  /* R32: bit (t*8+j)==1 iff node j is an ancestor of (or equal to) node t */
  u32 ink_tree_dep_pack;   /* R32: 4-bit logical tree depth per node t in [0..7] */
  u32 ink_tree_split;      /* R32: first node t>=2 with par[t]==0 (T if linear chain) */
  int has_tr8m;            /* the code object has ie_gemv_raw_tr8_m (several raw GEMVs of one input in one launch) */
  int has_tr4m;            /* the code object has ie_gemv_raw_tr4_m (2 to 4 tokens, several raw GEMVs in one launch) */
  int has_raw_swiglu;      /* the code object has ie_gemv_raw_swiglu[|_tr4|_tr8] (fused gate + up + SwiGLU + Q8) */
  int has_iq4tr8;          /* the code object has ie_gemv_iq4xsq8_tr8 (IQ4_XS, 5 to 8 tokens, 2 rows per wave) */
  int has_iq4tr16;         /* the code object has ie_gemv_iq4xsq8_tr16 (IQ4_XS, up to 16 tokens, 2 rows per wave) */
  int has_rq32;            /* the code object has ie_gemm_rq32_* (128x32 short-prefill raw WMMA) */
  int has_rq_swiglu;       /* the code object has ie_gemm_rq_swiglu (fused dual-matrix prefill SwiGLU) */
  int has_rqt16;           /* the code object has ie_gemm_rq_t16_* (single-pass matrix-core verify, T <= 16) */
  int has_rqt16_m;         /* the code object has ie_gemm_rq_t16_m (multi-projection single-pass verify) */
  int has_rqt16_swiglu;    /* the code object has ie_gemm_rq_t16_swiglu (fused gate + up + SwiGLU, T <= 16) */
  int has_gdn_wmma;        /* the code object has ie_gdn_prep_ring / ie_gdn_wy_wmma / ie_gdn_seq_wmma_gnorm */
  int has_norm_mt;         /* the code object has ie_rmsnorm_mt (multi-WG RMSNorm + Q8) */
  int has_mtp_prenorm;     /* the code object has ie_mtp_prenorm_q8 (fused MTP embed + enorm + hnorm + Q8) */
  int has_tr16;            /* the code object has ie_gemv_raw_*_tr16 (coalesced 10..16 token raw GEMV) */
  int has_tr16m;           /* the code object has ie_gemv_raw_tr16_m (multi-matrix 10..16 token raw GEMV) */
  int has_swiglu_tr16;     /* the code object has ie_gemv_raw_swiglu_tr16 (fused gate+up+SwiGLU+Q8, 10..16 tokens) */
  int has_q_scale_f16;     /* the code object has ie_q_scale_f16 (scaled fp16 Q prepass for 4-WG/WGP ie_attn_pl) */
  int has_q4k_swiglu;      /* the code object has ie_gemv_q4k_swiglu (fused MAT_Q4K gate+up+SwiGLU+Q8, T <= 16) */
  int attn_long;           /* use ie_attn_long for the tokens at positions >= long_min (hd 256; IE_ATTN_LONG=0: never) */
  u32 long_min;            /* IE_ATTN_LONG_MIN: the first position that ie_attn_long computes */
  u32 *at_gcount;          /* ie_attn_long: finished splits per group, [token][head][group] (device, zero) */
  u32 at_gmax;             /* its groups per head: ceil(ceil(n_ctx / 128) / 8) */
  /* IE_ATTN=check: after each ie_attn_split, also run ie_attn into chk and
   * compare the two outputs on the host (max |a - b| / max |a|). */
  int attn_check;
  float *chk, *chk_a, *chk_b;
  double chk_worst;
  uint32_t chk_n;
  /* host time spent issuing the launches of a step (not waiting), and how many launches that was */
  double host_ms;
  uint32_t host_n;
  uint64_t host_launches;
  void *xh;         /* ie_gemm_q4k_h: the fp16 activations of a chunk (T x the largest Q4_K K) */
  size_t xh_bytes;
  void *rq;         /* ie_requant_sb output (prompt chunks): T x the largest Q4_K input row */
  size_t rq_bytes;
  void *kc, *vc;    /* KV cache: kvb bytes per element (2: fp16, the kernels have ie_kv_f16; 4: f32; 1: q8_0, see kvq8) */
  u32 kvb;
  u32 kvq8;         /* the q8_0 KV cache (IE_KV_Q8=1; the kernels have ie_kv_q8, see KVROWP in ie_kernels.c) */
  u32 kvq4;         /* the 4-bit KV cache (IE_KV_Q4=1; the kernels have ie_kv_q4) */
  size_t kvrowb;    /* bytes of one row (position) of the KV cache of a layer: n_kv heads of hd values */
  float *rcos, *rsin;
  float *ring, *st; /* qwen35 linear attention state (see cpu.c) */
  float *sc_ring;   /* inkling2 sconv 4-tap ring buffer (m->sc_tot * 4 floats) */
  size_t sc_ring_bytes;
  u32 *ink_ctx_dev;         /* inkling2 HIP Graph T=1 decode device ctx: [0]=tok, [1]=pos */
  u32 *ink_ctx_host;        /* pinned host mirror of ink_ctx_dev */
  u32 *ink_toks_dev;        /* inkling2 prefill batched embed token ids on device */
  u32 *ink_toks_pin;        /* 16 * 512 * 4 = 32 KiB pinned ring for K_INK_EMBED_Q8_BATCH_R2 H2D */
  hipEvent_t ink_toks_ev[16]; /* per-slot H2D completion fence preventing ring overwrite on >16 queued chunks */
  u32 ink_toks_ring;
  u32 ink_toks_cap;
  u32 *am_pin;              /* 256 B pinned host buffer for T=1 argmax, verify argmax, and DSpark zbuf D2H */
  u32 *ink_lc_count;        /* inkling2 R6 wfg_gate_norm per-(ctrl, t_tile) atomic counters */
  hipStream_t ink_gstream;  /* dedicated non-blocking stream for inkling2 HIP Graph capture/replay */
  void *ink_pgraph[8];      /* R35 piecewise hipGraphExec_t[0..ink_n_pieces-1] for inkling2 T=1 decode */
  hipEvent_t ink_pev[8];    /* R35 WDDM HostQueue retirement markers between pieces */
  u32 ink_n_pieces;
  uint64_t ink_graph_key;   /* (swa_nsplit << 48) | (glob_nsplit << 32) | (mhash & 0xffffffffULL) */
  uint64_t ink_mask;        /* active IE_INKLING_KERNEL_MASK (bits 0..63) */
  uint64_t ink_mask2;       /* active IE_INKLING_KERNEL_MASK2 / upper hex128 (bits 64..127) */
  int ink_graph_cap;        /* 1 while capturing T=1 decode graph -> emit *_ctx kernels */
  int in_step;              /* 1 while inside gpu_step() T=1 decode run_ops */
  u32 ink_swa_nsplit, ink_glob_nsplit;
  /* R34: Low-VGPR (<=64) WDDM-guarded sub-span persistent glue dispatcher (bit 90) */
  void *ink_sp_dev;         /* 64 * 16 * 80 B = 80 KiB device task ring */
  void *ink_sp_pin;         /* 64 * 16 * 80 B = 80 KiB pinned host task ring */
  hipEvent_t ink_sp_ev[64];
  u32 *ink_sp_sync;         /* 16 B device sync control block (zero-init once; sync[2]=poison_flag) */
  u32 ink_sp_ring;
  int ink_sp_poisoned, ink_sp_used;
  float *snap_ring[IE_SNAP_MAX], *snap_st[IE_SNAP_MAX]; /* the prompt cache snapshots (gpu_snap), allocated at the first save: in host memory
                                     * (pageable; 2 x 160 MB less VRAM at 128K), IE_SNAP_HOST=0: in VRAM */
  char *snap_h[IE_SNAP_MAX];
  char *snap_kv[IE_SNAP_MAX];  /* sliding-window KV rings (inkling2/flint), host RAM under snap_host(), else VRAM */
  char *snap_sc[IE_SNAP_MAX];  /* inkling2 sconv ring snapshots */
  char *snap_dskv[IE_SNAP_MAX]; /* dspark sliding-window KV rings + frnn, host RAM under snap_host(), else VRAM */
  int live_snap, live_hbuf;    /* snapshot index matching the live GPU state / hbuf row 0 (-1 when dirty) */
  uint32_t live_snap_pos;      /* position at which live_snap was saved/restored (guards SWA ring wrap) */
  int q16_ready_op;            /* op index of next OP_ATTN whose scaled fp16 Q16 was pre-emitted into b->xh (-1: none) */
  char *snap_pin;              /* 2 x SNAP_PIN_CH pinned staging buffer for pipelined host<->device snap_copy */
  hipEvent_t snap_ev[2];
  hipStream_t snap_cs;
  /* the chunked linear attention of prompt chunks (ie_gdn_prep/wy/seq):
   * scratch for T tokens, or NULL (graph T <= 16 or no linear layers) */
  float *cq_qk, *cq_v, *cq_bg, *cq_u, *cq_w, *cq_m, *cq_g;
  void *tok0, *tok1; /* token_embd */
  dop *d;
  /* host -> device map of uploaded arrays */
  const void **hk;
  void **hv;
  uint32_t nh, caph;
  slab_list us; /* the blocks those arrays are in (slab_take) */
  hipEvent_t e0, e1;
  /* IE_PROFILE=1: one event after each op; the time between two events is
   * the GPU time of that op (with the gap before it). */
  int prof;
  hipEvent_t *pev;
  double *pms;
  uint32_t psteps;
  /* IE_HOST_STREAM=1 (opt-in): the host layers' arrays stay in pinned memory and are copied by
   * DMA into hs_ns VRAM slots during each pass (hs_before / hs_after); streamed layer s always runs from slot
   * s % hs_ns, so the ops' weight pointers are fixed. */
  int hs_on;
  u32 hs_n, hs_ns;
  int32_t *hs_op;            /* op -> streamed layer whose weights it reads, or -1 */
  u32 *hs_last;              /* the last op of each streamed layer */
  struct hs_arr { const void *src; size_t bytes, off; } *hs_a; /* the arrays of all streamed layers, by layer */
  u32 *hs_a0;                /* streamed layer s: arrays hs_a[hs_a0[s] .. hs_a0[s + 1]) */
  char *hs_slot[16];
  hipStream_t hs_cs;         /* the copy stream (non-blocking) */
  hipEvent_t *hs_done;       /* streamed layer s is in its slot */
  hipEvent_t hs_free[16];    /* the slot's last user has run */
  int hs_pending;            /* the streamed layer being run (its slot is freed after its last op) */
} gpu_backend;

/* The copy of streamed layer s into its slot, on the copy stream, after the slot's previous user ran. */
static void hs_copy(gpu_backend *b, u32 s, int wait) {
  const u32 sl = s % b->hs_ns;
  if (wait) HIP(H.StreamWaitEvent(b->hs_cs, b->hs_free[sl], 0));
  for (u32 k = b->hs_a0[s]; k < b->hs_a0[s + 1]; k++)
    HIP(H.MemcpyAsync(b->hs_slot[sl] + b->hs_a[k].off, b->hs_a[k].src, b->hs_a[k].bytes, hipMemcpyHostToDevice, b->hs_cs));
  HIP(H.EventRecord(b->hs_done[s], b->hs_cs));
}
/* Streamed layer s has run: free its slot and fill it with the next layer that uses it (cyclically, so the copies
 * of the next pass's first layers overlap the end of this one). */
static void hs_after(gpu_backend *b, u32 s) {
  HIP(H.EventRecord(b->hs_free[s % b->hs_ns], cur_stream));
  hs_copy(b, (s + b->hs_ns) % b->hs_n, 1);
}
/* Before op i: finish the previous streamed layer if op i is past it; wait for op i's layer to be in its slot. */
static void hs_before(gpu_backend *b, u32 i) {
  if (b->hs_pending >= 0 && i > b->hs_last[b->hs_pending]) {
    hs_after(b, (u32)b->hs_pending);
    b->hs_pending = -1;
  }
  const int s = b->hs_op[i];
  if (s >= 0 && s != b->hs_pending) {
    HIP(H.StreamWaitEvent(cur_stream, b->hs_done[s], 0));
    b->hs_pending = s;
  }
}

/* flint: ie_attn_gq (the q heads of a kv head together) for a token at
 * position pos when pos + 1 >= gq_min, else ie_attn_sw; the whole step the
 * same (the megakernel build). Default 1: ie_attn_gq at every position
 * (UD-Q4_K_XL, ctx 4096: 5.79 against 5.85 ms per token). */
#ifndef IE_GQ_MIN
#define IE_GQ_MIN 1u
#endif
/* flint chunks: ie_gemm_kx (WMMA) for Q4_K / Q5_K / IQ4_XS when T >= IE_GEMM_MIN
 * (default 17) and nb % 4 == 0; else ie_gexp_t (the arithmetic of decode).
 * IE_GEMM_KX=0: always ie_gexp_t. */
static int gx_wmma(const gpu_backend *b, u32 T, u32 nb, u32 kind) {
  (void)b;
  static int on = -1, gmin = -1;
  if (on < 0) on = getenv("IE_GEMM_KX") ? atoi(getenv("IE_GEMM_KX")) : 1; /* debug: a kind number = only that kind */
  if (gmin < 0) gmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
  return on && (on == 1 || (u32)on == kind) && (int)T >= gmin && nb % 4u == 0u &&
         (kind == (u32)MAT_Q4 || kind == (u32)MAT_Q4K || kind == (u32)MAT_Q5K || kind == (u32)MAT_IQ4XS || kind == (u32)MAT_IQ4NL);
}
#define IE_MS_MAXP 8192u /* the pairs (token, choice) of a chunk: kernels/ie_kernels.c MS_MAXP */
static int gq_use(const gpu_backend *b, uint32_t pos) { return b->attn_gq && pos + 1u >= b->gq_min; }

static void *upload(gpu_backend *b, const void *host, size_t bytes) {
  for (uint32_t i = 0; i < b->nh; i++)
    if (b->hk[i] == host) return b->hv[i]; /* tied output: one copy */
  void *dev = slab_take(&b->us, bytes);
  HIP(H.Memcpy(dev, host, bytes, hipMemcpyHostToDevice));
  if (b->nh == b->caph) {
    b->caph = b->caph ? 2 * b->caph : 64;
    b->hk = realloc(b->hk, b->caph * sizeof(void *));
    b->hv = realloc(b->hv, b->caph * sizeof(void *));
  }
  b->hk[b->nh] = host;
  b->hv[b->nh++] = dev;
  return dev;
}

static void upload_mat(gpu_backend *b, const mat *w, void **w0, void **w1) {
  if (w->d0) { /* streamed at load */
    *w0 = w->d0, *w1 = w->d1;
    return;
  }
  const size_t ne = w->n_exp > 1 ? w->n_exp : 1; /* experts: n_exp matrices one after the other */
  if (w->kind == MAT_RAWQ) { /* raw GGUF blocks: qs only */
    *w0 = NULL;
    *w1 = w->d1 ? w->d1 : upload(b, w->qs, (size_t)ie_rawq_bytes(w) * ne);
    return;
  }
  if (w->kind != MAT_F32) {
    *w0 = upload(b, w->qw, (size_t)mat_qw_words(w->kind, w->rows, w->cols) * 4 * ne);
    *w1 = upload(b, w->qs, (size_t)mat_qs_n(w->kind, w->rows, w->cols) * 2 * ne);
  } else {
    *w0 = upload(b, w->f, (size_t)w->rows * w->cols * 4 * ne);
    *w1 = NULL;
  }
}

static uint32_t gpu_step(backend *bk, uint32_t tok, uint32_t pos, float *logits, double *ms);
static void gpu_run(backend *bk, int sec, const uint32_t *toks, uint32_t T, uint32_t pos, int last_only, uint32_t *out,
                    float *prob);
static void ds_split_setup(gpu_backend *b);
static void ds_shortlist_setup(gpu_backend *b, const char *hsaco);
static void ds_ctx_drain(gpu_backend *b);

/* ie-lab (backend.h): weights changed in place. A streamed matrix lives in w->d0 (words, or floats) and w->d1
 * (scales); Q4_0, Q8_0 and F32 store row r at r * (words or floats per row), so rows r0 .. of a fused matrix are
 * one contiguous range of each array. Pinned host arrays (w->host) are written directly. */
static void patch_bytes(void *dst, int host, const void *src, size_t n) {
  if (host) memcpy(dst, src, n);
  else HIP(H.Memcpy(dst, src, n, hipMemcpyHostToDevice));
}
int gpu_patch_mat(backend *bk, const mat *w, uint32_t r0, const mat *part) {
  (void)bk;
  if (!w->d0 || part->kind != w->kind || part->cols != w->cols || r0 + part->rows > w->rows || w->n_exp > 1) return 0;
  HIP(H.DeviceSynchronize());
  if (w->kind == MAT_F32) {
    patch_bytes((float *)w->d0 + (size_t)r0 * w->cols, w->host, part->f, (size_t)part->rows * w->cols * 4);
    return 1;
  }
  if (w->kind != MAT_Q8 && w->kind != MAT_Q4) return 0;
  const size_t words = w->kind == MAT_Q8 ? 8 : 4; /* u32 per 32-weight block */
  patch_bytes((uint32_t *)w->d0 + (size_t)r0 * w->nb * words, w->host, part->qw, (size_t)part->rows * w->nb * words * 4);
  patch_bytes((uint16_t *)w->d1 + (size_t)r0 * w->nb, w->host, part->qs, (size_t)part->rows * w->nb * 2);
  return 1;
}
int gpu_patch_vec(backend *bk, const vec *v) {
  gpu_backend *b = (gpu_backend *)bk;
  const graph *g = b->g;
  const size_t n = (size_t)v->n * 4;
  int done = 0;
  HIP(H.DeviceSynchronize());
#define PUT(p) do { if (p) { HIP(H.Memcpy((p), v->f, n, hipMemcpyHostToDevice)); done++; } } while (0)
  for (uint32_t i = 0; i < g->n_ops; i++) {
    const op *o = &g->ops[i];
    if (o->v == v) PUT(o->w ? b->d[i].bias : b->d[i].w0);
    if (o->nv == v) PUT(b->d[i].nw);
    if (o->kind == OP_ATTN && o->sk == v) PUT(b->d[i].w1);
    if (o->kind == OP_GDN) {
      const layer *L = &b->m->l[o->layer];
      if (&L->conv == v) PUT(b->d[i].w0);
      if (&L->dt_bias == v) PUT(b->d[i].w1);
      if (&L->ssm_a == v) PUT(b->d[i].bias);
      if (&L->ssm_norm == v) PUT(b->d[i].nw);
    }
  }
#undef PUT
  return done;
}
static void gpu_accept(backend *bk, uint32_t k);
static int gpu_tree_ok(backend *bk, uint32_t pos, uint32_t T);
static void tree_begin(gpu_backend *b, u32 pos, u32 T);
static void gpu_accept_tree(backend *bk, const uint8_t *path, uint32_t k);
static void gpu_snap(backend *bk, int save, int k, int hbuf, uint32_t pos);
static u32 ink_moe_slot_max(const gpu_backend *b) {
  if (!b->k[K_INK_GEXP_RAW_IQ3XXS_SLOT_R6] && !b->k[K_INK_GEXP_IQ4NL_SLOT_R6]) return 0u;
  static int env_v = -2;
  if (env_v == -2) {
    const char *e = getenv("IE_INK_MOE_SLOT_MAX");
    env_v = (e && *e) ? atoi(e) : -1;
  }
  if (env_v >= 0) return (u32)env_v;
  return b->k[K_INK_MOE_SORT_COMPACT_R6] ? 2u : 16u;
}
static u32 ink_qtile_min(void) {
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("IE_INK_QTILE_MIN");
    v = (e && *e) ? atoi(e) : 2048;
  }
  return v > 0 ? (u32)v : 0u;
}
static size_t gpu_snap_io(backend *bk, int k, int out, void *buf, uint32_t pos);
static void gpu_kv_io(backend *bk, int out, uint32_t p0, uint32_t n, void *buf);
static void gpu_kv_trim(backend *bk, uint32_t keep_pos);
static void *gpu_host_alloc(backend *bk, size_t n);
static void gpu_host_free(backend *bk, void *p);
static void kv_ensure(gpu_backend *b, u32 need);
static void ds_kv_ensure(gpu_backend *b, u32 need);
static int kv_all_full(const graph *g, const model *m) {
  for (uint32_t i = 0; i < m->n_kvl; i++)
    if (g->kv_len[i] != g->n_ctx) return 0;
  return 1;
}
static void gpu_read_rows(backend *bk, int id, uint32_t r0, uint32_t n, float *dst);
static int gpu_topk(backend *bk, int id, uint32_t r0, uint32_t n, uint32_t nv, uint32_t K, float *val, uint32_t *idx);
static int topk_fits(uint32_t n, uint32_t nv, uint32_t K);
static int ink_moe_adap_mode(void) {
  static int m = -1;
  if (m < 0) m = getenv("IE_INK_MOE_ADAP_MODE") ? atoi(getenv("IE_INK_MOE_ADAP_MODE")) : 0;
  return m;
}
static void topk_prefetch(gpu_backend *b, uint32_t n);
static void topk_prefetch_clear(void);
static void ink_fused_q4k_head_run(gpu_backend *b, uint32_t T, int fold, uint32_t *out);
static void gpu_copy_rows(backend *bk, int dst, uint32_t d0, int src, uint32_t r0, uint32_t n);
static void gpu_close(backend *bk);
static double op_wbytes(const op *o);
static double ds_now_ms(void);

/* IE_TRACE=FILE records actual dispatches and their logical op spans. The
 * optional IE_TRACE_GPU=1 sidecar records HIP event times on the launch stream. */
static FILE *tr_f;
static FILE *tr_gpu;
static const char *tr_phase = "-";
typedef struct { int first, last, layer; double bytes; } tr_ops;
static tr_ops tr_op = {-1, -1, -1, 0}, tr_mk_op;
static uint32_t tr_call, tr_n, tr_pos, tr_T;
enum { TR_EVENT_CAP = 4096, TR_HIP_NOT_READY = 600 };
typedef struct { hipEvent_t begin, end; uint32_t launch; } tr_event;
static tr_event *tr_events;
static unsigned tr_head, tr_pending, tr_dropped, tr_invalid, tr_reported;
static uint32_t tr_stride = 1;

static void tr_drain(void) {
  while (tr_pending) {
    tr_event *e = &tr_events[tr_head];
    const hipError_t status = H.EventQuery(e->end);
    if (status == TR_HIP_NOT_READY) break;
    HIP(status);
    float ms;
    HIP(H.EventElapsedTime(&ms, e->begin, e->end));
    if (isfinite(ms) && ms >= 0) fprintf(tr_gpu, "%u,%.3f\n", e->launch, ms * 1000.0);
    else tr_invalid++;
    tr_head = (tr_head + 1u) % TR_EVENT_CAP;
    tr_pending--;
  }
}

static tr_event *tr_begin(void) {
  if (!tr_gpu) return NULL;
  if (tr_stride > 1u) {
    /* Hash the ordinal so repeated graph ops do not alias a fixed stride. */
    uint32_t sample = tr_n * 0x9e3779b9u;
    sample = (sample ^ (sample >> 16)) * 0x85ebca6bu;
    if ((sample ^ (sample >> 13)) % tr_stride) return NULL;
  }
  if (tr_pending == TR_EVENT_CAP) tr_drain();
  if (tr_pending == TR_EVENT_CAP) { tr_dropped++; return NULL; }
  const unsigned i = (tr_head + tr_pending) % TR_EVENT_CAP;
  tr_event *e = &tr_events[i];
  if (!e->begin) {
    HIP(H.EventCreate(&e->begin));
    HIP(H.EventCreate(&e->end));
  }
  e->launch = tr_n;
  HIP(H.EventRecord(e->begin, cur_stream));
  return e;
}

static void tr_launch(const char *name, unsigned groups, unsigned T, tr_ops *ops, unsigned commands, tr_event *e) {
  if (!tr_f) return;
  if (e) {
    HIP(H.EventRecord(e->end, cur_stream));
    tr_pending++;
  }
  fprintf(tr_f, "%u,%u,%u,%u,%s,%d,%d,%s,%u,%u,%.0f,%d,%u\n", tr_call, tr_pos, tr_T, tr_n++, tr_phase,
          ops->first, ops->layer, name, groups, T, ops->bytes, ops->last, commands);
  ops->bytes = 0; /* logical weight bytes appear once per op or fused span */
}

static void tr_span(const graph *g, unsigned i, unsigned n) {
  if (!tr_f) return;
  tr_op = (tr_ops){(int)i, (int)(i + n - 1u), g->ops[i].layer, 0};
  for (unsigned j = 0; j < n; j++) tr_op.bytes += op_wbytes(&g->ops[i + j]);
}

static void tr_mk_add(unsigned count) {
  if (!tr_f) return;
  if (count == 1u) tr_mk_op = tr_op;
  else {
    tr_mk_op.last = tr_op.last;
    if (tr_mk_op.layer != tr_op.layer) tr_mk_op.layer = -1;
    tr_mk_op.bytes += tr_op.bytes;
  }
  tr_op.bytes = 0;
}

void gpu_trace_flush(void) {
  if (tr_f) fflush(tr_f);
  if (tr_gpu) {
    tr_drain();
    fflush(tr_gpu);
    if (tr_reported != tr_dropped + tr_invalid) {
      tr_reported = tr_dropped + tr_invalid;
      fprintf(stderr, "GPU trace: %u samples skipped (queue full), %u invalid HIP durations\n", tr_dropped, tr_invalid);
    }
  }
}
/* Start a run of the trace: phase, first position, tokens. */
static void tr_run(const char *phase, uint32_t pos, uint32_t T) {
  if (!tr_f) return;
  if (tr_gpu) tr_drain();
  tr_phase = phase, tr_pos = pos, tr_T = T, tr_call++;
  tr_op = (tr_ops){-1, -1, -1, 0};
}

/* IE_SLAB_MB=N (default 0: one allocation per array): arrays packed into blocks of N MiB (the block with the least room
 * that fits; an array over half a block gets its own; new blocks zeroed at once). Saluki then has about 60 device
 * allocations instead of about 1000, but no more VRAM: Windows cuts this process's budget (15.06 -> about 9.5 GiB)
 * when its VRAM use passes what the desktop leaves free (about 10.2 GiB with 4.4 GiB used by other programs; IE_VRAM_TRACE=1
 * shows it), whatever the number of allocations, and then evicts whole allocations, which with large blocks moves more
 * to shared memory (7.3-7.7 GiB left in VRAM against 8.4-9.6 with one per array). slab_list: above. */
static size_t slab_bytes(void) {
  static size_t s = (size_t)-1;
  if (s == (size_t)-1) {
    const char *e = getenv("IE_SLAB_MB");
    s = (size_t)(e && atoi(e) > 0 ? atoi(e) : 0) << 20;
  }
  return s;
}
static void slab_keep(slab_list *s, void *p, char *cur, size_t left) {
  if (s->n == s->cap) {
    s->cap = s->cap ? 2 * s->cap : 64;
    s->a = realloc(s->a, s->cap * sizeof(void *));
    s->cur = realloc(s->cur, s->cap * sizeof(char *));
    s->left = realloc(s->left, s->cap * sizeof(size_t));
    if (!s->a || !s->cur || !s->left) ie_die("out of memory");
  }
  s->a[s->n] = p, s->cur[s->n] = cur, s->left[s->n++] = left;
}
static void *slab_take(slab_list *s, size_t bytes) {
  const size_t padded = bytes + 128u; /* >= 128 zeroed slack bytes so 128-byte coalesced raw-block loads never cross bounds */
  const size_t sb = slab_bytes(), need = (padded + 4095u) & ~(size_t)4095u;
  void *p;
  if (!sb || need > sb / 2u) {
    HIP(H.Malloc(&p, padded));
    HIP(H.Memset(p, 0, padded));
    slab_keep(s, p, NULL, 0);
    return p;
  }
  uint32_t best = s->n;
  for (uint32_t i = 0; i < s->n; i++)
    if (s->cur[i] && s->left[i] >= need && (best == s->n || s->left[i] < s->left[best])) best = i;
  if (best == s->n) {
    HIP(H.Malloc(&p, sb));
    HIP(H.Memset(p, 0, sb));
    slab_keep(s, p, (char *)p, sb);
  }
  p = s->cur[best], s->cur[best] += need, s->left[best] -= need;
  return p;
}
static void slab_free(slab_list *s) {
  for (uint32_t i = 0; i < s->n; i++) H.Free(s->a[i]);
  free(s->a), free(s->cur), free(s->left);
  memset(s, 0, sizeof *s);
}

/* Streaming load: the device arrays of the streamed matrices (freed at close). */
static slab_list sdev;
static uint32_t n_sdev; /* arrays */
static size_t sdev_bytes;
static void *shost[4096]; /* the arrays of mat.host matrices: pinned host memory */
static uint32_t n_shost;
static size_t shost_bytes;
static void *sdev_put(void *host, size_t bytes, int in_host) {
  void *dev;
  if (in_host && n_shost < 4096u) {
    HIP(H.HostMalloc(&dev, bytes + 128u, 0));
    memcpy(dev, host, bytes);
    memset((char *)dev + bytes, 0, 128u);
    shost[n_shost++] = dev, shost_bytes += bytes;
    vram_trace("pinned", bytes);
    free(host);
    return dev;
  }
  dev = slab_take(&sdev, bytes);
  HIP(H.Memcpy(dev, host, bytes, hipMemcpyHostToDevice));
  n_sdev++, sdev_bytes += bytes;
  free(host);
  return dev;
}
static void gpu_sink(mat *w) {
  if (!w->rows || w->d0) return;
  const size_t ne = w->n_exp > 1 ? w->n_exp : 1; /* experts: n_exp matrices one after the other */
  if (w->kind == MAT_RAWQ) { /* raw GGUF blocks: qs only */
    if (!w->qs) return;
    w->d1 = sdev_put(w->qs, (size_t)ie_rawq_bytes(w) * ne, w->host), w->qs = NULL;
    return;
  }
  if (w->kind == MAT_F32) {
    if (!w->f) return;
    w->d0 = sdev_put(w->f, (size_t)w->rows * w->cols * 4 * ne, w->host), w->f = NULL;
  } else {
    if (!w->qw) return;
    w->d0 = sdev_put(w->qw, (size_t)mat_qw_words(w->kind, w->rows, w->cols) * 4 * ne, w->host), w->qw = NULL;
    w->d1 = sdev_put(w->qs, (size_t)mat_qs_n(w->kind, w->rows, w->cols) * 2 * ne, w->host), w->qs = NULL;
  }
}
void gpu_stream_init(void) {
  hip_load();
  HIP(H.Init(0));
  HIP(H.SetDevice(0));
  ie_mat_sink = gpu_sink;
}

/* The megakernel (IE_MK=1, kernels/ie_mk.c): while mk.rec is set, the
 * launches of the kernels of the current build (mk.set: decode or MTP) go
 * to a command list (256 bytes each: {id, groups x, groups y, join}, then
 * the arguments packed as a launch packs them); mk_flush copies the list to
 * the device (asynchronously, from a ring of pinned host buffers) and runs
 * it as one cooperative launch of mk.np[set] workgroups. A launch of
 * another kernel flushes the list first, so the order is always the order
 * of the launches. */
#include "ie_mk_gen.h"
#include "../prof/sm_profiler.h"
enum { MK_CMD = 256, MK_CAP = 4096, MK_ARGEND = 232, MK_RING = 4 };
static struct {
  int on, rec, set;
  hipModule_t mod[MK_NSETS];
  hipFunction_t f[MK_NSETS];
  u32 np[MK_NSETS];
  int id[K_N];       /* kernel -> MK_ id, or -1 */
  unsigned char *cmd; /* host list: mk.ring[mk.slot] */
  unsigned char *ring[MK_RING]; /* pinned */
  hipEvent_t ev[MK_RING];       /* the copy of a ring buffer is done */
  int slot, mtp, light, free;
  int off[MK_NSETS]; /* a run of this kind had a kernel the build does not have */
  int nocoop[MK_NSETS]; /* Windows: cooperative launch is unavailable (719); a plain launch of np workgroups + the bounded barrier */
  void *dring[MK_RING]; /* device lists (one per ring buffer: a launch reads its own) */
  u32 *ws;           /* device: barrier counter, abort flag */
  volatile u32 *hflag; /* pinned: the abort flag after the last launch */
  u32 n, base, pf, pfall, join;
  uint64_t joined;
  uint64_t launches, cmds;
  /* IE_MK_PROF=1: per command the clock at its end; per kernel the sum */
  unsigned long *dprof, *hprof;
  int kof[MK_CAP];
  double pus[K_N];
  uint32_t pcnt[K_N], psteps;
  /* IE_MK_TRACE=FILE: the timeline of one launch (IE_MK_TRACE_AT: which,
   * default 10) per workgroup, to FILE (Chrome trace, prof/sm_profiler.h) */
  smp_buffer_t trace;
  const char *trace_file;
  uint64_t trace_at;
} mk;

/* The prefetch of the barrier after command i: the first bytes (up to
 * mk.pf, IE_MK_PF MB, default 16) of the weights of the next GEMV command j;
 * all the barriers before j share the claim cursor of j. */
/* The GEMV kernels whose weights can be streamed ahead: the repacked ones and
 * the raw-block ones (the latter are 64% of a 27B IQ2 decode). */
static int mk_is_gemv(int k) {
  return k == K_GEMV_Q4K || k == K_GEMV_Q6K || k == K_GEMV_Q8 || (k >= K_GEMV_RAW_Q2K && k <= K_GEMV_RAW_IQ3S);
}
/* Bytes of the first weight array of a GEMV command (exact for the raw kinds:
 * rows * (nb/8) * block bytes, which the type in the command selects). */
static uint64_t mk_wbytes(int k, const unsigned char *c) {
  u32 rows, nb, type = 0;
  if (k >= K_GEMV_RAW_Q2K && k <= K_GEMV_RAW_IQ3S) {
    memcpy(&rows, c + 40, 4), memcpy(&nb, c + 44, 4), memcpy(&type, c + 48, 4);
    return (uint64_t)rows * (nb / 8u) * (uint64_t)ie_rawq_bs(type);
  }
  memcpy(&rows, c + 48, 4), memcpy(&nb, c + 52, 4);
  return (uint64_t)rows * nb * (k == K_GEMV_Q8 ? 32u : 16u);
}
/* Weights are immutable, so the next GEMV's stream can be started while this
 * phase waits (Hazy's weight prefetch, Cohere's "stream before the dependency
 * is satisfied"). The size is now exact for every kind it targets, and the
 * 64 KB margin keeps the touch inside the matrix; IE_MK_PF stays off by
 * default because s_prefetch_data past an allocation faulted once on a small
 * model. */
static void mk_prefetch(void) {
  if (!mk.pf) return;
  for (u32 i = 0; i + 1u < mk.n; i++) {
    const int ki = mk.kof[i];
    if (mk.pfall == 0 && (ki == K_GEMV_Q4K || ki == K_GEMV_Q6K || ki == K_GEMV_Q8)) continue; /* memory busy */
    u32 j = i + 1u;
    for (; j < mk.n && !mk_is_gemv(mk.kof[j]); j++) {
    }
    if (j >= mk.n) continue;
    unsigned char *cj = mk.cmd + (size_t)j * MK_CMD;
    uint64_t bytes = mk_wbytes(mk.kof[j], cj);
    bytes = bytes > 65536u ? bytes - 65536u : 0u; /* stay inside the matrix */
    if (bytes > mk.pf) bytes = mk.pf;
    if (!bytes) continue;
    uint64_t qw;
    memcpy(&qw, cj + 16, 8); /* the first weight pointer of the command */
    unsigned char *d = mk.cmd + (size_t)i * MK_CMD;
    const u32 w[4] = {(u32)qw, (u32)(qw >> 32), (u32)bytes, j};
    memcpy(d + 232, w, sizeof w);
  }
}

/* A strided data range: n pieces of len bytes, piece t at base + t stride
 * (n = 1: one interval). */
typedef struct { uint64_t base, len, stride, n; } mk_sr;
typedef struct { mk_sr r, w, b, s; } mk_rng; /* activations read, result written, bias, residual */
static mk_sr mk_sr1(uint64_t base, uint64_t len, uint64_t stride, uint64_t n) {
  mk_sr x = {base, base ? len : 0u, stride, n ? n : 1u};
  return x;
}
static int mk_ivl(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1) { return a0 < b1 && b0 < a1; }
/* 1: the ranges may overlap. Exact for one interval each and for two ranges
 * with one stride whose pieces fit in it; else the hull intervals. */
static int mk_ovl(const mk_sr *x, const mk_sr *y) {
  if (!x->len || !y->len) return 0;
  const uint64_t x1 = x->base + (x->n - 1u) * x->stride + x->len, y1 = y->base + (y->n - 1u) * y->stride + y->len;
  if (!mk_ivl(x->base, x1, y->base, y1)) return 0;
  if (x->n > 1u && x->stride == y->stride && x->len <= x->stride && y->len <= y->stride) {
    const uint64_t S = x->stride, d = (y->base >= x->base ? y->base - x->base : x->base - y->base) % S;
    const uint64_t lo = y->base >= x->base ? x->len : y->len, hi = y->base >= x->base ? y->len : x->len;
    /* the later piece starts d after the earlier one, in every frame */
    return !(lo <= d && d + hi <= S);
  }
  return 1;
}
/* The data ranges of a GEMV command (MAT_Q4K / Q6K / Q8 or F32, one token
 * or several). 0: not a GEMV of these, or with the fused RMSNorm (its counter). */
static int mk_gemv_rng(u32 i, mk_rng *g) {
  const int k = mk.kof[i];
  const unsigned char *c = mk.cmd + (size_t)i * MK_CMD + 16;
  uint64_t p[8];
  u32 rows, n2, t[4];
  memset(g, 0, sizeof *g);
  if (k == K_GEMV_Q4K || k == K_GEMV_Q6K || k == K_GEMV_Q8 || k == K_GEMV_Q5K || k == K_GEMV_IQ4XS ||
      k == K_INK_GEMV_Q8_NB2 || k == K_INK_GEMV_Q8_SMALL || k == K_INK_GEMV_Q8_MED_R2) {
    /* qw qs xq y rows nb bias res nw ... */
    memcpy(p, c, 32), memcpy(&rows, c + 32, 4), memcpy(&n2, c + 36, 4), memcpy(p + 4, c + 40, 24);
    if (p[6]) return 0;
    g->r = mk_sr1(p[2], (uint64_t)n2 * 40u, 0, 1), g->w = mk_sr1(p[3], (uint64_t)rows * 4u, 0, 1);
    g->s = mk_sr1(p[5], (uint64_t)rows * 4u, 0, 1);
  } else if (k == K_GEMV_F32) {
    /* w x y rows cols bias res nw ... */
    memcpy(p, c, 24), memcpy(&rows, c + 24, 4), memcpy(&n2, c + 28, 4), memcpy(p + 4, c + 32, 24);
    if (p[6]) return 0;
    g->r = mk_sr1(p[1], (uint64_t)n2 * 4u, 0, 1), g->w = mk_sr1(p[2], (uint64_t)rows * 4u, 0, 1);
    g->s = mk_sr1(p[5], (uint64_t)rows * 4u, 0, 1);
  } else if (k >= K_GEMV_RAW_Q2K_TR && k <= K_GEMV_RAW_IQ3S_TR) {
    /* q xq y rows nb T bias res nw ny nq eps count xs ys rs: T tokens; xs,
     * ys and rs in bytes */
    u32 rows2, nb2, T2, xs2, ys2, rs2;
    memcpy(p, c, 24), memcpy(&rows2, c + 24, 4), memcpy(&nb2, c + 28, 4), memcpy(&T2, c + 32, 4), memcpy(p + 3, c + 40, 24);
    memcpy(&xs2, c + 96, 4), memcpy(&ys2, c + 100, 4), memcpy(&rs2, c + 104, 4);
    if (p[5]) return 0; /* nw unused by the TR kernels, but keep the guard */
    g->r = mk_sr1(p[1], (uint64_t)nb2 * 40u, xs2, T2), g->w = mk_sr1(p[2], (uint64_t)rows2 * 4u, ys2, T2);
    g->s = mk_sr1(p[4], (uint64_t)rows2 * 4u, rs2, T2);
    g->b = mk_sr1(p[3], (uint64_t)rows2 * 4u, 0, 1);
    return 1;
  } else if (k >= K_GEMV_RAW_Q2K && k <= K_GEMV_RAW_IQ3S) {
    /* q xq y rows nb type bias res nw ny nq eps count xs ys rs.
     * Same shape of ranges as the repacked GEMVs, one pointer slot later:
     * p[0] = the raw blocks, p[1] = xq, p[2] = y, then bias, res, nw.
     * Telling mk_join about them is what lets the independent raw GEMVs of a
     * layer (q, k, v / gate, up with mixed ggml types) share one phase
     * instead of meeting at a grid barrier each. */
    memcpy(p, c, 24), memcpy(&rows, c + 24, 4), memcpy(&n2, c + 28, 4), memcpy(p + 3, c + 40, 24);
    if (p[5]) return 0; /* the fused RMSNorm: its counter orders the phase */
    g->r = mk_sr1(p[1], (uint64_t)n2 * 40u, 0, 1), g->w = mk_sr1(p[2], (uint64_t)rows * 4u, 0, 1);
    g->s = mk_sr1(p[4], (uint64_t)rows * 4u, 0, 1);
    g->b = mk_sr1(p[3], (uint64_t)rows * 4u, 0, 1);
    return 1;
  } else if (k == K_GEMV_Q4K_TR || k == K_GEMV_Q6K_TR || k == K_GEMV_Q8_TR || k == K_GEMV_Q8_TS) {
    /* qw qs xq y rows nb bias res T xs ys rs: T tokens; xs in bytes, ys and rs in floats */
    memcpy(p, c, 32), memcpy(&rows, c + 32, 4), memcpy(&n2, c + 36, 4), memcpy(p + 4, c + 40, 16), memcpy(t, c + 56, 16);
    g->r = mk_sr1(p[2], (uint64_t)n2 * 40u, t[1], t[0]), g->w = mk_sr1(p[3], (uint64_t)rows * 4u, t[2] * 4ull, t[0]);
    g->s = mk_sr1(p[5], (uint64_t)rows * 4u, t[3] * 4ull, t[0]);
  } else if (k == K_GEXP) {
    /* qw qs xq y rows nb route qwn qsn astr bstr kind; grid y: the K slots (xq: astr apart, or one input; y: bstr apart) */
    u32 gy, as, bs;
    memcpy(p, c, 32), memcpy(&rows, c + 32, 4), memcpy(&n2, c + 36, 4), memcpy(&as, c + 56, 4), memcpy(&bs, c + 60, 4);
    memcpy(&gy, c - 8, 4);
    g->r = mk_sr1(p[2], (uint64_t)n2 * 40u, as, as ? gy : 1u), g->w = mk_sr1(p[3], (uint64_t)rows * 4u, bs, gy);
    return 1; /* no bias, no residual; the route is read (a GEMV phase does not write it) */
  } else if (k == K_GEMV_F32_T) {
    /* w x y rows cols bias res xs ys rs (bytes); grid y: the tokens */
    u32 gy;
    memcpy(p, c, 24), memcpy(&rows, c + 24, 4), memcpy(&n2, c + 28, 4), memcpy(p + 4, c + 32, 16), memcpy(t, c + 48, 12);
    memcpy(&gy, c - 8, 4);
    g->r = mk_sr1(p[1], (uint64_t)n2 * 4u, t[0], gy), g->w = mk_sr1(p[2], (uint64_t)rows * 4u, t[1], gy);
    g->s = mk_sr1(p[5], (uint64_t)rows * 4u, t[2], gy);
  } else if (k == K_INK_GEMV_F32_ROUTER_VER_R4) {
    /* w x y rows cols bias res xs ys rs T (bytes) */
    memcpy(p, c, 24), memcpy(&rows, c + 24, 4), memcpy(&n2, c + 28, 4), memcpy(p + 4, c + 32, 16), memcpy(t, c + 48, 16);
    g->r = mk_sr1(p[1], (uint64_t)n2 * 4u, t[0], t[3]), g->w = mk_sr1(p[2], (uint64_t)rows * 4u, t[1], t[3]);
    g->s = mk_sr1(p[5], (uint64_t)rows * 4u, t[2], t[3]);
  } else
    return 0;
  g->b = mk_sr1(p[4], (uint64_t)rows * 4u, 0, 1);
  return 1;
}
/* Command i joins the phase of the commands ph .. i - 1 (no barrier before
 * it: IE_MK_JOIN, default on) when all are GEMVs and no range i writes is
 * read or written by the phase and no range i reads is written by it. */
static void mk_join(void) {
  u32 ph = 0;
  for (u32 i = 1; i < mk.n; i++) {
    unsigned char *c = mk.cmd + (size_t)i * MK_CMD;
    mk_rng gi, gj;
    int ok = mk.join && mk_gemv_rng(i, &gi);
    for (u32 j = ph; ok && j < i; j++) {
      if (!mk_gemv_rng(j, &gj)) ok = 0;
      else if (mk_ovl(&gi.w, &gj.r) || mk_ovl(&gi.w, &gj.w) || mk_ovl(&gi.w, &gj.b) || mk_ovl(&gi.w, &gj.s) ||
               mk_ovl(&gj.w, &gi.r) || mk_ovl(&gj.w, &gi.b) || mk_ovl(&gj.w, &gi.s))
        ok = 0;
    }
    static int jd = -1;
    if (jd < 0) jd = getenv("IE_MK_JDEBUG") != NULL;
    if (jd && mk.launches == 2) {
      mk_rng gx;
      fprintf(stderr, "join %u %s: %s (gemv %d, phase from %u)\n", i, kname[mk.kof[i]], ok ? "joined" : "barrier", mk_gemv_rng(i, &gx), ph);
    }
    u32 f;
    memcpy(&f, c + 12, 4);
    f = (f & ~1u) | (ok ? 1u : 0u);
    memcpy(c + 12, &f, 4);
    if (!ok) ph = i;
    else mk.joined++;
  }
}

static void mk_flush(void) {
  if (!mk.n) return;
  mk_join();
  mk_prefetch();
  const u32 np = mk.np[mk.set];
  void *dc = mk.dring[mk.slot];
  HIP(H.MemcpyAsync(dc, mk.cmd, (size_t)mk.n * MK_CMD, hipMemcpyHostToDevice, cur_stream));
  uint64_t *tp = mk.trace && mk.launches == mk.trace_at ? smp_device_ptr(mk.trace) : NULL;
  void *a[] = {&dc, &mk.n, &mk.ws, &mk.base, &mk.dprof, &tp};
  tr_event *event = tr_begin();
  if (mk.nocoop[mk.set]) HIP(H.ModuleLaunchKernel(mk.f[mk.set], np, 1, 1, 256, 1, 1, 0, cur_stream, a, NULL));
  else HIP(H.ModuleLaunchCooperativeKernel(mk.f[mk.set], np, 1, 1, 256, 1, 1, 0, cur_stream, a));
  tr_launch("ie_mk", np, 1, &tr_mk_op, mk.n, event);
  HIP(H.EventRecord(mk.ev[mk.slot], cur_stream)); /* the copy and the launch are done */
  HIP(H.MemcpyAsync((void *)mk.hflag, mk.ws + 1, 4, hipMemcpyDeviceToHost, cur_stream));
  u32 nbar = 0;
  for (u32 i = 1; i < mk.n; i++) {
    u32 f;
    memcpy(&f, mk.cmd + (size_t)i * MK_CMD + 12, 4);
    nbar += (f & 1u) == 0u;
  }
  mk.base += np * nbar;
  mk.launches++, mk.cmds += mk.n;
  if (tp) {
    smp_print_stats(mk.trace, stderr);
    smp_export(mk.trace, mk.trace_file);
    smp_destroy(mk.trace), mk.trace = NULL;
  }
  if (mk.dprof) { /* the first command: from its predecessor's end is not known; skip it */
    HIP(H.Memcpy(mk.hprof, mk.dprof, (size_t)mk.n * 8u, hipMemcpyDeviceToHost));
    for (u32 i = 1; i + 1u < mk.n; i++) mk.pus[mk.kof[i]] += (mk.hprof[i] - mk.hprof[i - 1]) * 0.01, mk.pcnt[mk.kof[i]]++;
    mk.psteps++;
  }
  mk.n = 0;
  /* the next list goes to the next ring buffers, once their last launch is done */
  mk.slot = (mk.slot + 1) % MK_RING;
  HIP(H.EventSynchronize(mk.ev[mk.slot]));
  mk.cmd = mk.ring[mk.slot];
}
/* After a run (its results read with a blocking copy: every launch is done):
 * stop if a grid barrier of the megakernel timed out. */
static void mk_check(u32 pos) {
  if (mk.launches && *mk.hflag) ie_die("megakernel: a grid barrier timed out (run at position %u)", pos);
}
/* The MTP build of the megakernel has the multi-token GEMVs with 4 rows per
 * wave (ie_gemv_q4kq8_tr, ie_gemv_q6kq8_tr; bitwise equal to the others):
 * the 8-row ones (ts2, tr8) need about 192 VGPRs alone, and with them the
 * megakernel spills (IE_MK_LIGHT=0: keep them, launched apart). */
/* the builds (the sets of tools/mk_gen.py, in order), next to the kernels */
static const char *const mk_file[MK_NSETS] = {"ie_mk.hsaco", "ie_mk_mtp.hsaco", "ie_mk_flint.hsaco", "ie_mk_flintl.hsaco",
                                              "ie_mk_inkling.hsaco", "ie_mk_inkling_ver.hsaco"};
static const char *const mk_setname[MK_NSETS] = {"decode", "verify", "flint decode", "flint decode (long)",
                                                 "inkling decode", "inkling verify"};
/* the builds a model uses: flint its two, inkling2 its two, the others decode and MTP (the others are not loaded: off) */
static int mk_need(const model *m, int st) {
  if (m->arch == ARCH_FLINT) return st == MKSET_FLINT || st == MKSET_FLINTL;
  if (m->arch == ARCH_INKLING2) return st == MKSET_INKLING || st == MKSET_INKLING_VER;
  return st == MKSET_DECODE || st == MKSET_MTP;
}
static int mk_light(void) { return mk.rec && (mk.set == MKSET_MTP || mk.set == MKSET_INKLING_VER) && mk.light; }
/* 1 when kernel k is allowed in the current run: either megakernel recording is off, or k is in the active build. */
static int mk_ok_k(int k) { return !mk.rec || (mk.id[k] >= 0 && mk_in_set[mk.set][mk.id[k]]); }
/* 1: the launch went to the list */
static int mk_put(int k, unsigned gx, unsigned gy, unsigned lds_bytes, void **args) {
  if (!mk.rec) return 0;
  if (lds_bytes || mk.id[k] < 0 || !mk_in_set[mk.set][mk.id[k]]) {
    /* a kernel the build does not have: the runs of this kind go back to
     * launches (a list cut at every such kernel is slower than launches) */
    if (!mk.off[mk.set])
      fprintf(stderr, "gpu: megakernel off for %s runs (%s is not in it)\n", mk_setname[mk.set],
              kname[k]);
    mk.off[mk.set] = 1;
    mk_flush();
    return 0;
  }
  if (mk.n == MK_CAP) mk_flush();
  const int id = mk.id[k];
  mk.kof[mk.n] = k;
  unsigned char *c = mk.cmd + (size_t)mk.n * MK_CMD;
  memset(c, 0, MK_CMD);
  /* bit 1: the waves need not meet between workgroup bodies (no LDS, no
   * workgroup barrier): the GEMVs without the fused norm, the multi-token
   * GEMVs, the element-wise kernels */
  const int fk = k == K_GEMV_Q4K || k == K_GEMV_Q6K || k == K_GEMV_Q8 || k == K_GEMV_Q4 || k == K_GEMV_Q5K || k == K_GEMV_IQ4XS ||
                 k == K_INK_GEMV_Q8_NB2 || (k >= K_GEMV_RAW_Q2K && k <= K_GEMV_RAW_IQ3S_TR);
  const int fr = (fk && !*(void **)args[8]) || k == K_GEMV_Q4K_TR || k == K_GEMV_Q6K_TR || k == K_GEMV_Q8_TR ||
                 k == K_GEMV_Q8_TS || k == K_GEMV_Q8_T || k == K_SWIGLU || k == K_EMBED_Q8 || k == K_EMBED_Q4 || k == K_EMBED_F32 ||
                 k == K_ROPE_KV || k == K_QUANT || k == K_GEXP || k == K_MOE_SUM || k == K_MOE_ROUTE ||
                 k == K_ADD || k == K_ADD_T || k == K_COPY_ROWS || k == K_KV || k == K_KV_T || k == K_GATE || k == K_LOGIT_CAP ||
                 k == K_SCONV_RING || k == K_INK_VCLIP || k == K_INK_SCONV || k == K_INK_LOOP_GATE || k == K_INK_SITU ||
                 k == K_INK_MOE_SUM || k == K_INK_SCONV_ADD_R2 || k == K_INK_SCONV_HIST_R4 || k == K_INK_GATE_Q8_R2 ||
                 k == K_INK_EMBED_Q8_BATCH_R2 || k == K_INK_GEXP_IQ4NL || k == K_INK_GEXP_IQ4NL_R3 ||
                 k == K_INK_GEXP_IQ4NL_VER_R4 || k == K_INK_GEXP_RAW_IQ3XXS || k == K_INK_GEXP_RAW_IQ3XXS_R3 ||
                 k == K_INK_GEXP_RAW_IQ3XXS_VER_R4;
  const u32 h[4] = {(u32)id, gx, gy, fr && mk.free ? 2u : 0u};
  memcpy(c, h, sizeof h);
  size_t off = 16;
  for (int i = 0; i < mk_nargs[id]; i++) {
    const size_t sz = mk_argsz[id][i];
    off = (off + sz - 1) / sz * sz;
    if (off + sz > MK_ARGEND) ie_die("megakernel: the arguments of %s do not fit", kname[k]);
    memcpy(c + off, args[i], sz);
    off += sz;
  }
  mk.n++;
  return 1;
}

/* The device arrays of a matrix and their bytes (as gpu_sink made them): 1 or 2. */
static u32 mat_arrays(const mat *w, const void **p, size_t *n) {
  const size_t ne = w->n_exp > 1 ? w->n_exp : 1;
  if (w->kind == MAT_RAWQ) return p[0] = w->d1, n[0] = (size_t)ie_rawq_bytes(w) * ne, 1u;
  if (w->kind == MAT_F32) return p[0] = w->d0, n[0] = (size_t)w->rows * w->cols * 4 * ne, 1u;
  p[0] = w->d0, n[0] = (size_t)mat_qw_words(w->kind, w->rows, w->cols) * 4 * ne;
  p[1] = w->d1, n[1] = (size_t)mat_qs_n(w->kind, w->rows, w->cols) * 2 * ne;
  return 2u;
}

/* IE_HOST_STREAM: the layers whose matrices are in pinned host memory (IE_HOST_LAYERS / IE_HOST_SET) are copied by
 * DMA into IE_HOST_SLOTS (default 3, lowered to a divisor of the layer count) VRAM slots while the model runs,
 * instead of being read over PCIe by every GEMV (about 55 against 27 GB/s on PCIe 5.0, and the copies overlap the
 * other layers' work where copies and kernels overlap). Unset: the GEMVs read pinned memory directly. */
static void hs_setup(gpu_backend *b, const graph *g) {
  b->hs_on = 0, b->hs_pending = -1;
  if (b->m->arch == ARCH_INKLING2) return; /* 48 visits reuse 24 physical layers */
  const char *e = getenv("IE_HOST_STREAM");
  if (!(e && e[0] == '1')) return; /* opt-in: on this Windows runtime host->device copies do not overlap kernels
                                    * (tools/copy_overlap_probe.py: 19.3 + 21.0 ms alone, 40.4 together), so the
                                    * streamed layers measured slower than direct reads (Ripple-H, 16 layers: 158
                                    * against 88 ms/token) */
  /* the streamed layers: those with a pinned matrix, in op order */
  int32_t lay_s[4096];
  for (u32 l = 0; l < 4096u; l++) lay_s[l] = -1;
  u32 n = 0, na = 0, cap = 0;
  struct hs_arr *a = NULL;
  u32 *a0 = NULL;
  const mat **seen = NULL;
  u32 nseen = 0;
  b->hs_op = malloc(g->n_ops * sizeof(int32_t));
  for (u32 i = 0; i < g->n_ops; i++) b->hs_op[i] = -1;
  for (u32 i = 0; i < g->n_main; i++) {
    const op *o = &g->ops[i];
    if (!o->w || !o->w->host || o->layer < 0 || o->layer >= 4096) continue;
    if (lay_s[o->layer] < 0) {
      lay_s[o->layer] = (int32_t)n++;
      a0 = realloc(a0, (n + 1) * sizeof(u32));
      a0[n - 1] = na, a0[n] = na;
      b->hs_last = realloc(b->hs_last, n * sizeof(u32));
    }
    const u32 s = (u32)lay_s[o->layer];
    if (s + 1u != n) ie_die("IE_HOST_STREAM: the ops of a host layer are not contiguous (layer %d)", o->layer);
    b->hs_op[i] = (int32_t)s, b->hs_last[s] = i;
    int dup = 0;
    for (u32 k = 0; k < nseen; k++) dup |= seen[k] == o->w;
    if (dup) continue;
    seen = realloc(seen, (nseen + 1) * sizeof(*seen)), seen[nseen++] = o->w;
    const void *p[2];
    size_t nb[2];
    const u32 c = mat_arrays(o->w, p, nb);
    for (u32 k = 0; k < c; k++) {
      if (na == cap) a = realloc(a, (cap = cap ? 2 * cap : 64) * sizeof(*a));
      const size_t off = na > a0[s] ? (a[na - 1].off + a[na - 1].bytes + 255u) & ~(size_t)255u : 0u;
      a[na++] = (struct hs_arr){p[k], nb[k], off};
    }
    a0[n] = na;
  }
  free(seen);
  if (!n) {
    free(b->hs_op), b->hs_op = NULL, free(a), free(a0);
    return;
  }
  const char *se = getenv("IE_HOST_SLOTS");
  u32 ns = se && atoi(se) > 0 ? (u32)atoi(se) : 3u;
  if (ns > 16u) ns = 16u;
  if (ns > n) ns = n;
  while (n % ns) ns--; /* the slots cycle over passes: layer s and s + n use the same slot */
  size_t sb = 0;
  for (u32 s = 0; s < n; s++) {
    const struct hs_arr *l = &a[a0[s + 1] - 1];
    if (l->off + l->bytes > sb) sb = l->off + l->bytes;
  }
  for (u32 k = 0; k < ns; k++) {
    HIP(H.Malloc((void **)&b->hs_slot[k], sb));
    HIP(H.EventCreate(&b->hs_free[k]));
  }
  b->hs_done = malloc(n * sizeof(hipEvent_t));
  for (u32 s = 0; s < n; s++) HIP(H.EventCreate(&b->hs_done[s]));
  HIP(H.StreamCreateWithFlags(&b->hs_cs, 1u)); /* hipStreamNonBlocking: no implicit sync with the default stream */
  /* every op reading a streamed array now reads its slot copy */
  for (u32 i = 0; i < g->n_main; i++) {
    const int s = b->hs_op[i];
    if (s < 0) continue;
    char *base = b->hs_slot[(u32)s % ns];
    for (u32 k = a0[s]; k < a0[s + 1]; k++) {
      if (b->d[i].w0 == a[k].src) b->d[i].w0 = base + a[k].off;
      if (b->d[i].w1 == a[k].src) b->d[i].w1 = base + a[k].off;
    }
  }
  b->hs_n = n, b->hs_ns = ns, b->hs_a = a, b->hs_a0 = a0, b->hs_on = 1;
  /* The launches go to a created (blocking) stream: hipStreamWaitEvent on the NULL stream holds the host until the
   * event completes (Windows, measured: a wait for an 18 ms copy took 25 ms of host time), which would starve the GPU
   * at every streamed layer; on a created stream it returns at once. A blocking stream keeps its implicit order with
   * the NULL stream, so the synchronous reads of results stay after the work. */
  {
    hipStream_t cs;
    HIP(H.StreamCreate(&cs));
    cur_stream = cs;
  }
  for (u32 s = 0; s < ns; s++) hs_copy(b, s, 0); /* the first pass's first layers */
  size_t tot = 0;
  for (u32 k = 0; k < na; k++) tot += a[k].bytes;
  fprintf(stderr, "gpu: %u host layers (%.2f GB) streamed by DMA through %u VRAM slots of %.1f MB (IE_HOST_STREAM=1; unset: read over PCIe)\n",
          n, tot / 1e9, ns, sb / 1e6);
}

static void gpu_set_rope(backend *bk, uint32_t pos0, uint32_t n, const float *cos_tab, const float *sin_tab);

backend *gpu_open(const model *m, const graph *g, const char *hsaco) {
  hip_load();
  HIP(H.Init(0));
  int n = 0;
  HIP(H.GetDeviceCount(&n));
  if (n < 1) ie_die("GPU backend: no HIP device");
  HIP(H.SetDevice(0));

  /* IE_KV_Q8=1: the q8_0 KV cache, whose kernels are in NAME_q8.hsaco next to NAME.hsaco (build-win.sh); IE_KV_Q4=1: the
   * 4-bit KV cache, NAME_q4.hsaco */
  const char *kqe = getenv("IE_KV_Q8"), *kq4e = getenv("IE_KV_Q4");
  char *q8path = NULL;
  if (kqe && kqe[0] == '1' && kq4e && kq4e[0] == '1') ie_die("IE_KV_Q8=1 and IE_KV_Q4=1: one KV cache format");
  if ((kqe && kqe[0] == '1') || (kq4e && kq4e[0] == '1')) {
    const char *sfx = kqe && kqe[0] == '1' ? "_q8.hsaco" : "_q4.hsaco", *vn = kqe && kqe[0] == '1' ? "IE_KV_Q8=1" : "IE_KV_Q4=1";
    const size_t hl = strlen(hsaco);
    if (hl < 6 || strcmp(hsaco + hl - 6, ".hsaco")) ie_die("%s: %s is not a .hsaco file", vn, hsaco);
    if (hl >= 9 && !strcmp(hsaco + hl - 9, sfx)) q8path = strdup(hsaco);
    else {
      q8path = malloc(hl + 4);
      memcpy(q8path, hsaco, hl - 6);
      strcpy(q8path + hl - 6, sfx);
    }
    hsaco = q8path;
  }
  FILE *f = fopen(hsaco, "rb");
  if (!f) ie_die("GPU backend: cannot open %s (build it with make)", hsaco);
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *img = malloc((size_t)sz);
  if (fread(img, 1, (size_t)sz, f) != (size_t)sz) ie_die("cannot read %s", hsaco);
  fclose(f);

  gpu_backend *b = calloc(1, sizeof *b);
  b->base.step = gpu_step;
  b->base.run = gpu_run;
  b->base.copy_rows = gpu_copy_rows;
  b->base.accept = gpu_accept;
  b->base.tree_ok = gpu_tree_ok;
  b->base.accept_tree = gpu_accept_tree;
  b->base.snap = gpu_snap;
  b->base.snap_io = gpu_snap_io;
  b->base.kv_io = gpu_kv_io;
  b->base.kv_trim = gpu_kv_trim;
  b->base.snap_f32b = (size_t)m->n_rec * m->n_vh * m->sd * m->sd * 4;
  b->base.host_alloc = gpu_host_alloc;
  b->base.host_free = gpu_host_free;
  b->base.read_rows = gpu_read_rows;
  b->base.topk = gpu_topk;
  b->base.set_rope = gpu_set_rope;
  b->base.close = gpu_close;
  b->m = m;
  b->g = g;
  b->live_snap = -1;
  b->live_hbuf = -1;
  b->q16_ready_op = -1;
  HIP(H.ModuleLoadData(&b->mod, img));
  free(img);
  for (int i = 0; i < K_SALUKI_LORA_B; i++) {
    if (i == K_ATTN_LONG) { /* optional: a code object without it keeps the older attention kernels; heads of 128: ie_attn_long128 */
      b->has_long = H.ModuleGetFunction(&b->k[i], b->mod, m->hd == 128u ? "ie_attn_long128" : kname[i]) == 0;
      continue;
    }
    if (i == K_TOPK_MAX || i == K_TOPK_SEL) { /* optional: a code object without them keeps ie_topk_part + ie_topk_merge */
      const int ok = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      if (i == K_TOPK_MAX) b->has_topk_fast = ok;
      else b->has_topk_fast &= ok;
      continue;
    }
    if (i == K_ATTN_PL) { /* optional too (not in the f32 KV build): prompt chunks keep ie_attn_fa2 */
      b->has_pl = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i == K_GEMV_IQ4XS_TR8) { /* optional: without it IQ4_XS keeps ie_gemv_iq4xsq8_tr */
      b->has_iq4tr8 = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i == K_GEMV_Q8_TS) { /* optional: without it small Q8 GEMVs keep ie_gemv_q8q8_t */
      b->has_q8ts = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i >= K_GEMV_Q4K_TS16 && i <= K_GEMV_Q6K_TS4) { /* optional: without them 9 to 16 tokens keep the _tr kernels */
      const int ok = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      b->has_ts16 = i == K_GEMV_Q4K_TS16 ? ok : b->has_ts16 && ok;
      continue;
    }
    if (i == K_COPY_LIST) { /* optional: without it no tree verify */
      b->has_cl = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i == K_DF_ATTN_G) { /* optional: without it DFlash keeps ie_df_attn */
      b->has_dfg = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i == K_ATTN_SPLIT_TP) { /* optional: without it verifies use ie_attn_split_t */
      b->has_split_tp = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i == K_ATTN_LONG_E) { /* optional: without it every launch uses ie_attn_long */
      b->has_long_e = H.ModuleGetFunction(&b->k[i], b->mod, m->hd == 128u ? "ie_attn_long128_e" : kname[i]) == 0;
      continue;
    }
    if (i == K_ATTN_LONG_MP) { /* optional: single-KV-pass multi-pair-group Q8 long attention */
      b->has_long_mp = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i == K_GEMV_RAW_TR8_M) { /* optional: without it every raw GEMV has its own launch */
      b->has_tr8m = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i == K_GEMV_RAW_TR4_M) { /* optional: without it 2 to 4 tokens launch each raw GEMV apart */
      b->has_tr4m = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i >= K_GEMV_RAW_SWIGLU && i <= K_GEMV_RAW_SWIGLU_TR8) { /* optional: fused raw gate + up + SwiGLU + Q8 */
      const int ok = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      b->has_raw_swiglu = i == K_GEMV_RAW_SWIGLU ? ok : b->has_raw_swiglu && ok;
      continue;
    }
    if (i == K_GDN1P) { /* optional: without it ie_gdn1 */
      b->has_gdn1p = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i == K_QKN_ROPE_KV_ROT || i == K_ATTN_UNROT) { /* optional: IE_KV_ROT=1 needs both */
      const int ok = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      b->kvrot = i == K_QKN_ROPE_KV_ROT ? ok : b->kvrot && ok;
      continue;
    }
    if (i == K_GEMV_IQ4XS_TR16) {
      b->has_iq4tr16 = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i >= K_GEMM_RAW32_Q2K && i <= K_GEMM_RAW32_IQ3S) {
      const int ok = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      b->has_rq32 = i == K_GEMM_RAW32_Q2K ? ok : b->has_rq32 && ok;
      continue;
    }
    if (i == K_GEMM_RQ_SWIGLU) {
      b->has_rq_swiglu = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i >= K_GEMM_RQ_T16_Q2K && i <= K_GEMM_RQ_T16_IQ3S) {
      const int ok = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      b->has_rqt16 = i == K_GEMM_RQ_T16_Q2K ? ok : b->has_rqt16 && ok;
      continue;
    }
    if (i == K_GEMM_RQ_T16_M) {
      b->has_rqt16_m = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i == K_GEMM_RQ_T16_SWIGLU) {
      b->has_rqt16_swiglu = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i >= K_GDN_PREP_RING && i <= K_GDN_SEQ_WMMA_GNORM) {
      const int ok = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      b->has_gdn_wmma = i == K_GDN_PREP_RING ? ok : b->has_gdn_wmma && ok;
      continue;
    }
    if (i == K_RMSNORM_MT) {
      b->has_norm_mt = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i == K_MTP_PRENORM_Q8) {
      b->has_mtp_prenorm = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i >= K_GEMV_RAW_Q2K_TR16 && i <= K_GEMV_RAW_IQ3S_TR16) {
      const int ok = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      b->has_tr16 = i == K_GEMV_RAW_Q2K_TR16 ? ok : b->has_tr16 && ok;
      continue;
    }
    if (i == K_GEMV_RAW_TR16_M) {
      b->has_tr16m = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i == K_GEMV_RAW_SWIGLU_TR16) {
      b->has_swiglu_tr16 = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i == K_Q_SCALE_F16) {
      b->has_q_scale_f16 = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    if (i == K_GEMV_Q4K_SWIGLU) {
      b->has_q4k_swiglu = H.ModuleGetFunction(&b->k[i], b->mod, kname[i]) == 0;
      continue;
    }
    HIP(H.ModuleGetFunction(&b->k[i], b->mod, kname[i]));
  }
  const char *saluki_path = getenv("IE_SALUKI_KERNELS");
  const char *saluki_mask = getenv("IE_SALUKI_KERNEL_MASK");
  const char *saluki_mask_hi = getenv("IE_SALUKI_KERNEL_MASK_HI");
  const u32 saluki_base_bits = (u32)(K_SALUKI_RQ128_Q2K - K_SALUKI_LORA_B);          /* bits 0..20 */
  const u32 saluki_ext_bits = (u32)(K_SALUKI_END - K_SALUKI_ATTN_PL);                 /* bits 23..30 (R8), 31..42 (R9), 43..56 (R10), 57..66 (R11) */
  const u32 saluki_total_bits = saluki_base_bits + 2u + saluki_ext_bits;             /* 67 bits (0..66) */
  const uint64_t saluki_all_lo = saluki_total_bits >= 64u ? ~0ULL : ((1ULL << (saluki_total_bits & 63u)) - 1ULL);
  const uint64_t saluki_all_hi = saluki_total_bits <= 64u ? 0ULL :
                                 (saluki_total_bits >= 128u ? ~0ULL : ((1ULL << ((saluki_total_bits - 64u) & 63u)) - 1ULL));
  const uint64_t saluki_r8 = (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_RQ128_IQ2S_PIPE - K_SALUKI_ATTN_PL))) - 1ULL; /* 2147483647 */
  const uint64_t saluki_r9_win3 = saluki_r8                                          /* 283467841535 (R8 + 3 promoted R9 winners) */
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_RQ128_IQ2S_PIPE - K_SALUKI_ATTN_PL)))
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_RQ128_IQ3S_PIPE - K_SALUKI_ATTN_PL)))
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_IQ3XXS_TR9_PIPE1 - K_SALUKI_ATTN_PL)));
  const uint64_t saluki_r10_win8 = saluki_r9_win3                                    /* 39943341881688063 = 0x8de841ffffffff (R9-win3 + 8 promoted R10 winners) */
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_RQ128_IQ2XXS_R10 - K_SALUKI_ATTN_PL)))
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_RQ128_IQ1S_R10 - K_SALUKI_ATTN_PL)))
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_ATTN_PL_R10 - K_SALUKI_ATTN_PL)))
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_ATTN_LONG_MP_R10 - K_SALUKI_ATTN_PL)))
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_ATTN_LONG_MP_TR_R10 - K_SALUKI_ATTN_PL)))
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_IQ2XXS_TR9_R10 - K_SALUKI_ATTN_PL)))
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_IQ1S_TR9_R10 - K_SALUKI_ATTN_PL)))
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_IQ1M_TR9_R10 - K_SALUKI_ATTN_PL)));
  const uint64_t saluki_r11_win8_lo = saluki_r10_win8                                /* 0x7e8de841ffffffff (R10-win8 + R11 bits 57..62) */
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_RQ128_IQ2XXS_R11 - K_SALUKI_ATTN_PL)))
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_RQ128_IQ2S_R11 - K_SALUKI_ATTN_PL)))
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_GEMM_Q8R_R11 - K_SALUKI_ATTN_PL)))
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_QKN_ROPE_KV_R11 - K_SALUKI_ATTN_PL)))
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_ATTN_LONG_MP_R11 - K_SALUKI_ATTN_PL)))
    | (1ULL << (saluki_base_bits + 2u + (u32)(K_SALUKI_GDN_WY_WMMA_R11 - K_SALUKI_ATTN_PL)));
  const uint64_t saluki_r11_win8_hi =                                                /* 0x6 (R11 bits 65, 66 -> combined 0x67e8de841ffffffff) */
      (1ULL << ((saluki_base_bits + 2u + (u32)(K_SALUKI_LORA_B_R11 - K_SALUKI_ATTN_PL)) - 64u))
    | (1ULL << ((saluki_base_bits + 2u + (u32)(K_SALUKI_TOPK_R11 - K_SALUKI_ATTN_PL)) - 64u));
  const uint64_t saluki_default_lo = saluki_r11_win8_lo;                             /* 64 adapted kernels (R11-win8) */
  const uint64_t saluki_default_hi = saluki_r11_win8_hi;
  uint64_t mask_lo = saluki_default_lo, mask_hi = saluki_default_hi;
  int saluki_mask_hex128 = 0;
  if (saluki_mask) {
    const char *p = saluki_mask;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
      p += 2;
      mask_lo = 0ULL;
      mask_hi = 0ULL;
      u32 hex_digits = 0u;
      for (; *p; p++) {
        uint64_t nib;
        if (*p >= '0' && *p <= '9') nib = (uint64_t)(*p - '0');
        else if (*p >= 'a' && *p <= 'f') nib = (uint64_t)(*p - 'a' + 10);
        else if (*p >= 'A' && *p <= 'F') nib = (uint64_t)(*p - 'A' + 10);
        else break;
        mask_hi = (mask_hi << 4) | (mask_lo >> 60);
        mask_lo = (mask_lo << 4) | nib;
        hex_digits++;
      }
      saluki_mask_hex128 = (hex_digits > 16u);
    } else {
      mask_lo = (uint64_t)strtoull(saluki_mask, NULL, 0);
      mask_hi = 0ULL;
    }
  }
  if (!saluki_mask_hex128 && saluki_mask_hi && *saluki_mask_hi) {
    mask_hi = (uint64_t)strtoull(saluki_mask_hi, NULL, 0);
  }
  if (saluki_path && *saluki_path && strcmp(saluki_path, "0") &&
      ((mask_lo & saluki_all_lo) | (mask_hi & saluki_all_hi))) {
    if (H.ModuleLoad(&b->saluki_mod, saluki_path) == 0) {
      for (int i = K_SALUKI_LORA_B; i < K_SALUKI_END; i++) {
        u32 b_idx;
        if (i < K_SALUKI_RQ128_Q2K) b_idx = (u32)(i - K_SALUKI_LORA_B);
        else if (i < K_SALUKI_RQ64_Q2K) b_idx = saluki_base_bits;                    /* bit 21 = 2097152: all 8 rq128 */
        else if (i < K_SALUKI_ATTN_PL) b_idx = saluki_base_bits + 1u;                /* bit 22 = 4194304: all 8 rq64 */
        else b_idx = saluki_base_bits + 2u + (u32)(i - K_SALUKI_ATTN_PL);            /* bits 23..30: R8; 31..42: R9; 43..56: R10; 57..66: R11 */
        const int bit_on = (b_idx < 64u) ? ((mask_lo & (1ULL << (b_idx & 63u))) != 0ULL)
                         : (b_idx < 128u) ? ((mask_hi & (1ULL << ((b_idx - 64u) & 63u))) != 0ULL)
                                          : 0;
        hipFunction_t fn;
        if (bit_on &&
            H.ModuleGetFunction(&fn, b->saluki_mod, kname[i]) == 0) {
          b->k[i] = fn;
          fprintf(stderr, "gpu: enabled %s\n", kname[i]);
        }
      }
    } else {
      fprintf(stderr, "gpu: could not load optional kernels %s; using base module\n", saluki_path);
      b->saluki_mod = NULL;
    }
  }
  /* Auto-load architecture-agnostic 1-launch ie_saluki_topk_r11 when Saluki module wasn't explicitly set (IE_TOPK_R11=0: off). */
  if (!b->k[K_SALUKI_TOPK_R11] && !(saluki_path && *saluki_path) && !(saluki_mask && !mask_lo && !mask_hi) &&
      !(getenv("IE_TOPK_R11") && getenv("IE_TOPK_R11")[0] == '0')) {
    char tk_auto[1024];
    const char *tk_path = NULL;
    const char *sl1 = strrchr(hsaco, '/'), *sl2 = strrchr(hsaco, '\\');
    const char *sl = sl1 && sl2 ? (sl1 > sl2 ? sl1 : sl2) : (sl1 ? sl1 : sl2);
    if (sl) {
      const size_t dlen = (size_t)(sl - hsaco) + 1u;
      if (dlen + sizeof("ie_saluki_kernels.hsaco") < sizeof(tk_auto)) {
        memcpy(tk_auto, hsaco, dlen);
        strcpy(tk_auto + dlen, "ie_saluki_kernels.hsaco");
        tk_path = tk_auto;
      }
    } else {
      tk_path = "ie_saluki_kernels.hsaco";
    }
    if (tk_path && !b->saluki_mod && H.ModuleLoad(&b->saluki_mod, tk_path) != 0) b->saluki_mod = NULL;
    if (b->saluki_mod) {
      hipFunction_t fn;
      if (H.ModuleGetFunction(&fn, b->saluki_mod, kname[K_SALUKI_TOPK_R11]) == 0) {
        b->k[K_SALUKI_TOPK_R11] = fn;
        fprintf(stderr, "gpu: enabled %s\n", kname[K_SALUKI_TOPK_R11]);
      }
    }
  }
  {
    const char *ink_env = getenv("IE_INKLING_KERNELS");
    uint64_t ink_mask = 0ULL, ink_mask2 = 0ULL;
    ie_ink_parse_mask(&ink_mask, &ink_mask2);
    {
      const char *ge = getenv("IE_INKLING_GRAPH"), *pe = getenv("IE_INKLING_PWGRAPH");
      if (m->arch == ARCH_INKLING2 && ink_mask != 0ULL &&
          ((ge && atoi(ge) > 0) || (pe && pe[0] == '1')))
        ink_mask |= (1ULL << 26) | (1ULL << 33) | (1ULL << 34) | (1ULL << 35);
    }
    b->ink_mask = ink_mask;
    b->ink_mask2 = ink_mask2;
    char ink_auto[1024];
    const char *ink_path = NULL;
    if (ink_env && *ink_env) {
      if (strcmp(ink_env, "0") != 0) ink_path = ink_env;
    } else if (m->arch == ARCH_INKLING2) {
      const char *sl1 = strrchr(hsaco, '/'), *sl2 = strrchr(hsaco, '\\');
      const char *sl = sl1 && sl2 ? (sl1 > sl2 ? sl1 : sl2) : (sl1 ? sl1 : sl2);
      if (sl) {
        const size_t dlen = (size_t)(sl - hsaco) + 1u;
        if (dlen + sizeof("ie_inkling_kernels.hsaco") < sizeof(ink_auto)) {
          memcpy(ink_auto, hsaco, dlen);
          strcpy(ink_auto + dlen, "ie_inkling_kernels.hsaco");
          ink_path = ink_auto;
        }
      } else {
        ink_path = "ie_inkling_kernels.hsaco";
      }
    }
    if (ink_path && (ink_mask | ink_mask2)) {
      if (H.ModuleLoad(&b->inkling_mod, ink_path) == 0) {
        for (int i = K_INK_REL_ATTN_PF; i < K_N; i++) {
          const u32 bi = (u32)ink_bit[i - K_INK_REL_ATTN_PF];
          const int bit_on = (bi < 64u) ? (((ink_mask >> bi) & 1ULL) != 0ULL)
                           : (bi < 128u) ? (((ink_mask2 >> (bi - 64u)) & 1ULL) != 0ULL) : 0;
          hipFunction_t fn;
          if (bit_on && H.ModuleGetFunction(&fn, b->inkling_mod, kname[i]) == 0) {
            b->k[i] = fn;
          }
        }
        if (ink_mask2)
          fprintf(stderr, "gpu: loaded %s (mask 0x%llx) mask2 0x%llx\n", ink_path,
                  (unsigned long long)ink_mask, (unsigned long long)ink_mask2);
        else
          fprintf(stderr, "gpu: loaded %s (mask 0x%llx)\n", ink_path, (unsigned long long)ink_mask);
      } else {
        b->inkling_mod = NULL;
      }
    }
    if (m->arch == ARCH_INKLING2) {
      HIP(H.Malloc((void **)&b->ink_ctx_dev, 16u));
      HIP(H.HostMalloc((void **)&b->ink_ctx_host, 64u, 0));
      b->ink_ctx_host[0] = 0u;
      b->ink_ctx_host[1] = 0u;
      b->ink_ctx_host[2] = 0u;
      if (b->k[K_INK_SUBSPAN_GLUE_R8]) {
        const size_t sp_bytes = 64u * 16u * 80u;
        HIP(H.Malloc(&b->ink_sp_dev, sp_bytes));
        if (H.HostMalloc(&b->ink_sp_pin, sp_bytes, 0) != 0) b->ink_sp_pin = NULL;
        for (u32 k = 0; k < 64u; k++) HIP(H.EventCreate(&b->ink_sp_ev[k]));
        HIP(H.Malloc((void **)&b->ink_sp_sync, 16u));
        HIP(H.Memset(b->ink_sp_sync, 0, 16u));
      }
      b->ink_toks_cap = g->T ? g->T : 16u;
      HIP(H.Malloc((void **)&b->ink_toks_dev, (size_t)b->ink_toks_cap * 4u));
      if (H.HostMalloc((void **)&b->ink_toks_pin, 16u * 512u * 4u, 0) != 0) b->ink_toks_pin = NULL;
      if (b->ink_toks_pin) {
        for (u32 k = 0; k < 16u; k++) HIP(H.EventCreate(&b->ink_toks_ev[k]));
      }
      const u32 lc_tiles = (g->T + 63u) / 64u;
      const size_t lc_cb = (size_t)(4u * (lc_tiles ? lc_tiles : 1u)) * 4u;
      HIP(H.Malloc((void **)&b->ink_lc_count, lc_cb));
      HIP(H.Memset(b->ink_lc_count, 0, lc_cb));
    }
  }
  {
    hipFunction_t f;
    b->kvq8 = H.ModuleGetFunction(&f, b->mod, "ie_kv_q8") == 0;
    b->kvq4 = H.ModuleGetFunction(&f, b->mod, "ie_kv_q4") == 0;
    b->kvb = b->kvq8 || b->kvq4 ? 1u : H.ModuleGetFunction(&f, b->mod, "ie_kv_f16") == 0 ? 2u : 4u;
    if (kqe && kqe[0] == '1' && !b->kvq8) ie_die("IE_KV_Q8=1: %s has no q8 KV kernels", hsaco);
    if (kq4e && kq4e[0] == '1' && !b->kvq4) ie_die("IE_KV_Q4=1: %s has no 4-bit KV kernels", hsaco);
    b->kvrowb = b->kvq4   ? (size_t)m->n_kv * (m->hd / 2u + m->hd / 8u)
                : b->kvq8 ? (size_t)m->n_kv * (m->hd + m->hd / 16u)
                          : (size_t)m->n_kv * m->hd * b->kvb;
    /* the disk prompt cache's rows (kv_io): the layers whose cache holds every position; a sliding window layer's ring
     * goes with the snapshot instead (gpu_snap_io) */
    u32 n_full_kv = 0;
    for (u32 i = 0; i < m->n_kvl; i++) n_full_kv += g->kv_len[i] >= g->n_ctx;
    b->base.kv_rowb = b->kvrowb * 2u * n_full_kv;
    const char *mk_env = getenv("IE_MK");
    const char *mki_env = m->arch == ARCH_INKLING2 ? getenv("IE_MK_INKLING") : NULL;
    const int mk_req = mki_env ? (mki_env[0] == '1') : (mk_env && mk_env[0] == '1');
    if (b->kvq4) { /* hd / 2 bytes of nibbles and hd / 32 pairs of fp16 (d, m) per head: value = fmaf(q, d, m) */
      if ((m->arch != ARCH_QWEN35 && m->arch != ARCH_INKLING2 && !m->qk_norm) || m->hd % 32u || m->hd > 256u) /* qwen35, qwen3, inkling2 */
        ie_die("IE_KV_Q4=1: the 4-bit KV cache is for the qwen35 and qwen3 architectures with head size a multiple of 32 up to 256 (this model: arch %d, head size %u)",
               (int)m->arch, (unsigned)m->hd);
      if (mk_req) ie_die("IE_KV_Q4=1 does not work with the megakernel (IE_MK=1 / IE_MK_INKLING=1): its attention reads an fp16 cache");
      fprintf(stderr, "gpu: KV cache 4 bit (%zu bytes per position and layer, q8_0: %zu, fp16: %zu)\n", b->kvrowb,
              (size_t)m->n_kv * (m->hd + m->hd / 16u), (size_t)m->n_kv * m->hd * 2u);
    }
    if (b->kvq8) {
      /* q8 rows: hd int8 values + hd / 32 fp16 scales per head; the converted kernels are the qwen35 ones */
      if ((m->arch != ARCH_QWEN35 && m->arch != ARCH_INKLING2 && !m->qk_norm) || m->hd % 32u || m->hd > 256u) /* qwen35, qwen3, inkling2 */
        ie_die("IE_KV_Q8=1: the q8 KV cache is for the qwen35 and qwen3 architectures with head size a multiple of 32 up to 256 (this model: arch %d, head size %u)",
               (int)m->arch, (unsigned)m->hd);
      if (mk_req) ie_die("IE_KV_Q8=1 does not work with the megakernel (IE_MK=1 / IE_MK_INKLING=1): its attention reads an fp16 cache");
      fprintf(stderr, "gpu: KV cache q8_0 (%zu bytes per position and layer, fp16: %zu)\n", b->kvrowb,
              (size_t)m->n_kv * m->hd * 2u);
    }
    { /* IE_KV_ROT=1: the rotated KV cache (Walsh-Hadamard per head; off by default) */
      const char *re = getenv("IE_KV_ROT");
      const int want = re && re[0] == '1';
      if (want && !b->kvrot) ie_die("IE_KV_ROT=1: %s has no ie_qkn_rope_kv_rot / ie_attn_unrot", hsaco);
      if (want && (m->arch != ARCH_QWEN35 || m->hd < 32u || m->hd > 256u || (m->hd & (m->hd - 1u))))
        ie_die("IE_KV_ROT=1: for the qwen35 architecture with a power-of-two head size 32 .. 256 (head size %u)", (unsigned)m->hd);
      if (want && mk_req) ie_die("IE_KV_ROT=1 does not work with the megakernel (IE_MK=1 / IE_MK_INKLING=1)");
      b->kvrot = want;
      if (want) fprintf(stderr, "gpu: KV cache rotated (Walsh-Hadamard, %u values per head)\n", (unsigned)m->hd);
    }
    if (mk_req && m->arch == ARCH_INKLING2 && b->kvb != 2u)
      ie_die("IE_MK / IE_MK_INKLING requires an f16 KV module (kvb=%u)", b->kvb);
  }
  free(q8path);
  /* The megakernel (IE_MK=1; off by default: separate launches are faster
   * on gfx1201, research/raw-gemv-experiments.md): decode steps (ie_mk.hsaco) and
   * verify runs (ie_mk_mtp.hsaco) through it; the files next to the kernels. IE_MK_NP,
   * IE_MK_NP_MTP: the workgroups, default 128: two per CU, the most that are
   * resident with 192 VGPRs and 14 KB of LDS (the hardware places a
   * workgroup in one CU: the occupancy query, per WGP, says 4) */
  /* IE_MK=1 (or IE_MK_INKLING=1 for Inkling-2; IE_MK_INKLING=0 forces off): on. Never enabled for non-f16/rotated KV. */
  const char *mke = getenv("IE_MK");
  const char *mkie = m->arch == ARCH_INKLING2 ? getenv("IE_MK_INKLING") : NULL;
  int mkon = mkie ? (mkie[0] == '1') : (mke && mke[0] == '1');
  if (b->kvq8 || b->kvq4 || b->kvb != 2u || b->kvrot) mkon = 0;
  if (mkon) {
    const char *sl = strrchr(hsaco, '/'), *sb = strrchr(hsaco, '\\');
    if (!sl || (sb && sb > sl)) sl = sb;
    const size_t dl = sl ? (size_t)(sl - hsaco + 1) : 0;
    for (int st = 0; st < MK_NSETS && mkon; st++) {
      if (!mk_need(m, st)) continue;
      char *path = malloc(dl + 32);
      memcpy(path, hsaco, dl);
      strcpy(path + dl, mk_file[st]);
      FILE *mf = fopen(path, "rb");
      if (!mf) fprintf(stderr, "gpu: megakernel off (no %s)\n", path), mkon = 0;
      else fclose(mf);
      free(path);
    }
  }
  if (mkon) {
    const char *sl = strrchr(hsaco, '/'), *sb = strrchr(hsaco, '\\');
    if (!sl || (sb && sb > sl)) sl = sb;
    const size_t dl = sl ? (size_t)(sl - hsaco + 1) : 0;
    for (int st = 0; st < MK_NSETS; st++) {
      if (!mk_need(m, st)) {
        mk.off[st] = 1;
        continue;
      }
      char *path = malloc(dl + 32);
      memcpy(path, hsaco, dl);
      strcpy(path + dl, mk_file[st]);
      FILE *mf = fopen(path, "rb");
      if (!mf) ie_die("IE_MK: cannot open %s (make %s)", path, path);
      fseek(mf, 0, SEEK_END);
      const long ms = ftell(mf);
      fseek(mf, 0, SEEK_SET);
      char *mi = malloc((size_t)ms);
      if (fread(mi, 1, (size_t)ms, mf) != (size_t)ms) ie_die("cannot read %s", path);
      fclose(mf);
      HIP(H.ModuleLoadData(&mk.mod[st], mi));
      free(mi), free(path);
      HIP(H.ModuleGetFunction(&mk.f[st], mk.mod[st], "ie_mk"));
    }
    const char *skip = getenv("IE_MK_SKIP"); /* debug: kernels launched as usual (comma list) */
    for (int i = 0; i < K_N; i++) {
      mk.id[i] = -1;
      for (int j = 0; j < MK_N; j++)
        if (!strcmp(kname[i], mk_names[j])) mk.id[i] = j;
      if (skip) {
        const char *q = strstr(skip, kname[i]);
        const size_t kl = strlen(kname[i]);
        if (q && (q[kl] == ',' || q[kl] == 0) && (q == skip || q[-1] == ',')) mk.id[i] = -1;
      }
    }
    mk.np[MKSET_DECODE] = getenv("IE_MK_NP") ? (u32)atoi(getenv("IE_MK_NP")) : 128u;
    mk.np[MKSET_MTP] = getenv("IE_MK_NP_MTP") ? (u32)atoi(getenv("IE_MK_NP_MTP")) : 128u;
    mk.np[MKSET_FLINT] = mk.np[MKSET_FLINTL] = mk.np[MKSET_DECODE];
    mk.np[MKSET_INKLING] = getenv("IE_MK_NP") ? (u32)atoi(getenv("IE_MK_NP")) : 96u;
    mk.np[MKSET_INKLING_VER] = getenv("IE_MK_NP_MTP") ? (u32)atoi(getenv("IE_MK_NP_MTP")) : mk.np[MKSET_INKLING];
    /* IE_MK_MTP: 3 (default) verify runs only, 1 verify runs and drafts, 2 drafts only, 0 neither
     * (measured, MTP code 7 drafts: launches 94.2, 3: 99.2, 1: 98.4 tokens/s; a draft has
     * few launches, and a cooperative launch costs more than an ordinary one) */
    mk.mtp = getenv("IE_MK_MTP") ? atoi(getenv("IE_MK_MTP")) : 3;
    mk.light = !getenv("IE_MK_LIGHT") || atoi(getenv("IE_MK_LIGHT")) != 0;
    mk.free = !getenv("IE_MK_FREE") || atoi(getenv("IE_MK_FREE")) != 0; /* 0: the waves meet before every body */
    mk.pfall = getenv("IE_MK_PFALL") != NULL;
    mk.join = !getenv("IE_MK_JOIN") || atoi(getenv("IE_MK_JOIN")) != 0;
    /* IE_MK_PF (MB): in the barriers after small kernels the waiting workgroups prefetch the next GEMV's weights;
     * default 0 for Inkling-2 (off), 16 for others. */
    const u32 pf_def = m->arch == ARCH_INKLING2 ? 0u : 16u;
    mk.pf = (getenv("IE_MK_PF") ? (u32)atoi(getenv("IE_MK_PF")) : pf_def) << 20;
    for (int r = 0; r < MK_RING; r++) {
      HIP(H.HostMalloc((void **)&mk.ring[r], (size_t)MK_CAP * MK_CMD, 0));
      HIP(H.EventCreate(&mk.ev[r]));
    }
    mk.cmd = mk.ring[0];
    HIP(H.HostMalloc((void **)&mk.hflag, 64, 0));
    *mk.hflag = 0;
    for (int r = 0; r < MK_RING; r++) HIP(H.Malloc(&mk.dring[r], (size_t)MK_CAP * MK_CMD));
    HIP(H.Malloc((void **)&mk.ws, 64));
    HIP(H.Memset(mk.ws, 0, 64));
    if (getenv("IE_MK_PROF")) {
      HIP(H.Malloc((void **)&mk.dprof, (size_t)MK_CAP * 8u));
      mk.hprof = malloc((size_t)MK_CAP * 8u);
    }
    /* Prepare a 2-noop command list (1 real mk_grid_sync_pf barrier) to prove all np workgroups are truly co-resident. */
    memset(mk.ring[0], 0, 2u * MK_CMD);
    const u32 noop_h[4] = {0u, 1u, 0u, 0u}; /* tot = gx * gy = 0 skips body; i=0 -> i=1 runs 1 grid barrier */
    memcpy(mk.ring[0], noop_h, sizeof noop_h);
    memcpy(mk.ring[0] + MK_CMD, noop_h, sizeof noop_h);
    HIP(H.Memcpy(mk.dring[0], mk.ring[0], 2u * MK_CMD, hipMemcpyHostToDevice));
    int coop_attr = -1; /* hipDeviceAttributeCooperativeLaunch: 0 = the platform cannot do it */
    if (H.DeviceGetAttribute) H.DeviceGetAttribute(&coop_attr, 10, 0);
    for (int st = 0; st < MK_NSETS; st++) {
      if (!mk_need(m, st)) continue;
      const int is_ink = st == MKSET_INKLING || st == MKSET_INKLING_VER;
      const char *npe = (st == MKSET_MTP || st == MKSET_INKLING_VER)
                            ? (getenv("IE_MK_NP_MTP") ? getenv("IE_MK_NP_MTP") : getenv("IE_MK_NP"))
                            : getenv("IE_MK_NP");
#ifdef _WIN32
      if (!npe && is_ink && mk.np[st] > 96u) mk.np[st] = 96u; /* WDDM safety cap: never exceed 96 blocks */
#endif
      if (coop_attr == 0) { /* Windows/WDDM: occupancy cap + real 2-noop 1-barrier co-residency probe */
        int cu = 0, occ = 0;
        if (!npe && H.DeviceGetAttribute && H.DeviceGetAttribute(&cu, 63, 0) == 0 && cu > 0 && H.ModuleOccupancy &&
            H.ModuleOccupancy(&occ, mk.f[st], 256, 0) == 0 && occ > 0) {
          const u32 fit = (u32)(is_ink ? (occ >= 2 ? occ - 1 : 1) : (occ >= 3 ? occ - 1 : occ)) * (u32)cu;
          if (fit < mk.np[st]) mk.np[st] = fit;
        }
        int ok = 0;
        for (;;) {
          HIP(H.Memset(mk.ws, 0, 64));
          u32 n2 = 2u, b0 = 0u;
          void *nul = NULL;
          void *a0[] = {&mk.dring[0], &n2, &mk.ws, &b0, &nul, &nul};
          hipError_t pe = H.ModuleLaunchKernel(mk.f[st], mk.np[st], 1, 1, 256, 1, 1, 0, cur_stream, a0, NULL);
          if (pe == 0) {
            HIP(H.DeviceSynchronize());
            u32 wsh[2] = {0u, 0u};
            HIP(H.Memcpy(wsh, mk.ws, 8u, hipMemcpyDeviceToHost));
            if (wsh[1] == 0u) { ok = 1; break; }
            fprintf(stderr, "gpu: %s barrier probe timed out at np=%u; stepping down\n", mk_setname[st], mk.np[st]);
          } else {
            fprintf(stderr, "gpu: %s plain launch refused at np=%u (%d)\n", mk_setname[st], mk.np[st], pe);
          }
          if (mk.np[st] > 96u) mk.np[st] = 96u;
          else if (mk.np[st] > 64u) mk.np[st] = 64u;
          else if (mk.np[st] > 32u) mk.np[st] = 32u;
          else break;
        }
        HIP(H.Memset(mk.ws, 0, 64));
        if (ok) {
          mk.nocoop[st] = 1;
          fprintf(stderr, "gpu: %s runs: no cooperative launch on this platform (attribute 0); plain launch of %u "
                          "workgroups (occupancy %d x %d CUs, barrier probe ok)\n", mk_setname[st], mk.np[st], occ, cu);
          continue;
        }
        fprintf(stderr, "gpu: megakernel off for %s runs (co-residency probe failed)\n", mk_setname[st]);
        mk.off[st] = 1;
        continue;
      }
      for (;;) {
        HIP(H.Memset(mk.ws, 0, 64));
        u32 n2 = 2u, b0 = 0u;
        void *nul = NULL;
        void *a0[] = {&mk.dring[0], &n2, &mk.ws, &b0, &nul, &nul};
        hipError_t ce = H.ModuleLaunchCooperativeKernel(mk.f[st], mk.np[st], 1, 1, 256, 1, 1, 0, cur_stream, a0);
        if (getenv("IE_MK_DEBUG")) fprintf(stderr, "mk probe %s np=%u -> %d (%s)\n", mk_setname[st], mk.np[st], ce, H.GetErrorString ? H.GetErrorString(ce) : "?");
        if (ce == 0) {
          HIP(H.DeviceSynchronize());
          u32 wsh[2] = {0u, 0u};
          HIP(H.Memcpy(wsh, mk.ws, 8u, hipMemcpyDeviceToHost));
          if (wsh[1] == 0u) break;
        }
        if (mk.np[st] <= 32u) {
          int cu = 0, occ = 0;
          u32 np0 = mk.np[st];
          if (H.DeviceGetAttribute &&
              H.DeviceGetAttribute(&cu, 63 /* hipDeviceAttributeMultiprocessorCount */, 0) == 0 && cu > 0 &&
              H.ModuleOccupancy && H.ModuleOccupancy(&occ, mk.f[st], 256, 0) == 0 && occ > 0 &&
              (u32)occ * (u32)cu < mk.np[st])
            mk.np[st] = (u32)occ * (u32)cu;
          HIP(H.Memset(mk.ws, 0, 64));
          hipError_t pe = H.ModuleLaunchKernel(mk.f[st], mk.np[st], 1, 1, 256, 1, 1, 0, cur_stream, a0, NULL);
          if (pe == 0) {
            HIP(H.DeviceSynchronize());
            u32 wsh[2] = {0u, 0u};
            HIP(H.Memcpy(wsh, mk.ws, 8u, hipMemcpyDeviceToHost));
            if (wsh[1] == 0u) {
              mk.nocoop[st] = 1;
              fprintf(stderr, "gpu: %s runs: cooperative launch is unavailable; a plain launch of %u workgroups "
                              "(occupancy %d x %d CUs, base %u; barrier probe ok)\n",
                      mk_setname[st], mk.np[st], occ, cu, np0);
              break;
            }
          }
          fprintf(stderr, "gpu: megakernel off for %s runs (no cooperative launch: %d)\n", mk_setname[st], pe);
          mk.off[st] = 1;
          break;
        }
        mk.np[st] /= 2u;
      }
      HIP(H.Memset(mk.ws, 0, 64));
    }
    mk.base = 0u;
    HIP(H.DeviceSynchronize());
    if ((mk.trace_file = getenv("IE_MK_TRACE")) && mk.trace_file[0]) {
      u32 nmax = 0;
      for (int st = 0; st < MK_NSETS; st++) nmax = mk.np[st] > nmax ? mk.np[st] : nmax;
      mk.trace_at = getenv("IE_MK_TRACE_AT") ? (uint64_t)atoll(getenv("IE_MK_TRACE_AT")) : 10u;
      if ((mk.trace = smp_create(nmax, 1, 16384, 1))) {
        for (int j = 0; j < MK_N && j < 30; j++) smp_register_event(mk.trace, (uint32_t)j, mk_names[j] + 3);
        if (MK_N > 30) smp_register_event(mk.trace, 30, "other");
        smp_register_event(mk.trace, 31, "grid barrier");
      }
    }
    mk.on = 1;
    static const char *const mkm[4] = {"off", "verify runs and drafts", "drafts", "verify runs"};
    if (m->arch == ARCH_INKLING2)
      fprintf(stderr, "gpu: megakernel on (inkling decode: %u workgroups; inkling verify: %s, %u workgroups)\n",
              mk.np[MKSET_INKLING], mk.mtp >= 0 && mk.mtp <= 3 ? mkm[mk.mtp] : "?", mk.np[MKSET_INKLING_VER]);
    else
      fprintf(stderr, "gpu: megakernel on (decode: %u workgroups; MTP: %s, %u workgroups)\n",
              mk.np[MKSET_DECODE], mk.mtp >= 0 && mk.mtp <= 3 ? mkm[mk.mtp] : "?", mk.np[MKSET_MTP]);
  }

  HIP(H.Malloc((void **)&b->arena, g->arena));
  HIP(H.Memset(b->arena, 0, g->arena));
  HIP(H.Malloc((void **)&b->am_part, 16u * ARGMAX_GROUPS * 12u));
  if (H.HostMalloc((void **)&b->am_pin, 256u, 0) != 0) b->am_pin = NULL;
  if ((m->arch == ARCH_FLINT || m->arch == ARCH_INKLING2) && m->n_exp) {
    HIP(H.Malloc((void **)&b->ms_offs, (m->n_exp + 1u) * 4u));
    HIP(H.Malloc((void **)&b->ms_list, IE_MS_MAXP * 4u));
    HIP(H.Malloc((void **)&b->ms_packs, 512u * 4u));
    HIP(H.Malloc((void **)&b->ms_npacks, 4u));
    HIP(H.Malloc((void **)&b->ms_packs1, 512u * 4u));
    HIP(H.Malloc((void **)&b->ms_npacks1, 4u));
    HIP(H.Malloc((void **)&b->ms_packsM, 512u * 4u));
    HIP(H.Malloc((void **)&b->ms_npacksM, 4u));
  }
  b->ink_moe_all_slot = (m->arch == ARCH_INKLING2 && m->n_exp > 0u);
  if (b->ink_moe_all_slot) {
    for (u32 i = 0; i < g->n_ops; i++) {
      if (g->ops[i].kind != OP_GEMV_EXP) continue;
      const mat *w = g->ops[i].w;
      const int ok_iq3 = w && w->kind == MAT_RAWQ && w->sub == 18u && w->rows > 0u && w->nb >= 8u && (w->nb & 7u) == 0u;
      const int ok_iq4 = w && w->kind == MAT_IQ4NL && w->rows > 0u && w->nb > 0u;
      if (!ok_iq3 && !ok_iq4) { b->ink_moe_all_slot = 0; break; }
    }
  }
  /* a verify run of up to 8 tokens (7 MTP drafts) needs 8 states; with the
   * prompt lookup (src/gen.c, up to 15 drafts) 16 */
  b->ns = g->T > 4u && (g->mtp_h >= 0 || g->df_feat >= 0 || g->ds_feat >= 0) ? (g->T >= 16u && spec_verify_max() > 8u ? 16u : 8u) : 4u;
  HIP(H.Malloc((void **)&b->am_count, 16u * 4u));
  HIP(H.Memset(b->am_count, 0, 16u * 4u)); /* the kernel sets it back to 0 */
  HIP(H.Malloc((void **)&b->gm_count, 4u));
  HIP(H.Memset(b->gm_count, 0, 4u)); /* the kernel sets it back to 0 */
  {
    const size_t rcnt_b = (size_t)((m->dim + 31u) / 32u ? (m->dim + 31u) / 32u : 1u) * 4u;
    const size_t qh_b = (size_t)(m->n_head ? m->n_head : 1u) * 4u;
    HIP(H.Malloc((void **)&b->ink_moe_rcount, rcnt_b));
    HIP(H.Memset(b->ink_moe_rcount, 0, rcnt_b));
    HIP(H.Malloc((void **)&b->ink_qhead_cnt, qh_b));
    HIP(H.Memset(b->ink_qhead_cnt, 0, qh_b));
    HIP(H.Malloc((void **)&b->ink_sconv_rcnt, 16u * 4u));
    HIP(H.Memset(b->ink_sconv_rcnt, 0, 16u * 4u));
  }
  b->at_tmax = 16u; /* ie_attn_split_t: up to 16 tokens (speculative verify) */
  HIP(H.Malloc((void **)&b->at_count, m->n_head * 4u * b->at_tmax));
  HIP(H.Memset(b->at_count, 0, m->n_head * 4u * b->at_tmax)); /* the kernel sets it back to 0 */
  const char *ae = getenv("IE_ATTN");
  /* ie_attn_long (kernels/ie_kernels.c) takes the positions from long_min on; the older split kernels then only cover the
   * positions below it, so their part area (at_pstride per token) is small and the LDS limit of their merge weights
   * (IE_ATT_MAX_SPLIT) never applies. ie_attn_long keeps its partial results (at_lstride per token) in the same area.
   * model.c sizes OP_ATTN's scores buffer, the part area of one token, the same way (ie_att_part_floats). */
  b->long_min = ie_att_long_min();
  const int lw = b->has_long && !mk.on && ie_att_long_ok(m);
  const uint32_t old_ctx = lw && b->long_min < g->n_ctx ? b->long_min : g->n_ctx;
  b->at_pstride = m->n_head * ie_att_max_split(old_ctx) * (m->hd + 2u);
  b->at_lstride = lw ? m->n_head * ie_atl_parts(g->n_ctx) * IE_ATL_PSZ : 0u;
  b->at_ltmax = 0u;
  b->at_part = NULL; /* allocated below, once the number of verify tokens is known */
  b->attn_split = m->hd <= IE_ATT_MAX_HD && ie_att_max_split(old_ctx) <= IE_ATT_MAX_SPLIT && !(ae && !strcmp(ae, "old"));
  { /* the scores buffer of OP_ATTN (model.c) holds the part area of one token */
    size_t sc = (size_t)-1;
    for (uint32_t i = 0; i < g->n_ops; i++)
      if (g->ops[i].kind == OP_ATTN && g->bufs[g->ops[i].c].stride == 0u && g->bufs[g->ops[i].c].size < sc) sc = g->bufs[g->ops[i].c].size;
    const size_t need = (size_t)(b->at_pstride > b->at_lstride ? b->at_pstride : b->at_lstride) * 4u;
    if (b->attn_split && sc < need) { /* sized for ie_attn_long, which this build does not run */
      b->attn_split = 0;
      fprintf(stderr, "gpu: split attention off: the scores buffer (%zu bytes) is too small for %u positions without ie_attn_long\n", sc, g->n_ctx);
    }
  }
  b->attn_check = b->attn_split && ae && !strcmp(ae, "check");
  b->attn_gq = !(getenv("IE_ATTN_SW") && atoi(getenv("IE_ATTN_SW")));
  b->flint = m->arch == ARCH_FLINT;
  b->gq_min = getenv("IE_ATTN_GQ_MIN") ? (uint32_t)atoi(getenv("IE_ATTN_GQ_MIN")) : IE_GQ_MIN;
  b->gq_ch = getenv("IE_ATTN_GQ_CH") ? (uint32_t)atoi(getenv("IE_ATTN_GQ_CH")) : 256u;
  b->gq_ch_fixed = getenv("IE_ATTN_GQ_CH") != NULL;
  if (b->gq_ch < 1u) ie_die("IE_ATTN_GQ_CH: >= 1");
  b->gq_s = getenv("IE_ATTN_GQ_S") ? (uint32_t)atoi(getenv("IE_ATTN_GQ_S")) : 32u;
  if (b->gq_s < 1u) ie_die("IE_ATTN_GQ_S: >= 1");
  b->attn_kv = b->attn_split && !(ae && (!strcmp(ae, "split") || !strcmp(ae, "check"))) && ie_att_ch(0) % 8u == 0u &&
               IE_ATT_MAX_SPLIT <= 4096u && m->hd % 4u == 0u; /* ie_attn_kv: the merge weights in its LDS buffer, float4 loads */
  {
    const uint32_t nsl = (g->n_ctx + IE_ATL_CH - 1u) / IE_ATL_CH, ngl = (nsl + IE_ATL_G - 1u) / IE_ATL_G; /* splits and groups of ie_attn_long */
    b->attn_long = lw && b->attn_kv && b->at_lstride >= 4u;
    b->at_gmax = ngl + 1u;
    if (b->attn_long) {
      const size_t gn = (size_t)b->at_tmax * m->n_head * b->at_gmax;
      HIP(H.Malloc((void **)&b->at_gcount, gn * 4u));
      HIP(H.Memset(b->at_gcount, 0, gn * 4u)); /* the kernel sets each counter back to 0 */
    }
  }
  if (b->attn_check) {
    HIP(H.Malloc((void **)&b->chk, m->n_head * m->hd * 4u));
    b->chk_a = ie_alloc(m->n_head * m->hd * 4u);
    b->chk_b = ie_alloc(m->n_head * m->hd * 4u);
  }
  if (g->T > 1u) { /* ie_requant_sb's buffer */
    size_t mx = 0;
    const int ri = getenv("IE_RAW_GEMM_I256") != NULL && getenv("IE_RAW_GEMM_I256")[0] > '0'; /* also the raw GEMM's requantized tokens */
    for (uint32_t i = 0; i < g->n_ops; i++)
      if ((g->ops[i].kind == OP_GEMV_Q4K || (ri && g->ops[i].kind == OP_GEMV_RAW)) && (size_t)g->bufs[g->ops[i].a].stride > mx)
        mx = g->bufs[g->ops[i].a].stride;
    if (mx) {
      b->rq_bytes = mx * g->T;
      HIP(H.Malloc(&b->rq, b->rq_bytes));
    }
    size_t mk = 0; /* the largest Q4_K K (fp16 bytes per token), or scaled fp16 Q for ie_attn_pl */
    for (uint32_t i = 0; i < g->n_ops; i++)
      if (g->ops[i].kind == OP_GEMV_Q4K && (size_t)g->ops[i].w->nb * 64u > mk) mk = (size_t)g->ops[i].w->nb * 64u;
    if (b->has_pl && m->hd == 256u && (size_t)m->n_head * 512u > mk) mk = (size_t)m->n_head * 512u;
    if (mk) {
      b->xh_bytes = mk * g->T;
      HIP(H.Malloc(&b->xh, b->xh_bytes));
    }
  }
  const size_t kv = (size_t)g->kv_total * b->kvrowb + 4;
  int has_full_kv = 0;
  for (uint32_t i = 0; i < m->n_kvl; i++)
    if (g->kv_len[i] == g->n_ctx) has_full_kv = 1;
  b->kv_grow = (!mk.on || m->arch == ARCH_INKLING2) && m->arch != ARCH_FLINT && (kv_all_full(g, m) || (m->arch == ARCH_INKLING2 && has_full_kv)) &&
               !(getenv("IE_KV_GROW") && getenv("IE_KV_GROW")[0] == '0');
  if (b->kv_grow) {
    const uint32_t na = m->n_kvl ? m->n_kvl : 1u;
    b->kv_step = getenv("IE_KV_GROW_STEP") && atoi(getenv("IE_KV_GROW_STEP")) > 0 ? (u32)atoi(getenv("IE_KV_GROW_STEP")) : 4096u;
    b->kcl = calloc(na, sizeof(void *)), b->vcl = calloc(na, sizeof(void *));
    if (!b->kcl || !b->vcl) ie_die("out of memory");
    /* a kernel may read a few rows past the last position (a tile of 32 keys): kv_slack rows are mapped beyond it */
    const size_t va_rows = (size_t)g->n_ctx + KV_SLACK;
    b->kv_grow = 2; /* separate arrays unless VMM works (for mixed archs like inkling2, SWA stays compact in b->us) */
    if (H.MemAddressReserve && H.MemCreate && H.MemMap && H.MemSetAccess && H.MemUnmap && H.MemRelease && H.MemAddressFree &&
        H.MemGetAllocationGranularity && !(getenv("IE_KV_VMM") && getenv("IE_KV_VMM")[0] == '0')) {
      hipMemAllocProp prop = {1 /* pinned */, 0, 1 /* device */, 0, NULL, 0, 0, 0};
      size_t gran = 0;
      const size_t want = (getenv("IE_KV_CHUNK_MB") && atoi(getenv("IE_KV_CHUNK_MB")) > 0 ? (size_t)atoi(getenv("IE_KV_CHUNK_MB")) : 4u) << 20;
      if (H.MemGetAllocationGranularity(&gran, &prop, 0) == 0 && gran) {
        b->kv_chunk = (want + gran - 1) / gran * gran;
        b->kv_va = (va_rows * b->kvrowb + 4 + b->kv_chunk - 1) / b->kv_chunk * b->kv_chunk;
        int ok = 1;
        for (uint32_t i = 0; i < na * 2u && ok; i++) {
          if (g->kv_len[i >> 1] < g->n_ctx) continue;
          void **slot = i & 1u ? &b->vcl[i >> 1] : &b->kcl[i >> 1];
          ok = H.MemAddressReserve(slot, b->kv_va, 0, NULL, 0) == 0;
        }
        if (ok) {
          b->kv_grow = 1;
          b->kv_nchunk = b->kv_va / b->kv_chunk;
          b->kv_h = calloc((size_t)na * 2u * b->kv_nchunk, sizeof(void *));
          if (!b->kv_h) ie_die("out of memory");
        } else { /* give back what was reserved */
          for (uint32_t i = 0; i < na; i++) {
            if (g->kv_len[i] < g->n_ctx) continue;
            if (b->kcl[i]) H.MemAddressFree(b->kcl[i], b->kv_va), b->kcl[i] = NULL;
            if (b->vcl[i]) H.MemAddressFree(b->vcl[i], b->kv_va), b->vcl[i] = NULL;
          }
        }
      }
    }
    for (uint32_t i = 0; i < na; i++) {
      if (b->kv_grow == 1 && g->kv_len[i] == g->n_ctx) continue;
      const size_t rows = g->kv_len[i] < g->n_ctx ? (size_t)g->kv_len[i] : va_rows;
      b->kcl[i] = slab_take(&b->us, rows * b->kvrowb + 4);
      b->vcl[i] = slab_take(&b->us, rows * b->kvrowb + 4);
    }
    b->kv_cap = 0;
    kv_ensure(b, 1u);
  } else {
    HIP(H.Malloc((void **)&b->kc, kv));
    HIP(H.Malloc((void **)&b->vc, kv));
  }
  const char *gre = getenv("IE_GDN_RECORDS");
  /* 16 linear attention states (verify runs of up to 16 tokens) only if IE_GDN_RECORDS=0 and they
   * fit with 512 MB to spare; with IE_GDN_RECORDS (default), only 1 state + 15 compact records are allocated */
  if (b->ns == 16u && ((gre && gre[0] == '0') || !m->n_rec || m->sd != 128u)) {
    size_t fr = 0, tot = 0;
    const size_t s16 = (size_t)m->n_rec * 16u * m->n_vh * m->sd * m->sd * 4;
    if (H.MemGetInfo(&fr, &tot) == 0 && fr < s16 + ((size_t)512 << 20)) {
      b->ns = 8u;
      fprintf(stderr, "gpu: %.0f MB free: 8 linear attention states (verify runs of up to 8 tokens)\n", fr / 1e6);
    }
  }
  b->base.verify_max = b->ns > 4u ? b->ns : 1u;
  /* the part area (T > 1): at_tmax tokens of the older kernels' partial results (small when ie_attn_long takes the long
   * positions) or, if more, at_ltmax tokens of ie_attn_long's: the verify run (one launch; a longer chunk of up to 16
   * tokens goes in several, a token's result does not depend on the others) */
  if (b->attn_long) {
    b->at_ltmax = b->base.verify_max < b->at_tmax ? b->base.verify_max : b->at_tmax;
    /* a long context: the area holds at_ltmax tokens at the context's full length, halved while it passes 256 MB (131072
     * positions: 29 MB per token, 8 tokens). A launch packs its tokens at the stride its own positions need (at_lpack), so
     * shorter contexts fit more of them: 85K positions 12 tokens, 131072 8. Each token launched apart re-reads the whole
     * KV cache of the layer: at 85K a 6-token verify in launches of 4 + 2 took 60.6 ms, in one 40.3. */
    while (b->at_ltmax > 4u && (size_t)b->at_lstride * 4u * b->at_ltmax > ((size_t)256 << 20)) b->at_ltmax /= 2u;
    const char *lt = getenv("IE_ATTN_LONG_T");
    if (lt && atoi(lt) >= 1) b->at_ltmax = (uint32_t)atoi(lt) < b->at_tmax ? (uint32_t)atoi(lt) : b->at_tmax;
    b->at_lpack = !(getenv("IE_ATTN_LONG_PACK") && getenv("IE_ATTN_LONG_PACK")[0] == '0');
  }
  if (g->T > 1u && (b->attn_split || m->arch == ARCH_INKLING2)) {
    const size_t po = b->attn_split ? (size_t)b->at_pstride * b->at_tmax : 0u;
    const size_t pl = b->attn_split ? (size_t)b->at_lstride * b->at_ltmax : 0u;
    const u32 ivmax = b->base.verify_max > 1u ? b->base.verify_max : 1u;
    const size_t pi = (m->arch == ARCH_INKLING2 && ivmax > 1u) ? (size_t)ivmax * m->n_head * 64u * 130u : 0u;
    size_t pf = po > pl ? po : pl;
    if (pi > pf) pf = pi;
    if (pf) {
      HIP(H.Malloc((void **)&b->at_part, pf * 4u));
      b->at_part_floats = pf;
    }
  }
  /* IE_GDN_RECORDS=0: a state slot per verify token (ie_gdn1 writes them all). Default: one slot, and a
   * record of each later token of the verify run, which gpu_accept replays on the slot (ie_gdn_replay) */
  b->vmax = b->base.verify_max;
  size_t recbytes = 0;
  if (m->n_rec && b->vmax > 1u && !(gre && gre[0] == '0') && m->sd == 128u) {
    b->ns = 1u;
    recbytes = (size_t)m->n_rec * (b->vmax - 1u) * m->n_vh * GDN_RECF * 4;
    HIP(H.Malloc((void **)&b->rec_buf[0], recbytes + 4));
    HIP(H.Malloc((void **)&b->rec_buf[1], recbytes + 4));
    b->rec_idx = 0;
    b->rec = b->rec_buf[0];
    b->rec_bytes = recbytes;
  }
  const size_t rbytes = (size_t)m->n_rec * GDN_RING * m->conv_dim * 4 + 4,
               sbytes = (size_t)m->n_rec * b->ns * m->n_vh * m->sd * m->sd * 4 + 4;
  HIP(H.Malloc((void **)&b->ring, rbytes));
  if (g->df_feat >= 0) { /* a ring of window + 32 positions per layer (the block: up to 16) */
    b->df_ring = m->df->window + 32u < g->n_ctx + 16u ? m->df->window + 32u : g->n_ctx + 16u;
    const size_t dkv = (size_t)m->df->n_layer * b->df_ring * m->df->n_kv * m->df->hd * 4u;
    HIP(H.Malloc((void **)&b->dfk, dkv));
    HIP(H.Malloc((void **)&b->dfv, dkv));
    fprintf(stderr, "gpu: dflash KV cache %.1f MB\n", 2.0 * dkv / 1e6);
  }
  HIP(H.Malloc((void **)&b->st, sbytes));
  if (m->sc_tot) {
    b->sc_ring_bytes = (size_t)m->sc_tot * 4u * 4u;
    HIP(H.Malloc((void **)&b->sc_ring, b->sc_ring_bytes + 4u));
    HIP(H.Memset(b->sc_ring, 0, b->sc_ring_bytes + 4u));
    if (b->vmax > 1u) {
      b->sc_hist_stride = (m->sc_tot + 3u) & ~3u;
      HIP(H.Malloc((void **)&b->sc_hist, (size_t)b->sc_hist_stride * b->vmax * 4u));
    }
  }
  if (m->n_rec && g->T > 16u && !(getenv("IE_GDN_CHUNK") && getenv("IE_GDN_CHUNK")[0] == '0')) {
    const size_t T = g->T, nb = (T + 31u) / 32u, nv = m->n_vh;
    HIP(H.Malloc((void **)&b->cq_qk, T * m->n_kh * 256u * 4u));
    HIP(H.Malloc((void **)&b->cq_v, T * nv * 128u * 4u));
    HIP(H.Malloc((void **)&b->cq_bg, T * nv * 2u * 4u));
    HIP(H.Malloc((void **)&b->cq_u, T * nv * 128u * 4u));
    HIP(H.Malloc((void **)&b->cq_w, T * nv * 128u * 4u));
    HIP(H.Malloc((void **)&b->cq_m, nb * nv * 32u * 32u * 4u));
    HIP(H.Malloc((void **)&b->cq_g, T * nv * 4u));
  }
  const size_t rt = (size_t)g->n_ctx * (m->n_rot / 2) * 4;
  b->rcos = upload(b, g->rope_cos, rt);
  b->rsin = upload(b, g->rope_sin, rt);
  upload_mat(b, &m->tok, &b->tok0, &b->tok1);

  if (g->ds_feat >= 0 && m->ds) {
    const dspark *d = m->ds;
    if (!b->k[K_DS_DFLARE_MIX_R4] || !b->k[K_DS_BLOCK_INIT_R4] || !b->k[K_DS_ATTN_R4] ||
        !b->k[K_DS_CONCAT_Q8_R4] || !b->k[K_DS_RNN_STEP_R4] || !b->k[K_DS_ADD_Q8_R4])
      ie_die("dspark: required ie_dspark_* kernels missing from ie_inkling_kernels.hsaco");
    if (b->kvq8 || b->kvq4 || b->kvb != 2u) ie_die("dspark: requires f16 KV module (IE_KV_Q8=0, IE_KV_Q4=0)");
    if (d->n_layer > 8u) ie_die("dspark: %u layers, the K/V tables hold 8", d->n_layer);
    const u32 dim = d->dim, qd = d->n_head * d->hd, kvd = d->n_kv * d->hd, r = d->markov_rank, B = d->block;
    size_t dkv_bytes = 0, dkv_full = 0;
    const u32 full_cap = g->n_ctx + 16u;
    b->ds.kv_rowb = (size_t)kvd * 2u;
    b->ds.kv_grow = H.MemAddressReserve && H.MemCreate && H.MemMap && H.MemSetAccess && H.MemUnmap && H.MemRelease &&
                    H.MemAddressFree && H.MemGetAllocationGranularity &&
                    !(getenv("IE_DS_KV_GROW") && getenv("IE_DS_KV_GROW")[0] == '0');
    if (b->ds.kv_grow) {
      hipMemAllocProp prop = {1 /* pinned */, 0, 1 /* device */, 0, NULL, 0, 0, 0};
      size_t gran = 0;
      const size_t want = (getenv("IE_KV_CHUNK_MB") && atoi(getenv("IE_KV_CHUNK_MB")) > 0 ? (size_t)atoi(getenv("IE_KV_CHUNK_MB")) : 4u) << 20;
      if (H.MemGetAllocationGranularity(&gran, &prop, 0) == 0 && gran) {
        b->ds.kv_chunk = (want + gran - 1) / gran * gran;
        b->ds.kv_va = ((size_t)full_cap * b->ds.kv_rowb + b->ds.kv_chunk - 1) / b->ds.kv_chunk * b->ds.kv_chunk;
        b->ds.kv_nchunk = b->ds.kv_va / b->ds.kv_chunk;
        b->ds.kv_step = getenv("IE_KV_GROW_STEP") && atoi(getenv("IE_KV_GROW_STEP")) > 0 ? (u32)atoi(getenv("IE_KV_GROW_STEP")) : 4096u;
      } else
        b->ds.kv_grow = 0;
    }
    for (u32 il = 0; il < d->n_layer; il++) {
      u32 cap = full_cap;
      if (d->l[il].win) {
        u32 rw = 1u;
        while (rw < d->l[il].win + (g->T > 64u ? g->T : 64u)) rw <<= 1u;
        if (rw < cap) cap = rw;
      }
      b->ds.kv_cap[il] = cap;
      const size_t lbytes = (size_t)cap * kvd * 2u;
      if (b->ds.kv_grow && cap == full_cap && H.MemAddressReserve((void **)&b->ds.kc[il], b->ds.kv_va, 0, NULL, 0) == 0) {
        if (H.MemAddressReserve((void **)&b->ds.vc[il], b->ds.kv_va, 0, NULL, 0) == 0) {
          b->ds.kv_vm[il] = 1;
          dkv_full += 2u * lbytes;
          continue;
        }
        H.MemAddressFree(b->ds.kc[il], b->ds.kv_va), b->ds.kc[il] = NULL;
      }
      HIP(H.Malloc((void **)&b->ds.kc[il], lbytes));
      HIP(H.Malloc((void **)&b->ds.vc[il], lbytes));
      dkv_bytes += 2u * lbytes;
    }
    int n_vm = 0;
    for (u32 il = 0; il < d->n_layer; il++) n_vm += b->ds.kv_vm[il];
    if (!n_vm) b->ds.kv_grow = 0;
    if (b->ds.kv_grow) {
      b->ds.kv_h = calloc((size_t)16u * b->ds.kv_nchunk, sizeof(void *));
      if (!b->ds.kv_h) ie_die("out of memory");
      ds_kv_ensure(b, 1u); /* the first step now, with the GPU idle */
    }
    const u32 rn = g->n_ctx + 32u, h2 = d->hd / 2u;
    float *hcos = ie_alloc((size_t)rn * h2 * 4u), *hsin = ie_alloc((size_t)rn * h2 * 4u);
    for (u32 p = 0; p < rn; p++)
      for (u32 j = 0; j < h2; j++) {
        double th = (double)p * pow((double)d->rope_base, -2.0 * (double)j / (double)d->hd);
        hcos[(size_t)p * h2 + j] = (float)cos(th);
        hsin[(size_t)p * h2 + j] = (float)sin(th);
      }
    b->ds.rcos = upload(b, hcos, (size_t)rn * h2 * 4u);
    b->ds.rsin = upload(b, hsin, (size_t)rn * h2 * 4u);
    free(hcos), free(hsin);
    b->ds.fuse_w = upload(b, d->fuse_weight.f, d->fuse_weight.n * 4u);
    b->ds.mask_embed = upload(b, d->mask_embed.f, d->mask_embed.n * 4u);
    b->ds.out_norm = upload(b, d->out_norm.f, d->out_norm.n * 4u);
    b->ds.tok_norm = m->tok_norm.n ? upload(b, m->tok_norm.f, m->tok_norm.n * 4u) : NULL;
    b->ds.mk_dflare_norm = upload(b, d->mk_dflare_norm.f, d->mk_dflare_norm.n * 4u);
    b->ds.mk_rnn_bias = upload(b, d->mk_rnn_bias.f, d->mk_rnn_bias.n * 4u);
    b->ds.mk_hidden_bias = upload(b, d->mk_hidden_bias.f, d->mk_hidden_bias.n * 4u);
    upload_mat(b, &d->mk_rnn_proj, &b->ds.mk_rnn_w0, &b->ds.mk_rnn_w1);
    upload_mat(b, &d->mk_hidden_out, &b->ds.mk_ho_w0, &b->ds.mk_ho_w1);
    upload_mat(b, &d->mk_w1, &b->ds.mk_w1_w0, &b->ds.mk_w1_w1);
    const mat *hmat = m->ds_head.rows ? &m->ds_head : &m->out;
    upload_mat(b, hmat, &b->ds.head_w0, &b->ds.head_w1);
    b->ds.head_rows = hmat->rows;
    b->ds.head_kind = hmat->kind;
    if (hmat->kind != MAT_Q8 && hmat->kind != MAT_Q4K && hmat->kind != MAT_Q6K)
      ie_die("dspark: the output matrix must be Q8_0, Q4_K or Q6_K (kind %d)", hmat->kind);
    for (u32 il = 0; il < d->n_layer; il++) {
      const ds_layer *L = &d->l[il];
      b->ds.dflare_norm[il] = upload(b, L->dflare_norm.f, L->dflare_norm.n * 4u);
      b->ds.attn_norm[il] = upload(b, L->attn_norm.f, L->attn_norm.n * 4u);
      b->ds.ffn_norm[il] = upload(b, L->ffn_norm.f, L->ffn_norm.n * 4u);
      b->ds.q_norm[il] = upload(b, L->q_norm.f, L->q_norm.n * 4u);
      b->ds.k_norm[il] = upload(b, L->k_norm.f, L->k_norm.n * 4u);
      b->ds.k_ctx_norm[il] = upload(b, L->k_ctx_norm.f, L->k_ctx_norm.n * 4u);
      upload_mat(b, &L->wq, &b->ds.wq0[il], &b->ds.wq1[il]);
      upload_mat(b, &L->wk, &b->ds.wk0[il], &b->ds.wk1[il]);
      upload_mat(b, &L->wv, &b->ds.wv0[il], &b->ds.wv1[il]);
      upload_mat(b, &L->wk_ctx, &b->ds.wkc0[il], &b->ds.wkc1[il]);
      upload_mat(b, &L->wv_ctx, &b->ds.wvc0[il], &b->ds.wvc1[il]);
      upload_mat(b, &L->wo, &b->ds.wo0[il], &b->ds.wo1[il]);
      upload_mat(b, &L->wgu, &b->ds.wgu0[il], &b->ds.wgu1[il]);
      upload_mat(b, &L->wdown, &b->ds.wd0[il], &b->ds.wd1[il]);
    }
    b->ds.ctx_q8_str = ((u32)q8_bytes(dim / 32u) + 255u) & ~255u;
    b->ds.ctx_kv_str = (2u * kvd * 4u + 255u) & ~255u;
    b->ds.h_str = dim * 4u;
    b->ds.xq_str = b->ds.ctx_q8_str;
    b->ds.q_str = (qd * 4u + 255u) & ~255u;
    b->ds.kv_str = (kvd * 4u + 255u) & ~255u;
    b->ds.aq_str = ((u32)q8_bytes(qd / 32u) + 255u) & ~255u;
    b->ds.gu_str = (2u * d->ffn * 4u + 255u) & ~255u;
    b->ds.gq_str = ((u32)q8_bytes(d->ffn / 32u) + 255u) & ~255u;
    HIP(H.Malloc((void **)&b->ds.ctx_q8, (size_t)g->T * b->ds.ctx_q8_str));
    HIP(H.Malloc((void **)&b->ds.ctx_kv, (size_t)g->T * b->ds.ctx_kv_str));
    HIP(H.Malloc((void **)&b->ds.frnn, (size_t)dim * 4u));
    HIP(H.Malloc((void **)&b->ds.h, (size_t)B * b->ds.h_str));
    HIP(H.Malloc((void **)&b->ds.xn, (size_t)B * b->ds.h_str));
    HIP(H.Malloc((void **)&b->ds.xq, (size_t)B * b->ds.xq_str));
    HIP(H.Malloc((void **)&b->ds.q, (size_t)B * b->ds.q_str));
    HIP(H.Malloc((void **)&b->ds.k, (size_t)B * b->ds.kv_str));
    HIP(H.Malloc((void **)&b->ds.v, (size_t)B * b->ds.kv_str));
    HIP(H.Malloc((void **)&b->ds.att, (size_t)B * b->ds.q_str));
    HIP(H.Malloc((void **)&b->ds.aq, (size_t)B * b->ds.aq_str));
    HIP(H.Malloc((void **)&b->ds.gu, (size_t)B * b->ds.gu_str));
    HIP(H.Malloc((void **)&b->ds.gq, (size_t)B * b->ds.gq_str));
    HIP(H.Malloc((void **)&b->ds.hidden, (size_t)B * b->ds.h_str));
    HIP(H.Malloc((void **)&b->ds.state, (size_t)r * 4u));
    HIP(H.Malloc((void **)&b->ds.cat_q8, ((size_t)q8_bytes((2u * r + 2u * dim) / 32u) + 255u) & ~255u));
    HIP(H.Malloc((void **)&b->ds.raw768, (size_t)3u * r * 4u));
    HIP(H.Malloc((void **)&b->ds.lat_q8, ((size_t)q8_bytes(r / 32u) + 255u) & ~255u));
    HIP(H.Malloc((void **)&b->ds.delta, (size_t)dim * 4u));
    HIP(H.Malloc((void **)&b->ds.h_q8, b->ds.xq_str));
    HIP(H.Malloc((void **)&b->ds.logits, (size_t)b->ds.head_rows * 4u));
    HIP(H.Malloc((void **)&b->ds.argmax, 8u));
    HIP(H.Malloc(&b->ds.zbuf, 32768u));
    HIP(H.Memset(b->ds.zbuf, 0, 32768u));
    if (H.HostMalloc(&b->ds.zpin, 8192u, 0) != 0) b->ds.zpin = NULL;
    b->ds.qkind = d->l[0].wq.kind;
    b->ds.ready = 1;
    /* Disk prompt cache: include DSpark's full-length global KV layers in kv_rowb (rings go in gpu_snap_io). */
    for (u32 il = 0; il < d->n_layer; il++)
      if (b->ds.kv_cap[il] >= g->n_ctx + 16u) b->base.kv_rowb += 2u * b->ds.kv_rowb;
    if (b->ds.kv_grow)
      fprintf(stderr, "gpu: dspark KV cache %.1f MB mapped now of %.1f MB at the cap (global layers mapped as the context grows: "
              "%u of %u positions, chunks of %zu MiB, steps of %u; IE_DS_KV_GROW=0: one array)\n",
              (dkv_bytes + (size_t)n_vm * 2u * b->ds.kv_mapped) / 1e6, (dkv_bytes + dkv_full) / 1e6, b->ds.kv_rows, full_cap,
              b->ds.kv_chunk >> 20, b->ds.kv_step);
    fprintf(stderr, "gpu: dspark KV cache %.1f MB (caps %u/%u/%u), head %u rows %s, dense %s\n",
            (dkv_bytes + dkv_full) / 1e6, b->ds.kv_cap[0], d->n_layer > 1 ? b->ds.kv_cap[1] : 0u,
            d->n_layer > 2 ? b->ds.kv_cap[2] : 0u, b->ds.head_rows,
            b->ds.head_kind == MAT_Q4K ? "Q4_K" : b->ds.head_kind == MAT_Q6K ? "Q6_K" : "Q8_0",
            b->ds.qkind == MAT_Q4K ? "Q4_K" : "Q8_0");
    const char *sp_env = getenv("IE_DS_SPLIT");
    if (sp_env && sp_env[0] && sp_env[0] != '0') ds_split_setup(b);
    ds_shortlist_setup(b, hsaco);
    const char *ov_env = getenv("IE_DS_OVERLAP");
    const int ov_req = ov_env ? atoi(ov_env) : 0;
    if ((ov_req == 1 || ov_req == 2) && H.StreamCreateWithFlags) {
      if (H.StreamCreateWithFlags(&b->ds.ctx_stream, 1u) == 0 && b->ds.ctx_stream) {
        HIP(H.EventCreate(&b->ds.ev_feat_ready));
        HIP(H.EventCreate(&b->ds.ev_ctx_done));
        b->ds.ov_mode = ov_req;
        fprintf(stderr, "gpu: dspark multi-stream ds_ctx overlap enabled (mode %d)\n", ov_req);
      }
    }
  }

  b->d = calloc(g->n_ops, sizeof(dop));
  for (uint32_t i = 0; i < g->n_ops; i++) {
    const op *o = &g->ops[i];
    if (o->w) upload_mat(b, o->w, &b->d[i].w0, &b->d[i].w1);
    if (o->v && o->w) b->d[i].bias = upload(b, o->v->f, o->v->n * 4); /* GEMV bias */
    else if (o->v) b->d[i].w0 = upload(b, o->v->f, o->v->n * 4);
    if (o->nv) b->d[i].nw = upload(b, o->nv->f, o->nv->n * 4);
    if (o->kind == OP_ATTN && o->sk) b->d[i].w1 = upload(b, o->sk->f, o->sk->n * 4); /* flint: the sinks */
    if (o->kind == OP_GDN) { /* conv weights, dt bias, A, norm weight */
      const layer *L = &m->l[o->layer];
      b->d[i].w0 = upload(b, L->conv.f, L->conv.n * 4);
      b->d[i].w1 = upload(b, L->dt_bias.f, L->dt_bias.n * 4);
      b->d[i].bias = upload(b, L->ssm_a.f, L->ssm_a.n * 4);
      b->d[i].nw = upload(b, L->ssm_norm.f, L->ssm_norm.n * 4);
    }
  }
  hs_setup(b, g);
  HIP(H.EventCreate(&b->e0));
  HIP(H.EventCreate(&b->e1));
  const char *pe = getenv("IE_PROFILE");
  b->prof = pe && pe[0] >= '1' && pe[0] <= '3' ? pe[0] - '0' : 0; /* 1: decode steps; 2: prompt chunks; 3: MTP drafts */
  if (b->prof) {
    b->pev = calloc(b->g->n_ops, sizeof(hipEvent_t));
    b->pms = calloc(b->g->n_ops, sizeof(double));
    if (!b->pev || !b->pms) ie_die("out of memory");
    for (uint32_t i = 0; i < b->g->n_ops; i++) HIP(H.EventCreate(&b->pev[i]));
  }
  const char *te = getenv("IE_TRACE");
  if (te && te[0]) {
    if (!(tr_f = fopen(te, "w"))) ie_die("cannot write %s", te);
    tr_call = tr_n = tr_pos = tr_T = tr_head = tr_pending = tr_dropped = 0;
    tr_reported = tr_invalid = 0;
    fprintf(tr_f, "call,pos,T_run,launch,phase,op,layer,kernel,groups,T,weight_bytes,op_end,commands\n");
    const char *timing = getenv("IE_TRACE_GPU");
    if (timing && !strcmp(timing, "1")) {
      if (!H.EventQuery) ie_die("IE_TRACE_GPU requires hipEventQuery");
      size_t bytes = strlen(te) + sizeof ".gpu.csv";
      char *path = ie_alloc(bytes);
      snprintf(path, bytes, "%s.gpu.csv", te);
      if (!(tr_gpu = fopen(path, "w"))) ie_die("cannot write %s", path);
      free(path);
      tr_events = calloc(TR_EVENT_CAP, sizeof *tr_events);
      if (!tr_events) ie_die("cannot allocate GPU trace events");
      const char *stride = getenv("IE_TRACE_GPU_STRIDE");
      tr_stride = stride ? (uint32_t)strtoul(stride, NULL, 10) : 1u;
      if (!tr_stride) tr_stride = 1u;
      fprintf(tr_gpu, "launch,gpu_us\n");
    }
  }
  if (n_sdev) fprintf(stderr, "gpu: %u arrays (%.2f GB) streamed at load, in %u allocations (IE_SLAB_MB=N: blocks of N MiB)\n", n_sdev, sdev_bytes / 1e9, sdev.n);
  if (n_shost) fprintf(stderr, "gpu: token embedding in pinned host memory (%.2f GB; IE_EMBED_HOST=0: in VRAM)\n", shost_bytes / 1e9);
  if (b->kv_grow == 1)
    fprintf(stderr, "gpu: KV cache mapped as the context grows (chunks of %zu MiB, steps of %u positions; IE_KV_GROW=0: one array of %u positions)\n",
            b->kv_chunk >> 20, b->kv_step, g->n_ctx);
  else if (b->kv_grow == 2)
    fprintf(stderr, "gpu: KV cache in separate arrays per layer (no virtual memory in the runtime)\n");
  fprintf(stderr, "gpu: %u arrays uploaded (%u allocations), arena %u bytes, KV cache %zu bytes, linear state %zu bytes\n", b->nh,
          b->us.n, g->arena, 2 * kv, rbytes + sbytes + 2 * recbytes);
  { /* IE_KV_PREMAP=N: back the first N positions of the KV cache now, with every array uploaded and the GPU idle */
    const char *pm = getenv("IE_KV_PREMAP");
    const long pmn = pm ? atol(pm) : 4096L;
    if (b->kv_grow == 1 && pmn > 1) kv_ensure(b, (u32)(pmn < (long)g->n_ctx ? pmn : (long)g->n_ctx));
  }
  {
    const int sh = !(getenv("IE_SNAP_HOST") && getenv("IE_SNAP_HOST")[0] == '0');
    b->base.snap_in_ram = sh;
  }
  {
    size_t fr = 0, tot = 0; /* what is left for the run: a context that fills the card makes the driver move buffers to system memory */
    if (H.MemGetInfo(&fr, &tot) == 0)
      fprintf(stderr, "gpu: VRAM free after loading: %.0f MB of %.0f MB; %u device allocations\n", fr / 1e6, tot / 1e6, n_dalloc);
  }
  return &b->base;
}

/* debug, IE_DBG_TWICE=NAME: every launch of kernel NAME runs twice (wrong outputs where it updates in place): a
 * verify's time grows by what that kernel costs there, launch gaps included */
static int dbg_twice(int k) {
  static int t = -2;
  if (t == -2) {
    const char *e = getenv("IE_DBG_TWICE");
    t = -1;
    for (int i = 0; e && i < K_N; i++)
      if (!strcmp(kname[i], e)) t = i;
  }
  return k == t;
}
/* IE_FLUSH_EVERY=N: an event recorded on the stream after every N launches (0: none). On Windows the runtime's queue
 * thread keeps all the launches between two event records as one batch, and when the host waits for the end of a run
 * it first waits for the GPU and then retires the batch's commands one at a time, before it reports any of them as
 * done: a decode step of 2,222 launches is reported about 5 ms after the GPU has finished it (hipEventElapsedTime
 * 6.5 ms, host 13 ms). A record ends the batch (HostQueue::loop flushes on the marker), so the queue thread retires
 * the earlier batches while the GPU runs the later ones. Not while a stream captures a graph (cur_stream set). */
static hipEvent_t fl_ev;
static int fl_every = -2; /* -2: uninit, -1: auto (128 in Inkling-2 T=1 gpu_step only), 0: off, >0: explicit global */
static unsigned fl_cnt;
static void fl_mark(const gpu_backend *b) {
  if (fl_every == -2) {
    const char *e = getenv("IE_FLUSH_EVERY");
    if (e && *e) {
      fl_every = atoi(e);
      if (fl_every < 0) fl_every = 0;
      if (fl_every) fprintf(stderr, "gpu: an event record every %d launches (IE_FLUSH_EVERY)\n", fl_every);
    } else {
      fl_every = -1;
    }
  }
  const int lim = (fl_every < 0) ? ((b->in_step && b->m->arch == ARCH_INKLING2) ? 128 : 0) : fl_every;
  if (!lim || cur_stream || ++fl_cnt < (unsigned)lim) return;
  fl_cnt = 0;
  if (!fl_ev) HIP(H.EventCreate(&fl_ev));
  HIP(H.EventRecord(fl_ev, NULL));
}
static uint64_t launch_n; /* kernel launches issued so far */
static void launch_grid(gpu_backend *b, int k, unsigned groups, unsigned T, unsigned lds_bytes, void **args) {
  if (mk_put(k, groups, T, lds_bytes, args)) { tr_mk_add(mk.n); return; }
  tr_event *event = tr_begin();
  HIP(H.ModuleLaunchKernel(b->k[k], groups, T, 1, 256, 1, 1, lds_bytes, cur_stream, args, NULL));
  launch_n++;
  tr_launch(kname[k], groups, T, &tr_op, 1, event);
  if (dbg_twice(k)) {
    event = tr_begin();
    HIP(H.ModuleLaunchKernel(b->k[k], groups, T, 1, 256, 1, 1, lds_bytes, cur_stream, args, NULL));
    launch_n++;
    tr_launch(kname[k], groups, T, &tr_op, 1, event);
  }
  fl_mark(b);
}
static void launch_lds(gpu_backend *b, int k, unsigned groups, unsigned lds_bytes, void **args) {
  launch_grid(b, k, groups, 1, lds_bytes, args);
}
/* groups x T workgroups: grid dimension y is the token (kernels with byte strides) */
static void launch_t(gpu_backend *b, int k, unsigned groups, unsigned T, void **args) {
  launch_grid(b, k, groups, T, 0, args);
}
static void launch(gpu_backend *b, int k, unsigned groups, void **args) { launch_lds(b, k, groups, 0, args); }

/* Tokens per launch of the multi-token GEMVs (ie_gemv_raw_*_tr,
 * ie_gemv_iq4xsq8_tr; up to 8): IE_TR_TMAX, default 8 (one weight read for
 * a verify of up to 8 tokens: 4 drafts on code 13.4 -> 11.5 ms/token vs 4). */
static u32 tr_tmax(void) {
  static u32 n = 0;
  if (!n) {
    const char *e = getenv("IE_TR_TMAX");
    n = e ? (u32)atoi(e) : 8u;
    if (n < 1u || n > 8u) n = 8u;
  }
  return n;
}

/* The raw-block GEMV kernel of a ggml type (one kernel per format). */
static int raw_kernel(u32 type) {
  switch (type) {
    case GGML_Q2_K: return K_GEMV_RAW_Q2K;
    case GGML_IQ2_XXS: return K_GEMV_RAW_IQ2XXS;
    case GGML_IQ2_XS: return K_GEMV_RAW_IQ2XS;
    case GGML_IQ3_XXS: return K_GEMV_RAW_IQ3XXS;
    case GGML_IQ1_S: return K_GEMV_RAW_IQ1S;
    case GGML_IQ2_S: return K_GEMV_RAW_IQ2S;
    case GGML_IQ1_M: return K_GEMV_RAW_IQ1M;
    case GGML_IQ3_S: return K_GEMV_RAW_IQ3S;
    default: ie_die("MAT_RAWQ: no GEMV kernel for ggml type %u", type);
  }
}
/* The multi-token companion (up to 8 tokens per weight block). */
static int raw_kernel_tr(u32 type) {
  return raw_kernel(type) + (K_GEMV_RAW_Q2K_TR - K_GEMV_RAW_Q2K);
}
/* The same for 2 to 4 tokens (a verify of up to 3 drafts): ie_gemv_raw_*_tr4, two rows per wave with the
 * activation loads of the tokens shared by both rows, so rows / 16 workgroups (kernels/ie_kernels.c). Outputs
 * are bitwise those of the _tr kernels. IE_TR4=0: the _tr kernels for every token count. Megakernel runs keep
 * the _tr kernels (the only ones in its sets). */
static int raw_kernel_tr4(u32 type) {
  return raw_kernel(type) + (K_GEMV_RAW_Q2K_TR4 - K_GEMV_RAW_Q2K);
}
/* Multi-token Q4_K / Q6_K GEMVs (verify runs, prompt chunks of up to 16), all bitwise equal to the one-token kernel.
 * The _ts kernels split the tokens evenly over the waves of a row group (each computes ceil(T / waves) tokens), so
 * the waves read the same weights in step, the later ones from the cache.
 *   IE_TS2_MIN (default 2): n to 8 tokens use ie_gemv_*_ts2 (else ie_gemv_*_tr8);
 *   IE_TS16_MIN (default 9): Q4_K, n to 16 tokens use IE_TS16 (absent or 4: ie_gemv_q4kq8_ts4, 4 waves of 4 tokens;
 *   2: _ts16, 2 waves of 8; 0: the _tr kernels); Q6_K uses _ts16 from TS16_MIN (14) tokens (its _ts4 is slower).
 * MiMo-9B verify run at 55 tokens (ms, before -> after; the old 4 + 2 split of 6 tokens read the weights twice):
 * T 2 11.9 -> 10.9, 4 12.9 -> 11.6, 6 16.5 -> 12.2, 7 14.9 -> 12.7, 11 16.2 -> 14.5, 12 16.5 -> 14.7, 16 17.7 -> 17.3. */
#define TS16_MIN 14u
static int ts16_mode(const gpu_backend *b) {
  static int v = -1;
  if (v < 0) v = getenv("IE_TS16") ? atoi(getenv("IE_TS16")) : 4;
  return b->has_ts16 && !mk_light() ? v : 0;
}
static u32 ts2_min(void) {
  static int v = -1;
  if (v < 0) v = getenv("IE_TS2_MIN") ? atoi(getenv("IE_TS2_MIN")) : 2;
  return (u32)v;
}
static u32 ts16_min(void) {
  static int v = -1;
  if (v < 0) v = getenv("IE_TS16_MIN") ? atoi(getenv("IE_TS16_MIN")) : 9;
  return (u32)v;
}
static int tr4_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_TR4") && getenv("IE_TR4")[0] == '0');
  return on;
}
static unsigned tr4_groups(u32 rows) {
  const unsigned g = (rows + 15u) / 16u;
  return g >= 253u && g <= 256u ? 257u : g; /* as gemv_groups */
}
/* The same for 5 to 8 tokens (a verify of up to 7 drafts): ie_gemv_raw_*_tr8, with the token count fixed at compile
 * time and the rows per wave of tr8_rpw (IE_TR8_Rn and TR8_R8 in kernels/ie_kernels.c: the two must agree). Outputs are bitwise those of the _tr kernels. IE_TR8=0: the _tr kernels for 5 to 8 tokens.
 * Megakernel runs keep the _tr kernels. */
static int raw_kernel_tr8(u32 type) {
  return raw_kernel(type) + (K_GEMV_RAW_Q2K_TR8 - K_GEMV_RAW_Q2K);
}
/* the most tokens of the _tr8 kernels: 9, a verify of 8 drafts (MTP_DRAFT_MAX) */
#define TR8_TMAX 9u
static int tr8_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_TR8") && getenv("IE_TR8")[0] == '0');
  return on;
}
static unsigned tr8_rpw(u32 T, u32 type) { /* rows per wave at T tokens (8 tokens: 4 for IQ1_S and IQ1_M, TR8_R8) */
  static const unsigned rpw[10] = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2}; /* 9 tokens (8 drafts): IE_TR8_R9 */
  return T == 8u && (type == GGML_IQ1_S || type == GGML_IQ1_M) ? 4u : rpw[T];
}
static unsigned tr8_groups(u32 rows, u32 T, u32 type) {
  const unsigned g = (rows + 8u * tr8_rpw(T, type) - 1u) / (8u * tr8_rpw(T, type));
  return g >= 253u && g <= 256u ? 257u : g; /* as gemv_groups */
}
/* Several raw GEMVs in one launch (ie_gemv_raw_tr8_m, kernels/ie_kernels.c): up to 3 raw GEMV ops in a row that read
 * the same input (the gate and up of a layer, its q / k / v or qkv / z projections), of any types, for a run of 5 to 8
 * tokens. Each row is computed as in its own launch (the same kernel code and rows per wave), so the outputs are
 * bitwise the same; the launches saved save the GPU's gap between kernels and the tail of each. IE_TR8_MULTI=0: one
 * launch per op. */
/* ie_attn_long_e instead of ie_attn_long for the launches of fewer than 64 workgroups (less than one per CU): the V
 * loads start with the K loads. Same arithmetic. IE_ATTN_LONG_E=0: always ie_attn_long. */
static int asp_on(void) { /* IE_ATTN_SPLIT_P=0: ie_attn_split_t for the verify below ie_attn_long's positions */
  static int on = -1;
  if (on < 0) on = !(getenv("IE_ATTN_SPLIT_P") && getenv("IE_ATTN_SPLIT_P")[0] == '0');
  return on;
}
static int atl_e_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_ATTN_LONG_E") && getenv("IE_ATTN_LONG_E")[0] == '0');
  return on;
}
static int atl_mp_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_ATTN_LONG_MP") && getenv("IE_ATTN_LONG_MP")[0] == '0');
  return on;
}
static int mtp_kv_fast_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_MTP_KV_FAST") && getenv("IE_MTP_KV_FAST")[0] == '0');
  return on;
}
static int async_copy_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_ASYNC_COPY") && getenv("IE_ASYNC_COPY")[0] == '0');
  return on;
}
static int tr8m_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_TR8_MULTI") && getenv("IE_TR8_MULTI")[0] == '0');
  return on;
}
/* IQ4_XS for 5 to 8 tokens: ie_gemv_iq4xsq8_tr8, two rows per wave (rows / 16 workgroups), the activation loads of each
 * token shared by both rows. Outputs are bitwise those of ie_gemv_iq4xsq8_tr. IE_IQ4_TR8=0: ie_gemv_iq4xsq8_tr. */
static int iq4_tr8_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_IQ4_TR8") && getenv("IE_IQ4_TR8")[0] == '0');
  return on;
}
/* The types ie_gemv_raw_tr8_m takes, bit n: GGML type n (all 8 raw types: Q2_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S,
 * IQ2_S, IQ1_M, IQ3_S), as IE_TR8M_TYPES of kernels/ie_kernels.c; IE_TR8M_TYPES=hex: for a kernel file built with others. */
static int tr8m_type(u32 type) {
  static u32 m = 0;
  if (!m) m = getenv("IE_TR8M_TYPES") ? (u32)strtoul(getenv("IE_TR8M_TYPES"), NULL, 16) : 0x206F0400u;
  return type < 32u && (m >> type & 1u);
}
static int tr4m_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_TR4_MULTI") && getenv("IE_TR4_MULTI")[0] == '0');
  return on;
}
static int tr4m_type(u32 type) {
  static u32 m = 0;
  if (!m) m = getenv("IE_TR4M_TYPES") ? (u32)strtoul(getenv("IE_TR4M_TYPES"), NULL, 16) : 0x206F0400u;
  return type < 32u && (m >> type & 1u);
}
static unsigned tr8m_rpw(u32 T) { /* rows per wave of ie_gemv_raw_tr8_m: those of Q2_K (IQ1_S and IQ1_M too at 8 tokens) */
  return tr8_rpw(T, GGML_Q2_K);
}
static int multi_q8_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_MULTI_Q8") && getenv("IE_MULTI_Q8")[0] == '0');
  return on;
}
static int rqt16_min(void) { /* IE_RQ_T16_MIN: fewest verify tokens for ie_gemm_rq_t16* (default 0: off, coalesced _tr16 is faster) */
  static int v = -1;
  if (v < 0) v = getenv("IE_RQ_T16_MIN") ? atoi(getenv("IE_RQ_T16_MIN")) : 0;
  return v;
}
static int rqt16m_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_T16_MULTI") && getenv("IE_T16_MULTI")[0] == '0');
  return on;
}
static int tr16_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_TR16") && getenv("IE_TR16")[0] == '0');
  return on;
}
static int tr16m_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_TR16_MULTI") && getenv("IE_TR16_MULTI")[0] == '0');
  return on;
}
static int raw_kernel_tr16(u32 type) {
  return raw_kernel(type) + (K_GEMV_RAW_Q2K_TR16 - K_GEMV_RAW_Q2K);
}
static unsigned tr16_groups(u32 rows) {
  const unsigned g = (rows + 15u) / 16u;
  return g >= 253u && g <= 256u ? 257u : g;
}
static int rq32_max(void) { /* IE_RQ32_MAX: max tokens for 32-token tile ie_gemm_rq32_* before 64-token ie_gemm_rq_* */
  static int v = -1;
  if (v < 0) v = getenv("IE_RQ32_MAX") ? atoi(getenv("IE_RQ32_MAX")) : 64;
  return v;
}
static int buf_disjoint(const graph *g, int a, int b) {
  if (a < 0 || b < 0) return 1;
  if (a == b) return 0;
  const buf *ba = &g->bufs[a], *bb = &g->bufs[b];
  return ba->off + ba->size <= bb->off || bb->off + bb->size <= ba->off;
}
/* op j may share the launch of ops i .. j - 1: a raw GEMV (or unbiased Q8 GEMV, e.g. GDN beta/alpha) of the same input,
 * which reads nothing they write and writes nowhere they read or write */
static int tr_gemv_joins(const gpu_backend *b, const graph *g, u32 i, u32 j) {
  const op *f = &g->ops[i], *o = &g->ops[j];
  const int kind_ok = o->kind == OP_GEMV_RAW || (o->kind == OP_GEMV_Q8 && !b->d[j].bias && multi_q8_on());
  if (!kind_ok || !o->w || o->a != f->a || o->aoff != f->aoff || o->w->nb != f->w->nb || b->d[j].nw) return 0;
  if (!buf_disjoint(g, o->a, o->b) || (o->res >= 0 && !buf_disjoint(g, o->res, o->b))) return 0;
  for (u32 k = i; k < j; k++) {
    const op *p = &g->ops[k];
    if (!buf_disjoint(g, o->a, p->b)) return 0;
    if (o->res >= 0 && !buf_disjoint(g, o->res, p->b)) return 0;
    if (p->res >= 0 && !buf_disjoint(g, p->res, o->b)) return 0;
    if (o->b == p->b) {
      if (o->boff + o->w->rows * 4u > p->boff && p->boff + p->w->rows * 4u > o->boff) return 0;
    } else if (!buf_disjoint(g, o->b, p->b)) {
      return 0;
    }
  }
  return 1;
}
static int tr4m_joins(const gpu_backend *b, const graph *g, u32 i, u32 j) {
  const op *o = &g->ops[j];
  const int t_ok = o->kind == OP_GEMV_Q8 ? (!b->d[j].bias && multi_q8_on()) : tr4m_type(o->w ? o->w->sub : 32u);
  return t_ok && tr_gemv_joins(b, g, i, j);
}
static u32 raw_tr4_multi(gpu_backend *b, u32 i, u32 i1, u32 T) {
  const graph *g = b->g;
  u32 n = 1;
  if (b->d[i].nw || !tr4m_type(g->ops[i].w->sub)) return 0;
  while (n < 3u && i + n < i1 && tr4m_joins(b, g, i, i + n)) n++;
  if (n < 2u) return 0;
  void *q[3] = {NULL, NULL, NULL}, *y[3] = {NULL, NULL, NULL}, *res[3] = {NULL, NULL, NULL}, *bias[3] = {NULL, NULL, NULL};
  u32 rows[3] = {0, 0, 0}, type[3] = {0, 0, 0}, ys[3] = {0, 0, 0}, rs[3] = {0, 0, 0}, ng[3] = {0, 0, 0};
  for (u32 k = 0; k < n; k++) {
    const op *o = &g->ops[i + k];
    y[k] = b->arena + g->bufs[o->b].off + o->boff, ys[k] = g->bufs[o->b].stride;
    if (o->res >= 0) res[k] = b->arena + g->bufs[o->res].off, rs[k] = g->bufs[o->res].stride;
    rows[k] = o->w->rows;
    if (o->kind == OP_GEMV_Q8) {
      q[k] = b->d[i + k].w1, bias[k] = b->d[i + k].w0, type[k] = 8u;
      ng[k] = ie_gemv_ngroups(rows[k]) * T;
    } else {
      q[k] = b->d[i + k].w1, bias[k] = b->d[i + k].bias, type[k] = o->w->sub;
      ng[k] = (rows[k] + 15u) / 16u;
    }
  }
  const op *f = &g->ops[i];
  void *A = b->arena + g->bufs[f->a].off + f->aoff;
  u32 nb = f->w->nb, xs = g->bufs[f->a].stride, g1 = ng[0], g2 = g1 + ng[1], tot = g2 + ng[2];
  if (tot >= 253u && tot <= 256u) tot = 257u; /* as gemv_groups; the workgroups past the last matrix do nothing */
  void *args[] = {&A, &nb, &T, &xs, &g1, &g2,
                  &q[0], &y[0], &rows[0], &type[0], &ys[0], &res[0], &rs[0], &bias[0],
                  &q[1], &y[1], &rows[1], &type[1], &ys[1], &res[1], &rs[1], &bias[1],
                  &q[2], &y[2], &rows[2], &type[2], &ys[2], &res[2], &rs[2], &bias[2]};
  tr_span(g, i, n);
  launch(b, K_GEMV_RAW_TR4_M, tot, args);
  return n;
}
static int tr8m_joins(const gpu_backend *b, const graph *g, u32 i, u32 j) {
  const op *o = &g->ops[j];
  const int t_ok = o->kind == OP_GEMV_Q8 ? (!b->d[j].bias && multi_q8_on()) : tr8m_type(o->w ? o->w->sub : 32u);
  return t_ok && tr_gemv_joins(b, g, i, j);
}
/* ops i, i + 1, .. (before i1) of a run of T tokens (5 to 9) in one launch if two or more go together: the number
 * of ops launched (0: none, the caller launches op i alone) */
static u32 raw_tr8_multi(gpu_backend *b, u32 i, u32 i1, u32 T) {
  const graph *g = b->g;
  u32 n = 1;
  if (b->d[i].nw || !tr8m_type(g->ops[i].w->sub)) return 0;
  while (n < 3u && i + n < i1 && tr8m_joins(b, g, i, i + n)) n++;
  if (n < 2u) return 0;
  void *q[3] = {NULL, NULL, NULL}, *y[3] = {NULL, NULL, NULL}, *res[3] = {NULL, NULL, NULL}, *bias[3] = {NULL, NULL, NULL};
  u32 rows[3] = {0, 0, 0}, type[3] = {0, 0, 0}, ys[3] = {0, 0, 0}, rs[3] = {0, 0, 0}, ng[3] = {0, 0, 0};
  for (u32 k = 0; k < n; k++) {
    const op *o = &g->ops[i + k];
    y[k] = b->arena + g->bufs[o->b].off + o->boff, ys[k] = g->bufs[o->b].stride;
    if (o->res >= 0) res[k] = b->arena + g->bufs[o->res].off, rs[k] = g->bufs[o->res].stride;
    rows[k] = o->w->rows;
    if (o->kind == OP_GEMV_Q8) {
      q[k] = b->d[i + k].w1, bias[k] = b->d[i + k].w0, type[k] = 8u;
      ng[k] = ie_gemv_ngroups(rows[k]) * T;
    } else {
      q[k] = b->d[i + k].w1, bias[k] = b->d[i + k].bias, type[k] = o->w->sub;
      ng[k] = (rows[k] + 8u * tr8m_rpw(T) - 1u) / (8u * tr8m_rpw(T));
    }
  }
  const op *f = &g->ops[i];
  void *A = b->arena + g->bufs[f->a].off + f->aoff;
  u32 nb = f->w->nb, xs = g->bufs[f->a].stride, g1 = ng[0], g2 = g1 + ng[1], tot = g2 + ng[2];
  if (tot >= 253u && tot <= 256u) tot = 257u; /* as gemv_groups; the workgroups past the last matrix do nothing */
  void *args[] = {&A, &nb, &T, &xs, &g1, &g2,
                  &q[0], &y[0], &rows[0], &type[0], &ys[0], &res[0], &rs[0], &bias[0],
                  &q[1], &y[1], &rows[1], &type[1], &ys[1], &res[1], &rs[1], &bias[1],
                  &q[2], &y[2], &rows[2], &type[2], &ys[2], &res[2], &rs[2], &bias[2]};
  tr_span(g, i, n);
  if (b->k[K_SALUKI_IQ1S_Q8_TR8] && n == 2u && T == 8u && nb == 160u &&
      type[0] == GGML_IQ1_S && rows[0] == 16384u && type[1] == GGML_Q8_0 && rows[1] == 96u) {
    void *pair[] = {&A, &xs, &q[0], &y[0], &ys[0], &res[0], &rs[0], &bias[0],
                            &q[1], &y[1], &ys[1], &res[1], &rs[1], &bias[1]};
    launch(b, K_SALUKI_IQ1S_Q8_TR8, rows[0] / 32u + ng[1], pair);
  } else if (b->k[K_SALUKI_GEMV_RAW_TR8_M_R11] && T == 8u && nb > 0u && (nb & 7u) == 0u) {
    int r11_ok = 1;
    u32 ng9[3] = {0, 0, 0};
    for (u32 k = 0; k < n; k++) {
      if (type[k] == GGML_Q8_0) {
        ng9[k] = ng[k];
      } else {
        const uint64_t bytes = (uint64_t)rows[k] * (uint64_t)(nb >> 3u) * (uint64_t)ie_rawq_bs(type[k]);
        if (bytes > 0x7fffffffULL) { r11_ok = 0; break; }
        ng9[k] = (rows[k] + 31u) / 32u;
      }
    }
    if (r11_ok) {
      u32 g1_9 = ng9[0], g2_9 = g1_9 + ng9[1], tot9 = g2_9 + ng9[2];
      if (tot9 >= 253u && tot9 <= 256u) tot9 = 257u;
      void *args9[] = {&A, &nb, &T, &xs, &g1_9, &g2_9,
                       &q[0], &y[0], &rows[0], &type[0], &ys[0], &res[0], &rs[0], &bias[0],
                       &q[1], &y[1], &rows[1], &type[1], &ys[1], &res[1], &rs[1], &bias[1],
                       &q[2], &y[2], &rows[2], &type[2], &ys[2], &res[2], &rs[2], &bias[2]};
      launch(b, K_SALUKI_GEMV_RAW_TR8_M_R11, tot9, args9);
    } else {
      launch(b, K_GEMV_RAW_TR8_M, tot, args);
    }
  } else launch(b, K_GEMV_RAW_TR8_M, tot, args);
  return n;
}
static u32 raw_tr16_multi(gpu_backend *b, u32 i, u32 i1, u32 T) {
  const graph *g = b->g;
  u32 n = 1;
  if (b->d[i].nw || !tr8m_type(g->ops[i].w->sub)) return 0;
  while (n < 3u && i + n < i1 && tr8m_joins(b, g, i, i + n)) n++;
  if (n < 2u) return 0;
  void *q[3] = {NULL, NULL, NULL}, *y[3] = {NULL, NULL, NULL}, *res[3] = {NULL, NULL, NULL}, *bias[3] = {NULL, NULL, NULL};
  u32 rows[3] = {0, 0, 0}, type[3] = {0, 0, 0}, ys[3] = {0, 0, 0}, rs[3] = {0, 0, 0}, ng[3] = {0, 0, 0};
  for (u32 k = 0; k < n; k++) {
    const op *o = &g->ops[i + k];
    y[k] = b->arena + g->bufs[o->b].off + o->boff, ys[k] = g->bufs[o->b].stride;
    if (o->res >= 0) res[k] = b->arena + g->bufs[o->res].off, rs[k] = g->bufs[o->res].stride;
    rows[k] = o->w->rows;
    if (o->kind == OP_GEMV_Q8) {
      q[k] = b->d[i + k].w1, bias[k] = b->d[i + k].w0, type[k] = 8u;
      ng[k] = ie_gemv_ngroups(rows[k]) * T;
    } else {
      q[k] = b->d[i + k].w1, bias[k] = b->d[i + k].bias, type[k] = o->w->sub;
      ng[k] = (rows[k] + 15u) / 16u;
    }
  }
  const op *f = &g->ops[i];
  void *A = b->arena + g->bufs[f->a].off + f->aoff;
  u32 nb = f->w->nb, xs = g->bufs[f->a].stride, g1 = ng[0], g2 = g1 + ng[1], tot = g2 + ng[2];
  if (tot >= 253u && tot <= 256u) tot = 257u;
  void *args[] = {&A, &nb, &T, &xs, &g1, &g2,
                  &q[0], &y[0], &rows[0], &type[0], &ys[0], &res[0], &rs[0], &bias[0],
                  &q[1], &y[1], &rows[1], &type[1], &ys[1], &res[1], &rs[1], &bias[1],
                  &q[2], &y[2], &rows[2], &type[2], &ys[2], &res[2], &rs[2], &bias[2]};
  tr_span(g, i, n);
  launch(b, K_GEMV_RAW_TR16_M, tot, args);
  return n;
}
static int raw_swiglu_type(u32 type);
static int t16m_joins(const gpu_backend *b, const graph *g, u32 i, u32 j) {
  const op *o = &g->ops[j];
  const int t_ok = o->kind == OP_GEMV_Q8 ? (!b->d[j].bias && multi_q8_on()) : raw_swiglu_type(o->w ? o->w->sub : 32u);
  return t_ok && tr_gemv_joins(b, g, i, j);
}
static u32 raw_t16_multi(gpu_backend *b, u32 i, u32 i1, u32 T) {
  const graph *g = b->g;
  const op *f = &g->ops[i];
  u32 n = 1;
  if (b->d[i].nw || !f->w || f->w->nb < 8u || (f->w->nb & 7u) != 0u || !raw_swiglu_type(f->w->sub)) return 0;
  while (n < 3u && i + n < i1 && t16m_joins(b, g, i, i + n)) n++;
  if (n < 2u) return 0;
  void *q[3] = {NULL, NULL, NULL}, *qs[3] = {NULL, NULL, NULL}, *y[3] = {NULL, NULL, NULL};
  void *res[3] = {NULL, NULL, NULL}, *bias[3] = {NULL, NULL, NULL};
  u32 rows[3] = {0, 0, 0}, type[3] = {0, 0, 0}, ys[3] = {0, 0, 0}, rs[3] = {0, 0, 0}, ng[3] = {0, 0, 0};
  for (u32 k = 0; k < n; k++) {
    const op *o = &g->ops[i + k];
    y[k] = b->arena + g->bufs[o->b].off + o->boff, ys[k] = g->bufs[o->b].stride;
    if (o->res >= 0) res[k] = b->arena + g->bufs[o->res].off, rs[k] = g->bufs[o->res].stride;
    rows[k] = o->w->rows;
    if (o->kind == OP_GEMV_Q8) {
      q[k] = b->d[i + k].w0, qs[k] = b->d[i + k].w1, bias[k] = b->d[i + k].bias, type[k] = 8u;
    } else {
      q[k] = b->d[i + k].w1, qs[k] = NULL, bias[k] = b->d[i + k].bias, type[k] = o->w->sub;
    }
    ng[k] = (rows[k] + 127u) / 128u;
  }
  void *A = b->arena + g->bufs[f->a].off + f->aoff;
  u32 nb = f->w->nb, xs = g->bufs[f->a].stride, g1 = ng[0], g2 = g1 + ng[1], tot = g2 + ng[2];
  void *args[] = {&A, &nb, &T, &xs, &g1, &g2,
                  &q[0], &y[0], &rows[0], &type[0], &ys[0], &res[0], &rs[0], &bias[0],
                  &q[1], &y[1], &rows[1], &type[1], &ys[1], &res[1], &rs[1], &bias[1],
                  &q[2], &y[2], &rows[2], &type[2], &ys[2], &res[2], &rs[2], &bias[2],
                  &qs[0], &qs[1], &qs[2]};
  tr_span(g, i, n);
  launch(b, K_GEMM_RQ_T16_M, tot, args);
  return n;
}
/* Prompt chunks of raw blocks on the matrix cores (ie_gemm_rq_* / ie_gemm_rq32_*, kernels/ie_gemm_raw.inc):
 * for T >= IE_GEMM_MIN tokens (default 17: verify runs stay on TR / rq_t16),
 * T >= IE_RAW_GEMM_MIN and whole 256-weight blocks.
 * IE_RAW_GEMM=0: always the TR kernels. IE_RQ_R (128) rows x IE_RQ_T (64) or 64 x IE_RQ32_T (64) tokens per workgroup. */
#ifndef IE_RQ_R
#define IE_RQ_R 128u
#endif
#ifndef IE_RI_R
#define IE_RI_R 128u
#endif
#ifndef IE_RI_T
#define IE_RI_T 64u /* tokens per workgroup of ie_gemm_ri_* (RI_T of the kernels) */
#endif
#ifndef IE_RQ_T
#define IE_RQ_T 64u
#endif
#ifndef IE_RQ32_T
#define IE_RQ32_T 64u
#endif
static int raw_gemm_ok(u32 T, u32 nb, u32 type) {
  (void)type;
  static int on = -1, gmin = -1, rmin = -1;
  if (on < 0) on = !(getenv("IE_RAW_GEMM") && getenv("IE_RAW_GEMM")[0] == '0');
  if (gmin < 0) gmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
  if (rmin < 0) rmin = getenv("IE_RAW_GEMM_MIN") ? atoi(getenv("IE_RAW_GEMM_MIN")) : 0;
  const int tmin = rmin > 0 ? rmin : 17;
  return on && (int)T >= gmin && (int)T >= tmin && nb >= 8u && nb % 8u == 0u;
}
static int gdump_on(void) {
  static int on = -1;
  if (on < 0) on = getenv("IE_DUMP_GPU_OPS") != NULL && getenv("IE_DUMP_GPU_OPS")[0] != '0';
  return on;
}
static int deadb_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_SKIP_DEAD_B") && getenv("IE_SKIP_DEAD_B")[0] == '0') && !gdump_on();
  return on;
}
static int raw_swiglu_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_RAW_SWIGLU") && getenv("IE_RAW_SWIGLU")[0] == '0');
  return on;
}
static int raw_swiglu_type(u32 type) {
  return type < 32u && (0x206F0400u >> type & 1u);
}
/* Fused raw IQ/K gate + up + SwiGLU + Q8 quantization across decode/verify (ie_gemv_raw_swiglu[|_tr4|_tr8],
 * ie_gemm_rq_t16_swiglu) and prefill chunks (ie_gemm_rq_swiglu): matches both the 2-op concatenated wgu + OP_SWIGLU
 * pattern and the 3-op unconcatenated wgate + wup + OP_SWIGLU pattern. */
static u32 raw_swiglu_fuse(gpu_backend *b, u32 i, u32 i1, u32 T) {
  const graph *g = b->g;
  const op *f = &g->ops[i];
  if (T < 1u || f->kind != OP_GEMV_RAW || b->d[i].nw || b->d[i].bias || f->res >= 0 || f->boff != 0u) return 0;
  u32 nb = f->w->nb;
  if (!nb || nb % 8u != 0u || !raw_swiglu_type(f->w->sub)) return 0;
  static int psw_max = -2;
  if (psw_max == -2) {
    const char *em = getenv("IE_RQ_SWIGLU_MAX"), *es = getenv("IE_RQ_SWIGLU");
    if (es && es[0] == '0') psw_max = 0;
    else if (em) psw_max = atoi(em);
    else if (es && es[0] == '1') psw_max = 65536;
    else psw_max = 0;
  }
  const int is_pf = T > 1u && raw_gemm_ok(T, nb, f->w->sub);
  if (is_pf && (!b->has_rq_swiglu || psw_max <= 0 || (int)T > psw_max)) return 0;
  const int is_t16 = !is_pf && rqt16_min() > 0 && T >= (u32)rqt16_min() && T <= 16u && b->has_rqt16_swiglu;
  const int is_tr16 = !is_pf && !is_t16 && T >= 10u && T <= 16u && b->has_swiglu_tr16 && tr16_on() && !mk.rec;
  if (!is_pf && !is_t16 && !is_tr16) {
    if (!b->has_raw_swiglu || T > TR8_TMAX) return 0;
    if (T >= 2u && (T > tr_tmax() && !(T == TR8_TMAX && tr_tmax() == 8u))) return 0;
    if (T >= 2u && T <= 4u && !tr4_on()) return 0;
    if (T >= 5u && !tr8_on()) return 0;
  }
  const op *sw = NULL;
  void *q0 = NULL, *q1 = NULL;
  u32 type0 = 0, type1 = 0, ffn = 0, n_ops = 0;
  if (i + 1u < i1 && g->ops[i + 1u].kind == OP_SWIGLU) { /* 2-op concatenated wgu + OP_SWIGLU */
    sw = &g->ops[i + 1u];
    ffn = sw->n;
    if (!ffn || ffn % 32u != 0u || f->w->rows != 2u * ffn) return 0;
    if (sw->a != f->b || sw->aoff != 0u || sw->b != f->b || sw->boff != ffn * 4u || sw->qo < 0 || sw->clamp != 0.0f) return 0;
    type0 = type1 = f->w->sub;
    q0 = b->d[i].w1;
    q1 = (char *)b->d[i].w1 + (size_t)ffn * (nb / 8u) * ie_rawq_bs(type0);
    n_ops = 2u;
  } else if (i + 2u < i1 && g->ops[i + 1u].kind == OP_GEMV_RAW && g->ops[i + 2u].kind == OP_SWIGLU) { /* 3-op gate + up + OP_SWIGLU */
    const op *u = &g->ops[i + 1u];
    sw = &g->ops[i + 2u];
    ffn = sw->n;
    if (!ffn || ffn % 32u != 0u || f->w->rows != ffn || u->w->rows != ffn || u->w->nb != nb) return 0;
    if (b->d[i + 1u].nw || b->d[i + 1u].bias || u->res >= 0 || !raw_swiglu_type(u->w->sub)) return 0;
    if ((T > 1u && raw_gemm_ok(T, nb, u->w->sub)) != is_pf) return 0;
    if (u->a != f->a || u->aoff != f->aoff || u->b != f->b || u->boff != ffn * 4u) return 0;
    if (sw->a != f->b || sw->aoff != 0u || sw->b != f->b || sw->boff != ffn * 4u || sw->qo < 0 || sw->clamp != 0.0f) return 0;
    type0 = f->w->sub;
    type1 = u->w->sub;
    q0 = b->d[i].w1;
    q1 = b->d[i + 1u].w1;
    n_ops = 3u;
  } else {
    return 0;
  }
  void *A = b->arena + g->bufs[f->a].off + f->aoff;
  void *Y = (sw->dead_b && deadb_on()) ? NULL : (void *)(b->arena + g->bufs[sw->a].off);
  if (!buf_disjoint(g, f->a, sw->qo)) return 0;
  if (Y && (!buf_disjoint(g, f->a, sw->a) || !buf_disjoint(g, sw->a, sw->qo))) return 0;
  void *OQ = b->arena + g->bufs[sw->qo].off;
  u32 rows = ffn, xs = g->bufs[f->a].stride, ys = g->bufs[sw->a].stride, qs = g->bufs[sw->qo].stride;
  void *args[] = {&q0, &type0, &q1, &type1, &A, &Y, &OQ, &rows, &nb, &T, &xs, &ys, &qs};
  tr_span(g, i, n_ops);
  if (is_pf) {
    launch(b, K_GEMM_RQ_SWIGLU, ((rows + 127u) / 128u) * ((T + 31u) / 32u), args);
  } else if (is_t16) {
    launch(b, K_GEMM_RQ_T16_SWIGLU, (rows + 127u) / 128u, args);
  } else {
    unsigned gr = rows / 32u;
    if (gr >= 253u && gr <= 256u) gr = 257u;
    const int kern = is_tr16 ? K_GEMV_RAW_SWIGLU_TR16
                             : (T == 1u ? K_GEMV_RAW_SWIGLU : T <= 4u ? K_GEMV_RAW_SWIGLU_TR4 : K_GEMV_RAW_SWIGLU_TR8);
    launch(b, kern, gr, args);
  }
  return n_ops;
}
static int q4k_swiglu_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_Q4K_SWIGLU") && getenv("IE_Q4K_SWIGLU")[0] == '0');
  return on;
}
/* Fused MAT_Q4K gate + up + SwiGLU + Q8 quantization for T = 1 .. 16 tokens (e.g. Saluki blk.64 MTP FFN). */
static u32 q4k_swiglu_fuse(gpu_backend *b, u32 i, u32 i1, u32 T) {
  if (!b->has_q4k_swiglu || T < 1u || T > 16u) return 0;
  static int gmin = -1;
  if (gmin < 0) gmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
  if (T > 1u && (int)T >= gmin) return 0;
  const graph *g = b->g;
  const op *f = &g->ops[i];
  if (f->kind != OP_GEMV_Q4K || b->d[i].nw || b->d[i].bias || f->res >= 0 || f->boff != 0u) return 0;
  u32 nb = f->w->nb;
  if (!nb) return 0;
  const op *sw = NULL;
  void *qw0 = NULL, *qs0 = NULL, *qw1 = NULL, *qs1 = NULL;
  u32 rows0 = 0, rows1 = 0, roff1 = 0, ffn = 0, n_ops = 0;
  if (i + 1u < i1 && g->ops[i + 1u].kind == OP_SWIGLU) { /* 2-op concatenated wgu + OP_SWIGLU */
    sw = &g->ops[i + 1u];
    ffn = sw->n;
    if (!ffn || ffn % 32u != 0u || f->w->rows != 2u * ffn) return 0;
    if (sw->a != f->b || sw->aoff != 0u || sw->b != f->b || sw->boff != ffn * 4u || sw->qo < 0 || sw->clamp != 0.0f) return 0;
    qw0 = qw1 = b->d[i].w0;
    qs0 = qs1 = b->d[i].w1;
    rows0 = rows1 = f->w->rows;
    roff1 = ffn;
    n_ops = 2u;
  } else if (i + 2u < i1 && g->ops[i + 1u].kind == OP_GEMV_Q4K && g->ops[i + 2u].kind == OP_SWIGLU) { /* 3-op gate + up + OP_SWIGLU */
    const op *u = &g->ops[i + 1u];
    sw = &g->ops[i + 2u];
    ffn = sw->n;
    if (!ffn || ffn % 32u != 0u || f->w->rows != ffn || u->w->rows != ffn || u->w->nb != nb) return 0;
    if (b->d[i + 1u].nw || b->d[i + 1u].bias || u->res >= 0) return 0;
    if (u->a != f->a || u->aoff != f->aoff || u->b != f->b || u->boff != ffn * 4u) return 0;
    if (sw->a != f->b || sw->aoff != 0u || sw->b != f->b || sw->boff != ffn * 4u || sw->qo < 0 || sw->clamp != 0.0f) return 0;
    qw0 = b->d[i].w0;
    qs0 = b->d[i].w1;
    rows0 = ffn;
    qw1 = b->d[i + 1u].w0;
    qs1 = b->d[i + 1u].w1;
    rows1 = ffn;
    roff1 = 0u;
    n_ops = 3u;
  } else {
    return 0;
  }
  void *A = b->arena + g->bufs[f->a].off + f->aoff;
  void *Y = (sw->dead_b && deadb_on()) ? NULL : (void *)(b->arena + g->bufs[sw->a].off);
  if (!buf_disjoint(g, f->a, sw->qo)) return 0;
  if (Y && (!buf_disjoint(g, f->a, sw->a) || !buf_disjoint(g, sw->a, sw->qo))) return 0;
  void *OQ = b->arena + g->bufs[sw->qo].off;
  u32 rows = ffn, xs = g->bufs[f->a].stride, ys = g->bufs[sw->a].stride, qs = g->bufs[sw->qo].stride;
  void *args[] = {&qw0, &qs0, &rows0, &qw1, &qs1, &rows1, &roff1, &A, &Y, &OQ, &rows, &nb, &T, &xs, &ys, &qs};
  unsigned gr = rows / 32u;
  if (gr >= 253u && gr <= 256u) gr = 257u;
  tr_span(g, i, n_ops);
  launch(b, K_GEMV_Q4K_SWIGLU, gr, args);
  return n_ops;
}
/* Tokens per wave of ie_gemv_q8q8_ts (1 to 4, IE_Q8_TPW; default 1); IE_Q8_TS=0 (0 here): small Q8 GEMVs keep ie_gemv_q8q8_t. */
static unsigned q8ts_tpw(void) {
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("IE_Q8_TS"), *t = getenv("IE_Q8_TPW");
    v = e && e[0] == '0' ? 0 : t ? atoi(t) : 1;
    if (v > 4) v = 4;
    if (v < 0) v = 0;
  }
  return (unsigned)v;
}
static unsigned gemv_groups(u32 rows);
/* IE_RAW_GEMM_CHECK=1 (debug): after each raw GEMM, the same op by the TR kernels into a scratch buffer, and
 * the worst relative difference max |gemm - tr| / max |tr| over all ops so far is printed whenever it grows
 * (=2: every op). The outputs include bias and residual. */
static void raw_gemm_check(gpu_backend *b, void *w1, void *A, void *B, void *bias, void *R, u32 rows, u32 nb, u32 T, u32 xs, u32 ys,
                           u32 rs, u32 type) {
  static void *dev = NULL;
  static size_t cap = 0;
  static double worst = 0;
  static unsigned n = 0;
  const size_t need = (size_t)T * ys;
  if (need > cap) {
    if (dev) H.Free(dev);
    HIP(H.Malloc(&dev, need));
    cap = need;
  }
  void *NW = NULL, *NY = NULL, *NQ = NULL;
  float eps = 0.0f;
  const int rt = raw_kernel_tr(type);
  const unsigned gr = gemv_groups(rows);
  const u32 tmax = tr_tmax();
  for (u32 t0 = 0; t0 < T;) {
    u32 TT = T - t0 < tmax ? T - t0 : tmax;
    void *At = (void *)((uint8_t *)A + (size_t)t0 * xs), *Bt = (void *)((uint8_t *)dev + (size_t)t0 * ys);
    void *Rt = R ? (void *)((uint8_t *)R + (size_t)t0 * rs) : NULL;
    void *ta[] = {&w1, &At, &Bt, &rows, &nb, &TT, &bias, &Rt, &NW, &NY, &NQ, &eps, &b->gm_count, &xs, &ys, &rs};
    launch(b, rt, gr, ta);
    t0 += TT;
  }
  HIP(H.DeviceSynchronize());
  float *y1 = malloc(need), *y2 = malloc(need);
  if (!y1 || !y2) ie_die("out of memory");
  HIP(H.Memcpy(y1, B, need, hipMemcpyDeviceToHost));
  HIP(H.Memcpy(y2, dev, need, hipMemcpyDeviceToHost));
  double mx = 0, df = 0;
  for (u32 t = 0; t < T; t++)
    for (u32 r = 0; r < rows; r++) {
      const double u = y1[(size_t)t * (ys / 4u) + r], v = y2[(size_t)t * (ys / 4u) + r];
      if (fabs(v) > mx) mx = fabs(v);
      if (!(fabs(u - v) <= df)) df = fabs(u - v); /* a NaN counts */
    }
  const double rel = mx > 0 ? df / mx : 0;
  n++;
  const char *e = getenv("IE_RAW_GEMM_CHECK");
  if (rel > worst || (e && e[0] == '2')) {
    if (rel > worst) worst = rel;
    fprintf(stderr, "raw gemm check #%u type %u %ux%u T=%u: max |gemm - tr| / max |tr| = %.2e (worst so far %.2e)\n", n, type, rows,
            nb * 32u, T, rel, worst);
  }
  free(y1), free(y2);
}
/* GEMV grids: on the R9700 (gfx1201, 64 CUs) a GEMV with 253 .. 256
 * workgroups runs 2-3 times slower than with 252 or 257 (measured with
 * tools/gemv_bench.c: 2048 rows x 48 blocks, 17.0 us at 256 workgroups,
 * 6.5 us at 257). The cause is not known. A GEMV grid in that range gets
 * extra workgroups up to 257; they have no row (r >= rows) and do nothing. */
static unsigned gemv_groups(u32 rows) {
  const unsigned g = ie_gemv_ngroups(rows);
  return g >= 253u && g <= 256u ? 257u : g;
}

/* Launch op i for token t of a run: tok is its token id, pos its position.
 * For OP_GDN and, when T > 1, OP_GEMV_Q8, one launch does all T tokens (t
 * = 0). slot: the linear attention state slot to read (ie_gdn). */
/* The row of position p in the KV cache: entry kv_off of attention layer kvi, slot p % kv_len. */
#define KVROW(kvi, p) ((size_t)g->kv_off[(kvi)] + (p) % g->kv_len[(kvi)])
/* The K (v = 0) or V cache of attention layer kvi, from its slot `row` (the layer's own array, or the one for all layers). */
static inline char *kv_ptr(const gpu_backend *b, int v, uint32_t kvi, size_t row) {
  if (b->kv_grow) return (char *)(v ? b->vcl[kvi] : b->kcl[kvi]) + row * b->kvrowb;
  return (char *)(v ? b->vc : b->kc) + ((size_t)b->g->kv_off[kvi] + row) * b->kvrowb;
}
/* IE_KV_GROW: back positions 0 .. need - 1 (and KV_SLACK rows more) of every attention layer with physical memory: whole
 * chunks, a step of kv_step positions at a time. Nothing is copied, the arrays keep their addresses. */
static void kv_ensure(gpu_backend *b, u32 need) {
  const graph *g = b->g;
  if (b->kv_grow != 1 || need <= b->kv_cap) return;
  if (need > g->n_ctx) ie_die("KV cache: %u positions needed, the context is %u", need, g->n_ctx);
  u32 want = need + b->kv_step - 1u;
  want = want / b->kv_step * b->kv_step;
  if (want > g->n_ctx) want = g->n_ctx;
  size_t bytes = (((size_t)want + KV_SLACK) * b->kvrowb + 4 + b->kv_chunk - 1) / b->kv_chunk * b->kv_chunk;
  if (bytes > b->kv_va) bytes = b->kv_va;
  const hipMemAllocProp prop = {1, 0, 1, 0, NULL, 0, 0, 0};
  const hipMemAccDesc acc = {1, 0, 3 /* read and write */};
  const size_t na = b->m->n_kvl ? b->m->n_kvl : 1u;
  if (bytes > b->kv_mapped) {
    for (size_t a = 0; a < na * 2u; a++) {
      if (g->kv_len[a >> 1] < g->n_ctx) continue;
      char *base = (char *)(a & 1u ? b->vcl[a >> 1] : b->kcl[a >> 1]);
      for (size_t off = b->kv_mapped; off < bytes; off += b->kv_chunk) {
        void **h = &b->kv_h[a * b->kv_nchunk + off / b->kv_chunk];
        HIP(H.MemCreate(h, b->kv_chunk, &prop, 0));
        HIP(H.MemMap(base + off, b->kv_chunk, 0, *h, 0));
      }
      HIP(H.MemSetAccess(base + b->kv_mapped, bytes - b->kv_mapped, &acc, 1));
    }
    b->kv_mapped = bytes;
  }
  /* positions the mapped chunks cover with the slack behind them */
  const size_t rows = (bytes - 4) / b->kvrowb;
  b->kv_cap = rows > KV_SLACK ? (u32)(rows - KV_SLACK < g->n_ctx ? rows - KV_SLACK : g->n_ctx) : 0u;
}

/* IE_DS_KV_GROW: back positions 0 .. need - 1 of the drafter's global layers with physical memory, a step of kv_step
 * positions at a time, in whole chunks. Nothing is copied: the arrays keep their addresses. */
static void ds_kv_ensure(gpu_backend *b, u32 need) {
  if (!b->ds.kv_grow || need <= b->ds.kv_rows) return;
  const u32 full_cap = b->g->n_ctx + 16u;
  if (need > full_cap) ie_die("dspark KV cache: %u positions needed, the cap is %u", need, full_cap);
  uint64_t want = ((uint64_t)need + b->ds.kv_step - 1u) / b->ds.kv_step * b->ds.kv_step;
  if (want > full_cap) want = full_cap;
  size_t bytes = ((size_t)want * b->ds.kv_rowb + b->ds.kv_chunk - 1) / b->ds.kv_chunk * b->ds.kv_chunk;
  if (bytes > b->ds.kv_va) bytes = b->ds.kv_va;
  if (bytes > b->ds.kv_mapped) {
    const hipMemAllocProp prop = {1, 0, 1, 0, NULL, 0, 0, 0};
    const hipMemAccDesc acc = {1, 0, 3 /* read and write */};
    for (u32 a = 0; a < 16u; a++) {
      if (!b->ds.kv_vm[a >> 1]) continue;
      char *base = (char *)(a & 1u ? b->ds.vc[a >> 1] : b->ds.kc[a >> 1]);
      for (size_t off = b->ds.kv_mapped; off < bytes; off += b->ds.kv_chunk) {
        void **h = &b->ds.kv_h[a * b->ds.kv_nchunk + off / b->ds.kv_chunk];
        HIP(H.MemCreate(h, b->ds.kv_chunk, &prop, 0));
        HIP(H.MemMap(base + off, b->ds.kv_chunk, 0, *h, 0));
      }
      HIP(H.MemSetAccess(base + b->ds.kv_mapped, bytes - b->ds.kv_mapped, &acc, 1));
    }
    b->ds.kv_mapped = bytes;
    if (getenv("IE_VRAM_TRACE") && getenv("IE_VRAM_TRACE")[0] == '1') fprintf(stderr, "gpu: dspark KV cache mapped to %zu positions (%.1f MB per array)\n",
                           bytes / b->ds.kv_rowb, bytes / 1e6);
  }
  const size_t rows = bytes / b->ds.kv_rowb;
  b->ds.kv_rows = rows < full_cap ? (u32)rows : full_cap;
}

static int kv_shrink_on(void) {
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("IE_KV_SHRINK");
    v = !(e && e[0] == '0');
  }
  return v;
}

static size_t kv_shrink(gpu_backend *b, u32 keep) {
  const graph *g = b->g;
  if (b->kv_grow != 1 || !kv_shrink_on() || !H.MemUnmap || !H.MemRelease) return 0;
  uint64_t want = (uint64_t)(keep > 1u ? keep : 1u) + (uint64_t)b->kv_step;
  want = ((want + b->kv_step - 1u) / b->kv_step) * b->kv_step;
  if (want > g->n_ctx) want = g->n_ctx;
  size_t bytes = (((size_t)want + KV_SLACK) * b->kvrowb + 4 + b->kv_chunk - 1) / b->kv_chunk * b->kv_chunk;
  if (bytes > b->kv_va) bytes = b->kv_va;
  if (bytes >= b->kv_mapped) return 0;
  const size_t na = b->m->n_kvl ? b->m->n_kvl : 1u;
  size_t freed = 0;
  for (size_t a = 0; a < na * 2u; a++) {
    if (g->kv_len[a >> 1] < g->n_ctx) continue;
    char *base = (char *)(a & 1u ? b->vcl[a >> 1] : b->kcl[a >> 1]);
    for (size_t off = bytes; off < b->kv_mapped; off += b->kv_chunk) {
      void **h = &b->kv_h[a * b->kv_nchunk + off / b->kv_chunk];
      HIP(H.MemUnmap(base + off, b->kv_chunk));
      HIP(H.MemRelease(*h));
      *h = NULL;
      freed += b->kv_chunk;
    }
  }
  b->kv_mapped = bytes;
  const size_t rows = bytes > 4 ? (bytes - 4) / b->kvrowb : 0u;
  b->kv_cap = rows > KV_SLACK ? (u32)(rows - KV_SLACK < g->n_ctx ? rows - KV_SLACK : g->n_ctx) : 0u;
  return freed;
}

static size_t ds_kv_shrink(gpu_backend *b, u32 keep) {
  if (!b->ds.ready || !b->ds.kv_grow || !kv_shrink_on() || !H.MemUnmap || !H.MemRelease) return 0;
  const u32 full_cap = b->g->n_ctx + 16u;
  uint64_t want = (uint64_t)(keep > 1u ? keep : 1u) + (uint64_t)b->ds.kv_step;
  want = ((want + b->ds.kv_step - 1u) / b->ds.kv_step) * b->ds.kv_step;
  if (want > full_cap) want = full_cap;
  size_t bytes = ((size_t)want * b->ds.kv_rowb + b->ds.kv_chunk - 1) / b->ds.kv_chunk * b->ds.kv_chunk;
  if (bytes > b->ds.kv_va) bytes = b->ds.kv_va;
  if (bytes >= b->ds.kv_mapped) return 0;
  size_t freed = 0;
  for (u32 a = 0; a < 16u; a++) {
    if (!b->ds.kv_vm[a >> 1]) continue;
    char *base = (char *)(a & 1u ? b->ds.vc[a >> 1] : b->ds.kc[a >> 1]);
    for (size_t off = bytes; off < b->ds.kv_mapped; off += b->ds.kv_chunk) {
      void **h = &b->ds.kv_h[a * b->ds.kv_nchunk + off / b->ds.kv_chunk];
      HIP(H.MemUnmap(base + off, b->ds.kv_chunk));
      HIP(H.MemRelease(*h));
      *h = NULL;
      freed += b->ds.kv_chunk;
    }
  }
  b->ds.kv_mapped = bytes;
  const size_t rows = bytes / b->ds.kv_rowb;
  b->ds.kv_rows = rows < full_cap ? (u32)rows : full_cap;
  return freed;
}

static void gpu_kv_trim(backend *bk, uint32_t keep_pos) {
  gpu_backend *b = (gpu_backend *)bk;
  if (!kv_shrink_on()) return;
  if (b->kv_grow != 1 && !(b->ds.ready && b->ds.kv_grow)) return;
  ds_ctx_drain(b);
  if (b->snap_cs) HIP(H.StreamSynchronize(b->snap_cs));
  HIP(H.StreamSynchronize(cur_stream));
  const size_t f0 = kv_shrink(b, keep_pos);
  const size_t f1 = ds_kv_shrink(b, keep_pos);
  if (f0 + f1) {
    size_t fr = 0, tot = 0;
    if (H.MemGetInfo && H.MemGetInfo(&fr, &tot) == 0)
      fprintf(stderr, "gpu: KV cache shrunk to %u pos (freed %.0f MB VRAM; %.0f MB free)\n",
              b->kv_cap, (double)(f0 + f1) / 1e6, (double)fr / 1e6);
    else
      fprintf(stderr, "gpu: KV cache shrunk to %u pos (freed %.0f MB VRAM)\n",
              b->kv_cap, (double)(f0 + f1) / 1e6);
  }
}

/* The row of token t in a buffer: its own, or the last one if the buffer keeps fewer rows (graph_cap_logits). */
static inline u32 buf_row(const buf *B, u32 t) { return B->rows && t >= B->rows ? B->rows - 1u : t; }

/* IE_NORM_F=0: the RMSNorm op on ie_rmsnorm, not ie_rmsnorm_f (all its loads issued at once). */
static int normf_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_NORM_F") && getenv("IE_NORM_F")[0] == '0');
  return on;
}

static void launch_op(gpu_backend *b, uint32_t i, u32 t, u32 tok, u32 pos, u32 T, u32 slot, u32 wfrom) {
  const model *m = b->m;
  const graph *g = b->g;
  const u32 h2 = m->n_rot / 2, kvd = m->n_kv * m->hd, hd = m->hd, nhd = m->n_head, nkv = m->n_kv, nctx = g->n_ctx;
  (void)slot;
  {
    const op *o = &g->ops[i];
#define TP(id, o_) ((id) >= 0 ? (void *)(b->arena + g->bufs[id].off + (o_) + (size_t)buf_row(&g->bufs[id], t) * g->bufs[id].stride) : NULL)
    void *A = TP(o->a, o->aoff);
    void *B = TP(o->b, o->boff);
    void *C = TP(o->c, o->coff);
    void *NW = b->d[i].nw;
    void *NY = TP(o->nout, 0);
    void *NQ = TP(o->nq, 0);
    float eps = m->eps;
    void *w0 = b->d[i].w0, *w1 = b->d[i].w1, *bias = b->d[i].bias;
    void *Q = TP(o->qo, 0);
    void *R = TP(o->res, 0);
#undef TP
    u32 n = o->n;
    const unsigned elem_groups = (n + 255u) / 256u;
    switch (o->kind) {
      case OP_EMBED: { /* o->w: the MTP head's own table (IE_MTP_EMBED), uploaded as the op's weight */
        const mat *tm = o->w ? o->w : &m->tok;
        void *t0 = o->w ? w0 : b->tok0, *t1 = o->w ? w1 : b->tok1;
        float sc = o->escale > 0.0f ? o->escale : 1.0f; /* ricursive: the embedding scale */
        if (tm->kind == MAT_Q4 || tm->kind == MAT_Q8) {
          u32 nb = tm->nb;
          if (b->ink_graph_cap && T == 1u && tm->kind == MAT_Q8 && b->k[K_INK_EMBED_Q8_CTX_R2] &&
              (o->escale <= 0.0f || o->escale == 1.0f) && nb > 0u) {
            void *ca[] = {&t0, &t1, &A, &nb, &b->ink_ctx_dev};
            launch(b, K_INK_EMBED_Q8_CTX_R2, (nb + 7u) / 8u, ca);
          } else {
            void *args[] = {&A, &t0, &t1, &nb, &tok, &sc};
            launch(b, tm->kind == MAT_Q4 ? K_EMBED_Q4 : K_EMBED_Q8, 1, args);
          }
        } else {
          u32 cols = tm->cols;
          void *args[] = {&A, &t0, &cols, &tok, &sc};
          launch(b, K_EMBED_F32, 1, args);
        }
        break;
      }
      case OP_RMSNORM: {
        u32 xs = g->bufs[o->a].stride, ys = g->bufs[o->b].stride, qs = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
        void *Bout = (o->dead_b && Q && !mk.rec && deadb_on()) ? NULL : B;
        void *args[] = {&A, &w0, &Bout, &n, &eps, &Q, &xs, &ys, &qs};
        static int nmt = -1;
        if (nmt < 0) nmt = !(getenv("IE_NORM_MT") && getenv("IE_NORM_MT")[0] == '0');
        static int ntmin = -1; /* IE_NORM_T_MIN: the fewest tokens for one workgroup per token (fallback when IE_NORM_MT=0) */
        if (ntmin < 0) ntmin = getenv("IE_NORM_T_MIN") ? atoi(getenv("IE_NORM_T_MIN")) : 9;
        if (T >= 1u && nmt && b->k[K_INK_RMSNORM_MT] && mk_ok_k(K_INK_RMSNORM_MT) && (n & 31u) == 0u && n >= 32u &&
            ((uintptr_t)A & 15u) == 0u && (T == 1u || (xs & 15u) == 0u) && ((uintptr_t)w0 & 15u) == 0u &&
            (!Bout || (((uintptr_t)Bout & 15u) == 0u && (T == 1u || (ys & 15u) == 0u))))
          launch_t(b, K_INK_RMSNORM_MT, 1u, T, args);
        else if (T >= 1u && nmt && b->has_norm_mt && !mk.rec && (n & 31u) == 0u && ((uintptr_t)A & 15u) == 0u &&
            (T == 1u || (xs & 15u) == 0u) && ((uintptr_t)w0 & 15u) == 0u &&
            (!Bout || (o->a != o->b && ((uintptr_t)Bout & 15u) == 0u && (T == 1u || (ys & 15u) == 0u))))
          launch_t(b, K_RMSNORM_MT, (n + 1023u) / 1024u, T, args);
        else if ((int)T >= ntmin) launch_t(b, K_RMSNORM_T, 1u, T, args); /* many tokens: one workgroup per token */
        else if (normf_on()) launch_t(b, K_RMSNORM_F, (n + 255u) / 256u, T, args); /* the same, every load in flight */
        else launch_t(b, K_RMSNORM, (n + 255u) / 256u, T, args); /* one block of 32 per wave */
        break;
      }
      case OP_QUANT: {
        u32 nb = n / 32u, xs = g->bufs[o->a].stride, qs = g->bufs[o->b].stride;
        void *args[] = {&A, &B, &nb, &xs, &qs};
        launch_t(b, K_QUANT, (nb + 7u) / 8u, T, args);
        break;
      }
      case OP_GEMV_Q4: {
        u32 rows = o->w->rows, nb = o->w->nb;
        void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &NW, &NY, &NQ, &eps, &b->gm_count};
        launch(b, K_GEMV_Q4, gemv_groups(rows), args);
        break;
      }
      case OP_GEMV_IQ4NL: {
        u32 rows = o->w->rows, nb = o->w->nb;
        if (T > 1u) {
          if (NW) ie_die("IQ4_NL GEMV of several tokens: no fused norm");
          u32 xs = g->bufs[o->a].stride, ys = g->bufs[o->b].stride, rs = o->res >= 0 ? g->bufs[o->res].stride : 0u;
          static int nl_kxmin = -1;
          if (nl_kxmin < 0) nl_kxmin = getenv("IE_IQ4_KX_MIN") ? atoi(getenv("IE_IQ4_KX_MIN")) : 17;
          if (bias == NULL && (int)T >= nl_kxmin && gx_wmma(b, T, nb, (u32)o->w->kind) && !getenv("IE_GEMM_KX_NOD")) {
            void *nul = NULL;
            u32 zero = 0u, one = 1u, kind = (u32)o->w->kind;
            void *ga[] = {&w0, &w1, &A, &B, &rows, &nb, &nul, &nul, &T, &zero, &zero, &xs, &ys, &one, &kind, &R, &rs};
            launch_t(b, K_GEMM_KX, (rows + 127u) / 128u, (T + 63u) / 64u, ga);
          } else {
            for (u32 t0 = 0; t0 < T; t0 += 8u) {
              u32 TT = T - t0 < 8u ? T - t0 : 8u;
              void *At = (uint8_t *)A + (size_t)t0 * xs, *Bt = (uint8_t *)B + (size_t)t0 * ys;
              void *Rt = R ? (void *)((uint8_t *)R + (size_t)t0 * rs) : NULL;
              void *ta[] = {&w0, &w1, &At, &Bt, &rows, &nb, &TT, &bias, &Rt, &xs, &ys, &rs};
              launch(b, K_GEMV_IQ4NL_TR, gemv_groups(rows), ta);
            }
          }
          break;
        }
        void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &NW, &NY, &NQ, &eps, &b->gm_count};
        launch(b, K_GEMV_IQ4NL, gemv_groups(rows), args);
        break;
      }
      case OP_GEMV_Q6K: {
        u32 rows = o->w->rows, nb = o->w->nb;
        if (T > 1) { /* all tokens: 8 at a time (8 rows per wave) or 16 at a time */
          if (NW) ie_die("a GEMV with a fused norm (IE_NORM_FUSE) cannot do several tokens");
          u32 xs = g->bufs[o->a].stride, ys = g->bufs[o->b].stride / 4u, rs = o->res >= 0 ? g->bufs[o->res].stride / 4u : 0u;
          static int gmin6 = -1;
          if (gmin6 < 0) gmin6 = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
          if ((int)T >= gmin6 && nb % 8u == 0u) { /* WMMA */
            void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &T, &xs, &ys, &rs};
            launch(b, K_GEMM_Q6KR, ((rows + 127u) / 128u) * ((T + 63u) / 64u), args); /* GK_R x GK_T */
            break;
          }
          const u32 step = T <= 8u && !mk_light() ? 8u : 16u; /* <= 8 tokens (verify): 8 rows per wave */
          for (u32 t0 = 0; t0 < T; t0 += step) {
            u32 tn = T - t0 < step ? T - t0 : step;
            void *Ab = (char *)A + (size_t)t0 * xs, *Bb = (char *)B + (size_t)t0 * ys * 4u;
            const void *Rb = R ? (const char *)R + (size_t)t0 * rs * 4u : NULL;
            void *ta[] = {&w0, &w1, &Ab, &Bb, &rows, &nb, &bias, &Rb, &tn, &xs, &ys, &rs};
            const u32 g8 = step == 8u ? (rows + 63u) / 64u : (rows + 31u) / 32u;
            const int t16 = step == 16u && tn >= TS16_MIN ? ts16_mode(b) : 0;
            if (step == 8u && tn >= ts2_min()) { /* the tokens split over 2 waves: more waves in flight */
              const u32 g2 = (rows + 31u) / 32u;
              launch(b, K_GEMV_Q6K_TS2, g2 >= 253u && g2 <= 256u ? 257u : g2, ta);
            } else if (t16 == 2 || t16 == 4) { /* 14 to 16 tokens: 8 rows per wave, the tokens split over 2 waves */
              const u32 gs = (rows + 31u) / 32u;
              launch(b, K_GEMV_Q6K_TS16, gs >= 253u && gs <= 256u ? 257u : gs, ta);
            } else
              launch(b, step == 8u ? K_GEMV_Q6K_TR8 : K_GEMV_Q6K_TR, g8 >= 253u && g8 <= 256u ? 257u : g8, ta);
          }
          break;
        }
        void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &NW, &NY, &NQ, &eps, &b->gm_count};
        launch(b, K_GEMV_Q6K, gemv_groups(rows), args);
        break;
      }
      case OP_GEMV_Q4K: {
        u32 rows = o->w->rows, nb = o->w->nb;
        if (T > 1) { /* all tokens, 16 at a time: the weights read once per 16 tokens */
          if (NW) ie_die("a GEMV with a fused norm (IE_NORM_FUSE) cannot do several tokens");
          u32 xs = g->bufs[o->a].stride, ys = g->bufs[o->b].stride / 4u, rs = o->res >= 0 ? g->bufs[o->res].stride / 4u : 0u;
          static int gmin = -1;
          if (gmin < 0) gmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
          if ((int)T >= gmin && nb % 8u == 0u) { /* WMMA */
            /* IE_GEMM_H=1: ie_gemm_q4k_h (fp16 activations, the weights dequantized to fp16). Off by default:
             * not faster than ie_gemm_q4kr (512 tokens: 34816 x 5120 1.982 vs 1.957 ms; 5120 x 17408 1.269 vs 1.006) */
            static int gh = -1;
            if (gh < 0) gh = getenv("IE_GEMM_H") && getenv("IE_GEMM_H")[0] == '1';
            if (gh && b->xh && rows % 64u == 0u && (size_t)T * nb * 64u <= b->xh_bytes) {
              void *X = b->xh;
              u32 hsb = nb * 64u, hs = nb * 32u; /* fp16 row: bytes, halves */
              void *ca[] = {&A, &X, &nb, &xs, &hsb};
              launch_t(b, K_Q8_F16, (nb * 32u + 255u) / 256u, T, ca);
              void *ga[] = {&w0, &w1, &X, &B, &rows, &nb, &bias, &R, &T, &hs, &ys, &rs};
              static int ht = -1; /* IE_GEMM_HT=256|512: tokens per workgroup (default: by the shape) */
              if (ht < 0) ht = getenv("IE_GEMM_HT") ? atoi(getenv("IE_GEMM_HT")) : 0;
              const int t256 = ht ? ht == 256 : rows < 16384u;
              if (t256) launch(b, K_GEMM_Q4K_H256, (rows / 64u) * ((T + 255u) / 256u), ga);
              else launch(b, K_GEMM_Q4K_H, (rows / 64u) * ((T + 511u) / 512u), ga);
              break;
            }
            static int sb = -1; /* IE_Q4K_SB=1: activations requantized with one scale per 256 (ie_gemm_q4kr_sb) */
            if (sb < 0) sb = getenv("IE_Q4K_SB") && getenv("IE_Q4K_SB")[0] == '1';
            if (sb && b->rq && (size_t)T * xs <= b->rq_bytes) {
              void *Q = b->rq;
              void *ra[] = {&A, &Q, &nb, &xs, &xs};
              launch_t(b, K_REQUANT_SB, (nb / 8u + 7u) / 8u, T, ra);
              void *args[] = {&w0, &w1, &Q, &B, &rows, &nb, &bias, &R, &T, &xs, &ys, &rs};
              launch(b, K_GEMM_Q4KR_SB, ((rows + 127u) / 128u) * ((T + 63u) / 64u), args);
              break;
            }
            void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &T, &xs, &ys, &rs};
            launch(b, K_GEMM_Q4KR, ((rows + 127u) / 128u) * ((T + 63u) / 64u), args); /* GK_R x GK_T */
            break;
          }
          const u32 gr = (rows + 31u) / 32u;
          for (u32 t0 = 0; t0 < T; t0 += 16u) {
            u32 tn = T - t0 < 16u ? T - t0 : 16u;
            void *Ab = (char *)A + (size_t)t0 * xs, *Bb = (char *)B + (size_t)t0 * ys * 4u;
            const void *Rb = R ? (const char *)R + (size_t)t0 * rs * 4u : NULL;
            void *ta[] = {&w0, &w1, &Ab, &Bb, &rows, &nb, &bias, &Rb, &tn, &xs, &ys, &rs};
            if (mk_light()) { /* the megakernel: 4 rows per wave (fewer registers) */
              launch(b, K_GEMV_Q4K_TR, gr >= 253u && gr <= 256u ? 257u : gr, ta);
            } else if (b->k[K_INK_GEMV_Q4K_HEAD_VER_R7] && !mk.rec && tn >= 2u && tn <= 8u &&
                       rows > 0u && nb > 0u && (nb & 7u) == 0u) {
              u32 unpad = 0u;
              float cap = 0.0f;
              void *qa[] = {&w0, &w1, &Ab, &Bb, &rows, &nb, &bias, &Rb, &tn, &xs, &ys, &rs, &unpad, &cap};
              launch(b, K_INK_GEMV_Q4K_HEAD_VER_R7, gr >= 253u && gr <= 256u ? 257u : gr, qa);
            } else if (tn >= ts2_min() && tn <= 8u) { /* 8 rows per wave, the tokens split over 2 waves: more waves in flight */
              launch(b, K_GEMV_Q4K_TS2, gr >= 253u && gr <= 256u ? 257u : gr, ta);
            } else if (tn <= 8u) { /* 8 rows per wave: half the activation loads per weight byte */
              const u32 g8 = (rows + 63u) / 64u;
              launch(b, K_GEMV_Q4K_TR8, g8 >= 253u && g8 <= 256u ? 257u : g8, ta);
            } else if (tn >= ts16_min() && (ts16_mode(b) == 2 || ts16_mode(b) == 4)) { /* 8 rows per wave, tokens split */
              const int t4 = ts16_mode(b) == 4;
              const u32 gs = t4 ? (rows + 15u) / 16u : (rows + 31u) / 32u;
              launch(b, t4 ? K_GEMV_Q4K_TS4 : K_GEMV_Q4K_TS16, gs >= 253u && gs <= 256u ? 257u : gs, ta);
            } else {
              launch(b, K_GEMV_Q4K_TR, gr >= 253u && gr <= 256u ? 257u : gr, ta);
            }
          }
          break;
        }
        void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &NW, &NY, &NQ, &eps, &b->gm_count};
        launch(b, K_GEMV_Q4K, gemv_groups(rows), args);
        break;
      }
      case OP_GEMV_Q8: {
        u32 rows = o->w->rows, nb = o->w->nb;
        if (T > 1) { /* all tokens, the weights read once */
          if (NW) ie_die("a GEMV with a fused norm (IE_NORM_FUSE) cannot do several tokens");
          u32 xs = g->bufs[o->a].stride, ys = g->bufs[o->b].stride / 4u, rs = o->res >= 0 ? g->bufs[o->res].stride / 4u : 0u;
          void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &T, &xs, &ys, &rs};
          /* many tokens (prompt chunks): 4 rows per wave, 4 times less
           * activation traffic; few tokens (verify): one row per wave, more
           * workgroups. The same results either way. */
          static int trmin = -1, gmin = -1;
          if (trmin < 0) trmin = getenv("IE_TR_MIN") ? atoi(getenv("IE_TR_MIN")) : 2;
          if (gmin < 0) gmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
          /* ie_gemm_q8r (weights straight into registers, 256 rows per
           * workgroup) when nb % 4 == 0; else ie_gemm_q8 (128 rows).
           * IE_GEMM=old: always ie_gemm_q8; IE_GEMM=h: the fp16 WMMA kernel. */
          static int gsel = -1;
          if (gsel < 0) gsel = !getenv("IE_GEMM") ? 0 : getenv("IE_GEMM")[0] == 'h' ? 1 : 2;
          const int gk = gsel == 1 ? K_GEMM_H : gsel == 0 && nb % 4u == 0u ? K_GEMM_Q8R : K_GEMM_Q8;
          const u32 grows = gk == K_GEMM_Q8R ? 256u : 128u;
          if ((int)T >= gmin) { /* many tokens: the matrix instruction (WMMA) */
            if (b->k[K_INK_GEMM_Q8_NB2] && !mk.rec && rows > 0u && nb == 2u) {
              launch(b, K_INK_GEMM_Q8_NB2, ((rows + 127u) / 128u) * ((T + 63u) / 64u), args);
            } else if (gk == K_GEMM_Q8R && b->k[K_INK_GEMM_Q8_SMALL_R7] && !mk.rec &&
                rows > 0u && rows <= 512u && nb >= 4u && (nb & 3u) == 0u &&
                getenv("IE_INK_Q8_SMALL_R7") && getenv("IE_INK_Q8_SMALL_R7")[0] == '1') {
              const u32 gq = ((rows + 31u) / 32u) * ((T + 63u) / 64u);
              launch(b, K_INK_GEMM_Q8_SMALL_R7, gq >= 253u && gq <= 256u ? 257u : gq, args);
            } else if (gk == K_GEMM_Q8R && (b->k[K_INK_GEMM_Q8_64X64_R9] || b->k[K_INK_GEMM_Q8_64X64_R7]) && !mk.rec &&
                rows > 512u && rows <= 1024u && nb >= 4u && (nb & 3u) == 0u) {
              const u32 gq = ((rows + 63u) / 64u) * ((T + 63u) / 64u);
              const int k64 = b->k[K_INK_GEMM_Q8_64X64_R9] ? K_INK_GEMM_Q8_64X64_R9 : K_INK_GEMM_Q8_64X64_R7;
              launch(b, k64, gq >= 253u && gq <= 256u ? 257u : gq, args);
            } else if (gk == K_GEMM_Q8R && b->k[K_INK_GEMM_Q8_SMALL] && !mk.rec &&
                rows > 0u && rows <= 384u && nb >= 4u && (nb & 3u) == 0u && nb <= 65536u && xs <= 0x0FFFFFFFu) {
              const u32 gq = ((rows + 15u) / 16u) * ((T + 15u) / 16u);
              launch(b, K_INK_GEMM_Q8_SMALL, gq >= 253u && gq <= 256u ? 257u : gq, args);
            } else if (gk == K_GEMM_Q8R && (b->k[K_INK_GEMM_Q8_MED_R9] || b->k[K_INK_GEMM_Q8_MED_R7]) && !mk.rec &&
                rows >= 512u && nb >= 4u && (nb & 3u) == 0u) {
              const int km = b->k[K_INK_GEMM_Q8_MED_R9] ? K_INK_GEMM_Q8_MED_R9 : K_INK_GEMM_Q8_MED_R7;
              launch(b, km, ((rows + 127u) / 128u) * ((T + 63u) / 64u), args);
            } else if (gk == K_GEMM_Q8R && b->k[K_INK_GEMM_Q8_MED_R2] && !mk.rec &&
                rows >= 512u && nb >= 4u && (nb & 3u) == 0u) {
              launch(b, K_INK_GEMM_Q8_MED_R2, ((rows + 127u) / 128u) * ((T + 63u) / 64u), args);
            } else if (gk == K_GEMM_Q8R && b->k[K_SALUKI_GEMM_Q8R_R11] && !mk.rec &&
                rows <= 128u && nb >= 8u && (nb & 7u) == 0u && nb <= 65536u && xs <= 0x0FFFFFFFu) {
              const u32 gq = ((rows + 15u) / 16u) * ((T + 15u) / 16u);
              launch(b, K_SALUKI_GEMM_Q8R_R11, gq >= 253u && gq <= 256u ? 257u : gq, args);
            } else {
              launch(b, gk, ((rows + grows - 1u) / grows) * ((T + 63u) / 64u), args);
            }
            if (getenv("IE_GEMM_CHECK") && !bias && !R) { /* debug: compare with ie_gemv_q8q8_t, 16 tokens at a time (not with bias/residual) */
              static float *dev = NULL;
              static size_t cap = 0;
              const size_t need = (size_t)T * ys * 4;
              if (need > cap) { if (dev) H.Free(dev); HIP(H.Malloc((void **)&dev, need)); cap = need; }
              void *nul = NULL;
              u32 one = 1;
              (void)one;
              float *y1 = malloc(need), *y2 = malloc(need);
              void *ga[] = {&w0, &w1, &A, &dev, &rows, &nb, &nul, &nul, &T, &xs, &ys, &rs};
              launch(b, gk, ((rows + grows - 1u) / grows) * ((T + 63u) / 64u), ga);
              for (u32 t0 = 0; t0 < T; t0 += 16) {
                u32 tn = T - t0 < 16 ? T - t0 : 16;
                char *Ab = (char *)A + (size_t)t0 * xs;
                float *Bb = (float *)B + (size_t)t0 * ys;
                void *ta[] = {&w0, &w1, &Ab, &Bb, &rows, &nb, &nul, &nul, &tn, &xs, &ys, &rs};
                launch(b, K_GEMV_Q8_T, gemv_groups(rows), ta);
              }
              HIP(H.Memcpy(y1, dev, need, hipMemcpyDeviceToHost));
              HIP(H.Memcpy(y2, B, need, hipMemcpyDeviceToHost));
              double mx = 0, d = 0;
              for (u32 t = 0; t < T; t++)
                for (u32 r = 0; r < rows; r++) {
                  double u = y1[(size_t)t * ys + r], v = y2[(size_t)t * ys + r];
                  if (fabs(v) > mx) mx = fabs(v);
                  if (fabs(u - v) > d) d = fabs(u - v);
                }
              fprintf(stderr, "gemm check %ux%u T=%u: max |gemm - gemv| / max |gemv| = %.2e\n", rows, o->w->cols, T, mx > 0 ? d / mx : 0);
              free(y1), free(y2);
            }
          } else if (b->k[K_INK_GEMV_Q8_BW_R7] && !mk.rec && T >= 2u && T <= 16u &&
                     rows >= 256u && nb >= 8u && (nb & 7u) == 0u) {
            const u32 gr = (rows + 31u) / 32u;
            launch(b, K_INK_GEMV_Q8_BW_R7, gr >= 253u && gr <= 256u ? 257u : gr, args);
          } else if ((int)T >= trmin && rows >= 256u) { /* fewer rows (Qwen3.5's 96-row beta/alpha): more workgroups below */
            static unsigned rpw = 0; /* rows per wave of ie_gemv_q8q8_tr (GEMV_R) */
            if (!rpw) rpw = getenv("IE_TR_R") ? (unsigned)atoi(getenv("IE_TR_R")) : 4u;
            unsigned gr = (rows + 8u * rpw - 1u) / (8u * rpw);
            launch(b, K_GEMV_Q8_TR, gr >= 253u && gr <= 256u ? 257u : gr, args);
          } else if (b->k[K_INK_GEMV_Q8_SMALL_VER_R5] && mk_ok_k(K_INK_GEMV_Q8_SMALL_VER_R5) &&
                     rows > 0u && rows <= 128u && T >= 2u && T <= 16u) {
            launch(b, K_INK_GEMV_Q8_SMALL_VER_R5, rows, args);
          } else if (b->has_q8ts && rows < 256u && T >= 2u && T <= 16u && q8ts_tpw()) {
            /* few rows (the 96-row beta/alpha): the tokens split over workgroups too, ie_gemv_q8q8_ts */
            unsigned tpw = q8ts_tpw(), gr = ie_gemv_ngroups(rows); const unsigned tot = gr * ((T + tpw - 1u) / tpw);
            void *ta[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &T, &xs, &ys, &rs, &gr, &tpw};
            launch(b, K_GEMV_Q8_TS, tot >= 253u && tot <= 256u ? 257u : tot, ta);
          } else {
            launch(b, K_GEMV_Q8_T, gemv_groups(rows), args);
          }
          break;
        }
        void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &NW, &NY, &NQ, &eps, &b->gm_count};
        static int q8_bw_t1 = -1;
        if (q8_bw_t1 < 0) q8_bw_t1 = getenv("IE_INK_Q8_BW_T1") ? atoi(getenv("IE_INK_Q8_BW_T1")) : 1;
        if (b->k[K_INK_GEMV_Q8_NB2] && mk_ok_k(K_INK_GEMV_Q8_NB2) && rows > 0u && nb == 2u) {
          launch(b, K_INK_GEMV_Q8_NB2, (rows + 63u) / 64u, args);
        } else if (b->k[K_INK_GEMV_Q8_SMALL] && mk_ok_k(K_INK_GEMV_Q8_SMALL) && rows > 0u && rows <= 64u && nb >= 32u) {
          launch(b, K_INK_GEMV_Q8_SMALL, rows, args);
        } else if (q8_bw_t1 && b->k[K_INK_GEMV_Q8_BW_R7] && !mk.rec && !NW &&
                   rows >= 128u && rows <= 8192u && nb >= 8u && (nb & 7u) == 0u) {
          u32 one = 1u, z0 = 0u;
          void *ba[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &one, &z0, &z0, &z0};
          const u32 gr = (rows + 31u) / 32u;
          launch(b, K_INK_GEMV_Q8_BW_R7, gr >= 253u && gr <= 256u ? 257u : gr, ba);
        } else if (b->k[K_INK_GEMV_Q8_MED_R2] && mk_ok_k(K_INK_GEMV_Q8_MED_R2) && rows >= 128u && rows <= 8192u && nb >= 32u && (nb & 7u) == 0u) {
          launch(b, K_INK_GEMV_Q8_MED_R2, (rows + 31u) / 32u, args);
        } else {
          launch(b, K_GEMV_Q8, gemv_groups(rows), args);
        }
        break;
      }
      case OP_GEMV_F32: {
        u32 rows = o->w->rows, cols = o->w->cols;
        static int gmin = -1, wave32 = -1;
        if (gmin < 0) gmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
        if (wave32 < 0) {
          const char *e = getenv("IE_LORA_WAVE32");
          wave32 = e && !strcmp(e, "1");
        }
        /* With 32 columns the usual workgroup reduction has seven empty
         * waves. The wave kernel uses the same 32-lane reduction. */
        const int narrow = wave32 && cols == 32u && !NW && (T == 1 || (int)T < gmin);
        if (T > 1 || narrow) { /* all tokens in one launch (grid y) */
          if (NW) ie_die("a GEMV with a fused norm (IE_NORM_FUSE) cannot do several tokens");
          u32 xs = g->bufs[o->a].stride, ys = g->bufs[o->b].stride, rs = o->res >= 0 ? g->bufs[o->res].stride : 0u;
          void *ta[] = {&w0, &A, &B, &rows, &cols, &bias, &R, &xs, &ys, &rs};
          static int fs = -1; /* IE_GEMM_F32_S=0: off */
          if (fs < 0) fs = !(getenv("IE_GEMM_F32_S") && getenv("IE_GEMM_F32_S")[0] == '0');
          if (!narrow && (int)T >= gmin && fs && rows <= 1024u) { /* small matrix: 8 rows x 8 tokens per workgroup */
            if (wave32 && b->k[K_SALUKI_LORA_B_R11] && !mk.rec &&
                cols == 32u && rows == 1024u && R && !bias && (((uintptr_t)w0) & 15u) == 0u) {
              unsigned gx = (rows + 31u) / 32u;
              if (gx >= 253u && gx <= 256u) gx = 257u;
              u32 t_tile = 32u;
              void *la[] = {&w0, &A, &B, &R, &rows, &T, &xs, &ys, &rs, &t_tile};
              launch_t(b, K_SALUKI_LORA_B_R11, gx, (T + t_tile - 1u) / t_tile, la);
            } else if (wave32 && b->k[K_SALUKI_LORA_B_T32] && b->k[K_SALUKI_LORA_A_T4] && !mk.rec &&
                cols == 32u && rows == 1024u && R && !bias) {
              unsigned gx = (rows + 31u) / 32u;
              if (gx >= 253u && gx <= 256u) gx = 257u;
              void *la[] = {&w0, &A, &B, &R, &rows, &T, &xs, &ys, &rs};
              launch_t(b, K_SALUKI_LORA_B_T32, gx, (T + 31u) / 32u, la);
            } else if (b->k[K_SALUKI_LORA_A_T1] && !mk.rec && rows == 32u && T <= 96u &&
                       cols >= 1024u && (cols % 1024u) == 0u && !bias && !R &&
                       (xs & 15u) == 0u && (((uintptr_t)A) & 15u) == 0u) {
              u32 TT = T;
              void *sa[] = {&w0, &A, &B, &rows, &cols, &xs, &ys, &TT};
              launch(b, K_SALUKI_LORA_A_T1, T >= 253u && T <= 256u ? 257u : T, sa);
            } else if (b->k[K_SALUKI_LORA_A_T4] && !mk.rec && rows == 32u &&
                       cols >= 256u && (cols % 256u) == 0u && !bias && !R &&
                       (xs & 15u) == 0u && (((uintptr_t)A) & 15u) == 0u) {
              unsigned gx = (T + 3u) / 4u;
              if (gx >= 253u && gx <= 256u) gx = 257u;
              u32 TT = T;
              void *sa[] = {&w0, &A, &B, &rows, &cols, &xs, &ys, &TT};
              launch(b, K_SALUKI_LORA_A_T4, gx, sa);
            } else if (b->k[K_INK_ROUTER_ROUTE_PF_R7] && !mk.rec && rows > 0u && cols > 0u && (cols & 3u) == 0u &&
                       (xs & 15u) == 0u && (((uintptr_t)w0) & 15u) == 0u && (((uintptr_t)A) & 15u) == 0u) {
              u32 TT = T, K0 = 0u, ost0 = 0u;
              float sc0 = 0.0f;
              void *nul = NULL;
              void *sa[] = {&w0, &A, &B, &rows, &cols, &bias, &R, &xs, &ys, &rs, &TT, &K0, &nul, &sc0, &nul, &ost0, &nul};
              launch_t(b, K_INK_ROUTER_ROUTE_PF_R7, (rows + 31u) / 32u, (T + 63u) / 64u, sa);
            } else if (b->k[K_INK_GEMM_F32_ROUTER_R3] && !mk.rec && rows > 0u && cols > 0u && (cols & 3u) == 0u &&
                       (xs & 15u) == 0u && (((uintptr_t)w0) & 15u) == 0u && (((uintptr_t)A) & 15u) == 0u) {
              u32 TT = T;
              void *sa[] = {&w0, &A, &B, &rows, &cols, &bias, &R, &xs, &ys, &rs, &TT};
              launch_t(b, K_INK_GEMM_F32_ROUTER_R3, (rows + 31u) / 32u, (T + 31u) / 32u, sa);
            } else if (b->k[K_INK_GEMM_F32_ROUTER_R2] && !mk.rec && rows > 0u && cols > 0u && (cols & 3u) == 0u &&
                       (xs & 15u) == 0u && (((uintptr_t)w0) & 15u) == 0u && (((uintptr_t)A) & 15u) == 0u) {
              u32 TT = T;
              void *sa[] = {&w0, &A, &B, &rows, &cols, &bias, &R, &xs, &ys, &rs, &TT};
              launch_t(b, K_INK_GEMM_F32_ROUTER_R2, (rows + 31u) / 32u, (T + 63u) / 64u, sa);
            } else {
              u32 TT = T;
              void *sa[] = {&w0, &A, &B, &rows, &cols, &bias, &R, &xs, &ys, &rs, &TT};
              launch(b, K_GEMM_F32_S, ((rows + 7u) / 8u) * ((T + 7u) / 8u), sa);
            }
          } else if (wave32 && b->k[K_SALUKI_LORA_B_R11] && !mk.rec && cols == 32u &&
                     (rows >= 5120u || (rows == 1024u && (int)T < gmin)) && T >= 3u && R && !bias &&
                     (((uintptr_t)w0) & 15u) == 0u) {
            unsigned gx = (rows + 31u) / 32u;
            if (gx >= 253u && gx <= 256u) gx = 257u;
            u32 t_tile = rows >= 12288u ? 256u : (rows >= 5120u ? 128u : 32u);
            void *la[] = {&w0, &A, &B, &R, &rows, &T, &xs, &ys, &rs, &t_tile};
            launch_t(b, K_SALUKI_LORA_B_R11, gx, (T + t_tile - 1u) / t_tile, la);
          } else if (wave32 && b->k[K_SALUKI_LORA_B_T32] && !mk.rec && cols == 32u &&
                     (rows >= 5120u || (rows == 1024u && (int)T < gmin)) && T >= 3u && R && !bias) {
            unsigned gx = (rows + 31u) / 32u;
            if (gx >= 253u && gx <= 256u) gx = 257u;
            void *la[] = {&w0, &A, &B, &R, &rows, &T, &xs, &ys, &rs};
            launch_t(b, K_SALUKI_LORA_B_T32, gx, (T + 31u) / 32u, la);
          } else if (wave32 && b->k[K_SALUKI_LORA_B] && !mk.rec && cols == 32u && rows >= 5120u && T >= 3u && R && !bias) {
            void *la[] = {&w0, &A, &B, &R, &rows, &T, &xs, &ys, &rs};
            launch_t(b, K_SALUKI_LORA_B, gemv_groups(rows), (T + 3u) / 4u, la);
          } else if (narrow || (int)T >= gmin) launch_t(b, K_GEMV_F32_TW, gemv_groups(rows), T, ta); /* one row per wave */
          else if (b->k[K_SALUKI_LORA_A_V4] && !mk.rec && rows == 32u && !bias && !R && T >= 2u) {
            u32 TT = T;
            void *va[] = {&w0, &A, &B, &rows, &cols, &xs, &ys, &TT};
            launch_t(b, K_SALUKI_LORA_A_V4, rows, (T + 3u) / 4u, va);
          } else if (b->k[K_INK_GEMV_F32_ROUTER_VER_R4] && mk_ok_k(K_INK_GEMV_F32_ROUTER_VER_R4) && T >= 2u && T <= 16u && rows > 0u && cols > 0u) {
            u32 TT = T;
            void *sa[] = {&w0, &A, &B, &rows, &cols, &bias, &R, &xs, &ys, &rs, &TT};
            launch_t(b, K_INK_GEMV_F32_ROUTER_VER_R4, (rows + 3u) / 4u, 1u, sa);
          } else launch_t(b, K_GEMV_F32_T, rows >= 253u && rows <= 256u ? 257u : rows, T, ta); /* one row per workgroup */
          break;
        }
        void *args[] = {&w0, &A, &B, &rows, &cols, &bias, &R, &NW, &NY, &NQ, &eps, &b->gm_count};
        launch(b, K_GEMV_F32, rows >= 253u && rows <= 256u ? 257u : rows, args); /* one row per workgroup */
        break;
      }
      case OP_GEMV_Q5K:
      case OP_GEMV_IQ4XS: {
        u32 rows = o->w->rows, nb = o->w->nb;
        static int iq4_kxmin = -1;
        if (iq4_kxmin < 0) iq4_kxmin = getenv("IE_IQ4_KX_MIN") ? atoi(getenv("IE_IQ4_KX_MIN")) : 17;
        if (T > 1u && o->kind == OP_GEMV_IQ4XS && b->m->arch != ARCH_FLINT &&
            (bias != NULL || (int)T < iq4_kxmin || !(gx_wmma(b, T, nb, (u32)o->w->kind) && !getenv("IE_GEMM_KX_NOD")))) {
          /* verify / short prefill: ie_gemv_iq4xsq8_tr / _tr8 / _tr16, the weights read once in one pass up to 16 tokens */
          if (NW) ie_die("IQ4_XS GEMV of several tokens: no fused norm");
          u32 xs = g->bufs[o->a].stride, ys = g->bufs[o->b].stride, rs = o->res >= 0 ? g->bufs[o->res].stride : 0u;
          const int use_tr16 = T >= 10u && T <= 16u && b->has_iq4tr16 && iq4_tr8_on() && !mk.rec;
          const u32 tmax = use_tr16 ? 16u : (tr_tmax() == 8u && T == TR8_TMAX && b->has_iq4tr8 && iq4_tr8_on() && !mk.rec ? TR8_TMAX : tr_tmax());
          for (u32 t0 = 0; t0 < T;) {
            u32 TT = T - t0 < tmax ? T - t0 : tmax;
            void *At = (void *)((uint8_t *)A + (size_t)t0 * xs), *Bt = (void *)((uint8_t *)B + (size_t)t0 * ys);
            void *Rt = R ? (void *)((uint8_t *)R + (size_t)t0 * rs) : NULL;
            void *ta[] = {&w0, &w1, &At, &Bt, &rows, &nb, &TT, &bias, &Rt, &xs, &ys, &rs};
            if (TT >= 10u && TT <= 16u && b->has_iq4tr16 && iq4_tr8_on() && !mk.rec) {
              const u32 g16 = (rows + 15u) / 16u;
              launch(b, K_GEMV_IQ4XS_TR16, g16 >= 253u && g16 <= 256u ? 257u : g16, ta);
            } else if (TT >= 5u && TT <= TR8_TMAX && b->has_iq4tr8 && iq4_tr8_on() && !mk.rec) {
              const u32 g16 = (rows + 15u) / 16u;
              launch(b, K_GEMV_IQ4XS_TR8, g16 >= 253u && g16 <= 256u ? 257u : g16, ta); /* as gemv_groups */
            } else launch(b, K_GEMV_IQ4XS_TR, gemv_groups(rows), ta);
            t0 += TT;
          }
          break;
        }
        if (T > 1u) { /* all tokens: ie_gemm_kx (WMMA, GK_R x GK_T per workgroup) or ie_gexp_t */
          if (NW || bias) ie_die("Q5_K / IQ4_XS GEMV of several tokens: no fused norm, no bias");
          void *nul = NULL;
          u32 zero = 0u, one = 1u, kind = (u32)o->w->kind, xs = g->bufs[o->a].stride, ys = g->bufs[o->b].stride,
              rs = o->res >= 0 ? g->bufs[o->res].stride : 0u;
          void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &nul, &nul, &T, &zero, &zero, &xs, &ys, &one, &kind, &R, &rs};
          if (gx_wmma(b, T, nb, kind) && !getenv("IE_GEMM_KX_NOD")) launch_t(b, K_GEMM_KX, (rows + 127u) / 128u, (T + 63u) / 64u, args);
          else {
            if (nb > 128u) ie_die("Q5_K / IQ4_XS GEMV of several tokens: %u blocks per row (at most 128)", nb);
            launch(b, K_GEXP_T, (rows + 63u) / 64u, args); /* GXT_RB rows per workgroup */
          }
          break;
        }
        void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &NW, &NY, &NQ, &eps, &b->gm_count};
        launch(b, o->kind == OP_GEMV_Q5K ? K_GEMV_Q5K : K_GEMV_IQ4XS, gemv_groups(rows), args);
        break;
      }
      case OP_GEMV_RAW: {
        /* raw GGUF IQ/K blocks: ie_gemv_raw, one row per wave; T > 1 uses
         * grid y (the token), the kernel reads the byte strides */
        u32 rows = o->w->rows, nb = o->w->nb, type = o->w->sub;
        u32 xs = g->bufs[o->a].stride, ys = g->bufs[o->b].stride;
        u32 rs = o->res >= 0 ? g->bufs[o->res].stride : 0u;
        if (T > 1u && NW) ie_die("MAT_RAWQ GEMV of several tokens: no fused norm");
        void *args[] = {&w1, &A, &B, &rows, &nb, &type, &bias, &R, &NW, &NY, &NQ, &eps, &b->gm_count, &xs, &ys, &rs};
        if (T > 1u && raw_gemm_ok(T, nb, type)) {
          /* prompt chunk: ie_gemm_rq_<type> / ie_gemm_rq32_<type>, a weight tile decoded once and multiplied with
           * IE_RQ_T (64) or IE_RQ32_T (32) tokens by the matrix cores (kernels/ie_gemm_raw.inc). */
          const buf *ba = &g->bufs[o->a], *bb = &g->bufs[o->b];
          if (!ba->stride || !bb->stride || T > g->T) ie_die("MAT_RAWQ GEMM: %u tokens, strides %u / %u, graph T %u", T, ba->stride, bb->stride, g->T);
          /* every token's Q8 row / output row lies inside its own part of the buffer (stride xs / ys bytes) */
          if ((size_t)o->aoff + 40u * nb > xs || xs % 16u || o->aoff % 16u || (size_t)T * xs > ba->size) ie_die("MAT_RAWQ GEMM: activation offset %u + %u blocks of 40 bytes in a stride of %u, %u tokens (buffer %u)", o->aoff, nb, xs, T, ba->size);
          if ((size_t)o->boff + 4u * rows > ys || ys % 4u || o->boff % 4u || (size_t)T * ys > bb->size) ie_die("MAT_RAWQ GEMM: output offset %u + %u rows in a stride of %u, %u tokens (buffer %u)", o->boff, rows, ys, T, bb->size);
          if (o->res >= 0 && (rs < 4u * rows || rs % 4u || (size_t)T * rs > g->bufs[o->res].size)) ie_die("MAT_RAWQ GEMM: residual stride %u for %u rows, %u tokens", rs, rows, T);
          const u32 nrt = (rows + IE_RQ_R - 1u) / IE_RQ_R, ntt = (T + IE_RQ_T - 1u) / IE_RQ_T;
          if (!rows || (uint64_t)nrt * ntt > 0x7fffffffull) ie_die("MAT_RAWQ GEMM: grid of %u x %u workgroups", nrt, ntt);
          void *ga[] = {&w1, &A, &B, &rows, &nb, &T, &bias, &R, &xs, &ys, &rs};
          /* IE_RAW_GEMM_I256 (default 0: exact per-32 Q8_0 scales via LDS-LUT pipelined ie_gemm_rq_* / ie_gemm_rq32_*;
           * 1: tokens requantized with one scale per 256 via ie_requant_sb + ie_gemm_ri_<type>). */
          static int i256 = -1;
          if (i256 < 0) i256 = getenv("IE_RAW_GEMM_I256") ? getenv("IE_RAW_GEMM_I256")[0] - '0' : 0;
          if (i256 > 0 && b->rq && (size_t)T * xs <= b->rq_bytes) {
            const u32 nri = (rows + IE_RI_R - 1u) / IE_RI_R, ntr = (T + IE_RI_T - 1u) / IE_RI_T;
            if ((uint64_t)nri * ntr > 0x7fffffffull) ie_die("MAT_RAWQ GEMM: grid of %u x %u workgroups", nri, ntr);
            void *Q = b->rq;
            void *ra[] = {&A, &Q, &nb, &xs, &xs};
            launch_t(b, K_REQUANT_SB, (nb / 8u + 7u) / 8u, T, ra);
            void *gi[] = {&w1, &Q, &B, &rows, &nb, &T, &bias, &R, &xs, &ys, &rs};
            if (i256 == 2) launch(b, K_GEMM_RAW_Q2K + (raw_kernel(type) - K_GEMV_RAW_Q2K), nrt * ntt, gi);
            else launch(b, K_GEMM_RAWI_Q2K + (raw_kernel(type) - K_GEMV_RAW_Q2K), nri * ntr, gi);
          } else if (b->has_rq32 && rq32_max() > 0 && (int)T <= rq32_max()) {
            const u32 nr32 = (rows + 63u) / 64u, nt32 = (T + IE_RQ32_T - 1u) / IE_RQ32_T;
            launch(b, K_GEMM_RAW32_Q2K + (raw_kernel(type) - K_GEMV_RAW_Q2K), nr32 * nt32, ga);
          } else {
            const int ridx = raw_kernel(type) - K_GEMV_RAW_Q2K;
            const u32 ntt128 = (T + 127u) / 128u;
            const u32 g128 = nrt * ntt128;
            const int k128 = (ridx >= 0 && ridx < 8)
                ? ((type == GGML_IQ2_XXS && b->k[K_SALUKI_RQ128_IQ2XXS_R11]) ? K_SALUKI_RQ128_IQ2XXS_R11
                   : (type == GGML_IQ2_XXS && b->k[K_SALUKI_RQ128_IQ2XXS_R10]) ? K_SALUKI_RQ128_IQ2XXS_R10
                   : (type == GGML_IQ2_XXS && b->k[K_SALUKI_RQ128_IQ2XXS_SB]) ? K_SALUKI_RQ128_IQ2XXS_SB
                   : (type == GGML_Q2_K && b->k[K_SALUKI_RQ128_Q2K_R10]) ? K_SALUKI_RQ128_Q2K_R10
                   : (type == GGML_Q2_K && b->k[K_SALUKI_RQ128_Q2K_PIPE]) ? K_SALUKI_RQ128_Q2K_PIPE
                   : (type == GGML_IQ1_S && b->k[K_SALUKI_RQ128_IQ1S_R10]) ? K_SALUKI_RQ128_IQ1S_R10
                   : (type == GGML_IQ2_S && b->k[K_SALUKI_RQ128_IQ2S_R11] && xs <= 0x01FFFFFFu) ? K_SALUKI_RQ128_IQ2S_R11
                   : (type == GGML_IQ2_S && b->k[K_SALUKI_RQ128_IQ2S_PIPE]) ? K_SALUKI_RQ128_IQ2S_PIPE
                   : (type == GGML_IQ3_S && b->k[K_SALUKI_RQ128_IQ3S_PIPE]) ? K_SALUKI_RQ128_IQ3S_PIPE
                   : (type == GGML_IQ2_XS && b->k[K_SALUKI_RQ128_IQ2XS_PIPE]) ? K_SALUKI_RQ128_IQ2XS_PIPE
                   : (type == GGML_IQ1_M && b->k[K_SALUKI_RQ128_IQ1M_PIPE]) ? K_SALUKI_RQ128_IQ1M_PIPE
                   : (type == GGML_IQ3_XXS && b->k[K_SALUKI_RQ128_IQ3XXS_PIPE]) ? K_SALUKI_RQ128_IQ3XXS_PIPE
                   : (b->k[K_SALUKI_RQ128_Q2K + ridx] ? K_SALUKI_RQ128_Q2K + ridx : -1))
                : -1;
            if (k128 >= 0 && T >= 193u && (((T - 1u) & 64u) != 0u) && g128 >= 80u) {
              launch(b, k128, g128 >= 253u && g128 <= 256u ? 257u : g128, ga);
            } else if (ridx >= 0 && ridx < 8 && b->k[K_SALUKI_RQ64_Q2K + ridx]) {
              const u32 g64 = nrt * ntt;
              launch(b, K_SALUKI_RQ64_Q2K + ridx, g64 >= 253u && g64 <= 256u ? 257u : g64, ga);
            } else {
              launch(b, K_GEMM_RAW_Q2K + ridx, nrt * ntt, ga);
            }
          }
          static int chk = -1;
          if (chk < 0) chk = getenv("IE_RAW_GEMM_CHECK") != NULL;
          if (chk) raw_gemm_check(b, w1, A, B, bias, R, rows, nb, T, xs, ys, rs, type);
          break;
        }
        if (T > 1u) {
          if (b->has_rqt16 && !mk.rec && rqt16_min() > 0 && T >= (u32)rqt16_min() && T <= 16u && nb >= 8u && (nb & 7u) == 0u) {
            void *ga[] = {&w1, &A, &B, &rows, &nb, &T, &bias, &R, &xs, &ys, &rs};
            launch(b, K_GEMM_RQ_T16_Q2K + (raw_kernel(type) - K_GEMV_RAW_Q2K), (rows + 127u) / 128u, ga);
            break;
          }
          /* prefill / verify: tr_tmax() tokens per weight-block read (or 10..16 in one coalesced _tr16 launch) */
          const int rt = raw_kernel_tr(type);
          unsigned gr = gemv_groups(rows);
          u32 t0 = 0;
          const int use_tr16 = T >= 10u && T <= 16u && b->has_tr16 && tr16_on() && !mk.rec;
          const u32 tmax = use_tr16 ? 16u : (tr_tmax() == 8u && T == TR8_TMAX && tr8_on() && !mk.rec ? TR8_TMAX : tr_tmax());
          while (t0 < T) {
            u32 TT = T - t0 < tmax ? T - t0 : tmax;
            void *At = (void *)((uint8_t *)A + (size_t)t0 * xs);
            void *Bt = (void *)((uint8_t *)B + (size_t)t0 * ys);
            void *Rt = R ? (void *)((uint8_t *)R + (size_t)t0 * rs) : NULL;
            void *ta[] = {&w1, &At, &Bt, &rows, &nb, &TT, &bias, &Rt, &NW, &NY, &NQ, &eps, &b->gm_count, &xs, &ys, &rs};
            if (TT >= 2u && TT <= 4u && tr4_on() && !mk.rec) launch(b, raw_kernel_tr4(type), tr4_groups(rows), ta);
            else if (TT >= 5u && TT <= TR8_TMAX && tr8_on() && !mk.rec) {
              const int saluki_shape = (rows == 5120u && nb == 17408u / 32u) ||
                                       ((rows == 17408u || rows == 10240u) && nb == 5120u / 32u);
              const int saluki_xxs_tr9 = (rows == 5120u && nb == 17408u / 32u) ||
                                         ((rows == 17408u || rows == 12288u) && nb == 5120u / 32u);
              const int saluki_xxs_tr9_p1 = (rows == 17408u || rows == 12288u || rows == 10240u) && nb == 5120u / 32u;
              const int saluki_xxs_r10 = (rows == 5120u && nb == 17408u / 32u) ||
                                         ((rows == 17408u || rows == 12288u || rows == 10240u) && nb == 5120u / 32u);
              if (b->k[K_SALUKI_IQ2XXS_R4_LDS] && type == GGML_IQ2_XXS && TT == 8u && saluki_xxs_r10)
                launch(b, K_SALUKI_IQ2XXS_R4_LDS, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ2XXS_TR9_R10] && type == GGML_IQ2_XXS && TT == 8u &&
                       rows == 5120u && nb == 17408u / 32u)
                launch(b, K_SALUKI_IQ2XXS_TR9_R10, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ2XXS_TR9_PIPE1] && type == GGML_IQ2_XXS && TT == 8u && saluki_xxs_tr9_p1)
                launch(b, K_SALUKI_IQ2XXS_TR9_PIPE1, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ2XXS_TR9] && type == GGML_IQ2_XXS && TT == 8u && saluki_xxs_tr9)
                launch(b, K_SALUKI_IQ2XXS_TR9, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ2XXS_TR8] && type == GGML_IQ2_XXS && TT == 8u && saluki_shape)
                launch(b, K_SALUKI_IQ2XXS_TR8, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ2S_R4_LDS] && type == GGML_IQ2_S && TT == 8u && rows == 17408u && nb == 160u)
                launch(b, K_SALUKI_IQ2S_R4_LDS, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ2S_TR8_PIPE1] && type == GGML_IQ2_S && TT == 8u && rows == 17408u && nb == 160u)
                launch(b, K_SALUKI_IQ2S_TR8_PIPE1, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ2S_TR8] && type == GGML_IQ2_S && TT == 8u && rows == 17408u && nb == 160u)
                launch(b, K_SALUKI_IQ2S_TR8, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ1S_TR9_R10] && type == GGML_IQ1_S && TT == 8u &&
                       rows == 5120u && nb == 544u)
                launch(b, K_SALUKI_IQ1S_TR9_R10, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ1S_TR8_PIPE1] && type == GGML_IQ1_S && TT == 8u &&
                       rows == 17408u && nb == 160u && !bias && !Rt)
                launch(b, K_SALUKI_IQ1S_TR8_PIPE1, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ1S_TR8] && type == GGML_IQ1_S && TT == 8u &&
                       rows == 17408u && nb == 160u && !bias && !Rt)
                launch(b, K_SALUKI_IQ1S_TR8, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ1S_DOWN_TR8] && type == GGML_IQ1_S && TT == 8u &&
                       rows == 5120u && nb == 544u)
                launch(b, K_SALUKI_IQ1S_DOWN_TR8, rows / 32u, ta);
              else if (b->k[K_SALUKI_Q2K_TR9_R11] && type == GGML_Q2_K && TT == 8u &&
                       (rows == 17408u || rows == 12288u) && nb == 160u)
                launch(b, K_SALUKI_Q2K_TR9_R11, rows / 32u, ta);
              else if (b->k[K_SALUKI_Q2K_R4_LDS] && type == GGML_Q2_K && TT == 8u &&
                       (rows == 17408u || rows == 12288u) && nb == 160u)
                launch(b, K_SALUKI_Q2K_R4_LDS, rows / 32u, ta);
              else if (b->k[K_SALUKI_Q2K_TR8_R4_PIPE1] && type == GGML_Q2_K && TT == 8u &&
                       (rows == 17408u || rows == 12288u) && nb == 160u)
                launch(b, K_SALUKI_Q2K_TR8_R4_PIPE1, rows / 32u, ta);
              else if (b->k[K_SALUKI_Q2K_TR8_R4] && type == GGML_Q2_K && TT == 8u &&
                       (rows == 17408u || rows == 12288u) && nb == 160u)
                launch(b, K_SALUKI_Q2K_TR8_R4, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ2XS_R4_LDS] && type == GGML_IQ2_XS && TT == 8u &&
                       (rows == 17408u || rows == 12288u || rows == 10240u) && nb == 160u)
                launch(b, K_SALUKI_IQ2XS_R4_LDS, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ2XS_TR9_PIPE1] && type == GGML_IQ2_XS && TT == 8u &&
                       (rows == 17408u || rows == 12288u || rows == 10240u) && nb == 160u)
                launch(b, K_SALUKI_IQ2XS_TR9_PIPE1, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ2XS_TR9] && type == GGML_IQ2_XS && TT == 8u &&
                       rows == 17408u && nb == 160u)
                launch(b, K_SALUKI_IQ2XS_TR9, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ1M_TR9_R10] && type == GGML_IQ1_M && TT == 8u &&
                       rows == 17408u && nb == 160u && !bias && !Rt)
                launch(b, K_SALUKI_IQ1M_TR9_R10, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ1M_TR8_R4_PIPE1] && type == GGML_IQ1_M && TT == 8u &&
                       rows == 17408u && nb == 160u && !bias && !Rt)
                launch(b, K_SALUKI_IQ1M_TR8_R4_PIPE1, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ1M_TR8_R4] && type == GGML_IQ1_M && TT == 8u &&
                       rows == 17408u && nb == 160u && !bias && !Rt)
                launch(b, K_SALUKI_IQ1M_TR8_R4, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ3S_TR8_R4] && type == GGML_IQ3_S && TT == 8u &&
                       rows == 248320u && nb == 160u && !bias && !Rt)
                launch(b, K_SALUKI_IQ3S_TR8_R4, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ3S_R4_LDS] && type == GGML_IQ3_S && TT == 8u &&
                       ((rows == 17408u && nb == 160u && !bias && !Rt) || (rows == 5120u && nb == 544u)))
                launch(b, K_SALUKI_IQ3S_R4_LDS, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ3S_TR8_R2_PIPE1] && type == GGML_IQ3_S && TT == 8u &&
                       rows == 17408u && nb == 160u && !bias && !Rt)
                launch(b, K_SALUKI_IQ3S_TR8_R2_PIPE1, rows / 16u, ta);
              else if (b->k[K_SALUKI_IQ3S_TR8_R2] && type == GGML_IQ3_S && TT == 8u &&
                       rows == 17408u && nb == 160u && !bias && !Rt)
                launch(b, K_SALUKI_IQ3S_TR8_R2, rows / 16u, ta);
              else if (b->k[K_SALUKI_IQ3XXS_TR9_PIPE1] && type == GGML_IQ3_XXS && TT == 8u &&
                       rows == 17408u && nb == 160u)
                launch(b, K_SALUKI_IQ3XXS_TR9_PIPE1, rows / 32u, ta);
              else if (b->k[K_SALUKI_IQ3XXS_TR9] && type == GGML_IQ3_XXS && TT == 8u &&
                       rows == 17408u && nb == 160u)
                launch(b, K_SALUKI_IQ3XXS_TR9, rows / 32u, ta);
              else launch(b, raw_kernel_tr8(type), tr8_groups(rows, TT, type), ta);
            }
            else if (TT >= 10u && TT <= 16u && b->has_tr16 && tr16_on() && !mk.rec) launch(b, raw_kernel_tr16(type), tr16_groups(rows), ta);
            else launch(b, rt, gr, ta);
            t0 += TT;
          }
          break;
        }
        /* one row per wave, lanes over 32-weight sub-blocks: all 32 lanes are
         * busy at any row width, so no narrow-matrix variant is needed */
        launch(b, raw_kernel(type), gemv_groups(rows), args);
        break;
      }
      case OP_MOE_ROUTE: {
        u32 K = o->nh;
        float sc = m->escale;
        void *EB = b->d[i].w0; /* the router bias (o->v), or NULL */
        u32 ls = g->bufs[o->a].stride, ost = g->bufs[o->b].stride;
        void *args[] = {&A, &n, &K, &EB, &sc, &B, &ls, &ost};
        launch_t(b, K_MOE_ROUTE, 1, T, args);
        if (T > 1u) { /* the pairs by expert for ie_gexp_t */
          u32 ne = m->n_exp;
          if (T * K > IE_MS_MAXP || ne > 1024u) ie_die("flint chunk: %u pairs (at most %u), %u experts (at most 1024)", T * K, IE_MS_MAXP, ne);
          const u32 slot_max = ink_moe_slot_max(b);
          const int skip_sort = T >= 2u && T <= 16u && T <= slot_max && b->ink_moe_all_slot && !mk.rec &&
                                b->k[K_INK_GEXP_RAW_IQ3XXS_SLOT_R6] && b->k[K_INK_GEXP_IQ4NL_SLOT_R6];
          const u32 mp_a = T * K, mp_b = ne + (T * K) / 4u;
          const u32 max_packs = mp_a < mp_b ? mp_a : mp_b;
          if (skip_sort) {
            /* direct per-slot R6 MoE kernels read route directly without sort */
          } else if (T >= 2u && T <= 16u && !mk.rec && b->k[K_INK_MOE_SORT_ADAP_R8] &&
                     ink_moe_adap_mode() == 1 && b->ms_packs1 && b->ms_npacks1 && b->ms_packsM && b->ms_npacksM &&
                     ne <= 256u && T * K <= 4096u && max_packs <= 512u) {
            void *sa[] = {&B, &ost, &T, &K, &ne, &b->ms_offs, &b->ms_list,
                          &b->ms_packs1, &b->ms_npacks1, &b->ms_packsM, &b->ms_npacksM};
            launch(b, K_INK_MOE_SORT_ADAP_R8, 1, sa);
          } else if (T >= 2u && T <= 16u && b->k[K_INK_MOE_SORT_COMPACT_R6] && mk_ok_k(K_INK_MOE_SORT_COMPACT_R6) &&
                     b->ms_packs && b->ms_npacks && ne <= 256u && T * K <= 4096u && max_packs <= 160u) {
            void *sa[] = {&B, &ost, &T, &K, &ne, &b->ms_offs, &b->ms_list, &b->ms_packs, &b->ms_npacks};
            launch(b, K_INK_MOE_SORT_COMPACT_R6, 1, sa);
          } else if (T >= 17u && !mk.rec && b->k[K_INK_MOE_PF_WORKLIST_R7] &&
                     b->ms_packs && b->ms_npacks && ne > 0u && ne <= 256u && T * K <= 8192u &&
                     (T * K < ne + (T * K) / 64u ? T * K : ne + (T * K) / 64u) <= 512u) {
            void *sa[] = {&B, &ost, &T, &K, &ne, &b->ms_offs, &b->ms_list, &b->ms_packs, &b->ms_npacks};
            launch(b, K_INK_MOE_PF_WORKLIST_R7, 1, sa);
          } else {
            void *sa[] = {&B, &ost, &T, &K, &ne, &b->ms_offs, &b->ms_list};
            launch(b, (b->k[K_INK_MOE_SORT] && mk_ok_k(K_INK_MOE_SORT) && T * K <= (mk.rec ? 512u : 8192u) && ne <= (mk.rec ? 256u : 1024u)) ? K_INK_MOE_SORT : K_MOE_SORT, 1, sa);
          }
        }
        break;
      }
      case OP_GEMV_EXP: {
        const mat *w = o->w;
        if (w->kind == MAT_RAWQ) {
          u32 rows = w->rows, nb = w->nb, as = o->astr, bs = o->bstr, type = w->sub, q_bytes = (u32)ie_rawq_bytes(w);
          if (T > 1u) {
            if (g->bufs[o->b].rows && T > g->bufs[o->b].rows)
              ie_die("OP_GEMV_EXP: exp_gu capped to %u rows (T=%u) but fused IQ3+SITU R6 did not run", g->bufs[o->b].rows, T);
            const u32 K = o->nh, at = g->bufs[o->a].stride, bt = g->bufs[o->b].stride;
            if (bt != K * bs || (as && at != K * as)) ie_die("raw expert GEMV of a chunk: token strides %u, %u (slots %u, %u)", at, bt, as, bs);
            u32 xs = as ? as : at, xdiv = as ? 1u : K, zero = 0u;
            void *nul = NULL;
            static int rq_exp_min = -1;
            if (rq_exp_min < 0) rq_exp_min = getenv("IE_GEMM_RQ_EXP_MIN") ? atoi(getenv("IE_GEMM_RQ_EXP_MIN")) : 32;
            const u32 mp_a = T * K, mp_b = m->n_exp + (T * K) / 4u;
            const u32 max_packs = mp_a < mp_b ? mp_a : mp_b;
            if (type == 18u /* GGML_IQ3_XXS */ && (int)T >= rq_exp_min && rows % 32u == 0u && nb >= 8u && (nb & 7u) == 0u &&
                !getenv("IE_GEMM_RQ_EXP_NO")) {
              void *ta[] = {&w1, &A, &B, &rows, &nb, &b->ms_list, &b->ms_offs, &zero, &q_bytes, &xs, &bs, &xdiv, &nul, &zero};
              const int kk = (b->k[K_INK_GEMM_RQ_EXP_IQ3XXS_R3] && !mk.rec && rows > 0u)
                                 ? K_INK_GEMM_RQ_EXP_IQ3XXS_R3
                                 : (b->k[K_INK_GEMM_RQ_EXP_IQ3XXS_R2] && !mk.rec && rows > 0u)
                                       ? K_INK_GEMM_RQ_EXP_IQ3XXS_R2
                                       : (b->k[K_INK_GEMM_RQ_EXP_IQ3XXS] && !mk.rec && rows > 0u) ? K_INK_GEMM_RQ_EXP_IQ3XXS
                                                                                                  : K_GEMM_RQ_EXP_IQ3XXS;
              launch_t(b, kk, (rows + 127u) / 128u, m->n_exp, ta);
            } else if (T >= 2u && T <= 16u && T <= ink_moe_slot_max(b) &&
                       b->k[K_INK_GEXP_RAW_IQ3XXS_SLOT_R6] && mk_ok_k(K_INK_GEXP_RAW_IQ3XXS_SLOT_R6) &&
                       type == 18u && rows > 0u && nb >= 8u && (nb & 7u) == 0u) {
              u32 rst = g->bufs[o->c].stride, KK = K;
              void *ta[] = {&w1, &A, &B, &rows, &nb, &C, &rst, &KK, &q_bytes, &xs, &bs, &xdiv, &nul, &zero};
              launch_t(b, K_INK_GEXP_RAW_IQ3XXS_SLOT_R6, (rows + 31u) / 32u, T * K, ta);
            } else if (T >= 2u && T <= 16u &&
                       b->k[K_INK_GEXP_RAW_IQ3XXS_COMPACT_R6] && mk_ok_k(K_INK_GEXP_RAW_IQ3XXS_COMPACT_R6) &&
                       b->k[K_INK_MOE_SORT_COMPACT_R6] && b->ms_packs && b->ms_npacks &&
                       type == 18u && rows > 0u && nb >= 8u && (nb & 7u) == 0u &&
                       m->n_exp <= 256u && T * K <= 4096u && max_packs <= 160u) {
              void *ta[] = {&w1, &A, &B, &rows, &nb, &b->ms_list, &b->ms_packs, &b->ms_npacks, &q_bytes, &xs, &bs, &xdiv, &nul, &zero};
              launch_t(b, K_INK_GEXP_RAW_IQ3XXS_COMPACT_R6, (rows + 31u) / 32u, max_packs, ta);
            } else if (b->k[K_INK_GEXP_RAW_IQ3XXS_VER_R4] && mk_ok_k(K_INK_GEXP_RAW_IQ3XXS_VER_R4) && T >= 2u && T <= 16u &&
                       type == 18u && rows > 0u && nb >= 8u && (nb & 7u) == 0u) {
              void *ta[] = {&w1, &A, &B, &rows, &nb, &b->ms_list, &b->ms_offs, &zero, &q_bytes, &xs, &bs, &xdiv, &nul, &zero};
              launch_t(b, K_INK_GEXP_RAW_IQ3XXS_VER_R4, (rows + 31u) / 32u, m->n_exp, ta);
            } else {
              void *ta[] = {&w1, &A, &B, &rows, &nb, &b->ms_list, &b->ms_offs, &zero, &q_bytes, &xs, &bs, &xdiv, &type, &nul, &zero};
              launch_t(b, K_GEXP_RAW_T, (rows + 15u) / 16u, m->n_exp, ta);
            }
          } else {
            void *args[] = {&w1, &A, &B, &rows, &nb, &C, &q_bytes, &as, &bs, &type};
            const int kg = (type == 18u && rows > 0u && nb >= 8u && (nb & 7u) == 0u)
                               ? ((b->k[K_INK_GEXP_RAW_IQ3XXS_R3] && mk_ok_k(K_INK_GEXP_RAW_IQ3XXS_R3))
                                      ? K_INK_GEXP_RAW_IQ3XXS_R3
                                      : (b->k[K_INK_GEXP_RAW_IQ3XXS] && mk_ok_k(K_INK_GEXP_RAW_IQ3XXS))
                                            ? K_INK_GEXP_RAW_IQ3XXS
                                            : -1)
                               : -1;
            if (kg >= 0) {
              launch_t(b, kg, (rows + 31u) / 32u, o->nh, args);
            } else {
              launch_t(b, K_GEXP_RAW, gemv_groups(rows), o->nh, args);
            }
          }
          break;
        }
        if (w->kind != MAT_Q4 && w->kind != MAT_Q8 && w->kind != MAT_Q4K && w->kind != MAT_Q6K &&
            w->kind != MAT_Q5K && w->kind != MAT_IQ4XS && w->kind != MAT_IQ4NL)
          ie_die("expert GEMV: kind %d has no GPU kernel", w->kind);
        u32 rows = w->rows, nb = w->nb, as = o->astr, bs = o->bstr, kind = (u32)w->kind;
        u32 qwn = (u32)mat_qw_words(w->kind, w->rows, w->cols), qsn = (u32)mat_qs_n(w->kind, w->rows, w->cols);
        /* IE_GEXP_TWO=1: the gate/up experts (nb 64) two rows per wave too. Off by
         * default: measured slower (flint UD-Q4_K_XL: 1.45 vs 1.26 ms per token in ie_gexp) */
        static int two_env = -1;
        if (two_env < 0) two_env = getenv("IE_GEXP_TWO") && getenv("IE_GEXP_TWO")[0] == '1';
        u32 two_off = !two_env;
        if (T > 1u) { /* a prompt chunk: by expert (ie_moe_sort in OP_MOE_ROUTE), ie_gexp_t */
          const u32 K = o->nh, at = g->bufs[o->a].stride, bt = g->bufs[o->b].stride;
          if (bt != K * bs || (as && at != K * as)) ie_die("expert GEMV of a chunk: token strides %u, %u (slots %u, %u)", at, bt, as, bs);
          u32 xs = as ? as : at, xdiv = as ? 1u : K, zero = 0u;
          void *nul = NULL;
          void *ta[] = {&w0, &w1, &A, &B, &rows, &nb, &b->ms_list, &b->ms_offs, &zero, &qwn, &qsn, &xs, &bs, &xdiv, &kind, &nul, &zero};
          if (gx_wmma(b, T, nb, kind) && !getenv("IE_GEMM_KX_NOE")) {
            const u32 mt_a = T * K, mt_b = m->n_exp + (T * K) / 64u;
            const u32 max_tiles = mt_a < mt_b ? mt_a : mt_b;
            if ((b->k[K_INK_GEMM_IQ4NL_EXP_R9] || b->k[K_INK_GEMM_IQ4NL_EXP_R7]) && b->k[K_INK_MOE_PF_WORKLIST_R7] && !mk.rec &&
                kind == (u32)MAT_IQ4NL && nb > 0u && (nb & 3u) == 0u &&
                b->ms_packs && b->ms_npacks && m->n_exp <= 256u && T * K <= 8192u && max_tiles <= 512u) {
              void *t7[] = {&w0, &w1, &A, &B, &rows, &nb, &b->ms_list, &b->ms_packs, &b->ms_npacks, &qwn, &qsn, &xs, &bs, &xdiv, &kind, &nul, &zero};
              const int ki4 = b->k[K_INK_GEMM_IQ4NL_EXP_R9] ? K_INK_GEMM_IQ4NL_EXP_R9 : K_INK_GEMM_IQ4NL_EXP_R7;
              launch_t(b, ki4, (rows + 127u) / 128u, max_tiles, t7);
            } else if (b->k[K_INK_GEMM_IQ4NL_EXP_R2] && !mk.rec && kind == (u32)MAT_IQ4NL && nb > 0u && (nb & 3u) == 0u) {
              launch_t(b, K_INK_GEMM_IQ4NL_EXP_R2, (rows + 127u) / 128u, m->n_exp, ta);
            } else if (b->k[K_INK_GEMM_IQ4NL_EXP] && !mk.rec && kind == (u32)MAT_IQ4NL && nb > 0u && (nb & 3u) == 0u) {
              launch_t(b, K_INK_GEMM_IQ4NL_EXP, (rows + 127u) / 128u, m->n_exp, ta);
            } else {
              launch_t(b, K_GEMM_KX, (rows + 127u) / 128u, m->n_exp, ta); /* WMMA, GK_R rows per workgroup */
            }
            if (getenv("IE_KX_CHECK")) { /* debug: against ie_gexp_t (into a copy), the worst pair and row */
              const size_t need = (size_t)T * K * bs;
              float *y1 = malloc(need), *y2 = malloc(need);
              void *dev;
              HIP(H.Malloc(&dev, need));
              HIP(H.Memcpy(dev, B, need, hipMemcpyDeviceToDevice));
              void *t2[] = {&w0, &w1, &A, &dev, &rows, &nb, &b->ms_list, &b->ms_offs, &zero, &qwn, &qsn, &xs, &bs, &xdiv, &kind, &nul, &zero};
              HIP(H.Memcpy(y1, B, need, hipMemcpyDeviceToHost));
              launch_t(b, K_GEXP_T, (rows + 63u) / 64u, m->n_exp, t2);
              HIP(H.Memcpy(y2, dev, need, hipMemcpyDeviceToHost));
              double mx = 0, d = 0;
              size_t wi = 0;
              int nn = 0;
              for (size_t p = 0; p < (size_t)T * K; p++)
                for (u32 r = 0; r < rows; r++) {
                  const double u = y1[p * (bs / 4) + r], v = y2[p * (bs / 4) + r];
                  if (u != u) nn++;
                  if (fabs(v) > mx) mx = fabs(v);
                  if (!(fabs(u - v) <= d)) d = fabs(u - v), wi = p * 100000 + r;
                }
              fprintf(stderr, "kx check layer %d %ux%u kind %u: max |kx - gexp_t| %.3e of %.3e (pair %zu row %zu), %d nan\n", o->layer, rows, w->cols,
                      kind, d, mx, wi / 100000, wi % 100000, nn);
              H.Free(dev), free(y1), free(y2);
            }
          }
          else if (T >= 2u && T <= 16u && T <= ink_moe_slot_max(b) &&
                   b->k[K_INK_GEXP_IQ4NL_SLOT_R6] && mk_ok_k(K_INK_GEXP_IQ4NL_SLOT_R6) &&
                   kind == (u32)MAT_IQ4NL && rows > 0u && nb > 0u) {
            u32 rst = g->bufs[o->c].stride, KK = K;
            void *sa[] = {&w0, &w1, &A, &B, &rows, &nb, &C, &rst, &KK, &qwn, &qsn, &xs, &bs, &xdiv, &kind, &nul, &zero};
            launch_t(b, K_INK_GEXP_IQ4NL_SLOT_R6, (rows + 31u) / 32u, T * K, sa);
          } else if (T >= 2u && T <= 16u && !mk.rec &&
                     b->k[K_INK_GEXP_IQ4NL_NJ1_R8] && b->k[K_INK_GEXP_IQ4NL_NJM_R8] &&
                     ((ink_moe_adap_mode() >= 1 && b->k[K_INK_MOE_SORT_ADAP_R8] && b->ms_packs1 && b->ms_npacks1 && b->ms_packsM && b->ms_npacksM) ||
                      (ink_moe_adap_mode() == 0 && b->k[K_INK_MOE_SORT_COMPACT_R6] && b->ms_packs && b->ms_npacks)) &&
                     kind == (u32)MAT_IQ4NL && rows > 0u && nb > 0u &&
                     m->n_exp <= 256u && T * K <= 4096u &&
                     (T * K < m->n_exp + (T * K) / 4u ? T * K : m->n_exp + (T * K) / 4u) <= 160u) {
            const int sep = (ink_moe_adap_mode() >= 1 && b->k[K_INK_MOE_SORT_ADAP_R8]);
            const u32 mp_a = T * K, mp_b = m->n_exp + (T * K) / 4u;
            const u32 max_packs = mp_a < mp_b ? mp_a : mp_b;
            const u32 mp1 = sep ? (T * K < m->n_exp ? T * K : m->n_exp) : max_packs;
            const u32 mpM = sep ? ((T * K) / 2u < max_packs ? (T * K) / 2u : max_packs) : max_packs;
            void *p1 = sep ? b->ms_packs1 : b->ms_packs, *np1 = sep ? b->ms_npacks1 : b->ms_npacks;
            void *pM = sep ? b->ms_packsM : b->ms_packs, *npM = sep ? b->ms_npacksM : b->ms_npacks;
            void *ca1[] = {&w0, &w1, &A, &B, &rows, &nb, &b->ms_list, &p1, &np1, &qwn, &qsn, &xs, &bs, &xdiv, &kind, &nul, &zero};
            void *caM[] = {&w0, &w1, &A, &B, &rows, &nb, &b->ms_list, &pM, &npM, &qwn, &qsn, &xs, &bs, &xdiv, &kind, &nul, &zero};
            if (mp1) launch_t(b, K_INK_GEXP_IQ4NL_NJ1_R8, (rows + 31u) / 32u, mp1, ca1);
            if (mpM) launch_t(b, K_INK_GEXP_IQ4NL_NJM_R8, (rows + 31u) / 32u, mpM, caM);
          } else if (T >= 2u && T <= 16u &&
                     (b->k[K_INK_GEXP_IQ4NL_BW_COMPACT_R7] || (b->k[K_INK_GEXP_IQ4NL_COMPACT_R6] && mk_ok_k(K_INK_GEXP_IQ4NL_COMPACT_R6))) &&
                     b->k[K_INK_MOE_SORT_COMPACT_R6] && b->ms_packs && b->ms_npacks &&
                     kind == (u32)MAT_IQ4NL && rows > 0u && nb > 0u &&
                     m->n_exp <= 256u && T * K <= 4096u &&
                     (T * K < m->n_exp + (T * K) / 4u ? T * K : m->n_exp + (T * K) / 4u) <= 160u) {
            const u32 mp_a = T * K, mp_b = m->n_exp + (T * K) / 4u;
            const u32 max_packs = mp_a < mp_b ? mp_a : mp_b;
            void *ca[] = {&w0, &w1, &A, &B, &rows, &nb, &b->ms_list, &b->ms_packs, &b->ms_npacks, &qwn, &qsn, &xs, &bs, &xdiv, &kind, &nul, &zero};
            const int kc4 = (b->k[K_INK_GEXP_IQ4NL_BW_COMPACT_R7] && !mk.rec) ? K_INK_GEXP_IQ4NL_BW_COMPACT_R7 : K_INK_GEXP_IQ4NL_COMPACT_R6;
            launch_t(b, kc4, (rows + 31u) / 32u, max_packs, ca);
          } else if (b->k[K_INK_GEXP_IQ4NL_VER_R4] && mk_ok_k(K_INK_GEXP_IQ4NL_VER_R4) && T >= 2u && T <= 16u &&
                     kind == (u32)MAT_IQ4NL && rows > 0u && nb > 0u) {
            launch_t(b, K_INK_GEXP_IQ4NL_VER_R4, (rows + 31u) / 32u, m->n_exp, ta);
          } else {
            if (nb > 128u) ie_die("expert GEMV of a chunk: %u blocks per row (at most 128)", nb);
            launch_t(b, K_GEXP_T, (rows + 63u) / 64u, m->n_exp, ta); /* GXT_RB rows per workgroup */
          }
          break;
        }
        void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &C, &qwn, &qsn, &as, &bs, &kind, &two_off};
        const int kg4 = (kind == (u32)MAT_IQ4NL)
                            ? ((b->k[K_INK_GEXP_IQ4NL_BW_T1_R7] && !mk.rec)
                                   ? K_INK_GEXP_IQ4NL_BW_T1_R7
                                   : (b->k[K_INK_GEXP_IQ4NL_R3] && mk_ok_k(K_INK_GEXP_IQ4NL_R3))
                                         ? K_INK_GEXP_IQ4NL_R3
                                         : (b->k[K_INK_GEXP_IQ4NL] && mk_ok_k(K_INK_GEXP_IQ4NL))
                                               ? K_INK_GEXP_IQ4NL
                                               : -1)
                            : -1;
        if (kg4 >= 0) {
          launch_t(b, kg4, (rows + 31u) / 32u, o->nh, args);
        } else {
          /* nb <= 64: two rows per wave, 16 per workgroup (ie_gexp) */
          const unsigned gx = nb <= 16u || (nb <= 64u && !two_off) ? (rows + 15u) / 16u : gemv_groups(rows);
          launch_t(b, K_GEXP, gx, o->nh, args);
        }
        break;
      }
      case OP_MOE_SWIGLU: { /* per slot: ie_swiglu with the slot strides (grid y = slot) */
        void *Bu = (char *)A + (size_t)n * 4u;
        u32 as = o->astr, qs = o->bstr; /* the slot strides */
        float cl0 = 0.0f;
        void *args[] = {&A, &Bu, &n, &Q, &as, &qs, &cl0};
        if (T > 1u && (g->bufs[o->a].stride != o->nh * as || g->bufs[o->qo].stride != o->nh * qs))
          ie_die("expert SwiGLU of a chunk: token strides %u, %u (slots %u, %u)", g->bufs[o->a].stride, g->bufs[o->qo].stride, as, qs);
        launch_t(b, K_SWIGLU, elem_groups, o->nh * T, args); /* the pairs t K + k: grid y */
        break;
      }
      case OP_MOE_SUM: {
        u32 K = o->nh;
        u32 ys = g->bufs[o->a].stride, os = g->bufs[o->b].stride, rts = g->bufs[o->c].stride, rs = o->res >= 0 ? g->bufs[o->res].stride : 0u;
        void *args[] = {&A, &B, &C, &R, &n, &K, &ys, &os, &rts, &rs};
        if (b->k[K_INK_MOE_SUM] && mk_ok_k(K_INK_MOE_SUM) && (n & 3u) == 0u && (ys & 15u) == 0u && (os & 15u) == 0u && (!R || (rs & 15u) == 0u)) {
          launch_t(b, K_INK_MOE_SUM, ((n >> 2u) + 255u) / 256u, T, args);
        } else {
          launch_t(b, K_MOE_SUM, elem_groups, T, args);
        }
        break;
      }
      case OP_BIAS: {
        void *args[] = {&A, &w0, &n};
        launch(b, K_BIAS, elem_groups, args);
        break;
      }
      case OP_ADD: {
        if (T > 1u) {
          u32 as = g->bufs[o->a].stride, bs = g->bufs[o->b].stride, cs = g->bufs[o->c].stride;
          void *args[] = {&A, &B, &C, &n, &as, &bs, &cs};
          launch_t(b, K_ADD_T, elem_groups, T, args);
        } else {
          void *args[] = {&A, &B, &C, &n};
          launch(b, K_ADD, elem_groups, args);
        }
        break;
      }
      case OP_COPY: {
        u32 as = g->bufs[o->a].stride / 4u, bs = g->bufs[o->b].stride / 4u;
        void *args[] = {&A, &B, &n, &as, &bs};
        launch_t(b, K_COPY_ROWS, elem_groups, T, args);
        break;
      }
      case OP_DF_CONV: {
        const dflash *d = m->df;
        u32 which = o->nh, gs = d->group, as = g->bufs[o->a].stride / 4u, bs = g->bufs[o->b].stride / 4u,
            ds = g->bufs[o->c].stride / 4u, rs = o->res >= 0 ? g->bufs[o->res].stride / 4u : 0u;
        void *args[] = {&A, &B, &C, &w0, &R, &n, &which, &gs, &as, &bs, &ds, &rs};
        launch_t(b, K_DF_CONV, elem_groups, T, args);
        break;
      }
      case OP_DF_QKV: {
        const dflash *d = m->df;
        const size_t lofs = (size_t)o->layer * b->df_ring * d->n_kv * d->hd;
        float *kc = b->dfk + lofs, *vc = b->dfv + lofs;
        u32 xs = g->bufs[o->a].stride / 4u, nh = o->nh, nk = d->n_kv, dh = d->hd, ring = b->df_ring;
        float de = d->eps, rb = d->rope_base;
        void *qw = w0;
        void *args[] = {&A, &xs, &qw, &NW, &kc, &vc, &nh, &nk, &dh, &pos, &de, &rb, &ring};
        launch_t(b, K_DF_QKV, nh + 2u * nk, T, args);
        break;
      }
      case OP_DF_ATTN: {
        const dflash *d = m->df;
        const size_t lofs = (size_t)o->layer * b->df_ring * d->n_kv * d->hd;
        float *kc = b->dfk + lofs, *vc = b->dfv + lofs;
        u32 qs = g->bufs[o->a].stride / 4u, os = g->bufs[o->b].stride / 4u, nh = o->nh, nk = d->n_kv, dh = d->hd, win = d->window,
            ring = b->df_ring;
        void *args[] = {&A, &qs, &B, &os, &kc, &vc, &pos, &T, &nh, &nk, &dh, &win, &ring};
        /* 4 query heads per KV head, hd 128: ie_df_attn_g (each K/V row read once for the 4 heads; IE_DFLASH_ATTN=0:
         * ie_df_attn, a workgroup per query head) */
        static int dfg = -1;
        if (dfg < 0) dfg = !(getenv("IE_DFLASH_ATTN") && getenv("IE_DFLASH_ATTN")[0] == '0');
        if (dfg && b->has_dfg && nh == 4u * nk && dh == 128u) launch_t(b, K_DF_ATTN_G, nk, T, args);
        else launch_t(b, K_DF_ATTN, nh, T, args);
        break;
      }
      case OP_SWIGLU: {
        u32 as = g->bufs[o->a].stride, qs = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
        u32 n_arg = (o->dead_b && Q && !mk.rec && deadb_on()) ? (n | 0x80000000u) : n;
        float cl = o->clamp;
        void *args[] = {&A, &B, &n_arg, &Q, &as, &qs, &cl};
        const int use_saluki_sw = b->k[K_SALUKI_SWIGLU_V4] && !mk.rec && T >= 17u && n > 0u && (n & 31u) == 0u &&
                                  ((uintptr_t)A & 15u) == 0u && ((uintptr_t)B & 15u) == 0u && (as & 15u) == 0u &&
                                  (!Q || (((uintptr_t)Q & 3u) == 0u && (qs & 3u) == 0u));
        if (use_saluki_sw) {
          const u32 sw_g = (n + 1023u) / 1024u;
          launch_t(b, K_SALUKI_SWIGLU_V4, sw_g >= 253u && sw_g <= 256u ? 257u : sw_g, T, args);
        } else {
          launch_t(b, K_SWIGLU, elem_groups, T, args);
        }
        break;
      }
      case OP_VRES: {
        u32 bs = g->bufs[o->b].stride, as = g->bufs[o->a].stride, cs = g->bufs[o->c].stride,
            qs = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
        float lam = o->lam;
        void *args[] = {&B, &A, &C, &n, &Q, &bs, &as, &cs, &qs, &lam};
        launch_t(b, K_VRES, elem_groups, T, args);
        break;
      }
      case OP_GATE: {
        u32 as = g->bufs[o->a].stride, bs = g->bufs[o->b].stride, qs = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
        if (T > 1u && b->k[K_INK_GATE_Q8_R2] && mk_ok_k(K_INK_GATE_Q8_R2) && (n & 31u) == 0u &&
            (((uintptr_t)A) & 15u) == 0u && (((uintptr_t)B) & 15u) == 0u &&
            (!Q || (((uintptr_t)Q) & 3u) == 0u) &&
            (((as | bs) & 15u) == 0u && (!Q || (qs & 3u) == 0u))) {
          void *Bout = (m->arch == ARCH_INKLING2 && Q != NULL && !gdump_on()) ? NULL : A;
          void *ga[] = {&A, &B, &Bout, &Q, &n, &as, &bs, &as, &qs};
          launch_t(b, K_INK_GATE_Q8_R2, (n + 1023u) / 1024u, T, ga);
        } else {
          void *args[] = {&A, &B, &n, &Q, &as, &bs, &qs};
          launch_t(b, K_GATE, elem_groups, T, args);
        }
        break;
      }
      case OP_ROPE: {
        float *cs = b->rcos + (size_t)pos * h2, *sn = b->rsin + (size_t)pos * h2;
        u32 nh = o->nh, neox = m->rope == ROPE_NEOX;
        void *args[] = {&A, &cs, &sn, &nh, (void *)&hd, &neox};
        launch(b, K_ROPE, (nh * h2 + 255u) / 256u, args);
        break;
      }
      case OP_ROPE_KV: {
        float *cs = b->rcos + (size_t)pos * h2, *sn = b->rsin + (size_t)pos * h2;
        const u32 kvi = m->arch == ARCH_INKLING2 ? o->kvi : m->l[o->layer].kvi;
        void *kc = kv_ptr(b, 0, kvi, pos % g->kv_len[kvi]), *vc = kv_ptr(b, 1, kvi, pos % g->kv_len[kvi]);
        u32 nq = o->nh, neox = m->rope == ROPE_NEOX;
        void *args[] = {&A, &B, &C, &kc, &vc, &cs, &sn, &nq, (void *)&nkv, (void *)&hd, &neox};
        launch(b, K_ROPE_KV, ((nq + nkv) * h2 + kvd + 255u) / 256u, args);
        break;
      }
      case OP_QKN_ROPE_KV: {
        float *cs = b->rcos + (size_t)pos * h2, *sn = b->rsin + (size_t)pos * h2;
        const u32 kvi = m->arch == ARCH_INKLING2 ? o->kvi : m->l[o->layer].kvi;
        const u32 kv_pos = (o->layer == (int)m->n_layer && graph_mtp_tr(g)->on) ? pos + graph_mtp_tr(g)->slot_base : pos;
        void *kc = kv_ptr(b, 0, kvi, kv_pos % g->kv_len[kvi]), *vc = kv_ptr(b, 1, kvi, kv_pos % g->kv_len[kvi]);
        if (m->arch == ARCH_FLINT || m->arch == ARCH_RICURSIVE) { /* q heads of qstr floats, no RoPE for a norope layer; T tokens (grid y) */
          void *kl = kv_ptr(b, 0, kvi, 0), *vl = kv_ptr(b, 1, kvi, 0);
          u32 nq = o->nh, nrot = o->norope ? 0u : m->n_rot, qstr = o->qstr, ins = g->bufs[o->a].stride, qos = g->bufs[o->b].stride,
              len = g->kv_len[kvi], h2r = h2;
          void *args[] = {&A, &B, &kl, &vl, &w0, &NW, &b->rcos, &b->rsin, &nq, (void *)&nkv, (void *)&hd, &nrot, &eps, &qstr,
                          &ins, &qos, &pos, &len, &h2r};
          launch_t(b, K_QKN_ROPE_KV2, nq + 2u * nkv, T, args);
          break;
        }
        u32 nq = o->nh, nrot = m->n_rot, qstr = o->qstr; /* qwen35: q heads are [q | gate] (2 hd); qwen3: hd */
        u32 ins = g->bufs[o->a].stride, qos = g->bufs[o->b].stride;
        if (b->tr_on) cs = b->tr_rope, sn = b->tr_rope + 16u * h2; /* a tree: node t's row is of position pos + its depth */
        b->q16_ready_op = -1;
        if (b->kv_only && g->i_mtp_wq >= 0) {
          void *Akv = (char *)A + (size_t)nq * qstr * 4u, *B0 = NULL, *w0_null = NULL;
          u32 nq0 = 0u;
          const int use_r11_kv = b->k[K_SALUKI_QKN_ROPE_KV_R11] && b->kvq8 && !b->kvrot && !mk.rec &&
                                 hd == 256u && T > 0u && (nrot == 0u || nrot == 64u || nrot == 128u || nrot == 256u) &&
                                 (qstr & 3u) == 0u && ((uintptr_t)Akv & 15u) == 0u && (T == 1u || (ins & 15u) == 0u) &&
                                 ((uintptr_t)kc & 15u) == 0u && ((uintptr_t)vc & 15u) == 0u && ((uintptr_t)NW & 15u) == 0u &&
                                 (nrot == 0u || (((uintptr_t)cs & 15u) == 0u && ((uintptr_t)sn & 15u) == 0u));
          if (use_r11_kv) {
            void *q16_null = NULL;
            u32 TT = T;
            void *akv11[] = {&Akv, &B0, &kc, &vc, &w0_null, &NW, &cs, &sn, &nq0, (void *)&nkv, (void *)&hd, &nrot, &eps, &ins, &qos, &qstr, &TT, &q16_null};
            const u32 tpw = T >= 128u ? 4u : 1u;
            launch_t(b, K_SALUKI_QKN_ROPE_KV_R11, (2u * nkv + 7u) >> 3u, (T + tpw - 1u) / tpw, akv11);
            break;
          }
          void *akv[] = {&Akv, &B0, &kc, &vc, &w0_null, &NW, &cs, &sn, &nq0, (void *)&nkv, (void *)&hd, &nrot, &eps, &ins, &qos, &qstr};
          launch_t(b, b->kvrot ? K_QKN_ROPE_KV_ROT : K_QKN_ROPE_KV, 2u * nkv, T, akv);
          break;
        }
        {
          static int q11_agmin = -1, q11_fa = -1, q11_pl_env = -1, q11_pl_min = 0, q11_fa_min_env = -2, q11_pl_q16 = -1;
          if (q11_agmin < 0) {
            q11_agmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
            q11_fa = !(getenv("IE_ATTN_FA") && getenv("IE_ATTN_FA")[0] == '0');
            q11_pl_env = !(getenv("IE_ATTN_PL") && getenv("IE_ATTN_PL")[0] == '0');
            q11_pl_min = getenv("IE_ATTN_PL_MIN") ? atoi(getenv("IE_ATTN_PL_MIN")) : 0;
            q11_fa_min_env = getenv("IE_ATTN_FA_MIN") ? atoi(getenv("IE_ATTN_FA_MIN")) : -1;
            q11_pl_q16 = !(getenv("IE_PL_Q16") && getenv("IE_PL_Q16")[0] == '0');
          }
          const int q11_pl_ok = b->has_pl && q11_pl_env && b->kvq8 && b->kvb != 4u && !b->kvq4 && hd == 256u && nhd / nkv <= 6u;
          const int q11_fa_min = q11_fa_min_env >= 0 ? q11_fa_min_env : (q11_pl_ok ? 64 : 1024);
          const u32 q11_qn = nq * 256u;
          const int fuse_q16 =
              !b->kv_only && !b->kvrot && !b->tr_on && !mk.rec && !b->attn_check && !gdump_on() && t == 0u &&
              T > 1u && (int)T >= q11_agmin && nq == nhd &&
              q11_fa && q11_pl_ok && (pos + T > (u32)q11_fa_min) && ((int)(pos + T) >= q11_pl_min) &&
              q11_pl_q16 && b->has_q_scale_f16 && b->xh && (size_t)T * q11_qn * 2u <= b->xh_bytes &&
              i + 1u < g->n_ops && g->ops[i + 1u].kind == OP_ATTN &&
              g->ops[i + 1u].a == o->b && g->ops[i + 1u].layer == o->layer &&
              o->boff == 0u && g->ops[i + 1u].aoff == 0u &&
              ((uintptr_t)B & 15u) == 0u && (qos & 15u) == 0u;
          void *Q16 = fuse_q16 ? b->xh : NULL;
          void *Bout = (fuse_q16 && deadb_on()) ? NULL : B;
          const int use_r11 = b->k[K_SALUKI_QKN_ROPE_KV_R11] && b->kvq8 && !b->kvrot && !mk.rec &&
                              hd == 256u && T > 0u && (nrot == 0u || nrot == 64u || nrot == 128u || nrot == 256u) &&
                              (qstr & 3u) == 0u && ((uintptr_t)A & 15u) == 0u && (T == 1u || (ins & 15u) == 0u) &&
                              ((uintptr_t)kc & 15u) == 0u && ((uintptr_t)vc & 15u) == 0u &&
                              ((uintptr_t)NW & 15u) == 0u && (nq == 0u || ((uintptr_t)w0 & 15u) == 0u) &&
                              (!Bout || (((uintptr_t)Bout & 15u) == 0u && (T == 1u || (qos & 15u) == 0u))) &&
                              (nrot == 0u || (((uintptr_t)cs & 15u) == 0u && ((uintptr_t)sn & 15u) == 0u));
          if (use_r11) {
            u32 TT = T;
            void *a11[] = {&A, &Bout, &kc, &vc, &w0, &NW, &cs, &sn, &nq, (void *)&nkv, (void *)&hd, &nrot, &eps, &ins, &qos, &qstr, &TT, &Q16};
            const u32 tpw = T >= 128u ? 4u : 1u;
            if (fuse_q16) b->q16_ready_op = (int)(i + 1u);
            launch_t(b, K_SALUKI_QKN_ROPE_KV_R11, (nq + 2u * nkv + 7u) >> 3u, (T + tpw - 1u) / tpw, a11);
            break;
          }
        }
        void *args[] = {&A, &B, &kc, &vc, &w0, &NW, &cs, &sn, &nq, (void *)&nkv, (void *)&hd, &nrot, &eps, &ins, &qos, &qstr};
        launch_t(b, b->kvrot ? K_QKN_ROPE_KV_ROT : K_QKN_ROPE_KV, nq + 2u * nkv, T, args);
        break;
      }
      case OP_GDN: {
        const layer *L = &m->l[o->layer];
        float *ring = b->ring + (size_t)L->sti * GDN_RING * m->conv_dim;
        float *st = b->st + (size_t)L->sti * b->ns * m->n_vh * m->sd * m->sd;
        u32 ns = b->ns;
        /* a verify run with records: the records of tokens 1 .. T - 1 of this layer */
        float *rec = NULL;
        if (b->rec && wfrom == 0u && T > 1u) {
          if (T > b->vmax) ie_die("a verify run of %u tokens; the linear attention records hold %u", T, b->vmax);
          rec = b->rec + (size_t)L->sti * (b->vmax - 1u) * m->n_vh * GDN_RECF;
          b->rec_n = T - 1u;
        }
        void *cw = b->d[i].w0, *dtb = b->d[i].w1, *sa = b->d[i].bias, *nw = b->d[i].nw;
        u32 cd = m->conv_dim, nk = m->n_kh, nv = m->n_vh;
        u32 is = g->bufs[o->a].stride / 4u, os = g->bufs[o->b].stride / 4u, qs = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
        /* a prompt chunk (wfrom > 0) may have more tokens than the ring has
         * slots: ie_gdn does not write the ring, ie_ring_store does after it */
        u32 rw = wfrom == 0;
        if (T <= 8u || (wfrom == 0u && T <= b->vmax)) { /* decode and verify: one kernel, the norm fused */
          void *trp = b->tr_on ? (void *)b->tr_dev : NULL;
          void *Bout = (o->dead_b && Q && !mk.rec && deadb_on()) ? NULL : B;
          void *lrec = b->lazy_mask ? (void *)b->lazy_rec : NULL;
          u32 lmask = b->lazy_mask, r_idx = L->sti;
          uint64_t lrls = b->lazy_rls;
          void *a1[] = {&A, &Bout, &Q, &ring, &st, &cw, &dtb, &sa, &nw, &pos, &cd, &nk, &nv, &eps, &T, &is, &os, &qs, &slot, &wfrom, &rw, &ns, &rec, &trp,
                        &lrec, &lmask, &lrls, &r_idx};
          /* IE_GDN1P=0: ie_gdn1 for every run. Verify runs of 2 .. 16 tokens: ie_gdn1p (not recorded into the megakernel) */
          static int gp = -1;
          if (gp < 0) gp = !(getenv("IE_GDN1P") && getenv("IE_GDN1P")[0] == '0');
          const int k1p = (gp || b->tr_on) && b->has_gdn1p && T >= 2u && T <= 16u && !mk.rec;
          if (b->tr_on && !k1p) ie_die("a tree verify needs ie_gdn1p");
          if (k1p) {
            const int saluki_gdn = b->k[K_SALUKI_GDN1P] && !trp && rec && T == 8u && nk == 16u && nv == 48u &&
                                   cd == 10240u && ns == 1u && slot == 0u && wfrom == 0u && rw == 1u;
            const int saluki_gdn_tr = b->k[K_SALUKI_GDN1P_TR] && trp && rec && T == 8u && nk == 16u && nv == 48u &&
                                      cd == 10240u && ns == 1u && slot == 0u && wfrom == 0u && rw == 1u;
            launch(b, saluki_gdn ? K_SALUKI_GDN1P : saluki_gdn_tr ? K_SALUKI_GDN1P_TR : K_GDN1P, nv, a1);
          }
          else {
            a1[23] = NULL; /* ie_gdn1: no tree argument */
            launch(b, K_GDN1, nv, a1);
          }
          if (!rw) {
            void *a2[] = {&A, &ring, &pos, &T, &cd, &is};
            launch(b, K_RING_STORE, (3u * cd + 255u) / 256u, a2);
          }
          break;
        }
        static int cmin = -1;
        if (cmin < 0) cmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
        if (b->cq_qk && wfrom == T - 1u && (int)T >= cmin && m->sd == 128u) {
          /* the chunked form: only the state after the last token is kept */
          static int gchk = -1, gdn_sc = -1;
          if (gchk < 0) gchk = getenv("IE_GDN_CHECK") != NULL;
          if (gdn_sc < 0) gdn_sc = getenv("IE_GDN_SCALAR") != NULL && getenv("IE_GDN_SCALAR")[0] != '0';
          float *ref = NULL, *sref = NULL;
          const size_t ob = (size_t)T * os * 4u, sb = (size_t)nv * 16384u * 4u;
          float *sw = st + (size_t)((slot + T - 1u) % ns) * nv * 16384u;
          /* debug: the sequential kernel first, its output and state kept (not
           * when the state after the chunk goes to the slot it is read from) */
          if (gchk && (slot + T - 1u) % ns == slot) gchk = 2;
          else if (gchk == 2) gchk = 1;
          if (gchk == 1) {
            void *args[] = {&A, &B, &ring, &st, &cw, &dtb, &sa, &pos, &cd, &nk, &nv, &eps, &T, &is, &os, &slot, &wfrom, &rw, &ns, &rec};
            launch(b, K_GDN, nv * 4u, args);
            ref = malloc(ob), sref = malloc(sb);
            HIP(H.Memcpy(ref, B, ob, hipMemcpyDeviceToHost));
            HIP(H.Memcpy(sref, sw, sb, hipMemcpyDeviceToHost));
          }
          if (b->has_gdn_wmma && !gdn_sc && !mk.rec) {
            void *p1[] = {&A, &ring, &cw, &dtb, &sa, &pos, &T, &cd, &nk, &nv, &eps, &is, &b->cq_qk, &b->cq_v, &b->cq_bg};
            launch(b, K_GDN_PREP_RING, 2u * nk + nv, p1);
            void *p2[] = {&b->cq_qk, &b->cq_v, &b->cq_bg, &b->cq_u, &b->cq_w, &b->cq_m, &b->cq_g, &T, &nk, &nv};
            if (b->k[K_SALUKI_GDN_WY_WMMA_R11] && !mk.rec && nk == 16u && nv == 48u) {
              u32 gr = ((T + 31u) / 32u) * 16u;
              if (gr >= 253u && gr <= 256u) gr = 257u;
              launch(b, K_SALUKI_GDN_WY_WMMA_R11, gr, p2);
            } else {
              launch(b, K_GDN_WY_WMMA, nv * ((T + 31u) / 32u), p2);
            }
            const int saluki_gdn_seq = b->k[K_SALUKI_GDN_SEQ_WMMA_GNORM] && !mk.rec && nk == 16u && nv == 48u && cd == 10240u;
            if (gchk != 1) {
              void *Bout = (o->dead_b && Q && deadb_on()) ? NULL : B;
              void *p3[] = {&b->cq_qk, &b->cq_u, &b->cq_w, &b->cq_m, &b->cq_g, &A, &nw, &Bout, &Q, &cd, &eps, &is, &os, &qs, &st, &slot, &ns, &pos, &T, &nk, &nv};
              launch(b, saluki_gdn_seq ? K_SALUKI_GDN_SEQ_WMMA_GNORM : K_GDN_SEQ_WMMA_GNORM, nv, p3);
              break;
            }
            void *nw0 = NULL, *oq0 = NULL;
            void *p3[] = {&b->cq_qk, &b->cq_u, &b->cq_w, &b->cq_m, &b->cq_g, &A, &nw0, &B, &oq0, &cd, &eps, &is, &os, &qs, &st, &slot, &ns, &pos, &T, &nk, &nv};
            launch(b, saluki_gdn_seq ? K_SALUKI_GDN_SEQ_WMMA_GNORM : K_GDN_SEQ_WMMA_GNORM, nv, p3);
            rw = 1u;
          } else {
            void *p1[] = {&A, &ring, &cw, &dtb, &sa, &pos, &cd, &nk, &nv, &eps, &is, &b->cq_qk, &b->cq_v, &b->cq_bg};
            launch_t(b, K_GDN_PREP, 1, T, p1);
            void *p2[] = {&b->cq_qk, &b->cq_v, &b->cq_bg, &b->cq_u, &b->cq_w, &b->cq_m, &b->cq_g, &T, &nk, &nv};
            launch(b, K_GDN_WY, nv * ((T + 31u) / 32u), p2);
            void *p3[] = {&b->cq_qk, &b->cq_u, &b->cq_w, &b->cq_m, &b->cq_g, &B, &os, &st, &slot, &ns, &pos, &T, &nk, &nv};
            launch(b, K_GDN_SEQ, nv * 4u, p3); /* 4 workgroups per value head (GD_SPLIT) */
          }
          if (gchk == 1) {
            float *x = malloc(ob), *y = malloc(sb);
            HIP(H.Memcpy(x, B, ob, hipMemcpyDeviceToHost));
            HIP(H.Memcpy(y, sw, sb, hipMemcpyDeviceToHost));
            double eo = 0, mo = 0, es = 0, ms = 0;
            for (u32 t = 0; t < T; t++)
              for (u32 k = 0; k < nv * 128u; k++) {
                const double a = ref[(size_t)t * os + k], d = fabs(a - x[(size_t)t * os + k]);
                if (d > eo) eo = d;
                if (fabs(a) > mo) mo = fabs(a);
              }
            for (size_t k = 0; k < sb / 4u; k++) {
              const double d = fabs(sref[k] - y[k]);
              if (d > es) es = d;
              if (fabs(sref[k]) > ms) ms = fabs(sref[k]);
            }
            fprintf(stderr, "gdn check layer %u T %u pos %u: out %.3g / %.3g, state %.3g / %.3g\n", o->layer, T, pos, eo, mo, es, ms);
            free(x), free(y), free(ref), free(sref);
          }
        } else {
          void *args[] = {&A, &B, &ring, &st, &cw, &dtb, &sa, &pos, &cd, &nk, &nv, &eps, &T, &is, &os, &slot, &wfrom, &rw, &ns, &rec};
          launch(b, K_GDN, nv * 4u, args); /* verify runs of 9 tokens and more, prompt chunks: 4 workgroups per value head (GD_SPLIT) */
        }
        if (!rw) {
          void *a2[] = {&A, &ring, &pos, &T, &cd, &is};
          launch(b, K_RING_STORE, (3u * cd + 255u) / 256u, a2);
        }
        u32 isb = is * 4u, osb = os * 4u; /* the gated norm, all tokens: grid y */
        u32 nv_gn = (o->dead_b && Q && !mk.rec && deadb_on()) ? (nv | 0x80000000u) : nv;
        void *a3[] = {&B, &A, &nw, &Q, &nv_gn, &cd, &eps, &isb, &osb, &qs};
        launch_t(b, K_GNORM, (nv + 7u) / 8u, T, a3);
        break;
      }
      case OP_KV: {
        const u32 kvi = m->arch == ARCH_INKLING2 ? o->kvi : m->l[o->layer].kvi;
        if (T > 1u || b->kvq8 || b->kvq4) {
          void *kc = kv_ptr(b, 0, kvi, 0), *vc = kv_ptr(b, 1, kvi, 0);
          u32 kvlen = g->kv_len[kvi], ks = g->bufs[o->a].stride, vs = g->bufs[o->b].stride;
          void *args[] = {&A, &B, &kc, &vc, (void *)&hd, (void *)&nkv, &pos, &kvlen, &ks, &vs};
          launch_t(b, K_KV_T, elem_groups, T, args);
        } else if (b->ink_graph_cap && b->k[K_INK_KV_STORE_CTX_R2] && !b->kvq8 && !b->kvq4 &&
                   g->kv_len[kvi] > 0u && (n & 3u) == 0u) {
          void *kc0 = kv_ptr(b, 0, kvi, 0), *vc0 = kv_ptr(b, 1, kvi, 0);
          u32 kvlen = g->kv_len[kvi];
          void *ca[] = {&A, &B, &kc0, &vc0, &kvlen, &n, &b->ink_ctx_dev};
          launch(b, K_INK_KV_STORE_CTX_R2, ((n >> 2u) + 255u) / 256u, ca);
        } else {
          void *kc = kv_ptr(b, 0, kvi, pos % g->kv_len[kvi]), *vc = kv_ptr(b, 1, kvi, pos % g->kv_len[kvi]);
          void *args[] = {&A, &B, &kc, &vc, &n};
          launch(b, K_KV, elem_groups, args);
        }
        break;
      }
      case OP_ATTN: {
        const int q16_pre = (b->q16_ready_op == (int)i);
        b->q16_ready_op = -1;
        const u32 kvi = m->arch == ARCH_INKLING2 ? o->kvi : m->l[o->layer].kvi;
        void *kc = kv_ptr(b, 0, kvi, 0), *vc = kv_ptr(b, 1, kvi, 0);
        if (m->arch == ARCH_FLINT || m->arch == ARCH_RICURSIVE) { /* one token: window, ring, sinks (ie_attn_sw) */
          if (T > 1u) { /* a prompt chunk: ie_attn_fa_sw (window, ring, sinks), then the Q8 copy */
            u32 len = g->kv_len[kvi], win = o->win, qs = g->bufs[o->a].stride, os = g->bufs[o->b].stride;
            void *SK = b->d[i].w1;
            if (hd > 128u || hd % 32u) ie_die("flint chunk attention: hd %u", hd);
            void *fa[] = {&A, &kc, &vc, &B, &pos, (void *)&hd, (void *)&nhd, (void *)&nkv, &qs, &os, &T, &win, &len, &SK};
            launch(b, K_ATTN_FA_SW, nkv * ((T + 15u) / 16u) * ((nhd / nkv + 7u) / 8u), fa);
            if (Q) {
              u32 nb = nhd * hd / 32u, qst = g->bufs[o->qo].stride;
              void *qa[] = {&B, &Q, &nb, &os, &qst};
              launch_t(b, K_QUANT, (nb + 7u) / 8u, T, qa);
            }
            break;
          }
          u32 len = g->kv_len[kvi], p0 = o->win && pos + 1u > o->win ? pos + 1u - o->win : 0u, np = pos + 1u - p0;
          /* ie_attn_gq (the q heads of a kv head together, chunks of 256
           * positions) when it can; IE_ATTN_SW=1: ie_attn_sw */
          const int gq = gq_use(b, pos) && nhd / nkv <= 8u && hd <= 128u && hd % 8u == 0u;
          /* ie_attn_gq: passes of 256 positions; at most gq_s splits per kv head
           * (long positions: fewer, longer splits, a cheap merge) */
          u32 ch = gq ? b->gq_ch : ie_att_ch(pos);
          if (gq && !b->gq_ch_fixed) {
            const u32 per = (np + b->gq_s - 1u) / b->gq_s;
            ch = per <= 256u ? 256u : (per + 255u) & ~255u;
          }
          u32 nsplit = (np + ch - 1u) / ch;
          void *SK = b->d[i].w1; /* the sinks (gpu_open), or NULL */
          void *args[] = {&A, &kc, &vc, &C, &B, &p0, &np, &len, (void *)&hd, (void *)&nhd, (void *)&nkv, &nsplit, &ch,
                          &b->at_count, &Q, &SK};
          launch(b, gq ? K_ATTN_GQ : K_ATTN_SW, (gq ? nkv : nhd) * nsplit, args);
          break;
        }
        /* IE_KV_ROT=1: the attention kernels run without the gate and the Q8 copy; ie_attn_unrot after them rotates
         * the output back and applies both */
        const int ogt = o->gt >= 0 && !b->kvrot;
        void *const Qr = Q;
        const u32 Tr = T; /* ie_attn_long may leave T at its lower tokens */
        const int skip_att_b = o->dead_b && Qr && !mk.rec && !b->kvrot && !b->attn_check && deadb_on();
        void *Bout = skip_att_b ? NULL : B;
        do {
        void *Q = b->kvrot ? NULL : Qr;
        void *GT = ogt ? b->arena + g->bufs[o->gt].off + o->gtoff + (size_t)t * g->bufs[o->gt].stride : NULL;
        u32 gs = o->gstride;
        static int agmin = -1;
        if (agmin < 0) agmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
        if (T > 1 && (int)T >= agmin) { /* a prompt chunk on the fast path: all tokens in one launch */
          GT = ogt ? b->arena + g->bufs[o->gt].off + o->gtoff : NULL;
          u32 qst = g->bufs[o->a].stride, ost = g->bufs[o->b].stride, oqst = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
          u32 gst = o->gt >= 0 ? g->bufs[o->gt].stride : 0u;
          /* ie_attn_fa (fp16 WMMA flash attention; IE_ATTN_FA=0: off): K/V read once per 16 tokens and
           * group of query heads, Q K^T and P V on the matrix units */
          static int fa = -1, pl = -1, pl_min = 0, fa_min_env = -2;
          if (fa < 0) {
            fa = !(getenv("IE_ATTN_FA") && getenv("IE_ATTN_FA")[0] == '0');
            pl = b->has_pl && !(getenv("IE_ATTN_PL") && getenv("IE_ATTN_PL")[0] == '0');
            pl_min = getenv("IE_ATTN_PL_MIN") ? atoi(getenv("IE_ATTN_PL_MIN")) : 0;
            fa_min_env = getenv("IE_ATTN_FA_MIN") ? atoi(getenv("IE_ATTN_FA_MIN")) : -1;
          }
          const int pl_ok = pl && b->kvb != 4u && !b->kvq4 && hd == 256u && nhd / nkv <= 6u;
          const int fa_min = fa_min_env >= 0 ? fa_min_env : (pl_ok ? 64 : 1024);
          if (fa && pos + T > (u32)fa_min && nhd / nkv <= 8u && hd % 32u == 0u && hd <= 256u) {
            u32 TT = T;
            const u32 ntile = (T + 15u) / 16u, nparts = (hd + 127u) / 128u;
            /* ie_attn_fa2 (IE_ATTN_FA2=0: off): S once per block for all columns, the output tiles on all 8 waves */
            static int fa2 = -1;
            if (fa2 < 0) fa2 = !(getenv("IE_ATTN_FA2") && getenv("IE_ATTN_FA2")[0] == '0');
            if (pl_ok && (int)(pos + T) >= pl_min) {
              void *Q16 = NULL;
              u32 qn = nhd * 256u;
              if (q16_pre) {
                Q16 = b->xh;
              } else if (b->has_q_scale_f16 && b->xh && (size_t)TT * qn * 2u <= b->xh_bytes &&
                  ((uintptr_t)A & 15u) == 0u && (qst & 15u) == 0u && !(getenv("IE_PL_Q16") && getenv("IE_PL_Q16")[0] == '0')) {
                Q16 = b->xh;
                void *qa[] = {&A, &Q16, &qn, &qst};
                launch_t(b, K_Q_SCALE_F16, (qn + 1023u) / 1024u, TT, qa);
              }
              void *apl[] = {&A, &kc, &vc, &Bout, &pos, (void *)&hd, (void *)&nhd, (void *)&nkv, &GT, &gs, &qst, &ost, &gst, &TT, &Q, &oqst, &Q16};
              const int use_saluki_pl_r10 = b->k[K_SALUKI_ATTN_PL_R10] && b->kvq8 && !mk.rec && hd == 256u && nhd == 24u && nkv == 4u && ntile == 32u;
              const int use_saluki_pl = b->k[K_SALUKI_ATTN_PL] && b->kvq8 && !mk.rec && hd == 256u && nhd == 24u && nkv == 4u;
              const u32 pl_g = nkv * ntile;
              launch(b, use_saluki_pl_r10 ? K_SALUKI_ATTN_PL_R10 : use_saluki_pl ? K_SALUKI_ATTN_PL : K_ATTN_PL,
                     (use_saluki_pl_r10 || use_saluki_pl) && pl_g >= 253u && pl_g <= 256u ? 257u : pl_g, apl);
            } else {
              void *a1[] = {&A, &kc, &vc, &B, &pos, (void *)&hd, (void *)&nhd, (void *)&nkv, &GT, &gs, &qst, &ost, &gst, &TT};
              if (fa2 && hd == 256u && nhd / nkv <= 6u) launch(b, K_ATTN_FA2, nkv * ntile, a1);
              else launch(b, K_ATTN_FA, nkv * nparts * ntile * ((nhd / nkv + 7u) / 8u), a1);
              if (Q) {
                u32 nbq = nhd * hd / 32u;
                void *a2[] = {&B, &Q, &nbq, &ost, &oqst};
                launch_t(b, K_ATTN_FA_Q8, (nbq + 7u) / 8u, T, a2);
              }
            }
            break;
          }
          void *args[] = {&A, &kc, &vc, &Bout, &pos, (void *)&hd, (void *)&nhd, (void *)&nkv, &Q, &GT, &gs, &qst, &ost, &oqst, &gst};
          if (nhd / nkv <= 8u) launch_t(b, K_ATTN_PFG, nkv, T, args); /* K/V read once per group of query heads */
          else launch_t(b, K_ATTN_PF, nhd, T, args);
          break;
        }
        /* decode and verify at long positions (or any tree/MTP frontier run): ie_attn_long (kernels/ie_kernels.c). */
        if (b->attn_long && T <= b->at_tmax && (pos + T > b->long_min || b->tr_on)) {
          const u32 j0 = (b->tr_on || pos >= b->long_min) ? 0u : b->long_min - pos;
          u32 nhv = nhd, nkvv = nkv, gsv = gs, gmx = b->at_gmax;
          const u32 grp = nhd / nkv;
          u32 qst = g->bufs[o->a].stride, ost = g->bufs[o->b].stride, oqst = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
          u32 gst = o->gt >= 0 ? g->bufs[o->gt].stride : 0u;
          const size_t area = (size_t)b->at_lstride * b->at_ltmax; /* floats of the part area (T > 1) */
          static int mwin = -1;
          if (mwin < 0) mwin = getenv("IE_MTP_DRAFT_WIN") ? atoi(getenv("IE_MTP_DRAFT_WIN")) : 2048;
          for (u32 s = j0, Tl = 0; s < T; s += Tl) {
            const int use_atp = (T > 1u || b->tr_on) && b->at_part != NULL;
            u32 p0 = pos + s, pst = use_atp ? b->at_lstride : 0u, tmx = b->at_ltmax;
            void *kcl = kc, *vcl = vc;
            { /* IE_DBG_ATL_WIN=OFF,LEN or OFF:LEN (debug, wrong outputs): attention over the LEN positions from OFF only */
              static long woff = -2, wlen = 0;
              if (woff == -2) {
                const char *e = getenv("IE_DBG_ATL_WIN");
                woff = -1;
                if (e && sscanf(e, "%ld%*c%ld", &woff, &wlen) != 2) woff = -1;
              }
              if (woff >= 0 && wlen > 0 && (u32)(woff + wlen) <= pos) {
                kcl = (char *)kc + (size_t)woff * b->kvrowb, vcl = (char *)vc + (size_t)woff * b->kvrowb;
                p0 = (u32)wlen + s;
              }
            }
            Tl = T - s < b->at_tmax ? T - s : b->at_tmax;
            const u32 wbeg = (o->layer == (int)m->n_layer && hd == 256u && mwin > 0 && p0 > 4096u && p0 > 64u + (u32)mwin) ? p0 - (u32)mwin : 0u;
            const u32 wskip = wbeg >= 256u ? wbeg / IE_ATL_CH - 1u : 0u;
            const u32 kspan0 = b->tr_on ? (u32)b->tr_tab[IE_TR_KEND] : Tl;
            if (use_atp && b->at_lpack) { /* the stride of these positions' splits and groups (at most at_lstride) */
              const u32 sm = (p0 + kspan0 - 1u) / IE_ATL_CH + 1u - wskip;
              pst = nhd * (sm + (sm + IE_ATL_G - 1u) / IE_ATL_G) * IE_ATL_PSZ;
              const size_t fit = area / pst;
              tmx = fit < b->at_tmax ? (u32)fit : b->at_tmax;
            }
            if (Tl > tmx) Tl = tmx;
            const u32 kspan = b->tr_on ? (u32)b->tr_tab[IE_TR_KEND] : Tl;
            const u32 smx = (p0 + kspan - 1u) / IE_ATL_CH + 1u - wskip;
            const u32 npg = (grp * Tl + 15u) / 16u;
            u32 npg_arg = npg | (wbeg << 5);
            if (!Tl || !kspan || p0 + kspan > g->n_ctx || (use_atp && ((size_t)nhd * (smx + (smx + IE_ATL_G - 1u) / IE_ATL_G) * IE_ATL_PSZ > pst ||
                                                                       (size_t)pst * Tl > area)))
              ie_die("ie_attn_long: positions %u .. %u, %u tokens: beyond the context (%u) or the part area", p0, p0 + kspan - 1u, Tl, g->n_ctx);
            void *GL = ogt ? (T == 1u ? GT : b->arena + g->bufs[o->gt].off + o->gtoff) : NULL;
            if (GL) GL = (char *)GL + (size_t)s * gst;
            void *QL = (char *)A + (size_t)s * qst, *OL = (skip_att_b && hd == 256u) ? NULL : (char *)B + (size_t)s * ost, *OQ = Q ? (char *)Q + (size_t)s * oqst : NULL;
            void *PL = use_atp ? (void *)b->at_part : C;
            void *CL = b->at_count + (size_t)s * nhd;
            void *GC = b->at_gcount + (size_t)s * nhd * b->at_gmax;
            void *trp = b->tr_on ? (void *)b->tr_dev : NULL;
            if (b->tr_on && (s || Tl != T)) ie_die("a tree verify in more than one ie_attn_long launch");
            void *args[] = {&QL, &kcl, &vcl, &PL, &OL, &p0, &nhv, &nkvv, &Tl, &npg_arg, &CL, &GC, &gmx, &OQ, &GL, &gsv, &qst, &ost, &oqst, &gst, &pst, &trp};
            static int saluki_mp_min = -1;
            if (saluki_mp_min < 0) {
              const char *e = getenv("IE_SALUKI_ATTN_MIN_POS");
              saluki_mp_min = e && *e ? atoi(e) : 2048;
              if (saluki_mp_min < 0) saluki_mp_min = 0;
            }
            const int use_mp = b->kvq8 && b->has_long_mp && hd == 256u && npg >= 2u && npg <= 3u && atl_mp_on();
            const int use_saluki_mp_r11 = use_mp && b->k[K_SALUKI_ATTN_LONG_MP_R11] && !mk.rec && !wbeg && use_atp &&
                                          Tl == 8u && nhv == 24u && nkvv == 4u && p0 >= (u32)saluki_mp_min;
            const int use_saluki_mp_r10 = use_mp && b->k[K_SALUKI_ATTN_LONG_MP_R10] && !mk.rec && !trp && !wbeg &&
                                          Tl == 8u && nhv == 24u && nkvv == 4u && p0 >= (u32)saluki_mp_min;
            const int use_saluki_mp_p1 = use_mp && b->k[K_SALUKI_ATTN_LONG_MP_PIPE1] && !mk.rec && !trp && !wbeg &&
                                         Tl == 8u && nhv == 24u && nkvv == 4u && p0 >= (u32)saluki_mp_min;
            const int use_saluki_mp = use_mp && b->k[K_SALUKI_ATTN_LONG_MP] && !mk.rec && !trp && !wbeg &&
                                      Tl == 8u && nhv == 24u && nkvv == 4u && p0 >= (u32)saluki_mp_min;
            const int use_saluki_mp_tr_r10 = use_mp && b->k[K_SALUKI_ATTN_LONG_MP_TR_R10] && !mk.rec && trp && !wbeg &&
                                             Tl == 8u && nhv == 24u && nkvv == 4u && p0 >= (u32)saluki_mp_min;
            const int use_saluki_mp_tr_ns = use_mp && b->k[K_SALUKI_ATTN_LONG_MP_TR_NOSPILL] && !mk.rec && trp && !wbeg &&
                                            Tl == 8u && nhv == 24u && nkvv == 4u && p0 >= (u32)saluki_mp_min;
            const int use_saluki_mp_tr = use_mp && b->k[K_SALUKI_ATTN_LONG_MP_TR] && !mk.rec && trp && !wbeg &&
                                         Tl == 8u && nhv == 24u && nkvv == 4u && p0 >= (u32)saluki_mp_min;
            const u32 nwg = use_mp ? nkv * smx : nkv * smx * npg;
            const u32 mp_nwg = (use_saluki_mp_r11 || use_saluki_mp_r10 || use_saluki_mp_tr_r10) && nwg >= 253u && nwg <= 256u ? 257u : nwg;
            { /* IE_DBG_ATL=N: print the first N launches' arguments (debug) */
              static int dn = -1;
              if (dn < 0) dn = getenv("IE_DBG_ATL") ? atoi(getenv("IE_DBG_ATL")) : 0;
              static u32 dfrom = 0;
              if (dn < 0 || !dfrom) dfrom = getenv("IE_DBG_ATL_FROM") ? (u32)atoi(getenv("IE_DBG_ATL_FROM")) : 1u;
              if (dn > 0 && p0 >= dfrom) {
                dn--;
                fprintf(stderr, "atl%s: p0 %u Tl %u npg %u smx %u nwg %u nh %u nkv %u gmx %u pst %u qst %u ost %u oqst %u gst %u gs %u "
                        "q %p k %p v %p part %p out %p cnt %p gc %p oq %p gate %p\n", use_mp ? "_mp" : "", p0, Tl, npg, smx, nwg, nhv, nkvv, gmx, pst, qst, ost,
                        oqst, gst, gsv, QL, kc, vc, PL, OL, CL, GC, OQ, GL);
              }
            }
            launch(b, use_mp ? (use_saluki_mp_r11 ? K_SALUKI_ATTN_LONG_MP_R11 :
                                use_saluki_mp_r10 ? K_SALUKI_ATTN_LONG_MP_R10 :
                                use_saluki_mp_p1 ? K_SALUKI_ATTN_LONG_MP_PIPE1 :
                                use_saluki_mp ? K_SALUKI_ATTN_LONG_MP :
                                use_saluki_mp_tr_r10 ? K_SALUKI_ATTN_LONG_MP_TR_R10 :
                                use_saluki_mp_tr_ns ? K_SALUKI_ATTN_LONG_MP_TR_NOSPILL :
                                use_saluki_mp_tr ? K_SALUKI_ATTN_LONG_MP_TR : K_ATTN_LONG_MP) :
                              (b->has_long_e && nwg < 64u && atl_e_on() ? K_ATTN_LONG_E : K_ATTN_LONG), mp_nwg, args);
          }
          if (j0 == 0u) break;
          T = j0; /* the lower tokens: the kernels below */
        }
        /* decode and verify: ie_attn_kv (bitwise equal to ie_attn_split per
         * token) from IE_ATTN_KV_MIN positions (default 768). Measured on the
         * R9700 (27B, one token, ms/token in attention, ie_attn_kv vs
         * ie_attn_split): 100 positions 0.67 vs 0.39, 600: 0.91 vs 0.88,
         * 900: 1.09 vs 1.09, 7300: 3.44 vs 7.43. */
        static int kv_min = -1;
        if (kv_min < 0) kv_min = getenv("IE_ATTN_KV_MIN") ? atoi(getenv("IE_ATTN_KV_MIN")) : 768;
        if (b->tr_on) ie_die("a tree verify below ie_attn_long's positions");
        if (b->attn_kv && T <= b->at_tmax && pos + T > (u32)kv_min) {
          u32 ch = ie_att_ch(pos), smax = ie_att_nsplit(pos + T - 1u);
          if (ie_att_ch(pos + T - 1u) != ch) ie_die("ie_attn_kv: the chunk depends on the position");
          if (T == 1u) GT = ogt ? b->arena + g->bufs[o->gt].off + o->gtoff + (size_t)t * g->bufs[o->gt].stride : NULL;
          else GT = ogt ? b->arena + g->bufs[o->gt].off + o->gtoff : NULL;
          u32 qst = g->bufs[o->a].stride, ost = g->bufs[o->b].stride, oqst = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
          u32 gst = o->gt >= 0 ? g->bufs[o->gt].stride : 0u;
          /* one token: the part area is C (as ie_attn_split); more: at_part, at_pstride floats per token */
          void *P = T == 1u ? C : (void *)b->at_part;
          u32 pst = T == 1u ? 0u : b->at_pstride, TT = T, nb = (nhd / nkv * T + 7u) / 8u;
          /* splits per workgroup: about 128 workgroups (2 per CU). Measured on the
           * R9700 (27B, one token): at 7300 positions 8 splits per workgroup
           * (116 workgroups) 3.44 ms/token in attention, 1 split 4.42; at 900
           * positions 1 split (116) 1.09, 8 splits 2.21. It does not change the result. */
          static int spw_env = -1;
          if (spw_env < 0) spw_env = getenv("IE_ATTN_SPW") ? atoi(getenv("IE_ATTN_SPW")) : 0;
          u32 spw = spw_env > 0 ? (u32)spw_env : (u32)(smax * nkv * nb / 128u);
          if (spw < 1u) spw = 1u;
          if (spw > 16u) spw = 16u;
          const u32 ngs = (smax + spw - 1u) / spw;
          void *args[] = {&A, &kc, &vc, &P, &Bout, &pos, (void *)&hd, (void *)&nhd, (void *)&nkv, &smax, &ch,
                          &b->at_count, &Q, &GT, &gs, &qst, &ost, &oqst, &gst, &pst, &TT, &nb, &spw};
          launch(b, K_ATTN_KV, nkv * ngs * nb, args);
          break;
        }
        if (b->attn_split && T > 1) { /* verify: all tokens in one launch (ie_attn_split_t: bitwise equal per token) */
          u32 ch = ie_att_ch(pos), smax = ie_att_nsplit(pos + T - 1u);
          if (ie_att_ch(pos + T - 1u) != ch) ie_die("ie_attn_split_t: the chunk depends on the position");
          if (T > b->at_tmax) ie_die("ie_attn_split_t: %u tokens, room for %u", T, b->at_tmax);
          u32 qst = g->bufs[o->a].stride, ost = g->bufs[o->b].stride, oqst = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
          u32 gst = o->gt >= 0 ? g->bufs[o->gt].stride : 0u, pst = b->at_pstride;
          GT = ogt ? b->arena + g->bufs[o->gt].off + o->gtoff : NULL;
          void *args[] = {&A, &kc, &vc, &b->at_part, &Bout, &pos, (void *)&hd, (void *)&nhd, (void *)&nkv, &smax, &ch,
                          &b->at_count, &Q, &GT, &gs, &qst, &ost, &oqst, &gst, &pst};
          launch_t(b, b->has_split_tp && ch <= 32u && asp_on() ? K_ATTN_SPLIT_TP : K_ATTN_SPLIT_T, nhd * smax, T, args);
          break;
        }
        if (b->attn_split) {
          u32 ch = ie_att_ch(pos), nsplit = ie_att_nsplit(pos);
          void *args[] = {&A, &kc, &vc, &C, &Bout, &pos, (void *)&hd, (void *)&nhd, (void *)&nkv, &nsplit, &ch, &b->at_count, &Q, &GT, &gs};
          launch(b, K_ATTN_SPLIT, nhd * nsplit, args);
          if (b->attn_check) {
            void *nq = NULL;
            void *a2[] = {&A, &kc, &vc, &C, &b->chk, &pos, (void *)&nctx, (void *)&hd, (void *)&nhd, (void *)&nkv, &nq, &GT, &gs};
            launch(b, K_ATTN, nhd, a2);
            const size_t nb = (size_t)nhd * hd * 4u;
            HIP(H.Memcpy(b->chk_a, B, nb, hipMemcpyDeviceToHost));
            HIP(H.Memcpy(b->chk_b, b->chk, nb, hipMemcpyDeviceToHost));
            double mx = 0, d = 0;
            for (u32 j = 0; j < nhd * hd; j++) {
              const double x = b->chk_b[j], y = b->chk_a[j];
              if (fabs(x) > mx) mx = fabs(x);
              if (fabs(x - y) > d) d = fabs(x - y);
            }
            if (mx > 0 && d / mx > b->chk_worst) b->chk_worst = d / mx;
            b->chk_n++;
          }
        } else {
          void *args[] = {&A, &kc, &vc, &C, &Bout, &pos, (void *)&nctx, (void *)&hd, (void *)&nhd, (void *)&nkv, &Q, &GT, &gs};
          launch(b, K_ATTN, nhd, args);
        }
        } while (0);
        if (b->kvrot) {
          void *GU = o->gt >= 0 ? b->arena + g->bufs[o->gt].off + o->gtoff + (Tr == 1u ? (size_t)t * g->bufs[o->gt].stride : 0u) : NULL;
          u32 gsu = o->gstride, ost = g->bufs[o->b].stride, oqst = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
          u32 gst = o->gt >= 0 ? g->bufs[o->gt].stride : 0u;
          void *QU = Qr;
          void *args[] = {&B, &QU, &GU, (void *)&hd, (void *)&nhd, &gsu, &ost, &oqst, &gst};
          launch_t(b, K_ATTN_UNROT, nhd, Tr, args);
        }
        break;
      }
      case OP_VCLIP: {
        u32 nh = o->nh, qhd = o->qstr, norope = o->norope, win = o->win;
        u32 as = g->bufs[o->a].stride, bs = g->bufs[o->b].stride;
        float clamp = o->clamp, esc = o->escale;
        if (b->ink_graph_cap && T == 1u && b->k[K_INK_VCLIP_CTX_R2] && !mk.rec && (qhd == 128u || qhd == 16u) &&
            ((uintptr_t)A & 15u) == 0u && ((uintptr_t)B & 15u) == 0u && (!w0 || ((uintptr_t)w0 & 15u) == 0u)) {
          u32 gx = (qhd == 128u) ? ((nh + 7u) / 8u) : ((nh + 63u) / 64u);
          if (gx > 256u) gx = 256u;
          if (!gx) gx = 1u;
          void *va[] = {&A, &B, &w0, &nh, &qhd, &clamp, &norope, &esc, &win, &b->ink_ctx_dev};
          launch(b, K_INK_VCLIP_CTX_R2, gx, va);
        } else if (b->k[K_INK_VCLIP] && mk_ok_k(K_INK_VCLIP) && (qhd == 128u || qhd == 16u) &&
            ((uintptr_t)A & 15u) == 0u && ((uintptr_t)B & 15u) == 0u && (!w0 || ((uintptr_t)w0 & 15u) == 0u) &&
            (T == 1u || (((as | bs) & 15u) == 0u))) {
          u32 TT = T, total_h = T * nh;
          u32 gx = (qhd == 128u) ? ((total_h + 7u) / 8u) : ((total_h + 63u) / 64u);
          if (gx > 256u) gx = 256u;
          if (!gx) gx = 1u;
          void *va[] = {&A, &B, &w0, &nh, &qhd, &clamp, &norope, &esc, &win, &pos, &TT, &as, &bs};
          launch(b, K_INK_VCLIP, gx, va);
        } else {
          void *args[] = {&A, &B, &w0, &nh, &qhd, &clamp, &norope, &esc, &win, &pos, &as, &bs};
          launch_t(b, K_VCLIP, (nh + 7u) / 8u, T, args);
        }
        break;
      }
      case OP_ATTN_RES: {
        u32 M = o->nh, K = o->qstr, as = o->a >= 0 ? g->bufs[o->a].stride : 0u, cs = o->c >= 0 ? g->bufs[o->c].stride : 0u,
            bs = g->bufs[o->b].stride;
        int has_c = o->c >= 0;
        float *Qp = (float *)w0 + o->roff;
        void *args[] = {&A, &C, &Qp, &B, &M, &K, &n, &eps, &as, &cs, &bs, &has_c};
        const int ok_ar = b->k[K_INK_ATTN_RES] && mk_ok_k(K_INK_ATTN_RES) && M >= 1u && M <= 64u && K >= 1u && K <= 8u && n >= 1u &&
                          ((n & 3u) != 0u || ((!A || ((uintptr_t)A & 15u) == 0u) && (!has_c || ((uintptr_t)C & 15u) == 0u) &&
                                              ((uintptr_t)Qp & 15u) == 0u && ((uintptr_t)B & 15u) == 0u &&
                                              (T == 1u || ((!A || (as & 15u) == 0u) && (!has_c || (cs & 15u) == 0u) && (bs & 15u) == 0u))));
        launch_t(b, ok_ar ? K_INK_ATTN_RES : K_ATTN_RES, 1u, T, args);
        break;
      }
      case OP_LOOP_GATE: {
        void *GT = o->gt >= 0 ? b->arena + g->bufs[o->gt].off + o->gtoff + (T == 1u ? (size_t)t * g->bufs[o->gt].stride : 0u) : NULL;
        u32 as = g->bufs[o->a].stride, cs = g->bufs[o->c].stride, gs = g->bufs[o->gt].stride, bs = g->bufs[o->b].stride;
        if (b->k[K_INK_LOOP_GATE] && mk_ok_k(K_INK_LOOP_GATE) && (n & 3u) == 0u &&
            ((uintptr_t)A & 15u) == 0u && ((uintptr_t)C & 15u) == 0u && ((uintptr_t)GT & 15u) == 0u && ((uintptr_t)B & 15u) == 0u &&
            (T == 1u || (((as | cs | gs | bs) & 15u) == 0u))) {
          u32 TT = T, total4 = T * (n >> 2u);
          u32 gx = (T == 1u) ? ((total4 + 255u) / 256u) : ((total4 + 1023u) / 1024u);
          if (!gx) gx = 1u;
          void *la[] = {&A, &C, &GT, &B, &n, &TT, &as, &cs, &gs, &bs};
          launch(b, K_INK_LOOP_GATE, gx, la);
        } else {
          void *args[] = {&A, &C, &GT, &B, &n, &as, &cs, &gs, &bs};
          launch_t(b, K_LOOP_GATE, elem_groups, T, args);
        }
        break;
      }
      case OP_SCONV: {
        float *ring = b->sc_ring + (size_t)o->scoff * 4u;
        u32 nh = o->nh, as = g->bufs[o->a].stride, cs = o->c >= 0 ? g->bufs[o->c].stride : 0u, bs = g->bufs[o->b].stride;
        float esc = o->escale;
        u32 TT = T;
        if (b->tr_on && m->arch == ARCH_INKLING2 && b->ink_tree_split > 0u && b->ink_tree_split < T) {
          /* R32 exact 2-branch depth-1 tree sconv: Branch A [0..s-1] at pos, write x[0]->ring[pos%4], Branch B [s..T-1] at pos+1. */
          const u32 s_br = b->ink_tree_split, nb_br = T - s_br;
          const int use_ik = b->k[K_INK_SCONV] && mk_ok_k(K_INK_SCONV) && (n & 3u) == 0u &&
                             (!C || (nh > 0u && ((n / nh) & 3u) == 0u)) &&
                             ((uintptr_t)A & 15u) == 0u && ((uintptr_t)B & 15u) == 0u && ((uintptr_t)w0 & 15u) == 0u &&
                             ((uintptr_t)ring & 15u) == 0u && (!C || ((uintptr_t)C & 15u) == 0u) &&
                             (((as | bs | (C ? cs : 0u)) & 15u) == 0u);
          u32 ta = s_br, tb = nb_br, t1 = 1u, pos_b = pos + 1u;
          void *aa[] = {&A, &w0, &C, &ring, &B, &n, &nh, &esc, &pos, &ta, &as, &cs, &bs};
          if (use_ik) launch_t(b, K_INK_SCONV, ((n >> 2u) + 255u) / 256u, (ta + 7u) / 8u, aa);
          else launch_t(b, K_SCONV, elem_groups, ta, aa);
          void *ra0[] = {&A, &ring, &n, &pos, &t1, &as};
          launch(b, K_SCONV_RING, elem_groups, ra0);
          void *Ab = (char *)A + (size_t)s_br * as;
          void *Bb = (char *)B + (size_t)s_br * bs;
          void *Cb = C ? ((char *)C + (size_t)s_br * cs) : NULL;
          void *ab[] = {&Ab, &w0, &Cb, &ring, &Bb, &n, &nh, &esc, &pos_b, &tb, &as, &cs, &bs};
          if (use_ik) launch_t(b, K_INK_SCONV, ((n >> 2u) + 255u) / 256u, (tb + 7u) / 8u, ab);
          else launch_t(b, K_SCONV, elem_groups, tb, ab);
          if (b->sc_defer) {
            float *hp = b->sc_hist + (size_t)o->scoff;
            u32 hrs = b->sc_hist_stride;
            void *ha[] = {&A, &hp, &n, &TT, &as, &hrs};
            launch_t(b, K_INK_SCONV_HIST_R4, ((n >> 2u) + 255u) / 256u, T, ha);
          }
          break;
        }
        void *args[] = {&A, &w0, &C, &ring, &B, &n, &nh, &esc, &pos, &TT, &as, &cs, &bs};
        if (b->ink_graph_cap && T == 1u && b->k[K_INK_SCONV_CTX_R2] && !mk.rec && (n & 3u) == 0u &&
            (!C || (nh > 0u && ((n / nh) & 3u) == 0u)) &&
            ((uintptr_t)A & 15u) == 0u && ((uintptr_t)B & 15u) == 0u && ((uintptr_t)w0 & 15u) == 0u &&
            ((uintptr_t)ring & 15u) == 0u && (!C || ((uintptr_t)C & 15u) == 0u)) {
          void *ca[] = {&A, &w0, &C, &ring, &B, &n, &nh, &esc, &b->ink_ctx_dev};
          launch(b, K_INK_SCONV_CTX_R2, ((n >> 2u) + 255u) / 256u, ca);
        } else if (b->k[K_INK_SCONV] && mk_ok_k(K_INK_SCONV) && (n & 3u) == 0u && (!C || (nh > 0u && ((n / nh) & 3u) == 0u)) &&
            ((uintptr_t)A & 15u) == 0u && ((uintptr_t)B & 15u) == 0u && ((uintptr_t)w0 & 15u) == 0u &&
            ((uintptr_t)ring & 15u) == 0u && (!C || ((uintptr_t)C & 15u) == 0u) &&
            (T == 1u || (((as | bs | (C ? cs : 0u)) & 15u) == 0u))) {
          launch_t(b, K_INK_SCONV, ((n >> 2u) + 255u) / 256u, (T + 7u) / 8u, args);
        } else {
          launch_t(b, K_SCONV, elem_groups, T, args);
        }
        if (T > 1u) {
          if (b->sc_defer) {
            float *hp = b->sc_hist + (size_t)o->scoff;
            u32 hrs = b->sc_hist_stride;
            void *ha[] = {&A, &hp, &n, &TT, &as, &hrs};
            launch_t(b, K_INK_SCONV_HIST_R4, ((n >> 2u) + 255u) / 256u, T, ha);
          } else {
            void *ra[] = {&A, &ring, &n, &pos, &TT, &as};
            launch(b, K_SCONV_RING, elem_groups, ra);
          }
        }
        break;
      }
      case OP_REL_ATTN: {
        u32 kvi = o->kvi, kvlen = g->kv_len[kvi], nh = o->nh, d_rel = o->qstr, relext = m->rel_extent, win = o->win;
        void *kc = kv_ptr(b, 0, kvi, 0), *vc = kv_ptr(b, 1, kvi, 0);
        if (T > 1u) {
          u32 qs = g->bufs[o->a].stride, rs = g->bufs[o->nq].stride, ss = g->bufs[o->res].stride, os = g->bufs[o->b].stride;
          u32 TT = T;
          void *args[] = {&A, &kc, &vc, &NQ, &w0, &R, &B, &pos, &TT, &kvlen, (void *)&hd, &nh, (void *)&nkv, &d_rel,
                          &relext, &win, &qs, &rs, &ss, &os};
          if (b->tr_on && b->k[K_INK_REL_ATTN_VER_TREE_R8] && !b->kvq8 && !b->kvq4 && !mk.rec &&
              T >= 2u && T <= 8u && hd == 128u && d_rel == 16u && nkv > 0u &&
              nh == nkv * 3u && kvlen > 0u && ((qs | os) & 15u) == 0u && ((uintptr_t)A & 15u) == 0u) {
            u32 np_pfx = (win > 0u && pos + T > win) ? win : pos;
            u32 nsplit = (np_pfx >= 128u && b->at_part && b->at_count) ? (np_pfx + 63u) / 64u : 1u;
            if (nsplit > 64u) nsplit = 64u;
            while (nsplit > 1u && (size_t)T * nh * nsplit * 130u > b->at_part_floats) nsplit >>= 1u;
            void *part_p = (void *)b->at_part;
            uint64_t tmask = b->ink_tree_mask;
            u32 tdep = b->ink_tree_dep_pack;
            void *va[] = {&A, &kc, &vc, &NQ, &w0, &R, &part_p, &B, &pos, &TT, &kvlen, (void *)&hd, &nh, (void *)&nkv,
                          &d_rel, &relext, &win, &nsplit, &qs, &rs, &ss, &os, &b->at_count, &tmask, &tdep};
            launch_t(b, K_INK_REL_ATTN_VER_TREE_R8, nkv * nsplit, T, va);
          } else if (b->k[K_INK_REL_ATTN_VER_SWA_QTILE_R7] && !b->kvq8 && !b->kvq4 && mk_ok_k(K_INK_REL_ATTN_VER_SWA_QTILE_R7) &&
              win > 0u && T >= 2u && T <= 16u && hd == 128u && d_rel == 16u && nkv > 0u &&
              nh == nkv * 3u && kvlen > 0u && ((qs | os) & 15u) == 0u && ((uintptr_t)A & 15u) == 0u) {
            u32 np_u = (pos + T > win) ? (win + 3u) : (pos + T);
            u32 nsplit = (np_u >= 128u && b->at_part && b->at_count) ? (np_u + 63u) / 64u : 1u;
            if (nsplit > 64u) nsplit = 64u;
            void *part_p = (void *)b->at_part;
            void *va[] = {&A, &kc, &vc, &NQ, &w0, &R, &part_p, &B, &pos, &TT, &kvlen, (void *)&hd, &nh, (void *)&nkv,
                          &d_rel, &relext, &win, &nsplit, &qs, &rs, &ss, &os, &b->at_count};
            launch_t(b, K_INK_REL_ATTN_VER_SWA_QTILE_R7, nkv * nsplit, (T + 3u) / 4u, va);
          } else if (b->k[K_INK_REL_ATTN_VER_GLOB_WMMA_R7] && !b->kvq8 && !b->kvq4 && !mk.rec &&
              win == 0u && pos >= 2048u && T >= 2u && T <= 8u && hd == 128u && d_rel == 16u && nkv > 0u &&
              nh == nkv * 3u && kvlen > 0u && ((qs | os) & 15u) == 0u && ((uintptr_t)A & 15u) == 0u) {
            u32 np = pos + T;
            u32 nsplit = (np >= 256u && b->at_part && b->at_count) ? (np + 255u) / 256u : 1u;
            if (nsplit > 64u) nsplit = 64u;
            while (nsplit > 1u && (size_t)T * nh * nsplit * 130u > b->at_part_floats) nsplit >>= 1u;
            void *part_p = (void *)b->at_part;
            void *va[] = {&A, &kc, &vc, &NQ, &w0, &R, &part_p, &B, &pos, &TT, &kvlen, (void *)&hd, &nh, (void *)&nkv,
                          &d_rel, &relext, &win, &nsplit, &qs, &rs, &ss, &os, &b->at_count};
            launch_t(b, K_INK_REL_ATTN_VER_GLOB_WMMA_R7, nkv * nsplit, 1u, va);
          } else if (b->k[K_INK_REL_ATTN_VER_QTILE_R6] && !b->kvq8 && !b->kvq4 && mk_ok_k(K_INK_REL_ATTN_VER_QTILE_R6) &&
              win == 0u && pos >= ink_qtile_min() && T >= 2u && T <= 16u && hd == 128u && d_rel == 16u && nkv > 0u &&
              nh == nkv * 3u && kvlen > 0u && ((qs | os) & 15u) == 0u && ((uintptr_t)A & 15u) == 0u) {
            u32 np = pos + T;
            u32 nsplit = (np >= 128u && b->at_part && b->at_count) ? (np + 63u) / 64u : 1u;
            if (nsplit > 64u) nsplit = 64u;
            void *part_p = (void *)b->at_part;
            void *va[] = {&A, &kc, &vc, &NQ, &w0, &R, &part_p, &B, &pos, &TT, &kvlen, (void *)&hd, &nh, (void *)&nkv,
                          &d_rel, &relext, &win, &nsplit, &qs, &rs, &ss, &os, &b->at_count};
            launch_t(b, K_INK_REL_ATTN_VER_QTILE_R6, nkv * nsplit, (T + 3u) / 4u, va);
          } else if (b->k[K_INK_REL_ATTN_VER_R4] && !b->kvq8 && !b->kvq4 && mk_ok_k(K_INK_REL_ATTN_VER_R4) &&
              T >= 2u && T <= 16u && hd == 128u && d_rel == 16u && nkv > 0u &&
              nh == nkv * 3u && kvlen > 0u && ((qs | os) & 15u) == 0u && ((uintptr_t)A & 15u) == 0u) {
            u32 np = (win > 0u && pos + T > win) ? win : (pos + T);
            u32 nsplit = (np >= 128u && b->at_part && b->at_count) ? (np + 63u) / 64u : 1u;
            if (nsplit > 64u) nsplit = 64u;
            void *part_p = (void *)b->at_part;
            void *va[] = {&A, &kc, &vc, &NQ, &w0, &R, &part_p, &B, &pos, &TT, &kvlen, (void *)&hd, &nh, (void *)&nkv,
                          &d_rel, &relext, &win, &nsplit, &qs, &rs, &ss, &os, &b->at_count};
            launch_t(b, K_INK_REL_ATTN_VER_R4, nkv * nsplit, T, va);
          } else if (b->k[K_INK_REL_ATTN_PF_SPLIT_R7] && !b->kvq8 && !b->kvq4 && !mk.rec &&
              win == 0u && pos >= 2048u && T >= 17u && T <= 96u && hd == 128u && d_rel == 16u &&
              nkv > 0u && nh == nkv * 3u && kvlen > 0u && b->at_part && b->at_count &&
              ((T + 31u) / 32u) * nkv <= m->n_head * b->at_tmax) {
            const u32 ntile = (T + 31u) / 32u;
            u32 nsplit = (128u + ntile * nkv - 1u) / (ntile * nkv);
            if (nsplit < 2u) nsplit = 2u;
            if (nsplit > 8u) nsplit = 8u;
            while (nsplit > 1u && (size_t)T * nh * nsplit * 130u > b->at_part_floats) nsplit--;
            if (nsplit >= 2u) {
              void *part_p = (void *)b->at_part;
              void *va[] = {&A, &kc, &vc, &NQ, &w0, &R, &part_p, &B, &pos, &TT, &kvlen, (void *)&hd, &nh, (void *)&nkv,
                            &d_rel, &relext, &win, &nsplit, &qs, &rs, &ss, &os, &b->at_count};
              launch_t(b, K_INK_REL_ATTN_PF_SPLIT_R7, ntile * nsplit, nkv, va);
            } else {
              const int ka = b->k[K_INK_REL_ATTN_PF_R7] ? K_INK_REL_ATTN_PF_R7
                           : (b->k[K_INK_REL_ATTN_PF_R3] ? K_INK_REL_ATTN_PF_R3 : K_INK_REL_ATTN_PF);
              launch_t(b, ka, ntile, nkv, args);
            }
          } else if ((b->k[K_INK_REL_ATTN_PF_R7] || b->k[K_INK_REL_ATTN_PF_R3] || b->k[K_INK_REL_ATTN_PF]) &&
              !b->kvq8 && !b->kvq4 && !mk.rec && hd == 128u && d_rel == 16u && nkv > 0u && nh == nkv * 3u && kvlen > 0u) {
            const int ka = b->k[K_INK_REL_ATTN_PF_R7] ? K_INK_REL_ATTN_PF_R7
                         : (b->k[K_INK_REL_ATTN_PF_R3] ? K_INK_REL_ATTN_PF_R3 : K_INK_REL_ATTN_PF);
            launch_t(b, ka, (T + 31u) / 32u, nkv, args);
          } else {
            launch_t(b, K_REL_ATTN_PF, (T + 7u) / 8u, nh, args);
          }
        } else {
          u32 np = (win > 0u && pos + 1u > win) ? win : (pos + 1u);
          u32 max_sp = (u32)(g->bufs[o->c].size / ((size_t)nh * (hd + 2u) * 4u));
          if (max_sp > 64u) max_sp = 64u;
          if (max_sp == 0u) max_sp = 1u;
          if (b->ink_graph_cap && b->k[K_INK_REL_ATTN_DEC_CTX_R2] && !b->kvq8 && !b->kvq4 && !mk.rec &&
              hd == 128u && d_rel == 16u && nkv > 0u && nh == 3u * nkv && kvlen > 0u) {
            u32 nsplit = win > 0u ? b->ink_swa_nsplit : b->ink_glob_nsplit;
            if (nsplit > max_sp) nsplit = max_sp;
            if (!nsplit) nsplit = 1u;
            void *ca[] = {&A, &kc, &vc, &NQ, &w0, &R, &C, &B, &kvlen, (void *)&hd, &nh, (void *)&nkv, &d_rel,
                          &relext, &win, &nsplit, &b->at_count, &b->ink_ctx_dev};
            launch(b, K_INK_REL_ATTN_DEC_CTX_R2, nkv * nsplit, ca);
          } else {
            const int use_dec = b->k[K_INK_REL_ATTN_DEC] && !b->kvq8 && !b->kvq4 && mk_ok_k(K_INK_REL_ATTN_DEC) &&
                                hd == 128u && d_rel == 16u && nkv > 0u && nh == 3u * nkv;
            u32 nsplit = use_dec ? ((np >= 128u && b->at_count) ? (np + 63u) / 64u : 1u)
                                 : ((np >= 256u && b->at_count) ? (np + 127u) / 128u : 1u);
            if (nsplit > max_sp) nsplit = max_sp;
            u32 ch = (np + nsplit - 1u) / nsplit;
            nsplit = (np + ch - 1u) / ch;
            void *args[] = {&A, &kc, &vc, &NQ, &w0, &R, &C, &B, &pos, &kvlen, (void *)&hd, &nh, (void *)&nkv, &d_rel,
                            &relext, &win, &nsplit, &ch, &b->at_count};
            if (use_dec) launch(b, K_INK_REL_ATTN_DEC, nkv * nsplit, args);
            else launch(b, K_REL_ATTN, nh * nsplit, args);
          }
        }
        break;
      }
      case OP_SITU: {
        if (g->bufs[o->a].rows && T > g->bufs[o->a].rows)
          ie_die("OP_SITU: exp_gu capped to %u rows (T=%u) without R6 fusion", g->bufs[o->a].rows, T);
        u32 as = o->astr, qs = o->bstr;
        void *Bout = o->b >= 0 ? B : (Q ? NULL : A);
        u32 bs = o->b >= 0 ? (g->bufs[o->b].stride / o->nh) : (Q ? 0u : as);
        float beta = o->clamp, lbeta = o->lam;
        if (T > 1u && (g->bufs[o->a].stride != o->nh * as || (o->qo >= 0 && g->bufs[o->qo].stride != o->nh * qs)))
          ie_die("expert SITU of a chunk: token strides %u, %u (slots %u, %u)", g->bufs[o->a].stride,
                 o->qo >= 0 ? g->bufs[o->qo].stride : 0u, as, qs);
        void *args[] = {&A, &Bout, &Q, &n, &as, &bs, &qs, &beta, &lbeta};
        if (b->k[K_INK_SITU] && mk_ok_k(K_INK_SITU) && (n & 31u) == 0u && (as & 15u) == 0u &&
            (!Bout || (bs & 15u) == 0u) && (!Q || (qs & 3u) == 0u)) {
          const u32 gx = (o->nh * T >= 32u) ? 1u : (((n >> 5u) + 31u) / 32u);
          launch_t(b, K_INK_SITU, gx, o->nh * T, args);
        } else {
          launch_t(b, K_SITU, elem_groups, o->nh * T, args);
        }
        break;
      }
      case OP_LOGIT_CAP: {
        u32 unpad = o->nh, as = g->bufs[o->a].stride, bs = g->bufs[o->b].stride;
        float cap = o->clamp;
        void *args[] = {&A, &B, &n, &unpad, &cap, &as, &bs};
        launch_t(b, K_LOGIT_CAP, elem_groups, T, args);
        break;
      }
      case OP_ARGMAX: {
        if (b->skip_argmax && i < g->n_main) break; /* sampled run: the top K is taken instead, the MTP head's argmax still runs */
        u32 ng = ARGMAX_GROUPS;
        void *args[] = {&A, &n, &B, &b->am_part, &b->am_count, &ng};
        launch(b, K_ARGMAX, ng, args);
        break;
      }
    }
  }
}

/* IE_DUMP_GPU_OPS=1: print every op's output like the CPU backend's IE_DUMP_OPS,
 * to find the first op where the two backends disagree (debug only). */
static void dump_gpu_op(gpu_backend *b, uint32_t i, u32 t) {
  const graph *g = b->g;
  const op *o = &g->ops[i];
  /* the op's own output: OP_ADD writes c, OP_GATE rewrites a, the rest write b */
  const int to_c = o->kind == OP_ADD, to_a = o->kind == OP_GATE;
  const int ob = to_c ? o->c : to_a ? o->a : (o->b >= 0 ? o->b : o->a);
  if (ob < 0 || o->n == 0) return;
  const u32 oo = to_c ? o->coff : to_a ? o->aoff : (o->b >= 0 ? o->boff : o->aoff);
  const size_t off = oo + (size_t)buf_row(&g->bufs[ob], t) * g->bufs[ob].stride;
  u32 n = o->n;
  if (g->bufs[ob].size <= oo) return;
  if (n > 1024u) n = 1024u; /* the CPU dump prints the same 1024 floats */
  if (n > (g->bufs[ob].size - oo) / 4u) n = (g->bufs[ob].size - oo) / 4u;
  float *h = (float *)malloc((size_t)n * 4);
  if (!h) return;
  HIP(H.DeviceSynchronize());
  HIP(H.Memcpy(h, b->arena + g->bufs[ob].off + off, (size_t)n * 4, hipMemcpyDeviceToHost));
  double ss = 0.0;
  for (u32 j = 0; j < n; j++) ss += (double)h[j] * h[j];
  fprintf(stderr, "gop %3u kind %3u layer %2d n %5u |out| %.6g [", i, o->kind, o->layer, n, sqrt(ss));
  for (u32 j = 0; j < n && j < 4; j++) fprintf(stderr, " %.6g", h[j]);
  fprintf(stderr, "]");
  if (o->qo >= 0 && o->n % 32u == 0u) { /* fingerprint of the fused Q8 copy */
    const u32 nb = o->n / 32u;
    uint8_t *qh = (uint8_t *)malloc(40u * nb);
    if (qh) {
      HIP(H.Memcpy(qh, b->arena + g->bufs[o->qo].off, 40u * (size_t)nb, hipMemcpyDeviceToHost));
      u32 hh = 2166136261u; double ds = 0.0; unsigned long long sa = 0;
      for (u32 j = 0; j < 32u * nb; j++) hh = (hh ^ qh[j]) * 16777619u;
      for (u32 b2 = 0; b2 < nb; b2++) {
        float d2; u32 a2;
        memcpy(&d2, qh + 32u * nb + 4u * b2, 4); memcpy(&a2, qh + 36u * nb + 4u * b2, 4);
        ds += d2; sa += a2;
      }
      fprintf(stderr, " q8 %u %08x %.9g %llu", nb, hh, ds, sa);
    }
    free(qh);
  }
  fprintf(stderr, "\n");
  free(h);
}

static int gdn_eager_replay(void) {
  static int on = -1;
  if (on < 0) {
    const char *e = getenv("IE_GDN_EAGER_REPLAY");
    on = e != NULL && e[0] != '0';
  }
  return on;
}

static void gpu_flush_gdn_replay(gpu_backend *b) {
  if (!b->lazy_mask || !b->lazy_rec) return;
  const model *m = b->m;
  uint64_t sls = (uint64_t)m->n_vh * m->sd * m->sd, rls = b->lazy_rls;
  u32 nv = m->n_vh, mask = b->lazy_mask;
  void *rec = b->lazy_rec;
  b->lazy_mask = 0;
  b->lazy_rec = NULL;
  if (mk.n) mk_flush();
  const int saved_rec = mk.rec;
  mk.rec = 0;
  void *args[] = {&b->st, &rec, &nv, &mask, &sls, &rls};
  launch(b, K_GDN_REPLAY, m->n_rec * nv, args);
  mk.rec = saved_rec;
}

static u32 mtp_prenorm_fuse(gpu_backend *b, u32 i, u32 i1, const u32 *toks, u32 T) {
  const graph *g = b->g;
  const model *m = b->m;
  static int on = -1;
  if (on < 0) on = !(getenv("IE_MTP_PRENORM") && getenv("IE_MTP_PRENORM")[0] == '0');
  if (!on || !b->has_mtp_prenorm || g->mtp_h < 0 || mk.rec) return 0;
  /* Case 1 (T <= 16, i == n_main): fuse OP_EMBED + OP_RMSNORM(e) + OP_RMSNORM(h) [+ OP_QUANT(cat)].
   * Case 2 (T > 16, i == n_main + 1): OP_EMBED already ran; fuse OP_RMSNORM(e) + OP_RMSNORM(h) [+ OP_QUANT(cat)]. */
  const int with_emb = (i == g->n_main && T >= 1u && T <= 16u);
  if (!with_emb && !(i == g->n_main + 1u && T > 16u)) return 0;
  const u32 ir = with_emb ? i + 1u : i;
  if (ir + 1u >= i1) return 0;
  const op *o0 = with_emb ? &g->ops[i] : NULL;
  const op *o1 = &g->ops[ir], *o2 = &g->ops[ir + 1u];
  if (o0 && (o0->kind != OP_EMBED || o0->a != o1->a || o0->aoff != 0u || o0->n != o1->n)) return 0;
  u32 dim = o1->n;
  if (!dim || (dim & 31u) != 0u) return 0;
  if (o1->kind != OP_RMSNORM || o1->aoff != 0u || o1->boff != 0u || o1->qo >= 0 || !b->d[ir].w0) return 0;
  if (o2->kind != OP_RMSNORM || o2->a != g->mtp_h || o2->aoff != 0u || o2->b != o1->b || o2->boff != dim * 4u ||
      o2->n != dim || o2->qo >= 0 || !b->d[ir + 1u].w0)
    return 0;
  const op *o3 = (ir + 2u < i1 && g->ops[ir + 2u].kind == OP_QUANT && g->ops[ir + 2u].a == o1->b &&
                  g->ops[ir + 2u].aoff == 0u && g->ops[ir + 2u].boff == 0u && g->ops[ir + 2u].n == 2u * dim)
                     ? &g->ops[ir + 2u]
                     : NULL;
  const u32 n_ops = (with_emb ? 3u : 2u) + (o3 ? 1u : 0u);
  u32 tok_kind = 0u;
  void *tw0 = NULL, *tw1 = NULL;
  float escale = 1.0f;
  if (with_emb) {
    const mat *tm = o0->w ? o0->w : &m->tok;
    tw0 = o0->w ? b->d[i].w0 : b->tok0;
    tw1 = o0->w ? b->d[i].w1 : b->tok1;
    escale = o0->escale > 0.0f ? o0->escale : 1.0f;
    tok_kind = tm->kind == MAT_Q4 ? 1u : tm->kind == MAT_Q8 ? 2u : tm->kind == MAT_F32 ? 3u : 0u;
    if (!tok_kind) return 0;
    if (tok_kind == 3u && (((uintptr_t)tw0 & 15u) != 0u || (dim & 3u) != 0u)) return 0;
  }
  int cat_live = !o3 || !deadb_on() || o1->b == g->mtp_r || o1->b == g->mtp_a || o1->b == g->mtp_y;
  int e_live = 0;
  for (u32 j = i + n_ops; j < g->n_ops; j++) {
    const op *p = &g->ops[j];
    if (p->a == o1->b || p->b == o1->b || p->c == o1->b || p->res == o1->b || p->gt == o1->b) cat_live = 1;
    if (with_emb && (p->a == o0->a || p->b == o0->a || p->c == o0->a || p->res == o0->a || p->gt == o0->a)) e_live = 1;
  }
  if (with_emb && (o0->a == g->mtp_r || o0->a == g->mtp_a || o0->a == g->mtp_y)) e_live = 1;
  if (cat_live && (!buf_disjoint(g, o2->a, o1->b) || (!with_emb && !buf_disjoint(g, o1->a, o1->b)))) return 0;
  if (o3 && (!buf_disjoint(g, o2->a, o3->b) || (!with_emb && !buf_disjoint(g, o1->a, o3->b)))) return 0;
  if (cat_live && o3 && !buf_disjoint(g, o1->b, o3->b)) return 0;
  if (e_live && (!buf_disjoint(g, o2->a, o0->a) || (cat_live && !buf_disjoint(g, o0->a, o1->b)) || (o3 && !buf_disjoint(g, o0->a, o3->b))))
    return 0;
  void *E_in = with_emb ? NULL : (void *)(b->arena + g->bufs[o1->a].off);
  void *E_out = (with_emb && e_live) ? (void *)(b->arena + g->bufs[o0->a].off) : NULL;
  void *H_in = b->arena + g->bufs[o2->a].off;
  void *ew = b->d[ir].w0, *hw = b->d[ir + 1u].w0;
  void *cat = cat_live ? (void *)(b->arena + g->bufs[o1->b].off) : NULL;
  void *catq = o3 ? (void *)(b->arena + g->bufs[o3->b].off) : NULL;
  u32 es = g->bufs[o1->a].stride, hs = g->bufs[o2->a].stride, cs = g->bufs[o1->b].stride, qs = o3 ? g->bufs[o3->b].stride : 0u;
  if (((uintptr_t)H_in & 15u) || (hs & 15u) || ((uintptr_t)ew & 15u) || ((uintptr_t)hw & 15u)) return 0;
  if (E_in && (((uintptr_t)E_in & 15u) || (es & 15u))) return 0;
  if (E_out && (((uintptr_t)E_out & 15u) || (es & 15u))) return 0;
  if (cat && (((uintptr_t)cat & 15u) || (cs & 15u))) return 0;
  float eps = m->eps;
  uint64_t tp[8] = {0};
  if (with_emb)
    for (u32 t = 0; t < T; t++) tp[t >> 1] |= (uint64_t)toks[t] << (32u * (t & 1u));
  void *args[] = {&E_in, &E_out, &H_in, &ew, &hw, &cat, &catq, &tw0, &tw1, &dim, &eps, &escale, &tok_kind,
                  &es, &hs, &cs, &qs, &tp[0], &tp[1], &tp[2], &tp[3], &tp[4], &tp[5], &tp[6], &tp[7]};
  tr_span(g, i, n_ops);
  launch_t(b, K_MTP_PRENORM_Q8, 2u * ((dim + 1023u) / 1024u), T, args);
  return n_ops;
}

/* IE_INKLING_T1_FUSE=MASK (default 0): the fused kernels that so far run for prompt chunks and verifies only, also
 * for one token (a decode step, a draft of the main model, a one-token verify). They do not change what the GPU
 * computes per step, they are fewer launches: 1 the loop controller's two norms and their Q8 copy
 * (ie_inkling_lc_prenorm_q8_r2, 2 launches less per controller, 96 controllers per token), 2 its gate
 * (ie_inkling_lc_wfg_gate_r2, 2 less), 4 the K/V conv, clip and store (ie_inkling_kv_sconv_vclip_store_r2, bit 25 of
 * IE_INKLING_KERNEL_MASK, 4 or 5 less per layer visit), 8 a conv with its residual add (ie_inkling_sconv_add_r2). */
static unsigned ink_t1_fuse(void) {
  static int v = -1;
  if (v < 0) {
    const char *e = getenv("IE_INKLING_T1_FUSE");
    v = e ? (int)(strtoul(e, NULL, 0) & 15u) : 0;
    if (v) fprintf(stderr, "gpu: inkling fused kernels for one token: mask %d (IE_INKLING_T1_FUSE)\n", v);
  }
  return (unsigned)v;
}

static u32 ink_r2_fuse(gpu_backend *b, u32 i, u32 i1, const u32 *toks, u32 T, u32 pos, u32 slot, u32 wfrom) {
  const graph *g = b->g;
  const model *m = b->m;
  const op *o0 = &g->ops[i];
  const int k0 = o0->kind;
  const int ctx_on = b->ink_graph_cap && T == 1u;
  const unsigned t1f = T == 1u ? ink_t1_fuse() : 0u;

  /* R7 Wave 4 bit 76: [OP_GEMV_Q4K, OP_LOGIT_CAP] (T=1..8) -> ie_inkling_gemv_q4k_head_ver_r7 */
  if (k0 == OP_GEMV_Q4K && !mk.rec && T >= 1u && T <= 8u && i + 1u < i1 &&
      b->k[K_INK_GEMV_Q4K_HEAD_VER_R7] && !o0->nv &&
      o0->w->rows > 0u && o0->w->nb > 0u && (o0->w->nb & 7u) == 0u && o0->boff == 0u) {
    const op *o1 = &g->ops[i + 1u];
    static int q4k_min_t = -1;
    if (q4k_min_t < 0) q4k_min_t = getenv("IE_INK_Q4K_HEAD_MIN_T") ? atoi(getenv("IE_INK_Q4K_HEAD_MIN_T")) : 1;
    if ((int)T >= q4k_min_t && o1->kind == OP_LOGIT_CAP && o1->a == o0->b && o1->aoff == 0u && o1->boff == 0u &&
        o1->n == o0->w->rows && (o1->b == o0->b || buf_disjoint(g, o0->a, o1->b))) {
      void *qw = b->d[i].w0, *qs = b->d[i].w1, *bias = b->d[i].bias;
      void *xq = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *y = b->arena + g->bufs[o1->b].off + o1->boff;
      void *res = o0->res >= 0 ? (b->arena + g->bufs[o0->res].off) : NULL;
      u32 rows = o0->w->rows, nb = o0->w->nb, TT = T;
      u32 xs = g->bufs[o0->a].stride, ys = g->bufs[o1->b].stride / 4u;
      u32 rs = o0->res >= 0 ? g->bufs[o0->res].stride / 4u : 0u;
      u32 unpad = o1->nh;
      float cap = o1->clamp;
      u32 gr = (rows + 31u) / 32u;
      void *fa[] = {&qw, &qs, &xq, &y, &rows, &nb, &bias, &res, &TT, &xs, &ys, &rs, &unpad, &cap};
      tr_span(g, i, 2u);
      launch(b, K_INK_GEMV_Q4K_HEAD_VER_R7, gr >= 253u && gr <= 256u ? 257u : gr, fa);
      return 2u;
    }
  }

  /* R6 bit 50 (Exp #2): 2 adjacent K=2 OP_ATTN_RES bank reads (msg_qm, msg_kv) -> ie_inkling_attn_res_4out_r6 */
  if (k0 == OP_ATTN_RES && i + 1u < i1 && b->k[K_INK_ATTN_RES_4OUT_R6] && mk_ok_k(K_INK_ATTN_RES_4OUT_R6)) {
    const op *o1 = &g->ops[i + 1u];
    if (o1->kind == OP_ATTN_RES && o0->c < 0 && o1->c < 0 && o0->a >= 0 && o0->a == o1->a && o0->aoff == o1->aoff &&
        o0->nh == o1->nh && o0->qstr == 2u && o1->qstr == 2u && o0->n == o1->n && o0->nh >= 1u && o0->nh <= 64u &&
        o0->n >= 1u && o0->b >= 0 && o1->b >= 0 &&
        buf_disjoint(g, o0->a, o0->b) && buf_disjoint(g, o0->a, o1->b) && buf_disjoint(g, o0->b, o1->b)) {
      void *a = b->arena + g->bufs[o0->a].off + o0->aoff, *c_nul = NULL;
      float *q0 = (float *)b->d[i].w0 + o0->roff;
      float *q1 = (float *)b->d[i + 1u].w0 + o1->roff;
      void *b0 = b->arena + g->bufs[o0->b].off + o0->boff;
      void *b1 = b->arena + g->bufs[o1->b].off + o1->boff;
      u32 M = o0->nh, n = o0->n;
      float eps = m->eps;
      u32 as = g->bufs[o0->a].stride, cs = 0u, bs0 = g->bufs[o0->b].stride, bs1 = g->bufs[o1->b].stride;
      const int al_ok = (n & 3u) != 0u || (
          ((uintptr_t)a & 15u) == 0u && ((uintptr_t)q0 & 15u) == 0u && ((uintptr_t)q1 & 15u) == 0u &&
          ((uintptr_t)b0 & 15u) == 0u && ((uintptr_t)b1 & 15u) == 0u &&
          (T == 1u || ((as | bs0 | bs1) & 15u) == 0u));
      if (al_ok) {
        void *args[] = {&a, &c_nul, &q0, &q1, &b0, &b1, &M, &n, &eps, &as, &cs, &bs0, &bs1};
        tr_span(g, i, 2u);
        launch_t(b, K_INK_ATTN_RES_4OUT_R6, 1u, T, args);
        return 2u;
      }
    }
  }

  /* R7 Fusion E2 bit 67: non-loop [OP_ATTN_RES(K=1..3, n=2304), K x OP_RMSNORM[+Q8]] -> ie_inkling_attn_res_norm_q8_r7 */
  if (k0 == OP_ATTN_RES && !mk.rec && b->k[K_INK_ATTN_RES_NORM_Q8_R7] &&
      o0->n == 2304u && o0->nh >= 1u && o0->nh <= 32u && o0->qstr >= 1u && o0->qstr <= 3u &&
      o0->b >= 0 && i + o0->qstr < i1 && g->bufs[o0->b].last <= i + o0->qstr) {
    const u32 K = o0->qstr, M = o0->nh, n = 2304u;
    void *nw[3] = {NULL, NULL, NULL}, *ny[3] = {NULL, NULL, NULL}, *nq[3] = {NULL, NULL, NULL};
    u32 ys[3] = {0u, 0u, 0u}, qs[3] = {0u, 0u, 0u};
    int yb[3] = {-1, -1, -1}, qb[3] = {-1, -1, -1};
    int ok = 1;
    for (u32 k = 0u; ok && k < K; k++) {
      const op *rn = &g->ops[i + 1u + k];
      if (rn->kind != OP_RMSNORM || rn->a != o0->b || rn->aoff != k * 2304u * 4u ||
          rn->n != 2304u || !rn->v || rn->v == &m->ones_dim || rn->boff != 0u) {
        ok = 0; break;
      }
      nw[k] = b->d[i + 1u + k].w0;
      const int y_dead = rn->dead_b && deadb_on();
      if (!y_dead && rn->b >= 0) {
        ny[k] = b->arena + g->bufs[rn->b].off + rn->boff;
        yb[k] = rn->b;
        ys[k] = g->bufs[rn->b].stride;
      }
      if (rn->qo >= 0) {
        nq[k] = b->arena + g->bufs[rn->qo].off;
        qb[k] = rn->qo;
        qs[k] = g->bufs[rn->qo].stride;
      }
      if ((!ny[k] && !nq[k]) || ((uintptr_t)nw[k] & 15u) != 0u ||
          (ny[k] && ((uintptr_t)ny[k] & 15u) != 0u) ||
          (nq[k] && ((uintptr_t)nq[k] & 3u) != 0u) ||
          (T > 1u && ((ny[k] && (ys[k] & 15u) != 0u) || (nq[k] && (qs[k] & 3u) != 0u)))) {
        ok = 0; break;
      }
    }
    if (ok) {
      int outb[6] = {yb[0], yb[1], yb[2], qb[0], qb[1], qb[2]};
      for (int a = 0; ok && a < 6; a++) {
        if (outb[a] < 0) continue;
        if (o0->a >= 0 && !buf_disjoint(g, o0->a, outb[a])) ok = 0;
        if (o0->c >= 0 && !buf_disjoint(g, o0->c, outb[a])) ok = 0;
        for (int d = a + 1; ok && d < 6; d++)
          if (outb[d] >= 0 && !buf_disjoint(g, outb[a], outb[d])) ok = 0;
      }
    }
    if (ok) {
      void *a_p = o0->a >= 0 ? (b->arena + g->bufs[o0->a].off + o0->aoff) : NULL;
      void *c_p = o0->c >= 0 ? (b->arena + g->bufs[o0->c].off + o0->coff) : NULL;
      float *Qp = (float *)b->d[i].w0 + o0->roff;
      u32 as = o0->a >= 0 ? g->bufs[o0->a].stride : 0u;
      u32 cs = o0->c >= 0 ? g->bufs[o0->c].stride : 0u;
      int has_c = o0->c >= 0;
      float eps = m->eps;
      if ((!a_p || ((uintptr_t)a_p & 15u) == 0u) && (!has_c || ((uintptr_t)c_p & 15u) == 0u) &&
          ((uintptr_t)Qp & 15u) == 0u &&
          (T == 1u || ((!a_p || (as & 15u) == 0u) && (!has_c || (cs & 15u) == 0u)))) {
        u32 M_u = M, K_u = K, n_u = n;
        void *fa[] = {&a_p, &c_p, &Qp,
                      &nw[0], &ny[0], &nq[0], &nw[1], &ny[1], &nq[1], &nw[2], &ny[2], &nq[2],
                      &M_u, &K_u, &n_u, &eps,
                      &as, &cs, &ys[0], &qs[0], &ys[1], &qs[1], &ys[2], &qs[2], &has_c};
        tr_span(g, i, 1u + K);
        launch_t(b, K_INK_ATTN_RES_NORM_Q8_R7, 1u, T, fa);
        return 1u + K;
      }
    }
  }

  /* R6 bit 51 (Exp #2): 3x (or 1x MLP) [OP_RMSNORM(ones), OP_RMSNORM(ones), OP_QUANT(2*dim)] -> ie_inkling_lc_prenorm_q8_4x_r6 */
  if (k0 == OP_RMSNORM && b->k[K_INK_LC_PRENORM_Q8_4X_R6] && mk_ok_k(K_INK_LC_PRENORM_Q8_4X_R6) && o0->v == &m->ones_dim) {
    for (u32 nc = 3u; nc >= 1u; nc = (nc == 3u) ? 1u : 0u) {
      if (i + 3u * nc - 1u >= i1) continue;
      void *inp[4] = {NULL, NULL, NULL, NULL}, *msg[4] = {NULL, NULL, NULL, NULL}, *q[4] = {NULL, NULL, NULL, NULL};
      u32 is[4] = {0, 0, 0, 0}, ms[4] = {0, 0, 0, 0}, dim = o0->n, qs = 0u;
      int ib[4] = {-1, -1, -1, -1}, mb[4] = {-1, -1, -1, -1}, qb[4] = {-1, -1, -1, -1};
      int ok = (dim >= 32u && (dim & 31u) == 0u);
      for (u32 c = 0u; ok && c < nc; c++) {
        const op *r0 = &g->ops[i + 3u * c], *r1 = &g->ops[i + 3u * c + 1u], *rq = &g->ops[i + 3u * c + 2u];
        if (r0->kind != OP_RMSNORM || r1->kind != OP_RMSNORM || rq->kind != OP_QUANT ||
            r0->v != &m->ones_dim || r1->v != &m->ones_dim || r0->qo >= 0 || r1->qo >= 0 ||
            r0->b < 0 || r0->b != r1->b || r0->b != rq->a ||
            r0->boff != 0u || r1->boff != dim * 4u || rq->aoff != 0u || rq->boff != 0u ||
            r0->n != dim || r1->n != dim || rq->n != 2u * dim) { ok = 0; break; }
        ib[c] = r0->a; mb[c] = r1->a; qb[c] = rq->b;
        inp[c] = b->arena + g->bufs[r0->a].off + r0->aoff;
        msg[c] = b->arena + g->bufs[r1->a].off + r1->aoff;
        q[c] = b->arena + g->bufs[rq->b].off + rq->boff;
        is[c] = g->bufs[r0->a].stride; ms[c] = g->bufs[r1->a].stride;
        if (c == 0u) qs = g->bufs[rq->b].stride;
        else if (g->bufs[rq->b].stride != qs) { ok = 0; break; }
        if (((uintptr_t)inp[c] & 15u) != 0u || ((uintptr_t)msg[c] & 15u) != 0u || ((uintptr_t)q[c] & 3u) != 0u) { ok = 0; break; }
        if (T > 1u && (((is[c] | ms[c]) & 15u) != 0u || (qs & 3u) != 0u)) { ok = 0; break; }
      }
      for (u32 a = 0u; ok && a < nc; a++)
        for (u32 d = 0u; ok && d < nc; d++) {
          if (a != d && !buf_disjoint(g, qb[a], qb[d])) ok = 0;
          if (!buf_disjoint(g, ib[a], qb[d]) || !buf_disjoint(g, mb[a], qb[d])) ok = 0;
        }
      if (!ok) continue;
      int same_inp = (nc > 1u);
      for (u32 c = 1u; same_inp && c < nc; c++)
        if (inp[c] != inp[0] || (T > 1u && is[c] != is[0])) same_inp = 0;
      int down_next = 0;
      if (same_inp && nc == 3u && T <= 16u && i + 11u < i1 &&
          b->k[K_INK_LC_DOWN_4X_R6] && mk_ok_k(K_INK_LC_DOWN_4X_R6)) {
        const op *d0 = &g->ops[i + 9u], *d1 = &g->ops[i + 10u], *d2 = &g->ops[i + 11u];
        if (d0->kind == OP_GEMV_Q8 && d1->kind == OP_GEMV_Q8 && d2->kind == OP_GEMV_Q8 &&
            d0->a == qb[0] && d1->a == qb[1] && d2->a == qb[2] &&
            d0->aoff == 0u && d1->aoff == 0u && d2->aoff == 0u &&
            !d0->v && !d1->v && !d2->v && d0->res < 0 && d1->res < 0 && d2->res < 0 && !d0->nv && !d1->nv && !d2->nv &&
            d0->w->rows > 0u && d0->w->nb > 0u && (d0->w->nb & 1u) == 0u &&
            d1->w->rows == d0->w->rows && d2->w->rows == d0->w->rows &&
            d1->w->nb == d0->w->nb && d2->w->nb == d0->w->nb &&
            g->bufs[d0->b].stride == g->bufs[d1->b].stride && g->bufs[d0->b].stride == g->bufs[d2->b].stride &&
            buf_disjoint(g, d0->b, d1->b) && buf_disjoint(g, d0->b, d2->b) && buf_disjoint(g, d1->b, d2->b) &&
            buf_disjoint(g, qb[0], d0->b) && buf_disjoint(g, qb[0], d1->b) && buf_disjoint(g, qb[0], d2->b) &&
            buf_disjoint(g, qb[1], d0->b) && buf_disjoint(g, qb[1], d1->b) && buf_disjoint(g, qb[1], d2->b) &&
            buf_disjoint(g, qb[2], d0->b) && buf_disjoint(g, qb[2], d1->b) && buf_disjoint(g, qb[2], d2->b))
          down_next = 1;
      }
      float eps = m->eps;
      u32 n_ctrl = nc, flags = same_inp ? (down_next ? 3u : 1u) : 0u;
      void *args[] = {&inp[0], &inp[1], &inp[2], &inp[3], &msg[0], &msg[1], &msg[2], &msg[3],
                      &q[0], &q[1], &q[2], &q[3], &dim, &eps, &n_ctrl, &flags,
                      &is[0], &is[1], &is[2], &is[3], &ms[0], &ms[1], &ms[2], &ms[3], &qs};
      tr_span(g, i, 3u * nc);
      launch_t(b, K_INK_LC_PRENORM_Q8_4X_R6, (flags & 1u) ? (1u + n_ctrl) : (2u * n_ctrl), T, args);
      return 3u * nc;
    }
  }

  /* R6 bit 52 (Exp #2): 3x OP_GEMV_Q8 loop down projections (T<=16) -> ie_inkling_lc_down_4x_r6 */
  if (k0 == OP_GEMV_Q8 && T >= 1u && T <= 16u && i + 2u < i1 &&
      b->k[K_INK_LC_DOWN_4X_R6] && mk_ok_k(K_INK_LC_DOWN_4X_R6) &&
      !o0->v && o0->res < 0 && !o0->nv && o0->w->rows > 0u && o0->w->nb > 0u) {
    const op *d[3] = {o0, &g->ops[i + 1u], &g->ops[i + 2u]};
    u32 rows = o0->w->rows, nb = o0->w->nb, xs = g->bufs[o0->a].stride, ys = g->bufs[o0->b].stride;
    int ok = 1;
    void *qw[4] = {NULL, NULL, NULL, NULL}, *qs[4] = {NULL, NULL, NULL, NULL};
    void *xq[4] = {NULL, NULL, NULL, NULL}, *y[4] = {NULL, NULL, NULL, NULL};
    for (u32 c = 0u; ok && c < 3u; c++) {
      if (d[c]->kind != OP_GEMV_Q8 || d[c]->v || d[c]->res >= 0 || d[c]->nv ||
          d[c]->w->rows != rows || d[c]->w->nb != nb ||
          g->bufs[d[c]->a].stride != xs || g->bufs[d[c]->b].stride != ys) { ok = 0; break; }
      qw[c] = b->d[i + c].w0; qs[c] = b->d[i + c].w1;
      xq[c] = b->arena + g->bufs[d[c]->a].off + d[c]->aoff;
      y[c] = b->arena + g->bufs[d[c]->b].off + d[c]->boff;
      if (((uintptr_t)qw[c] & 3u) != 0u || ((uintptr_t)qs[c] & 3u) != 0u ||
          ((uintptr_t)xq[c] & 3u) != 0u || ((uintptr_t)y[c] & 3u) != 0u) { ok = 0; break; }
    }
    if (ok && (T == 1u || ((xs | ys) & 3u) == 0u)) {
      for (u32 a = 0u; ok && a < 3u; a++)
        for (u32 c = 0u; ok && c < 3u; c++) {
          if (a != c && !buf_disjoint(g, d[a]->b, d[c]->b)) ok = 0;
          if (!buf_disjoint(g, d[a]->a, d[c]->b)) ok = 0;
        }
      if (ok) {
        u32 shared_inp = 0u;
        if ((nb & 1u) == 0u && i >= 9u && d[0]->aoff == 0u && d[1]->aoff == 0u && d[2]->aoff == 0u) {
          const op *p0 = &g->ops[i - 9u], *p1 = &g->ops[i - 6u], *p2 = &g->ops[i - 3u];
          const op *q0 = &g->ops[i - 7u], *q1 = &g->ops[i - 4u], *q2 = &g->ops[i - 1u];
          if (p0->kind == OP_RMSNORM && p1->kind == OP_RMSNORM && p2->kind == OP_RMSNORM &&
              p0->v == &m->ones_dim && p1->v == &m->ones_dim && p2->v == &m->ones_dim &&
              p0->a == p1->a && p0->aoff == p1->aoff && p0->a == p2->a && p0->aoff == p2->aoff &&
              q0->kind == OP_QUANT && q1->kind == OP_QUANT && q2->kind == OP_QUANT &&
              q0->b == d[0]->a && q1->b == d[1]->a && q2->b == d[2]->a)
            shared_inp = 1u;
        }
        u32 TT = T, n_ctrl = 3u;
        void *args[] = {&qw[0], &qs[0], &xq[0], &y[0], &qw[1], &qs[1], &xq[1], &y[1],
                        &qw[2], &qs[2], &xq[2], &y[2], &qw[3], &qs[3], &xq[3], &y[3],
                        &rows, &nb, &TT, &n_ctrl, &shared_inp, &xs, &ys};
        u32 gx = n_ctrl * rows;
        if (gx >= 253u && gx <= 256u) gx = 257u;
        tr_span(g, i, 3u);
        launch(b, K_INK_LC_DOWN_4X_R6, gx, args);
        return 3u;
      }
    }
  }

  /* R6 bit 53 (Exp #2): T<=16 3x (or 1x MLP) [QUANT(64), GEMV_Q8(wfg,nb=2), LOOP_GATE] + OP_RMSNORM(qo) -> ie_inkling_lc_wfg_gate_norm_r6
   * (T>=17 prefill stays on K_INK_LC_WFG_GATE_R2 + K_INK_RMSNORM_MT, which is ~50-60 ms faster at C512.) */
  if (k0 == OP_QUANT && T <= 16u && o0->n == 64u && b->k[K_INK_LC_WFG_GATE_NORM_R6] &&
      mk_ok_k(K_INK_LC_WFG_GATE_NORM_R6) && b->ink_lc_count) {
    for (u32 nc = 3u; nc >= 1u; nc = (nc == 3u) ? 1u : 0u) {
      if (i + 4u * nc - 1u >= i1) continue;
      void *qw[4] = {NULL}, *qs[4] = {NULL}, *h[4] = {NULL}, *inp[4] = {NULL}, *msg[4] = {NULL};
      void *tmp[4] = {NULL}, *nw[4] = {NULL}, *ny[4] = {NULL}, *nq[4] = {NULL};
      int hb[4] = {-1, -1, -1, -1}, ib[4] = {-1, -1, -1, -1}, mb[4] = {-1, -1, -1, -1};
      int tb[4] = {-1, -1, -1, -1}, nyb[4] = {-1, -1, -1, -1}, nqb[4] = {-1, -1, -1, -1};
      u32 is[4] = {0}, ms[4] = {0};
      u32 dim = 0u, hs = 0u, ts = 0u, ys = 0u, qs_out = 0u;
      int ok = 1, any_ny = 0;
      for (u32 c = 0u; ok && c < nc; c++) {
        const u32 iq = i + 3u * c, ig = iq + 1u, il = iq + 2u, inr = i + 3u * nc + c;
        const op *oq = &g->ops[iq], *og = &g->ops[ig], *ol = &g->ops[il], *on = &g->ops[inr];
        if (oq->kind != OP_QUANT || oq->n != 64u || oq->boff != 0u ||
            og->kind != OP_GEMV_Q8 || og->a != oq->b || og->aoff != 0u || og->boff != 0u ||
            og->v || og->res >= 0 || og->nv || og->w->nb != 2u ||
            ol->kind != OP_LOOP_GATE || ol->gt != og->b || ol->gtoff != 0u || ol->boff != 0u ||
            ol->n < 32u || (ol->n & 31u) != 0u || og->w->rows != 2u * ol->n ||
            on->kind != OP_RMSNORM || on->a != ol->b || on->aoff != 0u || on->boff != 0u ||
            on->n != ol->n || !on->v || on->qo < 0) { ok = 0; break; }
        if (c == 0u) {
          dim = ol->n; hs = g->bufs[oq->a].stride; ts = g->bufs[ol->b].stride;
          ys = g->bufs[on->b].stride; qs_out = g->bufs[on->qo].stride;
        } else if (ol->n != dim || g->bufs[oq->a].stride != hs || g->bufs[ol->b].stride != ts ||
                   g->bufs[on->b].stride != ys || g->bufs[on->qo].stride != qs_out) { ok = 0; break; }
        hb[c] = oq->a; ib[c] = ol->a; mb[c] = ol->c; tb[c] = ol->b; nqb[c] = on->qo;
        qw[c] = b->d[ig].w0; qs[c] = b->d[ig].w1;
        h[c] = b->arena + g->bufs[oq->a].off + oq->aoff;
        inp[c] = b->arena + g->bufs[ol->a].off + ol->aoff;
        msg[c] = b->arena + g->bufs[ol->c].off + ol->coff;
        tmp[c] = b->arena + g->bufs[ol->b].off + ol->boff;
        nw[c] = b->d[inr].w0;
        nq[c] = b->arena + g->bufs[on->qo].off;
        if (on->dead_b && deadb_on()) {
          ny[c] = NULL; nyb[c] = -1;
        } else {
          ny[c] = b->arena + g->bufs[on->b].off + on->boff;
          nyb[c] = on->b;
          any_ny = 1;
        }
        is[c] = g->bufs[ol->a].stride; ms[c] = g->bufs[ol->c].stride;
        if (((uintptr_t)qw[c] & 3u) != 0u || ((uintptr_t)qs[c] & 3u) != 0u ||
            ((uintptr_t)h[c] & 3u) != 0u || ((uintptr_t)inp[c] & 3u) != 0u || ((uintptr_t)msg[c] & 3u) != 0u ||
            ((uintptr_t)tmp[c] & 15u) != 0u || ((uintptr_t)nw[c] & 15u) != 0u || ((uintptr_t)nq[c] & 3u) != 0u ||
            (ny[c] && ((uintptr_t)ny[c] & 15u) != 0u)) { ok = 0; break; }
        if (T > 1u && (((is[c] | ms[c]) & 3u) != 0u)) { ok = 0; break; }
      }
      if (ok && T > 1u && ((ts & 15u) != 0u || (any_ny && (ys & 15u) != 0u) || ((hs | qs_out) & 3u) != 0u)) ok = 0;
      for (u32 a = 0u; ok && a < nc; a++)
        for (u32 d = 0u; ok && d < nc; d++) {
          if (a != d) {
            if (!buf_disjoint(g, tb[a], tb[d]) || !buf_disjoint(g, nqb[a], nqb[d])) ok = 0;
            if (nyb[a] >= 0 && nyb[d] >= 0 && !buf_disjoint(g, nyb[a], nyb[d])) ok = 0;
          }
          if (!buf_disjoint(g, hb[a], tb[d]) || !buf_disjoint(g, ib[a], tb[d]) || !buf_disjoint(g, mb[a], tb[d])) ok = 0;
          if (!buf_disjoint(g, hb[a], nqb[d]) || !buf_disjoint(g, ib[a], nqb[d]) || !buf_disjoint(g, mb[a], nqb[d])) ok = 0;
          if (!buf_disjoint(g, tb[a], nqb[d])) ok = 0;
          if (nyb[d] >= 0) {
            if (!buf_disjoint(g, hb[a], nyb[d]) || !buf_disjoint(g, ib[a], nyb[d]) ||
                !buf_disjoint(g, mb[a], nyb[d]) || !buf_disjoint(g, tb[a], nyb[d]) || !buf_disjoint(g, nqb[a], nyb[d])) ok = 0;
          }
        }
      if (!ok) continue;
      u32 nb = 2u, TT = T, n_ctrl = nc;
      float eps = m->eps;
      void *args[] = {
          &qw[0], &qs[0], &h[0], &inp[0], &msg[0], &tmp[0], &nw[0], &ny[0], &nq[0],
          &qw[1], &qs[1], &h[1], &inp[1], &msg[1], &tmp[1], &nw[1], &ny[1], &nq[1],
          &qw[2], &qs[2], &h[2], &inp[2], &msg[2], &tmp[2], &nw[2], &ny[2], &nq[2],
          &qw[3], &qs[3], &h[3], &inp[3], &msg[3], &tmp[3], &nw[3], &ny[3], &nq[3],
          &b->ink_lc_count, &dim, &nb, &TT, &n_ctrl, &eps,
          &hs, &is[0], &is[1], &is[2], &is[3], &ms[0], &ms[1], &ms[2], &ms[3],
          &ts, &ys, &qs_out};
      tr_span(g, i, 4u * nc);
      if (T <= 16u) launch_t(b, K_INK_LC_WFG_GATE_NORM_R6, (dim + 31u) / 32u, n_ctrl, args);
      else launch_t(b, K_INK_LC_WFG_GATE_NORM_R6, (dim + 127u) / 128u, n_ctrl * ((T + 63u) / 64u), args);
      return 4u * nc;
    }
  }

  /* g11 bit 23: [OP_RMSNORM(ones), OP_RMSNORM(ones), OP_QUANT(2*dim)] -> ie_inkling_lc_prenorm_q8_r2 (T>1 prefill) */
  if (k0 == OP_RMSNORM && (T > 1u || (t1f & 1u)) && i + 2u < i1 && b->k[K_INK_LC_PRENORM_Q8_R2] && mk_ok_k(K_INK_LC_PRENORM_Q8_R2)) {
    const op *o1 = &g->ops[i + 1u], *o2 = &g->ops[i + 2u];
    if (o1->kind == OP_RMSNORM && o2->kind == OP_QUANT &&
        o0->v == &m->ones_dim && o1->v == &m->ones_dim && o0->qo < 0 && o1->qo < 0 &&
        o0->b >= 0 && o0->b == o1->b && o0->b == o2->a &&
        o0->boff == 0u && o1->boff == o0->n * 4u && o2->aoff == 0u &&
        o0->n == o1->n && o2->n == 2u * o0->n && o0->n >= 32u && (o0->n & 31u) == 0u) {
      void *inp = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *msg = b->arena + g->bufs[o1->a].off + o1->aoff;
      void *q = b->arena + g->bufs[o2->b].off + o2->boff;
      void *cat = NULL;
      u32 dim = o0->n;
      float eps = m->eps;
      u32 is = g->bufs[o0->a].stride, ms = g->bufs[o1->a].stride, qs = g->bufs[o2->b].stride, cs = 0u;
      if (((uintptr_t)inp & 15u) == 0u && ((uintptr_t)msg & 15u) == 0u &&
          (T == 1u || ((is | ms) & 15u) == 0u)) {
        void *args[] = {&inp, &msg, &q, &cat, &dim, &eps, &is, &ms, &qs, &cs};
        tr_span(g, i, 3u);
        launch_t(b, K_INK_LC_PRENORM_Q8_R2, 2u, T, args);
        return 3u;
      }
    }
  }

  /* g11 bit 24: [OP_QUANT(64), OP_GEMV_Q8(wfg 2*dim x 64, nb=2), OP_LOOP_GATE] -> ie_inkling_lc_wfg_gate_r2 (T>=2) */
  if (k0 == OP_QUANT && i + 2u < i1 && b->k[K_INK_LC_WFG_GATE_R2] && mk_ok_k(K_INK_LC_WFG_GATE_R2) && (T >= 2u || (t1f & 2u))) {
    const op *o1 = &g->ops[i + 1u], *o2 = &g->ops[i + 2u];
    if (o1->kind == OP_GEMV_Q8 && o2->kind == OP_LOOP_GATE &&
        o1->a == o0->b && o1->aoff == 0u && o0->boff == 0u && !o1->v && o1->res < 0 && !o1->nv &&
        o2->gt == o1->b && o2->gtoff == 0u && o1->boff == 0u &&
        o1->w->nb == 2u && o0->n == 64u && o2->n > 0u && o1->w->rows == 2u * o2->n) {
      void *qw = b->d[i + 1u].w0, *qs = b->d[i + 1u].w1;
      void *h = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *inp = b->arena + g->bufs[o2->a].off + o2->aoff;
      void *msg = b->arena + g->bufs[o2->c].off + o2->coff;
      void *out = b->arena + g->bufs[o2->b].off + o2->boff;
      u32 dim = o2->n, nb = 2u, TT = T;
      u32 hs = g->bufs[o0->a].stride, is = g->bufs[o2->a].stride, ms = g->bufs[o2->c].stride, os = g->bufs[o2->b].stride;
      if (((uintptr_t)qw & 3u) == 0u && ((uintptr_t)qs & 3u) == 0u &&
          (T == 1u || ((hs | is | ms | os) & 3u) == 0u)) {
        void *args[] = {&qw, &qs, &h, &inp, &msg, &out, &dim, &nb, &TT, &hs, &is, &ms, &os};
        tr_span(g, i, 3u);
        if (T == 1u) launch_t(b, K_INK_LC_WFG_GATE_R2, (dim + 31u) / 32u, 1u, args);
        else launch_t(b, K_INK_LC_WFG_GATE_R2, (dim + 127u) / 128u, (T + 63u) / 64u, args);
        return 3u;
      }
    }
  }

  /* R7 Fusion C1 bit 61: T=1..16 [sconv(k), vclip(k), sconv(v), vclip(v), kv_store] (+ inline sc_hist/ring) */
  const int sc_tree = b->tr_on && b->ink_tree_split > 0u && b->ink_tree_split < T;
  if (k0 == OP_SCONV && !sc_tree && !mk.rec && !ctx_on && T >= 1u && T <= 16u && !b->kvq8 && !b->kvq4 &&
      m->n_kv == 8u && m->hd == 128u && o0->n == 1024u &&
      b->k[K_INK_KV_SCONV_VCLIP_STORE_R7] && i + 4u < i1) {
    const op *okc = &g->ops[i + 1u], *ov = &g->ops[i + 2u], *ovc = &g->ops[i + 3u], *okv = &g->ops[i + 4u];
    if (okc->kind == OP_VCLIP && ov->kind == OP_SCONV && ovc->kind == OP_VCLIP && okv->kind == OP_KV &&
        okc->a == o0->b && okc->aoff == 0u && ov->n == 1024u && ovc->a == ov->b && ovc->aoff == 0u &&
        okv->a == okc->b && okv->b == ovc->b && o0->escale == ov->escale &&
        okc->clamp > 0.0f && !okc->v && !okc->norope && !ovc->v && ovc->clamp == 0.0f &&
        g->kv_len[okv->kvi] > 0u) {
      void *ka = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *vk = b->d[i].w0;
      void *ck = o0->c >= 0 ? (b->arena + g->bufs[o0->c].off + o0->coff) : NULL;
      float *ring_k = b->sc_ring + (size_t)o0->scoff * 4u;
      float *hist_k = (T > 1u && b->sc_defer) ? (b->sc_hist + (size_t)o0->scoff) : NULL;
      void *va = b->arena + g->bufs[ov->a].off + ov->aoff;
      void *vv = b->d[i + 2u].w0;
      void *cv = ov->c >= 0 ? (b->arena + g->bufs[ov->c].off + ov->coff) : NULL;
      float *ring_v = b->sc_ring + (size_t)ov->scoff * 4u;
      float *hist_v = (T > 1u && b->sc_defer) ? (b->sc_hist + (size_t)ov->scoff) : NULL;
      void *kc0 = kv_ptr(b, 0, okv->kvi, 0), *vc0 = kv_ptr(b, 1, okv->kvi, 0);
      u32 kas = g->bufs[o0->a].stride, cks = o0->c >= 0 ? g->bufs[o0->c].stride : 0u;
      u32 vas = g->bufs[ov->a].stride, cvs = ov->c >= 0 ? g->bufs[ov->c].stride : 0u;
      u32 hrs = hist_k ? b->sc_hist_stride : 0u;
      if (((uintptr_t)ka & 15u) == 0u && ((uintptr_t)vk & 15u) == 0u && ((uintptr_t)ring_k & 15u) == 0u &&
          ((uintptr_t)va & 15u) == 0u && ((uintptr_t)vv & 15u) == 0u && ((uintptr_t)ring_v & 15u) == 0u &&
          ((uintptr_t)kc0 & 15u) == 0u && ((uintptr_t)vc0 & 15u) == 0u &&
          (!ck || ((uintptr_t)ck & 15u) == 0u) && (!cv || ((uintptr_t)cv & 15u) == 0u) &&
          (!hist_k || (((uintptr_t)hist_k & 15u) == 0u && ((uintptr_t)hist_v & 15u) == 0u && (hrs & 3u) == 0u)) &&
          (T == 1u || ((kas | vas | (ck ? cks : 0u) | (cv ? cvs : 0u)) & 15u) == 0u)) {
        float sc_esc = o0->escale, k_clamp = okc->clamp, v_eps = ovc->escale;
        u32 kvlen = g->kv_len[okv->kvi], TT = T;
        void *fa[] = {&ka, &vk, &ck, &ring_k, &hist_k, &va, &vv, &cv, &ring_v, &hist_v,
                      &kc0, &vc0, &sc_esc, &k_clamp, &v_eps, &pos, &TT, &kvlen,
                      &kas, &cks, &vas, &cvs, &hrs};
        tr_span(g, i, 5u);
        launch(b, K_INK_KV_SCONV_VCLIP_STORE_R7, 1u, fa);
        return 5u;
      }
    }
  }

  /* g12 bits 25/26 & R8 bit 75: 5-op KV pipeline [sconv(k), vclip(k), (opt v_dyn), sconv(v), vclip(v), kv_store] (T>1 prefill or graph ctx) */
  if (k0 == OP_SCONV && !sc_tree && !mk.rec && (T > 1u || ctx_on || (t1f & 4u)) && !b->kvq8 && !b->kvq4 && m->n_kv == 8u && m->hd == 128u && o0->n == 1024u) {
    const int use_r8_kv = T >= 17u && !ctx_on && !b->sc_defer &&
                          b->k[K_INK_KV_SCONV_VCLIP_STORE_PF_R8] != NULL && b->ink_sconv_rcnt != NULL;
    const int have_kv = use_r8_kv ||
                        (ctx_on ? (b->k[K_INK_KV_SCONV_VCLIP_STORE_CTX_R2] != NULL)
                                : (b->k[K_INK_KV_SCONV_VCLIP_STORE_R2] != NULL &&
                                   (T == 1u || b->k[K_INK_KV_RING_STORE_R2] != NULL)));
    if (have_kv) {
      u32 n_span = 0u;
      int has_vdyn = 0;
      const op *okc = NULL, *ov = NULL, *ovc = NULL, *okv = NULL;
      if (i + 4u < i1 && g->ops[i + 1u].kind == OP_VCLIP && g->ops[i + 2u].kind == OP_SCONV &&
          g->ops[i + 3u].kind == OP_VCLIP && g->ops[i + 4u].kind == OP_KV) {
        okc = &g->ops[i + 1u]; ov = &g->ops[i + 2u]; ovc = &g->ops[i + 3u]; okv = &g->ops[i + 4u];
        n_span = 5u;
      } else if (i + 5u < i1 && g->ops[i + 1u].kind == OP_VCLIP && g->ops[i + 2u].kind == OP_GEMV_Q8 &&
                 g->ops[i + 3u].kind == OP_SCONV && g->ops[i + 4u].kind == OP_VCLIP && g->ops[i + 5u].kind == OP_KV) {
        const op *ovd = &g->ops[i + 2u];
        okc = &g->ops[i + 1u]; ov = &g->ops[i + 3u]; ovc = &g->ops[i + 4u]; okv = &g->ops[i + 5u];
        if (ovd->b == ov->c && ovd->a != o0->b && ovd->a != okc->b) {
          has_vdyn = 1;
          n_span = 6u;
        }
      }
      if (n_span && okc->a == o0->b && okc->aoff == 0u && ov->n == 1024u && ovc->a == ov->b && ovc->aoff == 0u &&
          okv->a == okc->b && okv->b == ovc->b && o0->escale == ov->escale &&
          okc->clamp > 0.0f && !okc->v && !okc->norope && !ovc->v && ovc->clamp == 0.0f &&
          g->kv_len[okv->kvi] > 0u) {
        const u32 iv_idx = has_vdyn ? (i + 3u) : (i + 2u);
        void *ka = b->arena + g->bufs[o0->a].off + o0->aoff;
        void *vk = b->d[i].w0;
        void *ck = o0->c >= 0 ? (b->arena + g->bufs[o0->c].off + o0->coff) : NULL;
        float *ring_k = b->sc_ring + (size_t)o0->scoff * 4u;
        void *va = b->arena + g->bufs[ov->a].off + ov->aoff;
        void *vv = b->d[iv_idx].w0;
        void *cv = ov->c >= 0 ? (b->arena + g->bufs[ov->c].off + ov->coff) : NULL;
        float *ring_v = b->sc_ring + (size_t)ov->scoff * 4u;
        void *kc0 = kv_ptr(b, 0, okv->kvi, 0), *vc0 = kv_ptr(b, 1, okv->kvi, 0);
        u32 kas = g->bufs[o0->a].stride, cks = o0->c >= 0 ? g->bufs[o0->c].stride : 0u;
        u32 vas = g->bufs[ov->a].stride, cvs = ov->c >= 0 ? g->bufs[ov->c].stride : 0u;
        if (((uintptr_t)ka & 15u) == 0u && ((uintptr_t)vk & 15u) == 0u && ((uintptr_t)ring_k & 15u) == 0u &&
            ((uintptr_t)va & 15u) == 0u && ((uintptr_t)vv & 15u) == 0u && ((uintptr_t)ring_v & 15u) == 0u &&
            ((uintptr_t)kc0 & 15u) == 0u && ((uintptr_t)vc0 & 15u) == 0u &&
            (!ck || ((uintptr_t)ck & 15u) == 0u) && (!cv || ((uintptr_t)cv & 15u) == 0u) &&
            (T == 1u || ((kas | vas | (ck ? cks : 0u) | (cv ? cvs : 0u)) & 15u) == 0u)) {
          if (has_vdyn) {
            tr_span(g, i + 2u, 1u);
            launch_op(b, i + 2u, 0, toks[0], pos, T, slot, wfrom);
          }
          float sc_esc = o0->escale, k_clamp = okc->clamp, v_eps = ovc->escale;
          u32 kvlen = g->kv_len[okv->kvi], TT = T;
          tr_span(g, i, n_span);
          if (use_r8_kv) {
            void *fa[] = {&ka, &vk, &ck, &ring_k, &va, &vv, &cv, &ring_v, &kc0, &vc0,
                          &sc_esc, &k_clamp, &v_eps, &pos, &TT, &kvlen, &kas, &cks, &vas, &cvs,
                          &b->ink_sconv_rcnt};
            launch_t(b, K_INK_KV_SCONV_VCLIP_STORE_PF_R8, 1u, (T + 31u) / 32u, fa);
          } else if (ctx_on) {
            void *ca[] = {&ka, &vk, &ck, &ring_k, &va, &vv, &cv, &ring_v, &kc0, &vc0,
                          &sc_esc, &k_clamp, &v_eps, &b->ink_ctx_dev, &kvlen};
            launch(b, K_INK_KV_SCONV_VCLIP_STORE_CTX_R2, 1u, ca);
          } else {
            void *fa[] = {&ka, &vk, &ck, &ring_k, &va, &vv, &cv, &ring_v, &kc0, &vc0,
                          &sc_esc, &k_clamp, &v_eps, &pos, &TT, &kvlen, &kas, &cks, &vas, &cvs};
            launch_t(b, K_INK_KV_SCONV_VCLIP_STORE_R2, 1u, T == 1u ? 1u : (T + 3u) / 4u, fa);
            if (T > 1u) {
              if (b->sc_defer) {
                float *hpk = b->sc_hist + (size_t)o0->scoff;
                float *hpv = b->sc_hist + (size_t)ov->scoff;
                u32 nk = o0->n, nv = ov->n, hrs = b->sc_hist_stride;
                void *hak[] = {&ka, &hpk, &nk, &TT, &kas, &hrs};
                void *hav[] = {&va, &hpv, &nv, &TT, &vas, &hrs};
                launch_t(b, K_INK_SCONV_HIST_R4, ((nk >> 2u) + 255u) / 256u, T, hak);
                launch_t(b, K_INK_SCONV_HIST_R4, ((nv >> 2u) + 255u) / 256u, T, hav);
              } else {
                void *ra[] = {&ka, &va, &ring_k, &ring_v, &pos, &TT, &kas, &vas};
                launch(b, K_INK_KV_RING_STORE_R2, 1u, ra);
              }
            }
          }
          return n_span;
        }
      }
    }
  }

  /* R7 Fusion B2 bit 63: 4-op [OP_GEMV_EXP(IQ4_NL), OP_MOE_SUM, OP_SCONV(c=-1), OP_ADD] (+ inline sc_hist/ring) */
  if (k0 == OP_GEMV_EXP && !sc_tree && !mk.rec && !ctx_on && T >= 1u && T <= 16u && i + 3u < i1 &&
      o0->w->kind == MAT_IQ4NL && o0->boff == 0u && b->ink_moe_rcount &&
      ((T == 1u && b->k[K_INK_EXP_DOWN_TAIL_T1_R7]) || (T >= 2u && b->k[K_INK_EXP_DOWN_TAIL_VER_R7]))) {
    /* IE_INK_MOE_TAIL_MODE: 0=hybrid (B1 on T==1, B2 on T=2..16), 1=B1 all, 2=B2 all */
    static int moe_tail_mode = -1;
    if (moe_tail_mode < 0) {
      const char *e = getenv("IE_INK_MOE_TAIL_MODE");
      moe_tail_mode = e ? atoi(e) : 0;
    }
    const int adap_nob2 = (T >= 2u && b->k[K_INK_MOE_SORT_ADAP_R8] && ink_moe_adap_mode() == 1);
    const int allow_b2 = !adap_nob2 && ((moe_tail_mode == 2) || (moe_tail_mode == 0 && T >= 2u));
    const op *o1 = &g->ops[i + 1u], *o2 = &g->ops[i + 2u], *o3 = &g->ops[i + 3u];
    const u32 rows = o0->w->rows, nb = o0->w->nb, K = o0->nh;
    if (allow_b2 && rows > 0u && (rows & 3u) == 0u && nb > 0u && K > 0u && o0->bstr == rows * 4u &&
        o1->kind == OP_MOE_SUM && o1->res < 0 && o1->a == o0->b && o1->aoff == 0u &&
        o1->c == o0->c && o1->coff == o0->coff && o1->n == rows && o1->nh == K &&
        o2->kind == OP_SCONV && o2->c < 0 && o2->a == o1->b && o2->aoff == 0u && o2->n == rows &&
        o3->kind == OP_ADD && o3->n == rows && (o3->a == o2->b || o3->b == o2->b) &&
        buf_disjoint(g, o0->a, o3->c) && buf_disjoint(g, o0->b, o3->c) && buf_disjoint(g, o0->c, o3->c) &&
        buf_disjoint(g, o0->a, o0->b) && buf_disjoint(g, o0->c, o0->b)) {
      const int rb = (o3->a == o2->b) ? o3->b : o3->a;
      const u32 roff = (o3->a == o2->b) ? o3->boff : o3->aoff;
      void *qw = b->d[i].w0, *qs = b->d[i].w1;
      void *xq = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *yd = b->arena + g->bufs[o0->b].off + o0->boff;
      void *rt = b->arena + g->bufs[o0->c].off + o0->coff;
      void *v = b->d[i + 2u].w0;
      float *ring = b->sc_ring + (size_t)o2->scoff * 4u;
      void *res = b->arena + g->bufs[rb].off + roff;
      void *out = b->arena + g->bufs[o3->c].off + o3->coff;
      u32 r_u = rows, nb_u = nb, K_u = K;
      u32 qwn = (u32)mat_qw_words(o0->w->kind, rows, o0->w->cols);
      u32 qsn = (u32)mat_qs_n(o0->w->kind, rows, o0->w->cols);
      float esc = o2->escale;
      if (((uintptr_t)yd & 15u) == 0u && ((uintptr_t)v & 15u) == 0u && ((uintptr_t)ring & 15u) == 0u &&
          ((uintptr_t)res & 15u) == 0u && ((uintptr_t)out & 15u) == 0u) {
        if (T == 1u) {
          u32 astr = o0->astr, bstr = o0->bstr;
          void *fa[] = {&qw, &qs, &xq, &yd, &r_u, &nb_u, &rt, &K_u, &qwn, &qsn, &astr, &bstr,
                        &v, &ring, &res, &out, &esc, &pos, &b->ink_moe_rcount};
          tr_span(g, i, 4u);
          launch_t(b, K_INK_EXP_DOWN_TAIL_T1_R7, (rows + 31u) / 32u, K, fa);
          return 4u;
        } else {
          const int skip_sort = T <= ink_moe_slot_max(b) && b->ink_moe_all_slot &&
                                b->k[K_INK_GEXP_RAW_IQ3XXS_SLOT_R6] && b->k[K_INK_GEXP_IQ4NL_SLOT_R6];
          const u32 mp_a = T * K, mp_b = m->n_exp + (T * K) / 4u;
          const u32 max_packs = mp_a < mp_b ? mp_a : mp_b;
          const u32 at = g->bufs[o0->a].stride, ys = o0->bstr, yts = g->bufs[o0->b].stride;
          const u32 rts = g->bufs[o0->c].stride, rs = g->bufs[rb].stride, os = g->bufs[o3->c].stride;
          float *hist = b->sc_defer ? (b->sc_hist + (size_t)o2->scoff) : NULL;
          u32 hrs = hist ? b->sc_hist_stride : 0u;
          if (!skip_sort && b->k[K_INK_MOE_SORT_COMPACT_R6] && b->ms_list && b->ms_packs && b->ms_npacks &&
              m->n_exp <= 256u && T * K <= 4096u && max_packs <= 160u &&
              yts == K * ys && (!o0->astr || at == K * o0->astr) &&
              ((yts | rs | os) & 15u) == 0u &&
              (!hist || (((uintptr_t)hist & 15u) == 0u && (hrs & 3u) == 0u))) {
            u32 xs = o0->astr ? o0->astr : at, ys_u = ys, xdiv = o0->astr ? 1u : K;
            u32 TT = T, rts_u = rts, yts_u = yts, rs_u = rs, os_u = os;
            void *fa[] = {&qw, &qs, &xq, &yd, &r_u, &nb_u,
                          &b->ms_list, &b->ms_packs, &b->ms_npacks,
                          &qwn, &qsn, &xs, &ys_u, &xdiv,
                          &rt, &K_u, &rts_u, &v, &ring, &res, &out, &hist,
                          &esc, &pos, &TT, &yts_u, &rs_u, &os_u, &hrs, &b->ink_moe_rcount};
            tr_span(g, i, 4u);
            launch_t(b, K_INK_EXP_DOWN_TAIL_VER_R7, (rows + 31u) / 32u, max_packs, fa);
            return 4u;
          }
        }
      }
    }
  }

  /* R7 Fusion B1 bit 63: 3-op [OP_MOE_SUM(res=-1), OP_SCONV(c=-1), OP_ADD] (+ inline sc_hist/ring) */
  if (k0 == OP_MOE_SUM && !sc_tree && !mk.rec && !ctx_on && T >= 1u && T <= 16u && i + 2u < i1 && o0->res < 0 &&
      b->k[K_INK_MOE_SUM_SCONV_ADD_R7]) {
    const op *o1 = &g->ops[i + 1u], *o2 = &g->ops[i + 2u];
    if (o1->kind == OP_SCONV && o1->c < 0 && o1->a == o0->b && o1->aoff == 0u && o1->n == o0->n &&
        o2->kind == OP_ADD && o2->n == o0->n && (o2->a == o1->b || o2->b == o1->b) &&
        (o0->n & 3u) == 0u && o0->nh > 0u &&
        buf_disjoint(g, o0->a, o2->c) && buf_disjoint(g, o0->c, o2->c)) {
      const int rb = (o2->a == o1->b) ? o2->b : o2->a;
      const u32 roff = (o2->a == o1->b) ? o2->boff : o2->aoff;
      void *y = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *rt = b->arena + g->bufs[o0->c].off + o0->coff;
      void *v = b->d[i + 1u].w0;
      float *ring = b->sc_ring + (size_t)o1->scoff * 4u;
      void *res = b->arena + g->bufs[rb].off + roff;
      void *out = b->arena + g->bufs[o2->c].off + o2->coff;
      float *hist = (T > 1u && b->sc_defer) ? (b->sc_hist + (size_t)o1->scoff) : NULL;
      u32 n = o0->n, K = o0->nh, TT = T;
      u32 ys = g->bufs[o0->a].stride, rts = g->bufs[o0->c].stride;
      u32 rs = g->bufs[rb].stride, os = g->bufs[o2->c].stride;
      u32 hrs = hist ? b->sc_hist_stride : 0u;
      float esc = o1->escale;
      if (((uintptr_t)y & 15u) == 0u && ((uintptr_t)v & 15u) == 0u && ((uintptr_t)ring & 15u) == 0u &&
          ((uintptr_t)res & 15u) == 0u && ((uintptr_t)out & 15u) == 0u &&
          (T == 1u || ((ys | rs | os) & 15u) == 0u) &&
          (!hist || (((uintptr_t)hist & 15u) == 0u && (hrs & 3u) == 0u))) {
        void *fa[] = {&y, &rt, &v, &ring, &res, &out, &hist, &n, &K, &esc, &pos, &TT, &ys, &rts, &rs, &os, &hrs};
        tr_span(g, i, 3u);
        launch(b, K_INK_MOE_SUM_SCONV_ADD_R7, ((n >> 2u) + 255u) / 256u, fa);
        return 3u;
      }
    }
  }

  /* g13 bit 28 (or g16 bit 34 ctx): T==1 [OP_MOE_SUM(res=-1), OP_SCONV(c=-1), OP_ADD] -> moe_sum_sconv_add */
  if (k0 == OP_MOE_SUM && !mk.rec && T == 1u && i + 2u < i1 && o0->res < 0 &&
      (ctx_on ? (b->k[K_INK_MOE_SUM_SCONV_ADD_CTX_R2] != NULL) : (b->k[K_INK_MOE_SUM_SCONV_ADD_R2] != NULL))) {
    const op *o1 = &g->ops[i + 1u], *o2 = &g->ops[i + 2u];
    if (o1->kind == OP_SCONV && o1->c < 0 && o1->a == o0->b && o1->aoff == 0u && o1->n == o0->n &&
        o2->kind == OP_ADD && o2->n == o0->n && (o2->a == o1->b || o2->b == o1->b) && (o0->n & 3u) == 0u && o0->nh > 0u &&
        buf_disjoint(g, o0->c, o2->c)) {
      const int rb = (o2->a == o1->b) ? o2->b : o2->a;
      const u32 roff = (o2->a == o1->b) ? o2->boff : o2->aoff;
      void *y = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *rt = b->arena + g->bufs[o0->c].off + o0->coff;
      void *v = b->d[i + 1u].w0;
      float *ring = b->sc_ring + (size_t)o1->scoff * 4u;
      void *res = b->arena + g->bufs[rb].off + roff;
      void *out = b->arena + g->bufs[o2->c].off + o2->coff;
      u32 n = o0->n, K = o0->nh;
      float esc = o1->escale;
      if (((uintptr_t)y & 15u) == 0u && ((uintptr_t)v & 15u) == 0u && ((uintptr_t)ring & 15u) == 0u &&
          ((uintptr_t)res & 15u) == 0u && ((uintptr_t)out & 15u) == 0u) {
        tr_span(g, i, 3u);
        if (ctx_on) {
          void *ca[] = {&y, &rt, &v, &ring, &res, &out, &n, &K, &esc, &b->ink_ctx_dev};
          launch(b, K_INK_MOE_SUM_SCONV_ADD_CTX_R2, ((n >> 2u) + 255u) / 256u, ca);
        } else {
          void *fa[] = {&y, &rt, &v, &ring, &res, &out, &n, &K, &esc, &pos};
          launch(b, K_INK_MOE_SUM_SCONV_ADD_R2, ((n >> 2u) + 255u) / 256u, fa);
        }
        return 3u;
      }
    }
  }

  /* R7/R9 Fusion E3 (bit 68 r7 / bit 98 r9): T=1..16 [OP_GEMV_Q8(wo), OP_SCONV(c=-1), OP_ADD | OP_COPY] (+ inline sc_hist/ring) */
  if (k0 == OP_GEMV_Q8 && !sc_tree && !mk.rec && !ctx_on && T >= 1u && T <= 16u && i + 2u < i1 &&
      ((T == 1u && b->k[K_INK_WO_SCONV_POST_T1_R9]) || (T >= 2u && b->k[K_INK_WO_SCONV_POST_T8_R9]) || b->k[K_INK_WO_SCONV_POST_R7]) &&
      !o0->v && o0->res < 0 && !o0->nv && o0->boff == 0u &&
      o0->w->rows > 0u && o0->w->nb >= 8u && (o0->w->nb & 7u) == 0u) {
    static int wo_r9_mode = -1;
    if (wo_r9_mode < 0) wo_r9_mode = getenv("IE_INK_WO_R9_MODE") ? atoi(getenv("IE_INK_WO_R9_MODE")) : 1; /* 1=T1 only, 2=T2..16 only, 3=both */
    const int use_r9_t1 = (T == 1u && b->k[K_INK_WO_SCONV_POST_T1_R9] && (wo_r9_mode & 1));
    const int use_r9_t8 = (T >= 2u && b->k[K_INK_WO_SCONV_POST_T8_R9] && (wo_r9_mode & 2));
    if (use_r9_t1 || use_r9_t8 || b->k[K_INK_WO_SCONV_POST_R7]) {
    const op *o1 = &g->ops[i + 1u], *o2 = &g->ops[i + 2u];
    if (o1->kind == OP_SCONV && o1->c < 0 && o1->a == o0->b && o1->aoff == 0u && o1->n == o0->w->rows &&
        g->bufs[o0->b].last <= i + 2u) {
      const int is_add = o2->kind == OP_ADD && o2->n == o1->n && (o2->a == o1->b || o2->b == o1->b) &&
                         g->bufs[o1->b].last <= i + 2u && buf_disjoint(g, o0->a, o2->c);
      const int is_cpy = !is_add && o2->kind == OP_COPY && o2->n == o1->n && o2->a != o1->b && o2->b != o1->b &&
                         buf_disjoint(g, o0->a, o1->b) && buf_disjoint(g, o0->a, o2->b) && buf_disjoint(g, o1->b, o2->b);
      if (is_add || is_cpy) {
        const int rb = is_add ? ((o2->a == o1->b) ? o2->b : o2->a) : o2->a;
        const u32 roff = is_add ? ((o2->a == o1->b) ? o2->boff : o2->aoff) : o2->aoff;
        const int ob = is_add ? o2->c : o1->b;
        const u32 ooff = is_add ? o2->coff : o1->boff;
        if (buf_disjoint(g, rb, ob) && (!is_cpy || buf_disjoint(g, rb, o2->b))) {
          void *qw = b->d[i].w0, *qs = b->d[i].w1;
          void *xq = b->arena + g->bufs[o0->a].off + o0->aoff;
          void *v = b->d[i + 1u].w0;
          float *ring = b->sc_ring + (size_t)o1->scoff * 4u;
          void *res = b->arena + g->bufs[rb].off + roff;
          void *out = b->arena + g->bufs[ob].off + ooff;
          void *blk_dst = is_cpy ? (b->arena + g->bufs[o2->b].off + o2->boff) : NULL;
          float *hist = (T > 1u && b->sc_defer) ? (b->sc_hist + (size_t)o1->scoff) : NULL;
          u32 dim = o0->w->rows, nb = o0->w->nb, TT = T;
          float esc = o1->escale;
          u32 xs = g->bufs[o0->a].stride, rs = g->bufs[rb].stride, os = g->bufs[ob].stride;
          u32 bds = is_cpy ? g->bufs[o2->b].stride : 0u;
          u32 hrs = hist ? b->sc_hist_stride : 0u;
          if (((uintptr_t)v & 15u) == 0u && ((uintptr_t)ring & 15u) == 0u) {
            void *fa[] = {&qw, &qs, &xq, &v, &ring, &res, &out, &blk_dst, &hist,
                          &dim, &nb, &esc, &pos, &TT, &xs, &rs, &os, &bds, &hrs};
            const int kw = use_r9_t1 ? K_INK_WO_SCONV_POST_T1_R9 : (use_r9_t8 ? K_INK_WO_SCONV_POST_T8_R9 : K_INK_WO_SCONV_POST_R7);
            u32 gx = (kw == K_INK_WO_SCONV_POST_T8_R9) ? ((dim + 15u) / 16u) : ((dim + 31u) / 32u);
            if (gx >= 253u && gx <= 256u) gx = 257u;
            tr_span(g, i, 3u);
            launch(b, kw, gx, fa);
            return 3u;
          }
        }
      }
    }
    }
  }

  /* R7 Fusion C2 bit 62: T=1..16 [OP_SCONV(c=-1), OP_ADD | OP_COPY] (+ inline sc_hist/ring) -> ie_inkling_sconv_post_r7 */
  if (k0 == OP_SCONV && !sc_tree && !mk.rec && !ctx_on && T >= 1u && T <= 16u && o0->c < 0 && (o0->n & 3u) == 0u &&
      i + 1u < i1 && b->k[K_INK_SCONV_POST_R7]) {
    const op *o1 = &g->ops[i + 1u];
    const int is_add = o1->kind == OP_ADD && o1->n == o0->n && (o1->a == o0->b || o1->b == o0->b) &&
                       (T == 1u || buf_disjoint(g, o0->a, o1->c));
    const int is_cpy = !is_add && o1->kind == OP_COPY && o1->n == o0->n && o1->a != o0->b && o1->b != o0->b &&
                       buf_disjoint(g, o0->a, o1->b) && buf_disjoint(g, o0->b, o1->b) &&
                       (T == 1u || buf_disjoint(g, o0->a, o0->b));
    if (is_add || is_cpy) {
      const int rb = is_add ? ((o1->a == o0->b) ? o1->b : o1->a) : o1->a;
      const u32 roff = is_add ? ((o1->a == o0->b) ? o1->boff : o1->aoff) : o1->aoff;
      const int ob = is_add ? o1->c : o0->b;
      const u32 ooff = is_add ? o1->coff : o0->boff;
      void *A = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *v = b->d[i].w0;
      float *ring = b->sc_ring + (size_t)o0->scoff * 4u;
      void *res = b->arena + g->bufs[rb].off + roff;
      void *out = b->arena + g->bufs[ob].off + ooff;
      void *blk_dst = is_cpy ? (b->arena + g->bufs[o1->b].off + o1->boff) : NULL;
      float *hist = (T > 1u && b->sc_defer) ? (b->sc_hist + (size_t)o0->scoff) : NULL;
      u32 n = o0->n, TT = T;
      u32 as = g->bufs[o0->a].stride, rs = g->bufs[rb].stride, os = g->bufs[ob].stride;
      u32 bds = is_cpy ? g->bufs[o1->b].stride : 0u;
      u32 hrs = hist ? b->sc_hist_stride : 0u;
      float esc = o0->escale;
      if (((uintptr_t)A & 15u) == 0u && ((uintptr_t)v & 15u) == 0u && ((uintptr_t)ring & 15u) == 0u &&
          ((uintptr_t)res & 15u) == 0u && ((uintptr_t)out & 15u) == 0u &&
          (!blk_dst || ((uintptr_t)blk_dst & 15u) == 0u) &&
          (!hist || (((uintptr_t)hist & 15u) == 0u && (hrs & 3u) == 0u)) &&
          (T == 1u || ((as | rs | os | (blk_dst ? bds : 0u)) & 15u) == 0u)) {
        void *fa[] = {&A, &v, &ring, &res, &out, &blk_dst, &hist, &n, &esc, &pos, &TT, &as, &rs, &os, &bds, &hrs};
        tr_span(g, i, 2u);
        launch(b, K_INK_SCONV_POST_R7, ((n >> 2u) + 255u) / 256u, fa);
        return 2u;
      }
    }
  }

  /* R8 Wave 3 bit 75: T >= 17 prefill [OP_SCONV(c=-1), OP_ADD | OP_COPY] + race-free ring commit -> ie_inkling_sconv_post_pf_r8 */
  if (k0 == OP_SCONV && !mk.rec && !ctx_on && !b->sc_defer && T >= 17u && o0->c < 0 && (o0->n & 3u) == 0u &&
      i + 1u < i1 && b->k[K_INK_SCONV_POST_PF_R8] && b->ink_sconv_rcnt &&
      (((o0->n >> 2u) + 255u) / 256u) <= 16u) {
    const op *o1 = &g->ops[i + 1u];
    const int is_add = o1->kind == OP_ADD && o1->n == o0->n && (o1->a == o0->b || o1->b == o0->b) &&
                       buf_disjoint(g, o0->a, o1->c);
    const int is_cpy = !is_add && o1->kind == OP_COPY && o1->n == o0->n && o1->a != o0->b && o1->b != o0->b &&
                       buf_disjoint(g, o0->a, o1->b) && buf_disjoint(g, o0->b, o1->b) && buf_disjoint(g, o0->a, o0->b);
    if (is_add || is_cpy) {
      const int rb = is_add ? ((o1->a == o0->b) ? o1->b : o1->a) : o1->a;
      const u32 roff = is_add ? ((o1->a == o0->b) ? o1->boff : o1->aoff) : o1->aoff;
      const int ob = is_add ? o1->c : o0->b;
      const u32 ooff = is_add ? o1->coff : o0->boff;
      void *A = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *v = b->d[i].w0;
      float *ring = b->sc_ring + (size_t)o0->scoff * 4u;
      void *res = b->arena + g->bufs[rb].off + roff;
      void *out = b->arena + g->bufs[ob].off + ooff;
      void *blk_dst = is_cpy ? (b->arena + g->bufs[o1->b].off + o1->boff) : NULL;
      u32 n = o0->n, TT = T;
      u32 as = g->bufs[o0->a].stride, rs = g->bufs[rb].stride, os = g->bufs[ob].stride;
      u32 bds = is_cpy ? g->bufs[o1->b].stride : 0u;
      float esc = o0->escale;
      if (((uintptr_t)A & 15u) == 0u && ((uintptr_t)v & 15u) == 0u && ((uintptr_t)ring & 15u) == 0u &&
          ((uintptr_t)res & 15u) == 0u && ((uintptr_t)out & 15u) == 0u &&
          (!blk_dst || ((uintptr_t)blk_dst & 15u) == 0u) &&
          ((as | rs | os | (blk_dst ? bds : 0u)) & 15u) == 0u) {
        void *fa[] = {&A, &v, &ring, &res, &out, &blk_dst, &n, &esc, &pos, &TT, &as, &rs, &os, &bds, &b->ink_sconv_rcnt};
        tr_span(g, i, 2u);
        launch_t(b, K_INK_SCONV_POST_PF_R8, ((n >> 2u) + 255u) / 256u, (T + 31u) / 32u, fa);
        return 2u;
      }
    }
  }

  /* g13 bit 27 (or g16 bit 34 ctx): [OP_SCONV, OP_ADD] -> ie_inkling_sconv_add_r2 (T>1 prefill or graph ctx) */
  if (k0 == OP_SCONV && !sc_tree && (T > 1u || ctx_on || (t1f & 8u)) && i + 1u < i1 &&
      (ctx_on ? (b->k[K_INK_SCONV_ADD_CTX_R2] != NULL && !mk.rec) : (b->k[K_INK_SCONV_ADD_R2] != NULL && mk_ok_k(K_INK_SCONV_ADD_R2)))) {
    const op *o1 = &g->ops[i + 1u];
    if (o1->kind == OP_ADD && o1->n == o0->n && (o1->a == o0->b || o1->b == o0->b) && (o0->n & 3u) == 0u &&
        (T == 1u || buf_disjoint(g, o0->a, o1->c))) {
      const int rb = (o1->a == o0->b) ? o1->b : o1->a;
      const u32 roff = (o1->a == o0->b) ? o1->boff : o1->aoff;
      void *A = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *v = b->d[i].w0;
      void *C = o0->c >= 0 ? (b->arena + g->bufs[o0->c].off + o0->coff) : NULL;
      float *ring = b->sc_ring + (size_t)o0->scoff * 4u;
      void *res = b->arena + g->bufs[rb].off + roff;
      void *out = b->arena + g->bufs[o1->c].off + o1->coff;
      u32 n = o0->n, nh = o0->nh, TT = T;
      u32 as = g->bufs[o0->a].stride, cs = o0->c >= 0 ? g->bufs[o0->c].stride : 0u;
      u32 rs = g->bufs[rb].stride, os = g->bufs[o1->c].stride;
      float esc = o0->escale;
      if ((!C || (nh > 0u && ((n / nh) & 3u) == 0u)) &&
          ((uintptr_t)A & 15u) == 0u && ((uintptr_t)v & 15u) == 0u && ((uintptr_t)ring & 15u) == 0u &&
          ((uintptr_t)res & 15u) == 0u && ((uintptr_t)out & 15u) == 0u && (!C || ((uintptr_t)C & 15u) == 0u) &&
          (T == 1u || ((as | rs | os | (C ? cs : 0u)) & 15u) == 0u)) {
        tr_span(g, i, 2u);
        if (ctx_on) {
          void *ca[] = {&A, &v, &C, &ring, &res, &out, &n, &nh, &esc, &b->ink_ctx_dev};
          launch(b, K_INK_SCONV_ADD_CTX_R2, ((n >> 2u) + 255u) / 256u, ca);
        } else {
          void *fa[] = {&A, &v, &C, &ring, &res, &out, &n, &nh, &esc, &pos, &TT, &as, &cs, &rs, &os};
          launch_t(b, K_INK_SCONV_ADD_R2, ((n >> 2u) + 255u) / 256u, (T + 7u) / 8u, fa);
          if (T > 1u) {
            if (b->sc_defer) {
              float *hp = b->sc_hist + (size_t)o0->scoff;
              u32 hrs = b->sc_hist_stride;
              void *ha[] = {&A, &hp, &n, &TT, &as, &hrs};
              launch_t(b, K_INK_SCONV_HIST_R4, ((n >> 2u) + 255u) / 256u, T, ha);
            } else {
              void *ra[] = {&A, &ring, &n, &pos, &TT, &as};
              launch(b, K_SCONV_RING, (n + 255u) / 256u, ra);
            }
          }
        }
        return 2u;
      }
    }
  }

  /* R9 bit 102 / R7 bit 70 (Wave 3) / R6 bit 57: T >= 17 [OP_GEMV_EXP(IQ3_XXS egu), OP_SITU(Q8)] -> worklist r9/r7 or r6 */
  if (k0 == OP_GEMV_EXP && !mk.rec && T >= 17u && i + 1u < i1 &&
      (b->k[K_INK_GEMM_RQ_EXP_IQ3XXS_SITU_R9] || b->k[K_INK_GEMM_RQ_EXP_IQ3XXS_SITU_R7] || b->k[K_INK_GEMM_RQ_EXP_IQ3XXS_SITU_R6]) &&
      !getenv("IE_GEMM_RQ_EXP_NO") &&
      o0->w->kind == MAT_RAWQ && o0->w->sub == 18u) {
    const op *o1 = &g->ops[i + 1u];
    if (o1->kind == OP_SITU && o1->a == o0->b && o1->aoff == 0u && o0->boff == 0u &&
        o1->nh == o0->nh && o1->b < 0 && o1->qo >= 0 &&
        o1->n > 0u && (o1->n & 31u) == 0u && o0->w->rows == 2u * o1->n &&
        o0->w->nb >= 8u && (o0->w->nb & 7u) == 0u &&
        g->bufs[o1->qo].stride == o1->nh * o1->bstr && (o1->bstr & 3u) == 0u &&
        (o0->astr == 0u || g->bufs[o0->a].stride == o0->nh * o0->astr) &&
        buf_disjoint(g, o0->a, o1->qo)) {
      void *qw = b->d[i].w1;
      void *X = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *Qout = b->arena + g->bufs[o1->qo].off;
      u32 eff = o1->n, nb = o0->w->nb, np = 0u, q_bytes = (u32)ie_rawq_bytes(o0->w);
      u32 xs = o0->astr ? o0->astr : g->bufs[o0->a].stride;
      u32 qstr = o1->bstr, xdiv = o0->astr ? 1u : o0->nh;
      float beta = o1->clamp, lbeta = o1->lam;
      const u32 mt_a = T * o0->nh, mt_b = m->n_exp + (T * o0->nh) / 64u;
      const u32 max_tiles = mt_a < mt_b ? mt_a : mt_b;
      if ((b->k[K_INK_GEMM_RQ_EXP_IQ3XXS_SITU_R9] || b->k[K_INK_GEMM_RQ_EXP_IQ3XXS_SITU_R7]) &&
          b->k[K_INK_MOE_PF_WORKLIST_R7] &&
          b->ms_packs && b->ms_npacks && m->n_exp <= 256u && T * o0->nh <= 8192u && max_tiles <= 512u) {
        void *args[] = {&qw, &X, &Qout, &eff, &nb, &b->ms_list, &b->ms_packs, &b->ms_npacks, &q_bytes, &xs, &qstr, &xdiv, &beta, &lbeta};
        const int ki3 = b->k[K_INK_GEMM_RQ_EXP_IQ3XXS_SITU_R9] ? K_INK_GEMM_RQ_EXP_IQ3XXS_SITU_R9 : K_INK_GEMM_RQ_EXP_IQ3XXS_SITU_R7;
        tr_span(g, i, 2u);
        launch_t(b, ki3, (eff + 63u) / 64u, max_tiles, args);
        return 2u;
      } else if (b->k[K_INK_GEMM_RQ_EXP_IQ3XXS_SITU_R6]) {
        void *args[] = {&qw, &X, &Qout, &eff, &nb, &b->ms_list, &b->ms_offs, &np, &q_bytes, &xs, &qstr, &xdiv, &beta, &lbeta};
        tr_span(g, i, 2u);
        launch_t(b, K_INK_GEMM_RQ_EXP_IQ3XXS_SITU_R6, (eff + 63u) / 64u, m->n_exp, args);
        return 2u;
      }
    }
  }

  /* R7 Fusion E1 bit 66 / R8 Adaptive MoE bit 86: T=2..16 compact [OP_GEMV_EXP(IQ3_XXS egu), OP_SITU(Q8)] */
  if (k0 == OP_GEMV_EXP && !mk.rec && T >= 2u && T <= 16u && i + 1u < i1 &&
      ((b->k[K_INK_GEXP_IQ3XXS_SITU_NJ1_R8] && b->k[K_INK_GEXP_IQ3XXS_SITU_NJM_R8]) || b->k[K_INK_GEXP_IQ3XXS_SITU_COMPACT_R7]) &&
      o0->w->kind == MAT_RAWQ && o0->w->sub == 18u) {
    const op *o1 = &g->ops[i + 1u];
    const u32 K = o0->nh;
    const int skip_sort = T <= ink_moe_slot_max(b) && b->ink_moe_all_slot &&
                          b->k[K_INK_GEXP_RAW_IQ3XXS_SLOT_R6] && b->k[K_INK_GEXP_IQ4NL_SLOT_R6];
    const u32 mp_a = T * K, mp_b = m->n_exp + (T * K) / 4u;
    const u32 max_packs = mp_a < mp_b ? mp_a : mp_b;
    const int use_adap = b->k[K_INK_GEXP_IQ3XXS_SITU_NJ1_R8] && b->k[K_INK_GEXP_IQ3XXS_SITU_NJM_R8];
    const int sep_packs = use_adap && ink_moe_adap_mode() >= 1 && b->k[K_INK_MOE_SORT_ADAP_R8] &&
                          b->ms_packs1 && b->ms_npacks1 && b->ms_packsM && b->ms_npacksM;
    const int has_packs = sep_packs || (b->k[K_INK_MOE_SORT_COMPACT_R6] && b->ms_packs && b->ms_npacks);
    if (!skip_sort && has_packs && b->ms_list &&
        m->n_exp <= 256u && T * K <= 4096u && max_packs <= 160u &&
        o1->kind == OP_SITU && o1->a == o0->b && o1->aoff == 0u && o0->boff == 0u &&
        o1->nh == K && o1->b < 0 && o1->qo >= 0 && g->bufs[o0->b].last <= i + 1u &&
        o1->n > 0u && (o1->n & 31u) == 0u && o0->w->rows == 2u * o1->n &&
        o0->w->nb >= 8u && (o0->w->nb & 7u) == 0u &&
        g->bufs[o1->qo].stride == K * o1->bstr && (o1->bstr & 3u) == 0u &&
        (o0->astr == 0u || g->bufs[o0->a].stride == K * o0->astr) &&
        buf_disjoint(g, o0->a, o1->qo)) {
      void *qw = b->d[i].w1;
      void *X = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *bout = NULL;
      void *qout = b->arena + g->bufs[o1->qo].off;
      u32 eff = o1->n, nb = o0->w->nb, q_bytes = (u32)ie_rawq_bytes(o0->w);
      u32 xs = o0->astr ? o0->astr : g->bufs[o0->a].stride;
      u32 bstr = 0u, qstr = o1->bstr, xdiv = o0->astr ? 1u : K;
      float beta = o1->clamp, lbeta = o1->lam;
      tr_span(g, i, 2u);
      if (use_adap) {
        const u32 mp1 = sep_packs ? (T * K < m->n_exp ? T * K : m->n_exp) : max_packs;
        const u32 mpM = sep_packs ? ((T * K) / 2u < max_packs ? (T * K) / 2u : max_packs) : max_packs;
        void *p1 = sep_packs ? b->ms_packs1 : b->ms_packs, *np1 = sep_packs ? b->ms_npacks1 : b->ms_npacks;
        void *pM = sep_packs ? b->ms_packsM : b->ms_packs, *npM = sep_packs ? b->ms_npacksM : b->ms_npacks;
        void *fa1[] = {&qw, &X, &bout, &qout, &eff, &nb, &b->ms_list, &p1, &np1,
                       &q_bytes, &xs, &bstr, &qstr, &xdiv, &beta, &lbeta};
        void *faM[] = {&qw, &X, &bout, &qout, &eff, &nb, &b->ms_list, &pM, &npM,
                       &q_bytes, &xs, &bstr, &qstr, &xdiv, &beta, &lbeta};
        if (mp1) launch_t(b, K_INK_GEXP_IQ3XXS_SITU_NJ1_R8, eff / 32u, mp1, fa1);
        if (mpM) launch_t(b, K_INK_GEXP_IQ3XXS_SITU_NJM_R8, eff / 32u, mpM, faM);
      } else {
        void *fa[] = {&qw, &X, &bout, &qout, &eff, &nb, &b->ms_list, &b->ms_packs, &b->ms_npacks,
                      &q_bytes, &xs, &bstr, &qstr, &xdiv, &beta, &lbeta};
        launch_t(b, K_INK_GEXP_IQ3XXS_SITU_COMPACT_R7, eff / 32u, max_packs, fa);
      }
      return 2u;
    }
  }

  /* g18 bit 37 / g09 bit 19: T==1 [OP_GEMV_EXP(IQ3_XXS egu), OP_SITU(Q8)] -> ie_inkling_gexp_iq3xxs_situ_r3/r2 */
  if (k0 == OP_GEMV_EXP && !mk.rec && T == 1u && i + 1u < i1 &&
      (b->k[K_INK_GEXP_IQ3XXS_SITU_R3] || b->k[K_INK_GEXP_IQ3XXS_SITU_R2]) &&
      o0->w->kind == MAT_RAWQ && o0->w->sub == 18u) {
    const op *o1 = &g->ops[i + 1u];
    if (o1->kind == OP_SITU && o1->a == o0->b && o1->aoff == 0u && o0->boff == 0u &&
        o1->nh == o0->nh && o1->n > 0u && (o1->n & 31u) == 0u && o0->w->rows == 2u * o1->n &&
        o0->w->nb >= 8u && (o0->w->nb & 7u) == 0u && o1->b < 0 && o1->qo >= 0 &&
        buf_disjoint(g, o0->a, o1->qo)) {
      void *qw = b->d[i].w1;
      void *xq = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *bout = NULL;
      void *qout = b->arena + g->bufs[o1->qo].off;
      void *rt = b->arena + g->bufs[o0->c].off + o0->coff;
      u32 eff = o1->n, nb = o0->w->nb, q_bytes = (u32)ie_rawq_bytes(o0->w);
      u32 astr = o0->astr, bstr = 0u, qstr = o1->bstr;
      float beta = o1->clamp, lbeta = o1->lam;
      void *args[] = {&qw, &xq, &bout, &qout, &eff, &nb, &rt, &q_bytes, &astr, &bstr, &qstr, &beta, &lbeta};
      const int ks = b->k[K_INK_GEXP_IQ3XXS_SITU_R3] ? K_INK_GEXP_IQ3XXS_SITU_R3 : K_INK_GEXP_IQ3XXS_SITU_R2;
      tr_span(g, i, 2u);
      launch_t(b, ks, (eff + 31u) / 32u, o0->nh, args);
      return 2u;
    }
  }

  /* R7/R9 Fusion D1 (bit 64 r7 T<=4 / bit 96 r9 CT=8 T=2..8): (wqkv | wq+wk+wv) + q_clip [+ k_dyn + v_dyn] */
  if (k0 == OP_GEMV_Q8 && !mk.rec && !ctx_on && T >= 1u &&
      ((b->k[K_INK_QKV_QCLIP_T8_R9] && T <= 8u) || (b->k[K_INK_QKV_QCLIP_R7] && T <= 4u)) &&
      i + 1u < i1 && b->ink_qhead_cnt && m->hd == 128u && m->n_head > 0u && m->n_kv > 0u &&
      !o0->v && o0->res < 0 && !o0->nv && o0->boff == 0u &&
      o0->w->nb >= 8u && (o0->w->nb & 7u) == 0u) {
    static int d1_r9_t1 = -1;
    if (d1_r9_t1 < 0) d1_r9_t1 = getenv("IE_INK_D1_R9_T1") ? atoi(getenv("IE_INK_D1_R9_T1")) : 1;
    const int use_d1_r9 = b->k[K_INK_QKV_QCLIP_T8_R9] && (T >= 2u || d1_r9_t1 || !b->k[K_INK_QKV_QCLIP_R7]);
    const u32 nh = m->n_head, nkv = m->n_kv, hd = 128u, qd = nh * hd, kvd = nkv * hd, nb = o0->w->nb;
    u32 n_span = 0u, i_qc = 0u;
    int packed = 0, has_dyn = 0;
    if (o0->w->rows == qd + 2u * kvd && g->ops[i + 1u].kind == OP_VCLIP) {
      const op *oqc = &g->ops[i + 1u];
      if (oqc->a == o0->b && oqc->aoff == 0u && oqc->n == qd && oqc->nh == nh && oqc->qstr == hd) {
        packed = 1; n_span = 2u; i_qc = i + 1u;
      }
    } else if (o0->w->rows == qd && i + 3u < i1 &&
               g->ops[i + 1u].kind == OP_GEMV_Q8 && g->ops[i + 2u].kind == OP_GEMV_Q8 &&
               g->ops[i + 3u].kind == OP_VCLIP) {
      const op *ok = &g->ops[i + 1u], *ov = &g->ops[i + 2u], *oqc = &g->ops[i + 3u];
      if (!ok->v && ok->res < 0 && !ok->nv && ok->boff == 0u && ok->w->rows == kvd && ok->w->nb == nb &&
          !ov->v && ov->res < 0 && !ov->nv && ov->boff == 0u && ov->w->rows == kvd && ov->w->nb == nb &&
          oqc->a == o0->b && oqc->aoff == 0u && oqc->n == qd && oqc->nh == nh && oqc->qstr == hd) {
        n_span = 4u; i_qc = i + 3u;
        if (i + 5u < i1 && g->ops[i + 4u].kind == OP_GEMV_Q8 && g->ops[i + 5u].kind == OP_GEMV_Q8) {
          const op *okd = &g->ops[i + 4u], *ovd = &g->ops[i + 5u];
          if (!okd->v && okd->res < 0 && !okd->nv && okd->boff == 0u && okd->w->nb == nb && okd->w->rows > 0u &&
              !ovd->v && ovd->res < 0 && !ovd->nv && ovd->boff == 0u && ovd->w->nb == nb && ovd->w->rows > 0u &&
              okd->a == ok->a && okd->aoff == ok->aoff && ovd->a == ov->a && ovd->aoff == ov->aoff) {
            has_dyn = 1; n_span = 6u;
          }
        }
      }
    }
    if (n_span) {
      const op *oqc = &g->ops[i_qc];
      const op *ok = packed ? o0 : &g->ops[i + 1u];
      const op *ov = packed ? o0 : &g->ops[i + 2u];
      const op *okd = has_dyn ? &g->ops[i + 4u] : NULL;
      const op *ovd = has_dyn ? &g->ops[i + 5u] : NULL;
      const int inb[3] = {o0->a, ok->a, ov->a};
      const int outb[6] = {o0->b, oqc->b, packed ? -1 : ok->b, packed ? -1 : ov->b,
                           has_dyn ? okd->b : -1, has_dyn ? ovd->b : -1};
      int ok_dj = 1;
      for (int a = 0; ok_dj && a < 6; a++) {
        if (outb[a] < 0) continue;
        for (int c = 0; ok_dj && c < 3; c++)
          if (!buf_disjoint(g, inb[c], outb[a])) ok_dj = 0;
        for (int d = a + 1; ok_dj && d < 6; d++)
          if (outb[d] >= 0 && !buf_disjoint(g, outb[a], outb[d])) ok_dj = 0;
      }
      if (ok_dj) {
        void *qw_q = b->d[i].w0, *qs_q = b->d[i].w1;
        void *xq_q = b->arena + g->bufs[o0->a].off + o0->aoff;
        void *y_q_raw = b->arena + g->bufs[o0->b].off + o0->boff;
        void *y_q_clip = b->arena + g->bufs[oqc->b].off + oqc->boff;
        void *qw_k = packed ? (void *)((uint32_t *)qw_q + (size_t)qd * nb * 8u) : b->d[i + 1u].w0;
        void *qs_k = packed ? (void *)((uint16_t *)qs_q + (size_t)qd * nb) : b->d[i + 1u].w1;
        void *xq_k = packed ? xq_q : (b->arena + g->bufs[ok->a].off + ok->aoff);
        void *y_k = packed ? (void *)((float *)y_q_raw + qd) : (b->arena + g->bufs[ok->b].off + ok->boff);
        void *qw_v = packed ? (void *)((uint32_t *)qw_q + (size_t)(qd + kvd) * nb * 8u) : b->d[i + 2u].w0;
        void *qs_v = packed ? (void *)((uint16_t *)qs_q + (size_t)(qd + kvd) * nb) : b->d[i + 2u].w1;
        void *xq_v = packed ? xq_q : (b->arena + g->bufs[ov->a].off + ov->aoff);
        void *y_v = packed ? (void *)((float *)y_q_raw + qd + kvd) : (b->arena + g->bufs[ov->b].off + ov->boff);
        void *qw_kd = has_dyn ? b->d[i + 4u].w0 : NULL, *qs_kd = has_dyn ? b->d[i + 4u].w1 : NULL;
        void *y_kd = has_dyn ? (b->arena + g->bufs[okd->b].off + okd->boff) : NULL;
        void *qw_vd = has_dyn ? b->d[i + 5u].w0 : NULL, *qs_vd = has_dyn ? b->d[i + 5u].w1 : NULL;
        void *y_vd = has_dyn ? (b->arena + g->bufs[ovd->b].off + ovd->boff) : NULL;
        void *v_qclip = b->d[i_qc].w0;
        u32 rows_kd = has_dyn ? okd->w->rows : 0u, rows_vd = has_dyn ? ovd->w->rows : 0u;
        u32 nh_u = nh, nkv_u = nkv, hd_u = hd, nb_u = nb, norope = oqc->norope, win = oqc->win, TT = T;
        float clamp = oqc->clamp, esc = oqc->escale;
        u32 xs_q = g->bufs[o0->a].stride, xs_k = g->bufs[ok->a].stride, xs_v = g->bufs[ov->a].stride;
        u32 ys_qr = g->bufs[o0->b].stride, ys_qc = g->bufs[oqc->b].stride;
        u32 ys_k = g->bufs[ok->b].stride, ys_v = g->bufs[ov->b].stride;
        u32 ys_kd = has_dyn ? g->bufs[okd->b].stride : 0u, ys_vd = has_dyn ? g->bufs[ovd->b].stride : 0u;
        if (((uintptr_t)y_q_raw & 15u) == 0u && ((uintptr_t)y_q_clip & 15u) == 0u &&
            (!v_qclip || ((uintptr_t)v_qclip & 15u) == 0u) &&
            (T == 1u || ((ys_qr | ys_qc) & 15u) == 0u)) {
          u32 gx = use_d1_r9
                       ? ((qd >> 4u) + ((kvd + 15u) >> 4u) + ((kvd + 15u) >> 4u) +
                          (rows_kd ? ((rows_kd + 15u) >> 4u) : 0u) + (rows_vd ? ((rows_vd + 15u) >> 4u) : 0u))
                       : ((qd >> 5u) + ((kvd + 31u) >> 5u) + ((kvd + 31u) >> 5u) +
                          (rows_kd ? ((rows_kd + 31u) >> 5u) : 0u) + (rows_vd ? ((rows_vd + 31u) >> 5u) : 0u));
          if (gx >= 253u && gx <= 256u) gx = 257u;
          void *fa[] = {&qw_q, &qs_q, &xq_q, &y_q_raw, &y_q_clip,
                        &qw_k, &qs_k, &xq_k, &y_k,
                        &qw_v, &qs_v, &xq_v, &y_v,
                        &qw_kd, &qs_kd, &y_kd,
                        &qw_vd, &qs_vd, &y_vd,
                        &v_qclip, &b->ink_qhead_cnt,
                        &nh_u, &nkv_u, &hd_u, &rows_kd, &rows_vd, &nb_u,
                        &clamp, &norope, &esc, &win, &pos, &TT,
                        &xs_q, &xs_k, &xs_v,
                        &ys_qr, &ys_qc, &ys_k, &ys_v, &ys_kd, &ys_vd};
          tr_span(g, i, n_span);
          launch(b, use_d1_r9 ? K_INK_QKV_QCLIP_T8_R9 : K_INK_QKV_QCLIP_R7, gx, fa);
          return n_span;
        }
      }
    }
  }

  /* R7/R9 Fusion D2 (bit 65 r7 T<=4 / bit 97 r9 CT=8 T=2..8): [wrel, r_clip(d_rel=16), wsink, wagt] */
  if (k0 == OP_GEMV_Q8 && !mk.rec && !ctx_on && T >= 1u &&
      ((b->k[K_INK_REL_CLIP_SINK_GATE_T8_R9] && T <= 8u) || (b->k[K_INK_REL_CLIP_SINK_GATE_R7] && T <= 4u)) &&
      i + 3u < i1 && m->n_head > 0u &&
      !o0->v && o0->res < 0 && !o0->nv && o0->boff == 0u &&
      o0->w->nb >= 8u && (o0->w->nb & 7u) == 0u && o0->w->rows == m->n_head * 16u) {
    static int d2_r9_t1 = -1;
    if (d2_r9_t1 < 0) d2_r9_t1 = getenv("IE_INK_D2_R9_T1") ? atoi(getenv("IE_INK_D2_R9_T1")) : 1;
    const int use_d2_r9 = b->k[K_INK_REL_CLIP_SINK_GATE_T8_R9] && (T >= 2u || d2_r9_t1 || !b->k[K_INK_REL_CLIP_SINK_GATE_R7]);
    const op *o1 = &g->ops[i + 1u], *o2 = &g->ops[i + 2u], *o3 = &g->ops[i + 3u];
    const u32 nh = m->n_head, d_rel = 16u, nb = o0->w->nb;
    if (o1->kind == OP_VCLIP && o1->a == o0->b && o1->aoff == 0u && o1->n == nh * d_rel && o1->nh == nh && o1->qstr == d_rel &&
        o2->kind == OP_GEMV_Q8 && o2->a == o0->a && o2->aoff == o0->aoff &&
        !o2->v && o2->res < 0 && !o2->nv && o2->w->nb == nb && o2->w->rows == nh &&
        o3->kind == OP_GEMV_Q8 && o3->a == o0->a && o3->aoff == o0->aoff &&
        !o3->v && o3->res < 0 && !o3->nv && o3->w->nb == nb && o3->w->rows > 0u &&
        buf_disjoint(g, o0->a, o1->b) && buf_disjoint(g, o0->a, o2->b) && buf_disjoint(g, o0->a, o3->b) &&
        buf_disjoint(g, o1->b, o2->b) && buf_disjoint(g, o1->b, o3->b) && buf_disjoint(g, o2->b, o3->b)) {
      void *qw_r = b->d[i].w0, *qs_r = b->d[i].w1, *v_rc = b->d[i + 1u].w0;
      void *y_rc = b->arena + g->bufs[o1->b].off + o1->boff;
      void *qw_s = b->d[i + 2u].w0, *qs_s = b->d[i + 2u].w1;
      void *y_s = b->arena + g->bufs[o2->b].off + o2->boff;
      void *qw_g = b->d[i + 3u].w0, *qs_g = b->d[i + 3u].w1;
      void *y_g = b->arena + g->bufs[o3->b].off + o3->boff;
      void *xq = b->arena + g->bufs[o0->a].off + o0->aoff;
      u32 nh_u = nh, dr_u = d_rel, rows_g = o3->w->rows, nb_u = nb;
      u32 norope = o1->norope, win = o1->win, TT = T;
      float clamp = o1->clamp, esc = o1->escale;
      u32 xs = g->bufs[o0->a].stride, ys_rc = g->bufs[o1->b].stride;
      u32 ys_s = g->bufs[o2->b].stride, ys_g = g->bufs[o3->b].stride;
      if (((uintptr_t)y_rc & 15u) == 0u && (!v_rc || ((uintptr_t)v_rc & 15u) == 0u) &&
          (T == 1u || (ys_rc & 15u) == 0u)) {
        u32 gx = use_d2_r9 ? (nh + ((nh + 15u) >> 4u) + ((rows_g + 15u) >> 4u))
                           : (((nh + 7u) >> 3u) + ((nh + 31u) >> 5u) + ((rows_g + 31u) >> 5u));
        if (gx >= 253u && gx <= 256u) gx = 257u;
        void *fa[] = {&qw_r, &qs_r, &v_rc, &y_rc,
                      &qw_s, &qs_s, &y_s,
                      &qw_g, &qs_g, &y_g,
                      &xq, &nh_u, &dr_u, &rows_g, &nb_u,
                      &clamp, &norope, &esc, &win, &pos, &TT,
                      &xs, &ys_rc, &ys_s, &ys_g};
        tr_span(g, i, 4u);
        launch(b, use_d2_r9 ? K_INK_REL_CLIP_SINK_GATE_T8_R9 : K_INK_REL_CLIP_SINK_GATE_R7, gx, fa);
        return 4u;
      }
    }
  }

  /* g10 bit 21: T==1 adjacent [OP_GEMV_Q8, OP_GEMV_Q8] sharing xq (wsink + wagt) -> ie_inkling_gemv_q8_multi2_r2 */
  if (k0 == OP_GEMV_Q8 && !mk.rec && T == 1u && i + 1u < i1 && b->k[K_INK_GEMV_Q8_MULTI2_R2]) {
    const op *o1 = &g->ops[i + 1u];
    if (o1->kind == OP_GEMV_Q8 && o0->a == o1->a && o0->aoff == o1->aoff &&
        o0->res < 0 && o1->res < 0 && !o0->nv && !o1->nv &&
        o0->w->nb == o1->w->nb && o0->w->nb >= 8u && (o0->w->nb & 7u) == 0u &&
        o0->w->rows > 0u && o1->w->rows > 0u) {
      void *qw0 = b->d[i].w0, *qs0 = b->d[i].w1, *bias0 = b->d[i].bias;
      void *y0 = b->arena + g->bufs[o0->b].off + o0->boff;
      u32 rows0 = o0->w->rows;
      void *qw1 = b->d[i + 1u].w0, *qs1 = b->d[i + 1u].w1, *bias1 = b->d[i + 1u].bias;
      void *y1 = b->arena + g->bufs[o1->b].off + o1->boff;
      u32 rows1 = o1->w->rows;
      void *xq = b->arena + g->bufs[o0->a].off + o0->aoff;
      u32 nb = o0->w->nb;
      void *args[] = {&qw0, &qs0, &bias0, &y0, &rows0, &qw1, &qs1, &bias1, &y1, &rows1, &xq, &nb};
      tr_span(g, i, 2u);
      launch(b, K_INK_GEMV_Q8_MULTI2_R2, ((rows0 + 31u) / 32u) + ((rows1 + 31u) / 32u), args);
      return 2u;
    }
  }

  /* R7 Fusion A bit 59: T==1 exact [OP_GEMV_F32(router), OP_MOE_ROUTE] -> ie_inkling_router_route_exact_t1_r7 */
  if (k0 == OP_GEMV_F32 && !mk.rec && T == 1u && i + 1u < i1 && b->k[K_INK_ROUTER_ROUTE_EXACT_T1_R7] &&
      !o0->v && o0->res < 0 && !o0->nv && o0->w->rows > 0u && o0->w->rows <= 256u && o0->w->cols > 0u &&
      b->gm_count) {
    const op *o1 = &g->ops[i + 1u];
    if (o1->kind == OP_MOE_ROUTE && o1->a == o0->b && o1->aoff == 0u && o0->boff == 0u &&
        o1->n == o0->w->rows && o1->nh > 0u && o1->nh <= 32u &&
        buf_disjoint(g, o0->a, o0->b) && buf_disjoint(g, o0->a, o1->b) && buf_disjoint(g, o0->b, o1->b)) {
      void *w = b->d[i].w0;
      void *x = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *lg = b->arena + g->bufs[o0->b].off + o0->boff;
      void *eb = b->d[i + 1u].w0;
      void *rt = b->arena + g->bufs[o1->b].off + o1->boff;
      u32 rows = o0->w->rows, cols = o0->w->cols, K = o1->nh;
      float sc = m->escale;
      u32 gx = (rows >= 253u && rows <= 256u) ? 257u : rows;
      void *args[] = {&w, &x, &lg, &rows, &cols, &K, &eb, &sc, &rt, &b->gm_count};
      tr_span(g, i, 2u);
      launch(b, K_INK_ROUTER_ROUTE_EXACT_T1_R7, gx, args);
      return 2u;
    }
  }

  /* R7 Fusion A bit 60: T=2..16 exact [OP_GEMV_F32(router), OP_MOE_ROUTE (+ compact sort)] -> ie_inkling_router_route_sort_ver_r7 */
  if (k0 == OP_GEMV_F32 && !mk.rec && T >= 2u && T <= 16u && i + 1u < i1 && b->k[K_INK_ROUTER_ROUTE_SORT_VER_R7] &&
      !(b->k[K_INK_MOE_SORT_ADAP_R8] && ink_moe_adap_mode() == 1) &&
      !o0->v && o0->res < 0 && !o0->nv && o0->w->rows > 0u && o0->w->rows <= 256u && o0->w->cols > 0u &&
      b->gm_count) {
    const op *o1 = &g->ops[i + 1u];
    if (o1->kind == OP_MOE_ROUTE && o1->a == o0->b && o1->aoff == 0u && o0->boff == 0u &&
        o1->n == o0->w->rows && o1->nh > 0u && o1->nh <= 32u && T * o1->nh <= 512u &&
        buf_disjoint(g, o0->a, o0->b) && buf_disjoint(g, o0->a, o1->b) && buf_disjoint(g, o0->b, o1->b)) {
      const u32 K = o1->nh, ne = m->n_exp;
      const u32 slot_max = ink_moe_slot_max(b);
      const int skip_sort = T <= slot_max && b->ink_moe_all_slot &&
                            b->k[K_INK_GEXP_RAW_IQ3XXS_SLOT_R6] && b->k[K_INK_GEXP_IQ4NL_SLOT_R6];
      const u32 mp_a = T * K, mp_b = ne + (T * K) / 4u;
      const u32 max_packs = mp_a < mp_b ? mp_a : mp_b;
      const int need_compact = !skip_sort && o0->w->rows == ne && b->k[K_INK_MOE_SORT_COMPACT_R6] &&
                               b->ms_offs && b->ms_list && b->ms_packs && b->ms_npacks &&
                               ne <= 256u && max_packs <= 160u;
      if (skip_sort || need_compact) {
        void *w = b->d[i].w0;
        void *x = b->arena + g->bufs[o0->a].off + o0->aoff;
        void *lg = b->arena + g->bufs[o0->b].off + o0->boff;
        void *eb = b->d[i + 1u].w0;
        void *rt = b->arena + g->bufs[o1->b].off + o1->boff;
        u32 rows = o0->w->rows, cols = o0->w->cols, TT = T, KK = K;
        u32 xs = g->bufs[o0->a].stride, ls = g->bufs[o0->b].stride, rst = g->bufs[o1->b].stride;
        float sc = m->escale;
        void *offs_p = need_compact ? (void *)b->ms_offs : NULL;
        void *list_p = need_compact ? (void *)b->ms_list : NULL;
        void *packs_p = need_compact ? (void *)b->ms_packs : NULL;
        void *npacks_p = need_compact ? (void *)b->ms_npacks : NULL;
        void *args[] = {&w, &x, &lg, &rows, &cols, &TT, &KK, &eb, &sc, &rt, &xs, &ls, &rst,
                        &offs_p, &list_p, &packs_p, &npacks_p, &b->gm_count};
        tr_span(g, i, 2u);
        launch(b, K_INK_ROUTER_ROUTE_SORT_VER_R7, (rows + 3u) / 4u, args);
        if (need_compact && b->k[K_INK_MOE_SORT_ADAP_R8] && ink_moe_adap_mode() == 2 &&
            b->ms_packs1 && b->ms_npacks1 && b->ms_packsM && b->ms_npacksM) {
          u32 ne_u = ne;
          void *sa[] = {&rt, &rst, &TT, &KK, &ne_u, &b->ms_offs, &b->ms_list,
                        &b->ms_packs1, &b->ms_npacks1, &b->ms_packsM, &b->ms_npacksM};
          launch(b, K_INK_MOE_SORT_ADAP_R8, 1u, sa);
        }
        return 2u;
      }
    }
  }

  /* R7 Wave 3 bit 74: T >= 17 [OP_GEMV_F32(router), OP_MOE_ROUTE] -> ie_inkling_router_route_pf_r7 (+ worklist/sort) */
  if (k0 == OP_GEMV_F32 && !mk.rec && T >= 17u && i + 1u < i1 && b->k[K_INK_ROUTER_ROUTE_PF_R7] &&
      !o0->v && o0->res < 0 && !o0->nv && o0->w->rows > 0u && o0->w->rows <= 256u &&
      o0->w->cols > 0u && (o0->w->cols & 3u) == 0u && b->at_count &&
      ((T + 63u) / 64u) <= m->n_head * b->at_tmax) {
    const op *o1 = &g->ops[i + 1u];
    if (o1->kind == OP_MOE_ROUTE && o1->a == o0->b && o1->aoff == 0u && o0->boff == 0u &&
        o1->n == o0->w->rows && o1->nh > 0u && o1->nh <= 32u &&
        buf_disjoint(g, o0->a, o0->b) && buf_disjoint(g, o0->a, o1->b) && buf_disjoint(g, o0->b, o1->b)) {
      void *w = b->d[i].w0;
      void *x = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *lg = b->arena + g->bufs[o0->b].off + o0->boff;
      void *eb = b->d[i + 1u].w0;
      void *rt = b->arena + g->bufs[o1->b].off + o1->boff;
      void *nul = NULL;
      u32 rows = o0->w->rows, cols = o0->w->cols, TT = T, K = o1->nh;
      u32 xs = g->bufs[o0->a].stride, ys = g->bufs[o0->b].stride, rs = 0u, ost = g->bufs[o1->b].stride;
      float sc = m->escale;
      if ((xs & 15u) == 0u && (((uintptr_t)w) & 15u) == 0u && (((uintptr_t)x) & 15u) == 0u) {
        void *fa[] = {&w, &x, &lg, &rows, &cols, &nul, &nul, &xs, &ys, &rs, &TT,
                      &K, &eb, &sc, &rt, &ost, &b->at_count};
        tr_span(g, i, 2u);
        launch_t(b, K_INK_ROUTER_ROUTE_PF_R7, (rows + 31u) / 32u, (T + 63u) / 64u, fa);
        u32 ne = m->n_exp;
        if (T * K > IE_MS_MAXP || ne > 1024u)
          ie_die("flint chunk: %u pairs (at most %u), %u experts (at most 1024)", T * K, IE_MS_MAXP, ne);
        const u32 mt_a = T * K, mt_b = ne + (T * K) / 64u;
        const u32 max_tiles = mt_a < mt_b ? mt_a : mt_b;
        if (b->k[K_INK_MOE_PF_WORKLIST_R7] && b->ms_packs && b->ms_npacks &&
            ne > 0u && ne <= 256u && T * K <= 8192u && max_tiles <= 512u) {
          void *sa[] = {&rt, &ost, &TT, &K, &ne, &b->ms_offs, &b->ms_list, &b->ms_packs, &b->ms_npacks};
          launch(b, K_INK_MOE_PF_WORKLIST_R7, 1, sa);
        } else {
          void *sa[] = {&rt, &ost, &TT, &K, &ne, &b->ms_offs, &b->ms_list};
          launch(b, (b->k[K_INK_MOE_SORT] && mk_ok_k(K_INK_MOE_SORT) && T * K <= 8192u && ne <= 1024u) ? K_INK_MOE_SORT : K_MOE_SORT, 1, sa);
        }
        return 2u;
      }
    }
  }

  /* g14 bit 31: T==1 [OP_GEMV_F32(router), OP_MOE_ROUTE] -> ie_inkling_router_route_dec_r2 */
  if (k0 == OP_GEMV_F32 && !mk.rec && T == 1u && i + 1u < i1 && b->k[K_INK_ROUTER_ROUTE_DEC_R2] &&
      !o0->v && o0->res < 0 && !o0->nv && o0->w->rows > 0u && o0->w->rows <= 256u &&
      o0->w->cols > 0u && (o0->w->cols & 3u) == 0u) {
    const op *o1 = &g->ops[i + 1u];
    if (o1->kind == OP_MOE_ROUTE && o1->a == o0->b && o1->aoff == 0u && o0->boff == 0u &&
        o1->n == o0->w->rows && o1->nh > 0u && o1->nh <= 32u) {
      void *w = b->d[i].w0;
      void *x = b->arena + g->bufs[o0->a].off + o0->aoff;
      void *lg = b->arena + g->bufs[o0->b].off + o0->boff;
      void *eb = b->d[i + 1u].w0;
      void *rt = b->arena + g->bufs[o1->b].off + o1->boff;
      u32 rows = o0->w->rows, cols = o0->w->cols, K = o1->nh;
      float sc = m->escale;
      if (((uintptr_t)w & 15u) == 0u && ((uintptr_t)x & 15u) == 0u && b->gm_count) {
        void *args[] = {&w, &x, &lg, &rows, &cols, &K, &eb, &sc, &rt, &b->gm_count};
        tr_span(g, i, 2u);
        launch(b, K_INK_ROUTER_ROUTE_DEC_R2, (rows + 3u) / 4u, args);
        return 2u;
      }
    }
  }

  /* R34 bit 90: Low-VGPR (<=64) WDDM-guarded sub-span persistent glue dispatcher (ie_inkling_subspan_glue_r8). */
  if (!ctx_on && !mk.rec && !b->ink_sp_poisoned && b->k[K_INK_SUBSPAN_GLUE_R8] &&
      b->ink_sp_dev && b->ink_sp_sync && T >= 1u && T <= 16u) {
    typedef struct __attribute__((aligned(16))) {
      u32 op, flags, n, a0, a1, a2, a3, f0, f1, a4;
      unsigned long long p0, p1, p2, p3, p4;
    } ink_r8_sp_task_t;
    _Static_assert(sizeof(ink_r8_sp_task_t) == 80u, "ink_r8_sp_task_t must be 80 bytes");
    static int sp_min_env = -1, sp_ng_env = -1;
    if (sp_min_env < 0) {
      sp_min_env = getenv("IE_INK_SUBSPAN_MIN") ? atoi(getenv("IE_INK_SUBSPAN_MIN")) : 2;
      if (sp_min_env < 1) sp_min_env = 1;
      sp_ng_env = getenv("IE_INK_SUBSPAN_NG") ? atoi(getenv("IE_INK_SUBSPAN_NG")) : 32;
      if (sp_ng_env < 1) sp_ng_env = 1;
      if (sp_ng_env > 64) sp_ng_env = 64; /* hard WDDM co-residency cap <= 64 */
    }
    ink_r8_sp_task_t tasks[16];
    int wbufs[32];
    u32 nt = 0u, nw = 0u;
    for (u32 j = i; j < i1 && nt < 16u; j++) {
      const op *oj = &g->ops[j];
      ink_r8_sp_task_t tk;
      memset(&tk, 0, sizeof tk);
      int rb[3] = {-1, -1, -1}, wb[2] = {-1, -1};
      int ok = 0;
      if (oj->kind == OP_COPY && oj->n > 0u && (oj->n & 3u) == 0u) {
        void *s_p = b->arena + g->bufs[oj->a].off + oj->aoff;
        void *d_p = b->arena + g->bufs[oj->b].off + oj->boff;
        const u32 ss = g->bufs[oj->a].stride, ds = g->bufs[oj->b].stride;
        if (((uintptr_t)s_p & 15u) == 0u && ((uintptr_t)d_p & 15u) == 0u && (T == 1u || ((ss | ds) & 15u) == 0u)) {
          tk.op = 1u; tk.n = oj->n; tk.a0 = T; tk.a1 = ss; tk.a2 = ds;
          tk.p0 = (unsigned long long)(uintptr_t)s_p; tk.p1 = (unsigned long long)(uintptr_t)d_p;
          rb[0] = oj->a; wb[0] = oj->b; ok = 1;
        }
      } else if (oj->kind == OP_ADD && oj->n > 0u && (oj->n & 3u) == 0u) {
        void *a_p = b->arena + g->bufs[oj->a].off + oj->aoff;
        void *b_p = b->arena + g->bufs[oj->b].off + oj->boff;
        void *c_p = b->arena + g->bufs[oj->c].off + oj->coff;
        const u32 as = g->bufs[oj->a].stride, bs = g->bufs[oj->b].stride, cs = g->bufs[oj->c].stride;
        if (((uintptr_t)a_p & 15u) == 0u && ((uintptr_t)b_p & 15u) == 0u && ((uintptr_t)c_p & 15u) == 0u &&
            (T == 1u || ((as | bs | cs) & 15u) == 0u)) {
          tk.op = 0u; tk.n = oj->n; tk.a0 = T; tk.a1 = as; tk.a2 = bs; tk.a3 = cs;
          tk.p0 = (unsigned long long)(uintptr_t)a_p; tk.p1 = (unsigned long long)(uintptr_t)b_p;
          tk.p2 = (unsigned long long)(uintptr_t)c_p;
          rb[0] = oj->a; rb[1] = oj->b; wb[0] = oj->c; ok = 1;
        }
      } else if (oj->kind == OP_RMSNORM && oj->n == 2304u && oj->v != NULL) {
        void *x_p = b->arena + g->bufs[oj->a].off + oj->aoff;
        void *w_p = b->d[j].w0;
        void *q_p = oj->qo >= 0 ? (b->arena + g->bufs[oj->qo].off) : NULL;
        void *y_p = (oj->dead_b && deadb_on() && q_p) ? NULL : (b->arena + g->bufs[oj->b].off + oj->boff);
        const u32 xs = g->bufs[oj->a].stride, ys = g->bufs[oj->b].stride, qs = oj->qo >= 0 ? g->bufs[oj->qo].stride : 0u;
        if (((uintptr_t)x_p & 15u) == 0u && ((uintptr_t)w_p & 15u) == 0u &&
            (!y_p || ((uintptr_t)y_p & 15u) == 0u) && (T == 1u || ((xs | ys | qs) & 15u) == 0u)) {
          tk.op = 3u; tk.n = 2304u; tk.a0 = T; tk.a1 = xs; tk.a2 = ys; tk.a3 = qs;
          float eps = m->eps; memcpy(&tk.f0, &eps, 4u);
          tk.p0 = (unsigned long long)(uintptr_t)x_p; tk.p1 = (unsigned long long)(uintptr_t)w_p;
          tk.p2 = (unsigned long long)(uintptr_t)y_p; tk.p3 = (unsigned long long)(uintptr_t)q_p;
          rb[0] = oj->a; wb[0] = oj->b; wb[1] = oj->qo; ok = 1;
        }
      } else if (oj->kind == OP_GATE && oj->n > 0u && (oj->n & 31u) == 0u) {
        void *a_p = b->arena + g->bufs[oj->a].off + oj->aoff;
        void *gt_p = b->arena + g->bufs[oj->b].off + oj->boff;
        void *q_p = oj->qo >= 0 ? (b->arena + g->bufs[oj->qo].off) : NULL;
        void *o_p = (oj->dead_b && deadb_on() && q_p) ? NULL : a_p;
        const u32 as = g->bufs[oj->a].stride, gs = g->bufs[oj->b].stride, qs = oj->qo >= 0 ? g->bufs[oj->qo].stride : 0u;
        if (((uintptr_t)a_p & 15u) == 0u && ((uintptr_t)gt_p & 15u) == 0u && (T == 1u || ((as | gs | qs) & 15u) == 0u)) {
          tk.op = 2u; tk.n = oj->n; tk.a0 = T; tk.a1 = as; tk.a2 = gs; tk.a3 = as; tk.a4 = qs;
          tk.p0 = (unsigned long long)(uintptr_t)a_p; tk.p1 = (unsigned long long)(uintptr_t)gt_p;
          tk.p2 = (unsigned long long)(uintptr_t)o_p; tk.p3 = (unsigned long long)(uintptr_t)q_p;
          rb[0] = oj->a; rb[1] = oj->b; wb[0] = oj->a; wb[1] = oj->qo; ok = 1;
        }
      } else if (oj->kind == OP_VCLIP && (oj->qstr == 128u || oj->qstr == 16u)) {
        void *a_p = b->arena + g->bufs[oj->a].off + oj->aoff;
        void *b_p = b->arena + g->bufs[oj->b].off + oj->boff;
        void *v_p = oj->v ? b->d[j].w0 : NULL;
        const u32 as = g->bufs[oj->a].stride, bs = g->bufs[oj->b].stride;
        if (((uintptr_t)a_p & 15u) == 0u && ((uintptr_t)b_p & 15u) == 0u &&
            (!v_p || ((uintptr_t)v_p & 15u) == 0u) && (T == 1u || ((as | bs) & 15u) == 0u)) {
          tk.op = 4u; tk.flags = oj->norope ? 2u : 0u;
          tk.n = oj->nh; tk.a0 = oj->qstr; tk.a1 = T; tk.a2 = oj->win; tk.a3 = pos; tk.a4 = as;
          float cl = oj->clamp, es = oj->escale; memcpy(&tk.f0, &cl, 4u); memcpy(&tk.f1, &es, 4u);
          tk.p0 = (unsigned long long)(uintptr_t)a_p; tk.p1 = (unsigned long long)(uintptr_t)b_p;
          tk.p2 = (unsigned long long)(uintptr_t)v_p; tk.p3 = (unsigned long long)bs;
          rb[0] = oj->a; wb[0] = oj->b; ok = 1;
        }
      } else if (oj->kind == OP_SITU && oj->b < 0 && oj->qo >= 0 && oj->n > 0u && (oj->n & 31u) == 0u &&
                 g->bufs[oj->a].stride == oj->nh * oj->astr && g->bufs[oj->qo].stride == oj->nh * oj->bstr) {
        void *a_p = b->arena + g->bufs[oj->a].off + oj->aoff;
        void *q_p = b->arena + g->bufs[oj->qo].off;
        tk.op = 7u; tk.n = oj->n; tk.a0 = T * oj->nh; tk.a1 = oj->astr; tk.a2 = oj->bstr;
        float bt = oj->clamp, lb = oj->lam; memcpy(&tk.f0, &bt, 4u); memcpy(&tk.f1, &lb, 4u);
        tk.p0 = (unsigned long long)(uintptr_t)a_p; tk.p1 = (unsigned long long)(uintptr_t)q_p;
        rb[0] = oj->a; wb[0] = oj->qo; ok = 1;
      }
      if (!ok) break;
      int hazard = 0;
      for (int r = 0; r < 3 && !hazard; r++)
        if (rb[r] >= 0)
          for (u32 w = 0u; w < nw; w++)
            if (!buf_disjoint(g, rb[r], wbufs[w])) { hazard = 1; break; }
      if (hazard && nt > 0u) {
        tasks[nt - 1u].flags |= 1u;
        nw = 0u;
      }
      tasks[nt++] = tk;
      for (int w = 0; w < 2; w++)
        if (wb[w] >= 0 && nw < 32u) wbufs[nw++] = wb[w];
    }
    if ((int)nt >= sp_min_env) {
      const u32 ridx = b->ink_sp_ring++;
      const u32 rslot = ridx & 63u;
      if (ridx >= 64u && b->ink_sp_ev[rslot]) HIP(H.EventSynchronize(b->ink_sp_ev[rslot]));
      const size_t bytes = (size_t)nt * sizeof(ink_r8_sp_task_t);
      ink_r8_sp_task_t *dev_t = (ink_r8_sp_task_t *)b->ink_sp_dev + (size_t)rslot * 16u;
      if (b->ink_sp_pin) {
        ink_r8_sp_task_t *pin_t = (ink_r8_sp_task_t *)b->ink_sp_pin + (size_t)rslot * 16u;
        memcpy(pin_t, tasks, bytes);
        HIP(H.MemcpyAsync(dev_t, pin_t, bytes, hipMemcpyHostToDevice, cur_stream));
      } else {
        HIP(H.Memcpy(dev_t, tasks, bytes, hipMemcpyHostToDevice));
      }
      u32 ng_u = (u32)sp_ng_env;
      unsigned long long bar_lim = 25000000ull, krn_lim = 90000000ull;
      void *spa[] = {&dev_t, &nt, &ng_u, &b->ink_sp_sync, &bar_lim, &krn_lim};
      tr_span(g, i, nt);
      launch(b, K_INK_SUBSPAN_GLUE_R8, ng_u, spa);
      if (b->ink_sp_ev[rslot]) HIP(H.EventRecord(b->ink_sp_ev[rslot], cur_stream));
      b->ink_sp_used = 1;
      return nt;
    }
  }

  return 0u;
}

/* Run the ops [i0, i1) for T tokens toks at positions pos .. pos + T - 1. */
static void run_ops(gpu_backend *b, uint32_t i0, uint32_t i1, const u32 *toks, u32 T, u32 pos, u32 wfrom) {
  const u32 slot = b->slot;
  b->live_snap = -1;
  b->live_hbuf = -1;
  b->q16_ready_op = -1;
  if (!i0) {
    b->rec_n = 0; /* a new model run without an accept: as accept(0) */
    if (pos == 0u) {
      b->lazy_mask = 0;
      b->lazy_rec = NULL;
    }
  }
  /* a prompt chunk (wfrom > 0 on the main model, more than one token): the model head only for
   * the last token, the MTP head not at all (its drafts are not needed) */
  const graph *g = b->g;
  const int chunk = wfrom > 0 && i0 < g->n_main && !b->base.all_logits;
  const struct ie_mtp_tree *mtr = graph_mtp_tr(g);
  const u32 kv_base = (i0 >= g->n_main && mtr->on) ? pos + mtr->slot_base : pos;
  for (u32 t = 0; t < T; t++) {
    if (toks[t] >= b->m->vocab) ie_die("token %u >= vocab %u", toks[t], b->m->vocab);
    if (kv_base + t >= g->n_ctx) ie_die("position %u >= context %u", kv_base + t, g->n_ctx);
  }
  if (T > g->T) ie_die("%u tokens in one run; the graph has room for %u", T, g->T);
  u32 kv_need = kv_base + T;
  if (b->tr_on && pos + (u32)b->tr_tab[IE_TR_KEND] > kv_need) kv_need = pos + (u32)b->tr_tab[IE_TR_KEND];
  kv_ensure(b, kv_need);
  int lazy_in_1p = 0;
  if (b->lazy_mask) {
    u32 ngdn = 0;
    for (uint32_t j = i0; j < i1; j++)
      if (g->ops[j].kind == OP_GDN) ngdn++;
    if (ngdn > 0u) {
      static int gp_env = -1;
      if (gp_env < 0) gp_env = !(getenv("IE_GDN1P") && getenv("IE_GDN1P")[0] == '0');
      const int use_1p = b->m->arch != ARCH_FLINT && (gp_env || b->tr_on) && b->has_gdn1p &&
                         T >= 2u && T <= 16u && (T <= 8u || (wfrom == 0u && T <= b->vmax)) && !mk.rec;
      if (use_1p && ngdn == b->m->n_rec) lazy_in_1p = 1;
      else gpu_flush_gdn_replay(b);
    }
  }
  for (uint32_t i = i0; i < i1; i++) {
    if (b->kv_only && (int32_t)i == g->i_mtp_wq) {
      if (b->prof == 3) HIP(H.EventRecord(b->pev[i], NULL));
      continue;
    }
    const int k = g->ops[i].kind;
    if (b->hs_on) hs_before(b, i); /* IE_HOST_STREAM: this op's layer in its slot, the last layer's slot refilled */
    tr_span(g, i, 1);
    if (chunk && i >= g->i_head && i < g->n_main) { /* last token only (its row T - 1) */
      launch_op(b, i, T - 1, toks[T - 1], pos + T - 1, 1, slot, wfrom);
      continue;
    }
    if (chunk && g->mtp_logits >= 0 && i >= g->i_mtp_head && i < g->n_mtp) continue;
    if (T > 1 && !mtr->on && g->mtp_logits >= 0 && i >= g->i_mtp_head && i < g->n_mtp) { /* MTP catch-up: only the last token's draft is used */
      launch_op(b, i, T - 1, toks[T - 1], pos + T - 1, 1, slot, wfrom);
      continue;
    }
    static int nobatch = -1;
    if (nobatch < 0) nobatch = getenv("IE_GEMV_NOBATCH") != NULL; /* debug: one GEMV launch per token */
    /* fused MTP prenorm + Q8 quantization (D3.2: 4 ops in 1 launch for T <= 16, 3 ops for T > 16) */
    if ((i == g->n_main || i == g->n_main + 1u) && !nobatch && !mk.rec && !gdump_on() && !b->base.ext_rows) {
      const u32 n = mtp_prenorm_fuse(b, i, i1, toks, T);
      if (n) {
        if (b->prof)
          for (u32 j = i; j < i + n; j++)
            if (j < g->n_main || b->prof == 3) HIP(H.EventRecord(b->pev[j], NULL));
        i += n - 1u;
        continue;
      }
    }
    /* Inkling-2 R2 batched prefill/verify Q8 embedding (T launches -> 1 launch) */
    if (k == OP_EMBED && T > 1u && !b->base.ext_rows && b->k[K_INK_EMBED_Q8_BATCH_R2] && mk_ok_k(K_INK_EMBED_Q8_BATCH_R2) && !gdump_on()) {
      const op *oe = &g->ops[i];
      const mat *tm = oe->w ? oe->w : &b->m->tok;
      if (tm->kind == MAT_Q8 && (oe->escale <= 0.0f || oe->escale == 1.0f) && tm->nb > 0u) {
        if (T > b->ink_toks_cap) {
          if (b->ink_toks_dev) H.Free(b->ink_toks_dev);
          HIP(H.Malloc((void **)&b->ink_toks_dev, (size_t)T * 4u));
          b->ink_toks_cap = T;
        }
        if (b->ink_toks_pin && T <= 512u) {
          const u32 ridx = b->ink_toks_ring++;
          const u32 rslot = ridx & 15u;
          if (ridx >= 16u && b->ink_toks_ev[rslot]) HIP(H.EventSynchronize(b->ink_toks_ev[rslot]));
          u32 *tp = b->ink_toks_pin + (size_t)rslot * 512u;
          memcpy(tp, toks, (size_t)T * 4u);
          HIP(H.MemcpyAsync(b->ink_toks_dev, tp, (size_t)T * 4u, hipMemcpyHostToDevice, cur_stream));
          if (b->ink_toks_ev[rslot]) HIP(H.EventRecord(b->ink_toks_ev[rslot], cur_stream));
        } else {
          HIP(H.Memcpy(b->ink_toks_dev, toks, (size_t)T * 4u, hipMemcpyHostToDevice));
        }
        void *t0 = oe->w ? b->d[i].w0 : b->tok0, *t1 = oe->w ? b->d[i].w1 : b->tok1;
        void *out = b->arena + g->bufs[oe->a].off + oe->aoff;
        u32 nb = tm->nb, os = g->bufs[oe->a].stride, TT = T;
        void *ea[] = {&t0, &t1, &b->ink_toks_dev, &out, &nb, &os, &TT};
        launch_t(b, K_INK_EMBED_Q8_BATCH_R2, (nb + 7u) / 8u, T, ea);
        if (b->prof && (i < g->n_main || b->prof == 3)) HIP(H.EventRecord(b->pev[i], NULL));
        continue;
      }
    }
    /* Inkling-2 R2 consecutive-op fusions (g09, g10, g11, g12, g13, g14) */
    if (b->m->arch == ARCH_INKLING2 && !nobatch && !gdump_on() && !(chunk && i >= g->i_head && i < g->n_main)) {
      const u32 nf = ink_r2_fuse(b, i, i1, toks, T, pos, slot, wfrom);
      if (nf) {
        if (b->prof)
          for (u32 j = i; j < i + nf; j++)
            if (j < g->n_main || b->prof == 3) HIP(H.EventRecord(b->pev[j], NULL));
        i += nf - 1u;
        continue;
      }
    }
    /* external vision embeddings injected directly into OP_EMBED output rows */
    if (k == OP_EMBED && b->base.ext_rows) {
      const op *o = &g->ops[i];
      const size_t stride = g->bufs[o->a].stride;
      const size_t rowb = (size_t)b->m->dim * 4u;
      for (u32 t = 0; t < T;) {
        if (!b->base.ext_rows[t]) {
          launch_op(b, i, t, toks[t], pos + t, 1, slot, wfrom);
          t++;
        } else {
          u32 t1 = t + 1u;
          while (t1 < T && b->base.ext_rows[t1] &&
                 b->base.ext_rows[t1] == b->base.ext_rows[t] + (size_t)(t1 - t) * b->m->dim && stride == rowb)
            t1++;
          void *dst = b->arena + g->bufs[o->a].off + o->aoff + (size_t)buf_row(&g->bufs[o->a], t) * stride;
          HIP(H.Memcpy(dst, b->base.ext_rows[t], (size_t)(t1 - t) * rowb, hipMemcpyHostToDevice));
          t = t1;
        }
      }
      if (gdump_on()) dump_gpu_op(b, i, 0);
      if (b->prof && (i < g->n_main || b->prof == 3)) HIP(H.EventRecord(b->pev[i], NULL));
      continue;
    }
    /* fused raw gate + up + SwiGLU + Q8 quantization across decode, verify (tr4/tr8/t16), and prefill chunks (rq_swiglu) */
    if (k == OP_GEMV_RAW && T >= 1u && !nobatch && !mk.rec &&
        (b->has_raw_swiglu || b->has_rqt16_swiglu || b->has_rq_swiglu) && raw_swiglu_on() && !gdump_on()) {
      const u32 n = raw_swiglu_fuse(b, i, i1, T);
      if (n) {
        if (b->prof)
          for (u32 j = i; j < i + n; j++)
            if (j < g->n_main || b->prof == 3) HIP(H.EventRecord(b->pev[j], NULL));
        i += n - 1u;
        continue;
      }
    }
    /* fused MAT_Q4K gate + up + SwiGLU + Q8 quantization across decode and verify (T = 1 .. 16, e.g. MTP blk.64 FFN) */
    if (k == OP_GEMV_Q4K && T >= 1u && T <= 16u && !nobatch && !mk.rec && b->has_q4k_swiglu &&
        q4k_swiglu_on() && !gdump_on()) {
      const u32 n = q4k_swiglu_fuse(b, i, i1, T);
      if (n) {
        if (b->prof)
          for (u32 j = i; j < i + n; j++)
            if (j < g->n_main || b->prof == 3) HIP(H.EventRecord(b->pev[j], NULL));
        i += n - 1u;
        continue;
      }
    }
    /* a verify of 2 to 4 tokens: raw (+ unbiased Q8) GEMVs of one input together (raw_tr4_multi) */
    if (k == OP_GEMV_RAW && T >= 2u && T <= 4u && T <= tr_tmax() && !chunk && !nobatch && !mk.rec && b->has_tr4m &&
        tr4_on() && tr4m_on() && !gdump_on()) {
      const u32 n = raw_tr4_multi(b, i, i1, T);
      if (n) {
        if (b->prof)
          for (u32 j = i; j < i + n; j++)
            if (j < g->n_main || b->prof == 3) HIP(H.EventRecord(b->pev[j], NULL));
        i += n - 1u;
        continue;
      }
    }
    /* a verify of 5 to 9 tokens (when < rqt16_min): raw (+ unbiased Q8) GEMVs of one input together (raw_tr8_multi) */
    if (k == OP_GEMV_RAW && T >= 5u && (rqt16_min() <= 0 || T < (u32)rqt16_min() || !b->has_rqt16_m) &&
        (T <= tr_tmax() || (T == TR8_TMAX && tr_tmax() == 8u)) && T <= TR8_TMAX && !chunk && !nobatch && !mk.rec &&
        b->has_tr8m && tr8_on() && tr8m_on() && !gdump_on()) {
      const u32 n = raw_tr8_multi(b, i, i1, T);
      if (n) {
        if (b->prof)
          for (u32 j = i; j < i + n; j++)
            if (j < g->n_main || b->prof == 3) HIP(H.EventRecord(b->pev[j], NULL));
        i += n - 1u;
        continue;
      }
    }
    /* a verify of 10 to 16 tokens: coalesced multi-projection raw GEMV (raw_tr16_multi) */
    if (k == OP_GEMV_RAW && T >= 10u && T <= 16u && (rqt16_min() <= 0 || T < (u32)rqt16_min() || !b->has_rqt16_m) &&
        !chunk && !nobatch && !mk.rec && b->has_tr16m && tr16_on() && tr16m_on() && !gdump_on()) {
      const u32 n = raw_tr16_multi(b, i, i1, T);
      if (n) {
        if (b->prof)
          for (u32 j = i; j < i + n; j++)
            if (j < g->n_main || b->prof == 3) HIP(H.EventRecord(b->pev[j], NULL));
        i += n - 1u;
        continue;
      }
    }
    /* a verify of rqt16_min .. 16 tokens: single-pass matrix-core multi-projection (raw_t16_multi) */
    if (k == OP_GEMV_RAW && rqt16_min() > 0 && T >= (u32)rqt16_min() && T <= 16u && !chunk && !nobatch &&
        !mk.rec && b->has_rqt16_m && rqt16m_on() && !gdump_on()) {
      const u32 n = raw_t16_multi(b, i, i1, T);
      if (n) {
        if (b->prof)
          for (u32 j = i; j < i + n; j++)
            if (j < g->n_main || b->prof == 3) HIP(H.EventRecord(b->pev[j], NULL));
        i += n - 1u;
        continue;
      }
    }
    /* one launch for all tokens: GDN (sequential inside), Q8 GEMV (weights
     * read once), and the per-token kernels with a token grid dimension */
    static int gmin = -1;
    if (gmin < 0) gmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
    static int fpf = -1; /* IE_FLINT_PF=0: flint prompt chunks one token at a time (debug) */
    if (fpf < 0) fpf = !(getenv("IE_FLINT_PF") && getenv("IE_FLINT_PF")[0] == '0');
    const int all = b->m->arch == ARCH_FLINT
                        ? fpf && (k == OP_RMSNORM || k == OP_QUANT || k == OP_SWIGLU || k == OP_COPY || k == OP_GEMV_Q8 ||
                                  k == OP_GEMV_Q4K || k == OP_GEMV_Q6K || k == OP_GEMV_F32 || k == OP_GEMV_Q5K ||
                                  k == OP_GEMV_IQ4XS || k == OP_QKN_ROPE_KV || k == OP_ATTN || k == OP_MOE_ROUTE ||
                                  k == OP_GEMV_EXP || k == OP_MOE_SWIGLU || k == OP_MOE_SUM)
                        : b->m->arch == ARCH_INKLING2
                              ? (k == OP_RMSNORM || k == OP_QUANT || k == OP_SWIGLU || k == OP_GATE || k == OP_ADD ||
                                 k == OP_COPY || k == OP_GEMV_IQ4NL || k == OP_GEMV_Q8 || k == OP_GEMV_Q4K ||
                                 k == OP_GEMV_Q6K || k == OP_GEMV_F32 || k == OP_GEMV_IQ4XS || k == OP_GEMV_RAW ||
                                 k == OP_KV || k == OP_MOE_ROUTE || k == OP_GEMV_EXP || k == OP_SITU || k == OP_MOE_SUM ||
                                 k == OP_VCLIP || k == OP_ATTN_RES || k == OP_LOOP_GATE || k == OP_SCONV ||
                                 k == OP_REL_ATTN || k == OP_LOGIT_CAP)
                              :
                    k == OP_GDN || ((k == OP_GEMV_Q8 || k == OP_GEMV_Q4K || k == OP_GEMV_Q6K || k == OP_GEMV_F32 || k == OP_GEMV_RAW || k == OP_GEMV_IQ4XS) && !nobatch) || k == OP_RMSNORM || k == OP_QUANT || k == OP_SWIGLU ||
                    k == OP_QKN_ROPE_KV || k == OP_COPY || k == OP_DF_CONV || k == OP_DF_QKV || k == OP_DF_ATTN || (k == OP_ATTN && (int)T >= gmin) /* the fast (not bitwise) prefill path */
                    || (k == OP_ATTN && b->attn_split && T <= b->at_tmax && b->at_part); /* verify: ie_attn_split_t (bitwise) */
    if (all) launch_op(b, i, 0, toks[0], pos, T, slot, wfrom);
    else
      for (u32 t = 0; t < T; t++) launch_op(b, i, t, toks[t], pos + t, 1, slot, wfrom);
    if (gdump_on()) dump_gpu_op(b, i, 0);
    if (b->prof && (i < g->n_main || b->prof == 3)) HIP(H.EventRecord(b->pev[i], NULL));
  }
  if (lazy_in_1p) {
    b->lazy_mask = 0;
    b->lazy_rec = NULL;
  }
  if (b->hs_on && b->hs_pending >= 0 && i1 > b->hs_last[b->hs_pending]) { /* the pass's last streamed layer */
    hs_after(b, (u32)b->hs_pending);
    b->hs_pending = -1;
  }
}

static u32 ink_nsplit_bucket(u32 np, u32 max_sp) {
  u32 n = (np >= 128u) ? (np + 63u) / 64u : 1u;
  if (n > max_sp) n = max_sp;
  u32 b = 1u;
  while (b < n && b < max_sp) b <<= 1u;
  if (b > max_sp) b = max_sp;
  return b ? b : 1u;
}

static uint32_t gpu_step(backend *bk, uint32_t tok, uint32_t pos, float *logits, double *ms) {
  gpu_backend *b = (gpu_backend *)bk;
  const model *m = b->m;
  const graph *g = b->g;
  static int gb = -1;
  if (gb < 0) gb = getenv("IE_GRAPH_BENCH") ? atoi(getenv("IE_GRAPH_BENCH")) : 0;
  if (gb > 0 && pos > 0) { /* debug: one step as a HIP graph, replayed gb times, against the stream */
    hipStream_t s;
    HIP(H.StreamCreateWithFlags ? H.StreamCreateWithFlags(&s, 1u) : H.StreamCreate(&s));
    cur_stream = s;
    HIP(H.DeviceSynchronize());
    HIP(H.EventRecord(b->e0, s));
    for (int r = 0; r < gb; r++) run_ops(b, 0, g->n_main, &tok, 1, pos, 0);
    HIP(H.EventRecord(b->e1, s));
    HIP(H.EventSynchronize(b->e1));
    float ts = 0, tg = 0;
    HIP(H.EventElapsedTime(&ts, b->e0, b->e1));
    void *gr, *ge;
    HIP(H.StreamBeginCapture(s, 2));
    run_ops(b, 0, g->n_main, &tok, 1, pos, 0);
    HIP(H.StreamEndCapture(s, &gr));
    HIP(H.GraphInstantiate(&ge, gr, NULL, NULL, 0));
    HIP(H.GraphLaunch(ge, s));
    HIP(H.StreamSynchronize(s));
    HIP(H.EventRecord(b->e0, s));
    for (int r = 0; r < gb; r++) HIP(H.GraphLaunch(ge, s));
    HIP(H.EventRecord(b->e1, s));
    HIP(H.EventSynchronize(b->e1));
    HIP(H.EventElapsedTime(&tg, b->e0, b->e1));
    fprintf(stderr, "graph bench: stream %.3f ms/step, graph %.3f ms/step\n", ts / gb, tg / gb);
    exit(0);
  }
  static int ink_gr_env = -1, ink_gr_pieces = -1;
  if (ink_gr_env < 0) {
    const char *e = getenv("IE_INKLING_GRAPH");
    ink_gr_env = (e && *e) ? atoi(e) : 0; /* 0: off, 1: monolithic, 2: piecewise (R35 default 4 x 12 visits) */
    const char *pw = getenv("IE_INKLING_PWGRAPH");
    if (pw && pw[0] == '1' && !ink_gr_env) ink_gr_env = 2;
    const char *pe = getenv("IE_INKLING_GRAPH_PIECES");
    ink_gr_pieces = (pe && atoi(pe) >= 1) ? atoi(pe) : (ink_gr_env >= 2 ? 4 : 1);
    if (ink_gr_pieces > 8) ink_gr_pieces = 8;
  }
  const int use_ink_graph =
      ink_gr_env && m->arch == ARCH_INKLING2 && !b->prof && !gdump_on() && !mk.on && !b->hs_on &&
      !tr_f && !tr_gpu && !b->base.ext_rows && !b->kvq8 && !b->kvq4 &&
      b->ink_ctx_dev && b->ink_ctx_host &&
      b->k[K_INK_EMBED_Q8_CTX_R2] && b->k[K_INK_VCLIP_CTX_R2] && b->k[K_INK_SCONV_CTX_R2] &&
      b->k[K_INK_KV_STORE_CTX_R2] && b->k[K_INK_REL_ATTN_DEC_CTX_R2] &&
      H.StreamBeginCapture && H.StreamEndCapture && H.GraphInstantiate && H.GraphLaunch;
  if (use_ink_graph) {
    if (tok >= m->vocab) ie_die("token %u >= vocab %u", tok, m->vocab);
    if (pos >= g->n_ctx) ie_die("position %u >= context %u", pos, g->n_ctx);
    kv_ensure(b, pos + 1u);
    if (!b->ink_gstream) {
      HIP(H.StreamCreateWithFlags ? H.StreamCreateWithFlags(&b->ink_gstream, 1u) : H.StreamCreate(&b->ink_gstream));
      for (u32 p = 0; p < 8u; p++) HIP(H.EventCreate(&b->ink_pev[p]));
    }
    const u32 swa_np = (pos + 1u > 1024u) ? 1024u : (pos + 1u);
    const u32 swa_sp = b->at_count ? ink_nsplit_bucket(swa_np, 64u) : 1u;
    const u32 glob_sp = b->at_count ? ink_nsplit_bucket(pos + 1u, 64u) : 1u;
    u32 np_cfg = (u32)ink_gr_pieces;
    if (np_cfg < 1u) np_cfg = 1u;
    if (np_cfg > 1u && g->n_visits_rec < np_cfg) np_cfg = 1u;
    const uint64_t mhash = (b->ink_mask ^ (b->ink_mask >> 32)) ^
                           ((b->ink_mask2 ^ (b->ink_mask2 >> 32)) * 0x9e3779b9ULL) ^
                           ((uint64_t)np_cfg * 0x85ebca6bULL);
    const uint64_t key = ((uint64_t)swa_sp << 48) | ((uint64_t)glob_sp << 32) | (mhash & 0xffffffffULL);
    if (!b->ink_n_pieces || !b->ink_pgraph[0] || b->ink_graph_key != key) {
      if (b->ink_n_pieces && H.GraphExecDestroy) {
        HIP(H.StreamSynchronize(b->ink_gstream));
        for (u32 p = 0; p < b->ink_n_pieces; p++)
          if (b->ink_pgraph[p]) H.GraphExecDestroy(b->ink_pgraph[p]), b->ink_pgraph[p] = NULL;
      }
      b->ink_n_pieces = 0u;
      u32 p_lo[8], p_hi[8];
      p_lo[0] = 0u;
      for (u32 p = 1u; p < np_cfg; p++) {
        const u32 v = (p * g->n_visits_rec) / np_cfg;
        p_lo[p] = g->visit_start[v];
        p_hi[p - 1u] = p_lo[p];
      }
      p_hi[np_cfg - 1u] = g->n_main;
      b->ink_swa_nsplit = swa_sp;
      b->ink_glob_nsplit = glob_sp;
      b->ink_graph_cap = 1;
      cur_stream = b->ink_gstream;
      for (u32 p = 0u; p < np_cfg; p++) {
        void *gr = NULL;
        HIP(H.StreamBeginCapture(b->ink_gstream, 2));
        run_ops(b, p_lo[p], p_hi[p], &tok, 1, pos, 0);
        HIP(H.StreamEndCapture(b->ink_gstream, &gr));
        HIP(H.GraphInstantiate(&b->ink_pgraph[p], gr, NULL, NULL, 0));
        if (H.GraphDestroy) H.GraphDestroy(gr);
      }
      b->ink_graph_cap = 0;
      cur_stream = NULL;
      b->ink_n_pieces = np_cfg;
      b->ink_graph_key = key;
      fprintf(stderr, "gpu: inkling graph captured (pieces=%u swa=%u glob=%u)\n", np_cfg, swa_sp, glob_sp);
    }
    if (ms) HIP(H.EventRecord(b->e0, b->ink_gstream));
    struct timespec h0, h1;
    clock_gettime(CLOCK_MONOTONIC, &h0);
    b->ink_ctx_host[0] = tok;
    b->ink_ctx_host[1] = pos;
    HIP(H.MemcpyAsync(b->ink_ctx_dev, b->ink_ctx_host, 8u, hipMemcpyHostToDevice, b->ink_gstream));
    for (u32 p = 0u; p < b->ink_n_pieces; p++) {
      HIP(H.GraphLaunch(b->ink_pgraph[p], b->ink_gstream));
      if (p + 1u < b->ink_n_pieces) HIP(H.EventRecord(b->ink_pev[p], b->ink_gstream));
    }
    clock_gettime(CLOCK_MONOTONIC, &h1);
    if (pos > 0) b->host_ms += (h1.tv_sec - h0.tv_sec) * 1e3 + (h1.tv_nsec - h0.tv_nsec) / 1e6, b->host_n++;
    if (ms) HIP(H.EventRecord(b->e1, b->ink_gstream));
    if (logits)
      HIP(H.MemcpyAsync(logits, b->arena + g->bufs[g->logits].off, m->vocab * 4, hipMemcpyDeviceToHost, b->ink_gstream));
    HIP(H.MemcpyAsync(&b->ink_ctx_host[2], b->arena + g->bufs[g->argmax].off, 4, hipMemcpyDeviceToHost, b->ink_gstream));
    HIP(H.StreamSynchronize(b->ink_gstream));
    if (ms) {
      float t = 0;
      HIP(H.EventElapsedTime(&t, b->e0, b->e1));
      *ms = t;
    }
    return b->ink_ctx_host[2];
  }
  static int st_on = -1; /* IE_STEP_TIME=1: a step's host time by part (launches, the wait for the GPU, the read of the token) */
  static double st_ms[4];
  static u32 st_n;
  if (st_on < 0) st_on = getenv("IE_STEP_TIME") != NULL;
  const int need_ev = (ms != NULL) || (b->prof == 1) || st_on;
  const double st0 = st_on ? ds_now_ms() : 0;
  if (need_ev) HIP(H.EventRecord(b->e0, cur_stream));
  struct timespec h0, h1;
  const uint64_t l0 = launch_n;
  clock_gettime(CLOCK_MONOTONIC, &h0);
  tr_run("decode", pos, 1);
  mk.set = b->flint ? (gq_use(b, pos) ? MKSET_FLINTL : MKSET_FLINT)
                    : m->arch == ARCH_INKLING2 ? MKSET_INKLING : MKSET_DECODE;
  mk.rec = mk.on && !mk.off[mk.set] && !b->kvq8 && !b->kvq4 && b->kvb == 2u && !b->kvrot;
  fl_cnt = 0;
  b->in_step = 1;
  run_ops(b, 0, g->n_main, &tok, 1, pos, 0); /* state: slot -> slot */
  b->in_step = 0;
  mk.rec = 0;
  mk_flush();
  clock_gettime(CLOCK_MONOTONIC, &h1);
  if (pos > 0) {
    b->host_ms += (h1.tv_sec - h0.tv_sec) * 1e3 + (h1.tv_nsec - h0.tv_nsec) / 1e6, b->host_n++;
    b->host_launches += launch_n - l0;
  }
  const double st1 = st_on ? ds_now_ms() : 0;
  if (need_ev) HIP(H.EventRecord(b->e1, cur_stream));
  u32 r = 0;
  double st2 = 0;
  if (logits) HIP(H.MemcpyAsync(logits, b->arena + g->bufs[g->logits].off, m->vocab * 4, hipMemcpyDeviceToHost, cur_stream));
  if (b->am_pin) {
    HIP(H.MemcpyAsync(b->am_pin, b->arena + g->bufs[g->argmax].off, 4, hipMemcpyDeviceToHost, cur_stream));
    HIP(H.StreamSynchronize(cur_stream));
    st2 = st_on ? ds_now_ms() : 0;
    r = b->am_pin[0];
  } else {
    if (need_ev) HIP(H.EventSynchronize(b->e1));
    st2 = st_on ? ds_now_ms() : 0;
    HIP(H.Memcpy(&r, b->arena + g->bufs[g->argmax].off, 4, hipMemcpyDeviceToHost));
  }
  float t = 0;
  if (need_ev) {
    HIP(H.EventElapsedTime(&t, b->e0, b->e1));
    if (ms) *ms = t;
    if (b->prof == 1 && pos > 0) { /* skip the first step (warm up) */
      for (uint32_t i = 0; i < g->n_main; i++) {
        float d = 0;
        HIP(H.EventElapsedTime(&d, i ? b->pev[i - 1] : b->e0, b->pev[i]));
        b->pms[i] += d;
      }
      b->psteps++;
    }
  }
  if (st_on && pos > 0) {
    st_ms[0] += st1 - st0, st_ms[1] += st2 - st1, st_ms[2] += ds_now_ms() - st2, st_ms[3] += t;
    if (++st_n % 32u == 0u)
      fprintf(stderr, "step time: launches %.3f ms, wait %.3f ms, read %.3f ms per token (GPU clock %.3f ms; %u steps)\n",
              st_ms[0] / st_n, st_ms[1] / st_n, st_ms[2] / st_n, st_ms[3] / st_n, st_n);
  }
  if (b->ink_sp_used && b->ink_sp_sync) {
    u32 pf = 0u;
    HIP(H.Memcpy(&pf, b->ink_sp_sync + 2u, 4u, hipMemcpyDeviceToHost));
    b->ink_sp_used = 0;
    if (pf != 0u) {
      b->ink_sp_poisoned = 1;
      fprintf(stderr, "gpu: ie_inkling_subspan_glue_r8 POISONED (flag=%u) -> latching off for process\n", pf);
    }
  }
  mk_check(pos);
  return r;
}

static int gpu_topk(backend *bk, int id, uint32_t r0, uint32_t n, uint32_t nv, uint32_t K, float *val, uint32_t *idx);
static int gpu_topk_lse(backend *bk, int id, uint32_t r0, uint32_t n, uint32_t nv, uint32_t K, float *val, uint32_t *idx, float *lse);
static void mtp_tree_begin(gpu_backend *b, const struct ie_mtp_tree *mtr, u32 pos, u32 T);

static double ds_now_ms(void) {
#ifdef _WIN32
  LARGE_INTEGER f, t;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&t);
  return 1e3 * (double)t.QuadPart / (double)f.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return 1e3 * (double)ts.tv_sec + 1e-6 * (double)ts.tv_nsec;
#endif
}

static void ds_gemv(gpu_backend *b, int kind, void *w0, void *w1, void *X, void *Y, u32 rows, u32 nb,
                    void *bias, void *R, u32 T, u32 xs, u32 ys, u32 rs, float eps) {
  void *nul = NULL;
  if (T == 1u) {
    void *a[] = {&w0, &w1, &X, &Y, &rows, &nb, &bias, &R, &nul, &nul, &nul, &eps, &b->gm_count};
    const int k = kind == MAT_Q4K ? K_GEMV_Q4K : kind == MAT_Q6K ? K_GEMV_Q6K : kind == MAT_Q4 ? K_GEMV_Q4 : K_GEMV_Q8;
    launch(b, k, gemv_groups(rows), a);
    return;
  }
  if (kind == MAT_Q4K || kind == MAT_Q6K) { /* the launches of OP_GEMV_Q4K / OP_GEMV_Q6K (run_ops) */
    const int q6 = kind == MAT_Q6K;
    void *a[] = {&w0, &w1, &X, &Y, &rows, &nb, &bias, &R, &T, &xs, &ys, &rs};
    if (T >= 17u && nb % 8u == 0u) {
      launch(b, q6 ? K_GEMM_Q6KR : K_GEMM_Q4KR, ((rows + 127u) / 128u) * ((T + 63u) / 64u), a);
    } else if (T <= 8u) {
      const u32 g8 = (rows + 63u) / 64u;
      launch(b, q6 ? K_GEMV_Q6K_TR8 : K_GEMV_Q4K_TR8, g8 >= 253u && g8 <= 256u ? 257u : g8, a);
    } else if (T <= 16u) {
      const u32 gr = (rows + 31u) / 32u;
      launch(b, q6 ? K_GEMV_Q6K_TR : K_GEMV_Q4K_TR, gr >= 253u && gr <= 256u ? 257u : gr, a);
    } else {
      ie_die("dspark: a %s matrix of %u columns for %u tokens has no kernel", q6 ? "Q6_K" : "Q4_K", nb * 32u, T);
    }
    return;
  }
  void *a[] = {&w0, &w1, &X, &Y, &rows, &nb, &bias, &R, &T, &xs, &ys, &rs};
  if (T >= 17u && nb % 4u == 0u) {
    const u32 gk = (b->k[K_INK_GEMM_Q8_MED_R9] && rows >= 512u) ? K_INK_GEMM_Q8_MED_R9
                   : (b->k[K_INK_GEMM_Q8_MED_R7] && rows >= 512u) ? K_INK_GEMM_Q8_MED_R7
                   : (b->k[K_INK_GEMM_Q8_MED_R2] ? K_INK_GEMM_Q8_MED_R2 : K_GEMM_Q8R);
    const u32 gr = (gk == K_INK_GEMM_Q8_MED_R9 || gk == K_INK_GEMM_Q8_MED_R7 || gk == K_INK_GEMM_Q8_MED_R2) ? 128u : 256u;
    launch(b, gk, ((rows + gr - 1u) / gr) * ((T + 63u) / 64u), a);
  } else {
    const u32 gr = (rows + 31u) / 32u;
    launch(b, K_GEMV_Q8_TR, gr >= 253u && gr <= 256u ? 257u : gr, a);
  }
}

static void ds_gemv_zsync(gpu_backend *b, int kind, void *w0, void *w1, void *X, void *Y, u32 rows, u32 nb,
                          void *bias, void *R, void *ds_ctrl) {
  void *a[] = {&w0, &w1, &X, &Y, &rows, &nb, &bias, &R, &ds_ctrl};
  const int k = kind == MAT_Q4K ? K_DS_GEMV_Q4K_ZSYNC_R5
                                : kind == MAT_Q6K ? K_DS_GEMV_Q6K_ZSYNC_R5 : K_DS_GEMV_Q8_ZSYNC_R5;
  launch(b, k, gemv_groups(rows), a);
}

/* IE_DS_SPLIT=1|q4 (default Q4_0: 29.0 MB) or IE_DS_SPLIT=q8 (Q8_0: 54.7 MB): the DSpark draft head in two parts.
 * Draft i's logits are W q8(hidden_i + bias) + P lat_i with P = W Hout ([vocab, r]): W is read once per step for all
 * drafts together, and each draft reads compact P. */
static void ds_split_setup(gpu_backend *b) {
  const dspark *d = b->m->ds;
  const u32 dim = d->dim, r = d->markov_rank, r_nb = r / 32u, nb_dim = dim / 32u, hrows = b->ds.head_rows, B = d->block;
  const char *se = getenv("IE_DS_SPLIT");
  const int use_q8 = se && (se[0] == 'q' || se[0] == 'Q') && se[1] == '8';
  if (r % 32u || (use_q8 ? !ie_q8_sizes_ok(hrows, r_nb) : !ie_sizes_ok(hrows, r_nb))) {
    fprintf(stderr, "gpu: dspark split head not used (%u rows x rank %u is outside %s layout)\n", hrows, r, use_q8 ? "Q8_0" : "Q4_0");
    return;
  }
  const double t0 = ds_now_ms();
  float eps = d->eps;
  void *nul = NULL;
  const size_t lq = q8_bytes(r_nb);
  uint8_t *oh = ie_alloc(lq);
  float *cols = ie_alloc((size_t)r * hrows * 4u);
  HIP(H.Malloc((void **)&b->ds.zero, (size_t)dim * 4u));
  HIP(H.Memset(b->ds.zero, 0, (size_t)dim * 4u));
  for (u32 j = 0; j < r; j++) { /* column j: W q8(Hout e_j) */
    const float one = 1.0f;
    const u32 s1 = 1u;
    memset(oh, 0, lq);
    oh[j] = 1u;                                          /* the int8 value */
    memcpy(oh + 32u * r_nb + 4u * (j / 32u), &one, 4u);  /* its block's scale */
    memcpy(oh + 36u * r_nb + 4u * (j / 32u), &s1, 4u);   /* and sum */
    HIP(H.Memcpy(b->ds.lat_q8, oh, lq, hipMemcpyHostToDevice));
    ds_gemv(b, d->mk_hidden_out.kind, b->ds.mk_ho_w0, b->ds.mk_ho_w1, b->ds.lat_q8, b->ds.delta, dim, r_nb, NULL, NULL, 1u, 0u, 0u, 0u, eps);
    void *ada[] = {&b->ds.delta, &b->ds.zero, &nul, &b->ds.h_q8, (void *)&dim};
    launch(b, K_DS_ADD_Q8_R4, (dim + 255u) / 256u, ada);
    ds_gemv(b, b->ds.head_kind, b->ds.head_w0, b->ds.head_w1, b->ds.h_q8, b->ds.logits, hrows, nb_dim, NULL, NULL, 1u, 0u, 0u, 0u, eps);
    HIP(H.Memcpy(cols + (size_t)j * hrows, b->ds.logits, (size_t)hrows * 4u, hipMemcpyDeviceToHost));
  }
  mat P;
  memset(&P, 0, sizeof P);
  P.rows = hrows, P.cols = r, P.nb = r_nb;
  size_t wb = 0, sb = 0;
  if (use_q8) {
    uint8_t *q8 = ie_alloc((size_t)hrows * r_nb * 34u);
    for (u32 v = 0; v < hrows; v++)
      for (u32 bl = 0; bl < r_nb; bl++) {
        float x[32], amax = 0.0f;
        for (u32 j = 0; j < 32u; j++) x[j] = cols[(size_t)(bl * 32u + j) * hrows + v], amax = fmaxf(amax, fabsf(x[j]));
        uint8_t *o = q8 + ((size_t)v * r_nb + bl) * 34u;
        const uint16_t h = ie_f32_to_f16(amax / 127.0f);
        const float dq = ie_f16_to_f32(h), id = dq != 0.0f ? 1.0f / dq : 0.0f;
        memcpy(o, &h, 2u);
        for (u32 j = 0; j < 32u; j++) {
          float q = roundf(x[j] * id);
          q = q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q;
          o[2u + j] = (uint8_t)(int8_t)q;
        }
      }
    free(cols), free(oh);
    P.kind = MAT_Q8;
    repack_q8(&P, q8);
    free(q8);
    wb = (size_t)hrows * r_nb * 32u, sb = (size_t)hrows * r_nb * 2u;
  } else {
    uint8_t *q4 = ie_alloc((size_t)hrows * r_nb * 18u);
    for (u32 v = 0; v < hrows; v++)
      for (u32 bl = 0; bl < r_nb; bl++) {
        float x[32], amax = 0.0f, mx = 0.0f;
        for (u32 j = 0; j < 32u; j++) {
          x[j] = cols[(size_t)(bl * 32u + j) * hrows + v];
          if (fabsf(x[j]) > amax) amax = fabsf(x[j]), mx = x[j];
        }
        uint8_t *o = q4 + ((size_t)v * r_nb + bl) * 18u;
        const uint16_t h = ie_f32_to_f16(mx / -8.0f);
        const float dq = ie_f16_to_f32(h), id = dq != 0.0f ? 1.0f / dq : 0.0f;
        memcpy(o, &h, 2u);
        for (u32 j = 0; j < 16u; j++) {
          int q0 = (int)floorf(x[j] * id + 8.5f), q1 = (int)floorf(x[16u + j] * id + 8.5f);
          if (q0 < 0) q0 = 0; else if (q0 > 15) q0 = 15;
          if (q1 < 0) q1 = 0; else if (q1 > 15) q1 = 15;
          o[2u + j] = (uint8_t)(q0 | (q1 << 4));
        }
      }
    free(cols), free(oh);
    P.kind = MAT_Q4;
    repack_q4(&P, q4);
    free(q4);
    wb = (size_t)hrows * r_nb * 16u, sb = (size_t)hrows * r_nb * 2u;
  }
  b->ds.p_kind = P.kind;
  HIP(H.Malloc(&b->ds.p_w0, wb));
  HIP(H.Malloc(&b->ds.p_w1, sb));
  HIP(H.Memcpy(b->ds.p_w0, P.qw, wb, hipMemcpyHostToDevice));
  HIP(H.Memcpy(b->ds.p_w1, P.qs, sb, hipMemcpyHostToDevice));
  ie_mat_free(&P);
  HIP(H.Malloc((void **)&b->ds.base, (size_t)B * hrows * 4u));
  HIP(H.Malloc((void **)&b->ds.hq, (size_t)B * b->ds.xq_str));
  fprintf(stderr, "gpu: dspark split head: P %u x %u %s (%.1f MB) in %.0f ms; a step's drafts read output.weight once\n",
          hrows, r, use_q8 ? "Q8_0" : "Q4_0", (wb + sb) / 1e6, ds_now_ms() - t0);
}

/* R30: MicroSpec dynamic prompt union + online miss-rate fallback certificate for 60K+4K DSpark shortlist. */
static void gpu_ds_set_shortlist(backend *bk, const uint32_t *prompt, uint32_t n_prompt) {
  gpu_backend *b = (gpu_backend *)bk;
  if (!b->ds.id_map_dev || !b->m->ds) return;
  const u32 unpad = b->m->ds->vocab_unpadded;
  const size_t bmb = (b->m->vocab + 7u) / 8u;
  memset(b->ds.dyn_bm, 0, bmb);
  b->ds.sl_bypass = 0;
  b->ds.sl_miss_ema = 0.0f;
  u32 nd = 0u;
  u32 cap = (b->ds.n_core < 65536u) ? (65536u - b->ds.n_core) : 0u;
  if (cap > 4096u) cap = 4096u;
  for (u32 i = 0; i < n_prompt && nd < cap; i++) {
    const u32 t = prompt[i];
    if (t >= unpad) continue;
    if ((b->ds.core_bm[t >> 3] >> (t & 7u)) & 1u) continue;
    if ((b->ds.dyn_bm[t >> 3] >> (t & 7u)) & 1u) continue;
    b->ds.dyn_bm[t >> 3] |= (uint8_t)(1u << (t & 7u));
    b->ds.id_dyn_pin[nd++] = t;
  }
  b->ds.n_dyn = nd;
  b->ds.n_act = b->ds.n_core + nd;
  if (nd > 0u)
    HIP(H.MemcpyAsync(b->ds.id_map_dev + b->ds.n_core, b->ds.id_dyn_pin, (size_t)nd * 4u,
                      hipMemcpyHostToDevice, cur_stream));
}

static void gpu_ds_note_tokens(backend *bk, const uint32_t *toks, uint32_t n_toks) {
  gpu_backend *b = (gpu_backend *)bk;
  if (!b->ds.id_map_dev || !b->m->ds) return;
  const u32 unpad = b->m->ds->vocab_unpadded;
  u32 cap = (b->ds.n_core < 65536u) ? (65536u - b->ds.n_core) : 0u;
  if (cap > 4096u) cap = 4096u;
  for (u32 i = 0; i < n_toks; i++) {
    const u32 t = toks[i];
    if (t >= unpad) continue;
    const int in_sl = (((b->ds.core_bm[t >> 3] >> (t & 7u)) | (b->ds.dyn_bm[t >> 3] >> (t & 7u))) & 1u) != 0u;
    b->ds.sl_miss_ema = 0.90f * b->ds.sl_miss_ema + (in_sl ? 0.0f : 0.10f);
    if (!in_sl && b->ds.n_dyn < cap) {
      b->ds.dyn_bm[t >> 3] |= (uint8_t)(1u << (t & 7u));
      const u32 idx = b->ds.n_dyn++;
      b->ds.id_dyn_pin[idx] = t;
      b->ds.n_act = b->ds.n_core + b->ds.n_dyn;
      HIP(H.MemcpyAsync(b->ds.id_map_dev + b->ds.n_core + idx, b->ds.id_dyn_pin + idx, 4u,
                        hipMemcpyHostToDevice, cur_stream));
    }
    if (b->ds.sl_miss_ema > 0.12f) b->ds.sl_bypass = 1;
  }
}

static void ds_shortlist_setup(gpu_backend *b, const char *hsaco) {
  if (!b->ds.ready || !b->ds.p_w1 || b->ds.p_kind != MAT_Q4 || b->ds.head_kind != MAT_Q4K) return;
  if (!b->k[K_INK_DS_BASE_Q4K_IDMAP_R7] || !b->k[K_INK_DS_GEMV_P_Q4_IDMAP_ZSYNC_R7] ||
      !b->k[K_INK_DS_CAP_SAMPLE_IDMAP_ZSYNC_R7]) return;
  const char *e = getenv("IE_DS_SHORTLIST");
  if (e && e[0] == '0' && e[1] == '\0') return;
  static const char *rel_def = "data/inkling2-shortlist/inkling2-frspec-61440.manifest.json";
  char cand[1024];
  const char *mpath = NULL;
  if (e && e[0] && strcmp(e, "1") != 0 && strcmp(e, "60k") != 0) {
    mpath = e;
  } else {
    const char *pfx[4] = {"", "../", "../../", "../../../"};
    for (int k = 0; k < 4 && !mpath; k++) {
      snprintf(cand, sizeof cand, "%s%s", pfx[k], rel_def);
      FILE *fp = fopen(cand, "rb");
      if (fp) { fclose(fp); mpath = cand; }
    }
    if (!mpath && hsaco) {
      const char *sl1 = strrchr(hsaco, '/'), *sl2 = strrchr(hsaco, '\\');
      const char *sl = sl1 && sl2 ? (sl1 > sl2 ? sl1 : sl2) : (sl1 ? sl1 : sl2);
      if (sl) {
        const size_t dlen = (size_t)(sl - hsaco) + 1u;
        if (dlen + 128u < sizeof cand) {
          memcpy(cand, hsaco, dlen);
          snprintf(cand + dlen, sizeof cand - dlen, "../../../%s", rel_def);
          FILE *fp = fopen(cand, "rb");
          if (fp) { fclose(fp); mpath = cand; }
        }
      }
    }
  }
  if (!mpath) {
    fprintf(stderr, "gpu: dspark shortlist manifest not found; keeping full vocab\n");
    return;
  }
  u32 n_core = 0u;
  char err[256] = {0};
  u32 *ids = ie_load_vocab_shortlist(mpath, b->m->vocab, &n_core, err, sizeof err);
  if (!ids) {
    fprintf(stderr, "gpu: dspark shortlist refused (%s); keeping full vocab\n", err);
    return;
  }
  const u32 unpad = b->m->ds->vocab_unpadded;
  int ok = (n_core > 0u && n_core <= 61440u);
  for (u32 i = 0; ok && i < n_core; i++) if (ids[i] >= unpad) ok = 0;
  if (!ok) {
    free(ids);
    fprintf(stderr, "gpu: dspark shortlist invalid count/id (%u); keeping full vocab\n", n_core);
    return;
  }
  const size_t bmb = (b->m->vocab + 7u) / 8u;
  b->ds.id_core_host = ids;
  b->ds.core_bm = (uint8_t *)calloc(bmb, 1u);
  b->ds.dyn_bm = (uint8_t *)calloc(bmb, 1u);
  if (!b->ds.core_bm || !b->ds.dyn_bm) ie_die("out of memory (dspark shortlist bitmap)");
  for (u32 i = 0; i < n_core; i++) b->ds.core_bm[ids[i] >> 3] |= (uint8_t)(1u << (ids[i] & 7u));
  HIP(H.Malloc((void **)&b->ds.id_map_dev, 65536u * 4u));
  HIP(H.Memcpy(b->ds.id_map_dev, ids, (size_t)n_core * 4u, hipMemcpyHostToDevice));
  if (H.HostMalloc((void **)&b->ds.id_dyn_pin, 4096u * 4u, 0) == 0 && b->ds.id_dyn_pin) {
    b->ds.id_dyn_pinned = 1;
  } else {
    b->ds.id_dyn_pin = (u32 *)ie_alloc(4096u * 4u);
    b->ds.id_dyn_pinned = 0;
  }
  b->ds.n_core = n_core;
  b->ds.n_dyn = 0u;
  b->ds.n_act = n_core;
  b->ds.sl_bypass = 0;
  b->ds.sl_miss_ema = 0.0f;
  b->base.ds_set_shortlist = gpu_ds_set_shortlist;
  b->base.ds_note_tokens = gpu_ds_note_tokens;
  fprintf(stderr, "gpu: dspark shortlist %u core + %u dynamic tail (+0.256 MiB VRAM, 0 MB weight copy)\n",
          n_core, 65536u - n_core);
}

/* R33: Multi-stream ds_ctx drain & factored KV / frnn helpers. */
static void ds_ctx_drain(gpu_backend *b) {
  if (b->ds.ctx_pending) {
    HIP(H.EventSynchronize(b->ds.ev_ctx_done));
    b->ds.ctx_pending = 0;
  }
}

static void ds_ctx_kv_launch(gpu_backend *b, u32 pos, u32 T, int dsprof, double *tp1, double *tp2, double *tp3, double *tp0) {
  const dspark *d = b->m->ds;
  const graph *g = b->g;
  const u32 dim = d->dim, kvd = d->n_kv * d->hd, nkv = d->n_kv, hd = d->hd, h2 = hd / 2u;
  float eps = d->eps;
  void *nul = NULL;
  const u32 fs = g->bufs[g->ds_feat].stride;
  float *feat = (float *)(b->arena + g->bufs[g->ds_feat].off);
  float *f0 = feat, *f1 = feat + dim, *f2 = feat + 2u * dim, *f3 = feat + 3u * dim;
  u32 TT = T, qs = b->ds.ctx_q8_str, os0 = 0u, nb_dim = dim / 32u;
  u32 kv_ys = b->ds.ctx_kv_str / 4u, rs0 = 0u;
  for (u32 il = 0; il < d->n_layer; il++) {
    u32 li = il;
    void *ma[] = {&f0, &f1, &f2, &f3, &b->ds.fuse_w, &b->ds.dflare_norm[il], &nul, &b->ds.ctx_q8,
                  &li, (void *)&dim, &eps, (void *)&fs, &os0, &qs, &TT};
    launch_t(b, K_DS_DFLARE_MIX_R4, 1u, T, ma);
    if (dsprof && tp1 && tp0) { HIP(H.DeviceSynchronize()); *tp1 += ds_now_ms() - *tp0; *tp0 = ds_now_ms(); }
    float *kb = b->ds.ctx_kv, *vb = b->ds.ctx_kv + kvd;
    ds_gemv(b, d->l[il].wk_ctx.kind, b->ds.wkc0[il], b->ds.wkc1[il], b->ds.ctx_q8, kb, kvd, nb_dim, NULL, NULL, T, qs, kv_ys, rs0, eps);
    ds_gemv(b, d->l[il].wv_ctx.kind, b->ds.wvc0[il], b->ds.wvc1[il], b->ds.ctx_q8, vb, kvd, nb_dim, NULL, NULL, T, qs, kv_ys, rs0, eps);
    if (dsprof && tp2 && tp0) { HIP(H.DeviceSynchronize()); *tp2 += ds_now_ms() - *tp0; *tp0 = ds_now_ms(); }
    const u32 cap = b->ds.kv_cap[il];
    u32 done = 0u, nq0 = 0u, nrot = hd, ins = b->ds.ctx_kv_str, qos0 = 0u, qstr = hd;
    while (done < T) {
      const u32 p_cur = pos + done, row = p_cur % cap;
      u32 cnt = T - done;
      if (row + cnt > cap) cnt = cap - row;
      void *src = (char *)b->ds.ctx_kv + (size_t)done * ins;
      uint16_t *kc = b->ds.kc[il] + (size_t)row * kvd;
      uint16_t *vc = b->ds.vc[il] + (size_t)row * kvd;
      float *cs = b->ds.rcos + (size_t)p_cur * h2, *sn = b->ds.rsin + (size_t)p_cur * h2;
      void *ra[] = {&src, &nul, &kc, &vc, &nul, &b->ds.k_ctx_norm[il], &cs, &sn, &nq0, (void *)&nkv, (void *)&hd, &nrot, &eps, &ins, &qos0, &qstr};
      launch_t(b, K_QKN_ROPE_KV, 2u * nkv, cnt, ra);
      done += cnt;
    }
    if (dsprof && tp3 && tp0) { HIP(H.DeviceSynchronize()); *tp3 += ds_now_ms() - *tp0; *tp0 = ds_now_ms(); }
  }
}

static void ds_ctx_frnn_launch(gpu_backend *b, u32 row_idx) {
  const dspark *d = b->m->ds;
  const graph *g = b->g;
  const u32 dim = d->dim, fs = g->bufs[g->ds_feat].stride;
  float eps = d->eps;
  void *nul = NULL;
  float *feat = (float *)(b->arena + g->bufs[g->ds_feat].off);
  float *fl = (float *)((char *)feat + (size_t)row_idx * fs);
  float *l0 = fl, *l1 = fl + dim, *l2 = fl + 2u * dim, *l3 = fl + 3u * dim;
  u32 li3 = 3u, one = 1u, zero = 0u;
  void *fa[] = {&l0, &l1, &l2, &l3, &b->ds.fuse_w, &b->ds.mk_dflare_norm, &b->ds.frnn, &nul,
                &li3, (void *)&dim, &eps, (void *)&fs, &zero, &zero, &one};
  launch_t(b, K_DS_DFLARE_MIX_R4, 1u, 1u, fa);
}

static void gpu_run(backend *bk, int sec, const uint32_t *toks, uint32_t T, uint32_t pos, int last_only, uint32_t *out,
                    float *prob) {
  gpu_backend *b = (gpu_backend *)bk;
  const graph *g = b->g;
  { /* IE_VRAM_TRACE=1: before the first runs, then every 64th */
    static uint32_t nr;
    if (nr < 6u || nr % 64u == 0u) {
      char w[16];
      snprintf(w, sizeof w, "run%u", nr);
      vram_trace(w, 0);
    }
    nr++;
  }
  if (sec == 1 && g->mtp_argmax < 0) ie_die("the graph has no MTP ops");
  if (sec == 4) { /* the MTP layer only (its KV cache), not its output head */
    tr_run("mtp_kv", pos, T);
    const int fast = g->mtp_a < 0 && g->i_mtp_kv > g->n_main && mtp_kv_fast_on();
    b->kv_only = fast;
    run_ops(b, g->n_main, fast ? g->i_mtp_kv : g->i_mtp_head, toks, T, pos, 0);
    b->kv_only = 0;
    (void)last_only, (void)out, (void)prob;
    return;
  }
  if (sec >= 2) { /* DFlash / DSpark: 2 the context ops, 3 the block / draft ops */
    if (g->ds_feat >= 0 && b->ds.ready) {
      const dspark *d = b->m->ds;
      const u32 dim = d->dim, qd = d->n_head * d->hd, kvd = d->n_kv * d->hd, r = d->markov_rank;
      const u32 nh = d->n_head, nkv = d->n_kv, hd = d->hd, h2 = hd / 2u;
      float eps = d->eps;
      void *nul = NULL;
      if (sec == 2) {
        tr_run("ds_ctx", pos, T);
        b->live_snap = -1;
        if (b->ds.ctx_pending && pos == b->ds.ctx_pos && T <= b->ds.ctx_T) {
          /* R33: KV rows [pos .. pos + T - 1] were already computed concurrently on ctx_stream; GPU-only wait + 1-WG frnn! */
          HIP(H.StreamWaitEvent(cur_stream, b->ds.ev_ctx_done, 0));
          b->ds.ctx_pending = 0;
          ds_ctx_frnn_launch(b, T - 1u);
          (void)last_only;
          return;
        }
        if (b->ds.ctx_pending) {
          HIP(H.StreamWaitEvent(cur_stream, b->ds.ev_ctx_done, 0));
          b->ds.ctx_pending = 0;
        }
        static int dsprof_ctx = -1;
        if (dsprof_ctx < 0) dsprof_ctx = getenv("IE_DS_PROF") && getenv("IE_DS_PROF")[0] == '1';
        const int dsprof = dsprof_ctx;
        double tp0 = 0, tp1 = 0, tp2 = 0, tp3 = 0;
        if (dsprof) { HIP(H.DeviceSynchronize()); tp0 = ds_now_ms(); }
        for (u32 il = 0; il < d->n_layer; il++)
          if (b->ds.kv_vm[il]) { ds_kv_ensure(b, pos + T); break; }
        ds_ctx_kv_launch(b, pos, T, dsprof, &tp1, &tp2, &tp3, &tp0);
        ds_ctx_frnn_launch(b, T - 1u);
        if (dsprof && T >= 16u) {
          HIP(H.DeviceSynchronize());
          fprintf(stderr, "ds_ctx(pos=%u, T=%u): dflare=%.2f ms, gemm=%.2f ms, rope_kv=%.2f ms, frnn=%.2f ms\n",
                  pos, T, tp1, tp2, tp3, ds_now_ms() - tp0);
        }
      } else {
        /* sec == 3: DSpark block forward (B = d->block) + sequential Markov RNN (nd = T steps) */
        tr_run("ds_draft", pos, T);
        static int dsprof_dr = -1;
        if (dsprof_dr < 0) dsprof_dr = getenv("IE_DS_PROF") && getenv("IE_DS_PROF")[0] == '1';
        const int dsprof = dsprof_dr;
        double td0 = 0, td_init = 0, td_blk = 0, td_rnn = 0, td_head = 0;
        if (dsprof) { HIP(H.DeviceSynchronize()); td0 = ds_now_ms(); }
        const u32 B = d->block, nd = T;
        ds_kv_ensure(b, pos); /* the block attention reads context rows 0 .. pos - 1 */
        u32 nb_dim = dim / 32u, nb_qd = qd / 32u, nb_ffn = d->ffn / 32u, anchor = toks[0];
        float meps = b->m->eps;
        static int zsync_env = -1, split_all_env = -1;
        if (zsync_env < 0) zsync_env = !(getenv("IE_DS_ZSYNC") && getenv("IE_DS_ZSYNC")[0] == '0');
        if (split_all_env < 0) split_all_env = getenv("IE_DS_SPLIT_ALL") && getenv("IE_DS_SPLIT_ALL")[0] == '1';
        const int split = b->ds.p_w1 != NULL;
        const int use_split_now = split && (nd >= 4u || split_all_env);
        const int ds_samp_on = b->base.ds_samp && b->base.ds_samp->on && b->base.ds_samp->temp > 0.0f;
        const int use_sl = use_split_now && b->ds.id_map_dev && !b->ds.sl_bypass &&
                           b->ds.n_act > 0u && b->ds.n_act <= 65536u &&
                           b->ds.head_kind == MAT_Q4K && b->ds.p_kind == MAT_Q4 && nd <= 8u &&
                           b->k[K_INK_DS_BASE_Q4K_IDMAP_R7] && b->k[K_INK_DS_GEMV_P_Q4_IDMAP_ZSYNC_R7] &&
                           b->k[K_INK_DS_CAP_SAMPLE_IDMAP_ZSYNC_R7];
        const int fused_ds_p = !use_sl && use_split_now && b->ds.p_kind == MAT_Q4 && b->k[K_DS_GEMV_P_CAP_SAMPLE_ZSYNC_R8];
        const int zsync_base = zsync_env && !dsprof && !b->m->ds_head.rows && nd <= 16u && b->ds.zbuf &&
                               b->k[K_DS_BLOCK_INIT_ZSYNC_R5] && b->k[K_DS_CONCAT_Q8_ZSYNC_R5] &&
                               b->k[K_DS_RNN_STEP_ZSYNC_R5] && b->k[K_DS_ADD_Q8_ZSYNC_R5] &&
                               b->k[K_DS_GEMV_Q8_ZSYNC_R5] && b->k[K_DS_GEMV_Q4K_ZSYNC_R5] && b->k[K_DS_GEMV_Q6K_ZSYNC_R5] &&
                               (!use_split_now || b->ds.p_kind != MAT_Q4 || b->k[K_DS_GEMV_P_Q4_ZSYNC_R5] || fused_ds_p || use_sl);
        const int zsync_samp = ds_samp_on && zsync_base && b->ds.zpin && (b->k[K_DS_CAP_SAMPLE_ZSYNC_R7] || fused_ds_p || use_sl);
        const int zsync = zsync_samp || (zsync_base && (b->k[K_DS_CAP_ARGMAX_ZSYNC_R5] || fused_ds_p || use_sl));
        char *zb = (char *)b->ds.zbuf;
        void *ds_ctrl = zb;
        if (zsync) {
          u32 rr = r;
          void *ba[] = {&b->tok0, &b->tok1, &b->ds.tok_norm, &b->ds.mask_embed, &b->ds.h, &b->ds.state, &ds_ctrl,
                        &anchor, (void *)&dim, &nb_dim, &meps, (void *)&B, &rr};
          launch_t(b, K_DS_BLOCK_INIT_ZSYNC_R5, 1u, B, ba);
        } else {
          void *ba[] = {&b->tok0, &b->tok1, &b->ds.tok_norm, &b->ds.mask_embed, &b->ds.h, &anchor, (void *)&dim, &nb_dim, &meps, (void *)&B};
          launch_t(b, K_DS_BLOCK_INIT_R4, 1u, B, ba);
        }
        if (dsprof) { HIP(H.DeviceSynchronize()); td_init = ds_now_ms() - td0; td0 = ds_now_ms(); }
        u32 BB = B, hs = b->ds.h_str, xqs = b->ds.xq_str, qs = b->ds.q_str, kvs = b->ds.kv_str;
        u32 aqs = b->ds.aq_str, gus = b->ds.gu_str, gqs = b->ds.gq_str;
        u32 h_ys = hs / 4u, q_ys = qs / 4u, kv_ys = kvs / 4u, gu_ys = gus / 4u, rs0 = 0u;
        for (u32 il = 0; il < d->n_layer; il++) {
          /* pre-attn RMSNorm + Q8 */
          void *na[] = {&b->ds.h, &b->ds.attn_norm[il], &b->ds.xn, (void *)&dim, &eps, &b->ds.xq, &hs, &hs, &xqs};
          launch_t(b, b->k[K_INK_RMSNORM_MT] ? K_INK_RMSNORM_MT : K_RMSNORM_T, 1u, B, na);
          /* Q, K, V projections */
          ds_gemv(b, d->l[il].wq.kind, b->ds.wq0[il], b->ds.wq1[il], b->ds.xq, b->ds.q, qd, nb_dim, NULL, NULL, BB, xqs, q_ys, rs0, eps);
          ds_gemv(b, d->l[il].wk.kind, b->ds.wk0[il], b->ds.wk1[il], b->ds.xq, b->ds.k, kvd, nb_dim, NULL, NULL, BB, xqs, kv_ys, rs0, eps);
          ds_gemv(b, d->l[il].wv.kind, b->ds.wv0[il], b->ds.wv1[il], b->ds.xq, b->ds.v, kvd, nb_dim, NULL, NULL, BB, xqs, kv_ys, rs0, eps);
          /* per-head Q/K RMSNorm + NeoX RoPE in-place (pos .. pos + B - 1) */
          float *cs = b->ds.rcos + (size_t)pos * h2, *sn = b->ds.rsin + (size_t)pos * h2;
          u32 nkv0 = 0u, nrot = hd, qstr = hd;
          void *rqa[] = {&b->ds.q, &b->ds.q, &nul, &nul, &b->ds.q_norm[il], &nul, &cs, &sn, (void *)&nh, &nkv0, (void *)&hd, &nrot, &eps, &qs, &qs, &qstr};
          void *rka[] = {&b->ds.k, &b->ds.k, &nul, &nul, &b->ds.k_norm[il], &nul, &cs, &sn, (void *)&nkv, &nkv0, (void *)&hd, &nrot, &eps, &kvs, &kvs, &qstr};
          launch_t(b, K_QKN_ROPE_KV, nh, B, rqa);
          launch_t(b, K_QKN_ROPE_KV, nkv, B, rka);
          /* bidirectional block attention over context KV [0..pos-1] + block KV [0..B-1] */
          u32 ctx_len = pos, cap = b->ds.kv_cap[il], win = d->l[il].win;
          if (b->k[K_DS_ATTN_LONG_R7] && hd == 128u && nkv > 0u && nh == nkv * 3u && B > 0u && ((qs | kvs) & 15u) == 0u) {
            const u32 ctx_start = (win > 0u && ctx_len > win) ? (ctx_len - win) : 0u;
            const u32 ctx_n = (cap > 0u && ctx_len > ctx_start) ? (ctx_len - ctx_start) : 0u;
            u32 nsp = (ctx_n >= 256u && b->at_part && b->at_count) ? (ctx_n + 255u) / 256u : 1u;
            if (nsp > 32u) nsp = 32u;
            const u32 btiles = (B + 7u) >> 3u;
            if (!b->at_count || btiles * nkv > b->m->n_head * b->at_tmax) nsp = 1u;
            while (nsp > 1u && (size_t)B * nh * nsp * 132u > b->at_part_floats) nsp >>= 1u;
            void *part_p = nsp > 1u ? (void *)b->at_part : NULL;
            void *cnt_p = nsp > 1u ? (void *)b->at_count : NULL;
            void *aa7[] = {&b->ds.q, &b->ds.kc[il], &b->ds.vc[il], &b->ds.k, &b->ds.v, &b->ds.att,
                           &ctx_len, &cap, &win, &BB, (void *)&nh, (void *)&nkv, (void *)&hd,
                           &qs, &kvs, &qs, &nsp, &part_p, &cnt_p};
            launch_t(b, K_DS_ATTN_LONG_R7, nkv * nsp, btiles, aa7);
          } else {
            void *aa[] = {&b->ds.q, &b->ds.kc[il], &b->ds.vc[il], &b->ds.k, &b->ds.v, &b->ds.att,
                          &ctx_len, &cap, &win, &BB, (void *)&nh, (void *)&nkv, (void *)&hd, &qs, &kvs, &qs};
            launch_t(b, K_DS_ATTN_R4, nh, B, aa);
          }
          /* quantize att -> aq, o_proj + residual h */
          void *qaa[] = {&b->ds.att, &b->ds.aq, &nb_qd, &qs, &aqs};
          launch_t(b, K_QUANT, (nb_qd + 7u) / 8u, B, qaa);
          ds_gemv(b, d->l[il].wo.kind, b->ds.wo0[il], b->ds.wo1[il], b->ds.aq, b->ds.h, dim, nb_qd, NULL, b->ds.h, BB, aqs, h_ys, h_ys, eps);
          /* pre-FFN RMSNorm + Q8 */
          void *fna[] = {&b->ds.h, &b->ds.ffn_norm[il], &b->ds.xn, (void *)&dim, &eps, &b->ds.xq, &hs, &hs, &xqs};
          launch_t(b, b->k[K_INK_RMSNORM_MT] ? K_INK_RMSNORM_MT : K_RMSNORM_T, 1u, B, fna);
          /* gate+up -> SwiGLU + Q8 -> down + residual h (fused Q4_K SwiGLU when available) */
          u32 gu_rows = 2u * d->ffn, ffn_n = d->ffn;
          if (d->l[il].wgu.kind == MAT_Q4K && b->has_q4k_swiglu && q4k_swiglu_on() && BB <= 16u && (ffn_n % 32u) == 0u) {
            void *qw0 = b->ds.wgu0[il], *qs0 = b->ds.wgu1[il], *Y = NULL;
            u32 rows0 = gu_rows, roff1 = ffn_n;
            void *sa[] = {&qw0, &qs0, &rows0, &qw0, &qs0, &rows0, &roff1, &b->ds.xq, &Y, &b->ds.gq,
                          &ffn_n, &nb_dim, &BB, &xqs, &gus, &gqs};
            u32 gr = ffn_n / 32u;
            if (gr >= 253u && gr <= 256u) gr = 257u;
            launch(b, K_GEMV_Q4K_SWIGLU, gr, sa);
          } else {
            ds_gemv(b, d->l[il].wgu.kind, b->ds.wgu0[il], b->ds.wgu1[il], b->ds.xq, b->ds.gu, gu_rows, nb_dim, NULL, NULL, BB, xqs, gu_ys, rs0, eps);
            void *gu_up = (char *)b->ds.gu + (size_t)ffn_n * 4u;
            float cl0 = 0.0f;
            void *swa[] = {&b->ds.gu, &gu_up, &ffn_n, &b->ds.gq, &gus, &gqs, &cl0};
            launch_t(b, K_SWIGLU, (ffn_n + 255u) / 256u, B, swa);
          }
          ds_gemv(b, d->l[il].wdown.kind, b->ds.wd0[il], b->ds.wd1[il], b->ds.gq, b->ds.h, dim, nb_ffn, NULL, b->ds.h, BB, gqs, h_ys, h_ys, eps);
        }
        /* final output_norm -> b->ds.hidden [B, dim] */
        void *ona[] = {&b->ds.h, &b->ds.out_norm, &b->ds.hidden, (void *)&dim, &eps, &nul, &hs, &hs, &rs0};
        launch_t(b, b->k[K_INK_RMSNORM_MT] ? K_INK_RMSNORM_MT : K_RMSNORM_T, 1u, B, ona);
        if (dsprof) { HIP(H.DeviceSynchronize()); td_blk = ds_now_ms() - td0; td0 = ds_now_ms(); }
        const float pmin = prob ? prob[0] : 0.0f;
        const u32 rnn_rows = 3u * r, rnn_nb = (2u * r + 2u * dim) / 32u, r_nb = r / 32u;
        const u32 hrows = b->ds.head_rows;
        const u32 unpad = b->m->ds_head.rows ? hrows : d->vocab_unpadded;
        float cap = d->logit_cap;
        u32 ng_am = ARGMAX_GROUPS, zero = 0u, n_ran = 0u;
        if (zsync) {
          /* R5/R7 zero-sync device-side Markov loop: 0 intermediate CPU syncs, 1 single pinned D2H copy at end. */
          typedef struct {
            u32 ctrl[4];             /* off    0 */
            u32 draft_ids[16];       /* off   16 */
            float draft_probs[16];   /* off   80 */
            uint64_t rng_state;      /* off  144 */
            float draft_entropy[16]; /* off  152 */
            u32 q_n[16];             /* off  216 */
            u32 pad[2];              /* off  280 -> 288 is 16-B aligned */
            u32 q_ids[16][32];       /* off  288 */
            float q_probs[16][32];   /* off 2336 */
          } ds_zsamp_hdr;
          _Static_assert(sizeof(ds_zsamp_hdr) <= 8192u, "ds_zsamp_hdr fits in 8 KiB zpin");
          void *draft_ids = zb + 16u, *draft_probs = zb + 80u;
          void *part_max = zb + 256u, *part_id = zb + 768u, *part_sum = zb + 1280u;
          if (zsync_samp) {
            *(uint64_t *)b->ds.zpin = b->base.ds_samp->rng;
            HIP(H.MemcpyAsync(zb + offsetof(ds_zsamp_hdr, rng_state), b->ds.zpin, 8u, hipMemcpyHostToDevice, cur_stream));
          }
          if (use_split_now) {
            /* Single-pass base = output.weight * q8(hidden_j + bias) for all nd drafts (no 2-pass mispredicts). */
            for (u32 j = 0; j < nd; j++) {
              float *hj = (float *)((char *)b->ds.hidden + (size_t)(j + 1u) * hs);
              uint8_t *qo = b->ds.hq + (size_t)j * xqs;
              void *aba[] = {&hj, &b->ds.zero, &b->ds.mk_hidden_bias, &qo, (void *)&dim};
              launch(b, K_DS_ADD_Q8_R4, (dim + 255u) / 256u, aba);
            }
            if (use_sl) {
              u32 n_act = b->ds.n_act, nd_u = nd, ys_f32 = hrows;
              void *ba7[] = {&b->ds.head_w0, &b->ds.head_w1, &b->ds.hq, &b->ds.id_map_dev, &b->ds.base,
                             (void *)&hrows, &n_act, &nb_dim, &nd_u, &xqs, &ys_f32};
              launch(b, K_INK_DS_BASE_Q4K_IDMAP_R7, (n_act + 31u) / 32u, ba7);
            } else {
              ds_gemv(b, b->ds.head_kind, b->ds.head_w0, b->ds.head_w1, b->ds.hq, b->ds.base, hrows, nb_dim,
                      NULL, NULL, nd, xqs, hrows, 0u, eps);
            }
          }
          for (u32 i = 0; i < nd; i++) {
            float *hi = (float *)((char *)b->ds.hidden + (size_t)(i + 1u) * hs);
            void *ca[] = {&b->ds.state, &b->ds.mk_w1_w0, &b->ds.mk_w1_w1, &hi, &b->ds.frnn, &b->ds.cat_q8,
                          &ds_ctrl, (void *)&r, (void *)&dim};
            launch(b, K_DS_CONCAT_Q8_ZSYNC_R5, (2u * r + 2u * dim + 255u) / 256u, ca);
            ds_gemv_zsync(b, d->mk_rnn_proj.kind, b->ds.mk_rnn_w0, b->ds.mk_rnn_w1, b->ds.cat_q8, b->ds.raw768,
                          rnn_rows, rnn_nb, b->ds.mk_rnn_bias, NULL, ds_ctrl);
            void *sa[] = {&b->ds.raw768, &b->ds.state, &nul, &b->ds.lat_q8, &ds_ctrl, (void *)&r};
            launch(b, K_DS_RNN_STEP_ZSYNC_R5, 1u, sa);
            u32 sidx = i;
            if (use_sl) {
              float *brow = b->ds.base + (size_t)i * hrows;
              u32 n_act = b->ds.n_act;
              void *pa7[] = {&b->ds.p_w0, &b->ds.p_w1, &b->ds.lat_q8, &brow, &b->ds.id_map_dev, &b->ds.logits,
                             (void *)&hrows, &n_act, (void *)&r_nb, &ds_ctrl};
              launch(b, K_INK_DS_GEMV_P_Q4_IDMAP_ZSYNC_R7, (n_act + 31u) / 32u, pa7);
              ie_ds_samp *ds = zsync_samp ? b->base.ds_samp : NULL;
              static float ds_spmin_sl = -1.0f, ds_sa_sl = -1.0f, ds_sb_sl = -1.0f;
              if (ds_spmin_sl < 0.0f) {
                ds_spmin_sl = getenv("IE_DS_SAMP_PMIN") ? (float)atof(getenv("IE_DS_SAMP_PMIN")) : 0.45f;
                ds_sa_sl = getenv("IE_DS_SURV_A") ? (float)atof(getenv("IE_DS_SURV_A")) : 0.92f;
                ds_sb_sl = getenv("IE_DS_SURV_B") ? (float)atof(getenv("IE_DS_SURV_B")) : 0.30f;
              }
              u32 ng_s = 64u, K_d = ds ? (ds->top_k ? (ds->top_k <= 32u ? ds->top_k : 32u) : 20u) : 1u, K_str = 32u;
              float temp_v = ds ? ds->temp : 0.0f, topp_v = ds ? ds->top_p : 1.0f;
              float pmin_v = ds ? (ds->fs_on ? ds->surv_pmin : ds_spmin_sl) : pmin;
              float surv_a = ds ? (ds->fs_on ? ds->surv_a : ds_sa_sl) : 0.0f;
              float surv_b = ds ? (ds->fs_on ? ds->surv_b : ds_sb_sl) : 0.0f;
              void *pv = zb + 8192u, *pi = zb + 16384u, *ps = zb + 24576u;
              void *rs = ds ? (zb + offsetof(ds_zsamp_hdr, rng_state)) : NULL;
              void *de = ds ? (zb + offsetof(ds_zsamp_hdr, draft_entropy)) : NULL;
              void *qn = ds ? (zb + offsetof(ds_zsamp_hdr, q_n)) : NULL;
              void *qi = ds ? (zb + offsetof(ds_zsamp_hdr, q_ids)) : NULL;
              void *qp = ds ? (zb + offsetof(ds_zsamp_hdr, q_probs)) : NULL;
              void *ama7[] = {&b->ds.logits, &b->ds.id_map_dev, &pv, &pi, &ps, &b->am_count, &ds_ctrl, &rs,
                              &draft_ids, &draft_probs, &de, &qn, &qi, &qp,
                              &n_act, &ng_s, &K_d, &K_str, &sidx,
                              &cap, &temp_v, &topp_v, &pmin_v, &surv_a, &surv_b};
              launch(b, K_INK_DS_CAP_SAMPLE_IDMAP_ZSYNC_R7, ng_s, ama7);
            } else if (fused_ds_p) {
              float *brow = b->ds.base + (size_t)i * hrows;
              void *id_map_nul = NULL;
              static float ds_spmin8 = -1.0f, ds_sa8 = -1.0f, ds_sb8 = -1.0f;
              if (ds_spmin8 < 0.0f) {
                ds_spmin8 = getenv("IE_DS_SAMP_PMIN") ? (float)atof(getenv("IE_DS_SAMP_PMIN")) : 0.45f;
                ds_sa8 = getenv("IE_DS_SURV_A") ? (float)atof(getenv("IE_DS_SURV_A")) : 0.92f;
                ds_sb8 = getenv("IE_DS_SURV_B") ? (float)atof(getenv("IE_DS_SURV_B")) : 0.30f;
              }
              ie_ds_samp *ds = zsync_samp ? b->base.ds_samp : NULL;
              u32 ng_s = 64u, K_d = ds ? (ds->top_k ? (ds->top_k <= 32u ? ds->top_k : 32u) : 20u) : 1u, K_str = 32u;
              float temp_v = ds ? ds->temp : 0.0f, topp_v = ds ? ds->top_p : 1.0f;
              float pmin_v = ds ? (ds->fs_on ? ds->surv_pmin : ds_spmin8) : pmin;
              float surv_a = ds ? (ds->fs_on ? ds->surv_a : ds_sa8) : 0.0f;
              float surv_b = ds ? (ds->fs_on ? ds->surv_b : ds_sb8) : 0.0f;
              void *pv = zb + 8192u, *pi = zb + 16384u, *ps = zb + 24576u;
              void *rs = ds ? (zb + offsetof(ds_zsamp_hdr, rng_state)) : NULL;
              void *de = ds ? (zb + offsetof(ds_zsamp_hdr, draft_entropy)) : NULL;
              void *qn = ds ? (zb + offsetof(ds_zsamp_hdr, q_n)) : NULL;
              void *qi = ds ? (zb + offsetof(ds_zsamp_hdr, q_ids)) : NULL;
              void *qp = ds ? (zb + offsetof(ds_zsamp_hdr, q_probs)) : NULL;
              void *fa[] = {&b->ds.p_w0, &b->ds.p_w1, &b->ds.lat_q8, &brow, &id_map_nul,
                            &pv, &pi, &ps, &b->am_count, &ds_ctrl, &rs,
                            &draft_ids, &draft_probs, &de, &qn, &qi, &qp,
                            (void *)&hrows, (void *)&unpad, (void *)&r_nb, &ng_s, &K_d, &K_str, &sidx,
                            &cap, &temp_v, &topp_v, &pmin_v, &surv_a, &surv_b};
              launch(b, K_DS_GEMV_P_CAP_SAMPLE_ZSYNC_R8, ng_s, fa);
            } else {
            if (use_split_now) {
              float *brow = b->ds.base + (size_t)i * hrows;
              if (b->ds.p_kind == MAT_Q4) {
                void *pa[] = {&b->ds.p_w0, &b->ds.p_w1, &b->ds.lat_q8, &brow, &b->ds.logits,
                              (void *)&hrows, (void *)&r_nb, &ds_ctrl};
                launch(b, K_DS_GEMV_P_Q4_ZSYNC_R5, (hrows + 31u) / 32u, pa);
              } else {
                ds_gemv_zsync(b, MAT_Q8, b->ds.p_w0, b->ds.p_w1, b->ds.lat_q8, b->ds.logits,
                              hrows, r_nb, NULL, brow, ds_ctrl);
              }
            } else {
              ds_gemv_zsync(b, d->mk_hidden_out.kind, b->ds.mk_ho_w0, b->ds.mk_ho_w1, b->ds.lat_q8, b->ds.delta,
                            dim, r_nb, NULL, NULL, ds_ctrl);
              void *ada[] = {&hi, &b->ds.delta, &b->ds.mk_hidden_bias, &b->ds.h_q8, &ds_ctrl, (void *)&dim};
              launch(b, K_DS_ADD_Q8_ZSYNC_R5, (dim + 255u) / 256u, ada);
              ds_gemv_zsync(b, b->ds.head_kind, b->ds.head_w0, b->ds.head_w1, b->ds.h_q8, b->ds.logits,
                            hrows, nb_dim, NULL, NULL, ds_ctrl);
            }
            if (zsync_samp) {
              ie_ds_samp *ds = b->base.ds_samp;
              static float ds_spmin = -1.0f, ds_sa = -1.0f, ds_sb = -1.0f;
              if (ds_spmin < 0.0f) {
                ds_spmin = getenv("IE_DS_SAMP_PMIN") ? (float)atof(getenv("IE_DS_SAMP_PMIN")) : 0.45f;
                ds_sa = getenv("IE_DS_SURV_A") ? (float)atof(getenv("IE_DS_SURV_A")) : 0.92f;
                ds_sb = getenv("IE_DS_SURV_B") ? (float)atof(getenv("IE_DS_SURV_B")) : 0.30f;
              }
              u32 ng_s = 64u, K_d = ds->top_k ? (ds->top_k <= 32u ? ds->top_k : 32u) : 20u, K_str = 32u;
              float temp_v = ds->temp, topp_v = ds->top_p;
              float pmin_v = ds->fs_on ? ds->surv_pmin : ds_spmin;
              float surv_a = ds->fs_on ? ds->surv_a : ds_sa;
              float surv_b = ds->fs_on ? ds->surv_b : ds_sb;
              void *pv = zb + 8192u, *pi = zb + 16384u, *ps = zb + 24576u;
              void *rs = zb + offsetof(ds_zsamp_hdr, rng_state);
              void *de = zb + offsetof(ds_zsamp_hdr, draft_entropy);
              void *qn = zb + offsetof(ds_zsamp_hdr, q_n);
              void *qi = zb + offsetof(ds_zsamp_hdr, q_ids);
              void *qp = zb + offsetof(ds_zsamp_hdr, q_probs);
              void *ama[] = {&b->ds.logits, &pv, &pi, &ps, &b->am_count, &ds_ctrl, &rs,
                             &draft_ids, &draft_probs, &de, &qn, &qi, &qp,
                             (void *)&unpad, &ng_s, &K_d, &K_str, &sidx,
                             &cap, &temp_v, &topp_v, &pmin_v, &surv_a, &surv_b};
              launch(b, K_DS_CAP_SAMPLE_ZSYNC_R7, ng_s, ama);
            } else {
              float pmin_v = pmin;
              void *ama[] = {&b->ds.logits, &part_max, &part_id, &part_sum, &b->am_count, &ds_ctrl,
                             &draft_ids, &draft_probs, (void *)&unpad, &cap, &pmin_v, &sidx};
              launch(b, K_DS_CAP_ARGMAX_ZSYNC_R5, ng_am, ama);
            }
            }
          }
          if (zsync_samp) {
            HIP(H.MemcpyAsync(b->ds.zpin, b->ds.zbuf, sizeof(ds_zsamp_hdr), hipMemcpyDeviceToHost, cur_stream));
            HIP(H.StreamSynchronize(cur_stream));
            const ds_zsamp_hdr *zh = (const ds_zsamp_hdr *)b->ds.zpin;
            ie_ds_samp *ds = b->base.ds_samp;
            ds->used = 1;
            ds->rng = zh->rng_state;
            static float ds_drop0_h = -1.0f;
            if (ds_drop0_h < 0.0f) ds_drop0_h = getenv("IE_DS_DROP0_H") ? (float)atof(getenv("IE_DS_DROP0_H")) : 1.65f;
            const float drop0 = ds->fs_on ? ds->drop0_h : ds_drop0_h;
            u32 got = zh->ctrl[2] < nd ? zh->ctrl[2] : nd;
            if (got > 0u && drop0 > 0.0f && zh->draft_entropy[0] > drop0) got = 0u;
            for (u32 i = 0; i < got; i++) {
              if (out) out[i] = zh->draft_ids[i];
              if (prob) prob[i] = zh->draft_probs[i];
              ds->out_entropy[i] = zh->draft_entropy[i];
              const u32 m_i = zh->q_n[i] <= 32u ? zh->q_n[i] : 32u;
              ds->out_n[i] = m_i;
              if (m_i) {
                memcpy(ds->out_id[i], zh->q_ids[i], (size_t)m_i * 4u);
                memcpy(ds->out_q[i], zh->q_probs[i], (size_t)m_i * 4u);
              }
            }
            for (u32 i = got; i < nd; i++) {
              if (out) out[i] = 0u;
              if (prob) prob[i] = 0.0f;
              ds->out_n[i] = 0u;
              ds->out_entropy[i] = 0.0f;
            }
          } else {
            struct { u32 ctrl[4]; u32 ids[16]; float probs[16]; } zh;
            if (b->am_pin && sizeof zh <= 256u) {
              HIP(H.MemcpyAsync(b->am_pin, b->ds.zbuf, sizeof zh, hipMemcpyDeviceToHost, cur_stream));
              HIP(H.StreamSynchronize(cur_stream));
              memcpy(&zh, b->am_pin, sizeof zh);
            } else {
              HIP(H.Memcpy(&zh, b->ds.zbuf, sizeof zh, hipMemcpyDeviceToHost));
            }
            const u32 got = zh.ctrl[2] < nd ? zh.ctrl[2] : nd;
            for (u32 i = 0; i < got; i++) {
              if (out) out[i] = zh.ids[i];
              if (prob) prob[i] = zh.probs[i];
            }
            for (u32 i = got; i < nd; i++) {
              if (out) out[i] = 0u;
              if (prob) prob[i] = 0.0f;
            }
          }
        } else {
          /* legacy host-synchronized Markov loop (IE_DS_ZSYNC=0, shortlist head, or IE_DS_PROF=1) */
          HIP(H.Memset(b->ds.state, 0, (size_t)r * 4u));
          u32 prev_tok = anchor;
          static u32 ran_last;
          u32 have = 0u;
          for (u32 i = 0; i < nd; i++) {
            n_ran++;
            if (dsprof) { td0 = ds_now_ms(); }
            if (use_split_now && i >= have) {
              u32 n = nd - have;
              if (!have && ran_last && ran_last + 1u < n) n = ran_last + 1u;
              for (u32 j = have; j < have + n; j++) {
                float *hj = (float *)((char *)b->ds.hidden + (size_t)(j + 1u) * hs);
                uint8_t *qo = b->ds.hq + (size_t)j * xqs;
                void *aba[] = {&hj, &b->ds.zero, &b->ds.mk_hidden_bias, &qo, (void *)&dim};
                launch(b, K_DS_ADD_Q8_R4, (dim + 255u) / 256u, aba);
              }
              ds_gemv(b, b->ds.head_kind, b->ds.head_w0, b->ds.head_w1, b->ds.hq + (size_t)have * xqs,
                      b->ds.base + (size_t)have * hrows, hrows, nb_dim, NULL, NULL, n, xqs, hrows, 0u, eps);
              have += n;
              if (dsprof) { HIP(H.DeviceSynchronize()); td_head += ds_now_ms() - td0; td0 = ds_now_ms(); }
            }
            float *hi = (float *)((char *)b->ds.hidden + (size_t)(i + 1u) * hs);
            void *ca[] = {&b->ds.state, &b->ds.mk_w1_w0, &b->ds.mk_w1_w1, &hi, &b->ds.frnn, &b->ds.cat_q8, &prev_tok, (void *)&r, (void *)&dim};
            launch(b, K_DS_CONCAT_Q8_R4, (2u * r + 2u * dim + 255u) / 256u, ca);
            ds_gemv(b, d->mk_rnn_proj.kind, b->ds.mk_rnn_w0, b->ds.mk_rnn_w1, b->ds.cat_q8, b->ds.raw768, rnn_rows, rnn_nb, b->ds.mk_rnn_bias, NULL, 1u, 0u, 0u, 0u, eps);
            void *sa[] = {&b->ds.raw768, &b->ds.state, &nul, &b->ds.lat_q8, (void *)&r};
            launch(b, K_DS_RNN_STEP_R4, 1u, sa);
            if (use_split_now) { /* logits = base_i + P lat_i */
              if (dsprof) { HIP(H.DeviceSynchronize()); td_rnn += ds_now_ms() - td0; td0 = ds_now_ms(); }
              ds_gemv(b, b->ds.p_kind, b->ds.p_w0, b->ds.p_w1, b->ds.lat_q8, b->ds.logits, hrows, r_nb, NULL,
                      b->ds.base + (size_t)i * hrows, 1u, 0u, 0u, 0u, eps);
            } else {
              ds_gemv(b, d->mk_hidden_out.kind, b->ds.mk_ho_w0, b->ds.mk_ho_w1, b->ds.lat_q8, b->ds.delta, dim, r_nb, NULL, NULL, 1u, 0u, 0u, 0u, eps);
              void *ada[] = {&hi, &b->ds.delta, &b->ds.mk_hidden_bias, &b->ds.h_q8, (void *)&dim};
              launch(b, K_DS_ADD_Q8_R4, (dim + 255u) / 256u, ada);
              if (dsprof) { HIP(H.DeviceSynchronize()); td_rnn += ds_now_ms() - td0; td0 = ds_now_ms(); }
              ds_gemv(b, b->ds.head_kind, b->ds.head_w0, b->ds.head_w1, b->ds.h_q8, b->ds.logits, hrows, nb_dim, NULL, NULL, 1u, 0u, 0u, 0u, eps);
            }
            void *lca[] = {&b->ds.logits, &b->ds.logits, (void *)&hrows, (void *)&unpad, &cap, &zero, &zero};
            launch_t(b, K_LOGIT_CAP, (hrows + 255u) / 256u, 1u, lca);
            void *ama[] = {&b->ds.logits, (void *)&unpad, &b->ds.argmax, &b->am_part, &b->am_count, &ng_am};
            launch(b, K_ARGMAX, ng_am, ama);
            u32 res[2];
            HIP(H.Memcpy(res, b->ds.argmax, 8u, hipMemcpyDeviceToHost));
            if (dsprof) { td_head += ds_now_ms() - td0; }
            const u32 tok = b->m->ds_head.rows ? model_draft_tok(b->m, res[0]) : res[0];
            float pr_i;
            memcpy(&pr_i, &res[1], 4u);
            if (out) out[i] = tok;
            if (prob) prob[i] = pr_i;
            prev_tok = tok;
            if (pr_i < pmin) {
              for (u32 j = i + 1u; j < nd; j++) {
                if (out) out[j] = 0u;
                if (prob) prob[j] = 0.0f;
              }
              break;
            }
          }
          ran_last = n_ran;
          if (dsprof) {
            static int ds_p_cnt = 0;
            if (ds_p_cnt++ < 5)
              fprintf(stderr, "ds_draft(pos=%u, ran=%u/%u): init=%.2f ms, 3layers=%.2f ms, rnn=%.2f ms, lm_head+am=%.2f ms\n",
                      pos, n_ran, nd, td_init, td_blk, td_rnn, td_head);
          }
        }
      }
      (void)last_only;
      return;
    }
    if (g->df_feat < 0) ie_die("the graph has no DFlash ops");
    tr_run(sec == 2 ? "df_ctx" : "df_block", pos, T);
    run_ops(b, sec == 2 ? g->i_dfc : g->i_dfb, sec == 2 ? g->i_dfb : g->n_ops, toks, T, pos, 0);
    (void)last_only, (void)out, (void)prob;
    return;
  }
  /* IE_PROFILE=2: the ops of the prompt chunks (ms per chunk in the report) */
  static u32 prof_from = 0xFFFFFFFFu; /* IE_PROFILE_FROM=N: only the chunks that start at position N or later */
  if (prof_from == 0xFFFFFFFFu) prof_from = getenv("IE_PROFILE_FROM") ? (u32)atoi(getenv("IE_PROFILE_FROM")) : 0u;
  const int pc = b->prof == 2 && !sec && T > 1u && pos >= prof_from;
  /* IE_PROFILE=3: the ops of the MTP draft runs (one token), reported per op as IE_PROFILE_CSV (gpu_close) */
  const int pd = b->prof == 3 && sec == 1 && T == 1u;
  if (pc || pd) HIP(H.EventRecord(b->e0, NULL));
  tr_run(sec ? "mtp" : last_only ? "prefill" : "verify", pos, T);
  /* P3: non-final prefill chunk (out == NULL, all_logits == 0): run 0 .. i_head asynchronously, skipping output.weight,
   * OP_ARGMAX, and the blocking D2H wait! */
  if (!sec && !out && !b->base.all_logits) {
    mk.rec = 0;
    run_ops(b, 0, g->i_head, toks, T, pos, last_only && T > 1u ? T - 1u : 0u);
    mk_flush();
    if (pc) {
      HIP(H.DeviceSynchronize());
      hipEvent_t prev = b->e0;
      for (uint32_t i = 0; i < g->i_head; i++) {
        float d = 0;
        if (H.EventElapsedTime(&d, prev, b->pev[i]) == 0) b->pms[i] += d, prev = b->pev[i];
      }
      b->psteps++;
    }
    mk_check(pos);
    return;
  }
  /* verify runs and MTP drafts (not prompt chunks): the MTP / Inkling verify megakernel (IE_MK_MTP=0: off) */
  const int mk_kv_ok = !b->kvq8 && !b->kvq4 && b->kvb == 2u && !b->kvrot;
  if (b->m->arch == ARCH_INKLING2) {
    mk.set = T == 1u ? MKSET_INKLING : MKSET_INKLING_VER;
    mk.rec = mk.on && mk_kv_ok && !mk.off[mk.set] && mk.mtp && !sec && !(last_only && T > 1u) && T <= 16u &&
             (mk.mtp == 1 || mk.mtp == 3);
  } else {
    mk.rec = mk.on && mk_kv_ok && !mk.off[MKSET_MTP] && mk.mtp && !(last_only && T > 1u) && (mk.mtp == 1 || (mk.mtp == 2) == (sec != 0));
    mk.set = MKSET_MTP;
  }
  /* A verify run whose rows the caller samples from the top K (backend.h sample_k): the top K is taken right behind the
   * model's output matrix and read back in the one wait that ends this run, and the argmax of each row is not computed. */
  static int fold_on = -1; /* IE_TOPK_FOLD=0: the old way, argmax read, then the top K in a second launch and wait */
  static int fold_am_on = -1; /* IE_ARGMAX_BATCH=0: T separate ie_argmax launches instead of 1-launch batched K=1 */
  if (fold_on < 0) fold_on = !(getenv("IE_TOPK_FOLD") && getenv("IE_TOPK_FOLD")[0] == '0');
  if (fold_am_on < 0) fold_am_on = !(getenv("IE_ARGMAX_BATCH") && getenv("IE_ARGMAX_BATCH")[0] == '0');
  topk_prefetch_clear();
  const u32 nv_tok = b->m->vocab_unpadded ? b->m->vocab_unpadded : b->m->vocab;
  const int fold = fold_on && !sec && !last_only && b->base.sample_k && b->base.topk && topk_fits(T, nv_tok, b->base.sample_k);
  const int fold_am_r7 = fold_on && fold_am_on && !fold && !sec && !last_only && !prob && out && T > 1u && T <= 16u &&
                         !b->prof && !mk.rec && b->k[K_INK_ARGMAX_BATCHED_R7] != NULL;
  const int fold_am = !fold_am_r7 && fold_on && fold_am_on && !fold && !sec && !last_only && !prob && out && T > 1u &&
                      !b->prof && !mk.rec && b->k[K_SALUKI_TOPK_R11] && topk_fits(T, nv_tok, 1u);
  const int fused_q4k_head =
      !sec && !last_only && !b->base.all_logits && !prob && !b->prof && !mk.rec &&
      T >= 1u && T <= 8u && b->k[K_INK_Q4K_HEAD_CAP_TOPK_R7] &&
      (fold ? (b->base.sample_k >= 1u && b->base.sample_k <= 32u) : (out != NULL)) &&
      g->i_head + 1u < g->n_main &&
      g->ops[g->i_head].kind == OP_GEMV_Q4K && !g->ops[g->i_head].nv &&
      g->ops[g->i_head].res < 0 && g->ops[g->i_head].boff == 0u &&
      g->ops[g->i_head].w->rows > 0u && g->ops[g->i_head].w->nb > 0u &&
      (g->ops[g->i_head].w->nb & 7u) == 0u &&
      g->ops[g->i_head + 1u].kind == OP_LOGIT_CAP &&
      g->ops[g->i_head + 1u].a == g->ops[g->i_head].b &&
      g->ops[g->i_head + 1u].aoff == 0u && g->ops[g->i_head + 1u].boff == 0u &&
      g->ops[g->i_head + 1u].n == g->ops[g->i_head].w->rows &&
      g->ops[g->i_head + 1u].b == g->logits &&
      (g->i_head + 2u == g->n_main || (g->i_head + 3u == g->n_main && g->ops[g->i_head + 2u].kind == OP_ARGMAX));
  b->skip_argmax = fold || fold_am || fold_am_r7 || fused_q4k_head;
  const int tree = !sec && !last_only && b->base.tree_par;
  if (tree) {
    if (!gpu_tree_ok(&b->base, pos, T)) ie_die("a tree verify of %u nodes at %u it cannot run", T, pos);
    tree_begin(b, pos, T);
  }
  struct ie_mtp_tree *mtr = graph_mtp_tr(g);
  const int mtree = sec == 1 && mtr->on;
  if (mtree) {
    mk.rec = 0;
    mtp_tree_begin(b, mtr, pos, T);
  }
  const u32 nv_mtp = b->m->out_draft.rows ? b->m->out_draft.rows : b->m->vocab;
  const int fold_mtp = fold_on && sec == 1 && mtr->topk_k && b->base.topk && topk_fits(mtr->topk_n, nv_mtp, mtr->topk_k);
  if (fold_mtp) mk.rec = 0;
  b->tr_on = tree || mtree;
  const u32 i1 = sec ? (fold_mtp ? g->n_mtp - 1u : g->n_mtp) : (fused_q4k_head ? g->i_head : g->n_main);
  const u32 wf = mtree ? 0u : (last_only ? T - 1u : 0u);
  const int defer_sc = !sec && b->m->arch == ARCH_INKLING2 && T > 1u && !last_only && wf == 0u &&
                       b->sc_hist && b->k[K_INK_SCONV_HIST_R4] && b->k[K_INK_SCONV_COMMIT_R4];
  b->sc_defer = defer_sc;
  if (defer_sc) b->sc_ver_pos = pos, b->sc_ver_T = T;
  if (!sec && pos == 0u) ds_ctx_drain(b);
  const int do_ds_ov = !sec && b->ds.ov_mode && b->ds.ctx_stream && b->ds.ready && !last_only &&
                       T >= 2u && !tree && !b->prof && !gdump_on() && !mk.rec && !fused_q4k_head &&
                       g->ds_feat >= 0 && g->i_head + 1u < i1;
  if (do_ds_ov) {
    const u32 split_op = (b->ds.ov_mode == 1) ? g->i_head : (g->i_head + 1u);
    for (u32 il = 0; il < b->m->ds->n_layer; il++)
      if (b->ds.kv_vm[il]) { ds_kv_ensure(b, pos + T); break; }
    if (b->ds.ctx_pending) {
      HIP(H.StreamWaitEvent(cur_stream, b->ds.ev_ctx_done, 0));
      b->ds.ctx_pending = 0;
    }
    run_ops(b, 0, split_op, toks, T, pos, wf);
    b->sc_defer = 0;
    HIP(H.EventRecord(b->ds.ev_feat_ready, cur_stream));
    {
      hipStream_t saved = cur_stream;
      cur_stream = b->ds.ctx_stream;
      HIP(H.StreamWaitEvent(cur_stream, b->ds.ev_feat_ready, 0));
      ds_ctx_kv_launch(b, pos, T, 0, NULL, NULL, NULL, NULL);
      HIP(H.EventRecord(b->ds.ev_ctx_done, cur_stream));
      cur_stream = saved;
      b->ds.ctx_pending = 1;
      b->ds.ctx_pos = pos;
      b->ds.ctx_T = T;
    }
    run_ops(b, split_op, i1, toks, T, pos, wf);
  } else {
    run_ops(b, sec ? g->n_main : 0, i1, toks, T, pos, wf);
    b->sc_defer = 0;
  }
  b->tr_on = 0;
  b->skip_argmax = 0;
  mk.rec = 0;
  mk_flush();
  if (fused_q4k_head) {
    ink_fused_q4k_head_run(b, T, fold, out);
    mk_check(pos);
    return;
  }
  if (pc) {
    HIP(H.DeviceSynchronize()); /* the events are on the NULL stream: run with IE_STREAM=0 for exact times */
    hipEvent_t prev = b->e0;
    for (uint32_t i = 0; i < g->n_main; i++) { /* an op a chunk skips has no event: time 0 */
      float d = 0;
      if (H.EventElapsedTime(&d, prev, b->pev[i]) == 0) b->pms[i] += d, prev = b->pev[i];
    }
    b->psteps++;
  }
  if (pd) {
    HIP(H.DeviceSynchronize());
    hipEvent_t prev = b->e0;
    for (uint32_t i = g->n_main; i < g->n_mtp; i++) {
      float d = 0;
      if (H.EventElapsedTime(&d, prev, b->pev[i]) == 0) b->pms[i] += d, prev = b->pev[i];
    }
    b->psteps++;
  }
  if (fold_mtp) {
    if (gpu_topk_lse(&b->base, g->mtp_logits, mtr->topk_row, mtr->topk_n, nv_mtp, mtr->topk_k, mtr->topk_v, mtr->topk_i, mtr->topk_lse))
      mtr->topk_done = 1;
    if (out) {
      for (u32 t = 0; t < T; t++) {
        out[t] = 0;
        if (prob) prob[t] = 0.0f;
      }
    }
    mk_check(pos);
    return;
  }
  if (fold) {
    topk_prefetch(b, T); /* launches the top K and waits for it */
    if (out) {
      for (u32 t = 0; t < T; t++) {
        out[t] = 0; /* no argmax this run */
        if (prob) prob[t] = 0.0f;
      }
    }
    mk_check(pos);
    return;
  }
  if (fold_am_r7) {
    const buf *lgb = &g->bufs[g->logits], *amb = &g->bufs[g->argmax];
    void *lx = b->arena + lgb->off, *lo = b->arena + amb->off;
    u32 n_arg = nv_tok, xs_b = lgb->stride, os_b = amb->stride, np_arg = 0u;
    u32 ng_arg = T >= 4u ? 32u : 64u;
    float cap0 = 0.0f;
    void *ama[] = {&lx, &n_arg, &xs_b, &cap0, &np_arg, &lo, &os_b, &b->am_part, &b->am_count, &ng_arg};
    launch_t(b, K_INK_ARGMAX_BATCHED_R7, ng_arg, T, ama);
    const size_t am_bytes = (size_t)T * amb->stride;
    if (b->am_pin && am_bytes <= 256u) {
      HIP(H.MemcpyAsync(b->am_pin, lo, am_bytes, hipMemcpyDeviceToHost, cur_stream));
      HIP(H.StreamSynchronize(cur_stream));
      for (u32 t = 0; t < T; t++) out[t] = b->am_pin[t * amb->stride / 4u];
    } else {
      u32 *tmp = malloc(am_bytes);
      if (!tmp) ie_die("out of memory");
      HIP(H.Memcpy(tmp, lo, am_bytes, hipMemcpyDeviceToHost));
      for (u32 t = 0; t < T; t++) out[t] = tmp[t * amb->stride / 4u];
      free(tmp);
    }
    mk_check(pos);
    return;
  }
  if (fold_am) {
    float am_val[16];
    if (!gpu_topk(&b->base, (int)g->logits, 0, T, nv_tok, 1u, am_val, out)) ie_die("batched verify argmax: sizes do not fit");
    mk_check(pos);
    return;
  }
  if (!out) {
    mk_check(pos);
    return;
  }
  const buf *am = &g->bufs[sec ? g->mtp_argmax : g->argmax];
  const size_t am_bytes = (size_t)T * am->stride;
  u32 *tmp = NULL;
  if (b->am_pin && am_bytes <= 256u) {
    HIP(H.MemcpyAsync(b->am_pin, b->arena + am->off, am_bytes, hipMemcpyDeviceToHost, cur_stream));
    HIP(H.StreamSynchronize(cur_stream));
    tmp = b->am_pin;
  } else {
    tmp = malloc(am_bytes);
    if (!tmp) ie_die("out of memory");
    HIP(H.Memcpy(tmp, b->arena + am->off, am_bytes, hipMemcpyDeviceToHost));
  }
  for (u32 t = 0; t < T; t++) {
    out[t] = tmp[t * am->stride / 4u];
    if (prob) memcpy(&prob[t], &tmp[t * am->stride / 4u + 1u], 4);
  }
  if (tmp != b->am_pin) free(tmp);
  mk_check(pos);
}

/* The prompt cache snapshot: the current state slot of each linear layer,
 * the whole conv ring (it is indexed by position), and row 0 of hbuf. It is kept in host memory (a save or restore is
 * a copy of about 160 MB over PCIe, tens of ms; it is 2 snapshots less in VRAM, which matters at 128K); IE_SNAP_HOST=0:
 * in VRAM. */
enum { SNAP_PIN_CH = 4u << 20 };
static int snap_host(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_SNAP_HOST") && getenv("IE_SNAP_HOST")[0] == '0');
  return on;
}
static int snap_skip_on(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_SNAP_SKIP") && getenv("IE_SNAP_SKIP")[0] == '0');
  return on;
}
static int snap_compact_swa(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_SNAP_COMPACT_SWA") && getenv("IE_SNAP_COMPACT_SWA")[0] == '0');
  return on;
}
static int snap_pipe_enabled(void) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_SNAP_PIPE") && getenv("IE_SNAP_PIPE")[0] == '0');
  return on;
}
static void snap_stream_ensure(gpu_backend *b) {
  if (!b->snap_cs) HIP(H.StreamCreateWithFlags(&b->snap_cs, 1u));
}
typedef struct {
  int on;
  int in_flight[2];
  void *d2h_dst[2];
  size_t d2h_len[2];
  size_t slot;
} snap_pipe;
static void snap_copy_pipe(gpu_backend *b, void *dst, const void *src, size_t n, int save, snap_pipe *pp) {
  if (!n) return;
  if (!snap_host()) {
    HIP(H.Memcpy(dst, src, n, hipMemcpyDeviceToDevice));
    return;
  }
  if (!pp || !pp->on) {
    HIP(H.Memcpy(dst, src, n, save ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice));
    return;
  }
  char *pin[2] = {b->snap_pin, b->snap_pin + SNAP_PIN_CH};
  for (size_t off = 0; off < n; off += SNAP_PIN_CH) {
    const size_t len = n - off < SNAP_PIN_CH ? n - off : (size_t)SNAP_PIN_CH;
    const size_t s = pp->slot;
    if (save) {
      if (pp->in_flight[s]) {
        HIP(H.EventSynchronize(b->snap_ev[s]));
        memcpy(pp->d2h_dst[s], pin[s], pp->d2h_len[s]);
        pp->in_flight[s] = 0;
      }
      HIP(H.MemcpyAsync(pin[s], (const char *)src + off, len, hipMemcpyDeviceToHost, b->snap_cs));
      HIP(H.EventRecord(b->snap_ev[s], b->snap_cs));
      pp->d2h_dst[s] = (char *)dst + off;
      pp->d2h_len[s] = len;
      pp->in_flight[s] = 1;
    } else {
      if (pp->in_flight[s]) HIP(H.EventSynchronize(b->snap_ev[s]));
      memcpy(pin[s], (const char *)src + off, len);
      HIP(H.MemcpyAsync((char *)dst + off, pin[s], len, hipMemcpyHostToDevice, b->snap_cs));
      HIP(H.EventRecord(b->snap_ev[s], b->snap_cs));
      pp->in_flight[s] = 1;
    }
    pp->slot ^= 1u;
  }
}
static void snap_pipe_flush(gpu_backend *b, snap_pipe *pp, int save) {
  if (!pp || !pp->on) return;
  char *pin[2] = {b->snap_pin, b->snap_pin + SNAP_PIN_CH};
  if (save) {
    for (int k = 0; k < 2; k++) {
      const size_t s = (pp->slot + (size_t)k) & 1u;
      if (pp->in_flight[s]) {
        HIP(H.EventSynchronize(b->snap_ev[s]));
        memcpy(pp->d2h_dst[s], pin[s], pp->d2h_len[s]);
        pp->in_flight[s] = 0;
      }
    }
  } else {
    if (pp->in_flight[0] || pp->in_flight[1]) HIP(H.StreamSynchronize(b->snap_cs));
    pp->in_flight[0] = pp->in_flight[1] = 0;
  }
}
static void snap_ring_slice(gpu_backend *b, char *sp, char *gpu_base, size_t rowb, u32 len, u32 w, u32 pos, int save, snap_pipe *pp) {
  if (w >= len) {
    snap_copy_pipe(b, save ? (void *)sp : (void *)gpu_base, save ? (void *)gpu_base : (void *)sp, (size_t)len * rowb, save, pp);
    return;
  }
  if (!save) {
    static int poison = -1;
    if (poison < 0) poison = (getenv("IE_SNAP_POISON_SWA") && getenv("IE_SNAP_POISON_SWA")[0] == '1');
    if (poison) {
      /* Debug verification: fill entire GPU SWA ring with 0x7E7E (f16 qNaN) before scattering valid rows */
      snap_pipe_flush(b, pp, 0);
      HIP(H.Memset(gpu_base, 0x7E, (size_t)len * rowb));
      HIP(H.StreamSynchronize(cur_stream));
    }
  }
  const u32 keep = pos < w ? pos : w;
  if (keep > 0u) {
    const u32 r0 = (pos - keep) % len;
    const u32 first = (len - r0) < keep ? (len - r0) : keep;
    const u32 second = keep - first;
    const size_t n1 = (size_t)first * rowb;
    snap_copy_pipe(b, save ? (void *)sp : (void *)(gpu_base + (size_t)r0 * rowb),
                   save ? (void *)(gpu_base + (size_t)r0 * rowb) : (void *)sp, n1, save, pp);
    if (second > 0u) {
      const size_t n2 = (size_t)second * rowb;
      snap_copy_pipe(b, save ? (void *)(sp + n1) : (void *)gpu_base,
                     save ? (void *)gpu_base : (void *)(sp + n1), n2, save, pp);
    }
  }
  if (save && keep < w) {
    const size_t zoff = (size_t)keep * rowb, zlen = (size_t)(w - keep) * rowb;
    if (snap_host()) memset(sp + zoff, 0, zlen);
    else HIP(H.Memset(sp + zoff, 0, zlen));
  }
}
static inline u32 snap_swa_rows(const model *m, const graph *g, u32 i) {
  const u32 len = g->kv_len[i];
  if (snap_compact_swa() && m->kv_win && m->kv_win[i] && m->kv_win[i] < len) return m->kv_win[i];
  return len;
}
/* snapshot k's arrays (at its first use) */
static void snap_alloc(gpu_backend *b, int k) {
  const model *m = b->m;
  const size_t per = (size_t)m->n_vh * m->sd * m->sd * 4, rbytes = (size_t)m->n_rec * GDN_RING * m->conv_dim * 4;
  const size_t sh = b->g->bufs[b->g->h_out].stride + 4; /* the rows of h buffers have one size */
  if (snap_host()) {
    b->snap_ring[k] = malloc(rbytes + 4), b->snap_st[k] = malloc((size_t)m->n_rec * per + 4), b->snap_h[k] = malloc(sh);
    b->snap_sc[k] = b->sc_ring_bytes ? malloc(b->sc_ring_bytes + 4) : NULL;
    if (!b->snap_ring[k] || !b->snap_st[k] || !b->snap_h[k] || (b->sc_ring_bytes && !b->snap_sc[k])) ie_die("out of memory (prompt cache snapshot)");
    if (!b->snap_pin) {
      if (H.HostMalloc((void **)&b->snap_pin, 2u * SNAP_PIN_CH, 0) == 0 && b->snap_pin) {
        snap_stream_ensure(b);
        HIP(H.EventCreate(&b->snap_ev[0]));
        HIP(H.EventCreate(&b->snap_ev[1]));
      } else {
        b->snap_pin = NULL;
      }
    }
  } else {
    HIP(H.Malloc((void **)&b->snap_ring[k], rbytes + 4));
    HIP(H.Malloc((void **)&b->snap_st[k], (size_t)m->n_rec * per + 4));
    HIP(H.Malloc((void **)&b->snap_h[k], sh));
    if (b->sc_ring_bytes) HIP(H.Malloc((void **)&b->snap_sc[k], b->sc_ring_bytes + 4));
  }
}
static inline int ds_is_ring(const gpu_backend *b, u32 il) { return b->ds.kv_cap[il] < b->g->n_ctx + 16u; }
static inline u32 ds_snap_rows(const gpu_backend *b, u32 il) {
  const u32 cap = b->ds.kv_cap[il];
  const u32 win = b->m->ds->l[il].win;
  if (snap_compact_swa() && win && win < cap) return win;
  return cap;
}
static size_t ds_snap_bytes(const gpu_backend *b) {
  if (!b->ds.ready || !b->m->ds) return 0;
  size_t nr = 0;
  for (u32 il = 0; il < b->m->ds->n_layer; il++)
    if (ds_is_ring(b, il)) nr += ds_snap_rows(b, il);
  return 2u * nr * b->ds.kv_rowb + (size_t)b->m->ds->dim * 4u;
}
static void gpu_snap(backend *bk, int save, int k, int hbuf, uint32_t pos) {
  gpu_backend *b = (gpu_backend *)bk;
  ds_ctx_drain(b);
  const model *m = b->m;
  const graph *g = b->g;
  const size_t per = (size_t)m->n_vh * m->sd * m->sd * 4, rbytes = (size_t)m->n_rec * GDN_RING * m->conv_dim * 4;
  const size_t hb = hbuf >= 0 ? g->bufs[hbuf].stride : 0;
  if (!b->snap_ring[k]) {
    if (!save) ie_die("gpu_snap: no snapshot %d to restore", k);
    snap_alloc(b, k);
  }
  size_t nr = 0;
  for (uint32_t i = 0; i < m->n_kvl; i++)
    if (g->kv_len[i] < g->n_ctx) nr += snap_swa_rows(m, g, i);
  const size_t ds_bytes = ds_snap_bytes(b);
  if (!save && snap_skip_on() && b->live_snap == k && (hbuf < 0 || b->live_hbuf == hbuf) &&
      ((!nr && !b->sc_ring_bytes && !ds_bytes) || (m->arch == ARCH_INKLING2 && b->live_snap_pos == pos))) {
    static int poison_chk = -1;
    if (poison_chk < 0) poison_chk = (getenv("IE_SNAP_POISON_SWA") && getenv("IE_SNAP_POISON_SWA")[0] == '1');
    if (!poison_chk) {
      b->lazy_mask = 0;
      b->lazy_rec = NULL;
      b->rec_n = 0;
      return;
    }
  }
  if (save) gpu_flush_gdn_replay(b);
  else {
    b->lazy_mask = 0;
    b->lazy_rec = NULL;
    b->rec_n = 0;
  }
  snap_pipe pp = {0};
  pp.on = snap_host() && b->snap_pin && snap_pipe_enabled();
  if (pp.on) HIP(H.StreamSynchronize(cur_stream));
  float *snap_ring = b->snap_ring[k], *snap_st = b->snap_st[k];
  char *snap_h = b->snap_h[k];
  if (b->ns == 1u && m->n_rec) {
    snap_copy_pipe(b, save ? (void *)snap_st : (void *)b->st, save ? (void *)b->st : (void *)snap_st, (size_t)m->n_rec * per, save, &pp);
  } else {
    for (uint32_t l = 0; l < m->n_rec; l++) {
      char *live = (char *)b->st + ((size_t)l * b->ns + b->slot) * per, *keep = (char *)snap_st + (size_t)l * per;
      snap_copy_pipe(b, save ? keep : live, save ? live : keep, per, save, &pp);
    }
  }
  if (rbytes) snap_copy_pipe(b, save ? (void *)snap_ring : (void *)b->ring, save ? (void *)b->ring : (void *)snap_ring, rbytes, save, &pp);
  if (b->sc_ring_bytes) snap_copy_pipe(b, save ? (void *)b->snap_sc[k] : (void *)b->sc_ring, save ? (void *)b->sc_ring : (void *)b->snap_sc[k], b->sc_ring_bytes, save, &pp);
  if (hb) {
    char *row = b->arena + g->bufs[hbuf].off;
    snap_copy_pipe(b, save ? snap_h : row, save ? row : snap_h, hb, save, &pp);
  }
  /* a ring slot is reused by a later position: keep the live window (or full ring when IE_SNAP_COMPACT_SWA=0) */
  const size_t kvb = b->kvrowb;
  if (nr) {
    if (!b->snap_kv[k]) {
      if (snap_host()) {
        b->snap_kv[k] = malloc(2u * nr * kvb);
        if (!b->snap_kv[k]) ie_die("out of memory (prompt cache KV ring snapshot)");
      } else {
        HIP(H.Malloc((void **)&b->snap_kv[k], 2u * nr * kvb));
      }
    }
    char *sp = b->snap_kv[k];
    for (uint32_t i = 0; i < m->n_kvl; i++) {
      if (g->kv_len[i] >= g->n_ctx) continue;
      const u32 len = g->kv_len[i], w = snap_swa_rows(m, g, i);
      const size_t wn = (size_t)w * kvb;
      char *kp = kv_ptr(b, 0, i, 0), *vp = kv_ptr(b, 1, i, 0);
      snap_ring_slice(b, sp, kp, kvb, len, w, pos, save, &pp);
      sp += wn;
      snap_ring_slice(b, sp, vp, kvb, len, w, pos, save, &pp);
      sp += wn;
    }
  }
  if (ds_bytes) {
    if (!b->snap_dskv[k]) {
      if (snap_host()) {
        b->snap_dskv[k] = malloc(ds_bytes);
        if (!b->snap_dskv[k]) ie_die("out of memory (dspark KV ring snapshot)");
      } else {
        HIP(H.Malloc((void **)&b->snap_dskv[k], ds_bytes));
      }
    }
    char *sp = b->snap_dskv[k];
    for (u32 il = 0; il < m->ds->n_layer; il++) {
      if (!ds_is_ring(b, il)) continue;
      const u32 len = b->ds.kv_cap[il], w = ds_snap_rows(b, il);
      const size_t wn = (size_t)w * b->ds.kv_rowb;
      snap_ring_slice(b, sp, (char *)b->ds.kc[il], b->ds.kv_rowb, len, w, pos, save, &pp);
      sp += wn;
      snap_ring_slice(b, sp, (char *)b->ds.vc[il], b->ds.kv_rowb, len, w, pos, save, &pp);
      sp += wn;
    }
    const size_t fb = (size_t)m->ds->dim * 4u;
    snap_copy_pipe(b, save ? (void *)sp : (void *)b->ds.frnn, save ? (void *)b->ds.frnn : (void *)sp, fb, save, &pp);
  }
  snap_pipe_flush(b, &pp, save);
  if ((!nr && !ds_bytes) || m->arch == ARCH_INKLING2) {
    b->live_snap = k;
    b->live_snap_pos = pos;
    b->live_hbuf = hbuf;
  }
}
/* backend.snap_io / kv_io (the disk prompt cache): snapshot k is the linear attention states, the conv ring slots a run
 * from position pos reads (the conv of token pos looks back at the inputs of positions pos - 3 .. pos - 1, d_conv - 1
 * of them: ie_gdn reads the ring only there, before the launch's own tokens, and every later position is read after its
 * slot is written, also after a verify run's rollback) and an h row. The other slots read as zero after a restore.
 * inkling2 adds what gpu_snap keeps besides: its short-conv ring (sc_ring, every visit's last 4 inputs, indexed by
 * pos & 3), the compact (or full) K/V rings of its sliding window visits, and (when DSpark is loaded) DSpark's SWA KV rings + frnn.
 * kv_io then moves the full-length layers (including DSpark's global layers). Other models with rings (flint) are off. */
static size_t gpu_snap_io(backend *bk, int k, int out, void *buf, uint32_t pos) {
  gpu_backend *b = (gpu_backend *)bk;
  ds_ctx_drain(b);
  const model *m = b->m;
  const graph *g = b->g;
  size_t nr = 0; /* ring rows (sliding window layers), as gpu_snap counts them */
  for (uint32_t i = 0; i < m->n_kvl; i++)
    if (g->kv_len[i] < g->n_ctx) nr += snap_swa_rows(m, g, i);
  if (nr && m->arch != ARCH_INKLING2) return SIZE_MAX;
  const uint32_t rs = m->d_conv - 1u; /* ring slots kept per layer */
  const size_t per = (size_t)m->n_vh * m->sd * m->sd * 4, rowb = (size_t)m->conv_dim * 4, rall = (size_t)m->n_rec * GDN_RING * rowb;
  const size_t ds_bytes = ds_snap_bytes(b);
  const size_t part[6] = {(size_t)m->n_rec * per, (size_t)m->n_rec * rs * rowb, g->bufs[g->h_out].stride + 4,
                          m->arch == ARCH_INKLING2 ? b->sc_ring_bytes : 0, 2u * nr * b->kvrowb, ds_bytes};
  const size_t total = part[0] + part[1] + part[2] + part[3] + part[4] + part[5];
  if (!buf) return total;
  if (!b->snap_ring[k]) {
    if (out) ie_die("gpu_snap_io: no snapshot %d to write out", k);
    snap_alloc(b, k);
  }
  const int host = snap_host();
  if (part[4] && !b->snap_kv[k]) { /* gpu_snap allocates the same size at its first save */
    if (out) ie_die("gpu_snap_io: snapshot %d has no KV rings to write out", k);
    if (host) {
      b->snap_kv[k] = malloc(part[4]);
      if (!b->snap_kv[k]) ie_die("out of memory (prompt cache KV ring snapshot)");
    } else {
      HIP(H.Malloc((void **)&b->snap_kv[k], part[4]));
    }
  }
  if (part[5] && !b->snap_dskv[k]) {
    if (out) ie_die("gpu_snap_io: snapshot %d has no DSpark rings to write out", k);
    if (host) {
      b->snap_dskv[k] = malloc(part[5]);
      if (!b->snap_dskv[k]) ie_die("out of memory (dspark KV ring snapshot)");
    } else {
      HIP(H.Malloc((void **)&b->snap_dskv[k], part[5]));
    }
  }
  if (!out && b->live_snap == k) b->live_snap = -1, b->live_hbuf = -1;
  void *mine[6] = {b->snap_st[k], b->snap_ring[k], b->snap_h[k], b->snap_sc[k], b->snap_kv[k], b->snap_dskv[k]};
  char *p = buf;
  for (int i = 0; i < 6; p += part[i++]) {
    if (!part[i] || i == 1) continue;
    if (host) memcpy(out ? (void *)p : mine[i], out ? mine[i] : (void *)p, part[i]);
    else HIP(H.Memcpy(out ? (void *)p : mine[i], out ? mine[i] : (void *)p, part[i], out ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice));
  }
  if (part[1]) {
    char *ring = mine[1], *q = buf;
    q += part[0];
    if (!out) { /* slots not in the file: zero */
      if (host) memset(ring, 0, rall);
      else HIP(H.Memset(ring, 0, rall));
    }
    for (uint32_t l = 0; l < m->n_rec; l++)
      for (uint32_t j = 0; j < rs; j++, q += rowb) {
        char *slot = ring + ((size_t)l * GDN_RING + ((pos + GDN_RING - rs + j) & (GDN_RING - 1u))) * rowb;
        if (host) memcpy(out ? (void *)q : (void *)slot, out ? (void *)slot : (void *)q, rowb);
        else HIP(H.Memcpy(out ? (void *)q : (void *)slot, out ? (void *)slot : (void *)q, rowb, out ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice));
      }
  }
  return total;
}
/* pinned host memory for the disk cache's buffers (NULL if the driver refuses) */
static void *gpu_host_alloc(backend *bk, size_t n) {
  (void)bk;
  void *p = NULL;
  if (H.HostMalloc(&p, n, 0) != 0 || !p) return NULL;
  if ((uintptr_t)p & 4095u) {
    H.HostFree(p);
    return NULL;
  }
  return p;
}
static void gpu_host_free(backend *bk, void *p) {
  (void)bk;
  if (p) H.HostFree(p);
}
static void gpu_kv_io(backend *bk, int out, uint32_t p0, uint32_t n, void *buf) {
  gpu_backend *b = (gpu_backend *)bk;
  ds_ctx_drain(b);
  if (!n) {
    if (!out && !buf && p0) {
      kv_ensure(b, p0);
      if (b->ds.ready) ds_kv_ensure(b, p0);
    }
    return;
  }
  if (!out) {
    b->live_snap = -1;
    kv_ensure(b, p0 + n);
    if (b->ds.ready) ds_kv_ensure(b, p0 + n);
  }
  const int async_io = snap_pipe_enabled();
  if (async_io) {
    snap_stream_ensure(b);
    HIP(H.StreamSynchronize(cur_stream));
  }
  const int kind = out ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice;
  const size_t len = (size_t)n * b->kvrowb;
  char *p = buf;
  for (uint32_t i = 0; i < b->m->n_kvl; i++) {
    if (b->g->kv_len[i] < b->g->n_ctx) continue; /* a ring (sliding window): in the snapshot, gpu_snap_io */
    for (int v = 0; v < 2; v++, p += len) {
      char *rows = kv_ptr(b, v, i, p0);
      if (async_io) HIP(H.MemcpyAsync(out ? (void *)p : rows, out ? (void *)rows : (void *)p, len, kind, b->snap_cs));
      else HIP(H.Memcpy(out ? (void *)p : rows, out ? (void *)rows : (void *)p, len, kind));
    }
  }
  if (b->ds.ready && b->m->ds) {
    const size_t dlen = (size_t)n * b->ds.kv_rowb;
    for (u32 il = 0; il < b->m->ds->n_layer; il++) {
      if (ds_is_ring(b, il)) continue;
      for (int v = 0; v < 2; v++, p += dlen) {
        char *rows = (char *)(v ? b->ds.vc[il] : b->ds.kc[il]) + (size_t)p0 * b->ds.kv_rowb;
        if (async_io) HIP(H.MemcpyAsync(out ? (void *)p : rows, out ? (void *)rows : (void *)p, dlen, kind, b->snap_cs));
        else HIP(H.Memcpy(out ? (void *)p : rows, out ? (void *)rows : (void *)p, dlen, kind));
      }
    }
  }
  if (async_io) HIP(H.StreamSynchronize(b->snap_cs));
}

/* Debug: IE_GDN_HASH=1 prints a hash of the linear attention state (the current slot of every layer) after each
 * accept, to compare IE_GDN_RECORDS=0 and 1; IE_GDN_DUMP=N,FILE also writes the state after accept N to FILE
 * (the first mismatch of two such files: tools or cmp -l). */
static void gdn_state_check(gpu_backend *b, uint32_t k) {
  static int hash = -1, dump_at = -1;
  static const char *dump_file;
  static unsigned n_acc;
  if (hash < 0) {
    hash = getenv("IE_GDN_HASH") != NULL;
    const char *e = getenv("IE_GDN_DUMP");
    if (e && strchr(e, ',')) dump_at = atoi(e), dump_file = strchr(e, ',') + 1;
  }
  const model *m = b->m;
  if (!(hash || dump_at >= 0) || !m->n_rec) return;
  gpu_flush_gdn_replay(b);
  const size_t per = (size_t)m->n_vh * m->sd * m->sd * 4;
  unsigned char *buf = malloc(per);
  FILE *f = (int)n_acc == dump_at ? fopen(dump_file, "wb") : NULL;
  uint64_t hs = 1469598103934665603ull;
  HIP(H.DeviceSynchronize());
  for (uint32_t l = 0; l < m->n_rec; l++) {
    HIP(H.Memcpy(buf, (char *)b->st + ((size_t)l * b->ns + b->slot) * per, per, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < per; i++) hs = (hs ^ buf[i]) * 1099511628211ull;
    if (f) fwrite(buf, 1, per, f);
  }
  if (f) fclose(f);
  free(buf);
  fprintf(stderr, "gdn state after accept %u (k=%u): %016llx\n", n_acc++, k, (unsigned long long)hs);
}

static void gpu_accept(backend *bk, uint32_t k) {
  tr_run("commit", tr_pos, k + 1u);
  gpu_backend *b = (gpu_backend *)bk;
  b->live_snap = -1;
  b->live_hbuf = -1;
  b->q16_ready_op = -1;
  b->slot = (b->slot + k) % b->ns;
  /* with records: the slot holds the state after token 0 of the verify run; apply the records of tokens 1 .. k */
  if (b->rec_n && k) {
    if (k > b->rec_n) ie_die("accept %u: the last verify run has records for %u tokens", k, b->rec_n);
    const model *m = b->m;
    /* the kernel reads these strides as 64-bit values: uint64_t here (unsigned long is 32 bits on Windows) */
    uint64_t sls = (uint64_t)m->n_vh * m->sd * m->sd, rls = (uint64_t)(b->vmax - 1u) * m->n_vh * GDN_RECF;
    u32 nv = m->n_vh, n = k;
    /* the largest float the replay touches in the state and in the records, against the allocations */
    if (b->ns != 1u || (uint64_t)m->n_rec * rls * 4 > b->rec_bytes || (uint64_t)(n - 1u) * nv * GDN_RECF + nv * GDN_RECF > rls)
      ie_die("gdn replay: out of the state or record allocation (ns %u, k %u)", b->ns, k);
    u32 mask = ((2u << n) - 1u) & ~1u; /* the nodes 1 .. k */
    gpu_flush_gdn_replay(b);
    if (gdn_eager_replay() || !b->rec_buf[1]) {
      void *args[] = {&b->st, &b->rec, &nv, &mask, &sls, &rls};
      launch(b, K_GDN_REPLAY, m->n_rec * nv, args);
    } else {
      b->lazy_rec = b->rec;
      b->lazy_mask = mask;
      b->lazy_rls = rls;
      b->rec_idx ^= 1;
      b->rec = b->rec_buf[b->rec_idx];
    }
  }
  b->rec_n = 0;
  if (b->sc_ver_T) {
    u32 tot = b->m->sc_tot, pos0 = b->sc_ver_pos, n_acc = k + 1u, hrs = b->sc_hist_stride;
    if (n_acc > b->sc_ver_T) ie_die("sconv commit: n_acc %u > ver_T %u", n_acc, b->sc_ver_T);
    void *ca[] = {&b->sc_ring, &b->sc_hist, &tot, &pos0, &n_acc, &hrs};
    launch(b, K_INK_SCONV_COMMIT_R4, (tot + 255u) / 256u, ca);
    b->sc_ver_T = 0;
  }
  if (b->ink_sp_used && b->ink_sp_sync) {
    u32 pf = 0u;
    HIP(H.Memcpy(&pf, b->ink_sp_sync + 2u, 4u, hipMemcpyDeviceToHost));
    b->ink_sp_used = 0;
    if (pf != 0u) {
      b->ink_sp_poisoned = 1;
      fprintf(stderr, "gpu: ie_inkling_subspan_glue_r8 POISONED (flag=%u) -> latching off for process\n", pf);
    }
  }
  gdn_state_check(b, k);
}

/* ---- tree verify ---- */
/* the node table (kernels/ie_kernels.c TR_*) */
#define TR_DEP 0u
#define TR_PATH 16u
#define TR_PAR 272u
#define TR_A1 288u
#define TR_ANC 336u
static void tree_table(unsigned char *t, const uint8_t *par, u32 n) {
  memset(t, 0, 384);
  memset(t + TR_A1, 0xFF, 48);
  unsigned short anc[16] = {0};
  for (u32 j = 0; j < n; j++) {
    const u32 p = j ? par[j] : 0u, d = j ? t[TR_DEP + p] + 1u : 0u;
    t[TR_DEP + j] = (unsigned char)d, t[TR_PAR + j] = (unsigned char)p;
    if (j) anc[j] = (unsigned short)(anc[p] | (1u << j));
    for (u32 a = j, e = d;; a = t[TR_PAR + a], e--) { /* the path: node a at depth e */
      t[TR_PATH + 16u * j + e] = (unsigned char)a;
      if (!e) break;
    }
    for (u32 x = 1; x <= 3u && x <= d; x++) t[TR_A1 + 16u * (x - 1u) + j] = t[TR_PATH + 16u * j + d - x];
  }
  memcpy(t + TR_ANC, anc, 32);
  t[IE_TR_KEND] = (unsigned char)n;
}
static int gpu_tree_ok(backend *bk, uint32_t pos, uint32_t T) {
  gpu_backend *b = (gpu_backend *)bk;
  static int on = -1; /* IE_TREE=0: no tree verify */
  if (on < 0) on = !(getenv("IE_TREE") && getenv("IE_TREE")[0] == '0');
  if (b->m->arch == ARCH_INKLING2) {
    const int ink2 = b->k[K_INK_REL_ATTN_VER_TREE_R8] && b->kvb == 2u && !b->kvq8 && !b->kvq4 && !b->kvrot &&
                     b->sc_hist && b->k[K_INK_SCONV_HIST_R4] && b->k[K_INK_SCONV_COMMIT_R4];
    return on && b->has_cl && ink2 && b->at_part && b->at_count && !mk.on &&
           T >= 2u && T <= 8u && T <= b->vmax && pos + T <= b->g->n_ctx;
  }
  /* qwen35: the linear attention records and an fp16 or q8_0 KV cache; qwen3 (dense, heads of 128): ie_attn_long128 has
   * the tree path for every cache format (not the rotated one) */
  const int q35 = b->has_gdn1p && b->rec && b->ns == 1u && (b->kvb == 2u || b->kvq8) && !b->kvq4 && !b->kvrot && b->m->arch == ARCH_QWEN35;
  const int q3 = b->m->qk_norm && !b->m->n_rec && b->m->hd == 128u && !b->kvrot;
  const u32 sm = (pos + T - 1u) / IE_ATL_CH + 1u;
  const size_t pst = (size_t)b->m->n_head * (sm + (sm + IE_ATL_G - 1u) / IE_ATL_G) * IE_ATL_PSZ;
  const int fit = T <= b->at_ltmax || (b->at_lpack && pst * T <= (size_t)b->at_lstride * b->at_ltmax);
  return on && b->has_cl && (q35 || q3) && b->attn_long && b->at_part && !mk.on && T >= 2u && T <= 16u && fit &&
         (q3 || T <= b->vmax) && T <= b->at_tmax && pos + T <= b->g->n_ctx;
}
enum { TR_PIN_RING = 16, CL_PIN_RING = 16 };
static void tree_alloc(gpu_backend *b) {
  if (b->tr_dev) return;
  const model *m = b->m;
  const graph *g = b->g;
  const u32 h2 = m->n_rot / 2u;
  HIP(H.Malloc((void **)&b->tr_dev, 512));
  if (h2 > 0u) HIP(H.Malloc((void **)&b->tr_rope, (size_t)32u * h2 * 4u));
  HIP(H.Malloc((void **)&b->cl_dev, 65536));
  size_t row = 2u * (size_t)m->n_kvl * b->kvrowb + (size_t)m->n_rec * m->conv_dim * 4u + 2048u;
  if (g->h_out >= 0) row += g->bufs[g->h_out].stride + 256u;
  if (g->df_feat >= 0) row += g->bufs[g->df_feat].stride + 256u;
  if (g->ds_feat >= 0) row += g->bufs[g->ds_feat].stride + 256u;
  b->cl_scr_bytes = 16u * (row + 256u * (2u * m->n_kvl + m->n_rec + 4u));
  HIP(H.Malloc((void **)&b->cl_scr, b->cl_scr_bytes));
  if (H.HostMalloc((void **)&b->tr_pin, (size_t)TR_PIN_RING * 512u, 0) != 0) b->tr_pin = NULL;
  if (H.HostMalloc((void **)&b->cl_pin, (size_t)CL_PIN_RING * 65536u, 0) != 0) b->cl_pin = NULL;
  b->cl_host = malloc((size_t)16u * (2u * m->n_layer + 4u) * 3u * 8u);
  if (!b->cl_host) ie_die("out of memory");
}
static void tr_tab_upload(gpu_backend *b) {
  if (b->tr_pin && async_copy_on()) {
    unsigned char *p = b->tr_pin + (size_t)(b->tr_ring++ % TR_PIN_RING) * 512u;
    memcpy(p, b->tr_tab, 384);
    HIP(H.MemcpyAsync(b->tr_dev, p, 384, hipMemcpyHostToDevice, cur_stream));
  } else {
    HIP(H.Memcpy(b->tr_dev, b->tr_tab, 384, hipMemcpyHostToDevice));
  }
}
/* n copies (src, dst, bytes) in two launches through the scratch (sources and destinations may overlap) */
static void copy_list(gpu_backend *b, const unsigned long long *e, u32 n) {
  if (!n) return;
  if (!b->cl_dev) tree_alloc(b);
  const size_t bytes = (size_t)n * 2u * 24u;
  if (bytes > 65536u) ie_die("copy list: %u entries (%zu bytes > 65536)", n, bytes);
  const int async = b->cl_pin && async_copy_on();
  unsigned long long *l = async ? (unsigned long long *)((char *)b->cl_pin + (size_t)(b->cl_ring++ % CL_PIN_RING) * 65536u)
                                : (unsigned long long *)malloc(bytes);
  if (!l) ie_die("out of memory");
  size_t off = 0;
  for (u32 i = 0; i < n; i++) {
    l[3 * i] = e[3 * i], l[3 * i + 1] = (unsigned long long)(size_t)(b->cl_scr + off), l[3 * i + 2] = e[3 * i + 2];
    l[3 * (n + i)] = (unsigned long long)(size_t)(b->cl_scr + off), l[3 * (n + i) + 1] = e[3 * i + 1], l[3 * (n + i) + 2] = e[3 * i + 2];
    off += (size_t)((e[3 * i + 2] + 255u) & ~255ull);
  }
  if (off > b->cl_scr_bytes) ie_die("copy list: %u entries, %zu bytes", n, off);
  if (async) HIP(H.MemcpyAsync(b->cl_dev, l, bytes, hipMemcpyHostToDevice, cur_stream));
  else {
    HIP(H.Memcpy(b->cl_dev, l, bytes, hipMemcpyHostToDevice));
    free(l);
  }
  void *a1[] = {&b->cl_dev, &n};
  launch(b, K_COPY_LIST, n, a1);
  unsigned long long *l2 = b->cl_dev + 3u * n;
  void *a2[] = {&l2, &n};
  launch(b, K_COPY_LIST, n, a2);
}
/* before a tree run: the table, the nodes' RoPE rows */
static void tree_begin(gpu_backend *b, u32 pos, u32 T) {
  const model *m = b->m;
  const u32 h2 = m->n_rot / 2u;
  tree_alloc(b);
  tree_table(b->tr_tab, b->base.tree_par, T);
  if (m->arch == ARCH_INKLING2) {
    uint64_t tmask = 0ULL;
    u32 tdep = 0u, split = T;
    for (u32 t = 0; t < T; t++) {
      u32 row = 0u, a = t;
      for (;;) {
        row |= (1u << a);
        if (a == 0u) break;
        a = b->base.tree_par[a];
      }
      tmask |= ((uint64_t)row << (t * 8u));
      tdep |= ((u32)(b->tr_tab[TR_DEP + t] & 0xFu) << (t * 4u));
      if (t >= 2u && b->base.tree_par[t] == 0u && split == T) split = t;
    }
    b->ink_tree_mask = tmask;
    b->ink_tree_dep_pack = tdep;
    b->ink_tree_split = split;
    b->tr_pos = pos;
    return;
  }
  tr_tab_upload(b);
  unsigned long long e[32 * 3];
  for (u32 t = 0; t < T; t++) {
    const size_t p = (size_t)pos + b->tr_tab[TR_DEP + t];
    e[6 * t] = (unsigned long long)(size_t)(b->rcos + p * h2), e[6 * t + 1] = (unsigned long long)(size_t)(b->tr_rope + (size_t)t * h2);
    e[6 * t + 2] = h2 * 4u;
    e[6 * t + 3] = (unsigned long long)(size_t)(b->rsin + p * h2), e[6 * t + 4] = (unsigned long long)(size_t)(b->tr_rope + (16u + t) * (size_t)h2);
    e[6 * t + 5] = h2 * 4u;
  }
  copy_list(b, e, 2u * T);
  b->tr_pos = pos;
}
/* before a batched MTP frontier run (sec == 1 && mtr->on): node depths, KV slot paths, RoPE rows */
static void mtp_tree_begin(gpu_backend *b, const struct ie_mtp_tree *mtr, u32 pos, u32 T) {
  const model *m = b->m;
  const u32 h2 = m->n_rot / 2u;
  tree_alloc(b);
  memset(b->tr_tab, 0, 384);
  u32 kend = 1u;
  for (u32 t = 0; t < T; t++) {
    const u32 d = mtr->dep[t];
    b->tr_tab[TR_DEP + t] = (unsigned char)d;
    memcpy(b->tr_tab + TR_PATH + 16u * t, mtr->path[t], d + 1u);
    if (d + 1u > kend) kend = d + 1u;
  }
  b->tr_tab[IE_TR_KEND] = (unsigned char)kend;
  tr_tab_upload(b);
  unsigned long long e[32 * 3];
  for (u32 t = 0; t < T; t++) {
    const size_t p = (size_t)pos + mtr->dep[t];
    e[6 * t] = (unsigned long long)(size_t)(b->rcos + p * h2), e[6 * t + 1] = (unsigned long long)(size_t)(b->tr_rope + (size_t)t * h2);
    e[6 * t + 2] = h2 * 4u;
    e[6 * t + 3] = (unsigned long long)(size_t)(b->rsin + p * h2), e[6 * t + 4] = (unsigned long long)(size_t)(b->tr_rope + (16u + t) * (size_t)h2);
    e[6 * t + 5] = h2 * 4u;
  }
  copy_list(b, e, 2u * T);
  b->tr_pos = pos;
}
/* the accepted path into place: KV rows of the attention layers, conv ring rows, h_out and df_feat rows (row path[i]
 * to row i), then the linear attention state: the records of the path's nodes */
static void gpu_accept_tree(backend *bk, const uint8_t *path, uint32_t k) {
  tr_run("commit", tr_pos, k + 1u);
  gpu_backend *b = (gpu_backend *)bk;
  b->live_snap = -1;
  b->live_hbuf = -1;
  b->q16_ready_op = -1;
  const model *m = b->m;
  const graph *g = b->g;
  const u32 pos = b->tr_pos;
  if (!b->cl_host) tree_alloc(b);
  unsigned long long *e = b->cl_host;
  u32 n = 0;
#define CL(s, d, nb) (e[3 * n] = (unsigned long long)(size_t)(s), e[3 * n + 1] = (unsigned long long)(size_t)(d), e[3 * n + 2] = (nb), n++)
  for (u32 i = 1; i <= k; i++) {
    const u32 f = path[i];
    if (f == i) continue;
    for (u32 l = 0; l < m->n_layer; l++) {
      const layer *L = &m->l[l];
      if (L->rec) {
        float *r = b->ring + (size_t)L->sti * GDN_RING * m->conv_dim;
        CL(r + (size_t)((pos + f) & (GDN_RING - 1u)) * m->conv_dim, r + (size_t)((pos + i) & (GDN_RING - 1u)) * m->conv_dim,
           (unsigned long long)m->conv_dim * 4u);
      } else {
        const u32 kl = g->kv_len[L->kvi];
        for (int v = 0; v < 2; v++) CL(kv_ptr(b, v, L->kvi, (pos + f) % kl), kv_ptr(b, v, L->kvi, (pos + i) % kl), b->kvrowb);
      }
    }
    const int rb[3] = {g->h_out, g->df_feat, g->ds_feat};
    for (u32 q = 0; q < 3u; q++)
      if (rb[q] >= 0) {
        const buf *B = &g->bufs[rb[q]];
        char *base = (char *)b->arena + B->off;
        CL(base + (size_t)buf_row(B, f) * B->stride, base + (size_t)buf_row(B, i) * B->stride, B->stride);
      }
  }
#undef CL
  copy_list(b, e, n);
  if (b->sc_ver_T) {
    /* R32 Inkling-2 tree sconv commit: compact accepted path rows path[1..k] -> 1..k in sc_hist (path[j] >= j always). */
    for (u32 j = 1u; j <= k; j++) {
      if (path[j] != j)
        HIP(H.MemcpyAsync(b->sc_hist + (size_t)j * b->sc_hist_stride,
                          b->sc_hist + (size_t)path[j] * b->sc_hist_stride,
                          (size_t)b->m->sc_tot * 4u, hipMemcpyDeviceToDevice, cur_stream));
    }
    u32 tot = b->m->sc_tot, pos0 = b->sc_ver_pos, n_acc = k + 1u, hrs = b->sc_hist_stride;
    void *ca[] = {&b->sc_ring, &b->sc_hist, &tot, &pos0, &n_acc, &hrs};
    launch(b, K_INK_SCONV_COMMIT_R4, (tot + 255u) / 256u, ca);
    b->sc_ver_T = 0;
  }
  /* the state: records of the path's nodes onto the slot (the state after node 0) */
  if (b->rec_n && k) {
    uint64_t sls = (uint64_t)m->n_vh * m->sd * m->sd, rls = (uint64_t)(b->vmax - 1u) * m->n_vh * GDN_RECF;
    u32 nv = m->n_vh, mask = 0;
    for (u32 i = 1; i <= k; i++) {
      if (path[i] > b->rec_n || (i > 1u && path[i] <= path[i - 1u])) ie_die("accept_tree: path node %u", path[i]);
      mask |= 1u << path[i];
    }
    gpu_flush_gdn_replay(b);
    if (gdn_eager_replay() || !b->rec_buf[1]) {
      void *args[] = {&b->st, &b->rec, &nv, &mask, &sls, &rls};
      launch(b, K_GDN_REPLAY, m->n_rec * nv, args);
    } else {
      b->lazy_rec = b->rec;
      b->lazy_mask = mask;
      b->lazy_rls = rls;
      b->rec_idx ^= 1;
      b->rec = b->rec_buf[b->rec_idx];
    }
  }
  b->rec_n = 0;
  b->base.tree_par = NULL;
}

static void gpu_read_rows(backend *bk, int id, uint32_t r0, uint32_t n, float *dst) {
  gpu_backend *b = (gpu_backend *)bk;
  const buf *B = &b->g->bufs[id];
  HIP(H.Memcpy(dst, b->arena + B->off + (size_t)buf_row(B, r0) * B->stride, (size_t)n * B->stride, hipMemcpyDeviceToHost));
}

/* top K on the GPU (ie_topk_part, ie_topk_merge): see ie_kernels.c */
enum { TK_MAX = 64, TK_CS = 4096, TK_ROWS = 16, TK_NG = 64 };
static int topk_fits(uint32_t n, uint32_t nv, uint32_t K) {
  return K != 0u && K <= TK_MAX && n <= TK_ROWS && (nv + TK_CS - 1u) / TK_CS <= TK_NG;
}
/* IE_TOPK_FAST=0: always the K-pass kernels (ie_topk_part, ie_topk_merge); else ie_topk_max + ie_topk_sel when K is at
 * most the number of segments (same result, see ie_kernels.c). IE_TOPK_CHECK=1: every call is also done the slow way
 * and compared; a difference ends the run. */
static int topk_fast_on(const gpu_backend *b, uint32_t nv, uint32_t K) {
  static int on = -1;
  if (on < 0) on = !(getenv("IE_TOPK_FAST") && getenv("IE_TOPK_FAST")[0] == '0');
  return on && b->has_topk_fast && K <= (nv + TK_CS - 1u) / TK_CS;
}
enum { TF_CAP = 1024 }; /* candidates per row (ie_kernels.c TF_CAP) */
/* The launches of the top K of rows r0 .. r0 + n - 1 of buffer id; the result stays on the device. Returns it
 * (TK_ROWS * TK_MAX values, then as many indices; row r at r * K), or NULL if the sizes do not fit. */
static float *topk_launch(gpu_backend *b, int id, uint32_t r0, uint32_t n, uint32_t nv, uint32_t K, int fast, int need_lse) {
  const u32 ng = (nv + TK_CS - 1u) / TK_CS;
  if (!topk_fits(n, nv, K)) return NULL;
  const buf *B = &b->g->bufs[id];
  const size_t tk_base_f = (size_t)TK_ROWS * (TK_NG + 1u) * TK_MAX * 2u; /* 133,120 floats */
  if (!b->tk_buf) {
    HIP(H.Malloc((void **)&b->tk_buf, (tk_base_f + 2064u + TK_ROWS) * 4u));
    HIP(H.Memset(b->tk_buf + tk_base_f + 2064u, 0, (size_t)TK_ROWS * 4u));
  }
  if (!b->tk_pin) HIP(H.HostMalloc((void **)&b->tk_pin, (size_t)TK_ROWS * TK_MAX * 2u * 4u, 0));
  float *cv = b->tk_buf, *ov = cv + (size_t)TK_ROWS * TK_NG * TK_MAX * 2u;
  u32 *ci = (u32 *)(cv + (size_t)TK_ROWS * TK_NG * TK_MAX), *oi = (u32 *)(ov + (size_t)TK_ROWS * TK_MAX);
  void *x = b->arena + B->off + (size_t)buf_row(B, r0) * B->stride;
  u32 xs = B->stride / 4u, cs = (nv + ng - 1u) / ng, nc = ng * K;
  tr_run("sample", tr_pos, n);
  if (fast) { /* candidate lists in cv / ci (TF_CAP per row), then the segment maxima and the counters */
    if ((b->k[K_INK_TOPK_BATCHED_R7] || b->k[K_SALUKI_TOPK_R11]) && !mk.rec && nv > 0u && n > 0u) {
      u32 lse_flag = need_lse ? 1u : 0u;
      float *smax_r11 = b->tk_buf + tk_base_f;
      u32 *done_r11 = (u32 *)(smax_r11 + 2064u);
      if (b->k[K_INK_TOPK_BATCHED_R7]) {
        float cap0 = 0.0f;
        void *a[] = {&x, &nv, &xs, &K, &cs, (void *)&ng, &cap0, &lse_flag, &smax_r11, &cv, &ci, &done_r11, &ov, &oi};
        launch_t(b, K_INK_TOPK_BATCHED_R7, ng, n, a);
      } else {
        void *a[] = {&x, &nv, &xs, &K, &cs, (void *)&ng, &lse_flag, &smax_r11, &cv, &ci, &done_r11, &ov, &oi};
        launch_t(b, K_SALUKI_TOPK_R11, ng, n, a);
      }
      return ov;
    }
    float *smax = cv + (size_t)TK_ROWS * TF_CAP;
    u32 *cnt = ci + (size_t)TK_ROWS * TF_CAP, *fin = cnt + TK_ROWS;
    void *a1[] = {&x, &nv, &xs, &cs, &smax, &cnt, &fin};
    launch_t(b, K_TOPK_MAX, ng, n, a1);
    void *a2[] = {&x, &nv, &xs, &K, &cs, (void *)&ng, &smax, &cv, &ci, &cnt, &fin, &ov, &oi};
    launch_t(b, K_TOPK_SEL, ng, n, a2);
    return ov;
  }
  void *a1[] = {&x, &nv, &xs, &K, &cs, (void *)&ng, &cv, &ci};
  launch_t(b, K_TOPK_PART, ng, n, a1);
  void *a2[] = {&cv, &ci, &nc, &K, &ov, &oi};
  launch(b, K_TOPK_MERGE, n, a2);
  return ov;
}
static unsigned tk_calls, tk_slow;
static void topk_check_report(void) {
  fprintf(stderr, "top K check: %u calls, %u of them needed the slow kernels (too many candidates), no difference\n", tk_calls, tk_slow);
}
/* The top K of n rows, in res (values, then indices: TK_ROWS * TK_MAX * 2 floats); 0 if the sizes do not fit. */
static int topk_fetch_ex(gpu_backend *b, int id, uint32_t r0, uint32_t n, uint32_t nv, uint32_t K, float *res, float *lse) {
  const int fast = topk_fast_on(b, nv, K);
  float *ov = topk_launch(b, id, r0, n, nv, K, fast, lse != NULL);
  if (!ov) return 0;
  const size_t bytes = (size_t)TK_ROWS * TK_MAX * 2u * 4u;
  if (b->tk_pin) {
    HIP(H.MemcpyAsync(b->tk_pin, ov, bytes, hipMemcpyDeviceToHost, cur_stream));
    HIP(H.StreamSynchronize(cur_stream));
    memcpy(res, b->tk_pin, bytes);
  } else {
    HIP(H.Memcpy(res, ov, bytes, hipMemcpyDeviceToHost));
  }
  int again = 0; /* a row with more than TF_CAP candidates: the slow kernels for the whole call */
  if (fast)
    for (uint32_t r = 0; r < n; r++) again |= ((const u32 *)(res + TK_ROWS * TK_MAX))[(size_t)r * K] == 0xFFFFFFFFu;
  if (lse) {
    if (fast && !again) {
      const float *lse_src = ((b->k[K_INK_TOPK_BATCHED_R7] || b->k[K_SALUKI_TOPK_R11]) && !mk.rec && nv > 0u && n > 0u)
          ? (b->tk_buf + (size_t)TK_ROWS * (TK_NG + 1u) * TK_MAX * 2u + 2048u)
          : (b->tk_buf + (size_t)TK_ROWS * TF_CAP + 2048u);
      HIP(H.Memcpy(lse, lse_src, (size_t)n * 4u, hipMemcpyDeviceToHost));
    } else {
      for (uint32_t r = 0; r < n; r++) lse[r] = NAN;
    }
  }
  if (again) {
    ov = topk_launch(b, id, r0, n, nv, K, 0, 0);
    if (b->tk_pin) {
      HIP(H.MemcpyAsync(b->tk_pin, ov, bytes, hipMemcpyDeviceToHost, cur_stream));
      HIP(H.StreamSynchronize(cur_stream));
      memcpy(res, b->tk_pin, bytes);
    } else {
      HIP(H.Memcpy(res, ov, bytes, hipMemcpyDeviceToHost));
    }
  }
  static int chk = -1;
  if (chk < 0) {
    chk = getenv("IE_TOPK_CHECK") != NULL && getenv("IE_TOPK_CHECK")[0] == '1';
    if (chk) atexit(topk_check_report);
  }
  if (chk) {
    static float slow[TK_ROWS * TK_MAX * 2];
    tk_calls++, tk_slow += (unsigned)again;
    ov = topk_launch(b, id, r0, n, nv, K, 0, 0);
    HIP(H.Memcpy(slow, ov, bytes, hipMemcpyDeviceToHost));
    if (memcmp(slow, res, (size_t)n * K * 4u) || memcmp(slow + TK_ROWS * TK_MAX, res + TK_ROWS * TK_MAX, (size_t)n * K * 4u))
      ie_die("top K check: the fast and the slow kernels differ (rows %u, K %u)", n, K);
  }
  return 1;
}
static int topk_fetch(gpu_backend *b, int id, uint32_t r0, uint32_t n, uint32_t nv, uint32_t K, float *res) {
  return topk_fetch_ex(b, id, r0, n, nv, K, res, NULL);
}
/* The top K that a verify run took right behind the model's output matrix (gpu_run, backend.h sample_k), and what it is of. */
static struct {
  int valid;
  uint32_t id, n, nv, K;
  float res[TK_ROWS * TK_MAX * 2]; /* values, then indices */
} tk_pre;
static void topk_prefetch_clear(void) { tk_pre.valid = 0; }
/* Launch the top K of the T rows of the logits of the run just launched and wait for it: the one wait of a run that
 * does not read the argmax. */
static void topk_prefetch(gpu_backend *b, uint32_t T) {
  const uint32_t id = b->g->logits, nv = b->m->vocab_unpadded ? b->m->vocab_unpadded : b->m->vocab, K = b->base.sample_k;
  if (!topk_fetch(b, (int)id, 0, T, nv, K, tk_pre.res)) ie_die("top K of the verify run: sizes do not fit");
  tk_pre.id = id, tk_pre.n = T, tk_pre.nv = nv, tk_pre.K = K, tk_pre.valid = 1;
}
/* R7 bit 84: persistent 1-launch fused Q4_K output head + softcap + Argmax/Top-K (0 g->logits traffic). */
static void ink_fused_q4k_head_run(gpu_backend *b, uint32_t T, int fold, uint32_t *out) {
  const graph *g = b->g;
  const u32 ih = g->i_head;
  const op *o0 = &g->ops[ih], *o1 = &g->ops[ih + 1u];
  const size_t tk_base_f = (size_t)TK_ROWS * (TK_NG + 1u) * TK_MAX * 2u;
  if (!b->tk_buf) {
    HIP(H.Malloc((void **)&b->tk_buf, (tk_base_f + 2064u + TK_ROWS) * 4u));
    HIP(H.Memset(b->tk_buf + tk_base_f + 2064u, 0, (size_t)TK_ROWS * 4u));
  }
  if (!b->tk_pin) HIP(H.HostMalloc((void **)&b->tk_pin, (size_t)TK_ROWS * TK_MAX * 2u * 4u, 0));
  float *part_v = b->tk_buf;
  u32 *part_i = (u32 *)(b->tk_buf + 32768u);
  float *ov_d = b->tk_buf + (size_t)TK_ROWS * TK_NG * TK_MAX * 2u;
  u32 *oi_d = (u32 *)(ov_d + (size_t)TK_ROWS * TK_MAX);
  void *qw = b->d[ih].w0, *qs = b->d[ih].w1, *bias = b->d[ih].bias;
  void *xq = b->arena + g->bufs[o0->a].off + o0->aoff;
  u32 rows = o0->w->rows, unpad = o1->nh, nb = o0->w->nb, TT = T, xs = g->bufs[o0->a].stride;
  float cap = o1->clamp;
  u32 K = fold ? b->base.sample_k : 1u, ng = 128u;
  const buf *amb = &g->bufs[g->argmax];
  void *out_am = fold ? NULL : (b->arena + amb->off);
  u32 am_stride = fold ? 0u : amb->stride;
  void *ov_p = fold ? (void *)ov_d : NULL, *oi_p = fold ? (void *)oi_d : NULL;
  void *fa[] = {&qw, &qs, &xq, &bias, &rows, &unpad, &nb, &TT, &xs, &cap, &K, &ng,
                &part_v, &part_i, &b->am_count, &out_am, &am_stride, &ov_p, &oi_p};
  launch(b, K_INK_Q4K_HEAD_CAP_TOPK_R7, ng, fa);
  if (fold) {
    const size_t bytes = (size_t)TK_ROWS * TK_MAX * 2u * 4u;
    HIP(H.MemcpyAsync(b->tk_pin, ov_d, bytes, hipMemcpyDeviceToHost, cur_stream));
    HIP(H.StreamSynchronize(cur_stream));
    memcpy(tk_pre.res, b->tk_pin, bytes);
    const u32 nv = b->m->vocab_unpadded ? b->m->vocab_unpadded : b->m->vocab;
    tk_pre.id = g->logits, tk_pre.n = T, tk_pre.nv = nv, tk_pre.K = K, tk_pre.valid = 1;
    if (out) for (u32 t = 0; t < T; t++) out[t] = 0u;
  } else {
    const size_t am_bytes = (size_t)T * amb->stride;
    if (b->am_pin && am_bytes <= 256u) {
      HIP(H.MemcpyAsync(b->am_pin, out_am, am_bytes, hipMemcpyDeviceToHost, cur_stream));
      HIP(H.StreamSynchronize(cur_stream));
      for (u32 t = 0; t < T; t++) out[t] = b->am_pin[t * amb->stride / 4u];
    } else {
      u32 *tmp = malloc(am_bytes);
      if (!tmp) ie_die("out of memory");
      HIP(H.Memcpy(tmp, out_am, am_bytes, hipMemcpyDeviceToHost));
      for (u32 t = 0; t < T; t++) out[t] = tmp[t * amb->stride / 4u];
      free(tmp);
    }
  }
}
static int gpu_topk_lse(backend *bk, int id, uint32_t r0, uint32_t n, uint32_t nv, uint32_t K, float *val, uint32_t *idx, float *lse) {
  gpu_backend *b = (gpu_backend *)bk;
  static float tmp[TK_ROWS * TK_MAX * 2]; /* values, then indices: one copy */
  const float *res = tmp;
  if (!lse && tk_pre.valid && (uint32_t)id == tk_pre.id && r0 == 0 && n == tk_pre.n && nv == tk_pre.nv && K == tk_pre.K) {
    tk_pre.valid = 0; /* already taken by the run */
    res = tk_pre.res;
  } else if (!topk_fetch_ex(b, id, r0, n, nv, K, tmp, lse)) {
    return 0;
  }
  memcpy(val, res, (size_t)n * K * 4u);
  memcpy(idx, res + TK_ROWS * TK_MAX, (size_t)n * K * 4u);
  return 1;
}
static int gpu_topk(backend *bk, int id, uint32_t r0, uint32_t n, uint32_t nv, uint32_t K, float *val, uint32_t *idx) {
  return gpu_topk_lse(bk, id, r0, n, nv, K, val, idx, NULL);
}

static void gpu_copy_rows(backend *bk, int dst, uint32_t d0, int src, uint32_t r0, uint32_t n) {
  gpu_backend *b = (gpu_backend *)bk;
  if (dst == b->live_hbuf) b->live_hbuf = -1;
  const graph *g = b->g;
  const buf *D = &g->bufs[dst];
  char *dp = b->arena + D->off + (size_t)d0 * D->stride;
  if (src < 0) {
    HIP(H.Memset(dp, 0, (size_t)n * D->stride));
    return;
  }
  const buf *S = &g->bufs[src];
  if (S->stride != D->stride) ie_die("copy_rows: strides differ");
  /* a device-to-device copy in the launch stream: no wait for the host (the blocking one drains the GPU first and
   * wakes the host before the next launches can begin); IE_ASYNC_COPY=0: the blocking copy */
  static int async = -1;
  if (async < 0) async = !(getenv("IE_ASYNC_COPY") && getenv("IE_ASYNC_COPY")[0] == '0');
  void *sp = b->arena + S->off + (size_t)r0 * S->stride;
  if (async) HIP(H.MemcpyAsync(dp, sp, (size_t)n * D->stride, hipMemcpyDeviceToDevice, cur_stream));
  else HIP(H.Memcpy(dp, sp, (size_t)n * D->stride, hipMemcpyDeviceToDevice));
}

/* The kernel of op o (as gpu_step launches it). */
static int op_kernel(const gpu_backend *b, const op *o) {
  const model *m = b->m;
  switch (o->kind) {
    case OP_EMBED: {
      const mat *tm = o->w ? o->w : &m->tok;
      return tm->kind == MAT_Q4 ? K_EMBED_Q4 : tm->kind == MAT_Q8 ? K_EMBED_Q8 : K_EMBED_F32;
    }
    case OP_QKN_ROPE_KV: return K_QKN_ROPE_KV;
    case OP_GDN: return K_GDN;
    case OP_RMSNORM: return K_RMSNORM;
    case OP_QUANT: return K_QUANT;
    case OP_GEMV_Q4: return K_GEMV_Q4;
    case OP_GEMV_IQ4NL: return K_GEMV_IQ4NL;
    case OP_GEMV_Q4K: return K_GEMV_Q4K;
    case OP_GEMV_Q6K: return K_GEMV_Q6K;
    case OP_GEMV_Q8: return K_GEMV_Q8;
    case OP_GEMV_F32: return K_GEMV_F32;
    case OP_BIAS: return K_BIAS;
    case OP_ADD: return K_ADD;
    case OP_SWIGLU: return K_SWIGLU;
    case OP_VRES: return K_VRES;
    case OP_GATE: return K_GATE;
    case OP_ROPE: return K_ROPE;
    case OP_KV: return K_KV;
    case OP_ROPE_KV: return K_ROPE_KV;
    case OP_ATTN: return (m->arch == ARCH_FLINT || m->arch == ARCH_RICURSIVE) ? (b->attn_gq && b->gq_min <= 1u ? K_ATTN_GQ : K_ATTN_SW) : b->attn_long ? K_ATTN_LONG : b->attn_kv ? K_ATTN_KV : b->attn_split ? K_ATTN_SPLIT : K_ATTN;
    case OP_GEMV_Q5K: return K_GEMV_Q5K;
    case OP_GEMV_IQ4XS: return K_GEMV_IQ4XS;
    case OP_MOE_ROUTE: return K_MOE_ROUTE;
    case OP_MOE_SUM: return K_MOE_SUM;
    case OP_MOE_SWIGLU: return K_SWIGLU;
    case OP_GEMV_EXP: return o->w->kind == MAT_RAWQ ? K_GEXP_RAW : K_GEXP;
    case OP_GEMV_RAW: return raw_kernel(o->w->sub);
    case OP_VCLIP: return K_VCLIP;
    case OP_ATTN_RES: return K_ATTN_RES;
    case OP_LOOP_GATE: return K_LOOP_GATE;
    case OP_SCONV: return K_SCONV;
    case OP_REL_ATTN: return K_REL_ATTN;
    case OP_SITU: return K_SITU;
    case OP_LOGIT_CAP: return K_LOGIT_CAP;
    default: return K_ARGMAX;
  }
}

/* Weight bytes that op o reads (0 if it reads no weight matrix). */
static double op_wbytes(const op *o) {
  if (o->kind == OP_GEMV_RAW) return (double)o->w->rows * (o->w->nb / 8u) * ie_rawq_bs(o->w->sub);
  if (o->kind == OP_GEMV_Q4 || o->kind == OP_GEMV_Q4K || o->kind == OP_GEMV_IQ4NL) return (double)o->w->rows * o->w->nb * 18.0;
  if (o->kind == OP_GEMV_Q6K) return (double)o->w->rows * o->w->nb * 26.25;
  if (o->kind == OP_GEMV_Q8) return (double)o->w->rows * o->w->nb * 34.0;
  if (o->kind == OP_GEMV_F32) return (double)o->w->rows * o->w->cols * 4.0;
  if (o->kind == OP_GEMV_Q5K) return (double)o->w->rows * o->w->nb * 22.0;
  if (o->kind == OP_GEMV_IQ4XS) return (double)o->w->rows * o->w->nb * 17.0;
  if (o->kind == OP_GEMV_EXP) {
    const int k = o->w->kind;
    if (k == MAT_RAWQ) return (double)o->w->rows * (o->w->nb / 8u) * ie_rawq_bs(o->w->sub) * o->nh;
    const double per = k == MAT_Q8 ? 34.0 : k == MAT_Q5K ? 22.0 : k == MAT_IQ4XS ? 17.0 : k == MAT_Q6K ? 26.25 : 18.0;
    return (double)o->w->rows * o->w->nb * per * o->nh;
  }
  return 0;
}

/* Per kernel and matrix shape: ops per token, ms per token, share, GB/s. */
static void prof_report(gpu_backend *b) {
  const graph *g = b->g;
  if (!b->psteps) return;
  enum { MAXR = 64 };
  struct { int k; u32 rows, cols; double ms, bytes; u32 cnt; } r[MAXR];
  int nr = 0;
  double total = 0;
  for (uint32_t i = 0; i < g->n_main; i++) {
    const op *o = &g->ops[i];
    int k = op_kernel(b, o);
    u32 rows = o->w && (o->kind == OP_GEMV_Q4 || o->kind == OP_GEMV_Q4K || o->kind == OP_GEMV_Q6K || o->kind == OP_GEMV_Q8 || o->kind == OP_GEMV_F32) ? o->w->rows : 0;
    u32 cols = rows ? o->w->cols : 0;
    int j = 0;
    while (j < nr && !(r[j].k == k && r[j].rows == rows && r[j].cols == cols)) j++;
    if (j == nr) {
      if (nr == MAXR) continue;
      r[nr].k = k, r[nr].rows = rows, r[nr].cols = cols, r[nr].ms = 0, r[nr].bytes = 0, r[nr].cnt = 0;
      nr++;
    }
    r[j].ms += b->pms[i] / b->psteps;
    r[j].bytes += op_wbytes(o);
    r[j].cnt++;
    total += b->pms[i] / b->psteps;
  }
  fprintf(stderr, "profile: %u steps, %.3f ms/token in the ops\n", b->psteps, total);
  fprintf(stderr, "  %-14s %13s %6s %9s %6s %10s %8s\n", "kernel", "rows x cols", "ops", "ms/token", "share",
          "MB/token", "GB/s");
  for (int pass = 0; pass < nr; pass++) { /* largest time first */
    int best = -1;
    for (int j = 0; j < nr; j++)
      if (r[j].cnt && (best < 0 || r[j].ms > r[best].ms)) best = j;
    if (best < 0) break;
    char shape[32] = "-";
    if (r[best].rows) snprintf(shape, sizeof shape, "%ux%u", r[best].rows, r[best].cols);
    fprintf(stderr, "  %-14s %13s %6u %9.4f %5.1f%%", kname[r[best].k], shape, r[best].cnt, r[best].ms,
            100.0 * r[best].ms / total);
    if (r[best].bytes > 0)
      fprintf(stderr, " %10.2f %8.1f\n", r[best].bytes / 1e6, r[best].bytes / (r[best].ms * 1e6));
    else
      fprintf(stderr, " %10s %8s\n", "-", "-");
    r[best].cnt = 0;
  }
}

/* Per layer: ms per token, ops, weight MB, GB/s, and the split into the
 * attention half (up to the first residual add) and the FFN half. Ops
 * outside the layers (embedding, output norm, output matrix, argmax) are
 * one row each. IE_PROFILE_CSV=FILE also writes every op as CSV. */
static void prof_layers(gpu_backend *b) {
  const graph *g = b->g;
  const model *m = b->m;
  const uint32_t nl = m->n_layer;
  double *ms = calloc(nl, sizeof(double)), *att = calloc(nl, sizeof(double)), *gemv = calloc(nl, sizeof(double));
  double *by = calloc(nl, sizeof(double));
  uint32_t *cnt = calloc(nl, sizeof(uint32_t));
  int *half = calloc(nl, sizeof(int));
  if (!ms || !att || !gemv || !by || !cnt || !half) ie_die("out of memory");
  double total = 0, in_layers = 0;
  for (uint32_t i = 0; i < g->n_main; i++) total += b->pms[i] / b->psteps;
  fprintf(stderr, "profile by layer (ms/token):\n");
  fprintf(stderr, "  %-7s %9s %9s %9s %9s %6s %6s %9s %7s\n", "layer", "total", "attn", "ffn", "gemv", "share", "ops",
          "MB", "GB/s");
  for (uint32_t i = 0; i < g->n_main; i++) {
    const op *o = &g->ops[i];
    double t = b->pms[i] / b->psteps;
    if (o->layer < 0) {
      double wb = op_wbytes(o);
      fprintf(stderr, "  %-7s %9.4f %9s %9s %9s %5.1f%% %6u", kname[op_kernel(b, o)] + 3, t, "-", "-", "-",
              100.0 * t / total, 1u);
      if (wb > 0) fprintf(stderr, " %9.2f %7.1f\n", wb / 1e6, wb / (t * 1e6));
      else fprintf(stderr, " %9s %7s\n", "-", "-");
      continue;
    }
    uint32_t l = (uint32_t)o->layer;
    ms[l] += t, cnt[l]++, by[l] += op_wbytes(o);
    if (o->kind == OP_GEMV_Q4 || o->kind == OP_GEMV_Q4K || o->kind == OP_GEMV_Q6K || o->kind == OP_GEMV_Q8 || o->kind == OP_GEMV_F32) gemv[l] += t;
    if (!half[l]) att[l] += t;
    if (o->kind == OP_ADD) half[l] = 1; /* the first add ends the attention half */
    in_layers += t;
  }
  for (uint32_t l = 0; l < nl; l++)
    fprintf(stderr, "  %-7u %9.4f %9.4f %9.4f %9.4f %5.1f%% %6u %9.2f %7.1f\n", l, ms[l], att[l], ms[l] - att[l],
            gemv[l], 100.0 * ms[l] / total, cnt[l], by[l] / 1e6, ms[l] > 0 ? by[l] / (ms[l] * 1e6) : 0.0);
  double sa = 0, sg = 0, sb = 0;
  uint32_t sc = 0;
  for (uint32_t l = 0; l < nl; l++) sa += att[l], sg += gemv[l], sb += by[l], sc += cnt[l];
  fprintf(stderr, "  %-7s %9.4f %9.4f %9.4f %9.4f %5.1f%% %6u %9.2f %7.1f\n", "layers", in_layers, sa, in_layers - sa,
          sg, 100.0 * in_layers / total, sc, sb / 1e6, sb / (in_layers * 1e6));
  fprintf(stderr, "  %-7s %9.4f\n", "all", total);

  const char *csv = getenv("IE_PROFILE_CSV");
  if (csv && csv[0]) {
    FILE *f = fopen(csv, "w");
    if (!f) ie_die("cannot write %s", csv);
    fprintf(f, "op,layer,kernel,rows,cols,ms_per_token,weight_bytes\n");
    for (uint32_t i = 0; i < g->n_main; i++) {
      const op *o = &g->ops[i];
      int mat = o->kind == OP_GEMV_Q4 || o->kind == OP_GEMV_Q4K || o->kind == OP_GEMV_Q6K || o->kind == OP_GEMV_Q8 || o->kind == OP_GEMV_F32 ||
                o->kind == OP_GEMV_RAW;
      fprintf(f, "%u,%d,%s,%u,%u,%.6f,%.0f\n", i, o->layer, kname[op_kernel(b, o)], mat ? o->w->rows : 0,
              mat ? o->w->cols : 0, b->pms[i] / b->psteps, op_wbytes(o));
    }
    fclose(f);
    fprintf(stderr, "profile: wrote %u ops to %s\n", g->n_main, csv);
  }
  free(ms), free(att), free(gemv), free(by), free(cnt), free(half);
}

static void gpu_close(backend *bk) {
  vram_trace("close", 0);
  gpu_backend *b = (gpu_backend *)bk;
  if (b->host_n && b->host_launches)
    fprintf(stderr, "gpu: host issue time %.3f ms/token (%.0f launches per token for %u ops)\n", b->host_ms / b->host_n,
            (double)b->host_launches / b->host_n, b->g->n_ops);
  else if (b->host_n)
    fprintf(stderr, "gpu: host issue time %.3f ms/token (%u ops per token)\n", b->host_ms / b->host_n, b->g->n_ops);
  if (b->attn_check) {
    fprintf(stderr, "attn check: %u calls, max |split - old| / max |old| = %.2e\n", b->chk_n, b->chk_worst);
    H.Free(b->chk), free(b->chk_a), free(b->chk_b);
  }
  if (b->prof == 3 && b->psteps) { /* the MTP draft ops: ms per draft run, by kernel and (IE_PROFILE_CSV) per op */
    const graph *g = b->g;
    double tot = 0;
    for (uint32_t i = g->n_main; i < g->n_mtp; i++) tot += b->pms[i] / b->psteps;
    fprintf(stderr, "profile of the MTP draft runs (%u runs): %.4f ms per run, %u ops\n", b->psteps, tot, g->n_mtp - g->n_main);
    const char *csv = getenv("IE_PROFILE_CSV");
    FILE *f = csv && csv[0] ? fopen(csv, "w") : NULL;
    if (f) fprintf(f, "op,layer,kernel,rows,cols,ms_per_token,weight_bytes\n");
    for (uint32_t i = g->n_main; i < g->n_mtp; i++) {
      const op *o = &g->ops[i];
      const int mat = o->kind == OP_GEMV_Q4 || o->kind == OP_GEMV_Q4K || o->kind == OP_GEMV_Q6K || o->kind == OP_GEMV_Q8 ||
                      o->kind == OP_GEMV_F32 || o->kind == OP_GEMV_RAW || o->kind == OP_GEMV_IQ4XS || o->kind == OP_GEMV_Q5K;
      if (f)
        fprintf(f, "%u,%d,%s,%u,%u,%.6f,%.0f\n", i, o->layer, kname[op_kernel(b, o)], mat ? o->w->rows : 0,
                mat ? o->w->cols : 0, b->pms[i] / b->psteps, op_wbytes(o));
    }
    if (f) fclose(f), fprintf(stderr, "profile: wrote the MTP ops to %s\n", csv);
  } else if (b->prof && b->psteps) prof_report(b), prof_layers(b);
  if (mk.on && mk.psteps) {
    double tot = 0;
    for (int k = 0; k < K_N; k++) tot += mk.pus[k];
    fprintf(stderr, "megakernel: %.1f commands joined per launch\n", (double)mk.joined / mk.launches);
    fprintf(stderr, "megakernel profile: %u launches, %.3f ms per launch (commands 2 .. n-1, barrier included)\n",
            mk.psteps, tot / 1e3 / mk.psteps);
    for (int k = 0; k < K_N; k++)
      if (mk.pcnt[k])
        fprintf(stderr, "  %-18s %5.1f per launch %9.2f us each %8.3f ms per launch\n", kname[k],
                (double)mk.pcnt[k] / mk.psteps, mk.pus[k] / mk.pcnt[k], mk.pus[k] / 1e3 / mk.psteps);
  }
  HIP(H.DeviceSynchronize());
  gpu_trace_flush();
  if (b->saluki_mod) H.ModuleUnload(b->saluki_mod);
  if (b->inkling_mod) H.ModuleUnload(b->inkling_mod);
  for (u32 p = 0; p < 8u; p++) {
    if (b->ink_pgraph[p] && H.GraphExecDestroy) H.GraphExecDestroy(b->ink_pgraph[p]);
    if (b->ink_pev[p]) H.EventDestroy(b->ink_pev[p]);
  }
  if (b->ink_gstream && H.StreamDestroy) H.StreamDestroy(b->ink_gstream);
  if (b->ink_sp_dev) H.Free(b->ink_sp_dev);
  if (b->ink_sp_pin) H.HostFree(b->ink_sp_pin);
  for (u32 k = 0; k < 64u; k++) if (b->ink_sp_ev[k]) H.EventDestroy(b->ink_sp_ev[k]);
  if (b->ink_sp_sync) H.Free(b->ink_sp_sync);
  if (b->ink_ctx_dev) H.Free(b->ink_ctx_dev);
  if (b->ink_ctx_host) H.HostFree(b->ink_ctx_host);
  if (b->ink_toks_dev) H.Free(b->ink_toks_dev);
  if (b->ink_lc_count) H.Free(b->ink_lc_count);
  if (b->ms_packs) H.Free(b->ms_packs);
  if (b->ms_npacks) H.Free(b->ms_npacks);
  if (tr_f) fprintf(stderr, "trace: wrote %u launches in %u runs\n", tr_n, tr_call), fclose(tr_f), tr_f = NULL;
  if (tr_gpu) {
    fclose(tr_gpu), tr_gpu = NULL;
    for (unsigned i = 0; i < TR_EVENT_CAP; i++) {
      if (!tr_events[i].begin) continue;
      HIP(H.EventDestroy(tr_events[i].begin));
      HIP(H.EventDestroy(tr_events[i].end));
    }
    free(tr_events), tr_events = NULL;
  }
  slab_free(&b->us);
  slab_free(&sdev);
  n_sdev = 0, sdev_bytes = 0;
  for (uint32_t i = 0; i < n_shost; i++) H.HostFree(shost[i]);
  n_shost = 0, shost_bytes = 0;
  H.Free(b->arena), H.Free(b->kc), H.Free(b->vc), H.Free(b->ring), H.Free(b->sc_ring), H.Free(b->sc_hist), H.Free(b->st), H.Free(b->rec_buf[0]), H.Free(b->rec_buf[1]);
  if (b->ds.ready) {
    if (b->ds.kv_grow) {
      for (u32 a = 0; a < 16u; a++) {
        if (!b->ds.kv_vm[a >> 1]) continue;
        uint16_t **slot = a & 1u ? &b->ds.vc[a >> 1] : &b->ds.kc[a >> 1];
        for (size_t off = 0; off < b->ds.kv_mapped; off += b->ds.kv_chunk) {
          H.MemUnmap((char *)*slot + off, b->ds.kv_chunk);
          H.MemRelease(b->ds.kv_h[a * b->ds.kv_nchunk + off / b->ds.kv_chunk]);
        }
        H.MemAddressFree(*slot, b->ds.kv_va), *slot = NULL;
      }
      free(b->ds.kv_h), b->ds.kv_h = NULL;
    }
    for (u32 il = 0; il < 8u; il++) H.Free(b->ds.kc[il]), H.Free(b->ds.vc[il]);
    H.Free(b->ds.ctx_q8), H.Free(b->ds.ctx_kv), H.Free(b->ds.frnn);
    H.Free(b->ds.h), H.Free(b->ds.xn), H.Free(b->ds.xq), H.Free(b->ds.q), H.Free(b->ds.k), H.Free(b->ds.v);
    H.Free(b->ds.att), H.Free(b->ds.aq), H.Free(b->ds.gu), H.Free(b->ds.gq), H.Free(b->ds.hidden);
    H.Free(b->ds.state), H.Free(b->ds.cat_q8), H.Free(b->ds.raw768), H.Free(b->ds.lat_q8), H.Free(b->ds.delta);
    H.Free(b->ds.h_q8), H.Free(b->ds.logits), H.Free(b->ds.argmax);
    if (b->ds.zbuf) H.Free(b->ds.zbuf);
    if (b->ds.zpin) H.HostFree(b->ds.zpin);
    if (b->ds.id_map_dev) H.Free(b->ds.id_map_dev);
    if (b->ds.id_dyn_pin) {
      if (b->ds.id_dyn_pinned) H.HostFree(b->ds.id_dyn_pin);
      else free(b->ds.id_dyn_pin);
    }
    free(b->ds.id_core_host), free(b->ds.core_bm), free(b->ds.dyn_bm);
    if (b->ds.ev_feat_ready) H.EventDestroy(b->ds.ev_feat_ready);
    if (b->ds.ev_ctx_done) H.EventDestroy(b->ds.ev_ctx_done);
    if (b->ds.ctx_stream && H.StreamDestroy) H.StreamDestroy(b->ds.ctx_stream);
    if (b->ds.p_w1) H.Free(b->ds.p_w0), H.Free(b->ds.p_w1), H.Free(b->ds.base), H.Free(b->ds.hq);
    if (b->ds.zero) H.Free(b->ds.zero);
  }
  if (b->kv_grow == 1) {
    const size_t na = b->m->n_kvl ? b->m->n_kvl : 1u;
    for (size_t a = 0; a < na * 2u; a++) {
      if (b->g->kv_len[a >> 1] < b->g->n_ctx) continue;
      char *base = (char *)(a & 1u ? b->vcl[a >> 1] : b->kcl[a >> 1]);
      for (size_t off = 0; off < b->kv_mapped; off += b->kv_chunk) {
        H.MemUnmap(base + off, b->kv_chunk);
        H.MemRelease(b->kv_h[a * b->kv_nchunk + off / b->kv_chunk]);
      }
      H.MemAddressFree(base, b->kv_va);
    }
    free(b->kv_h), free(b->kcl), free(b->vcl);
  } else if (b->kv_grow) { /* the arrays were in b->us (slab_free above) */
    free(b->kcl), free(b->vcl);
  }
  for (int k = 0; k < IE_SNAP_MAX; k++) {
    if (snap_host()) {
      if (b->snap_ring[k]) free(b->snap_ring[k]), free(b->snap_st[k]), free(b->snap_h[k]), free(b->snap_sc[k]);
      free(b->snap_kv[k]), free(b->snap_dskv[k]);
    } else {
      if (b->snap_ring[k]) H.Free(b->snap_ring[k]), H.Free(b->snap_st[k]), H.Free(b->snap_h[k]), H.Free(b->snap_sc[k]);
      if (b->snap_kv[k]) H.Free(b->snap_kv[k]);
      if (b->snap_dskv[k]) H.Free(b->snap_dskv[k]);
    }
  }
  H.Free(b->am_part), H.Free(b->am_count), H.Free(b->at_count), H.Free(b->at_gcount), H.Free(b->at_part), H.Free(b->tk_buf), H.Free(b->gm_count), H.Free(b->ink_moe_rcount), H.Free(b->ink_qhead_cnt), H.Free(b->ink_sconv_rcnt), H.Free(b->ms_packs1), H.Free(b->ms_npacks1), H.Free(b->ms_packsM), H.Free(b->ms_npacksM);
  H.Free(b->tr_dev), H.Free(b->tr_rope), H.Free(b->cl_dev), H.Free(b->cl_scr);
  if (b->tr_pin) H.HostFree(b->tr_pin);
  if (b->cl_pin) H.HostFree(b->cl_pin);
  if (b->tk_pin) H.HostFree(b->tk_pin);
  if (b->am_pin) H.HostFree(b->am_pin);
  if (b->ink_toks_pin) {
    for (u32 k = 0; k < 16u; k++) if (b->ink_toks_ev[k]) H.EventDestroy(b->ink_toks_ev[k]);
    H.HostFree(b->ink_toks_pin);
  }
  if (b->snap_pin) {
    H.EventDestroy(b->snap_ev[0]), H.EventDestroy(b->snap_ev[1]);
    H.HostFree(b->snap_pin);
  }
  if (b->snap_cs && H.StreamDestroy) H.StreamDestroy(b->snap_cs);
  free(b->cl_host);
  if (fl_ev) H.EventDestroy(fl_ev), fl_ev = NULL;
  fl_every = -2, fl_cnt = 0;
  if (b->prof) {
    for (uint32_t i = 0; i < b->g->n_ops; i++) H.EventDestroy(b->pev[i]);
    free(b->pev), free(b->pms);
  }
  free(b->hk), free(b->hv), free(b->d), free(b);
}

static void gpu_set_rope(backend *bk, uint32_t pos0, uint32_t n, const float *cos_tab, const float *sin_tab) {
  gpu_backend *b = (gpu_backend *)bk;
  const u32 h2 = b->m->n_rot / 2u;
  if (!n || pos0 + n > b->g->n_ctx) return;
  const size_t off = (size_t)pos0 * h2;
  const size_t bytes = (size_t)n * h2 * 4u;
  HIP(H.Memcpy(b->rcos + off, cos_tab, bytes, hipMemcpyHostToDevice));
  HIP(H.Memcpy(b->rsin + off, sin_tab, bytes, hipMemcpyHostToDevice));
}

int gpu_vis_ready(void) { return H.lib != NULL; }
void *gpu_vis_alloc(size_t bytes) {
  void *p = NULL;
  HIP(H.Malloc(&p, bytes ? bytes : 16u));
  return p;
}
void gpu_vis_free(void *p) {
  if (p) H.Free(p);
}
void gpu_vis_h2d(void *dst, const void *src, size_t bytes) {
  if (bytes) HIP(H.Memcpy(dst, src, bytes, hipMemcpyHostToDevice));
}
void gpu_vis_d2h(void *dst, const void *src, size_t bytes) {
  if (bytes) HIP(H.Memcpy(dst, src, bytes, hipMemcpyDeviceToHost));
}
void *gpu_vis_load_mod(const char *path) {
  if (!H.lib || !path) return NULL;
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz <= 0) { fclose(f); return NULL; }
  void *buf = malloc((size_t)sz);
  if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
  fclose(f);
  hipModule_t mod = NULL;
  hipError_t e = H.ModuleLoadData(&mod, buf);
  free(buf);
  return e == 0 ? (void *)mod : NULL;
}
void *gpu_vis_get_fn(void *mod, const char *name) {
  hipFunction_t fn = NULL;
  if (!mod || H.ModuleGetFunction(&fn, (hipModule_t)mod, name) != 0) return NULL;
  return (void *)fn;
}
void gpu_vis_unload_mod(void *mod) {
  if (mod && H.ModuleUnload) H.ModuleUnload((hipModule_t)mod);
}
void gpu_vis_launch(void *fn, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, void **args) {
  HIP(H.ModuleLaunchKernel((hipFunction_t)fn, gx, gy, gz, bx, by, bz, 0, cur_stream, args, NULL));
}
void gpu_vis_sync(void) {
  if (cur_stream && H.StreamSynchronize) HIP(H.StreamSynchronize(cur_stream));
  else HIP(H.DeviceSynchronize());
}
