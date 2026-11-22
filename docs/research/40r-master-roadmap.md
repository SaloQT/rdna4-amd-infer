# Inkling-2-Mini (IQ3_XXS) — 40-Round Master Optimization Roadmap (`R01..R40`)

**Target:** AMD Radeon RX 9070 XT (`gfx1201`, RDNA4, `wave32`, 64 CUs, 640 GB/s, 16 GiB VRAM, Windows WDDM/PAL) running `amd-infer` (`ref/tr8-team-merge`).  
**Baseline (`R00`, measured #191):** Phase 7 defaults (`IE_INKLING_KERNEL_MASK=0x67dfc0369c3ffff`, `DCACHE_VERSION=3`): `T=1` `109.19 tok/s` (`9.158 ms/tok` wall, GPU `7.14 ms`, `2.005 ms` WDDM retirement gap, `1,574` launches/step), DSpark `5x0.6` 4K `172.50 tok/s` (prefill `4,644.2 tok/s`, TTFT `882.0 ms`), 16K `138.60 tok/s` (prefill `4,368.0 tok/s`, TTFT `3,750.9 ms`), live `ie-serve` sampled `146.67 tok/s`, `sampling 0.399 ms/step` (`topk_prefetch` 100%-miss `200058 != 201024`!), `multi_req_hit_ttft 2,113.0 ms`, warm `.iec` TTFT `220.0 ms` (`disk 171.0 ms`).  
**Research Backing:** `research/inkling2/40r-research-bank.md` (Waves 1–4), `research/inkling2/40r-research-wave5-8.md` (Waves 5–8), and `research/inkling2/r00-phase7-trace.md` (live trace #191).

---

## 1. 128-Bit Kernel Mask Expansion Specification (`ink_mask` + `ink_mask2`)

Currently `src/hip.c:1731-1766` uses `uint64_t ink_mask` with **`59` bits (`0..58`)** allocated (`IE_INK_DEFAULT_MASK = 0x67dfc0369c3ffffULL`), leaving only **5 bits (`59..63`)** before overflow. Mirror the proven Saluki 128-bit loader already in `src/hip.c:1686-1724`:

```c
// src/hip.c struct gpu_backend (near line 601):
uint64_t ink_mask;   // bits 0..63  (IE_INKLING_KERNEL_MASK low 64 bits)
uint64_t ink_mask2;  // bits 64..127 (IE_INKLING_KERNEL_MASK2, OR upper hex digits of >16-digit IE_INKLING_KERNEL_MASK)

// Loader rule (mirroring saluki_mask_lo / saluki_mask_hi at src/hip.c:1690-1717):
// 1. Parse IE_INKLING_KERNEL_MASK: if >16 hex digits, split into (ink_mask2, ink_mask); else low 64 bits.
// 2. If not hex128 and IE_INKLING_KERNEL_MASK2 is set, parse ink_mask2 = strtoull(...).
// 3. Bit lookup for ink_bit[i - K_INK_REL_ATTN_PF]:
const u32 bi = (u32)ink_bit[i - K_INK_REL_ATTN_PF];
const int bit_on = (bi < 64u)  ? (((ink_mask  >> bi)         & 1ULL) != 0ULL)
                 : (bi < 128u) ? (((ink_mask2 >> (bi - 64u)) & 1ULL) != 0ULL) : 0;
// 4. Default macros:
//    IE_INK_DEFAULT_MASK  (bits 0..63)
//    IE_INK_DEFAULT_MASK2 (bits 64..127)
// 5. Log line: "gpu: loaded %s (mask 0x%llx mask2 0x%llx)\n"
// 6. Fix ink_graph_key (src/hip.c:5934): mix BOTH 64-bit masks (not ink_mask & 0xffffffffULL)!
//    uint64_t mhash = (b->ink_mask ^ (b->ink_mask >> 32)) ^ ((b->ink_mask2 ^ (b->ink_mask2 >> 32)) * 0x9e3779b9ULL);
//    uint64_t key   = ((uint64_t)swa_sp << 48) | ((uint64_t)glob_sp << 32) | (mhash & 0xffffffffULL);
```

- **Bits `0..58`**: Frozen Phase 1–7 kernels (unchanged).
- **Bits `59..63`** (`ink_mask`): Assigned to Wave 2 Batch 1 (`R06–R09`).
- **Bits `64..91`** (`ink_mask2`, i.e., `1ULL << (bit - 64)`): Assigned to Wave 1/2/3/4/5/6/7 new kernels (`R03`, `R05`, `R10–R37`), leaving **36 spare bits (`92..127`)** for follow-ups.

---

## 2. Harness Flag Legend & Parity Rules

- **Harness Modes (`bench/inkling2_40r_harness.py`):**
  - `core`: `t1_233`, `ds5_233`, `ds5_4k`, `long_16k` (~48s trace-off wall clock + exact SHA parity against R00).
  - `+cache`: add `--with-cache` (4K disk-cache cold restart TTFT, `.iec` size, exact SHA).
  - `+serve`: add `--with-serve` (live `ie-serve` sampled chat `temp=1.0`, `sampling_ms_step`, 3-req side-request prefix retention TTFT).
  - `+vbench`: add `--with-vbench` (`IE_VERIFY_BENCH=1` `T=1..16` micro-profile).
  - `+kld`: add `--with-kld` (logit KLD audit against R00 fp32 reference).
- **Parity Rules:**
  - `exact-SHA`: 100% bit-for-bit identical greedy token SHA-256 across all prompts (`--status promoted` enforced by harness).
  - `f32-tol / KLD<=0.001`: Exact greedy SHA preferred; where WMMA FP16/FMA reassociation shifts a near-tie, verify `KLD <= 0.001` (`--allow-nonexact --with-kld`) and 0 degradation.
  - `lossless-stochastic`: Greedy (`temp=0`) MUST remain `exact-SHA`; sampled (`temp>0`) must be mathematically unbiased (`min(1, p/q)` + `max(0, p-q)` residual, verified via distribution / chi-square & DRY exactness).
- **Ownership Rules:**
  - `gpu-integrator` exclusively edits `ref/tr8-team-merge/src/*` and `kernels/ie_inkling_kernels.c`, builds HSACO/host, and executes all GPU benchmarks.
  - `kernel-author-a` and `kernel-author-b` author strictly disjoint `kernels/ie_inkling_r{7,8,9,10}_*.inc` files and verify 0 warnings / 0/0 VGPR spills via CPU-only `tmp_ka.hsaco` / `tmp_kb.hsaco`.
  - `harness-and-ui` authors read-only host patch specs and updates harness/dashboard when needed.

---

## 3. Master 40-Round Schedule (`R01..R40`, Ordered by Compounding ROI)

### WAVE 1 (`R01–R05`): TTFT, Cache Immunity, GPU Sampling/DRY, WDDM Scrub & Probabilistic DSpark *(synced with #191/#193/#194)*

| Rd | Name | Target | Verified Open-Source / Paper Source | Owner | Target File(s) | Mask Bit | Parity Rule | Harness Flags |
|---:|---|---|---|---|---|---|---|---|
| **R01** | Ephemeral Side-Request Cache Immunity + Host-RAM `ck_limit=8` + SWA `live_snap` Skip (`2113 ms -> ~10 ms`, kills `98–155 ms` H2D) | **TTFT** | **SGLang RadixAttention / SLRU / HiCache** ([arXiv:2312.07104](https://arxiv.org/abs/2312.07104), [`sgl-project/sglang`](https://github.com/sgl-project/sglang) `radix_cache.py`, `evict_policy.py`) | `gpu-integrator` *(spec: `harness-and-ui`)* | `src/gen.c` (`ck_limit`, `3361-3423`), `src/hip.c:6800,6877` (`live_snap`) | Host (`IE_SIDE_REQ_MAX`, `IE_CACHE_CKPT_SWA`) | `exact-SHA` (+ Saluki-safe) | `core +serve +cache` |
| **R02** | Fix 100%-Miss `topk_prefetch` (`vocab_unpadded=200058`) + $O(1)$ Hash-Chain DRY + Batched Verify Argmax/Top-K (`0.72–1.12 ms` `ie_argmax` -> 1 launch!) | **Decode** | **FlashInfer** ([arXiv:2501.01005](https://arxiv.org/abs/2501.01005), [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer) `sampling.cuh`, `topk.cuh`) & **llama.cpp PR #9702** ([`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) `llama-sampler.cpp`) | `gpu-integrator` + `kernel-author-a` | `src/hip.c:6534,7396`, `src/gen.c:146`, `kernels/ie_inkling_r7_verify_topk.inc` *(assigned #194)* | `ink_mask2` bits **92, 93** (`argmax_batched`, `topk_batched`) | `exact-SHA` (0-bit diff) | `core +serve` |
| **R03** | WDDM HIP Dispatch Scrub (`ink_mask2` 128-bit + 64-bit `ink_graph_key`, `hipStreamNonBlocking`, Pinned Scalar D2H, `IE_FLUSH_EVERY` / Packed Args) | **Decode / TTFT** | **ROCm/clr** ([`ROCm/clr`](https://github.com/ROCm/clr) `rocclr/platform/kernel.cpp` `captureHIPArgs`, Issue `#13762`) | `gpu-integrator` *(spec: `harness-and-ui`)* | `src/hip.c` (`1731-1767`, `2515-2548`, `5890-5970`) | Host (`IE_FLUSH_EVERY`, `IE_HIP_PACKED_ARGS`, adds `ink_mask2`) | `exact-SHA` | `core +vbench` |
| **R04** | Overlapped Async Pinned Disk-Cache Streaming (`ReadFile` overlapped with `hipMemcpyHtoDAsync` + GPU `dc_unpack16`) | **TTFT** | **LMCache & Mooncake** ([arXiv:2510.09665](https://arxiv.org/abs/2510.09665), [arXiv:2407.00079](https://arxiv.org/abs/2407.00079), [`LMCache/LMCache`](https://github.com/LMCache/LMCache) `local_disk_backend.py`, [`kvcache-ai/Mooncake`](https://github.com/kvcache-ai/Mooncake)) | `gpu-integrator` *(spec: `harness-and-ui`)* | `src/gen.c` (`dc_rd_start/wait`, `dc_load`), `src/hip.c` | Host (`IE_DISK_PIPE`) | `exact-SHA` (+ Saluki-safe) | `core +cache` |
| **R05** | Lossless Probabilistic DSpark (`qnode` + GPU `xorshift64*` into `spec_rej_step()`) + DSpark Global Layer-1 Split-KV Attn (`1.45 ms` @ 16K!) | **Decode (Sampled & 16K)** | **Leviathan / Chen** ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192), [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)), **DSpark** ([arXiv:2607.05147](https://arxiv.org/abs/2607.05147)), **vLLM PR #69028** & `rejection_sampler.py`, **FlashInfer Split-KV** | `kernel-author-a` -> `gpu-integrator` | `src/gen.c:1621-1648`, `src/hip.c:6355-6413`, `kernels/ie_inkling_r7_dspark_r7.inc` *(assigned #194)* | `ink_mask2` bits **94, 95** (`ds_cap_sample_zsync`, `ds_attn_long`) | `lossless-stochastic` (`temp=0` exact-SHA!) | `core +serve` |

---

### WAVE 2 (`R06–R12`): Decode Launch Super-Fusions (`1,574 -> ~675–818` Launches/Step on `gfx1201`)

| Rd | Name | Target | Verified Open-Source / Paper Source | Owner | Target File(s) | Mask Bit | Parity Rule | Harness Flags |
|---:|---|---|---|---|---|---|---|---|
| **R06** | **Fusion A:** Exact Router + Sigmoid/Top-8 (`T=1`) & Router + Top-8 + Compact Sort (`T<=16`) | **Decode** | **ROCm/aiter** ([`ROCm/aiter`](https://github.com/ROCm/aiter) `fused_moe.py` `fused_topk`/`moe_sorting`) & **SGLang** (`sgl-kernel/`) | `kernel-author-a` -> `gpu-integrator` | `kernels/ie_inkling_r7_router_fuse.inc` *(assigned #183)* | `ink_mask` bit **59** (`T=1`), bit **60** (`T<=16`) | `exact-SHA` (0-bit diff) | `core +vbench` |
| **R07** | **Fusion C1:** Fused K/V ShortConv + VClip + f16 KV-Store `[+ deferred hist]` (40 SWA + 8 Global) | **Decode** | **Dao-AILab `causal-conv1d`** ([`Dao-AILab/causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d) `causal_conv1d_update.cu` `kIsCircularBuffer`) | `kernel-author-a` -> `gpu-integrator` | `kernels/ie_inkling_r7_kv_sconv_fuse.inc` *(assigned #183)* + 1-line `v_dyn` hoist in `src/model.c` | `ink_mask` bit **61** | `exact-SHA` (0-bit diff) | `core +vbench` |
| **R08** | **Fusion C2:** Fused `attn_pc` ShortConv + Residual Add / Global Block Copy `[+ deferred hist]` (48 visits) | **Decode** | **Dao-AILab `causal-conv1d`** (`causal_conv1d_update_kernel` epilogue fusion) | `kernel-author-a` -> `gpu-integrator` | `kernels/ie_inkling_r7_kv_sconv_fuse.inc` *(assigned #183)* | `ink_mask` bit **62** | `exact-SHA` (0-bit diff) | `core +vbench` |
| **R09** | **Fusion B:** Fused `exp_down` (`IQ4_NL`) + Ordered 8-Slot Combine + `ffn_pc` ShortConv + Residual Add (48 visits) | **Decode** | **vLLM Modular MoE / AITER** ([`vllm-project/vllm`](https://github.com/vllm-project/vllm) `modular_kernel.py`, `kMoEFinalizeARResidualRMSNorm`) | `kernel-author-a` -> `gpu-integrator` | `kernels/ie_inkling_r7_moe_tail.inc` *(assigned #183)* + `use_last(mq,yd,rt)` in `src/model.c:3305` | `ink_mask` bit **63** *(last bit of `ink_mask`!)* | `exact-SHA` (0-bit diff) | `core +vbench` |
| **R10** | **Fusion D1:** Unified QKV `Q8_0` GEMV + `q_clip` `[+ k_dyn/v_dyn]` (`T=1` & `T<=16`, 48 visits) | **Decode** | **ROCm/aiter** (`aiter/ops/fused_qk_norm_rope_cache_quant.py` architecture) & **FlashInfer** | `kernel-author-a` -> `gpu-integrator` | `kernels/ie_inkling_r7_attn_prep.inc` *(assigned #194)* + `use_last(kq,vq)` in `src/model.c` | `ink_mask2` bit **64** (`0x1`) | `exact-SHA` (0-bit diff) | `core +vbench` |
| **R11** | **Fusion D2:** Unified `wrel + r_clip + wsink + wagt` 4->1 `Q8_0` GEMV Launch (48 visits) | **Decode** | **ROCm/aiter** multi-output shared-input `Q8_0` fusion & **vLLM** `fused_qkv` pattern | `kernel-author-a` -> `gpu-integrator` | `kernels/ie_inkling_r7_attn_prep.inc` *(assigned #194)* | `ink_mask2` bit **65** (`0x2`) | `exact-SHA` (0-bit diff) | `core +vbench` |
| **R12** | **Fusion E:** Verify `exp_gu_iq3 + SITU + Q8` (`E1`), Non-Loop `attn_res + RMSNorm` (`E2`), `wo + attn_pc + res` (`E3`) & SWA C1+D1 Merge | **Decode** | **SGLang** (`sgl-kernel/` consolidated norm/activation fusions) & **vLLM** modular epilogue fusion | `kernel-author-a` -> `gpu-integrator` | `kernels/ie_inkling_r8_bonus_fuse.inc` | `ink_mask2` bits **66, 67, 68** | `exact-SHA` (0-bit diff) | `core +vbench` |

---

### WAVE 3 (`R13–R18`): Prefill WMMA GEMM & Sparse MoE on `gfx1201` (4K `4.56K -> 5.5K+ tok/s`, 16K `3.74K -> 4.6K+ tok/s`)

| Rd | Name | Target | Verified Open-Source / Paper Source | Owner | Target File(s) | Mask Bit | Parity Rule | Harness Flags |
|---:|---|---|---|---|---|---|---|---|
| **R13** | Flat Dynamic Largest-First MoE Prefill Tile Worklist (`16/32/48/64` pairs, `N=16` Few-Row MMQ) | **Prefill** | **SonicMoE** ([arXiv:2512.14080](https://arxiv.org/abs/2512.14080)), **ScatterMoE** ([arXiv:2403.08245](https://arxiv.org/abs/2403.08245), [`shawntan/scattermoe`](https://github.com/shawntan/scattermoe)), **MegaBlocks** ([arXiv:2211.15841](https://arxiv.org/abs/2211.15841)), **llama.cpp PR #32065** | `kernel-author-b` -> `gpu-integrator` | `kernels/ie_inkling_r8_pf_worklist.inc` | `ink_mask2` bit **69** (`0x20`) | `exact-SHA` (0-bit diff) | `core` |
| **R14** | Prefill `IQ3_XXS + SITU` `<=128 VGPR` (3 WGs/CU): `h0/h1->w0/w1` 0-Bank-Conflict, `0x4B400000`, `r11_g01` Pipelined A/dx + Worklist | **Prefill** | **AMD RDNA4 WMMA Blogs** ([FP16](https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp16-gemm/README.html), [FP8](https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp8-gemm/README.html), [`ROCm/rocm-blogs`](https://github.com/ROCm/rocm-blogs)) & **`ik_llama.cpp`** ([`ikawrakow/ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) `mmq.cuh`) | `kernel-author-b` -> `gpu-integrator` | `kernels/ie_inkling_r8_iq3_situ_pf.inc` | `ink_mask2` bit **70** (`0x40`) | `exact-SHA` (0-bit diff) | `core` |
| **R15** | Prefill `IQ4_NL` Down `<=128 VGPR` (3 WGs/CU): `f16 sw`, Mid-K `rx` Stage, 32-bit VOffsets, `2×f32x4` Stores + Worklist | **Prefill** | **ROCm `ck_tile` `gfx12_t`** ([`ROCm/composable_kernel`](https://github.com/ROCm/composable_kernel) `include/ck_tile/ops/gemm/pipeline/`) & **`ik_llama.cpp`** `IQ4_NL` | `kernel-author-b` -> `gpu-integrator` | `kernels/ie_inkling_r8_iq4nl_pf.inc` | `ink_mask2` bit **71** (`0x80`) | `exact-SHA` (0-bit diff) | `core` |
| **R16** | Prefill Dense `Q8_0` (`q8_med_r8` `<=128 VGPR` 3 WGs/CU + `q8_small_r8` `32x64/64x64` + L2/MALL 4×4 Swizzle) | **Prefill** | **AMD RDNA4 WMMA Blogs** (ping-pong XOR LDS + L2 block swizzling, `ROCm/rocm-blogs`) | `kernel-author-b` -> `gpu-integrator` | `kernels/ie_inkling_r8_dense_pf.inc` | `ink_mask2` bits **72, 73** | `exact-SHA` (0-bit diff) | `core` |
| **R17** | Prefill F32 Router `32x64x16` 4×4 FMA (`<=96 VGPR`, 4 WGs/CU) + Fused `OP_MOE_ROUTE` | **Prefill** | **ROCm `ck_tile`** intrawave/interwave GEMM scheduling & **ROCm/aiter** `fused_topk` | `kernel-author-b` -> `gpu-integrator` | `kernels/ie_inkling_r8_dense_pf.inc` | `ink_mask2` bit **74** | `f32-tol / KLD<=0.001` (exact-SHA expected) | `core +kld` |
| **R18** | Prefill Fused ShortConv + VClip + f16 KV-Store & Chunk-Size (`512` vs `1024`) Co-Tuning | **Prefill** | **Dao-AILab `causal-conv1d`** (`causal_conv1d_fwd`) & **vLLM** chunked prefill batching | `kernel-author-a` + `gpu-integrator` | `kernels/ie_inkling_r8_sconv_pf.inc` + `src/serve.c`/`serve-inkling2.ps1` | `ink_mask2` bit **75** | `exact-SHA` | `core` |

---

### WAVE 4 (`R19–R23`): Long-Context 16K–32K Prefill & Verify Attention + `Q4_K` Output Head on `gfx1201`

| Rd | Name | Target | Verified Open-Source / Paper Source | Owner | Target File(s) | Mask Bit | Parity Rule | Harness Flags |
|---:|---|---|---|---|---|---|---|---|
| **R19** | `Q4_K` Output Head 1-Pass Verify (`8` lanes/row, `4` rows/wave, `<=96 VGPR`, fixes `ts2` 2× DRAM read, fused `OP_LOGIT_CAP`) | **Decode** | **`ggml-org/llama.cpp`** Q4_K MMVQ (`ggml-cuda`) & **Saluki R8/R9** multi-row wave32 GEMV | `kernel-author-b` -> `gpu-integrator` | `kernels/ie_inkling_r7_q4k_head.inc` *(assigned #185)* | `ink_mask2` bit **76** | `exact-SHA` (0-bit diff) | `core +vbench` |
| **R20** | Verify SWA `win=1024` Multi-Query `QT=4..8` Attention (cuts 40-layer SWA KV read `1.01 -> 0.17–0.34 GB`/step) | **Decode** | **FlashInfer** ([arXiv:2501.01005](https://arxiv.org/abs/2501.01005) `include/flashinfer/attention`) & **ROCm/aiter** SWA Wave32 (`Issue #4329` LDS cap) | `kernel-author-b` -> `gpu-integrator` | `kernels/ie_inkling_r7_attn_ver.inc` *(assigned #185)* | `ink_mask2` bit **77** | `f32-tol / KLD<=0.001` (exact-SHA expected) | `core +vbench +kld` |
| **R21** | Verify Global Single-Pass `T<=8` FP16 WMMA Attention (1× Global KV read vs `qtile_r6`'s 2× at `T=5..8`, `<=128 VGPR`) | **Decode (16K–32K)** | **ROCm `ck_tile` FMHA `gfx12_t`** (`include/ck_tile/ops/fmha/`), **`llama.cpp`** `fattn-wmma-f16.cu` | `kernel-author-b` -> `gpu-integrator` | `kernels/ie_inkling_r7_attn_ver.inc` *(assigned #185)* | `ink_mask2` bit **78** | `f32-tol / KLD<=0.001` | `core +vbench +kld` |
| **R22** | Prefill 16K–32K Two-Phase Far/Near WMMA Attention (0 `% kv_len` divides, 94–97% zero-`rb` fast path, pipelined `D=3`, Wave-2 causal pairing) | **Prefill (4K–32K)** | **ROCm/aiter `fmha_gfx1201` / FlyDSL** ([`ROCm/aiter`](https://github.com/ROCm/aiter)) & **FlashDecoding++** ([arXiv:2311.01282](https://arxiv.org/abs/2311.01282)) | `kernel-author-b` -> `gpu-integrator` | `kernels/ie_inkling_r7_attn_pf.inc` *(assigned #185)* | `ink_mask2` bit **79** | `f32-tol / KLD<=0.001` | `core +kld` |
| **R23** | Prefill Long-Context Incremental Split-KV Attention (`ntile*n_kv < 128 && pos0 >= 2048`) + `NB=32` Far-Field Loop | **TTFT / Prefill (16K–32K)** | **FlashInfer Split-KV & Cascade Inference** ([arXiv:2501.01005](https://arxiv.org/abs/2501.01005), [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer)) | `kernel-author-b` -> `gpu-integrator` | `kernels/ie_inkling_r7_attn_pf.inc` *(assigned #185)* / `ie_inkling_r8_attn_splitkv.inc` | `ink_mask2` bit **80** | `f32-tol / KLD<=0.001` | `core +serve +kld` |

---

### WAVE 5 (`R24–R28`): Decode GEMV Bandwidth, FLUTE Repacking, Fused Head+Top-K & Adaptive MoE

| Rd | Name | Target | Verified Open-Source / Paper Source | Owner | Target File(s) | Mask Bit | Parity Rule | Harness Flags |
|---:|---|---|---|---|---|---|---|---|
| **R24** | Reversible Load-Time `FLUTE` Lane-Aligned `IQ3_XXS` SoA Repacking (`98 B -> 98 B`, `0 MB` extra VRAM, coalesced `u32x4`) | **Decode / Prefill** | **FLUTE** ([arXiv:2407.10960](https://arxiv.org/abs/2407.10960), [`HanGuo97/flute`](https://github.com/HanGuo97/flute) `flute/utils.py`, `flute/csrc/`) | `kernel-author-b` + `gpu-integrator` | `src/hip.c` (`gpu_sink`), `kernels/ie_inkling_r9_iq3_soa.inc` | `ink_mask2` bit **81** (`IE_INKLING_IQ3_REPACK`) | `exact-SHA` (0-bit diff) | `core +vbench` |
| **R25** | Multi-Row Wave-Uniform `IQ4_NL` & `Q8_0` Decode/Verify GEMVs (`v_perm_b32` + `s_wait_loadcnt` double-buffer, `<=96 VGPR`) | **Decode** | **Marlin** ([`IST-DASLab/marlin`](https://github.com/IST-DASLab/marlin)) & **ExLlamaV3** ([`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3)) | `kernel-author-b` -> `gpu-integrator` | `kernels/ie_inkling_r9_gemv_bw.inc` | `ink_mask2` bits **82, 83** | `exact-SHA` (0-bit diff) | `core +vbench` |
| **R26** | Fused Persistent `Q4_K` Output Head + Softcap + Tile-Local Online Top-K / Argmax (`0 B` `g->logits` write+reread) | **Decode** | **RadiK** ([arXiv:2501.14336](https://arxiv.org/abs/2501.14336), [`leefige/radik`](https://github.com/leefige/radik)), **AIR Top-K** ([`NVIDIA/cccl`](https://github.com/NVIDIA/cccl)), [`lsnchow/LLaDa-CUDA-fused-kernel`](https://github.com/lsnchow/LLaDa-CUDA-fused-kernel) | `kernel-author-b` -> `gpu-integrator` | `kernels/ie_inkling_r9_head_topk.inc` + `src/hip.c` | `ink_mask2` bit **84** | `exact-SHA` (0-bit diff) | `core +serve +vbench` |
| **R27** | Fused DSpark `P*lat + base + softcap + online top-K/sample` (Zero `ds.logits` Materialization in `zsync`) | **Decode (DSpark)** | **FlashInfer** (`sampling.cuh`), **RadiK** ([arXiv:2501.14336](https://arxiv.org/abs/2501.14336)) | `kernel-author-a` -> `gpu-integrator` | `kernels/ie_inkling_r9_dspark_head_fuse.inc` | `ink_mask2` bit **85** | `exact-SHA` (`temp=0`) / `lossless-stochastic` | `core +serve` |
| **R28** | Adaptive `T=2..16` MoE Verify Crossover (`slot_r6` vs `compact_r6` vs `N=16` Few-Row WMMA for Hot Experts) | **Decode (Verify)** | **`ggml-org/llama.cpp` PR #32065** (*few-row MMQ for MUL_MAT_ID*, `mmq.cuh`) & **ik_llama.cpp** `mmq_id.cu` | `kernel-author-b` -> `gpu-integrator` | `kernels/ie_inkling_r9_moe_adapt.inc` | `ink_mask2` bit **86** | `exact-SHA` (0-bit diff) | `core +vbench` |

---

### WAVE 6 (`R29–R33`): ForeSpark Stopping, Certified Shortlists, SAM Hybrid, Tree Verify & Multi-Stream Overlap

| Rd | Name | Target | Verified Open-Source / Paper Source | Owner | Target File(s) | Mask Bit | Parity Rule | Harness Flags |
|---:|---|---|---|---|---|---|---|---|
| **R29** | ForeSpark Entropy-Calibrated Cumulative Survival-Probability Stopping (Replaces Static `pmin` at `temp>0` & `temp=0`) | **Decode (DSpark)** | **ForeSpark** ([arXiv:2611.13442](https://arxiv.org/abs/2611.13442), [`facebookresearch/ForeSpark`](https://github.com/facebookresearch/ForeSpark)) & **DSpark** ([arXiv:2607.05147](https://arxiv.org/abs/2607.05147)) | `gpu-integrator` + `kernel-author-a` | `src/gen.c`, `src/hip.c`, `kernels/ie_inkling_r7_dspark_r7.inc` | `ink_mask2` bit **87** (`IE_DS_FORESPARK`) | `exact-SHA` (`temp=0`) / `lossless-stochastic` | `core +serve` |
| **R30** | Certified 60K `FR-Spec` Core + 4K `MicroSpec` Dynamic Prompt Shortlist (`0 MB` VRAM `id_map` Gather -> `ds_draft < 0.9 ms`) | **Decode (DSpark)** | **FR-Spec** ([arXiv:2502.14856](https://arxiv.org/abs/2502.14856), [`thunlp/FR-Spec`](https://github.com/thunlp/FR-Spec)), **MicroSpec** ([arXiv:2605.26444](https://arxiv.org/abs/2605.26444)), **SpecPilot** ([arXiv:2611.02501](https://arxiv.org/abs/2611.02501)) | `kernel-author-b` + `gpu-integrator` | `tools/build_inkling2_frspec.py`, `src/gen.c`, `src/hip.c`, `kernels/ie_inkling_r10_frspec_head.inc` | `ink_mask2` bit **88** (`IE_DS_SHORTLIST`) | `exact-SHA` (100% lossless!) | `core +serve` |
| **R31** | $O(1)$ Online Suffix Automaton (`SAM`) Pre-Check Hybrid (`0 ms` Draft on `ml>=6` Code/JSON Repeats + Auto `ds_ctx` Catch-Up) | **Decode (Code/Agent)** | **SAM Decoding** ([arXiv:2411.10666](https://arxiv.org/abs/2411.10666)), **SuffixDecoding** ([arXiv:2411.04975](https://arxiv.org/abs/2411.04975), [`snowflakedb/ArcticInference`](https://github.com/snowflakedb/ArcticInference)), **Lookahead** ([arXiv:2402.02057](https://arxiv.org/abs/2402.02057)), **PLD** | `gpu-integrator` *(spec: `harness-and-ui`)* | `src/gen.c:1619-1648` | Host (`IE_SAM_LOOKUP`) | `exact-SHA` (100% lossless!) | `core +serve` |
| **R32** | Tiny 8-Node `EAGLE-2` / `Sequoia` Dynamic Tree Verify (`u8 tree_mask` Attention + `tpar` ShortConv for Uncertain Steps) | **Decode (Sampled/Hard)** | **EAGLE-2** ([arXiv:2406.16858](https://arxiv.org/abs/2406.16858), [`SafeAILab/EAGLE`](https://github.com/SafeAILab/EAGLE)), **Sequoia** ([arXiv:2402.12374](https://arxiv.org/abs/2402.12374), [`Infini-AI-Lab/Sequoia`](https://github.com/Infini-AI-Lab/Sequoia)), **SpecInfer** ([arXiv:2305.09781](https://arxiv.org/abs/2305.09781)) | `kernel-author-b` + `gpu-integrator` | `src/gen.c:1776-1824`, `kernels/ie_inkling_r10_tree_ver.inc` | `ink_mask2` bit **89** (`IE_DS_TREE`) | `exact-SHA` (`temp=0`) / `lossless-stochastic` | `core +serve +vbench` |
| **R33** | Multi-Stream Overlap (`stream_ds` Speculative `ds_ctx` Pre-Projection Overlapped with Target `lm_head` + `topk` + Host Rejection) | **Decode (DSpark)** | **ROCm/HIP** `hipStreamNonBlocking` + `hipEventRecord` cross-stream async pipeline | `gpu-integrator` | `src/hip.c` (`stream_ds`, `ev_feat_ready`), `src/gen.c` | Host (`IE_DS_OVERLAP`) | `exact-SHA` | `core +serve` |

---

### WAVE 7 (`R34–R37`): Low-VGPR Sub-Span Persistence, Piecewise HIP Graphs, Arena Compaction & Cold-Expert Tier

| Rd | Name | Target | Verified Open-Source / Paper Source | Owner | Target File(s) | Mask Bit | Parity Rule | Harness Flags |
|---:|---|---|---|---|---|---|---|---|
| **R34** | Low-VGPR (`<=64 VGPR`) Non-GEMV Glue & L2-Hot `T=1` MoE Sub-Span Persistent Kernel (Grid `<=64`, WDDM Wall-Clock Guarded) | **Decode (`T=1`)** | **Mirage MPK** ([arXiv:2512.22219](https://arxiv.org/abs/2512.22219), OSDI'26, [`mirage-project/mirage`](https://github.com/mirage-project/mirage)) & **MonoMoE** ([arXiv:2609.04244](https://arxiv.org/abs/2609.04244)) | `kernel-author-a` -> `gpu-integrator` | `kernels/ie_inkling_r10_subspan.inc` | `ink_mask2` bit **90** (`IE_INKLING_SUBSPAN`) | `exact-SHA` (0-bit diff) | `core` |
| **R35** | Piecewise 12-Visit HIP Graph Capture Interleaved with WDDM Retirement Markers + 128-Bit `ink_graph_key` | **Decode (`T=1` & `T<=8`)** | **vLLM Piecewise CUDA/HIP Graphs** ([`vllm-project/vllm`](https://github.com/vllm-project/vllm) `vllm/compilation/cuda_piecewise_backend.py`) | `gpu-integrator` | `src/hip.c:5914-5975` | Host (`IE_INKLING_GRAPH=2` piecewise) | `exact-SHA` | `core +vbench` |
| **R36** | Arena Lifetime Compaction (`exp_gu` `T_cap=16`, Prune Dead `k_buf`/`v_buf`/`exp_down` Buffers, `IE_PLAN_DUMP=1` Audit) | **VRAM / Long-Ctx** | `amd-infer` `plan_arena()` (`src/model.c:2497, 3050`) + `r6_iq3_situ_pf.inc` §91–136 contract | `gpu-integrator` | `src/model.c` (`inkling_build`, `plan_arena`) | Host (`IE_ARENA_COMPACT`) | `exact-SHA` (+ `ie_plan_check`) | `core +vbench` |
| **R37** | Long-Context (`32K–128K`) Cold-Expert Host-Backed VRAM LRU Cache Tier (Pin Looped `blk.6..17`, Cache Outer `blk.0..5,18..23`) | **VRAM / 32K–128K** | **`ggml-org/llama.cpp` PR #29887** (*"llama : add a GPU cache for MoE experts kept in host memory"*, [`ggml-org/llama.cpp#29887`](https://github.com/ggml-org/llama.cpp/pull/29887)) | `kernel-author-a` + `gpu-integrator` | `src/hip.c`, `kernels/ie_inkling_r10_moe_cache.inc` | `ink_mask2` bit **91** (`IE_MOE_CACHE_MB`) | `exact-SHA` | `core` + 32K/64K VRAM stress |

---

### WAVE 8 (`R38–R40`): Full-Stack Re-Trace Co-Tuning, 32K Multi-Turn Serve Validation & Final Certification

| Rd | Name | Target | Verified Open-Source / Paper Source | Owner | Target File(s) | Mask Bit | Parity Rule | Harness Flags |
|---:|---|---|---|---|---|---|---|---|
| **R38** | Post-Wave-7 Full GPU Trace Audit + Global Parameter Co-Tuning (`draft`, `survival_tau`, `temp_d`, `chunk`, `IE_FLUSH_EVERY`, `PACKED_ARGS`) | **All** | **SpecPilot** ([arXiv:2611.02501](https://arxiv.org/abs/2611.02501)) ceiling-guided deployment co-tuning | `gpu-integrator` | `ref/tr8-team-merge/serve-inkling2.ps1`, `src/hip.c` defaults | Promoted default tuning | `exact-SHA` (`temp=0`) / `lossless-stochastic` | `core +vbench +serve` |
| **R39** | 32K Multi-Turn + Interleaved 152-Tok Side-Request + Cold-Disk-Resume + `DRY=0.8` Sampled Live `ie-serve` Stress & SLO Audit | **TTFT / Live Serve** | Live `ie-serve` workload validation (`task-723` reproduction + 32K endurance) | `gpu-integrator` + `harness-and-ui` | `bench/inkling2_40r_harness.py` (stress suite), `src/serve.c` / `src/gen.c` polish | N/A | `exact-SHA` + 0 WDDM eviction + 0 memory leaks | `core +cache +serve` |
| **R40** | Final Default Mask Promotion (`ink_mask` + `ink_mask2`), Saluki & 3D-Coder Shared-Binary Regression Gate, `AGENTS.md` & Final Dashboard Closeout | **All (Release Gate)** | Shared-binary safety gates (`saluki-cache-check-task.ps1`, `coder3d-serve-smoke-task.ps1`) + full 40-round ledger | `gpu-integrator` + `harness-and-ui` | `src/hip.c` (`IE_INK_DEFAULT_MASK/2`), `serve-inkling2.ps1`, `AGENTS.md`, `results/inkling2/40round-dashboard.html` | Final frozen `IE_INK_DEFAULT_MASK` + `MASK2` | `exact-SHA` + `KLD<=0.001` + Saluki/3D-Coder PASS | `core +vbench +cache +serve +kld` |

---

## 4. Complete Disjoint Mask-Bit & File Allocation Map (Zero Collisions)

| Bit(s) | Mask Reg | Round(s) | `.inc` File | Owner |
|---|---|---|---|---|
| `0..58` | `ink_mask` | Phase 1–7 | `r1_g01..r6_rel_attn_qtile` (existing) | Frozen |
| `59, 60` | `ink_mask` | R06 | `ie_inkling_r7_router_fuse.inc` | `kernel-author-a` |
| `61, 62` | `ink_mask` | R07, R08 | `ie_inkling_r7_kv_sconv_fuse.inc` | `kernel-author-a` |
| `63` | `ink_mask` | R09 | `ie_inkling_r7_moe_tail.inc` | `kernel-author-a` |
| `64, 65` | `ink_mask2` (`bits 0..1`) | R10, R11 | `ie_inkling_r7_attn_prep.inc` *(assigned #194)* | `kernel-author-a` |
| `66, 67, 68` | `ink_mask2` (`bits 2..4`) | R12 | `ie_inkling_r8_bonus_fuse.inc` | `kernel-author-a` |
| `69` | `ink_mask2` (`bit 5`) | R13 | `ie_inkling_r8_pf_worklist.inc` | `kernel-author-b` |
| `70` | `ink_mask2` (`bit 6`) | R14 | `ie_inkling_r8_iq3_situ_pf.inc` | `kernel-author-b` |
| `71` | `ink_mask2` (`bit 7`) | R15 | `ie_inkling_r8_iq4nl_pf.inc` | `kernel-author-b` |
| `72, 73, 74` | `ink_mask2` (`bits 8..10`) | R16, R17 | `ie_inkling_r8_dense_pf.inc` | `kernel-author-b` |
| `75` | `ink_mask2` (`bit 11`) | R18 | `ie_inkling_r8_sconv_pf.inc` | `kernel-author-a` |
| `76` | `ink_mask2` (`bit 12`) | R19 | `ie_inkling_r7_q4k_head.inc` | `kernel-author-b` |
| `77, 78` | `ink_mask2` (`bits 13..14`) | R20, R21 | `ie_inkling_r7_attn_ver.inc` | `kernel-author-b` |
| `79, 80` | `ink_mask2` (`bits 15..16`) | R22, R23 | `ie_inkling_r7_attn_pf.inc` / `r8_attn_splitkv.inc` | `kernel-author-b` |
| `81` | `ink_mask2` (`bit 17`) | R24 | `ie_inkling_r9_iq3_soa.inc` | `kernel-author-b` |
| `82, 83` | `ink_mask2` (`bits 18..19`) | R25 | `ie_inkling_r9_gemv_bw.inc` | `kernel-author-b` |
| `84` | `ink_mask2` (`bit 20`) | R26 | `ie_inkling_r9_head_topk.inc` | `kernel-author-b` |
| `85` | `ink_mask2` (`bit 21`) | R27 | `ie_inkling_r9_dspark_head_fuse.inc` | `kernel-author-a` |
| `86` | `ink_mask2` (`bit 22`) | R28 | `ie_inkling_r9_moe_adapt.inc` | `kernel-author-b` |
| `87` | `ink_mask2` (`bit 23`) | R29 | `ie_inkling_r7_dspark_r7.inc` (ForeSpark stopping mode) | `kernel-author-a` |
| `88` | `ink_mask2` (`bit 24`) | R30 | `ie_inkling_r10_frspec_head.inc` | `kernel-author-b` |
| `89` | `ink_mask2` (`bit 25`) | R32 | `ie_inkling_r10_tree_ver.inc` | `kernel-author-b` |
| `90` | `ink_mask2` (`bit 26`) | R34 | `ie_inkling_r10_subspan.inc` | `kernel-author-a` |
| `91` | `ink_mask2` (`bit 27`) | R37 | `ie_inkling_r10_moe_cache.inc` | `kernel-author-a` |
| `92, 93` | `ink_mask2` (`bits 28..29`) | R02 | `ie_inkling_r7_verify_topk.inc` *(assigned #194)* | `kernel-author-a` |
| `94, 95` | `ink_mask2` (`bits 30..31`) | R05 | `ie_inkling_r7_dspark_r7.inc` *(assigned #194)* | `kernel-author-a` |
| `96..127` | `ink_mask2` (`bits 32..63`) | Spare | Reserved for opportunistic follow-ups / splits | — |
