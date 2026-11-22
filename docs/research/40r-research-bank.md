# Inkling-2-Mini (IQ3_XXS) 40-Round Optimization Campaign — Research & Open-Source Code Bank

**Target Platform:** AMD Radeon RX 9070 XT (`gfx1201`, RDNA4, `wave32`, 64 CUs / 32 WGPs, 640 GB/s HBM/GDDR6, 16 GiB VRAM, Windows WDDM, ROCm Clang 21 AMDGPU C11).  
**Engine & Model:** `ref/tr8-team-merge` (`amd-infer`), `Inkling-2-Mini-Uncensored-IQ3_XXS-fast.gguf` (24 physical / 48 logical looped visits, 40 SWA `win=1024` + 8 Global, `dim=2304`, `n_head=24`, `n_kv=8` GQA-3, `hd=128`, `d_rel=16`, `rel_extent=1024`, 4-tap `shortconv`, 128E/K=8 MoE: `IQ3_XXS` gate/up + `IQ4_NL` down + F32 router, `vocab=200,058` unpadded / `201,024` padded, `Q4_K` output head) + `Inkling-2-Mini-DSpark-Q4_K.gguf` (`block=8`, 3 dense layers + Markov RNN head).  
**Phase 7 Baseline (`mask=0x67dfc0369c3ffff`, `DCACHE_VERSION=3`):**
- Plain `T=1`: `105.8 tok/s` (`9.45 ms/tok`, `~1,574` launches/step)
- DSpark `5x0.6` 4K code: `142.0 tok/s` (`148.4` greedy / `130.4` sampled `ie-serve`)
- 4K cold prefill: `~4,560 tok/s` (`~892 ms`); 16K prefill: `~3,740 tok/s` (`~4.4 s`)
- 24K verify (`qtile_r6` `QT=4`): `-14..-18%`; compact 1024-row SWA `.iec` disk cache: `181 MB` / `1,110 ms` load
- **Live `ie-serve` Telemetry Smoking Guns:**
  1. **152-token side request evicted 26.8K cached conversation -> `4.4 s` re-prefill** (`src/gen.c:2013-2017, 3361-3423`)
  2. **CPU sampling + DRY penalty = `0.44 ms/step`** (`src/gen.c:117-231, 1304-1360, 1933-1934, 3533`)
  3. **16K–31.6K incremental prefill = `4.1–4.4 s`** (`kernels/ie_inkling_r1_g01.inc`, `r6_iq3_situ_pf.inc`, `r2_g15.inc`)
  4. **Temp=1.0 sampled chat DSpark acceptance = `1.63–2.20 tok/step`** because DSpark proposes greedy point-masses with `memset(qnode, 0, ...)` (`src/gen.c:1621-1648`) despite `spec_rej_step()` already implementing Leviathan `min(1, p/q)` rejection!

---

## Hard `gfx1201` / `amd-infer` Constraints (Apply to ALL Kernel Waves)

1. **Occupancy Cliffs on `gfx1201` (256 KiB VGPR/CU, 64 KiB LDS/CU, `NT=256` = 8 waves/WG):**
   - `<= 96 VGPR` & `<= 16 KiB LDS` -> **4 WGs/CU** (max latency hiding for memory-bound decode/verify)
   - `<= 128 VGPR` & `<= 21 KiB LDS` -> **3 WGs/CU** (sweet spot for verify attention & fused decode)
   - `<= 160 VGPR` & `<= 32 KiB LDS` -> **2 WGs/CU** (required floor for WMMA prefill GEMM/attention; `161+ VGPR` collapses to 1 WG/CU!)
   - **0 VGPR spills (`0/0`)** mandatory on every compiled kernel.
2. **Clang 21 `__builtin_bit_cast` Lane Bug:** NEVER pass vector `.y/.z/.w` lanes directly into `__builtin_bit_cast` (Clang 21 miscompiles to `.x`). Always extract to `const u32 tmp = v.y;` first.
3. **Arena Aliasing Contract (`src/model.c` + `src/hip.c`):** `plan_arena()` reuses input buffers immediately after their last unfused consumer. Any fused kernel reading `o0->a` while writing `o1->qo`/`o1->b` MUST guard with `buf_disjoint(g, ...)` in `hip.c` AND extend `use_last(B, ...)` in `model.c`.
4. **WDDM Grid/VRAM Rules:** Keep per-tensor streaming (`IE_SLAB_MB` unset), keep total resident VRAM comfortably under Windows WDDM's ~12.5–13.0 GiB per-process eviction cliff, and never exceed persistent grid caps (`<= 96` WGs with poison guard).
5. **Exact Parity Gate:** Every exact fusion/GEMM/attention upgrade must pass exact greedy SHA-256 parity (or `f32-tol <= 1e-5` / `KLD <= 0.001` only where explicitly documented for WMMA/probabilistic paths).

---

# WAVE 1 (Rounds 1–5): TTFT, Multi-Slot Serving Cache, Async Disk Streaming & GPU Sampling/DRY

## 1.1 Multi-Slot Prefix Retention & Side-Request Eviction Immunity

### Open-Source References & Papers
- **SGLang RadixAttention & Eviction Policies** — *SGLang: Efficient Execution of Structured Language Model Programs* ([arXiv:2312.07104](https://arxiv.org/abs/2312.07104))
  - Repo: [`sgl-project/sglang`](https://github.com/sgl-project/sglang)
  - Core files:
    - `python/sglang/srt/mem_cache/radix_cache.py` (`match_prefix()`, `inc_lock_ref()` / `dec_lock_ref()`, `evict()` min-heap over `evictable_leaves`)
    - `python/sglang/srt/mem_cache/evict_policy.py` (`--radix-eviction-policy`: `lru`, `lfu`, `slru` [probationary vs protected scan-resistant segments], `priority`)
    - `python/sglang/srt/mem_cache/hi_radix_cache.py` (**HiCache**: L1 GPU HBM -> L2 host CPU DRAM -> L3 NVMe/remote unified radix tree covering both Full KV and SWA/recurrent checkpoints)
- **LMCache & CacheBlend** — *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference* ([arXiv:2510.09665](https://arxiv.org/abs/2510.09665)), *CacheBlend* ([arXiv:2405.16444](https://arxiv.org/abs/2405.16444))
  - Repo: [`LMCache/LMCache`](https://github.com/LMCache/LMCache)
  - Core files: `lmcache/v1/storage_backend/local_disk_backend.py` (pinned host DRAM pool + async background priority queue where prefetch/read beats background write so disk writes never stall TTFT).
- **Mooncake Store** — *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving* ([arXiv:2407.00079](https://arxiv.org/abs/2407.00079))
  - Repo: [`kvcache-ai/Mooncake`](https://github.com/kvcache-ai/Mooncake) (DRAM + SSD pooled KV storage with parallel striped I/O and async prefetch).

### Exact Root Cause in `amd-infer` (`src/gen.c`)
1. **Single live KV sequence clobbered on `L == 0` (`src/gen.c:3361-3423`):** When a 152-token side request (e.g., Open WebUI title/tag/follow-up prompt) arrives while a 26.8K conversation is live in GPU KV, `L = 0`. Line 3371 flushes to disk ONLY if `c->dc` is open; then lines 3419–3423 unconditionally erase all checkpoints `pos > 0` and overwrite `c->kv_toks` + GPU KV positions `[0..152+out)` with the side request! If `IE_DISK_CACHE` is off—or if the 26.8K conversation wasn't freshly checkpointed/restored—the very next turn of the 26.8K chat triggers a **`4.4 s` full re-prefill**.
2. **Hardcoded SWA checkpoint cap (`src/gen.c:2013-2017`):** `ck_limit()` forces `return 2u;` whenever ANY layer has `kv_len[i] < n_ctx` (all 40 SWA layers of Inkling-2!), assuming SWA snapshots consume VRAM. Phase 6 already moved Inkling-2 `snap_kv` to **host RAM (`0 MB` extra VRAM)**, so clamping to 2 slots starves multi-turn / system-prompt / branch retention for zero reason!

### Concrete Implementation Blueprints (Rounds 1–2)
1. **Recipe W1-A — Zero-Clobber Prefix-Safe Side-Request Swap (`src/gen.c` + `src/hip.c`):**
   - **Key Inkling-2 geometry insight:** Only **8 physical/logical global layers** (plus 1 DSpark global layer) store full-length KV rows (`kv_rowb` = `8 * 2 * 8 * 128 * 2` = `32 KiB/pos` + `4 KiB/pos` DSpark = **`36 KiB/pos`**). A 152-tok side request (+ 64 output tokens = 216 total positions) ONLY touches `216 * 36 KiB = 7.6 MiB` of global KV (even 1,024 positions is only `36 MiB`), plus the 40 SWA rings (~`160 MiB`, already snapshotted in host RAM via `snap_kv`)!
   - **Algorithm (SLRU Scan-Resistant Ephemeral Slot):**
     - Trigger when `cache && L < 64 && c->kv_n >= 2048 && (n_prompt + p->max_new) <= side_cap` (default `side_cap = 1024` or `IE_SIDE_REQ_MAX`, configurable; also guard if `n_prompt <= 1024 && c->kv_n >= 4 * n_prompt`).
     - Before running the side request:
       1. Save live conversation's current SWA/sconv/DSpark rings into a reserved host-RAM shadow snapshot slot (`snap_shadow`), and DMA-save ONLY the `[0 .. M)` global KV rows (`M = min(c->kv_n, n_prompt + p->max_new + spec_room)`) via `b->kv_io(b, 0, M, 1, pinned_stash)` (`~7–36 MiB`, **`< 3 ms`** over PCIe 4.0/5.0!).
       2. Run the side request with `cache = 0` (no checkpoint pollution, no `c->kv_toks` clobber, no `dp_flush("switch")` thrash!).
       3. Immediately after the side request finishes (before returning or on next non-side request): DMA-restore `[0 .. M)` global KV rows via `b->kv_io(b, 0, M, 0, pinned_stash)` + restore SWA rings from `snap_shadow` (`< 10 ms` total!).
     - **Impact:** 26.8K conversation survives 152-tok side requests with **0 re-prefill (`4,400 ms -> ~10 ms`, 440× faster)**, **0 MB extra VRAM**, **100% bitwise-exact KV restoration**, even when `IE_DISK_CACHE` is OFF!
2. **Recipe W1-B — Lift `ck_limit()` for Host-RAM SWA Snapshots + Cost-Weighted SLRU Eviction (`src/gen.c:2013-2073`):**
   - Add a `b->snap_in_ram` flag (set `1` on Inkling-2 where `snap_kv` is host-allocated). When `snap_in_ram == 1`, allow `ck_limit()` up to `IE_CACHE_CKPT_SWA` (default `6..8` slots, `~1.0–1.3 GiB` host RAM, `0 MB` VRAM).
   - Upgrade `ck_slot()` eviction (`gen.c:2064-2068`) with **SGLang SLRU + recompute-cost priority**: protect checkpoints with `pos >= 4096` or `hits >= 1` in a protected tier; evict probationary short-prefix `GEN_CK_EVERY` entries first, and add a host-RAM **L2 multi-conversation pool** (`IE_RAM_CACHE_GB`, reusing `dcache`'s existing `dc_file.img` RAM tier even without disk writes!) so switching between 2–4 active chats restores from pinned host RAM in **`~120–250 ms`** instead of `4.4 s` prefill.
3. **Recipe W1-C — Async Pinned Disk-Cache Double-Buffered Streaming & Background Prefetch (`src/gen.c:2496-2535, 2987-3080`):**
   - Look at `dc_rd_start()` (`gen.c:2496-2519`): it already uses Windows `FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING` + `ReadFile` chunks! **Why does 181 MB still take `1,110 ms` (~163 MB/s on NVMe)?**
     - Check (1) `dc_load()` serializes `dc_rd_wait()` of the whole checkpoint/blocks BEFORE calling `b->kv_io()` / `b->snap_io()` H2D DMA, AND (2) `dc_get()` caps pinned buffers at `pinned_limit / 2` and falls back to **unpinned `_aligned_malloc`**, forcing HIP's synchronous pageable staging copy on WDDM, AND (3) `d->chunk` / `CreateEventA` per chunk + `dc_unpack16()` runs single-threaded on CPU AFTER read completes!
   - **LMCache / Mooncake Fix:**
     - Pipeline `ReadFile(chunk i+1)` overlapped with `hipMemcpyHtoDAsync(chunk i, copy_stream)` using **two 16–32 MiB persistent pinned ring buffers** (never free/re-pin per load!).
     - Move `dc_unpack16` (if `DC_F_STATE16` is used) to a tiny GPU dequant kernel (`100 µs` on `gfx1201` @ 640 GB/s vs ~`40 ms` single-threaded CPU!).
     - **Target:** 181 MB cold disk resume drops from `1,110 ms` -> **`~180–300 ms`** (NVMe + PCIe overlapped streaming).

---

## 1.2 GPU Top-K / Top-P Sampling & Fast DRY Penalty (`0.44 ms/step -> < 0.06 ms/step`)

### Open-Source References & Papers
- **FlashInfer Sorting-Free Pivot Rejection Sampling & Multi-CTA Radix Top-K** — *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving* ([arXiv:2501.01005](https://arxiv.org/abs/2501.01005))
  - Repo: [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer)
  - Core files:
    - [`include/flashinfer/sampling.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/sampling.cuh) (`TopKTopPSamplingFromProb`, Dual-Pivot Rejection Sampling: initializes pivot=0, draws candidate via filtered block scan without $O(V \log V)$ sort, checks count $\le K$ / mass $\le P$, raises pivot to rejected candidate's prob -> converges in 4–6 passes!).
    - [`include/flashinfer/topk.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/topk.cuh) (Radix-based histogram top-K: 11-bit / 8-bit MSB radix passes over IEEE-754 float order-preserving bitkeys `key = (bits & 0x80000000) ? ~bits : (bits ^ 0x80000000)` to isolate the exact top-$K$ threshold in 2–3 linear passes!).
- **llama.cpp DRY Penalty & GPU Backend Sampling**
  - Repo: [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)
  - PR **#9702** (DRY sampler in [`src/llama-sampler.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp)), GPU backend sampling offload (`ggml_cuda_argsort` / top-k).

### Exact Root Causes in `amd-infer` (`src/gen.c:117-231, 1304-1360, 1933-1934, 3533`)
1. **Hidden `K > 64` CPU Fallback Cliff:** When a client omits `top_k` (setting `top_k=0` with `top_p=0.95`), `sample_k()` (`gen.c:45-47`) returns `1000u`. Both `sp->cv/ci` (allocated `16 * 64 * 4` at `gen.c:1933`) and `gen.c:3533` (`sample_k <= 64u`) reject `K > 64` -> triggers full `V=200,058` float D2H copy (`800 KiB × 6 verify rows = 4.8 MB/step`) + CPU `O(V log K)` min-heap!
2. **Even for `K = 20` (`serve-inkling2.ps1` default):** `b->topk` runs a multi-stage GPU reduction + separate D2H readback (`gpu_cands()` sync), and when a DRY Bloom-filter hit occurs on long outputs (`dry_n = 4K..20K`), `dry_apply()` (`gen.c:146-157`) does a **linear scan** over all `dry_h[S..dry_n]` tokens!

### Concrete `gfx1201` Wave32 Implementation Blueprints (Round 3)
1. **Recipe W1-D — Fused 1-Launch `gfx1201` Radix/Threshold Top-K Kernel (`V = 200,058`, `K <= 64` AND `K <= 1024`):**
   - **Algorithm (FlashInfer `topk.cuh` adapted to `gfx1201` wave32):**
     - **Case 1 (`K <= 32`, covers Inkling-2 default `top_k = 20`):** Two-stage fused kernel or single-pass 64-WG partial top-K + atomic last-WG merge (same pattern as `ie_inkling_rel_attn_ver_qtile_r6` self-resetting counter!):
       - Stage 1 (Grid `(64, T_rows, 1)`, `NT=256` = 8 waves, 1 `float4` coalesced load/lane): Each WG scans `200,058 / 64 = 3,126` logits (~12 iterations/thread). Maintain a per-wave register top-20 via **bitonic/insertion merge against a wave-wide `min_val` SGPR** (`__builtin_amdgcn_readfirstlane` of the 20th largest element!): **99.5% of logits fail `val > min_val` in 1 VALU compare and skip heap updates completely!**
       - Write `64 * K` (`64 * 20 * 8 B = 10 KiB`) into scratch; last WG (`atomicAdd == 63`) reduces those `1,280` candidates to exact sorted top-K (tie-break smaller `id` first!) in 1 wave using LDS bitonic sort and writes directly to a **pinned host-mapped output buffer** (`hipHostMalloc` zero-copy readback — eliminates `hipMemcpyDtoH` call overhead!).
       - **Bandwidth math:** `200,058 * 4 B = 800 KiB` read @ `600 GB/s` = **`1.3 µs/row`** (`~8 µs` for 6 verify rows; +~`15 µs` single-launch overhead = **`~25 µs/step`** vs `440 µs/step`!).
     - **Case 2 (`K = 65..1024`, i.e., `top_k=0, top_p<1`):** FlashInfer 2-pass **8-bit MSB radix histogram** over order-preserving `u32` float keys in LDS (`256 × u32 = 1 KiB` histogram/pass) finds the exact $K$-th threshold `thresh` in 2 passes, extracts only elements `>= thresh`, and runs FlashInfer's exact top-p nucleus cutoff on GPU!
2. **Recipe W1-E — $O(1)$ Amortized DRY Index (`src/gen.c:55-172`):**
   - Replace the linear scan `for (i = S+A-1; i < dry_n; i++)` in `dry_apply()` with a tiny open-addressed hash table / chained head-array indexed by the last 2-token bigram hash `h2 = (tok[p-1] * 2654435761u ^ tok[p]) & 4095u` (or `dry_hash` 12-bit bucket -> linked list `next_occ[pos]` of prior positions with the exact same `dry_allowed`-window hash!). On a Bloom hit, visit ONLY the 1–3 matching positions in `< 200 ns` instead of scanning 20,000 tokens. Keep DRY on the tiny top-K (`<= 64`) candidate list on CPU (`< 1 µs`) OR pass the `<= 8` `(id, penalty)` pairs into the GPU sampler kernel if full GPU sampling is enabled.

---

## 1.3 Chunked Prefill & `lm_head` Skipping

### Open-Source References
- **vLLM Chunked Prefill & Logit Skipping** (`vllm-project/vllm`): `vllm/v1/worker/gpu_model_runner.py` & `vllm/model_executor/models/` — skips `lm_head` (`output.weight`) projection on all intermediate chunks and on tokens `0 .. T-2` of the final chunk (only computing `[1, vocab]` for the last prompt token!).
- **Verify in `amd-infer` (`src/gen.c:1970, 3449-3457`):** `prefill_range` already passes `need_last = 0` for non-final chunks and runs the last token separately (`n_prompt - 1 .. n_prompt`). **Check `src/hip.c`:** Ensure intermediate chunks (`T=512`) also skip **final `rms_norm` + `lm_head`** completely (saves `~1.5 ms`/chunk if accidentally executed) AND check **optimal `--chunk` size at 16K–32K** (`chunk=512` vs `chunk=1024`: at `T=1024`, average MoE expert load rises from `32 -> 64` pairs/expert, doubling IQ3_XXS/IQ4_NL weight reuse per load — verify peak arena VRAM stays below WDDM cliff!).

---

# WAVE 2 (Rounds 6–10): Decode Launch-Reduction Fusions (`1,574 -> ~500–650` launches/step on `gfx1201`)

> **Why Launch Count Matters on Windows WDDM:** Each HIP kernel launch on WDDM adds `~2.5–3.5 µs` of host/command-buffer submission overhead (and inter-kernel L2 cache eviction). At `1,574` launches/step (`9.45 ms` wall vs `~5.2 ms` pure GPU memory time!), **~4.2 ms/step (~44% of T=1 wall time!) is launch overhead & inter-kernel stalls**. Cutting `1,574 -> ~550` launches recovers **`~2.5–3.2 ms/tok`** (`105.8 -> 140–155 tok/s` plain `T=1`!) AND trims `~3–5 ms` off every DSpark `T=6..8` verify pass!

### Open-Source References
- **ROCm/aiter** ([`github.com/ROCm/aiter`](https://github.com/ROCm/aiter)):
  - `aiter/ops/fused_qk_norm_rope_cache_quant.py` & `csrc/kernels/` — fuses QK-norm + positional transform + KV cache write (+ optional quant) in a single pass.
  - `aiter/fused_moe.py` & `csrc/kernels/` (`fused_topk`, `moe_sorting`) — fuses router gating + top-K selection + compact token-to-expert sorting.
- **vLLM & SGLang Fused MoE Pipelines:**
  - [`vllm-project/vllm`](https://github.com/vllm-project/vllm): `vllm/model_executor/layers/fused_moe/modular_kernel.py`, `fused_moe.py`, FlashInfer/AITER `kMoEFinalizeARResidualRMSNorm` fusion pass (fusing MoE down-projection combine + residual add + next-layer RMSNorm!).
  - [`sgl-project/sglang`](https://github.com/sgl-project/sglang): `sgl-kernel/` & `jit_kernel/` consolidated gate+topk+sort and fused residual+RMSNorm kernels.
- **Dao-AILab `causal-conv1d`** ([`github.com/Dao-AILab/causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d)):
  - `csrc/causal_conv1d_update.cu` (`causal_conv1d_update_kernel`, `kIsCircularBuffer`) — fuses circular state ring read + 4-tap depthwise conv dot product + bias + epilogue activation + in-place ring write into **1 single kernel launch** without shifting history elements!

### Concrete `gfx1201` Wave32 Fusion Blueprints (48 Logical Visits)

#### Recipe W2-A — Fused Router GEMV + Sigmoid/Softmax + Top-8 + Compact Sort (`3 -> 1` launch/visit = `-96` launches/step)
- **Inkling-2 Shape:** `ffn_gate_inp` is F32 `[128, 2304]`, `K = 8`, `E = 128`, `T = 1` (decode) or `T = 2..16` (verify).
- **`gfx1201` Wave32 Single-WG / Multi-WG Design:**
  - **For `T = 1` (single token):** Launch `Grid = (1, 1, 1)`? NO — `128 × 2304 × 4 B = 1.18 MB` needs all 64 CUs (`Grid = (64, 1, 1)` = 2 rows/WG, `NT=256`, 4 waves/row -> `1.18 MB / 600 GB/s = 2.0 µs`!) with a **self-resetting atomic last-WG epilogue** (`atomicAdd(done_cnt, 1) == 63`):
    - WGs `0..63` compute 2 router logits each (`r = 2*wgid + {0,1}`), write `scores[128]` into L2-hot global scratch (`512 B`!).
    - The last WG (wave 0..3, 128 lanes = **1 lane per expert!**) loads all 128 logits into registers, applies exact Inkling-2 router activation (sigmoid/softmax + bias), runs a **wave32 + 4-wave LDS bitonic / top-8 selection** (preserving exact tie-breaking: smaller expert ID wins!), normalizes the 8 routing weights, and writes `route_ids[8]`, `route_w[8]`, AND directly populates `offs[129]`, `list[8]`, `packs[8]`, `*n_packs = 8` (for `T=1`, 8 distinct active experts = 8 single-pair packs!).
  - **For `T = 2..16` (verify):** Grid `(64, T, 1)` or `(64, 1, 1)` looping over `T` tokens in registers (reads the `1.18 MB` F32 router weight **ONCE** across all `T` tokens instead of `T` times -> **6–8× router traffic cut!**), and the single last WG builds `route[T, 8]` + runs `ie_inkling_moe_sort_compact_r6`'s exact prefix-scan/pack emitter in LDS (`17 KiB`) — **eliminating both `OP_TOPK` and `ie_inkling_moe_sort_compact_r6` as separate launches!**
  - **Resource Budget:** `<= 64 VGPR`, `17.5 KiB LDS` (last WG only) / `0 LDS` for GEMV waves, `0` spills, exact bitwise parity!

#### Recipe W2-B — Fused `exp_down` (`IQ4_NL`) + Ordered 8-Slot Combine + Residual Add (+ Optional Next Pre-Norm/Q8) (`3..4 -> 1` launch/visit = `-96..-144` launches/step)
- **Inkling-2 Shape:** `ffn_down_exps` is `IQ4_NL [2304, 1152, 128]`, `nb = 36`, `K = 8`, `T = 1` (and `T = 2..16`).
- **Why Unfused Hurts:** Today, `exp_down` writes `8 × 2304 × 4 B = 73.7 KiB` (per token) to `exp_down` scratch, then `moe_sum` reloads all 8 slots, multiplies by `route_w[0..7]` in slot order `0..7`, sums + adds residual `x_res`, writes `x_out`, and then the next visit's `rmsnorm` + `quant_q8` reloads `x_out`!
- **`gfx1201` Wave32 Exact-Order Fusion (`T = 1`):**
  - Look at the geometry: `rows = 2304 = 72 × 32` rows. Launch `Grid = (72, 1, 1)` (`NT = 256` = 8 waves/WG, **1 wave per active expert `k = 0..7`**, each wave computes the SAME 32 output rows `r = wgid * 32 + lane` across `nb = 36` blocks of `IQ4_NL`!).
  - Wait — check bandwidth & CU utilization: `72` WGs across `64` CUs (`~1.125` WGs/CU) with `8` waves/WG = `256` threads/CU! Or `Grid = (144, 1, 1)` (`16` rows/WG, 2 waves/expert -> `2.25` WGs/CU, better CU balance!).
  - Inside each WG (`16` or `32` rows): waves `k = 0..7` compute their expert's exact FP32 dot product for those rows, store `lds_slot[k][r_local]` (`8 × 32 × 4 B = 1,024 B` LDS!), hit ONE `lbarrier()`, and then wave 0 reduces in **EXACT canonical slot order `k = 0 -> 7`**:
    ```c
    #pragma clang fp contract(off)
    float sum = 0.0f;
    for (u32 k = 0; k < 8u; k++) sum += lds_slot[k][r_local] * rw[k];
    float out_v = x_res[r] + sum;
    ```
    **100% BITWISE IDENTICAL to `exp_down` + `moe_sum` + residual add!** Zero `exp_down` global buffer write/read (`-147 KiB/visit` = `-7 MB/step` L2 traffic saved)!
  - **Bonus — Optional Next-Visit `attn_res` / `RMSNorm + Q8` Tail (`T=1`, vLLM/AITER `kMoEFinalizeResidualRMSNorm`):** Using the same atomic last-WG pattern (`atomicAdd(done_cnt, 1) == grid_x - 1`), the last WG reduces partial `sum_sq` from `lds -> global[72]` (just `288 B`!), computes `inv_rms`, and writes next visit's `norm_q8` — saving another **48–96 launches/step** (be careful to preserve exact `use_last` and check whether non-repeated vs repeated visits feed `attn_res` bank update first; for repeated visits, fuse residual add + bank write!).

#### Recipe W2-C — Fused SWA 4-Tap Short-Conv + Value-Clip + Ring-Store (+ Residual) (`3..4 -> 1` launch/visit = `-120` launches/step across 40 SWA visits)
- **Inspired by:** `Dao-AILab/causal-conv1d` (`causal_conv1d_update_kernel`, `kIsCircularBuffer`).
- **Inkling-2 SWA Op Sequence (40 visits):** After attention output / before projection, Inkling-2 runs `shortconv` (4-tap circular history ring) + `vclip` (`clamp`) + residual / ring update (plus `sconv_hist_r4` + `sconv_commit_r4` on verify!).
- **`gfx1201` Wave32 Design (`T = 1` and `T = 2..16`):**
  - `dim = 2304` (`72` Q8/F32 blocks of 32). Launch `Grid = (9, T, 1)` or `(18, T, 1)` (`NT = 256` = 1 channel per thread, or `Grid = (72, 1, 1)` for `T = 1` -> wait: `2304` channels is tiny, `Grid = (9, 1, 1)` @ `NT=256` = 1 channel/thread reads all 4 history taps `[pos-3, pos-2, pos-1, cur]` directly from the circular ring `ring[(pos - tap) & 3]` using `f32x4` coalesced loads, computes the 4-tap FMA in exact order, applies `vclip` in registers, writes output AND commits the new ring slot `(pos & 3)` in ONE single kernel!).
  - For verify `T = 2..16`: Keep a single fused `sconv_verify_clip_r7` kernel where 1 WG handles 256 channels across ALL `t = 0..T-1` tokens in registers (sliding the 4-tap register window `h0, h1, h2, h3` forward across `t = 0..T-1`!), writes all `T` clipped outputs AND writes the speculative history ring `sc_hist[t]` in **1 launch instead of 3**, with **0 redundant history reads**!
  - **Parity:** 100% bitwise exact (`#pragma clang fp contract(off)` matching `ie_inkling_sconv_r1` + `vclip`).

#### Recipe W2-D — Fused QKV Projection + Sink/Rel-Proj + KV-Cache Append (`4..6 -> 1..2` launches/visit = `-144..-192` launches/step)
- **Inspired by:** `ROCm/aiter` `fused_qk_norm_rope_cache_quant` & `FlashInfer` fused QKV+KV append.
- **Inkling-2 Attention Input Projections (per visit, all Q8_0 weights reading the EXACT SAME normalized `attn_in_q8`!):**
  - Look at `sota-plan.md` line 20:
    - `attn_q`: `[3072, 2304]` (`24 * 128`)
    - `attn_k`: `[1024, 2304]` (`8 * 128`)
    - `attn_v`: `[1024, 2304]` (`8 * 128`)
    - `attn_gate` / `rel_r` (`[384, 2304]` = `24 * 16`) + `sink` (`[24, 2304]`)!
  - Notice that on **40 SWA visits**, `Q`, `K`, `V`, `R`, `sink` ALL read the **exact same `attn_in_q8` vector (`2,880 B` — fits 100% in LDS / L1!)**, and `K` & `V` immediately get cast `f32 -> f16` and stored into `kc[pos % kv_len]` / `vc[pos % kv_len]`!
  - **Multi-Output Unified Q8_0 GEMV + Direct f16 KV-Store Kernel (`T = 1` and `T = 2..16`):**
    - Partition `wgid()` seamlessly across the concatenated output rows:
      - `wgid` in `[0 .. 95]`: `Q` (`3072` rows, 32 rows/WG) -> write `q_buf` (`f32`)
      - `wgid` in `[96 .. 127]`: `K` (`1024` rows) -> cast `(f16)` directly in registers & store straight to `kc[(pos + t) % kv_len]`! (Eliminates `OP_KV_STORE` launch AND intermediate `k_buf`!)
      - `wgid` in `[128 .. 159]`: `V` (`1024` rows) -> cast `(f16)` directly in registers & store straight to `vc[(pos + t) % kv_len]`!
      - `wgid` in `[160 .. 171]`: `R` (`384` rows) -> write `r_buf` (`f32`)
      - `wgid == 172`: `sink` (`24` rows) -> write `sink_buf` (`f32`)
    - **1 single launch** (`Grid = (173, 1, 1)` for `T=1`, `(173, 1, 1)` multi-token for `T=2..16`) replaces **5–6 separate launches per visit = `-192..-240` launches/step** across 48 visits!
    - **Global visits (8 visits):** On global visits with distinct `q_inp_norm`, `k_inp_norm`, `v_inp_norm` (`r6_4ctrl.inc` line 15), pass 3 input pointers `(xq_q, xq_k, xq_v)` — the exact same kernel dispatches cleanly by `wgid` range (`Q/R/sink` read `xq_q`, `K` reads `xq_k`, `V` reads `xq_v`)!
    - **Parity:** 100% bitwise exact vs individual Q8_0 GEMV + `f16` KV store!

---

# WAVE 3 (Rounds 11–15): Prefill WMMA GEMM & MoE on `gfx1201` (4K `4,560 -> 5,500+ tok/s`, 16K `3,740 -> 4,600+ tok/s`)

### Open-Source References & Papers
1. **AMD ROCm Official RDNA4 (`gfx1201`) WMMA GEMM Optimization Blogs & Code:**
   - Part 1: [*FP16 GEMM Optimization on AMD RDNA 3 and RDNA 4 Architectures*](https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp16-gemm/README.html)
   - Part 2: [*FP8 GEMM Optimization on AMD RDNA 4 Architecture*](https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp8-gemm/README.html)
   - Exact GitHub code: [`ROCm/rocm-blogs`](https://github.com/ROCm/rocm-blogs) (`blogs/software-tools-optimization/rdna-fp16-gemm/` and `rdna-fp8-gemm/`).
   - **Key `gfx1201` Tricks from AMD's RDNA4 Series:**
     - **XOR LDS Swizzle without Padding:** `swizzled_col = ((k / unit) ^ (row % blocks)) * unit` (zero padding overhead, zero LDS bank conflicts on both `ds_write_b128` and `ds_read_b128`).
     - **Ping-Pong Double-Buffered LDS + Deep K-Slicing:** Overlap `global_load_dwordx4` for `k+1` into LDS slot `buf^1` while issuing back-to-back `v_wmma_i32_16x16x16_iu8` / `v_wmma_f32_16x16x16_f16` out of `buf`, separated by `__builtin_amdgcn_sched_barrier(0)` and `s_setprio` / intrawave scheduling so LLVM doesn't hoist loads and blow VGPRs past 160!
     - **L2 Swizzled Tile Order (XCD/WGP Group Swizzle):** Remap `(bx, by)` in 4×4 or 8×4 macro-blocks so concurrent CUs share the same `X` activation / weight tiles in the RX 9070 XT's 64 MiB Infinity Cache (L3/MALL) + L2!
2. **ROCm Composable Kernel (`ck_tile`) for `gfx1201`:**
   - Repo: [`ROCm/composable_kernel`](https://github.com/ROCm/composable_kernel) (`include/ck_tile/ops/gemm/pipeline/`, `ck_tile::gfx12_t`).
   - **Key Trick:** Decoupled **intrawave** (compute-bound: 2-stage register prefetch + ping-pong LDS) vs **interwave** (memory-bound: smaller tile, higher occupancy 2–3 WGs/CU) pipeline selection.
3. **`ggml-org/llama.cpp` PR #32065 & `mmq.cuh`:**
   - Repo: [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)
   - **PR #32065** (*"CUDA: few-row MMQ for MUL_MAT_ID"*, Nov 2026) in `ggml/src/ggml-cuda/mmq.cuh`: Tightens dispatch for sparse MoE experts with small token counts (`1..16` tokens) via dedicated **few-row MMQ (`N=16` / `N=8` column tile)** bridging the gap between `MMVQ` and full `MMQ` (eliminating 2×–4× padded-zero WMMA waste on cold experts!).
4. **`ikawrakow/ik_llama.cpp` (`IQ3_XXS` / `IQ4_NL` MMQ):**
   - Repo: [`ikawrakow/ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) (`ggml/src/ggml-cuda/mmq.cuh`, `mmq_id.cu`, `load_tiles_mmq_t`).
   - **Key Trick:** Specialized `load_tiles` for `IQ4_NL` (4-bit non-linear LUT without codebook indirection) and `IQ3_XXS` (256-entry `iq3xxs_grid` codebook + sign bits), decoupling tile loading/dequantization into LDS from Tensor/WMMA dot products.
5. **FLUTE — LUT-Quantized GPU GEMM** (*Fast Matrix Multiplications for Lookup Table-Quantized LLMs*, [arXiv:2407.10960](https://arxiv.org/abs/2407.10960), [`HanGuo97/flute`](https://github.com/HanGuo97/flute)):
   - **Key Trick #1 (LDS LUT Bank-Conflict Elimination via Duplication):** `iq3xxs_grid` is only **`1 KiB` (256 × `u32`)**! When 32 lanes of a `gfx1201` wave simultaneously index random entries `lut[idx]` in a single 1 KiB LDS table, **average ~3–4-way LDS bank conflicts serialize every IQ3_XXS decode!** Replicating `lut` (or padding/bank-interleaving `lut[idx][copy]` across banks, or `4×` duplicated `4 KiB` LUT where lane `l` reads copy `l & 3`? Wait: on RDNA4 LDS has 32 banks of 4 bytes — duplicating `lut[256]` into **4 copies offset by bank**, or striding `lut[idx * 4 + (lane & 3)]`? Wait: let's check the exact bank math: bank = `(byte_addr >> 2) & 31`. If `lut` is `u32 lut[256]`, entry `idx` sits in bank `idx & 31`. Replicating `lut` DOES NOT change `idx & 31` unless different lane groups map to different bank subsets OR you use **2-byte / 4-byte vectorized sub-table**; in FLUTE for 4-bit LUTs (16 entries = 16 banks), 2 copies fill all 32 banks! For `IQ4_NL` (`16 × i8 = 16 B` LUT!), `ie_iq4_map` ALREADY uses **0-LDS `v_perm_b32` / register LUT** (`16` bytes fit in 4 SGPRs — **0 bank conflicts!**). For `IQ3_XXS` (`256 × u32`), check whether `ds_read_b32` random gather conflicts can be cut by **2× LUT interleaving by sub-half** or **register/L1 constant cache (`s_load` / global RO `__builtin_nontemporal` off)** vs LDS, AND check **FLUTE offline/load-time lane-aligned repacking**!).
   - **Key Trick #2 (Offline/Load-Time Weight Repacking — "FLUTE Lane-Aligned Packing"):** At GGUF load time in `src/hip.c` (`0 ms` runtime cost!), Inkling-2's `IQ3_XXS` (`780 MB` active, ~`6 GB` total) and `IQ4_NL` (`~4.5 GB`) weights can be repacked from 256-element super-block AoS layout into **lane-aligned SoA / 128-bit coalesced wave32 layout** (just like `ie_iq1m_grid8.inc` and Saluki's repacked formats!), eliminating unaligned `u32x4p` loads and bit-extraction shifts (`>> 9`, `& 0x1FF`, sign-bit scatter) inside every single prefill & verify K-loop!
6. **MegaBlocks / ScatterMoE / SonicMoE Dynamic Tile Worklist:**
   - **MegaBlocks** ([arXiv:2211.15841](https://arxiv.org/abs/2211.15841), [`databricks/megablocks`](https://github.com/databricks/megablocks)) — block-CSR sparse MoE GEMM.
   - **ScatterMoE** ([arXiv:2403.08245](https://arxiv.org/abs/2403.08245), [`shawntan/scattermoe`](https://github.com/shawntan/scattermoe)) — `ParallelLinear` / `scatter2scatter` fusing gather/scatter directly into the GEMM tile loop (no replicated token copy buffer!).
   - **SonicMoE** ([arXiv:2512.14080](https://arxiv.org/abs/2512.14080)) — IO- and tile-aware MoE scheduling with dynamic tile worklists and zero-padding tile partitioning.

### Concrete `gfx1201` Prefill Recipes (Rounds 11–15)

#### Recipe W3-A — Dynamic Flat Tile Worklist + `N=16` Few-Row Mode for Prefill `IQ3_XXS+SITU` (`r6_iq3_situ_pf.inc`) & `IQ4_NL` (`r2_g15.inc`)
- **Current Bottleneck:** In both `r6_iq3_situ_pf.inc` and `r2_g15.inc`, `Grid = (18, 128, 1)` assigns **1 WG per `(row_tile, expert)`** and loops serially over `ceil(cnt / 64)` token tiles!
  - Problem 1 (**Zipfian Tail Stragglers**): At `T=512, K=8, E=128`, hot experts get `96..180` pairs (2–3× serial K-loops!) while cold experts get `0..16` pairs and finish immediately, leaving CUs idle at the tail of every single MoE visit (48× per chunk!).
  - Problem 2 (**`N=32` Padding Waste on Cold Experts**, llama.cpp PR #32065): ~35–45% of active experts at `T=512` receive `cnt <= 16` tokens, yet `r6_iq3_situ_pf.inc` & `r2_g15.inc` run `WJ=2` (`32`-token tile) — doing **2× wasted WMMA instructions** on zero-padded columns!
- **Implementation:**
  1. Extend `ie_inkling_moe_sort_r1` (1 WG, `< 5 µs`) to also emit a **flat tile worklist `pf_tiles[max_tiles]`** (packed `u32`: `e[7:0] | lo[19:8] | nj_code[23:20]` where tile width is adaptively `64`, `32`, or **`16`** (`WJ=1`!), sorted **largest-tile-first** so 64-pair tiles launch first and 16-pair tiles fill tail slots — zero tail stragglers!).
  2. Dispatch `Grid = (18, max_tiles, 1)` (early-return if `by >= *n_tiles`, identical to `r6_compact_moe.inc`!) — every hot expert's 2–3 tiles run **in parallel across CUs**, AND every `<= 16`-pair expert runs a **1-tile (`WJ=1`) `128x16` micro-loop** (halving WMMA + LDS activation traffic + SITU epilogue cost, and cutting VGPRs to `<= 112`!).
  3. **Exact Parity:** 100% bitwise exact vs `r6_iq3_situ_pf.inc` & `r2_g15.inc` (same per-32 INT8 WMMA + FP32 FMA K-loop order!).

#### Recipe W3-B — `IQ3_XXS` Ping-Pong LDS & VGPR Trim (`<= 160` VGPR Guaranteed -> 2 WGs/CU) + Load-Time Weight Repack
- Audit `r6_iq3_situ_pf.inc`'s compiled VGPR count (R3's `g17` hit `178 VGPR` = 1 WG/CU!). Apply AMD RDNA4 blog's **strict 2-stage K-loop `sched_barrier(0)`** + **register-pressure-aware `IQ3_XXS` LUT**:
  - Test **load-time `IQ3_XXS` repacking** (FLUTE lane-aligned layout in `src/hip.c`, opt-in via env/flag first, 0 extra VRAM): pre-unpack the 9-bit indices & 7-bit sign masks into aligned `u16`/`u32` words per 16-row WMMA strip so the prefill K-loop issues **1 coalesced `u32x4` load per lane** with zero unaligned shifts!
  - Apply **L2/MALL 4×4 tile-swizzle** (`wgid` remap) on dense Q8_0 prefill GEMMs (`3072x2304`, `2304x3072`, `5120x2304` in `r2_g09.inc` / `ie_gemm_q8`) and test **ping-pong `128x64` WMMA (`v_wmma_i32_16x16x16_iu8_w32_gfx12`) with `KB=4` deep K-slice** from AMD's RDNA4 blog.

#### Recipe W3-C — Q4_K `output.weight` (`201,024 × 2,304`) Batched Verify GEMV (`T=2..8`)
- Look at `sota-plan.md` §4 Step 5.2: during DSpark `T=6..8` verify, `output.weight` (`260.5 MB` Q4_K) must be read **ONCE** across all `T` tokens (`0.42 ms` @ 620 GB/s), NOT `T` times (`2.5–3.4 ms`)! Audit whether `K_GEMV_Q4K_TR` / `r5_dspark_ver.inc` hits >= 85% peak bandwidth at `T=6..8`; if not, add a `gfx1201` wave32 4-row/wave `T=2..8` register-blocked `Q4_K` verify GEMV (matching Saluki's `ie_saluki_q2k_tr8_r4` / `iq1s` pattern: 1 weight decode shared across `T` token `Q8_1` dot4s via `ie_dot4_ss`).

---

# WAVE 4 (Rounds 16–20): Long-Context 16K–32K Prefill & Verify Attention on `gfx1201`

### Open-Source References & Papers
1. **ROCm/aiter & FlyDSL `fmha_gfx1201` + `ck_tile` FMHA:**
   - [`ROCm/aiter`](https://github.com/ROCm/aiter) (`triton_attn`, `flydsl_flash_attn_func`, Issue `#4329`: warns that 3D unified attention on Wave32 `gfx1201` can blow the 64 KiB LDS limit if `NB`/head-dim tile isn't bounded -> stay `<= 24–32 KiB` LDS for 2–3 WGs/CU!).
   - [`ROCm/composable_kernel`](https://github.com/ROCm/composable_kernel) (`include/ck_tile/ops/fmha/`, `gfx12_t` WMMA `v_wmma_f32_16x16x16_f16` attention).
2. **FlashInfer Cascade / Split-KV & FlashDecoding++:**
   - **FlashInfer** ([arXiv:2501.01005](https://arxiv.org/abs/2501.01005), [`flashinfer-ai/flashinfer/include/flashinfer/attention`](https://github.com/flashinfer-ai/flashinfer/tree/main/include/flashinfer/attention)):
     - **Split-KV** for both decode/verify AND long-context incremental prefill (`pos0 >> T`): splits KV sequence length across `S` WGs, writes `{m, l, O}` partials (`520 B`/slot), merges with a single reduction or last-WG atomic combine.
     - **Cascade Inference:** Separates shared/far-field prefix KV from local suffix KV so each region uses its optimal tile & occupancy!
   - **FlashDecoding++** ([arXiv:2311.01282](https://arxiv.org/abs/2311.01282)):
     - **Unified-Max / Lazy Softmax Rescaling:** Amortizes `o *= exp2(m_old - m_new)` across larger K-blocks (`NB=32/64`) instead of rescaling all 64 output accumulators every 16 keys!
3. **`ggml-org/llama.cpp` RDNA4 FlashAttention (`wave32`):**
   - [`ggml/src/ggml-cuda/fattn-wmma-f16.cu`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-wmma-f16.cu) & `fattn-mma-f16.cuh` & `fattn-vec.cuh` (WMMA `wave32` tiling on `gfx1201`, `kq_stride = 32/64`, split-K auto-selection when `n_tiles < n_cu`).

### Exact Audit of Current Inkling-2 Attention (`ie_inkling_r1_g01.inc`, `r3_g20.inc`, `r6_rel_attn_qtile.inc`)
1. **Why 16K–31.6K Global Prefill Takes `4.1–4.4 s`:**
   - Current promoted prefill attention is `r1_g01` (**174 VGPR, 32.1 KiB LDS = 1 WG/CU!**; `r3_g20` bit 41 is OFF in `0x67dfc0369c3ffff`).
   - `NB = 16` forces **`2,048` `lbarrier()`s + `2,048` online-softmax rescale loops** per WG at 32K context!
   - Inner producer loop executes **`p % kv_len` integer modulo** 12 times per 16-key block (`24,576` integer divisions per WG at 32K!), even though on global layers (`win == 0`) `p < kv_len` ALWAYS (0 modulo needed!) and on SWA layers `kv_len == 1024` (`p & 1023u` = 1-cycle AND!).
   - Zero Split-KV in prefill: at incremental chunk `T = 64..256` with `pos0 = 24K`, grid is only `(2..8, 8, 1) = 16..64` WGs -> **half the 64 CUs sit idle** while each WG walks 24K keys serially!
2. **Verify Attention (`r6_rel_attn_qtile.inc`) Headroom:**
   - Already cuts 24K global verify by `-14..-18%` via `QT=4` (`12` queries sharing 1 K/V load).
   - **Next wins:** (a) Support `QT=6` (exact fit for DSpark `5+1 = 6` tokens in **1 single pass** instead of 2 tiles of `4 + 2` -> cuts global K/V reads another **33%** on default `ds5`!!), OR (b) **WMMA-based `T=6..8` Verify Attention** (pack `6 × 3 = 18` -> `16` or `2×16` queries into 1–2 `16x16x16` FP16 WMMA tiles against `NB=16` keys — converts `r6_rel_attn_qtile`'s scalar FMA `QK3` / `UPD` loops into native RDNA4 matrix cores!).

### Concrete `gfx1201` Wave32 Attention Recipes (Rounds 16–20)
1. **Recipe W4-A — Far-Field / Near-Field Split-Path `NB=32` Global Prefill Attention (`r7_attn_pf_global`):**
   - For `win == 0` (8 global visits): split the KV range `[0 .. kend)` into:
     - **Far-field fast loop** `[0 .. max(0, qpos_min - 1024))` (`94–97%` of all blocks at 16K–32K!): **ZERO `rel_bias`, ZERO causal mask, ZERO `p % kv_len` modulo!** Drop `rb` + `rbsc` from LDS completely (`-15.7 KiB` LDS -> fits `NB = 32` double-buffered K/V in `<= 24 KiB` LDS or `16 KiB` single-stage + register prefetch, `<= 128 VGPR`, **2–3 WGs/CU**!). Process **32 keys per `lbarrier()` and per softmax rescale** (2× `16x16` QK WMMAs -> 1 unified `max`/`exp2`/rescale -> 2× `PV` WMMAs -> **2× fewer barriers & rescalings**!).
     - **Near-field tail loop** `[max(0, qpos_min - 1024) .. kend)` (`<= 64` blocks): exact `R * rel_bias` + causal mask (identical to `r3_g20`).
2. **Recipe W4-B — Prefill Split-KV for Long Incremental Context (`pos0 >= 4096 && win == 0`):**
   - Whenever `((T + 31)/32) * n_kv < 128` (e.g., `T <= 256`, or even `T = 512` at `pos0 >= 16K` with `nsplit_kv = 2..4`), split the far-field KV range across `nsplit_kv` WGs writing `{M, L, O[128]}` partials into existing `at_part` scratch + atomic self-resetting last-WG merge (exact same mechanism as `r6_rel_attn_qtile`!). Saturates all 64 CUs even on 64-token incremental prefill at 31.6K context!
3. **Recipe W4-C — SWA Bounded-Window Fast Path (`win == 1024`, 40 visits):**
   - Replace `p % kv_len` with `p & (kv_len - 1u)` when `(kv_len & (kv_len - 1u)) == 0u` (guaranteed power-of-two ring!).
   - Bound producer loads strictly to `[qpos_min + 1 - 1024 .. qpos_max + 1)` (`<= 66` blocks for `T=32`), keep `r3_g20`'s `<= 144 VGPR` / `22.4 KiB LDS` (2 WGs/CU), and promote bit 41 (or improved `r7_attn_pf_swa` with power-of-two mask + `NB=32`)!
4. **Recipe W4-D — `QT=6` / WMMA `T=6..8` Verify Global Attention:**
   - Extend `ie_inkling_rel_attn_ver_qtile_r6` from `QT=4` to **`QT=6` (`18` queries = 6 tokens × 3 GQA heads)** or **`WMMA 16x16x16` (`QT=5..8`)** for DSpark `5x0.6` (`T=6`) and `7x0.5` (`T=8`) verify -> reads the 16K–32K global KV cache **ONCE per step** instead of twice (`-33%..-50%` global attention verify bandwidth!).

---

# WAVES 5–7 (Rounds 21–40): Speculative Decoding Breakthroughs & Sub-Span Persistent / Piecewise Graphs

> **Highest Single-Round ROI in the Entire 40-Round Campaign:** Wave 5 Recipe W5-A (Probabilistic DSpark Proposals) directly fixes the live telemetry bug causing sampled chat acceptance to collapse to `1.63–2.20 tok/step`!

## 5.1 Probabilistic DSpark Proposals + Confidence Scheduling (Wave 5, Rounds 21–25)
- **Papers & Code:**
  - **Speculative Sampling** — Leviathan et al. ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192)), Chen et al. ([arXiv:2302.01318](https://arxiv.org/abs/2302.01318))
  - **DSpark** — *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation* ([arXiv:2607.05147](https://arxiv.org/abs/2607.05147), [`deepseek-ai/DeepSpec`](https://github.com/deepseek-ai/DeepSpec))
  - **ForeSpark** — *ForeSpark: Joint Draft and Verification Budget Scheduling for Parallel Speculative Decoding* ([arXiv:2611.13442](https://arxiv.org/abs/2611.13442), [`facebookresearch/ForeSpark`](https://github.com/facebookresearch/ForeSpark))
  - **vLLM Rejection Sampler** — [`vllm/v1/sample/rejection_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py) (`draft_probs` $q(x)$ vs `NO_DRAFT_PROBS` fallback!)
- **Exact Root Cause & Fix in `amd-infer` (`src/gen.c:1621-1648` & `1304-1442`):**
  - `spec_rej_step()` ALREADY implements exact lossless Leviathan rejection (`min(1, p(x)/q(x))` + residual `norm(max(0, p - q))`) whenever `qnode[j].n > 0`!
  - **Bug:** `src/gen.c:1646` executes `memset(qnode, 0, sizeof qnode)` on the DSpark (`dsm`) branch, forcing point-mass rejection (`P(accept) = p(argmax)` -> capped at `~0.35` at `temp=1.0`, `~1.54` tok/step even for a perfect drafter!).
  - **Recipe W5-A (Lossless Probabilistic DSpark Proposal):**
    1. When `sp->temp > 0.0f` and `rej_on`: in DSpark's Markov draft loop (`hip.c` `run(b, 3, ...)`), run GPU top-$K$ (`K = sample_k(sp, V)`, e.g. `20`!) on each draft step's logits (with softcap `30.0`), temperature-scale + nucleus-truncate `q(x)` with the **exact target sampler params `(temp, top_k, top_p)`** (or slightly sharper draft temp `T_draft = 0.85..1.0 * T`, mathematically lossless for ANY `q(x)` as long as `qnode[j]` records the true sampling distribution `q(x)` used to draw `d[j+1]`!).
    2. Sample `d[j+1] ~ qnode[j]` (feeding `d[j+1]` into the next Markov RNN step!), and pass `qnode[0..nd-1]` straight into `spec_rej_step()`!
    3. **Expected Gain:** Acceptance jumps from `1.63–2.20` -> **`3.5–5.0+ tok/step`** on `temp=1.0` sampled chat (**+50–100% live `ie-serve` chat tok/s!**) while staying **100% mathematically identical** to the target distribution $p(x)$ (verify via chi-squared / KLD <= 0.001 + exact greedy SHA when `temp=0`)!
  - **Recipe W5-B (DSpark/ForeSpark Adaptive Survival-Probability Cutoff):** Replace fixed scalar `pmin = 0.6` with **cumulative survival-probability budgeting**: stop drafting at depth $j$ when $\prod_{i=1}^j \hat{\alpha}_i < \tau_{\text{hw}}(j)$ where $\tau_{\text{hw}}(j) = \frac{t_{\text{draft}} + t_{\text{ver}}(j+1) - t_{\text{ver}}(j)}{t_{\text{ver}}(j)}$ (marginal cost of verifying 1 extra token vs expected accepted tokens, ForeSpark `arXiv:2611.13442`!).

## 5.2 Certified Frequency-Ranked & In-Context Vocab Shortlist for DSpark `lm_head` (Wave 5/6)
- **Papers & Code:**
  - **FR-Spec** ([arXiv:2502.14856](https://arxiv.org/abs/2502.14856), [`thunlp/FR-Spec`](https://github.com/thunlp/FR-Spec)) — static frequency-ranked shortlist (e.g., top `32K` or `48K` tokens out of `200K` -> **4–6× smaller draft `lm_head`!**), **100% lossless** because target still verifies full `200,058` vocab!
  - **MicroSpec** ([arXiv:2605.26444](https://arxiv.org/abs/2605.26444)) — lightweight **in-context vocabulary** (union of top-`K_freq` static tokens + unique tokens appearing in the current prompt/conversation -> captures rare code identifiers & domain terms that naive static frequency lists miss!).
  - **SpecPilot** ([arXiv:2611.02501](https://arxiv.org/abs/2611.02501)) — ceiling-guided autonomous system tuning for draft head & verify formats.
- **Recipe W5-C (Hybrid FR-Spec + MicroSpec Draft Head for Inkling-2):**
  - Recall `sota-plan.md` §3: naive `0..32767` truncation failed because Inkling-2's BPE puts common code tokens at high IDs, AND Phase 6 already cut `ds_draft` from `4.17 -> 1.83 ms` via split `Q4_K`/`Q4_0` `P`!
  - Build a **48K/64K frequency-ranked `id_map[65536]`** (ranked on code + chat corpus using Inkling-2's actual tokenizer) + **dynamic prompt token union** (MicroSpec: append up to `4K` unique prompt tokens not in top-`60K`, mapped into a compact `64K × 2304` Q4_K / Q4_0 draft head -> **`80 MB` instead of `260 MB` = `0.13 ms/step` draft `lm_head`, cutting total `ds_draft` to `< 0.9 ms`!**), 100% exact lossless parity!

## 5.3 Hybrid Prompt N-Gram / Suffix-Tree Lookup + Tiny Dynamic Tree Verify (Wave 6)
- **Papers & Code:**
  - **Prompt Lookup Decoding (PLD)** ([`apoorvumang/prompt-lookup-decoding`](https://github.com/apoorvumang/prompt-lookup-decoding)) & **REST** ([arXiv:2311.08252](https://arxiv.org/abs/2311.08252))
  - **SuffixDecoding** ([arXiv:2411.04975](https://arxiv.org/abs/2411.04975), [`snowflakedb/ArcticInference`](https://github.com/snowflakedb/ArcticInference)) — CPU suffix-tree matcher scaling draft length dynamically on agentic/code-edit/tool-call repetitions at ~`0.02 ms` draft cost!
  - **EAGLE-2** ([arXiv:2406.16858](https://arxiv.org/abs/2406.16858), [`SafeAILab/EAGLE`](https://github.com/SafeAILab/EAGLE)), **Sequoia** ([arXiv:2402.12374](https://arxiv.org/abs/2402.12374), [`Infini-AI-Lab/Sequoia`](https://github.com/Infini-AI-Lab/Sequoia)), **SpecInfer** ([arXiv:2305.09781](https://arxiv.org/abs/2305.09781)) — dynamic confidence-pruned tiny token trees (`8–12` nodes, e.g., branching top-2 at depths 1–2 when top-1 prob `< 0.55`).
- **Recipe W6-A (Suffix-Tree Hybrid with DSpark):** `src/gen.c:1634-1643` ALREADY hooks `lookup_draft()` after DSpark, **but only if `nd >= lk_agree` and DSpark ran first!** Upgrade to **SuffixDecoding pre-check**: when the last `>= 5` tokens have an exact high-confidence suffix match in the prompt/session history (common in code refactoring & tool JSON!), bypass DSpark drafting completely (`0 ms` draft!) and submit up to `12–15` suffix tokens directly to verify!
- **Recipe W6-B (Tiny 8-Node EAGLE-2/Sequoia Tree Verify for Inkling-2):** `amd-infer` ALREADY has `tree_par`, `accept_tree`, and `spec_rej_step` multi-child tree verification (`src/gen.c:1776-1824`)! Adding a `tree_mask` bitmask (8-bit `u8` ancestor mask fits in 1 SGPR!) to `ie_inkling_rel_attn_ver_r4` / `qtile_r6` and `sconv_hist_r4` lets DSpark verify a **2-branch × depth-4 (8-node) dynamic tree** at the exact same `T=8` verify cost (`~17.9 ms`), boosting sampled `temp=1.0` acceptance another **+15–25%**!

## 5.4 Piecewise HIP Graphs & Low-VGPR Sub-Span Persistent Kernels (Wave 7)
- **Papers & Code:**
  - **vLLM Piecewise CUDA/HIP Graphs** ([`vllm/compilation/cuda_piecewise_backend.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/cuda_piecewise_backend.py)) — splits the graph around dynamic-position ops (attention/KV-pos) and captures static sub-segments (MoE + Loop Controllers + Norms) into reusable HIP graphs (`FULL_AND_PIECEWISE` mode), bypassing WDDM's `~1.99 ms` full-graph re-instantiation/update overhead!
  - **Mirage Persistent Kernel (MPK)** — *MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs* ([arXiv:2512.22219](https://arxiv.org/abs/2512.22219), [`mirage-project/mirage`](https://github.com/mirage-project/mirage)) — SM/CU-level fine-grained task graph with decentralized in-kernel scheduling and cross-operator software pipelining.
- **Recipe W7-A (Why Full `IE_MK=1` Regressed -13% in Phase 6, and How Sub-Span Persistent Wins on `gfx1201`):**
  - Phase 6 showed monolithic `IE_MK=1` dropped bandwidth `596 -> 276 GB/s` because uniting attention + MoE + loop controllers in ONE mega-kernel blew VGPRs (1 WG/CU) and forced grid-wide spin barriers!
  - **Sub-Span Persistent Alternative (MPK-style):** Instead of one monolithic 48-visit kernel, build **2 tiny, ultra-low-VGPR (`<= 64 VGPR`, 4 WGs/CU, `grid <= 64`) sub-span kernels**:
    1. **Persistent MoE Sub-Span (`Router -> Top8 -> IQ3_XXS+SITU -> IQ4_NL+Combine+Residual`)**: 1 single launch per visit (`48` launches total instead of `240+`!) using `grid = 64` (1 WG/CU, WDDM-safe with wall-clock guard!) and fast L2-resident scratch (`exp_mid_q8` is only `11.5 KiB` for `T=1` — fits 100% in L2!).
    2. **Piecewise HIP Graph (`hipGraphExecKernelNodeSetParams` ONLY on attention `pos` + `sconv` `pos` — or static-param indirect `pos_dev` pointer in VRAM!):** In Phase 2, `IE_INKLING_GRAPH=1` paid `~1.99 ms` WDDM CPU replay overhead partly from parameter updates/graph splitting. Testing **1 device-side `u32 *d_pos` indirect pointer** (updated via 1 tiny 4-byte `hipMemcpyHtoDAsync` or in-kernel `d_pos++`!) enables **zero-update HIP graph replay** (or 8-visit piecewise chunks) — benchmark against Wave 2's fused eager baseline (`~550` launches) and promote ONLY if wall-clock wins on WDDM!

---

## Quick-Reference Master Priority Table (Waves 1–7)

| Wave | Rounds | Target Bottleneck | Top Concrete Blueprint | Primary Reference | Est. Impact |
|---|---|---|---|---|---|
| **W1** | R1–R2 | 152-tok side req evicts 26.8K chat (`4.4s` re-prefill) | **W1-A** Prefix-safe side-request swap (`<10 ms`, 0 VRAM) + **W1-B** `ck_limit` lift + SLRU | SGLang `radix_cache.py` / `evict_policy.py` ([arXiv:2312.07104](https://arxiv.org/abs/2312.07104)) | **4,400 -> 10 ms TTFT** on side-req resume |
| **W1** | R3–R4 | `0.44 ms/step` CPU sampling + DRY; `1.11s` disk load | **W1-D** 1-launch `gfx1201` wave32 radix/threshold top-K (`K<=1024`) + **W1-E** $O(1)$ DRY index + **W1-C** double-buffered pinned disk-DMA | FlashInfer `sampling.cuh`/`topk.cuh` ([arXiv:2501.01005](https://arxiv.org/abs/2501.01005)), LMCache ([arXiv:2510.09665](https://arxiv.org/abs/2510.09665)) | **-0.38 ms/step** decode; disk load **1,110 -> ~250 ms** |
| **W2** | R6–R10 | `1,574` launches/step (`~4.2 ms` launch tax on WDDM) | **W2-A** Router+Top8+Sort (`-96`), **W2-B** `IQ4_NL`+Combine+Res (`-96..-144`), **W2-C** ShortConv+Clip+Ring (`-120`), **W2-D** Unified QKV+R+Sink+f16-KV (`-192..-240`) | `ROCm/aiter`, `vLLM` modular MoE, `Dao-AILab/causal-conv1d` | **-550..-650 launches/step** (`T=1`: **106 -> ~135–145 tok/s**) |
| **W3** | R11–R15 | 4K/16K MoE prefill stragglers, `N=32` pad waste, VGPRs | **W3-A** Flat dynamic tile worklist + `N=16` few-row MMQ + **W3-B** RDNA4 ping-pong XOR LDS & `<=160` VGPR + load-time repack | AMD RDNA4 GEMM blogs, `llama.cpp` PR #32065, FLUTE ([arXiv:2407.10960](https://arxiv.org/abs/2407.10960)), SonicMoE ([arXiv:2512.14080](https://arxiv.org/abs/2512.14080)) | **+15–25%** 4K/16K prefill (`4.56K -> 5.4K+ tok/s`) |
| **W4** | R16–R20 | 16K–32K global prefill (`4.4s`), `p%kv_len`, `NB=16`, `QT=4` | **W4-A** Far-field 0-`rb` `NB=32` WMMA + **W4-B** Prefill Split-KV + **W4-C** SWA `& (kv_len-1)` + **W4-D** `QT=6`/WMMA verify | `ROCm/composable_kernel` `ck_tile`, FlashInfer ([arXiv:2501.01005](https://arxiv.org/abs/2501.01005)), FlashDecoding++ ([arXiv:2311.01282](https://arxiv.org/abs/2311.01282)) | **-25–40%** 16K–32K prefill & verify attention |
| **W5** | R21–R25 | Sampled `temp=1.0` DSpark accept `1.63–2.20 tok/step` | **W5-A** Probabilistic DSpark `q(x)` proposals into `spec_rej_step()` + **W5-B** ForeSpark survival cutoff + **W5-C** FR-Spec/MicroSpec 64K draft head | Leviathan ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192)), DSpark ([arXiv:2607.05147](https://arxiv.org/abs/2607.05147)), ForeSpark ([arXiv:2611.13442](https://arxiv.org/abs/2611.13442)), FR-Spec ([arXiv:2502.14856](https://arxiv.org/abs/2502.14856)), MicroSpec ([arXiv:2605.26444](https://arxiv.org/abs/2605.26444)) | Sampled chat **+50–100% tok/s** (`130 -> 195–240+ tok/s`), 100% lossless! |
| **W6–W7** | R26–R40 | Code-edit suffix hits, sampled tree depth, WDDM dispatch | **W6-A** SuffixDecoding pre-check, **W6-B** 8-node EAGLE-2/Sequoia tree verify, **W7-A** low-VGPR MoE sub-span / indirect-pos HIP graphs | SuffixDecoding ([arXiv:2411.04975](https://arxiv.org/abs/2411.04975)), EAGLE-2 ([arXiv:2406.16858](https://arxiv.org/abs/2406.16858)), Mirage MPK ([arXiv:2512.22219](https://arxiv.org/abs/2512.22219)), vLLM `cuda_piecewise_backend.py` | Code-edit **250–350+ tok/s**; sampled chat **+15–25%** further |
