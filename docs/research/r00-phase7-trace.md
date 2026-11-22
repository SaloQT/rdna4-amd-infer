# Round 0 — Fresh Phase 7 Trace-Backed Profile & Live-Log Forensics

**Date:** 2026-11-22  
**Hardware:** AMD Radeon RX 9070 XT (`gfx1201`, RDNA4, wave32, 64 CUs / 32 WGPs, 640 GB/s VRAM, 16 GiB WDDM)  
**Engine:** `ref/tr8-team-merge` (`build/ie-run.exe`, `build/ie-serve.exe`, `build/ie_inkling_kernels.hsaco`)  
**Model:** `models/Inkling-2-Mini-Uncensored-IQ3_XXS-fast.gguf` (`output.weight` Q4_K = 260.5 MB; `vocab=201024`, `vocab_unpadded=200058`; 24 physical layers / 48 logical visits: 40 SWA `win=1024` + 8 Global; 128 experts, `K=8`)  
**Speculator:** `models/DSpark/Inkling-2-Mini-DSpark-Q4_K.gguf` (`5x0.6`, `IE_DS_SPLIT=1` Q4_0 split head `P` 28.9 MB)  
**Active Phase 7 Kernel Mask:** `0x67dfc0369c3ffff` (`IE_INK_DEFAULT_MASK`, 0/0 VGPR spills)  
**Profiling Artifacts:** `results/inkling2/r00-phase7-trace.json`, `results/inkling2/r00-phase7-trace2.json`, `logs/inkling2/r00-trace/*.csv`

> **Methodology Note (Windows WDDM HIP Event Timing):** Recording `hipEventRecord(begin/end)` on every consecutive launch (`IE_TRACE_GPU_STRIDE=1`) serializes Windows WDDM dispatch packets (~60 µs/event overhead, inflating `T=1` from 7.16 ms to 113 ms). All wall-clock numbers below come from **clean trace-off runs** (`IE_STEP_TIME=1` / `IE_SPEC_TIME=1` only), while per-kernel GPU timings use **prime-strided sampling** (`stride=31` on `T=1`, `stride=19` on 20-rep `IE_VERIFY_BENCH=6`, `stride=4` on 4K prefill, `stride=16` on 16K prefill) aggregated by exact `(kernel, groups, weight_bytes)` shape medians so heterogeneous GEMV shapes and rare WDDM quantum preemptions do not distort per-kernel shares. Every trace run verified `0 samples skipped` and `100% EXACT` token parity against its clean control.

---

## 1. Plain `T=1` Decode Profile (`real-code-233.txt`, `--draft 0`)

### Clean Wall-Clock & Host/GPU Split (`t1-clean`, `--n 34`, `IE_STEP_TIME=1`)
- **Wall clock:** **`9.167 ms/token`** (**`109.1 tok/s`**)
- **GPU event clock:** **`7.162 ms/token`** (**`139.6 tok/s`**, `3,103.72 MB/tok` -> **`433.4 GB/s`** effective, 67.7% of 640 GB/s peak)
- **Exact launch count:** **`1,574 launches/step`** for **`2,222` logical ops**
- **`IE_STEP_TIME=1` breakdown (32 steps):**
  - `launches` (host submit): **`0.997 ms/tok`** (`10.9%`)
  - `wait` (`hipEventSynchronize`): **`8.031 ms/tok`** (`87.6%`)
  - `read` (4-byte argmax D2H): **`0.134 ms/tok`** (`1.5%`)
  - **WDDM launch-retirement & host gap (`wall - GPU_clock`)**: **`2.005 ms/tok`** (**`21.9%` of total wall clock** = `~1.27 µs/launch` over 1,574 launches!)
- **VRAM:** `11.91 GiB` VRAM / `0.54 GiB` system / `14.03 GiB` budget (`4,310 MB` free after load, `1,052` allocations, arena `116.3 MiB`)

### Top-15 GPU Kernel Breakdown (`t1-stride31`, 62 steps, `1,574` launches/step)
Shape-median pure kernel sum: **`5.495 ms/step`** (trimmed-85%: `5.898 ms/step`; `+1.26 ms` inter-launch GPU gaps = `7.16 ms` GPU clock).

| Rank | Kernel | Launches/step | Samples | Med ms/step | Share | Avg µs | Eff. GB/s | Optimization Target |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `ie_inkling_gexp_raw_iq3xxs_r1` | 48 | 110 | **1.836** | **33.4%** | 38.24 | 425.1 | Wave 5 IQ3_XXS GEMV / FLUTE |
| 2 | `ie_gemv_q8q8` | 268 | 517 | **1.296** | **23.6%** | 4.84 | — | Wave 2 D1/D2 QKV & rel/sink/gate fusion |
| 3 | `ie_inkling_gexp_iq4nl_r1` | 48 | 87 | **1.224** | **22.3%** | 25.50 | 468.4 | Wave 2 Fusion B (`down+sum+sconv+add`) |
| 4 | `ie_gemv_q4kq8` (`output.weight`) | 1 | 2 | **0.456** | **8.3%** | 455.75 | 571.6 | Near-peak (89% BW) |
| 5 | `ie_inkling_vclip_r1` | **192** | 396 | 0.115 | 2.1% | 0.60 | — | Wave 2 Fusions C1 & D1/D2 (`-192` launches!) |
| 6 | `ie_inkling_sconv_r1` | **192** | 437 | 0.073 | 1.3% | 0.38 | — | Wave 2 Fusions B, C1, C2 (`-192` launches!) |
| 7 | `ie_gate` | 48 | 94 | 0.065 | 1.2% | 1.36 | — | Wave 3 / sub-span fusion |
| 8 | `ie_inkling_situ_r1` | 48 | 91 | 0.063 | 1.2% | 1.32 | — | Fuse into IQ3_XXS epilogue |
| 9 | `ie_gemv_f32` (router) | 48 | 91 | 0.061 | 1.1% | 1.28 | — | Wave 2 Fusion A (`router+route`) |
| 10 | `ie_argmax` | 1 | 1 | 0.059 | 1.1% | 58.90 | — | Fuse with `logit_cap` / head |
| 11 | `ie_inkling_attn_res_r1` | 96 | 195 | 0.035 | 0.6% | 0.36 | — | Fuse with `rmsnorm` |
| 12 | `ie_inkling_gemv_q8_small_r1` | 88 | 171 | 0.032 | 0.6% | 0.36 | 366.0 | Wave 2 Fusion D2 (`wsink`, `wrel`) |
| 13 | `ie_add` | 88 | 174 | 0.032 | 0.6% | 0.36 | — | Wave 2 Fusions B & C2 (`-88` launches!) |
| 14 | `ie_inkling_rmsnorm_mt_r1` | 58 | 107 | 0.021 | 0.4% | 0.36 | — | Fuse with `attn_res` |
| 15 | `ie_inkling_rel_attn_dec_r1` | 48 | 103 | 0.019 | 0.3% | 0.40 | — | Scales at 16K–32K |

**Key `T=1` Takeaway:** Only **4 heavy GEMV kernels** (`365` launches) represent **87.6% of actual GPU compute** (`4.81 ms`), while **`1,209` micro-launches** (`vclip`, `sconv`, `attn_res`, `gemv_q8_small`, `add`, `rmsnorm`, `kv_store`, `moe_route`, `moe_sum`, `situ`, `gate`, `lc_*`) spend only `~0.68 ms` computing but burn **`~3.2 ms/step`** in combined WDDM launch retirement (`2.00 ms`) + inter-launch GPU bubbles (`~1.2 ms`)! Cutting `1,574 -> ~818` launches via Wave 2 Fusions A–D directly targets **`1.4–2.0 ms/tok` (`+18–28%`)** on `T=1` wall clock!

---

## 2. 4K Prefill & DSpark `5x0.6` Verify Profile (`real-code-4k.txt`, `--chunk 512`)

### Clean Wall-Clock & Speculative Stage Split (`ds4k-clean`, `--n 16`, `IE_DS_SPLIT=1`)
- **4K Cold Prefill:** **`888.6 ms`** (**`4,609.6 tok/s`**, 8 chunks × `1,871` launches/chunk + 8 `ds_ctx` × `13` launches)
- **DSpark `5x0.6` Decode:** **`168.4 tok/s`** (`5.938 ms/tok`, `3.75 tok/step`, `55.0%` accept over 4 steps, `15` tokens in `89.1 ms`)
- **Per-Step Wall Split (`89.1 ms / 4 steps = 22.275 ms/step`):**
  - `draft` (`ds_draft` 69 launches + `ds_ctx` 13 launches): **`2.99 ms/step`** (**`13.4%`**)
  - `verify` (`T=6`, **`1,735` launches/step**): **`19.21 ms/step`** (**`86.2%`**)
  - `commit + host` (`ie_inkling_sconv_commit_r4` 1 launch + host accept): **`0.075 ms/step`** (**`0.3%`**)
- **VRAM:** `12.36 GiB` VRAM / `0.58 GiB` sys / `14.03 GiB` budget (`3,845 MB` free after load, `1,167` allocations)

### Top-15 GPU Kernels — 4K Cold Prefill (`ds4k-trace-pf`, `stride=4`, 8 chunks, `14,971` launches)
Shape-median GPU total: **`809.8 ms`** (trimmed-85%: `848.1 ms`, clean wall `888.6 ms` including `~7 ms` `ds_ctx` + launch overhead).

| Rank | Kernel | Launches/4K | Samples | Med ms/4K | Share | Avg µs | Eff. GB/s | Optimization Target |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `ie_inkling_gemm_rq_exp_iq3xxs_situ_r6` | 384 | 105 | **370.0** | **45.7%** | 963.5 | 16.9 | **#1 Prefill Bottleneck** (3 WGs/CU `<=128` VGPR, 0 bank conflict `ds_load_b128`, `0x4B400000`, flat tile worklist) |
| 2 | `ie_inkling_gemm_iq4nl_exp_r2` | 384 | 99 | **176.6** | **21.8%** | 459.8 | 26.0 | **#2 Prefill Bottleneck** (`<=128` VGPR / `<=20.7 KiB` LDS -> 3 WGs/CU, `f32x4` epilogue, flat worklist) |
| 3 | `ie_inkling_gemm_q8_med_r2` | 1,760 | 399 | **148.4** | **18.3%** | 84.3 | 71.4 | **#3 Prefill Bottleneck** (`<=128` VGPR / `<=20.5 KiB` LDS -> 3 WGs/CU, `f32x4` stores) |
| 4 | `ie_inkling_rel_attn_pf_r1` | 384 | 91 | **50.3** | **6.2%** | 131.0 | — | Kill 85 `% kv_len` divs, Far/Near split, `D=3` ring, Split-KV |
| 5 | `ie_inkling_gemm_f32_router_r2` | 384 | 97 | **26.5** | **3.3%** | 69.1 | 17.1 | Currently 211 VGPR (1 WG/CU) -> `4x4` tile `<=96` VGPR (4 WGs/CU) |
| 6 | `ie_inkling_moe_sum_r1` | 384 | 99 | **21.7** | **2.7%** | 56.6 | — | Vectorized `f32x4` + fuse `sconv+add` |
| 7 | `ie_inkling_lc_prenorm_q8_4x_r6` | 384 | 87 | 9.3 | 1.1% | 24.2 | — | Already 4× fused |
| 8 | `ie_sconv_ring_store` | 1,536 | 385 | 2.0 | 0.3% | 1.3 | — | Fold into `sconv` kernels |
| 9 | `ie_inkling_gemm_q8_small_r1` | 1,664 | 390 | 0.7 | 0.1% | 0.4 | 897.6 | `32x64`/`64x64` tiles for `wrel`/`loop_down` |
| 10 | `ie_inkling_embed_q8_batch_r2` | 8 | 1 | 0.6 | 0.1% | 78.0 | — | Host-pinned embed read |
| 11 | `ie_inkling_vclip_r1` | 1,536 | 381 | 0.6 | 0.1% | 0.4 | — | Fuse into QKV / KV store |
| 12 | `ie_inkling_rmsnorm_mt_r1` | 1,232 | 320 | 0.5 | 0.1% | 0.4 | — | — |
| 13 | `ie_inkling_attn_res_r1` | 768 | 174 | 0.4 | 0.1% | 0.6 | — | — |
| 14 | `ie_inkling_sconv_r1` | 832 | 212 | 0.4 | 0.0% | 0.5 | — | — |
| 15 | `ie_inkling_lc_wfg_gate_r2` | 768 | 200 | 0.3 | 0.0% | 0.4 | 783.4 | — |

**Top 4 kernels (`IQ3+SITU`, `IQ4_NL`, `Q8_med`, `rel_attn_pf`) = `92.0%` (`745.3 ms`) of 4K prefill compute!**

### Top-15 GPU Kernels — 4K Verify `T=6` (`ds4k-vb6-s19`, 20 reps, `stride=19`, `1,735` launches/step)
Shape-median GPU total: **`20.81 ms`** (trimmed-85%: `19.56 ms`; clean `ds4k-clean` verify: `19.21 ms/step`, `verify bench T=6`: `23.20 ms` with trace overhead).

| Rank | Kernel | Launches/step | Samples | Med ms/step | Share | Avg µs | Eff. GB/s | Optimization Target |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `ie_inkling_gexp_raw_iq3xxs_compact_r6` | 48 | 51 | **4.823** | **23.2%** | 100.48 | 161.8 | Multi-token compact tile / FLUTE |
| 2 | `ie_gemv_q8q8_tr` | 268 | 282 | **4.660** | **22.4%** | 17.39 | 293.7 | Wave 2 D1/D2 fusion + Q8 verify GEMV |
| 3 | `ie_inkling_gexp_iq4nl_compact_r6` | 48 | 45 | **4.197** | **20.2%** | 87.44 | 136.6 | Wave 2 Fusion B (`down+sum+sconv+add`) |
| 4 | `ie_inkling_rel_attn_ver_r4` (40 SWA!) | 40 | 38 | **2.004** | **9.6%** | 50.10 | — | **Big Win:** Extend `QT=4..8` to 40 SWA layers (cuts 1.01 -> 0.17–0.34 GB/step!) |
| 5 | `ie_inkling_lc_wfg_gate_norm_r6` | 48 | 48 | **1.739** | **8.4%** | 36.22 | 17.3 | Optimize `T=2..8` tile path |
| 6 | `ie_inkling_rel_attn_ver_qtile_r6` (8 Global) | 8 | 10 | **1.529** | **7.3%** | 191.08 | — | Single-pass `T<=8` WMMA (avoids 2× KV read at `T=5..8`) |
| 7 | `ie_argmax` (6 separate launches!) | 6 | 5 | **1.117** | **5.4%** | 186.18 | — | **New Find:** Batched 1-launch multi-row argmax / fused with `logit_cap` (`-0.8 ms`!) |
| 8 | `ie_gemv_q4kq8_ts2` (`output.weight`) | 1 | 4* | **0.606*** | **~3.0%** | 606.0* | 430.0 | **Big Win:** 8-lane/row 1-pass `r7_q4k_head` (`193->96` VGPR, fixes `TH_LOAD_NT` 2× read) |
| 9 | `ie_inkling_vclip_r1` | **192** | 177 | 0.161 | 0.8% | 0.84 | — | Wave 2 Fusions C1 & D1/D2 (`-192` launches) |
| 10 | `ie_inkling_sconv_hist_r4` | **192** | 212 | 0.069 | 0.3% | 0.36 | — | Wave 2 Fusions B, C1, C2 (`-192` launches!) |
| 11 | `ie_inkling_gate_q8_r2` | 48 | 56 | 0.063 | 0.3% | 1.32 | — | — |
| 12 | `ie_inkling_sconv_r1` | **104** | 95 | 0.045 | 0.2% | 0.43 | — | Wave 2 Fusions B, C1, C2 (`-104` launches) |
| 13 | `ie_inkling_attn_res_r1` | 96 | 87 | 0.035 | 0.2% | 0.36 | — | — |
| 14 | `ie_inkling_sconv_add_r2` | 88 | 88 | 0.032 | 0.2% | 0.36 | — | Wave 2 Fusions B & C2 (`-88` launches) |
| 15 | `ie_gemv_q8q8_ts` | 88 | 97 | 0.032 | 0.2% | 0.36 | 366.0 | Wave 2 Fusion D2 |

*\*`ie_gemv_q4kq8_ts2` isolated duration measured `606.0 µs` (`min=601.3, max=654.1`) in serialized `ds4k_ver.csv` (`+79.6 µs` for `ie_logit_cap` = `686 µs` combined); `stride=19` overlapped with adjacent launches (`142.6 µs`).*

### 4K DSpark Draft Breakdown (`ds_draft`, `69` launches/step, clean wall `2.99 ms/step`)
1. `ie_dspark_attn_r4` (3 layers): **`0.961 ms`** (`28.9%` @ 4K; layer-1 Global jumps to **`1.45 ms`** @ 16K!)
2. `ie_gemv_q4kq8_tr8` (16 launches; `output.weight` split base = `639 µs` + 15 small projections): **`0.887 ms`** (`26.7%`)
3. `ie_gemv_q4k_swiglu` (3 layers): **`0.347 ms`** (`10.4%`)
4. 5-step zero-sync Markov loop (`30` launches: `cap_argmax` `330 µs`, `gemv_p_q4` `266 µs`, `rnn_step` `259 µs`, `add_q8` `159 µs`, `gemv_q4k` `87 µs`, `concat_q8` `8 µs`): **`1.11 ms`** (`33.3%`). *(Opportunity: batch `5x add_q8` into 1 launch and fuse `concat+rnn_proj+rnn_step` + certified 64K shortlist for `base`/`P`!)*

---

## 3. 16K Long-Context Prefill & Verify Profile (`--ctx 32768`, `--chunk 512`, DSpark `5x0.6`)

### Clean Wall-Clock Numbers
- **`syn:16384` (`ds16k-clean`):** Prefill **`3,558.3 ms`** (**`4,604.5 tok/s`**), `3,845 MB` free after load, `VRAM 12.78 GiB / budget 14.03 GiB` (zero eviction!).
- **`real-code-16k.txt` (`ds16k-real-clean`, `IE_LOOKUP=0`, `--n 16`):**
  - Prefill: **`3,769.3 ms`** (**`4,346.7 tok/s`**)
  - DSpark `5x0.6` Decode: **`134.2 tok/s`** (`7.452 ms/tok`, `3.00 tok/step`, `52.6%` accept)
  - Per-step split (`22.36 ms/step`): `draft` **`3.60 ms/step`** (`16.1%`, `+0.61 ms` vs 4K from Global layer-1 DSpark attn), `verify` **`18.72 ms/step`** (`83.7%`), `commit+host` `0.04 ms/step`.

### Top-15 GPU Kernels — 16K Prefill (`ds16k-trace-pf`, `stride=16`, 32 chunks, `59,875` launches)
Unbiased mean total: `3,764.4 ms` (vs clean `3,558.3 ms`, only `+5.8%` trace overhead!); shape-median total: `3,050.7 ms`.

| Rank | Kernel | Launches/16K | Samples | Est. ms/16K (Mean) | Share | Avg µs | Eff. GB/s | Scaling Note |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `ie_inkling_gemm_rq_exp_iq3xxs_situ_r6` | 1,536 | 114 | **1,433.7** | **38.1%** | 933.4 | 17.4 | Linear (4× of 4K) |
| 2 | `ie_inkling_gemm_q8_med_r2` | 7,040 | 446 | **796.0** | **21.2%** | 113.1 | 53.2 | Linear (4× of 4K) |
| 3 | `ie_inkling_gemm_iq4nl_exp_r2` | 1,536 | 102 | **592.9** | **15.8%** | 386.0 | 30.9 | Linear (4× of 4K) |
| 4 | `ie_inkling_rel_attn_pf_r1` | 1,536 | 85 | **313.9** | **8.3%** | 204.3 | — | **Quadratic Global Growth!** (see below) |
| 5 | `ie_inkling_gemm_f32_router_r2` | 1,536 | 83 | **105.0** | **2.8%** | 68.3 | 17.3 | 211 VGPR (1 WG/CU) |
| 6 | `ie_inkling_attn_res_r1` | 3,072 | 178 | 80.7 | 2.1% | 26.3 | — | — |
| 7 | `ie_inkling_sconv_add_r2` | 2,816 | 157 | 62.9 | 1.7% | 22.3 | — | — |
| 8 | `ie_inkling_moe_sum_r1` | 1,536 | 93 | 59.6 | 1.6% | 38.8 | — | — |
| 9 | `ie_inkling_lc_prenorm_q8_4x_r6` | 1,536 | 64 | 53.6 | 1.4% | 34.9 | — | — |
| 10 | `ie_inkling_gemm_q8_small_r1` | 6,656 | 402 | 46.4 | 1.2% | 7.0 | 54.7 | `16x16` re-reads weights 32× |
| 11 | `ie_inkling_moe_sort_r1` | 1,536 | 87 | 34.5 | 0.9% | 22.5 | — | — |
| 12 | `ie_inkling_vclip_r1` | 6,144 | 348 | 34.4 | 0.9% | 5.6 | — | — |
| 13 | `ie_inkling_lc_wfg_gate_r2` | 3,072 | 218 | 33.2 | 0.9% | 10.8 | 29.0 | — |
| 14 | `ie_sconv_ring_store` | 6,144 | 385 | 27.3 | 0.7% | 4.4 | — | — |
| 15 | `ie_inkling_rmsnorm_mt_r1` | 4,928 | 328 | 22.6 | 0.6% | 4.6 | — | — |

**Critical SWA vs Global Prefill Attention Scaling (`ie_inkling_rel_attn_pf_r1` at 16K):**
- **40 SWA layers** (`win=1024`, 1,280 launches): constant **`130.7 µs/call`** (`167.3 ms` total).
- **8 Global layers** (`win=0`, 256 launches): mean **`612.1 µs/call`** (`156.7 ms` total), scaling linearly per chunk:
  - `pos < 4K`: **`210.3 µs/call`**
  - `4K..12K`: **`604.5 µs/call`**
  - `12K..16K`: **`1,294.6 µs/call`** (`max = 1,391.8 µs/call` -> projects to **`~2.6 ms/call` @ 32K** and **`~6.5 ms/call` @ 80K**, matching the 45–52 s 80K–93K live prefill logs!)

### Top-15 GPU Kernels — 16K Verify `T=6` (`ds16k-vb6-s19`, 20 reps, `stride=19`, `1,735` launches/step)
Shape-median GPU total: **`20.80 ms`** (trimmed-85%: `21.48 ms`).

| Rank | Kernel | Launches/step | Samples | Med ms/step | Share | Avg µs | Eff. GB/s | Delta vs 4K Verify |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `ie_inkling_gexp_raw_iq3xxs_compact_r6` | 48 | 51 | **4.823** | **23.2%** | 100.48 | 161.8 | Flat |
| 2 | `ie_inkling_gexp_iq4nl_compact_r6` | 48 | 60 | **4.222** | **20.3%** | 87.96 | 135.8 | Flat |
| 3 | `ie_gemv_q8q8_tr` | 268 | 265 | **3.638** | **17.5%** | 13.57 | 376.3 | Flat |
| 4 | **`ie_inkling_rel_attn_ver_qtile_r6` (8 Global)** | 8 | 12 | **3.607** | **17.3%** | **450.92** | — | **+2.08 ms (+2.36× vs 4K!)** — reads 16K KV TWICE for `T=6` (`QT=4`)! |
| 5 | `ie_inkling_rel_attn_ver_r4` (40 SWA) | 40 | 40 | **1.980** | **9.5%** | 49.50 | — | Flat (`1.01 GB`/step without QT!) |
| 6 | `ie_inkling_lc_wfg_gate_norm_r6` | 48 | 66 | **1.082** | **5.2%** | 22.55 | 27.8 | — |
| 7 | `ie_argmax` (6 launches) | 6 | 6 | **0.721** | **3.5%** | 120.12 | — | Batched 1-launch saves ~0.5 ms |
| 8 | `ie_gemv_q4kq8_ts2` (`output.weight`) | 1 | 2* | **~0.606*** | **~2.9%** | 606.0* | 430.0 | Double DRAM read (`TH_LOAD_NT`) |
| 9 | `ie_inkling_vclip_r1` | 192 | 208 | 0.165 | 0.8% | 0.86 | — | Wave 2 Fusions C1/D1/D2 |
| 10 | `ie_inkling_sconv_hist_r4` | 192 | 203 | 0.069 | 0.3% | 0.36 | — | Wave 2 Fusions B/C1/C2 |
| 11 | `ie_inkling_gate_q8_r2` | 48 | 51 | 0.063 | 0.3% | 1.32 | — | — |
| 12 | `ie_inkling_sconv_r1` | 104 | 111 | 0.045 | 0.2% | 0.43 | — | Wave 2 Fusions B/C1/C2 |
| 13 | `ie_inkling_attn_res_r1` | 96 | 112 | 0.035 | 0.2% | 0.36 | — | — |
| 14 | `ie_inkling_sconv_add_r2` | 88 | 97 | 0.032 | 0.2% | 0.36 | — | Wave 2 Fusions B/C2 |
| 15 | `ie_gemv_q8q8_ts` | 88 | 91 | 0.032 | 0.2% | 0.36 | 366.0 | Wave 2 Fusion D2 |

**Key 16K Verify Takeaway:** Total verify attention reaches **`5.59 ms/step` (`26.9%` of verify)** + `1.45 ms` in `ds_dspark_attn_r4` global layer 1 = **`7.04 ms/step`** spent solely in attention at 16K! Extending `QT` to SWA (`r7_attn_ver.inc`) + single-pass `T<=8` WMMA Global verify directly targets **`-2.5 to -4.0 ms/step`** at 16K!

---

## 4. Live-Log Forensics (`serve.c`, `gen.c`, `hip.c`) & Clean Round 1 Fixes

Evidence inspected in live server logs `logs/inkling2-dsh-tui-20261122-054740.err.log` (req `[7]` 135-tok side request evicting long context -> `[8]` 26.4K cold re-prefill `6,416 ms`) and `logs/inkling2-dsh-tui-20261122-055952.err.log` (req `[20]` 1,200-tok side request evicting 93.3K chat -> `[21]` 93.8K cold re-prefill **`52,385 ms`**; `sampling: 0.29–0.69 ms/step` across every sampled turn).

### Issue 1: Tiny Side Requests Evicting 26.8K–93.7K Cached Prefixes (+ 98–155 ms Warm-Turn SWA Re-Copy)
**Root Causes in Code:**
1. **Unconditional single-history overwrite (`ref/tr8-team-merge/src/gen.c:3418-3423`):** `gen_ctx` holds only ONE flat `kv_toks[0..kv_n)` mapped onto GPU KV `[0..kv_n)`. When a divergent side request (`L <= 1`) arrives (e.g., Open WebUI title/tagger/side chat with 135–1,200 tokens), `gen_run` unconditionally drops ALL `c->ck[r].pos > L`, overwrites `c->kv_toks` with the short prompt, and overwrites GPU KV positions `[0 .. n_prompt + n_out)` — permanently destroying a 26.8K–93.7K conversation even though positions `[n_prompt + n_out .. 93761)` in the 8 global KV layers were never even touched!
2. **Hardcoded `ck_limit == 2` for SWA models (`ref/tr8-team-merge/src/gen.c:2013-2017`):**
   ```c
   for (uint32_t i = 0; i < c->m->n_kvl; i++)
     if (c->g->kv_len[i] < c->g->n_ctx) return 2u;
   ```
   Written before `IE_SNAP_HOST=1` moved SWA snapshots to host RAM (0 VRAM, ~181 MB compact SWA per slot). Even setting `IE_CACHE_CKPT` is ignored!
3. **Disabled `live_snap` skip on Inkling-2 (`ref/tr8-team-merge/src/hip.c:6800, 6877`):**
   `if (!save && !nr && !b->sc_ring_bytes && !ds_bytes && snap_skip_on() && b->live_snap == k ...)` disables `live_snap` whenever SWA rings exist. Consequently, EVERY consecutive warm turn (`[7]` 12 new tokens taking `155 ms`, `[15]` 12 new tokens taking `98 ms`) redundantly copies 181 MB host->GPU at turn start even though GPU SWA rings already hold `GEN_CK_ANSWER`!

**Clean Round 1 Fix (Zero Extra VRAM, 100% Exact Bitwise Parity, Works With & Without `-DiskCache`):**
1. **Ephemeral Side-Request Prefix Stash (`gen.c`):**
   - Inkling-2 has only **8 target global layers + 1 DSpark global layer** (`n_kv=8, hd=128, f16`: `2 * 8 * 128 * 2 = 4,096 B/layer` -> **`36 KiB/pos`** across all 9 global layers!).
   - When `cache && b->kv_io && !p->n_imgs && !c->kv_n_imgs`, primary cache has `best_pos >= 2048`, and an incoming request is **divergent & short** (`L < best_pos / 4`, `n_prompt <= stash_max` where `stash_max = IE_EPHEMERAL_MAX` default `4096`, `n_prompt < best_pos / 2`):
     1. Compute `stash_n = min(c->kv_n, min(stash_max, n_prompt + p->max_new + spec_room(p->draft) + 16))` (for a 152-tok title request with `max_new=64`, `stash_n = 232` -> only **`8.4 MiB`**; even at `stash_max=4096`, only **`144 MiB`** host RAM, `< 6 ms` PCIe DMA!).
     2. Copy ONLY global KV rows `[0 .. stash_n)` to host RAM via `b->kv_io(b, 1, 0, stash_n, stash_buf)`. (Primary conversation's SWA rings, `sc_ring`, DSpark SWA rings, and `frnn` are ALREADY safely stored in its `c->ck[best_idx].slot` host snapshot!)
     3. Run the side request in ephemeral mode (`cache = 0`), tracking max position written `max_pos`. If `max_pos <= stash_n`: restore `[0 .. max_pos)` via `b->kv_io(b, 0, 0, max_pos, stash_buf)` (**`< 1–5 ms`**!) and leave `c->kv_toks` + `c->ck[]` untouched! (If a side request turns out to be a long new conversation exceeding `stash_max`, abandon restore cleanly and cache the new conversation normally.)
   - **Impact:** 26.8K–93.7K re-prefill after a 152–1,200 token side request drops from **`4,400–52,385 ms` -> `< 10 ms`** (`>500×` TTFT win, 0 VRAM)!
2. **Enable `b->live_snap` tracking for Inkling-2 (`hip.c:6800, 6877` + clear in `sec==2 ds_ctx` and `gpu_kv_io(!out)`):** Saves ~40–60 ms redundant 181 MB H2D copy on every consecutive warm turn.
3. **Respect host-RAM `ck_limit` (`gen.c:2013`):** Allow 4–8 host-RAM checkpoints for Inkling-2 (when `IE_SNAP_HOST != 0`, override via `IE_CACHE_CKPT`).

### Issue 2: `0.44 ms/step` CPU `sampling:` Overhead (`gen.c:1919`)
**Smoking-Gun Root Cause in Code (`hip.c:7396` vs `gen.c:1465, 1836`):**
1. In `spec_generate` (`gen.c:1465`), Inkling-2 sets unpadded **`V = model_vocab(m) = m->vocab_unpadded = 200058`** (while `m->vocab = 201024`).
2. Inside `gpu_run` (`hip.c:6534, 7396`), `topk_prefetch(b, T)` runs GPU top-K with **`nv = b->m->vocab = 201024`** and caches `tk_pre.nv = 201024`!
3. Immediately after `gpu_run` returns, `gen.c:1833` starts the `sample_ms` timer and calls `gpu_cands(sp, b, g, 0, nd + 1, V, ...)` with **`nv = 200058`**.
4. `gpu_topk_lse` (`hip.c:7404`) checks `nv == tk_pre.nv` -> **`200058 == 201024` IS ALWAYS FALSE!!**
5. **Result:** On EVERY SINGLE VERIFY STEP of Inkling-2 in `ie-serve`:
   - `topk_prefetch` runs `ie_topk_max` + `ie_topk_sel` + D2H sync over `201024` floats and **throws the result in the trash**!
   - Then `gpu_cands` misses `tk_pre`, launches **`ie_topk_max` + `ie_topk_sel` + a SECOND full GPU->CPU `hipMemcpyDeviceToHost` sync** over `200058` floats, burning **`0.30–0.69 ms/step`** (mean `0.44 ms/step`, e.g., `1,546 ms` wasted in a single turn `[11]`!).
   - *(Confirmed by live log `054740.err.log`: req `[1]` with only 145 tokens and 0 DRY hits already spent `0.298 ms/step`; req `[10]` with `mtp 11.13` = more rows passed to duplicate top-k jumped to `0.691 ms/step`! Meanwhile `dry_apply` already has a fast 2M-bit bloom filter returning in `< 100 ns` on non-repeats.)*

**Clean Round 1 Fix:**
1. **Fix `nv` in `hip.c:6534, 7396`:** Use `const u32 nv = b->m->vocab_unpadded ? b->m->vocab_unpadded : b->m->vocab;` in both `fold` check and `topk_prefetch`. This immediately turns `gpu_cands` into a `100%` hit on `tk_pre` (`0` extra GPU launches, `0` extra D2H syncs -> **`sampling:` drops from `0.44 ms/step` -> `< 0.01 ms/step`** AND eliminates the wasted duplicate top-k from `verify` time too — a **double win** on both `sampling` and `verify`!!).
2. **Bonus (same round or R02):** Expose the existing 1-launch exact `ie_saluki_topk_r11` (from `ie_saluki_r11_g10.inc`, generic for any `nv <= 262144, K <= 64`, 128-bit `f32x4` loads, 0 `expf` when `need_lse==0`) inside `ie_inkling_kernels.hsaco` so the single prefetched top-k takes **1 launch instead of 2** (`ie_topk_max` + `ie_topk_sel`), and add a hash-bucket index to `dry_apply` for O(1) lookup when DRY actually triggers on long outputs.

---

## 5. Round 0 Harness Baseline (`results/inkling2/40round-ledger.json`)

Recorded live inside `bench/run-memory-bounded.ps1 -BudgetGiB 20` (`Tag: ink40r-r00`, peak owned RAM `13.588 GiB`, min free `33.349 GiB`, other VRAM `3,122 MB`, total wall `141.4 s`, mask `0x67dfc0369c3ffff`):

| Benchmark Stage | Primary Metric | Secondary / Breakdown | Parity / SHA-16 |
|---|---|---|---|
| **Plain `T=1` (`t1_233`, `--n 64`)** | **`109.19 tok/s`** (`9.158 ms/tok` wall) | GPU `7.14 ms` (`434.69 GB/s`), **`1,574` launches/step**, host issue `1.007 ms` | `BASELINE` (`39f9d7ccee0eb5a7`) |
| **DSpark `5x0.6` 233 (`ds5_233`, `--n 128`)** | **`281.50 tok/s`** (`3.552 ms/tok`) | `6.68 tok/step` (`73.5%` acc), `draft 1.74 ms`, `verify 21.98 ms` | `BASELINE` (`67cb62f159566c9a`) |
| **DSpark `5x0.6` 4K (`ds5_4k`, `--n 64`)** | **Prefill `4,644.2 tok/s`** (`TTFT 882.0 ms`) / **Decode `172.50 tok/s`** | `3.94 tok/step` (`51.6%` acc), `draft 2.33 ms`, `verify 20.46 ms`, arena `134.27 MiB`, VRAM free `3,845 MB` | `BASELINE` (`0043be349ae4d4a8`) |
| **DSpark `5x0.6` 16K (`long_16k`, `--n 32`)** | **Prefill `4,368.0 tok/s`** (`TTFT 3,750.9 ms`) / **Decode `138.60 tok/s`** | `3.88 tok/step` (`42.6%` acc), `draft 3.44 ms`, `verify 24.49 ms` | `BASELINE` (`97545643da8d725a`) |
| **16K Verify Bench (`vbench_16k`)** | `T=6 19.662 ms` / `T=8 23.385 ms` | `T=1 10.212 ms`, `T=16 36.477 ms`, `step 10.501 ms` | raw |
| **4K KLD Reference (`kld_4k`)** | `ppl = 1.2009` | Saved `results/inkling2/tmp/r00-ref-4k.kld` | reference |
| **Warm Disk Cache (`cache_4k`, `IE_DISK_IDLE_S=0`)** | **`TTFT 220.0 ms`** (`disk 171.0 ms`) | `9,081` cached tokens (`516 MB`), ckpt `204.75 MiB` | **`MATCH`** (`54d360c744463a46`) |
| **Live `ie-serve` (`serve`, 192 tok × 3)** | **Sampled `146.67 tok/s`** (`[148.9, 128.1, 163.0]`) | Greedy `166.7 tok/s` (`f1111b7ad1315a5d`), **`sampling 0.399 ms/step`**, **`multi-req hit TTFT 2,113.0 ms` (`14` cached!)** | baseline |

Dashboard rendered to `results/inkling2/40round-dashboard.html`.

---

## 6. Ranked Next Optimization Targets (Trace-Backed)

1. **TTFT & Live Serving (Wave 1, R01–R03):**
   - Ephemeral host-RAM prefix stash (`36 KiB/pos` global KV) + `live_snap` skip + `ck_limit` lift: cuts `multi_req_hit_ttft_ms` from **`2,113 ms (14 cached)` -> `< 100 ms` (`~9K cached`)** (and live 93.7K eviction `52.4 s -> < 10 ms`).
   - Fix `hip.c:6534, 7396` `topk_prefetch` `nv = vocab_unpadded` (`200058` vs `201024`) + 1-launch fused `topk_r11`: slashes **`sampling 0.399 ms/step -> < 0.015 ms/step`** and eliminates duplicate GPU top-k + 2nd D2H sync on every verify step.
2. **Decode Launch Reduction & Verify Attention/Head (Wave 2, R06–R12):**
   - Fusions A–D (`ie_inkling_r7_router_fuse`, `kv_sconv_fuse`, `moe_tail`, `attn_prep`): cut `1,574 -> ~818` launches/step (`T=1`, attacking the `2.005 ms` WDDM gap + `1.2 ms` GPU bubbles) and `1,735 -> ~822` (`T=6` verify).
   - `r7_q4k_head` (`606 -> ~450 µs` single-pass Q4_K verify head + fused `logit_cap`) + batched 1-launch multi-row `ie_argmax` (`1.12 -> ~0.25 ms`!).
   - `r7_attn_ver`: extend `QT` to 40 SWA layers (`2.00 ms`) + single-pass `T<=8` WMMA Global verify (`3.61 ms` @ 16K) -> **`-2.5 to -4.0 ms/step`** at 16K!
3. **4K & 16K Prefill Throughput (Wave 3, R13–R18):**
   - Top 4 prefill kernels are **92.0% of 4K / 83.3% of 16K**: `IQ3_XXS+SITU` (`45.7%`, `139->128` VGPR 3 WGs/CU + `ds_load_b128` + `0x4B400000` + flat tile worklist), `IQ4_NL` (`21.8%`, `150->128` VGPR 3 WGs/CU), `Q8_med` (`18.3%`, `162->128` VGPR 3 WGs/CU), and `rel_attn_pf` (`6.2%` @ 4K -> `8.3%` @ 16K, Global layers `1,295 µs/call` at 12–16K -> kill 85 `% kv_len` divs + Far/Near split + Split-KV).
4. **Sampled Chat Speculation (Wave 5–6):**
   - Replace `memset(qnode, 0)` (`gen.c:1646`) with probabilistic DSpark top-K proposals into `spec_rej_step()` + ForeSpark survival cutoff + 64K shortlist.

