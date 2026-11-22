# Inkling-2-Mini (IQ3_XXS) 40-Round Campaign — Waves 5–8 Deep Research & Implementation Bank (Rounds 21–40)

**Hardware & Toolchain:** AMD Radeon RX 9070 XT (`gfx1201`, RDNA4, `wave32`, 64 CUs / 32 WGPs, 640 GB/s GDDR6, 64 MiB Infinity Cache L3, 16 GiB VRAM, Windows WDDM/PAL, ROCm Clang 21 AMDGPU C11).  
**Companion Master Doc:** `research/inkling2/40r-research-bank.md` (Waves 1–4, Rounds 1–20).  
**Scope:** Concrete open-source code patterns, exact GitHub repos/PRs, arXiv papers, `amd-infer` (`ref/tr8-team-merge`) line-level hooks, and `gfx1201` wave32 recipes for **Wave 5 (Decode GEMV & Quantization)**, **Wave 6 (DSpark & Speculative Algorithms)**, **Wave 7 (Systems, WDDM Dispatch & Sub-Span Persistence)**, and **Wave 8 (Full-Stack Composition, Auto-Tuning & Regression Polish)**.

---

# WAVE 5 (Rounds 21–25): Decode GEMV, Lane-Aligned Repacking & Fused Head+Top-K on `gfx1201`

## 5.1 Reversible Lane-Aligned Weight Repacking for `IQ3_XXS` & `IQ4_NL` (R21–R22)

### Open-Source References & Papers
- **FLUTE** — *Fast Matrix Multiplications for Lookup Table-Quantized LLMs* ([arXiv:2407.10960](https://arxiv.org/abs/2407.10960))
  - Repo: [`HanGuo97/flute`](https://github.com/HanGuo97/flute) (`flute/utils.py` `pack`, `flute/csrc/`)
  - **Core Algorithmic Tricks:**
    1. **Offline/Load-Time Lane-Aligned Weight Restructuring:** Eliminates sub-byte unaligned loads and per-element bit-shifting inside the hot K-loop by pre-permuting packed indices and sign bits into the exact 32-lane word order consumed by each wave's vector load (`u32x4` = 128-bit coalesced load per lane).
    2. **LUT Duplication / Bank-Conflict Avoidance:** Eliminates shared-memory bank serialization when 32 lanes perform random LUT gathers concurrently.
- **Marlin** ([`IST-DASLab/marlin`](https://github.com/IST-DASLab/marlin)) & **ExLlamaV3** ([`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3))
  - **Core Tricks:** Weight-repacked register-file dequantization — never materialize dequantized weights to LDS/VRAM; keep packed activations wave-uniform in SGPR/LDS, stream weights with async 128-bit loads directly into VGPRs, unpack via byte-permute/ALU directly into `v_dot4` / MMA inputs, and process **4–8 output rows per wave** so every activation vector load is amortized 4–8×!
- **`ggml-org/llama.cpp` PR #32065 & `ikawrakow/ik_llama.cpp`**
  - [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) **PR #32065** (*"CUDA: few-row MMQ for MUL_MAT_ID"*, `ggml/src/ggml-cuda/mmq.cuh`): tightens dispatch (`MMVQ` -> few-row `MMQ` -> full `MMQ`) for small-batch MoE routing (`T = 1..16`).
  - [`ikawrakow/ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) (`ggml/src/ggml-cuda/mmq.cuh`, `mmq_id.cu`, `load_tiles_mmq_t`): specialized `IQ3_XXS` & `IQ4_NL` dequant-to-dot4 pipelines.

### Concrete `gfx1201` Wave32 Engineering Recipes (`IQ3_XXS` & `IQ4_NL`)

#### Recipe W5-1 — Zero-Overhead Load-Time Reversible Repacking for `IQ3_XXS` (`src/hip.c` `gpu_sink` + `ie_inkling_r8_iq3_repack.inc`)
- **Current Cost on Inkling-2:** `ffn_gate_up_exps` is `IQ3_XXS` (`[2304, 2304, 128]`, `780.3 MB/tok` active — **#1 MoE bandwidth & VALU consumer**!). Raw GGML `block_iq3_xxs` is **98 bytes per 256 elements** (not a power-of-two stride! Forces unaligned `ink_r6_pf_ld4` loads, 9-bit bit-field extraction across byte boundaries, and 7-bit sign-table lookup per 8 weights).
- **Reversible Lane-Aligned SoA Layout (0 extra GGUF disk changes, 100% lossless bit-exact repacking inside `gpu_sink()` at `src/hip.c:996`!):**
  - Inkling-2 `nb = 72` (`2304 / 32`) = exactly **`9` super-blocks of 256 weights** per row (`9 × 98 B = 882 B/row`).
  - At load time (opt-in via `IE_INKLING_IQ3_REPACK=1`, reversible in memory, `0` GGUF changes, **+~3% VRAM** or pad 98 -> 104 B? Better: **0 extra bytes via SoA split!**):
    - Split each expert's `[2304, 9]` super-blocks into **3 contiguous 16-byte-aligned SoA arrays**:
      1. `d_arr`: `f16` super-block scales `[2304, 9]` (padded to `[2304, 10]` or `[2304, 16]` -> `41 KB/expert`, `5.2 MB` total — 100% coalesced!)
      2. `grid_idx`: `u8` 8-bit grid indices (64 bytes/super-block = exact `4 × u32x4` aligned!) interleaved across **4 rows per wave** (`[2304/4, 9, 4_rows, 64_bytes]`) so a single 128-bit `global_load_b128` per lane fetches 16 contiguous grid indices with **zero unaligned crossing**!
      3. `signs_scales`: `u32` packed 32-bit `(4 × 7-bit signs + 4-bit sub-scale)` (`32 B/super-block = 2 × u32x4`, 100% aligned, **zero cross-byte bit-field splits!**).
  - **Total Memory Delta:** `98 B -> 96 B (grid+signs) + 2 B (d)` = **EXACTLY `98 B`/super-block (`0.0 MB` extra VRAM!!)** — 100% aligned 128-bit loads for BOTH decode `T=1`, verify `T=2..16`, and prefill WMMA!
  - **ISA Win on `gfx1201`:** Eliminates unaligned `u32x4p` splits and cuts bit-unpacking VALU instructions by ~30–40%, letting `v_dot4_i32_i8` (`ie_dot4_ss`) + `s_wait_loadcnt` run at pure DRAM roofline (~`580–610 GB/s`)!

#### Recipe W5-2 — `gfx1201` `v_dot4_i32_i8` + `v_perm_b32` Multi-Row Wave-Uniform GEMV (`IQ4_NL` & `Q8_0`)
- **Hardware Idioms for `gfx1201` (RDNA4, wave32):**
  1. **`v_perm_b32` Register LUT for `IQ4_NL`:** `ie_iq4_map` in `ie_kernels.c` already maps 4 nibbles via `v_perm_b32` (0 LDS, 0 bank conflicts!). Ensure every `T=1` and `T=2..8` verify `IQ4_NL` GEMV uses **4 rows/wave (8 lanes/row)** with **`1` shared `X` Q8_0 activation read cached in SGPRs/LDS** across all 4 rows (Marlin/ExLlamaV3 pattern) and **2-step `s_wait_loadcnt` double-buffering** (issue `w_next` global load before `ie_iq4_map` + `ie_dot4_ss` of `w_cur`)!
  2. **Wave-Uniform Activation Caching for `T=2..8` Verify:** In `r6_compact_moe.inc`, `ink_r6_cm_iq3_dot` already decodes weights once across up to `nj=4` routed pairs per pack (`packs[p]`). For **dense `Q8_0` verify (`attn_q/k/v/o`, `wrel`, `loop_down`)**, share each `Q8_0` weight row load across ALL `T` tokens (`T = 2..8`, 1 weight read, `T` `ie_dot4_ss` accumulators per lane -> **4–8× weight traffic reduction**), keeping `<= 96 VGPR` (4 WGs/CU, 0 spills) by limiting unroll factor with `__builtin_amdgcn_sched_barrier(0)`.
  3. **Clang 21 `v_dot4_i32_i8` Signedness & Bitcast Safety:** Always route through `ie_dot4_ss` (`__builtin_amdgcn_sdot4` / `v_dot4_i32_i8`) and extract `u32x4` fields to scalar `const u32` before any bit-cast!

---

## 5.2 Fused `Q4_K` Output Head + Tile-Local Online Top-K / Argmax (R23–R25)

### Open-Source References & Papers
- **RadiK** — *Scalable and Optimized GPU-Parallel Radix Top-K Selection* ([arXiv:2501.14336](https://arxiv.org/abs/2501.14336), [`leefige/radik`](https://github.com/leefige/radik))
- **AIR Top-K / GridSelect** — [`NVIDIA/cccl`](https://github.com/NVIDIA/cccl) (CUB `BlockTopK` / `DeviceTopK`) & [`anilshanbhag/gpu-topk`](https://github.com/anilshanbhag/gpu-topk)
- **Fused LM-Head + Online Argmax/Confidence** — [`lsnchow/LLaDa-CUDA-fused-kernel`](https://github.com/lsnchow/LLaDa-CUDA-fused-kernel) & [`ROCm/composable_kernel`](https://github.com/ROCm/composable_kernel) (`ck_tile` row-wise top-k/softmax)

### Exact Bottleneck in `amd-infer` (`src/hip.c:6381-6401`, `kernels/ie_kernels.c:5565-5730`)
- Today, EVERY target verify (`T=6..8`) AND EVERY DSpark draft step (`5×`) runs **3 separate global-memory passes over `V = 200,058` logits (`800 KiB × T`)**:
  1. `ie_gemv_q4kq8_*` (or `K_DS_GEMV_P_Q4_ZSYNC_R5`) writes `200,058 × T × 4 B` (`4.8–6.4 MB`) to `g->logits` in VRAM!
  2. `K_LOGIT_CAP` reads & rewrites all `4.8–6.4 MB` (`2` more passes!) for `softcap = 30.0`!
  3. `ie_topk_max` + `ie_topk_sel` (or `K_ARGMAX` / `K_DS_CAP_ARGMAX_ZSYNC_R5`) reads all `4.8–6.4 MB` back AGAIN (2 launches: `ie_topk_max` + `ie_topk_sel`!).
- **Total wasted VRAM traffic per verify step:** `~20–25 MB` + **3–4 kernel launches**!

### Concrete `gfx1201` Wave32 Recipe (`W5-3` — Fused `Q4_K` Head + Softcap + Tile-Local Top-K)
- **Attach directly onto `kernel-author-b`'s `ie_inkling_gemv_q4k_head_ver_r7` (`8` lanes/row, `4` rows/wave, `32` rows/WG, `Grid = (6282, 1, 1)` or persistent `Grid = (256, 1, 1)` looping over `~24` row-tiles/WG!):**
  - **Why Persistent `Grid = (192..256, 1, 1)` (`3–4` WGs/CU) Is Magical Here:**
    - Instead of `6,282` tiny WGs writing `200,058` floats to VRAM, **each of 256 WGs** loops over `~785` vocab rows (`25` tiles of 32 rows), computing exact 4-acc `Q4_K` dot products for `T = 1..8` tokens AND applying `30.0f * tanhf(x / 30.0f)` **in registers**!
    - **If `temp == 0` (greedy):** Each WG maintains ONLY `(max_val[T], max_id[T])` in `2*T` registers (`16` VGPRs max!), reduces across the WG in LDS (`256 B`), writes 1 `(val, id)` per token to `part[256][T]` (`16 KiB` total!), and the **last WG (`atomicAdd == 255`)** reduces those 256 partials -> **0 bytes written to `g->logits`! Saves 3 launches + 20 MB VRAM traffic!**
    - **If `temp > 0` (`K <= 32`, e.g., `top_k = 20`):** Each WG maintains a tiny **wave-thresholded candidate buffer in LDS** (`K=20` sorted pairs per token, updated ONLY when `logit > lds_min_k[t]` — >99% of vocab rows fail that 1-cycle compare!), emits its local top-20 (`256 × 20 × 8 B = 40 KiB` total!) at WG exit, and the **last WG** merges the 256 × 20 candidates in LDS and writes directly into a **host-pinned zero-copy `tk_res` buffer**!
    - **DSpark Draft Head (`P * lat_i + base_i`):** Same exact fusion inside `K_DS_GEMV_P_Q4_ZSYNC_R5` — fuse `base_i + P*lat_i -> softcap -> tile-local top-K + softmax/LSE -> stochastic sample` into **1 single launch per draft step** (`0` `b->ds.logits` global writes, cuts `5` launches/step + `12 MB` traffic from `ds_draft`!).

---

# WAVE 6 (Rounds 26–30): DSpark Probabilistic Speculation, Certified Shortlists, Suffix Automaton & Tree Verify

## 6.1 Exact Probabilistic DSpark Proposal + Lossless `spec_rej_step()` (R26 — #1 Sampled Chat Win!)

### Open-Source References & Papers
- **Speculative Sampling** — Leviathan et al. ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192)), Chen et al. ([arXiv:2302.01318](https://arxiv.org/abs/2302.01318))
- **DSpark** — *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation* ([arXiv:2607.05147](https://arxiv.org/abs/2607.05147), [`deepseek-ai/DeepSpec`](https://github.com/deepseek-ai/DeepSpec))
- **vLLM Inkling-2-Mini & Rejection Sampler:**
  - [`vllm-project/vllm`](https://github.com/vllm-project/vllm) **PR #69028** (`Inkling2LoopForConditionalGeneration` + `--speculative-config '{"method":"dspark",...}'`)
  - [`vllm/v1/sample/rejection_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py) — contrasts exact `draft_probs` $q(x)$ rejection (`min(1, p(x)/q(x))`) against the low-acceptance `NO_DRAFT_PROBS` ($q(x)=1$ point-mass) fallback!

### Exact Math & `amd-infer` Wiring (`src/gen.c:1304-1442, 1621-1648` + `src/hip.c:6355-6413`)
1. **Why Current Sampled Acceptance Is Trapped at `1.63–2.20 tok/step`:**
   - At `src/gen.c:1646`, DSpark clears `memset(qnode, 0, sizeof qnode)`.
   - Therefore `spec_rej_step()` skips the Leviathan branch (`gen.c:1381: if (qd && qd->n > 0u)`) and executes line 1418 (`rng_u01(sp) < pd[hit]`), which is **greedy point-mass acceptance** $\mathbb{P}(\text{acc}) = p(x^*)$. At `temp=1.0, top_p=0.95, top_k=20`, typical chat entropy yields $p(x^*) \approx 0.35\text{--}0.50 \implies \mathbb{E}[\text{tok/step}] \approx 1.54\text{--}2.00$ even with an **oracle** drafter!
2. **Exact Lossless Probabilistic Protocol (Zero-Sync GPU Loop!):**
   - For any draft distribution $q_j(x)$ supported on a candidate set $S_j$ (`|S_j| = K`, e.g., `K = 20` matching `top_k`, or `K = 32`), drawing $d_{j+1} \sim q_j$ and accepting with probability $\min\big(1, \frac{p_j(d_{j+1})}{q_j(d_{j+1})}\big)$ (and on rejection sampling the bonus token from $\tilde{p}_j(x) = \frac{\max(0, p_j(x) - q_j(x))}{\sum_y \max(0, p_j(y) - q_j(y))}$) produces **THE EXACT TARGET DISTRIBUTION $p_j(x)$** (Theorem 1, Leviathan `arXiv:2211.17192`)!
   - **Zero-Sync `hip.c:6355` Implementation (`K_DS_CAP_SAMPLE_ZSYNC_R8`):**
     - Pass `(temp, top_p, top_k, rng_seed)` in `ds_ctrl` (`zb` header).
     - At each Markov step `i = 0..nd-1`, instead of `K_DS_CAP_ARGMAX_ZSYNC_R5` picking greedy argmax:
       1. Extract exact top-$K$ (`K = min(sample_k(sp, V), 32)`) logits $(v_0 \ge \dots \ge v_{K-1}, \text{id}_0 \dots \text{id}_{K-1})$ of DSpark's softcapped logits.
       2. In the final reduction wave (1 wave = 32 lanes, `< 2 µs`!), compute `q[c] = exp((v[c] - v[0]) / temp_d)` (`temp_d = temp`, optionally tunable `0.85..1.0 * temp` — **lossless for ANY `temp_d`!**), apply exact `top_p` nucleus cutoff, normalize `q[0..m-1]`, and draw `d_i` using a GPU `xorshift64*` seeded from `ds_ctrl`!
       3. Write `prev_tok = d_i` into `ds_ctrl` (so step `i+1`'s Markov RNN conditions on the **actually sampled** draft token `d_i` — critical for multi-token suffix coherence at `temp=1.0`!), AND store `{m, id[0..m-1], q[0..m-1]}` into `zb` (`nd * (4 + 32*8) <= 2.1 KiB`!).
       4. At the end of all `nd` steps, ONE single `~2.2 KiB` D2H `hipMemcpy` populates `d[1..nd]` AND `qnode[0..nd-1]` in `gen.c`!
       5. **`DRY` Safety:** Because `spec_rej_step()` applies `dry_apply()` to target `pd` BEFORE computing `min(1, pd/qd)` and `max(0, pd - qd)`, DRY loop suppression stays **100% exact** even though the drafter doesn't need to know DRY!
   - **Expected Gain:** Sampled acceptance jumps from `1.63–2.20` -> **`3.5–5.0+ tok/step`** (**+50–100% live `ie-serve` chat speed**), with `temp=0` staying 100% bit-identical to Phase 7!

---

## 6.2 ForeSpark Survival-Probability & Work-Conserving Budget Cutoff (R27)

### Open-Source References & Papers
- **DSpark** ([arXiv:2607.05147](https://arxiv.org/abs/2607.05147), [`deepseek-ai/DeepSpec`](https://github.com/deepseek-ai/DeepSpec)) — confidence head estimating prefix survival probability $S_j = \prod_{i=1}^j \hat{\alpha}_i$.
- **ForeSpark** — *ForeSpark: Joint Draft and Verification Budget Scheduling for Parallel Speculative Decoding* ([arXiv:2611.13442](https://arxiv.org/abs/2611.13442), [`facebookresearch/ForeSpark`](https://github.com/facebookresearch/ForeSpark)) — replaces static `pmin` with hardware-calibrated concave marginal-gain stopping!

### Concrete Implementation Recipe (`W6-2` — Calibrated Survival-Probability Early Exit)
- **Current Limitation (`hip.c:6398`):** Current `zsync` stops when a single token's greedy prob `pr[i] < pmin (0.6)`. Under **probabilistic drafting (`temp=1.0`)**, a token with `q(d_i) = 0.25` can STILL have **90% acceptance** if $p(d_i) \approx 0.25$!! A naive `q(d_i) < 0.6` check would prematurely abort almost EVERY step at `temp=1.0`!
- **Correct Metric for Probabilistic Drafting (Entropy / Top-P Concentration + Survival Proxy):**
  - Instead of raw `q(d_i) < pmin`, estimate per-step expected acceptance $\hat{\alpha}_i$ from DSpark's **confidence/entropy** (e.g., $\hat{\alpha}_i = \text{clamp}(a - b \cdot H(q_i), 0, 1)$ where $H(q_i) = -\sum_c q_i(c) \log q_i(c)$ is computed in 5 cycles from the already-normalized top-K `q[0..m-1]`, or calibrated from online EMA of observed acceptance binned by $H(q_i)$!).
  - Track cumulative survival $S_j = \prod_{i=1}^j \hat{\alpha}_i$. Stop drafting step $j+1$ inside `zsync` when the marginal expected accepted tokens $S_{j+1}$ no longer exceeds the measured hardware marginal verify+draft ratio:
    $$S_{j+1} < \frac{\Delta t_{\text{draft}} + \Delta t_{\text{verify}}(j+1)}{\text{ms\_per\_tok}_{\text{ema}}}$$
    On Inkling-2 RX 9070 XT (`t_verify(T)` = `12.6 / 14.7 / 16.4 / 17.9 ms` for `T=2/4/6/8`, so $\Delta t_{\text{ver}} \approx 0.85\text{ ms/tok}$, $\Delta t_{\text{draft}} \approx 0.20\text{ ms/tok}$): verifying 1 more token only costs `~1.05 ms` vs `~5.5 ms/tok` value -> **optimal cutoff is $S_{j+1} \approx 0.18\text{--}0.25$**, dynamically drafting `7` tokens on easy code/boilerplates and `2–3` tokens on high-entropy creative turns!

---

## 6.3 Certified Frequency-Ranked + In-Context Vocab Shortlist for DSpark `lm_head` (R28)

### Open-Source References & Papers
- **FR-Spec** — *Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling* ([arXiv:2502.14856](https://arxiv.org/abs/2502.14856), [`thunlp/FR-Spec`](https://github.com/thunlp/FR-Spec)) — static frequency-ranked shortlist (e.g., `32K`/`48K` out of `200K`), **100% lossless** because target verifies full vocab.
- **MicroSpec** — *Accelerating Speculative Decoding with Lightweight In-Context Vocabularies* ([arXiv:2605.26444](https://arxiv.org/abs/2605.26444)) — dynamically augments a compact core shortlist with **unique tokens from the current prompt/context window** so domain identifiers/code symbols/rare entities NEVER fall out of the draft shortlist!
- **SpecPilot** — *Autonomous Algorithm Discovery and Systems Optimization for Speculative Decoding* ([arXiv:2611.02501](https://arxiv.org/abs/2611.02501)) — ceiling-guided tuning of draft head quantization & shortlist coverage.

### Concrete `amd-infer` Recipe (`W6-3` — 60K Static + 4K In-Context MicroSpec Draft Head)
- **Why Naive 32K Failed Earlier (`sota-plan.md` §3):** Inkling-2's 200K BPE puts common code tokens at high IDs (`> 32768`), so naive `0..32767` truncation broke code acceptance (`7.47 -> 1.87`).
- **Two-Tier Certified 64K Shortlist (`60K` FR-Spec Core + `4K` Dynamic Prompt Union):**
  1. Offline script (1-time, 60s): count Inkling-2 token frequencies across code + chat + math corpora, save sorted `id_core[61440]` into a tiny 240 KB file (`models/DSpark/inkling2-frspec-60k.bin`) and pack `P_core` (`[61440, 256]` `Q4_0` = **`8.8 MB`** vs `28.9 MB`!) + `H_core` (`[61440, 2304]` `Q4_K` = **`79.6 MB`** vs `260.5 MB` — OR gather rows directly via `id_map[65536]` from target's existing `output.weight` with **0 MB extra VRAM!!**).
  2. **ZERO-VRAM Gathering Trick (MicroSpec!):** Instead of duplicating `output.weight`, pass `const G u32 *id_map` (`65,536 × 4 B = 256 KiB`!) to `ds_gemv_q4k_gather` and `K_DS_GEMV_P_Q4_ZSYNC_R5`:
     - Slots `0..61439` hold the static FR-Spec core IDs.
     - At prompt prefill (`gen_run`, `< 0.5 ms` on CPU!), scan `prompt[0..n_prompt)` with a 25 KB bitmap, append up to `4,096` unique in-context tokens not in `id_core` into slots `61440..65535`, and upload the `16 KiB` tail of `id_map` to VRAM!
  3. **Result:** `base` (`output.weight` read) and `P` read BOTH drop by **3.07×** (`260 MB -> 85 MB`, `28.9 MB -> 9.4 MB`), cutting `ds_draft` from **`1.83 ms -> ~0.85 ms`**, with **0 MB extra VRAM**, **>99.3% token coverage on arbitrary codebases**, and **100% exact lossless parity**!

---

## 6.4 Suffix-Automaton / N-Gram Lookup Hybrid & Tiny 8-Node Tree Verify (R29–R30)

### Open-Source References & Papers
- **SAM Decoding** — *Speculative Decoding via Suffix Automaton* ([arXiv:2411.10666](https://arxiv.org/abs/2411.10666)) — $O(1)$-per-token online Suffix Automaton (SAM) maintaining longest exact suffix matches across the entire 32K prompt + generated output in `< 5 µs/step` on CPU!
- **SuffixDecoding** ([arXiv:2411.04975](https://arxiv.org/abs/2411.04975), [`snowflakedb/ArcticInference`](https://github.com/snowflakedb/ArcticInference)) — dynamic speculation length scaling for code-editing & agentic loops.
- **Prompt Lookup Decoding (PLD)** ([`apoorvumang/prompt-lookup-decoding`](https://github.com/apoorvumang/prompt-lookup-decoding)) & **Lookahead Decoding** ([arXiv:2402.02057](https://arxiv.org/abs/2402.02057), [`hao-ai-lab/LookaheadDecoding`](https://github.com/hao-ai-lab/LookaheadDecoding)) & **SGLang** (`sgl-project/sglang` ngram/SAM speculative worker).
- **EAGLE-2** ([arXiv:2406.16858](https://arxiv.org/abs/2406.16858), [`SafeAILab/EAGLE`](https://github.com/SafeAILab/EAGLE)), **Sequoia** ([arXiv:2402.12374](https://arxiv.org/abs/2402.12374), [`Infini-AI-Lab/Sequoia`](https://github.com/Infini-AI-Lab/Sequoia)), **SpecInfer** ([arXiv:2305.09781](https://arxiv.org/abs/2305.09781)).

### Concrete Recipes (`W6-4` & `W6-5`)
1. **Recipe W6-4 — Zero-Draft-Cost SAM Pre-Check + Exact DSpark Catch-Up (`src/gen.c:1619-1648`):**
   - Today (`gen.c:1634-1643`), `lookup_draft()` ONLY runs **AFTER** DSpark already paid `1.83 ms` to draft! And `lookup_draft()` does a linear scan over `out[0..n]` (slow at 32K!).
   - Build a tiny **online Suffix Automaton (SAM)** on CPU (`~24 B/state`, `~1.5 MB` for 32K tokens, built in `< 1 ms` during prefill, updated in `O(1)` per accepted token!).
   - **Pre-Check Policy:** Before calling `b->run(b, 3, ...)` (`ds_draft`):
     - Query SAM for the longest suffix match `ml` ending at `out[n-1]`.
     - If `ml >= 6` (strong exact match in code edit / tool JSON / copy-paste!) AND `sp->temp <= 0.3f` (or greedy): **SKIP `ds_draft` completely (`0.00 ms` draft!)** and submit up to `12–15` SAM continuation tokens directly to target verify!
     - **CRITICAL DSpark State Catch-Up Invariant:** Notice `src/gen.c:1619` ALREADY executes `if (dsm && ing) b->run(b, 2, &out[P - ing], ing, P - ing, 0, NULL, NULL);` using the `ing = k + 1` accepted tokens' target hidden features from the verify pass! So even when a step skips `ds_draft` (`sec=3`), the next step's `ds_ctx` (`sec=2`) automatically catches DSpark's KV + `frnn` up on ALL `k+1` accepted SAM tokens with **0 state desync**!
2. **Recipe W6-5 — Tiny 8-Node EAGLE-2/Sequoia Dynamic Tree Verify (`T <= 8`, `src/gen.c:1776-1824`):**
   - `amd-infer` ALREADY has `b->tree_par`, `b->accept_tree`, `spec_rej_step` multi-child rejection, and `b->tree_ok`!
   - When DSpark's step-1 top-1 probability `< 0.55` (uncertain branch point, common in sampled chat & reasoning), fork **2 branches** at depth 1 (top-2 tokens from DSpark's Markov step 1, each rolled 3 steps -> `1 + 2*3 = 7` or `8` total nodes = **exact same `T=8` verify slot**!).
   - **Kernel Extension Needed for Inkling-2 (`<= 50` lines):** Add an optional `u8 tree_mask[8]` (8-bit ancestor bitmask per node, fits in 2 `u32` SGPRs!) to `ie_inkling_rel_attn_ver_r4` / `r7_attn_ver` (mask out non-ancestor intra-tree keys `kp >= pos0`) AND pass `tpar[t]` to `sconv_verify` so each tree node reads its parent's 3-tap shortconv history instead of `t-1`!
   - **Expected Gain:** +15–25% acceptance on high-entropy `temp=1.0` chat and hard reasoning prompts at identical `T=8` verify latency!

---

# WAVE 7 (Rounds 31–35): Windows WDDM Dispatch, Low-VGPR Sub-Span Persistence, Multi-Stream Overlap & Cold-Expert Tier

## 7.1 Windows WDDM / `ROCm/clr` Launch Retirement & Argument Packing (R31–R32 — MASSIVE Hidden Win!)

### Open-Source References & Exact `src/hip.c` Smoking Guns
- **ROCm/clr (`rocclr/platform/kernel.cpp` & Issue `#13762`)** — [`ROCm/clr`](https://github.com/ROCm/clr):
  - `KernelParameters::captureHIPArgs` handles `kernelParams` (`void **args`) by iterating per-argument descriptor metadata and pointer-chasing every `args[i]` on the CPU!
  - Passing a contiguous packed struct via **`extra = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &packed, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END }`** (`kernelParams = NULL`) avoids per-arg pointer chasing (`memcpy` path in `rocclr`), **WARNING (`ROCm/clr #13762`):** `sz` MUST exactly match the AMDGPU `.args` msgpack struct size/alignment or `rocclr` silently zero-fills trailing pointers!
- **Exact Smoking Guns Found in `ref/tr8-team-merge/src/hip.c`:**
  1. **`HostQueue::loop` WDDM Retirement Stall (`src/hip.c:2515-2534`)!** Look at the repo's own comment: on Windows PAL/WDDM, `HostQueue::loop` buffers ALL `1,574` launches into ONE batch and retires them **serially AFTER the GPU finishes** (`~3–5 ms` dead wall time: `GPU = 5.21 ms` vs `Wall = 9.45 ms`!), UNLESS an event marker flushes the batch mid-step (`IE_FLUSH_EVERY`) — **and `IE_FLUSH_EVERY` defaults to `0` (OFF)!!**
  2. **Legacy Blocking Streams (`src/hip.c:5890, 5930`):** `hipStreamCreate(&b->ink_gstream)` uses the default **blocking** stream (implicit sync with legacy stream 0 on WDDM!) instead of `hipStreamCreateWithFlags(&s, 1u)` (`hipStreamNonBlocking`, which `hip.c:1422` and `6646` already use!).
  3. **Pageable Stack D2H Copy (`src/hip.c:5969`):** `hipMemcpyAsync(&r, ...)` copies to a stack `u32 r` (pageable memory -> forces synchronous host staging under WDDM!).
  4. **32-bit Graph Cache Key Bug (`src/hip.c:5934`):** `b->ink_mask & 0xffffffffULL` ignores Phase 7 bits `32..58`!
- **vLLM Piecewise CUDA/HIP Graphs** — [`vllm/compilation/cuda_piecewise_backend.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/cuda_piecewise_backend.py) (`PIECEWISE` and `FULL_AND_PIECEWISE` modes).

### Concrete Implementation Recipes (R31–R32)
1. **Recipe W7-1 — Sweep & Promote WDDM `IE_FLUSH_EVERY` + `hipStreamNonBlocking` + Pinned Scalar Return (`src/hip.c:2523-2548, 5930, 5969`):**
   - Benchmark `IE_FLUSH_EVERY = {0, 32, 64, 96, 128, 192, 256}` on post-Wave-2 (`~850` launches/step) plain `T=1` AND `ds5` verify! Because `HostQueue::loop` retires earlier chunks in parallel while the GPU executes later layers, setting optimal `fl_every` (e.g., every 4–6 visits = ~`64–128` launches) on Windows WDDM directly slashes the `~3–4 ms` post-GPU host-retirement gap!
   - Switch `b->ink_gstream` and default compute stream to `hipStreamNonBlocking` (`flags=1`), and replace all hot-loop stack `hipMemcpyAsync(&r, ...)` D2H reads with a **pre-allocated 4 KiB `hipHostMalloc` pinned return struct**!
2. **Recipe W7-2 — Packed `HIP_LAUNCH_PARAM_BUFFER_POINTER` Fast Launch Helper (`src/hip.c:2539`):**
   - Generate a compile-time/init-time table of kernel arg layouts (`offset[a]`, `size[a]`, `total_bytes`) from `.hsaco` msgpack metadata (or `IE_MK` already computes exact packed arg sizes in `ie_mk_gen.h`!! Look at `src/hip.c:1022-1028`: **`ie_mk_gen.h` ALREADY HAS EVERY KERNEL'S EXACT PACKED ARGUMENT LAYOUT!**).
   - Pack `args` into a 256-byte stack buffer and pass `NULL, extra` (`HIP_LAUNCH_PARAM_BUFFER_POINTER`) behind an opt-in `IE_HIP_PACKED_ARGS=1` A/B flag; verify exact `total_bytes` against `rocclr` (never trigger `#13762` zero-fill!) and promote if it trims `host issue time` on WDDM.
3. **Recipe W7-3 — Re-Test `IE_INKLING_GRAPH=1` & Piecewise 12-Visit HIP Graphs Post-Wave-2 (`src/hip.c:5914-5975`):**
   - Fix `ink_graph_key` to hash full 64-bit `b->ink_mask` (`ink_mask ^ (ink_mask >> 32)` + `swa_sp` + `glob_sp`).
   - Phase 2's `IE_INKLING_GRAPH=1` had `~2,222` graph nodes (`+0.2 ms` regression on WDDM). Once Wave 2 cuts node count to **`~750–850` nodes** AND splits capture into **4–8 piecewise HIP graphs** (e.g., 12 visits per graph, interleaved with event markers so `HostQueue::loop` retires Graph `k` while GPU runs Graph `k+1` — combining `vLLM`'s `PIECEWISE` capture with WDDM retirement pipelining!), re-benchmark against fused eager and promote ONLY if wall-clock wins!

---

## 7.2 Low-VGPR (`<= 64 VGPR`) Non-GEMV Glue Sub-Span Persistence (R33)

### Open-Source References & Papers
- **Mirage Persistent Kernel (MPK)** — *MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs* ([arXiv:2512.22219](https://arxiv.org/abs/2512.22219), OSDI'26 ecosystem, [`mirage-project/mirage`](https://github.com/mirage-project/mirage) `mpk` branch) — SM/CU-level fine-grained task graph with decentralized worklist scheduling and cross-operator pipelining.
- **MonoMoE** — *MonoMoE: An Efficient Fused Mega-kernel for Quantized MoE Decoding* ([arXiv:2609.04244](https://arxiv.org/abs/2609.04244)) — weight-major persistent MoE mega-kernel partitioning thread blocks over expert weights and fusing routing, top-$k$, quantization, projections, and combine into 1 launch!

### Why Monolithic `IE_MK=1` Failed in Phase 6 — And Why **Glue-Only / MoE-Only Sub-Spans** Win on `gfx1201`
- **Phase 6 Post-Mortem (`#161`):** Monolithic `IE_MK=1` dropped `T=1` bandwidth from **`596 -> 276 GB/s` (`-13.2%` slower!)** because compiling ALL heavy GEMVs + attention + norms into ONE giant LLVM function spiked VGPRs to the max of the heaviest op (~`140–170` VGPR -> 1 WG/CU everywhere!) and forced global grid-barriers between every op!
- **Recipe W7-4 — Two-Class Surgical Sub-Span Persistence (`<= 64 VGPR`, `Grid = 64`, WDDM Wall-Clock Guarded):**
  - Keep **heavy GEMVs** (`IQ3_XXS`, `IQ4_NL`, `Q8_0` QKV/O, `Q4_K` head) and **attention** as standalone optimal-occupancy kernels (or fused Wave 2 kernels), NEVER inflating their VGPRs!
  - ONLY collapse **adjacent non-GEMV glue ops** (norms, clips, shortconv, residual adds, attention-residual bank updates, Q8 quantizations — all `<= 48 VGPR`, `<= 8 KiB LDS`, `4 WGs/CU`!) into a **single low-VGPR mini-dispatcher** (or a **MonoMoE-style `T=1` persistent MoE block** `Router -> Top8 -> IQ3+SITU -> IQ4+Combine+Sconv+Add` where `exp_mid_q8` is only `11.5 KiB` — stays 100% L2-hot across the entire sub-span at `<= 64 VGPR`, `Grid = 64` = 1 WG/CU, wall-clock poison guard `250 ms`!).
  - **Impact:** Eliminates another `~150–250` tiny launches/step WITHOUT sacrificing a single GB/s on the big GEMVs (`>= 580 GB/s` preserved!).

---

## 7.3 Multi-Stream Overlap (`ds_ctx` vs Target Head/Sampling) & Cold-Expert VRAM Tier (R34–R35)

### Open-Source References
- **`ggml-org/llama.cpp` PR #29887** — *"llama : add a GPU cache for MoE experts kept in host memory"* ([`ggml-org/llama.cpp#29887`](https://github.com/ggml-org/llama.cpp/pull/29887)) — fixed-address VRAM slot cache for hot MoE experts backed by pinned host RAM, exploiting MoE's Zipfian skew (15–20% hot experts carry ~80% of tokens!) to prevent catastrophic WDDM whole-tensor eviction when KV cache grows to `32K–64K+`!

### Concrete Implementation Recipes (R34–R35)
1. **Recipe W7-5 — Multi-Stream Overlap of Target `lm_head` + `topk` + Host Rejection with DSpark `ds_ctx` (`src/hip.c` + `src/gen.c`):**
   - Look at the DSpark step timeline (`src/gen.c:1619, 1830-1890`):
     1. Step $t$: Target verify runs 48 visits -> produces `h_out[0..nd]` AND `ds_feat[0..nd]` -> runs `output.weight` (`~0.5 ms`) -> `topk` (`~0.06 ms`) -> host `spec_rej_step()` -> `b->accept(k)`.
     2. Step $t+1$: `b->run(b, 2, ...)` (`ds_ctx`, `~0.15–0.30 ms`) ingests the `k+1` accepted tokens' features (`ds_feat[0..k]`) into DSpark's KV + `frnn`!
   - **Overlap Trick (Dual `hipStreamNonBlocking` Streams):** As soon as visit 48 finishes producing `ds_feat[0..nd]` (BEFORE `output.weight` `lm_head` and `topk` finish!), record `ev_feat_ready` on `stream_main` and launch a speculative `ds_ctx` pre-projection (`dflare_mix` + `K/V` projections for all `nd+1` rows) on `stream_ds` **concurrently with target `lm_head` (`output.weight`) + host rejection sampling**! Once `k` is known, only the `k+1` accepted rows commit to DSpark's KV/frnn (`0.02 ms`)!
   - **Also overlap:** `dp_flush` / `snap_kv` D2H copies on `b->snap_cs` (`hip.c:6646`, already `hipStreamNonBlocking`!) with decode steps so checkpointing NEVER adds a single millisecond to TTFT or inter-token latency!
2. **Recipe W7-6 — Arena Lifetime Compaction (`src/model.c`) + Cold-Expert Host-Backed VRAM Cache (`llama.cpp` PR #29887, `src/hip.c`):**
   - **Arena Compaction:** Audit `IE_PLAN_DUMP=1` after Wave 2/3 fusions (especially once `exp_gu` is capped to `T_cap = 16` saving `-34.9 MiB`, `k_buf`/`v_buf` eliminated by direct `f16` KV store, and `exp_down` eliminated by Fusion B!). Every `100 MiB` reclaimed from `g->arena` adds `~2,800` extra tokens of **eviction-free WDDM VRAM headroom**!
   - **Long-Context (`32K–128K`) Cold-Expert VRAM Cache Tier (PR #29887 pattern, opt-in for `ctx > 32K`):**
     - At `32K` ctx, Inkling-2 sits right near Windows WDDM's `~12.5–13.0 GiB` cliff (`12.07 GB` weights + `~1.1 GB` KV + DSpark). If the user pushes to `64K–128K` ctx (`+1.2–3.5 GB` KV), WDDM silently evicts random looped MoE layers `6..17` to PCIe (`400 -> 26 GB/s`, 4× slowdown!).
     - Instead of letting WDDM blindly evict hot looped layers: pin the **12 looped core layers (`blk.6..17`, executed 3×/tok)** in VRAM, and allow non-looped outer MoE layers (`blk.0..5, 18..23`, executed 1×/tok) to use a **PR #29887 fixed-slot LRU VRAM expert cache** (e.g., `64/128` hot expert slots in VRAM, cold experts in `hipHostMalloc` pinned RAM fetched asynchronously on miss!) -> graceful degradation (`~10–15%` slower at 64K–128K) instead of a `3.5×` WDDM eviction cliff!

---

# WAVE 8 (Rounds 36–40): Full-Stack Composition, Workload Auto-Tuning & Final Parity/Regression Certification

| Round | Focus | Concrete Action & Acceptance Gate |
|---|---|---|
| **R36** | **Combined Full-Stack Re-Tuning** | Re-sweep DSpark `(draft=5..8, pmin/survival_tau, temp_d=0.85..1.0)` + `chunk={512,1024}` + `IE_FLUSH_EVERY={0,64,128}` on the fully fused Wave 1–7 stack (optimal cutoffs shift once `ds_draft < 0.9 ms` and `T=6..8` verify drops by `35%+`!). |
| **R37** | **Live `ie-serve` Multi-Turn & Side-Request Stress Gate** | 32K multi-turn coding + interleaved 152-tok title/summary side requests + cold disk-cache restart + `DRY=0.8` 4K generation + `temp=1.0` sampled chat acceptance & `sampling_ms_step` verification. |
| **R38** | **VRAM / WDDM Cliff & Long-Context `32K` Endurance Audit** | Verify zero WDDM eviction across `4K / 16K / 32K` with `IE_VRAM_TRACE=1`, confirm peak arena + KV + DSpark fits comfortably inside WDDM budget with `-BudgetGiB 20`. |
| **R39** | **Saluki & 3D-Coder Shared-Binary Regression Gate** | Run `bench/saluki-cache-check-task.ps1` and `bench/coder3d-serve-smoke-task.ps1` inside `-BudgetGiB 20` memory wrapper to prove **0 regressions** across all shared `amd-infer` engines! |
| **R40** | **Default Mask Promotion, Cleanup & Final 40-Round Dashboard** | Promote ONLY measured wall-clock winners passing strict parity gates (`0/0` spills, exact greedy SHA, `KLD <= 0.001` for non-exact/probabilistic), update `serve-inkling2.ps1` + `AGENTS.md`, and render final `results/inkling2/40round-dashboard.html`. |

---

## Verified Reference Index (Waves 5–8)

1. **FLUTE:** Guo et al., *Fast Matrix Multiplications for Lookup Table-Quantized LLMs*, [arXiv:2407.10960](https://arxiv.org/abs/2407.10960) | [`HanGuo97/flute`](https://github.com/HanGuo97/flute)
2. **Marlin & ExLlamaV3:** [`IST-DASLab/marlin`](https://github.com/IST-DASLab/marlin) | [`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3)
3. **`llama.cpp` & `ik_llama.cpp`:** [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) (PR `#32065` few-row MMQ `MUL_MAT_ID`, PR `#29887` GPU cache for host MoE experts, PR `#9702` DRY) | [`ikawrakow/ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp)
4. **GPU Top-K & Fused Head:** **RadiK** ([arXiv:2501.14336](https://arxiv.org/abs/2501.14336), [`leefige/radik`](https://github.com/leefige/radik)) | **AIR Top-K** ([`NVIDIA/cccl`](https://github.com/NVIDIA/cccl)) | [`lsnchow/LLaDa-CUDA-fused-kernel`](https://github.com/lsnchow/LLaDa-CUDA-fused-kernel) | [`ROCm/composable_kernel`](https://github.com/ROCm/composable_kernel)
5. **Speculative Sampling & DSpark/ForeSpark:** Leviathan et al. ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192)) | Chen et al. ([arXiv:2302.01318](https://arxiv.org/abs/2302.01318)) | **DSpark** ([arXiv:2607.05147](https://arxiv.org/abs/2607.05147), [`deepseek-ai/DeepSpec`](https://github.com/deepseek-ai/DeepSpec)) | **ForeSpark** ([arXiv:2611.13442](https://arxiv.org/abs/2611.13442), [`facebookresearch/ForeSpark`](https://github.com/facebookresearch/ForeSpark)) | **vLLM Inkling-2 PR #69028** & [`vllm/v1/sample/rejection_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py)
6. **Certified Vocab Shortlists:** **FR-Spec** ([arXiv:2502.14856](https://arxiv.org/abs/2502.14856), [`thunlp/FR-Spec`](https://github.com/thunlp/FR-Spec)) | **MicroSpec** ([arXiv:2605.26444](https://arxiv.org/abs/2605.26444)) | **SpecPilot** ([arXiv:2611.02501](https://arxiv.org/abs/2611.02501))
7. **Suffix / Lookup / Tree Speculation:** **SAM Decoding** ([arXiv:2411.10666](https://arxiv.org/abs/2411.10666)) | **SuffixDecoding** ([arXiv:2411.04975](https://arxiv.org/abs/2411.04975), [`snowflakedb/ArcticInference`](https://github.com/snowflakedb/ArcticInference)) | **Lookahead** ([arXiv:2402.02057](https://arxiv.org/abs/2402.02057), [`hao-ai-lab/LookaheadDecoding`](https://github.com/hao-ai-lab/LookaheadDecoding)) | **PLD** ([`apoorvumang/prompt-lookup-decoding`](https://github.com/apoorvumang/prompt-lookup-decoding)) | **EAGLE-2** ([arXiv:2406.16858](https://arxiv.org/abs/2406.16858), [`SafeAILab/EAGLE`](https://github.com/SafeAILab/EAGLE)) | **Sequoia** ([arXiv:2402.12374](https://arxiv.org/abs/2402.12374), [`Infini-AI-Lab/Sequoia`](https://github.com/Infini-AI-Lab/Sequoia)) | **SpecInfer** ([arXiv:2305.09781](https://arxiv.org/abs/2305.09781))
8. **WDDM / Persistent / Piecewise Graphs:** [`ROCm/clr`](https://github.com/ROCm/clr) (`rocclr/platform/kernel.cpp` `captureHIPArgs`, Issue `#13762`) | [`vllm/compilation/cuda_piecewise_backend.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/cuda_piecewise_backend.py) | **Mirage MPK** ([arXiv:2512.22219](https://arxiv.org/abs/2512.22219), [`mirage-project/mirage`](https://github.com/mirage-project/mirage)) | **MonoMoE** ([arXiv:2609.04244](https://arxiv.org/abs/2609.04244))
