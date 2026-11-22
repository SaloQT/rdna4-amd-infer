# TileLang experiments (2026-10-10, amd-gpu-host, R9700 / gfx1201)

Environment: container `stilldeadcode/vllm-radiance:0.9.3` (TileLang 0.1.10, PyTorch 2.11 + ROCm 7.14), with these set at run time:
`ROCM_PATH=/opt/rocm/core-7.14 LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib`, `tilelang.compile(..., target="auto")`.
Stop ie-serve before running the experiments (it uses 30.8 GB of VRAM).

| File | Contents |
|---|---|
| `q4k_gemm_current_tiling.py` | The current tiling of `ie_gemm_q4kr` (description only; TileLang 0.1.10 has no int8 matrix instructions on RDNA, so it cannot run) |
| `gemm_fp16_sweep.py` | PyTorch fp16 / int8 matrix multiplication, plus a tiling sweep of TileLang fp16 matrix multiplication |
| `q4k_dequant_fp16.py` | Q4_K (the engine's data layout) dequantized to fp16 inside the kernel, then fp16 matrix instructions; compared against a PyTorch reference |

Shape 512 × 34816 × 5120 (one prefill chunk of 512 tokens, the largest Q4_K matrix):

| Implementation | ms | trillion ops/s |
|---|---|---|
| PyTorch fp16 (hipBLASLt) | 1.269 | 143.8 |
| PyTorch int8 (`torch._int_mm`) | 0.859 | 212.5 |
| TileLang fp16, 512 × 64 × 32, 256 threads | 1.60 | 114 |
| TileLang fp16, 128 × 128 × 32 | 4.08 | 44.7 |
| TileLang Q4_K → fp16, 512 × 64 × 32 | 1.827 | 99.9 (equivalent), relative error 2.2e-4 |
| Engine `ie_gemm_q4kr` (int8, `IE_PROFILE=2`) | about 1.96 | about 93 |

Takeaway: fastest when one workgroup covers all 512 tokens of the chunk, so the weights are read from VRAM only once per chunk. Configurations with more than 64 KB of LDS fail to launch.

## Comparison in the same timing setup (`ab_q4k.py`, 2026-10-10)

The engine's hsaco is loaded directly with ctypes and timed the same way as TileLang. 512 tokens per chunk:

| Shape (rows × K) | `ie_gemm_q4kr` (int8) | `ie_gemm_q4k_h` (fp16, 512 / 256-token chunks) | TileLang fp16 + conversion |
|---|---|---|---|
| 34816 × 5120 | 1.953 ms | 2.206 / 2.336 | about 1.86 |
| 5120 × 17408 | 0.966 ms | 1.312 / 1.264 | - |
| 10240 × 5120 | 0.583 ms | 0.718 / 0.736 | - |

The int8 → fp16 conversion (`ie_q8_f16`) takes 0.02 to 0.05 ms. In-engine timing (`IE_PROFILE=2`) matches these numbers (1.96 vs 1.953 ms).
Conclusion: the TileLang fp16 dequantization version is about 5% faster than the int8 version; the hand-written fp16 version is slower, so it is off by default (`IE_GEMM_H=1` turns it on).

## Why the hand-written fp16 version is slower than TileLang (2026-10-10)

`ab_q4k.py` now also calls the TileLang kernel with the same input data, after a 3-second warm-up (without the warm-up the first shape is more than 30% slower). 512 tokens per chunk, ms:

| Implementation | 34816 × 5120 | 5120 × 17408 | 10240 × 5120 |
|---|---|---|---|
| int8 `ie_gemm_q4kr` | 1.957 | 1.006 | 0.586 |
| fp16 hand-written, register prefetch of the next block (original) | 2.226 | 1.370 | 0.683 |
| fp16 hand-written, no prefetch (current `ie_gemm_q4k_h`) | 1.982 | 1.269 | 0.661 |
| fp16 hand-written, no prefetch, read all fragments of one K half-step first | 2.154 | 1.283 | 0.676 |
| TileLang fp16 (512 × 64 × 32) | 1.988 | 1.178 | 0.610 |

1. The main cause is register prefetch: without it, the largest matrix drops from 2.226 to 1.982 ms, on par with TileLang.
2. The smaller matrices are still 5% to 8% behind. Guess: each VRAM read does a 64-bit address addition (the row width is a runtime parameter; in TileLang it is a compile-time constant, using immediate offsets), and the work is split across waves differently (TileLang: each wave handles 128 tokens × 32 rows). Not verified.
3. TileLang's times vary quite a bit between runs: 1.83, 1.92, 1.99 ms. The earlier "TileLang is about 5% faster than int8" came from the faster run; in repeated measurements the two are level, and on the other two shapes TileLang is slower.
4. Conclusion: on these three shapes, the fp16 dequantization approach is never faster than the current int8 kernel.

## Remaining gap: controlled experiments on the row-width constant and LDS swizzle (2026-10-10, `ab_q4k_layout.py`)

Two experiments, each run twice (ms, 512 tokens per chunk). The experiment kernels are not kept in the code:

| Implementation | 34816 × 5120 | 5120 × 17408 | 10240 × 5120 |
|---|---|---|---|
| int8 `ie_gemm_q4kr` | 1.955 / 2.046 | 0.998 / 1.016 | 0.596 / 0.597 |
| fp16 `ie_gemm_q4k_h` (current) | 2.056 / 2.110 | 1.277 / 1.285 | 0.640 / 0.682 |
| A: row width 5120 as a compile-time constant | 1.999 / 2.123 | - | 0.660 / 0.650 |
| B: LDS swizzle (64-byte rows, block index ^ ((row >> 2) & 3); LDS 46 KB → 37 KB) | 1.993 / 2.081 | 1.255 / 1.259 | 0.687 / 0.694 |
| A + B | 2.109 / 2.114 | - | 0.680 / 0.678 |
| TileLang | 1.984 / 1.985 | 1.176 / 1.184 | 0.607 / 0.617 |

1. The two runs differ by 3% to 6%; the changes from A and B are within that range, or slower. So neither the 64-bit address computation nor the LDS layout (and the higher workgroup residency that B brings) explains the gap.
2. The machine code of the inner loop is almost identical: per K step, 32 matrix instructions, 16 vs 20 LDS reads, 9 vs 8 LDS writes, 8 vs 8 VRAM reads.
3. The remaining gap exists only on the two smaller shapes (5% to 8%); the cause was not found. Possibilities not verified: register count (235 vs 202, which affects the number of resident waves), how the weight reads are coalesced, how the results are written back.
