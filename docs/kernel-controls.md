# Kernel masks and rollback controls

Companion to the [README](../README.md). Every promoted kernel can be switched off with an environment variable, so a regression can be bisected without rebuilding.

- **Inkling-2 Defaults:** `IE_INK_DEFAULT_MASK = 0xfe7dfc0369c3ffff`, `IE_INK_DEFAULT_MASK2 = 0x20001c7f103d36b`, `IE_INK_DEFAULT_MASK3 = 0x3` (192-bit mask, plus promoted `R53 - R89` env-gated kernels/hooks with explicit `=0` opt-outs)
  - Disable all custom Inkling kernels: `IE_INKLING_KERNELS=0`
  - Individual opt-out toggles: `IE_INK_H17_EXACT=0`, `IE_INK_H20_Q4=0`, `IE_VIS_ROPE_TBL=0`, `IE_VIS_ROPE_K=0`, `IE_INK_R17_DS_QKV=0`, `IE_INKLING_IQ4_REPACK=0`, `IE_INK_R17_IQ4_PF64=0`, `IE_INK_ROUTER_R13=0`, `IE_INK_R17_FLUTE_PF64=0`, `IE_INKLING_Q8_REPACK=0`, `IE_H38_ED_OVERLAP=0`, `IE_SAM_PERSIST=0`, `IE_DS_SHORTLIST=0`, `IE_DS_SHORTLIST_TIER2=0`, `IE_DS_ADAP_D=0`, `IE_SAM_LOOKUP=0`, `IE_EPHEMERAL_CACHE=0`, `IE_EPHEM_FAST=0`, `IE_DISK_IORING=0`, `IE_KV_H2D_PIPE=0`, `IE_DISK_PREWARM=0`, `IE_TOK_FAST_BPE=0`, `IE_TOK_THREADS=1`, `IE_TOK_SEG_CACHE=0`, `IE_DRY_HASH=0`, `IE_DS_PROB=0`, `IE_DS_SPLIT=0`, `IE_SNAP_HOST=0`, `IE_DEFER_CKSAVE=0`, `IE_PACE_FIRST_FAST=0`
  - Long-context 128K-256K controls (auto-enabled by `serve-inkling2.ps1` at `-Ctx > 131072`): `IE_INK_GLOB_KV=q4rot` (global-only `Q4_0 + FWHT-128` KV), `IE_DS_GLOB_WIN=8192` (or `auto16k`), `IE_DS_ROPE_OTF=1` (on-the-fly FP64 Cody-Waite DSpark RoPE, auto-on at `n_ctx >= 131072`), `IE_MOE_CACHE_SLOTS=auto` (outer-layer host-backed VRAM SLRU expert cache)
  - Opt-in experiments kept off by default: `IE_DISK_LZ4=1` + `IE_DCACHE_UNSHUF=1` (v4 byte-shuffled LZ4 disk cache + GPU unshuffle), `IE_RAM_CACHE_GB=N` (OS-guarded Host-RAM Radix L2 / RAM-only mode), `IE_INKLING_Q4K_REPACK=1`, `IE_INK_R17_FLUTE_VER=1`, `IE_INK_R17_SWA_PF=1`, `IE_INKLING_GRAPH=2` (piecewise HIP graphs), `IE_DS_TREE=1` (8-node tree verify), `IE_DS_FORESPARK=1`, `IE_DS_OVERLAP=1`, `IE_MK_INKLING=1`
- **Saluki-27B Defaults:** `IE_SALUKI_KERNEL_MASK = 0x67e8de841ffffffff` (81 compiled kernels across R0-R11: `rq128`/`rq64` raw-quant WMMA, `attn_pl_r10`, `attn_long_mp_r11`, `gdn_wy_wmma_r11`, `qkn_rope_kv_r11`, `lora_b_r11`, `topk_r11`)
  - Disable all custom Saluki kernels: `IE_SALUKI_KERNELS=0`
  - KV modes: `-Kv q8` (default, 128K), `-Kv q4` (`IE_KV_Q4=1 IE_KV_ROT=1` rotated 4-bit), `-Kv f16`
- **Re-render the Interactive Dashboard:**
  ```bash
  python scripts/render_40r_dashboard.py \
    --in docs/40round-ledger.json --out docs/index.html
  ```
