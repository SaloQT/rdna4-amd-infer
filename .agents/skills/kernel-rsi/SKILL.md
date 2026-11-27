---
name: kernel-rsi
description: Optimise this engine's gfx1201 GPU kernels one at a time in a measured improve-until-it-stops loop. Each edit is built, checked bit-identical against the best version so far on a captured real launch, timed cold on the GPU, kept or reverted, and logged to a JSONL ledger that report.mjs turns into charts. Use when asked to optimise, tune or speed up kernels, run the kernel RSI loop, find the slowest kernels, or split kernel work across a /swarm (coordinator, one worker per kernel, verifier).
---

# Kernel RSI loop

Make one kernel faster through many small measured attempts, each built on the best version so far, and stop when the
attempts stop paying. The script `scripts/rsi.mjs` does all the measuring and bookkeeping. You supply the ideas and
the edits, one idea per attempt, so that every gain and every failure is attributable to a single change.

All commands run from the repository root: `node .agents/skills/kernel-rsi/scripts/rsi.mjs <command>`, written
`rsi` below. `rsi` with no command prints its usage.

## How it measures

- **Capture.** `IE_KCAP` (`engine/src/hip.c`) records one real launch of a kernel during a benchmark scenario: grid,
  arguments, and every byte of memory it reads. `engine/tools/kbench.c` replays it alone.
- **Exactness.** Every attempt must produce output bit-identical to the best version on the captured inputs. The
  project gates every round on exact output hashes, so anything else is useless here.
- **Timing.** Cold (8 rotating copies of the captured memory so each launch reads DRAM, as inside the model), 15
  interleaved rounds. The noise floor is about ±0.5%. An attempt is kept only if it is at least 1% faster twice in a
  row and not over 1% slower on the held-out captures (another context length or verify width).
- **One GPU job at a time.** Every GPU step takes the lock `<runsDir>/.gpu-lock` and also waits while the lab
  harness holds `ref/.gpu-lock`, so parallel workers queue instead of disturbing each other's timings. Building runs
  on the CPU and needs no lock.
- **Stopping.** The loop for a kernel stops after 8 attempts without a kept gain, after 40 attempts, or when the last
  three kept gains total under 1.5%. Thresholds live in `scripts/rsi.config.json`.

## Setup (once per engine build)

1. Build the engine and the replay tool if they are older than the source:
   `sh engine/tools/build-host.sh run` and `gcc -std=c11 -O2 -o engine/build/kbench.exe engine/tools/kbench.c`
   (inside `engine/`). The kernel code objects in `engine/build/` must match the source; `rsi init` says so when they
   do not.
2. Rank the kernels of a scenario: `rsi discover vb16k_t6`. It traces one verify benchmark, captures every kernel
   that ran, times each alone, and writes `queue-vb16k_t6.json`, sorted by milliseconds per run. Other scenarios:
   `t1_16k` (one-token decode), `vb4k_t6`, `vb16k_t8`, `pf4k` (prefill). Each engine run takes about a minute; the
   captures need several GB of disk.
3. `rsi queue` shows the ranking and each kernel's loop state; `rsi next` names the highest-ranked kernel whose loop
   has not stopped. Skip kernels marked NOT DETERMINISTIC (atomics decide some output bytes, so two runs of the same
   build differ and bit-identity cannot judge an edit) or REPLAY MISMATCH (recapture with `rsi capture`).

## The loop for one kernel

1. `rsi init KERNEL` copies `engine/kernels` and `engine/core` into the kernel's working directory, builds the
   baseline, captures the held-out scenarios, and measures the baseline. It prints the path of the working copy.
   Edit only that copy; the repository is touched only when a verified result lands.
2. Repeat until `rsi eval` prints `STOP`:
   1. `rsi status KERNEL` lists the attempts kept so far, every idea already tried and why it was dropped, outcomes
      by tag, and the stop state. Read it before each idea so you build on what worked and never repeat a failure.
   2. `rsi isa KERNEL` shows registers, occupancy and the instruction mix of the hottest loops of the current best
      build (`--dump N` prints loop N). Read the kernel source in the working copy next to it.
   3. Pick one idea and write down what you expect it to change in the ISA or the timing, and why.
   4. Make the smallest edit in `work/kernels` that tests the idea.
   5. `rsi eval KERNEL --idea "one line: what changed and why" --tag TAG`. TAG is one of memory, alu, parallelism,
      sync, schedule, occupancy, algorithm. It builds, checks registers and spills, checks exactness, times, keeps or
      reverts the working copy, logs the attempt, and prints the decision and the stop state.
   6. If the result surprised you, look at why (`rsi isa KERNEL` on the new best, or reason about the rejected diff in
      `attempts/NNN.diff`) before the next idea. After three misses in a row, `status` suggests a different tag:
      switch kinds of idea rather than polishing one that has stopped working.
3. `rsi verify KERNEL` re-measures baseline against best on every capture with more rounds and writes `best.diff`.
4. Landing a verified result is a normal change to the repository: on a work branch, `git apply` the kernel's
   `best.diff` (repo-relative) from the repository root, rebuild with `engine/build-win.sh`, run the full benchmark
   harness for the exact output hashes and the end-to-end speed, then commit and open a PR. The single-kernel gain only
   counts once the whole model still produces the reference hashes.

Decisions in the ledger: `accepted`, `no_gain`, `slower`, `not_exact` (output differs), `not_confirmed` (the gain did
not repeat), `holdout_fail`, `spill`, `compile_fail`, `suspect` (faster than 2x in one step, held for a verifier),
`error`.

## Where ideas come from

Good attempts are specific to what the ISA shows. Useful directions on gfx1201 (RDNA4, wave32, 64 CUs, 64 MB
Infinity Cache, 640 GB/s):

- **memory**: wider loads (`global_load_b128`), more loads in flight before the first wait, one load shared by
  several outputs, scalar loads for wave-uniform data, fewer passes over the same bytes.
- **alu**: work done identically in every lane (reductions, `expf`, scale math) done once per wave or per lane
  group; operand layouts that let pairs of FMAs dual-issue (VOPD); integer tricks in place of conversions.
- **parallelism**: workgroup counts that fill 64 CUs without a straggling last round; splitting serial chains.
- **sync**: fewer `s_barrier`s; staging through LDS so waves do not wait on each other per step.
- **schedule**: removing `s_wait_*cnt 0` chains, prefetching the next iteration, loop shapes that keep constant
  register indices.
- **occupancy**: VGPR counts below the steps at 96 (16 waves/SIMD) and 120 (12); LDS per workgroup.
- **algorithm**: a different decomposition with the same arithmetic order.

Rules that keep outputs bit-identical: keep the order of every floating-point reduction (the same pair tree, the
same sequential order per output); keep separate roundings separate (`#pragma clang fp contract(off)` where the
compiler might fuse); keep the same math functions (`__builtin_expf` is not `exp2` of a scaled argument). Reordering
independent work, loads, barriers and integer math is free.

Before trying an idea, check `docs/40round-ledger.json` and the README round notes for rounds that already tried it
on this kernel; the notes say why rejected ideas failed.

## What a valid speedup is

The capture fixes the grid, the arguments and the inputs, and the rules below keep the gain real outside the replay:

- Do the same work for any input. A kernel that recognises the captured data, a grid size or a pointer value and
  takes a shortcut is wrong even when the replay passes.
- Keep the kernel's signature, grid contract and the meaning of every output byte, including counters and partial
  buffers other kernels read.
- Change only kernel sources under `work/kernels`. Host changes (launch geometry, fusing launches) are real
  improvements but need a separate full-model change and the harness, not this loop.
- Treat anything faster than about 2x as wrong until shown otherwise; the loop holds it as `suspect`.

## Running it as a /swarm (Antigravity) or with subagents

The kernels are independent pieces of work: each has its own working copy, captures, ledger and state file, and the
GPU lock serialises the measurements. That is the split `/swarm` wants. A prompt for the coordinator:

> /swarm Optimise the kernels in the kernel-rsi queue for scenario vb16k_t6 with the kernel-rsi skill. Give each of
> the top N kernels from `node .agents/skills/kernel-rsi/scripts/rsi.mjs queue vb16k_t6` its own worker. A worker owns
> only its kernel's working copy and runs the loop in the skill until `rsi eval` prints STOP, then runs `rsi verify`.
> Add one verifier. Done means every worker's kernel is stopped and verified, and the verifier has signed off.

- **Coordinator**: assigns kernels from `rsi queue` (or `rsi next` as workers finish), at most about 4 workers at a
  time because the GPU lock makes more of them wait. Runs `node .agents/skills/kernel-rsi/scripts/report.mjs` at the
  end and posts the summary.
- **Worker `@KERNEL`**: owns `<runsDir>/kernels/KERNEL/` only. Runs the loop above. After each `rsi eval`, posts one
  line to the board: `@coordinator KERNEL #N DECISION DELTA% best X us (xSPEEDUP)`. On STOP runs `rsi verify KERNEL`
  and posts the result. Never edits `engine/` in the repository.
- **Verifier `@verifier`**: for each stopped kernel, reads every accepted diff in `attempts/` for shortcuts and
  input-specific behaviour (the rules above), looks at any `suspect` attempt, runs `rsi verify KERNEL` itself, and
  replies to the worker with sign-off or the problem. A verifier on a different model than the workers catches more.
- Landing is done once, by the coordinator or the user, after the swarm: the full harness run needs the whole GPU.

Claude Code subagents work the same way: one subagent per kernel with the worker brief above, one reviewer with the
verifier brief.

## Data files and charts

Everything lives under `runsDir` from `scripts/rsi.config.json`:

- `queue-SCENARIO.json`: kernels with launches per run, microseconds per launch (cold, alone), ms per run, share.
- `captures/SCENARIO/KERNEL.kcap`: the captured launches.
- `kernels/KERNEL/ledger.jsonl`: one JSON object per attempt: `ts, kernel, attempt, idea, tag, worker, files, diff,
  decision, reason, res {vgpr, sgpr, lds, waves, vspill, sspill, priv}, ref_us, cand_us, delta_pct, confirm_pct,
  holdouts[], best_us, speedup, wall_s`. Attempt 0 is the baseline.
- `kernels/KERNEL/state.json`, `best-src/`, `best.hsaco`, `baseline.hsaco`, `attempts/NNN.diff`, `work/`.

`node .agents/skills/kernel-rsi/scripts/report.mjs` merges every ledger into `all.jsonl` and `summary.csv` and writes
`report.html`: speedup and milliseconds saved per kernel, and per kernel every attempt coloured by decision over the
best-so-far line.
