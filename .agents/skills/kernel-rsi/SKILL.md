---
name: kernel-rsi
description: Optimise this engine's gfx1201 GPU kernels automatically, one at a time, in a measured improve-until-it-stops loop that lands each result on main. Each edit is built, checked bit-identical against the best version so far on a captured real launch, timed cold on the GPU, kept or reverted, and logged to a JSONL ledger that report.mjs turns into charts; a finished kernel is verified, rebuilt, gated on the exact output hashes and merged. Use when asked to optimise, tune or speed up kernels, run the kernel RSI loop, find the slowest kernels, or split kernel work across a /swarm or subagents.
---

# Kernel RSI loop

Make the engine's kernels faster through many small measured attempts, each built on the best version so far, stop
each kernel when its attempts stop paying, and land every gain on `main`. The script `scripts/rsi.mjs` does all the
measuring, bookkeeping, ranking and landing. You supply the ideas and the edits, one idea per attempt, so that every
gain and every failure is attributable to a single change.

All commands run from the repository root: `node .agents/skills/kernel-rsi/scripts/rsi.mjs <command>`, written
`rsi` below. `rsi` with no command prints its usage.

## The cycle (repeat until claim prints "none")

1. `rsi claim vb16k_t6 --worker NAME` hands you the next kernel. It keeps the queue current by itself: when the engine
   build changed since the queue was made (a kernel landed, the engine was rebuilt), it traces, captures and times
   every kernel again before choosing. It ranks by each kernel's current time (a running loop's best so far), skips
   kernels that are finished, claimed by a live worker, below 0.5% of the step, or not judgeable (output decided by
   atomics), and sets up the kernel's loop if it has none. It prints the working copy to edit and what to do next.
   When it prints `none`, everything worth doing is done: stop.
2. If claim says the loop has stopped, go to step 4. Otherwise optimise, one attempt at a time, until `rsi eval`
   prints `STOP`:
   1. `rsi status KERNEL`: attempts kept so far, every idea already tried and why it was dropped, outcomes by tag,
      the stop state, and after a restart on newer engine code the earlier wins to re-apply first. Read it before each
      idea, so you build on what worked and never repeat a failure.
   2. `rsi isa KERNEL`: registers, occupancy and the instruction mix of the hottest loops of the current best build
      (`--dump N` prints loop N). Read the kernel source in the working copy next to it.
   3. Pick one idea and write down what you expect it to change in the ISA or the timing, and why.
   4. Make the smallest edit in the working copy (`work/kernels`) that tests it.
   5. `rsi eval KERNEL --idea "one line: what changed and why" --tag TAG --worker NAME`, TAG one of memory, alu,
      parallelism, sync, schedule, occupancy, algorithm. It builds, rejects spills, checks exactness, times, keeps or
      reverts the working copy, logs the attempt, and prints the decision and the stop state.
   6. When a result surprises you, look at why (`rsi isa` on the new best, or the rejected `attempts/NNN.diff`) before
      the next idea. After three misses in a row, `status` names a different tag: switch kinds of idea.
3. Stop when eval prints `STOP` (8 attempts without a kept gain, 40 attempts, or the last three gains under 1.5% in
   all; thresholds in `scripts/rsi.config.json`).
4. `rsi finish KERNEL` ends the kernel. With no gain it marks the kernel done. Otherwise it re-measures baseline against
   best on every capture and, if that holds, lands it: on a fresh branch from an up-to-date `main` it applies the
   patch, rebuilds everything with `engine/build-win.sh`, runs the four exact-output stages of the benchmark harness
   against the latest promoted round's hashes, then commits, pushes, opens a PR and rebase-merges it. If the patch no
   longer applies (another kernel landed in the same file), the loop restarts on the new code at the next claim,
   keeping its history. If the gate fails, `main` and `engine/build` are restored and the kernel is marked done with
   the reason. Landing waits while the repository has uncommitted changes or is not on `main`.
5. Back to step 1.

Every step that uses the GPU takes `<runsDir>/.gpu-lock` (and waits while the lab harness holds `ref/.gpu-lock`), and
landing also takes `<runsDir>/.repo-lock`, so any number of workers can run the cycle at once: they queue for the GPU
instead of disturbing each other's timings, and only one changes the repository at a time.

## How it measures

- **Capture.** `IE_KCAP` (`engine/src/hip.c`) records one real launch of each kernel of a benchmark scenario: grid,
  arguments, and every byte of memory it reads. `engine/tools/kbench.c` replays one alone. Each loop keeps its own
  hard links to its captures, so a re-discovery never changes them under a running loop.
- **Exactness.** Every attempt must produce output bit-identical to the best version on the captured inputs; the
  project gates every round on exact output hashes, and `finish` checks them again on the whole model before landing.
- **Timing.** Cold (8 rotating copies of the captured memory, so each launch reads DRAM as inside the model), 15
  interleaved rounds; the noise floor is about ±0.5%. An attempt is kept only if it is at least 1% faster twice in a
  row and not over 1% slower on the held-out captures (another context length and verify width).

Decisions in the ledger: `accepted`, `no_gain`, `slower`, `not_exact` (output differs), `not_confirmed` (the gain did
not repeat), `holdout_fail`, `spill`, `compile_fail`, `suspect` (faster than 2x in one step, held back), `error`; and
the markers `baseline`, `rebase` (restart on newer code), `verified`, `verify_fail`, `landed`, `land_fail`,
`land_conflict`, `land_blocked`, `land_error`, `done`.

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
- Change only kernel sources under the working copy. Host changes (launch geometry, fusing launches) are real
  improvements but need a separate full-model change, not this loop.
- Treat anything faster than about 2x as wrong until shown otherwise; the loop holds it as `suspect`.

## Running it as a /swarm (Antigravity) or with subagents

`/swarm Optimise the kernels in the kernel-rsi queue for scenario vb16k_t6 with the kernel-rsi skill` is enough. The
roles:

- **Coordinator**: starts up to 4 workers (the GPU lock makes more of them wait) and one verifier, and gives each
  worker the brief below. When every worker reports that claim printed `none`, runs
  `node .agents/skills/kernel-rsi/scripts/report.mjs` and posts the summary from `summary.csv` with the PR links.
- **Worker `@worker-N`**: runs the cycle above with `--worker worker-N`, kernel after kernel, until claim prints
  `none`. Posts one board line per attempt (`@coordinator KERNEL #N DECISION DELTA% best X us xSPEEDUP`) and one per
  finish (landed with the PR link, or why not). Edits only the working copy claim printed; never edits `engine/` in
  the repository.
- **Verifier `@verifier`**: reviews each kernel a worker finishes: reads every accepted `attempts/NNN.diff` for
  shortcuts and input-specific behaviour (the rules above) and looks at any `suspect` attempt. A landed change that
  breaks the rules is reverted with a PR of its own, and the board says why. A verifier on a different model than
  the workers catches more.

Claude Code subagents work the same way: a few subagents with the worker brief, one with the verifier brief.

## Data files and charts

Everything lives under `runsDir` from `scripts/rsi.config.json`:

- `queue-SCENARIO.json`: the build hash, and per kernel launches per run, microseconds per launch (cold, alone),
  ms per run and share at discovery. `rsi queue` shows it ranked by current time with each loop's state and worker.
- `captures/SCENARIO/KERNEL.kcap`: the captured launches.
- `kernels/KERNEL/ledger.jsonl`: one JSON object per attempt or event: `ts, kernel, attempt, idea, tag, worker, files,
  diff, decision, reason, res {vgpr, sgpr, lds, waves, vspill, sspill, priv}, ref_us, cand_us, delta_pct,
  confirm_pct, holdouts[], best_us, speedup, wall_s` (landing records add `pr`, `metrics`, `checks`).
- `kernels/KERNEL/state.json`, `claim.json`, `captures/`, `best-src/`, `best.hsaco`, `baseline.hsaco`,
  `best.diff`, `attempts/NNN.diff`, `work/`.

`node .agents/skills/kernel-rsi/scripts/report.mjs` merges every ledger into `all.jsonl` and `summary.csv` and writes
`report.html`: speedup and milliseconds saved per kernel, and per kernel every attempt coloured by decision over the
best-so-far line.
