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

## The cycle (repeat until claim prints "none" or "busy")

Use one worker name throughout (`--worker NAME`), different from every other worker's.

1. `rsi claim vb16k_t6 --worker NAME` hands you the next kernel. First it checks the machine (`rsi doctor`): tools,
   models, disk space, an interrupted landing left in the repository, and whether `engine/build` and `kbench` are
   current (it rebuilds them when not). It refuses to start while the repository has uncommitted changes under
   `engine/`. When the engine build changed since the queue was made (a kernel landed, the engine was rebuilt), it
   traces, captures and times every kernel again. It ranks by each kernel's current time, skips kernels that are
   finished, claimed by a live worker, waiting to land, below 0.5% of the step, or not judgeable (output decided by
   atomics), sets up the kernel's loop if it has none, and prints the working copy and what to do next.
   - `none`: nothing is left to do (it lists kernels that need a person: blocked 8 times in a row, or waiting for
     `rsi approve`). Stop.
   - `busy`: the rest is claimed by other live workers. Stop; they pick up any kernel waiting to retry.
   - `wait`: kernels are waiting to retry a landing (claim already waited up to 10 minutes for them). Run claim again.
2. If claim says the loop has stopped, go to step 4. Otherwise optimise, one attempt at a time, until `rsi eval`
   prints `STOP`:
   1. `rsi status KERNEL`: attempts kept so far, every idea already tried and why it was dropped, outcomes by tag,
      the stop state, and after a restart on newer engine code the earlier wins to re-apply first. Read it before each
      idea, so you build on what worked and never repeat a failure.
   2. `rsi isa KERNEL`: registers, occupancy and the instruction mix of the hottest loops of the current best build
      (`--dump N` prints loop N). Read the kernel source in the working copy next to it.
   3. Pick one idea and write down what you expect it to change in the ISA or the timing, and why.
   4. Make the smallest edit that tests it, in the working copy's `kernels/` only. Do not run another eval of the same
      kernel at the same time.
   5. `rsi eval KERNEL --idea "one line: what changed and why" --tag TAG --worker NAME`, TAG one of memory, alu,
      parallelism, sync, schedule, occupancy, algorithm. It builds, rejects spills, checks exactness, times, keeps or
      reverts the working copy, logs the attempt, and prints the decision and the stop state. Two results are not
      attempts and leave your edit in place: `NOISY` (the GPU timed the unchanged kernel far from its usual time, e.g.
      another program used the GPU) and `BUSY` (the GPU stayed in use). Run the same eval again after a minute. When
      the unchanged kernel settles at a new level (clocks, power plan, a driver update), eval re-baselines to it by
      itself and judges the attempt against that.
   6. When a result surprises you, look at why (`rsi isa` on the new best, or the rejected `attempts/NNN.diff`) before
      the next idea. After three misses in a row, `status` names a different tag: switch kinds of idea.
3. Stop when eval prints `STOP` (8 attempts without a kept gain, 40 attempts, the last three gains under 1.5% in all,
   or a kernel found not to be deterministic; thresholds in `scripts/rsi.config.json`).
4. `rsi finish KERNEL --worker NAME` ends the kernel and releases it:
   - no gain: marked done;
   - otherwise it re-measures baseline against best on every capture (once more if only the timing missed) and lands
     it: on a fresh branch from an up-to-date `main` it applies the patch, rebuilds everything with
     `engine/build-win.sh`, checks the gain again against `main`'s own build, runs the four exact-output stages of the
     benchmark harness against the latest promoted round's hashes, commits, pushes, opens a PR and rebase-merges it.
     If `main` moves at any point before the merge, the PR is closed and the landing starts again on the new `main`,
     so nothing merges that the gate did not run on.
   - the patch no longer applies, does not build, or no longer gains on the current `main`: the loop restarts on the
     current code at the next claim, keeping its history;
   - the gate fails because of the change: `main` and `engine/build` are restored and the kernel is marked done;
   - anything outside the change (repository not on `main` or dirty, network, GitHub, a broken build of `main`, a
     gate that gives no hashes, `main` itself not matching the reference): the kernel waits ("waiting to land") and a
     later claim retries it. A landing interrupted at any step resumes from that step.
5. Back to step 1.

When a command fails, its exit code says what kind of failure it was: 1 a mistake in the command or the loop (read
the message), 3 the machine or a tool (fix it, then run the command again), 4 the GPU stayed in use (run it again
later), 5 waiting on the repository or another worker. A worker that cannot fix an exit 3 stops and reports it.

Every step that uses the GPU takes `<runsDir>/.gpu-lock`, waits while the lab harness holds `ref/.gpu-lock` or a
chat server, benchmark or other engine is running, and landing also takes `<runsDir>/.repo-lock`. So any number of
workers can run the cycle at once: they queue for the GPU instead of disturbing each other's timings, and only one
changes the repository at a time. Locks of crashed or killed workers are taken over.

## How it measures

- **Capture.** `IE_KCAP` (`engine/src/hip.c`) records one real launch of each kernel of a benchmark scenario: grid,
  arguments, and every byte of memory it reads. `engine/tools/kbench.c` replays one alone. Each loop also captures
  its kernel's launch from the middle of the step (another layer) as an extra held-out case, and keeps hard links to
  its captures so a re-discovery never changes them under a running loop.
- **Exactness.** An attempt must match the best version byte for byte on the captured inputs, including the memory
  just past each buffer (no stray writes), give the same bytes on a second run (no races), and leave the same state
  after two launches back to back (counters reset for the next launch). An edit to a shared file is also built into
  every other code object that reads it (the q8/q4 KV builds, the six megakernel builds, other units): no kernel may
  newly spill, and every changed kernel with a capture is checked for exactness and speed. `finish` checks the gain
  again on the current `main` and the exact output hashes and whole-model speed on the full engine.
- **Timing.** Cold (rotating copies of the captured memory, so each launch reads DRAM as inside the model), 15
  interleaved rounds; the noise floor is about ±0.5%. An attempt is kept only if it is at least 1% faster twice in a
  row and not over 1% slower on any held-out capture.
- **Scope.** Only kernel sources change. An edit to the working copy's `core/`, `prof/` or generated headers is rejected
  (`out_of_scope`) and reset.

Decisions in the ledger: `accepted`, `no_gain`, `slower`, `not_exact`, `not_confirmed` (the gain did not repeat),
`holdout_fail`, `collateral` (another kernel the edit changed got slower or inexact), `spill`, `compile_fail`,
`out_of_scope`, `suspect` (faster than 2x in one step, held back), `fault` (GPU fault or failed launch), `hang`,
`nondeterministic` (stops the loop; earlier gains still land), `error`; `noisy` and `busy` (not attempts); and the
markers `baseline`, `rebase` (restart on newer code), `rebaseline`, `verified`, `verify_fail`, `approved`,
`land_blocked`, `land_conflict`, `land_retry`, `land_fail`, `land_error`, `landed`, `done`, `reopen`.

A restart on newer code moves the old loop (its best sources, patch and attempt diffs) to `history/KERNEL/<time>/`;
`status` lists the wins from before the restart and where their diffs are, so they can be re-applied first.

Other commands: `rsi doctor` (the preflight on its own), `rsi queue` (ranking and every loop's state and worker),
`rsi reopen KERNEL` (a finished kernel's loop starts again on the current code at the next claim), `rsi release
KERNEL --worker NAME` (drop your claim), `rsi approve KERNEL` (with `requireApproval` set in the config, a verified
kernel lands only after this).

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

- **Coordinator**: first runs `rsi doctor vb16k_t6` and stops with its message if it fails. Starts up to 4 workers
  (the GPU lock makes more of them wait) and one verifier, each worker with a name no other worker uses, and gives
  each the brief below. Workers can share the workspace: every loop has its own working copy, and building, capturing
  and landing always go through the repository set in `scripts/rsi.config.json`. When every worker has stopped,
  runs `rsi queue vb16k_t6`: a kernel still `running` or `stopped` with no worker (its worker died; its claim
  expires after 2 hours of silence) gets a new worker. Then runs `node .agents/skills/kernel-rsi/scripts/report.mjs`
  and posts the summary from `summary.csv` with the PR links and any kernels waiting to land with the reason.
- **Worker `@worker-N`**: runs the cycle above with `--worker worker-N`, kernel after kernel, until claim prints
  `none` or `busy`. Posts one board line per attempt (`@coordinator KERNEL #N DECISION DELTA% best X us xSPEEDUP`) and
  one per finish (landed with the PR link, or why not). Edits only the working copy claim printed; never edits
  `engine/` in the repository. On exit code 3 (machine or tool) it posts the message and stops.
- **Verifier `@verifier`**: reviews each kernel a worker finishes: reads every accepted `attempts/NNN.diff` for
  shortcuts and input-specific behaviour (the rules above) and looks at any `suspect` attempt. A landed change that
  breaks the rules is reverted with a PR of its own (after `rsi reopen KERNEL`), and the board says why. A verifier on
  a different model than the workers catches more. With `requireApproval: true` in `scripts/rsi.config.json`, nothing
  lands until the verifier has read the kernel's diffs and run `rsi approve KERNEL`; claim lists the kernels waiting
  for it.

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
  `best.diff`, `attempts/NNN.diff`, `work/`, `core-src/`. When a kernel lands or is done, its builds, captures and
  working copy are deleted (several GB per kernel); the ledger, state, `best-src/`, `best.diff` and attempt diffs stay.
- `engine/build/.rsi-build-broken` and `.rsi-build-head`: whether `engine/build` matches the checked-out code (the
  next claim rebuilds it when not).
- `.gate-broken`: `main` itself did not match the reference hashes (landing waits; fix `main` or
  `docs/40round-ledger.json`, then delete it).

`node .agents/skills/kernel-rsi/scripts/report.mjs` merges every ledger into `all.jsonl` and `summary.csv` and writes
`report.html`: speedup and milliseconds saved per kernel, and per kernel every attempt coloured by decision over the
best-so-far line.

## Tests

Run these after changing `scripts/rsi.mjs`, `engine/tools/kbench.c` or the capture hook. The first two need no GPU and
touch nothing outside `%TEMP%\kernel-rsi-tests` (a throwaway clone of this repository, a local bare `origin`, and stubs
for the build, the benchmark harness and `gh`, all in `tests/`):

- `node .agents/skills/kernel-rsi/tests/run_tests.mjs`: every landing path, each forced by the stubs. Success; keeping
  another kernel's landing in the same file; a dirty repository, a user's branch, a person's `rsi/...` branch; a
  patch that no longer applies; build failures (patched only, and on `main` too); a gate mismatch with and without
  `main` matching, missing hashes, a one-off mismatch, a whole-model slowdown, the GPU fault marker, `main` known not
  to match; push, PR and merge failures and resuming after them; a PR that will not merge; `main` moving with engine
  and with docs-only changes; a merge whose status cannot be read; recovery from a crashed landing; a stale lock. About
  a minute.
- `node .agents/skills/kernel-rsi/tests/run_claim_tests.mjs`: six workers claiming at once, what is never handed out,
  re-claims, stale claims, `busy` / `wait` / `none`, and claim ownership. About 20 seconds.
- `sh .agents/skills/kernel-rsi/tests/e2e.sh`: the real flow on the GPU (preflight, discovery, loop setup, an eval that
  must come back `no_gain`, an out-of-scope edit, status, release), in its own runs directory so real loops are
  untouched. About 10 minutes the first time, a few after.
