---
name: kernel-rsi
description: Optimise this engine's gfx1201 GPU kernels automatically, one at a time, in a measured improve-until-it-stops loop that lands each result on main. Each edit is built, checked bit-identical against the best version so far on a captured real launch, timed cold on the GPU, kept or reverted, and logged to a JSONL ledger that report.mjs turns into charts; a finished kernel is verified, rebuilt, gated on the exact output hashes and merged. Use when asked to optimise, tune or speed up kernels, run the kernel RSI loop, find the slowest kernels, or split kernel work across subagents.
---

The skill is shared with Antigravity and lives at `.agents/skills/kernel-rsi/SKILL.md` in this repository, with its
scripts in `.agents/skills/kernel-rsi/scripts/`. Read that file and follow it.
