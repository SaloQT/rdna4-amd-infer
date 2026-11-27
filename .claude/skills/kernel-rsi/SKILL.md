---
name: kernel-rsi
description: Optimise this engine's gfx1201 GPU kernels automatically, one at a time, in a measured improve-until-it-stops loop. Each edit is built, checked bit-identical against the best version so far on a captured real launch replayed alone on the GPU, timed cold, kept or reverted, and logged to a JSONL ledger that report.mjs turns into charts; a finished kernel ends as a verified patch, landed on main through the full benchmark gate only on request (rsi land) or with autoLand on. Use when asked to optimise, tune or speed up kernels, run the kernel RSI loop, find the slowest kernels, or split kernel work across subagents.
---

The skill is shared with Antigravity and lives at `.agents/skills/kernel-rsi/SKILL.md` in this repository, with its
scripts in `.agents/skills/kernel-rsi/scripts/`. Read that file and follow it.
