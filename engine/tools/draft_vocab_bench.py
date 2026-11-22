"""Interleaved speculative-decoding comparison of ie-run settings (env vars), every run through ref\\gpu-run.ps1.

  python tools/draft_vocab_bench.py [--prompts code,prose,list,rep] [--reps 3] [--n 64] [--draft 3]
      [--timeout 300] [--config LABEL:KEY=VAL,KEY=VAL ...] [--out build/spec/bench.jsonl]

Per prompt one run without drafts gives the reference ids; then, per rep, the prompts in order and for each prompt the
configs (the order of the configs alternates by rep). A run that does not end with exit code 0 stops everything (a
fault leaves ref\\.gpu-fault; nothing is re-run). Prints ms/token, acceptance, draft and verify ms/step per run, a
median table per (prompt, config) and whether every run's ids equal the draft-0 ids. One run at a time.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
lab = os.path.dirname(os.path.dirname(root))

ap = argparse.ArgumentParser()
ap.add_argument('--prompts', default='code,prose,list,rep')
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--n', type=int, default=64)
ap.add_argument('--draft', type=int, default=3)
ap.add_argument('--timeout', type=int, default=300)
ap.add_argument('--config', action='append', default=[], help='LABEL:KEY=VAL,KEY=VAL (default: base: and 64k:IE_DRAFT_VOCAB=64k)')
ap.add_argument('--model', default=os.path.join(lab, 'models', 'Underdog-Saluki-27B-1.0-IQ2-mix.gguf'))
ap.add_argument('--mtp', default=os.path.join(lab, 'models', 'MTP', 'mtp-Saluki-aligned.gguf'))
ap.add_argument('--out', default=os.path.join(root, 'build', 'spec', 'bench.jsonl'))
ap.add_argument('--no-ref', action='store_true', help='skip the draft-0 runs (no id check against plain decoding)')
a = ap.parse_args()

configs = []
for c in a.config or ['base:', '64k:IE_DRAFT_VOCAB=64k']:
    label, _, kv = c.partition(':')
    configs.append((label, dict(x.split('=', 1) for x in kv.split(',') if x)))

gpu_run = os.path.join(os.path.dirname(root), 'gpu-run.ps1')
pat = {
    'ms': re.compile(r'gpu: \d+ tokens in [\d.]+ ms, ([\d.]+) ms/token, [\d.]+ tokens/s \(speculative'),
    'acc': re.compile(r'mtp: (\d+) steps, (\d+) drafts accepted of (\d+) \(([\d.]+)%\), ([\d.]+) tokens per step'),
    'st': re.compile(r'spec time: drafts ([\d.]+) ms/step, verify ([\d.]+) ms/step'),
    'lk': re.compile(r'lookup: (\d+) of (\d+) steps drafted from the text'),
    'head': re.compile(r'mtp: draft head Q4_0 \((?:(\d+) rows, )?([\d.]+) MB\)'),
    'arena': re.compile(r'arena (\d+) bytes'),
    'plain': re.compile(r'gpu: \d+ steps, ([\d.]+) ms/token'),
    # the engine fell back from 16 to 8 linear attention states (verify runs of up to 8): MB free at that moment
    'lin': re.compile(r'linear state (\d+) bytes'),
    'cap8': re.compile(r'gpu: ([\d.]+) MB free: 8 linear attention states'),
}


def run(prompt, draft, env_extra):
    env = dict(os.environ)
    for k in [k for k in env if k.startswith('IE_')]:
        del env[k]  # only the config's settings, not the caller's
    env.update(env_extra)
    env['IE_SPEC_TIME'] = '1'
    cmd = ['pwsh', '-NoProfile', '-File', gpu_run, '-TimeoutSec', str(a.timeout), '--',
           os.path.join('build', 'ie-run.exe'), a.model, '--backend', 'gpu', '--hsaco', os.path.join('build', 'ie_kernels.hsaco'),
           '--tokens-file', os.path.join('build', 'spec', 't_%s.txt' % prompt), '--n', str(a.n), '--mtp', a.mtp, '--draft', str(draft)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, cwd=root)
    if p.returncode:
        sys.exit('STOP: exit %d for %s draft %d %s\n%s' % (p.returncode, prompt, draft, env_extra, p.stderr[-1500:]))
    m = re.search(r'^ids:(.*)$', p.stdout, re.M)
    if not m:
        sys.exit('STOP: no ids line:\n' + p.stderr[-1500:])
    r = {'prompt': prompt, 'draft': draft, 'env': env_extra, 'ids': m.group(1).split()}
    for key in ('ms', 'acc', 'st', 'lk', 'head', 'arena', 'plain', 'cap8', 'lin'):
        mm = pat[key].search(p.stderr)
        r[key] = [float(x) if x is not None else None for x in mm.groups()] if mm else None
    return r


prompts = a.prompts.split(',')
os.makedirs(os.path.dirname(a.out), exist_ok=True)
out = open(a.out, 'a')
ref = {}
if not a.no_ref:
    for pr in prompts:
        r = run(pr, 0, {})
        ref[pr] = r['ids']
        print('%-6s draft 0   %7.3f ms/token  (%d ids)' % (pr, r['plain'][0] if r['plain'] else -1, len(r['ids'])), flush=True)
        out.write(json.dumps(dict(r, label='draft0')) + '\n')
        out.flush()
runs = []
for rep in range(a.reps):
    order = configs if rep % 2 == 0 else configs[::-1]
    for pr in prompts:
        for label, env in order:
            r = run(pr, a.draft, env)
            r['label'], r['rep'] = label, rep
            r['same'] = None if pr not in ref else r['ids'] == ref[pr]
            runs.append(r)
            out.write(json.dumps(r) + '\n')
            out.flush()
            ms, acc, st = r['ms'][0], r['acc'], r['st']
            print('rep %d %-6s %-8s %7.3f ms/token  acc %5.1f%% (%d/%d) %.2f tok/step  draft %.2f verify %.2f ms/step  ids %s%s' % (
                rep, pr, label, ms, acc[3], acc[1], acc[2], acc[4], st[0] if st else -1, st[1] if st else -1,
                {None: '-', True: 'equal', False: 'DIFFERENT'}[r['same']],
                '  [verify capped at 8: only %.0f MB free]' % r['cap8'][0] if r['cap8'] else ''), flush=True)

print('\nmedians over %d reps (spec ms/token; acceptance %% of drafts; draft ms/step; verify ms/step)' % a.reps)
print('%-6s %-8s %9s %8s %9s %10s %8s %6s' % ('prompt', 'config', 'ms/token', 'accept%', 'tok/step', 'draft ms', 'verify', 'ids'))
for pr in prompts:
    for label, _ in configs:
        rs = [r for r in runs if r['prompt'] == pr and r['label'] == label]
        med = lambda f: statistics.median(f(r) for r in rs)  # noqa: E731
        same = 'n/a' if pr not in ref else ('equal' if all(r['same'] for r in rs) else 'DIFFERENT')
        print('%-6s %-8s %9.3f %8.1f %9.2f %10.2f %8.2f %6s' % (
            pr, label, med(lambda r: r['ms'][0]), med(lambda r: r['acc'][3]), med(lambda r: r['acc'][4]),
            med(lambda r: r['st'][0] if r['st'] else float('nan')), med(lambda r: r['st'][1] if r['st'] else float('nan')), same))
