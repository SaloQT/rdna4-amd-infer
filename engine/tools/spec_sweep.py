"""MTP draft sweep for ie-run: per prompt, greedy decode without drafts, then
with each draft setting; ms/token, acceptance, and whether the generated ids
equal the no-draft ids. Runs one at a time.

  python tools/spec_sweep.py [--n 128] [--drafts 1,2,3] [--pmin 0.6,0.4]
         [--mtp FILE] [--extra "--temp 0.7 --seed 1"] TOKENS_FILE...
"""
import argparse
import os
import re
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
lab = os.path.dirname(os.path.dirname(root))

ap = argparse.ArgumentParser()
ap.add_argument('files', nargs='+')
ap.add_argument('--n', type=int, default=128)
ap.add_argument('--drafts', default='1,2,3,4,5,6,7')
ap.add_argument('--pmin', default='0.6')
ap.add_argument('--model', default=os.path.join(lab, 'models', 'Underdog-Saluki-27B-1.0-IQ2-mix.gguf'))
ap.add_argument('--mtp', default=os.path.join(lab, 'models', 'MTP', 'mtp-Saluki-aligned.gguf'))
ap.add_argument('--exe', default=os.path.join(root, 'build', 'ie-run.exe'))
ap.add_argument('--hsaco', default=os.path.join(root, 'build', 'ie_kernels.hsaco'))
ap.add_argument('--extra', default='')
ap.add_argument('--no-base', action='store_true', help='skip the no-draft run (no id check)')
a = ap.parse_args()


def run(tfile, draft, pmin):
    cmd = [a.exe, a.model, '--backend', 'gpu', '--hsaco', a.hsaco, '--tokens-file', tfile, '--n', str(a.n)]
    cmd += a.extra.split()
    if draft:
        cmd += ['--mtp', a.mtp, '--draft', str(draft), '--draft-pmin', str(pmin)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
    if p.returncode:
        sys.exit('ie-run failed (%d): %s' % (p.returncode, p.stderr[-800:]))
    ids = re.search(r'^ids:(.*)$', p.stdout, re.M).group(1).split()
    if draft:
        ms = float(re.search(r'gpu: \d+ tokens in [\d.]+ ms, ([\d.]+) ms/token, [\d.]+ tokens/s \(speculative', p.stderr).group(1))
        m = re.search(r'mtp: \d+ steps, (\d+) drafts accepted of (\d+) \(([\d.]+)%\), ([\d.]+) tokens per step', p.stderr)
        acc = '%s%% %s/step' % (m.group(3), m.group(4))
    else:
        ms = float(re.search(r'gpu: \d+ steps, ([\d.]+) ms/token', p.stderr).group(1))
        acc = '-'
    return ids, ms, acc


for f in a.files:
    name = os.path.basename(f)
    base_ids, base_ms = None, None
    if not a.no_base:
        base_ids, base_ms, _ = run(f, 0, 0)
        print('%-12s draft 0          %7.3f ms/token  %5.1f tok/s' % (name, base_ms, 1000 / base_ms), flush=True)
    for pm in a.pmin.split(','):
        for d in a.drafts.split(','):
            ids, ms, acc = run(f, int(d), pm)
            same = '-' if base_ids is None else ('ids equal' if ids == base_ids else
                   'ids DIFFER at %d' % next((i for i, (x, y) in enumerate(zip(ids, base_ids)) if x != y), min(len(ids), len(base_ids))))
            gain = '' if base_ms is None else '%+6.1f%%' % ((base_ms / ms - 1) * 100)
            print('%-12s draft %s pmin %-4s %7.3f ms/token  %5.1f tok/s %s  %-20s %s' % (name, d, pm, ms, 1000 / ms, gain, acc, same), flush=True)
