"""ie-serve greedy check for the draft vocabulary: per setting one server (started through ref\\gpu-run.ps1), two small
identical requests (build/spec/chat_small.json, max_tokens 64, temperature 0; the second one runs on the prompt cache),
then the server is stopped (by the pid of the ie-serve.exe that our gpu-run started, so gpu-run exits and frees the lock).
The completions of all settings must be identical.

  python tools/serve_vocab_check.py [--port 8137] [--timeout 420] [--setting LABEL:DRAFT:KEY=VAL,KEY=VAL ...]
  default settings: draft0 (--draft 0), base (--draft 3), 64k (--draft 3 with IE_DRAFT_VOCAB=64k)
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
lab = os.path.dirname(os.path.dirname(root))

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=8137)
ap.add_argument('--timeout', type=int, default=420, help='gpu-run timeout; the server is stopped long before')
ap.add_argument('--wait', type=int, default=240, help='seconds to wait for the model to load')
ap.add_argument('--setting', action='append', default=[])
ap.add_argument('--out', default=os.path.join(root, 'build', 'spec', 'serve_check.json'))
a = ap.parse_args()

settings = []
for s in a.setting or ['draft0:0:', 'base:3:', '64k:3:IE_DRAFT_VOCAB=64k']:
    label, draft, kv = s.split(':', 2)
    settings.append((label, int(draft), dict(x.split('=', 1) for x in kv.split(',') if x)))

model = os.path.join(lab, 'models', 'Underdog-Saluki-27B-1.0-IQ2-mix.gguf')
mtp = os.path.join(lab, 'models', 'MTP', 'mtp-Saluki-aligned.gguf')
gpu_run = os.path.join(os.path.dirname(root), 'gpu-run.ps1')
body = open(os.path.join(root, 'build', 'spec', 'chat_small.json'), 'rb').read()
base = 'http://127.0.0.1:%d' % a.port


def server_pids(parent):
    """The ie-serve.exe started by our gpu-run (pid parent): its direct child, never another server with the same image name."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='ie-serve.exe' AND ParentProcessId=%d\" | ForEach-Object { $_.ProcessId }" % parent)
    p = subprocess.run(['pwsh', '-NoProfile', '-Command', ps], capture_output=True, text=True)
    return [int(x) for x in p.stdout.split()]


def get(path, timeout=5):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.read().decode('utf-8')


def ask():
    req = urllib.request.Request(base + '/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode('utf-8'))


results = {}
for label, draft, env_extra in settings:
    env = {k: v for k, v in os.environ.items() if not k.startswith('IE_')}
    env.update(env_extra)
    cmd = ['pwsh', '-NoProfile', '-File', gpu_run, '-TimeoutSec', str(a.timeout), '--', os.path.join('build', 'ie-serve.exe'), model,
           '--backend', 'gpu', '--hsaco', os.path.join('build', 'ie_kernels.hsaco'), '--mtp', mtp, '--port', str(a.port), '--ctx', '4096',
           '--draft', str(draft), '--name', 'saluki']
    logp = os.path.join(root, 'build', 'spec', 'serve_%s.log' % label)
    log = open(logp, 'wb')
    print('== %s: draft %d %s' % (label, draft, env_extra), flush=True)
    proc = subprocess.Popen(cmd, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    ok = False
    try:
        while time.time() - t0 < a.wait and proc.poll() is None:
            try:
                if '"ok"' in get('/health', 2):
                    ok = True
                    break
            except Exception:
                time.sleep(2)
        if not ok:
            print('   server not ready (gpu-run exit %s); see %s' % (proc.poll(), logp))
        else:
            print('   ready after %.0f s' % (time.time() - t0), flush=True)
            answers = []
            for i in range(2):
                r = ask()
                answers.append(r)
                c = r['choices'][0]['message']
                print('   request %d: %d tokens, finish %s, content %r' % (i + 1, r['usage'].get('completion_tokens', -1), r['choices'][0].get('finish_reason'),
                                                                  (c.get('content') or '')[:60]), flush=True)
            results[label] = answers
    finally:
        # stop only the ie-serve that our gpu-run started (by pid); gpu-run then returns by itself and frees the GPU lock
        for pid in server_pids(proc.pid):
            subprocess.run(['pwsh', '-NoProfile', '-Command', 'Stop-Process -Id %d -Force' % pid], capture_output=True)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print('   gpu-run did not return 60 s after the server was stopped; NOT killing it (it holds the lock): look at pid %d' % proc.pid)
            sys.exit(3)
        log.close()
        print('   stopped (gpu-run exit %s)' % proc.returncode, flush=True)

json.dump(results, open(a.out, 'w'), indent=1)
ref = None
for label, _, _ in settings:
    if label not in results:
        print('%-8s NO RESULT' % label)
        continue
    texts = [r['choices'][0]['message'].get('content') for r in results[label]]
    ref = ref if ref is not None else texts[0]
    print('%-8s completions %s' % (label, 'identical to the first setting' if all(t == ref for t in texts) else 'DIFFERENT'))
