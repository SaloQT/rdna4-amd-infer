"""Disk prompt cache check for Inkling-2 on ie-serve (run under ref/gpu-run.ps1, which holds the GPU lock).

usage: python tools/ink_dcache_check.py [--base TREE]     (TREE: another engine tree with build/ie-serve.exe)

Phase 1: this workspace's ie-serve with a fresh IE_DISK_CACHE dir; prompt A (src/json.c, json.h, gen.h in a user message) twice: a cold
         run, then a resume from the in-memory checkpoint. Waits for the idle flush ("checkpoints written"), stops.
Phase 2: the same server and dir restarted; prompt A once: must continue from the disk cache with first-token logits
         (IE_CACHE_DEBUG=1) identical to phase 1's memory resume.
With --base TREE:
Phase 3: this tree's ie-serve with no cache dir; prompt A twice.
Phase 4: TREE's ie-serve with no cache dir; prompt A twice: phase 3 must match its text.
Phase 5: TREE again, prompt A once: whether a cold prefill reproduces its first-token logits across processes at
         all (it decides whether phase 3 vs 4 logits are a gate).
Arguments as serve-inkling2.ps1 -Draft 0 passes them (no DSpark: the disk cache is off with a drafter), port 8097,
temperature 0, 8 output tokens, the -fast model file when it exists.
Exit 1 when a check fails. Server logs: results/inkling2/dcache-check/phaseN.err.log.
"""
import json, os, re, shutil, subprocess, sys, time, urllib.request

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LAB = os.path.abspath(os.path.join(WS, '..', '..'))
BASE = os.path.abspath(sys.argv[sys.argv.index('--base') + 1]) if '--base' in sys.argv else None
MODEL = os.path.join(LAB, 'models', 'Inkling-2-Mini-Uncensored-IQ3_XXS-fast.gguf')
if not os.path.isfile(MODEL):
    MODEL = os.path.join(LAB, 'models', 'Inkling-2-Mini-Uncensored-IQ3_XXS.gguf')
OUT = os.path.join(LAB, 'results', 'inkling2', 'dcache-check')
CACHE = os.path.join(OUT, 'dcache')
PORT = 8097


def serve_args(tree):
    return [os.path.join(tree, 'build', 'ie-serve.exe'), MODEL, '--backend', 'gpu',
            '--hsaco', os.path.join(tree, 'build', 'ie_kernels.hsaco'), '--port', str(PORT), '--ctx', '32768',
            '--name', 'inkling2-uncensored', '--chunk', '512', '--think-temp', '1.0', '--top-k', '20',
            '--dry-multiplier', '0.8', '--dry-base', '1.75', '--dry-allowed-length', '24', '--tool-temp', '0',
            '--effort', 'high']


def read_log(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read().splitlines()


def start(tree, phase, disk):
    env = {k: v for k, v in os.environ.items() if not k.startswith('IE_')}
    env['IE_CACHE_DEBUG'] = '1'
    if disk:
        env['IE_DISK_CACHE'] = CACHE
    log = os.path.join(OUT, 'phase%d.err.log' % phase)
    fe = open(log, 'w')
    p = subprocess.Popen(serve_args(tree), env=env, stdout=subprocess.DEVNULL, stderr=fe, cwd=tree)
    for _ in range(360):
        if p.poll() is not None:
            raise RuntimeError('phase %d: server exited %s' % (phase, p.returncode))
        try:
            urllib.request.urlopen('http://127.0.0.1:%d/health' % PORT, timeout=2).read()
            return p, fe, log
        except Exception:
            time.sleep(0.5)
    stop(p, fe)
    raise RuntimeError('phase %d: not ready after 180 s' % phase)


def stop(p, fe):
    if p.poll() is None:
        p.kill()
        p.wait(30)
    fe.close()
    time.sleep(2)


def ask(text):
    body = {'model': 'inkling2-uncensored', 'temperature': 0, 'max_tokens': 8, 'stream': False,
            'messages': [{'role': 'user', 'content': text}]}
    req = urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions' % PORT, json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    r = json.loads(urllib.request.urlopen(req, timeout=240).read())
    msg = r['choices'][0]['message']
    return {'text': (msg.get('reasoning_content') or '') + '|' + (msg.get('content') or ''),
            'prompt': r['usage']['prompt_tokens'],
            'cached': r['usage'].get('prompt_tokens_details', {}).get('cached_tokens'),
            'prompt_ms': r['usage'].get('prompt_ms')}


def logits(lines):
    return [l for l in lines if l.startswith('cache: first token logits')]


def vals(line):
    return line.split('):', 1)[1].strip()


def wait_flush(log):
    # the idle flush (IE_DISK_IDLE_S, default 10 s after the last request) and the disk thread's write after it, at most 90 s
    for _ in range(180):
        l = read_log(log)
        last = max([i for i, s in enumerate(l) if re.match(r'^\[\d+\] prompt', s)] or [-1])
        idle = max([i for i, s in enumerate(l) if s.startswith('disk cache: idle:')] or [-1])
        w = max([i for i, s in enumerate(l) if 'checkpoints written' in s] or [-1])
        if w > idle > last >= 0:
            return True
        time.sleep(0.5)
    return False


def show(log):
    for s in read_log(log):
        if re.match(r'^\[\d+\] prompt|^disk cache|^cache: first token', s):
            print('  ' + s)


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.path.isdir(CACHE):
        shutil.rmtree(CACHE)
    src = ''  # about 7K tokens: past the rings of the sliding window visits (2048 slots at --chunk 512), several blocks
    for name in ('json.c', 'json.h', 'gen.h'):
        src += open(os.path.join(WS, 'src', name), encoding='utf-8', errors='replace').read() + '\n'
    A = 'Summarize in one sentence what this C file does.\n\n' + src
    ok = True
    res = {}

    print('== phase 1: disk cache on, A cold then A from memory', flush=True)
    p, fe, log1 = start(WS, 1, True)
    try:
        res['p1a'] = ask(A)
        res['p1b'] = ask(A)
        flushed = wait_flush(log1)
    finally:
        stop(p, fe)
    print('  p1 cold  :', res['p1a'])
    print('  p1 memory:', res['p1b'])
    print('  idle flush seen:', flushed)
    show(log1)
    lg1 = logits(read_log(log1))
    files = os.listdir(CACHE) if os.path.isdir(CACHE) else []
    print('  cache files: %d, %.1f MB' % (len(files), sum(os.path.getsize(os.path.join(CACHE, f)) for f in files) / 1e6))
    for f in sorted(files):
        print('    %s %d' % (f, os.path.getsize(os.path.join(CACHE, f))))
    ok &= flushed and len(lg1) >= 2

    print('== phase 2: restart on the same dir, A', flush=True)
    p, fe, log2 = start(WS, 2, True)
    try:
        res['p2'] = ask(A)
    finally:
        stop(p, fe)
    print('  p2 disk  :', res['p2'])
    show(log2)
    l2 = read_log(log2)
    lg2 = logits(l2)
    cont = any('disk cache: continued from' in s for s in l2)
    same = len(lg1) >= 2 and len(lg2) >= 1 and vals(lg1[1]) == vals(lg2[0])
    print('  continued from disk: %s; first-token logits disk vs memory: %s' % (cont, 'identical' if same else 'DIFFERENT'))
    print('  prompt ms cold %s, memory %s, disk %s; cached %s of %s' % (res['p1a']['prompt_ms'], res['p1b']['prompt_ms'],
          res['p2']['prompt_ms'], res['p2']['cached'], res['p2']['prompt']))
    ok &= cont and same and res['p2']['cached'] is not None and res['p2']['cached'] >= res['p2']['prompt'] - 64
    ok &= res['p2']['text'] == res['p1b']['text']

    if not BASE:
        json.dump(res, open(os.path.join(OUT, 'check.json'), 'w'), indent=1)
        print('RESULT', 'PASS' if ok else 'FAIL')
        sys.exit(0 if ok else 1)

    print('== phase 3: this build, no cache dir', flush=True)
    p, fe, log3 = start(WS, 3, False)
    try:
        res['p3a'] = ask(A)
        res['p3b'] = ask(A)
    finally:
        stop(p, fe)
    show(log3)
    l3 = read_log(log3)
    l3_no_disk = not any(s.startswith('disk cache') for s in l3)
    print('  disk cache lines:', [s for s in l3 if s.startswith('disk cache')])

    print('== phase 4: ink5-base build, no cache dir', flush=True)
    p, fe, log4 = start(BASE, 4, False)
    try:
        res['p4a'] = ask(A)
        res['p4b'] = ask(A)
    finally:
        stop(p, fe)
    show(log4)
    lg3, lg4 = logits(l3), logits(read_log(log4))
    same34 = len(lg3) >= 2 and len(lg4) >= 2 and [vals(x) for x in lg3[:2]] == [vals(x) for x in lg4[:2]]
    text34 = res['p3a']['text'] == res['p4a']['text'] and res['p3b']['text'] == res['p4b']['text']
    print('  no-cache-dir build vs base: logits %s, texts %s' % ('identical' if same34 else 'DIFFERENT',
          'identical' if text34 else 'DIFFERENT'))
    cold13 = len(lg1) >= 1 and len(lg3) >= 1 and vals(lg1[0]) == vals(lg3[0])
    print('  cold logits phase 1 (disk on) vs phase 3 (off): %s' % ('identical' if cold13 else 'DIFFERENT'))

    print('== phase 5: ink5-base build again, no cache dir (does the base reproduce its own cold logits?)', flush=True)
    p, fe, log5 = start(BASE, 5, False)
    try:
        res['p5a'] = ask(A)
    finally:
        stop(p, fe)
    show(log5)
    lg5 = logits(read_log(log5))
    same45 = len(lg4) >= 1 and len(lg5) >= 1 and vals(lg4[0]) == vals(lg5[0])
    print('  base cold logits, phase 4 vs phase 5: %s' % ('identical' if same45 else 'DIFFERENT'))
    # Cold prefill logits are compared between processes only when the base itself reproduces them.
    ok &= text34 and l3_no_disk and (same34 or not same45)
    for k, v in res.items():
        print('  %s %s' % (k, v))
    json.dump(res, open(os.path.join(OUT, 'check.json'), 'w'), indent=1)
    print('RESULT', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
