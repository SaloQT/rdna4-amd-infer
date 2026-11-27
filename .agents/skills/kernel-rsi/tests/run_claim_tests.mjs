// Claim behaviour without the GPU: existing loops, a queue for the current build, workers claiming in parallel.
//   node .agents/skills/kernel-rsi/tests/run_claim_tests.mjs     (no GPU; works in %TEMP%/kernel-rsi-tests or $RSI_TEST_DIR)
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url)); /* the stubs live next to this file */
const T = process.env.RSI_TEST_DIR || path.join(os.tmpdir(), 'kernel-rsi-tests'); /* throwaway clones, origin, runs, modes */
fs.mkdirSync(T, { recursive: true });
process.env.RSI_TEST_DIR = T; /* the stubs, run by rsi.mjs, find their state here */
const REAL = path.resolve(HERE, '../../../..'); /* the repository the tests clone */
const RSI = path.join(HERE, '..', 'scripts', 'rsi.mjs');
const repo = path.join(T, 'repo'), origin = path.join(T, 'origin.git'), runs = path.join(T, 'runs');
const env = { ...process.env, RSI_CONFIG: path.join(T, 'config.json') };
const SC = 'vb16k_t6';
const K = ['ie_inkling_rel_attn_ver_r4', 'ie_inkling_gexp_iq4nl_wave_ver_r13', 'ie_inkling_qkv_sconv_kv_r11', 'ie_inkling_rel_clip_sink_gate_t8_r9',
  'ie_inkling_gemv_q8_wave_ht_r13', 'ie_inkling_gemv_q4k_head_ver_r7'];
const NONDET = 'ie_inkling_router_route_sort_ver_r7', TINY = 'ie_inkling_rmsnorm_mt_r1';

function setup() {
  for (const p of [origin, repo, runs]) fs.rmSync(p, { recursive: true, force: true });
  spawnSync('git', ['clone', '-q', '--bare', '--branch', 'main', REAL, origin]);
  spawnSync('git', ['clone', '-q', origin, repo]);
  const b = path.join(repo, 'engine', 'build');
  fs.mkdirSync(b, { recursive: true });
  for (const f of fs.readdirSync(path.join(REAL, 'engine', 'build')))
    if (/^(ie-run\.exe|kbench\.exe|ie_kernels\.hsaco|ie_inkling_kernels\.hsaco)$|\.h$/.test(f)) {
      fs.copyFileSync(path.join(REAL, 'engine', 'build', f), path.join(b, f));
      const t = new Date(Date.now() + 5000); fs.utimesSync(path.join(b, f), t, t);
    }
  const h = crypto.createHash('sha1');
  for (const f of ['ie-run.exe', 'ie_kernels.hsaco', 'ie_inkling_kernels.hsaco']) h.update(fs.readFileSync(path.join(b, f)));
  const rows = [...K, NONDET, TINY].map((k, i) => ({ kernel: k, unit: 'inkling', perStep: 48, us: 100 - i * 10, msPerStep: (100 - i * 10) * 48 / 1000,
    replay: k !== NONDET, nondet: k === NONDET, share: k === TINY ? 0.1 : 10 }));
  fs.mkdirSync(runs, { recursive: true });
  fs.writeFileSync(path.join(runs, `queue-${SC}.json`), JSON.stringify({ scenario: SC, about: 'test', buildHash: h.digest('hex').slice(0, 16), totalMsPerStep: 10, kernels: rows }));
  const res = { vgpr: 68, waves: 16 };
  for (const k of K) {
    const d = path.join(runs, 'kernels', k);
    fs.mkdirSync(path.join(d, 'base-src'), { recursive: true }); /* a current loop: claim must not set it up again (GPU) */
    const st = { kernel: k, unit: 'inkling', scenario: SC, holdouts: [], baseline: { us: 50, noisePct: 0.2, res, attempt: 0 }, best: { attempt: 0, us: 50, res },
      speedup: 1, attempts: 0, lastAccept: 0, accepts: [] };
    if (k === K[4]) st.done = { reason: 'no gain' };
    if (k === K[5]) { st.blockedUntil = new Date(Date.now() + 3600000).toISOString(); st.blockedWhy = 'test'; st.verified = { ok: true, attempt: 0 }; }
    fs.writeFileSync(path.join(d, 'state.json'), JSON.stringify(st));
  }
  /* a complete test config: stubs for the harness, build and GitHub; no real GPU locks, processes or fault marker */
  const cfg = { harness: path.join(HERE, 'stub_harness.py'), buildScript: path.join(HERE, 'stub_build.sh'), landGainCheck: false,
    labGpuLock: path.join(T, 'no-lock'), labLocks: [], labFaultMarker: path.join(T, 'gpu-fault'), gpuBusyProcesses: [], minFreeGB: 1, blockedRetryMinutes: 0 };
  fs.writeFileSync(path.join(T, 'config.json'), JSON.stringify({ ...cfg, repo, runsDir: runs, ghCmd: [process.execPath, path.join(HERE, 'stub_gh.mjs')], claimWaitMinutes: 0 }));
}
const claimSync = (w) => { const r = spawnSync(process.execPath, [RSI, 'claim', SC, '--worker', w], { encoding: 'utf8', env }); return (r.stdout || '') + (r.stderr || ''); };
const claimed = (out) => (out.match(/claimed (\S+) for/) || [])[1] || null;
const results = [];
const check = (name, ok, out) => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '\n' + out}`); };

setup();
// six workers at once: the four claimable loops go to four different workers, two get "busy"
const outs = await Promise.all(['w1', 'w2', 'w3', 'w4', 'w5', 'w6'].map((w) => new Promise((res) => {
  const p = spawn(process.execPath, [RSI, 'claim', SC, '--worker', w], { env });
  let o = '';
  p.stdout.on('data', (d) => (o += d)); p.stderr.on('data', (d) => (o += d));
  p.on('close', () => res(o));
})));
const got = outs.map(claimed).filter(Boolean);
check('parallel claims: each claimable kernel to exactly one worker', got.length === 4 && new Set(got).size === 4 && got.every((k) => K.slice(0, 4).includes(k)), outs.join('\n----\n'));
check('parallel claims: the others are told "busy"', outs.filter((o) => /^busy:/m.test(o)).length === 2, outs.join('\n----\n'));
check('done, waiting-to-land, nondeterministic and tiny kernels are never handed out', !got.some((k) => [K[4], K[5], NONDET, TINY].includes(k)), got.join(', '));
const wi = outs.findIndex((o) => claimed(o)); /* the order of parallel claims varies: take a worker that got one */
const wname = 'w' + (wi + 1), w1k = claimed(outs[wi]);
const again = claimSync(wname);
check('a worker claiming again gets its own kernel back', claimed(again) === w1k, again);
// a claim whose worker has been silent past claimStaleHours is taken over
const cp = path.join(runs, 'kernels', w1k, 'claim.json');
const old = new Date(Date.now() - 3 * 3600000).toISOString();
fs.writeFileSync(cp, JSON.stringify({ worker: wname, since: old, beat: old }));
const take = claimSync('w9');
check('a stale claim is taken over by another worker', claimed(take) === w1k, take);
// all claimed by live workers -> busy; all finished -> none
for (const k of K.slice(0, 4)) {
  const sp = path.join(runs, 'kernels', k, 'state.json'), st = JSON.parse(fs.readFileSync(sp, 'utf8'));
  st.done = { reason: 'no gain' }; fs.writeFileSync(sp, JSON.stringify(st));
}
const wait = claimSync('w10');
check('only a kernel waiting to retry is left (no other worker): "wait"', /^wait:/m.test(wait), wait);
const sp5 = path.join(runs, 'kernels', K[5], 'state.json'), st5 = JSON.parse(fs.readFileSync(sp5, 'utf8'));
st5.blockCount = 8; fs.writeFileSync(sp5, JSON.stringify(st5));
const none = claimSync('w11');
check('a kernel blocked 8 times needs a person: "none", naming it', /^none:/m.test(none) && none.includes(K[5]) && /needs a person/.test(none), none);
// eval/finish refuse a kernel claimed by someone else
setup();
const a = claimSync('alice'), ka = claimed(a);
const r = spawnSync(process.execPath, [RSI, 'finish', ka, '--worker', 'bob'], { encoding: 'utf8', env });
check("finish refuses another worker's claim", /claimed by alice/.test(r.stdout + r.stderr), r.stdout + r.stderr);
const f2 = spawnSync(process.execPath, [RSI, 'claim', SC], { encoding: 'utf8', env });
check('claim without --worker is refused', /needs --worker/.test(f2.stdout + f2.stderr), f2.stdout + f2.stderr);
// landing off: a stopped loop with a verified best is finished for the swarm and listed as ready to land
setup();
for (const k of K.slice(0, 4)) {
  const sp = path.join(runs, 'kernels', k, 'state.json'), st = JSON.parse(fs.readFileSync(sp, 'utf8'));
  st.attempts = 9; st.lastAccept = 1; st.best = { ...st.best, attempt: 1 }; st.speedup = 1.05; st.verified = { ok: true, attempt: 1 };
  fs.writeFileSync(sp, JSON.stringify(st));
}
{ /* the fixture's waiting kernel would make claim answer "wait": finish it for this case */
  const sp = path.join(runs, 'kernels', K[5], 'state.json'), st = JSON.parse(fs.readFileSync(sp, 'utf8'));
  st.done = { reason: 'test' }; fs.writeFileSync(sp, JSON.stringify(st));
}
const cfg2 = JSON.parse(fs.readFileSync(path.join(T, 'config.json'), 'utf8'));
fs.writeFileSync(path.join(T, 'config.json'), JSON.stringify({ ...cfg2, autoLand: false }));
const ready = claimSync('w20');
check('landing off: verified, stopped kernels are not handed out and are listed as ready to land', /^none:/m.test(ready) && /ready to land/.test(ready) && K.slice(0, 4).every((k) => ready.includes(k)), ready);

console.log(`\n${results.filter(Boolean).length}/${results.length} passed`);
process.exit(results.every(Boolean) ? 0 : 1);
