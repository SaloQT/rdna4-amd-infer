// Drives rsi.mjs land through every landing path against a throwaway clone, a local bare origin and stubs.
//   node .agents/skills/kernel-rsi/tests/run_tests.mjs     (no GPU; works in %TEMP%/kernel-rsi-tests or $RSI_TEST_DIR)
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url)); /* the stubs live next to this file */
const T = process.env.RSI_TEST_DIR || path.join(os.tmpdir(), 'kernel-rsi-tests'); /* throwaway clones, origin, runs, modes */
fs.mkdirSync(T, { recursive: true });
process.env.RSI_TEST_DIR = T; /* the stubs, run by rsi.mjs, find their state here */
const REAL = path.resolve(HERE, '../../../..'); /* the repository the tests clone */
const RSI = path.join(HERE, '..', 'scripts', 'rsi.mjs');
const K = 'ie_inkling_rel_attn_ver_r4';
const INC = 'engine/kernels/ie_inkling_r4_attn.inc';
const repo = path.join(T, 'repo'), origin = path.join(T, 'origin.git'), runs = path.join(T, 'runs');
const sh = (cmd, args, cwd = T) => spawnSync(cmd, args, { cwd, encoding: 'utf8' });
const git = (...a) => sh('git', a, repo);
const ogit = (...a) => sh('git', ['-C', origin, ...a]);
const kd = path.join(runs, 'kernels', K);
const state = () => JSON.parse(fs.readFileSync(path.join(kd, 'state.json'), 'utf8'));
const mode = (f, m) => fs.writeFileSync(path.join(T, f), m);

function setup() {
  for (const p of [origin, repo, runs]) fs.rmSync(p, { recursive: true, force: true });
  for (const f of ['gate.mode', 'build.mode', 'gh.mode', 'gh.json', 'gate.calls', 'build.calls', 'gh.calls']) fs.rmSync(path.join(T, f), { force: true });
  sh('git', ['clone', '-q', '--bare', '--branch', 'main', REAL, origin]);
  sh('git', ['clone', '-q', origin, repo]);
  const b = path.join(repo, 'engine', 'build');
  fs.mkdirSync(b, { recursive: true });
  for (const f of fs.readdirSync(path.join(REAL, 'engine', 'build')))
    if (/^(ie-run\.exe|ie_kernels\.hsaco|ie_inkling_kernels\.hsaco)$|\.h$/.test(f)) {
      fs.copyFileSync(path.join(REAL, 'engine', 'build', f), path.join(b, f));
      const t = new Date(Date.now() + 5000); /* built after the checkout, as a real build would be */
      fs.utimesSync(path.join(b, f), t, t);
    }
  // the verified loop: best.diff adds a marker comment
  const inc = path.join(repo, INC);
  const orig = fs.readFileSync(inc, 'utf8');
  fs.writeFileSync(inc, '/* RSI-TEST-MARKER */\n' + orig);
  const diff = git('diff', '--', INC).stdout;
  git('checkout', '-q', '--', INC);
  fs.mkdirSync(kd, { recursive: true });
  fs.writeFileSync(path.join(kd, 'best.diff'), diff);
  const res = { vgpr: 68, sgpr: 52, lds: 12484, priv: 0, vspill: 0, sspill: 0, waves: 16 };
  fs.writeFileSync(path.join(kd, 'state.json'), JSON.stringify({ kernel: K, unit: 'inkling', scenario: 'vb16k_t6', holdouts: [], perStep: 40,
    baseline: { us: 46, min: 45, noisePct: 0.2, res, attempt: 0 }, best: { attempt: 2, us: 43, res }, speedup: 46 / 43, attempts: 2, lastAccept: 2,
    accepts: [{ attempt: 2, gainPct: -7 }], verified: { ok: true, attempt: 2, files: ['ie_inkling_r4_attn.inc'],
      results: [{ scenario: 'vb16k_t6', baseline_us: 46, best_us: 43, delta_pct: -7, ok: true }] } }));
  fs.writeFileSync(path.join(kd, 'ledger.jsonl'), JSON.stringify({ ts: new Date().toISOString(), kernel: K, attempt: 2, idea: 'test idea', tag: 'alu', decision: 'accepted', delta_pct: -7 }) + '\n');
  fs.writeFileSync(path.join(T, 'config.json'), JSON.stringify({ repo, runsDir: runs, harness: path.join(HERE, 'stub_harness.py'),
    ghCmd: [process.execPath, path.join(HERE, 'stub_gh.mjs')], buildScript: path.join(HERE, 'stub_build.sh'), landGainCheck: false,
    labGpuLock: path.join(T, 'no-lock'), labLocks: [], labFaultMarker: path.join(T, 'gpu-fault'), gpuBusyProcesses: [], minFreeGB: 1, blockedRetryMinutes: 0 }));
  fs.rmSync(path.join(T, 'gpu-fault'), { force: true });
}
function land() {
  const r = spawnSync(process.execPath, [RSI, 'land', K], { encoding: 'utf8', env: { ...process.env, RSI_CONFIG: path.join(T, 'config.json') } });
  return (r.stdout || '') + (r.stderr || '');
}
const originMarker = () => ogit('show', `main:${INC}`).stdout.includes('RSI-TEST-MARKER');
const onMainClean = () => git('rev-parse', '--abbrev-ref', 'HEAD').stdout.trim() === 'main' && !git('status', '--porcelain', '--untracked-files=no').stdout.trim();
const localMarker = () => fs.readFileSync(path.join(repo, INC), 'utf8').includes('RSI-TEST-MARKER');
const remoteRsi = () => ogit('branch', '--list', 'rsi/*').stdout.trim();
const broken = () => fs.existsSync(path.join(repo, 'engine', 'build', '.rsi-build-broken'));

const results = [];
function test(name, fn) {
  setup();
  let ok = false, why = '';
  try { const r = fn(); ok = r === true; if (!ok) why = String(r); } catch (e) { why = e.stack; }
  results.push({ name, ok, why });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '\n      ' + why.split('\n').join('\n      ')}`);
}
const expect = (cond, msg, out) => (cond ? true : `${msg}\n--- output ---\n${out}`);

test('success: lands, origin has the change, repo on main, build current', () => {
  const out = land(), st = state();
  return expect(st.landed && originMarker() && onMainClean() && localMarker() && !broken() && !remoteRsi(), 'not landed cleanly', out);
});
test('another kernel landed in the same file after this loop started: the landing keeps both changes', () => {
  // the loop's snapshots: base-src = main when the loop started, best-src = base + this loop's change
  const kdirs = path.join(repo, 'engine', 'kernels');
  fs.cpSync(kdirs, path.join(kd, 'base-src'), { recursive: true });
  fs.cpSync(kdirs, path.join(kd, 'best-src'), { recursive: true });
  const bf = path.join(kd, 'best-src', 'ie_inkling_r4_attn.inc');
  fs.writeFileSync(bf, '/* RSI-TEST-MARKER */\n' + fs.readFileSync(bf, 'utf8'));
  // meanwhile another landing appends to the same file on main
  fs.appendFileSync(path.join(repo, INC), '/* OTHER-LANDING */\n');
  git('commit', '-q', '-am', 'another kernel landed'); git('push', '-q', 'origin', 'main');
  const pr = spawnSync(process.execPath, [RSI, 'patch', K], { encoding: 'utf8', env: { ...process.env, RSI_CONFIG: path.join(T, 'config.json') } });
  const out = pr.stdout + pr.stderr + land(), st = state();
  const f = ogit('show', `main:${INC}`).stdout;
  return expect(st.landed && f.includes('RSI-TEST-MARKER') && f.includes('OTHER-LANDING') && onMainClean(), 'the other landing was reverted or the change lost', out);
});
test('dirty repository: blocked, nothing changed', () => {
  fs.appendFileSync(path.join(repo, 'README.md'), 'x');
  const out = land(), st = state();
  return expect(st.blockedUntil && !st.landed && !st.landing && !originMarker(), 'expected blocked', out);
});
test('repository on a user branch: blocked', () => {
  git('checkout', '-q', '-b', 'user-work');
  const out = land(), st = state();
  return expect(st.blockedUntil && /not main/.test(st.blockedWhy) && git('rev-parse', '--abbrev-ref', 'HEAD').stdout.trim() === 'user-work', 'expected blocked on user branch', out);
});
test('patch conflicts with main: restart, repo back on main', () => {
  const inc = path.join(repo, INC), lines = fs.readFileSync(inc, 'utf8').split('\n');
  lines[0] = '/* changed by someone else */';
  fs.writeFileSync(inc, lines.join('\n'));
  git('commit', '-q', '-am', 'conflicting change'); git('push', '-q', 'origin', 'main');
  const out = land(), st = state();
  return expect(st.needsReinit && !st.landed && onMainClean() && !git('branch', '--list', 'rsi/*').stdout.trim(), 'expected restart', out);
});
test('patched build fails, main builds: restart', () => {
  mode('build.mode', 'failpatch');
  const out = land(), st = state();
  return expect(st.needsReinit && onMainClean() && !localMarker() && !broken(), 'expected restart after build failure', out);
});
test('build fails on main too: blocked (environment), not done', () => {
  mode('build.mode', 'fail');
  const out = land(), st = state();
  return expect(st.blockedUntil && !st.done && !st.needsReinit && onMainClean() && broken(), 'expected blocked with build marked broken', out);
});
test('engine/build behind main and main does not build: blocked before any branch', () => {
  mode('build.mode', 'fail');
  const t = new Date(Date.now() - 86400000);
  for (const f of ['ie-run.exe', 'ie_kernels.hsaco', 'ie_inkling_kernels.hsaco']) fs.utimesSync(path.join(repo, 'engine', 'build', f), t, t);
  const out = land(), st = state();
  return expect(st.blockedUntil && !st.done && onMainClean() && broken() && !git('branch', '--list', 'rsi/*').stdout.trim(), 'expected blocked before branching', out);
});
test('gate mismatch with the patch, main matches: done land_fail', () => {
  mode('gate.mode', 'patchbad');
  const out = land(), st = state();
  return expect(st.done?.reason === 'land_fail' && onMainClean() && !originMarker() && !remoteRsi(), 'expected land_fail', out);
});
test('gate mismatch on main as well: blocked, reference flagged', () => {
  mode('gate.mode', 'allbad');
  const out = land(), st = state();
  return expect(st.blockedUntil && !st.done && fs.existsSync(path.join(runs, '.gate-broken')) && onMainClean(), 'expected blocked with .gate-broken', out);
});
test('gate gives no hashes: blocked after a retry', () => {
  mode('gate.mode', 'missing');
  const out = land(), st = state();
  const calls = fs.readFileSync(path.join(T, 'gate.calls'), 'utf8').trim().split('\n').length;
  return expect(st.blockedUntil && !st.done && calls === 2 && onMainClean(), `expected blocked after 2 gate calls (got ${calls})`, out);
});
test('push fails, then succeeds on the next land: resumes and lands', () => {
  git('remote', 'set-url', '--push', 'origin', path.join(T, 'nowhere.git'));
  const out1 = land(), st1 = state();
  if (!(st1.landing?.stage === 'committed' && st1.blockedUntil && onMainClean() && broken())) return `first land should block at push\n${out1}`;
  git('remote', 'set-url', '--push', 'origin', origin);
  const out2 = land(), st2 = state();
  return expect(st2.landed && originMarker() && onMainClean() && broken(), 'expected resume to land (and build marked stale)', out1 + out2);
});
test('PR create fails, then lands on retry', () => {
  mode('gh.mode', 'createfail');
  const out1 = land(), st1 = state();
  if (!(st1.landing?.stage === 'pushed' && st1.blockedUntil)) return `should block at PR create\n${out1}`;
  mode('gh.mode', 'ok');
  const out2 = land();
  return expect(state().landed && originMarker() && !remoteRsi(), 'expected landed on retry', out1 + out2);
});
test('merge fails: blocked; retry lands', () => {
  mode('gh.mode', 'mergefail');
  const out1 = land(), st1 = state();
  if (!(st1.landing?.stage === 'pr' && st1.blockedUntil && onMainClean())) return `should block at merge\n${out1}`;
  mode('gh.mode', 'ok');
  const out2 = land();
  return expect(state().landed && originMarker(), 'expected landed on retry', out1 + out2);
});
test('merge fails every time: done land_error after the limit', () => {
  mode('gh.mode', 'mergefail');
  let out = '';
  for (let i = 0; i < 3; i++) out += land();
  const st = state();
  return expect(st.done?.reason === 'land_error' && onMainClean(), 'expected land_error after 3 tries', out);
});
test('PR not mergeable once: closed, landed again from scratch', () => {
  mode('gh.mode', 'conflict-once');
  const out = land(), st = state();
  const prs = JSON.parse(fs.readFileSync(path.join(T, 'gh.json'), 'utf8')).prs;
  return expect(st.landed && originMarker() && prs.length === 2 && prs[0].state === 'CLOSED' && prs[1].state === 'MERGED', 'expected close + second PR merged', out);
});
test('main moves while the PR is open: not merged on an ungated tree, landed again on the new main', () => {
  mode('gh.mode', 'movemain-once');
  const out = land(), st = state();
  const prs = JSON.parse(fs.readFileSync(path.join(T, 'gh.json'), 'utf8')).prs;
  const gates = fs.readFileSync(path.join(T, 'gate.calls'), 'utf8').trim().split('\n').length;
  const unrelated = ogit('log', '--format=%s', 'main').stdout.includes('unrelated change');
  return expect(st.landed && originMarker() && unrelated && prs[0].state === 'CLOSED' && gates === 2, `expected re-gate on new main (gates ${gates})`, out);
});
test("a person's branch named rsi/... with uncommitted work is not touched: blocked", () => {
  git('checkout', '-q', '-b', 'rsi/my-own-work');
  fs.appendFileSync(path.join(repo, 'README.md'), 'my uncommitted notes\n');
  const out = land(), st = state();
  const kept = fs.readFileSync(path.join(repo, 'README.md'), 'utf8').includes('my uncommitted notes');
  return expect(st.blockedUntil && kept && git('rev-parse', '--abbrev-ref', 'HEAD').stdout.trim() === 'rsi/my-own-work', 'the branch or its work was touched', out);
});
test('stray untracked files (repo root and engine/kernels) do not matter: lands only the patch', () => {
  fs.writeFileSync(path.join(repo, 'scratch.log'), 'x');
  fs.writeFileSync(path.join(repo, 'engine', 'kernels', 'notes.txt'), 'x');
  const out = land(), st = state();
  const files = ogit('show', '--name-only', '--format=', 'main').stdout.trim().split('\n');
  return expect(st.landed && files.length === 1 && files[0] === INC, `expected only ${INC} committed, got ${files.join(', ')}`, out);
});
test('a docs-only commit reaches main while the PR is open: lands without re-gating', () => {
  mode('gh.mode', 'movedocs-once');
  const out = land(), st = state();
  const gates = fs.readFileSync(path.join(T, 'gate.calls'), 'utf8').trim().split('\n').length;
  return expect(st.landed && originMarker() && gates === 1, `expected one gate (got ${gates})`, out);
});
test('the merge goes through but its status cannot be read: recognised as merged', () => {
  mode('gh.mode', 'viewfail-after-merge');
  const out = land(), st = state();
  return expect(st.landed && originMarker() && onMainClean(), 'expected landed', out);
});
test('a one-off gate mismatch: the re-run passes and it lands', () => {
  mode('gate.mode', 'flaky-once');
  const out = land();
  return expect(state().landed && originMarker(), 'expected landed after the re-run', out);
});
test('the whole model runs slower with the change: blocked, not landed', () => {
  mode('gate.mode', 'slow');
  const out = land(), st = state();
  return expect(st.blockedUntil && !st.landed && !st.done && !originMarker() && onMainClean(), 'expected blocked for speed', out);
});
test('GPU fault marker present: blocked before any build', () => {
  fs.writeFileSync(path.join(T, 'gpu-fault'), 'x');
  const out = land(), st = state();
  return expect(st.blockedUntil && !fs.existsSync(path.join(T, 'build.calls')) && onMainClean(), 'expected blocked with no build', out);
});
test('main already known not to match the reference: blocked before any build', () => {
  fs.mkdirSync(runs, { recursive: true });
  fs.writeFileSync(path.join(runs, '.gate-broken'), JSON.stringify({ since: new Date().toISOString() }));
  const out = land(), st = state();
  return expect(st.blockedUntil && !fs.existsSync(path.join(T, 'build.calls')) && !fs.existsSync(path.join(T, 'gate.calls')), 'expected blocked with no build or gate', out);
});
test('a landing that crashed mid-way (patch applied on an rsi/ branch): recovered and landed', () => {
  fs.mkdirSync(runs, { recursive: true });
  fs.writeFileSync(path.join(runs, '.landing-branch'), JSON.stringify({ branch: 'rsi/crashed-test', kernel: K }));
  git('checkout', '-q', '-b', 'rsi/crashed-test');
  fs.writeFileSync(path.join(repo, INC), '/* RSI-TEST-MARKER */\n' + fs.readFileSync(path.join(repo, INC), 'utf8'));
  const out = land(), st = state();
  return expect(st.landed && originMarker() && onMainClean() && !git('branch', '--list', 'rsi/crashed-test').stdout.trim(), 'expected recovery then landing', out);
});
test('a stale repository lock from a dead process is taken over', () => {
  fs.mkdirSync(path.join(runs, '.repo-lock'), { recursive: true });
  fs.writeFileSync(path.join(runs, '.repo-lock', 'owner.json'), JSON.stringify({ pid: 999999, what: 'dead', since: new Date().toISOString() }));
  const out = land();
  return expect(state().landed, 'expected the stale lock to be taken over', out);
});

const failed = results.filter((r) => !r.ok).length;
console.log(`\n${results.length - failed}/${results.length} passed`);
process.exit(failed ? 1 : 0);
