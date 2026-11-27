#!/usr/bin/env node
// rsi.mjs -- the measurement and landing side of the kernel-rsi loop. It finds the kernels a scenario runs, captures
// and ranks them, keeps a working copy of the kernel sources per kernel, builds each edit, checks it bit-identical and
// times it against the best version so far (engine/tools/kbench.c), keeps or reverts it, logs every attempt to a JSONL
// ledger, and lands a finished kernel on main (patch, full rebuild, exact-output gate, commit, PR, merge). The agent
// supplies the ideas and edits; this script decides nothing about code.
//
// The automatic cycle of a worker:
//   node rsi.mjs claim SCENARIO --worker NAME   next kernel to work on; "wait" (claim again) while kernels wait to retry,
//                                               "busy" when the rest is claimed by live workers, "none" when done
//   node rsi.mjs eval KERNEL --idea "TEXT" --tag TAG --worker NAME   one attempt, until it prints STOP
//   node rsi.mjs finish KERNEL --worker NAME    verify, land on main if it gained (or record why not), release
// Inspection and parts:
//   node rsi.mjs doctor [SCENARIO]              check tools, models, disk, builds and the repository
//   node rsi.mjs queue [SCENARIO]               the queue ranked by current time, with each kernel's loop state
//   node rsi.mjs status KERNEL [--json]         best so far, every idea tried, stop decision
//   node rsi.mjs isa KERNEL [--attempt N] [--dump LOOP]   instruction mix of the hot loops of a build
//   node rsi.mjs discover SCENARIO [--fresh]    trace + capture + rank the kernels of a scenario (claim does it)
//   node rsi.mjs init KERNEL [--scenario S] [--holdout S1,S2] [--force]   set a loop up (claim does it)
//   node rsi.mjs verify KERNEL                  re-measure baseline vs best on every capture (finish does it)
//   node rsi.mjs land KERNEL                    land a verified best on main (finish does it)
//   node rsi.mjs approve KERNEL                 let a verified kernel land (when requireApproval is set)
//   node rsi.mjs patch KERNEL                   write best.diff (base-src -> best-src) without measuring
//   node rsi.mjs release KERNEL --worker NAME | reopen KERNEL | capture SCENARIO KERNEL[,KERNEL...]
// Machine paths, scenarios and thresholds: rsi.config.json next to this file ($RSI_CONFIG overrides it).
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

// ---------------------------------------------------------------- errors, config
// kind: 'error' (a fault in the loop or its inputs), 'env' (machine or tools: retry later), 'blocked' (waits for the
// repository or a person), 'busy' (the GPU stayed in use)
class Fail extends Error {
  constructor(msg, kind = 'error') { super(msg); this.kind = kind; }
}
const die = (m, kind) => { throw new Fail(m, kind); };

const HERE = path.dirname(fileURLToPath(import.meta.url));
function loadConfig() {
  const read = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
  let c = read(path.join(HERE, 'rsi.config.json'));
  if (process.env.RSI_CONFIG) c = { ...c, ...read(process.env.RSI_CONFIG) };
  const fix = (v) => (typeof v === 'string' ? v.replaceAll('{lab}', c.lab) : v);
  for (const k of Object.keys(c)) c[k] = fix(c[k]);
  for (const k of Object.keys(c.prompts)) c.prompts[k] = fix(c.prompts[k]);
  return c;
}
const C = loadConfig();
// The repository that builds, captures and lands: the configured one, else the main worktree of the repository this
// file is in (a worker in a git worktree or a copy still lands through one place).
function findRepo() {
  if (C.repo && fs.existsSync(path.join(C.repo, 'engine'))) return path.resolve(C.repo);
  const r = spawnSync('git', ['rev-parse', '--path-format=absolute', '--git-common-dir'], { cwd: HERE, encoding: 'utf8' });
  return r.status === 0 ? path.dirname(r.stdout.trim()) : path.resolve(HERE, '../../../..');
}
const REPO = findRepo();
const ENGINE = path.join(REPO, 'engine');
const RUNS = C.runsDir;
const UNITS = {
  base: { src: 'kernels/ie_kernels.c', hsaco: 'ie_kernels.hsaco', warn: true },
  inkling: { src: 'kernels/ie_inkling_kernels.c', hsaco: 'ie_inkling_kernels.hsaco', warn: true },
  saluki: { src: 'kernels/ie_saluki_kernels.c', hsaco: 'ie_saluki_kernels.hsaco', warn: false },
  vision: { src: 'kernels/ie_vision_kernels.c', hsaco: 'ie_vision_kernels.hsaco', warn: true },
};
// every code object build-win.sh makes from the kernel sources: an edit to a shared file can change any of them
const VARIANTS = [
  ...Object.entries(UNITS).map(([u, d]) => ({ name: u, unit: u, src: d.src, defs: [], warn: d.warn })),
  { name: 'base_q8', src: 'kernels/ie_kernels.c', defs: ['-DIE_KV_Q8'], warn: true },
  { name: 'base_q4', src: 'kernels/ie_kernels.c', defs: ['-DIE_KV_Q4'], warn: true },
  ...['DECODE', 'MTP', 'FLINT', 'FLINTL', 'INKLING', 'INKLING_VER'].map((s) => ({ name: 'mk_' + s.toLowerCase(), src: 'kernels/ie_mk.c', defs: ['-DMKS_' + s, '-DIE_MK_VGPR=184'], warn: true })),
];
const WIN = process.platform === 'win32';
const exe = (name) => path.join(C.llvmBin, name + (WIN ? '.exe' : ''));
const pad = (n) => String(n).padStart(3, '0');
const now = () => new Date().toISOString();
const ago = (ts) => Date.now() - Date.parse(ts || 0);
const sleep = (ms) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
const slash = (p) => p.replaceAll('\\', '/');

// ---------------------------------------------------------------- files
function readJson(p, d = null) {
  if (!fs.existsSync(p)) return d;
  try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch (e) { die(`${p} is not valid JSON (${e.message})`); }
}
// write to a temporary file and rename it over the old one, so a crash never leaves half a file
function writeJson(p, v) {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  const tmp = `${p}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(v, null, 2) + '\n');
  for (let i = 0; ; i++) {
    try { fs.renameSync(tmp, p); return; } catch (e) { if (i >= 20) throw e; sleep(100); } /* a reader holds it */
  }
}
const kdir = (k) => path.join(RUNS, 'kernels', k);
const statePath = (k) => path.join(kdir(k), 'state.json');
const ledgerPath = (k) => path.join(kdir(k), 'ledger.jsonl');
const capDir = (sk) => path.join(RUNS, 'captures', sk);
const capPath = (sk, k) => path.join(capDir(sk), k + '.kcap');
const claimPath = (k) => path.join(kdir(k), 'claim.json');
// a loop's own capture: a hard link to the shared one, taken at init, so a later discovery cannot change it
const loopCap = (k, sk) => {
  const p = path.join(kdir(k), 'captures', sk.replace('@', '_') + '.kcap');
  return fs.existsSync(p) ? p : capPath(sk, k);
};
function linkCapture(k, sk) {
  const dst = path.join(kdir(k), 'captures', sk.replace('@', '_') + '.kcap');
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.rmSync(dst, { force: true });
  try { fs.linkSync(capPath(sk, k), dst); } catch { fs.copyFileSync(capPath(sk, k), dst); }
}
// the ledger, skipping a line a crash cut short
function ledger(k) {
  const p = ledgerPath(k);
  if (!fs.existsSync(p)) return [];
  const out = [];
  for (const l of fs.readFileSync(p, 'utf8').split('\n')) {
    if (!l.trim()) continue;
    try { out.push(JSON.parse(l)); } catch { /* torn line */ }
  }
  return out;
}
function appendLedger(k, rec) {
  const p = ledgerPath(k);
  fs.mkdirSync(path.dirname(p), { recursive: true });
  let lead = '';
  if (fs.existsSync(p)) {
    const s = fs.statSync(p).size;
    if (s > 0) { const fd = fs.openSync(p, 'r'), b = Buffer.alloc(1); fs.readSync(fd, b, 0, 1, s - 1); fs.closeSync(fd); if (b[0] !== 10) lead = '\n'; }
  }
  fs.appendFileSync(p, lead + JSON.stringify(rec) + '\n');
}
function copyDir(src, dst) { fs.rmSync(dst, { recursive: true, force: true }); fs.cpSync(src, dst, { recursive: true }); }
// engine directories outside kernels/ that the kernel sources include (../core/ie_core.h, ../prof/sm_profiler.h): a
// working copy needs them to build every code object, and an edit to them is out of scope (they are the whole engine's)
const SHARED = ['core', 'prof'];
// the shared directories and the generated headers (build/) from one root to another
function copyShared(from, to) {
  for (const s of [...SHARED, 'build']) if (fs.existsSync(path.join(from, s))) copyDir(path.join(from, s), path.join(to, s));
}
function sameTree(a, b) {
  if (!fs.existsSync(a) || !fs.existsSync(b)) return fs.existsSync(a) === fs.existsSync(b);
  const la = fs.readdirSync(a).sort(), lb = fs.readdirSync(b).sort();
  if (la.join('\0') !== lb.join('\0')) return false;
  for (const f of la) {
    const pa = path.join(a, f), pb = path.join(b, f);
    if (fs.statSync(pa).isDirectory()) { if (!sameTree(pa, pb)) return false; } else if (!fs.readFileSync(pa).equals(fs.readFileSync(pb))) return false;
  }
  return true;
}
function newestMtime(dir) {
  let t = 0;
  if (!fs.existsSync(dir)) return 0;
  for (const f of fs.readdirSync(dir)) {
    const p = path.join(dir, f), s = fs.statSync(p);
    t = Math.max(t, s.isDirectory() ? newestMtime(p) : s.mtimeMs);
  }
  return t;
}
function freeGB(p) {
  try { const s = fs.statfsSync(p); return (s.bavail * s.bsize) / 2 ** 30; } catch { return Infinity; }
}

// ---------------------------------------------------------------- processes
// never throws: the caller decides what a failure means. stdout alone is what a patch is made of; out has both streams.
// A timeout kills only the direct child, so o.orphans names programs it may have left running (killed then).
function run(cmd, argv, o = {}) {
  const r = spawnSync(cmd, argv, { encoding: 'utf8', maxBuffer: 512 << 20, timeout: o.timeout ?? 900000, cwd: o.cwd, env: o.env, windowsHide: true });
  const timedOut = !!(r.error && r.error.code === 'ETIMEDOUT');
  if (timedOut && o.orphans && WIN) for (const im of o.orphans) spawnSync('taskkill', ['/F', '/IM', im], { windowsHide: true });
  return { code: r.status, stdout: r.stdout || '', out: (r.stdout || '') + (r.stderr || '') + (r.error && !timedOut ? `\n${r.error.message}` : ''), timedOut, missing: r.error?.code === 'ENOENT' };
}
// GitHub CLI and Python, overridable (a test points them at stubs): config ghCmd / pythonCmd as [program, ...args]
const gh = (argv, o) => { const c = C.ghCmd || ['gh']; return run(c[0], [...c.slice(1), ...argv], o); };
const py = (argv, o) => { const c = C.pythonCmd || ['python']; return run(c[0], [...c.slice(1), ...argv], o); };
function runningNames() {
  const r = WIN ? run('tasklist', ['/FO', 'CSV', '/NH'], { timeout: 30000 }) : run('ps', ['-A', '-o', 'comm='], { timeout: 30000 });
  if (r.code !== 0) return [];
  return r.out.split('\n').map((l) => (WIN ? (l.match(/^"([^"]+)"/) || [])[1] : path.basename(l.trim()))).filter(Boolean).map((s) => s.toLowerCase());
}
function engineEnv(extra) {
  const env = {};
  for (const [k, v] of Object.entries(process.env)) if (!k.startsWith('IE_')) env[k] = v;
  const pk = Object.keys(env).find((k) => k.toUpperCase() === 'PATH') || 'PATH';
  env[pk] = C.rocmBin.replaceAll('/', path.sep) + path.delimiter + (env[pk] || '');
  return { ...env, ...extra };
}

// ---------------------------------------------------------------- locks
// A lock is a directory with owner.json. Its owner may be dead (taken over at once) or, with Windows reusing process
// ids, look alive long after: a lock older than its age limit is taken over too. Every held lock is removed when this
// process exits for any reason, Ctrl-C included.
const held = new Map();
function alive(pid) { try { process.kill(pid, 0); return true; } catch (e) { return e.code === 'EPERM'; } }
function releaseAll() {
  for (const p of held.keys()) {
    try { if (JSON.parse(fs.readFileSync(path.join(p, 'owner.json'), 'utf8')).pid === process.pid) fs.rmSync(p, { recursive: true, force: true }); } catch { /* gone */ }
  }
  held.clear();
}
process.on('exit', releaseAll);
for (const s of ['SIGINT', 'SIGTERM', 'SIGBREAK', 'SIGHUP']) { try { process.on(s, () => { releaseAll(); process.exit(130); }); } catch { /* not on this OS */ } }
// A stale lock is taken over by renaming it aside first: of two waiters that both find it stale, only one rename
// succeeds, so they cannot both end up holding it.
function takeOver(lockPath) {
  const aside = `${lockPath}.stale.${process.pid}.${Date.now()}`;
  try { fs.renameSync(lockPath, aside); } catch { return; }
  fs.rmSync(aside, { recursive: true, force: true });
}
// Programs outside this loop that use the GPU: the lab harness's locks (one left by a killed harness goes stale after
// 30 min with no engine running), and other engines, servers and benchmarks.
function gpuBusy(ignore = []) {
  for (const p of C.labLocks || [C.labGpuLock]) {
    if (!fs.existsSync(p)) continue;
    const old = Date.now() - fs.statSync(p).mtimeMs > 30 * 60000;
    if (old && p === C.labGpuLock && !runningNames().some((n) => n === 'ie-run.exe' || n === 'ie-serve.exe')) {
      console.log(`removing the lab's stale ${p} (over 30 min old, no engine running)`);
      fs.rmSync(p, { recursive: true, force: true });
      continue;
    }
    return `the lab harness (${path.basename(p)})`;
  }
  const other = runningNames().filter((n) => C.gpuBusyProcesses.includes(n) && !ignore.includes(n));
  return other.length ? `${[...new Set(other)].join(', ')} running` : null;
}
function withLock(lockPath, what, fn, { waitMin = 90, maxHours = 8, gpu = false } = {}) {
  if (held.has(lockPath)) return fn();
  fs.mkdirSync(path.dirname(lockPath), { recursive: true });
  const t0 = Date.now();
  let said = '';
  for (;;) {
    if (gpu && C.labFaultMarker && fs.existsSync(C.labFaultMarker))
      die(`the GPU fault marker ${C.labFaultMarker} is present: a GPU fault needs a person to check the card, then delete the file`, 'env');
    let busy = null;
    try {
      fs.mkdirSync(lockPath);
      fs.writeFileSync(path.join(lockPath, 'owner.json'), JSON.stringify({ pid: process.pid, what, since: now() }));
      held.set(lockPath, what);
      if (gpu && (busy = gpuBusy())) { held.delete(lockPath); fs.rmSync(lockPath, { recursive: true, force: true }); }
      if (!busy) break;
    } catch (e) {
      if (e.code !== 'EEXIST' && e.code !== 'EPERM') throw e;
      let o = null;
      try { o = JSON.parse(fs.readFileSync(path.join(lockPath, 'owner.json'), 'utf8')); } catch { /* being written, or never was */ }
      const age = o ? ago(o.since) : (() => { try { return Date.now() - fs.statSync(lockPath).mtimeMs; } catch { return 0; } })();
      if ((o && !alive(o.pid)) || age > maxHours * 3600000 || (!o && age > 120000)) { takeOver(lockPath); continue; }
      busy = o ? `${o.what} (pid ${o.pid}, since ${o.since.slice(11, 19)})` : 'another job';
    }
    if (Date.now() - t0 > waitMin * 60000) die(`waited ${waitMin} min: ${busy}`, 'busy');
    if (said !== busy) { console.log(`waiting: ${busy}`); said = busy; }
    sleep(3000);
  }
  try { return fn(); } finally {
    held.delete(lockPath);
    try { if (JSON.parse(fs.readFileSync(path.join(lockPath, 'owner.json'), 'utf8')).pid === process.pid) fs.rmSync(lockPath, { recursive: true, force: true }); } catch { /* already gone */ }
  }
}
const withGpu = (what, fn) => withLock(path.join(RUNS, '.gpu-lock'), what, fn, { gpu: true, maxHours: 6 });
const withRepo = (what, fn) => withLock(path.join(RUNS, '.repo-lock'), what, fn, { waitMin: 240, maxHours: 12 });
// only init creates a kernel's folder: anything else on a kernel with no loop is a mistake, not a new loop
const withKernel = (k, what, fn) => {
  if (!fs.existsSync(kdir(k))) die(`no loop for ${k}: claim it (or init it)`);
  return withLock(path.join(kdir(k), '.lock'), what, fn, { waitMin: 240, maxHours: 12 });
};

// ---------------------------------------------------------------- git and builds
const git = (argv, o = {}) => run('git', argv, { cwd: REPO, timeout: 300000, ...o });
function mustGit(argv, what, kind = 'error') {
  const r = git(argv);
  if (r.code !== 0) die(`${what}: git ${argv.join(' ')}\n${r.out.slice(-1500)}`, kind);
  return r.out.trim();
}
const branchNow = () => git(['rev-parse', '--abbrev-ref', 'HEAD']).out.trim();
const dirtyEngine = () => git(['status', '--porcelain', '--untracked-files=no', '--', 'engine']).out.trim();
const dirtyAll = () => git(['status', '--porcelain', '--untracked-files=no']).out.trim();
// both describe engine/build itself, so they live there (ignored by git), shared by every runs directory
const BROKEN = () => path.join(ENGINE, 'build', '.rsi-build-broken');
const BUILT = () => path.join(ENGINE, 'build', '.rsi-build-head'); /* the commit engine/build was built from */
const headSha = () => git(['rev-parse', 'HEAD']).out.trim();
// the whole engine (host + every code object) with build-win.sh; while it runs, and if it fails, .build-broken says
// that engine/build may not match the checked-out source. On success .build-head records the commit (null when the
// tree had uncommitted engine changes, e.g. a landing's patch before its commit).
function buildAll(why) {
  fs.mkdirSync(RUNS, { recursive: true });
  fs.writeFileSync(BROKEN(), JSON.stringify({ since: now(), why }));
  fs.rmSync(BUILT(), { force: true });
  const r = run(C.sh, [C.buildScript || 'build-win.sh'], { cwd: ENGINE, env: { ...process.env, LLVM_BIN: C.llvmBin }, timeout: 3600000, orphans: ['ie-run.exe'] });
  if (r.code !== 0) die(`build-win.sh failed (${why})${r.timedOut ? ': timed out' : ''}:\n${r.out.slice(-2500)}`, 'env');
  fs.rmSync(BROKEN(), { force: true });
  if (!dirtyEngine()) fs.writeFileSync(BUILT(), JSON.stringify({ sha: headSha(), ts: now() }));
}
// engine/build behind the checked-out code: compared by commit when the build recorded one (a checkout or pull that
// leaves newer file times but the same engine code is not stale), else by file times
function buildStale() {
  const outs = ['ie-run.exe', 'ie_kernels.hsaco', 'ie_inkling_kernels.hsaco'].map((f) => path.join(ENGINE, 'build', f));
  if (outs.some((p) => !fs.existsSync(p))) return 'engine/build is missing outputs';
  const b = readJson(BUILT(), null);
  if (b?.sha && git(['cat-file', '-e', b.sha + '^{commit}']).code === 0) {
    const same = git(['diff', '--quiet', b.sha, 'HEAD', '--', 'engine/src', 'engine/kernels', 'engine/core', 'engine/build-win.sh', 'engine/tools/mk_gen.py', 'engine/tools/build-inkling-kernels.sh', 'engine/tools/build-saluki-kernels.sh', 'engine/tools/build-vision-kernels.sh', 'engine/Makefile']).code === 0;
    return same && !dirtyEngine() ? null : 'the engine code changed since engine/build was built';
  }
  const built = Math.min(...outs.map((p) => fs.statSync(p).mtimeMs));
  const src = Math.max(...['src', 'kernels', ...SHARED].map((d) => newestMtime(path.join(ENGINE, d))));
  return src > built + 2000 ? 'engine sources are newer than engine/build' : null;
}
function ensureKbench() {
  const src = path.join(ENGINE, 'tools', 'kbench.c'), bin = path.join(ENGINE, 'build', 'kbench' + (WIN ? '.exe' : ''));
  if (fs.existsSync(bin) && fs.statSync(bin).mtimeMs >= fs.statSync(src).mtimeMs) return;
  withGpu('rebuild kbench', () => { /* no kbench runs while it is replaced */
    const tmp = bin + '.new' + (WIN ? '.exe' : '');
    const r = run(C.gcc, ['-std=c11', '-O2', '-o', tmp, src, ...(WIN ? [] : ['-ldl'])], { cwd: ENGINE, timeout: 300000 });
    if (r.code !== 0) die('building kbench failed:\n' + r.out.slice(-1500), 'env');
    fs.rmSync(bin, { force: true });
    fs.renameSync(tmp, bin);
  });
}

// A landing that crashed can leave the repository on its branch. Only the branch recorded in .landing-branch (written
// when a landing creates it) is touched, never a person's branch however it is named. Back to main: a branch with no
// commit of its own is deleted (it held only the applied patch); one with a commit is kept for the landing to resume.
const LANDING = () => path.join(RUNS, '.landing-branch');
function recoverRepo() {
  const br = branchNow();
  const rec = readJson(LANDING(), null);
  if (!rec || rec.branch !== br) return null;
  const ahead = +git(['rev-list', '--count', `main..${br}`]).out.trim() || 0;
  git(['reset', '-q', '--hard']);
  git(['checkout', '-q', 'main']);
  if (!ahead) git(['branch', '-q', '-D', br]);
  fs.rmSync(LANDING(), { force: true });
  fs.writeFileSync(BROKEN(), JSON.stringify({ since: now(), why: `recovered from ${br}` }));
  return `recovered the repository from an interrupted landing on ${br}${ahead ? ' (its commit is kept for the landing to resume)' : ''}`;
}

// ---------------------------------------------------------------- preflight
function doctor({ scenario, fix = true, quiet = false } = {}) {
  const problems = [], notes = [];
  const need = (p, what) => { if (!p || !fs.existsSync(p)) problems.push(`missing ${what}: ${p}`); };
  need(C.model, 'model'); need(C.dspark, 'DSpark model'); need(C.harness, 'benchmark harness'); need(C.sh, 'sh');
  for (const t of ['amdclang', 'ld.lld', 'llvm-readelf', 'llvm-objdump']) need(exe(t), t);
  for (const p of Object.values(C.prompts)) need(p, 'prompt');
  need(path.join(C.rocmBin, WIN ? 'amdhip64_7.dll' : ''), 'HIP runtime');
  if (scenario && !C.scenarios[scenario]) problems.push(`unknown scenario ${scenario} (${Object.keys(C.scenarios).join(', ')})`);
  if (run(C.gcc, ['--version'], { timeout: 30000 }).code !== 0) problems.push(`${C.gcc} is not runnable (kbench needs it)`);
  /* only landing needs these: a problem here makes landings wait, it does not stop the optimising */
  if (py(['--version'], { timeout: 30000 }).code !== 0) notes.push('warning: python is not runnable (rsi isa and the landing gate need it)');
  if (C.autoLand && gh(['auth', 'status'], { timeout: 60000 }).code !== 0) notes.push('warning: gh is not logged in: finished kernels will wait to land');
  if (C.autoLand && git(['remote', 'get-url', 'origin']).code !== 0) notes.push('warning: the repository has no origin remote: finished kernels will wait to land');
  if (fs.existsSync(path.join(RUNS, '.gate-broken'))) notes.push(`warning: main did not match the reference hashes at ${readJson(path.join(RUNS, '.gate-broken'), {}).since}: landings wait (fix main or docs/40round-ledger.json, then delete .gate-broken)`);
  if (C.labFaultMarker && fs.existsSync(C.labFaultMarker)) problems.push(`the GPU fault marker ${C.labFaultMarker} is present: check the card, then delete it`);
  for (const [where, p] of [['runs', RUNS], ['repository', REPO]]) {
    fs.mkdirSync(p, { recursive: true });
    const g = freeGB(p);
    if (g < C.minFreeGB) problems.push(`only ${g.toFixed(1)} GB free on the ${where} drive (needs ${C.minFreeGB})`);
  }
  if (!problems.length && fix) {
    withRepo('preflight', () => {
      const rec = recoverRepo();
      if (rec) notes.push(rec);
      const br = branchNow();
      if (br !== 'main') notes.push(`the repository is on ${br}, not main: results wait to land until it is back on main`);
      const dirty = dirtyEngine();
      if (dirty) die(`the repository has uncommitted changes under engine/ (${dirty.split('\n').length} files): commit or stash them first, so the loops measure and land against committed code`, 'blocked');
      const stale = fs.existsSync(BROKEN()) ? `the last build did not finish (${readJson(BROKEN(), {}).why || '?'})` : buildStale();
      if (stale) {
        console.log(`${stale}: rebuilding engine/build ...`);
        buildAll(stale); /* under the repository lock: captures and landings, the users of engine/build, take it too */
        notes.push('rebuilt engine/build');
      }
    });
    ensureKbench();
  }
  if (!quiet) for (const n of notes) console.log(n);
  if (problems.length) die('preflight:\n  ' + problems.join('\n  '), 'env');
  return notes;
}

// ---------------------------------------------------------------- engine runs, captures
function ieRun(sc, extra, what) {
  const s = C.scenarios[sc];
  if (!s) die(`unknown scenario ${sc}`);
  const argv = [C.model, '--backend', 'gpu', '--hsaco', 'build/ie_kernels.hsaco', '--tokens-file', C.prompts[s.prompt],
    '--ctx', '32768', '--n', '16', '--draft', '7', '--dspark', C.dspark, '--draft-pmin', '0.5'];
  const env = engineEnv({ IE_VERIFY_BENCH: String(s.T), IE_VERIFY_BENCH_TEXT: '1', ...extra });
  const r = run(path.join(ENGINE, 'build', 'ie-run.exe'), argv, { cwd: ENGINE, env, timeout: 900000 });
  if (r.code !== 0) die(`ie-run (${what}, ${sc}) ${r.timedOut ? 'timed out after 15 min' : `failed with code ${r.code}`}:\n${r.out.slice(-2000)}`, 'env');
  return r;
}
// sk is a scenario, or SCENARIO@mid: each kernel's launch in the middle of its run (another layer than the first)
function capture(sk, kernels, perStep = {}) {
  const [sc, mod] = sk.split('@');
  const s = C.scenarios[sc];
  if (!s) die(`unknown scenario ${sc}`);
  const names = kernels.map((k) => (mod === 'mid' ? `${k}@${Math.floor((perStep[k] || 1) / 2)}` : k));
  const est = kernels.length * 0.6;
  if (freeGB(RUNS) < est + C.minFreeGB / 2) die(`not enough disk for ${kernels.length} captures (${freeGB(RUNS).toFixed(1)} GB free)`, 'env');
  fs.mkdirSync(capDir(sk), { recursive: true });
  withGpu(`capture ${sk}`, () => {
    for (const k of kernels) fs.rmSync(capPath(sk, k), { force: true }); /* inside the lock: no eval is reading them now */
    ieRun(sc, { IE_KCAP: names.join(','), IE_KCAP_PHASE: s.phase, IE_KCAP_DIR: capDir(sk) }, 'capture');
  });
  const got = kernels.filter((k) => fs.existsSync(capPath(sk, k)));
  if (got.length < kernels.length) console.log(`not captured in ${sk}: ${kernels.filter((k) => !got.includes(k)).join(', ')}`);
  return got;
}

// ---------------------------------------------------------------- code objects
const symCache = new Map();
function symbols(hsaco) {
  const key = hsaco + ':' + (fs.existsSync(hsaco) ? fs.statSync(hsaco).mtimeMs : 0);
  if (!symCache.has(key)) {
    const out = run(exe('llvm-readelf'), ['--notes', hsaco]).out;
    symCache.set(key, out);
  }
  return symCache.get(key);
}
function unitOf(kernel) {
  for (const [u, d] of Object.entries(UNITS)) {
    const p = path.join(ENGINE, 'build', d.hsaco);
    if (fs.existsSync(p) && new RegExp(`\\n\\s+\\.name:\\s+${kernel}\\s`).test(symbols(p))) return u;
  }
  return null;
}
function resources(hsaco, kernel) {
  for (const blk of symbols(hsaco).split(/\n  - \.args:/)) {
    const m = blk.match(/\n\s+\.name:\s+(\S+)/);
    if (!m || m[1] !== kernel) continue;
    const g = (k) => { const v = blk.match(new RegExp('\\n\\s+\\.' + k + ':\\s+(\\d+)')); return v ? +v[1] : -1; };
    const vgpr = g('vgpr_count');
    return { vgpr, sgpr: g('sgpr_count'), lds: g('group_segment_fixed_size'), priv: g('private_segment_fixed_size'),
      vspill: g('vgpr_spill_count'), sspill: g('sgpr_spill_count'), waves: vgpr > 0 ? Math.min(16, Math.floor(1536 / (Math.ceil(vgpr / 8) * 8))) : 0 };
  }
  return null;
}
// a hash of each kernel's machine code (addresses and encodings left out), to see which kernels an edit changed
function codeHashes(hsaco) {
  const r = run(exe('llvm-objdump'), ['-d', '--mcpu=gfx1201', '--no-show-raw-insn', hsaco], { timeout: 600000 });
  const map = {};
  let cur = null, h = null;
  for (const line of r.out.split('\n')) {
    const m = line.match(/^[0-9a-f]+ <([^>]+)>:\s*$/);
    if (m) { if (cur) map[cur] = h.digest('hex'); cur = m[1]; h = crypto.createHash('sha1'); continue; }
    if (cur) h.update(line.replace(/\/\/.*$/, '').trim() + '\n');
  }
  if (cur) map[cur] = h.digest('hex');
  return map;
}
// build one unit from a source root (kernels/, core/, build/*.h) with the flags of build-win.sh
// the source files a variant's build reads (the compiler's dependency list), by file name
function deps(root, v) {
  const r = run(exe('amdclang'), ['-x', 'c', '-std=c11', '--target=amdgcn-amd-amdhsa', '-mcpu=gfx1201', '-nogpulib', '-Ibuild', ...v.defs, '-MM', v.src], { cwd: root, timeout: 300000 });
  if (r.code !== 0) return null;
  return [...new Set(r.stdout.replace(/\\\r?\n/g, ' ').replace(/^[^:]*:/, '').split(/\s+/).filter(Boolean).map((f) => path.basename(f)))];
}
// unit: a UNITS name, or a VARIANTS entry (its extra -D defines)
function compile(root, unit, out) {
  const d = typeof unit === 'string' ? { ...UNITS[unit], defs: [] } : unit;
  const ll = out + '.ll';
  const tgt = ['--target=amdgcn-amd-amdhsa', '-mcpu=gfx1201', '-nogpulib', '-O3'];
  const tidy = () => { fs.rmSync(ll, { force: true }); fs.rmSync(out + '.o', { force: true }); };
  let r = run(exe('amdclang'), ['-x', 'c', '-std=c11', ...(d.warn ? ['-Wall', '-Wextra'] : []), ...tgt, '-Ibuild', ...d.defs, '-S', '-emit-llvm', '-o', ll, d.src], { cwd: root, timeout: 1200000 });
  if (r.code !== 0) { tidy(); return { ok: false, log: r.timedOut ? 'the compiler ran over 20 min' : r.out }; }
  fs.writeFileSync(ll, fs.readFileSync(ll, 'utf8').replace(/^(attributes #\d+ = \{.*)"target-cpu"/gm, '$1"amdgpu-flat-work-group-size"="256,256" "target-cpu"'));
  r = run(exe('amdclang'), ['-nogpulib', ...tgt.slice(0, 2), '-O3', '-c', '-o', out + '.o', ll], { timeout: 1200000 });
  if (r.code !== 0) { tidy(); return { ok: false, log: r.timedOut ? 'the code generator ran over 20 min' : r.out }; }
  r = run(exe('ld.lld'), ['-shared', '-o', out, out + '.o']);
  tidy();
  return r.code === 0 ? { ok: true, log: '' } : { ok: false, log: r.out };
}

// ---------------------------------------------------------------- kbench
const KB_KIND = { 0: 'ok', 3: 'replay', 4: 'nondet', 5: 'differs', 6: 'notiming', 7: 'fault' };
// never throws for a verdict: returns { kind, replay, identical, ref, cand, deltaPct, why, out }
function kbench(cap, ref, cand, o = {}) {
  const kb = path.join(ENGINE, 'build', 'kbench' + (WIN ? '.exe' : ''));
  const capMB = fs.statSync(cap).size / 2 ** 20;
  const cold = Math.max(1, Math.min(o.cold ?? C.kbench.cold, Math.floor(C.kbench.vramBudgetMB / Math.max(capMB, 1))));
  const argv = [cap, ref, ...(cand ? [cand] : []), '--cold', String(cold), '-r', String(o.rounds ?? C.kbench.rounds), '-n', String(C.kbench.launches)];
  const r = run(kb, argv, { env: engineEnv({}), timeout: (o.timeoutSec ?? C.kbench.timeoutSec) * 1000 });
  const num = (re) => { const m = r.out.match(re); return m ? [+m[1], +m[2]] : null; };
  const res = {
    kind: r.timedOut ? 'hang' : KB_KIND[r.code] || 'error', cold,
    replay: /replay matches engine: yes/.test(r.out),
    identical: cand ? /candidate output: bit-identical/.test(r.out) : null,
    ref: num(/ref : median\s+([\d.]+) us\s+min\s+([\d.]+)/),
    cand: cand ? num(/cand: median\s+([\d.]+) us\s+min\s+([\d.]+)/) : null,
    why: ((r.out.match(/DIFFERS from REF \(([^)]*)\)/) || [])[1] || '') + ((r.out.match(/^  (?:REF vs CAND|CAND run|two launches)[^\n]*/m) || [''])[0] ? ': ' + r.out.match(/^  (?:REF vs CAND|CAND run|two launches)[^\n]*/m)[0].trim() : ''),
    out: r.out.slice(-1500),
  };
  if (res.kind === 'ok' && (!res.ref || (cand && !res.cand))) res.kind = 'error';
  res.deltaPct = res.ref && res.cand ? (100 * (res.cand[0] - res.ref[0])) / res.ref[0] : null;
  if (res.kind === 'fault' || res.kind === 'hang') sleep(15000); /* let the driver recover the GPU */
  return res;
}

// ---------------------------------------------------------------- state
// A new best is staged (best-src.next, best.hsaco.next, promote.json) before state.json names it, then swapped in.
// Loading under the kernel's lock finishes or drops a promotion a crash interrupted; a reader without the lock (status,
// queue) only reads, so it cannot drop a promotion an eval is half-way through.
function loadState(k) {
  const d = kdir(k), st = readJson(statePath(k));
  if (!st) return null;
  const pj = path.join(d, 'promote.json');
  if (fs.existsSync(pj) && held.has(path.join(d, '.lock'))) {
    const p = readJson(pj, {});
    if (st.best.attempt === p.attempt) {
      if (fs.existsSync(path.join(d, 'best-src.next'))) { fs.rmSync(path.join(d, 'best-src'), { recursive: true, force: true }); fs.renameSync(path.join(d, 'best-src.next'), path.join(d, 'best-src')); }
      if (fs.existsSync(path.join(d, 'best.hsaco.next'))) fs.renameSync(path.join(d, 'best.hsaco.next'), path.join(d, 'best.hsaco'));
    } else {
      fs.rmSync(path.join(d, 'best-src.next'), { recursive: true, force: true });
      fs.rmSync(path.join(d, 'best.hsaco.next'), { force: true });
    }
    fs.rmSync(pj, { force: true });
  }
  return st;
}
const saveState = (k, st) => writeJson(statePath(k), st);

function claimOf(k) {
  let c = null;
  try { c = JSON.parse(fs.readFileSync(claimPath(k), 'utf8')); } catch { return null; }
  const led = ledger(k);
  const last = Math.max(Date.parse(c.since || 0), Date.parse(c.beat || 0), led.length ? Date.parse(led[led.length - 1].ts) : 0);
  return Date.now() - last > C.claimStaleHours * 3600000 ? null : c; /* a silent worker's claim expires */
}
function heartbeat(k) {
  const c = readJson(claimPath(k), null);
  if (c) writeJson(claimPath(k), { ...c, beat: now() });
}
// a live claim belongs to its worker: eval, finish and release from anyone else (or from no one named) are refused
function checkOwner(k, worker) {
  const c = claimOf(k);
  if (c && c.worker !== worker) die(`${k} is claimed by ${c.worker}${worker ? '' : ' (pass --worker NAME)'}`, 'blocked');
}
// attempts and patience count from the loop's current baseline: a restart on newer code gets a fresh allowance
function stopCheck(st) {
  if (st.stopped) return { stop: true, why: st.stopped };
  const used = st.attempts - (st.baseline?.attempt || 0);
  const since = st.attempts - st.lastAccept;
  const last = st.accepts.slice(-3).map((a) => -a.gainPct);
  if (used >= C.stop.maxAttempts) return { stop: true, why: `reached ${C.stop.maxAttempts} attempts` };
  if (since >= C.stop.patience) return { stop: true, why: `no accepted gain in the last ${since} attempts (patience ${C.stop.patience})` };
  if (last.length === 3 && last.reduce((a, b) => a + b, 0) < C.stop.diminishingPct && since >= Math.ceil(C.stop.patience / 2))
    return { stop: true, why: `diminishing returns: the last 3 gains total ${last.reduce((a, b) => a + b, 0).toFixed(2)}% and ${since} attempts since` };
  return { stop: false, why: `${since}/${C.stop.patience} attempts since the last gain, ${used}/${C.stop.maxAttempts} in all` };
}
// heavy files of a finished loop; the ledger, state, best-src and best.diff stay
function prune(k) {
  const d = kdir(k);
  for (const p of ['captures', 'work', 'stage', 'base-src', 'core-src', 'best.hsaco', 'baseline.hsaco', 'main.hsaco', 'best-src.next', 'best.hsaco.next'])
    fs.rmSync(path.join(d, p), { recursive: true, force: true });
  const ad = path.join(d, 'attempts');
  if (fs.existsSync(ad)) for (const f of fs.readdirSync(ad)) if (f.endsWith('.hsaco')) fs.rmSync(path.join(ad, f), { force: true });
}

// ---------------------------------------------------------------- discover, queue
function cmdDiscover(sc, o = {}) {
  const s = C.scenarios[sc];
  if (!s) die(`unknown scenario ${sc}`);
  return withRepo(`discover ${sc}`, () => {
    const hash = buildHash();
    if (o.fresh) { fs.rmSync(capDir(sc), { recursive: true, force: true }); fs.rmSync(capDir(sc + '@mid'), { recursive: true, force: true }); }
    const csv = path.join(RUNS, 'discover', sc + '.csv');
    fs.mkdirSync(path.dirname(csv), { recursive: true });
    fs.rmSync(csv, { force: true });
    console.log(`1/3 tracing the launches of ${sc} (${s.about}) ...`);
    withGpu(`trace ${sc}`, () => ieRun(sc, { IE_TRACE: csv }, 'trace'));
    if (!fs.existsSync(csv)) die('the engine wrote no trace', 'env');
    const lines = fs.readFileSync(csv, 'utf8').trim().split('\n');
    const hdr = lines[0].split(',');
    const ix = (n) => hdr.indexOf(n);
    const counts = {}, calls = new Set(), groups = {};
    for (const l of lines.slice(1)) {
      const f = l.split(',');
      if (f.length < hdr.length || f[ix('phase')] !== s.phase) continue;
      if (s.phase === 'verify' && +f[ix('T_run')] !== s.T) continue;
      calls.add(f[ix('call')]);
      const k = f[ix('kernel')];
      counts[k] = (counts[k] || 0) + 1;
      groups[k] = f[ix('groups')] + 'x' + f[ix('T')];
    }
    if (!calls.size) die(`no ${s.phase} launches in the trace of ${sc}`, 'env');
    const kernels = Object.keys(counts).filter((k) => unitOf(k));
    console.log(`2/3 capturing ${kernels.length} kernels (${calls.size} ${s.phase} runs traced) ...`);
    const todo = kernels.filter((k) => !fs.existsSync(capPath(sc, k)));
    if (todo.length) capture(sc, todo);
    console.log('3/3 timing each capture alone (cold) ...');
    const rows = [];
    for (const k of kernels) {
      if (!fs.existsSync(capPath(sc, k))) continue;
      const unit = unitOf(k);
      const r = withGpu(`rank ${k}`, () => kbench(capPath(sc, k), path.join(ENGINE, 'build', UNITS[unit].hsaco), null, { rounds: 5 }));
      const perStep = counts[k] / calls.size;
      const nondet = r.kind === 'nondet', ok = r.kind === 'ok';
      if (nondet) console.log(`  ${k}: not deterministic (atomics decide some output bytes); the loop cannot judge it`);
      else if (!ok) console.log(`  ${k}: ${r.kind === 'replay' ? 'the replay does not match the engine' : 'kbench ' + r.kind}; skipped`);
      rows.push({ kernel: k, unit, perStep: +perStep.toFixed(2), grid: groups[k], us: ok ? r.ref[0] : 0, msPerStep: ok ? +((r.ref[0] * perStep) / 1000).toFixed(4) : 0, replay: ok, nondet });
    }
    /* no capture replaying at all is a fault in the tools (kbench, the capture hook), not in every kernel at once */
    if (rows.length && !rows.some((r) => r.replay || r.nondet)) die(`no capture of ${sc} replays (${rows.length} tried): kbench or the capture is broken; the queue was not written`, 'env');
    rows.sort((a, b) => b.msPerStep - a.msPerStep);
    const total = rows.reduce((a, r) => a + r.msPerStep, 0) || 1;
    for (const r of rows) r.share = +((100 * r.msPerStep) / total).toFixed(1);
    writeJson(path.join(RUNS, 'queue-' + sc + '.json'), { scenario: sc, about: s.about, created: now(), buildHash: hash, runsTraced: calls.size, totalMsPerStep: +total.toFixed(3), kernels: rows });
    printQueue(sc);
  });
}
function buildHash() {
  const h = crypto.createHash('sha1');
  for (const f of ['ie-run.exe', 'ie_kernels.hsaco', 'ie_inkling_kernels.hsaco']) {
    const p = path.join(ENGINE, 'build', f);
    if (fs.existsSync(p)) h.update(fs.readFileSync(p));
  }
  return h.digest('hex').slice(0, 16);
}
// the queue of a scenario for the current build, discovered again (fresh captures) when the build changed; two workers
// that both notice wait for each other on the repository lock and the second finds the new queue
function ensureQueue(sc) {
  const p = path.join(RUNS, 'queue-' + sc + '.json');
  const q0 = readJson(p);
  if (q0 && q0.buildHash === buildHash()) return q0;
  return withRepo(`queue ${sc}`, () => {
    const q = readJson(p);
    if (q && q.buildHash === buildHash()) return q;
    console.log(q ? `the engine build changed since the queue of ${sc} was made: discovering again ...` : `no queue for ${sc} yet: discovering ...`);
    cmdDiscover(sc, { fresh: !!q });
    return readJson(p);
  });
}
function current(r) {
  const st = readJson(statePath(r.kernel));
  const f = st && !st.landed && st.speedup > 0 ? 1 / st.speedup : 1;
  return { ...r, st, curUs: r.us * f, curMs: r.msPerStep * f };
}
function ranked(q) {
  const rows = q.kernels.map(current).sort((a, b) => b.curMs - a.curMs);
  const total = rows.reduce((a, r) => a + r.curMs, 0) || 1;
  for (const r of rows) r.curShare = (100 * r.curMs) / total;
  return rows;
}
function loopState(k) {
  const st = readJson(statePath(k));
  const c = claimOf(k);
  const who = c ? ` @${c.worker}` : '';
  if (!st) return 'not started' + who;
  if (st.landed) return `landed x${st.speedup.toFixed(3)} (${st.landed.pr || 'merged'})`;
  if (st.done) return `done: ${st.done.reason}`;
  if (st.landing) return `landing (${st.landing.stage})${who}`;
  if (st.blockedUntil && Date.parse(st.blockedUntil) > Date.now()) return `waiting to land until ${st.blockedUntil.slice(11, 16)}: ${st.blockedWhy}`;
  const s = stopCheck(st);
  return `${s.stop ? 'stopped' : 'running'} ${st.attempts} tries, x${st.speedup.toFixed(3)}${st.verified?.ok ? ', verified' : ''}${who}`;
}
function printQueue(sc) {
  const files = sc ? ['queue-' + sc + '.json'] : fs.existsSync(RUNS) ? fs.readdirSync(RUNS).filter((f) => /^queue-.*\.json$/.test(f)) : [];
  if (!files.length || (sc && !fs.existsSync(path.join(RUNS, files[0])))) die('no queue yet: claim makes one (or run discover SCENARIO)');
  for (const f of files) {
    const q = readJson(path.join(RUNS, f));
    const stale = q.buildHash !== buildHash();
    console.log(`\n${q.scenario}: ${q.about}; ${q.totalMsPerStep} ms per run at discovery (cold, each kernel alone)${stale ? '\n  (the engine build changed since: the next claim discovers again)' : ''}`);
    console.log('  #  share  ms/run now  us/launch  per-run  kernel  [loop]');
    ranked(q).forEach((r, i) => console.log(`${String(i + 1).padStart(3)} ${r.curShare.toFixed(1).padStart(5)}%  ${r.curMs.toFixed(3).padStart(10)}  ${r.curUs.toFixed(2).padStart(9)}  ${String(r.perStep).padStart(7)}  ${r.kernel}${r.nondet ? ' (NOT DETERMINISTIC)' : r.replay ? '' : ' (NOT REPLAYABLE)'}  [${loopState(r.kernel)}]`));
  }
}

// ---------------------------------------------------------------- claim
// what a kernel needs: 'optimise', 'finish', or null (finished, waiting, needs a person, or not judgeable)
const waitingNow = (st) => st?.blockedUntil && Date.parse(st.blockedUntil) > Date.now();
const needsPerson = (st) => !!st && !st.landed && !st.done && ((st.blockCount || 0) >= (C.maxBlocks || 8) || (C.requireApproval && st.verified?.ok && !st.approved && !st.landing && stopCheck(st).stop));
function need(r) {
  if (!r.replay || r.nondet) return null;
  const st = r.st;
  if (st?.landed || st?.done || waitingNow(st) || needsPerson(st)) return null;
  if (!st || st.needsReinit) return 'optimise';
  if (st.landing) return 'finish';
  return stopCheck(st).stop ? 'finish' : 'optimise';
}
function takeClaim(k, worker) {
  fs.mkdirSync(kdir(k), { recursive: true });
  const c = (() => { try { return JSON.parse(fs.readFileSync(claimPath(k), 'utf8')); } catch { return null; } })();
  if (c && c.worker !== worker && claimOf(k)) return false;
  writeJson(claimPath(k), { worker, since: now(), beat: now() });
  return true;
}
function cmdClaim(sc, o) {
  if (!sc || !C.scenarios[sc]) die(`usage: claim SCENARIO --worker NAME (scenarios: ${Object.keys(C.scenarios).join(', ')})`);
  if (typeof o.worker !== 'string' || !o.worker) die('claim needs --worker NAME (a name no other worker uses)');
  const worker = o.worker;
  doctor({ scenario: sc });
  // nothing to claim but kernels waiting for a retry (a landing blocked by the network, a busy repository): wait here
  // for the earliest retry, up to claimWaitMinutes, then say "wait" so the worker claims again
  const t0 = Date.now(), budget = (C.claimWaitMinutes ?? 10) * 60000;
  let pick = null;
  for (;;) {
    const q = ensureQueue(sc);
    let soonest = null;
    pick = withLock(path.join(RUNS, '.claim-lock'), `claim for ${worker}`, () => {
      const rows = ranked(q);
      const mine = rows.find((r) => need(r) && readJson(claimPath(r.kernel), {})?.worker === worker && claimOf(r.kernel));
      if (mine) return mine;
      const free = rows.find((r) => need(r) && r.share >= C.minSharePct && takeClaim(r.kernel, worker));
      if (free) return free;
      const live = rows.filter((r) => r.replay && !r.nondet && r.share >= C.minSharePct && r.st && !r.st.landed && !r.st.done);
      const busy = rows.filter((r) => need(r) && claimOf(r.kernel)).map((r) => `${r.kernel} @${claimOf(r.kernel).worker}`);
      const waiting = live.filter((r) => waitingNow(r.st) && !needsPerson(r.st));
      const person = live.filter((r) => needsPerson(r.st)).map((r) => `${r.kernel} (${C.requireApproval && r.st.verified?.ok && !r.st.approved ? 'waits for rsi approve' : r.st.blockedWhy})`);
      /* other workers still active: they claim again when they finish and pick the waiting kernels up then */
      if (busy.length) { console.log(`busy: the rest of ${sc} is claimed by live workers: ${busy.join(', ')}${waiting.length ? `; ${waiting.length} waiting to retry` : ''}`); return null; }
      if (waiting.length) { soonest = Math.min(...waiting.map((r) => Date.parse(r.st.blockedUntil))); return null; }
      console.log(`none: nothing left to do in ${sc}${person.length ? `; needs a person: ${person.join(', ')}` : ` (every kernel above ${C.minSharePct}% is finished)`}`);
      return null;
    }, { waitMin: 120, maxHours: 0.5 });
    if (pick || soonest === null) break;
    const left = budget - (Date.now() - t0);
    if (left <= 0) { console.log(`wait: kernels are waiting to retry (the next at ${new Date(soonest).toISOString().slice(11, 16)} UTC); run claim again`); return; }
    const ms = Math.max(5000, Math.min(soonest - Date.now() + 1000, left));
    console.log(`kernels are waiting to retry: waiting ${Math.round(ms / 1000)} s ...`);
    sleep(ms);
  }
  if (!pick) return;
  const k = pick.kernel;
  try {
    if (!pick.st || pick.st.needsReinit || !fs.existsSync(path.join(kdir(k), 'base-src'))) cmdInit(k, { scenario: sc, force: true, keepHistory: !!pick.st?.best, perStep: pick.perStep });
  } catch (e) {
    fs.rmSync(claimPath(k), { force: true });
    if (e instanceof Fail) {
      const st = readJson(statePath(k)) || { kernel: k, attempts: 0, lastAccept: 0, accepts: [], speedup: 1, needsReinit: true };
      if (e.kind === 'error') st.done = { reason: 'setup failed: ' + e.message.split('\n')[0].slice(0, 200), ts: now() }; /* the next claim moves on */
      else { st.blockedUntil = new Date(Date.now() + C.blockedRetryMinutes * 60000).toISOString(); st.blockedWhy = 'setup: ' + e.message.split('\n')[0].slice(0, 160); st.blockCount = (st.blockCount || 0) + 1; }
      saveState(k, st);
    }
    throw e;
  }
  const st = loadState(k);
  const act = need(current(pick));
  console.log(`\nclaimed ${k} for ${worker}: ${act === 'finish' ? 'its loop has stopped; run finish' : 'optimise it'} (${pick.curShare.toFixed(1)}% of ${sc}, best ${st.best.us.toFixed(2)} us, ${st.attempts} attempts so far)`);
  console.log(`work copy: ${slash(path.join(kdir(k), 'work', 'kernels'))}`);
  console.log(act === 'finish' ? `next: rsi finish ${k} --worker ${worker}` : `next: rsi status ${k}, then edit the work copy and rsi eval ${k} --worker ${worker} until STOP, then rsi finish ${k} --worker ${worker}`);
}
function cmdRelease(k, o) {
  if (!o.force) checkOwner(k, o.worker);
  fs.rmSync(claimPath(k), { force: true });
  console.log(`released ${k}`);
}
// with requireApproval, a verified kernel lands only after a reviewer approves it
function cmdApprove(k, o) {
  withKernel(k, `approve ${k}`, () => {
    const st = loadState(k);
    if (!st?.verified?.ok) die(`${k} has no verified result to approve`);
    st.approved = { by: o.worker || 'reviewer', ts: now(), attempt: st.best.attempt };
    saveState(k, st);
    appendLedger(k, { ts: now(), kernel: k, attempt: st.attempts, decision: 'approved', by: st.approved.by });
  });
  console.log(`${k} approved: the next claim lands it`);
}
// a finished kernel's loop starts again on the current code at the next claim, keeping its history
function cmdReopen(k) {
  withKernel(k, `reopen ${k}`, () => {
    const st = loadState(k);
    if (!st) die(`no loop for ${k}`);
    st.history = [...(st.history || []), { landed: st.landed || null, done: st.done || null, ts: now() }];
    delete st.landed; delete st.done; delete st.landing; delete st.blockedUntil; delete st.blockedWhy;
    st.needsReinit = true;
    saveState(k, st);
    appendLedger(k, { ts: now(), kernel: k, attempt: st.attempts, decision: 'reopen' });
  });
  console.log(`${k} reopens at the next claim`);
}

// ---------------------------------------------------------------- init
function cmdInit(k, o) {
  if (!k) die('usage: init KERNEL [--scenario S] [--holdout S1,S2] [--force]');
  fs.mkdirSync(kdir(k), { recursive: true }); /* init is what creates a loop */
  return withKernel(k, `init ${k}`, () => withRepo(`init ${k}`, () => {
    const unit = unitOf(k);
    if (!unit) die(`${k} is in no code object under engine/build`);
    const d = kdir(k);
    const old = loadState(k);
    if (old && !o.force) die(`${k} already has a loop (status ${k}); --force starts over`);
    let sc = o.scenario, perStep = o.perStep;
    for (const f of fs.existsSync(RUNS) ? fs.readdirSync(RUNS).filter((f) => /^queue-.*\.json$/.test(f)) : []) {
      const q = readJson(path.join(RUNS, f)), r = q.kernels.find((x) => x.kernel === k);
      if (r && (!sc || q.scenario === sc)) { sc ??= q.scenario; perStep ??= r.perStep; break; }
    }
    if (!sc) die('no queue lists this kernel: pass --scenario');
    const holdouts = typeof o.holdout === 'string' ? o.holdout.split(',').filter(Boolean) : [...(C.defaultHoldouts[sc] || []), ...((perStep || 1) >= 2 ? [sc + '@mid'] : [])];
    if (!fs.existsSync(capPath(sc, k))) capture(sc, [k]);
    if (!fs.existsSync(capPath(sc, k))) die(`${k} does not run in ${sc}`);
    for (const h of holdouts) if (!fs.existsSync(capPath(h, k))) capture(h, [k], { [k]: perStep });
    // keepHistory (a restart on newer engine code): the ledger and attempt numbers stay, so status still lists every
    // idea tried; the baseline, best and patience start again
    const keep = o.keepHistory && old;
    const oldLedger = keep && fs.existsSync(ledgerPath(k)) ? fs.readFileSync(ledgerPath(k), 'utf8') : '';
    const claim = fs.existsSync(claimPath(k)) ? fs.readFileSync(claimPath(k)) : null;
    // the old loop (its best-src, best.diff and attempt diffs: the gains to re-apply) moves to history/, not away
    let archive = null;
    if (fs.existsSync(d) && fs.readdirSync(d).some((f) => !['claim.json', '.lock', 'state.json', 'ledger.jsonl'].includes(f))) {
      archive = path.join(RUNS, 'history', k, now().replace(/[:.]/g, '-'));
      fs.mkdirSync(path.dirname(archive), { recursive: true });
      const keepLock = held.has(path.join(d, '.lock'));
      fs.mkdirSync(archive);
      for (const f of fs.readdirSync(d)) if (f !== '.lock' && f !== 'claim.json') fs.renameSync(path.join(d, f), path.join(archive, f));
      for (const p of ['captures', 'work', 'stage', 'core-src', 'best.hsaco', 'baseline.hsaco', 'main.hsaco']) fs.rmSync(path.join(archive, p), { recursive: true, force: true });
      if (!keepLock) fs.rmSync(d, { recursive: true, force: true });
    } else if (!held.has(path.join(d, '.lock'))) fs.rmSync(d, { recursive: true, force: true });
    for (const f of fs.existsSync(d) ? fs.readdirSync(d) : []) if (f !== '.lock') fs.rmSync(path.join(d, f), { recursive: true, force: true });
    fs.mkdirSync(path.join(d, 'attempts'), { recursive: true });
    if (oldLedger) fs.writeFileSync(ledgerPath(k), oldLedger);
    if (claim) fs.writeFileSync(claimPath(k), claim);
    const hold = [];
    linkCapture(k, sc);
    for (const h of holdouts) if (fs.existsSync(capPath(h, k))) { linkCapture(k, h); hold.push(h); }
    const work = path.join(d, 'work');
    for (const sub of ['kernels', ...SHARED]) if (fs.existsSync(path.join(ENGINE, sub))) fs.cpSync(path.join(ENGINE, sub), path.join(work, sub), { recursive: true });
    fs.mkdirSync(path.join(work, 'build'), { recursive: true });
    for (const f of fs.readdirSync(path.join(ENGINE, 'build')).filter((f) => f.endsWith('.h'))) fs.copyFileSync(path.join(ENGINE, 'build', f), path.join(work, 'build', f));
    copyDir(path.join(work, 'kernels'), path.join(d, 'best-src'));
    /* base-src: the sources the loop started from; the patch that lands is base-src -> best-src, so it carries only
     * this loop's changes, never the reverse of what landed on main since */
    copyDir(path.join(work, 'kernels'), path.join(d, 'base-src'));
    copyShared(work, path.join(d, 'core-src'));
    console.log(`building the baseline ${UNITS[unit].hsaco} from a copy of engine/ ...`);
    const c = compile(work, unit, path.join(d, 'baseline.hsaco'));
    if (!c.ok) die('the baseline does not build:\n' + c.log.slice(-2500), 'env');
    fs.copyFileSync(path.join(d, 'baseline.hsaco'), path.join(d, 'best.hsaco'));
    const same = fs.readFileSync(path.join(d, 'baseline.hsaco')).equals(fs.readFileSync(path.join(ENGINE, 'build', UNITS[unit].hsaco)));
    const res = resources(path.join(d, 'baseline.hsaco'), k);
    const r = withGpu(`baseline ${k}`, () => kbench(loopCap(k, sc), path.join(d, 'best.hsaco'), path.join(d, 'best.hsaco')));
    if (r.kind !== 'ok') die(`the baseline replay of ${k} gives ${r.kind}:\n${r.out}`, r.kind === 'notiming' || r.kind === 'hang' ? 'env' : 'error');
    const n0 = keep ? old.attempts : 0;
    const st = { kernel: k, unit, scenario: sc, holdouts: hold, perStep, created: now(), sameAsEngineBuild: same, buildHash: buildHash(),
      baseline: { us: r.ref[0], min: r.ref[1], noisePct: +Math.abs(r.deltaPct).toFixed(2), res, attempt: n0 },
      best: { attempt: n0, us: r.ref[0], res }, speedup: 1, attempts: n0, lastAccept: n0, accepts: [], history: [...(keep ? old.history || [] : []), ...(archive ? [{ archive: slash(archive), ts: now(), best: old?.best || null, speedup: old?.speedup ?? null }] : [])] };
    saveState(k, st);
    appendLedger(k, { ts: now(), kernel: k, attempt: n0, decision: keep ? 'rebase' : 'baseline', scenario: sc, holdouts: hold, ref_us: r.ref[0], cand_us: r.ref[0], delta_pct: 0, noise_pct: st.baseline.noisePct, res, best_us: r.ref[0], speedup: 1 });
    console.log(`${k}: ${unit}, scenario ${sc}${hold.length ? ', holdouts ' + hold.join(', ') : ''}`);
    console.log(`baseline ${r.ref[0].toFixed(2)} us (self A/B ${r.deltaPct.toFixed(2)}%), VGPR ${res.vgpr} (${res.waves} waves), LDS ${res.lds}${same ? '' : ', NOTE: not byte-identical to engine/build'}`);
  }));
}

// ---------------------------------------------------------------- eval
// An editor on Windows may save a file with CRLF; the repository's sources use LF. A file whose line endings changed
// gets the original ones back, so the diff (and the commit that lands) holds only the real change.
function keepLineEndings(bestDir, workDir) {
  for (const f of fs.readdirSync(workDir)) {
    const w = path.join(workDir, f), b = path.join(bestDir, f);
    if (!fs.existsSync(b) || !fs.statSync(w).isFile()) continue;
    const wt = fs.readFileSync(w, 'latin1'), bt = fs.readFileSync(b, 'latin1');
    if (wt === bt) continue;
    const bCrlf = bt.includes('\r\n'), wCrlf = wt.includes('\r\n');
    if (!bCrlf && wCrlf) fs.writeFileSync(w, wt.replace(/\r\n/g, '\n'), 'latin1');
    else if (bCrlf && !wCrlf) fs.writeFileSync(w, wt.replace(/\n/g, '\r\n'), 'latin1');
  }
}
function cmdEval(k, o) {
  if (!k || !o.idea || !o.tag) die('usage: eval KERNEL --idea "TEXT" --tag TAG --worker NAME');
  if (!C.tags.includes(o.tag)) die(`tag must be one of ${C.tags.join(', ')}`);
  return withKernel(k, `eval ${k}`, () => {
    const d = kdir(k);
    const st = loadState(k);
    if (!st || st.needsReinit) die(`${k} has no current loop: claim it`);
    if (st.landed || st.done) die(`${k} is finished (${st.landed ? 'landed' : st.done.reason}); claim another kernel`);
    if (st.landing) die(`${k} is being landed; run finish`);
    checkOwner(k, o.worker);
    const stop = stopCheck(st);
    if (stop.stop && !o.force) die(`the loop has stopped (${stop.why}): run finish ${k}`);
    heartbeat(k);
    const work = path.join(d, 'work'), best = path.join(d, 'best-src');
    const n = st.attempts + 1;
    const ap = path.join(d, 'attempts', pad(n));
    const rec = { ts: now(), kernel: k, attempt: n, idea: String(o.idea).slice(0, 500), tag: o.tag, worker: o.worker || null, scenario: st.scenario };
    const t0 = Date.now();
    const revert = () => {
      copyDir(best, path.join(work, 'kernels'));
      copyShared(path.join(d, 'core-src'), work);
    };
    // decision: counted attempts end here; 'noisy' and 'busy' are not attempts and leave the edit in place
    const finish = (decision, extra = {}) => {
      Object.assign(rec, extra, { decision, wall_s: +((Date.now() - t0) / 1000).toFixed(1) });
      if (decision === 'noisy' || decision === 'busy') {
        appendLedger(k, rec);
        console.log(`#${n} ${decision.toUpperCase()}: ${rec.reason}. The edit is still in the work copy: run the same eval again.`);
        fs.rmSync(ap + '.hsaco', { force: true });
        return;
      }
      if (decision === 'accepted') {
        copyDir(path.join(stage, 'kernels'), path.join(d, 'best-src.next'));
        fs.copyFileSync(ap + '.hsaco', path.join(d, 'best.hsaco.next'));
        writeJson(path.join(d, 'promote.json'), { attempt: n });
        st.speedup *= rec.ref_us / rec.cand_us;
        st.best = { attempt: n, us: rec.cand_us, res: rec.res };
        st.lastAccept = n;
        st.accepts.push({ attempt: n, gainPct: +rec.delta_pct.toFixed(2) });
      } else {
        revert();
        fs.rmSync(ap + '.hsaco', { force: true });
      }
      /* the loop stops; the gains kept before this still land through finish */
      if (decision === 'nondeterministic') st.stopped = 'the best version turned out not deterministic on its capture';
      for (const f of fs.readdirSync(path.join(d, 'attempts'))) if (f.startsWith(pad(n) + '.') && f.endsWith('.hsaco') && f !== pad(n) + '.hsaco') {
        const v = f.slice(pad(n).length + 1, -'.hsaco'.length); /* a variant built for this attempt: kept as the new best's, else dropped */
        if (decision === 'accepted') { fs.mkdirSync(path.join(d, 'variants'), { recursive: true }); fs.renameSync(path.join(d, 'attempts', f), path.join(d, 'variants', `${v}-${n}.hsaco`)); }
        else fs.rmSync(path.join(d, 'attempts', f), { force: true });
      }
      st.attempts = n;
      delete st.verified;
      rec.best_us = st.best.us;
      rec.speedup = +st.speedup.toFixed(4);
      saveState(k, st);
      appendLedger(k, rec);
      loadState(k); /* completes the promotion */
      const s = stopCheck(st);
      const dl = rec.delta_pct != null ? ` ${rec.delta_pct >= 0 ? '+' : ''}${rec.delta_pct.toFixed(2)}%` : '';
      console.log(`#${n} ${decision.toUpperCase()}${dl}  ${rec.reason || ''}`.trim());
      console.log(`best ${st.best.us.toFixed(2)} us (attempt ${st.best.attempt}), total x${st.speedup.toFixed(3)} vs baseline`);
      console.log(s.stop ? `STOP: ${s.why}; run finish ${k}` : `continue: ${s.why}`);
    };
    // only kernel sources may change: core/ and the generated headers are shared with the whole engine
    if ([...SHARED, 'build'].some((s) => !sameTree(path.join(work, s), path.join(d, 'core-src', s))))
      return finish('out_of_scope', { reason: 'the edit changed the working copy outside kernels/ (core/, prof/ or build/); only kernels/ may change (reset)' });
    keepLineEndings(best, path.join(work, 'kernels'));
    // the attempt is this snapshot of the work copy: it is what gets built, measured, kept and diffed, so an edit made
    // while the eval waits for the GPU cannot slip into the best version unmeasured
    const stage = path.join(d, 'stage');
    copyDir(path.join(work, 'kernels'), path.join(stage, 'kernels'));
    copyShared(path.join(d, 'core-src'), stage);
    const diff = run('git', ['diff', '--no-index', '--no-color', '--', best, path.join(stage, 'kernels')]).stdout;
    if (!diff.trim()) die('no change in work/kernels against the best version');
    rec.files = [...new Set([...diff.matchAll(/^\+\+\+ (?:"?b\/)?(.*?)"?$/gm)].map((m) => path.basename(m[1].trim())).filter((f) => f !== '/dev/null'))];
    fs.writeFileSync(ap + '.diff', diff);
    rec.diff = path.basename(ap) + '.diff';
    console.log(`#${n}: building ${UNITS[st.unit].hsaco} with ${rec.files.join(', ')} changed ...`);
    const c = compile(stage, st.unit, ap + '.hsaco');
    if (!c.ok) return finish('compile_fail', { reason: (c.log.split('\n').filter((l) => /error/.test(l)).slice(0, 6).join(' | ') || c.log.slice(-400)).slice(0, 600) });
    const res = resources(ap + '.hsaco', k);
    rec.res = res;
    if (!res) return finish('compile_fail', { reason: `the build has no kernel ${k}` });
    if (res.vspill > 0 || res.sspill > 0 || res.priv > 0) return finish('spill', { reason: `VGPR ${res.vgpr}, spills ${res.vspill}/${res.sspill}, scratch ${res.priv} B` });
    if (res.waves < st.best.res.waves) rec.note = `occupancy ${st.best.res.waves} -> ${res.waves} waves/SIMD (VGPR ${st.best.res.vgpr} -> ${res.vgpr})`;
    // Other kernels the edit changed (a shared helper or header), in this code object and in every other one built from
    // a touched file (the q8/q4 KV builds, the megakernels, other units): none may spill; those with a capture in this
    // scenario must stay exact and not slower; the rest are noted.
    const changedIn = (baseH, candH) => { const a = codeHashes(baseH), b = codeHashes(candH); return Object.keys(b).filter((s) => a[s] && a[s] !== b[s] && !s.endsWith('.kd')); };
    const collat = []; /* { kernel, variant, base, cand } */
    const noted = [];
    for (const s of changedIn(path.join(d, 'best.hsaco'), ap + '.hsaco')) if (s !== k) collat.push({ kernel: s, variant: st.unit, base: path.join(d, 'best.hsaco'), cand: ap + '.hsaco' });
    const depsFile = path.join(d, 'deps.json');
    let depMap = readJson(depsFile, null);
    if (!depMap || depMap.attempt !== st.best.attempt) {
      depMap = { attempt: st.best.attempt, map: {} };
      for (const v of VARIANTS) depMap.map[v.name] = deps(stage, v);
      writeJson(depsFile, depMap);
    }
    for (const v of VARIANTS) {
      if (v.name === st.unit) continue;
      const dl = depMap.map[v.name];
      if (dl && !rec.files.some((f) => dl.includes(f))) continue; /* does not read a touched file (unknown: build it) */
      const candV = `${ap}.${v.name}.hsaco`, baseV = path.join(d, 'variants', `${v.name}-${st.best.attempt}.hsaco`);
      console.log(`#${n}: ${rec.files.join(', ')} ${dl ? 'is also built into' : 'may also be built into'} ${v.name}: building it ...`);
      /* the best version's build of this variant first: if even that does not build here, the problem is the working
       * copy's environment, not this edit, so the variant is noted and skipped rather than charged to the attempt */
      if (!fs.existsSync(baseV)) {
        const root = path.join(d, 'bestroot');
        copyDir(best, path.join(root, 'kernels')); copyShared(path.join(d, 'core-src'), root);
        fs.mkdirSync(path.dirname(baseV), { recursive: true });
        if (!compile(root, v, baseV).ok) { noted.push(`${v.name} (does not build from the best version either: not compared)`); continue; }
      }
      const cv = compile(stage, v, candV);
      if (!cv.ok) return finish('compile_fail', { reason: `${v.name}: ` + (cv.log.split('\n').filter((l) => /error/.test(l)).slice(0, 4).join(' | ') || cv.log.slice(-300)).slice(0, 500) });
      for (const s of changedIn(baseV, candV)) {
        const rc = resources(candV, s), rb = resources(baseV, s);
        if (rc && (rc.vspill > (rb?.vspill || 0) || rc.sspill > (rb?.sspill || 0) || rc.priv > (rb?.priv || 0)))
          return finish('spill', { reason: `${v.name}: ${s} now spills (VGPR ${rc.vgpr}, spills ${rc.vspill}/${rc.sspill}, scratch ${rc.priv} B)` });
        if (v.unit) collat.push({ kernel: s, variant: v.name, base: baseV, cand: candV });
        else noted.push(`${v.name}:${s}`);
      }
    }
    const otherCaps = collat.filter((c) => fs.existsSync(capPath(st.scenario, c.kernel)));
    for (const c of collat) if (!otherCaps.includes(c)) noted.push(`${c.variant}:${c.kernel}`);
    rec.others = [...otherCaps.map((c) => `${c.variant}:${c.kernel}`), ...noted];
    try {
      return withGpu(`eval ${k} #${n}`, () => {
        const bad = (r) => {
          if (r.kind === 'differs') return finish('not_exact', { reason: r.why || 'output differs' });
          if (r.kind === 'nondet') return finish('nondeterministic', { reason: 'the best version is not deterministic on this capture' });
          if (r.kind === 'replay') return finish('error', { reason: 'the replay no longer matches the capture (restart the loop: rsi reopen)' });
          if (r.kind === 'fault') return finish('fault', { reason: 'GPU fault or failed launch: ' + (r.out.match(/HIP error[^\n]*/) || [''])[0] });
          if (r.kind === 'hang') return finish('hang', { reason: `kbench ran over ${C.kbench.timeoutSec} s` });
          if (r.kind === 'notiming') return finish('noisy', { reason: 'the GPU timer gave no valid times' });
          return finish('error', { reason: 'kbench: ' + r.out.slice(-300) });
        };
        let r1 = kbench(loopCap(k, st.scenario), path.join(d, 'best.hsaco'), ap + '.hsaco');
        if (r1.kind !== 'ok') return bad(r1);
        // the reference should time like the loop's best: far off means the GPU was shared or throttled meanwhile
        const off = (r) => Math.abs(r.ref[0] - st.best.us) / st.best.us * 100;
        if (off(r1) > C.accept.noiseGuardPct) {
          const first = r1.ref[0];
          sleep(20000);
          r1 = kbench(loopCap(k, st.scenario), path.join(d, 'best.hsaco'), ap + '.hsaco');
          if (r1.kind !== 'ok') return bad(r1);
          if (off(r1) > C.accept.noiseGuardPct) {
            /* the same level twice: the machine changed (clocks, power plan, driver), not a passing load; it is the
             * new level of the best version, and the attempt is judged against it as usual */
            if (Math.abs(r1.ref[0] - first) / first * 100 <= 1) {
              appendLedger(k, { ts: now(), kernel: k, attempt: st.attempts, decision: 'rebaseline', reason: `the best version now times ${r1.ref[0].toFixed(2)} us, not ${st.best.us.toFixed(2)} us`, best_us: r1.ref[0] });
              console.log(`the best version now times ${r1.ref[0].toFixed(2)} us steadily (was ${st.best.us.toFixed(2)} us): measuring against that`);
              st.best.us = r1.ref[0];
              saveState(k, st);
            } else return finish('noisy', { reason: `the unchanged kernel timed ${first.toFixed(2)} then ${r1.ref[0].toFixed(2)} us against its usual ${st.best.us.toFixed(2)} us (another GPU load?)` });
          }
        }
        Object.assign(rec, { ref_us: r1.ref[0], cand_us: r1.cand[0], cand_min_us: r1.cand[1], delta_pct: +r1.deltaPct.toFixed(3) });
        const g = C.accept.minGainPct;
        if (r1.ref[0] / r1.cand[0] >= C.accept.suspectSpeedup) return finish('suspect', { reason: `x${(r1.ref[0] / r1.cand[0]).toFixed(2)} in one step: held back (check it does all its work)` });
        if (r1.deltaPct > -g) return finish(r1.deltaPct > g ? 'slower' : 'no_gain', { reason: `needs <= -${g}%` });
        const r2 = kbench(loopCap(k, st.scenario), path.join(d, 'best.hsaco'), ap + '.hsaco');
        if (r2.kind !== 'ok') return bad(r2);
        const avg = (r1.deltaPct + r2.deltaPct) / 2;
        rec.confirm_pct = +r2.deltaPct.toFixed(3);
        rec.ref_us = (r1.ref[0] + r2.ref[0]) / 2, rec.cand_us = (r1.cand[0] + r2.cand[0]) / 2, rec.delta_pct = +avg.toFixed(3);
        if (r2.deltaPct > -g / 2 || avg > -g) return finish('not_confirmed', { reason: `second run ${r2.deltaPct.toFixed(2)}%` });
        rec.holdouts = [];
        for (const h of st.holdouts) {
          const rh = kbench(loopCap(k, h), path.join(d, 'best.hsaco'), ap + '.hsaco');
          if (rh.kind === 'nondet' || rh.kind === 'replay') { rec.holdouts.push({ scenario: h, skipped: rh.kind }); continue; }
          if (rh.kind !== 'ok') return rh.kind === 'differs' ? finish('holdout_fail', { reason: `${h}: ${rh.why}` }) : bad(rh);
          rec.holdouts.push({ scenario: h, delta_pct: +rh.deltaPct.toFixed(3) });
          if (rh.deltaPct > C.accept.holdoutTolPct) return finish('holdout_fail', { reason: `${h}: ${rh.deltaPct.toFixed(2)}% slower` });
        }
        rec.collateral = [];
        for (const c of otherCaps) {
          const cap = capPath(st.scenario, c.kernel);
          if (!fs.existsSync(cap)) { rec.collateral.push({ kernel: c.kernel, skipped: 'capture being refreshed' }); continue; }
          const ro = kbench(cap, c.base, c.cand, { rounds: 7 });
          if (['nondet', 'replay', 'error', 'notiming'].includes(ro.kind)) { rec.collateral.push({ kernel: c.kernel, skipped: ro.kind }); continue; }
          if (ro.kind !== 'ok') return finish('collateral', { reason: `${c.variant}:${c.kernel} (also changed by this edit): ${ro.kind === 'differs' ? ro.why : ro.kind}` });
          rec.collateral.push({ kernel: c.kernel, variant: c.variant, delta_pct: +ro.deltaPct.toFixed(3) });
          if (ro.deltaPct > C.accept.holdoutTolPct) return finish('collateral', { reason: `${c.variant}:${c.kernel} (also changed by this edit) is ${ro.deltaPct.toFixed(2)}% slower` });
        }
        return finish('accepted', { reason: [rec.note, noted.length ? `also changes ${noted.join(', ')} (no capture in ${st.scenario}: not timed)` : ''].filter(Boolean).join('; ') });
      });
    } catch (e) {
      if (e instanceof Fail && e.kind === 'busy') return finish('busy', { reason: e.message.split('\n')[0] });
      throw e;
    }
  });
}

// ---------------------------------------------------------------- status, isa
function cmdStatus(k, o) {
  const st = loadState(k);
  if (!st) die(`no loop for ${k}`);
  const led = ledger(k).filter((r) => r.attempt > 0 && r.tag && r.decision !== 'noisy' && r.decision !== 'busy');
  const since0 = st.baseline?.attempt || 0;
  const s = st.landed ? { stop: true, why: 'landed' } : st.done ? { stop: true, why: 'done: ' + st.done.reason } : st.best ? stopCheck(st) : { stop: true, why: 'not set up' };
  const byTag = {};
  for (const t of C.tags) byTag[t] = { tried: 0, accepted: 0 };
  for (const r of led) if (byTag[r.tag]) { byTag[r.tag].tried++; if (r.decision === 'accepted') byTag[r.tag].accepted++; }
  const recentTags = led.slice(-3).map((r) => r.tag);
  const fresh = C.tags.filter((t) => !recentTags.includes(t)).sort((a, b) => byTag[a].tried - byTag[b].tried);
  const out = { kernel: k, unit: st.unit, scenario: st.scenario, holdouts: st.holdouts, baseline_us: st.baseline?.us, noise_pct: st.baseline?.noisePct,
    best: st.best, speedup: +(st.speedup || 1).toFixed(4), attempts: st.attempts, n_accepted: (st.accepts || []).length, stop: s, verified: st.verified || null,
    landed: st.landed || null, done: st.done || null, landing: st.landing || null, claim: claimOf(k),
    byTag, suggestTag: led.length >= 3 && led.slice(-3).every((r) => r.decision !== 'accepted') ? fresh[0] : null,
    accepted: led.filter((r) => r.decision === 'accepted' && r.attempt > since0).map((r) => ({ attempt: r.attempt, idea: r.idea, delta_pct: r.delta_pct })),
    acceptedBeforeRestart: led.filter((r) => r.decision === 'accepted' && r.attempt <= since0).map((r) => ({ attempt: r.attempt, idea: r.idea, delta_pct: r.delta_pct, diff: r.diff })),
    failed: led.filter((r) => r.decision !== 'accepted').map((r) => ({ attempt: r.attempt, tag: r.tag, decision: r.decision, idea: r.idea, delta_pct: r.delta_pct ?? null, reason: r.reason || '' })) };
  if (o.json) { console.log(JSON.stringify(out, null, 2)); return; }
  console.log(`${k} (${st.unit}) on ${st.scenario}${st.holdouts?.length ? ', holdouts ' + st.holdouts.join(', ') : ''} [${loopState(k)}]`);
  if (st.best) console.log(`baseline ${st.baseline.us.toFixed(2)} us, best ${st.best.us.toFixed(2)} us (attempt ${st.best.attempt}, VGPR ${st.best.res.vgpr}), total x${st.speedup.toFixed(3)}, noise ~${st.baseline.noisePct}%`);
  console.log(`${st.attempts} attempts, ${(st.accepts || []).length} accepted; ${s.stop ? 'STOP: ' : 'continue: '}${s.why}`);
  if (out.accepted.length) { console.log('\nkept (in order):'); for (const a of out.accepted) console.log(`  #${a.attempt} ${a.delta_pct.toFixed(2)}%  ${a.idea}`); }
  if (out.acceptedBeforeRestart.length) {
    const arch = (st.history || []).filter((h) => h.archive).map((h) => h.archive);
    console.log(`
kept before the loop restarted on newer engine code at attempt ${since0} (not in the current best; re-apply them first; their diffs are in ${arch.length ? arch.join(", ") + "/attempts" : "attempts/"}):`);
    for (const a of out.acceptedBeforeRestart) console.log(`  #${a.attempt} ${a.delta_pct.toFixed(2)}%  ${a.idea}  (${a.diff})`);
  }
  if (out.failed.length) { console.log('\ntried and dropped (do not repeat):'); for (const f of out.failed) console.log(`  #${f.attempt} [${f.tag}] ${f.decision}${f.delta_pct != null ? ' ' + f.delta_pct.toFixed(2) + '%' : ''}  ${f.idea}${f.reason ? '  -- ' + f.reason.slice(0, 160) : ''}`); }
  console.log('\nby tag: ' + C.tags.map((t) => `${t} ${byTag[t].accepted}/${byTag[t].tried}`).join(', '));
  if (out.suggestTag) console.log(`three misses in a row: switch to a different kind of idea, e.g. "${out.suggestTag}"`);
}
function cmdIsa(k, o) {
  const d = kdir(k);
  const h = o.attempt ? path.join(d, 'attempts', pad(+o.attempt) + '.hsaco') : path.join(d, 'best.hsaco');
  if (!fs.existsSync(h)) die(`no build at ${h} (only accepted attempts and the best keep a build)`);
  const res = resources(h, k);
  if (!res) die(`${h} has no kernel ${k}`);
  console.log(`${k}: VGPR ${res.vgpr} (${res.waves} waves/SIMD), SGPR ${res.sgpr}, LDS ${res.lds} B, spills ${res.vspill}/${res.sspill}, scratch ${res.priv}`);
  const r = run('python', [path.join(ENGINE, 'tools', 'kernel_isa.py'), h, k, ...(o.dump ? ['--dump', String(o.dump)] : [])], { env: { ...process.env, LLVM_BIN: C.llvmBin } });
  console.log(o.dump ? r.out : r.out.split('\n').slice(0, 12).join('\n'));
}

// ---------------------------------------------------------------- verify
// baseline against best on every capture with more rounds: identical everywhere, the primary at least half the gain
// threshold faster, no capture over the holdout tolerance slower. Timing that misses is measured once more.
function verifyOnce(k, st) {
  const d = kdir(k);
  return withGpu(`verify ${k}`, () => [st.scenario, ...st.holdouts].map((s, i) => {
    const r = kbench(loopCap(k, s), path.join(d, 'baseline.hsaco'), path.join(d, 'best.hsaco'), { rounds: 21 });
    if (r.kind === 'nondet' || r.kind === 'replay') return { scenario: s, skipped: r.kind, ok: i > 0 };
    if (r.kind !== 'ok') return { scenario: s, kind: r.kind, why: r.why, ok: false, exact: r.kind !== 'differs' };
    const ok = i === 0 ? r.deltaPct <= -C.accept.minGainPct / 2 : r.deltaPct <= C.accept.holdoutTolPct;
    return { scenario: s, baseline_us: r.ref[0], best_us: r.cand[0], delta_pct: +r.deltaPct.toFixed(3), ok, exact: true };
  }));
}
// best.diff: the loop's own changes, base-src -> best-src, with repository-relative paths and the blob ids of the
// base, so git apply --3way merges it onto a main that has moved on (never a diff against the live engine/kernels,
// which would carry the reverse of everything landed since the loop started). Returns the changed files.
function makePatch(k) {
  const d = kdir(k), baseSrc = path.join(d, 'base-src');
  if (!fs.existsSync(baseSrc)) die(`${k}'s loop predates base snapshots: rsi reopen ${k}`);
  const files = [...new Set(run('git', ['diff', '--no-index', '--no-color', '--name-only', '--', baseSrc, path.join(d, 'best-src')]).stdout
    .split('\n').filter(Boolean).map((f) => path.basename(f.trim().replace(/^"|"$/g, ''))))];
  const diff = files.map((f) => {
    const rel = 'engine/kernels/' + f;
    const a = path.join(baseSrc, f), b = path.join(d, 'best-src', f);
    const body = run('git', ['diff', '--no-index', '--no-color', '--full-index', '--', fs.existsSync(a) ? a : '/dev/null', fs.existsSync(b) ? b : '/dev/null']).stdout;
    return body.replace(/^diff --git .*$/m, `diff --git a/${rel} b/${rel}`).replace(/^--- (?!\/dev\/null).*$/m, `--- a/${rel}`).replace(/^\+\+\+ (?!\/dev\/null).*$/m, `+++ b/${rel}`);
  }).join('');
  fs.writeFileSync(path.join(d, 'best.diff'), diff);
  return files;
}
function cmdVerify(k) { return withKernel(k, `verify ${k}`, () => verifyLocked(k)); }
function verifyLocked(k) {
  const d = kdir(k);
  const st = loadState(k);
  if (!st) die(`no loop for ${k}`);
  if (st.best.attempt === (st.baseline.attempt || 0)) die('nothing accepted yet');
  let results = verifyOnce(k, st);
  if (!results.every((r) => r.ok) && results.every((r) => r.exact !== false)) { console.log('verify missed on timing: measuring once more ...'); results = verifyOnce(k, st); }
  const ok = results.every((r) => r.ok);
  const files = makePatch(k);
  st.verified = { ts: now(), ok, results, files, attempt: st.best.attempt };
  saveState(k, st);
  appendLedger(k, { ts: now(), kernel: k, attempt: st.attempts, decision: ok ? 'verified' : 'verify_fail', results, best_us: st.best.us, speedup: +st.speedup.toFixed(4) });
  for (const r of results) console.log(`${r.scenario}: ${r.skipped ? 'skipped (' + r.skipped + ')' : r.kind ? r.kind + ' ' + (r.why || '') : `baseline ${r.baseline_us.toFixed(2)} us -> best ${r.best_us.toFixed(2)} us (${r.delta_pct.toFixed(2)}%)`}${r.ok ? '' : '  <- fails'}`);
  console.log(ok ? `verified (patch: ${slash(path.join(d, 'best.diff'))}, changes ${files.join(', ')})` : 'NOT verified: it will not land.');
  return ok;
}

// ---------------------------------------------------------------- landing
// the reference output hashes and speeds: those of the latest promoted round in the repository's ledger
function refShas() {
  const led = readJson(path.join(REPO, 'docs', '40round-ledger.json'));
  const r = led && [...led.rounds].reverse().find((x) => x.status === 'promoted' && x.parity?.status === 'EXACT' && x.parity?.shas);
  if (!r) die('no promoted EXACT round with output hashes in docs/40round-ledger.json', 'blocked');
  return { round: r.round, shas: r.parity.shas, metrics: r.metrics || {} };
}
// whole-model speeds of a gate run below gateMinSpeedPct of the reference round's (one pass is noisy: a few % normal)
function slowStages(g) {
  const ref = refShas().metrics, pct = C.gateMinSpeedPct ?? 93;
  return ['t1_tok_s', 'ds5_233_tok_s'].filter((m) => g.metrics[m] != null && ref[m] && g.metrics[m] < (ref[m] * pct) / 100)
    .map((m) => `${m} ${g.metrics[m]} vs ${ref[m]}`);
}
// the four exact-output stages of the benchmark harness on engine/build: { ok, missing, checks, metrics }. A stage
// without a hash is run again once; so is a mismatch (a one-off would otherwise write a good kernel off).
function gate() {
  const ref = refShas();
  const once = () => {
    const r = py([C.harness, '--exe', path.join(ENGINE, 'build', 'ie-run.exe'), '--hsaco', path.join(ENGINE, 'build', 'ie_kernels.hsaco'),
      '--model', C.model, '--dspark', C.dspark], { env: engineEnv({}), timeout: 2700000, orphans: ['ie-run.exe', 'ie-serve.exe'] });
    const pats = { t1_sha: /Plain T=1.*sha=([0-9a-f]{16})/, ds5_233_sha: /DS5 233.*sha=([0-9a-f]{16})/, ds5_4k_sha: /DS5 4K.*sha=([0-9a-f]{16})/, ds5_16k_sha: /DS5 16K.*sha=([0-9a-f]{16})/ };
    const checks = {};
    for (const [key, re] of Object.entries(pats)) {
      const m = r.out.match(re);
      checks[key] = !m ? 'MISSING' : m[1] === (ref.shas[key] || '').slice(0, 16) ? 'MATCH' : 'MISMATCH';
    }
    const num = (re) => { const m = r.out.match(re); return m ? +m[1] : null; };
    const metrics = { t1_tok_s: num(/Plain T=1\s*:\s*([\d.]+) tok\/s/), ds5_233_tok_s: num(/DS5 233\s*:\s*([\d.]+) tok\/s/),
      ds5_4k_tok_s: num(/DS5 4K\s*:\s*decode\s*([\d.]+)/), ds5_16k_tok_s: num(/DS5 16K\s*:\s*decode\s*([\d.]+)/) };
    const v = Object.values(checks);
    return { ok: v.every((x) => x === 'MATCH'), mismatch: v.includes('MISMATCH'), missing: v.includes('MISSING'), checks, metrics, refRound: ref.round, log: r.out.slice(-3000) };
  };
  let g = once();
  if (!g.ok) {
    console.log(g.mismatch ? 'the gate reports a hash mismatch: running it once more to rule out a one-off ...' : 'the gate produced no hashes for some stages: running it once more ...');
    sleep(30000);
    const g2 = once();
    if (g2.ok || !g.mismatch) g = g2;
  }
  return g;
}
// what keeps any landing from proceeding now, before a build is spent on it
function landingHeld() {
  if (C.labFaultMarker && fs.existsSync(C.labFaultMarker)) return [`the GPU fault marker ${C.labFaultMarker} is present (a person must check the card)`, 24 * 60];
  const gb = readJson(path.join(RUNS, '.gate-broken'), null);
  if (gb && ago(gb.since) < 24 * 3600000) return [`main did not match the reference hashes at ${gb.since} (fix main or docs/40round-ledger.json, then delete .gate-broken)`, 24 * 60];
  if (gh(['auth', 'status'], { timeout: 60000 }).code !== 0) return ['gh is not logged in', 60];
  const busy = gpuBusy(['kbench.exe']); /* other workers' evals queue behind the landing */
  if (busy) return [`the GPU is in use: ${busy}`, 15];
  return null;
}
const onMain = () => branchNow() === 'main';
// from a landing branch with no commit back to main (the tree was clean before, so this drops only the patch)
function backToMain(br) {
  git(['reset', '-q', '--hard']);
  git(['checkout', '-q', 'main']);
  if (br && br !== 'main') git(['branch', '-q', '-D', br]);
  fs.rmSync(LANDING(), { force: true });
}
function block(k, st, why, retryMin = C.blockedRetryMinutes) {
  st.blockedUntil = new Date(Date.now() + retryMin * 60000).toISOString();
  st.blockedWhy = why;
  st.blockCount = (st.blockCount || 0) + 1; /* after maxBlocks in a row the kernel waits for a person */
  saveState(k, st);
  appendLedger(k, { ts: now(), kernel: k, attempt: st.attempts, decision: 'land_blocked', reason: why });
  console.log(`cannot land ${k} now: ${why}. It stays verified; a claim retries it after ${st.blockedUntil.slice(11, 16)}.`);
  return 'blocked';
}
function markDone(k, st, reason, extra = {}) {
  st.done = { reason, ts: now(), ...extra };
  delete st.landing;
  saveState(k, st);
  appendLedger(k, { ts: now(), kernel: k, attempt: st.attempts, decision: reason === 'landed' ? 'landed' : 'done', reason, ...extra });
  prune(k);
  return 'done';
}
// apply the verified best.diff on an up-to-date main, rebuild everything, check the gain on the current code, gate,
// commit; then push, PR and merge, re-gating from scratch if main moved meanwhile. Each published step is recorded in
// st.landing so a landing interrupted anywhere resumes. Returns 'landed', 'blocked', 'restart' or 'done'.
// Locks: the kernel's, then the repository's for the whole landing; the GPU's only around the GPU steps (the gain check
// and the gate), so other workers' evals run during the builds, pushes and merges.
function cmdLand(k) {
  return withKernel(k, `land ${k}`, () => withRepo(`land ${k}`, () => {
    let st = loadState(k);
    if (!st) die(`no loop for ${k}`);
    if (st.landed) return 'landed';
    if (!st.landing && (!st.verified?.ok || st.verified.attempt !== st.best.attempt)) die(`${k} has no verified best to land: run finish`);
    const rec = recoverRepo();
    if (rec) console.log(rec);
    if (!onMain()) return block(k, st, `the repository is on ${branchNow()}, not main`);
    if (dirtyAll()) return block(k, st, 'the repository has uncommitted changes');
    const held0 = landingHeld();
    if (held0) return block(k, st, held0[0], held0[1]);
    for (let attempt = 0; attempt < C.maxLandTries; attempt++) {
      st = loadState(k);
      if (!st.landing) {
        const r = landFresh(k, st);
        if (r !== 'committed') return r;
        st = loadState(k);
      }
      const p = publish(k, st);
      if (p !== 'main-moved') return p;
      console.log('main moved while landing: landing again on the new main ...');
    }
    return block(k, loadState(k), `main kept moving during ${C.maxLandTries} landings`);
  }));
}
function landFresh(k, st) {
  const d = kdir(k);
  const unit = st.unit, unitFile = path.join(ENGINE, 'build', UNITS[unit].hsaco);
  if (git(['fetch', '-q', 'origin'], { timeout: 300000 }).code !== 0) return block(k, st, 'git fetch failed (network?)');
  const pl = git(['pull', '-q', '--ff-only', 'origin', 'main'], { timeout: 300000 });
  if (pl.code !== 0) return block(k, st, 'local main cannot fast-forward to origin/main (diverged?): ' + pl.out.trim().split('\n').pop(), 24 * 60);
  const base = mustGit(['rev-parse', 'HEAD'], 'base');
  if (buildStale() || fs.existsSync(BROKEN())) {
    console.log('engine/build is behind main: rebuilding it first ...');
    try { buildAll('main before landing'); } catch (e) { return block(k, st, 'engine/build does not build on main: ' + e.message.split('\n')[0], 120); }
  }
  fs.copyFileSync(unitFile, path.join(d, 'main.hsaco'));
  const br = `rsi/${k.replace(/^ie_inkling_|^ie_/, '')}-${now().slice(0, 19).replace(/[-:T]/g, '')}`;
  // the files the patch changes, from the patch itself: untracked files lying around the repository are no concern
  const staged = [...new Set([...fs.readFileSync(path.join(d, 'best.diff'), 'utf8').matchAll(/^\+\+\+ b\/(\S+)$/gm)].map((m) => m[1]))];
  if (!staged.length || staged.some((f) => !f.startsWith('engine/kernels/'))) return markDone(k, st, 'land_error', { stage: 'apply', why: `the patch changes ${staged.join(', ') || 'nothing'}` });
  writeJson(LANDING(), { branch: br, kernel: k, since: now() }); /* recoverRepo touches this branch only */
  mustGit(['checkout', '-q', '-b', br], 'branch');
  const ap = git(['apply', '--3way', '--whitespace=nowarn', path.join(d, 'best.diff')]);
  const extra = git(['status', '--porcelain', '--untracked-files=no']).out.trim().split('\n').filter(Boolean).map((l) => l.slice(3).trim()).filter((f) => !staged.includes(f));
  if (ap.code !== 0 || extra.length) {
    backToMain(br);
    st.needsReinit = true;
    saveState(k, st);
    appendLedger(k, { ts: now(), kernel: k, attempt: st.attempts, decision: 'land_conflict', reason: (extra.length ? `the apply also changed ${extra.join(', ')}; ` : '') + ap.out.slice(-600) });
    console.log('the patch no longer applies to main (another change touched the same code): the loop restarts on the current code at the next claim, keeping its history.');
    return 'restart';
  }
  console.log('rebuilding engine/build with the patch (build-win.sh) ...');
  try { buildAll(`landing ${k}`); } catch (e) {
    backToMain(br);
    try { buildAll('main after a failed landing build'); } catch (e2) { return block(k, st, 'engine/build does not build on main either: ' + e2.message.split('\n')[0], 120); }
    st.needsReinit = true; saveState(k, st);
    appendLedger(k, { ts: now(), kernel: k, attempt: st.attempts, decision: 'land_conflict', reason: 'the patched engine does not build: ' + e.message.slice(0, 400) });
    console.log('the patched engine does not build on the current main: the loop restarts on the current code.');
    return 'restart';
  }
  const restore = (why) => { backToMain(br); try { buildAll(why); return null; } catch (e) { return block(k, st, 'engine/build does not build on main: ' + e.message.split('\n')[0], 120); } };
  const restart = (decision, why) => {
    const b = restore(`main after: ${why}`);
    if (b) return b;
    st.needsReinit = true; saveState(k, st);
    appendLedger(k, { ts: now(), kernel: k, attempt: st.attempts, decision, reason: why });
    console.log(`${why}: the loop restarts on the current code at the next claim, keeping its history.`);
    return 'restart';
  };
  // the gain on the current code: main's code object against the patched one, on the loop's capture
  const gn = C.landGainCheck === false ? { kind: 'skipped' } : withGpu(`land ${k}: gain`, () => kbench(loopCap(k, st.scenario), path.join(d, 'main.hsaco'), unitFile, { rounds: 15 }));
  if (gn.kind === 'ok' && gn.deltaPct > -C.accept.minGainPct / 2) return restart('land_conflict', `on the current main the change gains only ${gn.deltaPct.toFixed(2)}%`);
  if (gn.kind === 'differs') { const b = restore('main after an inexact landing'); return b || markDone(k, st, 'land_fail', { why: 'not bit-identical on the current main: ' + gn.why }); }
  if (gn.kind === 'fault' || gn.kind === 'hang') return restart('land_conflict', `the change ${gn.kind === 'fault' ? 'faults' : 'hangs'} on the current main`);
  if (gn.kind === 'notiming' || gn.kind === 'error') { const b = restore('main after an unmeasured landing'); return b || block(k, st, `the gain on main could not be measured (${gn.kind})`, 30); }
  /* 'replay' (main's code changed what this kernel reads: its capture no longer replays), 'nondet', 'skipped': the
   * gate below still checks the whole model */
  console.log('running the exact-output gate (benchmark harness, 4 stages) ...');
  let g = withGpu(`land ${k}: gate`, () => gate());
  if (g.ok && slowStages(g).length) {
    console.log(`whole-model speed below ${C.gateMinSpeedPct ?? 93}% of the reference (${slowStages(g).join(', ')}): running the gate once more ...`);
    g = withGpu(`land ${k}: gate`, () => gate());
    if (g.ok && slowStages(g).length) { const b = restore('main after a slow landing'); return b || block(k, st, `the whole model ran slower than the reference with the change (${slowStages(g).join(', ')})`, 120); }
  }
  if (!g.ok) {
    const b = restore('main after a failed gate');
    if (b) return b;
    if (!g.mismatch) return block(k, st, 'the gate gave no hashes for ' + Object.keys(g.checks).filter((x) => g.checks[x] === 'MISSING').join(', '));
    console.log('the gate failed: checking main itself against the reference ...');
    const gm = withGpu(`land ${k}: gate on main`, () => gate());
    if (!gm.ok) {
      if (gm.mismatch) fs.writeFileSync(path.join(RUNS, '.gate-broken'), JSON.stringify({ since: now(), checks: gm.checks }));
      return block(k, st, gm.mismatch ? `main itself does not match the R${gm.refRound} reference hashes ${JSON.stringify(gm.checks)}: update docs/40round-ledger.json or fix main` : 'the gate on main gave no hashes', gm.mismatch ? 24 * 60 : 30);
    }
    fs.rmSync(path.join(RUNS, '.gate-broken'), { force: true });
    return markDone(k, st, 'land_fail', { checks: g.checks, ref_round: g.refRound });
  }
  fs.rmSync(path.join(RUNS, '.gate-broken'), { force: true });
  const pct = (100 * (1 - 1 / st.speedup)).toFixed(1);
  const kept = ledger(k).filter((r) => r.decision === 'accepted' && r.attempt > (st.baseline.attempt || 0));
  git(['add', '--', ...staged]);
  mustGit(['commit', '-q', '-m', `perf: ${k} ${pct}% faster`, '-m', `Found by the kernel-rsi loop: ${kept.map((r) => r.idea).join('; ')}. Bit-identical; output hashes match R${g.refRound}.`], 'commit');
  st.landing = { branch: br, base, stage: 'committed', gate: { checks: g.checks, metrics: g.metrics, refRound: g.refRound }, gainOnMain: gn.kind === 'ok' ? +gn.deltaPct.toFixed(3) : null, pct, kept: kept.map((r) => ({ idea: r.idea, delta_pct: r.delta_pct })), tries: 0 };
  saveState(k, st);
  builtHere = br; /* engine/build holds this branch's build; the repository stays on it until the merge */
  writeJson(BUILT(), { sha: headSha(), ts: now() });
  st.landing.files = staged;
  saveState(k, st);
  return 'committed';
}
let builtHere = null;
// every way out of publish other than a merge goes back to main; engine/build then holds the patched build, so it is
// marked stale and the next preflight rebuilds it from main
function leaveBranch() {
  git(['checkout', '-q', 'main']);
  fs.writeFileSync(BROKEN(), JSON.stringify({ since: now(), why: 'a landing left engine/build with an unmerged change' }));
}
// commits between two points of main that could change what the gate measured: the engine, or the reference hashes
const engineMoved = (from, to) => git(['diff', '--name-only', from, to]).out.split('\n').map((f) => f.trim()).filter(Boolean)
  .filter((f) => f.startsWith('engine/') || f === 'docs/40round-ledger.json');
// a merge that landed on a main that moved between the last check and the merge: the merged tree is built and gated
// now; if it fails, the merge is reverted through a PR of its own
function gateMerged(k, st, L) {
  console.log('main moved just before the merge: building and gating the merged tree ...');
  buildAll(`merged ${L.pr}`);
  const g = withGpu(`land ${k}: gate after merge`, () => gate());
  if (g.ok) return null;
  const rb = `rsi/revert-${k.replace(/^ie_inkling_|^ie_/, '')}-${now().slice(0, 19).replace(/[-:T]/g, '')}`;
  git(['checkout', '-q', '-b', rb]);
  const rv = git(['revert', '--no-edit', 'HEAD']);
  const ok = rv.code === 0 && git(['push', '-q', '-u', 'origin', rb], { timeout: 300000 }).code === 0;
  let url = '';
  if (ok) {
    const pr = gh(['pr', 'create', '--base', 'main', '--head', rb, '--title', `revert: ${k} (fails the exact-output gate when merged with newer main)`, '--body', `The merge of ${L.pr} combined with commits that reached main in the same moment fails the exact-output gate (${JSON.stringify(g.checks)}). This reverts it.`], { cwd: REPO, timeout: 300000 });
    url = (pr.out.match(/https:\/\/\S+/) || [''])[0];
    if (pr.code === 0) gh(['pr', 'merge', url || rb, '--rebase', '--delete-branch'], { cwd: REPO, timeout: 600000 });
  }
  git(['checkout', '-q', 'main']); git(['pull', '-q', '--ff-only', 'origin', 'main'], { timeout: 300000 });
  fs.writeFileSync(BROKEN(), JSON.stringify({ since: now(), why: 'reverted a merge' }));
  return `the merged tree failed the gate (${JSON.stringify(g.checks)}); ${ok && url ? 'reverted by ' + url : 'REVERT IT BY HAND: ' + L.pr}`;
}
function publish(k, st) {
  const L = st.landing;
  const save = () => saveState(k, st);
  const blockOff = (why, min) => { leaveBranch(); return block(k, st, why, min); };
  // main moved: commits outside the engine and the reference (docs, tools) leave the gated result valid on the new base
  const moved = (om, when) => {
    if (om === L.base) return null;
    const touched = engineMoved(L.base, om);
    if (!touched.length) { console.log(`origin/main moved ${when} with no engine change: continuing on ${om.slice(0, 8)}`); L.base = om; save(); return null; }
    return abandon(`origin/main moved ${when} (${touched.slice(0, 3).join(', ')}${touched.length > 3 ? ', ...' : ''}): the gate must run on the merged tree`);
  };
  const abandon = (why) => { /* close what was published and land again from scratch */
    if (L.pr) gh(['pr', 'close', L.pr, '--delete-branch'], { cwd: REPO, timeout: 300000 });
    else if (L.stage !== 'committed') git(['push', '-q', 'origin', '--delete', L.branch], { timeout: 300000 });
    leaveBranch(); git(['branch', '-q', '-D', L.branch]);
    appendLedger(k, { ts: now(), kernel: k, attempt: st.attempts, decision: 'land_retry', reason: why });
    delete st.landing; save();
    return 'main-moved';
  };
  if (git(['rev-parse', '--verify', '-q', L.branch]).code !== 0 && L.stage === 'committed') { delete st.landing; save(); return 'main-moved'; }
  if (git(['fetch', '-q', 'origin'], { timeout: 300000 }).code !== 0) return blockOff('git fetch failed (network?)');
  const mv0 = moved(git(['rev-parse', 'origin/main']).out.trim(), 'since the gate');
  if (mv0) return mv0;
  if (L.stage === 'committed') {
    const p = git(['push', '-q', '-u', 'origin', L.branch], { timeout: 300000 });
    if (p.code !== 0) return blockOff('git push failed: ' + p.out.trim().split('\n').pop());
    L.stage = 'pushed'; save();
  }
  if (L.stage === 'pushed') {
    const ex = gh(['pr', 'view', L.branch, '--json', 'url,state', '-q', '.url+" "+.state'], { cwd: REPO, timeout: 120000 });
    if (ex.code === 0 && /OPEN/.test(ex.out)) L.pr = ex.out.trim().split(' ')[0];
    else {
      const body = path.join(kdir(k), 'pr-body.md');
      fs.writeFileSync(body, [`${k} runs ${L.pct}% faster (x${st.speedup.toFixed(3)}), measured cold on captured launches, with output bit-identical to the previous build.`, '',
        '**Kept changes:**', ...L.kept.map((r) => `- ${r.idea} (${r.delta_pct.toFixed(2)}%)`), '',
        '**Verification:**', ...st.verified.results.filter((r) => r.baseline_us).map((r) => `- ${r.scenario}: ${r.baseline_us.toFixed(2)} -> ${r.best_us.toFixed(2)} us (${r.delta_pct.toFixed(2)}%), bit-identical`),
        L.gainOnMain != null ? `- On main at ${L.base.slice(0, 8)}: ${L.gainOnMain.toFixed(2)}% against main's own build.` : '',
        `- Output hashes of the benchmark harness (t1_233, ds5_233, ds5_4k, ds5_16k) match R${L.gate.refRound}, on main at ${L.base.slice(0, 8)} with this change.`,
        `- Harness speeds (one pass): ${Object.entries(L.gate.metrics).filter(([, v]) => v != null).map(([m, v]) => `${m} ${v}`).join(', ')}.`].filter(Boolean).join('\n') + '\n');
      const pr = gh(['pr', 'create', '--base', 'main', '--head', L.branch, '--title', `perf: ${k} ${L.pct}% faster`, '--body-file', body], { cwd: REPO, timeout: 300000 });
      fs.rmSync(body, { force: true });
      if (pr.code !== 0) return blockOff('gh pr create failed: ' + pr.out.trim().split('\n').pop());
      L.pr = (pr.out.match(/https:\/\/\S+/) || [''])[0];
    }
    L.stage = 'pr'; save();
  }
  if (L.stage === 'pr') {
    /* main may have moved while the PR was opened: a rebase-merge would then combine the change with code the gate
     * never ran on */
    if (git(['fetch', '-q', 'origin'], { timeout: 300000 }).code !== 0) return blockOff('git fetch failed (network?)');
    const mv1 = moved(git(['rev-parse', 'origin/main']).out.trim(), 'before the merge');
    if (mv1) return mv1;
    const mg = gh(['pr', 'merge', L.pr || L.branch, '--rebase', '--delete-branch'], { cwd: REPO, timeout: 600000 });
    let state = '';
    for (let i = 0; i < 3 && !state; i++) { state = gh(['pr', 'view', L.pr || L.branch, '--json', 'state', '-q', '.state'], { cwd: REPO, timeout: 120000 }).out.trim(); if (!state) sleep(5000); }
    if (state !== 'MERGED' && git(['fetch', '-q', 'origin'], { timeout: 300000 }).code === 0) {
      /* the merge may have gone through while its status could not be read: main then holds the branch's files */
      const om = git(['rev-parse', 'origin/main']).out.trim();
      if (om !== L.base && L.files?.length && git(['diff', '--quiet', om, L.branch, '--', ...L.files]).code === 0) state = 'MERGED';
    }
    if (state !== 'MERGED') {
      L.tries = (L.tries || 0) + 1; save();
      if (/conflict|not mergeable/i.test(mg.out)) return abandon('the PR does not merge cleanly');
      leaveBranch();
      if (L.tries >= C.maxLandTries) return markDone(k, st, 'land_error', { stage: 'merge', pr: L.pr, why: mg.out.trim().split('\n').pop() });
      return block(k, st, 'gh pr merge failed: ' + mg.out.trim().split('\n').pop());
    }
    git(['checkout', '-q', 'main']);
    const pl = git(['pull', '-q', '--ff-only', 'origin', 'main'], { timeout: 300000 });
    git(['branch', '-q', '-D', L.branch]);
    fs.rmSync(LANDING(), { force: true });
    if (pl.code !== 0) fs.writeFileSync(BROKEN(), JSON.stringify({ since: now(), why: `merged ${L.pr} but main did not update locally` }));
    /* the merge commit's parent is the main it went onto: anything other than the gated base with engine changes in
     * between means the merged tree was never gated (main moved in the last seconds) */
    let reverted = null;
    if (pl.code === 0) {
      const parent = git(['rev-parse', 'HEAD^']).out.trim();
      if (parent !== L.base && engineMoved(L.base, parent).length) {
        try { reverted = gateMerged(k, st, L); } catch (e) { reverted = `the merged tree could not be gated: ${e.message.split('\n')[0]}; CHECK ${L.pr} BY HAND`; }
      }
    }
    if (reverted) {
      st.needsReinit = true; delete st.landing; save();
      appendLedger(k, { ts: now(), kernel: k, attempt: st.attempts, decision: 'land_fail', reason: reverted, pr: L.pr });
      console.log(reverted);
      return 'restart';
    }
    st.landed = { ts: now(), pr: L.pr, branch: L.branch, base: L.base, checks: L.gate.checks, metrics: L.gate.metrics, refRound: L.gate.refRound };
    delete st.landing; delete st.blockedUntil; delete st.blockedWhy; delete st.blockCount;
    save();
    appendLedger(k, { ts: now(), kernel: k, attempt: st.attempts, decision: 'landed', pr: L.pr, metrics: L.gate.metrics, speedup: +st.speedup.toFixed(4), best_us: st.best.us });
    prune(k);
    console.log(`landed ${k} (${L.pct}% faster): ${L.pr}`);
    return 'landed';
  }
  leaveBranch();
  return 'blocked';
}

// ---------------------------------------------------------------- finish
function cmdFinish(k, o) {
  const out = withKernel(k, `finish ${k}`, () => {
    const st = loadState(k);
    if (!st) die(`no loop for ${k}`);
    checkOwner(k, o.worker);
    if (st.landed || st.done) { console.log(`${k} is already finished (${st.landed ? 'landed' : st.done.reason})`); return 'done'; }
    if (st.needsReinit) { console.log(`${k} restarts on the current code at the next claim`); return 'restart'; }
    if (!st.landing) {
      if (st.best.attempt === (st.baseline.attempt || 0)) { console.log(`${k}: no attempt beat the baseline; marked done.`); return markDone(k, st, 'no gain'); }
      const ok = st.verified?.ok && st.verified.attempt === st.best.attempt ? true : cmdVerify(k);
      if (!ok) return markDone(k, loadState(k), 'verify_fail');
      if (!C.autoLand) { console.log(`autoLand is off in rsi.config.json: land it with "rsi land ${k}"`); return 'verified'; }
      const s2 = loadState(k);
      if (C.requireApproval && s2.approved?.attempt !== s2.best.attempt) { console.log(`${k} is verified and waits for a reviewer: rsi approve ${k}`); return 'awaiting approval'; }
    }
    return cmdLand(k);
  });
  fs.rmSync(claimPath(k), { force: true });
  console.log(`released ${k} (${out}). Next: rsi claim SCENARIO --worker NAME`);
}

// ---------------------------------------------------------------- doctor, main
function cmdDoctor(sc) {
  const notes = doctor({ scenario: sc, quiet: true });
  for (const n of notes) console.log(n);
  console.log(`ok: tools, models, disk (${freeGB(RUNS).toFixed(0)} GB free), engine/build ${buildStale() ? 'STALE' : 'current'}, repository on ${branchNow()}${dirtyAll() ? ' (uncommitted changes)' : ''}, kbench current`);
  if (fs.existsSync(path.join(RUNS, '.gate-broken'))) console.log(`warning: .gate-broken: ${fs.readFileSync(path.join(RUNS, '.gate-broken'), 'utf8')}`);
  if (fs.existsSync(BROKEN())) console.log(`warning: engine/build may not match the source: ${fs.readFileSync(BROKEN(), 'utf8')}`);
}
// options that always take a value (so an idea that starts with "--" is still the idea); the rest are flags unless a
// value follows; --name=value works for any
const VALUED = new Set(['idea', 'tag', 'worker', 'scenario', 'holdout', 'attempt', 'dump']);
function args(argv) {
  const pos = [], opt = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const eq = a.indexOf('=');
      if (eq > 0) { opt[a.slice(2, eq)] = a.slice(eq + 1); continue; }
      const key = a.slice(2);
      if (i + 1 < argv.length && (VALUED.has(key) || !argv[i + 1].startsWith('--'))) opt[key] = argv[++i];
      else opt[key] = true;
    } else pos.push(a);
  }
  return { pos, opt };
}
const [cmd, ...rest] = process.argv.slice(2);
const { pos, opt } = args(rest);
const need1 = (what) => { if (!pos[0]) die(`usage: ${cmd} ${what}`); return pos[0]; };
try {
  switch (cmd) {
    case 'claim': cmdClaim(pos[0], opt); break;
    case 'eval': cmdEval(need1('KERNEL --idea "TEXT" --tag TAG --worker NAME'), opt); break;
    case 'finish': cmdFinish(need1('KERNEL --worker NAME'), opt); break;
    case 'land': cmdLand(need1('KERNEL')); break;
    case 'release': cmdRelease(need1('KERNEL --worker NAME [--force]'), opt); break;
    case 'approve': cmdApprove(need1('KERNEL [--worker NAME]'), opt); break;
    case 'reopen': cmdReopen(need1('KERNEL')); break;
    case 'doctor': cmdDoctor(pos[0]); break;
    case 'discover': cmdDiscover(need1('SCENARIO'), opt); break;
    case 'capture': if (!pos[1]) die('usage: capture SCENARIO KERNEL[,KERNEL...]'); console.log('captured: ' + withRepo('capture', () => capture(pos[0], pos[1].split(','))).join(', ')); break;
    case 'queue': printQueue(pos[0]); break;
    case 'init': cmdInit(need1('KERNEL'), opt); break;
    case 'status': cmdStatus(need1('KERNEL'), opt); break;
    case 'isa': cmdIsa(need1('KERNEL'), opt); break;
    case 'verify': cmdVerify(need1('KERNEL')); break;
    case 'patch': console.log('changed: ' + makePatch(need1('KERNEL')).join(', ')); break;
    default: {
      const lines = fs.readFileSync(fileURLToPath(import.meta.url), 'utf8').split('\n').slice(1);
      console.log(lines.slice(0, lines.findIndex((l) => !l.startsWith('//'))).map((l) => l.replace(/^\/\/ ?/, '')).join('\n'));
      process.exitCode = cmd ? 1 : 0;
    }
  }
} catch (e) {
  if (!(e instanceof Fail)) throw e;
  console.error(`rsi: ${e.message}`);
  if (e.kind === 'env') console.error('(a machine or tool problem: fix it and run the command again)');
  if (e.kind === 'busy') console.error('(the GPU stayed in use: run the command again later)');
  process.exitCode = e.kind === 'error' ? 1 : e.kind === 'env' ? 3 : e.kind === 'busy' ? 4 : 5;
}
