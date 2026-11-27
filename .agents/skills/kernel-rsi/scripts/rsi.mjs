#!/usr/bin/env node
// rsi.mjs -- the measurement side of the kernel-rsi loop: find the kernels a scenario runs, capture and rank them,
// then for one kernel keep a working copy of the kernel sources, build each edit, check its output is bit-identical
// and time it against the best version so far (tools/kbench.c), keep or revert it, and log every attempt to a JSONL
// ledger. The agent supplies the ideas and edits; this script decides nothing about code.
//
//   node rsi.mjs discover SCENARIO             trace + capture + rank the kernels of a scenario -> queue
//   node rsi.mjs queue [SCENARIO]               the ranked queue and each kernel's loop state
//   node rsi.mjs next [SCENARIO]                the highest-ranked kernel whose loop has not stopped (for a coordinator)
//   node rsi.mjs init KERNEL [--scenario S] [--holdout S1,S2] [--force]
//   node rsi.mjs eval KERNEL --idea "TEXT" --tag TAG [--worker NAME]
//   node rsi.mjs status KERNEL [--json]         best so far, recent attempts, failed ideas, stop decision
//   node rsi.mjs isa KERNEL [--attempt N] [--dump LOOP]   loop instruction mix of the best (or an attempt's) build
//   node rsi.mjs verify KERNEL                  independent re-measure of baseline vs best on every capture
//   node rsi.mjs capture SCENARIO KERNEL[,KERNEL...]      (re)capture launches for a scenario
// Machine paths, scenarios and thresholds: rsi.config.json next to this file ($RSI_CONFIG overrides it).
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '../../../..');
const ENGINE = path.join(REPO, 'engine');
const UNITS = {
  base: { src: 'kernels/ie_kernels.c', hsaco: 'ie_kernels.hsaco', warn: true },
  inkling: { src: 'kernels/ie_inkling_kernels.c', hsaco: 'ie_inkling_kernels.hsaco', warn: true },
  saluki: { src: 'kernels/ie_saluki_kernels.c', hsaco: 'ie_saluki_kernels.hsaco', warn: false },
  vision: { src: 'kernels/ie_vision_kernels.c', hsaco: 'ie_vision_kernels.hsaco', warn: true },
};

// ---------------------------------------------------------------- config, helpers
function loadConfig() {
  const read = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
  let c = read(path.join(HERE, 'rsi.config.json'));
  if (process.env.RSI_CONFIG) c = { ...c, ...read(process.env.RSI_CONFIG) };
  const lab = c.lab;
  const fix = (v) => (typeof v === 'string' ? v.replaceAll('{lab}', lab) : v);
  for (const k of Object.keys(c)) c[k] = fix(c[k]);
  for (const k of Object.keys(c.prompts)) c.prompts[k] = fix(c.prompts[k]);
  return c;
}
const C = loadConfig();
const RUNS = C.runsDir;
const die = (m) => { console.error('rsi: ' + m); process.exit(1); };
const exe = (name) => path.join(C.llvmBin, name + (process.platform === 'win32' ? '.exe' : ''));
const pad = (n) => String(n).padStart(3, '0');
const now = () => new Date().toISOString();
const readJson = (p, d) => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : d);
const writeJson = (p, v) => { fs.mkdirSync(path.dirname(p), { recursive: true }); fs.writeFileSync(p, JSON.stringify(v, null, 2) + '\n'); };
const kdir = (k) => path.join(RUNS, 'kernels', k);
const ledgerPath = (k) => path.join(kdir(k), 'ledger.jsonl');
const capPath = (sc, k) => path.join(RUNS, 'captures', sc, k + '.kcap');
function ledger(k) {
  const p = ledgerPath(k);
  if (!fs.existsSync(p)) return [];
  return fs.readFileSync(p, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
}
const appendLedger = (k, rec) => fs.appendFileSync(ledgerPath(k), JSON.stringify(rec) + '\n');

function args(argv) {
  const pos = [], opt = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      if (i + 1 < argv.length && !argv[i + 1].startsWith('--')) opt[key] = argv[++i];
      else opt[key] = true;
    } else pos.push(a);
  }
  return { pos, opt };
}
function run(cmd, argv, o = {}) {
  const r = spawnSync(cmd, argv, { encoding: 'utf8', maxBuffer: 256 << 20, timeout: o.timeout ?? 900000, cwd: o.cwd, env: o.env });
  if (r.error) die(`${path.basename(cmd)}: ${r.error.message}`);
  return { code: r.status, out: (r.stdout || '') + (r.stderr || '') };
}
function copyDir(src, dst) { fs.rmSync(dst, { recursive: true, force: true }); fs.cpSync(src, dst, { recursive: true }); }

// ---------------------------------------------------------------- GPU lock (one GPU job at a time, across workers)
const LOCK = path.join(RUNS, '.gpu-lock');
function alive(pid) { try { process.kill(pid, 0); return true; } catch { return false; } }
function withGpu(what, fn) {
  fs.mkdirSync(RUNS, { recursive: true });
  const t0 = Date.now();
  let said = false;
  for (;;) {
    let busy = fs.existsSync(C.labGpuLock) ? 'the lab harness (ref/.gpu-lock)' : null;
    if (!busy) {
      try {
        fs.mkdirSync(LOCK);
        fs.writeFileSync(path.join(LOCK, 'owner.json'), JSON.stringify({ pid: process.pid, what, since: now() }));
        break;
      } catch {
        const o = readJson(path.join(LOCK, 'owner.json'), null);
        if (o && !alive(o.pid)) { fs.rmSync(LOCK, { recursive: true, force: true }); continue; }
        busy = o ? `${o.what} (pid ${o.pid})` : 'another job';
      }
    }
    if (Date.now() - t0 > 45 * 60000) die('waited 45 min for the GPU: ' + busy);
    if (!said) { console.log(`waiting for the GPU: ${busy}`); said = true; }
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 2000);
  }
  try { return fn(); } finally { fs.rmSync(LOCK, { recursive: true, force: true }); }
}

// ---------------------------------------------------------------- engine runs
function engineEnv(extra) {
  const env = {};
  for (const [k, v] of Object.entries(process.env)) if (!k.startsWith('IE_')) env[k] = v;
  const pk = Object.keys(env).find((k) => k.toUpperCase() === 'PATH') || 'PATH';
  env[pk] = C.rocmBin.replaceAll('/', path.sep) + path.delimiter + (env[pk] || '');
  return { ...env, ...extra };
}
function ieRun(sc, extra) {
  const s = C.scenarios[sc];
  if (!s) die(`unknown scenario ${sc} (${Object.keys(C.scenarios).join(', ')})`);
  const argv = [C.model, '--backend', 'gpu', '--hsaco', 'build/ie_kernels.hsaco', '--tokens-file', C.prompts[s.prompt],
    '--ctx', '32768', '--n', '16', '--draft', '7', '--dspark', C.dspark, '--draft-pmin', '0.5'];
  const env = engineEnv({ IE_VERIFY_BENCH: String(s.T), IE_VERIFY_BENCH_TEXT: '1', ...extra });
  return run(path.join(ENGINE, 'build', 'ie-run.exe'), argv, { cwd: ENGINE, env, timeout: 600000 });
}
function capture(sc, kernels) {
  const s = C.scenarios[sc];
  const dir = path.join(RUNS, 'captures', sc);
  fs.mkdirSync(dir, { recursive: true });
  const r = withGpu(`capture ${sc}`, () => ieRun(sc, { IE_KCAP: kernels.join(','), IE_KCAP_PHASE: s.phase, IE_KCAP_DIR: dir }));
  const missing = kernels.filter((k) => !fs.existsSync(capPath(sc, k)));
  if (missing.length) console.log(`not launched in phase ${s.phase} of ${sc}: ${missing.join(', ')}`);
  if (!/kcap: done/.test(r.out) && missing.length === kernels.length) die('capture failed:\n' + r.out.slice(-3000));
  return kernels.filter((k) => !missing.includes(k));
}

// ---------------------------------------------------------------- code objects
let symCache = null;
function unitOf(kernel) {
  if (!symCache) {
    symCache = {};
    for (const [u, d] of Object.entries(UNITS)) {
      const p = path.join(ENGINE, 'build', d.hsaco);
      if (!fs.existsSync(p)) continue;
      const out = run(exe('llvm-readelf'), ['--notes', p]).out;
      for (const m of out.matchAll(/\n\s+\.name:\s+(\S+)/g)) symCache[m[1]] ??= u;
    }
  }
  return symCache[kernel] || null;
}
function resources(hsaco, kernel) {
  const out = run(exe('llvm-readelf'), ['--notes', hsaco]).out;
  for (const blk of out.split(/\n  - \.args:/)) {
    const m = blk.match(/\n\s+\.name:\s+(\S+)/);
    if (!m || m[1] !== kernel) continue;
    const g = (k) => { const v = blk.match(new RegExp('\\n\\s+\\.' + k + ':\\s+(\\d+)')); return v ? +v[1] : -1; };
    const vgpr = g('vgpr_count');
    return { vgpr, sgpr: g('sgpr_count'), lds: g('group_segment_fixed_size'), priv: g('private_segment_fixed_size'),
      vspill: g('vgpr_spill_count'), sspill: g('sgpr_spill_count'), waves: vgpr > 0 ? Math.min(16, Math.floor(1536 / (Math.ceil(vgpr / 8) * 8))) : 0 };
  }
  return null;
}
// build one unit from a source root (kernels/, core/, build/*.h) with the flags of build-win.sh
function compile(root, unit, out) {
  const d = UNITS[unit];
  const ll = out + '.ll';
  const tgt = ['--target=amdgcn-amd-amdhsa', '-mcpu=gfx1201', '-nogpulib', '-O3'];
  let r = run(exe('amdclang'), ['-x', 'c', '-std=c11', ...(d.warn ? ['-Wall', '-Wextra'] : []), ...tgt, '-Ibuild', '-S', '-emit-llvm', '-o', ll, d.src], { cwd: root });
  if (r.code !== 0) return { ok: false, log: r.out };
  const txt = fs.readFileSync(ll, 'utf8').replace(/^(attributes #\d+ = \{.*)"target-cpu"/gm, '$1"amdgpu-flat-work-group-size"="256,256" "target-cpu"');
  fs.writeFileSync(ll, txt);
  r = run(exe('amdclang'), ['-nogpulib', '--target=amdgcn-amd-amdhsa', '-mcpu=gfx1201', '-O3', '-c', '-o', out + '.o', ll]);
  if (r.code !== 0) return { ok: false, log: r.out };
  r = run(exe('ld.lld'), ['-shared', '-o', out, out + '.o']);
  fs.rmSync(ll, { force: true }); fs.rmSync(out + '.o', { force: true });
  return r.code === 0 ? { ok: true, log: '' } : { ok: false, log: r.out };
}

// ---------------------------------------------------------------- kbench
function kbench(cap, ref, cand, o = {}) {
  const kb = path.join(ENGINE, 'build', 'kbench.exe');
  if (!fs.existsSync(kb)) die('engine/build/kbench.exe is missing: gcc -std=c11 -O2 -o build/kbench.exe tools/kbench.c');
  const argv = [cap, ref, ...(cand ? [cand] : []), '--cold', String(o.cold ?? C.kbench.cold), '-r', String(o.rounds ?? C.kbench.rounds), '-n', String(C.kbench.launches)];
  const r = run(kb, argv, { env: engineEnv({}), timeout: 600000 });
  const num = (re) => { const m = r.out.match(re); return m ? [+m[1], +m[2]] : null; };
  const res = {
    replay: /replay matches engine: yes/.test(r.out),
    identical: cand ? /candidate output: bit-identical/.test(r.out) : null,
    ref: num(/ref : median\s+([\d.]+) us\s+min\s+([\d.]+)/),
    cand: cand ? num(/cand: median\s+([\d.]+) us\s+min\s+([\d.]+)/) : null,
    firstDiff: (r.out.match(/allocation \d+: \d+ bytes differ.*/) || [null])[0],
    raw: r.out,
  };
  if (r.code === 4 && o.soft) return { ...res, replay: false, nondet: true, ref: null, deltaPct: null };
  if (r.code === 4) die('the kernel is not deterministic (two runs of the same build differ), so bit-identity cannot judge it:\n' + r.out.slice(-1500));
  if (r.code === 3 && o.soft) return { ...res, replay: false, ref: null, deltaPct: null };
  if (r.code === 3) die('the replay does not reproduce the engine output (capture incomplete?), so nothing was timed:\n' + r.out.slice(-1500));
  if (r.code !== 0 || !res.ref) die('kbench failed:\n' + r.out.slice(-2000));
  res.deltaPct = res.cand ? (100 * (res.cand[0] - res.ref[0])) / res.ref[0] : null;
  return res;
}

// ---------------------------------------------------------------- commands
function cmdCapture(sc, list) {
  if (!sc || !list) die('usage: capture SCENARIO KERNEL[,KERNEL...]');
  const got = capture(sc, list.split(','));
  console.log(`captured ${got.length}: ${got.join(', ')}`);
}

function cmdDiscover(sc) {
  const s = C.scenarios[sc];
  if (!s) die(`unknown scenario ${sc}`);
  const csv = path.join(RUNS, 'discover', sc + '.csv');
  fs.mkdirSync(path.dirname(csv), { recursive: true });
  console.log(`1/3 tracing the launches of ${sc} (${s.about}) ...`);
  withGpu(`trace ${sc}`, () => ieRun(sc, { IE_TRACE: csv }));
  const lines = fs.readFileSync(csv, 'utf8').trim().split('\n');
  const hdr = lines[0].split(',');
  const ix = (n) => hdr.indexOf(n);
  const counts = {}, calls = new Set(), groups = {};
  for (const l of lines.slice(1)) {
    const f = l.split(',');
    if (f[ix('phase')] !== s.phase) continue;
    if (s.phase === 'verify' && +f[ix('T_run')] !== s.T) continue;
    calls.add(f[ix('call')]);
    const k = f[ix('kernel')];
    counts[k] = (counts[k] || 0) + 1;
    groups[k] = f[ix('groups')] + 'x' + f[ix('T')];
  }
  if (!calls.size) die(`no ${s.phase} launches in the trace`);
  const kernels = Object.keys(counts).filter((k) => unitOf(k));
  console.log(`2/3 capturing ${kernels.length} kernels (${calls.size} ${s.phase} runs traced) ...`);
  const todo = kernels.filter((k) => !fs.existsSync(capPath(sc, k)));
  if (todo.length) capture(sc, todo);
  console.log('3/3 timing each capture alone (cold) ...');
  const rows = [];
  for (const k of kernels) {
    if (!fs.existsSync(capPath(sc, k))) continue;
    const unit = unitOf(k);
    const r = withGpu(`rank ${k}`, () => kbench(capPath(sc, k), path.join(ENGINE, 'build', UNITS[unit].hsaco), null, { rounds: 5, soft: true }));
    const perStep = counts[k] / calls.size;
    if (r.nondet) console.log(`  ${k}: not deterministic (atomics decide some output bytes); the loop cannot judge it`);
    else if (!r.replay) console.log(`  ${k}: the replay does not match the engine; recapture (rsi capture ${sc} ${k}) before working on it`);
    rows.push({ kernel: k, unit, perStep: +perStep.toFixed(2), grid: groups[k], us: r.ref ? r.ref[0] : 0, msPerStep: r.ref ? +((r.ref[0] * perStep) / 1000).toFixed(4) : 0, replay: r.replay, nondet: !!r.nondet });
  }
  rows.sort((a, b) => b.msPerStep - a.msPerStep);
  const total = rows.reduce((a, r) => a + r.msPerStep, 0);
  for (const r of rows) r.share = +((100 * r.msPerStep) / total).toFixed(1);
  writeJson(path.join(RUNS, 'queue-' + sc + '.json'), { scenario: sc, about: s.about, created: now(), runsTraced: calls.size, totalMsPerStep: +total.toFixed(3), kernels: rows });
  printQueue(sc);
}

function loopState(k) {
  const st = readJson(path.join(kdir(k), 'state.json'), null);
  if (!st) return 'not started';
  const s = stopCheck(st, ledger(k));
  return `${s.stop ? 'stopped' : 'running'} ${st.attempts} tries, x${st.speedup.toFixed(3)}${st.verified ? ', verified' : ''}`;
}
function printQueue(sc) {
  const files = sc ? ['queue-' + sc + '.json'] : fs.existsSync(RUNS) ? fs.readdirSync(RUNS).filter((f) => /^queue-.*\.json$/.test(f)) : [];
  if (!files.length) die('no queue yet: run discover SCENARIO');
  for (const f of files) {
    const q = readJson(path.join(RUNS, f));
    console.log(`\n${q.scenario}: ${q.about}; kernels ${q.totalMsPerStep} ms per run (cold, each alone)`);
    console.log('  #  share   ms/run  us/launch  per-run  kernel  [loop]');
    q.kernels.forEach((r, i) => console.log(`${String(i + 1).padStart(3)} ${String(r.share).padStart(5)}%  ${r.msPerStep.toFixed(3).padStart(7)}  ${r.us.toFixed(2).padStart(9)}  ${String(r.perStep).padStart(7)}  ${r.kernel}${r.nondet ? ' (NOT DETERMINISTIC)' : r.replay ? '' : ' (REPLAY MISMATCH)'}  [${loopState(r.kernel)}]`));
  }
}
function cmdNext(sc) {
  const files = sc ? ['queue-' + sc + '.json'] : fs.existsSync(RUNS) ? fs.readdirSync(RUNS).filter((f) => /^queue-.*\.json$/.test(f)) : [];
  if (!files.length) die('no queue yet: run discover SCENARIO');
  for (const f of files) {
    const q = readJson(path.join(RUNS, f));
    for (const r of q.kernels) {
      if (!r.replay) continue;
      const st = readJson(path.join(kdir(r.kernel), 'state.json'), null);
      if (!st || !stopCheck(st, ledger(r.kernel)).stop) { console.log(`${r.kernel} ${q.scenario} ${r.share}%`); return; }
    }
  }
  console.log('none: every queued kernel has stopped');
}

function cmdInit(k, o) {
  if (!k) die('usage: init KERNEL [--scenario S] [--holdout S1,S2] [--force]');
  const unit = unitOf(k);
  if (!unit) die(`${k} is in no code object under engine/build`);
  const d = kdir(k);
  if (fs.existsSync(path.join(d, 'state.json')) && !o.force) die(`${k} already has a loop (status ${k}); --force starts over`);
  let sc = o.scenario;
  if (!sc) {
    const qs = fs.existsSync(RUNS) ? fs.readdirSync(RUNS).filter((f) => /^queue-.*\.json$/.test(f)) : [];
    for (const f of qs) if (readJson(path.join(RUNS, f)).kernels.some((r) => r.kernel === k)) { sc = readJson(path.join(RUNS, f)).scenario; break; }
  }
  if (!sc) die('no queue lists this kernel: pass --scenario');
  const holdouts = o.holdout === true || o.holdout === undefined ? C.defaultHoldouts[sc] || [] : o.holdout.split(',').filter(Boolean);
  for (const s of [sc, ...holdouts]) if (!fs.existsSync(capPath(s, k))) capture(s, [k]);
  if (!fs.existsSync(capPath(sc, k))) die(`${k} does not run in ${sc}`);
  const hold = holdouts.filter((s) => fs.existsSync(capPath(s, k)));
  fs.rmSync(d, { recursive: true, force: true });
  fs.mkdirSync(path.join(d, 'attempts'), { recursive: true });
  const work = path.join(d, 'work');
  for (const sub of ['kernels', 'core']) fs.cpSync(path.join(ENGINE, sub), path.join(work, sub), { recursive: true });
  fs.mkdirSync(path.join(work, 'build'), { recursive: true });
  for (const f of fs.readdirSync(path.join(ENGINE, 'build')).filter((f) => f.endsWith('.h'))) fs.copyFileSync(path.join(ENGINE, 'build', f), path.join(work, 'build', f));
  copyDir(path.join(work, 'kernels'), path.join(d, 'best-src'));
  console.log(`building the baseline ${UNITS[unit].hsaco} from a copy of engine/ ...`);
  const c = compile(work, unit, path.join(d, 'baseline.hsaco'));
  if (!c.ok) die('baseline build failed:\n' + c.log.slice(-3000));
  fs.copyFileSync(path.join(d, 'baseline.hsaco'), path.join(d, 'best.hsaco'));
  const same = Buffer.compare(fs.readFileSync(path.join(d, 'baseline.hsaco')), fs.readFileSync(path.join(ENGINE, 'build', UNITS[unit].hsaco))) === 0;
  const res = resources(path.join(d, 'baseline.hsaco'), k);
  const r = withGpu(`baseline ${k}`, () => kbench(capPath(sc, k), path.join(d, 'best.hsaco'), path.join(d, 'best.hsaco')));
  if (!r.replay) die('the replay does not reproduce the engine output: recapture before optimising');
  const st = { kernel: k, unit, scenario: sc, holdouts: hold, created: now(), sameAsEngineBuild: same,
    baseline: { us: r.ref[0], min: r.ref[1], noisePct: +Math.abs(r.deltaPct).toFixed(2), res },
    best: { attempt: 0, us: r.ref[0], res }, speedup: 1, attempts: 0, lastAccept: 0, accepts: [] };
  writeJson(path.join(d, 'state.json'), st);
  appendLedger(k, { ts: now(), kernel: k, attempt: 0, decision: 'baseline', scenario: sc, holdouts: hold, ref_us: r.ref[0], cand_us: r.ref[0], delta_pct: 0, noise_pct: st.baseline.noisePct, res, best_us: r.ref[0], speedup: 1 });
  console.log(`${k}: ${unit}, scenario ${sc}${hold.length ? ', holdouts ' + hold.join(', ') : ''}`);
  console.log(`baseline ${r.ref[0].toFixed(2)} us (self A/B ${r.deltaPct.toFixed(2)}%), VGPR ${res.vgpr} (${res.waves} waves), LDS ${res.lds}, build ${same ? 'identical to engine/build' : 'NOT identical to engine/build (rebuild engine/build first?)'}`);
  console.log(`edit sources in ${work.replaceAll('\\', '/')}/kernels, then: node rsi.mjs eval ${k} --idea "..." --tag TAG`);
}

function stopCheck(st, led) {
  const since = st.attempts - st.lastAccept;
  const last = st.accepts.slice(-3).map((a) => -a.gainPct);
  if (st.attempts >= C.stop.maxAttempts) return { stop: true, why: `reached ${C.stop.maxAttempts} attempts` };
  if (since >= C.stop.patience) return { stop: true, why: `no accepted gain in the last ${since} attempts (patience ${C.stop.patience})` };
  if (last.length === 3 && last.reduce((a, b) => a + b, 0) < C.stop.diminishingPct && since >= Math.ceil(C.stop.patience / 2))
    return { stop: true, why: `diminishing returns: the last 3 gains total ${last.reduce((a, b) => a + b, 0).toFixed(2)}% and ${since} attempts since` };
  return { stop: false, why: `${since}/${C.stop.patience} attempts since the last gain, ${st.attempts}/${C.stop.maxAttempts} in all` };
}

function cmdEval(k, o) {
  if (!k || !o.idea || !o.tag) die('usage: eval KERNEL --idea "TEXT" --tag TAG [--worker NAME]');
  if (!C.tags.includes(o.tag)) die(`tag must be one of ${C.tags.join(', ')}`);
  const d = kdir(k);
  const st = readJson(path.join(d, 'state.json'), null);
  if (!st) die(`no loop for ${k}: init it first`);
  const work = path.join(d, 'work'), best = path.join(d, 'best-src');
  const n = st.attempts + 1;
  const diff = run('git', ['diff', '--no-index', '--no-color', '--', best, path.join(work, 'kernels')]).out;
  if (!diff.trim()) die('no change in work/kernels against the best version');
  const files = [...diff.matchAll(/^\+\+\+ b?(.*)$/gm)].map((m) => path.basename(m[1].trim().replace(/^"|"$/g, "")));
  const ap = path.join(d, 'attempts', pad(n));
  fs.writeFileSync(ap + '.diff', diff);
  const rec = { ts: now(), kernel: k, attempt: n, idea: String(o.idea), tag: o.tag, worker: o.worker || null, files, diff: path.basename(ap) + '.diff', scenario: st.scenario };
  const t0 = Date.now();
  const revert = () => copyDir(best, path.join(work, 'kernels'));
  const finish = (decision, extra = {}) => {
    Object.assign(rec, extra, { decision, wall_s: +((Date.now() - t0) / 1000).toFixed(1) });
    if (decision === 'accepted') {
      copyDir(path.join(work, 'kernels'), best);
      fs.copyFileSync(ap + '.hsaco', path.join(d, 'best.hsaco'));
      const ratio = rec.ref_us / rec.cand_us;
      st.speedup *= ratio;
      st.best = { attempt: n, us: rec.cand_us, res: rec.res };
      st.lastAccept = n;
      st.accepts.push({ attempt: n, gainPct: +rec.delta_pct.toFixed(2) });
    } else {
      revert();
      if (fs.existsSync(ap + '.hsaco')) fs.rmSync(ap + '.hsaco');
    }
    st.attempts = n;
    rec.best_us = st.best.us;
    rec.speedup = +st.speedup.toFixed(4);
    writeJson(path.join(d, 'state.json'), st);
    appendLedger(k, rec);
    const s = stopCheck(st, ledger(k));
    const dl = rec.delta_pct != null ? ` ${rec.delta_pct >= 0 ? '+' : ''}${rec.delta_pct.toFixed(2)}%` : '';
    console.log(`#${n} ${decision.toUpperCase()}${dl}  ${rec.reason || ''}`.trim());
    console.log(`best ${st.best.us.toFixed(2)} us (attempt ${st.best.attempt}), total x${st.speedup.toFixed(3)} vs baseline`);
    console.log(s.stop ? `STOP: ${s.why}` : `continue: ${s.why}`);
  };
  console.log(`#${n}: building ${UNITS[st.unit].hsaco} with ${files.join(', ')} changed ...`);
  const c = compile(work, st.unit, ap + '.hsaco');
  if (!c.ok) return finish('compile_fail', { reason: c.log.split('\n').filter((l) => /error/.test(l)).slice(0, 6).join(' | ').slice(0, 600) });
  const res = resources(ap + '.hsaco', k);
  rec.res = res;
  if (!res) return finish('compile_fail', { reason: `the build has no kernel ${k}` });
  if (res.vspill > 0 || res.sspill > 0 || res.priv > 0) return finish('spill', { reason: `VGPR ${res.vgpr}, spills ${res.vspill}/${res.sspill}, scratch ${res.priv} B` });
  if (res.waves < st.best.res.waves) rec.note = `occupancy ${st.best.res.waves} -> ${res.waves} waves/SIMD (VGPR ${st.best.res.vgpr} -> ${res.vgpr})`;
  return withGpu(`eval ${k} #${n}`, () => {
    const r1 = kbench(capPath(st.scenario, k), path.join(d, 'best.hsaco'), ap + '.hsaco');
    Object.assign(rec, { ref_us: r1.ref[0], cand_us: r1.cand[0], cand_min_us: r1.cand[1], delta_pct: +r1.deltaPct.toFixed(3) });
    if (!r1.replay) return finish('error', { reason: 'replay no longer matches the engine output (capture stale?)' });
    if (!r1.identical) return finish('not_exact', { reason: r1.firstDiff || 'outputs differ' });
    const g = C.accept.minGainPct;
    if (r1.ref[0] / r1.cand[0] >= C.accept.suspectSpeedup) return finish('suspect', { reason: `x${(r1.ref[0] / r1.cand[0]).toFixed(2)} in one step: check it skips no work before trusting it (verifier)` });
    if (r1.deltaPct > -g) return finish(r1.deltaPct > g ? 'slower' : 'no_gain', { reason: `needs <= -${g}%` });
    const r2 = kbench(capPath(st.scenario, k), path.join(d, 'best.hsaco'), ap + '.hsaco');
    const avg = (r1.deltaPct + r2.deltaPct) / 2;
    rec.confirm_pct = +r2.deltaPct.toFixed(3);
    rec.ref_us = (r1.ref[0] + r2.ref[0]) / 2, rec.cand_us = (r1.cand[0] + r2.cand[0]) / 2, rec.delta_pct = +avg.toFixed(3);
    if (r2.deltaPct > -g / 2 || avg > -g) return finish('not_confirmed', { reason: `second run ${r2.deltaPct.toFixed(2)}%` });
    rec.holdouts = [];
    for (const h of st.holdouts) {
      const rh = kbench(capPath(h, k), path.join(d, 'best.hsaco'), ap + '.hsaco');
      rec.holdouts.push({ scenario: h, identical: rh.identical, delta_pct: +rh.deltaPct.toFixed(3) });
      if (!rh.identical) return finish('holdout_fail', { reason: `${h}: outputs differ` });
      if (rh.deltaPct > C.accept.holdoutTolPct) return finish('holdout_fail', { reason: `${h}: ${rh.deltaPct.toFixed(2)}% slower` });
    }
    return finish('accepted', { reason: rec.note || '' });
  });
}

function cmdStatus(k, o) {
  const st = readJson(path.join(kdir(k), 'state.json'), null);
  if (!st) die(`no loop for ${k}`);
  const led = ledger(k).filter((r) => r.attempt > 0);
  const s = stopCheck(st, led);
  const byTag = {};
  for (const t of C.tags) byTag[t] = { tried: 0, accepted: 0 };
  for (const r of led) { byTag[r.tag].tried++; if (r.decision === 'accepted') byTag[r.tag].accepted++; }
  const recentTags = led.slice(-3).map((r) => r.tag);
  const fresh = C.tags.filter((t) => !recentTags.includes(t)).sort((a, b) => byTag[a].tried - byTag[b].tried);
  const out = { kernel: k, unit: st.unit, scenario: st.scenario, holdouts: st.holdouts, baseline_us: st.baseline.us, noise_pct: st.baseline.noisePct,
    best: st.best, speedup: +st.speedup.toFixed(4), attempts: st.attempts, n_accepted: st.accepts.length, stop: s, verified: st.verified || null,
    byTag, suggestTag: led.length >= 3 && led.slice(-3).every((r) => r.decision !== 'accepted') ? fresh[0] : null,
    accepted: led.filter((r) => r.decision === 'accepted').map((r) => ({ attempt: r.attempt, idea: r.idea, delta_pct: r.delta_pct })),
    failed: led.filter((r) => r.decision !== 'accepted').map((r) => ({ attempt: r.attempt, tag: r.tag, decision: r.decision, idea: r.idea, delta_pct: r.delta_pct ?? null, reason: r.reason || '' })) };
  if (o.json) { console.log(JSON.stringify(out, null, 2)); return; }
  console.log(`${k} (${st.unit}) on ${st.scenario}${st.holdouts.length ? ', holdouts ' + st.holdouts.join(', ') : ''}`);
  console.log(`baseline ${st.baseline.us.toFixed(2)} us, best ${st.best.us.toFixed(2)} us (attempt ${st.best.attempt}, VGPR ${st.best.res.vgpr}), total x${st.speedup.toFixed(3)}, noise ~${st.baseline.noisePct}%`);
  console.log(`${st.attempts} attempts, ${st.accepts.length} accepted; ${s.stop ? 'STOP: ' : 'continue: '}${s.why}`);
  if (out.accepted.length) { console.log('\nkept (in order):'); for (const a of out.accepted) console.log(`  #${a.attempt} ${a.delta_pct.toFixed(2)}%  ${a.idea}`); }
  if (out.failed.length) { console.log('\ntried and dropped (do not repeat):'); for (const f of out.failed) console.log(`  #${f.attempt} [${f.tag}] ${f.decision}${f.delta_pct != null ? ' ' + f.delta_pct.toFixed(2) + '%' : ''}  ${f.idea}${f.reason ? '  -- ' + f.reason.slice(0, 160) : ''}`); }
  console.log('\nby tag: ' + C.tags.map((t) => `${t} ${byTag[t].accepted}/${byTag[t].tried}`).join(', '));
  if (out.suggestTag) console.log(`three misses in a row: switch to a different kind of idea, e.g. "${out.suggestTag}"`);
}

function cmdIsa(k, o) {
  const d = kdir(k);
  const h = o.attempt ? path.join(d, 'attempts', pad(+o.attempt) + '.hsaco') : path.join(d, 'best.hsaco');
  if (!fs.existsSync(h)) die(`no build at ${h} (rejected attempts keep only their diff)`);
  const res = resources(h, k);
  console.log(`${k}: VGPR ${res.vgpr} (${res.waves} waves/SIMD), SGPR ${res.sgpr}, LDS ${res.lds} B, spills ${res.vspill}/${res.sspill}, scratch ${res.priv}`);
  const r = run('python', [path.join(ENGINE, 'tools', 'kernel_isa.py'), h, k, ...(o.dump ? ['--dump', String(o.dump)] : [])], { env: { ...process.env, LLVM_BIN: C.llvmBin } });
  console.log(o.dump ? r.out : r.out.split('\n').slice(0, 12).join('\n'));
}

function cmdVerify(k) {
  const d = kdir(k);
  const st = readJson(path.join(d, 'state.json'), null);
  if (!st) die(`no loop for ${k}`);
  if (st.best.attempt === 0) die('nothing accepted yet');
  const results = withGpu(`verify ${k}`, () => [st.scenario, ...st.holdouts].map((s) => {
    const r = kbench(capPath(s, k), path.join(d, 'baseline.hsaco'), path.join(d, 'best.hsaco'), { rounds: 21 });
    return { scenario: s, replay: r.replay, identical: r.identical, baseline_us: r.ref[0], best_us: r.cand[0], delta_pct: +r.deltaPct.toFixed(3) };
  }));
  const ok = results.every((r) => r.replay && r.identical && r.delta_pct < 0);
  const all = run('git', ['diff', '--no-index', '--no-color', '--name-only', '--', path.join(ENGINE, 'kernels'), path.join(d, 'best-src')]).out;
  const files = [...new Set(all.split('\n').filter(Boolean).map((f) => path.basename(f.trim().replace(/^"|"$/g, ''))))];
  // one repo-relative patch (git apply from the repository root), file by file
  const diff = files.map((f) => {
    const rel = 'engine/kernels/' + f;
    const body = run('git', ['diff', '--no-index', '--no-color', '--', path.join(ENGINE, 'kernels', f), path.join(d, 'best-src', f)]).out;
    return body.replace(/^diff --git .*$/m, `diff --git a/${rel} b/${rel}`).replace(/^--- .*$/m, `--- a/${rel}`).replace(/^\+\+\+ .*$/m, `+++ b/${rel}`);
  }).join('');
  fs.writeFileSync(path.join(d, 'best.diff'), diff);
  st.verified = { ts: now(), ok, results, files };
  writeJson(path.join(d, 'state.json'), st);
  appendLedger(k, { ts: now(), kernel: k, attempt: st.attempts, decision: ok ? 'verified' : 'verify_fail', results, best_us: st.best.us, speedup: +st.speedup.toFixed(4) });
  for (const r of results) console.log(`${r.scenario}: baseline ${r.baseline_us.toFixed(2)} us -> best ${r.best_us.toFixed(2)} us (${r.delta_pct.toFixed(2)}%), ${r.identical ? 'bit-identical' : 'OUTPUT DIFFERS'}${r.replay ? '' : ', REPLAY MISMATCH'}`);
  console.log(ok ? `verified. To land, on a work branch: git apply ${path.join(d, 'best.diff').replaceAll('\\', '/')} (changes ${files.join(', ')}), rebuild, then run the full SHA gate.` : 'NOT verified: do not land.');
}

// ---------------------------------------------------------------- main
const [cmd, ...rest] = process.argv.slice(2);
const { pos, opt } = args(rest);
switch (cmd) {
  case 'discover': cmdDiscover(pos[0]); break;
  case 'capture': cmdCapture(pos[0], pos[1]); break;
  case 'queue': printQueue(pos[0]); break;
  case 'next': cmdNext(pos[0]); break;
  case 'init': cmdInit(pos[0], opt); break;
  case 'eval': cmdEval(pos[0], opt); break;
  case 'status': cmdStatus(pos[0], opt); break;
  case 'isa': cmdIsa(pos[0], opt); break;
  case 'verify': cmdVerify(pos[0]); break;
  default:
    {
      const lines = fs.readFileSync(fileURLToPath(import.meta.url), 'utf8').split('\n').slice(1);
      const end = lines.findIndex((l) => !l.startsWith('//'));
      console.log(lines.slice(0, end).map((l) => l.replace(/^\/\/ ?/, '')).join('\n'));
    }
    process.exit(cmd ? 1 : 0);
}
