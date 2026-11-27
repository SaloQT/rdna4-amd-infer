#!/usr/bin/env node
// report.mjs -- the kernel-rsi ledgers as charts and flat files.
//   node report.mjs [--out DIR]     (default: the runs directory of rsi.config.json)
// Writes all.jsonl (every record of every kernel, one JSON object per line), summary.csv (one row per kernel) and
// report.html (self-contained: inline SVG, no scripts or network; sized to fit one screen at any number of kernels).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
let C = JSON.parse(fs.readFileSync(path.join(HERE, 'rsi.config.json'), 'utf8'));
if (process.env.RSI_CONFIG) C = { ...C, ...JSON.parse(fs.readFileSync(process.env.RSI_CONFIG, 'utf8')) };
const RUNS = C.runsDir.replaceAll('{lab}', C.lab);
const oi = process.argv.indexOf('--out');
const OUT = oi > 0 ? process.argv[oi + 1] : RUNS;
const KDIR = path.join(RUNS, 'kernels');
if (!fs.existsSync(KDIR)) { console.error('report: no kernel loops under ' + KDIR); process.exit(1); }

// attempt outcomes; 'noisy' and 'busy' are measurements that did not happen, not attempts
const COLORS = { accepted: '#1a9850', no_gain: '#9e9e9e', slower: '#f46d43', not_exact: '#d73027', not_confirmed: '#fdae61',
  holdout_fail: '#e08214', collateral: '#b35806', spill: '#8073ac', compile_fail: '#542788', out_of_scope: '#2d004b',
  suspect: '#000000', fault: '#c51b7d', hang: '#8e0152', nondeterministic: '#4d9221', error: '#67001f' };
const NOT_ATTEMPTS = new Set(['noisy', 'busy']);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const short = (k) => k.replace(/^ie_inkling_/, '').replace(/^ie_/, '');
const readLines = (p) => fs.readFileSync(p, 'utf8').split('\n').filter((l) => l.trim()).flatMap((l) => { try { return [JSON.parse(l)]; } catch { return []; } });

const perStep = {};
for (const f of fs.readdirSync(RUNS).filter((f) => /^queue-.*\.json$/.test(f))) {
  try {
    const q = JSON.parse(fs.readFileSync(path.join(RUNS, f), 'utf8'));
    for (const r of q.kernels) perStep[r.kernel + '|' + q.scenario] = r.perStep;
  } catch { /* a queue being rewritten */ }
}
const kernels = [];
const all = [];
for (const k of fs.readdirSync(KDIR)) {
  const sp = path.join(KDIR, k, 'state.json'), lp = path.join(KDIR, k, 'ledger.jsonl');
  if (!fs.existsSync(sp)) continue;
  let st;
  try { st = JSON.parse(fs.readFileSync(sp, 'utf8')); } catch { continue; }
  const led = fs.existsSync(lp) ? readLines(lp) : [];
  all.push(...led);
  if (!st.baseline || !st.best) { kernels.push({ k, st, led, perStep: null, savedMs: null, setupOnly: true }); continue; }
  const ps = perStep[k + '|' + st.scenario] ?? null;
  const savedMs = ps != null ? ((st.baseline.us - st.baseline.us / st.speedup) * ps) / 1000 : null;
  kernels.push({ k, st, led, perStep: ps, savedMs });
}
kernels.sort((a, b) => (b.savedMs ?? 0) - (a.savedMs ?? 0) || (b.st.speedup || 1) - (a.st.speedup || 1));
all.sort((a, b) => (String(a.ts) < String(b.ts) ? -1 : 1));
fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(path.join(OUT, 'all.jsonl'), all.map((r) => JSON.stringify(r)).join('\n') + '\n');

function stateOf(st) {
  if (st.landed) return 'landed';
  if (st.done) return 'done: ' + String(st.done.reason).replaceAll('_', ' ');
  if (st.landing) return 'landing';
  if (st.blockedUntil && Date.parse(st.blockedUntil) > Date.now()) return 'waiting to land';
  if (!st.best) return 'not set up';
  if (st.verified?.ok && st.verified.attempt === st.best.attempt) return 'verified';
  return st.attempts - st.lastAccept >= C.stop.patience || st.attempts >= C.stop.maxAttempts ? 'stopped' : 'running';
}
const csvCell = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replaceAll('"', '""')}"` : String(v));
const csvRows = [['kernel', 'unit', 'scenario', 'attempts', 'accepted', 'baseline_us', 'best_us', 'speedup', 'launches_per_run', 'saved_ms_per_run', 'state', 'pr']];
for (const x of kernels) {
  const st = x.st;
  csvRows.push([x.k, st.unit ?? '', st.scenario ?? '', st.attempts ?? 0, (st.accepts || []).length, st.baseline ? st.baseline.us.toFixed(3) : '',
    st.baseline ? (st.baseline.us / st.speedup).toFixed(3) : '', (st.speedup || 1).toFixed(4), x.perStep ?? '', x.savedMs != null ? x.savedMs.toFixed(4) : '', stateOf(st), st.landed?.pr || '']);
}
fs.writeFileSync(path.join(OUT, 'summary.csv'), csvRows.map((r) => r.map(csvCell).join(',')).join('\n') + '\n');

// ---- charts
function kernelChart(x) {
  const W = 360, H = 150, L = 42, R = 8, T = 8, B = 20;
  if (x.setupOnly) return `<svg viewBox="0 0 ${W} ${H}" class="ch"><text x="${W / 2}" y="${H / 2}" class="ax" text-anchor="middle">not set up: ${esc(x.st.done?.reason || '')}</text></svg>`;
  const att = x.led.filter((r) => r.tag && !NOT_ATTEMPTS.has(r.decision));
  const pts = att.filter((r) => r.cand_us);
  const first = x.led.find((r) => r.decision === 'baseline' || r.decision === 'rebase');
  const base = first ? first.best_us : x.st.baseline.us;
  const ys = [base, ...pts.map((r) => r.cand_us), ...x.led.map((r) => r.best_us).filter(Boolean)];
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (hi - lo < base * 0.02) { lo -= base * 0.01; hi += base * 0.01; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const n = Math.max(x.st.attempts || 0, ...att.map((r) => r.attempt), 1);
  const sx = (a) => L + ((W - L - R) * a) / n, sy = (v) => T + ((H - T - B) * (hi - v)) / (hi - lo);
  let s = `<svg viewBox="0 0 ${W} ${H}" class="ch" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${esc(x.k)} attempts">`;
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4, y = sy(v);
    s += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" class="grid"/><text x="${L - 4}" y="${y + 3}" class="ax" text-anchor="end">${v.toFixed(1)}</text>`;
  }
  s += `<text x="${L}" y="${H - 5}" class="ax">attempt</text><text x="${W - R}" y="${H - 5}" class="ax" text-anchor="end">${n}</text>`;
  let d = `M${sx(0)},${sy(base)}`;
  for (const r of x.led.filter((r) => r.decision === 'accepted' || r.decision === 'rebase'))
    d += ` H${sx(r.attempt)} V${sy(r.best_us)}`; /* accepted: the best steps down; rebase: a restart on newer engine code */
  s += `<path d="${d} H${sx(n)}" class="best"/>`;
  for (const r of att.filter((r) => !r.cand_us))
    s += `<path d="M${sx(r.attempt) - 3},${T + 3} l6,6 m0,-6 l-6,6" stroke="${COLORS[r.decision] || '#555'}" stroke-width="1.5"><title>#${r.attempt} ${esc(r.decision)}: ${esc(r.idea)}</title></path>`;
  for (const r of pts)
    s += `<circle cx="${sx(r.attempt)}" cy="${sy(r.cand_us)}" r="${r.decision === 'accepted' ? 4 : 3}" fill="${COLORS[r.decision] || '#555'}"><title>#${r.attempt} [${esc(r.tag)}] ${esc(r.decision)} ${r.delta_pct?.toFixed?.(2) ?? ''}%: ${esc(r.idea)}</title></circle>`;
  return s + '</svg>';
}
function overview() {
  const rows = kernels.filter((x) => !x.setupOnly);
  const W = 560, rowH = 18, L = 220, H = 10 + rowH * Math.max(rows.length, 1);
  const maxSp = Math.max(1.05, ...rows.map((x) => x.st.speedup));
  let s = `<svg viewBox="0 0 ${W} ${H}" class="ov" preserveAspectRatio="xMinYMin meet" role="img" aria-label="speedup per kernel">`;
  rows.forEach((x, i) => {
    const y = 6 + i * rowH, w = ((W - L - 130) * (x.st.speedup - 1)) / (maxSp - 1);
    s += `<text x="${L - 6}" y="${y + 10}" class="lab" text-anchor="end">${esc(short(x.k))}</text>`;
    s += `<rect x="${L}" y="${y}" width="${Math.max(0, w)}" height="${rowH - 5}" fill="${x.st.landed ? '#1a9850' : '#a6d96a'}"/>`;
    s += `<text x="${L + Math.max(0, w) + 4}" y="${y + 10}" class="lab">x${x.st.speedup.toFixed(3)}${x.savedMs != null ? ` · ${x.savedMs.toFixed(3)} ms` : ''}</text>`;
  });
  return s + '</svg>';
}
const totalSaved = kernels.reduce((a, x) => a + (x.st.landed ? x.savedMs ?? 0 : 0), 0);
const pending = kernels.reduce((a, x) => a + (!x.st.landed ? x.savedMs ?? 0 : 0), 0);
const totalAttempts = kernels.reduce((a, x) => a + x.led.filter((r) => r.tag && !NOT_ATTEMPTS.has(r.decision)).length, 0);
const nLanded = kernels.filter((x) => x.st.landed).length;
const legend = Object.entries(COLORS).map(([k, c]) => `<span><i style="background:${c}"></i>${k.replaceAll('_', ' ')}</span>`).join('');
const MAX_CARDS = 24; /* more would not be readable on one screen; the rest are in summary.csv */
const shown = kernels.slice(0, MAX_CARDS);
const cards = shown.map((x) => {
  const st = x.st;
  const tags = {};
  for (const r of x.led.filter((r) => r.tag && !NOT_ATTEMPTS.has(r.decision))) { tags[r.tag] ??= [0, 0]; tags[r.tag][1]++; if (r.decision === 'accepted') tags[r.tag][0]++; }
  const link = st.landed?.pr ? ` <a href="${esc(st.landed.pr)}">PR</a>` : '';
  const line = x.setupOnly ? esc(stateOf(st))
    : `${esc(st.scenario)} · ${st.baseline.us.toFixed(1)} → ${(st.baseline.us / st.speedup).toFixed(1)} µs · <b>x${st.speedup.toFixed(3)}</b> · ${st.attempts} tries, ${(st.accepts || []).length} kept · ${esc(stateOf(st))}${link}`;
  return `<section class="card"><h3 title="${esc(x.k)}">${esc(short(x.k))}</h3><p>${line}</p>${kernelChart(x)}<p class="tags">${Object.entries(tags).map(([t, [a, n]]) => `${t} ${a}/${n}`).join(' · ') || 'no attempts yet'}</p></section>`;
}).join('\n');
// a grid with as many columns as keeps the cards near the chart's shape, every row the same height
const n = Math.max(shown.length, 1);
const cols = Math.min(n, Math.max(1, Math.round(Math.sqrt(n * 1.6))));
const rowsN = Math.ceil(n / cols);
const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kernel optimisation loop</title><style>
*{box-sizing:border-box}html,body{margin:0;height:100%;overflow:hidden}body{font:12px system-ui,sans-serif;color:#222;display:flex;flex-direction:column;padding:8px 12px;gap:6px}
header{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}h1{font-size:16px;margin:0}h3{font-size:12px;margin:0;overflow-wrap:anywhere;line-height:1.2}
.kpi b{font-size:15px}.legend{display:flex;gap:8px;flex-wrap:wrap;font-size:11px}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:3px}
main{flex:1;min-height:0;display:grid;grid-template-columns:minmax(300px,30%) 1fr;gap:10px}
.left{min-height:0;overflow:hidden;display:flex;flex-direction:column;gap:4px}.ov{width:100%;flex:1;min-height:0}
.grid-cards{min-height:0;overflow:hidden;display:grid;grid-template-columns:repeat(${cols},minmax(0,1fr));grid-template-rows:repeat(${rowsN},minmax(0,1fr));gap:6px}
.card{border:1px solid #ddd;border-radius:6px;padding:4px 6px;min-width:0;min-height:0;display:flex;flex-direction:column;overflow:hidden}
.card p{margin:1px 0;color:#555;font-size:${n > 12 ? 10 : 12}px}.ch{width:100%;flex:1;min-height:0;display:block}
.grid{stroke:#eee}.ax{font-size:9px;fill:#666}.lab{font-size:10px;fill:#333}.best{fill:none;stroke:#1a9850;stroke-width:1.5}.tags{font-size:10px}</style></head><body>
<header><h1>Kernel optimisation loop</h1><span class="kpi"><b>${kernels.length}</b> kernels, <b>${nLanded}</b> landed${kernels.length > shown.length ? ` (charts for the top ${shown.length}; all in summary.csv)` : ''}</span><span class="kpi"><b>${totalAttempts}</b> attempts</span>
<span class="kpi"><b>${totalSaved.toFixed(3)} ms</b> saved per run by landed kernels${pending > 0 ? `, ${pending.toFixed(3)} ms more not landed yet` : ''} (kernel time, cold)</span><span class="legend">${legend}</span></header>
<main><div class="left"><h3>Speedup per kernel (bar: x over baseline · ms saved per run; dark: landed)</h3>${overview()}</div><div class="grid-cards">${cards}</div></main>
</body></html>`;
fs.writeFileSync(path.join(OUT, 'report.html'), html);
console.log(`wrote ${path.join(OUT, 'report.html')}, summary.csv, all.jsonl (${kernels.length} kernels, ${all.length} records)`);
