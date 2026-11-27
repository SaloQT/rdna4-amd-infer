#!/usr/bin/env node
// report.mjs -- the kernel-rsi ledgers as charts and flat files.
//   node report.mjs [--out DIR]     (default: the runs directory of rsi.config.json)
// Writes all.jsonl (every attempt of every kernel, one JSON object per line), summary.csv (one row per kernel) and
// report.html (self-contained: inline SVG, no scripts or network).
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

const COLORS = { accepted: '#1a9850', no_gain: '#9e9e9e', slower: '#f46d43', not_exact: '#d73027', not_confirmed: '#fdae61',
  holdout_fail: '#e08214', spill: '#8073ac', compile_fail: '#542788', suspect: '#000000', error: '#67001f' };
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

const perStep = {};
for (const f of fs.readdirSync(RUNS).filter((f) => /^queue-.*\.json$/.test(f))) {
  const q = JSON.parse(fs.readFileSync(path.join(RUNS, f), 'utf8'));
  for (const r of q.kernels) perStep[r.kernel + '|' + q.scenario] = r.perStep;
}
const kernels = [];
const all = [];
for (const k of fs.readdirSync(KDIR)) {
  const sp = path.join(KDIR, k, 'state.json'), lp = path.join(KDIR, k, 'ledger.jsonl');
  if (!fs.existsSync(sp) || !fs.existsSync(lp)) continue;
  const st = JSON.parse(fs.readFileSync(sp, 'utf8'));
  const led = fs.readFileSync(lp, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
  all.push(...led);
  const ps = perStep[k + '|' + st.scenario] ?? null;
  const savedMs = ps != null ? ((st.baseline.us - st.baseline.us / st.speedup) * ps) / 1000 : null;
  kernels.push({ k, st, led, perStep: ps, savedMs });
}
kernels.sort((a, b) => (b.savedMs ?? 0) - (a.savedMs ?? 0) || b.st.speedup - a.st.speedup);
all.sort((a, b) => (a.ts < b.ts ? -1 : 1));
fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(path.join(OUT, 'all.jsonl'), all.map((r) => JSON.stringify(r)).join('\n') + '\n');
const csvRows = [['kernel', 'unit', 'scenario', 'attempts', 'accepted', 'baseline_us', 'best_us', 'speedup', 'launches_per_run', 'saved_ms_per_run', 'verified']];
for (const x of kernels)
  csvRows.push([x.k, x.st.unit, x.st.scenario, x.st.attempts, x.st.accepts.length, x.st.baseline.us.toFixed(3), (x.st.baseline.us / x.st.speedup).toFixed(3),
    x.st.speedup.toFixed(4), x.perStep ?? '', x.savedMs != null ? x.savedMs.toFixed(4) : '', x.st.verified ? (x.st.verified.ok ? 'yes' : 'failed') : 'no']);
fs.writeFileSync(path.join(OUT, 'summary.csv'), csvRows.map((r) => r.join(',')).join('\n') + '\n');

// ---- charts
function kernelChart(x) {
  const W = 360, H = 170, L = 46, R = 8, T = 10, B = 24;
  const pts = x.led.filter((r) => r.attempt > 0 && r.cand_us);
  const base = x.st.baseline.us;
  const ys = [base, ...pts.map((r) => r.cand_us), ...x.led.map((r) => r.best_us).filter(Boolean)];
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (hi - lo < base * 0.02) { lo -= base * 0.01; hi += base * 0.01; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const n = Math.max(x.st.attempts, 1);
  const sx = (a) => L + ((W - L - R) * a) / n, sy = (v) => T + ((H - T - B) * (hi - v)) / (hi - lo);
  let s = `<svg viewBox="0 0 ${W} ${H}" class="ch" role="img" aria-label="${esc(x.k)} attempts">`;
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4, y = sy(v);
    s += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" class="grid"/><text x="${L - 4}" y="${y + 3}" class="ax" text-anchor="end">${v.toFixed(1)}</text>`;
  }
  s += `<text x="${L}" y="${H - 6}" class="ax">attempt</text><text x="${W - R}" y="${H - 6}" class="ax" text-anchor="end">${n}</text>`;
  // best-so-far step line
  let best = base, d = `M${sx(0)},${sy(best)}`;
  for (const r of x.led.filter((r) => r.attempt > 0 && r.decision !== 'verified' && r.decision !== 'verify_fail')) {
    if (r.decision === 'accepted') { d += ` H${sx(r.attempt)} V${sy(r.best_us)}`; best = r.best_us; }
  }
  d += ` H${sx(n)}`;
  s += `<path d="${d}" class="best"/>`;
  for (const r of x.led.filter((r) => r.attempt > 0 && !r.cand_us && r.decision !== 'verified' && r.decision !== 'verify_fail'))
    s += `<path d="M${sx(r.attempt) - 3},${T + 3} l6,6 m0,-6 l-6,6" stroke="${COLORS[r.decision] || '#555'}" stroke-width="1.5"><title>#${r.attempt} ${esc(r.decision)}: ${esc(r.idea)}</title></path>`;
  for (const r of pts)
    s += `<circle cx="${sx(r.attempt)}" cy="${sy(r.cand_us)}" r="${r.decision === 'accepted' ? 4 : 3}" fill="${COLORS[r.decision] || '#555'}"><title>#${r.attempt} [${esc(r.tag)}] ${esc(r.decision)} ${r.delta_pct?.toFixed?.(2) ?? ''}%: ${esc(r.idea)}</title></circle>`;
  return s + '</svg>';
}
function overview() {
  const W = 560, rowH = 18, L = 250, H = 20 + rowH * kernels.length;
  const maxSp = Math.max(1.05, ...kernels.map((x) => x.st.speedup));
  let s = `<svg viewBox="0 0 ${W} ${H}" class="ov" role="img" aria-label="speedup per kernel">`;
  kernels.forEach((x, i) => {
    const y = 14 + i * rowH, w = ((W - L - 120) * (x.st.speedup - 1)) / (maxSp - 1);
    s += `<text x="${L - 6}" y="${y + 10}" class="lab" text-anchor="end">${esc(x.k.replace(/^ie_inkling_/, ''))}</text>`;
    s += `<rect x="${L}" y="${y}" width="${Math.max(0, w)}" height="${rowH - 5}" fill="${x.st.verified?.ok ? '#1a9850' : '#66bd63'}"/>`;
    s += `<text x="${L + Math.max(0, w) + 4}" y="${y + 10}" class="lab">x${x.st.speedup.toFixed(3)}${x.savedMs != null ? ` · ${(x.savedMs).toFixed(3)} ms` : ''}</text>`;
  });
  return s + '</svg>';
}
const totalSaved = kernels.reduce((a, x) => a + (x.savedMs ?? 0), 0);
const totalAttempts = kernels.reduce((a, x) => a + x.st.attempts, 0);
const legend = Object.entries(COLORS).map(([k, c]) => `<span><i style="background:${c}"></i>${k.replace('_', ' ')}</span>`).join('');
const cards = kernels.map((x) => {
  const tags = {};
  for (const r of x.led.filter((r) => r.attempt > 0 && r.tag)) { tags[r.tag] ??= [0, 0]; tags[r.tag][1]++; if (r.decision === 'accepted') tags[r.tag][0]++; }
  const stop = x.st.attempts - x.st.lastAccept >= C.stop.patience || x.st.attempts >= C.stop.maxAttempts;
  return `<section class="card"><h3 title="${esc(x.k)}">${esc(x.k.replace(/^ie_inkling_/, ''))}</h3>
<p>${esc(x.st.scenario)} · ${x.st.baseline.us.toFixed(1)} → ${(x.st.baseline.us / x.st.speedup).toFixed(1)} µs · <b>x${x.st.speedup.toFixed(3)}</b> · ${x.st.attempts} tries, ${x.st.accepts.length} kept · ${x.st.verified?.ok ? 'verified' : stop ? 'stopped' : 'running'}</p>
${kernelChart(x)}<p class="tags">${Object.entries(tags).map(([t, [a, n]]) => `${t} ${a}/${n}`).join(' · ') || 'no attempts yet'}</p></section>`;
}).join('\n');
const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kernel optimisation loop</title><style>
*{box-sizing:border-box}html,body{margin:0;height:100%;overflow:hidden}body{font:12px system-ui,sans-serif;color:#222;display:flex;flex-direction:column;padding:10px 14px;gap:8px}
header{display:flex;align-items:baseline;gap:18px;flex-wrap:wrap}h1{font-size:16px;margin:0}h3{font-size:12px;margin:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi b{font-size:15px}.legend{display:flex;gap:10px;flex-wrap:wrap}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:3px}
main{flex:1;min-height:0;display:grid;grid-template-columns:minmax(320px,34%) 1fr;gap:12px}
.left{min-height:0;overflow:hidden;display:flex;flex-direction:column}.ov{width:100%;height:auto;max-height:100%}
.grid-cards{min-height:0;overflow:hidden;display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));grid-auto-rows:min-content;gap:8px;align-content:start}
.card{border:1px solid #ddd;border-radius:6px;padding:6px 8px;min-width:0}.card p{margin:2px 0;color:#555}
.ch{width:100%;height:auto;display:block}.grid{stroke:#eee}.ax,.lab{font-size:9px;fill:#666}.lab{font-size:10px;fill:#333}.best{fill:none;stroke:#1a9850;stroke-width:1.5}
.tags{font-size:10px}</style></head><body>
<header><h1>Kernel optimisation loop</h1><span class="kpi"><b>${kernels.length}</b> kernels</span><span class="kpi"><b>${totalAttempts}</b> attempts</span>
<span class="kpi"><b>${totalSaved.toFixed(3)} ms</b> saved per run (kernel time, cold)</span><span class="legend">${legend}</span></header>
<main><div class="left"><h3>Speedup per kernel (bar: x over baseline · ms saved per run)</h3>${overview()}</div><div class="grid-cards">${cards}</div></main>
</body></html>`;
fs.writeFileSync(path.join(OUT, 'report.html'), html);
console.log(`wrote ${path.join(OUT, 'report.html')}, summary.csv, all.jsonl (${kernels.length} kernels, ${all.length} records)`);
