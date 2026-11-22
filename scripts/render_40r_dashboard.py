#!/usr/bin/env python3
"""Standalone generator for the Inkling-2-Mini 40-Round Telemetry HTML Dashboard.

Reads results/inkling2/40round-ledger.json and renders a 100% self-contained
dark telemetry HTML dashboard (inline CSS + inline JS + inline JSON, zero CDN links,
works offline via file://) to results/inkling2/40round-dashboard.html.

Usage:
  python bench/render_40r_dashboard.py [--in LEDGER.json] [--out DASHBOARD.html]
"""

import argparse
import json
import os
import sys

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IN = os.path.join(LAB, "results", "inkling2", "40round-ledger.json")
DEFAULT_OUT = os.path.join(LAB, "results", "inkling2", "40round-dashboard.html")

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RX 9070 XT Local LLM Benchmarks (RDNA4 / gfx1201): Inkling-2-Mini &amp; Saluki-27B | rdna4-amd-infer</title>
<meta name="description" content="AMD Radeon RX 9070 XT (RDNA4 / gfx1201) C11 LLM inference engine with 186 zero-spill custom AMDGPU kernels. No ROCm install on Windows. Inkling-2-Mini IQ3_XXS and Saluki-27B Qwen3.8 IQ2-mix benchmarks.">
<meta name="keywords" content="RX 9070 XT LLM, AMD RDNA4 LLM, gfx1201, Inkling-2-Mini 26B IQ3_XXS benchmark, Qwen3.8 27B 16GB VRAM, Saluki 27B IQ2-mix, local LLM Windows no ROCm, GGUF inference engine, DSpark speculative decoding, AMDGPU WMMA, wave32 kernels, 128K context 16GB">
<meta name="robots" content="index, follow">
<link rel="canonical" href="https://saloqt.github.io/rdna4-amd-infer/">
<meta property="og:type" content="website">
<meta property="og:site_name" content="rdna4-amd-infer">
<meta property="og:title" content="RX 9070 XT Local LLM Engine (RDNA4 / gfx1201): Inkling-2-Mini IQ3_XXS &amp; Saluki-27B IQ2-mix">
<meta property="og:description" content="186 zero-spill custom AMDGPU kernels for AMD RX 9070 XT (16GB VRAM). Pure C11, no Windows ROCm install, matched OMP prefill and medium-reasoning decode benchmarks.">
<meta property="og:url" content="https://saloqt.github.io/rdna4-amd-infer/">
<meta property="og:image" content="https://saloqt.github.io/rdna4-amd-infer/progress-chart.svg">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="rdna4-amd-infer: RX 9070 XT (RDNA4) Local LLM Engine &amp; 37-Round Telemetry">
<meta name="twitter:description" content="186 custom gfx1201 kernels on RX 9070 XT 16GB Windows: Inkling-2-Mini IQ3_XXS and Saluki-27B IQ2-mix matched OMP prefill + decode benchmarks. No system ROCm required.">
<meta name="twitter:image" content="https://saloqt.github.io/rdna4-amd-infer/progress-chart.svg">
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "SoftwareSourceCode",
      "name": "rdna4-amd-infer",
      "description": "Pure C11 + AMDGPU gfx1201 local LLM inference engine for AMD Radeon RX 9070 XT (RDNA4) with 186 zero-spill custom kernels, DSpark/MTP speculative decoding, and no system ROCm install required on Windows.",
      "url": "https://saloqt.github.io/rdna4-amd-infer/",
      "codeRepository": "https://github.com/SaloQT/rdna4-amd-infer",
      "programmingLanguage": ["C", "AMDGPU Assembly", "Python"],
      "operatingSystem": ["Windows 11", "Linux"],
      "runtimePlatform": "AMD Radeon RX 9070 XT (RDNA4 / gfx1201, 16GB VRAM)",
      "keywords": "RX 9070 XT, RDNA4, gfx1201, local LLM, GGUF, Inkling-2-Mini, Qwen3.8-27B, Saluki-27B, DSpark, speculative decoding, WMMA"
    },
    {
      "@type": "FAQPage",
      "mainEntity": [
        {
          "@type": "Question",
          "name": "Do I need to install AMD ROCm on Windows to run LLMs on an RX 9070 XT?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "No. rdna4-amd-infer compiles native gfx1201 AMDGPU kernels using user-space rocm-sdk Python wheels on top of standard Windows Adrenalin drivers without a system ROCm toolkit install."
          }
        },
        {
          "@type": "Question",
          "name": "How fast does Inkling-2-Mini (26B IQ3_XXS) run on an AMD RX 9070 XT 16GB?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "On a single RX 9070 XT 16GB on Windows WDDM, rdna4-amd-infer reaches 189.3 tok/s on a medium-reasoning HTML snake prompt (148.2 tok/s at 30K-42K OMP context, 306.4 tok/s short code peak), 3,792 tok/s on 28.7K cold OMP prefill (7.58 s TTFT), and 5,196 tok/s at 4K prefill."
          }
        },
        {
          "@type": "Question",
          "name": "Can Qwen3.8-27B (Saluki-27B IQ2-mix) run 128K context inside 16GB VRAM on RX 9070 XT?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "Yes. With growing q8 KV cache (4.85 GB at 128K vs 12.1 GB FP16), 64 custom gfx1201 kernels, and MTP7 speculative decoding, Saluki-27B hits 142.1 tok/s on medium-reasoning HTML snake (109.7 tok/s avg at 24K-40K OMP context, 166 tok/s writing code), 1,134-1,242 tok/s 23.8K cold OMP prefill, and 0.61 s disk-cache restarts."
          }
        }
      ]
    }
  ]
}
</script>
<style>
:root {
  --bg: #090a0d;
  --bg-elev: #0e1015;
  --card: #111318;
  --card-hover: #171a22;
  --border: #232732;
  --border-hi: #363c4d;
  --text: #f5f7fa;
  --muted: #8c95a6;
  --dim: #636c7e;
  --cyan: #00d4ff;
  --emerald: #00e599;
  --violet: #ff6b35;
  --amber: #ffaa00;
  --rose: #ff3b30;
  --teal: #00d4ff;
  --blue: #00d4ff;
  --font-sans: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  --font-mono: 'Cascadia Code', 'JetBrains Mono', Consolas, ui-monospace, SFMono-Regular, Menlo, monospace;
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  background: radial-gradient(1000px 480px at 12% -12%, rgba(237, 28, 36, 0.11), transparent 58%),
              radial-gradient(850px 440px at 90% -5%, rgba(255, 170, 0, 0.06), transparent 55%),
              var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 13.5px;
  line-height: 1.45;
  -webkit-font-smoothing: antialiased;
}
.wrap {
  max-width: 1580px;
  margin: 0 auto;
  padding: 20px 24px 48px;
}

/* Header */
.hero {
  background: linear-gradient(180deg, #14171f 0%, #101218 100%);
  border: 1px solid var(--border);
  border-top: 3px solid #ed1c24;
  border-radius: 6px;
  padding: 18px 22px;
  margin-bottom: 18px;
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45);
}
.repo-link {
  color: #f5f7fa;
  text-decoration: none;
  font-family: var(--font-mono);
  font-size: 11.5px;
  padding: 4px 10px;
  border-radius: 4px;
  border: 1px solid rgba(237, 28, 36, 0.55);
  background: rgba(237, 28, 36, 0.14);
}
.repo-link:hover { background: rgba(237, 28, 36, 0.28); border-color: #ff3b30; }
.hero-top {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: flex-start;
  gap: 14px;
}
.hero-title {
  margin: 0;
  font-size: 21px;
  font-weight: 700;
  letter-spacing: -0.02em;
  display: flex;
  align-items: center;
  gap: 10px;
}
.live-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--emerald);
  box-shadow: 0 0 12px var(--emerald);
  display: inline-block;
  flex-shrink: 0;
}
.hero-sub {
  margin: 5px 0 0;
  color: var(--muted);
  font-size: 12.5px;
  font-family: var(--font-mono);
}
.hero-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.pill {
  font-family: var(--font-mono);
  font-size: 11.5px;
  padding: 4px 10px;
  border-radius: 4px;
  border: 1px solid var(--border-hi);
  background: rgba(19, 28, 49, 0.85);
  color: var(--text);
  white-space: nowrap;
}
.pill.em { border-color: rgba(34, 197, 94, 0.45); color: #86efac; background: rgba(34, 197, 94, 0.1); }
.pill.am { border-color: rgba(245, 158, 11, 0.45); color: #fcd34d; background: rgba(245, 158, 11, 0.1); }
.pill.cy { border-color: rgba(56, 189, 248, 0.45); color: #7dd3fc; background: rgba(56, 189, 248, 0.1); }
.pill.ro { border-color: rgba(244, 63, 94, 0.45); color: #fda4af; background: rgba(244, 63, 94, 0.1); }
.progress-row {
  margin-top: 14px;
  display: flex;
  align-items: center;
  gap: 14px;
}
.progress-track {
  flex: 1;
  height: 7px;
  background: #090a0d;
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
  display: flex;
}
.progress-fill-prom {
  height: 100%;
  background: linear-gradient(90deg, var(--cyan), var(--emerald));
  transition: width 0.3s ease;
}
.progress-fill-rej {
  height: 100%;
  background: rgba(245, 158, 11, 0.65);
  transition: width 0.3s ease;
}
.progress-label {
  font-family: var(--font-mono);
  font-size: 11.5px;
  color: var(--muted);
  white-space: nowrap;
}

/* Awaiting Banner */
.banner {
  background: rgba(56, 189, 248, 0.08);
  border: 1px solid rgba(56, 189, 248, 0.35);
  border-radius: 10px;
  padding: 12px 16px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  font-size: 13px;
  color: #bae6fd;
}
.banner code {
  font-family: var(--font-mono);
  font-size: 11.5px;
  background: rgba(7, 11, 20, 0.7);
  padding: 2px 6px;
  border-radius: 4px;
  color: var(--text);
}

/* KPI Grid */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}
@media (max-width: 1320px) { .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 720px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.kpi-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 13px 14px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.15s ease, transform 0.15s ease;
}
.kpi-card:hover { border-color: var(--border-hi); transform: translateY(-1px); }
.kpi-card::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--accent, var(--cyan));
  opacity: 0.85;
}
.kpi-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}
.kpi-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted);
}
.kpi-tag {
  font-family: var(--font-mono);
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.05);
  color: var(--muted);
}
.kpi-val {
  font-family: var(--font-mono);
  font-size: 24px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text);
  font-variant-numeric: tabular-nums;
  line-height: 1.15;
}
.kpi-unit {
  font-size: 12px;
  font-weight: 500;
  color: var(--muted);
  margin-left: 4px;
}
.kpi-deltas {
  margin-top: 7px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  font-family: var(--font-mono);
  font-size: 11px;
}
.delta {
  padding: 1px 6px;
  border-radius: 4px;
  font-weight: 600;
}
.delta.pos { background: rgba(34, 197, 94, 0.14); color: #86efac; }
.delta.neg { background: rgba(244, 63, 94, 0.14); color: #fda4af; }
.delta.neu { background: rgba(142, 161, 194, 0.14); color: var(--muted); }
.kpi-sub {
  margin-top: 6px;
  font-family: var(--font-mono);
  font-size: 10.8px;
  color: var(--dim);
  display: flex;
  justify-content: space-between;
  gap: 6px;
}
.kpi-sub strong { color: var(--muted); font-weight: 600; }

/* Historical Collapsible Strip */
.hist-box {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: 12px;
  margin-bottom: 18px;
  overflow: hidden;
}
.hist-box summary {
  list-style: none;
  cursor: pointer;
  padding: 11px 16px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  user-select: none;
  transition: background 0.15s ease;
}
.hist-box summary::-webkit-details-marker { display: none; }
.hist-box summary:hover { background: rgba(255, 255, 255, 0.02); }
.hist-title {
  font-size: 12.5px;
  font-weight: 600;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 8px;
}
.hist-chev {
  display: inline-block;
  transition: transform 0.2s ease;
  color: var(--cyan);
  font-family: var(--font-mono);
}
.hist-box[open] .hist-chev { transform: rotate(90deg); }
.hist-preview {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--muted);
}
.hist-body {
  border-top: 1px solid var(--border);
  padding: 12px 16px 14px;
  overflow-x: auto;
}
.hist-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--font-mono);
  font-size: 11.5px;
}
.hist-table th, .hist-table td {
  text-align: left;
  padding: 7px 10px;
  border-bottom: 1px solid rgba(29, 43, 72, 0.65);
  white-space: nowrap;
}
.hist-table th { color: var(--dim); font-weight: 600; font-size: 10.5px; text-transform: uppercase; }
.hist-table td.notes-cell { white-space: normal; font-family: var(--font-sans); color: var(--muted); min-width: 240px; }
.hist-table tr:last-child td { border-bottom: none; }

/* Charts Toolbar & Grid */
.section-bar {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.section-title {
  font-size: 14px;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 8px;
}
.seg-ctrl {
  display: inline-flex;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 2px;
  gap: 2px;
}
.seg-btn {
  background: transparent;
  border: none;
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 11px;
  padding: 5px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.15s ease;
}
.seg-btn:hover { color: var(--text); }
.seg-btn.active {
  background: var(--card-hover);
  color: var(--cyan);
  box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.35);
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 22px;
}
@media (max-width: 1040px) { .charts-grid { grid-template-columns: 1fr; } }
.chart-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px 12px;
  display: flex;
  flex-direction: column;
}
.chart-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 8px;
}
.chart-title {
  font-size: 13.5px;
  font-weight: 700;
  color: var(--text);
}
.chart-sub {
  font-size: 11px;
  color: var(--dim);
  font-family: var(--font-mono);
}
.chart-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 8px;
}
.leg-item {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-family: var(--font-mono);
  font-size: 10.8px;
  padding: 2px 8px;
  border-radius: 4px;
  border: 1px solid var(--border);
  background: rgba(7, 11, 20, 0.45);
  color: var(--text);
  cursor: pointer;
  user-select: none;
  transition: opacity 0.15s ease, border-color 0.15s ease;
}
.leg-item:hover { border-color: var(--border-hi); }
.leg-item.off { opacity: 0.35; text-decoration: line-through; }
.leg-swatch { width: 10px; height: 3px; border-radius: 2px; display: inline-block; }
.chart-svg-wrap {
  position: relative;
  width: 100%;
  height: 278px;
}
.chart-svg-wrap svg {
  width: 100%;
  height: 100%;
  display: block;
  overflow: visible;
}

/* Floating Tooltip */
.tooltip {
  position: fixed;
  pointer-events: none;
  z-index: 9999;
  background: rgba(9, 14, 27, 0.97);
  border: 1px solid var(--border-hi);
  border-radius: 10px;
  padding: 10px 12px;
  min-width: 245px;
  max-width: 350px;
  box-shadow: 0 14px 36px rgba(0, 0, 0, 0.55);
  font-size: 12px;
  opacity: 0;
  transform: translateY(4px);
  transition: opacity 0.1s ease, transform 0.1s ease;
}
.tooltip.show { opacity: 1; transform: translateY(0); }
.tt-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 6px;
  margin-bottom: 6px;
}
.tt-round { font-family: var(--font-mono); font-weight: 700; color: var(--text); font-size: 12px; }
.tt-name { color: var(--muted); font-size: 11.5px; margin-bottom: 6px; }
.tt-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  font-family: var(--font-mono);
  font-size: 11.5px;
  padding: 2px 0;
}
.tt-key { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); }
.tt-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.tt-hyp {
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid rgba(29, 43, 72, 0.7);
  color: var(--dim);
  font-size: 11px;
  line-height: 1.35;
}

/* Ledger Table */
.ledger-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}
.ledger-controls {
  padding: 13px 16px;
  border-bottom: 1px solid var(--border);
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
}
.filter-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.fbtn {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 11.5px;
  padding: 5px 11px;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s ease;
}
.fbtn:hover { border-color: var(--border-hi); color: var(--text); }
.fbtn.active {
  background: rgba(56, 189, 248, 0.14);
  border-color: rgba(56, 189, 248, 0.55);
  color: #bae6fd;
}
.fbtn .cnt { opacity: 0.75; margin-left: 4px; font-size: 10.5px; }
.search-box {
  display: flex;
  align-items: center;
  gap: 8px;
}
.search-input {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 12px;
  padding: 6px 11px;
  border-radius: 8px;
  width: 240px;
  outline: none;
}
.search-input:focus { border-color: var(--cyan); }
.table-wrap { overflow-x: auto; }
table.ledger {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
table.ledger th {
  position: sticky;
  top: 0;
  background: #0b1222;
  color: var(--dim);
  font-family: var(--font-mono);
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  text-align: left;
  padding: 10px 10px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
table.ledger th.num, table.ledger td.num { text-align: right; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
table.ledger td {
  padding: 9px 10px;
  border-bottom: 1px solid rgba(29, 43, 72, 0.55);
  vertical-align: top;
}
table.ledger tbody tr.main-row { cursor: pointer; transition: background 0.12s ease; }
table.ledger tbody tr.main-row:hover { background: rgba(56, 189, 248, 0.05); }
table.ledger tbody tr.main-row.status-promoted td:first-child { box-shadow: inset 3px 0 0 var(--emerald); }
table.ledger tbody tr.main-row.status-rejected td:first-child { box-shadow: inset 3px 0 0 var(--amber); }
table.ledger tbody tr.main-row.status-baseline td:first-child { box-shadow: inset 3px 0 0 var(--cyan); }
table.ledger tr.detail-row { display: none; background: #090a0d; }
table.ledger tr.detail-row.open { display: table-row; }
table.ledger tr.detail-row td { padding: 12px 16px; border-bottom: 1px solid var(--border); }

.badge {
  display: inline-block;
  font-family: var(--font-mono);
  font-size: 10.2px;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 5px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  white-space: nowrap;
}
.b-promoted { background: rgba(34, 197, 94, 0.16); color: #86efac; border: 1px solid rgba(34, 197, 94, 0.4); }
.b-rejected { background: rgba(245, 158, 11, 0.15); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.4); }
.b-baseline { background: rgba(56, 189, 248, 0.15); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.4); }
.b-exact { background: rgba(34, 197, 94, 0.13); color: #4ade80; }
.b-kld { background: rgba(96, 165, 250, 0.15); color: #93c5fd; }
.b-mismatch { background: rgba(244, 63, 94, 0.18); color: #fda4af; }
.b-target-decode { background: rgba(167, 139, 250, 0.16); color: #c4b5fd; }
.b-target-prefill { background: rgba(56, 189, 248, 0.16); color: #7dd3fc; }
.b-target-ttft { background: rgba(245, 158, 11, 0.16); color: #fcd34d; }
.b-target-all { background: rgba(45, 212, 191, 0.16); color: #5eead4; }

.src-link {
  color: var(--cyan);
  text-decoration: none;
  font-size: 11.5px;
  display: inline-block;
  margin-top: 2px;
}
.src-link:hover { text-decoration: underline; }
.m-main { color: var(--text); font-weight: 600; display: block; white-space: nowrap; }
.m-sub { font-size: 10.5px; display: block; white-space: nowrap; margin-top: 1px; }
.m-sub.pos { color: #86efac; }
.m-sub.neg { color: #fda4af; }
.m-sub.dim { color: var(--dim); }
.hyp-cell { max-width: 280px; color: var(--muted); font-size: 11.8px; }
.name-cell { min-width: 185px; }
.name-cell .rname { font-weight: 600; color: var(--text); display: block; }
.sha-mini { font-family: var(--font-mono); font-size: 10px; color: var(--dim); display: block; margin-top: 3px; }
.detail-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
  font-family: var(--font-mono);
  font-size: 11.2px;
}
.detail-panel {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
}
.detail-panel h4 {
  margin: 0 0 6px;
  font-size: 10.5px;
  text-transform: uppercase;
  color: var(--cyan);
  letter-spacing: 0.05em;
}
.detail-kv { display: flex; justify-content: space-between; gap: 8px; padding: 2px 0; color: var(--muted); }
.detail-kv span:last-child { color: var(--text); text-align: right; word-break: break-all; }
.empty-state {
  text-align: center;
  padding: 36px 20px;
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 12.5px;
}
.foot {
  margin-top: 22px;
  text-align: center;
  color: var(--dim);
  font-family: var(--font-mono);
  font-size: 11px;
}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <div class="hero-top">
      <div>
        <h1 class="hero-title"><span class="live-dot" id="liveDot"></span><span id="campaignTitle">Inkling-2-Mini 40-Round Optimization Campaign</span></h1>
        <p class="hero-sub" id="campaignSub"></p>
      </div>
      <div class="hero-pills" id="heroPills"></div>
      <a class="repo-link" href="https://github.com/SaloQT/rdna4-amd-infer" rel="noopener">GitHub Repo &rarr;</a>
    </div>
    <div class="progress-row">
      <div class="progress-track">
        <div class="progress-fill-prom" id="progProm" style="width:0%"></div>
        <div class="progress-fill-rej" id="progRej" style="width:0%"></div>
      </div>
      <div class="progress-label" id="progLabel">0 / 40 rounds</div>
    </div>
  </header>

  <div class="banner" id="awaitBanner" style="display:none;">
    <div>
      <strong>Awaiting Round 0 (Phase 7) Live Baseline:</strong>
      Historical milestones are loaded below. Run <code>bench/inkling2-40r-task.ps1</code> inside
      <code>bench/run-memory-bounded.ps1 -BudgetGiB 20</code> with <code>--record-round --round 0</code> to populate R00.
    </div>
    <span class="pill cy">READY</span>
  </div>

  <section class="kpi-grid" id="kpiGrid"></section>

  <details class="hist-box" id="histBox">
    <summary>
      <span class="hist-title"><span class="hist-chev">&#9656;</span> Historical Pre-Campaign Milestones (First Impl v0 &rarr; R1-R4 &rarr; Phase 5 &rarr; Phase 6 &rarr; Phase 7 Ref)</span>
      <span class="hist-preview" id="histPreview"></span>
    </summary>
    <div class="hist-body">
      <table class="hist-table" id="histTable"></table>
    </div>
  </details>

  <div class="section-bar">
    <div class="section-title">Telemetry Frontiers (R00 Phase 7 Baseline &rarr; R40) <span class="pill" id="chartHint">Solid step-line = Promoted Frontier &middot; Green = Promoted &middot; Amber = Rejected</span></div>
    <div class="seg-ctrl" id="axisToggle">
      <button class="seg-btn active" data-axis="full">Full R00-R40 Axis</button>
      <button class="seg-btn" data-axis="fit">Zoom Populated</button>
    </div>
  </div>

  <section class="charts-grid">
    <div class="chart-card">
      <div class="chart-head">
        <span class="chart-title">1. Decode Throughput (tok/s)</span>
        <span class="chart-sub">Higher is better &middot; true trace-off wall clock</span>
      </div>
      <div class="chart-legend" id="leg-decode"></div>
      <div class="chart-svg-wrap" id="chart-decode"></div>
    </div>

    <div class="chart-card">
      <div class="chart-head">
        <span class="chart-title">2. Prefill Throughput (tok/s)</span>
        <span class="chart-sub">Higher is better &middot; chunk=512 f16 KV</span>
      </div>
      <div class="chart-legend" id="leg-prefill"></div>
      <div class="chart-svg-wrap" id="chart-prefill"></div>
    </div>

    <div class="chart-card">
      <div class="chart-head">
        <span class="chart-title">3. TTFT Latency (ms)</span>
        <span class="chart-sub">Lower is better &middot; cold 4K, cold 16K &amp; warm disk restart</span>
      </div>
      <div class="chart-legend" id="leg-ttft"></div>
      <div class="chart-svg-wrap" id="chart-ttft"></div>
    </div>

    <div class="chart-card">
      <div class="chart-head">
        <span class="chart-title">4. Efficiency &amp; Systems</span>
        <span class="chart-sub">Launches/step &amp; VRAM Free (left) &middot; % Frontier Gain vs R00 (right)</span>
      </div>
      <div class="chart-legend" id="leg-sys"></div>
      <div class="chart-svg-wrap" id="chart-sys"></div>
    </div>
  </section>

  <section class="ledger-card">
    <div class="ledger-controls">
      <div class="filter-pills" id="filterPills"></div>
      <div class="search-box">
        <input class="search-input" id="searchInput" type="search" placeholder="Filter name, wave, source, hypothesis...">
        <button class="fbtn" id="sortBtn" title="Toggle round sort order">Order: R00 &uarr;</button>
      </div>
    </div>
    <div class="table-wrap">
      <table class="ledger">
        <thead>
          <tr>
            <th>Round</th>
            <th>Target</th>
            <th>Experiment &amp; Research Source</th>
            <th>Hypothesis</th>
            <th>Verdict</th>
            <th>Parity / SHA</th>
            <th class="num">Plain T=1</th>
            <th class="num">DS5 233</th>
            <th class="num">DS5 4K</th>
            <th class="num">PF 4K</th>
            <th class="num">PF 16K</th>
            <th class="num">TTFT (ms)</th>
            <th class="num">Sys / Serve</th>
          </tr>
        </thead>
        <tbody id="ledgerBody"></tbody>
      </table>
    </div>
  </section>

  <footer class="foot" id="footNote"></footer>
</div>

<div class="tooltip" id="tooltip"></div>

<script id="ledger-data" type="application/json">__LEDGER_JSON__</script>
<script>
(function () {
  "use strict";

  const DATA = JSON.parse(document.getElementById("ledger-data").textContent);
  const ROUNDS = (DATA.rounds || []).slice().sort((a, b) => (a.round || 0) - (b.round || 0));
  const HIST = DATA.historical_milestones || [];
  const CAMP = DATA.campaign || {};
  const TARGET_ROUNDS = CAMP.target_rounds || 40;

  // Historical anchors from verified baselines
  const STOCK = HIST.find((h) => h.id === "stock_r0") || { t1_tok_s: 80.0, pf4k_tok_s: 1118.0, ttft_4k_ms: 3663.7, launches_per_step: 2222 };
  const PHASE5 = HIST.find((h) => h.id === "phase5") || { serve_sampled_tok_s: 118.2, warm_ttft_4k_ms: 2033.0 };
  const PHASE6 = HIST.find((h) => h.id === "phase6") || { ds5_4k_tok_s: 125.07, serve_sampled_tok_s: 130.4 };
  const PHASE7_REF = HIST.find((h) => h.id === "phase7_ref") || {};
  const R00 = ROUNDS.find((r) => r.round === 0) || null;
  const BASE_M = R00 ? (R00.metrics || {}) : PHASE7_REF;

  const STATE = {
    axisMode: "full",
    filter: "all",
    search: "",
    sortAsc: true,
    hiddenSeries: {},
    expanded: {}
  };

  function isNum(v) { return typeof v === "number" && Number.isFinite(v); }
  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
  function fmt(v, d = 1) {
    if (!isNum(v)) return "--";
    if (Math.abs(v) >= 1000 && d === 0) return Math.round(v).toLocaleString("en-US");
    return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  function pctGain(cur, base, lowerBetter) {
    if (!isNum(cur) || !isNum(base) || base === 0) return null;
    return lowerBetter ? ((base - cur) / base) * 100.0 : ((cur - base) / base) * 100.0;
  }
  function fmtDelta(p) {
    if (!isNum(p)) return { text: "--", cls: "neu" };
    const sign = p >= 0 ? "+" : "";
    const cls = Math.abs(p) < 0.05 ? "neu" : (p > 0 ? "pos" : "neg");
    return { text: sign + p.toFixed(1) + "%", cls: cls };
  }
  function isAcceptedStatus(st) {
    return st === "baseline" || st === "promoted";
  }

  // Compute composite % frontier gain vs R00 for each round
  function computeCompositeGain(metrics, base) {
    const comps = [
      pctGain(metrics.t1_tok_s, base.t1_tok_s, false),
      pctGain(metrics.ds5_4k_tok_s, base.ds5_4k_tok_s, false),
      pctGain(metrics.ds5_233_tok_s, base.ds5_233_tok_s, false),
      pctGain(metrics.pf4k_tok_s, base.pf4k_tok_s, false),
      pctGain(metrics.pf16k_tok_s, base.pf16k_tok_s, false),
      pctGain(metrics.ttft_4k_ms, base.ttft_4k_ms, true),
      pctGain(metrics.warm_ttft_4k_ms, base.warm_ttft_4k_ms, true)
    ].filter((x) => isNum(x));
    if (!comps.length) return null;
    return comps.reduce((a, b) => a + b, 0) / comps.length;
  }

  // Precompute frontier & composite gain series
  const FRONTIER = {};
  const METRIC_SPECS = {
    t1_tok_s: { lower: false },
    ds5_4k_tok_s: { lower: false },
    ds5_233_tok_s: { lower: false },
    serve_sampled_tok_s: { lower: false },
    pf4k_tok_s: { lower: false },
    pf16k_tok_s: { lower: false },
    ttft_4k_ms: { lower: true },
    ttft_16k_ms: { lower: true },
    warm_ttft_4k_ms: { lower: true },
    launches_per_step: { lower: true },
    vram_free_mb: { lower: false },
    arena_mb: { lower: true }
  };

  function computeBest(key) {
    const lower = (METRIC_SPECS[key] || {}).lower || false;
    let bestVal = null;
    let bestRound = null;
    for (const r of ROUNDS) {
      if (!isAcceptedStatus(r.status)) continue;
      const v = (r.metrics || {})[key];
      if (!isNum(v)) continue;
      if (bestVal === null || (lower ? v < bestVal : v > bestVal)) {
        bestVal = v;
        bestRound = r.round;
      }
    }
    return { val: bestVal, round: bestRound };
  }

  // Attach derived composite_gain_pct on frontier snapshot per round
  const runningBest = {};
  for (const r of ROUNDS) {
    const m = r.metrics || {};
    if (isAcceptedStatus(r.status)) {
      for (const k of Object.keys(METRIC_SPECS)) {
        const v = m[k];
        if (!isNum(v)) continue;
        const lower = METRIC_SPECS[k].lower;
        if (runningBest[k] === undefined || (lower ? v < runningBest[k] : v > runningBest[k])) {
          runningBest[k] = v;
        }
      }
    }
    r._frontier_gain_pct = isAcceptedStatus(r.status)
      ? computeCompositeGain(runningBest, BASE_M)
      : computeCompositeGain(m, BASE_M);
    r._round_gain_pct = computeCompositeGain(m, BASE_M);
  }
  for (const k of Object.keys(METRIC_SPECS)) FRONTIER[k] = computeBest(k);

  // ---------------- HEADER & PROGRESS ----------------
  function renderHeader() {
    document.getElementById("campaignTitle").textContent = CAMP.title || "Inkling-2-Mini 40-Round Optimization Campaign";
    document.getElementById("campaignSub").textContent =
      (CAMP.hardware || "RX 9070 XT gfx1201") + "  |  " + (CAMP.model || "Inkling-2-Mini-fast") + "  |  " + (CAMP.engine || "amd-infer");

    const promoted = ROUNDS.filter((r) => r.status === "promoted").length;
    const rejected = ROUNDS.filter((r) => r.status === "rejected").length;
    const optRounds = ROUNDS.filter((r) => r.round > 0).length;
    const exactOk = ROUNDS.filter((r) => r.parity && (r.parity.status === "EXACT" || r.parity.status === "BASELINE" || r.parity.status === "KLD-OK")).length;
    const latestMask = (ROUNDS.slice().reverse().find((r) => isAcceptedStatus(r.status) && r.ink_mask) || ROUNDS.slice().reverse().find((r) => r.ink_mask) || {}).ink_mask || "0x67dfc0369c3ffff";

    const pills = [
      `<span class="pill cy">MASK ${esc(latestMask)}</span>`,
      `<span class="pill em">PROMOTED ${promoted}</span>`,
      `<span class="pill am">REJECTED ${rejected}</span>`,
      `<span class="pill ${exactOk === ROUNDS.length ? "em" : "ro"}">PARITY ${exactOk}/${ROUNDS.length}</span>`,
      `<span class="pill">UPDATED ${esc((CAMP.updated_utc || "--").replace("T", " ").replace("Z", " UTC"))}</span>`
    ];
    document.getElementById("heroPills").innerHTML = pills.join("");

    const promPct = Math.min(100, (promoted / TARGET_ROUNDS) * 100);
    const rejPct = Math.min(100 - promPct, (rejected / TARGET_ROUNDS) * 100);
    document.getElementById("progProm").style.width = promPct.toFixed(2) + "%";
    document.getElementById("progRej").style.width = rejPct.toFixed(2) + "%";
    document.getElementById("progLabel").textContent =
      `${optRounds} / ${TARGET_ROUNDS} optimization rounds` + (R00 ? " (+ R00 baseline)" : " (awaiting R00)");

    const banner = document.getElementById("awaitBanner");
    banner.style.display = R00 ? "none" : "flex";
    if (!R00) {
      document.getElementById("liveDot").style.background = "var(--cyan)";
      document.getElementById("liveDot").style.boxShadow = "0 0 12px var(--cyan)";
    }
  }

  // ---------------- KPI CARDS ----------------
  function renderKPIs() {
    const bT1 = FRONTIER.t1_tok_s;
    const bDs4k = FRONTIER.ds5_4k_tok_s;
    const bDs233 = FRONTIER.ds5_233_tok_s;
    const bServe = FRONTIER.serve_sampled_tok_s;
    const bPf4k = FRONTIER.pf4k_tok_s;
    const bPf16k = FRONTIER.pf16k_tok_s;
    const bTtft4k = FRONTIER.ttft_4k_ms;
    const bTtft16k = FRONTIER.ttft_16k_ms;
    const bWarm = FRONTIER.warm_ttft_4k_ms;
    const bLaunch = FRONTIER.launches_per_step;
    const bVram = FRONTIER.vram_free_mb;
    const bArena = FRONTIER.arena_mb;

    function kpiCard(opts) {
      const valStr = isNum(opts.val) ? fmt(opts.val, opts.dec) : (isNum(opts.refFallback) ? fmt(opts.refFallback, opts.dec) : "--");
      const rBadge = opts.round !== null && opts.round !== undefined ? ("R" + String(opts.round).padStart(2, "0")) : (R00 ? "--" : "P7 REF");
      const dR00 = fmtDelta(pctGain(opts.val, opts.r00, opts.lower));
      const dStock = isNum(opts.val) && isNum(opts.stock)
        ? (opts.lower
            ? ((opts.stock / opts.val).toFixed(2) + "x faster vs " + opts.stockLabel)
            : ((opts.val / opts.stock).toFixed(2) + "x vs " + opts.stockLabel))
        : ("ref " + opts.stockLabel);

      return `
        <div class="kpi-card" style="--accent:${opts.accent}">
          <div class="kpi-head">
            <span class="kpi-label">${esc(opts.label)}</span>
            <span class="kpi-tag">${esc(rBadge)}</span>
          </div>
          <div class="kpi-val">${valStr}<span class="kpi-unit">${esc(opts.unit)}</span></div>
          <div class="kpi-deltas">
            <span class="delta ${dR00.cls}">${R00 ? "vs R00 " + dR00.text : "R00 pending"}</span>
            <span style="color:var(--muted);font-size:10.5px;">${esc(dStock)}</span>
          </div>
          <div class="kpi-sub">
            <span>${esc(opts.subL || "")}</span>
            <strong>${esc(opts.subR || "")}</strong>
          </div>
        </div>
      `;
    }

    const latestAccepted = ROUNDS.slice().reverse().find((r) => isAcceptedStatus(r.status));
    const compGain = latestAccepted && isNum(latestAccepted._frontier_gain_pct) ? latestAccepted._frontier_gain_pct : 0;

    const cards = [
      kpiCard({
        label: "Plain T=1 Decode",
        accent: "var(--cyan)",
        val: bT1.val,
        round: bT1.round,
        refFallback: PHASE7_REF.t1_tok_s,
        r00: BASE_M.t1_tok_s,
        stock: STOCK.t1_tok_s || 80,
        stockLabel: "80 v0 impl",
        unit: "tok/s",
        dec: 2,
        lower: false,
        subL: "R00: " + (isNum(BASE_M.t1_tok_s) ? fmt(BASE_M.t1_tok_s, 2) + " tok/s" : "--"),
        subR: isNum(bT1.val) ? fmt(1000 / bT1.val, 2) + " ms/t" : ""
      }),
      kpiCard({
        label: "DSpark 4K Decode",
        accent: "var(--emerald)",
        val: bDs4k.val,
        round: bDs4k.round,
        refFallback: PHASE7_REF.ds5_4k_tok_s,
        r00: BASE_M.ds5_4k_tok_s,
        stock: PHASE6.ds5_4k_tok_s || 125.07,
        stockLabel: "125 P6",
        unit: "tok/s",
        dec: 2,
        lower: false,
        subL: "R00: " + (isNum(BASE_M.ds5_4k_tok_s) ? fmt(BASE_M.ds5_4k_tok_s, 2) + " tok/s" : "--"),
        subR: "5x0.6 split"
      }),
      kpiCard({
        label: "DSpark 233 / Live Chat",
        accent: "var(--violet)",
        val: bDs233.val,
        round: bDs233.round,
        refFallback: PHASE7_REF.ds5_233_tok_s,
        r00: BASE_M.ds5_233_tok_s,
        stock: STOCK.t1_tok_s || 80,
        stockLabel: "80 v0 impl",
        unit: "tok/s",
        dec: 1,
        lower: false,
        subL: "Sampled Chat: " + (isNum(bServe.val) ? fmt(bServe.val, 1) + " tok/s" : (PHASE6.serve_sampled_tok_s || 130.4) + " P6"),
        subR: "P5 Fable: " + (PHASE5.serve_sampled_tok_s || 118.2)
      }),
      kpiCard({
        label: "4K Cold Prefill",
        accent: "var(--blue)",
        val: bPf4k.val,
        round: bPf4k.round,
        refFallback: PHASE7_REF.pf4k_tok_s,
        r00: BASE_M.pf4k_tok_s,
        stock: STOCK.pf4k_tok_s || 1118,
        stockLabel: "1118 v0 impl",
        unit: "tok/s",
        dec: 0,
        lower: false,
        subL: "16K PF: " + (isNum(bPf16k.val) ? fmt(bPf16k.val, 0) + " tok/s" : "--"),
        subR: "chunk=512"
      }),
      kpiCard({
        label: "Cold 4K TTFT",
        accent: "var(--amber)",
        val: bTtft4k.val,
        round: bTtft4k.round,
        refFallback: PHASE7_REF.ttft_4k_ms,
        r00: BASE_M.ttft_4k_ms,
        stock: STOCK.ttft_4k_ms || 3663.7,
        stockLabel: "3664ms v0 impl",
        unit: "ms",
        dec: 1,
        lower: true,
        subL: "Warm 4K: " + (isNum(bWarm.val) ? fmt(bWarm.val, 0) + "ms" : (isNum(PHASE7_REF.warm_ttft_4k_ms) ? fmt(PHASE7_REF.warm_ttft_4k_ms, 0) + "ms P7" : "--")),
        subR: "16K: " + (isNum(bTtft16k.val) ? fmt(bTtft16k.val, 0) + "ms" : "--")
      }),
      kpiCard({
        label: "Launches & VRAM",
        accent: "var(--teal)",
        val: bLaunch.val,
        round: bLaunch.round,
        refFallback: PHASE7_REF.launches_per_step,
        r00: BASE_M.launches_per_step,
        stock: STOCK.launches_per_step || 2222,
        stockLabel: "2222 v0 impl",
        unit: "/step",
        dec: 0,
        lower: true,
        subL: "Free: " + (isNum(bVram.val) ? fmt(bVram.val, 0) + " MB" : "--") + (isNum(bArena.val) ? " | Arena " + fmt(bArena.val, 0) + " MiB" : ""),
        subR: "Frontier " + (compGain >= 0 ? "+" : "") + compGain.toFixed(1) + "%"
      })
    ];
    document.getElementById("kpiGrid").innerHTML = cards.join("");
  }

  // ---------------- HISTORICAL STRIP ----------------
  function renderHistorical() {
    const preview = [
      `First Impl (v0): 80 tok/s · 1,118 PF`,
      `R1-R3: 4,563 PF (4.08x)`,
      `Phase 5: 118.2 sampled`,
      `Phase 6: 130.4 sampled · 264.5 DS5`,
      `Phase 7 Ref: 105.8 T=1 · 142.0 DS4K · 1,574 launches`
    ];
    document.getElementById("histPreview").innerHTML = preview.map((p) => `<span class="pill">${esc(p)}</span>`).join("");

    let html = `<thead><tr>
      <th>Stage</th><th>Date</th><th>Plain T=1</th><th>DS5 4K</th><th>DS5 233</th>
      <th>Sampled Chat</th><th>4K Prefill</th><th>Cold 4K TTFT</th><th>Warm 4K TTFT</th><th>Launches</th><th>Key Engineering Milestones</th>
    </tr></thead><tbody>`;
    for (const h of HIST) {
      html += `<tr>
        <td><strong>${esc(h.phase || h.id)}</strong></td>
        <td style="color:var(--dim)">${esc(h.date || "--")}</td>
        <td>${isNum(h.t1_tok_s) ? fmt(h.t1_tok_s, 2) + " tok/s" : "--"}</td>
        <td>${isNum(h.ds5_4k_tok_s) ? fmt(h.ds5_4k_tok_s, 2) + " tok/s" : "--"}</td>
        <td>${isNum(h.ds5_233_tok_s) ? fmt(h.ds5_233_tok_s, 1) + " tok/s" : "--"}</td>
        <td>${isNum(h.serve_sampled_tok_s) ? fmt(h.serve_sampled_tok_s, 1) + " tok/s" : "--"}</td>
        <td>${isNum(h.pf4k_tok_s) ? fmt(h.pf4k_tok_s, 0) + " tok/s" : "--"}</td>
        <td>${isNum(h.ttft_4k_ms) ? fmt(h.ttft_4k_ms, 1) + " ms" : "--"}</td>
        <td>${isNum(h.warm_ttft_4k_ms) ? fmt(h.warm_ttft_4k_ms, 0) + " ms" : "--"}</td>
        <td>${isNum(h.launches_per_step) ? fmt(h.launches_per_step, 0) : "--"}</td>
        <td class="notes-cell">${esc(h.notes || "")}</td>
      </tr>`;
    }
    html += "</tbody>";
    document.getElementById("histTable").innerHTML = html;
  }

  // ---------------- INTERACTIVE SVG CHARTS ----------------
  const CHART_DEFS = [
    {
      id: "decode",
      el: "chart-decode",
      leg: "leg-decode",
      unit: "tok/s",
      lower: false,
      series: [
        { key: "t1_tok_s", label: "Plain T=1", color: "#ffaa00", dec: 2 },
        { key: "ds5_4k_tok_s", label: "DSpark 4K", color: "#00d4ff", dec: 2 },
        { key: "ds5_233_tok_s", label: "DSpark 233", color: "#00e599", dec: 1 },
        { key: "serve_sampled_tok_s", label: "Live Sampled Chat", color: "#ff6b35", dec: 1 }
      ]
    },
    {
      id: "prefill",
      el: "chart-prefill",
      leg: "leg-prefill",
      unit: "tok/s",
      lower: false,
      series: [
        { key: "pf4k_tok_s", label: "4K Cold Prefill", color: "#00d4ff", dec: 0 },
        { key: "pf16k_tok_s", label: "16K Long-Context Prefill", color: "#ffaa00", dec: 0 }
      ]
    },
    {
      id: "ttft",
      el: "chart-ttft",
      leg: "leg-ttft",
      unit: "ms",
      lower: true,
      series: [
        { key: "ttft_4k_ms", label: "Cold 4K TTFT", color: "#00d4ff", dec: 1 },
        { key: "ttft_16k_ms", label: "Cold 16K TTFT", color: "#ff6b35", dec: 1 },
        { key: "warm_ttft_4k_ms", label: "Warm Restart 4K TTFT", color: "#00e599", dec: 0 }
      ]
    },
    {
      id: "sys",
      el: "chart-sys",
      leg: "leg-sys",
      unit: "",
      dualAxis: true,
      series: [
        { key: "launches_per_step", label: "Launches/step", color: "#ff3b30", dec: 0, lower: true, axis: "left", unit: "" },
        { key: "vram_free_mb", label: "VRAM Free (MiB)", color: "#00d4ff", dec: 0, lower: false, axis: "left", unit: " MB" },
        { key: "_frontier_gain_pct", label: "% Gain vs R00 Frontier", color: "#00e599", dec: 1, lower: false, axis: "right", unit: "%", derived: true }
      ]
    }
  ];

  function getMetricVal(r, s) {
    if (s.derived) return r[s.key];
    return (r.metrics || {})[s.key];
  }

  function niceBounds(vals, forceZero) {
    if (!vals.length) return [0, 100];
    let mn = Math.min.apply(null, vals);
    let mx = Math.max.apply(null, vals);
    if (mn === mx) {
      const pad = Math.max(Math.abs(mx) * 0.15, 5);
      mn = forceZero ? Math.min(0, mn - pad) : mn - pad;
      mx = mx + pad;
    } else {
      const span = mx - mn;
      mn = forceZero ? Math.min(0, mn - span * 0.08) : Math.max(0, mn - span * 0.14);
      mx = mx + span * 0.16;
    }
    return [mn, mx];
  }

  function buildFrontierStepPath(points, xScale, yScale, lower) {
    const accepted = points.filter((p) => isAcceptedStatus(p.r.status));
    if (!accepted.length) return { d: "", bestPoints: [] };
    let best = null;
    const coords = [];
    const bestPoints = [];
    for (const p of accepted) {
      const improved = best === null || (lower ? p.val < best : p.val > best);
      if (improved) {
        best = p.val;
        bestPoints.push(p);
      }
      coords.push({ x: xScale(p.r.round), y: yScale(best), val: best, round: p.r.round });
    }
    let d = `M ${coords[0].x.toFixed(1)} ${coords[0].y.toFixed(1)}`;
    for (let i = 1; i < coords.length; i++) {
      // Horizontal step then vertical jump
      d += ` L ${coords[i].x.toFixed(1)} ${coords[i - 1].y.toFixed(1)} L ${coords[i].x.toFixed(1)} ${coords[i].y.toFixed(1)}`;
    }
    return { d: d, bestPoints: bestPoints };
  }

  function renderLegend(def) {
    const el = document.getElementById(def.leg);
    el.innerHTML = def.series.map((s) => {
      const off = !!STATE.hiddenSeries[def.id + ":" + s.key];
      return `<button type="button" class="leg-item ${off ? "off" : ""}" data-chart="${def.id}" data-key="${s.key}">
        <span class="leg-swatch" style="background:${s.color}"></span>${esc(s.label)}
      </button>`;
    }).join("");
    el.querySelectorAll(".leg-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        const k = btn.dataset.chart + ":" + btn.dataset.key;
        STATE.hiddenSeries[k] = !STATE.hiddenSeries[k];
        renderChart(def);
      });
    });
  }

  const tooltipEl = document.getElementById("tooltip");
  function showTooltip(evt, def, roundObj) {
    const statusCls = roundObj.status === "promoted" ? "b-promoted" : (roundObj.status === "rejected" ? "b-rejected" : "b-baseline");
    const rows = def.series.map((s) => {
      const v = getMetricVal(roundObj, s);
      if (!isNum(v)) return "";
      const lower = s.lower !== undefined ? s.lower : !!def.lower;
      const baseV = s.derived ? 0 : BASE_M[s.key];
      const deltaStr = s.derived
        ? ""
        : (R00 && isNum(baseV) && roundObj.round > 0 ? " (" + fmtDelta(pctGain(v, baseV, lower)).text + ")" : "");
      const unit = s.unit !== undefined ? s.unit : (def.unit ? " " + def.unit : "");
      return `<div class="tt-row">
        <span class="tt-key"><span class="tt-dot" style="background:${s.color}"></span>${esc(s.label)}</span>
        <span style="color:var(--text)">${fmt(v, s.dec)}${esc(unit)}<span style="color:var(--muted);font-size:10.5px">${esc(deltaStr)}</span></span>
      </div>`;
    }).filter(Boolean).join("");

    const paritySt = (roundObj.parity && roundObj.parity.status) || "--";
    tooltipEl.innerHTML = `
      <div class="tt-head">
        <span class="tt-round">R${String(roundObj.round).padStart(2, "0")}${roundObj.wave ? " · W" + esc(roundObj.wave) : ""}</span>
        <span><span class="badge ${statusCls}">${esc(roundObj.status)}</span></span>
      </div>
      <div class="tt-name">${esc(roundObj.name || "Untitled")} <span style="color:var(--dim)">[${esc(paritySt)}]</span></div>
      ${rows || '<div class="tt-row"><span class="tt-key">No metrics in this chart</span></div>'}
      ${roundObj.hypothesis ? `<div class="tt-hyp">${esc(roundObj.hypothesis.slice(0, 150))}${roundObj.hypothesis.length > 150 ? "..." : ""}</div>` : ""}
    `;
    tooltipEl.classList.add("show");
    const pad = 16;
    const tw = tooltipEl.offsetWidth || 270;
    const th = tooltipEl.offsetHeight || 150;
    let x = evt.clientX + 16;
    let y = evt.clientY - 20;
    if (x + tw + pad > window.innerWidth) x = evt.clientX - tw - 16;
    if (y + th + pad > window.innerHeight) y = window.innerHeight - th - pad;
    if (y < pad) y = pad;
    tooltipEl.style.left = x + "px";
    tooltipEl.style.top = y + "px";
  }

  function hideTooltip() {
    tooltipEl.classList.remove("show");
  }

  function renderChart(def) {
    renderLegend(def);
    const wrap = document.getElementById(def.el);
    const W = Math.max(480, wrap.clientWidth || 640);
    const H = 278;
    const padL = 58;
    const padR = def.dualAxis ? 52 : 20;
    const padT = 18;
    const padB = 34;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const activeSeries = def.series.filter((s) => !STATE.hiddenSeries[def.id + ":" + s.key]);
    const maxObserved = ROUNDS.length ? Math.max.apply(null, ROUNDS.map((r) => r.round || 0)) : 0;
    const maxX = STATE.axisMode === "full" ? TARGET_ROUNDS : Math.max(4, maxObserved + 1);
    const xScale = (r) => padL + (maxX === 0 ? 0 : (r / maxX) * plotW);

    const leftVals = [];
    const rightVals = [];
    for (const s of activeSeries) {
      for (const r of ROUNDS) {
        const v = getMetricVal(r, s);
        if (!isNum(v)) continue;
        if (def.dualAxis && s.axis === "right") rightVals.push(v);
        else leftVals.push(v);
      }
    }

    const [yMinL, yMaxL] = niceBounds(leftVals, false);
    const [yMinR, yMaxR] = niceBounds(rightVals, true);
    const yScaleL = (v) => padT + plotH - ((v - yMinL) / (yMaxL - yMinL || 1)) * plotH;
    const yScaleR = (v) => padT + plotH - ((v - yMinR) / (yMaxR - yMinR || 1)) * plotH;

    let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
    // Gridlines (5 ticks)
    const ticks = 5;
    for (let i = 0; i <= ticks; i++) {
      const f = i / ticks;
      const y = padT + plotH - f * plotH;
      const lv = yMinL + f * (yMaxL - yMinL);
      svg += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" stroke="#1b2742" stroke-dasharray="3,4" stroke-width="1"/>`;
      svg += `<text x="${padL - 8}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" fill="#6f84ab" font-family="var(--font-mono)" font-size="10.5">${fmt(lv, lv >= 100 ? 0 : 1)}</text>`;
      if (def.dualAxis && rightVals.length) {
        const rv = yMinR + f * (yMaxR - yMinR);
        svg += `<text x="${W - padR + 8}" y="${(y + 3.5).toFixed(1)}" text-anchor="start" fill="#4ade80" font-family="var(--font-mono)" font-size="10.5">${rv >= 0 ? "+" : ""}${fmt(rv, 1)}%</text>`;
      }
    }

    // X-axis ticks
    const stepX = maxX <= 10 ? 1 : (maxX <= 20 ? 2 : 5);
    for (let x = 0; x <= maxX; x += stepX) {
      const px = xScale(x);
      svg += `<line x1="${px.toFixed(1)}" y1="${padT + plotH}" x2="${px.toFixed(1)}" y2="${padT + plotH + 4}" stroke="#2e4370" stroke-width="1"/>`;
      svg += `<text x="${px.toFixed(1)}" y="${padT + plotH + 18}" text-anchor="middle" fill="#6f84ab" font-family="var(--font-mono)" font-size="10">R${String(x).padStart(2, "0")}</text>`;
    }
    svg += `<line x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" stroke="#2e4370" stroke-width="1.2"/>`;

    if (!ROUNDS.length || (!leftVals.length && !rightVals.length)) {
      svg += `<text x="${(padL + plotW / 2).toFixed(1)}" y="${(padT + plotH / 2).toFixed(1)}" text-anchor="middle" fill="#5d7094" font-family="var(--font-mono)" font-size="12">No measurements recorded yet for this chart</text>`;
      svg += "</svg>";
      wrap.innerHTML = svg;
      return;
    }

    // Crosshair group (updated on hover)
    svg += `<line class="crosshair" x1="0" y1="${padT}" x2="0" y2="${padT + plotH}" stroke="#ff3b30" stroke-opacity="0" stroke-dasharray="3,3" stroke-width="1"/>`;

    // Plot each series: frontier step-line + raw connectors + promoted/rejected/baseline markers + key labels
    const usedLabelBoxes = [];
    function canPlaceLabel(lx, ly) {
      for (const b of usedLabelBoxes) {
        if (Math.abs(b.x - lx) < 34 && Math.abs(b.y - ly) < 13) return false;
      }
      usedLabelBoxes.push({ x: lx, y: ly });
      return true;
    }

    for (const s of activeSeries) {
      const ySc = (def.dualAxis && s.axis === "right") ? yScaleR : yScaleL;
      const lower = s.lower !== undefined ? s.lower : !!def.lower;
      const pts = [];
      for (const r of ROUNDS) {
        const v = getMetricVal(r, s);
        if (isNum(v)) pts.push({ r: r, val: v, x: xScale(r.round), y: ySc(v) });
      }
      if (!pts.length) continue;

      // Subtle dashed actual trend line across all measured rounds
      if (pts.length > 1) {
        const rawD = pts.map((p, idx) => (idx === 0 ? "M " : "L ") + p.x.toFixed(1) + " " + p.y.toFixed(1)).join(" ");
        svg += `<path d="${rawD}" fill="none" stroke="${s.color}" stroke-opacity="0.24" stroke-width="1.2" stroke-dasharray="2,3"/>`;
      }

      // Cumulative best frontier step-line
      const fr = buildFrontierStepPath(pts, xScale, ySc, lower);
      if (fr.d) {
        svg += `<path d="${fr.d}" fill="none" stroke="${s.color}" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/>`;
      }

      // Markers
      for (const p of pts) {
        const st = p.r.status;
        if (st === "baseline") {
          svg += `<rect x="${(p.x - 4.5).toFixed(1)}" y="${(p.y - 4.5).toFixed(1)}" width="9" height="9" transform="rotate(45 ${p.x.toFixed(1)} ${p.y.toFixed(1)})" fill="${s.color}" stroke="#090a0d" stroke-width="1.6"/>`;
        } else if (st === "promoted") {
          svg += `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4.6" fill="#00e599" stroke="${s.color}" stroke-width="2"/>`;
        } else {
          // Rejected marker: amber/crimson downward triangle
          const x = p.x, y = p.y;
          svg += `<polygon points="${x.toFixed(1)},${(y + 5.2).toFixed(1)} ${(x - 4.8).toFixed(1)},${(y - 3.8).toFixed(1)} ${(x + 4.8).toFixed(1)},${(y - 3.8).toFixed(1)}" fill="#ffaa00" stroke="#ff3b30" stroke-width="1.2"/>`;
        }
      }

      // Label frontier improvements (start + latest best)
      const labelTargets = fr.bestPoints.length <= 4
        ? fr.bestPoints
        : [fr.bestPoints[0], fr.bestPoints[fr.bestPoints.length - 1]];
      for (const p of labelTargets) {
        const ly = p.y - 9;
        if (canPlaceLabel(p.x, ly)) {
          svg += `<text x="${p.x.toFixed(1)}" y="${Math.max(padT + 9, ly).toFixed(1)}" text-anchor="middle" fill="${s.color}" font-family="var(--font-mono)" font-size="9.5" font-weight="600">R${String(p.r.round).padStart(2, "0")}</text>`;
        }
      }
    }

    // Invisible interactive overlay for smooth hover snapping
    svg += `<rect class="hover-pad" x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="transparent" style="cursor:crosshair"/>`;
    svg += "</svg>";
    wrap.innerHTML = svg;

    const svgEl = wrap.querySelector("svg");
    const hoverPad = svgEl.querySelector(".hover-pad");
    const cross = svgEl.querySelector(".crosshair");

    hoverPad.addEventListener("mousemove", (evt) => {
      const rect = svgEl.getBoundingClientRect();
      const mx = ((evt.clientX - rect.left) / rect.width) * W;
      let nearest = null;
      let bestDist = Infinity;
      for (const r of ROUNDS) {
        const d = Math.abs(xScale(r.round) - mx);
        if (d < bestDist) {
          bestDist = d;
          nearest = r;
        }
      }
      if (nearest && bestDist < 60) {
        const cx = xScale(nearest.round);
        cross.setAttribute("x1", cx.toFixed(1));
        cross.setAttribute("x2", cx.toFixed(1));
        cross.setAttribute("stroke-opacity", "0.65");
        showTooltip(evt, def, nearest);
      } else {
        cross.setAttribute("stroke-opacity", "0");
        hideTooltip();
      }
    });
    hoverPad.addEventListener("mouseleave", () => {
      cross.setAttribute("stroke-opacity", "0");
      hideTooltip();
    });
  }

  function renderAllCharts() {
    for (const def of CHART_DEFS) renderChart(def);
  }

  // ---------------- FILTERABLE 40-ROUND LEDGER TABLE ----------------
  const FILTERS = [
    { id: "all", label: "All" },
    { id: "promoted", label: "Promoted" },
    { id: "rejected", label: "Rejected" },
    { id: "Decode", label: "Decode" },
    { id: "Prefill", label: "Prefill" },
    { id: "TTFT", label: "TTFT" }
  ];

  function matchesFilter(r, f) {
    if (f === "all") return true;
    if (f === "promoted" || f === "rejected") return r.status === f;
    return r.target === f || r.target === "All";
  }

  function renderFilterPills() {
    const el = document.getElementById("filterPills");
    el.innerHTML = FILTERS.map((f) => {
      const cnt = ROUNDS.filter((r) => matchesFilter(r, f.id)).length;
      return `<button type="button" class="fbtn ${STATE.filter === f.id ? "active" : ""}" data-f="${f.id}">${esc(f.label)}<span class="cnt">(${cnt})</span></button>`;
    }).join("");
    el.querySelectorAll(".fbtn").forEach((b) => {
      b.addEventListener("click", () => {
        STATE.filter = b.dataset.f;
        renderFilterPills();
        renderTable();
      });
    });
  }

  function metricCell(val, baseVal, dec, unit, lower, extraSub) {
    if (!isNum(val)) return `<span class="m-sub dim">--</span>`;
    const d = (R00 && isNum(baseVal)) ? fmtDelta(pctGain(val, baseVal, lower)) : null;
    const sub = extraSub
      ? `<span class="m-sub dim">${esc(extraSub)}</span>`
      : "";
    const dHtml = d && d.text !== "--"
      ? `<span class="m-sub ${d.cls}">${esc(d.text)}</span>`
      : "";
    return `<span class="m-main">${fmt(val, dec)}${unit ? " " + unit : ""}</span>${dHtml}${sub}`;
  }

  function renderTable() {
    const body = document.getElementById("ledgerBody");
    const q = STATE.search.trim().toLowerCase();
    let list = ROUNDS.filter((r) => {
      if (!matchesFilter(r, STATE.filter)) return false;
      if (!q) return true;
      const hay = [
        "r" + String(r.round).padStart(2, "0"),
        "w" + (r.wave || ""),
        r.name, r.target, r.status, r.hypothesis, r.notes,
        (r.source || {}).title, (r.source || {}).url
      ].join(" ").toLowerCase();
      return hay.includes(q);
    });
    list.sort((a, b) => STATE.sortAsc ? (a.round - b.round) : (b.round - a.round));

    if (!list.length) {
      body.innerHTML = `<tr><td colspan="13" class="empty-state">${
        ROUNDS.length === 0
          ? "No rounds recorded in results/inkling2/40round-ledger.json yet. Run Round 0 baseline to begin!"
          : "No rounds match the active filter / search query."
      }</td></tr>`;
      return;
    }

    const html = [];
    for (const r of list) {
      const m = r.metrics || {};
      const p = r.parity || {};
      const src = r.source || {};
      const stCls = r.status === "promoted" ? "b-promoted" : (r.status === "rejected" ? "b-rejected" : "b-baseline");
      const tgtCls = "b-target-" + String(r.target || "all").toLowerCase();
      const pSt = p.status || "PENDING";
      const pCls = (pSt === "EXACT" || pSt === "BASELINE") ? "b-exact" : (pSt === "KLD-OK" ? "b-kld" : "b-mismatch");
      const shas = p.shas || {};
      const shortSha = (shas.ds5_4k_sha || shas.t1_sha || m.ds5_4k_sha || m.t1_sha || "").slice(0, 10);
      const isR0 = r.round === 0;
      const baseRef = isR0 ? {} : BASE_M;

      let srcHtml = "";
      if (src.url) {
        srcHtml = `<a class="src-link" href="${esc(src.url)}" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation()">&#8599; ${esc(src.title || src.url)}</a>`;
      } else if (src.title) {
        srcHtml = `<span class="m-sub dim">${esc(src.title)}</span>`;
      }

      const ttftMain = isNum(m.ttft_4k_ms) ? fmt(m.ttft_4k_ms, 1) + " ms" : "--";
      const ttftSubParts = [];
      if (isNum(m.ttft_16k_ms)) ttftSubParts.push("16K:" + fmt(m.ttft_16k_ms, 0));
      if (isNum(m.warm_ttft_4k_ms)) ttftSubParts.push("W:" + fmt(m.warm_ttft_4k_ms, 0));
      const ttftDelta = (!isR0 && isNum(m.ttft_4k_ms) && isNum(baseRef.ttft_4k_ms)) ? fmtDelta(pctGain(m.ttft_4k_ms, baseRef.ttft_4k_ms, true)) : null;

      const sysParts = [];
      if (isNum(m.serve_sampled_tok_s)) sysParts.push("chat " + fmt(m.serve_sampled_tok_s, 1));
      if (isNum(m.launches_per_step)) sysParts.push(fmt(m.launches_per_step, 0) + " lnch");
      if (isNum(m.vram_free_mb)) sysParts.push(fmt(m.vram_free_mb, 0) + " MB");

      const openCls = STATE.expanded[r.round] ? "open" : "";

      html.push(`
        <tr class="main-row status-${esc(r.status)}" data-round="${r.round}">
          <td style="white-space:nowrap;font-family:var(--font-mono)">
            <strong>R${String(r.round).padStart(2, "0")}</strong>
            ${r.wave ? `<span class="m-sub dim">W${esc(r.wave)}</span>` : ""}
          </td>
          <td><span class="badge ${tgtCls}">${esc(r.target || "All")}</span></td>
          <td class="name-cell"><span class="rname">${esc(r.name)}</span>${srcHtml}</td>
          <td class="hyp-cell">${esc(r.hypothesis || r.notes || "--")}</td>
          <td><span class="badge ${stCls}">${esc(r.status)}</span></td>
          <td>
            <span class="badge ${pCls}">${esc(pSt)}</span>
            ${shortSha ? `<span class="sha-mini">${esc(shortSha)}</span>` : ""}
            ${isNum(p.kld_mean) ? `<span class="sha-mini">KLD ${p.kld_mean.toFixed(5)}</span>` : ""}
          </td>
          <td class="num">${metricCell(m.t1_tok_s, baseRef.t1_tok_s, 2, "", false, isNum(m.t1_ms_tok) ? fmt(m.t1_ms_tok, 2) + " ms" : "")}</td>
          <td class="num">${metricCell(m.ds5_233_tok_s, baseRef.ds5_233_tok_s, 1, "", false, isNum(m.ds5_233_tps_step) ? fmt(m.ds5_233_tps_step, 2) + " t/s" : "")}</td>
          <td class="num">${metricCell(m.ds5_4k_tok_s, baseRef.ds5_4k_tok_s, 2, "", false, isNum(m.ds5_4k_tps_step) ? fmt(m.ds5_4k_tps_step, 2) + " t/s" : "")}</td>
          <td class="num">${metricCell(m.pf4k_tok_s, baseRef.pf4k_tok_s, 0, "", false, "")}</td>
          <td class="num">${metricCell(m.pf16k_tok_s, baseRef.pf16k_tok_s, 0, "", false, isNum(m.ds5_16k_tok_s) ? "dec " + fmt(m.ds5_16k_tok_s, 1) : "")}</td>
          <td class="num">
            <span class="m-main">${ttftMain}</span>
            ${ttftDelta ? `<span class="m-sub ${ttftDelta.cls}">${esc(ttftDelta.text)}</span>` : ""}
            ${ttftSubParts.length ? `<span class="m-sub dim">${esc(ttftSubParts.join(" · "))}</span>` : ""}
          </td>
          <td class="num">
            <span class="m-main" style="font-weight:500">${esc(sysParts[0] || "--")}</span>
            ${sysParts.length > 1 ? `<span class="m-sub dim">${esc(sysParts.slice(1).join(" · "))}</span>` : ""}
          </td>
        </tr>
        <tr class="detail-row ${openCls}" data-detail="${r.round}">
          <td colspan="13">
            <div class="detail-grid">
              <div class="detail-panel">
                <h4>Notes &amp; Config</h4>
                <div class="detail-kv"><span>Mask</span><span>${esc(r.ink_mask || "--")}</span></div>
                <div class="detail-kv"><span>Env</span><span>${esc(JSON.stringify(r.env || {}))}</span></div>
                <div class="detail-kv"><span>Bench Wall</span><span>${isNum(m.total_bench_wall_s) ? fmt(m.total_bench_wall_s, 1) + " s" : "--"}</span></div>
                <div style="margin-top:6px;color:var(--muted);font-family:var(--font-sans)">${esc(r.notes || "No extra notes.")}</div>
              </div>
              <div class="detail-panel">
                <h4>Parity SHAs</h4>
                <div class="detail-kv"><span>t1_233</span><span>${esc((shas.t1_sha || m.t1_sha || "--").slice(0, 16))} (${esc((p.checks || {}).t1_sha || "--")})</span></div>
                <div class="detail-kv"><span>ds5_233</span><span>${esc((shas.ds5_233_sha || m.ds5_233_sha || "--").slice(0, 16))} (${esc((p.checks || {}).ds5_233_sha || "--")})</span></div>
                <div class="detail-kv"><span>ds5_4k</span><span>${esc((shas.ds5_4k_sha || m.ds5_4k_sha || "--").slice(0, 16))} (${esc((p.checks || {}).ds5_4k_sha || "--")})</span></div>
                <div class="detail-kv"><span>long_16k</span><span>${esc((shas.ds5_16k_sha || m.ds5_16k_sha || "--").slice(0, 16))} (${esc((p.checks || {}).ds5_16k_sha || "--")})</span></div>
              </div>
              <div class="detail-panel">
                <h4>Decode &amp; Verify Breakdown</h4>
                <div class="detail-kv"><span>T=1 GPU / GB/s</span><span>${isNum(m.t1_gpu_ms_tok) ? fmt(m.t1_gpu_ms_tok, 2) + " ms / " + fmt(m.t1_gbs, 0) + " GB/s" : "--"}</span></div>
                <div class="detail-kv"><span>DS233 draft/ver</span><span>${isNum(m.ds5_233_draft_ms) ? fmt(m.ds5_233_draft_ms, 2) + " / " + fmt(m.ds5_233_verify_ms, 2) + " ms" : "--"}</span></div>
                <div class="detail-kv"><span>DS4K draft/ver</span><span>${isNum(m.ds5_4k_draft_ms) ? fmt(m.ds5_4k_draft_ms, 2) + " / " + fmt(m.ds5_4k_verify_ms, 2) + " ms" : "--"}</span></div>
                <div class="detail-kv"><span>VBench T=6 / T=8</span><span>${isNum(m.vb_t6_ms) ? fmt(m.vb_t6_ms, 2) + " / " + fmt(m.vb_t8_ms, 2) + " ms" : "--"}</span></div>
              </div>
              <div class="detail-panel">
                <h4>Cache, Serve &amp; VRAM</h4>
                <div class="detail-kv"><span>Warm 4K / Ckpt</span><span>${isNum(m.warm_ttft_4k_ms) ? fmt(m.warm_ttft_4k_ms, 0) + " ms / " + fmt(m.ckpt_mb, 1) + " MiB" : "--"}</span></div>
                <div class="detail-kv"><span>Sampling ms/step</span><span>${isNum(m.sampling_ms_step) ? fmt(m.sampling_ms_step, 3) + " ms" : "--"}</span></div>
                <div class="detail-kv"><span>Multi-req hit TTFT</span><span>${isNum(m.multi_req_hit_ttft_ms) ? fmt(m.multi_req_hit_ttft_ms, 0) + " ms (" + fmt(m.multi_req_cached_tokens, 0) + " tok)" : "--"}</span></div>
                <div class="detail-kv"><span>Arena / VRAM / Budget</span><span>${isNum(m.arena_mb) ? fmt(m.arena_mb, 1) + " MiB" : "--"} / ${isNum(m.vram_used_gib) ? fmt(m.vram_used_gib, 2) + " GiB" : "--"} / ${isNum(m.vram_budget_gib) ? fmt(m.vram_budget_gib, 2) + " GiB" : "--"}</span></div>
              </div>
            </div>
          </td>
        </tr>
      `);
    }
    body.innerHTML = html.join("");
    body.querySelectorAll("tr.main-row").forEach((tr) => {
      tr.addEventListener("click", () => {
        const rn = Number(tr.dataset.round);
        STATE.expanded[rn] = !STATE.expanded[rn];
        const det = body.querySelector(`tr.detail-row[data-detail="${rn}"]`);
        if (det) det.classList.toggle("open", !!STATE.expanded[rn]);
      });
    });
  }

  // ---------------- EVENTS & INIT ----------------
  document.getElementById("axisToggle").querySelectorAll(".seg-btn").forEach((b) => {
    b.addEventListener("click", () => {
      STATE.axisMode = b.dataset.axis;
      document.getElementById("axisToggle").querySelectorAll(".seg-btn").forEach((x) => x.classList.toggle("active", x === b));
      renderAllCharts();
    });
  });

  document.getElementById("searchInput").addEventListener("input", (e) => {
    STATE.search = e.target.value || "";
    renderTable();
  });

  document.getElementById("sortBtn").addEventListener("click", (e) => {
    STATE.sortAsc = !STATE.sortAsc;
    e.target.textContent = STATE.sortAsc ? "Order: R00 ↑" : "Order: R40 ↓";
    renderTable();
  });

  document.getElementById("footNote").textContent =
    "Offline self-contained telemetry dashboard generated by bench/render_40r_dashboard.py from results/inkling2/40round-ledger.json · Zero external CDN dependencies.";

  renderHeader();
  renderKPIs();
  renderHistorical();
  renderAllCharts();
  renderFilterPills();
  renderTable();

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(renderAllCharts, 120);
  });
})();
</script>
</body>
</html>
"""


def render_dashboard(in_path=DEFAULT_IN, out_path=DEFAULT_OUT):
    with open(in_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    # Escape closing script tags safely for inline embedding
    raw_json = json.dumps(ledger, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__LEDGER_JSON__", raw_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    os.replace(tmp, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Render Inkling-2 40-round HTML dashboard from ledger JSON")
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    ap.add_argument("--out", dest="out_path", default=DEFAULT_OUT)
    args = ap.parse_args()
    in_p = args.in_path if os.path.isabs(args.in_path) else os.path.join(LAB, args.in_path)
    out_p = args.out_path if os.path.isabs(args.out_path) else os.path.join(LAB, args.out_path)
    render_dashboard(in_p, out_p)
    print("rendered %s -> %s" % (os.path.relpath(in_p, LAB), os.path.relpath(out_p, LAB)))


if __name__ == "__main__":
    main()
