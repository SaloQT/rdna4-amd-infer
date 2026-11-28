#!/usr/bin/env python3
"""Render docs/progress-chart.svg (the README frontier chart) from docs/40round-ledger.json.

Every series is a running best over promoted rounds (the baseline counts), drawn
on a numeric R00..R<last> axis, so a new ledger round moves the chart without
hand-editing the SVG.

    python scripts/render_progress_chart.py [--ledger PATH] [--out PATH]
"""
import argparse
import json
import math
import os
from html import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ledger values that were measured on a different workload than the series they
# sit in, so they must not set the frontier: (metric, round) -> reason.
NOT_COMPARABLE = {
    ("ds5_16k_tok_s", 67): "R67 measured real 16K code (176 -> 208 tok/s), not the standard 16K bench",
    ("serve_sampled_tok_s", 87): "R87 live-serve sampled 236.7 tok/s exceeds its own greedy 209.3 tok/s",
}

# Per-round highlight tiles shown under the charts: (strip title, accent, [(label, text, colour)]).
EM, CY, AM = "#00e599", "#00d4ff", "#ffaa00"
STRIPS = [
    (None, None, [
        ("PREFIX-HIT TTFT", "2,113 -> 226 ms (9.35x)", EM),
        ("CPU SAMPLING + DRY", "0.399 -> 0.005 ms (80x)", CY),
        ("16K T=8 VERIFY", "23.4 -> 17.5 ms (-25.0%)", AM),
    ]),
    ("R88 Promoted Highlights Strip", EM, [
        ("R88 WMMA VISION (1280px)", "83.8 -> 75.6 ms (-9.8%)", EM),
        ("R88 DSPARK DRAFT (233 / 4K)", "1.32 -> 1.05 ms (-20.5%)", CY),
        ("R88 NT-LOAD + HOST -O3", "+5.4% T=1 · +6.5% DS233", AM),
    ]),
    ("R89 Promoted Highlights Strip (against R88)", CY, [
        ("R89 WMMA VISION (1280px, vs R88)", "76.3 -> 73.4 ms (-3.9%)", EM),
        ("R89 DSPARK DRAFT (233, vs R88)", "1.09 -> 1.01 ms (-7.3%)", CY),
        ("R89 16K VERIFY T=8 (vs R88)", "17.83 -> 17.54 ms (-1.6%)", AM),
    ]),
]

MONO = "ui-monospace,Consolas,monospace"
SANS = "system-ui,Segoe UI,sans-serif"
PX0, PX1 = 76.0, 798.0  # plot x range
RED, ORANGE = "#ff3b30", "#ff6b35"


def fmt(v, dec=0):
    return f"{v:,.{dec}f}"


def esc(s):
    return escape(s, quote=False)


def accepted(r):
    return r["status"] in ("promoted", "baseline")


def frontier(rounds, key, lower=False):
    """Running best per round (None until the metric first appears)."""
    best, out, setter = None, [], {}
    for r in rounds:
        v = r["metrics"].get(key)
        if accepted(r) and isinstance(v, (int, float)) and (key, r["round"]) not in NOT_COMPARABLE:
            if best is None or (v < best if lower else v > best):
                best = v
                setter = r["round"]
        out.append((r["round"], best, setter if best is not None else None))
    return out


def nice_ticks(lo, hi, want=5):
    span = hi - lo
    raw = span / want
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.ceil(lo / step) * step
    ticks = []
    t = first
    while t <= hi + 1e-9:
        ticks.append(t)
        t += step
    return ticks


class Panel:
    def __init__(self, top, height, plot_top, plot_bot, last_round):
        self.top, self.height = top, height
        self.pt, self.pb = plot_top, plot_bot
        self.last = last_round
        self.lo = self.hi = None

    def x(self, rnd):
        return PX0 + (PX1 - PX0) * rnd / self.last

    def y(self, v):
        return self.pb - (v - self.lo) / (self.hi - self.lo) * (self.pb - self.pt - 10) - 4

    def set_range(self, series):
        vals = [v for s in series for _, v, _ in s if v is not None]
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.08 or 1
        self.lo, self.hi = lo - pad, hi + pad


def x_ticks(last):
    ticks = list(range(0, last, 10))
    if last - ticks[-1] < 5:
        ticks.pop()
    return ticks + [last]


def panel_frame(p, accent, title, legend_rows, out):
    out.append(f'  <rect x="18" y="{p.top}" width="804" height="{p.height}" rx="4" fill="#111318" stroke="#232732"/>')
    out.append(f'  <rect x="18" y="{p.top + 10}" width="3" height="18" rx="1" fill="{accent}"/>')
    out.append(f'  <text x="34" y="{p.top + 24}" fill="#f5f7fa" font-size="13" font-weight="700" font-family="{MONO}">{esc(title)}</text>')
    out.append(f'  <line x1="34" y1="{p.top + 34}" x2="806" y2="{p.top + 34}" stroke="#232732" stroke-width="1"/>')
    out.append("")
    for (x_sw, x_txt, x_val, row, colour, label, value) in legend_rows:
        y = p.top + 49 + row * 20
        out.append(f'  <rect x="{x_sw}" y="{y}" width="14" height="3" fill="{colour}"/>')
        out.append(f'  <text x="{x_txt}" y="{y + 6}" fill="#cbd2df" font-size="12" font-family="{MONO}">{esc(label)}</text>')
        out.append(f'  <text x="{x_val}" y="{y + 6}" fill="{colour}" font-size="12" font-weight="700" text-anchor="end" font-family="{MONO}">{esc(value)}</text>')
        out.append("")


def panel_axes(p, ticks, out, tick_fmt):
    out.append(f'  <line x1="34" y1="{p.top + 87}" x2="806" y2="{p.top + 87}" stroke="#1d212b" stroke-width="1"/>')
    for t in ticks:
        y = p.y(t)
        out.append(f'  <line x1="76" y1="{y:.1f}" x2="798" y2="{y:.1f}" stroke="#1d212b" stroke-width="1" stroke-dasharray="3,4"/>')
        out.append(f'  <text x="66" y="{y + 4.5:.1f}" fill="#7d8799" font-size="11.5" text-anchor="end" font-family="{MONO}">{tick_fmt(t)}</text>')
    for r in x_ticks(p.last):
        x = p.x(r)
        out.append(f'  <line x1="{x:.1f}" y1="{p.pt}" x2="{x:.1f}" y2="{p.pb}" stroke="#171a22" stroke-width="1"/>')
        out.append(f'  <text x="{x:.1f}" y="{p.pb + 19}" fill="#7d8799" font-size="11.5" text-anchor="middle" font-family="{MONO}">R{r:02d}</text>')


def step_path(p, series):
    pts = [(p.x(r), p.y(v)) for r, v, _ in series if v is not None]
    d, prev_y = [], None
    for i, (x, y) in enumerate(pts):
        if i == 0:
            d.append(f"M {x:.1f} {y:.1f}")
        elif y != prev_y:
            d.append(f"L {x:.1f} {prev_y:.1f} L {x:.1f} {y:.1f}")
        prev_y = y
    d.append(f"L {PX1:.1f} {prev_y:.1f}")
    return " ".join(d), prev_y


def draw_series(p, series, colour, width, fill, out):
    d, end_y = step_path(p, series)
    if fill:
        out.append(f'  <path d="{d} L {PX1:.1f} {p.pb:.1f} L {PX0:.1f} {p.pb:.1f} Z" fill="url(#{fill})"/>')
    out.append(f'  <path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"/>')
    return end_y


def end_dots(p, ends, out):
    for colour, y in ends:
        out.append(f'  <circle cx="{PX1:.1f}" cy="{y:.1f}" r="3.8" fill="{colour}" stroke="#111318" stroke-width="1.6"/>')


def pct(new, old, lower=False):
    return (old - new) / old * 100 if lower else (new - old) / old * 100


def render(ledger):
    rounds = ledger["rounds"]
    last = rounds[-1]["round"]
    by = {r["round"]: r for r in rounds}
    base = by[0]["metrics"]
    v0 = next(m for m in ledger["historical_milestones"] if m["id"] == "stock_r0")

    def f(key, lower=False):
        return frontier(rounds, key, lower)

    ds233, ds16k, sampled, t1 = f("ds5_233_tok_s"), f("ds5_16k_tok_s"), f("serve_sampled_tok_s"), f("t1_tok_s")
    pf4k, pf16k = f("pf4k_tok_s"), f("pf16k_tok_s")
    launches = f("launches_per_step", lower=True)

    def now(series):
        return series[-1][1], series[-1][2]

    out = []
    w, h = 840, 1198
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="RX 9070 XT RDNA4 Inkling-2-Mini R00-R{last} LLM Optimization Progression">')
    out.append("""  <defs>
    <linearGradient id="radeonBar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#ed1c24"/>
      <stop offset="55%" stop-color="#ff5733"/>
      <stop offset="100%" stop-color="#ffaa00"/>
    </linearGradient>""")
    for gid, col, op in (("fillEm", EM, 0.14), ("fillCy", CY, 0.14), ("fillRd", RED, 0.16)):
        out.append(f"""    <linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{col}" stop-opacity="{op}"/>
      <stop offset="100%" stop-color="{col}" stop-opacity="0.0"/>
    </linearGradient>""")
    out.append("  </defs>\n")
    out.append(f'  <rect width="{w}" height="{h}" rx="6" fill="#090a0d" stroke="#232732" stroke-width="1"/>')
    out.append('  <rect x="1" y="1" width="838" height="3" rx="1" fill="url(#radeonBar)"/>')
    out.append(f'  <g transform="translate(24.0,21.0) scale(0.7917)" fill="none" stroke="{EM}" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 6-7"/><path d="M16 7h4v4"/></g>')
    out.append(f'  <text x="50" y="37" fill="#f5f7fa" font-size="18" font-weight="700" font-family="{SANS}">Inkling-2-Mini: R00-R{last} Frontier Progression (RX 9070 XT)</text>')
    out.append(f'  <text x="24" y="58" fill="#8c95a6" font-size="12" font-family="{MONO}">Running best over promoted rounds · R00 -&gt; R{last} · 218 zero-spill kernels</text>')
    out.append("")

    # ---- Panel 1: decode ----
    p1 = Panel(76, 318, 176, 362, last)
    p1.set_range([ds233, ds16k, sampled, t1])
    d233, d16 = now(ds233), now(ds16k)
    dsm, t1n = now(sampled), now(t1)
    t1_round = by[t1n[1]]["metrics"]
    gpu = 1000 / t1_round["t1_gpu_ms_tok"] if t1_round.get("t1_gpu_ms_tok") else None
    t1_txt = f"{base['t1_tok_s']:.1f} -> {t1n[0]:.1f} wall" + (f" / {gpu:.1f} GPU" if gpu else "") + f" @R{t1n[1]}"
    out.append("  <!-- PANEL 1: DECODE -->")
    panel_frame(p1, EM, "1. DECODE FRONTIER (tok/s · higher is better)", [
        (34, 56, 404, 0, EM, "DS5-233 Code", f"{base['ds5_233_tok_s']:.1f} -> {d233[0]:.1f} (+{pct(d233[0], base['ds5_233_tok_s']):.1f}%) @R{d233[1]}"),
        (34, 56, 404, 1, CY, "DS5-16K Ctx", f"{base['ds5_16k_tok_s']:.1f} -> {d16[0]:.1f} (+{pct(d16[0], base['ds5_16k_tok_s']):.1f}%) @R{d16[1]}"),
        (436, 458, 806, 0, ORANGE, "Live Sampled", f"{base['serve_sampled_tok_s']:.1f} -> {dsm[0]:.1f} (+{pct(dsm[0], base['serve_sampled_tok_s']):.1f}%) @R{dsm[1]}"),
        (436, 458, 806, 1, AM, "Plain T=1", t1_txt),
    ], out)
    out.append(f'  <line x1="420" y1="118" x2="420" y2="158" stroke="#1d212b" stroke-width="1"/>')
    panel_axes(p1, nice_ticks(p1.lo, p1.hi), out, lambda t: fmt(t))
    ends = [
        (EM, draw_series(p1, ds233, EM, 2.6, "fillEm", out)),
        (ORANGE, draw_series(p1, sampled, ORANGE, 2.4, None, out)),
        (CY, draw_series(p1, ds16k, CY, 2.4, None, out)),
        (AM, draw_series(p1, t1, AM, 2.4, None, out)),
    ]
    end_dots(p1, ends, out)
    out.append("")

    # ---- Panel 2: prefill ----
    p2 = Panel(408, 288, 508, 664, last)
    p2.set_range([pf4k, pf16k])
    a, b = now(pf4k), now(pf16k)
    m4, m16 = by[a[1]]["metrics"], by[b[1]]["metrics"]
    x_v0 = a[0] / v0["pf4k_tok_s"]
    t4 = f" · {m4['ttft_4k_ms']:.0f} ms" if m4.get("ttft_4k_ms") else ""
    t16 = f" · {m16['ttft_16k_ms'] / 1000:.2f} s" if m16.get("ttft_16k_ms") else ""
    out.append("  <!-- PANEL 2: PREFILL -->")
    panel_frame(p2, CY, f"2. COLD PREFILL FRONTIER (tok/s · {x_v0:.2f}x vs our v0 impl {fmt(v0['pf4k_tok_s'])})", [
        (34, 56, 806, 0, CY, "4K Cold Prefill", f"{fmt(base['pf4k_tok_s'])} -> {fmt(a[0])} tok/s{t4} (+{pct(a[0], base['pf4k_tok_s']):.1f}%) @R{a[1]}"),
        (34, 56, 806, 1, AM, "16K Cold Prefill", f"{fmt(base['pf16k_tok_s'])} -> {fmt(b[0])} tok/s{t16} (+{pct(b[0], base['pf16k_tok_s']):.1f}%) @R{b[1]}"),
    ], out)
    panel_axes(p2, nice_ticks(p2.lo, p2.hi), out, lambda t: fmt(t))
    ends = [(CY, draw_series(p2, pf4k, CY, 2.4, "fillCy", out)), (AM, draw_series(p2, pf16k, AM, 2.4, None, out))]
    end_dots(p2, ends, out)
    out.append("")

    # ---- Panel 3: launches ----
    p3 = Panel(710, 394, 788, 936, last)
    p3.set_range([launches])
    ln, lr = now(launches)
    drop = pct(ln, v0["launches_per_step"], lower=True)
    legend = (f"{fmt(v0['launches_per_step'])} (v0) -> {fmt(base['launches_per_step'])} (R00) "
              f"-> {fmt(ln)} (R{lr}, -{drop:.1f}%)")
    l256 = by[last]["metrics"].get("launches_per_step_256k")
    if l256:
        legend += f" | 256K: {fmt(l256)} (R{last})"
    out.append("  <!-- PANEL 3: LAUNCHES -->")
    panel_frame(p3, RED, "3. WDDM KERNEL LAUNCHES / STEP (lower is better)", [
        (34, 56, 806, 0, RED, "Plain T=1 Launches / Step", legend),
    ], out)
    panel_axes(p3, nice_ticks(p3.lo, p3.hi), out, lambda t: fmt(t))
    ends = [(RED, draw_series(p3, launches, RED, 2.6, "fillRd", out))]
    end_dots(p3, ends, out)
    out.append("")

    # ---- Highlight strips ----
    out.append("  <!-- Key Latency Wins Strip (strictly BELOW X-axis) -->")
    y = 970
    for i, (title, accent, tiles) in enumerate(STRIPS):
        hgt = 52 if i == 0 else 54
        if title:
            out.append(f"  <!-- {title} -->")
        out.append(f'  <rect x="34" y="{y}" width="772" height="{hgt}" rx="2" fill="#090a0d" stroke="#232732"/>')
        if accent:
            out.append(f'  <rect x="34" y="{y}" width="3" height="{hgt}" rx="1" fill="{accent}"/>')
        for x in (291, 548):
            out.append(f'  <line x1="{x}" y1="{y}" x2="{x}" y2="{y + hgt}" stroke="#232732"/>')
        for x, (label, text, colour) in zip((48, 305, 562), tiles):
            out.append(f'  <text x="{x}" y="{y + 19}" fill="#7d8799" font-size="10.5" font-family="{MONO}">{esc(label)}</text>')
            out.append(f'  <text x="{x}" y="{y + hgt - 10 + (0 if i == 0 else 1)}" fill="{colour}" font-size="13" font-weight="700" font-family="{MONO}">{esc(text)}</text>')
        out.append("")
        y += hgt + 12
    out.append(f'  <text x="24" y="{h - 11}" fill="#7d8799" font-size="11.5" font-family="{MONO}">RX 9070 XT (16 GB, WDDM) · 218 zero-spill Inkling-2 kernels · R00-R{last} · Early Tok-1 TTFT</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=os.path.join(ROOT, "docs", "40round-ledger.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "progress-chart.svg"))
    args = ap.parse_args()
    with open(args.ledger, encoding="utf-8") as fh:
        ledger = json.load(fh)
    svg = render(ledger)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)
    print(f"wrote {args.out} (R00-R{ledger['rounds'][-1]['round']})")


if __name__ == "__main__":
    main()
