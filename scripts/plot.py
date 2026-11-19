#!/usr/bin/env python3
# Figures and markdown tables from results/*.jsonl only. By default it uses the newest sweep file of
# each GPU, so cuFFTDx and Monarch numbers always come from the same session.
#   python3 scripts/plot.py                      figures into docs/figures/<gpu>/
#   python3 scripts/plot.py --tables             also print the tables used in docs/results.md
#   python3 scripts/plot.py results/x.jsonl ...  explicit files instead
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, NullLocator  # noqa: E402

# categorical slots 1 and 2 of the dataviz reference palette, validated for CVD on a light surface
COLOR = {"cufftdx": "#2a78d6", "monarch": "#eb6834"}
NAME = {"cufftdx": "cuFFTDx", "monarch": "Monarch (cuBLASDx)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PRECS = ["fp16", "fp32"]


def newest_sweeps(root):
    def key(p):
        m = re.search(r"(\d{4}-\d{2}-\d{2})_sweep(?:_(\d+))?\.jsonl$", p)
        return (m[1], int(m[2] or 1))
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*/"))):
        files = [p for p in glob.glob(os.path.join(d, "*_sweep*.jsonl")) if re.search(r"_sweep(_\d+)?\.jsonl$", p)]
        if files:
            out.append(max(files, key=key))
    return out


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [dict(json.loads(l), source=p) for l in f if l.strip()]
    return rows


def gbps_iqr(r):
    # time p25 -> fastest, so it bounds GB/s from above
    bytes_ = r["gbps_eff"] * r["time_ms_median"]
    return bytes_ / r["time_ms_p75"], bytes_ / r["time_ms_p25"]


def split(rows):
    # the cuFFTDx tuning grid and the fp32-compute / fp16-storage variant are analysed by final.py; these
    # tables keep the library's default configuration
    rows = [r for r in rows if r.get("variant") != "grid" and r.get("compute", r.get("precision")) == r.get("precision")]
    timed = [r for r in rows if r.get("status") == "ok" and r["experiment"] != "sass"]
    big = max(r["batch"] for r in timed)
    thr = [r for r in timed if r["batch"] == big]
    lat = [r for r in timed if r["batch"] == r["ffts_per_block"]]
    return thr, lat, [r for r in rows if r.get("experiment") == "sass"], [r for r in rows if r.get("status") == "skipped"]


def gbps_cell(r):
    if not r:
        return "-"
    lo, hi = gbps_iqr(r)
    return f"{r['gbps_eff']:.1f} [{lo:.1f}, {hi:.1f}]"


def label(r):
    return f"{r['n1']}x{r['n2']}" if r["impl"] == "monarch" else ""


def best_by_n(rows, impl, prec):
    best = {}
    for r in rows:
        if r["impl"] == impl and r["precision"] == prec and (r["n"] not in best or r["gbps_eff"] > best[r["n"]]["gbps_eff"]):
            best[r["n"]] = r
    return [best[n] for n in sorted(best)]


def gaps(ns, ys):
    # a skipped N (e.g. fp32 Monarch 2048, out of smem) breaks the line instead of being interpolated
    xo, yo = [], []
    for i, (n, y) in enumerate(zip(ns, ys)):
        if i and n > 2 * ns[i - 1]:
            xo.append(2 * ns[i - 1])
            yo.append(float("nan"))
        xo.append(n)
        yo.append(y)
    return xo, yo


def style(ax, ylabel, logy=False):
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("N", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def xticks(ax, ns):
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], rotation=0)
    ax.minorticks_off()


def end_label(ax, rs, key, impl):
    r = rs[-1]
    ax.annotate(NAME[impl].split(" ")[0], (r["n"], r[key]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)


def fig_throughput(thr, gpu, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, prec in zip(axes, PRECS):
        ns = sorted({r["n"] for r in thr if r["precision"] == prec})
        for impl, marker in (("cufftdx", "o"), ("monarch", "s")):
            rs = best_by_n(thr, impl, prec)
            if not rs:
                continue
            lo, hi = zip(*(gbps_iqr(r) for r in rs))
            y = [r["gbps_eff"] for r in rs]
            ax.errorbar(*gaps([r["n"] for r in rs], y), yerr=[gaps([r["n"] for r in rs], v)[1] for v in
                        ([a - b for a, b in zip(y, lo)], [b - a for a, b in zip(y, hi)])],
                        color=COLOR[impl], marker=marker, markersize=6, linewidth=2, capsize=0, label=NAME[impl] +
                        (", best n1xn2 per N" if impl == "monarch" else ""))
            end_label(ax, rs, "gbps_eff", impl)
            if impl == "monarch":
                best = {id(r) for r in rs}
                others = [r for r in thr if r["impl"] == impl and r["precision"] == prec and id(r) not in best]
                ax.scatter([r["n"] for r in others], [r["gbps_eff"] for r in others], marker=marker, s=36,
                           facecolors="white", edgecolors=COLOR[impl], linewidths=1.5, zorder=3,
                           label="Monarch, other n1xn2")
                for r, xy, ha in [(r, (0, -13), "center") for r in rs] + [(r, (7, 0), "left") for r in others]:
                    ax.annotate(label(r), (r["n"], r["gbps_eff"]), xytext=xy, textcoords="offset points",
                                ha=ha, va="center", fontsize=7, color=MUTED)
        style(ax, "effective GB/s (read + write)" if prec == "fp16" else "")
        xticks(ax, ns)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{prec}" + (" (fp32 accumulate)" if prec == "fp16" else ""), color=INK, fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"Throughput, batch {thr[0]['batch']} FFTs, {gpu}; bars = IQR over 50 launches", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "throughput_vs_n.png"), dpi=150)
    plt.close(fig)


def fig_error(thr, gpu, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, prec in zip(axes, PRECS):
        ns = sorted({r["n"] for r in thr if r["precision"] == prec})
        for impl, marker in (("cufftdx", "o"), ("monarch", "s")):
            rs = sorted((r for r in thr if r["impl"] == impl and r["precision"] == prec), key=lambda r: r["n"])
            if not rs:
                continue
            # every Monarch factorization is shown; the line joins the most accurate one per N
            line = {}
            for r in rs:
                if r["n"] not in line or r["rel_l2_median"] < line[r["n"]]["rel_l2_median"]:
                    line[r["n"]] = r
            lr = [line[n] for n in sorted(line)]
            ax.plot(*gaps([r["n"] for r in lr], [r["rel_l2_median"] for r in lr]), color=COLOR[impl], marker=marker,
                    markersize=6, linewidth=2, label=NAME[impl])
            extra = [r for r in rs if r is not line[r["n"]]]
            ax.scatter([r["n"] for r in extra], [r["rel_l2_median"] for r in extra], marker=marker, s=36,
                       facecolors="white", edgecolors=COLOR[impl], linewidths=1.5, zorder=1)
            end_label(ax, lr, "rel_l2_median", impl)
        style(ax, "relative L2 error vs FP64 (median of 5 seeds)" if prec == "fp16" else "", logy=True)
        # the fp32 panel spans less than one decade, where default log ticks are unlabeled
        lo, hi = ax.get_ylim()
        ax.set_yticks([lo * (hi / lo) ** (i / 4) for i in range(5)])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
        ax.yaxis.set_minor_locator(NullLocator())
        xticks(ax, ns)
        ax.set_title(prec, color=INK, fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Accuracy vs cuFFT Z2Z, {gpu}; hollow = other Monarch n1xn2", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "error_vs_n.png"), dpi=150)
    plt.close(fig)


def fig_scatter(thr, gpu, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for impl, marker in (("cufftdx", "o"), ("monarch", "s")):
        for prec in PRECS:
            rs = best_by_n(thr, impl, prec)
            if not rs:
                continue
            filled = prec == "fp32"
            ax.scatter([r["rel_l2_median"] for r in rs], [r["gbps_eff"] for r in rs], marker=marker, s=40,
                       facecolors=COLOR[impl] if filled else "white", edgecolors=COLOR[impl], linewidths=1.5,
                       label=f"{NAME[impl]} {prec}", zorder=3)
            # alternate sides so neighbouring N in a cluster don't print on top of each other
            for i, r in enumerate(rs):
                ax.annotate(str(r["n"]), (r["rel_l2_median"], r["gbps_eff"]), xytext=(6 if i % 2 else -6, 0),
                            textcoords="offset points", ha="left" if i % 2 else "right", va="center",
                            fontsize=6.5, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("relative L2 error vs FP64 (lower is better)", color=MUTED)
    ax.set_ylabel(f"effective GB/s at batch {thr[0]['batch']}", color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"Speed vs accuracy, {gpu}\nlabel = N; Monarch: best n1xn2 per N; hollow = fp16, filled = fp32",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "speed_vs_accuracy.png"), dpi=150)
    plt.close(fig)


def tables(thr, lat, sass, skipped):
    print("### Throughput and accuracy, best Monarch factorization per N\n")
    print("| prec | N | cuFFTDx GB/s [IQR] | Monarch n1xn2 | Monarch GB/s [IQR] | Monarch / cuFFTDx | cuFFTDx rel L2 | Monarch rel L2 |")
    print("|---|---:|---:|---|---:|---:|---:|---:|")
    for prec in PRECS:
        c = {r["n"]: r for r in best_by_n(thr, "cufftdx", prec)}
        m = {r["n"]: r for r in best_by_n(thr, "monarch", prec)}
        for n in sorted(set(c) | set(m)):
            cr, mr = c.get(n), m.get(n)
            cells = [prec, str(n), gbps_cell(cr), label(mr) if mr else "-", gbps_cell(mr),
                     f"{mr['gbps_eff'] / cr['gbps_eff']:.2f}" if cr and mr else "-",
                     f"{cr['rel_l2_median']:.2e}" if cr else "-", f"{mr['rel_l2_median']:.2e}" if mr else "-"]
            print("| " + " | ".join(cells) + " |")
    print("\n### Every Monarch configuration\n")
    print("| prec | N | n1xn2 | FFTs/block | smem B | blocks/SM | GB/s | rel L2 | max abs |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted((r for r in thr if r["impl"] == "monarch"), key=lambda r: (r["precision"], r["n"], r["n1"])):
        print(f"| {r['precision']} | {r['n']} | {label(r)} | {r['ffts_per_block']} | {r['smem_bytes']} | "
              f"{r['blocks_per_sm']} | {r['gbps_eff']:.1f} | {r['rel_l2_median']:.2e} | {r['max_abs_median']:.2e} |")
    for r in skipped:
        print(f"| {r['precision']} | {r['n']} | {label(r)} | {r['ffts_per_block']} | {r['smem_bytes']} | "
              f"- | skipped: {r['reason']} | | |")
    print("\n### Latency: one block's group of FFTs, grid = 1\n")
    print("| prec | impl | N | n1xn2 | FFTs | median us [IQR] |")
    print("|---|---|---:|---|---:|---:|")
    for r in sorted(lat, key=lambda r: (r["precision"], r["n"], r["impl"], r.get("n1", 0))):
        print(f"| {r['precision']} | {r['impl']} | {r['n']} | {label(r)} | {r['batch']} | "
              f"{1e3 * r['time_ms_median']:.1f} [{1e3 * r['time_ms_p25']:.1f}, {1e3 * r['time_ms_p75']:.1f}] |")
    if sass:
        print("\n### SASS, static instruction counts per kernel\n")
        print("| impl | prec | N | n1xn2 | MMA | FFMA | HFMA2 | total |")
        print("|---|---|---:|---|---:|---:|---:|---:|")
        for r in sorted(sass, key=lambda r: (r["impl"], r["precision"], r["n"], r.get("n1", 0))):
            print(f"| {r['impl']} | {r['precision']} | {r['n']} | {label(r)} | {r['mma']} | {r['ffma']} | "
                  f"{r['hfma2']} | {r['sass_total']} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--tables", action="store_true")
    a = ap.parse_args()

    paths = a.files or newest_sweeps(a.results)
    if not paths:
        raise SystemExit(f"no *_sweep*.jsonl under {a.results}/<gpu>/; run scripts/sweep.py on the GPU first")
    by_gpu = defaultdict(list)
    for r in load(paths):
        by_gpu[os.path.basename(os.path.dirname(r["source"]))].append(r)
    for gpu, rows in by_gpu.items():
        thr, lat, sass, skipped = split(rows)
        out = os.path.join(a.out, gpu)
        os.makedirs(out, exist_ok=True)
        name = rows[0]["gpu"] if "gpu" in rows[0] else gpu
        fig_throughput(thr, name, out)
        fig_error(thr, name, out)
        fig_scatter(thr, name, out)
        print(f"{gpu}: {len(thr)} throughput rows from {sorted({r['source'] for r in rows})} -> {out}")
        if a.tables:
            tables(thr, lat, sass, skipped)


if __name__ == "__main__":
    main()
