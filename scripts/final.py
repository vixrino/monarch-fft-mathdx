#!/usr/bin/env python3
# Final comparison from three sweep files of one session (sweep.py --grid --repeat 3), in repeat order.
# Selection is fixed on repeat 1 and reported on repeats 2-3 (docs/decisions.md, "Selection and repeats"):
#   - cuFFTDx tuned:   fastest EPT x FPB (grid or default) at the throughput batch, accuracy passing;
#   - cuFFTDx default: the library's suggested configuration;
#   - cuFFTDx accurate: lowest rel L2 over the grid (accuracy only; seeds are fixed, so repeats do not change it);
#   - Monarch:         fastest factorization n1 x n2;
#   - c32_*:           the same three roles for cuFFTDx computing in fp32 on fp16 storage (precision "fp16",
#                      compute "fp32"); only in sweeps from the commit that added it.
# A ratio whose range over repeats 2-3 contains 1.0 is reported as a tie.
#   python3 scripts/final.py results/tesla_t4/<d>_sweep_a.jsonl <..._b> <..._c> [--figures docs/figures/tesla_t4]
import argparse
import json
import math
import os
from collections import defaultdict

import matplotlib
import matplotlib.lines  # noqa: F401
import matplotlib.ticker  # noqa: F401

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import plot  # noqa: E402  (style helpers and palette)

BIG = 1 << 14


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


C32 = "#7b4fc9"  # third series: cuFFTDx with fp32 compute on fp16 storage


def key(r):
    if r["impl"] == "monarch":
        return ("monarch", r["precision"], r["n"], f"{r['n1']}x{r['n2']}")
    impl = "c32" if r.get("compute", r["precision"]) != r["precision"] else "cufftdx"
    return (impl, r["precision"], r["n"], f"e{r['ept']}f{r['ffts_per_block']}", r.get("variant", "default"))


def throughput(rows):
    # one record per configuration at the throughput batch (the default config also appears in the grid:
    # both are kept, under different keys)
    return {key(r): r for r in rows if r.get("status") == "ok" and r.get("experiment") != "sass" and r["batch"] >= BIG}


def select(rep1):
    sel = {}
    for k, r in rep1.items():
        if not r["acc_pass"]:
            continue
        impl, prec, n = k[:3]
        p = "c32_" if impl == "c32" else ""
        roles = ["monarch"] if impl == "monarch" else [p + "tuned"] + ([p + "default"] if k[4] == "default" else [])
        for role in roles:
            cur = sel.get((role, prec, n))
            if role.endswith("default") or cur is None or r["gbps_eff"] > rep1[cur]["gbps_eff"]:
                sel[(role, prec, n)] = k
        if impl != "monarch":
            cur = sel.get((p + "accurate", prec, n))
            if cur is None or r["rel_l2_median"] < rep1[cur]["rel_l2_median"]:
                sel[(p + "accurate", prec, n)] = k
    return sel


def stats(xs):
    xs = sorted(xs)
    return sum(xs) / len(xs), xs[0], xs[-1]


def fmt_range(m, lo, hi, p=2):
    return f"{m:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def cfg_name(k):
    return k[3] if k[0] == "monarch" else f"EPT {k[3].split('f')[0][1:]}, FPB {k[3].split('f')[1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", nargs=3)
    ap.add_argument("--figures")
    a = ap.parse_args()
    reps = [throughput(load(p)) for p in a.sweeps]
    sel = select(reps[0])
    later = reps[1:]

    def g(role, prec, n, rep):
        k = sel.get((role, prec, n))
        return rep.get(k) if k else None

    print(f"Selection on {os.path.basename(a.sweeps[0])}; ratios from {', '.join(os.path.basename(p) for p in a.sweeps[1:])}.\n")
    print("| prec | N | cuFFTDx default | cuFFTDx tuned (config) | Monarch (n1xn2) | Monarch / tuned [range] | "
          "tuned / default [range] | verdict |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    c32_rows = []
    table = defaultdict(dict)
    for prec in plot.PRECS:
        for n in sorted({k[2] for k in sel if k[1] == prec}):
            vals = {role: [g(role, prec, n, r) for r in later] for role in ("default", "tuned", "monarch", "c32_tuned")}
            gb = {role: stats([r["gbps_eff"] for r in v]) if all(v) else None for role, v in vals.items()}
            row = [prec, str(n)]
            row.append(f"{gb['default'][0]:.1f}" if gb["default"] else "-")
            row.append(f"{gb['tuned'][0]:.1f} ({cfg_name(sel[('tuned', prec, n)])})" if gb["tuned"] else "-")
            row.append(f"{gb['monarch'][0]:.1f} ({sel[('monarch', prec, n)][3]})" if gb["monarch"] else "-")
            ratio = verdict = "-"
            if gb["monarch"] and gb["tuned"]:
                rs = stats([m["gbps_eff"] / t["gbps_eff"] for m, t in zip(vals["monarch"], vals["tuned"])])
                table[prec][n] = rs
                ratio = fmt_range(*rs)
                verdict = "tie" if rs[1] <= 1 <= rs[2] else ("Monarch faster" if rs[1] > 1 else "cuFFTDx faster")
            td = "-"
            if gb["tuned"] and gb["default"]:
                td = fmt_range(*stats([t["gbps_eff"] / d["gbps_eff"] for t, d in zip(vals["tuned"], vals["default"])]))
            print("| " + " | ".join(row + [ratio, td, verdict]) + " |")
            if all(vals["c32_tuned"]):
                c = stats([r["gbps_eff"] for r in vals["c32_tuned"]])
                line = [prec, str(n), f"{c[0]:.1f} ({cfg_name(sel[('c32_tuned', prec, n)])})"]
                for other in ("tuned", "monarch"):
                    if all(vals[other]):
                        rs = stats([o["gbps_eff"] / t["gbps_eff"] for o, t in zip(vals[other], vals["c32_tuned"])])
                        v = "tie" if rs[1] <= 1 <= rs[2] else ("faster" if rs[1] > 1 else "slower")
                        line.append(f"{fmt_range(*rs)} ({v})")
                    else:
                        line.append("-")
                c32_rows.append(line)
    if c32_rows:
        print("\n### cuFFTDx with fp32 compute on fp16 storage (tuned on repeat 1, repeats 2-3)\n")
        print("| prec | N | fp32-compute GB/s (config) | cuFFTDx fp16 tuned / it [range] | Monarch / it [range] |")
        print("|---|---:|---:|---:|---:|")
        for line in c32_rows:
            print("| " + " | ".join(line) + " |")

    print("\n### Accuracy of the selected configurations (repeat 1; median over 5 seeds)\n")
    print("| prec | N | impl | total rel L2 | compute rel L2 | total / compute |")
    print("|---|---:|---|---:|---:|---:|")
    err = defaultdict(list)
    for prec in plot.PRECS:
        for n in sorted({k[2] for k in sel if k[1] == prec}):
            for role in ("default", "tuned", "accurate", "c32_default", "c32_tuned", "c32_accurate", "monarch"):
                k = sel.get((role, prec, n))
                if not k:
                    continue
                r = reps[0][k]
                t, c = r["rel_l2_median"], r.get("rel_l2_compute_median")
                err[(role, prec)].append((n, t, c))
                print(f"| {prec} | {n} | {role} ({cfg_name(k)}) | {t:.2e} | {c:.2e} | {t / c:.2f} |" if c else
                      f"| {prec} | {n} | {role} | {t:.2e} | - | - |")
    for role in ("tuned", "accurate", "c32_tuned", "c32_accurate"):
        if not any(err[(role, p)] for p in plot.PRECS):
            continue
        print(f"\nError ratio cuFFTDx {role} / Monarch:")
        for prec in plot.PRECS:
            mon = {n: (t, c) for n, t, c in err[("monarch", prec)]}
            for n, t, c in err[(role, prec)]:
                if n in mon and c and mon[n][1]:
                    print(f"  {prec} N={n}: total {t / mon[n][0]:.2f}x, compute {c / mon[n][1]:.2f}x")

    if a.figures:
        os.makedirs(a.figures, exist_ok=True)
        figures(sel, reps, later, err, a.figures)


# Bar charts in the layout of vendor benchmark slides: grouped bars per N, baselines in grays, the subject
# (Monarch) in the accent color, the key ratio printed above its bar, the setup in a footnote. Neutral palette:
# no vendor branding, since this is not an NVIDIA result.
GRAY_LIGHT, GRAY_DARK = "#c4c4c4", "#6b6b6b"


def bars(ax, ns, series, logy=False):
    # series: [(label, color, {n: (value, lo, hi, compute_or_None)})]; returns x position of each (series, n)
    w = 0.8 / len(series)
    pos = {}
    # log axis: every bar starts at the same decade, below the smallest value with margin, so no bar reads as zero
    floor = 10 ** math.floor(math.log10(min(min(v[0], v[3] or v[0]) for _, _, vals in series for v in vals.values()) / 1.5)) if logy else 0
    for i, (label, color, vals) in enumerate(series):
        for j, n in enumerate(ns):
            if n not in vals:
                continue
            v, lo, hi, c = vals[n]
            x = j - 0.4 + w * (i + 0.5)
            pos[(label, n)] = (x, v, hi)
            ax.bar(x, v - floor, w * 0.9, bottom=floor, color=color, label=label if j == min(k for k, m in enumerate(ns) if m in vals) else None,
                   zorder=2)
            if hi > lo:
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=plot.INK, linewidth=0.8, capsize=1.5, zorder=3)
            if c is not None:
                ax.hlines(c, x - w * 0.45, x + w * 0.45, color=plot.INK, linewidth=1.6, zorder=3)
    ax.set_xticks(range(len(ns)), [str(n) for n in ns])
    ax.set_xlabel("FFT size N", color=plot.MUTED)
    if logy:
        ax.set_yscale("log")
        top = max(v[0] for _, _, vals in series for v in vals.values())
        ticks = [m * 10 ** e for e in range(int(math.log10(floor)), int(math.log10(top)) + 2) for m in (1, 2, 3, 5)
                 if floor <= m * 10 ** e <= top * 1.3]
        ax.set_ylim(floor, top * 1.35)
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda y, _: f"{y / 10 ** math.floor(math.log10(y)):.0f}e{math.floor(math.log10(y))}"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(True, axis="y", color=plot.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(plot.MUTED)
    ax.tick_params(colors=plot.MUTED, labelsize=9, length=0)
    return pos


def annotate(ax, pos, label, ns, ratio, logy=False):
    # one label per group, above its tallest bar and centered on the group: above the Monarch bar alone it ran
    # into the taller neighbouring bars
    tops = {}
    for (_, n), (_, _, hi) in pos.items():
        tops[n] = max(tops.get(n, 0), hi)
    for j, n in enumerate(ns):
        if (label, n) in pos and n in ratio:
            ax.annotate(f"{ratio[n]:.2f}x", (j, tops[n]), xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=8.5, fontweight="bold", color=plot.INK)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * (1.6 if logy else 1.1))


def header(fig, title, subtitle, foot):
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=13, fontweight="bold", color=plot.INK)
    fig.text(0.012, 0.915, subtitle, ha="left", va="top", fontsize=9.5, color=plot.MUTED)
    fig.text(0.012, 0.012, foot, ha="left", va="bottom", fontsize=7.5, color=plot.MUTED)


def figures(sel, reps, later, err, out):
    r0 = next(r for r in reps[0].values())
    foot = (f"{r0['gpu']}, MathDx {r0['mathdx']}, CUDA {r0['cuda_runtime'] // 1000}.{r0['cuda_runtime'] % 1000 // 10}, "
            f"batch 2^14 FFTs. Configurations chosen on repeat 1; bars are the mean of repeats 2-3, whiskers their range.")
    have_c32 = any(k[0].startswith("c32") for k in sel)
    ref, ref_name = ("c32_tuned", "cuFFTDx fp32-compute tuned") if have_c32 else ("tuned", "cuFFTDx tuned")

    def gb(role, prec, n):
        v = [rep.get(sel[(role, prec, n)]) for rep in later] if (role, prec, n) in sel else []
        return stats([x["gbps_eff"] for x in v]) if v and all(v) else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, prec in zip(axes, plot.PRECS):
        ns = sorted({n for (_, p, n) in sel if p == prec})
        series = []
        for role, color, name in (("default", GRAY_LIGHT, "cuFFTDx default"), ("tuned", GRAY_DARK, "cuFFTDx tuned"),
                                  ("c32_tuned", C32, "cuFFTDx fp32 compute, fp16 I/O, tuned"),
                                  ("monarch", plot.COLOR["monarch"], "Monarch (cuBLASDx)")):
            vals = {n: (*gb(role, prec, n), None) for n in ns if gb(role, prec, n)}
            if vals:
                series.append((name, color, vals))
        pos = bars(ax, ns, series)
        r = ref if any(gb(ref, prec, n) for n in ns) else "tuned"
        ratio = {n: gb("monarch", prec, n)[0] / gb(r, prec, n)[0] for n in ns if gb("monarch", prec, n) and gb(r, prec, n)}
        annotate(ax, pos, "Monarch (cuBLASDx)", ns, ratio)
        ax.set_title(f"{prec} I/O", loc="left", fontsize=10.5, color=plot.INK, fontweight="bold")
        if prec == "fp16":
            ax.set_ylabel("effective GB/s (higher is better)", color=plot.MUTED)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h), labels.append(l)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=4, frameon=False, fontsize=9)
    header(fig, "Throughput", f"Label above each group: Monarch / {ref_name}"
           + (" (fp16 I/O), / cuFFTDx tuned (fp32 I/O)." if have_c32 else "."), foot)
    fig.tight_layout(rect=(0, 0.13, 1, 0.88))
    fig.savefig(os.path.join(out, "throughput_tuned.png"), dpi=150)

    acc_ref = "c32_accurate" if have_c32 else "accurate"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, prec in zip(axes, plot.PRECS):
        ns = sorted({p[0] for k in err if k[1] == prec for p in err[k]})
        series = []
        for role, color, name in (("tuned", GRAY_LIGHT, "cuFFTDx fastest"), ("accurate", GRAY_DARK, "cuFFTDx most accurate"),
                                  ("c32_accurate", C32, "cuFFTDx fp32 compute, fp16 I/O, most accurate"),
                                  ("monarch", plot.COLOR["monarch"], "Monarch (cuBLASDx)")):
            vals = {n: (t, t, t, c) for n, t, c in err[(role, prec)]}
            if vals:
                series.append((name, color, vals))
        pos = bars(ax, ns, series, logy=True)
        ref_err = {n: t for n, t, _ in (err[(acc_ref, prec)] or err[("accurate", prec)])}
        ratio = {n: ref_err[n] / t for n, t, _ in err[("monarch", prec)] if n in ref_err}
        annotate(ax, pos, "Monarch (cuBLASDx)", ns, ratio, logy=True)
        ax.set_title(f"{prec} I/O", loc="left", fontsize=10.5, color=plot.INK, fontweight="bold")
        if prec == "fp16":
            ax.set_ylabel("relative L2 error vs FP64 (lower is better)", color=plot.MUTED)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h), labels.append(l)
    tick = matplotlib.lines.Line2D([], [], color=plot.INK, linewidth=1.6)
    handles.append(tick), labels.append("error of the transform alone (input already rounded)")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=3, frameon=False, fontsize=9)
    acc_name = ("cuFFTDx fp32-compute most accurate / Monarch (fp16 I/O), cuFFTDx most accurate / Monarch (fp32 I/O)"
                if have_c32 else "cuFFTDx most accurate / Monarch")
    header(fig, "Accuracy", f"Bars: total error vs exact input. Label above each group: {acc_name}.",
           foot.replace("bars are the mean of repeats 2-3, whiskers their range", "median over 5 seeds"))
    fig.tight_layout(rect=(0, 0.15, 1, 0.88))
    fig.savefig(os.path.join(out, "error_decomposition.png"), dpi=150)
    print(f"\nfigures in {out}")


if __name__ == "__main__":
    main()
