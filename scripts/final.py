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


# Layout after the performance charts in the cuFFTDx docs (examples page, 26.10): throughput relative to a
# baseline normalized to 1.0 and drawn in a light color, the ratio printed on each bar, legend and setup note
# below. Neutral palette and default font: these are not NVIDIA results.
BASE, DEFAULT, ACCURATE = "#bdbdbd", "#7f7f7f", "#3f3f3f"
ROLE_COLOR = {"tuned": BASE, "default": DEFAULT, "accurate": ACCURATE, "c32_tuned": C32, "c32_accurate": C32,
              "monarch": plot.COLOR["monarch"]}


def axis_style(ax):
    ax.grid(True, axis="y", color=plot.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(plot.INK)
    ax.tick_params(colors=plot.INK, labelsize=9)
    ax.tick_params(axis="y", length=0)


def legend_below(fig, axes, extra=()):
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h), labels.append(l)
    for h, l in extra:
        handles.append(h), labels.append(l)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=4, frameon=False, fontsize=9)


def figures(sel, reps, later, err, out):
    r0 = next(iter(reps[0].values()))
    setup = (f"{r0['gpu']}, MathDx {r0['mathdx']}, CUDA {r0['cuda_runtime'] // 1000}.{r0['cuda_runtime'] % 1000 // 10}, "
             f"batch 2^14 FFTs per launch.")
    have_c32 = any(k[0].startswith("c32") for k in sel)

    def ratios(role, prec, n):
        if (role, prec, n) not in sel or ("tuned", prec, n) not in sel:
            return None
        v = [rep.get(sel[(role, prec, n)]) for rep in later]
        t = [rep.get(sel[("tuned", prec, n)]) for rep in later]
        if not (all(v) and all(t)):
            return None
        return stats([a["gbps_eff"] / b["gbps_eff"] for a, b in zip(v, t)])

    roles = [("tuned", "cuFFTDx tuned (= 1.0)"), ("default", "cuFFTDx default")]
    if have_c32:
        roles.append(("c32_tuned", "cuFFTDx fp32 compute, fp16 I/O, tuned"))
    roles.append(("monarch", "Monarch (cuBLASDx)"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), sharey=True)
    for ax, prec in zip(axes, plot.PRECS):
        ns = sorted({n for (_, p, n) in sel if p == prec})
        present = [(r, name) for r, name in roles if any(ratios(r, prec, n) for n in ns)]
        w = 0.8 / len(present)
        for i, (role, name) in enumerate(present):
            for j, n in enumerate(ns):
                rs = ratios(role, prec, n)
                if not rs:
                    continue
                x = j - 0.4 + w * (i + 0.5)
                m, lo, hi = rs
                ax.bar(x, m, w * 0.88, color=ROLE_COLOR[role], zorder=2,
                       label=name if j == min(k for k, nn in enumerate(ns) if ratios(role, prec, nn)) else None)
                if role == "tuned":
                    continue
                if hi > lo:
                    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=plot.INK, linewidth=0.7, capsize=1.5, zorder=3)
                ax.annotate(f"{m:.2f}", (x, hi), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                            fontsize=7, color=plot.INK, rotation=90 if len(present) > 3 else 0)
        ax.axhline(1.0, color=plot.INK, linestyle="--", linewidth=0.9, zorder=1)
        ax.set_xticks(range(len(ns)), [str(n) for n in ns])
        ax.set_xlabel("FFT size", color=plot.INK)
        ax.set_title(f"{prec} I/O", fontsize=10.5, color=plot.INK)
        axis_style(ax)
    axes[0].set_ylabel("throughput relative to tuned cuFFTDx (higher is better)", color=plot.INK)
    axes[0].set_ylim(0, max(1.15, axes[0].get_ylim()[1] * 1.06))
    legend_below(fig, axes)
    fig.text(0.5, 0.02, setup + " Configurations chosen on repeat 1; bars are the mean ratio over repeats 2-3, whiskers "
             "their range. Tuned can fall below default when the repeat-1 winner did not keep its lead.",
             ha="center", va="bottom", fontsize=7.5, color=plot.MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(out, "throughput_tuned.png"), dpi=150)

    eroles = [("tuned", "cuFFTDx fastest", "s"), ("accurate", "cuFFTDx most accurate", "o")]
    if have_c32:
        eroles.append(("c32_accurate", "cuFFTDx fp32 compute, fp16 I/O, most accurate", "D"))
    eroles.append(("monarch", "Monarch (cuBLASDx)", "o"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    for ax, prec in zip(axes, plot.PRECS):
        for role, name, mk in eroles:
            pts = err[(role, prec)]
            if not pts:
                continue
            c = ROLE_COLOR[role]
            ns = [p[0] for p in pts]
            ax.plot(ns, [p[1] for p in pts], color=c, marker=mk, markersize=6, linewidth=1.8,
                    linestyle="--" if role == "tuned" else "-", label=name)
            if role != "tuned":
                ax.plot(ns, [p[2] for p in pts], color=c, marker=mk, markersize=8, markerfacecolor="white",
                        markeredgewidth=1.4, linestyle=":", linewidth=1.2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        allns = sorted({p[0] for k in err if k[1] == prec for p in err[k]})
        ax.set_xticks(allns, [str(n) for n in allns])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ys = [v for k in err if k[1] == prec for p in err[k] for v in p[1:] if v]
        lo, hi = math.floor(math.log10(min(ys))), math.floor(math.log10(max(ys)))
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(
            [m * 10 ** e for e in range(lo, hi + 1) for m in (1, 1.5, 2, 3, 5, 7) if min(ys) / 1.1 <= m * 10 ** e <= max(ys) * 1.1]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda y, _: f"{y / 10 ** math.floor(math.log10(y)):g}e{math.floor(math.log10(y))}"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("FFT size", color=plot.INK)
        ax.set_title(f"{prec} I/O", fontsize=10.5, color=plot.INK)
        axis_style(ax)
    axes[0].set_ylabel("relative L2 error vs FP64 (lower is better)", color=plot.INK)
    hollow = matplotlib.lines.Line2D([], [], color=plot.MUTED, marker="o", markerfacecolor="white", linestyle=":")
    filled = matplotlib.lines.Line2D([], [], color=plot.MUTED, marker="o", linestyle="-")
    legend_below(fig, axes, [(filled, "total: exact input"), (hollow, "compute only: input already rounded")])
    fig.text(0.5, 0.02, setup + " Median over 5 seeds. Most accurate = lowest error over the EPT x FPB grid; "
             "fastest = the tuned configuration of the throughput figure.",
             ha="center", va="bottom", fontsize=7.5, color=plot.MUTED)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(out, "error_decomposition.png"), dpi=150)
    print(f"\nfigures in {out}")


if __name__ == "__main__":
    main()
