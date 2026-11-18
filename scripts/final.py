#!/usr/bin/env python3
# Final comparison from three sweep files of one session (sweep.py --grid --repeat 3), in repeat order.
# Selection is fixed on repeat 1 and reported on repeats 2-3 (docs/decisions.md, "Selection and repeats"):
#   - cuFFTDx tuned:   fastest EPT x FPB (grid or default) at the throughput batch, accuracy passing;
#   - cuFFTDx default: the library's suggested configuration;
#   - cuFFTDx accurate: lowest rel L2 over the grid (accuracy only; seeds are fixed, so repeats do not change it);
#   - Monarch:         fastest factorization n1 x n2.
# A ratio whose range over repeats 2-3 contains 1.0 is reported as a tie.
#   python3 scripts/final.py results/tesla_t4/<d>_sweep_a.jsonl <..._b> <..._c> [--figures docs/figures/tesla_t4]
import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import plot  # noqa: E402  (style helpers and palette)

BIG = 1 << 14


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def key(r):
    if r["impl"] == "monarch":
        return ("monarch", r["precision"], r["n"], f"{r['n1']}x{r['n2']}")
    return ("cufftdx", r["precision"], r["n"], f"e{r['ept']}f{r['ffts_per_block']}", r.get("variant", "default"))


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
        for role in (["monarch"] if impl == "monarch" else ["tuned"] + (["default"] if k[4] == "default" else [])):
            cur = sel.get((role, prec, n))
            if role == "default" or cur is None or r["gbps_eff"] > rep1[cur]["gbps_eff"]:
                sel[(role, prec, n)] = k
        if impl == "cufftdx":
            cur = sel.get(("accurate", prec, n))
            if cur is None or r["rel_l2_median"] < rep1[cur]["rel_l2_median"]:
                sel[("accurate", prec, n)] = k
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
    table = defaultdict(dict)
    for prec in plot.PRECS:
        for n in sorted({k[2] for k in sel if k[1] == prec}):
            vals = {role: [g(role, prec, n, r) for r in later] for role in ("default", "tuned", "monarch")}
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

    print("\n### Accuracy of the selected configurations (repeat 1; median over 5 seeds)\n")
    print("| prec | N | impl | total rel L2 | compute rel L2 | total / compute |")
    print("|---|---:|---|---:|---:|---:|")
    err = defaultdict(list)
    for prec in plot.PRECS:
        for n in sorted({k[2] for k in sel if k[1] == prec}):
            for role in ("default", "tuned", "accurate", "monarch"):
                k = sel.get((role, prec, n))
                if not k:
                    continue
                r = reps[0][k]
                t, c = r["rel_l2_median"], r.get("rel_l2_compute_median")
                err[(role, prec)].append((n, t, c))
                print(f"| {prec} | {n} | {role} ({cfg_name(k)}) | {t:.2e} | {c:.2e} | {t / c:.2f} |" if c else
                      f"| {prec} | {n} | {role} | {t:.2e} | - | - |")
    for role in ("tuned", "accurate"):
        print(f"\nError ratio cuFFTDx {role} / Monarch:")
        for prec in plot.PRECS:
            mon = {n: (t, c) for n, t, c in err[("monarch", prec)]}
            for n, t, c in err[(role, prec)]:
                if n in mon and c and mon[n][1]:
                    print(f"  {prec} N={n}: total {t / mon[n][0]:.2f}x, compute {c / mon[n][1]:.2f}x")

    if a.figures:
        os.makedirs(a.figures, exist_ok=True)
        figures(sel, reps, later, err, a.figures)


def figures(sel, reps, later, err, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, prec in zip(axes, plot.PRECS):
        for role, color, ls, mk, name in (("default", plot.COLOR["cufftdx"], "--", "s", "cuFFTDx default"),
                                         ("tuned", plot.COLOR["cufftdx"], "-", "o", "cuFFTDx tuned"),
                                         ("monarch", plot.COLOR["monarch"], "-", "o", "Monarch")):
            ns = sorted(n for (r, p, n) in sel if r == role and p == prec)
            ys, lo, hi = [], [], []
            for n in ns:
                v = [rep.get(sel[(role, prec, n)]) for rep in later]
                m, l, h = stats([x["gbps_eff"] for x in v if x]) if all(v) else (float("nan"),) * 3
                ys.append(m), lo.append(m - l), hi.append(h - m)
            ax.errorbar(ns, ys, yerr=[lo, hi], color=color, linestyle=ls, marker=mk, markersize=5, linewidth=2,
                        capsize=3, label=name)
        plot.style(ax, "effective GB/s, batch 2^14" if prec == "fp16" else "")
        plot.xticks(ax, sorted({n for (_, p, n) in sel if p == prec}))
        ax.set_title(prec, color=plot.INK, fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Throughput, configuration fixed on repeat 1, mean and range over repeats 2-3",
                 color=plot.INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out, "throughput_tuned.png"), dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, prec in zip(axes, plot.PRECS):
        for role, color, mk, name, both in (("tuned", plot.COLOR["cufftdx"], "s", "cuFFTDx fastest", False),
                                            ("accurate", plot.COLOR["cufftdx"], "o", "cuFFTDx most accurate", True),
                                            ("monarch", plot.COLOR["monarch"], "o", "Monarch", True)):
            pts = err[(role, prec)]
            if not pts:
                continue
            ns = [p[0] for p in pts]
            ax.plot(ns, [p[1] for p in pts], color=color, marker=mk, linestyle="--" if not both else "-",
                    linewidth=2, label=f"{name}, total")
            if both:
                ax.plot(ns, [p[2] for p in pts], color=color, marker="^", linestyle=":", linewidth=2,
                        label=f"{name}, compute only")
        plot.style(ax, "relative L2 error vs FP64" if prec == "fp16" else "", logy=True)
        plot.xticks(ax, sorted({p[0] for k in err if k[1] == prec for p in err[k]}))
        ax.set_title(prec, color=plot.INK, fontsize=11)
    # legend outside the axes: inside, it covered the fp32 curves
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Error vs exact input (total) and vs input rounded to the I/O type (compute only)",
                 color=plot.INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(out, "error_decomposition.png"), dpi=150)
    print(f"\nfigures in {out}")


if __name__ == "__main__":
    main()
