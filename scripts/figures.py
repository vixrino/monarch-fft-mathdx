# README figures, laid out like the cuFFTDx docs charts; neutral palette, since these are not NVIDIA results.
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

from tables import PRECS, stats  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MONARCH = "#eb6834"
C32 = "#7b4fc9"  # third series: cuFFTDx with fp32 compute on fp16 storage
BASE, DEFAULT, ACCURATE = "#a8a8a8", "#5f5f5f", "#2f6f8f"
ROLE_COLOR = {"tuned": BASE, "default": DEFAULT, "accurate": ACCURATE, "c32_tuned": C32, "c32_accurate": C32,
              "monarch": MONARCH}


def axis_style(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=9)
    ax.tick_params(axis="y", length=0)


def legend_below(fig, axes, extra=()):
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h), labels.append(lab)
    for h, lab in extra:
        handles.append(h), labels.append(lab)
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
        roles.append(("c32_tuned", "cuFFTDx mixed (fp16 I/O, fp32 compute), tuned"))
    roles.append(("monarch", "Monarch (cuBLASDx)"))
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.9), sharey=True)
    for ax, prec in zip(axes, PRECS):
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
                    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=INK, linewidth=0.7, capsize=1.5, zorder=3)
                ax.annotate(f"{m:.2f}", (x, hi), xytext=(0, 2.5), textcoords="offset points", ha="center", va="bottom",
                            fontsize=6.5 if len(present) > 3 else 7, color=INK, zorder=4,
                            bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))
        ax.axhline(1.0, color=INK, linestyle="--", linewidth=0.9, zorder=1)
        ax.set_xticks(range(len(ns)), [str(n) for n in ns])
        ax.set_xlabel("FFT size", color=INK)
        ax.set_title(f"{prec} I/O", fontsize=10.5, color=INK)
        axis_style(ax)
    axes[0].set_ylabel("throughput relative to tuned cuFFTDx (higher is better)", color=INK)
    axes[0].set_ylim(0, max(1.15, axes[0].get_ylim()[1] * 1.06))
    legend_below(fig, axes)
    fig.text(0.5, 0.02, setup + " Configurations chosen on repeat 1; bars are the mean ratio over repeats 2-3, whiskers "
             "their range. Tuned can fall below default when the repeat-1 winner did not keep its lead.",
             ha="center", va="bottom", fontsize=7.5, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(out, "throughput_tuned.png"), dpi=150)

    eroles = [("tuned", "cuFFTDx fastest", "s"), ("accurate", "cuFFTDx most accurate", "o")]
    if have_c32:
        eroles.append(("c32_accurate", "cuFFTDx mixed (fp16 I/O, fp32 compute), most accurate", "D"))
    eroles.append(("monarch", "Monarch (cuBLASDx)", "o"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    for ax, prec in zip(axes, PRECS):
        for role, name, mk in eroles:
            pts = err[(role, prec)]
            if not pts:
                continue
            c = ROLE_COLOR[role]
            # NaN where a size was not run, so the line breaks there
            allns = sorted({p[0] for k in err if k[1] == prec for p in err[k]})
            got = {p[0]: p for p in pts}
            ns = allns
            tot = [got[n][1] if n in got else float("nan") for n in ns]
            cmp_ = [got[n][2] if n in got and got[n][2] else float("nan") for n in ns]
            ax.plot(ns, tot, color=c, marker=mk, markersize=6, linewidth=1.8,
                    linestyle="--" if role == "tuned" else "-", label=name)
            if role != "tuned":
                ax.plot(ns, cmp_, color=c, marker=mk, markersize=8, markerfacecolor="white",
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
        ax.set_xlabel("FFT size", color=INK)
        ax.set_title(f"{prec} I/O", fontsize=10.5, color=INK)
        axis_style(ax)
    axes[0].set_ylabel("relative L2 error vs FP64 (lower is better)", color=INK)
    hollow = matplotlib.lines.Line2D([], [], color=MUTED, marker="o", markerfacecolor="white", linestyle=":")
    filled = matplotlib.lines.Line2D([], [], color=MUTED, marker="o", linestyle="-")
    legend_below(fig, axes, [(filled, "total: exact input"), (hollow, "compute only: input already rounded")])
    fig.text(0.5, 0.02, setup + " Median over 5 seeds. Most accurate = lowest error over the EPT x FPB grid; "
             "fastest = the tuned configuration of the throughput figure.\nLines break where a size was not run: Monarch fp32 "
             "N = 2048 needs more than 64 KB of shared memory, so its N = 4096 point stands alone.",
             ha="center", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(out, "error_decomposition.png"), dpi=150)
    print(f"\nfigures in {out}")
