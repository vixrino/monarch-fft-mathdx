#!/usr/bin/env python3
# Selection on repeat 1, ratios over repeats 2-3 (docs/decisions.md, "Selection and repeats").
#   python3 scripts/final.py results/tesla_t4/<d>_sweep_a.jsonl <..._b> <..._c> [--figures docs/figures/tesla_t4]
import argparse
import json
import os
from collections import defaultdict

from tables import PRECS, stats

BIG = 1 << 14


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def key(r):
    if r["impl"] == "monarch":
        return ("monarch", r["precision"], r["n"], f"{r['n1']}x{r['n2']}")
    impl = "c32" if r.get("compute", r["precision"]) != r["precision"] else "cufftdx"
    return (impl, r["precision"], r["n"], f"e{r['ept']}f{r['ffts_per_block']}", r.get("variant", "default"))


def throughput(rows):
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
    for prec in PRECS:
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
    for prec in PRECS:
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
        if not any(err[(role, p)] for p in PRECS):
            continue
        print(f"\nError ratio cuFFTDx {role} / Monarch:")
        for prec in PRECS:
            mon = {n: (t, c) for n, t, c in err[("monarch", prec)]}
            for n, t, c in err[(role, prec)]:
                if n in mon and c and mon[n][1]:
                    print(f"  {prec} N={n}: total {t / mon[n][0]:.2f}x, compute {c / mon[n][1]:.2f}x")

    if a.figures:
        import figures  # matplotlib only when drawing

        os.makedirs(a.figures, exist_ok=True)
        figures.figures(sel, reps, later, err, a.figures)


if __name__ == "__main__":
    main()
