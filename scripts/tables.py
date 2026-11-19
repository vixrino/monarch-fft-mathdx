#!/usr/bin/env python3
# Markdown tables of the default configurations (Monarch factorizations, latency, SASS); newest sweep by default.
#   python3 scripts/tables.py [results/<gpu>/<date>_sweep.jsonl ...]
import argparse
import glob
import json
import os
import re

PRECS = ["fp16", "fp32"]


def stats(xs):
    xs = sorted(xs)
    return sum(xs) / len(xs), xs[0], xs[-1]


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
            rows += [dict(json.loads(line), source=p) for line in f if line.strip()]
    return rows


def gbps_iqr(r):
    # time p25 -> fastest, so it bounds GB/s from above
    bytes_ = r["gbps_eff"] * r["time_ms_median"]
    return bytes_ / r["time_ms_p75"], bytes_ / r["time_ms_p25"]


def split(rows):
    # default configurations only; the grid and the mixed variant are final.py's (mixed SASS rows kept)
    rows = [r for r in rows if r.get("variant") != "grid"]
    mixed = [dict(r, precision=r["precision"] + " (fp32 compute)") for r in rows
             if r.get("experiment") == "sass" and r.get("compute", r["precision"]) != r["precision"]]
    rows = [r for r in rows if r.get("compute", r.get("precision")) == r.get("precision")]
    timed = [r for r in rows if r.get("status") == "ok" and r["experiment"] != "sass"]
    big = max(r["batch"] for r in timed)
    thr = [r for r in timed if r["batch"] == big]
    lat = [r for r in timed if r["batch"] == r["ffts_per_block"]]
    return thr, lat, [r for r in rows if r.get("experiment") == "sass"] + mixed, [r for r in rows if r.get("status") == "skipped"]


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
    a = ap.parse_args()
    paths = a.files or newest_sweeps(a.results)
    if not paths:
        raise SystemExit(f"no *_sweep*.jsonl under {a.results}/<gpu>/; run scripts/sweep.py on the GPU first")
    tables(*split(load(paths)))


if __name__ == "__main__":
    main()
