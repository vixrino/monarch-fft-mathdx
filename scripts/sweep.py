#!/usr/bin/env python3
# Both binaries and their SASS counts in one GPU session, into a new results/<gpu>/<date>_sweep[_k].jsonl.
# --grid adds the cuFFTDx tuning grid, --repeat R writes R files in a row.
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

BINARIES = ["cufftdx_block_fft", "monarch_fft"]
GRID = [f"cufftdx_grid_{n}" for n in (256, 512, 1024, 2048, 4096, 8192, 16384)]


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def new_path(root):
    gpu = sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").splitlines()[0].lower().replace(" ", "_")
    d = os.path.join(root, gpu)
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"{datetime.date.today().isoformat()}_sweep")
    path, k = f"{stem}.jsonl", 2
    while os.path.exists(path):
        path, k = f"{stem}_{k}.jsonl", k + 1
    return path


# sass_check.sh shortens names; these recover the configuration from what it prints
MONARCH = re.compile(r"monarch<(__half|float), \(unsigned int\)(\d+), \(unsigned int\)(\d+), \(unsigned int\)(\d+)>")
CUFFTDX = re.compile(r"^block_fft N=(\d+) (__half|float)( io=__half2)?$")
PREC = {"__half": "fp16", "float": "fp32"}
# git_dirty ignores outputs the runner writes
SRC_ONLY = (":(exclude)results", ":(exclude)docs/figures")


def sass_records(binary, prov):
    out = sh("bash", "scripts/sass_check.sh", binary)
    for line in out.splitlines()[1:]:
        total, mma, ffma, hfma2, fn = line.split(None, 4)
        r = {"experiment": "sass", "binary": os.path.basename(binary), "function": fn, "sass_total": int(total),
             "mma": int(mma), "ffma": int(ffma), "hfma2": int(hfma2), "counts": "static"}
        if m := MONARCH.search(fn):
            r.update(impl="monarch", precision=PREC[m[1]], n1=int(m[2]), n2=int(m[3]), n=int(m[2]) * int(m[3]),
                     ffts_per_block=int(m[4]))
        elif m := CUFFTDX.match(fn):
            r.update(impl="cufftdx", precision="fp16" if m[3] else PREC[m[2]], compute=PREC[m[2]], n=int(m[1]))
        else:
            continue  # table/fill/widen helper kernels
        yield {**r, **prov}


def one_sweep(a, rep):
    path = new_path(a.results)
    print("writing", path, flush=True)
    ok = True
    bins = BINARIES + [g for g in GRID if os.path.exists(os.path.join(a.build, g))] * a.grid
    for b in bins:
        r = subprocess.run([os.path.join(a.build, b), "--out", path])
        ok &= r.returncode == 0
        print(f"{b}: exit {r.returncode}", flush=True)

    prov = {"git_commit": sh("git", "rev-parse", "--short=7", "HEAD"),
            "git_dirty": bool(sh("git", "status", "--porcelain", "--untracked-files=no", "--", ".", *SRC_ONLY)),
            "cuobjdump": sh("cuobjdump", "--version").splitlines()[-1],
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cmd": " ".join(sys.argv), "repeat": rep, "repeats": a.repeat}
    with open(path, "a") as f:
        for b in BINARIES:
            for r in sass_records(os.path.join(a.build, b), prov):
                f.write(json.dumps(r) + "\n")
                prec = r["precision"] + ("c32" if r.get("compute", r["precision"]) != r["precision"] else "")
                print(f"  sass {r['impl']:8s} {prec:7s} N={r['n']:5d} "
                      f"{str(r.get('n1', '')) + 'x' + str(r.get('n2', '')) if 'n1' in r else '':7s} "
                      f"mma={r['mma']:4d} ffma={r['ffma']:5d} hfma2={r['hfma2']:5d} total={r['sass_total']:6d}")
    print(path)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="build")
    ap.add_argument("--results", default="results")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    ok = True
    for rep in range(a.repeat):
        ok &= one_sweep(a, rep)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
