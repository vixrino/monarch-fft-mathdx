#!/usr/bin/env python3
# mma_probe in one GPU session: timing of the three cuBLASDx configurations (build/mma_probe --out) and their
# static SASS counts, written to one new results/<gpu>/<date>_mma_probe[_k].jsonl.
#   python3 scripts/probe.py [--build build] [--results results]
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

from sweep import SRC_ONLY, sh

VARIANT = {"0": "plain", "1": "suggested", "2": "forced"}


def new_path(root):
    gpu = sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").splitlines()[0].lower().replace(" ", "_")
    os.makedirs(os.path.join(root, gpu), exist_ok=True)
    stem = os.path.join(root, gpu, f"{datetime.date.today().isoformat()}_mma_probe")
    path, k = f"{stem}.jsonl", 2
    while os.path.exists(path):
        path, k = f"{stem}_{k}.jsonl", k + 1
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="build")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    binary = os.path.join(a.build, "mma_probe")
    path = new_path(a.results)
    rc = subprocess.run([binary, "--out", path]).returncode
    prov = {"git_commit": sh("git", "rev-parse", "--short=7", "HEAD"),
            "git_dirty": bool(sh("git", "status", "--porcelain", "--untracked-files=no", "--", ".", *SRC_ONLY)),
            "cuobjdump": sh("cuobjdump", "--version").splitlines()[-1],
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cmd": " ".join(sys.argv)}
    with open(path, "a") as f:
        lines = [l for l in sh("bash", "scripts/sass_check.sh", binary, "probe").splitlines()[1:] if l.strip()]
        for i, line in enumerate(lines):
            total, mma, ffma, hfma2, fn = line.split(None, 4)
            # template arguments start with the variant index (see mma_probe.cu); sass_check.sh sorts by name,
            # so the position is the fallback
            m = re.search(r"probe<(?:\(int\))?(\d)", fn)
            v = m[1] if m else str(i)
            r = {"experiment": "sass", "binary": "mma_probe", "variant": VARIANT.get(v, v), "function": fn,
                 "sass_total": int(total), "mma": int(mma), "ffma": int(ffma), "hfma2": int(hfma2),
                 "counts": "static", **prov}
            f.write(json.dumps(r) + "\n")
            print(f"  sass {r['variant']:9s} mma={r['mma']:4d} ffma={r['ffma']:5d} total={r['sass_total']:6d}")
    print(path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
