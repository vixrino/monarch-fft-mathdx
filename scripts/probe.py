#!/usr/bin/env python3
# mma_probe timing, error, SASS counts and top opcodes into a new results/<gpu>/<date>_mma_probe[_k].jsonl.
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
        lines = [s for s in sh("bash", "scripts/sass_check.sh", binary, "probe").splitlines()[1:] if s.strip()]
        for i, line in enumerate(lines):
            total, mma, ffma, hfma2, fn = line.split(None, 4)
            m = re.search(r"probe<(?:\(int\))?(\d)", fn)
            v = m[1] if m else str(i)
            r = {"experiment": "sass", "binary": "mma_probe", "variant": VARIANT.get(v, v), "function": fn,
                 "sass_total": int(total), "mma": int(mma), "ffma": int(ffma), "hfma2": int(hfma2),
                 "counts": "static", **prov}
            f.write(json.dumps(r) + "\n")
            print(f"  sass {r['variant']:9s} mma={r['mma']:4d} ffma={r['ffma']:5d} total={r['sass_total']:6d}")
        out = subprocess.run(["bash", "scripts/sass_check.sh", binary, "probe"], capture_output=True, text=True,
                             check=True, env={**os.environ, "OPS": "1"}).stdout
        ops = {}
        for line in out.splitlines():
            if not line.strip():
                continue
            count, op, fn = line.split(None, 2)
            m = re.search(r"probe<(?:\(int\))?(\d)", fn)
            ops.setdefault(VARIANT.get(m[1], m[1]) if m else fn, {})[op] = int(count)
        for v, hist in ops.items():
            f.write(json.dumps({"experiment": "sass_opcodes", "binary": "mma_probe", "variant": v,
                                "top_opcodes": hist, "counts": "static", **prov}) + "\n")
            print(f"  ops  {v:9s} " + " ".join(f"{o}={c}" for o, c in sorted(hist.items(), key=lambda x: -x[1])[:6]))
    print(path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
