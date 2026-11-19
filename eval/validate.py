#!/usr/bin/env python3
# Validate every eval task with its reference solution, on the GPU, before any agent sees it:
#   1. the hidden test builds against hidden/reference.cu and passes;
#   2. the public variant (-DEVAL_PUBLIC) builds and passes;
#   3. tasks with requires_mma have HMMA in the reference binary's SASS.
# The hidden test's timing becomes the task's reference timing, written with provenance to
# results/<gpu>/<date>_eval_reference[_k].jsonl.
#   MATHDX_ROOT=... python3 eval/validate.py [task_id ...]
import datetime
import glob
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "eval")


def nvcc_cmd(task_dir, sources, out, extra=(), link=True):
    mx = os.environ["MATHDX_ROOT"]
    return (["nvcc", "-std=c++17", "--expt-relaxed-constexpr", "-DCUBLASDX_NO_FATBIN_AVAILABLE", "-O3", "-DNDEBUG",
             "-arch=sm_75", f"-I{mx}/include", f"-I{mx}/external/cutlass/include", f"-I{EVAL}/common",
             f"-I{task_dir}"] + list(extra) + sources + (["-lcufft"] if link else ["-c"]) + ["-o", out])


def sh(argv, timeout, env=None):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, "", f"timed out after {timeout} s"
    return p.returncode, p.stdout, p.stderr


def run_test(exe, timeout):
    # the test writes its JSON line to $EVAL_RESULT; stdout belongs to the solution too, so it is not trusted
    path = exe + ".result.json"
    if os.path.exists(path):
        os.remove(path)
    rc, so, se = sh([exe], timeout, {**os.environ, "EVAL_RESULT": path})
    try:
        res = json.load(open(path))
    except (OSError, ValueError):
        return {"status": "no_result", "returncode": rc, "stdout": so[-1000:], "stderr": se[-1500:]}
    if rc != 0 and res.get("status") == "pass":
        res = {**res, "status": "crash_after_pass", "returncode": rc, "stderr": se[-1500:]}
    return res


def hmma_lines(exe):
    sass = sh(["cuobjdump", "-sass", exe], 300)[1]
    return sum("HMMA" in l for l in sass.splitlines())


def provenance():
    q = lambda *a: subprocess.run(a, capture_output=True, text=True).stdout.strip()
    return {"gpu": q("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").splitlines()[0],
            "driver": q("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").splitlines()[0],
            "nvcc": q("nvcc", "--version").splitlines()[-1],
            "mathdx_root": os.environ["MATHDX_ROOT"],
            "git_commit": q("git", "-C", ROOT, "rev-parse", "--short=7", "HEAD"),
            "git_dirty": bool(q("git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no", "--", ".",
                                 ":(exclude)results", ":(exclude)docs/figures")),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def validate(task_dir, tmp):
    meta = json.load(open(os.path.join(task_dir, "meta.json")))
    test, ref = os.path.join(task_dir, "hidden", "test.cu"), os.path.join(task_dir, "hidden", "reference.cu")
    out = {"experiment": "eval_reference", "task": meta["id"], "split": meta["split"]}
    for variant, extra in (("hidden", ()), ("public", ("-DEVAL_PUBLIC",))):
        exe = os.path.join(tmp, f"{meta['id']}_{variant}")
        rc, so, se = sh(nvcc_cmd(task_dir, [test, ref], exe, extra), 900)
        if rc != 0:
            out[variant] = {"status": "build_error", "log": (so + se)[-3000:]}
            continue
        out[variant] = run_test(exe, 600)
        if variant == "hidden" and meta.get("requires_mma"):
            out["hmma_lines"] = hmma_lines(exe)
    ok = all(out.get(v, {}).get("status") == "pass" for v in ("hidden", "public"))
    if meta.get("requires_mma"):
        ok &= out.get("hmma_lines", 0) > 0
    out["valid"] = ok
    return out


def main():
    ids = sys.argv[1:] or sorted(os.path.basename(d) for d in glob.glob(os.path.join(EVAL, "tasks", "*")))
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.splitlines()[0].strip().lower().replace(" ", "_")
    d = os.path.join(ROOT, "results", gpu)
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"{datetime.date.today().isoformat()}_eval_reference")
    path, k = f"{stem}.jsonl", 2
    while os.path.exists(path):
        path, k = f"{stem}_{k}.jsonl", k + 1
    prov = provenance()
    all_ok = True
    with tempfile.TemporaryDirectory() as tmp, open(path, "w") as f:
        for t in ids:
            r = validate(os.path.join(EVAL, "tasks", t), tmp)
            f.write(json.dumps({**r, **prov}) + "\n")
            h = r.get("hidden", {})
            print(f"{t:22s} valid={r['valid']!s:5s} hidden={h.get('status')} public={r.get('public', {}).get('status')} "
                  f"time_ms={h.get('time_ms')} hmma={r.get('hmma_lines', '-')}", flush=True)
            if not r["valid"]:
                all_ok = False
                print(json.dumps(r, indent=1)[:3000], flush=True)
    print(path)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
