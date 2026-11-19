# Whitelisted command execution and output parsers for the monarch-fft-mathdx MCP server. No MCP dependency, so the
# validation and parsing logic is unit-tested without a GPU (tests/test_mcp_runner.py).
#
# Safety model: every command is a fixed argv list built here (never a shell string), every user-supplied
# value is checked against an enum or a strict regex before it reaches an argv, and every call has a timeout.
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass

TARGETS = ("all", "monarch_fft", "cufftdx_block_fft", "mma_probe", "dx_smoke")
BINARIES = ("monarch_fft", "cufftdx_block_fft")
ARCHS = ("75", "80", "86", "89", "90")
KEY = re.compile(r"^(monarch:(fp16|fp32):\d{2,5}:\d{1,3}x\d{1,3}|cufftdx:(fp16|fp32):\d{2,5})$")
SASS_FILTER = re.compile(r"^[A-Za-z0-9_ ,:()<>=.]{0,80}$")
MAX_OUT = 20_000  # characters of raw output returned to the client


class Rejected(ValueError):
    """An argument outside the whitelist; raised before anything runs."""


@dataclass
class Result:
    argv: list
    returncode: int
    seconds: float
    stdout: str
    stderr: str
    timed_out: bool = False


def check(value, allowed, what):
    if value not in allowed:
        raise Rejected(f"{what} must be one of {list(allowed)}, got {value!r}")
    return value


def check_re(value, pattern, what):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise Rejected(f"{what} {value!r} does not match {pattern.pattern}")
    return value


class Runner:
    def __init__(self, repo, mathdx_root, build_dir="build_mcp"):
        self.repo = os.path.realpath(repo)
        self.mathdx_root = mathdx_root
        self.build = os.path.join(self.repo, build_dir)

    def run(self, argv, timeout):
        t0 = time.monotonic()
        try:
            p = subprocess.run(argv, cwd=self.repo, capture_output=True, text=True, timeout=timeout, shell=False)
            return Result(argv, p.returncode, time.monotonic() - t0, p.stdout, p.stderr)
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            return Result(argv, -1, time.monotonic() - t0, out, err, timed_out=True)

    def env_check(self):
        return self.run(["bash", "scripts/env_check.sh"], timeout=120)

    def configure(self, cuda_arch):
        check(cuda_arch, ARCHS, "cuda_arch")
        if not self.mathdx_root:
            raise Rejected("MATHDX_ROOT is not set in the server environment")
        return self.run(["cmake", "-S", ".", "-B", self.build, f"-DCUDA_ARCH={cuda_arch}",
                         f"-DMATHDX_ROOT={self.mathdx_root}"], timeout=300)

    def build_target(self, target, cuda_arch="75"):
        check(target, TARGETS, "target")
        c = self.configure(cuda_arch)
        if c.returncode != 0:
            return c
        argv = ["cmake", "--build", self.build, "-j"] + ([] if target == "all" else ["--target", target])
        return self.run(argv, timeout=900)

    def binary(self, name):
        path = os.path.join(self.build, check(name, BINARIES + ("mma_probe",), "binary"))
        if not os.path.isfile(path):
            raise Rejected(f"{os.path.relpath(path, self.repo)} is not built; call build(target={name!r}) first")
        return path

    def check_accuracy(self, binary):
        return self.run([self.binary(binary), "--check"], timeout=600)

    def benchmark(self, binary, key):
        check(binary, BINARIES, "binary")
        check_re(key, KEY, "key")
        if not key.startswith(binary.split("_")[0]):
            raise Rejected(f"key {key!r} does not belong to {binary}")
        fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="mcp_bench_")
        os.close(fd)
        try:
            r = self.run([self.binary(binary), "--only", key, "--out", path], timeout=600)
            with open(path) as f:
                records = [json.loads(l) for l in f if l.strip()]
        finally:
            os.unlink(path)
        return r, records

    def sass(self, binary, filt=""):
        check(binary, BINARIES + ("mma_probe",), "binary")
        check_re(filt, SASS_FILTER, "filter")
        argv = ["bash", "scripts/sass_check.sh", self.binary(binary)] + ([filt] if filt else [])
        return self.run(argv, timeout=180)


# nvcc / gcc diagnostics: path(line): error: msg   or   path:line:col: error: msg
NVCC_ERR = re.compile(r"^(?P<file>[^\s(:][^(:]*)\((?P<line>\d+)\): (?P<kind>error|catastrophic error)\s*:?\s*(?P<msg>.*)$")
GCC_ERR = re.compile(r"^(?P<file>[^\s:][^:]*):(?P<line>\d+):(?:(?P<col>\d+):)? (?P<kind>error|fatal error): (?P<msg>.*)$")
CMAKE_ERR = re.compile(r"^CMake Error(?: at (?P<file>[^:]+):(?P<line>\d+))?.*$")


def compiler_errors(text, limit=20):
    errs, seen = [], set()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = NVCC_ERR.match(line) or GCC_ERR.match(line)
        if m:
            d = m.groupdict()
            # nvcc prints either "detected during instantiation of ..." or "detected during:" plus an indented list
            ctx = [l.strip() for l in lines[i + 1:i + 5]
                   if l.startswith((" ", "\t")) and not (NVCC_ERR.match(l) or GCC_ERR.match(l))][:3]
            e = {"file": d["file"], "line": int(d["line"]), "message": d["msg"].strip(), "context": ctx}
        elif m := CMAKE_ERR.match(line):
            e = {"file": m["file"] or "CMakeLists.txt", "line": int(m["line"] or 0),
                 "message": " ".join(l.strip() for l in lines[i + 1:i + 3] if l.strip())}
        else:
            continue
        k = (e["file"], e["line"], e["message"])
        if k not in seen:
            seen.add(k)
            errs.append(e)
        if len(errs) >= limit:
            break
    return errs


CHECK_LINE = re.compile(r"^(?P<label>(monarch|cufftdx) .*?)\s+rel_l2=(?P<rel>\S+) max_abs=(?P<mx>\S+)(?:\s+compute_rel_l2=\S+)?\s+(?P<status>ok|FAIL)$")
SKIP_LINE = re.compile(r"^(?P<label>(monarch|cufftdx) .*?)\s+skipped: (?P<reason>.*)$")
MISMATCH = re.compile(r"^MISMATCH key=(?P<key>\S+) seed=(?P<seed>\d+) fft=(?P<fft>\d+) k=(?P<k>\d+) "
                      r"got=\((?P<gr>\S+),(?P<gi>\S+)\) ref=\((?P<rr>\S+),(?P<ri>\S+)\) "
                      r"rel_l2=(?P<rel>\S+) tol=(?P<tol>\S+)$")


def accuracy(text):
    configs, mismatches = [], []
    for line in text.splitlines():
        if m := CHECK_LINE.match(line.strip()):
            configs.append({"config": " ".join(m["label"].split()), "rel_l2": float(m["rel"]),
                            "max_abs": float(m["mx"]), "pass": m["status"] == "ok"})
        elif m := SKIP_LINE.match(line.strip()):
            configs.append({"config": " ".join(m["label"].split()), "skipped": m["reason"]})
        elif m := MISMATCH.match(line.strip()):
            mismatches.append({"key": m["key"], "seed": int(m["seed"]), "fft": int(m["fft"]), "k": int(m["k"]),
                               "got": [float(m["gr"]), float(m["gi"])], "ref": [float(m["rr"]), float(m["ri"])],
                               "rel_l2_seed0": float(m["rel"]), "tol": float(m["tol"])})
    return configs, mismatches


def sass_rows(text):
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split(None, 4)
        if len(parts) == 5 and parts[0].isdigit():
            rows.append({"function": parts[4], "total": int(parts[0]), "mma": int(parts[1]),
                         "ffma": int(parts[2]), "hfma2": int(parts[3])})
    return rows


def clip(s):
    return s if len(s) <= MAX_OUT else s[:MAX_OUT // 4] + "\n...[clipped]...\n" + s[-3 * MAX_OUT // 4:]
