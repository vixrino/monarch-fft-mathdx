#!/usr/bin/env python3
# monarch-fft-mathdx MCP server: build, test, benchmark and inspect the MathDx kernels of this repo on the GPU machine.
#   DXB_REPO=/path/to/repo MATHDX_ROOT=/path/to/nvidia/mathdx/26.10 python3 mcp/server.py      (stdio)
# The client chooses among fixed tools with enum / regex-checked arguments; it never passes a command line.
import os
import sys
from typing import Literal

from mcp.server.mcpserver import MCPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doc_links  # noqa: E402
from runner import Rejected, Runner, accuracy, clip, compiler_errors, sass_rows  # noqa: E402

REPO = os.environ.get("DXB_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MATHDX_ROOT = os.environ.get("MATHDX_ROOT", "")
runner = Runner(REPO, MATHDX_ROOT)

Target = Literal["all", "monarch_fft", "cufftdx_block_fft", "mma_probe", "dx_smoke"]
Binary = Literal["monarch_fft", "cufftdx_block_fft"]
Arch = Literal["75", "80", "86", "89", "90"]

server = MCPServer(
    name="monarch-fft-mathdx",
    instructions=(
        "Tools to build and validate cuFFTDx / cuBLASDx kernels in the monarch-fft-mathdx repo on a GPU machine. "
        "Typical loop: edit source, call diagnose(binary); it builds, checks accuracy against an FP64 cuFFT "
        "reference and checks the SASS for tensor-core instructions, and returns the first failing stage with "
        "links to the NVIDIA docs section to read. Resources: the official cuBLASDx agent guide (llms.txt and "
        "markdown) from the installed MathDx package, and an unofficial cuFFTDx guide written for this repo."
    ),
)


def summary(r):
    return {"argv": r.argv, "returncode": r.returncode, "seconds": round(r.seconds, 1), "timed_out": r.timed_out}


def rejected(e):
    return {"ok": False, "rejected": str(e)}


@server.tool()
def env_check() -> dict:
    """GPU, driver, nvcc, compute capability, MathDx version and whether ncu counters are readable."""
    r = runner.env_check()
    return {"ok": r.returncode == 0, **summary(r), "output": clip(r.stdout + r.stderr)}


@server.tool()
def build(target: Target = "all", cuda_arch: Arch = "75") -> dict:
    """Configure (CMake, Release) and build one target. Returns parsed compiler errors, not just raw logs."""
    try:
        r = runner.build_target(target, cuda_arch)
    except Rejected as e:
        return rejected(e)
    log = r.stdout + r.stderr
    errors = compiler_errors(log)
    return {"ok": r.returncode == 0, **summary(r), "errors": errors, "docs": doc_links.for_errors(errors),
            "log_tail": clip(log[-4000:])}


@server.tool()
def run_tests(binary: Binary = "monarch_fft") -> dict:
    """Accuracy check of every configuration in a binary against cuFFT Z2Z in FP64 (median rel L2 over 5 seeds).
    Failing configurations come with the worst element: FFT index, frequency k, value and reference."""
    try:
        r = runner.check_accuracy(binary)
    except Rejected as e:
        return rejected(e)
    configs, mismatches = accuracy(r.stderr)
    return {"ok": r.returncode == 0 and not r.timed_out, **summary(r), "configs": configs, "mismatches": mismatches}


@server.tool()
def benchmark(binary: Binary, key: str) -> dict:
    """Time one configuration: key is monarch:<fp16|fp32>:<N>:<n1>x<n2> or cufftdx:<fp16|fp32>:<N>.
    Returns the JSON records the binary writes (median and IQR over 50 launches, batch = one block and ~2^14)."""
    try:
        r, records = runner.benchmark(binary, key)
    except Rejected as e:
        return rejected(e)
    return {"ok": r.returncode == 0 and bool(records), **summary(r), "records": records,
            "note": "" if records else "no record: key not in this binary's sweep, or skipped"}


@server.tool()
def sass_check(binary: Binary, kernel_filter: str = "") -> dict:
    """Static SASS opcode counts per kernel (total, MMA, FFMA, HFMA2) from cuobjdump. MMA > 0 is the proof that
    a kernel uses tensor cores; these are static counts, not executed instructions."""
    try:
        r = runner.sass(binary, kernel_filter)
    except Rejected as e:
        return rejected(e)
    rows = sass_rows(r.stdout)
    return {"ok": r.returncode == 0, **summary(r), "kernels": rows,
            "kernels_with_mma": sum(1 for k in rows if k["mma"] > 0)}


@server.tool()
def diagnose(binary: Binary = "monarch_fft", cuda_arch: Arch = "75") -> dict:
    """Build, then check accuracy, then check tensor-core use; stop at the first failing stage and return what
    failed (compiler errors, failing indices and errors, or kernels without MMA) plus the doc sections to read."""
    try:
        b = runner.build_target(binary, cuda_arch)
    except Rejected as e:
        return rejected(e)
    if b.returncode != 0:
        log = b.stdout + b.stderr
        errors = compiler_errors(log)
        return {"stage": "build", "ok": False, **summary(b), "errors": errors,
                "docs": doc_links.for_errors(errors) if errors else (doc_links.for_kind("timeout") if b.timed_out else []),
                "log_tail": clip(log[-4000:]) if not errors else ""}

    t = runner.check_accuracy(binary)
    configs, mismatches = accuracy(t.stderr)
    if t.timed_out or t.returncode != 0:
        kind = "timeout" if t.timed_out else ("accuracy_monarch" if binary == "monarch_fft" else "accuracy_cufftdx")
        return {"stage": "accuracy", "ok": False, **summary(t), "failing": [c for c in configs if c.get("pass") is False],
                "mismatches": mismatches, "docs": doc_links.for_kind(kind),
                "stderr_tail": "" if mismatches else clip(t.stderr[-4000:])}

    if binary == "monarch_fft":
        s = runner.sass(binary, "__half")
        rows = sass_rows(s.stdout)
        missing = [k["function"] for k in rows if "monarch_fft" in k["function"] and k["mma"] == 0]
        if missing or not rows:
            return {"stage": "tensor_cores", "ok": False, "fp16_kernels_without_mma": missing,
                    "note": "accuracy passes, but these fp16 kernels compiled to scalar code",
                    "docs": doc_links.for_kind("no_tensor_cores")}
    return {"stage": "ok", "ok": True, "configs": configs}


def package_docs():
    return os.path.join(MATHDX_ROOT, "doc", "cublasdx", "html")


@server.resource("mathdx://cublasdx/llms.txt", mime_type="text/plain",
                 description="Official cuBLASDx llms.txt, from the installed MathDx package")
def cublasdx_llms() -> str:
    with open(os.path.join(package_docs(), "llms.txt")) as f:
        return f.read()


@server.resource("mathdx://cublasdx/markdown/{page}", mime_type="text/markdown",
                 description="Official cuBLASDx Agent BLAS Performance Guide page (markdown), e.g. "
                             "performance_guide_api_supermma.md")
def cublasdx_markdown(page: str) -> str:
    d = os.path.join(package_docs(), "markdown")
    pages = sorted(p for p in os.listdir(d) if p.endswith(".md"))
    if page not in pages:
        return f"unknown page {page!r}; available: {', '.join(pages)}"
    with open(os.path.join(d, page)) as f:
        return f.read()


@server.resource("monarch-fft-mathdx://agent_guides/cufftdx", mime_type="text/markdown",
                 description="UNOFFICIAL cuFFTDx agent guide written for this repo (NVIDIA publishes none)")
def cufftdx_guide() -> str:
    with open(os.path.join(REPO, "agent_guides", "cufftdx_agent_guide.md")) as f:
        return f.read()


if __name__ == "__main__":
    server.run("stdio")
