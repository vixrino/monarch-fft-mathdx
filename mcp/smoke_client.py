#!/usr/bin/env python3
# End-to-end check of the MCP server on the GPU machine: launches mcp/server.py over stdio, calls every tool,
# then runs diagnose on three copies of the repo with a known injected bug each, as an agent would see them:
#   compile   an undefined identifier in the Monarch kernel                    -> stage "build"
#   numeric   a conjugated twiddle                                              -> stage "accuracy"
#   no_mma    fp16 back on cuBLASDx's default layouts (correct, but no HMMA)    -> stage "tensor_cores"
# Usage (after the normal build): MATHDX_ROOT=... python3 mcp/smoke_client.py [--skip-injected]
import asyncio
import json
import os
import shutil
import sys
import tempfile

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL = "cuda/monarch/monarch_fft.cu"
BUGS = {
    "compile": ("a2(f * n1 + j1, i2) = data(", "a3(f * n1 + j1, i2) = data("),
    "numeric": ("re * w.y + im * w.x)", "-(re * w.y + im * w.x))"),
    "no_mma": ("static constexpr bool suggest = std::is_same_v<T, __half>;", "static constexpr bool suggest = false;"),
}


def copy_repo(bug):
    d = tempfile.mkdtemp(prefix=f"dxb_{bug}_")
    dst = os.path.join(d, "repo")
    shutil.copytree(REPO, dst, ignore=shutil.ignore_patterns("build*", ".git", "results", "__pycache__"))
    p = os.path.join(dst, KERNEL)
    src = open(p).read()
    old, new = BUGS[bug]
    assert src.count(old) == 1, f"injection point for {bug} not found once"
    open(p, "w").write(src.replace(old, new))
    return dst


# The SDK's stdio client forwards only a few safe variables (PATH, HOME, ...). The CUDA driver library is found
# through LD_LIBRARY_PATH on Colab, so without it nvidia-smi and every CUDA binary fail inside the server.
FORWARD = ("PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "HOME")


def params(repo):
    env = {k: os.environ[k] for k in FORWARD if k in os.environ}
    env.update(DXB_REPO=repo, MATHDX_ROOT=os.environ["MATHDX_ROOT"])
    return StdioServerParameters(command=sys.executable, args=[os.path.join(repo, "mcp", "server.py")], env=env)


async def call(c, tool, **args):
    r = await c.call_tool(tool, args)
    text = r.content[0].text
    return {"ok": False, "tool_error": text[:800]} if r.is_error else json.loads(text)


def show(title, d, keys):
    print(f"== {title}")
    print(json.dumps({k: d.get(k) for k in keys}, indent=1)[:2500])


async def clean():
    async with Client(params(REPO)) as c:
        print("tools:", sorted(t.name for t in (await c.list_tools()).tools))
        print("resources:", sorted(str(r.uri) for r in (await c.list_resources()).resources))
        llms = await c.read_resource("mathdx://cublasdx/llms.txt")
        print("cublasdx llms.txt first line:", llms.contents[0].text.splitlines()[0])
        e = await call(c, "env_check")
        show("env_check", {**e, "output": e.get("output", "")[-300:]}, ["ok", "returncode", "seconds", "output"])
        show("sass_check rejects a shell string", await call(c, "sass_check", binary="monarch_fft",
                                                              kernel_filter="x; rm -rf /"), ["ok", "rejected"])
        d = await call(c, "diagnose", binary="monarch_fft")
        show("diagnose, clean repo", d, ["stage", "ok", "failing", "mismatches", "stderr_tail", "fp16_kernels_without_mma"])
        s = await call(c, "sass_check", binary="monarch_fft", kernel_filter="__half")
        show("sass_check fp16 monarch", s, ["ok", "kernels_with_mma"])
        b = await call(c, "benchmark", binary="monarch_fft", key="monarch:fp16:256:16x16")
        rec = [(r["batch"], r["gbps_eff"], r["rel_l2_median"]) if r.get("status") == "ok" else r.get("status")
               for r in b.get("records", [])]
        show("benchmark monarch:fp16:256:16x16 (batch, GB/s, rel L2)", {**b, "records": rec}, ["ok", "records", "note"])
        show("run_tests before building cufftdx", await call(c, "run_tests", binary="cufftdx_block_fft"), ["ok", "rejected"])
        show("build cufftdx_block_fft", await call(c, "build", target="cufftdx_block_fft"), ["ok", "seconds", "errors"])
        t = await call(c, "run_tests", binary="cufftdx_block_fft")
        show("run_tests cufftdx", {**t, "n_configs": len(t.get("configs", []))}, ["ok", "n_configs", "mismatches"])


async def injected(bug):
    repo = copy_repo(bug)
    async with Client(params(repo)) as c:
        d = await call(c, "diagnose", binary="monarch_fft")
    keys = {"compile": ["stage", "errors", "docs"], "numeric": ["stage", "failing", "mismatches", "docs"],
            "no_mma": ["stage", "fp16_kernels_without_mma", "docs"]}[bug]
    if bug == "compile":
        d["errors"] = d.get("errors", [])[:2]
    if bug == "numeric":
        d["failing"], d["mismatches"] = d.get("failing", [])[:3], d.get("mismatches", [])[:3]
    show(f"diagnose with injected bug: {bug}", d, keys)
    shutil.rmtree(os.path.dirname(repo))


async def main():
    await clean()
    if "--skip-injected" not in sys.argv:
        for bug in BUGS:
            await injected(bug)


if __name__ == "__main__":
    asyncio.run(main())
