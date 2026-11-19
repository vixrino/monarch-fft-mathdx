# MCP server checks that need no GPU: argument whitelisting, output parsers, and an in-process MCP session.
import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp"))
import doc_links  # noqa: E402
import runner  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")


@pytest.fixture
def r(monkeypatch, tmp_path):
    (tmp_path / "build_mcp").mkdir()
    for b in runner.BINARIES:
        (tmp_path / "build_mcp" / b).touch()  # runs are mocked; the files only need to exist
    rn = runner.Runner(str(tmp_path), "/nonexistent/mathdx")
    calls = []
    monkeypatch.setattr(rn, "run", lambda argv, timeout: calls.append(argv) or runner.Result(argv, 0, 0.0, "", ""))
    rn.calls = calls
    return rn


@pytest.mark.parametrize("bad", ["all; rm -rf /", "../monarch_fft", "monarch_fft ", "bash", ""])
def test_rejects_unknown_targets(r, bad):
    with pytest.raises(runner.Rejected):
        r.build_target(bad)
    assert r.calls == []


@pytest.mark.parametrize("key", ["monarch:fp16:4096:64x64; ls", "monarch:fp64:256:16x16", "cufftdx:fp16:256:16x16",
                                 "monarch:fp16:256", "$(id)", "monarch:fp16:256:16x16\n"])
def test_rejects_bad_keys(r, key):
    with pytest.raises(runner.Rejected):
        r.benchmark("monarch_fft", key)
    assert r.calls == []


def test_key_must_match_binary(r):
    with pytest.raises(runner.Rejected):
        r.benchmark("cufftdx_block_fft", "monarch:fp16:256:16x16")


@pytest.mark.parametrize("f", ["x; rm", "a|b", "$(id)", "`id`", "a&&b", "x" * 81, "-o/tmp/x"])
def test_rejects_bad_sass_filter(r, f):
    with pytest.raises(runner.Rejected):
        r.sass("monarch_fft", f)
    assert r.calls == []


def test_commands_are_argv_lists(r):
    r.build_target("monarch_fft", "75")
    r.sass("monarch_fft", "__half, (unsigned int)64")
    assert all(isinstance(a, list) for a in r.calls)
    assert r.calls[-1][-1] == "__half, (unsigned int)64"  # passed as one argv element, never through a shell


def test_real_timeout(tmp_path):
    rn = runner.Runner(str(tmp_path), "")
    res = rn.run(["sleep", "5"], timeout=0.2)
    assert res.timed_out and res.returncode == -1


NVCC_LOG = """\
/content/monarch-fft-mathdx/cuda/monarch/monarch_fft.cu(95): error: identifier "a3" is undefined
          detected during instantiation of "void monarch_fft<M>(...) [with M=monarch<__half, 64U, 64U, 1U>]" at line 200
/content/monarch-fft-mathdx/cuda/monarch/monarch_fft.cu(95): error: identifier "a3" is undefined
          detected during instantiation of "void monarch_fft<M>(...) [with M=monarch<float, 64U, 64U, 1U>]" at line 200
/x/include/cublasdx/operators.hpp(42): error: static assertion failed with "cublasdx::Size: M must be > 0"
In file included from foo.cpp:1:
bar.hpp:10:5: error: 'suggest_layout_smem_q' is not a member of 'G'
2 errors detected in the compilation of "monarch_fft.cu".
"""


def test_compiler_errors():
    e = runner.compiler_errors(NVCC_LOG)
    assert [(x["file"].split("/")[-1], x["line"]) for x in e] == [("monarch_fft.cu", 95), ("operators.hpp", 42), ("bar.hpp", 10)]
    assert e[0]["context"][0].startswith("detected during")
    titles = {d["title"] for d in doc_links.for_errors(e)}
    assert "cuBLASDx operators" in titles and "cuBLASDx shared-memory tensors and layouts" in titles


CHECK_LOG = """\
monarch fp16 N=  256  16x16  fpb=16 smem= 17408  rel_l2=3.731e-04 max_abs=2.088e-02  ok
monarch fp16 N= 4096  64x64  fpb= 1 smem= 32768  rel_l2=7.112e-01 max_abs=4.100e+01  FAIL
MISMATCH key=monarch:fp16:4096:64x64 seed=0 fft=0 k=64 got=(1.234000e+00,-5.000000e-01) ref=(3.141500e+01,2.000000e+00) rel_l2=7.000e-01 tol=1.0e-02
monarch fp32 N= 2048  32x64  fpb= 2 smem= 73728  skipped: shared memory
ACCURACY CHECK FAILED
"""


def test_accuracy_parser():
    configs, mm = runner.accuracy(CHECK_LOG)
    assert [c.get("pass") for c in configs] == [True, False, None]
    assert configs[2]["skipped"] == "shared memory"
    assert mm == [{"key": "monarch:fp16:4096:64x64", "seed": 0, "fft": 0, "k": 64, "got": [1.234, -0.5],
                   "ref": [31.415, 2.0], "rel_l2_seed0": 0.7, "tol": 0.01}]


def test_sass_parser():
    out = ("  total    mma   ffma  hfma2  function\n"
           "   2264    128     64      0  void monarch_fft<monarch<__half, (unsigned int)16, (unsigned int)16>>(\n"
           "    968      0      0    128  block_fft N=1024 __half\n")
    rows = runner.sass_rows(out)
    assert [(x["total"], x["mma"]) for x in rows] == [(2264, 128), (968, 0)]


def test_mcp_session_in_process(monkeypatch):
    mcp = pytest.importorskip("mcp.client")  # the repo's mcp/ folder imports as a namespace package
    from mcp.client import Client

    monkeypatch.setenv("DXB_REPO", REPO)
    sys.path.insert(0, os.path.join(REPO, "mcp"))
    import server

    async def go():
        async with Client(server.server) as c:
            tools = {t.name for t in (await c.list_tools()).tools}
            assert tools == {"env_check", "build", "run_tests", "benchmark", "sass_check", "diagnose"}
            res = await c.call_tool("benchmark", {"binary": "monarch_fft", "key": "monarch:fp16:256:16x16; id"})
            assert json.loads(res.content[0].text)["rejected"].startswith("key")  # dict results arrive as JSON text
            guide = await c.read_resource("monarch-fft-mathdx://agent_guides/cufftdx")
            assert "UNOFFICIAL" in guide.contents[0].text
            with pytest.raises(Exception, match="Unknown resource"):  # no path traversal out of the docs folder
                await c.read_resource("mathdx://cublasdx/markdown/..%2F..%2Fsecret")
    asyncio.run(go())


def test_unbuilt_binary_is_a_clean_rejection(tmp_path):
    rn = runner.Runner(str(tmp_path), "")
    with pytest.raises(runner.Rejected, match="not built"):
        rn.check_accuracy("monarch_fft")
