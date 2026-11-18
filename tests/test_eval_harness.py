import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
harness = pytest.importorskip("harness")

SASS = """
\tFunction : _Z4gemmPK6__halfS1_Pfi
\t.headerflags    @"EF_CUDA_SM75 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM75)"
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   HMMA.1688.F32 R4, R8, R12, R4 ;
        /*0020*/              @!P0 FFMA R2, R3, R4, R2 ;
        /*0030*/                   HFMA2.MMA R5, -RZ, RZ, 0, 0 ;
\tFunction : _Z3fftP6float2
        /*0000*/                   FFMA R2, R3, R4, R2 ;
        /*0010*/               @P1 EXIT ;
"""


def test_parse_sass_counts_per_kernel():
    a, b = harness.parse_sass(SASS)
    assert (a["total"], a["mma"], a["ffma"], a["hfma2"]) == (4, 1, 1, 1)
    assert (b["total"], b["mma"], b["ffma"]) == (2, 0, 1)


@pytest.fixture
def run(tmp_path):
    mx = tmp_path / "mx"
    (mx / "doc" / "cublasdx" / "html").mkdir(parents=True)
    (mx / "doc" / "cublasdx" / "html" / "a.html").write_text("<html><script>x()</script><p>Use suggest_layout_smem_a here</p></html>")
    (mx / "include" / "cufftdx" / "database").mkdir(parents=True)
    (mx / "include" / "cufftdx" / "database" / "big.hpp.inc").write_text("suggest_layout_smem_a\n")
    (mx / "include" / "cufftdx" / "fft.hpp").write_text("line1\nstruct FFTsPerBlock;\n")
    task = tmp_path / "task"
    (task / "hidden").mkdir(parents=True)
    (task / "task.h").write_text("void solve();\n")
    corpus = harness.load_corpus(str(mx))
    return harness.Run(str(task), {"id": "t", "split": "dev", "library": "cufftdx"}, corpus, str(tmp_path))


def test_corpus_skips_database_and_strips_html(run):
    assert set(run.corpus) == {"doc/cublasdx/html/a.html", "include/cufftdx/fft.hpp"}
    assert "x()" not in " ".join(run.corpus["doc/cublasdx/html/a.html"])


def test_search_and_read(run):
    r = run.search_docs("ffts_PER")
    assert r["n_hits"] == 0
    r = run.search_docs("FFTsPerBlock")
    assert r["hits"] == ["include/cufftdx/fft.hpp:2: struct FFTsPerBlock;"]
    assert run.read_doc("include/cufftdx/fft.hpp", 2, 1)["text"] == "2: struct FFTsPerBlock;"
    bad = run.read_doc("../../etc/passwd")
    assert "error" in bad and "text" not in bad


def test_tools_before_write(run):
    assert run.compile()["ok"] is False
    assert run.grade() == {"compiles": False, "correct": False, "solved": False, "hidden": None}
    assert run.write_solution("int x;")["ok"] and os.path.exists(run.sol)
    r = run.write_solution("```cpp\n#include <cufftdx.hpp>\nint x;\n```")
    assert r["stripped_markdown_fence"] and open(run.sol).read() == "#include <cufftdx.hpp>\nint x;"
    assert not run.write_solution("int y; // ```")["stripped_markdown_fence"]


def test_misuse_pattern_ignores_case():
    assert harness.MATHDX.search("cufftDx.h: No such file or directory")
    assert harness.MATHDX.search("cuda_mathdx.h: No such file or directory")


def test_conditions_nest():
    assert set(harness.CONDITIONS["A"]) < set(harness.CONDITIONS["B"]) < set(harness.CONDITIONS["C"])
    assert all(n in harness.TOOLS for n in harness.CONDITIONS["C"])


def test_library_check(run):
    run.write_solution('#include <cufftdx.hpp>\nusing namespace cufftdx;\nvoid solve() {}')
    assert run.library_check()["library_used"]
    run.write_solution('#include <cufft.h>\n#include <cufftdx.hpp>\nvoid solve() { cufftExecC2C(p, a, b, 1); }')
    c = run.library_check()
    assert not c["library_used"] and len(c["host_library_calls"]) == 2
    run.write_solution('#include "cublasdx.hpp"\nvoid solve() {}')
    assert run.library_check()["library_missing"] == ["cufftdx"]


class Msg:
    def __init__(self, calls):
        self.tool_calls = [type("TC", (), {"function": type("F", (), {"name": n, "arguments": a})()})() for n, a in calls]
        self.content = ""

    def model_dump(self, **_):
        return {"role": "assistant", "content": ""}


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, **_):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return type("R", (), {"message": r, "prompt_eval_count": 100, "eval_count": 10})()


@pytest.fixture
def task(run, monkeypatch):
    open(os.path.join(run.task_dir, "spec.md"), "w").write("spec")
    monkeypatch.setattr(harness.Run, "grade", lambda self: {"compiles": False, "correct": False, "solved": False,
                                                              "hidden": None, "written": os.path.exists(self.sol)})
    return run


def test_submit_runs_after_other_calls(task, tmp_path):
    c = FakeClient([Msg([("submit", {}), ("write_solution", {"code": "int x;"})])])
    rec, msgs = harness.agent(c, "m", "A", task.task_dir, task.meta, {}, str(tmp_path))
    assert rec["end"] == "submit" and rec["iterations"] == 1 and rec["written"]


def test_server_error_and_bad_args_do_not_crash(task, tmp_path):
    c = FakeClient([Msg([("search_docs", {"query": None}), ("read_doc", {"path": "x", "start": "abc"}),
                         ("compile", {"code": "solution.cu"}), ("nope", {})]), ConnectionError("down")])
    rec, msgs = harness.agent(c, "m", "B", task.task_dir, task.meta, {"x": ["a"]}, str(tmp_path))
    assert rec["end"].startswith("error: ConnectionError") and rec["iterations"] == 2
    outs = [m["content"] for m in msgs if m["role"] == "tool"]
    assert "ValueError" in outs[1] and "no solution.cu yet" in outs[2] and "unknown tool" in outs[3]


def fake_exe(tmp_path, body):
    p = tmp_path / "t.sh"
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(0o755)
    return str(p)


def test_run_test_ignores_stdout_and_checks_returncode(tmp_path):
    v = harness.validate
    spoof = fake_exe(tmp_path, 'echo \'{"status":"fail"}\' > "$EVAL_RESULT"\necho \'{"status":"pass"}\'\n')
    assert v.run_test(spoof, 10)["status"] == "fail"
    crash = fake_exe(tmp_path, 'echo \'{"status":"pass"}\' > "$EVAL_RESULT"\nexit 139\n')
    assert v.run_test(crash, 10)["status"] == "crash_after_pass"
    assert v.run_test(fake_exe(tmp_path, "echo hi\n"), 10)["status"] == "no_result"
    assert v.run_test(fake_exe(tmp_path, "sleep 5\n"), 1)["status"] == "no_result"
