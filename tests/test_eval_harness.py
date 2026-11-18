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
    return harness.Run(str(task), {"id": "t"}, corpus, str(tmp_path))


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
    assert run.grade() == {"compiles": False, "correct": False, "hidden": None}
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
