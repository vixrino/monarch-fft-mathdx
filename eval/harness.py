#!/usr/bin/env python3
# One local model (ollama), one task at a time, under three conditions that differ only in the tools offered:
#   A  write_solution, compile, submit
#   B  A + search_docs, read_doc over the installed MathDx package (doc/, include/, example/)
#   C  B + run_public_test, sass_check, diagnose
# Same system prompt, same first message, same turn and context budget in every condition. The last solution.cu
# written is graded with the hidden test in the same session as the reference solution's timing.
#   MATHDX_ROOT=... python3 eval/harness.py [--tasks id ...] [--conditions A B C] [--model qwen3:8b]
import argparse
import datetime
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from html.parser import HTMLParser

import ollama

EVAL = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EVAL)
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, EVAL)
import doc_links  # noqa: E402
import validate  # noqa: E402
from runner import compiler_errors  # noqa: E402

MAX_TURNS = 16
NUM_CTX = 32768
CTX_STOP = 24000  # prompt tokens; leaves room for one more reply (NUM_PREDICT) and one tool result
NUM_PREDICT = 4096
MAX_TOOL_CHARS = 6000
OPTS = {"temperature": 0, "seed": 0, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}
MATHDX = re.compile(r"cufftdx|cublasdx|commondx|mathdx", re.I)
# host cuFFT / cuBLAS would solve the tasks without MathDx; the hidden test links cuFFT for its own reference
HOST_LIB = re.compile(r'#\s*include\s*[<"]((?:cufft|cufftXt|cublas|cublas_v2|cublasLt)\.h)[>"]|\b((?:cufft|cublas)[A-Z]\w*)\s*\(')
FENCE = re.compile(r"^\s*```[\w+-]*\n(.*?)\n?```\s*$", re.S)

SYSTEM = f"""You write CUDA C++ for an NVIDIA Tesla T4 (sm_75) with NVIDIA MathDx 26.10 (cuFFTDx 1.8.0, \
cuBLASDx 0.8.0), already installed on the include paths of the build command in the spec.
Implement the function declared in task.h in a single file, solution.cu, as the spec asks.
Write the complete file with write_solution each time you change it. Check your work with the other tools, then call \
submit. Your last written solution.cu is graded on hidden tests: correctness on sizes and seeds you do not see, \
then speed. You have at most {MAX_TURNS} turns."""

NUDGE = "Use a tool. Call submit when solution.cu is final."


def fn(name, desc, props=None, required=()):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": list(required)}}}


TOOLS = {
    "write_solution": fn("write_solution", "Replace solution.cu with the given complete source.",
                         {"code": {"type": "string"}}, ["code"]),
    "compile": fn("compile", "Compile solution.cu (nvcc -c, the spec's flags). Returns the parsed compiler errors."),
    "submit": fn("submit", "Finish. The last written solution.cu is graded."),
    "search_docs": fn("search_docs", "Case-insensitive search for a string in the installed MathDx package: "
                      "cuBLASDx docs (HTML, llms.txt, markdown guide), cuFFTDx/cuBLASDx headers, official examples. "
                      "Returns path:line: text hits.", {"query": {"type": "string"}}, ["query"]),
    "read_doc": fn("read_doc", "Read lines of one file from the MathDx package, by the path search_docs returns.",
                   {"path": {"type": "string"}, "start": {"type": "integer"}, "count": {"type": "integer"}},
                   ["path"]),
    "run_public_test": fn("run_public_test", "Build solution.cu with a small public test and run it on the GPU. "
                          "Returns pass/fail, relative L2 error per case and the index of the worst element."),
    "sass_check": fn("sass_check", "Compile solution.cu and count SASS instructions per kernel: total, tensor-core "
                     "MMA (HMMA), FFMA, HFMA2."),
    "diagnose": fn("diagnose", "Compile, run the public test, then check tensor-core use if the spec requires it. "
                   "Returns the first failing stage with details and the local doc pages to read."),
}
CONDITIONS = {"A": ["write_solution", "compile", "submit"]}
CONDITIONS["B"] = CONDITIONS["A"] + ["search_docs", "read_doc"]
CONDITIONS["C"] = CONDITIONS["B"] + ["run_public_test", "sass_check", "diagnose"]


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        self.skip += tag in ("script", "style")

    def handle_endtag(self, tag):
        self.skip -= tag in ("script", "style")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def load_corpus(mx):
    files = {}
    for top, exts in (("doc", (".md", ".txt", ".html")), ("include", (".hpp", ".h", ".cuh", ".inc")),
                      ("example", (".cu", ".cuh", ".hpp", ".md"))):
        for d, _, names in os.walk(os.path.join(mx, top)):
            if "database" in d.split(os.sep):  # cuFFTDx's generated kernel tables: 100 MB, no prose
                continue
            for n in names:
                if n.endswith(exts):
                    p = os.path.join(d, n)
                    text = open(p, errors="replace").read()
                    if n.endswith(".html"):
                        t = Text()
                        t.feed(text)
                        text = re.sub(r"\n\s*\n+", "\n\n", "".join(t.out))
                    files[os.path.relpath(p, mx)] = text.splitlines()
    return files


def clip(obj):
    s = json.dumps(obj) if not isinstance(obj, str) else obj
    return s if len(s) <= MAX_TOOL_CHARS else s[:MAX_TOOL_CHARS] + " ...[clipped]"


def sh(argv, timeout):
    try:
        return validate.sh(argv, timeout)
    except subprocess.TimeoutExpired:
        return None, "", f"timed out after {timeout} s"


def sass_counts(path):
    return parse_sass(sh(["cuobjdump", "-sass", path], 120)[1])


def parse_sass(text):
    rows, cur = [], None
    for line in text.splitlines():
        if m := re.match(r"\s*Function : (\S+)", line):
            cur = {"kernel": m[1][:100], "total": 0, "mma": 0, "ffma": 0, "hfma2": 0}
            rows.append(cur)
        elif cur and (m := re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)):
            op = m[1].split(".")[0]
            cur["total"] += 1
            cur["mma"] += op in ("HMMA", "MMA")
            cur["ffma"] += op == "FFMA"
            cur["hfma2"] += op == "HFMA2"
    return rows


def local_docs(errors):
    # doc_links points at docs.nvidia.com; only the cuBLASDx pages ship in the package, so map those to read_doc paths
    out = []
    for d in doc_links.for_errors(errors):
        if d["url"].startswith(doc_links.CB):
            page = d["url"][len(doc_links.CB):].split("#")[0]
            out.append({"title": d["title"], "path": f"doc/cublasdx/html/{page}"})
    return out


NO_MMA_DOCS = [{"title": t, "path": "doc/cublasdx/html/" + u[len(doc_links.CB):].split("#")[0]}
               for t, u in doc_links.KIND["no_tensor_cores"]]


class Run:
    def __init__(self, task_dir, meta, corpus, tmp):
        self.task_dir, self.meta, self.corpus = task_dir, meta, corpus
        self.work = tempfile.mkdtemp(dir=tmp, prefix=meta["id"] + "_")
        shutil.copy(os.path.join(task_dir, "task.h"), self.work)
        self.sol = os.path.join(self.work, "solution.cu")
        self.misuse, self.builds = set(), 0
        self.stub = os.path.join(self.work, "stub_main.cu")
        open(self.stub, "w").write('#include "task.h"\nint main() { return &solve == nullptr; }\n')

    def build(self, sources, out, extra=(), link=True, count_misuse=False):
        rc, so, se = sh(validate.nvcc_cmd(self.work, sources, out, extra, link), 600)
        self.builds += 1
        errs = compiler_errors(so + se)
        if count_misuse:  # only compile() and the final grade, which every condition runs alike
            self.misuse |= {e["message"] for e in errs
                            if (e["file"].endswith("solution.cu") or "mathdx" in e["file"].lower())
                            and MATHDX.search(e["message"] + " " + e["file"])}
        for e in errs:  # absolute paths would name hidden/test.cu
            e["file"] = os.path.basename(e["file"])
        return rc == 0, errs, (so + se)[-1500:].replace(self.task_dir, "<task>").replace(self.work, "<work>")

    def write_solution(self, code="", **_):
        # a markdown fence around the file is a chat-format slip, not a MathDx error: strip it in every condition
        code = str(code)
        m = FENCE.match(code)
        open(self.sol, "w").write(m[1] if m else code)
        return {"ok": True, "bytes": len(m[1] if m else code), "stripped_markdown_fence": bool(m)}

    def compile(self, **_):
        if not os.path.exists(self.sol):
            return {"ok": False, "error": "no solution.cu yet; call write_solution first"}
        # linked against a stub main so that a missing or mis-declared solve() fails here, not only at grading
        ok, errs, tail = self.build([self.sol, self.stub], os.path.join(self.work, "stub_exe"), count_misuse=True)
        return {"ok": ok, "errors": errs} if ok or errs else {"ok": False, "log_tail": tail}

    def search_docs(self, query="", **_):
        q, hits = str(query).lower(), []
        if not q:
            return {"error": "empty query"}
        rank = lambda p: ("doc/", "example/", "include/").index(p.split("/")[0] + "/")
        for p in sorted(self.corpus, key=rank):
            for i, line in enumerate(self.corpus[p]):
                if q in line.lower():
                    hits.append(f"{p}:{i + 1}: {line.strip()[:160]}")
        return {"n_hits": len(hits), "hits": hits[:30]}

    def read_doc(self, path="", start=1, count=120, **_):
        if path not in self.corpus:
            return {"error": f"unknown path {path!r}",
                    "close_matches": difflib.get_close_matches(str(path), list(self.corpus), 5, 0.4)}
        lines = self.corpus[path]
        start, count = max(1, int(start)), min(max(1, int(count)), 200)
        return {"path": path, "n_lines": len(lines),
                "text": "\n".join(f"{start + i}: {l}" for i, l in enumerate(lines[start - 1:start - 1 + count]))}

    def public_exe(self):
        if not os.path.exists(self.sol):
            return None, {"stage": "build", "error": "no solution.cu yet; call write_solution first"}
        exe = os.path.join(self.work, "public")
        ok, errs, tail = self.build([os.path.join(self.task_dir, "hidden", "test.cu"), self.sol], exe,
                                    ("-DEVAL_PUBLIC",))
        if not ok:
            mine = [e for e in errs if e["file"] == "solution.cu"]
            return None, {"stage": "build", "errors": mine or errs, **({} if errs else {"log_tail": tail})}
        return exe, None

    def run_public_test(self, **_):
        exe, err = self.public_exe()
        return err or validate.run_test(exe, 120)

    def sass_check(self, **_):
        if not os.path.exists(self.sol):
            return {"error": "no solution.cu yet; call write_solution first"}
        obj = self.sol + ".o"
        ok, errs, _ = self.build([self.sol], obj, link=False)
        return {"kernels": sass_counts(obj)} if ok else {"stage": "build", "errors": errs}

    def diagnose(self, **_):
        exe, err = self.public_exe()
        if err:
            return {**err, "docs": local_docs(err.get("errors", []))}
        res = validate.run_test(exe, 120)
        if res.get("status") != "pass":
            return {"stage": "accuracy", "result": res}
        if self.meta.get("requires_mma"):
            ks = sass_counts(exe)
            if not any(k["mma"] for k in ks):
                return {"stage": "tensor_cores", "kernels": ks, "docs": NO_MMA_DOCS}
        return {"stage": "ok", "result": res}

    def library_check(self):
        src = open(self.sol, errors="replace").read()
        need = {"cufftdx": ["cufftdx"], "cublasdx": ["cublasdx"], "both": ["cufftdx", "cublasdx"]}[self.meta["library"]]
        missing = [l for l in need if not re.search(rf'#\s*include\s*[<"]([\w/]*/)?{l}\.hpp[>"]', src)]
        host = sorted(set(HOST_LIB.findall(src)))
        return {"library_used": not missing and not host, "library_missing": missing, "host_library_calls": host}

    def grade(self):
        if not os.path.exists(self.sol):
            return {"compiles": False, "correct": False, "solved": False, "hidden": None}
        exe = os.path.join(self.work, "hidden_exe")
        ok, errs, tail = self.build([os.path.join(self.task_dir, "hidden", "test.cu"), self.sol], exe,
                                    count_misuse=True)
        g = {"compiles": ok, **self.library_check()}
        if not ok:
            g["hidden"] = {"status": "build_error", "errors": errs[:5], **({} if errs else {"log_tail": tail})}
        else:
            g["hidden"] = validate.run_test(exe, 300)
        g["correct"] = ok and g["hidden"].get("status") == "pass"
        if ok and self.meta.get("requires_mma"):
            g["hmma_lines"] = validate.hmma_lines(exe)
            g["correct"] &= g["hmma_lines"] > 0
        g["solved"] = g["correct"] and g["library_used"]
        return g


def agent(client, model, cond, task_dir, meta, corpus, tmp):
    run = Run(task_dir, meta, corpus, tmp)
    names = CONDITIONS[cond]
    spec, h = (open(os.path.join(task_dir, f)).read() for f in ("spec.md", "task.h"))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"spec.md:\n```\n{spec}\n```\n\ntask.h:\n```cpp\n{h}\n```"}]
    t0, tok_in, tok_out, end, calls, last = time.time(), 0, 0, "turns", {}, None
    for turn in range(1, MAX_TURNS + 1):
        try:
            r = client.chat(model=model, messages=msgs, tools=[TOOLS[n] for n in names], options=OPTS, think=False)
        except Exception as e:  # server OOM, restart or timeout: grade what was written, keep the sweep going
            end = f"error: {type(e).__name__}: {str(e)[:200]}"
            break
        last = r.prompt_eval_count
        tok_in, tok_out = tok_in + (r.prompt_eval_count or 0), tok_out + (r.eval_count or 0)
        msgs.append(r.message.model_dump(exclude_none=True))
        tcs = r.message.tool_calls or []
        if not tcs:
            msgs.append({"role": "user", "content": NUDGE})
        # a submit among several calls ends the run after the others, so a write next to it is not lost
        for tc in sorted(tcs, key=lambda tc: tc.function.name == "submit"):
            name, args = tc.function.name, tc.function.arguments or {}
            calls[name] = calls.get(name, 0) + 1
            if name == "submit":
                end = "submit"
                continue
            if name not in names:
                out = {"error": f"unknown tool {name!r}; available: {names}"}
            else:
                try:
                    out = getattr(run, name)(**args)
                except Exception as e:
                    out = {"error": f"{type(e).__name__}: {e}"}
            msgs.append({"role": "tool", "tool_name": name, "content": clip(out)})
        if end == "submit":
            break
        if (r.prompt_eval_count or 0) >= CTX_STOP:
            end = "context"
            break
    rec = {"task": meta["id"], "split": meta["split"], "library": meta["library"], "condition": cond,
           "model": model, "end": end, "iterations": turn, "wall_s": round(time.time() - t0, 1),
           "tokens_in": tok_in, "tokens_out": tok_out, "last_prompt_tokens": last, "tool_calls": calls,
           "builds": run.builds}
    rec.update(run.grade())
    rec["api_misuse_compile"] = len(run.misuse)
    rec["api_misuse_messages"] = sorted(run.misuse)[:20]
    rec["no_mma"] = bool(meta.get("requires_mma") and rec["compiles"] and not rec.get("hmma_lines"))
    rec["api_misuse"] = rec["api_misuse_compile"] + rec["no_mma"]
    return rec, msgs


def out_path(d, name):
    stem = os.path.join(d, f"{datetime.date.today().isoformat()}_{name}")
    path, k = f"{stem}.jsonl", 2
    while os.path.exists(path):
        path, k = f"{stem}_{k}.jsonl", k + 1
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--conditions", nargs="*", default=["A", "B", "C"], choices=list(CONDITIONS))
    ap.add_argument("--model", default="qwen3:8b")
    a = ap.parse_args()
    ids = a.tasks or sorted(os.path.basename(d) for d in os.listdir(os.path.join(EVAL, "tasks")) if d.startswith("d"))
    gpu = validate.provenance()["gpu"].strip().lower().replace(" ", "_")
    d = os.path.join(ROOT, "results", gpu)
    os.makedirs(d, exist_ok=True)
    runs_path = out_path(d, "eval_runs")
    tr_path = runs_path.replace("_eval_runs", "_eval_transcripts")
    client = ollama.Client(timeout=900)  # one turn is at most NUM_PREDICT tokens at ~40 tok/s
    digest = next((m.digest for m in client.list().models if m.model == a.model), None)
    prov = {**validate.provenance(), "model_digest": digest, "ollama": subprocess.run(
        ["ollama", "--version"], capture_output=True, text=True).stdout.strip(),
        "budget": {"max_turns": MAX_TURNS, "num_ctx": NUM_CTX, "ctx_stop": CTX_STOP, "num_predict": NUM_PREDICT,
                   "max_tool_chars": MAX_TOOL_CHARS, "temperature": 0, "seed": 0, "think": False},
        "argv": sys.argv}
    corpus = load_corpus(os.environ["MATHDX_ROOT"])
    print(f"corpus: {len(corpus)} files; writing {runs_path}", flush=True)
    with tempfile.TemporaryDirectory() as tmp, open(runs_path, "w") as fr, open(tr_path, "w") as ft:
        for t in ids:
            task_dir = os.path.join(EVAL, "tasks", t)
            meta = json.load(open(os.path.join(task_dir, "meta.json")))
            ref = validate.validate(task_dir, tmp)
            ref_ms = ref.get("hidden", {}).get("time_ms") if ref["valid"] else None
            print(f"{t}: reference valid={ref['valid']} time_ms={ref_ms}", flush=True)
            for c in a.conditions:
                rec, msgs = agent(client, a.model, c, task_dir, meta, corpus, tmp)
                ms = (rec.get("hidden") or {}).get("time_ms")
                rec.update({"ref_ms": ref_ms, "time_ms": ms if rec["correct"] else None,
                            "perf_vs_ref": round(ref_ms / ms, 3) if rec["correct"] and ref_ms and ms else None})
                fr.write(json.dumps({**rec, **prov}) + "\n")
                fr.flush()
                ft.write(json.dumps({"task": t, "condition": c, "messages": msgs}, default=str) + "\n")
                ft.flush()
                print(f"  {c}: end={rec['end']:7s} iters={rec['iterations']:2d} compiles={rec['compiles']!s:5s} "
                      f"correct={rec['correct']!s:5s} solved={rec['solved']!s:5s} misuse={rec['api_misuse']} perf={rec['perf_vs_ref']} "
                      f"tok={rec['tokens_in']}+{rec['tokens_out']} wall={rec['wall_s']}s tools={rec['tool_calls']}",
                      flush=True)
    print(runs_path)


if __name__ == "__main__":
    main()
