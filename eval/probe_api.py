#!/usr/bin/env python3
# Check, before writing the harness, that the local model can run an agent loop on the T4:
# one tool call, the tool result sent back, a final answer, and the token counts the eval will log.
#   python3 eval/probe_api.py [model]     (ollama serve must be running)
import sys
import time

import ollama

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"
opts = {"temperature": 0, "seed": 0, "num_ctx": 32768}
tools = [{"type": "function", "function": {
    "name": "compile", "description": "Compile solution.cu for sm_75 and return the compiler log.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
msgs = [{"role": "user", "content": "Call the compile tool on solution.cu, then report what the log says in one line."}]


def call():
    t = time.time()
    r = ollama.chat(model=model, messages=msgs, tools=tools, options=opts, think=False)
    print(f"{r.model}  {time.time() - t:.1f}s  prompt={r.prompt_eval_count} completion={r.eval_count}  "
          f"{r.eval_count / (r.eval_duration / 1e9):.0f} tok/s")
    return r.message


m = call()
assert m.tool_calls, f"no tool call, got: {m.content!r}"
tc = m.tool_calls[0]
print("tool call:", tc.function.name, tc.function.arguments)
msgs += [m, {"role": "tool", "tool_name": tc.function.name,
             "content": '{"ok": false, "log": "solution.cu(12): error: identifier \\"FFT\\" is undefined"}'}]
print("final:", call().content)
