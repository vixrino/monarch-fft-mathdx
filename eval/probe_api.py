#!/usr/bin/env python3
# Check, before writing the harness, that the key and plan in use can run an agent loop:
# one tool call, the tool result sent back, a final answer, and the token counts the eval will log.
#   MISTRAL_API_KEY=... python3 eval/probe_api.py [model]
import json
import os
import sys
import time

from mistralai.client import Mistral

model = sys.argv[1] if len(sys.argv) > 1 else "mistral-medium-latest"
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
tools = [{"type": "function", "function": {
    "name": "compile", "description": "Compile solution.cu for sm_75 and return the compiler log.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
msgs = [{"role": "user", "content": "Call the compile tool on solution.cu, then report what the log says in one line."}]


def call(**kw):
    t = time.time()
    r = client.chat.complete(model=model, messages=msgs, tools=tools, temperature=0, random_seed=0, max_tokens=200, **kw)
    u = r.usage
    print(f"{r.model}  {time.time() - t:.1f}s  prompt={u.prompt_tokens} completion={u.completion_tokens}")
    return r.choices[0].message


m = call(tool_choice="any")
tc = m.tool_calls[0]
print("tool call:", tc.function.name, tc.function.arguments)
msgs += [{"role": "assistant", "content": m.content or "", "tool_calls": [
             {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]},
         {"role": "tool", "name": tc.function.name, "tool_call_id": tc.id,
          "content": json.dumps({"ok": False, "log": "solution.cu(12): error: identifier \"FFT\" is undefined"})}]
time.sleep(1.1)  # free plan: about 1 request/s
print("final:", call(tool_choice="none").content)
