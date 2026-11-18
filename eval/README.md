# Agent evaluation: does documentation plus an MCP loop help an agent write MathDx code?

Second research question of this repo (CLAUDE.md). It compares one model, Qwen3 8B (`qwen3:8b`, served by
ollama on the same T4; docs/decisions.md says why it is not an API model), under three conditions, with the
same prompt, the same token budget and the same iteration budget:

| condition | tools the agent gets |
|---|---|
| A, agent alone | `write_solution`, `compile` (nvcc errors only), `submit` |
| B, + official docs | A + `search_docs` / `read_doc` over the installed MathDx package: docs, headers, examples |
| C, + docs + MCP | B + `run_public_test`, `sass_check`, `diagnose` (the checks of `mcp/server.py`, pointed at the task) |

Status: done. The 5 dev tasks pass with their reference solutions on the T4
(`results/tesla_t4/2026-11-18_eval_reference.jsonl`). The harness checks every held-out reference before the first
agent run and gives no agent run to a task whose reference fails. The held-out validation output of the final
session was not saved; the evidence that all 8 references passed is that every held-out record has agent runs
and a reference time (`ref_ms`). h09 and h10 were not written and are dropped from all
three conditions. The held-out run is below.

Run (Colab T4, after `ollama serve` and `ollama pull qwen3:8b`):
`python3 eval/harness.py --tasks d1_ifft_scaled --conditions A B C`. One record per run goes to
`results/<gpu>/<date>_eval_runs[_k].jsonl` and the full conversation to `..._eval_transcripts[_k].jsonl`.

## Why these three conditions

- **All three can compile.** Compiler errors are what any agent with a shell gets. The eval measures what
  documentation adds (B − A), and then what a validate/profile loop adds on top (C − B).
- **Only C can execute anything.** A and B never see a numerical result or the SASS. Giving them a test
  runner would make "agent alone" an agent with a test loop, which is most of what the MCP server provides.
  This choice is debatable; docs/decisions.md records it and the alternative.
- **Headers count as documentation.** B and C can read `include/` of the installed MathDx; A cannot. Reading
  the headers is what this repo's own rules require of an agent, so it belongs to the "docs" arm.

## Tasks

`tasks/<id>/` holds:
- **Visible to the agent:** `spec.md` (what to compute, layouts, sizes, tolerance, performance
  expectation) and `task.h` (the one function to implement).
- **Hidden:** `hidden/test.cu` (correctness on sizes and seeds the agent never sees, plus timing) and
  `hidden/reference.cu` (my solution, used to validate the test and for the reference timing).
- **Visible in condition C only:** the *result* of the public test, `hidden/test.cu` built with
  `-DEVAL_PUBLIC` (one small case), returned by `run_public_test`. The source is never shown.
- **Task metadata:** `meta.json` (split, library, which checks apply, e.g. "must use tensor cores").

None of the tasks is a copy of an official example or of this repo's kernels. Each changes the operation,
the data layout or the fusion so that a remembered example does not compile into a correct answer.
Several are adjacent to official examples on purpose, since that is where an agent's memory is most
likely to be confidently wrong.

| id | split | library | what |
|---|---|---|---|
| d1_ifft_scaled | dev | cuFFTDx | batched inverse C2C fp32, N = 1024, output scaled by 1/N |
| d2_gemm_batched | dev | cuBLASDx | batched real fp32 GEMM 32×32×32, A row-major, B and C column-major |
| d3_power_fp16 | dev | cuFFTDx | fp16 forward FFT N = 512, output fp32 power spectrum `|X[k]|²` |
| d4_gemm_relu_hmma | dev | cuBLASDx | fp16→fp32 64×64×64 GEMM + per-column bias + ReLU, must run on tensor cores |
| d5_fft_shift | dev | cuFFTDx | fp32 forward FFT N = 256 with output in fftshift order (zero frequency centered) |
| h01_fft_columns | held-out | cuFFTDx | FFT along the columns of a row-major 128 × 128 complex matrix, in place |
| h02_circ_xcorr | held-out | cuFFTDx | circular cross-correlation of signal pairs, N = 512, in one kernel |
| h03_thread_fft16 | held-out | cuFFTDx | thread-level FFT, N = 16, one FFT per thread, 2^20 FFTs |
| h04_ifft_fp16_natural | held-out | cuFFTDx | fp16 inverse FFT N = 1024 with natural-layout `__half2` I/O |
| h05_spectral_deriv | held-out | cuFFTDx | spectral derivative of a periodic signal: FFT, multiply by i·k, inverse, real part |
| h06_gemm_complex_hmma | held-out | cuBLASDx | complex fp16→fp32 GEMM 32×64×32, must run on tensor cores |
| h07_gemm_alpha_beta | held-out | cuBLASDx | fp32 C = α·Aᵀ·B + β·C, 64 × 32 × 48, in place on C |
| h08_gemm_chain | held-out | cuBLASDx | D = (A·B)·C for 32 × 32 matrices in one block, fp32 |
| h09_fft_then_gemm | held-out | both | not written (time); dropped from all conditions |
| h10_gemm_then_fft | held-out | both | not written (time); dropped from all conditions |

The dev tasks are for building and debugging the harness and may be looked at freely. The held-out tasks
are run once per condition at the end, with no prompt or harness change after the first held-out run.

## Metrics (per task, per condition)

- **compiles:** the final solution builds.
- **correct:** the hidden test passes, i.e. rel L2 against an FP64 reference under the spec's tolerance on
  every hidden size and seed (the largest checked case is the timed size: 2^14, 2^20 for h03, 128 matrices for
  h01), plus HMMA in the binary where required. The timed launch itself is not checked.
- **library_used:** solution.cu includes the required MathDx header(s) and calls no host cuFFT/cuBLAS.
  **solved** = correct and library_used; this is the headline count.
- **api_misuse:** number of distinct compiler messages that name a MathDx symbol or header, plus 1 when a
  "tensor cores required" task has no MMA. The other traps listed in meta.json are not checked automatically.
- **iterations:** model turns until the agent submits or the budget ends.
- **wall time** and **tokens:** input and output.
- **perf vs reference:** reference solution's median time divided by the agent solution's, both from the hidden
  test's timing launch in the same session (> 1 means the agent's code is faster).

## Reporting

Per-task tables only, plus counts per condition (written before the run; the result is below). With 8 held-out
tasks and one run per cell, I make no significance claims. A difference of one or two tasks is within what a different
seed or prompt wording could flip, and the write-up says so.

## Held-out results (Tesla T4, 2026-11-18, commit c5b6099)

One run per task and condition, `qwen3:8b` (digest `500a1f06...`), temperature 0, 16 turns, 4096 output tokens
per turn. Records: `results/tesla_t4/2026-11-18_eval_runs_{2,3,4,5}.jsonl`, conversations in the matching
`_eval_transcripts_` files. Each cell: final state; turns used, builds, api_misuse.

| task | library | A | B | C |
|---|---|---|---|---|
| h01_fft_columns | cuFFTDx | build error; 4, 0, 1 | build error; 16, 1, 1 | build error; 16, 1, 1 |
| h02_circ_xcorr | cuFFTDx | build error; 8, 3, 1 | build error; 16, 3, 1 | build error; 16, 6, 1 |
| h03_thread_fft16 | cuFFTDx | build error; 16, 8, 1 | build error; 16, 6, 1 | correct, host cuFFT, not solved; 7, 3, 1 |
| h04_ifft_fp16_natural | cuFFTDx | no solution; 2, 0, 0 | build error; 16, 6, 3 | build error; 16, 6, 4 |
| h05_spectral_deriv | cuFFTDx | no solution; 2, 0, 0 | build error; 12, 0, 1 | build error; 16, 6, 6 |
| h06_gemm_complex_hmma | cuBLASDx | no solution; 2, 0, 0 | build error; 16, 2, 21 | build error; 16, 3, 11 |
| h07_gemm_alpha_beta | cuBLASDx | no solution; 2, 0, 0 | build error; 16, 2, 7 | build error; 16, 2, 7 |
| h08_gemm_chain | cuBLASDx | build error; 2, 0, 1 | build error; 16, 3, 2 | build error; 16, 3, 6 |

Solved: A 0 of 8, B 0 of 8, C 0 of 8. This is a floor effect. It says this model, with this budget, does not
write MathDx code in any condition. It says nothing about whether docs or the MCP loop help a model that can.

What the transcripts do show:
- **The model does not know the library in its real form.** Several final builds fail on an invented header:
  `cuda_mathdx.h` (h01 in all three conditions, h02 A and B), `cudaFFTdx.h` (h03 B), `cublasDx.h` (h05 B). In h04 and h06-h08 under B
  and C, it includes the real `cufftdx.hpp` / `cublasdx.hpp` and then calls invented host-style functions
  (`cublasDxCreate`, `cufftDxExec`, `cublasDxGemmBatched`). cuFFTDx and cuBLASDx are header-only C++ templates
  with no handles. Doc use was thin: h05 made no doc calls under B or C, `read_doc` was used only in h06-h08,
  and many searches returned nothing.
- **The C loop was barely used.** Across the 8 C runs, `run_public_test` was called once (h03), and `sass_check`
  and `diagnose` never.
- **In A, four of eight runs wrote no solution (h04-h07).** The first turn used the whole 4096-token output budget
  and returned no tool call; after the harness's nudge ("Use a tool. Call submit when solution.cu is final.") the
  model called `submit`, which the harness accepts with no solution. The transcript does not keep what those 4096
  tokens were.
- **The one correct answer used the wrong library.** In h03, C, the model ran the public test, then submitted a
  host cuFFT solution that passes the hidden test. `library_used` catches it, so it is counted as not solved. This
  is the case the separate `solved` field exists for.

The value of the C loop therefore rests on qualitative evidence, not on this run. docs/agent_log.md records two
cases where Claude Code's code was numerically correct but ran without tensor cores, caught only by SASS. The MCP
smoke test shows `diagnose` reporting that failure in one call.

## Limitations of the eval

- One model, one run per cell, 8 tasks. No claim generalizes beyond Qwen3 8B with this prompt and budget.
- Condition C's prompt is longer, since it lists more tools.
- The HMMA check reads the whole binary, so a dead kernel containing HMMA would pass it (seen in d4 during
  development). d3 does not enforce that the FFT runs in fp16.
- h09 and h10 (FFT and GEMM fused) were not written.
- The hidden tests do not enforce every spec requirement: h04's fp16 computation (an fp32 FFT passes at 1e-2),
  block-level execution, and h03's one-FFT-per-thread structure (host cuFFT passes h03's test; only
  `library_used` rejects it).
- Grading trusts the solution process. The solution runs in the same process as the test, so a `solve()` that
  writes a passing result file and exits would pass; `-I eval/common` would also let a solution include a hidden
  reference by relative path. No run did either (checked in the transcripts), and h03 C's pass is genuine.
- In C, when a public build fails inside `test.cu`, the error context shows a few lines of the test source.
- `host_library_calls` also lists invented names such as `cublasDxCreate`, since the pattern matches any
  `cublas*` / `cufft*` call. `solved` does not depend on it.
