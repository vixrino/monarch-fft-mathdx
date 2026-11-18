# Agent evaluation: does documentation plus an MCP loop help an agent write MathDx code?

Second research question of this repo (CLAUDE.md). It compares one model, Qwen3 8B (`qwen3:8b`, served by
ollama on the same T4; docs/decisions.md says why it is not an API model), under three conditions, with the
same prompt, the same token budget and the same iteration budget:

| condition | tools the agent gets |
|---|---|
| A, agent alone | `write_solution`, `compile` (nvcc errors only), `submit` |
| B, + official docs | A + `search_docs` / `read_doc` over the installed MathDx package: docs, headers, examples |
| C, + docs + MCP | B + `run_public_test`, `sass_check`, `diagnose` (the checks of `mcp/server.py`, pointed at the task) |

Status: the 5 dev tasks pass with the reference solutions on the T4
(`results/tesla_t4/2026-11-17_eval_reference.jsonl`). `eval/harness.py` is written and unit-tested off GPU,
run once on d1 (dev run 1). Held-out tasks h01–h08 are written; h09 and h10 were not written and are
dropped from all three conditions.

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
  every hidden size and seed (including the timed batch, 2^14), plus HMMA in the binary where required.
- **library_used:** solution.cu includes the required MathDx header(s) and calls no host cuFFT/cuBLAS.
  **solved** = correct and library_used; this is the headline count.
- **api_misuse:** count of compiler errors that name a MathDx symbol, plus the hidden-test check for the
  traps in meta.json (e.g. a "tensor cores required" task whose kernel has no MMA).
- **iterations:** model turns until the agent submits or the budget ends.
- **wall time** and **tokens:** input, output, cached.
- **perf vs reference:** median time of the hidden test's timing launch, divided by the reference
  solution's time in the same session.

## Reporting

Per-task tables only, plus counts ("C solved 7 of 10, B 5 of 10"). With 10 held-out tasks and one run
per cell, I make no significance claims. A difference of one or two tasks is within what a different
seed or prompt wording could flip, and the write-up says so.
