# DxBench — project context for Claude Code

## What this project is

A small, rigorous research prototype answering one question:

> For batched 1D complex FFTs computed inside a single CUDA block, when does a
> Monarch / four-step factorization executed as small GEMMs with **cuBLASDx**
> (tensor cores) beat a conventional radix FFT with **cuFFTDx** — and what does
> it cost in numerical accuracy?

Secondary question (later phase): do NVIDIA's agent-facing docs (cuBLASDx
"Agent BLAS Performance Guide", llms.txt) plus an MCP compile/validate/profile
loop measurably improve an LLM agent's ability to write correct MathDx code?

The author is Lucas, an MS student applying to NVIDIA's Math Libraries (Dx) team.
The repo must be something he can fully explain and defend line by line in a
technical interview. Small and correct beats large and unfinished.

## Prior work (always credit, never claim as novel)

- FlashFFTConv (Fu et al., ICLR 2024) — Monarch decomposition of FFT on tensor cores.
- tcFFT (Li, Cheng, Lin, 2021, arXiv:2104.11471) — half-precision FFT on tensor cores.
- Split FP32 → 2×FP16 tensor-core FFT (SC18 poster and follow-ups).
- NVIDIA MathDx official examples (convolution_performance, gemm_fft, fused_gemm...).
  Our contribution is: block-level / fusable implementation inside MathDx,
  head-to-head vs cuFFTDx (not just cuFFT), crossover + error analysis, and the
  agent evaluation.

## Environment (read carefully)

- Lucas edits code on a **Mac with no NVIDIA GPU**. CUDA cannot be compiled or run there.
- GPU execution happens on **Google Colab or Kaggle (Tesla T4, sm_75)** via the
  runner notebook `gpu/runner.ipynb`, and later on a VM (ISEP lab or cloud) for
  Nsight Compute profiling.
- At the start of every session run `nvidia-smi` (or check `scripts/env_check.sh`
  output). If no GPU is present: write code and the exact commands to run, then
  STOP and ask Lucas to run them on the GPU and paste back the output.
- Notes on T4 (sm_75): FP16 tensor cores (HMMA) yes; TF32 and BF16 tensor cores NO
  (Ampere+). Do not plan TF32/BF16 experiments on T4.

## Non-negotiable rules

1. **Never invent results.** No fabricated timings, errors, profiler metrics,
   SASS, or "expected" numbers presented as measured. Every number in
   `results/` or in docs must come from a run, stored with its provenance
   (GPU model, driver, CUDA version, MathDx version, git commit, command line).
2. **Never write MathDx API from memory.** Before using any cuFFTDx/cuBLASDx
   operator, check the installed headers and the docs for the installed version
   (cuBLASDx agent docs: `markdown/` folder + `llms.txt` of the HTML docs).
   If unsure, say so and point to where to check.
3. **Tensor-core usage must be proven, not assumed**: dump SASS
   (`cuobjdump -sass` / `nvdisasm`) and grep for HMMA/MMA instructions, or use
   Nsight Compute tensor-pipe metrics. Record the evidence.
4. **You implement the whole project**, including the Monarch kernel and the
   derivation. Because Lucas must be able to explain everything afterwards:
   - Keep code comments minimal (see Code style); put the reasoning behind each
     sync, layout, precision and template parameter in docs/decisions.md.
   - Keep `docs/decisions.md` up to date: one entry per non-obvious decision
     (what, alternatives considered, why, evidence). Include failed attempts and
     what they taught.
5. Keep changes small and reviewable. One phase at a time. At the end of each
   phase, report: implemented / actually executed / tests passed / unverified /
   next step.
6. No arbitrary shell execution exposed through the MCP server: controlled,
   whitelisted build/run commands only.

## Code style — write like a strong engineering student, not like an assistant

- Terse and idiomatic. Code should read like a focused project by someone who
  knows CUDA well, in the style of good open-source research code (e.g. CUTLASS
  examples, FlashAttention, MathDx examples).
- Comments explain _why_, never _what_. No comment on obvious lines, no
  section-banner comments, no "Step 1 / Step 2", no docstrings restating the
  function name. A tricky line (layout, sync, twiddle precision) gets one short
  comment; the full reasoning goes in docs/decisions.md.
- No emojis, no exclamation marks, no "robust/comprehensive/seamless" wording,
  no marketing tone in README or docs. Plain, direct, slightly informal technical
  prose, first person singular where natural ("I use N1 = 64 because...").
- No defensive over-engineering: no unused abstractions, no config systems for
  two parameters, no wrapper classes around one call, no exhaustive input
  validation in benchmark code. Prefer a few well-named functions.
- Consistent, short names (n1, n2, smem, tw, x_ref). Error handling via one
  small CUDA_CHECK macro.
- README: short — question, how to run, main plots, findings, limitations,
  references, and a short "How this was built" section. No feature lists with
  bold headers, no "Key Features" sections.
- "How this was built" must be accurate and plain, e.g.: "Implemented with
  Claude Code as the main coding tool, working phase by phase from a written
  spec (CLAUDE.md). I defined the research question and scope, ran every
  experiment on GPU, and reviewed the results. Design decisions and failed
  attempts are logged in docs/decisions.md." Add one line on what the agent
  struggled with in practice (e.g. MathDx API errors it made), since that
  motivates the MCP/eval part. Do not overstate Lucas's share of the code.
- Commits: small, one logical change each, short lowercase imperative messages
  ("fix twiddle indexing for n2 > 32", "add fp32 path").
- Keep a running log in docs/agent_log.md of MathDx mistakes you made during
  development (wrong operator, illegal shape, layout bug) and how they were
  caught: it is real data for the agent evaluation in phase 6.
- Personal study material (walkthrough.md, interview_prep.md) goes in `notes/`,
  which is git-ignored.

## Technical conventions

- C++17, CUDA, CMake. Release builds with `-O3 -DNDEBUG`, `-arch=sm_75` by default
  (configurable `CUDA_ARCH`).
- Forward DFT convention: X[k] = Σ x[n]·exp(-2πi·nk/N), no normalization;
  inverse normalized by 1/N. Same everywhere (NumPy, CUDA, docs).
- Accuracy metric: relative L2 error ‖y − y_ref‖₂ / ‖y_ref‖₂ against an FP64
  reference, plus max abs error. Report median over ≥ 5 random inputs
  (fixed seeds), inputs ~ U(-1,1) complex unless stated.
- Timing: CUDA events, warm-up runs, ≥ 20 timed iterations, report median and
  IQR; throughput in FFTs/s and effective GB/s. Batch large enough to fill the
  GPU (e.g. 2^14+ FFTs) and also report a small-batch latency point.
- Results stored as JSON lines in `results/<gpu>/<date>_<experiment>.jsonl`,
  never edited by hand. Plots generated only from these files.

## Repository layout

```
reference/       NumPy FP64 reference + Monarch reference (Python)
tests/           pytest for reference; CUDA correctness tests
cuda/common/     harness: allocation, timing, error computation, JSON output
cuda/cufftdx/    baseline: conventional block FFT with cuFFTDx
cuda/monarch/    Monarch FFT with cuBLASDx
gpu/runner.ipynb Colab/Kaggle notebook: clone, fetch MathDx, build, run, save results
scripts/         env_check.sh, sweep.py, plot.py, sass_check.sh
results/         raw measured data only
docs/            derivation.md, methodology.md, results.md, decisions.md
mcp/             (phase 5) MCP server
eval/            (phase 6) agent evaluation tasks + harness
```

## Scope

MVP (must ship): cuFFTDx baseline vs cuBLASDx Monarch, N ∈ {256, 512, …, 4096+
as far as one block allows}, FP16 (FP32 accumulate) and FP32, on T4; accuracy
vs FP64; crossover plot; SASS proof of tensor-core use; short results write-up.
Stretch (only after MVP): split-FP32→2×FP16 Monarch for accuracy recovery,
hybrid FFT/GEMM stages, Ampere+ GPU with Nsight Compute roofline, MCP server,
agent evaluation, JAX FFI.
