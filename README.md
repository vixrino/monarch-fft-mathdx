# monarch-fft-mathdx: Monarch FFT on cuBLASDx vs cuFFTDx

For batched 1D complex FFTs computed inside one CUDA block, when does a
four-step / Monarch factorization N = n1·n2, run as two small GEMMs with
cuBLASDx on tensor cores, beat a radix FFT from cuFFTDx, and what does it cost
in accuracy?

I measured this on one GPU, a Tesla T4 (sm_75), with MathDx 26.10
(cuFFTDx 1.8.0, cuBLASDx 0.8.0) and CUDA 13.0.

## Findings

**On the T4, fp32 compute on fp16 storage dominates. The tensor-core Monarch FFT only beats pure-fp16 cuFFTDx.**

From three repeats of one sweep in one Colab session (`results/tesla_t4/2026-11-19_sweep{,_2,_3}.jsonl`), batch
16384. Configurations are chosen on repeat 1 and measured on repeats 2-3. The tables, the method and the
interpretations are in [docs/results.md](docs/results.md).

![throughput relative to tuned cuFFTDx](docs/figures/tesla_t4/throughput_tuned.png)
![total and compute-only error](docs/figures/tesla_t4/error_decomposition.png)

- **At equal storage (fp16 in, fp16 out), cuFFTDx with fp32 compute beats Monarch on both axes.** Monarch has
  1.35-1.37× its total error (3.7-3.8e-4 vs 2.8e-4) and runs at 0.80-0.85× of its throughput for N ≤ 1024, about
  0.5× at 2048-4096. No range over the repeats reaches 1.0.
- **That mixed variant sits on the fp16 storage floor.** Its error is 2.8e-4 total and 2.1e-4 compute-only at every
  N and every configuration, the same as an exact FFT whose input and output are rounded to fp16
  (`scripts/storage_floor.py`). It costs 0-5 % throughput against pure-fp16 cuFFTDx up to N = 1024, and 12-26 %
  above.
- **Against pure-fp16 cuFFTDx, Monarch is more accurate and slower.** 2.1× (N = 256) to 3.0× (N = 4096) less error
  than the most accurate fp16 configuration in a full EPT × FPB grid, at 0.80-0.83× of the fastest (N ≤ 1024) and
  0.41× above. Its advantage comes from accumulating in fp32, which the mixed variant also does, without Monarch's
  fp16 intermediate between the two GEMMs.
- **fp32:** the T4 has no fp32 tensor cores, so Monarch's GEMMs run on FFMA. It reaches 0.14-0.37× of tuned
  cuFFTDx's throughput, with 1.3-1.8× its best error.
- **Tensor cores, in SASS and ncu:** every fp16 Monarch kernel contains HMMA. In the two fp16 Monarch kernels I
  profiled (earlier session), the tensor pipe is active 19-33 % at runtime. The fp32 Monarch kernels and the
  default cuFFTDx kernels (fp16, mixed, fp32) contain none; the grid (tuned) cuFFTDx binaries were not
  SASS-checked.
- **Why Monarch slows down at large N (Nsight Compute, fp16 64×64, earlier session):** it is not memory-bound
  (47 % of peak memory throughput vs 85 % for cuFFTDx). It executes 2.8× the instructions and about 4.5× the
  shared-memory wavefronts, with heavy bank conflicts.

- **Agent eval: floor effect.** Qwen3 8B solved 0 of 8 held-out MathDx tasks alone, with the docs, and with docs
  plus the MCP compile/test/SASS loop. It invents headers and host-style APIs that cuFFTDx and cuBLASDx do not
  have. The one correct answer called host cuFFT and was rejected by the library check. Per-task table:
  [eval/README.md](eval/README.md).

Observed behavior on sm_75 that I did not expect: with cuBLASDx's plain shared-memory layouts, a complex fp16
GEMM with fp32 accumulation gets no HMMA (the sm_75 tuning records are all `universal_fma`). Only the
`suggest_layout_smem_*` / `suggest_accumulator` path reaches tensor cores, and only when the shape divides the
32×16×8 warp tile. For 8192 GEMMs of 64³ in one launch, the plain kernel takes 13.84 ms and the suggested one
2.33 ms (`results/tesla_t4/2026-11-19_mma_probe.jsonl`).
[docs/cublasdx_sm75_layouts.md](docs/cublasdx_sm75_layouts.md) has the repro, the instruction counts and the
header lines involved.

## Layout

```
reference/     NumPy FP64 reference and Monarch reference, pytest in tests/
cuda/common/   timing, FP64 cuFFT reference, error metrics, JSON-lines provenance
cuda/cufftdx/  baseline block FFT (follows NVIDIA's cuFFTDx examples)
cuda/monarch/  Monarch FFT with cuBLASDx, plus the HMMA probe
scripts/       env_check.sh, sass_check.sh, sweep.py, plot.py, final.py
gpu/           runner notebook for Colab / Kaggle
results/       raw measurements, never edited
docs/          derivation, decisions log, results, agent log
```

## How to run

You need an NVIDIA GPU. I used a Colab T4.

1. Open `gpu/runner.ipynb` in Colab with a GPU runtime and run all cells. It
   clones the repo, downloads MathDx, builds, runs `ctest`, the sweep and the SASS
   check, and downloads the sweep file.
2. Or by hand:

```
cmake -S . -B build -DCUDA_ARCH=75 -DMATHDX_ROOT=/path/to/nvidia/mathdx/26.10
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --build build --target cufftdx_grid -- -k   # cuFFTDx EPT x FPB grid, one binary per N
python3 scripts/sweep.py --grid --repeat 3        # three files results/<gpu>/<date>_sweep_<k>.jsonl
python3 scripts/final.py <sweep 1> <sweep 2> <sweep 3> --figures docs/figures/<gpu>
```

The Python reference and its tests run without a GPU:
`pip install -r requirements.txt && python3 -m pytest -q`.

## Limitations

- **One GPU and one session.** T4 only, so nothing here covers Ampere or Hopper, where tensor-core and fp32
  throughput differ. The final numbers are three repeats in one Colab session with unlocked clocks. Absolute
  throughput drifts down across the repeats at large N, most for the mixed variant (−22 % at N = 4096); ratios
  are taken within a repeat.
- **Tuning differs in kind.** cuFFTDx gets the full grid of its two exposed knobs. Monarch gets its factorization,
  plus structural choices I made by hand (persistent grid, shared F table, transposed GEMM 1).
- **Partial profiling.** Four fp16 kernels from an earlier session were profiled; the tuned and mixed cuFFTDx
  configurations were not.
- **Limited sizes.** Monarch covers N ≤ 4096 with power-of-two two-factor splits. fp32 N = 2048 is missing: at
  2 FFTs per block it needs 72 KB of shared memory; 1 FFT per block would fit, and I did not run it.
- **Agent eval is one small model, one run per cell.** 8 held-out tasks (h09 and h10 were not written), Qwen3 8B
  at temperature 0, which solved none in any condition. Condition C also gets a longer prompt (more tool
  descriptions). The dev task d3 does not enforce fp16 inside the kernel.
- **Not tried:** a Monarch variant that keeps the intermediate between the GEMMs in fp32, the split fp32 → 2× fp16
  scheme, and fusion with other work in the same kernel, which is the reason to have a block-level FFT.

## Next steps (not done)

- Split Monarch's fp16 intermediate into hi + lo fp16 parts (one extra GEMM) to remove its extra rounding and reach the storage floor.
- Rerun on Ampere or newer: TF32/BF16 tensor cores, and the sm80 cuBLASDx database lists HMMA atoms for these shapes.
- A fused case (FFT → pointwise → inverse FFT in one kernel), which is where block-level Dx libraries matter.

## Prior work

- **The Monarch decomposition on tensor cores** is from FlashFFTConv
  (Fu et al., ICLR 2024).
- **tcFFT** (Li, Cheng, Lin, 2021, arXiv:2104.11471) found its error at the
  level of half-precision cuFFT and attributed it to storing intermediates in
  half precision between merging stages (section 5.2). Monarch's two stages
  with fp32 accumulation and one fp16 intermediate are consistent with that.
- **Split fp32 → 2× fp16:** the ORNL/UTK work (Cheng, Sorna, D'Azevedo, SC18
  poster; Sorna et al., with Wong and Tomov) recovers accuracy on fp16 tensor
  cores by splitting each fp32 value into two fp16 values.
- **The baselines** follow NVIDIA's MathDx examples, including
  `08_mixed_precision/mixed_precision_fft_1d` for the fp32-compute variant.

What this repo adds is narrow:
- a block-level Monarch FFT built from MathDx pieces;
- a head-to-head comparison against cuFFTDx rather than cuFFT, including
  cuFFTDx's fp32-compute / fp16-storage mode, with the error measured against
  FP64 on the same inputs and split into input and compute parts;
- a record of which cuBLASDx configurations reach HMMA on sm_75.

## License

Apache-2.0, see [LICENSE](LICENSE). The cuFFTDx baseline follows NVIDIA's MathDx examples, also Apache-2.0.
MathDx itself is not redistributed here: the runner downloads it from NVIDIA.

## How this was built

I implemented it with Claude Code as the main coding tool, working phase by
phase from a written spec (CLAUDE.md). I defined the research question and the
scope, ran every experiment on a GPU, and reviewed the results. Design
decisions and failed attempts are logged in docs/decisions.md.

What the agent struggled with in practice was the MathDx API: which instruction
cuBLASDx actually selects. Twice it assumed the complex fp16 GEMMs would use
tensor cores, and twice they did not. Both times the correctness tests passed, and only
the SASS check caught it. docs/agent_log.md lists these mistakes. They motivated
the second part: an MCP server (mcp/) with whitelisted compile, test, SASS and
diagnose tools, and an evaluation (eval/) of whether docs and that loop help a
local model write MathDx code. With Qwen3 8B the answer is a floor effect, not
a yes or no.
