# monarch-fft-mathdx: Monarch FFT on cuBLASDx vs cuFFTDx

For batched 1D complex FFTs inside one CUDA block, when does a Monarch factorization N = n1·n2, run as two
cuBLASDx GEMMs on tensor cores, beat cuFFTDx, and what does it cost in accuracy? Measured on a Tesla T4 (sm_75)
with MathDx 26.10 (cuFFTDx 1.8.0, cuBLASDx 0.8.0) and CUDA 13.0.

## Findings

**On the T4, fp32 compute on fp16 storage dominates. The tensor-core Monarch FFT only beats pure-fp16 cuFFTDx, and only on accuracy.**

Three repeats in one session (`results/tesla_t4/2026-11-19_sweep{,_2,_3}.jsonl`), batch 16384, configurations
chosen on repeat 1 and measured on repeats 2-3. Tables and method: [docs/results.md](docs/results.md).

![throughput relative to tuned cuFFTDx](docs/figures/tesla_t4/throughput_tuned.png)
![total and compute-only error](docs/figures/tesla_t4/error_decomposition.png)

- **Same fp16 storage, cuFFTDx with fp32 compute wins on both axes.** Monarch has 1.35-1.37× its error and runs at
  0.80-0.85× of its throughput for N ≤ 1024, about 0.5× above.
- **That mixed variant sits on the fp16 storage floor:** 2.8e-4 total error at every N, the same as an exact FFT
  with fp16-rounded input and output (`scripts/storage_floor.py`).
- **Against pure-fp16 cuFFTDx, Monarch has 2.1-3.0× less error than its most accurate configuration** and runs at
  0.80-0.83× of its fastest (N ≤ 1024), 0.41× above. My reading: the gain comes from fp32 accumulation, which the
  mixed variant also has, not from the tensor cores.
- **fp32:** no fp32 tensor cores on the T4; Monarch reaches 0.14-0.37× of cuFFTDx's throughput.
- **Evidence:** every fp16 Monarch kernel contains HMMA (SASS). At N = 4096 it executes 2.8× cuFFTDx's
  instructions with heavy shared-memory bank conflicts (Nsight Compute, earlier session).
- **Agent eval:** Qwen3 8B solved 0 of 8 held-out MathDx tasks with or without docs and the MCP loop, a floor
  effect ([eval/README.md](eval/README.md)).

**cuBLASDx on sm_75:** with the plain shared-memory layouts, a complex fp16 GEMM with fp32 accumulator compiles to
HMUL2 + FADD, no HMMA: 5.9× slower and 770× the error of the suggested-layout kernel
(`results/tesla_t4/2026-11-19_mma_probe_3.jsonl`). Repro and header lines:
[docs/cublasdx_sm75_layouts.md](docs/cublasdx_sm75_layouts.md).

## Layout

```
cuda/cufftdx/  cuFFTDx baselines: fp16, fp32, fp16 storage with fp32 compute
cuda/monarch/  Monarch FFT with cuBLASDx, and the layout probe
cuda/common/   timing, FP64 cuFFT reference, error metrics, provenance
reference/     NumPy references (tests in tests/)
scripts/       sweep, analysis, figures, SASS checks
mcp/, eval/    MCP server and agent evaluation
results/       raw measurements, never edited
docs/          derivation, decisions, results
```

## How to run

Open `gpu/runner.ipynb` in Colab with a GPU runtime, or:

```
cmake -S . -B build -DCUDA_ARCH=75 -DMATHDX_ROOT=/path/to/nvidia/mathdx/26.10
cmake --build build -j && ctest --test-dir build --output-on-failure
cmake --build build --target cufftdx_grid -- -k
python3 scripts/sweep.py --grid --repeat 3
python3 scripts/final.py <sweep 1> <sweep 2> <sweep 3> --figures docs/figures/<gpu>
```

Without a GPU: `pip install -r requirements.txt && python3 -m pytest -q`.

## Limitations

- One T4, one session, unlocked clocks; large-N throughput drifts across repeats (ratios are taken per repeat).
- cuFFTDx is tuned over its two knobs (EPT × FPB); Monarch over its factorization plus hand-made structural choices.
- Monarch covers N ≤ 4096; fp32 N = 2048 was not run (72 KB of shared memory at 2 FFTs per block).
- Partial profiling; the agent eval is one small model, one run per cell, 8 tasks.

## Next steps (not done)

- Split Monarch's fp16 intermediate into hi + lo fp16 (one extra GEMM) to reach the storage floor.
- Rerun on Ampere or newer (TF32/BF16 tensor cores; the sm80 cuBLASDx database selects HMMA atoms).
- A fused case (FFT → pointwise → inverse FFT in one kernel), where block-level Dx libraries matter.

## Prior work

- **FlashFFTConv** (Fu et al., ICLR 2024): the Monarch decomposition on tensor cores.
- **tcFFT** (Li, Cheng, Lin, 2021, arXiv:2104.11471) found its error at the level of half-precision cuFFT and
  attributed it to fp16 intermediates between merging stages (section 5.2), consistent with these results.
- **Split fp32 → 2× fp16** (Cheng, Sorna, D'Azevedo, SC18 poster; Sorna et al. with Wong and Tomov, ORNL/UTK).
- **NVIDIA's MathDx examples**, which the baselines follow, including `08_mixed_precision`.

## License

Apache-2.0 ([LICENSE](LICENSE)). MathDx is not redistributed; the runner downloads it from NVIDIA.

## How this was built

Implemented with Claude Code as the main coding tool, phase by phase from a written spec (CLAUDE.md). I defined
the question and scope, ran every experiment on a GPU and reviewed the results; decisions and failed attempts are
in docs/decisions.md. The agent's main difficulty was which instruction cuBLASDx selects: twice it assumed the
complex fp16 GEMMs used tensor cores, tests passed, and only the SASS check caught it (docs/agent_log.md). That
motivated the MCP server and the agent evaluation.
