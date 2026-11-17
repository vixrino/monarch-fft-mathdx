# Results: Monarch FFT on cuBLASDx vs cuFFTDx, Tesla T4

Every number below comes from one sweep file, named at the top of each
table, produced by `scripts/sweep.py` and turned into tables and figures by
`scripts/plot.py --tables`. Statements tagged *(interpretation)* are my
reading of the numbers, not measurements.

Sweep file: `results/tesla_t4/2026-11-17_sweep.jsonl` (commit d481e92, one Colab session, tables printed by `python3 scripts/plot.py --tables`).

## Question

For batched 1D complex FFTs computed inside one CUDA block, when does a
four-step / Monarch factorization N = n1·n2, run as two small cuBLASDx GEMMs
on tensor cores, beat cuFFTDx's radix FFT, and what does it cost in accuracy?

## Method

- **Hardware and software.** Tesla T4 (sm_75, 40 SMs) on Google Colab.
  The sweep file records the driver, CUDA, cuFFT, MathDx (26.10.0: cuFFTDx 1.8.0,
  cuBLASDx 0.8.0) and git commit on every line.
- **Transform.** Forward C2C, X[k] = Σ x[n]·exp(-2πi·nk/N), no normalization,
  out of place. N from 256 up to the largest size each implementation fits in
  one block.
- **cuFFTDx baseline.** The `Block` FFT with the library's suggested
  elements per thread and FFTs per block, following NVIDIA's examples
  (`cuda/cufftdx/block_fft.cu`). fp16 uses cuFFTDx's implicit batching
  (two FFTs per `__half2`).
- **Monarch** (`cuda/monarch/monarch_fft.cu`, derivation in `docs/derivation.md`):
  - GEMM 1 multiplies the reshaped input by F_n1.
  - Twiddles are applied in fp32 on the accumulator in registers.
  - GEMM 2 multiplies by F_n2.
  - The output is transposed through shared memory.
  - fp16 means fp16 operands with fp32 accumulation, using cuBLASDx's suggested
    shared-memory layouts (the path that emits HMMA on sm_75, see
    `docs/decisions.md`). fp32 runs on FFMA: there is no fp32 tensor-core
    instruction on sm_75.
  - 128 threads per block, max(1, 4096/N) FFTs per block, persistent grid.
- **Accuracy.**
  - Inputs ~ U(-1, 1) complex from a counter-based hash.
  - Reference: cuFFT Z2Z (FP64) on the unrounded input, so the error includes
    rounding the input to fp16/fp32.
  - Metrics: relative L2 error over 64 FFTs per seed, median over 5 seeds, and max abs error.
- **Timing.**
  - CUDA events around each launch: 5 warm-up launches, 50 timed launches, median and IQR.
  - Two batch sizes:
    - ~2^14 FFTs (throughput);
    - one block's group of FFTs on one block (latency, launch overhead included).
  - Effective GB/s = 2 · batch · N · sizeof(complex T) / time, i.e. one read and one write.
- **Tensor-core evidence.** `scripts/sass_check.sh` counts MMA, FFMA and
  HFMA2 opcodes per kernel in `cuobjdump -sass`. The counts are static: an
  unrolled-or-not K loop changes how often each one executes.

## Throughput

![throughput vs N](figures/tesla_t4/throughput_vs_n.png)

Batch 16384 FFTs. Monarch row = fastest factorization at that N.

| prec | N | cuFFTDx GB/s [IQR] | Monarch n1xn2 | Monarch GB/s [IQR] | Monarch / cuFFTDx | cuFFTDx rel L2 | Monarch rel L2 |
|---|---:|---:|---|---:|---:|---:|---:|
| fp16 | 256 | 162.4 [161.5, 163.8] | 16x16 | 192.8 [190.7, 195.2] | 1.19 | 1.64e-03 | 3.73e-04 |
| fp16 | 512 | 186.1 [185.1, 187.6] | 16x32 | 192.5 [191.6, 193.4] | 1.03 | 1.22e-03 | 3.74e-04 |
| fp16 | 1024 | 183.0 [181.8, 183.4] | 32x32 | 165.6 [165.2, 167.4] | 0.90 | 1.97e-03 | 3.79e-04 |
| fp16 | 2048 | 220.2 [218.9, 222.1] | 64x32 | 104.4 [103.2, 108.8] | 0.47 | 2.12e-03 | 3.80e-04 |
| fp16 | 4096 | 228.8 [226.3, 229.7] | 64x64 | 90.3 [89.8, 90.8] | 0.39 | 2.30e-03 | 3.81e-04 |
| fp16 | 8192 | 242.1 [241.0, 242.7] | - | - | - | 2.44e-03 | - |
| fp16 | 16384 | 206.4 [205.2, 206.9] | - | - | - | 2.51e-03 | - |
| fp32 | 256 | 211.8 [211.2, 212.8] | 16x16 | 86.8 [86.6, 86.9] | 0.41 | 1.38e-07 | 1.39e-07 |
| fp32 | 512 | 231.4 [230.8, 231.9] | 32x16 | 72.3 [71.7, 72.9] | 0.31 | 1.71e-07 | 1.71e-07 |
| fp32 | 1024 | 231.6 [231.5, 231.9] | 32x32 | 61.8 [61.4, 61.8] | 0.27 | 1.85e-07 | 1.98e-07 |
| fp32 | 2048 | 230.6 [229.1, 231.2] | - | - | - | 1.79e-07 | - |
| fp32 | 4096 | 236.1 [235.9, 236.3] | 64x64 | 33.9 [33.7, 34.0] | 0.14 | 1.87e-07 | 2.81e-07 |
| fp32 | 8192 | 236.1 [235.9, 236.3] | - | - | - | 2.43e-07 | - |
| fp32 | 16384 | 214.9 [214.4, 215.3] | - | - | - | 2.21e-07 | - |

Every Monarch configuration (batch 16384):

| prec | N | n1xn2 | FFTs/block | smem B | blocks/SM | GB/s | rel L2 | max abs |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| fp16 | 256 | 8x32 | 16 | 20736 | 2 | 132.1 | 3.81e-04 | 1.87e-02 |
| fp16 | 256 | 16x16 | 16 | 18432 | 3 | 192.8 | 3.73e-04 | 2.09e-02 |
| fp16 | 512 | 16x32 | 8 | 21504 | 3 | 192.5 | 3.74e-04 | 2.96e-02 |
| fp16 | 512 | 32x16 | 8 | 21504 | 3 | 183.8 | 3.73e-04 | 2.76e-02 |
| fp16 | 1024 | 32x32 | 4 | 24576 | 2 | 165.6 | 3.79e-04 | 4.11e-02 |
| fp16 | 2048 | 32x64 | 2 | 36864 | 1 | 103.6 | 3.79e-04 | 6.18e-02 |
| fp16 | 2048 | 64x32 | 2 | 36864 | 1 | 104.4 | 3.80e-04 | 5.85e-02 |
| fp16 | 4096 | 64x64 | 1 | 49152 | 1 | 90.3 | 3.81e-04 | 9.20e-02 |
| fp32 | 256 | 8x32 | 16 | 41472 | 1 | 77.1 | 1.55e-07 | 1.19e-05 |
| fp32 | 256 | 16x16 | 16 | 34816 | 1 | 86.8 | 1.39e-07 | 9.18e-06 |
| fp32 | 512 | 16x32 | 8 | 43008 | 1 | 71.8 | 1.71e-07 | 1.93e-05 |
| fp32 | 512 | 32x16 | 8 | 43008 | 1 | 72.3 | 1.71e-07 | 1.46e-05 |
| fp32 | 1024 | 32x32 | 4 | 40960 | 1 | 61.8 | 1.98e-07 | 2.71e-05 |
| fp32 | 4096 | 64x64 | 1 | 65536 | 1 | 33.9 | 2.81e-07 | 8.52e-05 |
| fp32 | 2048 | 32x64 | 2 | 73728 | - | skipped: shared memory | | |
| fp32 | 2048 | 64x32 | 2 | 73728 | - | skipped: shared memory | | |

## Accuracy

![relative error vs N](figures/tesla_t4/error_vs_n.png)

Numbers: the rel L2 columns of the two tables above.

## Speed vs accuracy

![speed vs accuracy](figures/tesla_t4/speed_vs_accuracy.png)

## Latency (one block)

| prec | impl | N | n1xn2 | FFTs | median us [IQR] |
|---|---|---:|---|---:|---:|
| fp16 | cufftdx | 256 |  | 8 | 6.8 [6.4, 7.5] |
| fp16 | monarch | 256 | 8x32 | 16 | 14.9 [14.6, 15.6] |
| fp16 | monarch | 256 | 16x16 | 16 | 11.3 [10.8, 11.9] |
| fp16 | cufftdx | 512 |  | 4 | 6.1 [6.0, 6.2] |
| fp16 | monarch | 512 | 16x32 | 8 | 11.9 [11.2, 12.3] |
| fp16 | monarch | 512 | 32x16 | 8 | 13.5 [12.8, 14.1] |
| fp16 | cufftdx | 1024 |  | 2 | 6.7 [6.4, 7.4] |
| fp16 | monarch | 1024 | 32x32 | 4 | 14.3 [14.3, 14.6] |
| fp16 | cufftdx | 2048 |  | 2 | 8.1 [7.6, 8.5] |
| fp16 | monarch | 2048 | 32x64 | 2 | 15.2 [14.8, 16.0] |
| fp16 | monarch | 2048 | 64x32 | 2 | 15.1 [14.7, 15.7] |
| fp16 | cufftdx | 4096 |  | 2 | 8.7 [8.5, 9.4] |
| fp16 | monarch | 4096 | 64x64 | 1 | 16.7 [16.4, 17.6] |
| fp16 | cufftdx | 8192 |  | 2 | 14.8 [14.4, 15.4] |
| fp16 | cufftdx | 16384 |  | 2 | 25.8 [25.1, 26.2] |
| fp32 | cufftdx | 256 |  | 4 | 7.1 [6.7, 7.8] |
| fp32 | monarch | 256 | 8x32 | 16 | 31.3 [30.8, 32.1] |
| fp32 | monarch | 256 | 16x16 | 16 | 24.0 [23.2, 24.5] |
| fp32 | cufftdx | 512 |  | 2 | 7.3 [6.7, 7.9] |
| fp32 | monarch | 512 | 16x32 | 8 | 30.7 [30.4, 30.7] |
| fp32 | monarch | 512 | 32x16 | 8 | 30.8 [30.7, 31.2] |
| fp32 | cufftdx | 1024 |  | 1 | 8.2 [8.1, 8.3] |
| fp32 | monarch | 1024 | 32x32 | 4 | 36.9 [36.9, 37.2] |
| fp32 | cufftdx | 2048 |  | 1 | 9.2 [8.6, 10.0] |
| fp32 | cufftdx | 4096 |  | 1 | 9.5 [8.9, 10.0] |
| fp32 | monarch | 4096 | 64x64 | 1 | 66.0 [65.6, 66.8] |
| fp32 | cufftdx | 8192 |  | 1 | 15.4 [14.7, 15.9] |
| fp32 | cufftdx | 16384 |  | 1 | 24.6 [24.6, 25.0] |

The two implementations put different numbers of FFTs
in a block (cuFFTDx's suggestion vs 4096/N for Monarch), so this row compares
one launch of one block each, not equal work.

## Tensor-core evidence (SASS)

| impl | prec | N | n1xn2 | MMA | FFMA | HFMA2 | total |
|---|---|---:|---|---:|---:|---:|---:|
| cufftdx | fp16 | 256 |  | 0 | 0 | 84 | 720 |
| cufftdx | fp16 | 512 |  | 0 | 0 | 52 | 440 |
| cufftdx | fp16 | 1024 |  | 0 | 0 | 128 | 968 |
| cufftdx | fp16 | 2048 |  | 0 | 0 | 136 | 1016 |
| cufftdx | fp16 | 4096 |  | 0 | 0 | 148 | 1072 |
| cufftdx | fp16 | 8192 |  | 0 | 0 | 192 | 1296 |
| cufftdx | fp16 | 16384 |  | 0 | 0 | 192 | 1344 |
| cufftdx | fp32 | 256 |  | 0 | 96 | 0 | 592 |
| cufftdx | fp32 | 512 |  | 0 | 64 | 0 | 400 |
| cufftdx | fp32 | 1024 |  | 0 | 152 | 0 | 848 |
| cufftdx | fp32 | 2048 |  | 0 | 160 | 0 | 896 |
| cufftdx | fp32 | 4096 |  | 0 | 172 | 0 | 952 |
| cufftdx | fp32 | 8192 |  | 0 | 228 | 0 | 1184 |
| cufftdx | fp32 | 16384 |  | 0 | 228 | 0 | 1216 |
| monarch | fp16 | 256 | 8x32 | 128 | 64 | 0 | 3944 |
| monarch | fp16 | 256 | 16x16 | 128 | 64 | 0 | 2352 |
| monarch | fp16 | 512 | 16x32 | 192 | 64 | 0 | 2264 |
| monarch | fp16 | 512 | 32x16 | 192 | 64 | 0 | 2200 |
| monarch | fp16 | 1024 | 32x32 | 256 | 64 | 0 | 2192 |
| monarch | fp16 | 2048 | 32x64 | 384 | 64 | 0 | 2440 |
| monarch | fp16 | 2048 | 64x32 | 384 | 64 | 0 | 2480 |
| monarch | fp16 | 4096 | 64x64 | 512 | 64 | 0 | 2744 |
| monarch | fp32 | 256 | 8x32 | 0 | 5184 | 0 | 6320 |
| monarch | fp32 | 256 | 16x16 | 0 | 4160 | 0 | 5224 |
| monarch | fp32 | 512 | 16x32 | 0 | 6208 | 0 | 7336 |
| monarch | fp32 | 512 | 32x16 | 0 | 6208 | 0 | 7328 |
| monarch | fp32 | 1024 | 32x32 | 0 | 8256 | 0 | 9376 |
| monarch | fp32 | 2048 | 32x64 | 0 | 12352 | 0 | 13976 |
| monarch | fp32 | 2048 | 64x32 | 0 | 12352 | 0 | 13960 |
| monarch | fp32 | 4096 | 64x64 | 0 | 16448 | 0 | 18304 |

What has to hold, and does in this table:
- every fp16 Monarch kernel has MMA > 0;
- fp32 Monarch and all cuFFTDx kernels have MMA = 0.

The GEMM 1 of the 8×32 configuration (N = 8 columns) is expected to stay on
FMA: on sm_75 the suggested path only selects the 16×8×8 HMMA atom when the
GEMM divides a 32×16×8 warp tile (`docs/decisions.md`).

## Nsight Compute profile (fp16)

Files: `results/tesla_t4/2026-11-17_ncu_*.csv` (raw pages, commit ea51822).
They were produced by `scripts/profile.sh` with `ncu --set full`: one launch at batch 16384 per configuration.
The table is built from those CSVs.

| metric | cuFFTDx N=256 | Monarch 16×16 | cuFFTDx N=4096 | Monarch 64×64 |
|---|---:|---:|---:|---:|
| SM clock under ncu (MHz) | 581 | 585 | 579 | 585 |
| SM cycles elapsed (avg) | 95k | 160k | 1319k | 6517k |
| grid (blocks) | 2,048 | 120 | 8,192 | 40 |
| warps active per SM (of 32) | 25.1 | 11.2 | 23.6 | 4.0 |
| occupancy limit: shared mem / registers (blocks) | 16 / 14 | 3 / 4 | 4 / 3 | 1 / 4 |
| issue active (% of peak) | 21 | 44 | 33 | 19 |
| instructions executed | 2.9M | 10.8M | 70.1M | 199.4M |
| tensor pipe active (%) | 0.4 | 18.6 | 0.9 | 26.4 |
| fp16 pipe (% of peak) | 24.4 | 4.2 | 42.3 | 1.6 |
| ALU pipe (% of peak) | 9.1 | 46.9 | 11.1 | 16.7 |
| memory throughput (% of peak) | 83 | 45 | 85 | 38 |
| DRAM read + write (MB) | 42 | 39 | 616 | 632 |
| smem load wavefronts / bank conflicts | 0.28M / 0.15M | 0.75M / 0.06M | 4.42M / 0.23M | 44.61M / 25.73M |
| smem store wavefronts / bank conflicts | 0.54M / 0.41M | 1.33M / 0.94M | 13.00M / 8.79M | 34.19M / 27.90M |

Caveat: ncu locks the SM clock to base (about 585 MHz) during profiling,
while the timed runs use boost clocks. This slows the compute-bound Monarch
kernels more than the memory-bound cuFFTDx ones: under ncu, Monarch 16×16
takes more cycles than cuFFTDx N=256, although it is 1.2× faster in the timed
sweep. So I read only ratios and utilizations from this table, not durations
across kernels.

Measured:
- **Monarch 64×64 has one 128-thread block per SM.** That is 4 warps active out
  of 32, i.e. one warp per scheduler. The limit is shared memory: 48 KB per
  block, one block fits. Issue slots are 19 % active.
- **Shared-memory traffic in Monarch 64×64:**
  - 44.6M load wavefronts and 34.2M store wavefronts, against 4.4M and 13.0M
    for cuFFTDx N=4096.
  - Bank conflicts: 25.7M on loads and 27.9M on stores.
- **Instructions executed:** 199M for Monarch 64×64 vs 70M for cuFFTDx (2.8×).
- **DRAM traffic is the same** for both at N=4096 (616 vs 632 MB). cuFFTDx runs
  at 85 % of peak memory throughput, Monarch 64×64 at 38 %.
- **The tensor pipe is active 26 % (64×64) and 19 % (16×16)** for Monarch, and
  under 1 % for cuFFTDx. This confirms the SASS evidence with a runtime counter.
- **Monarch 16×16** runs 11 warps per SM (limit: 3 blocks by shared memory).
  Its ALU pipe is the busiest at 47 % of peak.

## Findings

Measured (sweep file above):

1. **fp16 throughput: Monarch is faster up to N = 512 and slower from N = 1024.**
   - N = 256: 192.8 vs 162.4 GB/s (1.19×).
   - N = 512: 192.5 vs 186.1 GB/s (1.03×). The IQRs do not overlap at either size.
   - N = 1024: 0.90×. N = 2048: 0.47×. N = 4096: 0.39×.
   - So the crossover lies between N = 512 and N = 1024 at batch 16384.
2. **fp16 accuracy: Monarch's relative L2 error is 3.7–3.8e-4 at every N.**
   cuFFTDx fp16 is 1.2e-3 to 2.5e-3 and grows with N. The ratio
   cuFFTDx / Monarch is 3.3× (N = 512) to 6.0× (N = 4096), on the same inputs
   and the same FP64 reference.
3. **fp32: Monarch is 0.14–0.41× cuFFTDx's throughput.** The errors are equal
   at N = 256 and 512 and higher for Monarch at 1024 (1.98e-7 vs 1.85e-7) and
   4096 (2.81e-7 vs 1.87e-7).
4. **Latency (one block):** Monarch takes 11–17 µs in fp16 and 24–66 µs in fp32;
   cuFFTDx takes 6–9 µs for N ≤ 4096. Different FFT counts per block, see above.
5. **Factorization matters.**
   - fp16 N = 256: 16×16 runs at 192.8 GB/s, 8×32 at 132.1. The 8×32 GEMM 1 is
     not on HMMA and its kernel has 3944 static instructions vs 2352.
   - N = 512: 16×32 is 5 % faster than 32×16.
   - N = 2048: the two orders are within the IQR.

Interpretation:

- *(interpretation)* The flat fp16 error suggests that rounding the input and
  the one intermediate (after the twiddle) to fp16 dominates. In the Monarch
  kernel every product is accumulated in fp32. A radix FFT with fp16
  butterflies rounds after each of its log2 N stages, which fits an error that
  grows with N. The Phase 1 CPU emulation of the same rounding points gave
  2.6–3.4e-4, close to the GPU numbers.
- *(interpretation, supported by the ncu table)* cuFFTDx is memory-bound at
  N = 4096 (85 % of peak memory throughput). Monarch 64×64 is not: it moves
  the same DRAM bytes at 38 %. At N = 4096 its time goes into the block itself,
  for three reasons:
  - with one warp per scheduler, there is nothing to hide shared-memory and
    tensor-core latency (19 % issue active);
  - it executes 2.8× the instructions, because the dense DFT costs
    N·(n1 + n2) complex multiply-adds per FFT against O(N log N);
  - its shared-memory traffic is about 4.5× cuFFTDx's, more than half of it
    bank-conflict replays.

  I have not separated these three effects. The shared-memory limit is the
  first thing to change (see next step in docs/decisions.md).
- *(interpretation)* fp32 Monarch is slow because on sm_75 its GEMMs run on
  FFMA: 4160–16448 static FFMA vs 64–228 in cuFFTDx. Static counts are not
  executed counts, so this is an indication, not a measurement of work.
- *(interpretation)* For small fp16 FFTs (N ≤ 512) on this GPU the tensor-core
  formulation gives both higher throughput and about 3–4× lower error than
  cuFFTDx's fp16 path. That is the regime where it is worth using. Whether it
  holds when the FFT is fused with other work in the same kernel (the reason
  to do it at block level) is not measured here.

## Limitations

- **One GPU.** T4 only (Turing, sm_75). It has fp16 HMMA but no TF32/BF16
  tensor cores, and cuBLASDx's sm_75 tuning database selects FMA for these
  shapes unless the suggested layouts are used. Nothing here says how Ampere or
  Hopper behave.
- **Profiling is partial.** Four fp16 kernels were profiled, at the base
  clock that ncu imposes. Nothing was profiled in fp32. The profile shows where
  time goes, but no fix has been tested yet.
- **Shared Colab instances.** Clocks are not locked, and runs in different
  sessions differ by a few percent (compare the earlier per-binary runs in
  `results/tesla_t4/`). Only numbers from the same sweep file are compared.
- **Limited sizes.**
  - N ≤ 4096 for Monarch; fp32 2048 is skipped because 72 KB of shared memory
    exceeds the T4's opt-in limit.
  - Only power-of-two sizes and two-factor splits.
- **One input distribution.** Uniform inputs only. Errors on structured or
  large-dynamic-range signals were not measured.

## Prior work

- **FlashFFTConv** (Fu et al., ICLR 2024) runs FFT convolutions as Monarch
  decompositions on tensor cores. The Monarch idea here is theirs.
- **tcFFT** (Li, Cheng, Lin, 2021, arXiv:2104.11471): half-precision FFT on
  tensor cores.
- **Split fp32 → 2× fp16 tensor-core FFTs** (SC18 poster and follow-ups) recover
  fp32-level accuracy from fp16 tensor cores. Not attempted here.
- **NVIDIA's MathDx examples** (convolution_performance, gemm_fft, fused_gemm,
  ...): the cuFFTDx baseline follows them.

What this project adds is narrow:
- a block-level, fusable Monarch FFT inside MathDx;
- compared head-to-head against cuFFTDx rather than cuFFT;
- with the error measured against FP64 on the same inputs.
