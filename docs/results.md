# Results: Monarch FFT on cuBLASDx vs cuFFTDx, Tesla T4

Every number below comes from the three sweep files of one Colab session (2026-11-18, commit a8f89ec; the records say `git_dirty: true`, so
the working tree had uncommitted changes when they ran):

- `results/tesla_t4/2026-11-18_sweep_3.jsonl` (repeat 1, used to select configurations);
- `results/tesla_t4/2026-11-18_sweep_4.jsonl` and `_sweep_5.jsonl` (repeats 2 and 3, used for the ratios).

Throughput and accuracy tables are printed by
`python3 scripts/final.py results/tesla_t4/2026-11-18_sweep_{3,4,5}.jsonl --figures docs/figures/tesla_t4`; the
per-configuration Monarch, latency and SASS tables by `python3 scripts/plot.py --tables
results/tesla_t4/2026-11-18_sweep_3.jsonl` (default cuFFTDx configuration only). Statements tagged
*(interpretation)* are my reading of the numbers, not measurements. Older sweeps are kept in `results/tesla_t4/`
and are not used here.

## Question

For batched 1D complex FFTs computed inside one CUDA block, when does a four-step / Monarch factorization
N = n1·n2, run as two small cuBLASDx GEMMs on tensor cores, beat cuFFTDx's radix FFT, and what does it cost in
accuracy?

## Method

- **Hardware and software.** Tesla T4 (sm_75, 40 SMs) on Google Colab, driver 580.82.07, CUDA 13.0, MathDx 26.10.0
  (cuFFTDx 1.8.0, cuBLASDx 0.8.0). Every line of the sweep files records these and the git commit.
- **Transform.** Forward C2C, X[k] = Σ x[n]·exp(-2πi·nk/N), no normalization, out of place. N from 256 up to the
  largest size each implementation fits in one block.
- **cuFFTDx, two baselines** (`cuda/cufftdx/block_fft.cu`). fp16 uses cuFFTDx's implicit batching (two FFTs per
  `__half2`).
  - *Default:* the `Block` FFT with the library's suggested elements per thread (EPT) and FFTs per block (FPB),
    following NVIDIA's examples.
  - *Tuned:* every EPT the sm_75 database lists for that N (2-32, fewer at large N) times FPB 1-32, minus the
    combinations the library's static_asserts reject. Per sweep, 94 fp16 and 98 fp32 grid configurations ran
    over all N; 5 and 7 more were skipped because they need more than 64 KB of shared memory.
- **Monarch** (`cuda/monarch/monarch_fft.cu`, derivation in `docs/derivation.md`):
  - GEMM 1 multiplies the reshaped input by F_n1. Twiddles are applied in fp32 on the accumulator in registers.
    GEMM 2 multiplies by F_n2. The output is transposed through shared memory.
  - fp16 means fp16 operands with fp32 accumulation, using cuBLASDx's suggested shared-memory layouts (the path
    that emits HMMA on sm_75, see `docs/cublasdx_sm75_layouts.md`). fp32 runs on FFMA: sm_75 has no fp32
    tensor-core instruction.
  - 128 threads per block, max(1, 4096/N) FFTs per block, persistent grid. Its tuning knob is the factorization
    n1 × n2.
- **Selection and repeats** (`docs/decisions.md`, "Selection and repeats"). The same sweep ran three times in one
  session.
  - Repeat 1 selects, per precision and N: the fastest accuracy-passing cuFFTDx configuration (grid or default),
    and the fastest Monarch factorization.
  - Repeats 2 and 3 are measured with that selection fixed. Ratios are the mean of the two repeats, with
    [min, max] over them.
  - A ratio whose range contains 1.0 is a tie.
- **Accuracy.**
  - Inputs ~ U(-1, 1) complex from a counter-based hash. Relative L2 error over 64 FFTs per seed, median over 5
    seeds.
  - Two FP64 references (cuFFT Z2Z):
    - *total:* on the exact input, so it includes rounding the input to fp16/fp32;
    - *compute only:* on the input after rounding to the I/O type, so it isolates the transform's own error.
  - Seeds are fixed, so accuracy is identical across repeats; it is reported from repeat 1.
- **Timing.**
  - CUDA events around each launch: 5 warm-up launches, then 50 timed launches; median and IQR.
  - Throughput at a batch of 16384 FFTs. Latency is one block's group of FFTs on one block, launch overhead
    included.
  - Effective GB/s = 2 · batch · N · sizeof(complex T) / time, i.e. one read and one write.
- **Tensor-core evidence.** `scripts/sass_check.sh` counts MMA, FFMA and HFMA2 opcodes per kernel in
  `cuobjdump -sass`. The counts are static, and they cover the default cuFFTDx binary, not the grid binaries.

## Throughput

![throughput, configuration fixed on repeat 1](figures/tesla_t4/throughput_tuned.png)

Batch 16384. GB/s columns are the mean of repeats 2-3. Configurations were chosen on repeat 1.

| prec | N | cuFFTDx default | cuFFTDx tuned (config) | Monarch (n1xn2) | Monarch / tuned [range] | tuned / default [range] | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| fp16 | 256 | 161.5 | 235.3 (EPT 2, FPB 4) | 196.0 (16x16) | 0.83 [0.83, 0.84] | 1.46 [1.45, 1.46] | cuFFTDx faster |
| fp16 | 512 | 186.7 | 238.8 (EPT 2, FPB 2) | 191.2 (16x32) | 0.80 [0.79, 0.82] | 1.28 [1.27, 1.29] | cuFFTDx faster |
| fp16 | 1024 | 182.7 | 232.9 (EPT 2, FPB 2) | 183.2 (32x32) | 0.79 [0.78, 0.79] | 1.28 [1.24, 1.31] | cuFFTDx faster |
| fp16 | 2048 | 220.8 | 237.4 (EPT 4, FPB 4) | 97.1 (64x32) | 0.41 [0.40, 0.42] | 1.08 [1.07, 1.08] | cuFFTDx faster |
| fp16 | 4096 | 231.1 | 228.7 (EPT 16, FPB 4) | 94.7 (64x64) | 0.41 [0.41, 0.42] | 0.99 [0.95, 1.03] | cuFFTDx faster |
| fp16 | 8192 | 219.3 | 219.3 (EPT 16, FPB 2) | - | - | 1.00 [1.00, 1.00] | - |
| fp16 | 16384 | 196.4 | 196.4 (EPT 16, FPB 2) | - | - | 1.00 [1.00, 1.00] | - |
| fp32 | 256 | 210.9 | 235.6 (EPT 2, FPB 2) | 85.8 (16x16) | 0.36 [0.36, 0.37] | 1.12 [1.12, 1.12] | cuFFTDx faster |
| fp32 | 512 | 231.1 | 231.7 (EPT 2, FPB 2) | 69.3 (32x16) | 0.30 [0.30, 0.30] | 1.00 [1.00, 1.01] | cuFFTDx faster |
| fp32 | 1024 | 231.9 | 240.5 (EPT 4, FPB 4) | 58.2 (32x32) | 0.24 [0.24, 0.25] | 1.04 [1.03, 1.04] | cuFFTDx faster |
| fp32 | 2048 | 233.8 | 232.8 (EPT 4, FPB 1) | - | - | 1.00 [0.98, 1.01] | - |
| fp32 | 4096 | 238.5 | 228.1 (EPT 8, FPB 2) | 31.8 (64x64) | 0.14 [0.14, 0.14] | 0.96 [0.91, 1.00] | cuFFTDx faster |
| fp32 | 8192 | 234.0 | 219.8 (EPT 16, FPB 1) | - | - | 0.94 [0.91, 0.96] | - |
| fp32 | 16384 | 205.5 | 205.5 (EPT 16, FPB 1) | - | - | 1.00 [1.00, 1.00] | - |

Where "tuned" equals the default configuration, the ratio is exactly 1.00. Where the tuned/default range sits
below 1 (fp32 8192, and partly fp16 4096, fp32 2048 and fp32 4096), the configuration that won repeat 1 did not keep its lead
in repeats 2-3. I report it as selected, per the protocol, rather than re-selecting.

Every Monarch configuration (repeat 1, batch 16384):

| prec | N | n1xn2 | FFTs/block | smem B | blocks/SM | GB/s | rel L2 | max abs |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| fp16 | 256 | 8x32 | 16 | 20736 | 2 | 132.0 | 3.81e-04 | 1.87e-02 |
| fp16 | 256 | 16x16 | 16 | 17408 | 3 | 195.7 | 3.73e-04 | 2.09e-02 |
| fp16 | 512 | 16x32 | 8 | 21504 | 3 | 191.6 | 3.74e-04 | 2.96e-02 |
| fp16 | 512 | 32x16 | 8 | 21504 | 3 | 185.2 | 3.73e-04 | 2.76e-02 |
| fp16 | 1024 | 32x32 | 4 | 20480 | 3 | 185.7 | 3.79e-04 | 4.11e-02 |
| fp16 | 2048 | 32x64 | 2 | 36864 | 1 | 102.8 | 3.79e-04 | 6.18e-02 |
| fp16 | 2048 | 64x32 | 2 | 36864 | 1 | 103.0 | 3.80e-04 | 5.85e-02 |
| fp16 | 4096 | 64x64 | 1 | 32768 | 2 | 102.3 | 3.81e-04 | 9.20e-02 |
| fp32 | 256 | 8x32 | 16 | 41472 | 1 | 76.6 | 1.55e-07 | 1.19e-05 |
| fp32 | 256 | 16x16 | 16 | 34816 | 1 | 87.6 | 1.39e-07 | 9.18e-06 |
| fp32 | 512 | 16x32 | 8 | 43008 | 1 | 72.0 | 1.71e-07 | 1.93e-05 |
| fp32 | 512 | 32x16 | 8 | 43008 | 1 | 72.9 | 1.71e-07 | 1.46e-05 |
| fp32 | 1024 | 32x32 | 4 | 40960 | 1 | 60.9 | 1.98e-07 | 2.71e-05 |
| fp32 | 4096 | 64x64 | 1 | 65536 | 1 | 33.1 | 2.81e-07 | 8.52e-05 |
| fp32 | 2048 | 32x64 | 2 | 73728 | - | skipped: shared memory | | |
| fp32 | 2048 | 64x32 | 2 | 73728 | - | skipped: shared memory | | |

## Accuracy

![total and compute-only error](figures/tesla_t4/error_decomposition.png)

Repeat 1, median over 5 seeds. "fastest" is the tuned configuration from the throughput table; "most accurate" is
the cuFFTDx configuration with the lowest total error anywhere in the grid.

| prec | N | cuFFTDx default | cuFFTDx fastest | cuFFTDx most accurate (config) | Monarch total | Monarch compute only | most accurate / Monarch, total | most accurate / Monarch, compute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 256 | 1.64e-03 | 7.97e-04 | 7.97e-04 (EPT 2, FPB 2) | 3.73e-04 | 3.24e-04 | 2.14 | 2.39 |
| fp16 | 512 | 1.22e-03 | 8.60e-04 | 8.60e-04 (EPT 2, FPB 2) | 3.74e-04 | 3.26e-04 | 2.30 | 2.58 |
| fp16 | 1024 | 1.97e-03 | 9.13e-04 | 9.13e-04 (EPT 2, FPB 2) | 3.79e-04 | 3.31e-04 | 2.41 | 2.70 |
| fp16 | 2048 | 2.12e-03 | 1.08e-03 | 9.64e-04 (EPT 2, FPB 2) | 3.80e-04 | 3.32e-04 | 2.54 | 2.85 |
| fp16 | 4096 | 2.30e-03 | 2.30e-03 | 1.14e-03 (EPT 4, FPB 2) | 3.81e-04 | 3.33e-04 | 2.99 | 3.38 |
| fp16 | 8192 | 2.44e-03 | 2.44e-03 | 1.59e-03 (EPT 8, FPB 2) | - | - | - | - |
| fp16 | 16384 | 2.51e-03 | 2.51e-03 | 2.51e-03 (EPT 16, FPB 2) | - | - | - | - |
| fp32 | 256 | 1.38e-07 | 1.04e-07 | 1.04e-07 (EPT 2, FPB 1) | 1.39e-07 | 1.37e-07 | 0.75 | 0.74 |
| fp32 | 512 | 1.71e-07 | 1.10e-07 | 1.10e-07 (EPT 2, FPB 1) | 1.71e-07 | 1.70e-07 | 0.64 | 0.64 |
| fp32 | 1024 | 1.85e-07 | 1.43e-07 | 1.17e-07 (EPT 2, FPB 1) | 1.98e-07 | 1.96e-07 | 0.59 | 0.58 |
| fp32 | 2048 | 1.79e-07 | 1.58e-07 | 1.23e-07 (EPT 2, FPB 1) | - | - | - | - |
| fp32 | 4096 | 1.87e-07 | 2.07e-07 | 1.59e-07 (EPT 4, FPB 1) | 2.81e-07 | 2.80e-07 | 0.57 | 0.56 |
| fp32 | 8192 | 2.43e-07 | 2.43e-07 | 2.24e-07 (EPT 8, FPB 1) | - | - | - | - |
| fp32 | 16384 | 2.21e-07 | 2.21e-07 | 2.21e-07 (EPT 16, FPB 1) | - | - | - | - |

For cuFFTDx the total and compute-only errors differ by 0-3 % at every N and precision (the full pairs are in the
`final.py` output). For Monarch fp16 the total error is 14-15 % above the compute-only error.

Within cuFFTDx, the error falls monotonically as EPT goes down, at every fp16 N in the grid. At fp16 N = 256 it
goes from 3.3e-3 (EPT 32) to 1.6e-3 (EPT 16, the default) to 8.0e-4 (EPT 2).

## Latency (one block)

Repeat 1, default cuFFTDx configuration.

| prec | impl | N | n1xn2 | FFTs | median us [IQR] |
|---|---|---:|---|---:|---:|
| fp16 | cufftdx | 256 |  | 8 | 6.5 [6.2, 6.9] |
| fp16 | monarch | 256 | 16x16 | 16 | 10.2 [9.8, 10.2] |
| fp16 | cufftdx | 512 |  | 4 | 5.8 [5.1, 6.1] |
| fp16 | monarch | 512 | 16x32 | 8 | 10.5 [10.3, 11.3] |
| fp16 | cufftdx | 1024 |  | 2 | 6.1 [6.1, 6.3] |
| fp16 | monarch | 1024 | 32x32 | 4 | 11.2 [10.7, 11.8] |
| fp16 | cufftdx | 2048 |  | 2 | 6.8 [6.3, 7.5] |
| fp16 | monarch | 2048 | 64x32 | 2 | 13.5 [12.8, 14.2] |
| fp16 | cufftdx | 4096 |  | 2 | 8.5 [8.2, 9.0] |
| fp16 | monarch | 4096 | 64x64 | 1 | 14.0 [13.3, 14.3] |
| fp32 | cufftdx | 256 |  | 4 | 8.2 [8.2, 8.6] |
| fp32 | monarch | 256 | 16x16 | 16 | 23.0 [22.7, 23.8] |
| fp32 | cufftdx | 1024 |  | 1 | 8.6 [8.2, 8.9] |
| fp32 | monarch | 1024 | 32x32 | 4 | 35.8 [35.2, 36.4] |
| fp32 | cufftdx | 4096 |  | 1 | 10.2 [9.9, 10.2] |
| fp32 | monarch | 4096 | 64x64 | 1 | 65.8 [65.5, 66.5] |

The two implementations put different numbers of FFTs in a block, so each row is one launch of one block, not
equal work. The full latency table, every factorization included, is in the `plot.py --tables` output.

## Tensor-core evidence (SASS)

Repeat 1, default binaries.

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
| monarch | fp16 | 256 | 16x16 | 128 | 64 | 0 | 2264 |
| monarch | fp16 | 512 | 16x32 | 192 | 64 | 0 | 2264 |
| monarch | fp16 | 512 | 32x16 | 192 | 64 | 0 | 2200 |
| monarch | fp16 | 1024 | 32x32 | 256 | 64 | 0 | 2104 |
| monarch | fp16 | 2048 | 32x64 | 384 | 64 | 0 | 2440 |
| monarch | fp16 | 2048 | 64x32 | 384 | 64 | 0 | 2480 |
| monarch | fp16 | 4096 | 64x64 | 512 | 64 | 0 | 2656 |
| monarch | fp32 | 256 | 8x32 | 0 | 5184 | 0 | 6320 |
| monarch | fp32 | 256 | 16x16 | 0 | 4160 | 0 | 5224 |
| monarch | fp32 | 512 | 16x32 | 0 | 6208 | 0 | 7336 |
| monarch | fp32 | 512 | 32x16 | 0 | 6208 | 0 | 7328 |
| monarch | fp32 | 1024 | 32x32 | 0 | 8256 | 0 | 9376 |
| monarch | fp32 | 2048 | 32x64 | 0 | 12352 | 0 | 13976 |
| monarch | fp32 | 2048 | 64x32 | 0 | 12352 | 0 | 13960 |
| monarch | fp32 | 4096 | 64x64 | 0 | 16448 | 0 | 18304 |

Every fp16 Monarch kernel has MMA > 0; fp32 Monarch and the cuFFTDx kernels have none. In the 8×32 configuration,
GEMM 1 (N = 8 columns) stays on FMA: the suggested path selects the 16×8×8 HMMA atom only when the GEMM divides a
32×16×8 warp tile (`docs/cublasdx_sm75_layouts.md`).

## Nsight Compute profile (fp16, earlier session)

This section is not from the final sweeps. Files: `results/tesla_t4/2026-11-17_ncu_*_2.csv` (commit 38e465c, the
same Monarch and default cuFFTDx kernels as the final sweeps). Monarch 64×64 before the shared-F change is from
`2026-11-17_ncu_monarch_fp16_4096_64x64.csv` (commit ea51822). Each is `scripts/profile.sh` with `ncu --set full`,
one launch at batch 16384. The tuned cuFFTDx configurations were not profiled.

| metric | cuFFTDx N=256 | Monarch 16×16 | cuFFTDx N=4096 | Monarch 64×64, separate F (ea51822) | Monarch 64×64, shared F (38e465c) |
|---|---:|---:|---:|---:|---:|
| shared memory per block (KB) | 4.1 | 17.4 | 16.4 | 49.2 | 32.8 |
| occupancy limit: shared mem / registers (blocks) | 16 / 14 | 3 / 4 | 4 / 3 | 1 / 4 | 2 / 4 |
| warps active per SM (of 32) | 24.6 | 11.2 | 23.4 | 4.0 | 7.6 |
| SM cycles elapsed (avg, at ~585 MHz) | 94k | 159k | 1325k | 6517k | 5371k |
| issue active (% of peak) | 21 | 44 | 33 | 19 | 23 |
| instructions executed | 2.9M | 10.8M | 70.1M | 199.4M | 199.4M |
| tensor pipe active (%) | 0.4 | 18.8 | 0.9 | 26.4 | 32.9 |
| fp16 pipe (% of peak) | 24.3 | 4.3 | 42.3 | 1.6 | 2.0 |
| ALU pipe (% of peak) | 9.1 | 47.2 | 11.0 | 16.7 | 20.3 |
| memory throughput (% of peak) | 82 | 45 | 85 | 38 | 47 |
| DRAM read + write (MB) | 42 | 39 | 616 | 632 | 662 |
| smem load wavefronts / bank conflicts | 0.28M / 0.15M | 0.74M / 0.06M | 4.42M / 0.23M | 44.61M / 25.73M | 44.61M / 25.74M |
| smem store wavefronts / bank conflicts | 0.54M / 0.41M | 1.33M / 0.93M | 12.99M / 8.80M | 34.19M / 27.90M | 34.12M / 27.80M |

During profiling, ncu locks the SM clock to base (about 585 MHz), while the timed runs use boost clocks. Across
kernels, I read only ratios and utilizations from this table.

## Findings

Measured (the three sweep files above):

1. **The fp16 speed advantage at small N goes away against a tuned cuFFTDx.** With the selection fixed on repeat
   1, Monarch is slower than tuned cuFFTDx at every N and precision, and no range contains 1.0.
   - fp16: 0.83× (N = 256), 0.80× (512), 0.79× (1024), 0.41× (2048 and 4096).
   - Against the default cuFFTDx configuration, the same Monarch runs are 1.21× faster at N = 256 (196.0 vs
     161.5 GB/s) and roughly equal at 512 and 1024. The earlier "Monarch faster at N ≤ 512" result depended on
     the default configuration.
   - Tuning helps cuFFTDx fp16 most at small N: 1.46× at N = 256, 1.28× at 512 and 1024, with EPT 2. At
     N ≥ 4096 the gain is small: at fp16 4096 a grid config won repeat 1 (233.4 vs 229.1 GB/s) but over repeats 2-3
     tuned/default is 0.99 [0.95, 1.03], a tie.
2. **The fp16 accuracy advantage holds against every cuFFTDx configuration.**
   - Monarch's relative L2 error is 3.7-3.8e-4 at every N.
   - The most accurate cuFFTDx configuration anywhere in the grid has 2.1× (N = 256) to 3.0× (N = 4096) more
     error. Against the fastest configuration, the ratio is 2.1× to 6.0×; against the default, 3.3× to 6.0×.
   - With the input-rounding error removed (compute-only reference), the ratio grows to 2.4-3.4× against the most
     accurate configuration. So input quantization does not explain the gap; it narrows it.
3. **fp32: Monarch is slower and less accurate.** It runs on FFMA at 0.14-0.36× of tuned cuFFTDx's throughput.
   Its error is 1.39-2.81e-7, against 1.04-1.59e-7 for the most accurate cuFFTDx configuration.
4. **Repeat-to-repeat variation within the session is small for the ratios, not for absolute throughput.** Most
   ratio ranges over repeats 2-3 are 0.00-0.03 wide. The widest are tuned/default at fp16 1024 (0.07), fp32 8192
   (0.05), fp16 4096 (0.08) and fp32 4096 (0.09). Absolute throughput drifts down across the repeats at large N:
   default cuFFTDx fp16 8192 goes 241.9, 226.5, 212.2 GB/s (−12 %), and fp16 16384 and fp32 8192 / 16384 drop
   4-6 %. That is why I compare ratios within a repeat, and only within a session.
5. **Factorization matters for Monarch.** fp16 N = 256: 16×16 runs at 195.7 GB/s, 8×32 at 132.0. The 8×32 GEMM 1
   is not on HMMA, and its kernel has 3944 static instructions vs 2264.

Interpretation:

- *(interpretation)* Monarch's fp16 error is flat in N. All its products accumulate in fp32 on HMMA, and it rounds
  to fp16 at about three points: the input, A2 after the twiddle, and the output. cuFFTDx's fp16 butterflies run
  in HFMA2 (SASS table), so every stage rounds to fp16, which fits an error that grows with N. The compute-only
  numbers fit this reading: they remove the input rounding, and Monarch's error drops by about 13 % while
  cuFFTDx's barely moves.
- *(interpretation)* I don't know why cuFFTDx's fp16 error drops as EPT goes down. A lower EPT means fewer
  elements per thread and more exchanges through shared memory, not fewer stages. I haven't read the generated
  code closely enough to explain it, and I note it as an observation.
- *(interpretation, from the ncu table)* At N = 4096, Monarch's time goes into the block itself, not into DRAM:
  it executes 2.8× cuFFTDx's instructions and about 4.5× its shared-memory wavefronts, with heavy bank conflicts.
  The dense DFT costs N·(n1 + n2) multiply-adds per FFT, against O(N log N).
- *(interpretation)* On this GPU, the case for the tensor-core formulation is accuracy at fp16 I/O, not speed:
  about 20 % less throughput than tuned cuFFTDx at N ≤ 1024, for 2-3× less error than any cuFFTDx fp16
  configuration. Whether that trade holds when the FFT is fused with other work in the same kernel (the reason
  to do it at block level) is not measured here.

## Limitations

- **One GPU.** T4 only (Turing, sm_75). It has fp16 HMMA but no TF32/BF16 tensor cores. Nothing here says how
  Ampere or Hopper behave.
- **One session for the final numbers.** Three repeats in one Colab session bound the within-session noise. They
  say nothing about session-to-session drift; clocks are not locked.
- **Tuning is asymmetric in kind.** cuFFTDx gets its two exposed knobs (EPT × FPB) over the legal range. Monarch
  gets its factorization, plus structural choices I made by hand (persistent grid, shared F table, transposed
  GEMM 1) that have no cuFFTDx counterpart. I did not write a custom cuFFTDx kernel.
- **Profiling is partial.** Four fp16 kernels, from an earlier session, at the base clock that ncu imposes. The
  tuned cuFFTDx configurations, and all fp32 kernels, were not profiled. The grid binaries were not SASS-checked.
- **Limited sizes.** N ≤ 4096 for Monarch, power-of-two two-factor splits only. fp32 2048 is missing: with the
  8 FFTs per 4096 elements I use everywhere (2 per block at N = 2048) it needs 72 KB of shared memory. One FFT per
  block would need 56 KB and fit; I did not run that configuration.
- **One input distribution.** Uniform inputs only.

## Prior work

- **FlashFFTConv** (Fu et al., ICLR 2024) runs FFT convolutions as Monarch decompositions on tensor cores. The
  Monarch idea here is theirs.
- **tcFFT** (Li, Cheng, Lin, 2021, arXiv:2104.11471): half-precision FFT on tensor cores.
- **Split fp32 → 2× fp16 tensor-core FFTs** (SC18 poster and follow-ups) recover fp32-level accuracy from fp16
  tensor cores. Not attempted here.
- **NVIDIA's MathDx examples** (convolution_performance, gemm_fft, fused_gemm, ...): the cuFFTDx baseline follows
  them.
