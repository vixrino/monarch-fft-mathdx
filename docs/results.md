# Results: Monarch FFT on cuBLASDx vs cuFFTDx, Tesla T4

Every number below comes from the three sweep files of one Colab session (2026-11-19, commit 83e3916,
`git_dirty: false`), unless a section says otherwise:

- `results/tesla_t4/2026-11-19_sweep.jsonl` (repeat 1, used to select configurations);
- `results/tesla_t4/2026-11-19_sweep_2.jsonl` and `_sweep_3.jsonl` (repeats 2 and 3, used for the ratios).

Throughput and accuracy tables come from
`python3 scripts/final.py results/tesla_t4/2026-11-19_sweep{,_2,_3}.jsonl --figures docs/figures/tesla_t4`. The
per-configuration Monarch, latency and SASS tables come from `python3 scripts/plot.py --tables
results/tesla_t4/2026-11-19_sweep.jsonl` (default cuFFTDx configuration). Statements tagged *(interpretation)* are
my reading of the numbers, not measurements. The 11-18 sweeps (without the mixed-precision baseline) give similar fp16
and fp32 ratios (Limitations has the largest difference) and are kept in `results/tesla_t4/`, but are not used
here.

## Question

For batched 1D complex FFTs computed inside one CUDA block, when does a four-step / Monarch factorization
N = n1·n2, run as two small cuBLASDx GEMMs on tensor cores, beat cuFFTDx's radix FFT, and what does it cost in
accuracy?

## Method

- **Hardware and software.** Tesla T4 (sm_75, 40 SMs) on Google Colab, driver 580.82.07, CUDA 13.0, MathDx 26.10.0
  (cuFFTDx 1.8.0, cuBLASDx 0.8.0). Every line of the sweep files records these and the git commit.
- **Transform.** Forward C2C, X[k] = Σ x[n]·exp(-2πi·nk/N), no normalization, out of place. N from 256 up to the
  largest size each implementation fits in one block.
- **cuFFTDx, three variants** (`cuda/cufftdx/block_fft.cu`), each as a default and a tuned configuration:
  - *fp16:* fp16 storage and fp16 arithmetic, with cuFFTDx's implicit batching (two FFTs per `__half2`).
  - *mixed:* fp16 storage (`__half2` in global memory, the same bytes as the fp16 variant and Monarch fp16), fp32
    arithmetic. Each element is widened on load and rounded once on store, as in the package's
    `08_mixed_precision/mixed_precision_fft_1d` example (`docs/decisions.md` has the comparison).
  - *fp32:* fp32 storage and arithmetic.
  - *Default* is the `Block` FFT with the library's suggested elements per thread (EPT) and FFTs per block (FPB).
    *Tuned* is the fastest of every EPT the sm_75 database lists for that N (2-32, fewer at large N) times FPB
    1-32, minus what the library's static_asserts reject. Per sweep, 94 fp16, 98 mixed and 98 fp32 grid
    configurations ran; 5, 7 and 7 more were skipped because they need more than 64 KB of shared memory.
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
  - Repeat 1 selects, per variant and N: the fastest accuracy-passing cuFFTDx configuration (grid or default), the
    most accurate one, and the fastest Monarch factorization.
  - Repeats 2 and 3 are measured with that selection fixed. Ratios are the mean of the two per-repeat ratios, with
    [min, max] over them.
  - A ratio whose range contains 1.0 is a tie.
- **Accuracy.**
  - Inputs ~ U(-1, 1) complex from a counter-based hash. Relative L2 error over 64 FFTs per seed, median over 5
    seeds.
  - Two FP64 references (cuFFT Z2Z):
    - *total:* on the exact input, so it includes rounding the input to the storage type;
    - *compute only:* on the input after rounding to the storage type, so it isolates the transform's own error,
      including the final rounding of the output.
  - Seeds are fixed, so accuracy is identical across repeats (and across sessions); it is reported from repeat 1.
- **Timing.**
  - CUDA events around each launch: 5 warm-up launches, then 50 timed launches; median and IQR.
  - Throughput at a batch of 16384 FFTs. Latency is one block's group of FFTs on one block, launch overhead
    included.
  - Effective GB/s = 2 · batch · N · sizeof(stored complex) / time, i.e. one read and one write. fp16 Monarch,
    cuFFTDx fp16 and cuFFTDx mixed move the same bytes.
- **Tensor-core evidence.** `scripts/sass_check.sh` counts MMA, FFMA and HFMA2 opcodes per kernel in
  `cuobjdump -sass`. The counts are static, and they cover the default binaries, not the grid binaries.

## Throughput

![throughput relative to tuned cuFFTDx](figures/tesla_t4/throughput_tuned.png)

fp16 storage, batch 16384. GB/s columns are the mean of repeats 2-3; configurations were chosen on repeat 1.

| N | cuFFTDx fp16 default | cuFFTDx fp16 tuned (config) | cuFFTDx mixed tuned (config) | Monarch (n1xn2) | Monarch / mixed [range] | Monarch / fp16 tuned [range] | mixed / fp16 tuned [range] |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 161.4 | 235.6 (EPT 2, FPB 4) | 231.0 (EPT 16, FPB 1) | 196.6 (16x16) | 0.85 [0.85, 0.85] | 0.83 [0.83, 0.84] | 0.98 [0.97, 0.99] |
| 512 | 187.0 | 238.6 (EPT 2, FPB 2) | 226.1 (EPT 4, FPB 2) | 191.6 (16x32) | 0.85 [0.83, 0.86] | 0.80 [0.80, 0.81] | 0.95 [0.94, 0.96] |
| 1024 | 198.3 | 222.4 (EPT 2, FPB 2) | 226.5 (EPT 4, FPB 1) | 180.6 (32x32) | 0.80 [0.74, 0.86] | 0.81 [0.78, 0.85] | 1.02 [0.99, 1.05] |
| 2048 | 221.1 | 236.6 (EPT 4, FPB 4) | 208.7 (EPT 8, FPB 2) | 97.3 (64x32) | 0.47 [0.43, 0.51] | 0.41 [0.40, 0.42] | 0.88 [0.79, 0.98] |
| 4096 | 230.6 | 234.1 (EPT 16, FPB 4) | 178.9 (EPT 16, FPB 1) | 95.0 (64x64) | 0.53 [0.52, 0.54] | 0.41 [0.41, 0.41] | 0.76 [0.75, 0.78] |
| 8192 | 223.8 | 223.8 (EPT 16, FPB 2) | 165.0 (EPT 16, FPB 1) | - | - | - | 0.74 [0.73, 0.74] |
| 16384 | 197.0 | 197.0 (EPT 16, FPB 2) | 145.8 (EPT 16, FPB 1) | - | - | - | 0.74 [0.73, 0.75] |

fp32 storage:

| N | cuFFTDx default | cuFFTDx tuned (config) | Monarch (n1xn2) | Monarch / tuned [range] | tuned / default [range] |
|---:|---:|---:|---:|---:|---:|
| 256 | 210.4 | 236.5 (EPT 2, FPB 4) | 87.1 (16x16) | 0.37 [0.36, 0.38] | 1.12 [1.12, 1.12] |
| 512 | 230.9 | 233.5 (EPT 2, FPB 2) | 69.4 (32x16) | 0.30 [0.29, 0.31] | 1.01 [0.97, 1.05] |
| 1024 | 232.2 | 240.2 (EPT 4, FPB 4) | 58.4 (32x32) | 0.24 [0.24, 0.25] | 1.03 [1.03, 1.04] |
| 2048 | 233.2 | 235.5 (EPT 4, FPB 1) | - | - | 1.01 [1.00, 1.02] |
| 4096 | 238.6 | 232.7 (EPT 8, FPB 2) | 31.9 (64x64) | 0.14 [0.14, 0.14] | 0.98 [0.94, 1.01] |
| 8192 | 238.5 | 231.6 (EPT 16, FPB 1) | - | - | 0.97 [0.91, 1.03] |
| 16384 | 205.2 | 205.2 (EPT 16, FPB 1) | - | - | 1.00 [1.00, 1.00] |

Where "tuned" equals the default configuration, the ratio is exactly 1.00. Where tuned/default sits below 1, the
configuration that won repeat 1 did not keep its lead in repeats 2-3; I report it as selected, per the protocol.
The fp16 tuned/default ratios are 1.46 [1.46, 1.46] (256), 1.28 [1.27, 1.28] (512), 1.13 [1.01, 1.25] (1024),
1.07 (2048), 1.02 [0.99, 1.04] (4096).

Every Monarch configuration (repeat 1, batch 16384):

| prec | N | n1xn2 | FFTs/block | smem B | blocks/SM | GB/s | rel L2 | max abs |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| fp16 | 256 | 8x32 | 16 | 20736 | 2 | 133.3 | 3.81e-04 | 1.87e-02 |
| fp16 | 256 | 16x16 | 16 | 17408 | 3 | 195.0 | 3.73e-04 | 2.09e-02 |
| fp16 | 512 | 16x32 | 8 | 21504 | 3 | 191.9 | 3.74e-04 | 2.96e-02 |
| fp16 | 512 | 32x16 | 8 | 21504 | 3 | 184.1 | 3.73e-04 | 2.76e-02 |
| fp16 | 1024 | 32x32 | 4 | 20480 | 3 | 181.4 | 3.79e-04 | 4.11e-02 |
| fp16 | 2048 | 32x64 | 2 | 36864 | 1 | 103.9 | 3.79e-04 | 6.18e-02 |
| fp16 | 2048 | 64x32 | 2 | 36864 | 1 | 104.1 | 3.80e-04 | 5.85e-02 |
| fp16 | 4096 | 64x64 | 1 | 32768 | 2 | 102.7 | 3.81e-04 | 9.20e-02 |
| fp32 | 256 | 8x32 | 16 | 41472 | 1 | 77.5 | 1.55e-07 | 1.19e-05 |
| fp32 | 256 | 16x16 | 16 | 34816 | 1 | 89.8 | 1.39e-07 | 9.18e-06 |
| fp32 | 512 | 16x32 | 8 | 43008 | 1 | 72.9 | 1.71e-07 | 1.93e-05 |
| fp32 | 512 | 32x16 | 8 | 43008 | 1 | 73.4 | 1.71e-07 | 1.46e-05 |
| fp32 | 1024 | 32x32 | 4 | 40960 | 1 | 61.8 | 1.98e-07 | 2.71e-05 |
| fp32 | 4096 | 64x64 | 1 | 65536 | 1 | 33.8 | 2.81e-07 | 8.52e-05 |
| fp32 | 2048 | 32x64 | 2 | 73728 | - | skipped: shared memory | | |
| fp32 | 2048 | 64x32 | 2 | 73728 | - | skipped: shared memory | | |

## Accuracy

![total and compute-only error](figures/tesla_t4/error_decomposition.png)

Repeat 1, median over 5 seeds. "fastest" is the tuned configuration from the throughput table; "most accurate" is
the configuration with the lowest total error anywhere in that variant's grid.

fp16 storage (mixed and Monarch: total / compute only):

| N | cuFFTDx fp16 default | cuFFTDx fp16 fastest | cuFFTDx fp16 most accurate (config) | cuFFTDx mixed most accurate (config) | Monarch | Monarch / mixed, total | Monarch / mixed, compute |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1.64e-03 | 7.97e-04 | 7.97e-04 (EPT 2, FPB 2) | 2.77e-04 / 2.07e-04 (EPT 2, FPB 1) | 3.73e-04 / 3.24e-04 | 1.35 | 1.57 |
| 512 | 1.22e-03 | 8.60e-04 | 8.60e-04 (EPT 2, FPB 2) | 2.78e-04 / 2.07e-04 (EPT 8, FPB 2) | 3.74e-04 / 3.26e-04 | 1.35 | 1.57 |
| 1024 | 1.97e-03 | 9.13e-04 | 9.13e-04 (EPT 2, FPB 2) | 2.78e-04 / 2.08e-04 (EPT 4, FPB 1) | 3.79e-04 / 3.31e-04 | 1.36 | 1.59 |
| 2048 | 2.12e-03 | 1.08e-03 | 9.64e-04 (EPT 2, FPB 2) | 2.77e-04 / 2.07e-04 (EPT 4, FPB 1) | 3.80e-04 / 3.32e-04 | 1.37 | 1.60 |
| 4096 | 2.30e-03 | 2.30e-03 | 1.14e-03 (EPT 4, FPB 2) | 2.78e-04 / 2.08e-04 (EPT 32, FPB 1) | 3.81e-04 / 3.33e-04 | 1.37 | 1.60 |
| 8192 | 2.44e-03 | 2.44e-03 | 1.59e-03 (EPT 8, FPB 2) | 2.77e-04 / 2.07e-04 (EPT 8, FPB 1) | - | - | - |
| 16384 | 2.51e-03 | 2.51e-03 | 2.51e-03 (EPT 16, FPB 2) | 2.78e-04 / 2.08e-04 (EPT 32, FPB 1) | - | - | - |

The mixed variant's error is the same for every configuration in its grid and every N: 2.77-2.78e-4 total,
2.07-2.08e-4 compute only (all 210 mixed records of repeat 1). That is the floor of fp16 storage itself: an exact
FFT whose output is rounded to fp16 has 2.07-2.08e-4, and 2.76-2.78e-4 with the input rounded too
(`python3 scripts/storage_floor.py`, NumPy, same metric). Against the most accurate pure-fp16 cuFFTDx
configuration, Monarch has 2.14× (N = 256) to 2.99× (N = 4096) less total error and 2.39-3.38× less compute-only
error, as in the earlier session.

fp32 storage:

| N | cuFFTDx default | cuFFTDx fastest | cuFFTDx most accurate (config) | Monarch | Monarch / most accurate |
|---:|---:|---:|---:|---:|---:|
| 256 | 1.38e-07 | 1.04e-07 | 1.04e-07 (EPT 2, FPB 1) | 1.39e-07 | 1.34 |
| 512 | 1.71e-07 | 1.10e-07 | 1.10e-07 (EPT 2, FPB 1) | 1.71e-07 | 1.55 |
| 1024 | 1.85e-07 | 1.43e-07 | 1.17e-07 (EPT 2, FPB 1) | 1.98e-07 | 1.69 |
| 2048 | 1.79e-07 | 1.58e-07 | 1.23e-07 (EPT 2, FPB 1) | - | - |
| 4096 | 1.87e-07 | 2.07e-07 | 1.59e-07 (EPT 4, FPB 1) | 2.81e-07 | 1.77 |
| 8192 | 2.43e-07 | 2.43e-07 | 2.24e-07 (EPT 8, FPB 1) | - | - |
| 16384 | 2.21e-07 | 2.21e-07 | 2.21e-07 (EPT 16, FPB 1) | - | - |

Within pure-fp16 cuFFTDx, the error falls monotonically as EPT goes down, at every N in the grid. At N = 256 it
goes from 1.6e-3 (EPT 16, the default) to 8.0e-4 (EPT 2).

## Latency (one block)

Repeat 1, default cuFFTDx configuration.

| prec | impl | N | n1xn2 | FFTs | median us [IQR] |
|---|---|---:|---|---:|---:|
| fp16 | cufftdx | 256 |  | 8 | 6.1 [6.1, 6.3] |
| fp16 | monarch | 256 | 16x16 | 16 | 11.8 [11.1, 12.2] |
| fp16 | cufftdx | 512 |  | 4 | 6.1 [5.6, 6.2] |
| fp16 | monarch | 512 | 16x32 | 8 | 12.1 [11.4, 12.3] |
| fp16 | cufftdx | 1024 |  | 2 | 6.5 [6.1, 7.0] |
| fp16 | monarch | 1024 | 32x32 | 4 | 12.2 [11.9, 12.3] |
| fp16 | cufftdx | 2048 |  | 2 | 6.8 [6.5, 7.8] |
| fp16 | monarch | 2048 | 64x32 | 2 | 14.4 [14.3, 14.8] |
| fp16 | cufftdx | 4096 |  | 2 | 8.6 [8.3, 9.3] |
| fp16 | monarch | 4096 | 64x64 | 1 | 14.1 [13.2, 14.3] |
| fp32 | cufftdx | 256 |  | 4 | 7.0 [6.4, 7.8] |
| fp32 | monarch | 256 | 16x16 | 16 | 22.9 [22.7, 23.7] |
| fp32 | cufftdx | 1024 |  | 1 | 8.0 [7.6, 8.2] |
| fp32 | monarch | 1024 | 32x32 | 4 | 37.0 [36.9, 37.9] |
| fp32 | cufftdx | 4096 |  | 1 | 9.1 [8.6, 9.7] |
| fp32 | monarch | 4096 | 64x64 | 1 | 66.1 [65.7, 66.8] |

The two implementations put different numbers of FFTs in a block, so each row is one launch of one block, not
equal work. The full latency table is in the `plot.py --tables` output. The mixed variant's latency records
are in the sweep files but not tabulated.

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
| cufftdx | fp16 (fp32 compute) | 256 |  | 0 | 96 | 0 | 688 |
| cufftdx | fp16 (fp32 compute) | 512 |  | 0 | 64 | 0 | 448 |
| cufftdx | fp16 (fp32 compute) | 1024 |  | 0 | 152 | 0 | 944 |
| cufftdx | fp16 (fp32 compute) | 2048 |  | 0 | 160 | 0 | 984 |
| cufftdx | fp16 (fp32 compute) | 4096 |  | 0 | 172 | 0 | 1048 |
| cufftdx | fp16 (fp32 compute) | 8192 |  | 0 | 228 | 0 | 1288 |
| cufftdx | fp16 (fp32 compute) | 16384 |  | 0 | 228 | 0 | 1320 |
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

Every fp16 Monarch kernel has MMA > 0; fp32 Monarch and all cuFFTDx kernels have none. The mixed kernels have the
fp32 kernels' FFMA counts and no HFMA2, so the transform itself runs in fp32; their extra instructions over the
fp32 kernels (e.g. 688 vs 592 at N = 256) fit the half-float conversions on load and store. In the 8×32 Monarch
configuration, GEMM 1 (N = 8 columns) stays on FMA: the suggested path selects the 16×8×8 HMMA atom only when the
GEMM divides a 32×16×8 warp tile (`docs/cublasdx_sm75_layouts.md`).

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

1. **At equal storage (fp16 in, fp16 out), cuFFTDx mixed is both more accurate and faster than Monarch.**
   - Error: Monarch has 1.35-1.37× the total error of the mixed variant (3.73-3.81e-4 vs 2.77-2.78e-4) and
     1.57-1.60× its compute-only error (3.24-3.33e-4 vs 2.07-2.08e-4), at every N from 256 to 4096.
   - Speed: Monarch runs at 0.85× (N = 256 and 512), 0.80× (1024), 0.47× (2048) and 0.53× (4096) of tuned
     mixed. No range reaches 1.0.
2. **Mixed costs almost nothing in speed at small N and loses at large N.** Against tuned pure-fp16 cuFFTDx it runs
   at 0.98× (256), 0.95× (512), 1.02× (1024, a tie), then 0.88× (2048), 0.76× (4096) and 0.74× (8192, 16384). It
   has 2.9× (N = 256) to 9.0× (N = 16384) less error than the most accurate pure-fp16 configuration.
3. **Against pure-fp16 cuFFTDx, Monarch is more accurate and slower.** 2.1-3.0× less total error than the most
   accurate fp16 configuration, at 0.83× (256), 0.80× (512), 0.81× (1024) and 0.41× (2048, 4096) of the fastest.
   Against cuFFTDx's default fp16 configuration Monarch was faster at N = 256 (196.6 vs 161.4 GB/s); EPT tuning
   removes that.
4. **fp32: Monarch is slower and less accurate.** It runs on FFMA at 0.14-0.37× of tuned cuFFTDx's throughput,
   with 1.3-1.8× the error of the most accurate cuFFTDx configuration.
5. **Run-to-run variation.** Most ratio ranges over repeats 2-3 are under 0.05 wide. The widest are fp16
   tuned/default at 1024 ([1.01, 1.25]), mixed/tuned at 2048 ([0.79, 0.98]), Monarch/mixed at 1024 ([0.74, 0.86])
   and 2048 ([0.43, 0.51]), and fp32 tuned/default at 8192 ([0.91, 1.03]). Absolute throughput drifts down across the three repeats at large N, most for the mixed variant:
   its default configuration at N = 4096 goes 219.0, 186.3, 171.6 GB/s (−22 %); pure fp16 at N = 8192 goes 243.3,
   230.7, 217.0 (−11 %). Ratios are taken within a repeat, which removes a drift common to both kernels but not one
   that hits one kernel more.

Interpretation:

- *(interpretation)* The mixed variant sits on the fp16 storage floor (Accuracy section): computing the
  transform in fp32 adds nothing measurable, and what remains is the rounding of the output, plus the input in
  the total. No fp16-storage FFT can do better on this metric.
- *(interpretation)* Monarch keeps its products in fp32 on HMMA, but rounds A2 to fp16 between the two GEMMs, then
  rounds the output. One fp16 rounding more than the mixed variant fits its larger compute-only error (1.57-1.60×),
  flat in N. Pure-fp16 cuFFTDx rounds after every butterfly stage in HFMA2, which fits an error
  that grows with N.
- *(interpretation)* The earlier reading of the Monarch result, "tensor cores make it more accurate", was wrong in
  attribution: the gain over pure-fp16 cuFFTDx comes from accumulating in fp32. A conventional FFT that computes in
  fp32 on fp16 storage gets that gain without the extra intermediate rounding, and on the T4 it is also faster.
- *(interpretation, from the ncu table)* At N = 4096, Monarch's time goes into the block itself, not into DRAM:
  it executes 2.8× cuFFTDx's instructions and about 4.5× its shared-memory wavefronts, with heavy bank conflicts.
  The dense DFT costs N·(n1 + n2) multiply-adds per FFT, against O(N log N).
- *(interpretation)* I don't know why pure-fp16 cuFFTDx's error drops as EPT goes down. I note it as an
  observation.
- *(interpretation)* What is left for the tensor-core formulation on this GPU is narrow: it beats pure-fp16
  cuFFTDx on accuracy only. Whether Monarch wins anywhere when the FFT is fused with other tensor-core work in the
  same kernel, or when the A2 intermediate is kept in fp32 (it then would not feed GEMM 2 as an fp16 operand), is
  not measured here.

## Limitations

- **One GPU.** T4 only (Turing, sm_75). It has fp16 HMMA but no TF32/BF16 tensor cores. Nothing here says how
  Ampere or Hopper behave, where both the tensor cores and the fp32 throughput differ.
- **One session for the final numbers.** Three repeats in one Colab session bound the within-session noise, and
  absolute throughput drifts across them (finding 5). Clocks are not locked. The 11-18 session gives similar ratios;
  the largest difference is fp16 tuned/default at N = 1024 (1.28 then, 1.13 [1.01, 1.25] now).
- **Tuning is asymmetric in kind.** cuFFTDx gets its two exposed knobs (EPT × FPB) over the legal range. Monarch
  gets its factorization, plus structural choices I made by hand (persistent grid, shared F table, transposed
  GEMM 1) that have no cuFFTDx counterpart. I did not write a custom cuFFTDx kernel, and the mixed variant does
  not use the example's experimental `Transposition` layout.
- **Profiling is partial.** Four fp16 kernels, from an earlier session, at the base clock that ncu imposes. The
  tuned and mixed cuFFTDx configurations, and all fp32 kernels, were not profiled. The grid binaries were not
  SASS-checked.
- **Limited sizes.** N ≤ 4096 for Monarch, power-of-two two-factor splits only. fp32 2048 is missing: with the
  8 FFTs per 4096 elements I use everywhere (2 per block at N = 2048) it needs 72 KB of shared memory. One FFT per
  block would need 56 KB and fit; I did not run that configuration.
- **One input distribution.** Uniform inputs only.
- **No Monarch variant with an fp32 intermediate,** and no fusion experiment.

## Prior work

- **FlashFFTConv** (Fu et al., ICLR 2024) runs FFT convolutions as Monarch decompositions on tensor cores. The
  Monarch idea here is theirs.
- **tcFFT** (Li, Cheng, Lin, 2021, arXiv:2104.11471), half-precision FFT on tensor cores. Its precision section
  (5.2, Table 4) reports the same average relative error as half-precision cuFFT and names the storage of
  intermediate results in half precision between merging stages as the main source of error. My numbers fit that
  explanation: Monarch has two merging stages and one fp16 intermediate, and the mixed cuFFTDx variant, with none,
  is more accurate still.
- **Split fp32 → 2× fp16 on tensor cores:** the SC18 student poster by Cheng and Sorna, supervised by D'Azevedo (ORNL), "Accelerating
  2D FFT: Exploit GPU Tensor Cores through Mixed-Precision", and Sorna et al. (with Wong and Tomov, UTK),
  "Optimizing the Fast Fourier Transform using Mixed Precision on Tensor Core Hardware". They split each fp32 value
  into two fp16 values to recover accuracy from fp16 tensor cores. Not attempted here.
- **NVIDIA's MathDx examples** (convolution_performance, mixed_precision_fft_1d, gemm_fft, fused_gemm, ...): the
  cuFFTDx baselines follow them.
