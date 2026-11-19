# cuBLASDx 0.8.0 on sm_75: complex fp16 × fp16 → fp32 GEMM with the plain layouts gets no MMA and fp16 products. Intended?

Posted as [NVIDIA/CUDALibrarySamples#378](https://github.com/NVIDIA/CUDALibrarySamples/issues/378).

A question, scoped to what I ran: complex `__half` inputs, `float` accumulator, sm_75 (Tesla T4), cuBLASDx 0.8.0.
I have not run real-valued fp16, other shapes or other GPUs.

## Environment

- Tesla T4 (sm_75), driver 580.82.07 (Google Colab)
- CUDA 13.0 (nvcc V13.0.88)
- MathDx 26.10.0: cuBLASDx 0.8.0, `nvidia-mathdx-26.10.0-cuda13.tar.gz`
- Build: `-std=c++17 --expt-relaxed-constexpr -O3 -DNDEBUG -arch=sm_75`, header-only target `mathdx::cublasdx_no_lto`

## What I ran

One 64×64×64 GEMM per block, `complex<half>` A and B, `complex<float>` accumulator, 128 threads, `MaxAlignment()`,
row-major A and B. Three kernels in one binary differ only in what selects the instruction:

- **plain:** `get_layout_smem_a/b()` and `get_accumulator()`
- **suggested:** `suggest_layout_smem_a/b()` and `suggest_accumulator()`
- **forced:** plain + `experimental::Tile<cublasdx::detail::SM75_16x8x8_C32C16C16C32_TN, 2, 2>`

Each launch runs 8192 GEMMs (8192 blocks). Timing is CUDA events around each launch, 5 warm-up launches, then the
median and IQR of 50. Every GEMM of the batch is checked against an fp64 host GEMM on small-integer inputs (exact
in fp16 and fp32): all three give max |err| = 0. A second test uses random U(-1, 1) complex inputs rounded to
fp16 and compares against an fp64 GEMM on the same rounded inputs, so only the GEMM's own arithmetic is measured
(64 GEMMs per seed, median of 5 seeds).

## Results (2026-11-19, Tesla T4)

| kernel | static HMMA | dominant opcodes | static SASS instructions | time per launch, median [IQR] | rel L2 error, random fp16 inputs | max abs error |
|---|---:|---|---:|---:|---:|---:|
| plain | 0 | 8192 HMUL2, 8192 FADD | 17224 | 13.82 ms [13.80, 13.84] | 2.03e-4 | 3.99e-3 |
| suggested | 256 | 256 HMMA | 880 | 2.34 ms [2.33, 2.35] | 2.64e-7 | 9.52e-6 |
| forced | 256 | 256 HMMA | 848 | 2.32 ms [2.32, 2.34] | 2.64e-7 | 9.52e-6 |

All three kernels use 32768 B of shared memory and run 2 blocks per SM. Two earlier runs of the same timing
(before the accuracy test was added) gave 13.84 / 13.80 ms for plain and 2.33 / 2.34 ms for suggested.

- **Speed.** The plain kernel is 5.9× slower. The times include loading A and B and storing C from global memory
  (64 KB per GEMM); at 2.33 ms that is about 230 GB/s, so the suggested kernels may be partly memory-bound and the
  ratio is a lower bound on the difference in the GEMM itself.
- **Accuracy.** The plain kernel's error is 770× the suggested one. Its SASS multiplies with HMUL2 (fp16 x fp16,
  fp16 result) and adds with FADD (fp32). My reading: each product is rounded to fp16 before the fp32 add, so the
  error is at the level of fp16 rounding (unit roundoff 2^-11 ≈ 4.9e-4) although the accumulator type is `float`.
  The suggested and forced kernels, on HMMA with fp32 accumulation, are at the fp32 level.

## Where the selection happens in the headers

All paths below are under `include/cublasdx/database/gemm/`.

- **Plain layouts.** `database_getter.hpp`, `get_tile_config()` (line 327), looks up the tuning database at
  `lower_bound_int_sqrt(M * N)` (line 338). It falls back to `get_mma_family<...>()` (line 429) only when no
  record matches.
  - `sm75/cfp16_cfp16_cfp32_tn.hpp.inc` has 24 records, all `mma_atom::universal_fma` (e.g. line 5:
    `generated_config_impl<128, mma_atom::universal_fma, 16, 8>`).
  - For comparison, `sm80/cfp16_cfp16_cfp32_tn.hpp.inc` lists 14 `SM75_16x8x8_C32C16C16C32_TN_CUBLASDX` and 28
    `SM80_16x8x16_C32C16C16C32_TN_CUBLASDX` records. I only read that file; I have no Ampere GPU.
- **Suggested layouts.** `suggested/suggested_supermma.hpp` (line 296) picks the instruction with
  `search_mma_list` (`suggested/common.hpp`, line 90). The warp grid is `get_floor_tile(num_warps)` (line 38),
  (2, 2) for 128 threads, and an atom is kept only if `cute::evenly_divides(GEMMShape, warp_tile × atom_shape)`:
  for `SM75_16x8x8_C32C16C16C32_TN`, M % 32 == 0, N % 16 == 0, K % 8 == 0.

## Documentation I compared against

- **Requirements, "Supported MMA Data Types"** (`requirements_func.html`): the row `half | half | float` reads
  "MMA for SM75+", and the note under the table says that if an MMA instruction exists for the combination of
  precisions of A, B and C, cuBLASDx uses it automatically on supported architectures, and FMA otherwise. The
  table lists precisions, not real vs complex; I assumed it covers `complex<half>` as well.
- **Agent BLAS Performance Guide, API Card: supermma** (`performance_guide_api_supermma.md`) describes the
  configuration with suggested layouts and the suggested accumulator. With that configuration (the *suggested*
  kernel) the observed behavior matches the card, including its divisibility rule. So this is not a contradiction
  of that card.

## Question

For complex `half × half → float` on sm_75 with the plain layouts:

1. Is it intended that this path selects no MMA (the sm_75 tuning records are all `universal_fma`), while the
   suggested path and a forced tile reach HMMA, 5.9× faster in this test?
2. Is it intended that the FMA path forms the products in fp16 (HMUL2) when the accumulator precision is `float`?
   With random inputs this gives 770× the error of the HMMA path, i.e. roughly fp16-level results from a
   descriptor that asks for fp32 accumulation.

If both are intended, would a sentence in the Requirements note help, e.g. that the MMA selection there assumes
the suggested layouts, and what precision the FMA fallback uses for the products?

## Minimal repro

[`cuda/monarch/mma_probe.cu`](https://github.com/vixrino/monarch-fft-mathdx/blob/main/cuda/monarch/mma_probe.cu),
one source file. The descriptor:

```cpp
using base = decltype(cublasdx::Size<64, 64, 64>() + cublasdx::Precision<__half, __half, float>() +
                      cublasdx::Type<cublasdx::type::complex>() + cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>() + cublasdx::MaxAlignment() +
                      cublasdx::Block() + cublasdx::BlockDim<128>() + cublasdx::SM<750>());
```

The kernel body is the same for the three kernels: offset to the block's GEMM, `copy` to shared memory,
`copy_wait`, `execute(sa, sb, acc)`, `partition_and_store`. Build, run and count:

```bash
cmake -S . -B build -DCUDA_ARCH=75 -DMATHDX_ROOT=<.../nvidia/mathdx/26.10> && cmake --build build --target mma_probe
./build/mma_probe                                  # exactness, timing and random-input error per kernel
cuobjdump -sass build/mma_probe | grep -c HMMA     # HMMA over the three kernels
cuobjdump -sass build/mma_probe | grep -c HMUL2    # HMUL2 (plain kernel)
```

The table is from
[`results/tesla_t4/2026-11-19_mma_probe_3.jsonl`](https://github.com/vixrino/monarch-fft-mathdx/blob/main/results/tesla_t4/2026-11-19_mma_probe_3.jsonl)
(commit 6d3b4ab; timing, error, SASS counts and opcode histograms, with GPU, driver, CUDA and MathDx versions).
The two earlier timing runs are `2026-11-19_mma_probe.jsonl` and `_2.jsonl` (commit d768996); their SASS records
have `variant` = `(`, a parsing bug in my script since fixed, and are identified by the `function` field.
