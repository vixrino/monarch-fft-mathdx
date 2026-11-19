# cuBLASDx 0.8.0 on sm_75: complex fp16 × fp16 → fp32 GEMM gets no MMA with the plain shared-memory layouts. Intended?

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
in fp16 and fp32): all three give max |err| = 0.

## Results (one session, 2026-11-19)

| kernel | static HMMA | static SASS instructions | shared memory / blocks per SM | time per launch, median [IQR] | GEMM TFLOP/s |
|---|---:|---:|---:|---:|---:|
| plain | 0 | 17224 | 32768 B / 2 | 13.84 ms [13.80, 15.35] | 1.24 |
| suggested | 256 | 880 | 32768 B / 2 | 2.33 ms [2.33, 2.35] | 7.36 |
| forced | 256 | 848 | 32768 B / 2 | 2.33 ms [2.32, 2.34] | 7.38 |

The plain kernel is 5.9× slower than the suggested one at the same shared memory and occupancy. The times include
loading A and B and storing C from global memory (64 KB per GEMM); at 2.33 ms that is about 230 GB/s, so the
suggested kernels may be partly memory-bound and the ratio is a lower bound on the difference in the GEMM itself.
My SASS count script finds no FFMA or HFMA2 opcode in the plain kernel either, so I have not identified which
instructions it uses beyond "no HMMA".

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

For complex `half × half → float` on sm_75, is it intended that the plain-layout path selects no MMA (the sm_75
tuning records are all `universal_fma`), while the suggested path and a forced tile reach HMMA at 5.9× the
throughput in this test? If it is intended, would a sentence in the Requirements note help, e.g. that the MMA
selection there assumes the suggested layouts?

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
./build/mma_probe                                  # max |err| and timing per kernel
cuobjdump -sass build/mma_probe | grep -c HMMA     # HMMA over the three kernels
```

The numbers above are in
[`results/tesla_t4/2026-11-19_mma_probe.jsonl`](https://github.com/vixrino/monarch-fft-mathdx/blob/main/results/tesla_t4/2026-11-19_mma_probe.jsonl)
(commit d768996, with GPU, driver, CUDA and MathDx versions). In that file the `variant` field of the three SASS
records reads `(`, a parsing bug in my script since fixed; the `function` field (`probe<(int)0, ...>` = plain,
`1` = suggested, `2` = forced) identifies them.
