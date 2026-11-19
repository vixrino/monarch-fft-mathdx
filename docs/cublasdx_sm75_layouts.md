# cuBLASDx on sm_75: complex fp16→fp32 GEMM uses FMA with the plain layouts, HMMA with the suggested ones

This describes observed behavior on one GPU. I am not claiming it is a bug; the question is at the end.

## Environment

- Tesla T4 (sm_75), driver 580.82.07 (Google Colab)
- CUDA 13.0.88 (nvcc V13.0.88)
- MathDx 26.10.0: cuBLASDx 0.8.0, cuFFTDx 1.8.0, `nvidia-mathdx-26.10.0-cuda13.tar.gz`
- Build: `-std=c++17 --expt-relaxed-constexpr -O3 -DNDEBUG -arch=sm_75`, header-only target `mathdx::cublasdx_no_lto`

## Observation

A 64×64×64 GEMM with `complex<half>` inputs and a `complex<float>` accumulator, 128 threads, `MaxAlignment()`,
compiled three ways that differ only in what selects the instruction. All three give the exact result on
small-integer inputs. The static SASS counts are per kernel, from `cuobjdump -sass`:

| variant | shared-memory layouts and accumulator | static HMMA | total SASS instructions |
|---|---|---:|---:|
| 0 | `get_layout_smem_a/b()`, `get_accumulator()` | 0 | 17216 |
| 1 | `suggest_layout_smem_a/b()`, `suggest_accumulator()` | 256 | 872 |
| 2 | variant 0 + `experimental::Tile<cublasdx::detail::SM75_16x8x8_C32C16C16C32_TN, 2, 2>` | 256 | 840 |

Variant 0 has no tensor-core instruction. I saw the same counts in two separate sessions, on 2026-11-17 and
2026-11-18. The SM 80 version of the same database file lists complex HMMA atoms for these sizes. I only read
that file; I have no Ampere GPU to run it.

## Where the selection happens in the headers

All paths below are under `include/cublasdx/database/gemm/`.

- **Plain layouts.** `database_getter.hpp`, `get_tile_config()` (line 327), looks up the tuning database at
  `lower_bound_int_sqrt(M * N)` (line 338). It falls back to `get_mma_family<...>()` (line 429) only when no
  record matches.
  - `sm75/cfp16_cfp16_cfp32_tn.hpp.inc` has 24 records and all of them are `mma_atom::universal_fma`. For
    example, line 5: `generated_config_impl<128, mma_atom::universal_fma, 16, 8>`.
  - The real-valued file `sm75/fp16_fp16_fp32_tn.hpp.inc` has 36 records, all `universal_fma` as well. I only
    read that file; I did not run a real fp16 GEMM with the plain layouts.
- **Suggested layouts.** `suggested/suggested_supermma.hpp` (line 296) picks the instruction with
  `search_mma_list` (`suggested/common.hpp`, line 90).
  - The warp grid is `get_floor_tile(num_warps)` (line 38), (2, 2) for 128 threads.
  - An atom is kept only if `cute::evenly_divides(GEMMShape, warp_tile × atom_shape)`. For
    `SM75_16x8x8_C32C16C16C32_TN` that means M % 32 == 0, N % 16 == 0 and K % 8 == 0. Otherwise `type = void`
    and the GEMM uses FMA.
  - In my code this applied to 8×N GEMMs; transposing the product so that the batch dimension became M gave HMMA.

## The documentation sentence it differs from

`doc/cublasdx/html/markdown/performance_guide_api_supermma.md` (Agent BLAS Performance Guide, API Card:
supermma) lists as tensor-core precisions "`half` inputs with a `half` or `float` accumulator from SM 75",
adds "The complex `Type` of each of those triples is on the table as well", and then says:

> Scalar FMA is chosen only for `M == 1` or `N == 1` (matrix-vector) and `K == 1` (outer product), for every
> precision; any other shape, including a small tile, uses the tensor-core instruction when the precision has
> one.

The card says it describes the configuration with suggested layouts and the suggested accumulator, and with
that configuration (variant 1) the observed behavior matches the card, including its divisibility bullet. The
quoted sentence is not limited to that configuration, though. With the plain layouts (variant 0), the same
precision and shape compile to FMA on sm_75. The card's own advice to "confirm the generated instruction in the
SASS rather than inferring it from divisibility alone" is what caught it.

I don't know whether the sm_75 database choice is a deliberate tuning result for T4 or not. I have not
measured variant 0 against variant 1 at this size beyond the instruction counts.

## Minimal repro

[`cuda/monarch/mma_probe.cu`](https://github.com/vixrino/monarch-fft-mathdx/blob/main/cuda/monarch/mma_probe.cu)
(one source file, about 110 lines). The descriptor:

```cpp
using base = decltype(cublasdx::Size<64, 64, 64>() + cublasdx::Precision<__half, __half, float>() +
                      cublasdx::Type<cublasdx::type::complex>() + cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>() + cublasdx::MaxAlignment() +
                      cublasdx::Block() + cublasdx::BlockDim<128>() + cublasdx::SM<750>());
```

The kernel body is the same in all variants: `copy` to shared memory, `copy_wait`, `execute(sa, sb, acc)`,
`partition_and_store`. Only the layout and accumulator calls differ. Build and count:

```bash
cmake -S . -B build -DCUDA_ARCH=75 -DMATHDX_ROOT=<.../nvidia/mathdx/26.10> && cmake --build build --target mma_probe
./build/mma_probe                                  # plain / suggested / forced: max |err| = 0
bash scripts/sass_check.sh build/mma_probe probe   # total, mma, ffma, hfma2 per kernel
cuobjdump -sass build/mma_probe | grep -c HMMA     # without the repo script: HMMA over all three kernels
```

## Question

Is FMA the intended instruction on sm_75 for this precision and shape when the plain layouts are used? If it is,
could the supermma card's sentence about scalar FMA be scoped to the suggested-layout configuration, or mention
that the tuning database can select FMA for other layouts?
