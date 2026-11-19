# cuFFTDx agent guide (UNOFFICIAL)

> **UNOFFICIAL. Original work for the monarch-fft-mathdx repo, not written, reviewed or endorsed by NVIDIA.**
> NVIDIA ships an agent-targeted guide for cuBLASDx (`llms.txt` plus `markdown/` in the cuBLASDx HTML
> docs). On 2026-11-17 I found no equivalent for cuFFTDx:
> - `https://docs.nvidia.com/cuda/cufftdx/llms.txt` returns 404;
> - the MathDx 26.10 package has no `doc/cufftdx`;
> - the cuFFTDx table of contents has no agent page.
>
> This guide copies the cuBLASDx guide's style (short, procedural, checkable) for cuFFTDx 1.8.0 (MathDx 26.10).
> Every operator, trait and method named here was checked in the installed headers, and the header path is
> given. Statements about performance on a specific GPU come from measurements in this repo and say so.
> When this guide disagrees with the official docs or headers, they win.

cuFFTDx is the MathDx device-extension FFT library. You compose an FFT description at compile time from
operators, then call `execute` inside your own kernel, so the FFT can be fused with surrounding work.
The library does not load or store global memory for you; that is your kernel's job, and most correctness
bugs live there.

## Start here

1. [Describe the FFT](#describe-the-fft): operators, and which ones have defaults.
2. [Write the kernel](#write-the-kernel): load, execute and store through `FFT::working_group`.
3. [fp16](#fp16-two-ffts-per-register): two FFTs per register.
4. [Verify](#verify): accuracy against an FP64 reference, then SASS.
5. [Tune](#tune): only after 4 passes; defaults first.

Reference pages:
- [Operators](https://docs.nvidia.com/cuda/cufftdx/api/operators.html)
- [Traits](https://docs.nvidia.com/cuda/cufftdx/api/traits.html)
- [Execution methods](https://docs.nvidia.com/cuda/cufftdx/api/methods.html)
- [Achieving High Performance](https://docs.nvidia.com/cuda/cufftdx/performance.html)

Shipped examples are in `example/cufftdx/` in the package. `02_simple_fft_block` and
`03_block_fft_performance` are the closest to the pattern below.

## Build card

- **CMake:**
  ```
  find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG PATHS <MATHDX_ROOT>)
  target_link_libraries(<target> mathdx::cufftdx)
  ```
  Header-only; include `<cufftdx.hpp>`.
- **Release flags:** `-O3 -DNDEBUG`, a real architecture for the GPU you run on
  (`CMAKE_CUDA_ARCHITECTURES 75` for T4), and the same value in the `SM<>` operator.
- **Per-architecture code:** wrap the kernel body in `CUFFTDX_SKIP_IF_NOT_APPLICABLE_SM_BLOCK(FFT) { ... }`
  (`cufftdx/detail/commondx_config.hpp`). This lets one source build for several architectures.

## Describe the FFT

A description is a sum of operators (`include/cufftdx/operators/`, `commondx/operators/`):

| operator | required | values / notes |
|---|---|---|
| `Size<N>()` | yes | Max N depends on SM and precision. sm_75: 16384 for fp16 and fp32, 8192 for fp64 (`CUFFTDX_DETAIL_SM750_*_MAX`, `cufftdx/detail/fft_sizes.hpp`). |
| `Type<fft_type::c2c>()` | default c2c | `c2c`, `c2r`, `r2c` (`operators/type.hpp`). R2C/C2R take `RealFFTOptions<complex_layout, real_mode>`; see `operators/real_fft_options.hpp`. |
| `Direction<fft_direction::forward>()` | yes for c2c | `forward`, `inverse` (`operators/direction.hpp`). Forward is unnormalized; scaling is yours. |
| `Precision<T>()` | default float | `__half`, `float`, `double` |
| `SM<CC*10>()` | yes for execution | e.g. `SM<750>()` |
| `Block()` or `Thread()` | yes for execution | `operators.hpp` |
| `ElementsPerThread<E>()` | no | Default: `FFT::elements_per_thread` of the description without it. |
| `FFTsPerBlock<B>()` | no | Default 1. `FFT::suggested_ffts_per_block` is the library's suggestion. |
| `BlockDim<X,Y,Z>()` | no | Only when surrounding code fixes the block shape. |

The library's defaults, as `03_block_fft_performance` uses them:

```cpp
using base = decltype(Block() + Size<N>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                      Precision<T>() + SM<750>());
using FFT  = decltype(base() + ElementsPerThread<base::elements_per_thread>() +
                      FFTsPerBlock<base::suggested_ffts_per_block>());
```

Traits to read before writing the kernel (`cufftdx/detail/fft_execution.hpp`, `traits/`):
- **Launch:** `block_dim`, `max_threads_per_block`.
- **Memory:** `shared_memory_size` is the dynamic shared memory in bytes. Above 48 KB, call
  `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`.
- **Per-thread storage:** `storage_size` (registers per thread, in `value_type`), `elements_per_thread`,
  `ffts_per_block`, `stride`, `input_ept` / `output_ept`, `input_length` / `output_length`.
- **Special cases:**
  - `requires_workspace`: true for sizes cuFFTDx runs with Bluestein. Create the workspace with
    `cufftdx::make_workspace<FFT>(error, stream)` (`cufftdx/detail/workspace.hpp`) and pass it to `execute`. Power-of-two sizes do not need one;
    `static_assert(!FFT::requires_workspace)` documents that.
  - `implicit_type_batching`: 2 for fp16, see below.

## Write the kernel

Block execution with registers plus shared-memory scratch (`execute(value_type*, void*)`,
`fft_execution.hpp:697`). The pattern, as used in `cuda/cufftdx/block_fft.cu` here and in the shipped
`common/block_io.hpp`:

```cpp
template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void k(const typename FFT::value_type* in,
                                                                typename FFT::value_type* out) {
  CUFFTDX_SKIP_IF_NOT_APPLICABLE_SM_BLOCK(FFT) {
    extern __shared__ __align__(alignof(float4)) cufftdx::byte smem[];
    typename FFT::value_type td[FFT::storage_size];
    const size_t off = (size_t(blockIdx.x) * FFT::ffts_per_block + FFT::working_group::batchid()) *
                       cufftdx::size_of<FFT>::value;
    if (FFT::working_group::is_thread_active())
      for (unsigned i = 0; i < FFT::input_ept; ++i) {
        const unsigned k = FFT::working_group::fftid() + i * FFT::stride;
        if (k < FFT::input_length) td[i] = in[off + k];
      }
    FFT().execute(td, smem);          // every thread of the block calls execute, active or not
    if (FFT::working_group::is_thread_active())
      for (unsigned i = 0; i < FFT::output_ept; ++i) {
        const unsigned k = FFT::working_group::fftid() + i * FFT::stride;
        if (k < FFT::output_length) out[off + k] = td[i];
      }
  }
}
// launch: grid = batch / FFT::ffts_per_block, block = FFT::block_dim, smem = FFT::shared_memory_size
```

Rules that cause silent wrong answers when broken:
- **Index with `FFT::working_group`** (`batchid()`, `fftid()`, `is_thread_active()`), not with
  `threadIdx` (`traits/detail/working_group.hpp`). Some older public samples index with `threadIdx`
  directly, but the 26.10 package samples go through `working_group`.
- **Element `i` of a thread is `fftid() + i * stride`.** It is not contiguous per thread.
- **Every thread calls `execute`.** Inactive threads guard only their loads and stores.
- **Launch with `FFT::block_dim`,** or with exactly the `BlockDim` you specified.
- **The grid covers `batch / ffts_per_block` blocks.** Pad the batch, or guard the tail block.

## fp16: two FFTs per register

For `Precision<__half>`, `FFT::implicit_type_batching == 2`:
- `value_type` is `complex<__half2>` in RRII layout: `x = (re_a, re_b)` and `y = (im_a, im_b)` hold two FFTs.
- `working_group::batchid()` only ranges over `[0, ffts_per_block / 2)`, so `ffts_per_block` must be even.

A natural-layout global array (one `__half2 = (re, im)` per element) has to be repacked on load and
unpacked on store. The shipped `common/fp16_common.hpp` pairs FFT `b` with FFT `b + ffts_per_block/2` of
the same block:

```cpp
td[i] = complex<__half2>(__lows2half2(in[off + k], in[off + pair + k]),
                         __highs2half2(in[off + k], in[off + pair + k]));   // pair = ffts_per_block/2 * N
out[off + k] = __lows2half2(td[i].x, td[i].y);  out[off + pair + k] = __highs2half2(td[i].x, td[i].y);
```

Skip this and the fp16 FFT still "runs". Its output mixes two FFTs, and an error check against a
reference catches it at once (relative error near 1).

## Verify

1. **Accuracy against an FP64 reference on the same inputs.**
   - The reference: cuFFT Z2Z on the unrounded fp64 input.
   - The metric: relative L2 error over a batch, median over several seeds, plus max abs error.
   - Measured on a T4 in this repo, forward c2c, U(-1, 1) inputs (`results/tesla_t4/2026-11-17_sweep_3.jsonl`):
     fp32 has rel L2 1.4–2.4e-7; fp16 has 1.2–2.5e-3, and the error grows with N (256 to 16384).
   - A bug in indexing or fp16 packing shows up as rel L2 of order 1, not 1e-3.
2. **Instructions** (`cuobjdump -sass`): cuFFTDx radix kernels are FMA code. On sm_75 this repo measured
   FFMA-only fp32 kernels and HFMA2-only fp16 kernels, with no MMA (`scripts/sass_check.sh`). If you expect
   tensor cores in an FFT, cuFFTDx is not where they come from.
3. **Resources:** check `cuobjdump -res-usage` for registers and local memory. A spill shows up as `LOCAL > 0`.

## Tune

Follow [Achieving High Performance](https://docs.nvidia.com/cuda/cufftdx/performance.html). In short:
1. **Start from the defaults** above and measure:
   - CUDA events per launch, warm-up first, median and IQR over at least 20 launches;
   - enough blocks to fill the GPU;
   - never benchmark zero-filled buffers (the cuBLASDx guide's
     [measurement protocol](https://docs.nvidia.com/cuda/cublasdx/markdown/blas_performance_guide.md#perf-guide-measurement-label)
     applies unchanged).
2. **Tune `ElementsPerThread` and `FFTsPerBlock` together, per size.** NVIDIA's page reports up to a further
   1.4× from this in its examples. Larger `FFTsPerBlock` also helps coalescing.
3. **Shared-memory layout**, with `cufftdx::experimental::Transposition<transposition::...>`
   (`operators/experimental/transposition.hpp`). Values: `any`, `none`, `by_component`,
   `by_component_padded`, `by_complex`.
   - Try `by_component` when shared memory limits occupancy.
   - Try `by_complex` when it does not.
   - The performance page has the constraints of `by_component_padded` and `none`.
4. **Thread mapping** for strided multi-dimensional data:
   `cufftdx::experimental::ThreadLayout<thread_layout::transposed>` (`operators/experimental/thread_layout.hpp`).
5. **Fusion is the point of a device-side FFT.** Fold pre- and post-processing into the load and store
   loops before tuning the FFT itself. A memory-bound FFT kernel gains more from one fewer global round
   trip than from any FFT parameter.

The experimental operators are in the `experimental` namespace in 26.10; recheck them on upgrade.

## Mistakes an agent made in this repo (from docs/agent_log.md)

- **Assuming package layout and CMake targets from the latest docs** instead of the installed package's
  config. Different MathDx releases ship different targets.
- **Truncated kernel names** in SASS tooling made fp16 and fp32 kernels look identical, so the
  per-precision checks were wrong until N and precision were parsed out of the name.
- The cuFFTDx baseline itself compiled and passed on the first GPU run. Every name had been looked up in
  the headers first. That is the habit this guide tries to encode.

## Sources

- **Installed headers:** MathDx 26.10, cuFFTDx 1.8.0, under `include/cufftdx/` and `include/commondx/`.
- **Shipped examples:** `example/cufftdx/02_simple_fft_block`, `03_block_fft_performance`, `common/block_io.hpp`,
  `common/fp16_common.hpp` (Apache-2.0, NVIDIA).
- **NVIDIA cuFFTDx docs:** https://docs.nvidia.com/cuda/cufftdx/ (operators, traits, methods,
  performance pages).
- **NVIDIA cuBLASDx agent guide** (for the style and the measurement protocol):
  https://docs.nvidia.com/cuda/cublasdx/llms.txt
