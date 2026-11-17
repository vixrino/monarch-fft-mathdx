# Decisions

One entry per non-obvious choice: what, alternatives, why, evidence.

## Phase 0

### MathDx package picked from the runtime's CUDA major version
The notebook reads `nvcc --version` and downloads the matching package.
The 26.x MathDx releases are CUDA 13 only; the newest CUDA 12 build is 25.12.1
(both checked on developer.nvidia.com/cufftdx-downloads on 2026-11-17). Colab
and Kaggle images change their CUDA version without notice, so hard-coding one
package would break silently. Downside: the two runtimes can end up on
different MathDx versions, so every result records `mathdx_VERSION`.
Unverified: whether 25.12.1-cuda12 behaves the same on sm_75 as 26.10; the
current cuFFTDx docs list CUDA 13.0+ and SM75+ as requirements.

### Header-only cuBLASDx target: `cublasdx_no_lto` on 26.x, `cublasdx` on 25.12
In 26.x the default `mathdx::cublasdx` target links an LTO fatbin and requires
`CUDA_SEPARABLE_COMPILATION` + `INTERPROCEDURAL_OPTIMIZATION` on the consumer;
on CUDA < 13.2 one must use `mathdx::cublasdx_fatbin` instead. The fatbin is
only needed for non-GEMM functions (TRSM etc.). The Monarch FFT only uses GEMM,
so the header-only `mathdx::cublasdx_no_lto` is enough.
The 25.12.1-cuda12 package has no `_no_lto` (or `_fatbin`) target at all: its
`lib/cmake/cublasdx/cublasdx-config.cmake` only links CUTLASS and commondx into
`mathdx::cublasdx`, i.e. it is already header-only. CMake picks `_no_lto` when
the target exists and falls back to `mathdx::cublasdx` otherwise. Evidence: I
listed both tarballs and read their cmake configs (2026-11-17).

### `find_package(mathdx ... PATHS ${MATHDX_ROOT} NO_DEFAULT_PATH)`
An explicit root and no default search path, so a stray install under
`/opt/nvidia/mathdx` can never be picked up instead of the package the
notebook just downloaded. Configure fails if `MATHDX_ROOT` is empty.

### `CUDA_ARCH=75` maps to `CMAKE_CUDA_ARCHITECTURES=75`
This builds both SASS and PTX for sm_75. SASS is what I grep for HMMA later;
the PTX costs nothing and allows JIT on a newer GPU.

### cuFFTDx smoke test built from the package's own examples
I build `convolution_performance` with NVIDIA's examples CMakeLists rather
than copying it into this repo. The target is `06_convolution_convolution_performance` in the
`example/cufftdx/CMakeLists.txt` of both 25.12.1 and 26.10.0 (`<dir>_<file>`);
the notebook still reads it back from `--target help` so a rename in a later
package does not break the runner.

### Package layout differs from the installation docs
The docs say the package unpacks to `<dir>/nvidia/mathdx/YY.MM/`. Both
tarballs actually have one extra top-level directory:
`nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12/`. The first Colab run
failed on this (`ls: cannot access '/content/mathdx/nvidia/mathdx/*/'`); the
notebook now globs `$DX_DIR/*/nvidia/mathdx/*/`.

### ncu permission check in env_check.sh
I expected Colab and Kaggle to block performance counters for non-admin users
(`ERR_NVGPUCTRPERM`), so the script reports that case explicitly. That
expectation was wrong for Colab: on the first run (T4, driver 580.82.07,
ncu 2025.3.1) `sm__cycles_elapsed.avg` was readable. So Nsight Compute metrics
may not need a separate VM, at least on T4. Kaggle not checked yet.

### `cmake_minimum_required(VERSION 3.26...3.31)`
With a plain `3.26`, policy CMP0144 (3.27) is OLD and CMake warns that it
ignores the `MATHDX_ROOT` variable as a find_package hint, because it looks
like the upper-case `<PackageName>_ROOT`. The version range turns CMP0144 on
and silences the warning; behaviour is the same since I pass `PATHS
${MATHDX_ROOT}` explicitly anyway. Colab ships CMake 3.31.10.

## Phase 0 environment (Colab, 2026-11-17)
Tesla T4 (sm_75), driver 580.82.07, CUDA 13.0.88, GCC 13.3.0, CMake 3.31.10,
MathDx 26.10.0 (cuFFTDx 1.8.0, cuBLASDx 0.8.0), commit fac318c.

## Phase 1

### Index maps n = n2·i1 + i2, k = j1 + n1·j2
This makes the input a plain row-major reshape to (n1, n2), so on the GPU a
block loads contiguous rows. The price is a transpose at the output. The
alternative (n = i1 + n1·i2) moves the transpose to the input instead; the
math is the same. I kept the transpose at the end because the later cuFFTDx
baseline also writes X in natural order and I want both kernels to read x
identically.

### Dense F_m with (j·i mod m) and one rounding per entry
See docs/derivation.md, "Twiddle and DFT-matrix precision". In fp64 the
reduction changes nothing measurable; it is there so the same generator can be
reused for fp32/fp16 tables without a hidden precision loss.

### fp16 emulation: where rounding happens
x, F_n1, F_n2 rounded to fp16; both GEMMs accumulate in fp32; twiddle multiply
in fp32; the intermediate c rounded back to fp16 because it is GEMM 2's
operand. Output stays fp32. Alternative: keep c in fp32 and only round the
matrices. Rejected because a tensor-core GEMM on T4 needs both operands in
fp16, so that would be optimistic. complex64 matmul is used instead of
4 real GEMMs: products of fp16 values are exact in fp32, so the difference
is only in summation order.

### Test tolerances
fp64: rel L2 < 1e-13. Worst case over all 66 factorizations, 5 inputs each,
was 6.6e-16 (N=16384, 128×128), local CPU run. fp16 emulation: rel L2 < 1e-2
is only a sanity bound; the printed table is the output.

### Mutation checks on the tests (local CPU, 2026-11-17)
Each bug injected into `reference/monarch.py`, `pytest -k "not fp16"`
(133 tests), then the file restored:

| injected bug                                              | failed / 133 |
|-----------------------------------------------------------|--------------|
| output read row-major, no final transpose                 | 133          |
| twiddle table read with stride n1 (`.reshape(n2, n1).T`)   | 121          |
| twiddle root w_n2 instead of w_N                          | 133          |
| twiddle sign +                                            | 133          |
| inverse scaled by 1/n1 instead of 1/N                     | 133          |

The 12 survivors of the stride bug are exactly the square cases (n1 = n2,
forward and inverse, N = 16 ... 16384): the twiddle table is symmetric there.
This is why the tests cover every factorization and not only n1 = n2.
Failed attempt worth noting: my first version of that mutation was
`twiddles(n2, n1).T`, which is identical to `twiddles(n1, n2)` and so
tested nothing; all 133 passed, which is how I noticed.

### fp16 emulation observation (CPU, not GPU)
Median rel L2 is 2.6e-4 to 3.4e-4 across all 66 cases and barely depends on
the factorization. Rounding only the input x to fp16 and doing an exact FFT
already gives 1.85e-4 (N=4096, 5 inputs). So in this model most of the error
comes from quantizing x, not from the GEMMs. To be checked against the GPU.
