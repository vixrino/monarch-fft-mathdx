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
