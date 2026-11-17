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

### `mathdx::cublasdx_no_lto` instead of `mathdx::cublasdx`
The default `mathdx::cublasdx` target links an LTO fatbin and requires
`CUDA_SEPARABLE_COMPILATION` + `INTERPROCEDURAL_OPTIMIZATION` on the consumer;
on CUDA < 13.2 one must use `mathdx::cublasdx_fatbin` instead. The fatbin is
only needed for non-GEMM functions (TRSM etc.). The Monarch FFT only uses GEMM,
so the header-only `no_lto` target is enough and keeps the build identical on
CUDA 12 and 13. Source: cuBLASDx installation docs. Revisit if a GEMM feature
I need turns out to depend on the fatbin.

### `find_package(mathdx ... PATHS ${MATHDX_ROOT} NO_DEFAULT_PATH)`
An explicit root and no default search path, so a stray install under
`/opt/nvidia/mathdx` can never be picked up instead of the package the
notebook just downloaded. Configure fails if `MATHDX_ROOT` is empty.

### `CUDA_ARCH=75` maps to `CMAKE_CUDA_ARCHITECTURES=75`
This builds both SASS and PTX for sm_75. SASS is what I grep for HMMA later;
the PTX costs nothing and allows JIT on a newer GPU.

### cuFFTDx smoke test built from the package's own examples
I build `convolution_performance` with NVIDIA's examples CMakeLists rather
than copying it into this repo. The target name (`06_convolution_convolution_performance`)
comes from the CUDALibrarySamples copy of that CMakeLists; the packaged copy may
differ, so the notebook reads the target name back from `--target help`
instead of hard-coding it.

### ncu permission check in env_check.sh
Colab and Kaggle run without root, and the driver defaults to restricting
performance counters to admins, so `ncu` is expected to fail there with
`ERR_NVGPUCTRPERM`. The script reports this explicitly so the plan for
Nsight Compute metrics (VM with admin access) does not rest on a guess.
