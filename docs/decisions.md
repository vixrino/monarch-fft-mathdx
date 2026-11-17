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

## Phase 2

### Every cuFFTDx name checked in the 26.10 headers before use
`Block`, `Size`, `Type`, `Direction`, `Precision`, `SM`, `ElementsPerThread`,
`FFTsPerBlock` (cufftdx/operators/, commondx/operators/sm.hpp);
`elements_per_thread`, `suggested_ffts_per_block`, `storage_size`, `stride`,
`input_ept`/`output_ept`, `input_length`/`output_length`, `block_dim`,
`max_threads_per_block`, `shared_memory_size`, `requires_workspace`,
`implicit_type_batching`, `working_group::{batchid, fftid, is_thread_active}`
(cufftdx/detail/fft_execution.hpp, cufftdx/traits/detail/working_group.hpp);
`execute(value_type*, void* smem)` (fft_execution.hpp:697);
`CUFFTDX_SKIP_IF_NOT_APPLICABLE_SM_BLOCK` (detail/commondx_config.hpp).
The `common/block_io.hpp` on GitHub (CUDALibrarySamples, master, checked
2026-11-17) indexes threads with `threadIdx` directly; the copy shipped in
the 26.10 package goes through `FFT::working_group`. I followed the package.
Max block size on sm_75 is 16384 for both fp16 and fp32
(`CUFFTDX_DETAIL_SM750_FP{16,32}_MAX`, detail/fft_sizes.hpp), so the sweep goes
to 16384. Configurations whose `shared_memory_size` exceeds the device opt-in
limit are recorded as `skipped_smem`, not silently dropped.

### EPT and FFTs-per-block: cuFFTDx defaults
`ElementsPerThread<base::elements_per_thread>` and
`FFTsPerBlock<base::suggested_ffts_per_block>`, as in
`block_fft_performance.hpp` with `UseSuggested`. The baseline should be what
a user gets from the library's own heuristics, not something I tuned to look
weak or strong. Tuning both sides is a later, separate experiment.

### fp16 implicit batching and global layout
cuFFTDx fp16 holds two FFTs per register: `complex<__half2>` in RRII layout
((re_a, re_b), (im_a, im_b)), and `batchid()` ranges over [0, fpb/2). I keep
the global array in the natural layout (one `__half2` = (re, im) per element,
FFTs contiguous) and pair FFT b with FFT b + fpb/2 of the same block, exactly
as `io_fp16` in common/block_io.hpp does. Alternative: store data pre-packed
as RRII in global memory. Rejected because the Monarch kernel and the
reference both use the natural layout, and the repack is a few register moves.

### Out-of-place kernel
The official performance example works in place and repeats `execute`
inside the kernel. I time whole launches out of place instead: repeated
in-place forward FFTs grow by ~sqrt(N) per pass and overflow fp16 within a
few iterations, and launch-level timing is what the Monarch kernel will be
compared on.

### Timing
One CUDA event pair per launch, 5 warm-up launches, 50 timed launches, median
and IQR. Two batches: one block (`ffts_per_block` FFTs; latency) and 2^14
rounded up to a multiple of fpb (throughput). Effective GB/s counts one read
and one write of the data: 2·batch·N·sizeof(complex T). Launch overhead is
inside the single-block number on purpose.

### Accuracy
Inputs are generated by a counter-based hash (splitmix64 of seed and element
index) directly on the GPU, so element i is identical in the fp64, fp32 and
fp16 buffers apart from the final rounding. The FP64 reference is cuFFT Z2Z
on the unrounded fp64 input, so the reported error includes quantizing the
input to T, same as the Phase 1 emulation. 5 seeds, 64 FFTs each (rounded up
to fpb), rel L2 over the whole batch per seed, median over seeds. `--check`
fails if the median exceeds 1e-5 (fp32) or 1e-2 (fp16): sanity bounds, not
expected values.

### Provenance
GPU name, SM, SM count, driver (NVML), CUDA runtime and driver API versions,
cuFFT version, MathDx version and git commit/dirty flag (baked in by CMake at
configure time), cuFFTDx version (`CUFFTDX_VERSION_*`), UTC timestamp and the
command line, in every JSON line. The commit is stale if one rebuilds without
reconfiguring; the runner always configures a fresh clone.

### SASS check
`scripts/sass_check.sh` counts opcodes in `cuobjdump -sass`, per kernel:
anything containing `MMA` (HMMA, IMMA, ...), `FFMA`, `HFMA2`. These are static
counts, not executed instructions. For the cuFFTDx baseline I expect zero
MMA; it is the reference point for the Monarch kernel's proof.

### Phase 2 run (Colab T4, commit 25961ee)
Raw data: `results/tesla_t4/2026-11-17_cufftdx_block_fft.jsonl`, copied
unchanged from the runner. What I take from it:

- All 14 configurations pass `--check`; none hit the smem limit. N = 16384
  uses exactly 65536 B, the T4 opt-in maximum, so it runs 1 block/SM.
- Batch 2^14: fp32 reaches 212-238 GB/s effective, fp16 166-236 GB/s.
  The T4 spec sheet lists 320 GB/s, so the baseline is at 66-74 % of peak
  DRAM bandwidth, i.e. memory-bound as expected for a single-pass block FFT.
  fp16 is slower than fp32 in GB/s for N <= 1024 (fp16 time is 0.62-0.64x fp32,
  not 0.5x), so halving the bytes does not halve the time at small N.
- fp32 rel L2 1.4e-7 to 2.4e-7. fp16 rel L2 1.2e-3 to 2.5e-3, growing with N.
- SASS (`sass_check.sh`): 0 MMA in every cuFFTDx kernel. fp32 kernels use
  FFMA, fp16 kernels use HFMA2 and no FFMA, i.e. the fp16 butterflies are
  computed in half precision.

The last two points together give a hypothesis for the Monarch kernel, not a
result: the cuFFTDx fp16 error (~2e-3) is about 7x the Phase 1 CPU emulation
of an fp16-operand / fp32-accumulate Monarch FFT (~3e-4), and the HFMA2-only
SASS is a plausible reason. If the GPU Monarch kernel reproduces the
emulation, tensor cores would buy accuracy over cuFFTDx fp16, not only speed.
To be measured in Phase 3.

## Phase 3

### cuBLASDx names and where I checked them (MathDx 26.10, cuBLASDx 0.8.0)
- `Size`, `Precision<PA,PB,PC>`, `Type<type::complex>`, `Function<function::MM>`,
  `Arrangement`, `MaxAlignment`, `Block`, `BlockDim`, `SM`: doc/cublasdx/html/api/operators.html.
- Register API `execute(tensor_a, tensor_b)` returning an accumulator, and
  "call `__syncthreads()` before any thread overwrites A or B" after it:
  api/gemm_methods.html, "Register API".
- Accumulator `get_results`, `is_thread_active`, `is_predicated`,
  `is_index_in_bounds`, `map_fragment_index` returning the (m, n) coordinate:
  api/other_tensors.html, "Accumulator"; implemented in
  include/cublasdx/detail/accumulator/rmem_accumulator.hpp.
- `get_layout_smem_a/b` follow `Arrangement` with default leading dimension;
  row-major A is M×K with LDA = K, row-major B is K×N with LDB = N:
  api/operators.html, Table 3.
- `size_of<BLAS>::{m,n,k}`: include/cublasdx/traits/blas_traits.hpp:108.
  `CUBLASDX_VERSION_{MAJOR,MINOR,PATCH}`: include/cublasdx/cublasdx_version.hpp.
  `CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM_BLOCK`: include/cublasdx/detail/commondx_config.hpp:52.
- `complex<__half>(float re, float im)` rounds with `__float2half_rn`, and
  `.real()/.imag()`: include/commondx/complex_types.hpp:220-240.
- `cublasdx::get<>` is named in the docs but I did not find it exported in the
  headers, so I use `cute::get<>`, which the cuBLASDx headers use themselves
  (include/cublasdx/detail/tensor.hpp:239).
- `BlockDim<X>`: the kernel must be launched with at least X threads, all
  participating and consecutive: api/operators.html, "BlockDim Operator".

### Which instruction each precision gets on sm_75
include/cublasdx/database/gemm/mma_atoms.hpp:150-200 (`get_mma_family`):
- A, B = half and C = float, complex: `SM75_16x8x8_C32C16C16C32_TN`. That is
  `ComplexMMA<SM75_16x8x8_F32F16F16F32_TN>`
  (detail/cute/extension/arch/complex_mma.hpp:169): 4 real 16x8x8 HMMAs with
  fp32 accumulators per complex tile (re·re, im·re, −im·im, re·im).
- A, B, C = float, complex: `mma_family<>`, i.e. no tensor-core atom on sm_75
  (no TF32 on Turing). The fp32 Monarch is an FFMA GEMM; it is there as the
  same-algorithm, same-precision comparison against cuFFTDx fp32.
To be confirmed in SASS: HMMA in the fp16 kernels, none in fp32.

### Legal shapes on sm_75
Size is only bounded by shared memory (requirements page, "Supported Maximal
Sizes"); shapes not divisible by the MMA tile are predicated (other_tensors,
`is_predicated`). n1 = 8 with M = 8 < 16 is in the sweep on purpose to
exercise that path.

### Precision<__half, __half, float>, not Precision<__half>
Phase 1 emulation and the fp16 cuFFTDx result point the same way: the fp16
error is dominated by where rounding happens. half×half→half would accumulate
n1 or n2 terms in fp16 (the requirements page says the same about half C).
The instruction count is the same either way (one HMMA.1688 per
real tile); whether the fp32-accumulate HMMA is slower on T4 I have not measured.

### Batching: stack FFTs along the free GEMM dimension
GEMM 1 is F_n1 (n1×n1) times B1 (n1 × fpb·n2): fpb FFTs side by side as extra
columns. GEMM 2 is A2 (fpb·n1 × n2) times F_n2: FFTs stacked as extra rows.
One GEMM per stage per group instead of fpb small ones; F is shared by all.
fpb = 4096 / N (at least 1), so a group is 4096 complex elements.

### Shared memory layout
[F_n1 | F_n2 | buf], all `complex<T>`. With n1 == n2 the one table is used
for both GEMMs (F_n is symmetric). buf (fpb·N) holds B1, then A2, then the
output in natural order: three uses, separated by `__syncthreads()`. The
intermediate never goes to shared memory in fp32: it comes out of GEMM 1 in
registers, gets the twiddle in fp32, and is rounded once to T when written
as A2. Smem: fp16 64×64 = 16 KB (one F) + 16 KB = 32 KB; fp32 64×64 = 64 KB,
exactly the T4 opt-in limit.
N = 8192 does not fit with two dense factors: 64×128 needs 16 KB + 64 KB for F
alone in fp16. Going further needs three factors or F in registers; out of
MVP scope.

### Table storage precision
- F_n1, F_n2: stored as `complex<T>`. They are tensor-core operands, so for
  fp16 they must be half; the rounding of F is inherent to the method and is
  part of the Phase 1 emulation.
- Twiddles: fp32 complex in global memory for both precisions. They multiply
  the fp32 accumulator before the single rounding to T, so an fp16 table
  would add a second rounding of the same size as the operand rounding, for
  nothing: the table is N·8 B (32 KB at N = 4096) and stays in L1/L2.
- All tables are computed in fp64 with `sincospi` on exact integer phases
  (j·k mod m) and rounded once (derivation.md, "Twiddle and DFT-matrix precision").

### Output precision
The fp16 kernel writes fp16, like the cuFFTDx fp16 baseline, so both move the
same bytes. This adds one rounding that the Phase 1 emulation does not have
(it kept the output in fp32); expect the GPU error to be somewhat above the
emulation for that reason.

### Persistent grid
F is loaded into shared memory once per block, then the block loops over
groups (`g += gridDim.x`), grid = min(groups, occupancy × SMs). Without the
loop each block would re-read n1² + n2² elements from L2 for 4096 elements of
data, i.e. as much table traffic as data traffic at N = 4096.

### Output transpose through shared memory
GEMM 2 results are written to buf at X_f[j1 + n1·j2] and copied out
contiguously, instead of storing fragments straight to global memory with
stride n1.

### Block size
128 threads for both GEMMs (`BlockDim<128>`), chosen once, not tuned. Tuning
BlockDim and smem layouts (swizzled `suggest_layout_smem_*`) is a later step;
the default layouts are used now because the manual indexing above assumes
plain row-major.

### Harness refactor
The accuracy and timing loop moved from block_fft.cu into
`measure()` in harness.cuh so both implementations use the same code
path: same seeds, batches, quantiles and JSON fields. Skipped
configurations are now recorded as `"status":"skipped"` with a reason.

### Phase 3, first GPU run (Colab T4, commit 53cd975)
Raw data: `results/tesla_t4/2026-11-17_monarch_block_fft.jsonl`.
- Correctness: all 14 configurations that fit pass `--check`. fp16 rel L2
  4.2e-4 to 4.6e-4, vs 2.6e-4 to 3.4e-4 in the Phase 1 CPU emulation; the gap
  is in the direction expected from the extra fp16 rounding of the output.
  fp32 rel L2 1.4e-7 to 2.8e-7, same range as cuFFTDx fp32.
  fp32 32×64 and 64×32 skipped (73728 B smem > 65536 B).
- Tensor cores: NOT shown. `sass_check.sh build/monarch_fft monarch` reports
  0 MMA in every fp16 kernel, with 10k-35k instructions and almost no FFMA or
  HFMA2. Since the filter keeps only names containing "monarch", HMMAs in a
  non-inlined device function would not be counted. Next run checks the whole
  binary (raw `grep -c HMMA`), the opcode histogram and register/local use.
  No speed comparison is meaningful until this is settled.
- Speed (recorded, not interpreted yet): fp16 Monarch 15.5-58 GB/s effective
  at batch 2^14 vs 166-236 GB/s for cuFFTDx fp16; occupancy 1-3 blocks of
  128 threads per SM.
