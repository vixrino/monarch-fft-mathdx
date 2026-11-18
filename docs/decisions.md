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

### Phase 3, second run: the fp16 GEMMs do not use tensor cores (commit 7cf55d3)
Evidence: `cuobjdump -sass build/monarch_fft | grep -c HMMA` = 0 for the whole
binary. Opcode histogram of the fp16 64×64 kernel: 16384 HMUL2 + 16384 FADD,
i.e. the complex GEMM runs as scalar half multiplies with fp32 adds.
128-168 registers, no local memory, so it is not a spill problem.

Cause, read in the headers after the run:
include/cublasdx/database/gemm/database_getter.hpp, `get_tile_config`: for a
non-GEMV shape cuBLASDx first looks up the tuning database at
sqrt(M·N) (then rounded to a multiple of 16), and only if no record matches
does it fall back to `get_mma_family` (the HMMA atoms I cited above).
include/cublasdx/database/gemm/sm75/cfp16_cfp16_cfp32_tn.hpp.inc has records
for 16, 32, 48 and 64, and all 24 of them are `mma_atom::universal_fma`; the
real fp16 file (fp16_fp16_fp32_tn) is all `universal_fma` as well. The sm80
version of the same file uses the SM75/SM80 complex HMMA atoms. All my GEMMs
have M·N = 4096, so sqrt = 64 hits a record and gets FMA.

So my earlier entry "half/float complex on sm_75 selects
SM75_16x8x8_C32C16C16C32_TN" was wrong: that is only the fallback.
The performance-guide card (doc/cublasdx/html/markdown/performance_guide_api_supermma.md)
says half inputs with a float accumulator use tensor cores from SM 75 for any
shape except M, N or K = 1; for the default layout path on sm_75 that does
not match what the headers do. I don't know whether the database choice is a
deliberate tuning result for T4 or an artifact; I only know what is selected.

Next step: `cuda/monarch/mma_probe.cu`, one 64×64×64 complex fp16 GEMM in
three variants that differ only in what selects the instruction:
0. default layouts and accumulator (what monarch_fft does now),
1. `suggest_layout_smem_*` + `suggest_accumulator` (the documented
   "supermma" path; database/gemm/suggested/suggested_supermma.hpp lists
   SM75_16x8x8_C32C16C16C32_TN for this precision),
2. `experimental::Tile<cublasdx::detail::SM75_16x8x8_C32C16C16C32_TN, 2, 2>`
   (include/cublasdx/operators/experimental/tile.hpp; overrides the database
   in `get_database_threads`). This uses an internal `detail` type, so it is
   the last resort.
Each variant is checked for an exact result on small-integer inputs, and
`sass_check.sh` on the binary shows which one emits HMMA. The Monarch kernel
will then use the first variant that does.

### mma_probe result and the fix (Colab T4, commit d4e0e83)
`./build/mma_probe` + `sass_check.sh build/mma_probe probe`, one 64×64×64
complex fp16→fp32 GEMM, 128 threads, all three exact on small-integer inputs:

| variant                                   | static MMA | total SASS |
|-------------------------------------------|-----------:|-----------:|
| 0 plain `get_layout_smem_*` + `get_accumulator` |      0 |      17216 |
| 1 `suggest_layout_smem_*` + `suggest_accumulator` | 256 |        872 |
| 2 `experimental::Tile<SM75_..C32C16C16C32_TN, 2, 2>` | 256 |      840 |

So on sm_75 the suggested layouts/accumulator route the GEMM to HMMA while
the plain ones take the database's `universal_fma`. I use variant 1: it is
the documented API, variant 2 depends on a `detail` type for 8 % fewer
static instructions. Counts are static; a K loop that is not unrolled
executes its HMMAs more than once, so 256 is not the executed count.

Consequences in monarch_fft.cu (fp16 only; fp32 keeps plain layouts since it
has no tensor-core path on sm_75 and the plain layouts let n1 == n2 share F):
- Every write into B1, A2 and the F tables goes through tensor coordinates
  `t(row, col)` instead of a raw index, because the suggested layouts are
  swizzled. The output staging in buf stays raw: it is my layout, written
  after the last GEMM read.
- F_n1 (A role of GEMM 1) and F_n2 (B role of GEMM 2) can no longer share a
  buffer even when n1 == n2: their suggested layouts differ. fp16 64×64 smem
  goes from 32 KB to 48 KB.

### Phase 3, fourth run: fp16 Monarch on HMMA (Colab T4, commit 2e5eade)
Raw data: results/tesla_t4/2026-11-17_monarch_block_fft_4.jsonl.
- ctest: cufftdx_accuracy and monarch_accuracy pass; every fp16 and fp32
  size passes `--check`. fp32 32×64 and 64×32 are still skipped (73728 B smem).
- SASS: 2048 HMMA lines in the binary; every fp16 `monarch_fft` instance has
  MMA > 0 (64 to 512 static), every fp32 instance has 0 MMA and runs on FFMA,
  as expected on sm_75. cuFFTDx kernels: 0 MMA, HFMA2/FFMA only.
- Accuracy (median rel L2 over 5 seeds × 64 FFTs): fp16 3.7-4.0e-4, the same
  range as the FMA path (4.2-4.6e-4) and below cuFFTDx fp16 (1.2-2.5e-3).
  fp32 1.4-2.8e-7.
- Speed at batch 2^14 (recorded, not interpreted yet): fp16 72-145 GB/s
  effective, up from 15-78 on the FMA path; cuFFTDx fp16 is 166-236.
- Open: the 16×16 and 16×32 fp16 kernels have ~6500 static instructions vs
  ~2100-2800 for the other shapes, with only 64/128 MMA. One of their two
  GEMMs (a K = 16 one with a 16-wide side) may still be on a non-MMA path.
  Not checked yet: needs the per-kernel opcode histogram.

### Why 8×32, 16×16 and 16×32 fp16 still had an FMA GEMM, and the transpose of GEMM 1
Evidence (Colab T4, commit 2e5eade): `OPS=1 sass_check.sh` on the fp16 16×16
kernel shows 2048 HMUL2 + 2048 FADD next to its 64 HMMA, so one of its two
GEMMs is on the scalar path.

Cause, in the headers: database/gemm/suggested/suggested_supermma.hpp,
`get_best_supermma_tiled_instruction`, tiles the warps with
`get_floor_tile(num_warps)` (suggested/common.hpp), which is (2, 2) for
128 threads. `search_mma_list` (common.hpp) then keeps an MMA atom only if
the GEMM shape is evenly divisible by warp tile × atom shape. For
SM75_16x8x8_C32C16C16C32_TN that is 32 × 16 × 8, so M % 32, N % 16 and
K % 8 must all hold; otherwise the instruction is void and the GEMM falls
back to FMA. GEMM 1 was F_n1 @ B1 with M = n1, so n1 = 8 and n1 = 16 failed
on M. GEMM 2 always had M = fpb·n1 ≥ 64 and passed.

Fix: compute GEMM 1 as A1 @ F_n1 with A1 = (fpb·n2 × n1), the transpose of
the old B1. Valid because F_n1 is symmetric, so (F_n1 @ B1)ᵀ = B1ᵀ @ F_n1.
Both GEMMs now put the batch on M and the DFT size on N = K. Then
- n1 = 16 passes (M = fpb·n2 ≥ 64, N = K = 16);
- n1 = 8 still fails on N % 16. I keep 8×32 in the sweep as the
  documented case where GEMM 1 stays on FMA; 16×16 covers N = 256 on HMMA.
Alternatives: 64 threads (warp tile (2, 1), so N % 8 would pass for n1 = 8),
but that halves the threads for every size; forcing the atom with
`experimental::Tile` uses a `detail` type. Neither is worth it for one shape.
Cost: none in smem; the load and epilogue index maps change, the GEMMs are the
same size.

### Phase 3, fifth run: transposed GEMM 1 (Colab T4, commit 14bbef2)
Raw data: results/tesla_t4/2026-11-17_monarch_block_fft_5.jsonl.
- ctest passes; all 14 runnable sizes pass `--check` (fp32 32×64 / 64×32
  still skipped for smem).
- SASS: 2176 HMMA in the binary. fp16 16×16 went from 6576 static
  instructions / 64 MMA to 2352 / 128, 16×32 from 6520 / 128 to 2264 / 192;
  they now match the other shapes. 8×32 is 3944 / 128 (GEMM 1 still FMA, as
  predicted) and uses 254 registers.
- fp16 rel L2 3.7-3.8e-4, unchanged within noise.
- Speed at batch 2^14, fp16: 16×16 174 GB/s (was 80), 16×32 173 (was 72),
  32×16 170, 32×32 146, 32×64 / 64×32 100, 64×64 87. cuFFTDx fp16 from the
  Phase 2 run: 166 (N=256), 187 (512), 183 (1024), 220 (2048), 230 (4096).
  These come from different Colab sessions, one run each; I don't compare
  them closely until the sweep runs both binaries in the same session.

## Phase 4

### One sweep file per session
`scripts/sweep.py` runs both binaries with the same `--out` file, then
appends one `"experiment":"sass"` line per kernel parsed from
`sass_check.sh`. The file name gets a `_k` suffix instead of being appended
to. `plot.py` uses, by default, only the newest sweep file of each GPU
directory. So cuFFTDx and Monarch numbers in a figure always come from one
Colab session: the earlier per-binary runs differ by a few percent between
sessions. Alternative: plot every
file and average. I rejected that because it mixes sessions with different
clocks and commits.

### What the figures show
- Throughput: the batch-2^14 rows. For Monarch the line joins the fastest
  factorization per N, and the others are hollow markers labeled n1xn2.
  Error bars are the IQR of the 50 launches, converted to GB/s.
- Error: same layout. The line joins the most accurate factorization.
- Scatter: one point per (impl, precision, N), with Monarch at its fastest
  factorization so the point matches the throughput line.
- Latency is a table, not a figure: the two implementations put different
  numbers of FFTs in one block, so the comparison is per launch, not per FFT.
- Colors: categorical slots 1 and 2 of the dataviz reference palette,
  validated with its CVD script (protan ΔE 24.7). Marker shape also encodes
  the implementation, so the figures read without color.

Correction to the entry above: I first justified it with "Monarch numbers of
the same commit moved by up to 10 % between runs 4 and 5", but runs 4 and 5
are different kernels (2e5eade vs 14bbef2). The real same-kernel comparison
is run 5 vs the first sweep: the monarch kernel source did not change between
14bbef2 and d481e92, and fp16 16×16 went from 174.3 to 192.8 GB/s (+11 %),
32×32 from 145.9 to 165.6 (+13 %). cuFFTDx fp16 N=256 went from 165.6
(Phase 2 run) to 162.4. So session-to-session variation on Colab is up to
~13 % here, which is why figures only use one sweep file.

### First sweep (Colab T4, commit d481e92)
Raw data: results/tesla_t4/2026-11-17_sweep.jsonl. Tables and figures in
docs/results.md and docs/figures/tesla_t4/ are generated from it by
`scripts/plot.py`. ctest passed, every size passed `--check`, and the SASS
table matches the per-binary runs (fp16 Monarch MMA > 0, all else 0).
Plot fix after seeing it: a skipped N (fp32 Monarch 2048) now breaks the line
instead of being drawn through.

## Phase 4b: Nsight Compute

### Profiling one configuration
Both binaries take `--only <key>` (`monarch:fp16:4096:64x64`,
`cufftdx:fp16:256`, ...) and `--profile`, which skips accuracy and timing
and launches the FFT kernel once at the throughput batch (~2^14 FFTs).
`scripts/profile.sh` runs `ncu --set full -k regex:'monarch_fft|block_fft'
--launch-count 1` on that. The kernel filter drops the table, fill and cuFFT
reference kernels, so the one profiled launch is the right one without
counting launches. I rejected `--launch-skip` on the normal run (5 accuracy +
55 + 55 timing launches): it silently profiles the wrong launch if the
harness changes.
The raw page of every metric goes to `results/<gpu>/<date>_ncu_<key>.csv`,
unedited. `.ncu-rep` files are git-ignored because they are large binaries;
the CSV holds the same metric values. `scripts/ncu_summary.py` prints a fixed
set of metrics by name pattern (time, SOL, occupancy and its limiters, pipe
utilization including tensor, shared-memory bank conflicts, stall reasons),
so I don't have to guess which names the Turing ncu exports.
Configurations profiled: Monarch fp16 64×64 (0.39× cuFFTDx) and 16×16
(1.19×), plus cuFFTDx fp16 at N = 4096 and 256 as references.

### Nsight Compute run (Colab T4, commit ea51822)
Raw data: results/tesla_t4/2026-11-17_ncu_{cufftdx_fp16_256,cufftdx_fp16_4096,
monarch_fp16_256_16x16,monarch_fp16_4096_64x64}.csv, plus a second sweep
results/tesla_t4/2026-11-17_sweep_2.jsonl (same kernels; batch-2^14
throughputs differ from the first sweep by -1.2 % to +6.6 %, and the fp16
Monarch/cuFFTDx ratios by at most 0.02, e.g. 1.19 -> 1.21 at N = 256). The summary
table is in docs/results.md.

What the profile says about Monarch fp16 64×64:
- 1 block of 128 threads per SM: `launch__occupancy_limit_shared_mem` = 1
  for 48 KB per block. Registers would allow 4.
- 4.0 warps active per SM, 19 % issue active.
- Shared-memory wavefronts and bank conflicts dwarf cuFFTDx's (loads 44.6M
  with 25.7M conflicts; stores 34.2M with 27.9M).
- The tensor pipe is active 26 %. Memory throughput is 38 % vs 85 % for
  cuFFTDx, with the same DRAM bytes.

ncu locks the SM clock at base (~585 MHz), so cycle counts across kernels with
different compute/memory balance do not reproduce the timed ranking. I use the
profile for utilizations and ratios only.

Next step this points to (not done): since GEMM 1 was transposed, g1 and g2
are the same cuBLASDx type when n1 == n2 (both `gemm<T, fpb·n, n, n>`). So the
suggested B layouts are identical too, and one F table can serve both GEMMs in
fp16, as it already does in fp32. For 64×64 that cuts shared memory from
48 KB to 32 KB, which allows 2 blocks per SM (64 KB per SM on T4, registers
allow 4). After that, look at the bank conflicts in the epilogue stores
(writes through the swizzled A2 layout in fragment order) and in the
transposed output write `buf[f*n + j1 + n1*j2]` (stride n1).

### Share F in fp16 when n1 == n2
`share_f = N1 == N2`, no longer `&& !suggest`. Since GEMM 1 was transposed,
g1 = gemm<T, fpb·n2, n1, n1> and g2 = gemm<T, fpb·n1, n2, n2> are the same
type when n1 == n2, so `suggest_layout_smem_b()` is the same layout for both.
A `static_assert(std::is_same_v<g1, g2>)` guards this. Expected shared memory
for fp16: 16×16 18432 → 17408 B, 32×32 24576 → 20480 B, 64×64 49152 →
32768 B. fp32 is unchanged. Whether 64×64 then gets 2 blocks per SM is
for the run to show (blocks_per_sm in the sweep, occupancy limit in ncu).

### Shared-F run (Colab T4, commit 38e465c)
Raw data: results/tesla_t4/2026-11-17_sweep_3.jsonl and
2026-11-17_ncu_*_2.csv. ctest passed and every size passed `--check`, with
accuracy identical to the previous sweep. Shared memory came out exactly as
predicted (17408 / 20480 / 32768 B).
- 32×32: 2 → 3 blocks per SM, 164.2 → 184.1 GB/s (vs sweep_2), now tied
  with cuFFTDx (1.01×, IQRs overlap).
- 64×64: 1 → 2 blocks per SM, 92.5 → 103.9 GB/s.
- 16×16: 3 blocks per SM before and after (registers allow 4 and shared
  memory 3), 195.0 → 196.4 GB/s, which is noise.
- ncu, 64×64: warps active 4.0 → 7.6 per SM and SM cycles 6517k → 5371k
  (−18 %, same base clock). Instructions and shared-memory wavefronts and
  conflicts are unchanged.
Remaining gap at N ≥ 2048 (0.45-0.48×): the instruction count and the
data-path bank conflicts. Next candidates, in order:
1. bank conflicts in the A2 epilogue writes and the transposed output write;
2. 256 threads per block, to get more warps without more shared memory.

## Phase 5: MCP server

### What NVIDIA publishes for agents (checked 2026-11-17)
- cuBLASDx: `llms.txt` plus the Agent BLAS Performance Guide in `markdown/`. Both are online
  (docs.nvidia.com/cuda/cublasdx/llms.txt, HTTP 200) and in the package at `doc/cublasdx/html/`. The two
  llms.txt files are the same size (4911 B).
- cuFFTDx: none. `docs.nvidia.com/cuda/cufftdx/llms.txt` returns 404, the package has no `doc/cufftdx`,
  and the HTML table of contents has no agent page. So I wrote
  `agent_guides/cufftdx_agent_guide.md`, labeled unofficial at the top, in the cuBLASDx guide's style.
  Every operator, trait and method in it was checked in the 26.10 headers, with the header path cited,
  and every performance claim is either quoted from NVIDIA's performance page or measured here.
- cuSolverDx and a MathDx-level llms.txt: 404 too.

### Server design
- **SDK:** `mcp` 2.3.0 (`mcp.server.mcpserver.MCPServer`, stdio). I read the installed package before
  writing: in 2.x, `FastMCP` is renamed and importing `mcp.server.fastmcp` raises. Most examples online
  are 1.x, so `mcp/requirements.txt` pins `>=2.3,<3`.
- **No shell.**
  - `mcp/runner.py` builds every command as an argv list, with `shell=False`.
  - Arguments are `Literal` enums in the tool signatures and are re-checked in the runner; free-form
    strings (benchmark key, SASS kernel filter) must match strict regexes.
  - The repo and MathDx paths come from the server's environment, never from the client.
  - Each command has a timeout (2–15 min), and returned logs are clipped at 20k characters.
- **`diagnose` order:** build, then accuracy, then SASS. It stops at the first failure, because an agent
  should fix one thing at a time. The SASS stage exists because of docs/agent_log.md: twice, the code was
  numerically right and still not on tensor cores.
- **Failing indices:** the harness now prints one `MISMATCH key=... fft=... k=... got=(...) ref=(...)` line
  per failing configuration. It gives the worst element of seed 0, so an agent sees which FFT and which
  frequency are wrong, not just a norm.
- **Doc links** (`mcp/doc_links.py`) map compiler-message patterns and failure kinds to NVIDIA pages.
  All 11 URLs returned HTTP 200 when written; anchors were checked in the shipped markdown.
- **Build directory:** `build_mcp/` inside the repo the server points at, so it does not disturb `build/`.
- **Testing:**
  - Locally without a GPU: `tests/test_mcp_runner.py` (25 tests: rejection, parsers on nvcc/gcc/CMake
    formats, a real timeout, an in-process session, a path-traversal attempt on the markdown resource).
  - On the GPU: `mcp/smoke_client.py` over stdio with three injected bugs. Not run yet.

### First GPU run of the MCP smoke test (Colab T4, commit 6451672)
Failed, for two reasons in my code, not in the kernels:
- **Environment.** The SDK's stdio client starts the server with a reduced environment
  (`DEFAULT_INHERITED_ENV_VARS` in mcp/client/stdio.py: PATH, HOME, ...). `LD_LIBRARY_PATH` is not in
  it, and Colab finds the CUDA driver library through it. So `env_check` failed in 0.0 s, every CUDA
  binary failed at load (`diagnose` reported stage "accuracy" on the clean repo, `benchmark` returned no
  records), and only `sass_check` worked, since cuobjdump needs no driver. The smoke client now forwards
  LD_LIBRARY_PATH and the CUDA variables, and mcp/README.md says to pass `-e LD_LIBRARY_PATH` when
  registering the server.
  The `diagnose` misreport is a lesson for the tool itself: a binary that cannot start looked like an
  accuracy failure. The accuracy stage already returns stderr when no mismatch line was parsed, but the
  smoke client did not print it; it does now.
- **Unbuilt binary.** `run_tests` on a binary that was never built raised FileNotFoundError inside the
  tool. It is now a clean rejection that says which `build` call to make.

### MCP smoke test passes (Colab T4, commit 7f792a2)
`python3 mcp/smoke_client.py`, server over stdio, output pasted from the notebook (not stored in results/,
since it is a functional test, not a measurement):
- clean repo:
  - `env_check` ok (4.4 s);
  - the shell string is rejected;
  - `diagnose(monarch_fft)` reaches stage "ok" (build, all accuracy checks, all fp16 kernels with MMA);
  - `sass_check` finds 8 fp16 kernels with MMA;
  - `benchmark(monarch:fp16:256:16x16)` returns the two records;
  - `run_tests(cufftdx_block_fft)` before building is a clean rejection, and after `build` (38 s) it
    passes 14 of 14 configurations.
- injected compile error: stage "build", file and line 114, `identifier "a3" is undefined`. No doc link:
  none of the patterns match a plain C++ error, which is right, since no MathDx page is relevant. The
  instantiation context was cut to "detected during:" because nvcc prints an indented list after it; the
  parser now keeps the indented lines.
- injected conjugated twiddle: stage "accuracy", every fp16 configuration fails with rel L2 ≈ 1.41 (√2,
  i.e. an error as large as the signal). The worst element shows the bug's signature directly:
  `got = (-10.273, -35.531)`, `ref = (-10.275, 35.521)`, i.e. the imaginary part has the wrong sign.
- injected default layouts in fp16: stage "tensor_cores". Accuracy passes and all 8 fp16 kernels are
  listed without MMA, with links to the supermma card and the porting guide's instruction check. This
  is the failure from docs/agent_log.md, now caught by one tool call.
The benchmark tool's 163.8 GB/s for fp16 16×16 differs from the sweeps (195–197 GB/s). It is one
configuration in a fresh process right after a build in the same session, and I have not looked into it;
the tool returns the record, it is not a measurement campaign.

## Phase 6: agent evaluation (design)

### Conditions
- **A:** write/read files plus `compile`.
- **B:** A plus official docs and the installed headers.
- **C:** B plus the MCP tools.
- **Model and budgets:** the same for all three, set at run time.

Alternative considered: give A and B a public-test runner too. Rejected because most of the MCP server's
value is executing things (accuracy with failing indices, SASS); with a runner in every arm, C − B would
measure only the doc links and the SASS check.

### Build command for tasks
No CMake per task. One nvcc line derived from the installed cuBLASDx config (`cublasdx-targets.cmake`,
`cublasdx-config.cmake`, target `cublasdx_no_lto`):
- flags: `-std=c++17 --expt-relaxed-constexpr -DCUBLASDX_NO_FATBIN_AVAILABLE -O3 -DNDEBUG -arch=sm_75`;
- includes: `$MATHDX_ROOT/include` and `$MATHDX_ROOT/external/cutlass/include`;
- link: `-lcufft` (hidden tests only, for the FP64 reference).

### Tasks
15 tasks: 5 dev, 10 held-out (table in eval/README.md). None copies an official example or this repo's
kernels. Some deliberately sit next to official examples (inverse FFT, fp16 I/O), because there a
remembered example is most likely to be confidently wrong. Two (d4, h06) require tensor cores and check
the SASS: the trap from docs/agent_log.md.

### Model under test: Mistral Medium
Lucas has a Mistral API key; `mistral-medium-latest` (Mistral Medium 3.5) is on Mistral's function-calling
list (docs.mistral.ai/capabilities/function_calling, checked 2026-11-17). The SDK is `mistralai` 3.7.0: the
import is `from mistralai.client import Mistral`, not the older `from mistralai import Mistral`, which
raises ImportError in 3.x. `chat.complete` takes `tools`, `tool_choice`, `parallel_tool_calls`,
`temperature` and `random_seed`.
A side benefit: the tasks and reference solutions were written with Claude, so not evaluating Claude
avoids grading a model on problems its own family wrote. The results describe Mistral Medium only, and
the write-up must not generalize them to "LLM agents".

### Model under test, revised: Qwen3 8B, local, through ollama (replaces Mistral Medium)
Lucas lost access to the Mistral API. I looked for a free API that can run the eval (45 agent runs of up to
16 turns, contexts up to ~24k tokens), from third-party summaries dated 2026, not from the providers' own pages:
- Groq free: about 8k tokens per minute per model, so one condition-B turn with docs in context can exceed it.
- Cerebras: the free tier became a $5 trial behind a card.
- Gemini Flash free: about 100 requests per day, so roughly a week of runs.

Chosen instead: `qwen3:8b` served by ollama on the same Colab T4. It is free, has no quota, and the model is
pinned by its digest, which every record stores. `eval/probe_api.py` checked one tool-call round trip on the
T4 on 2026-11-18: correct `compile` call with the right argument, ~40 tok/s generation, 131 s for the first
call (model load). The cost: an 8B model is much weaker than Mistral Medium, so all three conditions may fail
most tasks. That would be a result about this model, reported as such. The tasks and references were still
written with Claude, which is still not the model under test.

Settings, identical in A/B/C: temperature 0, seed 0, `think=False` (Qwen3's thinking mode would multiply
output tokens at 40 tok/s), num_ctx 32768, 4096 output tokens per turn, 16 turns, stop when a prompt reaches
24k tokens, tool results clipped to 6000 characters.

## Phase 6: harness

### Tools
- **Fewer, simpler tools.** `write_solution(code)` replaces write_file/read_file: there is one file, and an 8B
  model handles fewer tools better (assumption, not measured). The spec and task.h go in the first message.
- **B corpus = the installed MathDx package:** `doc/` (only cuBLASDx ships docs: HTML stripped to text,
  llms.txt, the markdown agent guide), `include/` without `cufftdx/database` (100 MB of generated kernel
  tables, no prose) and `example/`.
  - cuFFTDx has no prose docs in the package, so for cuFFTDx tasks B means headers plus examples. I did not
    add the online cuFFTDx HTML, to keep the corpus "what is on disk after installing MathDx".
  - The official examples count as documentation: NVIDIA ships them as such, and an agent with a shell would
    read them.
- **No unofficial guide anywhere.** `agent_guides/cufftdx_agent_guide.md` is in no condition, so C − B
  measures the tool loop, not my guide. It could become a fourth condition later.
- **Local doc pointers.** C's `diagnose` returns local `read_doc` paths. It reuses mcp/doc_links.py and keeps
  only the cuBLASDx entries, since the cuFFTDx ones point at online pages that are not in the corpus.

### Grading
- **Reference timing in the same session.** The harness re-validates each task's reference before the agent
  runs, and `perf_vs_ref` = reference time / agent time, so above 1 means faster than mine.
- **api_misuse** = number of distinct compiler error messages that mention cufftdx/cublasdx/commondx (in the
  message, its context or the file path), over every build in the run, plus 1 when a tensor-core task
  compiles without HMMA.
- **Tokens:** as ollama reports them (prompt_eval_count, eval_count), summed over turns. With ollama's
  prompt cache these may count only re-evaluated tokens; the probe's second call reported 220 prompt tokens
  for a ~220-token context, so it looked like full counts there. Not checked on long runs.

### Dev run 1 (2026-11-18, d1 only, qwen3:8b): what it changed in the harness
Data: `results/tesla_t4/2026-11-18_eval_runs.jsonl` and `..._eval_transcripts.jsonl`.
- **Outcome.** All three conditions ended with a build error on the first include line, a header that does
  not exist: A `cufftDx.h`, B `cuda_mathdx.h`, C `MathDx/cuda_mathdx.h`.
  - B and C searched for the invented name (0 hits) and kept it.
  - C never called run_public_test or diagnose.
  - Wall time was 231 s (A, including the model load), 101 s (B) and 87 s (C).
- **Harness bug 1, code fences.** Every write_solution call wrapped the file in a markdown fence ```` ```cpp ````.
  nvcc stopped at the missing header before reaching it, but a correct solution would have failed on line 1.
  write_solution now strips one surrounding fence, in every condition, and says so in its result. It is a
  chat-format slip, not MathDx knowledge.
- **Harness bug 2, case-sensitive misuse count.** The api_misuse regex was case-sensitive and did not include
  "mathdx", so all three runs reported misuse=0 although every build error was an invented MathDx header.
  It is now case-insensitive and includes mathdx.
- **Spec unchanged on purpose.** I did not add the header name (`cufftdx.hpp`) to the specs. Knowing it is
  part of what the eval measures, and B/C can find it with one search.
- **Note on the corpus.** The Colab package has 855 corpus files against 490 in my local copy: it also ships
  docs for other Dx libraries (cuRANDDx seen in C's search hits). The corpus is still "the installed package".

### Pre-run review by three subagents (2026-11-18), before dev run 2
Three reviewer agents read the harness, the grading and the five tasks against the ollama 0.6.3 source and the
dev-run-1 transcripts. What I fixed:
- **Spoofable result.** The solution runs in the same process as the test, so anything it printed last (an
  atexit handler, a destructor) could be read as the pass line. The test now writes its JSON line to the file
  in `$EVAL_RESULT`, and the harness reads only that file. A pass followed by a nonzero exit becomes
  `crash_after_pass`.
- **Library use not checked.** A plain CUDA kernel, or host cuFFT (the test links `-lcufft` for its own
  reference), passed d1, d3 and d5. `library_check` greps solution.cu for the required `.hpp` include and for
  host cuFFT/cuBLAS includes or calls. Records keep `correct` (numbers, plus HMMA where required) and add
  `library_used` and `solved = correct and library_used`.
- **`stream` never exercised.** The tests passed 0, so a solution that ignored the stream was timed correctly
  by accident. Tests now pass a `cudaStreamNonBlocking` stream and record the timing events on it. `dev::up`
  syncs the device, because a pageable cudaMemcpy can return before the copy lands and a non-blocking stream
  does not wait for it. The specs now say the stream is non-blocking.
- **Timed batch unchecked.** A solve that skipped large batches would have gotten a huge perf_vs_ref. Each
  hidden test now also checks batch 2^14, the timed size, with one extra seed.
- **Public test.** It now uses batch 7 (odd, and not a multiple of the FFTs per block at these sizes), not 4 or
  2, so it can expose a tail block or an unpaired fp16 FFT. No hidden test uses 7.
- **Link errors invisible.** compile() built with -c only, so a missing or static `solve`, or a second `main`,
  passed. It now links solution.cu against a stub main that takes `&solve`.
- **Crashes that would end the sweep.** An `AttributeError` from a tool, an ollama `ResponseError` or
  connection error, a `TimeoutExpired` in the reference validation, or a malformed JSON line. Tool calls now
  catch all exceptions. A failed chat call ends the run with `end="error: ..."` and still grades it. The client
  has a 900 s timeout (the default is none), and validate.sh returns on timeout.
- **submit next to a write.** submit now runs after the other calls in the same reply.
- **Misuse inflated in C.** C builds more often (public test, SASS, diagnose), so its misuse count grew from
  extra attempts. Misuse is now counted only from compile() and the final grade, which every condition runs.
  It looks at the message and the file only, not the echoed source line.
- **Wasted turns.** Tools ignore unexpected arguments: in run 1 the model passed `compile(code=...)` twice.
- **Hidden path leak.** Absolute paths in tool output named `hidden/test.cu`; errors now carry basenames.

Known gaps I am keeping, reported as limitations:
- **d4 tensor cores.** HMMA is counted over the whole binary, so a never-launched HMMA kernel, or hand-written
  WMMA next to a cublasdx include, would pass. Closing this needs runtime tensor-pipe counters (ncu) per run.
- **d3 fp16.** Nothing enforces fp16 compute: an fp32 FFT also meets 1e-2.
- **C's prompt is longer.** C's six extra tool schemas lengthen every prompt, so it reaches CTX_STOP a little
  earlier. The records keep last_prompt_tokens.

## Final session plan (2026-11-18)

### Fair baseline: cuFFTDx tuning grid
- **The criticism.** Comparing a tuned Monarch against cuFFTDx's suggested configuration is unfair: Monarch
  got a persistent grid, a shared F and a transposed GEMM 1.
- **The grid.** `cufftdx_grid_<N>` binaries (block_fft.cu with `DXB_GRID_N`) run every ElementsPerThread that
  cuFFTDx's sm_75 database lists for that size (`include/cufftdx/database/records/750/database_*_fwd.json`)
  against FFTsPerBlock in {1, 2, 4, …, 32}. They keep only what the library's static_asserts in
  `traits/detail/check_and_get_trait.hpp` allow (EPT divides N, at most 1024 threads, even FPB for fp16).
  Configurations that do not fit in shared memory are skipped at run time and recorded as skipped. Records
  carry `variant: grid`, and the suggested configuration carries `variant: default`.

### Selection and repeats
- **Protocol.** The sweep runs 3 times in one session (`sweep.py --grid --repeat 3`, 3 files). For both
  implementations, the best configuration (cuFFTDx EPT × FPB, Monarch factorization) is chosen on repeat 1
  only, by throughput at the 2^14 batch among the configs that pass accuracy. The ratios are then reported
  from repeats 2 and 3 with that choice fixed. This keeps the selection from fitting the noise it is then
  measured on.
- **Ties.** A ratio counts as a tie when its range over repeats 2 and 3 contains 1.0.
- **Fallback.** If repeats 2 and 3 are missing, best-of on the median of the available repeats for both
  sides, stated as such in results.md.

### Error decomposition
Each accuracy record now also has `rel_l2_compute_median`. It is measured against an FP64 reference
computed from the input after rounding to the kernel's I/O type (k_widen of the same buffer, then cuFFT
Z2Z). The existing `rel_l2_median`, against the exact fp64 input, still includes the input quantization,
which no fp16 FFT can undo.

### Held-out tasks
- **Written.** h01–h08, from the task table in eval/README.md, which was fixed before dev run 1. The specs
  follow the dev specs' format, including the non-blocking-stream sentence that the pre-run review added to
  every task. No held-out task was changed in response to a dev transcript.
- **Not written tonight:** h09_fft_then_gemm and h10_gemm_then_fft. They are dropped from all three
  conditions.
- **Validation.** Every held-out reference is validated on the GPU (`validate.py`, then again inside the
  harness) before any agent run. The harness skips a task whose reference fails, in all three conditions, and
  records `skipped_invalid_reference`.
- **Unchecked reference choices**, each of which only the GPU validation will confirm:
  - h02 and h05 pin the inverse FFT to the forward's EPT and FPB and static_assert the same stride, storage
    and block size, so the spectrum can be multiplied in registers.
  - h06 relies on the same suggested-layout HMMA path as d4 and mma_probe; 32×64×32 divides the 2×2 warp
    tile.
  - h08 re-stages the first product into the A operand through a plain smem buffer.

## Final session results (2026-11-18, sweeps 3-5, commit a8f89ec)

### What the tuned baseline changed
- **The N ≤ 512 speed claim is gone.** With EPT 2, cuFFTDx fp16 runs 1.46× (N = 256) and 1.28× (512, 1024)
  faster than its default configuration. Monarch is then 0.79-0.83× of it at N ≤ 1024, every range below 1.0.
  The README and results.md now lead with accuracy, and the speed result is reported as a loss.
- **The accuracy claim survives tuning.** EPT also changes cuFFTDx's fp16 error, monotonically: lower EPT, lower
  error, at every N in the grid. So "tuned" in speed is not "tuned" in accuracy, and I report both. `final.py`
  has a third cuFFTDx role, "accurate": the lowest total error anywhere in the grid. Monarch still has
  2.1-3.0× less error than it (2.4-3.4× compute only). I can't explain the EPT dependence from what I've read
  of cuFFTDx, and I say so in results.md.
- **Protocol outcome.** No selection had to be revisited. Three tuned/default ranges sit partly or fully below
  1.0 (fp16 4096, fp32 4096, fp32 8192): the config that won repeat 1 lost its lead in repeats 2-3. I kept the
  repeat-1 selection, since re-selecting on repeats 2-3 would undo the point of the protocol.

### plot.py keeps the default configuration
`plot.py` drops `variant == "grid"` rows, so its per-configuration, latency and SASS tables stay about the
library's default cuFFTDx and the binaries that were SASS-checked. Without this, its "best cuFFTDx" would
silently become a grid row with no SASS record.
