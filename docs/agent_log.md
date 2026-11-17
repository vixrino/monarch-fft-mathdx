# Agent log: MathDx mistakes during development

Each entry: what the agent (Claude Code) got wrong, how it was caught, fix.

## Phase 0
- Initially planned to link `mathdx::cublasdx`. Caught by reading the cuBLASDx
  installation docs before writing CMake: that target needs separable
  compilation + IPO and, on CUDA < 13.2, the `cublasdx_fatbin` variant instead.
  Switched to `cublasdx_no_lto` (GEMM only). Not a build failure, a docs check.
- Assumed the tarball unpacks to `nvidia/mathdx/YY.MM/` as the installation docs
  say. Caught by the first Colab run: the archive has an extra
  `nvidia-mathdx-<ver>/` root. Fixed the glob in the notebook.
- Assumed `mathdx::cublasdx_no_lto` exists in every package because the current
  docs list it. Caught by listing the 25.12.1-cuda12 tarball before the next GPU
  run: that package has no such target. CMake now falls back to
  `mathdx::cublasdx`, which is header-only in 25.12. Lesson: read the cmake
  config of the installed package, not the latest docs.

## Phase 2
- The cuFFTDx baseline compiled and passed on the first GPU run. Every
  operator and trait was looked up in the 26.10 headers before use (list in
  docs/decisions.md). The GitHub CUDALibrarySamples `block_io.hpp` indexes
  with `threadIdx` directly, the packaged one with `FFT::working_group`;
  reading the examples from the installed package avoided mixing the two.
- `sass_check.sh` worked on real SASS, but printed demangled names cut before
  the size/precision, so fp16 and fp32 rows were indistinguishable. Fixed by
  extracting N and precision from the name.

## Phase 3
- Claimed from `get_mma_family` (database/gemm/mma_atoms.hpp) that complex
  half×half→float on sm_75 would use the HMMA atom, and wrote it in
  decisions.md as "to be confirmed in SASS". It was wrong: the tuning
  database lookup in `get_tile_config` runs first and its sm_75 records pick
  `universal_fma`. Caught only by the SASS check (0 HMMA in the binary,
  HMUL2 + FADD in the histogram). Correctness tests passed throughout, so
  without the SASS rule this would have shipped as a "tensor-core FFT".
  Lesson: an instruction choice has to be traced through the whole selection
  path, or measured; reading the last function in the chain is not enough.
- My first `sass_check.sh` call filtered on "monarch", which would also have
  hidden HMMAs in a non-inlined device function. The second run counted the
  whole binary to rule that out before concluding.

- Phase 3: after switching to suggested layouts I assumed every fp16 GEMM
  would reach HMMA. The suggested path only picks the MMA atom when the GEMM
  shape divides warp tile × atom (32 × 16 × 8 for 128 threads on sm_75), so
  GEMM 1 with M = n1 = 16 or 8 stayed on HMUL2/FADD. Caught by the per-kernel
  static MMA counts (64 instead of ~128 for 16×16, with ~3× the instructions)
  and confirmed with the opcode histogram; cause read in
  suggested/suggested_supermma.hpp and suggested/common.hpp.

## Phase 5 (MCP tooling, not MathDx)
- Wrote the server against `mcp` 2.x after reading the installed SDK; most online examples are 1.x
  (`FastMCP`), whose import raises in 2.x. Caught before writing code, by reading the package.
- Assumed the stdio server inherits the caller's environment. The SDK forwards only a safe list without
  LD_LIBRARY_PATH, so on Colab every CUDA binary failed inside the server. Caught by the first GPU smoke
  run: `env_check` failed in 0.0 s while `sass_check` (no driver needed) worked. Worse, `diagnose`
  reported the load failure as an accuracy failure.
- `--only` filtered measurements but not "skipped" records, so `benchmark` returned records without
  timing fields. Caught by the second smoke run (KeyError in the client).
