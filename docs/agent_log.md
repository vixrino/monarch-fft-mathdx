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
  docs/decisions.md). The GitHub CUDALibrarySamples examples use an older
  thread-indexing API than the one in the package (`working_group`); reading
  the examples from the installed package avoided that mismatch.
- `sass_check.sh` worked on real SASS, but printed demangled names cut before
  the size/precision, so fp16 and fp32 rows were indistinguishable. Fixed by
  extracting N and precision from the name.
