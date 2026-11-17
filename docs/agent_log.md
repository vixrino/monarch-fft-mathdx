# Agent log: MathDx mistakes during development

Each entry: what the agent (Claude Code) got wrong, how it was caught, fix.

## Phase 0
- Initially planned to link `mathdx::cublasdx`. Caught by reading the cuBLASDx
  installation docs before writing CMake: that target needs separable
  compilation + IPO and, on CUDA < 13.2, the `cublasdx_fatbin` variant instead.
  Switched to `cublasdx_no_lto` (GEMM only). Not a build failure, a docs check.
