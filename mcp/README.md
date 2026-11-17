# DxBench MCP server

The server runs on the GPU machine and gives an agent six tools and three documentation resources, so it
can check MathDx code itself. The agent never gets a shell: every tool runs a fixed command, and its
arguments are enums or strict regexes checked before anything runs. Each command has a timeout.

| tool | what it runs | returns |
|---|---|---|
| `env_check()` | `scripts/env_check.sh` | GPU, driver, nvcc, MathDx version, ncu counter access |
| `build(target, cuda_arch)` | `cmake` configure + build of one whitelisted target into `build_mcp/` | parsed compiler errors (file, line, message, instantiation context), and doc links for them |
| `run_tests(binary)` | `<binary> --check` | per-configuration rel L2 / max abs and pass/fail; for failures, the worst element (FFT index, k, value, reference) |
| `benchmark(binary, key)` | `<binary> --only <key> --out <tmp>` | the JSON records (median, IQR, GB/s, accuracy, provenance) |
| `sass_check(binary, kernel_filter)` | `scripts/sass_check.sh` | static MMA / FFMA / HFMA2 / total counts per kernel |
| `diagnose(binary, cuda_arch)` | build, then accuracy, then SASS | the first failing stage, what failed, and the NVIDIA doc sections to read |

Resources:
- `mathdx://cublasdx/llms.txt` and `mathdx://cublasdx/markdown/{page}`: NVIDIA's official cuBLASDx agent
  guide, read from the installed MathDx package (`doc/cublasdx/html/`).
- `dxbench://agent_guides/cufftdx`: an **unofficial** cuFFTDx guide written for this repo
  (`agent_guides/cufftdx_agent_guide.md`). NVIDIA publishes no cuFFTDx equivalent as of 2026-11-17.

Run (stdio):

```
pip install -r mcp/requirements.txt
DXB_REPO=$PWD MATHDX_ROOT=/path/to/nvidia/mathdx/26.10 python3 mcp/server.py
```

Claude Code on the same machine:
`claude mcp add dxbench -e DXB_REPO=$PWD -e MATHDX_ROOT=... -e LD_LIBRARY_PATH=$LD_LIBRARY_PATH -- python3 mcp/server.py`.
MCP stdio clients start the server with a reduced environment (the Python SDK forwards PATH, HOME and a few
others). If the CUDA driver library is found through `LD_LIBRARY_PATH`, as on Colab, pass it explicitly, or
`nvidia-smi` and every CUDA binary fail inside the server while `sass_check` still works.

Checks:
- `tests/test_mcp_runner.py` runs without a GPU. It covers argument rejection (shell metacharacters,
  path traversal, keys for the wrong binary), the parsers on real log formats, timeouts, and an
  in-process MCP session.
- `mcp/smoke_client.py` runs on the GPU machine and launches the server over stdio. It calls every tool on
  the clean repo, then runs `diagnose` on three copies with one injected bug each: a compile error, a
  conjugated twiddle, and fp16 GEMMs moved back to cuBLASDx's default layouts. The last one is the mistake
  in docs/agent_log.md: it passes accuracy but has no HMMA.
