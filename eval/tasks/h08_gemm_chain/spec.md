# h08_gemm_chain: two chained GEMMs in one block

Implement `solve` from `task.h` with cuBLASDx, block execution, for a Tesla T4 (sm_75).

- `batch` independent products `D_b = (A_b * B_b) * C_b`, all real fp32, 32 x 32, row-major: element (i, j)
  of matrix b at `X[b*1024 + i*32 + j]` for X in A, B, C, D.
- The intermediate `A_b * B_b` stays on chip: one kernel, no intermediate array in global memory.
- `D` aliases none of the inputs. `batch` is any integer from 1 to 2^16. Launch on `stream`; do not
  synchronize. `stream` is a non-blocking stream (it does not wait for the legacy default stream).
- Accuracy: relative L2 error ≤ 1e-5 against an FP64 reference on the same inputs.

Build: `nvcc -std=c++17 --expt-relaxed-constexpr -DCUBLASDX_NO_FATBIN_AVAILABLE -O3 -arch=sm_75
-I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include`. Write your code in `solution.cu`.
