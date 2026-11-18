# d2_gemm_batched: batched small fp32 GEMM with mixed layouts

Implement `solve` from `task.h` with cuBLASDx, block execution, for a Tesla T4 (sm_75).

- `batch` independent products `C_b = A_b * B_b`, each 32 x 32 x 32, real fp32.
- Storage, matrix b at offset `b * 1024` in each array:
  - `A_b` row-major: element (i, k) at `A[b*1024 + i*32 + k]`;
  - `B_b` column-major: element (k, j) at `B[b*1024 + k + j*32]`;
  - `C_b` column-major: element (i, j) at `C[b*1024 + i + j*32]`. Overwrite C; its input content is
    undefined.
- `batch` is any integer from 1 to 2^16.
- Launch on `stream`; do not synchronize. `stream` is a non-blocking stream (it does not wait for the legacy default stream).
- Accuracy: relative L2 error ≤ 1e-5 against an FP64 reference on the same inputs.
- Performance: the reference solution computes one product per thread block with the operands staged in
  shared memory; a host loop launching one GEMM per matrix is far too slow.

Build: `nvcc -std=c++17 --expt-relaxed-constexpr -DCUBLASDX_NO_FATBIN_AVAILABLE -O3 -arch=sm_75
-I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include`. Write your code in `solution.cu`.
