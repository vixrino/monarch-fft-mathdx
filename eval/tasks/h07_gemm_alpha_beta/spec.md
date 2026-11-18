# h07_gemm_alpha_beta: transposed GEMM with alpha, beta, in place on C

Implement `solve` from `task.h` with cuBLASDx, block execution, for a Tesla T4 (sm_75).

- `batch` independent real fp32 updates `C_b = alpha * A_b^T * B_b + beta * C_b`, M = 64, N = 32, K = 48:
  - `A_b`: stored as a 48 x 64 row-major matrix, element (k, i) at `A[b*3072 + k*64 + i]`; the product
    uses its transpose (64 x 48);
  - `B_b`: 48 x 32 row-major, element (k, j) at `B[b*1536 + k*32 + j]`;
  - `C_b`: 64 x 32 row-major, element (i, j) at `C[b*2048 + i*32 + j]`, read and overwritten in place.
- `alpha` and `beta` are arbitrary floats (beta may be 0; C still holds finite values on input).
- `batch` is any integer from 1 to 2^16. Launch on `stream`; do not synchronize. `stream` is a non-blocking
  stream (it does not wait for the legacy default stream).
- Accuracy: relative L2 error ≤ 1e-5 against an FP64 reference on the same inputs.

Build: `nvcc -std=c++17 --expt-relaxed-constexpr -DCUBLASDX_NO_FATBIN_AVAILABLE -O3 -arch=sm_75
-I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include`. Write your code in `solution.cu`.
