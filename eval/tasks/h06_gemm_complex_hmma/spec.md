# h06_gemm_complex_hmma: complex fp16 GEMM on tensor cores

Implement `solve` from `task.h` with cuBLASDx, block execution, for a Tesla T4 (sm_75).

- `batch` independent complex products `C_b = A_b * B_b` with M = 32, N = 64, K = 32:
  - `A_b`: 32 x 32, complex half, row-major: element (i, k) is the `__half2` (low = real, high = imaginary)
    at `A[b*1024 + i*32 + k]`;
  - `B_b`: 32 x 64, complex half, row-major: element (k, j) at `B[b*2048 + k*64 + j]`;
  - `C_b`: 32 x 64, complex float, row-major: element (i, j) at `C[b*2048 + i*64 + j]` (`.x` real, `.y` imaginary).
- The products must accumulate in fp32 (inputs fp16, accumulator fp32).
- **The GEMM must run on the tensor cores.** The test inspects the SASS of your solution and fails it if
  no `HMMA` instruction is present, even when the numbers are right.
- `batch` is any integer from 1 to 2^16. Launch on `stream`; do not synchronize. `stream` is a non-blocking
  stream (it does not wait for the legacy default stream).
- Accuracy: relative L2 error ≤ 1e-4 against an FP64 reference computed from the same fp16 inputs.

Build: `nvcc -std=c++17 --expt-relaxed-constexpr -DCUBLASDX_NO_FATBIN_AVAILABLE -O3 -arch=sm_75
-I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include`. Write your code in `solution.cu`.
