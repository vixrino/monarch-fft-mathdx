# d4_gemm_relu_hmma: fp16 GEMM with bias and ReLU, on tensor cores

Implement `solve` from `task.h` with cuBLASDx, block execution, for a Tesla T4 (sm_75).

- `batch` independent problems, each `C_b = relu(A_b * B_b + bias)`:
  - `A_b`: 64 x 64, `__half`, row-major, element (i, k) at `A[b*4096 + i*64 + k]`;
  - `B_b`: 64 x 64, `__half`, row-major, element (k, j) at `B[b*4096 + k*64 + j]`;
  - `bias`: 64 floats, shared by all problems, added to column j: `+ bias[j]`;
  - `C_b`: 64 x 64, `float`, row-major, element (i, j) at `C[b*4096 + i*64 + j]`;
  - `relu(v) = max(v, 0)`.
- The products must accumulate in fp32 (inputs fp16, accumulator fp32).
- **The GEMM must run on the tensor cores.** The test inspects the SASS of your solution and fails it if
  no `HMMA` instruction is present, even when the numbers are right.
- `batch` is any integer from 1 to 2^16. Launch on `stream`; do not synchronize.
- Accuracy: relative L2 error ≤ 1e-4 against an FP64 reference computed from the same fp16 inputs.
- Performance: bias and ReLU are applied before the result reaches global memory, not in a second kernel.

Build: `nvcc -std=c++17 --expt-relaxed-constexpr -DCUBLASDX_NO_FATBIN_AVAILABLE -O3 -arch=sm_75
-I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include`. Write your code in `solution.cu`.
