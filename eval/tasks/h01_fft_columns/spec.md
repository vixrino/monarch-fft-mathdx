# h01_fft_columns: FFT along the columns of row-major matrices, in place

Implement `solve` from `task.h` with cuFFTDx, block execution, for a Tesla T4 (sm_75).

- `batch` complex fp32 matrices of 128 x 128, row-major: element (r, c) of matrix b at `x[b*16384 + r*128 + c]`
  (`.x` real part, `.y` imaginary part).
- Replace every column by its forward DFT, in place: `X[k][c] = sum_r x[r][c] * exp(-2*pi*i*r*k/128)`,
  no normalization. Rows are not transformed.
- `batch` is any integer from 1 to 1024. Launch on `stream`; do not synchronize. `stream` is a non-blocking stream (it does not wait for the legacy default stream).
- Accuracy: relative L2 error ≤ 1e-5 against an FP64 reference on the same inputs.
- Performance: one kernel; no separate transpose pass through global memory.

Build: `nvcc -std=c++17 --expt-relaxed-constexpr -DCUBLASDX_NO_FATBIN_AVAILABLE -O3 -arch=sm_75
-I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include`. Write your code in `solution.cu`.
