# d5_fft_shift: forward FFT with the output in centered (fftshift) order

Implement `solve` from `task.h` with cuFFTDx, block execution, for a Tesla T4 (sm_75).

- Input `x`: `batch` complex signals of length N = 256, contiguous, `float2` = (re, im).
- Output `y`: the forward DFT `X[b][k] = sum_n x[b][n] * exp(-2*pi*i*n*k/N)` (no scaling), written in
  centered order: `y[b][m] = X[b][(m + N/2) mod N]`, so the zero frequency lands at `y[b][N/2]`
  (NumPy's `fftshift`).
- `batch` is any integer from 1 to 2^16.
- `x` and `y` do not alias. Launch on `stream`; do not synchronize. `stream` is a non-blocking stream (it does not wait for the legacy default stream).
- Accuracy: relative L2 error ≤ 1e-5 against an FP64 reference on the same inputs.
- Performance: the reordering must not cost an extra pass over global memory.

Build: `nvcc -std=c++17 --expt-relaxed-constexpr -DCUBLASDX_NO_FATBIN_AVAILABLE -O3 -arch=sm_75
-I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include`. Write your code in `solution.cu`.
