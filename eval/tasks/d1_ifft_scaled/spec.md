# d1_ifft_scaled: batched inverse FFT with 1/N scaling

Implement `solve` from `task.h` with cuFFTDx, block execution, for a Tesla T4 (sm_75).

- Input `x`: `batch` complex signals of length N = 1024, contiguous, `float2` = (re, im). Signal b
  starts at `x + b * 1024`.
- Output `y`: the same shape, the inverse DFT of each signal scaled by 1/N:
  `y[b][k] = (1/N) * sum_n x[b][n] * exp(+2*pi*i*n*k/N)`.
- `batch` is any integer from 1 to 2^16; it is not necessarily a multiple of anything.
- `x` and `y` do not alias. Launch everything on `stream`; do not synchronize. `stream` is a non-blocking stream (it does not wait for the legacy default stream).
- Accuracy: relative L2 error ≤ 1e-5 against an FP64 reference on the same (fp32) inputs.
- Performance: one global read and one global write per element is enough; a separate scaling kernel
  is not.

Build: `nvcc -std=c++17 --expt-relaxed-constexpr -DCUBLASDX_NO_FATBIN_AVAILABLE -O3 -arch=sm_75
-I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include`. Write your code in `solution.cu`.
