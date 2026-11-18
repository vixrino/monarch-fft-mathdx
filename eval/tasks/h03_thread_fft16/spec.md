# h03_thread_fft16: many tiny FFTs, one per thread

Implement `solve` from `task.h` with cuFFTDx for a Tesla T4 (sm_75).

- `batch` complex fp32 signals of length 16: signal b is `x[b*16 + n]` (`.x` real, `.y` imaginary).
- Output the forward DFT `y[b*16 + k] = sum_n x[b*16 + n] * exp(-2*pi*i*n*k/16)`, no normalization.
- `x` and `y` do not alias. `batch` is any integer from 1 to 2^20. Launch on `stream`; do not synchronize.
  `stream` is a non-blocking stream (it does not wait for the legacy default stream).
- Accuracy: relative L2 error ≤ 1e-5 against an FP64 reference on the same inputs.
- Performance: the workload is 2^20 FFTs of 16 points; the reference computes each FFT inside one thread,
  with no shared memory and no synchronization.

Build: `nvcc -std=c++17 --expt-relaxed-constexpr -DCUBLASDX_NO_FATBIN_AVAILABLE -O3 -arch=sm_75
-I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include`. Write your code in `solution.cu`.
