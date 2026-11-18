# h05_spectral_deriv: spectral derivative of periodic signals

Implement `solve` from `task.h` with cuFFTDx, block execution, for a Tesla T4 (sm_75).

- `batch` real fp32 signals of length 1024: signal b is `x[b*1024 + n]`, the samples of one period of a
  2*pi-periodic function at `t_n = 2*pi*n/1024`.
- Output the spectral derivative, real fp32, same layout:
  - `X[k] = sum_n x[n] * exp(-2*pi*i*n*k/1024)`;
  - `D[k] = i * w(k) * X[k]` with `w(k) = k` for `k < 512`, `w(512) = 0`, `w(k) = k - 1024` for `k > 512`;
  - `dx[n] = Re( (1/1024) * sum_k D[k] * exp(+2*pi*i*n*k/1024) )`.
- `x` and `dx` do not alias. `batch` is any integer from 1 to 2^16. Launch on `stream`; do not synchronize.
  `stream` is a non-blocking stream (it does not wait for the legacy default stream).
- Accuracy: relative L2 error ≤ 1e-5 against an FP64 reference on the same inputs.
- Performance: one kernel; the spectrum stays on chip.

Build: `nvcc -std=c++17 --expt-relaxed-constexpr -DCUBLASDX_NO_FATBIN_AVAILABLE -O3 -arch=sm_75
-I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include`. Write your code in `solution.cu`.
