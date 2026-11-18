# h02_circ_xcorr: circular cross-correlation of signal pairs, in one kernel

Implement `solve` from `task.h` with cuFFTDx, block execution, for a Tesla T4 (sm_75).

- `batch` pairs of complex fp32 signals of length 512; pair p is `a[p*512 + m]` and `b[p*512 + m]`
  (`.x` real, `.y` imaginary).
- Output the circular cross-correlation `r[p*512 + n] = sum_m conj(a[p*512 + m]) * b[p*512 + (m + n) % 512]`
  for n in [0, 512).
- `a`, `b` and `r` do not alias. `batch` is any integer from 1 to 2^16. Launch on `stream`; do not
  synchronize. `stream` is a non-blocking stream (it does not wait for the legacy default stream).
- Accuracy: relative L2 error ≤ 1e-5 against an FP64 reference on the same inputs.
- Performance: one kernel; the spectra stay on chip (no intermediate arrays in global memory).

Build: `nvcc -std=c++17 --expt-relaxed-constexpr -DCUBLASDX_NO_FATBIN_AVAILABLE -O3 -arch=sm_75
-I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include`. Write your code in `solution.cu`.
