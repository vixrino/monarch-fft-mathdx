# h04_ifft_fp16_natural: half-precision inverse FFT with natural-layout I/O

Implement `solve` from `task.h` with cuFFTDx, block execution, in half precision, for a Tesla T4 (sm_75).

- `batch` complex signals of length 1024, natural layout: element n of signal b is the `__half2` at
  `x[b*1024 + n]`, low half = real part, high half = imaginary part. The output uses the same layout.
- Output `y[b*1024 + n] = (1/1024) * sum_k x[b*1024 + k] * exp(+2*pi*i*n*k/1024)`.
- `x` and `y` do not alias. `batch` is any integer from 1 to 2^16 (odd values included). Launch on
  `stream`; do not synchronize. `stream` is a non-blocking stream (it does not wait for the legacy default stream).
- The FFT itself runs in half precision (`__half`), not by converting to fp32.
- Accuracy: relative L2 error ≤ 1e-2 against an FP64 reference computed from the same (fp16) inputs.

Build: `nvcc -std=c++17 --expt-relaxed-constexpr -DCUBLASDX_NO_FATBIN_AVAILABLE -O3 -arch=sm_75
-I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include`. Write your code in `solution.cu`.
