# d3_power_fp16: power spectrum from a half-precision FFT

Implement `solve` from `task.h` with cuFFTDx, block execution, half precision, for a Tesla T4 (sm_75).

- Input `x`: `batch` complex signals of length N = 512, contiguous, one `__half2` per element holding
  (re, im) in natural order: element n of signal b is `x[b*512 + n]`, `.x` = real part, `.y` = imaginary.
- Output `p`: `batch * 512` floats, `p[b*512 + k] = |X_b[k]|^2` where `X_b` is the forward DFT of signal b
  (no scaling), computed in half precision with cuFFTDx.
- `batch` is any integer from 1 to 2^16.
- Launch on `stream`; do not synchronize.
- Accuracy: relative L2 error of `p` ≤ 1e-2 against an FP64 reference computed from the same fp16 inputs.
  An fp16 FFT is expected to be off by about 1e-3 relative; the tolerance leaves room for that, not for a
  layout bug.
- Performance: read `x` once, write `p` once, in one kernel.

Build: `nvcc -std=c++17 --expt-relaxed-constexpr -DCUBLASDX_NO_FATBIN_AVAILABLE -O3 -arch=sm_75
-I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include`. Write your code in `solution.cu`.
