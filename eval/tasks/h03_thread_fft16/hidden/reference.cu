#include <cufftdx.hpp>

#include "task.h"

using namespace cufftdx;

using FFT = decltype(Thread() + Size<kN>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                     Precision<float>() + SM<750>());

__launch_bounds__(256) __global__ void thread_fft(const float2* x, float2* y, int batch) {
  using C = typename FFT::value_type;
  const unsigned b = blockIdx.x * 256 + threadIdx.x;
  if (b >= unsigned(batch)) return;  // thread FFTs have no collective step, so early exit is safe
  C td[FFT::storage_size];
  const size_t off = size_t(b) * kN;
  for (unsigned i = 0; i < FFT::input_length; ++i) td[i] = C(x[off + i].x, x[off + i].y);
  FFT().execute(td);
  for (unsigned i = 0; i < FFT::output_length; ++i) y[off + i] = make_float2(td[i].x, td[i].y);
}

void solve(const float2* x, float2* y, int batch, cudaStream_t stream) {
  thread_fft<<<(batch + 255) / 256, 256, 0, stream>>>(x, y, batch);
}
