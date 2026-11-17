#include <cufftdx.hpp>

#include "task.h"

using namespace cufftdx;

using base = decltype(Block() + Size<kN>() + Type<fft_type::c2c>() + Direction<fft_direction::inverse>() +
                      Precision<float>() + SM<750>());
using FFT = decltype(base() + ElementsPerThread<base::elements_per_thread>() +
                     FFTsPerBlock<base::suggested_ffts_per_block>());

__launch_bounds__(FFT::max_threads_per_block) __global__ void ifft_scaled(const float2* x, float2* y, int batch) {
  using C = typename FFT::value_type;
  extern __shared__ __align__(alignof(float4)) cufftdx::byte smem[];
  C td[FFT::storage_size];
  const unsigned fft = blockIdx.x * FFT::ffts_per_block + FFT::working_group::batchid();
  // the tail block has idle FFT slots; their threads still take part in execute()
  const bool live = FFT::working_group::is_thread_active() && fft < unsigned(batch);
  const size_t off = size_t(fft) * kN;
  if (live)
    for (unsigned i = 0; i < FFT::input_ept; ++i) {
      const unsigned k = FFT::working_group::fftid() + i * FFT::stride;
      if (k < FFT::input_length) td[i] = C(x[off + k].x, x[off + k].y);
    }
  FFT().execute(td, smem);
  if (live)
    for (unsigned i = 0; i < FFT::output_ept; ++i) {
      const unsigned k = FFT::working_group::fftid() + i * FFT::stride;
      if (k < FFT::output_length) y[off + k] = make_float2(td[i].x * (1.f / kN), td[i].y * (1.f / kN));
    }
}

void solve(const float2* x, float2* y, int batch, cudaStream_t stream) {
  static bool once = [] {
    cudaFuncSetAttribute(ifft_scaled, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size);
    return true;
  }();
  (void)once;
  const unsigned grid = (batch + FFT::ffts_per_block - 1) / FFT::ffts_per_block;
  ifft_scaled<<<grid, FFT::block_dim, FFT::shared_memory_size, stream>>>(x, y, batch);
}
