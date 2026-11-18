#include <cufftdx.hpp>

#include "task.h"

using namespace cufftdx;

using base = decltype(Block() + Size<kR>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                      Precision<float>() + SM<750>());
using FFT = decltype(base() + ElementsPerThread<base::elements_per_thread>() +
                     FFTsPerBlock<base::suggested_ffts_per_block>());
static_assert(kC % FFT::ffts_per_block == 0, "a block's columns never straddle two matrices");

// One block owns ffts_per_block whole columns: each is read into registers before execute() and written
// after it, so in place is safe without any global synchronization.
__launch_bounds__(FFT::max_threads_per_block) __global__ void fft_columns(float2* x) {
  using C = typename FFT::value_type;
  extern __shared__ __align__(alignof(float4)) cufftdx::byte smem[];
  C td[FFT::storage_size];
  const unsigned col = blockIdx.x * FFT::ffts_per_block + FFT::working_group::batchid();
  float2* m = x + size_t(col / kC) * kR * kC + col % kC;
  const bool act = FFT::working_group::is_thread_active();
  if (act)
    for (unsigned i = 0; i < FFT::input_ept; ++i) {
      const unsigned r = FFT::working_group::fftid() + i * FFT::stride;
      if (r < FFT::input_length) td[i] = C(m[size_t(r) * kC].x, m[size_t(r) * kC].y);
    }
  FFT().execute(td, smem);
  if (act)
    for (unsigned i = 0; i < FFT::output_ept; ++i) {
      const unsigned k = FFT::working_group::fftid() + i * FFT::stride;
      if (k < FFT::output_length) m[size_t(k) * kC] = make_float2(td[i].x, td[i].y);
    }
}

void solve(float2* x, int batch, cudaStream_t stream) {
  static bool once = [] {
    cudaFuncSetAttribute(fft_columns, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size);
    return true;
  }();
  (void)once;
  fft_columns<<<batch * kC / FFT::ffts_per_block, FFT::block_dim, FFT::shared_memory_size, stream>>>(x);
}
