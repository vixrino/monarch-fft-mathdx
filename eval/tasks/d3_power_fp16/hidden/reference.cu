#include <cufftdx.hpp>

#include "task.h"

using namespace cufftdx;

using base = decltype(Block() + Size<kN>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                      Precision<__half>() + SM<750>());
using FFT = decltype(base() + ElementsPerThread<base::elements_per_thread>() +
                     FFTsPerBlock<base::suggested_ffts_per_block>());
static_assert(FFT::implicit_type_batching == 2 && FFT::ffts_per_block % 2 == 0, "fp16 packs two FFTs per register");

// One complex<__half2> register holds FFT a (low halves) and FFT a + fpb/2 (high halves), RRII layout.
__launch_bounds__(FFT::max_threads_per_block) __global__ void power_fp16(const __half2* x, float* p, int batch) {
  using C = typename FFT::value_type;
  constexpr unsigned half = FFT::ffts_per_block / 2;
  extern __shared__ __align__(alignof(float4)) cufftdx::byte smem[];
  C td[FFT::storage_size];
  const unsigned fa = blockIdx.x * FFT::ffts_per_block + FFT::working_group::batchid(), fb = fa + half;
  const bool act = FFT::working_group::is_thread_active();
  const bool la = act && fa < unsigned(batch), lb = act && fb < unsigned(batch);
  const __half2 zero = __floats2half2_rn(0.f, 0.f);
  for (unsigned i = 0; i < FFT::input_ept; ++i) {
    const unsigned k = FFT::working_group::fftid() + i * FFT::stride;
    if (act && k < FFT::input_length) {
      const __half2 a = la ? x[size_t(fa) * kN + k] : zero, b = lb ? x[size_t(fb) * kN + k] : zero;
      td[i] = C(__lows2half2(a, b), __highs2half2(a, b));
    }
  }
  FFT().execute(td, smem);
  for (unsigned i = 0; i < FFT::output_ept; ++i) {
    const unsigned k = FFT::working_group::fftid() + i * FFT::stride;
    if (act && k < FFT::output_length) {
      const float ra = __low2float(td[i].x), ia = __low2float(td[i].y);
      const float rb = __high2float(td[i].x), ib = __high2float(td[i].y);
      if (la) p[size_t(fa) * kN + k] = ra * ra + ia * ia;
      if (lb) p[size_t(fb) * kN + k] = rb * rb + ib * ib;
    }
  }
}

void solve(const __half2* x, float* p, int batch, cudaStream_t stream) {
  static bool once = [] {
    cudaFuncSetAttribute(power_fp16, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size);
    return true;
  }();
  (void)once;
  const unsigned grid = (batch + FFT::ffts_per_block - 1) / FFT::ffts_per_block;
  power_fp16<<<grid, FFT::block_dim, FFT::shared_memory_size, stream>>>(x, p, batch);
}
