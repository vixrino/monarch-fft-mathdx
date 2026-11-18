#include <cufftdx.hpp>

#include "task.h"

using namespace cufftdx;

template <fft_direction D>
using desc = decltype(Block() + Size<kN>() + Type<fft_type::c2c>() + Direction<D>() + Precision<float>() +
                      SM<750>());
using fwd_base = desc<fft_direction::forward>;
// inverse pinned to the forward's EPT and FPB: element i of a thread is bin fftid + i * stride in both
using FWD = decltype(fwd_base() + ElementsPerThread<fwd_base::elements_per_thread>() +
                     FFTsPerBlock<fwd_base::suggested_ffts_per_block>());
using INV = decltype(desc<fft_direction::inverse>() + ElementsPerThread<FWD::elements_per_thread>() +
                     FFTsPerBlock<FWD::ffts_per_block>());
static_assert(FWD::stride == INV::stride && FWD::storage_size == INV::storage_size &&
                  FWD::max_threads_per_block == INV::max_threads_per_block,
              "forward and inverse must share the register layout");
constexpr unsigned kSmem = FWD::shared_memory_size > INV::shared_memory_size ? FWD::shared_memory_size
                                                                             : INV::shared_memory_size;

__launch_bounds__(FWD::max_threads_per_block) __global__ void deriv(const float* x, float* dx, int batch) {
  using C = typename FWD::value_type;
  extern __shared__ __align__(alignof(float4)) cufftdx::byte smem[];
  C td[FWD::storage_size];
  const unsigned s = blockIdx.x * FWD::ffts_per_block + FWD::working_group::batchid();
  const bool live = FWD::working_group::is_thread_active() && s < unsigned(batch);
  const size_t off = size_t(s) * kN;
  for (unsigned i = 0; i < FWD::input_ept; ++i) {
    const unsigned n = FWD::working_group::fftid() + i * FWD::stride;
    if (live && n < FWD::input_length) td[i] = C(x[off + n], 0.f);
  }
  FWD().execute(td, smem);
  for (unsigned i = 0; i < FWD::output_ept; ++i) {
    const int k = int(FWD::working_group::fftid() + i * FWD::stride);
    const float w = k < kN / 2 ? float(k) : k == kN / 2 ? 0.f : float(k - kN);
    td[i] = C(-w * td[i].y, w * td[i].x);
  }
  __syncthreads();  // smem is reused by the inverse
  INV().execute(td, smem);
  for (unsigned i = 0; i < INV::output_ept; ++i) {
    const unsigned n = INV::working_group::fftid() + i * INV::stride;
    if (live && n < INV::output_length) dx[off + n] = td[i].x * (1.f / kN);
  }
}

void solve(const float* x, float* dx, int batch, cudaStream_t stream) {
  static bool once = [] {
    cudaFuncSetAttribute(deriv, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    return true;
  }();
  (void)once;
  const unsigned grid = (batch + FWD::ffts_per_block - 1) / FWD::ffts_per_block;
  deriv<<<grid, FWD::block_dim, kSmem, stream>>>(x, dx, batch);
}
