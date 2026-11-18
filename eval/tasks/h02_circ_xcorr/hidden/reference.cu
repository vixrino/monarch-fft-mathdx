#include <cufftdx.hpp>

#include "task.h"

using namespace cufftdx;

template <fft_direction D>
using desc = decltype(Block() + Size<kN>() + Type<fft_type::c2c>() + Direction<D>() + Precision<float>() +
                      SM<750>());
using fwd_base = desc<fft_direction::forward>;
// the inverse is pinned to the forward's EPT and FPB, so element i of a thread is frequency
// fftid + i * stride in both, and the product can be formed in registers
using FWD = decltype(fwd_base() + ElementsPerThread<fwd_base::elements_per_thread>() +
                     FFTsPerBlock<fwd_base::suggested_ffts_per_block>());
using INV = decltype(desc<fft_direction::inverse>() + ElementsPerThread<FWD::elements_per_thread>() +
                     FFTsPerBlock<FWD::ffts_per_block>());
static_assert(FWD::stride == INV::stride && FWD::storage_size == INV::storage_size &&
                  FWD::max_threads_per_block == INV::max_threads_per_block,
              "forward and inverse must share the register layout");
constexpr unsigned kSmem = FWD::shared_memory_size > INV::shared_memory_size ? FWD::shared_memory_size
                                                                             : INV::shared_memory_size;

__launch_bounds__(FWD::max_threads_per_block) __global__ void xcorr(const float2* a, const float2* b, float2* r,
                                                                     int batch) {
  using C = typename FWD::value_type;
  extern __shared__ __align__(alignof(float4)) cufftdx::byte smem[];
  C ta[FWD::storage_size], tb[FWD::storage_size];
  const unsigned p = blockIdx.x * FWD::ffts_per_block + FWD::working_group::batchid();
  const bool live = FWD::working_group::is_thread_active() && p < unsigned(batch);
  const size_t off = size_t(p) * kN;
  for (unsigned i = 0; i < FWD::input_ept; ++i) {
    const unsigned k = FWD::working_group::fftid() + i * FWD::stride;
    if (live && k < FWD::input_length) ta[i] = C(a[off + k].x, a[off + k].y), tb[i] = C(b[off + k].x, b[off + k].y);
  }
  FWD().execute(ta, smem);
  __syncthreads();  // smem is reused by the next execute
  FWD().execute(tb, smem);
  for (unsigned i = 0; i < FWD::storage_size; ++i)
    ta[i] = C(ta[i].x * tb[i].x + ta[i].y * tb[i].y, ta[i].x * tb[i].y - ta[i].y * tb[i].x);
  __syncthreads();
  INV().execute(ta, smem);
  for (unsigned i = 0; i < INV::output_ept; ++i) {
    const unsigned n = INV::working_group::fftid() + i * INV::stride;
    if (live && n < INV::output_length) r[off + n] = make_float2(ta[i].x * (1.f / kN), ta[i].y * (1.f / kN));
  }
}

void solve(const float2* a, const float2* b, float2* r, int batch, cudaStream_t stream) {
  static bool once = [] {
    cudaFuncSetAttribute(xcorr, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    return true;
  }();
  (void)once;
  const unsigned grid = (batch + FWD::ffts_per_block - 1) / FWD::ffts_per_block;
  xcorr<<<grid, FWD::block_dim, kSmem, stream>>>(a, b, r, batch);
}
