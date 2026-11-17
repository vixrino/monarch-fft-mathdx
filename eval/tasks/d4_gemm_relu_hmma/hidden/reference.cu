#include <cublasdx.hpp>

#include "task.h"

using GEMM = decltype(cublasdx::Size<kM, kM, kM>() + cublasdx::Precision<__half, __half, float>() +
                      cublasdx::Type<cublasdx::type::real>() + cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major, cublasdx::row_major>() +
                      cublasdx::MaxAlignment() + cublasdx::Block() + cublasdx::BlockDim<128>() + cublasdx::SM<750>());

// suggested smem layouts + suggested accumulator: the path that selects HMMA on sm_75 (repo docs/decisions.md)
__launch_bounds__(128) __global__ void gemm_relu(const __half* A, const __half* B, const float* bias, float* C) {
  extern __shared__ __align__(16) cublasdx::byte smem[];
  const size_t off = size_t(blockIdx.x) * kM * kM;
  auto ga = cublasdx::make_tensor(A + off, GEMM::get_layout_gmem_a());
  auto gb = cublasdx::make_tensor(B + off, GEMM::get_layout_gmem_b());
  auto [sa, sb] = cublasdx::slice_shared_memory_ab<GEMM>(smem);
  auto ta = cublasdx::make_tensor(sa, GEMM::suggest_layout_smem_a());
  auto tb = cublasdx::make_tensor(sb, GEMM::suggest_layout_smem_b());
  using al = cublasdx::alignment_of<GEMM>;
  cublasdx::copy<GEMM, al::a>(ga, ta);
  cublasdx::copy<GEMM, al::b>(gb, tb);
  cublasdx::copy_wait();
  auto acc = GEMM::suggest_accumulator();
  GEMM().execute(ta, tb, acc);
  auto r = acc.get_results();
  if (acc.is_thread_active())
    for (int i = 0; i < cublasdx::size(r); ++i) {
      if (acc.is_predicated() && !acc.is_index_in_bounds(i)) continue;
      const auto c = acc.map_fragment_index(i);
      const int row = cute::get<0>(c), col = cute::get<1>(c);
      C[off + row * kM + col] = fmaxf(r(i) + bias[col], 0.f);
    }
}

void solve(const __half* A, const __half* B, const float* bias, float* C, int batch, cudaStream_t stream) {
  const unsigned smem = cublasdx::get_shared_storage_size_ab<GEMM>();
  gemm_relu<<<batch, 128, smem, stream>>>(A, B, bias, C);
}
