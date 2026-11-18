#include <cublasdx.hpp>

#include "task.h"

// A^T with A stored row-major K x M is the M x K operand in column-major order: no transpose copy needed
using GEMM = decltype(cublasdx::Size<kM, kN, kK>() + cublasdx::Precision<float>() +
                      cublasdx::Type<cublasdx::type::real>() + cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>() +
                      cublasdx::MaxAlignment() + cublasdx::Block() + cublasdx::BlockDim<128>() + cublasdx::SM<750>());

__launch_bounds__(128) __global__ void gemm_ab(const float* A, const float* B, float* C, float alpha, float beta) {
  extern __shared__ __align__(16) cublasdx::byte smem[];
  const size_t b = blockIdx.x;
  auto ga = cublasdx::make_tensor(A + b * kK * kM, GEMM::get_layout_gmem_a());
  auto gb = cublasdx::make_tensor(B + b * kK * kN, GEMM::get_layout_gmem_b());
  float* c = C + b * kM * kN;
  auto [sa, sb] = cublasdx::slice_shared_memory_ab<GEMM>(smem);
  auto ta = cublasdx::make_tensor(sa, GEMM::get_layout_smem_a());
  auto tb = cublasdx::make_tensor(sb, GEMM::get_layout_smem_b());
  using al = cublasdx::alignment_of<GEMM>;
  cublasdx::copy<GEMM, al::a>(ga, ta);
  cublasdx::copy<GEMM, al::b>(gb, tb);
  cublasdx::copy_wait();
  auto acc = GEMM::get_accumulator();
  GEMM().execute(ta, tb, acc);
  // each C element is owned by exactly one thread's fragment, so the read-modify-write needs no sync
  auto r = acc.get_results();
  if (acc.is_thread_active())
    for (int i = 0; i < cublasdx::size(r); ++i) {
      if (acc.is_predicated() && !acc.is_index_in_bounds(i)) continue;
      const auto rc = acc.map_fragment_index(i);
      float& v = c[cute::get<0>(rc) * kN + cute::get<1>(rc)];
      v = alpha * r(i) + (beta == 0.f ? 0.f : beta * v);
    }
}

void solve(const float* A, const float* B, float* C, float alpha, float beta, int batch, cudaStream_t stream) {
  const unsigned smem = cublasdx::get_shared_storage_size_ab<GEMM>();
  gemm_ab<<<batch, 128, smem, stream>>>(A, B, C, alpha, beta);
}
