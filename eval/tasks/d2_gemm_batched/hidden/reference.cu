#include <cublasdx.hpp>

#include "task.h"

using GEMM = decltype(cublasdx::Size<kM, kM, kM>() + cublasdx::Precision<float>() +
                      cublasdx::Type<cublasdx::type::real>() + cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major, cublasdx::col_major>() +
                      cublasdx::MaxAlignment() + cublasdx::Block() + cublasdx::BlockDim<128>() + cublasdx::SM<750>());

__launch_bounds__(128) __global__ void gemm_batched(const float* A, const float* B, float* C) {
  extern __shared__ __align__(16) cublasdx::byte smem[];
  const size_t off = size_t(blockIdx.x) * kM * kM;
  auto ga = cublasdx::make_tensor(A + off, GEMM::get_layout_gmem_a());
  auto gb = cublasdx::make_tensor(B + off, GEMM::get_layout_gmem_b());
  auto gc = cublasdx::make_tensor(C + off, GEMM::get_layout_gmem_c());
  auto [sa, sb] = cublasdx::slice_shared_memory_ab<GEMM>(smem);
  auto ta = cublasdx::make_tensor(sa, GEMM::get_layout_smem_a());
  auto tb = cublasdx::make_tensor(sb, GEMM::get_layout_smem_b());
  using al = cublasdx::alignment_of<GEMM>;
  cublasdx::copy<GEMM, al::a>(ga, ta);
  cublasdx::copy<GEMM, al::b>(gb, tb);
  cublasdx::copy_wait();
  auto acc = GEMM::get_accumulator();
  GEMM().execute(ta, tb, acc);
  acc.partition_and_store(gc);
}

void solve(const float* A, const float* B, float* C, int batch, cudaStream_t stream) {
  const unsigned smem = cublasdx::get_shared_storage_size_ab<GEMM>();
  gemm_batched<<<batch, 128, smem, stream>>>(A, B, C);
}
