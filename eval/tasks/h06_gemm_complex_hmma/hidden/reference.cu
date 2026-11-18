#include <cublasdx.hpp>

#include "task.h"

using GEMM = decltype(cublasdx::Size<kM, kN, kK>() + cublasdx::Precision<__half, __half, float>() +
                      cublasdx::Type<cublasdx::type::complex>() + cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major, cublasdx::row_major>() +
                      cublasdx::MaxAlignment() + cublasdx::Block() + cublasdx::BlockDim<128>() + cublasdx::SM<750>());
using ta = typename GEMM::a_value_type;
using tc = typename GEMM::c_value_type;
static_assert(sizeof(ta) == sizeof(__half2) && sizeof(tc) == sizeof(float2), "interleaved (re, im) storage");

// suggested smem layouts + suggested accumulator: the path that selects HMMA on sm_75 (repo docs/decisions.md);
// 32 x 64 x 32 divides the 2 x 2 warp tile of the 16x8x8 complex atom
__launch_bounds__(128) __global__ void cgemm(const ta* A, const ta* B, tc* C) {
  extern __shared__ __align__(16) cublasdx::byte smem[];
  const size_t b = blockIdx.x;
  auto ga = cublasdx::make_tensor(A + b * kM * kK, GEMM::get_layout_gmem_a());
  auto gb = cublasdx::make_tensor(B + b * kK * kN, GEMM::get_layout_gmem_b());
  auto gc = cublasdx::make_tensor(C + b * kM * kN, GEMM::get_layout_gmem_c());
  auto [sa, sb] = cublasdx::slice_shared_memory_ab<GEMM>(smem);
  auto sta = cublasdx::make_tensor(sa, GEMM::suggest_layout_smem_a());
  auto stb = cublasdx::make_tensor(sb, GEMM::suggest_layout_smem_b());
  using al = cublasdx::alignment_of<GEMM>;
  cublasdx::copy<GEMM, al::a>(ga, sta);
  cublasdx::copy<GEMM, al::b>(gb, stb);
  cublasdx::copy_wait();
  auto acc = GEMM::suggest_accumulator();
  GEMM().execute(sta, stb, acc);
  acc.partition_and_store(gc);
}

void solve(const __half2* A, const __half2* B, float2* C, int batch, cudaStream_t stream) {
  const unsigned smem = cublasdx::get_shared_storage_size_ab<GEMM>();
  cgemm<<<batch, 128, smem, stream>>>(reinterpret_cast<const ta*>(A), reinterpret_cast<const ta*>(B),
                                      reinterpret_cast<tc*>(C));
}
