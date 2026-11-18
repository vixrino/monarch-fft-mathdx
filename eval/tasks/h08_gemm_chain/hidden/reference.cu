#include <cublasdx.hpp>

#include "task.h"

using GEMM = decltype(cublasdx::Size<kM, kM, kM>() + cublasdx::Precision<float>() +
                      cublasdx::Type<cublasdx::type::real>() + cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major, cublasdx::row_major>() +
                      cublasdx::MaxAlignment() + cublasdx::Block() + cublasdx::BlockDim<128>() + cublasdx::SM<750>());

// Both GEMMs have the same shape and type, so one description and one pair of smem operands serve both:
// T = A*B leaves the registers through a plain row-major smem buffer and is re-staged as the A operand.
__launch_bounds__(128) __global__ void chain(const float* A, const float* B, const float* C, float* D, unsigned ab) {
  extern __shared__ __align__(16) cublasdx::byte smem[];
  const size_t off = size_t(blockIdx.x) * kM * kM;
  float* t = reinterpret_cast<float*>(smem + ab);
  auto [sa, sb] = cublasdx::slice_shared_memory_ab<GEMM>(smem);
  auto ta = cublasdx::make_tensor(sa, GEMM::get_layout_smem_a());
  auto tb = cublasdx::make_tensor(sb, GEMM::get_layout_smem_b());
  using al = cublasdx::alignment_of<GEMM>;
  auto ga = cublasdx::make_tensor(A + off, GEMM::get_layout_gmem_a());
  auto gb = cublasdx::make_tensor(B + off, GEMM::get_layout_gmem_b());
  auto gc = cublasdx::make_tensor(C + off, GEMM::get_layout_gmem_b());
  auto gd = cublasdx::make_tensor(D + off, GEMM::get_layout_gmem_c());
  cublasdx::copy<GEMM, al::a>(ga, ta);
  cublasdx::copy<GEMM, al::b>(gb, tb);
  cublasdx::copy_wait();
  auto acc = GEMM::get_accumulator();
  GEMM().execute(ta, tb, acc);
  auto r = acc.get_results();
  if (acc.is_thread_active())
    for (int i = 0; i < cublasdx::size(r); ++i) {
      if (acc.is_predicated() && !acc.is_index_in_bounds(i)) continue;
      const auto rc = acc.map_fragment_index(i);
      t[cute::get<0>(rc) * kM + cute::get<1>(rc)] = r(i);
    }
  __syncthreads();  // every thread is done reading ta/tb and T is complete
  for (int e = threadIdx.x; e < kM * kM; e += 128) ta(e / kM, e % kM) = t[e];
  cublasdx::copy<GEMM, al::b>(gc, tb);
  cublasdx::copy_wait();
  __syncthreads();  // the plain stores into ta are not covered by copy_wait
  auto acc2 = GEMM::get_accumulator();
  GEMM().execute(ta, tb, acc2);
  acc2.partition_and_store(gd);
}

void solve(const float* A, const float* B, const float* C, float* D, int batch, cudaStream_t stream) {
  const unsigned ab = (cublasdx::get_shared_storage_size_ab<GEMM>() + 15) / 16 * 16;
  chain<<<batch, 128, ab + kM * kM * sizeof(float), stream>>>(A, B, C, D, ab);
}
