// Which cuBLASDx configuration emits HMMA on sm_75 for complex<half> x complex<half> -> complex<float>?
// One 64x64x64 GEMM per kernel; only the layout/accumulator/tile choice differs. Read the answer from
// scripts/sass_check.sh on this binary; the run checks each result against an fp64 host GEMM.
#include <cmath>
#include <cstdio>
#include <vector>

#include <cublasdx.hpp>

#include "cuda_check.cuh"

constexpr unsigned m = 64, threads = 128;

using base = decltype(cublasdx::Size<m, m, m>() + cublasdx::Precision<__half, __half, float>() +
                      cublasdx::Type<cublasdx::type::complex>() + cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>() + cublasdx::MaxAlignment() +
                      cublasdx::Block() + cublasdx::BlockDim<threads>() + cublasdx::SM<DXB_SM>());
// 2 x 2 warps of the 16x8x8 complex atom = 128 threads (include/cublasdx/detail/cute/extension/arch/complex_mma.hpp:169)
using forced = decltype(base() + cublasdx::experimental::Tile<cublasdx::detail::SM75_16x8x8_C32C16C16C32_TN, 2, 2>());

template <class G, bool Suggest>
__device__ auto smem_layout_a() {
  if constexpr (Suggest) return G::suggest_layout_smem_a();
  else return G::get_layout_smem_a();
}

template <class G, bool Suggest>
__device__ auto smem_layout_b() {
  if constexpr (Suggest) return G::suggest_layout_smem_b();
  else return G::get_layout_smem_b();
}

template <class G, bool Suggest>
__device__ auto accumulator() {
  if constexpr (Suggest) return G::suggest_accumulator();
  else return G::get_accumulator();
}

// Variant comes first so it survives the name truncation in sass_check.sh: 0 plain, 1 suggested, 2 forced tile
template <int Variant, class G, bool Suggest>
__launch_bounds__(threads) __global__ void probe(const typename G::a_value_type* a,
                                                 const typename G::b_value_type* b, typename G::c_value_type* c) {
  CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM_BLOCK(G) {
    extern __shared__ __align__(16) cublasdx::byte smem[];
    auto [sa, sb] = cublasdx::slice_shared_memory_ab<G>(smem);
    auto ga = cublasdx::make_tensor(a, G::get_layout_gmem_a());
    auto gb = cublasdx::make_tensor(b, G::get_layout_gmem_b());
    auto gc = cublasdx::make_tensor(c, G::get_layout_gmem_c());
    auto ta = cublasdx::make_tensor(sa, smem_layout_a<G, Suggest>());
    auto tb = cublasdx::make_tensor(sb, smem_layout_b<G, Suggest>());
    using al = cublasdx::alignment_of<G>;
    cublasdx::copy<G, al::a>(ga, ta);
    cublasdx::copy<G, al::b>(gb, tb);
    cublasdx::copy_wait();
    auto acc = accumulator<G, Suggest>();
    G().execute(ta, tb, acc);
    acc.partition_and_store(gc);
  }
}

template <int Variant, class G, bool Suggest>
bool run(const char* name) {
  using ta = typename G::a_value_type;
  using tc = typename G::c_value_type;
  // small integers: exact in fp16, products and sums exact in fp32, so the GEMM must match exactly
  std::vector<ta> ha(m * m), hb(m * m);
  std::vector<double> ar(m * m), ai(m * m), br(m * m), bi(m * m);
  for (unsigned i = 0; i < m * m; ++i) {
    ar[i] = int(i * 7 % 5) - 2, ai[i] = int(i * 3 % 7) - 3, br[i] = int(i * 5 % 3) - 1, bi[i] = int(i % 4) - 2;
    ha[i] = ta(float(ar[i]), float(ai[i]));
    hb[i] = ta(float(br[i]), float(bi[i]));
  }
  ta *da, *db;
  tc* dc;
  CUDA_CHECK(cudaMalloc(&da, m * m * sizeof(ta)));
  CUDA_CHECK(cudaMalloc(&db, m * m * sizeof(ta)));
  CUDA_CHECK(cudaMalloc(&dc, m * m * sizeof(tc)));
  CUDA_CHECK(cudaMemcpy(da, ha.data(), m * m * sizeof(ta), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), m * m * sizeof(ta), cudaMemcpyHostToDevice));
  const unsigned smem = cublasdx::get_shared_storage_size_ab<G>();
  CUDA_CHECK(cudaFuncSetAttribute(probe<Variant, G, Suggest>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  probe<Variant, G, Suggest><<<1, threads, smem>>>(da, db, dc);
  CUDA_CHECK(cudaGetLastError());
  std::vector<tc> hc(m * m);
  CUDA_CHECK(cudaMemcpy(hc.data(), dc, m * m * sizeof(tc), cudaMemcpyDeviceToHost));

  // A row-major (i, k) at i*m + k, B row-major (k, j) at k*m + j, C col-major (i, j) at i + j*m
  double err = 0;
  for (unsigned i = 0; i < m; ++i)
    for (unsigned j = 0; j < m; ++j) {
      double re = 0, im = 0;
      for (unsigned k = 0; k < m; ++k) {
        re += ar[i * m + k] * br[k * m + j] - ai[i * m + k] * bi[k * m + j];
        im += ar[i * m + k] * bi[k * m + j] + ai[i * m + k] * br[k * m + j];
      }
      err = std::fmax(err, std::fabs(hc[i + j * m].real() - re) + std::fabs(hc[i + j * m].imag() - im));
    }
  std::printf("%-10s max |err| = %g  %s\n", name, err, err == 0 ? "ok" : "MISMATCH");
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  return err == 0;
}

int main() {
  bool ok = run<0, base, false>("plain");
  ok &= run<1, base, true>("suggested");
  ok &= run<2, forced, false>("forced");
  return ok ? 0 : 1;
}
