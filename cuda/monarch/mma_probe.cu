// Which cuBLASDx configuration emits HMMA on sm_75 for complex<half> x complex<half> -> complex<float>, and
// what it costs: one 64x64x64 GEMM per block, a batch of them per launch; only the layout/accumulator/tile choice
// differs. Read the instruction from scripts/sass_check.sh on this binary. Every GEMM of the batch is checked
// against an fp64 host GEMM; with --out the timing goes to a JSON-lines file.
#include <cmath>
#include <cstdio>
#include <vector>

#include <cublasdx.hpp>

#include "harness.cuh"

constexpr unsigned m = 64, threads = 128;
constexpr unsigned batch = 8192;  // 205 blocks per SM on the T4's 40 SMs, far above residency
constexpr int warmup = 5, iters = 50;

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
    a += size_t(blockIdx.x) * m * m;
    b += size_t(blockIdx.x) * m * m;
    c += size_t(blockIdx.x) * m * m;
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
bool run(const char* name, FILE* f, const std::string& prov) {
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
  const size_t n = size_t(batch) * m * m;
  std::vector<ta> ba(n), bb(n);
  for (size_t g = 0; g < batch; ++g)
    std::copy(ha.begin(), ha.end(), ba.begin() + g * m * m), std::copy(hb.begin(), hb.end(), bb.begin() + g * m * m);
  ta *da, *db;
  tc* dc;
  CUDA_CHECK(cudaMalloc(&da, n * sizeof(ta)));
  CUDA_CHECK(cudaMalloc(&db, n * sizeof(ta)));
  CUDA_CHECK(cudaMalloc(&dc, n * sizeof(tc)));
  CUDA_CHECK(cudaMemcpy(da, ba.data(), n * sizeof(ta), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, bb.data(), n * sizeof(ta), cudaMemcpyHostToDevice));
  auto kernel = probe<Variant, G, Suggest>;
  const unsigned smem = cublasdx::get_shared_storage_size_ab<G>();
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  int occ = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, threads, smem));

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  std::vector<float> ms;
  for (int it = 0; it < warmup + iters; ++it) {
    CUDA_CHECK(cudaEventRecord(e0));
    kernel<<<batch, threads, smem>>>(da, db, dc);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float t;
    CUDA_CHECK(cudaEventElapsedTime(&t, e0, e1));
    if (it >= warmup) ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());
  auto q = [&](double p) {
    const double x = p * (ms.size() - 1);
    const size_t i = size_t(x);
    return i + 1 < ms.size() ? ms[i] + (x - i) * (ms[i + 1] - ms[i]) : ms[i];
  };
  std::vector<tc> bc(n);
  CUDA_CHECK(cudaMemcpy(bc.data(), dc, n * sizeof(tc), cudaMemcpyDeviceToHost));

  // A row-major (i, k) at i*m + k, B row-major (k, j) at k*m + j, C col-major (i, j) at i + j*m
  double err = 0;
  for (unsigned i = 0; i < m; ++i)
    for (unsigned j = 0; j < m; ++j) {
      double re = 0, im = 0;
      for (unsigned k = 0; k < m; ++k) {
        re += ar[i * m + k] * br[k * m + j] - ai[i * m + k] * bi[k * m + j];
        im += ar[i * m + k] * bi[k * m + j] + ai[i * m + k] * br[k * m + j];
      }
      for (size_t g = 0; g < batch; ++g) {
        const tc v = bc[g * m * m + i + j * m];
        err = std::fmax(err, std::fabs(v.real() - re) + std::fabs(v.imag() - im));
      }
    }
  // 8 real flops per complex multiply-add
  const double med = q(0.5), tflops = 8.0 * m * m * m * batch / (med * 1e-3) / 1e12;
  std::printf("%-10s max |err| = %g  %s  median %.3f ms [%.3f, %.3f]  %.2f TFLOP/s  smem %u B  %d blocks/SM\n", name,
              err, err == 0 ? "ok" : "MISMATCH", med, q(0.25), q(0.75), tflops, smem, occ);
  if (f)
    std::fprintf(f,
                 "{\"experiment\":\"mma_probe\",\"variant\":\"%s\",\"m\":%u,\"n\":%u,\"k\":%u,\"type\":\"complex\","
                 "\"precision\":\"fp16,fp16,fp32\",\"arrangement\":\"row_major,row_major\",\"threads\":%u,"
                 "\"cublasdx\":\"%d.%d.%d\",\"batch\":%u,\"smem_bytes\":%u,\"blocks_per_sm\":%d,\"warmup\":%d,"
                 "\"iters\":%d,\"time_ms_median\":%.6f,\"time_ms_p25\":%.6f,\"time_ms_p75\":%.6f,\"tflops\":%.4f,"
                 "\"max_abs_err\":%g,%s}\n",
                 name, m, m, m, threads, CUBLASDX_VERSION_MAJOR, CUBLASDX_VERSION_MINOR, CUBLASDX_VERSION_PATCH, batch, smem,
                 occ, warmup, iters, med, q(0.25), q(0.75), tflops, err, prov.c_str());
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  return err == 0;
}

int main(int argc, char** argv) {
  args a;
  if (!parse_args(argc, argv, a)) return 2;
  FILE* f = a.out ? std::fopen(a.out, "a") : nullptr;
  if (a.out && !f) {
    std::perror(a.out);
    return 2;
  }
  const std::string prov = provenance(argc, argv);
  bool ok = run<0, base, false>("plain", f, prov);
  ok &= run<1, base, true>("suggested", f, prov);
  ok &= run<2, forced, false>("forced", f, prov);
  if (f) std::fclose(f);
  return ok ? 0 : 1;
}
