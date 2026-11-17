// Batched block-level Monarch / four-step FFT, N = n1 * n2, as two cuBLASDx GEMMs.
// Index maps and math: reference/monarch.py and docs/derivation.md. Per group of fpb FFTs:
//
//   buf  <- x                as A1 = (fpb*n2 x n1), A1[f*n2 + i2, i1] = x_f[n2*i1 + i2]
//   acc1  = A1 @ F_n1        GEMM 1, M = fpb*n2, N = n1, K = n1 (= (F_n1 @ A1^T)^T, F_n1 is symmetric)
//   buf  <- acc1 * tw        as A2 = (fpb*n1 x n2), A2[f*n1 + j1, i2]; rounded to T
//   acc2  = A2 @ F_n2        GEMM 2, M = fpb*n1, N = n2, K = n2
//   buf  <- acc2             as X_f[j1 + n1*j2] (the transpose, plain indexing), rounded to T, copied out
// A1, A2 and the F tables use cuBLASDx's suggested (swizzled) smem layouts for fp16, plain row-major for fp32.
//
// Every cuBLASDx name used here is checked against the MathDx 26.10 headers/docs; see docs/decisions.md.
#include <type_traits>

#include <cublasdx.hpp>

#include "harness.cuh"

constexpr unsigned threads = 128;

template <class T, unsigned M, unsigned N, unsigned K>
using gemm = decltype(cublasdx::Size<M, N, K>() + cublasdx::Precision<T, T, float>() +
                      cublasdx::Type<cublasdx::type::complex>() + cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>() + cublasdx::MaxAlignment() +
                      cublasdx::Block() + cublasdx::BlockDim<threads>() + cublasdx::SM<DXB_SM>());

template <class T, unsigned N1, unsigned N2, unsigned FPB>
struct monarch {
  static constexpr unsigned n = N1 * N2;
  // both GEMMs put the batch on M: on sm_75 the HMMA path needs M % 32 == 0 (docs/decisions.md)
  using g1 = gemm<T, FPB * N2, N1, N1>;
  using g2 = gemm<T, FPB * N1, N2, N2>;
  using data = typename g1::a_value_type;  // cublasdx::complex<T>
  using io = std::conditional_t<std::is_same_v<T, __half>, __half2, float2>;
  static_assert(sizeof(data) == sizeof(io), "smem element and global element must share the (re, im) layout");
  // suggested layouts are what reach HMMA on sm_75 (mma_probe.cu, docs/decisions.md); fp32 has no tensor cores
  static constexpr bool suggest = std::is_same_v<T, __half>;
  // n1 == n2 makes g1 and g2 the same type, so both GEMMs see F through the same B layout and one table serves both
  static constexpr bool share_f = N1 == N2;
  static_assert(!share_f || std::is_same_v<g1, g2>, "a shared F table needs identical B layouts");
  static constexpr unsigned f_elems = N1 * N1 + (share_f ? 0 : N2 * N2);
  static constexpr unsigned smem = (f_elems + FPB * n) * sizeof(data);
};

template <class G, bool Suggest>
__device__ auto layout_a() {
  if constexpr (Suggest) return G::suggest_layout_smem_a();
  else return G::get_layout_smem_a();
}

template <class G, bool Suggest>
__device__ auto layout_b() {
  if constexpr (Suggest) return G::suggest_layout_smem_b();
  else return G::get_layout_smem_b();
}

template <class G, bool Suggest>
__device__ auto accumulator() {
  if constexpr (Suggest) return G::suggest_accumulator();
  else return G::get_accumulator();
}

template <class M>
__launch_bounds__(threads) __global__
void monarch_fft(const typename M::io* x, typename M::io* y, const typename M::io* f1g, const typename M::io* f2g,
                 const float2* tw, unsigned groups) {
  using g1 = typename M::g1;
  using g2 = typename M::g2;
  using data = typename M::data;
  using io = typename M::io;
  constexpr unsigned n1 = cublasdx::size_of<g1>::n, n2 = cublasdx::size_of<g2>::n, fpb = cublasdx::size_of<g2>::m / n1;
  constexpr unsigned n = M::n;
  constexpr bool sg = M::suggest;

  CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM_BLOCK(g1) {
    extern __shared__ __align__(16) cublasdx::byte smem[];
    data* f1 = reinterpret_cast<data*>(smem);
    data* f2 = M::share_f ? f1 : f1 + n1 * n1;
    data* buf = f1 + M::f_elems;
    const unsigned tid = threadIdx.x;

    // all smem writes below go through (row, col) tensor coordinates, so they are valid for swizzled layouts
    auto a1 = cublasdx::make_tensor(buf, layout_a<g1, sg>());
    auto b1 = cublasdx::make_tensor(f1, layout_b<g1, sg>());
    auto a2 = cublasdx::make_tensor(buf, layout_a<g2, sg>());
    auto b2 = cublasdx::make_tensor(f2, layout_b<g2, sg>());

    // F tables are loaded once per block and reused for every group (persistent grid)
    for (unsigned i = tid; i < n1 * n1; i += threads) b1(i / n1, i % n1) = reinterpret_cast<const data&>(f1g[i]);
    if constexpr (!M::share_f)
      for (unsigned i = tid; i < n2 * n2; i += threads) b2(i / n2, i % n2) = reinterpret_cast<const data&>(f2g[i]);

    for (unsigned g = blockIdx.x; g < groups; g += gridDim.x) {
      const size_t base = size_t(g) * fpb * n;

      for (unsigned e = tid; e < fpb * n; e += threads) {
        const unsigned f = e / n, i1 = (e % n) / n2, i2 = e % n2;
        a1(f * n2 + i2, i1) = reinterpret_cast<const data&>(x[base + e]);
      }
      __syncthreads();

      auto acc1 = accumulator<g1, sg>();
      g1().execute(a1, b1, acc1);
      auto r1 = acc1.get_results();
      __syncthreads();  // other warps may still read a1, which a2 overwrites
      if (acc1.is_thread_active()) {
        for (int i = 0; i < cublasdx::size(r1); ++i) {
          if (acc1.is_predicated() && !acc1.is_index_in_bounds(i)) continue;
          const auto c = acc1.map_fragment_index(i);
          const unsigned row = cute::get<0>(c), j1 = cute::get<1>(c);
          const unsigned f = row / n2, i2 = row % n2;
          const float2 w = tw[j1 * n2 + i2];
          const float re = r1(i).real(), im = r1(i).imag();
          // twiddle in fp32 on the fp32 accumulator; the single rounding to T happens here
          a2(f * n1 + j1, i2) = data(re * w.x - im * w.y, re * w.y + im * w.x);
        }
      }
      __syncthreads();

      auto acc2 = accumulator<g2, sg>();
      g2().execute(a2, b2, acc2);
      auto r2 = acc2.get_results();
      __syncthreads();
      if (acc2.is_thread_active()) {
        for (int i = 0; i < cublasdx::size(r2); ++i) {
          if (acc2.is_predicated() && !acc2.is_index_in_bounds(i)) continue;
          const auto c = acc2.map_fragment_index(i);
          const unsigned row = cute::get<0>(c), j2 = cute::get<1>(c);
          const unsigned f = row / n1, j1 = row % n1;
          buf[f * n + j1 + n1 * j2] = data(r2(i).real(), r2(i).imag());
        }
      }
      __syncthreads();

      for (unsigned e = tid; e < fpb * n; e += threads) y[base + e] = reinterpret_cast<const io&>(buf[e]);
      __syncthreads();  // next group's load overwrites buf
    }
  }
}

// Tables computed in fp64 with sincospi on exact integer phases, rounded once to the storage type.
template <class T>
__global__ void k_dft_matrix(T* f, unsigned m) {
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < m * m; i += gridDim.x * blockDim.x) {
    const unsigned j = i / m, k = i % m;
    double s, c;
    sincospi(-2.0 * double((j * k) % m) / m, &s, &c);
    put(f + i, c, s);
  }
}

__global__ void k_twiddles(float2* tw, unsigned n1, unsigned n2) {
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n1 * n2; i += gridDim.x * blockDim.x) {
    const unsigned j1 = i / n2, i2 = i % n2;
    double s, c;
    sincospi(-2.0 * double(j1 * i2) / (n1 * n2), &s, &c);
    put(tw + i, c, s);
  }
}

template <class T, unsigned N1, unsigned N2>
bool run(const args& a, FILE* f, const std::string& prov) {
  constexpr bool fp16 = std::is_same_v<T, __half>;
  constexpr unsigned n = N1 * N2;
  constexpr unsigned fpb = n >= 4096 ? 1 : 4096 / n;
  using M = monarch<T, N1, N2, fpb>;
  using io = typename M::io;
  auto kernel = monarch_fft<M>;

  int occ = 0;
  const bool fits = int(M::smem) <= smem_optin_max();
  if (fits) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, M::smem));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, threads, M::smem));
  }

  char c[512], label[128];
  std::snprintf(c, sizeof c,
                "\"impl\":\"monarch\",\"cublasdx\":\"%d.%d.%d\",\"precision\":\"%s\",\"n\":%u,\"n1\":%u,\"n2\":%u,"
                "\"ffts_per_block\":%u,\"block_dim\":[%u,1,1],\"smem_bytes\":%u,\"blocks_per_sm\":%d",
                CUBLASDX_VERSION_MAJOR, CUBLASDX_VERSION_MINOR, CUBLASDX_VERSION_PATCH, fp16 ? "fp16" : "fp32", n,
                N1, N2, fpb, threads, M::smem, occ);
  std::snprintf(label, sizeof label, "monarch %s N=%5u %3ux%-3u fpb=%2u smem=%6u", fp16 ? "fp16" : "fp32", n, N1,
                N2, fpb, M::smem);
  // same sanity bounds as the cuFFTDx baseline
  char key[64];
  std::snprintf(key, sizeof key, "monarch:%s:%u:%ux%u", fp16 ? "fp16" : "fp32", n, N1, N2);
  bench b{"monarch_block_fft", c, n, fpb, fp16 ? 1e-2 : 1e-5, label, key};
  if (!fits || occ == 0) {
    skip(f, b, prov, "shared memory");
    return true;
  }

  dbuf<io> f1(N1 * N1), f2(N2 * N2);
  dbuf<float2> tw(n);
  k_dft_matrix<<<64, 256>>>(f1.p, N1);
  k_dft_matrix<<<64, 256>>>(f2.p, N2);
  k_twiddles<<<64, 256>>>(tw.p, N1, N2);
  CUDA_CHECK(cudaGetLastError());

  const size_t resident = size_t(occ) * sm_count();
  return measure<io>(f, b, a, prov, [&](const dbuf<io>& x, dbuf<io>& y) {
    const size_t groups = x.n / n / fpb;
    const size_t grid = std::min(groups, resident);
    kernel<<<unsigned(grid), threads, M::smem>>>(x.p, y.p, f1.p, f2.p, tw.p, unsigned(groups));
    return grid;
  });
}

template <class T>
bool sweep(const args& a, FILE* f, const std::string& prov) {
  bool ok = true;
  ok &= run<T, 8, 32>(a, f, prov);  // GEMM 1 N = 8 is below the 16-wide HMMA warp tile: stays on FMA
  ok &= run<T, 16, 16>(a, f, prov);
  ok &= run<T, 16, 32>(a, f, prov);
  ok &= run<T, 32, 16>(a, f, prov);
  ok &= run<T, 32, 32>(a, f, prov);
  ok &= run<T, 32, 64>(a, f, prov);
  ok &= run<T, 64, 32>(a, f, prov);
  ok &= run<T, 64, 64>(a, f, prov);
  return ok;
}

int main(int argc, char** argv) {
  args a;
  if (!parse_args(argc, argv, a)) return 2;
  FILE* f = a.out ? std::fopen(a.out, "a") : stdout;
  if (!f) {
    std::perror(a.out);
    return 2;
  }
  const std::string prov = provenance(argc, argv);
  bool ok = sweep<__half>(a, f, prov);
  ok &= sweep<float>(a, f, prov);
  if (a.out) std::fclose(f);
  std::fprintf(stderr, ok ? "all accuracy checks passed\n" : "ACCURACY CHECK FAILED\n");
  return ok ? 0 : 1;
}
