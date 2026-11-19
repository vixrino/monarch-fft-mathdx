// Batched block-level 1D C2C forward FFT with cuFFTDx, fp16 and fp32: the baseline. A third variant stores
// fp16 (__half2 in global memory, like the fp16 path and Monarch fp16) but computes in fp32.
//
// Kernel structure, the load/store index math and the fp16 RRII packing follow the cuFFTDx
// examples shipped with MathDx 26.10 (example/cufftdx/02_simple_fft_block/simple_fft_block.cu,
// 03_block_fft_performance/block_fft_performance.hpp, common/block_io.hpp, common/fp16_common.hpp),
// Copyright NVIDIA Corporation, Apache-2.0. Differences: out-of-place, natural (RIRI) global
// layout for fp16, and my own harness for timing and accuracy.
#include <type_traits>
#include <utility>

#include <cufftdx.hpp>

#include "harness.cuh"

using namespace cufftdx;

// T: compute precision, S: storage precision of the global I/O.
// EPT = FPB = 0: the library's suggestion (the default baseline); otherwise one point of the tuning grid
template <class T, unsigned N, unsigned EPT = 0, unsigned FPB = 0, class S = T>
struct cfg {
  using base = decltype(Block() + Size<N>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                        Precision<T>() + SM<DXB_SM>());
  using fft = decltype(base() + ElementsPerThread<EPT ? EPT : base::elements_per_thread>() +
                       FFTsPerBlock<FPB ? FPB : base::suggested_ffts_per_block>());
  using io = std::conditional_t<std::is_same_v<S, __half>, __half2, float2>;
};

// (re_a, im_a), (re_b, im_b) -> ((re_a, re_b), (im_a, im_b)), from common/fp16_common.hpp
__device__ inline cufftdx::complex<__half2> to_rrii(__half2 a, __half2 b) {
  return cufftdx::complex<__half2>(__lows2half2(a, b), __highs2half2(a, b));
}
__device__ inline __half2 ri_lo(cufftdx::complex<__half2> v) { return __lows2half2(v.x, v.y); }
__device__ inline __half2 ri_hi(cufftdx::complex<__half2> v) { return __highs2half2(v.x, v.y); }

template <class FFT, class IO>
__launch_bounds__(FFT::max_threads_per_block) __global__ void block_fft(const IO* in, IO* out) {
  CUFFTDX_SKIP_IF_NOT_APPLICABLE_SM_BLOCK(FFT) {
    using complex_type = typename FFT::value_type;
    constexpr unsigned n = size_of<FFT>::value;
    constexpr bool fp16 = FFT::implicit_type_batching == 2;
    // fp16: one complex<__half2> register holds FFT b and FFT b + fpb/2 of this block,
    // so batchid() only ranges over [0, fpb/2)
    constexpr size_t pair = fp16 ? (FFT::ffts_per_block / 2) * size_t(n) : 0;

    extern __shared__ __align__(alignof(float4)) cufftdx::byte smem[];
    complex_type td[FFT::storage_size];

    const size_t off = (size_t(blockIdx.x) * FFT::ffts_per_block + FFT::working_group::batchid()) * n;
    const unsigned fftid = FFT::working_group::fftid();

    if (FFT::working_group::is_thread_active()) {
      for (unsigned i = 0; i < FFT::input_ept; ++i) {
        const unsigned k = fftid + i * FFT::stride;
        if (k < FFT::input_length) {
          if constexpr (fp16) {
            td[i] = to_rrii(in[off + k], in[off + pair + k]);
          } else if constexpr (std::is_same_v<IO, __half2>) {
            const float2 v = __half22float2(in[off + k]);
            td[i] = complex_type(v.x, v.y);
          } else {
            td[i] = reinterpret_cast<const complex_type*>(in)[off + k];
          }
        }
      }
    }

    FFT().execute(td, smem);

    if (FFT::working_group::is_thread_active()) {
      for (unsigned i = 0; i < FFT::output_ept; ++i) {
        const unsigned k = fftid + i * FFT::stride;
        if (k < FFT::output_length) {
          if constexpr (fp16) {
            out[off + k] = ri_lo(td[i]);
            out[off + pair + k] = ri_hi(td[i]);
          } else if constexpr (std::is_same_v<IO, __half2>) {
            out[off + k] = __floats2half2_rn(td[i].x, td[i].y);
          } else {
            reinterpret_cast<complex_type*>(out)[off + k] = td[i];
          }
        }
      }
    }
  }
}

template <class T, unsigned N, unsigned EPT = 0, unsigned FPB = 0, class S = T>
bool run(const args& a, FILE* f, const std::string& prov) {
  using FFT = typename cfg<T, N, EPT, FPB, S>::fft;
  using IO = typename cfg<T, N, EPT, FPB, S>::io;
  constexpr bool grid = EPT != 0;
  constexpr bool fp16 = std::is_same_v<T, __half>;
  constexpr bool io16 = std::is_same_v<S, __half>;
  const char* prec = io16 ? "fp16" : "fp32";
  const char* comp = fp16 ? "fp16" : "fp32";
  const char* kp = fp16 ? "fp16" : io16 ? "fp16c32" : "fp32";
  constexpr unsigned fpb = FFT::ffts_per_block;
  static_assert(!FFT::requires_workspace, "power-of-two sizes should not need a Bluestein workspace");
  static_assert(!fp16 || fpb % 2 == 0, "fp16 implicit batching needs an even FFTs-per-block");

  auto kernel = block_fft<FFT, IO>;
  int occ = 0;
  const bool fits = int(FFT::shared_memory_size) <= smem_optin_max();
  if (fits) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, FFT::max_threads_per_block,
                                                             FFT::shared_memory_size));
  }

  char c[512], label[128];
  std::snprintf(c, sizeof c,
                "\"impl\":\"cufftdx\",\"variant\":\"%s\",\"cufftdx\":\"%d.%d.%d\",\"precision\":\"%s\",\"compute\":\"%s\","
                "\"n\":%u,\"ept\":%u,"
                "\"ffts_per_block\":%u,\"block_dim\":[%u,%u,%u],\"smem_bytes\":%u,\"blocks_per_sm\":%d",
                grid ? "grid" : "default", CUFFTDX_VERSION_MAJOR, CUFFTDX_VERSION_MINOR, CUFFTDX_VERSION_PATCH,
                prec, comp, N,
                FFT::elements_per_thread, fpb, FFT::block_dim.x, FFT::block_dim.y, FFT::block_dim.z,
                FFT::shared_memory_size, occ);
  std::snprintf(label, sizeof label, "cufftdx%s %s N=%5u ept=%2u fpb=%3u smem=%6u", grid ? "-grid" : "",
                kp, N, FFT::elements_per_thread, fpb, FFT::shared_memory_size);
  // loose sanity bounds; fp16 includes rounding the input itself (see docs/decisions.md)
  char key[64];
  if (grid)
    std::snprintf(key, sizeof key, "cufftdx:%s:%u:e%u:f%u", kp, N, EPT, FPB);
  else
    std::snprintf(key, sizeof key, "cufftdx:%s:%u", kp, N);
  bench b{"cufftdx_block_fft", c, N, fpb, io16 ? 1e-2 : 1e-5, label, key};
  if (a.only && std::strcmp(a.only, key) != 0) return true;  // --only also silences skipped configs
  if (!fits) {
    skip(f, b, prov, "shared memory");
    return true;
  }
  return measure<IO>(f, b, a, prov, [&](const dbuf<IO>& x, dbuf<IO>& y) {
    const size_t grid = x.n / N / fpb;
    kernel<<<unsigned(grid), FFT::block_dim, FFT::shared_memory_size>>>(x.p, y.p);
    return grid;
  });
}

template <class T, class S, unsigned... Ns>
bool sweep(const args& a, FILE* f, const std::string& prov) {
  return (run<T, Ns, 0, 0, S>(a, f, prov) & ...);
}

#ifdef DXB_GRID_N
// Tuning grid for one N: every ElementsPerThread the sm_75 database lists for this size
// (include/cufftdx/database/records/750/database_{fp16,fp32}_fwd.json, the same lists for both precisions)
// times FFTsPerBlock in {1..32}, keeping what the library's own static_asserts allow: EPT divides N, at most
// 1024 threads, and an even FPB for fp16 (implicit batching). Shared memory is checked at run time.
template <class T, unsigned N, unsigned EPT, unsigned FPB>
constexpr bool legal() {
  constexpr bool fp16 = std::is_same_v<T, __half>;
  return N % EPT == 0 && (!fp16 || FPB % 2 == 0) && (N / EPT) * (fp16 ? FPB / 2 : FPB) <= 1024;
}

template <class T, class S, unsigned N, unsigned EPT, unsigned FPB>
bool run_if_legal(const args& a, FILE* f, const std::string& prov) {
  if constexpr (legal<T, N, EPT, FPB>())
    return run<T, N, EPT, FPB, S>(a, f, prov);
  else
    return true;
}

template <class T, class S, unsigned N, unsigned EPT, unsigned... Log2FPB>
bool grid_fpb(const args& a, FILE* f, const std::string& prov, std::integer_sequence<unsigned, Log2FPB...>) {
  return (run_if_legal<T, S, N, EPT, (1u << Log2FPB)>(a, f, prov) & ...);
}

template <class T, class S, unsigned N, unsigned... EPTs>
bool grid(const args& a, FILE* f, const std::string& prov) {
  return (grid_fpb<T, S, N, EPTs>(a, f, prov, std::make_integer_sequence<unsigned, 6>{}) & ...);
}

template <class T, class S = T>
bool grid_n(const args& a, FILE* f, const std::string& prov) {
  constexpr unsigned n = DXB_GRID_N;
  if constexpr (n <= 2048) return grid<T, S, n, 2, 4, 8, 16, 32>(a, f, prov);
  else if constexpr (n == 4096) return grid<T, S, n, 4, 8, 16, 32>(a, f, prov);
  else if constexpr (n == 8192) return grid<T, S, n, 8, 16, 32>(a, f, prov);
  else return grid<T, S, n, 16, 32>(a, f, prov);
}
#endif

int main(int argc, char** argv) {
  args a;
  if (!parse_args(argc, argv, a)) return 2;
  FILE* f = a.out ? std::fopen(a.out, "a") : stdout;
  if (!f) {
    std::perror(a.out);
    return 2;
  }
  const std::string prov = provenance(argc, argv);
#ifdef DXB_GRID_N
  bool ok = grid_n<float>(a, f, prov);
  ok &= grid_n<__half>(a, f, prov);
  ok &= grid_n<float, __half>(a, f, prov);
#else
  bool ok = sweep<float, float, 256, 512, 1024, 2048, 4096, 8192, 16384>(a, f, prov);
  ok &= sweep<__half, __half, 256, 512, 1024, 2048, 4096, 8192, 16384>(a, f, prov);
  ok &= sweep<float, __half, 256, 512, 1024, 2048, 4096, 8192, 16384>(a, f, prov);
#endif
  if (a.out) std::fclose(f);
  std::fprintf(stderr, ok ? "all accuracy checks passed\n" : "ACCURACY CHECK FAILED\n");
  return ok ? 0 : 1;
}
