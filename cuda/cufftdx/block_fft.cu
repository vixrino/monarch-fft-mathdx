// Batched block-level 1D C2C forward FFT with cuFFTDx, fp16 and fp32: the baseline.
//
// Kernel structure, the load/store index math and the fp16 RRII packing follow the cuFFTDx
// examples shipped with MathDx 26.10 (example/cufftdx/02_simple_fft_block/simple_fft_block.cu,
// 03_block_fft_performance/block_fft_performance.hpp, common/block_io.hpp, common/fp16_common.hpp),
// Copyright NVIDIA Corporation, Apache-2.0. Differences: out-of-place, natural (RIRI) global
// layout for fp16, and my own harness for timing and accuracy.
#include <cstring>
#include <type_traits>

#include <cufftdx.hpp>

#include "harness.cuh"

using namespace cufftdx;

template <class T, unsigned N>
struct cfg {
  using base = decltype(Block() + Size<N>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                        Precision<T>() + SM<DXB_SM>());
  using fft = decltype(base() + ElementsPerThread<base::elements_per_thread>() +
                       FFTsPerBlock<base::suggested_ffts_per_block>());
  using io = std::conditional_t<std::is_same_v<T, __half>, __half2, float2>;
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
          if constexpr (fp16)
            td[i] = to_rrii(in[off + k], in[off + pair + k]);
          else
            td[i] = reinterpret_cast<const complex_type*>(in)[off + k];
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
          } else {
            reinterpret_cast<complex_type*>(out)[off + k] = td[i];
          }
        }
      }
    }
  }
}

struct opts {
  bool check = false;
  const char* out = nullptr;
};

constexpr int acc_seeds = 5;
constexpr size_t acc_ffts = 64;
constexpr size_t big_batch = 1 << 14;

size_t round_up(size_t a, size_t b) { return (a + b - 1) / b * b; }

template <class T, unsigned N>
bool run(const opts& o, FILE* f, const std::string& prov) {
  using FFT = typename cfg<T, N>::fft;
  using IO = typename cfg<T, N>::io;
  constexpr bool fp16 = std::is_same_v<T, __half>;
  constexpr unsigned fpb = FFT::ffts_per_block;
  static_assert(!FFT::requires_workspace, "power-of-two sizes should not need a Bluestein workspace");
  static_assert(!fp16 || fpb % 2 == 0, "fp16 implicit batching needs an even FFTs-per-block");
  const char* prec = fp16 ? "fp16" : "fp32";
  // loose sanity bounds, not claims; fp16 includes rounding the input itself (see docs/decisions.md)
  const double tol = fp16 ? 1e-2 : 1e-5;

  auto kernel = block_fft<FFT, IO>;
  int smem_max, dev;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));

  char cfgs[512];
  std::snprintf(cfgs, sizeof cfgs,
                "\"impl\":\"cufftdx\",\"cufftdx\":\"%d.%d.%d\",\"precision\":\"%s\",\"n\":%u,\"ept\":%u,"
                "\"ffts_per_block\":%u,\"block_dim\":[%u,%u,%u],\"smem_bytes\":%u",
                CUFFTDX_VERSION_MAJOR, CUFFTDX_VERSION_MINOR, CUFFTDX_VERSION_PATCH, prec, N,
                FFT::elements_per_thread, fpb, FFT::block_dim.x, FFT::block_dim.y, FFT::block_dim.z,
                FFT::shared_memory_size);

  if (int(FFT::shared_memory_size) > smem_max) {
    std::fprintf(stderr, "%s N=%u: skipped, needs %u B smem > %d B\n", prec, N, FFT::shared_memory_size, smem_max);
    std::fprintf(f, "{\"experiment\":\"cufftdx_block_fft\",\"status\":\"skipped_smem\",%s,%s}\n", cfgs, prov.c_str());
    return true;
  }
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));
  auto launch = [&](const dbuf<IO>& x, dbuf<IO>& y) {
    kernel<<<unsigned(x.n / N / fpb), FFT::block_dim, FFT::shared_memory_size>>>(x.p, y.p);
  };

  // Accuracy: input generated in fp64, rounded to T for the kernel; the reference sees the unrounded fp64 input.
  const size_t ab = round_up(acc_ffts, fpb);
  dbuf<double2> x64(ab * N), r64(ab * N);
  dbuf<IO> x(ab * N), y(ab * N);
  std::vector<double> rel(acc_seeds), mx(acc_seeds);
  for (int s = 0; s < acc_seeds; ++s) {
    fill_random(x64, s);
    fill_random(x, s);
    fft_ref_fp64(x64, r64, N);
    launch(x, y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    err e = compare(to_host_fp64(y), to_host_fp64(r64));
    rel[s] = e.rel_l2;
    mx[s] = e.max_abs;
  }
  const double rel_med = quantile(rel, 0.5), mx_med = quantile(mx, 0.5);
  const bool pass = rel_med < tol;
  std::fprintf(stderr, "%s N=%5u ept=%2u fpb=%3u smem=%6u  rel_l2=%.3e max_abs=%.3e  %s\n", prec, N,
               FFT::elements_per_thread, fpb, FFT::shared_memory_size, rel_med, mx_med, pass ? "ok" : "FAIL");

  char acc[256];
  std::snprintf(acc, sizeof acc,
                "\"rel_l2_median\":%.6e,\"max_abs_median\":%.6e,\"acc_seeds\":%d,\"acc_batch\":%zu,"
                "\"acc_tol\":%.1e,\"acc_pass\":%s",
                rel_med, mx_med, acc_seeds, ab, tol, pass ? "true" : "false");
  if (o.check) return pass;

  int occ;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, FFT::max_threads_per_block,
                                                           FFT::shared_memory_size));
  for (size_t batch : {size_t(fpb), round_up(big_batch, fpb)}) {
    dbuf<IO> xb(batch * N), yb(batch * N);
    fill_random(xb, 0);
    timing t = time_ms([&] { launch(xb, yb); });
    const double s = t.median * 1e-3;
    const double bytes = 2.0 * batch * N * sizeof(IO);
    std::fprintf(f,
                 "{\"experiment\":\"cufftdx_block_fft\",\"status\":\"ok\",%s,\"batch\":%zu,\"grid\":%zu,"
                 "\"blocks_per_sm\":%d,\"time_ms_median\":%.6f,\"time_ms_p25\":%.6f,\"time_ms_p75\":%.6f,"
                 "\"warmup\":%d,\"iters\":%d,\"ffts_per_s\":%.6e,\"gbps_eff\":%.3f,%s,%s}\n",
                 cfgs, batch, batch / fpb, occ, t.median, t.p25, t.p75, t.warmup, t.iters, batch / s, bytes / s * 1e-9,
                 acc, prov.c_str());
    std::fprintf(stderr, "    batch=%6zu  median=%.4f ms  iqr=[%.4f, %.4f]  %.1f GB/s\n", batch, t.median, t.p25,
                 t.p75, bytes / s * 1e-9);
  }
  return pass;
}

template <class T, unsigned... Ns>
bool sweep(const opts& o, FILE* f, const std::string& prov) {
  return (run<T, Ns>(o, f, prov) & ...);
}

int main(int argc, char** argv) {
  opts o;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--check")) o.check = true;
    else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) o.out = argv[++i];
    else {
      std::fprintf(stderr, "usage: %s [--check] [--out results.jsonl]\n", argv[0]);
      return 2;
    }
  }
  FILE* f = o.out ? std::fopen(o.out, "a") : stdout;
  if (!f) {
    std::perror(o.out);
    return 2;
  }
  const std::string prov = provenance(argc, argv);
  bool ok = sweep<float, 256, 512, 1024, 2048, 4096, 8192, 16384>(o, f, prov);
  ok &= sweep<__half, 256, 512, 1024, 2048, 4096, 8192, 16384>(o, f, prov);
  if (o.out) std::fclose(f);
  std::fprintf(stderr, ok ? "all accuracy checks passed\n" : "ACCURACY CHECK FAILED\n");
  return ok ? 0 : 1;
}
