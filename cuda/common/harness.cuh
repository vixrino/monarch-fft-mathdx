#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cufft.h>
#include <nvml.h>

#include "cuda_check.cuh"

#define CUFFT_CHECK(x)                                                        \
  do {                                                                        \
    cufftResult r_ = (x);                                                     \
    if (r_ != CUFFT_SUCCESS) {                                                \
      std::fprintf(stderr, "%s:%d: cufft error %d\n", __FILE__, __LINE__,     \
                   int(r_));                                                  \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

template <class T>
struct dbuf {
  T* p = nullptr;
  size_t n = 0;
  explicit dbuf(size_t n) : n(n) { CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); }
  ~dbuf() { cudaFree(p); }
  dbuf(const dbuf&) = delete;
  dbuf& operator=(const dbuf&) = delete;
};

// Counter-based, so element i gets the same value for any batch size, buffer type or launch config.
__host__ __device__ inline uint64_t splitmix64(uint64_t z) {
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

__host__ __device__ inline double uniform_pm1(uint64_t seed, uint64_t i) {
  return double(splitmix64(seed * 0x2545f4914f6cdd1dull + i) >> 11) * 0x1.0p-52 - 1.0;
}

__device__ inline void put(double2* p, double re, double im) { *p = make_double2(re, im); }
__device__ inline void put(float2* p, double re, double im) { *p = make_float2(float(re), float(im)); }
__device__ inline void put(__half2* p, double re, double im) {
  // round straight from fp64, as numpy does in reference/monarch.py
  *p = __halves2half2(__double2half(re), __double2half(im));
}

__device__ inline double2 widen(float2 v) { return make_double2(v.x, v.y); }
__device__ inline double2 widen(__half2 v) { return make_double2(__low2float(v), __high2float(v)); }

template <class T>
__global__ void k_fill(T* x, size_t n, uint64_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    put(x + i, uniform_pm1(seed, 2 * i), uniform_pm1(seed, 2 * i + 1));
}

template <class T>
__global__ void k_widen(const T* x, double2* y, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    y[i] = widen(x[i]);
}

template <class T>
void fill_random(dbuf<T>& x, uint64_t seed) {
  k_fill<<<1024, 256>>>(x.p, x.n, seed);
  CUDA_CHECK(cudaGetLastError());
}

template <class T>
std::vector<double2> to_host_fp64(const dbuf<T>& x) {
  dbuf<double2> w(x.n);
  k_widen<<<1024, 256>>>(x.p, w.p, x.n);
  CUDA_CHECK(cudaGetLastError());
  std::vector<double2> h(x.n);
  CUDA_CHECK(cudaMemcpy(h.data(), w.p, x.n * sizeof(double2), cudaMemcpyDeviceToHost));
  return h;
}

inline std::vector<double2> to_host_fp64(const dbuf<double2>& x) {
  std::vector<double2> h(x.n);
  CUDA_CHECK(cudaMemcpy(h.data(), x.p, x.n * sizeof(double2), cudaMemcpyDeviceToHost));
  return h;
}

// FP64 reference: cuFFT Z2Z, forward, unnormalized, contiguous batch.
inline void fft_ref_fp64(const dbuf<double2>& x, dbuf<double2>& y, int n) {
  cufftHandle plan;
  int batch = int(x.n / n);
  CUFFT_CHECK(cufftPlanMany(&plan, 1, &n, nullptr, 1, n, nullptr, 1, n, CUFFT_Z2Z, batch));
  CUFFT_CHECK(cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(x.p),
                           reinterpret_cast<cufftDoubleComplex*>(y.p), CUFFT_FORWARD));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUFFT_CHECK(cufftDestroy(plan));
}

struct err {
  double rel_l2, max_abs;
};

inline err compare(const std::vector<double2>& y, const std::vector<double2>& ref) {
  double num = 0, den = 0, mx = 0;
  for (size_t i = 0; i < y.size(); ++i) {
    double dx = y[i].x - ref[i].x, dy = y[i].y - ref[i].y;
    num += dx * dx + dy * dy;
    den += ref[i].x * ref[i].x + ref[i].y * ref[i].y;
    mx = std::max(mx, std::hypot(dx, dy));
  }
  return {std::sqrt(num / den), mx};
}

inline double quantile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  double pos = q * (v.size() - 1);
  size_t lo = size_t(pos);
  size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (pos - lo) * (v[hi] - v[lo]);
}

struct timing {
  double median, p25, p75;
  int warmup, iters;
};

// One event pair per launch, so the spread across launches is visible (IQR), not averaged away.
template <class F>
timing time_ms(F launch, int warmup = 5, int iters = 50) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  std::vector<double> t(iters);
  for (auto& ti : t) {
    CUDA_CHECK(cudaEventRecord(a));
    launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    ti = ms;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return {quantile(t, 0.5), quantile(t, 0.25), quantile(t, 0.75), warmup, iters};
}

inline std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o + "\"";
}

#ifndef DXB_GIT_COMMIT
#define DXB_GIT_COMMIT "unknown"
#define DXB_GIT_DIRTY 1
#endif
#ifndef DXB_MATHDX_VERSION
#define DXB_MATHDX_VERSION "unknown"
#endif

// JSON members (no braces) shared by every record of one run.
inline std::string provenance(int argc, char** argv) {
  int dev;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  int rt, drv, cufft_ver;
  CUDA_CHECK(cudaRuntimeGetVersion(&rt));
  CUDA_CHECK(cudaDriverGetVersion(&drv));
  CUFFT_CHECK(cufftGetVersion(&cufft_ver));

  char driver[96] = "unknown";
  if (nvmlInit() == NVML_SUCCESS) {
    if (nvmlSystemGetDriverVersion(driver, sizeof driver) != NVML_SUCCESS) std::snprintf(driver, sizeof driver, "unknown");
    nvmlShutdown();
  }

  char ts[32];
  std::time_t now = std::time(nullptr);
  std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));

  std::string cmd;
  for (int i = 0; i < argc; ++i) cmd += (i ? " " : "") + std::string(argv[i]);

  char buf[1024];
  std::snprintf(buf, sizeof buf,
                "\"gpu\":%s,\"sm\":%d%d,\"sm_count\":%d,\"driver\":%s,\"cuda_runtime\":%d,\"cuda_driver_api\":%d,"
                "\"cufft\":%d,\"mathdx\":%s,\"git_commit\":%s,\"git_dirty\":%s,\"timestamp_utc\":%s,\"cmd\":%s",
                jstr(p.name).c_str(), p.major, p.minor, p.multiProcessorCount, jstr(driver).c_str(), rt, drv,
                cufft_ver, jstr(DXB_MATHDX_VERSION).c_str(), jstr(DXB_GIT_COMMIT).c_str(),
                DXB_GIT_DIRTY ? "true" : "false", jstr(ts).c_str(), jstr(cmd).c_str());
  return buf;
}
