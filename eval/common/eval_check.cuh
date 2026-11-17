// Shared by the eval tests (public and hidden): inputs, FP64 references, comparison, timing, one JSON line.
// Never shown to the agent, apart from public_test.cu files in condition C, which include it.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define EVAL_CHECK(x)                                                                              \
  do {                                                                                             \
    cudaError_t e_ = (x);                                                                          \
    if (e_ != cudaSuccess) {                                                                       \
      std::printf("{\"status\":\"cuda_error\",\"error\":\"%s\",\"line\":%d}\n", cudaGetErrorString(e_), \
                  __LINE__);                                                                       \
      std::exit(3);                                                                                \
    }                                                                                              \
  } while (0)

namespace eval {

inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// U(-1, 1), identical for a given (seed, i) whatever the element type
inline double uniform(uint64_t seed, uint64_t i) {
  return double(splitmix64(seed * 0x100000001B3ull + i) >> 11) * 0x1.0p-52 - 1.0;
}

template <class T>
struct dev {
  T* p = nullptr;
  size_t n = 0;
  explicit dev(size_t n_) : n(n_) { EVAL_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); }
  ~dev() { cudaFree(p); }
  dev(const dev&) = delete;
  void up(const std::vector<T>& h) { EVAL_CHECK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> down() const {
    std::vector<T> h(n);
    EVAL_CHECK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
};

// batched C2C in FP64 on the GPU; dir = CUFFT_FORWARD or CUFFT_INVERSE, unnormalized
inline std::vector<double2> fft64(const std::vector<double2>& x, int n, int dir) {
  const int batch = int(x.size() / n);
  dev<double2> d(x.size());
  d.up(x);
  cufftHandle plan;
  if (cufftPlanMany(&plan, 1, &n, nullptr, 1, n, nullptr, 1, n, CUFFT_Z2Z, batch) != CUFFT_SUCCESS ||
      cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(d.p), reinterpret_cast<cufftDoubleComplex*>(d.p),
                   dir) != CUFFT_SUCCESS) {
    std::printf("{\"status\":\"reference_error\"}\n");
    std::exit(3);
  }
  EVAL_CHECK(cudaDeviceSynchronize());
  cufftDestroy(plan);
  return d.down();
}

struct err {
  double rel_l2 = 0, max_abs = 0;
  size_t worst = 0;
};

// y and ref as flat arrays of doubles (complex data interleaved re, im)
inline err compare(const std::vector<double>& y, const std::vector<double>& ref) {
  double num = 0, den = 0;
  err e;
  for (size_t i = 0; i < y.size(); ++i) {
    const double d = std::isfinite(y[i]) ? y[i] - ref[i] : INFINITY;
    num += d * d;
    den += ref[i] * ref[i];
    if (!(std::fabs(d) <= e.max_abs)) e.max_abs = std::fabs(d), e.worst = i;
  }
  e.rel_l2 = std::sqrt(num / std::max(den, 1e-300));
  return e;
}

template <class F>
double median_ms(F launch, int warmup = 3, int iters = 20) {
  cudaEvent_t a, b;
  EVAL_CHECK(cudaEventCreate(&a));
  EVAL_CHECK(cudaEventCreate(&b));
  for (int i = 0; i < warmup; ++i) launch();
  std::vector<float> t(iters);
  for (int i = 0; i < iters; ++i) {
    EVAL_CHECK(cudaEventRecord(a));
    launch();
    EVAL_CHECK(cudaEventRecord(b));
    EVAL_CHECK(cudaEventSynchronize(b));
    EVAL_CHECK(cudaEventElapsedTime(&t[i], a, b));
  }
  EVAL_CHECK(cudaGetLastError());
  std::sort(t.begin(), t.end());
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return t[iters / 2];
}

// One line per run, parsed by the eval harness: overall pass, every (batch, seed) case, timing.
struct report {
  std::string task;
  bool pass = true;
  std::string cases;
  void add(size_t batch, uint64_t seed, const err& e, double tol) {
    const bool ok = e.rel_l2 <= tol;
    pass &= ok;
    char buf[256];
    std::snprintf(buf, sizeof buf, "%s{\"batch\":%zu,\"seed\":%llu,\"rel_l2\":%.3e,\"max_abs\":%.3e,\"worst\":%zu,\"ok\":%s}",
                  cases.empty() ? "" : ",", batch, (unsigned long long)seed, e.rel_l2, e.max_abs, e.worst,
                  ok ? "true" : "false");
    cases += buf;
  }
  void print(double ms, size_t timed_batch) const {
    std::printf("{\"task\":\"%s\",\"status\":\"%s\",\"cases\":[%s],\"time_ms\":%.6f,\"timed_batch\":%zu}\n",
                task.c_str(), pass ? "pass" : "fail", cases.c_str(), ms, timed_batch);
  }
};

}  // namespace eval
