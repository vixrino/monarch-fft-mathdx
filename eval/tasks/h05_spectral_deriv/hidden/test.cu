#include "eval_check.cuh"
#include "task.h"

static eval::err check(int batch, uint64_t seed) {
  const size_t n = size_t(batch) * kN;
  std::vector<float> h(n);
  std::vector<double2> h64(n);
  for (size_t i = 0; i < n; ++i) h[i] = float(eval::uniform(seed, i)), h64[i] = {h[i], 0.0};
  eval::dev<float> x(n), d(n);
  x.up(h);
  solve(x.p, d.p, batch, eval::stream());
  EVAL_CHECK(cudaGetLastError());
  EVAL_CHECK(cudaDeviceSynchronize());
  auto X = eval::fft64(h64, kN, CUFFT_FORWARD);
  for (size_t i = 0; i < n; ++i) {
    const int k = int(i % kN);
    const double w = k < kN / 2 ? k : k == kN / 2 ? 0 : k - kN;
    X[i] = {-w * X[i].y, w * X[i].x};
  }
  auto r = eval::fft64(X, kN, CUFFT_INVERSE);
  auto o = d.down();
  std::vector<double> got(o.begin(), o.end()), ref(n);
  for (size_t i = 0; i < n; ++i) ref[i] = r[i].x / kN;
  return eval::compare(got, ref);
}

int main() {
  eval::report rep{"h05_spectral_deriv"};
#ifdef EVAL_PUBLIC
  for (int b : {7}) rep.add(b, 1, check(b, 1), 1e-5);
#else
  for (int b : {1, 3, 257})
    for (uint64_t s : {91, 92}) rep.add(b, s, check(b, s), 1e-5);
  rep.add(1 << 14, 93, check(1 << 14, 93), 1e-5);
#endif
  const int tb = 1 << 14;
  eval::dev<float> x(size_t(tb) * kN), d(size_t(tb) * kN);
  EVAL_CHECK(cudaMemset(x.p, 0x3c, x.n * sizeof(float)));  // nonzero: zeros draw less power and inflate clocks
  const double ms = eval::median_ms([&] { solve(x.p, d.p, tb, eval::stream()); });
  rep.print(ms, tb);
  return rep.pass ? 0 : 1;
}
