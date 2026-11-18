#include "eval_check.cuh"
#include "task.h"

static eval::err check(int batch, uint64_t seed) {
  std::vector<float2> h(size_t(batch) * kN);
  std::vector<double2> h64(h.size());
  for (size_t i = 0; i < h.size(); ++i) {
    h[i] = {float(eval::uniform(seed, 2 * i)), float(eval::uniform(seed, 2 * i + 1))};
    h64[i] = {h[i].x, h[i].y};
  }
  eval::dev<float2> x(h.size()), y(h.size());
  x.up(h);
  solve(x.p, y.p, batch, eval::stream());
  EVAL_CHECK(cudaGetLastError());
  EVAL_CHECK(cudaDeviceSynchronize());
  auto r = eval::fft64(h64, kN, CUFFT_FORWARD);
  auto o = y.down();
  std::vector<double> got, ref;
  for (size_t i = 0; i < o.size(); ++i) {
    got.insert(got.end(), {o[i].x, o[i].y});
    ref.insert(ref.end(), {r[i].x, r[i].y});
  }
  return eval::compare(got, ref);
}

int main() {
  eval::report rep{"h03_thread_fft16"};
#ifdef EVAL_PUBLIC
  for (int b : {7}) rep.add(b, 1, check(b, 1), 1e-5);
#else
  for (int b : {1, 3, 1000, 100001})
    for (uint64_t s : {71, 72}) rep.add(b, s, check(b, s), 1e-5);
  rep.add(1 << 20, 73, check(1 << 20, 73), 1e-5);
#endif
  const int tb = 1 << 20;
  eval::dev<float2> x(size_t(tb) * kN), y(size_t(tb) * kN);
  EVAL_CHECK(cudaMemset(x.p, 0x3c, x.n * sizeof(float2)));  // nonzero: zeros draw less power and inflate clocks
  const double ms = eval::median_ms([&] { solve(x.p, y.p, tb, eval::stream()); });
  rep.print(ms, tb);
  return rep.pass ? 0 : 1;
}
