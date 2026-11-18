#include "eval_check.cuh"
#include "task.h"

static eval::err check(int batch, uint64_t seed) {
  std::vector<__half2> h(size_t(batch) * kN);
  std::vector<double2> h64(h.size());
  for (size_t i = 0; i < h.size(); ++i) {
    h[i] = __floats2half2_rn(float(eval::uniform(seed, 2 * i)), float(eval::uniform(seed, 2 * i + 1)));
    h64[i] = {__low2float(h[i]), __high2float(h[i])};
  }
  eval::dev<__half2> x(h.size());
  eval::dev<float> p(h.size());
  x.up(h);
  solve(x.p, p.p, batch, eval::stream());
  EVAL_CHECK(cudaGetLastError());
  EVAL_CHECK(cudaDeviceSynchronize());
  auto r = eval::fft64(h64, kN, CUFFT_FORWARD);
  auto o = p.down();
  std::vector<double> got(o.begin(), o.end()), ref(o.size());
  for (size_t i = 0; i < o.size(); ++i) ref[i] = r[i].x * r[i].x + r[i].y * r[i].y;
  return eval::compare(got, ref);
}

int main() {
  eval::report rep{"d3_power_fp16"};
#ifdef EVAL_PUBLIC
  for (int b : {7}) rep.add(b, 1, check(b, 1), 1e-2);
#else
  for (int b : {1, 2, 3, 501})
    for (uint64_t s : {31, 32}) rep.add(b, s, check(b, s), 1e-2);
  rep.add(1 << 14, 33, check(1 << 14, 33), 1e-2);
#endif
  const int tb = 1 << 14;
  eval::dev<__half2> x(size_t(tb) * kN);
  eval::dev<float> p(size_t(tb) * kN);
  EVAL_CHECK(cudaMemset(x.p, 0x2c, x.n * sizeof(__half2)));
  const double ms = eval::median_ms([&] { solve(x.p, p.p, tb, eval::stream()); });
  rep.print(ms, tb);
  return rep.pass ? 0 : 1;
}
