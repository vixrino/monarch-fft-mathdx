#include "eval_check.cuh"
#include "task.h"

static eval::err check(int batch, uint64_t seed) {
  std::vector<__half2> h(size_t(batch) * kN);
  std::vector<double2> h64(h.size());
  for (size_t i = 0; i < h.size(); ++i) {
    h[i] = __floats2half2_rn(float(eval::uniform(seed, 2 * i)), float(eval::uniform(seed, 2 * i + 1)));
    h64[i] = {__low2float(h[i]), __high2float(h[i])};
  }
  eval::dev<__half2> x(h.size()), y(h.size());
  x.up(h);
  solve(x.p, y.p, batch, eval::stream());
  EVAL_CHECK(cudaGetLastError());
  EVAL_CHECK(cudaDeviceSynchronize());
  auto r = eval::fft64(h64, kN, CUFFT_INVERSE);
  auto o = y.down();
  std::vector<double> got, ref;
  for (size_t i = 0; i < o.size(); ++i) {
    got.insert(got.end(), {__low2float(o[i]), __high2float(o[i])});
    ref.insert(ref.end(), {r[i].x / kN, r[i].y / kN});
  }
  return eval::compare(got, ref);
}

int main() {
  eval::report rep{"h04_ifft_fp16_natural"};
#ifdef EVAL_PUBLIC
  for (int b : {7}) rep.add(b, 1, check(b, 1), 1e-2);
#else
  for (int b : {1, 2, 3, 501})
    for (uint64_t s : {81, 82}) rep.add(b, s, check(b, s), 1e-2);
  rep.add(1 << 14, 83, check(1 << 14, 83), 1e-2);
#endif
  const int tb = 1 << 14;
  eval::dev<__half2> x(size_t(tb) * kN), y(size_t(tb) * kN);
  EVAL_CHECK(cudaMemset(x.p, 0x3c, x.n * sizeof(__half2)));  // nonzero: zeros draw less power and inflate clocks
  const double ms = eval::median_ms([&] { solve(x.p, y.p, tb, eval::stream()); });
  rep.print(ms, tb);
  return rep.pass ? 0 : 1;
}
