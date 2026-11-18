#include "eval_check.cuh"
#include "task.h"

constexpr size_t kA = size_t(kM) * kK, kB = size_t(kK) * kN, kCc = size_t(kM) * kN;

static eval::err check(int batch, uint64_t seed) {
  std::vector<__half2> a(batch * kA), b(batch * kB);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = __floats2half2_rn(float(eval::uniform(seed, 2 * i)), float(eval::uniform(seed, 2 * i + 1)));
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = __floats2half2_rn(float(eval::uniform(seed + 1000, 2 * i)), float(eval::uniform(seed + 1000, 2 * i + 1)));
  eval::dev<__half2> da(a.size()), db(b.size());
  eval::dev<float2> dc(batch * kCc);
  da.up(a);
  db.up(b);
  solve(da.p, db.p, dc.p, batch, eval::stream());
  EVAL_CHECK(cudaGetLastError());
  EVAL_CHECK(cudaDeviceSynchronize());
  auto c = dc.down();
  std::vector<double> got, ref;
  got.reserve(2 * c.size());
  ref.reserve(2 * c.size());
  std::vector<double> ar(kA), ai(kA), br(kB), bi(kB);
  for (int m = 0; m < batch; ++m) {
    for (size_t i = 0; i < kA; ++i) ar[i] = __low2float(a[m * kA + i]), ai[i] = __high2float(a[m * kA + i]);
    for (size_t i = 0; i < kB; ++i) br[i] = __low2float(b[m * kB + i]), bi[i] = __high2float(b[m * kB + i]);
    for (int i = 0; i < kM; ++i)
      for (int j = 0; j < kN; ++j) {
        double re = 0, im = 0;
        for (int k = 0; k < kK; ++k) {
          re += ar[i * kK + k] * br[k * kN + j] - ai[i * kK + k] * bi[k * kN + j];
          im += ar[i * kK + k] * bi[k * kN + j] + ai[i * kK + k] * br[k * kN + j];
        }
        const float2 g = c[m * kCc + i * kN + j];
        got.insert(got.end(), {g.x, g.y});
        ref.insert(ref.end(), {re, im});
      }
  }
  return eval::compare(got, ref);
}

int main() {
  eval::report rep{"h06_gemm_complex_hmma"};
#ifdef EVAL_PUBLIC
  for (int b : {7}) rep.add(b, 1, check(b, 1), 1e-4);
#else
  for (int b : {1, 3, 99})
    for (uint64_t s : {101, 102}) rep.add(b, s, check(b, s), 1e-4);
  rep.add(1 << 14, 103, check(1 << 14, 103), 1e-4);
#endif
  const int tb = 1 << 14;
  eval::dev<__half2> a(tb * kA), b(tb * kB);
  eval::dev<float2> c(tb * kCc);
  EVAL_CHECK(cudaMemset(a.p, 0x3c, a.n * sizeof(__half2)));  // nonzero: zeros draw less power and inflate clocks
  EVAL_CHECK(cudaMemset(b.p, 0x3c, b.n * sizeof(__half2)));
  const double ms = eval::median_ms([&] { solve(a.p, b.p, c.p, tb, eval::stream()); });
  rep.print(ms, tb);
  return rep.pass ? 0 : 1;
}
