#include "eval_check.cuh"
#include "task.h"

constexpr size_t kA = size_t(kK) * kM, kB = size_t(kK) * kN, kCc = size_t(kM) * kN;

static eval::err check(int batch, uint64_t seed, float alpha, float beta) {
  std::vector<float> a(batch * kA), b(batch * kB), c(batch * kCc);
  for (size_t i = 0; i < a.size(); ++i) a[i] = float(eval::uniform(seed, i));
  for (size_t i = 0; i < b.size(); ++i) b[i] = float(eval::uniform(seed + 1000, i));
  for (size_t i = 0; i < c.size(); ++i) c[i] = float(eval::uniform(seed + 2000, i));
  eval::dev<float> da(a.size()), db(b.size()), dc(c.size());
  da.up(a);
  db.up(b);
  dc.up(c);
  solve(da.p, db.p, dc.p, alpha, beta, batch, eval::stream());
  EVAL_CHECK(cudaGetLastError());
  EVAL_CHECK(cudaDeviceSynchronize());
  auto o = dc.down();
  std::vector<double> got(o.begin(), o.end()), ref(o.size());
  for (int m = 0; m < batch; ++m)
    for (int i = 0; i < kM; ++i)
      for (int j = 0; j < kN; ++j) {
        double s = 0;
        for (int k = 0; k < kK; ++k) s += double(a[m * kA + k * kM + i]) * b[m * kB + k * kN + j];
        const size_t ci = m * kCc + i * kN + j;
        ref[ci] = double(alpha) * s + double(beta) * c[ci];
      }
  return eval::compare(got, ref);
}

int main() {
  eval::report rep{"h07_gemm_alpha_beta"};
#ifdef EVAL_PUBLIC
  for (int b : {7}) rep.add(b, 1, check(b, 1, 1.5f, 0.5f), 1e-5);
#else
  rep.add(1, 111, check(1, 111, 1.f, 0.f), 1e-5);
  rep.add(3, 112, check(3, 112, -0.75f, 2.f), 1e-5);
  rep.add(301, 113, check(301, 113, 0.5f, -1.25f), 1e-5);
  rep.add(1 << 14, 114, check(1 << 14, 114, 2.f, 1.f), 1e-5);
#endif
  const int tb = 1 << 14;
  eval::dev<float> a(tb * kA), b(tb * kB), c(tb * kCc);
  EVAL_CHECK(cudaMemset(a.p, 0x3c, a.n * sizeof(float)));  // nonzero: zeros draw less power and inflate clocks
  EVAL_CHECK(cudaMemset(b.p, 0x3c, b.n * sizeof(float)));
  EVAL_CHECK(cudaMemset(c.p, 0, c.n * sizeof(float)));
  const double ms = eval::median_ms([&] { solve(a.p, b.p, c.p, 1.f, 0.f, tb, eval::stream()); });
  rep.print(ms, tb);
  return rep.pass ? 0 : 1;
}
