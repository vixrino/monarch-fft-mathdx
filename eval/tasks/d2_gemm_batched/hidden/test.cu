#include "eval_check.cuh"
#include "task.h"

constexpr size_t kE = kM * kM;

static eval::err check(int batch, uint64_t seed) {
  std::vector<float> a(batch * kE), b(batch * kE);
  for (size_t i = 0; i < a.size(); ++i) a[i] = float(eval::uniform(seed, i)), b[i] = float(eval::uniform(seed + 1000, i));
  eval::dev<float> da(a.size()), db(b.size()), dc(a.size());
  da.up(a);
  db.up(b);
  solve(da.p, db.p, dc.p, batch, 0);
  EVAL_CHECK(cudaGetLastError());
  EVAL_CHECK(cudaDeviceSynchronize());
  auto c = dc.down();
  std::vector<double> got(c.begin(), c.end()), ref(c.size());
  for (int m = 0; m < batch; ++m)
    for (int i = 0; i < kM; ++i)
      for (int j = 0; j < kM; ++j) {
        double s = 0;
        for (int k = 0; k < kM; ++k) s += double(a[m * kE + i * kM + k]) * b[m * kE + k + j * kM];
        ref[m * kE + i + j * kM] = s;
      }
  return eval::compare(got, ref);
}

int main() {
  eval::report rep{"d2_gemm_batched"};
#ifdef EVAL_PUBLIC
  for (int b : {2}) rep.add(b, 1, check(b, 1), 1e-5);
#else
  for (int b : {1, 5, 300})
    for (uint64_t s : {21, 22}) rep.add(b, s, check(b, s), 1e-5);
#endif
  const int tb = 1 << 14;
  eval::dev<float> a(tb * kE), b(tb * kE), c(tb * kE);
  EVAL_CHECK(cudaMemset(a.p, 0x3c, a.n * sizeof(float)));
  EVAL_CHECK(cudaMemset(b.p, 0x3c, b.n * sizeof(float)));
  const double ms = eval::median_ms([&] { solve(a.p, b.p, c.p, tb, 0); });
  rep.print(ms, tb);
  return rep.pass ? 0 : 1;
}
