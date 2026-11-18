#include "eval_check.cuh"
#include "task.h"

constexpr size_t kE = kM * kM;

static eval::err check(int batch, uint64_t seed) {
  std::vector<__half> a(batch * kE), b(batch * kE);
  std::vector<float> bias(kM);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = __float2half(float(eval::uniform(seed, i))), b[i] = __float2half(float(eval::uniform(seed + 1000, i)));
  // centered near zero so ReLU clips a good share of the outputs
  for (int j = 0; j < kM; ++j) bias[j] = float(2 * eval::uniform(seed + 2000, j));
  eval::dev<__half> da(a.size()), db(b.size());
  eval::dev<float> dbias(kM), dc(a.size());
  da.up(a);
  db.up(b);
  dbias.up(bias);
  solve(da.p, db.p, dbias.p, dc.p, batch, eval::stream());
  EVAL_CHECK(cudaGetLastError());
  EVAL_CHECK(cudaDeviceSynchronize());
  auto c = dc.down();
  std::vector<double> got(c.begin(), c.end()), ref(c.size());
  for (int m = 0; m < batch; ++m)
    for (int i = 0; i < kM; ++i)
      for (int j = 0; j < kM; ++j) {
        double s = bias[j];
        for (int k = 0; k < kM; ++k)
          s += double(__half2float(a[m * kE + i * kM + k])) * __half2float(b[m * kE + k * kM + j]);
        ref[m * kE + i * kM + j] = s > 0 ? s : 0;
      }
  return eval::compare(got, ref);
}

int main() {
  eval::report rep{"d4_gemm_relu_hmma"};
#ifdef EVAL_PUBLIC
  for (int b : {7}) rep.add(b, 1, check(b, 1), 1e-4);
#else
  for (int b : {1, 3, 77})
    for (uint64_t s : {41, 42}) rep.add(b, s, check(b, s), 1e-4);
  rep.add(1 << 14, 43, check(1 << 14, 43), 1e-4);
#endif
  const int tb = 1 << 14;
  eval::dev<__half> a(tb * kE), b(tb * kE);
  eval::dev<float> bias(kM), c(tb * kE);
  EVAL_CHECK(cudaMemset(a.p, 0x2c, a.n * sizeof(__half)));
  EVAL_CHECK(cudaMemset(b.p, 0x2c, b.n * sizeof(__half)));
  EVAL_CHECK(cudaMemset(bias.p, 0, kM * sizeof(float)));
  const double ms = eval::median_ms([&] { solve(a.p, b.p, bias.p, c.p, tb, eval::stream()); });
  rep.print(ms, tb);
  return rep.pass ? 0 : 1;
}
