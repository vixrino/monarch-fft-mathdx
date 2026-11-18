#include "eval_check.cuh"
#include "task.h"

constexpr size_t kE = size_t(kM) * kM;

static eval::err check(int batch, uint64_t seed) {
  std::vector<float> a(batch * kE), b(a.size()), c(a.size());
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = float(eval::uniform(seed, i)), b[i] = float(eval::uniform(seed + 1000, i)),
    c[i] = float(eval::uniform(seed + 2000, i));
  eval::dev<float> da(a.size()), db(a.size()), dc(a.size()), dd(a.size());
  da.up(a);
  db.up(b);
  dc.up(c);
  solve(da.p, db.p, dc.p, dd.p, batch, eval::stream());
  EVAL_CHECK(cudaGetLastError());
  EVAL_CHECK(cudaDeviceSynchronize());
  auto o = dd.down();
  std::vector<double> got(o.begin(), o.end()), ref(o.size()), t(kE);
  for (int m = 0; m < batch; ++m) {
    const size_t off = m * kE;
    for (int i = 0; i < kM; ++i)
      for (int j = 0; j < kM; ++j) {
        double s = 0;
        for (int k = 0; k < kM; ++k) s += double(a[off + i * kM + k]) * b[off + k * kM + j];
        t[i * kM + j] = s;
      }
    for (int i = 0; i < kM; ++i)
      for (int j = 0; j < kM; ++j) {
        double s = 0;
        for (int k = 0; k < kM; ++k) s += t[i * kM + k] * c[off + k * kM + j];
        ref[off + i * kM + j] = s;
      }
  }
  return eval::compare(got, ref);
}

int main() {
  eval::report rep{"h08_gemm_chain"};
#ifdef EVAL_PUBLIC
  for (int b : {7}) rep.add(b, 1, check(b, 1), 1e-5);
#else
  for (int b : {1, 3, 300})
    for (uint64_t s : {121, 122}) rep.add(b, s, check(b, s), 1e-5);
  rep.add(1 << 14, 123, check(1 << 14, 123), 1e-5);
#endif
  const int tb = 1 << 14;
  eval::dev<float> a(tb * kE), b(a.n), c(a.n), d(a.n);
  for (auto* p : {&a, &b, &c})  // nonzero: zeros draw less power and inflate clocks
    EVAL_CHECK(cudaMemset(p->p, 0x3c, p->n * sizeof(float)));
  const double ms = eval::median_ms([&] { solve(a.p, b.p, c.p, d.p, tb, eval::stream()); });
  rep.print(ms, tb);
  return rep.pass ? 0 : 1;
}
