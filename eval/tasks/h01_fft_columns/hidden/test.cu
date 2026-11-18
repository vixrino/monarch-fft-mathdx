#include "eval_check.cuh"
#include "task.h"

constexpr size_t kE = size_t(kR) * kC;

static eval::err check(int batch, uint64_t seed) {
  std::vector<float2> h(batch * kE);
  for (size_t i = 0; i < h.size(); ++i)
    h[i] = {float(eval::uniform(seed, 2 * i)), float(eval::uniform(seed, 2 * i + 1))};
  // columns gathered into contiguous rows for the FP64 batched reference
  std::vector<double2> cols(h.size());
  for (int b = 0; b < batch; ++b)
    for (int r = 0; r < kR; ++r)
      for (int c = 0; c < kC; ++c) {
        const float2 v = h[b * kE + r * kC + c];
        cols[(b * kC + c) * size_t(kR) + r] = {v.x, v.y};
      }
  eval::dev<float2> x(h.size());
  x.up(h);
  solve(x.p, batch, eval::stream());
  EVAL_CHECK(cudaGetLastError());
  EVAL_CHECK(cudaDeviceSynchronize());
  auto ref = eval::fft64(cols, kR, CUFFT_FORWARD);
  auto o = x.down();
  std::vector<double> got, want;
  for (int b = 0; b < batch; ++b)
    for (int r = 0; r < kR; ++r)
      for (int c = 0; c < kC; ++c) {
        const float2 g = o[b * kE + r * kC + c];
        const double2 w = ref[(b * kC + c) * size_t(kR) + r];
        got.insert(got.end(), {g.x, g.y});
        want.insert(want.end(), {w.x, w.y});
      }
  return eval::compare(got, want);
}

int main() {
  eval::report rep{"h01_fft_columns"};
#ifdef EVAL_PUBLIC
  for (int b : {7}) rep.add(b, 1, check(b, 1), 1e-5);
#else
  for (int b : {1, 3, 40})
    for (uint64_t s : {51, 52}) rep.add(b, s, check(b, s), 1e-5);
  rep.add(128, 53, check(128, 53), 1e-5);
#endif
  const int tb = 128;  // 2^14 columns
  eval::dev<float2> x(tb * kE);
  EVAL_CHECK(cudaMemset(x.p, 0x3c, x.n * sizeof(float2)));  // nonzero: zeros draw less power and inflate clocks
  const double ms = eval::median_ms([&] { solve(x.p, tb, eval::stream()); });
  rep.print(ms, tb);
  return rep.pass ? 0 : 1;
}
