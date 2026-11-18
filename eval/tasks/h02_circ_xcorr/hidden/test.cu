#include "eval_check.cuh"
#include "task.h"

static eval::err check(int batch, uint64_t seed) {
  const size_t n = size_t(batch) * kN;
  std::vector<float2> ha(n), hb(n);
  std::vector<double2> a64(n), b64(n);
  for (size_t i = 0; i < n; ++i) {
    ha[i] = {float(eval::uniform(seed, 2 * i)), float(eval::uniform(seed, 2 * i + 1))};
    hb[i] = {float(eval::uniform(seed + 500, 2 * i)), float(eval::uniform(seed + 500, 2 * i + 1))};
    a64[i] = {ha[i].x, ha[i].y};
    b64[i] = {hb[i].x, hb[i].y};
  }
  eval::dev<float2> a(n), b(n), r(n);
  a.up(ha);
  b.up(hb);
  solve(a.p, b.p, r.p, batch, eval::stream());
  EVAL_CHECK(cudaGetLastError());
  EVAL_CHECK(cudaDeviceSynchronize());
  // correlation theorem: R = conj(A) * B, r = IDFT(R) / N
  auto fa = eval::fft64(a64, kN, CUFFT_FORWARD), fb = eval::fft64(b64, kN, CUFFT_FORWARD);
  for (size_t i = 0; i < n; ++i)
    fa[i] = {fa[i].x * fb[i].x + fa[i].y * fb[i].y, fa[i].x * fb[i].y - fa[i].y * fb[i].x};
  auto ref = eval::fft64(fa, kN, CUFFT_INVERSE);
  auto o = r.down();
  std::vector<double> got, want;
  for (size_t i = 0; i < n; ++i) {
    got.insert(got.end(), {o[i].x, o[i].y});
    want.insert(want.end(), {ref[i].x / kN, ref[i].y / kN});
  }
  return eval::compare(got, want);
}

int main() {
  eval::report rep{"h02_circ_xcorr"};
#ifdef EVAL_PUBLIC
  for (int b : {7}) rep.add(b, 1, check(b, 1), 1e-5);
#else
  for (int b : {1, 3, 333})
    for (uint64_t s : {61, 62}) rep.add(b, s, check(b, s), 1e-5);
  rep.add(1 << 14, 63, check(1 << 14, 63), 1e-5);
#endif
  const int tb = 1 << 14;
  eval::dev<float2> a(size_t(tb) * kN), b(a.n), r(a.n);
  EVAL_CHECK(cudaMemset(a.p, 0x3c, a.n * sizeof(float2)));  // nonzero: zeros draw less power and inflate clocks
  EVAL_CHECK(cudaMemset(b.p, 0x3c, b.n * sizeof(float2)));
  const double ms = eval::median_ms([&] { solve(a.p, b.p, r.p, tb, eval::stream()); });
  rep.print(ms, tb);
  return rep.pass ? 0 : 1;
}
