#pragma once
#include <cuda_runtime.h>

constexpr int kN = 512;

// r[b][n] = sum_m conj(a[b][m]) * b_[b][(m + n) mod N], for every pair b in [0, batch). Asynchronous on `stream`.
void solve(const float2* a, const float2* b, float2* r, int batch, cudaStream_t stream);
