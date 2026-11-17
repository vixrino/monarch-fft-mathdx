#pragma once
#include <cuda_runtime.h>

constexpr int kN = 256;

// y[b][m] = DFT(x[b])[(m + N/2) % N], for b in [0, batch). Asynchronous on `stream`.
void solve(const float2* x, float2* y, int batch, cudaStream_t stream);
