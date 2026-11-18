#pragma once
#include <cuda_runtime.h>

constexpr int kN = 16;

// y[b] = forward DFT of x[b] (length 16), for b in [0, batch). Asynchronous on `stream`.
void solve(const float2* x, float2* y, int batch, cudaStream_t stream);
