#pragma once
#include <cuda_runtime.h>

constexpr int kN = 1024;

// y[b] = (1/N) * inverse DFT of x[b], for b in [0, batch). Asynchronous on `stream`.
void solve(const float2* x, float2* y, int batch, cudaStream_t stream);
