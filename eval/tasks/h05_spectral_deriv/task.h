#pragma once
#include <cuda_runtime.h>

constexpr int kN = 1024;

// dx[b] = spectral derivative of the periodic real signal x[b] (length N, one period of 2*pi). Asynchronous on `stream`.
void solve(const float* x, float* dx, int batch, cudaStream_t stream);
