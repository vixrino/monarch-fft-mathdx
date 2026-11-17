#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int kN = 512;

// p[b][k] = |DFT(x[b])[k]|^2 computed with a half-precision FFT. Asynchronous on `stream`.
void solve(const __half2* x, float* p, int batch, cudaStream_t stream);
