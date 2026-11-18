#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int kN = 1024;

// y[b] = (1/N) * inverse DFT of x[b] in half precision; element n of signal b is the __half2 (re, im) at
// x[b*N + n]. Asynchronous on `stream`.
void solve(const __half2* x, __half2* y, int batch, cudaStream_t stream);
