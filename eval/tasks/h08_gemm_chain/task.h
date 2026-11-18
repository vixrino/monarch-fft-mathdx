#pragma once
#include <cuda_runtime.h>

constexpr int kM = 32;  // all matrices 32 x 32

// D_b = (A_b * B_b) * C_b, all fp32 row-major, matrix b at offset b * 1024. Asynchronous on `stream`.
void solve(const float* A, const float* B, const float* C, float* D, int batch, cudaStream_t stream);
