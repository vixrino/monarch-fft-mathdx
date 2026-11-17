#pragma once
#include <cuda_runtime.h>

constexpr int kM = 32;  // M = N = K

// C_b = A_b * B_b; A row-major, B and C column-major, matrix b at offset b * 1024. Asynchronous on `stream`.
void solve(const float* A, const float* B, float* C, int batch, cudaStream_t stream);
