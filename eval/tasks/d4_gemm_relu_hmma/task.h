#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int kM = 64;  // M = N = K

// C_b = relu(A_b * B_b + bias[col]); A, B fp16 row-major, C fp32 row-major, matrix b at offset b * 4096.
void solve(const __half* A, const __half* B, const float* bias, float* C, int batch, cudaStream_t stream);
