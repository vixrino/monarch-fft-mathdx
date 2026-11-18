#pragma once
#include <cuda_runtime.h>

constexpr int kM = 64, kN = 32, kK = 48;

// C_b = alpha * A_b^T * B_b + beta * C_b, in place on C. A_b is K x M, B_b is K x N, C_b is M x N, all row-major.
void solve(const float* A, const float* B, float* C, float alpha, float beta, int batch, cudaStream_t stream);
