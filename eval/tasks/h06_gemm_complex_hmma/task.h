#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int kM = 32, kN = 64, kK = 32;

// C_b = A_b * B_b, complex; A (M x K) and B (K x N) as __half2 (re, im), C (M x N) as float2, all row-major.
void solve(const __half2* A, const __half2* B, float2* C, int batch, cudaStream_t stream);
