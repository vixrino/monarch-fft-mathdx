#pragma once
#include <cuda_runtime.h>

constexpr int kR = 128, kC = 128;  // rows, columns

// In place: every column of every matrix replaced by its forward DFT. Matrix b is row-major at x + b*kR*kC.
void solve(float2* x, int batch, cudaStream_t stream);
