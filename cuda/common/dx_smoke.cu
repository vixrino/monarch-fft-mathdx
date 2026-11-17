// Only checks that both Dx headers compile together for CUDA_ARCH and that a kernel runs.
#include <cufftdx.hpp>
#include <cublasdx.hpp>
#include "cuda_check.cuh"

__global__ void touch(int* p) { *p = 42; }

int main() {
  int* d;
  int h = 0;
  CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
  touch<<<1, 1>>>(d);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("%s sm_%d%d kernel %s\n", prop.name, prop.major, prop.minor,
              h == 42 ? "ok" : "FAILED");
  return h == 42 ? 0 : 1;
}
