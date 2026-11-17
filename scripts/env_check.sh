#!/usr/bin/env bash
# Usage: MATHDX_ROOT=<dir>/nvidia/mathdx/YY.MM bash scripts/env_check.sh
set -u

echo "== nvidia-smi"
nvidia-smi || { echo "no GPU visible"; exit 1; }

echo "== nvcc"
nvcc --version || { echo "nvcc not on PATH"; exit 1; }

echo "== GPU"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
echo "compute capability: sm_${cc}"

echo "== MathDx"
if [ -n "${MATHDX_ROOT:-}" ] && [ -d "$MATHDX_ROOT" ]; then
  echo "root: $MATHDX_ROOT"
  vf=$(find "$MATHDX_ROOT" -name 'mathdx-config-version.cmake' | head -1)
  if [ -n "$vf" ]; then
    grep -m1 'set(PACKAGE_VERSION "' "$vf"
  else
    echo "mathdx-config-version.cmake not found; dir name: $(basename "$MATHDX_ROOT")"
  fi
  for h in cufftdx.hpp cublasdx.hpp; do
    [ -f "$MATHDX_ROOT/include/$h" ] && echo "found include/$h" || echo "MISSING include/$h"
  done
else
  echo "MATHDX_ROOT not set or not a directory"
fi

echo "== ncu hardware counters"
ncu_bin=$(command -v ncu || ls /usr/local/cuda/bin/ncu 2>/dev/null || true)
if [ -z "$ncu_bin" ]; then
  echo "ncu: not installed"
  exit 0
fi
"$ncu_bin" --version | tail -1
tmp=$(mktemp -d)
cat > "$tmp/k.cu" <<'CU'
__global__ void k(float* p) { p[threadIdx.x] += 1.f; }
int main() {
  float* d;
  cudaMalloc(&d, 32 * sizeof(float));
  k<<<1, 32>>>(d);
  cudaDeviceSynchronize();
  cudaFree(d);
}
CU
nvcc -arch=sm_${cc} -o "$tmp/k" "$tmp/k.cu" || { echo "ncu: could not build test kernel"; exit 0; }
out=$("$ncu_bin" --metrics sm__cycles_elapsed.avg "$tmp/k" 2>&1)
if echo "$out" | grep -q ERR_NVGPUCTRPERM; then
  echo "ncu: ERR_NVGPUCTRPERM -- counters blocked for non-admin users on this machine."
  echo "     Needs root or NVreg_RestrictProfilingToAdminUsers=0; not fixable on Colab/Kaggle."
elif echo "$out" | grep -q 'sm__cycles_elapsed.avg'; then
  echo "ncu: hardware counters readable"
  echo "$out" | grep 'sm__cycles_elapsed.avg'
else
  echo "ncu: unexpected output"
  echo "$out" | tail -20
fi
rm -rf "$tmp"
