#!/usr/bin/env bash
# Usage: bash scripts/profile.sh <binary> <key>...      e.g. build/monarch_fft monarch:fp16:4096:64x64
# Nsight Compute, full section set, on one launch of one configuration at the throughput batch.
# Writes results/<gpu>/<date>_ncu_<key>[_k].csv (raw page, every metric) and the .ncu-rep next to it.
set -euo pipefail
bin=$1
shift
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr 'A-Z ' 'a-z_')
dir=results/$gpu
mkdir -p "$dir"
for key in "$@"; do
  stem="$dir/$(date -u +%F)_ncu_$(echo "$key" | tr ':' '_')"
  out=$stem
  k=2
  while [ -e "$out.csv" ]; do out=${stem}_$k; k=$((k + 1)); done
  # the binary launches only this configuration's FFT kernel once; -k drops the table/fill/reference kernels
  ncu --set full -k regex:'monarch_fft|block_fft' --launch-count 1 -o "$out" "$bin" --only "$key" --profile
  ncu --import "$out.ncu-rep" --page raw --csv > "$out.csv"
  echo "$out.csv"
done
