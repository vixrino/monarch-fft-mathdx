#!/usr/bin/env python3
# Workaround suggested on NVIDIA/CUDALibrarySamples#378: copy a MathDx root and comment out the sm75 fp16 GEMM
# tuning records, so cuBLASDx falls back to an MMA instruction. The original package is never touched.
#   python3 scripts/war_sm75.py <mathdx root> <copy>
import difflib
import os
import re
import shutil
import sys

REC = "include/cublasdx/database/gemm/database_records.hpp"
LINE = re.compile(r'^#include "cublasdx/database/gemm/sm75/c?fp16_c?fp16_c?fp(16|32)_tn\.hpp\.inc"$')
PATCH = "commented out the 4 sm75 fp16 / complex fp16 #include lines of " + REC


def main():
    src, dst = sys.argv[1:3]
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    path = os.path.join(dst, REC)
    old = open(path).read().splitlines(keepends=True)
    new = ["// " + s if LINE.match(s.rstrip("\n")) else s for s in old]
    changed = sum(a != b for a, b in zip(old, new))
    assert changed == 4, f"expected 4 sm75 fp16 includes, found {changed}"
    open(path, "w").writelines(new)
    sys.stdout.writelines(difflib.unified_diff(old, new, os.path.join(src, REC), path))
    print(PATCH)


if __name__ == "__main__":
    main()
