#!/usr/bin/env bash
# Usage: bash scripts/sass_check.sh <binary> [kernel-name-filter]
# Per kernel: total SASS instructions, tensor-core MMA (HMMA/IMMA/...), FFMA, HFMA2.
# Static counts in the binary, not executed counts.
set -euo pipefail
bin=$1
filter=${2:-}
demangle=$(command -v cu++filt || command -v c++filt || echo cat)

cuobjdump -sass "$bin" | awk '
  /Function :/ { fn = $3; next }
  /^[ \t]*\/\*[0-9a-f]+\*\// {
    line = $0
    sub(/^[ \t]*\/\*[0-9a-f]+\*\/[ \t]*/, "", line)
    sub(/^@!?[A-Z]*[0-9]* +/, "", line)          # predicate guard, e.g. @P0 or @!PT
    split(line, tok, /[ .;]/)
    op = tok[1]
    if (op == "") next
    total[fn]++
    if (op ~ /MMA/) mma[fn]++
    if (op == "FFMA") ffma[fn]++
    if (op == "HFMA2") hfma2[fn]++
  }
  END {
    for (f in total) printf "%d\t%d\t%d\t%d\t%s\n", total[f], mma[f] + 0, ffma[f] + 0, hfma2[f] + 0, f
  }' | sort -t$'\t' -k5 | while IFS=$'\t' read -r t m f h name; do
  pretty=$(echo "$name" | $demangle)
  if [ -n "$filter" ] && ! echo "$pretty" | grep -q -- "$filter"; then continue; fi
  printf "%7s %6s %6s %6s  %s\n" "$t" "$m" "$f" "$h" "${pretty:0:140}"
done | { printf "%7s %6s %6s %6s  %s\n" total mma ffma hfma2 kernel; cat; }
