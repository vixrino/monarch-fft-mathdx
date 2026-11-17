#!/usr/bin/env bash
# Usage: bash scripts/sass_check.sh <binary> [function-name-filter]
#        OPS=1 bash scripts/sass_check.sh <binary> [filter]    top opcodes per function instead
# Per function (kernels and non-inlined device functions): total SASS instructions,
# tensor-core MMA (HMMA/IMMA/...), FFMA, HFMA2. Static counts in the binary, not executed counts.
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
    ops[fn "\t" op]++
    if (op ~ /MMA/) mma[fn]++
    if (op == "FFMA") ffma[fn]++
    if (op == "HFMA2") hfma2[fn]++
  }
  END {
    if (ENVIRON["OPS"] != "") {
      for (k in ops) { split(k, kv, "\t"); printf "%d\t%s\t0\t0\t%s\n", ops[k], kv[2], kv[1] }
    } else {
      for (f in total) printf "%d\t%d\t%d\t%d\t%s\n", total[f], mma[f] + 0, ffma[f] + 0, hfma2[f] + 0, f
    }
  }' | sort -t$'\t' -k5 | while IFS=$'\t' read -r t m f h name; do
  # cuFFTDx descriptors demangle to several hundred characters; keep size and precision
  pretty=$(echo "$name" | $demangle |
    sed -E 's/^void ([a-z_]+)<.*Size<\(unsigned int\)([0-9]+)>.*Precision<([^>]+)>.*$/\1 N=\2 \3/')
  if [ -n "$filter" ] && ! echo "$pretty" | grep -q -- "$filter"; then continue; fi
  printf "%7s %6s %6s %6s  %s\n" "$t" "$m" "$f" "$h" "${pretty:0:140}"
done | if [ -n "${OPS:-}" ]; then
  # rows are: count op 0 0 function; keep the 12 most frequent opcodes per function
  sort -k5 -k1,1nr | awk '{ key = substr($0, index($0, $5)); if (++seen[key] <= 12) print }' |
    awk '{ printf "%7s %-12s %s\n", $1, $2, substr($0, index($0, $5)) }'
else
  { printf "%7s %6s %6s %6s  %s\n" total mma ffma hfma2 function; cat; }
fi
