#!/usr/bin/env python3
# Print a fixed set of metrics from the raw ncu CSVs written by profile.sh, side by side.
# Metric names are matched by pattern, so a name the installed ncu does not export just prints nothing.
#   python3 scripts/ncu_summary.py results/tesla_t4/*_ncu_*.csv
import csv
import re
import sys

PATTERNS = [
    r"^gpu__time_duration\.sum$",
    r"^sm__throughput\.avg\.pct_of_peak_sustained_elapsed$",
    r"^gpu__compute_memory_throughput\.avg\.pct_of_peak_sustained_elapsed$",
    r"^dram__throughput\.avg\.pct_of_peak_sustained_elapsed$",
    r"^dram__bytes_(read|write)\.sum$",
    r"^sm__warps_active\.avg\.pct_of_peak_sustained_active$",
    r"^launch__occupancy_limit_",
    r"^launch__registers_per_thread$",
    r"^launch__shared_mem_per_block",
    r"^sm__inst_executed_pipe_[a-z0-9_]+\.avg\.pct_of_peak_sustained_active$",
    r"^sm__pipe_[a-z0-9_]+_cycles_active\.avg\.pct_of_peak_sustained_active$",
    r"^smsp__inst_executed\.sum$",
    r"^l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_(ld|st)\.sum$",
    r"^l1tex__data_pipe_lsu_wavefronts_mem_shared_op_(ld|st)\.sum$",
    r"^smsp__average_warp_latency_issue_stalled_[a-z_]+\.ratio$",
    r"^smsp__average_warps_issue_stalled_[a-z_]+_per_issue_active\.ratio$",
]


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    head, units, data = rows[0], rows[1], rows[2:]
    return head, units, data


def main():
    cols, names = {}, []
    for p in sys.argv[1:]:
        head, units, data = read(p)
        for row in data:
            r = dict(zip(head, row))
            name = p.rsplit("_ncu_", 1)[-1].removesuffix(".csv")
            names.append(name)
            for h, u in zip(head, units):
                if any(re.search(pat, h) for pat in PATTERNS):
                    cols.setdefault((h, u), {})[name] = r[h]
    print("metric\tunit\t" + "\t".join(names))
    for (h, u), v in sorted(cols.items()):
        vals = [v.get(n, "") for n in names]
        # stall reasons and pipes that are zero everywhere only add noise
        if all(x in ("", "0", "0.00", "0.0") for x in vals):
            continue
        print(f"{h}\t{u}\t" + "\t".join(vals))


if __name__ == "__main__":
    main()
