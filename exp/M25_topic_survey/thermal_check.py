#!/usr/bin/env python
"""M25 B0 thermal gate (Opus 2026-11-25). Job 497945 on hgpn01: GPU 7 hit 93 C and ran at a median 345 MHz under the B1
GEMM+comm load (clocks_event_reasons 0x20 SW thermal / 0x40 HW thermal / 0x8 HW slowdown) while the other seven ran at
1590-1965 MHz; every 8-GPU collective waited for it, so B1 was unusable. This gate reads gpu_monitor.csv for the window of
the all-GPU GEMM burn and fails if any GPU is thermally limited.

usage: thermal_check.py <gpu_monitor.csv> <start 'YYYY/MM/DD HH:MM:SS'> <ngpu>
exit 0 = pass, 3 = fail, 2 = not enough samples (treated as pass by the caller, but recorded)
"""
import csv, statistics as st, sys

THERMAL_BITS = 0x20 | 0x40 | 0x08
MAX_TEMP_C = 88          # fail if any busy sample reaches this
MAX_THERMAL_FRAC = 0.10  # fail if more than 10 % of busy samples carry a thermal/HW-slowdown reason
MIN_MED_CLK = 1300       # fail if the busy-sample median SM clock is below this (MHz)


def main():
    path, start, ngpu = sys.argv[1], sys.argv[2], int(sys.argv[3])
    per = {i: [] for i in range(ngpu)}
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            row = [c.strip() for c in row]
            if len(row) < 8 or row[0][:19] < start:
                continue
            try:
                i = int(row[1]); clk = int(row[2].split()[0]); temp = int(row[5]); util = int(row[6].split()[0])
                rs = int(row[7], 16)
            except ValueError:
                continue
            if i in per and util >= 50:
                per[i].append((clk, temp, rs))
    bad, few = [], []
    for i, v in per.items():
        if len(v) < 20:
            few.append(i)
            print(f"gpu{i} busy_samples={len(v)} (too few)")
            continue
        med = st.median(c for c, _, _ in v); tmax = max(t for _, t, _ in v)
        frac = sum(1 for *_, rs in v if rs & THERMAL_BITS) / len(v)
        ok = tmax < MAX_TEMP_C and frac <= MAX_THERMAL_FRAC and med >= MIN_MED_CLK
        print(f"gpu{i} busy_samples={len(v)} clk_median={med} temp_max={tmax} thermal_frac={frac:.3f} {'ok' if ok else 'BAD'}")
        if not ok:
            bad.append(i)
    if bad:
        print(f"THERMAL_FAIL gpus={bad}")
        sys.exit(3)
    if few:
        print(f"THERMAL_UNKNOWN gpus={few}")
        sys.exit(2)
    print("THERMAL_OK")


if __name__ == "__main__":
    main()
