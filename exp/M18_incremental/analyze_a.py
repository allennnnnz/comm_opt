"""M18 step a analysis: launch-to-launch spread under the old vs new measurement method, and what slow configs have
in common (clocks, position in the run).

Usage: .venv/bin/python exp/M18_incremental/analyze_a.py [results/M18_incremental/a_<timestamp>]
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TARGET_T = [2, 4, 8, 16, 32, 64]  # 20K-640K at hidden 5120


def main():
    base = ROOT / "results" / "M18_incremental"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.glob("a_*") if p.is_dir())
    out = ROOT / "figures" / "M18_incremental" / run.name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in run.glob("*_s*.json"):
        m = re.match(r"(old|new)_s(\d+)$", f.stem)
        for r in json.load(open(f))["rows"]:
            rows.append(dict(mode=m.group(1), seed=int(m.group(2)), impl=r["impl"], order=r["order"], tokens=r["tokens"],
                             us=max(r["per_rank_us"]), correct=r["correct"], sm_min=min(r["sm_mhz"]),
                             mem_min=min(r["mem_mhz"]), pstate_max=max(r["pstate"])))
    df = pd.DataFrame(rows)
    df.to_csv(out / "launches.csv", index=False)
    print(f"run {run.name}: {len(df)} rows, all correct: {bool(df.correct.all())}")
    df["rel"] = df.us / df.groupby(["mode", "impl", "tokens"]).us.transform("median")
    sp = df.groupby(["mode", "impl", "tokens"]).us.agg(lambda s: s.max() / s.min()).unstack("tokens")
    sp.to_csv(out / "spread.csv")
    pd.set_option("display.width", 220)
    print("\nlaunch-to-launch spread (max/min of per-launch values):")
    print(sp.round(3).to_string())
    print("\nclocks seen before timing (min over ranks): SM", sorted(df.sm_min.unique()), "MEM", sorted(df.mem_min.unique()),
          "Pstate", sorted(df.pstate_max.unique()))
    slow = df[df.rel > 1.05].sort_values("rel", ascending=False)
    print(f"\nconfigs > 5% slower than their median: {len(slow)} (old {int((slow['mode'] == 'old').sum())}, new {int((slow['mode'] == 'new').sum())})")
    print(slow[["mode", "seed", "impl", "order", "tokens", "us", "rel", "sm_min", "mem_min", "pstate_max"]].head(25).to_string(index=False))
    print("\nmean rel by position in run (order of impl):")
    print(df.groupby(["mode", "order"]).rel.mean().unstack(0).round(3).to_string())
    if not {"old", "new"}.issubset(set(df["mode"])) or not {
        ("old", "P_v0_c64"),
        ("new", "P_v0_c64"),
    }.issubset(set(zip(df["mode"], df["impl"]))):
        print("\nverdict: INCOMPLETE RUN (need old/new P_v0_c64 data)")
        return
    o = sp.loc[("old", "P_v0_c64"), TARGET_T]
    n = sp.loc[("new", "P_v0_c64"), TARGET_T]
    holds = bool((n <= 1.05).all()) and bool((o > 1.05).any())
    print(f"\nP_v0_c64 spread at tokens {TARGET_T}: old {o.round(3).tolist()} new {n.round(3).tolist()}")
    print("verdict:", "NEW METHOD FIXES IT" if holds else "NOT AS EXPECTED")


if __name__ == "__main__":
    main()
