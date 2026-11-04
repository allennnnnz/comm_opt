"""M16 analysis: interleaving tests (old vs new kernel) and speed regression against M15.

Usage: .venv/bin/python exp/M16_epoch_fix/analyze.py [results/M16_epoch_fix/<timestamp>]
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
M15 = ROOT / "figures/M15_bf16/20261104_102644/summary.csv"
MID = [160 << 10, 320 << 10, 640 << 10]


def fmt(b):
    return f"{b / (1 << 20):.2f}M" if b >= 1 << 20 else f"{b >> 10}K"


def main():
    base = ROOT / "results" / "M16_epoch_fix"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if p.is_dir())
    out = ROOT / "figures" / "M16_epoch_fix" / run.name
    out.mkdir(parents=True, exist_ok=True)
    sched = pd.read_csv(run / "schedule.tsv", sep="\t", comment="#")
    rc = dict(zip(sched.tag, sched.rc))
    print(f"run {run.name}")
    for tag in ("test_old_eager", "test_new_eager", "test_new_graph"):
        f = run / f"{tag}.json"
        if f.exists():
            d = json.load(open(f))
            r = pd.DataFrame(d["results"])
            print(f"{tag}: rc={rc[tag]} calls={len(r)} correct={int(r.correct.sum())} all_correct={d['all_correct']} "
                  f"max_err={r.max_rel_err.max():.2e} tokens={sorted(set(r.tokens))} ctas={sorted(set(r.ctas))}")
        else:
            log = (run / f"{tag}.log").read_text().splitlines()
            done = [l for l in log if l.startswith("call ")]
            print(f"{tag}: rc={rc[tag]} (no result file; {len(done)} calls finished before it stopped; "
                  f"last: {done[-1] if done else 'none'})")
    p1 = rc["test_old_eager"] != 0 or not json.load(open(run / "test_old_eager.json"))["all_correct"] \
        if (run / "test_old_eager.json").exists() else rc["test_old_eager"] != 0
    p2 = all((run / f"{t}.json").exists() and json.load(open(run / f"{t}.json"))["all_correct"] and rc[t] == 0
             for t in ("test_new_eager", "test_new_graph"))
    print(f"P1 old kernel fails: {p1}\nP2 new kernel all correct (eager + graph): {p2}")

    rows = []
    for f in run.glob("perf_*_r*.json"):
        m = re.match(r"perf_(\w+?)_r(\d+)$", f.stem)
        for r in json.load(open(f))["rows"]:
            if r["error"]:
                continue
            rows.append(dict(dtype=m.group(1), impl=r["impl"], bytes=r["bytes"], rep=int(m.group(2)),
                             us=max(r["per_rank_us"]), correct=r["correct"]))
    df = pd.DataFrame(rows)
    df.to_csv(out / "perf_launches.csv", index=False)
    s = df.groupby(["dtype", "impl", "bytes"]).agg(med=("us", "median"), lo=("us", "min"), hi=("us", "max"),
                                                    n=("us", "count"), correct=("correct", "all")).reset_index()
    s.to_csv(out / "perf_summary.csv", index=False)
    old = pd.read_csv(M15)
    old = old[old.hidden == 5120]
    res = []
    for dt in ("float16", "bfloat16"):
        for b in sorted(s.bytes.unique()):
            new_p = s[(s.dtype == dt) & s.impl.str.startswith("P_ours") & (s.bytes == b)].med.min()
            new_u = s[(s.dtype == dt) & (s.impl == "U_vllm") & (s.bytes == b)].med.iloc[0]
            old_p = old[(old.dtype == dt) & old.impl.str.startswith("P_ours") & (old.bytes == b)].med.min()
            res.append(dict(dtype=dt, bytes=b, new=new_p, m15=old_p, change=new_p / old_p - 1, vllm=new_u,
                            speedup=new_u / new_p))
    res = pd.DataFrame(res)
    res.to_csv(out / "regression.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.assign(bytes=res.bytes.map(fmt)).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    p3a = bool((res.change.abs() <= 0.05).all())
    p3b = bool(all(res[(res.dtype == dt) & (res.bytes == b)].speedup.iloc[0] >= 1.3 for dt in ("float16", "bfloat16")
                   for b in MID))
    print(f"P3 within 5% of M15 at every size: {p3a}; mid sizes still >= 1.3x vs vLLM: {p3b}; all correct: {bool(s.correct.all())}")
    print("verdict:", "FIXED" if p2 and p3a and p3b else "FIXED BUT SLOWER" if p2 else "NOT FIXED")


if __name__ == "__main__":
    main()
