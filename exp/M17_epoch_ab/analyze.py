"""M17 analysis: m15 vs m16 vs m16nf vs vLLM measured in the same launches.

Usage: .venv/bin/python exp/M17_epoch_ab/analyze.py [results/M17_epoch_ab/<timestamp>]
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MID = [160 << 10, 320 << 10, 640 << 10]


def fmt(b):
    return f"{b / (1 << 20):.2f}M" if b >= 1 << 20 else f"{b >> 10}K"


def main():
    base = ROOT / "results" / "M17_epoch_ab"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if p.is_dir())
    out = ROOT / "figures" / "M17_epoch_ab" / run.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"run {run.name}")
    for m in ("eager", "graph"):
        f = run / f"test_m16nf_{m}.json"
        if f.exists():
            d = json.load(open(f))
            r = pd.DataFrame(d["results"])
            print(f"m16nf {m}: {int(r.correct.sum())}/{len(r)} correct, max err {r.max_rel_err.max():.2e}")
        else:
            print(f"m16nf {m}: no result file (see log)")
    p1 = all((run / f"test_m16nf_{m}.json").exists() and json.load(open(run / f"test_m16nf_{m}.json"))["all_correct"]
             for m in ("eager", "graph"))
    rows = []
    for f in run.glob("ab_r*.json"):
        rep = int(re.search(r"_r(\d+)", f.stem).group(1))
        for r in json.load(open(f))["rows"]:
            if r["error"]:
                continue
            ver = "vllm" if r["impl"] == "U_vllm" else r["impl"].split("_")[1]
            ctas = 0 if ver == "vllm" else int(r["impl"].split("_c")[1])
            rows.append(dict(ver=ver, ctas=ctas, bytes=r["bytes"], rep=rep, us=max(r["per_rank_us"]), correct=r["correct"]))
    df = pd.DataFrame(rows)
    df.to_csv(out / "launches.csv", index=False)
    print(f"all perf configs correct: {bool(df.correct.all())}")
    s = df.groupby(["ver", "ctas", "bytes"]).us.agg(med="median", lo="min", hi="max", n="count").reset_index()
    best = s.loc[s.groupby(["ver", "bytes"]).med.idxmin()].set_index(["ver", "bytes"])
    best.to_csv(out / "best.csv")
    piv = best.med.unstack(0)[["m15", "m16", "m16nf", "vllm"]]
    piv["m16/m15"] = piv.m16 / piv.m15
    piv["m16nf/m15"] = piv.m16nf / piv.m15
    for v in ("m15", "m16", "m16nf"):
        piv[f"vllm/{v}"] = piv.vllm / piv[v]
    piv.to_csv(out / "comparison.csv")
    pd.set_option("display.width", 220)
    show = piv.copy()
    show.index = [fmt(b) for b in show.index]
    print(show.round(3).to_string())
    print("best block counts:", best.ctas.unstack(0).rename(index=fmt).to_dict())
    p2 = int((piv["m16/m15"] > 1.05).sum())
    p3 = int(((piv["m16nf/m15"] - 1).abs() <= 0.05).sum())
    print(f"P1 m16nf interleaving all correct: {p1}\nP2 m16 > 1.05 x m15 at {p2}/5 sizes\nP3 |m16nf/m15 - 1| <= 5% at {p3}/5 sizes")
    print("verdict:", "ADOPT m16nf" if p1 and p3 == 5 else "COST IN GLOBAL EPOCH (discuss)" if p1 else "NO-FENCE UNSAFE")
    print("mid-size speedups vs vLLM (m16nf):", [round(piv.loc[b, "vllm/m16nf"], 3) for b in MID])


if __name__ == "__main__":
    main()
