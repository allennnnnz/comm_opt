"""M18 step b analysis: validation run, v0 + per-size block table vs v0 fixed 64 blocks vs vLLM.

Usage: .venv/bin/python exp/M18_incremental/analyze_b.py [results/M18_incremental/b_<timestamp>]
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def summarize(run, prefix):
    rows = []
    for f in run.glob(f"{prefix}_s*.json"):
        s = int(re.search(r"_s(\d+)$", f.stem).group(1))
        for r in json.load(open(f))["rows"]:
            rows.append(dict(seed=s, impl=r["impl"], ctas=r["ctas"], tokens=r["tokens"], us=max(r["per_rank_us"]),
                             correct=r["correct"] and not r["error"], sm=min(r["sm_mhz"])))
    return pd.DataFrame(rows)


def compare(df, new, old, label):
    """effective / worse per tokens: >= 3% median change and disjoint [min, max] over launches"""
    out = []
    for t in sorted(df.tokens.unique()):
        a = df[(df.impl == new) & (df.tokens == t)].us
        b = df[(df.impl == old) & (df.tokens == t)].us
        change = a.median() / b.median() - 1
        status = ("effective" if change <= -0.03 and a.max() < b.min() else
                  "worse" if change >= 0.03 and a.min() > b.max() else "no difference")
        out.append(dict(tokens=t, new=a.median(), old=b.median(), change=change, status=status))
    r = pd.DataFrame(out)
    print(f"\n{label}:\n{r.to_string(index=False, float_format=lambda v: f'{v:.3f}')}")
    return r


def main():
    base = ROOT / "results" / "M18_incremental"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.glob("b_*") if p.is_dir())
    out = ROOT / "figures" / "M18_incremental" / run.name
    out.mkdir(parents=True, exist_ok=True)
    cal = summarize(run, "calib")
    cal.to_csv(out / "calibration.csv", index=False)
    piv = cal.groupby(["tokens", "ctas"]).us.median().unstack("ctas")
    print(f"run {run.name}\ncalibration (median of 3, µs), all correct: {bool(cal.correct.all())}, SM {sorted(cal.sm.unique())}")
    print(piv.round(2).to_string())
    print("table:", json.load(open(run / "table_v0.json")))
    val = summarize(run, "valid")
    val.to_csv(out / "validation.csv", index=False)
    print(f"\nvalidation: all correct {bool(val.correct.all())}, SM {sorted(val.sm.unique())}")
    r1 = compare(val, "P_v0_tab", "P_v0_c64", "v0 + table vs v0 fixed 64 blocks")
    r1.to_csv(out / "table_vs_c64.csv", index=False)
    med = val.groupby(["impl", "tokens"]).us.median().unstack(0)
    med["x_vllm_c64"] = med.U_vllm / med.P_v0_c64
    med["x_vllm_tab"] = med.U_vllm / med.P_v0_tab
    med.to_csv(out / "vs_vllm.csv")
    print(f"\nvs vLLM (median µs):\n{med.round(3).to_string()}")
    n_eff = int((r1.status == "effective").sum())
    worse = r1[r1.status == "worse"].tokens.tolist()
    print(f"\nexpectation: >= 2 sizes effective -> {n_eff}; 16 tokens vs vLLM with table >= 1.3 -> {med.loc[16, 'x_vllm_tab']:.3f}")
    print("keep rule:", "KEEP (use table" + (f"; but keep 64 blocks at tokens {worse}" if worse else "") + ")" if n_eff >= 1
          else "DROP (no size effective)")


if __name__ == "__main__":
    main()
