"""M15 analysis: bf16 vs fp16 prototype (each against U_vllm of the same dtype, same run); pre-registered checks.

Usage: .venv/bin/python exp/M15_bf16/analyze.py [results/M15_bf16/<timestamp>]
"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MID = [160 << 10, 320 << 10, 640 << 10]
COL = {"float16": "#3987e5", "bfloat16": "#9c4a0d"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def fmt(b):
    return f"{b / (1 << 20):.2f}M" if b >= 1 << 20 else f"{b >> 10}K"


def main():
    base = ROOT / "results" / "M15_bf16"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if p.is_dir())
    out = ROOT / "figures" / "M15_bf16" / run.name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in run.glob("proto_*_h*_r*.json"):
        m = re.match(r"proto_(\w+?)_h(\d+)_r(\d+)$", f.stem)
        for r in json.load(open(f))["rows"]:
            if r["error"].startswith("SKIP"):
                continue
            ok = r["correct"] and not r["error"]
            rows.append(dict(dtype=m.group(1), impl=r["impl"], hidden=r["hidden"], bytes=r["bytes"], rep=int(m.group(3)),
                             us=max(r["per_rank_us"]) if ok else float("nan"), correct=ok, err=r["max_rel_err"],
                             error=r["error"][:80]))
    df = pd.DataFrame(rows)
    df.to_csv(out / "launches.csv", index=False)
    s = df.groupby(["dtype", "impl", "hidden", "bytes"]).agg(med=("us", "median"), lo=("us", "min"), hi=("us", "max"),
                                                              n=("us", "count"), correct=("correct", "all"),
                                                              err=("err", "max")).reset_index()
    s.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    bad = s[~s.correct | (s.n < 3)]
    print(f"run {run.name}: incorrect/incomplete: {bad[['dtype', 'impl', 'hidden', 'bytes', 'n']].to_dict('records') if len(bad) else 'none'}")
    po = s[s.impl.str.startswith("P_ours") & s.correct & (s.n == 3)]
    best = po.loc[po.groupby(["dtype", "hidden", "bytes"]).med.idxmin()].copy()
    best["ctas"] = best.impl.str.split("_c").str[1].astype(int)
    u = s[s.impl == "U_vllm"].set_index(["dtype", "hidden", "bytes"])
    t = s[s.impl.str.startswith("F_trt")].set_index(["dtype", "impl", "hidden", "bytes"])
    res = []
    for _, r in best.iterrows():
        dt = r["dtype"]  # r.dtype would be the Series' own dtype attribute
        v = u.loc[(dt, r.hidden, r.bytes)]
        trt = [t.loc[(dt, i, r.hidden, r.bytes)].med for i in ("F_trt_one", "F_trt_two")
               if (dt, i, r.hidden, r.bytes) in t.index]
        res.append(dict(dtype=dt, hidden=r.hidden, bytes=r.bytes, ctas=r.ctas, ours=r.med, ours_lo=r.lo,
                        ours_hi=r.hi, vllm=v.med, vllm_lo=v.lo, speedup=v.med / r.med, disjoint=r.hi < v.lo,
                        trt_best=min(trt) if trt else float("nan"), err_ours=r.err, err_vllm=v.err))
    res = pd.DataFrame(res).sort_values(["hidden", "bytes", "dtype"])
    res.to_csv(out / "speedup.csv", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    print("\n== Checks ==")
    bf = s[s.dtype == "bfloat16"]
    print(f"P1 bf16 all correct: {bool(bf.correct.all())}; max err bf16 ours {bf[bf.impl.str.startswith('P_')].err.max():.2e}, "
          f"fp16 ours {s[(s.dtype == 'float16') & s.impl.str.startswith('P_')].err.max():.2e}")
    piv = res.pivot_table(index=["hidden", "bytes"], columns="dtype", values="speedup")
    piv["diff"] = (piv.bfloat16 - piv.float16).abs()
    print("P2 |speedup(bf16) - speedup(fp16)| <= 0.1:")
    print(piv.round(3).to_string())
    print(f"P2: {int((piv['diff'] <= 0.1).sum())}/{len(piv)} sizes within 0.1")
    r5 = res[res.hidden == 5120].set_index(["dtype", "bytes"])
    p3 = all(r5.loc[("float16", b), "speedup"] >= 1.3 for b in MID)
    print(f"P3 fp16 160K/320K/640K >= 1.3x: {p3} ({[round(r5.loc[('float16', b), 'speedup'], 3) for b in MID]})")
    ok_bf = bool(bf.correct.all()) and all(r5.loc[("bfloat16", b), "speedup"] >= 1.3 and r5.loc[("bfloat16", b), "disjoint"] for b in MID)
    badbf = (not bf.correct.all()) or any(r5.loc[("bfloat16", b), "speedup"] < 1.15 for b in MID)
    print(f"verdict: {'BF16 USABLE' if ok_bf else 'BF16 PROBLEM' if badbf else 'IN BETWEEN (discuss)'} "
          f"(bf16 mid speedups {[round(r5.loc[('bfloat16', b), 'speedup'], 3) for b in MID]})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=SURFACE)
    for ax, h in zip(axes, (5120, 8192)):
        for dt, col in COL.items():
            d = res[(res.hidden == h) & (res.dtype == dt)].sort_values("bytes")
            ax.plot(d.bytes, d.speedup, color=col, linewidth=2, marker="o", markersize=4, label=f"{dt}: ours / vLLM")
        ax.axhline(1.3, color=MUTED, linestyle=":", linewidth=1)
        ax.axhline(1.0, color=GRID, linewidth=1)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(int(v))))
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(f"hidden {h}: speedup of the prototype over vLLM (same dtype, same run)", color=INK, fontsize=10,
                     loc="left")
        ax.set_xlabel("message size", color=INK2, fontsize=9)
        ax.set_ylabel("vLLM time / prototype time", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_bf16.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
