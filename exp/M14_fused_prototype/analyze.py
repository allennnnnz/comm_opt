"""M14 analysis: prototype vs vLLM (U_vllm) and TRT-LLM fused; pre-registered go / no-go.

Usage: .venv/bin/python exp/M14_fused_prototype/analyze.py [results/M14_fused_prototype/<timestamp>]
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
EST = {80 << 10: 12.5, 160 << 10: 12.9, 320 << 10: 13.6, 640 << 10: 17.1, 1280 << 10: 24.9}  # D029 estimate
COL = {"U_vllm": "#898781", "F_trt_one": "#86b6ef", "F_trt_two": "#3987e5", "P_ours": "#9c4a0d"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def fmt(b):
    return f"{b / (1 << 20):.2f}M" if b >= 1 << 20 else f"{b >> 10}K"


def main():
    base = ROOT / "results" / "M14_fused_prototype"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if p.is_dir())
    out = ROOT / "figures" / "M14_fused_prototype" / run.name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in run.glob("proto_h*_r*.json"):
        rep = int(re.search(r"_r(\d+)", f.stem).group(1))
        for r in json.load(open(f))["rows"]:
            if r["error"].startswith("SKIP"):
                continue
            ok = r["correct"] and not r["error"]
            rows.append(dict(impl=r["impl"], hidden=r["hidden"], tokens=r["tokens"], bytes=r["bytes"], rep=rep,
                             us=max(r["per_rank_us"]) if ok else float("nan"), correct=ok, err=r["max_rel_err"],
                             error=r["error"][:80]))
    df = pd.DataFrame(rows)
    df.to_csv(out / "launches.csv", index=False)
    s = df.groupby(["impl", "hidden", "tokens", "bytes"]).agg(med=("us", "median"), lo=("us", "min"), hi=("us", "max"),
                                                               n=("us", "count"), correct=("correct", "all"),
                                                               err=("err", "max")).reset_index()
    s.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    bad = s[~s.correct | (s.n < 3)]
    print(f"run {run.name}: incorrect/incomplete: {bad[['impl', 'hidden', 'tokens', 'n']].to_dict('records') if len(bad) else 'none'}")
    for h in sorted(s.hidden.unique()):
        piv = s[s.hidden == h].pivot(index="bytes", columns="impl", values="med")
        piv.index = [fmt(b) for b in piv.index]
        print(f"\nhidden {h}: median µs per op (max over ranks)\n{piv.round(2).to_string()}")
    # P_ours: best block count per size
    po = s[s.impl.str.startswith("P_ours") & s.correct & (s.n == 3)]
    best = po.loc[po.groupby(["hidden", "bytes"]).med.idxmin()].copy()
    best["ctas"] = best.impl.str.split("_c").str[1].astype(int)
    best["impl"] = "P_ours"
    best.to_csv(out / "ours_best.csv", index=False)
    print("\nP_ours best block count:")
    print(best[["hidden", "bytes", "ctas", "med", "lo", "hi", "err"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    u = s[(s.impl == "U_vllm")].set_index(["hidden", "bytes"])
    print("\nspeedup vs U_vllm (median ratio; disjoint = ours.hi < vllm.lo):")
    res = []
    for _, r in best.iterrows():
        v = u.loc[(r.hidden, r.bytes)]
        sp = v.med / r.med
        dj = r.hi < v.lo
        res.append(dict(hidden=r.hidden, bytes=r.bytes, ours=r.med, vllm=v.med, speedup=sp, disjoint=dj,
                        est=EST.get(r.bytes) if r.hidden == 5120 else None))
        print(f"  h{r.hidden} {fmt(r.bytes):>6s}: ours {r.med:6.2f} (c{r.ctas}) vs vLLM {v.med:6.2f}  x{sp:.2f}  disjoint {dj}"
              + (f"  (estimate {EST[r.bytes]:.1f})" if r.hidden == 5120 else ""))
    res = pd.DataFrame(res)
    res.to_csv(out / "speedup.csv", index=False)
    h5 = res[res.hidden == 5120].set_index("bytes")
    mid = [160 << 10, 320 << 10, 640 << 10]
    go = all(h5.loc[b, "speedup"] >= 1.3 and h5.loc[b, "disjoint"] for b in mid) and bool(po[po.hidden == 5120].correct.all())
    nogo = bool((h5.speedup < 1.15).all())
    print(f"\nverdict: {'GO' if go else 'NO-GO' if nogo else 'IN BETWEEN (discuss)'}")
    print("P1 hidden 8192 launches and is correct:", bool(len(po[po.hidden == 8192]) > 0 and po[po.hidden == 8192].correct.all()))
    t = s[(s.hidden == 5120) & s.impl.str.startswith("F_trt")].set_index(["impl", "bytes"])
    for b in sorted(h5.index):
        trt = [t.loc[(i, b)].med for i in ("F_trt_one", "F_trt_two") if (i, b) in t.index]
        print(f"P3 {fmt(b):>6s}: ours {h5.loc[b, 'ours']:.2f} vs TRT {[round(x, 2) for x in trt]}  "
              f"{'faster than all' if all(h5.loc[b, 'ours'] < x for x in trt) else 'no'}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=SURFACE)
    for ax, h in zip(axes, (5120, 8192)):
        g = s[s.hidden == h]
        for impl, col in COL.items():
            d = (best[best.hidden == h] if impl == "P_ours" else g[(g.impl == impl) & g.correct]).sort_values("bytes")
            if d.empty:
                continue
            ax.plot(d.bytes, d.med, color=col, linewidth=2.2 if impl == "P_ours" else 1.6, marker="o", markersize=4,
                    linestyle="-" if impl == "P_ours" else "--", label="ours (best blocks)" if impl == "P_ours" else impl)
            ax.fill_between(d.bytes, d.lo, d.hi, color=col, alpha=0.15, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(int(v))))
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(f"hidden {h} (fp16, TP=8): all-reduce + residual + RMSNorm", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("message size", color=INK2, fontsize=9)
        ax.set_ylabel("µs per op (CUDA graph, max over ranks)", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_prototype.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
