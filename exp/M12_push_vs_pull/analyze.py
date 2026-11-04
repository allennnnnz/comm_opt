"""M12 analysis: push (LL128) vs pull, and vs M11's vLLM phases and 4-byte store_slot.

Usage: .venv/bin/python exp/M12_push_vs_pull/analyze.py [results/M12_push_vs_pull/<timestamp>]
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SIZES = [80 << 10, 160 << 10, 320 << 10, 640 << 10, 1280 << 10]
M11 = ROOT / "figures/M11_before_prototype/20261104_090459"
COL = {"ag_pull": "#86b6ef", "ag_push": "#0d366b", "rs_pull": "#e5b98a", "rs_push": "#9c4a0d"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def fmt(b):
    return f"{b / (1 << 20):.2f}M" if b >= 1 << 20 else f"{b >> 10}K"


def main():
    base = ROOT / "results" / "M12_push_vs_pull"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if p.is_dir())
    out = ROOT / "figures" / "M12_push_vs_pull" / run.name
    out.mkdir(parents=True, exist_ok=True)
    it, sm = [], []
    for f in sorted(run.glob("ll128_probe_r*.csv")):
        rep = int(re.search(r"_r(\d+)", f.stem).group(1))
        for line in open(f):
            x = line.rstrip().split(",")
            if x[0] == "it":
                it.append(dict(rep=rep, v=x[1], S=int(x[2]), ctas=int(x[3]), us=float(x[5])))
            elif x[0] == "sum":
                sm.append(dict(rep=rep, v=x[1], S=int(x[2]), ctas=int(x[3]), viol=int(x[4]), tmo=int(x[5])))
    it, sm = pd.DataFrame(it), pd.DataFrame(sm)
    pl = it.groupby(["v", "S", "ctas", "rep"]).us.median().reset_index()
    t = pl.groupby(["v", "S", "ctas"]).us.agg(med="median", lo="min", hi="max", n="count").reset_index()
    t.to_csv(out / "times.csv", index=False)
    best = t.loc[t.groupby(["v", "S"]).med.idxmin()].set_index(["v", "S"])
    best.to_csv(out / "best.csv")
    q = sm.groupby("v")[["viol", "tmo"]].sum()
    pd.set_option("display.width", 200)
    piv = best.med.unstack(0)
    piv.index = [fmt(s) for s in piv.index]
    print(f"run {run.name}\nbest block count, median µs:\n{piv.round(2).to_string()}")
    print(best.reset_index()[["v", "S", "ctas", "med", "lo", "hi"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nP1 violations / timeouts:", q.to_dict())

    def ratio_check(a, b, factor):
        n = 0
        for S in SIZES:
            x, y = best.loc[(a, S)], best.loc[(b, S)]
            ok = x.med <= factor * y.med
            n += ok
            print(f"  {fmt(S):>6s}: {a} {x.med:.2f} vs {b} {y.med:.2f}  ratio {x.med / y.med:.2f}  {'ok' if ok else 'no'}")
        return n

    print("P2 ag_push <= 1.2 x ag_pull:")
    p2 = ratio_check("ag_push", "ag_pull", 1.2)
    print(f"P2: {p2}/5")
    print("P3 rs_push <= 1.2 x rs_pull:")
    p3 = ratio_check("rs_push", "rs_pull", 1.2)
    print(f"P3: {p3}/5")
    b = pd.read_csv(M11 / "b_breakdown.csv")
    b2 = b[b.algo == 2].set_index("bytes")
    a_best = pd.read_csv(M11 / "a_best.csv")
    p4 = 0
    print("P4 ag_push faster than M11 vLLM (mid barrier + stage 2):")
    for S in SIZES:
        x = best.loc[("ag_push", S)]
        ref = b2.loc[S, "mid"] + b2.loc[S, "stage2"]
        ok = x.hi < ref
        p4 += ok
        print(f"  {fmt(S):>6s}: {x.med:.2f} [{x.lo:.2f},{x.hi:.2f}] vs {ref:.2f} (stage 2 alone {b2.loc[S, 'stage2']:.2f})  {'faster' if ok else 'no'}")
    print(f"P4: {p4}/5")
    print("P5 rs_push vs M11 4-byte store_slot (need >= 1.5x faster at 640K, 1.25M):")
    for S in SIZES:
        x = best.loc[("rs_push", S)]
        s4 = a_best[(a_best.variant == "store_slot") & (a_best.bytes == S)].med.iloc[0]
        print(f"  {fmt(S):>6s}: rs_push {x.med:.2f} vs store_slot(4B) {s4:.2f}  speedup {s4 / x.med:.2f}")
    print("context: M11 vLLM stage 1 (critical path) and start+stage1:",
          {fmt(S): (round(b2.loc[S, 'stage1'], 2), round(b2.loc[S, 'start'] + b2.loc[S, 'stage1'], 2)) for S in SIZES})
    verdict = ("PUSH FAST ENOUGH" if q.loc["ag_push", "viol"] == 0 and q.loc["rs_push", "viol"] == 0 and p2 >= 3 and p4 >= 3
               else "PUSH NOT FAST ENOUGH" if p2 <= 1 else "IN BETWEEN (discuss)")
    print(f"verdict: {verdict}")

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    for v, c in COL.items():
        d = best.loc[v].reset_index()
        ax.plot(d.S, d.med, color=c, linewidth=2, marker="o", markersize=4,
                linestyle="--" if v.endswith("pull") else "-", label=v)
        ax.fill_between(d.S, d.lo, d.hi, color=c, alpha=0.15, linewidth=0)
    ax.plot(SIZES, [b2.loc[S, "mid"] + b2.loc[S, "stage2"] for S in SIZES], color=MUTED, linestyle=":",
            label="M11 vLLM mid barrier + stage 2")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(int(v))))
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("all-reduce message size (fp16)", color=INK2, fontsize=9)
    ax.set_ylabel("µs (max over blocks and GPUs)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("M12: push (LL128, 16 B) vs pull (16 B) on 8× A100 — best block count, median of 3",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_push_vs_pull.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
