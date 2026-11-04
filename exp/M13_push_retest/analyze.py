"""M13 analysis: rs_push vs rs_push_par vs rs_pull (single mode); large all-gather in steady state.

Usage: .venv/bin/python exp/M13_push_retest/analyze.py [results/M13_push_retest/<timestamp>]
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
M11 = ROOT / "figures/M11_before_prototype/20261104_090459/b_breakdown.csv"
COL = {"ag_pull": "#86b6ef", "ag_pull_rot": "#3987e5", "ag_push": "#0d366b", "rs_pull": "#e5b98a",
       "rs_push": "#9c4a0d", "rs_push_par": "#e5873a"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def fmt(b):
    return f"{b / (1 << 20):.2f}M" if b >= 1 << 20 else f"{b >> 10}K"


def summarize(df, keys, col):
    t = df.groupby(keys + ["ctas"])[col].agg(med="median", lo="min", hi="max", n="count").reset_index()
    return t.loc[t.groupby(keys).med.idxmin()].set_index(keys)


def main():
    base = ROOT / "results" / "M13_push_retest"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if p.is_dir())
    out = ROOT / "figures" / "M13_push_retest" / run.name
    out.mkdir(parents=True, exist_ok=True)
    it, st, sm = [], [], []
    for f in sorted(run.glob("m13_probe_r*.csv")):
        rep = int(re.search(r"_r(\d+)", f.stem).group(1))
        for line in open(f):
            x = line.rstrip().split(",")
            if x[0] == "it":
                it.append(dict(rep=rep, v=x[2], S=int(x[3]), ctas=int(x[4]), us=float(x[6])))
            elif x[0] == "st":
                st.append(dict(rep=rep, v=x[1], S=int(x[2]), ctas=int(x[3]), gpu=int(x[4]), per_op=float(x[5]),
                               kern=float(x[6]), kern_p90=float(x[7])))
            elif x[0] == "sum":
                sm.append(dict(rep=rep, mode=x[1], v=x[2], S=int(x[3]), ctas=int(x[4]), viol=int(x[5]), tmo=int(x[6])))
    it, st, sm = pd.DataFrame(it), pd.DataFrame(st), pd.DataFrame(sm)
    single = summarize(it.groupby(["v", "S", "ctas", "rep"]).us.median().reset_index(), ["v", "S"], "us")
    stl = st.groupby(["v", "S", "ctas", "rep"]).agg(kern=("kern", "max"), per_op=("per_op", "max")).reset_index()
    steady = summarize(stl, ["v", "S"], "kern")
    steady_op = stl.groupby(["v", "S", "ctas"]).per_op.median()
    steady["per_op"] = [steady_op.loc[(v, S, int(c))] for (v, S), c in steady.ctas.items()]
    single.to_csv(out / "single_best.csv")
    steady.to_csv(out / "steady_best.csv")
    pd.set_option("display.width", 220)
    q = sm.groupby(["mode", "v"])[["viol", "tmo"]].sum()
    print(f"run {run.name}\nviolations/timeouts:\n{q.to_string()}")
    for name, t in (("single (in-kernel µs)", single), ("steady (in-kernel µs, median launch)", steady)):
        piv = t.med.unstack(0)
        piv.index = [fmt(s) for s in piv.index]
        print(f"\n{name}, best block count:\n{piv.round(2).to_string()}")
    piv = steady.per_op.unstack(0)
    piv.index = [fmt(s) for s in piv.index]
    print(f"\nsteady per-op (events, µs):\n{piv.round(2).to_string()}")
    b = pd.read_csv(M11)
    b2 = b[b.algo == 2].set_index("bytes")

    print("\n== Checks ==")
    p1 = 0
    for S in SIZES:
        x, y, z = single.loc[("rs_push_par", S)], single.loc[("rs_pull", S)], single.loc[("rs_push", S)]
        ok = x.med <= 1.2 * y.med
        p1 += ok
        print(f"P1 {fmt(S):>6s}: rs_push_par {x.med:.2f} vs rs_pull {y.med:.2f} (x{x.med / y.med:.2f}); rs_push {z.med:.2f}  {'ok' if ok else 'no'}")
    print(f"P1: {p1}/5 (need >= 3)")
    for S in SIZES[3:]:
        x, y = single.loc[("ag_pull_rot", S)], single.loc[("ag_pull", S)]
        print(f"P2 {fmt(S):>6s}: ag_pull {y.med:.2f} -> ag_pull_rot {x.med:.2f}  speedup {y.med / x.med:.2f} (need >= 1.3, disjoint {x.hi < y.lo})")
    for S in SIZES[3:]:
        x = steady.loc[("ag_pull_rot", S)]
        ref = b2.loc[S, "stage2"]
        print(f"P3 {fmt(S):>6s}: steady ag_pull_rot {x.med:.2f} vs 1.3 x vLLM stage2 {1.3 * ref:.2f}  {'ok' if x.med <= 1.3 * ref else 'no'}")
    p4 = 0
    for S in SIZES[3:]:
        x, y = steady.loc[("ag_push", S)], steady.loc[("ag_pull_rot", S)]
        ok = x.med <= 1.2 * y.med
        p4 += ok
        print(f"P4 {fmt(S):>6s}: steady ag_push {x.med:.2f} vs ag_pull_rot {y.med:.2f} (x{x.med / y.med:.2f})  {'ok' if ok else 'no'}")
    p5 = 0
    for S in SIZES[:3]:
        x = steady.loc[("ag_push", S)]
        ref = b2.loc[S, "mid"] + b2.loc[S, "stage2"]
        ok = x.hi < ref
        p5 += ok
        print(f"P5 {fmt(S):>6s}: steady ag_push {x.med:.2f} [{x.lo:.2f},{x.hi:.2f}] vs vLLM mid+stage2 {ref:.2f}  {'faster' if ok else 'no'}")
    for S in SIZES[3:]:
        x = steady.loc[("ag_push", S)]
        ref = b2.loc[S, "mid"] + b2.loc[S, "stage2"]
        print(f"   (info) {fmt(S):>6s}: steady ag_push {x.med:.2f} vs vLLM mid+stage2 {ref:.2f}")
    verdict = ("PUSH FOR ALL SIZES" if p4 == 2 and p5 == 3 else "SWITCH BY SIZE" if p5 == 3 else "DISCUSS")
    print(f"P4 {p4}/2, P5 {p5}/3 -> {verdict}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=SURFACE, sharey=False)
    for ax, (title, t, vs) in zip(axes, (("single mode: reduce-scatter", single, ["rs_pull", "rs_push", "rs_push_par"]),
                                         ("steady mode: all-gather (in-kernel)", steady, ["ag_pull", "ag_pull_rot", "ag_push"]))):
        for v in vs:
            d = t.loc[v].reset_index()
            ax.plot(d.S, d.med, color=COL[v], linewidth=2, marker="o", markersize=4,
                    linestyle="--" if "pull" in v else "-", label=v)
            ax.fill_between(d.S, d.lo, d.hi, color=COL[v], alpha=0.15, linewidth=0)
        if "all-gather" in title:
            ax.plot(SIZES, [b2.loc[S, "stage2"] for S in SIZES], color=MUTED, linestyle=":", label="M11 vLLM stage 2")
            ax.plot(SIZES, [b2.loc[S, "mid"] + b2.loc[S, "stage2"] for S in SIZES], color=INK2, linestyle=":",
                    label="M11 vLLM mid barrier + stage 2")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(int(v))))
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("all-reduce message size (fp16)", color=INK2, fontsize=9)
        ax.set_ylabel("µs, best block count, median of 3", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_m13.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
