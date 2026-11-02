"""M4 analysis: evaluate the pre-registered criteria for F1-F5 and the GPU-set bias check.

Usage: .venv/bin/python exp/M4_repeat/analyze.py [results/M4_repeat/<timestamp>]
Writes to figures/M4_repeat/<timestamp>/.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp" / "lib"))
from nccljson import disjoint, load_run, summarize  # noqa: E402

MB, GB = 1 << 20, 1 << 30
SETS = ("A", "B")
GPU_COLOR = {2: "#86b6ef", 4: "#2a78d6", 8: "#104281"}  # ordinal ramp, validated in M0
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def evaluate(oop: pd.DataFrame, times: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    bw = oop.set_index(["coll", "ngpus", "gpuset", "size"])
    out = []

    def add(name, gs, ok, detail):
        out.append(dict(criterion=name, gpuset=gs, passed=bool(ok), detail=detail))

    for gs in SETS:
        ok, det = True, []
        for s in (16 * MB, 64 * MB):
            a, b = bw.loc[("sendrecv", 8, gs, s)], bw.loc[("sendrecv", 2, gs, s)]
            ok &= a.med <= 0.5 * b.med and disjoint(a, b)
            det.append(f"{s // MB}M: 8GPU {a.med:.1f} [{a.lo:.1f},{a.hi:.1f}] vs 2GPU {b.med:.1f} "
                       f"[{b.lo:.1f},{b.hi:.1f}] ratio {a.med / b.med:.2f}")
        add("F1", gs, ok, "; ".join(det))

        a, b = bw.loc[("alltoall", 8, gs, 4 * GB)], bw.loc[("alltoall", 8, gs, 1 * GB)]
        add("F2a", gs, a.med <= 0.97 * b.med and disjoint(a, b),
            f"4G {a.med:.1f} [{a.lo:.1f},{a.hi:.1f}] vs 1G {b.med:.1f} [{b.lo:.1f},{b.hi:.1f}] ratio {a.med / b.med:.3f}")

        t = times.set_index(["coll", "ngpus", "gpuset", "size", "place"])
        ip, op = t.loc[("alltoall", 2, gs, 1 * GB, "in_place")], t.loc[("alltoall", 2, gs, 1 * GB, "out_of_place")]
        add("F2b", gs, ip.med <= 0.95 * op.med and disjoint(ip, op),
            f"in-place {ip.med:.0f} us [{ip.lo:.0f},{ip.hi:.0f}] vs out-of-place {op.med:.0f} [{op.lo:.0f},{op.hi:.0f}] "
            f"ratio {ip.med / op.med:.3f}")

        ok, det = True, []
        for s in (2 * MB, 4 * MB):
            hits = []
            for c in ("all_gather", "reduce_scatter", "all_reduce"):
                a, b = bw.loc[(c, 2, gs, s)], bw.loc[(c, 8, gs, s)]
                if a.med <= 0.75 * b.med and disjoint(a, b):
                    hits.append(c)
                det.append(f"{c}@{s // MB}M ratio {a.med / b.med:.2f}")
            ok &= len(hits) >= 2
        add("F3", gs, ok, "; ".join(det))

        bc = bw.loc[("broadcast", 2, gs, 2 * GB)]
        best = max(("all_gather", "reduce_scatter", "all_reduce"), key=lambda c: bw.loc[(c, 8, gs, 4 * GB)].med)
        top = bw.loc[(best, 8, gs, 4 * GB)]
        add("F4", gs, bc.med >= 1.10 * top.med and disjoint(bc, top),
            f"broadcast x2 @2G {bc.med:.1f} [{bc.lo:.1f},{bc.hi:.1f}] vs {best} x8 @4G {top.med:.1f} "
            f"[{top.lo:.1f},{top.hi:.1f}] ratio {bc.med / top.med:.3f}")

        c = cv.set_index(["coll", "ngpus", "gpuset", "size"])
        hits, det = 0, []
        for s in (1 * MB, 2 * MB, 4 * MB, 8 * MB, 16 * MB):
            b, r = c.loc[("broadcast", 8, gs, s)].med, c.loc[("all_reduce", 8, gs, s)].med
            hits += b >= 20 and b >= 2 * r
            det.append(f"{s // MB}M cv {b:.1f}% vs all_reduce {r:.1f}%")
        add("F5", gs, hits >= 3, f"{hits}/5 sizes; " + "; ".join(det))

    for (coll, ng, s), g in oop.groupby(["coll", "ngpus", "size"]):
        if set(g.gpuset) != set(SETS):
            continue
        a, b = g[g.gpuset == "A"].iloc[0], g[g.gpuset == "B"].iloc[0]
        gap = abs(a.med - b.med) / max(a.med, b.med)
        if gap > 0.10 and disjoint(a, b):
            add("gpuset_bias", "A/B", True, f"{coll} x{ng} @{s / MB:g}M: A {a.med:.1f} vs B {b.med:.1f} ({100 * gap:.0f}%)")
    return pd.DataFrame(out)


def plot(oop, out):
    panels = [("sendrecv", "F1"), ("alltoall", "F2"), ("all_gather", "F3/F4"),
              ("reduce_scatter", "F3/F4"), ("all_reduce", "F3/F4"), ("broadcast", "F4/F5")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True, facecolor=SURFACE)
    for ax, (coll, tag) in zip(axes.flat, panels):
        ax.set_xscale("log", base=2)
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v / GB:g}G" if v >= GB else f"{v / MB:g}M"))
        ax.axhline(300, color=MUTED, linestyle="--", linewidth=1)
        for (ng, gs), g in oop[oop.coll == coll].groupby(["ngpus", "gpuset"]):
            g = g.sort_values("size")
            ls = "-" if gs == "A" else (0, (4, 2))
            ax.fill_between(g["size"], g.lo, g.hi, color=GPU_COLOR[ng], alpha=0.18, linewidth=0)
            ax.plot(g["size"], g.med, color=GPU_COLOR[ng], linewidth=2, linestyle=ls, label=f"{ng} GPUs, set {gs}")
        ax.set_title(f"{coll}  ({tag})", color=INK, fontsize=10, loc="left")
        ax.set_ylim(0, 320)
        ax.set_ylabel("busbw (GB/s)", color=INK2, fontsize=9)
        ax.set_xlabel("message size", color=INK2, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("M4: median of 5 launches (line) and min–max (band); solid = GPU set A, dashed = set B; "
                 "dashed gray = NVLink 300 GB/s", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_confirm.png", dpi=150)
    plt.close(fig)


def main():
    base = ROOT / "results" / "M4_repeat"
    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if p.is_dir())
    out = ROOT / "figures" / "M4_repeat" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)

    df = load_run(run_dir)
    df.to_csv(out / "runs.csv", index=False)
    keys = ["coll", "ngpus", "gpuset", "size"]
    oop = summarize(df[df.place == "out_of_place"], keys)
    times = summarize(df, keys + ["place"], "time_us")
    cv = summarize(df[df.place == "out_of_place"], keys, "cv_pct")
    oop.to_csv(out / "busbw_summary.csv", index=False)

    verdicts = evaluate(oop, times, cv)
    verdicts.to_csv(out / "verdicts.csv", index=False)
    plot(oop, out)

    sched = pd.read_csv(run_dir / "schedule.tsv", sep="\t", comment="#")
    print(f"run {run_dir.name}: {len(sched)} launches, rc!=0: {(sched.rc != 0).sum()}, "
          f"wrong results: {(df.nwrong > 0).sum()}, min reps per point: {oop.n.min()}")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 200)
    print(verdicts.to_string(index=False))


if __name__ == "__main__":
    main()
