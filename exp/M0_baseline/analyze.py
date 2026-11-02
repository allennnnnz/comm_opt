"""M0 analysis: nccl-tests JSON -> tables, anomaly list, and figures.

Usage: .venv/bin/python exp/M0_baseline/analyze.py [results/M0_baseline/<timestamp>]
Writes to figures/M0_baseline/<timestamp>/ (raw results are never modified, D2).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COLLS = ["all_reduce", "all_gather", "reduce_scatter", "alltoall", "broadcast", "sendrecv"]
NGPUS = [2, 4, 8]
NVLINK_PEAK = 300.0  # GB/s per GPU, one direction: 12 NVLink3 links x 25 GB/s (R8)

# dataviz palettes, validated with validate_palette.js (ordinal / categorical --pairs all)
GPU_COLOR = {2: "#86b6ef", 4: "#2a78d6", 8: "#104281"}
PROTO_COLOR = {"LL": "#2a78d6", "LL128": "#eb6834", "SIMPLE": "#1baf7a", "P2P": "#898781"}  # P2P: neutral, not a series
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def load(run_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(run_dir.glob("*.json")):
        coll, ng = f.stem.rsplit("_g", 1)
        try:
            results = json.load(open(f))["results"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"skip {f.name}: incomplete or invalid JSON ({e})", file=sys.stderr)
            continue
        for r in results:
            for place in ("out_of_place", "in_place"):
                it = r[f"{place}_per_iter"]
                t = r["tuning"]
                rows.append(dict(
                    coll=coll, ngpus=int(ng), size=r["size"], place=place,
                    time_us=r[place]["time"], algbw=r[place]["alg_bw"], busbw=r[place]["bus_bw"],
                    nwrong=r[place]["nwrong"], p50_us=it["p50_us"], p99_us=it["p99_us"],
                    min_us=it["min_us"], max_us=it["max_us"], cv_pct=it["cv_pct"],
                    impl=t["implementation"],
                    # alltoall/sendrecv run as grouped P2P and bypass the collective tuner
                    algo="P2P" if t["implementation"] == "P2P" else t["algo"],
                    proto="P2P" if t["implementation"] == "P2P" else t["proto"],
                    channels=t["#channels"], warps=t["#warps"]))
    df = pd.DataFrame(rows)
    # nccl-tests rounds tiny sizes up to whole elements per rank, so a size can repeat;
    # keep every row in summary.csv but analyze only the first occurrence.
    df["dup"] = df.groupby(["coll", "ngpus", "size", "place"]).cumcount()
    return df


def find_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag points that contradict the M0 card's expectations (candidates for G1)."""
    out = []
    oop = df[df.place == "out_of_place"]
    for (coll, ng), g in oop.groupby(["coll", "ngpus"]):
        g = g.sort_values("size")
        run_max = g.busbw.cummax().shift()
        drop = g[(g["size"] >= 64 << 10) & (g.busbw < 0.9 * run_max)]
        for _, r in drop.iterrows():
            out.append(dict(kind="busbw_drop", coll=coll, ngpus=ng, size=r["size"],
                            detail=f"busbw {r.busbw:.1f} < 90% of prior max {run_max[r.name]:.1f} GB/s "
                                   f"({r.algo}/{r.proto}/{r.channels}ch)"))
        big = g[g["size"] >= 1 << 20]
        for _, r in big[(big.cv_pct > 10) | (big.p99_us > 1.5 * big.p50_us)].iterrows():
            out.append(dict(kind="unstable", coll=coll, ngpus=ng, size=r["size"],
                            detail=f"cv {r.cv_pct:.1f}%, p99/p50 {r.p99_us / r.p50_us:.2f}"))
        sw = g[(g.algo != g.algo.shift()) | (g.proto != g.proto.shift()) | (g.channels != g.channels.shift())].iloc[1:]
        for _, r in sw.iterrows():
            out.append(dict(kind="tuner_switch", coll=coll, ngpus=ng, size=r["size"],
                            detail=f"-> {r.algo}/{r.proto}/{r.channels}ch"))
    # same collective and size, different GPU counts: NVSwitch is non-blocking, so large gaps are suspicious
    for (coll, size), g in oop[oop["size"] >= 1 << 20].groupby(["coll", "size"]):
        if len(g) > 1 and g.busbw.min() < 0.7 * g.busbw.max():
            lo, hi = g.loc[g.busbw.idxmin()], g.loc[g.busbw.idxmax()]
            out.append(dict(kind="gpu_count_gap", coll=coll, ngpus=int(lo.ngpus), size=size,
                            detail=f"{int(lo.ngpus)} GPUs {lo.busbw:.1f} ({lo.channels:g}ch) vs "
                                   f"{int(hi.ngpus)} GPUs {hi.busbw:.1f} GB/s ({hi.channels:g}ch)"))
    both = df.pivot_table(index=["coll", "ngpus", "size"], columns="place", values="time_us").reset_index()
    both = both[both["size"] >= 1 << 20]
    diff = both[(both.in_place / both.out_of_place - 1).abs() > 0.10]
    for _, r in diff.iterrows():
        out.append(dict(kind="inplace_gap", coll=r.coll, ngpus=r.ngpus, size=r["size"],
                        detail=f"in-place/out-of-place time = {r.in_place / r.out_of_place:.2f}"))
    if (df.nwrong > 0).any():
        for _, r in df[df.nwrong > 0].iterrows():
            out.append(dict(kind="WRONG_RESULT", coll=r.coll, ngpus=r.ngpus, size=r["size"], detail=r.place))
    return pd.DataFrame(out)


def fmt_bytes(b: float) -> str:
    for unit, s in (("G", 30), ("M", 20), ("K", 10)):
        if b >= 1 << s:
            return f"{b / (1 << s):g}{unit}"
    return f"{b:g}"


def style(ax, ylabel):
    ax.set_xscale("log", base=2)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt_bytes(v)))
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def plot_busbw(df, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True, facecolor=SURFACE)
    oop = df[df.place == "out_of_place"]
    for ax, coll in zip(axes.flat, COLLS):
        style(ax, "busbw (GB/s)")
        ax.axhline(NVLINK_PEAK, color=MUTED, linestyle="--", linewidth=1)
        ax.text(16, NVLINK_PEAK + 6, "NVLink peak 300 GB/s", color=MUTED, fontsize=7)
        for ng in NGPUS:
            g = oop[(oop.coll == coll) & (oop.ngpus == ng)].sort_values("size")
            if g.empty:
                continue
            ax.plot(g["size"], g.busbw, color=GPU_COLOR[ng], linewidth=2, label=f"{ng} GPUs")
            ax.text(g["size"].iloc[-1] * 1.3, g.busbw.iloc[-1], f"{ng}", color=INK2, fontsize=8, va="center")
        ax.set_title(coll, color=INK, fontsize=10, loc="left")
        ax.set_ylim(0, 330)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="center left")
    for ax in axes[1]:
        ax.set_xlabel("message size (bytes)", color=INK2, fontsize=9)
    fig.suptitle("M0: bus bandwidth vs message size (out-of-place, single run, exploratory)", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_busbw.png", dpi=150)
    plt.close(fig)


def plot_latency(df, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, facecolor=SURFACE)
    small = df[(df.place == "out_of_place") & (df["size"] <= 1 << 20)]
    for ax, coll in zip(axes.flat, COLLS):
        style(ax, "time per op (µs)")
        for ng in NGPUS:
            g = small[(small.coll == coll) & (small.ngpus == ng)].sort_values("size")
            if g.empty:
                continue
            ax.fill_between(g["size"], g.p50_us, g.p99_us, color=GPU_COLOR[ng], alpha=0.18, linewidth=0)
            ax.plot(g["size"], g.p50_us, color=GPU_COLOR[ng], linewidth=2, label=f"{ng} GPUs (p50; band to p99)")
        ax.set_title(coll, color=INK, fontsize=10, loc="left")
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("message size (bytes)", color=INK2, fontsize=9)
    fig.suptitle("M0: small-message latency, 8 B–1 MiB (p50 line, p99 band)", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_latency.png", dpi=150)
    plt.close(fig)


def plot_tuner(df, out):
    oop = df[(df.place == "out_of_place") & (df.impl != "N/A")]  # nccl-tests -U reports N/A for broadcast
    sizes = sorted(oop["size"].unique())
    rows = [(c, n) for c in COLLS for n in NGPUS if not oop[(oop.coll == c) & (oop.ngpus == n)].empty]
    fig, ax = plt.subplots(figsize=(15, 0.42 * len(rows) + 1.6), facecolor=SURFACE)
    for y, (c, n) in enumerate(rows):
        g = oop[(oop.coll == c) & (oop.ngpus == n)].set_index("size")
        for x, s in enumerate(sizes):
            if s not in g.index:
                continue
            r = g.loc[s]
            ax.add_patch(plt.Rectangle((x + 0.04, y + 0.06), 0.92, 0.88,
                                       color=PROTO_COLOR.get(r.proto, MUTED), linewidth=0))
            ink = "#0b0b0b" if r.proto == "SIMPLE" else "#ffffff"
            ax.text(x + 0.5, y + 0.5, f"{r.algo[0]}{r.channels}", ha="center", va="center", fontsize=6, color=ink)
    ax.set_xlim(0, len(sizes))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(sizes)) + 0.5, [fmt_bytes(s) for s in sizes], fontsize=7, color=MUTED, rotation=90)
    ax.set_yticks(np.arange(len(rows)) + 0.5, [f"{c} ×{n}" for c, n in rows], fontsize=8, color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in PROTO_COLOR.values()]
    ax.legend(handles, PROTO_COLOR.keys(), ncol=4, frameon=False, fontsize=8,
              loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.suptitle("M0: NCCL tuner choice per size — color = protocol, text = algorithm initial + #channels "
                 "(broadcast: not reported by nccl-tests)", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_tuner.png", dpi=150)
    plt.close(fig)


def main():
    base = ROOT / "results" / "M0_baseline"
    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if p.is_dir())
    out = ROOT / "figures" / "M0_baseline" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)

    df = load(run_dir)
    df.to_csv(out / "summary.csv", index=False)
    df = df[(df.dup == 0) & (df["size"] > 0)]  # size 0 = sub-element rounding, not a real message

    oop = df[df.place == "out_of_place"]
    peak = oop.loc[oop.groupby(["coll", "ngpus"]).busbw.idxmax(),
                   ["coll", "ngpus", "size", "busbw", "algo", "proto", "channels"]]
    peak["pct_of_nvlink"] = 100 * peak.busbw / NVLINK_PEAK
    lat = oop.loc[oop.groupby(["coll", "ngpus"])["size"].idxmin(), ["coll", "ngpus", "size", "p50_us", "p99_us"]]
    peak.rename(columns={"size": "peak_size"}) \
        .merge(lat.rename(columns={"size": "min_size", "p50_us": "min_size_p50_us", "p99_us": "min_size_p99_us"}),
               on=["coll", "ngpus"]) \
        .to_csv(out / "peaks.csv", index=False)

    anomalies = find_anomalies(df)
    anomalies.to_csv(out / "anomalies.csv", index=False)

    plot_busbw(df, out)
    plot_latency(df, out)
    plot_tuner(df, out)

    print(f"run: {run_dir.relative_to(ROOT)}  ->  {out.relative_to(ROOT)}")
    print(f"points: {len(df)}  collectives x gpus: {df.groupby(['coll', 'ngpus']).ngroups}")
    print(pd.read_csv(out / "peaks.csv").to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    if not anomalies.empty:
        print(anomalies.groupby("kind").size().to_string())


if __name__ == "__main__":
    main()
