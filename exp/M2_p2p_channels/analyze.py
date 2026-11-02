"""M2-P2P analysis: check the channel model and the pre-registered criteria for F1.

Usage: .venv/bin/python exp/M2_p2p_channels/analyze.py [results/M2_p2p_channels/<timestamp>]
Writes to figures/M2_p2p_channels/<timestamp>/.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp" / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nccljson import disjoint, load_run, summarize  # noqa: E402
from predict_channels import VARIANTS, channels, per_op_bytes  # noqa: E402

MB = 1 << 20
ORDER = list(VARIANTS)
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))  # validated, adjacent pairs
MARKER = dict(zip(ORDER, ["o", "s", "^", "D", "v"]))  # secondary encoding (contrast WARN relief)
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def model_check(df: pd.DataFrame) -> pd.DataFrame:
    obs = (df[df.place == "out_of_place"].groupby(["coll", "ngpus", "variant", "size"]).channels
           .agg(lambda s: sorted(set(s))).reset_index())
    obs["predicted"] = [channels(per_op_bytes(r.coll, r["size"], r.ngpus), r.ngpus, **VARIANTS[r.variant])
                        for _, r in obs.iterrows()]
    obs["match"] = [o == [p] for o, p in zip(obs.channels, obs.predicted)]
    return obs


def evaluate(bw: pd.DataFrame) -> pd.DataFrame:
    b = bw.set_index(["coll", "ngpus", "variant", "size"])
    out = []
    best = {}
    for s in (16 * MB, 64 * MB):
        target, base = b.loc[("sendrecv", 2, "default", s)], b.loc[("sendrecv", 8, "default", s)]
        for v in ORDER:
            r = b.loc[("sendrecv", 8, v, s)]
            out.append(dict(check="sendrecv8_vs_2gpu", size_MiB=s // MB, variant=v, med=r.med, lo=r.lo, hi=r.hi,
                            pct_of_2gpu_default=100 * r.med / target.med, x_8gpu_default=r.med / base.med,
                            disjoint_from_default=disjoint(r, base) if v != "default" else None))
        best[s] = max(ORDER, key=lambda v: b.loc[("sendrecv", 8, v, s)].med)
    res = pd.DataFrame(out)

    def pct(v, s):
        return res[(res.variant == v) & (res.size_MiB == s // MB)].pct_of_2gpu_default.iloc[0]

    def dis(v, s):
        return bool(res[(res.variant == v) & (res.size_MiB == s // MB)].disjoint_from_default.iloc[0])

    full = [v for v in ORDER[1:] if all(pct(v, s) >= 90 and dis(v, s) for s in (16 * MB, 64 * MB))]
    partial = any(50 <= pct(best[s], s) < 90 for s in (16 * MB, 64 * MB))
    not_cfg = all(res[(res.variant == best[s]) & (res.size_MiB == s // MB)].x_8gpu_default.iloc[0] < 1.5
                  for s in (16 * MB, 64 * MB))
    verdict = pd.DataFrame([
        dict(criterion="F1 fully a default-value problem", passed=bool(full), detail=f"variants: {full}"),
        dict(criterion="F1 partially a default-value problem", passed=partial,
             detail=f"best@16M {best[16 * MB]} {pct(best[16 * MB], 16 * MB):.0f}%, "
                    f"best@64M {best[64 * MB]} {pct(best[64 * MB], 64 * MB):.0f}% of 2-GPU default"),
        dict(criterion="F1 not a default-value problem", passed=not_cfg, detail="best < 1.5x 8-GPU default at both sizes"),
    ])

    side = []
    for (coll, ng), ref_v in ((("alltoall", 8), "default"), (("sendrecv", 2), "default")):
        for v in ORDER[1:]:
            for s in sorted(bw[(bw.coll == coll) & (bw.ngpus == ng)]["size"].unique()):
                if (coll, ng, v, s) not in b.index:
                    continue
                r, ref = b.loc[(coll, ng, v, s)], b.loc[(coll, ng, ref_v, s)]
                if r.med < 0.95 * ref.med and disjoint(r, ref):
                    side.append(dict(coll=coll, ngpus=ng, variant=v, size_MiB=s / MB,
                                     med=r.med, default_med=ref.med, change_pct=100 * (r.med / ref.med - 1)))
    return res, verdict, pd.DataFrame(side)


def plot(bw, ch, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    specs = [("sendrecv", 8, "busbw (GB/s)", "sendrecv ×8 vs settings"),
             ("alltoall", 8, "busbw (GB/s)", "alltoall ×8 vs settings"),
             ("sendrecv", 8, "channels per op", "sendrecv ×8: channels used (= CTAs / SMs)")]
    for ax, (coll, ng, ylab, title) in zip(axes, specs):
        ax.set_xscale("log", base=2)
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v / 1024 / MB:g}G" if v >= 1024 * MB else f"{v / MB:g}M"))
        src = ch if "channels" in ylab else bw
        col = "channels" if "channels" in ylab else "med"
        for v in ORDER:
            g = src[(src.coll == coll) & (src.ngpus == ng) & (src.variant == v)].sort_values("size")
            if col == "med":
                ax.fill_between(g["size"], g.lo, g.hi, color=COLOR[v], alpha=0.15, linewidth=0)
            ax.plot(g["size"], g[col], color=COLOR[v], linewidth=2, marker=MARKER[v], markersize=5, label=v)
        if coll == "sendrecv" and col == "med":
            ref = bw[(bw.coll == "sendrecv") & (bw.ngpus == 2) & (bw.variant == "default")].sort_values("size")
            ax.plot(ref["size"], ref.med, color=MUTED, linewidth=1.5, linestyle="--", label="2 GPUs, default")
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_ylabel(ylab, color=INK2, fontsize=9)
        ax.set_xlabel("message size", color=INK2, fontsize=9)
        ax.set_ylim(bottom=0)
    for ax in axes:  # 5 series: legend on every panel, no direct labels (they collide)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("M2-P2P: median of 5 launches (line), min–max (band)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_p2p_settings.png", dpi=150)
    plt.close(fig)


def main():
    base = ROOT / "results" / "M2_p2p_channels"
    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if p.is_dir())
    out = ROOT / "figures" / "M2_p2p_channels" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)

    df = load_run(run_dir)
    df.to_csv(out / "runs.csv", index=False)
    bw = summarize(df[df.place == "out_of_place"], ["coll", "ngpus", "variant", "size"])
    bw.to_csv(out / "busbw_summary.csv", index=False)
    mc = model_check(df)
    mc.to_csv(out / "channel_model_check.csv", index=False)
    res, verdict, side = evaluate(bw)
    res.to_csv(out / "sendrecv8_vs_2gpu.csv", index=False)
    verdict.to_csv(out / "verdicts.csv", index=False)
    side.to_csv(out / "side_effects.csv", index=False)
    ch = df[df.place == "out_of_place"].groupby(["coll", "ngpus", "variant", "size"]).channels.median().reset_index()
    plot(bw, ch, out)

    sched = pd.read_csv(run_dir / "schedule.tsv", sep="\t", comment="#")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 200)
    print(f"run {run_dir.name}: {len(sched)} launches, rc!=0: {(sched.rc != 0).sum()}, "
          f"wrong results: {(df.nwrong > 0).sum()}")
    print(f"channel model: {mc.match.sum()}/{len(mc)} points match")
    if (~mc.match).any():
        print(mc[~mc.match].to_string(index=False))
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(verdict.to_string(index=False))
    print(f"side effects (>5% worse, disjoint): {len(side)}")
    if len(side):
        print(side.to_string(index=False, float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    main()
