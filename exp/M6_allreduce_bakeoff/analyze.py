"""M6 analysis: per-implementation all-reduce latency, the pre-registered go/no-go ratio, and a fixed/data split.

Usage: .venv/bin/python exp/M6_allreduce_bakeoff/analyze.py [results/M6_allreduce_bakeoff/<timestamp>]
Writes to figures/M6_allreduce_bakeoff/<timestamp>/.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
N = 8
FOCUS = [64 << 10, 128 << 10, 256 << 10, 512 << 10, 1 << 20]
ORDER = ["nccl_eager", "nccl_graph", "nccl_symm_graph", "torch_nccl", "mscclpp_graph", "vllm_car",
         "symm_one_shot", "symm_two_shot"]
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]))
MARKER = dict(zip(ORDER, ["o", "s", "^", "D", "v", "P", "X", "*"]))  # secondary encoding (contrast WARN relief)
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def peak_busbw() -> float:
    p = pd.read_csv(ROOT / "figures/M0_baseline/20261102_134841/peaks.csv")
    return float(p[(p.coll == "all_reduce") & (p.ngpus == 8)].busbw.iloc[0]) * 1e9


def load(run: Path) -> pd.DataFrame:
    rows = []
    for f in run.glob("all_reduce_g8A_*_r*.json"):
        m = re.match(r"all_reduce_g8A_(\w+)_r(\d+)", f.stem)
        try:
            res = json.load(open(f))["results"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"skip {f.name}: {e}", file=sys.stderr)
            continue
        for r in res:
            if r["size"] > 0:
                rows.append(dict(impl=m.group(1), rep=int(m.group(2)), bytes=r["size"], us=r["out_of_place"]["time"],
                                 correct=r["out_of_place"]["nwrong"] == 0, kernel=r.get("tuning", {}).get("implementation", "")))  # MSCCL++ shim: no tuning info
    # torch_group_r*: fp16; torch_bf16symm_r*: SymmetricMemory one/two-shot rerun in bf16 (no fp16 kernels)
    for f in list(run.glob("torch_group_r*.json")) + list(run.glob("torch_bf16symm_r*.json")):
        rep = int(re.search(r"_r(\d+)", f.stem).group(1))
        for r in json.load(open(f))["rows"]:
            if r["error"]:  # e.g. NotImplementedError for symm_mem ops in fp16
                continue
            rows.append(dict(impl=r["impl"], rep=rep, bytes=r["bytes"], us=max(r["per_rank_us"]), correct=r["correct"],
                             kernel=""))
    df = pd.DataFrame(rows)
    return df[df.groupby(["impl", "rep", "bytes"]).cumcount() == 0]  # tiny sizes can repeat after rounding


def fmt(b):
    return f"{b >> 20}M" if b >= 1 << 20 else f"{b >> 10}K"


def plot(summ, bound, out):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    ax.axvspan(FOCUS[0], FOCUS[-1], color=GRID, alpha=0.5, linewidth=0)
    ax.text(FOCUS[0] * 1.05, 0.97, "decode batch 8–128 (Qwen2.5-32B)", transform=ax.get_xaxis_transform(),
            fontsize=7, color=INK2, va="top")
    for impl in ORDER:
        g = summ[summ.impl == impl].sort_values("bytes")
        if g.empty:
            continue
        ax.fill_between(g.bytes, g.lo, g.hi, color=COLOR[impl], alpha=0.15, linewidth=0)
        ax.plot(g.bytes, g.med, color=COLOR[impl], marker=MARKER[impl], markersize=5, linewidth=2, label=impl)
    b = bound.sort_values("bytes")
    ax.plot(b.bytes, b.bound_us, color=INK, linestyle="--", linewidth=1.5, label="achievable bound (L0 + 1.75·S/B)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(int(v))))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("message size (fp16)", color=INK2, fontsize=9)
    ax.set_ylabel("time per all-reduce (µs, max over ranks)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    ax.set_title("M6: all-reduce on 8× A100 (NVSwitch), CUDA graph unless eager — median of 5 launches, band = min–max",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_bakeoff.png", dpi=150)
    plt.close(fig)


def main():
    base = ROOT / "results" / "M6_allreduce_bakeoff"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if p.is_dir())
    out = ROOT / "figures" / "M6_allreduce_bakeoff" / run.name
    out.mkdir(parents=True, exist_ok=True)
    df = load(run)
    df.to_csv(out / "launches.csv", index=False)
    summ = df.groupby(["impl", "bytes"]).agg(med=("us", "median"), lo=("us", "min"), hi=("us", "max"),
                                            n=("us", "count"), all_correct=("correct", "all"),
                                            kernel=("kernel", "first")).reset_index()
    summ.to_csv(out / "summary.csv", index=False)

    B = peak_busbw()
    smallest = summ.bytes.min()
    L0 = float(summ[summ.bytes == smallest].med.min())
    L0_impl = summ[summ.bytes == smallest].sort_values("med").impl.iloc[0]
    sizes = sorted(summ.bytes.unique())
    bound = pd.DataFrame(dict(bytes=sizes, bound_us=[L0 + 2 * (N - 1) / N * s / B * 1e6 for s in sizes]))
    best = summ.loc[summ.groupby("bytes").med.idxmin(), ["bytes", "impl", "med"]].rename(columns={"impl": "best_impl",
                                                                                                 "med": "best_us"})
    verdict = best.merge(bound, on="bytes")
    verdict["R"] = verdict.best_us / verdict.bound_us
    verdict["R_L0_6us"] = verdict.best_us / (6.0 + 2 * (N - 1) / N * verdict.bytes / B * 1e6)
    verdict.to_csv(out / "ratio.csv", index=False)
    foc = verdict[verdict.bytes.isin(FOCUS)]
    n_big = int((foc.R >= 1.5).sum())
    decision = ("worth implementing" if n_big >= 3 else "gap already closed" if (foc.R <= 1.2).all() else "partial")

    split = []
    for impl, g in summ.groupby("impl"):
        g = g.set_index("bytes")
        if smallest not in g.index:
            continue
        fixed = g.loc[smallest, "med"]
        for s in FOCUS:
            if s in g.index:
                tot = g.loc[s, "med"]
                split.append(dict(impl=impl, bytes=s, total_us=tot, fixed_us=fixed, fixed_share=fixed / tot,
                                  data_us=tot - fixed, data_busbw_GBs=2 * (N - 1) / N * s / max(tot - fixed, 1e-9) / 1e3))
    pd.DataFrame(split).to_csv(out / "fixed_vs_data.csv", index=False)
    plot(summ, bound, out)

    pd.set_option("display.width", 250)
    print(f"run {run.name}: {len(df)} rows; launches per impl: {df.groupby('impl').rep.nunique().to_dict()}")
    print(f"incorrect points: {int((~summ.all_correct).sum())}")
    print(f"B = {B / 1e9:.1f} GB/s, L0 = {L0:.2f} us ({L0_impl} @ {fmt(smallest)})")
    piv = summ.pivot(index="bytes", columns="impl", values="med")[[c for c in ORDER if c in summ.impl.unique()]]
    piv.index = [fmt(i) for i in piv.index]
    print(piv.round(1).to_string())
    print(foc[["bytes", "best_impl", "best_us", "bound_us", "R", "R_L0_6us"]].to_string(index=False,
                                                                                       float_format=lambda v: f"{v:.2f}"))
    print(f"R >= 1.5 at {n_big}/5 focus sizes -> {decision}")
    k = summ[summ.impl == "nccl_symm_graph"][["bytes", "kernel"]]
    print("nccl_symm_graph kernel by size:", dict(zip([fmt(b) for b in k.bytes], k.kernel)))


if __name__ == "__main__":
    main()
