"""M5 inference analysis: end-to-end time split per config, corrected for profiler overhead.

GPU-side compute and transfer times come from the profiled window (kernel durations are barely affected
by the profiler); the denominator is the unprofiled step time (decode: full- minus half-length runs;
prefill: vLLM's own per-iteration elapsed time). "Other" = CPU overhead + waiting for other GPUs + idle.

Usage: .venv/bin/python exp/M5_e2e_comm_share/analyze_infer.py [results/M5_e2e_comm_share/<timestamp>]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from traces import decompose, summarize  # noqa: E402

M0_SUMMARY = ROOT / "figures/M0_baseline/20261102_134841/summary.csv"
ITER_RE = re.compile(r"Iteration\(\d+\): (\d+) context requests, (\d+) context tokens, (\d+) generation requests, "
                     r"(\d+) generation tokens, iteration elapsed time: ([\d.]+) ms")
COLOR = {"compute": "#2a78d6", "transfer": "#eb6834", "other": "#c3c2b7"}  # slots 1-2 validated; other = neutral
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def iteration_times(log: Path) -> list:
    return [tuple(map(float, m.groups())) for m in ITER_RE.finditer(log.read_text())]


def nccl_tests_time(nbytes: float, m0: pd.DataFrame) -> float:
    """M0 all_reduce x8 out-of-place time (us) at nbytes, log-log interpolated between measured sizes."""
    g = m0[(m0.coll == "all_reduce") & (m0.ngpus == 8) & (m0.place == "out_of_place") & (m0.dup == 0)
           & (m0["size"] > 0)].sort_values("size")
    return float(np.exp(np.interp(np.log(nbytes), np.log(g["size"]), np.log(g.time_us))))


def analyze_launch(d: Path, m0: pd.DataFrame) -> list:
    meta = json.load(open(d / "meta.json"))
    its = iteration_times(d / "run.log")
    rows = []
    for name, c in meta["configs"].items():
        files = sorted(d.glob(f"{name}_rank*.pt.trace.json.gz"))
        if c["phase"] == "decode":
            pat, mode = rf"^execute_context_0\(0\)_generation_{c['batch']}\({c['batch']}\)$", "cadence"
            t_unprof = 1e6 * float(np.median(c["unprofiled_step_s"]))
            tokens = c["batch"]
            eng = [t for a, b, g, gt, t in its if a == 0 and g == c["batch"]][-50:]
        else:
            pat, mode = rf"^execute_context_1\({c['input_len']}\)_generation_0\(0\)$", "span"
            eng = [t for a, b, g, gt, t in its if a == 1 and b == c["input_len"] and g == 0][-meta["timing_runs"]:]
            t_unprof = 1e3 * float(np.median(eng))
            tokens = c["input_len"]
        df, coll, names = decompose(files, pat, mode)
        s = summarize(df)
        n_steps = df.step.nunique()
        transfer = coll.transfer_us.sum() / n_steps if len(coll) else np.nan
        compute = (df.compute_only_us + df.overlap_us).groupby(df["rank"]).mean().mean()
        msg = tokens * meta["hidden_size"] * 2  # bf16 activations all-reduced after attention and MLP
        per_coll = float(coll.transfer_us.median()) if len(coll) else np.nan
        rows.append(dict(
            model=d.name.split("_")[1], seed=meta["seed"], config=name, phase=c["phase"], tokens_per_step=tokens,
            unprofiled_step_us=t_unprof, profiled_step_us=s["step_wall_us"],
            engine_iter_us=1e3 * float(np.median(eng)) if eng else np.nan,
            compute_pct=100 * compute / t_unprof, transfer_pct=100 * transfer / t_unprof,
            other_pct=100 * (1 - (compute + transfer) / t_unprof),
            raw_comm_kernel_pct=s["comm_total_pct"], raw_wait_pct=s["exposed_wait_pct"], raw_idle_pct=s["idle_pct"],
            overlap_pct=s["overlap_pct"], colls_per_step=len(coll) / n_steps if len(coll) else 0,
            msg_bytes=msg, transfer_per_coll_us=per_coll, nccl_tests_us=nccl_tests_time(msg, m0),
            comm_kernels=";".join(sorted({re.sub(r"[<(].*", "", n).strip() for n in names}))))
    return rows


def plot(agg: pd.DataFrame, out: Path):
    order = agg.sort_values(["model", "phase", "tokens_per_step"])
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(order) + 1.5), facecolor=SURFACE)
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for part, label in (("compute", "computation"), ("transfer", "communication (actual transfer)"),
                        ("other", "other: CPU overhead, waiting for other GPUs, idle")):
        v = order[f"{part}_pct"].clip(lower=0).to_numpy()
        ax.barh(y, v, left=left, color=COLOR[part], height=0.7, label=label, edgecolor=SURFACE, linewidth=2)
        left += v
    for yi, (_, r) in zip(y, order.iterrows()):
        ax.text(r.compute_pct + r.transfer_pct / 2, yi, f"{r.transfer_pct:.0f}%", ha="center", va="center",
                fontsize=7, color=INK)
        ax.text(101, yi, f"{r.unprofiled_step_us / 1e3:.2f} ms/step", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [f"{r.model}  decode, batch {int(r.tokens_per_step)}" if r.phase == "decode"
                      else f"{r.model}  prefill, {int(r.tokens_per_step)} tokens" for _, r in order.iterrows()],
                  fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xlabel("share of end-to-end step time (%)", color=INK2, fontsize=9)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    fig.suptitle("M5 inference (vLLM, TP=8): where each step's time goes — median of 3 launches",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_infer_breakdown.png", dpi=150)
    plt.close(fig)


def main():
    base = ROOT / "results" / "M5_e2e_comm_share"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if (p / "env.txt").exists())
    out = ROOT / "figures" / "M5_e2e_comm_share" / run.name
    out.mkdir(parents=True, exist_ok=True)
    m0 = pd.read_csv(M0_SUMMARY)
    rows = [r for d in sorted(run.glob("infer_*_s*")) if (d / "meta.json").exists() for r in analyze_launch(d, m0)]
    df = pd.DataFrame(rows)
    df.to_csv(out / "infer_launches.csv", index=False)
    num = [c for c in df.columns if df[c].dtype.kind in "fi" and c != "seed"]
    agg = df.groupby(["model", "config", "phase"])[num].median().reset_index()
    spread = df.groupby(["model", "config"])["transfer_pct"].agg(["min", "max"]).reset_index()
    agg = agg.merge(spread, on=["model", "config"]).rename(columns={"min": "transfer_pct_min", "max": "transfer_pct_max"})
    agg["comm_kernels"] = agg.config.map(df.groupby("config").comm_kernels.first())
    agg.to_csv(out / "infer_summary.csv", index=False)
    plot(agg, out)
    pd.set_option("display.width", 250)
    cols = ["model", "config", "unprofiled_step_us", "profiled_step_us", "compute_pct", "transfer_pct",
            "transfer_pct_min", "transfer_pct_max", "other_pct", "raw_comm_kernel_pct", "overlap_pct",
            "colls_per_step", "msg_bytes", "transfer_per_coll_us", "nccl_tests_us"]
    print(agg.sort_values(["model", "phase", "tokens_per_step"])[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(agg[["model", "config", "comm_kernels"]].to_string(index=False))


if __name__ == "__main__":
    main()
