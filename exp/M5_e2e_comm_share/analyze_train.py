"""M5 training analysis: end-to-end step split for torchtitan runs, corrected for profiler overhead.

Per launch: profiled steps 16-20 (ProfilerStep#15-19) give GPU-side times; the denominator is the
unprofiled step time from torchtitan's per-step tps on steps 6-12 (profiler fully off before step 13):
step_s = global_tokens / (8 GPUs x tps per GPU). "Hidden cost" = extra time compute kernels take while
a comm kernel runs, vs the same kernel alone (Lee et al. 2507.03114).

Usage: .venv/bin/python exp/M5_e2e_comm_share/analyze_train.py [results/M5_e2e_comm_share/<timestamp>]
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from traces import COMM_RE, decompose, intersect, length, load_rank, overlap_slowdown, summarize, union  # noqa: E402

GLOBAL_TOKENS = 8 * 2048
STEP_RE = r"^ProfilerStep#\d+$"
LOG_RE = re.compile(r"step: +(\d+) .*?tps: ([\d,]+)")
COLOR = {"compute": "#2a78d6", "hidden": "#1baf7a", "transfer": "#eb6834", "other": "#c3c2b7"}
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"


def unprofiled_step_us(log: Path) -> float:
    tps = [float(t.replace(",", "")) for s, t in LOG_RE.findall(re.sub(r"\x1b\[[0-9;]*m", "", log.read_text()))
           if 6 <= int(s) <= 12]
    return 1e6 * GLOBAL_TOKENS / (8 * float(np.median(tps)))


def exposed_by_type(f: Path) -> dict:
    """Exposed (not overlapped by compute) comm time per collective type, over the captured steps."""
    import gzip, json
    ev = json.load(gzip.open(f))["traceEvents"]
    _, _, steps, _ = load_rank(f, STEP_RE, "cadence", merge_by_name=True)
    s0, s1 = steps[0][0], steps[-1][1]
    k = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("dur", 0) > 0
         and s0 <= e["ts"] < s1]
    comp = union(np.array([(e["ts"], e["ts"] + e["dur"]) for e in k if not COMM_RE.search(e["name"])],
                          dtype=float).reshape(-1, 2))
    out = {}
    for e in k:
        if e["cat"] != "kernel" or not COMM_RE.search(e["name"]):
            continue
        iv = np.array([[e["ts"], e["ts"] + e["dur"]]])
        typ = re.sub(r"ncclDevKernel_|_RING.*|_TREE.*|\(.*", "", e["name"])
        out[typ] = out.get(typ, 0.0) + e["dur"] - length(intersect(iv, comp))
    return {t: v / len(steps) for t, v in out.items()}, (s0, s1), len(steps)


def analyze_run(d: Path) -> dict:
    files = sorted((d / "profiling/traces/iteration_20").glob("rank*_trace.json.gz"))
    df, coll, _ = decompose(files, STEP_RE, "cadence", merge_by_name=True)
    s = summarize(df)
    t = unprofiled_step_us(d / "run.log")
    per_rank = df.groupby("rank").mean(numeric_only=True)
    compute = float((per_rank.compute_only_us + per_rank.overlap_us).mean())
    transfer = float(per_rank.exposed_transfer_us.mean())
    hidden, by_type = [], {}
    for f in files:
        types, window, n = exposed_by_type(f)
        for k, v in types.items():
            by_type[k] = by_type.get(k, 0.0) + v / len(files)
        hidden.append(overlap_slowdown(f, window)["extra_us"] / n)
    hidden_us = float(np.mean(hidden))
    m = re.match(r"train_llama3_(\w+)_tp(\d)_dp(\d)_r(\d)", d.name)
    row = dict(model=m.group(1), tp=int(m.group(2)), dp=int(m.group(3)), rep=int(m.group(4)),
               unprofiled_step_us=t, profiled_step_us=s["step_wall_us"],
               compute_pct=100 * (compute - hidden_us) / t, hidden_cost_pct=100 * hidden_us / t,
               transfer_pct=100 * transfer / t, other_pct=100 * (1 - (compute + transfer) / t),
               comm_cost_pct=100 * (transfer + hidden_us) / t,
               raw_comm_kernel_pct=s["comm_total_pct"], raw_overlap_pct=s["overlap_pct"],
               raw_wait_pct=s["exposed_wait_pct"], raw_idle_pct=s["idle_pct"])
    row.update({f"exposed_{k}_pct": 100 * v / t for k, v in by_type.items()})
    return row


def plot(agg: pd.DataFrame, out: Path):
    order = agg.sort_values(["model", "tp"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(order) + 1.6), facecolor=SURFACE)
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for part, col, label in (("compute", "compute_pct", "computation"),
                             ("hidden", "hidden_cost_pct", "compute slowdown while overlapped (hidden comm cost)"),
                             ("transfer", "transfer_pct", "communication not hidden (actual transfer)"),
                             ("other", "other_pct", "other: CPU overhead, waiting, idle")):
        v = order[col].clip(lower=0).to_numpy()
        ax.barh(y, v, left=left, color=COLOR[part], height=0.7, label=label, edgecolor=SURFACE, linewidth=2)
        left += v
    for yi, (_, r) in zip(y, order.iterrows()):
        ax.text(r.compute_pct + r.hidden_cost_pct + r.transfer_pct / 2, yi, f"{r.transfer_pct:.0f}%",
                ha="center", va="center", fontsize=7, color=INK)
        ax.text(101, yi, f"{r.unprofiled_step_us / 1e3:.0f} ms/step", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [f"Llama-3 {r.model.upper()}  TP={r.tp} × FSDP={r.dp}" for _, r in order.iterrows()],
                  fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xlabel("share of end-to-end training step time (%)", color=INK2, fontsize=9)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.suptitle("M5 training (torchtitan, 8 GPUs, 8 × 2048 tokens/step): where each step's time goes — median of 3",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig_train_breakdown.png", dpi=150)
    plt.close(fig)


def main():
    base = ROOT / "results" / "M5_e2e_comm_share"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir()
                                                                   if (p / "schedule.txt").exists())
    out = ROOT / "figures" / "M5_e2e_comm_share" / run.name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(run.glob("train_*_r*")):
        if (d / "profiling/traces/iteration_20").exists():
            rows.append(analyze_run(d))
        else:
            print(f"skip {d.name}: no traces", file=sys.stderr)
    df = pd.DataFrame(rows).fillna(0.0)
    df.to_csv(out / "train_launches.csv", index=False)
    agg = df.drop(columns="rep").groupby(["model", "tp", "dp"]).median().reset_index()
    spread = df.groupby(["model", "tp", "dp"]).transfer_pct.agg(["min", "max"]).reset_index()
    agg = agg.merge(spread, on=["model", "tp", "dp"]).rename(columns={"min": "transfer_pct_min",
                                                                      "max": "transfer_pct_max"})
    agg.to_csv(out / "train_summary.csv", index=False)
    plot(agg, out)
    pd.set_option("display.width", 250)
    print(agg.sort_values(["model", "tp"]).to_string(index=False, float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    main()
