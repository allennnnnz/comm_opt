"""M7 analysis: NCCL symmetric kernel/CTA regret, CTA scaling, and the best-of-everything envelope vs the M6 bound.

Usage: .venv/bin/python exp/M7_where_slow/analyze.py [results/M7_where_slow/<timestamp>]
Writes to figures/M7_where_slow/<timestamp>/.
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
M6 = ROOT / "figures/M6_allreduce_bakeoff/20261103_083417"
N, L0, B = 8, 5.39, 231.5e9  # bound parameters fixed in M6 (ratio.csv / README)
FOCUS = [64 << 10, 128 << 10, 256 << 10, 512 << 10, 1 << 20]
REGRET_SIZES = [16 << 10, 32 << 10, 64 << 10, 128 << 10, 256 << 10]
CTA_SHOWN = {1: "#86b6ef", 4: "#3987e5", 16: "#1c5cab", 64: "#0d366b"}  # ordinal ramp, validated
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def fmt(b):
    return f"{b >> 20}M" if b >= 1 << 20 else f"{b >> 10}K"


def load(run: Path) -> pd.DataFrame:
    rows = []
    for f in run.glob("all_reduce_g8A_*_r*.json"):
        m = re.match(r"all_reduce_g8A_(\w+?)_r(\d+)$", f.stem)
        try:
            res = json.load(open(f))["results"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"skip {f.name}: {e}", file=sys.stderr)
            continue
        seen = set()
        for r in res:
            if r["size"] == 0 or r["size"] in seen:
                continue
            seen.add(r["size"])
            t = r.get("tuning", {})
            rows.append(dict(config=m.group(1), rep=int(m.group(2)), bytes=r["size"], us=r["out_of_place"]["time"],
                             correct=r["out_of_place"]["nwrong"] == 0, kernel_impl=t.get("implementation", ""),
                             ctas=t.get("#channels", "")))
    for f in run.glob("C_vllm_*_r*.json"):
        m = re.match(r"(C_vllm_\w+?)_r(\d+)$", f.stem)
        for r in json.load(open(f))["rows"]:
            if not r["error"]:
                rows.append(dict(config=m.group(1), rep=int(m.group(2)), bytes=r["bytes"], us=max(r["per_rank_us"]),
                                 correct=r["correct"], kernel_impl="", ctas=""))
    return pd.DataFrame(rows)


def summarize(df):
    return df.groupby(["config", "bytes"]).agg(med=("us", "median"), lo=("us", "min"), hi=("us", "max"),
                                              n=("us", "count"), correct=("correct", "all"),
                                              ctas=("ctas", "first")).reset_index()


def plot_ctas(s, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, facecolor=SURFACE)
    auto = s[s.config == "A_auto"].sort_values("bytes")
    for ax, (short, title) in zip(axes, (("agll", "AGxLL_R (gather then reduce, one-shot)"),
                                         ("rsag", "RSxLD_AGxST (reduce-scatter + all-gather, two-shot)"))):
        ax.axvspan(FOCUS[0], FOCUS[-1], color=GRID, alpha=0.5, linewidth=0)
        for c, col in CTA_SHOWN.items():
            g = s[(s.config == f"A_{short}_c{c}") & s.correct].sort_values("bytes")
            ax.plot(g.bytes, g.med, color=col, linewidth=2, marker="o", markersize=4, label=f"{c} CTAs")
        ax.plot(auto.bytes, auto.med, color=INK, linewidth=1.5, linestyle="--", label="NCCL auto choice")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(int(v))))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("message size (fp16)", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    axes[0].set_ylabel("time per all-reduce (µs, max over ranks)", color=INK2, fontsize=9)
    fig.suptitle("M7: NCCL 2.32.3 symmetric kernels on 8× A100 — time vs CTA count (median of 3); gray = decode batch 8–128",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_cta_sweep.png", dpi=150)
    plt.close(fig)


def main():
    base = ROOT / "results" / "M7_where_slow"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if p.is_dir())
    out = ROOT / "figures" / "M7_where_slow" / run.name
    out.mkdir(parents=True, exist_ok=True)
    df = load(run)
    df.to_csv(out / "launches.csv", index=False)
    s = summarize(df)
    s.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    bad = s[~s.correct].groupby("config").bytes.apply(lambda b: [fmt(x) for x in b])
    print(f"run {run.name}: {df.config.nunique()} configs, {len(df)} rows; incorrect: {bad.to_dict() if len(bad) else 'none'}")

    # A. NCCL symmetric: auto choice vs best forced (kernel, CTAs)
    forced = s[s.config.str.match(r"A_(agll|rsag)_c\d+") & s.correct]
    best = forced.loc[forced.groupby("bytes").med.idxmin()].set_index("bytes")
    auto = s[s.config == "A_auto"].set_index("bytes")
    reg = pd.DataFrame(dict(auto_us=auto.med, auto_ctas=auto.ctas, best_cfg=best.config, best_us=best.med))
    reg["auto_over_best"] = reg.auto_us / reg.best_us
    reg["disjoint"] = [auto.loc[b, "lo"] > best.loc[b, "hi"] for b in reg.index]
    reg.to_csv(out / "nccl_sym_regret.csv")
    hits = sum(bool(reg.loc[b, "auto_over_best"] >= 1.3 and reg.loc[b, "disjoint"]) for b in REGRET_SIZES)
    print("\nA. NCCL SYM auto vs best forced:")
    print(reg.rename(index=fmt).to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"criterion 'NCCL SYM picks badly': {hits}/5 sizes in 16K-256K -> {'HOLDS' if hits >= 3 else 'does not hold'}")
    piv = forced.pivot(index="bytes", columns="config", values="med")
    piv.index = [fmt(b) for b in piv.index]
    piv.to_csv(out / "cta_sweep_table.csv")

    # Envelope over all correct configs in M6 and M7
    m6 = pd.read_csv(M6 / "summary.csv").rename(columns={"impl": "config"})
    m6["config"] = "M6_" + m6.config
    allc = pd.concat([s[s.correct][["config", "bytes", "med"]],
                      m6[m6.all_correct][["config", "bytes", "med"]]]).reset_index(drop=True)
    env = allc.loc[allc.groupby("bytes").med.idxmin()].set_index("bytes")
    env["bound_us"] = [L0 + 2 * (N - 1) / N * b / B * 1e6 for b in env.index]
    env["R_env"] = env.med / env.bound_us
    m6best = m6[m6.all_correct].groupby("bytes").med.min()
    env["M6_best_us"] = m6best
    env["gain_vs_M6_best"] = env.M6_best_us / env.med
    env.to_csv(out / "envelope.csv")
    foc = env.loc[FOCUS]
    n15, all12 = int((foc.R_env >= 1.5).sum()), bool((foc.R_env <= 1.2).all())
    verdict = "selection closes most of the gap" if all12 else "new mechanism needed" if n15 >= 3 else "in between"
    print("\nEnvelope (best correct config, M6 + M7):")
    print(env.rename(index=fmt)[["config", "med", "bound_us", "R_env", "M6_best_us", "gain_vs_M6_best"]]
          .to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"focus 64K-1M: R_env>=1.5 at {n15}/5, all<=1.2: {all12} -> {verdict}")

    # B, C, D side questions
    side = s[s.config.isin(["A_auto", "B_auto", "B_agll", "B_rsag", "C_vllm_1stage", "C_vllm_2stage",
                            "D_mscclpp_sym0", "D_mscclpp_sym1"])]
    sp = side[side.bytes.isin([16 << 10] + FOCUS)].pivot(index="bytes", columns="config", values="med")
    sp.index = [fmt(b) for b in sp.index]
    print("\nB/C/D (median us):")
    print(sp.round(1).to_string())
    plot_ctas(s, out)


if __name__ == "__main__":
    main()
