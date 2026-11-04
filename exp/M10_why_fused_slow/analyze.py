"""M10 analysis: same bytes, different hidden -> does more blocks make row-per-block fused kernels faster?

Usage: .venv/bin/python exp/M10_why_fused_slow/analyze.py [results/M10_why_fused_slow/<timestamp>]
Writes to figures/M10_why_fused_slow/<timestamp>/.
"""
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
N = 8
TEST_SIZES = [320 << 10, 640 << 10, 1280 << 10]
H1_IMPLS = ["F_trt_one", "F_trt_two"]
CONTROLS = ["U_vllm", "F_lmd_force"]
HCOL = {5120: "#0d366b", 2560: "#3987e5", 1280: "#86b6ef"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def load(run: Path):
    rows, reg = [], []
    for f in run.glob("*.json"):
        m = re.match(r"(torch|reglimit)_h(\d+)_r(\d+)$", f.stem)
        if not m:
            continue
        for r in json.load(open(f))["rows"]:
            if r["error"].startswith("SKIP"):
                continue
            ok = r["correct"] and not r["error"]
            rec = dict(impl=r["impl"], hidden=r["hidden"], tokens=r["tokens"], bytes=r["bytes"], rep=int(m.group(3)),
                       us=max(r["per_rank_us"]) if ok else float("nan"), correct=ok, error=r["error"][:80])
            (reg if m.group(1) == "reglimit" else rows).append(rec)
    for f in run.glob("lmdforce_h*_r*.log"):
        m = re.match(r"lmdforce_h(\d+)_r(\d+)$", f.stem)
        text = f.read_text()
        recs = {}
        for line in text.splitlines():
            if line.startswith("M9\tfused\t"):
                _, _, rk, h, t, iso, b2b = line.split("\t")
                recs.setdefault(int(t), []).append(float(b2b))
        for t, v in recs.items():
            ok = len(v) == N and "abort" not in text.lower()
            rows.append(dict(impl="F_lmd_force", hidden=int(m.group(1)), tokens=t, bytes=t * int(m.group(1)) * 2,
                             rep=int(m.group(2)), us=max(v) if ok else float("nan"), correct=ok, error=""))
    return pd.DataFrame(rows), pd.DataFrame(reg)


def kernels(run: Path) -> pd.DataFrame:
    """Distinct launch configs per kernel family from the nsys traces."""
    out = []
    for rep in sorted(run.glob("nsys_*.nsys-rep")):
        db = rep.with_suffix(".sqlite")
        if not db.exists():
            subprocess.run(["/usr/local/cuda/bin/nsys", "export", "--type", "sqlite", "-o", str(db), str(rep)],
                           capture_output=True)
        con = sqlite3.connect(db)
        q = ("SELECT s.value AS name, k.gridX, k.blockX, k.registersPerThread AS regs, k.deviceId, "
             "k.end - k.start AS ns FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id")
        df = pd.read_sql(q, con)
        con.close()
        fam = {"allreduce_fusion_kernel_oneshot": "F_trt_one", "allreduce_fusion_kernel_twoshot": "F_trt_two",
               "cross_device_reduce": "U_vllm (AR)", "rms_norm": "U_vllm (norm)",
               "AllreduceResidualBiasRMSnorm": "F_lmd_force"}
        df["family"] = df.name.map(lambda n: next((v for k, v in fam.items() if k in n), None))
        df = df[df.family.notna() & (df.deviceId == 0)]
        g = df.groupby(["family", "gridX", "blockX", "regs"]).agg(launches=("ns", "count"),
                                                                    median_us=("ns", lambda x: x.median() / 1e3))
        out.append(g.reset_index().assign(trace=rep.stem))
    return pd.concat(out) if out else pd.DataFrame()


def fmt(b):
    return f"{b / (1 << 20):.2f}M" if b >= 1 << 20 else f"{b >> 10}K"


def main():
    base = ROOT / "results" / "M10_why_fused_slow"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if p.is_dir())
    out = ROOT / "figures" / "M10_why_fused_slow" / run.name
    out.mkdir(parents=True, exist_ok=True)
    df, reg = load(run)
    df.to_csv(out / "launches.csv", index=False)
    s = df.groupby(["impl", "hidden", "tokens", "bytes"]).agg(med=("us", "median"), lo=("us", "min"),
                                                               hi=("us", "max"), n=("us", "count"),
                                                               correct=("correct", "all")).reset_index()
    s.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    bad = s[~s.correct | (s.n < 3)]
    print(f"run {run.name}: {len(df)} rows; incorrect/incomplete: "
          f"{bad[['impl', 'hidden', 'tokens', 'n']].to_dict('records') if len(bad) else 'none'}")
    piv = s.pivot_table(index=["impl", "bytes"], columns="hidden", values="med")
    piv.index = [(i, fmt(b)) for i, b in piv.index]
    print("\nmedian µs (max over ranks), columns = hidden")
    print(piv[[5120, 2560, 1280]].round(1).to_string())

    def cell(impl, b, h):
        g = s[(s.impl == impl) & (s.bytes == b) & (s.hidden == h)]
        return None if g.empty or not g.correct.iloc[0] or g.n.iloc[0] < 3 else g.iloc[0]

    print("\nP1 (H1): hidden 1280 faster than 5120 by >= 1.5x, disjoint")
    p1 = 0
    for impl in H1_IMPLS:
        for b in TEST_SIZES:
            x, y = cell(impl, b, 1280), cell(impl, b, 5120)
            if x is None or y is None:
                print(f"  {impl:10s} {fmt(b):>6s}: n/a")
                continue
            ok = x.hi < y.lo and y.med / x.med >= 1.5
            p1 += ok
            print(f"  {impl:10s} {fmt(b):>6s}: {y.med:6.1f} -> {x.med:6.1f} µs  x{y.med / x.med:.2f}  {'YES' if ok else 'no'}")
    print("P2 (controls): |median(1280) / median(5120) - 1| <= 15%")
    p2 = 0
    for impl in CONTROLS:
        for b in TEST_SIZES:
            x, y = cell(impl, b, 1280), cell(impl, b, 5120)
            if x is None or y is None:
                print(f"  {impl:12s} {fmt(b):>6s}: n/a")
                continue
            ok = abs(x.med / y.med - 1) <= 0.15
            p2 += ok
            print(f"  {impl:12s} {fmt(b):>6s}: {y.med:6.1f} -> {x.med:6.1f} µs  ratio {x.med / y.med:.2f}  {'ok' if ok else 'NO'}")
    verdict = ("H1 HOLDS" if p1 >= 4 and p2 >= 5 else "H1 DOES NOT HOLD" if p1 <= 2 else "IN BETWEEN (discuss)")
    print(f"P1 {p1}/6, P2 {p2}/6 -> {verdict}")

    print("\nP3 (register limit), F_trt_one tokens=8:")
    if len(reg):
        print(reg.groupby("hidden").agg(launched_ok=("correct", "sum"), runs=("correct", "count"),
                                        first_error=("error", "first"), med_us=("us", "median")).to_string())
    k = kernels(run)
    if len(k):
        k.to_csv(out / "kernel_launch_configs.csv", index=False)
        print("\nP4: launch configs on GPU 0 (nsys)")
        print(k.drop(columns="trace").to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True, facecolor=SURFACE)
    for ax, impl in zip(axes, H1_IMPLS + CONTROLS):
        for h, col in HCOL.items():
            d = s[(s.impl == impl) & (s.hidden == h) & s.correct].sort_values("bytes")
            ax.plot(d.bytes, d.med, color=col, linewidth=2, marker="o", markersize=4, label=f"hidden {h}")
            ax.fill_between(d.bytes, d.lo, d.hi, color=col, alpha=0.15, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(int(v))))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(impl + ("  (control)" if impl in CONTROLS else ""), color=INK, fontsize=10, loc="left")
        ax.set_xlabel("message size (fp16)", color=INK2, fontsize=9)
    axes[0].set_ylabel("µs per op (max over ranks)", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("M10: same bytes, smaller hidden = more blocks for row-per-block kernels (TRT-LLM fused); "
                 "median of 3, band = [min, max]", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "fig_hidden_sweep.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
