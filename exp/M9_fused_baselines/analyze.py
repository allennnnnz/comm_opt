"""M9 analysis: fused vs unfused all-reduce + residual + RMSNorm on 8x A100; pre-registered criteria P1-P4.

Usage: .venv/bin/python exp/M9_fused_baselines/analyze.py [results/M9_fused_baselines/<timestamp>]
Writes to figures/M9_fused_baselines/<timestamp>/.

Per launch, an op's time is the max over the 8 ranks (torch: CUDA-graph back-to-back; LMDeploy: the added
back-to-back timing, 'b2b'). Across the 3 launches: median and [min, max].
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
N, L0, B = 8, 5.39, 231.5e9  # bound parameters fixed in M6
FOCUS_H, FOCUS_T = 5120, [8, 16, 32, 64, 128]
FUSED = ["F_trt_one", "F_trt_two", "F_lmd_auto", "F_lmd_force"]  # the card's "fused" set (F_ prefix)
COLORS = {"U_vllm": "#898781", "U_nccl": "#c3c2bc", "F_trt_one": "#3987e5", "F_trt_two": "#0d366b",
          "F_lmd_auto": "#e5873a", "F_lmd_force": "#9c4a0d", "L_ar": "#c9a4e0"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def bound_us(nbytes):
    return L0 + 2 * (N - 1) / N * nbytes / B * 1e6


def load(run: Path) -> pd.DataFrame:
    rows = []
    for f in run.glob("torch_h*_r*.json"):
        rep = int(re.search(r"_r(\d+)$", f.stem).group(1))
        for r in json.load(open(f))["rows"]:
            if r["error"].startswith("SKIP"):
                continue
            ok = r["correct"] and not r["error"]
            rows.append(dict(impl=r["impl"], hidden=r["hidden"], tokens=r["tokens"], bytes=r["bytes"], rep=rep,
                             us=max(r["per_rank_us"]) if ok else float("nan"), iso_us=float("nan"), correct=ok))
    for f in run.glob("lmd*_h*_r*.log"):
        m = re.match(r"(lmd|lmdforce)_h(\d+)_r(\d+)$", f.stem)
        kind, hidden, rep = m.group(1), int(m.group(2)), int(m.group(3))
        text = f.read_text()
        recs = {}
        for line in text.splitlines():
            if line.startswith("M9\t"):
                _, test, rk, h, t, iso, b2b = line.split("\t")
                recs.setdefault((test, int(t)), []).append((float(iso), float(b2b)))
        aborted = "abort" in text.lower() or "res_diff" in text
        for (test, t), v in recs.items():
            if test == "allreduce":
                if kind != "lmd":
                    continue  # identical code in both builds; keep one copy as a reference all-reduce
                impl = "L_ar"
            else:
                impl = "F_lmd_auto" if kind == "lmd" else "F_lmd_force"
            ok = len(v) == N and not aborted
            rows.append(dict(impl=impl, hidden=hidden, tokens=t, bytes=t * hidden * 2, rep=rep,
                             us=max(b for _, b in v) if ok else float("nan"), iso_us=max(i for i, _ in v),
                             correct=ok))
    return pd.DataFrame(rows)


def summarize(df):
    s = df.groupby(["impl", "hidden", "tokens", "bytes"]).agg(
        med=("us", "median"), lo=("us", "min"), hi=("us", "max"), n=("us", "count"),
        correct=("correct", "all"), iso_med=("iso_us", "median")).reset_index()
    s["bound"] = s.bytes.map(bound_us)
    s["R"] = s.med / s.bound
    return s


def get(s, impl, h, t):
    g = s[(s.impl == impl) & (s.hidden == h) & (s.tokens == t)]
    return None if g.empty or not g.correct.iloc[0] or g.n.iloc[0] < 3 else g.iloc[0]


def faster(s, a, b, h, t, factor=1.0):
    """'yes' if a is faster than b by >= factor with disjoint [min,max]; 'no' if b is faster (disjoint);
    'tie' if overlapping; 'n/a' if missing."""
    x, y = get(s, a, h, t), get(s, b, h, t)
    if x is None or y is None:
        return "n/a", float("nan")
    ratio = y.med / x.med
    if x.hi < y.lo and ratio >= factor:
        return "yes", ratio
    if x.hi < y.lo:
        return "faster but < factor", ratio
    if y.hi < x.lo:
        return "no", ratio
    return "tie", ratio


def fmt(b):
    return f"{b / (1 << 20):.2f}M" if b >= 1 << 20 else f"{b >> 10}K"


def plot(s, out):
    hs = sorted(s.hidden.unique())
    fig, axes = plt.subplots(1, len(hs), figsize=(6.5 * len(hs), 4.8), facecolor=SURFACE, squeeze=False)
    for ax, h in zip(axes[0], hs):
        g = s[s.hidden == h]
        bx = sorted(g.bytes.unique())
        ax.plot(bx, [bound_us(b) for b in bx], color=INK, linestyle=":", linewidth=1.2, label="attainable (M6 bound)")
        for impl, col in COLORS.items():
            d = g[(g.impl == impl) & g.correct].sort_values("bytes")
            if d.empty:
                continue
            ls = "--" if impl.startswith(("U_", "L_")) else "-"
            ax.plot(d.bytes, d.med, color=col, linestyle=ls, linewidth=2, marker="o", markersize=3.5, label=impl)
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
        ax.set_title(f"hidden {h} (fp16, TP=8)", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("all-reduce message size", color=INK2, fontsize=9)
        ax.set_ylabel("µs per AR + residual + RMSNorm (max over ranks)", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("M9: fused (solid) vs unfused (dashed) on 8× A100 — median of 3, band = [min, max]",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig_fused_vs_unfused.png", dpi=150)
    plt.close(fig)


def main():
    base = ROOT / "results" / "M9_fused_baselines"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if p.is_dir())
    out = ROOT / "figures" / "M9_fused_baselines" / run.name
    out.mkdir(parents=True, exist_ok=True)
    df = load(run)
    df.to_csv(out / "launches.csv", index=False)
    s = summarize(df)
    s.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    bad = s[~s.correct | (s.n < 3)]
    print(f"run {run.name}: {len(df)} rows; incorrect/incomplete: "
          f"{bad[['impl', 'hidden', 'tokens', 'n']].to_dict('records') if len(bad) else 'none'}")

    for h in sorted(s.hidden.unique()):
        piv = s[s.hidden == h].pivot(index="tokens", columns="impl", values="med")
        piv.insert(0, "bytes", [fmt(t * h * 2) for t in piv.index])
        piv["bound"] = [bound_us(t * h * 2) for t in piv.index]
        print(f"\nhidden {h}: median µs (b2b, max over ranks)")
        print(piv.round(1).to_string())
    iso = s[s.impl.isin(["L_ar", "F_lmd_auto", "F_lmd_force"])].pivot_table(
        index=["hidden", "tokens"], columns="impl", values="iso_med")
    print("\nLMDeploy original per-call timing (isolated, max over ranks, median µs):")
    print(iso.round(1).to_string())

    h = FOCUS_H
    print(f"\n== Pre-registered checks (hidden {h}) ==")
    for t in (8, 16, 32):
        print(f"P1 t={t}: F_trt_one faster than U_vllm by >=1.2x? {faster(s, 'F_trt_one', 'U_vllm', h, t, 1.2)}")
    print(f"P1 t=128: F_trt_two faster than F_trt_one? {faster(s, 'F_trt_two', 'F_trt_one', h, 128)}")
    for t in (8, 16):
        print(f"P2 t={t}: F_trt_one faster than F_trt_two? {faster(s, 'F_trt_one', 'F_trt_two', h, t)}")
    for t in (64, 128):
        print(f"P2 t={t}: F_trt_two faster than U_vllm by >=1.1x? {faster(s, 'F_trt_two', 'U_vllm', h, t, 1.1)}")
    for t in (8, 16, 32):
        print(f"P3 t={t}: F_lmd_auto faster than F_lmd_force? {faster(s, 'F_lmd_auto', 'F_lmd_force', h, t)}")

    env_rows = []
    for t in FOCUS_T:
        cands = [r for r in (get(s, i, h, t) for i in FUSED) if r is not None]
        allc = [r for r in (get(s, i, h, t) for i in s.impl.unique()) if r is not None]
        best = min(cands, key=lambda r: r.med)
        best_all = min(allc, key=lambda r: r.med)
        env_rows.append(dict(tokens=t, bytes=fmt(t * h * 2), bound=best.bound, fused_best=best.impl,
                             fused_med=best.med, fused_lo=best.lo, R=best.med / best.bound,
                             R_lo=best.lo / best.bound, overall_best=best_all.impl, overall_med=best_all.med,
                             R_overall=best_all.med / best_all.bound))
    env = pd.DataFrame(env_rows)
    env.to_csv(out / "fused_envelope.csv", index=False)
    print("\nP4 / criterion: best fused per size vs attainable")
    print(env.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    room = int(((env.R >= 1.3) & (env.R_lo >= 1.3)).sum())
    close = int((env.R <= 1.15).sum())
    verdict = "ROOM (prototype worth building)" if room >= 3 else "NO ROOM (drop fusion)" if close >= 3 else "IN BETWEEN (discuss)"
    print(f"sizes with R>=1.3 (min also >=1.3): {room}/5; sizes with R<=1.15: {close}/5 -> {verdict}")
    plot(s, out)


if __name__ == "__main__":
    main()
