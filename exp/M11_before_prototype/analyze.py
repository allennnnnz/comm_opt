"""M11 analysis: A (8-to-8 atomic reduce-scatter), B (vLLM time breakdown), C (NVLink utilization).

Usage: .venv/bin/python exp/M11_before_prototype/analyze.py [results/M11_before_prototype/<timestamp>]
Writes to figures/M11_before_prototype/<timestamp>/.
"""
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
SIZES = [80 << 10, 160 << 10, 320 << 10, 640 << 10, 1280 << 10]
VLLM_ALGO = {80 << 10: 1, 160 << 10: 1, 320 << 10: 2, 640 << 10: 2, 1280 << 10: 2}  # vLLM's own choice at TP=8
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
PHASE_COL = {"start barrier": "#e5873a", "stage 1": "#3987e5", "mid/end barrier": "#9c4a0d", "stage 2": "#0d366b",
             "gap": "#c3c2bc"}


def fmt(b):
    return f"{b / (1 << 20):.2f}M" if b >= 1 << 20 else f"{b >> 10}K"


def rows(f, tag):
    return [line.rstrip("\n").split(",") for line in open(f) if line.startswith(tag + ",")]


def three(series):
    return pd.Series(dict(med=series.median(), lo=series.min(), hi=series.max(), n=series.count()))


def part_a(run):
    it, sm = [], []
    for f in sorted(run.glob("rs_probe_r*.csv")):
        rep = int(re.search(r"_r(\d+)", f.stem).group(1))
        it += [dict(rep=rep, variant=r[1], bytes=int(r[2]), ctas=int(r[3]), us=float(r[5])) for r in rows(f, "it")]
        sm += [dict(rep=rep, variant=r[1], bytes=int(r[2]), ctas=int(r[3]), violations=int(r[4]), timeouts=int(r[5]),
                    err=float(r[6]), diff=float(r[7])) for r in rows(f, "sum")]
    it, sm = pd.DataFrame(it), pd.DataFrame(sm)
    per_launch = it.groupby(["variant", "bytes", "ctas", "rep"]).us.median().reset_index()
    t = per_launch.groupby(["variant", "bytes", "ctas"]).us.apply(three).unstack().reset_index()
    best = t.loc[t.groupby(["variant", "bytes"]).med.idxmin()].reset_index(drop=True)
    q = sm.groupby(["variant", "bytes"]).agg(violations=("violations", "sum"), timeouts=("timeouts", "sum"),
                                              max_err=("err", "max"), diff_frac=("diff", "mean")).reset_index()
    return t, best, q


def part_b(run):
    tm, ph = [], []
    for f in sorted(run.glob("car_*_r*.csv")):
        m = re.match(r"car_(timed|plain)_r(\d+)", f.stem)
        build, rep = m.group(1), int(m.group(2))
        tm += [dict(build=build, rep=rep, algo=int(r[1]), bytes=int(r[2]), blocks=int(r[3]), gpu=int(r[4]),
                    us=float(r[5]), err=float(r[6])) for r in rows(f, "time")]
        if build == "timed":
            ph += [dict(rep=rep, algo=int(r[1]), bytes=int(r[2]), gpu=int(r[3]), call=int(r[4]),
                        c0=float(r[9]), c1=float(r[10]), c2=float(r[11]), c3=float(r[12]), tot=float(r[13]))
                   for r in rows(f, "phase")]
    tm, ph = pd.DataFrame(tm), pd.DataFrame(ph)
    # per launch: per-op time = max over GPUs; per call: the GPU whose slowest block took longest (critical path)
    op = tm.groupby(["build", "rep", "algo", "bytes"]).agg(us=("us", "max"), err=("err", "max")).reset_index()
    crit = ph.loc[ph.groupby(["rep", "algo", "bytes", "call"]).tot.idxmax()]
    cl = crit.groupby(["rep", "algo", "bytes"])[["c0", "c1", "c2", "c3", "tot"]].median().reset_index()
    for c in ("c0", "c1", "c2", "c3", "tot"):
        cl[c] /= 1000.0  # ns -> µs
    cl = cl.merge(op[op.build == "timed"][["rep", "algo", "bytes", "us"]], on=["rep", "algo", "bytes"])
    cl["gap"] = cl.us - cl.tot
    cl["barrier_share"] = (cl.c0 + cl.c2) / (cl.c0 + cl.c1 + cl.c2 + cl.c3)
    cl["start_plus_stage1"] = cl.c0 + cl.c1
    agg = cl.groupby(["algo", "bytes"]).agg(
        start=("c0", "median"), stage1=("c1", "median"), mid=("c2", "median"), stage2=("c3", "median"),
        kernel=("tot", "median"), per_op=("us", "median"), gap=("gap", "median"),
        share=("barrier_share", "median"), share_lo=("barrier_share", "min"), share_hi=("barrier_share", "max"),
        sps=("start_plus_stage1", "median"), sps_lo=("start_plus_stage1", "min"),
        sps_hi=("start_plus_stage1", "max")).reset_index()
    ov = op.pivot_table(index=["algo", "bytes"], columns="build", values="us", aggfunc="median").reset_index()
    ov["overhead"] = ov.timed / ov.plain
    return agg, ov, op


def part_c(run, out):
    """Mean NVLink / SM utilization per NVTX range, aggregated inside SQLite (the table has ~4e8 rows)."""
    rep = run / "gpu_metrics.nsys-rep"
    if not rep.exists():
        return None
    db = out / "gpu_metrics.sqlite"  # derived file: kept with the figures, not in results/
    if not db.exists():
        subprocess.run(["/usr/local/cuda/bin/nsys", "export", "--type", "sqlite", "-o", str(db), str(rep)],
                       capture_output=True)
    want = {"NVLink TX Requests User Data [Throughput %]": "tx_req", "NVLink RX Requests User Data [Throughput %]":
            "rx_req", "NVLink TX Responses User Data [Throughput %]": "tx_rsp",
            "NVLink RX Responses User Data [Throughput %]": "rx_rsp", "SMs Active [Throughput %]": "sm_active"}
    con = sqlite3.connect(db)
    names = pd.read_sql("SELECT DISTINCT typeId, metricId, metricName FROM TARGET_INFO_GPU_METRICS", con)
    names = names[names.metricName.isin(want)]
    ids = ",".join(str(int(m)) for m in names.metricId.unique())
    nv = pd.read_sql("SELECT start, end, COALESCE(text, (SELECT value FROM StringIds WHERE id = textId)) AS name "
                     "FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start", con)
    nv.to_sql("m11_ranges", con, if_exists="replace", index=False)
    agg = pd.read_sql(
        f"SELECT r.name, g.typeId, g.metricId, AVG(g.value) AS mean, COUNT(*) AS n FROM GPU_METRICS g "
        f"JOIN m11_ranges r ON g.timestamp BETWEEN r.start AND r.end WHERE g.metricId IN ({ids}) "
        f"GROUP BY r.name, g.typeId, g.metricId", con)
    con.close()
    agg = agg.merge(names, on=["typeId", "metricId"])
    agg["metric"] = agg.metricName.map(want)
    agg["gpu"] = agg.typeId.map({t: i for i, t in enumerate(sorted(agg.typeId.unique()))})
    rows = []
    for name, g in agg.groupby("name", sort=False):
        rec = dict(range=name, samples_per_gpu=int(g.n.max()))
        for m in want.values():
            v = g[g.metric == m]["mean"]
            if len(v):
                rec[f"{m}_mean"], rec[f"{m}_max_gpu"] = v.mean(), v.max()
        rows.append(rec)
    order = {n: i for i, n in enumerate(nv.name)}
    return pd.DataFrame(rows).sort_values("range", key=lambda c: c.map(order))


def main():
    base = ROOT / "results" / "M11_before_prototype"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if p.is_dir())
    out = ROOT / "figures" / "M11_before_prototype" / run.name
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    t, best, q = part_a(run)
    t.to_csv(out / "a_times.csv", index=False)
    best.to_csv(out / "a_best.csv", index=False)
    q.to_csv(out / "a_quality.csv", index=False)
    print("A. per variant, best block count (median of 3 launch medians, µs):")
    pa = best.pivot(index="bytes", columns="variant", values="med")
    pa.index = [fmt(b) for b in pa.index]
    print(pa.round(2).to_string())
    print(best[["variant", "bytes", "ctas", "med", "lo", "hi"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nA. quality (all block counts, 3 launches):")
    print(q.to_string(index=False, float_format=lambda v: f"{v:.3e}"))

    agg, ov, op = part_b(run)
    agg.to_csv(out / "b_breakdown.csv", index=False)
    ov.to_csv(out / "b_overhead.csv", index=False)
    print("\nB. vLLM critical-path breakdown (µs, median over calls and launches):")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nB. stamping overhead (timed / plain per-op):")
    print(ov.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"max correctness error (vs fp32 host sum): {op.err.max():.2e}")

    # ---- pre-registered checks ----
    print("\n== Checks ==")
    a1 = q.groupby("variant").violations.sum()
    print("A1 violations per variant:", a1.to_dict(), "timeouts:", q.groupby("variant").timeouts.sum().to_dict())
    lim = {"atom_f32": 2 ** -10, "store_slot": 2 ** -10, "atom_f16": 2 ** -8, "atom_bf16": 2 ** -5}
    for v, l in lim.items():
        e = q[q.variant == v].max_err.max()
        print(f"A2 {v}: max err {e:.3e} <= {l:.3e}? {e <= l}")
    for v in lim:
        print(f"A3 {v}: mean differing fraction {q[q.variant == v].diff_frac.mean():.4f}")
    b2 = agg[agg.algo == 2].set_index("bytes")
    a4 = 0
    for b in SIZES:
        x = best[(best.variant == "atom_f16") & (best.bytes == b)].iloc[0]
        y = b2.loc[b]
        ok = x.hi < y.sps_lo
        a4 += ok
        print(f"A4 {fmt(b):>6s}: atom_f16 {x.med:.2f} [{x.lo:.2f},{x.hi:.2f}] (ctas {x.ctas}) vs vLLM2 start+stage1 "
              f"{y.sps:.2f} [{y.sps_lo:.2f},{y.sps_hi:.2f}] -> {'faster' if ok else 'not faster'}")
    print(f"A4: {a4}/5 (need >= 3)")
    for v in ("store_slot", "atom_f32"):
        n = 0
        for b in SIZES:
            x = best[(best.variant == v) & (best.bytes == b)].iloc[0]
            n += x.hi < b2.loc[b].sps_lo
        print(f"   {v}: faster than vLLM2 start+stage1 at {n}/5")
    f16 = best[(best.variant == "atom_f16") & (best.bytes == SIZES[-1])].med.iloc[0]
    slot = best[(best.variant == "store_slot") & (best.bytes == SIZES[-1])].med.iloc[0]
    print(f"A5 1.25M: atom_f16 {f16:.2f} vs store_slot {slot:.2f} -> f16 speed = {slot / f16:.2f} of slot (need >= 0.5)")
    for b in SIZES:
        s2 = b2.loc[b]
        s1 = agg[(agg.algo == 1) & (agg.bytes == b)].iloc[0]
        print(f"B1 {fmt(b):>6s}: 2-stage barrier share {s2.share:.2f} [{s2.share_lo:.2f},{s2.share_hi:.2f}]  "
              f"1-stage {s1.share:.2f} [{s1.share_lo:.2f},{s1.share_hi:.2f}]")
    print("B2 gap per op (µs):", agg.set_index(["algo", "bytes"]).gap.round(2).to_dict())
    print("Overhead <= 1.10:", ov.set_index(["algo", "bytes"]).overhead.le(1.10).to_dict())

    c = part_c(run, out)
    if c is not None:
        c.to_csv(out / "c_metrics.csv", index=False)
        print("\nC. NVLink / SM utilization per NVTX range (% , mean over GPUs; max_gpu = busiest GPU):")
        print(c.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    # figure: vLLM critical-path breakdown per size (vLLM's own algorithm choice) next to atom_f16 / store_slot
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    xs = range(len(SIZES))
    for i, b in enumerate(SIZES):
        r = agg[(agg.algo == VLLM_ALGO[b]) & (agg.bytes == b)].iloc[0]
        bottom = 0
        for lab, val in (("start barrier", r.start), ("stage 1", r.stage1), ("mid/end barrier", r.mid),
                         ("stage 2", r.stage2), ("gap", max(r.gap, 0))):
            ax.bar(i - 0.2, val, 0.36, bottom=bottom, color=PHASE_COL[lab], label=lab if i == 0 else None)
            bottom += val
        f = best[(best.variant == "atom_f16") & (best.bytes == b)].med.iloc[0]
        s = best[(best.variant == "store_slot") & (best.bytes == b)].med.iloc[0]
        ax.bar(i + 0.2, f, 0.18, color="#86b6ef", label="atom_f16 reduce-scatter" if i == 0 else None)
        ax.bar(i + 0.38, s, 0.18, color="#898781", label="store_slot reduce-scatter" if i == 0 else None)
    ax.set_xticks(list(xs), [f"{fmt(b)}\n(vLLM {VLLM_ALGO[b]}-stage)" for b in SIZES])
    ax.set_ylabel("µs", color=INK2, fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_title("M11: vLLM all-reduce critical path (left bars) vs 8-to-8 reduce-scatter probes (right bars)",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_breakdown.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
