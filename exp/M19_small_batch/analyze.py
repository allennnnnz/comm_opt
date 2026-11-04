"""M19 analysis: where the prototype's small-batch time goes, next to vLLM's all-reduce + norm.

Usage: .venv/bin/python exp/M19_small_batch/analyze.py [results/M19_small_batch/<timestamp>]
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
H = 5120
SMALL = [1, 2, 4]
PH = ["barrier", "pull_sum_residual", "norm_reduce", "write_out", "push", "recv_wait", "end"]


def torch_rows(run):
    rows, stamps = [], []
    for f in run.glob("torch_s*.json"):
        s = int(re.search(r"_s(\d+)$", f.stem).group(1))
        for r in json.load(open(f))["rows"]:
            rows.append(dict(seed=s, impl=r["impl"], tokens=r["tokens"], us=max(r["per_rank_us"]),
                             correct=r["correct"] and not r["error"], sm=min(r["sm_mhz"])))
            if r["impl"].startswith("S_"):
                for rank, st in enumerate(r["stamps"]):
                    if not st:
                        continue
                    base = dict(seed=s, tokens=r["tokens"], rank=rank, owner=st["owner"], kernel=st["kernel_us"],
                                b0_total=st["block0_total_us"])
                    base.update({f"crit_{k}": v for k, v in st["phases_us"].items()})
                    base.update({f"b0_{k}": v for k, v in st["block0_phases_us"].items()})
                    stamps.append(base)
    return pd.DataFrame(rows), pd.DataFrame(stamps)


def car_rows(run):
    tm, ph = [], []
    for f in run.glob("car_*_warm_s*.csv"):
        m = re.match(r"car_(timed|plain)_warm_s(\d+)$", f.stem)
        build, s = m.group(1), int(m.group(2))
        for line in open(f):
            x = line.strip().split(",")
            if x[0] == "time":
                tm.append(dict(build=build, seed=s, algo=int(x[1]), bytes=int(x[2]), gpu=int(x[4]), us=float(x[5])))
            elif x[0] == "phase" and build == "timed":
                ph.append(dict(seed=s, algo=int(x[1]), bytes=int(x[2]), gpu=int(x[3]), call=int(x[4]),
                               c0=float(x[9]), c1=float(x[10]), c2=float(x[11]), c3=float(x[12]), tot=float(x[13])))
    return pd.DataFrame(tm), pd.DataFrame(ph)


def main():
    base = ROOT / "results" / "M19_small_batch"
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else max(p for p in base.iterdir() if p.is_dir())
    out = ROOT / "figures" / "M19_small_batch" / run.name
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)
    tr, st = torch_rows(run)
    tr.to_csv(out / "torch_launches.csv", index=False)
    st.to_csv(out / "stamps.csv", index=False)
    print(f"run {run.name}: torch rows {len(tr)}, all correct {bool(tr.correct.all())}, SM {sorted(tr.sm.unique())}")
    per_op = tr.groupby(["impl", "tokens"]).us.median().unstack(0)
    per_op["norm_plus_gap"] = per_op.U_vllm - per_op.A_vllm
    per_op["stamp_overhead"] = per_op.S_v0s_c64 / per_op.P_v0_c64
    per_op["vllm_over_ours"] = per_op.U_vllm / per_op.P_v0_c64
    per_op.to_csv(out / "per_op.csv")
    print("\nper-op µs (median of 5 launches):")
    print(per_op.round(3).to_string())

    # prototype: per tokens, the rank with the longest kernel (median over launches of per-rank medians)
    g = st.groupby(["tokens", "rank"]).median(numeric_only=True).reset_index()
    g["owner"] = g.owner > 0.5  # the median turned the boolean into 0.0 / 1.0
    crit_rank = g.loc[g.groupby("tokens").kernel.idxmax()].set_index("tokens")
    proto = pd.DataFrame({"per_op": per_op.P_v0_c64, "kernel": crit_rank.kernel})
    proto["gap"] = proto.per_op - proto.kernel
    for k in PH:
        proto[k] = crit_rank[f"crit_{k}"]
    proto["crit_rank_owner"] = crit_rank.owner
    proto.to_csv(out / "proto_breakdown.csv")
    print("\nprototype, critical path (rank with the longest kernel; its longest block), µs:")
    print(proto.round(2).to_string())
    own = g[g.owner & (g["rank"] == 0)].set_index("tokens")
    owner_tbl = own[[f"b0_{k}" for k in PH] + ["b0_total"]].rename(columns=lambda c: c[3:])
    owner_tbl.to_csv(out / "owner_block0.csv")
    print("\nprototype, rank 0 block 0 (owns row 0), µs:")
    print(owner_tbl.round(2).to_string())
    non = g[~g.owner].groupby("tokens")[["kernel", "crit_recv_wait", "crit_barrier"]].median()
    if len(non):
        non["wait_share"] = non.crit_recv_wait / non.kernel
        print("\nprototype, ranks without a row (median over them), µs:")
        print(non.round(3).to_string())

    # vLLM all-reduce kernel phases (standalone, warmed up)
    tm, ph = car_rows(run)
    tm.to_csv(out / "car_times.csv", index=False)
    ph.to_csv(out / "car_phases.csv", index=False)
    op = tm.groupby(["build", "seed", "algo", "bytes"]).us.max().reset_index()
    ov = op.groupby(["build", "algo", "bytes"]).us.median().unstack("build")
    ov["overhead"] = ov.timed / ov.plain
    crit = ph.loc[ph.groupby(["seed", "algo", "bytes", "call"]).tot.idxmax()]
    cv = crit.groupby(["algo", "bytes"])[["c0", "c1", "c2", "c3", "tot"]].median() / 1000.0
    cv.columns = ["start_barrier", "stage1", "mid_or_end_barrier", "stage2", "kernel"]
    cv = cv.join(ov)
    cv["gap"] = cv.timed - cv.kernel
    cv.to_csv(out / "vllm_breakdown.csv")
    print("\nvLLM all-reduce (standalone harness, warmed up), µs; vLLM uses 1-stage below 256 KB:")
    print(cv.round(2).to_string())

    print("\n== Checks (tokens 1, 2, 4) ==")
    for t in SMALL:
        b = t * H * 2
        p = proto.loc[t]
        v1 = cv.loc[(1, b)]
        sync = (p.barrier + p.recv_wait) / p.kernel
        print(f"tokens {t}: ours per-op {p.per_op:.2f} = gap {p.gap:.2f} + kernel {p.kernel:.2f} "
              f"[barrier {p.barrier:.2f}, pull {p.pull_sum_residual:.2f}, norm {p.norm_reduce:.2f}, write {p.write_out:.2f}, "
              f"push {p.push:.2f}, recv/wait {p.recv_wait:.2f}, end {p.end:.2f}]")
        print(f"          vLLM per-op {per_op.loc[t, 'U_vllm']:.2f} = all-reduce only {per_op.loc[t, 'A_vllm']:.2f} "
              f"+ norm+gap {per_op.loc[t, 'norm_plus_gap']:.2f}; AR kernel (1-stage) {v1.kernel:.2f} "
              f"[start {v1.start_barrier:.2f}, read {v1.stage1:.2f}, end {v1.mid_or_end_barrier:.2f}]")
        print(f"          P1 barrier+recv_wait share {sync:.2f} (>= 0.5?)  P3 ours kernel - vLLM AR kernel = "
              f"{p.kernel - v1.kernel:.2f} (>= 2?), norm+gap {per_op.loc[t, 'norm_plus_gap']:.2f} (>= 1.5?)  "
              f"P4 stamp overhead {per_op.loc[t, 'stamp_overhead']:.3f} (<= 1.10?)")
    if len(non) and 1 in non.index:
        print(f"P2 tokens 1, ranks without a row: waiting share {non.loc[1, 'wait_share']:.2f} (>= 0.60?)")


if __name__ == "__main__":
    main()
