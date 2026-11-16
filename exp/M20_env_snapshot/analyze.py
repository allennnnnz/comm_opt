#!/usr/bin/env python3
"""M20 analysis: node comparison table + clock table + figures. All numbers computed here.
Usage: python3 analyze.py [results_dir]   (default results/M20_env_snapshot)
Writes: results/M20_env_snapshot/analysis/{node_table.md,clock_table.md,clock_rounds.csv}, figures/M20_clock_<job>.png
"""
import sys, re, glob, os, csv, statistics as st
from datetime import datetime
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "M20_env_snapshot")
AN = os.path.join(RES, "analysis"); os.makedirs(AN, exist_ok=True)
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)

def rd(p):
    try: return open(p, errors="replace").read()
    except OSError: return ""

def first(rx, s, g=1, flags=0):
    m = re.search(rx, s, flags); return m.group(g).strip() if m else "?"

def node_row(d):
    s = os.path.join(d, "snapshot")
    env = rd(os.path.join(d, "job_env.txt"))
    smi = rd(os.path.join(s, "nvidia-smi_q.txt"))
    topo = rd(os.path.join(s, "nvidia-smi_topo.txt"))
    lines = [l for l in topo.splitlines() if l.startswith("GPU")]
    gg = set()
    for i, l in enumerate(lines[:8]):
        cells = l.split()[1:9]
        gg.update(c for j, c in enumerate(cells) if j != i)
    nvs = rd(os.path.join(s, "nvidia-smi_nvlink_s.txt"))
    per_gpu = {}; cur = None
    for l in nvs.splitlines():
        m = re.match(r"GPU (\d+):", l)
        if m: cur = m.group(1); per_gpu[cur] = []
        m = re.search(r"Link (\d+): ([\d.]+) GB/s", l)
        if m and cur is not None: per_gpu[cur].append(float(m.group(2)))
    nvl = sorted({(len(v), tuple(sorted(set(v)))) for v in per_gpu.values()})
    ib = rd(os.path.join(s, "ibstat.txt"))
    ports = re.findall(r"State: (\w+).*?Rate: (\d+)", ib, re.S)
    ndr_active = sum(1 for st_, r in ports if st_ == "Active" and r == "400")
    q = rd(os.path.join(s, "nvidia-smi_L_persistence_mig.txt"))
    rows = [l.split(", ") for l in q.splitlines() if re.match(r"^\d+, ", l)]
    def col(i): return sorted({r[i].strip() for r in rows if len(r) > i})
    nv = rd(os.path.join(d, "nvls", "mcast_query.txt"))
    mc = sorted(set(re.findall(r"MULTICAST_SUPPORTED=(-?\d)", nv)))
    summ = rd(os.path.join(d, "nvls", "summary.txt"))
    return {
        "dir": os.path.basename(d),
        "node": first(r"node=(\S+)", env),
        "GPU": first(r"Product Name\s*:\s*(.+)", smi),
        "driver": first(r"Driver Version\s*:\s*(\S+)", smi),
        "VBIOS": ";".join(col(4)) if rows else first(r"VBIOS Version\s*:\s*(\S+)", smi),
        "persistence": ";".join(col(1)), "MIG": ";".join(col(2)), "ECC": ";".join(col(3)),
        "GPU-GPU topo cells": ",".join(sorted(gg)),
        "NVLink (links, GB/s)": str(nvl),
        "NDR Active ports": str(ndr_active),
        "PCIe gen/width cur": ";".join(sorted({r[5].strip()+"x"+r[6].strip() for r in rows if len(r) > 6})),
        "peermem": "yes" if "nvidia_peermem" in rd(os.path.join(s, "lsmod_grep.txt")) else "no",
        "MULTICAST_SUPPORTED": ",".join(mc) or "?",
        "NVLS runs (rc/wrong)": " | ".join(l.split(" start=")[0] + " " + l.split("wrong(oop/ip)=")[-1].split(" mode")[0] for l in summ.splitlines()),
    }

def ts(s): return datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
def num(x):
    m = re.search(r"[-\d.]+", x); return float(m.group()) if m else float("nan")

def pct(v, p):
    v = sorted(v); k = (len(v) - 1) * p / 100; f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)

def clock_rounds(d):
    out = []; C = os.path.join(d, "clock")
    marks = {}
    for l in rd(os.path.join(C, "marks.txt")).splitlines():
        m = re.match(r"round=(\d+) load_start_epoch=(\S+) load_end_epoch=(\S+)", l)
        if m: marks[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    for r, (t0, t1) in sorted(marks.items()):
        p = os.path.join(C, f"round{r}_samples.csv")
        rows = list(csv.reader(open(p, errors="replace")))[1:] if os.path.exists(p) else []
        data = {}
        for row in rows:
            if len(row) < 7: continue
            try: t = ts(row[0])
            except ValueError: continue
            data.setdefault(int(row[1]), []).append((t - t0, num(row[2]), num(row[4]), num(row[5]), row[6].strip()))
        per = []
        for g, s in data.items():
            load = [x for x in s if 0 <= x[0] <= t1 - t0]
            stable = [x for x in load if 5 <= x[0] <= 60]
            if len(stable) < 10: continue
            clk = [x[1] for x in stable]; med = st.median(clk)
            rise = None
            for x in load:  # first time from which clock stays >= 98% of median to the end of stable window
                if all(y[1] >= 0.98 * med for y in stable if y[0] >= x[0]) and x[1] >= 0.98 * med:
                    rise = x[0]; break
            reasons = sorted({x[4] for x in load})
            per.append(dict(gpu=g, med=med, p1=pct(clk, 1), p99=pct(clk, 99), rise=rise if rise is not None else float("nan"),
                            reasons=";".join(reasons), pmax=max(x[2] for x in load), tmax=max(x[3] for x in load)))
        if per:
            out.append(dict(round=r, load_s=t1 - t0, max_rise=max(p["rise"] for p in per), min_med=min(p["med"] for p in per),
                            min_p1=min(p["p1"] for p in per), max_p99=max(p["p99"] for p in per),
                            reasons="|".join(sorted({p["reasons"] for p in per})), per_gpu=per, raw=data, t0=t0, t1=t1))
    return out

def plot(d, rounds, name):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; no figures"); return
    fig, ax = plt.subplots(figsize=(10, 5))
    for rd_ in rounds:
        # slowest GPU (min median) per round
        g = min(rd_["per_gpu"], key=lambda p: p["med"])["gpu"]
        s = rd_["raw"][g]
        ax.plot([x[0] for x in s], [x[1] for x in s], lw=0.8, label=f"round {rd_['round']} (GPU {g})")
    ax.axvline(0, color="k", ls=":"); ax.set_xlabel("seconds from load start"); ax.set_ylabel("SM clock (MHz)")
    ax.set_title(f"M20 SM clock under 64 MB all_reduce load, {name} (slowest GPU per round)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"M20_clock_{name}.png"), dpi=130); plt.close(fig)

dirs = sorted(d for d in glob.glob(os.path.join(RES, "2*_job*")) if os.path.isdir(os.path.join(d, "snapshot")))
rows = [node_row(d) for d in dirs]
with open(os.path.join(AN, "node_table.md"), "w") as f:
    if rows:
        keys = [k for k in rows[0]]
        f.write("| item | " + " | ".join(r["dir"] for r in rows) + " |\n|---|" + "---|" * len(rows) + "\n")
        for k in keys[1:]:
            vals = [r[k] for r in rows]; flag = "" if len(set(vals)) == 1 else " **(differs)**"
            f.write(f"| {k}{flag} | " + " | ".join(vals) + " |\n")
crows = []; clines = ["| job dir | round | load s | max rise s | min median MHz | min p1 | max p99 | event reasons |", "|---|---|---|---|---|---|---|---|"]
for d in dirs:
    rr = clock_rounds(d)
    for r in rr:
        crows.append([os.path.basename(d), r["round"], r["load_s"], r["max_rise"], r["min_med"], r["min_p1"], r["max_p99"], r["reasons"]])
        clines.append(f"| {os.path.basename(d)} | {r['round']} | {r['load_s']:.1f} | {r['max_rise']:.2f} | {r['min_med']:.0f} | {r['min_p1']:.0f} | {r['max_p99']:.0f} | {r['reasons']} |")
    if rr:
        clines.append(f"| {os.path.basename(d)} ALL | 5 rounds | | rise med/min/max = {st.median(r['max_rise'] for r in rr):.2f}/{min(r['max_rise'] for r in rr):.2f}/{max(r['max_rise'] for r in rr):.2f} | min median {min(r['min_med'] for r in rr):.0f} / median {st.median(r['min_med'] for r in rr):.0f} | | | |")
        plot(d, rr, os.path.basename(d))
open(os.path.join(AN, "clock_table.md"), "w").write("\n".join(clines) + "\n")
with open(os.path.join(AN, "clock_rounds.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dir", "round", "load_s", "max_rise_s", "min_median_MHz", "min_p1", "max_p99", "reasons"]); w.writerows(crows)
print(f"{len(rows)} node dirs; tables in {AN}")
