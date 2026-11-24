#!/usr/bin/env python
"""M25 analysis: prints the numbers needed to judge each prediction in exp/M25_topic_survey/README.md (Opus judges).
Usage: analyze.py <result dir> [--csv-out DIR]   (pure stdlib; works in the vllm or analysis venv)
"""
import csv, glob, gzip, json, os, re, statistics as st, sys

R = sys.argv[1] if len(sys.argv) > 1 else "."
outdir = R


def med(x):
    return st.median(x) if x else float("nan")


def rng(x):
    return f"{min(x):.4g}..{max(x):.4g}" if x else "-"


def table(rows, header, name):
    print(f"\n## {name}")
    print("\t".join(header))
    for r in rows:
        print("\t".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r))
    with open(os.path.join(outdir, f"analysis_{name.split()[0]}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def opengz(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


# ------------------------------------------------------------------ B4
def b4():
    p = os.path.join(R, "b4", "summary.tsv")
    if not os.path.exists(p):
        return
    print("\n## B4 NVLS crash (N1: a1,a2 should show cuda999>0 in >=1; N2: e256 completes with wrong=0)")
    print(open(p).read())


# ------------------------------------------------------------------ B1
def b1():
    p = os.path.join(R, "b1", "overlap.jsonl")
    if not os.path.exists(p):
        print("\n## B1: no overlap.jsonl"); return
    rows = [json.loads(l) for l in open(p) if l.strip()]
    cell = {}
    for r in rows:
        cell.setdefault((r["cfg"], r["op"], r["size_mb"], r["gemm"]), []).append(r)
    M = {}
    out = []
    for k, rs in sorted(cell.items()):
        g = med([r["t_gemm_conc"] / r["t_gemm_alone"] - 1 for r in rs]) * 100
        c = med([r["t_comm_conc"] / r["t_comm_alone"] - 1 for r in rs]) * 100
        m = med([r["makespan"] for r in rs]); ca = med([r["t_comm_alone"] for r in rs])
        ga = med([r["t_gemm_alone"] for r in rs])
        M[k] = dict(g=g, c=c, mk=m, ca=ca)
        out.append(list(k) + [len(rs), ga, ca, g, c, m, rng([r["makespan"] for r in rs])])
    table(out, ["cfg", "op", "size_mb", "gemm", "n", "gemm_alone_ms", "comm_alone_ms", "gemm_slow_pct", "comm_slow_pct",
                "makespan_ms", "makespan_range"], "B1_cells")
    print("\n## B1 judgements")
    k = ("default", "all_reduce", 256, 8192)
    if k in M: print("O1 default all_reduce256 x gemm8192 gemm slowdown %:", round(M[k]["g"], 2))
    for g in (8192, 4096):
        a, b = ("ring_cta32", "all_reduce", 256, g), ("ring_cta4", "all_reduce", 256, g)
        if a in M and b in M and M[b]["g"] > 0:
            print(f"O2 gemm{g}: slow cta32={M[a]['g']:.2f}% cta4={M[b]['g']:.2f}% ratio={M[a]['g']/M[b]['g']:.2f}")
        a, b = ("nvls", "all_reduce", 256, g), ("ring", "all_reduce", 256, g)
        if a in M and b in M:
            print(f"O3 gemm{g}: slow nvls={M[a]['g']:.2f}% ring(default CTAs)={M[b]['g']:.2f}%")
    # O4 per (op,size,gemm)
    n4 = 0; lines = []
    for op, s, g in sorted({(k[1], k[2], k[3]) for k in M}):
        cf = {k[0]: v for k, v in M.items() if k[1:] == (op, s, g)}
        if len(cf) < 2: continue
        ba = min(cf, key=lambda c: cf[c]["ca"]); bm = min(cf, key=lambda c: cf[c]["mk"])
        diff = (cf[ba]["mk"] - cf[bm]["mk"]) / cf[bm]["mk"] * 100
        ok = ba != bm and diff >= 5; n4 += ok
        lines.append([op, s, g, ba, bm, diff, ok, max(v["g"] for v in cf.values())])
    table(lines, ["op", "size_mb", "gemm", "best_comm_alone_cfg", "best_makespan_cfg", "makespan_diff_pct", "O4_cell", "max_gemm_slow_pct"], "B1_O4")
    print("O4 cells with different best cfg and diff>=5%:", n4, "of", len(lines))


# ------------------------------------------------------------------ B2
def b2():
    fs = sorted(glob.glob(os.path.join(R, "b2", "*", "result.json")))
    if not fs:
        print("\n## B2: no result.json"); return
    dec, pre, prof, info = [], [], [], []
    for f in fs:
        r = json.load(open(f)); n = r["name"]
        bk = ""
        ap = os.path.join(os.path.dirname(f), "allreduce_backend.txt")
        if os.path.exists(ap): bk = open(ap).read().strip()[:300]
        info.append([n, r.get("startup_s", float("nan")), r.get("max_num_batched_tokens", ""), len(r["errors"]), bk])
        for b, d in r["decode"].items():
            dec.append([n, b, med(d["ms_per_token"]), rng(d["ms_per_token"]), med(d["t1_s"])])
        for L, d in r["prefill"].items():
            pre.append([n, L, med(d["t_s"]), rng(d["t_s"])])
        for b, d in r["profile"].items():
            cs = d.get("class_share")
            if cs:
                prof.append([n, b, d["n_kernels"], d["kernel_time_us_total"]] + [cs.get(c, 0.0) for c in
                            ("allreduce", "other_collective", "gemm", "attention", "other")])
            else:
                prof.append([n, b, 0, 0, "", "", "", "", ""])
    table(info, ["engine", "startup_s", "max_batched", "n_errors", "allreduce_backend_log"], "B2_engines")
    table(dec, ["engine", "batch", "ms_per_token_median", "range", "t1_s_median"], "B2_decode")
    table(pre, ["engine", "len", "prefill_s_median", "range"], "B2_prefill")
    table(prof, ["engine", "batch", "n_kernels", "kernel_us", "share_allreduce", "share_other_coll", "share_gemm", "share_attn", "share_other"], "B2_profile")
    # I5: KV bytes / p2p bandwidth vs prefill time (72B, TP=4, 16k)
    try:
        cfgj = json.load(open("/work/allen0117/comm_opt_env/models/Qwen2.5-72B-Instruct/config.json"))
        hd = cfgj["hidden_size"] // cfgj["num_attention_heads"]
        kv = 2 * cfgj["num_hidden_layers"] * cfgj["num_key_value_heads"] * hd * 2 * 16384
        print(f"\nI5: KV bytes for 16384 tokens (72B, bf16) = {kv/1e9:.3f} GB")
        if not B3: load_b3()
        bw = sr_busbw()
        pf = [r for r in pre if r[0] == "q72_tp4" and r[1] == "16384"]
        if bw and pf:
            t = kv / (bw * 1e9)
            print(f"I5: p2p busbw (sr_default >=256MB median) = {bw:.1f} GB/s -> transfer {t*1e3:.2f} ms; prefill TP4 16k = {pf[0][2]*1e3:.1f} ms; ratio = {t/pf[0][2]*100:.2f}%")
    except Exception as e:
        print("I5 not computed:", e)


# ------------------------------------------------------------------ B3
ROWRE = re.compile(r"^\s*(\d+)\s+(\d+)\s+float\s+(?:sum|none)\s+-?\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)")


def parse_run(p):
    out = {}
    for l in opengz(p):
        m = ROWRE.match(l)
        if m:
            out[int(m.group(1))] = dict(t_us=float(m.group(3)), bus=float(m.group(5)), wrong=m.group(6), t_ip=float(m.group(7)), bus_ip=float(m.group(9)))
    return out


B3 = {}


def load_b3():
    for p in sorted(glob.glob(os.path.join(R, "b3", "runs", "*__run*.log*"))):
        m = re.match(r"(.+)__run(\d+)\.(small|big)\.log", os.path.basename(p))
        if not m: continue
        for size, d in parse_run(p).items():
            B3.setdefault((m.group(1), size), []).append(d)


def sr_busbw():
    v = [d["bus"] for (c, s), ds in B3.items() if c == "sr_default" and s >= 256 * 2**20 for d in ds]
    return med(v) if v else None


def b3():
    load_b3()
    if not B3:
        print("\n## B3: no parsed rows"); return
    cfgs = sorted({c for c, _ in B3}); sizes = sorted({s for _, s in B3})
    rows = []
    for c in cfgs:
        for s in sizes:
            ds = B3.get((c, s))
            if ds:
                rows.append([c, s, len(ds), med([d["bus"] for d in ds]), rng([d["bus"] for d in ds]), med([d["t_us"] for d in ds]), ",".join(sorted({d["wrong"] for d in ds}))])
    table(rows, ["cfg", "bytes", "n", "busbw_oop_median", "busbw_range", "time_us_median", "wrong"], "B3_curves")
    print("\n## B3 judgements")
    for s in sizes:
        if s >= 512 * 2**20:
            ds = B3.get(("ar_ring_simple", s))
            if ds: print(f"C1 ar_ring_simple {s}B busbw median {med([d['bus'] for d in ds]):.1f}")
    for s in sizes:
        if s >= 256 * 2**20 and ("ar_default", s) in B3 and ("ar_nvls_off", s) in B3:
            print(f"C2 {s}B default/nvls_off = {med([d['bus'] for d in B3[('ar_default', s)]]) / med([d['bus'] for d in B3[('ar_nvls_off', s)]]):.3f}")
    for s in sizes:
        if 4 * 2**20 <= s <= 128 * 2**20 and ("ar_default", s) in B3:
            base = [d["bus"] for d in B3[("ar_default", s)]]
            best = None
            for c in ("ar_ring_ll", "ar_ring_ll128", "ar_ring_simple", "ar_tree", "ar_nvls", "ar_nvls_off"):
                ds = B3.get((c, s))
                if ds and len(ds) == len(base) and len(base) >= 2:
                    gain_min = min(a["bus"] / b for a, b in zip(sorted(ds, key=lambda d: d["bus"]), sorted(base))) - 1
                    if best is None or gain_min > best[1]: best = (c, gain_min)
            if best: print(f"C3 {s}B best forced algo {best[0]} min-over-launch gain vs default {best[1]*100:.1f}%")
    for s in sizes:
        if s <= 1024 and ("ar_default", s) in B3:
            print(f"C4 ar_default {s}B time_us median {med([d['t_us'] for d in B3[('ar_default', s)]]):.2f}")
    for s in sizes:
        if s >= 256 * 2**20 and ("sr_default", s) in B3:
            print(f"C5 sr_default {s}B busbw {med([d['bus'] for d in B3[('sr_default', s)]]):.1f}")


# ------------------------------------------------------------------ clocks
def clocks():
    # Opus 2026-11-25: job 498237 has a replacement monitor gpu_monitor_2.csv (original monitor stopped in-job); read all
    ps = sorted(glob.glob(os.path.join(R, "gpu_monitor*.csv")) + glob.glob(os.path.join(R, "gpu_monitor*.csv.gz")))
    if not ps: return
    print("\n# monitor files:", [os.path.basename(q) for q in ps])
    sm, allsm = [], []
    for r in (row for q in ps for row in csv.DictReader(opengz(q), skipinitialspace=True)):
        try:
            c = float((r.get("clocks.current.sm [MHz]") or r["clocks.sm [MHz]"]).split()[0]); u = float(r["utilization.gpu [%]"].split()[0])
        except Exception:
            continue
        allsm.append(c)
        if u >= 50: sm.append(c)
    print(f"\n## K1 SM clock (MHz): median while util>=50%: {med(sm):.0f} (n={len(sm)}); all samples median {med(allsm):.0f} (n={len(allsm)})")


print("# M25 analysis of", R)
print(open(os.path.join(R, "blocks.tsv")).read() if os.path.exists(os.path.join(R, "blocks.tsv")) else "")
for f in (b4, b1, b2, b3, clocks):
    try:
        f()
    except Exception as e:
        print(f"!! {f.__name__} failed: {e!r}")
