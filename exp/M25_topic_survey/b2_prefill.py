#!/usr/bin/env python
"""M25 B2 follow-up (Opus 2026-11-25 20:30, post-measurement): all-reduce share of kernel time in the PREFILL steps that the
b>=8 profile windows happened to catch (steps cut as in b2_kernels.py; prefill = median all-reduce > 50 us). Not in the pre-written rules.
usage: b2_prefill.py <result dir> [...]"""
# prefill-step all-reduce share from kept rank-0 traces (steps cut by CPS like b2_kernels.py)
import glob, gzip, json, os, sys
sys.path.insert(0, "/home/allen0117/comm_opt/exp/M25_topic_survey")
from vllm_bench import classify
for R in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(R, "b2", "*", "*_rank0*.trace.json.gz"))):
        eng = os.path.basename(os.path.dirname(f)); b = os.path.basename(f)[len(eng)+2:].split("_")[0]
        mdl = json.load(open(os.path.join(os.path.dirname(f), "result.json")))["model"]
        cps = 2 * json.load(open(os.path.join(mdl, "config.json")))["num_hidden_layers"] + 1
        ev = json.load(gzip.open(f, "rt")); ev = ev["traceEvents"] if isinstance(ev, dict) else ev
        ks = sorted([e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"], key=lambda e: e["ts"])
        idx = [i for i, e in enumerate(ks) if classify(e["name"]) == "allreduce"]
        if not idx or len(idx) % cps: continue
        n = len(idx) // cps
        for s in range(n):
            seg = ks[idx[s*cps]: idx[(s+1)*cps] if s+1 < n else len(ks)]
            ar = sorted(float(e["dur"]) for e in seg if classify(e["name"]) == "allreduce")
            if ar[len(ar)//2] <= 50: continue
            tot = sum(float(e["dur"]) for e in seg); sa = sum(ar)
            print(f"{os.path.basename(R)[-14:]}\t{eng}\tb{b}\tstep{s}\tkernel_ms={tot/1e3:.1f}\tar_ms={sa/1e3:.1f}\tshare={sa/tot:.3f}\tar_med_us={ar[len(ar)//2]:.0f}")
