#!/usr/bin/env python
"""M25 B2 follow-up (Opus 2026-11-25): per-call all-reduce kernel durations from the kept rank-0 torch traces.

Why: the class_share in result.json sums kernel durations; an all-reduce kernel's duration includes the time it spins waiting
for the slowest rank, so the share mixes transfer time with rank skew (q72_tp8 b1: 41.6 % vs 23.6 % in the identical
q72_tp8_mb8k engine). Here we report, per engine x batch:
  steps        = number of decode steps in the trace (= number of distinct 'execute_context_*' annotations, else n/a)
  ar_calls     = all-reduce kernel count; ar_per_step
  ar_us p10/p50/mean per call
  share_sum    = sum(ar) / sum(all kernels)                       (same as result.json, includes waiting)
  share_p10    = ar_calls * p10(ar) / (sum(all kernels) - sum(ar) + ar_calls * p10(ar))
                 (lower-bound style estimate: every call costs only its fast-path duration)
  step_ms      = GPU span / steps
Decode-only view (added after a first look): the profile window of b>=8 also caught prefill chunks (all-reduce ~1.3 ms per
call, 285 MB). Steps are cut by all-reduce count (calls per step = 2*num_hidden_layers+1 from the model config); a step is
prefill if its median all-reduce > 50 us. For decode steps we report
  dec_steps, dec_step_kernel_ms, dec_share_sum (all all-reduce time), dec_share_x1 (excluding each step's FIRST all-reduce,
  which absorbs inter-rank launch skew; the single 26 ms call in q72_tp8 b1 is such a first call)
usage: b2_kernels.py <result dir> [more dirs...]
"""
import glob, gzip, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm_bench import classify  # same classification rules as the job


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def one(path):
    with gzip.open(path, "rt") as f:
        tr = json.load(f)
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    kern = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
    steps = len({e["name"] for e in ev if str(e.get("name", "")).startswith("execute_context_")}) or None
    ar = [float(e["dur"]) for e in kern if classify(e["name"]) == "allreduce"]
    tot = sum(float(e["dur"]) for e in kern)
    span = (max(e["ts"] + e["dur"] for e in kern) - min(e["ts"] for e in kern)) if kern else 0
    p10, p50 = pct(ar, 0.10), pct(ar, 0.50)
    # decode-only by step segmentation
    ks = sorted(kern, key=lambda e: e["ts"])
    idx = [i for i, e in enumerate(ks) if classify(e["name"]) == "allreduce"]
    dec = None
    if CPS and idx and len(idx) % CPS == 0:
        dk, dsum, dx1, nst = 0.0, 0.0, 0.0, 0
        nsteps = len(idx) // CPS
        for s_ in range(nsteps):
            a0 = idx[s_ * CPS]; a1 = idx[(s_ + 1) * CPS] if s_ + 1 < nsteps else len(ks)
            seg = ks[a0:a1]
            ards = [float(e["dur"]) for e in seg if classify(e["name"]) == "allreduce"]
            if pct(ards, 0.5) > 50:
                continue  # prefill step
            nst += 1; dk += sum(float(e["dur"]) for e in seg); dsum += sum(ards); dx1 += sum(ards[1:])
        if nst:
            dec = {"dec_steps": nst, "dec_step_kernel_ms": dk / nst / 1e3, "dec_share_sum": dsum / dk,
                   "dec_share_x1": dx1 / (dk - (dsum - dx1))}
    non_ar = tot - sum(ar)
    est = len(ar) * p10
    return {"steps": steps, "ar_calls": len(ar), "ar_p10_us": p10, "ar_p50_us": p50,
            "ar_mean_us": (sum(ar) / len(ar)) if ar else float("nan"),
            "share_sum": sum(ar) / tot if tot else float("nan"),
            "share_p10": est / (non_ar + est) if (non_ar + est) else float("nan"),
            "kernel_ms": tot / 1e3, "span_ms": span / 1e3,
            "dec": dec, "ar_names": sorted({e["name"][:60] for e in kern if classify(e["name"]) == "allreduce"})}


CPS = None


def main():
    global CPS
    print("run\tengine\tbatch\tsteps\tar_calls\tar_per_step\tar_p10_us\tar_p50_us\tar_mean_us\tshare_sum\tshare_p10\tkernel_ms\tspan_ms\tstep_ms\tdec_steps\tdec_step_kernel_ms\tdec_share_sum\tdec_share_x1\tar_kernel_names")
    for R in sys.argv[1:]:
        for f in sorted(glob.glob(os.path.join(R, "b2", "*", "*_rank0*.trace.json.gz"))):
            eng = os.path.basename(os.path.dirname(f))
            b = os.path.basename(f)[len(eng) + 2:].split("_rank0")[0]
            mdl = json.load(open(os.path.join(os.path.dirname(f), "result.json")))["model"]
            cfg = json.load(open(os.path.join(mdl, "config.json")))
            CPS = 2 * cfg["num_hidden_layers"] + 1
            r = one(f)
            st = r["steps"]
            print("\t".join(str(x) for x in [os.path.basename(R.rstrip("/")), eng, b, st, r["ar_calls"],
                  f"{r['ar_calls'] / st:.1f}" if st else "", f"{r['ar_p10_us']:.2f}", f"{r['ar_p50_us']:.2f}",
                  f"{r['ar_mean_us']:.2f}", f"{r['share_sum']:.3f}", f"{r['share_p10']:.3f}", f"{r['kernel_ms']:.2f}",
                  f"{r['span_ms']:.2f}", f"{r['span_ms'] / st:.2f}" if st else "",
                  *( [r["dec"]["dec_steps"], f"{r['dec']['dec_step_kernel_ms']:.3f}", f"{r['dec']['dec_share_sum']:.3f}",
                      f"{r['dec']['dec_share_x1']:.3f}"] if r["dec"] else ["", "", "", ""]), ";".join(r["ar_names"])]))


if __name__ == "__main__":
    main()
