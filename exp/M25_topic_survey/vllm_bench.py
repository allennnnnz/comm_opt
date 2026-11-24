#!/usr/bin/env python
"""M25 B2: one vLLM engine per process (dummy weights). Per-token time, prefill time, and profiler kernel breakdown.

Examples
  vllm_bench.py --name q72_tp8 --model /work/.../models/Qwen2.5-72B-Instruct --tp 8 --maxlen 17408 \
      --decode 1,8,32,128 --prefill 1024,4096,16384 --prof 1,8,32,128 --outdir DIR
Everything is written incrementally to DIR/result.json so a timeout still leaves partial data.

Per-token time = (t128 - t1) / 127 for the same batch of random-token prompts (prefix caching is OFF so both calls
pay the same prefill). Prefill = batch 1, input length L, 1 generated token. Profiler: vLLM torch profiler on all workers;
the rank-0 trace is parsed into a kernel-time summary (classification rules in CLASS_RULES below).
"""
import argparse, glob, gzip, hashlib, json, os, re, shutil, sys, time, zlib

CLASS_RULES = [  # first match wins, matched on lower-cased kernel name
    ("allreduce", ["cross_device_reduce", "allreduce", "all_reduce", "oneshot", "twoshot", "one_shot", "two_shot"]),
    ("other_collective", ["nccl", "allgather", "all_gather", "reducescatter", "reduce_scatter", "alltoall", "all_to_all",
                          "dispatch", "combine", "sendrecv", "broadcast"]),
    ("attention", ["fmha", "flash", "attention", "attn", "paged", "mla", "fa3", "flashinfer", "prepare_inputs",
                   "softmax"]),
    ("gemm", ["gemm", "cutlass", "cublas", "nvjet", "xmma", "gemv", "splitk", "fused_moe", "grouped", "matmul",
              "ampere_", "hopper_", "sm90_", "sm80_"]),
]


def classify(name):
    n = name.lower()
    for cls, pats in CLASS_RULES:
        if any(p in n for p in pats):
            return cls
    return "other"


def summarize_trace(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        tr = json.load(f)
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    kern = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
    ann = {}
    for e in ev:
        n = e.get("name", "")
        if n.startswith("execute_context_"):
            ann[n] = ann.get(n, 0) + 1
    tot = {}
    per_name = {}
    for e in kern:
        c = classify(e["name"])
        d = float(e.get("dur", 0.0))
        tot[c] = tot.get(c, 0.0) + d
        per_name[e["name"]] = per_name.get(e["name"], [0, 0.0])
        per_name[e["name"]][0] += 1
        per_name[e["name"]][1] += d
    total = sum(tot.values())
    span = 0.0
    if kern:
        span = max(e["ts"] + e["dur"] for e in kern) - min(e["ts"] for e in kern)
    top = sorted(per_name.items(), key=lambda kv: -kv[1][1])[:40]
    return {"n_kernels": len(kern), "kernel_time_us_total": total, "gpu_span_us": span, "class_us": tot,
            "class_share": {k: (v / total if total else 0.0) for k, v in tot.items()},
            "annotations": ann, "top_kernels": [(n, c, d, classify(n)) for n, (c, d) in top]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--eager", type=int, default=0)
    ap.add_argument("--maxlen", type=int, default=2048)
    ap.add_argument("--decode", default="")
    ap.add_argument("--prefill", default="")
    ap.add_argument("--prof", default="")
    ap.add_argument("--in-len", type=int, default=1024)
    ap.add_argument("--out-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prof-iters", type=int, default=6)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--max-batched", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--trace-root", required=True, help="where raw torch traces are written (outside results/)")
    ap.add_argument("--keep-trace-mb", type=float, default=50.0)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    trace_dir = os.path.join(a.trace_root, a.name)
    os.makedirs(trace_dir, exist_ok=True)
    res = {"name": a.name, "model": a.model, "tp": a.tp, "ep": a.ep, "eager": a.eager, "maxlen": a.maxlen,
           "decode": {}, "prefill": {}, "profile": {}, "errors": []}
    rpath = os.path.join(a.outdir, "result.json")

    def save():
        with open(rpath, "w") as f:
            json.dump(res, f, indent=1)

    import numpy as np
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    def prompts(b, L, tag):
        r = np.random.default_rng(zlib.crc32(f"{tag}-{b}-{L}".encode()))
        return [TokensPrompt(prompt_token_ids=r.integers(1000, 100000, size=L).tolist()) for _ in range(b)]

    def gen(pr, ntok):
        sp = SamplingParams(max_tokens=ntok, min_tokens=ntok, ignore_eos=True, temperature=0.0)
        t0 = time.perf_counter()
        llm.generate(pr, sp, use_tqdm=False)
        return time.perf_counter() - t0

    kw = dict(model=a.model, tokenizer=a.model, load_format="dummy", dtype="bfloat16", tensor_parallel_size=a.tp,
              enable_expert_parallel=bool(a.ep), enforce_eager=bool(a.eager), max_model_len=a.maxlen,
              gpu_memory_utilization=a.gpu_util, enable_prefix_caching=False, max_num_seqs=a.max_seqs,
              seed=0, disable_log_stats=True,
              profiler_config={"profiler": "torch", "torch_profiler_dir": trace_dir,
                               "torch_profiler_with_stack": False, "torch_profiler_use_gzip": True,
                               "torch_profiler_dump_cuda_time_total": False})
    if a.max_batched:
        kw["max_num_batched_tokens"] = a.max_batched
    # Opus 2026-11-25: the all-reduce+RMSNorm fusion pass calls FlashInfer trtllm_mnnvl kernels, whose JIT needs nvcc>=12.8
    # (not available: CUDA<=12.6, driver 12.4); validation 497906 failed in graph mode. Keep vLLM's plain all-reduce path.
    if not a.eager:
        kw["compilation_config"] = {"pass_config": {"fuse_allreduce_rms": False}}
    res["fuse_allreduce_rms"] = False
    t_init0 = time.perf_counter()
    llm = LLM(**kw)
    gen(prompts(1, 16, "warm"), 2)  # first generate = engine ready
    res["startup_s"] = time.perf_counter() - t_init0
    save()

    # ---- decode time per token
    for b in [int(x) for x in a.decode.split(",") if x]:
        try:
            pr = prompts(b, a.in_len, "dec")
            gen(pr, a.out_tokens)  # warm-up (1x)
            t1 = [gen(pr, 1) for _ in range(a.reps)]
            tN = [gen(pr, a.out_tokens) for _ in range(a.reps)]
            per = [(tn - tt) / (a.out_tokens - 1) * 1000 for tt, tn in zip(t1, tN)]
            res["decode"][str(b)] = {"t1_s": t1, "tN_s": tN, "ms_per_token": per, "in_len": a.in_len,
                                     "out_tokens": a.out_tokens}
        except Exception as e:  # keep going
            res["errors"].append(f"decode b={b}: {e!r}")
        save()

    # ---- prefill time
    for L in [int(x) for x in a.prefill.split(",") if x]:
        try:
            pr = prompts(1, L, "pre")
            gen(pr, 1)
            ts = [gen(pr, 1) for _ in range(a.reps)]
            res["prefill"][str(L)] = {"t_s": ts}
        except Exception as e:
            res["errors"].append(f"prefill L={L}: {e!r}")
        save()

    # ---- profiler (decode steps only: skip prefill iterations of this batch, then record prof_iters steps)
    try:
        mb = int(llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens)
    except Exception:
        mb = a.max_batched if a.max_batched else 8192
    res["max_num_batched_tokens"] = mb
    for b in [int(x) for x in a.prof.split(",") if x]:
        try:
            pr = prompts(b, a.in_len, "prof")
            gen(pr, 4)  # warm shapes
            n_prefill_iters = -(-b * a.in_len // mb)
            prefix = f"{a.name}_b{b}"
            llm.start_profile(prefix, delay_iterations=n_prefill_iters + 2, max_iterations=a.prof_iters)
            gen(pr, a.prof_iters + 12)
            llm.stop_profile()  # no-op if max_iterations already stopped it
            time.sleep(3)
            files = sorted(glob.glob(os.path.join(trace_dir, f"{prefix}*rank0*.json*")))
            if not files:
                files = sorted(glob.glob(os.path.join(trace_dir, f"{prefix}*.json*")))
            if not files:
                res["errors"].append(f"profile b={b}: no trace file in {trace_dir}")
                res["profile"][str(b)] = {"trace_files": [], "all_files": os.listdir(trace_dir)}
            else:
                f0 = files[0]
                summ = summarize_trace(f0)
                size_mb = os.path.getsize(f0) / 1e6
                dst = f0
                if size_mb <= a.keep_trace_mb:
                    shutil.copy(f0, a.outdir)
                    dst = os.path.join(a.outdir, os.path.basename(f0))
                else:
                    h = hashlib.sha256(open(f0, "rb").read()).hexdigest()
                    summ["large_trace"] = {"path": f0, "sha256": h, "size_mb": size_mb}
                summ["trace_file"] = dst
                summ["trace_size_mb"] = size_mb
                res["profile"][str(b)] = summ
        except Exception as e:
            res["errors"].append(f"profile b={b}: {e!r}")
        save()

    save()
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:
        res["errors"].append(f"shutdown: {e!r}")
        save()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
