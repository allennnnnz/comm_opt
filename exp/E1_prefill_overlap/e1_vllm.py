#!/usr/bin/env python
"""E1 block A: vLLM prefill baseline.  ONE vLLM engine per process (dummy weights), one launch.

Per launch: for every input length L (random order, seed recorded), send REPS single requests (1 prompt of L tokens, 1 output token); the
first DISCARD are dropped.  Per request: wall time of llm.generate (TTFT proxy, the request is a single prefill step) and, if vLLM provides
them (RequestOutput.metrics, needs disable_log_stats=False), queued/scheduled/first-token timestamps -> queue time and prefill time.
Then PROF_REPS extra requests per L are profiled with the vLLM torch profiler (rank 0): from each trace we take the GPU kernel time sum,
the all-reduce kernel time sum and share (classification = M25 vllm_bench.classify), the all-reduce kernel names, the GPU span of the step
and the all-reduce call count.  One trace per L is copied (gz) into the output directory.
Writes DIR/result.json (incrementally) and DIR/per_request.tsv.
"""
import argparse, glob, gzip, json, os, random, shutil, sys, time, zlib

sys.path.insert(0, "/home/allen0117/comm_opt/exp/M25_topic_survey")
from vllm_bench import classify, summarize_trace  # noqa: E402  (only stdlib imports at module level)


def ar_calls(path):
    """all-reduce kernel durations (us) in a trace + distinct names."""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        tr = json.load(f)
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    ks = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
    ar = [e for e in ks if classify(e["name"]) == "allreduce"]
    names = {}
    for e in ar:
        names[e["name"][:120]] = names.get(e["name"][:120], 0) + 1
    return [float(e["dur"]) for e in ar], names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--lens", default="2048,4096,8192,16384")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--discard", type=int, default=2)
    ap.add_argument("--prof-reps", type=int, default=3)
    ap.add_argument("--maxlen", type=int, default=17408)
    ap.add_argument("--max-batched", type=int, default=17408)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--launch", type=int, required=True)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--trace-root", required=True)
    ap.add_argument("--keep-trace-mb", type=float, default=50.0)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    trace_dir = os.path.join(a.trace_root, f"launch{a.launch}")
    os.makedirs(trace_dir, exist_ok=True)
    lens = [int(x) for x in a.lens.split(",")]
    order = lens[:]
    random.Random(a.seed).shuffle(order)
    res = {"model": a.model, "tp": a.tp, "launch": a.launch, "seed": a.seed, "order": order, "reps": a.reps, "discard": a.discard,
           "max_num_batched_tokens": a.max_batched, "maxlen": a.maxlen, "requests": {}, "profile": {}, "errors": [],
           "fuse_allreduce_rms": False, "enforce_eager": False}
    rpath = os.path.join(a.outdir, "result.json")

    def save():
        with open(rpath, "w") as f:
            json.dump(res, f, indent=1)

    import numpy as np
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    def prompt(L):
        r = np.random.default_rng(zlib.crc32(f"e1-{a.seed}-{L}".encode()))
        return [TokensPrompt(prompt_token_ids=r.integers(1000, 100000, size=L).tolist())]

    sp = SamplingParams(max_tokens=1, min_tokens=1, ignore_eos=True, temperature=0.0)

    def one(pr):
        t0 = time.perf_counter()
        out = llm.generate(pr, sp, use_tqdm=False)
        return time.perf_counter() - t0, out

    # same engine settings as M25 B2 q72_tp8 (graph mode, fuse_allreduce_rms off); disable_log_stats=False so RequestOutput.metrics is filled
    kw = dict(model=a.model, tokenizer=a.model, load_format="dummy", dtype="bfloat16", tensor_parallel_size=a.tp,
              enforce_eager=False, max_model_len=a.maxlen, gpu_memory_utilization=a.gpu_util, enable_prefix_caching=False,
              max_num_seqs=128, max_num_batched_tokens=a.max_batched, seed=0, disable_log_stats=False,
              compilation_config={"pass_config": {"fuse_allreduce_rms": False}},
              profiler_config={"profiler": "torch", "torch_profiler_dir": trace_dir, "torch_profiler_with_stack": False,
                               "torch_profiler_use_gzip": True, "torch_profiler_dump_cuda_time_total": False})
    t0 = time.perf_counter()
    llm = LLM(**kw)
    one([TokensPrompt(prompt_token_ids=list(range(1000, 1016)))])
    res["startup_s"] = time.perf_counter() - t0
    save()

    tsv = open(os.path.join(a.outdir, "per_request.tsv"), "w")
    tsv.write("launch\tL\tidx\tdiscarded\twall_s\tqueue_s\tprefill_s\tn_prompt_tokens\n")
    for L in order:
        rows = []
        try:
            pr = prompt(L)
            for i in range(a.reps):
                wall, out = one(pr)
                m = getattr(out[0], "metrics", None)
                q = pf = None
                if m is not None and getattr(m, "scheduled_ts", 0):
                    q = m.scheduled_ts - m.queued_ts
                    pf = m.first_token_ts - m.scheduled_ts
                disc = i < a.discard
                rows.append({"idx": i, "discarded": disc, "wall_s": wall, "queue_s": q, "prefill_s": pf,
                             "n_prompt_tokens": len(out[0].prompt_token_ids)})
                tsv.write(f"{a.launch}\t{L}\t{i}\t{int(disc)}\t{wall:.6f}\t{'' if q is None else f'{q:.6f}'}\t"
                          f"{'' if pf is None else f'{pf:.6f}'}\t{len(out[0].prompt_token_ids)}\n")
                tsv.flush()
        except Exception as e:
            res["errors"].append(f"requests L={L}: {e!r}")
        res["requests"][str(L)] = rows
        save()

    # ---- profiled requests (rank 0 trace), delay 0 / no max: profiler is on for exactly one prefill step
    for L in order:
        res["profile"][str(L)] = []
        for k in range(a.prof_reps):
            try:
                pr = prompt(L)
                prefix = f"e1_L{L}_p{k}"
                llm.start_profile(prefix, delay_iterations=0, max_iterations=0)
                wall, _ = one(pr)
                llm.stop_profile()
                time.sleep(4)
                files = sorted(glob.glob(os.path.join(trace_dir, f"{prefix}*rank0*.json*")))
                if not files:
                    files = sorted(glob.glob(os.path.join(trace_dir, f"{prefix}*.json*")))
                if not files:
                    res["errors"].append(f"profile L={L} k={k}: no trace in {trace_dir}")
                    continue
                f0 = files[0]
                summ = summarize_trace(f0)
                durs, names = ar_calls(f0)
                durs_s = sorted(durs)
                kt = summ["kernel_time_us_total"]
                entry = {"k": k, "profiled_wall_s": wall, "trace": os.path.basename(f0), "trace_size_mb": os.path.getsize(f0) / 1e6,
                         "kernel_time_us_total": kt, "allreduce_us_total": summ["class_us"].get("allreduce", 0.0),
                         "allreduce_share": summ["class_share"].get("allreduce", 0.0), "gpu_span_us": summ["gpu_span_us"],
                         "n_kernels": summ["n_kernels"], "n_allreduce_calls": len(durs),
                         "allreduce_p50_us": durs_s[len(durs_s) // 2] if durs_s else None, "allreduce_kernel_names": names,
                         "class_us": summ["class_us"], "annotations": summ["annotations"]}
                if k == 0:
                    if entry["trace_size_mb"] <= a.keep_trace_mb:
                        shutil.copy(f0, a.outdir)
                        entry["kept_trace"] = os.path.join(a.outdir, os.path.basename(f0))
                    else:
                        entry["kept_trace"] = None
                res["profile"][str(L)].append(entry)
            except Exception as e:
                res["errors"].append(f"profile L={L} k={k}: {e!r}")
            save()

    save()
    tsv.close()
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:
        res["errors"].append(f"shutdown: {e!r}")
        save()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
