"""M5 inference workload: vLLM offline engine, TP=8; for each config, profile a steady window of engine steps.

One launch loads the model once and runs every config in a seeded random order (R11). Each config gets
a profiled window (torch profiler traces, one file per worker rank, prefix = config name) and unprofiled
timing runs so the profiler's own overhead can be checked.

Usage (inside .venv-vllm):
  python exp/M5_e2e_comm_share/infer.py --model Qwen/Qwen2.5-32B-Instruct \
      --configs prefill:1:512,decode:8:128 --seed 0 --out DIR
  config = prefill:<batch>:<input_len>  or  decode:<batch>:<input_len>
"""
import argparse
import json
import os
import random
import time

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights come from the shared HF cache

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--configs", required=True)
p.add_argument("--output-len", type=int, default=256, help="decode: tokens generated per request")
p.add_argument("--tp", type=int, default=8)
p.add_argument("--profile-steps", type=int, default=16)
p.add_argument("--timing-runs", type=int, default=3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", seed=a.seed,
          max_model_len=None, gpu_memory_utilization=0.85,  # model's own limit (TinyLlama: 2048)
          enable_prefix_caching=False,  # repeated prefill must recompute, not hit the cache
          disable_log_stats=False, enable_logging_iteration_details=True,  # unprofiled per-step engine time
          profiler_config={"profiler": "torch", "torch_profiler_dir": os.path.abspath(a.out),
                           "torch_profiler_with_stack": False, "ignore_frontend": True})
vocab = llm.get_tokenizer().vocab_size
rng = np.random.default_rng(a.seed)
hf = llm.llm_engine.model_config.hf_config


def prompts(n, length):
    return [TokensPrompt(prompt_token_ids=rng.integers(100, vocab - 100, length).tolist()) for _ in range(n)]


def run(phase, batch, length, out_len=None):
    n = 1 if phase == "prefill" else (out_len or a.output_len)
    llm.generate(prompts(batch, length), SamplingParams(max_tokens=n, ignore_eos=True), use_tqdm=False)


def timed(*args):
    t0 = time.perf_counter()
    run(*args)
    return time.perf_counter() - t0


configs = [(c.split(":")[0], int(c.split(":")[1]), int(c.split(":")[2])) for c in a.configs.split(",")]
random.Random(a.seed).shuffle(configs)
meta = dict(vars(a), vocab=vocab, hidden_size=hf.hidden_size, num_layers=hf.num_hidden_layers,
            max_num_batched_tokens=llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens,
            order=[f"{ph}_b{b}_l{l}" for ph, b, l in configs], configs={})
for phase, batch, length in configs:
    name = f"{phase}_b{batch}_l{length}"
    run(phase, batch, length)  # warmup for this shape
    # decode: skip prefill and early decode steps, capture a steady full-batch window
    steps_per_call = -(-batch * length // meta["max_num_batched_tokens"]) if phase == "prefill" else a.output_len
    llm.start_profile(name, delay_iterations=32 if phase == "decode" else 0, max_iterations=a.profile_steps)
    for _ in range(1 if phase == "decode" else -(-a.profile_steps // steps_per_call) + 1):
        run(phase, batch, length)
    llm.stop_profile()
    # unprofiled per-step time; decode uses a full and a half-length run so the prefill time cancels out
    if phase == "prefill":
        walls = [timed(phase, batch, length) for _ in range(a.timing_runs)]
        step_s = [w / steps_per_call for w in walls]
    else:
        half = a.output_len // 2
        walls = [(timed(phase, batch, length), timed(phase, batch, length, half)) for _ in range(a.timing_runs)]
        step_s = [(f - h) / (a.output_len - half) for f, h in walls]
    meta["configs"][name] = dict(phase=phase, batch=batch, input_len=length, steps_per_call=steps_per_call,
                                 unprofiled_wall_s=walls, unprofiled_step_s=step_s)
    print(f"{name}: unprofiled step {1e3 * np.median(step_s):.3f} ms", flush=True)
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print("done")
