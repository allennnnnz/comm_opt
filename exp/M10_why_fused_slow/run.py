"""M10: is "no thread block clusters -> one block per token row -> too few blocks" why fused kernels lose on A100?

Same bytes, different hidden: smaller hidden = more tokens = more blocks for row-per-block kernels (TRT-LLM fused),
while vLLM (blocks depend on bytes) and LMDeploy fused (packs several rows per block when hidden is small) keep
their block counts. Reuses M9's bench_fused.py and LMDeploy test_comm (forced fused) unchanged.

Also: register-limit check (hidden 6400 vs 6656, F_trt_one), and one nsys trace at hidden 5120 to record the actual
grid/block/registers of every kernel.

Usage: python3 exp/M10_why_fused_slow/run.py          (refuses to run while other GPU processes exist)
       DRY_RUN=1 python3 exp/M10_why_fused_slow/run.py
"""
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from ncclrun import ROOT, gpu_processes, gpu_state, snapshot_env  # noqa: E402

SEED = 20261105
REPS = 3
SIZES = [160 << 10, 320 << 10, 640 << 10, 1280 << 10]
HIDDENS = [5120, 2560, 1280]
BENCH = ROOT / "exp/M9_fused_baselines/bench_fused.py"
LMD_FORCE = ROOT / "third_party/lmdeploy/build_force/bin/test_comm"
NSYS = "/usr/local/cuda/bin/nsys"
CUDA = {"CUDA_HOME": "/usr/local/cuda", "PATH": f"/usr/local/cuda/bin:{os.environ['PATH']}"}


def tokens(h):
    return ",".join(str(b // (2 * h)) for b in SIZES)


out = ROOT / "results" / "M10_why_fused_slow" / time.strftime("%Y%m%d_%H%M%S")
jobs = [(k, h, r) for k in ("torch", "lmdforce") for h in HIDDENS for r in range(REPS)]
jobs += [("reglimit", h, r) for h in (6400, 6656) for r in range(REPS)]
random.Random(SEED).shuffle(jobs)
jobs += [("nsys_torch", 5120, 0), ("nsys_lmd", 5120, 0)]  # traces last: launch parameters only


def torchrun(tag, hidden, toks, impls, rep):
    return [str(ROOT / ".venv-vllm/bin/torchrun"), "--nproc-per-node", "8", "--master-port", "29631", str(BENCH),
            "--out", str(out / f"{tag}.json"), "--seed", str(rep), "--hidden", str(hidden), "--tokens", toks,
            "--impls", impls]


def build(kind, hidden, rep):
    tag = f"{kind}_h{hidden}_r{rep}"
    lmd_env = {"M9_HIDDEN": str(hidden), "M9_TOKENS": tokens(hidden), "M9_WARMUP": "10", "M9_ITERS": "1000",
               "M9_B2B": "20"}
    if kind == "torch":
        return tag, torchrun(tag, hidden, tokens(hidden), "U_vllm,F_trt_one,F_trt_two", rep), CUDA
    if kind == "reglimit":
        return tag, torchrun(tag, hidden, "8", "F_trt_one", rep), CUDA
    if kind == "lmdforce":
        return tag, [str(LMD_FORCE), "cuda-ipc"], lmd_env
    trace = ["-t", "cuda", "-o", str(out / tag), "--force-overwrite", "true"]
    if kind == "nsys_torch":
        return tag, [NSYS, "profile", *trace, *torchrun(tag, hidden, tokens(hidden),
                                                       "U_vllm,F_trt_one,F_trt_two", 0)], CUDA
    return tag, [NSYS, "profile", *trace, str(LMD_FORCE), "cuda-ipc"], {**lmd_env, "M9_ITERS": "50"}


if os.environ.get("DRY_RUN"):
    for j in jobs:
        tag, cmd, env = build(*j)
        print(tag, " ".join(f"{k}={v}" for k, v in env.items() if k != "PATH"), shlex.join(cmd))
    print(f"{len(jobs)} launches", file=sys.stderr)
    sys.exit(0)
if procs := gpu_processes():
    sys.exit(f"GPUs have running processes; not measuring (R13):\n{procs}")
out.mkdir(parents=True)
snapshot_env(out)
with open(out / "schedule.tsv", "w") as sched:
    sched.write(f"# seed={SEED}\norder\ttag\tkind\thidden\trep\tgpu_state_before\tstart\tseconds\trc\n")
    for i, (kind, hidden, rep) in enumerate(jobs):
        if procs := gpu_processes():
            sys.exit(f"stopped before job {i}: other GPU processes appeared:\n{procs}")
        tag, cmd, env = build(kind, hidden, rep)
        state, t0 = gpu_state(), time.time()
        with open(out / f"{tag}.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env}).returncode
        sched.write(f"{i}\t{tag}\t{kind}\t{hidden}\t{rep}\t{state}\t{time.strftime('%H:%M:%S', time.localtime(t0))}\t"
                    f"{time.time() - t0:.1f}\t{rc}\n")
        sched.flush()
        print(f"[{i + 1}/{len(jobs)}] {tag} rc={rc} {time.time() - t0:.1f}s", flush=True)
print(f"done: {out}")
