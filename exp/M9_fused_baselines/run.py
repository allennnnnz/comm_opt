"""M9: existing fused all-reduce + residual + RMSNorm on 8x A100 vs unfused, 3 launches each, seeded random order.

Launch kinds (each covers all token counts for one hidden size):
  torch     bench_fused.py: U_vllm, U_nccl, F_trt_one, F_trt_two (impl order shuffled per launch)
  lmd       LMDeploy test_comm, original dispatch (<= 512 KB falls back to all-reduce + separate norm)
  lmdforce  LMDeploy test_comm built with -DM9_FORCE_FUSED (fused kernel at every size)

Usage: python3 exp/M9_fused_baselines/run.py          (refuses to run while other GPU processes exist)
       DRY_RUN=1 python3 exp/M9_fused_baselines/run.py
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

SEED = 20261104
REPS = 3
TOKENS = {5120: "1,8,16,32,64,128", 8192: "1,8,16,32,64,128,256"}
LMD = ROOT / "third_party/lmdeploy"
LMD_BIN = {"lmd": LMD / "build/bin/test_comm", "lmdforce": LMD / "build_force/bin/test_comm"}
CUDA = {"CUDA_HOME": "/usr/local/cuda", "PATH": f"/usr/local/cuda/bin:{os.environ['PATH']}"}

out = ROOT / "results" / "M9_fused_baselines" / time.strftime("%Y%m%d_%H%M%S")
jobs = [(kind, h, r) for kind in ("torch", "lmd", "lmdforce") for h in TOKENS for r in range(REPS)]
random.Random(SEED).shuffle(jobs)


def build(kind, hidden, rep):
    tag = f"{kind}_h{hidden}_r{rep}"
    if kind == "torch":
        cmd = [str(ROOT / ".venv-vllm/bin/torchrun"), "--nproc-per-node", "8", "--master-port", "29629",
               str(ROOT / "exp/M9_fused_baselines/bench_fused.py"), "--out", str(out / f"{tag}.json"),
               "--seed", str(rep), "--hidden", str(hidden), "--tokens", TOKENS[hidden]]
        return tag, cmd, CUDA
    env = {"M9_HIDDEN": str(hidden), "M9_TOKENS": TOKENS[hidden], "M9_WARMUP": "10", "M9_ITERS": "1000",
           "M9_B2B": "20"}
    return tag, [str(LMD_BIN[kind]), "cuda-ipc"], env


if os.environ.get("DRY_RUN"):
    for j in jobs:
        tag, cmd, env = build(*j)
        print(tag, " ".join(f"{k}={v}" for k, v in env.items() if k != "PATH"), shlex.join(cmd))
    print(f"{len(jobs)} launches", file=sys.stderr)
    sys.exit(0)
for b in LMD_BIN.values():
    if not b.exists():
        sys.exit(f"missing {b}")
if procs := gpu_processes():
    sys.exit(f"GPUs have running processes; not measuring (R13):\n{procs}")
out.mkdir(parents=True)
snapshot_env(out)
with open(out / "env.txt", "a") as f:
    for c in (f"git -C {LMD} log -1 --format='lmdeploy %H'", f"git -C {LMD} diff --stat",
              f"{ROOT}/.venv-vllm/bin/python -c 'import flashinfer, vllm, torch; "
              "print(\"flashinfer\", flashinfer.__version__, \"vllm\", vllm.__version__, \"torch\", torch.__version__)'",
              "/usr/local/cuda/bin/nvcc --version | tail -2"):
        f.write(subprocess.run(c, shell=True, capture_output=True, text=True).stdout)
subprocess.run(["cp", str(ROOT / "exp/M9_fused_baselines/lmdeploy_m9.patch"), str(out)])
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
