"""M6: all-reduce bake-off. 4 nccl-tests variants + 1 torch-harness group, 5 launches each, seeded random order.

Usage: python3 exp/M6_allreduce_bakeoff/run.py          (refuses to run while other GPU processes exist)
       DRY_RUN=1 python3 exp/M6_allreduce_bakeoff/run.py
"""
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from ncclrun import ROOT, Job, command, gpu_processes, gpu_state, snapshot_env  # noqa: E402

SEED = 20261105
REPS = 5
MSCCLPP = ROOT / "third_party/mscclpp/build/lib/libmscclpp_nccl.so"
BASE = ("-f", "2", "-g", "1", "-n", "40", "-w", "10", "-a", "3", "-U", "1", "-c", "1", "-d", "half")
NCCL_VARIANTS = {  # variant -> (extra nccl-tests args, extra env)
    "nccl_eager": ((), {}),
    "nccl_graph": (("-G", "20"), {}),
    "nccl_symm_graph": (("-R", "2", "-G", "20"), {}),
    "mscclpp_graph": (("-G", "20"), {"LD_PRELOAD": str(MSCCLPP)}),
}

out = ROOT / "results" / "M6_allreduce_bakeoff" / time.strftime("%Y%m%d_%H%M%S")
jobs = [("nccltests", v, r) for v in NCCL_VARIANTS for r in range(REPS)] + [("torch", "torch_group", r) for r in range(REPS)]
random.Random(SEED).shuffle(jobs)


def build(kind, variant, rep):
    tag = f"{variant}_r{rep}"
    if kind == "nccltests":
        extra, env = NCCL_VARIANTS[variant]
        job = Job("all_reduce", 8, "A", "1K", "8M", rep, variant, env, args=BASE + extra)
        return job.tag, command(job, out)  # job.tag also names the -J json and NCCL debug logs
    return tag, [str(ROOT / ".venv-vllm/bin/torchrun"), "--nproc-per-node", "8", "--master-port", "29621",
                 str(ROOT / "exp/M6_allreduce_bakeoff/bench_torch.py"), "--out", str(out / f"{tag}.json"),
                 "--seed", str(rep)]


if os.environ.get("DRY_RUN"):
    for j in jobs:
        print(shlex.join(build(*j)[1]))
    sys.exit(0)
if not MSCCLPP.exists():
    sys.exit(f"missing {MSCCLPP}; build MSCCL++ first")
if procs := gpu_processes():
    sys.exit(f"GPUs have running processes; not measuring (R13):\n{procs}")
out.mkdir(parents=True)
snapshot_env(out)
with open(out / "schedule.tsv", "w") as sched:
    sched.write(f"# seed={SEED}\norder\ttag\tkind\tvariant\trep\tgpu_state_before\tstart\tseconds\trc\n")
    for i, (kind, variant, rep) in enumerate(jobs):
        if procs := gpu_processes():
            sys.exit(f"stopped before job {i}: other GPU processes appeared:\n{procs}")
        tag, cmd = build(kind, variant, rep)
        state, t0 = gpu_state(), time.time()
        with open(out / f"{tag}.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        sched.write(f"{i}\t{tag}\t{kind}\t{variant}\t{rep}\t{state}\t{time.strftime('%H:%M:%S', time.localtime(t0))}\t"
                    f"{time.time() - t0:.1f}\t{rc}\n")
        sched.flush()
        print(f"[{i + 1}/{len(jobs)}] {tag} rc={rc} {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
print(f"done: {out}", file=sys.stderr)
