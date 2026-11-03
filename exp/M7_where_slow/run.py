"""M7: where is all-reduce slow? Sweep NCCL symmetric kernel x CTA count, low-latency-nccl, vLLM 1/2-stage,
MSCCL++ symmetric memory. 22 configs x 3 launches, seeded random order.

Usage: python3 exp/M7_where_slow/run.py          (refuses to run while other GPU processes exist)
       DRY_RUN=1 python3 exp/M7_where_slow/run.py
"""
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from ncclrun import BIN, ROOT, Job, command, gpu_processes, gpu_state, snapshot_env  # noqa: E402

SEED = 20261106
REPS = 3
LLNCCL_LIB = ROOT / "third_party/low-latency-nccl/build/lib"
LLNCCL_BIN = ROOT / "third_party/nccl-tests-llnccl/build"  # rebuilt against low-latency-nccl's own headers
MSCCLPP = ROOT / "third_party/mscclpp/build/lib/libmscclpp_nccl.so"
BASE = ("-f", "2", "-g", "1", "-n", "40", "-w", "10", "-a", "3", "-U", "1", "-c", "1", "-d", "half")
SYM = ("-R", "2", "-G", "20")

configs = {}  # name -> (kind, env, extra args)
configs["A_auto"] = ("nccl", {}, SYM)
for k, short in (("AllReduce_AGxLL_R", "agll"), ("AllReduce_RSxLD_AGxST", "rsag")):
    for c in (1, 2, 4, 8, 16, 32, 64):
        configs[f"A_{short}_c{c}"] = ("nccl", {"NCCL_SYM_KERNEL": k, "NCCL_SYM_CTAS": str(c)}, SYM)
configs["B_auto"] = ("llnccl", {"LD_LIBRARY_PATH": str(LLNCCL_LIB)}, SYM)
configs["B_agll"] = ("llnccl", {"LD_LIBRARY_PATH": str(LLNCCL_LIB), "NCCL_SYM_KERNEL": "AllReduce_AGxLL_R"}, SYM)
configs["B_rsag"] = ("llnccl", {"LD_LIBRARY_PATH": str(LLNCCL_LIB), "NCCL_SYM_KERNEL": "AllReduce_RSxLD_AGxST"}, SYM)
configs["C_vllm_1stage"] = ("torch", {"VLLM_CUSTOM_ALLREDUCE_ALGO": "1stage"}, ())
configs["C_vllm_2stage"] = ("torch", {"VLLM_CUSTOM_ALLREDUCE_ALGO": "2stage"}, ())
configs["D_mscclpp_sym0"] = ("nccl", {"LD_PRELOAD": str(MSCCLPP), "MSCCLPP_NCCL_SYMMETRIC_MEMORY": "0"}, ("-G", "20"))
configs["D_mscclpp_sym1"] = ("nccl", {"LD_PRELOAD": str(MSCCLPP), "MSCCLPP_NCCL_SYMMETRIC_MEMORY": "1"}, ("-G", "20"))

out = ROOT / "results" / "M7_where_slow" / time.strftime("%Y%m%d_%H%M%S")
jobs = [(name, r) for name in configs for r in range(REPS)]
random.Random(SEED).shuffle(jobs)


def build(name, rep):
    kind, env, extra = configs[name]
    if kind == "torch":
        cmd = [str(ROOT / ".venv-vllm/bin/torchrun"), "--nproc-per-node", "8", "--master-port", "29621",
               str(ROOT / "exp/M6_allreduce_bakeoff/bench_torch.py"), "--out", str(out / f"{name}_r{rep}.json"),
               "--seed", str(rep), "--impls", "vllm_car"]
        return f"{name}_r{rep}", cmd, env
    job = Job("all_reduce", 8, "A", "1K", "8M", rep, name, env, args=BASE + extra)
    cmd = command(job, out)
    if kind == "llnccl":
        cmd = [str(LLNCCL_BIN / "all_reduce_perf") if c == str(BIN / "all_reduce_perf") else c for c in cmd]
    return job.tag, cmd, {}


if os.environ.get("DRY_RUN"):
    for j in jobs:
        tag, cmd, env = build(*j)
        print(" ".join(f"{k}={v}" for k, v in env.items()), shlex.join(cmd))
    print(f"{len(jobs)} launches", file=sys.stderr)
    sys.exit(0)
for p in (MSCCLPP, LLNCCL_LIB / "libnccl.so.2", LLNCCL_BIN / "all_reduce_perf"):
    if not p.exists():
        sys.exit(f"missing {p}")
if procs := gpu_processes():
    sys.exit(f"GPUs have running processes; not measuring (R13):\n{procs}")
out.mkdir(parents=True)
snapshot_env(out)
with open(out / "schedule.tsv", "w") as sched:
    sched.write(f"# seed={SEED}\norder\ttag\tconfig\trep\tgpu_state_before\tstart\tseconds\trc\n")
    for i, (name, rep) in enumerate(jobs):
        if procs := gpu_processes():
            sys.exit(f"stopped before job {i}: other GPU processes appeared:\n{procs}")
        tag, cmd, env = build(name, rep)
        state, t0 = gpu_state(), time.time()
        with open(out / f"{tag}.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env}).returncode
        sched.write(f"{i}\t{tag}\t{name}\t{rep}\t{state}\t{time.strftime('%H:%M:%S', time.localtime(t0))}\t"
                    f"{time.time() - t0:.1f}\t{rc}\n")
        sched.flush()
        print(f"[{i + 1}/{len(jobs)}] {tag} rc={rc} {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
print(f"done: {out}", file=sys.stderr)
