"""Shared runner for nccl-tests experiments: one MPI rank per GPU, randomized job order.

Each job is one independent process launch (R3). Jobs are shuffled with a recorded seed
(R11, D6) and the executed order, per-job GPU clock/temperature, and return codes are written
to schedule.tsv next to the raw outputs.
"""
import os
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "third_party" / "nccl-tests" / "build"
GPUSETS = {  # set A matches M0; set B uses other physical GPUs / rank order (R11)
    2: {"A": "0,1", "B": "6,7"},
    4: {"A": "0,1,2,3", "B": "4,5,6,7"},
    8: {"A": "0,1,2,3,4,5,6,7", "B": "0,2,4,6,1,3,5,7"},
}


@dataclass
class Job:
    coll: str
    ngpus: int
    gpuset: str
    minbytes: str
    maxbytes: str
    rep: int
    variant: str = "default"
    env: dict = field(default_factory=dict)
    args: tuple = ("-f", "2", "-g", "1", "-n", "40", "-w", "10", "-a", "1", "-U", "1", "-I", "1")

    @property
    def tag(self):
        return f"{self.coll}_g{self.ngpus}{self.gpuset}_{self.variant}_r{self.rep}"


def gpu_processes() -> str:
    """GPU processes other than our own nccl-tests binaries (which may linger briefly after a job)."""
    lines = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True).stdout.splitlines()
    return "\n".join(line for line in lines if line.strip() and str(BIN) not in line)


def gpu_state() -> str:
    q = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.split("\n")
    return ";".join(line.replace(", ", "/") for line in q if line)


def snapshot_env(out: Path):
    cmds = [
        "date -Is", "hostname",
        "nvidia-smi --query-gpu=index,name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu --format=csv",
        "nvidia-smi topo -m",
        f"git -C {ROOT / 'third_party/nccl-tests'} log -1 --format='nccl-tests %h'",
        f"git -C {ROOT} log -1 --format='comm_opt %h'",
        "ls -l /usr/lib/x86_64-linux-gnu/libnccl.so.2*",
    ]
    with open(out / "env.txt", "w") as f:
        for c in cmds:
            f.write(subprocess.run(c, shell=True, capture_output=True, text=True).stdout)


def command(job: Job, out: Path) -> list:
    env = {"CUDA_VISIBLE_DEVICES": GPUSETS[job.ngpus][job.gpuset], "NCCL_DEBUG": "INFO",
           "NCCL_DEBUG_SUBSYS": "INIT,ENV", "NCCL_DEBUG_FILE": str(out / f"{job.tag}.nccl.%h.%p.log"), **job.env}
    cmd = ["mpirun", "-np", str(job.ngpus), "--bind-to", "none", "--mca", "btl", "self,vader"]
    for k, v in env.items():
        cmd += ["-x", f"{k}={v}"]
    return cmd + [str(BIN / f"{job.coll}_perf"), "-b", job.minbytes, "-e", job.maxbytes, *job.args,
                  "-J", str(out / f"{job.tag}.json")]


def run_all(jobs: list, name: str, seed: int):
    out = ROOT / "results" / name / time.strftime("%Y%m%d_%H%M%S")
    dry = bool(os.environ.get("DRY_RUN"))
    random.Random(seed).shuffle(jobs)
    if dry:
        for j in jobs:
            print(shlex.join(command(j, out)))
        print(f"{len(jobs)} jobs, seed {seed}", file=sys.stderr)
        return
    if procs := gpu_processes():
        sys.exit(f"GPUs have running processes; not measuring (R13):\n{procs}")
    out.mkdir(parents=True)
    snapshot_env(out)
    with open(out / "schedule.tsv", "w") as sched:
        sched.write(f"# seed={seed}\norder\ttag\tcoll\tngpus\tgpuset\tvariant\trep\tenv\tgpu_state_before\t"
                    "start\tseconds\trc\n")
        for i, j in enumerate(jobs):
            if procs := gpu_processes():  # someone else started using the GPUs: stop, don't compete
                sys.exit(f"stopped before job {i}: other GPU processes appeared:\n{procs}")
            state, t0 = gpu_state(), time.time()
            with open(out / f"{j.tag}.log", "w") as log:
                rc = subprocess.run(command(j, out), stdout=log, stderr=subprocess.STDOUT).returncode
            sched.write(f"{i}\t{j.tag}\t{j.coll}\t{j.ngpus}\t{j.gpuset}\t{j.variant}\t{j.rep}\t"
                        f"{' '.join(f'{k}={v}' for k, v in j.env.items())}\t{state}\t"
                        f"{time.strftime('%H:%M:%S', time.localtime(t0))}\t{time.time() - t0:.1f}\t{rc}\n")
            sched.flush()
            print(f"[{i + 1}/{len(jobs)}] {j.tag} rc={rc} {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    print(f"done: {out}", file=sys.stderr)
