"""M11: build the three probes, run each 3 times in seeded random order, then one nsys GPU-metrics pass.

Launches: rs_probe (part A), car_timed (part B, stamped), car_plain (part B, unstamped, overhead check),
3 seeds each, shuffled; then nsys --gpu-metrics-set=ga100 over car_plain and rs_probe in "metrics" mode (part C).

Usage: python3 exp/M11_before_prototype/run.py          (refuses to run while other GPU processes exist)
       DRY_RUN=1 python3 exp/M11_before_prototype/run.py
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

SEED = 20261107
REPS = 3
EXP = ROOT / "exp/M11_before_prototype"
NVCC = "/usr/local/cuda/bin/nvcc"
NSYS = "/usr/local/cuda/bin/nsys"
out = ROOT / "results" / "M11_before_prototype" / time.strftime("%Y%m%d_%H%M%S")
BIN = out / "bin"
BUILDS = {"rs_probe": ("rs_probe.cu", []), "car_timed": ("car_timed.cu", []), "car_plain": ("car_timed.cu", ["-DNO_STAMPS"])}

jobs = [(name, r) for name in BUILDS for r in range(REPS)]
random.Random(SEED).shuffle(jobs)
jobs += [("nsys_metrics", 0)]


def cmd_for(name, rep):
    if name != "nsys_metrics":
        return [str(BIN / name), str(rep)]
    script = f"{BIN / 'car_plain'} metrics 500 0 && {BIN / 'rs_probe'} metrics 500 0"
    return [NSYS, "profile", "-t", "cuda,nvtx", "--gpu-metrics-devices=all", "--gpu-metrics-set=ga100",
            "--gpu-metrics-frequency=100000", "-o", str(out / "gpu_metrics"), "--force-overwrite", "true",
            "bash", "-c", script]


if os.environ.get("DRY_RUN"):
    for j in jobs:
        print(j[0], j[1], shlex.join(cmd_for(*j)))
    sys.exit(0)
if procs := gpu_processes():
    sys.exit(f"GPUs have running processes; not measuring (R13):\n{procs}")
BIN.mkdir(parents=True)
snapshot_env(out)
with open(out / "env.txt", "a") as f:
    f.write(subprocess.run(f"{NVCC} --version | tail -2; {NSYS} --version", shell=True, capture_output=True,
                           text=True).stdout)
for name, (src, flags) in BUILDS.items():
    cmd = [NVCC, "-O3", "-arch=sm_80", "-std=c++20", *flags, "-o", str(BIN / name), str(EXP / src)]
    r = subprocess.run(cmd, cwd=EXP, capture_output=True, text=True)
    (out / f"build_{name}.log").write_text(shlex.join(cmd) + "\n" + r.stdout + r.stderr)
    if r.returncode:
        sys.exit(f"build {name} failed, see build_{name}.log")
subprocess.run(["cp", str(EXP / "rs_probe.cu"), str(EXP / "car_timed.cu"), str(out)])
with open(out / "schedule.tsv", "w") as sched:
    sched.write(f"# seed={SEED}\norder\ttag\trep\tgpu_state_before\tstart\tseconds\trc\n")
    for i, (name, rep) in enumerate(jobs):
        if procs := gpu_processes():
            sys.exit(f"stopped before job {i}: other GPU processes appeared:\n{procs}")
        tag = f"{name}_r{rep}"
        state, t0 = gpu_state(), time.time()
        with open(out / f"{tag}.csv" if name != "nsys_metrics" else out / f"{tag}.log", "w") as fo:
            rc = subprocess.run(cmd_for(name, rep), stdout=fo, stderr=subprocess.STDOUT).returncode
        sched.write(f"{i}\t{tag}\t{rep}\t{state}\t{time.strftime('%H:%M:%S', time.localtime(t0))}\t"
                    f"{time.time() - t0:.1f}\t{rc}\n")
        sched.flush()
        print(f"[{i + 1}/{len(jobs)}] {tag} rc={rc} {time.time() - t0:.1f}s", flush=True)
print(f"done: {out}")
