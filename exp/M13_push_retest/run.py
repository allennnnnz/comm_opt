"""M13: build m13_probe, run it 3 times (seeds 0..2) in seeded random order.

Usage: python3 exp/M13_push_retest/run.py          (refuses to run while other GPU processes exist)
"""
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from ncclrun import ROOT, gpu_processes, gpu_state, snapshot_env  # noqa: E402

SEED = 20261109
EXP = ROOT / "exp/M13_push_retest"
NVCC = "/usr/local/cuda/bin/nvcc"
out = ROOT / "results" / "M13_push_retest" / time.strftime("%Y%m%d_%H%M%S")
reps = list(range(3))
random.Random(SEED).shuffle(reps)
if procs := gpu_processes():
    sys.exit(f"GPUs have running processes; not measuring (R13):\n{procs}")
(out / "bin").mkdir(parents=True)
snapshot_env(out)
with open(out / "env.txt", "a") as f:
    f.write(subprocess.run(f"{NVCC} --version | tail -2", shell=True, capture_output=True, text=True).stdout)
cmd = [NVCC, "-O3", "-arch=sm_80", "-std=c++20", "-o", str(out / "bin/m13_probe"), str(EXP / "m13_probe.cu")]
r = subprocess.run(cmd, capture_output=True, text=True)
(out / "build.log").write_text(shlex.join(cmd) + "\n" + r.stdout + r.stderr)
if r.returncode:
    sys.exit("build failed")
subprocess.run(["cp", str(EXP / "m13_probe.cu"), str(out)])
with open(out / "schedule.tsv", "w") as sched:
    sched.write(f"# seed={SEED}\norder\ttag\tgpu_state_before\tstart\tseconds\trc\n")
    for i, rep in enumerate(reps):
        if procs := gpu_processes():
            sys.exit(f"stopped before job {i}: other GPU processes appeared:\n{procs}")
        tag = f"m13_probe_r{rep}"
        state, t0 = gpu_state(), time.time()
        with open(out / f"{tag}.csv", "w") as fo:
            rc = subprocess.run([str(out / "bin/m13_probe"), str(rep)], stdout=fo, stderr=subprocess.STDOUT).returncode
        sched.write(f"{i}\t{tag}\t{state}\t{time.strftime('%H:%M:%S', time.localtime(t0))}\t{time.time() - t0:.1f}\t{rc}\n")
        sched.flush()
        print(f"[{i + 1}/3] {tag} rc={rc} {time.time() - t0:.1f}s", flush=True)
print(f"done: {out}")
