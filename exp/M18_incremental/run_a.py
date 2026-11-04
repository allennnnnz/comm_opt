"""M18 step a: old (no warm-up, 3 warm replays) vs new (2 s all-GPU warm-up, 50 warm replays) measurement method,
5 launches each, seeded interleaved order; v0 at 16/32/64/108 blocks and vLLM; clocks recorded per config.

Usage: python3 exp/M18_incremental/run_a.py
"""
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from ncclrun import ROOT, gpu_processes, gpu_state, snapshot_env  # noqa: E402

EXP = ROOT / "exp/M18_incremental"
TORCHRUN = str(ROOT / ".venv-vllm/bin/torchrun")
PY = str(ROOT / ".venv-vllm/bin/python")
CUDA = {"CUDA_HOME": "/usr/local/cuda", "PATH": f"/usr/local/cuda/bin:{os.environ['PATH']}"}
MODES = {"old": ["--warmup-ms", "0", "--warm-replays", "3"], "new": ["--warmup-ms", "2000", "--warm-replays", "50"]}
IMPLS = "U_vllm,P_v0_c16,P_v0_c32,P_v0_c64,P_v0_c108"
out = ROOT / "results" / "M18_incremental" / ("a_" + time.strftime("%Y%m%d_%H%M%S"))
if procs := gpu_processes():
    sys.exit(f"GPUs have running processes; not measuring (R13):\n{procs}")
(out / "ext").mkdir(parents=True)
snapshot_env(out)
code = (f"from torch.utils.cpp_extension import load; load(name='m18_v0', sources=['{EXP / 'kernels/v0.cu'}'], "
        f"extra_cuda_cflags=['-O3','-gencode=arch=compute_80,code=sm_80'], build_directory='{out / 'ext/v0'}', "
        f"verbose=True)")
(out / "ext/v0").mkdir()
r = subprocess.run([PY, "-c", code], capture_output=True, text=True, env={**os.environ, **CUDA})
(out / "build_v0.log").write_text(r.stdout + r.stderr)
if r.returncode:
    sys.exit("build failed")
subprocess.run(["cp", str(EXP / "kernels/v0.cu"), str(EXP / "bench_inc.py"), str(out)])
jobs = [(m, s) for m in MODES for s in range(5)]
random.Random(20261113).shuffle(jobs)
with open(out / "schedule.tsv", "w") as sched:
    sched.write("order\ttag\tgpu_state_before\tstart\tseconds\trc\n")
    for i, (m, s) in enumerate(jobs):
        if procs := gpu_processes():
            sys.exit(f"stopped before job {i}: other GPU processes appeared:\n{procs}")
        tag = f"{m}_s{s}"
        cmd = ["timeout", "--kill-after=10", "1500", TORCHRUN, "--nproc-per-node", "8", "--master-port", "29681",
               str(EXP / "bench_inc.py"), "--versions", f"v0={EXP / 'kernels/v0.cu'}", "--impls", IMPLS,
               "--seed", str(s), "--ext-root", str(out / "ext"), "--out", str(out / f"{tag}.json"), *MODES[m]]
        state, t0 = gpu_state(), time.time()
        with open(out / f"{tag}.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **CUDA}).returncode
        sched.write(f"{i}\t{tag}\t{state}\t{time.strftime('%H:%M:%S', time.localtime(t0))}\t{time.time() - t0:.1f}\t{rc}\n")
        sched.flush()
        print(f"[{i + 1}/{len(jobs)}] {tag} rc={rc} {time.time() - t0:.1f}s", flush=True)
print(f"done: {out}")
