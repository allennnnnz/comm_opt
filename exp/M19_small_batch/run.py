"""M19: small-batch diagnosis. 5 torch launches (U_vllm, A_vllm, P_v0_c64, S_v0s_c64) and 5+5 launches of the
warmed-up vLLM phase harness (stamped / plain), seeded interleaved order; M18 measurement method.

Usage: python3 exp/M19_small_batch/run.py
"""
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from ncclrun import ROOT, gpu_processes, gpu_state, snapshot_env  # noqa: E402

EXP = ROOT / "exp/M19_small_batch"
TORCHRUN = str(ROOT / ".venv-vllm/bin/torchrun")
PY = str(ROOT / ".venv-vllm/bin/python")
NVCC = "/usr/local/cuda/bin/nvcc"
CUDA = {"CUDA_HOME": "/usr/local/cuda", "PATH": f"/usr/local/cuda/bin:{os.environ['PATH']}"}
VERS = {"v0": ROOT / "exp/M18_incremental/kernels/v0.cu", "v0s": EXP / "kernels/v0_stamped.cu"}
out = ROOT / "results" / "M19_small_batch" / time.strftime("%Y%m%d_%H%M%S")
if procs := gpu_processes():
    sys.exit(f"GPUs have running processes; not measuring (R13):\n{procs}")
(out / "bin").mkdir(parents=True)
snapshot_env(out)
for v, src in VERS.items():
    (out / "ext" / v).mkdir(parents=True)
    code = (f"from torch.utils.cpp_extension import load; load(name='m18_{v}', sources=['{src}'], "
            f"extra_cuda_cflags=['-O3','-gencode=arch=compute_80,code=sm_80'], build_directory='{out / 'ext' / v}', verbose=True)")
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, env={**os.environ, **CUDA})
    (out / f"build_{v}.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        sys.exit(f"build {v} failed")
for name, flags in (("car_timed_warm", []), ("car_plain_warm", ["-DNO_STAMPS"])):
    r = subprocess.run([NVCC, "-O3", "-arch=sm_80", "-std=c++20", *flags, "-o", str(out / "bin" / name),
                        str(EXP / "car_timed_warm.cu")], capture_output=True, text=True, cwd=EXP)
    (out / f"build_{name}.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        sys.exit(f"build {name} failed")
subprocess.run(["cp", str(VERS["v0"]), str(VERS["v0s"]), str(EXP / "bench_m19.py"), str(EXP / "car_timed_warm.cu"), str(out)])
versions = ",".join(f"{v}={p}" for v, p in VERS.items())
jobs = [("torch", s) for s in range(5)] + [("car_timed_warm", s) for s in range(5)] + [("car_plain_warm", s) for s in range(5)]
random.Random(20261114).shuffle(jobs)
with open(out / "schedule.tsv", "w") as sched:
    sched.write("order\ttag\tgpu_state_before\tstart\tseconds\trc\n")
    for i, (kind, s) in enumerate(jobs):
        if procs := gpu_processes():
            sys.exit(f"stopped before job {i}: other GPU processes appeared:\n{procs}")
        tag = f"{kind}_s{s}"
        if kind == "torch":
            cmd = ["timeout", "--kill-after=10", "1500", TORCHRUN, "--nproc-per-node", "8", "--master-port", "29685",
                   str(EXP / "bench_m19.py"), "--versions", versions, "--impls", "U_vllm,A_vllm,P_v0_c64,S_v0s_c64",
                   "--tokens", "1,2,4,8,16,32,64,128", "--seed", str(s), "--ext-root", str(out / "ext"),
                   "--warmup-ms", "2000", "--warm-replays", "50", "--discard-first", "1", "--out", str(out / f"{tag}.json")]
            fout = out / f"{tag}.log"
        else:
            cmd = ["timeout", "--kill-after=10", "900", str(out / "bin" / kind), str(s)]
            fout = out / f"{tag}.csv"
        state, t0 = gpu_state(), time.time()
        with open(fout, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **CUDA}).returncode
        sched.write(f"{i}\t{tag}\t{state}\t{time.strftime('%H:%M:%S', time.localtime(t0))}\t{time.time() - t0:.1f}\t{rc}\n")
        sched.flush()
        print(f"[{i + 1}/{len(jobs)}] {tag} rc={rc} {time.time() - t0:.1f}s", flush=True)
print(f"done: {out}")
