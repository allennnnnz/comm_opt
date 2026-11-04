"""M14: build the prototype extension, then 3 launches x {hidden 5120, 8192} in seeded random order.

Usage: python3 exp/M14_fused_prototype/run.py          (refuses to run while other GPU processes exist)
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

SEED = 20261110
EXP = ROOT / "exp/M14_fused_prototype"
PY = ROOT / ".venv-vllm/bin/python"
CUDA = {"CUDA_HOME": "/usr/local/cuda", "PATH": f"/usr/local/cuda/bin:{os.environ['PATH']}"}
IMPLS = {5120: "U_vllm,F_trt_one,F_trt_two,P_ours_c16,P_ours_c32,P_ours_c64,P_ours_c108",
         8192: "U_vllm,P_ours_c16,P_ours_c32,P_ours_c64,P_ours_c108"}  # TRT-LLM cannot launch at 8192 on A100 (M9)
out = ROOT / "results" / "M14_fused_prototype" / time.strftime("%Y%m%d_%H%M%S")
jobs = [(h, r) for h in IMPLS for r in range(3)]
random.Random(SEED).shuffle(jobs)
if procs := gpu_processes():
    sys.exit(f"GPUs have running processes; not measuring (R13):\n{procs}")
ext = out / "ext"
ext.mkdir(parents=True)
snapshot_env(out)
build = (f"from torch.utils.cpp_extension import load; load(name='m14_fused_ar_norm', "
         f"sources=['{EXP / 'fused_ar_norm.cu'}'], extra_cuda_cflags=['-O3','-gencode=arch=compute_80,code=sm_80'], "
         f"build_directory='{ext}', verbose=True)")
r = subprocess.run([str(PY), "-c", build], capture_output=True, text=True, env={**os.environ, **CUDA})
(out / "build.log").write_text(r.stdout + r.stderr)
if r.returncode:
    sys.exit("extension build failed, see build.log")
subprocess.run(["cp", str(EXP / "fused_ar_norm.cu"), str(EXP / "bench_proto.py"), str(out)])
with open(out / "env.txt", "a") as f:
    f.write(subprocess.run("/usr/local/cuda/bin/nvcc --version | tail -2", shell=True, capture_output=True,
                           text=True).stdout)
with open(out / "schedule.tsv", "w") as sched:
    sched.write(f"# seed={SEED}\norder\ttag\thidden\trep\tgpu_state_before\tstart\tseconds\trc\n")
    for i, (h, rep) in enumerate(jobs):
        if procs := gpu_processes():
            sys.exit(f"stopped before job {i}: other GPU processes appeared:\n{procs}")
        tag = f"proto_h{h}_r{rep}"
        cmd = [str(ROOT / ".venv-vllm/bin/torchrun"), "--nproc-per-node", "8", "--master-port", "29641",
               str(EXP / "bench_proto.py"), "--hidden", str(h), "--tokens", "8,16,32,64,128", "--impls", IMPLS[h],
               "--seed", str(rep), "--ext-dir", str(ext), "--out", str(out / f"{tag}.json")]
        state, t0 = gpu_state(), time.time()
        with open(out / f"{tag}.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **CUDA}).returncode
        sched.write(f"{i}\t{tag}\t{h}\t{rep}\t{state}\t{time.strftime('%H:%M:%S', time.localtime(t0))}\t"
                    f"{time.time() - t0:.1f}\t{rc}\n")
        sched.flush()
        print(f"[{i + 1}/{len(jobs)}] {tag} rc={rc} {time.time() - t0:.1f}s", flush=True)
print(f"done: {out}")
