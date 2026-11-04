"""M17: build m15 / m16 / m16nf, run the interleaving test on m16nf (eager, graph), then 3 launches of bench_ab.py
(all three versions and vLLM interleaved in one process), fp16, hidden 5120.

Usage: python3 exp/M17_epoch_ab/run.py          (refuses to run while other GPU processes exist)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from ncclrun import ROOT, gpu_processes, gpu_state, snapshot_env  # noqa: E402

EXP = ROOT / "exp/M17_epoch_ab"
PY = ROOT / ".venv-vllm/bin/python"
TORCHRUN = str(ROOT / ".venv-vllm/bin/torchrun")
CUDA = {"CUDA_HOME": "/usr/local/cuda", "PATH": f"/usr/local/cuda/bin:{os.environ['PATH']}"}
SRC = {"m15": ("m15_fused_ar_norm", ROOT / "exp/M15_bf16/fused_ar_norm.cu"),
       "m16": ("m16_fused_ar_norm", ROOT / "exp/M16_epoch_fix/fused_ar_norm.cu"),
       "m16nf": ("m17nf_fused_ar_norm", EXP / "fused_ar_norm_nf.cu")}
out = ROOT / "results" / "M17_epoch_ab" / time.strftime("%Y%m%d_%H%M%S")
if procs := gpu_processes():
    sys.exit(f"GPUs have running processes; not measuring (R13):\n{procs}")
out.mkdir(parents=True)
snapshot_env(out)
for k, (name, src) in SRC.items():
    d = out / "ext" / k
    d.mkdir(parents=True)
    code = (f"from torch.utils.cpp_extension import load; load(name='{name}', sources=['{src}'], "
            f"extra_cuda_cflags=['-O3','-gencode=arch=compute_80,code=sm_80'], build_directory='{d}', verbose=True)")
    r = subprocess.run([str(PY), "-c", code], capture_output=True, text=True, env={**os.environ, **CUDA})
    (out / f"build_{k}.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        sys.exit(f"build {k} failed")
    subprocess.run(["cp", str(src), str(out / f"{k}_{src.name}")])
subprocess.run(["cp", str(EXP / "bench_ab.py"), str(out)])
jobs = [(f"test_m16nf_{m}", [TORCHRUN, "--nproc-per-node", "8", "--master-port", "29671",
                             str(ROOT / "exp/M16_epoch_fix/test_interleave.py"), "--src", str(SRC["m16nf"][1]),
                             "--name", SRC["m16nf"][0], "--ext-dir", str(out / "ext/m16nf"), "--mode", m])
        for m in ("eager", "graph")]
jobs += [(f"ab_r{r}", [TORCHRUN, "--nproc-per-node", "8", "--master-port", "29671", str(EXP / "bench_ab.py"),
                       "--dtype", "float16", "--hidden", "5120", "--tokens", "8,16,32,64,128", "--seed", str(r),
                       "--ext-root", str(out / "ext")]) for r in range(3)]
with open(out / "schedule.tsv", "w") as sched:
    sched.write("order\ttag\tgpu_state_before\tstart\tseconds\trc\n")
    for i, (tag, cmd) in enumerate(jobs):
        if procs := gpu_processes():
            sys.exit(f"stopped before job {i}: other GPU processes appeared:\n{procs}")
        state, t0 = gpu_state(), time.time()
        with open(out / f"{tag}.log", "w") as log:
            rc = subprocess.run(["timeout", "--kill-after=10", "1500", *cmd, "--out", str(out / f"{tag}.json")],
                                stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **CUDA}).returncode
        sched.write(f"{i}\t{tag}\t{state}\t{time.strftime('%H:%M:%S', time.localtime(t0))}\t{time.time() - t0:.1f}\t{rc}\n")
        sched.flush()
        print(f"[{i + 1}/{len(jobs)}] {tag} rc={rc} {time.time() - t0:.1f}s", flush=True)
print(f"done: {out}")
