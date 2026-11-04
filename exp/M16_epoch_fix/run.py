"""M16: (1) old M15 kernel under the interleaving test (expected to fail; 120 s timeout), (2) new kernel eager and
graph interleaving tests, (3) speed regression: new kernel vs vLLM, fp16 and bf16, hidden 5120, 3 launches each.

Usage: python3 exp/M16_epoch_fix/run.py          (refuses to run while other GPU processes exist)
"""
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from ncclrun import ROOT, gpu_processes, gpu_state, snapshot_env  # noqa: E402

SEED = 20261112
EXP = ROOT / "exp/M16_epoch_fix"
PY = ROOT / ".venv-vllm/bin/python"
TORCHRUN = str(ROOT / ".venv-vllm/bin/torchrun")
CUDA = {"CUDA_HOME": "/usr/local/cuda", "PATH": f"/usr/local/cuda/bin:{os.environ['PATH']}"}
out = ROOT / "results" / "M16_epoch_fix" / time.strftime("%Y%m%d_%H%M%S")
if procs := gpu_processes():
    sys.exit(f"GPUs have running processes; not measuring (R13):\n{procs}")
out.mkdir(parents=True)
snapshot_env(out)
EXTS = {"old": ("m15_fused_ar_norm", ROOT / "exp/M15_bf16/fused_ar_norm.cu"),
        "new": ("m16_fused_ar_norm", EXP / "fused_ar_norm.cu")}
for key, (name, src) in EXTS.items():
    d = out / f"ext_{key}"
    d.mkdir()
    code = (f"from torch.utils.cpp_extension import load; load(name='{name}', sources=['{src}'], "
            f"extra_cuda_cflags=['-O3','-gencode=arch=compute_80,code=sm_80'], build_directory='{d}', verbose=True)")
    r = subprocess.run([str(PY), "-c", code], capture_output=True, text=True, env={**os.environ, **CUDA})
    (out / f"build_{key}.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        sys.exit(f"build {key} failed")
subprocess.run(["cp", str(EXP / "fused_ar_norm.cu"), str(EXP / "test_interleave.py"), str(EXP / "bench_proto.py"),
                str(out)])

jobs = [("test_old_eager", ["timeout", "--kill-after=10", "120", TORCHRUN, "--nproc-per-node", "8", "--master-port",
                            "29661", str(EXP / "test_interleave.py"), "--src", str(EXTS["old"][1]), "--name",
                            EXTS["old"][0], "--ext-dir", str(out / "ext_old"), "--mode", "eager"])]
for mode in ("eager", "graph"):
    jobs.append((f"test_new_{mode}", [TORCHRUN, "--nproc-per-node", "8", "--master-port", "29661",
                                      str(EXP / "test_interleave.py"), "--src", str(EXTS["new"][1]), "--name",
                                      EXTS["new"][0], "--ext-dir", str(out / "ext_new"), "--mode", mode]))
perf = [(dt, r) for dt in ("float16", "bfloat16") for r in range(3)]
random.Random(SEED).shuffle(perf)
for dt, r in perf:
    jobs.append((f"perf_{dt}_r{r}", [TORCHRUN, "--nproc-per-node", "8", "--master-port", "29661",
                                     str(EXP / "bench_proto.py"), "--dtype", dt, "--hidden", "5120",
                                     "--tokens", "8,16,32,64,128", "--seed", str(r),
                                     "--impls", "U_vllm,P_ours_c16,P_ours_c32,P_ours_c64,P_ours_c108",
                                     "--ext-dir", str(out / "ext_new")]))
with open(out / "schedule.tsv", "w") as sched:
    sched.write(f"# seed={SEED}\norder\ttag\tgpu_state_before\tstart\tseconds\trc\n")
    for i, (tag, cmd) in enumerate(jobs):
        if procs := gpu_processes():
            sys.exit(f"stopped before job {i}: other GPU processes appeared:\n{procs}")
        cmd = cmd + ["--out", str(out / f"{tag}.json")]
        state, t0 = gpu_state(), time.time()
        with open(out / f"{tag}.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **CUDA}).returncode
        sched.write(f"{i}\t{tag}\t{state}\t{time.strftime('%H:%M:%S', time.localtime(t0))}\t{time.time() - t0:.1f}\t{rc}\n")
        sched.flush()
        print(f"[{i + 1}/{len(jobs)}] {tag} rc={rc} {time.time() - t0:.1f}s", flush=True)
        if tag == "test_old_eager":
            time.sleep(5)  # let a killed run release the GPUs before the next job's R13 check
print(f"done: {out}")
