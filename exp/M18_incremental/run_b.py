"""M18 step b: per-size block count. Calibration run (3 launches, blocks 16..108) -> table; validation run
(5 fresh launches): v0 fixed 64 blocks vs v0 with the table vs vLLM. New measurement method (step a).

Usage: python3 exp/M18_incremental/run_b.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import statistics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from ncclrun import ROOT, gpu_processes, gpu_state, snapshot_env  # noqa: E402

EXP = ROOT / "exp/M18_incremental"
TORCHRUN = str(ROOT / ".venv-vllm/bin/torchrun")
PY = str(ROOT / ".venv-vllm/bin/python")
CUDA = {"CUDA_HOME": "/usr/local/cuda", "PATH": f"/usr/local/cuda/bin:{os.environ['PATH']}"}
METHOD = ["--warmup-ms", "2000", "--warm-replays", "50", "--discard-first", "1"]
CTAS = [16, 24, 32, 48, 64, 80, 96, 108]
TOKENS = "1,2,4,8,16,32,64,128"
out = ROOT / "results" / "M18_incremental" / ("b_" + time.strftime("%Y%m%d_%H%M%S"))
if procs := gpu_processes():
    sys.exit(f"GPUs have running processes; not measuring (R13):\n{procs}")
(out / "ext/v0").mkdir(parents=True)
snapshot_env(out)
code = (f"from torch.utils.cpp_extension import load; load(name='m18_v0', sources=['{EXP / 'kernels/v0.cu'}'], "
        f"extra_cuda_cflags=['-O3','-gencode=arch=compute_80,code=sm_80'], build_directory='{out / 'ext/v0'}', verbose=True)")
r = subprocess.run([PY, "-c", code], capture_output=True, text=True, env={**os.environ, **CUDA})
(out / "build_v0.log").write_text(r.stdout + r.stderr)
if r.returncode:
    sys.exit("build failed")
subprocess.run(["cp", str(EXP / "kernels/v0.cu"), str(EXP / "bench_inc.py"), str(out)])
sched = open(out / "schedule.tsv", "w")
sched.write("order\ttag\tgpu_state_before\tstart\tseconds\trc\n")
n = 0


def launch(tag, impls, seed, extra=()):
    global n
    if procs := gpu_processes():
        sys.exit(f"stopped before {tag}: other GPU processes appeared:\n{procs}")
    cmd = ["timeout", "--kill-after=10", "1500", TORCHRUN, "--nproc-per-node", "8", "--master-port", "29683",
           str(EXP / "bench_inc.py"), "--versions", f"v0={EXP / 'kernels/v0.cu'}", "--impls", impls,
           "--tokens", TOKENS, "--seed", str(seed), "--ext-root", str(out / "ext"), "--out", str(out / f"{tag}.json"),
           *METHOD, *extra]
    state, t0 = gpu_state(), time.time()
    with open(out / f"{tag}.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **CUDA}).returncode
    sched.write(f"{n}\t{tag}\t{state}\t{time.strftime('%H:%M:%S', time.localtime(t0))}\t{time.time() - t0:.1f}\t{rc}\n")
    sched.flush()
    n += 1
    print(f"[{n}] {tag} rc={rc} {time.time() - t0:.1f}s", flush=True)
    return rc


# calibration
for s in range(3):
    launch(f"calib_s{s}", ",".join(f"P_v0_c{c}" for c in CTAS), 100 + s)
med = {}
for s in range(3):
    for r in json.load(open(out / f"calib_s{s}.json"))["rows"]:
        if r["correct"] and not r["error"]:
            med.setdefault((r["tokens"], r["ctas"]), []).append(max(r["per_rank_us"]))
table = {}
for t in sorted({k[0] for k in med}):
    cand = {c: statistics.median(v) for (tt, c), v in med.items() if tt == t and len(v) == 3}
    table[t] = min(cand, key=cand.get)
json.dump(table, open(out / "table_v0.json", "w"), indent=1)
print("table:", table, flush=True)
# validation
for s in range(5):
    launch(f"valid_s{s}", "U_vllm,P_v0_c64,P_v0_tab", 200 + s, ["--table", str(out / "table_v0.json")])
sched.close()
print(f"done: {out}")
