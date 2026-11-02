"""M5 inference driver: 3 independent launches per model, each running every config in its own random order.

Usage: python3 exp/M5_e2e_comm_share/run_infer.py   (refuses to run while other GPU processes exist)
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp" / "lib"))
from ncclrun import gpu_processes  # noqa: E402

MODELS = {  # name -> (HF id, configs)
    "tinyllama": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                  "prefill:1:512,prefill:1:2000,decode:1:128,decode:8:128,decode:32:128,decode:128:128"),
    "qwen32b": ("Qwen/Qwen2.5-32B-Instruct",
                "prefill:1:512,prefill:1:2048,prefill:1:8192,decode:1:128,decode:8:128,decode:32:128,decode:128:128"),
}
SEEDS = (0, 1, 2)

out = ROOT / "results" / "M5_e2e_comm_share" / time.strftime("%Y%m%d_%H%M%S")
out.mkdir(parents=True)
subprocess.run(f"nvidia-smi --query-gpu=index,name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu --format=csv "
               f"> {out}/env.txt; git -C {ROOT} log -1 --format='comm_opt %h' >> {out}/env.txt", shell=True)
for model, (hf_id, configs) in MODELS.items():
    for seed in SEEDS:
        if procs := gpu_processes():
            sys.exit(f"stopped: other GPU processes appeared (R13):\n{procs}")
        d = out / f"infer_{model}_s{seed}"
        d.mkdir()
        # sample GPU clocks and temperature every 100 ms while the workload runs (Chopper: DVFS matters)
        mon = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,clocks.sm,temperature.gpu,power.draw",
                                "--format=csv,noheader", "-lms", "100"], stdout=open(d / "gpu_monitor.csv", "w"))
        t0 = time.time()
        with open(d / "run.log", "w") as log:
            rc = subprocess.run([str(ROOT / ".venv-vllm/bin/python"), str(ROOT / "exp/M5_e2e_comm_share/infer.py"),
                                 "--model", hf_id, "--configs", configs, "--seed", str(seed), "--out", str(d)],
                                stdout=log, stderr=subprocess.STDOUT).returncode
        mon.terminate()
        print(f"{model} seed {seed}: rc={rc} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
print(f"done: {out}", file=sys.stderr)
