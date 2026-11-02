"""M5 training driver: every (model, TP x FSDP layout) x 3 launches, in a seeded random order.

Usage: python3 exp/M5_e2e_comm_share/run_train.py   (refuses to run while other GPU processes exist)
"""
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp" / "lib"))
from ncclrun import gpu_processes  # noqa: E402

CONFIGS = [f"llama3_{m}_tp{tp}_dp{8 // tp}" for m in ("1b", "8b") for tp in (8, 4, 2)]
REPS = 3
SEED = 20261104

out = ROOT / "results" / "M5_e2e_comm_share" / time.strftime("%Y%m%d_%H%M%S")
out.mkdir(parents=True)
jobs = [(c, r) for c in CONFIGS for r in range(REPS)]
random.Random(SEED).shuffle(jobs)
(out / "schedule.txt").write_text(f"# seed={SEED}\n" + "\n".join(f"train_{c}_r{r}" for c, r in jobs) + "\n")
env = dict(os.environ, PYTHONPATH=str(ROOT / "exp/M5_e2e_comm_share"), HF_HUB_OFFLINE="1")
for c, r in jobs:
    if procs := gpu_processes():
        sys.exit(f"stopped: other GPU processes appeared (R13):\n{procs}")
    d = out / f"train_{c}_r{r}"
    d.mkdir()
    mon = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,clocks.sm,temperature.gpu,power.draw",
                            "--format=csv,noheader", "-lms", "100"], stdout=open(d / "gpu_monitor.csv", "w"))
    t0 = time.time()
    with open(d / "run.log", "w") as log:
        rc = subprocess.run([str(ROOT / ".venv-titan/bin/torchrun"), "--nproc-per-node", "8", "--master-port", "29611",
                             "-m", "torchtitan.train", "--module", "titan_recipes", "--config", c,
                             "--output-dir", str(d)], stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    mon.terminate()
    print(f"{c} r{r}: rc={rc} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
print(f"done: {out}", file=sys.stderr)
