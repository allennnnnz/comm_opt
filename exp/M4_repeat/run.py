"""M4: repeat the M0 points behind F1-F5, 5 independent launches each, randomized order, two GPU sets.

Usage: python3 exp/M4_repeat/run.py          (refuses to run while other GPU processes exist)
       DRY_RUN=1 python3 exp/M4_repeat/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from ncclrun import Job, run_all  # noqa: E402

SEED = 20261102
REPS = 5
PLAN = [  # (collective, GPU counts, min size, max size)  # phenomena covered
    ("sendrecv", (2, 4, 8), "4M", "256M"),       # F1
    ("alltoall", (2, 8), "64M", "4G"),           # F2
    ("all_gather", (2, 8), "1M", "4G"),          # F3, F4
    ("reduce_scatter", (2, 8), "1M", "4G"),      # F3, F4
    ("all_reduce", (2, 8), "1M", "4G"),          # F3, F4, F5 control
    ("broadcast", (2, 8), "1M", "2G"),           # F4, F5
]

jobs = [Job(coll, ng, gs, lo, hi, rep)
        for coll, ngs, lo, hi in PLAN for ng in ngs for gs in ("A", "B") for rep in range(REPS)]
run_all(jobs, "M4_repeat", SEED)
