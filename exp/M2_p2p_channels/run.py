"""M2-P2P: do NCCL P2P channel settings remove the F1 sendrecv plateau? 5 launches per variant, randomized.

Usage: python3 exp/M2_p2p_channels/run.py
       DRY_RUN=1 python3 exp/M2_p2p_channels/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from ncclrun import Job, run_all  # noqa: E402

SEED = 20261103
REPS = 5
VARIANTS = {  # see README.md for the channel counts each one should produce (from NCCL 2.32.3 source)
    "default": {},
    "minp2p64": {"NCCL_MIN_P2P_NCHANNELS": "64"},
    "chunk128k": {"NCCL_P2P_NVL_CHUNKSIZE": "131072"},
    "chunk64k": {"NCCL_P2P_NVL_CHUNKSIZE": "65536"},
    "minp2p64_chunk128k": {"NCCL_MIN_P2P_NCHANNELS": "64", "NCCL_P2P_NVL_CHUNKSIZE": "131072"},
}

jobs = []
for v, env in VARIANTS.items():
    for rep in range(REPS):
        jobs.append(Job("sendrecv", 8, "A", "1M", "1G", rep, v, env))
        jobs.append(Job("alltoall", 8, "A", "1M", "1G", rep, v, env))
for v in ("default", "chunk128k", "chunk64k"):  # 2-GPU reference: does a smaller chunk hurt the baseline?
    for rep in range(REPS):
        jobs.append(Job("sendrecv", 2, "A", "1M", "1G", rep, v, VARIANTS[v]))
run_all(jobs, "M2_p2p_channels", SEED)
