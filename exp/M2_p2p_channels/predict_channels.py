"""Predict NCCL 2.32.3 per-op P2P channel counts, ported from the source (single node).

src/graph/paths.cc  ncclTopoComputeP2pChannels: p2pnChannels = pow2Up(max(min(collChannels, MAX_P2P), MIN_P2P))
src/enqueue/enqueue.cc scheduleP2pTasksToPlan:  nChannelsMin halves while nChannelsMin * nRanks > p2pnChannels
src/enqueue/enqueue.cc addP2pToPlan:            start at min(nChannelsMin, bytes / (chunk/8)); double while the
                                                 per-channel part exceeds chunk*32 and nChannels <= max/2
Usage: python3 exp/M2_p2p_channels/predict_channels.py   (checks itself against M0 first)
"""
from math import ceil

COLL_CHANNELS = 24   # "24 coll channels" in M0 NCCL INIT logs
PER_PEER_TOPO = 32   # "32 p2p channels per peer" in M0 NCCL INIT logs
MB = 1 << 20


def pow2up(x):
    p = 1
    while p < x:
        p *= 2
    return p


def channels(nbytes, nranks, min_p2p=1, max_p2p=64, chunk=512 << 10):
    p2p_nch = pow2up(max(min(COLL_CHANNELS, max_p2p), min_p2p))
    per_peer = min(p2p_nch, PER_PEER_TOPO)
    nmin = per_peer
    while nmin * nranks > p2p_nch and nmin > 1:
        nmin //= 2
    min_part, max_part = chunk // 8, chunk * 32
    n = min(nmin, ceil(nbytes / min_part))
    part = max(min_part, ceil(nbytes / n))
    while part > max_part and n <= per_peer // 2:
        n *= 2
        part = ceil(nbytes / n)
    return n


# M0 observations (fig_tuner.png): size -> channels
M0 = {("sendrecv", 8): {4 * MB: 4, 64 * MB: 4, 128 * MB: 8, 256 * MB: 16, 512 * MB: 32},
      ("sendrecv", 2): {512 << 10: 8, 1 * MB: 16, 256 * MB: 16, 512 * MB: 32},
      ("alltoall", 8): {1 * MB: 2, 2 * MB: 4, 512 * MB: 4, 1024 * MB: 8, 2048 * MB: 16, 4096 * MB: 32}}
VARIANTS = {"default": {}, "minp2p64": dict(min_p2p=64), "chunk128k": dict(chunk=128 << 10),
            "chunk64k": dict(chunk=64 << 10), "minp2p64_chunk128k": dict(min_p2p=64, chunk=128 << 10)}


def per_op_bytes(coll, size, nranks):
    return size // nranks if coll == "alltoall" else size  # alltoall sends size/nranks to each peer


if __name__ == "__main__":
    bad = [(k, s, obs, channels(per_op_bytes(k[0], s, k[1]), k[1]))
           for k, pts in M0.items() for s, obs in pts.items()
           if channels(per_op_bytes(k[0], s, k[1]), k[1]) != obs]
    print("model vs M0:", "all match" if not bad else f"MISMATCH {bad}")
    sizes = [MB << i for i in range(11)]  # 1 MiB .. 1 GiB
    for coll, ng in (("sendrecv", 8), ("alltoall", 8), ("sendrecv", 2)):
        print(f"\n{coll} x{ng}  " + " ".join(f"{s // MB:>5}M" for s in sizes))
        for v, kw in VARIANTS.items():
            if ng == 2 and v not in ("default", "chunk128k", "chunk64k"):
                continue
            print(f"  {v:<20}" + " ".join(f"{channels(per_op_bytes(coll, s, ng), ng, **kw):>6}" for s in sizes))
