"""M18 benchmark: several prototype versions and vLLM interleaved in one process, timed as M14-M17
(CUDA graph of K ops, R replays, per-op = elapsed / (R*K), max over ranks), plus the SM clock seen before each config.

Implementations
  U_vllm          vLLM custom all-reduce, then vLLM fused_add_rms_norm
  P_<ver>_c<N>    prototype version <ver> with N blocks
  P_<ver>_tab     prototype version <ver>, blocks from --table (JSON {tokens: blocks}, calibrated in another run)

Warm-up: --warmup-ms of all-GPU busy work (matmul + NCCL all-reduce) before the first config, and --warm-replays graph
replays before timing each config (M14-M17 used 0 and 3).

Usage (8 ranks, .venv-vllm):
  torchrun --nproc-per-node 8 exp/M18_incremental/bench_inc.py --versions v0=exp/M18_incremental/kernels/v0.cu \
      --impls U_vllm,P_v0_c64 --ext-root DIR --out F.json
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--hidden", type=int, default=5120)
p.add_argument("--tokens", default="1,2,4,8,16,32,64,128")
p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
p.add_argument("--graph-ops", type=int, default=20)
p.add_argument("--replays", type=int, default=20)
p.add_argument("--warmup-ms", type=int, default=0)
p.add_argument("--warm-replays", type=int, default=3)
p.add_argument("--versions", required=True, help="comma-separated name=path.cu")
p.add_argument("--impls", required=True)
p.add_argument("--table", default="", help="JSON file {tokens: blocks} for P_<ver>_tab")
p.add_argument("--ext-root", required=True)
p.add_argument("--eps", type=float, default=1e-6)
a = p.parse_args()

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
dist.init_process_group("nccl", device_id=dev)
cpu_group = dist.new_group(backend="gloo")
DT = getattr(torch, a.dtype)
H = a.hidden
tokens = [int(t) for t in a.tokens.split(",")]
max_tokens = max(tokens)
impls = a.impls.split(",")
table = {int(k): int(v) for k, v in json.load(open(a.table)).items()} if a.table else {}

import pynvml  # noqa: E402
import vllm._custom_ops  # noqa: E402,F401
from torch.utils.cpp_extension import load  # noqa: E402
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce  # noqa: E402

EXT = {}
for item in a.versions.split(","):
    ver, src = item.split("=")
    EXT[ver] = load(name=f"m18_{ver}", sources=[src], extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"],
                    build_directory=str(Path(a.ext_root) / ver), verbose=False)

pynvml.nvmlInit()
NVH = pynvml.nvmlDeviceGetHandleByIndex(rank)  # CUDA_VISIBLE_DEVICES unset: CUDA index == NVML index on this machine


def gpu_clocks():
    return (pynvml.nvmlDeviceGetClockInfo(NVH, pynvml.NVML_CLOCK_SM),
            pynvml.nvmlDeviceGetClockInfo(NVH, pynvml.NVML_CLOCK_MEM), pynvml.nvmlDeviceGetPerformanceState(NVH))


car = CustomAllreduce(cpu_group, dev, max_size=16 << 20)
gamma = (1 + 0.1 * torch.randn(H, generator=torch.Generator().manual_seed(7))).to(DT).to(dev)
GPR = math.ceil(math.ceil(H / 4 / 15) / 4)
LAND_ROWS = max(math.ceil(max_tokens / world), min(max_tokens, 32))  # two-shot rows per rank, or one-shot rows (<= 32)
x_buf = symm_mem.empty(max_tokens * H, dtype=DT, device=dev)
land = symm_mem.empty(2 * world * LAND_ROWS * GPR * 64, dtype=torch.int64, device=dev)
flags = symm_mem.empty(128 * world, dtype=torch.int32, device=dev)
x_ptrs = list(symm_mem.rendezvous(x_buf, group=dist.group.WORLD).buffer_ptrs)
land_ptrs = list(symm_mem.rendezvous(land, group=dist.group.WORLD).buffer_ptrs)
flag_ptrs = list(symm_mem.rendezvous(flags, group=dist.group.WORLD).buffer_ptrs)
ctr = torch.zeros(128, dtype=torch.int64, device=dev)


def sync_all():
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)


def reset_proto():
    sync_all()
    land.zero_()
    flags.zero_()
    ctr.zero_()
    sync_all()


def parse(impl):
    """'P_v0_c64' -> ('v0', 64); 'P_v0_tab' -> ('v0', None)"""
    _, ver, sel = impl.split("_", 2)
    return ver, None if sel == "tab" else int(sel[1:])


def ours(ver, nt, res, out, ctas):
    EXT[ver].fused_ar_norm(x_ptrs, land_ptrs, flag_ptrs, ctr, res, gamma, out, rank, nt, H, a.eps, ctas,
                           math.ceil(nt / world), GPR)
    return out


def vllm_unfused(x, res):
    out = car.custom_all_reduce(x)
    torch.ops._C.fused_add_rms_norm(out, res, gamma, a.eps)
    return out


def blocks_for(impl, nt):
    ver, c = parse(impl)
    return ver, (table[nt] if c is None else c)


def make(impl, nt):
    x = 0.01 * torch.randn(nt, H, dtype=DT, device=dev)
    res = torch.randn(nt, H, dtype=DT, device=dev)
    if impl == "U_vllm":
        return lambda: vllm_unfused(x, res)
    out = torch.empty_like(x)
    x_buf[: nt * H].copy_(x.view(-1))
    reset_proto()
    ver, ctas = blocks_for(impl, nt)
    return lambda: ours(ver, nt, res, out, ctas)


def check(impl, nt) -> float:
    g = torch.Generator(device=dev).manual_seed(rank + 1)
    x = torch.randn(nt, H, dtype=torch.float32, device=dev, generator=g).to(DT)
    res0 = torch.randn(nt, H, dtype=torch.float32, device=dev,
                       generator=torch.Generator(device=dev).manual_seed(99)).to(DT)
    total = x.float()
    dist.all_reduce(total)
    ref_res = res0.float() + total
    ref_norm = ref_res * torch.rsqrt(ref_res.pow(2).mean(-1, keepdim=True) + a.eps) * gamma.float()
    res = res0.clone()
    rows = slice(0, nt)
    if impl.startswith("P_"):
        x_buf[: nt * H].copy_(x.view(-1))
        reset_proto()
        ver, ctas = blocks_for(impl, nt)
        out = ours(ver, nt, res, torch.empty_like(x), ctas)
        rp = math.ceil(nt / world)
        rows = slice(min(nt, rank * rp), min(nt, (rank + 1) * rp))
    else:
        sync_all()
        out = vllm_unfused(x.clone(), res)
    torch.cuda.synchronize()
    e1 = float((out.float() - ref_norm).abs().max() / ref_norm.abs().max())
    e2 = 0.0
    if rows.stop > rows.start:
        e2 = float((res[rows].float() - ref_res[rows]).abs().max() / ref_res[rows].abs().max())
    return max(e1, e2)


def bench(impl, nt):
    fn = make(impl, nt)
    for _ in range(5):
        fn()
    sync_all()
    graph = torch.cuda.CUDAGraph()
    if impl == "U_vllm":
        with car.capture():
            with torch.cuda.graph(graph):
                for _ in range(a.graph_ops):
                    fn()
    else:
        with torch.cuda.graph(graph):
            for _ in range(a.graph_ops):
                fn()
    for _ in range(a.warm_replays):
        graph.replay()
    sync_all()
    clock = gpu_clocks()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(a.replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000 / (a.replays * a.graph_ops)
    del graph
    return us, clock


if a.warmup_ms > 0:  # all GPUs busy until clocks have ramped
    m = torch.randn(4096, 4096, dtype=DT, device=dev)
    big = torch.randn(8 << 20, dtype=DT, device=dev)
    t0 = time.time()
    while (time.time() - t0) * 1000 < a.warmup_ms:
        for _ in range(10):
            m = (m @ m).clamp_(-1, 1)
        dist.all_reduce(big)
        torch.cuda.synchronize()
    sync_all()

random.Random(a.seed).shuffle(impls)
rows_out = []
for idx, impl in enumerate(impls):
    for nt in tokens:
        try:
            rel_err = check(impl, nt)
            us, clock = bench(impl, nt)
            err = ""
        except Exception as e:
            rel_err, us, clock, err = float("nan"), float("nan"), (-1, -1, -1), f"{type(e).__name__}: {str(e)[:200]}"
        gathered = [None] * world
        dist.all_gather_object(gathered, (us, rel_err, clock, err), group=cpu_group)
        if rank == 0:
            worst = max(g[1] for g in gathered)
            ctas = blocks_for(impl, nt)[1] if impl.startswith("P_") else 0
            rows_out.append(dict(impl=impl, ctas=ctas, order=idx, dtype=a.dtype, hidden=H, tokens=nt, bytes=nt * H * 2,
                                 per_rank_us=[g[0] for g in gathered], sm_mhz=[g[2][0] for g in gathered],
                                 mem_mhz=[g[2][1] for g in gathered], pstate=[g[2][2] for g in gathered],
                                 max_rel_err=worst, correct=bool(worst <= 2 ** -5),
                                 error=next((g[3] for g in gathered if g[3]), "")))
            print(f"{impl:12s} t{nt:>4d} max {max(g[0] for g in gathered):8.2f} us  sm min {min(g[2][0] for g in gathered):5d} "
                  f"mem min {min(g[2][1] for g in gathered):5d} P{max(g[2][2] for g in gathered)}  err {worst:.1e} "
                  f"{rows_out[-1]['error'][:50]}", flush=True)
if rank == 0:
    json.dump(dict(args=vars(a), torch=torch.__version__, order=impls, rows=rows_out), open(a.out, "w"), indent=1)
sync_all()
dist.destroy_process_group()
