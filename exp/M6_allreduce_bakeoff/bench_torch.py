"""M6 torch-side all-reduce implementations, timed identically inside CUDA graphs.

Implementations: torch.distributed NCCL (bundled NCCL 2.30.7), vLLM custom all-reduce, PyTorch
SymmetricMemory one-shot and two-shot. For each (impl, size): eager warmup, capture K back-to-back
all-reduces in one CUDA graph (decode issues many all-reduces back to back), replay R times, per-op
time = elapsed / (R*K) on every rank. Correctness is checked once per size against NCCL.

Usage (8 ranks, inside .venv-vllm):
  torchrun --nproc-per-node 8 exp/M6_allreduce_bakeoff/bench_torch.py --out FILE.json --seed 0
"""
import argparse
import contextlib
import json
import os
import random

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--min-bytes", type=int, default=1 << 10)
p.add_argument("--max-bytes", type=int, default=8 << 20)
p.add_argument("--graph-ops", type=int, default=20, help="all-reduces captured per graph")
p.add_argument("--replays", type=int, default=20, help="timed graph replays")
a = p.parse_args()

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
dist.init_process_group("nccl", device_id=dev)
cpu_group = dist.new_group(backend="gloo")
group_name = dist.group.WORLD.group_name
DT = torch.float16  # fp16, not bf16: nccl-tests disables bf16 under the MSCCL++ shim (it reports version 0.9.0)
sizes = []
s = a.min_bytes
while s <= a.max_bytes:
    sizes.append(s)
    s *= 2

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce  # noqa: E402

car = CustomAllreduce(cpu_group, dev, max_size=a.max_bytes * 2)
symm_buf = symm_mem.empty(a.max_bytes // 2, dtype=DT, device=dev)
symm_mem.rendezvous(symm_buf, group=dist.group.WORLD)


def make(impl, nbytes):
    """Return (fn, capture_ctx) for one implementation at one size; fn returns the result tensor."""
    n = nbytes // 2
    if impl == "torch_nccl":
        x = torch.randn(n, dtype=DT, device=dev)
        return (lambda: (dist.all_reduce(x), x)[1]), contextlib.nullcontext
    if impl == "vllm_car":
        x = torch.randn(n, dtype=DT, device=dev)
        return (lambda: car.custom_all_reduce(x)), car.capture
    x = symm_buf[:n]
    x.copy_(torch.randn(n, dtype=DT, device=dev))
    if impl == "symm_one_shot":
        return (lambda: torch.ops.symm_mem.one_shot_all_reduce(x, "sum", group_name)), contextlib.nullcontext
    if impl == "symm_two_shot":
        return (lambda: (torch.ops.symm_mem.two_shot_all_reduce_(x, "sum", group_name), x)[1]), contextlib.nullcontext
    raise ValueError(impl)


def check(impl, nbytes) -> float:
    """One eager call on known bf16 data; returns max |out - exact| / max |exact|.

    The exact sum is an fp32 all-reduce of the same bf16 inputs (bf16 NCCL is not a good reference: it
    rounds at every ring hop). two_shot is in-place, so it runs on a fresh copy in the symmetric buffer.
    """
    n = nbytes // 2
    g = torch.Generator(device=dev).manual_seed(rank + 1)
    src = torch.randn(n, dtype=torch.float32, device=dev, generator=g).to(DT)
    exact = src.float()
    dist.all_reduce(exact)
    if impl == "torch_nccl":
        out = src.clone()
        dist.all_reduce(out)
    elif impl == "vllm_car":
        out = car.custom_all_reduce(src.clone())
    else:
        x = symm_buf[:n]
        x.copy_(src)
        torch.cuda.synchronize()  # peers read x directly: it must be written before they start
        dist.barrier(group=cpu_group)
        out = (torch.ops.symm_mem.one_shot_all_reduce(x, "sum", group_name) if impl == "symm_one_shot"
               else (torch.ops.symm_mem.two_shot_all_reduce_(x, "sum", group_name), x)[1].clone())
    torch.cuda.synchronize()
    return float((out.float() - exact).abs().max() / exact.abs().max())


def bench(impl, nbytes) -> float:
    fn, ctx = make(impl, nbytes)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    graph = torch.cuda.CUDAGraph()
    with ctx():
        with torch.cuda.graph(graph):
            for _ in range(a.graph_ops):
                fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(a.replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000 / (a.replays * a.graph_ops)
    del graph
    return us


impls = ["torch_nccl", "vllm_car", "symm_one_shot", "symm_two_shot"]
random.Random(a.seed).shuffle(impls)
rows = []
for impl in impls:
    for nbytes in sizes:
        if impl == "vllm_car" and not car.should_custom_ar(torch.empty(nbytes // 2, dtype=DT, device=dev)):
            continue
        try:
            rel_err = check(impl, nbytes)
            us = bench(impl, nbytes)
            err = ""
        except Exception as e:  # record and move on: an implementation may not support a size or this GPU
            rel_err, us, err = float("nan"), float("nan"), f"{type(e).__name__}: {str(e)[:200]}"
        gathered = [None] * world
        dist.all_gather_object(gathered, (us, rel_err, err), group=cpu_group)
        if rank == 0:
            worst = max(g[1] for g in gathered)
            rows.append(dict(impl=impl, bytes=nbytes, per_rank_us=[g[0] for g in gathered], max_rel_err=worst,
                             correct=bool(worst <= 2 ** -5), error=next((g[2] for g in gathered if g[2]), "")))
            print(f"{impl:14s} {nbytes:>9d} B  max {max(g[0] for g in gathered):8.2f} us  rel_err {worst:.2e} "
                  f"{rows[-1]['error'][:60]}", flush=True)
if rank == 0:
    json.dump(dict(args=vars(a), torch=torch.__version__, nccl=".".join(map(str, torch.cuda.nccl.version())),
                   order=impls, rows=rows), open(a.out, "w"), indent=1)
dist.destroy_process_group()
