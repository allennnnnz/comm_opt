"""M16 interleaving test: calls with different block counts and token counts, back to back, no state reset in between.

Each call: fresh inputs (x in symmetric memory, residual replicated), run the prototype, compare the normalized output
(all rows) and the residual (this rank's rows) with an fp32 reference (<= 2^-5 counts as correct).
  eager mode: the calls are issued directly
  graph mode: one CUDA graph per (tokens, blocks) combination (one call each), replayed in a random order
The same script runs the old (M15, per-block epoch) kernel to show the bug; run it under an outer timeout.

Usage (8 ranks, .venv-vllm):
  torchrun --nproc-per-node 8 exp/M16_epoch_fix/test_interleave.py --src exp/M16_epoch_fix/fused_ar_norm.cu \
      --name m16_fused_ar_norm --ext-dir DIR --mode eager --out F.json
"""
import argparse
import json
import math
import os
import random

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.utils.cpp_extension import load

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--name", required=True)
p.add_argument("--ext-dir", required=True)
p.add_argument("--mode", choices=["eager", "graph"], default="eager")
p.add_argument("--calls", type=int, default=60)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--hidden", type=int, default=5120)
p.add_argument("--out", required=True)
a = p.parse_args()

TOKENS = [1, 3, 8, 13, 40, 64, 128]
CTAS = [16, 108, 32, 64]
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
dist.init_process_group("nccl", device_id=dev)
cpu_group = dist.new_group(backend="gloo")
H, DT, EPS = a.hidden, torch.float16, 1e-6
ext = load(name=a.name, sources=[a.src], extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"],
           build_directory=a.ext_dir, verbose=False)
gamma = (1 + 0.1 * torch.randn(H, generator=torch.Generator().manual_seed(7))).to(DT).to(dev)
GPR = math.ceil(math.ceil(H / 4 / 15) / 4)
MAXT = max(TOKENS)
x_buf = symm_mem.empty(MAXT * H, dtype=DT, device=dev)
land = symm_mem.empty(2 * world * math.ceil(MAXT / world) * GPR * 64, dtype=torch.int64, device=dev)
flags = symm_mem.empty(128 * world, dtype=torch.int32, device=dev)
land.zero_()
flags.zero_()
x_ptrs = list(symm_mem.rendezvous(x_buf, group=dist.group.WORLD).buffer_ptrs)
land_ptrs = list(symm_mem.rendezvous(land, group=dist.group.WORLD).buffer_ptrs)
flag_ptrs = list(symm_mem.rendezvous(flags, group=dist.group.WORLD).buffer_ptrs)
ctr = torch.zeros(128, dtype=torch.int64, device=dev)  # M16: [0]=epoch, [1]=done count; M15: one per block
torch.cuda.synchronize()
dist.barrier(group=cpu_group)

rng = random.Random(a.seed)  # same sequence on every rank
seq = [(rng.choice(TOKENS), rng.choice(CTAS)) for _ in range(a.calls)]


def call(nt, ctas, res, out):
    ext.fused_ar_norm(x_ptrs, land_ptrs, flag_ptrs, ctr, res, gamma, out, rank, nt, H, EPS, ctas,
                      math.ceil(nt / world), GPR)


def inputs(i, nt):
    g = torch.Generator(device=dev).manual_seed(1000 * i + rank)
    x = torch.randn(nt, H, generator=g, device=dev).to(DT)
    res0 = torch.randn(nt, H, generator=torch.Generator(device=dev).manual_seed(77 + i), device=dev).to(DT)
    total = x.float()
    dist.all_reduce(total)  # reference only (NCCL)
    ref_res = res0.float() + total
    ref_norm = ref_res * torch.rsqrt(ref_res.pow(2).mean(-1, keepdim=True) + EPS) * gamma.float()
    return x, res0, ref_res, ref_norm


def error(nt, out, res, ref_res, ref_norm):
    e = float((out.float() - ref_norm).abs().max() / ref_norm.abs().max())
    rp = math.ceil(nt / world)
    rows = slice(min(nt, rank * rp), min(nt, (rank + 1) * rp))
    if rows.stop > rows.start:
        e = max(e, float((res[rows].float() - ref_res[rows]).abs().max() / ref_res[rows].abs().max()))
    return e


graphs = {}
if a.mode == "graph":
    for nt, ctas in sorted(set(seq)):
        res, out = torch.empty(nt, H, dtype=DT, device=dev), torch.empty(nt, H, dtype=DT, device=dev)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            call(nt, ctas, res, out)
        graphs[(nt, ctas)] = (gr, res, out)
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)

results = []
for i, (nt, ctas) in enumerate(seq):
    x, res0, ref_res, ref_norm = inputs(i, nt)
    x_buf[: nt * H].copy_(x.view(-1))
    if a.mode == "eager":
        res, out = res0.clone(), torch.empty(nt, H, dtype=DT, device=dev)
        call(nt, ctas, res, out)
    else:
        gr, res, out = graphs[(nt, ctas)]
        res.copy_(res0)
        gr.replay()
    torch.cuda.synchronize()
    e = error(nt, out, res, ref_res, ref_norm)
    worst = [None] * world
    dist.all_gather_object(worst, e, group=cpu_group)
    results.append(dict(call=i, tokens=nt, ctas=ctas, max_rel_err=max(worst), correct=max(worst) <= 2 ** -5))
    if rank == 0:
        print(f"call {i:3d} tokens {nt:4d} ctas {ctas:4d} err {max(worst):.2e} {'ok' if max(worst) <= 2 ** -5 else 'WRONG'}",
              flush=True)
if rank == 0:
    json.dump(dict(args=vars(a), seq=seq, results=results, all_correct=all(r["correct"] for r in results)),
              open(a.out, "w"), indent=1)
    print(f"done: {sum(r['correct'] for r in results)}/{len(results)} correct", flush=True)
dist.destroy_process_group()
