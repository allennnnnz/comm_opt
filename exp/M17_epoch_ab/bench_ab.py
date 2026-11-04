"""M17 (three kernel versions in one process; otherwise M16's bench): the fused prototype (pull reduce-scatter + residual + RMSNorm + push all-gather) vs vLLM and TRT-LLM fused,
timed exactly as M9 (CUDA graph of K ops, R replays, per-op = elapsed / (R*K), max over ranks).

Implementations
  U_vllm         vLLM custom all-reduce, then vLLM fused_add_rms_norm (vLLM 0.33.0 on A100)
  F_trt_one/two  TRT-LLM fused one-shot / two-shot (FlashInfer 0.7.2.post1 sources rebuilt for sm_80, as M9)
  P_ours_c<N>    this prototype with N blocks

Correctness (once per impl and size): normalized output (all rows) and residual (rows the impl updates: all rows for
U_/F_, this rank's rows for P_ours) against an fp32 reference; max |out - ref| / max |ref| <= 2^-5 counts as correct.

Usage (8 ranks, .venv-vllm, extension prebuilt by run.py):
  torchrun --nproc-per-node 8 exp/M17_epoch_ab/bench_ab.py --hidden 5120 --tokens 8,16 --out F.json
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--hidden", type=int, default=5120)
p.add_argument("--tokens", default="8,16,32,64,128")
p.add_argument("--graph-ops", type=int, default=20)
p.add_argument("--replays", type=int, default=20)
p.add_argument("--impls", default="U_vllm," + ",".join(f"P_{v}_c{c}" for v in ("m15", "m16", "m16nf") for c in (16, 32, 64, 108)))
p.add_argument("--eps", type=float, default=1e-6)
p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
p.add_argument("--ext-root", required=True, help="directory holding one prebuilt build dir per version")
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

import vllm._custom_ops  # noqa: E402,F401
from torch.utils.cpp_extension import load  # noqa: E402
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = {"m15": ("m15_fused_ar_norm", ROOT / "exp/M15_bf16/fused_ar_norm.cu"),
       "m16": ("m16_fused_ar_norm", ROOT / "exp/M16_epoch_fix/fused_ar_norm.cu"),
       "m16nf": ("m17nf_fused_ar_norm", ROOT / "exp/M17_epoch_ab/fused_ar_norm_nf.cu")}
EXT = {k: load(name=n, sources=[str(s)], extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"],
               build_directory=str(Path(a.ext_root) / k), verbose=False) for k, (n, s) in SRC.items()}

use_trt = any(i.startswith("F_trt") for i in impls)
if use_trt:
    import flashinfer.comm.trtllm_ar as fi_ar
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import common_nvcc_flags, gen_jit_spec

    SM80_FLAGS = ["-gencode=arch=compute_80,code=sm_80"] + common_nvcc_flags
    fi_ar.gen_trtllm_comm_module = lambda: gen_jit_spec(
        "trtllm_comm_sm80", [jit_env.FLASHINFER_CSRC_DIR / f for f in
                             ("trtllm_allreduce.cu", "trtllm_allreduce_fusion.cu", "trtllm_moe_allreduce_fusion.cu")],
        extra_cuda_cflags=SM80_FLAGS)
    fi_handles, fi_ws = fi_ar.trtllm_create_ipc_workspace_for_all_reduce_fusion(rank, world, max_tokens, H,
                                                                                group=cpu_group)

car = CustomAllreduce(cpu_group, dev, max_size=16 << 20)
gamma = (1 + 0.1 * torch.randn(H, generator=torch.Generator().manual_seed(7))).to(DT).to(dev)

# prototype buffers in symmetric memory
GPR = math.ceil(math.ceil(H / 4 / 15) / 4)  # 4-line LL128 groups per row (15 payload u64 per line)
MAX_ROWS_PER = math.ceil(max_tokens / world)
x_buf = symm_mem.empty(max_tokens * H, dtype=DT, device=dev)
land = symm_mem.empty(2 * world * MAX_ROWS_PER * GPR * 64, dtype=torch.int64, device=dev)
flags = symm_mem.empty(128 * world, dtype=torch.int32, device=dev)
x_ptrs = list(symm_mem.rendezvous(x_buf, group=dist.group.WORLD).buffer_ptrs)
land_ptrs = list(symm_mem.rendezvous(land, group=dist.group.WORLD).buffer_ptrs)
flag_ptrs = list(symm_mem.rendezvous(flags, group=dist.group.WORLD).buffer_ptrs)
ctr = torch.zeros(128, dtype=torch.int64, device=dev)


def reset_proto():
    """Zero landing areas, barrier flags and epoch counters on every rank before a new configuration."""
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    land.zero_()
    flags.zero_()
    ctr.zero_()
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)


def ours(nt, res, out, ctas, ver):
    EXT[ver].fused_ar_norm(x_ptrs, land_ptrs, flag_ptrs, ctr, res, gamma, out, rank, nt, H, a.eps, ctas,
                      math.ceil(nt / world), GPR)
    return out


def trt(x, res, norm_out, nt, oneshot):
    fi_ar.trtllm_allreduce_fusion(
        allreduce_in=x, world_size=world, world_rank=rank, token_num=nt, hidden_dim=H, workspace_ptrs=fi_ws,
        launch_with_pdl=False, trigger_completion_at_end=True, fp32_acc=True,
        pattern_code=fi_ar.AllReduceFusionPattern.kARResidualRMSNorm, use_oneshot=oneshot, allreduce_out=None,
        residual_in=res, residual_out=res, norm_out=norm_out, quant_out=None, scale_out=None, rms_gamma=gamma,
        rms_eps=a.eps, scale_factor=None, layout_code=None)
    return norm_out


def vllm_unfused(x, res):
    out = car.custom_all_reduce(x)
    torch.ops._C.fused_add_rms_norm(out, res, gamma, a.eps)
    return out


def make(impl, nt):
    x = 0.01 * torch.randn(nt, H, dtype=DT, device=dev)
    res = torch.randn(nt, H, dtype=DT, device=dev)
    if impl == "U_vllm":
        return lambda: vllm_unfused(x, res)
    out = torch.empty_like(x)
    if impl.startswith("F_trt"):
        return lambda: trt(x, res, out, nt, impl == "F_trt_one")
    x_buf[: nt * H].copy_(x.view(-1))
    reset_proto()
    ver, ctas = impl.split("_")[1], int(impl.split("_c")[1])
    return lambda: ours(nt, res, out, ctas, ver)


def check(impl, nt) -> float:
    g = torch.Generator(device=dev).manual_seed(rank + 1)
    x = torch.randn(nt, H, dtype=torch.float32, device=dev, generator=g).to(DT)
    res0 = torch.randn(nt, H, dtype=torch.float32, device=dev,
                       generator=torch.Generator(device=dev).manual_seed(99)).to(DT)  # replicated across ranks
    total = x.float()
    dist.all_reduce(total)
    ref_res = res0.float() + total
    ref_norm = ref_res * torch.rsqrt(ref_res.pow(2).mean(-1, keepdim=True) + a.eps) * gamma.float()
    res = res0.clone()
    rows = slice(0, nt)
    if impl.startswith("P_"):
        x_buf[: nt * H].copy_(x.view(-1))
        reset_proto()
        out = ours(nt, res, torch.empty_like(x), int(impl.split("_c")[1]), impl.split("_")[1])
        rp = math.ceil(nt / world)
        rows = slice(min(nt, rank * rp), min(nt, (rank + 1) * rp))
    else:
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        out = vllm_unfused(x.clone(), res) if impl == "U_vllm" else trt(x, res, torch.empty_like(x), nt,
                                                                         impl == "F_trt_one")
    torch.cuda.synchronize()
    e1 = float((out.float() - ref_norm).abs().max() / ref_norm.abs().max())
    e2 = 0.0
    if rows.stop > rows.start:
        e2 = float((res[rows].float() - ref_res[rows]).abs().max() / ref_res[rows].abs().max())
    return max(e1, e2)


def bench(impl, nt) -> float:
    fn = make(impl, nt)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
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


random.Random(a.seed).shuffle(impls)
rows_out = []
for impl in impls:
    for nt in tokens:
        skip = ""
        if impl == "U_vllm" and not car.should_custom_ar(torch.empty(nt * H, dtype=DT, device=dev)):
            skip = "vllm custom AR declines this size"
        if impl == "F_trt_two" and nt <= world:
            skip = "two-shot needs tokens > world size (FlashInfer assertion)"
        if skip:
            us, rel_err, err = float("nan"), float("nan"), "SKIP: " + skip
        else:
            try:
                rel_err = check(impl, nt)
                us = bench(impl, nt)
                err = ""
            except Exception as e:
                rel_err, us, err = float("nan"), float("nan"), f"{type(e).__name__}: {str(e)[:200]}"
        gathered = [None] * world
        dist.all_gather_object(gathered, (us, rel_err, err), group=cpu_group)
        if rank == 0:
            worst = max(g[1] for g in gathered)
            rows_out.append(dict(impl=impl, dtype=a.dtype, hidden=H, tokens=nt, bytes=nt * H * 2,
                                 per_rank_us=[g[0] for g in gathered], max_rel_err=worst,
                                 correct=bool(worst <= 2 ** -5), error=next((g[2] for g in gathered if g[2]), "")))
            print(f"{impl:12s} h{H} t{nt:>4d} max {max(g[0] for g in gathered):8.2f} us  rel_err {worst:.2e} "
                  f"{rows_out[-1]['error'][:60]}", flush=True)
if rank == 0:
    import vllm
    json.dump(dict(args=vars(a), torch=torch.__version__, vllm=vllm.__version__, order=impls, rows=rows_out),
              open(a.out, "w"), indent=1)
torch.cuda.synchronize()
dist.barrier(group=cpu_group)
if use_trt:
    fi_ar.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(fi_handles, group=cpu_group)
dist.destroy_process_group()
