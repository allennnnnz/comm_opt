"""M9 torch-side: all-reduce + residual add + RMSNorm on 8x A100, fused vs unfused, timed like M6.

Implementations
  U_vllm     vLLM custom all-reduce, then vLLM fused_add_rms_norm (what vLLM 0.33.0 runs on A100)
  U_nccl     torch.distributed NCCL all-reduce, then fused_add_rms_norm
  F_trt_one  TensorRT-LLM fused one-shot Lamport kernel (FlashInfer 0.7.2.post1 sources, rebuilt for sm_80)
  F_trt_two  TensorRT-LLM fused two-shot sync kernel (same build; FlashInfer asserts tokens > world size)

FlashInfer only builds its trtllm_comm module for SM 9x/10x/12x. We build the same three source files for
sm_80 under a different module name and hand that module to flashinfer.comm.trtllm_ar; the installed
package is not modified.

Timing (same as M6): warm up, capture K back-to-back ops in one CUDA graph, replay R times, per-op time =
elapsed / (R*K) on every rank. Correctness: one eager call on known data per (impl, tokens), norm output and
updated residual compared with an fp32 reference; max |out - ref| / max |ref| <= 2^-5 counts as correct.

Usage (8 ranks, .venv-vllm):
  torchrun --nproc-per-node 8 exp/M9_fused_baselines/bench_fused.py --hidden 5120 --tokens 1,8,16 --out F.json
"""
import argparse
import json
import os
import random

import torch
import torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--hidden", type=int, default=5120)
p.add_argument("--tokens", default="1,8,16,32,64,128")
p.add_argument("--graph-ops", type=int, default=20, help="ops captured per graph")
p.add_argument("--replays", type=int, default=20, help="timed graph replays")
p.add_argument("--impls", default="U_vllm,U_nccl,F_trt_one,F_trt_two")
p.add_argument("--eps", type=float, default=1e-6)
a = p.parse_args()

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
dist.init_process_group("nccl", device_id=dev)
cpu_group = dist.new_group(backend="gloo")
DT = torch.float16
H = a.hidden
tokens = [int(t) for t in a.tokens.split(",")]
max_tokens = max(tokens)

import vllm._custom_ops  # noqa: E402,F401  registers torch.ops._C.fused_add_rms_norm
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce  # noqa: E402

import flashinfer.comm.trtllm_ar as fi_ar  # noqa: E402
from flashinfer.jit import env as jit_env  # noqa: E402
from flashinfer.jit.core import common_nvcc_flags, gen_jit_spec  # noqa: E402

SM80_FLAGS = ["-gencode=arch=compute_80,code=sm_80"] + common_nvcc_flags


def gen_trtllm_comm_module_sm80():
    return gen_jit_spec("trtllm_comm_sm80",
                        [jit_env.FLASHINFER_CSRC_DIR / f for f in
                         ("trtllm_allreduce.cu", "trtllm_allreduce_fusion.cu", "trtllm_moe_allreduce_fusion.cu")],
                        extra_cuda_cflags=SM80_FLAGS)


fi_ar.gen_trtllm_comm_module = gen_trtllm_comm_module_sm80  # get_trtllm_comm_module looks this name up at call time

car = CustomAllreduce(cpu_group, dev, max_size=16 << 20)
fi_handles, fi_ws = fi_ar.trtllm_create_ipc_workspace_for_all_reduce_fusion(
    rank, world, max_tokens, H, group=cpu_group)
gamma = (1 + 0.1 * torch.randn(H, generator=torch.Generator().manual_seed(7))).to(DT).to(dev)


def fused(x, res, norm_out, nt, oneshot):
    fi_ar.trtllm_allreduce_fusion(
        allreduce_in=x, world_size=world, world_rank=rank, token_num=nt, hidden_dim=H, workspace_ptrs=fi_ws,
        launch_with_pdl=False, trigger_completion_at_end=True, fp32_acc=True,
        pattern_code=fi_ar.AllReduceFusionPattern.kARResidualRMSNorm, use_oneshot=oneshot, allreduce_out=None,
        residual_in=res, residual_out=res, norm_out=norm_out, quant_out=None, scale_out=None, rms_gamma=gamma,
        rms_eps=a.eps, scale_factor=None, layout_code=None)
    return norm_out


def unfused(impl, x, res):
    if impl == "U_vllm":
        out = car.custom_all_reduce(x)
    else:
        out = x.clone()
        dist.all_reduce(out)
    torch.ops._C.fused_add_rms_norm(out, res, gamma, a.eps)  # out <- norm(res + out) * gamma, res <- res + out
    return out


def make(impl, nt):
    """Return fn; inputs are small so the residual stays far from fp16 overflow over all calls."""
    x = 0.01 * torch.randn(nt, H, dtype=DT, device=dev)
    res = torch.randn(nt, H, dtype=DT, device=dev)
    if impl.startswith("U_"):
        return lambda: unfused(impl, x, res)
    norm_out = torch.empty_like(x)
    return lambda: fused(x, res, norm_out, nt, impl == "F_trt_one")


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
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    if impl.startswith("U_"):
        out = unfused(impl, x.clone(), res)
    else:
        out = fused(x, res, torch.empty_like(x), nt, impl == "F_trt_one")
    torch.cuda.synchronize()
    e1 = float((out.float() - ref_norm).abs().max() / ref_norm.abs().max())
    e2 = float((res.float() - ref_res).abs().max() / ref_res.abs().max())
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


impls = a.impls.split(",")
random.Random(a.seed).shuffle(impls)
rows = []
for impl in impls:
    for nt in tokens:
        nbytes = nt * H * 2
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
            except Exception as e:  # record and move on
                rel_err, us, err = float("nan"), float("nan"), f"{type(e).__name__}: {str(e)[:200]}"
        gathered = [None] * world
        dist.all_gather_object(gathered, (us, rel_err, err), group=cpu_group)
        if rank == 0:
            worst = max(g[1] for g in gathered)
            rows.append(dict(impl=impl, hidden=H, tokens=nt, bytes=nbytes, per_rank_us=[g[0] for g in gathered],
                             max_rel_err=worst, correct=bool(worst <= 2 ** -5),
                             error=next((g[2] for g in gathered if g[2]), "")))
            print(f"{impl:10s} h{H} t{nt:>4d} {nbytes:>9d} B  max {max(g[0] for g in gathered):8.2f} us  "
                  f"rel_err {worst:.2e} {rows[-1]['error'][:60]}", flush=True)
if rank == 0:
    import flashinfer
    import vllm
    json.dump(dict(args=vars(a), torch=torch.__version__, nccl=".".join(map(str, torch.cuda.nccl.version())),
                   vllm=vllm.__version__, flashinfer=flashinfer.__version__, fi_sm80_flags=SM80_FLAGS,
                   order=impls, rows=rows), open(a.out, "w"), indent=1)
fi_ar.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(fi_handles, group=cpu_group)
dist.destroy_process_group()
