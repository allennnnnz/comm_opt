#!/usr/bin/env python
"""E1 block B: operator-level two-way slowdown of (all-reduce, compute) running at the same time. torchrun, one mechanism per process.

Grid (exp card section B): T in {1024,2048,4096,8192} tokens x 5 compute ops (qkv, o_proj, gate_up, down, attn), per-GPU TP8 shapes of Qwen2.5-72B.
all-reduce size = T x 8192 x 2 B.  Mechanism = NCCL environment set by the shell (NCCL_ALGO / NCCL_PROTO / NCCL_MAX_CTAS) or --mech symm
(torch symm_mem multimem all-reduce, the path vLLM 0.32.0 uses for <= 64 MB on Hopper TP8; only cells with T <= 4096).

Per cell, 3 measurements per iteration (10 warm-up iterations discarded, 50 timed): comm alone, compute alone, both at once
(two streams, both start after a shared start event; every time is a CUDA-event interval from that start event).
IMPLEMENTATION CHOICE (flagged to Opus in R09): one timed "iteration" runs R = 8192 / T chunk pairs back to back (R x T = 8192 tokens), i.e. R
all-reduces on the comm stream against R compute ops on the compute stream, because a single 16 MB all-reduce / small GEMM is only tens of us
and event timing + launch skew would dominate.  All times in the jsonl are for the whole iteration (R pairs); divide by R for per-pair time.
Each of the 5 numbers per iteration is max-reduced over ranks (R7) through a gloo group (NCCL_ALGO=NVLS has no float64 all-reduce).
Compute data: random bf16 from torch.manual_seed(SEED + rank).  Attention: causal SDPA, 8 Q heads, 1 KV head, head_dim 128, enable_gqa=True
(falls back to KV expanded to 8 heads if the installed SDPA refuses GQA; the mode used is written in every row).
"""
import argparse, json, os, sys, time
import torch
import torch.distributed as dist
import torch.nn.functional as F

TS = [1024, 2048, 4096, 8192]
COMPS = ["qkv", "o_proj", "gate_up", "down", "attn"]
KN = {"qkv": (8192, 1280), "o_proj": (1024, 8192), "gate_up": (8192, 7392), "down": (3696, 8192)}  # (K, N) per GPU
HID = 8192
SEED = 20261125
SYMM_MAX_BYTES = 64 * 1024 * 1024  # vLLM SYMM_MEM_ALL_REDUCE_MAX_SIZES["9.0"][8]
CTRL = None


def maxreduce(vals):
    t = torch.tensor(vals, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=CTRL)
    return t.tolist()


class Symm:
    """Same call sequence as vllm SymmMemCommunicator.all_reduce (copy in, multimem_all_reduce_, copy out)."""

    def __init__(self, dev, ws):
        import torch.distributed._symmetric_memory as sm
        self.sm = sm
        self.buf = sm.empty(SYMM_MAX_BYTES // 2, device=dev, dtype=torch.bfloat16)
        self.group = dist.group.WORLD
        handle = sm.rendezvous(self.buf, self.group.group_name)
        self.multicast = handle.multicast_ptr != 0
        want_multi = ws in (4, 6, 8) or os.environ.get("E1_SYMM_FORCE_MULTIMEM") == "1"  # vLLM _WORLD_SIZES_MULTIMEM["9.0"]
        self.algo = "multimem" if (want_multi and self.multicast) else "two_shot"
        self.tried_multi = want_multi

    def __call__(self, x):
        n = x.numel()
        v = self.buf[:n]
        v.copy_(x.view(-1))
        if self.algo == "multimem":
            torch.ops.symm_mem.multimem_all_reduce_(v, "sum", self.group.group_name)
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(v, "sum", self.group.group_name)
        x.view(-1).copy_(v)


def make_comm(T, mech, symm, dev):
    buf = torch.zeros(T * HID, dtype=torch.bfloat16, device=dev)
    if mech == "symm":
        if buf.numel() * 2 > SYMM_MAX_BYTES:
            return None
        return lambda: symm(buf)   # runs on the current stream, no work handle
    return lambda: dist.all_reduce(buf, async_op=True)


def make_compute(T, comp, dev, gqa_state):
    if comp in KN:
        K, N = KN[comp]
        x = (torch.randn(T, K, device=dev) * 0.5).to(torch.bfloat16)
        w = (torch.randn(K, N, device=dev) * K ** -0.5).to(torch.bfloat16)
        out = torch.empty(T, N, device=dev, dtype=torch.bfloat16)
        return lambda: torch.mm(x, w, out=out)
    q = torch.randn(1, 8, T, 128, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, 1, T, 128, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, 1, T, 128, device=dev, dtype=torch.bfloat16)
    ke, ve = k.expand(1, 8, T, 128).contiguous(), v.expand(1, 8, T, 128).contiguous()

    def f():
        if gqa_state["mode"] == "gqa":
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return F.scaled_dot_product_attention(q, ke, ve, is_causal=True)
    try:
        f()
    except Exception as e:  # SDPA backend without GQA support
        gqa_state["mode"] = "expanded_kv"
        gqa_state["why"] = repr(e)[:200]
        f()
    return f


class Runner:
    def __init__(self, dev):
        self.dev = dev
        self.ms = torch.cuda.Stream(device=dev)
        self.cs = torch.cuda.Stream(device=dev)
        self.sleep_cycles = int(torch.cuda.get_device_properties(dev).clock_rate * 2.0)  # ~2 ms

    def start(self):
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda._sleep(self.sleep_cycles)
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def enq_comp(self, fn, n, s):
        self.ms.wait_event(s)
        with torch.cuda.stream(self.ms):
            for _ in range(n):
                fn()
            e = torch.cuda.Event(enable_timing=True)
            e.record()
        return e

    def enq_comm(self, fn, n, s):
        self.cs.wait_event(s)
        with torch.cuda.stream(self.cs):
            ws = [fn() for _ in range(n)]
            for w in ws:
                if w is not None:
                    w.wait()
            e = torch.cuda.Event(enable_timing=True)
            e.record()
        return e

    def comp_alone(self, fn, n):
        s = self.start(); e = self.enq_comp(fn, n, s); torch.cuda.synchronize()
        return s.elapsed_time(e)

    def comm_alone(self, fn, n):
        s = self.start(); e = self.enq_comm(fn, n, s); torch.cuda.synchronize()
        return s.elapsed_time(e)

    def conc(self, comp_fn, comm_fn, n):
        s = self.start()
        e1 = self.enq_comp(comp_fn, n, s)
        e2 = self.enq_comm(comm_fn, n, s)
        torch.cuda.synchronize()
        return s.elapsed_time(e1), s.elapsed_time(e2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", required=True)
    ap.add_argument("--launch", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--ts", default=",".join(map(str, TS)))
    a = ap.parse_args()

    rank = int(os.environ["RANK"]); ws = int(os.environ["WORLD_SIZE"]); lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    dist.init_process_group("nccl", device_id=dev)
    global CTRL
    CTRL = dist.new_group(backend="gloo")
    torch.manual_seed(SEED + rank)
    r = Runner(dev)
    symm = Symm(dev, ws) if a.mech == "symm" else None
    gqa = {"mode": "gqa"}
    envs = {k: os.environ[k] for k in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MAX_CTAS", "NCCL_MIN_CTAS", "NCCL_NVLS_ENABLE") if k in os.environ}
    # touch NCCL once so communicator creation is outside every timed cell
    w = dist.all_reduce(torch.zeros(1 << 20, dtype=torch.bfloat16, device=dev), async_op=True); w.wait(); torch.cuda.synchronize()

    fo = open(a.out, "a") if rank == 0 else None
    for T in [int(x) for x in a.ts.split(",")]:
        R = HID // T
        comm = make_comm(T, a.mech, symm, dev)
        if comm is None:
            if rank == 0:
                print(f"skip T={T}: {T * HID * 2 >> 20} MB > symm_mem limit", flush=True)
            continue
        for comp in COMPS:
            cf = make_compute(T, comp, dev, gqa)
            rows = []
            for it in range(a.warm + a.iters):
                ta = r.comm_alone(comm, R)
                tb = r.comp_alone(cf, R)
                tcp, tcm = r.conc(cf, comm, R)
                if it >= a.warm:
                    rows.append([ta, tb, tcp, tcm, max(tcp, tcm)])
            # one gloo max-reduce per cell: (n_iter x 5), each element is the max over ranks
            flat = maxreduce([x for row in rows for x in row])
            n = len(rows)
            m = [flat[i * 5:(i + 1) * 5] for i in range(n)]
            if rank == 0:
                fo.write(json.dumps({
                    "mech": a.mech, "launch": a.launch, "T": T, "comp": comp, "R": R, "world_size": ws,
                    "warm": a.warm, "iters": a.iters, "seed": SEED, "attn_mode": gqa["mode"], "envs": envs,
                    "symm_algo": symm.algo if symm else None,
                    "comm_alone_ms": [x[0] for x in m], "comp_alone_ms": [x[1] for x in m],
                    "comp_conc_ms": [x[2] for x in m], "comm_conc_ms": [x[3] for x in m],
                    "makespan_ms": [x[4] for x in m]}) + "\n")
                fo.flush()
                print(f"cell mech={a.mech} T={T} comp={comp} R={R} comm_alone={sorted(x[0] for x in m)[n // 2]:.3f} "
                      f"comp_alone={sorted(x[1] for x in m)[n // 2]:.3f} makespan={sorted(x[4] for x in m)[n // 2]:.3f}", flush=True)
            del cf
        del comm
    if fo:
        fo.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
