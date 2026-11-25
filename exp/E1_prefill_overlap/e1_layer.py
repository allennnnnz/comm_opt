#!/usr/bin/env python
"""E1 block C: 8 stacked Qwen2.5-72B transformer layers (per-GPU TP8 shapes), prefill of L tokens, simple chunked overlap prototype.

torchrun, one communication mechanism per process (NCCL environment from the shell, or --mech symm).

Layer: RMSNorm -> QKV GEMM (8192->1280) -> causal attention (8 Q heads, 1 KV head, d=128) -> o_proj (1024->8192) -> all-reduce -> (+residual)
       -> RMSNorm -> gate_up GEMM (8192->7392) -> SiLU*mul -> down GEMM (3696->8192) -> all-reduce -> (+residual).   No RoPE, no KV cache.
Versions
  C0  sequential, all-reduce on the compute stream (nothing overlaps)            k=1, mode "seq"
  C1  all-reduce removed (the upper bound)                                       k=1, mode "noar"
  C2  token dimension cut into k in {2,4} chunks; chunk i's all-reduce runs on a separate stream while chunk i+1 computes; across layers
      chunk i of layer l+1 only waits for chunk i's second all-reduce of layer l.   mode "overlap"
Chunk j's attention sees K/V of chunks 0..j (queries are the chunk's rows; mask = torch.nn.attention.bias.causal_lower_right(Lc, end)).
ALL versions (also C0/C1) use that same SDPA call so the attention kernel is the same; K/V are expanded from 1 to 8 heads (.contiguous()) per
chunk call (same in every version).
Correctness: every C2 setting is compared with C0 (same weights, same input, same process): relative Frobenius error
||out - ref||_F / ||ref||_F <= 1e-2 (bf16) on every rank (max over ranks).  Failing settings are written with valid=false ("INVALID").
Weights: randn * K^-0.5 with torch.Generator seed SEED*1000 + rank*100 + layer (different per rank = a different TP shard); input randn*0.5.
Timing per setting: 5 warm-up + 20 timed iterations; each iteration is one CUDA-event interval on the compute stream, max over ranks (gloo).
"""
import argparse, json, os, time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

SEED = 20261125
HID, NQ, D = 8192, 8, 128
QKV_N, O_K, GU_N, DN_K = 1280, 1024, 7392, 3696
NL = 8
SYMM_MAX_BYTES = 64 * 1024 * 1024
CTRL = None


def maxreduce(vals):
    t = torch.tensor(vals, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=CTRL)
    return t.tolist()


class Symm:  # same as e1_pairs.Symm; vLLM uses it for <= 64 MB and NCCL for larger
    def __init__(self, dev, ws):
        import torch.distributed._symmetric_memory as sm
        self.buf = sm.empty(SYMM_MAX_BYTES // 2, device=dev, dtype=torch.bfloat16)
        self.group = dist.group.WORLD
        handle = sm.rendezvous(self.buf, self.group.group_name)
        multi = (ws in (4, 6, 8) or os.environ.get("E1_SYMM_FORCE_MULTIMEM") == "1") and handle.multicast_ptr != 0
        self.algo = "multimem" if multi else "two_shot"

    def __call__(self, x):
        n = x.numel(); v = self.buf[:n]
        v.copy_(x.view(-1))
        if self.algo == "multimem":
            torch.ops.symm_mem.multimem_all_reduce_(v, "sum", self.group.group_name)
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(v, "sum", self.group.group_name)
        x.view(-1).copy_(v)


class Model:
    def __init__(self, dev, rank, mech, ws):
        self.dev, self.mech = dev, mech
        self.symm = Symm(dev, ws) if mech == "symm" else None
        self.w = []
        for l in range(NL):
            g = torch.Generator(device=dev).manual_seed(SEED * 1000 + rank * 100 + l)

            def W(k, n):
                return (torch.randn(k, n, device=dev, generator=g) * k ** -0.5).to(torch.bfloat16)
            self.w.append(dict(qkv=W(HID, QKV_N), o=W(O_K, HID), gu=W(HID, GU_N), dn=W(DN_K, HID),
                               n1=torch.ones(HID, device=dev, dtype=torch.bfloat16), n2=torch.ones(HID, device=dev, dtype=torch.bfloat16)))
        self.ms = torch.cuda.current_stream(dev)   # compute stream
        self.cs = torch.cuda.Stream(device=dev)    # communication stream
        self.alloc(0)

    def alloc(self, L):
        self.L = L
        dev = self.dev
        z = lambda *s: torch.zeros(*s, device=dev, dtype=torch.bfloat16)
        torch.cuda.empty_cache()
        self.X = [z(L, HID) for _ in range(NL + 1)] if L else []
        self.H = [z(L, HID) for _ in range(NL)] if L else []
        self.O = [z(L, HID) for _ in range(NL)] if L else []
        self.D = [z(L, HID) for _ in range(NL)] if L else []
        self.QKV = z(L, QKV_N) if L else None

    def set_input(self, seed):
        g = torch.Generator(device=self.dev).manual_seed(seed)
        self.X[0].copy_((torch.randn(self.L, HID, device=self.dev, generator=g) * 0.5).to(torch.bfloat16))

    # -- all-reduce of a contiguous row slice, issued on the CURRENT stream
    def ar(self, buf):
        if self.mech == "symm" and buf.numel() * 2 <= SYMM_MAX_BYTES:
            self.symm(buf)
            return
        w = dist.all_reduce(buf, async_op=True)
        w.wait()

    def forward(self, k, mode, nl=NL):
        """Runs NL layers over rows [0, L) in k chunks; returns nothing (result in self.X[NL])."""
        L = self.L
        Lc = L // k
        ms, cs = self.ms, self.cs
        ar1 = [[None] * k for _ in range(NL)]   # events: all-reduce 1 of (layer, chunk) done
        ar2 = [[None] * k for _ in range(NL)]
        ov = mode == "overlap"

        def issue(buf, store, l, i):
            if mode == "noar":
                return
            if not ov:
                self.ar(buf)               # on the compute stream, blocking in stream order
                return
            ready = torch.cuda.Event(); ready.record(ms)
            cs.wait_event(ready)
            with torch.cuda.stream(cs):
                self.ar(buf)
                done = torch.cuda.Event(); done.record(cs)
            store[l][i] = done

        for l in range(nl):
            w = self.w[l]
            # ---- part A of every chunk: (prologue: x_l = h_{l-1} + d_{l-1}), norm, qkv, attention, o_proj, all-reduce 1
            for i in range(k):
                s, e = i * Lc, (i + 1) * Lc
                if l > 0:
                    if ar2[l - 1][i] is not None:
                        ms.wait_event(ar2[l - 1][i])
                    torch.add(self.H[l - 1][s:e], self.D[l - 1][s:e], out=self.X[l][s:e])
                xn = F.rms_norm(self.X[l][s:e], (HID,), w["n1"], 1e-6)
                torch.mm(xn, w["qkv"], out=self.QKV[s:e])
                q = self.QKV[s:e, :NQ * D].reshape(Lc, NQ, D).transpose(0, 1).unsqueeze(0)
                kk = self.QKV[:e, NQ * D:NQ * D + D].reshape(1, 1, e, D).expand(1, NQ, e, D).contiguous()
                vv = self.QKV[:e, NQ * D + D:NQ * D + 2 * D].reshape(1, 1, e, D).expand(1, NQ, e, D).contiguous()
                at = F.scaled_dot_product_attention(q, kk, vv, attn_mask=causal_lower_right(Lc, e))
                at = at.transpose(1, 2).reshape(Lc, NQ * D)
                torch.mm(at, w["o"], out=self.O[l][s:e])
                issue(self.O[l][s:e], ar1, l, i)
            # ---- part B of every chunk: wait all-reduce 1, residual, norm, gate_up, silu*mul, down, all-reduce 2
            for i in range(k):
                s, e = i * Lc, (i + 1) * Lc
                if ov and ar1[l][i] is not None:
                    ms.wait_event(ar1[l][i])
                torch.add(self.X[l][s:e], self.O[l][s:e], out=self.H[l][s:e])
                hn = F.rms_norm(self.H[l][s:e], (HID,), w["n2"], 1e-6)
                gu = torch.mm(hn, w["gu"])
                act = F.silu(gu[:, :DN_K]) * gu[:, DN_K:]
                torch.mm(act, w["dn"], out=self.D[l][s:e])
                issue(self.D[l][s:e], ar2, l, i)
        # ---- epilogue: x_NL = h_{NL-1} + d_{NL-1}
        for i in range(k):
            s, e = i * Lc, (i + 1) * Lc
            if ar2[nl - 1][i] is not None:
                ms.wait_event(ar2[nl - 1][i])
            torch.add(self.H[nl - 1][s:e], self.D[nl - 1][s:e], out=self.X[nl][s:e])

    def timed(self, k, mode):
        torch.cuda.synchronize(); dist.barrier(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        self.forward(k, mode)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", required=True)
    ap.add_argument("--launch", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--lens", default="2048,4096,8192,16384")
    ap.add_argument("--ks", default="2,4")
    ap.add_argument("--tol", type=float, default=1e-2)
    a = ap.parse_args()

    rank = int(os.environ["RANK"]); ws = int(os.environ["WORLD_SIZE"]); lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    dist.init_process_group("nccl", device_id=dev)
    global CTRL
    CTRL = dist.new_group(backend="gloo")
    envs = {k: os.environ[k] for k in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MAX_CTAS", "NCCL_MIN_CTAS", "NCCL_NVLS_ENABLE") if k in os.environ}
    m = Model(dev, rank, a.mech, ws)
    w = dist.all_reduce(torch.zeros(1 << 20, dtype=torch.bfloat16, device=dev), async_op=True); w.wait(); torch.cuda.synchronize()
    fo = open(a.out, "a") if rank == 0 else None

    def emit(row):
        if rank == 0:
            row.update({"mech": a.mech, "launch": a.launch, "world_size": ws, "envs": envs, "seed": SEED, "tol_rel_fro": a.tol,
                        "symm_algo": m.symm.algo if m.symm else None})
            fo.write(json.dumps(row) + "\n"); fo.flush()
            t = row.get("ms")
            print(f"C L={row['L']} version={row['version']} k={row['k']} valid={row.get('valid')} err={row.get('rel_err')} "
                  f"median_ms={sorted(t)[len(t) // 2] if t else None}", flush=True)

    for L in [int(x) for x in a.lens.split(",")]:
        m.alloc(L)
        m.set_input(SEED)
        m.forward(1, "seq"); torch.cuda.synchronize()
        ref = m.X[NL].clone()
        refn = ref.float().norm()
        m.forward(1, "seq", nl=1); torch.cuda.synchronize()   # diagnostic: error after layer 1 only (rounding noise vs. a logic bug)
        ref1 = m.X[1].clone(); ref1n = ref1.float().norm()
        plan = [("C0", 1, "seq", None), ("C1", 1, "noar", None)] + [("C2", int(k), "overlap", int(k)) for k in a.ks.split(",")]
        for ver, k, mode, chk in plan:
            valid, err, err1, err_seq = True, None, None, None
            if chk is not None:
                m.X[1].zero_()
                m.forward(k, mode, nl=1); torch.cuda.synchronize()
                err1 = maxreduce([((m.X[1].float() - ref1.float()).norm() / ref1n).item()])[0]
                m.X[NL].zero_()
                m.forward(k, mode); torch.cuda.synchronize()
                err = maxreduce([((m.X[NL].float() - ref.float()).norm() / refn).item()])[0]
                valid = bool(err <= a.tol)
                # diagnostic: overlap vs the SAME chunking run sequentially (same shapes/kernels -> should match to ~0); isolates scheduling/sync bugs
                # from the rounding difference caused by chunking itself
                ov_out = m.X[NL].clone()
                m.X[NL].zero_(); m.forward(k, "seq"); torch.cuda.synchronize()
                err_seq = maxreduce([((ov_out.float() - m.X[NL].float()).norm() / ov_out.float().norm()).item()])[0]
                del ov_out
            for _ in range(a.warm):
                m.timed(k, mode)
            ts = [m.timed(k, mode) for _ in range(a.iters)]
            flat = maxreduce(ts)   # max over ranks, per iteration
            emit({"L": L, "version": ver, "k": k, "mode": mode, "valid": valid, "rel_err": err, "rel_err_layer1": err1, "rel_err_vs_seq_same_k": err_seq,
                  "warm": a.warm, "iters": a.iters, "ms": flat})
        m.alloc(0)
    if fo:
        fo.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
