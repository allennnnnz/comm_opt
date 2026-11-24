#!/usr/bin/env python
"""M25 B1: collective communication and GEMM running at the same time (single node, torchrun).

Modes
  --calib FILE     calibration launch (not a measurement): decide n_g (GEMMs per 'GEMM-alone' ~100 ms) and
                   n_c (collective calls per cell so comm-alone ~ 50% of GEMM time); writes FILE (calib.json)
  --calib-in FILE  measurement launch: read the same FILE, run all (or filtered) cells, append JSONL rows to --out

Sizes: all_reduce input, reduce_scatter input, all_gather output are all SIZE MB of bf16.
Concurrent run: default stream sleeps ~1 ms, records a start event; GEMM stream and comm stream both wait on it,
then each enqueues its own work; times are CUDA events from the shared start to each stream's end.
Per-rank times are max-reduced over ranks (every metric is the cross-rank maximum).
"""
import argparse, json, os, time
import torch
import torch.distributed as dist

OPS = ["all_reduce", "reduce_scatter", "all_gather"]
SIZES_MB = [4, 32, 256]
GEMMS = [8192, 4096]
REPS = 3


def make_comm(op, size_mb, ws, dev):
    numel = size_mb * 1024 * 1024 // 2  # bf16
    numel -= numel % ws
    if op == "all_reduce":
        x = torch.full((numel,), 0.001, dtype=torch.bfloat16, device=dev)
        return lambda: dist.all_reduce(x, async_op=True)
    if op == "reduce_scatter":
        inp = torch.full((numel,), 0.001, dtype=torch.bfloat16, device=dev)
        out = torch.empty(numel // ws, dtype=torch.bfloat16, device=dev)
        return lambda: dist.reduce_scatter_tensor(out, inp, async_op=True)
    if op == "all_gather":
        out = torch.empty(numel, dtype=torch.bfloat16, device=dev)
        inp = torch.full((numel // ws,), 0.001, dtype=torch.bfloat16, device=dev)
        return lambda: dist.all_gather_into_tensor(out, inp, async_op=True)
    raise ValueError(op)


class Runner:
    def __init__(self, dev, ws):
        self.dev, self.ws = dev, ws
        self.gs = torch.cuda.Stream(device=dev)
        self.cs = torch.cuda.Stream(device=dev)
        # ~1 ms of GPU sleep; clock rate (kHz) from device properties
        khz = torch.cuda.get_device_properties(dev).clock_rate
        self.sleep_cycles = int(khz * 1.0)  # kHz * 1 ms = cycles
        self.mats = {}

    def gemm_fn(self, n):
        if n not in self.mats:
            a = torch.randn(n, n, device=self.dev, dtype=torch.bfloat16)
            b = torch.randn(n, n, device=self.dev, dtype=torch.bfloat16)
            c = torch.empty(n, n, device=self.dev, dtype=torch.bfloat16)
            self.mats[n] = (a, b, c)
        a, b, c = self.mats[n]
        return lambda: torch.mm(a, b, out=c)

    def _start(self):
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda._sleep(self.sleep_cycles)
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def _enq_gemm(self, fn, n, start):
        self.gs.wait_event(start)
        with torch.cuda.stream(self.gs):
            for _ in range(n):
                fn()
            e = torch.cuda.Event(enable_timing=True)
            e.record()
        return e

    def _enq_comm(self, fn, n, start):
        self.cs.wait_event(start)
        with torch.cuda.stream(self.cs):
            works = [fn() for _ in range(n)]
            for w in works:
                w.wait()
            e = torch.cuda.Event(enable_timing=True)
            e.record()
        return e

    def gemm_alone(self, fn, n):
        s = self._start()
        e = self._enq_gemm(fn, n, s)
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    def comm_alone(self, fn, n):
        s = self._start()
        e = self._enq_comm(fn, n, s)
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    def conc(self, gfn, ng, cfn, nc):
        s = self._start()
        eg = self._enq_gemm(gfn, ng, s)
        ec = self._enq_comm(cfn, nc, s)
        torch.cuda.synchronize()
        return s.elapsed_time(eg), s.elapsed_time(ec)


CTRL = None  # gloo group for timing bookkeeping (Opus 2026-11-25: job 497945 NVLS cfgs died here -- NCCL_ALGO=NVLS has no
             # float64 all-reduce; control-plane reductions must not go through the NCCL config under test)


def maxreduce(vals, dev):
    t = torch.tensor(vals, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=CTRL)
    return t.tolist()


def per_call_ms(r, fn, kind, dev, n=20):
    fn_warm = [fn() for _ in range(3)] if kind == "comm" else [fn() for _ in range(3)]
    if kind == "comm":
        for w in fn_warm:
            w.wait()
    torch.cuda.synchronize()
    t = r.comm_alone(fn, n) if kind == "comm" else r.gemm_alone(fn, n)
    return maxreduce([t / n], dev)[0]


def cell_list(filt):
    cells = [(op, s, g) for op in OPS for s in SIZES_MB for g in GEMMS]
    if filt:
        want = {tuple(x.split(":")) for x in filt.split(",")}
        cells = [c for c in cells if (c[0], str(c[1]), str(c[2])) in want]
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib")
    ap.add_argument("--calib-in")
    ap.add_argument("--out")
    ap.add_argument("--cfg", default="default")
    ap.add_argument("--launch", type=int, default=0)
    ap.add_argument("--cells", default="")
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()

    rank = int(os.environ["RANK"]); ws = int(os.environ["WORLD_SIZE"]); lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    dist.init_process_group("nccl", device_id=dev)
    global CTRL
    CTRL = dist.new_group(backend="gloo")
    r = Runner(dev, ws)
    cells = cell_list(a.cells)

    # global warm-up: touch every op and both GEMM sizes once
    for op in OPS:
        fn = make_comm(op, 4, ws, dev)
        for w in [fn() for _ in range(5)]:
            w.wait()
    for g in GEMMS:
        gf = r.gemm_fn(g)
        for _ in range(5):
            gf()
    torch.cuda.synchronize()

    if a.calib:
        calib = {"n_g": {}, "t_gemm_ms": {}, "n_c": {}, "t_comm_ms": {}, "world_size": ws}
        for g in GEMMS:
            gf = r.gemm_fn(g)
            t = per_call_ms(r, gf, "gemm", dev, n=50)
            ng = max(1, int(round(100.0 / t)))
            calib["n_g"][str(g)] = ng
            calib["t_gemm_ms"][str(g)] = t
        for op in OPS:
            for s in SIZES_MB:
                cf = make_comm(op, s, ws, dev)
                t = per_call_ms(r, cf, "comm", dev, n=20)
                calib["t_comm_ms"][f"{op}|{s}"] = t
                for g in GEMMS:
                    tg = calib["n_g"][str(g)] * calib["t_gemm_ms"][str(g)]
                    calib["n_c"][f"{op}|{s}|{g}"] = max(1, int(round(0.5 * tg / t)))
        if rank == 0:
            with open(a.calib, "w") as f:
                json.dump(calib, f, indent=1)
            print("calib written", a.calib, json.dumps(calib["n_g"]), flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    calib = json.load(open(a.calib_in))
    fo = open(a.out, "a") if rank == 0 else None
    for (op, s, g) in cells:
        ng = calib["n_g"][str(g)]
        nc = calib["n_c"][f"{op}|{s}|{g}"]
        cf = make_comm(op, s, ws, dev)
        gf = r.gemm_fn(g)
        # per-cell warm-up (one concurrent run, discarded)
        r.conc(gf, ng, cf, nc)
        for rep in range(a.reps):
            tga = r.gemm_alone(gf, ng)
            tca = r.comm_alone(cf, nc)
            tgc, tcc = r.conc(gf, ng, cf, nc)
            mk = max(tgc, tcc)
            vals = maxreduce([tga, tca, tgc, tcc, mk], dev)
            if rank == 0:
                fo.write(json.dumps({
                    "cfg": a.cfg, "launch": a.launch, "op": op, "size_mb": s, "gemm": g, "rep": rep,
                    "n_g": ng, "n_c": nc,
                    "t_gemm_alone": vals[0], "t_comm_alone": vals[1],
                    "t_gemm_conc": vals[2], "t_comm_conc": vals[3], "makespan": vals[4]}) + "\n")
                fo.flush()
    if fo:
        fo.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
