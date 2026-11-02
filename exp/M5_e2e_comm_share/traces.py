"""Split each step's wall time on every GPU into compute-only / exposed comm / overlap / idle.

Input: PyTorch (Kineto) traces, one per rank. Two ways to delimit a step on the GPU timeline:
- mode="cadence": from the start of its step annotation to the start of the next one, so CPU-side gaps
  between back-to-back steps count as idle (end-to-end view). The last step has no successor and is dropped.
- mode="span": the step annotation's own [start, end). Used when consecutive annotated steps are not
  back-to-back (e.g. one offline generate() call per prefill step).

Communication kernels are matched by name (COMM_RE), not by substrings like "reduce" or "gather",
which also appear in compute kernels (cublasLt splitKreduce, vLLM gather_block_tables).
"""
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

COMM_RE = re.compile(r"nccl|cross_device_reduce|custom_?all_?reduce|one_shot_all_reduce|two_shot_all_reduce|"
                     r"multimem|symm_mem", re.I)
GPU_WORK_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def union(iv: np.ndarray) -> np.ndarray:
    """Merge overlapping [start, end) intervals."""
    if len(iv) == 0:
        return iv.reshape(0, 2)
    iv = iv[np.argsort(iv[:, 0])]
    out = [iv[0].copy()]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append(np.array([s, e]))
    return np.array(out)


def length(iv: np.ndarray) -> float:
    return float((iv[:, 1] - iv[:, 0]).sum()) if len(iv) else 0.0


def intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection of two merged interval lists."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s, e = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        if s < e:
            out.append((s, e))
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return np.array(out).reshape(-1, 2)


def clip(iv: np.ndarray, s: float, e: float) -> np.ndarray:
    iv = iv[(iv[:, 1] > s) & (iv[:, 0] < e)].copy()
    iv[:, 0] = np.maximum(iv[:, 0], s)
    iv[:, 1] = np.minimum(iv[:, 1], e)
    return iv


def load_rank(path: Path, step_re: str, mode: str = "cadence", merge_by_name: bool = False):
    opener = gzip.open if path.suffix == ".gz" else open
    ev = json.load(opener(path))["traceEvents"]
    work = [e for e in ev if e.get("cat") in GPU_WORK_CATS and e.get("dur", 0) > 0]
    is_comm = np.array([bool(COMM_RE.search(e["name"])) and e["cat"] == "kernel" for e in work])
    iv = np.array([(e["ts"], e["ts"] + e["dur"]) for e in work], dtype=float).reshape(-1, 2)
    ann = [(e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev
           if e.get("cat") == "gpu_user_annotation" and re.search(step_re, e["name"])]
    if merge_by_name:  # one step split into several GPU annotation segments (torch ProfilerStep#N)
        span = {}
        for s, t, n in ann:
            a, b = span.get(n, (s, t))
            span[n] = (min(a, s), max(b, t))
        ann = list(span.values())
    ann = sorted((s, t) for s, t, *_ in ann)
    if mode == "cadence":
        steps = [(s, n) for (s, _), (n, _) in zip(ann, ann[1:])]
    else:
        steps = ann
    comm_kernels = [(e["ts"], e["dur"], e["name"]) for e, c in zip(work, is_comm) if c]
    return iv, is_comm, steps, comm_kernels


def decompose(trace_files: list, step_re: str, mode: str = "cadence", merge_by_name: bool = False) -> tuple:
    """Return (per-rank-step breakdown, per-collective table, comm kernel name counts).

    Transfer vs waiting: the j-th comm kernel in the window is the same collective on every rank, so its
    shortest duration across ranks approximates the transfer (the last rank to arrive barely waits); the
    rest of each rank's duration is waiting for slower ranks. Exposed comm is split in the same proportion
    (exact when comm does not overlap compute, as in TP inference). Requires equal comm-kernel counts.
    """
    files = sorted(trace_files, key=lambda p: int(re.search(r"rank(\d+)", p.name).group(1)))
    ranks = [load_rank(f, step_re, mode, merge_by_name) for f in files]
    windows = [[(ts, dur) for ts, dur, _ in ck if any(s <= ts < e for s, e in st)] for _, _, st, ck in ranks]
    counts = {len(w) for w in windows}
    transfer = (np.min([[d for _, d in w] for w in windows], axis=0) if len(counts) == 1 and counts != {0}
                else None)
    rows, names = [], {}
    for rank, ((iv, is_comm, steps, comm_k), win) in enumerate(zip(ranks, windows)):
        comm_all, comp_all = union(iv[is_comm]), union(iv[~is_comm])
        win_ts = np.array([ts for ts, _ in win])
        win_dur = np.array([d for _, d in win])
        for i, (s, e) in enumerate(steps):
            c, k = union(clip(comm_all, s, e)), union(clip(comp_all, s, e))
            both = intersect(c, k)
            busy = length(union(np.vstack([c, k]))) if len(c) + len(k) else 0.0
            exposed = length(c) - length(both)
            sel = (win_ts >= s) & (win_ts < e)
            frac = transfer[sel].sum() / win_dur[sel].sum() if transfer is not None and win_dur[sel].sum() else np.nan
            rows.append(dict(rank=rank, step=i, wall_us=e - s, compute_only_us=length(k) - length(both),
                             exposed_comm_us=exposed, exposed_transfer_us=exposed * frac,
                             exposed_wait_us=exposed * (1 - frac), overlap_us=length(both), idle_us=e - s - busy,
                             comm_total_us=length(c), n_comm_kernels=int(sel.sum())))
        for _, _, n in comm_k:
            names[n[:120]] = names.get(n[:120], 0) + 1
    coll = pd.DataFrame()
    if transfer is not None:
        durs = np.array([[d for _, d in w] for w in windows])
        coll = pd.DataFrame(dict(idx=np.arange(len(transfer)), transfer_us=transfer,
                                 med_us=np.median(durs, axis=0), max_us=durs.max(axis=0)))
    return pd.DataFrame(rows), coll, names


def summarize(df: pd.DataFrame) -> dict:
    """Time shares over all captured steps: ratio of sums per rank, then the mean across ranks.

    Ratio of sums keeps compute_only + exposed + overlap + idle = 100% on every rank, and the mean across
    ranks preserves that sum (a median per component would not).
    """
    parts = ["compute_only_us", "exposed_comm_us", "exposed_transfer_us", "exposed_wait_us", "overlap_us", "idle_us",
             "comm_total_us"]
    per_rank = df.groupby("rank")[["wall_us"] + parts].sum()
    share = per_rank[parts].div(per_rank.wall_us, axis=0)
    out = {f"{c.replace('_us', '')}_pct": 100 * float(share[c].mean()) for c in share}
    out["step_wall_us"] = float(df.wall_us.median())
    out["n_steps"] = int(df.groupby("rank").size().min())
    return out


def overlap_slowdown(trace_file: Path, window: tuple | None = None, min_alone: int = 5) -> dict:
    """Hidden cost of overlap (Lee et al. 2507.03114): compute kernels that run while a comm kernel runs,
    compared with the same kernel (name + grid) running alone. Returns the extra time those overlapped
    instances took over the alone median, summed, and the overlapped compute time it applies to."""
    opener = gzip.open if trace_file.suffix == ".gz" else open
    ev = json.load(opener(trace_file))["traceEvents"]
    k = [e for e in ev if e.get("cat") == "kernel" and e.get("dur", 0) > 0
         and (window is None or window[0] <= e["ts"] < window[1])]
    comm = union(np.array([(e["ts"], e["ts"] + e["dur"]) for e in k if COMM_RE.search(e["name"])],
                          dtype=float).reshape(-1, 2))
    rows = []
    for e in k:
        if COMM_RE.search(e["name"]):
            continue
        s, t = e["ts"], e["ts"] + e["dur"]
        ov = length(intersect(np.array([[s, t]]), comm)) > 0.5 * e["dur"] if len(comm) else False
        rows.append((e["name"][:160], str(e.get("args", {}).get("grid")), e["dur"], ov))
    df = pd.DataFrame(rows, columns=["name", "grid", "dur", "overlapped"])
    if df.empty:
        return dict(extra_us=0.0, overlapped_compute_us=0.0, matched_overlapped_us=0.0)
    alone = df[~df.overlapped].groupby(["name", "grid"]).dur.agg(["median", "count"])
    alone = alone[alone["count"] >= min_alone]
    ovl = df[df.overlapped].join(alone, on=["name", "grid"], how="inner")
    return dict(extra_us=float((ovl.dur - ovl["median"]).sum()), overlapped_compute_us=float(df[df.overlapped].dur.sum()),
                matched_overlapped_us=float(ovl.dur.sum()))
