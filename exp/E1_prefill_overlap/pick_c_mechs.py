#!/usr/bin/env python
"""E1: choose the two non-default communication mechanisms for block C from block B results (stdlib only).

Rule (spec S09 / exp card): for every mechanism and every (T, comp) cell take the median (all iterations of all launches pooled) of the
concurrent completion time (makespan_ms); ratio = median(mech) / median(default) in the same cell; score = geometric mean of the ratios over
the cells the mechanism has (symm_mem only has T <= 4096 cells -> compared only on those cells, and the number of cells is written);
the two non-default mechanisms with the smallest score are chosen.
usage: pick_c_mechs.py <b dir>   -> writes <b dir>/c_mechs.txt  (lines "mech1=<name>", "mech2=<name>" + the score table)
"""
import glob, json, math, os, statistics as st, sys


def main():
    b = sys.argv[1]
    cells = {}
    for f in sorted(glob.glob(os.path.join(b, "*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cells.setdefault((r["mech"], r["T"], r["comp"]), []).extend(r["makespan_ms"])
    med = {k: st.median(v) for k, v in cells.items()}
    mechs = sorted({k[0] for k in med})
    out = []
    if "default" not in mechs:
        out.append("ERROR no default mech data")
        open(os.path.join(b, "c_mechs.txt"), "w").write("\n".join(out) + "\n")
        sys.exit(1)
    scores = {}
    for m in mechs:
        if m == "default":
            continue
        rs = [med[k] / med[("default", k[1], k[2])] for k in med if k[0] == m and ("default", k[1], k[2]) in med]
        if rs:
            scores[m] = (math.exp(sum(math.log(x) for x in rs) / len(rs)), len(rs))
    ranked = sorted(scores.items(), key=lambda kv: kv[1][0])
    for i, (m, (s, n)) in enumerate(ranked[:2], 1):
        out.append(f"mech{i}={m}")
    out.append("# geometric-mean ratio of median concurrent completion time vs default (smaller = better), and number of cells compared")
    for m, (s, n) in ranked:
        out.append(f"# {m}\tscore={s:.4f}\tcells={n}" + ("\t(symm_mem: only cells with T<=4096)" if m == "symm" else ""))
    open(os.path.join(b, "c_mechs.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
