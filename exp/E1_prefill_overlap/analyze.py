#!/usr/bin/env python
"""E1 analysis: prints, for every expectation on the exp card (A1, A2, B1-B4, C1-C3), the numbers needed to judge it, plus whether each
rule of the card's reading rules holds.  No verdict text: the judgement is written by Opus.
usage: analyze.py [--c-valid card|seq] <result dir> [...]   (--c-valid seq: a C2 setting counts as valid when it equals the same-k sequential run, rel err <= 1e-4, instead of the card's 1e-2 vs C0)     (one dir = one machine run; writes <dir>/analysis.json for each)
Statistics: median; 95% bootstrap CI of the median (2000 resamples of the pooled iterations of all launches); per-launch medians listed.
Slowdown = ratio concurrent/alone computed per iteration (the three measurements of one iteration are adjacent in time), median over the pooled iterations.
Overlap efficiency (per iteration) = (t_comm_alone + t_comp_alone - makespan) / min(t_comm_alone, t_comp_alone)  (share of the shorter side that was hidden).
Definitions of the card's rules used here (written so Opus can check them):
  A1  all-reduce share of GPU kernel time of one prefill step, median over all profiled steps of that L, in [0.25, 0.38]
  A2  |median(prefill time) - median(kernel time sum)| / median(kernel time sum) <= 0.10 ; prefill time = vLLM first_token_ts - scheduled_ts if
      vLLM gave it, else wall time of the request; both are printed
  B1  default mech: fraction of the 20 cells whose median comm slowdown >= 1.30 is >= 0.5; median over cells of the median comp slowdown in [1.05, 1.30]
  B2a median over cells of (comp_slow(cta8)-1)/(comp_slow(default)-1) < 0.5 ; B2b median over cells with T>=4096 of comm_alone(cta8)/comm_alone(default) >= 1.20
      (cells where default comp slowdown <= 1.01 are left out of B2a); fractions of cells are printed too
  B3  per cell the mech with the smallest median makespan; 'single best' = the mech with the smallest geometric mean of median makespan over the
      20 cells (only mechs that have all 20 cells); count cells where the per-cell best is >= 5% faster than the single best (needs >= 3)
  B4  default mech: median over cells of the median overlap efficiency in [0.40, 0.80]
  C1  (T_C0 - T_C1) / T_C0 per L in [0.25, 0.35]
  C2  fraction of the gap realised = (T_C0 - T_C2) / (T_C0 - T_C1) with the better of k=2,4 (by median time), default mech; per L in [0.20, 0.60]
  C3  per L: realised fraction of the best of the two non-default mechs (b/c_mechs.txt) minus realised fraction of default >= 0.10; needs >= 2 L
"""
import glob, json, math, os, statistics as st, sys
import numpy as np

RNG = np.random.default_rng(20261125)


def med(x):
    return float(np.median(x))


def ci(x, stat=np.median, n=2000):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    b = stat(x[idx], axis=1)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def ratio_ci(num, den, n=2000):
    """95% CI of median(num)/median(den), resampling each side independently."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    a = np.median(num[RNG.integers(0, len(num), size=(n, len(num)))], axis=1)
    b = np.median(den[RNG.integers(0, len(den), size=(n, len(den)))], axis=1)
    r = a / b
    return (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


def f(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def fci(c, d=3):
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


def load_jsonl(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        for line in open(p):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sec_a(D, R):
    print("\n=== A: vLLM prefill baseline")
    files = sorted(glob.glob(os.path.join(D, "a", "*", "result.json")))
    if not files:
        print("no a/*/result.json"); return
    res = [json.load(open(p)) for p in files]
    print("launches:", len(res), "orders:", [r["order"] for r in res], "seeds:", [r["seed"] for r in res], "errors:", [r["errors"] for r in res])
    lens = sorted({int(L) for r in res for L in r["profile"]})
    out = {}
    print("L\tn_prof\tAR_share_median\tCI95\tper_launch_medians\tkernel_sum_ms\tgpu_span_ms\tprefill_ms(vllm)\twall_ms\tA2_diff_prefill\tA2_diff_wall\tn_AR_calls\tAR_p50_us")
    for L in lens:
        ent = [e for r in res for e in r["profile"].get(str(L), [])]
        if not ent:
            continue
        sh = [e["allreduce_share"] for e in ent]
        per = [med([e["allreduce_share"] for e in r["profile"].get(str(L), [])]) for r in res if r["profile"].get(str(L))]
        ks = med([e["kernel_time_us_total"] for e in ent]) / 1e3
        sp = med([e["gpu_span_us"] for e in ent]) / 1e3
        reqs = [q for r in res for q in r["requests"].get(str(L), []) if not q["discarded"]]
        wall = med([q["wall_s"] for q in reqs]) * 1e3 if reqs else float("nan")
        pf = [q["prefill_s"] for q in reqs if q["prefill_s"] is not None]
        pfm = med(pf) * 1e3 if pf else float("nan")
        d_pf = abs(pfm - ks) / ks if pf else float("nan")
        d_w = abs(wall - ks) / ks
        names = sorted({n for e in ent for n in e["allreduce_kernel_names"]})
        print(f"{L}\t{len(ent)}\t{f(med(sh), 4)}\t{fci(ci(sh), 4)}\t{[round(x, 4) for x in per]}\t{f(ks, 2)}\t{f(sp, 2)}\t{f(pfm, 2)}\t{f(wall, 2)}\t{f(d_pf)}\t{f(d_w)}\t"
              f"{med([e['n_allreduce_calls'] for e in ent])}\t{f(med([e['allreduce_p50_us'] for e in ent if e['allreduce_p50_us']]), 1)}")
        print(f"   all-reduce kernel names at L={L}: {names}")
        out[L] = {"share_median": med(sh), "share_ci": ci(sh), "per_launch": per, "kernel_sum_ms": ks, "gpu_span_ms": sp, "prefill_ms": pfm,
                  "wall_ms": wall, "a2_diff_prefill": d_pf, "a2_diff_wall": d_w, "ar_kernel_names": names}
    R["A"] = out
    n_in = sum(1 for v in out.values() if 0.25 <= v["share_median"] <= 0.38)
    n_ge25 = sum(1 for v in out.values() if v["share_median"] >= 0.25)
    n_lt20 = sum(1 for v in out.values() if v["share_median"] < 0.20)
    print(f"A1: L with share in [0.25,0.38]: {n_in}/{len(out)} ; L with share >= 0.25: {n_ge25}/{len(out)} ; L with share < 0.20: {n_lt20}/{len(out)} ; "
          f"A1 holds (all L in range): {n_in == len(out)}")
    ok2 = [v["a2_diff_prefill"] <= 0.10 for v in out.values() if not math.isnan(v["a2_diff_prefill"])]
    ok2w = [v["a2_diff_wall"] <= 0.10 for v in out.values()]
    print(f"A2: all L within 10% using vLLM prefill time: {all(ok2) if ok2 else 'n/a'} ; using wall time: {all(ok2w)}")


def sec_b(D, R):
    print("\n=== B: operator-level pairs")
    rows = load_jsonl(os.path.join(D, "b", "*.jsonl"))
    if not rows:
        print("no b/*.jsonl"); return
    cells = {}
    for r in rows:
        cells.setdefault((r["mech"], r["T"], r["comp"]), []).append(r)
    mechs = sorted({k[0] for k in cells})
    print("mechs:", mechs, "launches per mech:", {m: sorted({r['launch'] for r in rows if r['mech'] == m}) for m in mechs},
          "attn_mode:", sorted({r["attn_mode"] for r in rows}), "symm_algo:", sorted({str(r["symm_algo"]) for r in rows if r["mech"] == "symm"}))
    S = {}
    print("mech\tT\tcomp\tn\tcomm_alone_ms\tcomp_alone_ms\tcomm_slow\tCI95\tcomp_slow\tCI95\tmakespan_ms\tCI95\teff\tCI95\tmakespan_per_launch")
    for (m, T, c) in sorted(cells):
        rs = cells[(m, T, c)]
        a = np.concatenate([r["comm_alone_ms"] for r in rs]); b = np.concatenate([r["comp_alone_ms"] for r in rs])
        cp = np.concatenate([r["comp_conc_ms"] for r in rs]); cm = np.concatenate([r["comm_conc_ms"] for r in rs]); mk = np.concatenate([r["makespan_ms"] for r in rs])
        sc, sp = cm / a, cp / b
        eff = (a + b - mk) / np.minimum(a, b)
        d = {"comm_alone": med(a), "comp_alone": med(b), "comm_slow": med(sc), "comm_slow_ci": ci(sc), "comp_slow": med(sp), "comp_slow_ci": ci(sp),
             "makespan": med(mk), "makespan_ci": ci(mk), "eff": med(eff), "eff_ci": ci(eff), "R": rs[0]["R"],
             "makespan_launch": [med(r["makespan_ms"]) for r in rs], "n": int(len(mk))}
        S[(m, T, c)] = d
        print(f"{m}\t{T}\t{c}\t{d['n']}\t{f(d['comm_alone'])}\t{f(d['comp_alone'])}\t{f(d['comm_slow'])}\t{fci(d['comm_slow_ci'])}\t{f(d['comp_slow'])}\t{fci(d['comp_slow_ci'])}\t"
              f"{f(d['makespan'])}\t{fci(d['makespan_ci'])}\t{f(d['eff'])}\t{fci(d['eff_ci'])}\t{[round(x, 3) for x in d['makespan_launch']]}")
    R["B"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in S.items()}
    # B1
    dc = {(k[1], k[2]): v for k, v in S.items() if k[0] == "default"}
    if dc:
        fr = sum(1 for v in dc.values() if v["comm_slow"] >= 1.30) / len(dc)
        mc = med([v["comp_slow"] for v in dc.values()])
        print(f"B1 (default, {len(dc)} cells): fraction of cells with comm slowdown >= 1.30 = {f(fr)} (needs >= 0.5: {fr >= 0.5}); median over cells of comp slowdown = {f(mc)} "
              f"(in [1.05,1.30]: {1.05 <= mc <= 1.30}); B1 holds: {fr >= 0.5 and 1.05 <= mc <= 1.30}")
    # B2
    c8 = {(k[1], k[2]): v for k, v in S.items() if k[0] == "cta8"}
    if dc and c8:
        ex = [((c8[k]["comp_slow"] - 1) / (dc[k]["comp_slow"] - 1), k) for k in dc if k in c8 and dc[k]["comp_slow"] > 1.01]
        m2a = med([x for x, _ in ex]) if ex else float("nan")
        f2a = sum(1 for x, _ in ex if x < 0.5) / len(ex) if ex else float("nan")
        big = [c8[k]["comm_alone"] / dc[k]["comm_alone"] for k in dc if k in c8 and k[0] >= 4096]
        m2b = med(big) if big else float("nan")
        f2b = sum(1 for x in big if x >= 1.2) / len(big) if big else float("nan")
        print(f"B2a: median over {len(ex)} cells of (comp_slow_cta8-1)/(comp_slow_default-1) = {f(m2a)} (<0.5: {m2a < 0.5}); fraction of cells < 0.5 = {f(f2a)}")
        print(f"B2b: median over {len(big)} cells (T>=4096) of comm_alone(cta8)/comm_alone(default) = {f(m2b)} (>=1.20: {m2b >= 1.2}); fraction of cells >= 1.20 = {f(f2b)}; B2 holds: {m2a < 0.5 and m2b >= 1.2}")
        for k in sorted(dc):
            if k in c8:
                print(f"   B2 cell T={k[0]} {k[1]}: comp_slow default {f(dc[k]['comp_slow'])} cta8 {f(c8[k]['comp_slow'])} ; comm_alone default {f(dc[k]['comm_alone'])} cta8 {f(c8[k]['comm_alone'])}")
    # B3
    allcells = sorted(dc)
    if allcells:
        full = [m for m in mechs if all((m, T, c) in S for (T, c) in allcells)]
        gm = {m: math.exp(sum(math.log(S[(m, T, c)]["makespan"]) for (T, c) in allcells) / len(allcells)) for m in full}
        best_fixed = min(gm, key=gm.get)
        n3 = 0
        print(f"B3: mechs with all {len(allcells)} cells: {full}; single best by geomean makespan: {best_fixed} {{{', '.join(f'{m}:{gm[m]:.4f}' for m in full)}}}")
        for (T, c) in allcells:
            av = {m: S[(m, T, c)]["makespan"] for m in mechs if (m, T, c) in S}
            bm = min(av, key=av.get); gain = 1 - av[bm] / av[best_fixed]
            n3 += gain >= 0.05
            print(f"   B3 cell T={T} {c}: best={bm} {f(av[bm])} ms ; single-best({best_fixed}) {f(av[best_fixed])} ms ; gain={f(gain)}")
        print(f"B3: cells with per-cell best >= 5% faster than single best: {n3} (needs >= 3: {n3 >= 3})")
    # B4
    for m in mechs:
        e = [S[k]["eff"] for k in S if k[0] == m]
        print(f"B4 mech={m}: median overlap efficiency over {len(e)} cells = {f(med(e))} (in [0.40,0.80]: {0.40 <= med(e) <= 0.80})")


def sec_c(D, R):
    print("\n=== C: layer-stack prototype")
    rows = load_jsonl(os.path.join(D, "c", "*.jsonl"))
    if not rows:
        print("no c/*.jsonl"); return
    mechs = sorted({r["mech"] for r in rows})
    print("mechs:", mechs, "launches:", {m: sorted({r['launch'] for r in rows if r['mech'] == m}) for m in mechs})
    seqmode = os.environ.get("E1_C_VALID", "card") == "seq"
    if seqmode:
        for r in rows:
            if r["rel_err_vs_seq_same_k"] is not None:
                r["valid"] = r["rel_err_vs_seq_same_k"] <= 1e-4
        print("NOTE: C validity judged by --c-valid seq (overlap vs same-k sequential <= 1e-4), NOT the card's 1e-2 vs C0")
    inval = [(r["mech"], r["launch"], r["L"], r["k"], r["rel_err"]) for r in rows if not r["valid"]]
    print("INVALID settings (rel_fro_err > tol):", inval if inval else "none", "; max rel_err among C2:", max([r["rel_err"] for r in rows if r["rel_err"] is not None], default=None))
    print("C2 correctness (mech, launch, L, k, rel_fro_err after 8 layers, rel_fro_err after layer 1, err vs same-k sequential, valid):")
    for r in rows:
        if r["rel_err"] is not None:
            print(f"   {r['mech']}\t{r['launch']}\t{r['L']}\t{r['k']}\t{r['rel_err']:.5f}\t{r.get('rel_err_layer1')}\t{r.get('rel_err_vs_seq_same_k')}\t{r['valid']}")
    lens = sorted({r["L"] for r in rows})
    T = {}
    for r in rows:
        key = (r["mech"], r["L"], r["version"] + (str(r["k"]) if r["version"] == "C2" else ""))
        if r["valid"]:
            T.setdefault(key, []).append(r)
    print("mech\tL\tversion\tn\tmedian_ms\tCI95\tper_launch_medians")
    M = {}
    for key in sorted(T):
        ms = np.concatenate([r["ms"] for r in T[key]])
        M[key] = ms
        print(f"{key[0]}\t{key[1]}\t{key[2]}\t{len(ms)}\t{f(med(ms), 2)}\t{fci(ci(ms), 2)}\t{[round(med(r['ms']), 2) for r in T[key]]}")
    out = {}
    frac_tab = {}
    for m in mechs:
        for L in lens:
            c0, c1 = M.get((m, L, "C0")), M.get((m, L, "C1"))
            if c0 is None or c1 is None:
                continue
            up = 1 - med(c1) / med(c0)
            upci = tuple(1 - x for x in ratio_ci(c1, c0))[::-1]
            cand = {k: M[(m, L, f"C2{k}")] for k in (2, 4) if (m, L, f"C2{k}") in M}
            if not cand:
                print(f"C {m} L={L}: C1 gain {f(up)} {fci(upci)}; no valid C2"); continue
            kb = min(cand, key=lambda k: med(cand[k]))
            c2 = cand[kb]
            gap = med(c0) - med(c1)
            frac = (med(c0) - med(c2)) / gap
            # CI of the realised fraction: resample the three samples independently
            n = 2000
            A = np.median(c0[RNG.integers(0, len(c0), size=(n, len(c0)))], axis=1)
            B = np.median(c1[RNG.integers(0, len(c1), size=(n, len(c1)))], axis=1)
            Cc = np.median(c2[RNG.integers(0, len(c2), size=(n, len(c2)))], axis=1)
            fr = (A - Cc) / (A - B)
            frci = (float(np.percentile(fr, 2.5)), float(np.percentile(fr, 97.5)))
            kfr = {k: (med(c0) - med(cand[k])) / gap for k in cand}
            print(f"C mech={m} L={L}: C1 gain (upper bound) = {f(up)} CI95 {fci(upci)} ; C2 best k={kb} realised fraction of gap = {f(frac)} CI95 {fci(frci)} ; per k: {{{', '.join(f'{k}:{v:.3f}' for k, v in kfr.items())}}} ; "
                  f"C2 speedup over C0 = {f(med(c0) / med(c2))}")
            out[(m, L)] = {"c1_gain": up, "c1_gain_ci": upci, "k_best": kb, "frac": frac, "frac_ci": frci, "frac_by_k": kfr}
            frac_tab[(m, L)] = frac
    R["C"] = {f"{k[0]}|{k[1]}": v for k, v in out.items()}
    d = {L: out[("default", L)] for L in lens if ("default", L) in out}
    if d:
        print("C1: L with gain in [0.25,0.35]:", {L: f(v["c1_gain"]) + (" ok" if 0.25 <= v["c1_gain"] <= 0.35 else " OUT") for L, v in d.items()})
        print("C2 (default): L with realised fraction in [0.20,0.60]:", {L: f(v["frac"]) + (" ok" if 0.20 <= v["frac"] <= 0.60 else " OUT") for L, v in d.items()})
        n_le60 = sum(1 for v in d.values() if v["frac"] <= 0.60)
        n_gt80 = sum(1 for v in d.values() if v["frac"] > 0.80)
        print(f"C2: L with fraction <= 0.60: {n_le60}/{len(d)} (needs >= 3 for the core rule) ; L with fraction > 0.80: {n_gt80}/{len(d)} (all L > 0.80 -> 'topic weak')")
        cm = [m for m in mechs if m != "default"]
        n3 = 0
        for L in d:
            best = max(((frac_tab[(m, L)], m) for m in cm if (m, L) in frac_tab), default=None)
            if best:
                diff = best[0] - d[L]["frac"]
                n3 += diff >= 0.10
                print(f"   C3 L={L}: default {f(d[L]['frac'])} ; best non-default {best[1]} {f(best[0])} ; difference (pct points/100) {f(diff)}")
        print(f"C3: L with improvement >= 0.10: {n3} (needs >= 2: {n3 >= 2})")
    # card-level rules (per machine)
    a = R.get("A", {})
    if a and d:
        n_ge25 = sum(1 for v in a.values() if v["share_median"] >= 0.25)
        n_le60 = sum(1 for v in d.values() if v["frac"] <= 0.60)
        print(f"RULE core (this machine): A1 L with share >= 0.25: {n_ge25} (needs >= 3 on both machines) AND C2 L with fraction <= 0.60: {n_le60} (needs >= 3)")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--c-valid":
        os.environ["E1_C_VALID"] = args[1]; args = args[2:]
    for D in args:
        print(f"##### {D}")
        R = {}
        for fn in (sec_a, sec_b, sec_c):
            try:
                fn(D, R)
            except Exception as e:
                import traceback
                print(f"ERROR in {fn.__name__}: {e!r}")
                traceback.print_exc()
        with open(os.path.join(D, "analysis.json"), "w") as fo:
            json.dump(R, fo, indent=1, default=str)


if __name__ == "__main__":
    main()
