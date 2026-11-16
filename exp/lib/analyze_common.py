"""Shared analysis for M21 / M23 (S06). numpy only; matplotlib optional (figures skipped if missing).

Input: job result dirs written by exp/lib/nccl_bench.sh (runs.tsv, runs/*.log.gz, cotenants_summary.txt, pairs.tsv).
Per (job, cfg, size): out-of-place and in-place time_us / busbw from each of the 5 independent launches (kind=run);
across launches: median and 95% percentile-bootstrap CI of the median. Algorithm actually used comes from the
probe launch (kind=probe, NCCL_DEBUG_SUBSYS=TUNING: lines "<coll>: <bytes> Bytes -> Algo X proto Y channel{Lo..Hi}={a..b}").
"""
import csv, glob, gzip, os, re, sys
from collections import defaultdict
import numpy as np

ROW = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(-?\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+|N/A)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+|N/A)\s*$')
ALGO = re.compile(r'(\d+) Bytes -> Algo (\S+) proto (\S+)')
RNG = np.random.default_rng(20261117)


def opengz(p):
    return gzip.open(p, 'rt', errors='replace') if p.endswith('.gz') else open(p, errors='replace')


def parse_log(path):
    """-> list of dict(size, oop_time, oop_bw, oop_wrong, ip_time, ip_bw, ip_wrong) ; also first 'Avg bus bandwidth' ignored."""
    rows = []
    with opengz(path) as f:
        for line in f:
            m = ROW.match(line)
            if m:
                g = m.groups()
                rows.append(dict(size=int(g[0]), oop_time=float(g[5]), oop_bw=float(g[7]), oop_wrong=g[8],
                                 ip_time=float(g[9]), ip_bw=float(g[11]), ip_wrong=g[12]))
    return rows


def parse_algos(path):
    """probe log -> {size: (algo, proto)} (rank-0-ish first occurrence per size)"""
    d = {}
    with opengz(path) as f:
        for line in f:
            m = ALGO.search(line)
            if m:
                d.setdefault(int(m.group(1)), (m.group(2), m.group(3)))
    return d


def cotenants(jobdir):
    """-> max other_jobs_on_our_nodes over start/end tags, or None if no record (job predates the helper)"""
    p = os.path.join(jobdir, 'cotenants_summary.txt')
    if not os.path.exists(p):
        return None
    vals = [int(m.group(1)) for m in re.finditer(r'other_jobs_on_our_nodes=(\d+)', open(p).read())]
    return max(vals) if vals else None


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    med = float(np.median(x))
    if len(x) == 1:
        return (med, med, med)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    meds = np.median(x[idx], axis=1)
    return (med, float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def load_jobs(pattern):
    """Collect per-launch data. returns (data, meta, algos)
    data[(job, cfg)][size] = list of dict (one per launch, only launches whose segments all had rc==0 contribute rows from that segment)
    meta[(job,cfg)] = dict(nodes, shared, rc_fail)  ; algos[(job,cfg)] = {size:(algo,proto)}"""
    data = defaultdict(lambda: defaultdict(list)); meta = {}; algos = {}
    for jd in sorted(glob.glob(pattern)):
        if not os.path.isfile(os.path.join(jd, 'runs.tsv')):
            continue
        job = os.path.basename(jd)
        shared = cotenants(jd)
        for r in csv.reader(open(os.path.join(jd, 'runs.tsv')), delimiter='\t'):
            if len(r) < 12:
                continue
            seq, cfg, kind, rep, seg, rc, t0, t1, mode, nodes, gpus, desc = r[:12]
            key = (job, cfg)
            m = meta.setdefault(key, dict(nodes=nodes, shared=shared, rc_fail=0, mode=mode, desc=desc, n_runs=0))
            lp = os.path.join(jd, 'runs', f'{cfg}__{kind}{rep}.{seg}.log.gz')
            if not os.path.exists(lp):
                continue
            if kind == 'probe':
                algos.setdefault(key, {}).update(parse_algos(lp))
                continue
            if int(rc) != 0:
                m['rc_fail'] += 1
            for row in parse_log(lp):   # rows are kept even if rc!=0 (partial sweep), but the rc is counted/flagged
                row['rep'] = int(rep)
                data[key][row['size']].append(row)
            m['n_runs'] += 1
    return data, meta, algos


def summarize(data, meta, algos, out_csv):
    rows = []
    for (job, cfg), bysize in sorted(data.items()):
        m = meta[(job, cfg)]
        al = algos.get((job, cfg), {})
        for size, lst in sorted(bysize.items()):
            if size == 0:
                continue
            rec = dict(job=job, cfg=cfg, nodes=m['nodes'], size=size, n=len(lst),
                       shared_flag=('unknown' if m['shared'] is None else int(m['shared'] > 0)),
                       rc_fail_segments=m['rc_fail'],
                       algo_proto_probe=('/'.join(al[size]) if size in al else ''),
                       wrong_any=int(any(str(x['oop_wrong']) not in ('0',) or str(x['ip_wrong']) not in ('0',) for x in lst)))
            for place in ('oop', 'ip'):
                for met in ('time', 'bw'):
                    med, lo, hi = boot_ci([x[f'{place}_{met}'] for x in lst])
                    rec[f'{place}_{met}_med'] = med; rec[f'{place}_{met}_lo'] = lo; rec[f'{place}_{met}_hi'] = hi
            rec['oop_bw_halfwidth_pct'] = (100 * (rec['oop_bw_hi'] - rec['oop_bw_lo']) / 2 / rec['oop_bw_med']) if rec['oop_bw_med'] else ''
            rows.append(rec)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if rows:
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'wrote {out_csv}: {len(rows)} rows; shared_flag: unknown = job has no cotenants_summary.txt, 1 = other Slurm jobs were on our nodes at start/end')
    return rows


def get_plt():
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; return plt
    except ImportError:
        print('matplotlib missing: figures skipped (CSV still written)', file=sys.stderr); return None


def lineplot(plt, rows, cfgs, metric, title, path, jobfilter=None, ylabel=None, logy=False):
    """rows: summary rows; cfgs: list of (label, cfg-name[, job-substring]); error bars = 95% bootstrap CI"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for item in cfgs:
        label, cfg = item[0], item[1]
        sel = sorted([r for r in rows if r['cfg'] == cfg and (len(item) < 3 or item[2] in r['job'])], key=lambda r: r['size'])
        if not sel:
            continue
        x = [r['size'] for r in sel]; y = np.array([r[f'oop_{metric}_med'] for r in sel])
        lo = y - np.array([r[f'oop_{metric}_lo'] for r in sel]); hi = np.array([r[f'oop_{metric}_hi'] for r in sel]) - y
        ax.errorbar(x, y, yerr=[lo, hi], marker='o', ms=3, lw=1, capsize=2, label=label)
    ax.set_xscale('log', base=2)
    if logy or metric == 'time':
        ax.set_yscale('log')
    ax.set_xlabel('message size (bytes)'); ax.set_ylabel(ylabel or ('bus bandwidth (GB/s)' if metric == 'bw' else 'time (us)'))
    ax.set_title(title, fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True); fig.savefig(path, dpi=130); plt.close(fig)
    print('figure', path)
