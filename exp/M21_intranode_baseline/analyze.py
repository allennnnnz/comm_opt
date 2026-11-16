#!/usr/bin/env python3
"""M21 analysis: python analyze.py [results glob]. Output: figures/M21_summary.csv and figures/M21_*.png (S06)."""
import sys
sys.path.insert(0, '/home/allen0117/comm_opt/exp/lib')
from analyze_common import *

ROOT = '/home/allen0117/comm_opt'
pat = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/results/M21_intranode_baseline/*_job*'
data, meta, algos = load_jobs(pat)
rows = summarize(data, meta, algos, f'{ROOT}/figures/M21_summary.csv')
plt = get_plt()
if plt and rows:
    for c in ['all_reduce', 'all_gather', 'reduce_scatter', 'broadcast', 'reduce', 'alltoall']:
        for met in ('bw', 'time'):
            cf = [(f'{n} GPU NVLS {s}', f'{c}_g{n}_nvls{s}') for n in (2, 4, 8) for s in ('on', 'off')]
            lineplot(plt, rows, cf, met, f'M21 {c} (median, 95% bootstrap CI over 5 launches)', f'{ROOT}/figures/M21_{c}_{met}.png')
    lineplot(plt, rows, [('default (NVLS on)', 'all_reduce_g8_nvlson'), ('Ring', 'allreduce_g8_algoRing'),
                         ('Tree', 'allreduce_g8_algoTree'), ('NVLS', 'allreduce_g8_algoNVLS')], 'bw',
             'M21 8-GPU all_reduce by forced algorithm', f'{ROOT}/figures/M21_algo_bw.png')
    lineplot(plt, rows, [(f'Ring/{p}', f'allreduce_g8_ring_{p}') for p in ('LL', 'LL128', 'Simple')], 'time',
             'M21 8-GPU all_reduce, Ring, by protocol', f'{ROOT}/figures/M21_proto_time.png')
    # NVLS on/off ratio of busbw (R2/R4)
    import collections
    by = {(r['cfg'], r['size']): r for r in rows}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in ['all_reduce', 'all_gather', 'reduce_scatter']:
        xs = sorted(s for (cfg, s) in by if cfg == f'{c}_g8_nvlson' and (f'{c}_g8_nvlsoff', s) in by)
        ax.plot(xs, [by[(f'{c}_g8_nvlson', s)]['oop_bw_med'] / by[(f'{c}_g8_nvlsoff', s)]['oop_bw_med'] for s in xs], marker='o', ms=3, label=c)
    ax.set_xscale('log', base=2); ax.axhline(1, color='k', lw=.5); ax.set_xlabel('message size (bytes)')
    ax.set_ylabel('busbw ratio NVLS on / off (8 GPU)'); ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f'{ROOT}/figures/M21_nvls_ratio.png', dpi=130)
