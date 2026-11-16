#!/usr/bin/env python3
"""M23 analysis: python analyze.py [results glob]. Output: figures/M23_summary.csv, figures/M23_q7_nodepairs.md, figures/M23_*.png (S06).
Algorithm column = algo/proto from the probe launch's TUNING log (what NCCL actually chose), not the requested NCCL_ALGO."""
import sys
sys.path.insert(0, '/home/allen0117/comm_opt/exp/lib')
from analyze_common import *

ROOT = '/home/allen0117/comm_opt'
pat = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/results/M23_internode_baseline/*_job*'
data, meta, algos = load_jobs(pat)
rows = summarize(data, meta, algos, f'{ROOT}/figures/M23_summary.csv')
by = {(r['job'], r['cfg'], r['size']): r for r in rows}
# Q7: same-group vs cross-group pair (H100 job), all_reduce default/ring, 8 B time and >=1 GB busbw
lines = ['# M23 Q7 node-pair comparison (median [95% CI] over 5 launches; out-of-place)', '',
         '| job | cfg | size | time_us | busbw GB/s | shared_flag | nodes |', '|---|---|---|---|---|---|---|']
for r in rows:
    if re.search(r'n2(same|cross|same2_nocross)_(ar_default|ar_ring)$', r['cfg']) and (r['size'] == 8 or r['size'] >= 1 << 30):
        lines.append(f"| {r['job']} | {r['cfg']} | {r['size']} | {r['oop_time_med']:.2f} [{r['oop_time_lo']:.2f}, {r['oop_time_hi']:.2f}] | "
                     f"{r['oop_bw_med']:.2f} [{r['oop_bw_lo']:.2f}, {r['oop_bw_hi']:.2f}] | {r['shared_flag']} | {r['nodes']} |")
open(f'{ROOT}/figures/M23_q7_nodepairs.md', 'w').write('\n'.join(lines) + '\n')
plt = get_plt()
if plt and rows:
    scopes = sorted({r['cfg'].split('_')[0] for r in rows})
    for met in ('bw', 'time'):
        for sc in scopes:
            cf = [(a, f'{sc}_{a}') for a in ('ar_default', 'ar_ring', 'ar_tree', 'ar_nvlstree', 'arP0_default', 'arP1_default', 'arP1_cdirect', 'arP1_cchain')]
            lineplot(plt, rows, cf, met, f'M23 {sc} all_reduce by algorithm / plugin', f'{ROOT}/figures/M23_{sc}_allreduce_{met}.png')
            cf = [(a, f'{sc}_{a}') for a in ('ag', 'rs', 'a2a', 'sr1', 'sr8', 'ar_default_gdroff', 'sr1_gdroff', 'sr8_gdroff')]
            lineplot(plt, rows, cf, met, f'M23 {sc} other collectives / sendrecv / GDR off', f'{ROOT}/figures/M23_{sc}_other_{met}.png')
