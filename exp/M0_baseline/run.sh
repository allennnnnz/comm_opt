#!/usr/bin/env bash
# M0 baseline: 6 collectives x 8 B..4 GiB x {2,4,8} GPUs, one MPI rank per GPU.
# Usage: exp/M0_baseline/run.sh            (refuses to run if any GPU process exists)
#        DRY_RUN=1 exp/M0_baseline/run.sh  (print commands only)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=$ROOT/third_party/nccl-tests/build
OUT=$ROOT/results/M0_baseline/$(date +%Y%m%d_%H%M%S)
COLLS=(all_reduce all_gather reduce_scatter alltoall broadcast sendrecv)
declare -A DEVS=([2]="0,1" [4]="0,1,2,3" [8]="0,1,2,3,4,5,6,7")
ARGS=(-b 8 -e 4G -f 2 -g 1 -n 20 -w 5 -a 1 -U 1 -I 1)

if [[ -z "${DRY_RUN:-}" && -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPUs have running processes; not measuring." >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 1
fi


if [[ -z "${DRY_RUN:-}" ]]; then
  mkdir -p "$OUT"
  { date -Is; hostname; nvidia-smi --query-gpu=index,name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu --format=csv
    nvidia-smi topo -m; git -C "$ROOT/third_party/nccl-tests" log -1 --format='nccl-tests %h'
    ls -l /usr/lib/x86_64-linux-gnu/libnccl.so.2*; } > "$OUT/env.txt" 2>&1
fi

for ng in 2 4 8; do
  for coll in "${COLLS[@]}"; do
    tag=${coll}_g${ng}
    echo "[$(date +%T)] $tag" >&2
    cmd=(mpirun -np "$ng" --bind-to none --mca btl self,vader
      -x CUDA_VISIBLE_DEVICES="${DEVS[$ng]}" -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
      -x NCCL_DEBUG_FILE="$OUT/$tag.nccl.%h.%p.log"
      "$BIN/${coll}_perf" "${ARGS[@]}" -J "$OUT/$tag.json")
    if [[ -n "${DRY_RUN:-}" ]]; then echo "${cmd[*]}"; continue; fi
    "${cmd[@]}" > "$OUT/$tag.log" 2>&1 || echo "  FAILED: $tag (see $OUT/$tag.log)" >&2
  done
done
echo "done: $OUT" >&2
