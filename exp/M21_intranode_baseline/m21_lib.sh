# M21 job body (sourced by node_a.sbatch / node_b.sbatch after $OUT exists and cotenants.sh was sourced).
# usage: m21_main <a|b> <deadline_seconds>
source /home/allen0117/comm_opt/exp/lib/nccl_bench.sh

m21_plan() {  # m21_plan <a|b> <planfile>
  local part=$1 pf=$2 c n v bin gpus
  : > "$pf"
  local colls_a="all_reduce all_gather reduce_scatter" colls_b="broadcast reduce alltoall"
  local colls; [[ $part == a ]] && colls=$colls_a || colls=$colls_b
  for c in $colls; do
    bin=${c}_perf
    for n in 2 4 8; do
      gpus=$(seq -s, 0 $((n-1)))
      add_cfg "$pf" "${c}_g${n}_nvlson"  $bin 1 $n "$gpus" 0 "-" block all
      add_cfg "$pf" "${c}_g${n}_nvlsoff" $bin 1 $n "$gpus" 0 "NCCL_NVLS_ENABLE=0" block all
    done
  done
  if [[ $part == a ]]; then
    for v in Ring Tree NVLS; do
      add_cfg "$pf" "allreduce_g8_algo${v}" all_reduce_perf 1 8 0,1,2,3,4,5,6,7 0 "NCCL_ALGO=$v" block all
    done
    for v in LL LL128 Simple; do
      add_cfg "$pf" "allreduce_g8_ring_${v}" all_reduce_perf 1 8 0,1,2,3,4,5,6,7 0 "NCCL_ALGO=Ring NCCL_PROTO=$v" block all
    done
  fi
}

m21_warmup() {  # default: 30 s of all_reduce 64 MB on all 8 GPUs (M21 card, Opus addendum); length may be changed after M20 P7
  local secs=${WARMUP_SECS:-30}
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu --format=csv > "$OUT/warmup_clocks_before.csv" 2>&1
  export GPU_LIST=0,1,2,3,4,5,6,7; set_ld 0
  local t0=$SECONDS
  # timeout kills the srun step after $secs (rc 124 is the expected "finished" code)
  NCCL_DEBUG=WARN timeout "$secs" $LIB/launch.sh $LAUNCH_MODE 1 8 block all $NCCL_TESTS/all_reduce_perf -b 64M -e 64M -n 100000000 -w 5 -c 0 -a 3 > "$OUT/warmup.log" 2>&1
  local wrc=$?
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu --format=csv > "$OUT/warmup_clocks_after.csv" 2>&1
  echo "warmup requested=${secs}s elapsed=$((SECONDS-t0))s rc=$wrc" | tee -a "$OUT/summary.txt"
}

m21_main() {
  local part=$1 deadline=$2
  mkdir -p "$OUT/runs"
  {
    echo "job_id=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION node=$(hostname) nodelist=$SLURM_JOB_NODELIST part=$part start=$(date -Is)"
    echo "git_commit=$(git -C /home/allen0117/comm_opt rev-parse HEAD) dirty=$(git -C /home/allen0117/comm_opt status --porcelain | wc -l)"
    module list 2>&1
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.sm,clocks.max.sm,temperature.gpu --format=csv
    nvidia-smi topo -m
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
    echo "NCCL_TESTS=$NCCL_TESTS nccl-tests commit $(git -C /home/allen0117/comm_opt/third_party/nccl-tests rev-parse --short HEAD)"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "SEG_SMALL_ARGS=$SEG_SMALL_ARGS  SEG_BIG_ARGS=$SEG_BIG_ARGS  PROBE_ARGS=$PROBE_ARGS"
  } > "$OUT/job_env.txt" 2>&1
  others=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
  if [[ -n $others ]]; then echo "GPU processes found before measuring; aborting: $others" | tee -a "$OUT/summary.txt"; return 1; fi
  NNODES=1 NPROCS_PER_NODE=8
  probe_launcher 1 8 || { echo "no launcher works" | tee -a "$OUT/summary.txt"; return 1; }
  m21_warmup
  m21_plan "$part" "$OUT/plan_unshuffled.tsv"
  shuffle_plan 20261117 "$OUT/plan_unshuffled.tsv" "$OUT/plan_order.tsv"   # actual execution order = plan_order.tsv
  echo "configs=$(cut -f1 $OUT/plan_unshuffled.tsv | sort -u | wc -l) launches=$(wc -l < $OUT/plan_order.tsv) seed=20261117" | tee -a "$OUT/summary.txt"
  SEG_TIMEOUT=300 run_plan "$OUT/plan_order.tsv" "$deadline"
  echo "end=$(date -Is)" >> "$OUT/job_env.txt"
}
