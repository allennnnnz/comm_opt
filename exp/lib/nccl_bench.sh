# Shared helpers for M21 / M23 (sourced by job scripts after $OUT exists). Written by S06 (2026-11-17).
# Plan line (TSV): cfg kind rep bin nnodes ppn gpulist plugin(0/1) envs("-" or "K=V K=V") dist(block|cyclic) nodes(all|a,b)
LIB=/home/allen0117/comm_opt/exp/lib
HPCX=/work/HPC_software/LMOD/nvidia/packages/hpc_sdk-24.7/Linux_x86_64/24.7/comm_libs/12.5/hpcx/latest
source /home/allen0117/comm_opt/exp/lib/nano5_env.sh
LD_BASE=$LD_LIBRARY_PATH
LAUNCH_MODE=""
SEG_SMALL_ARGS="-b 8 -e 64M -f 2 -n 2000 -w 5 -a 3"     # -a 3 = MAX across ranks (common.cu:121,357-359)
SEG_BIG_ARGS="-b 128M -e 8G -f 2 -n 20 -w 5 -a 3"
PROBE_ARGS="-b 8 -e 8G -f 2 -n 1 -w 0 -c 0 -a 3"        # one launch per config: algorithm/proto selection log (TUNING)

set_ld() {  # set_ld <plugin 0|1>  (venv NCCL always first; SHARP plugin dirs only when plugin=1; no hpcx module)
  if [[ $1 == 1 ]]; then
    export LD_LIBRARY_PATH="$NCCL_LIB_DIR:$HPCX/nccl_rdma_sharp_plugin/lib:$HPCX/sharp/lib:$HPCX/ucx/lib:$LD_BASE"
  else
    export LD_LIBRARY_PATH="$LD_BASE"
  fi
}
wrongcols() { grep -E '^ +[0-9]+ +[0-9]+ +float +(sum|none)' "$1" | awk '{print $9"/"$13}' | sort -u | tr '\n' ' '; }

# probe_launcher: first mode whose tiny all_reduce exits 0 and prints result rows; result in $LAUNCH_MODE, log in $OUT/launcher_probe.txt
probe_launcher() {  # probe_launcher <nnodes> <ppn>
  local nn=$1 ppn=$2 m
  set_ld 0
  export GPU_LIST=0,1,2,3,4,5,6,7
  for m in pmix_v4 pmi2 mpirun; do
    echo "### probe launcher $m" | tee -a "$OUT/launcher_probe.txt"
    NCCL_DEBUG=WARN timeout 300 $LIB/launch.sh $m $nn $ppn block all $NCCL_TESTS/all_reduce_perf -b 8 -e 1M -f 8 -n 5 -w 2 > "$OUT/launcher_probe_$m.log" 2>&1
    local rc=$?
    echo "    rc=$rc rows_wrong(oop/ip)=$(wrongcols "$OUT/launcher_probe_$m.log")" | tee -a "$OUT/launcher_probe.txt"
    if [[ $rc == 0 && -n $(wrongcols "$OUT/launcher_probe_$m.log") ]]; then LAUNCH_MODE=$m; echo "    -> use $m" | tee -a "$OUT/launcher_probe.txt"; return 0; fi
  done
  return 1
}

# run_one "<plan line>" <seq> ; writes $OUT/runs/<cfg>__<kind><rep>.<seg>.log.gz and appends to $OUT/runs.tsv (one row per segment, own exit code)
run_one() {
  local line=$1 seq=$2 cfg kind rep bin nn ppn gpus plugin envs dist nodes
  IFS=$'\t' read -r cfg kind rep bin nn ppn gpus plugin envs dist nodes <<< "$line"
  set_ld "$plugin"
  local envarr=() kv
  for kv in $envs; do [[ $kv != - ]] && envarr+=("$kv"); done
  local subsys=INIT,NET segs="small big"
  [[ $kind == probe ]] && { subsys=INIT,NET,TUNING; segs=probe; }
  local seg args base rc t0
  for seg in $segs; do
    case $seg in small) args=$SEG_SMALL_ARGS ;; big) args=$SEG_BIG_ARGS ;; probe) args=$PROBE_ARGS ;; esac
    # alltoall needs count >= 4*nranks floats; start at 512 B (min size we use for it)
    if [[ $bin == *alltoall_perf && $seg != big ]]; then args=${args/-b 8 /-b 512 }; fi
    [[ $bin == *all_reduce_perf ]] && args="$args -o sum"
    base="$OUT/runs/${cfg}__${kind}${rep}.${seg}"
    t0=$(date -Is)
    env ${envarr[@]+"${envarr[@]}"} NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=$subsys GPU_LIST=$gpus \
      timeout "${SEG_TIMEOUT:-600}" $LIB/launch.sh $LAUNCH_MODE $nn $ppn $dist $nodes $NCCL_TESTS/$bin $args > "$base.log" 2>&1
    rc=$?
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$seq" "$cfg" "$kind" "$rep" "$seg" "$rc" "$t0" "$(date -Is)" "$LAUNCH_MODE" "$nodes" "$gpus" "$envs plugin=$plugin args=$args" >> "$OUT/runs.tsv"
    echo "seq=$seq cfg=$cfg $kind$rep $seg rc=$rc wrong=$(wrongcols "$base.log")" | tee -a "$OUT/summary.txt"
    gzip -f "$base.log"
  done
}

# shuffle_plan <seed> <in> <out> : seeded shuffle, stdout-free
shuffle_plan() {
  python3 - "$1" "$2" "$3" <<'PY'
import random, sys
seed, src, dst = int(sys.argv[1]), sys.argv[2], sys.argv[3]
lines = [l.rstrip("\n") for l in open(src) if l.strip()]
random.Random(seed).shuffle(lines)
open(dst, "w").write("\n".join(lines) + "\n")
PY
}

# run_plan <plan file> <deadline_seconds> : run every line in order; skip (and record) lines once SECONDS > deadline
run_plan() {
  local plan=$1 deadline=$2 seq=0 line cfg
  while IFS= read -r line; do
    seq=$((seq+1))
    if (( SECONDS > deadline )); then
      echo "seq=$seq SKIPPED_TIME_BUDGET $line" | tee -a "$OUT/summary.txt"; continue
    fi
    run_one "$line" "$seq"
  done < "$plan"
}

# add_cfg <planfile> <cfg> <bin> <nn> <ppn> <gpus> <plugin> <envs|-> <dist> <nodes> : 5 reps + 1 probe
add_cfg() {
  local pf=$1 cfg=$2 bin=$3 nn=$4 ppn=$5 gpus=$6 plugin=$7 envs=$8 dist=$9 nodes=${10} r
  for r in 1 2 3 4 5; do printf '%s\trun\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cfg" "$r" "$bin" "$nn" "$ppn" "$gpus" "$plugin" "$envs" "$dist" "$nodes" >> "$pf"; done
  printf '%s\tprobe\t0\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cfg" "$bin" "$nn" "$ppn" "$gpus" "$plugin" "$envs" "$dist" "$nodes" >> "$pf"
}
