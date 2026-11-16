# M20 shared helpers (sourced by node.sbatch and sharp.sbatch).
# launch_mode is chosen by probe_launcher; launch runs a binary with NPROCS ranks (one GPU each).
HPCX=/work/HPC_software/LMOD/nvidia/packages/hpc_sdk-24.7/Linux_x86_64/24.7/comm_libs/12.5/hpcx/latest
LAUNCH_MODE=""
wrongcols() {  # print "#wrong" column values (out-of-place, in-place) seen in a nccl-tests log
  grep -E '^ +[0-9]+ +[0-9]+ +float +(sum|none)' "$1" | awk '{print $9"/"$13}' | sort -u | tr '\n' ' '
}
launch() {  # launch <mode> <bin> <args...>  ; uses NNODES, NPROCS_PER_NODE
  local mode=$1; shift
  case $mode in
    pmix_v4) srun --mpi=pmix_v4 --nodes=$NNODES --ntasks-per-node=$NPROCS_PER_NODE --gpus-per-node=8 "$@" ;;
    pmi2)    srun --mpi=pmi2    --nodes=$NNODES --ntasks-per-node=$NPROCS_PER_NODE --gpus-per-node=8 "$@" ;;
    mpirun)
      local xs=() v
      for v in LD_LIBRARY_PATH NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_ALGO NCCL_NVLS_ENABLE NCCL_COLLNET_ENABLE SHARP_COLL_LOG_LEVEL SHARP_COLL_ENABLE_SAT; do
        [[ -n ${!v+x} ]] && xs+=(-x $v)
      done
      mpirun -np $((NNODES*NPROCS_PER_NODE)) --map-by ppr:${NPROCS_PER_NODE}:node "${xs[@]}" "$@" ;;
  esac
}
# run_cfg <name> <timeout_s> <bin> <args...> : uses $LAUNCH_MODE; writes $OUT/<name>.log and appends to $OUT/summary.txt
run_cfg() {
  local name=$1 to=$2; shift 2
  local t0; t0=$(date -Is)
  timeout $to bash -c "$(declare -f launch); NNODES=$NNODES NPROCS_PER_NODE=$NPROCS_PER_NODE; launch $LAUNCH_MODE \"\$@\"" _ "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc start=$t0 end=$(date -Is) wrong(oop/ip)=$(wrongcols "$OUT/$name.log") mode=$LAUNCH_MODE" | tee -a "$OUT/summary.txt"
  return $rc
}
# probe_launcher <name> <timeout_s> <bin> <args...> : first mode that rc==0 AND has result rows wins; first run's log is the config's log
probe_launcher() {
  local name=$1 to=$2; shift 2
  local m
  for m in pmix_v4 pmi2 mpirun; do
    LAUNCH_MODE=$m
    echo "### probe launcher $m for $name" | tee -a "$OUT/attempts.txt"
    if run_cfg "$name" $to "$@" && [[ -n $(wrongcols "$OUT/$name.log") ]]; then
      echo "    -> $m works" | tee -a "$OUT/attempts.txt"; return 0
    fi
    echo "    -> $m failed (see $name.log; kept as $name.failed_$m.log)" | tee -a "$OUT/attempts.txt"
    mv "$OUT/$name.log" "$OUT/$name.failed_$m.log"
  done
  LAUNCH_MODE=""
  return 1
}

# co-tenancy record at start and exit (D054); read at run time, so queued jobs also get it
source /home/allen0117/comm_opt/exp/lib/cotenants.sh
