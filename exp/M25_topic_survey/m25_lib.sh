#!/bin/bash
# M25 shared library. Sourced by job*.sbatch and validate.sbatch AT RUN TIME (so it can be fixed while a job is queued).
# Control variables (all optional):
#   M25_MODE=full|smoke   M25_NGPU=8|2   M25_BLOCKS="B4 B1 B2 B3"   M25_TAG=<suffix for result dir>
#   M25_OUTBASE=<results dir>  M25_GLOBAL_MIN=112  M25_LIM_B4/B1/B2/B3=<minutes>  M25_B2_EAGER=0|1
# Entry point: m25_main.  B0 (env record, GPU check, warm-up, GPU monitor) always runs first.
set -uo pipefail
ROOT=/home/allen0117/comm_opt
EXPD=$ROOT/exp/M25_topic_survey
ENVROOT=/work/allen0117/comm_opt_env
M25_MODE=${M25_MODE:-full}
NGPU=${M25_NGPU:-8}
M25_BLOCKS=${M25_BLOCKS:-"B4 B1 B2 B3"}
M25_OUTBASE=${M25_OUTBASE:-$ROOT/results/M25_topic_survey}
source $ROOT/exp/lib/nccl_bench.sh      # nano5_env.sh (modules, venv, NCCL path, NCCL_TESTS), shuffle_plan, wrongcols
CACHE=$ENVROOT/cache
mkdir -p $CACHE/vllm $CACHE/inductor $CACHE/triton $CACHE/xdg $CACHE/flashinfer
export VLLM_CACHE_ROOT=$CACHE/vllm TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton
export XDG_CACHE_HOME=$CACHE/xdg FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
MODELS=$ENVROOT/models
SEED=20261117

if [[ $M25_MODE == smoke ]]; then
  GLOBAL_MIN=${M25_GLOBAL_MIN:-36}; LIM_B4=${M25_LIM_B4:-3}; LIM_B1=${M25_LIM_B1:-6}; LIM_B2=${M25_LIM_B2:-20}; LIM_B3=${M25_LIM_B3:-4}
  WARM_S=5; REPS_B3=1; REPS_B1=1
else
  GLOBAL_MIN=${M25_GLOBAL_MIN:-112}; LIM_B4=${M25_LIM_B4:-8}; LIM_B1=${M25_LIM_B1:-30}; LIM_B2=${M25_LIM_B2:-40}; LIM_B3=${M25_LIM_B3:-25}
  WARM_S=60; REPS_B3=2; REPS_B1=2
fi

# ---------------------------------------------------------------- time budget helpers
elapsed() { echo $(( SECONDS - M25_T0 )); }
blk_begin() {  # blk_begin <name> <limit_minutes>
  BLK=$1; BLK_START=$(elapsed); BLK_END=$(( BLK_START + $2*60 ))
  (( BLK_END > GLOBAL_MIN*60 )) && BLK_END=$(( GLOBAL_MIN*60 ))
  BLK_RUN=0; BLK_FAIL=0; BLK_SKIP=0; BLK_T0_ISO=$(date -Is)
  echo "=== $BLK begin elapsed=${BLK_START}s block_end=${BLK_END}s" | tee -a $OUT/summary.txt
}
blk_end() {  # blk_end <status>
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$BLK" "$BLK_T0_ISO" "$(date -Is)" "$BLK_START" "$(elapsed)" "$1" "run=$BLK_RUN fail=$BLK_FAIL" "skipped=$BLK_SKIP" >> $OUT/blocks.tsv
  echo "=== $BLK end status=$1 run=$BLK_RUN fail=$BLK_FAIL skipped=$BLK_SKIP elapsed=$(elapsed)s" | tee -a $OUT/summary.txt
}
can_run() { (( $(elapsed) + $1 <= BLK_END )); }   # can_run <estimated_seconds>
cap() { local rem=$(( BLK_END - $(elapsed) )); (( rem < 20 )) && rem=20; (( rem < $1 )) && echo $rem || echo $1; }   # cap <want_s>
skip_rec() { BLK_SKIP=$((BLK_SKIP+1)); echo "SKIPPED(time) $BLK $*" | tee -a $OUT/summary.txt; }
rec_rc() { BLK_RUN=$((BLK_RUN+1)); (( $1 != 0 )) && BLK_FAIL=$((BLK_FAIL+1)); return 0; }

m25_mpirun() {  # m25_mpirun <np> <cmd...> ; exports names listed in M25_XENV in addition to the usual ones
  local np=$1; shift; local xs=() v
  for v in LD_LIBRARY_PATH NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE CUDA_VISIBLE_DEVICES ${M25_XNAMES:-}; do
    [[ -n $v && -n ${!v+x} ]] && xs+=(-x "$v")
  done
  # job has 1 Slurm task -> --oversubscribe (validation 497760: "not enough slots").
  # Opus 2026-11-25: main job 497945 (8 ranks, hgpn01) died in MPI_Allgather with UCX "wireup.c:408 no remote ep address" ->
  # single node: bypass UCX (pml ob1 + shared-memory btl); MPI only carries nccl-tests' bootstrap, not the measured traffic.
  # </dev/null: mpirun read the rest of B3's plan file from stdin, so B3 stopped after its first entry.
  mpirun -np "$np" --oversubscribe --bind-to none --mca pml ob1 --mca btl self,vader --mca osc ^ucx "${xs[@]}" "$@" < /dev/null
}
with_env() {  # with_env "K=V K=V" cmd... : run cmd in a subshell with those variables exported (and passed to mpirun)
  local envs=$1; shift
  ( names=""; for kv in $envs; do [[ $kv == - ]] && continue; export "$kv"; names="$names ${kv%%=*}"; done; export M25_XNAMES="$names"; "$@" )
}
gpu_clean() {  # wait until no compute process is left on the GPUs; kill our own leftovers if needed
  local i
  for i in 1 2 3 4 5 6; do
    [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]] && return 0
    sleep 5
  done
  echo "leftover GPU processes after wait: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')" >> $OUT/summary.txt
  pkill -u "$USER" -f 'VLLM::' 2>/dev/null; pkill -u "$USER" -f vllm_bench.py 2>/dev/null; sleep 5
}

# ---------------------------------------------------------------- B0
b0() {
  blk_begin B0 8
  {
    echo "job_id=${SLURM_JOB_ID:-none} job_name=${SLURM_JOB_NAME:-none} partition=${SLURM_JOB_PARTITION:-none}"
    echo "node=$(hostname) start=$(date -Is) mode=$M25_MODE ngpu=$NGPU blocks=$M25_BLOCKS global_min=$GLOBAL_MIN"
    echo "git_commit=$(git -C $ROOT rev-parse HEAD) dirty=$(git -C $ROOT status --porcelain | wc -l)"
    module list 2>&1
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.sm,clocks.max.sm,temperature.gpu --format=csv
    nvidia-smi topo -m
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    python -c "import sys,torch;print('python',sys.executable);print('torch',torch.__version__,'cuda_avail',torch.cuda.is_available(),'devices',torch.cuda.device_count(),'nccl',torch.cuda.nccl.version())" 2>&1
    python -c "import vllm;print('vllm',vllm.__version__)" 2>&1 | tail -1
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  } > $OUT/job_env.txt 2>&1
  local others; others=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
  if [[ -n $others ]]; then
    echo "GPU processes found before measuring: $others" | tee -a $OUT/job_env.txt $OUT/summary.txt
    if [[ $M25_MODE == full ]]; then blk_end ABORT; echo "ABORT" >> $OUT/summary.txt; return 1; fi
  fi
  nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active \
    --format=csv -lms 500 > $OUT/gpu_monitor.csv 2> $OUT/gpu_monitor.err &
  MON_PID=$!
  # warm-up: calibrate -n so that all_reduce 64 MB runs about WARM_S seconds
  mkdir -p $OUT/b0
  local bin=$NCCL_TESTS/all_reduce_perf a="-b 64M -e 64M -g 1 -c 0"
  NCCL_DEBUG=WARN timeout 120 bash -c "$(declare -f m25_mpirun); m25_mpirun $NGPU $bin $a -n 200 -w 20" > $OUT/b0/calibration.log 2>&1
  local cal niter
  cal=$(grep -E '^ +[0-9]+ +[0-9]+ +float +sum' $OUT/b0/calibration.log | awk '{print $6" "$10}')
  niter=$(echo "$cal" | awk -v s=$WARM_S '{t=$1+$2; if(t>0) printf "%d", s*1e6/t; else print 100000}')
  [[ -z $niter || $niter -lt 1 ]] && niter=100000
  echo "warmup calibration oop_us ip_us=$cal n=$niter" | tee -a $OUT/summary.txt
  local t0=$SECONDS
  NCCL_DEBUG=WARN timeout $((WARM_S*3+60)) bash -c "$(declare -f m25_mpirun); m25_mpirun $NGPU $bin $a -n $niter -w 5" > $OUT/b0/warmup.log 2>&1
  local rc=$?
  echo "warmup rc=$rc seconds=$((SECONDS-t0))" | tee -a $OUT/summary.txt
  rec_rc $rc
  # thermal gate (Opus 2026-11-25; job 497945 hgpn01 GPU 7 thermally throttled to ~345 MHz under B1 load): 8-GPU GEMM burn
  local bs=${M25_BURN_S:-120} ts i trc bpids=()
  ts=$(date '+%Y/%m/%d %H:%M:%S')
  for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$i timeout $((bs+90)) python -c "
import torch,time
a=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16);b=a.clone();c=torch.empty_like(a);n=0;t=time.time()
while time.time()-t<$bs:
    for _ in range(20): torch.mm(a,b,out=c)
    torch.cuda.synchronize(); n+=20
print('gemms',n,'tflops',n*2*8192**3/(time.time()-t)/1e12)" > $OUT/b0/burn_gpu$i.log 2>&1 &
    bpids+=($!)
  done
  wait "${bpids[@]}"   # NOT a bare wait: that also waits for the GPU monitor and hung job 498237 (rescued by hand)
  sleep 2
  python $EXPD/thermal_check.py $OUT/gpu_monitor.csv "$ts" $NGPU > $OUT/b0/thermal.txt 2>&1; trc=$?
  for i in $(seq 0 $((NGPU-1))); do echo "burn gpu$i: $(tail -1 $OUT/b0/burn_gpu$i.log)"; done | tee -a $OUT/summary.txt
  echo "thermal gate rc=$trc: $(tail -1 $OUT/b0/thermal.txt)" | tee -a $OUT/summary.txt
  cat $OUT/b0/thermal.txt >> $OUT/summary.txt
  if [[ $trc == 3 && $M25_MODE == full ]]; then
    echo "ABORT: thermally limited GPU on $(hostname); resubmit with --exclude=$(hostname)" | tee -a $OUT/summary.txt
    blk_end ABORT_THERMAL; return 1
  fi
  blk_end $([[ $rc == 0 ]] && echo OK || echo WARN)
  return 0
}

# ---------------------------------------------------------------- B4  NVLS crash reproduction
b4() {
  blk_begin B4 $LIM_B4
  mkdir -p $OUT/b4; local B=$OUT/b4
  local bin=$NCCL_TESTS/all_reduce_perf
  local base_args="-b 8 -e 64M -f 4 -g 1"; [[ $M25_MODE == smoke ]] && base_args="-b 8 -e 4M -f 8 -g 1"
  local big_args="-b 8 -e 256M -f 4 -g 1";  [[ $M25_MODE == smoke ]] && big_args="-b 8 -e 8M -f 8 -g 1"
  # name | extra env | args.  a1,a2 = exact M20 nvls_a_default repeat (NCCL 2.29.7, mpirun, INFO INIT,ENV,NVLS); e256 = N2;
  # v1/v2 = diagnostic variants (multicast group is created at nvls.cc:257 cuMulticastAddDevice; both shrink the group)
  local plan=("b4_a1|-|$base_args" "b4_a2|-|$base_args" "b4_e256|-|$big_args"
              "b4_v1_nvlsch4|NCCL_NVLS_NCHANNELS=4|$base_args" "b4_v2_chunk16k|NCCL_NVLS_CHUNKSIZE=16384|$base_args")
  local p name envs args rc
  for p in "${plan[@]}"; do
    IFS='|' read -r name envs args <<< "$p"
    can_run 60 || { skip_rec $name; continue; }
    with_env "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,NVLS $envs" \
      timeout $(cap 150) bash -c "$(declare -f m25_mpirun); m25_mpirun $NGPU $bin $args" > $B/$name.log 2>&1
    rc=$?
    local f999; f999=$(grep -c "Cuda failure 999" $B/$name.log)
    printf '%s\t%s\t%s\trc=%s\tcuda999_lines=%s\twrong=%s\n' "$name" "$envs" "$args" "$rc" "$f999" "$(wrongcols $B/$name.log)" | tee -a $B/summary.tsv $OUT/summary.txt
    rec_rc $rc
    gzip -f $B/$name.log
    gpu_clean
  done
  blk_end DONE
}

# ---------------------------------------------------------------- B1  collective + GEMM concurrency
declare -A B1ENV=(
  [default]="-" [ring]="NCCL_ALGO=Ring"
  [ring_cta4]="NCCL_ALGO=Ring NCCL_MIN_CTAS=4 NCCL_MAX_CTAS=4" [ring_cta8]="NCCL_ALGO=Ring NCCL_MIN_CTAS=8 NCCL_MAX_CTAS=8"
  [ring_cta16]="NCCL_ALGO=Ring NCCL_MIN_CTAS=16 NCCL_MAX_CTAS=16" [ring_cta32]="NCCL_ALGO=Ring NCCL_MIN_CTAS=32 NCCL_MAX_CTAS=32"
  [nvls]="NCCL_ALGO=NVLS" [nvls_nch4]="NCCL_ALGO=NVLS NCCL_NVLS_NCHANNELS=4" [nvls_nch8]="NCCL_ALGO=NVLS NCCL_NVLS_NCHANNELS=8"
  [efficiency]="NCCL_CTA_POLICY=EFFICIENCY")
b1_torchrun() {  # b1_torchrun <timeout_s> <log> <args...> ; uses current env
  local to=$1 log=$2; shift 2
  timeout $to torchrun --standalone --nproc-per-node $NGPU $EXPD/overlap_bench.py "$@" > $log 2>&1
}
b1() {
  blk_begin B1 $LIM_B1
  mkdir -p $OUT/b1/nccl_logs; local B=$OUT/b1
  local cfgs="default ring ring_cta4 ring_cta8 ring_cta16 ring_cta32 nvls nvls_nch4 nvls_nch8 efficiency" cells=""
  if [[ $M25_MODE == smoke ]]; then
    cfgs="default ring_cta4"; cells="all_reduce:4:8192,reduce_scatter:32:4096,all_gather:256:8192"
  fi
  # calibration launch (not a measurement): default NCCL settings
  local t0=$SECONDS rc
  b1_torchrun $(cap 300) $B/calib_launch.log --calib $B/calib.json; rc=$?
  echo "B1 calibration rc=$rc seconds=$((SECONDS-t0))" | tee -a $OUT/summary.txt
  rec_rc $rc
  if [[ ! -s $B/calib.json ]]; then echo "no calib.json; B1 aborted" | tee -a $OUT/summary.txt; blk_end FAIL_CALIB; return 0; fi
  # plan: every cfg x REPS_B1 launches, shuffled with the fixed seed; actual order recorded
  : > $B/plan_unshuffled.tsv; local c r
  for c in $cfgs; do for r in $(seq 1 $REPS_B1); do printf '%s\t%s\n' $c $r >> $B/plan_unshuffled.tsv; done; done
  shuffle_plan $SEED $B/plan_unshuffled.tsv $B/plan_order.tsv
  local seq=0 cfg launch envs d
  printf 'seq\tcfg\tlaunch\trc\tseconds\tstart\n' > $B/launches.tsv
  while IFS=$'\t' read -r cfg launch; do
    seq=$((seq+1))
    can_run 100 || { skip_rec "$cfg launch$launch"; continue; }
    envs=${B1ENV[$cfg]}; d=$B/nccl_logs/${cfg}_L${launch}; mkdir -p $d
    t0=$SECONDS
    with_env "$envs NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV NCCL_DEBUG_FILE=$d/nccl_%h_%p.log" \
      b1_torchrun $(cap 240) $B/launch_${cfg}_L${launch}.log --calib-in $B/calib.json --out $B/overlap.jsonl --cfg $cfg --launch $launch ${cells:+--cells $cells} --reps 3
    rc=$?
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' $seq $cfg $launch $rc $((SECONDS-t0)) "$(date -Is)" >> $B/launches.tsv
    echo "B1 seq=$seq cfg=$cfg launch=$launch rc=$rc seconds=$((SECONDS-t0))" | tee -a $OUT/summary.txt
    rec_rc $rc
    tar -czf $d.tar.gz -C $B/nccl_logs ${cfg}_L${launch} 2>/dev/null && rm -rf $d
    gzip -f $B/launch_${cfg}_L${launch}.log
    gpu_clean
  done < $B/plan_order.tsv
  blk_end DONE
}

# ---------------------------------------------------------------- B2  vLLM
b2_engine() {  # b2_engine <name> <model> <tp> <ep> <eager> <maxlen> <decode> <prefill> <prof> <maxbatched> [envs|-] [timeout_s]
  local name=$1 model=$2 tp=$3 ep=$4 eager=$5 maxlen=$6 dec=$7 pre=$8 prof=$9 mb=${10} envs=${11:--} to=${12:-720}
  local d=$OUT/b2/$name; mkdir -p $d
  can_run 300 || { skip_rec "engine $name"; return; }
  local t0=$SECONDS rc envv=()
  [[ $envs != - ]] && read -r -a envv <<< "$envs"
  echo "$envs" > $d/extra_env.txt
  env "${envv[@]}" timeout $(cap $to) python $EXPD/vllm_bench.py --name $name --model $MODELS/$model --tp $tp --ep $ep --eager $eager \
    --maxlen $maxlen --decode "$dec" --prefill "$pre" --prof "$prof" ${mb:+--max-batched $mb} --outdir $d --trace-root $ENVROOT/traces/$(basename $OUT) \
    > $d/engine.log 2>&1
  rc=$?
  printf '%s\t%s\t%s\n' $name $rc $((SECONDS-t0)) >> $OUT/b2/engines.tsv
  echo "B2 engine=$name rc=$rc seconds=$((SECONDS-t0))" | tee -a $OUT/summary.txt
  rec_rc $rc
  grep -h "all-reduce backends" $d/engine.log | head -2 > $d/allreduce_backend.txt
  gzip -f $d/engine.log
  gpu_clean
}
b2() {
  blk_begin B2 $LIM_B2
  # Opus 2026-11-25: vLLM 0.32 dispatches FLASHINFER all-reduce first; its JIT needs nvcc >= 12.8 (--compress-mode=size,
  # flashinfer/jit/core.py:693) but this cluster has CUDA <= 12.6 and driver 12.4 -> validation 497896 failed. Use vLLM's other
  # backends (CUSTOM, SYMM_MEM, PYNCCL). gcc module gives JIT/Triton a full host compiler.
  export VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0 VLLM_USE_FLASHINFER_SAMPLER=0   # validation 497916: FlashInfer sampling JIT also fails (same nvcc>=12.8 need)
  module load gcc/12.5.0 2>/dev/null || true
  mkdir -p $OUT/b2 $ENVROOT/traces/$(basename $OUT)
  printf 'engine\trc\tseconds\n' > $OUT/b2/engines.tsv
  if [[ $M25_MODE == smoke ]]; then
    b2_engine q7_tp2        Qwen2.5-7B-Instruct 2 0 0 2048 "1,8" "1024" "1,8" ""
    b2_engine q7_tp2_eager  Qwen2.5-7B-Instruct 2 0 1 2048 ""    ""     "1"   ""
    b2_engine q30_tp2       Qwen3-30B-A3B       2 0 0 2048 "1"   ""     "1"   ""
    b2_engine q30_tp2_ep    Qwen3-30B-A3B       2 1 0 2048 "1"   ""     "1"   ""
  else
    # Opus 2026-11-25: in 497945 q72_tp8 hung for 10 min right after torch.compile (first 17408-token dummy run, ~285 MB
    # all-reduce; q72_tp4 and q30_tp8 with 16384x2048 were fine). Same engine again with NCCL INIT/NVLS logging, then two
    # diagnostic variants that isolate (a) message size (8192-token batches) and (b) NVLS (disabled).
    b2_engine q72_tp8        Qwen2.5-72B-Instruct 8 0 0 17408 "1,8,32,128" "1024,4096,16384" "1,8,32,128" 17408 \
      "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NVLS,ENV" 480
    b2_engine q72_tp8_mb8k   Qwen2.5-72B-Instruct 8 0 0 17408 "1,8,32,128" "1024,4096,16384" "1,8,32,128" 8192 \
      "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NVLS,ENV" 420
    b2_engine q72_tp8_nonvls Qwen2.5-72B-Instruct 8 0 0 17408 "1,8,32,128" "1024,4096,16384" "1,8,32,128" 17408 \
      "NCCL_NVLS_ENABLE=0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV" 420
    b2_engine q72_tp4     Qwen2.5-72B-Instruct 4 0 0 17408 "1,32"       "1024,4096,16384" "1,32"       17408
    b2_engine q30_tp8     Qwen3-30B-A3B        8 0 0 2048  "1,32,128"   ""               "1,32,128"   ""
    b2_engine q30_tp8_ep  Qwen3-30B-A3B        8 1 0 2048  "1,32,128"   ""               "1,32,128"   ""
    if [[ ${M25_B2_EAGER:-0} == 1 ]]; then   # profile-only engines in eager mode (used when CUDA graph traces lack kernels)
      b2_engine q72_tp8_eager    Qwen2.5-72B-Instruct 8 0 1 2048 "" "" "1,8,32,128" ""
      b2_engine q30_tp8_eager    Qwen3-30B-A3B        8 0 1 2048 "" "" "1,32,128" ""
      b2_engine q30_tp8_ep_eager Qwen3-30B-A3B        8 1 1 2048 "" "" "1,32,128" ""
    fi
  fi
  blk_end DONE
}

# ---------------------------------------------------------------- B3  nccl-tests curves
b3() {
  blk_begin B3 $LIM_B3
  mkdir -p $OUT/b3/runs; local B=$OUT/b3
  local small="-b 8 -e 64M -f 2 -n 200 -w 5 -a 3" big="-b 128M -e 1G -f 2 -n 20 -w 5 -a 3" probe="-b 8 -e 1G -f 2 -n 1 -w 0 -c 0 -a 3"
  if [[ $M25_MODE == smoke ]]; then small="-b 8 -e 1M -f 8 -n 20 -w 2 -a 3"; big="-b 128M -e 256M -f 2 -n 5 -w 2 -a 3"; probe="-b 8 -e 256M -f 8 -n 1 -w 0 -c 0 -a 3"; fi
  # plan columns: cfg kind rep bin np cvd envs
  local pf=$B/plan_unshuffled.tsv; : > $pf
  add3() {  # add3 cfg bin np cvd envs
    local r; for r in $(seq 1 $REPS_B3); do printf '%s\trun\t%s\t%s\t%s\t%s\t%s\n' $1 $r $2 $3 $4 "$5" >> $pf; done
    printf '%s\tprobe\t0\t%s\t%s\t%s\t%s\n' $1 $2 $3 $4 "$5" >> $pf
  }
  if [[ $M25_MODE == smoke ]]; then
    add3 ar_default all_reduce_perf $NGPU - "-"
    add3 rs_default reduce_scatter_perf $NGPU - "-"
    add3 a2a_default alltoall_perf $NGPU - "-"
    add3 sr_default sendrecv_perf 2 - "-"
  else
    add3 ar_default      all_reduce_perf 8 - "-"
    add3 ar_ring_ll      all_reduce_perf 8 - "NCCL_ALGO=Ring NCCL_PROTO=LL"
    add3 ar_ring_ll128   all_reduce_perf 8 - "NCCL_ALGO=Ring NCCL_PROTO=LL128"
    add3 ar_ring_simple  all_reduce_perf 8 - "NCCL_ALGO=Ring NCCL_PROTO=Simple"
    add3 ar_tree         all_reduce_perf 8 - "NCCL_ALGO=Tree"
    add3 ar_nvls         all_reduce_perf 8 - "NCCL_ALGO=NVLS"
    add3 ar_nvls_off     all_reduce_perf 8 - "NCCL_NVLS_ENABLE=0"
    add3 rs_default      reduce_scatter_perf 8 - "-"
    add3 rs_nvls_off     reduce_scatter_perf 8 - "NCCL_NVLS_ENABLE=0"
    add3 ag_default      all_gather_perf 8 - "-"
    add3 ag_nvls_off     all_gather_perf 8 - "NCCL_NVLS_ENABLE=0"
    add3 a2a_default     alltoall_perf 8 - "-"
    add3 sr_default      sendrecv_perf 2 0,1 "-"
    add3 sr_memcpy       sendrecv_perf 2 0,1 "NCCL_P2P_USE_CUDA_MEMCPY=1"
    add3 ar4_default     all_reduce_perf 4 0,1,2,3 "-"
    add3 ar2_default     all_reduce_perf 2 0,1 "-"
  fi
  shuffle_plan $((SEED+1)) $pf $B/plan_order.tsv
  printf 'seq\tcfg\tkind\trep\tseg\trc\tstart\tend\tenvs\targs\n' > $B/runs.tsv
  local seq=0 cfg kind rep bin np cvd envs seg args seglist base rc t0 sub
  while IFS=$'\t' read -r cfg kind rep bin np cvd envs; do
    seq=$((seq+1))
    can_run 70 || { skip_rec "seq=$seq $cfg $kind$rep"; continue; }
    seglist="small big"; [[ $kind == probe ]] && seglist=probe
    sub=INIT; [[ $kind == probe ]] && sub=INIT,TUNING
    for seg in $seglist; do
      case $seg in small) args=$small ;; big) args=$big ;; probe) args=$probe ;; esac
      [[ $bin == alltoall_perf && $seg != big ]] && args=${args/-b 8 /-b 512 }
      base=$B/runs/${cfg}__${kind}${rep}.${seg}; t0=$(date -Is)
      local cv=""; [[ $cvd != - ]] && cv="CUDA_VISIBLE_DEVICES=$cvd"
      with_env "$envs $cv NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=$sub" \
        timeout $(cap 200) bash -c "$(declare -f m25_mpirun); m25_mpirun $np $NCCL_TESTS/$bin $args" > $base.log 2>&1
      rc=$?
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $seq $cfg $kind $rep $seg $rc "$t0" "$(date -Is)" "$envs $cv" "$args" >> $B/runs.tsv
      echo "B3 seq=$seq cfg=$cfg $kind$rep $seg rc=$rc wrong=$(wrongcols $base.log)" | tee -a $OUT/summary.txt
      rec_rc $rc; gzip -f $base.log
    done
    gpu_clean
  done < $B/plan_order.tsv
  blk_end DONE
}

# ---------------------------------------------------------------- finish
m25_cleanup() {
  [[ -n ${MON_PID:-} ]] && kill $MON_PID 2>/dev/null
  gzip -f $OUT/gpu_monitor.csv 2>/dev/null
  cotenants_record end
}
m25_main() {
  M25_T0=$SECONDS
  OUT=$M25_OUTBASE/$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}${M25_TAG:+_$M25_TAG}
  mkdir -p $OUT; : > $OUT/blocks.tsv; : > $OUT/summary.txt
  printf 'block\tstart\tend\tstart_s\tend_s\tstatus\tcounts\tskipped\n' > $OUT/blocks.tsv
  source $ROOT/exp/lib/cotenants.sh       # records start; sets its own EXIT trap which we replace below
  trap m25_cleanup EXIT
  echo "OUT=$OUT"
  b0 || return 1
  local b
  for b in $M25_BLOCKS; do
    if (( $(elapsed) > GLOBAL_MIN*60 - 120 )); then BLK=$b; BLK_START=$(elapsed); BLK_END=$BLK_START; BLK_T0_ISO=$(date -Is); BLK_RUN=0; BLK_FAIL=0; BLK_SKIP=1; blk_end SKIPPED_GLOBAL; continue; fi
    case $b in B4) b4 ;; B1) b1 ;; B2) b2 ;; B3) b3 ;; esac
  done
  echo "end=$(date -Is) total_elapsed=$(elapsed)s" | tee -a $OUT/job_env.txt $OUT/summary.txt
  echo "--- blocks.tsv"; cat $OUT/blocks.tsv
  echo "done: $OUT"
}
