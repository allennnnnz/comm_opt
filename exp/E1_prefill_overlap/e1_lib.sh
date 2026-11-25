#!/bin/bash
# E1 shared library, sourced by job_*.sbatch / validate.sbatch AT RUN TIME.  Spec: docs/specs/S09_E1_impl.md, card: exp/E1_prefill_overlap/README.md
# Borrows exp/M25_topic_survey/m25_lib.sh (env, B0 = warm-up + 8-GPU GEMM burn + thermal gate, time-budget helpers blk_begin/blk_end/can_run/cap/
# skip_rec/rec_rc, with_env, gpu_clean, m25_cleanup).  Those functions only use $OUT, $M25_MODE, $NGPU, $GLOBAL_MIN, $WARM_S, $M25_T0 -> we set them.
# Control variables:  E1_MODE=full|smoke  E1_NGPU=8|2  E1_TAG=<suffix>  E1_OUTBASE=<results dir>  E1_BLOCKS="A B C"
E1_MODE=${E1_MODE:-full}
export M25_MODE=$E1_MODE M25_NGPU=${E1_NGPU:-8} M25_BLOCKS="${E1_BLOCKS:-A B C}"
if [[ $E1_MODE == smoke ]]; then
  export M25_GLOBAL_MIN=${M25_GLOBAL_MIN:-27} M25_BURN_S=${M25_BURN_S:-20}
else
  export M25_GLOBAL_MIN=${M25_GLOBAL_MIN:-290}      # job time limit is 300 min; leave room for cleanup + analyze
fi
source /home/allen0117/comm_opt/exp/M25_topic_survey/m25_lib.sh     # also sources nccl_bench.sh (venv, shuffle_plan) and sets ROOT, ENVROOT, MODELS, NGPU, SEED
set +e   # m25_lib sets -uo pipefail only; keep going on failed blocks
E1D=$ROOT/exp/E1_prefill_overlap
E1_SEED=20261125
E1_OUTBASE=${E1_OUTBASE:-$ROOT/results/E1_prefill_overlap}

if [[ $E1_MODE == smoke ]]; then
  LIM_A=${E1_LIM_A:-12}; LIM_B=${E1_LIM_B:-6}; LIM_C=${E1_LIM_C:-6}
  A_MODEL=Qwen2.5-7B-Instruct; A_TP=2; A_LENS="2048,4096"; A_REPS=4; A_DISC=1; A_PROF=1; A_NL=1; A_TO=600
  B_MECHS="default cta8 symm"; B_NL=1; B_WARM=3; B_ITERS=10
  C_NL=1; C_WARM=2; C_ITERS=5
else
  LIM_A=${E1_LIM_A:-90}; LIM_B=${E1_LIM_B:-90}; LIM_C=${E1_LIM_C:-60}
  A_MODEL=Qwen2.5-72B-Instruct; A_TP=8; A_LENS="2048,4096,8192,16384"; A_REPS=10; A_DISC=2; A_PROF=3; A_NL=3; A_TO=1200
  B_MECHS="default nvls ring_ll128 ring_simple cta8 cta16 symm"; B_NL=3; B_WARM=10; B_ITERS=50
  C_NL=3; C_WARM=5; C_ITERS=20
fi
declare -A MECH_ENV=(
  [default]="-" [nvls]="NCCL_ALGO=NVLS" [ring_ll128]="NCCL_ALGO=Ring NCCL_PROTO=LL128" [ring_simple]="NCCL_ALGO=Ring NCCL_PROTO=Simple"
  [cta8]="NCCL_MAX_CTAS=8" [cta16]="NCCL_MAX_CTAS=16" [symm]="-")

# Opus 2026-11-25 (coordinator): full mode B0 limit 10 min (m25_lib.sh b0 hardcodes 8) -> wrap blk_begin
eval "$(declare -f blk_begin | sed '1s/blk_begin/m25_blk_begin/')"
blk_begin() { if [[ $1 == B0 && $E1_MODE == full ]]; then m25_blk_begin B0 10; else m25_blk_begin "$@"; fi; }

# redefine: also kill our own leftovers (m25's version only knows vllm_bench.py)
gpu_clean() {
  local i
  for i in 1 2 3 4 5 6; do
    [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]] && return 0
    sleep 5
  done
  echo "leftover GPU processes after wait: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')" >> $OUT/summary.txt
  pkill -u "$USER" -f 'VLLM::' 2>/dev/null; pkill -u "$USER" -f e1_vllm.py 2>/dev/null
  pkill -u "$USER" -f e1_pairs.py 2>/dev/null; pkill -u "$USER" -f e1_layer.py 2>/dev/null; sleep 5
}

e1_torchrun() {  # e1_torchrun <timeout_s> <log> <script args...> ; uses the current environment; stdin from /dev/null (torchrun/mpirun can eat a plan file)
  local to=$1 log=$2; shift 2
  timeout $to torchrun --standalone --nproc-per-node $NGPU "$@" > $log 2>&1 < /dev/null
}

# ---------------------------------------------------------------- A  vLLM prefill baseline
a_block() {
  blk_begin A $LIM_A
  # same workaround as M25 b2: FlashInfer JIT needs nvcc >= 12.8, cluster has <= 12.6
  export VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0 VLLM_USE_FLASHINFER_SAMPLER=0
  module load gcc/12.5.0 2>/dev/null || true
  mkdir -p $OUT/a $ENVROOT/traces/$(basename $OUT)
  printf 'launch\trc\tseconds\n' > $OUT/a/launches.tsv
  local n d t0 rc
  for n in $(seq 1 $A_NL); do
    can_run 420 || { skip_rec "A launch$n"; continue; }
    d=$OUT/a/$n; mkdir -p $d; t0=$SECONDS
    timeout $(cap $A_TO) python $E1D/e1_vllm.py --model $MODELS/$A_MODEL --tp $A_TP --lens "$A_LENS" --reps $A_REPS --discard $A_DISC \
      --prof-reps $A_PROF --seed $((E1_SEED+n)) --launch $n --outdir $d --trace-root $ENVROOT/traces/$(basename $OUT)/a$n \
      > $d/engine.log 2>&1 < /dev/null
    rc=$?
    printf '%s\t%s\t%s\n' $n $rc $((SECONDS-t0)) >> $OUT/a/launches.tsv
    echo "A launch=$n rc=$rc seconds=$((SECONDS-t0))" | tee -a $OUT/summary.txt
    rec_rc $rc
    grep -h "all-reduce backends" $d/engine.log | head -2 > $d/allreduce_backend.txt
    gzip -f $d/engine.log
    gpu_clean
  done
  blk_end DONE
}

# ---------------------------------------------------------------- B  operator-level pairs
env_check() {  # env_check <mech> <launch> <logdir> : did each NCCL variable take effect? which algo/proto/channels per all-reduce size?
  local mech=$1 launch=$2 d=$3 kv k v n
  for kv in ${MECH_ENV[$mech]}; do
    [[ $kv == - ]] && continue
    k=${kv%%=*}; v=${kv#*=}
    n=$(cat $d/nccl_*.log 2>/dev/null | grep -c "$k set by environment to $v")
    printf '%s\t%s\t%s\tset_by_environment_lines=%s\n' $mech $launch "$kv" $n >> $OUT/b/env_check.tsv
  done
  cat $d/nccl_*.log 2>/dev/null | grep -h "AllReduce: [0-9]* Bytes -> Algo" \
    | sed -E 's/.*AllReduce: ([0-9]+) Bytes -> Algo ([A-Z0-9_]+) proto ([A-Za-z0-9]+) channel\{Lo\.\.Hi\}=\{([0-9]+)\.\.([0-9]+)\}.*/\1\t\2\t\3\t\4..\5/' \
    | sort -u | awk -v m=$mech -v l=$launch '{print m"\t"l"\talgo_line\tbytes="$1" algo="$2" proto="$3" channels="$4}' >> $OUT/b/env_check.tsv
}
b_block() {
  blk_begin B $LIM_B
  mkdir -p $OUT/b/nccl_logs
  printf 'mech\tlaunch\tset_check\n' > $OUT/b/env_check.tsv
  local pu=$OUT/b/plan_unshuffled.tsv m r; : > $pu
  for m in $B_MECHS; do for r in $(seq 1 $B_NL); do printf '%s\t%s\n' $m $r >> $pu; done; done
  shuffle_plan $((E1_SEED+11)) $pu $OUT/b/plan_order.tsv
  printf 'seq\tmech\tlaunch\trc\tseconds\tstart\n' > $OUT/b/launches.tsv
  local seq=0 mech launch d t0 rc
  while IFS=$'\t' read -r mech launch <&3; do
    seq=$((seq+1))
    can_run 150 || { skip_rec "B $mech launch$launch"; continue; }
    d=$OUT/b/nccl_logs/${mech}_l${launch}; mkdir -p $d; t0=$SECONDS
    with_env "${MECH_ENV[$mech]} NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING NCCL_DEBUG_FILE=$d/nccl_%h_%p.log" \
      e1_torchrun $(cap 420) $OUT/b/${mech}_l${launch}.log $E1D/e1_pairs.py --mech $mech --launch $launch --out $OUT/b/${mech}_l${launch}.jsonl \
        --warm $B_WARM --iters $B_ITERS
    rc=$?
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' $seq $mech $launch $rc $((SECONDS-t0)) "$(date -Is)" >> $OUT/b/launches.tsv
    echo "B seq=$seq mech=$mech launch=$launch rc=$rc seconds=$((SECONDS-t0))" | tee -a $OUT/summary.txt
    rec_rc $rc
    env_check $mech $launch $d
    tar -czf $d.tar.gz -C $OUT/b/nccl_logs ${mech}_l${launch} 2>/dev/null && rm -rf $d
    gzip -f $OUT/b/${mech}_l${launch}.log
    gpu_clean
  done 3< $OUT/b/plan_order.tsv
  # choose C's mechanisms from B (stdlib script); fall back with a recorded reason if B produced nothing usable
  python $E1D/pick_c_mechs.py $OUT/b > $OUT/b/pick_c_mechs.log 2>&1
  if ! grep -q '^mech2=' $OUT/b/c_mechs.txt 2>/dev/null; then
    printf 'mech1=cta8\nmech2=ring_simple\n# FALLBACK: pick_c_mechs.py failed or B had too little data (see pick_c_mechs.log); using NCCL_MAX_CTAS=8 and Ring+Simple per spec S09\n' > $OUT/b/c_mechs.txt
    echo "B: c_mechs fallback used" | tee -a $OUT/summary.txt
  fi
  cat $OUT/b/c_mechs.txt | tee -a $OUT/summary.txt
  blk_end DONE
}

# ---------------------------------------------------------------- C  one-layer-stack prototype
c_block() {
  blk_begin C $LIM_C
  mkdir -p $OUT/c $OUT/c/nccl_logs
  local m1 m2
  m1=$(sed -n 's/^mech1=//p' $OUT/b/c_mechs.txt 2>/dev/null | head -1); m2=$(sed -n 's/^mech2=//p' $OUT/b/c_mechs.txt 2>/dev/null | head -1)
  if [[ -z $m1 || -z $m2 ]]; then   # B block not run / failed before writing: same fallback as spec
    m1=cta8; m2=ring_simple; mkdir -p $OUT/b
    printf 'mech1=cta8\nmech2=ring_simple\n# FALLBACK: no b/c_mechs.txt (B did not run or failed); NCCL_MAX_CTAS=8 and Ring+Simple per spec S09\n' > $OUT/b/c_mechs.txt
  fi
  echo "C mechs: default $m1 $m2" | tee -a $OUT/summary.txt
  local pu=$OUT/c/plan_unshuffled.tsv m r; : > $pu
  for m in default $m1 $m2; do for r in $(seq 1 $C_NL); do printf '%s\t%s\n' $m $r >> $pu; done; done
  shuffle_plan $((E1_SEED+12)) $pu $OUT/c/plan_order.tsv
  printf 'seq\tmech\tlaunch\trc\tseconds\tstart\n' > $OUT/c/launches.tsv
  local seq=0 mech launch t0 rc d
  while IFS=$'\t' read -r mech launch <&3; do
    seq=$((seq+1))
    can_run 200 || { skip_rec "C $mech launch$launch"; continue; }
    d=$OUT/c/nccl_logs/${mech}_l${launch}; mkdir -p $d; t0=$SECONDS
    with_env "${MECH_ENV[$mech]} NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV NCCL_DEBUG_FILE=$d/nccl_%h_%p.log" \
      e1_torchrun $(cap 600) $OUT/c/${mech}_l${launch}.log $E1D/e1_layer.py --mech $mech --launch $launch --out $OUT/c/${mech}_l${launch}.jsonl \
        --warm $C_WARM --iters $C_ITERS
    rc=$?
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' $seq $mech $launch $rc $((SECONDS-t0)) "$(date -Is)" >> $OUT/c/launches.tsv
    echo "C seq=$seq mech=$mech launch=$launch rc=$rc seconds=$((SECONDS-t0))" | tee -a $OUT/summary.txt
    rec_rc $rc
    tar -czf $d.tar.gz -C $OUT/c/nccl_logs ${mech}_l${launch} 2>/dev/null && rm -rf $d
    gzip -f $OUT/c/${mech}_l${launch}.log
    gpu_clean
  done 3< $OUT/c/plan_order.tsv
  blk_end DONE
}

# ---------------------------------------------------------------- main
e1_main() {
  M25_T0=$SECONDS
  OUT=$E1_OUTBASE/$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}${E1_TAG:+_$E1_TAG}
  mkdir -p $OUT; : > $OUT/summary.txt
  printf 'block\tstart\tend\tstart_s\tend_s\tstatus\tcounts\tskipped\n' > $OUT/blocks.tsv
  source $ROOT/exp/lib/cotenants.sh
  trap m25_cleanup EXIT          # kills the GPU monitor, gzips gpu_monitor.csv, records co-tenants at end
  echo "OUT=$OUT"
  b0 || return 1
  local b
  for b in $M25_BLOCKS; do
    if (( $(elapsed) > GLOBAL_MIN*60 - 120 )); then
      BLK=$b; BLK_START=$(elapsed); BLK_END=$BLK_START; BLK_T0_ISO=$(date -Is); BLK_RUN=0; BLK_FAIL=0; BLK_SKIP=1; blk_end SKIPPED_GLOBAL; continue
    fi
    case $b in A) a_block ;; B) b_block ;; C) c_block ;; esac
  done
  echo "end=$(date -Is) total_elapsed=$(elapsed)s" | tee -a $OUT/job_env.txt $OUT/summary.txt
  echo "--- blocks.tsv"; cat $OUT/blocks.tsv
  python $E1D/analyze.py --c-valid seq $OUT > $OUT/analyze_stdout_cvalid_seq.txt 2>&1; mv $OUT/analysis.json $OUT/analysis_cvalid_seq.json 2>/dev/null
  python $E1D/analyze.py $OUT > $OUT/analyze_stdout.txt 2>&1; echo "analyze rc=$?" | tee -a $OUT/summary.txt
  echo "done: $OUT"
}
