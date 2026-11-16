# M23 job body (sourced by h100_4node.sbatch / h200_2node.sbatch after $OUT exists and cotenants.sh was sourced).
# usage: m23_main <h100|h200> <deadline_seconds>
source /home/allen0117/comm_opt/exp/lib/nccl_bench.sh
G8=0,1,2,3,4,5,6,7

# all_reduce algorithm / plugin matrix (card: default, Ring, Tree, NVLSTree; plugin+COLLNET=0 control; plugin+COLLNET=1: default, CollnetDirect, CollnetChain)
m23_allreduce_cfgs() {  # <pf> <scope> <nn> <nodes>
  local pf=$1 sc=$2 nn=$3 nodes=$4 b=all_reduce_perf
  add_cfg "$pf" "${sc}_ar_default"  $b $nn 8 $G8 0 "-" block $nodes
  add_cfg "$pf" "${sc}_ar_ring"     $b $nn 8 $G8 0 "NCCL_ALGO=Ring" block $nodes
  add_cfg "$pf" "${sc}_ar_tree"     $b $nn 8 $G8 0 "NCCL_ALGO=Tree" block $nodes
  add_cfg "$pf" "${sc}_ar_nvlstree" $b $nn 8 $G8 0 "NCCL_ALGO=NVLSTree" block $nodes
  add_cfg "$pf" "${sc}_arP0_default" $b $nn 8 $G8 1 "NCCL_COLLNET_ENABLE=0" block $nodes
  add_cfg "$pf" "${sc}_arP0_ring"    $b $nn 8 $G8 1 "NCCL_COLLNET_ENABLE=0 NCCL_ALGO=Ring" block $nodes
  add_cfg "$pf" "${sc}_arP1_default" $b $nn 8 $G8 1 "NCCL_COLLNET_ENABLE=1" block $nodes
  add_cfg "$pf" "${sc}_arP1_cdirect" $b $nn 8 $G8 1 "NCCL_COLLNET_ENABLE=1 NCCL_ALGO=CollnetDirect" block $nodes
  add_cfg "$pf" "${sc}_arP1_cchain"  $b $nn 8 $G8 1 "NCCL_COLLNET_ENABLE=1 NCCL_ALGO=CollnetChain" block $nodes
  # Opus 2026-11-17: cross-node NVLS = intra-node NVLS + inter-node IB SHARP (NCCL v2.32.6 src/tuning/nvls.cc:149); core of direction A
  add_cfg "$pf" "${sc}_arP1_nvls"    $b $nn 8 $G8 1 "NCCL_COLLNET_ENABLE=1 NCCL_ALGO=NVLS" block $nodes
}
m23_other_cfgs() {  # <pf> <scope> <nn> <nodes>
  local pf=$1 sc=$2 nn=$3 nodes=$4
  add_cfg "$pf" "${sc}_ag"  all_gather_perf     $nn 8 $G8 0 "-" block $nodes
  add_cfg "$pf" "${sc}_rs"  reduce_scatter_perf $nn 8 $G8 0 "-" block $nodes
  add_cfg "$pf" "${sc}_a2a" alltoall_perf       $nn 8 $G8 0 "-" block $nodes
}
m23_twonode_cfgs() {  # <pf> <scope> <nodes> <with_gdroff 0|1>   (sendrecv: 1 GPU/node, and 8 GPUs/node in a cyclic rank order so every send/recv crosses nodes)
  local pf=$1 sc=$2 nodes=$3 gdr=$4
  m23_allreduce_cfgs "$pf" "$sc" 2 "$nodes"
  m23_other_cfgs "$pf" "$sc" 2 "$nodes"
  add_cfg "$pf" "${sc}_sr1" sendrecv_perf 2 1 0 0 "-" block $nodes
  add_cfg "$pf" "${sc}_sr8" sendrecv_perf 2 8 $G8 0 "-" cyclic $nodes
  if [[ $gdr == 1 ]]; then
    add_cfg "$pf" "${sc}_ar_default_gdroff" all_reduce_perf 2 8 $G8 0 "NCCL_NET_GDR_LEVEL=LOC" block $nodes
    add_cfg "$pf" "${sc}_ar_ring_gdroff"    all_reduce_perf 2 8 $G8 0 "NCCL_ALGO=Ring NCCL_NET_GDR_LEVEL=LOC" block $nodes
    add_cfg "$pf" "${sc}_sr1_gdroff" sendrecv_perf 2 1 0 0 "NCCL_NET_GDR_LEVEL=LOC" block $nodes
    add_cfg "$pf" "${sc}_sr8_gdroff" sendrecv_perf 2 8 $G8 0 "NCCL_NET_GDR_LEVEL=LOC" cyclic $nodes
  fi
}

# node pairs inside a 4-node allocation. hgpn01-06 = group A, hgpn17-21 = group B (card). Writes "label<TAB>nodeA,nodeB" lines.
m23_pairs() {
  python3 - "$@" <<'PY'
import sys, itertools, re
nodes = sys.argv[1:]
grp = lambda h: 'A' if int(re.sub(r'\D', '', h)) <= 6 else 'B'
combos = list(itertools.combinations(nodes, 2))
same = [c for c in combos if grp(c[0]) == grp(c[1])]
cross = [c for c in combos if grp(c[0]) != grp(c[1])]
out = []
if same:
    s1 = same[0]; out.append(("same", s1))
    if cross:
        dj = [c for c in cross if not set(c) & set(s1)]
        out.append(("cross", (dj or cross)[0]))
    else:
        dj = [c for c in same if not set(c) & set(s1)]
        out.append(("same2_nocross", (dj or same[1:2] or [None])[0]))
for lab, c in out:
    if c: print(f"{lab}\t{c[0]},{c[1]}")
PY
}

m23_main() {
  local kind=$1 deadline=$2 pf="$OUT/plan_unshuffled.tsv"
  mkdir -p "$OUT/runs"
  NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
  {
    echo "job_id=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION nodelist=$SLURM_JOB_NODELIST nnodes=${#NODES[@]} kind=$kind start=$(date -Is)"
    echo "git_commit=$(git -C /home/allen0117/comm_opt rev-parse HEAD) dirty=$(git -C /home/allen0117/comm_opt status --porcelain | wc -l)"
    module list 2>&1
    srun -N ${#NODES[@]} --ntasks-per-node=1 bash -c 'echo "== $(hostname -s)"; nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.sm,clocks.max.sm --format=csv,noheader; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader'
    srun -N 1 -n 1 nvidia-smi topo -m
    echo "nccl-tests commit $(git -C /home/allen0117/comm_opt/third_party/nccl-tests rev-parse --short HEAD)"
    echo "LD_LIBRARY_PATH(no plugin)=$LD_BASE"
    echo "SHARP plugin LD_LIBRARY_PATH prefix=$NCCL_LIB_DIR:$HPCX/nccl_rdma_sharp_plugin/lib:$HPCX/sharp/lib:$HPCX/ucx/lib"
    echo "SEG_SMALL_ARGS=$SEG_SMALL_ARGS  SEG_BIG_ARGS=$SEG_BIG_ARGS  PROBE_ARGS=$PROBE_ARGS"
  } > "$OUT/job_env.txt" 2>&1
  others=$(srun -N ${#NODES[@]} --ntasks-per-node=1 nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
  if [[ -n $others ]]; then echo "GPU processes found before measuring; aborting: $others" | tee -a "$OUT/summary.txt"; return 1; fi
  probe_launcher ${#NODES[@]} 8 || { echo "no launcher works" | tee -a "$OUT/summary.txt"; return 1; }
  : > "$pf"
  if [[ $kind == h100 ]]; then
    m23_allreduce_cfgs "$pf" n4 4 all
    m23_other_cfgs "$pf" n4 4 all
    m23_pairs "${NODES[@]}" > "$OUT/pairs.tsv"
    local lab nd
    while IFS=$'\t' read -r lab nd; do m23_twonode_cfgs "$pf" "n2${lab}" "$nd" 1; done < "$OUT/pairs.tsv"
    echo "pairs:"; cat "$OUT/pairs.tsv"
  else
    m23_twonode_cfgs "$pf" h200n2 all 0
  fi
  shuffle_plan 20261118 "$pf" "$OUT/plan_order.tsv"
  echo "configs=$(cut -f1 $pf | sort -u | wc -l) launches=$(wc -l < $OUT/plan_order.tsv) seed=20261118" | tee -a "$OUT/summary.txt"
  SEG_TIMEOUT=600 run_plan "$OUT/plan_order.tsv" "$deadline"
  echo "end=$(date -Is)" >> "$OUT/job_env.txt"
}
