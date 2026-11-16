#!/bin/bash
#SBATCH --account=MST114546
#SBATCH --partition=normal
#SBATCH --job-name=M20sharp
#SBATCH --dependency=singleton
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/home/allen0117/comm_opt/results/M20_env_snapshot/slurm/%x-%j.out
#SBATCH --error=/home/allen0117/comm_opt/results/M20_env_snapshot/slurm/%x-%j.err
# M20 P10: 2-node SHARP / CollNet check, configs (a)-(f) + AG/RS. Functionality only; speeds not interpreted.
# Config (a) doubles as S01's 2-node H100 smoke test.
set -uo pipefail
ROOT=/home/allen0117/comm_opt
EXPD=$ROOT/exp/M20_env_snapshot
OUT=$ROOT/results/M20_env_snapshot/$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}_sharp2n
mkdir -p "$OUT"
source $ROOT/exp/lib/nano5_env.sh
source $EXPD/common.sh
NNODES=2; NPROCS_PER_NODE=8
NCCLD=$NCCL_LIB_DIR
BASE_LD=$LD_LIBRARY_PATH
PLUG_LD=$NCCLD:$HPCX/nccl_rdma_sharp_plugin/lib:$HPCX/sharp/lib:$HPCX/ucx/lib:$BASE_LD

{
  echo "job_id=$SLURM_JOB_ID job_name=$SLURM_JOB_NAME partition=$SLURM_JOB_PARTITION nodes=$SLURM_JOB_NODELIST start=$(date -Is)"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD) dirty=$(git -C "$ROOT" status --porcelain | wc -l)"
  module list 2>&1
  srun --nodes=2 --ntasks-per-node=1 nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
  srun --nodes=2 --ntasks-per-node=1 nvidia-smi topo -m
  python -c "import torch;print('torch',torch.__version__,'nccl',torch.cuda.nccl.version())" 2>&1
  echo "BASE_LD=$BASE_LD"; echo "PLUG_LD=$PLUG_LD"
} > "$OUT/job_env.txt" 2>&1

others=$(srun --nodes=2 --ntasks-per-node=1 nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
if [[ -n "$others" ]]; then echo "GPU processes found; aborting: $others" >&2; echo "ABORT: $others" >> "$OUT/job_env.txt"; exit 1; fi

# per-node records
srun --nodes=2 --ntasks-per-node=1 bash -c 'echo "### $(hostname)"; ibstat; ip -br addr | grep -i ib' > $OUT/ibstat_ipib_per_node.txt 2>&1
srun --nodes=2 --ntasks-per-node=1 bash -c "echo '### '\$(hostname); ldd $HPCX/nccl_rdma_sharp_plugin/lib/libnccl-net.so; echo '--- with plugin LD_LIBRARY_PATH'; LD_LIBRARY_PATH=$PLUG_LD ldd $HPCX/nccl_rdma_sharp_plugin/lib/libnccl-net.so" > $OUT/ldd_plugin_per_node.txt 2>&1

export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,COLL,TUNING
AR=$NCCL_TESTS/all_reduce_perf; AG=$NCCL_TESTS/all_gather_perf; RS=$NCCL_TESTS/reduce_scatter_perf
ARGS="-b 8 -e 64M -f 4 -n 5 -w 2"
TO=420

# (a) no plugin, default; also decides the launcher
export LD_LIBRARY_PATH=$BASE_LD
probe_launcher a_noplugin_default $TO $AR $ARGS
if [[ -z $LAUNCH_MODE ]]; then echo "no launcher worked; stopping" | tee -a $OUT/summary.txt; echo "end=$(date -Is)" >> $OUT/job_env.txt; exit 1; fi

export LD_LIBRARY_PATH=$PLUG_LD
NCCL_COLLNET_ENABLE=0 run_cfg b_plugin_collnet0 $TO $AR $ARGS
NCCL_COLLNET_ENABLE=1 run_cfg c_plugin_collnet1 $TO $AR $ARGS
NCCL_COLLNET_ENABLE=1 NCCL_ALGO=CollnetDirect run_cfg d_collnet1_CollnetDirect $TO $AR $ARGS
NCCL_COLLNET_ENABLE=1 NCCL_ALGO=CollnetChain run_cfg e_collnet1_CollnetChain $TO $AR $ARGS
NCCL_COLLNET_ENABLE=1 NCCL_ALGO=NVLS run_cfg f_collnet1_NVLS $TO $AR $ARGS
NCCL_COLLNET_ENABLE=1 run_cfg c_allgather_collnet1 $TO $AG $ARGS
NCCL_COLLNET_ENABLE=1 run_cfg c_reducescatter_collnet1 $TO $RS $ARGS
echo "end=$(date -Is)" >> "$OUT/job_env.txt"
echo "done: $OUT"
