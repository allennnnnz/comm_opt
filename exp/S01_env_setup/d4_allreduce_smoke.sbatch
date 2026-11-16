#!/bin/bash
#SBATCH --account=MST114546
#SBATCH --partition=dev
#SBATCH --job-name=S01d4
#SBATCH --dependency=singleton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/home/allen0117/comm_opt/results/S01_env_setup/slurm/%x-%j.out
#SBATCH --error=/home/allen0117/comm_opt/results/S01_env_setup/slurm/%x-%j.err
# S01 steps D4/D5: nccl-tests all_reduce smoke test (functionality only, speeds are NOT interpreted).
# Override partition/nodes/job-name on the submit command line. Tries pmix_v4, then pmi2, then mpirun.
set -uo pipefail
ROOT=/home/allen0117/comm_opt
ENVROOT=/work/allen0117/comm_opt_env
NNODES=${SLURM_NNODES:-1}
OUT=$ROOT/results/S01_env_setup/$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}
mkdir -p "$OUT"
module load cuda/12.4 openmpi/4.1.6
{
  echo "job_id=$SLURM_JOB_ID job_name=$SLURM_JOB_NAME partition=$SLURM_JOB_PARTITION nodes=$NNODES nodelist=$SLURM_JOB_NODELIST start=$(date -Is)"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD)"
  module list 2>&1
  srun -N1 -n1 nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
  srun -N1 -n1 nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
} > "$OUT/job_env.txt" 2>&1

export LD_LIBRARY_PATH=$ENVROOT/venvs/vllm/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
BIN=$ROOT/third_party/nccl-tests/build/all_reduce_perf
ARGS="-b 8 -e 128M -f 4 -g 1 -n 5 -w 2"

try() {  # name, command...
  local name=$1; shift
  echo "### try $name: $*" | tee -a "$OUT/attempts.txt"
  "$@" > "$OUT/allreduce_$name.log" 2>&1
  local rc=$?
  local wrong
  wrong=$(grep -E '^ +[0-9]+ +[0-9]+ +float +sum' "$OUT/allreduce_$name.log" | awk '{print $(NF-1)}' | sort -u | tr '\n' ' ')
  echo "    rc=$rc  #wrong values seen (out-of-place/in-place): $wrong" | tee -a "$OUT/attempts.txt"
  return $rc
}

try pmix_v4 srun --mpi=pmix_v4 --nodes=$NNODES --ntasks-per-node=8 --gpus-per-node=8 $BIN $ARGS \
 || try pmi2 srun --mpi=pmi2 --nodes=$NNODES --ntasks-per-node=8 --gpus-per-node=8 $BIN $ARGS \
 || try mpirun mpirun -np $((8*NNODES)) --map-by ppr:8:node $BIN $ARGS

echo "end=$(date -Is)" >> "$OUT/job_env.txt"
echo "done: $OUT"
