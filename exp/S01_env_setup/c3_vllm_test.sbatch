#!/bin/bash
#SBATCH --account=MST114546
#SBATCH --partition=dev
#SBATCH --job-name=S01c3
#SBATCH --dependency=singleton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/home/allen0117/comm_opt/results/S01_env_setup/slurm/%x-%j.out
#SBATCH --error=/home/allen0117/comm_opt/results/S01_env_setup/slurm/%x-%j.err
# S01 step C3: torch checks, torchrun all_reduce, vLLM TP=8 generation. Not a measurement.
set -uo pipefail
ROOT=/home/allen0117/comm_opt
ENVROOT=/work/allen0117/comm_opt_env
OUT=$ROOT/results/S01_env_setup/$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}
mkdir -p "$OUT"
module load cuda/12.4 openmpi/4.1.6
{
  echo "job_id=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD)"
  module list 2>&1
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
} > "$OUT/job_env.txt" 2>&1

source $ENVROOT/venvs/vllm/bin/activate
export HF_HOME=$ENVROOT/hf_cache HF_HUB_OFFLINE=1
cd $ROOT/exp/S01_env_setup

echo "### 1 torch checks" ; python c3_check_torch.py > "$OUT/c3_1_torch.txt" 2>&1; echo "rc=$?" >> "$OUT/c3_1_torch.txt"
echo "### 2 torchrun all_reduce"
NCCL_DEBUG=INFO torchrun --nproc-per-node 8 c3_allreduce.py > "$OUT/c3_2_allreduce.txt" 2>&1; echo "rc=$?" >> "$OUT/c3_2_allreduce.txt"
echo "### 3 vllm TP=8"
VLLM_LOGGING_LEVEL=DEBUG NCCL_DEBUG=INFO python c3_vllm_tp8.py > "$OUT/c3_3_vllm_full.log" 2>&1; echo "rc=$?" >> "$OUT/c3_3_vllm_full.log"
grep -iE "all.?reduce|symm|pynccl|nccl|flashinfer|custom_ar|communicator|P2P|NVLS|nvlink" "$OUT/c3_3_vllm_full.log" > "$OUT/c3_3_vllm_comm_lines.txt"
grep -E "vllm.__version__|^TOKENS|^OUTPUT_TEXT|^rc=" "$OUT/c3_3_vllm_full.log" > "$OUT/c3_3_vllm_result.txt"
echo "end=$(date -Is)" >> "$OUT/job_env.txt"
echo "done: $OUT"
