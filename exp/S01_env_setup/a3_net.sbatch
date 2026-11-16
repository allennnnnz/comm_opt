#!/bin/bash
#SBATCH --account=MST114546
#SBATCH --partition=dev
#SBATCH --job-name=S01a3
#SBATCH --dependency=singleton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:02:00
#SBATCH --output=/home/allen0117/comm_opt/results/S01_env_setup/slurm/%x-%j.out
#SBATCH --error=/home/allen0117/comm_opt/results/S01_env_setup/slurm/%x-%j.err
# S01 step A3: can the compute node reach the internet? (not a measurement, 1 GPU, non-exclusive)
set -uo pipefail
ROOT=/home/allen0117/comm_opt
OUT=$ROOT/results/S01_env_setup/$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}
mkdir -p "$OUT"
{
  echo "job_id=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
  nvidia-smi | head -5
} > "$OUT/job_env.txt" 2>&1
for u in https://pypi.org/simple/ https://huggingface.co https://github.com; do
  echo "=== curl -sI -m 20 $u"; curl -sI -m 20 "$u" 2>&1 | head -5; echo "exit=$?"
done > "$OUT/net.txt" 2>&1
echo "end=$(date -Is)" >> "$OUT/job_env.txt"
echo "done: $OUT"
