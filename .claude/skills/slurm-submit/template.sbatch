#!/bin/bash
# Standard Nano5 job script for comm_opt. Copy into exp/<ID>_<name>/job.sbatch and fill the <...> fields.
# Submit with .claude/skills/slurm-submit/submit.sh (it adds the mail options and runs the checks).
#SBATCH --account=MST114546
#SBATCH --partition=<dev|normal|normal2|4nodes>
#SBATCH --job-name=<ID>
#SBATCH --dependency=singleton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=/home/allen0117/comm_opt/results/<ID>_<name>/slurm/%x-%j.out
#SBATCH --error=/home/allen0117/comm_opt/results/<ID>_<name>/slurm/%x-%j.err

set -euo pipefail

ROOT=/home/allen0117/comm_opt
OUT=$ROOT/results/<ID>_<name>/$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}
mkdir -p "$OUT"

module load cuda/12.4 openmpi/4.1.6

# --- environment record (traceability: job, node, commit, GPUs, topology) ---
{
  echo "job_id=$SLURM_JOB_ID job_name=$SLURM_JOB_NAME partition=$SLURM_JOB_PARTITION"
  echo "node=$(hostname) start=$(date -Is)"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD) dirty=$(git -C "$ROOT" status --porcelain | wc -l)"
  module list 2>&1
  srun -n 1 nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.sm,clocks.max.sm,temperature.gpu --format=csv
  srun -n 1 nvidia-smi topo -m
} > "$OUT/job_env.txt" 2>&1

# --- no other process may be on our GPUs (rule R13) ---
others=$(srun -n 1 nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
if [[ -n "$others" ]]; then
  echo "GPU processes found before measuring; aborting:" >&2
  echo "$others" >&2
  exit 1
fi

### BODY — replace with the experiment's commands; write all data under $OUT ###
srun -n 1 --gpus-per-task=8 <program and args>
### END BODY ###

echo "end=$(date -Is)" >> "$OUT/job_env.txt"
echo "done: $OUT"
