#!/bin/bash
#SBATCH --account=MST114546
#SBATCH --partition=4nodes
#SBATCH --job-name=M23h100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=09:00:00
#SBATCH --output=/home/allen0117/comm_opt/results/M23_internode_baseline/slurm/%x-%j.out
#SBATCH --error=/home/allen0117/comm_opt/results/M23_internode_baseline/slurm/%x-%j.err
# M23 h100_4node (see exp/M23_internode_baseline/m23_lib.sh). 5 independent launches per config, seeded shuffle.
# Submit with: submit.sh <this> --dependency=afterok:475444:475459. Launches after 29700 s are skipped and recorded.
set -uo pipefail
ROOT=/home/allen0117/comm_opt
OUT=$ROOT/results/M23_internode_baseline/$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}_h100
mkdir -p "$OUT"
source /home/allen0117/comm_opt/exp/lib/cotenants.sh
source $ROOT/exp/M23_internode_baseline/m23_lib.sh
m23_main h100 29700
echo "done: $OUT"
