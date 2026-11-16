#!/bin/bash
#SBATCH --account=MST114546
#SBATCH --partition=dev
#SBATCH --job-name=M21b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=01:20:00
#SBATCH --output=/home/allen0117/comm_opt/results/M21_intranode_baseline/slurm/%x-%j.out
#SBATCH --error=/home/allen0117/comm_opt/results/M21_intranode_baseline/slurm/%x-%j.err
# M21 part b (see exp/M21_intranode_baseline/m21_lib.sh). Single H100 node, 5 independent launches per config, seeded shuffle.
# Override partition/exclusions on the submit command line. Time budget guard: launches after 4300 s are skipped and recorded.
set -uo pipefail
ROOT=/home/allen0117/comm_opt
OUT=$ROOT/results/M21_intranode_baseline/$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}_b
mkdir -p "$OUT"
source /home/allen0117/comm_opt/exp/lib/cotenants.sh
source $ROOT/exp/M21_intranode_baseline/m21_lib.sh
m21_main b 4300
echo "done: $OUT"
