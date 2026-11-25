#!/bin/bash
#SBATCH --account=MST114546
#SBATCH --partition=normal2
#SBATCH --job-name=E1h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=400G
#SBATCH --time=05:00:00
#SBATCH --output=/home/allen0117/comm_opt/results/E1_prefill_overlap/slurm/%x-%j.out
#SBATCH --error=/home/allen0117/comm_opt/results/E1_prefill_overlap/slurm/%x-%j.err
# E1 main job (h200, whole node). All logic in e1_lib.sh (read at run time). Blocks: B0 (thermal gate) A B C. Time limits: B0 10 / A 90 / B 90 / C 60 min.
set -uo pipefail
export E1_MODE=full E1_NGPU=8 E1_TAG=h200
source /home/allen0117/comm_opt/exp/E1_prefill_overlap/e1_lib.sh
e1_main
