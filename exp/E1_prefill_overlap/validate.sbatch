#!/bin/bash
#SBATCH --account=MST114546
#SBATCH --partition=dev
#SBATCH --job-name=E1val
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=2
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/home/allen0117/comm_opt/results/E1_prefill_overlap/validation/%x-%j.out
#SBATCH --error=/home/allen0117/comm_opt/results/E1_prefill_overlap/validation/%x-%j.err
# E1 validation (2 GPUs, not a measurement): same e1_lib.sh as the main jobs, smoke sizes, every code path once.
set -uo pipefail
export E1_MODE=smoke E1_NGPU=2 E1_TAG=val E1_OUTBASE=/home/allen0117/comm_opt/results/E1_prefill_overlap/validation
source /home/allen0117/comm_opt/exp/E1_prefill_overlap/e1_lib.sh
e1_main
