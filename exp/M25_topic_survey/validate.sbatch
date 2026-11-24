#!/bin/bash
#SBATCH --account=MST114546
#SBATCH --partition=dev
#SBATCH --job-name=M25val
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=2
#SBATCH --time=00:40:00
#SBATCH --output=/home/allen0117/comm_opt/results/M25_topic_survey/validation/%x-%j.out
#SBATCH --error=/home/allen0117/comm_opt/results/M25_topic_survey/validation/%x-%j.err
# M25 validation (2 GPUs, not a measurement): same m25_lib.sh as the main job, smoke sizes, every code path once.
set -uo pipefail
export M25_MODE=smoke M25_NGPU=2 M25_BLOCKS="B4 B1 B2 B3" M25_OUTBASE=/home/allen0117/comm_opt/results/M25_topic_survey/validation M25_TAG=val
source /home/allen0117/comm_opt/exp/M25_topic_survey/m25_lib.sh
m25_main
echo "--- analyze.py smoke run"
python /home/allen0117/comm_opt/exp/M25_topic_survey/analyze.py "$OUT" > "$OUT/analyze_stdout.txt" 2>&1; echo "analyze rc=$?"
tail -30 "$OUT/analyze_stdout.txt"
