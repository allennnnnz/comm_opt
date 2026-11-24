#!/bin/bash
#SBATCH --account=MST114546
#SBATCH --partition=dev
#SBATCH --job-name=M25b
#SBATCH --dependency=singleton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/home/allen0117/comm_opt/results/M25_topic_survey/slurm/%x-%j.out
#SBATCH --error=/home/allen0117/comm_opt/results/M25_topic_survey/slurm/%x-%j.err
# M25 topic survey main job. All logic is in m25_lib.sh (read at run time). Blocks: B2
set -uo pipefail
export M25_MODE=full M25_NGPU=8 M25_BLOCKS="B2" M25_TAG=b
export M25_B2_EAGER=${M25_B2_EAGER:-0}
source /home/allen0117/comm_opt/exp/M25_topic_survey/m25_lib.sh
m25_main
