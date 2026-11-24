#!/bin/bash
#SBATCH --account=MST114546
#SBATCH --partition=dev
#SBATCH --job-name=M25fix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=2
#SBATCH --time=00:20:00
#SBATCH --output=/home/allen0117/comm_opt/results/M25_topic_survey/validation/%x-%j.out
#SBATCH --error=/home/allen0117/comm_opt/results/M25_topic_survey/validation/%x-%j.err
# Opus 2026-11-25: quick check of the fixes after main job 497945 (not a measurement):
#  T1 8 MPI ranks wire up (old flags vs new m25_mpirun flags; 8 ranks on 2 GPUs -> MPI_Allgather must pass, then a CUDA
#     device error from nccl-tests is expected and fine)
#  T2 m25_mpirun no longer eats a while-read loop's stdin
#  T3 overlap_bench.py with NCCL_ALGO=NVLS reaches the end (maxreduce on gloo)
set -uo pipefail
export M25_MODE=smoke M25_NGPU=2
source /home/allen0117/comm_opt/exp/M25_topic_survey/m25_lib.sh
O=/home/allen0117/comm_opt/results/M25_topic_survey/validation/fixcheck_${SLURM_JOB_ID}; mkdir -p $O; cd $O
bin=$NCCL_TESTS/all_reduce_perf
exec 0</dev/null   # T1a uses plain mpirun; keep it off the job's stdin
echo "== T1a old flags, 8 ranks"
timeout 90 mpirun -np 8 --oversubscribe --bind-to none -x LD_LIBRARY_PATH $bin -b 8 -e 8 -g 1 -n 1 -w 0 > t1a.log 2>&1
echo "rc=$? ucx_wireup_lines=$(grep -c 'no remote ep address' t1a.log) allgather_err=$(grep -c 'MPI_Allgather' t1a.log)"
echo "== T1b new flags, 8 ranks"
timeout 90 bash -c "$(declare -f m25_mpirun); m25_mpirun 8 $bin -b 8 -e 8 -g 1 -n 1 -w 0" > t1b.log 2>&1
echo "rc=$? ucx_wireup_lines=$(grep -c 'no remote ep address' t1b.log) allgather_err=$(grep -c 'MPI_Allgather' t1b.log)"
grep -i "error\|invalid\|device" t1b.log | grep -v -i hwloc | head -5
echo "== T1c new flags, 2 ranks, real all_reduce"
timeout 90 bash -c "$(declare -f m25_mpirun); m25_mpirun 2 $bin -b 8 -e 64M -f 8 -g 1 -n 20 -w 5" > t1c.log 2>&1
echo "rc=$?"; grep -E "^ +[0-9]+ |Avg bus|wrong" t1c.log | tail -4
echo "== T2 stdin"
printf 'a\nb\nc\n' | while read -r x; do timeout 60 bash -c "$(declare -f m25_mpirun); m25_mpirun 2 $bin -b 8 -e 8 -g 1 -n 1 -w 0" > /dev/null 2>&1; echo "line=$x"; done
echo "== T3 overlap_bench NVLS (2 GPUs)"
timeout 300 torchrun --standalone --nproc-per-node 2 $EXPD/overlap_bench.py --calib calib.json \
  --cells all_reduce:4:8192 > t3_calib.log 2>&1; echo "calib rc=$?"
NCCL_ALGO=NVLS timeout 300 torchrun --standalone --nproc-per-node 2 $EXPD/overlap_bench.py --calib-in calib.json --out t3.jsonl \
  --cfg nvls --launch 1 --cells all_reduce:4:8192 --reps 1 > t3.log 2>&1; echo "nvls rc=$? lines=$(cat t3.jsonl 2>/dev/null | wc -l)"
grep -i "error" t3.log | head -5
gzip -f *.log
echo done
