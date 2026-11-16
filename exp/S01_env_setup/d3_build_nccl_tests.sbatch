#!/bin/bash
#SBATCH --account=MST114546
#SBATCH --partition=dev
#SBATCH --job-name=S01d3
#SBATCH --dependency=singleton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/home/allen0117/comm_opt/results/S01_env_setup/slurm/%x-%j.out
#SBATCH --error=/home/allen0117/comm_opt/results/S01_env_setup/slurm/%x-%j.err
# S01 step D3: compile nccl-tests against the NCCL shipped with the vLLM venv's PyTorch. Not a measurement.
set -uo pipefail
ROOT=/home/allen0117/comm_opt
ENVROOT=/work/allen0117/comm_opt_env
OUT=$ROOT/results/S01_env_setup/$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}
mkdir -p "$OUT"
module load gcc/11.5.0 cuda/12.4 openmpi/4.1.6
{
  echo "job_id=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "nccl-tests: $(git -C $ROOT/third_party/nccl-tests describe --tags) $(git -C $ROOT/third_party/nccl-tests rev-parse HEAD)"
  module list 2>&1; echo "CUDA_HOME=$CUDA_HOME MPI_HOME=$MPI_HOME NCCL_HOME=$ENVROOT/nccl_home"
  which nvcc mpicc; nvcc --version
} > "$OUT/job_env.txt" 2>&1
cd $ROOT/third_party/nccl-tests
make clean > /dev/null 2>&1
CMD="make -j12 MPI=1 MPI_HOME=$MPI_HOME CUDA_HOME=$CUDA_HOME NCCL_HOME=$ENVROOT/nccl_home NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90"
echo "$CMD" > "$OUT/make_cmd.txt"
$CMD > "$OUT/make.log" 2>&1; echo "make rc=$?" | tee -a "$OUT/make.log"
ls -la build/*_perf >> "$OUT/make.log" 2>&1
LD_LIBRARY_PATH=$ENVROOT/nccl_home/lib:$LD_LIBRARY_PATH ldd build/all_reduce_perf > "$OUT/ldd.txt" 2>&1
echo "end=$(date -Is)" >> "$OUT/job_env.txt"
echo "done: $OUT"
