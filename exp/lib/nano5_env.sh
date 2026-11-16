# Source this from job scripts on Nano5 compute nodes: `source /home/allen0117/comm_opt/exp/lib/nano5_env.sh`
# (S01, 2026-11-16). Contains no credentials.
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export COMM_OPT_ROOT="$_REPO"
export ENVROOT=/work/allen0117/comm_opt_env

module load cuda/12.4 openmpi/4.1.6

# shellcheck disable=SC1091
source "$ENVROOT/venvs/vllm/bin/activate"

export HF_HOME="$ENVROOT/hf_cache"
# the venv's NCCL (nvidia-nccl-cu12, same one PyTorch loads) goes first
_NCCL_LIB="$ENVROOT/venvs/vllm/lib/python3.12/site-packages/nvidia/nccl/lib"
export NCCL_LIB_DIR="$_NCCL_LIB"
export LD_LIBRARY_PATH="$_NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_TESTS="$_REPO/third_party/nccl-tests/build"
