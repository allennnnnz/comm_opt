#!/bin/bash
# Per-rank wrapper: Slurm overwrites CUDA_VISIBLE_DEVICES, so select the GPUs here (after Slurm set its own).
# GPU_LIST (e.g. 0,1,2,3) comes from the environment of the launch; the actual list is also echoed once per node.
[[ -n "${GPU_LIST:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU_LIST"
if [[ "${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}" == 0 ]]; then
  echo "RANKWRAP host=$(hostname -s) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
fi
exec "$@"
