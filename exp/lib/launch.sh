#!/bin/bash
# launch.sh <mode> <nnodes> <ppn> <dist block|cyclic> <nodes all|a,b> <cmd...>  -- one GPU per rank; execs the launcher so signals/timeouts reach it.
mode=$1 nn=$2 ppn=$3 dist=$4 nodes=$5; shift 5
LIB=$(cd "$(dirname "$0")" && pwd)
nl=(); [[ $nodes != all ]] && nl=(--nodelist=$nodes)
case $mode in
  pmix_v4|pmi2) exec srun --mpi=$mode -N $nn --ntasks-per-node=$ppn --gpus-per-node=8 --distribution=$dist "${nl[@]}" bash $LIB/rankwrap.sh "$@" ;;
  mpirun)
    xs=(); rankby=slot
    for v in LD_LIBRARY_PATH NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_ALGO NCCL_PROTO NCCL_NVLS_ENABLE NCCL_COLLNET_ENABLE NCCL_NET_GDR_LEVEL GPU_LIST; do
      [[ -n ${!v+x} ]] && xs+=(-x $v)
    done
    [[ $dist == cyclic ]] && rankby=node
    hostarg=(); [[ $nodes != all ]] && hostarg=(--host "$(echo $nodes | sed "s/,/:$ppn,/g"):$ppn")
    exec mpirun -np $((nn*ppn)) --map-by ppr:$ppn:node --rank-by $rankby "${hostarg[@]}" "${xs[@]}" bash $LIB/rankwrap.sh "$@" ;;
  *) echo "unknown mode $mode" >&2; exit 2 ;;
esac
