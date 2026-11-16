#!/bin/bash
#SBATCH --account=MST114546
#SBATCH --partition=dev
#SBATCH --job-name=M20node
#SBATCH --dependency=singleton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=/home/allen0117/comm_opt/results/M20_env_snapshot/slurm/%x-%j.out
#SBATCH --error=/home/allen0117/comm_opt/results/M20_env_snapshot/slurm/%x-%j.err
# M20 single-node job: (1) snapshot + SHARP precheck, (2) NVLS checks a/b/c, (3) clock behaviour x5.
# Override partition / exclusions on the submit command line.
set -uo pipefail
ROOT=/home/allen0117/comm_opt
EXPD=$ROOT/exp/M20_env_snapshot
OUT=$ROOT/results/M20_env_snapshot/$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}
mkdir -p "$OUT"
source $ROOT/exp/lib/nano5_env.sh
source $EXPD/common.sh
NNODES=1; NPROCS_PER_NODE=8
BIN=$NCCL_TESTS/all_reduce_perf

{
  echo "job_id=$SLURM_JOB_ID job_name=$SLURM_JOB_NAME partition=$SLURM_JOB_PARTITION node=$(hostname) start=$(date -Is)"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD) dirty=$(git -C "$ROOT" status --porcelain | wc -l)"
  module list 2>&1
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.sm,clocks.max.sm,temperature.gpu --format=csv
  python -c "import torch;print('torch',torch.__version__,'nccl',torch.cuda.nccl.version())" 2>&1
  echo "NCCL_TESTS=$NCCL_TESTS"; echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
} > "$OUT/job_env.txt" 2>&1

# --- R13: nothing else on the GPUs ---
others=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
if [[ -n "$others" ]]; then echo "GPU processes found; aborting: $others" >&2; echo "ABORT: $others" >> "$OUT/job_env.txt"; exit 1; fi

### 1. snapshot (raw outputs, one file per command)
S=$OUT/snapshot; mkdir -p $S
snap() { local f=$1; shift; { echo "# cmd: $*"; "$@"; echo "# rc=$?"; } > "$S/$f" 2>&1; }
snapsh() { local f=$1; shift; { echo "# cmd: $*"; bash -c "$*"; echo "# rc=$?"; } > "$S/$f" 2>&1; }
snap nvidia-smi_q.txt nvidia-smi -q
snap nvidia-smi_topo.txt nvidia-smi topo -m
snap nvidia-smi_nvlink_s.txt nvidia-smi nvlink -s
snap nvidia-smi_nvlink_c.txt nvidia-smi nvlink -c
snap nvidia-smi_q_clock_perf_power.txt nvidia-smi -q -d CLOCK,PERFORMANCE,POWER
snapsh nvidia-smi_L_persistence_mig.txt 'nvidia-smi -L; nvidia-smi --query-gpu=index,persistence_mode,mig.mode.current,ecc.mode.current,vbios_version,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max --format=csv'
snap ibstat.txt ibstat
snap ibv_devinfo_v.txt ibv_devinfo -v
snap lscpu.txt lscpu
snap numactl_H.txt numactl -H
snap lspci_tv.txt lspci -tv
snapsh lspci_vv_ib_gpu.txt 'lspci -vv 2>/dev/null | grep -E "^[0-9a-f:.]+ |LnkCap:|LnkSta:" | grep -E -A2 "NVIDIA|Mellanox"'
snapsh lsmod_grep.txt 'lsmod | grep -E "nvidia|peermem|mlx|gdr"'
snap uname_a.txt uname -a
snapsh proc_cmdline.txt 'cat /proc/cmdline'
snapsh module_list.txt 'module list 2>&1'
snapsh clock_perm_readonly.txt 'id; ls -l /dev/nvidia0 /dev/nvidiactl 2>&1; nvidia-smi -q -d CLOCK | grep -iE "applications|default|locked|auto boost"; echo "(only reading; no -lgc/-ac attempted)"'
snapsh nccl_version.txt 'python -c "import torch;print(torch.cuda.nccl.version())"; ls -l $NCCL_LIB_DIR; strings $NCCL_LIB_DIR/libnccl.so.2 | grep -m3 -E "NCCL version|^2\.[0-9]+\.[0-9]+"'
snapsh fabric_grep.txt 'nvidia-smi -q | grep -iE -A6 "Fabric"'

### 1b. single-node SHARP precheck (P10 precursor)
SH=$OUT/sharp_precheck; mkdir -p $SH
{ echo "# cmd: ip -br addr | grep -i ib"; ip -br addr | grep -i ib; echo "# rc=$?"; } > $SH/ip_br_addr_ib.txt 2>&1
PLUG=$HPCX/nccl_rdma_sharp_plugin/lib/libnccl-net.so
{ echo "# ldd, LD_LIBRARY_PATH = venv NCCL + original"; ldd $PLUG; } > $SH/ldd_plugin_before.txt 2>&1
export HPCX
NEWLD=$NCCL_LIB_DIR:$HPCX/nccl_rdma_sharp_plugin/lib:$HPCX/sharp/lib:$HPCX/ucx/lib:${LD_LIBRARY_PATH}
{ echo "# ldd, LD_LIBRARY_PATH = $NEWLD"; LD_LIBRARY_PATH=$NEWLD ldd $PLUG; } > $SH/ldd_plugin_after.txt 2>&1
# first NDR (400) Active port from ibstat
IBDEV=$(ibstat 2>/dev/null | awk '/^CA /{gsub(/\x27/,"",$2); ca=$2} /Port [0-9]+:/{p=$2; sub(":","",p)} /State: Active/{act=1} /Rate: 400/{if(act&&!done){print ca":"p; done=1}} /Port GUID/{act=0}')
echo "IBDEV=$IBDEV" > $SH/ibdev_choice.txt
if [[ -z $IBDEV ]]; then echo "no NDR Active port parsed from ibstat" >> $SH/ibdev_choice.txt; IBDEV=mlx5_0:1; echo "fallback to mlx5_0:1" >> $SH/ibdev_choice.txt; fi
{ echo "# cmd: SHARP_COLL_LOG_LEVEL=3 sharp_hello -d $IBDEV -v 3 (timeout 60)"; LD_LIBRARY_PATH=$NEWLD SHARP_COLL_LOG_LEVEL=3 timeout 60 $HPCX/sharp/bin/sharp_hello -d $IBDEV -v 3; echo "# rc=$?"; } > $SH/sharp_hello.txt 2>&1
{ echo "# cmd: SHARP_COLL_ENABLE_SAT=1 SHARP_COLL_LOG_LEVEL=3 sharp_hello -d $IBDEV -v 3 (timeout 60)"; LD_LIBRARY_PATH=$NEWLD SHARP_COLL_ENABLE_SAT=1 SHARP_COLL_LOG_LEVEL=3 timeout 60 $HPCX/sharp/bin/sharp_hello -d $IBDEV -v 3; echo "# rc=$?"; } > $SH/sharp_hello_sat.txt 2>&1

### 2. NVLS checks
N=$OUT/nvls; mkdir -p $N
nvcc -arch=sm_90 $EXPD/mcast_query.cu -o $N/mcast_query -lcuda > $N/mcast_compile.txt 2>&1
echo "nvcc rc=$?" >> $N/mcast_compile.txt
[[ -x $N/mcast_query ]] && $N/mcast_query > $N/mcast_query.txt 2>&1 || echo "mcast_query not run (compile failed; see mcast_compile.txt)" > $N/mcast_query.txt
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,NVLS
SAVE_OUT=$OUT; OUT=$N
ARGS="-b 8 -e 64M -f 4 -g 1"
probe_launcher nvls_a_default 300 $BIN $ARGS
if [[ -n $LAUNCH_MODE ]]; then
  NCCL_ALGO=NVLS run_cfg nvls_b_algo_nvls 300 $BIN $ARGS
  NCCL_NVLS_ENABLE=0 run_cfg nvls_c_nvls_disabled 300 $BIN $ARGS
else
  echo "no launcher worked; NVLS b/c and clock rounds skipped" | tee -a $SAVE_OUT/summary.txt
fi
cp $N/summary.txt $SAVE_OUT/nvls_summary.txt 2>/dev/null
OUT=$SAVE_OUT
unset NCCL_DEBUG NCCL_DEBUG_SUBSYS

### 3. clock behaviour: calibrate -n, then 5 rounds
if [[ -n $LAUNCH_MODE ]]; then
  C=$OUT/clock; mkdir -p $C
  LARGS="-b 64M -e 64M -g 1 -c 0"
  timeout 300 bash -c "$(declare -f launch); NNODES=1 NPROCS_PER_NODE=8; launch $LAUNCH_MODE $BIN $LARGS -n 200 -w 20" > $C/calibration.log 2>&1
  # columns: ... time(oop)=$6 ... time(ip)=$10 ; iteration pair costs t_oop+t_ip when -n applies to both
  CAL=$(grep -E '^ +[0-9]+ +[0-9]+ +float +sum' $C/calibration.log | awk '{print $6" "$10}')
  NITER=$(echo "$CAL" | awk '{t=$1+$2; if(t>0) printf "%d", 60e6/t; else print 100000}')
  [[ -z $NITER || $NITER -lt 1 ]] && NITER=100000
  echo "calibration: oop_us ip_us = $CAL ; n=$NITER (60 s / (oop+ip))" | tee $C/calibration_result.txt
  for r in 1 2 3 4 5; do
    nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv -lms 100 > $C/round${r}_samples.csv 2> $C/round${r}_samples.err &
    SP=$!
    sleep 10
    T0=$(date +%s.%N); T0I=$(date '+%Y/%m/%d %H:%M:%S.%3N')
    timeout 200 bash -c "$(declare -f launch); NNODES=1 NPROCS_PER_NODE=8; launch $LAUNCH_MODE $BIN $LARGS -n $NITER -w 5" > $C/round${r}_load.log 2>&1
    RC=$?
    T1=$(date +%s.%N); T1I=$(date '+%Y/%m/%d %H:%M:%S.%3N')
    sleep 3
    kill $SP 2>/dev/null; wait $SP 2>/dev/null
    echo "round=$r load_start_epoch=$T0 load_end_epoch=$T1 load_start=$T0I load_end=$T1I rc=$RC n=$NITER sampler_alive=$(kill -0 $SP 2>/dev/null && echo yes || echo no)" | tee -a $C/marks.txt
  done
  pgrep -u $USER -a nvidia-smi >> $C/leftover_check.txt 2>&1 || echo "no nvidia-smi left" >> $C/leftover_check.txt
fi
echo "end=$(date -Is)" >> "$OUT/job_env.txt"
echo "done: $OUT"
