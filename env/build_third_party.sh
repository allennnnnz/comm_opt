#!/usr/bin/env bash
# Fetch and build the external tools at the exact commits and with the exact commands used in this project.
# Everything lands in third_party/ (not in git). Logs: third_party/<name>-*.log
#
#   nccl-tests        NVIDIA/nccl-tests @ afd59ab, MPI build, sm_80, against the system NCCL (/usr)      M0-M7
#   mscclpp           microsoft/mscclpp @ d5ce6c72, NCCL shim lib/libmscclpp_nccl.so (LD_PRELOAD)        M6, M7
#   low-latency-nccl  ss16118/low-latency-nccl branch siyshen/low-latency @ 5357eff (NCCL 2.29.2 fork)    M7
#   nccl-tests-llnccl nccl-tests @ afd59ab rebuilt against low-latency-nccl's headers                    M7
#   lmdeploy          InternLM/lmdeploy @ 3228395 + exp/M9_fused_baselines/lmdeploy_m9.patch,
#                     test_comm built twice: build/ (original dispatch), build_force/ (-DM9_FORCE_FUSED)  M9, M10
#
# Usage: bash env/build_third_party.sh [component ...]   (default: all, in the order above)
#        FETCH_ONLY=1 bash env/build_third_party.sh ...    (clone and check out, no build)
#        THIRD_PARTY=/other/dir bash env/build_third_party.sh ...
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TP=${THIRD_PARTY:-$ROOT/third_party}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
GENCODE="-gencode=arch=compute_80,code=sm_80"
JOBS=${JOBS:-32}
mkdir -p "$TP"

mpi_home() { dirname "$(dirname "$(readlink -f "$(command -v mpirun)")")"; }

fetch() {  # fetch <url> <dir> <commit> [branch]
  local url=$1 dir=$2 commit=$3 branch=${4:-}
  if [ ! -d "$TP/$dir/.git" ]; then
    if [ -n "$branch" ]; then git clone -q --branch "$branch" "$url" "$TP/$dir"; else git clone -q "$url" "$TP/$dir"; fi
  fi
  git -C "$TP/$dir" checkout -q "$commit"
  echo "   $dir at $(git -C "$TP/$dir" rev-parse --short HEAD)"
}

build_nccl_tests() {
  fetch https://github.com/NVIDIA/nccl-tests.git nccl-tests afd59ab
  [ -n "${FETCH_ONLY:-}" ] && return
  make -C "$TP/nccl-tests" -j"$JOBS" CUDA_HOME="$CUDA_HOME" NCCL_HOME=/usr MPI=1 MPI_HOME="$(mpi_home)" \
    NVCC_GENCODE="$GENCODE" > "$TP/nccl-tests-build.log" 2>&1
  ls "$TP/nccl-tests/build/all_reduce_perf" > /dev/null && echo "   built nccl-tests"
}

build_mscclpp() {
  fetch https://github.com/microsoft/mscclpp.git mscclpp d5ce6c72
  [ -n "${FETCH_ONLY:-}" ] && return
  mkdir -p "$TP/mscclpp/build"
  (cd "$TP/mscclpp/build" &&
    cmake -DCMAKE_BUILD_TYPE=Release -DMSCCLPP_GPU_ARCHS=80 -DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF -DMSCCLPP_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" .. > "$TP/mscclpp-cmake.log" 2>&1 &&
    make -j"$JOBS" > "$TP/mscclpp-build.log" 2>&1)
  ls "$TP/mscclpp/build/lib/libmscclpp_nccl.so" > /dev/null && echo "   built mscclpp"
}

build_low_latency_nccl() {
  fetch https://github.com/ss16118/low-latency-nccl.git low-latency-nccl 5357eff siyshen/low-latency
  [ -n "${FETCH_ONLY:-}" ] && return
  make -C "$TP/low-latency-nccl" -j"$JOBS" src.build CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$GENCODE" \
    > "$TP/llnccl-build.log" 2>&1
  ls "$TP/low-latency-nccl/build/lib/libnccl.so.2" > /dev/null && echo "   built low-latency-nccl"
}

build_nccl_tests_llnccl() {
  [ -d "$TP/nccl-tests/.git" ] || build_nccl_tests
  [ -d "$TP/low-latency-nccl/build" ] || [ -n "${FETCH_ONLY:-}" ] || build_low_latency_nccl
  rm -rf "$TP/nccl-tests-llnccl" && cp -r "$TP/nccl-tests" "$TP/nccl-tests-llnccl"
  git -C "$TP/nccl-tests-llnccl" checkout -q afd59ab
  [ -n "${FETCH_ONLY:-}" ] && return
  make -C "$TP/nccl-tests-llnccl" clean > /dev/null 2>&1 || true
  make -C "$TP/nccl-tests-llnccl" -j"$JOBS" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$TP/low-latency-nccl/build" MPI=1 \
    MPI_HOME="$(mpi_home)" NVCC_GENCODE="$GENCODE" > "$TP/nccl-tests-llnccl-build.log" 2>&1
  ls "$TP/nccl-tests-llnccl/build/all_reduce_perf" > /dev/null && echo "   built nccl-tests-llnccl"
}

build_lmdeploy() {
  fetch https://github.com/InternLM/lmdeploy.git lmdeploy 3228395546dfb2f6427faf39ec0c0c285f4c378a
  if git -C "$TP/lmdeploy" apply --check "$ROOT/exp/M9_fused_baselines/lmdeploy_m9.patch" 2>/dev/null; then
    git -C "$TP/lmdeploy" apply "$ROOT/exp/M9_fused_baselines/lmdeploy_m9.patch" && echo "   applied lmdeploy_m9.patch"
  else
    echo "   lmdeploy_m9.patch already applied (or does not apply cleanly; check git -C $TP/lmdeploy diff)"
  fi
  [ -n "${FETCH_ONLY:-}" ] && return
  local common=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80-real -DBUILD_TEST=ON -DBUILD_PY_FFI=OFF
                -DBUILD_MULTI_GPU=ON -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc")
  mkdir -p "$TP/lmdeploy/build" "$TP/lmdeploy/build_force"
  (cd "$TP/lmdeploy/build" && cmake .. "${common[@]}" > "$TP/lmdeploy-cmake.log" 2>&1 &&
    make -j"$JOBS" test_comm > "$TP/lmdeploy-build.log" 2>&1)
  (cd "$TP/lmdeploy/build_force" && cmake .. "${common[@]}" -DCMAKE_CUDA_FLAGS=-DM9_FORCE_FUSED \
    > "$TP/lmdeploy-force-cmake.log" 2>&1 && make -j"$JOBS" test_comm > "$TP/lmdeploy-force-build.log" 2>&1)
  ls "$TP/lmdeploy/build/bin/test_comm" "$TP/lmdeploy/build_force/bin/test_comm" > /dev/null && echo "   built lmdeploy test_comm (x2)"
}

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(nccl-tests mscclpp low-latency-nccl nccl-tests-llnccl lmdeploy)
for t in "${targets[@]}"; do
  echo "== $t"
  case $t in
    nccl-tests) build_nccl_tests ;;
    mscclpp) build_mscclpp ;;
    low-latency-nccl) build_low_latency_nccl ;;
    nccl-tests-llnccl) build_nccl_tests_llnccl ;;
    lmdeploy) build_lmdeploy ;;
    *) echo "unknown component $t" >&2; exit 1 ;;
  esac
done
