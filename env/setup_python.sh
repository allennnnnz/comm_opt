#!/usr/bin/env bash
# Recreate the three Python environments with the exact package versions used in this project.
#   .venv        analysis (pandas, matplotlib, ...)        <- requirements-analysis.txt
#   .venv-vllm   vLLM 0.33.0, torch 2.14.0 (CUDA 13.0)     <- requirements-vllm.txt   (M5 inference, M6-M18)
#   .venv-titan  torchtitan 0.4.0, torch 2.15.0            <- requirements-titan.txt  (M5 training)
# The requirement files are `uv pip freeze` output of the environments that produced the results.
#
# Usage: bash env/setup_python.sh [analysis|vllm|titan ...]     (default: all three)
#        CHECK_ONLY=1 bash env/setup_python.sh                    (resolve only, install nothing)
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "uv not found; install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

declare -A DIR=([analysis]=.venv [vllm]=.venv-vllm [titan]=.venv-titan)
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(analysis vllm titan)

for t in "${targets[@]}"; do
  dir=${DIR[$t]:?unknown environment $t (analysis, vllm, titan)}
  req=requirements-$t.txt
  if [ -n "${CHECK_ONLY:-}" ]; then
    tmp=$(mktemp -d)
    uv venv -q "$tmp/venv" --python 3.12
    echo "== $t: resolving $req"
    uv pip install --dry-run -q --python "$tmp/venv/bin/python" -r "$req" && echo "   ok: all pinned versions resolvable"
    rm -rf "$tmp"
    continue
  fi
  if [ -d "$dir" ]; then
    echo "== $t: $dir already exists, skipped (delete it to rebuild)"
    continue
  fi
  echo "== $t: creating $dir from $req"
  uv venv -q "$dir" --python 3.12
  uv pip install -q --python "$dir/bin/python" -r "$req"
done
