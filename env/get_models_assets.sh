#!/usr/bin/env bash
# Download the models and data used in M5 at the exact revisions, and rebuild assets/ (not in git).
#   Qwen/Qwen2.5-32B-Instruct           @ 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd   (~65 GB, M5 inference)
#   TinyLlama/TinyLlama-1.1B-Chat-v1.0  @ fe8a4ea1ffedaf415f4da2f062534de366a451e6   (~2 GB, M5 inference + tokenizer)
#   assets/c4_test/data.json            torchtitan v0.4.0 tests/assets/c4_test        (M5 training data)
#   assets/tokenizer_tinyllama/         tokenizer files copied from the TinyLlama snapshot (M5 training)
# Models go to the Hugging Face cache (~/.cache/huggingface/hub, or $HF_HOME). Only M5 needs them.
#
# Usage: bash env/get_models_assets.sh [--skip-qwen]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$ROOT/.venv-vllm/bin/python
[ -x "$PY" ] || { echo "run env/setup_python.sh vllm first" >&2; exit 1; }

download() {  # download <repo> <revision>
  "$PY" - "$1" "$2" <<'EOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2])
print("  ", sys.argv[1], "->", path)
EOF
}

[ "${1:-}" = "--skip-qwen" ] || download Qwen/Qwen2.5-32B-Instruct 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
download TinyLlama/TinyLlama-1.1B-Chat-v1.0 fe8a4ea1ffedaf415f4da2f062534de366a451e6

mkdir -p "$ROOT/assets/c4_test" "$ROOT/assets/tokenizer_tinyllama"
curl -sfL -o "$ROOT/assets/c4_test/data.json" \
  https://raw.githubusercontent.com/pytorch/torchtitan/v0.4.0/tests/assets/c4_test/data.json
SNAP=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', revision='fe8a4ea1ffedaf415f4da2f062534de366a451e6'))")
for f in tokenizer.json tokenizer_config.json special_tokens_map.json tokenizer.model; do
  [ -e "$SNAP/$f" ] && cp -L "$SNAP/$f" "$ROOT/assets/tokenizer_tinyllama/"
done
echo "assets:"; ls -la "$ROOT/assets/c4_test" "$ROOT/assets/tokenizer_tinyllama" | grep -v "^total"
