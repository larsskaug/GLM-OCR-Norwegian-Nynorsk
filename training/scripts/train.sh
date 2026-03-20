#!/usr/bin/env bash
# Run from the repo root: bash training/scripts/train.sh [lora|full] [gpu_ids]
# Examples:
#   bash training/scripts/train.sh lora 0
#   bash training/scripts/train.sh full 0,1

set -euo pipefail

MODE="${1:-lora}"
GPUS="${2:-0}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA_FACTORY="$REPO_ROOT/LLaMA-Factory"
CONFIG="$REPO_ROOT/training/config/glm_ocr_nynorsk_${MODE}.yaml"

if [[ ! -f "$CONFIG" ]]; then
  echo "Unknown mode '$MODE'. Use 'lora' or 'full'." >&2
  exit 1
fi

cd "$LLAMA_FACTORY"

DISABLE_VERSION_CHECK=1 PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$GPUS" \
  llamafactory-cli train "$CONFIG"
