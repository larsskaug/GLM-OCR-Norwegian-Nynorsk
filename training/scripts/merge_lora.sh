#!/usr/bin/env bash
# Merges LoRA adapter weights back into the base model.
# Run from the repo root: bash training/scripts/merge_lora.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA_FACTORY="$REPO_ROOT/LLaMA-Factory"

cd "$LLAMA_FACTORY"

llamafactory-cli export \
  --model_name_or_path zai-org/GLM-OCR \
  --adapter_name_or_path saves/glm-ocr-nynorsk/lora/sft \
  --template glm_ocr \
  --export_dir saves/glm-ocr-nynorsk/lora/sft/merged \
  --trust_remote_code true
