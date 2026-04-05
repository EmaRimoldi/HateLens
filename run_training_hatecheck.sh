#!/usr/bin/env bash
# LoRA fine-tuning on HateCheck (non-interactive; override CONFIG with first arg).
set -euo pipefail
cd "$(dirname "$0")"
CONFIG="${1:-experiments/TinyLlama/config.yaml}"
exec uv run hatelens train "$CONFIG" --dataset hatecheck
