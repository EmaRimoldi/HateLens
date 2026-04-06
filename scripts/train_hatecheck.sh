#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/models/tinyllama.yaml}"
exec uv run hatelens train "$CONFIG" --dataset hatecheck
