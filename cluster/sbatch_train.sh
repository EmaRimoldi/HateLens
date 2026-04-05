#!/usr/bin/env bash
#SBATCH --job-name=hatelens-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# One-shot training. Usage:
#   export HATELENS_ROOT=/path/to/HateLens && cd "$HATELENS_ROOT" && mkdir -p logs
#   sbatch cluster/sbatch_train.sh dynahate experiments/TinyLlama/config.yaml
# Args: <dataset> <config_yaml>

set -euo pipefail
DATASET="${1:-dynahate}"
CONFIG="${2:-experiments/TinyLlama/config.yaml}"
ROOT="${HATELENS_ROOT:-$(pwd)}"
cd "$ROOT"
mkdir -p logs

if command -v uv >/dev/null 2>&1; then
  exec uv run hatelens train "$CONFIG" --dataset "$DATASET"
fi
exec .venv/bin/hatelens train "$CONFIG" --dataset "$DATASET"
