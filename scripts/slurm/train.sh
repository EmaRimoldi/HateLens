#!/usr/bin/env bash
#SBATCH --job-name=hatelens-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/slurm-%j.out
#SBATCH --error=outputs/logs/slurm-%j.err
#
# Usage (from repo root, after tuning #SBATCH for your cluster):
#   export HATELENS_ROOT="$(pwd)"
#   mkdir -p outputs/logs
#   sbatch scripts/slurm/train.sh dynahate configs/models/tinyllama.yaml
#
# Args: <dataset> <config_yaml>

set -euo pipefail
DATASET="${1:-dynahate}"
CONFIG="${2:-configs/models/tinyllama.yaml}"
ROOT="${HATELENS_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p outputs/logs

if command -v uv >/dev/null 2>&1; then
  exec uv run hatelens train "$CONFIG" --dataset "$DATASET"
fi
exec .venv/bin/hatelens train "$CONFIG" --dataset "$DATASET"
