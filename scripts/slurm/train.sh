#!/usr/bin/env bash
#SBATCH --job-name=hatelens-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# Align with agent_swarms-style jobs (see configs/experiment_opus.yaml: partition + single GPU).
#SBATCH --partition=pi_tpoggio
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/slurm-%j.out
#SBATCH --error=outputs/logs/slurm-%j.err
#
# MIT Engaging (ORCD): partition/GPU examples — see
# https://mabdel-03.github.io/Engaging-ModdedGPT-Blog/getting-started/
#   pi_tpoggio + A100:  sbatch -p pi_tpoggio --gres=gpu:a100:1 scripts/slurm/train.sh ...
#   general pool:       sbatch -p mit_normal_gpu --gres=gpu:h100:1 ...
# CLI flags override these #SBATCH lines.
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

# Full GPU runs: login shells often export HATELENS_SMOKE=1 from docs/CI; that caps steps.
export HATELENS_SMOKE=0

if command -v uv >/dev/null 2>&1; then
  exec uv run hatelens train "$CONFIG" --dataset "$DATASET"
fi
exec .venv/bin/hatelens train "$CONFIG" --dataset "$DATASET"
