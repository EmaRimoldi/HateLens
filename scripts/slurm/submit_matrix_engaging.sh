#!/usr/bin/env bash
# Submit multiple HateLens training jobs in parallel on MIT Engaging (Slurm).
#
# Quick-start context: https://mabdel-03.github.io/Engaging-ModdedGPT-Blog/getting-started/
#
# Override queue/GPU via environment (defaults match Poggio lab pool from that guide):
#   SLURM_PARTITION=pi_tpoggio          # or mit_normal_gpu, ou_bcs_normal, mit_quicktest, …
#   SLURM_GRES=gpu:1                    # or gpu:a100:1, gpu:h100:1 per site and partition
#   SLURM_CPUS_PER_TASK=8 SLURM_MEM=64G SLURM_TIME=08:00:00
#
# Optional CUDA modules (Engaging module stack) before batch jobs — load manually or uncomment in batch:
#   module load cuda/12.4.0 cudnn/9.8.0.87-cuda12
#
# Usage (repo root):
#   export HATELENS_ROOT="$(pwd)"
#   mkdir -p outputs/logs
#   bash scripts/slurm/submit_matrix_engaging.sh
#
# Print commands only (no sbatch):
#   ENGAGING_DRY_RUN=1 bash scripts/slurm/submit_matrix_engaging.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export HATELENS_ROOT="${HATELENS_ROOT:-$ROOT}"
mkdir -p outputs/logs

PART="${SLURM_PARTITION:-pi_tpoggio}"
GRES="${SLURM_GRES:-gpu:1}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
MEM="${SLURM_MEM:-64G}"
TIME="${SLURM_TIME:-08:00:00}"

sbatch_base=(
  sbatch
  -p "$PART"
  --gres="$GRES"
  --cpus-per-task="$CPUS"
  --mem="$MEM"
  --time="$TIME"
)

echo "Submitting parallel jobs: partition=$PART gres=$GRES" | tee outputs/logs/engaging-matrix-submit.log

run_id=0
submit() {
  local dataset="$1"
  local cfg="$2"
  run_id=$((run_id + 1))
  if [[ "${ENGAGING_DRY_RUN:-0}" == "1" ]]; then
    echo "  [dry $run_id] ${sbatch_base[*]} scripts/slurm/train.sh $dataset $cfg" | tee -a outputs/logs/engaging-matrix-submit.log
    return 0
  fi
  local jid
  jid="$("${sbatch_base[@]}" scripts/slurm/train.sh "$dataset" "$cfg" | awk '{print $4}')"
  echo "  [$run_id] job=$jid dataset=$dataset config=$cfg" | tee -a outputs/logs/engaging-matrix-submit.log
}

# --- Group 1: legacy binary ---
submit dynahate configs/models/tinyllama-legacy.yaml
submit hateeval configs/models/tinyllama-legacy.yaml

# --- Group 2: binary vs structured baseline ---
submit dynahate configs/experiments/paper_matrix/tinyllama_binary_compare.yaml
submit dynahate configs/models/tinyllama-structured.yaml

# --- Groups 3–4: rationale / consistency ablations ---
submit dynahate configs/experiments/paper_matrix/structured_dynahate_no_rationale.yaml
submit dynahate configs/experiments/paper_matrix/structured_dynahate_consistency.yaml

# --- Combined supervision (requires HateXplain/HF cache on node) ---
submit dynahate_hatexplain configs/models/tinyllama-structured.yaml

# --- Group 6 — PEFT sanity (LoRA vs QLoRA vs DoRA), outputs isolated per config ---
submit dynahate configs/experiments/paper_matrix/tinyllama_peft_lora.yaml
submit dynahate configs/experiments/paper_matrix/tinyllama_qlora.yaml
submit dynahate configs/experiments/paper_matrix/tinyllama_dora.yaml

echo "Done. Monitor: squeue -u \"$USER\""
echo "After checkpoints exist, eval bundle (same partition/GRES):"
echo "  sbatch -p \"$PART\" --gres=\"$GRES\" --cpus-per-task=\"$CPUS\" --mem=\"$MEM\" --time=\"$TIME\" scripts/slurm/eval_bundle.sh"
