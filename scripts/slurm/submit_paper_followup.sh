#!/usr/bin/env bash
# Submit only paper follow-up jobs (ablations + PEFT sweep + isolated binary compare).
# Does not retrain legacy Group 1 or the main structured_dynahate / dynahate_hatexplain runs.
#
# Usage (repo root):
#   export HATELENS_ROOT="$(pwd)"
#   mkdir -p outputs/logs
#   bash scripts/slurm/submit_paper_followup.sh
#
# Dry run:
#   ENGAGING_DRY_RUN=1 bash scripts/slurm/submit_paper_followup.sh
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

echo "Paper follow-up submits: partition=$PART gres=$GRES" | tee outputs/logs/paper-followup-submit.log

submit() {
  local dataset="$1"
  local cfg="$2"
  if [[ "${ENGAGING_DRY_RUN:-0}" == "1" ]]; then
    echo "  [dry] ${sbatch_base[*]} scripts/slurm/train.sh $dataset $cfg" | tee -a outputs/logs/paper-followup-submit.log
    return 0
  fi
  local jid
  jid="$("${sbatch_base[@]}" scripts/slurm/train.sh "$dataset" "$cfg" | awk '{print $4}')"
  echo "  job=$jid dataset=$dataset config=$cfg" | tee -a outputs/logs/paper-followup-submit.log
}

submit dynahate configs/experiments/paper_matrix/structured_dynahate_no_rationale.yaml
submit dynahate configs/experiments/paper_matrix/structured_dynahate_consistency.yaml
submit dynahate configs/experiments/paper_matrix/tinyllama_binary_compare.yaml
submit dynahate configs/experiments/paper_matrix/tinyllama_peft_lora.yaml
submit dynahate configs/experiments/paper_matrix/tinyllama_qlora.yaml
submit dynahate configs/experiments/paper_matrix/tinyllama_dora.yaml

echo "After jobs finish, run eval bundle (avoid afterok if you scancel train jobs):"
echo "  sbatch -p \"$PART\" --gres=\"$GRES\" --cpus-per-task=\"$CPUS\" --mem=\"${SLURM_EVAL_MEM:-48G}\" --time=\"${SLURM_EVAL_TIME:-04:00:00}\" scripts/slurm/eval_bundle.sh"
