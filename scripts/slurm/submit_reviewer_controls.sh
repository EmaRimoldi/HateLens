#!/usr/bin/env bash
# Submit all reviewer-requested controls and multi-seed replicates.
#
# Adds:
#   1. Bin-FE        — FEAT_EXTR head, no aux losses (isolates head-path confound)
#   2. HX-no-rat     — HateXplain data, rationale loss disabled (isolates data vs. rationale)
#   3. Bin-C s42/s456 — Bin-C with seeds 42 and 456
#   4. Struct s42/s456 — Struct with seeds 42 and 456
#
# After ALL train jobs finish run eval:
#   sbatch -p $PART --gres=gpu:1 scripts/slurm/eval_bundle.sh
#
# Usage (from repo root):
#   export HATELENS_ROOT="$(pwd)"
#   mkdir -p outputs/logs
#   bash scripts/slurm/submit_reviewer_controls.sh
#
# Dry run:
#   ENGAGING_DRY_RUN=1 bash scripts/slurm/submit_reviewer_controls.sh
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

LOG=outputs/logs/reviewer-controls-submit.log
echo "Reviewer controls submit: partition=$PART gres=$GRES  $(date)" | tee "$LOG"

submit() {
  local dataset="$1"
  local cfg="$2"
  local label="$3"
  if [[ "${ENGAGING_DRY_RUN:-0}" == "1" ]]; then
    echo "  [dry] $label  dataset=$dataset  cfg=$cfg" | tee -a "$LOG"
    return 0
  fi
  local jid
  jid="$("${sbatch_base[@]}" scripts/slurm/train.sh "$dataset" "$cfg" | awk '{print $4}')"
  echo "  job=$jid  $label  dataset=$dataset  cfg=$cfg" | tee -a "$LOG"
}

echo "" | tee -a "$LOG"
echo "── Reviewer control 1: Bin-FE (FEAT_EXTR, no aux loss) ─────────────────" | tee -a "$LOG"
submit dynahate \
  configs/experiments/paper_matrix/bin_feat_extr.yaml \
  "Bin-FE"

echo "" | tee -a "$LOG"
echo "── Reviewer control 2: HX-no-rat (HX data, rat_loss=0) ─────────────────" | tee -a "$LOG"
submit dynahate_hatexplain \
  configs/experiments/paper_matrix/structured_hx_no_rationale.yaml \
  "HX-no-rat"

echo "" | tee -a "$LOG"
echo "── Multi-seed: Bin-C seed 42 ────────────────────────────────────────────" | tee -a "$LOG"
submit dynahate \
  configs/experiments/paper_matrix/tinyllama_binary_compare_seed42.yaml \
  "Bin-C-s42"

echo "── Multi-seed: Bin-C seed 456 ───────────────────────────────────────────" | tee -a "$LOG"
submit dynahate \
  configs/experiments/paper_matrix/tinyllama_binary_compare_seed456.yaml \
  "Bin-C-s456"

echo "" | tee -a "$LOG"
echo "── Multi-seed: Struct seed 42 ───────────────────────────────────────────" | tee -a "$LOG"
submit dynahate \
  configs/experiments/paper_matrix/structured_dynahate_seed42.yaml \
  "Struct-s42"

echo "── Multi-seed: Struct seed 456 ──────────────────────────────────────────" | tee -a "$LOG"
submit dynahate \
  configs/experiments/paper_matrix/structured_dynahate_seed456.yaml \
  "Struct-s456"

echo "" | tee -a "$LOG"
echo "All jobs submitted. Monitor with: squeue -u \$USER" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "After all train jobs finish, run:" | tee -a "$LOG"
echo "  sbatch -p \"$PART\" --gres=\"$GRES\" --cpus-per-task=\"$CPUS\" \\" | tee -a "$LOG"
echo "    --mem=48G --time=04:00:00 scripts/slurm/eval_bundle.sh" | tee -a "$LOG"
