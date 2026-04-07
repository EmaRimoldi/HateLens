#!/usr/bin/env bash
#SBATCH --job-name=hatelens-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --partition=pi_tpoggio
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/slurm-eval-%j.out
#SBATCH --error=outputs/logs/slurm-eval-%j.err
#
# Engaging queues: https://mabdel-03.github.io/Engaging-ModdedGPT-Blog/getting-started/
#
# Run after structured training checkpoints exist. Adjust paths if your run dirs differ.
# Optional: sbatch --dependency=afterok:<train_jobid> scripts/slurm/eval_bundle.sh
#
set -euo pipefail
ROOT="${HATELENS_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p outputs/logs outputs/eval_runs

export HATELENS_SMOKE=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

run_eval() {
  local ckpt="$1"
  local name="$2"
  shift 2
  if [[ ! -d "$ckpt" ]]; then
    echo "SKIP (missing): $ckpt" | tee -a outputs/logs/slurm-eval-last.skip
    return 0
  fi
  uv run hatelens eval-run --checkpoint "$ckpt" --run-name "$name" "$@"
}

# Binary baselines (Group 1)
run_eval outputs/runs/tinyllama/dynahate/best_checkpoint exp_g1_binary_dynahate \
  --in-domain dynahate hateeval --cross dynahate:hateeval hateeval:dynahate --hatecheck

run_eval outputs/runs/tinyllama/hateeval/best_checkpoint exp_g1_binary_hateeval \
  --in-domain dynahate hateeval --cross dynahate:hateeval hateeval:dynahate --hatecheck

# Structured (Group 2+) when present
run_eval outputs/runs/tinyllama/structured_dynahate/best_checkpoint exp_g2_structured_dynahate \
  --in-domain dynahate hateeval --cross dynahate:hateeval hateeval:dynahate --hatecheck --rationale --rationale-max-samples 128

run_eval outputs/runs/tinyllama/structured_dynahate_hatexplain/best_checkpoint exp_structured_dh_hx \
  --in-domain dynahate hateeval --cross dynahate_hatexplain:hateeval --hatecheck --rationale --rationale-max-samples 128

# Ablations (Group 3–4): separate checkpoint dirs via structured_output_suffix
run_eval outputs/runs/tinyllama/structured_dynahate_no_rationale/best_checkpoint exp_ablation_no_rationale \
  --in-domain dynahate hateeval --cross dynahate:hateeval hateeval:dynahate --hatecheck --rationale --rationale-max-samples 128

run_eval outputs/runs/tinyllama/structured_dynahate_consistency/best_checkpoint exp_ablation_consistency \
  --in-domain dynahate hateeval --cross dynahate:hateeval hateeval:dynahate --hatecheck --rationale --rationale-max-samples 128

# Group 2b + 6 (optional paper matrix)
run_eval outputs/runs/tinyllama_binary_compare/dynahate/best_checkpoint exp_g2_binary_compare_dynahate \
  --in-domain dynahate hateeval --cross dynahate:hateeval hateeval:dynahate --hatecheck

run_eval outputs/runs/tinyllama_peft_lora/dynahate/best_checkpoint exp_g6_peft_lora \
  --in-domain dynahate hateeval --cross dynahate:hateeval hateeval:dynahate --hatecheck

run_eval outputs/runs/tinyllama_qlora/dynahate/best_checkpoint exp_g6_peft_qlora \
  --in-domain dynahate hateeval --cross dynahate:hateeval hateeval:dynahate --hatecheck

run_eval outputs/runs/tinyllama_dora/dynahate/best_checkpoint exp_g6_peft_dora \
  --in-domain dynahate hateeval --cross dynahate:hateeval hateeval:dynahate --hatecheck

echo "Eval bundle finished job $SLURM_JOB_ID"
