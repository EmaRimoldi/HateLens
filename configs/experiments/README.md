# Experiment matrix (follow-up paper)

Configs are grouped by letter (see project brief). Legacy behaviour remains under `configs/models/tinyllama-legacy.yaml` and original `tinyllama.yaml`.

| Group | Purpose |
|-------|---------|
| A | Legacy replication (TinyLlama / Phi-2 / OPT + DynaHate / HateCheck) |
| B | Modern small models + LoRA / QLoRA / DoRA |
| C–H | Structured supervision, rationale, consistency, cross-dataset, efficiency, policy pilot |

Use `peft_type` (`lora`, `dora`, `qlora`, …) and `quantization` (`none`, `4bit`, `8bit`) in YAML. Run:

`uv run hatelens train configs/models/<config>.yaml --dataset dynahate`

HateCheck is **evaluation / functional robustness** in this codebase; training on `hatecheck` remains supported for legacy parity but is not recommended as a primary objective for the new framework.

## SLURM: train chain + eval bundle

From repo root (tune `#SBATCH` in scripts for your site):

```bash
export HATELENS_ROOT="$(pwd)"
mkdir -p outputs/logs
J1=$(sbatch --parsable scripts/slurm/train.sh dynahate configs/models/tinyllama-structured.yaml)
J2=$(sbatch --parsable --dependency=afterok:$J1 scripts/slurm/train.sh dynahate_hatexplain configs/models/tinyllama-structured.yaml)
sbatch --dependency=afterok:$J2 scripts/slurm/eval_bundle.sh
```

`scripts/slurm/eval_bundle.sh` runs `eval-run` for binary DynaHate/HateEval checkpoints and for structured runs **if** `outputs/runs/tinyllama/structured_*/best_checkpoint` exist (skips missing dirs). See also `configs/experiments/paper_matrix/README.md`.

**Hang dopo il save + dipendenze Slurm:** se fai `scancel` sul training perché il job resta `RUNNING` senza log, i job con `--dependency=afterok:…` **non partono** (il padre non è `COMPLETED`). Rilancia a mano il prossimo step, es. `sbatch scripts/slurm/train.sh …` oppure solo la valutazione: `sbatch scripts/slurm/eval_bundle.sh`.

**Parallel matrix on MIT Engaging** (partitions, `sbatch`, `pi_tpoggio`, `mit_normal_gpu`): [Engaging quick-start](https://mabdel-03.github.io/Engaging-ModdedGPT-Blog/getting-started/). Submit many trainings at once (tune `SLURM_PARTITION` / `SLURM_GRES`; use `ENGAGING_DRY_RUN=1` to print commands only):

```bash
export HATELENS_ROOT="$(pwd)" SLURM_PARTITION=pi_tpoggio SLURM_GRES=gpu:a100:1
bash scripts/slurm/submit_matrix_engaging.sh
```
