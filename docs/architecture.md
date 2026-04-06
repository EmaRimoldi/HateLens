# Architecture

## Overview

HateLens fine-tunes **decoder-only** language models for **binary sequence classification** (hate vs not) using **PEFT LoRA** on attention projections (`k_proj`, `v_proj` by default), with optional **merged adapters** for fast CPU/GPU inference.

```text
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│ DynaHate CSV    │     │ HF AutoModel     │     │ HF Trainer           │
│ HateCheck split │────▶│ + LoRA (PEFT)    │────▶│ outputs/runs/...     │
└─────────────────┘     └──────────────────┘     └──────────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
  hatelens.datasets      hatelens.train_pipeline    best_checkpoint/ (adapter)
                         hatelens.modeling          + trainer checkpoints
```

## Packages

| Module | Role |
|--------|------|
| `hatelens.datasets` | Build `DatasetDict` for DynaHate / HateCheck (+ metadata split for diagnostics) |
| `hatelens.paths` | `repo_root()`, `data_dir()`, `outputs_dir()` |
| `hatelens.modeling` | Load hub or local checkpoints; **detect PEFT adapters** and merge for inference |
| `hatelens.train_pipeline` | Single training path for both datasets; W&B **opt-in** (`WANDB_ENABLED=1`) |
| `hatelens.evaluation` | **Batched** inference + sklearn metrics |
| `hatelens.diagnostics` | HateCheck **per-functionality** accuracy/F1 table |
| `hatelens.lime_scores` | Optional LIME word weights (requires `[lime]` extra) |
| `hatelens.cli` | `hatelens evaluate`, `train`, `lime`, `diagnose-hatecheck` |

## Data flow

- **DynaHate**: `data/DynaHate/dynahate_v0.2.3.csv` — columns `text`, `label`, `split`.
- **HateCheck**: `data/hatecheck/hatecheck_split.csv` — stratified train/val/test with `test_case`, `label`, and metadata (`functionality`, …).

Repository root is resolved from `hatelens.paths.repo_root()` (or `HATELENS_ROOT`).

## Configs and outputs

- **YAML**: `configs/models/*.yaml` — passed to `hatelens train <path> --dataset …`.
- **Training**: `outputs/runs/<model>/<dataset>/` (HF `TrainingArguments.output_dir` + `best_checkpoint/` subdir for final adapter save).
- **Eval / LIME**: `outputs/eval/`, `outputs/lime/` (gitignored except `outputs/.gitkeep`).

## Compatibility

Legacy `utils/datasets.py` re-exports `hatelens.datasets` when `src/` is on `sys.path`. Thin wrappers live under `scripts/` (`evaluate_models.py`, `compute_lime_scores.py`, `trainer_*.py`).
