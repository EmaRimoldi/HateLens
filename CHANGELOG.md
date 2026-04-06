# Changelog

## 0.4.0 — 2026-04-06 (research framework extension)

### Added

- **Model registry** (`hatelens/registry.py`) and **PEFT factory** (`hatelens/peft_factory.py`): LoRA, QLoRA (optional `bitsandbytes`), DoRA / AdaLoRA / PiSSA with fallbacks.
- **Unified example schema** (`hatelens/schema.py`), **row mapping** (`hatelens/mapping.py`), **HateXplain loader stub** (`hatelens/loaders/hatexplain.py`).
- **Calibration metrics** (ECE, Brier), **consistency losses** (JS / KL utilities), **structured JSON parsing**, **policy preprompt** helper, **distillation cache** helpers.
- **Evaluation bundle** (`hatelens/evaluation_suite.py`) building on extended `classification_metrics` (**f1_binary**, **f1_macro**, macro precision/recall).
- **Training**: config hash + git HEAD logging; **smoke mode** (`smoke_test` / `HATELENS_SMOKE`); subset train/val/**test** before tokenization; `max_length` in YAML; Transformers 5 **`processing_class`** instead of `tokenizer`.
- **Configs**: `configs/models/tinyllama-legacy.yaml`, `configs/models/qwen2.5-1.5b.yaml`, `configs/smoke/tinyllama_dynahate.yaml`, `configs/experiments/README.md`.
- **Docs**: `docs/framework-audit-2026.md` (audit + migration + execution plan).
- **Tests**: metrics, mapping, parsing, calibration, consistency, registry; `tests/conftest.py` for BLAS thread limits.

### Changed

- **DynaHate** splits now retain `target`, `type`, `level` when present for structured supervision downstream.
- **`pyproject.toml`**: optional `hatelens[quant]` for QLoRA.

## 0.3.0 — 2026-04-05

### Changed (repository layout)

- **Removed** committed experiment artifacts: `results/`, `checkpoints/`, and old `experiments/` YAML tree.
- **Configs** live under `configs/models/*.yaml` (TinyLlama, Phi-2, OPT).
- **All generated files** go under `outputs/` (`runs/`, `logs/`, `eval/`, `lime/`) — gitignored except `outputs/.gitkeep`.
- **Scripts**: `scripts/train_*.sh`, `scripts/slurm/train.sh`, legacy Python entrypoints under `scripts/`.
- **Notebook**: `notebooks/plot_lime_barplots.ipynb` (paths aligned with `outputs/lime/`).
- **Default adapter paths** for eval/LIME: `outputs/runs/tinyllama/<dataset>/best_checkpoint` (override with env or `--adapter`).
- **Training** appends `--dataset` to `output_dir` and `logging_dir` so one YAML serves both DynaHate and HateCheck.

### Added

- `hatelens.paths.outputs_dir`; CLI flags `--adapter`, `--eval-output` where relevant.

## 0.2.0 — 2026-04-05

### Added

- Installable package `hatelens` (`src/hatelens/`) with CLI `hatelens`.
- Correct **PEFT adapter loading** + optional merge for inference (`hatelens.modeling`).
- **Unified training** pipeline with dataset flag (`hatelens train CONFIG --dataset dynahate|hatecheck`).
- **Batched evaluation** and CSV exports (`hatelens evaluate`).
- **HateCheck functionality breakdown** (`hatelens diagnose-hatecheck`).
- `uv` + `pyproject.toml`, pytest suite, Ruff; CI workflow template in `docs/github-actions-ci.yml` (copy to `.github/workflows/` when using a PAT with `workflow` scope).
- Documentation: `docs/architecture.md`, `docs/repo-audit.md`, `docs/related-work.md`, `docs/cluster-runbook.md`, `docs/experiments.md`.
- `cluster/sbatch_train.sh` SLURM template.

### Changed

- W&B logging is **opt-in** via `WANDB_ENABLED=1` (optional `hatelens[wandb]`).
- `run_training_*.sh` scripts are non-interactive and call `uv run hatelens train`.
- `utils/datasets.py` is a compatibility shim to `hatelens.datasets`.

### Fixed

- HateCheck CSV resolution no longer depends on process working directory alone (`data_dir()`).
- Post-fine-tune evaluation compatible with **adapter-only** checkpoints.

### Removed

- ~500 lines of commented dead code from legacy trainer scripts (replaced by `train_pipeline`).
