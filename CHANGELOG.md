# Changelog

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
