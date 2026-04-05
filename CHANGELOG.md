# Changelog

## 0.2.0 — 2026-04-05

### Added

- Installable package `hatelens` (`src/hatelens/`) with CLI `hatelens`.
- Correct **PEFT adapter loading** + optional merge for inference (`hatelens.modeling`).
- **Unified training** pipeline with dataset flag (`hatelens train CONFIG --dataset dynahate|hatecheck`).
- **Batched evaluation** and CSV exports (`hatelens evaluate`).
- **HateCheck functionality breakdown** (`hatelens diagnose-hatecheck`).
- `uv` + `pyproject.toml`, pytest suite, Ruff, GitHub Actions CI.
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
