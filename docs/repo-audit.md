# Repository audit (baseline → v0.2)

## Issues found in the original tree

1. **Incorrect post-FT loading**: Checkpoints are PEFT adapters; `AutoModelForSequenceClassification.from_pretrained(adapter_dir)` does not reconstruct LoRA fine-tuned weights. **Fixed** in `hatelens.modeling.load_sequence_classifier` (base + `PeftModel` + optional `merge_and_unload()`).
2. **Massive commented-out code** in `trainer_dynahate.py` / `trainer_hatecheck.py` (~250 lines each). **Removed**; training lives in `hatelens.train_pipeline`.
3. **Hard-coded W&B entity** and implicit W&B dependency for training. **Replaced** with opt-in `WANDB_ENABLED=1` and optional `pip install 'hatelens[wandb]'`.
4. **HateCheck dataset path**: `create_hatecheck_dataset` used `./data/hatecheck/...` cwd-relative. **Fixed** with `data_dir()` relative to repo root.
5. **No installable package / no tests / no CI**. **Added** `pyproject.toml`, `pytest`, GitHub Actions, `uv` workflow.
6. **LIME not in pinned dependencies** while scripts import `lime`. **Documented** as extra `hatelens[lime]`.
7. **Duplicate / messy imports** in `utils/datasets.py`. **Centralized** in `hatelens.datasets` + thin shim.
8. **Sequential per-example evaluation** in `evaluate_models.py` (slow). **Replaced** with batched `predict_batch` in `hatelens.evaluation`.
9. **Jupyter checkpoints** committed under `experiments/**/.ipynb_checkpoints`. **Removed** and **gitignored**.

## What was preserved

- Data files, experiment YAML layout, checkpoint artifacts, and high-level training hyperparameters.
- DynaHate / HateCheck semantics (binary labels, text fields).

## Risks / follow-ups

- Full `hatelens evaluate` downloads the base TinyLlama weights (~2GB+) and needs sufficient RAM/VRAM.
- `phi-2` / `OPT` configs may need different `target_modules` or batch sizes on your hardware; not revalidated in this pass.
