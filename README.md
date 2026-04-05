# HateLens

**Tiny decoder LMs + LoRA for efficient, explainable hate speech detection** on [DynaHate](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset) and [HateCheck](https://github.com/paul-rottger/hatecheck-data).

## Why this matters

HateLens packages the full loop: **PEFT fine-tuning**, **correct LoRA inference** (merge adapters for speed), **batched evaluation**, optional **LIME** word attributions, and **HateCheck functionality diagnostics** (per test-type metrics). The repository keeps **data and configs** in Git; **runs, plots, and checkpoints** live under `outputs/` (gitignored except `.gitkeep`).

---

## Repository layout

| Path | Role |
|------|------|
| `src/hatelens/` | Python package and CLI |
| `configs/models/*.yaml` | LoRA + trainer hyperparameters |
| `data/` | DynaHate CSV and HateCheck splits (versioned) |
| `outputs/` | **All generated files** (training, eval, LIME) — not committed |
| `scripts/` | Shell helpers and legacy Python entrypoints |
| `scripts/slurm/train.sh` | SLURM template |
| `notebooks/` | Example plots (e.g. LIME barplots) |
| `docs/` | Architecture, experiments, cluster runbook |
| `tests/` | `pytest` |

---

## Install

**Python 3.10+**. Recommended: [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/EmaRimoldi/HateLens.git
cd HateLens
uv sync
uv sync --extra lime    # optional: LIME explainability
uv sync --extra wandb   # optional: training logging
```

Set `HATELENS_ROOT` to the repo root if you run commands from another directory.

---

## Using the framework

### 1. Train (LoRA)

Pick a config under `configs/models/` and a dataset:

```bash
uv run hatelens train configs/models/tinyllama.yaml --dataset dynahate
uv run hatelens train configs/models/tinyllama.yaml --dataset hatecheck
```

**Outputs**

- Checkpoints: `outputs/runs/<model>/<dataset>/best_checkpoint/` (PEFT adapter + tokenizer files)
- HF Trainer checkpoints + TensorBoard logs: `outputs/runs/<model>/<dataset>/` and `outputs/logs/<model>/<dataset>/`

**Weights & Biases** (optional):

```bash
export WANDB_ENABLED=1
export WANDB_ENTITY=your-team   # optional
uv run hatelens train configs/models/tinyllama.yaml --dataset hatecheck
```

**Shell shortcuts**

```bash
./scripts/train_dynahate.sh
./scripts/train_hatecheck.sh configs/models/phi-2.yaml
```

### 2. Evaluate (pre- vs post-FT)

Uses the base model from Hugging Face and, if present, your fine-tuned adapter.

Default adapter paths (after TinyLlama training):

- `outputs/runs/tinyllama/dynahate/best_checkpoint`
- `outputs/runs/tinyllama/hatecheck/best_checkpoint`

Override with `--adapter` or `HATELENS_CKPT_DYNAHATE` / `HATELENS_CKPT_HATECHECK`.

```bash
uv run hatelens evaluate --hatecheck --batch-size 16
uv run hatelens evaluate --dynahate --plots --adapter /path/to/best_checkpoint
```

**Writes** `outputs/eval/<dataset>/metrics_summary.csv` (and optional `comparison_bar.png`).

### 3. HateCheck diagnostics (by functionality)

Requires a trained HateCheck adapter:

```bash
uv run hatelens diagnose-hatecheck --batch-size 16
# or
uv run hatelens diagnose-hatecheck --adapter outputs/runs/tinyllama/hatecheck/best_checkpoint
```

**Writes** `outputs/eval/hatecheck/functionality_breakdown.csv`.

### 4. LIME (optional extra)

```bash
uv run hatelens lime --hatecheck
```

**Writes** pickles under `outputs/lime/` (e.g. `positive_pre_FT_hatecheck.pkl`). Plot them with `notebooks/plot_lime_barplots.ipynb`.

### 5. Cluster (SLURM)

See `docs/cluster-runbook.md`. Template:

```bash
export HATELENS_ROOT="$(pwd)"
mkdir -p outputs/logs
sbatch scripts/slurm/train.sh hatecheck configs/models/tinyllama.yaml
```

Edit `#SBATCH` headers in `scripts/slurm/train.sh` for your partition, GPU, walltime, and memory.

---

## Legacy scripts

`scripts/evaluate_models.py`, `scripts/compute_lime_scores.py`, and `scripts/trainer_*.py` forward to the same CLI. Prefer `uv run hatelens …`.

---

## Configuration

YAML fields are documented inline in `configs/models/*.yaml`. Common knobs: `model_checkpoint`, `r`, `lora_alpha`, `target_modules`, `output_dir` / `logging_dir` (per-dataset subfolders are appended automatically).

---

## Authors (original project)

| Name | SCIPER |
|------|--------|
| [Vittoria Meroni](https://github.com/vittoriameroni) | 386722 |
| [Emanuele Rimoldi](https://github.com/EmaRimoldi) | 377013 |
| [Simone Vicentini](https://github.com/SimoVice/) | 378204 |

## Data citations

- Vidgen et al., EMNLP 2021 — DynaHate.
- Röttger et al., ACL 2021 — HateCheck.

## License

Apache-2.0 (`LICENSE`). Dataset licenses remain **CC BY 4.0** per upstream publishers.

## Safety

This repository contains **real examples of hateful text** in datasets and logs. Handle outputs accordingly.
