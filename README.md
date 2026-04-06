# HateLens

**Efficient decoder LMs + LoRA for hate-speech detection** with **pre/post fine-tuning evaluation**, **HateCheck functionality diagnostics**, and optional **LIME** word attributions — on [DynaHate](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset) and [HateCheck](https://github.com/paul-rottger/hatecheck-data).

---

## The problem

Moderation and safety systems do not only need **high average accuracy**. They need models that teams can **ship cheaply**, **debug when they fail**, and **inspect under adaptation**. A classifier that looks strong on one slice can still break on targeted functional tests (slurs, quoted hate, negation, identity attacks, etc.). Explaining *which words* drive a prediction is still a practical bridge between model behavior and human review — even when full interpretability remains an open research problem.

HateLens is built around that gap: **parameter-efficient fine-tuning** on small open decoder LMs, **systematic functional testing** via HateCheck metadata, and **optional post-hoc attributions** (LIME) — in one reproducible codebase.

---

## What HateLens provides

| Capability | Why it matters |
|------------|----------------|
| **PEFT / LoRA training** | Adapt 1B-class decoders without full fine-tunes; configs for TinyLlama, Phi-2, OPT. |
| **Correct LoRA inference** | Load and **merge** adapters for reliable pre vs post–fine-tune comparison (common footgun in PEFT repos). |
| **Batched evaluation** | Fast metrics + CSV summaries; optional comparison plots. |
| **HateCheck `diagnose-hatecheck`** | Per-**functionality** accuracy/F1 — surfaces *where* the model fails, not only *how much*. |
| **LIME (optional extra)** | Word-level attributions for qualitative analysis and notebooks. |
| **Clean artifact layout** | Data and YAML in Git; **all** generated files under `outputs/` (gitignored except `.gitkeep`). |
| **Cluster-ready** | SLURM template under `scripts/slurm/train.sh`. |

For adjacent public work in the same model class (TinyLlama / LoRA hate detection), see e.g. [HateTinyLLM](https://arxiv.org/abs/2405.01577) — HateLens focuses on a **full evaluation + diagnostics + explainability loop** and engineering hygiene, not a new benchmark score. Nuanced limits (what we do *not* claim) are in [`docs/related-work.md`](docs/related-work.md).

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
| `docs/` | Architecture, experiments, cluster runbook, related work |
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
