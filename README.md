# HateLens

**Tiny decoder LMs + LoRA for efficient, explainable hate speech detection** on [DynaHate](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset) and [HateCheck](https://github.com/paul-rottger/hatecheck-data).

[![CI](https://github.com/EmaRimoldi/HateLens/actions/workflows/ci.yml/badge.svg)](https://github.com/EmaRimoldi/HateLens/actions/workflows/ci.yml)

## Why this matters

Moderation and research systems need models that are **small enough to deploy**, **fine-tunable on modest GPUs**, and **inspectable** when a decision is contested. HateLens keeps that entire loop in one repo: PEFT fine-tuning, correct loading of **adapter checkpoints**, fast **batched evaluation**, optional **LIME** word attributions, and a **HateCheck functionality breakdown** so failures line up with the benchmark’s functional test types.

## Install

Requires **Python 3.10+**. Recommended: [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/EmaRimoldi/HateLens.git
cd HateLens
uv sync                    # core
uv sync --extra lime       # + LIME explainability
uv sync --extra wandb      # + Weights & Biases (training logs)
```

Legacy `pip install -r requirements.txt` remains possible for older workflows; new development targets `pyproject.toml`.

## Quick start

```bash
# Metrics on test split (downloads TinyLlama base from Hugging Face if not cached)
uv run hatelens evaluate --hatecheck --batch-size 16
uv run hatelens evaluate --dynahate --batch-size 16 --plots

# HateCheck: per-functionality accuracy/F1 (uses bundled adapter + metadata)
uv run hatelens diagnose-hatecheck --batch-size 16

# LoRA training (W&B only if WANDB_ENABLED=1)
uv run hatelens train experiments/TinyLlama/config.yaml --dataset dynahate
./run_training_hatecheck.sh experiments/TinyLlama/config.yaml

# LIME (optional extra)
uv run hatelens lime --hatecheck
```

Set `HATELENS_ROOT` if you run commands outside the repo tree.

## Project layout

| Path | Purpose |
|------|---------|
| `src/hatelens/` | Library + CLI |
| `data/` | DynaHate CSV + HateCheck splits |
| `experiments/*/config.yaml` | LoRA + trainer hyperparameters |
| `checkpoints/` | Example **LoRA adapters** (not full dense weights) |
| `cluster/sbatch_train.sh` | SLURM template |
| `docs/` | Architecture, audit, experiments, cluster runbook, related work |

## Authors (original course project)

| Name | SCIPER |
|------|--------|
| [Vittoria Meroni](https://github.com/vittoriameroni) | 386722 |
| [Emanuele Rimoldi](https://github.com/EmaRimoldi) | 377013 |
| [Simone Vicentini](https://github.com/SimoVice/) | 378204 |

## Data citations

- Vidgen et al., EMNLP 2021 — DynaHate.
- Röttger et al., ACL 2021 — HateCheck.

## License

Apache-2.0 (code). Dataset licenses remain **CC BY 4.0** per upstream publishers.

## Safety note

This repository contains **real examples of hateful text** in datasets and model outputs. Handle logs and screenshots accordingly.
