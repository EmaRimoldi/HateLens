# Cluster runbook

Patterns are aligned with the **agent_swarms** / autoresearch style: one SLURM job per training run, explicit working directory, logs under `outputs/logs/`.

**GitHub Actions:** the workflow lives at `docs/github-actions-ci.yml` until you copy it to `.github/workflows/ci.yml` (PAT pushes without the `workflow` scope are rejected for workflow files).

## Prerequisites

- Clone repo; `cd` into it.
- `curl -LsSf https://astral.sh/uv/install.sh | sh` (or your site’s `uv` module).
- `uv sync --extra wandb` if you use Weights & Biases.

## Environment

```bash
export HATELENS_ROOT="$(pwd)"
# Optional: Hugging Face cache / token
export HF_HOME="$SCRATCH/hf"
export HF_TOKEN=...   # if your site requires authenticated pulls
```

## One-shot GPU job

Edit `#SBATCH` headers in `scripts/slurm/train.sh` for **partition**, **GPU GRES**, **time**, and **memory** at your center.

```bash
mkdir -p outputs/logs
sbatch scripts/slurm/train.sh dynahate configs/models/tinyllama.yaml
# or
sbatch scripts/slurm/train.sh hatecheck configs/models/tinyllama.yaml
```

Monitor:

```bash
squeue -u "$USER"
tail -f outputs/logs/slurm-<jobid>.out
```

## Weights & Biases (optional)

```bash
export WANDB_ENABLED=1
export WANDB_ENTITY=your-team
```

## Persistent worker pattern (advanced)

For many short trials on one long GPU allocation, reuse the **trigger-file loop** described in `agent_swarms`’s `training_harness.py` (`worker_loop.sh` + `run.trigger`). HateLens does not ship that loop by default; this runbook keeps a **single `hatelens train` invocation** per job for clarity.
