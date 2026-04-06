# Experiments

## Hypotheses (engineering validation)

1. **H1**: Merging LoRA adapters into the base classifier yields consistent labels vs keeping the `PeftModel` wrapper (sanity check).
2. **H2**: Batched evaluation matches sequential metrics within floating-point tolerance (regression guard).
3. **H3**: HateCheck **functionality** buckets show heterogeneous F1 — descriptive error analysis.

## Metrics

Accuracy, macro-F1, precision, recall, ROC-AUC, PR-AUC (where defined).

## Commands

```bash
uv sync --extra dev --extra lime
hatelens evaluate --hatecheck --batch-size 16 --plots
hatelens evaluate --dynahate --batch-size 16
hatelens diagnose-hatecheck --batch-size 16
hatelens lime --hatecheck   # requires hatelens[lime]
```

## Outputs (local, gitignored)

- `outputs/eval/<dataset>/metrics_summary.csv`
- `outputs/eval/hatecheck/functionality_breakdown.csv`
- `outputs/lime/*.pkl`
- Optional plots when `--plots` is set.

## Full training

```bash
export WANDB_ENABLED=1   # optional
./scripts/train_dynahate.sh configs/models/tinyllama.yaml
./scripts/train_hatecheck.sh configs/models/tinyllama.yaml
```

## Note on compute

Evaluation loads **TinyLlama-1.1B** from Hugging Face unless cached; plan **~2GB+** download and **>8GB RAM** for comfortable CPU runs; GPU recommended.
