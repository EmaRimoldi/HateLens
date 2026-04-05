# Experiments

## Hypotheses (engineering validation)

1. **H1**: Merging LoRA adapters into the base classifier yields the same labels as keeping the `PeftModel` wrapper for inference on the test split (sanity; optional manual check).
2. **H2**: Batched evaluation matches sequential evaluation metrics within floating-point tolerance (regression guard; add explicit test if needed).
3. **H3**: HateCheck **functionality** buckets show heterogeneous F1 — motivating targeted error analysis (descriptive, not a novelty claim).

## Metrics

- Accuracy, macro-F1, precision, recall, ROC-AUC, PR-AUC (where defined).

## Commands

```bash
uv sync --extra dev --extra lime
hatelens evaluate --hatecheck --batch-size 16 --plots
hatelens evaluate --dynahate --batch-size 16
hatelens diagnose-hatecheck --batch-size 16
hatelens lime --hatecheck   # requires hatelens[lime]
```

## Outputs

- `results/<dataset>/metrics_summary.csv`
- `results/hatecheck/functionality_breakdown.csv`
- Optional plots under `results/<dataset>/` when `--plots` is set.

## Full training

```bash
export WANDB_ENABLED=1   # optional
./run_training_dynahate.sh experiments/TinyLlama/config.yaml
./run_training_hatecheck.sh experiments/TinyLlama/config.yaml
```

## Note on compute

End-to-end evaluation loads **TinyLlama-1.1B** from Hugging Face unless cached; plan **~2GB+** download and **>8GB RAM** for comfortable CPU runs; GPU recommended.
