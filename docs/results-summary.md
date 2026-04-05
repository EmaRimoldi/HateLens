# Results summary

## Automated metrics in CI

Continuous integration runs **unit tests only** (datasets, diagnostics helper, checkpoint file layout). It does **not** download TinyLlama or run GPU inference.

## Local / cluster evaluation

Run:

```bash
uv run hatelens evaluate --hatecheck --batch-size 16
uv run hatelens diagnose-hatecheck --batch-size 16
```

Outputs:

- `results/hatecheck/metrics_summary.csv`
- `results/hatecheck/functionality_breakdown.csv`

**Resource note**: evaluating the default 1.1B base model requires sufficient **RAM/VRAM** and a Hugging Face cache (or `HF_TOKEN` for rate limits). If a run is killed (e.g. exit code 137), reduce batch size, use a GPU node, or point `HATELENS_BASE_MODEL` to a smaller public classifier for smoke tests.

## Not claimed

No new accuracy numbers are asserted in this refactor; reproduce and report your own tables from the CSVs above.
