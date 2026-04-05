# Results summary

## Automated metrics in CI

Continuous integration runs **unit tests only** (datasets, layout, diagnostics helper). It does **not** download TinyLlama or run GPU inference.

## Local / cluster evaluation

Run:

```bash
uv run hatelens evaluate --hatecheck --batch-size 16
uv run hatelens diagnose-hatecheck --batch-size 16
```

Outputs (under `outputs/`, not committed):

- `outputs/eval/hatecheck/metrics_summary.csv`
- `outputs/eval/hatecheck/functionality_breakdown.csv`

**Resource note**: the default base model needs sufficient **RAM/VRAM** and a Hugging Face cache. If a run is killed (e.g. exit code 137), reduce batch size, use a GPU node, or set `HATELENS_BASE_MODEL` to a smaller classifier for smoke tests.

## Not claimed

The repository does **not** ship precomputed metric tables; reproduce and archive your own CSVs from runs.
