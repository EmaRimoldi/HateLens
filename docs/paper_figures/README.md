# Paper figures and frozen results

## Scope (what this work demonstrates)

- **Binary vs structured** detection with TinyLlama + PEFT (LoRA family): moving from a single hate label to **multi-task structured heads** (target, hate type, explicitness, rationale).
- **Supervision ablations**: training **without rationale** vs **with consistency** vs full structured setup.
- **Extra supervision**: **DynaHate + HateXplain** joint structured training.
- **Generalization**: **HateEval** (in-domain and cross-style) and **HateCheck** functional stress tests.
- **PEFT sanity**: comparable **LoRA vs QLoRA vs DoRA** under the same training budget (see configs).

Figures are generated from `outputs/eval_runs/*/metrics.json` (not committed; regenerate on cluster).

## Files

| File | Role |
|------|------|
| `fig1_main_comparison.{pdf,png}` | DynaHate, HateEval, HateCheck F1 for main systems |
| `fig2_peft_hatecheck.{pdf,png}` | HateCheck F1 for PEFT triplet |
| `results_summary.json` | Machine-readable key metrics + scope string |
| `tables/*.csv` | Snapshot of manuscript tables (`export-tables`) |

## Regenerate

From repo root (after eval runs exist):

```bash
uv run python scripts/generate_paper_figures.py
uv run hatelens export-tables outputs/eval_runs/exp_*/metrics.json --kind efficiency --out-csv docs/paper_figures/tables/efficiency.csv
```
