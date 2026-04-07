# Minimal follow-up paper experiment matrix

Commands assume repository root and `uv sync`. Training writes under `outputs/runs/...`; evaluation writes under `outputs/eval_runs/<run_name>/` via `hatelens eval-run`.

## Group 1 — Legacy replication

```bash
uv run hatelens train configs/models/tinyllama-legacy.yaml --dataset dynahate
uv run hatelens train configs/models/tinyllama-legacy.yaml --dataset hateeval
```

## Group 2 — Binary vs structured (TinyLlama)

```bash
uv run hatelens train configs/experiments/paper_matrix/tinyllama_binary_compare.yaml --dataset dynahate
uv run hatelens train configs/models/tinyllama-structured.yaml --dataset dynahate
```

## Group 3 — Rationale ablation

Writes to `outputs/runs/tinyllama/structured_dynahate_no_rationale/` (suffix avoids overwriting the full structured run).

```bash
uv run hatelens train configs/experiments/paper_matrix/structured_dynahate_no_rationale.yaml --dataset dynahate
uv run hatelens train configs/models/tinyllama-structured.yaml --dataset dynahate
```

## Group 4 — Consistency ablation

Writes to `outputs/runs/tinyllama/structured_dynahate_consistency/`.

```bash
uv run hatelens train configs/models/tinyllama-structured.yaml --dataset dynahate
uv run hatelens train configs/experiments/paper_matrix/structured_dynahate_consistency.yaml --dataset dynahate
```

## Group 5 — Robustness (HateCheck)

After training a binary or structured checkpoint, evaluate on HateCheck:

```bash
uv run hatelens eval-run --checkpoint outputs/runs/tinyllama/hatecheck/best_checkpoint --run-name binary_hatecheck --in-domain hatecheck --hatecheck
uv run hatelens eval-run --checkpoint outputs/runs/tinyllama/structured_dynahate/best_checkpoint --run-name structured_hatecheck --in-domain hatecheck --hatecheck
```

## Group 6 — PEFT sanity (LoRA vs QLoRA vs DoRA)

Use `tinyllama_peft_lora.yaml` (r=8, 1 epoch, isolated under `outputs/runs/tinyllama_peft_lora/`) so LoRA matches the QLoRA/DoRA cards and does not overwrite legacy `outputs/runs/tinyllama/dynahate`.

```bash
uv run hatelens train configs/experiments/paper_matrix/tinyllama_peft_lora.yaml --dataset dynahate
uv run hatelens train configs/experiments/paper_matrix/tinyllama_qlora.yaml --dataset dynahate
uv run hatelens train configs/experiments/paper_matrix/tinyllama_dora.yaml --dataset dynahate
```

QLoRA requires GPU + optional `uv sync --extra quant` (bitsandbytes).

## Unified evaluation + tables

```bash
uv run hatelens eval-run --config configs/eval/minimal.yaml
uv run hatelens export-tables outputs/eval_runs/tinyllama_dynahate_smoke/metrics.json --kind hatecheck --out-md /tmp/hc.md
```

Cross-dataset pairs are **informational labels** for the run; the runner always evaluates the **fixed checkpoint** on each requested **test** split (`cross_dataset` entries use the `test` dataset’s held-out split).

Example third pair (after training `structured_dynahate_hatexplain`): add to your eval YAML or CLI `--cross dynahate_hatexplain:hateeval` — the checkpoint is still the **saved adapter** from the combined training run; the `train` key documents that supervision mix.

See also `configs/eval/paper_followup.yaml` for a template that includes this pair.
