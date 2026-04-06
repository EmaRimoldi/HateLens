# Evaluation (`hatelens eval-run`)

The unified runner loads a **binary** (`AutoModelForSequenceClassification` + PEFT adapter) or **structured** (`StructuredHateModel` + `peft_adapter/` + `vocab/` + `structured_*.pt`) checkpoint and writes:

- `outputs/eval_runs/<run_name>/metrics.json`
- `outputs/eval_runs/<run_name>/metrics.csv`
- `outputs/eval_runs/<run_name>/predictions.jsonl`
- `outputs/eval_runs/<run_name>/rationale_examples.jsonl` (rationale section or skip message)
- `outputs/eval_runs/<run_name>/efficiency.json`

## Config file

See `configs/eval/minimal.yaml`. Override paths via CLI flags when needed.

## Cross-dataset evaluation

Each `cross_dataset` entry `{train: A, test: B}` records the experimental **intent** (trained on A, evaluated on B). The implementation evaluates the **same checkpoint** on dataset **B**’s `split` (default `test`). It does not re-train.

## Limitations

- **HateXplain** may require Hugging Face access / cache; failures surface as dataset errors in `metrics.json`.
- **Rationale metrics** require a structured checkpoint and sufficient span-aligned examples.
- **QLoRA evaluation** may require matching `quantization` when loading structured adapters trained with 4-bit weights (not enabled in the default structured path).

For onboarding, start with the main `README.md`.
