# HateLens

**Parameter-efficient (LoRA-family) fine-tuning of small decoder LMs** for **binary hate-speech detection**, with an **extended research pipeline** for **structured multi-task supervision**, optional **rationale** and **pair-consistency** objectives, and a **unified evaluation runner** (in-domain, cross-dataset, HateCheck robustness, calibration, efficiency).

This repository supports:

- **Legacy replication path** — the original TinyLlama + LoRA + DynaHate / HateCheck workflow (CLI `train` + `evaluate` + `diagnose-hatecheck`).
- **Follow-up research pipeline** — `training_mode: binary | structured` in YAML, HateEval / HateXplain loaders, structured multi-head model + trainer, and `hatelens eval-run` for paper-style evaluation exports.

If you only need the original paper-style workflow, use **binary** configs under `configs/models/` and the legacy `evaluate` command. If you are running the extended experiments, prefer **`training_mode: structured`** and **`hatlens eval-run`**.

---

## Project overview

- **What HateLens is**: a compact codebase for fine-tuning open decoder LMs (TinyLlama, Phi-2, OPT, etc.) with PEFT (LoRA / optional QLoRA / DoRA when supported), plus systematic evaluation on DynaHate, HateEval, HateXplain, and functional testing via HateCheck.
- **Legacy scope**: binary LoRA fine-tuning + batched eval + HateCheck diagnostics (+ optional LIME).
- **Extended scope**: structured multi-head prediction (main + auxiliary labels), optional rationale token supervision (HateXplain-aligned spans when available), optional pair-consistency regularization (when pair metadata exists), unified `eval-run` exports under `outputs/eval_runs/`.
- **Binary mode** (`training_mode: binary` or omitted): Hugging Face `AutoModelForSequenceClassification` + LoRA; standard hate vs not-hate logits.
- **Structured mode** (`training_mode: structured`): backbone + PEFT with `task_type: FEATURE_EXTRACTION`, multi-head classifier + token rationale head; losses are combined in `StructuredTrainer` (main + aux + optional rationale + optional consistency).

---

## Repository features (verified in code)

| Feature | Implementation |
|--------|------------------|
| Legacy binary training | `train_pipeline.run_training` when `training_mode` is binary |
| Structured multi-task training | `structured_train.run_structured_training` + `StructuredHateModel` |
| Rationale-aware training | `use_rationale` + HateXplain span alignment (`rationale_align.py`) |
| Consistency-aware training | `use_consistency` + pair relations in `StructuredTrainer` |
| HateEval | `datasets.create_hateeval_dataset` (place TSVs under `data/HateEval/`) |
| HateXplain | `structured_data.load_hatexplain_hf` (requires HF access / cache) |
| HateCheck evaluation | `eval-run --hatecheck` + `diagnose-hatecheck` |
| Calibration metrics | ECE + Brier in `evaluation_calibration.py` and `eval-run` bundles |
| Efficiency metrics | `outputs/.../efficiency.json` from `eval-run` (params, disk, timing) |
| PEFT variants | `peft_type` + `quantization` in YAML (`lora`, `qlora`, `dora`, …) |

**Not claimed**: true bottleneck Adapter layers (Hugging Face `AdapterConfig`) — the repo standardizes on **LoRA-family PEFT** unless you extend `peft_factory.py` yourself.

---

## Installation

**Python 3.10+**. Recommended: [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/EmaRimoldi/HateLens.git
cd HateLens
uv sync
```

Optional extras:

```bash
uv sync --extra lime     # LIME attributions
uv sync --extra wandb      # experiment tracking
uv sync --extra quant      # bitsandbytes / QLoRA-style loading
```

Set `HATELENS_ROOT` to the repo root if you launch commands from elsewhere.

**GPU**: training/eval-run benefits from CUDA; CPU runs are possible for smoke tests but will be slow for full epochs.

**Hugging Face**: gated models may need `huggingface-cli login` or `HF_TOKEN`. HateXplain loading uses `datasets.load_dataset` and may download on first use.

---

## Dataset support

| Dataset | Role | Notes |
|--------|------|------|
| **DynaHate** | Primary train/eval for hate vs not | CSV under `data/DynaHate/` (see `datasets.download_dynahate`). |
| **HateEval** | Train/eval (English SemEval 2019) | Place `train_en.tsv` / `dev_en.tsv` under `data/HateEval/` or call `download_hateeval_tsvs()` (may fail on strict SSL; manual download is OK). |
| **HateXplain** | Rationale supervision (structured) | Loaded via Hugging Face; requires network or cached files. |
| **HateCheck** | Functional robustness eval | `data/hatecheck/hatecheck_split.csv` — stratified splits. |

Schema mappings are implemented in `datasets.py` and `structured_data.py` (binary label column `label`, text fields `text` / `test_case` / `post` as appropriate).

---

## Training modes

### Binary (`training_mode: binary` or omitted)

- Standard sequence classification head on top of the decoder.
- Datasets: `dynahate`, `hatecheck`, `hateeval`.

### Structured (`training_mode: structured`)

- Backbone: `AutoModel` + PEFT with **`task_type: FEATURE_EXTRACTION`** (see configs).
- Heads: main (hate vs not) + auxiliary vocabularies + token rationale head.
- **Rationale**: enabled with `use_rationale: true` (HateXplain rows only contribute span losses when spans parse).
- **Consistency**: `use_consistency: true` uses pair metadata (`pair_id`, `pair_relation`) when present; many batches may have zero pair terms (loss is gated).
- **Masked labels**: `IGNORE_INDEX` (`-100`) for padding / unknown auxiliaries as in `labels.py`.

---

## How to run legacy experiments

TinyLlama on DynaHate / HateEval (binary configs such as `tinyllama.yaml` or `tinyllama-legacy.yaml`):

```bash
uv run hatelens train configs/models/tinyllama-legacy.yaml --dataset dynahate
uv run hatelens train configs/models/tinyllama-legacy.yaml --dataset hateeval
```

Phi-2 / OPT on DynaHate:

```bash
uv run hatelens train configs/models/phi-2.yaml --dataset dynahate
uv run hatelens train configs/models/opt-1.3b.yaml --dataset dynahate
```

Smoke / CI-style run (small subsets via env + YAML `smoke_test`):

```bash
export HATELENS_SMOKE=1
uv run hatelens train configs/smoke/tinyllama_dynahate.yaml --dataset dynahate
```

Legacy evaluation (pre/post adapter):

```bash
uv run hatelens evaluate --dynahate --batch-size 16
uv run hatelens evaluate --hatecheck --batch-size 16
```

---

## How to run structured experiments

Structured TinyLlama examples:

```bash
uv run hatelens train configs/models/tinyllama-structured.yaml --dataset dynahate
uv run hatelens train configs/models/tinyllama-structured.yaml --dataset hateeval
uv run hatelens train configs/models/tinyllama-structured.yaml --dataset hatexplain
uv run hatelens train configs/models/tinyllama-structured.yaml --dataset dynahate_hatexplain
```

Rationale / consistency ablations (see `configs/experiments/paper_matrix/`):

```bash
uv run hatelens train configs/experiments/paper_matrix/structured_dynahate_no_rationale.yaml --dataset dynahate
uv run hatelens train configs/experiments/paper_matrix/structured_dynahate_consistency.yaml --dataset dynahate
```

Artifacts per run directory:

- `config_resolved.yaml`, `train_metrics.json` (runtime + Trainer metrics)
- `best_checkpoint/` with tokenizer + adapter (binary) **or** `peft_adapter/`, `vocab/`, `structured_model.pt`, `structured_heads.pt` (structured)

---

## How to run evaluation

### Unified runner (`eval-run`)

Configuration example: `configs/eval/minimal.yaml`.

```bash
uv run hatelens eval-run --config configs/eval/minimal.yaml
```

Quick CLI without YAML (requires a real checkpoint path on disk):

```bash
uv run hatelens eval-run \
  --checkpoint outputs/runs/tinyllama/dynahate/best_checkpoint \
  --run-name my_eval \
  --in-domain dynahate hateeval \
  --cross dynahate:hateeval hateeval:dynahate \
  --hatecheck
```

**Cross-dataset entries** label the *intent* (trained on A, evaluated on B). The runner always evaluates the **given checkpoint** on dataset **B**’s test split (it does not re-train).

Outputs (under `outputs/eval_runs/<run_name>/`):

- `metrics.json`, `metrics.csv`
- `predictions.jsonl`
- `rationale_examples.jsonl` (structured rationale metrics or a skip message)
- `efficiency.json` (parameters, checkpoint bytes, simple latency / throughput; CUDA peak memory when available)

### Table export helper

```bash
uv run hatelens export-tables path/to/metrics.json --kind hatecheck --out-md /tmp/table.md
```

### Legacy commands (still supported)

```bash
uv run hatelens diagnose-hatecheck --batch-size 16
```

More detail: [`docs/evaluation.md`](docs/evaluation.md).

---

## Output directory guide

| Artifact | Location |
|---------|----------|
| Training runs | `outputs/runs/<model>/<dataset>/` (binary) or `.../structured_<dataset>/` (structured) |
| Best checkpoint | `.../best_checkpoint/` — binary: PEFT adapter + tokenizer; structured: `peft_adapter/`, `vocab/`, `structured_model.pt`, `structured_heads.pt` |
| Resolved config + train metrics | `config_resolved.yaml`, `train_metrics.json` in the run directory (parent of `best_checkpoint/`) |
| Trainer validation snapshot | `eval_summary.json` — last epoch’s `eval_*` keys from Hugging Face; **not** a substitute for `eval-run` |
| Unified eval exports | `outputs/eval_runs/<run_name>/` |
| Eval artifacts | `metrics.json`, `metrics.csv`, `predictions.jsonl`, `rationale_examples.jsonl`, `efficiency.json` |
| Legacy eval CSV | `outputs/eval/<dataset>/metrics_summary.csv` |

---

## Cluster / batch jobs (SLURM)

GPU training jobs should **not** inherit `HATELENS_SMOKE=1` from your shell. The template script resets smoke off:

```bash
export HATELENS_ROOT="$(pwd)"
mkdir -p outputs/logs
sbatch scripts/slurm/train.sh dynahate configs/models/tinyllama-legacy.yaml
sbatch scripts/slurm/train.sh dynahate configs/models/tinyllama-structured.yaml
```

Chained train → train → **batch eval**: see `scripts/slurm/eval_bundle.sh` and `configs/experiments/README.md` (`sbatch --dependency=afterok:…`).

Tune `#SBATCH` headers in `scripts/slurm/train.sh` for your partition and wall time. Some clusters report an older kernel; if the job **finishes training** but SLURM stays `RUNNING` with no new log lines, **`scancel`** after verifying `best_checkpoint/` and `train_metrics.json` are written.

---

## Minimal follow-up paper reproduction

Step-by-step (configs for Groups 1–6: `configs/experiments/paper_matrix/README.md`):

1. **Legacy binary (Group 1)** — `tinyllama-legacy` on `dynahate` and `hateeval`; outputs `outputs/runs/tinyllama/<dataset>/`.
2. **Binary vs structured (Group 2)** — `tinyllama_binary_compare.yaml` vs `tinyllama-structured.yaml` on `dynahate`.
3. **Rationale ablation (Group 3)** — `structured_dynahate_no_rationale.yaml` vs full `tinyllama-structured.yaml`.
4. **Consistency ablation (Group 4)** — `tinyllama-structured.yaml` vs `structured_dynahate_consistency.yaml` on `dynahate`.
5. **HateCheck robustness (Group 5)** — `eval-run` with `--hatecheck` (and `--in-domain hatecheck` if you only want functional robustness) for the best binary and structured checkpoints.
6. **PEFT sanity (Group 6)** — `tinyllama.yaml` vs `tinyllama_qlora.yaml` vs `tinyllama_dora.yaml` (QLoRA: `uv sync --extra quant`, GPU).

**Evaluation**

- Template with in-domain, cross pairs (including `dynahate_hatexplain` → `hateeval` metadata), HateCheck, optional rationale, calibration, efficiency:

  ```bash
  uv run hatelens eval-run --config configs/eval/paper_followup.yaml
  ```

- Minimal smoke template: `configs/eval/minimal.yaml`.

**Compare runs**

```bash
uv run hatelens export-tables outputs/eval_runs/run_a/metrics.json outputs/eval_runs/run_b/metrics.json \
  --kind binary_vs_structured --out-csv compare_binary_structured.csv --out-md compare.md
uv run hatelens export-tables outputs/eval_runs/*/metrics.json --kind hatecheck --out-csv hatecheck_compare.csv
uv run hatelens export-tables run1/metrics.json run2/metrics.json --kind efficiency --out-md efficiency.md
```

`--kind` choices: `binary_vs_structured`, `rationale`, `consistency`, `cross` (rows from each file’s `cross_dataset` block), `hatecheck`, `efficiency`.

---

## Current limitations / future work

- **HateXplain** requires Hugging Face access; offline environments must cache datasets ahead of time.
- **HateEval** download scripts may fail under strict SSL — manual placement of TSVs is supported.
- **QLoRA / 4-bit structured training** is not the default structured path; structured training uses full-precision backbone loading unless you extend it.
- **True bottleneck adapters** (non-LoRA adapters) are not the default implementation.
- **`eval-run` rationale metrics** depend on span extraction success; counts may be low if records lack usable spans.
- **GPU memory** scales with model size; use smoke configs or smaller checkpoints when developing on laptops.

---

## Repository layout (high level)

| Path | Role |
|------|------|
| `src/hatelens/` | Package + CLI (`train`, `eval-run`, `export-tables`, …) |
| `configs/models/` | Base training YAML profiles |
| `configs/eval/` | `eval-run` YAML templates |
| `configs/experiments/paper_matrix/` | Follow-up paper matrix configs + README |
| `data/` | Versioned CSV / splits (not all datasets are redistributable) |
| `outputs/` | All generated artifacts (gitignored) |
| `docs/` | Architecture / evaluation notes |
| `tests/` | Pytest |

---

## Configuration

YAML fields are documented inline in `configs/models/*.yaml`. Important keys:

- `training_mode`: `binary` (default) or `structured`
- `task_type`: `SEQ_CLS` (binary) or `FEATURE_EXTRACTION` (structured backbone)
- `peft_type`, `quantization`, loss weights (`aux_loss_weight`, `rationale_loss_weight`, `consistency_loss_weight`)

---

## Legacy scripts

`scripts/evaluate_models.py`, `scripts/compute_lime_scores.py`, and `scripts/trainer_*.py` forward to the CLI where possible. Prefer `uv run hatelens …`.

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
