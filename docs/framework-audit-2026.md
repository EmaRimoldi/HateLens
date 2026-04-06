# HateLens framework audit (2026) — Phases 0–8 roadmap

This document satisfies the **repo audit** request: repository map, migration plan, and execution plan. The **original paper** in public materials refers to **HateTinyLLM**-class setups (TinyLlama, Phi-2, OPT); this repo historically shipped **DynaHate + HateCheck** (not HateEval). **HateEval** is not wired in the codebase yet; add it as a new loader under the unified schema.

---

## A. Repository map

### Training entry points

- **Primary**: `uv run hatelens train <config.yaml> --dataset {dynahate,hatecheck}` → `hatelens/cli.py` → `hatelens/train_pipeline.py::run_training`.
- **Legacy shims**: `scripts/trainer_dynahate.py`, `scripts/trainer_hatecheck.py` (forward to the same pipeline).

### Dataset loaders

- `hatelens/datasets.py`: **DynaHate** (CSV + official splits), **HateCheck** (`hatecheck_split.csv`), optional **Gab** stub.
- `hatelens/mapping.py`: rows → `UnifiedExample`.
- `hatelens/loaders/hatexplain.py`: Hugging Face candidates (may require network / token).

### Model wrappers

- `hatelens/modeling.py`: `load_sequence_classifier` (hub or **PEFT adapter** + merge).
- `hatelens/registry.py`: string keys → hub IDs.
- `hatelens/peft_factory.py`: base classifier + **LoRA / QLoRA / DoRA / AdaLoRA / PiSSA** (with graceful fallback).

### PEFT integration points

- Training: `hatelens/train_pipeline.py` (was inline `LoraConfig` + `get_peft_model`; now factory).
- Inference: `hatelens/modeling.py` (`PeftModel.from_pretrained` + merge).

### Evaluation scripts

- `hatelens/cli.py`: `evaluate`, `diagnose-hatecheck`, `lime`.
- `hatelens/evaluation.py`: batched predictions + **binary + macro** metrics.
- `hatelens/evaluation_suite.py`, `hatelens/evaluation_calibration.py`: ECE, Brier, bundles.

### Config system

- YAML under `configs/models/*.yaml`; optional `configs/smoke/` for CPU/CI smoke.
- Keys: `model_checkpoint` (hub ID or **registry key**), `peft_type`, `quantization`, LoRA hyperparameters, `smoke_test`, `max_length`, etc.

### Logging / checkpointing

- HF `TrainingArguments` → `outputs/runs/<model>/<dataset>/`, `best_checkpoint/`.
- Run metadata: **config hash** + **git HEAD** logged at train start (`train_pipeline`).

### Binary-only assumptions (hard-coded spots)

- `hatelens/modeling.py`: `num_labels=2`, `ID2LABEL` / `LABEL2ID`.
- `hatelens/train_pipeline.py`: `AutoModelForSequenceClassification` with 2 labels.
- `hatelens/evaluation.py`: hate probability = softmax[:, 1].
- `hatelens/datasets.py`: DynaHate binarised to 0/1; HateCheck `label` int.
- **CLI** `evaluate`: `--dynahate` / `--hatecheck` only.

---

## B. Migration plan

### Modify

- `src/hatelens/train_pipeline.py` — registry, PEFT factory, metrics, smoke, `processing_class`, subset-before-tokenize, test-split truncation in smoke.
- `src/hatelens/evaluation.py` — explicit **f1_binary**, **f1_macro**, macro precision/recall.
- `src/hatelens/datasets.py` — preserve DynaHate **target / type / level** columns.

### Add

- `src/hatelens/registry.py`, `peft_factory.py`, `schema.py`, `mapping.py`
- `src/hatelens/loaders/hatexplain.py`, `src/hatelens/parsing/structured_output.py`
- `src/hatelens/losses/consistency.py`, `src/hatelens/evaluation_calibration.py`, `evaluation_suite.py`
- `src/hatelens/prompting.py`, `src/hatelens/distill.py`
- `configs/models/tinyllama-legacy.yaml`, `configs/models/qwen2.5-1.5b.yaml`, `configs/smoke/tinyllama_dynahate.yaml`
- `configs/experiments/README.md`, this doc
- Tests: `tests/test_metrics.py`, `test_mapping.py`, `test_parsing.py`, `test_calibration.py`, `test_consistency_numpy.py`, `conftest.py`

### Remain untouched (compat)

- `data/` CSVs and HateCheck preprocessing scripts.
- `scripts/evaluate_models.py` shims.
- Default **legacy** commands and paths (`outputs/runs/tinyllama/...`).

### Risk points

- **Transformers 5.x**: `Trainer` no longer accepts `tokenizer=`; use `processing_class=` (**fixed**).
- **Gated models** (Llama, some Gemma): need `HF_TOKEN`.
- **QLoRA**: requires optional `bitsandbytes`.
- **CPU smoke** on 1B+ models: very slow; prefer GPU or a future tiny smoke model config.

### Order of implementation

1. Phase 0 — metrics, smoke, legacy YAML, Trainer fix (**done in this PR**).
2. Phase 1 — registry + PEFT factory (**done**); extend configs for more models.
3. Phase 2 — unified schema loaders (HateXplain, HateEval stub, HateBench stubs).
4. Phase 3 — structured multi-task or JSON generation training loop.
5. Phases 4–6 — distill cache, rationale metrics, consistency loss in Trainer, policy prompts (**scaffolding in place**).
6. Phase 7 — full evaluation runner (cross-dataset, efficiency profiling).
7. Phase 8 — experiment YAML matrix + shell driver.

---

## C. Execution plan

| Phase | Smoke test | Minimal experiment | Full experiment |
|-------|------------|--------------------|-----------------|
| 0 | `uv run hatelens train configs/smoke/tinyllama_dynahate.yaml --dataset dynahate` | 1 epoch TinyLlama DynaHate | 3 epochs + `evaluate` + `diagnose-hatecheck` |
| 1 | Train with `peft_type: lora` + `quantization: none` | Qwen2.5 1.5B LoRA DynaHate | LoRA vs QLoRA vs DoRA sweep |
| 2 | `pytest` + optional `load_hatexplain_unified()` online | Map DynaHate+HateXplain to schema | Multi-dataset tables |
| 3–5 | Unit tests for losses/parsing | One structured run | Ablations C–E |
| 7–8 | `run_binary_eval_bundle` on saved preds | Cross-dataset eval script | Full matrix + seeds 13,21,42 |

---

## Baseline reproduction notes (Phase 0)

- **Commands**: full legacy — `uv run hatelens train configs/models/tinyllama-legacy.yaml --dataset dynahate` (and `hatecheck` if desired). Smoke — `configs/smoke/tinyllama_dynahate.yaml` with `smoke_test: true`.
- **Metric definitions**: `precision` / `recall` / `f1` = **hate-class (positive) binary**; `f1_macro` = macro-averaged F1. Training and `evaluate` now align with `classification_metrics` (previously HF `evaluate` load used unqualified `f1`, effectively binary for binary tasks but not documented).
- **Discrepancy vs “HateEval”**: repository targets **HateCheck** for functional tests; **HateEval** is listed in the paper brief but **not** implemented as a first-class dataset here yet.

---

## Pending / optional (not blocking legacy)

- HateEval, HateBench, SoftHateBench loaders.
- Multi-head or generative structured training loop (beyond stubs).
- Teacher distillation runs (cache helpers exist).
- Automated multi-seed orchestration and full efficiency benchmarks (VRAM, throughput).
