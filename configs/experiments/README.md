# Experiment matrix (follow-up paper)

Configs are grouped by letter (see project brief). Legacy behaviour remains under `configs/models/tinyllama-legacy.yaml` and original `tinyllama.yaml`.

| Group | Purpose |
|-------|---------|
| A | Legacy replication (TinyLlama / Phi-2 / OPT + DynaHate / HateCheck) |
| B | Modern small models + LoRA / QLoRA / DoRA |
| C–H | Structured supervision, rationale, consistency, cross-dataset, efficiency, policy pilot |

Use `peft_type` (`lora`, `dora`, `qlora`, …) and `quantization` (`none`, `4bit`, `8bit`) in YAML. Run:

`uv run hatelens train configs/models/<config>.yaml --dataset dynahate`

HateCheck is **evaluation / functional robustness** in this codebase; training on `hatecheck` remains supported for legacy parity but is not recommended as a primary objective for the new framework.
