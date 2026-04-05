# Refactor plan

Execution branch: `refactor/professional-v2`.

1. **Baseline** — clone, map entry points, reproduce issues (PEFT load, path fragility, missing deps).
2. **Package** — `hatelens` installable module, `uv`, tests, CI.
3. **Correctness** — adapter-aware `load_sequence_classifier`, batched eval.
4. **Innovation (bounded)** — HateCheck `functionality` diagnostics CSV + CLI.
5. **Ops** — SLURM template mirroring `agent_swarms` one-shot job style; docs runbook.
6. **Docs** — README, architecture, related work, changelog.

See `CHANGELOG.md` and `docs/repo-audit.md` for outcomes.
