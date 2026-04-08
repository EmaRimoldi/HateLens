#!/usr/bin/env python3
"""Build publication-quality figures from outputs/eval_runs/*/metrics.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _f1_in(m: dict, split: str) -> float:
    return float(m["in_domain"][split]["f1"])


def _f1_hc(m: dict) -> float:
    return float(m["hatecheck"]["overall"]["f1"])


def main() -> int:
    root = _repo_root()
    er = root / "outputs" / "eval_runs"
    if not er.is_dir():
        print(f"Missing {er}", file=sys.stderr)
        return 1

    def mj(name: str) -> dict:
        return _load(er / name / "metrics.json")

    # --- Main comparison (scope: binary vs structured + supervision + DH+HX) ---
    rows = [
        ("Binary (legacy)", mj("exp_g1_binary_dynahate")),
        ("Binary (compare)", mj("exp_g2_binary_compare_dynahate")),
        ("Structured", mj("exp_g2_structured_dynahate")),
        ("− Rationale", mj("exp_ablation_no_rationale")),
        ("+ Consistency", mj("exp_ablation_consistency")),
        ("Structured\nDynaHate+HateXplain", mj("exp_structured_dh_hx")),
    ]
    labels = [r[0] for r in rows]
    dh = [_f1_in(m, "dynahate") for _, m in rows]
    he = [_f1_in(m, "hateeval") for _, m in rows]
    hc = [_f1_hc(m) for _, m in rows]

    # PEFT
    peft = [
        ("LoRA", _f1_hc(mj("exp_g6_peft_lora"))),
        ("QLoRA", _f1_hc(mj("exp_g6_peft_qlora"))),
        ("DoRA", _f1_hc(mj("exp_g6_peft_dora"))),
    ]

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Okabe–Ito (colorblind-friendly)
    c = {
        "dh": "#0072B2",
        "he": "#E69F00",
        "hc": "#009E73",
        "peft": ["#56B4E9", "#D55E00", "#CC79A7"],
    }

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        }
    )

    x = np.arange(len(labels))
    w = 0.25

    fig1, ax1 = plt.subplots(figsize=(7.2, 3.8))
    kw = dict(edgecolor="0.2", linewidth=0.4)
    ax1.bar(x - w, dh, width=w, label="DynaHate (in-domain)", color=c["dh"], **kw)
    ax1.bar(x, he, width=w, label="HateEval (in-domain)", color=c["he"], **kw)
    ax1.bar(x + w, hc, width=w, label="HateCheck (robustness)", color=c["hc"], **kw)
    ax1.set_ylabel("F1")
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=22, ha="right")
    ax1.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="0.4")
    ax1.set_title("Main comparison: detection F1 across benchmarks")
    ax1.axhline(0.5, color="0.75", linewidth=0.6, linestyle="--", zorder=0)
    ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.8)
    fig1.tight_layout()
    out_dir = root / "docs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out_dir / f"fig1_main_comparison.{ext}"
        fig1.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(4.2, 3.4))
    px = np.arange(len(peft))
    ax2.bar(px, [p[1] for p in peft], color=c["peft"], edgecolor="0.2", linewidth=0.4)
    ax2.set_xticks(px)
    ax2.set_xticklabels([p[0] for p in peft])
    ax2.set_ylabel("HateCheck F1 (overall)")
    ax2.set_ylim(0.75, 0.92)
    ax2.set_title("PEFT sanity (1 epoch, matched card)")
    ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.8)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(out_dir / f"fig2_peft_hatecheck.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    # Summary JSON for the paper / reproducibility
    key_labels = [
        "binary_legacy",
        "binary_compare",
        "structured",
        "ablation_no_rationale",
        "ablation_consistency",
        "structured_dh_hatexplain",
    ]
    summary = {
        "scope": (
            "Binary vs structured hate detection with TinyLlama+LoRA; multi-task structured heads; "
            "rationale and pair-consistency ablations; DynaHate+HateXplain joint training; "
            "cross-benchmark (HateEval) and HateCheck robustness; "
            "PEFT variant sanity (LoRA vs QLoRA vs DoRA)."
        ),
        "main_comparison": {
            key_labels[i]: {
                "dynahate_f1": float(dh[i]),
                "hateeval_f1": float(he[i]),
                "hatecheck_f1": float(hc[i]),
            }
            for i in range(len(key_labels))
        },
        "peft_hatecheck_f1": {peft[i][0]: float(peft[i][1]) for i in range(len(peft))},
    }
    (out_dir / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote figures and results_summary.json under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
