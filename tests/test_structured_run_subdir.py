"""Structured run directory naming (ablations must not clobber default checkpoint)."""

from hatelens.structured_train import structured_run_subdir


def test_structured_run_subdir_default():
    assert structured_run_subdir("dynahate", {}) == "structured_dynahate"


def test_structured_run_subdir_with_suffix():
    assert structured_run_subdir("dynahate", {"structured_output_suffix": "_no_rationale"}) == (
        "structured_dynahate_no_rationale"
    )


def test_structured_run_subdir_suffix_adds_underscore():
    assert structured_run_subdir("dynahate", {"structured_output_suffix": "consistency"}) == (
        "structured_dynahate_consistency"
    )
