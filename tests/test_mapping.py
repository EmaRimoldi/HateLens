from hatelens.mapping import dynahate_row_to_unified, example_to_json_target


def test_dynahate_unified_carries_target_and_type():
    ex = dynahate_row_to_unified(
        {"text": "x", "label": 1, "target": "women", "type": "derogation"},
        idx=0,
        split="train",
    )
    assert ex.label == "hate"
    assert ex.target_group == "women"
    assert ex.hate_type == "derogation"
    js = example_to_json_target(ex)
    assert "women" in js and "derogation" in js
