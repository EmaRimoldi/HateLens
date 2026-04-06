from hatelens.parsing.structured_output import normalize_prediction, parse_structured_json


def test_parse_json_with_extra_text():
    raw = 'Sure: {"label": "hate", "target_group": "x", "hate_type": "y", "explicitness": "explicit", "rationale": "z"}'
    p = parse_structured_json(raw)
    assert p is not None
    n = normalize_prediction(p)
    assert n["label"] == "hate"
