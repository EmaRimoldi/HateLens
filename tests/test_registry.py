from hatelens.registry import resolve_model_id


def test_resolve_registry_key():
    assert "Qwen" in resolve_model_id("qwen2.5-1.5b")


def test_resolve_passthrough():
    assert resolve_model_id("org/custom-model") == "org/custom-model"
