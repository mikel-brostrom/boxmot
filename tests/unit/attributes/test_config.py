from dataclasses import FrozenInstanceError

import pytest

from boxmot.attributes.config import AttributeConfig, resolve_attribute_config
from boxmot.utils.config import ConfigurationError


def _config(**overrides):
    values = {"model": "vision", "prompt": "Describe the object.", "schema": {"type": "object"}}
    values.update(overrides)
    return AttributeConfig(**values)


def test_resolves_builtin_profile_and_detaches_custom_schema(tmp_path):
    config = resolve_attribute_config("person-appearance")
    assert config.model == "qwen3.5:9b"
    assert config.class_ids == (0,)
    assert config.batch_size == 5
    assert config.sample_every == 30
    assert config.schema["properties"]["pose"]["enum"][-1] == "Unknown"
    assert "gender" not in config.schema["properties"]
    with pytest.raises(FrozenInstanceError):
        config.model = "changed"
    schema = {"type": "object", "properties": {"color": {"type": "string"}}}
    custom = _config(schema=schema, class_ids=[2, 2, 7], display_fields=["color"])
    schema["properties"].clear()
    assert "color" in custom.schema["properties"]
    assert custom.class_ids == (2, 7)
    assert custom.display_fields == ("color",)
    assert resolve_attribute_config(custom) is custom
    path = tmp_path / "custom.yaml"
    path.write_text("model: vision\nprompt: Describe\nschema: {type: object}\n", encoding="utf-8")
    assert resolve_attribute_config(path).model == "vision"
    assert resolve_attribute_config({"model": "vision", "prompt": "Describe", "schema": {"type": "object"}}).model


@pytest.mark.parametrize(
    "values",
    [
        {"backend": "other"},
        {"model": " "},
        {"prompt": ""},
        {"host": "localhost:11434"},
        {"host": "http://localhost:bad"},
        {"sample_every": 0},
        {"sample_every": True},
        {"batch_size": 6},
        {"retries": -1},
        {"request_timeout_s": float("inf")},
        {"request_timeout_s": 0},
        {"max_crop_side": 0},
        {"jpeg_quality": 101},
        {"class_ids": [True]},
        {"class_ids": [-1]},
        {"display_fields": "color"},
        {"think": "false"},
        {"think": 0},
        {"response_format": "text"},
        {"response_format": None},
        {"response_format": "json_object"},
        {"api_key_env": ""},
        {"api_key_env": "Bearer secret"},
        {"api_key_env": 7},
        {"schema": {"type": "array"}},
        {"schema": {"type": "object", "$ref": "https://example.com/schema.json"}},
        {"schema": {"type": "object", "$dynamicRef": "other.json#anchor"}},
        {"schema": {"type": "object", "default": float("nan")}},
    ],
)
def test_rejects_invalid_settings(values):
    with pytest.raises(ConfigurationError):
        _config(**values)


def test_rejects_unknown_or_missing_authored_fields():
    with pytest.raises(ConfigurationError, match="Unknown"):
        resolve_attribute_config({"wrong_key": 3})
    with pytest.raises(ConfigurationError, match="Missing"):
        resolve_attribute_config({"model": "vision"})


def test_accepts_local_schema_references():
    schema = {
        "type": "object",
        "$defs": {"word": {"type": "string", "$anchor": "word"}},
        "properties": {"color": {"$ref": "#/$defs/word"}, "shape": {"$ref": "#word"}},
    }
    assert _config(schema=schema).schema == schema


def test_backend_specific_defaults_and_vllm_profile():
    assert _config().host == "http://localhost:11434"
    assert _config(backend="vllm").host == "http://localhost:8000"
    assert _config(backend="vllm", host="https://vision.example/v1").host == "https://vision.example/v1"
    assert _config(api_key_env="ATTRIBUTE_API_KEY").api_key_env == "ATTRIBUTE_API_KEY"
    config = resolve_attribute_config("person-appearance-vllm")
    ollama = resolve_attribute_config("person-appearance")
    assert config.backend == "vllm"
    assert config.model == "Qwen/Qwen3.5-9B"
    assert config.host == "http://localhost:8000"
    assert config.schema == ollama.schema
    assert config.prompt == ollama.prompt
    assert config.display_fields == ollama.display_fields
    assert config.think is False
    assert config.response_format == "json_schema"


def test_json_object_mode_and_optional_thinking_from_yaml(tmp_path):
    """Provider profiles can opt out of schema and thinking request extensions."""
    path = tmp_path / "provider.yaml"
    path.write_text(
        "backend: vllm\nmodel: vision\nprompt: Describe\nschema: {type: object}\n"
        "response_format: json_object\nthink: null\n",
        encoding="utf-8",
    )
    config = resolve_attribute_config(path)
    assert config.response_format == "json_object"
    assert config.think is None
    assert _config(think=None).think is None


def test_reference_named_properties_and_examples_are_not_schema_references():
    schema = {
        "type": "object",
        "properties": {"$ref": {"type": "string"}},
        "examples": [{"$ref": "https://example.org"}],
    }
    assert _config(schema=schema).schema == schema
