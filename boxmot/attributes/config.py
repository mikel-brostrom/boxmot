"""Configuration for optional per-track visual attribute extraction."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from boxmot.configs import CONFIG_ROOT
from boxmot.utils.config import ConfigurationError, load_yaml_mapping, resolve_config_path, validate_config_id


def _validate_schema_references(schema: dict[str, Any]) -> None:
    """Reject external references without interpreting examples as subschemas."""
    def visit(value: Any) -> None:
        if not isinstance(value, dict):
            return
        for name in ("$ref", "$dynamicRef", "$recursiveRef"):
            if name in value and (not isinstance(value[name], str) or not value[name].startswith("#")):
                raise ConfigurationError("Attribute schema references must be local fragments beginning with '#'.")
        for name in ("$defs", "definitions", "properties", "patternProperties", "dependentSchemas", "dependencies"):
            children = value.get(name)
            if isinstance(children, dict):
                for child in children.values():
                    visit(child)
        for name in (
            "allOf", "anyOf", "oneOf", "prefixItems", "items", "additionalItems", "contains",
            "additionalProperties", "unevaluatedProperties", "unevaluatedItems", "propertyNames",
            "not", "if", "then", "else",
        ):
            children = value.get(name)
            for child in children if isinstance(children, list) else (children,):
                visit(child)

    visit(schema)


@dataclass(frozen=True, slots=True)
class AttributeConfig:
    """Authored extraction settings; inference dependencies load only when used."""

    model: str
    prompt: str
    schema: dict[str, Any]
    backend: str = "ollama"
    host: str | None = None
    class_ids: tuple[int, ...] | None = None
    sample_every: int = 30
    batch_size: int = 5
    request_timeout_s: float = 120
    retries: int = 2
    display_fields: tuple[str, ...] = ()
    max_crop_side: int = 640
    jpeg_quality: int = 90
    think: bool | None = False  # None leaves the provider's thinking setting unchanged.
    api_key_env: str | None = None
    database_url_env: str = "BOXMOT_ATTRIBUTES_DATABASE_URL"
    response_format: str = "json_schema"  # vLLM only; Ollama always sends the schema.

    def __post_init__(self) -> None:
        """Validate portable settings and detach mutable authored schema data."""
        if self.backend not in ("ollama", "vllm"):
            raise ConfigurationError("Attribute backend must be 'ollama' or 'vllm'.")
        if self.response_format not in ("json_schema", "json_object"):
            raise ConfigurationError("Attribute response_format must be 'json_schema' or 'json_object'.")
        if self.backend == "ollama" and self.response_format != "json_schema":
            raise ConfigurationError("Attribute response_format 'json_object' requires the 'vllm' backend.")
        if self.host is None:
            object.__setattr__(
                self, "host", "http://localhost:11434" if self.backend == "ollama" else "http://localhost:8000"
            )
        for name in ("model", "prompt", "host"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"Attribute {name} must be a nonempty string.")
        try:
            endpoint = urlsplit(self.host)
            endpoint.port
        except ValueError as exc:
            raise ConfigurationError("Attribute host must be a valid HTTP(S) server URL.") from exc
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc or endpoint.query or endpoint.fragment:
            raise ConfigurationError("Attribute host must be an HTTP(S) server URL without query or fragment.")
        for name, minimum, maximum in (
            ("sample_every", 1, None),
            ("batch_size", 1, 5),
            ("retries", 0, None),
            ("max_crop_side", 1, None),
            ("jpeg_quality", 1, 100),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ConfigurationError(f"Attribute {name} must be an integer >= {minimum}.")
            if maximum is not None and value > maximum:
                raise ConfigurationError(f"Attribute {name} must be <= {maximum}.")
        if (
            isinstance(self.request_timeout_s, bool)
            or not isinstance(self.request_timeout_s, (int, float))
            or not math.isfinite(self.request_timeout_s)
            or self.request_timeout_s <= 0
        ):
            raise ConfigurationError("Attribute request_timeout_s must be finite and positive.")
        if self.think is not None and not isinstance(self.think, bool):
            raise ConfigurationError("Attribute think must be a bool or null.")
        if self.api_key_env is not None and (
            not isinstance(self.api_key_env, str) or not self.api_key_env.isidentifier()
        ):
            raise ConfigurationError("Attribute api_key_env must be a nonempty environment variable name.")
        if (
            not isinstance(self.database_url_env, str)
            or not self.database_url_env.isascii()
            or not self.database_url_env.isidentifier()
        ):
            raise ConfigurationError("Attribute database_url_env must be a nonempty environment variable name.")
        if self.class_ids is not None:
            if not isinstance(self.class_ids, (list, tuple)) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.class_ids
            ):
                raise ConfigurationError("Attribute class_ids must contain nonnegative integer class IDs.")
            object.__setattr__(self, "class_ids", tuple(dict.fromkeys(self.class_ids)))
        if not isinstance(self.display_fields, (list, tuple)) or any(
            not isinstance(value, str) or not value.strip() for value in self.display_fields
        ):
            raise ConfigurationError("Attribute display_fields must contain nonempty field paths.")
        object.__setattr__(self, "display_fields", tuple(self.display_fields))
        if not isinstance(self.schema, Mapping) or self.schema.get("type") != "object":
            raise ConfigurationError("Attribute schema must declare an object root with type: object.")
        try:
            schema = json.loads(json.dumps(dict(self.schema), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("Attribute schema must contain JSON-compatible values.") from exc
        _validate_schema_references(schema)
        object.__setattr__(self, "schema", schema)


def resolve_attribute_config(reference: AttributeConfig | str | Path | Mapping[str, Any]) -> AttributeConfig:
    """Resolve an immutable config, authored mapping, YAML path, or catalog ID."""
    if isinstance(reference, AttributeConfig):
        return reference
    if isinstance(reference, (str, Path)):
        path = resolve_config_path(CONFIG_ROOT / "attributes", reference, "attribute")
        payload = load_yaml_mapping(path)
        if "id" in payload:
            validate_config_id(payload["id"], path=path, label="attribute")
    elif isinstance(reference, Mapping):
        payload = dict(reference)
    else:
        raise TypeError("Attributes must be an AttributeConfig, mapping, YAML path, or built-in profile ID.")
    payload.pop("id", None)
    unknown = set(payload) - {item.name for item in fields(AttributeConfig)}
    if unknown:
        raise ConfigurationError(f"Unknown attribute configuration fields: {', '.join(sorted(unknown))}.")
    missing = {"model", "prompt", "schema"} - set(payload)
    if missing:
        raise ConfigurationError(f"Missing attribute configuration fields: {', '.join(sorted(missing))}.")
    return AttributeConfig(**payload)
