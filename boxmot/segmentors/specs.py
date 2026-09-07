"""Immutable segmentor construction specifications."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | tuple["JSONValue", ...]

_BACKEND_PATTERN = re.compile(r"[a-z][a-z0-9_-]*")
_OPTION_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PRECISIONS = frozenset(("fp16", "fp32", "bf16"))
_GEOMETRY_MODES = frozenset(("auto", "aabb", "obb"))


def _validate_json_value(value: JSONValue, *, location: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must not contain non-finite numbers.")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_json_value(item, location=f"{location}[{index}]")
        return
    raise TypeError(f"{location} must contain only immutable JSON values, not {type(value).__name__}.")


@dataclass(frozen=True, slots=True)
class SegmentorSpec:
    """Everything needed to construct and fingerprint a segmentor runtime."""

    backend: str
    artifact: str | None = None
    artifact_sha256: str | None = None
    device: str = "cpu"
    precision: str = "fp32"
    options: tuple[tuple[str, JSONValue], ...] = ()
    preprocessing: str = "default"
    geometry_mode: str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or _BACKEND_PATTERN.fullmatch(self.backend) is None:
            raise ValueError("backend must be a non-empty canonical lowercase identifier.")
        if self.artifact is not None and (
            not isinstance(self.artifact, str) or not self.artifact or self.artifact != self.artifact.strip()
        ):
            raise ValueError("artifact must be a non-empty canonical string when provided.")
        if self.artifact_sha256 is not None:
            if not isinstance(self.artifact_sha256, str) or _SHA256_PATTERN.fullmatch(self.artifact_sha256) is None:
                raise ValueError("artifact_sha256 must be a lowercase 64-character SHA-256 digest.")
            if self.artifact is None:
                raise ValueError("artifact_sha256 requires an artifact reference.")
        if not isinstance(self.device, str) or not self.device or self.device != self.device.strip():
            raise ValueError("device must be a non-empty canonical string.")
        if not isinstance(self.preprocessing, str) or _BACKEND_PATTERN.fullmatch(self.preprocessing) is None:
            raise ValueError("preprocessing must be a non-empty canonical lowercase identifier.")
        if self.geometry_mode not in _GEOMETRY_MODES:
            available = ", ".join(sorted(_GEOMETRY_MODES))
            raise ValueError(f"geometry_mode must be one of: {available}.")
        if self.precision not in _PRECISIONS:
            available = ", ".join(sorted(_PRECISIONS))
            raise ValueError(f"precision must be one of: {available}.")
        if not isinstance(self.options, tuple):
            raise TypeError("options must be a tuple of (key, value) pairs.")

        keys: list[str] = []
        for index, option in enumerate(self.options):
            if not isinstance(option, tuple) or len(option) != 2:
                raise TypeError(f"options[{index}] must be a (key, value) tuple.")
            key, value = option
            if not isinstance(key, str) or _OPTION_PATTERN.fullmatch(key) is None:
                raise ValueError(f"options[{index}] key must be a canonical lowercase identifier.")
            keys.append(key)
            _validate_json_value(value, location=f"option {key!r}")
        if len(keys) != len(set(keys)):
            raise ValueError("option keys must be unique.")
        if keys != sorted(keys):
            raise ValueError("option keys must be sorted in canonical order.")

    def option_values(self) -> dict[str, JSONValue]:
        """Return backend options as a new mutable mapping."""
        return dict(self.options)


__all__ = ("JSONScalar", "JSONValue", "SegmentorSpec")
