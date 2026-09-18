"""Immutable ReID encoder construction specifications."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, TypeAlias, overload

from boxmot.reid._model_names import ReIDName

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | tuple["JSONValue", ...]

_IDENTIFIER_PATTERN = re.compile(r"[a-z][a-z0-9_-]*")
_OPTION_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PRECISIONS = frozenset(("fp16", "fp32", "bf16"))


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
class ReIDEncoderSpec:
    """Everything needed to construct and fingerprint an appearance encoder."""

    backend: str
    artifact: str | None = None
    artifact_sha256: str | None = None
    device: str = "cpu"
    precision: str = "fp32"
    options: tuple[tuple[str, JSONValue], ...] = ()
    preprocessing: str = "default"

    def __post_init__(self) -> None:
        for name, value in (
            ("backend", self.backend),
            ("preprocessing", self.preprocessing),
        ):
            if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{name} must be a non-empty canonical lowercase identifier.")
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


@dataclass(frozen=True, slots=True, init=False)
class ReIDConfig:
    """Reusable appearance-model selection and inference settings.

    Omitted overrides inherit the selected profile's settings. Construction is
    lazy when this configuration is passed to a tracker; creating this object
    does not resolve artifacts or download weights.

    """

    model: str | Path = "osnet-x0-25-msmt17"
    device: str | None = None
    precision: Literal["fp16", "fp32"] | None = None
    preprocessing: str | None = None
    batch_size: int | None = None
    image_size: tuple[int, int] | None = None
    embedding_dim: int | None = None
    allow_download: bool = True

    @overload
    def __init__(
        self,
        model: ReIDName = "osnet-x0-25-msmt17",
        *,
        device: str | None = None,
        precision: Literal["fp16", "fp32"] | None = None,
        preprocessing: str | None = None,
        batch_size: int | None = None,
        image_size: tuple[int, int] | None = None,
        embedding_dim: int | None = None,
        allow_download: bool = True,
    ) -> None: ...

    @overload
    def __init__(
        self,
        model: str | Path = "osnet-x0-25-msmt17",
        *,
        device: str | None = None,
        precision: Literal["fp16", "fp32"] | None = None,
        preprocessing: str | None = None,
        batch_size: int | None = None,
        image_size: tuple[int, int] | None = None,
        embedding_dim: int | None = None,
        allow_download: bool = True,
    ) -> None: ...

    def __init__(
        self,
        model: str | Path = "osnet-x0-25-msmt17",
        *,
        device: str | None = None,
        precision: Literal["fp16", "fp32"] | None = None,
        preprocessing: str | None = None,
        batch_size: int | None = None,
        image_size: tuple[int, int] | None = None,
        embedding_dim: int | None = None,
        allow_download: bool = True,
    ) -> None:
        """Select a model and retain only explicit inference overrides.

        Args:
            model: Built-in model/profile name, custom weights, or encoder YAML path.
                Defaults to OSNet x0.25 trained on MSMT17.
            device: Inference device override, such as ``"cpu"`` or ``"cuda:0"``.
            precision: FP16 or FP32 inference override; the selected backend validates
                support during encoder construction.
            preprocessing: Crop preprocessing override, such as ``"resize"`` or
                ``"resize_pad"``. Crop geometry follows the input detections.
            batch_size: Maximum number of crops per inference batch; the encoder
                default is 64 when neither the profile nor this config specifies it.
            image_size: Crop height and width override for the selected encoder.
            embedding_dim: Descriptor width when the backend cannot declare it.
            allow_download: Allow downloading missing model artifacts on resolution.
        """
        if not isinstance(model, (str, Path)):
            raise TypeError("reid.model must be a model name, string path, or Path.")
        if isinstance(model, str) and (not model or model != model.strip()):
            raise ValueError("reid.model must be a non-empty canonical string.")
        if device is not None and (not isinstance(device, str) or not device or device != device.strip()):
            raise ValueError("reid.device must be a non-empty canonical string or None.")
        if precision is not None and (not isinstance(precision, str) or precision not in {"fp16", "fp32"}):
            raise ValueError("reid.precision must be fp16, fp32, or None.")
        if preprocessing is not None and (
            not isinstance(preprocessing, str) or _IDENTIFIER_PATTERN.fullmatch(preprocessing) is None
        ):
            raise ValueError("reid.preprocessing must be a canonical lowercase identifier or None.")
        for name, value in (("batch_size", batch_size), ("embedding_dim", embedding_dim)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"reid.{name} must be a positive integer or None.")
        if image_size is not None and (
            not isinstance(image_size, tuple)
            or len(image_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_size)
        ):
            raise ValueError("reid.image_size must be a two-integer (height, width) tuple or None.")
        if not isinstance(allow_download, bool):
            raise TypeError("reid.allow_download must be bool.")
        for name, value in (
            ("model", str(model)),
            ("device", device),
            ("precision", precision),
            ("preprocessing", preprocessing),
            ("batch_size", batch_size),
            ("image_size", image_size),
            ("embedding_dim", embedding_dim),
            ("allow_download", allow_download),
        ):
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        """Return YAML-ready settings, retaining omitted profile overrides."""
        values: dict[str, Any] = {"model": str(self.model)}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name != "model" and value is not None:
                values[item.name] = list(value) if item.name == "image_size" else value
        return values

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ReIDConfig:
        """Validate a friendly YAML mapping without resolving its model."""
        if not isinstance(values, Mapping):
            raise TypeError("reid must contain a configuration mapping.")
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise TypeError(f"Unexpected reid field {next(iter(unknown))!r}.")
        payload = dict(values)
        if isinstance(payload.get("image_size"), list):
            payload["image_size"] = tuple(payload["image_size"])
        return cls(**payload)


__all__ = ("JSONScalar", "JSONValue", "ReIDConfig", "ReIDEncoderSpec")
