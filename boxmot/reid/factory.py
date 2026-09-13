"""Factory for canonical appearance encoder components."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, overload

from boxmot.components.artifacts import require_resolved_artifact
from boxmot.components.registry import LazyComponentRegistry
from boxmot.reid._model_names import ReIDName
from boxmot.reid.protocols import AppearanceEncoder, EncoderRequirements
from boxmot.reid.specs import ReIDEncoderSpec

ReIDEncoderFactory = Callable[[ReIDEncoderSpec], AppearanceEncoder]

_REID_ENCODER_FACTORIES: LazyComponentRegistry[ReIDEncoderFactory] = LazyComponentRegistry(
    "ReID",
    {
        "pytorch": "boxmot.reid.adapters:create_python_reid_encoder",
        "torchscript": "boxmot.reid.adapters:create_python_reid_encoder",
        "onnx": "boxmot.reid.adapters:create_python_reid_encoder",
        "openvino": "boxmot.reid.adapters:create_python_reid_encoder",
        "tensorrt": "boxmot.reid.adapters:create_python_reid_encoder",
        "tflite": "boxmot.reid.adapters:create_python_reid_encoder",
        "coreml": "boxmot.reid.adapters:create_python_reid_encoder",
        "native": "boxmot.reid.adapters:create_native_reid_encoder",
    },
)


@overload
def create_reid_encoder(
    spec: ReIDName,
    *,
    device: str | None = None,
    precision: str | None = None,
    preprocessing: str | None = None,
    options: Mapping[str, Any] | None = None,
    allow_download: bool = True,
) -> AppearanceEncoder: ...


@overload
def create_reid_encoder(
    spec: ReIDEncoderSpec | str | Path | Mapping[str, Any],
    *,
    device: str | None = None,
    precision: str | None = None,
    preprocessing: str | None = None,
    options: Mapping[str, Any] | None = None,
    allow_download: bool = True,
) -> AppearanceEncoder: ...


def create_reid_encoder(
    spec: ReIDEncoderSpec | str | Path | Mapping[str, Any],
    *,
    device: str | None = None,
    precision: str | None = None,
    preprocessing: str | None = None,
    options: Mapping[str, Any] | None = None,
    allow_download: bool = True,
) -> AppearanceEncoder:
    """Construct an encoder from a resolved spec, profile, YAML, or model artifact.

    Explicit runtime keywords override the selected configuration. ``options``
    merges backend settings by key, preserving authored settings not overridden.
    References resolve and verify model artifacts, downloading missing weights
    when allowed; an explicit ``ReIDEncoderSpec`` must already be resolved.
    """
    if not isinstance(spec, (ReIDEncoderSpec, str, Path, Mapping)):
        raise TypeError(
            f"spec must be a ReIDEncoderSpec, name, path, or configuration mapping, not {type(spec).__name__}."
        )
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("options must be a mapping when provided.")
    if type(allow_download) is not bool:
        raise TypeError("allow_download must be a boolean.")
    if not isinstance(spec, ReIDEncoderSpec):
        from boxmot.reid.config import resolve_reid_spec

        spec, _ = resolve_reid_spec(spec, allow_download=allow_download)
    overrides = {
        name: value
        for name, value in (("device", device), ("precision", precision), ("preprocessing", preprocessing))
        if value is not None
    }
    if options is not None:
        from boxmot.components.resolution import component_options

        overrides["options"] = component_options({**spec.option_values(), **options})
    if overrides:
        spec = replace(spec, **overrides)
    require_resolved_artifact(
        spec.artifact,
        spec.artifact_sha256,
        component=f"ReID backend {spec.backend!r}",
    )
    factory = _REID_ENCODER_FACTORIES.resolve(spec.backend)
    encoder = factory(spec)
    if (
        not callable(getattr(encoder, "encode", None))
        or inspect.getattr_static(encoder, "embedding_dim", None) is None
        or not isinstance(getattr(encoder, "requirements", None), EncoderRequirements)
    ):
        raise TypeError(f"ReID backend {spec.backend!r} does not implement the appearance encoder protocol.")
    return encoder


__all__ = ("create_reid_encoder",)
