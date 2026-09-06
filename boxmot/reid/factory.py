"""Factory for canonical appearance encoder components."""

from __future__ import annotations

import inspect
from collections.abc import Callable

from boxmot.components.artifacts import require_resolved_artifact
from boxmot.components.registry import LazyComponentRegistry
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


def create_reid_encoder(spec: ReIDEncoderSpec) -> AppearanceEncoder:
    """Construct the appearance encoder described by ``spec``."""
    if not isinstance(spec, ReIDEncoderSpec):
        raise TypeError(f"spec must be a ReIDEncoderSpec, not {type(spec).__name__}.")
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
