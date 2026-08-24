"""Lazy registry for ReID runtime backends."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from boxmot.reid.core.formats import ReIDFormat


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """Import location for the backend serving one artifact format."""

    format_id: str
    module: str
    class_name: str

    def resolve(self) -> type[Any]:
        """Import and return the configured backend class."""
        return getattr(import_module(self.module), self.class_name)


BACKEND_SPECS = {
    spec.format_id: spec
    for spec in (
        BackendSpec("pytorch", "boxmot.reid.backends.pytorch_backend", "PyTorchBackend"),
        BackendSpec("torchscript", "boxmot.reid.backends.torchscript_backend", "TorchscriptBackend"),
        BackendSpec("onnx", "boxmot.reid.backends.onnx_backend", "ONNXBackend"),
        BackendSpec("openvino", "boxmot.reid.backends.openvino_backend", "OpenVinoBackend"),
        BackendSpec("tensorrt", "boxmot.reid.backends.tensorrt_backend", "TensorRTBackend"),
        BackendSpec("coreml", "boxmot.reid.backends.coreml_backend", "CoreMLBackend"),
        BackendSpec("tflite", "boxmot.reid.backends.tflite_backend", "TFLiteBackend"),
    )
}


def get_backend_class(format_: ReIDFormat | str) -> type[Any]:
    """Resolve the backend class registered for an artifact format."""
    format_id = format_.id if isinstance(format_, ReIDFormat) else str(format_)
    try:
        spec = BACKEND_SPECS[format_id]
    except KeyError as exc:
        raise KeyError(f"No ReID backend registered for format {format_id!r}") from exc
    return spec.resolve()


__all__ = ("BACKEND_SPECS", "BackendSpec", "get_backend_class")
