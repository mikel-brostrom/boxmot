"""Lazy registry for ReID model exporters."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from boxmot.reid.core.formats import ReIDFormat


@dataclass(frozen=True, slots=True)
class ExporterSpec:
    """Import location for the exporter producing one artifact format."""

    format_id: str
    module: str
    class_name: str

    def resolve(self) -> type[Any]:
        """Import and return the configured exporter class."""
        return getattr(import_module(self.module), self.class_name)


EXPORTER_SPECS = {
    spec.format_id: spec
    for spec in (
        ExporterSpec("torchscript", "boxmot.reid.exporters.torchscript_exporter", "TorchScriptExporter"),
        ExporterSpec("onnx", "boxmot.reid.exporters.onnx_exporter", "ONNXExporter"),
        ExporterSpec("openvino", "boxmot.reid.exporters.openvino_exporter", "OpenVINOExporter"),
        ExporterSpec("tensorrt", "boxmot.reid.exporters.tensorrt_exporter", "EngineExporter"),
        ExporterSpec("coreml", "boxmot.reid.exporters.coreml_exporter", "CoreMLExporter"),
        ExporterSpec("tflite", "boxmot.reid.exporters.tflite_exporter", "TFLiteExporter"),
    )
}


def get_exporter_class(format_: ReIDFormat | str) -> type[Any]:
    """Resolve the exporter class registered for an artifact format."""
    format_id = format_.id if isinstance(format_, ReIDFormat) else str(format_)
    try:
        spec = EXPORTER_SPECS[format_id]
    except KeyError as exc:
        raise KeyError(f"No ReID exporter registered for format {format_id!r}") from exc
    return spec.resolve()


__all__ = ("EXPORTER_SPECS", "ExporterSpec", "get_exporter_class")
