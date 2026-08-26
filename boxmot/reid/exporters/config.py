"""Configuration defaults for ReID export workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

EXPORT_DEFAULTS_PATH = Path(__file__).resolve().parent / "defaults.yaml"


@dataclass(frozen=True, slots=True)
class ExportModeDefaults:
    """Typed export defaults shared by the CLI and Python API."""

    batch_size: int
    imgsz: tuple[int, int]
    device: str
    optimize: bool
    dynamic: bool
    simplify: bool
    opset: int
    workspace: int
    weights: str
    half: bool
    coreml_batch_buckets: tuple[int, ...]
    coreml_minimum_deployment_target: str
    coreml_compute_units: str
    coreml_timeout: float
    coreml_max_memory_gb: float
    tflite_quantize: str
    tflite_calibration_data: str | None
    tflite_calibration_samples: int
    tflite_calibration_preprocess: str
    tflite_calibration_seed: int
    tflite_calibration_update: str
    tflite_static_activation_bits: int
    include: tuple[str, ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExportModeDefaults":
        """Build typed defaults from the package-local YAML mapping."""
        image_size = tuple(int(value) for value in values.get("imgsz", (256, 128)))
        if len(image_size) != 2:
            raise ValueError("Export imgsz must contain exactly two integers")
        return cls(
            batch_size=int(values.get("batch_size", 1)),
            imgsz=(image_size[0], image_size[1]),
            device=str(values.get("device", "cpu")),
            optimize=bool(values.get("optimize", False)),
            dynamic=bool(values.get("dynamic", False)),
            simplify=bool(values.get("simplify", False)),
            opset=int(values.get("opset", 17)),
            workspace=int(values.get("workspace", 4)),
            weights=str(values.get("weights", "osnet_x0_25_msmt17")),
            half=bool(values.get("half", False)),
            coreml_batch_buckets=tuple(int(value) for value in values.get("coreml_batch_buckets", (1, 8, 16, 32))),
            coreml_minimum_deployment_target=str(
                values.get("coreml_minimum_deployment_target", "macOS15")
            ),
            coreml_compute_units=str(values.get("coreml_compute_units", "CPUAndGPU")),
            coreml_timeout=float(values.get("coreml_timeout", 600.0)),
            coreml_max_memory_gb=float(values.get("coreml_max_memory_gb", 16.0)),
            tflite_quantize=str(values.get("tflite_quantize", "none")),
            tflite_calibration_data=(
                None
                if values.get("tflite_calibration_data") in {None, ""}
                else str(values["tflite_calibration_data"])
            ),
            tflite_calibration_samples=int(values.get("tflite_calibration_samples", 256)),
            tflite_calibration_preprocess=str(values.get("tflite_calibration_preprocess", "resize")),
            tflite_calibration_seed=int(values.get("tflite_calibration_seed", 0)),
            tflite_calibration_update=str(values.get("tflite_calibration_update", "minmax")),
            tflite_static_activation_bits=int(values.get("tflite_static_activation_bits", 16)),
            include=tuple(str(value) for value in values.get("include", ("onnx",))),
        )


def load_export_defaults() -> ExportModeDefaults:
    """Load and validate the package-local ReID export defaults."""
    with open(EXPORT_DEFAULTS_PATH, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Export defaults must contain a YAML mapping: {EXPORT_DEFAULTS_PATH}")
    return ExportModeDefaults.from_mapping(payload)


__all__ = ("EXPORT_DEFAULTS_PATH", "ExportModeDefaults", "load_export_defaults")
