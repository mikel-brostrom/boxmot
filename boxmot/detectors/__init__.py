from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from boxmot.detectors.base import Detections
    from boxmot.detectors.detector import Detector
    from boxmot.detectors.registry import (
        default_conf,
        default_imgsz,
        get_detector_class,
        get_detector_url,
        get_runtime_detector_cfg,
        is_rtdetr_model,
        is_seg_model,
        is_ultralytics_model,
        is_yolox_model,
        load_detector_cfg,
        resolve_detector_cfg_path,
    )

_EXPORTS = {
    "Detector": ("boxmot.detectors.detector", "Detector"),
    "Detections": ("boxmot.detectors.base", "Detections"),
    "default_conf": ("boxmot.detectors.registry", "default_conf"),
    "default_imgsz": ("boxmot.detectors.registry", "default_imgsz"),
    "get_detector_class": ("boxmot.detectors.registry", "get_detector_class"),
    "get_detector_url": ("boxmot.detectors.registry", "get_detector_url"),
    "get_runtime_detector_cfg": ("boxmot.detectors.registry", "get_runtime_detector_cfg"),
    "is_rtdetr_model": ("boxmot.detectors.registry", "is_rtdetr_model"),
    "is_seg_model": ("boxmot.detectors.registry", "is_seg_model"),
    "is_ultralytics_model": ("boxmot.detectors.registry", "is_ultralytics_model"),
    "is_yolox_model": ("boxmot.detectors.registry", "is_yolox_model"),
    "load_detector_cfg": ("boxmot.detectors.registry", "load_detector_cfg"),
    "resolve_detector_cfg_path": ("boxmot.detectors.registry", "resolve_detector_cfg_path"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
