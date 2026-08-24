from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

from boxmot.detectors.config import (
    detector_config_to_runtime,
    iter_detector_config_paths,
    load_detector_config,
)
from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.config import ConfigurationError

checker = RequirementsChecker()

ULTRALYTICS_MODELS = ("yolov8", "yolov9", "yolov10", "yolo11", "yolo12", "yolo26", "sam")
RTDETR_MODELS = ("rtdetr_v2_r50vd", "rtdetr_v2_r18vd", "rtdetr_v2_r101vd")
YOLOX_MODELS = ("yolox_n", "yolox_s", "yolox_m", "yolox_l", "yolox_x")


@dataclass(frozen=True)
class DetectorBackendSpec:
    """Lazy import and optional-dependency metadata for one detector family."""

    matches: Callable[[str | Path], bool]
    module: str
    class_name: str
    requirements: tuple[str, ...] = ()
    requirement_args: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _model_name(name: str | Path) -> str:
    """Return a case-insensitive filename for family routing."""
    return Path(str(name)).name.lower()


def _check_model(name: str | Path, markers: tuple[str, ...]) -> bool:
    return any(marker in _model_name(name) for marker in markers)


def is_seg_model(name) -> bool:
    """Return True if the model name indicates a segmentation (mask-producing) model.

    Detection: yolo11n.pt, yolov8l.pt
    Segmentation: yolo11n-seg.pt, yolov8l-seg.pt
    """
    stem = Path(str(name)).stem.lower()
    return "-seg" in stem or "_seg" in stem


def _detector_name_key(name) -> str:
    """Normalize detector names so config lookup tolerates separator variants."""
    return Path(str(name)).stem.lower().replace("-", "").replace("_", "")


def is_ultralytics_model(model: str | Path) -> bool:
    return _check_model(model, ULTRALYTICS_MODELS)


def is_yolox_model(model: str | Path) -> bool:
    return _check_model(model, YOLOX_MODELS)


def is_rtdetr_model(model: str | Path) -> bool:
    return _check_model(model, RTDETR_MODELS)


DETECTOR_BACKENDS = (
    DetectorBackendSpec(
        matches=is_yolox_model,
        module="boxmot.detectors.yolox",
        class_name="YoloXDetector",
        requirements=("yolox", "tabulate", "thop"),
        requirement_args={"yolox": ("--no-deps",)},
    ),
    DetectorBackendSpec(
        matches=is_ultralytics_model,
        module="boxmot.detectors.ultralytics",
        class_name="UltralyticsDetector",
    ),
    DetectorBackendSpec(
        matches=is_rtdetr_model,
        module="boxmot.detectors.rtdetr",
        class_name="RTDetrDetector",
        requirements=("transformers[torch]", "timm"),
    ),
)


def _matching_detector_configs(model: str | Path) -> list[tuple[Path, dict, str]]:
    """Return validated detector profiles/checkpoints matching a model stem."""
    model_key = _detector_name_key(model)
    if not model_key:
        return []

    matches: list[tuple[Path, dict, str]] = []
    for config_path in iter_detector_config_paths():
        try:
            config = load_detector_config(config_path)
        except (ConfigurationError, FileNotFoundError, OSError):
            continue
        for checkpoint_name, checkpoint in config["checkpoints"].items():
            if _detector_name_key(checkpoint["path"]) == model_key:
                matches.append((config_path, config, checkpoint_name))
    return matches


def resolve_detector_cfg_path(model: str | Path) -> Path | None:
    """Return the detector profile whose checkpoint model matches ``model``."""
    matches = _matching_detector_configs(model)
    return matches[0][0] if matches else None


def load_detector_cfg(model: str | Path) -> dict:
    """Load a detector config matching the detector model stem."""
    matches = _matching_detector_configs(model)
    if not matches:
        return {}
    _, config, checkpoint_name = matches[0]
    return detector_config_to_runtime(config, checkpoint_name)


def get_detector_url(model: str | Path) -> str | None:
    """Return the configured detector download URL for a detector model, if any."""
    model_url = load_detector_cfg(model).get("model_url")
    return str(model_url) if model_url else None


def get_runtime_detector_cfg(model: str | Path, detector_cfg: dict | None = None) -> dict:
    """Return runtime detector settings, letting detector-config defaults override benchmark values."""
    runtime_cfg = dict(detector_cfg) if isinstance(detector_cfg, dict) else {}
    model_cfg = load_detector_cfg(model)
    if model_cfg:
        runtime_cfg.update(model_cfg)
    return runtime_cfg


def default_imgsz(model: str | Path) -> list[int]:
    """Return the detector fallback image size when no benchmark config is active."""
    detector_cfg = load_detector_cfg(model)
    if "imgsz" in detector_cfg:
        return list(detector_cfg["imgsz"])
    if is_yolox_model(model):
        return [1080, 1920]
    return [640, 640]


def default_conf(model: str | Path) -> float:
    """Return the detector fallback confidence threshold when no benchmark config is active."""
    detector_cfg = load_detector_cfg(model)
    if "conf" in detector_cfg:
        return float(detector_cfg["conf"])
    return 0.01


def get_detector_class(model: str | Path):
    """Return the detector backend class that matches the provided model reference."""
    model_name = str(model)

    for backend in DETECTOR_BACKENDS:
        if backend.matches(model_name):
            for package in backend.requirements:
                try:
                    pkg_name = package.split("[")[0].split("=")[0]
                    __import__(pkg_name)
                except ImportError:
                    args = backend.requirement_args.get(pkg_name, ())
                    checker.check_packages((package,), extra_args=args)

            return getattr(import_module(backend.module), backend.class_name)

    LOGGER.error(f"Failed to infer inference mode from yolo model name: {model_name}")
    LOGGER.error("Supported models must contain one of the following:")
    LOGGER.error(f"  Ultralytics: {ULTRALYTICS_MODELS}")
    LOGGER.error(f"  RTDetr: {RTDETR_MODELS}")
    LOGGER.error(f"  YOLOX: {YOLOX_MODELS}")
    LOGGER.error(
        "By using these names, the default COCO-trained models will be downloaded automatically. "
        "For custom models, the filename must include one of these substrings to route it to the "
        "correct package and architecture."
    )
    raise SystemExit(1)


__all__ = (
    "ULTRALYTICS_MODELS",
    "RTDETR_MODELS",
    "YOLOX_MODELS",
    "default_conf",
    "default_imgsz",
    "get_detector_class",
    "get_detector_url",
    "get_runtime_detector_cfg",
    "is_rtdetr_model",
    "is_seg_model",
    "is_ultralytics_model",
    "is_yolox_model",
    "load_detector_cfg",
    "resolve_detector_cfg_path",
)
