from __future__ import annotations

from pathlib import Path

import yaml

from boxmot.utils import BENCHMARK_CONFIGS
from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import RequirementsChecker

checker = RequirementsChecker()

ULTRALYTICS_MODELS = {"yolov8", "yolov9", "yolov10", "yolo11", "yolo12", "yolo26", "sam"}
RTDETR_MODELS = {"rtdetr_v2_r50vd", "rtdetr_v2_r18vd", "rtdetr_v2_r101vd"}
YOLOX_MODELS = {"yolox_n", "yolox_s", "yolox_m", "yolox_l", "yolox_x"}

# Suffixes/keywords that indicate a segmentation (mask-producing) model
_SEG_MARKERS = {"-seg", "_seg", "seg."}


def _check_model(name, markers):
    """Check if model name contains any of the markers."""
    return any(marker in str(name) for marker in markers)


def is_seg_model(name) -> bool:
    """Return True if the model name indicates a segmentation (mask-producing) model.

    Detection: yolo11n.pt, yolov8l.pt
    Segmentation: yolo11n-seg.pt, yolov8l-seg.pt
    """
    stem = Path(str(name)).stem.lower()
    return any(marker in stem or stem.endswith(marker.rstrip(".")) for marker in _SEG_MARKERS)


def _detector_name_key(name) -> str:
    """Normalize detector names so config lookup tolerates separator variants."""
    return Path(str(name)).stem.lower().replace("-", "").replace("_", "")


def is_ultralytics_model(yolo_name):
    return _check_model(yolo_name, ULTRALYTICS_MODELS)


def is_yolox_model(yolo_name):
    return _check_model(yolo_name, YOLOX_MODELS)


def is_rtdetr_model(yolo_name):
    return _check_model(yolo_name, RTDETR_MODELS)


def resolve_detector_cfg_path(yolo_name):
    """Return the detector-config YAML path whose detector model matches ``yolo_name``."""
    model_key = _detector_name_key(yolo_name)
    if not model_key:
        return None

    matches: list[Path] = []
    for cfg_path in sorted(BENCHMARK_CONFIGS.glob("*.yaml")):
        try:
            with open(cfg_path, "r") as handle:
                cfg = yaml.safe_load(handle) or {}
        except Exception:
            continue
        detector_cfg = cfg.get("detector")
        if not isinstance(detector_cfg, dict):
            continue
        detector_model = detector_cfg.get("model") or detector_cfg.get("default_model")
        if detector_model and _detector_name_key(detector_model) == model_key:
            matches.append(cfg_path)

    if matches:
        return matches[0]

    return None


def load_detector_cfg(yolo_name):
    """Load a detector config matching the detector model stem."""
    cfg_path = resolve_detector_cfg_path(yolo_name)
    if cfg_path is None:
        return {}

    with open(cfg_path, "r") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        return {}

    if isinstance(cfg.get("detector"), dict):
        detector_cfg = dict(cfg["detector"])
        detector_cfg.setdefault("id", detector_cfg.get("id") or f"{Path(cfg_path).stem}_detector")
        return detector_cfg

    return dict(cfg)


def get_detector_url(yolo_name):
    """Return the configured detector download URL for a detector model, if any."""
    detector_cfg = load_detector_cfg(yolo_name)
    model_url = detector_cfg.get("model_url") or detector_cfg.get("url")
    return str(model_url) if model_url else None


def get_runtime_detector_cfg(yolo_name, detector_cfg=None):
    """Return runtime detector settings, letting detector-config defaults override benchmark values."""
    runtime_cfg = dict(detector_cfg) if isinstance(detector_cfg, dict) else {}
    model_cfg = load_detector_cfg(yolo_name)
    if model_cfg:
        runtime_cfg.update(model_cfg)
    return runtime_cfg


def default_imgsz(yolo_name):
    """Return the detector fallback image size when no benchmark config is active."""
    detector_cfg = load_detector_cfg(yolo_name)
    if "imgsz" in detector_cfg:
        return list(detector_cfg["imgsz"])
    if is_yolox_model(yolo_name):
        return [1080, 1920]
    return [640, 640]


def default_conf(yolo_name):
    """Return the detector fallback confidence threshold when no benchmark config is active."""
    detector_cfg = load_detector_cfg(yolo_name)
    if "conf" in detector_cfg:
        return float(detector_cfg["conf"])
    return 0.01


def get_detector_class(yolo_model):
    """Return the detector backend class that matches the provided model reference."""
    model_name = str(yolo_model)

    detectors = [
        (
            is_yolox_model,
            ("yolox", "tabulate", "thop"),
            {"yolox": ["--no-deps"]},
            "boxmot.detectors.yolox",
            "YoloXDetector",
        ),
        (
            is_ultralytics_model,
            (),
            {},
            "boxmot.detectors.ultralytics",
            "UltralyticsDetector",
        ),
        (
            is_rtdetr_model,
            ("transformers[torch]", "timm"),
            {},
            "boxmot.detectors.rtdetr",
            "RTDetrDetector",
        ),
    ]

    for check_func, packages, extra_args, module_path, class_name in detectors:
        if check_func(model_name):
            for package in packages:
                try:
                    pkg_name = package.split("[")[0].split("=")[0]
                    __import__(pkg_name)
                except ImportError:
                    args = extra_args.get(pkg_name, [])
                    checker.check_packages((package,), extra_args=args)

            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)

    LOGGER.error(f"Failed to infer inference mode from yolo model name: {model_name}")
    LOGGER.error("Supported models must contain one of the following:")
    LOGGER.error(f"  Ultralytics: {ULTRALYTICS_MODELS}")
    LOGGER.error(f"  RTDetr: {RTDETR_MODELS}")
    LOGGER.error(f"  YOLOX: {YOLOX_MODELS}")
    LOGGER.error(
        "By using these names, the default COCO-trained models will be downloaded automatically. "
        "For custom models, the filename must include one of these substrings to route it to the correct package and architecture."
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
