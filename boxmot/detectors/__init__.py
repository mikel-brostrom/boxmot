# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from pathlib import Path

import yaml

from boxmot.detectors.detector import Detections

from boxmot.utils import DETECTOR_CONFIGS, logger as LOGGER
from boxmot.utils.checks import RequirementsChecker

checker = RequirementsChecker()

ULTRALYTICS_MODELS = {"yolov8", "yolov9", "yolov10", "yolo11", "yolo12", "yolo26", "sam"}
RTDETR_MODELS = {"rtdetr_v2_r50vd", "rtdetr_v2_r18vd", "rtdetr_v2_r101vd"}
YOLOX_MODELS = {"yolox_n", "yolox_s", "yolox_m", "yolox_l", "yolox_x"}


def _check_model(name, markers):
    """Check if model name contains any of the markers."""
    return any(m in str(name) for m in markers)


def is_ultralytics_model(yolo_name):
    return _check_model(yolo_name, ULTRALYTICS_MODELS)


def is_yolox_model(yolo_name):
    return _check_model(yolo_name, YOLOX_MODELS)


def is_rtdetr_model(yolo_name):
    return _check_model(yolo_name, RTDETR_MODELS)


def resolve_detector_cfg_path(yolo_name):
    """Return the matching detector YAML path for a model name, if one exists."""
    stem = Path(str(yolo_name)).name
    stem = Path(stem).stem.lower()
    if not stem or not DETECTOR_CONFIGS.exists():
        return None

    for suffix in (".yaml", ".yml"):
        exact = DETECTOR_CONFIGS / f"{stem}{suffix}"
        if exact.exists():
            return exact

    matches = sorted(
        p
        for pattern in ("*.yaml", "*.yml")
        for p in DETECTOR_CONFIGS.glob(pattern)
        if p.stem.lower() == stem
    )
    return matches[0] if matches else None


def load_detector_cfg(yolo_name):
    """Load a detector config YAML matching the detector model stem, if present."""
    cfg_path = resolve_detector_cfg_path(yolo_name)
    if cfg_path is None:
        return {}

    with open(cfg_path, "r") as handle:
        cfg = yaml.safe_load(handle) or {}
    return dict(cfg) if isinstance(cfg, dict) else {}


def get_runtime_detector_cfg(yolo_name, detector_cfg=None):
    """Return runtime detector settings, letting a model-matched YAML override benchmark values."""
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
    """
    Determines and returns the appropriate detector class based on the model name.
    Handles dependency checks and imports dynamically.
    """
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
                    # Simple import check for package name (stripping version/extras)
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
    exit()
