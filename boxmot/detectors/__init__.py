# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from pathlib import Path

from boxmot.detectors.detector import Detections

import yaml

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


def _load_detector_cfg(yolo_name) -> dict:
    """Load ``boxmot/configs/detectors/<stem>.yaml`` and return its contents (or empty dict)."""
    stem = Path(str(yolo_name)).stem
    cfg_path = DETECTOR_CONFIGS / f"{stem}.yaml"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            return cfg or {}
        except Exception as e:
            LOGGER.warning(f"Could not read detector config {cfg_path}: {e}")
    return {}


def default_imgsz(yolo_name):
    """Return the default image size from detector config YAML.

    Reads ``imgsz`` from ``boxmot/configs/detectors/<stem>.yaml``.
    Falls back to ``[640, 640]`` with a warning when the field or the file is absent.
    """
    cfg = _load_detector_cfg(yolo_name)
    if "imgsz" in cfg:
        return list(cfg["imgsz"])
    stem = Path(str(yolo_name)).stem
    LOGGER.warning(
        f"No 'imgsz' found in detector config for '{stem}'. "
        f"Using [640, 640]. Consider adding imgsz to "
        f"boxmot/configs/detectors/{stem}.yaml"
    )
    return [640, 640]


def default_conf(yolo_name):
    """Return the default confidence threshold from detector config YAML.

    Reads ``conf`` from ``boxmot/configs/detectors/<stem>.yaml``.
    Falls back to ``0.25`` with a warning when the field or the file is absent.
    """
    cfg = _load_detector_cfg(yolo_name)
    if "conf" in cfg:
        return float(cfg["conf"])
    stem = Path(str(yolo_name)).stem
    LOGGER.warning(
        f"No 'conf' found in detector config for '{stem}'. "
        f"Using 0.25. Consider adding conf to "
        f"boxmot/configs/detectors/{stem}.yaml"
    )
    return 0.25


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

