# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import RequirementsChecker

checker = RequirementsChecker()

ULTRALYTICS_MODELS = {"yolov8", "yolov9", "yolov10", "yolo11", "yolo12", "sam"}
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


def default_imgsz(yolo_name):
    if is_ultralytics_model(yolo_name):
        return [640, 640]
    elif is_yolox_model(yolo_name):
        return [800, 1440]
    else:
        return [640, 640]


def get_yolo_inferer(yolo_model):
    """
    Determines and returns the appropriate inference strategy class based on the model name.
    Handles dependency checks and imports dynamically.
    """
    model_name = str(yolo_model)

    strategies = [
        (
            is_yolox_model,
            ("yolox", "tabulate", "thop"),
            {"yolox": ["--no-deps"]},
            "boxmot.detectors.yolox",
            "YoloXStrategy",
        ),
        (
            is_ultralytics_model,
            (),
            {},
            "boxmot.detectors.ultralytics",
            "UltralyticsStrategy",
        ),
        (
            is_rtdetr_model,
            ("transformers[torch]", "timm"),
            {},
            "boxmot.detectors.rtdetr",
            "RTDetrStrategy",
        ),
    ]

    for check_func, packages, extra_args, module_path, class_name in strategies:
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

