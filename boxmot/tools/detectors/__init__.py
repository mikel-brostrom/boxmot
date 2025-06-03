# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import RequirementsChecker

checker = RequirementsChecker()

UL_MODELS = ["yolov8", "yolov9", "yolov10", "yolo11", "yolo12", "rtdetr", "sam"]


def is_ultralytics_model(yolo_name):
    return any(yolo in str(yolo_name) for yolo in UL_MODELS)


def is_yolox_model(yolo_name):
    return "yolox" in str(yolo_name)


def default_imgsz(yolo_name):
    if is_ultralytics_model(yolo_name):
        return [640, 640]
    elif is_yolox_model(yolo_name):
        return [800, 1440]
    else:
        return [640, 640]


def get_yolo_inferer(yolo_model):

    if is_yolox_model(yolo_model):
        try:
            import yolox  # for linear_assignment

            assert yolox.__version__
        except (ImportError, AssertionError, AttributeError):
            checker.check_packages(("yolox",), cmds=["--no-deps"])
            checker.check_packages(("tabulate",))  # needed dependency
            checker.check_packages(("thop",))  # needed dependency
        from .yolox import YoloXStrategy

        return YoloXStrategy
    elif is_ultralytics_model(yolo_model):
        # ultralytics already installed when running track.py
        from .yolov8 import Yolov8Strategy

        return Yolov8Strategy
    elif "rf-detr" in str(yolo_model):
        try:
            import rfdetr
        except (ImportError, AssertionError, AttributeError):
            checker.check_packages(("onnxruntime",))  # needed dependency
            checker.check_packages(("rfdetr",))  # needed dependency
        from .rfdetr import RFDETRStrategy

        return RFDETRStrategy
    elif "yolo_nas" in str(yolo_model):
        try:
            import super_gradients  # for linear_assignment

            assert super_gradients.__version__
        except (ImportError, AssertionError, AttributeError):
            checker.check_packages(("super-gradients==3.1.3",))  # install
        from .yolonas import YoloNASStrategy

        return YoloNASStrategy
    else:
        LOGGER.error("Failed to infer inference mode from yolo model name")
        LOGGER.error("Your model name has to contain either yolox, yolo_nas or yolov8")
        exit()
