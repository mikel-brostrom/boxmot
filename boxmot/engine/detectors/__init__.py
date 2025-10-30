# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

"""
BoxMOT Detector Interface
=========================

This module provides a standardized interface for object detectors.

New Interface (Recommended):
    >>> from boxmot.engine.detectors import YOLOX, Ultralytics, RFDETR
    >>> detector = YOLOX("yolox_s.pt", device="cpu", conf_thres=0.5)
    >>> boxes = detector("image.jpg")

Legacy Interface (Backward Compatibility):
    >>> from boxmot.engine.detectors import get_yolo_inferer
    >>> strategy = get_yolo_inferer("yolox_s.pt")
    >>> model = strategy(model="yolox_s.pt", device="cpu", args=args)
"""

from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import RequirementsChecker

checker = RequirementsChecker()

# ============================================================================
# NEW STANDARDIZED DETECTOR INTERFACE
# ============================================================================

# Base classes and utilities
from boxmot.engine.detectors.base import Detector, resolve_image

# Detector implementations
try:
    from boxmot.engine.detectors.yolox_detector import YOLOX
except ImportError:
    YOLOX = None

try:
    from boxmot.engine.detectors.ultralytics import Ultralytics
except ImportError:
    Ultralytics = None

try:
    from boxmot.engine.detectors.rfdetr_detector import RFDETR
except ImportError:
    RFDETR = None

# ============================================================================
# LEGACY INTERFACE (Backward Compatibility)
# ============================================================================

# Supported model types
ULTRALYTICS_MODELS = ["yolov8", "yolov9", "yolov10", "yolo11", "yolo12", "rtdetr", "sam"]


def is_ultralytics_model(yolo_name):
    """Check if model name corresponds to an Ultralytics model."""
    return any(yolo in str(yolo_name) for yolo in ULTRALYTICS_MODELS)


def is_yolox_model(yolo_name):
    """Check if model name corresponds to a YOLOX model."""
    return "yolox" in str(yolo_name)


def default_imgsz(yolo_name):
    """
    Get default image size for a model.
    
    Args:
        yolo_name: Model name or path
        
    Returns:
        list: [width, height] for the model
    """
    if is_ultralytics_model(yolo_name):
        return [640, 640]
    elif is_yolox_model(yolo_name):
        return [800, 1440]
    else:
        return [640, 640]


def get_yolo_inferer(yolo_model):
    """
    Get the appropriate detector strategy for a model (legacy interface).
    
    Args:
        yolo_model: Path to model weights
        
    Returns:
        Detector strategy class
        
    Note:
        This is the legacy interface. For new code, use:
        YOLOX(), Ultralytics(), or RFDETR() directly.
    """
    # YOLOX models
    if is_yolox_model(yolo_model):
        try:
            import yolox
            assert yolox.__version__
        except (ImportError, AssertionError, AttributeError):
            checker.check_packages(("yolox",), extra_args=["--no-deps"])
            checker.check_packages(("tabulate",))
            checker.check_packages(("thop",))
        from .yolox import YoloXStrategy
        return YoloXStrategy
    
    # Ultralytics models (YOLOv8, v9, v10, v11, RT-DETR, etc.)
    elif is_ultralytics_model(yolo_model):
        from .ultralytics import Ultralytics
        return Ultralytics
    
    # RF-DETR models
    elif "rf-detr" in str(yolo_model):
        try:
            import rfdetr
        except (ImportError, AssertionError, AttributeError):
            checker.check_packages(("onnxruntime",))
            checker.check_packages(("rfdetr",))
        from .rfdetr import RFDETR
        return RFDETR
    
    # Unknown model type
    else:
        LOGGER.error("Failed to infer inference mode from yolo model name")
        LOGGER.error("Your model name has to contain either yolox, yolo_nas or yolov8")
        exit()


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # New standardized interface
    'Detector',
    'resolve_image',
    'YOLOX',
    'Ultralytics',
    'RFDETR',
    # Legacy interface
    'get_yolo_inferer',
    'is_ultralytics_model',
    'is_yolox_model',
    'default_imgsz',
]
