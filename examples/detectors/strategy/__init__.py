from .yolonas import YoloNASStrategy
from .yolov8 import Yolov8Strategy
from .yolox import YoloXStrategy

YOLO_SWITCH = {
    'yolox': YoloXStrategy,
    'yolov8': Yolov8Strategy,
    'yolo_nas': YoloNASStrategy
}


def find_yolo_engine(yolo_model):
    for key in YOLO_SWITCH.keys():
        if key in str(yolo_model):
            return YOLO_SWITCH[key]
