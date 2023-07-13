from .yolox import YoloXStrategy
from .yolov8 import Yolov8Strategy
from .yolonas import YoloNASStrategy

YOLO_SWITCH = {
    'yolox': YoloXStrategy,
    'yolov8': Yolov8Strategy,
    'yolo_nas': YoloNASStrategy
}

def find_yolo_engine(yolo_model):
    for key in YOLO_SWITCH.keys():
        if key in str(yolo_model):
            return YOLO_SWITCH[key]
    
