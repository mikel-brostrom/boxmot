

def create_detector(args):
    yolo_model = str(args.yolo_model)
    if 'triton' in yolo_model:
        from .yolov8_triton import YoloV8SegPoseAPI
        return YoloV8SegPoseAPI(model_name=yolo_model, confidence_threshold=args.conf)

    elif 'yolov8' in yolo_model:
        from .yolov8 import YOLOv8_wrapper
        return YOLOv8_wrapper(args)
        
    elif 'yolox' in yolo_model:
        from .yolox import YOLOX_wrapper
        return YOLOX_wrapper(args)