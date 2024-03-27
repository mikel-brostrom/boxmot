from copy import deepcopy
from ultralytics import YOLO


class YOLOv8_wrapper:
    def __init__(self, args):
        self.args = deepcopy(args)
        self.yolov8 = YOLO(args.yolo_model.name)
    
    def inference(self, img_path):
        outputs = self.yolov8([img_path],
            conf=self.args.conf,
            iou=self.args.iou,
            agnostic_nms=self.args.agnostic_nms,
            stream=True,
            device=self.args.device,
            verbose=self.args.verbose,
            exist_ok=self.args.exist_ok,
            project=self.args.project,
            name=self.args.name,
            classes=self.args.classes,
            imgsz=self.args.imgsz,
        )

        results = []
        for output in outputs:
            dets = []
            for box in output.boxes:
                bbox = box.xyxy[0].tolist()  # Convert from tensor to list
                conf = box.conf.item()  # Get confidence score
                cls = box.cls.item()  # Get confidence score
                dets.append([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1], conf, cls])
            results.append(dets)
        return results[0]