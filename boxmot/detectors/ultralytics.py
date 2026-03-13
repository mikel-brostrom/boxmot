# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import numpy as np
from ultralytics import YOLO

from boxmot.detectors.detector import Detections, Detector


class UltralyticsDetector(Detector):
    """
    Detector wrapper for Ultralytics YOLO models (YOLOv8, YOLO11, etc.).

    All Ultralytics internals are contained here. The public interface
    follows the same Detector contract as YoloXDetector and RTDetrDetector.
    """

    def __init__(self, model, device, imgsz=None):
        self.device = device
        self.imgsz = imgsz  # passed through to YOLO.predict
        self._yolo = YOLO(str(model))
        self.names = self._yolo.names or {}
        self.pt = True
        self.stride = 32

    def preprocess(self, images):
        raise NotImplementedError("Use __call__ directly for UltralyticsDetector")

    def process(self, preprocessed):
        raise NotImplementedError("Use __call__ directly for UltralyticsDetector")

    def postprocess(self, detections):
        raise NotImplementedError("Use __call__ directly for UltralyticsDetector")

    def __call__(self, images: list, conf, iou, classes, agnostic_nms) -> list:
        yolo_results = self._yolo.predict(
            source=images,
            conf=conf,
            iou=iou,
            classes=classes,
            agnostic_nms=agnostic_nms,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
            stream=False,
        )
        detections = []
        for r in yolo_results:
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy().reshape(-1, 1)
                cls  = r.boxes.cls.cpu().numpy().reshape(-1, 1)
                dets = np.concatenate([xyxy, conf, cls], axis=1)
            else:
                dets = np.empty((0, 6))
            detections.append(Detections(
                dets=dets,
                orig_img=r.orig_img,
                path=r.path or "",
                names=r.names or self.names,
            ))
        return detections
