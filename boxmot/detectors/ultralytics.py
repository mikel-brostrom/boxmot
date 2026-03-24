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

    @staticmethod
    def _as_numpy(values) -> np.ndarray:
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "numpy"):
            values = values.numpy()
        return np.asarray(values, dtype=np.float32)

    def _extract_dets(self, result) -> np.ndarray:
        if result.obb is not None:
            if len(result.obb) == 0:
                return np.empty((0, 7), dtype=np.float32)
            xywhr = self._as_numpy(result.obb.xywhr)
            conf = self._as_numpy(result.obb.conf).reshape(-1, 1)
            cls = self._as_numpy(result.obb.cls).reshape(-1, 1)
            return np.concatenate([xywhr, conf, cls], axis=1)

        if result.boxes is not None:
            if len(result.boxes) == 0:
                return np.empty((0, 6), dtype=np.float32)
            xyxy = self._as_numpy(result.boxes.xyxy)
            conf = self._as_numpy(result.boxes.conf).reshape(-1, 1)
            cls = self._as_numpy(result.boxes.cls).reshape(-1, 1)
            return np.concatenate([xyxy, conf, cls], axis=1)

        return np.empty((0, 6), dtype=np.float32)

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
            dets = self._extract_dets(r)
            detections.append(Detections(
                dets=dets,
                orig_img=r.orig_img,
                path=r.path or "",
                names=r.names or self.names,
            ))
        return detections
