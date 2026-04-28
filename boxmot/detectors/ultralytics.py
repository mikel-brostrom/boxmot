# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

from boxmot.detectors.base import BaseDetectorBackend, Detections
from boxmot.detectors.registry import get_detector_url, is_ultralytics_model
from boxmot.utils import logger as LOGGER
from boxmot.utils.download import download_file
from boxmot.utils.misc import resolve_model_path


class UltralyticsDetector(BaseDetectorBackend):
    """
    Detector wrapper for Ultralytics YOLO models (YOLOv8, YOLO11, etc.).

    All Ultralytics internals are contained here. The public interface
    follows the same Detector contract as YoloXDetector and RTDetrDetector.
    """

    def __init__(self, model, device, imgsz=None):
        model_path = resolve_model_path(model)
        detector_url = get_detector_url(model_path)
        if detector_url and not model_path.exists():
            LOGGER.info("Downloading detector weights...")
            download_file(url=detector_url, dest=model_path, overwrite=False)
        elif is_ultralytics_model(model_path.name) and not model_path.exists():
            LOGGER.info("Downloading detector weights...")
            attempt_download_asset(model_path, release="latest")

        self.device = device
        self.imgsz = imgsz  # passed through to YOLO.predict
        self._yolo = self._load_yolo(model_path)
        self.names = self._yolo.names or {}
        self.pt = True
        self.stride = 32

    @staticmethod
    def _is_corrupt_weights_error(exc: Exception) -> bool:
        message = str(exc)
        return "PytorchStreamReader failed reading zip archive" in message or "failed finding central directory" in message

    def _load_yolo(self, model_path: Path):
        try:
            return YOLO(str(model_path))
        except RuntimeError as exc:
            if not (model_path.exists() and is_ultralytics_model(model_path.name) and self._is_corrupt_weights_error(exc)):
                raise

            LOGGER.warning(f"Detector weights appear corrupted, removing and re-downloading {model_path}")
            model_path.unlink(missing_ok=True)
            attempt_download_asset(model_path, release="latest")
            return YOLO(str(model_path))

    def preprocess(self, images):
        return images

    def process(self, preprocessed, conf, iou, classes, agnostic_nms):
        return self._yolo.predict(
            source=preprocessed,
            conf=conf,
            iou=iou,
            classes=classes,
            agnostic_nms=agnostic_nms,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
            stream=False,
        )

    def postprocess(self, detections):
        processed = []
        for result in detections:
            dets = self._extract_dets(result)
            processed.append(Detections(
                dets=dets,
                orig_img=result.orig_img,
                path=result.path or "",
                names=result.names or self.names,
            ))
        return processed

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
        preprocessed = self.preprocess(images)
        yolo_results = self.process(
            preprocessed,
            conf=conf,
            iou=iou,
            classes=classes,
            agnostic_nms=agnostic_nms,
        )
        return self.postprocess(yolo_results)
