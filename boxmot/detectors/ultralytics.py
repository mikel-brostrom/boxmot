# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from pathlib import Path
from typing import Any, Optional

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

    Wraps Ultralytics' :class:`BasePredictor` so the standard
    ``preprocess`` / ``process`` / ``postprocess`` contract maps onto
    ``predictor.preprocess`` (letterbox + tensor upload),
    ``predictor.inference`` (pure forward) and
    ``predictor.postprocess`` (NMS + scale-back). This lets timing
    instrumentation report each stage independently instead of collapsing
    everything into the inference bucket.
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

        # Lazily initialised on the first call: ``predictor`` only exists
        # after ``YOLO.predict`` runs once, and we don't know the conf/iou
        # overrides until then.
        self._predictor = None
        self._last_orig_imgs: Optional[list[np.ndarray]] = None
        self._last_preprocessed: Any = None

    @staticmethod
    def _is_corrupt_weights_error(exc: Exception) -> bool:
        message = str(exc)
        return (
            "PytorchStreamReader failed reading zip archive" in message
            or "failed finding central directory" in message
        )

    def _load_yolo(self, model_path: Path):
        try:
            return YOLO(str(model_path))
        except RuntimeError as exc:
            if not (
                model_path.exists()
                and is_ultralytics_model(model_path.name)
                and self._is_corrupt_weights_error(exc)
            ):
                raise

            LOGGER.warning(f"Detector weights appear corrupted, removing and re-downloading {model_path}")
            model_path.unlink(missing_ok=True)
            attempt_download_asset(model_path, release="latest")
            return YOLO(str(model_path))

    def _ensure_predictor(self, conf=None, iou=None, classes=None, agnostic_nms=None) -> None:
        """Ensure ``self._yolo.predictor`` exists and is configured.

        Ultralytics only instantiates the predictor on the first ``predict``
        call. We use a 1×1 dummy frame to trigger that — this is dwarfed by
        the real per-frame work and only happens once per detector instance.
        """
        if self._predictor is None:
            dummy = np.zeros((32, 32, 3), dtype=np.uint8)
            self._yolo.predict(
                source=dummy,
                conf=0.25 if conf is None else float(conf),
                iou=0.7 if iou is None else float(iou),
                classes=classes,
                agnostic_nms=False if agnostic_nms is None else bool(agnostic_nms),
                device=self.device,
                imgsz=self.imgsz,
                verbose=False,
                save=False,
                stream=False,
            )
            self._predictor = self._yolo.predictor
        # Refresh args used by ``predictor.postprocess`` for NMS thresholds.
        if conf is not None:
            self._predictor.args.conf = float(conf)
        if iou is not None:
            self._predictor.args.iou = float(iou)
        if classes is not None:
            self._predictor.args.classes = classes
        if agnostic_nms is not None:
            self._predictor.args.agnostic_nms = bool(agnostic_nms)

    @staticmethod
    def _as_image_list(images) -> list[np.ndarray]:
        if isinstance(images, list):
            return images
        return [images]

    def preprocess(self, images, **kwargs):
        self._ensure_predictor()
        ims = self._as_image_list(images)
        self._last_orig_imgs = ims
        # ``predictor.batch`` is read by some postprocess paths to recover
        # source paths; the third element (``s``) is unused by the BoxMOT
        # adapter so we leave it as None.
        self._predictor.batch = ([""] * len(ims), ims, None)
        preprocessed = self._predictor.preprocess(ims)
        self._last_preprocessed = preprocessed
        return preprocessed

    def process(self, preprocessed, **kwargs):
        self._ensure_predictor()
        return self._predictor.inference(preprocessed)

    def postprocess(self, raw_preds, conf=None, iou=None, classes=None, agnostic_nms=None, **kwargs):
        self._ensure_predictor(conf=conf, iou=iou, classes=classes, agnostic_nms=agnostic_nms)
        preprocessed = kwargs.get("preprocessed", self._last_preprocessed)
        orig_imgs = kwargs.get("orig_imgs", self._last_orig_imgs) or []
        results = self._predictor.postprocess(raw_preds, preprocessed, orig_imgs)
        processed: list[Detections] = []
        for result in results:
            dets, masks = self._extract_dets(result)
            processed.append(Detections(
                dets=dets,
                orig_img=result.orig_img,
                path=result.path or "",
                names=result.names or self.names,
                masks=masks,
            ))
        return processed

    @staticmethod
    def _as_numpy(values) -> np.ndarray:
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "numpy"):
            values = values.numpy()
        return np.asarray(values, dtype=np.float32)

    def _extract_dets(self, result) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract detections and optional masks from an Ultralytics result.

        Returns:
            (dets, masks) where masks is (N, H, W) uint8 or None.
        """
        masks = None

        if result.obb is not None:
            if len(result.obb) == 0:
                return np.empty((0, 7), dtype=np.float32), None
            xywhr = self._as_numpy(result.obb.xywhr)
            conf = self._as_numpy(result.obb.conf).reshape(-1, 1)
            cls = self._as_numpy(result.obb.cls).reshape(-1, 1)
            return np.concatenate([xywhr, conf, cls], axis=1), None

        if result.boxes is not None:
            if len(result.boxes) == 0:
                return np.empty((0, 6), dtype=np.float32), None
            xyxy = self._as_numpy(result.boxes.xyxy)
            conf = self._as_numpy(result.boxes.conf).reshape(-1, 1)
            cls = self._as_numpy(result.boxes.cls).reshape(-1, 1)
            dets = np.concatenate([xyxy, conf, cls], axis=1)

            # Extract masks from segmentation models
            if result.masks is not None and len(result.masks) > 0:
                mask_data = result.masks.data  # (N, H_model, W_model) tensor
                if hasattr(mask_data, "cpu"):
                    mask_data = mask_data.cpu()
                if hasattr(mask_data, "numpy"):
                    mask_data = mask_data.numpy()
                masks = (mask_data > 0.5).astype(np.uint8)

            return dets, masks

        return np.empty((0, 6), dtype=np.float32), None

    def __call__(self, images: list, conf, iou, classes, agnostic_nms) -> list:
        preprocessed = self.preprocess(images)
        raw = self.process(preprocessed)
        return self.postprocess(
            raw,
            conf=conf,
            iou=iou,
            classes=classes,
            agnostic_nms=agnostic_nms,
        )
